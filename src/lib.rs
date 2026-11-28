//! PyO3 bindings exposing compiled stateful Rust execution plans (`FixedPlan`, `MovingPlan`,
//! `CurvesPlan`) with unified f64/f32 dispatch and cached `WindowPlan` / `MovingSchedule`.

mod baseline;
mod compat;
mod detect;
mod quantile;

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use baseline::{
    build_moving_schedule, build_window_plan, compute_pixel_climatology, detrend_pixel,
    MovingSchedule, WindowPlan,
};
use compat::{
    BoundPolicy, CompatConfig, DetrendOrder, EventMode, Feb29Mode, MovingBoundary, SmoothNanPolicy,
};
use detect::{detect_pixel_with_curves, DetectParams, EventRecord, CATEGORY_MISSING};
use quantile::QuantileMethod;

// ============================================================================
// Unified f64 / f32 matrix view (eliminates duplicate *_f32_rs wrappers)
// ============================================================================

/// Zero-copy Python 2D array wrapper accepting either f64 or f32 C-contiguous tiles.
#[derive(FromPyObject)]
enum FloatArray2<'py> {
    F64(PyReadonlyArray2<'py, f64>),
    F32(PyReadonlyArray2<'py, f32>),
}

/// Borrowed flat matrix view whose rows promote f32 tiles to f64 inside Rayon workers.
enum FloatMatrix<'a> {
    F64(&'a [f64], usize, usize),
    F32(&'a [f32], usize, usize),
}

/// Borrow a 2D NumPy buffer only if strictly C-contiguous (rejects Fortran-order memory).
fn as_c_slice_2d<'a, T: numpy::Element>(
    arr: &'a PyReadonlyArray2<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err(format!("{name} must be C-contiguous")));
    }
    arr.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

impl<'py> FloatArray2<'py> {
    fn as_matrix(&'py self, name: &str) -> PyResult<FloatMatrix<'py>> {
        match self {
            Self::F64(a) => {
                let s = as_c_slice_2d(a, name)?;
                let sh = a.shape();
                Ok(FloatMatrix::F64(s, sh[0], sh[1]))
            }
            Self::F32(a) => {
                let s = as_c_slice_2d(a, name)?;
                let sh = a.shape();
                Ok(FloatMatrix::F32(s, sh[0], sh[1]))
            }
        }
    }
}

impl<'a> FloatMatrix<'a> {
    #[inline]
    fn nrows(&self) -> usize {
        match *self {
            Self::F64(_, r, _) | Self::F32(_, r, _) => r,
        }
    }

    #[inline]
    fn ncols(&self) -> usize {
        match *self {
            Self::F64(_, _, c) | Self::F32(_, _, c) => c,
        }
    }

    #[inline]
    fn row_f64<'b>(&'a self, p: usize, buf: &'b mut Vec<f64>) -> &'b [f64]
    where
        'a: 'b,
    {
        match *self {
            Self::F64(s, _, c) => &s[p * c..(p + 1) * c],
            Self::F32(s, _, c) => {
                buf.clear();
                buf.extend(s[p * c..(p + 1) * c].iter().map(|&v| v as f64));
                buf.as_slice()
            }
        }
    }
}

// ============================================================================
// String parsers & boundary validators (called once at plan construction)
// ============================================================================

fn parse_compat_and_detect(
    quantile_method: &str,
    feb29_mode: &str,
    smooth_nan_policy: &str,
    strict_inequality: bool,
    var_ddof: usize,
    event_mode: &str,
    min_duration: usize,
    max_gap: usize,
    max_pad_length: usize,
    lower_bound: Option<f64>,
    bound_policy: &str,
) -> PyResult<(CompatConfig, DetectParams)> {
    let qm = match quantile_method {
        "type7" | "python" | "linear" => QuantileMethod::Type7,
        "type5" | "matlab" | "hazen" => QuantileMethod::Type5,
        s => return Err(PyValueError::new_err(format!("Unsupported quantile_method: '{s}'"))),
    };
    let f29 = match feb29_mode {
        "interp_59_61" => Feb29Mode::Interp5961,
        "window_direct" => Feb29Mode::WindowDirect,
        "heatwaver_series" => Feb29Mode::HeatwaveRSeries,
        "oliver_matrix" => Feb29Mode::OliverMatrix,
        s => return Err(PyValueError::new_err(format!("Unsupported feb29_mode: '{s}'"))),
    };
    let snp = match smooth_nan_policy {
        "omit" => SmoothNanPolicy::OmitNan,
        "propagate" => SmoothNanPolicy::PropagateNan,
        s => return Err(PyValueError::new_err(format!("Unsupported smooth_nan_policy: '{s}'"))),
    };
    let em = match event_mode {
        "high" | "mhw" => EventMode::High,
        "low" | "mcs" => EventMode::Low,
        "both" => EventMode::Both,
        s => return Err(PyValueError::new_err(format!("Unsupported event mode: '{s}'"))),
    };
    let bp = match bound_policy {
        "flag" => BoundPolicy::Flag,
        "mask" => BoundPolicy::Mask,
        "floor" => BoundPolicy::Floor,
        s => return Err(PyValueError::new_err(format!("Unsupported bound_policy: '{s}'"))),
    };
    let compat = CompatConfig {
        quantile_method: qm,
        strict_inequality,
        feb29_mode: f29,
        smooth_nan_policy: snp,
        var_ddof,
    };
    let params = DetectParams {
        event_mode: em,
        min_duration,
        max_gap,
        max_pad_length,
        lower_bound,
        bound_policy: bp,
        compat,
    };
    Ok((compat, params))
}

fn parse_detrend_order(s: &str) -> PyResult<DetrendOrder> {
    match s {
        "none" => Ok(DetrendOrder::None),
        "linear" => Ok(DetrendOrder::Linear),
        "quadratic" => Ok(DetrendOrder::Quadratic),
        _ => Err(PyValueError::new_err(format!("Unsupported detrend_order: '{s}'"))),
    }
}

fn parse_moving_boundary(s: &str) -> PyResult<MovingBoundary> {
    match s {
        "clamp" => Ok(MovingBoundary::Clamp),
        "shrink" => Ok(MovingBoundary::Shrink),
        _ => Err(PyValueError::new_err(format!("Unsupported moving_boundary: '{s}'"))),
    }
}

/// Validate 1-based DOY values against `1..=year_len` before indexing climatology tables.
fn validate_doys(doys: &[u16], year_len: usize, name: &str) -> PyResult<()> {
    if year_len == 0 {
        return Err(PyValueError::new_err("year_len must be >= 1"));
    }
    if let Some(&bad) = doys.iter().find(|&&d| d == 0 || (d as usize) > year_len) {
        return Err(PyValueError::new_err(format!(
            "{name} contains out-of-range DOY {bad} (expected 1..={year_len})"
        )));
    }
    Ok(())
}

/// Validate that active pixel indices are in-bounds and strictly ascending for CSR assembly.
fn validate_active_indices(active_idx: &[i64], n_active: usize, total_grid_pixels: usize) -> PyResult<()> {
    if active_idx.len() != n_active {
        return Err(PyValueError::new_err(format!(
            "active_pixel_indices length {} != n_active {n_active}",
            active_idx.len()
        )));
    }
    let mut prev: i64 = -1;
    for &idx in active_idx {
        if idx < 0 || (idx as usize) >= total_grid_pixels {
            return Err(PyValueError::new_err(format!(
                "active_pixel_indices entry {idx} out of bounds [0, {total_grid_pixels})"
            )));
        }
        if idx <= prev {
            return Err(PyValueError::new_err("active_pixel_indices must be strictly increasing"));
        }
        prev = idx;
    }
    Ok(())
}

/// Split a flat per-tile output buffer into per-pixel mutable slices (or `None` when disabled).
fn opt_chunks_mut<T>(buf: &mut [T], chunk_len: usize, n_active: usize) -> Vec<Option<&mut [T]>> {
    if buf.is_empty() || chunk_len == 0 {
        (0..n_active).map(|_| None).collect()
    } else {
        buf.chunks_mut(chunk_len).map(Some).collect()
    }
}

// ============================================================================
// Single SoA event packer & unified Rayon tile runner
// ============================================================================

macro_rules! define_packed_events {
    ($($field:ident : $ty:ty),* $(,)?) => {
        struct PackedEvents {
            $($field: Vec<$ty>,)*
            grid_offsets: Vec<i64>,
        }

        fn pack_events(
            per_pixel_events: Vec<Vec<EventRecord>>,
            total_grid_pixels: usize,
            active_pixel_indices: &[i64],
        ) -> PackedEvents {
            let cap: usize = per_pixel_events.iter().map(|v| v.len()).sum();
            let mut pe = PackedEvents {
                $($field: Vec::with_capacity(cap),)*
                grid_offsets: vec![0i64; total_grid_pixels + 1],
            };
            let mut cur = 0usize;
            let mut count = 0i64;
            for g in 0..total_grid_pixels {
                pe.grid_offsets[g] = count;
                if cur < active_pixel_indices.len() && active_pixel_indices[cur] as usize == g {
                    for ev in &per_pixel_events[cur] {
                        $(pe.$field.push(ev.$field);)*
                    }
                    count += per_pixel_events[cur].len() as i64;
                    cur += 1;
                }
            }
            pe.grid_offsets[total_grid_pixels] = count;
            pe
        }

        fn packed_to_pydict<'py>(py: Python<'py>, pe: PackedEvents) -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            $(d.set_item(stringify!($field), PyArray1::from_vec(py, pe.$field))?;)*
            d.set_item("grid_offsets", PyArray1::from_vec(py, pe.grid_offsets))?;
            Ok(d)
        }
    };
}

define_packed_events! {
    pixel_idx: i64,
    event_type: i8,
    index_start: i32,
    index_peak: i32,
    index_end: i32,
    duration: i32,
    int_max: f64,
    int_mean: f64,
    int_std: f64,
    int_var: f64,
    int_cum: f64,
    int_max_rel_thresh: f64,
    int_mean_rel_thresh: f64,
    int_cum_rel_thresh: f64,
    rate_onset: f64,
    rate_decline: f64,
    category_max: i8,
    p_moderate: f64,
    p_strong: f64,
    p_severe: f64,
    p_extreme: f64,
    truncated_start: bool,
    truncated_end: bool,
    near_lower_bound: bool,
}

#[derive(Clone, Copy)]
struct TileFlags {
    event_mode: EventMode,
    clim_stride: usize,
    clim: bool,
    hi: bool,
    lo: bool,
    trend: bool,
    intensity: bool,
    intensity_high: bool,
    intensity_low: bool,
    daily_cat: bool,
    event_cat: bool,
}

impl TileFlags {
    fn ts_only(event_mode: EventMode, intensity: bool, daily_cat: bool, event_cat: bool) -> Self {
        let is_both = intensity && matches!(event_mode, EventMode::Both);
        Self {
            event_mode,
            clim_stride: 0,
            clim: false,
            hi: false,
            lo: false,
            trend: false,
            intensity,
            intensity_high: is_both,
            intensity_low: is_both,
            daily_cat,
            event_cat,
        }
    }

    fn with_clim(
        event_mode: EventMode,
        clim_stride: usize,
        return_clim: bool,
        trend: bool,
        intensity: bool,
        daily_cat: bool,
        event_cat: bool,
    ) -> Self {
        let is_both = intensity && matches!(event_mode, EventMode::Both);
        Self {
            event_mode,
            clim_stride,
            clim: return_clim,
            hi: return_clim && matches!(event_mode, EventMode::High | EventMode::Both),
            lo: return_clim && matches!(event_mode, EventMode::Low | EventMode::Both),
            trend,
            intensity,
            intensity_high: is_both,
            intensity_low: is_both,
            daily_cat,
            event_cat,
        }
    }
}

struct PixelOuts<'a> {
    clim: Option<&'a mut [f64]>,
    hi: Option<&'a mut [f64]>,
    lo: Option<&'a mut [f64]>,
    trend: Option<&'a mut [f64]>,
    intensity: Option<&'a mut [f64]>,
    intensity_high: Option<&'a mut [f64]>,
    intensity_low: Option<&'a mut [f64]>,
    daily_cat: Option<&'a mut [i8]>,
    event_cat: Option<&'a mut [i8]>,
}

struct TileOutBufs {
    packed: PackedEvents,
    clim: Vec<f64>,
    hi: Vec<f64>,
    lo: Vec<f64>,
    trend: Vec<f64>,
    intensity: Vec<f64>,
    intensity_high: Vec<f64>,
    intensity_low: Vec<f64>,
    daily_cat: Vec<i8>,
    event_cat: Vec<i8>,
}

/// Single shared Rayon tile executor used by all 3 compiled plan classes.
fn run_tile_parallel<S, I, F>(
    n_active: usize,
    n_det: usize,
    total_grid_pixels: usize,
    active_idx: &[i64],
    flags: TileFlags,
    run_when_empty_det: bool,
    init_scratch: I,
    per_pixel: F,
) -> TileOutBufs
where
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize, PixelOuts<'_>) -> Vec<EventRecord> + Sync + Send,
{
    let mut clim = if flags.clim { vec![f64::NAN; n_active * flags.clim_stride] } else { Vec::new() };
    let mut hi = if flags.hi { vec![f64::NAN; n_active * flags.clim_stride] } else { Vec::new() };
    let mut lo = if flags.lo { vec![f64::NAN; n_active * flags.clim_stride] } else { Vec::new() };
    let mut trend = if flags.trend { vec![f64::NAN; n_active * 3] } else { Vec::new() };
    let mut intensity = if flags.intensity { vec![f64::NAN; n_active * n_det] } else { Vec::new() };
    let mut intensity_high = if flags.intensity_high { vec![f64::NAN; n_active * n_det] } else { Vec::new() };
    let mut intensity_low = if flags.intensity_low { vec![f64::NAN; n_active * n_det] } else { Vec::new() };
    let mut daily_cat = if flags.daily_cat { vec![CATEGORY_MISSING; n_active * n_det] } else { Vec::new() };
    let mut event_cat = if flags.event_cat { vec![CATEGORY_MISSING; n_active * n_det] } else { Vec::new() };

    let per_pixel_events: Vec<Vec<EventRecord>> = if n_active > 0 && (n_det > 0 || run_when_empty_det) {
        let s_slots = opt_chunks_mut(&mut clim, flags.clim_stride, n_active);
        let h_slots = opt_chunks_mut(&mut hi, flags.clim_stride, n_active);
        let l_slots = opt_chunks_mut(&mut lo, flags.clim_stride, n_active);
        let t_slots = opt_chunks_mut(&mut trend, 3, n_active);
        let i_slots = opt_chunks_mut(&mut intensity, n_det, n_active);
        let ih_slots = opt_chunks_mut(&mut intensity_high, n_det, n_active);
        let il_slots = opt_chunks_mut(&mut intensity_low, n_det, n_active);
        let d_slots = opt_chunks_mut(&mut daily_cat, n_det, n_active);
        let e_slots = opt_chunks_mut(&mut event_cat, n_det, n_active);

        s_slots
            .into_par_iter()
            .zip(h_slots.into_par_iter())
            .zip(l_slots.into_par_iter())
            .zip(t_slots.into_par_iter())
            .zip(i_slots.into_par_iter())
            .zip(ih_slots.into_par_iter())
            .zip(il_slots.into_par_iter())
            .zip(d_slots.into_par_iter())
            .zip(e_slots.into_par_iter())
            .enumerate()
            .map_init(init_scratch, |scratch, (p, ((((((((s, h), l), tr), i), ih), il), dc), ec))| {
                per_pixel(
                    scratch,
                    p,
                    PixelOuts {
                        clim: s,
                        hi: h,
                        lo: l,
                        trend: tr,
                        intensity: i,
                        intensity_high: ih,
                        intensity_low: il,
                        daily_cat: dc,
                        event_cat: ec,
                    },
                )
            })
            .collect()
    } else {
        vec![Vec::new(); n_active]
    };

    let packed = pack_events(per_pixel_events, total_grid_pixels, active_idx);
    TileOutBufs {
        packed,
        clim,
        hi,
        lo,
        trend,
        intensity,
        intensity_high,
        intensity_low,
        daily_cat,
        event_cat,
    }
}

fn vec_to_opt_array2<'py, T: numpy::Element>(
    py: Python<'py>,
    flat: Vec<T>,
    shape: (usize, usize),
    enabled: bool,
) -> PyResult<Py<PyAny>> {
    if !enabled {
        return Ok(py.None());
    }
    let a = Array2::from_shape_vec(shape, flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(a.into_pyarray(py).into_any().unbind())
}

fn vec_to_opt_array3<'py>(
    py: Python<'py>,
    flat: Vec<f64>,
    shape: (usize, usize, usize),
    enabled: bool,
) -> PyResult<Py<PyAny>> {
    if !enabled {
        return Ok(py.None());
    }
    let a = Array3::from_shape_vec(shape, flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(a.into_pyarray(py).into_any().unbind())
}

impl TileOutBufs {
    fn into_pydict<'py>(
        self,
        py: Python<'py>,
        n_active: usize,
        n_det: usize,
        flags: TileFlags,
        clim_3d_years: Option<usize>,
        year_len: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("events", packed_to_pydict(py, self.packed)?)?;
        let int_obj = vec_to_opt_array2(py, self.intensity, (n_active, n_det), flags.intensity)?;
        out.set_item("intensity_ts", int_obj.clone_ref(py))?;
        let hi_obj = if flags.intensity_high {
            vec_to_opt_array2(py, self.intensity_high, (n_active, n_det), true)?
        } else if flags.intensity && matches!(flags.event_mode, EventMode::High) {
            int_obj.clone_ref(py)
        } else {
            py.None()
        };
        let lo_obj = if flags.intensity_low {
            vec_to_opt_array2(py, self.intensity_low, (n_active, n_det), true)?
        } else if flags.intensity && matches!(flags.event_mode, EventMode::Low) {
            int_obj
        } else {
            py.None()
        };
        out.set_item("intensity_high_ts", hi_obj)?;
        out.set_item("intensity_low_ts", lo_obj)?;
        out.set_item("daily_category_ts", vec_to_opt_array2(py, self.daily_cat, (n_active, n_det), flags.daily_cat)?)?;
        out.set_item("event_category_ts", vec_to_opt_array2(py, self.event_cat, (n_active, n_det), flags.event_cat)?)?;

        if flags.clim_stride > 0 {
            if let Some(n_uy) = clim_3d_years {
                out.set_item("clim", vec_to_opt_array3(py, self.clim, (n_active, n_uy, year_len), flags.clim)?)?;
                out.set_item("thresh_high", vec_to_opt_array3(py, self.hi, (n_active, n_uy, year_len), flags.hi)?)?;
                out.set_item("thresh_low", vec_to_opt_array3(py, self.lo, (n_active, n_uy, year_len), flags.lo)?)?;
            } else {
                out.set_item("clim", vec_to_opt_array2(py, self.clim, (n_active, year_len), flags.clim)?)?;
                out.set_item("thresh_high", vec_to_opt_array2(py, self.hi, (n_active, year_len), flags.hi)?)?;
                out.set_item("thresh_low", vec_to_opt_array2(py, self.lo, (n_active, year_len), flags.lo)?)?;
            }
        }
        if flags.trend {
            out.set_item("trend", vec_to_opt_array2(py, self.trend, (n_active, 3), true)?)?;
        }
        Ok(out)
    }
}

/// Expand DOY-indexed climatology/threshold curves onto each detection step.
fn expand_doy_curves(
    doys: &[u16],
    s_row: &[f64],
    h_row: &[f64],
    l_row: &[f64],
    seas_t: &mut Vec<f64>,
    th_h_t: &mut Vec<f64>,
    th_l_t: &mut Vec<f64>,
) {
    seas_t.clear();
    th_h_t.clear();
    th_l_t.clear();
    for &d in doys {
        let di = (d as usize) - 1;
        seas_t.push(s_row[di]);
        th_h_t.push(h_row[di]);
        th_l_t.push(l_row[di]);
    }
}

// ============================================================================
// Compiled Plan 1: FixedPlan (FixedBaseline + DetrendedBaseline)
// ============================================================================

/// Compiled fixed/detrended execution plan caching `WindowPlan` across all spatial tiles.
#[pyclass(frozen)]
struct FixedPlan {
    plan: WindowPlan,
    n_clim: usize,
    year_len: usize,
    pctile_high: f64,
    pctile_low: f64,
    smooth_width: usize,
    min_valid_fraction: f64,
    compat: CompatConfig,
    detect_params: DetectParams,
    det_doys: Vec<u16>,
    detrend_order: DetrendOrder,
    fit_mask: Vec<u8>,
    clim_slice: (usize, usize),
    det_slice: (usize, usize),
}

impl FixedPlan {
    fn run_clim_and_detect_pixel(
        &self,
        clim_row: &[f64],
        det_row: &[f64],
        pixel_id: i64,
        outs: PixelOuts<'_>,
        scratch: &mut Vec<f64>,
        seas_t: &mut Vec<f64>,
        th_h_t: &mut Vec<f64>,
        th_l_t: &mut Vec<f64>,
        pad_buf: &mut Vec<f64>,
    ) -> Vec<EventRecord> {
        let clim = compute_pixel_climatology(
            clim_row,
            &self.plan,
            self.pctile_high,
            self.pctile_low,
            self.smooth_width,
            self.min_valid_fraction,
            &self.compat,
            scratch,
        );
        if let Some(ps) = outs.clim {
            ps.copy_from_slice(&clim.seas);
        }
        if let Some(ph) = outs.hi {
            ph.copy_from_slice(&clim.thresh_high);
        }
        if let Some(pl) = outs.lo {
            pl.copy_from_slice(&clim.thresh_low);
        }
        if det_row.is_empty() {
            return Vec::new();
        }
        expand_doy_curves(&self.det_doys, &clim.seas, &clim.thresh_high, &clim.thresh_low, seas_t, th_h_t, th_l_t);
        detect_pixel_with_curves(
            pixel_id,
            det_row,
            seas_t,
            th_h_t,
            th_l_t,
            &self.detect_params,
            pad_buf,
            outs.intensity,
            outs.intensity_high,
            outs.intensity_low,
            outs.daily_cat,
            outs.event_cat,
        )
    }
}

#[pymethods]
impl FixedPlan {
    #[new]
    #[pyo3(signature = (
        clim_years, clim_doys, clim_in_period, year_len, pctile_high, pctile_low,
        window_half_width, smooth_width, min_valid_fraction, quantile_method, feb29_mode,
        smooth_nan_policy, *, det_doys=None, event_mode="high", min_duration=5, max_gap=2,
        max_pad_length=0, lower_bound=None, bound_policy="flag", strict_inequality=true,
        var_ddof=1, detrend_order="none", fit_mask=None, clim_slice_start=0,
        clim_slice_stop=None, det_slice_start=0, det_slice_stop=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        clim_years: PyReadonlyArray1<'_, i32>,
        clim_doys: PyReadonlyArray1<'_, u16>,
        clim_in_period: PyReadonlyArray1<'_, u8>,
        year_len: usize,
        pctile_high: f64,
        pctile_low: f64,
        window_half_width: usize,
        smooth_width: usize,
        min_valid_fraction: f64,
        quantile_method: &str,
        feb29_mode: &str,
        smooth_nan_policy: &str,
        det_doys: Option<PyReadonlyArray1<'_, u16>>,
        event_mode: &str,
        min_duration: usize,
        max_gap: usize,
        max_pad_length: usize,
        lower_bound: Option<f64>,
        bound_policy: &str,
        strict_inequality: bool,
        var_ddof: usize,
        detrend_order: &str,
        fit_mask: Option<PyReadonlyArray1<'_, u8>>,
        clim_slice_start: usize,
        clim_slice_stop: Option<usize>,
        det_slice_start: usize,
        det_slice_stop: Option<usize>,
    ) -> PyResult<Self> {
        let c_years = clim_years.as_slice().map_err(|_| PyValueError::new_err("clim_years must be contiguous"))?;
        let c_doys = clim_doys.as_slice().map_err(|_| PyValueError::new_err("clim_doys must be contiguous"))?;
        let c_in = clim_in_period.as_slice().map_err(|_| PyValueError::new_err("clim_in_period must be contiguous"))?;
        let n_clim = c_years.len();
        if c_doys.len() != n_clim || c_in.len() != n_clim {
            return Err(PyValueError::new_err("clim_years, clim_doys, and clim_in_period must have equal length"));
        }
        validate_doys(c_doys, year_len, "clim_doys")?;

        let d_doys_vec = match det_doys {
            Some(ref arr) => {
                let s = arr.as_slice().map_err(|_| PyValueError::new_err("det_doys must be contiguous"))?;
                validate_doys(s, year_len, "det_doys")?;
                s.to_vec()
            }
            None => Vec::new(),
        };
        let (compat, detect_params) = parse_compat_and_detect(
            quantile_method, feb29_mode, smooth_nan_policy, strict_inequality, var_ddof,
            event_mode, min_duration, max_gap, max_pad_length, lower_bound, bound_policy,
        )?;
        let det_ord = parse_detrend_order(detrend_order)?;

        let fit_mask_vec = match fit_mask {
            Some(ref arr) => arr.as_slice().map_err(|_| PyValueError::new_err("fit_mask must be contiguous"))?.to_vec(),
            None => Vec::new(),
        };
        let c_stop = match clim_slice_stop {
            Some(v) => v,
            None => clim_slice_start
                .checked_add(n_clim)
                .ok_or_else(|| PyValueError::new_err("clim_slice_start + n_clim overflowed"))?,
        };
        let d_stop = match det_slice_stop {
            Some(v) => v,
            None => det_slice_start
                .checked_add(d_doys_vec.len())
                .ok_or_else(|| PyValueError::new_err("det_slice_start + det_doys length overflowed"))?,
        };
        if !fit_mask_vec.is_empty() {
            let n_full = fit_mask_vec.len();
            if clim_slice_start > c_stop || c_stop > n_full || (c_stop - clim_slice_start) != n_clim {
                return Err(PyValueError::new_err("clim_slice bounds invalid for fit_mask / clim_years length"));
            }
            if det_slice_start > d_stop || d_stop > n_full || (d_stop - det_slice_start) != d_doys_vec.len() {
                return Err(PyValueError::new_err("det_slice bounds invalid for fit_mask / det_doys length"));
            }
        }

        let plan = py.detach(|| build_window_plan(c_years, c_doys, c_in, year_len, window_half_width, compat.feb29_mode));
        Ok(Self {
            plan,
            n_clim,
            year_len,
            pctile_high,
            pctile_low,
            smooth_width,
            min_valid_fraction,
            compat,
            detect_params,
            det_doys: d_doys_vec,
            detrend_order: det_ord,
            fit_mask: fit_mask_vec,
            clim_slice: (clim_slice_start, c_stop),
            det_slice: (det_slice_start, d_stop),
        })
    }

    fn compute_climatology<'py>(
        &self,
        py: Python<'py>,
        clim_data: FloatArray2<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let mat = clim_data.as_matrix("clim_data")?;
        if mat.ncols() != self.n_clim {
            return Err(PyValueError::new_err("clim_data ncols must match clim_years length"));
        }
        let n_pixels = mat.nrows();
        let yl = self.year_len;
        let (s, h, l) = py.detach(|| {
            let (mut s, mut h, mut l) = (vec![f64::NAN; n_pixels * yl], vec![f64::NAN; n_pixels * yl], vec![f64::NAN; n_pixels * yl]);
            if n_pixels > 0 {
                s.par_chunks_mut(yl)
                    .zip(h.par_chunks_mut(yl))
                    .zip(l.par_chunks_mut(yl))
                    .enumerate()
                    .for_each_init(
                        || (Vec::<f64>::with_capacity(512), Vec::<f64>::with_capacity(self.n_clim)),
                        |(scratch, row_buf), (p, ((so, ho), lo))| {
                            let res = compute_pixel_climatology(
                                mat.row_f64(p, row_buf), &self.plan, self.pctile_high, self.pctile_low,
                                self.smooth_width, self.min_valid_fraction, &self.compat, scratch,
                            );
                            so.copy_from_slice(&res.seas);
                            ho.copy_from_slice(&res.thresh_high);
                            lo.copy_from_slice(&res.thresh_low);
                        },
                    );
            }
            (s, h, l)
        });
        Ok((
            Array2::from_shape_vec((n_pixels, yl), s).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py),
            Array2::from_shape_vec((n_pixels, yl), h).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py),
            Array2::from_shape_vec((n_pixels, yl), l).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py),
        ))
    }

    fn compute_detrended_climatology<'py>(
        &self,
        py: Python<'py>,
        full_data: FloatArray2<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let mat = full_data.as_matrix("full_data")?;
        if self.fit_mask.is_empty() || mat.ncols() != self.fit_mask.len() {
            return Err(PyValueError::new_err("full_data ncols must match fit_mask length"));
        }
        let n_pixels = mat.nrows();
        let yl = self.year_len;
        let (c0, c1) = self.clim_slice;
        let (s, h, l, tr) = py.detach(|| {
            let (mut s, mut h, mut l, mut tr) = (
                vec![f64::NAN; n_pixels * yl],
                vec![f64::NAN; n_pixels * yl],
                vec![f64::NAN; n_pixels * yl],
                vec![f64::NAN; n_pixels * 3],
            );
            if n_pixels > 0 {
                s.par_chunks_mut(yl)
                    .zip(h.par_chunks_mut(yl))
                    .zip(l.par_chunks_mut(yl))
                    .zip(tr.par_chunks_mut(3))
                    .enumerate()
                    .for_each_init(
                        || (Vec::<f64>::with_capacity(512), Vec::<f64>::with_capacity(mat.ncols())),
                        |(scratch, row_buf), (p, (((so, ho), lo), to))| {
                            let (det_full, coeffs) = detrend_pixel(mat.row_f64(p, row_buf), &self.fit_mask, self.detrend_order);
                            to.copy_from_slice(&coeffs);
                            let res = compute_pixel_climatology(
                                &det_full[c0..c1], &self.plan, self.pctile_high, self.pctile_low,
                                self.smooth_width, self.min_valid_fraction, &self.compat, scratch,
                            );
                            so.copy_from_slice(&res.seas);
                            ho.copy_from_slice(&res.thresh_high);
                            lo.copy_from_slice(&res.thresh_low);
                        },
                    );
            }
            (s, h, l, tr)
        });
        Ok((
            Array2::from_shape_vec((n_pixels, yl), s).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py),
            Array2::from_shape_vec((n_pixels, yl), h).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py),
            Array2::from_shape_vec((n_pixels, yl), l).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py),
            Array2::from_shape_vec((n_pixels, 3), tr).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py),
        ))
    }

    #[pyo3(signature = (clim_data, det_data, active_pixel_indices, total_grid_pixels, *, return_clim=false, return_intensity_ts=false, return_daily_category_ts=false, return_event_category_ts=false))]
    #[allow(clippy::too_many_arguments)]
    fn run_tile<'py>(
        &self,
        py: Python<'py>,
        clim_data: FloatArray2<'py>,
        det_data: FloatArray2<'py>,
        active_pixel_indices: PyReadonlyArray1<'py, i64>,
        total_grid_pixels: usize,
        return_clim: bool,
        return_intensity_ts: bool,
        return_daily_category_ts: bool,
        return_event_category_ts: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let c_mat = clim_data.as_matrix("clim_data")?;
        let d_mat = det_data.as_matrix("det_data")?;
        let active_idx = active_pixel_indices.as_slice().map_err(|_| PyValueError::new_err("active_pixel_indices must be contiguous"))?;
        let (n_active, n_det) = (d_mat.nrows(), d_mat.ncols());
        if c_mat.nrows() != n_active {
            return Err(PyValueError::new_err("clim_data and det_data must have matching n_active rows"));
        }
        if c_mat.ncols() != self.n_clim || n_det != self.det_doys.len() {
            return Err(PyValueError::new_err("Tile time length does not match compiled FixedPlan"));
        }
        validate_active_indices(active_idx, n_active, total_grid_pixels)?;

        let flags = TileFlags::with_clim(
            self.detect_params.event_mode, self.year_len, return_clim, false,
            return_intensity_ts, return_daily_category_ts, return_event_category_ts,
        );
        let bufs = py.detach(|| {
            run_tile_parallel(
                n_active, n_det, total_grid_pixels, active_idx, flags, true,
                || (Vec::with_capacity(512), Vec::with_capacity(n_det), Vec::with_capacity(n_det), Vec::with_capacity(n_det), Vec::with_capacity(n_det), Vec::with_capacity(self.n_clim), Vec::with_capacity(n_det)),
                |(scratch, st, ht, lt, pad, cbuf, dbuf), p, outs| {
                    self.run_clim_and_detect_pixel(
                        c_mat.row_f64(p, cbuf), d_mat.row_f64(p, dbuf), active_idx[p], outs,
                        scratch, st, ht, lt, pad,
                    )
                },
            )
        });
        bufs.into_pydict(py, n_active, n_det, flags, None, self.year_len)
    }

    #[pyo3(signature = (full_data, active_pixel_indices, total_grid_pixels, *, return_clim=false, return_intensity_ts=false, return_daily_category_ts=false, return_event_category_ts=false))]
    #[allow(clippy::too_many_arguments)]
    fn run_detrended_tile<'py>(
        &self,
        py: Python<'py>,
        full_data: FloatArray2<'py>,
        active_pixel_indices: PyReadonlyArray1<'py, i64>,
        total_grid_pixels: usize,
        return_clim: bool,
        return_intensity_ts: bool,
        return_daily_category_ts: bool,
        return_event_category_ts: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let f_mat = full_data.as_matrix("full_data")?;
        let active_idx = active_pixel_indices.as_slice().map_err(|_| PyValueError::new_err("active_pixel_indices must be contiguous"))?;
        let (n_active, n_full) = (f_mat.nrows(), f_mat.ncols());
        if self.fit_mask.is_empty() || n_full != self.fit_mask.len() {
            return Err(PyValueError::new_err("full_data ncols must match fit_mask length"));
        }
        let n_det = self.det_doys.len();
        validate_active_indices(active_idx, n_active, total_grid_pixels)?;

        let (c0, c1) = self.clim_slice;
        let (d0, d1) = self.det_slice;
        let flags = TileFlags::with_clim(
            self.detect_params.event_mode, self.year_len, return_clim, true,
            return_intensity_ts, return_daily_category_ts, return_event_category_ts,
        );
        let bufs = py.detach(|| {
            run_tile_parallel(
                n_active, n_det, total_grid_pixels, active_idx, flags, true,
                || (Vec::with_capacity(512), Vec::with_capacity(n_det), Vec::with_capacity(n_det), Vec::with_capacity(n_det), Vec::with_capacity(n_det), Vec::with_capacity(n_full)),
                |(scratch, st, ht, lt, pad, fbuf), p, mut outs| {
                    let (det_full, coeffs) = detrend_pixel(f_mat.row_f64(p, fbuf), &self.fit_mask, self.detrend_order);
                    if let Some(tr) = outs.trend.take() {
                        tr.copy_from_slice(&coeffs);
                    }
                    self.run_clim_and_detect_pixel(
                        &det_full[c0..c1], &det_full[d0..d1], active_idx[p], outs,
                        scratch, st, ht, lt, pad,
                    )
                },
            )
        });
        bufs.into_pydict(py, n_active, n_det, flags, None, self.year_len)
    }
}

// ============================================================================
// Compiled Plan 2: MovingPlan (cached MovingSchedule across all tiles)
// ============================================================================

/// Compiled moving-baseline plan caching deduplicated `MovingSchedule` windows across tiles.
#[pyclass(frozen)]
struct MovingPlan {
    sched: MovingSchedule,
    n_clim: usize,
    year_len: usize,
    pctile_high: f64,
    pctile_low: f64,
    smooth_width: usize,
    min_valid_fraction: f64,
    compat: CompatConfig,
    detect_params: DetectParams,
    det_doys: Vec<u16>,
}

#[pymethods]
impl MovingPlan {
    #[new]
    #[pyo3(signature = (
        clim_years, clim_doys, det_years, year_len, window_years, center, boundary,
        pctile_high, pctile_low, window_half_width, smooth_width, min_valid_fraction,
        quantile_method, feb29_mode, smooth_nan_policy, *, det_doys=None,
        event_mode="high", min_duration=5, max_gap=2, max_pad_length=0,
        lower_bound=None, bound_policy="flag", strict_inequality=true, var_ddof=1
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        clim_years: PyReadonlyArray1<'_, i32>,
        clim_doys: PyReadonlyArray1<'_, u16>,
        det_years: PyReadonlyArray1<'_, i32>,
        year_len: usize,
        window_years: usize,
        center: bool,
        boundary: &str,
        pctile_high: f64,
        pctile_low: f64,
        window_half_width: usize,
        smooth_width: usize,
        min_valid_fraction: f64,
        quantile_method: &str,
        feb29_mode: &str,
        smooth_nan_policy: &str,
        det_doys: Option<PyReadonlyArray1<'_, u16>>,
        event_mode: &str,
        min_duration: usize,
        max_gap: usize,
        max_pad_length: usize,
        lower_bound: Option<f64>,
        bound_policy: &str,
        strict_inequality: bool,
        var_ddof: usize,
    ) -> PyResult<Self> {
        let c_years = clim_years.as_slice().map_err(|_| PyValueError::new_err("clim_years must be contiguous"))?;
        let c_doys = clim_doys.as_slice().map_err(|_| PyValueError::new_err("clim_doys must be contiguous"))?;
        let d_years = det_years.as_slice().map_err(|_| PyValueError::new_err("det_years must be contiguous"))?;
        let n_clim = c_years.len();
        if c_doys.len() != n_clim {
            return Err(PyValueError::new_err("clim_years and clim_doys must have equal length"));
        }
        validate_doys(c_doys, year_len, "clim_doys")?;
        if window_years == 0 {
            return Err(PyValueError::new_err("window_years must be >= 1"));
        }

        let d_doys_vec = match det_doys {
            Some(ref arr) => {
                let s = arr.as_slice().map_err(|_| PyValueError::new_err("det_doys must be contiguous"))?;
                if s.len() != d_years.len() {
                    return Err(PyValueError::new_err("det_doys length must match det_years"));
                }
                validate_doys(s, year_len, "det_doys")?;
                s.to_vec()
            }
            None => Vec::new(),
        };
        let bnd = parse_moving_boundary(boundary)?;
        let (compat, detect_params) = parse_compat_and_detect(
            quantile_method, feb29_mode, smooth_nan_policy, strict_inequality, var_ddof,
            event_mode, min_duration, max_gap, max_pad_length, lower_bound, bound_policy,
        )?;
        let sched = py.detach(|| {
            build_moving_schedule(
                c_years, c_doys, d_years, year_len, window_years, center, bnd,
                window_half_width, compat.feb29_mode,
            )
        });
        Ok(Self {
            sched,
            n_clim,
            year_len,
            pctile_high,
            pctile_low,
            smooth_width,
            min_valid_fraction,
            compat,
            detect_params,
            det_doys: d_doys_vec,
        })
    }

    #[getter]
    fn unique_years<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        Array1::from_vec(self.sched.det_unique_years.clone()).into_pyarray(py)
    }

    fn compute_climatology<'py>(
        &self,
        py: Python<'py>,
        clim_data: FloatArray2<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
    )> {
        let mat = clim_data.as_matrix("clim_data")?;
        if mat.ncols() != self.n_clim {
            return Err(PyValueError::new_err("clim_data ncols must match clim_years length"));
        }
        let (n_pixels, n_uy, yl) = (mat.nrows(), self.sched.det_unique_years.len(), self.year_len);
        let stride = n_uy * yl;
        let (s, h, l) = py.detach(|| {
            let (mut s, mut h, mut l) = (vec![f64::NAN; n_pixels * stride], vec![f64::NAN; n_pixels * stride], vec![f64::NAN; n_pixels * stride]);
            if n_pixels > 0 && stride > 0 {
                let n_win = self.sched.plans.len();
                s.par_chunks_mut(stride)
                    .zip(h.par_chunks_mut(stride))
                    .zip(l.par_chunks_mut(stride))
                    .enumerate()
                    .for_each_init(
                        || (Vec::<f64>::with_capacity(512), Vec::<f64>::with_capacity(self.n_clim)),
                        |(scratch, row_buf), (p, ((so, ho), lo))| {
                            let row = mat.row_f64(p, row_buf);
                            let win_clims: Vec<_> = (0..n_win)
                                .map(|w| {
                                    compute_pixel_climatology(
                                        row, &self.sched.plans[w], self.pctile_high, self.pctile_low,
                                        self.smooth_width, self.min_valid_fraction, &self.compat, scratch,
                                    )
                                })
                                .collect();
                            for (yi, &wi) in self.sched.year_window_idx.iter().enumerate() {
                                let dst = yi * yl..(yi + 1) * yl;
                                so[dst.clone()].copy_from_slice(&win_clims[wi].seas);
                                ho[dst.clone()].copy_from_slice(&win_clims[wi].thresh_high);
                                lo[dst].copy_from_slice(&win_clims[wi].thresh_low);
                            }
                        },
                    );
            }
            (s, h, l)
        });
        Ok((
            self.unique_years(py),
            Array3::from_shape_vec((n_pixels, n_uy, yl), s).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py),
            Array3::from_shape_vec((n_pixels, n_uy, yl), h).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py),
            Array3::from_shape_vec((n_pixels, n_uy, yl), l).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py),
        ))
    }

    #[pyo3(signature = (clim_data, det_data, active_pixel_indices, total_grid_pixels, *, return_clim=false, return_intensity_ts=false, return_daily_category_ts=false, return_event_category_ts=false))]
    #[allow(clippy::too_many_arguments)]
    fn run_tile<'py>(
        &self,
        py: Python<'py>,
        clim_data: FloatArray2<'py>,
        det_data: FloatArray2<'py>,
        active_pixel_indices: PyReadonlyArray1<'py, i64>,
        total_grid_pixels: usize,
        return_clim: bool,
        return_intensity_ts: bool,
        return_daily_category_ts: bool,
        return_event_category_ts: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let c_mat = clim_data.as_matrix("clim_data")?;
        let d_mat = det_data.as_matrix("det_data")?;
        let active_idx = active_pixel_indices.as_slice().map_err(|_| PyValueError::new_err("active_pixel_indices must be contiguous"))?;
        let (n_active, n_det) = (d_mat.nrows(), d_mat.ncols());
        if c_mat.nrows() != n_active {
            return Err(PyValueError::new_err("clim_data and det_data must have matching n_active rows"));
        }
        if c_mat.ncols() != self.n_clim || n_det != self.det_doys.len() {
            return Err(PyValueError::new_err("Tile time length does not match compiled MovingPlan"));
        }
        validate_active_indices(active_idx, n_active, total_grid_pixels)?;

        let (n_uy, yl, n_win) = (self.sched.det_unique_years.len(), self.year_len, self.sched.plans.len());
        let flags = TileFlags::with_clim(
            self.detect_params.event_mode, n_uy * yl, return_clim, false,
            return_intensity_ts, return_daily_category_ts, return_event_category_ts,
        );
        let bufs = py.detach(|| {
            run_tile_parallel(
                n_active, n_det, total_grid_pixels, active_idx, flags, false,
                || (Vec::with_capacity(512), Vec::with_capacity(n_det), Vec::with_capacity(n_det), Vec::with_capacity(n_det), Vec::with_capacity(n_det), Vec::with_capacity(self.n_clim), Vec::with_capacity(n_det)),
                |(scratch, st, ht, lt, pad, cbuf, dbuf), p, outs| {
                    let c_row = c_mat.row_f64(p, cbuf);
                    let win_clims: Vec<_> = (0..n_win)
                        .map(|w| {
                            compute_pixel_climatology(
                                c_row, &self.sched.plans[w], self.pctile_high, self.pctile_low,
                                self.smooth_width, self.min_valid_fraction, &self.compat, scratch,
                            )
                        })
                        .collect();
                    st.clear();
                    ht.clear();
                    lt.clear();
                    for t in 0..n_det {
                        let wi = self.sched.det_window_idx[t];
                        let di = (self.det_doys[t] as usize) - 1;
                        st.push(win_clims[wi].seas[di]);
                        ht.push(win_clims[wi].thresh_high[di]);
                        lt.push(win_clims[wi].thresh_low[di]);
                    }
                    let evs = detect_pixel_with_curves(
                        active_idx[p], d_mat.row_f64(p, dbuf), st, ht, lt,
                        &self.detect_params, pad, outs.intensity, outs.intensity_high, outs.intensity_low,
                        outs.daily_cat, outs.event_cat,
                    );
                    if let Some(ps) = outs.clim {
                        for (yi, &wi) in self.sched.year_window_idx.iter().enumerate() {
                            ps[yi * yl..(yi + 1) * yl].copy_from_slice(&win_clims[wi].seas);
                        }
                    }
                    if let Some(ph) = outs.hi {
                        for (yi, &wi) in self.sched.year_window_idx.iter().enumerate() {
                            ph[yi * yl..(yi + 1) * yl].copy_from_slice(&win_clims[wi].thresh_high);
                        }
                    }
                    if let Some(pl) = outs.lo {
                        for (yi, &wi) in self.sched.year_window_idx.iter().enumerate() {
                            pl[yi * yl..(yi + 1) * yl].copy_from_slice(&win_clims[wi].thresh_low);
                        }
                    }
                    evs
                },
            )
        });
        let out = bufs.into_pydict(py, n_active, n_det, flags, Some(n_uy), yl)?;
        out.set_item("unique_years", self.unique_years(py))?;
        Ok(out)
    }
}

// ============================================================================
// Compiled Plan 3: CurvesPlan (FixedThreshold + ExternalClimatology)
// ============================================================================

/// Compiled detection-only plan for precomputed external climatologies or fixed thresholds.
#[pyclass(frozen)]
struct CurvesPlan {
    params: DetectParams,
    det_doys: Option<Vec<u16>>,
    year_len: usize,
}

#[pymethods]
impl CurvesPlan {
    #[new]
    #[pyo3(signature = (event_mode, min_duration, max_gap, max_pad_length, lower_bound, bound_policy, strict_inequality, var_ddof, *, det_doys=None, year_len=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        event_mode: &str,
        min_duration: usize,
        max_gap: usize,
        max_pad_length: usize,
        lower_bound: Option<f64>,
        bound_policy: &str,
        strict_inequality: bool,
        var_ddof: usize,
        det_doys: Option<PyReadonlyArray1<'_, u16>>,
        year_len: Option<usize>,
    ) -> PyResult<Self> {
        let (_, params) = parse_compat_and_detect(
            "type7", "interp_59_61", "omit", strict_inequality, var_ddof,
            event_mode, min_duration, max_gap, max_pad_length, lower_bound, bound_policy,
        )?;
        let (d_doys, yl) = match det_doys {
            Some(ref arr) => {
                let s = arr.as_slice().map_err(|_| PyValueError::new_err("det_doys must be contiguous"))?;
                let y = year_len.ok_or_else(|| PyValueError::new_err("year_len required with det_doys"))?;
                validate_doys(s, y, "det_doys")?;
                (Some(s.to_vec()), y)
            }
            None => (None, year_len.unwrap_or(0)),
        };
        Ok(Self { params, det_doys: d_doys, year_len: yl })
    }

    #[pyo3(signature = (det_data, seas_ts, thresh_high_ts, thresh_low_ts, active_pixel_indices, total_grid_pixels, *, return_intensity_ts=false, return_daily_category_ts=false, return_event_category_ts=false))]
    #[allow(clippy::too_many_arguments)]
    fn run_curves<'py>(
        &self,
        py: Python<'py>,
        det_data: FloatArray2<'py>,
        seas_ts: PyReadonlyArray2<'py, f64>,
        thresh_high_ts: PyReadonlyArray2<'py, f64>,
        thresh_low_ts: PyReadonlyArray2<'py, f64>,
        active_pixel_indices: PyReadonlyArray1<'py, i64>,
        total_grid_pixels: usize,
        return_intensity_ts: bool,
        return_daily_category_ts: bool,
        return_event_category_ts: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let d_mat = det_data.as_matrix("det_data")?;
        let (n_active, n_det) = (d_mat.nrows(), d_mat.ncols());
        let s_slice = as_c_slice_2d(&seas_ts, "seas_ts")?;
        let h_slice = as_c_slice_2d(&thresh_high_ts, "thresh_high_ts")?;
        let l_slice = as_c_slice_2d(&thresh_low_ts, "thresh_low_ts")?;
        let active_idx = active_pixel_indices.as_slice().map_err(|_| PyValueError::new_err("active_pixel_indices must be contiguous"))?;

        let expected = [n_active, n_det];
        if seas_ts.shape() != expected || thresh_high_ts.shape() != expected || thresh_low_ts.shape() != expected {
            return Err(PyValueError::new_err("seas_ts, thresh_high_ts, and thresh_low_ts must match det_data shape"));
        }
        validate_active_indices(active_idx, n_active, total_grid_pixels)?;

        let flags = TileFlags::ts_only(self.params.event_mode, return_intensity_ts, return_daily_category_ts, return_event_category_ts);
        let bufs = py.detach(|| {
            run_tile_parallel(
                n_active, n_det, total_grid_pixels, active_idx, flags, false,
                || (Vec::with_capacity(n_det), Vec::with_capacity(n_det)),
                |(pad, rbuf), p, outs| {
                    let sl = p * n_det..(p + 1) * n_det;
                    detect_pixel_with_curves(
                        active_idx[p], d_mat.row_f64(p, rbuf),
                        &s_slice[sl.clone()], &h_slice[sl.clone()], &l_slice[sl],
                        &self.params, pad, outs.intensity, outs.intensity_high, outs.intensity_low,
                        outs.daily_cat, outs.event_cat,
                    )
                },
            )
        });
        bufs.into_pydict(py, n_active, n_det, flags, None, 0)
    }

    #[pyo3(signature = (det_data, seas_doy, thresh_high_doy, thresh_low_doy, active_pixel_indices, total_grid_pixels, *, return_intensity_ts=false, return_daily_category_ts=false, return_event_category_ts=false))]
    #[allow(clippy::too_many_arguments)]
    fn run_doy_curves<'py>(
        &self,
        py: Python<'py>,
        det_data: FloatArray2<'py>,
        seas_doy: PyReadonlyArray2<'py, f64>,
        thresh_high_doy: PyReadonlyArray2<'py, f64>,
        thresh_low_doy: PyReadonlyArray2<'py, f64>,
        active_pixel_indices: PyReadonlyArray1<'py, i64>,
        total_grid_pixels: usize,
        return_intensity_ts: bool,
        return_daily_category_ts: bool,
        return_event_category_ts: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let doys = self.det_doys.as_deref().ok_or_else(|| PyValueError::new_err("CurvesPlan was constructed without det_doys"))?;
        let d_mat = det_data.as_matrix("det_data")?;
        let (n_active, n_det) = (d_mat.nrows(), d_mat.ncols());
        if n_det != doys.len() {
            return Err(PyValueError::new_err("det_data ncols must match det_doys length"));
        }
        let s_slice = as_c_slice_2d(&seas_doy, "seas_doy")?;
        let h_slice = as_c_slice_2d(&thresh_high_doy, "thresh_high_doy")?;
        let l_slice = as_c_slice_2d(&thresh_low_doy, "thresh_low_doy")?;
        let active_idx = active_pixel_indices.as_slice().map_err(|_| PyValueError::new_err("active_pixel_indices must be contiguous"))?;

        let yl = self.year_len;
        let expected = [n_active, yl];
        if seas_doy.shape() != expected || thresh_high_doy.shape() != expected || thresh_low_doy.shape() != expected {
            return Err(PyValueError::new_err("seas_doy, thresh_high_doy, and thresh_low_doy must have shape (n_active, year_len)"));
        }
        validate_active_indices(active_idx, n_active, total_grid_pixels)?;

        let flags = TileFlags::ts_only(self.params.event_mode, return_intensity_ts, return_daily_category_ts, return_event_category_ts);
        let bufs = py.detach(|| {
            run_tile_parallel(
                n_active, n_det, total_grid_pixels, active_idx, flags, false,
                || (Vec::with_capacity(n_det), Vec::with_capacity(n_det), Vec::with_capacity(n_det), Vec::with_capacity(n_det), Vec::with_capacity(n_det)),
                |(st, ht, lt, pad, rbuf), p, outs| {
                    let sl = p * yl..(p + 1) * yl;
                    expand_doy_curves(doys, &s_slice[sl.clone()], &h_slice[sl.clone()], &l_slice[sl], st, ht, lt);
                    detect_pixel_with_curves(
                        active_idx[p], d_mat.row_f64(p, rbuf), st, ht, lt,
                        &self.params, pad, outs.intensity, outs.intensity_high, outs.intensity_low,
                        outs.daily_cat, outs.event_cat,
                    )
                },
            )
        });
        bufs.into_pydict(py, n_active, n_det, flags, None, 0)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CATEGORY_MISSING", CATEGORY_MISSING)?;
    m.add_class::<FixedPlan>()?;
    m.add_class::<MovingPlan>()?;
    m.add_class::<CurvesPlan>()?;
    Ok(())
}
