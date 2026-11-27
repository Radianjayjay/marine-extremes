//! Climatology, detrending, and moving-baseline computation engines.

use std::collections::HashMap;

use crate::compat::{CompatConfig, DetrendOrder, Feb29Mode, MovingBoundary, SmoothNanPolicy};
use crate::quantile::{quantile_sorted, sort_finite};

/// A precomputed sample reference into a 1D pixel time series.
/// Using `u32` halves cache footprint for multi-decade / multi-century moving baselines.
#[derive(Debug, Clone, Copy)]
pub enum SampleRef {
    Direct(u32),
    /// Arithmetic mean of two indices (used for synthetic Feb 29 interpolation in non-leap years).
    Avg(u32, u32),
}

/// Precomputed DOY sample index plan shared across all spatial/depth pixels.
#[derive(Debug, Clone)]
pub struct WindowPlan {
    pub year_len: usize,
    /// For each 0-based DOY `0..year_len`, list of sample references to gather.
    pub doy_samples: Vec<Vec<SampleRef>>,
    /// If true, DOY index 59 (1-based DOY 60, Feb 29) is interpolated from DOY 58 and 60 before smoothing.
    pub interp_feb29_after: bool,
}

/// Result of computing a single climatology curve for one pixel.
#[derive(Debug, Clone)]
pub struct PixelClimatology {
    pub seas: Vec<f64>,
    pub thresh_high: Vec<f64>,
    pub thresh_low: Vec<f64>,
}

/// Schedule mapping each detection time step to one of several deduplicated moving climatology windows.
#[derive(Debug, Clone)]
pub struct MovingSchedule {
    /// Distinct (win_start_year, win_end_year) windows.
    #[allow(dead_code)]
    pub windows: Vec<(i32, i32)>,
    /// Precomputed WindowPlan for each distinct window.
    pub plans: Vec<WindowPlan>,
    /// For each detection time step `t`, index into `windows`/`plans`.
    pub det_window_idx: Vec<usize>,
    /// Unique sorted detection years corresponding to `year_window_idx`.
    pub det_unique_years: Vec<i32>,
    /// For each unique detection year, index into `windows`/`plans`.
    pub year_window_idx: Vec<usize>,
}

/// Detrend a single pixel's 1D time series using OLS (linear or quadratic), ignoring NaNs.
///
/// The fitted trend is centered around its mean over valid `fit_mask` points so that
/// the physical level (e.g., SST in deg C for freezing-point checks) is preserved while
/// secular trend is completely removed.
///
/// Returns `(detrended_series, coeffs)` where `coeffs = [c0, c1_per_step, c2_per_step2]`
/// in centered-step units (`dt = t - t_ref`).
pub fn detrend_pixel(
    values: &[f64],
    fit_mask: &[u8],
    order: DetrendOrder,
) -> (Vec<f64>, [f64; 3]) {
    let n = values.len();
    match order {
        DetrendOrder::None => (values.to_vec(), [0.0, 0.0, 0.0]),
        DetrendOrder::Linear => detrend_linear(values, fit_mask, n),
        DetrendOrder::Quadratic => detrend_quadratic(values, fit_mask, n),
    }
}

/// Fit linear OLS trend on valid `fit_mask` points and subtract centered slope.
fn detrend_linear(values: &[f64], fit_mask: &[u8], n: usize) -> (Vec<f64>, [f64; 3]) {
    let mut count = 0usize;
    let mut sum_t = 0.0f64;
    let mut sum_y = 0.0f64;
    for t in 0..n {
        if fit_mask[t] != 0 && values[t].is_finite() {
            count += 1;
            sum_t += t as f64;
            sum_y += values[t];
        }
    }
    if count < 2 {
        return (vec![f64::NAN; n], [f64::NAN, f64::NAN, 0.0]);
    }
    let nf = count as f64;
    let mean_t = sum_t / nf;
    let mean_y = sum_y / nf;

    let mut s_tt = 0.0f64;
    let mut s_ty = 0.0f64;
    for t in 0..n {
        if fit_mask[t] != 0 && values[t].is_finite() {
            let dt = t as f64 - mean_t;
            let dy = values[t] - mean_y;
            s_tt += dt * dt;
            s_ty += dt * dy;
        }
    }
    if s_tt <= 1e-18 {
        return (values.to_vec(), [mean_y, 0.0, 0.0]);
    }
    let slope = s_ty / s_tt;
    let mut out = vec![f64::NAN; n];
    for t in 0..n {
        let v = values[t];
        if v.is_finite() {
            // Subtract slope * (t - mean_t) so mean over fit_mask stays equal to mean_y.
            out[t] = v - slope * (t as f64 - mean_t);
        }
    }
    (out, [mean_y, slope, 0.0])
}

/// Fit quadratic OLS trend in normalized coordinates (falling back to linear if rank-deficient).
fn detrend_quadratic(values: &[f64], fit_mask: &[u8], n: usize) -> (Vec<f64>, [f64; 3]) {
    let mut count = 0usize;
    let mut sum_t = 0.0f64;
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    for t in 0..n {
        if fit_mask[t] != 0 && values[t].is_finite() {
            let tf = t as f64;
            count += 1;
            sum_t += tf;
            if tf < t_min {
                t_min = tf;
            }
            if tf > t_max {
                t_max = tf;
            }
        }
    }
    if count < 3 {
        return detrend_linear(values, fit_mask, n);
    }
    let mean_t = sum_t / (count as f64);
    let scale = ((t_max - t_min) * 0.5).max(1.0);

    // Build normalized 3x3 normal equations for u = (t - mean_t) / scale:
    // y ~ a0 + a1*u + a2*u^2
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s3 = 0.0f64;
    let mut s4 = 0.0f64;
    let mut sy0 = 0.0f64;
    let mut sy1 = 0.0f64;
    let mut sy2 = 0.0f64;

    for t in 0..n {
        if fit_mask[t] != 0 && values[t].is_finite() {
            let u = (t as f64 - mean_t) / scale;
            let u2 = u * u;
            let y = values[t];
            s0 += 1.0;
            s1 += u;
            s2 += u2;
            s3 += u2 * u;
            s4 += u2 * u2;
            sy0 += y;
            sy1 += u * y;
            sy2 += u2 * y;
        }
    }

    let Some([a0, a1, a2]) = solve_3x3(
        [[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]],
        [sy0, sy1, sy2],
    ) else {
        return detrend_linear(values, fit_mask, n);
    };

    // Mean of fitted polynomial over valid fit_mask equals sy0 / s0 by OLS normal equations.
    let trend_ref = sy0 / s0;

    let mut out = vec![f64::NAN; n];
    for t in 0..n {
        let v = values[t];
        if v.is_finite() {
            let u = (t as f64 - mean_t) / scale;
            let trend_val = a0 + a1 * u + a2 * u * u;
            out[t] = v - (trend_val - trend_ref);
        }
    }
    (out, [a0, a1 / scale, a2 / (scale * scale)])
}

/// Solve 3x3 linear system A x = b with partial pivoting.
fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut pivot_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..3 {
            let v = a[row][col].abs();
            if v > max_val {
                max_val = v;
                pivot_row = row;
            }
        }
        if max_val < 1e-18 || !max_val.is_finite() {
            return None;
        }
        if pivot_row != col {
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }
        let piv = a[col][col];
        for row in (col + 1)..3 {
            let factor = a[row][col] / piv;
            for k in col..3 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        let mut sum = b[i];
        for j in (i + 1)..3 {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
        if !x[i].is_finite() {
            return None;
        }
    }
    Some(x)
}

/// Build a reusable `WindowPlan` once from calendar arrays before parallel pixel processing.
///
/// - `clim_years`: year integer for each climatology time step
/// - `clim_doys`: 1-based DOY (`1..=year_len`). In 366-day leap-aligned Gregorian calendars,
///   non-leap years skip DOY 60 (Feb 28 = 59, Mar 1 = 61) and leap years have Feb 29 = 60.
/// - `clim_in_period`: 1 if step belongs to the nominal climatology period, 0 if buffer-only.
pub fn build_window_plan(
    clim_years: &[i32],
    clim_doys: &[u16],
    clim_in_period: &[u8],
    year_len: usize,
    window_half_width: usize,
    feb29_mode: Feb29Mode,
) -> WindowPlan {
    let n = clim_doys.len();
    let effective_mode = if year_len != 366 {
        Feb29Mode::WindowDirect
    } else {
        feb29_mode
    };

    match effective_mode {
        Feb29Mode::Interp5961 | Feb29Mode::WindowDirect => {
            let interp_feb29_after = effective_mode == Feb29Mode::Interp5961;
            let mut doy_samples = vec![Vec::new(); year_len];
            let mut included = vec![false; n];

            for d_zero in 0..year_len {
                let d_one = (d_zero + 1) as u16;
                if interp_feb29_after && d_one == 60 {
                    continue;
                }
                included.fill(false);
                let mut any_match = false;
                for t in 0..n {
                    if clim_in_period[t] != 0 && clim_doys[t] == d_one {
                        any_match = true;
                        let lo = t.saturating_sub(window_half_width);
                        let hi = t.saturating_add(window_half_width).min(n.saturating_sub(1));
                        for k in lo..=hi {
                            included[k] = true;
                        }
                    }
                }
                if any_match {
                    let mut list = Vec::new();
                    for (k, &inc) in included.iter().enumerate() {
                        if inc {
                            list.push(SampleRef::Direct(k as u32));
                        }
                    }
                    doy_samples[d_zero] = list;
                }
            }
            WindowPlan {
                year_len,
                doy_samples,
                interp_feb29_after,
            }
        }
        Feb29Mode::OliverMatrix => {
            // Replicates ecjoliver/marineHeatWaves:
            // Build (n_years, 366) grid over clim_in_period years, fill non-leap Feb 29 as
            // 0.5*(Feb 28 + Mar 1), wrap +/- W circularly over 0..366 per year, and interpolate
            // DOY 60 from DOY 59 & 61 after computing raw clim/thresh.
            let mut year_map: HashMap<i32, usize> = HashMap::new();
            let mut years_order: Vec<i32> = Vec::new();
            for t in 0..n {
                if clim_in_period[t] != 0 {
                    let y = clim_years[t];
                    if !year_map.contains_key(&y) {
                        year_map.insert(y, years_order.len());
                        years_order.push(y);
                    }
                }
            }
            let ny = years_order.len();
            let mut grid: Vec<Vec<Option<SampleRef>>> = vec![vec![None; 366]; ny];
            for t in 0..n {
                if clim_in_period[t] != 0 {
                    let yi = year_map[&clim_years[t]];
                    let d_zero = (clim_doys[t] as usize).saturating_sub(1);
                    if d_zero < 366 {
                        grid[yi][d_zero] = Some(SampleRef::Direct(t as u32));
                    }
                }
            }
            // Fill non-leap Feb 29 (index 59) as average of Feb 28 (58) and Mar 1 (60)
            for row in grid.iter_mut() {
                if row[59].is_none() {
                    if let (Some(SampleRef::Direct(a)), Some(SampleRef::Direct(b))) =
                        (row[58], row[60])
                    {
                        row[59] = Some(SampleRef::Avg(a, b));
                    }
                }
            }
            let mut doy_samples = vec![Vec::new(); 366];
            let w = window_half_width as isize;
            for d_zero in 0..366 {
                if d_zero == 59 {
                    continue;
                }
                let mut list = Vec::with_capacity(ny * (2 * window_half_width + 1));
                for yi in 0..ny {
                    for off in -w..=w {
                        let idx = (d_zero as isize + off).rem_euclid(366) as usize;
                        if let Some(sref) = grid[yi][idx] {
                            list.push(sref);
                        }
                    }
                }
                doy_samples[d_zero] = list;
            }
            WindowPlan {
                year_len: 366,
                doy_samples,
                interp_feb29_after: true,
            }
        }
        Feb29Mode::HeatwaveRSeries => {
            // Replicates heatwaveR::ts2clm:
            // Expand continuous time series so non-leap years have synthetic Feb 29 = 0.5*(Feb28+Mar1),
            // then slide a non-wrapping +/- W window along the expanded series (clamped at ends),
            // and interpolate DOY 60 from 59 & 61.
            let mut expanded_refs: Vec<SampleRef> = Vec::with_capacity(n + n / 365 + 2);
            let mut expanded_doys: Vec<u16> = Vec::with_capacity(n + n / 365 + 2);
            let mut expanded_in_period: Vec<u8> = Vec::with_capacity(n + n / 365 + 2);

            for t in 0..n {
                expanded_refs.push(SampleRef::Direct(t as u32));
                expanded_doys.push(clim_doys[t]);
                expanded_in_period.push(clim_in_period[t]);

                // If this step is Feb 28 (DOY 59) and the next step is Mar 1 (DOY 61) of the SAME year,
                // insert synthetic Feb 29 (DOY 60).
                if clim_doys[t] == 59
                    && t + 1 < n
                    && clim_doys[t + 1] == 61
                    && clim_years[t] == clim_years[t + 1]
                {
                    expanded_refs.push(SampleRef::Avg(t as u32, (t + 1) as u32));
                    expanded_doys.push(60);
                    expanded_in_period.push(clim_in_period[t] & clim_in_period[t + 1]);
                }
            }

            let m = expanded_refs.len();
            let mut doy_samples = vec![Vec::new(); 366];
            let mut included = vec![false; m];

            for d_zero in 0..366 {
                let d_one = (d_zero + 1) as u16;
                if d_one == 60 {
                    continue;
                }
                included.fill(false);
                let mut any_match = false;
                for pos in 0..m {
                    if expanded_in_period[pos] != 0 && expanded_doys[pos] == d_one {
                        any_match = true;
                        let lo = pos.saturating_sub(window_half_width);
                        let hi = pos.saturating_add(window_half_width).min(m.saturating_sub(1));
                        for idx in lo..=hi {
                            included[idx] = true;
                        }
                    }
                }
                if any_match {
                    let mut list = Vec::new();
                    for (k, &inc) in included.iter().enumerate() {
                        if inc {
                            list.push(expanded_refs[k]);
                        }
                    }
                    doy_samples[d_zero] = list;
                }
            }
            WindowPlan {
                year_len: 366,
                doy_samples,
                interp_feb29_after: true,
            }
        }
    }
}

/// Build deduplicated moving-baseline schedule for customizable moving windows
/// (e.g., centered 11-year, centered 31-year, or trailing 30-year).
pub fn build_moving_schedule(
    clim_years: &[i32],
    clim_doys: &[u16],
    det_years: &[i32],
    year_len: usize,
    window_years: usize,
    center: bool,
    boundary: MovingBoundary,
    window_half_width: usize,
    feb29_mode: Feb29Mode,
) -> MovingSchedule {
    let y_min = *clim_years.iter().min().unwrap_or(&0) as i64;
    let y_max = *clim_years.iter().max().unwrap_or(&0) as i64;
    let total_span = (y_max - y_min + 1).max(1);
    let max_shrink_win = total_span.saturating_mul(2).max(1);
    let eff_win: i64 = match boundary {
        MovingBoundary::Clamp => (window_years.max(1) as u128).min(total_span as u128) as i64,
        MovingBoundary::Shrink => (window_years.max(1) as u128).min(max_shrink_win as u128) as i64,
    };

    let half_left = if center { (eff_win - 1) / 2 } else { eff_win - 1 };
    let half_right = if center { eff_win - 1 - half_left } else { 0 };

    let mut det_unique_years: Vec<i32> = det_years.to_vec();
    det_unique_years.sort_unstable();
    det_unique_years.dedup();

    let mut win_to_idx: HashMap<(i32, i32), usize> = HashMap::new();
    let mut windows: Vec<(i32, i32)> = Vec::new();
    let mut year_to_win_idx: HashMap<i32, usize> = HashMap::new();
    let mut year_window_idx: Vec<usize> = Vec::with_capacity(det_unique_years.len());

    for &y in &det_unique_years {
        let y64 = y as i64;
        let mut ws = y64 - half_left;
        let mut we = y64 + half_right;
        match boundary {
            MovingBoundary::Clamp => {
                if ws < y_min {
                    let shift = y_min - ws;
                    ws = y_min;
                    we = (we + shift).min(y_max);
                }
                if we > y_max {
                    let shift = we - y_max;
                    we = y_max;
                    ws = (ws - shift).max(y_min);
                }
            }
            MovingBoundary::Shrink => {
                ws = ws.max(y_min);
                we = we.min(y_max);
            }
        }
        let (ws_i32, we_i32) = (ws as i32, we as i32);
        let idx = *win_to_idx.entry((ws_i32, we_i32)).or_insert_with(|| {
            let new_idx = windows.len();
            windows.push((ws_i32, we_i32));
            new_idx
        });
        year_to_win_idx.insert(y, idx);
        year_window_idx.push(idx);
    }

    let det_window_idx: Vec<usize> = det_years.iter().map(|y| year_to_win_idx[y]).collect();

    let mut in_period = vec![0u8; clim_years.len()];
    let mut plans = Vec::with_capacity(windows.len());
    for &(ws, we) in &windows {
        for (k, &cy) in clim_years.iter().enumerate() {
            in_period[k] = if cy >= ws && cy <= we { 1 } else { 0 };
        }
        plans.push(build_window_plan(
            clim_years,
            clim_doys,
            &in_period,
            year_len,
            window_half_width,
            feb29_mode,
        ));
    }

    MovingSchedule {
        windows,
        plans,
        det_window_idx,
        det_unique_years,
        year_window_idx,
    }
}

/// Compute climatology (`seas`, `thresh_high`, `thresh_low`) for one pixel using a precomputed `WindowPlan`.
pub fn compute_pixel_climatology(
    clim_values: &[f64],
    plan: &WindowPlan,
    pctile_high: f64,
    pctile_low: f64,
    smooth_width: usize,
    min_valid_fraction: f64,
    compat: &CompatConfig,
    scratch: &mut Vec<f64>,
) -> PixelClimatology {
    let l = plan.year_len;
    let mut raw_seas = vec![f64::NAN; l];
    let mut raw_high = vec![f64::NAN; l];
    let mut raw_low = vec![f64::NAN; l];

    let need_high = pctile_high.is_finite();
    let need_low = pctile_low.is_finite();

    for d in 0..l {
        if plan.interp_feb29_after && d == 59 {
            continue;
        }
        let refs = &plan.doy_samples[d];
        let expected = refs.len();
        if expected == 0 {
            continue;
        }
        scratch.clear();
        let mut sum = 0.0f64;
        for &sref in refs {
            let val = match sref {
                SampleRef::Direct(i) => clim_values[i as usize],
                SampleRef::Avg(a, b) => {
                    let va = clim_values[a as usize];
                    let vb = clim_values[b as usize];
                    if va.is_finite() && vb.is_finite() {
                        0.5 * (va + vb)
                    } else if va.is_finite() {
                        va
                    } else if vb.is_finite() {
                        vb
                    } else {
                        f64::NAN
                    }
                }
            };
            if val.is_finite() {
                scratch.push(val);
                sum += val;
            }
        }
        let valid = scratch.len();
        if valid == 0 {
            continue;
        }
        let frac = (valid as f64) / (expected as f64);
        if frac + 1e-12 < min_valid_fraction {
            continue;
        }

        raw_seas[d] = sum / (valid as f64);
        if need_high || need_low {
            sort_finite(scratch);
            if need_high {
                raw_high[d] = quantile_sorted(scratch, pctile_high, compat.quantile_method);
            }
            if need_low {
                raw_low[d] = quantile_sorted(scratch, pctile_low, compat.quantile_method);
            }
        }
    }

    if plan.interp_feb29_after && l == 366 {
        raw_seas[59] = mean_pair_omitnan(raw_seas[58], raw_seas[60]);
        if need_high {
            raw_high[59] = mean_pair_omitnan(raw_high[58], raw_high[60]);
        }
        if need_low {
            raw_low[59] = mean_pair_omitnan(raw_low[58], raw_low[60]);
        }
    }

    let mut seas = circular_smooth(&raw_seas, smooth_width, compat.smooth_nan_policy);
    let mut thresh_high = if need_high {
        circular_smooth(&raw_high, smooth_width, compat.smooth_nan_policy)
    } else {
        raw_high.clone()
    };
    let mut thresh_low = if need_low {
        circular_smooth(&raw_low, smooth_width, compat.smooth_nan_policy)
    } else {
        raw_low.clone()
    };

    // Do not let neighboring finite days bleed across circular smoothing into DOYs that
    // were intentionally masked out (e.g., sea-ice DOYs failing min_valid_fraction).
    for d in 0..l {
        if !raw_seas[d].is_finite() {
            seas[d] = f64::NAN;
        }
        if need_high && !raw_high[d].is_finite() {
            thresh_high[d] = f64::NAN;
        }
        if need_low && !raw_low[d].is_finite() {
            thresh_low[d] = f64::NAN;
        }
    }

    PixelClimatology {
        seas,
        thresh_high,
        thresh_low,
    }
}

/// Mean of two optional finite values (omits a single NaN; matches MATLAB nanmean).
#[inline]
fn mean_pair_omitnan(a: f64, b: f64) -> f64 {
    match (a.is_finite(), b.is_finite()) {
        (true, true) => 0.5 * (a + b),
        (true, false) => a,
        (false, true) => b,
        (false, false) => f64::NAN,
    }
}

/// Circular moving average smoothing across a periodic DOY array.
pub fn circular_smooth(arr: &[f64], width: usize, policy: SmoothNanPolicy) -> Vec<f64> {
    let l = arr.len();
    if width <= 1 || l <= 1 {
        return arr.to_vec();
    }
    let half_left = (width / 2) as isize;
    let half_right = (width - 1) as isize - half_left;
    let li = l as isize;
    let mut out = vec![f64::NAN; l];

    for i in 0..l {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        let mut has_nan = false;
        let ii = i as isize;
        for off in -half_left..=half_right {
            let idx = (ii + off).rem_euclid(li) as usize;
            let v = arr[idx];
            if v.is_finite() {
                sum += v;
                count += 1;
            } else {
                has_nan = true;
            }
        }
        out[i] = match policy {
            SmoothNanPolicy::OmitNan => {
                if count > 0 {
                    sum / (count as f64)
                } else {
                    f64::NAN
                }
            }
            SmoothNanPolicy::PropagateNan => {
                if !has_nan && count > 0 {
                    sum / (count as f64)
                } else {
                    f64::NAN
                }
            }
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_detrend_removes_slope_and_preserves_mean() {
        let n = 1000;
        let vals: Vec<f64> = (0..n).map(|t| 15.0 + 0.01 * t as f64).collect();
        let mask = vec![1u8; n];
        let (det, coeffs) = detrend_pixel(&vals, &mask, DetrendOrder::Linear);
        let orig_mean = vals.iter().sum::<f64>() / n as f64;
        assert!((coeffs[1] - 0.01).abs() < 1e-12);
        for &v in &det {
            assert!((v - orig_mean).abs() < 1e-10);
        }
    }

    #[test]
    fn test_quadratic_detrend_removes_curve() {
        let n = 500;
        let vals: Vec<f64> = (0..n)
            .map(|t| {
                let x = t as f64;
                10.0 + 0.02 * x + 1e-4 * x * x
            })
            .collect();
        let mask = vec![1u8; n];
        let (det, _coeffs) = detrend_pixel(&vals, &mask, DetrendOrder::Quadratic);
        let orig_mean = vals.iter().sum::<f64>() / n as f64;
        for &v in &det {
            assert!((v - orig_mean).abs() < 1e-9);
        }
    }

    #[test]
    fn test_circular_smooth_constant() {
        let arr = vec![20.0; 366];
        let sm = circular_smooth(&arr, 31, SmoothNanPolicy::OmitNan);
        for v in sm {
            assert!((v - 20.0).abs() < 1e-12);
        }
    }
}
