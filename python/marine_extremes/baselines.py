"""Baseline specifications and polymorphic PreparedBaseline executors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import xarray as xr

from marine_extremes import _core
from marine_extremes.calendar import CalendarInfo, build_period_mask, effective_feb29_mode, encode_time_axis
from marine_extremes.cf import cf_auxiliary_vars, resolve_time_dim
from marine_extremes.compat import CompatProfile
from marine_extremes.streaming import (
    OpenedInput,
    SpatialTile,
    estimate_max_tile_pixels,
    iter_spatial_tiles,
    load_tile_as_pixels_x_time,
    open_data_input,
)

DateLike = str | int | Any
DataInput = Any


# ============================================================================
# Public baseline specification dataclasses
# ============================================================================


@dataclass(frozen=True)
class FixedBaseline:
    """Fixed reference-period climatology baseline (classic Hobday et al. 2016)."""

    start: DateLike | None = None
    end: DateLike | None = None
    clim_data: DataInput | None = None
    clim_time: Any | None = None
    clim_var: str | None = None


@dataclass(frozen=True)
class DetrendedBaseline:
    """Detrended baseline removing linear or quadratic trends before climatology and detection."""

    method: Literal["linear", "quadratic"] = "linear"
    start: DateLike | None = None
    end: DateLike | None = None
    fit_period: tuple[DateLike | None, DateLike | None] | None = None

    def __post_init__(self) -> None:
        if self.method not in ("linear", "quadratic"):
            raise ValueError(
                f"DetrendedBaseline method must be 'linear' or 'quadratic', got '{self.method}'"
            )


@dataclass(frozen=True)
class MovingBaseline:
    """Customizable moving/shifting baseline window (e.g., centered 11-year or 31-year)."""

    window_years: int = 31
    center: bool = True
    boundary: Literal["clamp", "shrink"] = "clamp"
    clim_data: DataInput | None = None
    clim_time: Any | None = None
    clim_var: str | None = None

    def __post_init__(self) -> None:
        if self.window_years < 1:
            raise ValueError("MovingBaseline.window_years must be >= 1")
        if self.boundary not in ("clamp", "shrink"):
            raise ValueError("MovingBaseline.boundary must be 'clamp' or 'shrink'")


@dataclass(frozen=True)
class FixedThreshold:
    """Explicit physical/ecological threshold (scalar or spatial DataArray/ndarray)."""

    threshold_high: float | np.ndarray | xr.DataArray | None = None
    threshold_low: float | np.ndarray | xr.DataArray | None = None
    climatology: float | np.ndarray | xr.DataArray = 0.0


@dataclass(frozen=True)
class ExternalClimatology:
    """Reuse precomputed seasonal climatology and threshold arrays."""

    seas: xr.DataArray | np.ndarray | str | Path
    thresh_high: xr.DataArray | np.ndarray | str | Path | None = None
    thresh_low: xr.DataArray | np.ndarray | str | Path | None = None

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> "ExternalClimatology":
        """Construct from a Dataset returned by ``compute_climatology``."""
        seas = ds["seas"] if "seas" in ds else ds["clim"]
        th_h = ds["thresh_high"] if "thresh_high" in ds else None
        th_l = ds["thresh_low"] if "thresh_low" in ds else None
        return cls(seas=seas, thresh_high=th_h, thresh_low=th_l)


BaselineSpec = (
    FixedBaseline
    | DetrendedBaseline
    | MovingBaseline
    | FixedThreshold
    | ExternalClimatology
)


def normalize_baseline(baseline: BaselineSpec | tuple[DateLike, DateLike] | None) -> BaselineSpec:
    """Coerce shorthand baseline specifications into a BaselineSpec object."""
    if baseline is None:
        return FixedBaseline()
    if isinstance(
        baseline,
        (FixedBaseline, DetrendedBaseline, MovingBaseline, FixedThreshold, ExternalClimatology),
    ):
        return baseline
    if isinstance(baseline, Sequence) and not isinstance(baseline, (str, bytes)) and len(baseline) == 2:
        return FixedBaseline(start=baseline[0], end=baseline[1])
    raise TypeError(
        "baseline must be a FixedBaseline, DetrendedBaseline, MovingBaseline, "
        "FixedThreshold, ExternalClimatology, (start, end) tuple, or None"
    )


# ============================================================================
# Shared validation & external array helpers for baseline adapters
# ============================================================================


@dataclass(frozen=True)
class RunContext:
    """Validated run-wide parameters passed into the PreparedBaseline seam."""

    opened: OpenedInput
    cal_full: CalendarInfo
    cal_det: CalendarInfo
    det_slice: slice
    tile_dtype: np.dtype
    prof: CompatProfile
    feb29: str
    freq: Literal["daily", "monthly"]
    calendar: str | None
    var: str | None
    time_dim: str | None
    squeeze_singleton: bool | str | Sequence[str]
    ev_mode: Literal["high", "low", "both"]
    event_raw: str
    p_hi: float
    p_lo: float
    w_half: int
    sm_w: int
    min_dur: int
    mgap: int
    max_pad_length: int
    min_valid_fraction: float
    lower_bound: float | None
    bound_policy: str
    eff_return_clim: bool
    eff_return_int: bool
    eff_return_dcat: bool
    eff_return_ecat: bool
    writes_grids: bool


def _validate_percentiles(ev_mode: str, event_raw: str, p_hi: float, p_lo: float) -> None:
    """Verify required upper/lower percentiles are finite and ordered for the requested mode."""
    if ev_mode in ("high", "both") and math.isnan(p_hi):
        raise ValueError(f"event='{event_raw}' requires a non-None pctile_high.")
    if ev_mode in ("low", "both") and math.isnan(p_lo):
        raise ValueError(f"event='{event_raw}' requires a non-None pctile_low.")
    if ev_mode == "both" and p_lo >= p_hi:
        raise ValueError(f"event='both' requires pctile_low < pctile_high (got {p_lo} >= {p_hi}).")


def _open_and_validate_clim_source(
    clim_data: Any,
    clim_var: str | None,
    clim_time: Any,
    ctx: RunContext,
) -> tuple[OpenedInput, CalendarInfo]:
    """Open external climatology input and verify spatial grid and calendar compatibility."""
    ext = open_data_input(
        clim_data,
        var=clim_var or ctx.var,
        time=clim_time,
        time_dim=ctx.time_dim,
        squeeze_singleton=ctx.squeeze_singleton,
    )
    try:
        cal_c = encode_time_axis(ext.da.coords[ext.time_dim], freq=ctx.freq, calendar=ctx.calendar)
        if ext.spatial_dims != ctx.opened.spatial_dims or ext.spatial_shape != ctx.opened.spatial_shape:
            raise ValueError(
                f"External clim_data spatial grid {dict(zip(ext.spatial_dims, ext.spatial_shape))} "
                f"does not match data grid {dict(zip(ctx.opened.spatial_dims, ctx.opened.spatial_shape))}."
            )
        if cal_c.year_len != ctx.cal_full.year_len:
            raise ValueError(
                f"External clim_data calendar '{cal_c.calendar}' (year_len={cal_c.year_len}) "
                f"is incompatible with data calendar '{ctx.cal_full.calendar}' (year_len={ctx.cal_full.year_len})."
            )
        return ext, cal_c
    except BaseException:
        ext.close()
        raise


def _validate_and_broadcast_spatial(
    val: float | np.ndarray | xr.DataArray | None,
    tile: SpatialTile,
    spatial_dims: tuple[str, ...],
    spatial_shape: tuple[int, ...],
    n_det: int,
    default: float = float("nan"),
) -> np.ndarray:
    """Broadcast a scalar or spatial DataArray/ndarray to ``(tile.n_pixels, n_det)`` float64."""
    if val is None:
        return np.full((tile.n_pixels, n_det), default, dtype=np.float64)
    if np.ndim(val) == 0:
        return np.full((tile.n_pixels, n_det), float(val), dtype=np.float64)
    if isinstance(val, xr.DataArray):
        unknown = [d for d in val.dims if d not in spatial_dims]
        if unknown:
            raise ValueError(
                f"FixedThreshold DataArray has unexpected dims {unknown}; expected subset of {spatial_dims}."
            )
        for d in val.dims:
            expected_len = spatial_shape[spatial_dims.index(d)]
            if val.sizes[d] != expected_len:
                raise ValueError(
                    f"FixedThreshold DataArray dim '{d}' has length {val.sizes[d]}, expected {expected_len}."
                )
        sub = val.isel({d: tile.isel_dict[d] for d in spatial_dims if d in val.dims})
        present = [d for d in spatial_dims if d in sub.dims]
        arr = np.asarray(sub.transpose(*present).values, dtype=np.float64)
        shape_with_ones = tuple(tile.tile_shape[i] if d in present else 1 for i, d in enumerate(spatial_dims))
        tile_arr = np.broadcast_to(arr.reshape(shape_with_ones), tile.tile_shape).reshape(tile.n_pixels, 1)
        return np.ascontiguousarray(np.broadcast_to(tile_arr, (tile.n_pixels, n_det)), dtype=np.float64)
    arr = np.asarray(val, dtype=np.float64)
    if not spatial_dims:
        if arr.size == 1:
            return np.full((tile.n_pixels, n_det), float(arr.reshape(())), dtype=np.float64)
        raise ValueError(f"1D FixedThreshold expects scalar value, got shape {arr.shape}.")
    if arr.shape == spatial_shape:
        sub_arr = arr[tuple(tile.isel_dict[d] for d in spatial_dims)].reshape(tile.n_pixels, 1)
        return np.ascontiguousarray(np.broadcast_to(sub_arr, (tile.n_pixels, n_det)), dtype=np.float64)
    raise ValueError(f"FixedThreshold array shape {arr.shape} does not match spatial_shape {spatial_shape}.")


def _open_external_clim_array(
    obj: xr.DataArray | np.ndarray | str | Path | None,
    var_hints: tuple[str, ...],
) -> tuple[xr.DataArray | np.ndarray | None, list[Any]]:
    """Load an external climatology variable from memory or a NetCDF/Zarr path."""
    if obj is None or isinstance(obj, (xr.DataArray, np.ndarray)):
        return obj, []
    p = Path(obj)
    ds = xr.open_zarr(p, consolidated=False) if (p.is_dir() or p.suffix == ".zarr") else xr.open_dataset(p)
    try:
        for hint in var_hints:
            if hint in ds:
                return ds[hint], [ds]
        if len(ds.data_vars) == 1:
            return ds[next(iter(ds.data_vars))], [ds]
        aux = cf_auxiliary_vars(ds)
        primary = [str(v) for v in ds.data_vars if str(v) not in aux]
        if len(primary) == 1:
            return ds[primary[0]], [ds]
        raise KeyError(f"Could not find any of {var_hints} in external climatology file {p}")
    except BaseException:
        try:
            ds.close()
        except Exception:
            pass
        raise


def _extract_external_clim_tile(
    obj: xr.DataArray | np.ndarray | None,
    tile: SpatialTile,
    spatial_dims: tuple[str, ...],
    spatial_shape: tuple[int, ...],
    extra_dims: tuple[str, ...],
    extra_shape: tuple[int, ...],
    *,
    time_dim: str | None = None,
    time_slice: slice | None = None,
    n_full_time: int | None = None,
) -> np.ndarray:
    """Slice and broadcast an external climatology array onto ``(tile.n_pixels, *extra_shape)``."""
    if obj is None:
        return np.full((tile.n_pixels, *extra_shape), np.nan, dtype=np.float64)
    if isinstance(obj, xr.DataArray):
        missing = [d for d in extra_dims if d not in obj.dims]
        if missing:
            raise ValueError(f"ExternalClimatology DataArray missing required dims {missing}; has {obj.dims}.")
        allowed = set(spatial_dims) | set(extra_dims)
        unknown = [d for d in obj.dims if d not in allowed]
        if unknown:
            raise ValueError(
                f"ExternalClimatology DataArray has unexpected dims {unknown}; expected subset of {(*spatial_dims, *extra_dims)}."
            )
        for d in spatial_dims:
            if d in obj.dims:
                expected_len = spatial_shape[spatial_dims.index(d)]
                if obj.sizes[d] != expected_len:
                    raise ValueError(
                        f"ExternalClimatology DataArray dim '{d}' has length {obj.sizes[d]}, expected {expected_len}."
                    )
        sel: dict[str, Any] = {d: tile.isel_dict[d] for d in spatial_dims if d in obj.dims}
        if time_dim is not None and time_slice is not None and time_dim in obj.dims:
            t_len = obj.sizes[time_dim]
            n_det = extra_shape[0]
            if n_full_time is not None and t_len == n_full_time and t_len != n_det:
                sel[time_dim] = time_slice
            elif t_len != n_det:
                raise ValueError(
                    f"ExternalClimatology dim '{time_dim}' has length {t_len}; expected {n_det} (detect_period) or {n_full_time} (full input)."
                )
        sub = obj.isel(sel)
        present_sp = [d for d in spatial_dims if d in sub.dims]
        sub = sub.transpose(*present_sp, *extra_dims)
        actual_extra = tuple(sub.sizes[d] for d in extra_dims)
        if actual_extra != extra_shape:
            raise ValueError(f"ExternalClimatology dims {extra_dims} have shape {actual_extra}, expected {extra_shape}.")
        arr = np.asarray(sub.values, dtype=np.float64)
        shape_with_ones = tuple(tile.tile_shape[i] if d in present_sp else 1 for i, d in enumerate(spatial_dims)) + extra_shape
        tiled = np.broadcast_to(arr.reshape(shape_with_ones), (*tile.tile_shape, *extra_shape))
        return np.ascontiguousarray(tiled.reshape(tile.n_pixels, *extra_shape), dtype=np.float64)

    arr = np.asarray(obj, dtype=np.float64)
    if not spatial_dims:
        if time_slice is not None and n_full_time is not None and arr.shape == (n_full_time,) and arr.shape != extra_shape:
            arr = arr[time_slice]
        if arr.shape != extra_shape:
            raise ValueError(f"1D ExternalClimatology array shape {arr.shape} != expected {extra_shape}.")
        return np.ascontiguousarray(arr.reshape(1, *extra_shape), dtype=np.float64)
    n_sp = len(spatial_dims)
    if arr.ndim < n_sp or arr.shape[:n_sp] != spatial_shape:
        raise ValueError(
            f"ExternalClimatology array spatial shape {arr.shape[:n_sp]} does not match data spatial_shape {spatial_shape}."
        )
    idx_tuple: tuple[Any, ...] = tuple(tile.isel_dict[d] for d in spatial_dims)
    if time_slice is not None and n_full_time is not None and arr.shape[n_sp:] == (n_full_time,) and (n_full_time,) != extra_shape:
        idx_tuple = idx_tuple + (time_slice,)
    else:
        idx_tuple = idx_tuple + (slice(None),) * len(extra_shape)
    sub_arr = arr[idx_tuple]
    if sub_arr.shape != (*tile.tile_shape, *extra_shape):
        raise ValueError(
            f"ExternalClimatology array slice shape {sub_arr.shape} != expected {(*tile.tile_shape, *extra_shape)}."
        )
    return np.ascontiguousarray(sub_arr.reshape(tile.n_pixels, *extra_shape), dtype=np.float64)


# ============================================================================
# PreparedBaseline seam + 5 concrete polymorphic baseline adapters
# ============================================================================


class PreparedBaseline:
    """Deep polymorphic execution seam hiding baseline preparation, tile loading, and Rust dispatch."""

    produces_clim: bool = False
    produces_trend: bool = False
    extra_curves_count: int = 0
    moving_years: np.ndarray | None = None
    n_clim_eff: int = 0

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self._closers: list[Any] = []

    @property
    def want_clim_out(self) -> bool:
        return self.ctx.eff_return_clim and self.produces_clim

    @property
    def want_hi_out(self) -> bool:
        return self.want_clim_out and self.ctx.ev_mode in ("high", "both")

    @property
    def want_lo_out(self) -> bool:
        return self.want_clim_out and self.ctx.ev_mode in ("low", "both")

    @property
    def want_trend_out(self) -> bool:
        return self.ctx.writes_grids and self.produces_trend

    def run_tile(self, tile: SpatialTile) -> tuple[np.ndarray, dict[str, Any] | None]:
        raise NotImplementedError

    def close(self) -> None:
        for obj in self._closers:
            try:
                obj.close()
            except Exception:
                pass
        self._closers.clear()


class _PreparedFixed(PreparedBaseline):
    """Adapter for FixedBaseline (supports optional external clim_data and single-read union slice)."""
    produces_clim = True

    def __init__(self, base: FixedBaseline, ctx: RunContext) -> None:
        super().__init__(ctx)
        _validate_percentiles(ctx.ev_mode, ctx.event_raw, ctx.p_hi, ctx.p_lo)
        self._ext_opened: OpenedInput | None = None
        try:
            if base.clim_data is not None:
                ext, cal_c = _open_and_validate_clim_source(base.clim_data, base.clim_var, base.clim_time, ctx)
                self._ext_opened = ext
                self._closers.append(ext)
                clim_slice, in_period_sub = cal_c.slice_with_window_buffer(base.start, base.end, ctx.w_half)
                c_years = np.ascontiguousarray(cal_c.years[clim_slice], dtype=np.int32)
                c_doys = np.ascontiguousarray(cal_c.doys[clim_slice], dtype=np.uint16)
            else:
                clim_slice, in_period_sub = ctx.cal_full.slice_with_window_buffer(base.start, base.end, ctx.w_half)
                c_years = np.ascontiguousarray(ctx.cal_full.years[clim_slice], dtype=np.int32)
                c_doys = np.ascontiguousarray(ctx.cal_full.doys[clim_slice], dtype=np.uint16)

            self._clim_slice = clim_slice
            self.n_clim_eff = int(c_years.size)

            # Precompute single-read union optimization when climatology and detection share the same source
            self._union_slice: slice | None = None
            if self._ext_opened is None:
                u_start = min(ctx.det_slice.start, clim_slice.start)
                u_stop = max(ctx.det_slice.stop, clim_slice.stop)
                if (u_stop - u_start) <= 1.5 * ((ctx.det_slice.stop - ctx.det_slice.start) + (clim_slice.stop - clim_slice.start)):
                    self._union_slice = slice(u_start, u_stop)

            self._plan = _core.FixedPlan(
                c_years,
                c_doys,
                in_period_sub,
                ctx.cal_full.year_len,
                ctx.p_hi,
                ctx.p_lo,
                ctx.w_half,
                ctx.sm_w,
                float(ctx.min_valid_fraction),
                ctx.prof.quantile_method,
                ctx.feb29,
                ctx.prof.smooth_nan_policy,
                det_doys=ctx.cal_det.doys,
                event_mode=ctx.ev_mode,
                min_duration=ctx.min_dur,
                max_gap=ctx.mgap,
                max_pad_length=ctx.max_pad_length,
                lower_bound=ctx.lower_bound,
                bound_policy=ctx.bound_policy,
                strict_inequality=ctx.prof.strict_inequality,
                var_ddof=ctx.prof.var_ddof,
            )
        except BaseException:
            self.close()
            raise

    def run_tile(self, tile: SpatialTile) -> tuple[np.ndarray, dict[str, Any] | None]:
        ctx = self.ctx
        preloaded_clim: np.ndarray | None = None
        if self._union_slice is not None:
            u0 = self._union_slice.start
            full_tile = load_tile_as_pixels_x_time(ctx.opened.da, tile, ctx.opened.time_dim, self._union_slice, dtype=ctx.tile_dtype)
            det_tile = full_tile[:, ctx.det_slice.start - u0 : ctx.det_slice.stop - u0]
            preloaded_clim = full_tile[:, self._clim_slice.start - u0 : self._clim_slice.stop - u0]
            active_mask = np.any(np.isfinite(det_tile), axis=1)
            if self.want_clim_out:
                active_mask |= np.any(np.isfinite(preloaded_clim), axis=1)
        else:
            det_tile = load_tile_as_pixels_x_time(ctx.opened.da, tile, ctx.opened.time_dim, ctx.det_slice, dtype=ctx.tile_dtype)
            active_mask = np.any(np.isfinite(det_tile), axis=1)
            if self.want_clim_out:
                src = self._ext_opened if self._ext_opened is not None else ctx.opened
                preloaded_clim = load_tile_as_pixels_x_time(src.da, tile, src.time_dim, self._clim_slice, dtype=ctx.tile_dtype)
                active_mask |= np.any(np.isfinite(preloaded_clim), axis=1)

        active_local = np.flatnonzero(active_mask).astype(np.int64)
        if active_local.size == 0:
            return active_local, None

        if preloaded_clim is None:
            src = self._ext_opened if self._ext_opened is not None else ctx.opened
            preloaded_clim = load_tile_as_pixels_x_time(src.da, tile, src.time_dim, self._clim_slice, dtype=ctx.tile_dtype)
        clim_act = np.ascontiguousarray(preloaded_clim[active_local], dtype=ctx.tile_dtype)
        det_act = np.ascontiguousarray(det_tile[active_local], dtype=ctx.tile_dtype)
        res = self._plan.run_tile(
            clim_act,
            det_act,
            active_local,
            tile.n_pixels,
            return_clim=self.want_clim_out,
            return_intensity_ts=ctx.eff_return_int,
            return_daily_category_ts=ctx.eff_return_dcat,
            return_event_category_ts=ctx.eff_return_ecat,
        )
        return active_local, res


class _PreparedDetrended(PreparedBaseline):
    """Adapter for DetrendedBaseline (fused OLS detrend + climatology + detection in Rust)."""
    produces_clim = True
    produces_trend = True

    def __init__(self, base: DetrendedBaseline, ctx: RunContext) -> None:
        super().__init__(ctx)
        _validate_percentiles(ctx.ev_mode, ctx.event_raw, ctx.p_hi, ctx.p_lo)
        f_start, f_end = base.fit_period if base.fit_period is not None else (None, None)
        fit_mask = build_period_mask(ctx.cal_full, f_start, f_end)
        if not np.any(fit_mask):
            raise ValueError("DetrendedBaseline fit_period selected zero time steps.")
        clim_slice, in_period_sub = ctx.cal_full.slice_with_window_buffer(base.start, base.end, ctx.w_half)
        c_years = np.ascontiguousarray(ctx.cal_full.years[clim_slice], dtype=np.int32)
        c_doys = np.ascontiguousarray(ctx.cal_full.doys[clim_slice], dtype=np.uint16)
        self.n_clim_eff = int(ctx.cal_full.years.size)

        self._plan = _core.FixedPlan(
            c_years,
            c_doys,
            in_period_sub,
            ctx.cal_full.year_len,
            ctx.p_hi,
            ctx.p_lo,
            ctx.w_half,
            ctx.sm_w,
            float(ctx.min_valid_fraction),
            ctx.prof.quantile_method,
            ctx.feb29,
            ctx.prof.smooth_nan_policy,
            det_doys=ctx.cal_det.doys,
            event_mode=ctx.ev_mode,
            min_duration=ctx.min_dur,
            max_gap=ctx.mgap,
            max_pad_length=ctx.max_pad_length,
            lower_bound=ctx.lower_bound,
            bound_policy=ctx.bound_policy,
            strict_inequality=ctx.prof.strict_inequality,
            var_ddof=ctx.prof.var_ddof,
            detrend_order=base.method,
            fit_mask=fit_mask,
            clim_slice_start=clim_slice.start,
            clim_slice_stop=clim_slice.stop,
            det_slice_start=ctx.det_slice.start,
            det_slice_stop=ctx.det_slice.stop,
        )

    def run_tile(self, tile: SpatialTile) -> tuple[np.ndarray, dict[str, Any] | None]:
        ctx = self.ctx
        full_tile = load_tile_as_pixels_x_time(ctx.opened.da, tile, ctx.opened.time_dim, dtype=ctx.tile_dtype)
        det_tile = full_tile[:, ctx.det_slice]
        active_mask = np.any(np.isfinite(det_tile), axis=1)
        if self.want_clim_out or self.want_trend_out:
            active_mask |= np.any(np.isfinite(full_tile), axis=1)
        active_local = np.flatnonzero(active_mask).astype(np.int64)
        if active_local.size == 0:
            return active_local, None

        full_act = np.ascontiguousarray(full_tile[active_local], dtype=ctx.tile_dtype)
        res = self._plan.run_detrended_tile(
            full_act,
            active_local,
            tile.n_pixels,
            return_clim=self.want_clim_out,
            return_intensity_ts=ctx.eff_return_int,
            return_daily_category_ts=ctx.eff_return_dcat,
            return_event_category_ts=ctx.eff_return_ecat,
        )
        return active_local, res


class _PreparedMoving(PreparedBaseline):
    """Adapter for MovingBaseline using a cached deduplicated Rust MovingSchedule."""
    produces_clim = True

    def __init__(self, base: MovingBaseline, ctx: RunContext) -> None:
        super().__init__(ctx)
        _validate_percentiles(ctx.ev_mode, ctx.event_raw, ctx.p_hi, ctx.p_lo)
        self._ext_opened: OpenedInput | None = None
        try:
            if base.clim_data is not None:
                ext, cal_c = _open_and_validate_clim_source(base.clim_data, base.clim_var, base.clim_time, ctx)
                self._ext_opened = ext
                self._closers.append(ext)
                c_years, c_doys = cal_c.years, cal_c.doys
            else:
                c_years, c_doys = ctx.cal_full.years, ctx.cal_full.doys

            self.n_clim_eff = int(c_years.size)
            self._plan = _core.MovingPlan(
                c_years,
                c_doys,
                ctx.cal_det.years,
                ctx.cal_full.year_len,
                base.window_years,
                base.center,
                base.boundary,
                ctx.p_hi,
                ctx.p_lo,
                ctx.w_half,
                ctx.sm_w,
                float(ctx.min_valid_fraction),
                ctx.prof.quantile_method,
                ctx.feb29,
                ctx.prof.smooth_nan_policy,
                det_doys=ctx.cal_det.doys,
                event_mode=ctx.ev_mode,
                min_duration=ctx.min_dur,
                max_gap=ctx.mgap,
                max_pad_length=ctx.max_pad_length,
                lower_bound=ctx.lower_bound,
                bound_policy=ctx.bound_policy,
                strict_inequality=ctx.prof.strict_inequality,
                var_ddof=ctx.prof.var_ddof,
            )
            self.moving_years = self._plan.unique_years
        except BaseException:
            self.close()
            raise

    def run_tile(self, tile: SpatialTile) -> tuple[np.ndarray, dict[str, Any] | None]:
        ctx = self.ctx
        preloaded_clim: np.ndarray | None = None
        if self._ext_opened is None:
            full_tile = load_tile_as_pixels_x_time(ctx.opened.da, tile, ctx.opened.time_dim, dtype=ctx.tile_dtype)
            det_tile = full_tile[:, ctx.det_slice]
            preloaded_clim = full_tile
            active_mask = np.any(np.isfinite(det_tile), axis=1)
            if self.want_clim_out:
                active_mask |= np.any(np.isfinite(full_tile), axis=1)
        else:
            det_tile = load_tile_as_pixels_x_time(ctx.opened.da, tile, ctx.opened.time_dim, ctx.det_slice, dtype=ctx.tile_dtype)
            active_mask = np.any(np.isfinite(det_tile), axis=1)
            if self.want_clim_out:
                preloaded_clim = load_tile_as_pixels_x_time(self._ext_opened.da, tile, self._ext_opened.time_dim, dtype=ctx.tile_dtype)
                active_mask |= np.any(np.isfinite(preloaded_clim), axis=1)

        active_local = np.flatnonzero(active_mask).astype(np.int64)
        if active_local.size == 0:
            return active_local, None

        if preloaded_clim is None:
            assert self._ext_opened is not None
            preloaded_clim = load_tile_as_pixels_x_time(self._ext_opened.da, tile, self._ext_opened.time_dim, dtype=ctx.tile_dtype)
        clim_act = np.ascontiguousarray(preloaded_clim[active_local], dtype=ctx.tile_dtype)
        det_act = np.ascontiguousarray(det_tile[active_local], dtype=ctx.tile_dtype)
        res = self._plan.run_tile(
            clim_act,
            det_act,
            active_local,
            tile.n_pixels,
            return_clim=self.want_clim_out,
            return_intensity_ts=ctx.eff_return_int,
            return_daily_category_ts=ctx.eff_return_dcat,
            return_event_category_ts=ctx.eff_return_ecat,
        )
        return active_local, res


def _validate_fixed_threshold_spec(
    val: float | np.ndarray | xr.DataArray | None,
    spatial_dims: tuple[str, ...],
    spatial_shape: tuple[int, ...],
) -> None:
    """Eagerly verify FixedThreshold array dims/shape (runs even on all-NaN grids)."""
    if val is None or np.ndim(val) == 0:
        return
    if isinstance(val, xr.DataArray):
        unknown = [d for d in val.dims if d not in spatial_dims]
        if unknown:
            raise ValueError(
                f"FixedThreshold DataArray has unexpected dims {unknown}; expected subset of {spatial_dims}."
            )
        for d in val.dims:
            expected_len = spatial_shape[spatial_dims.index(d)]
            if val.sizes[d] != expected_len:
                raise ValueError(
                    f"FixedThreshold DataArray dim '{d}' has length {val.sizes[d]}, expected {expected_len}."
                )
        return
    arr = np.asarray(val)
    if not spatial_dims:
        if arr.size != 1:
            raise ValueError(f"1D FixedThreshold expects scalar value, got shape {arr.shape}.")
        return
    if arr.shape != spatial_shape:
        raise ValueError(f"FixedThreshold array shape {arr.shape} does not match spatial_shape {spatial_shape}.")


def _validate_external_clim_spec(
    obj: xr.DataArray | np.ndarray | None,
    spatial_dims: tuple[str, ...],
    spatial_shape: tuple[int, ...],
    extra_dims: tuple[str, ...],
    extra_shape: tuple[int, ...],
    *,
    time_dim: str | None = None,
    n_full_time: int | None = None,
) -> None:
    """Eagerly verify ExternalClimatology array dims/shape against the input grid."""
    if obj is None:
        return
    if isinstance(obj, xr.DataArray):
        missing = [d for d in extra_dims if d not in obj.dims]
        if missing:
            raise ValueError(f"ExternalClimatology DataArray missing required dims {missing}; has {obj.dims}.")
        allowed = set(spatial_dims) | set(extra_dims)
        unknown = [d for d in obj.dims if d not in allowed]
        if unknown:
            raise ValueError(
                f"ExternalClimatology DataArray has unexpected dims {unknown}; expected subset of {(*spatial_dims, *extra_dims)}."
            )
        for d in spatial_dims:
            if d in obj.dims:
                expected_len = spatial_shape[spatial_dims.index(d)]
                if obj.sizes[d] != expected_len:
                    raise ValueError(
                        f"ExternalClimatology DataArray dim '{d}' has length {obj.sizes[d]}, expected {expected_len}."
                    )
        for d, exp_len in zip(extra_dims, extra_shape):
            actual = obj.sizes[d]
            if d == time_dim and n_full_time is not None and actual == n_full_time:
                continue
            if actual != exp_len:
                if d == time_dim:
                    raise ValueError(
                        f"ExternalClimatology dim '{d}' has length {actual}; expected {exp_len} (detect_period) or {n_full_time} (full input)."
                    )
                raise ValueError(f"ExternalClimatology dims {extra_dims} dim '{d}' has length {actual}, expected {exp_len}.")
        return
    arr = np.asarray(obj)
    if not spatial_dims:
        valid_shapes = {extra_shape}
        if time_dim is not None and n_full_time is not None:
            valid_shapes.add((n_full_time,))
        if arr.shape not in valid_shapes:
            raise ValueError(f"1D ExternalClimatology array shape {arr.shape} != expected {extra_shape}.")
        return
    n_sp = len(spatial_dims)
    if arr.ndim < n_sp or arr.shape[:n_sp] != spatial_shape:
        raise ValueError(
            f"ExternalClimatology array spatial shape {arr.shape[:n_sp]} does not match data spatial_shape {spatial_shape}."
        )
    valid_tails = {extra_shape}
    if time_dim is not None and n_full_time is not None:
        valid_tails.add((n_full_time,))
    if arr.shape[n_sp:] not in valid_tails:
        raise ValueError(
            f"ExternalClimatology array trailing shape {arr.shape[n_sp:]} != expected {extra_shape}."
        )


def _index_year_coord(
    arr: xr.DataArray,
    det_years: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Map each detection timestamp's year to its integer index in an external 'year' coord."""
    ext_years = np.asarray(arr.coords["year"].values, dtype=np.int32)
    if ext_years.size == 0:
        raise ValueError("ExternalClimatology 'year' coordinate cannot be empty.")
    yr_to_i = {int(y): i for i, y in enumerate(ext_years)}
    missing_yrs = [int(y) for y in np.unique(det_years) if int(y) not in yr_to_i]
    if missing_yrs:
        raise ValueError(
            f"ExternalClimatology missing detection years {missing_yrs} "
            f"(available years: {ext_years.min()}..{ext_years.max()})."
        )
    yr_idx = np.array([yr_to_i[int(y)] for y in det_years], dtype=np.intp)
    return yr_idx, int(ext_years.size)


class _PreparedFixedThreshold(PreparedBaseline):
    """Adapter for scalar or spatial FixedThreshold via Rust CurvesPlan."""
    extra_curves_count = 3

    def __init__(self, base: FixedThreshold, ctx: RunContext) -> None:
        super().__init__(ctx)
        if ctx.ev_mode in ("high", "both") and base.threshold_high is None:
            raise ValueError(f"FixedThreshold requires threshold_high when event='{ctx.event_raw}'.")
        if ctx.ev_mode in ("low", "both") and base.threshold_low is None:
            raise ValueError(f"FixedThreshold requires threshold_low when event='{ctx.event_raw}'.")
        for spec_val in (base.climatology, base.threshold_high, base.threshold_low):
            _validate_fixed_threshold_spec(spec_val, ctx.opened.spatial_dims, ctx.opened.spatial_shape)
        self._base = base
        self._plan = _core.CurvesPlan(
            ctx.ev_mode,
            ctx.min_dur,
            ctx.mgap,
            ctx.max_pad_length,
            ctx.lower_bound,
            ctx.bound_policy,
            ctx.prof.strict_inequality,
            ctx.prof.var_ddof,
        )

    def run_tile(self, tile: SpatialTile) -> tuple[np.ndarray, dict[str, Any] | None]:
        ctx = self.ctx
        det_tile = load_tile_as_pixels_x_time(ctx.opened.da, tile, ctx.opened.time_dim, ctx.det_slice, dtype=ctx.tile_dtype)
        active_local = np.flatnonzero(np.any(np.isfinite(det_tile), axis=1)).astype(np.int64)
        if active_local.size == 0:
            return active_local, None

        n_det = int(ctx.cal_det.years.size)
        det_act = np.ascontiguousarray(det_tile[active_local], dtype=ctx.tile_dtype)
        seas_ts = _validate_and_broadcast_spatial(
            self._base.climatology, tile, ctx.opened.spatial_dims, ctx.opened.spatial_shape, n_det, default=0.0
        )[active_local]
        hi_ts = _validate_and_broadcast_spatial(
            self._base.threshold_high, tile, ctx.opened.spatial_dims, ctx.opened.spatial_shape, n_det, default=np.nan
        )[active_local]
        lo_ts = _validate_and_broadcast_spatial(
            self._base.threshold_low, tile, ctx.opened.spatial_dims, ctx.opened.spatial_shape, n_det, default=np.nan
        )[active_local]
        res = self._plan.run_curves(
            det_act,
            np.ascontiguousarray(seas_ts, dtype=np.float64),
            np.ascontiguousarray(hi_ts, dtype=np.float64),
            np.ascontiguousarray(lo_ts, dtype=np.float64),
            active_local,
            tile.n_pixels,
            return_intensity_ts=ctx.eff_return_int,
            return_daily_category_ts=ctx.eff_return_dcat,
            return_event_category_ts=ctx.eff_return_ecat,
        )
        return active_local, res


class _PreparedExternalClimatology(PreparedBaseline):
    """Adapter for precomputed ExternalClimatology in ('doy',), ('year', 'doy'), or ('time',) layout."""
    extra_curves_count = 3

    def __init__(self, base: ExternalClimatology, ctx: RunContext) -> None:
        super().__init__(ctx)
        if ctx.ev_mode in ("high", "both") and base.thresh_high is None:
            raise ValueError(f"ExternalClimatology requires thresh_high when event='{ctx.event_raw}'.")
        if ctx.ev_mode in ("low", "both") and base.thresh_low is None:
            raise ValueError(f"ExternalClimatology requires thresh_low when event='{ctx.event_raw}'.")
        try:
            self._seas, c1 = _open_external_clim_array(base.seas, ("seas", "clim"))
            self._closers.extend(c1)
            self._hi, c2 = _open_external_clim_array(base.thresh_high, ("thresh_high",))
            self._closers.extend(c2)
            self._lo, c3 = _open_external_clim_array(base.thresh_low, ("thresh_low",))
            self._closers.extend(c3)

            ref = self._seas if self._seas is not None else (self._hi if self._hi is not None else self._lo)
            self._doy_idx: np.ndarray | None = None
            self._yd_maps: dict[str, tuple[np.ndarray, int]] = {}
            self._ext_time_dim: str = ctx.opened.time_dim
            sp_dims, sp_shape = ctx.opened.spatial_dims, ctx.opened.spatial_shape
            yl = ctx.cal_full.year_len
            n_det = int(ctx.cal_det.years.size)

            if isinstance(ref, xr.DataArray) and "year" in ref.dims and "doy" in ref.dims:
                self._mode = "year_doy"
                self._doy_idx = ctx.cal_det.doys.astype(np.intp) - 1
                ref_map = _index_year_coord(ref, ctx.cal_det.years)
                for key, obj in (("seas", self._seas), ("hi", self._hi), ("lo", self._lo)):
                    if obj is None:
                        continue
                    ymap = _index_year_coord(obj, ctx.cal_det.years) if isinstance(obj, xr.DataArray) and "year" in obj.dims else ref_map
                    self._yd_maps[key] = ymap
                    _validate_external_clim_spec(obj, sp_dims, sp_shape, ("year", "doy"), (ymap[1], yl))
            else:
                ext_t_dim: str | None = None
                if isinstance(ref, xr.DataArray) and "doy" not in ref.dims:
                    if ctx.opened.time_dim in ref.dims:
                        ext_t_dim = ctx.opened.time_dim
                    else:
                        try:
                            cand = resolve_time_dim(ref, "auto")
                        except ValueError:
                            cand = None
                        if cand is not None and cand not in sp_dims:
                            ext_t_dim = str(cand)
                if ext_t_dim is not None:
                    self._mode = "time"
                    self._ext_time_dim = ext_t_dim
                    n_full_t = int(ctx.cal_full.years.size)
                    for obj in (self._seas, self._hi, self._lo):
                        _validate_external_clim_spec(
                            obj, sp_dims, sp_shape, (ext_t_dim,), (n_det,),
                            time_dim=ext_t_dim, n_full_time=n_full_t,
                        )
                else:
                    self._mode = "doy"
                    for obj in (self._seas, self._hi, self._lo):
                        _validate_external_clim_spec(obj, sp_dims, sp_shape, ("doy",), (yl,))

            self._plan = _core.CurvesPlan(
                ctx.ev_mode,
                ctx.min_dur,
                ctx.mgap,
                ctx.max_pad_length,
                ctx.lower_bound,
                ctx.bound_policy,
                ctx.prof.strict_inequality,
                ctx.prof.var_ddof,
                det_doys=ctx.cal_det.doys if self._mode == "doy" else None,
                year_len=yl if self._mode == "doy" else None,
            )
        except BaseException:
            self.close()
            raise

    def run_tile(self, tile: SpatialTile) -> tuple[np.ndarray, dict[str, Any] | None]:
        ctx = self.ctx
        det_tile = load_tile_as_pixels_x_time(ctx.opened.da, tile, ctx.opened.time_dim, ctx.det_slice, dtype=ctx.tile_dtype)
        active_local = np.flatnonzero(np.any(np.isfinite(det_tile), axis=1)).astype(np.int64)
        if active_local.size == 0:
            return active_local, None

        det_act = np.ascontiguousarray(det_tile[active_local], dtype=ctx.tile_dtype)
        n_det = int(ctx.cal_det.years.size)
        yl = ctx.cal_full.year_len
        sp_dims, sp_shape = ctx.opened.spatial_dims, ctx.opened.spatial_shape

        if self._mode == "doy":
            s_t = _extract_external_clim_tile(self._seas, tile, sp_dims, sp_shape, ("doy",), (yl,))[active_local]
            h_t = _extract_external_clim_tile(self._hi, tile, sp_dims, sp_shape, ("doy",), (yl,))[active_local]
            l_t = _extract_external_clim_tile(self._lo, tile, sp_dims, sp_shape, ("doy",), (yl,))[active_local]
            res = self._plan.run_doy_curves(
                det_act,
                np.ascontiguousarray(s_t, dtype=np.float64),
                np.ascontiguousarray(h_t, dtype=np.float64),
                np.ascontiguousarray(l_t, dtype=np.float64),
                active_local,
                tile.n_pixels,
                return_intensity_ts=ctx.eff_return_int,
                return_daily_category_ts=ctx.eff_return_dcat,
                return_event_category_ts=ctx.eff_return_ecat,
            )
            return active_local, res

        if self._mode == "year_doy":
            assert self._doy_idx is not None

            def _map_yd(key: str, obj: Any) -> np.ndarray:
                if obj is None:
                    return np.full((active_local.size, n_det), np.nan, dtype=np.float64)
                yr_idx, ny = self._yd_maps[key]
                raw = _extract_external_clim_tile(obj, tile, sp_dims, sp_shape, ("year", "doy"), (ny, yl))[active_local]
                return np.ascontiguousarray(raw[:, yr_idx, self._doy_idx], dtype=np.float64)

            s_c, h_c, l_c = _map_yd("seas", self._seas), _map_yd("hi", self._hi), _map_yd("lo", self._lo)
        else:
            n_full_t = int(ctx.cal_full.years.size)
            kw = dict(time_dim=self._ext_time_dim, time_slice=ctx.det_slice, n_full_time=n_full_t)
            s_c = np.ascontiguousarray(
                _extract_external_clim_tile(self._seas, tile, sp_dims, sp_shape, (self._ext_time_dim,), (n_det,), **kw)[active_local],
                dtype=np.float64,
            )
            h_c = np.ascontiguousarray(
                _extract_external_clim_tile(self._hi, tile, sp_dims, sp_shape, (self._ext_time_dim,), (n_det,), **kw)[active_local],
                dtype=np.float64,
            )
            l_c = np.ascontiguousarray(
                _extract_external_clim_tile(self._lo, tile, sp_dims, sp_shape, (self._ext_time_dim,), (n_det,), **kw)[active_local],
                dtype=np.float64,
            )

        res = self._plan.run_curves(
            det_act,
            s_c,
            h_c,
            l_c,
            active_local,
            tile.n_pixels,
            return_intensity_ts=ctx.eff_return_int,
            return_daily_category_ts=ctx.eff_return_dcat,
            return_event_category_ts=ctx.eff_return_ecat,
        )
        return active_local, res


_PREPARED_ADAPTERS: dict[type, type[PreparedBaseline]] = {
    FixedBaseline: _PreparedFixed,
    DetrendedBaseline: _PreparedDetrended,
    MovingBaseline: _PreparedMoving,
    FixedThreshold: _PreparedFixedThreshold,
    ExternalClimatology: _PreparedExternalClimatology,
}


def prepare_baseline(base: BaselineSpec, ctx: RunContext) -> PreparedBaseline:
    """Instantiate the concrete PreparedBaseline adapter for ``base``."""
    adapter_cls = _PREPARED_ADAPTERS.get(type(base))
    if adapter_cls is None:
        raise TypeError(f"Unsupported baseline type: {type(base)!r}")
    return adapter_cls(base, ctx)  # type: ignore[call-arg]


# ============================================================================
# Standalone climatology runner sharing the same compiled Rust plans
# ============================================================================


def run_climatology_only(
    base: BaselineSpec,
    opened: OpenedInput,
    *,
    tile_dtype: np.dtype,
    cal: CalendarInfo,
    prof: CompatProfile,
    p_hi: float,
    p_lo: float,
    w_half: int,
    sm_w: int,
    min_valid_fraction: float,
    freq: Literal["daily", "monthly"],
    mem_bytes: int,
) -> xr.Dataset:
    """Compute climatology and upper/lower thresholds using the compiled Rust plans."""
    feb29 = effective_feb29_mode(cal, prof.feb29_mode)
    total_pixels = int(math.prod(opened.spatial_shape)) if opened.spatial_shape else 1
    base_attrs = {"baseline": repr(base), "compat": prof.name, "calendar": cal.calendar, "freq": freq}

    if isinstance(base, MovingBaseline):
        plan = _core.MovingPlan(
            cal.years,
            cal.doys,
            cal.years,
            cal.year_len,
            base.window_years,
            base.center,
            base.boundary,
            p_hi,
            p_lo,
            w_half,
            sm_w,
            float(min_valid_fraction),
            prof.quantile_method,
            feb29,
            prof.smooth_nan_policy,
        )
        uy = plan.unique_years
        max_tile = estimate_max_tile_pixels(
            mem_bytes, cal.years.size, 0,
            return_intensity_ts=False, return_daily_category_ts=False, return_event_category_ts=False,
            moving_n_years=int(uy.size), year_len=cal.year_len, input_itemsize=tile_dtype.itemsize,
        )
        seas_out = np.full((total_pixels, uy.size, cal.year_len), np.nan, dtype=np.float64)
        hi_out = np.full_like(seas_out, np.nan)
        lo_out = np.full_like(seas_out, np.nan)
        for tile in iter_spatial_tiles(opened.spatial_dims, opened.spatial_shape, max_tile):
            tile_vals = load_tile_as_pixels_x_time(opened.da, tile, opened.time_dim, dtype=tile_dtype)
            active_mask = np.any(np.isfinite(tile_vals), axis=1)
            if not np.any(active_mask):
                continue
            _, s_act, h_act, l_act = plan.compute_climatology(np.ascontiguousarray(tile_vals[active_mask], dtype=tile_dtype))
            act_idx = np.flatnonzero(active_mask) + tile.flat_start
            seas_out[act_idx], hi_out[act_idx], lo_out[act_idx] = s_act, h_act, l_act

        dims = (*opened.spatial_dims, "year", "doy")
        shape = (*opened.spatial_shape, int(uy.size), cal.year_len)
        coords: dict[str, Any] = {d: opened.spatial_coords[d] for d in opened.spatial_dims}
        coords["year"] = uy
        coords["doy"] = np.arange(1, cal.year_len + 1, dtype=np.int32)
        return xr.Dataset(
            {"seas": (dims, seas_out.reshape(shape)), "thresh_high": (dims, hi_out.reshape(shape)), "thresh_low": (dims, lo_out.reshape(shape))},
            coords=coords,
            attrs=base_attrs,
        )

    assert isinstance(base, (FixedBaseline, DetrendedBaseline))
    clim_slice, in_period_sub = cal.slice_with_window_buffer(base.start, base.end, w_half)
    c_years = np.ascontiguousarray(cal.years[clim_slice], dtype=np.int32)
    c_doys = np.ascontiguousarray(cal.doys[clim_slice], dtype=np.uint16)

    is_detrended = isinstance(base, DetrendedBaseline)
    fit_mask: np.ndarray | None = None
    if is_detrended:
        f_start, f_end = base.fit_period if base.fit_period is not None else (None, None)
        fit_mask = build_period_mask(cal, f_start, f_end)
        if not np.any(fit_mask):
            raise ValueError("DetrendedBaseline fit_period selected zero time steps.")

    plan = _core.FixedPlan(
        c_years,
        c_doys,
        in_period_sub,
        cal.year_len,
        p_hi,
        p_lo,
        w_half,
        sm_w,
        float(min_valid_fraction),
        prof.quantile_method,
        feb29,
        prof.smooth_nan_policy,
        detrend_order=base.method if is_detrended else "none",
        fit_mask=fit_mask,
        clim_slice_start=clim_slice.start,
        clim_slice_stop=clim_slice.stop,
    )

    max_tile = estimate_max_tile_pixels(
        mem_bytes, cal.years.size if is_detrended else c_years.size, 0,
        return_intensity_ts=False, return_daily_category_ts=False, return_event_category_ts=False,
        year_len=cal.year_len, input_itemsize=tile_dtype.itemsize,
    )
    seas_out = np.full((total_pixels, cal.year_len), np.nan, dtype=np.float64)
    hi_out = np.full_like(seas_out, np.nan)
    lo_out = np.full_like(seas_out, np.nan)
    trend_out = np.full((total_pixels, 3), np.nan, dtype=np.float64) if is_detrended else None

    for tile in iter_spatial_tiles(opened.spatial_dims, opened.spatial_shape, max_tile):
        t_slice = None if is_detrended else clim_slice
        vals = load_tile_as_pixels_x_time(opened.da, tile, opened.time_dim, t_slice, dtype=tile_dtype)
        active_mask = np.any(np.isfinite(vals), axis=1)
        if not np.any(active_mask):
            continue
        act_vals = np.ascontiguousarray(vals[active_mask], dtype=tile_dtype)
        act_idx = np.flatnonzero(active_mask) + tile.flat_start
        if is_detrended:
            s_act, h_act, l_act, coeffs = plan.compute_detrended_climatology(act_vals)
            assert trend_out is not None
            trend_out[act_idx] = coeffs
        else:
            s_act, h_act, l_act = plan.compute_climatology(act_vals)
        seas_out[act_idx], hi_out[act_idx], lo_out[act_idx] = s_act, h_act, l_act

    dims = (*opened.spatial_dims, "doy")
    shape = (*opened.spatial_shape, cal.year_len)
    coords = {d: opened.spatial_coords[d] for d in opened.spatial_dims}
    coords["doy"] = np.arange(1, cal.year_len + 1, dtype=np.int32)
    dvars: dict[str, Any] = {
        "seas": (dims, seas_out.reshape(shape)),
        "thresh_high": (dims, hi_out.reshape(shape)),
        "thresh_low": (dims, lo_out.reshape(shape)),
    }
    if trend_out is not None:
        tr_dims = opened.spatial_dims if opened.spatial_dims else ()
        tr_shape = opened.spatial_shape if opened.spatial_shape else ()
        dvars["trend_intercept"] = (tr_dims, trend_out[:, 0].reshape(tr_shape))
        dvars["trend_slope_per_step"] = (tr_dims, trend_out[:, 1].reshape(tr_shape))
        dvars["trend_quadratic_per_step2"] = (tr_dims, trend_out[:, 2].reshape(tr_shape))
    return xr.Dataset(dvars, coords=coords, attrs=base_attrs)
