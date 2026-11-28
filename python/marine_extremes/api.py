"""High-level public API for ``marine-extremes``: ``compute_climatology`` and ``detect``."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import xarray as xr

from marine_extremes import _core
from marine_extremes.baselines import (
    BaselineSpec,
    ExternalClimatology,
    FixedBaseline,
    FixedThreshold,
    MovingBaseline,
    RunContext,
    normalize_baseline,
    prepare_baseline,
    run_climatology_only,
)
from marine_extremes.calendar import datetime64_resolution, effective_feb29_mode, encode_time_axis
from marine_extremes.cf import warn_if_units_mismatch
from marine_extremes.compat import CompatProfile, resolve_compat
from marine_extremes.io_formats import (
    GRID_CAPABLE_FORMATS,
    MatLayout,
    OutFormat,
    OutputSpec,
    create_result_sink,
    resolve_out_format,
)
from marine_extremes.result import DetectResult
from marine_extremes.streaming import (
    estimate_max_tile_pixels,
    iter_spatial_tiles,
    open_data_input,
    parse_memory_limit,
    resolve_working_dtype,
)

CATEGORY_MISSING: int = int(_core.CATEGORY_MISSING)


def _resolve_intensity_dtype(intensity_dtype: str) -> np.dtype:
    """Validate and map intensity_ts output dtype ('float64' or 'float32')."""
    if intensity_dtype == "float64":
        return np.dtype(np.float64)
    if intensity_dtype == "float32":
        return np.dtype(np.float32)
    raise ValueError(f"intensity_dtype must be 'float64' or 'float32', got '{intensity_dtype}'")


def _normalize_percentile(p: float | None, name: str) -> float:
    """Accept either [0, 1] (m_mhw style) or (1, 100] (heatwaveR/marineHeatWaves style)."""
    if p is None:
        return float("nan")
    val = float(p)
    if 1.0 < val <= 100.0:
        val /= 100.0
    if not (0.0 <= val <= 1.0):
        raise ValueError(f"{name} must be in [0, 1] or [0, 100], got {p!r}")
    return val


def _normalize_event_mode(event: str) -> Literal["high", "low", "both"]:
    """Map user-facing aliases ('mhw'/'mcs'/'high'/'low'/'both') to canonical modes."""
    e = str(event).lower()
    if e in ("high", "mhw"):
        return "high"
    if e in ("low", "mcs"):
        return "low"
    if e == "both":
        return "both"
    raise ValueError(
        f"Unsupported event='{event}'. Expected 'high', 'low', 'both', 'mhw', or 'mcs'."
    )


def _resolve_freq_defaults(
    freq: Literal["daily", "monthly"],
    window_half_width: int | None,
    smooth_width: int | None,
    min_duration: int | None,
    max_gap: int | None,
) -> tuple[int, int, int, int]:
    """Apply Hobday (2016) daily or Jacox (2020) monthly defaults for unset parameters."""
    if freq == "daily":
        return (
            5 if window_half_width is None else int(window_half_width),
            31 if smooth_width is None else int(smooth_width),
            5 if min_duration is None else int(min_duration),
            2 if max_gap is None else int(max_gap),
        )
    if freq == "monthly":
        return (
            0 if window_half_width is None else int(window_half_width),
            1 if smooth_width is None else int(smooth_width),
            1 if min_duration is None else int(min_duration),
            0 if max_gap is None else int(max_gap),
        )
    raise ValueError(f"freq must be 'daily' or 'monthly', got '{freq}'")


def _validate_common_params(
    *,
    min_valid_fraction: float,
    window_half_width: int,
    smooth_width: int,
    min_duration: int | None = None,
    max_gap: int | None = None,
    max_pad_length: int | None = None,
    lower_bound: float | None = None,
) -> None:
    """Validate scalar numeric bounds eagerly before opening files or compiling plans."""
    mvf = float(min_valid_fraction)
    if not math.isfinite(mvf) or not (0.0 <= mvf <= 1.0):
        raise ValueError(f"min_valid_fraction must be in [0.0, 1.0], got {min_valid_fraction!r}")
    if window_half_width < 0:
        raise ValueError(f"window_half_width must be >= 0, got {window_half_width!r}")
    if smooth_width < 1:
        raise ValueError(f"smooth_width must be >= 1, got {smooth_width!r}")
    if min_duration is not None and min_duration < 1:
        raise ValueError(f"min_duration must be >= 1, got {min_duration!r}")
    if max_gap is not None and max_gap < 0:
        raise ValueError(f"max_gap must be >= 0, got {max_gap!r}")
    if max_pad_length is not None and max_pad_length < 0:
        raise ValueError(f"max_pad_length must be >= 0, got {max_pad_length!r}")
    if lower_bound is not None and not math.isfinite(float(lower_bound)):
        raise ValueError(f"lower_bound must be finite or None, got {lower_bound!r}")


def compute_climatology(
    data: Any,
    *,
    var: str | None = None,
    time: Any | None = None,
    time_dim: str | None = "auto",
    squeeze_singleton: bool | str | Sequence[str] = False,
    baseline: BaselineSpec | tuple[Any, Any] | None = None,
    pctile_high: float | None = 0.90,
    pctile_low: float | None = 0.10,
    freq: Literal["daily", "monthly"] = "daily",
    window_half_width: int | None = None,
    smooth_width: int | None = None,
    min_valid_fraction: float = 0.0,
    compat: str | CompatProfile = "standard",
    calendar: str | None = None,
    working_dtype: Literal["auto", "float32", "float64"] = "auto",
    memory_limit: str | int = "4GB",
) -> xr.Dataset:
    """Compute seasonal climatology and upper/lower thresholds without running event detection."""
    base = normalize_baseline(baseline)
    if isinstance(base, (FixedThreshold, ExternalClimatology)):
        raise TypeError("compute_climatology requires FixedBaseline, DetrendedBaseline, or MovingBaseline.")

    prof = resolve_compat(compat)
    w_half, sm_w, _, _ = _resolve_freq_defaults(freq, window_half_width, smooth_width, None, None)
    _validate_common_params(min_valid_fraction=min_valid_fraction, window_half_width=w_half, smooth_width=sm_w)
    p_hi = _normalize_percentile(pctile_high, "pctile_high")
    p_lo = _normalize_percentile(pctile_low, "pctile_low")
    mem_bytes = parse_memory_limit(memory_limit)

    if isinstance(base, (FixedBaseline, MovingBaseline)) and base.clim_data is not None:
        opened = open_data_input(
            base.clim_data,
            var=base.clim_var or var,
            time=base.clim_time,
            time_dim=time_dim,
            squeeze_singleton=squeeze_singleton,
        )
    else:
        opened = open_data_input(
            data, var=var, time=time, time_dim=time_dim, squeeze_singleton=squeeze_singleton
        )
    try:
        tile_dtype = resolve_working_dtype(opened.da, working_dtype)
        cal = encode_time_axis(opened.da.coords[opened.time_dim], freq=freq, calendar=calendar)
        return run_climatology_only(
            base,
            opened,
            tile_dtype=tile_dtype,
            cal=cal,
            prof=prof,
            p_hi=p_hi,
            p_lo=p_lo,
            w_half=w_half,
            sm_w=sm_w,
            min_valid_fraction=min_valid_fraction,
            freq=freq,
            mem_bytes=mem_bytes,
        )
    finally:
        opened.close()


def detect(
    data: Any,
    *,
    var: str | None = None,
    time: Any | None = None,
    time_dim: str | None = "auto",
    squeeze_singleton: bool | str | Sequence[str] = False,
    event: str = "high",
    baseline: BaselineSpec | tuple[Any, Any] | None = None,
    detect_period: tuple[Any | None, Any | None] | None = None,
    pctile_high: float | None = 0.90,
    pctile_low: float | None = 0.10,
    freq: Literal["daily", "monthly"] = "daily",
    window_half_width: int | None = None,
    smooth_width: int | None = None,
    min_duration: int | None = None,
    max_gap: int | None = None,
    max_pad_length: int = 0,
    min_valid_fraction: float = 0.0,
    lower_bound: float | None = None,
    bound_policy: Literal["flag", "mask", "floor"] = "flag",
    compat: str | CompatProfile = "standard",
    calendar: str | None = None,
    return_clim: bool = True,
    return_intensity_ts: bool = True,
    return_daily_category_ts: bool = True,
    return_event_category_ts: bool = False,
    working_dtype: Literal["auto", "float32", "float64"] = "auto",
    intensity_dtype: Literal["float64", "float32"] = "float64",
    memory_limit: str | int = "4GB",
    out: str | Path | None = None,
    out_format: OutFormat | str | None = None,
    mat_layout: MatLayout | str = "v73",
) -> DetectResult:
    """Detect marine heatwaves, marine cold spells, or general marine extremes across 1D-4D data."""
    ev_mode = _normalize_event_mode(event)
    base = normalize_baseline(baseline)
    prof = resolve_compat(compat)
    out_path = Path(out) if out is not None else None
    resolved_fmt = resolve_out_format(out_path, out_format=out_format, mat_layout=mat_layout) if out_path is not None else None
    writes_grids = (resolved_fmt in GRID_CAPABLE_FORMATS) if resolved_fmt is not None else True

    w_half, sm_w, min_dur, mgap = _resolve_freq_defaults(
        freq, window_half_width, smooth_width, min_duration, max_gap
    )
    _validate_common_params(
        min_valid_fraction=min_valid_fraction,
        window_half_width=w_half,
        smooth_width=sm_w,
        min_duration=min_dur,
        max_gap=mgap,
        max_pad_length=int(max_pad_length),
        lower_bound=lower_bound,
    )
    p_hi = _normalize_percentile(pctile_high, "pctile_high") if ev_mode in ("high", "both") else float("nan")
    p_lo = _normalize_percentile(pctile_low, "pctile_low") if ev_mode in ("low", "both") else float("nan")
    mem_bytes = parse_memory_limit(memory_limit)
    int_out_dtype = _resolve_intensity_dtype(intensity_dtype)

    opened = open_data_input(
        data, var=var, time=time, time_dim=time_dim, squeeze_singleton=squeeze_singleton
    )
    prepared = None
    sink = None
    try:
        warn_if_units_mismatch(opened.da, lower_bound, stacklevel=3)
        tile_dtype = resolve_working_dtype(opened.da, working_dtype)
        cal_full = encode_time_axis(opened.da.coords[opened.time_dim], freq=freq, calendar=calendar)
        det_start, det_end = detect_period if detect_period is not None else (None, None)
        det_slice = cal_full.extract_period_slice(det_start, det_end)
        cal_det = cal_full.slice(det_slice)

        eff_return_clim = return_clim and writes_grids
        eff_return_int = return_intensity_ts and writes_grids
        eff_return_dcat = return_daily_category_ts and writes_grids
        eff_return_ecat = return_event_category_ts and writes_grids

        ctx = RunContext(
            opened=opened,
            cal_full=cal_full,
            cal_det=cal_det,
            det_slice=det_slice,
            tile_dtype=tile_dtype,
            prof=prof,
            feb29=effective_feb29_mode(cal_full, prof.feb29_mode),
            freq=freq,
            calendar=calendar,
            var=var,
            time_dim=time_dim,
            squeeze_singleton=squeeze_singleton,
            ev_mode=ev_mode,
            event_raw=event,
            p_hi=p_hi,
            p_lo=p_lo,
            w_half=w_half,
            sm_w=sm_w,
            min_dur=min_dur,
            mgap=mgap,
            max_pad_length=int(max_pad_length),
            min_valid_fraction=float(min_valid_fraction),
            lower_bound=lower_bound,
            bound_policy=bound_policy,
            eff_return_clim=eff_return_clim,
            eff_return_int=eff_return_int,
            eff_return_dcat=eff_return_dcat,
            eff_return_ecat=eff_return_ecat,
            writes_grids=writes_grids,
        )
        # 1. Compile baseline execution plan once for all spatial tiles.
        prepared = prepare_baseline(base, ctx)

        # 2. Size tiles to stay within memory_limit and open output sink (in-memory or file).
        max_tile = estimate_max_tile_pixels(
            mem_bytes,
            prepared.n_clim_eff,
            int(cal_det.years.size),
            return_intensity_ts=eff_return_int,
            return_daily_category_ts=eff_return_dcat,
            return_event_category_ts=eff_return_ecat,
            moving_n_years=int(prepared.moving_years.size) if (prepared.moving_years is not None and eff_return_clim) else 0,
            year_len=cal_full.year_len,
            input_itemsize=tile_dtype.itemsize,
            intensity_itemsize=int_out_dtype.itemsize,
            extra_curves_count=prepared.extra_curves_count,
        )

        metadata: dict[str, Any] = {
            "variable": opened.da.name or var or "value",
            "units": str(opened.da.attrs.get("units", "")),
            "event": ev_mode,
            "baseline": repr(base),
            "compat": prof.name,
            "calendar": cal_full.calendar,
            "date_resolution": datetime64_resolution(cal_det.raw_time, cal_det.is_cftime),
            "freq": freq,
            "working_dtype": str(tile_dtype),
            "intensity_dtype": str(int_out_dtype),
            "window_half_width": w_half,
            "smooth_width": sm_w,
            "min_duration": min_dur,
            "max_gap": mgap,
            "max_pad_length": int(max_pad_length),
            "min_valid_fraction": float(min_valid_fraction),
            "lower_bound": lower_bound,
            "bound_policy": bound_policy,
        }
        if resolved_fmt is not None:
            metadata["out_format"] = resolved_fmt

        spec = OutputSpec(
            spatial_dims=opened.spatial_dims,
            spatial_shape=opened.spatial_shape,
            spatial_coords=opened.spatial_coords,
            det_time=cal_det.raw_time,
            time_dim=opened.time_dim,
            year_len=cal_full.year_len,
            moving_years=prepared.moving_years,
            write_clim=prepared.want_clim_out,
            write_thresh_high=prepared.want_hi_out,
            write_thresh_low=prepared.want_lo_out,
            write_intensity_ts=eff_return_int,
            intensity_dtype=int_out_dtype,
            write_daily_category_ts=eff_return_dcat,
            write_event_category_ts=eff_return_ecat,
            write_trend=prepared.want_trend_out,
            write_events=True,
            attrs=metadata,
        )
        sink = create_result_sink(out_path, resolved_fmt, spec)

        # 3. Stream contiguous spatial tiles through Rust and assemble results.
        for tile in iter_spatial_tiles(opened.spatial_dims, opened.spatial_shape, max_tile):
            active_local, res_core = prepared.run_tile(tile)
            if res_core is None:
                sink.record_tile(tile, active_local, cal_det)
            else:
                sink.record_tile(
                    tile,
                    active_local,
                    cal_det,
                    raw_events=res_core["events"],
                    clim=res_core.get("clim"),
                    thresh_high=res_core.get("thresh_high"),
                    thresh_low=res_core.get("thresh_low"),
                    intensity_ts=res_core.get("intensity_ts"),
                    daily_category_ts=res_core.get("daily_category_ts"),
                    event_category_ts=res_core.get("event_category_ts"),
                    trend=res_core.get("trend"),
                )

        result = sink.finalize(cal_det)
        sink = None
        return result
    finally:
        if sink is not None:
            sink.close()
        if prepared is not None:
            prepared.close()
        opened.close()
