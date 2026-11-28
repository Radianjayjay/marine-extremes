"""Optional lightweight stdio MCP server exposing marine-extremes tools to AI clients."""

from __future__ import annotations

from collections import Counter
import datetime as dt
import functools
import math
import threading
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence

import numpy as np
import pandas as pd
import xarray as xr

import marine_extremes as mx
from marine_extremes.api import _normalize_event_mode
from marine_extremes.calendar import _is_cftime_obj, date_to_ymd_key, dates_to_yyyymmdd, encode_time_axis
from marine_extremes.cf import cf_auxiliary_vars, is_kelvin_units, resolve_time_dim, select_dataset_var
from marine_extremes.streaming import _squeeze_singleton_dims, open_path_dataset, resolve_input_paths

try:
    from pydantic import Field as _Field
except ImportError:  # pragma: no cover

    def _Field(*, description: str = "", **kwargs: Any) -> Any:
        return description


# MCP SDK v2 dispatches sync tools on anyio worker threads; serialize file I/O and warnings.catch_warnings.
_MCP_IO_LOCK = threading.Lock()


def _to_jsonable(val: Any) -> Any:
    """Convert NumPy, Pandas, datetime, cftime, bytes, and Path values into strict JSON primitives."""
    if val is None or val is pd.NaT or val is pd.NA:
        return None
    if isinstance(val, (bytes, bytearray, np.bytes_)):
        return bytes(val).decode("utf-8", errors="replace")
    if isinstance(val, (bool, str)):
        return val
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.timedelta64):
        return None if np.isnat(val) else str(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(val) if math.isfinite(float(val)) else None
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, np.datetime64):
        return None if np.isnat(val) else str(np.datetime_as_string(val, unit="D"))
    if isinstance(val, (dt.date, dt.datetime)) or _is_cftime_obj(val):
        return f"{val.year:04d}-{val.month:02d}-{val.day:02d}"
    if isinstance(val, dict):
        return {str(k): _to_jsonable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_jsonable(x) for x in val]
    if isinstance(val, np.ndarray):
        if np.issubdtype(val.dtype, np.datetime64) or np.issubdtype(val.dtype, np.timedelta64):
            return _to_jsonable(val[()]) if val.ndim == 0 else [_to_jsonable(x) for x in val]
        return _to_jsonable(val.tolist())
    return str(val)


def _validate_output_path(out: str, input_paths: Sequence[Path]) -> None:
    """Reject output paths that overlap input datasets or non-Zarr directories."""
    if not isinstance(out, str) or not out.strip():
        raise ValueError("Output path must be a non-empty string.")
    out_p = Path(out)
    out_res = out_p.resolve()
    for in_p in input_paths:
        in_res = in_p.resolve()
        if (
            out_res == in_res
            or out_res in in_res.parents
            or in_res in out_res.parents
            or (out_p.exists() and in_p.exists() and out_p.samefile(in_p))
        ):
            raise ValueError(
                f"Output path '{out}' overlaps input dataset path '{in_p}'; choose a different output path."
            )
    if out_p.exists() and out_p.is_dir():
        if out_p.suffix.lower() != ".zarr":
            raise ValueError(f"Output path '{out}' is an existing directory.")
        if not ((out_p / "zarr.json").exists() or (out_p / ".zgroup").exists()):
            raise ValueError(f"Refusing to overwrite non-Zarr directory '{out}' with Zarr output.")


def inspect_dataset(
    path: Annotated[str, _Field(description="Path, Zarr directory, or glob pattern to inspect.")],
    var: Annotated[str | None, _Field(description="Optional variable name, CF standard_name, or marine alias.")] = None,
    time_dim: Annotated[str | None, _Field(description="Time dimension name or 'auto' for CF inference.")] = "auto",
) -> dict[str, Any]:
    """Inspect NetCDF/Zarr variables, dimensions, time calendar, units, and CF recommendations."""
    with _MCP_IO_LOCK:
        ds, input_paths = open_path_dataset(path)
        try:
            aux = cf_auxiliary_vars(ds)
            data_vars = [
                {
                    "name": str(v),
                    "dims": [str(d) for d in da.dims],
                    "shape": [int(s) for s in da.shape],
                    "dtype": str(da.dtype),
                    "units": _to_jsonable(da.attrs.get("units", "")),
                    "standard_name": _to_jsonable(da.attrs.get("standard_name", "")),
                    "long_name": _to_jsonable(da.attrs.get("long_name", "")),
                    "is_auxiliary": str(v) in aux,
                }
                for v, da in ds.data_vars.items()
            ]

            da_sel: xr.DataArray | None = None
            tips: list[str] = []
            try:
                da_sel = select_dataset_var(ds, var, time_dim=time_dim)
                rec_var: str | None = str(da_sel.name)
            except (ValueError, KeyError) as exc:
                rec_var = None
                tips.append(str(exc))

            target = da_sel if da_sel is not None else ds
            resolved_t: str | None = None
            t_obj: Any = None
            try:
                t_obj = resolve_time_dim(target, time_dim)
                resolved_t = str(t_obj)
            except ValueError:
                pass

            time_range: dict[str, Any] | None = None
            if t_obj is not None and t_obj in target.coords:
                try:
                    cal = encode_time_axis(target.coords[t_obj])
                    time_range = {
                        "start": _to_jsonable(cal.raw_time[0]),
                        "end": _to_jsonable(cal.raw_time[-1]),
                        "n_steps": int(cal.raw_time.size),
                        "calendar": cal.calendar,
                    }
                except Exception as exc:
                    time_range = {"n_steps": int(ds.sizes[t_obj])}
                    tips.append(f"Time coordinate '{resolved_t}' could not be decoded as calendar dates: {exc}")

            singleton_dims = [str(d) for d in target.dims if str(d) != resolved_t and int(ds.sizes[d]) == 1]
            spatial_coords: dict[str, Any] = {}
            curvilinear: list[str] = []
            for d, c in ds.coords.items():
                if str(d) == resolved_t:
                    continue
                if c.dims == (d,):
                    arr = np.asarray(c.values)
                    if arr.dtype.kind in ("i", "u", "f"):
                        fin = arr[np.isfinite(arr)]
                        spatial_coords[str(d)] = {
                            "size": int(arr.size),
                            "min": _to_jsonable(np.min(fin)) if fin.size > 0 else None,
                            "max": _to_jsonable(np.max(fin)) if fin.size > 0 else None,
                        }
                    else:
                        spatial_coords[str(d)] = {"size": int(arr.size), "dtype": str(arr.dtype)}
                elif len(c.dims) >= 2:
                    curvilinear.append(str(d))

            if curvilinear:
                tips.append(f"Multidimensional coordinates {curvilinear} found; detect_extremes `region` requires 1D coords.")
            if da_sel is not None and is_kelvin_units(raw_u := da_sel.attrs.get("units")):
                tips.append(
                    f"Variable '{rec_var}' has units='{_to_jsonable(raw_u)}' (Kelvin); use Kelvin lower_bound (e.g., 271.35) or convert to degC."
                )
            if singleton_dims:
                tips.append(f"Singleton dims {singleton_dims} found; pass squeeze_singleton=True or dim name to detect_extremes.")

            return {
                "path": path,
                "n_files": len(input_paths),
                "dims": {str(k): int(v) for k, v in ds.sizes.items()},
                "data_vars": data_vars,
                "recommended_var": rec_var,
                "time_dim": resolved_t,
                "time_range": time_range,
                "singleton_dims": singleton_dims,
                "spatial_coords": spatial_coords,
                "notes": tips,
            }
        finally:
            ds.close()


def _build_baseline(
    baseline_type: str,
    baseline_start: str | int | None,
    baseline_end: str | int | None,
    moving_window_years: int,
    moving_center: bool,
    moving_boundary: Literal["clamp", "shrink"],
    threshold_high: float | None,
    threshold_low: float | None,
    threshold_climatology: float,
) -> Any:
    bt = baseline_type.strip().lower()
    if bt == "fixed":
        return mx.FixedBaseline(start=baseline_start, end=baseline_end)
    if bt in ("detrended_linear", "detrended_quadratic"):
        method = "linear" if bt == "detrended_linear" else "quadratic"
        return mx.DetrendedBaseline(method=method, start=baseline_start, end=baseline_end)
    if bt == "moving":
        return mx.MovingBaseline(window_years=moving_window_years, center=moving_center, boundary=moving_boundary)
    if bt == "fixed_threshold":
        return mx.FixedThreshold(threshold_high=threshold_high, threshold_low=threshold_low, climatology=threshold_climatology)
    raise ValueError(
        f"Unsupported baseline_type='{baseline_type}'. "
        "Expected 'fixed', 'detrended_linear', 'detrended_quadratic', 'moving', or 'fixed_threshold'."
    )


def _slice_region(da: xr.DataArray, region: dict[str, list[float]], *, time_dim: Any) -> xr.DataArray:
    """Slice ascending or descending monotonic 1D numeric spatial coordinates by [min, max] bounds."""
    for dim, bounds in region.items():
        if dim == time_dim:
            raise ValueError(f"Cannot slice time dimension '{dim}' via `region`; use `detect_start`/`detect_end` instead.")
        if dim not in da.dims:
            raise ValueError(f"Region dimension '{dim}' not in variable dims {da.dims}")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Region bounds for '{dim}' must be [min, max], got {bounds!r}")
        if any(isinstance(b, bool) for b in bounds):
            raise TypeError(f"Region bounds for '{dim}' must be numeric, got {bounds!r}")
        if dim not in da.coords or da.coords[dim].dims != (dim,):
            raise ValueError(f"Region slicing requires a 1D coordinate matching '{dim}'.")
        vals = np.asarray(da.coords[dim].values)
        if vals.dtype.kind not in ("i", "u", "f"):
            raise ValueError(f"Region slicing requires a numeric coordinate for '{dim}', got dtype {vals.dtype}.")
        b0, b1 = float(bounds[0]), float(bounds[1])
        if not (math.isfinite(b0) and math.isfinite(b1)):
            raise ValueError(f"Region bounds for '{dim}' must be finite, got {bounds!r}.")
        vf = vals.astype(np.float64, copy=False)
        if not np.all(np.isfinite(vf)):
            raise ValueError(f"Coordinate '{dim}' contains non-finite values and cannot be sliced via `region`.")
        if vf.size >= 2:
            diffs = np.diff(vf)
            if not (np.all(diffs >= 0) or np.all(diffs <= 0)):
                raise ValueError(f"Coordinate '{dim}' is not monotonic and cannot be sliced via `region`.")
        lo, hi = sorted((b0, b1))
        sl = slice(hi, lo) if (vf.size >= 2 and vf[0] > vf[-1]) else slice(lo, hi)
        da = da.sel({dim: sl})
        if da.sizes[dim] == 0:
            raise ValueError(f"Region {bounds} selected 0 grid points along '{dim}'.")
    return da


def detect_extremes(
    path: Annotated[str, _Field(description="Input NetCDF/Zarr file, directory, or glob pattern.")],
    out: Annotated[str, _Field(description="Single output target (.nc, .mat, .zarr, .npz, .parquet, .feather, .csv). Must not overlap `path`.")],
    var: Annotated[str | None, _Field(description="Variable name, CF standard_name, or marine alias; required if multi-variable.")] = None,
    time_dim: Annotated[str | None, _Field(description="Time dimension name or 'auto'.")] = "auto",
    squeeze_singleton: Annotated[bool | str | list[str], _Field(description="Length-1 spatial dim(s) to squeeze (e.g., True or 'zlev').")] = False,
    event: Annotated[Literal["high", "low", "both", "mhw", "mcs"], _Field(description="Event direction: 'high'/'mhw', 'low'/'mcs', or 'both'.")] = "both",
    baseline_type: Annotated[Literal["fixed", "detrended_linear", "detrended_quadratic", "moving", "fixed_threshold"], _Field(description="Climatology baseline mode.")] = "fixed",
    baseline_start: Annotated[str | int | None, _Field(description="Inclusive baseline start year or ISO date (e.g., '1982-01-01').")] = None,
    baseline_end: Annotated[str | int | None, _Field(description="Inclusive baseline end year or ISO date (e.g., '2011-12-31').")] = None,
    moving_window_years: int = 31,
    moving_center: bool = True,
    moving_boundary: Annotated[Literal["clamp", "shrink"], _Field(description="Edge policy for 'moving' baseline windows.")] = "clamp",
    threshold_high: float | None = None,
    threshold_low: float | None = None,
    threshold_climatology: float = 0.0,
    detect_start: Annotated[str | int | None, _Field(description="Inclusive detection start year or ISO date.")] = None,
    detect_end: Annotated[str | int | None, _Field(description="Inclusive detection end year or ISO date.")] = None,
    pctile_high: float | None = 0.90,
    pctile_low: float | None = 0.10,
    freq: Annotated[Literal["daily", "monthly"], _Field(description="Input temporal frequency.")] = "daily",
    min_duration: int | None = None,
    max_gap: int | None = None,
    max_pad_length: int = 0,
    min_valid_fraction: float = 0.0,
    lower_bound: Annotated[float | None, _Field(description="Physical lower bound (e.g., -1.8 degC or 271.35 K for sea-ice freezing limit).")] = None,
    bound_policy: Annotated[Literal["flag", "mask", "floor"], _Field(description="Action when values fall at/below `lower_bound`.")] = "flag",
    compat: Annotated[Literal["standard", "m_mhw", "m_mhw_python", "oliver", "heatwaveR"], _Field(description="Cross-tool numerical compatibility preset.")] = "standard",
    memory_limit: Annotated[str, _Field(description="Tile memory budget (e.g., '4GB', '512MB').")] = "4GB",
    working_dtype: Annotated[Literal["auto", "float32", "float64"], _Field(description="Working tile precision for detection compute.")] = "auto",
    intensity_dtype: Annotated[Literal["float32", "float64"], _Field(description="Storage dtype for daily intensity_ts output.")] = "float32",
    mat_layout: Annotated[Literal["v73", "detectc"], _Field(description="MATLAB .mat output layout when `out` ends with .mat.")] = "v73",
    region: Annotated[dict[str, list[float]] | None, _Field(description="Optional 1D spatial subset mapping dim -> [min, max], e.g. {'lat': [-35, -25], 'lon': [145, 160]}.")] = None,
    sample_limit: Annotated[int, _Field(description="Maximum sample event rows returned in response (0..50).")] = 10,
) -> dict[str, Any]:
    """Detect MHW/MCS/marine extreme events, write one self-contained file to ``out``, and return a compact summary."""
    if int(sample_limit) < 0:
        raise ValueError(f"sample_limit must be >= 0, got {sample_limit}")
    base = _build_baseline(
        baseline_type, baseline_start, baseline_end, moving_window_years,
        moving_center, moving_boundary, threshold_high, threshold_low, threshold_climatology,
    )
    det_period = (detect_start, detect_end) if (detect_start is not None or detect_end is not None) else None

    with _MCP_IO_LOCK:
        raw_ds: xr.Dataset | None = None
        res: mx.DetectResult | None = None
        try:
            if region is not None:
                raw_ds, input_paths = open_path_dataset(path)
                _validate_output_path(out, input_paths)
                da = select_dataset_var(raw_ds, var, time_dim=time_dim)
                resolved_t = resolve_time_dim(da, time_dim)
                data_input: Any = _slice_region(_squeeze_singleton_dims(da, resolved_t, squeeze_singleton), region, time_dim=resolved_t)
                detect_var, detect_time_dim, detect_squeeze = None, str(resolved_t), False
            else:
                input_paths, _ = resolve_input_paths(path)
                _validate_output_path(out, input_paths)
                data_input, detect_var, detect_time_dim, detect_squeeze = path, var, time_dim, squeeze_singleton

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                res = mx.detect(
                    data_input, var=detect_var, time_dim=detect_time_dim, squeeze_singleton=detect_squeeze,
                    event=event, baseline=base, detect_period=det_period, pctile_high=pctile_high,
                    pctile_low=pctile_low, freq=freq, min_duration=min_duration, max_gap=max_gap,
                    max_pad_length=max_pad_length, min_valid_fraction=min_valid_fraction,
                    lower_bound=lower_bound, bound_policy=bound_policy, compat=compat,
                    working_dtype=working_dtype, intensity_dtype=intensity_dtype,
                    memory_limit=memory_limit, out=out, mat_layout=mat_layout,
                )

            k = min(int(sample_limit), 50, len(res.events) if res.events is not None else 0)
            chunk0 = next(res.events.iter_chunks(chunk_rows=k), None) if (res.events is not None and k > 0) else None
            sample_df = chunk0.slice(0, k).to_pandas() if chunk0 is not None else pd.DataFrame()
            return {
                "out_path": str(res.out_path or out),
                "n_events": len(res.events) if res.events is not None else 0,
                "spatial_dims": list(res.events.spatial_dims) if res.events is not None else [],
                "spatial_shape": list(res.events.spatial_shape) if res.events is not None else [],
                "metadata": _to_jsonable(res.metadata),
                "warnings": [str(w.message) for w in caught if issubclass(w.category, UserWarning)],
                "sample_events": _to_jsonable(sample_df.to_dict(orient="records")),
            }
        finally:
            if res is not None:
                res.close()
            if raw_ds is not None:
                raw_ds.close()


def _filter_events_df(
    df: pd.DataFrame,
    *,
    event_type_norm: str | None,
    min_duration: int | None,
    min_category: int | None,
    start_key: int | None,
    end_key: int | None,
) -> pd.DataFrame:
    if df.empty:
        return df
    mask = np.ones(len(df), dtype=bool)
    if event_type_norm is not None:
        mask &= df["event_type"].astype(str).to_numpy() == event_type_norm
    if min_duration is not None:
        mask &= df["duration"].to_numpy() >= min_duration
    if min_category is not None:
        mask &= df["category_max"].to_numpy() >= min_category
    if start_key is not None or end_key is not None:
        keys = dates_to_yyyymmdd(df["date_start"])
        if start_key is not None:
            mask &= keys >= start_key
        if end_key is not None:
            mask &= keys <= end_key
    return df.loc[mask]


def query_events(
    result_path: Annotated[str, _Field(description="Saved marine-extremes output file (.nc, .mat, .zarr, .npz, .parquet, .feather, .csv).")],
    coords: Annotated[dict[str, Any] | None, _Field(description="Physical coordinates per spatial dim for O(1) CSR lookup (nearest match for numeric coords, exact for string coords; e.g., {'lat': -30.0, 'lon': 150.0}).")] = None,
    indices: Annotated[dict[str, int] | None, _Field(description="Zero-based integer indices per spatial dim for O(1) CSR lookup.")] = None,
    event_type: Annotated[Literal["high", "low", "mhw", "mcs", "both"] | None, _Field(description="Filter by event direction ('high'/'mhw', 'low'/'mcs', or 'both').")] = None,
    min_duration: int | None = None,
    min_category: int | None = None,
    start_date: Annotated[str | int | None, _Field(description="Inclusive lower bound on `date_start` (YYYY, YYYY-MM, or YYYY-MM-DD).")] = None,
    end_date: Annotated[str | int | None, _Field(description="Inclusive upper bound on `date_start` (YYYY, YYYY-MM, or YYYY-MM-DD).")] = None,
    sort_by: Annotated[str | None, _Field(description="Optional event column name to sort results by (e.g., 'duration', 'int_max').")] = None,
    descending: bool = True,
    limit: Annotated[int, _Field(description="Maximum event rows returned (0..500; 0 returns matched count only).")] = 50,
) -> dict[str, Any]:
    """Query events from a saved result by O(1) grid coordinate/index or bounded chunked filtering."""
    if coords is not None and indices is not None:
        raise ValueError("Pass either `coords` or `indices`, not both.")
    if int(limit) < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    lim = min(int(limit), 500)

    et_mode = _normalize_event_mode(str(event_type).strip()) if event_type is not None else None
    filter_kw: dict[str, Any] = {
        "event_type_norm": None if et_mode == "both" else et_mode,
        "min_duration": int(min_duration) if min_duration is not None else None,
        "min_category": int(min_category) if min_category is not None else None,
        "start_key": date_to_ymd_key(start_date, is_end=False) if start_date is not None else None,
        "end_key": date_to_ymd_key(end_date, is_end=True) if end_date is not None else None,
    }

    with _MCP_IO_LOCK:
        res = mx.open(result_path)
        try:
            ev = res.events
            if ev is None:
                raise ValueError(f"Result file '{result_path}' contains no event catalog.")

            if coords is not None or indices is not None:
                if coords is not None and "method" in coords and "method" not in ev.spatial_dims:
                    raise KeyError("Unexpected coordinate key 'method'")
                df = ev.sel(**coords) if coords is not None else ev.isel(**indices)  # type: ignore[arg-type]
                if sort_by is not None and sort_by not in df.columns:
                    raise KeyError(f"Unknown sort_by column '{sort_by}'. Available: {list(df.columns)}")
                df = _filter_events_df(df, **filter_kw)
                total_matched = len(df)
                if sort_by is not None and lim > 0 and not df.empty:
                    df = df.sort_values(sort_by, ascending=not descending, kind="mergesort")
                out_df = df.head(lim)
            else:
                total_matched, held_rows, parts = 0, 0, []
                for chunk in ev.iter_chunks():
                    if sort_by is not None and sort_by not in chunk.schema.names:
                        raise KeyError(f"Unknown sort_by column '{sort_by}'. Available: {list(chunk.schema.names)}")
                    f_df = _filter_events_df(chunk.to_pandas(), **filter_kw)
                    if f_df.empty:
                        continue
                    total_matched += len(f_df)
                    if lim == 0:
                        continue
                    if sort_by is None:
                        if held_rows < lim:
                            take = f_df.iloc[: lim - held_rows]
                            parts.append(take)
                            held_rows += len(take)
                    else:
                        merged = pd.concat([*parts, f_df], ignore_index=True) if parts else f_df
                        parts = [merged.sort_values(sort_by, ascending=not descending, kind="mergesort").head(lim)]
                out_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

            return {
                "result_path": str(result_path),
                "spatial_dims": list(ev.spatial_dims),
                "total_matched": total_matched,
                "returned": len(out_df),
                "truncated": total_matched > len(out_df),
                "events": _to_jsonable(out_df.to_dict(orient="records")),
            }
        finally:
            res.close()


def summarize_result(
    result_path: Annotated[str, _Field(description="Saved marine-extremes output file to summarize.")],
) -> dict[str, Any]:
    """Summarize metadata, spatial event counts, categories, and yearly distribution for a saved result."""
    with _MCP_IO_LOCK:
        res = mx.open(result_path)
        try:
            ev = res.events
            if ev is None:
                raise ValueError(f"Result file '{result_path}' contains no event catalog.")

            cmap = np.asarray(ev.count_map().values)
            by_type: Counter[str] = Counter()
            by_cat: Counter[int] = Counter()
            by_year: Counter[int] = Counter()
            dur_min: int | None = None
            dur_max: int | None = None
            dur_sum, n_total = 0, len(ev)
            max_abs_int: float | None = None

            for chunk in ev.iter_chunks():
                df = chunk.to_pandas()
                if df.empty:
                    continue
                vc = df["event_type"].astype(str).value_counts()
                by_type.update(dict(zip(vc.index.tolist(), vc.to_numpy().tolist())))
                for counter, arr in (
                    (by_cat, df["category_max"].to_numpy()),
                    (by_year, dates_to_yyyymmdd(df["date_start"]) // 10000),
                ):
                    vals, cnts = np.unique(arr, return_counts=True)
                    counter.update(dict(zip(vals.tolist(), cnts.tolist())))

                durs = df["duration"].to_numpy()
                dur_min = int(np.min(durs)) if dur_min is None else min(dur_min, int(np.min(durs)))
                dur_max = int(np.max(durs)) if dur_max is None else max(dur_max, int(np.max(durs)))
                dur_sum += int(np.sum(durs))

                abs_vals = np.abs(df["int_max"].to_numpy(dtype=np.float64))
                fin = abs_vals[np.isfinite(abs_vals)]
                if fin.size > 0:
                    abs_i = float(np.max(fin))
                    max_abs_int = abs_i if max_abs_int is None else max(max_abs_int, abs_i)

            return {
                "result_path": str(result_path),
                "metadata": _to_jsonable(res.metadata),
                "spatial_dims": list(ev.spatial_dims),
                "spatial_shape": list(ev.spatial_shape),
                "n_events": n_total,
                "spatial_counts": {
                    "n_pixels": int(cmap.size),
                    "pixels_with_events": int(np.count_nonzero(cmap)),
                    "max_events_per_pixel": int(np.max(cmap)) if cmap.size else 0,
                    "mean_events_per_pixel": float(np.mean(cmap)) if cmap.size else 0.0,
                },
                "by_event_type": dict(by_type),
                "by_category_max": {str(k): by_cat[k] for k in sorted(by_cat)},
                "by_start_year": {str(k): by_year[k] for k in sorted(by_year)},
                "duration": {"min": dur_min, "mean": (dur_sum / n_total) if n_total > 0 else None, "max": dur_max},
                "max_abs_intensity": _to_jsonable(max_abs_int),
            }
        finally:
            res.close()


def _wrap_tool(fn: Any, err_cls: type[Exception]) -> Any:
    """Wrap a tool function so expected input/I/O failures surface as MCP ToolError."""

    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (LookupError, ValueError, TypeError, OSError, ImportError, NotImplementedError) as exc:
            raise err_cls(str(exc)) from exc

    return _wrapped


def create_mcp_server() -> Any:
    """Create and register the ``marine-extremes`` MCP server (supports MCP SDK v2 and v1)."""
    server_cls: Any = None
    tool_error_cls: type[Exception] | None = None
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError

        server_cls, tool_error_cls = MCPServer, ToolError
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP

            server_cls = FastMCP
            try:
                from mcp.server.fastmcp.exceptions import ToolError as FastToolError

                tool_error_cls = FastToolError
            except ImportError:
                pass
        except ImportError as exc:
            raise ImportError(
                "The optional `mcp` package is required for marine-extremes-mcp. "
                "Install it with: pip install 'marine-extremes[mcp]'"
            ) from exc

    server = server_cls(
        "marine-extremes",
        instructions=(
            "Detect and query marine heatwaves (MHWs), marine cold spells (MCSs), and scalar marine extremes. "
            "Always inspect unknown datasets first; write large gridded detections to `out` instead of returning raw arrays."
        ),
    )
    for fn in (inspect_dataset, detect_extremes, query_events, summarize_result):
        server.tool()(_wrap_tool(fn, tool_error_cls) if tool_error_cls is not None else fn)
    return server


def main() -> None:
    """Entry point for the ``marine-extremes-mcp`` stdio server."""
    create_mcp_server().run()


if __name__ == "__main__":
    main()
