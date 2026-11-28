"""Optional lightweight stdio MCP server exposing marine-extremes tools to AI clients."""

from __future__ import annotations

import datetime as dt
import functools
import glob as glob_mod
import math
import threading
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence

import numpy as np
import pandas as pd
import xarray as xr

import marine_extremes as mx
from marine_extremes.calendar import _is_cftime_obj, date_to_ymd_key, dates_to_yyyymmdd, encode_time_axis
from marine_extremes.cf import cf_auxiliary_vars, is_kelvin_units, resolve_time_dim, select_dataset_var
from marine_extremes.io_formats import resolve_out_format
from marine_extremes.streaming import _squeeze_singleton_dims

try:
    from pydantic import Field as _Field
except ImportError:  # pragma: no cover

    def _Field(*, description: str = "", **kwargs: Any) -> Any:
        return description


# MCP SDK v2 dispatches sync tools via anyio worker threads; serialize file I/O and warnings capture.
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
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        f = float(val)
        return f if math.isfinite(f) else None
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, np.datetime64):
        return None if np.isnat(val) else str(np.datetime_as_string(val, unit="D"))
    if isinstance(val, pd.Timestamp):
        return None if pd.isna(val) else val.isoformat()[:10]
    if isinstance(val, (dt.date, dt.datetime)):
        return val.isoformat()[:10]
    if _is_cftime_obj(val):
        return f"{val.year:04d}-{val.month:02d}-{val.day:02d}"
    if isinstance(val, dict):
        return {str(k): _to_jsonable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_jsonable(x) for x in val]
    if isinstance(val, np.ndarray):
        return _to_jsonable(val.tolist())
    return str(val)


def _resolve_input_paths(path: str) -> tuple[list[Path], bool]:
    """Resolve an input file, Zarr store, or glob pattern to concrete existing path(s)."""
    p = Path(path)
    if p.exists():
        return [p], False
    if any(ch in path for ch in ("*", "?", "[")):
        matched = sorted(glob_mod.glob(path))
        if not matched:
            raise FileNotFoundError(f"No files matched glob pattern: {path!r}")
        return [Path(m) for m in matched], True
    raise FileNotFoundError(f"Input path does not exist: {p}")


def _open_raw_dataset(path: str) -> tuple[xr.Dataset, list[Path]]:
    """Open a single NetCDF/Zarr path or glob pattern read-only without eagerly loading arrays."""
    paths, is_glob = _resolve_input_paths(path)
    if is_glob:
        mf_kw: dict[str, Any] = {"combine": "by_coords"}
        if all(q.is_dir() or q.suffix == ".zarr" for q in paths):
            mf_kw.update(engine="zarr", consolidated=False)
        return xr.open_mfdataset([str(q) for q in paths], **mf_kw), paths
    p = paths[0]
    if p.is_dir() or p.suffix == ".zarr":
        return xr.open_zarr(p, consolidated=False), paths
    return xr.open_dataset(p), paths


def _validate_output_path(out: str, input_paths: Sequence[Path]) -> None:
    """Reject output paths that overlap input datasets or non-Zarr directories."""
    out_p = Path(out)
    out_res = out_p.resolve()
    for in_p in input_paths:
        in_res = in_p.resolve()
        if out_res == in_res or out_res in in_res.parents or in_res in out_res.parents:
            raise ValueError(
                f"Output path '{out}' overlaps input dataset path '{in_p}'; choose a different output path."
            )
    if out_p.exists() and out_p.is_dir():
        try:
            fmt = resolve_out_format(out_p, None)
        except ValueError as exc:
            raise ValueError(f"Output path '{out}' is an existing directory.") from exc
        if fmt != "zarr":
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
        ds, input_paths = _open_raw_dataset(path)
        n_files = len(input_paths)
        try:
            aux = cf_auxiliary_vars(ds)
            data_vars = [
                {
                    "name": str(vname),
                    "dims": [str(d) for d in da.dims],
                    "shape": [int(s) for s in da.shape],
                    "dtype": str(da.dtype),
                    "units": _to_jsonable(da.attrs.get("units", "")),
                    "standard_name": _to_jsonable(da.attrs.get("standard_name", "")),
                    "long_name": _to_jsonable(da.attrs.get("long_name", "")),
                    "is_auxiliary": str(vname) in aux,
                }
                for vname, da in ds.data_vars.items()
            ]

            da_sel: xr.DataArray | None = None
            var_note: str | None = None
            try:
                da_sel = select_dataset_var(ds, var, time_dim=time_dim)
                rec_var: str | None = str(da_sel.name)
            except (ValueError, KeyError) as exc:
                rec_var = None
                var_note = str(exc)

            target_obj: xr.Dataset | xr.DataArray = da_sel if da_sel is not None else ds
            resolved_t: str | None = None
            t_obj: Any = None
            try:
                t_obj = resolve_time_dim(target_obj, time_dim)
                resolved_t = str(t_obj)
            except ValueError:
                resolved_t = None

            tips: list[str] = []
            if var_note:
                tips.append(var_note)

            time_range: dict[str, Any] | None = None
            if t_obj is not None and t_obj in target_obj.coords:
                try:
                    cal = encode_time_axis(target_obj.coords[t_obj])
                    time_range = {
                        "start": _to_jsonable(cal.raw_time[0]),
                        "end": _to_jsonable(cal.raw_time[-1]),
                        "n_steps": int(cal.raw_time.size),
                        "calendar": cal.calendar,
                    }
                except Exception as exc:
                    time_range = {"n_steps": int(ds.sizes[t_obj])}
                    tips.append(f"Time coordinate '{resolved_t}' could not be decoded as calendar dates: {exc}")

            target_dims = [str(d) for d in target_obj.dims]
            singleton_dims = [d for d in target_dims if d != resolved_t and int(ds.sizes[d]) == 1]

            spatial_coords: dict[str, Any] = {}
            curvilinear_coords: list[str] = []
            for d in ds.coords:
                c = ds.coords[d]
                if str(d) == resolved_t:
                    continue
                if c.dims == (d,):
                    arr = np.asarray(c.values)
                    if np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.bool_):
                        finite = arr[np.isfinite(arr)]
                        spatial_coords[str(d)] = {
                            "size": int(arr.size),
                            "min": _to_jsonable(np.min(finite)) if finite.size > 0 else None,
                            "max": _to_jsonable(np.max(finite)) if finite.size > 0 else None,
                        }
                    else:
                        spatial_coords[str(d)] = {"size": int(arr.size), "dtype": str(arr.dtype)}
                elif len(c.dims) >= 2:
                    curvilinear_coords.append(str(d))

            if curvilinear_coords:
                tips.append(
                    f"Multidimensional coordinates {curvilinear_coords} found; detect_extremes `region` slicing requires 1D coordinates."
                )
            if da_sel is not None:
                raw_units = da_sel.attrs.get("units")
                if is_kelvin_units(raw_units):
                    tips.append(
                        f"Variable '{rec_var}' has units='{_to_jsonable(raw_units)}' (Kelvin); "
                        "use Kelvin lower_bound (e.g., 271.35) or convert to degC."
                    )
            if singleton_dims:
                tips.append(
                    f"Singleton dims {singleton_dims} found; pass squeeze_singleton=True or dim name when calling detect_extremes."
                )

            return {
                "path": path,
                "n_files": n_files,
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
    if bt == "detrended_linear":
        return mx.DetrendedBaseline(method="linear", start=baseline_start, end=baseline_end)
    if bt == "detrended_quadratic":
        return mx.DetrendedBaseline(method="quadratic", start=baseline_start, end=baseline_end)
    if bt == "moving":
        return mx.MovingBaseline(window_years=moving_window_years, center=moving_center, boundary=moving_boundary)
    if bt == "fixed_threshold":
        return mx.FixedThreshold(
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            climatology=threshold_climatology,
        )
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
        if not np.issubdtype(vals.dtype, np.number) or np.issubdtype(vals.dtype, np.bool_):
            raise ValueError(f"Region slicing requires a numeric coordinate for '{dim}', got dtype {vals.dtype}.")
        b0, b1 = float(bounds[0]), float(bounds[1])
        if not (math.isfinite(b0) and math.isfinite(b1)):
            raise ValueError(f"Region bounds for '{dim}' must be finite, got {bounds!r}.")
        vals_f = vals.astype(np.float64, copy=False)
        if not np.all(np.isfinite(vals_f)):
            raise ValueError(f"Coordinate '{dim}' contains non-finite values and cannot be sliced via `region`.")
        if vals_f.size >= 2:
            diffs = np.diff(vals_f)
            if not (np.all(diffs >= 0) or np.all(diffs <= 0)):
                raise ValueError(f"Coordinate '{dim}' is not monotonic and cannot be sliced via `region`.")
        lo, hi = sorted((b0, b1))
        sl = slice(hi, lo) if (vals_f.size >= 2 and vals_f[0] > vals_f[-1]) else slice(lo, hi)
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
    base = _build_baseline(
        baseline_type,
        baseline_start,
        baseline_end,
        moving_window_years,
        moving_center,
        moving_boundary,
        threshold_high,
        threshold_low,
        threshold_climatology,
    )
    input_paths, _ = _resolve_input_paths(path)
    _validate_output_path(out, input_paths)
    det_period = (detect_start, detect_end) if (detect_start is not None or detect_end is not None) else None

    with _MCP_IO_LOCK:
        raw_ds: xr.Dataset | None = None
        res: mx.DetectResult | None = None
        try:
            if region is not None:
                raw_ds, _ = _open_raw_dataset(path)
                da = select_dataset_var(raw_ds, var, time_dim=time_dim)
                resolved_t = resolve_time_dim(da, time_dim)
                da = _squeeze_singleton_dims(da, resolved_t, squeeze_singleton)
                data_input: Any = _slice_region(da, region, time_dim=resolved_t)
                detect_var = None
                detect_time_dim: str | None = str(resolved_t)
                detect_squeeze: bool | str | list[str] = False
            else:
                data_input = path
                detect_var = var
                detect_time_dim = time_dim
                detect_squeeze = squeeze_singleton

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                res = mx.detect(
                    data_input,
                    var=detect_var,
                    time_dim=detect_time_dim,
                    squeeze_singleton=detect_squeeze,
                    event=event,
                    baseline=base,
                    detect_period=det_period,
                    pctile_high=pctile_high,
                    pctile_low=pctile_low,
                    freq=freq,
                    min_duration=min_duration,
                    max_gap=max_gap,
                    max_pad_length=max_pad_length,
                    min_valid_fraction=min_valid_fraction,
                    lower_bound=lower_bound,
                    bound_policy=bound_policy,
                    compat=compat,
                    working_dtype=working_dtype,
                    intensity_dtype=intensity_dtype,
                    memory_limit=memory_limit,
                    out=out,
                    mat_layout=mat_layout,
                )

            k = min(max(int(sample_limit), 0), 50, len(res.events) if res.events is not None else 0)
            sample_df = pd.DataFrame()
            if res.events is not None and k > 0:
                first_chunk = next(res.events.iter_chunks(chunk_rows=k), None)
                if first_chunk is not None:
                    sample_df = first_chunk.slice(0, k).to_pandas()
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


def _normalize_query_event_type(event_type: str | None) -> str | None:
    """Validate and normalize query event_type filter ('both' -> unfiltered None)."""
    if event_type is None:
        return None
    et = str(event_type).strip().lower()
    if et == "both":
        return None
    if et in ("high", "mhw"):
        return "high"
    if et in ("low", "mcs"):
        return "low"
    raise ValueError(
        f"Unsupported event_type='{event_type}'. Expected 'high', 'low', 'mhw', 'mcs', or 'both'."
    )


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
        mask &= df["duration"].to_numpy() >= int(min_duration)
    if min_category is not None:
        mask &= df["category_max"].to_numpy() >= int(min_category)
    if start_key is not None or end_key is not None:
        keys = dates_to_yyyymmdd(df["date_start"])
        if start_key is not None:
            mask &= keys >= start_key
        if end_key is not None:
            mask &= keys <= end_key
    return df.loc[mask]


def query_events(
    result_path: Annotated[str, _Field(description="Saved marine-extremes output file (.nc, .mat, .zarr, .npz, .parquet, .feather, .csv).")],
    coords: Annotated[dict[str, Any] | None, _Field(description="Exact physical coordinates per spatial dim for O(1) CSR lookup (e.g., {'lat': -30.0, 'lon': 150.0}).")] = None,
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
    lim_raw = int(limit)
    if lim_raw < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    lim = min(lim_raw, 500)

    filter_kw: dict[str, Any] = {
        "event_type_norm": _normalize_query_event_type(event_type),
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
                if coords is not None:
                    if not ev.spatial_dims:
                        if coords:
                            raise KeyError(f"1D catalog takes no coordinate selectors, got {list(coords)}")
                    else:
                        missing = [d for d in ev.spatial_dims if d not in coords]
                        extra = [k for k in coords if k not in ev.spatial_dims]
                        if missing or extra:
                            raise KeyError(
                                f"Expected coordinates for exact spatial dims {ev.spatial_dims}; "
                                f"missing={missing}, unexpected={extra}"
                            )
                    df = ev.sel(**coords)
                else:
                    df = ev.isel(**indices)  # type: ignore[arg-type]

                if sort_by is not None and sort_by not in df.columns:
                    raise KeyError(f"Unknown sort_by column '{sort_by}'. Available: {list(df.columns)}")
                df = _filter_events_df(df, **filter_kw)
                total_matched = len(df)
                if lim == 0:
                    out_df = df.iloc[:0]
                else:
                    if sort_by is not None and not df.empty:
                        df = df.sort_values(sort_by, ascending=not descending, kind="mergesort")
                    out_df = df.head(lim)
            else:
                total_matched = 0
                parts: list[pd.DataFrame] = []
                held_rows = 0
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
            spatial_counts = {
                "n_pixels": int(cmap.size),
                "pixels_with_events": int(np.count_nonzero(cmap)),
                "max_events_per_pixel": int(np.max(cmap)) if cmap.size else 0,
                "mean_events_per_pixel": float(np.mean(cmap)) if cmap.size else 0.0,
            }

            by_type: dict[str, int] = {}
            by_cat: dict[str, int] = {}
            by_year: dict[str, int] = {}
            dur_min: int | None = None
            dur_max: int | None = None
            dur_sum = 0
            max_abs_int: float | None = None
            n_total = len(ev)

            for chunk in ev.iter_chunks():
                df = chunk.to_pandas()
                if df.empty:
                    continue
                for k, v in df["event_type"].astype(str).value_counts().items():
                    by_type[str(k)] = by_type.get(str(k), 0) + int(v)
                for k, v in df["category_max"].value_counts().items():
                    by_cat[str(int(k))] = by_cat.get(str(int(k)), 0) + int(v)
                yrs = dates_to_yyyymmdd(df["date_start"]) // 10000
                for k, v in pd.Series(yrs).value_counts().items():
                    by_year[str(int(k))] = by_year.get(str(int(k)), 0) + int(v)

                durs = df["duration"].to_numpy()
                c_min, c_max = int(np.min(durs)), int(np.max(durs))
                dur_min = c_min if dur_min is None else min(dur_min, c_min)
                dur_max = c_max if dur_max is None else max(dur_max, c_max)
                dur_sum += int(np.sum(durs))

                abs_vals = np.abs(df["int_max"].to_numpy(dtype=np.float64))
                finite = abs_vals[np.isfinite(abs_vals)]
                if finite.size > 0:
                    abs_i = float(np.max(finite))
                    max_abs_int = abs_i if max_abs_int is None else max(max_abs_int, abs_i)

            return {
                "result_path": str(result_path),
                "metadata": _to_jsonable(res.metadata),
                "spatial_dims": list(ev.spatial_dims),
                "spatial_shape": list(ev.spatial_shape),
                "n_events": n_total,
                "spatial_counts": spatial_counts,
                "by_event_type": by_type,
                "by_category_max": dict(sorted(by_cat.items())),
                "by_start_year": dict(sorted(by_year.items())),
                "duration": {
                    "min": dur_min,
                    "mean": (dur_sum / n_total) if n_total > 0 else None,
                    "max": dur_max,
                },
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

        server_cls = MCPServer
        tool_error_cls = ToolError
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP

            server_cls = FastMCP
            try:
                from mcp.server.fastmcp.exceptions import ToolError as FastToolError

                tool_error_cls = FastToolError
            except ImportError:
                tool_error_cls = None
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
