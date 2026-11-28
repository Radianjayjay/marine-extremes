"""Optional lightweight stdio MCP server exposing marine-extremes tools to AI clients."""

from __future__ import annotations

import datetime as dt
import glob as glob_mod
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import marine_extremes as mx
from marine_extremes.calendar import _is_cftime_obj, encode_time_axis
from marine_extremes.cf import cf_auxiliary_vars, resolve_time_dim, select_dataset_var


def _to_jsonable(val: Any) -> Any:
    """Convert NumPy, Pandas, datetime, cftime, and Path values into strict JSON primitives."""
    if val is None or isinstance(val, (bool, str)):
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
        return [_to_jsonable(x) for x in val.tolist()]
    if val is pd.NaT:
        return None
    return str(val)


def _open_raw_dataset(path: str) -> tuple[xr.Dataset, int]:
    """Open a single NetCDF/Zarr path or glob pattern read-only without eagerly loading arrays."""
    if any(ch in path for ch in ("*", "?", "[")):
        paths = sorted(glob_mod.glob(path))
        if not paths:
            raise FileNotFoundError(f"No files matched glob pattern: {path!r}")
        mf_kw: dict[str, Any] = {"combine": "by_coords"}
        if all(Path(q).is_dir() or Path(q).suffix == ".zarr" for q in paths):
            mf_kw.update(engine="zarr", consolidated=False)
        return xr.open_mfdataset(paths, **mf_kw), len(paths)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    if p.is_dir() or p.suffix == ".zarr":
        return xr.open_zarr(p, consolidated=False), 1
    return xr.open_dataset(p), 1


def inspect_dataset(
    path: str,
    var: str | None = None,
    time_dim: str | None = "auto",
) -> dict[str, Any]:
    """Inspect NetCDF/Zarr variables, dimensions, time calendar, units, and CF recommendations."""
    ds, n_files = _open_raw_dataset(path)
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
            except Exception:
                time_range = {"n_steps": int(ds.sizes[t_obj])}

        target_dims = [str(d) for d in target_obj.dims]
        singleton_dims = [d for d in target_dims if d != resolved_t and int(ds.sizes[d]) == 1]

        spatial_coords: dict[str, Any] = {}
        for d in ds.coords:
            c = ds.coords[d]
            if str(d) != resolved_t and c.dims == (d,):
                arr = np.asarray(c.values)
                if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
                    spatial_coords[str(d)] = {
                        "size": int(arr.size),
                        "min": _to_jsonable(np.nanmin(arr)),
                        "max": _to_jsonable(np.nanmax(arr)),
                    }
                else:
                    spatial_coords[str(d)] = {"size": int(arr.size), "dtype": str(arr.dtype)}

        tips: list[str] = []
        if var_note:
            tips.append(var_note)
        if da_sel is not None:
            u = str(da_sel.attrs.get("units", "")).strip()
            if u == "K" or "kelvin" in u.lower():
                tips.append(f"Variable '{rec_var}' has units='{u}' (Kelvin); use Kelvin lower_bound (e.g., 271.35) or convert to degC.")
        if singleton_dims:
            tips.append(f"Singleton dims {singleton_dims} found; pass squeeze_singleton=True or dim name when calling detect_extremes.")

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
    moving_boundary: str,
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
        return mx.MovingBaseline(window_years=moving_window_years, center=moving_center, boundary=moving_boundary)  # type: ignore[arg-type]
    if bt == "fixed_threshold":
        return mx.FixedThreshold(threshold_high=threshold_high, threshold_low=threshold_low, climatology=threshold_climatology)
    raise ValueError(
        f"Unsupported baseline_type='{baseline_type}'. "
        "Expected 'fixed', 'detrended_linear', 'detrended_quadratic', 'moving', or 'fixed_threshold'."
    )


def _slice_region(da: xr.DataArray, region: dict[str, list[float]]) -> xr.DataArray:
    """Slice ascending or descending 1D spatial coordinates cleanly by [min, max] bounds."""
    for dim, bounds in region.items():
        if dim not in da.dims:
            raise ValueError(f"Region dimension '{dim}' not in variable dims {da.dims}")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Region bounds for '{dim}' must be [min, max], got {bounds!r}")
        if dim not in da.coords or da.coords[dim].dims != (dim,):
            raise ValueError(f"Region slicing requires a 1D coordinate matching '{dim}'.")
        vals = np.asarray(da.coords[dim].values)
        lo, hi = sorted((float(bounds[0]), float(bounds[1])))
        sl = slice(hi, lo) if (vals.size >= 2 and vals[0] > vals[-1]) else slice(lo, hi)
        da = da.sel({dim: sl})
        if da.sizes[dim] == 0:
            raise ValueError(f"Region {bounds} selected 0 grid points along '{dim}'.")
    return da


def detect_extremes(
    path: str,
    out: str,
    var: str | None = None,
    time_dim: str | None = "auto",
    squeeze_singleton: bool | str | list[str] = False,
    event: str = "both",
    baseline_type: str = "fixed",
    baseline_start: str | int | None = None,
    baseline_end: str | int | None = None,
    moving_window_years: int = 31,
    moving_center: bool = True,
    moving_boundary: str = "clamp",
    threshold_high: float | None = None,
    threshold_low: float | None = None,
    threshold_climatology: float = 0.0,
    detect_start: str | int | None = None,
    detect_end: str | int | None = None,
    pctile_high: float | None = 0.90,
    pctile_low: float | None = 0.10,
    freq: str = "daily",
    min_duration: int | None = None,
    max_gap: int | None = None,
    max_pad_length: int = 0,
    min_valid_fraction: float = 0.0,
    lower_bound: float | None = None,
    bound_policy: str = "flag",
    compat: str = "standard",
    memory_limit: str = "4GB",
    working_dtype: str = "auto",
    intensity_dtype: str = "float32",
    mat_layout: str = "v73",
    region: dict[str, list[float]] | None = None,
    sample_limit: int = 10,
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
    det_period = (detect_start, detect_end) if (detect_start is not None or detect_end is not None) else None
    raw_ds: xr.Dataset | None = None
    res: mx.DetectResult | None = None
    try:
        if region is not None:
            raw_ds, _ = _open_raw_dataset(path)
            da = select_dataset_var(raw_ds, var, time_dim=time_dim)
            data_input: Any = _slice_region(da, region)
            detect_var = None
        else:
            data_input = path
            detect_var = var

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            res = mx.detect(
                data_input,
                var=detect_var,
                time_dim=time_dim,
                squeeze_singleton=squeeze_singleton,
                event=event,
                baseline=base,
                detect_period=det_period,
                pctile_high=pctile_high,
                pctile_low=pctile_low,
                freq=freq,  # type: ignore[arg-type]
                min_duration=min_duration,
                max_gap=max_gap,
                max_pad_length=max_pad_length,
                min_valid_fraction=min_valid_fraction,
                lower_bound=lower_bound,
                bound_policy=bound_policy,  # type: ignore[arg-type]
                compat=compat,
                working_dtype=working_dtype,  # type: ignore[arg-type]
                intensity_dtype=intensity_dtype,  # type: ignore[arg-type]
                memory_limit=memory_limit,
                out=out,
                mat_layout=mat_layout,
            )

        k = min(max(int(sample_limit), 0), 50, len(res.events) if res.events is not None else 0)
        sample_df = res.events._slice_rows(0, k) if (res.events is not None and k > 0) else pd.DataFrame()
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
    event_type: str | None,
    min_duration: int | None,
    min_category: int | None,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if df.empty:
        return df
    mask = np.ones(len(df), dtype=bool)
    if event_type is not None:
        et = {"mhw": "high", "mcs": "low"}.get(event_type.lower(), event_type.lower())
        mask &= df["event_type"].astype(str).to_numpy() == et
    if min_duration is not None:
        mask &= df["duration"].to_numpy() >= int(min_duration)
    if min_category is not None:
        mask &= df["category_max"].to_numpy() >= int(min_category)
    if start_date is not None or end_date is not None:
        ds_str = pd.Series(df["date_start"]).astype(str).str.slice(0, 10)
        if start_date is not None:
            mask &= (ds_str >= str(start_date)[:10]).to_numpy()
        if end_date is not None:
            mask &= (ds_str <= str(end_date)[:10]).to_numpy()
    return df.loc[mask]


def query_events(
    result_path: str,
    coords: dict[str, Any] | None = None,
    indices: dict[str, int] | None = None,
    event_type: str | None = None,
    min_duration: int | None = None,
    min_category: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    sort_by: str | None = None,
    descending: bool = True,
    limit: int = 50,
) -> dict[str, Any]:
    """Query events from a saved result by O(1) grid coordinate/index or bounded chunked filtering."""
    if coords is not None and indices is not None:
        raise ValueError("Pass either `coords` or `indices`, not both.")
    lim = min(max(int(limit), 1), 500)
    filter_kw = dict(
        event_type=event_type,
        min_duration=min_duration,
        min_category=min_category,
        start_date=start_date,
        end_date=end_date,
    )
    res = mx.open(result_path)
    try:
        ev = res.events
        if ev is None:
            raise ValueError(f"Result file '{result_path}' contains no event catalog.")

        if coords is not None or indices is not None:
            df = ev.sel(**coords) if coords is not None else ev.isel(**indices)  # type: ignore[arg-type]
            df = _filter_events_df(df, **filter_kw)
            total_matched = len(df)
            if sort_by is not None and not df.empty:
                if sort_by not in df.columns:
                    raise KeyError(f"Unknown sort_by column '{sort_by}'. Available: {list(df.columns)}")
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


def summarize_result(result_path: str) -> dict[str, Any]:
    """Summarize metadata, spatial event counts, categories, and yearly distribution for a saved result."""
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
            yrs = pd.Series(df["date_start"]).astype(str).str.slice(0, 4)
            for k, v in yrs.value_counts().items():
                by_year[str(k)] = by_year.get(str(k), 0) + int(v)

            durs = df["duration"].to_numpy()
            c_min, c_max = int(np.min(durs)), int(np.max(durs))
            dur_min = c_min if dur_min is None else min(dur_min, c_min)
            dur_max = c_max if dur_max is None else max(dur_max, c_max)
            dur_sum += int(np.sum(durs))

            abs_i = float(np.nanmax(np.abs(df["int_max"].to_numpy(dtype=np.float64))))
            if math.isfinite(abs_i):
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


def create_mcp_server() -> Any:
    """Create and register the ``marine-extremes`` MCP server (supports MCP SDK v2 and v1)."""
    server_cls: Any = None
    try:
        from mcp.server.mcpserver import MCPServer

        server_cls = MCPServer
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP

            server_cls = FastMCP
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
        server.tool()(fn)
    return server


def main() -> None:
    """Entry point for the ``marine-extremes-mcp`` stdio server."""
    create_mcp_server().run()


if __name__ == "__main__":
    main()
