"""Deep single-file persistence module: OutputSpec, ResultSink seam, and per-format adapters."""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import struct
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.feather as pa_feather
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq
import xarray as xr

from marine_extremes import _core
from marine_extremes.calendar import (
    CF_TIME_UNITS,
    CalendarInfo,
    cf_days_to_time,
    coerce_time_for_zarr,
    dates_to_yyyymmdd,
    parse_csv_date_series,
    time_to_cf_days,
    yyyymmdd_to_dates,
)
from marine_extremes.catalog import (
    CSV_STANDARD_DTYPES,
    DATE_EVENT_COLS,
    DiskEventStore,
    EventCatalog,
    MemoryEventStore,
    build_arrow_table_from_core,
    decode_event_column,
    empty_event_table,
    encode_event_type_codes,
    event_type_to_arrow_dict,
    is_str_dtype,
    to_str_list,
)
from marine_extremes.result import DetectResult
from marine_extremes.streaming import SpatialTile

CATEGORY_MISSING: int = int(_core.CATEGORY_MISSING)

OutFormat = Literal["netcdf", "mat", "mat_detectc", "zarr", "npz", "parquet", "feather", "csv"]
MatLayout = Literal["v73", "detectc"]

GRID_CAPABLE_FORMATS: frozenset[str] = frozenset({"netcdf", "mat", "mat_detectc", "zarr", "npz"})
TABULAR_ONLY_FORMATS: frozenset[str] = frozenset({"parquet", "feather", "csv"})

TREND_VAR_KEYS: tuple[tuple[str, str], ...] = (
    ("intercept", "trend_intercept"),
    ("slope_per_step", "trend_slope_per_step"),
    ("quadratic_per_step2", "trend_quadratic_per_step2"),
)
CLIM_VAR_NAMES: tuple[str, ...] = ("clim", "thresh_high", "thresh_low")
TS_VAR_NAMES: tuple[str, ...] = (
    "intensity_ts",
    "intensity_high_ts",
    "intensity_low_ts",
    "daily_category_ts",
    "event_category_ts",
)
TREND_VAR_NAMES: tuple[str, ...] = tuple(full for _, full in TREND_VAR_KEYS)
ALL_GRID_VAR_NAMES: frozenset[str] = frozenset((*CLIM_VAR_NAMES, *TS_VAR_NAMES, *TREND_VAR_NAMES))

_GRID_VAR_META: dict[str, tuple[str, str, bool]] = {
    "clim": (
        "Seasonal climatology",
        "Baseline seasonal climatology computed per calendar day-of-year or month.",
        True,
    ),
    "thresh_high": (
        "Upper extreme threshold",
        "Upper percentile or fixed threshold defining high extreme events (e.g., MHWs).",
        True,
    ),
    "thresh_low": (
        "Lower extreme threshold",
        "Lower percentile or fixed threshold defining low extreme events (e.g., MCSs).",
        True,
    ),
    "intensity_ts": (
        "Combined extreme event intensity anomaly (value - clim)",
        "Hobday-convention intensity anomaly (value - clim) inside active high or low events; 0.0 on non-event active steps, NaN on land/missing (active exceedance overrides opposite-tail gaps; high wins on gap ties).",
        True,
    ),
    "intensity_high_ts": (
        "High extreme event intensity anomaly (value - clim)",
        "Hobday-convention intensity anomaly (value - clim) inside active upper-tail events (e.g., MHWs); 0.0 on non-event active steps, NaN on land/missing.",
        True,
    ),
    "intensity_low_ts": (
        "Low extreme event intensity anomaly (value - clim)",
        "Hobday-convention intensity anomaly (value - clim) inside active lower-tail events (e.g., MCSs, negative values); 0.0 on non-event active steps, NaN on land/missing/masked.",
        True,
    ),
    "daily_category_ts": (
        "Daily Hobday event severity category",
        "+1..+4 high Cat I-IV, -1..-4 low Cat I-IV, 0 normal, -128 missing",
        False,
    ),
    "event_category_ts": (
        "Event peak severity category",
        "Event peak category (+1..+4 high, -1..-4 low) mapped across event span; 0 normal, -128 missing",
        False,
    ),
    "trend_intercept": (
        "Detrended baseline OLS intercept",
        "Ordinary least-squares intercept fitted per pixel before climatology computation.",
        True,
    ),
    "trend_slope_per_step": (
        "Detrended baseline linear slope per step",
        "Ordinary least-squares linear trend per time step.",
        True,
    ),
    "trend_quadratic_per_step2": (
        "Detrended baseline quadratic coefficient per step squared",
        "Ordinary least-squares quadratic trend coefficient per squared step.",
        True,
    ),
}


def grid_var_attrs(name: str, units: str = "") -> dict[str, Any]:
    """Build CF-friendly long_name, description, optional units, and category sentinel attributes."""
    meta = _GRID_VAR_META.get(name)
    if meta is None:
        return {}
    long_name, desc, has_units = meta
    attrs: dict[str, Any] = {"long_name": long_name, "description": desc}
    clean_u = units.strip()
    if has_units and clean_u:
        if name == "trend_slope_per_step":
            attrs["units"] = f"{clean_u} step-1"
        elif name == "trend_quadratic_per_step2":
            attrs["units"] = f"{clean_u} step-2"
        else:
            attrs["units"] = clean_u
    if name in ("daily_category_ts", "event_category_ts"):
        attrs["missing_value_code"] = CATEGORY_MISSING
    return attrs

_EXT_TO_FORMAT: dict[str, OutFormat] = {
    ".nc": "netcdf",
    ".nc4": "netcdf",
    ".cdf": "netcdf",
    ".mat": "mat",
    ".zarr": "zarr",
    ".npz": "npz",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".feather": "feather",
    ".ipc": "feather",
    ".csv": "csv",
}

_NP_TO_MATLAB_CLASS: dict[np.dtype, bytes] = {
    np.dtype(np.float64): b"double",
    np.dtype(np.float32): b"single",
    np.dtype(np.int8): b"int8",
    np.dtype(np.uint8): b"uint8",
    np.dtype(np.int16): b"int16",
    np.dtype(np.uint16): b"uint16",
    np.dtype(np.int32): b"int32",
    np.dtype(np.uint32): b"uint32",
    np.dtype(np.int64): b"int64",
    np.dtype(np.uint64): b"uint64",
    np.dtype(np.bool_): b"logical",
}
_MATLAB_CLASS_TO_NP: dict[bytes, np.dtype] = {v: k for k, v in _NP_TO_MATLAB_CLASS.items()}


# ============================================================================
# Canonical OutputSpec & shared staging / codec helpers
# ============================================================================


@dataclass(frozen=True)
class OutputSpec:
    """Compact specification replacing wide 19-parameter persistence constructors."""

    spatial_dims: tuple[str, ...]
    spatial_shape: tuple[int, ...]
    spatial_coords: dict[str, np.ndarray]
    det_time: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[ns]"))
    time_dim: str = "time"
    year_len: int = 366
    moving_years: np.ndarray | None = None
    write_clim: bool = False
    write_thresh_high: bool = False
    write_thresh_low: bool = False
    clim_dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float64))
    write_intensity_ts: bool = False
    write_intensity_high_ts: bool = False
    write_intensity_low_ts: bool = False
    intensity_dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float64))
    write_daily_category_ts: bool = False
    write_event_category_ts: bool = False
    write_trend: bool = False
    write_events: bool = True
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def total_pixels(self) -> int:
        return int(math.prod(self.spatial_shape)) if self.spatial_shape else 1

    @property
    def has_clim(self) -> bool:
        return self.write_clim or self.write_thresh_high or self.write_thresh_low

    @property
    def has_ts(self) -> bool:
        return (
            self.write_intensity_ts
            or self.write_intensity_high_ts
            or self.write_intensity_low_ts
            or self.write_daily_category_ts
            or self.write_event_category_ts
        ) and self.det_time.size > 0

    @property
    def has_grids(self) -> bool:
        return self.has_clim or self.has_ts or self.write_trend

    @property
    def clim_extra_ndims(self) -> int:
        return 2 if self.moving_years is not None else 1

    @property
    def clim_dims(self) -> tuple[str, ...]:
        return (*self.spatial_dims, "year", "doy") if self.moving_years is not None else (*self.spatial_dims, "doy")

    @property
    def clim_shape(self) -> tuple[int, ...]:
        if self.moving_years is not None:
            return (*self.spatial_shape, int(self.moving_years.shape[0]), int(self.year_len))
        return (*self.spatial_shape, int(self.year_len))

    @property
    def ts_dims(self) -> tuple[str, ...]:
        return (*self.spatial_dims, self.time_dim)

    @property
    def ts_shape(self) -> tuple[int, ...]:
        return (*self.spatial_shape, int(self.det_time.shape[0]))

    def grid_var_specs(self) -> list[tuple[str, tuple[str, ...], tuple[int, ...], np.dtype, Any]]:
        """Canonical ordered list of (name, dims, shape, dtype, fill_value) to persist."""
        out: list[tuple[str, tuple[str, ...], tuple[int, ...], np.dtype, Any]] = []
        if self.has_clim:
            for flag, name in (
                (self.write_clim, "clim"),
                (self.write_thresh_high, "thresh_high"),
                (self.write_thresh_low, "thresh_low"),
            ):
                if flag:
                    out.append((name, self.clim_dims, self.clim_shape, np.dtype(self.clim_dtype), np.nan))
        if self.has_ts:
            for flag, name in (
                (self.write_intensity_ts, "intensity_ts"),
                (self.write_intensity_high_ts, "intensity_high_ts"),
                (self.write_intensity_low_ts, "intensity_low_ts"),
            ):
                if flag:
                    out.append((name, self.ts_dims, self.ts_shape, np.dtype(self.intensity_dtype), np.nan))
            if self.write_daily_category_ts:
                out.append(("daily_category_ts", self.ts_dims, self.ts_shape, np.dtype(np.int8), np.int8(CATEGORY_MISSING)))
            if self.write_event_category_ts:
                out.append(("event_category_ts", self.ts_dims, self.ts_shape, np.dtype(np.int8), np.int8(CATEGORY_MISSING)))
        if self.write_trend:
            for name in TREND_VAR_NAMES:
                out.append((name, self.spatial_dims, self.spatial_shape, np.dtype(np.float64), np.nan))
        return out

    @classmethod
    def for_events_only(
        cls,
        spatial_dims: tuple[str, ...],
        spatial_shape: tuple[int, ...],
        spatial_coords: dict[str, np.ndarray],
        *,
        attrs: dict[str, Any] | None = None,
    ) -> "OutputSpec":
        return cls(
            spatial_dims=spatial_dims,
            spatial_shape=spatial_shape,
            spatial_coords=spatial_coords,
            write_events=True,
            attrs=dict(attrs or {}),
        )

    @classmethod
    def from_result(
        cls,
        result: DetectResult,
        fmt: OutFormat,
    ) -> tuple["OutputSpec", list[tuple[str, xr.DataArray, tuple[str, ...]]]]:
        """Extract an OutputSpec and ordered (name, DataArray, target_dims) sources from a DetectResult."""
        time_dim = "time"
        det_time = np.array([], dtype="datetime64[ns]")
        for ts_da in (
            result.intensity_ts,
            result.intensity_high_ts,
            result.intensity_low_ts,
            result.daily_category_ts,
            result.event_category_ts,
        ):
            if ts_da is not None:
                if result.events is not None:
                    extra = [d for d in ts_da.dims if d not in result.events.spatial_dims]
                    time_dim = extra[-1] if extra else ts_da.dims[-1]
                else:
                    time_dim = ts_da.dims[-1]
                det_time = np.asarray(ts_da.coords[time_dim].values)
                break

        ref_da = next(
            (
                da
                for da in (
                    result.intensity_ts,
                    result.intensity_high_ts,
                    result.intensity_low_ts,
                    result.daily_category_ts,
                    result.event_category_ts,
                    result.clim,
                    result.thresh_high,
                    result.thresh_low,
                )
                if da is not None
            ),
            None,
        )
        if result.events is not None:
            spatial_dims = result.events.spatial_dims
            spatial_shape = result.events.spatial_shape
            spatial_coords = result.events.spatial_coords
        elif ref_da is not None:
            non_spatial = {"doy", "year", "time", "pixel", time_dim}
            spatial_dims = tuple(d for d in ref_da.dims if d not in non_spatial)
            spatial_shape = tuple(int(ref_da.sizes[d]) for d in spatial_dims)
            spatial_coords = {d: np.asarray(ref_da.coords[d].values) for d in spatial_dims}
        else:
            spatial_dims, spatial_shape, spatial_coords = (), (), {}

        clim_ref = result.clim if result.clim is not None else (result.thresh_high if result.thresh_high is not None else result.thresh_low)
        year_len = int(clim_ref.sizes["doy"]) if (clim_ref is not None and "doy" in clim_ref.dims) else 366
        moving_years = (
            np.asarray(clim_ref.coords["year"].values, dtype=np.int32)
            if (clim_ref is not None and "year" in clim_ref.dims)
            else None
        )
        clim_dtype = np.dtype(clim_ref.dtype) if clim_ref is not None else np.dtype(np.float64)
        int_ref = next((da for da in (result.intensity_ts, result.intensity_high_ts, result.intensity_low_ts) if da is not None), None)
        int_dtype = np.dtype(int_ref.dtype) if int_ref is not None else np.dtype(np.float64)
        meta_out = {**result.metadata, "out_format": fmt}

        spec = cls(
            spatial_dims=spatial_dims,
            spatial_shape=spatial_shape,
            spatial_coords=spatial_coords,
            det_time=det_time,
            time_dim=time_dim,
            year_len=year_len,
            moving_years=moving_years,
            write_clim=result.clim is not None,
            write_thresh_high=result.thresh_high is not None,
            write_thresh_low=result.thresh_low is not None,
            clim_dtype=clim_dtype,
            write_intensity_ts=result.intensity_ts is not None,
            write_intensity_high_ts=result.intensity_high_ts is not None,
            write_intensity_low_ts=result.intensity_low_ts is not None,
            intensity_dtype=int_dtype,
            write_daily_category_ts=result.daily_category_ts is not None,
            write_event_category_ts=result.event_category_ts is not None,
            write_trend=result.trend is not None,
            write_events=result.events is not None,
            attrs=meta_out,
        )

        grid_sources: list[tuple[str, xr.DataArray, tuple[str, ...]]] = []
        for name, da, target_dims in (
            ("clim", result.clim, spec.clim_dims),
            ("thresh_high", result.thresh_high, spec.clim_dims),
            ("thresh_low", result.thresh_low, spec.clim_dims),
            ("intensity_ts", result.intensity_ts, spec.ts_dims),
            ("intensity_high_ts", result.intensity_high_ts, spec.ts_dims),
            ("intensity_low_ts", result.intensity_low_ts, spec.ts_dims),
            ("daily_category_ts", result.daily_category_ts, spec.ts_dims),
            ("event_category_ts", result.event_category_ts, spec.ts_dims),
        ):
            if da is not None:
                grid_sources.append((name, da, target_dims))
        if result.trend is not None:
            for short, full in TREND_VAR_KEYS:
                if short in result.trend:
                    grid_sources.append((full, result.trend[short], spec.spatial_dims))
        return spec, grid_sources


def _staging_path(path: Path, owner: Any) -> Path:
    """Build a unique sibling temp path for atomic replacement on finalize."""
    return path.parent / f".{path.name}.tmp.{os.getpid()}.{id(owner)}"


def _cleanup_path(path: Path | None) -> None:
    """Best-effort removal of an uncommitted staging file or directory."""
    if path is None or not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except Exception:
        pass


def _replace_directory(src_dir: Path, dst: Path) -> None:
    """Swap a staged Zarr directory into place with rollback on replacement failure."""
    backup: Path | None = None
    if dst.exists():
        if dst.is_dir():
            backup = dst.parent / f".{dst.name}.old.{os.getpid()}.{id(src_dir)}"
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(dst, backup)
        else:
            dst.unlink()
    try:
        os.replace(src_dir, dst)
    except BaseException:
        if backup is not None and backup.exists() and not dst.exists():
            os.replace(backup, dst)
        raise
    finally:
        if backup is not None and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _compute_bounded_chunks(
    shape: tuple[int, ...],
    itemsize: int,
    target_bytes: int = 4 * 1024 * 1024,
    max_last_axis: int = 4096,
) -> tuple[int, ...]:
    """Compute HDF5/NetCDF/Zarr chunk shape bounded to ``<= target_bytes`` (strictly < 4 GiB HDF5 limit)."""
    if not shape:
        return ()
    chunks = [max(1, min(int(s), 64)) for s in shape[:-1]]
    chunks.append(max(1, min(int(shape[-1]), max_last_axis)))
    elem_size = max(1, int(itemsize))
    for i in range(len(chunks) - 1):
        while chunks[i] > 1 and math.prod(chunks) * elem_size > target_bytes:
            chunks[i] = max(1, chunks[i] // 2)
    while chunks[-1] > 1 and math.prod(chunks) * elem_size > target_bytes:
        chunks[-1] = max(1, chunks[-1] // 2)
    return tuple(chunks)


def _str_to_s1_2d(values: Any, max_bytes: int) -> np.ndarray:
    """Encode strings as a fixed-width ``(N, max_bytes)`` UTF-8 ``S1`` array (avoids HDF5 vlen NC_STRING)."""
    strs = to_str_list(values)
    arr = np.zeros((len(strs), max_bytes), dtype="S1")
    for i, s in enumerate(strs):
        b = s.encode("utf-8")[:max_bytes]
        if b:
            arr[i, : len(b)] = np.frombuffer(b, dtype="S1")
    return arr


def _flatten_dict_cols(tbl: pa.Table) -> pa.Table:
    """Cast Arrow dictionary columns to plain strings for Feather/CSV compatibility."""
    new_cols = []
    new_fields = []
    for i, fld in enumerate(tbl.schema):
        col = tbl.column(i)
        if pa.types.is_dictionary(fld.type):
            col = col.cast(pa.string())
            fld = pa.field(fld.name, pa.string())
        new_cols.append(col)
        new_fields.append(fld)
    return pa.Table.from_arrays(new_cols, schema=pa.schema(new_fields))


def _serialize_spatial_coords(spatial_coords: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Serialize spatial coordinate arrays into JSON-safe dtype/value dicts."""
    out: dict[str, dict[str, Any]] = {}
    for k, arr in spatial_coords.items():
        a = np.asarray(arr)
        if is_str_dtype(a.dtype):
            out[k] = {"dtype": "str", "values": to_str_list(a.tolist())}
        else:
            out[k] = {"dtype": str(a.dtype), "values": a.tolist()}
    return out


def _deserialize_spatial_coords(raw: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    """Restore spatial coordinate arrays from JSON-serialized metadata."""
    out: dict[str, np.ndarray] = {}
    for k, spec in raw.items():
        if spec["dtype"] == "str":
            out[k] = np.array(spec["values"], dtype=object)
        else:
            out[k] = np.array(spec["values"], dtype=np.dtype(spec["dtype"]))
    return out


def _encode_csr_counts(grid_offsets: np.ndarray) -> str:
    """Delta-compress CSR grid_offsets as base64(zlib(counts)) for Parquet footer storage."""
    counts = np.diff(np.asarray(grid_offsets, dtype=np.int64)).astype(np.int64, copy=False)
    return base64.b64encode(zlib.compress(counts.tobytes(), level=6)).decode("ascii")


def _decode_csr_counts(encoded: str, max_bytes: int | None = None) -> np.ndarray:
    """Decode delta-compressed CSR counts back to cumulative int64 grid_offsets."""
    raw = base64.b64decode(encoded.encode("ascii"))
    if max_bytes is not None:
        dobj = zlib.decompressobj()
        data = dobj.decompress(raw, max_bytes)
        if dobj.unconsumed_tail:
            raise ValueError("Compressed CSR metadata exceeds expected grid size.")
    else:
        data = zlib.decompress(raw)
    counts = np.frombuffer(data, dtype=np.int64)
    offsets = np.empty(counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets


def _offsets_from_pixel_idx(pixel_idx: np.ndarray, n_pixels: int) -> np.ndarray:
    """Rebuild CSR grid_offsets from sorted per-event flat pixel indices."""
    offsets = np.zeros(n_pixels + 1, dtype=np.int64)
    if pixel_idx.size > 0 and n_pixels > 0:
        counts = np.bincount(pixel_idx.astype(np.int64, copy=False), minlength=n_pixels)
        np.cumsum(counts[:n_pixels], out=offsets[1:])
    return offsets


def _encode_event_col_numpy(
    fld: pa.Field,
    col: pa.ChunkedArray | pa.Array,
    *,
    bool_dtype: Any = np.bool_,
    str_max_bytes: int | None = None,
) -> np.ndarray:
    """Shared encoder converting one Arrow event column to its disk array representation."""
    name = fld.name
    if name == "event_type":
        return encode_event_type_codes(col)
    if name in DATE_EVENT_COLS:
        return dates_to_yyyymmdd(col)
    if pa.types.is_boolean(fld.type):
        return col.to_numpy(zero_copy_only=False).astype(bool_dtype)
    if str_max_bytes is not None:
        return _str_to_s1_2d(col.to_pylist(), str_max_bytes)
    if pa.types.is_string(fld.type) or pa.types.is_large_string(fld.type):
        return np.asarray(to_str_list(col.to_pylist()), dtype=str)
    vals = col.to_numpy(zero_copy_only=False)
    if is_str_dtype(vals.dtype):
        return np.asarray(to_str_list(vals.tolist()), dtype=str)
    return vals


def _common_meta_attrs(spec: OutputSpec, fmt: OutFormat, *, include_coords: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {
        "marine_extremes_format": fmt,
        "marine_extremes_metadata_json": json.dumps(spec.attrs, default=str),
        "marine_extremes_spatial_dims": json.dumps(list(spec.spatial_dims)),
        "marine_extremes_spatial_shape": json.dumps(list(spec.spatial_shape)),
        "marine_extremes_time_dim": spec.time_dim,
        "marine_extremes_year_len": int(spec.year_len),
        "marine_extremes_has_moving_years": bool(spec.moving_years is not None),
    }
    if include_coords:
        out["marine_extremes_spatial_coords_json"] = json.dumps(_serialize_spatial_coords(spec.spatial_coords))
    return out


# ============================================================================
# Format normalization & magic-byte detection
# ============================================================================


def normalize_mat_layout(mat_layout: str) -> MatLayout:
    lay = str(mat_layout).lower()
    if lay in ("v73", "hdf5", "struct"):
        return "v73"
    if lay in ("detectc", "cell", "v5"):
        return "detectc"
    raise ValueError(f"Unsupported mat_layout='{mat_layout}'. Expected 'v73' or 'detectc'.")


def resolve_out_format(
    path: str | Path,
    out_format: str | None = None,
    mat_layout: str = "v73",
) -> OutFormat:
    """Resolve single-file output format from explicit ``out_format`` or file extension."""
    lay = normalize_mat_layout(mat_layout)
    if out_format is not None:
        f = str(out_format).lower().lstrip(".")
        aliases: dict[str, OutFormat] = {
            "netcdf": "netcdf", "nc": "netcdf", "nc4": "netcdf", "netcdf4": "netcdf",
            "mat": "mat_detectc" if lay == "detectc" else "mat",
            "mat73": "mat_detectc" if lay == "detectc" else "mat",
            "matlab": "mat_detectc" if lay == "detectc" else "mat",
            "mat_detectc": "mat_detectc", "mat_cell": "mat_detectc", "detectc": "mat_detectc", "mat5": "mat_detectc",
            "zarr": "zarr",
            "npz": "npz", "numpy": "npz",
            "parquet": "parquet", "pq": "parquet",
            "feather": "feather", "ipc": "feather", "arrow": "feather",
            "csv": "csv",
        }
        if f in aliases:
            return aliases[f]
        raise ValueError(
            f"Unsupported out_format='{out_format}'. Supported formats: "
            "'netcdf' (.nc), 'mat' (.mat v7.3), 'mat_detectc' (.mat v5), 'zarr' (.zarr), "
            "'npz' (.npz), 'parquet' (.parquet), 'feather' (.feather), 'csv' (.csv)."
        )

    suffix = Path(path).suffix.lower()
    if suffix not in _EXT_TO_FORMAT:
        raise ValueError(
            f"Cannot infer output format from path '{path}'. "
            "Use a supported extension (.nc, .mat, .zarr, .npz, .parquet, .feather, .csv) "
            "or specify out_format explicitly."
        )
    fmt = _EXT_TO_FORMAT[suffix]
    return "mat_detectc" if (fmt == "mat" and lay == "detectc") else fmt


def detect_file_format(path: Path) -> OutFormat:
    """Detect format of an existing marine-extremes output using magic bytes, Zarr markers, or extension."""
    if not path.exists():
        raise FileNotFoundError(f"Output path does not exist: {path}")
    if path.is_dir():
        if (path / "zarr.json").exists() or (path / ".zgroup").exists():
            return "zarr"
        raise IsADirectoryError(
            f"Expected a marine-extremes output file or .zarr store, got plain directory '{path}'."
        )
    with open(path, "rb") as fh:
        head = fh.read(16)
    for magic, fmt in (
        (b"MATLAB 7.3", "mat"),
        (b"MATLAB 5.0", "mat_detectc"),
        (b"PAR1", "parquet"),
        (b"ARROW1", "feather"),
        (b"PK\x03\x04", "npz"),
        (b"\x89HDF\r\n\x1a\n", "netcdf"),
        (b"CDF", "netcdf"),
    ):
        if head.startswith(magic):
            return fmt  # type: ignore[return-value]
    suffix = path.suffix.lower()
    if suffix in _EXT_TO_FORMAT:
        return _EXT_TO_FORMAT[suffix]
    raise ValueError(f"Unable to detect marine-extremes file format for '{path}'.")


# ============================================================================
# Shared writer protocol & buffered-grid base
# ============================================================================


class _FileWriter(Protocol):
    writes_grids: bool

    def write_grid_tile(self, name: str, tile: SpatialTile, data: np.ndarray, extra_ndims: int) -> None: ...
    def write_empty_tile(self, tile: SpatialTile) -> None: ...
    def write_events_table(self, tbl: pa.Table) -> None: ...
    def finalize(self, empty_schema: pa.Schema | None, grid_offsets: np.ndarray | None) -> None: ...
    def close(self) -> None: ...


class _BufferedGridWriter:
    """Shared in-memory grid/event buffer for archive-style formats (MAT v5 detectc & NPZ)."""

    writes_grids: bool = True

    def __init__(self, path: Path, spec: OutputSpec, *, max_bytes: int | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.spec = spec
        self._schema: pa.Schema | None = None
        self._event_tables: list[pa.Table] = []

        var_specs = spec.grid_var_specs()
        est_bytes = sum((math.prod(shape) if shape else 1) * dt.itemsize for _, _, shape, dt, _ in var_specs)
        if max_bytes is not None and est_bytes > max_bytes:
            raise ValueError(
                f"Estimated uncompressed grid size (~{est_bytes / 1e9:.2f} GB) exceeds the "
                f"{max_bytes / 1e9:.1f} GB limit for this format. Use '.nc' or '.mat' (mat_layout='v73') instead."
            )
        self._grids: dict[str, np.ndarray] = {
            name: np.full(shape, fill, dtype=dt) for name, _, shape, dt, fill in var_specs
        }

    def write_grid_tile(self, name: str, tile: SpatialTile, data: np.ndarray, extra_ndims: int) -> None:
        buf = self._grids[name]
        if self.spec.spatial_dims:
            idx = tuple(tile.isel_dict[d] for d in self.spec.spatial_dims) + (slice(None),) * extra_ndims
            buf[idx] = data
        elif extra_ndims == 0:
            buf[()] = np.asarray(data).reshape(())
        else:
            buf[...] = data

    def write_empty_tile(self, tile: SpatialTile) -> None:
        return  # Buffers are pre-filled with NaN / CATEGORY_MISSING

    def write_events_table(self, tbl: pa.Table) -> None:
        if self._schema is None:
            self._schema = tbl.schema
        if self.spec.write_events and tbl.num_rows > 0:
            self._event_tables.append(tbl)

    def _concat_events(self, empty_schema: pa.Schema | None) -> pa.Table:
        schema = self._schema or empty_schema
        assert schema is not None
        return pa.concat_tables(self._event_tables) if self._event_tables else schema.empty_table()

    def close(self) -> None:
        self._grids.clear()
        self._event_tables.clear()


# ============================================================================
# Format Adapter 1: NetCDF4 (streaming grids + /events subgroup)
# ============================================================================


class _NetCDFWriter:
    writes_grids: bool = True

    def __init__(self, path: Path, spec: OutputSpec) -> None:
        import netCDF4 as nc4

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.spec = spec
        self._tmp_path = _staging_path(path, self)
        self._finalized = False
        self._schema: pa.Schema | None = None
        self._ev_grp: Any = None
        self._ev_cursor: int = 0
        self._str_coord_max_bytes: dict[str, int] = {}
        self._ev_str_max_bytes: dict[str, int] = {}

        ds = nc4.Dataset(str(self._tmp_path), mode="w", format="NETCDF4")
        self._ds = ds
        try:
            for k, v in spec.attrs.items():
                if v is not None:
                    ds.setncattr(k, int(v) if isinstance(v, bool) else (v if isinstance(v, (str, int, float)) else str(v)))
            for k, v in _common_meta_attrs(spec, "netcdf", include_coords=False).items():
                ds.setncattr(k, int(v) if isinstance(v, bool) else v)

            for d, n in zip(spec.spatial_dims, spec.spatial_shape):
                ds.createDimension(d, int(n))
                cvals = np.asarray(spec.spatial_coords[d])
                if is_str_dtype(cvals.dtype):
                    strs = to_str_list(cvals.tolist())
                    max_b = max(1, max((len(s.encode("utf-8")) for s in strs), default=1))
                    self._str_coord_max_bytes[d] = max_b
                    sdim = f"{d}_strlen"
                    ds.createDimension(sdim, max_b)
                    v = ds.createVariable(d, "S1", (d, sdim))
                    v._Encoding = "utf-8"
                    v[:] = _str_to_s1_2d(strs, max_b)
                else:
                    v = ds.createVariable(d, cvals.dtype, (d,))
                    v[:] = cvals

            if spec.has_clim:
                ds.createDimension("doy", int(spec.year_len))
                ds.createVariable("doy", "i4", ("doy",))[:] = np.arange(1, spec.year_len + 1, dtype=np.int32)
                if spec.moving_years is not None:
                    ds.createDimension("year", int(spec.moving_years.shape[0]))
                    ds.createVariable("year", "i4", ("year",))[:] = spec.moving_years.astype(np.int32)

            if spec.has_ts:
                ds.createDimension(spec.time_dim, int(spec.det_time.shape[0]))
                t_days, t_units, t_cal = time_to_cf_days(spec.det_time)
                v_t = ds.createVariable(spec.time_dim, "f8", (spec.time_dim,))
                v_t.units = t_units
                v_t.calendar = t_cal
                v_t[:] = t_days

            units_str = str(spec.attrs.get("units", "")).strip()
            for name, dims, shape, dt, fill in spec.grid_var_specs():
                chunks = _compute_bounded_chunks(shape, dt.itemsize) if dims else None
                kw: dict[str, Any] = {"zlib": True, "complevel": 1} if dims else {}
                if chunks:
                    kw["chunksizes"] = chunks
                if dt == np.dtype(np.int8):
                    # Omit _FillValue on int8 category vars so xarray.open_dataset does not upcast int8 -> float.
                    v = ds.createVariable(name, "i1", dims, **kw)
                else:
                    code = "f4" if dt == np.dtype(np.float32) else "f8"
                    v = ds.createVariable(name, code, dims, fill_value=fill, **kw)
                for ak, av in grid_var_attrs(name, units_str).items():
                    v.setncattr(ak, np.int8(av) if ak == "missing_value_code" else av)
        except BaseException:
            self.close()
            raise

    def _init_events_group(self, schema: pa.Schema) -> None:
        assert self._ds is not None
        grp = self._ds.createGroup("events")
        grp.createDimension("event", None)
        self._ev_grp = grp
        for fld in schema:
            name = fld.name
            if name == "event_type":
                v = grp.createVariable(name, "i1", ("event",), zlib=True, complevel=1)
                v.description = "+1 = high (MHW), -1 = low (MCS)"
            elif name in DATE_EVENT_COLS:
                v = grp.createVariable(name, "i4", ("event",), zlib=True, complevel=1)
                v.format = "YYYYMMDD"
            elif pa.types.is_boolean(fld.type):
                grp.createVariable(name, "i1", ("event",), zlib=True, complevel=1)
            elif pa.types.is_string(fld.type) or pa.types.is_large_string(fld.type):
                max_b = self._str_coord_max_bytes.get(name, 256)
                self._ev_str_max_bytes[name] = max_b
                sdim = f"{name}_ev_strlen"
                grp.createDimension(sdim, max_b)
                v = grp.createVariable(name, "S1", ("event", sdim), zlib=True, complevel=1)
                v._Encoding = "utf-8"
            else:
                grp.createVariable(name, fld.type.to_pandas_dtype(), ("event",), zlib=True, complevel=1)

    def write_grid_tile(self, name: str, tile: SpatialTile, data: np.ndarray, extra_ndims: int) -> None:
        assert self._ds is not None
        v = self._ds.variables[name]
        if self.spec.spatial_dims:
            idx = tuple(tile.isel_dict[d] for d in self.spec.spatial_dims) + (slice(None),) * extra_ndims
            v[idx] = data
        elif extra_ndims == 0:
            v.assignValue(float(np.asarray(data).reshape(())))
        else:
            v[:] = data

    def write_empty_tile(self, tile: SpatialTile) -> None:
        # NetCDF int8 category vars intentionally omit _FillValue; fill pure-land tiles explicitly here.
        if not self.spec.has_ts:
            return
        n_det = int(self.spec.det_time.shape[0])
        for name, enabled in (
            ("daily_category_ts", self.spec.write_daily_category_ts),
            ("event_category_ts", self.spec.write_event_category_ts),
        ):
            if enabled:
                miss = np.full((*tile.tile_shape, n_det), CATEGORY_MISSING, dtype=np.int8)
                self.write_grid_tile(name, tile, miss, 1)

    def write_events_table(self, tbl: pa.Table) -> None:
        if self._schema is None:
            self._schema = tbl.schema
        if not self.spec.write_events or tbl.num_rows == 0:
            return
        if self._ev_grp is None:
            self._init_events_group(tbl.schema)
        n = tbl.num_rows
        sl = slice(self._ev_cursor, self._ev_cursor + n)
        for fld in tbl.schema:
            name = fld.name
            max_b = self._ev_str_max_bytes.get(name)
            vals = _encode_event_col_numpy(fld, tbl.column(name), bool_dtype=np.int8, str_max_bytes=max_b)
            if max_b is not None:
                self._ev_grp.variables[name][sl, :] = vals
            else:
                self._ev_grp.variables[name][sl] = vals
        self._ev_cursor += n

    def finalize(self, empty_schema: pa.Schema | None, grid_offsets: np.ndarray | None) -> None:
        if self._ds is None:
            return
        if self.spec.write_events and grid_offsets is not None:
            schema = self._schema or empty_schema
            if self._ev_grp is None and schema is not None:
                self._init_events_group(schema)
            if self._ev_grp is not None:
                self._ev_grp.createDimension("grid_offsets", int(grid_offsets.shape[0]))
                self._ev_grp.createVariable("grid_offsets", "i8", ("grid_offsets",), zlib=True, complevel=1)[:] = grid_offsets
        self._ds.close()
        self._ds = None
        self._ev_grp = None
        os.replace(self._tmp_path, self.path)
        self._finalized = True

    def close(self) -> None:
        if self._ds is not None:
            try:
                self._ds.close()
            except Exception:
                pass
            self._ds = None
            self._ev_grp = None
        if not self._finalized:
            _cleanup_path(self._tmp_path)


def _read_nc_events_group(
    grp: Any,
    *,
    start: int | None = None,
    stop: int | None = None,
    calendar: str | None = None,
    date_resolution: str | None = None,
) -> pa.Table:
    cols: dict[str, Any] = {}
    is_slice = start is not None and stop is not None
    length = (stop - start) if is_slice else None
    for name, v in grp.variables.items():
        if name == "grid_offsets":
            continue
        is_char = v.dtype == np.dtype("S1") and v.ndim == 2
        if is_slice:
            assert start is not None and stop is not None and length is not None
            raw = np.asarray(v[start:stop]) if length > 0 else np.array([], dtype=str if is_char else v.dtype)
        else:
            raw = np.asarray(v[:]) if v.shape[0] > 0 else np.array([], dtype=str if is_char else v.dtype)
        cols[name] = decode_event_column(name, raw, calendar=calendar, date_resolution=date_resolution)
    return pa.table(cols)


class _NetCDFAdapter:
    def create_writer(self, path: Path, spec: OutputSpec) -> _NetCDFWriter:
        return _NetCDFWriter(path, spec)

    def _read_events(
        self,
        path: Path,
        *,
        start: int | None = None,
        stop: int | None = None,
        calendar: str | None = None,
        date_resolution: str | None = None,
    ) -> pa.Table:
        import netCDF4 as nc4

        with nc4.Dataset(str(path), "r") as ds:
            meta_json = getattr(ds, "marine_extremes_metadata_json", "{}")
            meta = json.loads(meta_json) if meta_json else {}
            cal = calendar if calendar is not None else meta.get("calendar")
            dres = date_resolution if date_resolution is not None else meta.get("date_resolution")
            return _read_nc_events_group(ds.groups["events"], start=start, stop=stop, calendar=cal, date_resolution=dres)

    def open(self, path: Path, *, events_in_memory: bool) -> tuple[xr.Dataset | None, EventCatalog | None, dict[str, Any], list[Any]]:
        import netCDF4 as nc4

        with nc4.Dataset(str(path), "r") as nc:
            meta_json = getattr(nc, "marine_extremes_metadata_json", "{}")
            metadata = json.loads(meta_json) if meta_json else {}
            cal = metadata.get("calendar")
            dres = metadata.get("date_resolution")
            spatial_dims = tuple(json.loads(getattr(nc, "marine_extremes_spatial_dims", "[]")))
            spatial_shape = tuple(json.loads(getattr(nc, "marine_extremes_spatial_shape", "[]")))
            spatial_coords = {
                d: (np.asarray(nc.variables[d][:]) if d in nc.variables else np.arange(n))
                for d, n in zip(spatial_dims, spatial_shape)
            }
            has_grids = any(v in nc.variables for v in ALL_GRID_VAR_NAMES)

            catalog = None
            if "events" in nc.groups:
                ev_grp = nc.groups["events"]
                grid_offsets = np.asarray(ev_grp.variables["grid_offsets"][:], dtype=np.int64)
                if events_in_memory:
                    store = MemoryEventStore(
                        _read_nc_events_group(ev_grp, calendar=cal, date_resolution=dres),
                        calendar=cal,
                        date_resolution=dres,
                    )
                else:
                    store = DiskEventStore(
                        path,
                        slice_fn=lambda s, e: self._read_events(path, start=s, stop=e, calendar=cal, date_resolution=dres),
                        read_all_fn=lambda: self._read_events(path, calendar=cal, date_resolution=dres),
                        calendar=cal,
                        date_resolution=dres,
                    )
                catalog = EventCatalog(None, grid_offsets, spatial_dims, spatial_shape, spatial_coords, store=store)

        grids_ds = xr.open_dataset(path) if has_grids else None
        return grids_ds, catalog, metadata, ([grids_ds] if grids_ds is not None else [])


# ============================================================================
# Format Adapter 2: MATLAB v7.3 HDF5 (streaming grids + lazy BackendArray)
# ============================================================================


def _write_mat_v73_header(path: Path) -> None:
    """Overwrite the 512-byte HDF5 userblock prefix with a valid 128-byte MATLAB v7.3 header."""
    desc = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created by marine-extremes"
    header = desc.ljust(116, b" ") + (b"\x00" * 8) + struct.pack("<H", 0x0200) + b"IM"
    assert len(header) == 128
    with open(path, "r+b") as fh:
        fh.seek(0)
        fh.write(header)


def _tag_matlab_dataset(ds: Any, np_dtype: np.dtype) -> None:
    """Attach MATLAB_class (and logical decode flag) attrs required by MATLAB load()."""
    dt = np.dtype(np_dtype)
    ds.attrs["MATLAB_class"] = np.bytes_(_NP_TO_MATLAB_CLASS.get(dt, b"double"))
    if dt == np.dtype(np.bool_):
        ds.attrs["MATLAB_int_decode"] = np.int32(1)


def _write_matlab_struct_attrs(grp: Any, field_names: list[str]) -> None:
    """Mark an HDF5 group as a MATLAB struct with ordered variable-length field names."""
    import h5py

    grp.attrs["MATLAB_class"] = np.bytes_(b"struct")
    if field_names:
        vlen_s1 = h5py.vlen_dtype(np.dtype("S1"))
        arr = np.empty(len(field_names), dtype=object)
        for i, fn in enumerate(field_names):
            arr[i] = np.frombuffer(fn.encode("ascii"), dtype="S1")
        grp.attrs.create("MATLAB_fields", arr, dtype=vlen_s1)


def _read_matlab_struct_field_order(grp: Any) -> list[str]:
    if "MATLAB_fields" in grp.attrs:
        order: list[str] = []
        for entry in grp.attrs["MATLAB_fields"]:
            name = b"".join(entry).decode("ascii")
            if name in grp:
                order.append(name)
        for k in grp.keys():
            if k not in order:
                order.append(k)
        return order
    return list(grp.keys())


def _write_mat_v73_dataset(
    parent: Any,
    name: str,
    arr: np.ndarray,
    *,
    logical_dtype: np.dtype | None = None,
    compress: bool = True,
) -> Any:
    a = np.asarray(arr)
    eff_dt = np.dtype(logical_dtype) if logical_dtype is not None else np.dtype(a.dtype)
    if a.size == 0:
        mat_dims = np.array([0, 0] if a.ndim == 0 else ([0, 1] if a.ndim == 1 else list(a.shape)), dtype=np.uint64)
        ds = parent.create_dataset(name, data=mat_dims)
        _tag_matlab_dataset(ds, eff_dt)
        ds.attrs["MATLAB_empty"] = np.uint8(1)
        return ds

    a2 = a.reshape(1, 1) if a.ndim == 0 else (a.reshape(1, -1) if a.ndim == 1 else np.transpose(a))
    if eff_dt == np.dtype(np.bool_) and a2.dtype != np.dtype(np.uint8):
        a2 = a2.astype(np.uint8, copy=False)
    ds = (
        parent.create_dataset(name, data=a2, compression="gzip", compression_opts=1)
        if (compress and a2.size >= 16)
        else parent.create_dataset(name, data=a2)
    )
    _tag_matlab_dataset(ds, eff_dt)
    return ds


class _H5TransposedBackendArray(xr.backends.BackendArray):
    """Lazy xarray BackendArray wrapping a transposed MATLAB v7.3 HDF5 dataset."""

    def __init__(self, ds: Any, logical_shape: tuple[int, ...]) -> None:
        self._ds = ds
        self.shape = tuple(int(s) for s in logical_shape)
        self.dtype = np.dtype(ds.dtype)

    def _raw_indexing_method(self, key: tuple[int | slice, ...]) -> np.ndarray:
        if not self.shape:
            return np.asarray(self._ds[...]).reshape(())
        pos_key: list[int | slice] = []
        flips: list[bool] = []
        for k, dim_len in zip(key, self.shape):
            if isinstance(k, slice) and k.step is not None and k.step < 0:
                idx = np.arange(*k.indices(dim_len))
                if idx.size == 0:
                    pos_key.append(slice(0, 0, 1))
                    flips.append(False)
                else:
                    pos_key.append(slice(int(idx[-1]), int(idx[0]) + 1, abs(k.step)))
                    flips.append(True)
            else:
                pos_key.append(k)
                flips.append(False)

        if len(self.shape) == 1 and self._ds.ndim == 2:
            arr = np.asarray(self._ds[pos_key[0], 0])
            return arr[::-1] if (flips[0] and arr.ndim == 1) else arr

        arr = np.transpose(np.asarray(self._ds[tuple(pos_key[::-1])]))
        flip_slices = [
            slice(None, None, -1) if do_flip else slice(None)
            for k, do_flip in zip(pos_key, flips)
            if not isinstance(k, (int, np.integer))
        ]
        return arr[tuple(flip_slices)] if any(s.step == -1 for s in flip_slices) else arr

    def __getitem__(self, key: Any) -> np.ndarray:
        from xarray.core import indexing

        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._raw_indexing_method
        )


class _MatV73Writer:
    writes_grids: bool = True

    def __init__(self, path: Path, spec: OutputSpec) -> None:
        import h5py

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.spec = spec
        self._tmp_path = _staging_path(path, self)
        self._finalized = False
        self._schema: pa.Schema | None = None
        self._ev_grp: Any = None
        self._ev_cursor: int = 0
        self._ev_field_dtypes: dict[str, np.dtype] = {}

        f = h5py.File(str(self._tmp_path), "w", userblock_size=512)
        self._f = f
        try:
            def _write_vec(name: str, arr: np.ndarray) -> None:
                if not is_str_dtype(arr.dtype):
                    _write_mat_v73_dataset(f, name, np.asarray(arr).reshape(-1), compress=False)

            for d in spec.spatial_dims:
                _write_vec(d, np.asarray(spec.spatial_coords[d]))
            if spec.has_clim:
                _write_vec("doy", np.arange(1, spec.year_len + 1, dtype=np.int32))
                if spec.moving_years is not None:
                    _write_vec("year", spec.moving_years.astype(np.int32))

            t_units = CF_TIME_UNITS
            t_cal = str(spec.attrs.get("calendar", "proleptic_gregorian"))
            if spec.has_ts:
                t_days, t_units, t_cal = time_to_cf_days(spec.det_time)
                _write_vec(spec.time_dim, t_days)
                _write_vec(f"{spec.time_dim}_yyyymmdd", dates_to_yyyymmdd(spec.det_time))

            for name, _, shape, dt, fill in spec.grid_var_specs():
                logical_shape = shape if shape else (1,)
                eff_shape = logical_shape if len(logical_shape) >= 2 else (1, logical_shape[0])
                if 0 in eff_shape:
                    ds = f.create_dataset(name, shape=eff_shape[::-1], dtype=dt, fillvalue=fill)
                else:
                    h5_chunks = _compute_bounded_chunks(eff_shape, dt.itemsize)[::-1]
                    ds = f.create_dataset(
                        name,
                        shape=eff_shape[::-1],
                        dtype=dt,
                        chunks=h5_chunks,
                        compression="gzip",
                        compression_opts=1,
                        fillvalue=fill,
                    )
                _tag_matlab_dataset(ds, dt)

            for k, v in _common_meta_attrs(spec, "mat", include_coords=False).items():
                f.attrs[k] = v
            str_coords = {
                d: spec.spatial_coords[d]
                for d in spec.spatial_dims
                if is_str_dtype(np.asarray(spec.spatial_coords[d]).dtype)
            }
            if str_coords:
                f.attrs["marine_extremes_spatial_coords_json"] = json.dumps(_serialize_spatial_coords(str_coords))
            f.attrs["marine_extremes_time_units"] = t_units
            f.attrs["marine_extremes_time_calendar"] = t_cal
        except BaseException:
            self.close()
            raise

    def _init_events_group(self, schema: pa.Schema) -> None:
        assert self._f is not None
        grp = self._f.create_group("events")
        self._ev_grp = grp
        self._ev_field_dtypes = {}
        for fld in schema:
            name = fld.name
            if name == "event_type":
                dt, tag_dt = np.dtype(np.int8), np.dtype(np.int8)
            elif name in DATE_EVENT_COLS:
                dt, tag_dt = np.dtype(np.int32), np.dtype(np.int32)
            elif pa.types.is_boolean(fld.type):
                dt, tag_dt = np.dtype(np.uint8), np.dtype(np.bool_)
            elif pa.types.is_string(fld.type) or pa.types.is_large_string(fld.type):
                warnings.warn(
                    f"MATLAB v7.3 events struct skips string column '{name}'; "
                    "use NetCDF/Zarr/Parquet/NPZ to preserve string columns.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            else:
                dt = tag_dt = np.dtype(fld.type.to_pandas_dtype())
            ds = grp.create_dataset(
                name, shape=(1, 0), maxshape=(1, None), chunks=(1, 4096),
                dtype=dt, compression="gzip", compression_opts=1,
            )
            _tag_matlab_dataset(ds, tag_dt)
            self._ev_field_dtypes[name] = tag_dt
        _write_matlab_struct_attrs(grp, list(self._ev_field_dtypes.keys()))

    def write_grid_tile(self, name: str, tile: SpatialTile, data: np.ndarray, extra_ndims: int) -> None:
        assert self._f is not None
        ds = self._f[name]
        if self.spec.spatial_dims:
            if len(self.spec.spatial_dims) + extra_ndims == 1:
                ds[tile.isel_dict[self.spec.spatial_dims[0]], 0] = data.reshape(-1)
            else:
                logical_idx = tuple(tile.isel_dict[d] for d in self.spec.spatial_dims) + (slice(None),) * extra_ndims
                ds[logical_idx[::-1]] = np.transpose(data)
        elif extra_ndims == 0:
            ds[0, 0] = data.reshape(())
        elif extra_ndims == 1:
            ds[:, 0] = data.reshape(-1)
        else:
            ds[...] = np.transpose(data)

    def write_empty_tile(self, tile: SpatialTile) -> None:
        return  # HDF5 datasets are created with fillvalue=NaN / CATEGORY_MISSING

    def write_events_table(self, tbl: pa.Table) -> None:
        if self._schema is None:
            self._schema = tbl.schema
        if not self.spec.write_events or tbl.num_rows == 0:
            return
        if self._ev_grp is None:
            self._init_events_group(tbl.schema)
        new_total = self._ev_cursor + tbl.num_rows
        for fld in tbl.schema:
            name = fld.name
            if name not in self._ev_grp:
                continue
            ds = self._ev_grp[name]
            ds.resize((1, new_total))
            ds[0, self._ev_cursor : new_total] = _encode_event_col_numpy(fld, tbl.column(name), bool_dtype=np.uint8)
        self._ev_cursor = new_total

    def finalize(self, empty_schema: pa.Schema | None, grid_offsets: np.ndarray | None) -> None:
        if self._f is None:
            return
        if self.spec.write_events and grid_offsets is not None:
            schema = self._schema or empty_schema
            if self._ev_grp is None and schema is not None:
                self._init_events_group(schema)
            if self._ev_grp is not None and self._ev_cursor == 0:
                for fname, fdt in self._ev_field_dtypes.items():
                    del self._ev_grp[fname]
                    _write_mat_v73_dataset(
                        self._ev_grp, fname, np.empty((0,), dtype=fdt), logical_dtype=fdt, compress=False
                    )
            _write_mat_v73_dataset(self._f, "grid_offsets", grid_offsets.reshape(-1), compress=False)
        self._f.close()
        self._f = None
        self._ev_grp = None
        _write_mat_v73_header(self._tmp_path)
        os.replace(self._tmp_path, self.path)
        self._finalized = True

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None
            self._ev_grp = None
        if not self._finalized:
            _cleanup_path(self._tmp_path)


def _read_mat_v73_events_group(
    grp: Any,
    *,
    start: int | None = None,
    stop: int | None = None,
    calendar: str | None = None,
    date_resolution: str | None = None,
) -> pa.Table:
    cols: dict[str, Any] = {}
    is_slice = start is not None and stop is not None
    length = (stop - start) if is_slice else None
    for name in _read_matlab_struct_field_order(grp):
        ds_h = grp[name]
        if (is_slice and (length is None or length <= 0)) or bool(ds_h.attrs.get("MATLAB_empty", 0)):
            mcls = bytes(ds_h.attrs.get("MATLAB_class", b"double"))
            raw = np.array([], dtype=_MATLAB_CLASS_TO_NP.get(mcls, np.float64))
        elif is_slice:
            assert start is not None and stop is not None
            raw = np.asarray(ds_h[0, start:stop]).reshape(-1)
        else:
            raw = np.asarray(ds_h[:]).reshape(-1)
        cols[name] = decode_event_column(name, raw, calendar=calendar, date_resolution=date_resolution)
    return pa.table(cols)


class _MatV73Adapter:
    def create_writer(self, path: Path, spec: OutputSpec) -> _MatV73Writer:
        return _MatV73Writer(path, spec)

    def _read_events(
        self,
        path: Path,
        *,
        start: int | None = None,
        stop: int | None = None,
        calendar: str | None = None,
        date_resolution: str | None = None,
    ) -> pa.Table:
        import h5py

        with h5py.File(str(path), "r") as f:
            meta = json.loads(f.attrs["marine_extremes_metadata_json"]) if "marine_extremes_metadata_json" in f.attrs else {}
            cal = calendar if calendar is not None else (
                meta.get("calendar")
                or (str(f.attrs["marine_extremes_time_calendar"]) if "marine_extremes_time_calendar" in f.attrs else None)
            )
            dres = date_resolution if date_resolution is not None else meta.get("date_resolution")
            return _read_mat_v73_events_group(f["events"], start=start, stop=stop, calendar=cal, date_resolution=dres)

    def open(self, path: Path, *, events_in_memory: bool) -> tuple[xr.Dataset | None, EventCatalog | None, dict[str, Any], list[Any]]:
        import h5py
        from xarray.core import indexing

        f = h5py.File(str(path), "r")
        try:
            metadata = json.loads(f.attrs["marine_extremes_metadata_json"]) if "marine_extremes_metadata_json" in f.attrs else {}
            cal = metadata.get("calendar") or (
                str(f.attrs["marine_extremes_time_calendar"]) if "marine_extremes_time_calendar" in f.attrs else None
            )
            dres = metadata.get("date_resolution")
            spatial_dims = tuple(json.loads(f.attrs["marine_extremes_spatial_dims"])) if "marine_extremes_spatial_dims" in f.attrs else ()
            spatial_shape = tuple(json.loads(f.attrs["marine_extremes_spatial_shape"])) if "marine_extremes_spatial_shape" in f.attrs else ()
            saved_coords = (
                _deserialize_spatial_coords(json.loads(f.attrs["marine_extremes_spatial_coords_json"]))
                if "marine_extremes_spatial_coords_json" in f.attrs
                else {}
            )
            spatial_coords = {
                d: (f[d][:].reshape(-1) if d in f else saved_coords.get(d, np.arange(n, dtype=np.int64)))
                for d, n in zip(spatial_dims, spatial_shape)
            }

            catalog = None
            if "events" in f and "grid_offsets" in f:
                grid_offsets = np.asarray(f["grid_offsets"][:], dtype=np.int64).reshape(-1)
                if events_in_memory:
                    store = MemoryEventStore(
                        _read_mat_v73_events_group(f["events"], calendar=cal, date_resolution=dres),
                        calendar=cal,
                        date_resolution=dres,
                    )
                else:
                    store = DiskEventStore(
                        path,
                        slice_fn=lambda s, e: self._read_events(path, start=s, stop=e, calendar=cal, date_resolution=dres),
                        read_all_fn=lambda: self._read_events(path, calendar=cal, date_resolution=dres),
                        calendar=cal,
                        date_resolution=dres,
                    )
                catalog = EventCatalog(None, grid_offsets, spatial_dims, spatial_shape, spatial_coords, store=store)

            grids_ds = None
            if any(g in f for g in ALL_GRID_VAR_NAMES):
                time_dim = str(f.attrs["marine_extremes_time_dim"])
                has_mov = bool(f.attrs["marine_extremes_has_moving_years"])
                coords: dict[str, Any] = dict(spatial_coords)
                if "doy" in f:
                    coords["doy"] = f["doy"][:].reshape(-1)
                if time_dim in f:
                    coords[time_dim] = cf_days_to_time(
                        f[time_dim][:],
                        str(f.attrs.get("marine_extremes_time_units", CF_TIME_UNITS)),
                        str(f.attrs.get("marine_extremes_time_calendar", "proleptic_gregorian")),
                    )
                elif f"{time_dim}_yyyymmdd" in f:
                    coords[time_dim] = yyyymmdd_to_dates(
                        f[f"{time_dim}_yyyymmdd"][:].reshape(-1),
                        calendar=str(f.attrs.get("marine_extremes_time_calendar", "proleptic_gregorian")),
                    )
                if has_mov and "year" in f:
                    coords["year"] = f["year"][:].reshape(-1)

                clim_dims = (*spatial_dims, "year", "doy") if has_mov else (*spatial_dims, "doy")
                ts_dims = (*spatial_dims, time_dim)
                units_str = str(metadata.get("units", "")).strip()
                data_vars: dict[str, xr.DataArray] = {}
                for names, dims in ((CLIM_VAR_NAMES, clim_dims), (TS_VAR_NAMES, ts_dims), (TREND_VAR_NAMES, spatial_dims)):
                    for vname in names:
                        if vname in f:
                            logical_shape = tuple(int(len(coords[d])) for d in dims)
                            lazy = indexing.LazilyIndexedArray(_H5TransposedBackendArray(f[vname], logical_shape))
                            data_vars[vname] = xr.DataArray(
                                lazy, dims=dims, name=vname, attrs=grid_var_attrs(vname, units_str)
                            )
                grids_ds = xr.Dataset(data_vars, coords=coords)
        except BaseException:
            try:
                f.close()
            except Exception:
                pass
            raise
        if grids_ds is not None:
            return grids_ds, catalog, metadata, [f]
        f.close()
        return None, catalog, metadata, []


# ============================================================================
# Format Adapter 3: Zarr v3 (streaming root grids + /events subgroup)
# ============================================================================


class _ZarrWriter:
    writes_grids: bool = True

    def __init__(self, path: Path, spec: OutputSpec) -> None:
        import dask.array as da_mod
        import zarr

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.spec = spec
        self._tmp_path = _staging_path(path, self)
        self._finalized = False
        self._schema: pa.Schema | None = None
        self._ev_grp: Any = None
        self._ev_cols_order: list[str] = []

        safe_attrs: dict[str, Any] = {
            k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in spec.attrs.items()
            if v is not None
        }
        safe_attrs.update(_common_meta_attrs(spec, "zarr"))

        try:
            if spec.has_grids:
                coords: dict[str, Any] = {}
                for d in spec.spatial_dims:
                    cvals = np.asarray(spec.spatial_coords[d])
                    coords[d] = np.asarray(to_str_list(cvals.tolist()), dtype=str) if is_str_dtype(cvals.dtype) else cvals
                if spec.has_clim:
                    coords["doy"] = np.arange(1, spec.year_len + 1, dtype=np.int32)
                    if spec.moving_years is not None:
                        coords["year"] = spec.moving_years.astype(np.int32)
                if spec.has_ts:
                    coords[spec.time_dim] = coerce_time_for_zarr(spec.det_time)

                units_str = str(spec.attrs.get("units", "")).strip()
                data_vars: dict[str, xr.DataArray] = {}
                for name, dims, shape, dt, fill in spec.grid_var_specs():
                    chunks = _compute_bounded_chunks(shape, dt.itemsize) if shape else ()
                    data_vars[name] = xr.DataArray(
                        da_mod.full(shape, fill, chunks=chunks, dtype=dt),
                        dims=dims,
                        name=name,
                        attrs=grid_var_attrs(name, units_str),
                    )
                xr.Dataset(data_vars, coords=coords, attrs=safe_attrs).to_zarr(
                    self._tmp_path, compute=False, mode="w", consolidated=False
                )
                self._group: Any = zarr.open_group(str(self._tmp_path), mode="r+")
            else:
                self._group = zarr.open_group(str(self._tmp_path), mode="w")
                for k, v in safe_attrs.items():
                    self._group.attrs[k] = v
        except BaseException:
            self.close()
            raise

    def _init_events_group(self, schema: pa.Schema) -> None:
        grp = self._group.create_group("events")
        self._ev_grp = grp
        self._ev_cols_order = []
        for fld in schema:
            name = fld.name
            if name == "event_type":
                dt: Any = np.int8
            elif name in DATE_EVENT_COLS:
                dt = np.int32
            elif pa.types.is_boolean(fld.type):
                dt = np.bool_
            elif pa.types.is_string(fld.type) or pa.types.is_large_string(fld.type):
                dt = str
            else:
                dt = fld.type.to_pandas_dtype()
            grp.create_array(name, shape=(0,), chunks=(4096,), dtype=dt)
            self._ev_cols_order.append(name)
        grp.attrs["event_columns"] = list(self._ev_cols_order)

    def write_grid_tile(self, name: str, tile: SpatialTile, data: np.ndarray, extra_ndims: int) -> None:
        arr = self._group[name]
        if self.spec.spatial_dims:
            idx = tuple(tile.isel_dict[d] for d in self.spec.spatial_dims) + (slice(None),) * extra_ndims
            arr[idx] = data
        elif extra_ndims == 0:
            arr[()] = float(np.asarray(data).reshape(()))
        else:
            arr[:] = data

    def write_empty_tile(self, tile: SpatialTile) -> None:
        # to_zarr(compute=False) uses Zarr's default 0 fill_value for int8; write CATEGORY_MISSING explicitly.
        if not self.spec.has_ts:
            return
        n_det = int(self.spec.det_time.shape[0])
        for name, enabled in (
            ("daily_category_ts", self.spec.write_daily_category_ts),
            ("event_category_ts", self.spec.write_event_category_ts),
        ):
            if enabled:
                miss = np.full((*tile.tile_shape, n_det), CATEGORY_MISSING, dtype=np.int8)
                self.write_grid_tile(name, tile, miss, 1)

    def write_events_table(self, tbl: pa.Table) -> None:
        if self._schema is None:
            self._schema = tbl.schema
        if not self.spec.write_events or tbl.num_rows == 0:
            return
        if self._ev_grp is None:
            self._init_events_group(tbl.schema)
        for fld in tbl.schema:
            vals = _encode_event_col_numpy(fld, tbl.column(fld.name), bool_dtype=np.bool_)
            self._ev_grp[fld.name].append(vals)

    def finalize(self, empty_schema: pa.Schema | None, grid_offsets: np.ndarray | None) -> None:
        if self.spec.write_events and grid_offsets is not None:
            schema = self._schema or empty_schema
            if self._ev_grp is None and schema is not None:
                self._init_events_group(schema)
            if self._ev_grp is not None:
                self._ev_grp.create_array("grid_offsets", data=np.asarray(grid_offsets, dtype=np.int64))
        self._ev_grp = None
        self._group = None
        _replace_directory(self._tmp_path, self.path)
        self._finalized = True

    def close(self) -> None:
        self._ev_grp = None
        self._group = None
        if not self._finalized:
            _cleanup_path(self._tmp_path)


def _read_zarr_events_group(
    grp: Any,
    *,
    start: int | None = None,
    stop: int | None = None,
    calendar: str | None = None,
    date_resolution: str | None = None,
) -> pa.Table:
    order = list(grp.attrs.get("event_columns", [k for k in grp.keys() if k != "grid_offsets"]))
    cols: dict[str, Any] = {}
    is_slice = start is not None and stop is not None
    length = (stop - start) if is_slice else None
    for name in order:
        arr_z = grp[name]
        if is_slice:
            assert start is not None and stop is not None and length is not None
            raw = np.asarray(arr_z[start:stop]) if length > 0 else np.array([], dtype=arr_z.dtype)
        else:
            raw = np.asarray(arr_z[:])
        cols[name] = decode_event_column(name, raw, calendar=calendar, date_resolution=date_resolution)
    return pa.table(cols)


class _ZarrAdapter:
    def create_writer(self, path: Path, spec: OutputSpec) -> _ZarrWriter:
        return _ZarrWriter(path, spec)

    def _read_events(
        self,
        path: Path,
        *,
        start: int | None = None,
        stop: int | None = None,
        calendar: str | None = None,
        date_resolution: str | None = None,
    ) -> pa.Table:
        import zarr

        root = zarr.open_group(str(path), mode="r")
        meta_json = root.attrs.get("marine_extremes_metadata_json", "{}")
        meta = json.loads(meta_json) if meta_json else {}
        cal = calendar if calendar is not None else meta.get("calendar")
        dres = date_resolution if date_resolution is not None else meta.get("date_resolution")
        return _read_zarr_events_group(root["events"], start=start, stop=stop, calendar=cal, date_resolution=dres)

    def open(self, path: Path, *, events_in_memory: bool) -> tuple[xr.Dataset | None, EventCatalog | None, dict[str, Any], list[Any]]:
        import zarr

        root = zarr.open_group(str(path), mode="r")
        meta_json = root.attrs.get("marine_extremes_metadata_json", "{}")
        metadata = json.loads(meta_json) if meta_json else {}
        cal = metadata.get("calendar")
        dres = metadata.get("date_resolution")
        spatial_dims = tuple(json.loads(root.attrs.get("marine_extremes_spatial_dims", "[]")))
        spatial_shape = tuple(json.loads(root.attrs.get("marine_extremes_spatial_shape", "[]")))
        saved_coords = (
            _deserialize_spatial_coords(json.loads(root.attrs["marine_extremes_spatial_coords_json"]))
            if "marine_extremes_spatial_coords_json" in root.attrs
            else {}
        )
        spatial_coords = {
            d: (np.asarray(root[d][:]) if d in root else saved_coords.get(d, np.arange(n, dtype=np.int64)))
            for d, n in zip(spatial_dims, spatial_shape)
        }
        has_grids = any(v in root for v in ALL_GRID_VAR_NAMES)

        catalog = None
        if "events" in root and "grid_offsets" in root["events"]:
            ev_grp = root["events"]
            grid_offsets = np.asarray(ev_grp["grid_offsets"][:], dtype=np.int64)
            if events_in_memory:
                store = MemoryEventStore(
                    _read_zarr_events_group(ev_grp, calendar=cal, date_resolution=dres),
                    calendar=cal,
                    date_resolution=dres,
                )
            else:
                store = DiskEventStore(
                    path,
                    slice_fn=lambda s, e: self._read_events(path, start=s, stop=e, calendar=cal, date_resolution=dres),
                    read_all_fn=lambda: self._read_events(path, calendar=cal, date_resolution=dres),
                    calendar=cal,
                    date_resolution=dres,
                )
            catalog = EventCatalog(None, grid_offsets, spatial_dims, spatial_shape, spatial_coords, store=store)

        grids_ds = xr.open_zarr(path, consolidated=False) if has_grids else None
        return grids_ds, catalog, metadata, ([grids_ds] if grids_ds is not None else [])


# ============================================================================
# Format Adapter 4: MATLAB v5 detectc cell layout
# ============================================================================


def _build_detectc_cell(
    table: pa.Table,
    grid_offsets: np.ndarray,
    spatial_dims: tuple[str, ...],
    spatial_shape: tuple[int, ...],
    *,
    extra_grid_bytes: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Pack CSR-sorted events into a MATLAB v5 object cell array mirroring m_mhw1.0 detectc.m."""
    n_events = table.num_rows
    n_pixels = int(grid_offsets.shape[0] - 1)
    ndim = len(spatial_shape)

    est_bytes = n_events * (10 + ndim) * 8 + n_pixels * 64 + extra_grid_bytes
    if est_bytes > 1_800_000_000:
        raise ValueError(
            f"Estimated MAT v5 size (~{est_bytes / 1e9:.2f} GB) approaches MATLAB v5's 2 GB limit. "
            "Use mat_layout='v73' (MATLAB v7.3 HDF5, no 2 GB limit) or '.nc' instead."
        )

    col_names = (
        ["mhw_onset", "mhw_end", "mhw_dur", "int_max", "int_mean", "int_std", "int_cum"]
        + [f"{d}_loc" for d in spatial_dims]
        + ["category", "event_type"]
    )
    if n_events > 0:
        onset = dates_to_yyyymmdd(table.column("date_start")).astype(np.float64)
        end = dates_to_yyyymmdd(table.column("date_end")).astype(np.float64)
        num_cols = [
            table.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
            for c in ("duration", "int_max", "int_mean", "int_std", "int_cum")
        ]
        locs = [
            table.column(f"{d}_idx").to_numpy(zero_copy_only=False).astype(np.float64) + 1.0
            for d in spatial_dims
        ]
        cat = table.column("category_max").to_numpy(zero_copy_only=False).astype(np.float64)
        ev_t = encode_event_type_codes(table.column("event_type")).astype(np.float64)
        full_mat = np.column_stack([onset, end, *num_cols, *locs, cat, ev_t])
    else:
        full_mat = np.empty((0, 9 + ndim), dtype=np.float64)

    mhw_cell = np.empty(spatial_shape if spatial_shape else (1, 1), dtype=object)
    empty_block = np.empty((0, 0), dtype=np.float64)
    flat_cell = mhw_cell.ravel()
    for p in range(n_pixels):
        s, e = int(grid_offsets[p]), int(grid_offsets[p + 1])
        flat_cell[p] = full_mat[s:e] if e > s else empty_block
    return mhw_cell, col_names


def _table_from_detectc_mat(
    mat: dict[str, Any],
    *,
    spatial_shape: tuple[int, ...] = (),
    calendar: str | None = None,
    date_resolution: str | None = None,
) -> pa.Table:
    col_names = [str(c.item()) if isinstance(c, np.ndarray) else str(c) for c in mat["MHW_columns"].ravel()]
    blocks = [b for b in mat["MHW"].ravel() if isinstance(b, np.ndarray) and b.size > 0]
    stacked = np.vstack(blocks) if blocks else np.empty((0, len(col_names)), dtype=np.float64)
    cols: dict[str, Any] = {}
    loc_dims: list[str] = []
    for i, cname in enumerate(col_names):
        v = stacked[:, i]
        if cname == "mhw_onset":
            cols["date_start"] = yyyymmdd_to_dates(v.astype(np.int64), calendar=calendar, date_resolution=date_resolution)
        elif cname == "mhw_end":
            cols["date_end"] = yyyymmdd_to_dates(v.astype(np.int64), calendar=calendar, date_resolution=date_resolution)
        elif cname == "mhw_dur":
            cols["duration"] = v.astype(np.int32)
        elif cname == "category":
            cols["category_max"] = v.astype(np.int8)
        elif cname == "event_type":
            cols["event_type"] = event_type_to_arrow_dict(v.astype(np.int8))
        elif cname.endswith("_loc"):
            dim = cname.removesuffix("_loc")
            loc_dims.append(dim)
            cols[f"{dim}_idx"] = v.astype(np.int32) - 1
        else:
            cols[cname] = v
    if stacked.shape[0] == 0:
        pixel_idx = np.array([], dtype=np.int64)
    elif spatial_shape and len(loc_dims) == len(spatial_shape):
        pixel_idx = np.ravel_multi_index(
            tuple(cols[f"{d}_idx"].astype(np.int64) for d in loc_dims), spatial_shape
        ).astype(np.int64)
    else:
        pixel_idx = np.zeros(stacked.shape[0], dtype=np.int64)
    return pa.table({"pixel_idx": pixel_idx, **cols})


class _MatDetectcWriter(_BufferedGridWriter):
    def __init__(self, path: Path, spec: OutputSpec) -> None:
        super().__init__(path, spec, max_bytes=1_800_000_000)

    def finalize(self, empty_schema: pa.Schema | None, grid_offsets: np.ndarray | None) -> None:
        import scipy.io as sio

        spec = self.spec
        payload: dict[str, Any] = {
            d: np.asarray(vals) for d, vals in spec.spatial_coords.items() if not is_str_dtype(vals.dtype)
        }
        if any(k in self._grids for k in CLIM_VAR_NAMES):
            payload["doy"] = np.arange(1, spec.year_len + 1, dtype=np.int32)
            if spec.moving_years is not None:
                payload["year"] = spec.moving_years.astype(np.int32)

        t_units = CF_TIME_UNITS
        t_cal = str(spec.attrs.get("calendar", "proleptic_gregorian"))
        if spec.det_time.size > 0 and any(k in self._grids for k in TS_VAR_NAMES):
            t_days, t_units, t_cal = time_to_cf_days(spec.det_time)
            payload[spec.time_dim] = t_days
            payload[f"{spec.time_dim}_yyyymmdd"] = dates_to_yyyymmdd(spec.det_time)

        rename = {
            "clim": "mclim",
            "thresh_high": "m90",
            "thresh_low": "m10",
            "intensity_ts": "mhw_ts",
        }
        for src_k, dst_k in rename.items():
            if src_k in self._grids:
                payload[dst_k] = self._grids[src_k]
        for extra_k in ("intensity_high_ts", "intensity_low_ts"):
            if extra_k in self._grids:
                payload[extra_k] = self._grids[extra_k]
        if "event_category_ts" in self._grids:
            payload["category_ts"] = self._grids["event_category_ts"]
            payload["event_category_ts"] = self._grids["event_category_ts"]
        elif "daily_category_ts" in self._grids:
            payload["category_ts"] = self._grids["daily_category_ts"]
        if "daily_category_ts" in self._grids:
            payload["daily_category_ts"] = self._grids["daily_category_ts"]
        for tname in TREND_VAR_NAMES:
            if tname in self._grids:
                payload[tname] = self._grids[tname]

        if spec.write_events and grid_offsets is not None:
            full_tbl = self._concat_events(empty_schema)
            mhw_cell, col_names = _build_detectc_cell(
                full_tbl, grid_offsets, spec.spatial_dims, spec.spatial_shape,
                extra_grid_bytes=sum(a.nbytes for a in self._grids.values()),
            )
            payload["MHW"] = mhw_cell
            payload["MHW_columns"] = np.array(col_names, dtype=object)
            payload["grid_offsets"] = grid_offsets

        meta_blob = {
            "format": "mat_detectc",
            "spatial_dims": list(spec.spatial_dims),
            "spatial_shape": list(spec.spatial_shape),
            "spatial_coords": _serialize_spatial_coords(spec.spatial_coords),
            "time_dim": spec.time_dim,
            "year_len": spec.year_len,
            "has_moving_years": spec.moving_years is not None,
            "time_units": t_units,
            "time_calendar": t_cal,
            "has_daily_category_ts": "daily_category_ts" in self._grids,
            "has_event_category_ts": "event_category_ts" in self._grids,
            "has_events": bool(spec.write_events and grid_offsets is not None),
            "metadata": spec.attrs,
        }
        payload["marine_extremes_meta_json"] = json.dumps(meta_blob, default=str)

        tmp_path = _staging_path(self.path, self)
        try:
            with open(tmp_path, "wb") as fh:
                sio.savemat(fh, payload, do_compression=True)
            os.replace(tmp_path, self.path)
        except BaseException:
            _cleanup_path(tmp_path)
            raise
        self.close()


class _MatDetectcAdapter:
    def create_writer(self, path: Path, spec: OutputSpec) -> _MatDetectcWriter:
        return _MatDetectcWriter(path, spec)

    def open(self, path: Path, *, events_in_memory: bool) -> tuple[xr.Dataset | None, EventCatalog | None, dict[str, Any], list[Any]]:
        import scipy.io as sio

        mat = sio.loadmat(str(path), squeeze_me=False)
        raw_mj = mat.get("marine_extremes_meta_json")
        blob = (
            json.loads(str(raw_mj.item()) if isinstance(raw_mj, np.ndarray) else str(raw_mj))
            if raw_mj is not None
            else {}
        )
        metadata = blob.get("metadata", {})
        cal = metadata.get("calendar") or blob.get("time_calendar")
        dres = metadata.get("date_resolution")
        spatial_dims = tuple(blob.get("spatial_dims", []))
        spatial_shape = tuple(blob.get("spatial_shape", []))
        time_dim = str(blob.get("time_dim", "time"))
        has_mov = bool(blob.get("has_moving_years", False))
        spatial_coords = _deserialize_spatial_coords(blob.get("spatial_coords", {}))
        for d, n in zip(spatial_dims, spatial_shape):
            if d not in spatial_coords:
                spatial_coords[d] = mat[d].reshape(-1) if d in mat else np.arange(n)

        coords: dict[str, Any] = dict(spatial_coords)
        if "doy" in mat:
            coords["doy"] = mat["doy"].reshape(-1)
        if has_mov and "year" in mat:
            coords["year"] = mat["year"].reshape(-1)
        if time_dim in mat:
            coords[time_dim] = cf_days_to_time(
                mat[time_dim],
                str(blob.get("time_units", CF_TIME_UNITS)),
                str(blob.get("time_calendar", "proleptic_gregorian")),
            )

        def _unwrap_v5(arr: np.ndarray, dims: tuple[str, ...]) -> xr.DataArray:
            a = np.asarray(arr)
            if len(dims) == 1 and a.ndim == 2 and a.shape[0] == 1:
                a = a.reshape(-1)
            elif len(dims) == 0:
                a = a.reshape(())
            return xr.DataArray(a, dims=dims)

        units_str = str(metadata.get("units", "")).strip()
        data_vars: dict[str, xr.DataArray] = {}
        clim_dims = (*spatial_dims, "year", "doy") if has_mov else (*spatial_dims, "doy")
        for mat_k, std_k in (("mclim", "clim"), ("m90", "thresh_high"), ("m10", "thresh_low")):
            if mat_k in mat:
                da_v = _unwrap_v5(mat[mat_k], clim_dims)
                da_v.attrs = grid_var_attrs(std_k, units_str)
                data_vars[std_k] = da_v
        ts_dims = (*spatial_dims, time_dim)
        if "mhw_ts" in mat:
            da_v = _unwrap_v5(mat["mhw_ts"], ts_dims)
            da_v.attrs = grid_var_attrs("intensity_ts", units_str)
            data_vars["intensity_ts"] = da_v
        for extra_k in ("intensity_high_ts", "intensity_low_ts"):
            if extra_k in mat:
                da_v = _unwrap_v5(mat[extra_k], ts_dims)
                da_v.attrs = grid_var_attrs(extra_k, units_str)
                data_vars[extra_k] = da_v
        if "daily_category_ts" in mat and blob.get("has_daily_category_ts", True):
            da_v = _unwrap_v5(mat["daily_category_ts"], ts_dims)
            da_v.attrs = grid_var_attrs("daily_category_ts", units_str)
            data_vars["daily_category_ts"] = da_v
        if "event_category_ts" in mat and blob.get("has_event_category_ts", True):
            da_v = _unwrap_v5(mat["event_category_ts"], ts_dims)
            da_v.attrs = grid_var_attrs("event_category_ts", units_str)
            data_vars["event_category_ts"] = da_v
        elif "category_ts" in mat and "daily_category_ts" not in data_vars and "event_category_ts" not in data_vars:
            da_v = _unwrap_v5(mat["category_ts"], ts_dims)
            da_v.attrs = grid_var_attrs("daily_category_ts", units_str)
            data_vars["daily_category_ts"] = da_v
        for trname in TREND_VAR_NAMES:
            if trname in mat:
                da_v = _unwrap_v5(mat[trname], spatial_dims)
                da_v.attrs = grid_var_attrs(trname, units_str)
                data_vars[trname] = da_v

        grids_ds = xr.Dataset(data_vars, coords=coords) if data_vars else None
        catalog = None
        if "MHW" in mat and "grid_offsets" in mat:
            tbl = _table_from_detectc_mat(mat, spatial_shape=spatial_shape, calendar=cal, date_resolution=dres)
            grid_offsets = np.asarray(mat["grid_offsets"], dtype=np.int64).reshape(-1)
            catalog = EventCatalog(
                tbl, grid_offsets, spatial_dims, spatial_shape, spatial_coords, calendar=cal, date_resolution=dres
            )
        return grids_ds, catalog, metadata, []


# ============================================================================
# Format Adapter 5: NumPy .npz (allow_pickle=False)
# ============================================================================


class _NpzWriter(_BufferedGridWriter):
    def finalize(self, empty_schema: pa.Schema | None, grid_offsets: np.ndarray | None) -> None:
        spec = self.spec
        payload: dict[str, Any] = dict(self._grids)
        for d, vals in spec.spatial_coords.items():
            carr = np.asarray(vals)
            payload[f"coord__{d}"] = np.asarray(to_str_list(carr.tolist()), dtype=str) if is_str_dtype(carr.dtype) else carr

        if any(k in self._grids for k in CLIM_VAR_NAMES):
            payload["doy"] = np.arange(1, spec.year_len + 1, dtype=np.int32)
            if spec.moving_years is not None:
                payload["year"] = spec.moving_years.astype(np.int32)

        t_units = CF_TIME_UNITS
        t_cal = str(spec.attrs.get("calendar", "proleptic_gregorian"))
        if spec.det_time.size > 0 and any(k in self._grids for k in TS_VAR_NAMES):
            t_days, t_units, t_cal = time_to_cf_days(spec.det_time)
            payload[f"coord__{spec.time_dim}"] = t_days

        event_cols_order: list[str] = []
        if spec.write_events and grid_offsets is not None:
            full_tbl = self._concat_events(empty_schema)
            payload["grid_offsets"] = grid_offsets
            for fld in full_tbl.schema:
                event_cols_order.append(fld.name)
                payload[f"event__{fld.name}"] = _encode_event_col_numpy(fld, full_tbl.column(fld.name), bool_dtype=np.bool_)

        meta_blob = {
            "format": "npz",
            "spatial_dims": list(spec.spatial_dims),
            "spatial_shape": list(spec.spatial_shape),
            "time_dim": spec.time_dim,
            "year_len": spec.year_len,
            "has_moving_years": spec.moving_years is not None,
            "time_units": t_units,
            "time_calendar": t_cal,
            "has_events": bool(spec.write_events and grid_offsets is not None),
            "event_columns": event_cols_order,
            "metadata": spec.attrs,
        }
        payload["marine_extremes_meta_json"] = np.frombuffer(json.dumps(meta_blob, default=str).encode("utf-8"), dtype=np.uint8)

        tmp_path = _staging_path(self.path, self)
        try:
            with open(tmp_path, "wb") as fh:
                np.savez_compressed(fh, **payload)
            os.replace(tmp_path, self.path)
        except BaseException:
            _cleanup_path(tmp_path)
            raise
        self.close()


class _NpzAdapter:
    def create_writer(self, path: Path, spec: OutputSpec) -> _NpzWriter:
        return _NpzWriter(path, spec)

    def open(self, path: Path, *, events_in_memory: bool) -> tuple[xr.Dataset | None, EventCatalog | None, dict[str, Any], list[Any]]:
        with np.load(path, allow_pickle=False) as npz:
            blob = json.loads(bytes(npz["marine_extremes_meta_json"].tobytes()).decode("utf-8"))
            metadata = blob.get("metadata", {})
            cal = metadata.get("calendar") or blob.get("time_calendar")
            dres = metadata.get("date_resolution")
            spatial_dims = tuple(blob.get("spatial_dims", []))
            spatial_shape = tuple(blob.get("spatial_shape", []))
            time_dim = str(blob.get("time_dim", "time"))
            has_mov = bool(blob.get("has_moving_years", False))
            spatial_coords = {d: npz[f"coord__{d}"] for d in spatial_dims if f"coord__{d}" in npz.files}

            coords: dict[str, Any] = dict(spatial_coords)
            if "doy" in npz.files:
                coords["doy"] = npz["doy"]
            if has_mov and "year" in npz.files:
                coords["year"] = npz["year"]
            if f"coord__{time_dim}" in npz.files:
                coords[time_dim] = cf_days_to_time(
                    npz[f"coord__{time_dim}"],
                    str(blob.get("time_units", CF_TIME_UNITS)),
                    str(blob.get("time_calendar", "proleptic_gregorian")),
                )

            units_str = str(metadata.get("units", "")).strip()
            clim_dims = (*spatial_dims, "year", "doy") if has_mov else (*spatial_dims, "doy")
            ts_dims = (*spatial_dims, time_dim)
            data_vars: dict[str, xr.DataArray] = {}
            for names, dims in ((CLIM_VAR_NAMES, clim_dims), (TS_VAR_NAMES, ts_dims), (TREND_VAR_NAMES, spatial_dims)):
                for vname in names:
                    if vname in npz.files:
                        arr = np.asarray(npz[vname])
                        if not dims:
                            arr = arr.reshape(())
                        data_vars[vname] = xr.DataArray(arr, dims=dims, name=vname, attrs=grid_var_attrs(vname, units_str))
            grids_ds = xr.Dataset(data_vars, coords=coords) if data_vars else None

            catalog = None
            if blob.get("has_events") and "grid_offsets" in npz.files:
                cols = {
                    cname: decode_event_column(cname, npz[f"event__{cname}"], calendar=cal, date_resolution=dres)
                    for cname in blob.get("event_columns", [])
                    if f"event__{cname}" in npz.files
                }
                catalog = EventCatalog(
                    pa.table(cols),
                    np.asarray(npz["grid_offsets"], dtype=np.int64),
                    spatial_dims,
                    spatial_shape,
                    spatial_coords,
                    calendar=cal,
                    date_resolution=dres,
                )
            return grids_ds, catalog, metadata, []


# ============================================================================
# Format Adapters 6-8: Tabular single files (Parquet, Feather, CSV)
# ============================================================================


class _TabularWriter:
    writes_grids: bool = False

    def __init__(self, path: Path, fmt: Literal["parquet", "feather", "csv"], spec: OutputSpec) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fmt = fmt
        self.spec = spec
        self._tmp_path = _staging_path(path, self)
        self._finalized = False
        self._schema: pa.Schema | None = None
        self._writer: Any = None
        self._sink: pa.OSFile | None = None
        self._base_meta = {
            "spatial_dims": list(spec.spatial_dims),
            "spatial_shape": list(spec.spatial_shape),
            "spatial_coords": _serialize_spatial_coords(spec.spatial_coords),
            "metadata": spec.attrs,
        }

    def _attach_schema_meta(self, tbl: pa.Table) -> pa.Table:
        existing = dict(tbl.schema.metadata or {})
        existing[b"marine_extremes_meta"] = json.dumps(self._base_meta, default=str).encode("utf-8")
        return tbl.replace_schema_metadata(existing)

    def write_grid_tile(self, name: str, tile: SpatialTile, data: np.ndarray, extra_ndims: int) -> None:
        return

    def write_empty_tile(self, tile: SpatialTile) -> None:
        return

    def write_events_table(self, tbl: pa.Table) -> None:
        if self._schema is None:
            self._schema = tbl.schema
        if tbl.num_rows == 0:
            return
        if self.fmt == "parquet":
            t_meta = self._attach_schema_meta(tbl)
            if self._writer is None:
                self._writer = pq.ParquetWriter(self._tmp_path, t_meta.schema, compression="zstd")
            self._writer.write_table(t_meta, row_group_size=65536)
        elif self.fmt == "feather":
            flat = self._attach_schema_meta(_flatten_dict_cols(tbl))
            if self._writer is None:
                self._sink = pa.OSFile(str(self._tmp_path), "wb")
                self._writer = pa_ipc.new_file(
                    self._sink, flat.schema, options=pa_ipc.IpcWriteOptions(compression="zstd")
                )
            self._writer.write_table(flat)
        else:
            flat = _flatten_dict_cols(tbl)
            if self._writer is None:
                self._sink = pa.OSFile(str(self._tmp_path), "wb")
                self._writer = pa_csv.CSVWriter(self._sink, flat.schema)
            self._writer.write_table(flat)

    def finalize(self, empty_schema: pa.Schema | None, grid_offsets: np.ndarray | None) -> None:
        schema = self._schema or empty_schema
        assert schema is not None
        if self.fmt == "parquet":
            if self._writer is None:
                t_empty = self._attach_schema_meta(schema.empty_table())
                self._writer = pq.ParquetWriter(self._tmp_path, t_empty.schema, compression="zstd")
            if grid_offsets is not None:
                self._writer.add_key_value_metadata({b"marine_extremes_csr": _encode_csr_counts(grid_offsets).encode("ascii")})
            self._writer.close()
            self._writer = None
        else:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            if self._sink is not None:
                self._sink.close()
                self._sink = None
            if not self._tmp_path.exists():
                empty_flat = _flatten_dict_cols(schema.empty_table())
                if self.fmt == "feather":
                    pa_feather.write_feather(self._attach_schema_meta(empty_flat), self._tmp_path, compression="zstd")
                else:
                    with pa.OSFile(str(self._tmp_path), "wb") as sink:
                        with pa_csv.CSVWriter(sink, empty_flat.schema) as w:
                            w.write_table(empty_flat)
        os.replace(self._tmp_path, self.path)
        self._finalized = True

    def close(self) -> None:
        for attr in ("_writer", "_sink"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        if not self._finalized:
            _cleanup_path(self._tmp_path)


class _ParquetAdapter:
    def create_writer(self, path: Path, spec: OutputSpec) -> _TabularWriter:
        return _TabularWriter(path, "parquet", spec)

    @staticmethod
    def _slice_events(path: Path, start: int, stop: int) -> pa.Table:
        pf = pq.ParquetFile(path)
        if stop <= start:
            return pf.schema_arrow.empty_table()
        rg_offset = 0
        parts: list[pa.Table] = []
        for rg_idx in range(pf.num_row_groups):
            rg_rows = pf.metadata.row_group(rg_idx).num_rows
            rg_end = rg_offset + rg_rows
            if rg_end > start and rg_offset < stop:
                rg_tbl = pf.read_row_group(rg_idx)
                s0 = max(0, start - rg_offset)
                s1 = min(rg_rows, stop - rg_offset)
                parts.append(rg_tbl.slice(s0, s1 - s0))
            rg_offset = rg_end
            if rg_offset >= stop:
                break
        return pa.concat_tables(parts) if parts else pf.schema_arrow.empty_table()

    def open(self, path: Path, *, events_in_memory: bool) -> tuple[xr.Dataset | None, EventCatalog | None, dict[str, Any], list[Any]]:
        pf = pq.ParquetFile(path)
        schema_meta = pf.schema_arrow.metadata or {}
        file_meta = pf.metadata.metadata or {}
        blob = json.loads(schema_meta[b"marine_extremes_meta"].decode("utf-8")) if b"marine_extremes_meta" in schema_meta else {}
        metadata = blob.get("metadata", {})
        cal, dres = metadata.get("calendar"), metadata.get("date_resolution")
        spatial_dims = tuple(blob.get("spatial_dims", []))
        spatial_shape = tuple(blob.get("spatial_shape", []))
        spatial_coords = _deserialize_spatial_coords(blob.get("spatial_coords", {}))
        n_pixels = int(math.prod(spatial_shape)) if spatial_shape else 1

        grid_offsets: np.ndarray | None = None
        if b"marine_extremes_csr" in file_meta:
            try:
                cand = _decode_csr_counts(file_meta[b"marine_extremes_csr"].decode("ascii"), max_bytes=(n_pixels + 1) * 16)
                if (
                    cand.size == n_pixels + 1
                    and cand[0] == 0
                    and np.all(np.diff(cand) >= 0)
                    and int(cand[-1]) == pf.metadata.num_rows
                ):
                    grid_offsets = cand
            except Exception:
                grid_offsets = None
        if grid_offsets is None:
            pix = pq.read_table(path, columns=["pixel_idx"]).column("pixel_idx").to_numpy(zero_copy_only=False)
            grid_offsets = _offsets_from_pixel_idx(pix, n_pixels)

        if events_in_memory:
            store: Any = MemoryEventStore(pq.read_table(path), calendar=cal, date_resolution=dres)
        else:
            store = DiskEventStore(
                path,
                slice_fn=lambda s, e: self._slice_events(path, s, e),
                read_all_fn=lambda: pq.read_table(path),
                calendar=cal,
                date_resolution=dres,
            )
        catalog = EventCatalog(None, grid_offsets, spatial_dims, spatial_shape, spatial_coords, store=store)
        return None, catalog, metadata, []


class _FeatherAdapter:
    def create_writer(self, path: Path, spec: OutputSpec) -> _TabularWriter:
        return _TabularWriter(path, "feather", spec)

    def open(self, path: Path, *, events_in_memory: bool) -> tuple[xr.Dataset | None, EventCatalog | None, dict[str, Any], list[Any]]:
        tbl = pa_feather.read_table(path)
        schema_meta = tbl.schema.metadata or {}
        blob = json.loads(schema_meta[b"marine_extremes_meta"].decode("utf-8")) if b"marine_extremes_meta" in schema_meta else {}
        metadata = blob.get("metadata", {})
        cal, dres = metadata.get("calendar"), metadata.get("date_resolution")
        spatial_dims = tuple(blob.get("spatial_dims", []))
        spatial_shape = tuple(blob.get("spatial_shape", []))
        spatial_coords = _deserialize_spatial_coords(blob.get("spatial_coords", {}))
        n_pixels = int(math.prod(spatial_shape)) if spatial_shape else 1
        if "pixel_idx" in tbl.column_names:
            pix = tbl.column("pixel_idx").to_numpy(zero_copy_only=False)
        elif spatial_dims and tbl.num_rows > 0 and all(f"{d}_idx" in tbl.column_names for d in spatial_dims):
            idx_tuple = tuple(tbl.column(f"{d}_idx").to_numpy(zero_copy_only=False).astype(np.int64) for d in spatial_dims)
            pix = np.ravel_multi_index(idx_tuple, spatial_shape).astype(np.int64)
        else:
            pix = np.zeros(tbl.num_rows, dtype=np.int64) if (not spatial_dims and tbl.num_rows > 0) else np.array([], dtype=np.int64)
        catalog = EventCatalog(
            tbl, _offsets_from_pixel_idx(pix, n_pixels), spatial_dims, spatial_shape, spatial_coords,
            calendar=cal, date_resolution=dres,
        )
        return None, catalog, metadata, []


class _CsvAdapter:
    def create_writer(self, path: Path, spec: OutputSpec) -> _TabularWriter:
        return _TabularWriter(path, "csv", spec)

    @staticmethod
    def read_events(
        path: Path,
        *,
        calendar: str | None = None,
        date_resolution: str | None = None,
    ) -> pa.Table:
        df = pd.read_csv(path)
        raw_dates: dict[str, pd.Series] = {}
        for col_name in list(df.columns):
            if col_name in CSV_STANDARD_DTYPES:
                df[col_name] = df[col_name].astype(CSV_STANDARD_DTYPES[col_name])
            elif col_name.endswith("_idx"):
                df[col_name] = df[col_name].astype(np.int32)
            elif col_name == "event_type":
                df[col_name] = pd.Categorical(df[col_name], categories=["low", "high"])
            elif col_name in DATE_EVENT_COLS:
                raw_dates[col_name] = df[col_name].copy()
                df[col_name] = parse_csv_date_series(df[col_name], calendar=calendar, date_resolution=date_resolution)
            elif len(df) == 0 and df[col_name].dtype == object:
                df[col_name] = df[col_name].astype("string")
        if calendar is None and date_resolution is None and raw_dates:
            if any(pd.api.types.is_string_dtype(df[c].dtype) for c in raw_dates):
                for c, orig in raw_dates.items():
                    df[c] = orig.astype("string")
        return pa.Table.from_pandas(df, preserve_index=False)

    def open(self, path: Path, *, events_in_memory: bool) -> tuple[xr.Dataset | None, EventCatalog | None, dict[str, Any], list[Any]]:
        tbl = self.read_events(path)
        idx_cols = [c for c in tbl.column_names if c.endswith("_idx") and c != "pixel_idx"]
        spatial_dims = tuple(c.removesuffix("_idx") for c in idx_cols)
        if spatial_dims and tbl.num_rows > 0:
            spatial_shape_list: list[int] = []
            spatial_coords: dict[str, np.ndarray] = {}
            for d in spatial_dims:
                idx_vals = tbl.column(f"{d}_idx").to_numpy(zero_copy_only=False).astype(np.int64)
                sz = int(idx_vals.max()) + 1
                spatial_shape_list.append(sz)
                if d in tbl.column_names:
                    cvals = tbl.column(d).to_numpy(zero_copy_only=False)
                    coord_arr = np.full(sz, np.nan, dtype=np.float64) if np.issubdtype(cvals.dtype, np.number) else np.empty(sz, dtype=object)
                    coord_arr[idx_vals] = cvals
                    spatial_coords[d] = coord_arr
                else:
                    spatial_coords[d] = np.arange(sz, dtype=np.int64)
            spatial_shape = tuple(spatial_shape_list)
        else:
            spatial_dims, spatial_shape, spatial_coords = (), (), {}

        n_pixels = int(math.prod(spatial_shape)) if spatial_shape else 1
        if spatial_shape and tbl.num_rows > 0:
            idx_tuple = tuple(tbl.column(f"{d}_idx").to_numpy(zero_copy_only=False).astype(np.int64) for d in spatial_dims)
            pix = np.ravel_multi_index(idx_tuple, spatial_shape)
        elif "pixel_idx" in tbl.column_names and tbl.num_rows > 0:
            pix = tbl.column("pixel_idx").to_numpy(zero_copy_only=False)
        elif not spatial_dims and tbl.num_rows > 0:
            pix = np.zeros(tbl.num_rows, dtype=np.int64)
        else:
            pix = np.array([], dtype=np.int64)

        if pix.size > 1 and np.any(pix[1:] < pix[:-1]):
            order = np.argsort(pix, kind="stable")
            tbl = tbl.take(pa.array(order))
            pix = pix[order]

        catalog = EventCatalog(tbl, _offsets_from_pixel_idx(pix, n_pixels), spatial_dims, spatial_shape, spatial_coords)
        return None, catalog, {}, []


_FORMAT_ADAPTERS: dict[OutFormat, Any] = {
    "netcdf": _NetCDFAdapter(),
    "mat": _MatV73Adapter(),
    "mat_detectc": _MatDetectcAdapter(),
    "zarr": _ZarrAdapter(),
    "npz": _NpzAdapter(),
    "parquet": _ParquetAdapter(),
    "feather": _FeatherAdapter(),
    "csv": _CsvAdapter(),
}


def open_file_writer(path: Path, fmt: OutFormat, spec: OutputSpec) -> _FileWriter:
    """Create a format-specific single-file writer from a compact OutputSpec."""
    if fmt not in _FORMAT_ADAPTERS:
        raise ValueError(f"Unsupported single-file format: {fmt}")
    return _FORMAT_ADAPTERS[fmt].create_writer(path, spec)


def open_single_file(
    path: str | Path,
    *,
    events_in_memory: bool = False,
) -> tuple[xr.Dataset | None, EventCatalog | None, dict[str, Any], list[Any]]:
    """Open any single-file ``marine-extremes`` output into ``(grids_ds, EventCatalog | None, metadata, closers)``."""
    p = Path(path)
    fmt = detect_file_format(p)
    return _FORMAT_ADAPTERS[fmt].open(p, events_in_memory=events_in_memory)


# ============================================================================
# ResultSink seam: unified tile output consumer (in-memory vs single-file)
# ============================================================================


class _BaseResultSink:
    """Shared tile-ingestion base accumulating global CSR grid_offsets across spatial tiles."""
    writes_grids: bool = True

    def __init__(self, spec: OutputSpec) -> None:
        self.spec = spec
        self._offsets = np.zeros(spec.total_pixels + 1, dtype=np.int64)
        self._running_events = 0
        self._empty_schema: pa.Schema | None = None

    def record_tile(
        self,
        tile: SpatialTile,
        active_local: np.ndarray,
        cal_det: CalendarInfo,
        *,
        raw_events: dict[str, np.ndarray] | None = None,
        clim: np.ndarray | None = None,
        thresh_high: np.ndarray | None = None,
        thresh_low: np.ndarray | None = None,
        intensity_ts: np.ndarray | None = None,
        intensity_high_ts: np.ndarray | None = None,
        intensity_low_ts: np.ndarray | None = None,
        daily_category_ts: np.ndarray | None = None,
        event_category_ts: np.ndarray | None = None,
        trend: np.ndarray | None = None,
    ) -> None:
        if active_local.size == 0:
            self._offsets[tile.flat_start : tile.flat_end + 1] = self._running_events
            self._on_empty_tile(tile)
            return

        assert raw_events is not None
        tile_offsets = raw_events["grid_offsets"]
        self._offsets[tile.flat_start : tile.flat_end + 1] = tile_offsets + self._running_events
        self._running_events += int(tile_offsets[-1])

        tile_tbl = build_arrow_table_from_core(
            raw_events,
            cal_det,
            self.spec.spatial_dims,
            self.spec.spatial_shape,
            self.spec.spatial_coords,
            pixel_offset_base=tile.flat_start,
        )
        if self._empty_schema is None:
            self._empty_schema = tile_tbl.schema
        self._on_events(tile_tbl)
        self._on_grids(
            tile,
            active_local,
            clim=clim,
            thresh_high=thresh_high,
            thresh_low=thresh_low,
            intensity_ts=intensity_ts,
            intensity_high_ts=intensity_high_ts,
            intensity_low_ts=intensity_low_ts,
            daily_category_ts=daily_category_ts,
            event_category_ts=event_category_ts,
            trend=trend,
        )

    def _resolve_empty_schema(self, cal_det: CalendarInfo) -> pa.Schema:
        if self._empty_schema is not None:
            return self._empty_schema
        return empty_event_table(
            cal_det, self.spec.spatial_dims, self.spec.spatial_shape, self.spec.spatial_coords
        ).schema

    def _on_empty_tile(self, tile: SpatialTile) -> None: ...
    def _on_events(self, tbl: pa.Table) -> None: ...
    def _on_grids(self, tile: SpatialTile, active_local: np.ndarray, **arrays: np.ndarray | None) -> None: ...
    def finalize(self, cal_det: CalendarInfo) -> DetectResult: ...
    def close(self) -> None: ...


class InMemoryResultSink(_BaseResultSink):
    """In-memory ResultSink adapter owning flat grid buffers and xarray DetectResult assembly."""

    writes_grids: bool = True

    def __init__(self, spec: OutputSpec) -> None:
        super().__init__(spec)
        n_pix = spec.total_pixels
        n_det = int(spec.det_time.shape[0])
        c_shape = (
            (n_pix, int(spec.moving_years.shape[0]), spec.year_len)
            if spec.moving_years is not None
            else (n_pix, spec.year_len)
        )
        self._events: list[pa.Table] = []
        self._clim = np.full(c_shape, np.nan, dtype=spec.clim_dtype) if spec.write_clim else None
        self._hi = np.full(c_shape, np.nan, dtype=spec.clim_dtype) if spec.write_thresh_high else None
        self._lo = np.full(c_shape, np.nan, dtype=spec.clim_dtype) if spec.write_thresh_low else None
        self._int = np.full((n_pix, n_det), np.nan, dtype=spec.intensity_dtype) if spec.write_intensity_ts else None
        self._int_hi = np.full((n_pix, n_det), np.nan, dtype=spec.intensity_dtype) if spec.write_intensity_high_ts else None
        self._int_lo = np.full((n_pix, n_det), np.nan, dtype=spec.intensity_dtype) if spec.write_intensity_low_ts else None
        self._dcat = np.full((n_pix, n_det), CATEGORY_MISSING, dtype=np.int8) if spec.write_daily_category_ts else None
        self._ecat = np.full((n_pix, n_det), CATEGORY_MISSING, dtype=np.int8) if spec.write_event_category_ts else None
        self._trend = np.full((n_pix, 3), np.nan, dtype=np.float64) if spec.write_trend else None

    def _on_empty_tile(self, tile: SpatialTile) -> None:
        return

    def _on_events(self, tbl: pa.Table) -> None:
        if tbl.num_rows > 0:
            self._events.append(tbl)

    def _on_grids(self, tile: SpatialTile, active_local: np.ndarray, **arrays: np.ndarray | None) -> None:
        act = active_local + tile.flat_start
        for buf, val in (
            (self._clim, arrays.get("clim")),
            (self._hi, arrays.get("thresh_high")),
            (self._lo, arrays.get("thresh_low")),
            (self._int, arrays.get("intensity_ts")),
            (self._int_hi, arrays.get("intensity_high_ts")),
            (self._int_lo, arrays.get("intensity_low_ts")),
            (self._dcat, arrays.get("daily_category_ts")),
            (self._ecat, arrays.get("event_category_ts")),
            (self._trend, arrays.get("trend")),
        ):
            if buf is not None and val is not None:
                buf[act] = val

    def finalize(self, cal_det: CalendarInfo) -> DetectResult:
        spec = self.spec
        schema = self._resolve_empty_schema(cal_det)
        full_table = pa.concat_tables(self._events) if self._events else schema.empty_table()
        catalog = EventCatalog(
            full_table,
            self._offsets,
            spec.spatial_dims,
            spec.spatial_shape,
            spec.spatial_coords,
            calendar=spec.attrs.get("calendar"),
            date_resolution=spec.attrs.get("date_resolution"),
        )
        sp_coords: dict[str, Any] = {d: spec.spatial_coords[d] for d in spec.spatial_dims}
        doy_coord = np.arange(1, spec.year_len + 1, dtype=np.int32)
        clim_coords = (
            {**sp_coords, "year": spec.moving_years, "doy": doy_coord}
            if spec.moving_years is not None
            else {**sp_coords, "doy": doy_coord}
        )
        ts_coords = {**sp_coords, spec.time_dim: spec.det_time}
        units_str = str(spec.attrs.get("units", "")).strip()

        def _da(buf: np.ndarray | None, name: str, dims: tuple[str, ...], shape: tuple[int, ...], coords: dict[str, Any]) -> xr.DataArray | None:
            return xr.DataArray(buf.reshape(shape), dims=dims, coords=coords, name=name, attrs=grid_var_attrs(name, units_str)) if buf is not None else None

        trend_ds = None
        if self._trend is not None:
            trend_ds = xr.Dataset(
                {
                    short: (spec.spatial_dims, self._trend[:, i].reshape(spec.spatial_shape), grid_var_attrs(full_name, units_str))
                    for i, (short, full_name) in enumerate(TREND_VAR_KEYS)
                },
                coords=sp_coords,
            )

        return DetectResult(
            events=catalog,
            clim=_da(self._clim, "clim", spec.clim_dims, spec.clim_shape, clim_coords),
            thresh_high=_da(self._hi, "thresh_high", spec.clim_dims, spec.clim_shape, clim_coords),
            thresh_low=_da(self._lo, "thresh_low", spec.clim_dims, spec.clim_shape, clim_coords),
            intensity_ts=_da(self._int, "intensity_ts", spec.ts_dims, spec.ts_shape, ts_coords),
            intensity_high_ts=_da(self._int_hi, "intensity_high_ts", spec.ts_dims, spec.ts_shape, ts_coords),
            intensity_low_ts=_da(self._int_lo, "intensity_low_ts", spec.ts_dims, spec.ts_shape, ts_coords),
            daily_category_ts=_da(self._dcat, "daily_category_ts", spec.ts_dims, spec.ts_shape, ts_coords),
            event_category_ts=_da(self._ecat, "event_category_ts", spec.ts_dims, spec.ts_shape, ts_coords),
            trend=trend_ds,
            metadata=dict(spec.attrs),
        )

    def close(self) -> None:
        return


class FileResultSink(_BaseResultSink):
    """Single-file ResultSink adapter delegating to a format writer with atomic staging."""

    def __init__(self, path: Path, fmt: OutFormat, spec: OutputSpec) -> None:
        super().__init__(spec)
        self.path = path
        self.fmt = fmt
        self._writer: _FileWriter | None = open_file_writer(path, fmt, spec)
        self.writes_grids = self._writer.writes_grids

    def _on_empty_tile(self, tile: SpatialTile) -> None:
        if self._writer is not None and self.writes_grids:
            self._writer.write_empty_tile(tile)

    def _on_events(self, tbl: pa.Table) -> None:
        assert self._writer is not None
        self._writer.write_events_table(tbl)

    def _on_grids(self, tile: SpatialTile, active_local: np.ndarray, **arrays: np.ndarray | None) -> None:
        if self._writer is None or not self.writes_grids:
            return
        spec = self.spec
        clim_extra_shape = (
            (int(spec.moving_years.shape[0]), spec.year_len) if spec.moving_years is not None else (spec.year_len,)
        )

        def _scatter_write(name: str, val: np.ndarray | None, enabled: bool, extra_shape: tuple[int, ...], dtype: Any, fill: Any) -> None:
            if not enabled or val is None:
                return
            full = np.full((tile.n_pixels, *extra_shape), fill, dtype=dtype)
            full[active_local] = val.astype(dtype, copy=False) if val.dtype != np.dtype(dtype) else val
            assert self._writer is not None
            self._writer.write_grid_tile(name, tile, full.reshape((*tile.tile_shape, *extra_shape)), len(extra_shape))
            del full

        _scatter_write("clim", arrays.pop("clim", None), spec.write_clim, clim_extra_shape, spec.clim_dtype, np.nan)
        _scatter_write("thresh_high", arrays.pop("thresh_high", None), spec.write_thresh_high, clim_extra_shape, spec.clim_dtype, np.nan)
        _scatter_write("thresh_low", arrays.pop("thresh_low", None), spec.write_thresh_low, clim_extra_shape, spec.clim_dtype, np.nan)
        n_det = int(spec.det_time.shape[0])
        _scatter_write("intensity_ts", arrays.pop("intensity_ts", None), spec.write_intensity_ts, (n_det,), spec.intensity_dtype, np.nan)
        _scatter_write("intensity_high_ts", arrays.pop("intensity_high_ts", None), spec.write_intensity_high_ts, (n_det,), spec.intensity_dtype, np.nan)
        _scatter_write("intensity_low_ts", arrays.pop("intensity_low_ts", None), spec.write_intensity_low_ts, (n_det,), spec.intensity_dtype, np.nan)
        _scatter_write("daily_category_ts", arrays.pop("daily_category_ts", None), spec.write_daily_category_ts, (n_det,), np.int8, CATEGORY_MISSING)
        _scatter_write("event_category_ts", arrays.pop("event_category_ts", None), spec.write_event_category_ts, (n_det,), np.int8, CATEGORY_MISSING)
        trend_arr = arrays.pop("trend", None)
        if spec.write_trend and trend_arr is not None:
            for ci, tname in enumerate(TREND_VAR_NAMES):
                _scatter_write(tname, trend_arr[:, ci], True, (), np.float64, np.nan)

    def finalize(self, cal_det: CalendarInfo) -> DetectResult:
        assert self._writer is not None
        schema = self._resolve_empty_schema(cal_det)
        try:
            self._writer.finalize(schema, self._offsets)
        finally:
            self.close()
        if self.fmt == "csv":
            # CSV files do not embed full spatial metadata/empty cells; preserve the run's full grid here.
            cal = self.spec.attrs.get("calendar")
            dres = self.spec.attrs.get("date_resolution")
            tbl_csv = _CsvAdapter.read_events(self.path, calendar=cal, date_resolution=dres)
            cat = EventCatalog(
                tbl_csv,
                self._offsets,
                self.spec.spatial_dims,
                self.spec.spatial_shape,
                self.spec.spatial_coords,
                calendar=cal,
                date_resolution=dres,
            )
            return DetectResult(events=cat, metadata=dict(self.spec.attrs), out_path=self.path)
        return DetectResult.open(self.path, events_in_memory=False)

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None


def create_result_sink(
    out_path: Path | None,
    resolved_fmt: OutFormat | None,
    spec: OutputSpec,
) -> InMemoryResultSink | FileResultSink:
    """Create either the in-memory or single-file ResultSink adapter."""
    if out_path is not None:
        assert resolved_fmt is not None
        return FileResultSink(out_path, resolved_fmt, spec)
    return InMemoryResultSink(spec)
