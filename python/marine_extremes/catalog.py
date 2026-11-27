"""Canonical event schema, EventStore seam, and CSR grid-indexed EventCatalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

from marine_extremes.calendar import CalendarInfo, format_date_column, yyyymmdd_to_dates

# ---------------------------------------------------------------------------
# Single authoritative event schema & column codecs
# ---------------------------------------------------------------------------

BOOL_EVENT_COLS: frozenset[str] = frozenset({"truncated_start", "truncated_end", "near_lower_bound"})
DATE_EVENT_COLS: frozenset[str] = frozenset({"date_start", "date_peak", "date_end"})

# Canonical ordered non-spatial, non-date, non-event_type columns emitted by the Rust engine.
EVENT_FIXED_COLS: tuple[tuple[str, Any], ...] = (
    ("index_start", np.int32),
    ("index_peak", np.int32),
    ("index_end", np.int32),
    ("duration", np.int32),
    ("int_max", np.float64),
    ("int_mean", np.float64),
    ("int_std", np.float64),
    ("int_var", np.float64),
    ("int_cum", np.float64),
    ("int_max_rel_thresh", np.float64),
    ("int_mean_rel_thresh", np.float64),
    ("int_cum_rel_thresh", np.float64),
    ("rate_onset", np.float64),
    ("rate_decline", np.float64),
    ("category_max", np.int8),
    ("p_moderate", np.float64),
    ("p_strong", np.float64),
    ("p_severe", np.float64),
    ("p_extreme", np.float64),
    ("truncated_start", bool),
    ("truncated_end", bool),
    ("near_lower_bound", bool),
)

CSV_STANDARD_DTYPES: dict[str, Any] = {"pixel_idx": np.int64, **dict(EVENT_FIXED_COLS)}


def is_str_dtype(dt: Any) -> bool:
    """Return True for Unicode, byte-string, object, or NumPy 2 StringDType ('T') dtypes."""
    return np.dtype(dt).kind in ("U", "S", "O", "T")


def to_str_list(vals: Any) -> list[str]:
    """Decode arbitrary string/bytes/None sequence elements into plain Python UTF-8 strings."""
    out: list[str] = []
    for x in vals:
        if x is None:
            out.append("")
        elif isinstance(x, (bytes, bytearray)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def event_type_to_arrow_dict(codes: np.ndarray) -> pa.DictionaryArray:
    """Convert +1/-1 int8 event_type codes into the canonical DictionaryArray schema."""
    if codes.size > 0:
        return pa.DictionaryArray.from_arrays(
            pa.array((codes > 0).astype(np.int8, copy=False)),
            pa.array(["low", "high"]),
        )
    return pa.DictionaryArray.from_arrays(
        pa.array([], type=pa.int8()),
        pa.array(["low", "high"]),
    )


def encode_event_type_codes(col: Any) -> np.ndarray:
    """Convert Arrow/NumPy 'high'/'low' event_type column to +1/-1 int8 codes."""
    if len(col) == 0:
        return np.array([], dtype=np.int8)
    arr = col.to_numpy(zero_copy_only=False) if isinstance(col, (pa.Array, pa.ChunkedArray)) else np.asarray(col)
    return np.where(arr == "high", 1, -1).astype(np.int8)


def decode_event_column(
    name: str,
    arr: np.ndarray,
    *,
    calendar: str | None = None,
    date_resolution: str | None = None,
) -> Any:
    """Decode one persisted event column back into its canonical Arrow/NumPy representation."""
    if name == "event_type":
        return event_type_to_arrow_dict(np.asarray(arr, dtype=np.int8))
    if name in DATE_EVENT_COLS:
        return yyyymmdd_to_dates(arr, calendar=calendar, date_resolution=date_resolution)
    if name in BOOL_EVENT_COLS:
        return np.asarray(arr, dtype=bool)
    a = np.asarray(arr)
    if a.dtype.kind == "T":
        return pa.array(a.tolist(), type=pa.string())
    if a.dtype.kind == "S":
        return pa.array(to_str_list(a.tolist()), type=pa.string())
    if a.size == 0 and a.dtype.kind in ("U", "O"):
        return pa.array([], type=pa.string())
    return a


def _add_spatial_event_cols(
    cols: dict[str, Any],
    pixel_idx: np.ndarray,
    spatial_dims: tuple[str, ...],
    spatial_shape: tuple[int, ...],
    spatial_coords: dict[str, np.ndarray],
) -> None:
    """Unravel flat pixel indices into per-dimension integer indices and physical coordinate values."""
    if not spatial_shape:
        return
    n_ev = pixel_idx.shape[0]
    unraveled = (
        np.unravel_index(pixel_idx, spatial_shape)
        if n_ev > 0
        else tuple(np.array([], dtype=np.int64) for _ in spatial_shape)
    )
    for dim, idx_arr in zip(spatial_dims, unraveled):
        cols[f"{dim}_idx"] = idx_arr.astype(np.int32, copy=False)
        coord_vals = spatial_coords[dim]
        if is_str_dtype(coord_vals.dtype):
            vals = to_str_list(coord_vals[idx_arr]) if n_ev > 0 else []
            cols[dim] = pa.array(vals, type=pa.string())
        else:
            cols[dim] = coord_vals[idx_arr] if n_ev > 0 else np.array([], dtype=coord_vals.dtype)


def build_arrow_table_from_core(
    raw_events: dict[str, np.ndarray],
    cal: CalendarInfo,
    spatial_dims: tuple[str, ...],
    spatial_shape: tuple[int, ...],
    spatial_coords: dict[str, np.ndarray],
    pixel_offset_base: int = 0,
) -> pa.Table:
    """Convert Rust SoA output dict into a richly typed PyArrow Table with coordinates and dates."""
    pixel_idx = raw_events["pixel_idx"].astype(np.int64, copy=False) + int(pixel_offset_base)
    cols: dict[str, Any] = {
        "pixel_idx": pixel_idx,
        "event_type": event_type_to_arrow_dict(raw_events["event_type"]),
    }
    _add_spatial_event_cols(cols, pixel_idx, spatial_dims, spatial_shape, spatial_coords)
    cols["date_start"] = format_date_column(cal.raw_time, raw_events["index_start"], cal.is_cftime)
    cols["date_peak"] = format_date_column(cal.raw_time, raw_events["index_peak"], cal.is_cftime)
    cols["date_end"] = format_date_column(cal.raw_time, raw_events["index_end"], cal.is_cftime)
    for name, _ in EVENT_FIXED_COLS:
        cols[name] = raw_events[name]
    return pa.table(cols)


def empty_event_table(
    cal: CalendarInfo,
    spatial_dims: tuple[str, ...],
    spatial_shape: tuple[int, ...],
    spatial_coords: dict[str, np.ndarray],
) -> pa.Table:
    """Synthesize a zero-row event table with the exact canonical schema."""
    empty_idx = np.array([], dtype=np.int32)
    raw: dict[str, np.ndarray] = {
        "pixel_idx": np.array([], dtype=np.int64),
        "event_type": np.array([], dtype=np.int8),
        **{name: (empty_idx if dt is np.int32 else np.array([], dtype=dt)) for name, dt in EVENT_FIXED_COLS},
    }
    return build_arrow_table_from_core(raw, cal, spatial_dims, spatial_shape, spatial_coords)


# ---------------------------------------------------------------------------
# EventStore seam: two concrete adapters (in-memory Arrow vs disk-backed)
# ---------------------------------------------------------------------------


class EventStore(Protocol):
    """Internal storage seam behind EventCatalog."""

    @property
    def is_disk_backed(self) -> bool: ...

    @property
    def storage_label(self) -> str: ...

    @property
    def calendar(self) -> str | None: ...

    @property
    def date_resolution(self) -> str | None: ...

    def slice_rows(self, start: int, stop: int) -> pa.Table: ...

    def to_arrow(self) -> pa.Table: ...

    def iter_chunks(self, n_rows: int, chunk_rows: int = 200_000) -> Iterator[pa.Table]: ...


class MemoryEventStore:
    """In-memory Arrow-table adapter for EventStore."""

    def __init__(
        self,
        table: pa.Table,
        *,
        calendar: str | None = None,
        date_resolution: str | None = None,
    ) -> None:
        self._table = table
        self._calendar = calendar
        self._date_resolution = date_resolution

    @property
    def is_disk_backed(self) -> bool:
        return False

    @property
    def storage_label(self) -> str:
        return "in-memory"

    @property
    def calendar(self) -> str | None:
        return self._calendar

    @property
    def date_resolution(self) -> str | None:
        return self._date_resolution

    def slice_rows(self, start: int, stop: int) -> pa.Table:
        length = stop - start
        if length <= 0:
            return self._table.slice(0, 0)
        return self._table.slice(start, length)

    def to_arrow(self) -> pa.Table:
        return self._table

    def iter_chunks(self, n_rows: int, chunk_rows: int = 200_000) -> Iterator[pa.Table]:
        yield self._table


class DiskEventStore:
    """Disk-backed lazy-slicing adapter for EventStore (zero dependency on format internals)."""

    def __init__(
        self,
        path: Path,
        *,
        slice_fn: Callable[[int, int], pa.Table],
        read_all_fn: Callable[[], pa.Table],
        calendar: str | None = None,
        date_resolution: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._slice_fn = slice_fn
        self._read_all_fn = read_all_fn
        self._calendar = calendar
        self._date_resolution = date_resolution

    @property
    def is_disk_backed(self) -> bool:
        return True

    @property
    def storage_label(self) -> str:
        return f"disk:{self._path}"

    @property
    def calendar(self) -> str | None:
        return self._calendar

    @property
    def date_resolution(self) -> str | None:
        return self._date_resolution

    def slice_rows(self, start: int, stop: int) -> pa.Table:
        if stop <= start:
            return self._slice_fn(0, 0)
        return self._slice_fn(start, stop)

    def to_arrow(self) -> pa.Table:
        return self._read_all_fn()

    def iter_chunks(self, n_rows: int, chunk_rows: int = 200_000) -> Iterator[pa.Table]:
        if n_rows <= 0:
            yield self._slice_fn(0, 0)
            return
        for s0 in range(0, n_rows, chunk_rows):
            yield self._slice_fn(s0, min(s0 + chunk_rows, n_rows))


# ---------------------------------------------------------------------------
# Deep CSR-indexed EventCatalog
# ---------------------------------------------------------------------------


class EventCatalog:
    """Grid-indexed columnar catalog of detected marine extreme events.

    Backed by an ``EventStore`` (either in-memory Arrow or lazy single-file disk storage)
    sorted by flattened spatial pixel index, paired with an ``int64`` CSR offset array
    ``grid_offsets`` of length ``n_pixels + 1``.
    """

    def __init__(
        self,
        table: pa.Table | None,
        grid_offsets: np.ndarray,
        spatial_dims: tuple[str, ...],
        spatial_shape: tuple[int, ...],
        spatial_coords: dict[str, np.ndarray],
        *,
        store: EventStore | None = None,
        calendar: str | None = None,
        date_resolution: str | None = None,
    ) -> None:
        if store is not None:
            self._store: EventStore = store
        elif table is not None:
            self._store = MemoryEventStore(table, calendar=calendar, date_resolution=date_resolution)
        else:
            raise ValueError("EventCatalog requires either `table` or `store`.")

        self.grid_offsets = np.asarray(grid_offsets, dtype=np.int64)
        self.spatial_dims = tuple(spatial_dims)
        self.spatial_shape = tuple(spatial_shape)
        self.spatial_coords = {k: np.asarray(v) for k, v in spatial_coords.items()}

        expected_pixels = int(np.prod(self.spatial_shape)) if self.spatial_shape else 1
        if self.grid_offsets.shape != (expected_pixels + 1,):
            raise ValueError(
                f"grid_offsets length {self.grid_offsets.shape[0]} does not match "
                f"expected grid size {expected_pixels} + 1"
            )

    @property
    def n_pixels(self) -> int:
        return int(self.grid_offsets.shape[0] - 1)

    @property
    def is_disk_backed(self) -> bool:
        return self._store.is_disk_backed

    def __len__(self) -> int:
        return int(self.grid_offsets[-1])

    def __repr__(self) -> str:
        shape_str = (
            ", ".join(f"{d}={n}" for d, n in zip(self.spatial_dims, self.spatial_shape))
            if self.spatial_dims
            else "1D"
        )
        return f"EventCatalog(n_events={len(self):,}, grid=({shape_str}), storage={self._store.storage_label})"

    def iter_chunks(self, chunk_rows: int = 200_000) -> Iterator[pa.Table]:
        """Yield Arrow table chunks without loading disk-backed catalogs fully into memory."""
        return self._store.iter_chunks(len(self), chunk_rows=chunk_rows)

    def _slice_rows(self, start: int, stop: int) -> pd.DataFrame:
        return self._store.slice_rows(start, stop).to_pandas()

    def isel(self, **indexers: int) -> pd.DataFrame:
        """O(1) event lookup for a single grid cell by integer indices via CSR grid_offsets."""
        if not self.spatial_dims:
            if indexers:
                raise ValueError(f"1D catalog takes no spatial indexers, got {list(indexers)}")
            return self._slice_rows(int(self.grid_offsets[0]), int(self.grid_offsets[1]))

        missing = [d for d in self.spatial_dims if d not in indexers]
        extra = [d for d in indexers if d not in self.spatial_dims]
        if missing or extra:
            raise KeyError(
                f"Expected indexers for exact spatial dims {self.spatial_dims}; "
                f"missing={missing}, unexpected={extra}"
            )
        idx_tuple = []
        for dim, size in zip(self.spatial_dims, self.spatial_shape):
            i = int(indexers[dim])
            if i < 0:
                i += size
            if i < 0 or i >= size:
                raise IndexError(f"Index {indexers[dim]} out of bounds for dim '{dim}' (size {size})")
            idx_tuple.append(i)
        flat = int(np.ravel_multi_index(tuple(idx_tuple), self.spatial_shape))
        return self._slice_rows(int(self.grid_offsets[flat]), int(self.grid_offsets[flat + 1]))

    def sel(self, method: str = "nearest", **coords: Any) -> pd.DataFrame:
        """O(1) event lookup for a single grid cell by physical coordinates."""
        if not self.spatial_dims:
            if coords:
                raise ValueError(f"1D catalog takes no coordinate selectors, got {list(coords)}")
            return self.isel()
        if method not in ("nearest", "exact"):
            raise ValueError("method must be 'nearest' or 'exact'")

        missing = [d for d in self.spatial_dims if d not in coords]
        extra = [d for d in coords if d not in self.spatial_dims]
        if missing or extra:
            raise KeyError(
                f"Expected coordinates for exact spatial dims {self.spatial_dims}; "
                f"missing={missing}, unexpected={extra}"
            )
        indexers: dict[str, int] = {}
        for dim in self.spatial_dims:
            arr = self.spatial_coords[dim]
            target = coords[dim]
            is_str_coord = is_str_dtype(arr.dtype)
            if method == "exact" or is_str_coord:
                if is_str_coord:
                    target_s = target.decode("utf-8") if isinstance(target, (bytes, bytearray)) else str(target)
                    matches = np.flatnonzero(
                        [
                            x is not None
                            and (x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)) == target_s
                            for x in arr.tolist()
                        ]
                    )
                else:
                    matches = np.where(arr == target)[0]
                if matches.size == 0:
                    raise KeyError(f"Value {target!r} not found in coordinate '{dim}'")
                indexers[dim] = int(matches[0])
            else:
                diffs = np.abs(arr.astype(np.float64) - float(target))
                if not np.any(np.isfinite(diffs)):
                    raise KeyError(f"Coordinate '{dim}' has no finite values to match {target!r}")
                indexers[dim] = int(np.nanargmin(diffs))
        return self.isel(**indexers)

    def __getitem__(self, item: int | tuple[int, ...]) -> pd.DataFrame:
        """Positional grid indexing mirroring a MATLAB cell array (``MHW{i, j}``)."""
        if not self.spatial_dims:
            return self.isel()
        if not isinstance(item, tuple):
            item = (item,)
        if len(item) != len(self.spatial_dims):
            raise IndexError(
                f"Expected {len(self.spatial_dims)} indices for dims {self.spatial_dims}, got {len(item)}"
            )
        return self.isel(**dict(zip(self.spatial_dims, item)))

    def count_map(self) -> xr.DataArray:
        """Instant spatial map of total detected event counts via CSR offsets (no table scan)."""
        counts = np.diff(self.grid_offsets)
        if not self.spatial_dims:
            return xr.DataArray(int(counts[0]), name="event_count")
        return xr.DataArray(
            counts.reshape(self.spatial_shape),
            dims=self.spatial_dims,
            coords={d: self.spatial_coords[d] for d in self.spatial_dims},
            name="event_count",
        )

    def to_arrow(self) -> pa.Table:
        """Return the full catalog as a PyArrow Table."""
        return self._store.to_arrow()

    def to_pandas(self) -> pd.DataFrame:
        """Return the full catalog as a Pandas DataFrame."""
        return self.to_arrow().to_pandas()

    def filter(
        self,
        *,
        event_type: str | None = None,
        min_duration: int | None = None,
        min_category: int | None = None,
        exclude_truncated: bool = False,
        exclude_near_lower_bound: bool = False,
    ) -> pd.DataFrame:
        """Convenience columnar filter returning a Pandas DataFrame."""
        df = self.to_pandas()
        if event_type is not None:
            et = {"mhw": "high", "mcs": "low"}.get(event_type.lower(), event_type.lower())
            df = df[df["event_type"] == et]
        if min_duration is not None:
            df = df[df["duration"] >= min_duration]
        if min_category is not None:
            df = df[df["category_max"] >= min_category]
        if exclude_truncated:
            if "truncated_start" not in df.columns or "truncated_end" not in df.columns:
                raise KeyError(
                    "Catalog does not contain truncated_start/truncated_end columns "
                    "(not stored by MATLAB v5 detectc layout)."
                )
            df = df[~df["truncated_start"] & ~df["truncated_end"]]
        if exclude_near_lower_bound:
            if "near_lower_bound" not in df.columns:
                raise KeyError(
                    "Catalog does not contain near_lower_bound column "
                    "(not stored by MATLAB v5 detectc layout)."
                )
            df = df[~df["near_lower_bound"]]
        return df

    def save(
        self,
        path: str | Path,
        *,
        fmt: str | None = None,
        mat_layout: str = "v73",
    ) -> Path:
        """Save this event catalog as ONE single self-contained file."""
        from marine_extremes.io_formats import OutputSpec, open_file_writer, resolve_out_format

        out_path = Path(path)
        out_fmt = resolve_out_format(out_path, out_format=fmt, mat_layout=mat_layout)
        attrs: dict[str, Any] = {"out_format": out_fmt}
        if self._store.calendar:
            attrs["calendar"] = self._store.calendar
        if self._store.date_resolution:
            attrs["date_resolution"] = self._store.date_resolution
        spec = OutputSpec.for_events_only(
            self.spatial_dims,
            self.spatial_shape,
            self.spatial_coords,
            attrs=attrs,
        )
        writer = open_file_writer(out_path, out_fmt, spec)
        ev_schema: pa.Schema | None = None
        try:
            for part in self.iter_chunks():
                if ev_schema is None:
                    ev_schema = part.schema
                writer.write_events_table(part)
            writer.finalize(ev_schema, self.grid_offsets)
        finally:
            writer.close()
        return out_path

    def to_mat(self, path: str | Path, layout: str = "v73") -> Path:
        """Export events to a single standalone MATLAB ``.mat`` file (``'v73'`` default or ``'detectc'`` v5)."""
        return self.save(path, fmt="mat", mat_layout=layout)

    @classmethod
    def open(cls, path: str | Path, in_memory: bool = False) -> "EventCatalog":
        """Open a single-file EventCatalog (``.parquet``, ``.feather``, ``.csv``, ``.nc``, ``.mat``, ``.zarr``, ``.npz``)."""
        from marine_extremes.io_formats import open_single_file

        _, catalog, _, closers = open_single_file(path, events_in_memory=in_memory)
        try:
            if catalog is None:
                raise ValueError(f"File '{path}' does not contain an event catalog.")
            return catalog
        finally:
            for c in closers:
                try:
                    c.close()
                except Exception:
                    pass
