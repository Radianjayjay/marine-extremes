"""Memory-bounded spatial/depth tiling and multi-file lazy input normalization."""

from __future__ import annotations

import glob as glob_mod
import itertools
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from marine_extremes.cf import resolve_time_dim, select_dataset_var

_MEM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)?\s*$", re.IGNORECASE)
_MEM_UNITS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1000,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1000**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1000**3,
    "gib": 1024**3,
    "t": 1024**4,
    "tb": 1000**4,
    "tib": 1024**4,
}


def parse_memory_limit(memory_limit: str | int | float) -> int:
    """Parse a human-readable memory budget (e.g., ``'4GB'``, ``'512MB'``) into bytes."""
    if isinstance(memory_limit, (int, float)):
        val = int(memory_limit)
        if val <= 0:
            raise ValueError("memory_limit must be positive")
        return val
    m = _MEM_RE.match(str(memory_limit))
    if not m:
        raise ValueError(f"Invalid memory_limit specification: {memory_limit!r}")
    num = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit not in _MEM_UNITS:
        raise ValueError(f"Unsupported memory unit '{unit}' in {memory_limit!r}")
    out = int(num * _MEM_UNITS[unit])
    if out <= 0:
        raise ValueError("memory_limit must be positive")
    return out


@dataclass
class OpenedInput:
    """Normalized xarray.DataArray along with any file handles to close after execution."""

    da: xr.DataArray
    time_dim: str
    spatial_dims: tuple[str, ...]
    spatial_shape: tuple[int, ...]
    spatial_coords: dict[str, np.ndarray]
    closers: list[Any]

    def close(self) -> None:
        """Close any underlying file/Dataset handles opened for this input."""
        for obj in self.closers:
            try:
                obj.close()
            except Exception:
                pass
        self.closers.clear()


def _squeeze_singleton_dims(
    da: xr.DataArray,
    resolved_time_dim: Any,
    squeeze_singleton: bool | str | Sequence[str],
) -> xr.DataArray:
    """Drop requested length-1 non-time dimensions from a normalized DataArray."""
    if squeeze_singleton is True:
        candidates: Sequence[str] = [d for d in da.dims if d != resolved_time_dim and da.sizes[d] == 1]
    elif isinstance(squeeze_singleton, str):
        candidates = (squeeze_singleton,)
    elif isinstance(squeeze_singleton, Sequence) and not isinstance(squeeze_singleton, (bytes, bytearray)):
        candidates = squeeze_singleton
    else:
        return da

    to_drop = []
    for d in candidates:
        if d in da.dims and d != resolved_time_dim:
            if da.sizes[d] != 1:
                raise ValueError(f"Cannot squeeze dim '{d}' of size {da.sizes[d]} != 1")
            to_drop.append(d)
    return da.squeeze(to_drop, drop=True) if to_drop else da


def resolve_input_paths(path: str | Path) -> tuple[list[Path], bool]:
    """Resolve an existing file/Zarr store or glob pattern to concrete path(s)."""
    p_str = str(path)
    if not p_str.strip():
        raise ValueError("Input path must be a non-empty string.")
    p = Path(p_str)
    if p.exists():
        return [p], False
    if any(ch in p_str for ch in ("*", "?", "[")):
        matched = sorted(glob_mod.glob(p_str))
        if not matched:
            raise FileNotFoundError(f"No files matched glob pattern: {p_str!r}")
        return [Path(m) for m in matched], True
    raise FileNotFoundError(f"Input path does not exist: {p}")


def open_path_dataset(data: str | Path | Sequence[str | Path]) -> tuple[xr.Dataset, list[Path]]:
    """Open a file, Zarr dir, glob, or path sequence lazily without loading arrays."""
    if isinstance(data, (str, Path)):
        paths, is_multi = resolve_input_paths(data)
    else:
        paths = [Path(p) for p in data]
        is_multi = True
    if is_multi:
        mf_kw: dict[str, Any] = {"combine": "by_coords"}
        if all(q.is_dir() or q.suffix == ".zarr" for q in paths):
            mf_kw.update(engine="zarr", consolidated=False)
        return xr.open_mfdataset([str(q) for q in paths], **mf_kw), paths
    p = paths[0]
    ds = xr.open_zarr(p, consolidated=False) if (p.is_dir() or p.suffix == ".zarr") else xr.open_dataset(p)
    return ds, paths


def open_data_input(
    data: Any,
    *,
    var: str | None = None,
    time: Any | None = None,
    time_dim: str | None = "auto",
    squeeze_singleton: bool | str | Sequence[str] = False,
) -> OpenedInput:
    """Normalize 1D/3D/4D/N-D inputs (ndarray, Series, DataArray, Dataset, paths, globs)."""
    closers: list[Any] = []
    synth_time_dim = "time" if (time_dim is None or time_dim == "auto") else str(time_dim)
    try:
        if isinstance(data, xr.DataArray):
            da = data
        elif isinstance(data, xr.Dataset):
            da = select_dataset_var(data, var, time_dim=time_dim)
        elif isinstance(data, pd.Series):
            t_vals = time if time is not None else data.index
            da = xr.DataArray(
                data.to_numpy(dtype=np.float64),
                dims=(synth_time_dim,),
                coords={synth_time_dim: t_vals},
                name=data.name or var or "value",
            )
        elif isinstance(data, Sequence) and not isinstance(data, (bytes, bytearray)) and len(data) == 0:
            raise ValueError("Input data sequence is empty.")
        elif isinstance(data, (str, Path)) or (
            isinstance(data, Sequence)
            and not isinstance(data, (bytes, bytearray))
            and isinstance(data[0], (str, Path))
        ):
            ds, _ = open_path_dataset(data)
            closers.append(ds)
            da = select_dataset_var(ds, var, time_dim=time_dim)
        else:
            arr = np.asarray(data)
            if arr.ndim == 0:
                raise ValueError("Input array must be at least 1D (got 0D scalar).")
            if time is None:
                raise ValueError("Passing a raw numpy array requires explicit `time=...`.")
            t_arr = np.asarray(time)
            if t_arr.ndim != 1:
                raise ValueError("`time` must be a 1D array.")
            n_t = t_arr.shape[0]
            t_coord: Any = (synth_time_dim, t_arr, dict(time.attrs)) if isinstance(time, xr.DataArray) else t_arr
            if arr.ndim == 1:
                if arr.shape[0] != n_t:
                    raise ValueError(f"1D array length {arr.shape[0]} != len(time) {n_t}")
                da = xr.DataArray(arr, dims=(synth_time_dim,), coords={synth_time_dim: t_coord}, name=var or "value")
            else:
                # Auto-detect whether time is axis 0 (Python/CF) or axis -1 (MATLAB m_mhw)
                if arr.shape[0] == n_t:
                    t_axis = 0
                elif arr.shape[-1] == n_t:
                    t_axis = arr.ndim - 1
                else:
                    raise ValueError(
                        f"Could not match len(time)={n_t} to first or last axis of shape {arr.shape}"
                    )
                dims = []
                spatial_i = 0
                for ax in range(arr.ndim):
                    if ax == t_axis:
                        dims.append(synth_time_dim)
                    else:
                        dims.append(f"dim_{spatial_i}")
                        spatial_i += 1
                coords: dict[str, Any] = {synth_time_dim: t_coord}
                for ax, dname in enumerate(dims):
                    if dname != synth_time_dim:
                        coords[dname] = np.arange(arr.shape[ax], dtype=np.int64)
                da = xr.DataArray(arr, dims=tuple(dims), coords=coords, name=var or "value")
            if isinstance(time, xr.DataArray) and time.encoding:
                da.coords[synth_time_dim].encoding.update(time.encoding)

        resolved_time_dim = resolve_time_dim(da, time_dim)
        da = _squeeze_singleton_dims(da, resolved_time_dim, squeeze_singleton)

        spatial_dims = tuple(d for d in da.dims if d != resolved_time_dim)
        spatial_shape = tuple(int(da.sizes[d]) for d in spatial_dims)
        spatial_coords: dict[str, np.ndarray] = {}
        for d in spatial_dims:
            if d in da.coords and da.coords[d].ndim == 1 and da.coords[d].dims == (d,):
                spatial_coords[d] = np.asarray(da.coords[d].values)
            else:
                spatial_coords[d] = np.arange(da.sizes[d], dtype=np.int64)

        return OpenedInput(
            da=da,
            time_dim=resolved_time_dim,
            spatial_dims=spatial_dims,
            spatial_shape=spatial_shape,
            spatial_coords=spatial_coords,
            closers=closers,
        )
    except BaseException:
        for obj in closers:
            try:
                obj.close()
            except Exception:
                pass
        raise


@dataclass(frozen=True)
class SpatialTile:
    """Rectangular spatial slice corresponding to a strictly contiguous flat pixel range."""

    isel_dict: dict[str, slice]
    flat_start: int
    flat_end: int
    tile_shape: tuple[int, ...]

    @property
    def n_pixels(self) -> int:
        return self.flat_end - self.flat_start


def iter_spatial_tiles(
    spatial_dims: tuple[str, ...],
    spatial_shape: tuple[int, ...],
    max_tile_pixels: int,
) -> Iterator[SpatialTile]:
    """Yield contiguous rectangular tiles partitioning ``0..prod(spatial_shape)`` in row-major order.

    Every yielded tile satisfies:
    1. Pure rectangular slice ``isel_dict`` for fast NetCDF/Zarr slicing.
    2. Strictly contiguous flat pixel range ``[flat_start, flat_end)`` so CSR offsets and
       streamed Parquet batches assemble in a single pass with zero global sorting.
    """
    if not spatial_dims:
        yield SpatialTile(isel_dict={}, flat_start=0, flat_end=1, tile_shape=())
        return

    max_tile_pixels = max(1, int(max_tile_pixels))
    total_pixels = int(math.prod(spatial_shape))
    if total_pixels <= max_tile_pixels:
        yield SpatialTile(
            isel_dict={d: slice(0, n) for d, n in zip(spatial_dims, spatial_shape)},
            flat_start=0,
            flat_end=total_pixels,
            tile_shape=spatial_shape,
        )
        return

    ndim = len(spatial_shape)
    # Trailing stride after axis `a`: number of pixels per unit step along axis `a`.
    strides = [1] * ndim
    for a in range(ndim - 2, -1, -1):
        strides[a] = strides[a + 1] * spatial_shape[a + 1]

    # Pick the outermost axis whose trailing stride fits inside max_tile_pixels.
    split_axis = ndim - 1
    for a in range(ndim):
        if strides[a] <= max_tile_pixels:
            split_axis = a
            break

    step = max(1, max_tile_pixels // strides[split_axis])
    # Align split step to 16-row HDF5 chunk boundaries so tiles do not trigger chunk read-modify-write.
    if step >= 16:
        step = (step // 16) * 16
    prefix_ranges = [range(spatial_shape[a]) for a in range(split_axis)]
    flat_cursor = 0

    for prefix in itertools.product(*prefix_ranges):
        axis_len = spatial_shape[split_axis]
        for start in range(0, axis_len, step):
            stop = min(axis_len, start + step)
            isel_dict: dict[str, slice] = {}
            tile_shape_list: list[int] = []
            for a in range(split_axis):
                isel_dict[spatial_dims[a]] = slice(prefix[a], prefix[a] + 1)
                tile_shape_list.append(1)
            isel_dict[spatial_dims[split_axis]] = slice(start, stop)
            tile_shape_list.append(stop - start)
            for a in range(split_axis + 1, ndim):
                isel_dict[spatial_dims[a]] = slice(0, spatial_shape[a])
                tile_shape_list.append(spatial_shape[a])

            n_pix = int(math.prod(tile_shape_list))
            yield SpatialTile(
                isel_dict=isel_dict,
                flat_start=flat_cursor,
                flat_end=flat_cursor + n_pix,
                tile_shape=tuple(tile_shape_list),
            )
            flat_cursor += n_pix


def resolve_working_dtype(da: xr.DataArray, working_dtype: str = "auto") -> np.dtype:
    """Choose compact float32 tile buffers for float32/packed-int16 inputs, or float64 otherwise."""
    if working_dtype == "float32":
        return np.dtype(np.float32)
    if working_dtype == "float64":
        return np.dtype(np.float64)
    if working_dtype != "auto":
        raise ValueError(f"working_dtype must be 'auto', 'float32', or 'float64', got '{working_dtype}'")
    compact = (np.dtype(np.float32), np.dtype(np.float16), np.dtype(np.int16), np.dtype(np.uint16))
    dt = np.dtype(da.dtype)
    if dt in compact:
        return np.dtype(np.float32)
    # Check raw on-disk CF encoding dtype (mask_and_scale upcasts int16+float64 scale_factor to float64)
    enc_dt = da.encoding.get("dtype")
    if enc_dt is not None and np.dtype(enc_dt) in compact:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def estimate_max_tile_pixels(
    memory_limit_bytes: int,
    n_clim_time: int,
    n_det_time: int,
    *,
    return_clim: bool = True,
    return_intensity_ts: bool,
    return_daily_category_ts: bool,
    return_event_category_ts: bool,
    moving_n_years: int = 0,
    year_len: int = 366,
    input_itemsize: int = 8,
    clim_itemsize: int = 8,
    intensity_itemsize: int = 8,
    n_intensity_arrays: int = 1,
    extra_curves_count: int = 0,
) -> int:
    """Estimate safe tile pixel count staying comfortably within ``memory_limit_bytes``."""
    bytes_per_pixel = input_itemsize * (n_clim_time + n_det_time) * 2
    if extra_curves_count > 0:
        bytes_per_pixel += 8 * extra_curves_count * n_det_time
    if return_clim:
        n_clim_curves = 3 * (moving_n_years if moving_n_years > 0 else 1) * year_len
        # Rust emits f64 clim/thresholds before optional float32 cast on scatter
        bytes_per_pixel += (8 + clim_itemsize) * n_clim_curves
    if return_intensity_ts and n_intensity_arrays > 0:
        # Rust allocates f64 intensity buffer(s) before optional f32 cast on scatter
        bytes_per_pixel += n_intensity_arrays * (8 + intensity_itemsize) * n_det_time
    if return_daily_category_ts:
        bytes_per_pixel += n_det_time * 2
    if return_event_category_ts:
        bytes_per_pixel += n_det_time * 2
    bytes_per_pixel = max(bytes_per_pixel * 2, 1024)
    return max(1, int(memory_limit_bytes // bytes_per_pixel))


def load_tile_as_pixels_x_time(
    da: xr.DataArray,
    tile: SpatialTile,
    time_dim: str,
    time_slice: slice | np.ndarray | None = None,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Read one rectangular spatial tile and return C-contiguous ``(n_pixels, n_time)``."""
    sel: dict[str, Any] = dict(tile.isel_dict)
    if time_slice is not None:
        sel[time_dim] = time_slice
    sub = da.isel(sel)
    spatial_dims = tuple(d for d in sub.dims if d != time_dim)
    ordered = sub.transpose(*spatial_dims, time_dim)
    arr = np.asarray(ordered.values, dtype=dtype)
    n_time = int(ordered.sizes[time_dim])
    return np.ascontiguousarray(arr.reshape(tile.n_pixels, n_time), dtype=dtype)
