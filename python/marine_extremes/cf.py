"""Lightweight built-in CF conventions metadata resolution with optional ``cf-xarray`` enhancement."""

from __future__ import annotations

import datetime as dt
import importlib.util
import math
import re
import sys
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from marine_extremes.calendar import _CF_TIME_UNITS_RE, _is_cftime_obj

_ALIAS_SPECS: tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("sst", "sea_surface_temperature"),
        ("sea_surface_temperature", "sea_surface_foundation_temperature", "sea_surface_skin_temperature", "sea_surface_subskin_temperature"),
        ("sst", "tos", "analysed_sst", "sea_surface_temperature", "t_surf", "sst_c"),
    ),
    (
        ("temp", "temperature", "sea_water_temperature"),
        ("sea_water_temperature", "sea_water_potential_temperature", "sea_water_conservative_temperature", "sea_surface_temperature", "sea_surface_foundation_temperature"),
        ("thetao", "temp", "temperature", "votemper", "to", "water_temp", "sst", "tos", "analysed_sst"),
    ),
    (
        ("salinity", "sss", "sea_water_salinity"),
        ("sea_water_salinity", "sea_water_practical_salinity", "sea_water_absolute_salinity", "sea_surface_salinity"),
        ("so", "salinity", "salt", "vosaline", "sss", "sos", "psal"),
    ),
    (
        ("chlorophyll", "chl"),
        ("mass_concentration_of_chlorophyll_in_sea_water", "mass_concentration_of_chlorophyll_a_in_sea_water"),
        ("chl", "chlor_a", "chla", "chlorophyll"),
    ),
    (
        ("oxygen", "o2"),
        ("mole_concentration_of_dissolved_molecular_oxygen_in_sea_water", "mass_concentration_of_oxygen_in_sea_water"),
        ("o2", "oxygen", "doxy"),
    ),
)

# Normalized lowercase alias -> (CF standard_names, common dataset variable names)
CF_VAR_ALIASES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    key: (stds, vnames) for keys, stds, vnames in _ALIAS_SPECS for key in keys
}

_COMMON_TIME_DIM_NAMES: frozenset[str] = frozenset({"time", "valid_time", "time_counter", "ocean_time", "t", "date", "dates"})
_KELVIN_UNITS: frozenset[str] = frozenset({"degk", "deg_k", "degree_k", "degrees_k", "kelvin", "kelvins", "degree_kelvin", "degrees_kelvin", "°k", "°_k"})
_CELSIUS_UNITS: frozenset[str] = frozenset({"degc", "deg_c", "degree_c", "degrees_c", "degree_celsius", "degrees_celsius", "celsius", "°c", "°_c"})
_CF_ACCESSOR_ERRORS = (KeyError, ValueError, TypeError, AttributeError)


def has_cf_xarray() -> bool:
    """Return True if the optional ``cf-xarray`` package is installed without importing it."""
    return importlib.util.find_spec("cf_xarray") is not None


def _get_cf_accessor(obj: xr.Dataset | xr.DataArray) -> Any | None:
    """Return ``obj.cf`` when optional ``cf_xarray`` is available, else ``None``."""
    if "cf_xarray" in sys.modules:
        return getattr(obj, "cf", None) if sys.modules["cf_xarray"] is not None else None
    if not has_cf_xarray():
        return None
    try:
        import cf_xarray  # noqa: F401
    except ImportError:
        return None
    return getattr(obj, "cf", None)


def _as_str(val: Any) -> str:
    return bytes(val).decode("utf-8", errors="replace") if isinstance(val, (bytes, bytearray, np.bytes_)) else str(val)


def _split_attr_tokens(val: Any) -> list[str]:
    if isinstance(val, (str, bytes, bytearray, np.bytes_)):
        return _as_str(val).split()
    items = val.ravel().tolist() if isinstance(val, np.ndarray) else (val if isinstance(val, Sequence) else ())
    return [tok for item in items for tok in _as_str(item).split()]


def _extract_grid_mapping_vars(val: Any, data_var_names: set[str]) -> set[str]:
    text = " ".join(_split_attr_tokens(val))
    if not text:
        return set()
    tokens = text.split() if ":" not in text else re.findall(r"(?:^|\s)([^\s:]+)\s*:", text)
    return {t for t in tokens if t in data_var_names}


def _extract_cell_measure_vars(val: Any, data_var_names: set[str]) -> set[str]:
    text = " ".join(_split_attr_tokens(val))
    return {m for m in re.findall(r"(?:^|\s)[^\s:]+\s*:\s*([^\s:]+)(?=\s|$)(?!\s*:)", text) if m in data_var_names}


def cf_auxiliary_vars(ds: xr.Dataset) -> set[str]:
    """Identify CF auxiliary ``data_vars`` (bounds, grid mappings, ancillary vars, cell measures)."""
    data_var_names = {str(v) for v in ds.data_vars}
    if not data_var_names:
        return set()

    aux: set[str] = set()
    for vname, var_obj in ds.variables.items():
        for attrs in (var_obj.attrs, var_obj.encoding):
            for attr_key in ("bounds", "climatology", "ancillary_variables", "coordinates"):
                if attr_key in attrs:
                    aux.update(t for t in _split_attr_tokens(attrs[attr_key]) if t in data_var_names)
            if "grid_mapping" in attrs:
                aux.update(_extract_grid_mapping_vars(attrs["grid_mapping"], data_var_names))
            if "cell_measures" in attrs:
                aux.update(_extract_cell_measure_vars(attrs["cell_measures"], data_var_names))
            if str(vname) in data_var_names and "grid_mapping_name" in attrs:
                aux.add(str(vname))

    cf_acc = _get_cf_accessor(ds)
    if cf_acc is not None:
        for prop in ("bounds", "cell_measures"):
            try:
                for names in getattr(cf_acc, prop).values():
                    aux.update(str(n) for n in names if str(n) in data_var_names)
            except _CF_ACCESSOR_ERRORS:
                pass
    return aux


def _dim_coord(obj: xr.Dataset | xr.DataArray, dim: Any) -> xr.DataArray | None:
    sources = (obj.coords, obj.data_vars) if isinstance(obj, xr.Dataset) else (obj.coords,)
    for src in sources:
        if dim in src and src[dim].dims == (dim,):
            return src[dim]
    return None


def _has_explicit_cf_time_axis(coord: xr.DataArray) -> bool:
    if _as_str(coord.attrs.get("axis", "")).strip().upper() == "T":
        return True
    if _as_str(coord.attrs.get("cartesian_axis", "")).strip().upper() == "T":
        return True
    return _as_str(coord.attrs.get("standard_name", "")).strip().lower() == "time"


def _has_cf_time_units(coord: xr.DataArray) -> bool:
    return any(
        (u := src.get("units")) is not None and _CF_TIME_UNITS_RE.match(_as_str(u))
        for src in (coord.attrs, coord.encoding)
    )


def _has_datetime_dtype(coord: xr.DataArray, dim: Any) -> bool:
    if np.issubdtype(coord.dtype, np.datetime64):
        return True
    if coord.dtype == object and coord.sizes[dim] > 0:
        sample = np.asarray(coord.isel({dim: slice(0, min(8, int(coord.sizes[dim])))}).values)
        for val in sample:
            if (
                val is None
                or val is pd.NaT
                or (isinstance(val, (float, np.floating)) and np.isnan(val))
                or (isinstance(val, np.datetime64) and np.isnat(val))
            ):
                continue
            return _is_cftime_obj(val) or isinstance(val, (dt.date, dt.datetime, pd.Timestamp))
    return False


def _pick_single_or_raise(candidates: list[Any], dims: tuple[Any, ...]) -> Any:
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Ambiguous time dimensions {candidates} in data dims {dims}; pass `time_dim=...` explicitly.")


def resolve_time_dim(obj: xr.Dataset | xr.DataArray, time_dim: str | None = "auto") -> Any:
    """Resolve an explicit or ``'auto'`` time dimension on an xarray Dataset or DataArray."""
    dims = tuple(obj.dims)
    if time_dim is not None and time_dim != "auto":
        if time_dim in dims:
            return time_dim
        raise ValueError(f"Time dimension '{time_dim}' not found in data dims {dims}")

    if "time" in dims:
        return "time"

    dim_coords = [(d, _dim_coord(obj, d)) for d in dims]

    # Tier 1: Explicit CF time axis / standard_name metadata
    t_attr = [d for d, coord in dim_coords if coord is not None and _has_explicit_cf_time_axis(coord)]
    if t_attr:
        return _pick_single_or_raise(t_attr, dims)

    # Tier 2: Decoded datetime64 / cftime / Python date dtype or CF "<unit> since ..." units
    t_val = [d for d, coord in dim_coords if coord is not None and (_has_datetime_dtype(coord, d) or _has_cf_time_units(coord))]
    if t_val:
        return _pick_single_or_raise(t_val, dims)

    # Tier 3: Optional cf-xarray enhancement (axes["T"] + guess_coord_axis)
    cf_acc = _get_cf_accessor(obj)
    if cf_acc is not None:
        found: list[Any] = []
        try:
            found = [d for d in cf_acc.axes.get("T", []) if d in dims]
            if not found:
                guessed = cf_acc.guess_coord_axis(verbose=False)
                guessed_acc = _get_cf_accessor(guessed)
                if guessed_acc is not None:
                    found = [d for d in guessed_acc.axes.get("T", []) if d in dims]
        except _CF_ACCESSOR_ERRORS:
            found = []
        if found:
            return _pick_single_or_raise(list(dict.fromkeys(found)), dims)

    # Tier 4: Common ocean/climate time dimension names (case-insensitive)
    t_name = [d for d in dims if _as_str(d).lower() in _COMMON_TIME_DIM_NAMES]
    if t_name:
        return _pick_single_or_raise(t_name, dims)

    raise ValueError(f"Could not infer time dimension from data dims {dims}; pass `time_dim=...` explicitly.")


def _match_by_standard_name(ds: xr.Dataset, candidates: list[str], targets_lower: set[str]) -> list[str]:
    return [
        name
        for name in candidates
        if (std := ds[name].attrs.get("standard_name")) is not None and _as_str(std).strip().lower() in targets_lower
    ]


def _resolve_cf_xarray_var(ds: xr.Dataset, var: str, aux: set[str]) -> list[str]:
    cf_acc = _get_cf_accessor(ds)
    if cf_acc is None:
        return []
    data_var_names = {str(v) for v in ds.data_vars}
    try:
        sub = cf_acc[[var]]
        if isinstance(sub, xr.Dataset):
            return [str(k) for k in sub.data_vars if str(k) in data_var_names and str(k) not in aux]
    except _CF_ACCESSOR_ERRORS:
        pass
    return []


def _pick_var(ds: xr.Dataset, hits: list[str], label: str) -> xr.DataArray | None:
    if len(hits) == 1:
        return ds[hits[0]]
    if len(hits) > 1:
        raise ValueError(f"Multiple variables {hits} match {label}; pass an exact `var=...`.")
    return None


def select_dataset_var(ds: xr.Dataset, var: str | None = None, *, time_dim: str | None = "auto") -> xr.DataArray:
    """Select a Dataset variable by exact name, CF ``standard_name``, alias, or sole primary var."""
    if var is not None:
        if var in ds.data_vars:
            return ds[var]

        aux = cf_auxiliary_vars(ds)
        all_vars = [str(v) for v in ds.data_vars]
        primary = [v for v in all_vars if v not in aux]
        target_std = {str(var).strip().lower()}

        std_hits = _match_by_standard_name(ds, primary, target_std) or _match_by_standard_name(ds, all_vars, target_std)
        if (picked := _pick_var(ds, std_hits, f"standard_name '{var}'")) is not None:
            return picked
        if (picked := _pick_var(ds, _resolve_cf_xarray_var(ds, var, aux), f"CF criterion '{var}'")) is not None:
            return picked

        alias = CF_VAR_ALIASES.get(str(var).strip().lower())
        if alias is not None:
            alias_stds, alias_names = alias
            if (picked := _pick_var(ds, _match_by_standard_name(ds, primary, set(alias_stds)), f"alias '{var}' standard_names")) is not None:
                return picked
            alias_name_set = set(alias_names)
            if (picked := _pick_var(ds, [v for v in primary if v.lower() in alias_name_set], f"alias '{var}'")) is not None:
                return picked

        raise KeyError(f"Variable '{var}' not found in dataset. Available: {list(ds.data_vars)}")

    all_vars = [str(v) for v in ds.data_vars]
    if not all_vars:
        raise ValueError("Dataset contains no data variables.")
    if len(all_vars) == 1:
        return ds[all_vars[0]]

    aux = cf_auxiliary_vars(ds)
    primary = [v for v in all_vars if v not in aux]
    if len(primary) > 1:
        if time_dim is not None and time_dim != "auto":
            t_dim = resolve_time_dim(ds, time_dim)
        else:
            try:
                t_dim = resolve_time_dim(ds, time_dim)
            except ValueError:
                t_dim = None
        if t_dim is not None:
            primary = [v for v in primary if t_dim in ds[v].dims] or primary

    if len(primary) == 1:
        return ds[primary[0]]
    raise ValueError(f"Dataset has multiple variables {primary or all_vars}; pass `var=...` explicitly.")


def is_kelvin_units(raw_units: Any) -> bool:
    """Return True if a CF ``units`` attribute represents Kelvin."""
    u = _as_str(raw_units).strip() if raw_units is not None else ""
    return bool(u) and (u == "K" or "_".join(u.lower().split()) in _KELVIN_UNITS)


def warn_if_units_mismatch(da: xr.DataArray, lower_bound: float | None, *, stacklevel: int = 3) -> None:
    """Warn (without modifying values) when temperature ``units`` and ``lower_bound`` conflict."""
    if lower_bound is None or not math.isfinite(lb := float(lower_bound)):
        return
    if (raw_units := da.attrs.get("units")) is None or not (units := _as_str(raw_units).strip()):
        return

    if is_kelvin_units(units) and lb < 100.0:
        warnings.warn(
            f"Input variable '{da.name or 'value'}' has units='{units}' (Kelvin), "
            f"but lower_bound={lower_bound!r} looks like Celsius (< 100). "
            "Check whether data or lower_bound needs Kelvin/Celsius conversion.",
            UserWarning,
            stacklevel=stacklevel,
        )
    elif (units == "C" or "_".join(units.lower().split()) in _CELSIUS_UNITS) and lb > 150.0:
        warnings.warn(
            f"Input variable '{da.name or 'value'}' has units='{units}' (Celsius), "
            f"but lower_bound={lower_bound!r} looks like Kelvin (> 150). "
            "Check whether data or lower_bound needs Kelvin/Celsius conversion.",
            UserWarning,
            stacklevel=stacklevel,
        )
