"""Tests for built-in lightweight CF resolution, optional cf-xarray enhancement, and units warnings."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marine_extremes as mx
import marine_extremes.cf as cf_mod
from marine_extremes.streaming import OpenedInput, open_data_input


def _configure_cf_mode(monkeypatch: pytest.MonkeyPatch, with_cf_xarray: bool) -> None:
    """Force built-in-only (False) or optional cf-xarray (True) resolution for isolated testing."""
    if not with_cf_xarray:
        monkeypatch.setattr(cf_mod, "_get_cf_accessor", lambda _obj: None)
    elif not cf_mod.has_cf_xarray():
        pytest.skip("cf-xarray not installed")


def _make_sst_cube(
    time_dim: str = "time",
    time_vals: np.ndarray | pd.DatetimeIndex | None = None,
    n_lat: int = 2,
    n_lon: int = 2,
    seed: int = 42,
) -> xr.DataArray:
    if time_vals is None:
        time_vals = pd.date_range("2000-01-01", "2004-12-31", freq="D")
    nt = len(time_vals)
    rng = np.random.default_rng(seed)
    t_idx = np.arange(nt, dtype=np.float64)
    seas = (20.0 + 3.0 * np.sin(2.0 * np.pi * t_idx / 365.25))[:, None, None]
    noise = rng.normal(0.0, 0.6, size=(nt, n_lat, n_lon))
    for i in range(1, nt):
        noise[i] = 0.8 * noise[i - 1] + 0.5 * noise[i]
    # Inject a clear warm spike so events are detected
    noise[500:512, :, :] += 3.5
    lats = np.linspace(-30.0, -30.0 + (n_lat - 1), n_lat)
    lons = np.linspace(150.0, 150.0 + (n_lon - 1), n_lon)
    return xr.DataArray(
        seas + noise,
        dims=(time_dim, "lat", "lon"),
        coords={time_dim: time_vals, "lat": lats, "lon": lons},
        name="sst",
    )


def test_resolve_time_dim_tiers_and_strict_explicit():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")

    # Fast path: "time"
    da_time = xr.DataArray(np.zeros((10, 2)), dims=("time", "lat"), coords={"time": dates, "lat": [0, 1]})
    assert cf_mod.resolve_time_dim(da_time, "auto") == "time"
    assert cf_mod.resolve_time_dim(da_time, None) == "time"

    # ERA5-like "valid_time" via datetime64 dtype
    da_era5 = xr.DataArray(np.zeros((10, 2)), dims=("valid_time", "lat"), coords={"valid_time": dates, "lat": [0, 1]})
    assert cf_mod.resolve_time_dim(da_era5, "auto") == "valid_time"
    assert cf_mod.resolve_time_dim(da_era5, None) == "valid_time"

    # NEMO/ORAS5-like "time_counter" via CF axis="T" (even on numeric coord)
    da_nemo = xr.DataArray(
        np.zeros((10, 2)),
        dims=("time_counter", "lat"),
        coords={"time_counter": ("time_counter", np.arange(10), {"axis": "T"}), "lat": [0, 1]},
    )
    assert cf_mod.resolve_time_dim(da_nemo, "auto") == "time_counter"

    # ROMS-like "ocean_time" with cftime 360_day & CF units
    cftime = pytest.importorskip("cftime")
    cf_dates = np.array([cftime.Datetime360Day(2000, 1, d + 1) for d in range(10)], dtype=object)
    da_roms = xr.DataArray(
        np.zeros((10, 2)),
        dims=("ocean_time", "lat"),
        coords={"ocean_time": ("ocean_time", cf_dates, {"units": "days since 1980-01-01"}), "lat": [0, 1]},
    )
    assert cf_mod.resolve_time_dim(da_roms, "auto") == "ocean_time"

    # Tier 4 case-insensitive common name fallback (numeric coord, no CF attrs)
    da_upper = xr.DataArray(
        np.zeros((10, 2)),
        dims=("VALID_TIME", "lat"),
        coords={"VALID_TIME": np.arange(10), "lat": [0, 1]},
    )
    assert cf_mod.resolve_time_dim(da_upper, "auto") == "VALID_TIME"

    # Ambiguous time dimensions raise ValueError
    da_ambig = xr.DataArray(
        np.zeros((5, 5)),
        dims=("valid_time", "time_counter"),
        coords={"valid_time": dates[:5], "time_counter": dates[:5]},
    )
    with pytest.raises(ValueError, match="Ambiguous time dimensions"):
        cf_mod.resolve_time_dim(da_ambig, "auto")

    # Missing time dimension raises ValueError
    da_notime = xr.DataArray(np.zeros((2, 3)), dims=("lat", "lon"), coords={"lat": [0, 1], "lon": [10, 11, 12]})
    with pytest.raises(ValueError, match="Could not infer time dimension"):
        cf_mod.resolve_time_dim(da_notime, "auto")

    # Explicit time_dim stays strict
    assert cf_mod.resolve_time_dim(da_time, "time") == "time"
    with pytest.raises(ValueError, match="Time dimension 'missing' not found"):
        cf_mod.resolve_time_dim(da_time, "missing")

    # Raw numpy arrays & pandas Series resolve to synthetic "time" under default time_dim="auto"
    opened_np = open_data_input(np.ones(10), time=dates)
    assert opened_np.time_dim == "time"
    opened_ser = open_data_input(pd.Series(np.ones(10), index=dates, name="sst"))
    assert opened_ser.time_dim == "time"


def test_resolve_time_dim_tiering_nat_nan_and_encoding_units():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")

    # Tier 1 explicit axis="T" beats Tier 2 encoding["units"] on another dim
    da_t1_wins = xr.DataArray(
        np.zeros((5, 5)),
        dims=("init_time", "valid_time"),
        coords={
            "init_time": ("init_time", np.arange(5)),
            "valid_time": ("valid_time", np.arange(5), {"axis": "T"}),
        },
    )
    da_t1_wins.coords["init_time"].encoding["units"] = "days since 2020-01-01"
    assert cf_mod.resolve_time_dim(da_t1_wins, "auto") == "valid_time"

    # Both encoding["units"] and datetime64 are Tier 2 -> raises Ambiguous time dimensions
    da_t2_ambig = xr.DataArray(
        np.zeros((5, 5)),
        dims=("init_time", "valid_time"),
        coords={
            "init_time": ("init_time", np.arange(5)),
            "valid_time": dates,
        },
    )
    da_t2_ambig.coords["init_time"].encoding["units"] = "days since 2020-01-01"
    with pytest.raises(ValueError, match="Ambiguous time dimensions"):
        cf_mod.resolve_time_dim(da_t2_ambig, "auto")

    # Object coord containing pd.NaT or np.float32('nan') before strings is NOT mistaken for datetime
    da_nat_str = xr.DataArray(
        np.zeros((3, 5)),
        dims=("station", "valid_time"),
        coords={
            "station": np.array([pd.NaT, "foo", "bar"], dtype=object),
            "valid_time": dates,
        },
    )
    assert cf_mod.resolve_time_dim(da_nat_str, "auto") == "valid_time"

    da_f32nan_str = xr.DataArray(
        np.zeros((3, 5)),
        dims=("station", "valid_time"),
        coords={
            "station": np.array([np.float32("nan"), "foo", "bar"], dtype=object),
            "valid_time": dates,
        },
    )
    assert cf_mod.resolve_time_dim(da_f32nan_str, "auto") == "valid_time"


@pytest.mark.parametrize("ext", [".nc", ".zarr", ".mat", ".parquet"])
def test_end_to_end_non_time_dim_roundtrip(tmp_path: Path, ext: str):
    da = _make_sst_cube(time_dim="valid_time")
    ds = xr.Dataset({"analysed_sst": da})
    ds["analysed_sst"].attrs["standard_name"] = "sea_surface_foundation_temperature"

    clim = mx.compute_climatology(ds, var="sst")
    assert clim["seas"].dims == ("lat", "lon", "doy")

    out_path = tmp_path / f"valid_time_out{ext}"
    res = mx.detect(ds, var="sst", event="both", out=out_path)
    if ext != ".parquet":
        assert res.intensity_ts.dims == ("lat", "lon", "valid_time")
    assert len(res.events) > 0

    reopened = mx.open(out_path)
    pd.testing.assert_frame_equal(
        reopened.events.sel(lat=-30.0, lon=150.0).reset_index(drop=True),
        res.events.sel(lat=-30.0, lon=150.0).reset_index(drop=True),
    )
    if ext != ".parquet":
        assert reopened.intensity_ts.dims == ("lat", "lon", "valid_time")
        np.testing.assert_allclose(reopened.clim.values, res.clim.values, equal_nan=True)
        np.testing.assert_allclose(reopened.intensity_ts.values, res.intensity_ts.values, equal_nan=True)
    reopened.close()
    res.close()


def test_mismatched_clim_data_and_external_climatology_time_dims():
    da = _make_sst_cube(time_dim="valid_time")
    da_clim = da.rename({"valid_time": "time"})

    # FixedBaseline(clim_data=...) with mismatched time dim ("time" vs "valid_time")
    res_fixed_clim = mx.detect(
        da,
        baseline=mx.FixedBaseline("2000-01-01", "2003-12-31", clim_data=da_clim),
    )
    assert res_fixed_clim.intensity_ts.dims == ("lat", "lon", "valid_time")

    # Time-series ExternalClimatology with mismatched time dim ("time" vs "valid_time")
    res_base = mx.detect(da, event="high")
    doy_idx = xr.DataArray(da["valid_time"].dt.dayofyear.values, dims=("time",))
    # Map simple DOY climatology to daily time-series layout under dim "time"
    seas_t = res_base.clim.sel(doy=doy_idx).drop_vars("doy").assign_coords(time=da_clim["time"])
    hi_t = res_base.thresh_high.sel(doy=doy_idx).drop_vars("doy").assign_coords(time=da_clim["time"])
    res_ext = mx.detect(
        da,
        event="high",
        baseline=mx.ExternalClimatology(seas=seas_t, thresh_high=hi_t),
    )
    assert res_ext.intensity_ts.dims == ("lat", "lon", "valid_time")
    assert len(res_ext.events) > 0


@pytest.mark.parametrize("with_cf_xarray", [False, True])
def test_var_none_filters_cf_auxiliary_and_refuses_multi_science_vars(
    monkeypatch: pytest.MonkeyPatch, with_cf_xarray: bool
):
    _configure_cf_mode(monkeypatch, with_cf_xarray)
    dates = pd.date_range("2000-01-01", periods=10, freq="D")
    nt = len(dates)

    # Empty dataset raises ValueError
    with pytest.raises(ValueError, match="no data variables"):
        cf_mod.select_dataset_var(xr.Dataset(), None)

    # 1. CMIP6-like Dataset: tos + time_bnds + lat_bnds + lon_bnds + 0D crs
    ds_cmip = xr.Dataset(
        {
            "tos": (
                ("time", "lat", "lon"),
                np.ones((nt, 2, 2)),
                {"standard_name": "sea_surface_temperature", "grid_mapping": "crs"},
            ),
            "time_bnds": (("time", "bnds"), np.zeros((nt, 2))),
            "lat_bnds": (("lat", "bnds"), np.zeros((2, 2))),
            "lon_bnds": (("lon", "bnds"), np.zeros((2, 2))),
            "crs": ((), 0, {"grid_mapping_name": "latitude_longitude"}),
        },
        coords={
            "time": ("time", dates, {"bounds": "time_bnds"}),
            "lat": ("lat", [-10.0, -9.0], {"bounds": "lat_bnds"}),
            "lon": ("lon", [120.0, 121.0], {"bounds": "lon_bnds"}),
        },
    )
    assert cf_mod.select_dataset_var(ds_cmip, None).name == "tos"

    # 2. GHRSST/OSTIA-like Dataset: ancillary_variables + extended grid_mapping + cell_measures
    ds_ostia = xr.Dataset(
        {
            "analysed_sst": (
                ("time", "lat", "lon"),
                np.full((nt, 2, 2), 295.0),
                {
                    "standard_name": "sea_surface_foundation_temperature",
                    "ancillary_variables": "analysis_error sea_ice_fraction mask",
                    "grid_mapping": "crs: lat lon",
                    "cell_measures": "area: areacello",
                },
            ),
            "analysis_error": (("time", "lat", "lon"), np.full((nt, 2, 2), 0.1)),
            "sea_ice_fraction": (("time", "lat", "lon"), np.zeros((nt, 2, 2))),
            "mask": (("time", "lat", "lon"), np.ones((nt, 2, 2))),
            "areacello": (("lat", "lon"), np.ones((2, 2))),
            "crs": ((), 0),
        },
        coords={"time": dates, "lat": [-10.0, -9.0], "lon": [120.0, 121.0]},
    )
    assert cf_mod.select_dataset_var(ds_ostia, None).name == "analysed_sst"

    # 3. Static 2D mask without CF attrs is filtered by presence of time dim
    ds_static_mask = xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), np.ones((nt, 2, 2))),
            "mask": (("lat", "lon"), np.ones((2, 2))),
        },
        coords={"time": dates, "lat": [-10.0, -9.0], "lon": [120.0, 121.0]},
    )
    assert cf_mod.select_dataset_var(ds_static_mask, None).name == "sst"

    # 4. OISST multi-science Dataset refuses to guess
    ds_oisst_multi = xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), np.ones((nt, 2, 2))),
            "anom": (("time", "lat", "lon"), np.zeros((nt, 2, 2))),
            "err": (("time", "lat", "lon"), np.zeros((nt, 2, 2))),
            "ice": (("time", "lat", "lon"), np.zeros((nt, 2, 2))),
        },
        coords={"time": dates, "lat": [-10.0, -9.0], "lon": [120.0, 121.0]},
    )
    with pytest.raises(ValueError, match="multiple variables"):
        cf_mod.select_dataset_var(ds_oisst_multi, None)

    # 5. Multi-variable thetao + so refuses to guess
    ds_ocean_multi = xr.Dataset(
        {
            "thetao": (("time", "lat", "lon"), np.ones((nt, 2, 2))),
            "so": (("time", "lat", "lon"), np.full((nt, 2, 2), 35.0)),
        },
        coords={"time": dates, "lat": [-10.0, -9.0], "lon": [120.0, 121.0]},
    )
    with pytest.raises(ValueError, match="multiple variables"):
        cf_mod.select_dataset_var(ds_ocean_multi, None)


@pytest.mark.parametrize("with_cf_xarray", [False, True])
def test_cf_auxiliary_vars_edge_cases_and_explicit_invalid_time_dim(
    monkeypatch: pytest.MonkeyPatch, with_cf_xarray: bool
):
    _configure_cf_mode(monkeypatch, with_cf_xarray)
    dates = pd.date_range("2000-01-01", periods=5, freq="D")

    # 1. cell_measures with space before colon does NOT mistake measure keyword "area" for auxiliary var
    ds_cell = xr.Dataset(
        {
            "tos": (("time", "lat"), np.ones((5, 2)), {"cell_measures": "area : areacello"}),
            "area": (("time", "lat"), np.ones((5, 2))),
            "areacello": (("lat",), np.ones(2)),
        },
        coords={"time": dates, "lat": [0.0, 1.0]},
    )
    aux_cell = cf_mod.cf_auxiliary_vars(ds_cell)
    assert "areacello" in aux_cell
    assert "area" not in aux_cell
    with pytest.raises(ValueError, match="multiple variables"):
        cf_mod.select_dataset_var(ds_cell, None)

    # 1b. Regex edge cases for cell_measures & grid_mapping (no-space, multi-pair, malformed)
    all_keys = {"sst", "volume", "volcello", "areacello", "crs", "lat"}
    assert cf_mod._extract_cell_measure_vars("area: areacello volume: volcello", all_keys) == {"areacello", "volcello"}
    assert cf_mod._extract_cell_measure_vars("area:areacello", all_keys) == {"areacello"}
    assert cf_mod._extract_cell_measure_vars("area: volume: volcello", all_keys) == {"volcello"}
    assert cf_mod._extract_cell_measure_vars(": sst", all_keys) == set()
    assert cf_mod._extract_cell_measure_vars("area::sst", all_keys) == set()
    assert cf_mod._extract_grid_mapping_vars("crs:lat:lon", all_keys) == {"crs"}

    # 2. grid_mapping with space before colon ("my_crs : lat") + bytes/ndarray attrs + coordinates attr
    ds_edge = xr.Dataset(
        {
            "tos": (
                ("valid_time", "j", "i"),
                np.ones((5, 2, 2)),
                {
                    "grid_mapping": b"my_crs : lat",
                    "ancillary_variables": np.array(["err1", "err2"]),
                    "coordinates": "nav_lat nav_lon",
                },
            ),
            "my_crs": ((), 0),
            "err1": (("valid_time", "j", "i"), np.zeros((5, 2, 2))),
            "err2": (("valid_time", "j", "i"), np.zeros((5, 2, 2))),
            "nav_lat": (("j", "i"), np.zeros((2, 2))),
            "nav_lon": (("j", "i"), np.zeros((2, 2))),
        },
        coords={"valid_time": dates, "j": [0, 1], "i": [0, 1]},
    )
    aux_edge = cf_mod.cf_auxiliary_vars(ds_edge)
    assert {"my_crs", "err1", "err2", "nav_lat", "nav_lon"} <= aux_edge
    assert cf_mod.select_dataset_var(ds_edge, None).name == "tos"

    # 2b. decode_cf() moves coordinates into encoding; reset_coords() still filters them
    ds_decoded = xr.decode_cf(ds_edge).reset_coords(["nav_lat", "nav_lon"])
    assert "coordinates" not in ds_decoded["tos"].attrs
    assert cf_mod.select_dataset_var(ds_decoded, None).name == "tos"

    # 3. Explicit invalid time_dim is never swallowed by select_dataset_var(ds, None)
    ds_static = xr.Dataset(
        {
            "sst": (("valid_time", "lat"), np.ones((5, 2))),
            "mask": (("lat",), np.ones(2)),
        },
        coords={"valid_time": dates, "lat": [0.0, 1.0]},
    )
    with pytest.raises(ValueError, match="Time dimension 'wrong_dim' not found"):
        cf_mod.select_dataset_var(ds_static, None, time_dim="wrong_dim")


def test_multifile_glob_1d_promoted_crs_filtered_without_cf_xarray(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(cf_mod, "_get_cf_accessor", lambda _obj: None)
    for yr in (2000, 2001):
        dates = pd.date_range(f"{yr}-01-01", f"{yr}-12-31", freq="D")
        vals = 20.0 + np.zeros((dates.size, 2, 2))
        if yr == 2001:
            vals[100:110, :, :] = 25.0
        ds = xr.Dataset(
            {
                "tos": (("valid_time", "lat", "lon"), vals),
                # Standalone 0D grid_mapping_name var NOT referenced by tos.attrs["grid_mapping"];
                # open_mfdataset(..., data_vars="all") concatenates it along valid_time into 1D!
                "crs": ((), 0, {"grid_mapping_name": "latitude_longitude"}),
            },
            coords={"valid_time": dates, "lat": [-10.0, -9.0], "lon": [140.0, 141.0]},
        )
        ds.to_netcdf(tmp_path / f"tos_{yr}.nc")

    res = mx.detect(str(tmp_path / "tos_*.nc"), event="high")
    assert res.intensity_ts.dims == ("lat", "lon", "valid_time")
    assert len(res.events) > 0


@pytest.mark.parametrize("with_cf_xarray", [False, True])
def test_var_resolution_exact_standard_name_aliases_and_errors(
    monkeypatch: pytest.MonkeyPatch, with_cf_xarray: bool
):
    _configure_cf_mode(monkeypatch, with_cf_xarray)
    dates = pd.date_range("2000-01-01", periods=5, freq="D")

    # Exact name takes priority over standard_name/alias
    ds_prio = xr.Dataset(
        {
            "sst": (("time", "lat"), np.ones((5, 2))),
            "other": (("time", "lat"), np.zeros((5, 2)), {"standard_name": "sea_surface_temperature"}),
        },
        coords={"time": dates, "lat": [0, 1]},
    )
    assert cf_mod.select_dataset_var(ds_prio, "sst").name == "sst"

    # standard_name exact match
    ds_std = xr.Dataset(
        {"tos": (("time", "lat"), np.ones((5, 2)), {"standard_name": "sea_surface_temperature"})},
        coords={"time": dates, "lat": [0, 1]},
    )
    assert cf_mod.select_dataset_var(ds_std, "sea_surface_temperature").name == "tos"

    # Alias "sst" -> OSTIA analysed_sst via sea_surface_foundation_temperature (with ancillary var)
    ds_ostia = xr.Dataset(
        {
            "analysed_sst": (
                ("time", "lat"),
                np.ones((5, 2)),
                {"standard_name": "sea_surface_foundation_temperature", "ancillary_variables": "err"},
            ),
            "err": (("time", "lat"), np.zeros((5, 2))),
        },
        coords={"time": dates, "lat": [0, 1]},
    )
    assert cf_mod.select_dataset_var(ds_ostia, "sst").name == "analysed_sst"

    # Alias "sst" -> CMIP6 tos without standard_name
    ds_tos_plain = xr.Dataset({"tos": (("time", "lat"), np.ones((5, 2)))}, coords={"time": dates, "lat": [0, 1]})
    assert cf_mod.select_dataset_var(ds_tos_plain, "sst").name == "tos"

    # Alias "salinity" -> so (with and without standard_name)
    ds_sal_std = xr.Dataset(
        {"so": (("time", "lat"), np.full((5, 2), 35.0), {"standard_name": "sea_water_salinity"})},
        coords={"time": dates, "lat": [0, 1]},
    )
    assert cf_mod.select_dataset_var(ds_sal_std, "salinity").name == "so"
    ds_sal_plain = xr.Dataset({"so": (("time", "lat"), np.full((5, 2), 35.0))}, coords={"time": dates, "lat": [0, 1]})
    assert cf_mod.select_dataset_var(ds_sal_plain, "salinity").name == "so"

    # Alias "temp" -> thetao (with and without standard_name)
    ds_temp_std = xr.Dataset(
        {"thetao": (("time", "lat"), np.ones((5, 2)), {"standard_name": "sea_water_potential_temperature"})},
        coords={"time": dates, "lat": [0, 1]},
    )
    assert cf_mod.select_dataset_var(ds_temp_std, "temp").name == "thetao"
    ds_temp_plain = xr.Dataset({"thetao": (("time", "lat"), np.ones((5, 2)))}, coords={"time": dates, "lat": [0, 1]})
    assert cf_mod.select_dataset_var(ds_temp_plain, "temp").name == "thetao"

    # Case-colliding alias variables (SST vs Sst) raise ValueError instead of silent dict overwrite
    ds_case_ambig = xr.Dataset(
        {
            "SST": (("time", "lat"), np.ones((5, 2))),
            "Sst": (("time", "lat"), np.zeros((5, 2))),
        },
        coords={"time": dates, "lat": [0, 1]},
    )
    with pytest.raises(ValueError, match="Multiple variables"):
        cf_mod.select_dataset_var(ds_case_ambig, "sst")

    # Ambiguous standard_name / alias match raises ValueError
    ds_ambig = xr.Dataset(
        {
            "sst_day": (("time", "lat"), np.ones((5, 2)), {"standard_name": "sea_surface_temperature"}),
            "sst_night": (("time", "lat"), np.ones((5, 2)), {"standard_name": "sea_surface_temperature"}),
        },
        coords={"time": dates, "lat": [0, 1]},
    )
    with pytest.raises(ValueError, match="Multiple variables"):
        cf_mod.select_dataset_var(ds_ambig, "sea_surface_temperature")
    with pytest.raises(ValueError, match="Multiple variables"):
        cf_mod.select_dataset_var(ds_ambig, "sst")

    # Missing variable raises KeyError
    with pytest.raises(KeyError, match="not found in dataset"):
        cf_mod.select_dataset_var(ds_tos_plain, "nonexistent")


@pytest.mark.parametrize("with_cf_xarray", [False, True])
def test_external_climatology_file_with_auxiliary_bounds_and_coords(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_cf_xarray: bool
):
    _configure_cf_mode(monkeypatch, with_cf_xarray)
    da = _make_sst_cube(time_dim="time")
    clim_ds = mx.compute_climatology(da)
    clim_nc = tmp_path / "ext_clim.nc"
    hi_nc = tmp_path / "ext_hi.nc"

    # Rename primary var and attach CF bounds + 2D coordinates stored in data_vars
    ds_s = xr.Dataset(
        {
            "climatology": (
                ("lat", "lon", "doy"),
                clim_ds["seas"].transpose("lat", "lon", "doy").values,
                {"coordinates": "nav_lat nav_lon"},
            ),
            "lat_bnds": (("lat", "bnds"), np.zeros((2, 2))),
            "lon_bnds": (("lon", "bnds"), np.zeros((2, 2))),
            "nav_lat": (("lat", "lon"), np.zeros((2, 2))),
            "nav_lon": (("lat", "lon"), np.zeros((2, 2))),
        },
        coords={
            "lat": ("lat", clim_ds["lat"].values, {"bounds": "lat_bnds"}),
            "lon": ("lon", clim_ds["lon"].values, {"bounds": "lon_bnds"}),
            "doy": clim_ds["doy"].values,
        },
    )
    ds_h = xr.Dataset(
        {
            "hi_val": (("lat", "lon", "doy"), clim_ds["thresh_high"].transpose("lat", "lon", "doy").values),
            "lat_bnds": (("lat", "bnds"), np.zeros((2, 2))),
        },
        coords={
            "lat": ("lat", clim_ds["lat"].values, {"bounds": "lat_bnds"}),
            "lon": clim_ds["lon"].values,
            "doy": clim_ds["doy"].values,
        },
    )
    ds_s.to_netcdf(clim_nc)
    ds_h.to_netcdf(hi_nc)

    res_ext = mx.detect(da, baseline=mx.ExternalClimatology(seas=clim_nc, thresh_high=hi_nc))
    res_dir = mx.detect(da)
    pd.testing.assert_frame_equal(res_ext.events.to_pandas(), res_dir.events.to_pandas())


def test_units_mismatch_warning_stacklevel_handle_close_and_udunits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    da_c = _make_sst_cube(time_dim="time")
    da_c.attrs["units"] = "degrees_C"
    da_k = da_c + 273.15
    da_k.attrs["units"] = "K"

    # Kelvin data + Celsius-like lower_bound -> UserWarning pointing to caller line (__file__)
    with warnings.catch_warnings(record=True) as rec_warn:
        warnings.simplefilter("always")
        res_k = mx.detect(da_k, event="mcs", lower_bound=-1.8)
    user_warns = [w for w in rec_warn if issubclass(w.category, UserWarning)]
    assert len(user_warns) == 1
    assert "Kelvin" in str(user_warns[0].message)
    assert user_warns[0].filename == __file__

    with warnings.catch_warnings(record=True) as rec_k:
        warnings.simplefilter("always")
        res_k_ref = mx.detect(da_k, event="mcs", lower_bound=271.35)
    assert not [w for w in rec_k if issubclass(w.category, UserWarning)]
    np.testing.assert_allclose(res_k.clim.values, res_k_ref.clim.values)

    # Space-separated UDUNITS & bytes units
    for c_unit in ("°C", "degrees Celsius", "deg C", "° C"):
        da_u = da_c.copy()
        da_u.attrs["units"] = c_unit
        with pytest.warns(UserWarning, match="Celsius"):
            cf_mod.warn_if_units_mismatch(da_u, 271.35, stacklevel=2)

    for k_unit in ("deg K", "degree Kelvin", b"K"):
        da_u = da_k.copy()
        da_u.attrs["units"] = k_unit
        with pytest.warns(UserWarning, match="Kelvin"):
            cf_mod.warn_if_units_mismatch(da_u, -1.8, stacklevel=2)

    # Non-finite lower_bound & matching units produce no warning
    with warnings.catch_warnings(record=True) as rec_ok:
        warnings.simplefilter("always")
        mx.detect(da_c, event="mcs", lower_bound=-1.8)
        mx.detect(da_k, event="mcs", lower_bound=None)
        cf_mod.warn_if_units_mismatch(da_k, float("-inf"))
        cf_mod.warn_if_units_mismatch(da_k, float("nan"))
    assert not [w for w in rec_ok if issubclass(w.category, UserWarning)]

    # Under warnings.simplefilter("error"), NetCDF file handles opened by detect() are still closed
    nc_k = tmp_path / "sst_k.nc"
    xr.Dataset({"sst": da_k}).to_netcdf(nc_k)
    close_calls = 0
    orig_close = OpenedInput.close

    def _spy_close(self: OpenedInput) -> None:
        nonlocal close_calls
        close_calls += 1
        orig_close(self)

    monkeypatch.setattr(OpenedInput, "close", _spy_close)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="Kelvin"):
            mx.detect(nc_k, event="mcs", lower_bound=-1.8)
    assert close_calls == 1


def test_fast_path_and_optional_cf_xarray_hooks_including_tier3_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
):
    dates = pd.date_range("2000-01-01", periods=5, freq="D")
    ds_fast = xr.Dataset(
        {"sst": (("time", "lat"), np.ones((5, 2)))},
        coords={"time": dates, "lat": [0.0, 1.0]},
    )

    # sys.modules["cf_xarray"] = None sentinel disables accessor even if ds has a "cf" variable
    ds_cf_var = xr.Dataset({"cf": (("time", "lat"), np.ones((5, 2)))}, coords={"time": dates, "lat": [0.0, 1.0]})
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "cf_xarray", None)
        assert cf_mod._get_cf_accessor(ds_cf_var) is None

    def _forbid_cf(_obj):
        raise AssertionError("_get_cf_accessor should not be called on fast path")

    monkeypatch.setattr(cf_mod, "_get_cf_accessor", _forbid_cf)
    assert cf_mod.select_dataset_var(ds_fast, "sst").name == "sst"
    assert cf_mod.resolve_time_dim(ds_fast, "auto") == "time"

    # Stubbed cf-xarray accessor tests (no global xarray accessor mutation)
    ds_stub = xr.Dataset(
        {
            "my_custom_sst": (("custom_t", "lat"), np.ones((5, 2))),
            "extra_bnds": (("lat", "bnds"), np.zeros((2, 2))),
        },
        coords={"custom_t": np.arange(5), "lat": [0.0, 1.0]},
    )

    class _StubAccessor:
        def __init__(self, obj, axes=None):
            self._obj = obj
            self.axes = axes or {}
            self.standard_names = {}
            self.bounds = {"lat": ["extra_bnds"]}
            self.grid_mapping_names = {}

        @property
        def cell_measures(self):
            raise TypeError("malformed cell_measures")

        def __getitem__(self, key):
            if key == ["my_criterion"] or key == "my_criterion":
                return self._obj[["my_custom_sst"]]
            raise KeyError(key)

        def guess_coord_axis(self, verbose: bool = False):
            return SimpleNamespace(dims=self._obj.dims, _guessed_axes={"T": ["custom_t"]})

    def _fake_get_cf_accessor(obj):
        if hasattr(obj, "_guessed_axes"):
            return _StubAccessor(ds_stub, axes=obj._guessed_axes)
        if hasattr(obj, "dims") and "t" in obj.dims and "step_idx" in obj.dims:
            return _StubAccessor(obj, axes={"T": ["step_idx", "t"]})
        return _StubAccessor(obj)

    monkeypatch.setattr(cf_mod, "_get_cf_accessor", _fake_get_cf_accessor)
    # Faulty cell_measures property must not drop bounds ("extra_bnds")
    assert "extra_bnds" in cf_mod.cf_auxiliary_vars(ds_stub)
    assert cf_mod.select_dataset_var(ds_stub, "my_criterion").name == "my_custom_sst"
    assert cf_mod.select_dataset_var(ds_stub, None).name == "my_custom_sst"
    assert cf_mod.resolve_time_dim(ds_stub, "auto") == "custom_t"

    # Tier 3 multi-time-dim ambiguity must raise ValueError instead of falling through to Tier 4 ("t")
    da_t3_ambig = xr.DataArray(
        np.zeros((5, 5)),
        dims=("step_idx", "t"),
        coords={"step_idx": np.arange(5), "t": np.arange(5)},
    )
    with pytest.raises(ValueError, match="Ambiguous time dimensions"):
        cf_mod.resolve_time_dim(da_t3_ambig, "auto")


@pytest.mark.skipif(not cf_mod.has_cf_xarray(), reason="cf-xarray not installed")
def test_real_cf_xarray_custom_criteria_integration():
    import cf_xarray

    dates = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    rng = np.random.default_rng(7)
    vals = 20.0 + rng.normal(0.0, 0.5, size=(dates.size, 2, 2))
    vals[200:212, :, :] += 4.0
    ds = xr.Dataset(
        {
            "temp_custom_42": (
                ("time", "lat", "lon"),
                vals,
                {"standard_name": "custom_ocean_temp", "units": "degC"},
            )
        },
        coords={"time": dates, "lat": [-10.0, -9.0], "lon": [140.0, 141.0]},
    )
    with cf_xarray.set_options(custom_criteria={"my_sst": {"standard_name": "custom_ocean_temp"}}):
        assert cf_mod.select_dataset_var(ds, "my_sst").name == "temp_custom_42"
        res = mx.detect(ds, var="my_sst", event="high")
        assert len(res.events) > 0


@pytest.mark.skipif(not cf_mod.has_cf_xarray(), reason="cf-xarray not installed")
def test_real_cf_xarray_curvilinear_cmip6_aux_and_time_axis():
    dates = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    nt = dates.size
    rng = np.random.default_rng(19)
    tos_vals = 22.0 + rng.normal(0.0, 0.5, size=(nt, 2, 3))
    tos_vals[150:162, :, :] += 3.5
    lat2d = np.array([[-10.0, -10.0, -10.0], [-9.0, -9.0, -9.0]])
    lon2d = np.array([[120.0, 121.0, 122.0], [120.2, 121.2, 122.2]])

    ds_cmip = xr.Dataset(
        {
            "tos": (
                ("time", "j", "i"),
                tos_vals,
                {
                    "standard_name": "sea_surface_temperature",
                    "units": "degC",
                    "cell_measures": "area: areacello",
                    "grid_mapping": "crs",
                },
            ),
            "time_bnds": (("time", "bnds"), np.zeros((nt, 2))),
            "vertices_latitude": (("j", "i", "vertices"), np.zeros((2, 3, 4))),
            "vertices_longitude": (("j", "i", "vertices"), np.zeros((2, 3, 4))),
            "areacello": (("j", "i"), np.ones((2, 3))),
            "crs": ((), 0, {"grid_mapping_name": "latitude_longitude"}),
        },
        coords={
            "time": ("time", dates, {"bounds": "time_bnds"}),
            "j": np.arange(2),
            "i": np.arange(3),
            "latitude": (("j", "i"), lat2d, {"bounds": "vertices_latitude"}),
            "longitude": (("j", "i"), lon2d, {"bounds": "vertices_longitude"}),
        },
    )
    aux = cf_mod.cf_auxiliary_vars(ds_cmip)
    assert {"time_bnds", "vertices_latitude", "vertices_longitude", "areacello", "crs"} <= aux
    assert cf_mod.select_dataset_var(ds_cmip, None).name == "tos"
    res = mx.detect(ds_cmip, event="high")
    assert res.clim.dims == ("j", "i", "doy")
    assert len(res.events) > 0

    # Non-standard dim resolved via real cf-xarray axes["T"] and guess_coord_axis()
    da_coord_attr = xr.DataArray(
        np.zeros((5, 2)),
        dims=("step_idx", "lat"),
        coords={"step_idx": ("step_idx", np.arange(5), {"_CoordinateAxisType": "Time"}), "lat": [0.0, 1.0]},
    )
    assert cf_mod.resolve_time_dim(da_coord_attr, "auto") == "step_idx"

    da_guess = xr.DataArray(
        np.zeros((5, 2)),
        dims=("time_step", "lat"),
        coords={"time_step": np.arange(5), "lat": [0.0, 1.0]},
    )
    assert cf_mod.resolve_time_dim(da_guess, "auto") == "time_step"

    # Real cf-xarray Tier 3 multi-time-dim ambiguity raises ValueError (does not fall through to "t")
    da_real_t3_ambig = xr.DataArray(
        np.zeros((5, 5)),
        dims=("step_idx", "t"),
        coords={
            "step_idx": ("step_idx", np.arange(5), {"_CoordinateAxisType": "Time"}),
            "t": ("t", np.arange(5), {"_CoordinateAxisType": "Time"}),
        },
    )
    with pytest.raises(ValueError, match="Ambiguous time dimensions"):
        cf_mod.resolve_time_dim(da_real_t3_ambig, "auto")
