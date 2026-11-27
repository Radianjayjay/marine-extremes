"""Tests for edge cases, 4D depth grids, CSR EventCatalog, and single-file streaming."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marxtreme as mx


def test_max_pad_length_only_applies_to_detection_not_climatology():
    dates = pd.date_range("2000-01-01", "2009-12-31", freq="D")
    sst = np.full(dates.size, 20.0, dtype=np.float64)
    # Inject a 10-day heatwave with a 2-day internal NaN gap in 2005
    s0 = int(np.where(dates == pd.Timestamp("2005-06-01"))[0][0])
    sst[s0 : s0 + 10] = 25.0
    sst[s0 + 3 : s0 + 5] = np.nan

    # Without padding: two short runs (len 3 and len 5) -> only len 5 detected (or len 3 dropped)
    r_nopad = mx.detect(
        sst,
        time=dates,
        baseline=mx.FixedThreshold(threshold_high=22.0, climatology=20.0),
        min_duration=5,
        max_gap=2,
        max_pad_length=0,
    )
    assert len(r_nopad.events) == 1
    assert r_nopad.events.to_pandas()["duration"].iloc[0] == 5

    # With max_pad_length=2: internal 2-day NaN linearly interpolated to 25.0 -> full 10-day event
    r_pad = mx.detect(
        sst,
        time=dates,
        baseline=mx.FixedThreshold(threshold_high=22.0, climatology=20.0),
        min_duration=5,
        max_gap=2,
        max_pad_length=2,
    )
    assert len(r_pad.events) == 1
    assert r_pad.events.to_pandas()["duration"].iloc[0] == 10

    # Verify climatology is identical regardless of max_pad_length
    r_c0 = mx.detect(sst, time=dates, max_pad_length=0)
    r_c2 = mx.detect(sst, time=dates, max_pad_length=2)
    np.testing.assert_allclose(r_c0.clim.values, r_c2.clim.values, equal_nan=True)


def test_min_valid_fraction_sea_ice_guard():
    dates = pd.date_range("2000-01-01", "2019-12-31", freq="D")
    sst = np.full(dates.size, 2.0, dtype=np.float64)
    # Simulate winter sea-ice: Jan-Feb missing in 16 out of 20 years (only 20% valid)
    winter_ice = (dates.month <= 2) & (dates.year >= 2004)
    sst[winter_ice] = np.nan

    clim_guarded = mx.compute_climatology(sst, time=dates, min_valid_fraction=0.5)
    # Deep winter (DOY 15) should be NaN under 50% requirement, summer (DOY 200) finite
    assert np.isnan(clim_guarded["seas"].values[14])
    assert np.isfinite(clim_guarded["seas"].values[199])


def test_lower_bound_policies_and_truncated_flags():
    dates = pd.date_range("2010-01-01", "2010-03-31", freq="D")
    sst = np.full(dates.size, -1.5, dtype=np.float64)
    # Event touching start of record (truncated_start=True)
    sst[0:8] = -1.82
    # Event touching end of record (truncated_end=True)
    sst[-7:] = -1.85

    base = mx.FixedThreshold(threshold_low=-1.79, climatology=-1.79)  # delta = 0.0 -> div-zero stress

    r_flag = mx.detect(sst, time=dates, event="mcs", baseline=base, lower_bound=-1.8, bound_policy="flag")
    df_flag = r_flag.events.to_pandas()
    assert len(df_flag) == 2
    assert bool(df_flag.iloc[0]["truncated_start"]) is True
    assert bool(df_flag.iloc[0]["truncated_end"]) is False
    assert bool(df_flag.iloc[1]["truncated_start"]) is False
    assert bool(df_flag.iloc[1]["truncated_end"]) is True
    assert df_flag["near_lower_bound"].all()
    assert (df_flag["category_max"] == 1).all()  # safely guarded against Inf/NaN

    r_mask = mx.detect(sst, time=dates, event="mcs", baseline=base, lower_bound=-1.8, bound_policy="mask")
    assert len(r_mask.events) == 2  # -1.79 > -1.8, still active

    base_frozen = mx.FixedThreshold(threshold_low=-1.80, climatology=-1.80)
    r_masked_out = mx.detect(
        sst, time=dates, event="mcs", baseline=base_frozen, lower_bound=-1.8, bound_policy="mask"
    )
    assert len(r_masked_out.events) == 0


def test_4d_depth_grid_matches_independent_surface_layers_and_csr_lookup():
    rng = np.random.default_rng(99)
    dates = pd.date_range("2000-01-01", "2009-12-31", freq="D")
    nt = dates.size
    depths = np.array([5.0, 50.0, 200.0])
    lats = np.array([-30.0, -29.0])
    lons = np.array([150.0, 151.0, 152.0])

    doy = dates.dayofyear.to_numpy()
    seas = 18.0 + 3.0 * np.sin(2.0 * np.pi * doy / 365.25)
    data = np.empty((nt, depths.size, lats.size, lons.size), dtype=np.float64)
    for k in range(depths.size):
        for i in range(lats.size):
            for j in range(lons.size):
                if (k == 2 and i == 0 and j == 0):
                    # Seafloor land mask at deep layer only!
                    data[:, k, i, j] = np.nan
                    continue
                eps = rng.normal(0.0, 0.5, size=nt)
                ar = np.zeros(nt)
                for t in range(1, nt):
                    ar[t] = 0.8 * ar[t - 1] + eps[t]
                data[:, k, i, j] = seas - 0.03 * depths[k] + ar

    da4 = xr.DataArray(
        data,
        dims=("time", "depth", "lat", "lon"),
        coords={"time": dates, "depth": depths, "lat": lats, "lon": lons},
        name="thetao",
    )

    res4 = mx.detect(da4, event="both", return_event_category_ts=True)

    # Verify seafloor pixel (k=2, i=0, j=0) has 0 events, NaN intensity_ts, CATEGORY_MISSING
    assert len(res4.events.isel(depth=2, lat=0, lon=0)) == 0
    assert np.all(np.isnan(res4.intensity_ts.isel(depth=2, lat=0, lon=0).values))
    assert np.all(res4.daily_category_ts.isel(depth=2, lat=0, lon=0).values == mx.CATEGORY_MISSING)

    # Verify non-event ocean days are 0.0 (not NaN)
    ocean_int = res4.intensity_ts.isel(depth=0, lat=0, lon=0).values
    assert np.all(np.isfinite(ocean_int))
    assert np.any(ocean_int == 0.0)

    # Verify depth layer k=1 matches running surface-only 3D detect on da4.isel(depth=1) identically
    res_layer1 = mx.detect(da4.isel(depth=1, drop=True), event="both")
    df_4d_pt = res4.events.sel(depth=50.0, lat=-29.0, lon=151.0).drop(columns=["pixel_idx", "depth_idx", "depth"])
    df_3d_pt = res_layer1.events.sel(lat=-29.0, lon=151.0).drop(columns=["pixel_idx"])
    pd.testing.assert_frame_equal(df_4d_pt.reset_index(drop=True), df_3d_pt.reset_index(drop=True))

    # Positional indexing mirroring detectc MHWc{k, i, j}
    pd.testing.assert_frame_equal(res4.events[1, 1, 1], res4.events.isel(depth=1, lat=1, lon=1))

    # Instant count_map matches actual per-grid counts
    cmap = res4.events.count_map()
    assert cmap.dims == ("depth", "lat", "lon")
    assert int(cmap.isel(depth=2, lat=0, lon=0)) == 0
    assert int(cmap.isel(depth=1, lat=1, lon=1)) == len(res4.events[1, 1, 1])


def test_multifile_glob_streaming_matches_inmemory_bit_for_bit(tmp_path: Path):
    """Write yearly NetCDF files, stream with tiny memory_limit forcing many tiles, verify exact parity."""
    rng = np.random.default_rng(123)
    lats = np.linspace(-20.0, -15.0, 6)
    lons = np.linspace(140.0, 146.0, 7)
    depths = np.array([0.0, 100.0])

    nc_dir = tmp_path / "yearly_nc"
    nc_dir.mkdir()
    full_parts = []

    for year in range(2000, 2010):
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        nt = dates.size
        doy = dates.dayofyear.to_numpy()
        seas = 22.0 + 3.0 * np.sin(2.0 * np.pi * doy / 365.25)[:, None, None, None]
        noise = rng.normal(0.0, 0.7, size=(nt, depths.size, lats.size, lons.size))
        # Simple temporal smoothing to generate multi-day events
        for t in range(1, nt):
            noise[t] = 0.8 * noise[t - 1] + 0.5 * noise[t]
        arr = seas + noise
        # Add land pixels & a pure-land slice
        arr[:, :, 0, :] = np.nan
        arr[:, 1, 2, 3] = np.nan

        ds = xr.Dataset(
            {"sst": (("time", "depth", "lat", "lon"), arr)},
            coords={"time": dates, "depth": depths, "lat": lats, "lon": lons},
        )
        ds.to_netcdf(nc_dir / f"sst_{year}.nc")
        full_parts.append(ds["sst"])

    full_da = xr.concat(full_parts, dim="time")

    res_mem = mx.detect(
        full_da,
        event="both",
        baseline=mx.FixedBaseline("2000-01-01", "2009-12-31"),
        return_event_category_ts=True,
        memory_limit="2GB",
    )

    out_file = tmp_path / "stream_out.nc"
    # Tiny memory_limit forces multiple spatial/depth tiles!
    res_disk = mx.detect(
        str(nc_dir / "sst_*.nc"),
        var="sst",
        event="both",
        baseline=mx.FixedBaseline("2000-01-01", "2009-12-31"),
        return_event_category_ts=True,
        memory_limit="1MB",
        out=out_file,
    )

    assert out_file.is_file()
    assert res_disk.events.is_disk_backed
    assert len(res_disk.events) == len(res_mem.events)
    assert len(res_disk.events) > 20

    pd.testing.assert_frame_equal(res_disk.events.to_pandas(), res_mem.events.to_pandas())
    np.testing.assert_array_equal(res_disk.events.count_map().values, res_mem.events.count_map().values)

    # Verify O(1) disk-backed sel/isel matches in-memory
    pd.testing.assert_frame_equal(
        res_disk.events.isel(depth=0, lat=3, lon=4).reset_index(drop=True),
        res_mem.events.isel(depth=0, lat=3, lon=4).reset_index(drop=True),
    )
    assert len(res_disk.events.isel(depth=0, lat=0, lon=0)) == 0

    # Verify single-file NetCDF gridded outputs match bit-for-bit
    np.testing.assert_allclose(res_disk.clim.values, res_mem.clim.values, equal_nan=True)
    np.testing.assert_allclose(res_disk.thresh_high.values, res_mem.thresh_high.values, equal_nan=True)
    np.testing.assert_allclose(res_disk.thresh_low.values, res_mem.thresh_low.values, equal_nan=True)
    np.testing.assert_allclose(res_disk.intensity_ts.values, res_mem.intensity_ts.values, equal_nan=True)
    np.testing.assert_array_equal(res_disk.daily_category_ts.values, res_mem.daily_category_ts.values)
    np.testing.assert_array_equal(res_disk.event_category_ts.values, res_mem.event_category_ts.values)

    # Verify mx.open() / DetectResult.open() reloads cleanly
    reopened = mx.open(out_file)
    assert len(reopened.events) == len(res_mem.events)
    pd.testing.assert_frame_equal(
        reopened.events.sel(depth=0.0, lat=-17.0, lon=143.0).reset_index(drop=True),
        res_mem.events.sel(depth=0.0, lat=-17.0, lon=143.0).reset_index(drop=True),
    )


def test_matlab_time_last_ndarray_and_pandas_series_inputs():
    dates = pd.date_range("2000-01-01", "2005-12-31", freq="D")
    rng = np.random.default_rng(17)
    arr_3d = 20.0 + rng.normal(0.0, 1.0, size=(3, 4, dates.size))  # MATLAB (lat, lon, time) order!
    res = mx.detect(arr_3d, time=dates, event="high")
    assert res.clim.shape == (3, 4, 366)

    s = pd.Series(arr_3d[0, 0, :], index=dates, name="sst")
    res_s = mx.detect(s, event="high")
    np.testing.assert_allclose(res_s.clim.values, res.clim.values[0, 0, :])


def test_float32_fast_path_and_squeeze_singleton_oisst_layout(tmp_path: Path):
    """Verify OISST-like (time, zlev=1, lat, lon) float32/int16 data uses compact f32 path cleanly."""
    rng = np.random.default_rng(808)
    dates = pd.date_range("2000-01-01", "2009-12-31", freq="D")
    nt = dates.size
    doy = dates.dayofyear.to_numpy()
    seas = (19.0 + 3.5 * np.sin(2.0 * np.pi * doy / 365.25))[:, None, None, None]
    noise = rng.normal(0.0, 0.6, size=(nt, 1, 3, 4))
    for t in range(1, nt):
        noise[t] = 0.8 * noise[t - 1] + 0.5 * noise[t]
    arr_f32 = (seas + noise).astype(np.float32)

    da_f32 = xr.DataArray(
        arr_f32,
        dims=("time", "zlev", "lat", "lon"),
        coords={"time": dates, "zlev": [0.0], "lat": [-30.0, -29.0, -28.0], "lon": [150.0, 151.0, 152.0, 153.0]},
        name="sst",
    )

    r_f32 = mx.detect(
        da_f32,
        event="both",
        squeeze_singleton="zlev",
        working_dtype="auto",
        intensity_dtype="float32",
    )
    r_f64 = mx.detect(
        da_f32.astype(np.float64),
        event="both",
        squeeze_singleton=True,
        working_dtype="float64",
    )

    assert r_f32.metadata["working_dtype"] == "float32"
    assert r_f32.intensity_ts.dtype == np.float32
    assert r_f32.clim.dims == ("lat", "lon", "doy")
    assert r_f64.clim.dims == ("lat", "lon", "doy")
    assert len(r_f32.events) == len(r_f64.events)
    np.testing.assert_allclose(r_f32.clim.values, r_f64.clim.values, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(r_f32.intensity_ts.values, r_f64.intensity_ts.values, rtol=1e-5, atol=1e-5)

    # Also test DetrendedBaseline & MovingBaseline under float32 fast path + streaming
    r_det_f32 = mx.detect(
        da_f32,
        event="both",
        baseline=mx.DetrendedBaseline("linear"),
        squeeze_singleton="zlev",
        intensity_dtype="float32",
        out=tmp_path / "f32_det.nc",
    )
    assert r_det_f32.intensity_ts.dtype == np.float32
    assert len(r_det_f32.events) > 10

