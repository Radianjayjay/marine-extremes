"""Adversarial edge-case, streaming-baseline, and real-world NetCDF stress tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marine_extremes as mx


def test_all_nan_pure_land_in_memory_and_streaming(tmp_path: Path):
    dates = pd.date_range("2000-01-01", "2005-12-31", freq="D")
    da = xr.DataArray(
        np.full((dates.size, 3, 4), np.nan, dtype=np.float64),
        dims=("time", "lat", "lon"),
        coords={"time": dates, "lat": [0.0, 1.0, 2.0], "lon": [10.0, 11.0, 12.0, 13.0]},
        name="sst",
    )

    r_mem = mx.detect(da, event="both", return_event_category_ts=True)
    assert len(r_mem.events) == 0
    assert np.all(r_mem.events.count_map().values == 0)
    assert np.all(np.isnan(r_mem.clim.values))
    assert np.all(np.isnan(r_mem.intensity_ts.values))
    assert np.all(r_mem.daily_category_ts.values == mx.CATEGORY_MISSING)

    out_file = tmp_path / "land_out.nc"
    r_disk = mx.detect(da, event="both", return_event_category_ts=True, out=out_file)
    assert out_file.is_file()
    assert len(r_disk.events) == 0
    assert np.all(r_disk.daily_category_ts.values == mx.CATEGORY_MISSING)
    assert np.all(r_disk.event_category_ts.values == mx.CATEGORY_MISSING)
    assert len(r_disk.events.isel(lat=1, lon=2)) == 0


def test_zero_events_constant_field_in_memory_and_streaming(tmp_path: Path):
    dates = pd.date_range("2000-01-01", "2005-12-31", freq="D")
    da = xr.DataArray(
        np.full((dates.size, 2, 2), 20.0, dtype=np.float64),
        dims=("time", "lat", "lon"),
        coords={"time": dates, "lat": [0.0, 1.0], "lon": [10.0, 11.0]},
        name="sst",
    )

    r_mem = mx.detect(da, event="both", compat="standard")
    assert len(r_mem.events) == 0
    # Valid ocean with no events must have intensity_ts == 0.0 and daily_category_ts == 0
    assert np.all(r_mem.intensity_ts.values == 0.0)
    assert np.all(r_mem.daily_category_ts.values == 0)

    r_disk = mx.detect(da, event="both", compat="standard", out=tmp_path / "zero_out.nc")
    assert len(r_disk.events) == 0
    assert np.all(r_disk.intensity_ts.values == 0.0)
    assert np.all(r_disk.daily_category_ts.values == 0)


def test_1d_streaming_for_fixed_detrended_and_moving_baselines(tmp_path: Path):
    rng = np.random.default_rng(404)
    dates = pd.date_range("1995-01-01", "2014-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    t_yr = np.arange(dates.size) / 365.25
    eps = rng.normal(0.0, 0.5, size=dates.size)
    ar = np.zeros_like(eps)
    for t in range(1, dates.size):
        ar[t] = 0.8 * ar[t - 1] + eps[t]
    sst = 19.0 + 3.0 * np.sin(2.0 * np.pi * doy / 365.25) + 0.05 * t_yr + ar
    da = xr.DataArray(sst, dims=("time",), coords={"time": dates}, name="sst")

    for name, base in [
        ("fixed", mx.FixedBaseline("1995-01-01", "2009-12-31")),
        ("detrend", mx.DetrendedBaseline(method="linear")),
        ("moving", mx.MovingBaseline(window_years=11, center=True, boundary="clamp")),
    ]:
        r_mem = mx.detect(da, event="both", baseline=base, detect_period=("2000-01-01", "2014-12-31"))
        r_disk = mx.detect(
            da,
            event="both",
            baseline=base,
            detect_period=("2000-01-01", "2014-12-31"),
            out=tmp_path / f"1d_{name}.nc",
        )
        pd.testing.assert_frame_equal(r_disk.events.to_pandas(), r_mem.events.to_pandas())
        np.testing.assert_allclose(r_disk.clim.values, r_mem.clim.values, equal_nan=True)
        np.testing.assert_allclose(r_disk.intensity_ts.values, r_mem.intensity_ts.values, equal_nan=True)
        np.testing.assert_array_equal(r_disk.daily_category_ts.values, r_mem.daily_category_ts.values)
        if name == "detrend":
            assert r_disk.trend is not None
            np.testing.assert_allclose(
                r_disk.trend["slope_per_step"].values.reshape(()),
                r_mem.trend["slope_per_step"].values.reshape(()),
            )


def test_detrended_and_moving_2d_streaming_with_detect_period_subset(tmp_path: Path):
    rng = np.random.default_rng(505)
    dates = pd.date_range("1990-01-01", "2019-12-31", freq="D")
    nt = dates.size
    doy = dates.dayofyear.to_numpy()
    t_yr = np.arange(nt) / 365.25
    data = np.empty((nt, 4, 5), dtype=np.float64)
    for i in range(4):
        for j in range(5):
            if i == 0 and j == 0:
                data[:, i, j] = np.nan
                continue
            eps = rng.normal(0.0, 0.5, size=nt)
            ar = np.zeros(nt)
            for t in range(1, nt):
                ar[t] = 0.8 * ar[t - 1] + eps[t]
            data[:, i, j] = 17.0 + 3.0 * np.sin(2.0 * np.pi * doy / 365.25) + 0.04 * t_yr + ar
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": dates, "lat": [10.0, 11.0, 12.0, 13.0], "lon": [100.0, 101.0, 102.0, 103.0, 104.0]},
        name="sst",
    )

    for label, base in [
        ("detrend_quad", mx.DetrendedBaseline(method="quadratic")),
        ("moving_11_trailing", mx.MovingBaseline(window_years=11, center=False, boundary="clamp")),
    ]:
        r_mem = mx.detect(da, event="both", baseline=base, detect_period=("2005-01-01", "2019-12-31"))
        r_disk = mx.detect(
            da,
            event="both",
            baseline=base,
            detect_period=("2005-01-01", "2019-12-31"),
            memory_limit="2MB",
            out=tmp_path / f"{label}.nc",
        )
        pd.testing.assert_frame_equal(r_disk.events.to_pandas(), r_mem.events.to_pandas())
        np.testing.assert_allclose(r_disk.clim.values, r_mem.clim.values, equal_nan=True)
        np.testing.assert_allclose(r_disk.intensity_ts.values, r_mem.intensity_ts.values, equal_nan=True)
        np.testing.assert_array_equal(r_disk.daily_category_ts.values, r_mem.daily_category_ts.values)
        if r_mem.trend is not None:
            assert r_disk.trend is not None
            np.testing.assert_allclose(
                r_disk.trend["slope_per_step"].values, r_mem.trend["slope_per_step"].values, equal_nan=True
            )


def test_chained_gap_merge_with_nan_and_subthreshold_island_rejection():
    dates = pd.date_range("2010-01-01", periods=60, freq="D")
    base = mx.FixedThreshold(threshold_high=22.0, climatology=20.0)

    # Case A: 5-day + 2-day gap (with 1 NaN!) + 5-day + 2-day gap + 5-day -> single 19-day event, no NaN poison
    v_chain = np.full(60, 20.0)
    v_chain[5:10] = 24.0
    v_chain[10] = np.nan
    v_chain[11] = 21.0
    v_chain[12:17] = 25.0
    v_chain[17:19] = 21.0
    v_chain[19:24] = 23.0
    r_a = mx.detect(v_chain, time=dates, baseline=base, min_duration=5, max_gap=2)
    df_a = r_a.events.to_pandas()
    assert len(df_a) == 1
    assert df_a["duration"].iloc[0] == 19
    assert np.isfinite(df_a["int_mean"].iloc[0])
    assert np.isfinite(df_a["int_std"].iloc[0])
    assert np.isfinite(df_a["int_cum"].iloc[0])

    # Case B: 5-day + 1-day gap + 3-day exceedance (< min_duration=5) + 1-day gap + 5-day
    # The 3-day island is filtered before gap merging -> effective gap is 5 days (> max_gap=2) -> 2 separate events!
    v_island = np.full(60, 20.0)
    v_island[5:10] = 24.0
    v_island[11:14] = 24.0  # only 3 days!
    v_island[15:20] = 24.0
    r_b = mx.detect(v_island, time=dates, baseline=base, min_duration=5, max_gap=2)
    assert len(r_b.events) == 2
    assert list(r_b.events.to_pandas()["duration"]) == [5, 5]


def test_single_year_with_31yr_moving_baseline_and_packed_int16_netcdf(tmp_path: Path):
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    lats = np.array([30.0, 20.0, 10.0])  # descending latitude like ERA5/OSTIA!
    lons = np.array([0.0, 180.0, 359.75])
    rng = np.random.default_rng(606)
    sst = 20.0 + rng.normal(0.0, 1.0, size=(dates.size, 3, 3))
    sst[100:110, 1, 1] = 26.0

    # Write CF-packed int16 NetCDF with custom time dimension name "t"
    nc_path = tmp_path / "packed_oisst.nc"
    ds = xr.Dataset(
        {"analysed_sst": (("t", "lat", "lon"), sst)},
        coords={"t": dates, "lat": lats, "lon": lons},
    )
    ds.to_netcdf(
        nc_path,
        encoding={"analysed_sst": {"dtype": "int16", "scale_factor": 0.01, "add_offset": 20.0, "_FillValue": -32768}},
    )

    res = mx.detect(
        nc_path,
        var="analysed_sst",
        time_dim="t",
        event="high",
        baseline=mx.MovingBaseline(window_years=31, center=True, boundary="clamp"),
    )
    assert res.clim.sizes["year"] == 1
    df_pt = res.events.sel(lat=20.0, lon=180.0)
    assert len(df_pt) >= 1
    assert (df_pt["duration"] >= 10).any()

    # Verify EventCatalog.save / open / filter & DetectResult.to_dataset()
    cat_file = tmp_path / "saved_cat.parquet"
    res.events.save(cat_file)
    assert cat_file.is_file()
    cat2 = mx.EventCatalog.open(cat_file)
    pd.testing.assert_frame_equal(cat2.sel(lat=20.0, lon=180.0), df_pt)
    filtered = cat2.filter(event_type="mhw", min_duration=10)
    assert len(filtered) >= 1
    ds_out = res.to_dataset()
    assert "clim" in ds_out and "intensity_ts" in ds_out
