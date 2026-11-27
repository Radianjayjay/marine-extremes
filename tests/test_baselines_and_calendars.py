"""Tests for Fixed, Detrended, Moving (11yr/31yr), FixedThreshold, ExternalClimatology, and CMIP6 calendars."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marxtreme as mx


def _synthetic_grid(
    n_years: int = 35,
    start_year: int = 1982,
    shape: tuple[int, ...] = (2, 3),
    slope_per_year: float = 0.0,
    seed: int = 7,
) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{start_year}-01-01", f"{start_year + n_years - 1}-12-31", freq="D")
    nt = dates.size
    doy = dates.dayofyear.to_numpy()
    t_years = np.arange(nt, dtype=np.float64) / 365.25
    seasonal = 16.0 + 4.0 * np.sin(2.0 * np.pi * (doy - 40) / 365.25)

    n_pix = int(np.prod(shape))
    data = np.empty((nt, n_pix), dtype=np.float64)
    for p in range(n_pix):
        eps = rng.normal(0.0, 0.5, size=nt)
        ar = np.zeros(nt)
        for t in range(1, nt):
            ar[t] = 0.8 * ar[t - 1] + eps[t]
        data[:, p] = seasonal + slope_per_year * t_years + ar + 0.5 * p

    dims = ("time", *[f"d{i}" for i in range(len(shape))])
    if len(shape) == 2:
        dims = ("time", "lat", "lon")
        coords = {
            "time": dates,
            "lat": np.linspace(-40.0, -39.0, shape[0]),
            "lon": np.linspace(145.0, 147.0, shape[1]),
        }
    else:
        coords = {"time": dates, **{d: np.arange(n) for d, n in zip(dims[1:], shape)}}
    return xr.DataArray(data.reshape((nt, *shape)), dims=dims, coords=coords, name="sst")


def test_detrended_baseline_removes_warming_before_detection():
    # Strong +0.08 degC/yr linear trend across 40 years (+3.2 degC warming)
    da = _synthetic_grid(n_years=40, start_year=1982, shape=(2, 2), slope_per_year=0.08, seed=11)

    res_fixed = mx.detect(da, event="both", baseline=mx.FixedBaseline("1982-01-01", "2001-12-31"))
    res_det = mx.detect(da, event="both", baseline=mx.DetrendedBaseline(method="linear"))

    assert res_det.trend is not None
    recovered_slope_per_year = res_det.trend["slope_per_step"].values * 365.25
    np.testing.assert_allclose(recovered_slope_per_year, 0.08, atol=0.01)

    df_fix = res_fixed.events.to_pandas()
    df_det = res_det.events.to_pandas()
    # Under fixed early baseline + strong warming, late decade has almost zero MCS and huge MHW count.
    late_fix_mcs = df_fix[(df_fix["event_type"] == "low") & (df_fix["date_start"] >= "2012-01-01")]
    late_det_mcs = df_det[(df_det["event_type"] == "low") & (df_det["date_start"] >= "2012-01-01")]
    assert len(late_det_mcs) > len(late_fix_mcs) + 5

    # Quadratic detrending also works cleanly
    res_quad = mx.detect(da, event="high", baseline=mx.DetrendedBaseline(method="quadratic"))
    assert res_quad.trend is not None
    assert len(res_quad.events) > 0


@pytest.mark.parametrize("window_years", [11, 31])
def test_moving_baseline_custom_windows_match_exact_fixed_slices(window_years):
    da = _synthetic_grid(n_years=40, start_year=1982, shape=(1, 2), slope_per_year=0.05, seed=21)
    res_mov = mx.detect(
        da,
        event="both",
        baseline=mx.MovingBaseline(window_years=window_years, center=True, boundary="clamp"),
    )
    assert res_mov.clim is not None
    assert res_mov.clim.dims == ("lat", "lon", "year", "doy")
    assert res_mov.clim.sizes["year"] == 40

    # Verify interior year 2005 uses exact centered window [2005 - half, 2005 + half]
    half = (window_years - 1) // 2
    y_check = 2005
    ref_clim_2005 = mx.compute_climatology(
        da,
        baseline=mx.FixedBaseline(f"{y_check - half}-01-01", f"{y_check + half}-12-31"),
    )
    np.testing.assert_allclose(
        res_mov.clim.sel(year=y_check).values,
        ref_clim_2005["seas"].values,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        res_mov.thresh_high.sel(year=y_check).values,
        ref_clim_2005["thresh_high"].values,
        rtol=1e-12,
        atol=1e-12,
    )

    # Verify clamped boundary year 1982 uses [1982, 1982 + window_years - 1]
    ref_clim_1982 = mx.compute_climatology(
        da,
        baseline=mx.FixedBaseline("1982-01-01", f"{1982 + window_years - 1}-12-31"),
    )
    np.testing.assert_allclose(
        res_mov.clim.sel(year=1982).values,
        ref_clim_1982["seas"].values,
        rtol=1e-12,
        atol=1e-12,
    )

    # Verify shrink boundary differs at edge year 1982 (uses [1982, 1982 + half])
    res_shrink = mx.compute_climatology(
        da,
        baseline=mx.MovingBaseline(window_years=window_years, center=True, boundary="shrink"),
    )
    ref_shrink_1982 = mx.compute_climatology(
        da,
        baseline=mx.FixedBaseline("1982-01-01", f"{1982 + half}-12-31"),
    )
    np.testing.assert_allclose(
        res_shrink["seas"].sel(year=1982).values,
        ref_shrink_1982["seas"].values,
        rtol=1e-12,
        atol=1e-12,
    )


def test_external_clim_data_cross_scenario_and_external_climatology_reuse():
    # Historical (1985-2004) -> detect on SSP future (2050-2069), fixing detectc.m:141-142 bug!
    hist = _synthetic_grid(n_years=20, start_year=1985, shape=(2, 2), slope_per_year=0.0, seed=31)
    future = _synthetic_grid(n_years=20, start_year=2050, shape=(2, 2), slope_per_year=0.0, seed=32) + 1.5

    res_cross = mx.detect(
        future,
        event="both",
        baseline=mx.FixedBaseline(start="1985-01-01", end="2004-12-31", clim_data=hist),
    )
    # Reuse standalone compute_climatology via ExternalClimatology -> identical events
    clim_ds = mx.compute_climatology(hist, baseline=mx.FixedBaseline("1985-01-01", "2004-12-31"))
    res_reuse = mx.detect(
        future,
        event="both",
        baseline=mx.ExternalClimatology.from_dataset(clim_ds),
    )
    pd.testing.assert_frame_equal(res_cross.events.to_pandas(), res_reuse.events.to_pandas())

    # Moving ExternalClimatology (year, doy) reuse
    mov_ds = mx.compute_climatology(
        hist,
        baseline=mx.MovingBaseline(window_years=11, center=True, boundary="clamp"),
    )
    res_mov_direct = mx.detect(
        hist,
        event="high",
        baseline=mx.MovingBaseline(window_years=11, center=True, boundary="clamp"),
    )
    res_mov_reuse = mx.detect(
        hist,
        event="high",
        baseline=mx.ExternalClimatology.from_dataset(mov_ds),
    )
    pd.testing.assert_frame_equal(res_mov_direct.events.to_pandas(), res_mov_reuse.events.to_pandas())


def test_fixed_threshold_salinity_extremes():
    # Salinity (PSU) high/low threshold test (e.g., freshening < 32.5, hypersaline > 36.5)
    dates = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    sal = np.full((dates.size, 2), 34.5, dtype=np.float64)
    sal[20:30, 0] = 37.0  # 10-day high salinity extreme at pixel 0
    sal[100:108, 1] = 31.8  # 8-day low salinity extreme at pixel 1
    da = xr.DataArray(sal, dims=("time", "station"), coords={"time": dates, "station": ["S1", "S2"]}, name="sos")

    res = mx.detect(
        da,
        event="both",
        baseline=mx.FixedThreshold(threshold_high=36.0, threshold_low=32.5, climatology=34.5),
        lower_bound=0.0,
    )
    df = res.events.to_pandas()
    assert len(df) == 2
    assert set(df["event_type"]) == {"high", "low"}
    assert df.loc[df["station"] == "S1", "duration"].iloc[0] == 10
    assert df.loc[df["station"] == "S2", "duration"].iloc[0] == 8


@pytest.mark.parametrize(
    "calendar_name,expected_year_len",
    [
        ("noleap", 365),
        ("360_day", 360),
        ("all_leap", 366),
        ("proleptic_gregorian", 366),
    ],
)
def test_cmip6_cftime_calendars(calendar_name, expected_year_len):
    n_years = 15
    t_idx = xr.date_range("2000-01-01", periods=n_years * expected_year_len, freq="D", calendar=calendar_name, use_cftime=True)
    rng = np.random.default_rng(77)
    vals = 20.0 + 3.0 * np.sin(2.0 * np.pi * np.arange(t_idx.size) / expected_year_len)
    noise = rng.normal(0.0, 0.5, size=t_idx.size)
    ar = np.zeros_like(noise)
    for t in range(1, t_idx.size):
        ar[t] = 0.8 * ar[t - 1] + noise[t]
    da = xr.DataArray(vals + ar, dims=("time",), coords={"time": t_idx}, name="tos")

    res = mx.detect(
        da,
        event="both",
        baseline=mx.FixedBaseline("2000-01-01", "2014-12-30"),
    )
    assert res.clim is not None
    assert res.clim.sizes["doy"] == expected_year_len
    assert res.metadata["calendar"] == calendar_name
    assert len(res.events) > 5


def test_monthly_frequency_jacox_mode():
    # 30 years of monthly SST with an explicit 3-month El Nino warm anomaly
    dates = pd.date_range("1991-01-01", "2020-12-01", freq="MS")
    rng = np.random.default_rng(55)
    months = dates.month.to_numpy()
    sst = 22.0 + 3.0 * np.sin(2.0 * np.pi * (months - 1) / 12.0) + rng.normal(0.0, 0.3, size=dates.size)
    # Inject a clear 3-month warm event in 2015-10..2015-12
    sst[24 * 12 + 9 : 24 * 12 + 12] += 3.5
    da = xr.DataArray(sst, dims=("time",), coords={"time": dates}, name="sst")

    res = mx.detect(da, event="high", freq="monthly")
    assert res.clim is not None
    assert res.clim.sizes["doy"] == 12
    df = res.events.to_pandas()
    assert len(df) >= 1
    hit = df[(df["date_start"] <= "2015-10-01") & (df["date_end"] >= "2015-12-01")]
    assert len(hit) == 1
    assert hit["duration"].iloc[0] >= 3
