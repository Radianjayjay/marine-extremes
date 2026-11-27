"""Focused regression tests covering all blocking & important findings from full code review."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marxtreme as mx
from marxtreme import _core
from marxtreme.calendar import date_to_ymd_key, dates_to_yyyymmdd
from marxtreme.io_formats import OutputSpec, _compute_bounded_chunks, open_file_writer


def _make_grid_da(
    start: str = "2000-01-01",
    end: str = "2004-12-31",
    shape: tuple[int, int] = (2, 3),
    seed: int = 11,
) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    nt = dates.size
    doy = dates.dayofyear.to_numpy()
    data = 18.0 + 3.0 * np.sin(2.0 * np.pi * doy / 365.25)[:, None, None] + rng.normal(0.0, 0.4, (nt, *shape))
    # Inject strong heatwave in June 2003
    s0 = int(np.where(dates == pd.Timestamp("2003-06-05"))[0][0])
    data[s0 : s0 + 12, :, :] += 5.0
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": dates, "lat": [10.0 + i for i in range(shape[0])], "lon": [120.0 + j for j in range(shape[1])]},
        name="sst",
    )


# ---------------------------------------------------------------------------
# B1: Atomic staging & inplace save(same_path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".nc", ".mat", ".zarr"])
def test_b1_inplace_save_same_path_preserves_data(tmp_path: Path, ext: str):
    da = _make_grid_da()
    p = tmp_path / f"inplace{ext}"
    r0 = mx.detect(da, baseline=("2000-01-01", "2002-12-31"), out=p)
    n_ev0 = len(r0.events)
    clim0 = r0.clim.values.copy()
    assert n_ev0 > 0
    r0.close()

    with mx.open(p) as r_lazy:
        r_lazy.save(p)

    with mx.open(p) as r_after:
        assert len(r_after.events) == n_ev0
        np.testing.assert_allclose(r_after.clim.values, clim0, equal_nan=True)
    # Zero leftover staging files
    leftovers = [f for f in tmp_path.iterdir() if ".tmp." in f.name]
    assert leftovers == []


def test_b1_failed_writer_preserves_existing_file(tmp_path: Path):
    da = _make_grid_da()
    p = tmp_path / "existing.nc"
    r = mx.detect(da, baseline=("2000-01-01", "2002-12-31"), out=p)
    orig_size = p.stat().st_size
    r.close()

    spec = OutputSpec(
        spatial_dims=("lat", "lon"),
        spatial_shape=(2, 3),
        spatial_coords={"lat": np.array([10.0, 11.0]), "lon": np.array([120.0, 121.0, 122.0])},
        det_time=da.coords["time"].values,
        write_clim=True,
        write_events=True,
    )
    w = open_file_writer(p, "netcdf", spec)
    # Abort without finalize -> existing file untouched, temp file removed
    w.close()
    assert p.stat().st_size == orig_size
    assert [f for f in tmp_path.iterdir() if ".tmp." in f.name] == []


# ---------------------------------------------------------------------------
# B2: Bounded chunk sizes on 4D large-time grids (strictly < 4 GiB HDF5 limit)
# ---------------------------------------------------------------------------


def test_b2_bounded_chunks_4d_large_time(tmp_path: Path):
    # Pure unit test on massive 4D shape (50, 180, 360, 15000) float64
    chunks = _compute_bounded_chunks((50, 180, 360, 15000), itemsize=8)
    assert int(np.prod(chunks)) * 8 <= 4 * 1024 * 1024

    # Live NetCDF4, MAT v7.3, and Zarr creation on 4D (depth=50, lat=4, lon=4, time=4000)
    import netCDF4 as nc4

    t = pd.date_range("2000-01-01", periods=4000, freq="D").to_numpy()
    spec = OutputSpec(
        spatial_dims=("depth", "lat", "lon"),
        spatial_shape=(50, 4, 4),
        spatial_coords={
            "depth": np.arange(50, dtype=np.float64),
            "lat": np.arange(4, dtype=np.float64),
            "lon": np.arange(4, dtype=np.float64),
        },
        det_time=t,
        write_intensity_ts=True,
        write_events=False,
    )
    nc_p = tmp_path / "chunk4d.nc"
    w_nc = open_file_writer(nc_p, "netcdf", spec)
    w_nc.finalize(None, None)
    with nc4.Dataset(str(nc_p), "r") as ds:
        c = ds.variables["intensity_ts"].chunking()
        assert int(np.prod(c)) * 8 <= 4 * 1024 * 1024

    for fname, fmt in (("chunk4d.mat", "mat"), ("chunk4d.zarr", "zarr")):
        w = open_file_writer(tmp_path / fname, fmt, spec)
        w.finalize(None, None)


# ---------------------------------------------------------------------------
# B3: File descriptor lifecycle & context manager & single-open
# ---------------------------------------------------------------------------


def test_b3_context_manager_and_no_fd_leak(tmp_path: Path):
    da = _make_grid_da()
    nc_p = tmp_path / "fd_test.nc"
    mat_p = tmp_path / "fd_test.mat"
    with mx.detect(da, baseline=("2000-01-01", "2002-12-31"), out=nc_p) as r:
        r.save(mat_p)

    if os.path.isdir("/proc/self/fd"):
        before = len(os.listdir("/proc/self/fd"))
        for _ in range(20):
            with mx.open(nc_p, events_in_memory=True) as r1:
                assert len(r1.events) > 0
            with mx.open(mat_p, events_in_memory=True) as r2:
                assert len(r2.events) > 0
            cat = mx.EventCatalog.open(nc_p)
            assert len(cat) > 0
        after = len(os.listdir("/proc/self/fd"))
        assert after - before <= 2


# ---------------------------------------------------------------------------
# B4: Rust PyO3 input validation (ValueError, never PanicException)
# ---------------------------------------------------------------------------


def test_b4_rust_ffi_rejects_bad_inputs_with_value_error():
    data = np.ones((2, 10), dtype=np.float64)
    years = np.full(10, 2000, dtype=np.int32)
    bad_doys = np.array([1, 2, 400, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint16)
    zero_doys = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint16)
    in_p = np.ones(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="year_len"):
        _core.FixedPlan(years, np.ones(10, dtype=np.uint16), in_p, 0, 0.9, 0.1, 5, 31, 0.0, "type7", "interp_59_61", "omit")
    with pytest.raises(ValueError, match="out-of-range DOY"):
        _core.FixedPlan(years, bad_doys, in_p, 366, 0.9, 0.1, 5, 31, 0.0, "type7", "interp_59_61", "omit")
    with pytest.raises(ValueError, match="out-of-range DOY"):
        _core.FixedPlan(years, zero_doys, in_p, 366, 0.9, 0.1, 5, 31, 0.0, "type7", "interp_59_61", "omit")

    plan = _core.CurvesPlan("high", 5, 2, 0, None, "flag", True, 1)
    seas = np.zeros((2, 10), dtype=np.float64)
    th = np.ones((2, 10), dtype=np.float64)
    tl = -np.ones((2, 10), dtype=np.float64)
    # Unsorted active_pixel_indices
    with pytest.raises(ValueError, match="strictly increasing"):
        plan.run_curves(data, seas, th, tl, np.array([1, 0], dtype=np.int64), 2)
    # Mismatched shape
    with pytest.raises(ValueError, match="shape"):
        plan.run_curves(data, np.zeros((1, 10), dtype=np.float64), th, tl, np.array([0, 1], dtype=np.int64), 2)


# ---------------------------------------------------------------------------
# B5: BoundPolicy Floor / Mask & NaN threshold variance safety
# ---------------------------------------------------------------------------


def test_b5_bound_policy_floor_rates_and_mask_gap_and_nan_thresh_var():
    n = 30
    dates = pd.date_range("2000-01-01", periods=n, freq="D")
    seas_da = xr.DataArray(np.full(n, 0.0), dims=("time",), coords={"time": dates})
    th_h_da = xr.DataArray(np.full(n, 10.0), dims=("time",), coords={"time": dates})
    th_l_da = xr.DataArray(np.full(n, -1.0), dims=("time",), coords={"time": dates})

    # Floor policy: raw value -5.0 floored at -1.8 -> peak anomaly is -1.8, onset uses floored prev day
    vals = np.full(n, -0.5, dtype=np.float64)
    vals[5:12] = -5.0
    vals[4] = -0.8  # above th_l (-1.0)
    r_floor = mx.detect(
        vals,
        time=dates,
        event="low",
        baseline=mx.ExternalClimatology(seas=seas_da, thresh_high=th_h_da, thresh_low=th_l_da),
        lower_bound=-1.8,
        bound_policy="floor",
    )
    ev_f = r_floor.events.to_pandas()
    assert ev_f["int_max"].iloc[0] == pytest.approx(-1.8)
    assert ev_f["rate_onset"].iloc[0] == pytest.approx(1.0)

    # Mask policy: gap day where baseline/threshold hits lower_bound is masked out from event mean/cum
    vals_m = np.full(n, 0.0, dtype=np.float64)
    vals_m[5:10] = -1.5
    vals_m[10] = -2.5
    vals_m[11:16] = -1.5
    th_l_mask = th_l_da.copy(deep=True)
    th_l_mask.values[10] = -1.8  # threshold hits lower_bound (-1.8) -> masked gap of length 1
    r_mask = mx.detect(
        vals_m,
        time=dates,
        event="low",
        baseline=mx.ExternalClimatology(seas=seas_da, thresh_high=th_h_da, thresh_low=th_l_mask),
        lower_bound=-1.8,
        bound_policy="mask",
    )
    ev_m = r_mask.events.to_pandas()
    assert ev_m["duration"].iloc[0] == 11
    assert ev_m["int_mean"].iloc[0] == pytest.approx(-1.5)  # masked day 10 (-2.5) excluded!

    # NaN threshold on a gap day inside merged event must NOT poison int_var
    vals_v = np.full(n, 20.0, dtype=np.float64)
    vals_v[5:10] = 25.0
    vals_v[10] = 21.0
    vals_v[11:16] = 25.0
    seas_v = xr.DataArray(np.full(n, 20.0), dims=("time",), coords={"time": dates})
    th_hv = xr.DataArray(np.full(n, 22.0), dims=("time",), coords={"time": dates})
    th_hv.values[10] = np.nan
    r_v = mx.detect(
        vals_v,
        time=dates,
        event="high",
        baseline=mx.ExternalClimatology(seas=seas_v, thresh_high=th_hv),
    )
    ev_v = r_v.events.to_pandas()
    assert np.isfinite(ev_v["int_var"].iloc[0])
    assert ev_v["int_var"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# B6: FixedThreshold & ExternalClimatology dim ordering, slicing, & clim alias
# ---------------------------------------------------------------------------


def test_b6_fixed_threshold_transposed_and_zonal_and_external_clim(tmp_path: Path):
    dates = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    nt = dates.size
    data = np.full((nt, 2, 3), 20.0, dtype=np.float64)
    data[50:60, 0, 2] = 25.0  # only cell (lat=0, lon=2) exceeds threshold 23
    data[50:60, 1, 0] = 25.0  # cell (lat=1, lon=0)
    da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": dates, "lat": [10.0, 20.0], "lon": [100.0, 110.0, 120.0]})

    # 1. Transposed (lon, lat) FixedThreshold
    th_grid = np.array([[23.0, 23.0, 23.0], [28.0, 28.0, 28.0]])  # (lat, lon): lat=0 thresh=23, lat=1 thresh=28
    th_transposed = xr.DataArray(th_grid.T, dims=("lon", "lat"), coords={"lon": da.lon, "lat": da.lat})
    r_tr = mx.detect(da, baseline=mx.FixedThreshold(threshold_high=th_transposed, climatology=20.0))
    assert len(r_tr.events.isel(lat=0, lon=2)) == 1
    assert len(r_tr.events.isel(lat=1, lon=0)) == 0

    # 2. 1D zonal (lat,) FixedThreshold broadcast across lon
    th_zonal = xr.DataArray([23.0, 28.0], dims=("lat",), coords={"lat": da.lat})
    r_zonal = mx.detect(da, baseline=mx.FixedThreshold(threshold_high=th_zonal, climatology=20.0))
    assert len(r_zonal.events.isel(lat=0, lon=2)) == 1
    assert len(r_zonal.events.isel(lat=1, lon=0)) == 0

    # 3. ExternalClimatology with (lat, lon, doy, year) dim order
    years = np.array([2000, 2001, 2002], dtype=np.int32)
    doys = np.arange(1, 367, dtype=np.int32)
    seas_yd = xr.DataArray(np.full((2, 3, 3, 366), 20.0), dims=("lat", "lon", "year", "doy"), coords={"lat": da.lat, "lon": da.lon, "year": years, "doy": doys})
    hi_yd = xr.DataArray(np.full((2, 3, 3, 366), 23.0), dims=("lat", "lon", "year", "doy"), coords={"lat": da.lat, "lon": da.lon, "year": years, "doy": doys})
    hi_yd.loc[dict(year=2000)] = 23.0
    hi_yd.loc[dict(year=2001)] = 30.0
    r_yd1 = mx.detect(da, baseline=mx.ExternalClimatology(seas=seas_yd, thresh_high=hi_yd))
    r_yd2 = mx.detect(
        da,
        baseline=mx.ExternalClimatology(
            seas=seas_yd.transpose("lat", "lon", "doy", "year"),
            thresh_high=hi_yd.transpose("lat", "lon", "doy", "year"),
        ),
    )
    pd.testing.assert_frame_equal(r_yd1.events.to_pandas(), r_yd2.events.to_pandas())

    # 4. ExternalClimatology with time dim + detect_period subset
    seas_t = xr.DataArray(np.full((2, 3, nt), 20.0), dims=("lat", "lon", "time"), coords={"lat": da.lat, "lon": da.lon, "time": dates})
    hi_t = xr.DataArray(np.full((2, 3, nt), 23.0), dims=("lat", "lon", "time"), coords={"lat": da.lat, "lon": da.lon, "time": dates})
    r_time = mx.detect(
        da,
        baseline=mx.ExternalClimatology(seas=seas_t, thresh_high=hi_t),
        detect_period=("2000-01-01", "2000-12-31"),
    )
    assert len(r_time.events) == 2

    # 5. Loading "clim" directly from a detect() .nc file
    det_nc = tmp_path / "det_clim.nc"
    gda = _make_grid_da()
    with mx.detect(gda, baseline=("2000-01-01", "2002-12-31"), out=det_nc):
        pass
    r_reuse = mx.detect(
        gda,
        baseline=mx.ExternalClimatology(seas=det_nc, thresh_high=det_nc),
    )
    assert len(r_reuse.events) > 0


# ---------------------------------------------------------------------------
# B7: CSV sel() with NaN land coords, string coords, & shuffled CSV
# ---------------------------------------------------------------------------


def test_b7_csv_nan_coords_string_coords_and_shuffled(tmp_path: Path):
    da = _make_grid_da(shape=(3, 2))
    # Make lat=0 completely land (NaN) -> CSV inferred lat coord has [NaN, 11.0, 12.0]
    da.values[:, 0, :] = np.nan
    csv_p = tmp_path / "events.csv"
    mx.detect(da, baseline=("2000-01-01", "2002-12-31"), out=csv_p)

    cat = mx.EventCatalog.open(csv_p)
    # Nearest to 11.0 must pick lat index 1 (NOT index 0 where coord is NaN)
    df_hit = cat.sel(lat=11.0, lon=120.0)
    assert len(df_hit) > 0
    assert (df_hit["lat_idx"] == 1).all()

    # Externally re-sort CSV by duration descending -> CSR offsets must still stay valid
    df_all = pd.read_csv(csv_p).sort_values("duration", ascending=False)
    shuffled_p = tmp_path / "shuffled.csv"
    df_all.to_csv(shuffled_p, index=False)
    cat_shuf = mx.EventCatalog.open(shuffled_p)
    df_cell = cat_shuf.isel(lat=2, lon=1)
    assert (df_cell["lat_idx"] == 2).all() and (df_cell["lon_idx"] == 1).all()

    # String spatial coordinate sel()
    dates = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    s_data = np.full((dates.size, 2), 20.0)
    s_data[100:110, 1] = 26.0
    da_str = xr.DataArray(s_data, dims=("time", "station"), coords={"time": dates, "station": ["S1", "S2"]})
    r_str = mx.detect(da_str, baseline=mx.FixedThreshold(threshold_high=23.0, climatology=20.0))
    assert len(r_str.events.sel(station="S2")) == 1
    assert len(r_str.events.sel(station="S1")) == 0


# ---------------------------------------------------------------------------
# B8: String coords + 0 events & detectc -> feather pixel_idx preservation
# ---------------------------------------------------------------------------


def test_b8_zero_events_string_coords_and_detectc_to_feather(tmp_path: Path):
    dates = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    da_str = xr.DataArray(
        np.full((dates.size, 2), 20.0),
        dims=("time", "station"),
        coords={"time": dates, "station": np.array(["A", "B"], dtype=object)},
    )
    for ext in (".nc", ".zarr", ".npz", ".parquet", ".feather", ".csv"):
        p = tmp_path / f"zero_str{ext}"
        with mx.detect(da_str, baseline=mx.FixedThreshold(threshold_high=25.0, climatology=20.0), out=p) as r:
            assert len(r.events) == 0

    mat_p = tmp_path / "zero_str.mat"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with mx.detect(da_str, baseline=mx.FixedThreshold(threshold_high=25.0, climatology=20.0), out=mat_p) as r:
            assert len(r.events) == 0
    assert any("skips string column" in str(w.message) for w in caught)

    # detectc -> reopen -> save to .feather -> reopen must keep non-zero offsets & events
    da = _make_grid_da()
    dc_p = tmp_path / "dc.mat"
    fth_p = tmp_path / "dc_conv.feather"
    r0 = mx.detect(da, baseline=("2000-01-01", "2002-12-31"), out=dc_p, mat_layout="detectc")
    n0 = len(r0.events)
    assert n0 > 0
    r0.save(fth_p)
    r_fth = mx.open(fth_p)
    assert len(r_fth.events) == n0
    assert int(r_fth.events.grid_offsets[-1]) == n0
    with pytest.raises(KeyError, match="detectc"):
        r0.events.filter(exclude_truncated=True)


# ---------------------------------------------------------------------------
# B9: .npz allow_pickle=False security & string coords round-trip
# ---------------------------------------------------------------------------


def test_b9_npz_forbids_pickle_and_roundtrips_strings(tmp_path: Path):
    # Craft an NPZ containing a pickled object array
    evil_p = tmp_path / "evil.npz"
    np.savez(evil_p, marxtreme_meta_json=np.array([{"evil": 1}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays cannot be loaded when allow_pickle=False"):
        mx.open(evil_p)

    dates = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    vals = np.full((dates.size, 2), 20.0)
    vals[60:70, 0] = 26.0
    da = xr.DataArray(vals, dims=("time", "station"), coords={"time": dates, "station": np.array(["ST_A", "ST_B"], dtype=object)})
    ok_p = tmp_path / "safe.npz"
    mx.detect(da, baseline=mx.FixedThreshold(threshold_high=23.0, climatology=20.0), out=ok_p)
    r = mx.open(ok_p)
    assert len(r.events.sel(station="ST_A")) == 1


# ---------------------------------------------------------------------------
# I1, I2, I4, I5, I6, I7: Validation, calendars, packed int16, 1D trend 0D
# ---------------------------------------------------------------------------


def test_i1_external_clim_data_validation():
    da = _make_grid_da(shape=(2, 3))
    da_wrong_shape = _make_grid_da(shape=(3, 2))
    with pytest.raises(ValueError, match="does not match"):
        mx.detect(da, baseline=mx.FixedBaseline(clim_data=da_wrong_shape))

    import cftime

    t360 = np.array([cftime.Datetime360Day(y, m, d) for y in range(2000, 2003) for m in range(1, 13) for d in range(1, 31)], dtype=object)
    da_360 = xr.DataArray(np.full((t360.size, 2, 3), 20.0), dims=("time", "lat", "lon"), coords={"time": t360, "lat": da.lat, "lon": da.lon})
    with pytest.raises(ValueError, match="incompatible"):
        mx.detect(da, baseline=mx.FixedBaseline(clim_data=da_360))


def test_i2_calendar_guards_post2262_and_360day_consistency(tmp_path: Path):
    # NaT rejected
    t_nat = np.array(["2000-01-01", "NaT", "2000-01-03"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="NaT"):
        mx.detect(np.ones(3), time=t_nat, baseline=mx.FixedThreshold(threshold_high=2.0))

    # Non-monotonic rejected
    t_rev = pd.to_datetime(["2000-01-02", "2000-01-01", "2000-01-03"])
    with pytest.raises(ValueError, match="strictly increasing"):
        mx.detect(np.ones(3), time=t_rev, baseline=mx.FixedThreshold(threshold_high=2.0))

    # Feb 29 in noleap rejected
    t_leap = pd.date_range("2000-02-27", "2000-03-02", freq="D")
    with pytest.raises(ValueError, match="February 29"):
        mx.detect(np.ones(t_leap.size), time=t_leap, calendar="noleap", baseline=mx.FixedThreshold(threshold_high=2.0))

    # 8-digit YYYYMMDD date boundary accepted; junk prefix rejected
    da = _make_grid_da()
    r8 = mx.detect(da, baseline=("20000101", "20021231"))
    assert len(r8.events) > 0
    with pytest.raises(ValueError, match="Could not parse date string"):
        mx.detect(da, baseline=("not-2000-01-01", "2002-12-31"))

    # Post-2262 datetime64[s] round-trips without overflow
    t_far = np.arange(np.datetime64("2295-01-01", "s"), np.datetime64("2297-01-01", "s"), np.timedelta64(1, "D"))
    vals_far = np.full(t_far.size, 20.0)
    vals_far[50:60] = 26.0
    nc_far = tmp_path / "far_2295.nc"
    with mx.detect(vals_far, time=t_far, baseline=mx.FixedThreshold(threshold_high=23.0, climatology=20.0), out=nc_far) as rf:
        assert len(rf.events) == 1
        assert str(rf.events.to_pandas()["date_start"].iloc[0]).startswith("2295-")

    # 360_day consistent string date dtype across cells with and without Feb 30
    import cftime

    t360 = np.array([cftime.Datetime360Day(y, m, d) for y in range(2000, 2003) for m in range(1, 13) for d in range(1, 31)], dtype=object)
    v360 = np.full((t360.size, 2), 20.0)
    v360[54:62, 0] = 26.0   # crosses Feb 25..Mar 2 including Feb 30
    v360[100:108, 1] = 26.0  # pure April (valid Gregorian dates too)
    da360 = xr.DataArray(v360, dims=("time", "site"), coords={"time": t360, "site": [0, 1]})
    nc360 = tmp_path / "c360.nc"
    with mx.detect(da360, baseline=mx.FixedThreshold(threshold_high=23.0, climatology=20.0), out=nc360) as r360:
        d0 = r360.events.isel(site=0)["date_start"].dtype
        d1 = r360.events.isel(site=1)["date_start"].dtype
        assert d0 == d1
        assert pd.api.types.is_string_dtype(d1)


def test_i4_cf_packed_int16_auto_float32(tmp_path: Path):
    da = _make_grid_da()
    nc_in = tmp_path / "packed_int16.nc"
    da.to_dataset(name="sst").to_netcdf(
        nc_in,
        encoding={"sst": {"dtype": "int16", "scale_factor": 0.01, "add_offset": 20.0, "_FillValue": -32768}},
    )
    r = mx.detect(nc_in, var="sst", baseline=("2000-01-01", "2002-12-31"), working_dtype="auto")
    assert r.metadata["working_dtype"] == "float32"


def test_i5_i6_i7_param_validation_event_both_bool_attr_and_1d_trend_0d(tmp_path: Path):
    da = _make_grid_da()
    # I5: invalid min_valid_fraction
    with pytest.raises(ValueError, match="min_valid_fraction"):
        mx.detect(da, min_valid_fraction=50.0)

    # I5: event="both" colliding High/Low events must not let Low gap clobber High peak category
    n = 20
    t20 = pd.date_range("2000-01-01", periods=n, freq="D")
    vals = np.full(n, 20.0)
    vals[3:5] = 25.0   # High part 1 (days 3-4)
    vals[5:7] = 15.0   # Low part 1 / High 2-day gap (days 5-6)
    vals[7:9] = 25.0   # High part 2 / Low 2-day gap (days 7-8)
    vals[9:11] = 15.0  # Low part 2 (days 9-10)
    res_b = mx.detect(
        vals,
        time=t20,
        event="both",
        baseline=mx.FixedThreshold(threshold_high=22.0, threshold_low=18.0, climatology=20.0),
        min_duration=2,
        max_gap=2,
        return_event_category_ts=True,
    )
    assert len(res_b.events) == 2
    assert int(res_b.event_category_ts.values[7]) > 0  # High exceed day NOT clobbered by Low gap
    assert int(res_b.daily_category_ts.values[7]) > 0

    # I5: pixel valid in clim period, all-NaN in detect_period still gets valid clim
    da_c = _make_grid_da()
    det_mask = da_c.coords["time"] >= pd.Timestamp("2003-01-01")
    da_c.values[det_mask, 0, 0] = np.nan
    r_c = mx.detect(da_c, baseline=("2000-01-01", "2002-12-31"), detect_period=("2003-01-01", "2004-12-31"))
    assert np.all(np.isfinite(r_c.clim.isel(lat=0, lon=0).values))

    # I6: bool attr in NetCDF writer
    nc_b = tmp_path / "bool_attr.nc"
    w_b = open_file_writer(nc_b, "netcdf", OutputSpec((), (), {}, write_events=False, attrs={"flag_bool": True}))
    w_b.finalize(None, None)

    # I7: 1D DetrendedBaseline trend is 0D scalar across all grid formats
    da1 = da.isel(lat=0, lon=0)
    r_mem = mx.detect(da1, baseline=mx.DetrendedBaseline(method="quadratic"))
    assert r_mem.trend["slope_per_step"].dims == ()
    for ext, kw in [
        (".nc", {}),
        (".mat", {"mat_layout": "v73"}),
        (".mat", {"mat_layout": "detectc"}),
        (".zarr", {}),
        (".npz", {}),
    ]:
        p = tmp_path / f"trend1d_{kw.get('mat_layout', ext.lstrip('.'))}{ext}"
        with mx.detect(da1, baseline=mx.DetrendedBaseline(method="quadratic"), out=p, **kw) as rd:
            assert rd.trend["slope_per_step"].dims == ()
            np.testing.assert_allclose(rd.trend["slope_per_step"].values, r_mem.trend["slope_per_step"].values)


# ---------------------------------------------------------------------------
# Round 2 regressions (Neon xhigh re-review + Carbon high independent review)
# ---------------------------------------------------------------------------


def test_r2_string_coords_with_events_no_sigsegv_zarr_csv_and_none_slot(tmp_path: Path):
    dates = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    vals = np.full((dates.size, 2), 20.0)
    vals[60:72, 0] = 26.0  # only S1 has an event; S2 has 0 events
    da = xr.DataArray(vals, dims=("time", "station"), coords={"time": dates, "station": ["S1", "S2"]})

    # All formats with events > 0 + string coords: NetCDF (no SIGSEGV!), Zarr (no StringDType crash!), CSV (keeps S2!)
    for ext in (".nc", ".zarr", ".npz", ".parquet", ".feather", ".csv"):
        p = tmp_path / f"str_events{ext}"
        with mx.detect(da, baseline=mx.FixedThreshold(threshold_high=23.0, climatology=20.0), out=p) as r:
            assert len(r.events) == 1
            for _ in range(5):
                df1 = r.events.sel(station="S1")
                df2 = r.events.sel(station="S2")
                df_all = r.events.to_pandas()
                assert len(df1) == 1 and str(df1["station"].iloc[0]) == "S1"
                assert len(df2) == 0
                assert len(df_all) == 1

    # Zero-event Zarr & CSV: isel/to_pandas & CSV->MAT export must not crash
    da0 = xr.DataArray(np.full((dates.size, 2), 20.0), dims=("time", "station"), coords={"time": dates, "station": ["S1", "S2"]})
    z0 = tmp_path / "zero_str_read.zarr"
    with mx.detect(da0, baseline=mx.FixedThreshold(threshold_high=25.0, climatology=20.0), out=z0) as rz0:
        assert len(rz0.events.isel(station=0)) == 0
        assert len(rz0.events.to_pandas()) == 0

    c0 = tmp_path / "zero_ev.csv"
    mx.detect(_make_grid_da() * 0.0 + 20.0, baseline=mx.FixedThreshold(threshold_high=25.0, climatology=20.0), out=c0)
    r_c0 = mx.open(c0)
    r_c0.events.to_mat(tmp_path / "zero_from_csv.mat")

    # CSV unpopulated string slot must NOT match sel(station="None")
    vals2 = np.full((dates.size, 2), 20.0)
    vals2[60:70, 1] = 26.0  # slot 0 has no events -> None in reopened CSV
    da2 = xr.DataArray(vals2, dims=("time", "station"), coords={"time": dates, "station": ["S0", "S1"]})
    c2 = tmp_path / "gap_str.csv"
    mx.detect(da2, baseline=mx.FixedThreshold(threshold_high=23.0, climatology=20.0), out=c2)
    cat_c2 = mx.EventCatalog.open(c2)
    with pytest.raises(KeyError):
        cat_c2.sel(station="None")


def test_r2_cross_2262_multitile_schema_and_lazy_consistency(tmp_path: Path):
    dates = np.arange(np.datetime64("2260-01-01", "s"), np.datetime64("2264-12-31", "s"), np.timedelta64(1, "D"))
    vals = np.full((dates.size, 3, 2), 20.0)
    # Tile 0 cell: event in 2260; middle cell: event crossing 2261-12-31 -> 2262-01-06; last cell: event in 2263
    i_2260 = int(np.where(dates == np.datetime64("2260-06-01", "s"))[0][0])
    i_cross = int(np.where(dates == np.datetime64("2261-12-28", "s"))[0][0])
    i_2263 = int(np.where(dates == np.datetime64("2263-06-01", "s"))[0][0])
    vals[i_2260 : i_2260 + 8, 0, 0] = 26.0
    vals[i_cross : i_cross + 10, 1, 0] = 26.0
    vals[i_2263 : i_2263 + 8, 2, 1] = 26.0
    da = xr.DataArray(vals, dims=("time", "lat", "lon"), coords={"time": dates, "lat": [0.0, 1.0, 2.0], "lon": [0.0, 1.0]})

    # Force multiple tiles via small memory_limit in-memory AND Parquet/NetCDF streaming
    r_mem = mx.detect(da, baseline=mx.FixedThreshold(threshold_high=23.0, climatology=20.0), memory_limit="200KB")
    df_mem = r_mem.events.to_pandas()
    assert len(df_mem) == 3
    assert df_mem["date_start"].dtype == df_mem["date_end"].dtype

    pq_p = tmp_path / "cross2262.parquet"
    r_pq = mx.detect(da, baseline=mx.FixedThreshold(threshold_high=23.0, climatology=20.0), memory_limit="200KB", out=pq_p)
    assert len(r_pq.events) == 3

    nc_p = tmp_path / "cross2262.nc"
    with mx.detect(da, baseline=mx.FixedThreshold(threshold_high=23.0, climatology=20.0), memory_limit="200KB", out=nc_p) as r_nc:
        d0 = r_nc.events.isel(lat=0, lon=0)["date_start"].dtype
        d2 = r_nc.events.isel(lat=2, lon=1)["date_start"].dtype
        assert d0 == d2


def test_r2_heatwaver_no_future_leak_and_shrink_window():
    # HeatwaveRSeries must NOT wrap 1980-01-01 to 2009-12-31 in MovingBaseline OR 1982-01-01 to 2012-01-01 in FixedBaseline
    dates = pd.date_range("1980-01-01", "2009-12-31", freq="D")
    vals = np.full(dates.size, 15.0)
    vals[dates >= pd.Timestamp("2009-01-01")] = 100.0
    da = xr.DataArray(vals, dims=("time",), coords={"time": dates})

    r_mov = mx.detect(
        da,
        baseline=mx.MovingBaseline(window_years=11, center=True, boundary="clamp"),
        compat="heatwaveR",
        detect_period=("1980-01-01", "1985-12-31"),
    )
    assert float(r_mov.clim.sel(year=1980, doy=1).values) == pytest.approx(15.0)

    r_fix = mx.detect(
        da,
        baseline=mx.FixedBaseline("1980-01-01", "2008-12-31"),
        compat="heatwaveR",
        smooth_width=1,
        detect_period=("1980-01-01", "1985-12-31"),
    )
    # 2009-01-01..05 buffer must NOT wrap into DOY 1 (1980-01-01 left tail)
    assert float(r_fix.clim.sel(doy=1).values) == pytest.approx(15.0)

    # MovingBaseline(window_years=31, boundary="shrink") on 20-year data (1990-2009):
    # year 1990 window must be [1990, 2005] (16 years), NOT [1990, 2000] (11 years)
    d20 = pd.date_range("1990-01-01", "2009-12-31", freq="D")
    v20 = np.where(d20.year <= 2000, 10.0, 20.0)
    c20 = mx.compute_climatology(
        xr.DataArray(v20, dims=("time",), coords={"time": d20}),
        baseline=mx.MovingBaseline(window_years=31, center=True, boundary="shrink"),
        smooth_width=1,
        window_half_width=0,
    )
    # For 1990: 11 years at 10.0 + 5 years (2001..2005) at 20.0 -> mean = (110 + 100) / 16 = 13.125
    assert float(c20["seas"].sel(year=1990, doy=15).values) == pytest.approx(13.125)


def test_r2_mat_v73_lazy_grids_and_to_dataset_to_netcdf(tmp_path: Path):
    from xarray.core import indexing

    da = _make_grid_da()
    mat_p = tmp_path / "lazy_v73.mat"
    with mx.detect(da, baseline=("2000-01-01", "2002-12-31"), out=mat_p) as r:
        assert isinstance(r.intensity_ts.variable._data, indexing.LazilyIndexedArray)
        # Negative-step slicing works cleanly on lazy BackendArray
        np.testing.assert_allclose(r.clim[::-1, :, :].values, r.clim.values[::-1, :, :], equal_nan=True)
        # B5: to_dataset().to_netcdf() works with default lower_bound=None
        ds_nc = tmp_path / "from_to_dataset.nc"
        r.to_dataset().to_netcdf(ds_nc)
        with xr.open_dataset(ds_nc) as check_ds:
            assert "clim" in check_ds


def test_r2_skewed_delta_floor_seas_and_mask_grids():
    n = 30
    dates = pd.date_range("2000-01-01", periods=n, freq="D")

    # Skewed low extreme: seas=32.67 < thresh_low=35.0 (delta > 0 for low event!)
    vals = np.full(n, 35.0)
    vals[5:12] = 5.0
    res_skew = mx.detect(
        vals,
        time=dates,
        event="low",
        baseline=mx.FixedThreshold(threshold_low=35.0, climatology=32.67),
        return_event_category_ts=True,
    )
    ev_sk = res_skew.events.to_pandas()
    assert int(ev_sk["category_max"].iloc[0]) == 1
    assert ev_sk["p_moderate"].iloc[0] == pytest.approx(1.0)
    assert int(res_skew.daily_category_ts.values[6]) == -1

    # BoundPolicy::Floor floors seas too: seas=-1.85, val=-1.90, lb=-1.80 -> floored anom=0.0 (never positive!)
    vals_f = np.full(n, -1.5)
    vals_f[5:12] = -1.90
    res_fl = mx.detect(
        vals_f,
        time=dates,
        event="low",
        baseline=mx.FixedThreshold(threshold_low=-1.75, climatology=-1.85),
        lower_bound=-1.80,
        bound_policy="floor",
    )
    assert res_fl.events.to_pandas()["int_max"].iloc[0] <= 0.0

    # BoundPolicy::Mask sets masked days to NaN / CATEGORY_MISSING (-128)
    th_l = xr.DataArray(np.full(n, -1.0), dims=("time",), coords={"time": dates})
    th_l.values[0:3] = -1.80  # masked
    res_m = mx.detect(
        np.full(n, -0.5),
        time=dates,
        event="low",
        baseline=mx.ExternalClimatology(
            seas=xr.DataArray(np.full(n, 0.0), dims=("time",), coords={"time": dates}),
            thresh_low=th_l,
        ),
        lower_bound=-1.80,
        bound_policy="mask",
        return_event_category_ts=True,
    )
    assert np.isnan(res_m.intensity_ts.values[0])
    assert int(res_m.daily_category_ts.values[0]) == mx.CATEGORY_MISSING
    assert int(res_m.daily_category_ts.values[10]) == 0


def test_r2_validations_5digit_year_julian_monthly_clim_active_and_parquet_csr(tmp_path: Path):
    da = _make_grid_da(shape=(2, 3))

    # Missing required thresholds rejected
    with pytest.raises(ValueError, match="pctile_high"):
        mx.detect(da, event="high", pctile_high=None)
    with pytest.raises(ValueError, match="threshold_high"):
        mx.detect(da, event="high", baseline=mx.FixedThreshold())
    with pytest.raises(ValueError, match="thresh_high"):
        mx.detect(da, event="high", baseline=mx.ExternalClimatology(seas=np.zeros((2, 3, 366))))

    # Oversized ExternalClimatology spatial dim rejected (no silent truncation)
    big_clim = xr.DataArray(np.full((5, 5, 366), 20.0), dims=("lat", "lon", "doy"))
    with pytest.raises(ValueError, match="length 5, expected 2"):
        mx.detect(da, baseline=mx.ExternalClimatology(seas=big_clim, thresh_high=big_clim + 2.0))

    # Mismatched FixedThreshold ndarray rejected even when shape happens to match a single tile
    with pytest.raises(ValueError, match="does not match spatial_shape"):
        mx.detect(da, baseline=mx.FixedThreshold(threshold_high=np.full((1, 1), 23.0)), memory_limit="20KB")

    # ISO string with fractional seconds & Z suffix accepted
    r_iso = mx.detect(da, baseline=("2000-01-01T00:00:00.000000000", "2002-12-31T23:59:59Z"))
    assert len(r_iso.events) > 0

    # freq="monthly" rejects daily timestamps in the same month
    with pytest.raises(ValueError, match="monthly"):
        mx.detect(da, freq="monthly")

    # 5-digit year dates_to_yyyymmdd must NOT truncate day
    assert int(dates_to_yyyymmdd(np.array(["10000-01-13"]))[0]) == 100000113

    # detrend_linear with < 2 valid points returns NaN (not raw un-detrended values)
    t3 = pd.date_range("2000-01-01", periods=3, freq="D")
    r_1pt = mx.detect(np.array([np.nan, 25.0, np.nan]), time=t3, baseline=mx.DetrendedBaseline(method="linear"))
    assert np.all(np.isnan(r_1pt.clim.values))
    assert np.isnan(float(r_1pt.trend["slope_per_step"].values))

    # Active mask when FixedBaseline is far from detect_period (non-union read) still computes clim
    d_long = pd.date_range("1990-01-01", "2019-12-31", freq="D")
    v_long = np.full((d_long.size, 2, 2), 20.0)
    v_long[d_long >= pd.Timestamp("2018-01-01"), 0, 0] = np.nan
    da_long = xr.DataArray(v_long, dims=("time", "lat", "lon"), coords={"time": d_long, "lat": [0.0, 1.0], "lon": [0.0, 1.0]})
    r_sep = mx.detect(da_long, baseline=("1990-01-01", "1992-12-31"), detect_period=("2018-01-01", "2019-12-31"))
    assert np.all(np.isfinite(r_sep.clim.isel(lat=0, lon=0).values))

    # compute_climatology honors base.clim_data
    da_alt = da + 10.0
    c_ext = mx.compute_climatology(da, baseline=mx.FixedBaseline("2000-01-01", "2002-12-31", clim_data=da_alt))
    assert float(c_ext["seas"].mean().values) > 25.0

    # Parquet filtered externally with stale marxtreme_csr falls back safely to pixel_idx
    import pyarrow.parquet as pq

    pq_orig = tmp_path / "orig.parquet"
    mx.detect(da, baseline=("2000-01-01", "2002-12-31"), out=pq_orig)
    pf = pq.ParquetFile(pq_orig)
    tbl = pq.read_table(pq_orig)
    assert tbl.num_rows > 1
    filtered = tbl.slice(0, 1)
    pq_stale = tmp_path / "stale_csr.parquet"
    with pq.ParquetWriter(pq_stale, filtered.schema) as pw:
        pw.write_table(filtered)
        pw.add_key_value_metadata(pf.metadata.metadata)  # stale CSR claiming > 1 rows!
    cat_stale = mx.EventCatalog.open(pq_stale)
    assert len(cat_stale) == 1
    assert int(cat_stale.grid_offsets[-1]) == 1


def test_r3_mask_high_both_zarr_pydate_fit_period_and_0d_input(tmp_path: Path):
    import datetime as dt

    # I1: BoundPolicy::Mask must NOT mask grid non-event days when event="high" or "both"
    n = 20
    t20 = pd.date_range("2000-01-01", periods=n, freq="D")
    vals = np.full(n, -1.60)
    vals[5:12] = -1.00
    for mode in ("high", "both"):
        res = mx.detect(
            vals,
            time=t20,
            event=mode,
            baseline=mx.FixedThreshold(threshold_high=-1.40, threshold_low=-1.80, climatology=-1.80),
            lower_bound=-1.80,
            bound_policy="mask",
            return_event_category_ts=True,
        )
        assert len(res.events) == 1
        assert float(res.intensity_ts.values[0]) == 0.0
        assert int(res.daily_category_ts.values[0]) == 0

    # I2: Python datetime.date time axis writes cleanly to .zarr
    py_dates = np.array([dt.date(2000, 1, 1) + dt.timedelta(days=i) for i in range(400)], dtype=object)
    v = np.full(400, 20.0)
    v[50:60] = 26.0
    zp = tmp_path / "pydate.zarr"
    with mx.detect(v, time=py_dates, baseline=mx.FixedThreshold(threshold_high=23.0, climatology=20.0), out=zp) as rz:
        assert len(rz.events) == 1

    # Nits: empty fit_period raises ValueError, 0D array raises ValueError, 5-digit np.datetime64 bound works
    da = _make_grid_da()
    with pytest.raises(ValueError, match="fit_period"):
        mx.detect(da, baseline=mx.DetrendedBaseline(fit_period=("1950-01-01", "1955-12-31")))
    with pytest.raises(ValueError, match="fit_period"):
        mx.compute_climatology(da, baseline=mx.DetrendedBaseline(fit_period=("1950-01-01", "1955-12-31")))
    with pytest.raises(ValueError, match="0D scalar"):
        mx.detect(np.array(20.0), time=pd.date_range("2000-01-01", periods=1))

    assert date_to_ymd_key(np.datetime64("10000-01-13"), is_end=False) == 100000113


def test_arch_fused_detrend_f32_multitile_and_eager_year_doy_and_disk_chunk_save(tmp_path: Path):
    # Fused 1-trip DetrendedBaseline on f32 + multi-tile matches f64 closely (no f64->f32->f64 round-trip)
    da = _make_grid_da(shape=(3, 2))
    r64 = mx.detect(da, baseline=mx.DetrendedBaseline(method="quadratic"), working_dtype="float64")
    r32 = mx.detect(da, baseline=mx.DetrendedBaseline(method="quadratic"), working_dtype="float32", memory_limit="200KB")
    assert len(r32.events) == len(r64.events)
    np.testing.assert_allclose(r32.trend["slope_per_step"].values, r64.trend["slope_per_step"].values, rtol=1e-4, atol=1e-6)

    # ExternalClimatology year_doy missing years fails fast at prepare time even on all-NaN land grid
    da_land = da * np.nan
    seas_yd = xr.DataArray(
        np.full((3, 2, 1, 366), 20.0),
        dims=("lat", "lon", "year", "doy"),
        coords={"lat": da.lat, "lon": da.lon, "year": [2000], "doy": np.arange(1, 367, dtype=np.int32)},
    )
    with pytest.raises(ValueError, match="missing detection years"):
        mx.detect(da_land, baseline=mx.ExternalClimatology(seas=seas_yd, thresh_high=seas_yd + 2.0))

    # Disk-backed EventStore iter_chunks save round-trip (.nc -> .zarr -> .parquet)
    nc_p, zarr_p, pq_p = tmp_path / "s1.nc", tmp_path / "s2.zarr", tmp_path / "s3.parquet"
    with mx.detect(da, baseline=("2000-01-01", "2002-12-31"), out=nc_p) as r_nc:
        r_nc.save(zarr_p)
    with mx.open(zarr_p) as r_z:
        r_z.save(pq_p)
    cat_pq = mx.EventCatalog.open(pq_p)
    assert len(cat_pq) > 0
    pd.testing.assert_frame_equal(cat_pq.to_pandas(), mx.EventCatalog.open(nc_p).to_pandas())


def test_r4_post_refactor_review_regressions(tmp_path: Path):
    import cftime

    from marxtreme.calendar import cf_days_to_time

    # B1: FixedPlan det_doys=None with invalid det_slice raises ValueError (never PanicException)
    years = np.full(10, 2000, dtype=np.int32)
    doys = np.arange(1, 11, dtype=np.uint16)
    in_p = np.ones(10, dtype=np.uint8)
    with pytest.raises(ValueError, match="det_slice"):
        _core.FixedPlan(
            years, doys, in_p, 366, 0.9, 0.1, 5, 31, 0.0, "type7", "interp_59_61", "omit",
            det_doys=None, detrend_order="linear", fit_mask=in_p,
            clim_slice_start=0, clim_slice_stop=10, det_slice_start=5, det_slice_stop=20,
        )
    with pytest.raises(ValueError, match="det_slice"):
        _core.FixedPlan(
            years, doys, in_p, 366, 0.9, 0.1, 5, 31, 0.0, "type7", "interp_59_61", "omit",
            det_doys=None, detrend_order="linear", fit_mask=in_p,
            clim_slice_start=0, clim_slice_stop=10, det_slice_start=0, det_slice_stop=5,
        )

    # B2: Fortran-contiguous 2D arrays rejected at PyO3 boundary (no silent row/col corruption)
    c_arr = np.vstack([np.full(10, 10.0), np.full(10, 50.0)])
    f_arr = np.asfortranarray(c_arr)
    fp = _core.FixedPlan(years, doys, in_p, 366, 0.9, 0.1, 5, 31, 0.0, "type7", "interp_59_61", "omit")
    with pytest.raises(ValueError, match="C-contiguous"):
        fp.compute_climatology(f_arr)
    cp = _core.CurvesPlan("high", 5, 2, 0, None, "flag", True, 1)
    with pytest.raises(ValueError, match="C-contiguous"):
        cp.run_curves(f_arr, np.zeros((2, 10)), np.ones((2, 10)), -np.ones((2, 10)), np.array([0, 1], dtype=np.int64), 2)
    with pytest.raises(ValueError, match="C-contiguous"):
        cp.run_curves(c_arr, np.asfortranarray(np.zeros((2, 10))), np.ones((2, 10)), -np.ones((2, 10)), np.array([0, 1], dtype=np.int64), 2)

    # B3: Failed _PreparedFixed with external clim_data file does not leak file descriptors
    da = _make_grid_da(shape=(2, 2))
    ext_nc = tmp_path / "ext_clim.nc"
    da.to_dataset(name="sst").to_netcdf(ext_nc)
    if os.path.isdir("/proc/self/fd"):
        before_fd = len(os.listdir("/proc/self/fd"))
        caught_excs = []
        for _ in range(10):
            try:
                mx.detect(da, baseline=mx.FixedBaseline("1900-01-01", "1905-12-31", clim_data=ext_nc, clim_var="sst"))
            except ValueError as exc:
                caught_excs.append(exc)
        after_fd = len(os.listdir("/proc/self/fd"))
        assert len(caught_excs) == 10
        assert after_fd - before_fd <= 2

    # B4: ExternalClimatology year_doy aligns each array by its own year coordinate labels & rejects empty year
    dates3 = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    v3 = np.full((dates3.size, 1, 1), 20.0)
    v3[50:60, 0, 0] = 25.0  # year 2000 anomaly
    da3 = xr.DataArray(v3, dims=("time", "lat", "lon"), coords={"time": dates3, "lat": [0.0], "lon": [0.0]})
    doys_366 = np.arange(1, 367, dtype=np.int32)
    seas_fwd = xr.DataArray(np.full((1, 1, 3, 366), 20.0), dims=("lat", "lon", "year", "doy"), coords={"lat": [0.0], "lon": [0.0], "year": [2000, 2001, 2002], "doy": doys_366})
    hi_rev = xr.DataArray(np.full((1, 1, 3, 366), 30.0), dims=("lat", "lon", "year", "doy"), coords={"lat": [0.0], "lon": [0.0], "year": [2002, 2001, 2000], "doy": doys_366})
    hi_rev.loc[dict(year=2000)] = 23.0  # year 2000 threshold is 23.0 at index 2 (reversed!)
    r_rev = mx.detect(da3, baseline=mx.ExternalClimatology(seas=seas_fwd, thresh_high=hi_rev))
    assert len(r_rev.events) == 1
    empty_yd = xr.DataArray(np.empty((1, 1, 0, 366)), dims=("lat", "lon", "year", "doy"), coords={"lat": [0.0], "lon": [0.0], "year": np.array([], dtype=np.int32), "doy": doys_366})
    with pytest.raises(ValueError, match="cannot be empty"):
        mx.detect(da3, baseline=mx.ExternalClimatology(seas=empty_yd, thresh_high=empty_yd))

    # I1: cf_days_to_time respects non-1970 units and pre-1582 standard/gregorian Julian dates
    t_1900 = cf_days_to_time(np.array([0.0, 1.0]), "days since 1900-01-01 00:00:00", "proleptic_gregorian")
    assert (t_1900[0].year, t_1900[0].month, t_1900[0].day) == (1900, 1, 1)
    t_1500 = np.array([cftime.DatetimeGregorian(1500, 2, 27) + __import__("datetime").timedelta(days=i) for i in range(15)], dtype=object)
    v_1500 = np.full(15, 20.0)
    v_1500[1:8] = 26.0
    mat_1500 = tmp_path / "g1500.mat"
    with mx.detect(v_1500, time=t_1500, baseline=mx.FixedThreshold(threshold_high=23.0, climatology=20.0), out=mat_1500) as r_1500:
        t_out = r_1500.intensity_ts.coords["time"].values
        assert (t_out[2].year, t_out[2].month, t_out[2].day) == (1500, 2, 29)

    # I2: cftime array with np.float32("nan") raises clean ValueError; bool date bound rejected
    t_bad = np.array([cftime.DatetimeNoLeap(2000, 1, 1), np.float32("nan")], dtype=object)
    with pytest.raises(ValueError, match="missing/NaT"):
        mx.detect(np.ones(2), time=t_bad, baseline=mx.FixedThreshold(threshold_high=2.0))
    with pytest.raises(TypeError, match="bool"):
        date_to_ymd_key(True)

    # I3: huge window_years in MovingBaseline(boundary="shrink") must not wrap negative
    c_huge = mx.compute_climatology(
        da.isel(lat=0, lon=0),
        baseline=mx.MovingBaseline(window_years=(1 << 31) + 10, center=True, boundary="shrink"),
        smooth_width=1,
        window_half_width=0,
    )
    assert np.all(np.isfinite(c_huge["seas"].values))

    # I4: CSV reopen keeps uniform string date dtype across date_start/peak/end on julian 1900-02-29
    t_jul = np.array([cftime.DatetimeJulian(1900, 2, 25) + __import__("datetime").timedelta(days=i) for i in range(15)], dtype=object)
    v_jul = np.full(15, 20.0)
    v_jul[1:9] = 26.0
    v_jul[4] = 30.0  # peak on 1900-02-29
    csv_jul = tmp_path / "jul1900.csv"
    mx.detect(v_jul, time=t_jul, baseline=mx.FixedThreshold(threshold_high=23.0, climatology=20.0), out=csv_jul)
    df_jul = mx.EventCatalog.open(csv_jul).to_pandas()
    assert df_jul["date_start"].dtype == df_jul["date_peak"].dtype == df_jul["date_end"].dtype
    assert pd.api.types.is_string_dtype(df_jul["date_start"].dtype)

    # I5: DetectResult.save() handles user-transposed grid DataArrays cleanly
    r_tr = mx.detect(da, baseline=("2000-01-01", "2002-12-31"))
    r_tr.clim = r_tr.clim.transpose("doy", "lon", "lat")
    nc_tr = tmp_path / "transposed_save.nc"
    r_tr.save(nc_tr)
    with mx.open(nc_tr) as r_tr_open:
        np.testing.assert_allclose(r_tr_open.clim.values, r_tr.clim.transpose("lat", "lon", "doy").values, equal_nan=True)

    # I6 & Nit1 & Nit3: pure-land eager shape validation, finite lower_bound, 0-length spatial MAT save
    da_land = da * np.nan
    with pytest.raises(ValueError, match="does not match spatial_shape"):
        mx.detect(da_land, baseline=mx.FixedThreshold(threshold_high=np.ones((9, 9))))
    with pytest.raises(ValueError, match="does not match data spatial_shape"):
        mx.detect(da_land, baseline=mx.ExternalClimatology(seas=np.ones((9, 9, 366)), thresh_high=np.ones((9, 9, 366))))
    with pytest.raises(ValueError, match="lower_bound must be finite"):
        mx.detect(da, lower_bound=float("nan"))
    da_zero = xr.DataArray(np.empty((10, 0, 2)), dims=("time", "lat", "lon"), coords={"time": pd.date_range("2000-01-01", periods=10), "lat": [], "lon": [0.0, 1.0]})
    with mx.detect(da_zero, baseline=mx.FixedThreshold(threshold_high=23.0), out=tmp_path / "zero_sp.mat") as r_zsp:
        assert len(r_zsp.events) == 0


