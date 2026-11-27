"""Tests for strictly single-file outputs (.nc, .mat v7.3, .mat detectc, .npz, .parquet, .feather, .csv)
and real MATLAB R2025b verification via load(), matfile(), and ncread().
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marxtreme as mx
from marxtreme import CATEGORY_MISSING, DetrendedBaseline, DetectResult, EventCatalog, FixedBaseline, MovingBaseline

MATLAB_BIN = Path("/usr/local/MATLAB/R2025b/bin/matlab")


def _make_synthetic_grid(
    *,
    n_years: int = 8,
    lat: int = 4,
    lon: int = 5,
    noon_timestamps: bool = False,
    seed: int = 123,
) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    start = "2010-01-01 12:00:00" if noon_timestamps else "2010-01-01"
    end = f"{2010 + n_years - 1}-12-31 12:00:00" if noon_timestamps else f"{2010 + n_years - 1}-12-31"
    times = pd.date_range(start, end, freq="D")
    n_t = len(times)
    doy = times.dayofyear.to_numpy()
    seasonal = 16.0 + 4.5 * np.sin(2.0 * np.pi * (doy - 30) / 365.25)

    data = np.empty((n_t, lat, lon), dtype=np.float64)
    for i in range(lat):
        for j in range(lon):
            if i == 0:
                # Entire first lat row is land (exercises pure-land tile branch)
                data[:, i, j] = np.nan
            else:
                noise = rng.normal(0.0, 0.7, size=n_t)
                series = seasonal + 0.5 * i - 0.3 * j + noise
                series[120:132] += 4.2
                series[450:462] -= 4.0
                series[900:910] += 3.8
                data[:, i, j] = series

    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": times,
            "lat": np.linspace(-44.0, -41.0, lat),
            "lon": np.linspace(145.0, 149.0, lon),
        },
        name="sst",
        attrs={"units": "degC"},
    )


@pytest.mark.parametrize(
    "filename,mat_layout,has_grids,disk_backed",
    [
        ("run.nc", "v73", True, True),
        ("run_v73.mat", "v73", True, True),
        ("run_detectc.mat", "detectc", True, False),
        ("run.zarr", "v73", True, True),
        ("run.npz", "v73", True, False),
        ("events.parquet", "v73", False, True),
        ("events.feather", "v73", False, False),
        ("events.csv", "v73", False, False),
    ],
)
def test_strictly_single_file_detect_and_reopen(
    tmp_path: Path,
    filename: str,
    mat_layout: str,
    has_grids: bool,
    disk_backed: bool,
) -> None:
    da = _make_synthetic_grid(noon_timestamps=True)
    ref = mx.detect(
        da,
        event="both",
        baseline=FixedBaseline("2010-01-01", "2015-12-31"),
        return_event_category_ts=True,
    )

    sub = tmp_path / f"case_{filename}_{mat_layout}"
    sub.mkdir()
    out_file = sub / filename

    res = mx.detect(
        da,
        event="both",
        baseline=FixedBaseline("2010-01-01", "2015-12-31"),
        return_event_category_ts=True,
        out=out_file,
        mat_layout=mat_layout,
        memory_limit="2MB",  # force multiple spatial tiles including pure-land tile
    )

    # STRICT SINGLE-TARGET INVARIANT: exactly 1 entry inside sub/, zero sidecars!
    entries = list(sub.iterdir())
    assert entries == [out_file]
    if filename.endswith(".zarr"):
        assert out_file.is_dir()
    else:
        assert out_file.is_file()
    assert res.out_path == out_file

    reopened = mx.open(out_file)
    assert reopened.events is not None
    assert reopened.events.is_disk_backed == disk_backed
    assert len(reopened.events) == len(ref.events) > 0

    # O(1) CSR cell lookup matches reference
    ref_cell = ref.events.isel(lat=2, lon=3)
    got_cell = reopened.events.isel(lat=2, lon=3)
    assert len(got_cell) == len(ref_cell) > 0
    np.testing.assert_array_equal(got_cell["duration"].to_numpy(), ref_cell["duration"].to_numpy())
    np.testing.assert_allclose(got_cell["int_max"].to_numpy(), ref_cell["int_max"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(got_cell["int_mean"].to_numpy(), ref_cell["int_mean"].to_numpy(), atol=1e-12)
    np.testing.assert_array_equal(got_cell["category_max"].to_numpy(), ref_cell["category_max"].to_numpy())
    np.testing.assert_array_equal(
        got_cell["event_type"].astype(str).to_numpy(),
        ref_cell["event_type"].astype(str).to_numpy(),
    )
    np.testing.assert_array_equal(
        pd.to_datetime(got_cell["date_start"]).to_numpy(dtype="datetime64[D]"),
        pd.to_datetime(ref_cell["date_start"]).to_numpy(dtype="datetime64[D]"),
    )

    if filename != "events.csv":
        np.testing.assert_array_equal(reopened.events.count_map().values, ref.events.count_map().values)

    if not has_grids:
        assert reopened.clim is None
        assert reopened.intensity_ts is None
        assert reopened.daily_category_ts is None
        return

    # Full gridded parity
    np.testing.assert_allclose(reopened.clim.values, ref.clim.values, rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(reopened.thresh_high.values, ref.thresh_high.values, rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(reopened.thresh_low.values, ref.thresh_low.values, rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(reopened.intensity_ts.values, ref.intensity_ts.values, rtol=0, atol=1e-12, equal_nan=True)

    assert reopened.daily_category_ts.dtype == np.int8
    assert reopened.event_category_ts.dtype == np.int8
    np.testing.assert_array_equal(reopened.daily_category_ts.values, ref.daily_category_ts.values)
    np.testing.assert_array_equal(reopened.event_category_ts.values, ref.event_category_ts.values)
    assert np.all(reopened.daily_category_ts.values[0, :, :] == CATEGORY_MISSING)
    assert np.all(reopened.event_category_ts.values[0, :, :] == CATEGORY_MISSING)

    # Noon timestamps (12:00:00) survive roundtrip intact
    np.testing.assert_array_equal(
        reopened.intensity_ts.coords["time"].values.astype("datetime64[s]"),
        ref.intensity_ts.coords["time"].values.astype("datetime64[s]"),
    )

    # NetCDF and Zarr stores are directly readable with plain xarray (root grids)
    if filename == "run.nc":
        with xr.open_dataset(out_file) as ds_root:
            assert "clim" in ds_root and "intensity_ts" in ds_root and "events" not in ds_root
        with xr.open_dataset(out_file, group="events") as ds_ev:
            assert "int_max" in ds_ev and "grid_offsets" in ds_ev
            assert ds_ev.sizes["event"] == len(ref.events)
    if filename == "run.zarr":
        with xr.open_zarr(out_file, consolidated=False) as ds_root:
            assert "clim" in ds_root and "intensity_ts" in ds_root and ds_root["daily_category_ts"].dtype == np.int8


def test_result_save_and_catalog_save_single_files(tmp_path: Path) -> None:
    da = _make_synthetic_grid()
    res = mx.detect(
        da,
        event="high",
        baseline=DetrendedBaseline(method="linear", start="2010-01-01", end="2015-12-31"),
        return_event_category_ts=True,
    )

    for fname, lay in [("s.nc", "v73"), ("s_v73.mat", "v73"), ("s_det.mat", "detectc"), ("s.zarr", "v73"), ("s.npz", "v73")]:
        d = tmp_path / f"save_{fname}"
        d.mkdir()
        p = res.save(d / fname, mat_layout=lay)
        assert list(d.iterdir()) == [p]
        r2 = DetectResult.open(p)
        assert r2.trend is not None
        np.testing.assert_allclose(r2.trend["slope_per_step"].values, res.trend["slope_per_step"].values, equal_nan=True)
        np.testing.assert_allclose(r2.clim.values, res.clim.values, equal_nan=True)
        assert len(r2.events) == len(res.events)

    # EventCatalog.save() across all formats
    for fname, lay in [
        ("c.parquet", "v73"),
        ("c.feather", "v73"),
        ("c.csv", "v73"),
        ("c.nc", "v73"),
        ("c_v73.mat", "v73"),
        ("c_det.mat", "detectc"),
        ("c.zarr", "v73"),
        ("c.npz", "v73"),
    ]:
        d = tmp_path / f"cat_{fname}"
        d.mkdir()
        p = res.events.save(d / fname, mat_layout=lay)
        assert list(d.iterdir()) == [p]
        c2 = EventCatalog.open(p)
        assert len(c2) == len(res.events)
        np.testing.assert_allclose(
            c2.isel(lat=1, lon=2)["int_cum"].to_numpy(),
            res.events.isel(lat=1, lon=2)["int_cum"].to_numpy(),
        )


def test_zero_events_1d_moving_4d_and_cftime_single_files(tmp_path: Path) -> None:
    # 1. Zero-event runs across ALL 7 single-file formats
    da = _make_synthetic_grid(n_years=6, lat=2, lon=2)
    da_flat = xr.full_like(da, 20.0)
    for fname, lay in [
        ("z.nc", "v73"),
        ("z_v73.mat", "v73"),
        ("z_det.mat", "detectc"),
        ("z.zarr", "v73"),
        ("z.npz", "v73"),
        ("z.parquet", "v73"),
        ("z.feather", "v73"),
        ("z.csv", "v73"),
    ]:
        p = tmp_path / fname
        r = mx.detect(
            da_flat,
            event="high",
            baseline=FixedBaseline("2010-01-01", "2014-12-31"),
            out=p,
            mat_layout=lay,
        )
        assert p.exists()
        assert len(r.events) == 0
        if fname != "z.csv":
            assert r.events.isel(lat=0, lon=0).empty

    # 2. 1D time series with DetrendedBaseline -> .nc, .mat (v73 & detectc), .zarr, .npz
    s1d = da.isel(lat=1, lon=1).drop_vars(["lat", "lon"])
    r1d_mem = mx.detect(s1d, event="high", baseline=DetrendedBaseline(method="linear", start="2010-01-01", end="2014-12-31"))
    for fname, lay in [("d1.nc", "v73"), ("d1_v73.mat", "v73"), ("d1_det.mat", "detectc"), ("d1.zarr", "v73"), ("d1.npz", "v73")]:
        r1d = mx.detect(
            s1d,
            event="high",
            baseline=DetrendedBaseline(method="linear", start="2010-01-01", end="2014-12-31"),
            out=tmp_path / fname,
            mat_layout=lay,
        )
        assert len(r1d.events) == len(r1d_mem.events) > 0
        np.testing.assert_allclose(r1d.clim.values, r1d_mem.clim.values, equal_nan=True)
        assert r1d.trend is not None

    # 3. MovingBaseline (year, doy) -> .nc, .mat (v73), .zarr, .npz
    r_mov_mem = mx.detect(da, event="high", baseline=MovingBaseline(window_years=3, boundary="clamp"))
    for fname in ("mov.nc", "mov.mat", "mov.zarr", "mov.npz"):
        r_mov = mx.detect(
            da,
            event="high",
            baseline=MovingBaseline(window_years=3, boundary="clamp"),
            out=tmp_path / fname,
        )
        assert r_mov.clim.dims == ("lat", "lon", "year", "doy")
        np.testing.assert_allclose(r_mov.clim.values, r_mov_mem.clim.values, equal_nan=True)

    # 4. 4D (time, depth, lat, lon) -> .nc, .mat (v73 & detectc), .zarr
    da4 = xr.concat([da, da - 2.0], dim=xr.DataArray([5.0, 50.0], dims=["depth"], name="depth")).transpose(
        "time", "depth", "lat", "lon"
    )
    r4 = mx.detect(da4, event="high", baseline=FixedBaseline("2010-01-01", "2014-12-31"))
    p4_det = r4.to_mat(tmp_path / "four_d_detectc.mat", layout="detectc")
    p4_v73 = r4.to_mat(tmp_path / "four_d_v73.mat", layout="v73")
    p4_nc = r4.save(tmp_path / "four_d.nc")
    p4_zarr = r4.save(tmp_path / "four_d.zarr")
    assert p4_det.is_file() and p4_v73.is_file() and p4_nc.is_file() and p4_zarr.is_dir()
    assert mx.open(p4_nc).clim.dims == ("depth", "lat", "lon", "doy")
    assert mx.open(p4_zarr).clim.dims == ("depth", "lat", "lon", "doy")

    # 5. CMIP6 360_day cftime calendar roundtrip through .nc, .mat, .zarr
    cftime = pytest.importorskip("cftime")
    t360 = xr.date_range("2010-01-01", "2015-12-30", freq="D", calendar="360_day", use_cftime=True)
    rng = np.random.default_rng(77)
    arr360 = 18.0 + 3.0 * np.sin(2.0 * np.pi * np.arange(len(t360)) / 360.0) + rng.normal(0, 0.5, len(t360))
    arr360[50:62] += 4.0  # spans Feb 21..Mar 2 including Feb 30 in 360_day!
    da360 = xr.DataArray(arr360, dims=("time",), coords={"time": t360}, name="tos")
    for fname in ("cmip_360.nc", "cmip_360.mat", "cmip_360.zarr"):
        r360 = mx.detect(da360, event="high", out=tmp_path / fname)
        assert r360.clim.sizes["doy"] == 360
        assert len(r360.events) >= 1


def test_invalid_formats_and_directory_rejected(tmp_path: Path) -> None:
    da = _make_synthetic_grid(n_years=4, lat=2, lon=2)
    with pytest.raises(ValueError, match="Cannot infer output format"):
        mx.detect(da, out=tmp_path / "bad_no_ext")
    with pytest.raises(ValueError, match="Unsupported out_format"):
        mx.detect(da, out=tmp_path / "out.xlsx", out_format="xlsx")
    with pytest.raises(ValueError, match="Unsupported mat_layout"):
        mx.detect(da, out=tmp_path / "out.mat", mat_layout="unknown")
    with pytest.raises(IsADirectoryError):
        mx.open(tmp_path)


@pytest.mark.skipif(not MATLAB_BIN.exists(), reason="Local MATLAB R2025b not installed")
def test_matlab_r2025b_loads_single_file_mat_and_nc(tmp_path: Path) -> None:
    """Verify real MATLAB R2025b loads every single-file .mat (v7.3 & v5 detectc) AND .nc output."""
    da = _make_synthetic_grid(n_years=7, lat=4, lon=5)
    base = DetrendedBaseline(method="linear", start="2010-01-01", end="2014-12-31")
    mem = mx.detect(da, event="both", baseline=base, return_event_category_ts=True)

    f_v73 = tmp_path / "single_v73.mat"
    mx.detect(da, event="both", baseline=base, return_event_category_ts=True, out=f_v73, mat_layout="v73", memory_limit="2MB")

    f_det = tmp_path / "single_detectc.mat"
    mx.detect(da, event="both", baseline=base, return_event_category_ts=True, out=f_det, mat_layout="detectc", memory_limit="2MB")

    f_nc = tmp_path / "single.nc"
    mx.detect(da, event="both", baseline=base, return_event_category_ts=True, out=f_nc, memory_limit="2MB")

    f_cat_v73 = mem.events.to_mat(tmp_path / "cat_v73.mat", layout="v73")

    da_zero = xr.full_like(da, 18.0)
    f_zero_v73 = tmp_path / "zero_v73.mat"
    mx.detect(da_zero, event="high", baseline=FixedBaseline("2010-01-01", "2014-12-31"), out=f_zero_v73, mat_layout="v73")
    f_zero_det = tmp_path / "zero_det.mat"
    mx.detect(da_zero, event="high", baseline=FixedBaseline("2010-01-01", "2014-12-31"), out=f_zero_det, mat_layout="detectc")

    da4 = xr.concat([da, da - 1.5], dim=xr.DataArray([5.0, 50.0], dims=["depth"], name="depth")).transpose(
        "time", "depth", "lat", "lon"
    )
    r4 = mx.detect(da4, event="high", baseline=FixedBaseline("2010-01-01", "2014-12-31"))
    f4_v73 = r4.to_mat(tmp_path / "d4_v73.mat", layout="v73")
    f4_det = r4.to_mat(tmp_path / "d4_det.mat", layout="detectc")

    n_ev = len(mem.events)
    n_t = da.sizes["time"]
    probe_clim = float(mem.clim.values[2, 3, 100])
    probe_int = float(mem.intensity_ts.values[2, 3, 125])
    probe_cat = int(mem.daily_category_ts.values[2, 3, 125])
    probe_slope = float(mem.trend["slope_per_step"].values[2, 3])
    cell_df = mem.events.isel(lat=2, lon=3)
    cell_n = len(cell_df)
    cell_first_dur = int(cell_df["duration"].iloc[0])
    cell_first_imax = float(cell_df["int_max"].iloc[0])
    first_global_imax = float(mem.events.to_pandas()["int_max"].iloc[0])

    script = tmp_path / "verify_single_files.m"
    script.write_text(
        f"""
try
    % 1. Single-file v7.3 .mat (grids + events struct + matfile partial read)
    s = load('{f_v73}');
    assert(isequal(size(s.clim), [4, 5, 366]), 'v73 clim shape mismatch');
    assert(isequal(size(s.intensity_ts), [4, 5, {n_t}]), 'v73 intensity_ts shape mismatch');
    assert(isa(s.daily_category_ts, 'int8'), 'v73 daily_category_ts must be int8');
    assert(all(s.daily_category_ts(1, :, :) == int8(-128), 'all'), 'v73 land row must be -128');
    assert(abs(s.clim(3, 4, 101) - {probe_clim:.15g}) < 1e-12, 'v73 clim probe mismatch');
    assert(abs(s.intensity_ts(3, 4, 126) - {probe_int:.15g}) < 1e-12, 'v73 intensity probe mismatch');
    assert(s.daily_category_ts(3, 4, 126) == int8({probe_cat}), 'v73 cat probe mismatch');
    assert(abs(s.trend_slope_per_step(3, 4) - {probe_slope:.15g}) < 1e-12, 'v73 trend mismatch');
    assert(isstruct(s.events), 'v73 events must be struct');
    assert(isequal(size(s.events.int_max), [{n_ev}, 1]), 'v73 events.int_max Nx1 mismatch');
    assert(isa(s.events.duration, 'int32') && isa(s.events.event_type, 'int8'));
    assert(islogical(s.events.truncated_start));
    assert(abs(s.events.int_max(1) - {first_global_imax:.15g}) < 1e-12);
    assert(isequal(size(s.grid_offsets), [21, 1]));

    mf = matfile('{f_v73}');
    assert(abs(mf.clim(3, 4, 101) - {probe_clim:.15g}) < 1e-12, 'matfile partial read mismatch');

    % 2. Single-file v5 detectc .mat (MHW cell + mclim/m90/mhw_ts/category_ts/trend)
    d = load('{f_det}');
    assert(iscell(d.MHW) && isequal(size(d.MHW), [4, 5]), 'detectc MHW shape mismatch');
    assert(isempty(d.MHW{{1, 1}}), 'detectc land cell must be empty');
    blk = d.MHW{{3, 4}};
    assert(size(blk, 1) == {cell_n} && blk(1, 3) == {cell_first_dur});
    assert(abs(blk(1, 4) - {cell_first_imax:.15g}) < 1e-12);
    assert(isequal(size(d.mclim), [4, 5, 366]));
    assert(abs(d.mclim(3, 4, 101) - {probe_clim:.15g}) < 1e-12);
    assert(isfield(d, 'mhw_ts') && isfield(d, 'category_ts') && isfield(d, 'trend_slope_per_step'));

    % 3. Single-file NetCDF4 .nc via MATLAB native ncread() (root grids + /events subgroup!)
    nc_clim = ncread('{f_nc}', '/clim');
    assert(isequal(size(nc_clim), [366, 5, 4]), 'nc clim Fortran-order shape mismatch');
    assert(abs(nc_clim(101, 4, 3) - {probe_clim:.15g}) < 1e-12, 'nc clim probe mismatch');
    nc_imax = ncread('{f_nc}', '/events/int_max');
    assert(numel(nc_imax) == {n_ev} && abs(nc_imax(1) - {first_global_imax:.15g}) < 1e-12);

    % 4. EventCatalog standalone v7.3 .mat
    ct = load('{f_cat_v73}');
    assert(isstruct(ct.events) && isequal(size(ct.events.int_max), [{n_ev}, 1]));

    % 5. Zero-event v7.3 and v5 files
    zv = load('{f_zero_v73}');
    assert(isstruct(zv.events) && isempty(zv.events.int_max) && isa(zv.events.int_max, 'double'));
    assert(isempty(zv.events.truncated_start) && islogical(zv.events.truncated_start));
    zd = load('{f_zero_det}');
    assert(iscell(zd.MHW) && isempty(zd.MHW{{2, 2}}));

    % 6. 4D single files
    d4v = load('{f4_v73}');
    assert(isequal(size(d4v.clim), [2, 4, 5, 366]));
    d4d = load('{f4_det}');
    assert(iscell(d4d.MHW) && isequal(size(d4d.MHW), [2, 4, 5]));

    disp('MATLAB_SINGLE_FILE_VERIFY_OK');
catch ME
    disp(getReport(ME));
    exit(1);
end
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [str(MATLAB_BIN), "-batch", f"run('{script}')"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"MATLAB verification failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "MATLAB_SINGLE_FILE_VERIFY_OK" in proc.stdout
