"""Minimal self-contained quickstart for marxtreme (no external data files required)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import marxtreme as mx


def make_demo_sst() -> xr.DataArray:
    """Create a 10-year 3D synthetic SST grid with land mask and injected MHW/MCS pulses."""
    rng = np.random.default_rng(42)
    times = pd.date_range("2010-01-01", "2019-12-31", freq="D")
    n_t = len(times)
    doy = times.dayofyear.to_numpy()
    seasonal = 18.0 + 4.0 * np.sin(2.0 * np.pi * (doy - 30) / 365.25)

    n_lat, n_lon = 5, 6
    data = np.empty((n_t, n_lat, n_lon), dtype=np.float32)
    for i in range(n_lat):
        for j in range(n_lon):
            if i == 0:
                data[:, i, j] = np.nan  # land row
            else:
                series = seasonal + 0.4 * i - 0.2 * j + rng.normal(0.0, 0.65, size=n_t)
                series[300:312] += 4.0  # MHW pulse
                series[1100:1112] -= 3.8  # MCS pulse
                data[:, i, j] = series.astype(np.float32)

    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": times,
            "lat": np.linspace(-44.0, -40.0, n_lat),
            "lon": np.linspace(145.0, 150.0, n_lon),
        },
        name="sst",
        attrs={"units": "degC"},
    )


def main() -> None:
    """Run an end-to-end demo covering detection, O(1) lookup, reopening, export, and baselines."""
    da = make_demo_sst()

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)

        # 1. Fixed baseline: detect MHW + MCS in a single pass, stream to single NetCDF4 file
        nc_path = out_dir / "demo_mhw_mcs.nc"
        res = mx.detect(
            da,
            event="both",
            baseline=mx.FixedBaseline("2010-01-01", "2017-12-31"),
            lower_bound=-1.8,
            bound_policy="flag",
            intensity_dtype="float32",
            out=nc_path,
        )
        print("NetCDF single-file run:", res)
        print("Total events:", len(res.events))
        print("Count map shape:", res.events.count_map().shape)
        print("Single-cell O(1) lookup (lat=-42, lon=148):")
        print(res.events.sel(lat=-42.0, lon=148.0)[["event_type", "date_start", "duration", "int_max", "category_max"]].head())

        # 2. Reopen single file directly
        reopened = mx.open(nc_path)
        assert len(reopened.events) == len(res.events)

        # 3. Export to other self-contained formats (.mat v7.3, .mat detectc, .zarr, .parquet)
        res.to_mat(out_dir / "demo_v73.mat", layout="v73")
        res.to_mat(out_dir / "demo_detectc.mat", layout="detectc")
        res.save(out_dir / "demo.zarr")
        res.events.save(out_dir / "demo_events.parquet")

        # 4. Detrended & Moving baselines
        res_det = mx.detect(da, event="high", baseline=mx.DetrendedBaseline(method="linear"))
        res_mov = mx.detect(da, event="high", baseline=mx.MovingBaseline(window_years=5, center=True))
        print(f"Detrended events: {len(res_det.events)}, Moving-5yr events: {len(res_mov.events)}")


if __name__ == "__main__":
    main()
