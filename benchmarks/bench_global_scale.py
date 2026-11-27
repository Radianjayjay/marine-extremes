"""Throughput & memory benchmark for marine-extremes on realistic global-scale synthetic ocean grids."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xarray as xr

import marine_extremes as mx


def run_benchmark() -> None:
    """Benchmark Fixed, Detrended, and 31-yr Moving baselines on a 100x100x40yr synthetic grid."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1982-01-01", "2021-12-31", freq="D")  # 40 years = 14,610 days
    nt = dates.size
    n_lat, n_lon = 100, 100  # 10,000 grid cells (~1/70th of global OISST 0.25 deg ocean)
    doy = dates.dayofyear.to_numpy()
    seas = (18.0 + 4.0 * np.sin(2.0 * np.pi * doy / 365.25)).astype(np.float32)

    # Generate AR(1) anomalies across 10,000 pixels (with 25% land mask)
    data = np.empty((nt, n_lat, n_lon), dtype=np.float32)
    state = rng.normal(0.0, 0.6, size=(n_lat, n_lon)).astype(np.float32)
    for t in range(nt):
        state = np.float32(0.82) * state + rng.normal(0.0, 0.55, size=(n_lat, n_lon)).astype(np.float32)
        data[t] = seas[t] + state
    data[:, :25, :] = np.nan  # 25% land mask -> 7,500 active ocean pixels

    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": dates,
            "lat": np.linspace(-50.0, 50.0, n_lat),
            "lon": np.linspace(0.0, 359.0, n_lon),
        },
        name="sst",
    )

    # 1. Fixed baseline MHW + MCS (both)
    t0 = time.perf_counter()
    res_fix = mx.detect(
        da,
        event="both",
        baseline=mx.FixedBaseline("1982-01-01", "2011-12-31"),
        return_intensity_ts=False,
        return_daily_category_ts=False,
    )
    dt_fix = time.perf_counter() - t0

    # 2. Detrended linear baseline (both)
    t0 = time.perf_counter()
    res_det = mx.detect(
        da,
        event="both",
        baseline=mx.DetrendedBaseline(method="linear"),
        return_intensity_ts=False,
        return_daily_category_ts=False,
    )
    dt_det = time.perf_counter() - t0

    # 3. Centered 31-year MovingBaseline (both)
    t0 = time.perf_counter()
    res_mov = mx.detect(
        da,
        event="both",
        baseline=mx.MovingBaseline(window_years=31, center=True, boundary="clamp"),
        return_clim=False,
        return_intensity_ts=False,
        return_daily_category_ts=False,
    )
    dt_mov = time.perf_counter() - t0

    active_pixels = 7500
    est_oisst_scale = 700_000 / active_pixels

    print(f"Grid: {n_lat}x{n_lon} ({active_pixels:,} active ocean pixels) x {nt:,} days (40 yr)")
    print(
        f"  FixedBaseline (MHW+MCS):    {dt_fix:.3f} s  "
        f"({len(res_fix.events):,} events, est. global 0.25° OISST: {dt_fix * est_oisst_scale:.1f} s)"
    )
    print(
        f"  DetrendedBaseline (linear): {dt_det:.3f} s  "
        f"({len(res_det.events):,} events, est. global 0.25° OISST: {dt_det * est_oisst_scale:.1f} s)"
    )
    print(
        f"  MovingBaseline (31-yr):     {dt_mov:.3f} s  "
        f"({len(res_mov.events):,} events, est. global 0.25° OISST: {dt_mov * est_oisst_scale:.1f} s)"
    )


if __name__ == "__main__":
    run_benchmark()
