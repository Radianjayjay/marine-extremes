---
name: marine-extremes
description: >
  Detect and analyze marine heatwaves (MHWs), marine cold spells (MCSs), and
  general scalar marine extremes (e.g. salinity anomalies) from 1D station time
  series, 3D surface grids (NOAA OISST, OSTIA, ERA5), 4D depth reanalyses
  (GLORYS, ORAS5, ARMOR3D, Argo), and CMIP6 multi-calendar NetCDF/Zarr datasets
  using the `marine-extremes` Python package. Use whenever a user asks to detect,
  classify, benchmark, or query MHW/MCS events or marine extremes.
---

# `marine-extremes` Usage Guide for AI Agents

Use `marine_extremes` (Rust + Rayon core with Python/Xarray orchestration) instead of writing slow Python loops or calling `marineHeatWaves` / `xmhw` / `m_mhw1.0`.

## Installation

```bash
pip install "marine-extremes[io]"        # full NetCDF4, HDF5/MAT, Zarr, cftime, Dask
# Optional extras: "marine-extremes[cf]" (cf-xarray), "marine-extremes[mcp]" (MCP server)
```

> Note: Until `v0.2.0` is published on PyPI, `[cf]`, `[mcp]`, and `marine-extremes-mcp` live on `main` (install from source via `maturin develop --release --extras io,mcp` or `pip install "marine-extremes[io,mcp] @ git+https://github.com/Radianjayjay/marine-extremes.git"`).

## Core Workflow

```python
import marine_extremes as mx

res = mx.detect(
    "sst_daily_*.nc",                   # file, glob, Zarr dir, xr.DataArray/Dataset, ndarray, Series
    var="sst",                          # exact name, standard_name, or alias ("sst", "temp", "salinity"); omit if sole science var
    time_dim="auto",                    # auto-infers "time", "valid_time", "time_counter", "ocean_time", etc.
    squeeze_singleton="zlev",           # drop singleton vertical dim (e.g. OISST zlev=1) or pass True
    event="both",                       # "high" | "mhw" | "low" | "mcs" | "both"
    baseline=mx.FixedBaseline("1982-01-01", "2011-12-31"),
    working_dtype="auto",               # compact float32 tiles for float32/int16 NetCDF with f64 accumulation
    intensity_dtype="float32",          # halves intensity_ts RAM and disk size
    memory_limit="4GB",
    out="outputs/mhw_mcs.nc",           # single-target .nc | .mat | .zarr | .npz | .parquet | .feather | .csv
)
```

### Fast Result Queries & Persistence

- **O(1) single-cell lookup** (lazy disk-backed on `.nc`, `.mat` v7.3, `.zarr`, `.parquet`):
  `df_pt = res.events.sel(lat=-42.5, lon=148.0)` or `res.events.isel(lat=10, lon=20)`
- **Instant spatial event count map** (from CSR `grid_offsets`, no table scan):
  `counts = res.events.count_map()`
- **Reopen saved output**: `res = mx.open("outputs/mhw_mcs.nc")` (use `with mx.open(...) as res:` or call `res.close()` when done)
- **Filter small/medium tables**: `res.events.filter(event_type="mhw", min_duration=5, min_category=2)`
- **Climatology only**: `clim_ds = mx.compute_climatology(data, var=..., baseline=...)`

## Baselines & Compatibility Presets

| Need | Argument |
| --- | --- |
| Classic Hobday (2016) fixed period | `baseline=mx.FixedBaseline("1982-01-01", "2011-12-31")` |
| Cross-scenario (e.g. historical -> SSP) | `baseline=mx.FixedBaseline("1982-01-01", "2014-12-31", clim_data=hist_nc)` |
| Detrended baseline (linear / quadratic OLS) | `baseline=mx.DetrendedBaseline("linear")` (coefficients in `res.trend`) |
| Sliding window baseline | `baseline=mx.MovingBaseline(window_years=31, center=True, boundary="clamp")` |
| Physical/ecological threshold | `baseline=mx.FixedThreshold(threshold_high=29.0, climatology=clim_da)` |
| Precomputed climatology | `baseline=mx.ExternalClimatology.from_dataset(clim_ds)` |
| Monthly Jacox (2020) mode | `freq="monthly"` (defaults: `window_half_width=0, smooth_width=1, min_duration=1, max_gap=0`) |
| Match reference tool conventions | `compat="standard"` \| `"m_mhw"` \| `"m_mhw_python"` \| `"oliver"` \| `"heatwaveR"` |

## Critical Ocean-Data Safeguards

1. **Multi-variable files never auto-guess `var`**:
   - **NOAA OISST** contains `sst`, `anom`, `err`, `ice` plus `zlev=1`. Always pass `var="sst", squeeze_singleton="zlev"`.
   - **ARMOR3D / multi-var models** require explicit `var="thetao"` / `"ugo"` / `"so"` or `standard_name`.
   - Single-science-variable files with CF bounds/ancillary vars (CMIP6 `tos`, OSTIA, ERA5 SST) work cleanly with `var=None` and `time_dim="auto"`.
2. **Kelvin vs. Celsius (`lower_bound` safety)**:
   - **OSTIA** (`units="kelvin"`) and **ERA5** (`units="K"`) store Kelvin! Passing `lower_bound=-1.8` on Kelvin triggers a `UserWarning`. Either convert `da = da - 273.15` (and update `da.attrs["units"] = "degC"`) or pass `lower_bound=271.35`.
   - Sea-ice edge guard: pass `min_valid_fraction=0.5`, `max_pad_length=2`, and `lower_bound=-1.8, bound_policy="flag"` (or `"mask"` / `"floor"`).
3. **Large grids**:
   - Always pass `out=...` and `memory_limit="4GB"` (or `"8GB"`) so results stream to disk tile-by-tile.
   - Never call `res.events.to_pandas()` on huge global catalogs just to check one point or counts; use `res.events.sel(...)`, `res.events.count_map()`, or `res.events.iter_chunks()`.
4. **Scope**:
   - `marine-extremes` performs point-wise 1D–4D extreme detection and single-file CSR persistence. It does not perform 3D/4D spatial connected-component object tracking or vertical ocean heat content integration.
