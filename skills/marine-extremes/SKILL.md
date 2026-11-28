---
name: marine-extremes
description: >
  Detect and analyze marine heatwaves (MHWs), marine cold spells (MCSs), and
  general scalar marine extremes (e.g. salinity anomalies) from 1D station time
  series, 3D surface grids (NOAA OISST, OSTIA, ERA5), 4D depth reanalyses
  (GLORYS, ORAS5, ARMOR3D, Argo), and CMIP6 multi-calendar NetCDF/Zarr datasets
  using the `marine-extremes` Python package. Inspects datasets first and
  guides users through interactive parameter selection to assemble `mx.detect(...)`
  without guessing. Use whenever a user asks to detect, classify, benchmark, or
  query MHW/MCS events or marine extremes.
---

# `marine-extremes` Usage Guide for AI Agents

Use `marine_extremes` (Rust + Rayon core with Python/Xarray orchestration) instead of writing slow Python loops or calling `marineHeatWaves` / `xmhw` / `m_mhw1.0`.

## Installation

```bash
pip install "marine-extremes[io]"        # full NetCDF4, HDF5/MAT, Zarr, cftime, Dask, cf-xarray
# Optional extras: "marine-extremes[cf]" (cf-xarray), "marine-extremes[mcp]" (MCP server)
```

## Interactive Parameter Wizard (Default Workflow)

When asked to run detection without explicit parameters, inspect the data first and collect unresolved settings interactively instead of guessing.

### Step 0 — Fast path
- Skip interactive questioning when the user already specified key settings (`event`, `baseline`, `out`, etc.) or explicitly asked to run with defaults.
- Still pause to clarify if a multi-variable file has no `var` specified or if Kelvin units conflict with a Celsius `lower_bound`.

### Step 1 — Auto-inspect the dataset before asking
Probe the target path first (via `marine_extremes.mcp_server.inspect_dataset(path)` or a brief `xarray` read) to determine:
- `data_vars`, shapes, and `dims`
- `singleton_dims` (for example `zlev=1` -> auto-fill `squeeze_singleton="zlev"`)
- `time_range` (`start`, `end`, `calendar`, daily vs. `freq="monthly"`)
- `units` (Celsius vs. Kelvin) and approximate grid size

Never ask the user for values already uniquely known from the file or prompt.

### Step 2 — Ask missing `mx.detect(...)` parameters (max 2 short rounds)
Use the host's structured multiple-choice tool (`AskUserQuestion` in Claude Code, `ask_question` in Antigravity / T3 Code, `request_user_input` in Codex); if unavailable, send one compact numbered choice list in chat and wait for replies. Put the recommended option first.

**Round 1 — Scientific definition and climatology** (skip items already known):
1. `var` *(only if multiple science variables exist)*: list discovered science variables with units and shapes.
2. `event`: `"both"` (MHW + MCS single pass, Recommended), `"high"` (`"mhw"`), or `"low"` (`"mcs"`).
3. `baseline` (climatology type):
   - 30-year fixed climatology `(Recommended)`: `mx.FixedBaseline("1982-01-01", "2011-12-31")` (or the dataset's first 30 years if `1982-2011` is not covered; classic Hobday 2016)
   - 30/31-year centered moving climatology: `mx.MovingBaseline(window_years=31, center=True, boundary="clamp")` (symmetric `±15` yr; pass `window_years=30` if strict 30-year width is preferred)
   - 11-year centered moving climatology: `mx.MovingBaseline(window_years=11, center=True, boundary="clamp")`
   - Detrended climatology: `mx.DetrendedBaseline(method="linear", ...)` (or `"quadratic"`)
   - Fixed threshold or precomputed climatology: `mx.FixedThreshold(...)` / `mx.ExternalClimatology(...)`
   *(Ask a quick follow-up for `boundary` `"clamp"` vs `"shrink"`, custom `(start, end)`, or `"linear"` vs `"quadratic"` only when unresolved.)*
4. `compat`: `"standard"` (Hobday 2016/2018, Recommended), `"m_mhw"` (MATLAB `m_mhw1.0`), `"oliver"` (`marineHeatWaves.py`), or `"heatwaveR"` (R `heatwaveR` / C++ `heatwave3`).

**Round 2 — Physical bounds, precision, memory, and output** (skip items already given):
1. `lower_bound` and `bound_policy` *(adapt for Step 1 units)*:
   - Celsius SST: `lower_bound=-1.8, bound_policy="flag"` `(Recommended)`, `"mask"` / `"floor"`, or `lower_bound=None`.
   - Kelvin SST (OSTIA/ERA5): convert `da = da - 273.15` and pass `lower_bound=-1.8, bound_policy="flag"` `(Recommended)`, keep Kelvin with `lower_bound=271.35, bound_policy="flag"`, or `lower_bound=None`.
2. `out` and `mat_layout`: `.nc` single file `(Recommended)`, `.mat` (`mat_layout="v73"` or `"detectc"`), `.zarr`, `.parquet` / `.csv` (event table only), or `out=None` (in-memory, small/1D data only).
3. `working_dtype`, `intensity_dtype`, daily grids, and `memory_limit`:
   - Fast & compact `(Recommended)`: `working_dtype="auto", intensity_dtype="float32", memory_limit="4GB"` (use `"8GB"` for global grids on >=16 GB RAM)
   - Events + climatology only: add `return_intensity_ts=False, return_daily_category_ts=False` (faster, much smaller disk output)
   - Full `float64`: `working_dtype="float64", intensity_dtype="float64"`

### Step 3 — Populate and execute `mx.detect(...)`
Fill every inferred and chosen argument explicitly into `res = mx.detect(...)`, show the concrete call briefly, run it, and report event counts, output path, and any warnings.

## Core Workflow

```python
import marine_extremes as mx

res = mx.detect(
    "data/OISST_1982_2025_r360x180.nc",  # file, glob, Zarr dir, xr.DataArray/Dataset, ndarray, Series
    var="sst",                          # exact name, standard_name, or alias; omit if sole science var
    time_dim="auto",                    # auto-infers "time", "valid_time", "time_counter", "ocean_time", etc.
    squeeze_singleton="zlev",           # drop singleton vertical dim (e.g. OISST zlev=1) or pass True
    event="both",                       # "high" | "mhw" | "low" | "mcs" | "both"
    baseline=mx.MovingBaseline(window_years=31, center=True, boundary="clamp"),
    compat="standard",                  # "standard" | "m_mhw" | "m_mhw_python" | "oliver" | "heatwaveR"
    lower_bound=-1.8,                   # physical lower bound (degC); use 271.35 for raw Kelvin SST
    bound_policy="flag",                # "flag" | "mask" | "floor"
    working_dtype="auto",               # compact float32 tiles for float32/int16 NetCDF with f64 accumulation
    intensity_dtype="float32",          # halves intensity_ts RAM and disk size
    memory_limit="8GB",
    out="outputs/oisst_mhw.nc",         # single-target .nc | .mat | .zarr | .npz | .parquet | .feather | .csv
)
```

### Fast Result Queries & Persistence

- **O(1) single-cell lookup** (lazy disk-backed on `.nc`, `.mat` v7.3, `.zarr`, `.parquet`):
  `df_pt = res.events.sel(lat=-42.5, lon=148.0)` or `res.events.isel(lat=10, lon=20)`
- **Instant spatial event count map** (from CSR `grid_offsets`, no table scan):
  `counts = res.events.count_map()`
- **Reopen saved output**: `res = mx.open("outputs/oisst_mhw.nc")` (use `with mx.open(...) as res:` or call `res.close()` when done)
- **Filter small/medium tables**: `res.events.filter(event_type="mhw", min_duration=5, min_category=2)`
- **Climatology only**: `clim_ds = mx.compute_climatology(data, var=..., baseline=...)`

## Baselines & Compatibility Presets

| Need | Argument |
| --- | --- |
| Classic Hobday (2016) fixed period | `baseline=mx.FixedBaseline("1982-01-01", "2011-12-31")` |
| Cross-scenario (e.g. historical -> SSP) | `baseline=mx.FixedBaseline("1982-01-01", "2014-12-31", clim_data=hist_nc)` |
| Detrended baseline (linear / quadratic OLS) | `baseline=mx.DetrendedBaseline("linear")` (coefficients in `res.trend`) |
| 30/31-yr or 11-yr centered moving baseline | `baseline=mx.MovingBaseline(window_years=31, center=True, boundary="clamp")` (or `window_years=11` / `30`) |
| Physical/ecological threshold | `baseline=mx.FixedThreshold(threshold_high=29.0, climatology=27.0)` (scalar or static spatial-only `DataArray`; use `ExternalClimatology` for seasonal clim) |
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
   - `marine-extremes` performs point-wise 1D-4D extreme detection and single-file CSR persistence. It does not perform 3D/4D spatial connected-component object tracking or vertical ocean heat content integration.
