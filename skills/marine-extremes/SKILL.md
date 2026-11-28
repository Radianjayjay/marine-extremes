---
name: marine-extremes
description: >
  Detect and analyze upper-tail ("high") and lower-tail ("low") marine
  extremes across 1D station time series, 3D surface grids (NOAA OISST, OSTIA,
  ERA5), 4D depth reanalyses (GLORYS, ORAS5, ARMOR3D, Argo), and CMIP6
  multi-calendar NetCDF/Zarr datasets using the `marine-extremes` Python
  package. Covers seawater temperature (MHWs/MCSs) as well as salinity, ocean
  acidity (pH), dissolved oxygen (hypoxia), CO2/pCO2, and chlorophyll. Inspects
  datasets and host capabilities first without slow trial runs, lets users
  pick scientific options and compute nodes/schedulers, launches gridded jobs
  non-blocking, and stops to offer periodic check-ins.
---

# `marine-extremes` Usage Guide for AI Agents

Use `marine_extremes` (Rust + Rayon single-node shared-memory core with Python/Xarray orchestration) instead of slow Python loops or calling `marineHeatWaves` / `xmhw` / `m_mhw1.0`.

## Installation

```bash
pip install "marine-extremes[io]"        # full NetCDF4, HDF5/MAT, Zarr, cftime, Dask, cf-xarray
# Optional extras: "marine-extremes[cf]" (cf-xarray), "marine-extremes[mcp]" (MCP server)
```

- **Older HPC Linux (`glibc < 2.28`, e.g., CentOS 7)**: check `ldd --version`. `pyarrow>=21` has no `manylinux2014` wheel and triggers a multi-minute Arrow C++ source build. Always pre-pin `pyarrow<21` with binary wheels only:
  ```bash
  pip install --only-binary=:all: "pyarrow<21" "marine-extremes[io]"
  ```

## Interactive Parameter Wizard (Default Workflow)

When asked to run detection without explicit parameters, inspect metadata and host capabilities first, then ask unresolved scientific and compute-node settings before running anything heavy.

### Step 0: Fast path
- Skip interactive questioning when the user already specified key settings (`event`, `baseline`, `out`, execution node/scheduler) or explicitly asked to run with defaults.
- Still clarify if a multi-variable file has no `var` specified, Kelvin units conflict with a Celsius `lower_bound`, or an HPC scheduler/node choice is needed for a large gridded run.

### Step 1: One fast read-only probe before asking
Run **one** lightweight read-only check combining dataset headers and host environment:
- **Dataset header** (`marine_extremes.mcp_server.inspect_dataset(path)` if installed, `ncdump -h`, or `xarray.open_dataset(..., chunks={})` without loading data): collect `data_vars`, shapes, `dims`, `singleton_dims` (e.g. `zlev=1` -> `squeeze_singleton="zlev"`), `time_range` (`start`, `end`, `calendar`, daily vs. `freq="monthly"`), `units` (Celsius vs. Kelvin), and approximate grid size.
- **Host check**: `hostname`, `nproc`, `free -h`, `ldd --version | head -n1` (`glibc < 2.28` -> `pyarrow<21`), `command -v sbatch qsub` (plus `sinfo` / `pbsnodes` only if a scheduler is present), and existing Python envs with `marine_extremes`.
- **Hard rule**: **Do NOT run trial detections, single-cell smoke tests, or timing benchmarks before asking the user.** Do not ask the user for values already uniquely known from the file or prompt.

### Step 2: Ask missing settings and compute target (max 2 short rounds)
Use the host's structured multiple-choice tool (`AskUserQuestion` in Claude Code, `ask_question` in Antigravity / T3 Code, `request_user_input` in Codex); if unavailable, send one compact numbered choice list in chat and wait. Put the recommended option first.

**Round 1: Scientific definition and climatology** (skip items already known):
1. `var` *(only if multiple science variables exist)*: list discovered science variables with units and shapes.
2. `event`: Uses variable-neutral names `"high"` / `"low"` because `marine-extremes` supports temperature, salinity, pH/acidity, dissolved oxygen, CO2/pCO2, and chlorophyll:
   - `"both"` `(Recommended)`: high + low extremes in one pass (MHW + MCS for temperature).
   - `"high"` (`"mhw"` alias): upper-tail events (MHWs, high salinity, high pCO2).
   - `"low"` (`"mcs"` alias): lower-tail events (MCSs, low salinity, low-pH acidification, low-O2 hypoxia).
3. `baseline` (climatology type):
   - 30-year fixed climatology `(Recommended)`: `mx.FixedBaseline("1982-01-01", "2011-12-31")` (or the dataset's first 30 years if `1982-2011` is not covered; classic Hobday 2016)
   - 30/31-year centered moving climatology: `mx.MovingBaseline(window_years=31, center=True, boundary="clamp")` (symmetric `±15` yr; use `window_years=30` if strict 30-year width is preferred)
   - 11-year centered moving climatology: `mx.MovingBaseline(window_years=11, center=True, boundary="clamp")`
   - Detrended climatology: `mx.DetrendedBaseline(method="linear", ...)` (or `"quadratic"`)
   - Fixed threshold or precomputed climatology: `mx.FixedThreshold(...)` / `mx.ExternalClimatology(...)`
   *(Ask a quick follow-up for `boundary` `"clamp"` vs `"shrink"`, custom `(start, end)`, or `"linear"` vs `"quadratic"` only when unresolved.)*
4. `compat`: `"standard"` (Hobday 2016/2018, Recommended), `"m_mhw"` (MATLAB `m_mhw1.0`), `"oliver"` (`marineHeatWaves.py`), or `"heatwaveR"` (R `heatwaveR` / C++ `heatwave3`).

**Round 2: Bounds, precision, output, and execution node** (skip items already given):
1. `lower_bound` and `bound_policy` *(adapt for Step 1 units and variable type)*:
   - Celsius SST: `lower_bound=-1.8, bound_policy="flag"` `(Recommended)`, `"mask"` / `"floor"`, or `lower_bound=None`.
   - Kelvin SST (OSTIA/ERA5): convert `da = da - 273.15` (`da.attrs["units"] = "degC"`) and pass `lower_bound=-1.8, bound_policy="flag"` `(Recommended)`, keep Kelvin with `lower_bound=271.35, bound_policy="flag"`, or `lower_bound=None`.
   - Non-temperature scalars: `lower_bound=0.0, bound_policy="flag"` (or `"mask"`) for non-negative fields (salinity, dissolved oxygen, chlorophyll), or `lower_bound=None` for pH/CO2 unless a domain cutoff applies.
2. `out` and `mat_layout`: `.nc` single file `(Recommended)`, `.mat` (`mat_layout="v73"` or `"detectc"`), `.zarr` (fastest for 4D `MovingBaseline` + daily writes), `.parquet` / `.csv` (event table only), or `out=None` (1D/small data only).
3. `working_dtype`, `clim_dtype`, `intensity_dtype`, daily grids, and `memory_limit`:
   - Fast & compact `(Recommended)`: `working_dtype="auto", clim_dtype="float32", intensity_dtype="float32"`. Set `memory_limit="10GB"` (or `>= 2.3x` raw uncompressed input size when RAM permits) for single-pass pre-read (`~17%` faster on global 1° OISST), or `"4GB"`/`"8GB"` for low-memory (`< 3.1 GB` RSS) tiled streaming. For `MovingBaseline` + `return_clim=True`, `clim_dtype="float32"` halves the 4D `(..., year, doy)` cube size and write time.
   - Events + intensity only: add `return_clim=False` (or omit daily grids too with `return_intensity_ts=False, return_daily_category_ts=False`).
   - Full `float64`: `working_dtype="float64", clim_dtype="float64", intensity_dtype="float64"`
4. **Execution target / compute node** *(for gridded runs or HPC environments)*:
   - **Current / service node** `(Recommended on standalone machines or when allowed)`: run in background locally (`nohup ... > run.log 2>&1 &`).
   - **SSH to a compute node**: specify hostname (e.g. `node01`); launch via `ssh <node> "cd <dir> && nohup ... > run.log 2>&1 < /dev/null &"`.
   - **PBS (`qsub`) / Slurm (`sbatch`) single-node Rayon** `(Recommended on HPC clusters)`: `marine-extremes` uses Rayon threads within one node (no MPI), so request `#PBS -l nodes=1:ppn=<N>` or `#SBATCH --nodes=1 --cpus-per-task=<N>` with `RAYON_NUM_THREADS=<N>`.
   - **PBS (`qsub`) / Slurm (`sbatch`) multi-node spatial-shard job array**: split space (e.g. latitude bands or depth slices) across array tasks (`qsub -J`/`-t` or `sbatch --array`), writing per-shard outputs (`part_${TASK_ID}.nc`). Never claim `mx.detect(...)` uses MPI across nodes.

### Step 3: Generate script, launch non-blocking, verify startup, and stop
- If a small 1D/local call finishes in a few seconds, run it directly and report event counts and output paths.
- For gridded or multi-minute jobs:
  1. Write a clean standalone script (e.g. `<out_dir>/run_detect.py`) with every inferred and chosen `mx.detect(...)` argument filled explicitly, plus a `.pbs` / `.slurm` job script when using a scheduler.
  2. Launch **non-blocking** (never wait synchronously for `mx.detect(...)` to finish):
     - Local/service node: `nohup <python> -u run_detect.py > run.log 2>&1 < /dev/null & echo $!`
     - SSH node: `ssh <node> 'cd <dir> && nohup <python> -u run_detect.py > run.log 2>&1 < /dev/null & echo $!'`
     - Scheduler: `qsub run_detect.pbs` or `sbatch run_detect.slurm`
  3. Wait ~2-5 s to confirm startup: check `ps -p <pid>` (or via `ssh`) / `qstat` / `squeue`, and read the first lines of `run.log` to ensure no immediate import, path, or argument error.
  4. **Stop the agent turn cleanly**: tell the user the job is now running, report the target node, `PID` or `Job ID`, script path, log path, and expected output path, and explicitly ask whether to set up periodic scheduled check-ins (using `/schedule`, `cron`, or host timer tools when supported).

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
    clim_dtype="float32",               # halves 4D MovingBaseline clim/thresh RAM, disk size & write time
    intensity_dtype="float32",          # halves intensity_ts / intensity_high_ts / intensity_low_ts size
    memory_limit="8GB",
    out="outputs/oisst_mhw.nc",         # single-target .nc | .mat | .zarr | .npz | .parquet | .feather | .csv
)
```

### Fast Result Queries & Persistence

- **Bounded-memory run summary** (safe on millions of events, never calls `.to_pandas()`):
  `stats = res.summary()` (or `res.events.summary()`); separate intensity series sit in `res.intensity_high_ts`, `res.intensity_low_ts`, and combined `res.intensity_ts`.
- **O(1) single-cell lookup** (lazy disk-backed on `.nc`, `.mat` v7.3, `.zarr`, `.parquet`):
  `df_pt = res.events.sel(lat=-42.5, lon=148.0)` or `res.events.isel(lat=10, lon=20)`
- **Instant spatial event count map** (from CSR `grid_offsets`, no table scan):
  `counts = res.events.count_map()`
- **Reopen saved output**: `res = mx.open("outputs/oisst_mhw.nc")` (use `with mx.open(...) as res:` or call `res.close()` when done)
- **Date columns rule**: `date_start`/`date_peak`/`date_end` normalize to midnight `datetime64[ns]` (`1678..2261` Gregorian; extract years via `.dt.year`, never `.str[:4]`), `datetime64[s]` out of range, or ISO strings for non-Gregorian `cftime`.
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
3. **Large grids & multi-agency timestamps**:
   - Pass `out=...` and `memory_limit="4GB"` or `"8GB"` for low-memory (`< 3.1 GB` RSS) tiled streaming, or `memory_limit="10GB"` (`>= 2.3x` raw uncompressed array size) on ~4 GB inputs for single-pass pre-read.
   - Never call `res.events.to_pandas()` on huge global catalogs just to compute totals or check one point; use `res.summary()`, `res.events.sel(...)`, `res.events.count_map()`, or `res.events.iter_chunks()`.
   - `freq="daily"` rejects duplicate calendar days (resample sub-daily inputs to daily first); tz-aware UTC, undecoded CF numeric coords, `YYYYMMDD` integers, and ISO strings all normalize cleanly.
4. **Low-event sign convention (`"low"` / `"mcs"`, low pH, low O2, low salinity)**:
   - `int_max`, `int_mean`, `int_cum`, `int_*_rel_thresh`, gridded `intensity_ts` / `intensity_low_ts`, and gridded `daily_category_ts` / `event_category_ts` (`-1` to `-4`, with `CATEGORY_MISSING = -128` via `missing_value_code`) are **negative** for `"low"` events (`intensity_high_ts` holds `>= 0` high events).
   - `int_std`, `int_var`, `rate_onset`, `rate_decline`, and event-table `category_max` (`1` to `4`) remain **non-negative magnitudes**; sort ascending when ranking the strongest low events.
5. **Scope**:
   - `marine-extremes` performs point-wise 1D-4D extreme detection and single-file CSR persistence. It does not perform 3D/4D spatial connected-component object tracking or vertical ocean heat content integration.
