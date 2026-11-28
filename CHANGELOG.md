# Changelog

All notable changes to `marine-extremes` are documented in this file.

## [0.1.2] - 2026-11-28

### Added
- Direction-specific Hobday intensity grids (`intensity_high_ts` and `intensity_low_ts`) alongside `intensity_ts`, plus CF metadata attributes (`long_name`, `units`, `description`, and `missing_value_code`) across in-memory results and saved `.nc`, `.zarr`, `.mat`, and `.npz` files.
- `clim_dtype` (`"float64"` or `"float32"`) option on `detect()` and MCP `detect_extremes` to halve disk and RAM usage for 4D `MovingBaseline` climatology and threshold arrays.
- Bounded-chunk `DetectResult.summary()` and `EventCatalog.summary()` methods, reused by `mcp_server.summarize_result`.
- Multi-agency timestamp normalization (`datetime64[D|s|ms|us|ns]`, timezone-aware datetimes and 4-/5-digit ISO strings converted to naive UTC, undecoded CF numeric axes, `YYYYMMDD` ints), uniform midnight event dates across formats, Zarr encoding for out-of-range years (`< 1` or `> 9999`), and strict duplicate-day rejection under `freq="daily"`.

### Changed
- Added single-pass pre-read for file-backed inputs (NetCDF/Zarr) when the raw array fits inside 45% of `memory_limit` (17-18% faster global 1° OISST runs under `memory_limit="10GB"`), 16-row HDF5 chunk alignment, faster zlib/gzip level 1 compression, and cleaned up NetCDF `events/grid_offsets` dimension naming.
- Updated bilingual READMEs with `v0.1.2` global OISST benchmarks and figures, `SKILL.md` with HPC `glibc < 2.28` `pyarrow<21`, pre-read `memory_limit`, scheduler/SSH node choices, and non-blocking execution rules, and `mcp_server.py` schema descriptions and sub-daily `inspect_dataset` reporting.

### Fixed
- Preserved CF time `units`/`calendar` attrs when passing an `xr.DataArray` as `time=` alongside a NumPy array.
- Enforced bounded `chunk_rows` slicing in `MemoryEventStore.iter_chunks()` and `chunk_rows >= 1` validation across memory and disk event stores.

## [0.1.1] - 2026-11-28

### Added
- Built-in lightweight CF metadata resolver (`marine_extremes.cf`) with optional `cf-xarray` (`[cf]`) enhancement: automatic non-`time` dimension inference (`valid_time`, `time_counter`, `ocean_time`), CF auxiliary variable filtering (`*_bnds`, `grid_mapping`, `ancillary_variables`, `cell_measures`), `standard_name` / marine alias variable selection, and Kelvin vs. Celsius `lower_bound` mismatch warnings.
- Universal AI Agent Skill ([skills/marine-extremes/SKILL.md](https://github.com/Radianjayjay/marine-extremes/blob/main/skills/marine-extremes/SKILL.md)) for coding agents (Claude Code, Codex, Cursor, Antigravity, Windsurf, OpenCode), including an auto-inspect interactive parameter wizard to assemble `mx.detect(...)` settings.
- Optional local stdio MCP server (`marine-extremes-mcp` / `[mcp]` extra, supporting MCP SDK v2 and v1) exposing `inspect_dataset`, `detect_extremes`, `query_events`, and `summarize_result` with path-overlap protection, `ToolError` diagnostics, and real stdio end-to-end tests.

## [0.1.0] - 2026-11-27

### Added
- Rust + Rayon parallel detection core (`marine_extremes._core`) compiled with PyO3 `abi3-py311`, releasing the Python GIL (`py.detach()`), skipping land pixels, and running a compact `float32` tile fast path (`working_dtype="auto"`) with internal `f64` accumulation.
- Single-pass MHW and MCS detection (`event="both"`), plus general upper/lower scalar extreme detection across 1D-4D gridded and station datasets.
- Baseline definitions: `FixedBaseline` (with optional external `clim_data`), `DetrendedBaseline` (`linear` and `quadratic` OLS detrending), `MovingBaseline` (sliding multi-year window with `clamp`/`shrink` boundaries), `FixedThreshold`, and `ExternalClimatology`.
- Native Gregorian and CMIP6 `cftime` calendars (`standard`, `proleptic_gregorian`, `noleap`/`365_day`, `360_day`, `all_leap`/`366_day`) and `freq="monthly"` detection.
- Explicit cross-tool compatibility profiles (`"standard"`, `"m_mhw"`, `"m_mhw_python"`, `"oliver"`, `"heatwaveR"`), verified against MATLAB R2025b `m_mhw1.0`, `ecjoliver/marineHeatWaves.py`, `mhw-detect`, `heatwave3`, and `xmhw` on real NOAA OISST v2.1 data.
- Memory-bounded out-of-core streaming (`memory_limit`) across multi-file NetCDF/Zarr globs and seven standalone single-target output formats (`.nc`, `.mat` v7.3 HDF5 and v5 `detectc` cell, `.zarr`, `.npz`, `.parquet`, `.feather`, `.csv`) with CSR `grid_offsets` for O(1) lazy grid lookup.
