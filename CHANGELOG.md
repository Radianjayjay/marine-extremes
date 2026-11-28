# Changelog

All notable changes to `marine-extremes` are documented in this file.

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
