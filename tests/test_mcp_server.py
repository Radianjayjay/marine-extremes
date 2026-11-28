"""Tests for the optional marine-extremes stdio MCP server tools."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from marine_extremes import mcp_server


def _write_sample_datasets(tmp_path: Path) -> dict[str, Path]:
    dates = pd.date_range("2000-01-01", "2004-12-31", freq="D")
    nt = dates.size
    rng = np.random.default_rng(123)
    doy = np.arange(nt, dtype=np.float64)
    seas = (20.0 + 3.0 * np.sin(2.0 * np.pi * doy / 365.25))[:, None, None]
    noise = rng.normal(0.0, 0.5, size=(nt, 4, 3))
    for t in range(1, nt):
        noise[t] = 0.8 * noise[t - 1] + 0.5 * noise[t]
    # Inject several strong warm and cold spells
    noise[200:212, :, :] += 4.0
    noise[600:615, :, :] += 4.5
    noise[900:910, :, :] -= 4.2
    noise[1200:1214, :, :] += 3.8
    sst = seas + noise

    # 1. OISST-like multi-var with singleton zlev=1 and descending lat
    oisst_p = tmp_path / "oisst_multi.nc"
    ds_oisst = xr.Dataset(
        {
            "sst": (("time", "zlev", "lat", "lon"), sst[:, None, :, :], {"units": "Celsius"}),
            "anom": (("time", "zlev", "lat", "lon"), noise[:, None, :, :]),
            "err": (("time", "zlev", "lat", "lon"), np.full((nt, 1, 4, 3), 0.1)),
            "ice": (("time", "zlev", "lat", "lon"), np.zeros((nt, 1, 4, 3))),
        },
        coords={
            "time": dates,
            "zlev": [0.0],
            "lat": [-28.0, -29.0, -30.0, -31.0],
            "lon": [150.0, 151.0, 152.0],
        },
    )
    ds_oisst.to_netcdf(oisst_p)

    # 2. Split yearly glob for glob inspection test
    for idx, yr_mask in enumerate((dates.year <= 2001, dates.year >= 2002)):
        ds_oisst.isel(time=yr_mask).to_netcdf(tmp_path / f"oisst_part_{idx}.nc")

    # 3. ERA5-like valid_time + Kelvin units
    era5_p = tmp_path / "era5_k.nc"
    ds_era5 = xr.Dataset(
        {
            "sst": (
                ("valid_time", "latitude", "longitude"),
                sst + 273.15,
                {"standard_name": "unknown", "units": "K"},
            )
        },
        coords={
            "valid_time": dates,
            "latitude": [-29.0, -30.0, -31.0, -32.0],
            "longitude": [150.0, 151.0, 152.0],
        },
    )
    ds_era5.to_netcdf(era5_p)

    # 4. CMIP6-like tos + bounds + crs
    cmip_p = tmp_path / "cmip_tos.nc"
    ds_cmip = xr.Dataset(
        {
            "tos": (
                ("time", "j", "i"),
                sst[:, :2, :2],
                {"standard_name": "sea_surface_temperature", "units": "degC", "grid_mapping": "crs"},
            ),
            "time_bnds": (("time", "bnds"), np.zeros((nt, 2))),
            "crs": ((), 0, {"grid_mapping_name": "latitude_longitude"}),
        },
        coords={"time": ("time", dates, {"bounds": "time_bnds"}), "j": [0, 1], "i": [0, 1]},
    )
    ds_cmip.to_netcdf(cmip_p)

    return {"oisst": oisst_p, "era5": era5_p, "cmip": cmip_p, "glob": tmp_path / "oisst_part_*.nc"}


def test_inspect_dataset_covers_oisst_era5_cmip_and_glob(tmp_path: Path):
    paths = _write_sample_datasets(tmp_path)

    info_oisst = mcp_server.inspect_dataset(str(paths["oisst"]))
    assert info_oisst["recommended_var"] is None
    assert info_oisst["singleton_dims"] == ["zlev"]
    assert len(info_oisst["data_vars"]) == 4
    assert all(not v["is_auxiliary"] for v in info_oisst["data_vars"])
    assert json.dumps(info_oisst, allow_nan=False)

    info_era5 = mcp_server.inspect_dataset(str(paths["era5"]))
    assert info_era5["recommended_var"] == "sst"
    assert info_era5["time_dim"] == "valid_time"
    assert any("Kelvin" in n for n in info_era5["notes"])

    info_cmip = mcp_server.inspect_dataset(str(paths["cmip"]))
    assert info_cmip["recommended_var"] == "tos"
    aux_flags = {v["name"]: v["is_auxiliary"] for v in info_cmip["data_vars"]}
    assert aux_flags == {"tos": False, "time_bnds": True, "crs": True}
    assert info_cmip["time_range"]["n_steps"] == 1827

    info_glob = mcp_server.inspect_dataset(str(paths["glob"]), var="sst")
    assert info_glob["n_files"] == 2
    assert info_glob["recommended_var"] == "sst"
    assert info_glob["time_range"]["n_steps"] == 1827


def test_detect_query_and_summarize_roundtrip(tmp_path: Path):
    paths = _write_sample_datasets(tmp_path)

    # 1. detect_extremes with region slicing (descending lat), singleton squeeze, and .nc output
    out_nc = tmp_path / "oisst_sub_mhw.nc"
    det_nc = mcp_server.detect_extremes(
        str(paths["oisst"]),
        out=str(out_nc),
        var="sst",
        squeeze_singleton="zlev",
        event="both",
        region={"lat": [-30.0, -29.0], "lon": [150.0, 151.0]},
        sample_limit=2,
    )
    assert out_nc.is_file()
    assert det_nc["spatial_dims"] == ["lat", "lon"]
    assert det_nc["spatial_shape"] == [2, 2]
    assert det_nc["n_events"] > 5
    assert len(det_nc["sample_events"]) == 2
    assert json.dumps(det_nc, allow_nan=False)

    # 2. detect_extremes on Kelvin ERA5 captures UserWarning and writes .parquet
    out_pq = tmp_path / "era5_detrended.parquet"
    det_era = mcp_server.detect_extremes(
        str(paths["era5"]),
        out=str(out_pq),
        baseline_type="detrended_linear",
        event="both",
        lower_bound=-1.8,
        sample_limit=0,
    )
    assert out_pq.is_file()
    assert any("Kelvin" in w for w in det_era["warnings"])
    assert det_era["sample_events"] == []

    # 3. Other baseline_types ("moving", "fixed_threshold")
    out_mov = tmp_path / "cmip_mov.nc"
    det_mov = mcp_server.detect_extremes(
        str(paths["cmip"]),
        out=str(out_mov),
        baseline_type="moving",
        moving_window_years=3,
        event="high",
    )
    assert det_mov["n_events"] > 0

    out_ft = tmp_path / "cmip_ft.nc"
    det_ft = mcp_server.detect_extremes(
        str(paths["cmip"]),
        out=str(out_ft),
        baseline_type="fixed_threshold",
        threshold_high=24.0,
        threshold_climatology=20.0,
        event="high",
    )
    assert det_ft["n_events"] > 0

    # 4. query_events: O(1) CSR coords & indices, mutual exclusion check, and chunked sort/filter
    q_coord = mcp_server.query_events(str(out_nc), coords={"lat": -30.0, "lon": 150.0}, event_type="mhw")
    assert q_coord["total_matched"] >= 1
    assert all(ev["event_type"] == "high" for ev in q_coord["events"])

    q_idx = mcp_server.query_events(str(out_nc), indices={"lat": 1, "lon": 0}, event_type="mhw")
    assert q_idx["events"] == q_coord["events"]

    with pytest.raises(ValueError, match="not both"):
        mcp_server.query_events(str(out_nc), coords={"lat": -30.0, "lon": 150.0}, indices={"lat": 0, "lon": 0})

    q_filt = mcp_server.query_events(
        str(out_nc),
        event_type="mhw",
        min_duration=5,
        start_date="2000-01-01",
        end_date="2004-12-31",
        sort_by="int_max",
        descending=True,
        limit=3,
    )
    assert q_filt["returned"] == min(3, q_filt["total_matched"])
    assert q_filt["truncated"] == (q_filt["total_matched"] > q_filt["returned"])
    int_vals = [ev["int_max"] for ev in q_filt["events"]]
    assert int_vals == sorted(int_vals, reverse=True)
    assert json.dumps(q_filt, allow_nan=False)

    with pytest.raises(KeyError, match="Unknown sort_by"):
        mcp_server.query_events(str(out_nc), sort_by="no_such_col")

    # 5. summarize_result
    summary = mcp_server.summarize_result(str(out_nc))
    assert summary["n_events"] == det_nc["n_events"]
    assert summary["spatial_counts"]["n_pixels"] == 4
    assert summary["spatial_counts"]["pixels_with_events"] > 0
    assert "high" in summary["by_event_type"] and "low" in summary["by_event_type"]
    assert summary["duration"]["min"] >= 5
    assert summary["max_abs_intensity"] > 0
    assert json.dumps(summary, allow_nan=False)


def test_real_mcp_server_registration_and_missing_dep_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("mcp")
    paths = _write_sample_datasets(tmp_path)

    srv = mcp_server.create_mcp_server()
    tools = asyncio.run(srv.list_tools())
    tool_names = {t.name for t in tools}
    assert tool_names == {"inspect_dataset", "detect_extremes", "query_events", "summarize_result"}

    call_res = asyncio.run(srv.call_tool("inspect_dataset", {"path": str(paths["cmip"])}))
    payload = json.loads(call_res.content[0].text)
    assert payload["recommended_var"] == "tos"

    # Simulate missing optional mcp package cleanly
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "mcp.server.mcpserver", None)
        mp.setitem(sys.modules, "mcp.server.fastmcp", None)
        with pytest.raises(ImportError, match="marine-extremes\\[mcp\\]"):
            mcp_server.create_mcp_server()
