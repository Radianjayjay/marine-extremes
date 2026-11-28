"""Tests for the optional marine-extremes stdio MCP server tools."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import sys
import sysconfig
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr

from marine_extremes import mcp_server


def _extract_tool_text(call_res: object) -> str:
    """Extract text payload across MCP SDK v2 CallToolResult and v1 FastMCP return shapes."""
    content = getattr(call_res, "content", None)
    if content is None:
        if isinstance(call_res, tuple) and len(call_res) == 2:
            content = call_res[0]
        else:
            content = call_res
    return content[0].text


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
    tool_map = {t.name: t for t in tools}
    assert set(tool_map) == {"inspect_dataset", "detect_extremes", "query_events", "summarize_result"}

    det_schema = getattr(tool_map["detect_extremes"], "input_schema", None) or tool_map["detect_extremes"].inputSchema
    props = det_schema["properties"]
    assert "enum" in props["baseline_type"]
    assert "fixed" in props["baseline_type"]["enum"]
    assert "enum" in props["event"]
    assert "description" in props["path"] and "description" in props["out"]

    call_res = asyncio.run(srv.call_tool("inspect_dataset", {"path": str(paths["cmip"])}))
    payload = json.loads(_extract_tool_text(call_res))
    assert payload["recommended_var"] == "tos"

    # Simulate missing optional mcp package cleanly
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "mcp.server.mcpserver", None)
        mp.setitem(sys.modules, "mcp.server.fastmcp", None)
        with pytest.raises(ImportError, match="marine-extremes\\[mcp\\]"):
            mcp_server.create_mcp_server()


def test_mcp_v1_fastmcp_fallback_and_tool_error(monkeypatch: pytest.MonkeyPatch):
    class _FakeToolError(Exception):
        pass

    class _FakeFastMCP:
        def __init__(self, name: str, instructions: str = "") -> None:
            self.name = name
            self.instructions = instructions
            self.tools: dict[str, object] = {}

        def tool(self):
            def _dec(fn):
                self.tools[fn.__name__] = fn
                return fn

            return _dec

    fake_fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    fake_fastmcp_mod.FastMCP = _FakeFastMCP  # type: ignore[attr-defined]
    fake_exc_mod = types.ModuleType("mcp.server.fastmcp.exceptions")
    fake_exc_mod.ToolError = _FakeToolError  # type: ignore[attr-defined]

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "mcp.server.mcpserver", None)
        mp.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp_mod)
        mp.setitem(sys.modules, "mcp.server.fastmcp.exceptions", fake_exc_mod)
        srv = mcp_server.create_mcp_server()

    assert set(srv.tools) == {"inspect_dataset", "detect_extremes", "query_events", "summarize_result"}
    with pytest.raises(_FakeToolError, match="does not exist"):
        srv.tools["inspect_dataset"](path="/nonexistent_fastmcp_test.nc")


def test_mcp_stdio_subprocess_end_to_end(tmp_path: Path):
    pytest.importorskip("mcp")
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    paths = _write_sample_datasets(tmp_path)
    out_nc = tmp_path / "stdio_oisst.nc"
    out_era = tmp_path / "stdio_era5.parquet"

    def _is_error(res: object) -> bool:
        return bool(getattr(res, "is_error", getattr(res, "isError", False)))

    def _json_payload(res: object) -> dict[str, object]:
        return json.loads(_extract_tool_text(res))

    cli_name = "marine-extremes-mcp.exe" if sys.platform == "win32" else "marine-extremes-mcp"
    cli_candidates = [Path(sys.executable).parent / cli_name]
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        cli_candidates.append(Path(scripts_dir) / cli_name)
    cli_path = next((p for p in cli_candidates if p.is_file()), None)
    if cli_path is not None:
        primary_params = StdioServerParameters(command=str(cli_path), args=[])
        has_cli = True
    else:
        primary_params = StdioServerParameters(command=sys.executable, args=["-m", "marine_extremes.mcp_server"])
        has_cli = False

    async def _run_primary() -> None:
        async with stdio_client(primary_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_res = await session.list_tools()
                assert {t.name for t in tools_res.tools} == {
                    "inspect_dataset",
                    "detect_extremes",
                    "query_events",
                    "summarize_result",
                }

                r_ins = await session.call_tool("inspect_dataset", {"path": str(paths["oisst"])})
                assert not _is_error(r_ins)
                p_ins = _json_payload(r_ins)
                assert p_ins["singleton_dims"] == ["zlev"]
                assert p_ins["recommended_var"] is None

                r_det = await session.call_tool(
                    "detect_extremes",
                    {
                        "path": str(paths["oisst"]),
                        "out": str(out_nc),
                        "var": "sst",
                        "squeeze_singleton": "zlev",
                        "event": "both",
                        "region": {"lat": [-30.0, -29.0], "lon": [150.0, 151.0]},
                        "sample_limit": 2,
                    },
                )
                assert not _is_error(r_det)
                p_det = _json_payload(r_det)
                assert out_nc.is_file()
                assert p_det["n_events"] > 5
                assert len(p_det["sample_events"]) == 2

                r_q = await session.call_tool(
                    "query_events",
                    {
                        "result_path": str(out_nc),
                        "coords": {"lat": -30.0, "lon": 150.0},
                        "event_type": "mhw",
                    },
                )
                assert not _is_error(r_q)
                p_q = _json_payload(r_q)
                assert p_q["total_matched"] >= 1

                r_sum = await session.call_tool("summarize_result", {"result_path": str(out_nc)})
                assert not _is_error(r_sum)
                p_sum = _json_payload(r_sum)
                assert p_sum["n_events"] == p_det["n_events"]

                # Ensure UserWarning from Kelvin lower_bound is captured cleanly without corrupting stdio JSON-RPC
                r_era = await session.call_tool(
                    "detect_extremes",
                    {
                        "path": str(paths["era5"]),
                        "out": str(out_era),
                        "lower_bound": -1.8,
                        "sample_limit": 0,
                    },
                )
                assert not _is_error(r_era)
                p_era = _json_payload(r_era)
                assert out_era.is_file()
                assert any("Kelvin" in w for w in p_era["warnings"])

                # Ensure ToolError preserves diagnostic messages for FileNotFoundError and IndexError
                r_err = await session.call_tool(
                    "inspect_dataset",
                    {"path": str(tmp_path / "does_not_exist.nc")},
                )
                assert _is_error(r_err)
                assert "does not exist" in _extract_tool_text(r_err)

                r_oob = await session.call_tool(
                    "query_events",
                    {"result_path": str(out_nc), "indices": {"lat": 999, "lon": 0}},
                )
                assert _is_error(r_oob)
                assert "out of bounds" in _extract_tool_text(r_oob)

    async def _run_module_handshake() -> None:
        params = StdioServerParameters(command=sys.executable, args=["-m", "marine_extremes.mcp_server"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_res = await session.list_tools()
                assert {t.name for t in tools_res.tools} == {
                    "inspect_dataset",
                    "detect_extremes",
                    "query_events",
                    "summarize_result",
                }

    asyncio.run(asyncio.wait_for(_run_primary(), timeout=60))
    if has_cli:
        asyncio.run(asyncio.wait_for(_run_module_handshake(), timeout=60))


def test_to_jsonable_edge_cases():
    cftime = pytest.importorskip("cftime")
    ts_5dig = pa.Table.from_arrays(
        [pa.array(np.array([np.datetime64("10001-02-03", "s")]))],
        names=["d"],
    ).to_pandas()["d"].iloc[0]
    sample = {
        "nat": pd.NaT,
        "na": pd.NA,
        "nan": float("nan"),
        "inf": float("inf"),
        "arr0d_num": np.array(3.14),
        "arr0d_str": np.array("degC"),
        "arr1d": np.array([1, 2]),
        "td": np.timedelta64(6, "h"),
        "td_nat": np.timedelta64("NaT", "ns"),
        "arr0d_td": np.array(np.timedelta64(1, "D")),
        "arr_dt64": np.array([np.datetime64("2000-01-01", "ns"), np.datetime64("NaT", "ns")]),
        "ts_5dig": ts_5dig,
        "bytes": b"K",
        "np_bytes": np.bytes_(b"degK"),
        "cftime": cftime.DatetimeNoLeap(2000, 3, 1),
        "date": dt.date(2001, 2, 3),
        "ts_nat": pd.Timestamp("NaT"),
    }
    encoded = mcp_server._to_jsonable(sample)
    assert encoded == {
        "nat": None,
        "na": None,
        "nan": None,
        "inf": None,
        "arr0d_num": 3.14,
        "arr0d_str": "degC",
        "arr1d": [1, 2],
        "td": "6 hours",
        "td_nat": None,
        "arr0d_td": "1 days",
        "arr_dt64": ["2000-01-01", None],
        "ts_5dig": "10001-02-03",
        "bytes": "K",
        "np_bytes": "degK",
        "cftime": "2000-03-01",
        "date": "2001-02-03",
        "ts_nat": None,
    }
    assert json.dumps(encoded, allow_nan=False)


def test_path_safety_region_and_literal_bracket_filename(tmp_path: Path):
    paths = _write_sample_datasets(tmp_path)

    # Reject out == input path and out matching a glob input member
    orig_bytes = paths["cmip"].read_bytes()
    with pytest.raises(ValueError, match="overlaps input"):
        mcp_server.detect_extremes(str(paths["cmip"]), out=str(paths["cmip"]))
    assert paths["cmip"].read_bytes() == orig_bytes

    glob_member = sorted(tmp_path.glob("oisst_part_*.nc"))[0]
    with pytest.raises(ValueError, match="overlaps input"):
        mcp_server.detect_extremes(str(paths["glob"]), out=str(glob_member), var="sst")

    # Reject overwriting non-Zarr directories (even with .zarr extension)
    keep_dir = tmp_path / "keep_dir"
    keep_dir.mkdir()
    (keep_dir / "precious.txt").write_text("keep")
    with pytest.raises(ValueError, match="existing directory"):
        mcp_server.detect_extremes(str(paths["cmip"]), out=str(keep_dir))
    assert (keep_dir / "precious.txt").read_text() == "keep"

    fake_zarr = tmp_path / "fake.zarr"
    fake_zarr.mkdir()
    (fake_zarr / "precious.txt").write_text("keep")
    with pytest.raises(ValueError, match="non-Zarr directory"):
        mcp_server.detect_extremes(str(paths["cmip"]), out=str(fake_zarr))
    assert (fake_zarr / "precious.txt").read_text() == "keep"

    # Single-point region slice + squeeze_singleton=True preserves the sliced spatial dim
    out_pt = tmp_path / "single_lat_slice.nc"
    det_pt = mcp_server.detect_extremes(
        str(paths["oisst"]),
        out=str(out_pt),
        var="sst",
        squeeze_singleton=True,
        region={"lat": [-30.0, -30.0], "lon": [150.0, 151.0]},
    )
    assert det_pt["spatial_dims"] == ["lat", "lon"]
    assert det_pt["spatial_shape"] == [1, 2]

    # Reject invalid region specs (time dim, non-monotonic coord, NaN bound, bool bound)
    with pytest.raises(ValueError, match="time dimension"):
        mcp_server.detect_extremes(
            str(paths["cmip"]),
            out=str(tmp_path / "bad_time_reg.nc"),
            region={"time": [2000, 2001]},
        )
    with pytest.raises(ValueError, match="finite"):
        mcp_server.detect_extremes(
            str(paths["oisst"]),
            out=str(tmp_path / "bad_nan_reg.nc"),
            var="sst",
            region={"lat": [float("nan"), -29.0]},
        )
    with pytest.raises(TypeError, match="numeric"):
        mcp_server.detect_extremes(
            str(paths["oisst"]),
            out=str(tmp_path / "bad_bool_reg.nc"),
            var="sst",
            region={"lat": [False, True]},  # type: ignore[dict-item]
        )

    non_mono_nc = tmp_path / "non_mono.nc"
    xr.Dataset(
        {"tos": (("time", "lon"), np.ones((5, 3)))},
        coords={"time": pd.date_range("2000-01-01", periods=5), "lon": [180.0, 190.0, 10.0]},
    ).to_netcdf(non_mono_nc)
    with pytest.raises(ValueError, match="monotonic"):
        mcp_server.detect_extremes(
            str(non_mono_nc),
            out=str(tmp_path / "bad_mono.nc"),
            region={"lon": [10.0, 190.0]},
        )

    # Literal filename containing '[' must not fail glob expansion or bypass path guard when sibling exists
    bracket_nc = tmp_path / "sst[v2].nc"
    bracket_nc.write_bytes(paths["cmip"].read_bytes())
    sibling_nc = tmp_path / "sstv.nc"
    sibling_nc.write_bytes(b"sibling")
    info_br = mcp_server.inspect_dataset(str(bracket_nc))
    assert info_br["recommended_var"] == "tos"

    det_br = mcp_server.detect_extremes(str(bracket_nc), out=str(tmp_path / "out_bracket.nc"))
    assert det_br["n_events"] > 0
    det_sib = mcp_server.detect_extremes(str(bracket_nc), out=str(sibling_nc))
    assert det_sib["n_events"] == det_br["n_events"]
    assert sibling_nc.stat().st_size > 100

    # Reject empty path / output / negative sample_limit cleanly
    with pytest.raises(ValueError, match="non-empty"):
        mcp_server.inspect_dataset("")
    with pytest.raises(ValueError, match="non-empty"):
        mcp_server.detect_extremes("", out=str(tmp_path / "empty_in.nc"))
    with pytest.raises(ValueError, match="sample_limit"):
        mcp_server.detect_extremes(str(paths["cmip"]), out=str(tmp_path / "neg_sl.nc"), sample_limit=-1)

    # timedelta64 coordinate must be reported as non-numeric in inspect_dataset and rejected in region slicing
    td_nc = tmp_path / "with_timedelta_coord.nc"
    xr.Dataset(
        {"tos": (("time", "step"), np.ones((5, 3)))},
        coords={
            "time": pd.date_range("2000-01-01", periods=5),
            "step": pd.to_timedelta([0, 1, 2], unit="D"),
        },
    ).to_netcdf(td_nc)
    info_td = mcp_server.inspect_dataset(str(td_nc))
    assert "min" not in info_td["spatial_coords"]["step"]
    assert "dtype" in info_td["spatial_coords"]["step"]
    with pytest.raises(ValueError, match="numeric coordinate"):
        mcp_server.detect_extremes(
            str(td_nc),
            out=str(tmp_path / "bad_td_reg.nc"),
            region={"step": [0, 1]},
        )


def test_query_events_and_summarize_edge_cases(tmp_path: Path):
    cftime = pytest.importorskip("cftime")
    paths = _write_sample_datasets(tmp_path)
    out_nc = tmp_path / "events.nc"
    mcp_server.detect_extremes(
        str(paths["oisst"]),
        out=str(out_nc),
        var="sst",
        squeeze_singleton="zlev",
        event="both",
    )

    # Partial YYYY / YYYY-MM date filters work properly via calendar keys; invalid dates raise ValueError
    q_year = mcp_server.query_events(str(out_nc), start_date="2001", end_date="2002", limit=500)
    assert q_year["total_matched"] > 0
    assert any(ev["date_start"].startswith("2002-") for ev in q_year["events"])
    assert all("2001-01-01" <= ev["date_start"] <= "2002-12-31" for ev in q_year["events"])

    with pytest.raises(ValueError, match="Could not parse date"):
        mcp_server.query_events(str(out_nc), end_date="not-a-date")

    # event_type="both" is allowed; invalid event_type raises ValueError even when limit=0 / empty
    q_both = mcp_server.query_events(str(out_nc), event_type="both", limit=0)
    assert q_both["total_matched"] > 0 and q_both["returned"] == 0 and q_both["events"] == []
    with pytest.raises(ValueError, match="Unsupported event_type"):
        mcp_server.query_events(str(out_nc), event_type="bogus")
    with pytest.raises(ValueError, match="limit must be >= 0"):
        mcp_server.query_events(str(out_nc), limit=-1)

    # coords={"method": ...} cannot hijack EventCatalog.sel(method=...), and sort_by validates on empty df
    with pytest.raises(KeyError, match="unexpected"):
        mcp_server.query_events(str(out_nc), coords={"method": "exact", "lat": -30.0, "lon": 150.0})
    with pytest.raises(KeyError, match="Unknown sort_by"):
        mcp_server.query_events(
            str(out_nc),
            coords={"lat": -30.0, "lon": 150.0},
            min_duration=99999,
            sort_by="no_such_col",
        )

    # Empty event catalog (0 events) + NaN int_max guard + all-NaN spatial coord without RuntimeWarning
    const_nc = tmp_path / "const_empty.nc"
    dates = pd.date_range("2000-01-01", periods=400, freq="D")
    xr.Dataset(
        {
            "sst": (
                ("time", "y", "x"),
                np.full((400, 2, 2), 20.0),
                {"units": "degK"},
            ),
        },
        coords={
            "time": dates,
            "y": [float("nan"), float("nan")],
            "x": [0.0, 1.0],
            "lat2d": (("y", "x"), np.array([[0.0, 1.0], [2.0, 3.0]])),
        },
    ).to_netcdf(const_nc)

    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always", RuntimeWarning)
        info_k = mcp_server.inspect_dataset(str(const_nc), var="sst")
        assert any("Kelvin" in n for n in info_k["notes"])
        assert any("Multidimensional coordinates" in n for n in info_k["notes"])
        assert info_k["spatial_coords"]["y"] == {"size": 2, "min": None, "max": None}

        out_empty = tmp_path / "empty_res.nc"
        mcp_server.detect_extremes(str(const_nc), out=str(out_empty), var="sst")
        sum_empty = mcp_server.summarize_result(str(out_empty))
        assert sum_empty["n_events"] == 0
        assert sum_empty["duration"] == {"min": None, "mean": None, "max": None}
        assert sum_empty["max_abs_intensity"] is None

        # Non-empty events with all-NaN int_max emit zero RuntimeWarning
        out_nan_int = tmp_path / "nan_int_max.nc"
        mcp_server.detect_extremes(
            str(const_nc),
            out=str(out_nan_int),
            var="sst",
            event="high",
            baseline_type="fixed_threshold",
            threshold_high=19.0,
            threshold_climatology=18.0,
        )
        netCDF4 = pytest.importorskip("netCDF4")

        with netCDF4.Dataset(out_nan_int, "r+") as nc:
            nc.groups["events"].variables["int_max"][:] = np.nan
        sum_nan_int = mcp_server.summarize_result(str(out_nan_int))
        assert sum_nan_int["n_events"] == 4
        assert sum_nan_int["max_abs_intensity"] is None
    assert not [w for w in wlog if issubclass(w.category, RuntimeWarning)]

    with pytest.raises(KeyError, match="Unknown sort_by"):
        mcp_server.query_events(str(out_empty), sort_by="no_such_col")

    # 5-digit cftime noleap roundtrip proves year extraction does not truncate year >= 10000
    # and by_start_year sorts numerically (9999 before 10000, not lexicographic)
    t_5y = [
        cftime.DatetimeNoLeap(yr, mo, d)
        for yr in (9999, 10000, 10001)
        for mo, nd in ((1, 31), (2, 28), (3, 31), (4, 30), (5, 31), (6, 30), (7, 31), (8, 31), (9, 30), (10, 31), (11, 30), (12, 31))
        for d in range(1, nd + 1)
    ]
    arr_cf = np.full(len(t_5y), 20.0)
    arr_cf[30:45] = 26.0
    arr_cf[365 + 30 : 365 + 45] = 26.0
    cf_nc = tmp_path / "cftime_10000.nc"
    xr.Dataset(
        {"tos": (("time", "lat"), arr_cf[:, None], {"units": "degC"})},
        coords={"time": t_5y, "lat": [-20.0]},
    ).to_netcdf(cf_nc)
    out_cf = tmp_path / "cftime_res.nc"
    mcp_server.detect_extremes(str(cf_nc), out=str(out_cf), event="high")
    sum_cf = mcp_server.summarize_result(str(out_cf))
    assert list(sum_cf["by_start_year"]) == ["9999", "10000"]


def test_skill_markdown_frontmatter():
    skill_path = Path(__file__).resolve().parents[1] / "skills" / "marine-extremes" / "SKILL.md"
    assert skill_path.is_file()
    text = skill_path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    parts = text.split("\n---\n", 1)
    assert len(parts) == 2
    frontmatter = parts[0]
    assert "name: marine-extremes" in frontmatter
    assert "description:" in frontmatter
