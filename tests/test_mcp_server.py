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
        content = call_res[0] if isinstance(call_res, tuple) and len(call_res) == 2 else call_res
    return content[0].text


def _is_error(res: object) -> bool:
    return bool(getattr(res, "is_error", getattr(res, "isError", False)))


def _write_sample_datasets(tmp_path: Path) -> dict[str, Path]:
    dates = pd.date_range("2000-01-01", "2004-12-31", freq="D")
    nt = dates.size
    rng = np.random.default_rng(123)
    seas = (20.0 + 3.0 * np.sin(2.0 * np.pi * np.arange(nt) / 365.25))[:, None, None]
    noise = rng.normal(0.0, 0.5, size=(nt, 4, 3))
    for t in range(1, nt):
        noise[t] = 0.8 * noise[t - 1] + 0.5 * noise[t]
    for s0, s1, amp in ((200, 212, 4.0), (600, 615, 4.5), (900, 910, -4.2), (1200, 1214, 3.8)):
        noise[s0:s1] += amp
    sst = seas + noise

    oisst_p = tmp_path / "oisst_multi.nc"
    ds_oisst = xr.Dataset(
        {
            "sst": (("time", "zlev", "lat", "lon"), sst[:, None, :, :], {"units": "Celsius"}),
            "anom": (("time", "zlev", "lat", "lon"), noise[:, None, :, :]),
            "err": (("time", "zlev", "lat", "lon"), np.full((nt, 1, 4, 3), 0.1)),
            "ice": (("time", "zlev", "lat", "lon"), np.zeros((nt, 1, 4, 3))),
        },
        coords={"time": dates, "zlev": [0.0], "lat": [-28.0, -29.0, -30.0, -31.0], "lon": [150.0, 151.0, 152.0]},
    )
    ds_oisst.to_netcdf(oisst_p)
    for idx, yr_mask in enumerate((dates.year <= 2001, dates.year >= 2002)):
        ds_oisst.isel(time=yr_mask).to_netcdf(tmp_path / f"oisst_part_{idx}.nc")

    era5_p = tmp_path / "era5_k.nc"
    xr.Dataset(
        {"sst": (("valid_time", "latitude", "longitude"), sst + 273.15, {"standard_name": "unknown", "units": "K"})},
        coords={"valid_time": dates, "latitude": [-29.0, -30.0, -31.0, -32.0], "longitude": [150.0, 151.0, 152.0]},
    ).to_netcdf(era5_p)

    cmip_p = tmp_path / "cmip_tos.nc"
    xr.Dataset(
        {
            "tos": (("time", "j", "i"), sst[:, :2, :2], {"standard_name": "sea_surface_temperature", "units": "degC", "grid_mapping": "crs"}),
            "time_bnds": (("time", "bnds"), np.zeros((nt, 2))),
            "crs": ((), 0, {"grid_mapping_name": "latitude_longitude"}),
        },
        coords={"time": ("time", dates, {"bounds": "time_bnds"}), "j": [0, 1], "i": [0, 1]},
    ).to_netcdf(cmip_p)

    return {"oisst": oisst_p, "era5": era5_p, "cmip": cmip_p, "glob": tmp_path / "oisst_part_*.nc"}


def test_inspect_dataset_covers_oisst_era5_cmip_and_glob(tmp_path: Path):
    paths = _write_sample_datasets(tmp_path)

    info_oisst = mcp_server.inspect_dataset(str(paths["oisst"]))
    assert info_oisst["recommended_var"] is None and info_oisst["singleton_dims"] == ["zlev"]
    assert len(info_oisst["data_vars"]) == 4 and all(not v["is_auxiliary"] for v in info_oisst["data_vars"])
    assert json.dumps(info_oisst, allow_nan=False)

    info_era5 = mcp_server.inspect_dataset(str(paths["era5"]))
    assert info_era5["recommended_var"] == "sst" and info_era5["time_dim"] == "valid_time"
    assert any("Kelvin" in n for n in info_era5["notes"])

    info_cmip = mcp_server.inspect_dataset(str(paths["cmip"]))
    assert info_cmip["recommended_var"] == "tos"
    assert {v["name"]: v["is_auxiliary"] for v in info_cmip["data_vars"]} == {"tos": False, "time_bnds": True, "crs": True}
    assert info_cmip["time_range"]["n_steps"] == 1827

    info_glob = mcp_server.inspect_dataset(str(paths["glob"]), var="sst")
    assert info_glob["n_files"] == 2 and info_glob["recommended_var"] == "sst" and info_glob["time_range"]["n_steps"] == 1827


def test_detect_query_and_summarize_roundtrip(tmp_path: Path):
    paths = _write_sample_datasets(tmp_path)

    out_nc = tmp_path / "oisst_sub_mhw.nc"
    det_nc = mcp_server.detect_extremes(
        str(paths["oisst"]), out=str(out_nc), var="sst", squeeze_singleton="zlev",
        event="both", region={"lat": [-30.0, -29.0], "lon": [150.0, 151.0]}, sample_limit=2,
    )
    assert out_nc.is_file() and det_nc["spatial_dims"] == ["lat", "lon"] and det_nc["spatial_shape"] == [2, 2]
    assert det_nc["n_events"] > 5 and len(det_nc["sample_events"]) == 2
    assert json.dumps(det_nc, allow_nan=False)

    out_pq = tmp_path / "era5_detrended.parquet"
    det_era = mcp_server.detect_extremes(
        str(paths["era5"]), out=str(out_pq), baseline_type="detrended_linear", event="both", lower_bound=-1.8, sample_limit=0,
    )
    assert out_pq.is_file() and any("Kelvin" in w for w in det_era["warnings"]) and det_era["sample_events"] == []

    assert mcp_server.detect_extremes(
        str(paths["cmip"]), out=str(tmp_path / "cmip_mov.nc"), baseline_type="moving", moving_window_years=3, event="high",
    )["n_events"] > 0
    assert mcp_server.detect_extremes(
        str(paths["cmip"]), out=str(tmp_path / "cmip_ft.nc"), baseline_type="fixed_threshold",
        threshold_high=24.0, threshold_climatology=20.0, event="high",
    )["n_events"] > 0

    q_coord = mcp_server.query_events(str(out_nc), coords={"lat": -30.0, "lon": 150.0}, event_type="mhw")
    assert q_coord["total_matched"] >= 1 and all(ev["event_type"] == "high" for ev in q_coord["events"])
    assert mcp_server.query_events(str(out_nc), indices={"lat": 1, "lon": 0}, event_type="mhw")["events"] == q_coord["events"]

    with pytest.raises(ValueError, match="not both"):
        mcp_server.query_events(str(out_nc), coords={"lat": -30.0, "lon": 150.0}, indices={"lat": 0, "lon": 0})

    q_filt = mcp_server.query_events(
        str(out_nc), event_type="mhw", min_duration=5, start_date="2000-01-01",
        end_date="2004-12-31", sort_by="int_max", descending=True, limit=3,
    )
    assert q_filt["returned"] == min(3, q_filt["total_matched"])
    assert q_filt["truncated"] == (q_filt["total_matched"] > q_filt["returned"])
    ints = [ev["int_max"] for ev in q_filt["events"]]
    assert ints == sorted(ints, reverse=True) and json.dumps(q_filt, allow_nan=False)

    with pytest.raises(KeyError, match="Unknown sort_by"):
        mcp_server.query_events(str(out_nc), sort_by="no_such_col")

    summary = mcp_server.summarize_result(str(out_nc))
    assert summary["n_events"] == det_nc["n_events"] and summary["spatial_counts"]["n_pixels"] == 4
    assert summary["spatial_counts"]["pixels_with_events"] > 0
    assert "high" in summary["by_event_type"] and "low" in summary["by_event_type"]
    assert summary["duration"]["min"] >= 5 and summary["max_abs_intensity"] > 0
    assert json.dumps(summary, allow_nan=False)


def test_real_mcp_server_registration_and_missing_dep_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("mcp")
    paths = _write_sample_datasets(tmp_path)

    srv = mcp_server.create_mcp_server()
    tool_map = {t.name: t for t in asyncio.run(srv.list_tools())}
    assert set(tool_map) == {"inspect_dataset", "detect_extremes", "query_events", "summarize_result"}

    props = (getattr(tool_map["detect_extremes"], "input_schema", None) or tool_map["detect_extremes"].inputSchema)["properties"]
    assert "fixed" in props["baseline_type"]["enum"] and "enum" in props["event"]
    assert "description" in props["path"] and "description" in props["out"]

    payload = json.loads(_extract_tool_text(asyncio.run(srv.call_tool("inspect_dataset", {"path": str(paths["cmip"])}))))
    assert payload["recommended_var"] == "tos"

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
            self.tools: dict[str, object] = {}

        def tool(self):
            return lambda fn: self.tools.setdefault(fn.__name__, fn)

    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    fastmcp_mod.FastMCP = _FakeFastMCP  # type: ignore[attr-defined]
    exc_mod = types.ModuleType("mcp.server.fastmcp.exceptions")
    exc_mod.ToolError = _FakeToolError  # type: ignore[attr-defined]

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "mcp.server.mcpserver", None)
        mp.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_mod)
        mp.setitem(sys.modules, "mcp.server.fastmcp.exceptions", exc_mod)
        srv = mcp_server.create_mcp_server()

    assert set(srv.tools) == {"inspect_dataset", "detect_extremes", "query_events", "summarize_result"}
    with pytest.raises(_FakeToolError, match="does not exist"):
        srv.tools["inspect_dataset"](path="/nonexistent_fastmcp_test.nc")


def test_mcp_stdio_subprocess_end_to_end(tmp_path: Path):
    pytest.importorskip("mcp")
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    paths = _write_sample_datasets(tmp_path)
    out_nc, out_era = tmp_path / "stdio_oisst.nc", tmp_path / "stdio_era5.parquet"
    cli_name = "marine-extremes-mcp.exe" if sys.platform == "win32" else "marine-extremes-mcp"
    cli_path = next(
        (p / cli_name for p in (Path(sys.executable).parent, Path(sysconfig.get_path("scripts") or "")) if (p / cli_name).is_file()),
        None,
    )
    primary = StdioServerParameters(command=str(cli_path) if cli_path else sys.executable, args=[] if cli_path else ["-m", "marine_extremes.mcp_server"])

    async def _run_primary() -> None:
        async with stdio_client(primary) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            assert {t.name for t in (await session.list_tools()).tools} == {"inspect_dataset", "detect_extremes", "query_events", "summarize_result"}

            r_ins = await session.call_tool("inspect_dataset", {"path": str(paths["oisst"])})
            assert not _is_error(r_ins) and json.loads(_extract_tool_text(r_ins))["singleton_dims"] == ["zlev"]

            r_det = await session.call_tool(
                "detect_extremes",
                {"path": str(paths["oisst"]), "out": str(out_nc), "var": "sst", "squeeze_singleton": "zlev",
                 "event": "both", "region": {"lat": [-30.0, -29.0], "lon": [150.0, 151.0]}, "sample_limit": 2},
            )
            p_det = json.loads(_extract_tool_text(r_det))
            assert not _is_error(r_det) and out_nc.is_file() and p_det["n_events"] > 5 and len(p_det["sample_events"]) == 2

            r_q = await session.call_tool("query_events", {"result_path": str(out_nc), "coords": {"lat": -30.0, "lon": 150.0}, "event_type": "mhw"})
            assert not _is_error(r_q) and json.loads(_extract_tool_text(r_q))["total_matched"] >= 1

            r_sum = await session.call_tool("summarize_result", {"result_path": str(out_nc)})
            assert not _is_error(r_sum) and json.loads(_extract_tool_text(r_sum))["n_events"] == p_det["n_events"]

            r_era = await session.call_tool("detect_extremes", {"path": str(paths["era5"]), "out": str(out_era), "lower_bound": -1.8, "sample_limit": 0})
            assert not _is_error(r_era) and out_era.is_file() and any("Kelvin" in w for w in json.loads(_extract_tool_text(r_era))["warnings"])

            r_err = await session.call_tool("inspect_dataset", {"path": str(tmp_path / "does_not_exist.nc")})
            assert _is_error(r_err) and "does not exist" in _extract_tool_text(r_err)

            r_oob = await session.call_tool("query_events", {"result_path": str(out_nc), "indices": {"lat": 999, "lon": 0}})
            assert _is_error(r_oob) and "out of bounds" in _extract_tool_text(r_oob)

    async def _run_module_handshake() -> None:
        mod_params = StdioServerParameters(command=sys.executable, args=["-m", "marine_extremes.mcp_server"])
        async with stdio_client(mod_params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            assert "detect_extremes" in {t.name for t in (await session.list_tools()).tools}

    asyncio.run(asyncio.wait_for(_run_primary(), timeout=60))
    if cli_path is not None:
        asyncio.run(asyncio.wait_for(_run_module_handshake(), timeout=60))


def test_to_jsonable_edge_cases():
    cftime = pytest.importorskip("cftime")
    ts_5dig = pa.Table.from_arrays([pa.array(np.array([np.datetime64("10001-02-03", "s")]))], names=["d"]).to_pandas()["d"].iloc[0]
    encoded = mcp_server._to_jsonable(
        {
            "nat": pd.NaT, "na": pd.NA, "nan": float("nan"), "inf": float("inf"),
            "arr0d_num": np.array(3.14), "arr0d_str": np.array("degC"), "arr1d": np.array([1, 2]),
            "td": np.timedelta64(6, "h"), "td_nat": np.timedelta64("NaT", "ns"),
            "arr0d_td": np.array(np.timedelta64(1, "D")),
            "arr_dt64": np.array([np.datetime64("2000-01-01", "ns"), np.datetime64("NaT", "ns")]),
            "ts_5dig": ts_5dig, "bytes": b"K", "np_bytes": np.bytes_(b"degK"),
            "cftime": cftime.DatetimeNoLeap(2000, 3, 1), "date": dt.date(2001, 2, 3), "ts_nat": pd.Timestamp("NaT"),
        }
    )
    assert encoded == {
        "nat": None, "na": None, "nan": None, "inf": None,
        "arr0d_num": 3.14, "arr0d_str": "degC", "arr1d": [1, 2],
        "td": "6 hours", "td_nat": None, "arr0d_td": "1 days",
        "arr_dt64": ["2000-01-01", None], "ts_5dig": "10001-02-03",
        "bytes": "K", "np_bytes": "degK", "cftime": "2000-03-01", "date": "2001-02-03", "ts_nat": None,
    }
    assert json.dumps(encoded, allow_nan=False)


def test_path_safety_region_and_literal_bracket_filename(tmp_path: Path):
    paths = _write_sample_datasets(tmp_path)

    orig = paths["cmip"].read_bytes()
    with pytest.raises(ValueError, match="overlaps input"):
        mcp_server.detect_extremes(str(paths["cmip"]), out=str(paths["cmip"]))
    assert paths["cmip"].read_bytes() == orig

    with pytest.raises(ValueError, match="overlaps input"):
        mcp_server.detect_extremes(str(paths["glob"]), out=str(sorted(tmp_path.glob("oisst_part_*.nc"))[0]), var="sst")

    keep_dir = tmp_path / "keep_dir"
    fake_zarr = tmp_path / "fake.zarr"
    for d, pat in ((keep_dir, "existing directory"), (fake_zarr, "non-Zarr directory")):
        d.mkdir()
        (d / "precious.txt").write_text("keep")
        with pytest.raises(ValueError, match=pat):
            mcp_server.detect_extremes(str(paths["cmip"]), out=str(d))
        assert (d / "precious.txt").read_text() == "keep"

    det_pt = mcp_server.detect_extremes(
        str(paths["oisst"]), out=str(tmp_path / "single_lat_slice.nc"), var="sst",
        squeeze_singleton=True, region={"lat": [-30.0, -30.0], "lon": [150.0, 151.0]},
    )
    assert det_pt["spatial_dims"] == ["lat", "lon"] and det_pt["spatial_shape"] == [1, 2]

    non_mono_nc = tmp_path / "non_mono.nc"
    xr.Dataset({"tos": (("time", "lon"), np.ones((5, 3)))}, coords={"time": pd.date_range("2000-01-01", periods=5), "lon": [180.0, 190.0, 10.0]}).to_netcdf(non_mono_nc)
    td_nc = tmp_path / "with_timedelta_coord.nc"
    xr.Dataset({"tos": (("time", "step"), np.ones((5, 3)))}, coords={"time": pd.date_range("2000-01-01", periods=5), "step": pd.to_timedelta([0, 1, 2], unit="D")}).to_netcdf(td_nc)
    info_td = mcp_server.inspect_dataset(str(td_nc))
    assert "min" not in info_td["spatial_coords"]["step"] and "dtype" in info_td["spatial_coords"]["step"]

    for src, reg, exc_cls, pat, var_name in (
        (paths["cmip"], {"time": [2000, 2001]}, ValueError, "time dimension", None),
        (paths["oisst"], {"lat": [float("nan"), -29.0]}, ValueError, "finite", "sst"),
        (paths["oisst"], {"lat": [False, True]}, TypeError, "numeric", "sst"),
        (non_mono_nc, {"lon": [10.0, 190.0]}, ValueError, "monotonic", None),
        (td_nc, {"step": [0, 1]}, ValueError, "numeric coordinate", None),
    ):
        with pytest.raises(exc_cls, match=pat):
            mcp_server.detect_extremes(str(src), out=str(tmp_path / "bad_reg.nc"), var=var_name, region=reg)  # type: ignore[arg-type]

    bracket_nc, sibling_nc = tmp_path / "sst[v2].nc", tmp_path / "sstv.nc"
    bracket_nc.write_bytes(orig)
    sibling_nc.write_bytes(b"sibling")
    assert mcp_server.inspect_dataset(str(bracket_nc))["recommended_var"] == "tos"
    n_br = mcp_server.detect_extremes(str(bracket_nc), out=str(tmp_path / "out_bracket.nc"))["n_events"]
    assert n_br > 0 and mcp_server.detect_extremes(str(bracket_nc), out=str(sibling_nc))["n_events"] == n_br
    assert sibling_nc.stat().st_size > 100

    with pytest.raises(ValueError, match="non-empty"):
        mcp_server.inspect_dataset("")
    with pytest.raises(ValueError, match="non-empty"):
        mcp_server.detect_extremes("", out=str(tmp_path / "empty_in.nc"))
    with pytest.raises(ValueError, match="sample_limit"):
        mcp_server.detect_extremes(str(paths["cmip"]), out=str(tmp_path / "neg_sl.nc"), sample_limit=-1)


def test_query_events_and_summarize_edge_cases(tmp_path: Path):
    cftime = pytest.importorskip("cftime")
    netCDF4 = pytest.importorskip("netCDF4")
    paths = _write_sample_datasets(tmp_path)
    out_nc = tmp_path / "events.nc"
    mcp_server.detect_extremes(str(paths["oisst"]), out=str(out_nc), var="sst", squeeze_singleton="zlev", event="both")

    q_year = mcp_server.query_events(str(out_nc), start_date="2001", end_date="2002", limit=500)
    assert q_year["total_matched"] > 0 and any(ev["date_start"].startswith("2002-") for ev in q_year["events"])
    assert all("2001-01-01" <= ev["date_start"] <= "2002-12-31" for ev in q_year["events"])

    q_both = mcp_server.query_events(str(out_nc), event_type="both", limit=0)
    assert q_both["total_matched"] > 0 and q_both["returned"] == 0 and q_both["events"] == []

    for kw, exc_cls, pat in (
        ({"end_date": "not-a-date"}, ValueError, "Could not parse date"),
        ({"event_type": "bogus"}, ValueError, "Unsupported event"),
        ({"limit": -1}, ValueError, "limit must be >= 0"),
        ({"coords": {"method": "exact", "lat": -30.0, "lon": 150.0}}, KeyError, "(?i)unexpected"),
        ({"coords": {"lat": -30.0, "lon": 150.0}, "min_duration": 99999, "sort_by": "no_such_col"}, KeyError, "Unknown sort_by"),
    ):
        with pytest.raises(exc_cls, match=pat):
            mcp_server.query_events(str(out_nc), **kw)  # type: ignore[arg-type]

    const_nc = tmp_path / "const_empty.nc"
    xr.Dataset(
        {"sst": (("time", "y", "x"), np.full((400, 2, 2), 20.0), {"units": "degK"})},
        coords={"time": pd.date_range("2000-01-01", periods=400, freq="D"), "y": [float("nan"), float("nan")], "x": [0.0, 1.0],
                "lat2d": (("y", "x"), np.array([[0.0, 1.0], [2.0, 3.0]]))},
    ).to_netcdf(const_nc)

    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always", RuntimeWarning)
        info_k = mcp_server.inspect_dataset(str(const_nc), var="sst")
        assert any("Kelvin" in n for n in info_k["notes"]) and any("Multidimensional coordinates" in n for n in info_k["notes"])
        assert info_k["spatial_coords"]["y"] == {"size": 2, "min": None, "max": None}

        out_empty = tmp_path / "empty_res.nc"
        mcp_server.detect_extremes(str(const_nc), out=str(out_empty), var="sst")
        sum_empty = mcp_server.summarize_result(str(out_empty))
        assert sum_empty["n_events"] == 0 and sum_empty["duration"] == {"min": None, "mean": None, "max": None} and sum_empty["max_abs_intensity"] is None

        out_nan_int = tmp_path / "nan_int_max.nc"
        mcp_server.detect_extremes(str(const_nc), out=str(out_nan_int), var="sst", event="high", baseline_type="fixed_threshold", threshold_high=19.0, threshold_climatology=18.0)
        with netCDF4.Dataset(out_nan_int, "r+") as nc:
            nc.groups["events"].variables["int_max"][:] = np.nan
        sum_nan_int = mcp_server.summarize_result(str(out_nan_int))
        assert sum_nan_int["n_events"] == 4 and sum_nan_int["max_abs_intensity"] is None
    assert not [w for w in wlog if issubclass(w.category, RuntimeWarning)]

    with pytest.raises(KeyError, match="Unknown sort_by"):
        mcp_server.query_events(str(out_empty), sort_by="no_such_col")

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
    xr.Dataset({"tos": (("time", "lat"), arr_cf[:, None], {"units": "degC"})}, coords={"time": t_5y, "lat": [-20.0]}).to_netcdf(cf_nc)
    out_cf = tmp_path / "cftime_res.nc"
    mcp_server.detect_extremes(str(cf_nc), out=str(out_cf), event="high")
    assert list(mcp_server.summarize_result(str(out_cf))["by_start_year"]) == ["9999", "10000"]


def test_multichunk_query_and_summarize_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from marine_extremes.catalog import EventCatalog

    paths = _write_sample_datasets(tmp_path)
    out_nc = tmp_path / "multi_chunk.nc"
    det = mcp_server.detect_extremes(str(paths["oisst"]), out=str(out_nc), var="sst", squeeze_singleton="zlev", event="both")
    assert det["n_events"] >= 20

    ref = (
        mcp_server.query_events(str(out_nc), event_type="mhw", limit=5),
        mcp_server.query_events(str(out_nc), sort_by="int_max", descending=True, limit=5),
        mcp_server.query_events(str(out_nc), event_type="mcs", limit=0),
        mcp_server.summarize_result(str(out_nc)),
    )

    orig_iter = EventCatalog.iter_chunks
    chunk_calls = 0

    def _small_chunks(self: EventCatalog, chunk_rows: int = 200_000):
        nonlocal chunk_calls
        for part in orig_iter(self, chunk_rows=3):
            chunk_calls += 1
            yield part

    monkeypatch.setattr(EventCatalog, "iter_chunks", _small_chunks)
    got = (
        mcp_server.query_events(str(out_nc), event_type="mhw", limit=5),
        mcp_server.query_events(str(out_nc), sort_by="int_max", descending=True, limit=5),
        mcp_server.query_events(str(out_nc), event_type="mcs", limit=0),
        mcp_server.summarize_result(str(out_nc)),
    )
    assert chunk_calls >= 12 and got == ref


def test_skill_markdown_frontmatter():
    skill_path = Path(__file__).resolve().parents[1] / "skills" / "marine-extremes" / "SKILL.md"
    assert skill_path.is_file()
    text = skill_path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    parts = text.split("\n---\n", 1)
    assert len(parts) == 2 and "name: marine-extremes" in parts[0] and "description:" in parts[0]
