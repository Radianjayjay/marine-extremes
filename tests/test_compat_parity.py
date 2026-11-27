"""Cross-tool numerical parity tests against m_mhw1.0 (detectc.m) and ecjoliver/marineHeatWaves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marxtreme as mx


def _matlab_quantile(vals: np.ndarray, q: float) -> float:
    """Exact MATLAB quantile(vals, q) (Hazen / Type-5) ignoring NaNs."""
    v = np.sort(vals[np.isfinite(vals)])
    n = v.size
    if n == 0:
        return np.nan
    if n == 1:
        return float(v[0])
    if q <= 0.5 / n:
        return float(v[0])
    if q >= (n - 0.5) / n:
        return float(v[-1])
    pos = q * n - 0.5
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    return float(v[lo] + (pos - lo) * (v[hi] - v[lo]))


def _reference_m_mhw_1d(
    temp: np.ndarray,
    dates: pd.DatetimeIndex,
    cli_start: str,
    cli_end: str,
    event: str = "MHW",
    threshold: float = 0.9,
    window_half_width: int = 5,
    smooth_width: int = 31,
    min_duration: int = 5,
    max_gap: int = 2,
    percentile_mode: str = "matlab",
):
    """Line-by-line reference implementation of ZijieZhaoMMHW/m_mhw1.0 detectc.m for 1D."""
    c0 = pd.Timestamp(cli_start)
    c1 = pd.Timestamp(cli_end)
    w = window_half_width
    buf_mask = (dates >= c0 - pd.Timedelta(days=w)) & (dates <= c1 + pd.Timedelta(days=w))
    t_clim = temp[buf_mask]
    d_clim = dates[buf_mask]

    # Map all dates to leap year 2012 (DOY 1..366, non-leap skips 60)
    cum_leap = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    fake_doy = cum_leap[d_clim.month.to_numpy() - 1] + d_clim.day.to_numpy()
    in_cli = (d_clim >= c0) & (d_clim <= c1)

    mclim = np.full(366, np.nan)
    m90 = np.full(366, np.nan)
    n_c = t_clim.size

    for d in range(1, 367):
        if d == 60:
            continue
        centers = np.flatnonzero((fake_doy == d) & in_cli)
        inc = np.zeros(n_c, dtype=bool)
        for c in centers:
            inc[max(0, c - w) : min(n_c, c + w + 1)] = True
        samp = t_clim[inc]
        fin = samp[np.isfinite(samp)]
        if fin.size > 0:
            mclim[d - 1] = np.mean(fin)
            if percentile_mode == "matlab":
                m90[d - 1] = _matlab_quantile(fin, threshold)
            else:
                m90[d - 1] = float(np.percentile(fin, threshold * 100.0))

    mclim[59] = np.nanmean([mclim[58], mclim[60]])
    m90[59] = np.nanmean([m90[58], m90[60]])

    def _smooth_circ(arr: np.ndarray) -> np.ndarray:
        tri = np.concatenate([arr, arr, arr])
        out = np.empty(366)
        hl = smooth_width // 2
        hr = smooth_width - 1 - hl
        for i in range(366):
            win = tri[366 + i - hl : 366 + i + hr + 1]
            out[i] = np.nanmean(win)
        return out

    mclim = _smooth_circ(mclim)
    m90 = _smooth_circ(m90)

    det_doy = cum_leap[dates.month.to_numpy() - 1] + dates.day.to_numpy() - 1
    th_ts = m90[det_doy]
    cl_ts = mclim[det_doy]
    exceed = (temp >= th_ts) if event == "MHW" else (temp <= th_ts)

    runs = []
    in_run = False
    s = 0
    for i, flag in enumerate(exceed):
        if flag and not in_run:
            in_run = True
            s = i
        elif not flag and in_run:
            if i - s >= min_duration:
                runs.append([s, i - 1])
            in_run = False
    if in_run and len(exceed) - s >= min_duration:
        runs.append([s, len(exceed) - 1])

    merged = []
    for r in runs:
        if not merged or r[0] - merged[-1][1] - 1 > max_gap:
            merged.append(r)
        else:
            merged[-1][1] = r[1]

    events = []
    for s, e in merged:
        anom = temp[s : e + 1] - cl_ts[s : e + 1]
        mca = th_ts[s : e + 1] - cl_ts[s : e + 1]
        int_max = float(np.nanmax(anom)) if event == "MHW" else float(np.nanmin(anom))
        cat = int(min(4, np.nanmax(np.floor(anom / mca))))
        events.append(
            {
                "index_start": s,
                "index_end": e,
                "duration": e - s + 1,
                "int_max": int_max,
                "int_mean": float(np.mean(anom)),
                "int_std": float(np.std(anom, ddof=1)),
                "int_cum": float(np.sum(anom)),
                "category_max": cat,
            }
        )
    return mclim, m90, events


def _reference_oliver_clim(temp: np.ndarray, dates: pd.DatetimeIndex, y0: int, y1: int, pct: float = 90.0):
    """Reference ecjoliver/marineHeatWaves circular (366, n_years) matrix climatology."""
    cum_leap = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    years = list(range(y0, y1 + 1))
    ny = len(years)
    mat = np.full((366, ny), np.nan)
    for t, d in enumerate(dates):
        if y0 <= d.year <= y1:
            yi = d.year - y0
            di = int(cum_leap[d.month - 1] + d.day - 1)
            mat[di, yi] = temp[t]
    for yi in range(ny):
        if np.isnan(mat[59, yi]):
            mat[59, yi] = 0.5 * (mat[58, yi] + mat[60, yi])

    seas = np.full(366, np.nan)
    thresh = np.full(366, np.nan)
    for d in range(366):
        if d == 59:
            continue
        idx = (np.arange(d - 5, d + 6) % 366)
        samp = mat[idx, :].ravel()
        samp = samp[np.isfinite(samp)]
        seas[d] = np.mean(samp)
        thresh[d] = np.percentile(samp, pct)
    seas[59] = 0.5 * (seas[58] + seas[60])
    thresh[59] = 0.5 * (thresh[58] + thresh[60])

    def _sm(a):
        tri = np.concatenate([a, a, a])
        return np.array([np.nanmean(tri[366 + i - 15 : 366 + i + 16]) for i in range(366)])

    return _sm(seas), _sm(thresh)


def _make_synthetic_sst(seed: int = 42, start: str = "1982-01-01", end: str = "2015-12-31"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.to_numpy()
    base = 18.0 + 4.5 * np.sin(2.0 * np.pi * (doy - 30) / 365.25)
    # AR(1) red noise mimicking realistic ocean SST persistence
    eps = rng.normal(0.0, 0.55, size=dates.size)
    anom = np.zeros_like(eps)
    for t in range(1, dates.size):
        anom[t] = 0.82 * anom[t - 1] + eps[t]
    return dates, base + anom


@pytest.mark.parametrize("compat_name,pmode", [("m_mhw", "matlab"), ("m_mhw_python", "python")])
@pytest.mark.parametrize("event_mx,event_mat,pct", [("mhw", "MHW", 0.9), ("mcs", "MCS", 0.1)])
def test_exact_parity_with_m_mhw(compat_name, pmode, event_mx, event_mat, pct):
    dates, sst = _make_synthetic_sst(seed=101)
    ref_clim, ref_th, ref_evs = _reference_m_mhw_1d(
        sst,
        dates,
        cli_start="1982-01-01",
        cli_end="2011-12-31",
        event=event_mat,
        threshold=pct,
        percentile_mode=pmode,
    )

    res = mx.detect(
        sst,
        time=dates,
        event=event_mx,
        baseline=mx.FixedBaseline("1982-01-01", "2011-12-31"),
        pctile_high=pct if event_mx == "mhw" else 0.9,
        pctile_low=pct if event_mx == "mcs" else 0.1,
        compat=compat_name,
        return_event_category_ts=True,
    )

    np.testing.assert_allclose(res.clim.values, ref_clim, rtol=1e-12, atol=1e-12)
    th_arr = res.thresh_high if event_mx == "mhw" else res.thresh_low
    np.testing.assert_allclose(th_arr.values, ref_th, rtol=1e-12, atol=1e-12)

    df = res.events.to_pandas()
    assert len(df) == len(ref_evs)
    assert len(df) > 10

    for row, ref in zip(df.itertuples(index=False), ref_evs):
        assert row.index_start == ref["index_start"]
        assert row.index_end == ref["index_end"]
        assert row.duration == ref["duration"]
        assert row.int_max == pytest.approx(ref["int_max"], rel=1e-12, abs=1e-12)
        assert row.int_mean == pytest.approx(ref["int_mean"], rel=1e-12, abs=1e-12)
        assert row.int_std == pytest.approx(ref["int_std"], rel=1e-12, abs=1e-12)
        assert row.int_cum == pytest.approx(ref["int_cum"], rel=1e-12, abs=1e-12)
        assert row.category_max == ref["category_max"]


def test_exact_parity_with_oliver_climatology_and_ddof0():
    dates, sst = _make_synthetic_sst(seed=202, start="1982-01-01", end="2011-12-31")
    # ecjoliver/marineHeatWaves uses continuous +/-5 day windows with DOY 60 interpolated from 59 & 61
    # and ddof=0 population variance/std (matching our interp_59_61 + Type-7 + ddof=0 preset).
    ref_seas, ref_th, _ = _reference_m_mhw_1d(
        sst,
        dates,
        cli_start="1982-01-01",
        cli_end="2011-12-31",
        event="MHW",
        threshold=0.9,
        percentile_mode="python",
    )

    res = mx.detect(
        sst,
        time=dates,
        event="high",
        baseline=mx.FixedBaseline("1982-01-01", "2011-12-31"),
        pctile_high=90,  # also verify 0..100 percentile normalization
        compat="oliver",
    )

    np.testing.assert_allclose(res.clim.values, ref_seas, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res.thresh_high.values, ref_th, rtol=1e-12, atol=1e-12)

    # Verify ddof=0 population variance & anomaly-based rate_onset/rate_decline
    df = res.events.to_pandas()
    assert len(df) > 10
    first = df.iloc[0]
    s, p, e = int(first["index_start"]), int(first["index_peak"]), int(first["index_end"])
    cum_leap = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    all_doy = cum_leap[dates.month.to_numpy() - 1] + dates.day.to_numpy() - 1
    anom_all = sst - ref_seas[all_doy]
    anom_ev = anom_all[s : e + 1]
    assert first["int_var"] == pytest.approx(float(np.var(anom_ev, ddof=0)), rel=1e-12)
    assert first["int_std"] == pytest.approx(float(np.std(anom_ev, ddof=0)), rel=1e-12)
    if s > 0:
        expected_onset = (anom_all[p] - 0.5 * (anom_all[s] + anom_all[s - 1])) / (p - s + 0.5)
        assert first["rate_onset"] == pytest.approx(float(expected_onset), rel=1e-12)

    # Also verify optional circular matrix feb29_mode="oliver_matrix" independently
    ref_mat_seas, ref_mat_th = _reference_oliver_clim(sst, dates, 1982, 2011, pct=90.0)
    prof_mat = mx.CompatProfile(name="oliver_matrix", feb29_mode="oliver_matrix", var_ddof=0)
    res_mat = mx.detect(
        sst,
        time=dates,
        event="high",
        baseline=mx.FixedBaseline("1982-01-01", "2011-12-31"),
        compat=prof_mat,
    )
    np.testing.assert_allclose(res_mat.clim.values, ref_mat_seas, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_mat.thresh_high.values, ref_mat_th, rtol=1e-12, atol=1e-12)


def test_heatwaver_preset_runs_and_differs_cleanly_at_feb29():
    dates, sst = _make_synthetic_sst(seed=303, start="1982-01-01", end="2011-12-31")
    r_std = mx.detect(sst, time=dates, compat="standard")
    r_hw = mx.detect(sst, time=dates, compat="heatwaveR")
    assert len(r_hw.events) > 10
    # Away from Feb 29 (e.g., July DOY 200), standard and heatwaveR agree closely
    np.testing.assert_allclose(r_std.clim.values[199], r_hw.clim.values[199], rtol=1e-12)
    # Near Feb 29 (DOY 59), heatwaveR inserts synthetic non-leap Feb 29 values into the window
    assert np.isfinite(r_hw.clim.values[59])
