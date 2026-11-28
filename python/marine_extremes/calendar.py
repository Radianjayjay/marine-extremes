"""Unified calendar encoding, period slicing, and bidirectional date/CF-time codecs."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa

# Cumulative days before each month (1-indexed month -> index month-1)
_CUM_DAYS_LEAP = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335], dtype=np.int32)
_CUM_DAYS_NOLEAP = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int32)

# Safe inclusive year range for pandas / NumPy datetime64[ns] without overflow.
PANDAS_NS_MIN_YEAR = 1678
PANDAS_NS_MAX_YEAR = 2261
CF_TIME_UNITS = "days since 1970-01-01 00:00:00"

# Calendars that cannot round-trip faithfully through proleptic-Gregorian np.datetime64
# (includes "julian" because e.g. 1900-02-29 exists in Julian even though its DOY year_len is 366).
NON_GREGORIAN_CALENDARS: frozenset[str] = frozenset(
    {"360_day", "noleap", "365_day", "all_leap", "366_day", "julian"}
)

_CF_TIME_UNITS_RE = re.compile(
    r"^\s*(?:seconds?|minutes?|hours?|days?|months?|years?)\s+since\s+", re.IGNORECASE
)
_DATE_RE = re.compile(
    r"^\s*(?:(\d{4})(\d{2})(\d{2})|(\d{1,6})(?:[-/](\d{1,2})(?:[-/](\d{1,2}))?)?)"
    r"(?:[T\s]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\s*$"
)
_ISO_FALLBACK_RE = re.compile(
    r"^\s*(\d{1,6})[-/](\d{1,2})[-/](\d{1,2})"
    r"(?:[T\s](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?\s*$"
)


def is_non_gregorian_calendar(calendar: str | None) -> bool:
    """Return True if ``calendar`` cannot safely round-trip via proleptic-Gregorian datetime64."""
    return bool(calendar and str(calendar).lower() in NON_GREGORIAN_CALENDARS)


def _years_in_ns_range(years: np.ndarray) -> bool:
    """Check whether all years fit inside Pandas/NumPy datetime64[ns] bounds [1678, 2261]."""
    if years.size == 0:
        return True
    return int(years.min()) >= PANDAS_NS_MIN_YEAR and int(years.max()) <= PANDAS_NS_MAX_YEAR


def _is_cftime_obj(obj: Any) -> bool:
    """Detect cftime datetime objects by duck typing."""
    return hasattr(obj, "calendar") and hasattr(obj, "year") and hasattr(obj, "month")


@dataclass(frozen=True)
class CalendarInfo:
    """Encoded calendar coordinates ready for the Rust engine and persistence."""

    years: np.ndarray      # int32, shape (n_time,)
    months: np.ndarray     # int16, shape (n_time,), 1..12
    days: np.ndarray       # int16, shape (n_time,), 1..31
    doys: np.ndarray       # uint16, shape (n_time,), 1..year_len
    year_len: int          # 366, 365, 360, or 12
    calendar: str          # standard, noleap, 360_day, all_leap, julian, etc.
    freq: Literal["daily", "monthly"]
    raw_time: np.ndarray   # original time coordinate values
    is_cftime: bool

    @property
    def ymd_keys(self) -> np.ndarray:
        """Sortable integer YYYYMMDD key safe across all calendars including 360_day."""
        return (
            self.years.astype(np.int64) * 10000
            + self.months.astype(np.int64) * 100
            + self.days.astype(np.int64)
        )

    def slice(self, sl: slice) -> "CalendarInfo":
        """Return a contiguous sub-calendar for ``sl``."""
        return CalendarInfo(
            years=np.ascontiguousarray(self.years[sl], dtype=np.int32),
            months=np.ascontiguousarray(self.months[sl], dtype=np.int16),
            days=np.ascontiguousarray(self.days[sl], dtype=np.int16),
            doys=np.ascontiguousarray(self.doys[sl], dtype=np.uint16),
            year_len=self.year_len,
            calendar=self.calendar,
            freq=self.freq,
            raw_time=self.raw_time[sl],
            is_cftime=self.is_cftime,
        )

    def extract_period_slice(self, start: Any | None, end: Any | None) -> slice:
        """Return contiguous slice covering ``[start, end]`` or raise ValueError."""
        mask = build_period_mask(self, start, end)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            raise ValueError("Detection period selected zero time steps.")
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        if not np.all(mask[lo:hi]):
            raise ValueError("Detection time range must be contiguous.")
        return slice(lo, hi)

    def slice_with_window_buffer(
        self,
        start: Any | None,
        end: Any | None,
        window_half_width: int,
    ) -> tuple[slice, np.ndarray]:
        """Return minimal contiguous time slice covering ``[start, end]`` plus ``+/- window_half_width``."""
        in_period = build_period_mask(self, start, end)
        idx = np.flatnonzero(in_period)
        if idx.size == 0:
            raise ValueError("Climatology period selected zero time steps.")
        lo = max(0, int(idx[0]) - int(window_half_width))
        hi = min(in_period.shape[0], int(idx[-1]) + int(window_half_width) + 1)
        return slice(lo, hi), np.ascontiguousarray(in_period[lo:hi], dtype=np.uint8)


def _check_strictly_increasing(raw: np.ndarray, vectorized: bool) -> None:
    """Reject unsorted or duplicate timestamps before indexing."""
    if raw.size <= 1:
        return
    bad = bool(np.any(raw[1:] <= raw[:-1])) if vectorized else any(raw[i] <= raw[i - 1] for i in range(1, raw.size))
    if bad:
        raise ValueError("time axis must be strictly increasing without duplicate timestamps")


def _extract_ymd_from_objects(objs: Any, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract contiguous (year, month, day) integer arrays from Python/cftime date objects."""
    years = np.fromiter((t.year for t in objs), dtype=np.int32, count=count)
    months = np.fromiter((t.month for t in objs), dtype=np.int16, count=count)
    days = np.fromiter((t.day for t in objs), dtype=np.int16, count=count)
    return years, months, days


def _ymd_from_datetime64(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract int32 years and int16 months/days from a strictly increasing ``datetime64`` array."""
    try:
        idx = pd.DatetimeIndex(raw)
        return (
            idx.year.to_numpy(dtype=np.int32),
            idx.month.to_numpy(dtype=np.int16),
            idx.day.to_numpy(dtype=np.int16),
        )
    except (OverflowError, pd.errors.OutOfBoundsDatetime):
        # Safe fallback for CMIP6/paleo/5-digit datetime64 outside pandas ns range
        d_strs = np.datetime_as_string(raw, unit="D")
        years = np.empty(raw.size, dtype=np.int32)
        months = np.empty(raw.size, dtype=np.int16)
        days = np.empty(raw.size, dtype=np.int16)
        for i, ds in enumerate(d_strs):
            s = str(ds).lstrip("+")
            neg = s.startswith("-")
            parts = (s[1:] if neg else s).split("-")
            years[i] = -int(parts[0]) if neg else int(parts[0])
            months[i] = int(parts[1])
            days[i] = int(parts[2])
        return years, months, days


def _extract_cf_time_attrs(time_values: Any) -> tuple[str | None, str | None]:
    """Read optional CF ``units`` and ``calendar`` attrs before ``np.asarray`` strips them."""
    cf_units: str | None = None
    cf_cal: str | None = None
    for src in (getattr(time_values, "attrs", None), getattr(time_values, "encoding", None)):
        if isinstance(src, dict):
            if cf_units is None and src.get("units") is not None:
                cf_units = str(src["units"]).strip()
            if cf_cal is None and src.get("calendar") is not None:
                cf_cal = str(src["calendar"]).strip().lower()
    return cf_units, cf_cal


def _parse_iso_strings_to_dt64(strs: list[str]) -> np.ndarray:
    """Parse date/datetime string arrays across both in-range ns and out-of-ns years."""
    try:
        try:
            idx = pd.to_datetime(strs, format="mixed")
        except (OverflowError, pd.errors.OutOfBoundsDatetime):
            raise
        except ValueError:
            idx = pd.to_datetime(strs, format="mixed", utc=True)
        if getattr(idx, "dtype", None) == object:
            idx = pd.to_datetime(strs, format="mixed", utc=True)
    except (OverflowError, pd.errors.OutOfBoundsDatetime, ValueError):
        parsed = []
        for s in strs:
            m_iso = _ISO_FALLBACK_RE.match(str(s))
            if m_iso:
                yy, mo, dd, hh, mm, ss, tz = m_iso.groups()
                dt_s = np.datetime64(
                    f"{int(yy):04d}-{int(mo):02d}-{int(dd):02d}"
                    f"T{int(hh or 0):02d}:{int(mm or 0):02d}:{int(ss or 0):02d}",
                    "s",
                )
                if tz and tz != "Z":
                    sign = 1 if tz[0] == "+" else -1
                    digits = tz[1:].replace(":", "")
                    off_s = sign * (int(digits[:2]) * 3600 + int(digits[2:4]) * 60)
                    dt_s = dt_s - np.timedelta64(off_s, "s")
                parsed.append(dt_s)
            else:
                key = date_to_ymd_key(s)
                y, m, d = key // 10000, (key % 10000) // 100, key % 100
                parsed.append(np.datetime64(f"{y:04d}-{m:02d}-{d:02d}T00:00:00", "s"))
        return np.array(parsed, dtype="datetime64[s]")
    if idx.isna().any():
        raise ValueError("time axis contains NaT values")
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.to_numpy()


def encode_time_axis(
    time_values: Any,
    freq: Literal["daily", "monthly"] = "daily",
    calendar: str | None = None,
    check_freq_unique: bool = True,
) -> CalendarInfo:
    """Encode a time axis into contiguous integer arrays for Rust.

    Supports ``numpy.datetime64``, ``pandas.DatetimeIndex``, ``xarray.CFTimeIndex``,
    arrays of ``cftime.datetime``, undecoded CF numeric coordinates, 8-digit ``YYYYMMDD``
    integers, ISO strings, and Python ``datetime.date``/``datetime.datetime``.
    """
    if freq not in ("daily", "monthly"):
        raise ValueError(f"freq must be 'daily' or 'monthly', got '{freq}'")

    cf_units, cf_cal = _extract_cf_time_attrs(time_values)
    if isinstance(time_values, pd.DatetimeIndex) and time_values.tz is not None:
        time_values = time_values.tz_convert("UTC").tz_localize(None)

    raw = np.asarray(time_values.values if hasattr(time_values, "values") else time_values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("time axis must be a non-empty 1D sequence")

    is_cftime = False
    detected_cal = calendar

    if np.issubdtype(raw.dtype, np.datetime64):
        if np.isnat(raw).any():
            raise ValueError("time axis contains NaT values")
        _check_strictly_increasing(raw, vectorized=True)
        years, months, days = _ymd_from_datetime64(raw)
    elif np.issubdtype(raw.dtype, np.bool_):
        raise ValueError("time axis cannot have boolean dtype")
    elif np.issubdtype(raw.dtype, np.number):
        if np.issubdtype(raw.dtype, np.floating) and not np.all(np.isfinite(raw)):
            raise ValueError("time axis contains missing/NaT values")
        if cf_units is not None and _CF_TIME_UNITS_RE.match(cf_units):
            try:
                import cftime
            except ImportError as exc:
                raise ImportError(
                    f"cftime is required to decode numeric CF time coordinates ('{cf_units}')."
                ) from exc
            eff_cal = (detected_cal or cf_cal or "standard").lower()
            decoded = np.asarray(cftime.num2date(raw, units=cf_units, calendar=eff_cal), dtype=object)
            _check_strictly_increasing(decoded, vectorized=False)
            years, months, days = _extract_ymd_from_objects(decoded, decoded.size)
            if not is_non_gregorian_calendar(eff_cal) and _years_in_ns_range(years):
                iso = [
                    f"{t.year:04d}-{t.month:02d}-{t.day:02d}T{t.hour:02d}:{t.minute:02d}:{t.second:02d}"
                    for t in decoded
                ]
                raw = np.array(iso, dtype="datetime64[ns]")
            else:
                raw = decoded
                is_cftime = True
                detected_cal = detected_cal or eff_cal
        elif np.issubdtype(raw.dtype, np.integer):
            vals_i = raw.astype(np.int64, copy=False)
            yr_i, mo_i, dy_i = vals_i // 10000, (vals_i % 10000) // 100, vals_i % 100
            if not (
                np.all((vals_i >= 10000101) & (vals_i <= 99991231))
                and np.all((mo_i >= 1) & (mo_i <= 12))
                and np.all((dy_i >= 1) & (dy_i <= 31))
            ):
                raise ValueError(
                    "Numeric time axis requires CF '<unit> since ...' units or valid 8-digit YYYYMMDD integers."
                )
            _check_strictly_increasing(vals_i, vectorized=True)
            iso = [f"{int(y):04d}-{int(m):02d}-{int(d):02d}" for y, m, d in zip(yr_i, mo_i, dy_i)]
            target_dt = "datetime64[ns]" if _years_in_ns_range(yr_i) else "datetime64[s]"
            try:
                raw = np.array(iso, dtype=target_dt)
            except ValueError as exc:
                raise ValueError(
                    "Numeric time axis requires CF '<unit> since ...' units or valid 8-digit YYYYMMDD integers."
                ) from exc
            years, months, days = _ymd_from_datetime64(raw)
        else:
            raise ValueError(
                "Numeric time axis requires CF '<unit> since ...' units or valid 8-digit YYYYMMDD integers."
            )
    else:
        if any(
            t is None or t is pd.NaT or (isinstance(t, (float, np.floating)) and np.isnan(t))
            for t in raw
        ):
            raise ValueError("time axis contains missing/NaT values")
        first = raw[0]
        if _is_cftime_obj(first):
            is_cftime = True
            _check_strictly_increasing(raw, vectorized=False)
            if detected_cal is None:
                detected_cal = str(getattr(first, "calendar", "standard")).lower()
            years, months, days = _extract_ymd_from_objects(raw, raw.size)
        elif isinstance(first, (dt.date, dt.datetime, pd.Timestamp)):
            if any(pd.isna(t) for t in raw):
                raise ValueError("time axis contains NaT values")
            has_tz = any(getattr(t, "tzinfo", None) is not None for t in raw)
            if has_tz:
                norm_objs = [
                    t.astimezone(dt.timezone.utc).replace(tzinfo=None)
                    if getattr(t, "tzinfo", None) is not None
                    else (t if isinstance(t, dt.datetime) else dt.datetime(t.year, t.month, t.day))
                    for t in raw
                ]
                yrs_tmp = np.fromiter((t.year for t in norm_objs), dtype=np.int64, count=len(norm_objs))
                dt_tgt = "datetime64[ns]" if _years_in_ns_range(yrs_tmp) else "datetime64[s]"
                raw = np.array(norm_objs, dtype=dt_tgt)
                _check_strictly_increasing(raw, vectorized=True)
                years, months, days = _ymd_from_datetime64(raw)
            else:
                _check_strictly_increasing(raw, vectorized=False)
                years, months, days = _extract_ymd_from_objects(raw, raw.size)
        elif isinstance(first, (str, bytes, bytearray, np.bytes_)) or raw.dtype.kind in ("U", "S", "T"):
            strs = [x.decode("utf-8") if isinstance(x, (bytes, bytearray, np.bytes_)) else str(x) for x in raw]
            raw = _parse_iso_strings_to_dt64(strs)
            _check_strictly_increasing(raw, vectorized=True)
            years, months, days = _ymd_from_datetime64(raw)
        else:
            idx = pd.DatetimeIndex(raw)
            if idx.isna().any():
                raise ValueError("time axis contains NaT values")
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            raw = idx.to_numpy()
            _check_strictly_increasing(raw, vectorized=True)
            years = idx.year.to_numpy(dtype=np.int32)
            months = idx.month.to_numpy(dtype=np.int16)
            days = idx.day.to_numpy(dtype=np.int16)

    cal_norm = (detected_cal or "standard").lower()

    if freq == "monthly":
        ym = years.astype(np.int64) * 12 + months.astype(np.int64)
        if check_freq_unique and ym.size > 1 and np.any(ym[1:] <= ym[:-1]):
            raise ValueError(
                "freq='monthly' requires strictly increasing unique (year, month) time steps "
                "(got multiple timestamps in the same month)."
            )
        year_len = 12
        doys = months.astype(np.uint16)
    else:
        ymd = years.astype(np.int64) * 10000 + months.astype(np.int64) * 100 + days.astype(np.int64)
        if check_freq_unique and ymd.size > 1 and np.any(ymd[1:] <= ymd[:-1]):
            raise ValueError(
                "freq='daily' requires strictly increasing unique calendar dates "
                "(got multiple timestamps on the same calendar day; resample sub-daily data to daily first)."
            )
        if cal_norm == "360_day":
            if np.any(days > 30):
                raise ValueError("360_day calendar cannot contain day > 30")
            year_len = 360
            doys = ((months.astype(np.int32) - 1) * 30 + days.astype(np.int32)).astype(np.uint16)
        elif cal_norm in ("noleap", "365_day"):
            if np.any((months == 2) & (days == 29)):
                raise ValueError(f"Calendar '{cal_norm}' cannot contain February 29")
            year_len = 365
            doys = (_CUM_DAYS_NOLEAP[months - 1] + days.astype(np.int32)).astype(np.uint16)
        elif cal_norm in ("all_leap", "366_day"):
            year_len = 366
            doys = (_CUM_DAYS_LEAP[months - 1] + days.astype(np.int32)).astype(np.uint16)
        else:
            # standard, gregorian, proleptic_gregorian, julian:
            # Align non-leap Mar 1..Dec 31 to DOY 61..366 so DOY 60 is reserved for Feb 29.
            year_len = 366
            doys = (_CUM_DAYS_LEAP[months - 1] + days.astype(np.int32)).astype(np.uint16)

    if int(doys.min()) < 1 or int(doys.max()) > year_len:
        raise ValueError(
            f"Encoded DOY out of bounds [1, {year_len}] for calendar '{cal_norm}' and freq '{freq}'"
        )

    return CalendarInfo(
        years=np.ascontiguousarray(years, dtype=np.int32),
        months=np.ascontiguousarray(months, dtype=np.int16),
        days=np.ascontiguousarray(days, dtype=np.int16),
        doys=np.ascontiguousarray(doys, dtype=np.uint16),
        year_len=year_len,
        calendar=cal_norm,
        freq=freq,
        raw_time=raw,
        is_cftime=is_cftime,
    )


def date_to_ymd_key(val: Any, *, is_end: bool = False) -> int:
    """Convert a scalar date specifier to an integer YYYYMMDD key without Gregorian-only crashes."""
    if isinstance(val, bool):
        raise TypeError("Unsupported date bound type: bool")
    if isinstance(val, (int, np.integer)):
        return int(val) * 10000 + (1231 if is_end else 101)
    if isinstance(val, str):
        m = _DATE_RE.match(val)
        if not m:
            raise ValueError(f"Could not parse date string: '{val}'")
        if m.group(1) is not None:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:
            y = int(m.group(4))
            mo = int(m.group(5)) if m.group(5) is not None else (12 if is_end else 1)
            d = int(m.group(6)) if m.group(6) is not None else (31 if is_end else 1)
        if not (1 <= mo <= 12) or not (1 <= d <= 31):
            raise ValueError(f"Invalid calendar month/day in date string: '{val}'")
        return y * 10000 + mo * 100 + d
    if isinstance(val, np.datetime64):
        if np.isnat(val):
            raise ValueError("Date bound cannot be NaT")
        parts = str(np.datetime_as_string(val, unit="D")).lstrip("+").split("-")
        return int(parts[0]) * 10000 + int(parts[1]) * 100 + int(parts[2])
    if hasattr(val, "year") and hasattr(val, "month") and hasattr(val, "day"):
        return int(val.year) * 10000 + int(val.month) * 100 + int(val.day)
    raise TypeError(f"Unsupported date bound type: {type(val)!r}")


def build_period_mask(
    cal: CalendarInfo,
    start: Any | None = None,
    end: Any | None = None,
) -> np.ndarray:
    """Build a contiguous uint8 mask (1 inside [start, end], 0 outside)."""
    keys = cal.ymd_keys
    mask = np.ones(keys.shape[0], dtype=bool)
    if start is not None:
        mask &= keys >= date_to_ymd_key(start, is_end=False)
    if end is not None:
        mask &= keys <= date_to_ymd_key(end, is_end=True)
    return np.ascontiguousarray(mask.astype(np.uint8))


def datetime64_resolution(raw_time: np.ndarray, is_cftime: bool) -> Literal["cftime", "ns", "s"]:
    """Determine a consistent date column resolution from the full ``raw_time`` axis."""
    if is_cftime:
        return "cftime"
    if raw_time.size == 0:
        return "ns"
    if np.issubdtype(raw_time.dtype, np.datetime64):
        yrs = raw_time.astype("datetime64[Y]").astype(np.int64) + 1970
        return "ns" if _years_in_ns_range(yrs) else "s"
    first = raw_time[0]
    if hasattr(first, "year"):
        yrs = np.fromiter((int(t.year) for t in raw_time), dtype=np.int64, count=raw_time.size)
        return "ns" if _years_in_ns_range(yrs) else "s"
    return "ns"


def coerce_time_for_zarr(det_time: np.ndarray) -> np.ndarray:
    """Coerce time arrays to Zarr/xarray-safe representations (incl. out-of-range years)."""
    arr = det_time
    if arr.dtype == object and arr.size > 0 and not hasattr(arr[0], "calendar"):
        dt_s = np.asarray(arr, dtype="datetime64[s]")
        yrs = dt_s.astype("datetime64[Y]").astype(np.int64) + 1970
        arr = dt_s.astype("datetime64[ns]") if _years_in_ns_range(yrs) else dt_s
    if np.issubdtype(arr.dtype, np.datetime64) and arr.size > 0:
        dt_s = arr.astype("datetime64[s]", copy=False)
        yrs = dt_s.astype("datetime64[Y]").astype(np.int64) + 1970
        if int(yrs.min()) < 1 or int(yrs.max()) > 9999:
            import cftime

            # xarray's Zarr/CF encoder falls back to cftime.date2num for years outside 1..9999,
            # which rejects raw datetime64 ints unless given cftime objects directly.
            cft = []
            for st in np.datetime_as_string(dt_s, unit="s"):
                date_p, time_p = str(st).split("T")
                neg = date_p.startswith("-")
                dp = date_p[1:] if neg else date_p
                yp, mp, dp_day = dp.split("-")
                hh, mm, ss = time_p.split(":")
                cft.append(
                    cftime.DatetimeProlepticGregorian(
                        -int(yp) if neg else int(yp), int(mp), int(dp_day), int(hh), int(mm), int(ss)
                    )
                )
            return np.asarray(cft, dtype=object)
    return arr


def format_date_column(raw_time: np.ndarray, indices: np.ndarray, is_cftime: bool) -> Any:
    """Map integer time indices to uniform date-only midnight values or cftime ISO strings."""
    res = datetime64_resolution(raw_time, is_cftime)
    if res == "cftime":
        if indices.size == 0:
            return pa.array([], type=pa.string())
        selected = raw_time[indices]
        return pa.array([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in selected], type=pa.string())
    target_dtype = f"datetime64[{res}]"
    if indices.size == 0:
        return np.array([], dtype=target_dtype)
    selected = raw_time[indices]
    if np.issubdtype(selected.dtype, np.datetime64):
        return selected.astype("datetime64[D]").astype(target_dtype, copy=False)
    return np.array([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in selected], dtype=target_dtype)


def dates_to_yyyymmdd(arr: Any) -> np.ndarray:
    """Convert Arrow/NumPy datetime, cftime, or ISO string date column to int32 YYYYMMDD."""
    if len(arr) == 0:
        return np.array([], dtype=np.int32)
    np_arr = arr.to_numpy(zero_copy_only=False) if isinstance(arr, (pa.Array, pa.ChunkedArray)) else np.asarray(arr)
    if np.issubdtype(np_arr.dtype, np.datetime64):
        try:
            idx = pd.DatetimeIndex(np_arr.astype("datetime64[s]", copy=False))
            return (
                idx.year.to_numpy(dtype=np.int32) * 10000
                + idx.month.to_numpy(dtype=np.int32) * 100
                + idx.day.to_numpy(dtype=np.int32)
            )
        except (OverflowError, pd.errors.OutOfBoundsDatetime):
            d_strs = np.datetime_as_string(np_arr, unit="D")
            return np.fromiter(
                (date_to_ymd_key(str(ds).lstrip("+")) for ds in d_strs),
                dtype=np.int32,
                count=len(d_strs),
            )
    return np.fromiter(
        (date_to_ymd_key(s if hasattr(s, "year") else str(s)) for s in np_arr),
        dtype=np.int32,
        count=len(np_arr),
    )


def yyyymmdd_to_dates(
    vals: np.ndarray,
    *,
    calendar: str | None = None,
    date_resolution: str | None = None,
) -> np.ndarray | pa.Array:
    """Convert int32 YYYYMMDD back to datetime64[ns]/datetime64[s] or ISO strings consistently."""
    vals = np.asarray(vals, dtype=np.int64)
    use_str = is_non_gregorian_calendar(calendar) or date_resolution == "cftime"
    if vals.size == 0:
        if use_str:
            return pa.array([], type=pa.string())
        return np.array([], dtype="datetime64[s]" if date_resolution == "s" else "datetime64[ns]")
    y = vals // 10000
    m = (vals % 10000) // 100
    d = vals % 100
    iso = [f"{int(yi):04d}-{int(mi):02d}-{int(di):02d}" for yi, mi, di in zip(y, m, d)]
    if use_str:
        return pa.array(iso, type=pa.string())
    target_dtype = "datetime64[s]" if (date_resolution == "s" or not _years_in_ns_range(y)) else "datetime64[ns]"
    try:
        return np.array(iso, dtype=target_dtype)
    except Exception:
        return pa.array(iso, type=pa.string())


def parse_csv_date_series(
    series: pd.Series,
    *,
    calendar: str | None = None,
    date_resolution: str | None = None,
) -> Any:
    """Decode a CSV date Series using the single calendar/resolution rule."""
    use_str = is_non_gregorian_calendar(calendar) or date_resolution == "cftime"
    if len(series) == 0:
        if use_str:
            return pd.Series([], dtype="string")
        return pd.Series([], dtype="datetime64[s]" if date_resolution == "s" else "datetime64[ns]")
    if use_str:
        return series.astype("string")
    if date_resolution == "s":
        try:
            return np.array(series.astype(str).tolist(), dtype="datetime64[s]")
        except Exception:
            return series.astype("string")
    try:
        parsed = pd.to_datetime(series, format="mixed")
        if parsed.isna().any():
            return series.astype("string")
        if date_resolution == "ns":
            return parsed.astype("datetime64[ns]")
        yrs = parsed.dt.year.to_numpy(dtype=np.int64)
        return parsed.astype("datetime64[ns]" if _years_in_ns_range(yrs) else "datetime64[s]")
    except Exception:
        try:
            arr = np.array(series.astype(str).tolist(), dtype="datetime64[s]")
            if np.isnat(arr).any():
                return series.astype("string")
            return arr
        except Exception:
            return series.astype("string")


def time_to_cf_days(raw_time: np.ndarray) -> tuple[np.ndarray, str, str]:
    """Encode a 1D time array into float64 numeric days for NetCDF/MAT/NPZ storage."""
    if raw_time.size == 0:
        return np.array([], dtype=np.float64), CF_TIME_UNITS, "proleptic_gregorian"
    if np.issubdtype(raw_time.dtype, np.datetime64):
        secs = raw_time.astype("datetime64[s]").astype(np.int64)
        return (secs / 86_400.0).astype(np.float64), CF_TIME_UNITS, "proleptic_gregorian"
    first = raw_time[0]
    if hasattr(first, "calendar"):
        import cftime

        cal = str(first.calendar)
        days = cftime.date2num(raw_time, units=CF_TIME_UNITS, calendar=cal)
        return np.asarray(days, dtype=np.float64), CF_TIME_UNITS, cal
    secs = np.asarray(raw_time, dtype="datetime64[s]").astype(np.int64)
    return (secs / 86_400.0).astype(np.float64), CF_TIME_UNITS, "proleptic_gregorian"


def cf_days_to_time(t_days: np.ndarray, t_units: str, t_cal: str) -> np.ndarray:
    """Decode float64 CF days back to datetime64[ns]/[s] or cftime object array."""
    t_days = np.asarray(t_days, dtype=np.float64).reshape(-1)
    if t_days.size == 0:
        return np.array([], dtype="datetime64[ns]")
    cal_l = str(t_cal).lower()
    # 1582-10-15 is -141427 days since 1970-01-01; earlier standard/gregorian dates follow Julian rules.
    can_use_dt64 = t_units == CF_TIME_UNITS and (
        cal_l == "proleptic_gregorian"
        or (cal_l in ("standard", "gregorian") and float(np.min(t_days)) >= -141427.0)
    )
    if can_use_dt64:
        secs = np.round(t_days * 86_400.0).astype(np.int64)
        dt_s = secs.astype("datetime64[s]")
        yrs = dt_s.astype("datetime64[Y]").astype(np.int64) + 1970
        return dt_s.astype("datetime64[ns]") if _years_in_ns_range(yrs) else dt_s
    import cftime

    return np.asarray(cftime.num2date(t_days, units=t_units, calendar=t_cal), dtype=object)


def effective_feb29_mode(cal: CalendarInfo, requested_mode: str) -> str:
    """Adapt feb29_mode for non-leap / 360-day / all-leap / monthly calendars."""
    if cal.freq == "monthly" or cal.year_len != 366:
        return "window_direct"
    if cal.calendar in ("all_leap", "366_day"):
        return "window_direct"
    return requested_mode
