"""Compatibility profiles governing exact numerical conventions across MHW/MCS tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QuantileMethod = Literal["type7", "type5"]
Feb29Mode = Literal["interp_59_61", "window_direct", "heatwaver_series", "oliver_matrix"]
SmoothNanPolicy = Literal["omit", "propagate"]


@dataclass(frozen=True)
class CompatProfile:
    """Explicit specification of numerical edge-case conventions.

    Attributes:
        name: Human-readable preset identifier.
        quantile_method: ``"type7"`` (NumPy/R default, ``heatwaveR``, ``marineHeatWaves``)
            or ``"type5"`` (Hazen / MATLAB ``quantile()``, ``m_mhw1.0`` default).
        strict_inequality: If True, exceedances use ``>`` / ``<`` (Hobday 2016,
            ``marineHeatWaves``, ``heatwaveR``). If False, use ``>=`` / ``<=`` (``m_mhw1.0``).
        feb29_mode: Handling of February 29 (DOY 60) in 366-day leap-aligned calendars.
        smooth_nan_policy: ``"omit"`` averages finite values inside the smoothing window;
            ``"propagate"`` returns NaN if any value inside the window is NaN.
        var_ddof: Degrees of freedom for event intensity variance/std (1 = sample, 0 = population).
    """

    name: str = "standard"
    quantile_method: QuantileMethod = "type7"
    strict_inequality: bool = True
    feb29_mode: Feb29Mode = "interp_59_61"
    smooth_nan_policy: SmoothNanPolicy = "omit"
    var_ddof: int = 1


PRESETS: dict[str, CompatProfile] = {
    "standard": CompatProfile(
        name="standard",
        quantile_method="type7",
        strict_inequality=True,
        feb29_mode="interp_59_61",
        smooth_nan_policy="omit",
        var_ddof=1,
    ),
    "m_mhw": CompatProfile(
        name="m_mhw",
        quantile_method="type5",
        strict_inequality=False,
        feb29_mode="interp_59_61",
        smooth_nan_policy="omit",
        var_ddof=1,
    ),
    "m_mhw_python": CompatProfile(
        name="m_mhw_python",
        quantile_method="type7",
        strict_inequality=False,
        feb29_mode="interp_59_61",
        smooth_nan_policy="omit",
        var_ddof=1,
    ),
    "oliver": CompatProfile(
        name="oliver",
        quantile_method="type7",
        strict_inequality=True,
        feb29_mode="interp_59_61",
        smooth_nan_policy="omit",
        var_ddof=0,
    ),
    "heatwaveR": CompatProfile(
        name="heatwaveR",
        quantile_method="type7",
        strict_inequality=True,
        feb29_mode="heatwaver_series",
        smooth_nan_policy="omit",
        var_ddof=1,
    ),
}


def resolve_compat(compat: str | CompatProfile) -> CompatProfile:
    """Resolve a preset name or CompatProfile instance."""
    if isinstance(compat, CompatProfile):
        return compat
    key = str(compat)
    if key in PRESETS:
        return PRESETS[key]
    # Case-insensitive fallback for convenience (e.g. "heatwaver")
    lower_map = {k.lower(): v for k, v in PRESETS.items()}
    if key.lower() in lower_map:
        return lower_map[key.lower()]
    valid = ", ".join(sorted(PRESETS.keys()))
    raise ValueError(f"Unknown compat preset '{compat}'. Expected one of: {valid}")
