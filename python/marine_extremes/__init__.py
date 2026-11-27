"""marine-extremes: Rust-accelerated detection of marine heatwaves, cold spells, and marine extremes."""

from marine_extremes.api import CATEGORY_MISSING, compute_climatology, detect
from marine_extremes.baselines import (
    DetrendedBaseline,
    ExternalClimatology,
    FixedBaseline,
    FixedThreshold,
    MovingBaseline,
)
from marine_extremes.catalog import EventCatalog
from marine_extremes.compat import PRESETS as COMPAT_PRESETS
from marine_extremes.compat import CompatProfile, resolve_compat
from marine_extremes.result import DetectResult

# Top-level convenience alias for reopening any self-contained marine-extremes output.
open = DetectResult.open

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CATEGORY_MISSING",
    "COMPAT_PRESETS",
    "CompatProfile",
    "DetrendedBaseline",
    "DetectResult",
    "EventCatalog",
    "ExternalClimatology",
    "FixedBaseline",
    "FixedThreshold",
    "MovingBaseline",
    "compute_climatology",
    "detect",
    "open",
    "resolve_compat",
]
