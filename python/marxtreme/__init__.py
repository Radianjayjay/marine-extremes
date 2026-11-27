"""marxtreme: Rust-accelerated detection of marine heatwaves, cold spells, and marine extremes."""

from marxtreme.api import CATEGORY_MISSING, compute_climatology, detect
from marxtreme.baselines import (
    DetrendedBaseline,
    ExternalClimatology,
    FixedBaseline,
    FixedThreshold,
    MovingBaseline,
)
from marxtreme.catalog import EventCatalog
from marxtreme.compat import PRESETS as COMPAT_PRESETS
from marxtreme.compat import CompatProfile, resolve_compat
from marxtreme.result import DetectResult

# Top-level convenience alias for reopening any self-contained marxtreme output.
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
