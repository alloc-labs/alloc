"""Alloc — GPU intelligence for ML training."""

from __future__ import annotations

import warnings as _warnings
_warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")
_warnings.filterwarnings("ignore", category=DeprecationWarning, module="pynvml")
_warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.cuda")
_warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.cuda")
del _warnings

__version__ = "0.0.13"

from alloc.ghost import ghost, GhostReport
from alloc.callbacks import AllocCallback as HuggingFaceCallback
from alloc.callbacks import AllocLightningCallback as LightningCallback

__all__ = ["ghost", "GhostReport", "HuggingFaceCallback", "LightningCallback", "__version__"]
