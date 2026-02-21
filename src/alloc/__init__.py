"""Alloc — GPU intelligence for ML training."""

from __future__ import annotations

__version__ = "0.0.1"

from alloc.ghost import ghost, GhostReport
from alloc.callbacks import AllocCallback as HuggingFaceCallback
from alloc.callbacks import AllocLightningCallback as LightningCallback

__all__ = ["ghost", "GhostReport", "HuggingFaceCallback", "LightningCallback", "__version__"]
