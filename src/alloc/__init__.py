"""Alloc — GPU intelligence for ML training."""

from __future__ import annotations

__version__ = "0.3.0"

from alloc.ghost import ghost, GhostReport
from alloc.callbacks import AllocCallback as HuggingFaceCallback

__all__ = ["ghost", "GhostReport", "HuggingFaceCallback", "__version__"]
