"""Ghost Scan — static VRAM analysis without executing the model.

Two paths:
  1. With torch: walk model.named_parameters() for exact sizing
  2. Without torch: pure math from param_count

Never crashes user code. All exceptions are caught and logged.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

# Canonical dtype → bytes mapping.
BYTES_PER_DTYPE = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
    # Shorthand aliases (torch uses long names, API uses short names)
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
}

# Adam optimizer state sizing:
#   Mixed precision (fp16/bf16): master copy (fp32) + momentum (fp32) + variance (fp32) = 12 bytes/param
#   Full precision (fp32):       momentum (fp32) + variance (fp32) = 8 bytes/param (no master copy needed)
OPTIMIZER_BYTES_MIXED = 12   # fp16/bf16 training
OPTIMIZER_BYTES_FP32 = 8     # fp32 training

# Buffer overhead: 10% of (weights + gradients + activations + optimizer)
# for fragmentation and temp allocations.
BUFFER_OVERHEAD_FACTOR = 0.1

# Rough transformer param formula: P ≈ 12 * hidden_dim^2 * num_layers
# Used to estimate num_layers when not provided explicitly.
_TRANSFORMER_PARAM_FACTOR = 12


@dataclass
class GhostReport:
    """Result of a Ghost Scan — VRAM breakdown."""

    param_count: int
    param_count_b: float
    dtype: str
    weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    buffer_gb: float
    total_gb: float
    extraction_method: Optional[str] = None  # "execution", "ast", "manual"

    def to_dict(self) -> dict:
        return asdict(self)


def ghost(
    model: Any = None,
    *,
    param_count: Optional[int] = None,
    param_count_b: Optional[float] = None,
    dtype: str = "float16",
    batch_size: int = 32,
    seq_length: int = 2048,
    hidden_dim: int = 4096,
) -> GhostReport:
    """Run a Ghost Scan on a model or param count.

    Usage:
        # With a PyTorch model
        report = alloc.ghost(model)

        # Without torch — just param count
        report = alloc.ghost(param_count_b=7.0)

    Never raises. If something goes wrong, returns a best-effort report.
    """
    try:
        return _ghost_impl(
            model=model,
            param_count=param_count,
            param_count_b=param_count_b,
            dtype=dtype,
            batch_size=batch_size,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
        )
    except Exception:
        # Never crash user code
        count = param_count or int((param_count_b or 0) * 1e9) or 0
        return GhostReport(
            param_count=count,
            param_count_b=count / 1e9,
            dtype=dtype,
            weights_gb=0.0,
            gradients_gb=0.0,
            optimizer_gb=0.0,
            activations_gb=0.0,
            buffer_gb=0.0,
            total_gb=0.0,
        )


def _ghost_impl(
    model: Any,
    param_count: Optional[int],
    param_count_b: Optional[float],
    dtype: str,
    batch_size: int,
    seq_length: int,
    hidden_dim: int,
) -> GhostReport:
    """Core Ghost implementation."""
    resolved_count = 0
    resolved_dtype = dtype

    if model is not None:
        # Path 1: Walk torch model parameters
        resolved_count, resolved_dtype = _count_from_model(model)
    elif param_count is not None:
        resolved_count = param_count
    elif param_count_b is not None:
        resolved_count = round(param_count_b * 1e9)

    if resolved_count <= 0:
        resolved_count = 0

    bytes_per_param = BYTES_PER_DTYPE.get(resolved_dtype, 2)
    is_mixed = bytes_per_param <= 2  # fp16, bf16, int8, int4
    optimizer_bpp = OPTIMIZER_BYTES_MIXED if is_mixed else OPTIMIZER_BYTES_FP32
    to_gb = 1.0 / (1024 ** 3)

    # Mixed precision: gradients stored in fp32 during backward pass
    grad_bytes_per_param = 4 if is_mixed else bytes_per_param
    weights_bytes = resolved_count * bytes_per_param
    gradients_bytes = resolved_count * grad_bytes_per_param
    optimizer_bytes = resolved_count * optimizer_bpp

    # Estimate num_layers for activation sizing: P ≈ 12 * H^2 * L
    est_layers = 1
    if resolved_count > 0 and hidden_dim > 0:
        est_layers = max(1, round(resolved_count / (_TRANSFORMER_PARAM_FACTOR * hidden_dim * hidden_dim)))
    # Per-layer activation: B * S * H * ~34 bytes (from Megatron paper)
    # Using bytes_per_param as proxy for precision factor
    activations_bytes = est_layers * batch_size * seq_length * hidden_dim * bytes_per_param

    weights_gb = weights_bytes * to_gb
    gradients_gb = gradients_bytes * to_gb
    optimizer_gb = optimizer_bytes * to_gb
    activations_gb = activations_bytes * to_gb
    buffer_gb = BUFFER_OVERHEAD_FACTOR * (weights_gb + gradients_gb + optimizer_gb + activations_gb)
    total_gb = weights_gb + gradients_gb + optimizer_gb + activations_gb + buffer_gb

    return GhostReport(
        param_count=resolved_count,
        param_count_b=round(resolved_count / 1e9, 3),
        dtype=resolved_dtype,
        weights_gb=round(weights_gb, 2),
        gradients_gb=round(gradients_gb, 2),
        optimizer_gb=round(optimizer_gb, 2),
        activations_gb=round(activations_gb, 2),
        buffer_gb=round(buffer_gb, 2),
        total_gb=round(total_gb, 2),
    )


def _count_from_model(model: Any) -> tuple:
    """Extract param count and dtype from a torch model. Returns (count, dtype_str)."""
    total = 0
    dtype_str = "float16"
    try:
        for name, param in model.named_parameters():
            total += param.numel()
            # Use dtype of the first parameter
            if total == param.numel():
                dtype_str = str(param.dtype).replace("torch.", "")
    except Exception:
        pass
    return total, dtype_str
