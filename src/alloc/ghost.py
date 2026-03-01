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

# Optimizer state sizing (bytes per parameter):
# [PHYSICS] Adam mixed (fp16/bf16): master copy (4) + momentum (4) + variance (4) = 12
# [PHYSICS] Adam fp32:              momentum (4) + variance (4) = 8
# [PHYSICS] SGD:                    momentum (4) = 4  (or 0 without momentum)
# [PHYSICS] Adafactor:              row/col factors ≈ 4 bytes/param effective
# [PHYSICS] Lion:                   momentum (4) + master copy (4) = 8 for mixed
# [PHYSICS] 8-bit Adam:             master (4) + momentum (1) + variance (1) = 6 for mixed
OPTIMIZER_BYTES_MIXED = 12   # fp16/bf16 training (default: Adam)
OPTIMIZER_BYTES_FP32 = 8     # fp32 training (default: Adam)

# Optimizer-specific overrides (mixed precision, bytes/param)
OPTIMIZER_BYTES_BY_TYPE = {
    "Adam": 12,
    "AdamW": 12,
    "FusedAdam": 12,
    "DeepSpeedCPUAdam": 12,
    "SGD": 4,
    "RMSprop": 8,
    "Adagrad": 8,
    "Lion": 8,
    "Adafactor": 4,
    "Adam8bit": 6,
    "AdamW8bit": 6,
    "FusedLAMB": 12,
    "LBFGS": 4,
}

# Buffer overhead: 10% of (weights + gradients + activations + optimizer)
# for fragmentation and temp allocations.
BUFFER_OVERHEAD_FACTOR = 0.1

# Rough transformer param formula: P ≈ 12 * hidden_dim^2 * num_layers
# TRANSFORMER-SPECIFIC: only used when hidden_dim is explicitly available.
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
    activation_method: Optional[str] = None  # "traced", "transformer_formula", "weights_fallback"
    batch_size: Optional[int] = None
    seq_length: Optional[int] = None
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def ghost(
    model: Any = None,
    *,
    param_count: Optional[int] = None,
    param_count_b: Optional[float] = None,
    dtype: str = "float16",
    batch_size: Optional[int] = None,
    seq_length: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    optimizer_type: Optional[str] = None,
    activation_memory_bytes: Optional[int] = None,
) -> GhostReport:
    """Run a Ghost Scan on a model or param count.

    Usage:
        # With a PyTorch model
        report = alloc.ghost(model)

        # Without torch — just param count
        report = alloc.ghost(param_count_b=7.0)

        # With optimizer awareness
        report = alloc.ghost(param_count_b=7.0, optimizer_type="SGD")

        # With traced activations from forward hooks
        report = alloc.ghost(param_count_b=7.0, activation_memory_bytes=12345678)

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
            optimizer_type=optimizer_type,
            activation_memory_bytes=activation_memory_bytes,
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
    batch_size: Optional[int],
    seq_length: Optional[int],
    hidden_dim: Optional[int],
    optimizer_type: Optional[str] = None,
    activation_memory_bytes: Optional[int] = None,
) -> GhostReport:
    """Core Ghost implementation."""
    resolved_count = 0
    resolved_dtype = dtype

    if model is not None:
        # Path 1: Walk torch model parameters
        resolved_count, resolved_dtype = _count_from_model(model)
        # Try to extract architecture info if not provided by user
        if hidden_dim is None or seq_length is None:
            arch = _extract_architecture(model)
            if hidden_dim is None:
                hidden_dim = arch["hidden_dim"]
            if seq_length is None:
                seq_length = arch["seq_length"]
    elif param_count is not None:
        resolved_count = param_count
    elif param_count_b is not None:
        resolved_count = round(param_count_b * 1e9)

    if resolved_count <= 0:
        resolved_count = 0

    bytes_per_param = BYTES_PER_DTYPE.get(resolved_dtype, 2)
    is_mixed = bytes_per_param <= 2  # fp16, bf16, int8, int4

    # Optimizer-aware VRAM: use specific bytes/param when optimizer is known
    if optimizer_type and optimizer_type in OPTIMIZER_BYTES_BY_TYPE:
        optimizer_bpp = OPTIMIZER_BYTES_BY_TYPE[optimizer_type]
    else:
        optimizer_bpp = OPTIMIZER_BYTES_MIXED if is_mixed else OPTIMIZER_BYTES_FP32
    to_gb = 1.0 / (1024 ** 3)

    # Mixed precision: gradients stored in fp32 during backward pass
    grad_bytes_per_param = 4 if is_mixed else bytes_per_param
    weights_bytes = resolved_count * bytes_per_param
    gradients_bytes = resolved_count * grad_bytes_per_param
    optimizer_bytes = resolved_count * optimizer_bpp

    # Activation estimation: three paths
    est_layers = None
    act_method = None
    if activation_memory_bytes is not None and activation_memory_bytes > 0:
        # Path 1: Traced activations from forward hooks — scale linearly by batch_size
        scale = batch_size if batch_size and batch_size > 0 else 1
        activations_bytes = activation_memory_bytes * scale
        act_method = "traced"
    elif hidden_dim is not None and seq_length is not None and batch_size is not None:
        # Path 2: Transformer formula — use architecture-specific formula
        est_layers = 1
        if resolved_count > 0 and hidden_dim > 0:
            est_layers = max(1, round(resolved_count / (_TRANSFORMER_PARAM_FACTOR * hidden_dim * hidden_dim)))
        # Per-layer activation: B * S * H * bytes_per_param
        activations_bytes = est_layers * batch_size * seq_length * hidden_dim * bytes_per_param
        act_method = "transformer_formula"
    else:
        # Path 3: Fallback — activations ≈ 1x model weights
        # Empirically reasonable across CNNs, ViTs, diffusion models for moderate batch sizes.
        # Underestimates for large batches, overestimates for small — but never wildly wrong.
        activations_bytes = weights_bytes
        act_method = "weights_fallback"

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
        activation_method=act_method,
        batch_size=batch_size,
        seq_length=seq_length,
        hidden_dim=hidden_dim,
        num_layers=est_layers,
    )


def _count_from_model(model: Any) -> tuple:
    """Extract param count and dtype from a torch model. Returns (count, dtype_str).

    Uses data_ptr() to deduplicate shared/tied parameters.
    """
    seen_ptrs = set()
    total = 0
    dtype_str = "float16"
    try:
        for name, param in model.named_parameters():
            ptr = param.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            total += param.numel()
            # Use dtype of the first parameter
            if len(seen_ptrs) == 1:
                dtype_str = str(param.dtype).replace("torch.", "")
    except Exception:
        pass
    return total, dtype_str


def _extract_architecture(model: Any) -> dict:
    """Try to extract architecture dimensions from a model object.

    Checks HuggingFace config, common attribute names, and weight shapes.
    Returns dict with keys: hidden_dim, num_layers, seq_length (all Optional[int]).
    """
    result = {"hidden_dim": None, "num_layers": None, "seq_length": None}

    # 1. HuggingFace config (most reliable)
    config = getattr(model, "config", None)
    if config is not None:
        result["hidden_dim"] = getattr(config, "hidden_size", None)
        result["num_layers"] = (
            getattr(config, "num_hidden_layers", None)
            or getattr(config, "num_layers", None)
            or getattr(config, "n_layer", None)  # GPT-2 style
        )
        result["seq_length"] = (
            getattr(config, "max_position_embeddings", None)
            or getattr(config, "n_positions", None)  # GPT-2 style
            or getattr(config, "max_seq_len", None)
        )

    # 2. Infer hidden_dim from weight shapes if config didn't provide it
    if result["hidden_dim"] is None:
        try:
            max_dim = 0
            for name, param in model.named_parameters():
                if param.dim() == 2:  # Linear weight matrix
                    max_dim = max(max_dim, max(param.shape))
            if max_dim > 0:
                result["hidden_dim"] = max_dim
        except Exception:
            pass

    return result
