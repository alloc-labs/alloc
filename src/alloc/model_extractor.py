"""Model extraction — get param count from user scripts.

Three extraction methods with fallback chain:
  1. Manual override (--param-count-b flag) → instant, no file needed
  2. Subprocess execution → import module, find nn.Module, count params
  3. AST parsing → find from_pretrained() calls, match against known models

Never crashes. Returns None when extraction fails.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    """Result of model extraction."""

    param_count: int            # raw parameter count
    dtype: str                  # "float32", "float16", etc.
    model_name: Optional[str]   # class name if found
    method: str                 # "execution" | "ast" | "manual"
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None
    seq_length: Optional[int] = None
    activation_memory_bytes: Optional[int] = None
    activation_method: Optional[str] = None  # "traced" | None


def extract_model_info(
    script: str,
    *,
    timeout: int = 60,
    param_count_b: Optional[float] = None,
) -> Optional[ModelInfo]:
    """Extract model info from a script with fallback chain.

    Returns ModelInfo or None if extraction fails.
    """
    # 1. Manual override — no file needed
    if param_count_b is not None:
        return ModelInfo(
            param_count=int(param_count_b * 1e9),
            dtype="float16",
            model_name=None,
            method="manual",
        )

    # 2. File must exist for remaining methods
    if not os.path.isfile(script):
        return None

    # 3. Try subprocess execution
    result = _extract_via_subprocess(script, timeout=timeout)
    if result is not None:
        return result

    # 4. Try AST parsing
    result = _extract_via_ast(script)
    if result is not None:
        return result

    return None


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------

def _extract_via_subprocess(
    script: str,
    *,
    timeout: int = 60,
) -> Optional[ModelInfo]:
    """Execute user script in isolated subprocess, extract model info.

    Invokes alloc.extractor_runner as a module in a subprocess with
    CUDA_VISIBLE_DEVICES="" to prevent GPU allocation. Results are
    communicated via a JSON sidecar file.
    """
    sidecar_fd = None
    sidecar_path = None

    try:
        # Create sidecar file for IPC
        sidecar_fd, sidecar_path = tempfile.mkstemp(suffix=".json", prefix="alloc_extract_")
        os.close(sidecar_fd)
        sidecar_fd = None

        script_abs = os.path.abspath(script)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""  # prevent GPU allocation

        subprocess.run(
            [sys.executable, "-m", "alloc.extractor_runner", sidecar_path, script_abs],
            timeout=timeout,
            capture_output=True,
            env=env,
        )

        # Read sidecar result
        try:
            with open(sidecar_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            return None

        if data.get("status") == "ok":
            return ModelInfo(
                param_count=data["param_count"],
                dtype=data.get("dtype", "float32"),
                model_name=data.get("model_name"),
                method="execution",
                hidden_dim=data.get("hidden_dim"),
                num_layers=data.get("num_layers"),
                seq_length=data.get("seq_length"),
                activation_memory_bytes=data.get("activation_memory_bytes"),
                activation_method=data.get("activation_method"),
            )

        return None

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
    finally:
        # Clean up sidecar temp file
        if sidecar_path and os.path.exists(sidecar_path):
            try:
                os.unlink(sidecar_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# AST-based extraction (static fallback)
# ---------------------------------------------------------------------------

from alloc.model_registry import KNOWN_MODELS


def _extract_via_ast(script: str) -> Optional[ModelInfo]:
    """Parse script AST, find from_pretrained() calls, match against known models."""
    try:
        with open(script, "r") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return None

    # Walk AST looking for .from_pretrained("model_name") calls
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Match *.from_pretrained(...)
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "from_pretrained":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                model_id = node.args[0].value
                param_count = KNOWN_MODELS.get(model_id)
                if param_count is not None:
                    return ModelInfo(
                        param_count=param_count,
                        dtype="float16",
                        model_name=model_id,
                        method="ast",
                    )

    return None
