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

# The extractor script that runs inside the subprocess.
# It imports the user's module, finds nn.Module classes/instances,
# and writes results to a sidecar JSON file.
_EXTRACTOR_SCRIPT = r'''
import importlib.util
import json
import os
import sys

def _patch_sys_exit():
    """Prevent sys.exit() from killing the extractor."""
    class _ExitCatch(SystemExit):
        pass

    _real_exit = sys.exit
    def _fake_exit(code=0):
        raise _ExitCatch(code)

    sys.exit = _fake_exit
    return _ExitCatch

def _count_params(model):
    """Count parameters in an nn.Module."""
    total = 0
    dtype_str = "float32"
    try:
        for i, (name, param) in enumerate(model.named_parameters()):
            total += param.numel()
            if i == 0:
                dtype_str = str(param.dtype).replace("torch.", "")
    except Exception:
        pass
    return total, dtype_str

def main():
    sidecar_path = sys.argv[1]
    script_path = sys.argv[2]

    result = {"status": "no_model"}

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        result = {"status": "error", "error": "torch not installed"}
        with open(sidecar_path, "w") as f:
            json.dump(result, f)
        return

    ExitCatch = _patch_sys_exit()

    # Add script's directory to sys.path for relative imports
    script_dir = os.path.dirname(os.path.abspath(script_path))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Import the user module
    try:
        spec = importlib.util.spec_from_file_location("__user_module__", script_path)
        if spec is None or spec.loader is None:
            result = {"status": "error", "error": "cannot load module"}
            with open(sidecar_path, "w") as f:
                json.dump(result, f)
            return

        module = importlib.util.module_from_spec(spec)
        module.__name__ = "__user_module__"  # skip if __name__ == "__main__" guards
        try:
            spec.loader.exec_module(module)
        except ExitCatch:
            pass  # script called sys.exit(), that's fine
        except SystemExit:
            pass  # catch real SystemExit too
    except Exception as e:
        result = {"status": "error", "error": str(e)[:200]}
        with open(sidecar_path, "w") as f:
            json.dump(result, f)
        return

    # Search for nn.Module instances in module globals
    models = []
    for attr_name in dir(module):
        try:
            obj = getattr(module, attr_name)
            if isinstance(obj, nn.Module):
                count, dtype_str = _count_params(obj)
                if count > 0:
                    models.append((count, dtype_str, attr_name))
        except Exception:
            continue

    # If no instances found, search for nn.Module subclasses and try instantiation
    if not models:
        for attr_name in dir(module):
            try:
                obj = getattr(module, attr_name)
                if (isinstance(obj, type)
                    and issubclass(obj, nn.Module)
                    and obj is not nn.Module):
                    try:
                        instance = obj()
                        count, dtype_str = _count_params(instance)
                        if count > 0:
                            models.append((count, dtype_str, attr_name))
                    except Exception:
                        continue
            except Exception:
                continue

    if models:
        # Pick largest model (most params = likely training target)
        models.sort(key=lambda m: m[0], reverse=True)
        best = models[0]
        result = {
            "status": "ok",
            "param_count": best[0],
            "dtype": best[1],
            "model_name": best[2],
        }
    else:
        result = {"status": "no_model"}

    with open(sidecar_path, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
'''


def _extract_via_subprocess(
    script: str,
    *,
    timeout: int = 60,
) -> Optional[ModelInfo]:
    """Execute user script in isolated subprocess, extract model info."""
    sidecar_fd = None
    sidecar_path = None
    extractor_fd = None
    extractor_path = None

    try:
        # Create sidecar file for IPC
        sidecar_fd, sidecar_path = tempfile.mkstemp(suffix=".json", prefix="alloc_extract_")
        os.close(sidecar_fd)
        sidecar_fd = None

        # Write extractor script to temp file
        extractor_fd, extractor_path = tempfile.mkstemp(suffix=".py", prefix="alloc_extractor_")
        os.write(extractor_fd, _EXTRACTOR_SCRIPT.encode("utf-8"))
        os.close(extractor_fd)
        extractor_fd = None

        script_abs = os.path.abspath(script)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""  # prevent GPU allocation

        proc = subprocess.run(
            [sys.executable, extractor_path, sidecar_path, script_abs],
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
            )

        return None

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
    finally:
        # Clean up temp files
        if sidecar_path and os.path.exists(sidecar_path):
            try:
                os.unlink(sidecar_path)
            except OSError:
                pass
        if extractor_path and os.path.exists(extractor_path):
            try:
                os.unlink(extractor_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# AST-based extraction (static fallback)
# ---------------------------------------------------------------------------

# Known HuggingFace model IDs → approximate param count
_KNOWN_MODELS = {
    "gpt2": int(124e6),
    "gpt2-medium": int(355e6),
    "gpt2-large": int(774e6),
    "gpt2-xl": int(1.5e9),
    "bert-base-uncased": int(110e6),
    "bert-large-uncased": int(340e6),
    "meta-llama/Llama-2-7b-hf": int(7e9),
    "meta-llama/Llama-2-13b-hf": int(13e9),
    "meta-llama/Llama-2-70b-hf": int(70e9),
    "mistralai/Mistral-7B-v0.1": int(7.24e9),
    "tiiuae/falcon-7b": int(7e9),
    "tiiuae/falcon-40b": int(40e9),
    "bigscience/bloom-560m": int(560e6),
    "bigscience/bloom-1b7": int(1.7e9),
    "bigscience/bloom-7b1": int(7.1e9),
    "EleutherAI/gpt-neo-125M": int(125e6),
    "EleutherAI/gpt-neo-1.3B": int(1.3e9),
    "EleutherAI/gpt-neo-2.7B": int(2.7e9),
    "EleutherAI/gpt-j-6B": int(6e9),
    "microsoft/phi-2": int(2.78e9),
    "google/gemma-2b": int(2.51e9),
    "google/gemma-7b": int(8.54e9),
}


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
                param_count = _KNOWN_MODELS.get(model_id)
                if param_count is not None:
                    return ModelInfo(
                        param_count=param_count,
                        dtype="float16",
                        model_name=model_id,
                        method="ast",
                    )

    return None
