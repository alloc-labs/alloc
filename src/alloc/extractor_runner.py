"""Standalone subprocess runner for model parameter extraction.

This module is invoked by model_extractor.py as:
    subprocess.run([python, "-m", "alloc.extractor_runner", sidecar_path, script_path])

It imports the user's training script in an isolated process, finds
nn.Module instances/classes, counts parameters, and writes results
to a JSON sidecar file. Runs with CUDA_VISIBLE_DEVICES="" to prevent
GPU allocation.

Never imported directly by the rest of the CLI — only executed as __main__.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys


def _patch_sys_exit():
    """Prevent sys.exit() from killing the extractor."""
    class _ExitCatch(SystemExit):
        pass

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
