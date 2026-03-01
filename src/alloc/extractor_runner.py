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


def _extract_architecture(model):
    """Extract architecture dimensions from model object."""
    info = {}
    config = getattr(model, "config", None)
    if config is not None:
        info["hidden_dim"] = getattr(config, "hidden_size", None)
        info["num_layers"] = (
            getattr(config, "num_hidden_layers", None)
            or getattr(config, "num_layers", None)
            or getattr(config, "n_layer", None)
        )
        info["seq_length"] = (
            getattr(config, "max_position_embeddings", None)
            or getattr(config, "n_positions", None)
            or getattr(config, "max_seq_len", None)
        )
        info["model_type"] = getattr(config, "model_type", None)
        info["vocab_size"] = getattr(config, "vocab_size", None)
        info["image_size"] = getattr(config, "image_size", None)
        info["num_channels"] = getattr(config, "num_channels", None)
    return info


def _build_dummy_input(model, arch_info):
    """Build a dummy input tensor for a forward pass. Returns Tensor or None.

    Three inference paths:
      1. HF transformer config: vocab_size → token IDs
      2. Vision config: image_size → image tensor
      3. Weight shape inference: first param shape → infer input
    """
    import torch

    # Path 1: HF transformer with vocab_size → token IDs
    vocab_size = arch_info.get("vocab_size")
    if vocab_size is not None and isinstance(vocab_size, int) and vocab_size > 0:
        seq_len = arch_info.get("seq_length") or 128
        seq_len = min(seq_len, 128)
        return torch.zeros(1, seq_len, dtype=torch.long)

    # Path 2: Vision config with image_size
    image_size = arch_info.get("image_size")
    if image_size is not None and isinstance(image_size, int) and image_size > 0:
        num_channels = arch_info.get("num_channels") or 3
        return torch.zeros(1, num_channels, image_size, image_size)

    # Path 3: Infer from first parameter shape
    try:
        for name, param in model.named_parameters():
            if param.dim() == 4:
                # Conv2d weight: (out_ch, in_ch, kH, kW) → image input
                in_ch = param.shape[1]
                return torch.zeros(1, in_ch, 32, 32)
            if param.dim() == 2:
                # Linear weight: (out_features, in_features) → vector input
                in_features = param.shape[1]
                return torch.zeros(1, in_features)
    except Exception:
        pass

    return None


def _measure_activations(model, arch_info):
    """Measure activation memory via forward hooks on CPU. batch_size=1.

    Returns {"activation_memory_bytes": int, "activation_method": "traced"} or {} on failure.
    """
    import torch

    hooks = []
    activation_sizes = []

    def _hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activation_sizes.append(output.numel() * output.element_size())
        elif isinstance(output, (tuple, list)):
            for t in output:
                if isinstance(t, torch.Tensor):
                    activation_sizes.append(t.numel() * t.element_size())

    try:
        # Register hooks on every leaf module
        for module in model.modules():
            children = list(module.children())
            if not children:  # leaf module
                hooks.append(module.register_forward_hook(_hook_fn))

        dummy_input = _build_dummy_input(model, arch_info)
        if dummy_input is None:
            return {}

        with torch.no_grad():
            model.eval()
            model(dummy_input)

        if not activation_sizes:
            return {}

        total_bytes = sum(activation_sizes)
        return {
            "activation_memory_bytes": total_bytes,
            "activation_method": "traced",
        }
    except Exception:
        return {}
    finally:
        for h in hooks:
            h.remove()


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

        # Try to extract architecture info from the best model
        best_model_obj = None
        for attr_name in dir(module):
            try:
                obj = getattr(module, attr_name)
                if isinstance(obj, nn.Module) and attr_name == best[2]:
                    best_model_obj = obj
                    break
            except Exception:
                continue

        arch_info = _extract_architecture(best_model_obj) if best_model_obj else {}

        # Measure activation memory via forward hooks
        activation_result = _measure_activations(best_model_obj, arch_info) if best_model_obj else {}

        result = {
            "status": "ok",
            "param_count": best[0],
            "dtype": best[1],
            "model_name": best[2],
            "hidden_dim": arch_info.get("hidden_dim"),
            "num_layers": arch_info.get("num_layers"),
            "seq_length": arch_info.get("seq_length"),
            "model_type": arch_info.get("model_type"),
            "activation_memory_bytes": activation_result.get("activation_memory_bytes"),
            "activation_method": activation_result.get("activation_method"),
        }
    else:
        result = {"status": "no_model"}

    with open(sidecar_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
