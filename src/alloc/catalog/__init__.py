"""GPU catalog — offline hardware specs and pricing for CLI.

Source of truth: apps/api/src/engine/catalog/gpus.v1.json
This is a bundled copy for offline CLI use. Update when the API catalog changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

_CATALOG_DIR = Path(__file__).parent

# Aliases for common shorthand names
_ALIASES = {
    "H100": "nvidia-h100-sxm-80gb",
    "H100-80GB": "nvidia-h100-sxm-80gb",
    "A100": "nvidia-a100-sxm-80gb",
    "A100-80GB": "nvidia-a100-sxm-80gb",
    "A100-40GB": "nvidia-a100-40gb",
    "A10G": "nvidia-a10g-24gb",
    "L40S": "nvidia-l40s-48gb",
    "L4": "nvidia-l4-24gb",
    "T4": "nvidia-t4-16gb",
    "V100": "nvidia-v100-32gb",
    "V100-32GB": "nvidia-v100-32gb",
    "V100-16GB": "nvidia-v100-16gb",
    "RTX-4090": "nvidia-rtx4090-24gb",
    "RTX-3090": "nvidia-rtx3090-24gb",
    "H200": "nvidia-h200-141gb",
    "H100-NVL": "nvidia-h100-nvl-94gb",
}


def _load_catalog() -> dict:
    """Load GPU catalog from bundled JSON."""
    with open(_CATALOG_DIR / "gpus.v1.json") as f:
        return json.load(f)


def _load_rate_card() -> dict:
    """Load default rate card from bundled JSON."""
    with open(_CATALOG_DIR / "default_rate_card.json") as f:
        return json.load(f)


def list_gpus() -> List[dict]:
    """Return all GPUs sorted by VRAM descending.

    Each entry has: id, display_name, vendor, vram_gb, architecture,
    bandwidth_gbps, bf16_tflops, tdp_watts, pricing.
    """
    catalog = _load_catalog()
    rate_card = _load_rate_card()

    result = []
    for gpu_id, spec in catalog.get("gpus", {}).items():
        pricing = rate_card.get("rates", {}).get(spec["display_name"], {})
        result.append({
            "id": gpu_id,
            "display_name": spec["display_name"],
            "vendor": spec.get("vendor", "nvidia"),
            "vram_gb": spec["vram_gb"],
            "architecture": spec.get("architecture", ""),
            "bandwidth_gbps": spec.get("bandwidth_gbps", 0),
            "bf16_tflops": spec.get("bf16_tflops", 0),
            "fp16_tflops": spec.get("fp16_tflops", 0),
            "fp32_tflops": spec.get("fp32_tflops", 0),
            "tf32_tflops": spec.get("tf32_tflops", 0),
            "tdp_watts": spec.get("tdp_watts", 0),
            "interconnect": spec.get("interconnect"),
            "pricing": pricing,
        })

    return sorted(result, key=lambda x: x["vram_gb"], reverse=True)


def get_default_rate(gpu_name: str) -> Optional[float]:
    """Look up the average default $/hr for a GPU by name or alias.

    Tries to match the probe-reported GPU name against catalog display names.
    Returns the average across clouds, or None if not found.
    """
    rate_card = _load_rate_card()
    rates = rate_card.get("rates", {})

    # Direct match by display name
    for display_name, cloud_rates in rates.items():
        if display_name.lower() in gpu_name.lower() or gpu_name.lower() in display_name.lower():
            vals = [v for v in cloud_rates.values() if isinstance(v, (int, float))]
            return sum(vals) / len(vals) if vals else None

    # Try aliases → display name
    for alias, stable_id in _ALIASES.items():
        if alias.lower() in gpu_name.lower():
            catalog = _load_catalog()
            spec = catalog.get("gpus", {}).get(stable_id)
            if spec:
                dn = spec.get("display_name", "")
                cloud_rates = rates.get(dn, {})
                vals = [v for v in cloud_rates.values() if isinstance(v, (int, float))]
                return sum(vals) / len(vals) if vals else None

    return None


def get_gpu(gpu_id: str) -> Optional[dict]:
    """Look up a GPU by stable ID or alias.

    Returns full spec dict or None if not found.
    """
    # Resolve aliases
    resolved = _ALIASES.get(gpu_id, gpu_id)

    catalog = _load_catalog()
    rate_card = _load_rate_card()

    spec = catalog.get("gpus", {}).get(resolved)
    if not spec:
        return None

    pricing = rate_card.get("rates", {}).get(spec["display_name"], {})
    return {
        "id": resolved,
        "display_name": spec["display_name"],
        "vendor": spec.get("vendor", "nvidia"),
        "vram_gb": spec["vram_gb"],
        "architecture": spec.get("architecture", ""),
        "bandwidth_gbps": spec.get("bandwidth_gbps", 0),
        "bf16_tflops": spec.get("bf16_tflops", 0),
        "fp16_tflops": spec.get("fp16_tflops", 0),
        "fp32_tflops": spec.get("fp32_tflops", 0),
        "tf32_tflops": spec.get("tf32_tflops", 0),
        "tdp_watts": spec.get("tdp_watts", 0),
        "interconnect": spec.get("interconnect"),
        "pricing": pricing,
    }
