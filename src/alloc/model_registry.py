"""Canonical model registry — single source of truth for known model param counts.

Used by:
  - model_extractor.py (AST fallback: HuggingFace model IDs → raw param count)
  - cli.py (alloc scan: short model names → param count in billions)

Add new models here. Do NOT maintain separate lookup tables elsewhere.
"""

from __future__ import annotations

from typing import Dict, Optional

# HuggingFace model IDs → raw parameter count (int).
# Used by AST extractor to match from_pretrained("model_id") calls.
KNOWN_MODELS: Dict[str, int] = {
    # GPT-2
    "gpt2": int(124e6),
    "gpt2-medium": int(355e6),
    "gpt2-large": int(774e6),
    "gpt2-xl": int(1.5e9),
    # BERT
    "bert-base-uncased": int(110e6),
    "bert-large-uncased": int(340e6),
    # Llama
    "meta-llama/Llama-2-7b-hf": int(7e9),
    "meta-llama/Llama-2-13b-hf": int(13e9),
    "meta-llama/Llama-2-70b-hf": int(70e9),
    # Mistral / Mixtral
    "mistralai/Mistral-7B-v0.1": int(7.24e9),
    "mistralai/Mixtral-8x7B-v0.1": int(46.7e9),
    # Falcon
    "tiiuae/falcon-7b": int(7e9),
    "tiiuae/falcon-40b": int(40e9),
    # Bloom
    "bigscience/bloom-560m": int(560e6),
    "bigscience/bloom-1b7": int(1.7e9),
    "bigscience/bloom-7b1": int(7.1e9),
    # GPT-Neo / GPT-J
    "EleutherAI/gpt-neo-125M": int(125e6),
    "EleutherAI/gpt-neo-1.3B": int(1.3e9),
    "EleutherAI/gpt-neo-2.7B": int(2.7e9),
    "EleutherAI/gpt-j-6B": int(6e9),
    # Phi
    "microsoft/phi-2": int(2.78e9),
    # Gemma
    "google/gemma-2b": int(2.51e9),
    "google/gemma-7b": int(8.54e9),
    # Qwen
    "Qwen/Qwen-7B": int(7.72e9),
    "Qwen/Qwen-14B": int(14.2e9),
    "Qwen/Qwen-72B": int(72.7e9),
    # DeepSeek
    "deepseek-ai/deepseek-llm-7b-base": int(6.9e9),
    "deepseek-ai/deepseek-llm-67b-base": int(67e9),
    # T5
    "t5-small": int(60e6),
    "t5-base": int(220e6),
    "t5-large": int(770e6),
    "google/t5-xl-lm-adapt": int(3e9),
    "google/t5-xxl-lm-adapt": int(11e9),
    # Vision — ViT
    "google/vit-base-patch16-224": int(86e6),
    "google/vit-large-patch16-224": int(307e6),
    # Vision — ResNet / ConvNets
    "microsoft/resnet-50": int(25.6e6),
    "microsoft/resnet-101": int(44.5e6),
    "microsoft/resnet-152": int(60.2e6),
    # Vision — CLIP
    "openai/clip-vit-base-patch32": int(151e6),
    "openai/clip-vit-large-patch14": int(428e6),
    # Vision — DINOv2
    "facebook/dinov2-base": int(86e6),
    "facebook/dinov2-large": int(307e6),
    # Diffusion — Stable Diffusion (UNet component)
    "stabilityai/stable-diffusion-2-1": int(865e6),
    # Whisper
    "openai/whisper-small": int(244e6),
    "openai/whisper-medium": int(769e6),
    "openai/whisper-large-v3": int(1.55e9),
}

# Short CLI names → param count in billions.
# Used by `alloc scan --model <name>` for quick lookups without a script.
# Maps normalized lowercase names to billions (float).
_CLI_MODEL_PARAMS: Dict[str, float] = {
    # Llama
    "llama-3-70b": 70.0,
    "llama-3-8b": 8.03,
    "llama-2-70b": 70.0,
    "llama-2-13b": 13.0,
    "llama-2-7b": 7.0,
    # Mistral / Mixtral
    "mistral-7b": 7.24,
    "mixtral-8x7b": 46.7,
    # GPT-2
    "gpt2": 0.124,
    "gpt2-medium": 0.355,
    "gpt2-large": 0.774,
    "gpt2-xl": 1.5,
    # BERT
    "bert-base": 0.110,
    "bert-large": 0.340,
    # T5
    "t5-small": 0.060,
    "t5-base": 0.220,
    "t5-large": 0.770,
    "t5-xl": 3.0,
    "t5-xxl": 11.0,
    # Falcon
    "falcon-7b": 7.0,
    "falcon-40b": 40.0,
    # Phi
    "phi-2": 2.78,
    # Gemma
    "gemma-2b": 2.51,
    "gemma-7b": 8.54,
    # Qwen
    "qwen-7b": 7.72,
    "qwen-14b": 14.2,
    "qwen-72b": 72.7,
    # DeepSeek
    "deepseek-7b": 6.9,
    "deepseek-67b": 67.0,
    # Vision — ViT
    "vit-base": 0.086,
    "vit-large": 0.307,
    # Vision — ResNet / ConvNets
    "resnet-50": 0.026,
    "resnet-101": 0.045,
    "resnet-152": 0.060,
    # Vision — CLIP
    "clip-vit-base": 0.151,
    "clip-vit-large": 0.428,
    # Vision — DINOv2
    "dinov2-base": 0.086,
    "dinov2-large": 0.307,
    # Diffusion
    "stable-diffusion-2": 0.865,
    # Whisper
    "whisper-small": 0.244,
    "whisper-medium": 0.769,
    "whisper-large": 1.55,
    # Bloom
    "bloom-560m": 0.560,
    "bloom-1b7": 1.7,
    "bloom-7b1": 7.1,
    # GPT-Neo / GPT-J
    "gpt-neo-125m": 0.125,
    "gpt-neo-1.3b": 1.3,
    "gpt-neo-2.7b": 2.7,
    "gpt-j-6b": 6.0,
}


def lookup_model_params(model: str) -> Optional[float]:
    """Look up model param count (in billions) by short CLI name.

    Returns None if model not found.
    """
    normalized = model.lower().strip()
    return _CLI_MODEL_PARAMS.get(normalized)
