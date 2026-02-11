"""Tests for multi-signal stability detection."""

from __future__ import annotations

import time
from alloc.probe import ProbeSample
from alloc.stability import (
    check_stability,
    MIN_SAMPLES_FOR_STABILITY,
    MIN_PEAK_VRAM_MB,
    RAMP_UP_SAMPLES,
)


def _make_samples(count, vram_mb=5000.0, gpu_util=60.0, power=200.0, interval_s=0.5):
    """Generate a list of ProbeSamples with constant metrics."""
    base_t = time.time()
    return [
        ProbeSample(
            timestamp=base_t + i * interval_s,
            memory_used_mb=vram_mb,
            memory_total_mb=80000.0,
            gpu_util_pct=gpu_util,
            power_watts=power,
        )
        for i in range(count)
    ]


def test_insufficient_samples():
    """<30 samples → not stable."""
    samples = _make_samples(10)
    result = check_stability(samples)
    assert not result.is_stable
    assert result.reason == "insufficient_samples"
    assert result.sample_count == 10


def test_below_min_vram():
    """Peak VRAM < 100MB → not stable (GPU not really in use)."""
    samples = _make_samples(50, vram_mb=50.0)
    result = check_stability(samples)
    assert not result.is_stable
    assert result.reason == "below_min_vram"


def test_stable_constant_metrics():
    """50 constant samples → stable, all sub-checks True."""
    samples = _make_samples(50, vram_mb=5000.0, gpu_util=60.0, power=200.0)
    result = check_stability(samples, poll_interval_ms=500)
    assert result.is_stable
    assert result.vram_plateau
    assert result.util_stable
    assert result.power_stable
    assert "vram_plateau" in result.reason
    assert "util_stable" in result.reason
    assert "power_stable" in result.reason
    assert result.current_peak_vram_mb == 5000.0


def test_unstable_rising_vram():
    """Linearly increasing VRAM → vram_plateau=False."""
    base_t = time.time()
    samples = [
        ProbeSample(
            timestamp=base_t + i * 0.5,
            memory_used_mb=1000.0 + i * 100.0,  # Keeps rising
            memory_total_mb=80000.0,
            gpu_util_pct=60.0,
            power_watts=200.0,
        )
        for i in range(50)
    ]
    result = check_stability(samples, poll_interval_ms=500)
    assert not result.is_stable
    assert not result.vram_plateau
    assert "vram_rising" in result.reason


def test_unstable_oscillating_util():
    """Alternating 20%/90% GPU util → util_stable=False."""
    base_t = time.time()
    samples = [
        ProbeSample(
            timestamp=base_t + i * 0.5,
            memory_used_mb=5000.0,
            memory_total_mb=80000.0,
            gpu_util_pct=20.0 if i % 2 == 0 else 90.0,
            power_watts=200.0,
        )
        for i in range(50)
    ]
    result = check_stability(samples, poll_interval_ms=500)
    assert not result.is_stable
    assert not result.util_stable
    assert "util_unstable" in result.reason


def test_ramp_up_ignored():
    """Chaotic first 20 samples + 30 stable → is_stable=True."""
    base_t = time.time()
    # 20 chaotic ramp-up samples
    ramp = [
        ProbeSample(
            timestamp=base_t + i * 0.5,
            memory_used_mb=5000.0,
            memory_total_mb=80000.0,
            gpu_util_pct=float(i * 5 % 100),  # Wild oscillation
            power_watts=float(100 + i * 10 % 200),
        )
        for i in range(RAMP_UP_SAMPLES)
    ]
    # 40 stable samples (enough for plateau window + stability window)
    stable = [
        ProbeSample(
            timestamp=base_t + (RAMP_UP_SAMPLES + i) * 0.5,
            memory_used_mb=5000.0,
            memory_total_mb=80000.0,
            gpu_util_pct=60.0,
            power_watts=200.0,
        )
        for i in range(40)
    ]
    samples = ramp + stable
    result = check_stability(samples, poll_interval_ms=500)
    assert result.is_stable


def test_vram_jitter_within_tolerance():
    """±30MB jitter on 5000MB base → vram_plateau=True (peak set early, stays)."""
    import random
    random.seed(42)
    base_t = time.time()
    # Set a clear peak early
    samples = [
        ProbeSample(
            timestamp=base_t + i * 0.5,
            memory_used_mb=5030.0 if i == 5 else 5000.0 + random.uniform(-30, 29),
            memory_total_mb=80000.0,
            gpu_util_pct=60.0,
            power_watts=200.0,
        )
        for i in range(50)
    ]
    result = check_stability(samples, poll_interval_ms=500)
    assert result.vram_plateau


def test_all_zeros_not_stable():
    """0 VRAM, 0 util → not stable (below min vram)."""
    samples = _make_samples(50, vram_mb=0.0, gpu_util=0.0, power=0.0)
    result = check_stability(samples)
    assert not result.is_stable
    assert result.reason == "below_min_vram"


def test_power_unstable_blocks():
    """Stable VRAM + stable util, but wild power oscillation → is_stable=False."""
    base_t = time.time()
    samples = [
        ProbeSample(
            timestamp=base_t + i * 0.5,
            memory_used_mb=5000.0,
            memory_total_mb=80000.0,
            gpu_util_pct=60.0,
            power_watts=100.0 if i % 2 == 0 else 300.0,  # Wild 200W swings
        )
        for i in range(50)
    ]
    result = check_stability(samples, poll_interval_ms=500)
    assert not result.is_stable
    assert not result.power_stable
    assert "power_unstable" in result.reason
