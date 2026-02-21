"""Multi-signal stability detection for calibrate-and-exit mode.

Pure functions — no side effects, no threading, no I/O.
Determines when GPU metrics have stabilized enough to stop monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from alloc.probe import ProbeSample

RAMP_UP_SAMPLES = 20
MIN_SAMPLES_FOR_STABILITY = 30
MIN_PEAK_VRAM_MB = 100.0
VRAM_PLATEAU_WINDOW_S = 15.0
UTIL_STABILITY_WINDOW = 20
GPU_UTIL_STD_THRESHOLD = 5.0
POWER_STD_THRESHOLD = 10.0


@dataclass
class StabilityResult:
    """Result of a stability check."""

    is_stable: bool
    reason: str
    vram_plateau: bool
    util_stable: bool
    power_stable: bool
    current_peak_vram_mb: float
    sample_count: int


def check_stability(samples, poll_interval_ms=500):
    # type: (List[ProbeSample], int) -> StabilityResult
    """Check if GPU metrics have stabilized across multiple signals."""
    count = len(samples)

    if count < MIN_SAMPLES_FOR_STABILITY:
        return StabilityResult(
            is_stable=False,
            reason="insufficient_samples",
            vram_plateau=False,
            util_stable=False,
            power_stable=False,
            current_peak_vram_mb=0.0,
            sample_count=count,
        )

    peak_vram = max(s.memory_used_mb for s in samples)

    if peak_vram < MIN_PEAK_VRAM_MB:
        return StabilityResult(
            is_stable=False,
            reason="below_min_vram",
            vram_plateau=False,
            util_stable=False,
            power_stable=False,
            current_peak_vram_mb=peak_vram,
            sample_count=count,
        )

    vram_plateau = _check_vram_plateau(samples, poll_interval_ms)

    post_ramp = samples[RAMP_UP_SAMPLES:]
    recent_util = [s.gpu_util_pct for s in post_ramp[-UTIL_STABILITY_WINDOW:]]
    recent_power = [s.power_watts for s in post_ramp[-UTIL_STABILITY_WINDOW:]]

    util_stable = _check_metric_stable(recent_util, GPU_UTIL_STD_THRESHOLD)
    power_stable = _check_metric_stable(recent_power, POWER_STD_THRESHOLD)

    is_stable = vram_plateau and util_stable and power_stable

    reasons = []
    if vram_plateau:
        reasons.append("vram_plateau")
    if util_stable:
        reasons.append("util_stable")
    if power_stable:
        reasons.append("power_stable")

    if not is_stable:
        failing = []
        if not vram_plateau:
            failing.append("vram_rising")
        if not util_stable:
            failing.append("util_unstable")
        if not power_stable:
            failing.append("power_unstable")
        reason = "+".join(failing)
    else:
        reason = "+".join(reasons)

    return StabilityResult(
        is_stable=is_stable,
        reason=reason,
        vram_plateau=vram_plateau,
        util_stable=util_stable,
        power_stable=power_stable,
        current_peak_vram_mb=peak_vram,
        sample_count=count,
    )


def _check_vram_plateau(samples, poll_interval_ms):
    # type: (List[ProbeSample], int) -> bool
    """Check if VRAM has plateaued."""
    if not samples:
        return False

    peak_vram = -1.0
    peak_index = 0
    for i, s in enumerate(samples):
        if s.memory_used_mb > peak_vram:
            peak_vram = s.memory_used_mb
            peak_index = i

    window_samples = int(VRAM_PLATEAU_WINDOW_S / (poll_interval_ms / 1000.0))
    return (len(samples) - 1 - peak_index) >= window_samples


def _check_metric_stable(values, threshold):
    # type: (List[float], float) -> bool
    """Check if population std dev of values is below threshold."""
    if not values:
        return False

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return variance ** 0.5 < threshold
