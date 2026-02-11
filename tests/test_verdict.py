"""Tests for verdict display helpers."""

from __future__ import annotations

from alloc.display import (
    _classify_bottleneck_local,
    _compute_confidence_local,
    _qualitative_recommendation,
    _stop_reason_label,
)


# --- _classify_bottleneck_local ---

def test_classify_underutilized():
    """Low util + low VRAM util → underutilized."""
    assert _classify_bottleneck_local(4000.0, 80000.0, 30.0) == "underutilized"


def test_classify_memory_bound():
    """Low util + high VRAM util → memory_bound."""
    assert _classify_bottleneck_local(68000.0, 80000.0, 30.0) == "memory_bound"


def test_classify_compute_bound():
    """High util → compute_bound regardless of VRAM."""
    assert _classify_bottleneck_local(40000.0, 80000.0, 90.0) == "compute_bound"


def test_classify_balanced():
    """Mid util + mid VRAM → balanced."""
    assert _classify_bottleneck_local(56000.0, 80000.0, 60.0) == "balanced"


def test_classify_unknown_no_total():
    """gpu_total_vram_mb=None, util in middle → unknown."""
    assert _classify_bottleneck_local(5000.0, None, 55.0) == "unknown"


def test_classify_unknown_no_total_low_util():
    """gpu_total_vram_mb=None, low util → underutilized."""
    assert _classify_bottleneck_local(5000.0, None, 30.0) == "underutilized"


def test_classify_unknown_no_total_high_util():
    """gpu_total_vram_mb=None, high util → compute_bound."""
    assert _classify_bottleneck_local(5000.0, None, 90.0) == "compute_bound"


# --- _compute_confidence_local ---

def test_confidence_minimal():
    """3 samples, 10s → baseline 0.3."""
    assert _compute_confidence_local(3, 10.0) == 0.3


def test_confidence_good_probe():
    """50 samples, 90s → 0.3 + 0.2 (samples) + 0.1 (duration) = 0.6."""
    assert _compute_confidence_local(50, 90.0) == 0.6


def test_confidence_cap_at_06():
    """200 samples, 600s, 1000 steps → would be 1.0, but capped at 0.6 (NVML_ONLY)."""
    assert _compute_confidence_local(200, 600.0, step_count=1000) == 0.6


def test_confidence_with_steps():
    """20 samples, 30s, 100 steps → 0.3 + 0.2 + 0.2 = 0.6 (capped)."""
    result = _compute_confidence_local(20, 30.0, step_count=100)
    assert result == 0.6


def test_confidence_few_samples_short():
    """5 samples, 10s → 0.3 + 0.1 = 0.4."""
    assert _compute_confidence_local(5, 10.0) == 0.4


# --- _qualitative_recommendation ---

def test_recommendation_underutilized_low_vram():
    """Underutilized with <30% VRAM → significantly oversized."""
    rec = _qualitative_recommendation("underutilized", 15.0, 20.0)
    assert rec is not None
    assert "significantly oversized" in rec


def test_recommendation_underutilized():
    """Underutilized with moderate VRAM → underutilized suggestion."""
    rec = _qualitative_recommendation("underutilized", 45.0, 20.0)
    assert rec is not None
    assert "underutilized" in rec.lower()


def test_recommendation_memory_bound():
    """Memory bound → FSDP suggestion."""
    rec = _qualitative_recommendation("memory_bound", 85.0, 30.0)
    assert rec is not None
    assert "FSDP" in rec


def test_recommendation_compute_bound_headroom():
    """Compute bound with VRAM headroom → batch size suggestion."""
    rec = _qualitative_recommendation("compute_bound", 50.0, 90.0)
    assert rec is not None
    assert "batch size" in rec.lower()


def test_recommendation_balanced():
    """Balanced → None."""
    assert _qualitative_recommendation("balanced", 70.0, 60.0) is None


def test_recommendation_compute_bound_full():
    """Compute bound with high VRAM → None."""
    assert _qualitative_recommendation("compute_bound", 85.0, 90.0) is None


# --- _stop_reason_label ---

def test_stop_reason_stable():
    label = _stop_reason_label("stable", 34.2)
    assert "auto-stopped" in label
    assert "34.2" in label


def test_stop_reason_stable_no_duration():
    label = _stop_reason_label("stable", None)
    assert "auto-stopped" in label


def test_stop_reason_timeout():
    label = _stop_reason_label("timeout")
    assert "timeout" in label


def test_stop_reason_process_exit():
    label = _stop_reason_label("process_exit")
    assert "process exited" in label


def test_stop_reason_probe_steps():
    label = _stop_reason_label("probe_steps")
    assert "probe-steps" in label


def test_stop_reason_unknown():
    label = _stop_reason_label(None)
    assert label == "unknown"


# --- build_verdict_dict ---

def test_build_verdict_dict():
    """build_verdict_dict returns a complete dict with expected keys."""
    from alloc.display import build_verdict_dict
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class FakeProbe:
        peak_vram_mb: float = 40000.0
        avg_gpu_util: float = 85.0
        avg_power_watts: float = 250.0
        duration_seconds: float = 60.0
        samples: list = field(default_factory=lambda: [{}] * 30)
        exit_code: Optional[int] = 0
        gpu_name: Optional[str] = "NVIDIA A100-SXM4-80GB"
        gpu_total_vram_mb: Optional[float] = 81920.0
        stop_reason: Optional[str] = "stable"
        calibration_duration_s: Optional[float] = 45.0

        @property
        def vram_utilization_pct(self):
            if self.gpu_total_vram_mb and self.gpu_total_vram_mb > 0:
                return (self.peak_vram_mb / self.gpu_total_vram_mb) * 100
            return None

    result = FakeProbe()
    d = build_verdict_dict(result, artifact_path="/tmp/test.json.gz", step_count=100)

    assert d["bottleneck"] == "compute_bound"
    assert 0 < d["confidence"] <= 0.6
    assert d["peak_vram_mb"] == 40000.0
    assert d["gpu_name"] == "NVIDIA A100-SXM4-80GB"
    assert d["artifact_path"] == "/tmp/test.json.gz"
    assert d["process_status"] == "success"
    assert d["recommendation"] is not None  # compute_bound with headroom → increase batch size
