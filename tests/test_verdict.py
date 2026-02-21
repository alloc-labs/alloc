"""Tests for verdict display helpers."""

from __future__ import annotations

from alloc.display import (
    _stop_reason_label,
)


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
    assert label == "unknown"


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

    assert d["peak_vram_mb"] == 40000.0
    assert d["gpu_name"] == "NVIDIA A100-SXM4-80GB"
    assert d["artifact_path"] == "/tmp/test.json.gz"
    assert d["process_status"] == "success"
    assert d["avg_gpu_util"] == 85.0
    assert d["sample_count"] == 30
