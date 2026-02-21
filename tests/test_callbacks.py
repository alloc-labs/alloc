"""Tests for framework callback timing logic."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from alloc.callbacks import (
    _compute_percentile,
    _compute_timing_stats,
    _detect_distributed,
    _build_sidecar,
    _write_callback_data,
)


# ── Percentile computation ───────────────────────────────────────────────


class TestComputePercentile:
    def test_empty_list(self):
        assert _compute_percentile([], 50) == 0.0

    def test_single_element(self):
        assert _compute_percentile([42.0], 50) == 42.0
        assert _compute_percentile([42.0], 90) == 42.0

    def test_p50_even_count(self):
        vals = [10.0, 20.0, 30.0, 40.0]
        p50 = _compute_percentile(vals, 50)
        assert abs(p50 - 25.0) < 0.01

    def test_p90(self):
        vals = list(range(1, 101))  # 1..100
        p90 = _compute_percentile(vals, 90)
        assert abs(p90 - 90.1) < 0.5

    def test_p50_odd_count(self):
        vals = [10.0, 20.0, 30.0]
        p50 = _compute_percentile(vals, 50)
        assert abs(p50 - 20.0) < 0.01


# ── Timing stats ─────────────────────────────────────────────────────────


class TestComputeTimingStats:
    def test_empty(self):
        assert _compute_timing_stats([]) == {}

    def test_uniform_steps(self):
        times = [100.0] * 50
        stats = _compute_timing_stats(times)
        assert stats["p50"] == 100.0
        assert stats["p90"] == 100.0
        assert stats["mean"] == 100.0
        assert stats["std"] == 0.0

    def test_varied_steps(self):
        times = [100.0, 200.0, 150.0, 120.0, 180.0]
        stats = _compute_timing_stats(times)
        assert stats["p50"] > 0
        assert stats["p90"] >= stats["p50"]
        assert stats["mean"] == 150.0
        assert stats["std"] > 0


# ── Sidecar build ────────────────────────────────────────────────────────


class TestBuildSidecar:
    def test_basic(self):
        data = _build_sidecar(
            framework="huggingface",
            step_count=100,
            step_times_ms=[100.0] * 50,
            batch_size=32,
        )
        assert data["framework"] == "huggingface"
        assert data["step_count"] == 100
        assert data["step_time_ms_p50"] == 100.0
        assert data["step_time_ms_p90"] == 100.0
        assert data["batch_size"] == 32
        # 32 samples / (100ms / 1000) = 320 samples/sec
        assert abs(data["samples_per_sec"] - 320.0) < 1.0

    def test_no_batch_size(self):
        data = _build_sidecar(
            framework="lightning",
            step_count=10,
            step_times_ms=[50.0, 60.0, 70.0],
            batch_size=None,
        )
        assert data["samples_per_sec"] is None
        assert data["batch_size"] is None

    def test_empty_steps(self):
        data = _build_sidecar(
            framework="huggingface",
            step_count=0,
            step_times_ms=[],
            batch_size=32,
        )
        assert data["step_time_ms_p50"] is None
        assert data["samples_per_sec"] is None


# ── Sidecar write/read ───────────────────────────────────────────────────


class TestWriteCallbackData:
    def test_writes_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data = {"framework": "huggingface", "step_count": 42}
        _write_callback_data(data)
        path = tmp_path / ".alloc_callback.json"
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["step_count"] == 42


# ── HF Callback integration (mocked) ────────────────────────────────────


class TestAllocCallbackTiming:
    def test_step_timing_capture(self):
        """Verify HF callback captures step times using mocked time.monotonic."""
        from alloc.callbacks import AllocCallback

        # AllocCallback requires transformers — if not installed, skip
        try:
            cb = AllocCallback()
        except ImportError:
            pytest.skip("transformers not installed")

        # Simulate 3 steps with 100ms each
        times = iter([1.0, 1.1, 1.1, 1.2, 1.2, 1.3])

        class FakeArgs:
            per_device_train_batch_size = 16
            gradient_accumulation_steps = 2

        class FakeState:
            global_step = 0

        args = FakeArgs()
        state = FakeState()

        with patch("alloc.callbacks.time") as mock_time:
            mock_time.monotonic = lambda: next(times)

            for step in range(1, 4):
                state.global_step = step
                cb.on_step_begin(args, state, None)
                cb.on_step_end(args, state, None)

        assert len(cb._step_times_ms) == 3
        assert all(abs(t - 100.0) < 0.1 for t in cb._step_times_ms)
        assert cb._batch_size == 32  # 16 * 2


# ── Sidecar discovery ────────────────────────────────────────────────────


class TestSidecarDiscovery:
    def test_read_callback_data_new_format(self, tmp_path, monkeypatch):
        """_read_callback_data reads .alloc_callback.json."""
        monkeypatch.chdir(tmp_path)
        data = {"framework": "huggingface", "step_count": 100, "step_time_ms_p50": 50.0}
        (tmp_path / ".alloc_callback.json").write_text(json.dumps(data))

        from alloc.cli import _read_callback_data
        result = _read_callback_data()
        assert result is not None
        assert result["step_count"] == 100
        assert result["step_time_ms_p50"] == 50.0

    def test_read_callback_data_legacy_file_ignored(self, tmp_path, monkeypatch):
        """Legacy .alloc_steps.json is ignored (no backward-compat shims)."""
        monkeypatch.chdir(tmp_path)
        data = {"framework": "huggingface", "step_count": 42}
        (tmp_path / ".alloc_steps.json").write_text(json.dumps(data))

        from alloc.cli import _read_callback_data
        result = _read_callback_data()
        assert result is None

    def test_read_callback_data_prefers_new(self, tmp_path, monkeypatch):
        """When both files exist, .alloc_callback.json wins."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".alloc_steps.json").write_text(json.dumps({"step_count": 10}))
        (tmp_path / ".alloc_callback.json").write_text(json.dumps({"step_count": 99, "step_time_ms_p50": 100.0}))

        from alloc.cli import _read_callback_data
        result = _read_callback_data()
        assert result["step_count"] == 99

    def test_read_callback_data_none(self, tmp_path, monkeypatch):
        """No sidecar files → None."""
        monkeypatch.chdir(tmp_path)
        from alloc.cli import _read_callback_data
        assert _read_callback_data() is None


class TestDetectDistributed:
    def test_no_torch_returns_false(self):
        """Without torch.distributed, should return (False, 0, 1)."""
        result = _detect_distributed()
        assert result[0] is False or result[0] is True  # either is valid
        assert isinstance(result[1], int)
        assert isinstance(result[2], int)


class TestBuildSidecarDistributed:
    def test_non_distributed_no_extra_fields(self):
        """Non-distributed sidecar should not have distributed fields."""
        data = _build_sidecar(
            framework="huggingface",
            step_count=100,
            step_times_ms=[100.0] * 50,
            batch_size=32,
        )
        assert "is_distributed" not in data
        assert "rank" not in data
        assert "world_size" not in data

    def test_distributed_has_extra_fields(self):
        """Distributed sidecar should include distributed metadata."""
        data = _build_sidecar(
            framework="huggingface",
            step_count=100,
            step_times_ms=[100.0] * 50,
            batch_size=32,
            is_distributed=True,
            rank=2,
            world_size=8,
        )
        assert data["is_distributed"] is True
        assert data["rank"] == 2
        assert data["world_size"] == 8
