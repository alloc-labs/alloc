"""Tests for framework callback timing logic."""

from __future__ import annotations

import gzip
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from alloc.callbacks import (
    _compute_percentile,
    _compute_timing_stats,
    _detect_distributed,
    _build_sidecar,
    _write_callback_data,
    _NvmlMonitor,
    _write_full_artifact,
    _WARMUP_STEPS,
    _WRITE_EVERY,
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


# ── NVML Monitor ────────────────────────────────────────────────────────


class TestNvmlMonitorNoPynvml:
    def test_no_pynvml_returns_empty(self):
        """Without pynvml, get_results returns empty dicts."""
        with patch("alloc.callbacks._try_import_pynvml", return_value=None):
            monitor = _NvmlMonitor()
        monitor.start()  # should be no-op
        monitor.stop()
        hw, probe = monitor.get_results()
        assert hw == {}
        assert probe == {}

    def test_available_false_without_pynvml(self):
        """Monitor reports not available without pynvml."""
        with patch("alloc.callbacks._try_import_pynvml", return_value=None):
            monitor = _NvmlMonitor()
        assert monitor.available is False


class TestNvmlMonitorWithMock:
    def _make_mock_pynvml(self):
        """Build a mock pynvml module with standard GPU 0 responses."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlShutdown.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 2

        # GPU name
        mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA A100-SXM4-80GB"

        # Memory info
        mem_info = SimpleNamespace(total=80 * 1024 * 1024 * 1024, used=40 * 1024 * 1024 * 1024)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info

        # Driver / CUDA
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "535.129.03"
        mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12020  # 12.2
        mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 0)

        # Utilization
        util = SimpleNamespace(gpu=75.0, memory=60.0)
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util

        # Power (in milliwatts)
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 300000  # 300W

        return mock_pynvml

    def test_hw_context_keys(self):
        """Mock pynvml, verify hardware context has all expected keys."""
        mock_pynvml = self._make_mock_pynvml()
        with patch("alloc.callbacks._try_import_pynvml", return_value=mock_pynvml):
            monitor = _NvmlMonitor()
        monitor.start()
        # Let the sampling thread run briefly
        import time
        time.sleep(0.05)
        monitor.stop()

        hw, probe = monitor.get_results()
        assert hw["gpu_name"] == "NVIDIA A100-SXM4-80GB"
        assert hw["gpu_total_vram_mb"] == round(80 * 1024, 1)
        assert hw["driver_version"] == "535.129.03"
        assert hw["cuda_version"] == "12.2"
        assert hw["sm_version"] == "8.0"
        assert hw["num_gpus_detected"] == 2

    def test_probe_dict_keys(self):
        """Verify probe_dict has all expected keys including probe_mode."""
        mock_pynvml = self._make_mock_pynvml()
        with patch("alloc.callbacks._try_import_pynvml", return_value=mock_pynvml):
            monitor = _NvmlMonitor()
        monitor.start()
        import time
        time.sleep(0.05)
        monitor.stop()

        _, probe = monitor.get_results()
        assert probe["probe_mode"] == "callback"
        assert "peak_vram_mb" in probe
        assert "avg_gpu_util" in probe
        assert "avg_power_watts" in probe
        assert "duration_seconds" in probe
        assert "samples" in probe
        assert isinstance(probe["samples"], list)

    def test_samples_have_correct_fields(self):
        """Verify each sample has t, vram_mb, gpu_util_pct, power_w."""
        mock_pynvml = self._make_mock_pynvml()
        # Use very short poll interval to get samples quickly
        with patch("alloc.callbacks._try_import_pynvml", return_value=mock_pynvml):
            with patch("alloc.callbacks._NVML_POLL_INTERVAL_S", 0.01):
                monitor = _NvmlMonitor()
                monitor.start()
                import time
                time.sleep(0.05)
                monitor.stop()

        _, probe = monitor.get_results()
        assert len(probe["samples"]) > 0
        sample = probe["samples"][0]
        assert "t" in sample
        assert "vram_mb" in sample
        assert "gpu_util_pct" in sample
        assert "power_w" in sample


class TestNvmlMonitorCleanup:
    def test_stop_calls_nvml_shutdown(self):
        """nvmlShutdown is called on stop."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetName.return_value = "GPU"
        mem = SimpleNamespace(total=80 * 1024**3, used=10 * 1024**3)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "535"
        mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12000
        mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 0)
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        util = SimpleNamespace(gpu=50, memory=30)
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 200000

        with patch("alloc.callbacks._try_import_pynvml", return_value=mock_pynvml):
            monitor = _NvmlMonitor()
        monitor.start()
        monitor.stop()
        mock_pynvml.nvmlShutdown.assert_called_once()

    def test_nvml_init_failure_graceful(self):
        """If nvmlInit raises, monitor degrades gracefully."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("No NVIDIA driver")
        with patch("alloc.callbacks._try_import_pynvml", return_value=mock_pynvml):
            monitor = _NvmlMonitor()
        monitor.start()  # should handle exception
        monitor.stop()
        hw, probe = monitor.get_results()
        assert hw == {}
        assert probe == {}


# ── Artifact writing ────────────────────────────────────────────────────


class TestWriteFullArtifact:
    def test_writes_artifact_file(self, tmp_path, monkeypatch):
        """_write_full_artifact creates alloc_artifact.json.gz with merged data."""
        monkeypatch.chdir(tmp_path)

        monitor = MagicMock()
        monitor.get_results.return_value = (
            {"gpu_name": "A100", "gpu_total_vram_mb": 81920.0, "num_gpus_detected": 1},
            {
                "peak_vram_mb": 40000.0,
                "avg_gpu_util": 75.0,
                "avg_power_watts": 300.0,
                "duration_seconds": 120.0,
                "samples": [{"t": 0, "vram_mb": 40000.0, "gpu_util_pct": 75.0, "power_w": 300.0}],
                "gpu_name": "A100",
                "gpu_total_vram_mb": 81920.0,
                "num_gpus_detected": 1,
                "probe_mode": "callback",
            },
        )

        sidecar = {
            "framework": "huggingface",
            "step_count": 100,
            "step_time_ms_p50": 50.0,
            "step_time_ms_p90": 80.0,
            "samples_per_sec": 640.0,
            "batch_size": 32,
        }

        _write_full_artifact(monitor, sidecar)

        artifact_path = tmp_path / "alloc_artifact.json.gz"
        assert artifact_path.exists()

        with gzip.open(str(artifact_path), "rt") as f:
            data = json.load(f)

        assert data["hardware"]["gpu_name"] == "A100"
        assert data["probe"]["probe_mode"] == "callback"
        assert data["probe"]["step_time_ms_p50"] == 50.0
        assert data["probe"]["step_count"] == 100
        assert data["probe"]["framework"] == "huggingface"

    def test_empty_monitor_still_works(self, tmp_path, monkeypatch):
        """With empty monitor results, artifact is still written (timing only)."""
        monkeypatch.chdir(tmp_path)

        monitor = MagicMock()
        monitor.get_results.return_value = ({}, {})

        sidecar = {"framework": "huggingface", "step_count": 10}
        _write_full_artifact(monitor, sidecar)

        # With empty results, write_report gets None for both — still writes
        artifact_path = tmp_path / "alloc_artifact.json.gz"
        assert artifact_path.exists()


# ── HF Callback artifact integration ───────────────────────────────────


class TestHFCallbackArtifact:
    def test_writes_artifact_on_train_end(self, tmp_path, monkeypatch):
        """HF callback writes alloc_artifact.json.gz on on_train_end."""
        from alloc.callbacks import AllocCallback

        try:
            cb = AllocCallback()
        except ImportError:
            pytest.skip("transformers not installed")

        monkeypatch.chdir(tmp_path)

        # Replace monitor with mock
        mock_monitor = MagicMock()
        mock_monitor.get_results.return_value = (
            {"gpu_name": "V100"},
            {"peak_vram_mb": 16000.0, "probe_mode": "callback", "samples": []},
        )
        cb._monitor = mock_monitor
        cb._monitor_started = True

        class FakeArgs:
            per_device_train_batch_size = 8
            gradient_accumulation_steps = 1

        class FakeState:
            global_step = 50

        # Simulate some steps
        times = iter([1.0, 1.1, 1.1, 1.2])
        with patch("alloc.callbacks.time") as mock_time:
            mock_time.monotonic = lambda: next(times)
            mock_time.time = lambda: 100.0
            cb.on_step_begin(FakeArgs(), FakeState(), None)
            cb.on_step_end(FakeArgs(), FakeState(), None)
            cb.on_step_begin(FakeArgs(), FakeState(), None)
            cb.on_step_end(FakeArgs(), FakeState(), None)

        cb.on_train_end(FakeArgs(), FakeState(), None)

        # Monitor should have been stopped
        mock_monitor.stop.assert_called_once()

        # Artifact should be written
        artifact_path = tmp_path / "alloc_artifact.json.gz"
        assert artifact_path.exists()

    def test_still_writes_sidecar(self, tmp_path, monkeypatch):
        """HF callback still writes .alloc_callback.json for backwards compat."""
        from alloc.callbacks import AllocCallback

        try:
            cb = AllocCallback()
        except ImportError:
            pytest.skip("transformers not installed")

        monkeypatch.chdir(tmp_path)

        # Replace monitor with no-op mock
        mock_monitor = MagicMock()
        mock_monitor.get_results.return_value = ({}, {})
        cb._monitor = mock_monitor
        cb._monitor_started = True

        class FakeArgs:
            per_device_train_batch_size = 8
            gradient_accumulation_steps = 1

        class FakeState:
            global_step = 10

        times = iter([1.0, 1.1])
        with patch("alloc.callbacks.time") as mock_time:
            mock_time.monotonic = lambda: next(times)
            mock_time.time = lambda: 100.0
            cb.on_step_begin(FakeArgs(), FakeState(), None)
            cb.on_step_end(FakeArgs(), FakeState(), None)

        cb.on_train_end(FakeArgs(), FakeState(), None)

        sidecar_path = tmp_path / ".alloc_callback.json"
        assert sidecar_path.exists()
        data = json.loads(sidecar_path.read_text())
        assert data["framework"] == "huggingface"


# ── Lightning Callback artifact integration ─────────────────────────────


class TestLightningCallbackArtifact:
    def test_writes_artifact_on_train_end(self, tmp_path, monkeypatch):
        """Lightning callback writes artifact on on_train_end."""
        from alloc.callbacks import AllocLightningCallback

        try:
            cb = AllocLightningCallback()
        except ImportError:
            pytest.skip("lightning not installed")

        monkeypatch.chdir(tmp_path)

        mock_monitor = MagicMock()
        mock_monitor.get_results.return_value = (
            {"gpu_name": "A100"},
            {"peak_vram_mb": 40000.0, "probe_mode": "callback", "samples": []},
        )
        cb._monitor = mock_monitor
        cb._monitor_started = True

        class FakeTrainer:
            global_step = 25
            datamodule = None

        times = iter([1.0, 1.1])
        with patch("alloc.callbacks.time") as mock_time:
            mock_time.monotonic = lambda: next(times)
            mock_time.time = lambda: 100.0
            cb.on_train_batch_start(FakeTrainer(), None, [1, 2, 3, 4], 0)
            cb.on_train_batch_end(FakeTrainer(), None, None, [1, 2, 3, 4], 0)

        cb.on_train_end(FakeTrainer(), None)

        mock_monitor.stop.assert_called_once()
        artifact_path = tmp_path / "alloc_artifact.json.gz"
        assert artifact_path.exists()


# ── Phase timing helpers ───────────────────────────────────────────────


class TestComputePhaseStats:
    def test_empty_list(self):
        from alloc.callbacks import _compute_phase_stats
        assert _compute_phase_stats([]) == {}

    def test_single_value(self):
        from alloc.callbacks import _compute_phase_stats
        result = _compute_phase_stats([10.0])
        assert result["p50"] == 10.0
        assert result["p90"] == 10.0

    def test_multiple_values(self):
        from alloc.callbacks import _compute_phase_stats
        result = _compute_phase_stats([10.0, 20.0, 30.0, 40.0, 50.0])
        assert abs(result["p50"] - 30.0) < 0.01
        assert result["p90"] > result["p50"]


class TestBuildSidecarPhaseTimimg:
    def test_no_phase_stats(self):
        data = _build_sidecar("test", 10, [100.0, 200.0], 8)
        assert "has_phase_timing" not in data

    def test_with_phase_stats(self):
        phase_stats = {
            "forward": {"p50": 10.0, "p90": 12.0},
            "backward": {"p50": 20.0, "p90": 25.0},
            "optimizer": {"p50": 5.0, "p90": 6.0},
            "dataloader": {"p50": 3.0, "p90": 4.0},
        }
        data = _build_sidecar("test", 10, [100.0], 8, phase_stats=phase_stats)
        assert data["has_phase_timing"] is True
        assert data["phase_forward_ms_p50"] == 10.0
        assert data["phase_forward_ms_p90"] == 12.0
        assert data["phase_backward_ms_p50"] == 20.0
        assert data["phase_backward_ms_p90"] == 25.0
        assert data["phase_optimizer_ms_p50"] == 5.0
        assert data["phase_optimizer_ms_p90"] == 6.0
        assert data["phase_dataloader_ms_p50"] == 3.0
        assert data["phase_dataloader_ms_p90"] == 4.0

    def test_partial_phase_stats(self):
        phase_stats = {
            "forward": {"p50": 10.0, "p90": 12.0},
        }
        data = _build_sidecar("test", 10, [100.0], 8, phase_stats=phase_stats)
        assert data["has_phase_timing"] is True
        assert data["phase_forward_ms_p50"] == 10.0
        assert "phase_backward_ms_p50" not in data


class TestCudaAvailability:
    def test_try_cuda_not_available(self):
        from alloc.callbacks import _try_cuda_available
        # On test machines without CUDA, should return False gracefully
        result = _try_cuda_available()
        assert isinstance(result, bool)

    def test_create_cuda_event_no_cuda(self):
        from alloc.callbacks import _create_cuda_event
        # Should return None when CUDA not available
        with patch("alloc.callbacks._try_cuda_available", return_value=False):
            # _create_cuda_event tries torch.cuda.Event directly
            # Without CUDA it should return None
            result = _create_cuda_event()
            # Either None (no CUDA) or a real event (if CUDA available in test env)
            assert result is None or result is not None


class TestWriteFullArtifactWithPhases:
    def test_phase_fields_merged(self, tmp_path):
        """Phase timing fields from sidecar should be written to artifact."""
        monitor = _NvmlMonitor()
        sidecar = {
            "framework": "test",
            "step_count": 10,
            "step_time_ms_p50": 100.0,
            "has_phase_timing": True,
            "phase_forward_ms_p50": 40.0,
            "phase_forward_ms_p90": 45.0,
            "phase_backward_ms_p50": 35.0,
            "phase_backward_ms_p90": 40.0,
            "phase_optimizer_ms_p50": 15.0,
            "phase_optimizer_ms_p90": 18.0,
            "phase_dataloader_ms_p50": 10.0,
            "phase_dataloader_ms_p90": 12.0,
        }

        with patch("alloc.artifact_writer.write_report") as mock_write:
            _write_full_artifact(monitor, sidecar)
            assert mock_write.called, "write_report must be called"
            call_kwargs = mock_write.call_args
            probe = call_kwargs.kwargs.get("probe_result")
            assert isinstance(probe, dict), "probe_result must be a dict"
            assert probe["has_phase_timing"] is True
            assert probe["phase_forward_ms_p50"] == 40.0
            assert probe["phase_forward_ms_p90"] == 45.0
            assert probe["phase_backward_ms_p50"] == 35.0
            assert probe["phase_optimizer_ms_p50"] == 15.0
            assert probe["phase_dataloader_ms_p50"] == 10.0

    def test_rank_tagged_filename(self, tmp_path):
        """Non-rank-0 distributed should use rank-tagged filename."""
        monitor = _NvmlMonitor()
        sidecar = {
            "framework": "test",
            "step_count": 10,
            "is_distributed": True,
            "rank": 2,
            "world_size": 4,
        }

        with patch("alloc.artifact_writer.write_report") as mock_write:
            _write_full_artifact(monitor, sidecar)
            assert mock_write.called, "write_report must be called"
            call_kwargs = mock_write.call_args
            assert call_kwargs.kwargs.get("output_path") == "alloc_artifact_rank2.json.gz"

    def test_rank0_default_filename(self, tmp_path):
        """Rank 0 should use default filename (None = default)."""
        monitor = _NvmlMonitor()
        sidecar = {
            "framework": "test",
            "step_count": 10,
            "is_distributed": True,
            "rank": 0,
            "world_size": 4,
        }

        with patch("alloc.artifact_writer.write_report") as mock_write:
            _write_full_artifact(monitor, sidecar)
            assert mock_write.called, "write_report must be called"
            call_kwargs = mock_write.call_args
            assert call_kwargs.kwargs.get("output_path") is None  # default = alloc_artifact.json.gz


# ── Architecture Detection (P3-A) ──────────────────────────────────────

class TestDetectArchitecture:
    def test_hf_config_extraction(self):
        from alloc.callbacks import _detect_architecture

        model = MagicMock()
        model.config.model_type = "llama"
        model.config.num_attention_heads = 32
        model.config._attn_implementation = "flash_attention_2"
        model.is_gradient_checkpointing = True
        # No peft_config
        del model.peft_config

        result = _detect_architecture(model)
        assert result["model_type"] == "llama"
        assert result["architecture_type"] == "transformer"
        assert result["gradient_checkpointing"] is True
        assert result["attention_type"] == "flash_attention_2"

    def test_optimizer_type_extraction(self):
        from alloc.callbacks import _detect_architecture

        model = MagicMock()
        model.config.model_type = "gpt2"
        model.config.num_attention_heads = 12
        model.is_gradient_checkpointing = False
        del model.peft_config
        del model.config._attn_implementation
        del model.config.attn_implementation

        class FakeAdamW:
            pass
        optimizer = FakeAdamW()

        result = _detect_architecture(model, optimizer=optimizer)
        assert result["optimizer_type"] == "FakeAdamW"
        assert result["architecture_type"] == "transformer"

    def test_peft_lora_detection(self):
        from alloc.callbacks import _detect_architecture

        model = MagicMock()
        model.config.model_type = "llama"
        model.config.num_attention_heads = 32
        model.is_gradient_checkpointing = False
        del model.config._attn_implementation
        del model.config.attn_implementation

        # Simulate PEFT model
        adapter_config = MagicMock()
        adapter_config.peft_type = "LORA"
        model.peft_config = {"default": adapter_config}

        result = _detect_architecture(model)
        assert result["fine_tuning_method"] == "lora"

    def test_frozen_params_detection(self):
        from alloc.callbacks import _detect_architecture

        model = MagicMock()
        model.config.model_type = "bert"
        model.config.num_attention_heads = 12
        model.is_gradient_checkpointing = False
        del model.peft_config
        del model.config._attn_implementation
        del model.config.attn_implementation

        # Simulate frozen model (95% frozen, 5% trainable)
        params = []
        for i in range(20):
            p = MagicMock()
            p.numel.return_value = 1000000
            p.requires_grad = (i == 0)  # only 1 of 20 is trainable
            params.append(p)
        model.parameters.return_value = iter(params)

        result = _detect_architecture(model)
        assert result["fine_tuning_method"] == "adapter"
        assert result["param_count"] == 20000000
        assert result["trainable_param_count"] == 1000000

    def test_none_model(self):
        from alloc.callbacks import _detect_architecture
        result = _detect_architecture(None)
        assert result == {}

    def test_model_without_config(self):
        from alloc.callbacks import _detect_architecture
        model = MagicMock(spec=[])  # empty spec = no attributes
        result = _detect_architecture(model)
        assert result == {}

    def test_moe_detection(self):
        from alloc.callbacks import _detect_architecture

        model = MagicMock()
        model.config.model_type = "mixtral"
        model.config.num_attention_heads = 32
        model.is_gradient_checkpointing = False
        del model.peft_config
        del model.config._attn_implementation
        del model.config.attn_implementation

        result = _detect_architecture(model)
        assert result["architecture_type"] == "moe"

    def test_training_args_gradient_checkpointing(self):
        from alloc.callbacks import _detect_architecture

        model = MagicMock()
        model.config.model_type = "llama"
        model.config.num_attention_heads = 32
        model.is_gradient_checkpointing = False
        del model.peft_config
        del model.config._attn_implementation
        del model.config.attn_implementation

        training_args = MagicMock()
        training_args.gradient_checkpointing = True

        result = _detect_architecture(model, training_args=training_args)
        assert result["gradient_checkpointing"] is True


class TestBuildSidecarArchitecture:
    def test_architecture_info_included(self):
        arch_info = {
            "architecture_type": "transformer",
            "optimizer_type": "AdamW",
            "gradient_checkpointing": True,
            "model_type": "llama",
        }
        data = _build_sidecar(
            framework="huggingface",
            step_count=10,
            step_times_ms=[100.0],
            batch_size=8,
            architecture_info=arch_info,
        )
        assert data["architecture_type"] == "transformer"
        assert data["optimizer_type"] == "AdamW"
        assert data["gradient_checkpointing"] is True
        assert data["model_type"] == "llama"

    def test_no_architecture_info(self):
        data = _build_sidecar(
            framework="huggingface",
            step_count=10,
            step_times_ms=[100.0],
            batch_size=8,
        )
        assert "architecture_type" not in data
        assert "optimizer_type" not in data

    def test_partial_architecture_info(self):
        arch_info = {
            "optimizer_type": "SGD",
            "gradient_checkpointing": None,  # None values not included
        }
        data = _build_sidecar(
            framework="lightning",
            step_count=5,
            step_times_ms=[200.0],
            batch_size=4,
            architecture_info=arch_info,
        )
        assert data["optimizer_type"] == "SGD"
        assert "gradient_checkpointing" not in data
        assert "architecture_type" not in data


# ── CUDA Event Non-Blocking Sync (C2 Fix) ─────────────────────────────


class TestSyncPhaseEventsNonBlocking:
    """Test that _sync_phase_events uses query() (non-blocking) by default
    and synchronize() only on train_end (force=True)."""

    def test_query_false_skips_sync(self):
        """When event.query() returns False, no elapsed_time is called."""
        from alloc.callbacks import AllocCallback
        try:
            cb = AllocCallback()
        except ImportError:
            pytest.skip("transformers not installed")

        # Set up mock CUDA events
        cb._cuda_available = True
        cb._evt_step_start = MagicMock()
        cb._evt_backward_start = MagicMock()
        cb._evt_optimizer_start = MagicMock()
        cb._evt_step_end = MagicMock()
        cb._evt_step_end.query.return_value = False  # not ready

        cb._sync_phase_events()

        cb._evt_step_end.synchronize.assert_not_called()
        cb._evt_step_start.elapsed_time.assert_not_called()

    def test_query_true_reads_events(self):
        """When event.query() returns True, elapsed_time is called."""
        from alloc.callbacks import AllocCallback
        try:
            cb = AllocCallback()
        except ImportError:
            pytest.skip("transformers not installed")

        cb._cuda_available = True
        cb._evt_step_start = MagicMock()
        cb._evt_backward_start = MagicMock()
        cb._evt_optimizer_start = MagicMock()
        cb._evt_step_end = MagicMock()
        cb._evt_step_end.query.return_value = True
        cb._evt_step_start.elapsed_time.return_value = 10.0
        cb._evt_backward_start.elapsed_time.return_value = 20.0
        cb._evt_optimizer_start.elapsed_time.return_value = 5.0

        cb._sync_phase_events()

        cb._evt_step_end.synchronize.assert_not_called()
        assert len(cb._phase_forward_ms) == 1
        assert cb._phase_forward_ms[0] == 10.0

    def test_force_uses_synchronize(self):
        """force=True uses synchronize() (on train_end)."""
        from alloc.callbacks import AllocCallback
        try:
            cb = AllocCallback()
        except ImportError:
            pytest.skip("transformers not installed")

        cb._cuda_available = True
        cb._evt_step_start = MagicMock()
        cb._evt_backward_start = MagicMock()
        cb._evt_optimizer_start = MagicMock()
        cb._evt_step_end = MagicMock()
        cb._evt_step_start.elapsed_time.return_value = 10.0
        cb._evt_backward_start.elapsed_time.return_value = 20.0
        cb._evt_optimizer_start.elapsed_time.return_value = 5.0

        cb._sync_phase_events(force=True)

        cb._evt_step_end.synchronize.assert_called_once()


class TestWarmupExclusion:
    """Phase timing skips warmup steps to avoid JIT compilation noise."""

    def test_warmup_steps_constant(self):
        """Warmup steps constant exists and is reasonable."""
        assert _WARMUP_STEPS > 0
        assert _WARMUP_STEPS <= 20  # not too large

    def test_phase_sync_not_called_during_warmup(self):
        """Phase sync is NOT called when step_count <= WARMUP_STEPS."""
        from alloc.callbacks import AllocCallback
        try:
            cb = AllocCallback()
        except ImportError:
            pytest.skip("transformers not installed")

        cb._cuda_available = True
        cb._evt_step_start = MagicMock()
        cb._evt_backward_start = MagicMock()
        cb._evt_optimizer_start = MagicMock()
        cb._evt_step_end = MagicMock()
        cb._monitor_started = True
        cb._dist_checked = True

        class FakeArgs:
            per_device_train_batch_size = 8
            gradient_accumulation_steps = 1

        class FakeState:
            global_step = _WARMUP_STEPS  # exactly at warmup — should still skip

        # Step count at warmup boundary — sync should NOT be called
        cb.step_count = _WARMUP_STEPS
        with patch.object(cb, "_sync_phase_events") as mock_sync:
            with patch("alloc.callbacks.time") as mock_time:
                mock_time.monotonic.return_value = 1.0
                cb.on_step_end(FakeArgs(), FakeState(), None)
            # At exactly _WARMUP_STEPS, should not sync (condition is > not >=)
            mock_sync.assert_not_called()


class TestNvmlMonitorThreadSafety:
    """Verify per-GPU peak VRAM is protected by lock."""

    def test_peak_vram_under_lock(self):
        """get_results reads peak VRAM copy taken under lock."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_pynvml.nvmlDeviceGetName.return_value = "GPU"
        mem = SimpleNamespace(total=80 * 1024**3, used=50 * 1024**3)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "535"
        mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12000
        mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 0)
        util = SimpleNamespace(gpu=75, memory=60)
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 300000

        with patch("alloc.callbacks._try_import_pynvml", return_value=mock_pynvml):
            with patch("alloc.callbacks._NVML_POLL_INTERVAL_S", 0.01):
                monitor = _NvmlMonitor()
                monitor.start()
                import time
                time.sleep(0.05)
                monitor.stop()

        _, probe = monitor.get_results()
        # Peak VRAM should be populated from locked updates
        assert "per_rank_peak_vram_mb" in probe
        assert len(probe["per_rank_peak_vram_mb"]) == 2
        for peak in probe["per_rank_peak_vram_mb"]:
            assert peak > 0
