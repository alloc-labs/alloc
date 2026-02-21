"""Tests for process-tree aware multi-GPU discovery."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from alloc.probe import _discover_gpu_indices, _get_child_pids, ProbeResult


def _mock_pynvml_multi_gpu(proc_pid, gpu_process_map):
    """Create mock pynvml that reports processes on specific GPUs.

    gpu_process_map: dict mapping gpu_index -> list of PIDs on that GPU
    """
    mock = MagicMock()
    mock.nvmlDeviceGetCount = MagicMock(return_value=len(gpu_process_map))

    handles = {}
    for idx in gpu_process_map:
        handles[idx] = MagicMock(name=f"handle_{idx}")

    def get_handle(idx):
        return handles.get(idx, MagicMock())

    mock.nvmlDeviceGetHandleByIndex = MagicMock(side_effect=get_handle)

    def get_procs(handle):
        for idx, h in handles.items():
            if handle == h:
                procs = []
                for pid in gpu_process_map[idx]:
                    p = MagicMock()
                    p.pid = pid
                    procs.append(p)
                return procs
        return []

    mock.nvmlDeviceGetComputeRunningProcesses = MagicMock(side_effect=get_procs)

    return mock


def test_single_gpu_discovery():
    """Process on one GPU should return that GPU index."""
    mock = _mock_pynvml_multi_gpu(
        proc_pid=1000,
        gpu_process_map={0: [1000], 1: [2000]},
    )
    with patch("alloc.probe._get_child_pids", return_value=[]):
        result = _discover_gpu_indices(1000, mock, fallback_index=0)
    assert result == [0]


def test_multi_gpu_discovery():
    """Process on two GPUs should return both indices."""
    mock = _mock_pynvml_multi_gpu(
        proc_pid=1000,
        gpu_process_map={0: [1000], 1: [1000], 2: [3000], 3: [4000]},
    )
    with patch("alloc.probe._get_child_pids", return_value=[]):
        result = _discover_gpu_indices(1000, mock, fallback_index=0)
    assert result == [0, 1]


def test_child_process_discovery():
    """Child processes on different GPUs should be discovered."""
    mock = _mock_pynvml_multi_gpu(
        proc_pid=1000,
        gpu_process_map={0: [1001], 1: [1002]},
    )
    with patch("alloc.probe._get_child_pids", side_effect=lambda pid: [1001, 1002] if pid == 1000 else []):
        result = _discover_gpu_indices(1000, mock, fallback_index=0)
    assert result == [0, 1]


def test_fallback_when_no_match():
    """When no GPU processes match, fallback to default index."""
    mock = _mock_pynvml_multi_gpu(
        proc_pid=1000,
        gpu_process_map={0: [9999], 1: [8888]},
    )
    with patch("alloc.probe._get_child_pids", return_value=[]):
        result = _discover_gpu_indices(1000, mock, fallback_index=0)
    assert result == [0]


def test_fallback_on_nvml_error():
    """When nvmlDeviceGetCount raises, fallback to default."""
    mock = MagicMock()
    mock.nvmlDeviceGetCount = MagicMock(side_effect=Exception("NVML error"))
    result = _discover_gpu_indices(1000, mock, fallback_index=2)
    assert result == [2]


def test_probe_result_multi_gpu_fields():
    """ProbeResult should carry multi-GPU metadata."""
    result = ProbeResult(
        num_gpus_detected=4,
        process_map=[
            {"gpu_index": 0},
            {"gpu_index": 1},
            {"gpu_index": 2},
            {"gpu_index": 3},
        ],
    )
    assert result.num_gpus_detected == 4
    assert len(result.process_map) == 4


def test_probe_result_defaults_single_gpu():
    """Default ProbeResult should indicate single GPU."""
    result = ProbeResult()
    assert result.num_gpus_detected == 1
    assert result.process_map is None
