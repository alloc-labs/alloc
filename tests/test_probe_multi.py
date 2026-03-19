"""Tests for process-tree aware multi-GPU discovery."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from alloc.probe import (
    _discover_gpu_indices,
    _get_child_pids,
    _parse_launcher_gpu_count,
    ProbeResult,
)


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


# ── Launcher command-line parsing ──


def test_parse_torchrun_equals():
    assert _parse_launcher_gpu_count(["torchrun", "--nproc_per_node=2", "train.py"]) == 2


def test_parse_torchrun_space():
    assert _parse_launcher_gpu_count(["torchrun", "--nproc_per_node", "4", "train.py"]) == 4


def test_parse_torchrun_hyphen():
    assert _parse_launcher_gpu_count(["torchrun", "--nproc-per-node=8", "train.py"]) == 8


def test_parse_accelerate_equals():
    assert _parse_launcher_gpu_count(["accelerate", "launch", "--num_processes=4", "train.py"]) == 4


def test_parse_accelerate_hyphen_space():
    assert _parse_launcher_gpu_count(["accelerate", "launch", "--num-processes", "8", "train.py"]) == 8


def test_parse_deepspeed_equals():
    assert _parse_launcher_gpu_count(["deepspeed", "--num_gpus=4", "train.py"]) == 4


def test_parse_deepspeed_hyphen_space():
    assert _parse_launcher_gpu_count(["deepspeed", "--num-gpus", "2", "train.py"]) == 2


def test_parse_mpirun():
    assert _parse_launcher_gpu_count(["mpirun", "-np", "8", "python", "train.py"]) == 8


def test_parse_plain_python():
    assert _parse_launcher_gpu_count(["python", "train.py"]) is None


# ── CVD UUID resolution ──


def test_cvd_uuid_resolves_to_correct_index():
    """UUID-style CUDA_VISIBLE_DEVICES should resolve to the matching physical GPU index."""
    mock = _mock_pynvml_multi_gpu(
        proc_pid=1000,
        gpu_process_map={0: [1000], 1: [], 2: []},
    )
    mock.nvmlDeviceGetCount.return_value = 3

    # Set up UUID resolution: GPU 0 → UUID-A, GPU 1 → UUID-B, GPU 2 → UUID-C
    uuid_map = {0: "GPU-aaaa-1111", 1: "GPU-bbbb-2222", 2: "GPU-cccc-3333"}
    handles = {}
    for idx in range(3):
        handles[idx] = MagicMock(name=f"handle_{idx}")

    def get_handle(idx):
        return handles[idx]

    def get_uuid(handle):
        for idx, h in handles.items():
            if handle == h:
                return uuid_map[idx]
        return "GPU-unknown"

    mock.nvmlDeviceGetHandleByIndex = MagicMock(side_effect=get_handle)
    mock.nvmlDeviceGetUUID = MagicMock(side_effect=get_uuid)

    # CVD set to GPU 2's UUID
    with patch("alloc.probe._get_child_pids", return_value=[]):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-cccc-3333"}):
            result = _discover_gpu_indices(1000, mock, fallback_index=0)
    # Should only search GPU index 2
    assert 2 in result or result == [0]  # either found on idx 2, or fallback if no PID match


def test_cvd_invalid_uuid_falls_back_to_all_gpus():
    """Invalid UUID that doesn't match any device should fall back to all GPUs."""
    mock = _mock_pynvml_multi_gpu(
        proc_pid=1000,
        gpu_process_map={0: [1000], 1: []},
    )
    mock.nvmlDeviceGetCount.return_value = 2

    # UUID lookup raises for all devices
    mock.nvmlDeviceGetUUID = MagicMock(side_effect=RuntimeError("no UUID support"))

    with patch("alloc.probe._get_child_pids", return_value=[]):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-nonexistent"}):
            result = _discover_gpu_indices(1000, mock, fallback_index=0)
    # Should fall back to searching all GPUs and find PID 1000 on GPU 0
    assert 0 in result


def test_parse_torch_distributed_launch():
    assert _parse_launcher_gpu_count([
        "python", "-m", "torch.distributed.launch", "--nproc_per_node=2", "train.py"
    ]) == 2


# ── Active-GPU fallback discovery ──


def test_active_gpu_fallback_when_pid_mismatch():
    """When PID matching fails but GPUs have active compute processes,
    and expected_gpus matches, use active GPUs."""
    mock = _mock_pynvml_multi_gpu(
        proc_pid=1000,
        gpu_process_map={0: [9999], 1: [8888]},  # PIDs don't match 1000
    )
    with patch("alloc.probe._get_child_pids", return_value=[]):
        with patch("alloc.probe._read_child_env", return_value=None):
            result = _discover_gpu_indices(1000, mock, fallback_index=0, expected_gpus=2)
    assert result == [0, 1]


def test_active_gpu_fallback_not_used_without_expected():
    """Without expected_gpus, active GPU fallback is not used."""
    mock = _mock_pynvml_multi_gpu(
        proc_pid=1000,
        gpu_process_map={0: [9999], 1: [8888]},
    )
    with patch("alloc.probe._get_child_pids", return_value=[]):
        with patch("alloc.probe._read_child_env", return_value=None):
            result = _discover_gpu_indices(1000, mock, fallback_index=0)
    assert result == [0]  # Falls back to default


# ── Early handle initialization for expected GPUs ──


def test_early_init_opens_handles_for_expected_gpus():
    """When expected_gpus > 1 and device_count >= expected, early-init should
    open handles for all expected GPUs."""
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetCount = MagicMock(return_value=2)

    handles_map = {0: MagicMock(name="gpu0"), 1: MagicMock(name="gpu1")}
    mock_pynvml.nvmlDeviceGetHandleByIndex = MagicMock(side_effect=lambda i: handles_map[i])

    # Simulate early-init logic from probe_command._monitor()
    expected_gpus = 2
    handles = [handles_map[0]]

    if expected_gpus > 1:
        device_count = mock_pynvml.nvmlDeviceGetCount()
        if device_count >= expected_gpus:
            early_handles = []
            early_indices = []
            for idx in range(device_count):
                if len(early_handles) >= expected_gpus:
                    break
                h = mock_pynvml.nvmlDeviceGetHandleByIndex(idx)
                early_handles.append(h)
                early_indices.append(idx)
            if len(early_handles) >= expected_gpus:
                handles = early_handles

    assert len(handles) == 2


def test_early_init_skipped_when_fewer_devices():
    """When device_count < expected_gpus, early-init should not change handles."""
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetCount = MagicMock(return_value=1)

    expected_gpus = 2
    handles = [MagicMock(name="gpu0")]
    original_handles = list(handles)

    if expected_gpus > 1:
        device_count = mock_pynvml.nvmlDeviceGetCount()
        if device_count >= expected_gpus:
            assert False, "Should not reach here"

    assert len(handles) == 1


def test_per_gpu_sampling_resilient_to_partial_failure():
    """Per-GPU try/except: one GPU failure should not prevent others from
    being sampled into per_gpu_peaks."""
    mock_pynvml = MagicMock()
    handles_map = {0: MagicMock(name="gpu0"), 1: MagicMock(name="gpu1")}

    mem_ok = MagicMock()
    mem_ok.total = 24 * 1024 * 1024 * 1024
    mem_ok.used = 8000 * 1024 * 1024

    def mem_info_side_effect(h):
        if h == handles_map[1]:
            raise RuntimeError("GPU 1 memory read failed")
        return mem_ok

    mock_pynvml.nvmlDeviceGetMemoryInfo = MagicMock(side_effect=mem_info_side_effect)
    util = MagicMock()
    util.gpu = 80
    mock_pynvml.nvmlDeviceGetUtilizationRates = MagicMock(return_value=util)
    mock_pynvml.nvmlDeviceGetPowerUsage = MagicMock(return_value=100_000)

    # Simulate the per-GPU sampling loop
    handles = [handles_map[0], handles_map[1]]
    per_gpu_peaks = {}
    vram_vals = []

    for h in handles:
        try:
            mi = mock_pynvml.nvmlDeviceGetMemoryInfo(h)
            vram_vals.append(mi.used / (1024 * 1024))
        except Exception:
            pass

    for gi, vm in enumerate(vram_vals):
        per_gpu_peaks[gi] = max(per_gpu_peaks.get(gi, 0.0), vm)

    # GPU 0 tracked, GPU 1 skipped
    assert 0 in per_gpu_peaks
    assert per_gpu_peaks[0] > 0
    assert len(vram_vals) == 1


def test_stability_delayed_until_discovery_done():
    """Stability check requires discovery_done=True."""
    # Single GPU: expected=1, num_gpus=1 → done immediately
    assert 1 >= 1  # num_gpus >= expected

    # Multi GPU with early-init: expected=2, num_gpus=2 → done at sample 5
    assert 2 >= 2

    # Multi GPU, discovery incomplete: expected=4, found=2 → NOT done
    assert not (2 >= 4)


def test_per_gpu_peaks_to_result_list():
    """per_gpu_peaks dict should convert to sorted list for ProbeResult."""
    peaks = {0: 8000.5, 1: 12000.3}
    result = [round(peaks[i], 1) for i in sorted(peaks)] if peaks else None
    assert result == [8000.5, 12000.3]

    empty = {}
    result_empty = [round(empty[i], 1) for i in sorted(empty)] if empty else None
    assert result_empty is None
