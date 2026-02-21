"""Tests for hardware context capture in probe."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from alloc.probe import ProbeResult


def _make_mock_pynvml(
    driver_version="535.129.03",
    cuda_driver_version=12020,
    compute_capability=(8, 0),
    gpu_name="NVIDIA A100-SXM4-80GB",
    total_vram_bytes=80 * 1024 * 1024 * 1024,
):
    """Create a mock pynvml module with hardware info methods."""
    mock = MagicMock()
    mock.nvmlInit = MagicMock()
    mock.nvmlShutdown = MagicMock()
    mock.nvmlDeviceGetHandleByIndex = MagicMock(return_value="handle")
    mock.nvmlDeviceGetName = MagicMock(return_value=gpu_name)

    mem_info = MagicMock()
    mem_info.total = total_vram_bytes
    mem_info.used = 1024 * 1024 * 1024  # 1GB
    mock.nvmlDeviceGetMemoryInfo = MagicMock(return_value=mem_info)

    util = MagicMock()
    util.gpu = 50.0
    mock.nvmlDeviceGetUtilizationRates = MagicMock(return_value=util)
    mock.nvmlDeviceGetPowerUsage = MagicMock(return_value=250000)  # 250W in mW

    mock.nvmlSystemGetDriverVersion = MagicMock(return_value=driver_version)
    mock.nvmlSystemGetCudaDriverVersion = MagicMock(return_value=cuda_driver_version)
    mock.nvmlDeviceGetCudaComputeCapability = MagicMock(return_value=compute_capability)

    return mock


def test_probe_result_has_hw_fields():
    """ProbeResult dataclass should have hardware fields."""
    result = ProbeResult(
        driver_version="535.129.03",
        cuda_version="12.2",
        sm_version="8.0",
    )
    assert result.driver_version == "535.129.03"
    assert result.cuda_version == "12.2"
    assert result.sm_version == "8.0"


def test_probe_result_hw_defaults_none():
    """Hardware fields default to None."""
    result = ProbeResult()
    assert result.driver_version is None
    assert result.cuda_version is None
    assert result.sm_version is None


def test_cuda_version_formatting():
    """CUDA driver version int should be formatted as major.minor."""
    # 12020 → 12.2, 11080 → 11.8
    mock = _make_mock_pynvml(cuda_driver_version=12020)
    # Simulate what probe does:
    cuda_ver_int = mock.nvmlSystemGetCudaDriverVersion()
    major = cuda_ver_int // 1000
    minor = (cuda_ver_int % 1000) // 10
    assert f"{major}.{minor}" == "12.2"


def test_cuda_version_formatting_11_8():
    """CUDA 11.8 formatting."""
    cuda_ver_int = 11080
    major = cuda_ver_int // 1000
    minor = (cuda_ver_int % 1000) // 10
    assert f"{major}.{minor}" == "11.8"


def test_sm_version_formatting():
    """SM version should be major.minor string."""
    sm_major, sm_minor = 9, 0
    assert f"{sm_major}.{sm_minor}" == "9.0"
