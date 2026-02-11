"""Tests for Ghost Scan VRAM math."""

from alloc.ghost import ghost, GhostReport


def test_7b_fp16_weights():
    """7B params in fp16 should be ~14GB weights."""
    report = ghost(param_count_b=7.0, dtype="fp16")
    assert isinstance(report, GhostReport)
    assert 12 < report.weights_gb < 16  # 7e9 * 2 / 1024^3 ≈ 13.04
    assert report.param_count_b == 7.0


def test_7b_fp32_weights_double():
    """fp32 weights should be 2x fp16."""
    fp16 = ghost(param_count_b=1.0, dtype="fp16")
    fp32 = ghost(param_count_b=1.0, dtype="fp32")
    assert abs(fp32.weights_gb - fp16.weights_gb * 2) < 0.1


def test_optimizer_adam():
    """Adam optimizer should add 12 bytes per param."""
    report = ghost(param_count_b=1.0, dtype="fp16")
    # 1e9 * 12 / 1024^3 ≈ 11.18 GB
    assert 10 < report.optimizer_gb < 13


def test_total_is_sum():
    """Total should be sum of components including buffer."""
    report = ghost(param_count_b=1.0, dtype="fp16")
    expected = report.weights_gb + report.gradients_gb + report.optimizer_gb + report.activations_gb + report.buffer_gb
    assert abs(report.total_gb - expected) < 0.1


def test_buffer_is_ten_percent():
    """Buffer should be ~10% of other components."""
    report = ghost(param_count_b=1.0, dtype="fp16")
    expected_buffer = 0.1 * (report.weights_gb + report.gradients_gb + report.optimizer_gb + report.activations_gb)
    assert abs(report.buffer_gb - expected_buffer) < 0.1


def test_param_count_input():
    """Should accept raw param_count (not billions)."""
    report = ghost(param_count=1_000_000_000)
    assert report.param_count == 1_000_000_000
    assert report.param_count_b == 1.0


def test_never_crash_none():
    """Ghost should never crash, even with None model."""
    report = ghost(None)
    assert isinstance(report, GhostReport)
    assert report.param_count == 0


def test_never_crash_bad_model():
    """Ghost should never crash with a non-torch object."""
    report = ghost("not a model")
    assert isinstance(report, GhostReport)


def test_to_dict():
    """GhostReport.to_dict() should return a plain dict."""
    report = ghost(param_count_b=1.0)
    d = report.to_dict()
    assert isinstance(d, dict)
    assert "total_gb" in d
    assert "param_count_b" in d


def test_70b_total_vram():
    """70B model should need substantial VRAM."""
    report = ghost(param_count_b=70.0, dtype="fp16")
    assert report.total_gb > 200  # weights + grads + optimizer alone > 200GB
    assert report.weights_gb > 100


def test_small_model():
    """125M model should need modest VRAM."""
    report = ghost(param_count_b=0.125, dtype="fp32")
    assert report.weights_gb < 1  # 125M * 4B ≈ 0.47 GB
    assert report.total_gb < 5
