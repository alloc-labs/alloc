"""Tests for Ghost Scan VRAM math."""

from alloc.ghost import ghost, GhostReport, _extract_architecture


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


def test_optimizer_adam_mixed():
    """Adam optimizer should add 12 bytes per param for mixed precision."""
    report = ghost(param_count_b=1.0, dtype="fp16")
    # 1e9 * 12 / 1024^3 ≈ 11.18 GB
    assert 10 < report.optimizer_gb < 13


def test_optimizer_adam_fp32():
    """Adam optimizer should add 8 bytes per param for fp32 (no master copy)."""
    report = ghost(param_count_b=1.0, dtype="fp32")
    # 1e9 * 8 / 1024^3 ≈ 7.45 GB
    assert 6 < report.optimizer_gb < 9


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


def test_70b_total_vram_with_transformer_fields():
    """70B model with explicit transformer fields should need substantial VRAM."""
    report = ghost(param_count_b=70.0, dtype="fp16", batch_size=32, seq_length=2048, hidden_dim=8192)
    assert report.total_gb > 200  # weights + grads + optimizer alone > 200GB
    assert report.weights_gb > 100


def test_small_model_with_transformer_fields():
    """125M model with explicit fields should need modest VRAM."""
    report = ghost(param_count_b=0.125, dtype="fp32", batch_size=8, seq_length=512, hidden_dim=768)
    assert report.weights_gb < 1  # 125M * 4B ≈ 0.47 GB
    assert report.total_gb < 10


def test_num_layers_estimation():
    """Estimated num_layers should be reasonable for known architectures."""
    from alloc.ghost import _TRANSFORMER_PARAM_FACTOR
    # 7B model with hidden_dim=4096 → ~34 layers (LLaMA-7B has 32)
    est = round(7e9 / (_TRANSFORMER_PARAM_FACTOR * 4096 * 4096))
    assert 28 <= est <= 40

    # 70B model with hidden_dim=8192 → ~87 layers (LLaMA-70B has 80)
    est = round(70e9 / (_TRANSFORMER_PARAM_FACTOR * 8192 * 8192))
    assert 70 <= est <= 100


def test_activations_scale_with_layers():
    """Larger models should have proportionally more activation memory."""
    small = ghost(param_count_b=1.0, dtype="fp16", batch_size=32, seq_length=2048, hidden_dim=2048)
    large = ghost(param_count_b=7.0, dtype="fp16", batch_size=32, seq_length=2048, hidden_dim=4096)
    # 7B model should have much more activation memory than 1B
    assert large.activations_gb > small.activations_gb * 2


def test_gradients_fp32_for_mixed_precision():
    """Mixed precision gradients should be fp32 (4 bytes), not fp16 (2 bytes)."""
    report = ghost(param_count_b=1.0, dtype="fp16")
    # 1e9 * 4 / 1024^3 ≈ 3.73 GB (fp32 gradients)
    assert 3.0 < report.gradients_gb < 5.0
    # Should be 2x the weight size (fp16 weights = ~1.86 GB, fp32 grads = ~3.73 GB)
    assert report.gradients_gb > report.weights_gb * 1.5


def test_gradients_same_as_weights_for_fp32():
    """For fp32 training, gradients should be same precision as weights."""
    report = ghost(param_count_b=1.0, dtype="fp32")
    # Both should be fp32: ~3.73 GB each
    assert abs(report.gradients_gb - report.weights_gb) < 0.1


# ---------------------------------------------------------------------------
# Architecture-agnostic tests (new)
# ---------------------------------------------------------------------------


def test_none_defaults_uses_generic_heuristic():
    """ghost() with no architecture fields should use generic activation heuristic."""
    report = ghost(param_count_b=7.0)
    assert isinstance(report, GhostReport)
    # Activations should be ≈ weights (1x heuristic)
    # 7B fp16 weights ≈ 13 GB, so activations ≈ 13 GB
    assert 10 < report.activations_gb < 16
    # num_layers should be None (generic path doesn't estimate layers)
    assert report.num_layers is None
    # batch_size, seq_length, hidden_dim should be None
    assert report.batch_size is None
    assert report.seq_length is None
    assert report.hidden_dim is None


def test_explicit_transformer_fields_match_previous_behavior():
    """Explicit transformer fields should produce transformer-specific activations."""
    report = ghost(param_count_b=7.0, batch_size=32, seq_length=2048, hidden_dim=4096)
    # Should use transformer formula, so num_layers should be set
    assert report.num_layers is not None
    assert report.num_layers > 0
    assert report.batch_size == 32
    assert report.seq_length == 2048
    assert report.hidden_dim == 4096


def test_non_transformer_resnet_scale():
    """ResNet-50 scale (26M params) should produce reasonable VRAM without transformer fields."""
    report = ghost(param_count_b=0.026, dtype="fp32")
    # 26M * 4B / 1024^3 ≈ 0.097 GB weights
    assert report.weights_gb < 0.2
    # Total should be reasonable (weights + grads + optimizer + activations + buffer)
    assert report.total_gb < 2
    # num_layers should be None
    assert report.num_layers is None


def test_model_config_extraction():
    """Model with .config should extract hidden_dim and num_layers."""

    class FakeConfig:
        hidden_size = 768
        num_hidden_layers = 12
        max_position_embeddings = 512

    class FakeModel:
        config = FakeConfig()

    arch = _extract_architecture(FakeModel())
    assert arch["hidden_dim"] == 768
    assert arch["num_layers"] == 12
    assert arch["seq_length"] == 512


def test_model_config_extraction_gpt2_style():
    """GPT-2 style config uses n_layer and n_positions."""

    class FakeConfig:
        hidden_size = None
        num_hidden_layers = None
        n_layer = 24
        n_positions = 1024
        max_position_embeddings = None

    class FakeModel:
        config = FakeConfig()

    arch = _extract_architecture(FakeModel())
    assert arch["num_layers"] == 24
    assert arch["seq_length"] == 1024


def test_model_without_config():
    """Model without .config should return all None."""

    class PlainModel:
        pass

    arch = _extract_architecture(PlainModel())
    assert arch["hidden_dim"] is None
    assert arch["num_layers"] is None
    assert arch["seq_length"] is None


def test_partial_transformer_fields():
    """Providing only hidden_dim but not batch_size/seq_length should use generic path."""
    report = ghost(param_count_b=7.0, hidden_dim=4096)
    # Missing batch_size and seq_length → generic path
    assert report.num_layers is None
    # Activations should be ≈ weights
    assert 10 < report.activations_gb < 16
