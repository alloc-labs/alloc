"""Tests for forward hook activation tracing in extractor_runner."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from alloc.extractor_runner import _measure_activations, _build_dummy_input


class SimpleSequential(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TupleOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 128)

    def forward(self, x):
        out = self.fc(x)
        return out, out * 2


class BrokenForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        raise RuntimeError("deliberate failure")


def test_measure_activations_sequential():
    """Simple nn.Sequential, verify bytes > 0."""
    model = SimpleSequential()
    arch_info = {}
    result = _measure_activations(model, arch_info)
    assert result.get("activation_method") == "traced"
    assert result["activation_memory_bytes"] > 0


def test_measure_activations_conv_model():
    """CNN model, verify correct activation size."""
    model = SimpleCNN()
    arch_info = {}
    result = _measure_activations(model, arch_info)
    assert result.get("activation_method") == "traced"
    assert result["activation_memory_bytes"] > 0


def test_measure_activations_fails_gracefully():
    """Model that can't forward → returns {}."""
    model = BrokenForwardModel()
    arch_info = {}
    result = _measure_activations(model, arch_info)
    assert result == {}


def test_measure_activations_tuple_output():
    """Model returning tuple of tensors still measures activations."""
    model = TupleOutputModel()
    arch_info = {}
    result = _measure_activations(model, arch_info)
    assert result.get("activation_method") == "traced"
    assert result["activation_memory_bytes"] > 0


def test_build_dummy_input_transformer_config():
    """Model with vocab_size → Long tensor."""
    class FakeModel:
        pass
    arch_info = {"vocab_size": 32000, "seq_length": 2048}
    inp = _build_dummy_input(FakeModel(), arch_info)
    assert inp is not None
    assert inp.dtype == torch.long
    assert inp.shape[0] == 1
    assert inp.shape[1] <= 128  # capped at 128


def test_build_dummy_input_vision_config():
    """Model with image_size → Float tensor."""
    class FakeModel:
        pass
    arch_info = {"image_size": 224, "num_channels": 3}
    inp = _build_dummy_input(FakeModel(), arch_info)
    assert inp is not None
    assert inp.shape == (1, 3, 224, 224)


def test_build_dummy_input_weight_inference():
    """Infer from first layer shape (Linear)."""
    model = SimpleSequential()
    arch_info = {}
    inp = _build_dummy_input(model, arch_info)
    assert inp is not None
    assert inp.shape == (1, 64)  # first Linear is (128, 64)


def test_build_dummy_input_conv_weight():
    """4D weight → image input."""
    model = SimpleCNN()
    arch_info = {}
    inp = _build_dummy_input(model, arch_info)
    assert inp is not None
    assert inp.dim() == 4
    assert inp.shape[1] == 3  # in_ch from Conv2d(3, 16, ...)


def test_build_dummy_input_returns_none():
    """No inferable shape → None."""
    class EmptyModel(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return x

    arch_info = {}
    inp = _build_dummy_input(EmptyModel(), arch_info)
    assert inp is None


def test_hooks_cleaned_up_on_failure():
    """Verify no dangling hooks after failure."""
    model = BrokenForwardModel()
    arch_info = {}

    # Get hook count before
    hooks_before = sum(len(m._forward_hooks) for m in model.modules())

    _measure_activations(model, arch_info)

    # Hooks should be cleaned up
    hooks_after = sum(len(m._forward_hooks) for m in model.modules())
    assert hooks_after == hooks_before


def test_count_params_dedup_shared():
    """Shared parameters (same data_ptr) should be counted only once."""
    from alloc.extractor_runner import _count_params

    class TiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 64)
            self.proj = nn.Linear(64, 100, bias=False)
            # Tie weights: same tensor
            self.proj.weight = self.embed.weight

        def forward(self, x):
            return self.proj(self.embed(x))

    model = TiedModel()
    count, dtype_str = _count_params(model)
    # embed.weight = 100*64 = 6400 params (shared with proj.weight)
    assert count == 6400
