"""Tests for qwen_hook.py: model patching for TurboQuant KV cache compression."""

import torch
import torch.nn as nn
import pytest

from turboquant_core.backends.qwen_hook import (
    patch_qwen35_with_tq,
    patch_qwen3_with_tq,
    unpatch_model,
    _get_model_layers,
    _get_attention_module,
    _apply_rotary_pos_emb,
)


# ---------------------------------------------------------------------------
# Mock model components
# ---------------------------------------------------------------------------

class MockLinear(nn.Module):
    """Simple linear layer for testing (no bias, identity-like)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


class MockAttention(nn.Module):
    """Minimal attention module matching HF Qwen attribute names."""
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        return hidden_states, None, None


class MockLayer(nn.Module):
    """Transformer layer with self_attn attribute."""
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.self_attn = MockAttention(hidden_size, num_heads, num_kv_heads, head_dim)


class MockModel(nn.Module):
    """Mock model with model.layers structure (HF Qwen style)."""
    def __init__(self, num_layers, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockLayer(hidden_size, num_heads, num_kv_heads, head_dim)
            for _ in range(num_layers)
        ])


def _make_qwen35_mock():
    """Create mock Qwen3.5-9B structure: 32 layers, GQA with 4 KV heads, 256 head_dim.

    Uses real head_dim (256) for correct TQ codebook construction but reduces
    Q head count to 8 (2 groups per KV head) to save memory.
    """
    return MockModel(
        num_layers=32, hidden_size=8 * 256,
        num_heads=8, num_kv_heads=4, head_dim=256,
    )


def _make_qwen3_mock():
    """Create mock Qwen3-8B structure: 36 layers, GQA with 8 KV heads, 128 head_dim.

    Uses real head_dim (128) for correct TQ codebook construction but reduces
    Q head count to 16 (2 groups per KV head) to save memory.
    """
    return MockModel(
        num_layers=36, hidden_size=16 * 128,
        num_heads=16, num_kv_heads=8, head_dim=128,
    )


# ---------------------------------------------------------------------------
# Tests: _get_model_layers
# ---------------------------------------------------------------------------

class TestGetModelLayers:
    def test_model_dot_layers(self):
        model = _make_qwen35_mock()
        layers = _get_model_layers(model)
        assert layers is not None
        assert len(layers) == 32

    def test_transformer_dot_h(self):
        """Models using transformer.h attribute path (GPT-2 style)."""
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.h = nn.ModuleList([nn.Linear(10, 10) for _ in range(4)])
        layers = _get_model_layers(model)
        assert layers is not None
        assert len(layers) == 4

    def test_unknown_structure_returns_none(self):
        model = nn.Module()
        model.encoder = nn.Linear(10, 10)
        assert _get_model_layers(model) is None


# ---------------------------------------------------------------------------
# Tests: _get_attention_module
# ---------------------------------------------------------------------------

class TestGetAttentionModule:
    def test_self_attn(self):
        layer = MockLayer(256, 4, 4, 64)
        attn = _get_attention_module(layer)
        assert attn is not None
        assert isinstance(attn, MockAttention)

    def test_attn_attribute(self):
        layer = nn.Module()
        layer.attn = nn.Linear(10, 10)
        attn = _get_attention_module(layer)
        assert attn is not None

    def test_no_attention_returns_none(self):
        layer = nn.Module()
        layer.mlp = nn.Linear(10, 10)
        assert _get_attention_module(layer) is None


# ---------------------------------------------------------------------------
# Tests: patch_qwen35_with_tq
# ---------------------------------------------------------------------------

class TestPatchQwen35:
    def test_returns_cache(self):
        model = _make_qwen35_mock()
        cache = patch_qwen35_with_tq(model, bit_width=4)
        assert cache is not None
        assert cache.num_layers == 32

    def test_compressible_layers(self):
        model = _make_qwen35_mock()
        cache = patch_qwen35_with_tq(model, bit_width=4)
        # GatedAttn at interval=4: indices {3,7,11,15,19,23,27,31}
        for i in range(32):
            expected = (i + 1) % 4 == 0
            assert cache.is_compressible(i) == expected, f"Layer {i}"

    def test_patches_only_compressible_layers(self):
        model = _make_qwen35_mock()
        cache = patch_qwen35_with_tq(model, bit_width=4)

        for i, layer in enumerate(model.model.layers):
            forward = layer.self_attn.forward
            if cache.is_compressible(i):
                # Patched layers have a plain function (via functools.wraps),
                # not the original bound method
                assert hasattr(forward, '__wrapped__'), \
                    f"Layer {i} should be patched"
            else:
                # Unpatched layers still have the original nn.Module.forward
                assert not hasattr(forward, '__wrapped__'), \
                    f"Layer {i} should NOT be patched"

    def test_raises_on_bad_model(self):
        model = nn.Module()
        model.register_parameter("dummy", nn.Parameter(torch.zeros(1)))
        with pytest.raises(ValueError, match="Could not find transformer layers"):
            patch_qwen35_with_tq(model)


# ---------------------------------------------------------------------------
# Tests: patch_qwen3_with_tq
# ---------------------------------------------------------------------------

class TestPatchQwen3:
    def test_returns_cache(self):
        model = _make_qwen3_mock()
        cache = patch_qwen3_with_tq(model, bit_width=4)
        assert cache is not None
        assert cache.num_layers == 36

    def test_all_layers_compressible(self):
        model = _make_qwen3_mock()
        cache = patch_qwen3_with_tq(model, bit_width=4)
        for i in range(36):
            assert cache.is_compressible(i), f"Layer {i} should be compressible"

    def test_all_layers_patched(self):
        model = _make_qwen3_mock()
        patch_qwen3_with_tq(model, bit_width=4)

        for i, layer in enumerate(model.model.layers):
            assert hasattr(layer.self_attn.forward, '__wrapped__'), \
                f"Layer {i} should be patched"

    def test_raises_on_bad_model(self):
        model = nn.Module()
        model.register_parameter("dummy", nn.Parameter(torch.zeros(1)))
        with pytest.raises(ValueError, match="Could not find transformer layers"):
            patch_qwen3_with_tq(model)

    def test_cache_dimensions(self):
        model = _make_qwen3_mock()
        cache = patch_qwen3_with_tq(model, bit_width=4)
        assert cache.kv_head_dim == 128
        assert cache.num_kv_heads == 8


# ---------------------------------------------------------------------------
# Tests: _apply_rotary_pos_emb
# ---------------------------------------------------------------------------

class TestRotaryPosEmb:
    def test_output_shape(self):
        bsz, heads, seq_len, head_dim = 1, 4, 16, 64
        q = torch.randn(bsz, heads, seq_len, head_dim)
        k = torch.randn(bsz, heads, seq_len, head_dim)
        cos = torch.randn(1, 1, seq_len, head_dim)
        sin = torch.randn(1, 1, seq_len, head_dim)
        q_out, k_out = _apply_rotary_pos_emb(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_no_nan(self):
        q = torch.randn(1, 2, 8, 32)
        k = torch.randn(1, 2, 8, 32)
        cos = torch.ones(1, 1, 8, 32)
        sin = torch.zeros(1, 1, 8, 32)
        q_out, k_out = _apply_rotary_pos_emb(q, k, cos, sin)
        assert not torch.isnan(q_out).any()
        assert not torch.isnan(k_out).any()


# ---------------------------------------------------------------------------
# Tests: Patched forward produces correct output shapes
# ---------------------------------------------------------------------------

class TestPatchedForwardShapes:
    def test_qwen35_patched_forward_output(self):
        """Verify patched attention forward produces [batch, seq, hidden] output."""
        model = _make_qwen35_mock()
        patch_qwen35_with_tq(model, bit_width=4)

        bsz, seq_len = 1, 8
        hidden_size = 8 * 256  # num_q_heads * head_dim (mock uses 8 Q heads)
        hidden_states = torch.randn(bsz, seq_len, hidden_size)

        # Test a compressible layer (index 3)
        layer = model.model.layers[3]
        output, _, _ = layer.self_attn(hidden_states)
        assert output.shape == (bsz, seq_len, hidden_size)

    def test_qwen3_patched_forward_output(self):
        """Verify patched attention forward produces [batch, seq, hidden] output."""
        model = _make_qwen3_mock()
        patch_qwen3_with_tq(model, bit_width=4)

        bsz, seq_len = 1, 8
        hidden_size = 16 * 128  # mock uses 16 Q heads
        hidden_states = torch.randn(bsz, seq_len, hidden_size)

        # All layers compressible; test layer 0
        layer = model.model.layers[0]
        output, _, _ = layer.self_attn(hidden_states)
        assert output.shape == (bsz, seq_len, hidden_size)


# ---------------------------------------------------------------------------
# Tests: GQA attention with TQ cache
# ---------------------------------------------------------------------------

class TestGQAWithCache:
    def test_qwen35_gqa_attention_shape(self):
        """Mock Qwen3.5: 8 Q heads, 4 KV heads → groups of 2."""
        model = _make_qwen35_mock()
        patch_qwen35_with_tq(model, bit_width=4)

        bsz, seq_len = 1, 4
        hidden_size = 8 * 256
        hidden_states = torch.randn(bsz, seq_len, hidden_size)

        # Run through compressible layer 3
        layer = model.model.layers[3]
        output, _, _ = layer.self_attn(hidden_states)
        assert output.shape == (bsz, seq_len, hidden_size)
        assert not torch.isnan(output).any()

    def test_qwen3_gqa_attention_shape(self):
        """Mock Qwen3-8B: 16 Q heads, 8 KV heads → groups of 2."""
        model = _make_qwen3_mock()
        patch_qwen3_with_tq(model, bit_width=4)

        bsz, seq_len = 1, 4
        hidden_size = 16 * 128
        hidden_states = torch.randn(bsz, seq_len, hidden_size)

        layer = model.model.layers[0]
        output, _, _ = layer.self_attn(hidden_states)
        assert output.shape == (bsz, seq_len, hidden_size)
        assert not torch.isnan(output).any()


# ---------------------------------------------------------------------------
# Tests: unpatch_model
# ---------------------------------------------------------------------------

class TestUnpatchModel:
    def test_unpatch_qwen35_restores_forward(self):
        model = _make_qwen35_mock()
        cache = patch_qwen35_with_tq(model, bit_width=4)

        # Verify compressible layers have the marker
        for i, layer in enumerate(model.model.layers):
            if cache.is_compressible(i):
                assert hasattr(layer.self_attn, '_tq_original_forward')

        count = unpatch_model(model)
        assert count == 8  # 8 GatedAttn layers

        # Verify all layers restored: no markers, no __wrapped__ (patched fn)
        for i, layer in enumerate(model.model.layers):
            assert not hasattr(layer.self_attn, '_tq_original_forward')
            assert not hasattr(layer.self_attn.forward, '__wrapped__'), \
                f"Layer {i} should be unpatched"

    def test_unpatch_qwen3_restores_all(self):
        model = _make_qwen3_mock()
        patch_qwen3_with_tq(model, bit_width=4)
        count = unpatch_model(model)
        assert count == 36

        for layer in model.model.layers:
            assert not hasattr(layer.self_attn, '_tq_original_forward')

    def test_unpatch_unpatched_model_returns_zero(self):
        model = _make_qwen35_mock()
        assert unpatch_model(model) == 0

    def test_unpatch_bad_model_returns_zero(self):
        model = nn.Module()
        model.encoder = nn.Linear(10, 10)
        assert unpatch_model(model) == 0

    def test_double_unpatch_is_safe(self):
        model = _make_qwen3_mock()
        patch_qwen3_with_tq(model, bit_width=4)
        assert unpatch_model(model) == 36
        assert unpatch_model(model) == 0  # second call is a no-op


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
