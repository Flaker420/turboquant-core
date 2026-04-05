"""Tests for PR0 correctness fixes and new features."""

import torch
import torch.nn as nn
import pytest

from turboquant_core.core import TQQuantizedCache
from turboquant_core.backends.qwen import (
    Qwen35KVBackend, Qwen3DenseKVBackend, Qwen25DenseKVBackend,
)
from turboquant_core.backends.qwen_hook import (
    patch_qwen35_with_tq, patch_qwen3_with_tq, unpatch_model,
)
from turboquant_core.adapters.workflow_eval import (
    TurboQuantAdapter, _detect_variant,
)


# ---------------------------------------------------------------------------
# Mock model components (shared with test_hooks.py)
# ---------------------------------------------------------------------------

class MockAttention(nn.Module):
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
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.self_attn = MockAttention(hidden_size, num_heads, num_kv_heads, head_dim)


class MockModel(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockLayer(hidden_size, num_heads, num_kv_heads, head_dim)
            for _ in range(num_layers)
        ])


def _make_qwen35_mock():
    return MockModel(32, 8 * 256, 8, 4, 256)

def _make_qwen3_mock():
    return MockModel(36, 16 * 128, 16, 8, 128)

def _make_qwen25_mock():
    """Mock Qwen2.5-3B: 36 layers, GQA with 2 KV heads, 128 head_dim, 16 Q heads."""
    return MockModel(36, 16 * 128, 16, 2, 128)


# ---------------------------------------------------------------------------
# Tests: Causal masking in patched attention
# ---------------------------------------------------------------------------

class TestCausalMask:
    def test_prefill_is_causal(self):
        """Multi-token prefill must not attend to future positions."""
        model = _make_qwen3_mock()
        cache = patch_qwen3_with_tq(model, bit_width=4)

        bsz, seq_len = 1, 8
        hidden_size = 16 * 128
        hidden_states = torch.randn(bsz, seq_len, hidden_size)

        # Run prefill through layer 0
        layer = model.model.layers[0]
        output, _, _ = layer.self_attn(hidden_states)

        # Output should be valid (no NaN from causal mask)
        assert not torch.isnan(output).any()
        assert output.shape == (bsz, seq_len, hidden_size)

    def test_causal_mask_prevents_future_attention(self):
        """Verify that position i cannot attend to position j > i.

        We test this by checking that two sequences with different future
        tokens produce identical outputs for earlier positions.
        """
        model = _make_qwen3_mock()
        cache = patch_qwen3_with_tq(model, bit_width=4)

        bsz = 1
        hidden_size = 16 * 128
        seq_len = 4

        # Create two inputs: same first 2 tokens, different last 2
        torch.manual_seed(42)
        base = torch.randn(bsz, 2, hidden_size)
        suffix_a = torch.randn(bsz, 2, hidden_size)
        suffix_b = torch.randn(bsz, 2, hidden_size)

        input_a = torch.cat([base, suffix_a], dim=1)
        input_b = torch.cat([base, suffix_b], dim=1)

        layer = model.model.layers[0]

        # Run input_a
        cache.clear()
        out_a, _, _ = layer.self_attn(input_a)

        # Run input_b
        cache.clear()
        out_b, _, _ = layer.self_attn(input_b)

        # First token output should be identical (it can only attend to itself)
        assert torch.allclose(out_a[:, 0, :], out_b[:, 0, :], atol=1e-5), \
            "Position 0 should not be affected by future tokens"

    def test_incremental_decode_after_prefill(self):
        """Verify that incremental single-token decode works after prefill."""
        model = _make_qwen3_mock()
        cache = patch_qwen3_with_tq(model, bit_width=4)

        bsz = 1
        hidden_size = 16 * 128

        # Prefill with 4 tokens
        prefill = torch.randn(bsz, 4, hidden_size)
        layer = model.model.layers[0]
        out_prefill, _, _ = layer.self_attn(prefill)
        assert out_prefill.shape == (bsz, 4, hidden_size)

        # Decode single token
        decode = torch.randn(bsz, 1, hidden_size)
        out_decode, _, _ = layer.self_attn(decode)
        assert out_decode.shape == (bsz, 1, hidden_size)
        assert not torch.isnan(out_decode).any()


# ---------------------------------------------------------------------------
# Tests: Residual windowing
# ---------------------------------------------------------------------------

class TestResidualWindow:
    def test_window_only_no_compression(self):
        """When total tokens <= window size, nothing should be compressed."""
        cache = TQQuantizedCache(
            num_layers=4, interval=1,
            kv_head_dim=128, num_kv_heads=2,
            bit_width=4, seed=42, residual_window=16,
        )
        K = torch.randn(1, 2, 8, 128)
        V = torch.randn(1, 2, 8, 128)
        cache.update(K, V, layer_idx=0)

        # All tokens should be in the window (no compressed cache)
        assert cache._cache[0] is None
        assert cache._window_k[0] is not None
        assert cache._window_k[0].shape[2] == 8
        assert cache.get_seq_length(0) == 8

    def test_overflow_triggers_compression(self):
        """Tokens exceeding window size should be compressed."""
        cache = TQQuantizedCache(
            num_layers=4, interval=1,
            kv_head_dim=128, num_kv_heads=2,
            bit_width=4, seed=42, residual_window=4,
        )
        # Add 8 tokens — 4 should overflow into compressed, 4 stay in window
        K = torch.randn(1, 2, 8, 128)
        V = torch.randn(1, 2, 8, 128)
        cache.update(K, V, layer_idx=0)

        assert cache._cache[0] is not None  # compressed portion exists
        assert cache._window_k[0].shape[2] == 4  # window has 4 tokens
        assert cache.get_seq_length(0) == 8  # total = compressed + window

    def test_window_compute_attention(self):
        """Attention should work with windowed cache."""
        cache = TQQuantizedCache(
            num_layers=4, interval=1,
            kv_head_dim=128, num_kv_heads=2,
            bit_width=4, seed=42, residual_window=4,
        )
        K = torch.randn(1, 2, 8, 128)
        V = torch.randn(1, 2, 8, 128)
        cache.update(K, V, layer_idx=0)

        Q = torch.randn(1, 2, 1, 128)
        output = cache.compute_attention(Q, layer_idx=0)
        assert output.shape == (1, 2, 1, 128)
        assert not torch.isnan(output).any()

    def test_window_zero_is_original_behavior(self):
        """residual_window=0 should behave identically to the original code."""
        cache_rw0 = TQQuantizedCache(
            num_layers=4, interval=1,
            kv_head_dim=128, num_kv_heads=2,
            bit_width=4, seed=42, residual_window=0,
        )
        K = torch.randn(1, 2, 8, 128)
        V = torch.randn(1, 2, 8, 128)
        cache_rw0.update(K, V, layer_idx=0)

        assert cache_rw0._cache[0] is not None
        assert cache_rw0._window_k[0] is None
        assert cache_rw0.get_seq_length(0) == 8

    def test_clear_resets_window(self):
        """cache.clear() should reset both compressed and window state."""
        cache = TQQuantizedCache(
            num_layers=4, interval=1,
            kv_head_dim=128, num_kv_heads=2,
            bit_width=4, seed=42, residual_window=4,
        )
        K = torch.randn(1, 2, 8, 128)
        V = torch.randn(1, 2, 8, 128)
        cache.update(K, V, layer_idx=0)
        cache.clear()

        assert cache._cache[0] is None
        assert cache._window_k[0] is None
        assert cache._window_v[0] is None
        assert cache.get_seq_length(0) == 0

    def test_incremental_window_growth(self):
        """Adding tokens one at a time should correctly fill and overflow window."""
        cache = TQQuantizedCache(
            num_layers=4, interval=1,
            kv_head_dim=128, num_kv_heads=2,
            bit_width=4, seed=42, residual_window=3,
        )
        # Add tokens one at a time
        for i in range(5):
            K = torch.randn(1, 2, 1, 128)
            V = torch.randn(1, 2, 1, 128)
            cache.update(K, V, layer_idx=0)

        assert cache.get_seq_length(0) == 5
        # Window should have exactly 3 tokens
        assert cache._window_k[0].shape[2] == 3
        # Compressed should have 2 tokens
        assert cache._compressed_seq_len(0) == 2


# ---------------------------------------------------------------------------
# Tests: Adapter fixes
# ---------------------------------------------------------------------------

class TestAdapterFixes:
    def test_reset_generation_state(self):
        """reset_generation_state clears the KV cache."""
        adapter = TurboQuantAdapter()
        model = _make_qwen3_mock()
        adapter.prepare_model(model, None, {"name": "Qwen/Qwen3-8B"}, {"settings": {}})

        # Simulate populating cache
        K = torch.randn(1, 8, 4, 128)
        V = torch.randn(1, 8, 4, 128)
        adapter._cache.update(K, V, layer_idx=0)
        assert adapter._cache.get_seq_length(0) == 4

        # Reset should clear
        adapter.reset_generation_state()
        assert adapter._cache.get_seq_length(0) == 0

    def test_reset_generation_state_before_prepare(self):
        """reset_generation_state should not raise if called before prepare."""
        adapter = TurboQuantAdapter()
        adapter.reset_generation_state()  # should not raise

    def test_update_params_positional_dict(self):
        """update_params should accept a positional dict argument."""
        adapter = TurboQuantAdapter()
        # This is how workflow-eval wrappers call it
        result = adapter.update_params({"bit_width": 8})
        assert result is False

    def test_update_params_kwargs(self):
        """update_params should also accept kwargs (original signature)."""
        adapter = TurboQuantAdapter()
        result = adapter.update_params(bit_width=8)
        assert result is False

    def test_update_params_no_args(self):
        """update_params should work with no arguments."""
        adapter = TurboQuantAdapter()
        result = adapter.update_params()
        assert result is False


# ---------------------------------------------------------------------------
# Tests: Qwen2.5-3B backend
# ---------------------------------------------------------------------------

class TestQwen25Backend:
    def test_defaults(self):
        backend = Qwen25DenseKVBackend()
        assert backend.num_layers == 36
        assert backend.kv_heads == 2
        assert backend.head_dim == 128

    def test_all_layers_compressible(self):
        backend = Qwen25DenseKVBackend()
        for i in range(36):
            assert backend.is_compressible(i)

    def test_compress_decompress_v(self):
        backend = Qwen25DenseKVBackend(bit_width=4)
        K = torch.randn(1, 2, 8, 128)
        V = torch.randn(1, 2, 8, 128)
        compressed = backend.compress(K, V, layer_idx=0)
        V_hat = backend.decompress_v(compressed)
        assert V_hat.shape == V.shape

    def test_compress_mse_only(self):
        backend = Qwen25DenseKVBackend(bit_width=4, key_strategy="mse")
        K = torch.randn(1, 2, 8, 128)
        V = torch.randn(1, 2, 8, 128)
        compressed = backend.compress(K, V, layer_idx=0)
        assert "k_qjl" not in compressed

    def test_attention_scores_shape(self):
        backend = Qwen25DenseKVBackend(bit_width=4)
        K = torch.randn(1, 2, 8, 128)
        V = torch.randn(1, 2, 8, 128)
        compressed = backend.compress(K, V, layer_idx=0)
        Q = torch.randn(1, 2, 1, 128)
        scores = backend.compute_attention_scores(Q, compressed)
        assert scores.shape == (1, 2, 1, 8)


class TestQwen25VariantDetection:
    def test_detect_qwen25_from_name(self):
        vid, cls = _detect_variant({"name": "Qwen/Qwen2.5-3B-Instruct"}, {})
        assert vid == "qwen25"
        assert cls is Qwen25DenseKVBackend

    def test_detect_qwen25_explicit(self):
        vid, cls = _detect_variant({"name": "any"}, {"model_variant": "qwen25"})
        assert vid == "qwen25"
        assert cls is Qwen25DenseKVBackend

    def test_adapter_prepare_qwen25(self):
        adapter = TurboQuantAdapter()
        model = _make_qwen25_mock()
        model_cfg = {"name": "Qwen/Qwen2.5-3B-Instruct"}
        policy_cfg = {"settings": {"bit_width": 4}}
        ret_model, ret_tok = adapter.prepare_model(model, None, model_cfg, policy_cfg)
        assert ret_model is model
        assert adapter._cache is not None
        assert adapter.get_state()["variant"] == "qwen25"


# ---------------------------------------------------------------------------
# Tests: Duplicated V quantization fix (Qwen35KVBackend)
# ---------------------------------------------------------------------------

class TestQwen35CompressFix:
    def test_compress_qjl_no_duplicate_v_quantization(self):
        """Verify V is quantized exactly once (bug fix: was called twice)."""
        backend = Qwen35KVBackend(bit_width=4, num_layers=4, full_attn_interval=1)
        K = torch.randn(1, 4, 8, 256)
        V = torch.randn(1, 4, 8, 256)

        compressed = backend.compress(K, V, layer_idx=0)

        # Decompress V and check it's close to original
        V_hat = backend.decompress_v(compressed)
        # Should be reasonable quality (not a completely different quantization)
        cosine_sim = torch.nn.functional.cosine_similarity(
            V.reshape(-1, 256), V_hat.reshape(-1, 256), dim=-1
        ).mean()
        assert cosine_sim > 0.8, f"V reconstruction too poor: {cosine_sim}"


# ---------------------------------------------------------------------------
# Tests: Residual window through hook
# ---------------------------------------------------------------------------

class TestResidualWindowHook:
    def test_patch_with_residual_window(self):
        """Patching with residual_window should propagate to cache."""
        model = _make_qwen3_mock()
        cache = patch_qwen3_with_tq(model, bit_width=4, residual_window=128)
        assert cache.residual_window == 128

    def test_adapter_with_residual_window(self):
        """Adapter should pass residual_window from settings."""
        adapter = TurboQuantAdapter()
        model = _make_qwen3_mock()
        model_cfg = {"name": "Qwen/Qwen3-8B"}
        policy_cfg = {"settings": {"bit_width": 4, "residual_window": 64}}
        adapter.prepare_model(model, None, model_cfg, policy_cfg)
        assert adapter._cache.residual_window == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
