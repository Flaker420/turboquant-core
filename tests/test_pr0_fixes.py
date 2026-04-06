"""Tests for PR0 correctness fixes and new features."""

import torch
import torch.nn as nn
import pytest

from turboquant_core.core import TQQuantizedCache
from turboquant_core.backends.qwen import (
    Qwen35KVBackend, Qwen25DenseKVBackend,
)
from turboquant_core.backends.qwen_hook import (
    patch_qwen3_with_tq, patch_qwen25_with_tq,
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
        patch_qwen3_with_tq(model, bit_width=4)

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
        patch_qwen3_with_tq(model, bit_width=4)

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

    def test_update_params_positional_dict_raises(self):
        """update_params is not supported and must raise NotImplementedError."""
        adapter = TurboQuantAdapter()
        with pytest.raises(NotImplementedError):
            adapter.update_params({"bit_width": 8})

    def test_update_params_kwargs_raises(self):
        adapter = TurboQuantAdapter()
        with pytest.raises(NotImplementedError):
            adapter.update_params(bit_width=8)

    def test_update_params_no_args_raises(self):
        adapter = TurboQuantAdapter()
        with pytest.raises(NotImplementedError):
            adapter.update_params()


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


# ---------------------------------------------------------------------------
# Tests: key_strategy threading through TQQuantizedCache
# ---------------------------------------------------------------------------

class TestKeyStrategyCache:
    def test_mse_only_cache_no_qjl(self):
        """MSE-only cache should not create a QJL projection."""
        cache = TQQuantizedCache(
            num_layers=4, interval=1,
            kv_head_dim=128, num_kv_heads=2,
            bit_width=4, seed=42, key_strategy="mse",
        )
        assert cache.key_strategy == "mse"
        assert cache.k_qjl is None

    def test_mse_qjl_cache_has_qjl(self):
        """MSE+QJL cache should have a QJL projection."""
        cache = TQQuantizedCache(
            num_layers=4, interval=1,
            kv_head_dim=128, num_kv_heads=2,
            bit_width=4, seed=42, key_strategy="mse+qjl",
        )
        assert cache.key_strategy == "mse+qjl"
        assert cache.k_qjl is not None

    def test_invalid_key_strategy_raises(self):
        """Invalid key_strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid key_strategy"):
            TQQuantizedCache(
                num_layers=4, interval=1,
                kv_head_dim=128, num_kv_heads=2,
                bit_width=4, key_strategy="invalid",
            )

    def test_mse_only_compress_and_attention(self):
        """MSE-only cache should compress/decompress without QJL fields."""
        cache = TQQuantizedCache(
            num_layers=4, interval=1,
            kv_head_dim=128, num_kv_heads=2,
            bit_width=4, seed=42, key_strategy="mse",
        )
        K = torch.randn(1, 2, 8, 128)
        V = torch.randn(1, 2, 8, 128)
        cache.update(K, V, layer_idx=0)

        entry = cache._cache[0]
        assert "k_qjl" not in entry
        assert "k_rn" not in entry

        Q = torch.randn(1, 2, 1, 128)
        output = cache.compute_attention(Q, layer_idx=0)
        assert output.shape == (1, 2, 1, 128)
        assert not torch.isnan(output).any()

    def test_mse_only_with_residual_window(self):
        """MSE-only + residual window should work together."""
        cache = TQQuantizedCache(
            num_layers=4, interval=1,
            kv_head_dim=128, num_kv_heads=2,
            bit_width=4, seed=42, key_strategy="mse",
            residual_window=4,
        )
        K = torch.randn(1, 2, 8, 128)
        V = torch.randn(1, 2, 8, 128)
        cache.update(K, V, layer_idx=0)

        Q = torch.randn(1, 2, 1, 128)
        output = cache.compute_attention(Q, layer_idx=0)
        assert output.shape == (1, 2, 1, 128)
        assert not torch.isnan(output).any()


class TestKeyStrategyHook:
    def test_patch_qwen3_mse_only(self):
        """Patching with key_strategy='mse' should propagate."""
        model = _make_qwen3_mock()
        cache = patch_qwen3_with_tq(model, bit_width=4, key_strategy="mse")
        assert cache.key_strategy == "mse"
        assert cache.k_qjl is None

    def test_patch_qwen25_mse_only(self):
        """patch_qwen25_with_tq with MSE-only should work end-to-end."""
        model = _make_qwen25_mock()
        cache = patch_qwen25_with_tq(model, bit_width=4, key_strategy="mse")
        assert cache.key_strategy == "mse"
        assert cache.kv_head_dim == 128
        assert cache.num_kv_heads == 2

        # Run a forward pass through layer 0
        bsz, seq_len = 1, 4
        hidden_size = 16 * 128
        hidden_states = torch.randn(bsz, seq_len, hidden_size)
        layer = model.model.layers[0]
        output, _, _ = layer.self_attn(hidden_states)
        assert output.shape == (bsz, seq_len, hidden_size)
        assert not torch.isnan(output).any()

    def test_adapter_key_strategy(self):
        """Adapter should pass key_strategy from settings."""
        adapter = TurboQuantAdapter()
        model = _make_qwen25_mock()
        model_cfg = {"name": "Qwen/Qwen2.5-3B-Instruct"}
        policy_cfg = {"settings": {"bit_width": 4, "key_strategy": "mse"}}
        adapter.prepare_model(model, None, model_cfg, policy_cfg)
        assert adapter._cache.key_strategy == "mse"

    def test_patch_qwen25_defaults(self):
        """patch_qwen25_with_tq should use Qwen2.5-3B defaults."""
        model = _make_qwen25_mock()
        cache = patch_qwen25_with_tq(model, bit_width=4)
        assert cache.num_layers == 36
        assert cache.kv_head_dim == 128
        assert cache.num_kv_heads == 2


# ---------------------------------------------------------------------------
# Tests: GQA + residual window (the bug the reviewer caught)
# ---------------------------------------------------------------------------

class TestGQAResidualWindow:
    def test_gqa_residual_window_prefill(self):
        """GQA model with residual_window should work during prefill."""
        model = _make_qwen3_mock()  # 16 Q heads, 8 KV heads → GQA
        patch_qwen3_with_tq(
            model, bit_width=4, residual_window=4, key_strategy="mse",
        )

        bsz, seq_len = 1, 8
        hidden_size = 16 * 128
        hidden_states = torch.randn(bsz, seq_len, hidden_size)

        layer = model.model.layers[0]
        output, _, _ = layer.self_attn(hidden_states)
        assert output.shape == (bsz, seq_len, hidden_size)
        assert not torch.isnan(output).any()

    def test_gqa_residual_window_overflow(self):
        """GQA with tokens exceeding residual window should compress overflow."""
        model = _make_qwen25_mock()  # 16 Q heads, 2 KV heads → GQA
        cache = patch_qwen25_with_tq(
            model, bit_width=4, residual_window=4, key_strategy="mse",
        )

        bsz = 1
        hidden_size = 16 * 128

        # Prefill with 8 tokens (4 overflow → compressed, 4 in window)
        layer = model.model.layers[0]
        hidden_states = torch.randn(bsz, 8, hidden_size)
        output, _, _ = layer.self_attn(hidden_states)
        assert output.shape == (bsz, 8, hidden_size)
        assert not torch.isnan(output).any()

        # Verify window + compressed split
        assert cache._window_k[0] is not None
        assert cache._window_k[0].shape[2] == 4
        assert cache._cache[0] is not None

    def test_gqa_residual_window_incremental_decode(self):
        """GQA with residual window should handle incremental decode."""
        model = _make_qwen25_mock()
        cache = patch_qwen25_with_tq(
            model, bit_width=4, residual_window=3, key_strategy="mse",
        )

        bsz = 1
        hidden_size = 16 * 128
        layer = model.model.layers[0]

        # Prefill with 4 tokens
        hidden_states = torch.randn(bsz, 4, hidden_size)
        output, _, _ = layer.self_attn(hidden_states)
        assert output.shape == (bsz, 4, hidden_size)

        # Decode 3 more tokens one at a time
        for _ in range(3):
            decode = torch.randn(bsz, 1, hidden_size)
            output, _, _ = layer.self_attn(decode)
            assert output.shape == (bsz, 1, hidden_size)
            assert not torch.isnan(output).any()

        # Total should be 7: 4 compressed + 3 in window
        assert cache.get_seq_length(0) == 7

    def test_gqa_residual_window_qjl(self):
        """GQA + residual window + QJL should also work."""
        model = _make_qwen3_mock()
        patch_qwen3_with_tq(
            model, bit_width=4, residual_window=4, key_strategy="mse+qjl",
        )

        bsz = 1
        hidden_size = 16 * 128
        hidden_states = torch.randn(bsz, 8, hidden_size)

        layer = model.model.layers[0]
        output, _, _ = layer.self_attn(hidden_states)
        assert output.shape == (bsz, 8, hidden_size)
        assert not torch.isnan(output).any()

    def test_gqa_window_only_no_compressed(self):
        """When all tokens fit in window, GQA should use FP16 path."""
        model = _make_qwen25_mock()
        cache = patch_qwen25_with_tq(
            model, bit_width=4, residual_window=16, key_strategy="mse",
        )

        bsz = 1
        hidden_size = 16 * 128
        hidden_states = torch.randn(bsz, 4, hidden_size)

        layer = model.model.layers[0]
        output, _, _ = layer.self_attn(hidden_states)
        assert output.shape == (bsz, 4, hidden_size)
        assert not torch.isnan(output).any()

        # All in window, nothing compressed
        assert cache._cache[0] is None
        assert cache._window_k[0] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
