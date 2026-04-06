"""Tests for the TurboQuantAdapter (workflow-eval integration)."""

import torch.nn as nn
import pytest

from turboquant_core.adapters.workflow_eval import (
    TurboQuantAdapter, _detect_variant, register_variant, _VARIANT_REGISTRY,
)
from turboquant_core.backends.qwen import Qwen35KVBackend, Qwen3DenseKVBackend
from turboquant_core.core import TQQuantizedCache


# ---------------------------------------------------------------------------
# Mock model components (mirrors test_hooks.py)
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
    return MockModel(
        num_layers=32, hidden_size=8 * 256,
        num_heads=8, num_kv_heads=4, head_dim=256,
    )


def _make_qwen3_mock():
    return MockModel(
        num_layers=36, hidden_size=16 * 128,
        num_heads=16, num_kv_heads=8, head_dim=128,
    )


# ---------------------------------------------------------------------------
# Tests: _detect_variant
# ---------------------------------------------------------------------------

class TestDetectVariant:
    def test_qwen35_from_name(self):
        vid, cls = _detect_variant({"name": "Qwen/Qwen3.5-9B"}, {})
        assert vid == "qwen35"
        assert cls is Qwen35KVBackend

    def test_qwen3_from_name(self):
        vid, cls = _detect_variant({"name": "Qwen/Qwen3-8B"}, {})
        assert vid == "qwen3"
        assert cls is Qwen3DenseKVBackend

    def test_explicit_override(self):
        vid, cls = _detect_variant(
            {"name": "Qwen/Qwen3.5-9B"}, {"model_variant": "qwen3"}
        )
        assert vid == "qwen3"
        assert cls is Qwen3DenseKVBackend

    def test_explicit_override_unknown_variant(self):
        vid, cls = _detect_variant({"name": "X"}, {"model_variant": "custom"})
        assert vid == "custom"
        assert cls is None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot detect model variant"):
            _detect_variant({"name": "GPT-4"}, {})

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="model_cfg\\['name'\\] is missing"):
            _detect_variant({}, {})

    def test_missing_name_with_model_name_key_hints(self):
        with pytest.raises(ValueError, match="model_name"):
            _detect_variant({"model_name": "Qwen/Qwen3.5-9B"}, {})


# ---------------------------------------------------------------------------
# Tests: register_variant (Phase 6)
# ---------------------------------------------------------------------------

class TestRegisterVariant:
    def test_register_and_detect(self):
        # Register a custom variant
        class FakeBackend:
            pass

        original_len = len(_VARIANT_REGISTRY)
        register_variant("Qwen4", "qwen4", FakeBackend)
        try:
            vid, cls = _detect_variant({"name": "Qwen4-16B"}, {})
            assert vid == "qwen4"
            assert cls is FakeBackend
        finally:
            # Clean up the registry
            _VARIANT_REGISTRY.pop(0)
            assert len(_VARIANT_REGISTRY) == original_len

    def test_register_takes_priority(self):
        """Newly registered variants are checked first (inserted at index 0)."""
        class OverrideBackend:
            pass

        register_variant("Qwen3", "qwen3_override", OverrideBackend)
        try:
            vid, cls = _detect_variant({"name": "Qwen3-8B"}, {})
            assert vid == "qwen3_override"
            assert cls is OverrideBackend
        finally:
            _VARIANT_REGISTRY.pop(0)


# ---------------------------------------------------------------------------
# Tests: TurboQuantAdapter
# ---------------------------------------------------------------------------

class TestTurboQuantAdapter:
    def test_name(self):
        adapter = TurboQuantAdapter()
        assert adapter.name == "turboquant"

    def test_import_path(self):
        from turboquant_core.adapters.workflow_eval import TurboQuantAdapter as A
        from turboquant_core.adapters import TurboQuantAdapter as B
        from turboquant_core import TurboQuantAdapter as C
        assert A is B is C

    def test_prepare_model_qwen35(self):
        adapter = TurboQuantAdapter()
        model = _make_qwen35_mock()
        model_cfg = {"name": "Qwen/Qwen3.5-9B"}
        policy_cfg = {"settings": {"bit_width": 4}}

        ret_model, ret_tok = adapter.prepare_model(model, None, model_cfg, policy_cfg)

        assert ret_model is model
        assert ret_tok is None
        assert adapter._cache is not None
        assert isinstance(adapter._cache, TQQuantizedCache)

    def test_prepare_model_qwen3(self):
        adapter = TurboQuantAdapter()
        model = _make_qwen3_mock()
        model_cfg = {"name": "Qwen/Qwen3-8B"}
        policy_cfg = {"settings": {"bit_width": 4}}

        ret_model, ret_tok = adapter.prepare_model(model, None, model_cfg, policy_cfg)

        assert ret_model is model
        assert adapter._cache is not None
        assert isinstance(adapter._cache, TQQuantizedCache)

    def test_prepare_model_unknown_raises(self):
        adapter = TurboQuantAdapter()
        model = _make_qwen35_mock()
        with pytest.raises(ValueError):
            adapter.prepare_model(model, None, {"name": "Unknown"}, {"settings": {}})

    def test_prepare_model_with_layout_override(self):
        """Phase 2: layout overrides from model_cfg are passed through."""
        adapter = TurboQuantAdapter()
        # Use a smaller mock matching the layout
        model = MockModel(
            num_layers=16, hidden_size=8 * 256,
            num_heads=8, num_kv_heads=4, head_dim=256,
        )
        model_cfg = {
            "name": "Qwen/Qwen3.5-9B",
            "layout": {"num_layers": 16, "full_attn_interval": 4},
        }
        policy_cfg = {"settings": {"bit_width": 4}}

        adapter.prepare_model(model, None, model_cfg, policy_cfg)
        assert adapter._cache.num_layers == 16

    def test_describe(self):
        adapter = TurboQuantAdapter()
        policy_cfg = {
            "settings": {
                "bit_width": 4, "seed": 123,
                "residual_window": 8, "key_strategy": "mse",
            },
        }
        desc = adapter.describe(policy_cfg)
        assert desc == {
            "adapter": "turboquant",
            "bit_width": 4,
            "seed": 123,
            "residual_window": 8,
            "key_strategy": "mse",
            "value_strategy": "mse",
        }

    def test_describe_defaults(self):
        adapter = TurboQuantAdapter()
        desc = adapter.describe({})
        assert desc == {
            "adapter": "turboquant",
            "bit_width": 4,
            "seed": 42,
            "residual_window": 0,
            "key_strategy": "mse+qjl",
            "value_strategy": "mse",
        }

    def test_cleanup(self):
        adapter = TurboQuantAdapter()
        model = _make_qwen35_mock()
        adapter.prepare_model(model, None, {"name": "Qwen/Qwen3.5-9B"}, {"settings": {}})
        assert adapter._cache is not None

        adapter.cleanup(model)
        assert adapter._cache is None

    def test_cleanup_noop_without_prepare(self):
        adapter = TurboQuantAdapter()
        adapter.cleanup(None)  # should not raise
        assert adapter._cache is None

    # -- can_revert / revert ---------------------------------------------------

    def test_can_revert_false_before_prepare(self):
        adapter = TurboQuantAdapter()
        assert adapter.can_revert() is False

    def test_can_revert_true_after_prepare(self):
        adapter = TurboQuantAdapter()
        model = _make_qwen35_mock()
        adapter.prepare_model(model, None, {"name": "Qwen/Qwen3.5-9B"}, {"settings": {}})
        assert adapter.can_revert() is True

    def test_revert_restores_model(self):
        adapter = TurboQuantAdapter()
        model = _make_qwen35_mock()

        adapter.prepare_model(model, None, {"name": "Qwen/Qwen3.5-9B"}, {"settings": {}})
        assert adapter.can_revert() is True
        assert hasattr(model.model.layers[3].self_attn.forward, '__wrapped__')

        result = adapter.revert(model)
        assert result is True
        assert adapter.can_revert() is False
        assert adapter._cache is None
        assert not hasattr(model.model.layers[3].self_attn.forward, '__wrapped__')

    def test_revert_without_prepare_returns_false(self):
        adapter = TurboQuantAdapter()
        model = _make_qwen35_mock()
        assert adapter.revert(model) is False

    def test_double_revert_returns_false(self):
        adapter = TurboQuantAdapter()
        model = _make_qwen35_mock()
        adapter.prepare_model(model, None, {"name": "Qwen/Qwen3.5-9B"}, {"settings": {}})
        assert adapter.revert(model) is True
        assert adapter.revert(model) is False

    # -- get_state (Phase 3) ---------------------------------------------------

    def test_get_state_before_prepare(self):
        adapter = TurboQuantAdapter()
        state = adapter.get_state()
        assert state == {
            "adapter": "turboquant",
            "variant": None,
            "bit_width": None,
            "seed": None,
            "patched": False,
            "backend": None,
        }

    def test_get_state_after_prepare(self):
        adapter = TurboQuantAdapter()
        model = _make_qwen3_mock()
        adapter.prepare_model(
            model, None,
            {"name": "Qwen/Qwen3-8B"},
            {"settings": {"bit_width": 2, "seed": 99}},
        )
        state = adapter.get_state()
        assert state == {
            "adapter": "turboquant",
            "variant": "qwen3",
            "bit_width": 2,
            "seed": 99,
            "patched": True,
            "backend": "Qwen3DenseKVBackend",
        }

    def test_get_state_after_revert(self):
        adapter = TurboQuantAdapter()
        model = _make_qwen3_mock()
        adapter.prepare_model(model, None, {"name": "Qwen/Qwen3-8B"}, {"settings": {}})
        adapter.revert(model)
        state = adapter.get_state()
        assert state["patched"] is False
        assert state["variant"] == "qwen3"

    def test_get_state_backend_qwen35(self):
        adapter = TurboQuantAdapter()
        model = _make_qwen35_mock()
        adapter.prepare_model(model, None, {"name": "Qwen/Qwen3.5-9B"}, {"settings": {}})
        assert adapter.get_state()["backend"] == "Qwen35KVBackend"

    # -- update_params ---------------------------------------------------------

    def test_update_params_raises(self):
        adapter = TurboQuantAdapter()
        with pytest.raises(NotImplementedError):
            adapter.update_params()

    def test_cleanup_unpatches_model(self):
        adapter = TurboQuantAdapter()
        model = _make_qwen35_mock()
        adapter.prepare_model(model, None, {"name": "Qwen/Qwen3.5-9B"}, {"settings": {}})
        assert hasattr(model.model.layers[3].self_attn.forward, '__wrapped__')
        adapter.cleanup(model)
        assert not hasattr(model.model.layers[3].self_attn.forward, '__wrapped__')
        assert adapter._cache is None
        assert adapter._patched is False
