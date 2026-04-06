"""Tests for _extract_layout schema reconciliation (Section A1, A2)."""
from pathlib import Path

import yaml

from turboquant_core.adapters.workflow_eval import _extract_layout


class TestExtractLayout:
    def test_nested_short_names(self):
        cfg = {"layout": {"num_layers": 32, "full_attn_interval": 4,
                          "kv_heads": 4, "head_dim": 256}}
        out = _extract_layout(cfg, "qwen35")
        assert out == {"num_layers": 32, "full_attn_interval": 4,
                       "kv_heads": 4, "head_dim": 256}

    def test_nested_long_name_alias(self):
        cfg = {"layout": {"full_attention_interval": 4}}
        out = _extract_layout(cfg, "qwen35")
        assert out == {"full_attn_interval": 4}

    def test_total_lm_layers_alias(self):
        cfg = {"layout": {"total_lm_layers": 36}}
        out = _extract_layout(cfg, "qwen3")
        assert out == {"num_layers": 36}

    def test_top_level_keys(self):
        cfg = {"name": "Qwen/Qwen3.5-9B", "num_layers": 32,
               "full_attention_interval": 4, "kv_heads": 4, "head_dim": 256}
        out = _extract_layout(cfg, "qwen35")
        assert out == {"num_layers": 32, "full_attn_interval": 4,
                       "kv_heads": 4, "head_dim": 256}

    def test_nested_overrides_top_level(self):
        cfg = {"num_layers": 99, "layout": {"num_layers": 32}}
        out = _extract_layout(cfg, "qwen35")
        assert out["num_layers"] == 32

    def test_unknown_nested_key_warns(self, caplog):
        cfg = {"layout": {"num_layers": 32, "bogus_key": 7}}
        with caplog.at_level("WARNING"):
            out = _extract_layout(cfg, "qwen35")
        assert "bogus_key" in caplog.text
        assert "bogus_key" not in out

    def test_qwen3_ignores_full_attn_interval(self):
        cfg = {"layout": {"full_attn_interval": 4, "num_layers": 36}}
        out = _extract_layout(cfg, "qwen3")
        assert out == {"num_layers": 36}

    def test_bundled_qwen35_yaml_loads(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "models" / "qwen35_9b.yaml"
        cfg = yaml.safe_load(path.read_text())
        out = _extract_layout(cfg, "qwen35")
        assert out == {
            "num_layers": 32,
            "full_attn_interval": 4,
            "kv_heads": 4,
            "head_dim": 256,
        }
        assert cfg["name"] == "Qwen/Qwen3.5-9B"
