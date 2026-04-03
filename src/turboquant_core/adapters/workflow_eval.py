"""
Adapter for turboquant-workflow-eval integration.

Provides a CompressionAdapter-compatible class that bridges the workflow-eval
framework to turboquant-core's model patching functions.

Usage in a workflow-eval policy YAML:

    adapter:
      import_path: "turboquant_core.adapters.workflow_eval:TurboQuantAdapter"
      settings:
        bit_width: 4
        seed: 42
"""

from __future__ import annotations

from ..backends.qwen_hook import patch_qwen35_with_tq, patch_qwen3_with_tq


class TurboQuantAdapter:
    """CompressionAdapter-compatible class for turboquant-workflow-eval.

    Duck-types the CompressionAdapter interface (no import dependency on
    the workflow-eval package).
    """

    name = "turboquant"

    def __init__(self):
        self._cache = None

    def prepare_model(self, model, tokenizer, model_cfg: dict, policy_cfg: dict):
        settings = policy_cfg.get("settings", {})
        bit_width = settings.get("bit_width", 4)
        seed = settings.get("seed", 42)

        variant = _detect_variant(model_cfg, settings)

        if variant == "qwen35":
            self._cache = patch_qwen35_with_tq(model, bit_width=bit_width, seed=seed)
        elif variant == "qwen3":
            self._cache = patch_qwen3_with_tq(model, bit_width=bit_width, seed=seed)
        else:
            raise ValueError(f"Unsupported model variant: {variant!r}")

        return model, tokenizer

    def describe(self, policy_cfg: dict) -> dict:
        settings = policy_cfg.get("settings", {})
        return {
            "adapter": self.name,
            "bit_width": settings.get("bit_width", 4),
            "seed": settings.get("seed", 42),
            "scope": settings.get("scope", "full_attention_only"),
        }

    def cleanup(self, model) -> None:
        if self._cache is not None:
            self._cache.clear()
            self._cache = None


def _detect_variant(model_cfg: dict, settings: dict) -> str:
    # Explicit override takes precedence
    if "model_variant" in settings:
        return settings["model_variant"]

    name = model_cfg.get("name", "")
    # Check "3.5" before "3" to avoid substring false match
    if "3.5" in name:
        return "qwen35"
    if "Qwen3" in name or "qwen3" in name.lower():
        return "qwen3"

    raise ValueError(
        f"Cannot detect model variant from model_cfg name {name!r}. "
        "Set 'model_variant' in policy settings."
    )
