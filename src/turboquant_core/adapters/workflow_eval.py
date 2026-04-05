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

from ..backends.qwen import Qwen35KVBackend, Qwen3DenseKVBackend
from ..backends.qwen_hook import patch_qwen35_with_tq, patch_qwen3_with_tq, unpatch_model


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

_VARIANT_REGISTRY: list[tuple[str, str, type]] = [
    ("Qwen3.5", "qwen35", Qwen35KVBackend),
    ("Qwen3", "qwen3", Qwen3DenseKVBackend),
]


def register_variant(pattern: str, variant_id: str, backend_cls: type):
    """Register a new model variant for auto-detection.

    Entries are matched in order (first match wins), so register more
    specific patterns (e.g. "Qwen3.5") before general ones (e.g. "Qwen3").

    Args:
        pattern: Substring to match against model_cfg["name"].
        variant_id: Short identifier returned by _detect_variant().
        backend_cls: Backend class associated with this variant.
    """
    _VARIANT_REGISTRY.insert(0, (pattern, variant_id, backend_cls))


class TurboQuantAdapter:
    """CompressionAdapter-compatible class for turboquant-workflow-eval.

    Duck-types the CompressionAdapter interface (no import dependency on
    the workflow-eval package).
    """

    name = "turboquant"

    def __init__(self):
        self._cache = None
        self._variant = None
        self._bit_width = None
        self._seed = None
        self._patched = False
        self._backend_name = None

    def prepare_model(self, model, tokenizer, model_cfg: dict, policy_cfg: dict):
        settings = policy_cfg.get("settings", {})
        bit_width = settings.get("bit_width", 4)
        seed = settings.get("seed", 42)
        key_strategy = settings.get("key_strategy", "mse+qjl")
        value_strategy = settings.get("value_strategy", "mse")

        self._bit_width = bit_width
        self._seed = seed

        variant, backend_cls = _detect_variant(model_cfg, settings)
        self._variant = variant
        self._backend_name = backend_cls.__name__ if backend_cls else None

        # Build layout kwargs from model_cfg if provided
        layout = _extract_layout(model_cfg, variant)

        if variant == "qwen35":
            self._cache = patch_qwen35_with_tq(
                model, bit_width=bit_width, seed=seed, **layout,
            )
        elif variant == "qwen3":
            self._cache = patch_qwen3_with_tq(
                model, bit_width=bit_width, seed=seed, **layout,
            )
        else:
            raise ValueError(f"Unsupported model variant: {variant!r}")

        self._patched = True
        return model, tokenizer

    def describe(self, policy_cfg: dict) -> dict:
        settings = policy_cfg.get("settings", {})
        return {
            "adapter": self.name,
            "bit_width": settings.get("bit_width", 4),
            "seed": settings.get("seed", 42),
            "scope": settings.get("scope", "full_attention_only"),
        }

    def can_revert(self) -> bool:
        """Return True if the model is currently patched and can be reverted."""
        return self._patched

    def revert(self, model) -> bool:
        """Unpatch the model, restoring original attention forward methods.

        Args:
            model: The patched model to revert.

        Returns:
            True if the model was successfully unpatched, False if not patched.
        """
        if not self._patched:
            return False
        unpatch_model(model)
        if self._cache is not None:
            self._cache.clear()
            self._cache = None
        self._patched = False
        return True

    def get_state(self) -> dict:
        """Return current adapter state for inspection."""
        return {
            "adapter": self.name,
            "variant": self._variant,
            "bit_width": self._bit_width,
            "seed": self._seed,
            "patched": self._patched,
            "backend": self._backend_name,
        }

    def update_params(self, **kwargs) -> bool:
        """Update compression parameters on a live model.

        Not yet supported — requires revert + re-prepare.

        Returns:
            False always.
        """
        return False

    def cleanup(self, model) -> None:
        if self._cache is not None:
            self._cache.clear()
            self._cache = None
        self._patched = False


def _detect_variant(model_cfg: dict, settings: dict) -> tuple[str, type | None]:
    """Detect model variant from config, returning (variant_id, backend_cls).

    Resolution order:
    1. Explicit ``model_variant`` in settings (registry lookup for backend_cls).
    2. Substring match against ``model_cfg["name"]`` using the variant registry.
    """
    # Explicit override
    if "model_variant" in settings:
        vid = settings["model_variant"]
        for _, variant_id, backend_cls in _VARIANT_REGISTRY:
            if variant_id == vid:
                return vid, backend_cls
        return vid, None

    name = model_cfg.get("name", "")
    for pattern, variant_id, backend_cls in _VARIANT_REGISTRY:
        if pattern in name or pattern.lower() in name.lower():
            return variant_id, backend_cls

    raise ValueError(
        f"Cannot detect model variant from model_cfg name {name!r}. "
        "Set 'model_variant' in policy settings or use register_variant()."
    )


def _extract_layout(model_cfg: dict, variant: str) -> dict:
    """Extract layout overrides from model_cfg, if present."""
    layout = model_cfg.get("layout", {})
    kwargs = {}
    if variant == "qwen35":
        for key in ("num_layers", "full_attn_interval", "kv_heads", "head_dim"):
            if key in layout:
                kwargs[key] = layout[key]
    elif variant == "qwen3":
        for key in ("num_layers", "kv_heads", "head_dim"):
            if key in layout:
                kwargs[key] = layout[key]
    return kwargs
