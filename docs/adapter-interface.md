# Adapter Interface

`turboquant-core` backends implement a minimal interface compatible with
[turboquant-workflow-eval](https://github.com/Flaker420/turboquant-workflow-eval)'s
adapter contract.

## Backend methods

```python
class Backend:
    def is_compressible(self, layer_idx: int) -> bool:
        """Whether this layer has a KV cache that can be compressed."""

    def compress(self, K: Tensor, V: Tensor, layer_idx: int) -> dict:
        """Compress K/V tensors. K,V shape: [batch, heads, seq_len, head_dim].
        Returns dict with keys: k_mse, k_n, v_idx, v_n, shape.
        When key_strategy='mse+qjl', also includes: k_qjl, k_rn."""

    def decompress_v(self, compressed: dict) -> Tensor:
        """Decompress V for attention output computation.
        Returns tensor with same shape as original V."""

    def compute_attention_scores(self, Q: Tensor, compressed: dict) -> Tensor:
        """Compute Q @ K^T (with QJL bias correction when key_strategy='mse+qjl').
        Q shape: [batch, heads, q_len, head_dim].
        Returns: [batch, heads, q_len, kv_len]."""
```

## Configurable constructors

Both backends accept keyword-only layout and strategy params:

```python
Qwen35KVBackend(bit_width=4, seed=42, device="cpu", *,
    num_layers=32, full_attn_interval=4, kv_heads=4, head_dim=256,
    key_strategy="mse+qjl", value_strategy="mse")

Qwen3DenseKVBackend(bit_width=4, seed=42, device="cpu", *,
    num_layers=36, kv_heads=8, head_dim=128,
    key_strategy="mse+qjl", value_strategy="mse")
```

All new params have defaults matching the standard model configs. Class
constants (`NUM_LAYERS`, `GA_HEAD_DIM`, etc.) are preserved for backward
compatibility.

## Algorithm selection

`key_strategy` controls K quantization:

| Strategy | K codebook bits | QJL correction | Use case |
|---|---|---|---|
| `"mse+qjl"` (default) | `bit_width - 1` | Yes (1-bit QJL) | Unbiased softmax(QK^T) |
| `"mse"` | `bit_width` | No | When QJL overhead is not justified |

`value_strategy` controls V quantization (currently only `"mse"` is supported).

Invalid strategies raise `ValueError` eagerly in the constructor.

## K vs V asymmetry

K gets TQ_prod (MSE + QJL) because it participates in the softmax(QK^T) inner product.
QJL corrects the quantization bias in this inner product.

V gets TQ_MSE only because it's weighted-averaged by attention scores — no inner product to debias.

## bit_width parameter semantics

When `bit_width=4` and `key_strategy="mse+qjl"`:
- **K** gets a **(4-1)=3 bit** MSE codebook (8 centroids) + 1-bit QJL = **4 bits total**
- **V** gets a **4-bit** MSE codebook (16 centroids) = **4 bits total**

When `key_strategy="mse"`:
- **K** gets a **4-bit** MSE codebook (16 centroids) = **4 bits total** (no QJL)
- **V** gets a **4-bit** MSE codebook (16 centroids) = **4 bits total**

## TurboQuantAdapter (workflow-eval integration)

The `TurboQuantAdapter` class duck-types the eval harness's `CompressionAdapter`
interface:

```python
class TurboQuantAdapter:
    name = "turboquant"

    def prepare_model(self, model, tokenizer, model_cfg, policy_cfg):
        """Patch model with TQ compressed KV cache."""

    def describe(self, policy_cfg) -> dict:
        """Return adapter metadata."""

    def can_revert(self) -> bool:
        """True if model is currently patched."""

    def revert(self, model) -> bool:
        """Unpatch model, restore original forwards, clear cache."""

    def get_state(self) -> dict:
        """Return {adapter, variant, bit_width, seed, patched, backend}."""

    def reset_generation_state(self) -> None:
        """Clear KV cache between generations (call before each prompt)."""

    def update_params(self, params=None, **kwargs) -> bool:
        """Not supported — raises NotImplementedError. Revert + re-prepare instead."""

    def cleanup(self, model) -> None:
        """Fully unpatch the model and clear cache (equivalent to revert)."""
```

The adapter reads these settings from `policy_cfg["settings"]`:

| Setting | Default | Description |
|---|---|---|
| `bit_width` | `4` | Total bits per value |
| `seed` | `42` | Random seed for rotation/QJL |
| `key_strategy` | `"mse+qjl"` | K quantization strategy |
| `value_strategy` | `"mse"` | V quantization strategy |
| `model_variant` | auto-detect | Explicit variant override |

Layout overrides can be provided either nested under `model_cfg["layout"]`
or at the top level of `model_cfg`. The long-form aliases
`full_attention_interval` (→ `full_attn_interval`) and `total_lm_layers`
(→ `num_layers`) are also accepted:

```python
# Nested form
model_cfg = {
    "name": "Qwen/Qwen3.5-9B",
    "layout": {"num_layers": 32, "full_attn_interval": 4, "kv_heads": 4, "head_dim": 256},
}

# Top-level form with long aliases (equivalent)
model_cfg = {
    "name": "Qwen/Qwen3.5-9B",
    "total_lm_layers": 32,
    "full_attention_interval": 4,
    "kv_heads": 4,
    "head_dim": 256,
}
```

A missing or empty `model_cfg["name"]` (with no `model_variant` override)
raises a clear `ValueError` referencing the `model_name` bridge — useful
when wiring through harness configs that use a different field name.

## Variant registry

Model variants are detected from `model_cfg["name"]` using a registry:

```python
from turboquant_core import register_variant

# Register before calling prepare_model()
register_variant("Qwen4", "qwen4", MyCustomBackend)
```

Built-in variants: `"Qwen3.5"` → `qwen35`, `"Qwen3"` → `qwen3`. Entries
are matched in order (first match wins); `register_variant()` inserts at
the front, so custom variants take priority.

## Model unpatching

```python
from turboquant_core import unpatch_model

count = unpatch_model(model)  # restores original attention forwards
```

Works for both qwen35 (selective) and qwen3 (all layers). Checks all
layers for the `_tq_original_forward` marker attribute.

## Hybrid models

For Qwen3.5-9B, `is_compressible()` returns `False` for DeltaNet layers (0-2, 4-6, etc.)
because their recurrent state is internal to the flash-linear-attention kernel and not
exposed through the `Qwen3_5DynamicCache` API.

## Model hook-in

For direct model integration (beyond the eval adapter), use `patch_qwen35_with_tq()`:

```python
from turboquant_core.backends.qwen_hook import patch_qwen35_with_tq

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", ...)
cache = patch_qwen35_with_tq(model, bit_width=4)
# model.generate() now uses compressed KV cache
# cache.clear() between generations
# unpatch_model(model) to restore originals
```

This monkey-patches the GatedAttn attention layers to compress K/V and use
QJL-corrected attention scores, while passing DeltaNet layers through unchanged.
