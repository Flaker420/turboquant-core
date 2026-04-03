# Adapter Interface

`turboquant-core` backends implement a minimal interface compatible with
[turboquant-workflow-eval](https://github.com/Flaker420/turboquant-workflow-eval)'s
adapter contract.

## Required methods

```python
class Backend:
    def is_compressible(self, layer_idx: int) -> bool:
        """Whether this layer has a KV cache that can be compressed."""

    def compress(self, K: Tensor, V: Tensor, layer_idx: int) -> dict:
        """Compress K/V tensors. K,V shape: [batch, heads, seq_len, head_dim].
        Returns dict with keys: k_mse, k_qjl, k_rn, k_n, v_idx, v_n, shape."""

    def decompress_v(self, compressed: dict) -> Tensor:
        """Decompress V for attention output computation.
        Returns tensor with same shape as original V."""

    def compute_attention_scores(self, Q: Tensor, compressed: dict) -> Tensor:
        """Compute unbiased Q @ K^T using QJL bias correction.
        Q shape: [batch, heads, q_len, head_dim].
        Returns: [batch, heads, q_len, kv_len]."""
```

## K vs V asymmetry

K gets TQ_prod (MSE + QJL) because it participates in the softmax(QK^T) inner product.
QJL corrects the quantization bias in this inner product.

V gets TQ_MSE only because it's weighted-averaged by attention scores — no inner product to debias.

## bit_width parameter semantics

When `bit_width=4` is passed to the backend:
- **K** gets a **(4-1)=3 bit** MSE codebook (8 centroids) + 1-bit QJL = **4 bits total**
- **V** gets a **4-bit** MSE codebook (16 centroids) = **4 bits total**

The same `bit_width` parameter produces different codebook sizes for K and V.
This is correct — K needs 1 bit reserved for the QJL residual correction.

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
```

This monkey-patches the GatedAttn attention layers to compress K/V and use
QJL-corrected attention scores, while passing DeltaNet layers through unchanged.
