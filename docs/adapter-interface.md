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
        """Compress K/V tensors. K,V shape: [batch, heads, seq_len, head_dim]."""

    def decompress_v(self, compressed: dict) -> Tensor:
        """Decompress V for attention output computation."""
```

## K vs V asymmetry

K gets TQ_prod (MSE + QJL) because it participates in the softmax(QK^T) inner product.
QJL corrects the quantization bias in this inner product.

V gets TQ_MSE only because it's weighted-averaged by attention scores — no inner product to debias.

## Hybrid models

For Qwen3.5-9B, `is_compressible()` returns `False` for DeltaNet layers (0-2, 4-6, etc.)
because their recurrent state is internal to the flash-linear-attention kernel and not
exposed through the `Qwen3_5DynamicCache` API.
