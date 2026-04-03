# turboquant-core

TurboQuant algorithm library with model-specific KV cache backends.

## What it is

Pure Python/PyTorch implementation of TurboQuant (ICLR 2026): Lloyd-Max codebook quantization with random rotation and QJL residual correction for unbiased inner product estimation.

## Algorithms

- **TQ_MSE** — MSE-optimal quantization via random rotation + Lloyd-Max scalar quantization
- **TQ_prod** — MSE + 1-bit QJL residual for unbiased inner product estimation
- **QJL** — Quantized Johnson-Lindenstrauss projection (sign-bit quantization)

## Model backends

| Backend | Model | KV layers | Strategy |
|---|---|---|---|
| `Qwen35KVBackend` | Qwen3.5-9B | 8 GatedAttn (of 32) | K→TQ_prod, V→TQ_MSE |
| `Qwen3DenseKVBackend` | Qwen3-8B | All 36 | K→TQ_prod, V→TQ_MSE |

```python
from turboquant_core.backends.qwen import Qwen35KVBackend

backend = Qwen35KVBackend(bit_width=4, device="cuda")
if backend.is_compressible(layer_idx=3):
    compressed = backend.compress(K, V, layer_idx=3)
    V_restored = backend.decompress_v(compressed)
```

## Integration with turboquant-workflow-eval

This library provides the real TQ backend for the adapter interface in [turboquant-workflow-eval](https://github.com/Flaker420/turboquant-workflow-eval).

## Reference

[TurboQuant: Online Vector Quantization](https://arxiv.org/abs/2504.19874) (ICLR 2026)

## License

Apache 2.0
