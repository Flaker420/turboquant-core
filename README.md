# turboquant-core

TurboQuant algorithm library with model-specific KV cache backends.

## What it is

Pure Python/PyTorch implementation of TurboQuant (ICLR 2026): Lloyd-Max codebook quantization with random rotation and QJL residual correction for unbiased inner product estimation.

## Algorithms

- **TQ_MSE** — MSE-optimal quantization via random rotation + Lloyd-Max scalar quantization
- **TQ_prod** — MSE + 1-bit QJL residual for unbiased inner product estimation
- **QJL** — Quantized Johnson-Lindenstrauss projection (sign-bit quantization)
- **STE** — Straight-through estimator for differentiable quantization

## Model backends

| Backend | Model | KV layers | Strategy |
|---|---|---|---|
| `Qwen35KVBackend` | Qwen3.5-9B | 8 GatedAttn (of 32) | K→TQ_prod, V→TQ_MSE |
| `Qwen3DenseKVBackend` | Qwen3-8B | All 36 | K→TQ_prod, V→TQ_MSE |

Backend constructors accept configurable layout params with keyword-only args:

```python
# Custom model dimensions (defaults match standard model configs)
backend = Qwen35KVBackend(
    bit_width=4, seed=42, device="cuda",
    num_layers=32, full_attn_interval=4, kv_heads=4, head_dim=256,
    key_strategy="mse+qjl",   # or "mse" for MSE-only (no QJL correction)
    value_strategy="mse",
)
```

### Compress / decompress

```python
from turboquant_core.backends.qwen import Qwen35KVBackend

backend = Qwen35KVBackend(bit_width=4, device="cuda")
if backend.is_compressible(layer_idx=3):
    compressed = backend.compress(K, V, layer_idx=3)
    V_restored = backend.decompress_v(compressed)
    attn_scores = backend.compute_attention_scores(Q, compressed)
```

### Model hook-in (drop-in KV cache replacement)

```python
from transformers import AutoModelForCausalLM
from turboquant_core import patch_qwen35_with_tq, patch_qwen3_with_tq

# Qwen3.5-9B (hybrid: 8 GatedAttn layers compressed, DeltaNet unchanged)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", ...)
cache = patch_qwen35_with_tq(model, bit_width=4)
# model.generate() now uses compressed KV cache
cache.clear()  # call between generations

# Qwen3-8B (dense: all 36 layers compressed)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", ...)
cache = patch_qwen3_with_tq(model, bit_width=4)
```

Patch functions also accept configurable layout overrides:

```python
cache = patch_qwen35_with_tq(model, bit_width=4,
    num_layers=32, full_attn_interval=4, kv_heads=4, head_dim=256)
```

### Unpatching (model revert)

```python
from turboquant_core import unpatch_model

count = unpatch_model(model)  # restores original forwards, returns count
```

## Codebook registry

```python
from turboquant_core import CodebookRegistry

cb = CodebookRegistry.precompute(256, 4)       # compute + cache + return
cached = CodebookRegistry.list_cached()         # [(256, 4), ...]
CodebookRegistry.clear()                        # drop all cached codebooks
```

## Variant registry

Register custom model variants for auto-detection by the adapter:

```python
from turboquant_core import register_variant

class MyBackend:
    ...

register_variant("Qwen4", "qwen4", MyBackend)
```

## Integration with turboquant-workflow-eval

The `TurboQuantAdapter` bridges `turboquant-core` to the eval harness:

```python
from turboquant_core import TurboQuantAdapter

adapter = TurboQuantAdapter()
adapter.prepare_model(model, tokenizer, model_cfg, policy_cfg)

adapter.can_revert()          # True if model is patched
adapter.revert(model)         # unpatch + clear cache
adapter.get_state()           # {"adapter", "variant", "bit_width", "seed", "patched", "backend"}
adapter.update_params()       # False (not yet supported)
```

Policy YAML settings:

```yaml
adapter:
  import_path: "turboquant_core.adapters.workflow_eval:TurboQuantAdapter"
  settings:
    bit_width: 4
    seed: 42
    key_strategy: "mse+qjl"   # or "mse"
    value_strategy: "mse"
```

See `docs/HANDOFF.md` for full wiring instructions.

## Install

```bash
pip install -e .
# With dev tools: pip install -e ".[dev]"
```

## Tests

```bash
pytest tests/ -v
```

## Benchmarks

```bash
python benchmarks/benchmark_kv_cache.py
```

## Reference

[TurboQuant: Online Vector Quantization](https://arxiv.org/abs/2504.19874) (ICLR 2026)

## License

Apache 2.0
