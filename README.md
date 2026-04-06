# turboquant-core

TurboQuant algorithm library with model-specific KV cache backends.

## What it is

Pure Python/PyTorch implementation of TurboQuant (ICLR 2026): Lloyd-Max codebook quantization with random rotation and optional QJL residual correction for KV cache compression.

## Algorithms

- **TQ_MSE** — MSE-optimal quantization via random rotation + Lloyd-Max scalar quantization
- **TQ_prod** — MSE + 1-bit QJL residual for unbiased inner product estimation
- **QJL** — Quantized Johnson-Lindenstrauss projection (sign-bit quantization)
- **STE** — Straight-through estimator for differentiable quantization

## Model backends

| Backend | Model | KV layers | Default K/V strategy |
|---|---|---|---|
| `Qwen25DenseKVBackend` | Qwen2.5-3B-Instruct | All 36 | Configurable: `mse` or `mse+qjl` / `mse` |
| `Qwen3DenseKVBackend` | Qwen3-8B | All 36 | Configurable: `mse` or `mse+qjl` / `mse` |
| `Qwen35KVBackend` | Qwen3.5-9B | 8 GatedAttn (of 32) | Configurable: `mse` or `mse+qjl` / `mse` |

All backends accept `key_strategy="mse"` (MSE-only, community-recommended for softmax attention) or `key_strategy="mse+qjl"` (original paper approach with QJL correction).

```python
backend = Qwen25DenseKVBackend(
    bit_width=4, seed=42, device="cuda",
    key_strategy="mse",       # MSE-only (no QJL) — recommended default
    value_strategy="mse",
)
```

### Model hook-in (drop-in KV cache replacement)

```python
from transformers import AutoModelForCausalLM
from turboquant_core import patch_qwen25_with_tq

# Qwen2.5-3B-Instruct (dense: all 36 layers, 2 KV heads, recommended ablation target)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", ...)
cache = patch_qwen25_with_tq(
    model, bit_width=4,
    key_strategy="mse",       # or "mse+qjl"
    residual_window=128,      # keep recent 128 tokens in FP16
)
# model.generate() now uses compressed KV cache with causal masking.
cache.clear()  # call between generations
```

Other models:

```python
from turboquant_core import patch_qwen3_with_tq, patch_qwen35_with_tq

# Qwen3-8B (dense: all 36 layers)
cache = patch_qwen3_with_tq(model, bit_width=4, key_strategy="mse", residual_window=128)

# Qwen3.5-9B (hybrid: 8 GatedAttn layers compressed, DeltaNet unchanged)
cache = patch_qwen35_with_tq(model, bit_width=4, key_strategy="mse", residual_window=128)
```

### Key configuration options

| Parameter | Values | Description |
|---|---|---|
| `bit_width` | 2-8 | Bits per quantized value |
| `key_strategy` | `"mse"`, `"mse+qjl"` | MSE-only (recommended) or MSE + QJL correction |
| `residual_window` | 0-256 | Recent tokens kept in FP16 (0 = compress all) |

### Compress / decompress (low-level)

```python
from turboquant_core.backends.qwen import Qwen25DenseKVBackend

backend = Qwen25DenseKVBackend(bit_width=4, key_strategy="mse")
compressed = backend.compress(K, V, layer_idx=0)
V_restored = backend.decompress_v(compressed)
attn_scores = backend.compute_attention_scores(Q, compressed)
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

# Call between generations to prevent cross-prompt cache contamination:
adapter.reset_generation_state()

adapter.can_revert()          # True if model is patched
adapter.revert(model)         # unpatch + clear cache
adapter.cleanup(model)        # equivalent to revert(model) — fully unpatches
adapter.get_state()           # {adapter, variant, bit_width, seed, patched, backend}
adapter.describe(policy_cfg)  # {adapter, bit_width, seed, residual_window, key_strategy}
```

`update_params()` is **not supported** and raises `NotImplementedError`. To
change compression parameters, call `revert(model)` followed by a fresh
`prepare_model(...)`.

Policy YAML settings:

```yaml
adapter:
  import_path: "turboquant_core.adapters.workflow_eval:TurboQuantAdapter"
  settings:
    bit_width: 4
    seed: 42
    key_strategy: "mse"          # or "mse+qjl"
    residual_window: 128         # recent tokens in FP16
```

Both `key_strategy` and `residual_window` are surfaced by `describe()` so
the eval harness can record them per row.

See `docs/HANDOFF.md` for full wiring instructions.

## Install

```bash
pip install -e .
# With dev tools: pip install -e ".[dev]"
```

### Supported versions

| Dependency | Range |
|---|---|
| Python | `>=3.10` |
| `torch` | `>=2.5,<3` |
| `transformers` | `>=4.45,<4.60` |

The upper bound on `transformers` exists because the Qwen attention hooks
in `backends/qwen_hook.py` reach into internal KV-cache APIs that have
historically broken across major releases.

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
