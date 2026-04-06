# Handoff: turboquant-core → turboquant-workflow-eval Integration

## Context

`turboquant-workflow-eval` is an existing evaluation harness for testing KV cache compression policies on Qwen3.5-9B. It currently includes a baseline pass-through adapter and a local Transformers-side patch adapter that quantizes K/V projection outputs on the 8 full-attention layers as a behavioral proxy.

`turboquant-core` is a new standalone library implementing the actual TurboQuant algorithms from the ICLR 2026 paper (arxiv:2504.19874). It provides the math (codebook computation, random rotation, Lloyd-Max quantization, QJL residual correction) and model-specific backends that expose a `compress`/`decompress` interface.

This document covers what `turboquant-core` implements, how to wire it into the eval harness, and what the reviewer should verify.

---

## What turboquant-core implements

### Core algorithms (`src/turboquant_core/core.py`)

- **CodebookRegistry** — Self-contained Lloyd-Max codebook solver using scipy. Precomputes and caches codebooks for (dimension, bit_width) pairs with O(log n) quantization via `torch.searchsorted`. Public API: `get()`, `precompute()`, `list_cached()`, `clear()`.
- **RotationCache** — Fast Walsh-Hadamard rotation (O(d log d), no matrix materialized). Randomized with seeded sign vectors.
- **tq_quantize_mse / tq_dequantize_mse** — TQ Algorithm 1: MSE-optimal quantization via random rotation + Lloyd-Max scalar quantization.
- **QJLProjection** — Quantized Johnson-Lindenstrauss: 1-bit sign quantization of `S @ x` for unbiased inner product estimation.
- **tq_quantize_prod** — TQ Algorithm 2: (b-1)-bit MSE + 1-bit QJL residual for unbiased inner products. Total effective bit-width is b.
- **TQQuantizedCache** — Drop-in compressed KV cache with incremental token-by-token updates. Implements `compute_attention()` with QJL-corrected scores for compressed layers, standard attention for raw layers.
- **tq_quantize_mse_ste** — Straight-through estimator for differentiable quantization (gradient pass-through).
- **TQActivationCheckpoint** / **TQLoRAStorage** — Compression utilities for activation checkpointing and LoRA weight storage.

### Model backends (`src/turboquant_core/backends/`)

| Backend | Model | Layers | KV Heads | Head Dim | Strategy |
|---|---|---|---|---|---|
| `Qwen35KVBackend` | Qwen3.5-9B | 8 of 32 (GatedAttn) | 4 | 256 | K→TQ_prod, V→TQ_MSE |
| `Qwen3DenseKVBackend` | Qwen3-8B | All 36 | 8 | 128 | K→TQ_prod, V→TQ_MSE |
| `Qwen25DenseKVBackend` | Qwen2.5-3B-Instruct | All 36 | 2 | 128 | K→TQ_prod, V→TQ_MSE |

Both backends implement: `is_compressible(layer_idx)`, `compress(K, V, layer_idx)`, `decompress_v(compressed)`, `compute_attention_scores(Q, compressed)`.

**Configurable constructors**: Both accept keyword-only layout params (`num_layers`, `kv_heads`, `head_dim`, etc.) and algorithm strategy params (`key_strategy`, `value_strategy`) with defaults matching the standard model configs.

### Model hook-in (`src/turboquant_core/backends/qwen_hook.py`)

- **`patch_qwen35_with_tq(model, bit_width=4)`** — Monkey-patches Qwen3.5-9B attention to use TQ compressed KV cache. Handles GQA expansion (24 Q heads / 4 KV heads). Accepts layout overrides.
- **`patch_qwen3_with_tq(model, bit_width=4)`** — Same for Qwen3-8B (all 36 layers, 32 Q heads / 8 KV heads). Accepts layout overrides.
- **`patch_qwen25_with_tq(model, bit_width=4)`** — Same for Qwen2.5-3B-Instruct (all 36 layers, 16 Q heads / 2 KV heads, head_dim 128). Accepts layout overrides and `key_strategy` / `value_strategy`. See `README.md` for the canonical usage example.
- **`unpatch_model(model)`** — Restores original attention forward methods on all TQ-patched layers.

### Adapter (`src/turboquant_core/adapters/workflow_eval.py`)

- **`TurboQuantAdapter`** — Duck-types the eval harness's `CompressionAdapter` interface. Supports `prepare_model()`, `can_revert()`, `revert()`, `get_state()`, `reset_generation_state()`, `describe()`, `cleanup()` (fully unpatches). `update_params()` is intentionally unsupported and raises `NotImplementedError` — callers must `revert` + `prepare_model` again.
- **`register_variant()`** — Register custom model variants for auto-detection.

---

## How to wire into turboquant-workflow-eval

### Step 1: Install turboquant-core

```bash
pip install -e ../turboquant-core
# Or: pip install git+https://github.com/Flaker420/turboquant-core.git
```

### Step 2: Use the built-in adapter

The eval harness already supports `TurboQuantAdapter` via import path. No custom adapter file needed:

```yaml
adapter:
  import_path: "turboquant_core.adapters.workflow_eval:TurboQuantAdapter"
  settings:
    bit_width: 4
    seed: 42
    key_strategy: "mse+qjl"    # optional, default
    value_strategy: "mse"       # optional, default; only "mse" is currently supported
```

Both `key_strategy` and `value_strategy` are forwarded by `TurboQuantAdapter.prepare_model()` into the underlying `patch_qwen{25,3,35}_with_tq()` call and recorded in `describe()` for run provenance. `value_strategy` is currently restricted to `"mse"`; passing any other value raises `ValueError` from `TQQuantizedCache`.

The eval harness's `TurboQuantAdapter` bridge delegates `can_revert()`, `revert()`, `get_state()`, `reset_generation_state()`, `describe()`, and `cleanup()` directly to core. `update_params()` raises `NotImplementedError` by design.

### Step 3 (alternative): Direct backend adapter

For reconstruction-quality experiments (compress → immediately decompress):

```python
from turboquant_core.backends.qwen import Qwen35KVBackend

class TurboQuantRealAdapter:
    """Adapter wrapping the real TQ backend for the eval harness.

    Compresses K/V with TQ, then immediately dequantizes and passes
    the reconstructed tensors forward. Measures reconstruction quality,
    not memory savings.
    """

    def __init__(self, bit_width=4, device="cuda"):
        self.backend = Qwen35KVBackend(bit_width=bit_width, device=device)

    def should_apply(self, layer_idx: int) -> bool:
        return self.backend.is_compressible(layer_idx)

    def process_kv(self, K, V, layer_idx):
        if not self.should_apply(layer_idx):
            return K, V

        compressed = self.backend.compress(K, V, layer_idx)
        V_reconstructed = self.backend.decompress_v(compressed)

        from turboquant_core.core import tq_dequantize_mse
        K_reconstructed = tq_dequantize_mse(
            compressed["k_mse"], compressed["k_n"],
            self.backend.k_cb, self.backend.k_rot
        ).reshape(K.shape)

        return K_reconstructed, V_reconstructed
```

Note: this adapter dequantizes K using only the MSE component, not the full TQ_prod reconstruction with QJL correction. This makes the K reconstruction *worse* than what a real TQ_prod backend would achieve — eval results are a conservative lower bound.

### Step 4: Run the comparison

```bash
make study \
  POLICY_CONFIGS=configs/policies/baseline.yaml,configs/policies/safe_template.yaml,configs/policies/tq_real_4bit.yaml \
  OUTPUT_DIR=outputs/study_tq_real
```

---

## What the reviewer should verify

### Correctness checks

1. **Codebook centroids match the paper.** For b=1: 2 centroids at ~±0.7979. For b=2: 4 centroids at ~±0.4528, ±1.5104. Run `pytest tests/test_core.py -k paper`.

2. **Rotation preserves norms.** `‖rotate(x)‖ = ‖x‖` for any x (orthogonal transform).

3. **MSE round-trip error.** At b=4, d=256: MSE/coord ≈ 0.0115 (paper Table 2, within 10%).

4. **K vs V asymmetry.** K uses `tq_quantize_prod` (MSE + QJL), V uses `tq_quantize_mse` (MSE only). K participates in QK^T inner product (where bias matters), V does not.

5. **Layer filtering.** `Qwen35KVBackend.is_compressible(i)` returns True for i ∈ {3,7,11,15,19,23,27,31} only. `Qwen3DenseKVBackend.is_compressible(i)` returns True for all layers.

### Integration checks

6. **Tensor shapes survive round-trip.** Feed K/V with shape [1, 4, 128, 256] through compress → decompress_v. Output V shape must match input.

7. **No NaN/Inf.** Compress/decompress on random and zero inputs. The `+ 1e-12` guards prevent division by zero.

8. **Adapter interface compatibility.** The methods (`is_compressible`, `compress`, `decompress_v`, `compute_attention_scores`) match `docs/adapter-interface.md`.

9. **Model revert.** Patch a model, call `can_revert()` (True), call `revert()`, verify original forward restored and `can_revert()` returns False.

10. **Algorithm selection.** `Qwen35KVBackend(key_strategy="mse")` — K uses MSE-only, no QJL correction. Invalid strategies raise `ValueError`.

### Known design decisions

11. **bit_width semantics.** When `bit_width=4` and `key_strategy="mse+qjl"`: K gets 3-bit MSE + 1-bit QJL, V gets 4-bit MSE. Same `bit_width` produces different codebook sizes for K and V. When `key_strategy="mse"`: both K and V get 4-bit MSE. Documented in `docs/adapter-interface.md`.

---

## File inventory

```
turboquant-core/
├── src/turboquant_core/
│   ├── __init__.py      # Public API exports (incl. unpatch_model, register_variant)
│   ├── core.py          # TQ algorithms: codebooks, rotation, MSE, QJL, prod,
│   │                    # TQQuantizedCache, STE gradient support
│   ├── backends/
│   │   ├── qwen.py      # Qwen35KVBackend (hybrid), Qwen3DenseKVBackend (dense)
│   │   └── qwen_hook.py # patch/unpatch functions
│   └── adapters/
│       ├── __init__.py   # Re-exports TurboQuantAdapter
│       └── workflow_eval.py  # TurboQuantAdapter, register_variant, _detect_variant
├── tests/
│   ├── test_core.py     # Core algorithm + backend tests (incl. strategy, registry)
│   ├── test_hooks.py    # Model hook-in, GQA, patched forward, unpatch tests
│   └── test_adapter.py  # Adapter, variant detection, revert, get_state tests
├── benchmarks/
│   └── benchmark_kv_cache.py  # Compression ratio, MSE, throughput
├── configs/models/
│   ├── qwen35_9b.yaml   # Probe-verified dims for Qwen3.5-9B
│   └── qwen3_8b.yaml    # Dims for Qwen3-8B
├── docs/
│   ├── adapter-interface.md  # Contract for eval harness integration
│   └── HANDOFF.md       # This file
├── .github/workflows/
│   └── ci.yml           # Pytest + ruff on Python 3.10-3.12
├── pyproject.toml       # pip install -e .
├── README.md
├── LICENSE              # Apache 2.0
└── .gitignore
```
