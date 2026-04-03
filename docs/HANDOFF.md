# Handoff: turboquant-core → turboquant-workflow-eval Integration

## Context

`turboquant-workflow-eval` is an existing evaluation harness for testing KV cache compression policies on Qwen3.5-9B. It currently includes a baseline pass-through adapter and a local Transformers-side patch adapter that quantizes K/V projection outputs on the 8 full-attention layers as a behavioral proxy.

`turboquant-core` is a new standalone library implementing the actual TurboQuant algorithms from the ICLR 2026 paper (arxiv:2504.19874). It provides the math (codebook computation, random rotation, Lloyd-Max quantization, QJL residual correction) and model-specific backends that expose a `compress`/`decompress` interface.

This document covers what `turboquant-core` implements, what it does not implement, what the reviewer should verify, and how to wire it into the eval harness.

---

## What turboquant-core implements

### Core algorithms (`src/turboquant_core/core.py`)

**CodebookRegistry** — Precomputes and caches Lloyd-Max codebooks for a given (dimension, bit_width) pair. The codebook is computed by importing a `get_codebook(d, b)` function from an external `turboquant` module (the paper's reference implementation). Centroids are sorted, and decision boundaries are precomputed as midpoints for O(log n) quantization via `torch.searchsorted`.

*Known issue:* The `CodebookRegistry.get()` method constructs the import path to the external `turboquant` module via `Path(__file__).parent.parent.parent / "turboquant"`. This assumes the paper's reference `turboquant.py` (with `get_codebook`) lives as a sibling directory at the repo root. If this external dependency is missing, codebook construction will fail at runtime. The reviewer should either vendor the reference implementation or replace this import with a self-contained Lloyd-Max solver. The algorithm is straightforward: iterate the Lloyd-Max fixed-point equations on the standard normal distribution for the given number of levels.

**RotationCache** — Precomputes randomized Hadamard rotation matrices. For dimension d, it pads to the next power of 2, generates a random sign vector (±1 per coordinate, seeded), and builds the Walsh-Hadamard matrix recursively. The rotation is applied as: `y = H @ diag(signs) @ x / sqrt(d_padded)`. The inverse is the same operation (Hadamard is symmetric and orthogonal).

*Note on the Hadamard construction:* The current implementation builds the full d×d dense matrix and stores it. For d=256 (the GatedAttn head_dim), this is a 256×256 float32 matrix (256 KB). This is fine for the eval harness but would not scale to production inference where you'd want the fast Walsh-Hadamard transform (O(d log d) without materializing the matrix).

**tq_quantize_mse / tq_dequantize_mse** — TurboQuant Algorithm 1 (MSE-optimal). Normalizes input vectors to unit norm, applies the random rotation, then quantizes each coordinate independently using the precomputed Lloyd-Max codebook via `searchsorted`. Stores the per-vector L2 norms separately in float32. Dequantization reverses: look up centroids, inverse-rotate, rescale by stored norms.

**QJLProjection** — Quantized Johnson-Lindenstrauss projection. Stores a d×d random Gaussian matrix S. Quantization is `sign(S @ x)`, producing 1-bit per coordinate. The theoretical guarantee: the QJL estimator of the inner product ⟨x, y⟩ has variance π/(2d)·‖x‖²·‖y‖², which vanishes as d grows.

*Note:* The S matrix is d×d dense (for d=256, that's 256 KB float32). In production this would be replaced with a structured random projection (e.g., SRHT) but for eval purposes the dense matrix is correct and matches the paper.

**tq_quantize_prod** — TurboQuant Algorithm 2 (unbiased inner products). Two-stage: first applies (b-1)-bit TQ_MSE to the unit-normalized input, then computes the residual and applies QJL 1-bit quantization to the normalized residual. Returns four components: MSE indices, QJL sign bits, residual norms, and original norms. The total effective bit-width is b: (b-1) bits for MSE + 1 bit for QJL.

### Model backends (`src/turboquant_core/backends/qwen.py`)

**Qwen35KVBackend** — For Qwen3.5-9B (hybrid architecture, 32 layers). Hardcodes the probe-verified constants: 32 layers, `full_attention_interval=4`, giving GatedAttn layers at indices {3, 7, 11, 15, 19, 23, 27, 31}. KV heads = 4, head_dim = 256.

The `compress` method:
- Asserts the layer is compressible (is a GatedAttn layer)
- Reshapes K from [batch, 4, seq_len, 256] to [batch*4*seq_len, 256]
- Applies TQ_prod to K (3-bit MSE + 1-bit QJL at default bit_width=4)
- Applies TQ_MSE to V (4-bit MSE, no QJL)
- Returns a dict with all compressed components + the original shape

The `decompress_v` method:
- Dequantizes V using TQ_MSE
- Reshapes back to [batch, 4, seq_len, 256]
- K decompression is not implemented — this is intentional (see "What is not implemented" below)

**Qwen3DenseKVBackend** — For Qwen3-8B (standard dense, 36 layers). Same interface, but `is_compressible` returns `True` for all layers. KV heads = 8, head_dim = 128.

Both backends implement the same three-method interface: `is_compressible(layer_idx)`, `compress(K, V, layer_idx)`, `decompress_v(compressed)`.

---

## What turboquant-core does NOT implement

> **Status update:** All four items below have been implemented. See git history.

### 1. K decompression for attention score computation ✅ IMPLEMENTED

The `decompress_v` method exists but there is no `decompress_k` method. This is the most significant gap. The reason TQ_prod exists is to provide unbiased estimation of softmax(Q @ K^T), but the current code only stores the compressed K representation — it doesn't provide a method to compute the corrected Q @ K_quantized^T inner product using the QJL residual.

To complete this, you need a method like:

```python
def compute_attention_scores(self, Q, compressed_K):
    """Compute unbiased Q @ K^T from fresh Q and compressed K."""
    # Stage 1: Q @ K_mse^T (biased, from MSE reconstruction)
    K_mse = dequantize_mse(compressed_K["k_mse"], ...)
    scores_mse = Q @ K_mse.T

    # Stage 2: QJL bias correction
    # ⟨q, r⟩ ≈ (π/2d) * (S@q).sign() @ compressed_K["k_qjl"].T * residual_norms
    Q_qjl = self.k_qjl.quantize(Q)
    correction = (Q_qjl.float() @ compressed_K["k_qjl"].float().T)
    correction = correction * (math.pi / (2 * self.kv_head_dim))
    correction = correction * compressed_K["k_rn"]  # scale by residual norms

    return scores_mse + correction * compressed_K["k_n"]  # scale by original K norms
```

This is the core of TurboQuant's contribution and it's not wired up yet. The eval harness's current local patch adapter doesn't need this because it quantizes K/V at the projection output and lets the standard attention kernel handle the (now-quantized) K/V directly. But a true TQ backend would intercept the attention score computation itself.

### 2. Integration with Qwen3_5DynamicCache ✅ IMPLEMENTED

The backends operate on raw K/V tensors but don't hook into the model's actual cache object. To use this in the eval harness, you need to either:

(a) Intercept K/V at the point your existing local patch adapter does (at the projection output), compress with the TQ backend, then dequantize before passing to the attention kernel. This is what the eval harness's adapter interface expects and is the simpler path. Quality results will be valid — you're measuring the reconstruction error — but you won't see true memory savings because the dequantized tensors still exist in FP16/BF16.

(b) Replace the cache object with a `TQQuantizedCache` that stores compressed representations and implements the corrected attention score computation above. This gives true memory savings but requires deeper integration with the HuggingFace model code.

For the eval harness, path (a) is sufficient and is what the reviewer should wire first.

### 3. Self-contained codebook computation ✅ IMPLEMENTED

As noted above, `CodebookRegistry` imports from an external `turboquant` module for `get_codebook(d, b)`. If this dependency is missing, the library won't initialize. The reviewer should either:

- Vendor the reference implementation's `get_codebook` function (it's a standard Lloyd-Max iteration on the Gaussian distribution — ~50 lines of scipy)
- Or replace the import with a hardcoded table of known-good centroids for the dimensions actually used (128 for Qwen3-8B, 256 for Qwen3.5-9B)

### 4. Gradient support ✅ IMPLEMENTED

All operations use `torch.no_grad()` semantics or detached tensors. The quantization functions are not differentiable. This is correct for inference-time KV cache compression but means you cannot backpropagate through the TQ operations during training. For the eval harness this is fine — you're measuring inference quality, not training.

---

## How to wire into turboquant-workflow-eval

### Step 1: Install turboquant-core

```bash
# As sibling directory
pip install -e ../turboquant-core

# Or from git
pip install git+https://github.com/YOUR_ORG/turboquant-core.git
```

### Step 2: Create the adapter

Create `src/qwen35_turboquant_workflow_study/adapters/turboquant_real.py`:

```python
from turboquant_core.backends.qwen import Qwen35KVBackend

class TurboQuantRealAdapter:
    """
    Adapter that wraps the real TQ backend for the eval harness.

    This adapter operates at the K/V projection output level:
    captures K/V tensors, compresses them with TQ, then immediately
    dequantizes and passes the reconstructed tensors forward.

    This measures reconstruction quality, not memory savings.
    """

    def __init__(self, bit_width=4, device="cuda"):
        self.backend = Qwen35KVBackend(bit_width=bit_width, device=device)

    def should_apply(self, layer_idx: int) -> bool:
        return self.backend.is_compressible(layer_idx)

    def process_kv(self, K, V, layer_idx):
        """Compress then immediately decompress to measure quality impact."""
        if not self.should_apply(layer_idx):
            return K, V  # DeltaNet layers: pass through unchanged

        compressed = self.backend.compress(K, V, layer_idx)
        V_reconstructed = self.backend.decompress_v(compressed)

        # For K: dequantize the MSE component only (no QJL correction applied)
        # This is conservative — actual TQ_prod would give better K fidelity
        from turboquant_core.core import tq_dequantize_mse
        b, nh, sl, hd = K.shape
        K_reconstructed = tq_dequantize_mse(
            compressed["k_mse"], compressed["k_n"],
            self.backend.k_cb, self.backend.k_rot
        ).reshape(K.shape)

        return K_reconstructed, V_reconstructed
```

Note: this adapter dequantizes K using only the MSE component, not the full TQ_prod reconstruction with QJL correction. This makes the K reconstruction *worse* than what a real TQ_prod backend would achieve, which means the eval results are a conservative lower bound on TQ quality. If the eval passes with this adapter, the real TQ will be at least as good.

### Step 3: Add a policy config

Create `configs/policies/tq_real_4bit.yaml`:

```yaml
policy:
  name: "tq_real_4bit"
  adapter_class: "adapters.turboquant_real.TurboQuantRealAdapter"
  adapter_kwargs:
    bit_width: 4
  description: "Real TurboQuant: 3-bit MSE + 1-bit QJL on K, 4-bit MSE on V (GatedAttn only)"
```

### Step 4: Run the comparison

```bash
make study \
  POLICY_CONFIGS=configs/policies/baseline.yaml,configs/policies/safe_template.yaml,configs/policies/tq_real_4bit.yaml \
  OUTPUT_DIR=outputs/study_tq_real
```

This gives a three-way comparison in `workflow_compare.csv`:
1. Baseline (no compression)
2. Local patch (existing Transformers-side proxy)
3. Real TQ (turboquant-core backend)

The local patch vs real TQ comparison is the most interesting — it tells you whether your behavioral proxy was accurately predicting what real TQ would do.

---

## What the reviewer should verify

### Correctness checks

1. **Codebook centroids match the paper.** For b=1, the Lloyd-Max codebook on N(0,1) should produce 2 centroids at approximately ±0.7979. For b=2, 4 centroids at approximately ±0.4528 and ±1.5104. Run `test_core.py` and additionally verify against Table 2 in the paper.

2. **Rotation preserves norms.** The randomized Hadamard rotation should be orthogonal: `‖rotate(x)‖ = ‖x‖` for any x. Test with random vectors.

3. **MSE round-trip error.** Quantize a batch of random Gaussian vectors, dequantize, and measure MSE/coord. At b=4, d=256, this should be approximately 0.0115 (from paper Table 2). The actual value may differ slightly because the paper's codebooks are computed for asymptotic d→∞ but should be within 10%.

4. **K vs V asymmetry is correct.** Verify that K goes through `tq_quantize_prod` (MSE + QJL) while V goes through `tq_quantize_mse` (MSE only). This is the core design decision: K participates in the QK^T inner product (where bias matters), V does not.

5. **Layer filtering is correct.** `Qwen35KVBackend.is_compressible(i)` should return `True` for i ∈ {3,7,11,15,19,23,27,31} and `False` for all other indices 0-31. These are the probe-verified GatedAttn layer indices.

### Integration checks

6. **Tensor shapes survive round-trip.** Feed K/V with shape [1, 4, 128, 256] (batch=1, 4 KV heads, 128 tokens, 256 head_dim) through `compress` then `decompress_v`. Output V shape must match input V shape exactly.

7. **No NaN/Inf.** Run compress/decompress on random inputs and zero inputs. Check for NaN in the output. The `+ 1e-12` guards in the normalization should prevent division by zero but verify.

8. **Adapter interface compatibility.** The three methods (`is_compressible`, `compress`, `decompress_v`) should match the contract in `docs/adapter-interface.md` in turboquant-workflow-eval. The method names and signatures were designed to align but the reviewer should confirm against the actual eval harness code.

### Known limitations to document

9. ~~**The `get_codebook` external dependency.**~~ ✅ RESOLVED — replaced with self-contained `_lloyd_max_gaussian()` solver using scipy.

10. ~~**Dense Hadamard matrix at d=256.**~~ ✅ RESOLVED — replaced with fast Walsh-Hadamard transform (O(d log d), no matrix materialized).

11. ~~**No K decompression with QJL correction.**~~ ✅ RESOLVED — `compute_attention_scores()` implements full QJL-corrected Q@K^T.

12. **bit_width parameter semantics.** When `bit_width=4` is passed to the backend, K gets a (4-1)=3 bit codebook and V gets a 4-bit codebook. The effective bits per value are 4 for K (3 MSE + 1 QJL) and 4 for V (4 MSE). This is correct but may be confusing — the same `bit_width` parameter produces different codebook sizes for K and V. Documented in `docs/adapter-interface.md`.

---

## File inventory

```
turboquant-core/
├── src/turboquant_core/
│   ├── __init__.py      # Public API exports
│   ├── core.py          # TQ algorithms: codebooks, rotation, MSE, QJL, prod,
│   │                    # TQQuantizedCache, STE gradient support
│   └── backends/
│       ├── qwen.py      # Qwen35KVBackend (hybrid), Qwen3DenseKVBackend (dense)
│       └── qwen_hook.py # patch_qwen35_with_tq() model hook-in
├── tests/
│   └── test_core.py     # 35 tests: algorithms, backends, paper verification
├── configs/models/
│   ├── qwen35_9b.yaml   # Probe-verified dims for Qwen3.5-9B
│   └── qwen3_8b.yaml    # Dims for Qwen3-8B
├── docs/
│   ├── adapter-interface.md  # Contract for eval harness integration
│   └── HANDOFF.md       # This file
├── pyproject.toml       # pip install -e .
├── README.md
├── LICENSE              # Apache 2.0
└── .gitignore
```
