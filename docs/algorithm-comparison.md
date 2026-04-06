# TurboQuant Algorithm Comparison

A comparison of **flaker420/turboquant-core** against [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) and community implementations listed in its README.

---

## Summary Matrix

| Feature | turboquant-core | tonbistudio (V3) | scos-lab | 0xSero | back2matching | TheTom/plus | RecursiveIntell | SCJedi/entropy |
|---|---|---|---|---|---|---|---|---|
| **Language** | Python/PyTorch | Python/PyTorch | Python/PyTorch | Python/Triton | Python/PyTorch | C/Metal + Python | Rust | Python |
| **Core Quant** | MSE + QJL | MSE-only (V3) | MSE-only | MSE + QJL | MSE-only | PolarQuant + WHT | PolarQuant + TQ + QJL | Token eviction (no quant) |
| **QJL Residual** | Yes (key strategy) | Removed in V3 | Removed | Yes | Removed | No | Yes | N/A |
| **Asymmetric K/V bits** | No (same bit_width) | Yes (K6/V4, K4/V2) | Yes | Yes (K3-4/V2-4) | Yes (K4/V2 default) | Yes (K=q8_0, V=turbo2-4) | No | N/A (eviction-based) |
| **Layer-adaptive** | No | Yes (protect early/late) | No | Selective (attn-only) | Yes (protected_layers) | Yes (boundary V) | No | Yes (per-head entropy) |
| **Residual window** | Yes (configurable, default 0) | Yes (128-token FP16) | No | No | Yes (128-token FP16) | No | No | No |
| **Bit-packing** | No (index tensors) | Yes (V3) | Yes | Yes | Yes | Yes (block-32) | Yes | N/A |
| **GPU kernels** | No | No | No | 3 Triton kernels | No | Metal GPU kernels | No | No |
| **Pip-installable** | No | No | No | No | Yes | llama.cpp integration | Cargo crate | No |
| **Models tested** | Qwen3.5-9B, Qwen3-8B, Qwen2.5-3B | Qwen2.5-3B | 8 models (GPT-2 to Qwen2.5-7B) | Qwen3.5-27B, Qwen3.5-35B | Qwen2.5 family, StableLM | 30+ models incl. Command-R+ 104B, Llama-70B | Generic | GPT-2 |

---

## Detailed Algorithm Comparisons

### 1. Quantization Core: MSE vs. MSE+QJL

**turboquant-core** implements both `TQ_MSE` and `TQ_Prod` (MSE+QJL) as the paper describes. The key strategy is configurable: `"mse+qjl"` uses (b-1)-bit MSE codebook + 1-bit QJL sign correction for keys, while values always use MSE-only.

**Community consensus (6 independent implementations)** found that QJL's theoretical unbiasedness does not survive softmax attention in practice -- the variance amplification through softmax degrades actual generation quality. This is the single most impactful divergence:

- **tonbistudio V3**, **scos-lab**, **back2matching** all dropped QJL entirely
- **0xSero** and **RecursiveIntell** retain QJL but acknowledge the debate
- **turboquant-core** retains QJL as the default key strategy

**Implication for turboquant-core**: The MSE+QJL path may underperform a pure MSE approach at equivalent bit budgets for real generation tasks, despite being theoretically motivated. The repo should consider benchmarking MSE-only keys against MSE+QJL on actual generation quality (not just MSE-per-coordinate).

### 2. Asymmetric K/V Bit Allocation

**turboquant-core** applies the same `bit_width` to both K and V. The only asymmetry is that K uses (b-1) bits for MSE + 1 bit for QJL when using the `"mse+qjl"` strategy, while V gets the full `b` bits of MSE.

**Most community implementations** discovered that keys and values have dramatically different precision requirements:
- Key norms range 172-778 vs. value norms of 2-4 (scos-lab measured up to 1274x ratio in smaller models)
- Keys need higher precision because errors are amplified through softmax
- Values tolerate aggressive compression because they're weighted-averaged

| Repo | Typical K/V config | Reasoning |
|---|---|---|
| tonbistudio V3 | K6/V4 or K4/V2 | Norm disparity; softmax amplification |
| scos-lab | 3.6-bit mixed | Outlier-aware; K channels at 8-bit |
| back2matching | K4/V2 default | "Keys matter more" |
| TheTom/plus | K=q8_0, V=turbo2 | "V compression is free" -- 2-bit values show zero measurable attention degradation |

**Implication for turboquant-core**: Adding independent `key_bit_width` and `value_bit_width` parameters could yield significantly better quality-compression tradeoffs. TheTom's finding that 2-bit values cause zero attention degradation (when keys are well-preserved) suggests turboquant-core is over-allocating bits to values.

### 3. Layer-Adaptive Precision

**turboquant-core** treats all compressible layers identically (same bit_width, same strategy). For Qwen3.5, it correctly identifies only 8/32 GatedAttn layers as compressible (skipping DeltaNet layers), but applies uniform precision to those 8.

**Community approaches**:
- **tonbistudio V3**: Protects early/late layers with extra bits, compresses middle layers aggressively
- **back2matching**: Configurable `protected_layers=[0, 1, -1, -2]` kept at full precision
- **TheTom/plus**: "Boundary V" protects first/last 2 layers, recovers 37-91% of quality gap
- **SCJedi**: Per-head entropy-based budget allocation (300x entropy variance across heads)

**Implication for turboquant-core**: The `TQGatedAttnKVCache` already tracks per-layer caches. Adding per-layer bit_width overrides would be straightforward and could recover significant quality at negligible cost.

### 4. Residual Windowing (Recent Token Protection)

**turboquant-core** does not implement residual windowing. All tokens are quantized equally regardless of recency.

**tonbistudio V3** and **back2matching** keep the most recent 128 tokens in full FP16 precision, only quantizing older cache entries. This is critical for generation quality:

| Config (tonbistudio) | 2K Retrieval | 4K Retrieval |
|---|---|---|
| K6/V4 + 128-token FP16 window | EXACT | EXACT |
| K4/V4 + 128-token FP16 window | PARTIAL | MISS |
| K4/V2, no window | MISS | MISS |

**Implication for turboquant-core**: This is arguably the highest-impact missing feature. Recent tokens dominate attention weights in autoregressive generation, so keeping them at full precision provides outsized quality benefits at minimal memory cost.

### 5. Bit-Packing & Actual Compression

**turboquant-core** stores quantization indices as standard int8/int16 tensors, plus float32 norm tensors. The reported ~1.9-2.0x compression ratio reflects this overhead.

**tonbistudio** explicitly called out that their V2 (similar to turboquant-core's approach) produced tensors **38% larger than uncompressed data** before implementing proper bit-packing in V3.

**Community implementations** with bit-packing achieve substantially higher effective compression:

| Repo | Compression Ratio | Method |
|---|---|---|
| turboquant-core | ~1.9-2.0x | Index tensors + f32 norms |
| 0xSero | 4.4-5.0x | Bit-packed K3/V2 |
| back2matching | ~3x (K4/V2) | Bit-packed asymmetric |
| TheTom/plus | 4.6-5.1x (turbo3) | Block-32 packed format |
| tonbistudio V3 | ~2-3x | Bit-packed + window overhead |

**Implication for turboquant-core**: Without bit-packing, the repo cannot achieve the compression ratios the algorithm is theoretically capable of. This is a fundamental gap for production use.

### 6. Rotation Implementation

All implementations use Walsh-Hadamard Transform (WHT) for random rotation. turboquant-core's O(d log d) in-place Fast WHT is competitive with every Python-based implementation.

**TheTom/plus** goes further with fp16 WHT execution on Metal GPU and half4 vectorized butterfly operations, achieving +38-45% decode speedup at long context on Apple Silicon.

**RecursiveIntell** uses Haar-distributed orthogonal matrices via QR decomposition of ChaCha8-seeded Gaussian matrices -- a different approach that's more expensive but provably uniform on the orthogonal group.

turboquant-core's approach is solid here and matches the paper faithfully.

### 7. Hardware Acceleration

**turboquant-core**: Pure PyTorch, CPU benchmarks only (~50k tok/s compress, ~100k tok/s decompress V).

| Repo | Hardware | Performance |
|---|---|---|
| 0xSero | RTX 5090 | 1907 tok/s prefill (+5.7%), 914K max tokens (2x capacity) |
| 0xSero | 8x RTX 3090 | 10K tok/s prefill, 30.9% KV savings per GPU |
| TheTom/plus | M5 Max | turbo4 = +33.9% decode speed vs q8_0 at 38K tokens |
| back2matching | Qwen 3B, 4K | 7.4 tok/s vs 2.5 tok/s baseline (196% improvement) |

### 8. Novel Approaches Not in turboquant-core

| Technique | Source | Description |
|---|---|---|
| **Attention-gated value decoding** | TheTom/plus | Skips V positions where softmax weight < 1e-6; +22.8% decode speedup at 32K context with no quality loss |
| **Entropy-adaptive eviction** | SCJedi | Per-head cache budget based on entropy; at 5x compression matches uniform at 2x; complementary to quantization |
| **Outlier-aware mixed precision** | scos-lab | High-magnitude K channels at 8-bit, others at 3-bit; 3.6-bit average with +2.1% PPL |
| **PolarQuant (angle quantization)** | RecursiveIntell, TheTom | Converts to polar coordinates, quantizes angles uniformly on [-pi, pi]; alternative to Lloyd-Max codebook |
| **OpenAI-compatible server** | back2matching | `turboquant-server` wraps compression behind standard API |

---

## Architecture Ceiling: Dense vs. Hybrid

A critical factor the community hasn't always made explicit: **compressible layer fraction caps end-to-end gains**.

If compressible layers are fraction `f` of total KV state and you achieve compression ratio `r` on them, end-to-end memory reduction is `1 / ((1 - f) + f / r)`.

| Model | Compressible layers | f (approx) | At r=5x | At r=10x | Hard ceiling |
|---|---|---|---|---|---|
| **Qwen2.5-3B** | 36/36 (all) | 1.0 | **5.0x** | **10.0x** | Unlimited |
| **Qwen3-8B** | 36/36 (all) | 1.0 | **5.0x** | **10.0x** | Unlimited |
| **Qwen3.5-9B** | 8/32 (GatedAttn only) | ~0.25 | **1.25x** | **1.29x** | **1.33x** |

Qwen3.5-9B's DeltaNet layers carry opaque recurrent state that TurboQuant cannot compress. Even perfect compression on the 8 GatedAttn layers yields at most 1.33x end-to-end. This means:

- **Qwen3-8B should be the showcase target** for demonstrating TurboQuant's compression value
- **Qwen3.5-9B should be framed as the hybrid, bounded-upside case** where the contribution is architectural compatibility, not headline compression ratios
- Published results should always report **end-to-end memory reduction**, not just per-compressible-layer ratios

0xSero independently warns of the same issue: hybrid/Mamba-like state limits overall TurboQuant impact on those models.

---

## Improvement Plan (Reviewer-Validated)

The following plan reflects consensus across this comparison, an independent code reviewer, and community evidence. The key reframing: **the next high-value artifact is a reproducible ablation table, not a code rewrite.** The algorithm works; the question is which configuration point wins for softmax attention on Qwen targets.

### PR0: Correctness and Evaluation Hygiene (Merged)

**Rationale**: If attention semantics are wrong, every downstream ablation is suspect. tonbistudio had to retract a headline result after discovering a no-compression bug. Fix correctness first.

- [x] **Causal mask in patched attention** -- `qwen_hook.py` now builds a proper causal mask. Both `_gqa_attention` and `TQQuantizedCache.compute_attention` accept `causal_mask` and `attention_mask`. Regression tests added for prefill causality and incremental decode. **Remaining**: padded-batch parity tests against unpatched attention.
- [x] **Automatic cache clearing** -- `reset_generation_state()` added to `TurboQuantAdapter`. Harnesses should call this between generations.
- [ ] **Explicit baseline selection** -- Evaluator can shuffle policy order and picks first row per prompt as baseline. Baseline must be explicit by policy name or `is_baseline: true` flag. *(Lives in turboquant-workflow-eval, not this repo.)*
- [x] **`update_params` is intentionally unsupported** -- Now raises `NotImplementedError` so silent param drift can't happen. Callers must `revert(model)` + `prepare_model(...)` to change settings.
- [ ] **Stricter reference-answer scoring** -- Numeric checker accepts any matching number anywhere in output. Tighten before using results as canonical. *(Lives in turboquant-workflow-eval, not this repo.)*
- [x] **Fix duplicated V quantization** -- `Qwen35KVBackend.compress()` was calling `tq_quantize_mse` twice for V. Fixed to call once.

### PR1: Algorithmic Ablation Matrix

**Rationale**: Community evidence strongly suggests the default configuration is suboptimal. Run a staged ablation on Qwen3-8B (all layers compressible, no architecture ceiling), then transfer winning configs to Qwen3.5-9B.

**Stage 1**: `mse` vs `mse+qjl` x residual window {0, 128, 256}
- Community prediction: MSE-only + window will win
- This settles the QJL question on turboquant-core's actual targets

**Stage 2** (conditional on Stage 1 winner): Asymmetric K/V splits
- K bits: {3, 4, 6} x V bits: {2, 3, 4}
- Note: optimal V precision is a function of K precision (TheTom reports "V is free" at K=q8_0; 0xSero says V2 is the bottleneck at K=3-bit)

**Stage 3**: Protected layers
- {none, first/last 1, first/last 2} on winning K/V config

**Stage 4**: Transfer top 3 configs to Qwen3.5-9B and measure end-to-end (not per-layer) gains

**Required code changes to enable ablation**:
- [ ] Add `key_bit_width` / `value_bit_width` independent parameters
- [x] Add `residual_window` parameter (keep recent N tokens in FP16) -- Merged in PR0.
- [ ] Add `protected_layers` list parameter
- [x] Keep QJL as an option but benchmark before deciding default -- `key_strategy` exposed in backends; now threaded through `TQQuantizedCache` and hooked attention path.
- [x] Add Qwen2.5-3B-Instruct backend as dense ablation target -- `Qwen25DenseKVBackend` merged in PR0.

**Measurement discipline** -- Every published row must include:
- Actual compressed-token count
- Residual-window length
- Protected-layer policy
- Honest bytes/token including all metadata (norms, QJL bits, projection matrices)
- Whether decode reconstructs full history or operates in compressed domain
- End-to-end memory reduction (not just per-compressible-layer ratio)

### PR2: Real Compression / Packing

**Rationale**: Without bit-packing, even the winning config from PR1 will show only ~2x compression. This is the gap between "algorithm works" and "library is useful."

- [ ] Pack MSE codes to actual 2/3/4 bits
- [ ] Pack QJL signs to 1-bit (if QJL survives PR1 ablation)
- [ ] Compress norms to fp16/bf16 or shared block scales
- [ ] Avoid materializing full dequantized V unless necessary
- [ ] Replace `torch.cat` append with paged/blockwise storage

### PR3: Compressed-Domain Compute and Kernels

**Rationale**: After packing, the bottleneck shifts to decode-time dequantization. This is where 0xSero (Triton) and TheTom (Metal) differentiate on runtime.

- [ ] Tiled attention over compressed chunks (avoid full-cache reconstruction)
- [ ] Fused unpack/dequant + score accumulation for K
- [ ] Consider attention-gated V decode (TheTom: +22.8% at 32K, model-agnostic)
- [ ] Vectorized PyTorch interim before Triton/CUDA kernels

### Publication Target

If turboquant-core publishes a rigorous ablation table on Qwen3-8B and Qwen3.5-9B with honest end-to-end numbers and full measurement metadata, it becomes more than another implementation -- it becomes the reference point the community cites. No other repo has done this systematically on these models with this level of accounting discipline.

---

## Community Findings: "V Compression Is Free" -- Nuanced

TheTom/plus reports 2-bit values with zero measurable attention degradation when keys are well-preserved. But 0xSero's audit says 2-bit values are the quality bottleneck in its stack and recommends 4-bit for quality-sensitive workloads.

These findings are not contradictory -- they reflect different K precision baselines:
- TheTom tests with K at q8_0 (8-bit), giving values enormous headroom
- 0xSero tests with K at 3-bit, where key noise compounds with value noise

The actionable conclusion: **optimal V precision is a function of K precision**. This is why asymmetric allocation must be independently tunable, not hardcoded. The ablation matrix in PR1 is designed to capture this interaction.

---

## Architectural Strengths of turboquant-core

Despite gaps in configuration and systems engineering, turboquant-core has notable strengths:

- **Faithful paper implementation**: Most complete reference of the original TurboQuant algorithms including QJL (which others dropped). QJL should remain available for retrieval/experimental paths even if it stops being the default for softmax attention.
- **Clean separation of concerns**: core.py / backends / adapters architecture is well-structured
- **Hybrid architecture support**: Correct handling of Qwen3.5's mixed GatedAttn/DeltaNet layers (only 0xSero also does this)
- **STE support**: Differentiable quantization path for fine-tuning (unique among all repos)
- **Eval adapter pattern**: Clean integration with evaluation harnesses
- **Comprehensive tests**: Theorem-validated unit tests with paper-matching MSE values
- **Existing `key_strategy` parameter**: Already exposes the MSE vs MSE+QJL lever; extending to asymmetric bits and windowing builds on existing design patterns
