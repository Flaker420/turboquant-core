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
| **Residual window** | No | Yes (128-token FP16) | No | No | Yes (128-token FP16) | No | No | No |
| **Bit-packing** | No (index tensors) | Yes (V3) | Yes | Yes | Yes | Yes (block-32) | Yes | N/A |
| **GPU kernels** | No | No | No | 3 Triton kernels | No | Metal GPU kernels | No | No |
| **Pip-installable** | No | No | No | No | Yes | llama.cpp integration | Cargo crate | No |
| **Models tested** | Qwen3.5-9B, Qwen3-8B | Qwen2.5-3B | 8 models (GPT-2 to Qwen2.5-7B) | Qwen3.5-27B, Qwen3.5-35B | Qwen2.5 family, StableLM | 30+ models incl. Command-R+ 104B, Llama-70B | Generic | GPT-2 |

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

## Recommended Priorities for turboquant-core

Based on community findings ranked by impact:

1. **Add residual windowing** -- Keep recent N tokens (e.g., 128) in FP16. Highest quality impact per engineering effort. Multiple repos validate this.

2. **Benchmark MSE-only vs MSE+QJL on generation quality** -- Six independent teams found MSE-only superior through softmax. turboquant-core should validate this on its Qwen targets before defaulting to QJL.

3. **Asymmetric K/V bit allocation** -- Independent `key_bit_width`/`value_bit_width` parameters. TheTom's "V compression is free" finding suggests major wins from giving keys more bits.

4. **Implement bit-packing** -- Current index-tensor storage limits practical compression to ~2x. Bit-packing could push to 4-5x.

5. **Layer-adaptive precision** -- Protect first/last layers at higher precision. Straightforward given existing per-layer architecture.

6. **Consider attention-gated V decoding** -- +22.8% decode speedup with no quality cost is compelling, though it's a decode optimization rather than a compression technique.

---

## Architectural Strengths of turboquant-core

Despite missing community-discovered optimizations, turboquant-core has notable strengths:

- **Faithful paper implementation**: Most complete reference of the original TurboQuant algorithms including QJL (which others dropped)
- **Clean separation of concerns**: core.py / backends / adapters architecture is well-structured
- **Hybrid architecture support**: Correct handling of Qwen3.5's mixed GatedAttn/DeltaNet layers (only 0xSero also does this)
- **STE support**: Differentiable quantization path for fine-tuning (unique among all repos)
- **Eval adapter pattern**: Clean integration with evaluation harnesses
- **Comprehensive tests**: Theorem-validated unit tests with paper-matching MSE values
