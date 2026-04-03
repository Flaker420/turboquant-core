"""Benchmark TurboQuant KV cache compression.

Measures: compression ratio, MSE per coordinate, and throughput
for both Qwen3.5-9B and Qwen3-8B backends.

Usage:
    python benchmarks/benchmark_kv_cache.py
"""

import time
import torch
from turboquant_core.backends.qwen import Qwen35KVBackend, Qwen3DenseKVBackend


def benchmark_backend(backend, name, K, V, layer_idx, warmup=5, iters=50):
    """Benchmark compress + decompress_v + compute_attention_scores."""
    b, nh, sl, hd = K.shape

    # Warmup
    for _ in range(warmup):
        compressed = backend.compress(K, V, layer_idx)
        backend.decompress_v(compressed)
        backend.compute_attention_scores(K, compressed)

    # Compress throughput
    t0 = time.perf_counter()
    for _ in range(iters):
        compressed = backend.compress(K, V, layer_idx)
    t_compress = (time.perf_counter() - t0) / iters

    # Decompress V throughput
    t0 = time.perf_counter()
    for _ in range(iters):
        V_out = backend.decompress_v(compressed)
    t_decompress = (time.perf_counter() - t0) / iters

    # Attention score throughput
    Q = torch.randn_like(K)
    t0 = time.perf_counter()
    for _ in range(iters):
        scores = backend.compute_attention_scores(Q, compressed)
    t_attention = (time.perf_counter() - t0) / iters

    # Quality: V reconstruction MSE
    V_out = backend.decompress_v(compressed)
    mse = ((V - V_out) ** 2).mean().item()
    mse_per_coord = mse  # already per-element from .mean()

    # Compression ratio
    original_bytes = K.nelement() * 2 + V.nelement() * 2  # bfloat16
    compressed_bytes = (
        compressed["k_mse"].nelement() * 1  # uint8
        + compressed["k_qjl"].nelement() * 1  # int8
        + compressed["k_rn"].nelement() * 4  # float32
        + compressed["k_n"].nelement() * 4  # float32
        + compressed["v_idx"].nelement() * 1  # uint8
        + compressed["v_n"].nelement() * 4  # float32
    )
    ratio = original_bytes / compressed_bytes

    tokens = b * sl
    print(f"\n{'=' * 60}")
    print(f"{name} (bit_width=4, {nh} KV heads, head_dim={hd})")
    print(f"  Input: batch={b}, seq_len={sl}, tokens={tokens}")
    print(f"  Compress:    {t_compress*1000:8.2f} ms  ({tokens/t_compress:,.0f} tok/s)")
    print(f"  Decompress V:{t_decompress*1000:8.2f} ms  ({tokens/t_decompress:,.0f} tok/s)")
    print(f"  Attn scores: {t_attention*1000:8.2f} ms  ({tokens/t_attention:,.0f} tok/s)")
    print(f"  V MSE/coord: {mse_per_coord:.6f}")
    print(f"  Compression: {ratio:.2f}x ({original_bytes/1024:.1f} KB -> {compressed_bytes/1024:.1f} KB)")


def main():
    torch.manual_seed(42)
    device = "cpu"  # Use "cuda" if available

    print("TurboQuant KV Cache Benchmark")
    print(f"Device: {device}")

    # Qwen3.5-9B: 4 KV heads, 256 head_dim
    backend_35 = Qwen35KVBackend(bit_width=4, device=device)
    for seq_len in [128, 512, 1024]:
        K = torch.randn(1, 4, seq_len, 256, device=device)
        V = torch.randn(1, 4, seq_len, 256, device=device)
        benchmark_backend(backend_35, f"Qwen3.5-9B (seq={seq_len})", K, V, layer_idx=3)

    # Qwen3-8B: 8 KV heads, 128 head_dim
    backend_3 = Qwen3DenseKVBackend(bit_width=4, device=device)
    for seq_len in [128, 512, 1024]:
        K = torch.randn(1, 8, seq_len, 128, device=device)
        V = torch.randn(1, 8, seq_len, 128, device=device)
        benchmark_backend(backend_3, f"Qwen3-8B (seq={seq_len})", K, V, layer_idx=0)


if __name__ == "__main__":
    main()
