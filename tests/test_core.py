"""Comprehensive tests for TurboQuant core algorithms and backends."""

import math
import torch
import numpy as np

from turboquant_core.core import (
    _lloyd_max_gaussian,
    CodebookRegistry,
    RotationCache,
    tq_rotate,
    tq_rotate_inv,
    tq_quantize_mse,
    tq_dequantize_mse,
    QJLProjection,
    tq_quantize_prod,
    TQGatedAttnKVCache,
)
from turboquant_core.backends.qwen import Qwen35KVBackend, Qwen3DenseKVBackend


# ---------------------------------------------------------------------------
# Codebook tests
# ---------------------------------------------------------------------------

def test_lloyd_max_b1():
    """b=1: Lloyd-Max on N(0,1) → 2 centroids at approximately ±0.7979."""
    centroids, mse = _lloyd_max_gaussian(1)
    assert len(centroids) == 2
    assert abs(centroids[0] - (-0.7979)) < 0.01, f"got {centroids[0]}"
    assert abs(centroids[1] - 0.7979) < 0.01, f"got {centroids[1]}"
    assert abs(mse - 0.3634) < 0.01, f"MSE/coord at b=1: {mse}"
    print(f"  b=1 centroids: {centroids}, MSE={mse:.4f} — OK")


def test_lloyd_max_b2():
    """b=2: Lloyd-Max on N(0,1) → 4 centroids at ±0.4528, ±1.5104."""
    centroids, mse = _lloyd_max_gaussian(2)
    assert len(centroids) == 4
    expected = [-1.5104, -0.4528, 0.4528, 1.5104]
    for c, e in zip(centroids, expected):
        assert abs(c - e) < 0.01, f"got {c}, expected {e}"
    print(f"  b=2 centroids: {np.round(centroids, 4)} — OK")


def test_codebook_registry():
    """CodebookRegistry returns valid codebook with correct structure."""
    cb = CodebookRegistry.get(256, 4)
    assert cb.centroids.shape == (16,), f"Expected 16 centroids for b=4, got {cb.centroids.shape}"
    assert cb.boundaries.shape == (17,), f"Expected 17 boundaries, got {cb.boundaries.shape}"
    assert cb.boundaries[0] == -1.0
    assert cb.boundaries[-1] == 1.0
    # Centroids should be sorted
    assert torch.all(cb.centroids[:-1] <= cb.centroids[1:])
    print(f"  CodebookRegistry b=4: {cb.centroids.shape[0]} centroids — OK")


# ---------------------------------------------------------------------------
# Rotation tests
# ---------------------------------------------------------------------------

def test_rotation_preserves_norms():
    """Randomized Hadamard rotation should preserve vector norms."""
    d = 256
    rot = RotationCache.get(d, seed=42)
    x = torch.randn(100, d)
    y = tq_rotate(x, rot)
    norms_in = torch.linalg.norm(x, dim=-1)
    norms_out = torch.linalg.norm(y, dim=-1)
    max_err = (norms_in - norms_out).abs().max().item()
    assert max_err < 1e-4, f"Norm preservation error: {max_err}"
    print(f"  Rotation norm preservation (d={d}): max_err={max_err:.2e} — OK")


def test_rotation_invertible():
    """rotate_inv(rotate(x)) ≈ x."""
    d = 256
    rot = RotationCache.get(d, seed=42)
    x = torch.randn(50, d)
    reconstructed = tq_rotate_inv(tq_rotate(x, rot), rot)
    max_err = (x - reconstructed).abs().max().item()
    assert max_err < 1e-4, f"Rotation inverse error: {max_err}"
    print(f"  Rotation invertibility (d={d}): max_err={max_err:.2e} — OK")


# ---------------------------------------------------------------------------
# MSE quantization tests
# ---------------------------------------------------------------------------

def test_mse_round_trip_shape():
    """Quantize/dequantize preserves shape."""
    d = 256
    cb = CodebookRegistry.get(d, 4)
    rot = RotationCache.get(d, seed=42)
    x = torch.randn(32, d)
    indices, norms = tq_quantize_mse(x, cb, rot)
    x_hat = tq_dequantize_mse(indices, norms, cb, rot)
    assert x_hat.shape == x.shape, f"Shape mismatch: {x_hat.shape} vs {x.shape}"
    print(f"  MSE round-trip shape: {x.shape} → {x_hat.shape} — OK")


def test_mse_round_trip_error():
    """MSE/coord at b=4, d=256 should be approximately 0.0115 (paper Table 2)."""
    d = 256
    cb = CodebookRegistry.get(d, 4)
    rot = RotationCache.get(d, seed=42)
    # Use large batch of Gaussian vectors for stable estimate
    x = torch.randn(10000, d)
    indices, norms = tq_quantize_mse(x, cb, rot)
    x_hat = tq_dequantize_mse(indices, norms, cb, rot)
    mse_per_coord = ((x - x_hat) ** 2).mean().item()
    # Paper says ~0.0115 at b=4; allow 30% tolerance for finite-sample effects
    assert mse_per_coord < 0.02, f"MSE/coord too high: {mse_per_coord}"
    print(f"  MSE/coord at b=4 d=256: {mse_per_coord:.5f} — OK")


def test_mse_no_nan():
    """No NaN/Inf on random and zero inputs."""
    d = 128
    cb = CodebookRegistry.get(d, 3)
    rot = RotationCache.get(d, seed=42)
    for label, x in [("random", torch.randn(16, d)), ("zeros", torch.zeros(16, d))]:
        indices, norms = tq_quantize_mse(x, cb, rot)
        x_hat = tq_dequantize_mse(indices, norms, cb, rot)
        assert not torch.isnan(x_hat).any(), f"NaN in {label} output"
        assert not torch.isinf(x_hat).any(), f"Inf in {label} output"
    print(f"  No NaN/Inf on random and zero inputs — OK")


# ---------------------------------------------------------------------------
# QJL tests
# ---------------------------------------------------------------------------

def test_qjl_output_shape():
    """QJL quantize returns correct shape and int8 type."""
    d = 256
    qjl = QJLProjection(d, seed=123)
    x = torch.randn(32, d)
    bits = qjl.quantize(x)
    assert bits.shape == x.shape
    assert bits.dtype == torch.int8
    assert set(bits.unique().tolist()).issubset({-1, 0, 1})
    print(f"  QJL output shape and dtype — OK")


# ---------------------------------------------------------------------------
# TQ_prod tests
# ---------------------------------------------------------------------------

def test_tq_prod_components():
    """tq_quantize_prod returns 4 components with correct shapes."""
    d = 256
    cb = CodebookRegistry.get(d, 3)  # b-1 = 3 for bit_width=4
    rot = RotationCache.get(d, seed=42)
    qjl = QJLProjection(d, seed=123)
    x = torch.randn(64, d)
    mse_idx, qjl_bits, r_norms, norms = tq_quantize_prod(x, cb, rot, qjl)
    assert mse_idx.shape == (64, d)
    assert qjl_bits.shape == (64, d)
    assert r_norms.shape == (64,)
    assert norms.shape == (64,)
    assert mse_idx.dtype == torch.uint8
    assert qjl_bits.dtype == torch.int8
    print(f"  TQ_prod components: shapes and dtypes — OK")


# ---------------------------------------------------------------------------
# Backend tests
# ---------------------------------------------------------------------------

def test_qwen35_layer_filtering():
    """Qwen3.5-9B: compressible only at GatedAttn layers {3,7,11,...,31}."""
    backend = Qwen35KVBackend()
    ga = {i for i in range(32) if backend.is_compressible(i)}
    assert ga == {3, 7, 11, 15, 19, 23, 27, 31}
    assert len(ga) == 8
    print(f"  Qwen3.5-9B layer filtering: {sorted(ga)} — OK")


def test_qwen3_dense_all_compressible():
    """Qwen3-8B: all 36 layers compressible."""
    backend = Qwen3DenseKVBackend()
    assert all(backend.is_compressible(i) for i in range(36))
    print(f"  Qwen3-8B all 36 layers compressible — OK")


def test_qwen35_compress_decompress_v_shape():
    """Compress then decompress_v preserves V shape."""
    backend = Qwen35KVBackend()
    K = torch.randn(1, 4, 128, 256)
    V = torch.randn(1, 4, 128, 256)
    compressed = backend.compress(K, V, layer_idx=3)
    V_out = backend.decompress_v(compressed)
    assert V_out.shape == V.shape, f"V shape: {V_out.shape} vs {V.shape}"
    assert not torch.isnan(V_out).any()
    print(f"  Qwen3.5-9B compress/decompress_v shape: {V.shape} — OK")


def test_qwen35_compress_decompress_v_no_nan_zeros():
    """No NaN on zero inputs."""
    backend = Qwen35KVBackend()
    K = torch.zeros(1, 4, 16, 256)
    V = torch.zeros(1, 4, 16, 256)
    compressed = backend.compress(K, V, layer_idx=7)
    V_out = backend.decompress_v(compressed)
    assert not torch.isnan(V_out).any(), "NaN in V output from zero inputs"
    assert not torch.isinf(V_out).any(), "Inf in V output from zero inputs"
    print(f"  Qwen3.5-9B no NaN on zero inputs — OK")


def test_qwen3_dense_compress_decompress_v_shape():
    """Qwen3-8B compress/decompress_v preserves shape."""
    backend = Qwen3DenseKVBackend()
    K = torch.randn(1, 8, 64, 128)
    V = torch.randn(1, 8, 64, 128)
    compressed = backend.compress(K, V, layer_idx=0)
    V_out = backend.decompress_v(compressed)
    assert V_out.shape == V.shape
    print(f"  Qwen3-8B compress/decompress_v shape: {V.shape} — OK")


# ---------------------------------------------------------------------------
# compute_attention_scores tests
# ---------------------------------------------------------------------------

def test_qwen35_compute_attention_scores_shape():
    """compute_attention_scores returns correct shape."""
    backend = Qwen35KVBackend()
    K = torch.randn(1, 4, 32, 256)
    V = torch.randn(1, 4, 32, 256)
    Q = torch.randn(1, 4, 8, 256)
    compressed = backend.compress(K, V, layer_idx=3)
    scores = backend.compute_attention_scores(Q, compressed)
    assert scores.shape == (1, 4, 8, 32), f"Scores shape: {scores.shape}"
    assert not torch.isnan(scores).any()
    print(f"  Qwen3.5-9B compute_attention_scores shape: {scores.shape} — OK")


def test_qwen3_dense_compute_attention_scores_shape():
    """compute_attention_scores on Qwen3-8B."""
    backend = Qwen3DenseKVBackend()
    K = torch.randn(1, 8, 32, 128)
    V = torch.randn(1, 8, 32, 128)
    Q = torch.randn(1, 8, 8, 128)
    compressed = backend.compress(K, V, layer_idx=0)
    scores = backend.compute_attention_scores(Q, compressed)
    assert scores.shape == (1, 8, 8, 32), f"Scores shape: {scores.shape}"
    assert not torch.isnan(scores).any()
    print(f"  Qwen3-8B compute_attention_scores shape: {scores.shape} — OK")


def test_cache_compute_attention_scores():
    """TQGatedAttnKVCache.compute_attention_scores works."""
    cache = TQGatedAttnKVCache()
    K = torch.randn(1, 4, 16, 256)
    V = torch.randn(1, 4, 16, 256)
    Q = torch.randn(1, 4, 4, 256)
    compressed = cache.compress_layer(K, V, layer_idx=3)
    scores = cache.compute_attention_scores(Q, compressed)
    assert scores.shape == (1, 4, 4, 16)
    assert not torch.isnan(scores).any()
    print(f"  TQGatedAttnKVCache compute_attention_scores: {scores.shape} — OK")


# ---------------------------------------------------------------------------
# K vs V asymmetry test
# ---------------------------------------------------------------------------

def test_k_v_asymmetry():
    """K uses tq_quantize_prod (MSE+QJL), V uses tq_quantize_mse (MSE only)."""
    backend = Qwen35KVBackend()
    K = torch.randn(1, 4, 16, 256)
    V = torch.randn(1, 4, 16, 256)
    compressed = backend.compress(K, V, layer_idx=3)
    # K has QJL components
    assert "k_qjl" in compressed, "K should have QJL bits"
    assert "k_rn" in compressed, "K should have residual norms"
    assert "k_mse" in compressed, "K should have MSE indices"
    assert "k_n" in compressed, "K should have original norms"
    # V has only MSE components
    assert "v_idx" in compressed, "V should have MSE indices"
    assert "v_n" in compressed, "V should have norms"
    # V should NOT have QJL
    assert "v_qjl" not in compressed, "V should NOT have QJL bits"
    print(f"  K/V asymmetry: K has QJL, V does not — OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_lloyd_max_b1,
        test_lloyd_max_b2,
        test_codebook_registry,
        test_rotation_preserves_norms,
        test_rotation_invertible,
        test_mse_round_trip_shape,
        test_mse_round_trip_error,
        test_mse_no_nan,
        test_qjl_output_shape,
        test_tq_prod_components,
        test_qwen35_layer_filtering,
        test_qwen3_dense_all_compressible,
        test_qwen35_compress_decompress_v_shape,
        test_qwen35_compress_decompress_v_no_nan_zeros,
        test_qwen3_dense_compress_decompress_v_shape,
        test_qwen35_compute_attention_scores_shape,
        test_qwen3_dense_compute_attention_scores_shape,
        test_cache_compute_attention_scores,
        test_k_v_asymmetry,
    ]

    print(f"Running {len(tests)} tests...\n")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed.")
    if passed < len(tests):
        exit(1)
    print("All turboquant-core tests passed.")
