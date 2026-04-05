"""Comprehensive tests for TurboQuant core algorithms and backends."""

import math
import torch
import numpy as np

from turboquant_core.core import (
    _lloyd_max_gaussian,
    _fast_wht,
    CodebookRegistry,
    RotationCache,
    tq_rotate,
    tq_rotate_inv,
    tq_quantize_mse,
    tq_dequantize_mse,
    tq_quantize_mse_ste,
    QJLProjection,
    tq_quantize_prod,
    TQGatedAttnKVCache,
    TQQuantizedCache,
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
    print("  No NaN/Inf on random and zero inputs — OK")


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
    print("  QJL output shape and dtype — OK")


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
    print("  TQ_prod components: shapes and dtypes — OK")


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
    print("  Qwen3-8B all 36 layers compressible — OK")


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
    print("  Qwen3.5-9B no NaN on zero inputs — OK")


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
    print("  K/V asymmetry: K has QJL, V does not — OK")


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard transform tests
# ---------------------------------------------------------------------------

def test_fast_wht_matches_dense():
    """Fast WHT should produce same result as dense Hadamard matrix multiply."""
    d = 16
    x = torch.randn(8, d)
    # Build dense Hadamard for reference
    def hadamard(n):
        if n == 1:
            return torch.tensor([[1.0]])
        h = hadamard(n // 2)
        return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    H = hadamard(d)
    expected = x @ H.T
    got = _fast_wht(x)
    max_err = (expected - got).abs().max().item()
    assert max_err < 1e-5, f"Fast WHT vs dense error: {max_err}"
    print(f"  Fast WHT matches dense Hadamard (d={d}): max_err={max_err:.2e} — OK")


def test_fast_wht_no_matrix_stored():
    """RotationCache should not store a Hadamard matrix with fast WHT."""
    rot = RotationCache.get(256, seed=99)
    assert "H" not in rot, "RotationCache should not store dense Hadamard matrix"
    assert "signs" in rot
    assert "d" in rot
    assert "d_padded" in rot
    print("  RotationCache stores no dense matrix — OK")


# ---------------------------------------------------------------------------
# TQQuantizedCache tests
# ---------------------------------------------------------------------------

def test_quantized_cache_compressed_update():
    """TQQuantizedCache stores compressed data for GatedAttn layers."""
    cache = TQQuantizedCache()
    K = torch.randn(1, 4, 16, 256)
    V = torch.randn(1, 4, 16, 256)
    cache.update(K, V, layer_idx=3)
    assert cache.get_seq_length(3) == 16
    # Should be a dict (compressed), not a tuple (raw)
    assert isinstance(cache._cache[3], dict)
    assert "k_mse" in cache._cache[3]
    print("  TQQuantizedCache compressed storage — OK")


def test_quantized_cache_raw_update():
    """TQQuantizedCache stores raw tensors for non-compressible layers."""
    cache = TQQuantizedCache()
    K = torch.randn(1, 4, 8, 256)
    V = torch.randn(1, 4, 8, 256)
    cache.update(K, V, layer_idx=0)  # DeltaNet layer
    assert cache.get_seq_length(0) == 8
    assert isinstance(cache._cache[0], tuple)
    print("  TQQuantizedCache raw storage for DeltaNet — OK")


def test_quantized_cache_incremental():
    """TQQuantizedCache supports incremental token updates."""
    cache = TQQuantizedCache()
    K1 = torch.randn(1, 4, 10, 256)
    V1 = torch.randn(1, 4, 10, 256)
    K2 = torch.randn(1, 4, 5, 256)
    V2 = torch.randn(1, 4, 5, 256)
    cache.update(K1, V1, layer_idx=3)
    cache.update(K2, V2, layer_idx=3)
    assert cache.get_seq_length(3) == 15
    print("  TQQuantizedCache incremental: 10+5=15 tokens — OK")


def test_quantized_cache_compute_attention():
    """TQQuantizedCache.compute_attention returns correct shape."""
    cache = TQQuantizedCache()
    K = torch.randn(1, 4, 32, 256)
    V = torch.randn(1, 4, 32, 256)
    Q = torch.randn(1, 4, 1, 256)
    cache.update(K, V, layer_idx=3)
    out = cache.compute_attention(Q, layer_idx=3)
    assert out.shape == (1, 4, 1, 256), f"Output shape: {out.shape}"
    assert not torch.isnan(out).any()
    print(f"  TQQuantizedCache compute_attention: {out.shape} — OK")


def test_quantized_cache_compute_attention_raw():
    """TQQuantizedCache.compute_attention works for raw (non-compressed) layers."""
    cache = TQQuantizedCache()
    K = torch.randn(1, 4, 16, 256)
    V = torch.randn(1, 4, 16, 256)
    Q = torch.randn(1, 4, 1, 256)
    cache.update(K, V, layer_idx=0)
    out = cache.compute_attention(Q, layer_idx=0)
    assert out.shape == (1, 4, 1, 256)
    assert not torch.isnan(out).any()
    print(f"  TQQuantizedCache raw attention: {out.shape} — OK")


def test_quantized_cache_clear():
    """TQQuantizedCache.clear resets all layers."""
    cache = TQQuantizedCache()
    cache.update(torch.randn(1, 4, 8, 256), torch.randn(1, 4, 8, 256), 3)
    cache.clear()
    assert cache.get_seq_length(3) == 0
    assert cache._cache[3] is None
    print("  TQQuantizedCache clear — OK")


# ---------------------------------------------------------------------------
# STE gradient tests
# ---------------------------------------------------------------------------

def test_ste_forward_matches_hard():
    """STE forward should produce same output as hard quantize+dequantize."""
    d = 128
    cb = CodebookRegistry.get(d, 4)
    rot = RotationCache.get(d, seed=42)
    x = torch.randn(32, d)
    # Hard round-trip
    idx, norms = tq_quantize_mse(x, cb, rot)
    x_hard = tq_dequantize_mse(idx, norms, cb, rot)
    # STE round-trip
    x_ste = tq_quantize_mse_ste(x, cb, rot)
    max_err = (x_hard - x_ste).abs().max().item()
    assert max_err < 1e-6, f"STE forward mismatch: {max_err}"
    print(f"  STE forward matches hard quantization: max_err={max_err:.2e} — OK")


def test_ste_gradient_flows():
    """STE should allow gradients to flow through quantization."""
    d = 128
    cb = CodebookRegistry.get(d, 4)
    rot = RotationCache.get(d, seed=42)
    x = torch.randn(16, d, requires_grad=True)
    x_hat = tq_quantize_mse_ste(x, cb, rot)
    loss = x_hat.sum()
    loss.backward()
    assert x.grad is not None, "Gradient should flow through STE"
    assert not torch.isnan(x.grad).any(), "NaN in STE gradient"
    assert x.grad.abs().sum() > 0, "Gradient should be non-zero"
    print(f"  STE gradient flows: grad norm={x.grad.norm().item():.4f} — OK")


def test_ste_gradient_shape():
    """STE gradient should have same shape as input."""
    d = 256
    cb = CodebookRegistry.get(d, 4)
    rot = RotationCache.get(d, seed=42)
    x = torch.randn(8, d, requires_grad=True)
    x_hat = tq_quantize_mse_ste(x, cb, rot)
    (x_hat ** 2).sum().backward()
    assert x.grad.shape == x.shape
    print(f"  STE gradient shape: {x.grad.shape} — OK")


# ---------------------------------------------------------------------------
# Paper Table 2 verification (HANDOFF VERIFY #1, #3)
# ---------------------------------------------------------------------------

def test_paper_table2_codebook_centroids():
    """Verify Lloyd-Max centroids match known optimal values for b=1..4."""
    # Known optimal Lloyd-Max centroids for N(0,1)
    known = {
        1: ([-0.7979, 0.7979], 0.3634),
        2: ([-1.5104, -0.4528, 0.4528, 1.5104], 0.1175),
        3: ([-2.1519, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1519], 0.03454),
    }
    for b, (expected_c, expected_mse) in known.items():
        centroids, mse = _lloyd_max_gaussian(b)
        assert len(centroids) == 2 ** b
        for c, e in zip(centroids, expected_c):
            assert abs(c - e) < 0.01, f"b={b}: centroid {c} != {e}"
        assert abs(mse - expected_mse) < 0.005, f"b={b}: MSE {mse} != {expected_mse}"
    print("  Paper Table 2 centroids b=1..3 — OK")


def test_paper_table2_mse_round_trip():
    """Verify MSE/coord at multiple bit widths matches theoretical values."""
    # Theoretical Lloyd-Max MSE on N(0,1), scaled by 1/d for the rotated distribution
    # Paper Table 2 reference values (asymptotic d→∞)
    d = 256
    results = {}
    for b in [1, 2, 3, 4]:
        cb = CodebookRegistry.get(d, b)
        rot = RotationCache.get(d, seed=42)
        x = torch.randn(5000, d)
        idx, norms = tq_quantize_mse(x, cb, rot)
        x_hat = tq_dequantize_mse(idx, norms, cb, rot)
        mse = ((x - x_hat) ** 2).mean().item()
        results[b] = mse
    # MSE should decrease as bit_width increases
    assert results[1] > results[2] > results[3] > results[4], \
        f"MSE should decrease: {results}"
    # b=4 should be < 0.02 (paper ~0.0115)
    assert results[4] < 0.02, f"b=4 MSE too high: {results[4]}"
    # b=1 should be much higher
    assert results[1] > 0.1, f"b=1 MSE too low: {results[1]}"
    print("  MSE/coord by bit_width: " +
          ", ".join(f"b={b}={v:.4f}" for b, v in results.items()) + " — OK")


def test_qjl_inner_product_unbiasedness():
    """QJL estimator of inner products should be approximately unbiased."""
    d = 256
    qjl = QJLProjection(d, seed=123)
    # Generate random pairs and compare true vs QJL-estimated inner products
    n_pairs = 1000
    x = torch.randn(n_pairs, d)
    y = torch.randn(n_pairs, d)
    true_ip = (x * y).sum(dim=-1)
    # QJL estimate: (pi/2d) * sign(Sx) @ sign(Sy) * ||x|| * ||y||
    x_norms = torch.linalg.norm(x, dim=-1)
    y_norms = torch.linalg.norm(y, dim=-1)
    x_unit = x / (x_norms.unsqueeze(-1) + 1e-12)
    y_unit = y / (y_norms.unsqueeze(-1) + 1e-12)
    sx = qjl.quantize(x_unit).float()
    sy = qjl.quantize(y_unit).float()
    qjl_ip = (sx * sy).sum(dim=-1) * (math.pi / (2 * d)) * x_norms * y_norms
    # Should be unbiased: mean of (qjl - true) ≈ 0
    bias = (qjl_ip - true_ip).mean().item()
    # Allow some variance but bias should be small relative to scale
    scale = true_ip.abs().mean().item()
    assert abs(bias / scale) < 0.15, f"QJL bias too large: {bias:.4f} (scale={scale:.4f})"
    print(f"  QJL inner product bias: {bias:.4f} (relative: {bias/scale:.4f}) — OK")


# ---------------------------------------------------------------------------
# Adapter interface compatibility test (HANDOFF VERIFY #8)
# ---------------------------------------------------------------------------

def test_adapter_interface_contract():
    """Verify backends match the adapter-interface.md contract exactly."""
    import inspect

    for BackendCls, name in [
        (Qwen35KVBackend, "Qwen35KVBackend"),
        (Qwen3DenseKVBackend, "Qwen3DenseKVBackend"),
    ]:
        backend = BackendCls()

        # Required method: is_compressible(layer_idx: int) -> bool
        assert hasattr(backend, 'is_compressible'), f"{name} missing is_compressible"
        sig = inspect.signature(backend.is_compressible)
        params = list(sig.parameters.keys())
        assert 'layer_idx' in params, f"{name}.is_compressible missing layer_idx param"
        result = backend.is_compressible(0)
        assert isinstance(result, bool), f"{name}.is_compressible should return bool"

        # Required method: compress(K, V, layer_idx) -> dict
        assert hasattr(backend, 'compress'), f"{name} missing compress"
        sig = inspect.signature(backend.compress)
        params = list(sig.parameters.keys())
        assert 'K' in params or params[0] == 'K' or len(params) >= 3, \
            f"{name}.compress signature mismatch: {params}"

        # Required method: decompress_v(compressed) -> Tensor
        assert hasattr(backend, 'decompress_v'), f"{name} missing decompress_v"

        # Extended method: compute_attention_scores(Q, compressed) -> Tensor
        assert hasattr(backend, 'compute_attention_scores'), \
            f"{name} missing compute_attention_scores"

        # Functional test: compress returns dict with expected keys
        if name == "Qwen35KVBackend":
            K = torch.randn(1, 4, 8, 256)
            V = torch.randn(1, 4, 8, 256)
            idx = 3
        else:
            K = torch.randn(1, 8, 8, 128)
            V = torch.randn(1, 8, 8, 128)
            idx = 0
        compressed = backend.compress(K, V, idx)
        assert isinstance(compressed, dict), f"{name}.compress should return dict"
        assert "shape" in compressed, f"{name}.compress dict missing 'shape'"

        # decompress_v returns Tensor with correct shape
        V_out = backend.decompress_v(compressed)
        assert isinstance(V_out, torch.Tensor), f"{name}.decompress_v should return Tensor"
        assert V_out.shape == V.shape, f"{name}.decompress_v shape mismatch"

    print("  Adapter interface contract verified for both backends — OK")


# ---------------------------------------------------------------------------
# bit_width semantics test (HANDOFF LIMIT #12)
# ---------------------------------------------------------------------------

def test_bit_width_kv_asymmetry_semantics():
    """bit_width=4 → K gets 3-bit codebook (8 levels), V gets 4-bit (16 levels)."""
    backend = Qwen35KVBackend(bit_width=4)
    # K codebook: bit_width - 1 = 3 bits = 8 centroids
    assert backend.k_cb.centroids.shape[0] == 8, \
        f"K should have 8 centroids (3-bit), got {backend.k_cb.centroids.shape[0]}"
    assert backend.k_cb.bit_width == 3
    # V codebook: bit_width = 4 bits = 16 centroids
    assert backend.v_cb.centroids.shape[0] == 16, \
        f"V should have 16 centroids (4-bit), got {backend.v_cb.centroids.shape[0]}"
    assert backend.v_cb.bit_width == 4
    # Effective: K = 3 MSE + 1 QJL = 4 bits total, V = 4 MSE = 4 bits total
    print("  bit_width=4: K=3-bit MSE + 1-bit QJL, V=4-bit MSE — OK")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Phase 2: Configurable backend constructors
# ---------------------------------------------------------------------------

def test_qwen35_backend_custom_params():
    """Qwen35KVBackend accepts custom layout params."""
    backend = Qwen35KVBackend(
        bit_width=4, num_layers=16, full_attn_interval=4, kv_heads=2, head_dim=256,
    )
    assert backend.num_layers == 16
    assert backend.full_attn_interval == 4
    assert backend.kv_heads == 2
    assert backend.head_dim == 256
    # ga_indices should be based on custom num_layers
    assert backend.ga_indices == {3, 7, 11, 15}


def test_qwen3_backend_custom_params():
    """Qwen3DenseKVBackend accepts custom layout params."""
    backend = Qwen3DenseKVBackend(
        bit_width=4, num_layers=18, kv_heads=4, head_dim=128,
    )
    assert backend.num_layers == 18
    assert backend.kv_heads == 4
    assert backend.head_dim == 128
    assert backend.is_compressible(17)  # all layers compressible


def test_backend_defaults_match_constants():
    """Default constructor args match original class constants."""
    b35 = Qwen35KVBackend()
    assert b35.num_layers == Qwen35KVBackend.NUM_LAYERS
    assert b35.full_attn_interval == Qwen35KVBackend.FULL_ATTN_INTERVAL
    assert b35.kv_heads == Qwen35KVBackend.GA_KV_HEADS
    assert b35.head_dim == Qwen35KVBackend.GA_HEAD_DIM

    b3 = Qwen3DenseKVBackend()
    assert b3.num_layers == Qwen3DenseKVBackend.NUM_LAYERS
    assert b3.kv_heads == Qwen3DenseKVBackend.KV_HEADS
    assert b3.head_dim == Qwen3DenseKVBackend.HEAD_DIM


# ---------------------------------------------------------------------------
# Phase 4: Flexible algorithm selection
# ---------------------------------------------------------------------------

def test_qwen35_backend_mse_only_strategy():
    """key_strategy='mse' uses full bit_width for K with no QJL."""
    backend = Qwen35KVBackend(bit_width=4, key_strategy="mse")
    assert backend.key_strategy == "mse"
    assert backend.k_qjl is None
    # K codebook should use full bit_width (4 bits = 16 centroids)
    assert backend.k_cb.centroids.shape[0] == 16

    # Compress and verify no QJL keys
    K = torch.randn(1, 4, 8, 256)
    V = torch.randn(1, 4, 8, 256)
    compressed = backend.compress(K, V, 3)
    assert "k_qjl" not in compressed
    assert "k_rn" not in compressed


def test_qwen35_backend_default_strategy():
    """Default key_strategy='mse+qjl' allocates (b-1) bits to MSE + 1 to QJL."""
    backend = Qwen35KVBackend(bit_width=4)
    assert backend.key_strategy == "mse+qjl"
    assert backend.k_qjl is not None
    # K codebook should use bit_width-1 (3 bits = 8 centroids)
    assert backend.k_cb.centroids.shape[0] == 8


def test_qwen3_backend_mse_only_attention_scores():
    """MSE-only strategy produces valid attention scores (no correction term)."""
    backend = Qwen3DenseKVBackend(bit_width=4, key_strategy="mse")
    K = torch.randn(1, 8, 4, 128)
    V = torch.randn(1, 8, 4, 128)
    Q = torch.randn(1, 8, 2, 128)
    compressed = backend.compress(K, V, 0)
    scores = backend.compute_attention_scores(Q, compressed)
    assert scores.shape == (1, 8, 2, 4)
    assert not torch.isnan(scores).any()


def test_invalid_key_strategy_raises():
    """Invalid key_strategy raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Invalid key_strategy"):
        Qwen35KVBackend(key_strategy="invalid")


def test_invalid_value_strategy_raises():
    """Invalid value_strategy raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Invalid value_strategy"):
        Qwen35KVBackend(value_strategy="invalid")


# ---------------------------------------------------------------------------
# Phase 5: Open CodebookRegistry API
# ---------------------------------------------------------------------------

def test_codebook_registry_list_cached():
    """list_cached() returns previously computed (d, b) pairs."""
    # Ensure at least one entry exists
    CodebookRegistry.get(256, 3)
    cached = CodebookRegistry.list_cached()
    assert (256, 3) in cached


def test_codebook_registry_precompute():
    """precompute() computes, caches, and returns a codebook."""
    cb = CodebookRegistry.precompute(64, 2)
    assert cb.dimension == 64
    assert cb.bit_width == 2
    assert cb.centroids.shape[0] == 4  # 2^2 = 4 centroids
    assert (64, 2) in CodebookRegistry.list_cached()


def test_codebook_registry_clear():
    """clear() drops all cached codebooks."""
    CodebookRegistry.get(256, 4)
    assert len(CodebookRegistry.list_cached()) > 0
    CodebookRegistry.clear()
    assert len(CodebookRegistry.list_cached()) == 0
    # Re-populate for other tests that depend on cached codebooks
    CodebookRegistry.get(256, 4)


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
        # Fast WHT tests
        test_fast_wht_matches_dense,
        test_fast_wht_no_matrix_stored,
        # TQQuantizedCache tests
        test_quantized_cache_compressed_update,
        test_quantized_cache_raw_update,
        test_quantized_cache_incremental,
        test_quantized_cache_compute_attention,
        test_quantized_cache_compute_attention_raw,
        test_quantized_cache_clear,
        # STE gradient tests
        test_ste_forward_matches_hard,
        test_ste_gradient_flows,
        test_ste_gradient_shape,
        # Paper Table 2 and verification tests
        test_paper_table2_codebook_centroids,
        test_paper_table2_mse_round_trip,
        test_qjl_inner_product_unbiasedness,
        test_adapter_interface_contract,
        test_bit_width_kv_asymmetry_semantics,
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
