"""
TurboQuant integration for Qwen3.5-9B (probe-verified).

Key probe findings for TQ:
  - DeltaNet state is OPAQUE: Qwen3_5DynamicCache returns None for DeltaNet layers
  - Only 8 GatedAttn layers have KV cache: K/V shape [batch, 4 heads, seq, 256]
  - TQ can only compress the GatedAttn KV cache (not DeltaNet internal state)
  - KV head_dim is 256 (not 128), num_kv_heads is 4 (not 8)
"""

import math

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Codebook registry
# ---------------------------------------------------------------------------

@dataclass
class TQCodebook:
    dimension: int
    bit_width: int
    centroids: torch.Tensor
    boundaries: torch.Tensor
    mse_per_coord: float


def _lloyd_max_gaussian(b: int, max_iter: int = 200, tol: float = 1e-12):
    """Compute Lloyd-Max optimal quantizer for the standard normal distribution.

    Returns (centroids, mse_per_coord) where centroids is a sorted numpy array
    of 2^b reproduction levels minimizing E[(X - Q(X))^2] for X ~ N(0,1).
    """
    n_levels = 1 << b
    # Initialize centroids at evenly-spaced quantiles
    quantiles = np.linspace(0, 1, n_levels + 2)[1:-1]
    centroids = norm.ppf(quantiles)

    for _ in range(max_iter):
        # Boundaries = midpoints between adjacent centroids
        boundaries = np.concatenate([[-np.inf],
                                     (centroids[:-1] + centroids[1:]) / 2.0,
                                     [np.inf]])
        # Update centroids as conditional expectations E[X | b_i < X < b_{i+1}]
        new_centroids = np.empty(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            # E[X | lo < X < hi] = (phi(lo) - phi(hi)) / (Phi(hi) - Phi(lo))
            p = norm.cdf(hi) - norm.cdf(lo)
            if p < 1e-15:
                new_centroids[i] = (lo + hi) / 2.0
            else:
                new_centroids[i] = (norm.pdf(lo) - norm.pdf(hi)) / p

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    # Compute MSE per coordinate: E[(X - Q(X))^2]
    boundaries = np.concatenate([[-np.inf],
                                 (centroids[:-1] + centroids[1:]) / 2.0,
                                 [np.inf]])
    mse = 0.0
    for i in range(n_levels):
        lo, hi = boundaries[i], boundaries[i + 1]
        p = norm.cdf(hi) - norm.cdf(lo)
        if p < 1e-15:
            continue
        c = centroids[i]
        # E[X|lo<X<hi] * P = phi(lo) - phi(hi), but x*phi(x)->0 as x->±inf
        phi_lo = norm.pdf(lo) if np.isfinite(lo) else 0.0
        phi_hi = norm.pdf(hi) if np.isfinite(hi) else 0.0
        ex_p = phi_lo - phi_hi
        # E[X^2|lo<X<hi]*P = P + lo*phi(lo) - hi*phi(hi)
        lo_phi_lo = lo * phi_lo if np.isfinite(lo) else 0.0
        hi_phi_hi = hi * phi_hi if np.isfinite(hi) else 0.0
        ex2_p = p + lo_phi_lo - hi_phi_hi
        mse += ex2_p - 2 * c * ex_p + c * c * p

    centroids.sort()
    return centroids, float(mse)


class CodebookRegistry:
    _cache: dict = {}

    @classmethod
    def get(cls, d: int, b: int, device=torch.device("cpu")) -> TQCodebook:
        key = (d, b)
        if key not in cls._cache:
            raw_centroids, mse_per_coord = _lloyd_max_gaussian(b)
            # Scale for the rotated unit-vector distribution: each coordinate ~ N(0, 1/d)
            scale = 1.0 / (d ** 0.5)
            centroids = torch.tensor(raw_centroids * scale, dtype=torch.float32)
            boundaries = torch.empty(len(centroids) + 1)
            boundaries[0] = -1.0
            boundaries[-1] = 1.0
            for i in range(len(centroids) - 1):
                boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0
            cls._cache[key] = TQCodebook(d, b, centroids, boundaries, mse_per_coord / d)

        cb = cls._cache[key]
        if device != torch.device("cpu"):
            return TQCodebook(cb.dimension, cb.bit_width,
                              cb.centroids.to(device), cb.boundaries.to(device), cb.mse_per_coord)
        return cb


# ---------------------------------------------------------------------------
# Rotation cache
# ---------------------------------------------------------------------------

class RotationCache:
    _cache: dict = {}

    @classmethod
    def get(cls, d: int, seed: int = 42, device=torch.device("cpu")):
        key = (d, seed)
        if key not in cls._cache:
            rng = np.random.default_rng(seed)
            d_pad = 1 << (d - 1).bit_length()
            signs = torch.tensor(rng.choice([-1.0, 1.0], size=d_pad), dtype=torch.float32)
            H = cls._hadamard(d_pad)
            cls._cache[key] = {"d": d, "d_padded": d_pad, "signs": signs, "H": H}
        e = cls._cache[key]
        if device != torch.device("cpu"):
            return {"d": e["d"], "d_padded": e["d_padded"],
                    "signs": e["signs"].to(device), "H": e["H"].to(device)}
        return e

    @staticmethod
    def _hadamard(n):
        if n == 1: return torch.tensor([[1.0]])
        h = RotationCache._hadamard(n // 2)
        return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)


# ---------------------------------------------------------------------------
# Core quantization
# ---------------------------------------------------------------------------

def tq_rotate(x, rot):
    d, dp, signs, H = rot["d"], rot["d_padded"], rot["signs"], rot["H"]
    if x.shape[-1] < dp:
        x = torch.nn.functional.pad(x, (0, dp - x.shape[-1]))
    return ((x * signs) @ H.T / (dp ** 0.5))[..., :d]

def tq_rotate_inv(y, rot):
    d, dp, signs, H = rot["d"], rot["d_padded"], rot["signs"], rot["H"]
    if y.shape[-1] < dp:
        y = torch.nn.functional.pad(y, (0, dp - y.shape[-1]))
    return ((y @ H.T / (dp ** 0.5)) * signs)[..., :d]

def tq_quantize_mse(x, cb, rot):
    norms = torch.linalg.norm(x, dim=-1, keepdim=True)
    y = tq_rotate(x / (norms + 1e-12), rot)
    y = y.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    indices = torch.searchsorted(cb.boundaries[1:-1].contiguous(), y.contiguous()).to(torch.uint8)
    return indices, norms.squeeze(-1)

def tq_dequantize_mse(indices, norms, cb, rot):
    y_hat = cb.centroids[indices.long()]
    return tq_rotate_inv(y_hat, rot) * norms.unsqueeze(-1)


# ---------------------------------------------------------------------------
# QJL
# ---------------------------------------------------------------------------

class QJLProjection:
    def __init__(self, d, seed=123, device=torch.device("cpu")):
        self.d = d
        self.S = torch.randn(d, d, generator=torch.Generator().manual_seed(seed), device=device)

    def quantize(self, x):
        return (x @ self.S.T).sign().to(torch.int8)

    def to(self, device):
        self.S = self.S.to(device)
        return self


def tq_quantize_prod(x, cb, rot, qjl):
    norms = torch.linalg.norm(x, dim=-1)
    x_unit = x / (norms.unsqueeze(-1) + 1e-12)
    mse_idx, _ = tq_quantize_mse(x_unit, cb, rot)
    x_hat = tq_dequantize_mse(mse_idx, torch.ones_like(norms), cb, rot)
    r = x_unit - x_hat
    r_norms = torch.linalg.norm(r, dim=-1)
    qjl_bits = qjl.quantize(r / (r_norms.unsqueeze(-1) + 1e-12))
    return mse_idx, qjl_bits, r_norms, norms


# ---------------------------------------------------------------------------
# High-level wrappers
# ---------------------------------------------------------------------------

class TQActivationCheckpoint:
    def __init__(self, d, bit_width=4, seed=42, device=torch.device("cpu")):
        self.cb = CodebookRegistry.get(d, bit_width, device)
        self.rot = RotationCache.get(d, seed, device)
        self._idx = self._norms = self._shape = None

    def save(self, x):
        self._shape = x.shape
        flat = x.reshape(-1, x.shape[-1])
        self._idx, self._norms = tq_quantize_mse(flat, self.cb, self.rot)

    def restore(self):
        return tq_dequantize_mse(self._idx, self._norms, self.cb, self.rot).reshape(self._shape)


class TQLoRAStorage:
    def __init__(self, d_in, d_out, rank, bit_width=4, seed=42, device=torch.device("cpu")):
        self.d_in, self.d_out, self.rank = d_in, d_out, rank
        self.cb_in = CodebookRegistry.get(d_in, bit_width, device)
        self.cb_out = CodebookRegistry.get(d_out, bit_width, device)
        self.rot_in = RotationCache.get(d_in, seed, device)
        self.rot_out = RotationCache.get(d_out, seed + 1, device)

    def compress(self, A, B):
        Ai, An = tq_quantize_mse(A, self.cb_in, self.rot_in)
        Bi, Bn = tq_quantize_mse(B.T, self.cb_out, self.rot_out)
        return {"A_idx": Ai, "A_n": An, "B_idx": Bi, "B_n": Bn}

    def decompress(self, s):
        A = tq_dequantize_mse(s["A_idx"], s["A_n"], self.cb_in, self.rot_in)
        B = tq_dequantize_mse(s["B_idx"], s["B_n"], self.cb_out, self.rot_out).T
        return A, B


# ---------------------------------------------------------------------------
# Hybrid KV cache compression (probe-verified)
# ---------------------------------------------------------------------------

class TQGatedAttnKVCache:
    """
    Compresses KV cache for the 8 GatedAttn layers in Qwen3.5-9B.

    Probe findings:
      - Qwen3_5DynamicCache.key_cache[i] is None for DeltaNet layers
      - For GatedAttn layers: K/V shape [batch, 4 heads, seq_len, 256]
      - Only 8 of 32 layers have compressible KV cache

    Strategy:
      K → TQ_prod (MSE + QJL) — K participates in softmax(QK^T)
      V → TQ_MSE only — V is weighted-averaged by attention scores
    """

    def __init__(self, num_layers=32, interval=4,
                 kv_head_dim=256, num_kv_heads=4,
                 bit_width=4, seed=42, device=torch.device("cpu")):
        self.ga_indices = {i for i in range(num_layers) if (i + 1) % interval == 0}
        self.kv_head_dim = kv_head_dim

        # K: (b-1)-bit MSE + 1-bit QJL
        self.k_cb = CodebookRegistry.get(kv_head_dim, bit_width - 1, device)
        self.k_rot = RotationCache.get(kv_head_dim, seed, device)
        self.k_qjl = QJLProjection(kv_head_dim, seed=seed + 50, device=device)

        # V: b-bit MSE only (no QJL — V is not inner-producted against Q)
        self.v_cb = CodebookRegistry.get(kv_head_dim, bit_width, device)
        self.v_rot = RotationCache.get(kv_head_dim, seed + 100, device)

    def is_gated_attn(self, layer_idx):
        return layer_idx in self.ga_indices

    def compress_layer(self, K, V, layer_idx):
        """K, V shape: [batch, num_heads, seq_len, head_dim]"""
        assert self.is_gated_attn(layer_idx)
        b, nh, sl, hd = K.shape
        Kf = K.reshape(b * nh * sl, hd)
        Vf = V.reshape(b * nh * sl, hd)

        k_mse, k_qjl, k_rnorms, k_norms = tq_quantize_prod(Kf, self.k_cb, self.k_rot, self.k_qjl)
        v_idx, v_norms = tq_quantize_mse(Vf, self.v_cb, self.v_rot)

        return {"k_mse": k_mse, "k_qjl": k_qjl, "k_rn": k_rnorms, "k_n": k_norms,
                "v_idx": v_idx, "v_n": v_norms, "shape": K.shape}

    def decompress_v(self, compressed):
        """Decompress V (MSE-only) for attention output computation."""
        s = compressed["shape"]
        Vf = tq_dequantize_mse(compressed["v_idx"], compressed["v_n"], self.v_cb, self.v_rot)
        return Vf.reshape(s)

    def compute_attention_scores(self, Q, compressed):
        """Compute unbiased Q @ K^T from fresh Q and compressed K using QJL correction.

        Q shape: [batch, num_heads, q_len, head_dim]
        Returns: [batch, num_heads, q_len, kv_len]
        """
        b, nh, kv_len, hd = compressed["shape"]
        q_len = Q.shape[2]

        # Stage 1: Q @ K_mse^T (biased, from MSE reconstruction)
        K_mse = tq_dequantize_mse(
            compressed["k_mse"], compressed["k_n"], self.k_cb, self.k_rot
        ).reshape(compressed["shape"])
        scores_mse = Q @ K_mse.transpose(-2, -1)

        # Stage 2: QJL bias correction on the residual
        # Reshape to [b*nh, seq, hd] for per-head batched matmul
        Q_flat = Q.reshape(b * nh * q_len, hd)
        Q_qjl = self.k_qjl.quantize(Q_flat).reshape(b * nh, q_len, hd)
        k_qjl = compressed["k_qjl"].reshape(b * nh, kv_len, hd)

        # [b*nh, q_len, hd] @ [b*nh, hd, kv_len] -> [b*nh, q_len, kv_len]
        correction = Q_qjl.float() @ k_qjl.float().transpose(-2, -1)
        correction = correction * (math.pi / (2 * hd))
        # k_rn and k_n are [b*nh*kv_len] -> [b*nh, 1, kv_len] for broadcasting
        correction = correction * compressed["k_rn"].reshape(b * nh, 1, kv_len)
        correction = correction * compressed["k_n"].reshape(b * nh, 1, kv_len)
        correction = correction.reshape(b, nh, q_len, kv_len)

        return scores_mse + correction
