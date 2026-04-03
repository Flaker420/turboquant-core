"""
Hook-in module for patching Qwen3.5-9B to use TurboQuant compressed KV cache.

Usage:
    from transformers import AutoModelForCausalLM
    from turboquant_core.backends.qwen_hook import patch_qwen35_with_tq

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", ...)
    cache = patch_qwen35_with_tq(model, bit_width=4)

    # Now model.generate() uses compressed KV cache automatically.
    # cache.clear() between generations.

This module monkey-patches the GatedAttn layers' attention forward to:
1. Compress K/V into the TQQuantizedCache after projection
2. Use QJL-corrected attention scores for K (instead of raw matmul)
3. Decompress V for the attention output
4. Pass through DeltaNet layers unchanged

Requires: transformers with Qwen3.5 support.
"""

import math
import functools
from typing import Optional

import torch

from ..core import (
    TQQuantizedCache,
    tq_dequantize_mse,
)


def patch_qwen35_with_tq(model, bit_width=4, seed=42, device=None):
    """Patch a Qwen3.5 model to use TurboQuant compressed KV cache.

    Args:
        model: A Qwen3.5 CausalLM model from transformers.
        bit_width: Total bits per value (K gets b-1 MSE + 1 QJL, V gets b MSE).
        seed: Random seed for rotation and QJL matrices.
        device: Device for TQ buffers. Defaults to model's device.

    Returns:
        TQQuantizedCache instance. Call cache.clear() between generations.
    """
    if device is None:
        device = next(model.parameters()).device

    cache = TQQuantizedCache(
        num_layers=32, interval=4,
        kv_head_dim=256, num_kv_heads=4,
        bit_width=bit_width, seed=seed, device=device,
    )

    # Find the attention layers in the model
    layers = _get_model_layers(model)
    if layers is None:
        raise ValueError(
            "Could not find transformer layers in model. "
            "Expected model.model.layers or similar structure."
        )

    for layer_idx, layer in enumerate(layers):
        if not cache.is_compressible(layer_idx):
            continue  # DeltaNet layers: no patching needed

        attn = _get_attention_module(layer)
        if attn is None:
            continue

        # Patch the attention forward
        _patch_attention_forward(attn, cache, layer_idx)

    return cache


def _get_model_layers(model):
    """Extract the list of transformer layers from a HuggingFace model."""
    # Try common attribute paths
    for path in ["model.layers", "transformer.h", "transformer.layers"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue
    return None


def _get_attention_module(layer):
    """Extract the self-attention module from a transformer layer."""
    for attr in ["self_attn", "attn", "attention"]:
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return None


def _patch_attention_forward(attn_module, cache, layer_idx):
    """Patch an attention module to use TQ compressed KV cache.

    The patched forward:
    1. Computes Q, K, V projections as normal
    2. Stores compressed K/V in the TQQuantizedCache
    3. Computes attention using QJL-corrected scores
    4. Returns the attention output
    """
    original_forward = attn_module.forward

    @functools.wraps(original_forward)
    def tq_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        # Get Q, K, V projections by calling the projection layers directly
        bsz, q_len, _ = hidden_states.size()

        # Standard QKV projection (these attributes are standard in HF Qwen models)
        Q = attn_module.q_proj(hidden_states)
        K = attn_module.k_proj(hidden_states)
        V = attn_module.v_proj(hidden_states)

        # Reshape to [batch, heads, seq_len, head_dim]
        num_q_heads = attn_module.num_heads
        num_kv_heads = attn_module.num_key_value_heads
        head_dim = attn_module.head_dim

        Q = Q.view(bsz, q_len, num_q_heads, head_dim).transpose(1, 2)
        K = K.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        V = V.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply rotary embeddings if available
        if hasattr(attn_module, 'rotary_emb'):
            cos, sin = attn_module.rotary_emb(V, position_ids)
            Q, K = _apply_rotary_pos_emb(Q, K, cos, sin)

        # Store compressed K/V in the TQ cache
        cache.update(K, V, layer_idx)

        # Compute attention using the full cache (including previous tokens)
        attn_output = cache.compute_attention(Q, layer_idx)

        # GQA: attn_output is [batch, num_kv_heads, q_len, head_dim]
        # Need to expand if num_q_heads > num_kv_heads
        if num_q_heads != num_kv_heads:
            # For GQA, Q was [batch, num_q_heads, ...] but cache.compute_attention
            # operates at KV head granularity. We need the full Q @ K^T with GQA expansion.
            # Fall back to manual attention with GQA for correctness.
            attn_output = _gqa_attention(Q, cache, layer_idx, num_q_heads, num_kv_heads)

        # Reshape back: [batch, heads, seq_len, head_dim] -> [batch, seq_len, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # Output projection
        attn_output = attn_module.o_proj(attn_output)

        return attn_output, None, past_key_value

    attn_module.forward = tq_forward


def _gqa_attention(Q, cache, layer_idx, num_q_heads, num_kv_heads):
    """Handle grouped-query attention with TQ cache.

    Qwen3.5-9B: 24 Q heads, 4 KV heads → groups of 6 Q heads per KV head.
    """
    entry = cache._cache[layer_idx]
    bsz = Q.shape[0]
    q_len = Q.shape[2]
    head_dim = Q.shape[3]
    kv_len = cache.get_seq_length(layer_idx)
    num_groups = num_q_heads // num_kv_heads

    if cache.is_compressible(layer_idx) and isinstance(entry, dict):
        shape = (bsz, num_kv_heads, kv_len, head_dim)

        # MSE-reconstructed K: [batch, kv_heads, kv_len, head_dim]
        K_mse = tq_dequantize_mse(
            entry["k_mse"], entry["k_n"], cache.k_cb, cache.k_rot
        ).reshape(shape)

        # Expand K for GQA: [batch, kv_heads, kv_len, hd] -> [batch, q_heads, kv_len, hd]
        K_mse_expanded = K_mse.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        K_mse_expanded = K_mse_expanded.reshape(bsz, num_q_heads, kv_len, head_dim)

        scores_mse = Q @ K_mse_expanded.transpose(-2, -1)

        # QJL correction at KV head granularity, then expand
        Q_grouped = Q.reshape(bsz, num_kv_heads, num_groups, q_len, head_dim)
        correction_per_kv = []
        for g in range(num_kv_heads):
            Q_g = Q_grouped[:, g].reshape(bsz * num_groups * q_len, head_dim)
            Q_qjl = cache.k_qjl.quantize(Q_g).reshape(bsz * num_groups, q_len, head_dim)
            k_qjl = entry["k_qjl"].reshape(bsz, kv_len, head_dim)
            # Need per-batch k_qjl for this KV head
            k_qjl_g = k_qjl[:, :, :].unsqueeze(1).expand(-1, num_groups, -1, -1)
            k_qjl_g = k_qjl_g.reshape(bsz * num_groups, kv_len, head_dim)

            corr = Q_qjl.float() @ k_qjl_g.float().transpose(-2, -1)
            corr = corr * (math.pi / (2 * head_dim))
            k_rn_g = entry["k_rn"].reshape(bsz, kv_len).unsqueeze(1).expand(-1, num_groups, -1)
            k_n_g = entry["k_n"].reshape(bsz, kv_len).unsqueeze(1).expand(-1, num_groups, -1)
            corr = corr * k_rn_g.reshape(bsz * num_groups, 1, kv_len)
            corr = corr * k_n_g.reshape(bsz * num_groups, 1, kv_len)
            correction_per_kv.append(corr.reshape(bsz, num_groups, q_len, kv_len))

        correction = torch.stack(correction_per_kv, dim=1)
        correction = correction.reshape(bsz, num_q_heads, q_len, kv_len)

        scores = (scores_mse + correction) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        # Decompress V and expand for GQA
        V_decompressed = tq_dequantize_mse(
            entry["v_idx"], entry["v_n"], cache.v_cb, cache.v_rot
        ).reshape(shape)
        V_expanded = V_decompressed.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        V_expanded = V_expanded.reshape(bsz, num_q_heads, kv_len, head_dim)

        return attn @ V_expanded
    else:
        # Raw (non-compressed) with GQA expansion
        K, V = entry
        K_expanded = K.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        K_expanded = K_expanded.reshape(bsz, num_q_heads, -1, head_dim)
        V_expanded = V.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        V_expanded = V_expanded.reshape(bsz, num_q_heads, -1, head_dim)

        scores = Q @ K_expanded.transpose(-2, -1) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return attn @ V_expanded


def _apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings (standard HF implementation)."""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
