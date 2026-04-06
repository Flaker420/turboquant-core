"""
Hook-in module for patching Qwen models to use TurboQuant compressed KV cache.

Supported models:
    - Qwen3.5-9B (hybrid: 8 GatedAttn layers compressed, DeltaNet unchanged)
    - Qwen3-8B (dense: all 36 layers compressed)
    - Qwen2.5-3B-Instruct (dense: all 36 layers, 2 KV heads)

Usage:
    from turboquant_core.backends.qwen_hook import (
        patch_qwen35_with_tq, patch_qwen3_with_tq, patch_qwen25_with_tq,
    )

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", ...)
    cache = patch_qwen25_with_tq(
        model, bit_width=4, key_strategy="mse", residual_window=128,
    )

    # model.generate() now uses compressed KV cache automatically.
    # Call cache.clear() between generations (or use adapter.reset_generation_state()).

This module monkey-patches attention layers' forward to:
1. Compress K/V into the TQQuantizedCache after projection
2. Apply causal masking for correct multi-token prefill
3. Compute attention scores (MSE-only or MSE+QJL corrected, per key_strategy)
4. When residual_window > 0, keep recent tokens in FP16 and compress older ones
5. Handle GQA head expansion for all three data layouts (window-only,
   compressed-only, compressed+window)
6. Pass through non-compressible layers unchanged (Qwen3.5 DeltaNet layers)

Requires: transformers with Qwen2.5/Qwen3/Qwen3.5 support.
"""

import math
import functools

import torch

from ..core import (
    TQQuantizedCache,
    tq_dequantize_mse,
)


def patch_qwen35_with_tq(model, bit_width=4, seed=42, device=None, *,
                         num_layers=32, full_attn_interval=4,
                         kv_heads=4, head_dim=256, residual_window=0,
                         key_strategy="mse+qjl", value_strategy="mse"):
    """Patch a Qwen3.5 model to use TurboQuant compressed KV cache.

    Args:
        model: A Qwen3.5 CausalLM model from transformers.
        bit_width: Total bits per value (K gets b-1 MSE + 1 QJL, V gets b MSE).
        seed: Random seed for rotation and QJL matrices.
        device: Device for TQ buffers. Defaults to model's device.
        num_layers: Number of transformer layers (default 32).
        full_attn_interval: GatedAttn layer interval (default 4).
        kv_heads: Number of KV heads (default 4).
        head_dim: Head dimension (default 256).

    Returns:
        TQQuantizedCache instance. Call cache.clear() between generations.
    """
    if device is None:
        device = next(model.parameters()).device

    cache = TQQuantizedCache(
        num_layers=num_layers, interval=full_attn_interval,
        kv_head_dim=head_dim, num_kv_heads=kv_heads,
        bit_width=bit_width, seed=seed, device=device,
        residual_window=residual_window,
        key_strategy=key_strategy,
        value_strategy=value_strategy,
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


def patch_qwen3_with_tq(model, bit_width=4, seed=42, device=None, *,
                        num_layers=36, kv_heads=8, head_dim=128,
                        residual_window=0, key_strategy="mse+qjl",
                        value_strategy="mse"):
    """Patch a Qwen3-8B model to use TurboQuant compressed KV cache.

    All layers are dense attention and compressible.

    Args:
        model: A Qwen3 CausalLM model from transformers.
        bit_width: Total bits per value (K gets b-1 MSE + 1 QJL, V gets b MSE).
        seed: Random seed for rotation and QJL matrices.
        device: Device for TQ buffers. Defaults to model's device.
        num_layers: Number of transformer layers (default 36).
        kv_heads: Number of KV heads (default 8).
        head_dim: Head dimension (default 128).

    Returns:
        TQQuantizedCache instance. Call cache.clear() between generations.
    """
    if device is None:
        device = next(model.parameters()).device

    cache = TQQuantizedCache(
        num_layers=num_layers, interval=1,
        kv_head_dim=head_dim, num_kv_heads=kv_heads,
        bit_width=bit_width, seed=seed, device=device,
        residual_window=residual_window,
        key_strategy=key_strategy,
        value_strategy=value_strategy,
    )

    layers = _get_model_layers(model)
    if layers is None:
        raise ValueError(
            "Could not find transformer layers in model. "
            "Expected model.model.layers or similar structure."
        )

    for layer_idx, layer in enumerate(layers):
        attn = _get_attention_module(layer)
        if attn is None:
            continue

        _patch_attention_forward(attn, cache, layer_idx)

    return cache


def patch_qwen25_with_tq(model, bit_width=4, seed=42, device=None, *,
                         num_layers=36, kv_heads=2, head_dim=128,
                         residual_window=0, key_strategy="mse+qjl",
                         value_strategy="mse"):
    """Patch a Qwen2.5 model to use TurboQuant compressed KV cache.

    Qwen2.5-3B-Instruct: 36 dense attention layers, 2 KV heads, head_dim 128.
    All layers are compressible (f=1.0). This is the recommended first
    ablation target due to small size and direct community comparability.

    Args:
        model: A Qwen2.5 CausalLM model from transformers.
        bit_width: Total bits per value.
        seed: Random seed for rotation and QJL matrices.
        device: Device for TQ buffers. Defaults to model's device.
        num_layers: Number of transformer layers (default 36).
        kv_heads: Number of KV heads (default 2).
        head_dim: Head dimension (default 128).
        residual_window: Number of recent tokens to keep in FP16 (default 0).
        key_strategy: "mse+qjl" or "mse" (default "mse+qjl").

    Returns:
        TQQuantizedCache instance. Call cache.clear() between generations.
    """
    if device is None:
        device = next(model.parameters()).device

    cache = TQQuantizedCache(
        num_layers=num_layers, interval=1,
        kv_head_dim=head_dim, num_kv_heads=kv_heads,
        bit_width=bit_width, seed=seed, device=device,
        residual_window=residual_window,
        key_strategy=key_strategy,
        value_strategy=value_strategy,
    )

    layers = _get_model_layers(model)
    if layers is None:
        raise ValueError(
            "Could not find transformer layers in model. "
            "Expected model.model.layers or similar structure."
        )

    for layer_idx, layer in enumerate(layers):
        attn = _get_attention_module(layer)
        if attn is None:
            continue

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
    1. Computes Q, K, V projections and applies rotary embeddings
    2. Stores K/V in the TQQuantizedCache (compressed or FP16 window)
    3. Builds a causal mask for correct multi-token prefill
    4. Computes attention via GQA-aware path (MSE-only or MSE+QJL)
    5. Returns (attn_output, None, past_key_value)
    """
    attn_module._tq_original_forward = attn_module.forward
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

        # Build causal mask: query positions can only attend to key positions <= their own
        kv_len = cache.get_seq_length(layer_idx)
        # Query positions are the last q_len positions in the sequence
        q_start = kv_len - q_len
        # causal_mask[i, j] = True means position i can attend to position j
        q_positions = torch.arange(q_start, kv_len, device=Q.device).unsqueeze(1)
        k_positions = torch.arange(kv_len, device=Q.device).unsqueeze(0)
        causal_mask = k_positions <= q_positions  # [q_len, kv_len]

        # Combine with external attention_mask if provided (e.g. padding mask)
        if attention_mask is not None:
            # attention_mask from HF is [batch, 1, q_len, kv_len] or [batch, 1, 1, kv_len]
            # with 0 for valid positions and large negative values for masked positions
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]
        else:
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]

        # Compute attention using the full cache (including previous tokens)
        if num_q_heads != num_kv_heads:
            # GQA: Q has more heads than K/V, need manual head expansion
            attn_output = _gqa_attention(
                Q, cache, layer_idx, num_q_heads, num_kv_heads,
                causal_mask=combined_mask, attention_mask=attention_mask,
            )
        else:
            attn_output = cache.compute_attention(
                Q, layer_idx,
                causal_mask=combined_mask, attention_mask=attention_mask,
            )

        # Reshape back: [batch, heads, seq_len, head_dim] -> [batch, seq_len, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # Output projection
        attn_output = attn_module.o_proj(attn_output)

        return attn_output, None, past_key_value

    attn_module.forward = tq_forward


def _apply_mask(scores, causal_mask, attention_mask):
    """Apply causal and padding masks to attention scores.

    Args:
        scores: [batch, heads, q_len, kv_len]
        causal_mask: [1, 1, q_len, kv_len] boolean (True = attend)
        attention_mask: Optional HF-style mask [batch, 1, q_len/1, kv_len]
            with 0 for valid, large negative for masked positions.
    """
    # Apply causal mask (True = can attend, False = masked)
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    # Apply padding mask if provided
    if attention_mask is not None:
        scores = scores + attention_mask
    return scores


def _expand_kv_for_gqa(tensor, num_kv_heads, num_groups, bsz, seq_len, head_dim):
    """Expand KV tensor from [bsz, kv_heads, seq, hd] to [bsz, q_heads, seq, hd]."""
    num_q_heads = num_kv_heads * num_groups
    expanded = tensor.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
    return expanded.reshape(bsz, num_q_heads, seq_len, head_dim)


def _gqa_attention(Q, cache, layer_idx, num_q_heads, num_kv_heads,
                   causal_mask=None, attention_mask=None):
    """Handle grouped-query attention with TQ cache.

    Supports three data layouts:
    1. Window-only: all tokens in FP16 (no compressed cache yet)
    2. Compressed-only: all tokens quantized (residual_window=0)
    3. Compressed + window: older tokens compressed, recent in FP16
    """
    entry = cache._cache[layer_idx]
    has_compressed = cache.is_compressible(layer_idx) and isinstance(entry, dict)
    has_window = cache._window_k[layer_idx] is not None
    bsz = Q.shape[0]
    q_len = Q.shape[2]
    head_dim = Q.shape[3]
    num_groups = num_q_heads // num_kv_heads

    if not cache.is_compressible(layer_idx):
        # Raw (non-compressed) with GQA expansion
        K, V = entry
        K_expanded = _expand_kv_for_gqa(K, num_kv_heads, num_groups, bsz, K.shape[2], head_dim)
        V_expanded = _expand_kv_for_gqa(V, num_kv_heads, num_groups, bsz, V.shape[2], head_dim)

        scores = Q @ K_expanded.transpose(-2, -1) / (head_dim ** 0.5)
        if causal_mask is not None:
            scores = _apply_mask(scores, causal_mask, attention_mask)
        attn = torch.softmax(scores, dim=-1)
        return attn @ V_expanded

    # --- Window-only path ---
    if not has_compressed and has_window:
        wk = cache._window_k[layer_idx]
        wv = cache._window_v[layer_idx]
        wk_exp = _expand_kv_for_gqa(wk, num_kv_heads, num_groups, bsz, wk.shape[2], head_dim)
        wv_exp = _expand_kv_for_gqa(wv, num_kv_heads, num_groups, bsz, wv.shape[2], head_dim)

        scores = Q @ wk_exp.transpose(-2, -1) / (head_dim ** 0.5)
        if causal_mask is not None:
            scores = _apply_mask(scores, causal_mask, attention_mask)
        attn = torch.softmax(scores, dim=-1)
        return attn @ wv_exp

    if not has_compressed and not has_window:
        raise ValueError(f"No cache for layer {layer_idx}")

    # --- Compressed path (possibly with residual window) ---
    compressed_len = cache._compressed_seq_len(layer_idx)
    c_shape = (bsz, num_kv_heads, compressed_len, head_dim)

    # MSE-reconstructed K from compressed cache
    K_mse = tq_dequantize_mse(
        entry["k_mse"], entry["k_n"], cache.k_cb, cache.k_rot
    ).reshape(c_shape)
    K_mse_expanded = _expand_kv_for_gqa(
        K_mse, num_kv_heads, num_groups, bsz, compressed_len, head_dim,
    )
    scores_mse = Q @ K_mse_expanded.transpose(-2, -1)

    if cache.key_strategy == "mse+qjl":
        # QJL correction at KV head granularity, then expand
        k_qjl_all = entry["k_qjl"].reshape(bsz, num_kv_heads, compressed_len, head_dim)
        k_rn_all = entry["k_rn"].reshape(bsz, num_kv_heads, compressed_len)
        k_n_all = entry["k_n"].reshape(bsz, num_kv_heads, compressed_len)

        Q_grouped = Q.reshape(bsz, num_kv_heads, num_groups, q_len, head_dim)
        correction_per_kv = []
        for g in range(num_kv_heads):
            Q_g = Q_grouped[:, g].reshape(bsz * num_groups * q_len, head_dim)
            Q_qjl = cache.k_qjl.quantize(Q_g).reshape(bsz * num_groups, q_len, head_dim)

            k_qjl_g = k_qjl_all[:, g].unsqueeze(1).expand(-1, num_groups, -1, -1)
            k_qjl_g = k_qjl_g.reshape(bsz * num_groups, compressed_len, head_dim)

            corr = Q_qjl.float() @ k_qjl_g.float().transpose(-2, -1)
            corr = corr * (math.pi / (2 * head_dim))
            k_rn_g = k_rn_all[:, g].unsqueeze(1).expand(-1, num_groups, -1)
            k_n_g = k_n_all[:, g].unsqueeze(1).expand(-1, num_groups, -1)
            corr = corr * k_rn_g.reshape(bsz * num_groups, 1, compressed_len)
            corr = corr * k_n_g.reshape(bsz * num_groups, 1, compressed_len)
            correction_per_kv.append(corr.reshape(bsz, num_groups, q_len, compressed_len))

        correction = torch.stack(correction_per_kv, dim=1)
        correction = correction.reshape(bsz, num_q_heads, q_len, compressed_len)

        compressed_scores = (scores_mse + correction) / (head_dim ** 0.5)
    else:
        compressed_scores = scores_mse / (head_dim ** 0.5)

    # Decompress V from compressed region
    V_compressed = tq_dequantize_mse(
        entry["v_idx"], entry["v_n"], cache.v_cb, cache.v_rot
    ).reshape(c_shape)
    V_compressed_expanded = _expand_kv_for_gqa(
        V_compressed, num_kv_heads, num_groups, bsz, compressed_len, head_dim,
    )

    if has_window:
        # Combine compressed scores with FP16 window scores
        wk = cache._window_k[layer_idx]
        wv = cache._window_v[layer_idx]
        wk_exp = _expand_kv_for_gqa(wk, num_kv_heads, num_groups, bsz, wk.shape[2], head_dim)
        wv_exp = _expand_kv_for_gqa(wv, num_kv_heads, num_groups, bsz, wv.shape[2], head_dim)
        window_scores = Q @ wk_exp.transpose(-2, -1) / (head_dim ** 0.5)

        all_scores = torch.cat([compressed_scores, window_scores], dim=-1)
        all_V = torch.cat([V_compressed_expanded, wv_exp], dim=2)
    else:
        all_scores = compressed_scores
        all_V = V_compressed_expanded

    if causal_mask is not None:
        all_scores = _apply_mask(all_scores, causal_mask, attention_mask)
    attn = torch.softmax(all_scores, dim=-1)
    return attn @ all_V


def unpatch_model(model):
    """Restore original forward methods on all TQ-patched attention modules.

    Iterates all transformer layers and, for any attention module that has the
    ``_tq_original_forward`` attribute (set during patching), restores the
    original forward and removes the marker attribute.

    Args:
        model: A patched HuggingFace model.

    Returns:
        int: Number of layers that were unpatched.
    """
    layers = _get_model_layers(model)
    if layers is None:
        return 0

    count = 0
    for layer in layers:
        attn = _get_attention_module(layer)
        if attn is None:
            continue
        if hasattr(attn, '_tq_original_forward'):
            attn.forward = attn._tq_original_forward
            del attn._tq_original_forward
            count += 1
    return count


def _apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings (standard HF implementation)."""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
