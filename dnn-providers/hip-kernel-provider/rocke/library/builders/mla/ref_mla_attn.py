# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python reference implementations for MLA prefill forward pass.

These functions compute the numerically correct MLA attention output
using PyTorch operations.  They serve as the correctness oracle for the
rocKE kernel test suite.

Tolerances (matching the unified attention gate):
  max_abs ≤ 4e-2   for bf16 computation

References:
  DESIGN.md §8.1   (correctness reference section)
  DESIGN.md §2.2–§2.3  (math formulation)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# MLA geometry constants (all supported models share these)
# ---------------------------------------------------------------------------
D_NOPE = 128   # qk_nope_dim
D_ROPE = 64    # qk_rope_dim
D_QK   = 192   # d_nope + d_rope  (effective QK head dim)
D_V    = 128   # v_head_dim
R_KV   = 512   # kv_lora_rank
R_Q    = 1536  # q_lora_rank  (not used in this file — W_UQ applied prior)


# ---------------------------------------------------------------------------
# Reference: MLA prefill forward (expanded-Q path)
# ---------------------------------------------------------------------------


def ref_mla_prefill_fwd(
    q: Tensor,
    c_kv: Tensor,
    k_rope: Tensor,
    w_uk_k: Tensor,
    w_uv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Optional[Tensor] = None,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
) -> Tensor:
    """Reference MLA prefill forward (expanded-Q input form).

    Implements the expanded-form attention:
        K_nope = c_kv @ w_uk_k^T          [S_k, D_NOPE]
        V      = c_kv @ w_uv^T             [S_k, D_V]
        K_exp  = concat(K_nope, K_rope)    [S_k, D_QK]
        score  = scale * q @ K_exp^T        [S_q, S_k]
        out    = softmax(score) @ V         [S_q, D_V]

    Args:
        q:            [total_q, H_q, D_QK=192]  bf16 — already expanded
        c_kv:         [S_k, R_KV=512]            bf16 — compressed KV latent
        k_rope:       [S_k, D_ROPE=64]           bf16 — RoPE keys
        w_uk_k:       [R_KV, D_NOPE=128]         bf16 — K up-projection weight
        w_uv:         [R_KV, D_V=128]            bf16 — V up-projection weight
        cu_seqlens_q: [B+1]                       i32 prefix sums of Q lengths
        cu_seqlens_k: [B+1] or None               i32 prefix sums of KV lengths;
                      if None, assumed equal to cu_seqlens_q (sq==sk prefill)
        causal:       whether to apply causal mask (default True for prefill)
        scale:        attention scale; defaults to 1/sqrt(D_QK)

    Returns:
        out: [total_q, H_q, D_V] bf16
    """
    if scale is None:
        scale = 1.0 / math.sqrt(D_QK)

    device = q.device
    dtype  = q.dtype
    total_q, hq, d_qk = q.shape
    batch = cu_seqlens_q.shape[0] - 1

    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q

    # Expand K and V in f32 for numerical stability
    c_kv_f32    = c_kv.float()     # [S_k, R_KV]
    w_uk_k_f32  = w_uk_k.float()  # [R_KV, D_NOPE]
    w_uv_f32    = w_uv.float()    # [R_KV, D_V]
    k_rope_f32  = k_rope.float()  # [S_k, D_ROPE]

    K_nope = c_kv_f32 @ w_uk_k_f32  # [S_k, D_NOPE]
    V_exp  = c_kv_f32 @ w_uv_f32    # [S_k, D_V]
    K_exp  = torch.cat([K_nope, k_rope_f32], dim=-1)  # [S_k, D_QK]

    out = torch.zeros(total_q, hq, D_V, dtype=torch.float32, device=device)

    for b_idx in range(batch):
        sq_start = int(cu_seqlens_q[b_idx])
        sq_end   = int(cu_seqlens_q[b_idx + 1])
        sk_start = int(cu_seqlens_k[b_idx])
        sk_end   = int(cu_seqlens_k[b_idx + 1])

        sq = sq_end - sq_start
        sk = sk_end - sk_start
        if sq == 0 or sk == 0:
            continue

        q_b = q[sq_start:sq_end].float()  # [sq, hq, D_QK]
        K_b = K_exp[sk_start:sk_end]       # [sk, D_QK]
        V_b = V_exp[sk_start:sk_end]       # [sk, D_V]

        # score [sq, hq, sk]
        score = scale * torch.einsum("qhd,kd->qhk", q_b, K_b)

        if causal:
            # Causal mask: position i can attend to positions 0..i
            q_pos = torch.arange(sq, device=device).unsqueeze(1)    # [sq, 1]
            k_pos = torch.arange(sk, device=device).unsqueeze(0)    # [1, sk]
            mask = k_pos > q_pos                                     # [sq, sk]
            score = score.masked_fill(mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(score, dim=-1)  # [sq, hq, sk]
        out[sq_start:sq_end] = torch.einsum("qhk,kd->qhd", attn, V_b)

    return out.to(dtype)


def ref_mla_prefill_fwd_latent(
    q_latent: Tensor,
    c_kv: Tensor,
    k_rope: Tensor,
    w_uq: Tensor,
    w_uk_k: Tensor,
    w_uv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Optional[Tensor] = None,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
) -> Tensor:
    """Reference MLA prefill forward (compressed-Q input form).

    Expands the compressed query via W_UQ then delegates to
    :func:`ref_mla_prefill_fwd`.

    Args:
        q_latent: [total_q, R_Q=1536] bf16 — compressed query latent
        w_uq:     [H_q, R_Q, D_QK]  bf16 — query up-projection (per head)
        (other args same as ref_mla_prefill_fwd)

    Returns:
        out: [total_q, H_q, D_V] bf16
    """
    total_q, r_q = q_latent.shape
    hq = w_uq.shape[0]

    # Expand: q[total_q, H_q, D_QK] = einsum("qr,hrd->qhd", q_latent, w_uq)
    q_expanded = torch.einsum(
        "qr,hrd->qhd",
        q_latent.float(),
        w_uq.float(),
    ).to(q_latent.dtype)

    return ref_mla_prefill_fwd(
        q_expanded, c_kv, k_rope, w_uk_k, w_uv,
        cu_seqlens_q, cu_seqlens_k,
        causal=causal, scale=scale,
    )


# ---------------------------------------------------------------------------
# Helpers for test generation
# ---------------------------------------------------------------------------


def make_mla_prefill_inputs(
    num_query_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    """Generate random but consistent MLA prefill inputs.

    Returns a dict with:
        q, c_kv, k_rope, w_uk_k, w_uv: Tensors
        block_table: [1, num_blocks] i32
        cu_seqlens_q: [2] i32
        scale: float
        seqlen_q, seqlen_k, num_query_heads: ints
    """
    torch.manual_seed(seed)
    total_q = seqlen_q
    total_k = seqlen_k
    num_blocks = (total_k + block_size - 1) // block_size
    scale = 1.0 / math.sqrt(D_QK)

    q      = torch.randn(total_q, num_query_heads, D_QK,  dtype=dtype, device=device) * 0.1
    c_kv   = torch.randn(total_k, R_KV,           dtype=dtype, device=device) * 0.1
    k_rope = torch.randn(total_k, D_ROPE,          dtype=dtype, device=device) * 0.1
    w_uk_k = torch.randn(R_KV, D_NOPE,            dtype=dtype, device=device) * 0.01
    w_uv   = torch.randn(R_KV, D_V,               dtype=dtype, device=device) * 0.01

    # Physical block layout: sequential blocks
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32, device=device)

    # Paged c_kv and k_rope: [num_blocks, block_size, R_KV/D_ROPE]
    # Pad last block with zeros if needed
    pad_k = num_blocks * block_size - total_k
    c_kv_pad   = torch.cat([c_kv,   torch.zeros(pad_k, R_KV,  dtype=dtype, device=device)], dim=0)
    k_rope_pad = torch.cat([k_rope, torch.zeros(pad_k, D_ROPE, dtype=dtype, device=device)], dim=0)
    c_kv_paged   = c_kv_pad.view(num_blocks, block_size, R_KV)
    k_rope_paged = k_rope_pad.view(num_blocks, block_size, D_ROPE)

    # q_latent: [total_q, hq, R_KV+D_ROPE=576] — raw Q latent for AITER Triton MLA.
    # In practice this is concat(q_abs[hq, R_KV], q_rope[hq, D_ROPE]) per token.
    # We synthesise it here as a random tensor with matching scale so that AITER
    # can be benchmarked for latency (correctness vs our kernel is not asserted).
    q_latent = torch.randn(total_q, num_query_heads, R_KV + D_ROPE,
                           dtype=dtype, device=device) * 0.1

    # kv_buffer: [num_blocks, block_size, 1, 576] = concat(c_KV, K_rope) per token.
    # This is the format expected by aiter.ops.triton.attention.mla.mla_prefill_fwd.
    kv_buf_flat = torch.cat([c_kv_pad, k_rope_pad], dim=-1)   # [nb*bs, 576]
    kv_buffer   = kv_buf_flat.view(num_blocks, block_size, 1, R_KV + D_ROPE)

    return {
        "q":              q,
        "q_latent":       q_latent,   # [sq, hq, 576] — for AITER Triton MLA
        "kv_buffer":      kv_buffer,  # [nb, bs, 1, 576] — for AITER Triton MLA
        "c_kv_paged":     c_kv_paged,
        "k_rope_paged":   k_rope_paged,
        "c_kv_flat":      c_kv,
        "k_rope_flat":    k_rope,
        "w_uk_k":         w_uk_k,
        "w_uv":           w_uv,
        "block_table":    block_table,
        "cu_seqlens_q":   cu_seqlens_q,
        "scale":          scale,
        "seqlen_q":       seqlen_q,
        "seqlen_k":       seqlen_k,
        "num_query_heads": num_query_heads,
        "block_size":     block_size,
        "num_blocks":     num_blocks,
    }
