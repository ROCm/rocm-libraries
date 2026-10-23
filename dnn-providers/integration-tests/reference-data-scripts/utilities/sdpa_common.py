#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Shared SDPA golden-reference helpers.

Imported by both `generate_sdpa_fwd_golden.py` and `generate_sdpa_bwd_golden.py`
so the two generators cannot drift apart. That matters most for masking and for
the forward pass itself: a backward bundle is generated from the forward O/LSE
of the same run (RFC 0011 §4.2 "Forward-Backward Generation Constraint"), so if
the two generators disagreed about what a bound means or how softmax is
computed, the backward golden data would encode that disagreement instead of
catching it.

Everything here is operation-level and reference-source-level; nothing here
knows about bundle layout, graph JSON, or metadata.
"""

import hashlib

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

DTYPE_MAP = {
    "bf16": {"torch": torch.bfloat16, "json": "bfloat16", "bytes": 2},
    "fp16": {"torch": torch.float16, "json": "half", "bytes": 2},
}

# Physical stride order per memory layout. The value at position i is the rank of
# dim i (0 = fastest-varying/unit stride). Logical dims stay [B, H, S, D]; only
# the strides change, so BSHD swaps the H and S ranks relative to BHSD.
LAYOUT_STRIDE_ORDER = {"bhsd": [3, 2, 1, 0], "bshd": [3, 1, 2, 0]}


def compute_contiguous_strides(dims):
    strides = []
    stride = 1
    for d in reversed(dims):
        strides.append(stride)
        stride *= d
    strides.reverse()
    return strides


def compute_strides(dims, layout="bhsd"):
    """Strides for the given layout. bhsd matches compute_contiguous_strides."""
    order = LAYOUT_STRIDE_ORDER.get(layout)
    if order is None or len(dims) != len(order):
        raise ValueError(f"no stride order for layout {layout!r} at rank {len(dims)}")
    stride = [1] * len(dims)
    acc = 1
    for idx in sorted(range(len(dims)), key=lambda i: order[i]):
        stride[idx] = acc
        acc *= dims[idx]
    return stride


def to_physical_layout(tensor, layout):
    """Reorder a rank-4 BHSD tensor to physical BSHD byte order when requested."""
    if layout == "bshd" and tensor.dim() == 4:
        return tensor.transpose(1, 2).contiguous()  # physical [B, S, H, D]
    return tensor


def build_bound_mask(s_q, s_kv, left_bound, right_bound, top_left_alignment):
    """Boolean keep-mask [S_q, S_kv] from the graph's diagonal-band parameters.

        This is the vectorized negation of the CPU reference executor's `isMasked()`
        (CpuFpReferenceSdpa.hpp), which both reference executors share, so the golden
        data encodes exactly the band the device-under-test is asked to produce:

          masked if right_bound >= 0 and skv >= sq + 1 + offset + right_bound
          masked if left_bound  >= 0 and skv <= sq + offset - left_bound

    left_bound is cuDNN's: it keeps that many keys including the diagonal
    (SdpaSemanticsRegister SEM-001), so 0 would keep no keys and is rejected.

        where offset is 0 for top-left alignment and (S_kv - S_q) for bottom-right.
        A bound of -1 (or None) means "unbounded" on that side. Returns None when
        neither bound constrains anything, so callers can take the dense fast path.
    """
    left = -1 if left_bound is None else left_bound
    right = -1 if right_bound is None else right_bound
    if left == 0:
        raise ValueError("left_bound = 0 keeps no keys (cuDNN rejects it)")
    if left < 0 and right < 0:
        return None

    row = torch.arange(s_q).unsqueeze(1)  # [S_q, 1]
    col = torch.arange(s_kv).unsqueeze(0)  # [1, S_kv]
    offset = 0 if top_left_alignment else (s_kv - s_q)

    keep = torch.ones(s_q, s_kv, dtype=torch.bool)
    if right >= 0:
        keep &= col < (row + 1 + offset + right)
    if left >= 0:
        keep &= col > (row + offset - left)
    return keep


def causal_window_bounds(causal, window_left, window_right):
    """Map the forward generator's causal/window arguments to band parameters.

    Returns (left_bound, right_bound, top_left_alignment) with -1 meaning
    unbounded. Resolved exactly as the forward graph serializes them (cuDNN setter
    composition): causal sets right_bound = 0, and an explicit --window-right
    replaces it. Taking the minimum instead would keep right_bound at 0 while the
    graph says window_right.
    """
    left_bound = window_left if window_left >= 0 else -1
    right_bound = window_right if window_right >= 0 else (0 if causal != "none" else -1)
    return left_bound, right_bound, causal != "bottom_right"


def build_mask(s_q, s_kv, causal, window_left, window_right):
    """Construct a boolean keep-mask [S_q, S_kv] for the given causal/window config.

    Returns None when no masking applies (the dense attention case), so callers
    can take the fast PyTorch path. The mask is True where attention is allowed.
    The bound math matches the CPU reference executor's leftBound/rightBound +
    topLeftAlignment semantics so golden output is reproducible on the CPU side.

    causal:
      - "none":          no causal constraint
      - "top_left":      diagonal anchored at (0,0); keep skv <= sq
      - "bottom_right":  diagonal anchored bottom-right; keep skv <= sq + (S_kv - S_q)
    """
    left_bound, right_bound, top_left = causal_window_bounds(
        causal, window_left, window_right
    )
    return build_bound_mask(s_q, s_kv, left_bound, right_bound, top_left)


def additive_mask(keep_mask):
    """Convert a boolean keep-mask to an additive FP32 mask (0 keep, -inf drop)."""
    additive = torch.zeros(keep_mask.shape, dtype=torch.float32)
    additive.masked_fill_(~keep_mask, float("-inf"))
    return additive


def combined_attn_mask(keep_mask, attn_bias=None):
    """The single additive mask that carries both the band and an explicit bias.

    The CPU reference adds the graph's attention-mask tensor to the scores first
    and only then forces the out-of-band entries to -inf
    (CpuFpReferenceSdpa.hpp, steps 2 and 3), so a finite bias can never rescue a
    masked position. Summing a finite bias with a -inf band gives -inf, which
    reproduces that ordering exactly.

    Returns None when there is neither a band nor a bias.
    """
    if keep_mask is None and attn_bias is None:
        return None
    if attn_bias is None:
        return additive_mask(keep_mask)
    combined = attn_bias.float()
    if keep_mask is not None:
        combined = combined + additive_mask(keep_mask)
    return combined


def compute_forward(Q, K, V, scale, H_q, H_kv, keep_mask=None, attn_bias=None):
    """SDPA forward via PyTorch Math backend (half->FP32->half).

    keep_mask is an optional boolean [S_q, S_kv] (True = attend); applied as an
    additive mask so causal/window compose uniformly with GQA. attn_bias is an
    optional additive score bias broadcastable to [B, H_q, S_q, S_kv].
    """
    attn_mask = combined_attn_mask(keep_mask, attn_bias)
    with sdpa_kernel(SDPBackend.MATH):
        O = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            scale=scale,
            enable_gqa=(H_q != H_kv),
        )
    return O


def compute_lse(Q, K, scale, H_q, H_kv, keep_mask=None, attn_bias=None):
    """Compute Log-Sum-Exp reference: logsumexp(Q @ K^T * scale, dim=-1).

    Returns shape [B, H_q, S_q, 1] in FP32. For fully-masked rows the LSE is
    -inf (log of zero), matching the CPU reference semantics.
    """
    Q_f = Q.float()
    K_f = K.float()

    if H_q != H_kv:
        gqa_ratio = H_q // H_kv
        K_f = K_f.repeat_interleave(gqa_ratio, dim=1)

    scores = torch.matmul(Q_f, K_f.transpose(-2, -1)) * scale
    attn_mask = combined_attn_mask(keep_mask, attn_bias)
    if attn_mask is not None:
        scores = scores + attn_mask
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)  # [B, H_q, S_q, 1]
    return lse


def save_tensor_bin(tensor, path):
    t = tensor.contiguous().cpu()
    if t.dtype in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        raw = t.view(torch.uint8).numpy().tobytes()
    else:
        raw = t.numpy().tobytes()
    with open(path, "wb") as f:
        f.write(raw)


def script_sha256(script_path):
    """SHA-256 of a generator script's contents — a git-independent version marker.

    It names the exact script bytes that emitted a bundle, so any edit to that
    script -- logic, metadata schema, even a comment -- makes the value recorded
    in already checked-in .meta.json files stale. That is the intended behaviour:
    those files record the generator that produced *their* tensors, so they are
    never rewritten to match a later revision of the script.
    """
    with open(script_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
