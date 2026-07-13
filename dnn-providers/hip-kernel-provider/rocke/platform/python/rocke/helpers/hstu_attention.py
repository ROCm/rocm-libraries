# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Reusable HSTU attention math helpers."""

from __future__ import annotations

from ..core.ir import IRBuilder, Value
from .activations import _sigmoid_via_exp2


__all__ = [
    "hstu_mask_keep",
    "hstu_silu_and_grad",
    "hstu_to_mask_id",
]


def hstu_silu_and_grad(b: IRBuilder, score: Value, alpha: float) -> tuple[Value, Value]:
    """Return ``(silu(alpha * score), d/d(alpha*score) silu(alpha * score))``.

    Matches FlyDSL's fast sigmoid chain: ``exp2`` + reciprocal, not libdevice
    ``exp``. The caller applies the outer ``alpha`` factor to dQ/dK epilogues.
    """
    sc = b.fmul(score, b.const_f32(alpha))
    sig = _sigmoid_via_exp2(b, sc)
    silu = b.fmul(sc, sig)
    one_minus_sig = b.fsub(b.const_f32(1.0), sig)
    grad = b.fmul(sig, b.fadd(b.const_f32(1.0), b.fmul(sc, one_minus_sig)))
    return silu, grad


def hstu_to_mask_id(
    b: IRBuilder,
    x: Value,
    *,
    max_id: Value,
    contextual_seq_len: int,
    has_targets: bool,
) -> Value:
    """Apply HSTU's contextual shift and optional target-tail clamp."""
    xid = x
    if contextual_seq_len > 0:
        xid = b.sub(xid, b.const_i32(contextual_seq_len - 1))
        xid = b.select(b.cmp_lt(xid, b.const_i32(0)), b.const_i32(0), xid)
    if has_targets:
        xid = b.select(b.cmp_gt(xid, max_id), max_id, xid)
    return xid


def hstu_mask_keep(
    b: IRBuilder,
    *,
    q_local: Value,
    k_local: Value,
    max_id: Value,
    max_attn_len: int,
    contextual_seq_len: int,
    has_targets: bool,
) -> Value:
    """HSTU causal/window/contextual mask predicate.

    This is not equivalent to standard FMHA causal masking: HSTU compares shifted
    logical IDs, keeps the diagonal explicitly, and opens the contextual prefix.
    """
    q_id = hstu_to_mask_id(
        b,
        q_local,
        max_id=max_id,
        contextual_seq_len=contextual_seq_len,
        has_targets=has_targets,
    )
    k_id = hstu_to_mask_id(
        b,
        k_local,
        max_id=max_id,
        contextual_seq_len=contextual_seq_len,
        has_targets=has_targets,
    )
    dist = b.sub(q_id, k_id)
    keep = b.lor(b.cmp_eq(q_local, k_local), b.cmp_gt(dist, b.const_i32(0)))
    if max_attn_len > 0:
        keep = b.land(keep, b.cmp_le(dist, b.const_i32(max_attn_len)))
    if contextual_seq_len > 0:
        ctx = b.land(b.cmp_eq(q_id, b.const_i32(0)), b.cmp_lt(k_id, max_id))
        keep = b.lor(keep, ctx)
    return keep
