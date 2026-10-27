# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Additive attention bias for the gfx950 tiled 2D and 3D split-KV kernels.

The fp32 bias is read at element
``seq*batch_stride + q_head*head_stride + q_row*sq_stride + col`` (Sk stride 1;
a stride of 0 broadcasts that dimension), where ``q_head`` is the query head.
It is added to the log2-domain score after any softcap, so the score is
``softcap(QK * scale) + bias``. Python engine only for now.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import F32, I32, IRBuilder, PtrType, Value


# Stride params in ABI order, after the pointer. Shared by the kernel params and
# the host launcher signature, which packs kernargs by position.
_STRIDES = ("batch", "head", "sq")


@dataclass(frozen=True)
class AdditiveBiasParams:
    ptr: Value
    batch_stride: Value
    head_stride: Value
    sq_stride: Value


def declare_additive_bias_params(b: IRBuilder) -> AdditiveBiasParams:
    """The four kernel params, appended after all others; must match
    :func:`append_additive_bias_signature`."""
    ptr = b.param("additive_bias_ptr", PtrType(F32, "global"), readonly=True, align=4)
    strides = [b.param(f"additive_bias_{s}_stride", I32) for s in _STRIDES]
    return AdditiveBiasParams(ptr, *strides)


def append_additive_bias_signature(sb) -> None:
    """Host launcher entries for :func:`declare_additive_bias_params`, same order."""
    sb.ptr("additive_bias_ptr", "f32")
    for s in _STRIDES:
        sb.scalar(f"additive_bias_{s}_stride", "i32")


def add_additive_bias(
    b: IRBuilder,
    score: Value,
    p: AdditiveBiasParams,
    *,
    seq_idx: Value,
    q_head: Value,
    q_row: Value,
    col: Value,
    mask: Value,
    rcp_ln2: Value,
) -> Value:
    """Return ``score + bias[idx] / ln2``. Where ``mask`` is false (padding rows,
    tail columns) the lane loads element 0 instead of an out-of-range index and
    adds 0."""
    base = b.add(b.mul(p.batch_stride, seq_idx), b.mul(p.head_stride, q_head))
    row = b.add(base, b.mul(p.sq_stride, q_row))
    idx = b.add(row, col)
    v = b.masked_global_load(p.ptr, idx, mask, b.const_f32(0.0), dtype=F32, align=4)
    return b.fadd(score, b.fmul(v, rcp_ln2))
