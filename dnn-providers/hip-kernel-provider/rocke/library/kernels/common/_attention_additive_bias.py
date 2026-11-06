# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Additive attention bias for the gfx950 tiled 2D, 3D split-KV and dense kernels.

The fp32 bias is read at element
``seq*batch_stride + q_head*head_stride + q_row*sq_stride + col`` (Sk stride 1;
a stride of 0 broadcasts that dimension), where ``q_head`` is the query head.
It is added to the log2-domain score: after any softcap in the tiled kernels
(``softcap(QK * scale) + bias``), before the mask in the dense kernel. The
builders are Python only; the C++ lowerer lowers the IR they emit.

Device side: :func:`declare_additive_bias_params` adds the four kernel params,
:func:`additive_bias_row` computes a query row's base index once, and
:func:`add_additive_bias_at` adds one element (:func:`add_additive_bias` does
both); :func:`additive_bias_elem_ptr` builds one 64-bit pointer per KV tile and
:func:`add_additive_bias_x4_at` adds four consecutive in-range keys with one
16-byte load at a constant element offset from it (the dense kernel's non-ragged
body). Host side: :func:`append_additive_bias_signature` adds the launcher
entries; :func:`additive_bias_strides` (strides derived from a tensor) or the
unified kernels' ``_validate_additive_bias`` (strides passed by the caller)
validates a launch, both through :func:`check_additive_bias_tensor` and
:func:`check_additive_bias_reach`; :func:`validate_dense_additive_bias` is the
dense runners' entry point (presence and strides; a varlen runner also bounds
its offsets by the spec maxima); :func:`additive_bias_launch_values` packs it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType, Value

# Stride params in ABI order, after the pointer. Shared by the kernel params and
# the host launcher signature, which packs kernargs by position.
_STRIDES = ("batch", "head", "sq")

# additive_bias_row / add_additive_bias_at / additive_bias_elem_ptr index the bias
# in i32 (additive_bias_elem_ptr also zero-extends the index, so it relies on it
# being non-negative too; the x4 loads then add only a host-constant offset); every
# host validator of a bias launch bounds the largest element index by this
# (through check_additive_bias_reach). The gfx1151 dispatch binding keeps its own
# copy of the bound; the hipDNN dense pack (Gfx950AttentionDenseNative.cpp) states
# it once, as the inclusive MAX_BIAS_INDEX = 2**31 - 1.
ADDITIVE_BIAS_INDEX_LIMIT = 2**31


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


def additive_bias_launch_values(
    bias: Any, strides: tuple[int, int, int]
) -> dict[str, Any]:
    """Launcher values for the four entries :func:`append_additive_bias_signature`
    declares, from the bias tensor and its :func:`additive_bias_strides`."""
    vals = {"additive_bias_ptr": bias}
    for s, stride in zip(_STRIDES, strides):
        vals[f"additive_bias_{s}_stride"] = int(stride)
    return vals


def additive_bias_row(
    b: IRBuilder, p: AdditiveBiasParams, *, seq_idx: Value, q_head: Value, q_row: Value
) -> Value:
    """Element index of key 0 in the bias row of ``(seq_idx, q_head, q_row)``.
    Loop-invariant for a query row, so a kernel can hoist it out of its KV loop."""
    b_off = b.mul(p.batch_stride, seq_idx)
    h_off = b.mul(p.head_stride, q_head)
    base = b.add(b_off, h_off)
    q_off = b.mul(p.sq_stride, q_row)
    return b.add(base, q_off)


def add_additive_bias_at(
    b: IRBuilder,
    score: Value,
    p: AdditiveBiasParams,
    *,
    row: Value,
    col: Value,
    mask: Value | None,
    rcp_ln2: Value,
    floor: Value | None = None,
) -> Value:
    """Return ``score + bias[row + col] * rcp_ln2`` for a ``row`` from
    :func:`additive_bias_row` with the same params. ``score`` is in log2 units
    (already scaled) and ``rcp_ln2`` is the constant 1/ln2 (= log2 e), which
    converts the natural-log bias to log2 units.

    Where ``mask`` is false the lane loads element 0 and adds 0. ``mask=None``
    emits a plain, unchecked load: only for callers whose every ``(row, col)``
    is in range, which the host validation of the launch has to guarantee.
    ``floor`` (natural-log units) clamps the bias from below, -inf and NaN
    included, for kernels whose mask is a finite sentinel. There is no upper
    clamp: a bias whose log2 value overflows fp32 (above about 2.3e38) makes
    the row NaN."""
    idx = b.add(row, col)
    if mask is None:
        v = b.global_load(p.ptr, idx, F32, align=4)
    else:
        v = b.masked_global_load(p.ptr, idx, mask, b.const_f32(0.0), dtype=F32, align=4)
    if floor is not None:
        v = b.fmax(v, floor)
    return b.fadd(score, b.fmul(v, rcp_ln2))


def additive_bias_elem_ptr(
    b: IRBuilder, p: AdditiveBiasParams, *, row: Value, col: Value
) -> Value:
    """Global pointer to ``bias[row + col]``, for loads at constant element offsets
    from it (see :func:`add_additive_bias_x4_at`): the backend folds a constant
    offset into the load's immediate, so a group of loads shares one 64-bit
    address instead of each sign-extending and scaling its own i32 index.

    ``row`` and ``col`` must address an element the launch reads (an in-range
    query row and key): only then does the host reach check
    (:func:`check_additive_bias_reach`; the hipDNN pack's copy) bound
    ``row + col`` below 2**31. With every stride >= 0 (all hosts refuse a
    negative one), the i32 index is then never negative and its zero-extension
    is exact. A padded row or tail key, like the ragged body's masked ones,
    may wrap the i32 add, and zero-extending that reads 4 GiB away. Use
    :func:`add_additive_bias_at` with a mask there."""
    idx = b.add(row, col)
    idx64 = b.zext(idx, I64)
    elem_bytes = b.const_i64(4)
    byte_off = b.mul(idx64, elem_bytes)
    return b.global_ptr_add(p.ptr, byte_off)


def add_additive_bias_x4_at(
    b: IRBuilder,
    scores: Sequence[Value],
    ptr: Value,
    *,
    offset: int,
    rcp_ln2: Value,
    floor: Value | None = None,
) -> list[Value]:
    """Vector form of :func:`add_additive_bias_at` for four consecutive keys:
    return ``scores[j] + ptr[offset + j] * rcp_ln2`` for ``j`` in 0..3, where
    ``ptr`` comes from :func:`additive_bias_elem_ptr` and ``offset`` is a constant
    element offset from it, read with one 16-byte load instead of four 4-byte
    ones. Each element gets the same floor clamp, scale and add as the scalar
    form, so the results are bitwise equal to it (pinned on the GPU against the
    scalar path).

    The load is plain and unchecked, like ``mask=None`` in the scalar form: all
    four keys must be in range. It is only element-aligned (``align=4``): the
    host contract does not guarantee 16-byte alignment of the address."""
    if len(scores) != 4:
        raise ValueError(
            f"add_additive_bias_x4_at takes exactly 4 scores, got {len(scores)}"
        )
    elem_off = b.const_i32(offset)
    vec = b.global_load_vN(ptr, elem_off, F32, 4, align=4)
    out: list[Value] = []
    for j, score in enumerate(scores):
        v = b.vec_extract(vec, j)
        if floor is not None:
            v = b.fmax(v, floor)
        scaled_bias = b.fmul(v, rcp_ln2)
        out.append(b.fadd(score, scaled_bias))
    return out


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
    row = additive_bias_row(b, p, seq_idx=seq_idx, q_head=q_head, q_row=q_row)
    return add_additive_bias_at(
        b, score, p, row=row, col=col, mask=mask, rcp_ln2=rcp_ln2
    )


def check_additive_bias_tensor(bias: Any, device: Any | None) -> None:
    """Raise ``ValueError`` unless ``bias`` is fp32 and on ``device``, the launch's
    Q device (``None`` skips the device check: host-only tests have no Q)."""
    if str(bias.dtype) != "torch.float32":
        raise ValueError(f"additive_bias must be float32, got {bias.dtype}")
    if device is not None and bias.device != device:
        raise ValueError(
            f"additive_bias must be on the same device as q, got "
            f"{bias.device} vs {device}"
        )


def check_additive_bias_reach(bias: Any, max_idx: int) -> None:
    """Check ``max_idx``, the largest element index a launch reads, against the
    kernels' i32 addressing (``NotImplementedError``: a too-big bias, not a
    malformed one) and against the elements ``bias``'s storage backs from its
    offset (``ValueError``: strides that do not fit the tensor)."""
    if max_idx >= ADDITIVE_BIAS_INDEX_LIMIT:
        raise NotImplementedError(
            f"additive_bias max element index {max_idx} exceeds the kernels' i32 "
            "bias addressing (< 2**31, i.e. 8 GiB of f32); broadcast a dimension "
            "(stride 0) or split the launch"
        )
    avail = bias.untyped_storage().nbytes() // bias.element_size() - int(
        bias.storage_offset()
    )
    if max_idx >= avail:
        raise ValueError(
            f"additive_bias strides reach element {max_idx} but the tensor only "
            f"backs {avail} elements from its storage offset"
        )


def additive_bias_strides(
    bias: Any,
    *,
    batch: int,
    num_query_heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    device: Any | None,
) -> tuple[int, int, int]:
    """Validate a rank-4 ``[B|1, Hq|1, Sq|1, Skv]`` fp32 bias tensor and return
    its ``(batch, head, sq)`` element strides, 0 on each broadcast (size-1) axis.

    ``device`` is the launch's Q device (``None`` skips the check). The largest
    element index the kernel can read must be below 2**31 (i32 addressing) and
    inside the tensor's storage. Raises ``ValueError`` for a malformed bias and
    ``NotImplementedError`` past the i32 limit (a too-big bias, not a malformed
    one). The key axis cannot broadcast: the kernels read ``col`` at unit stride.
    The unified kernels take caller-supplied strides and validate them with
    ``attention_unified._validate_additive_bias`` instead, through the same two
    checks.
    """
    check_additive_bias_tensor(bias, device)
    shape = tuple(int(n) for n in bias.shape)
    if len(shape) != 4:
        raise ValueError(
            f"additive_bias must be rank-4 [B|1, Hq|1, Sq|1, Skv], got {shape}"
        )
    for axis, (extent, full, label) in enumerate(
        (
            (shape[0], batch, "batch"),
            (shape[1], num_query_heads, "query head"),
            (shape[2], seqlen_q, "query"),
        )
    ):
        if extent not in (1, full):
            raise ValueError(
                f"additive_bias dim {axis} ({label}) must be 1 or {full}, got {extent}"
            )
    if shape[3] != seqlen_kv:
        raise ValueError(
            f"additive_bias key dim must equal seqlen_kv={seqlen_kv}, got {shape[3]}"
        )
    # A single-key axis is never stepped along, so its torch stride is meaningless.
    if seqlen_kv > 1 and int(bias.stride(3)) != 1:
        raise ValueError(
            "additive_bias must be contiguous along Skv (stride(3) == 1), "
            f"got strides {tuple(bias.stride())}"
        )
    strides = tuple(0 if shape[a] == 1 else int(bias.stride(a)) for a in range(3))
    # Each stride is an i32 kernarg; on a full axis max_idx >= stride (torch strides
    # are never negative), so the reach check also bounds every stride.
    max_idx = (
        (batch - 1) * strides[0]
        + (num_query_heads - 1) * strides[1]
        + (seqlen_q - 1) * strides[2]
        + (seqlen_kv - 1)
    )
    check_additive_bias_reach(bias, max_idx)
    return strides


def validate_dense_additive_bias(
    spec: Any, bias: Any, *, device: Any | None
) -> tuple[int, int, int] | None:
    """The dense runners' bias checks; returns the strides to launch with, or
    ``None`` when ``spec.use_additive_bias`` is off (a bias is then refused).

    The bias is sized by the spec (the maxima under varlen) and loaded unmasked,
    so a varlen runner must also bound its offsets by the spec maxima; that check
    is the runner's, which already owns the LSE path's identical one."""
    if not spec.use_additive_bias:
        if bias is not None:
            raise ValueError(
                "additive_bias provided but spec.use_additive_bias is False"
            )
        return None
    if bias is None:
        raise ValueError("spec.use_additive_bias=True requires additive_bias")
    return additive_bias_strides(
        bias,
        batch=int(spec.batch),
        num_query_heads=int(spec.num_query_heads),
        seqlen_q=int(spec.seqlen_q),
        seqlen_kv=int(spec.seqlen_kv),
        device=device,
    )
