# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Natural-log log-sum-exp (LSE) output from an existing base-2 softmax state.

Three consumers share the math here (``natural_lse_from_log2_stats``):

* the gfx942 / gfx950 dense prefill kernels: ``lse_store`` emits the optional
  LSE store, ``dense_row_has_key`` / ``zero_keyless_row`` the empty-row output
  masking, and ``validate_dense_lse`` is the host-side contract for the
  caller-owned LSE buffer;
* the unified split-KV reduce: the same IR helpers emit its LSE store, and
  ``validate_unified_lse`` is its host-side buffer contract.  Both validators
  sit on the common ``_check_lse_buffer``;
* the MFMA / WMMA attention forward inner bodies: ``make_*_lse_epilogue``
  produce *epilogue callables* for
  :class:`rocke.helpers.attention_fwd_ext.AttnFwdExt.epilogue_hook`.

Epilogue-hook contract
----------------------
The inner body calls the hook once per row slot after the K loop; the hook
stores one FP32 value per valid query row and returns the optional extra
row-validity predicate it was given (so the same predicate also zeroes the O
row).

* natural-log LSE: ``(m_log2 + log2(l)) * ln(2)`` where ``m_log2`` is the
  running max of the log2-domain scores and ``l`` the running sum of
  ``exp2(s - m_log2)``;
* stored as FP32;
* rows with no kept key store ``-inf``.  Whether a row is dead comes from the
  body's row validity (running-max sentinel and ``q_row < seqlen_q``) and from
  ``extra_valid`` when given -- never from ``l``;
* rows ``>= seqlen_q`` are not written at all.

The element index of row ``q_row`` of head ``h`` is
``(row_base + q_row) * row_stride + h * head_stride``.  Strides are in
elements and may be Python ints or i32 values.  Typical use:

* ``[B, H_q, S_q, 1]``: ``row_stride=1``, ``head_stride=S_q``, batch through
  ``batch_idx`` / ``batch_stride`` (a 64-bit pointer rebase);
* packed ``[T_q, H_q, 1]``: ``row_stride=H_q``, ``head_stride=1`` and
  ``row_base`` the first token of the sequence.

The MFMA accumulator replicates each row across 16 lanes and a lane owns 4
row slots; the WMMA accumulator gives a lane 8 row slots.  The body hands the
hook an ``is_row_leader`` predicate that is true on exactly one lane per row,
so every row is stored once; the two factories differ only in the slot count
they accept.  Nothing is allocated (no LDS, no scratch).
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from rocke.core.ir import I16, I32, I64, IRBuilder, Value, VectorType
from rocke.helpers.attention_fwd_ext import RowEpilogue, natural_lse_from_log2_stats

from ._attention_bwd_addr import as_i32

__all__ = [
    "LSE_ROWS_PER_LANE",
    "dense_row_has_key",
    "lse_row_value",
    "lse_store",
    "make_lse_epilogue",
    "make_mfma_lse_epilogue",
    "make_wmma_lse_epilogue",
    "validate_dense_lse",
    "zero_keyless_row",
]

# Row slots a lane holds per accumulator tile.
LSE_ROWS_PER_LANE = {"mfma": 4, "wmma": 8}


def lse_store(
    b: IRBuilder,
    dst: Value,
    element_offset: Value,
    m_log2: Value,
    l: Value,
    *,
    row_has_mass: Value,
) -> None:
    """Emit one FP32 store; the caller guards row existence and writer ownership."""
    safe_l = b.select(row_has_mass, l, b.const_f32(1.0))
    safe_m = b.select(row_has_mass, m_log2, b.const_f32(0.0))
    value = natural_lse_from_log2_stats(b, safe_m, safe_l)
    value = b.select(row_has_mass, value, b.const_f32(float("-inf")))
    b.global_store(dst, element_offset, value, align=4)


def dense_row_has_key(
    b, query_row, kv_length, *, causal, diagonal, window, right_bound=0
):
    """Key existence is independent of the finite score-mask sentinel.

    ``right_bound`` is the causal lookahead: the last visible key is
    ``diagonal + query_row + right_bound`` (ignored when not ``causal``).
    """
    last_key = b.sub(kv_length, b.const_i32(1))
    shifted_q = b.add(query_row, b.const_i32(diagonal)) if diagonal else query_row
    upper_q = b.add(shifted_q, b.const_i32(right_bound)) if right_bound else shifted_q
    upper = b.smin(last_key, upper_q) if causal else last_key
    lower = (
        b.smax(b.const_i32(0), b.sub(shifted_q, b.const_i32(window - 1)))
        if window
        else b.const_i32(0)
    )
    return b.cmp_le(lower, upper)


def zero_keyless_row(b, packed, row_has_key, dtype, width=4):
    """Zero O for rows without keys by masking the packed dtype bits.

    ``packed`` holds ``width`` 16-bit values (1, 2 or 4). A float select, before
    or after the cast, lets LLVM fuse multiply and convert into
    ``v_fma_mixlo_f16``, which changes nonempty FP16 O versus LSE-off; an
    integer AND on the same bits cannot fuse.
    """
    mask = b.select(row_has_key, b.const_i32(-1), b.const_i32(0))
    if width == 1:
        lane, lanes = I16, 1
        mask = b.trunc(mask, I16)
    elif width in (2, 4):
        lane, lanes = I32, width // 2
    else:
        raise ValueError(f"unsupported O store width {width}")
    bits = b.vector_and(
        b.bitcast(packed, VectorType(lane, lanes)), b.vector_splat(mask, lanes)
    )
    return b.bitcast(bits, VectorType(dtype, width))


def _check_lse_buffer(lse, q, expected, inputs):
    """Shared caller-owned LSE buffer contract: FP32, exact shape, contiguous,
    on q's GPU, and not overlapping any input or output tensor."""
    import torch

    if not isinstance(lse, torch.Tensor):
        raise ValueError("lse must be a torch tensor")
    if lse.dtype != torch.float32:
        raise ValueError("lse must have dtype float32")
    if tuple(lse.shape) != expected:
        raise ValueError(f"lse shape {tuple(lse.shape)} != {expected}")
    if not lse.is_contiguous():
        raise ValueError("lse must be contiguous")
    if not q.is_cuda or not lse.is_cuda or lse.device != q.device:
        raise ValueError("lse and q must be GPU tensors on the same device")

    lse_start = lse.data_ptr()
    lse_end = lse_start + lse.numel() * lse.element_size()
    for name, tensor in inputs.items():
        if tensor is None:
            continue
        if not tensor.is_cuda or tensor.device != q.device:
            raise ValueError(f"{name} must be on q's GPU device")
        if not tensor.numel():
            continue
        # Torch strides are nonnegative. Include holes conservatively when
        # checking the span of an input view against the contiguous output.
        span = 1 + sum(
            (extent - 1) * stride
            for extent, stride in zip(tensor.shape, tensor.stride())
        )
        start = tensor.data_ptr()
        end = start + span * tensor.element_size()
        if lse_start < end and start < lse_end:
            raise ValueError(f"lse overlaps {name}")


def validate_dense_lse(
    spec,
    q,
    k,
    v,
    out,
    lse,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    block_tables=None,
    kv_lens=None,
    sinks=None,
):
    """Validate only the new output contract; disabled callers keep their path."""
    if not spec.emit_lse:
        if lse is not None:
            raise ValueError("lse requires spec.emit_lse=True")
        return
    if lse is None:
        raise ValueError("emit_lse=True requires a caller-owned lse buffer")

    expected = (
        (q.shape[0], spec.num_query_heads, 1)
        if spec.varlen
        else (spec.batch, spec.num_query_heads, spec.seqlen_q, 1)
    )
    _check_lse_buffer(
        lse,
        q,
        expected,
        {
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_kv": cu_seqlens_kv,
            "block_tables": block_tables,
            "kv_lens": kv_lens,
            "sinks": sinks,
        },
    )

    if not spec.varlen:
        return
    for name, tensor, heads in (
        ("q", q, spec.num_query_heads),
        ("out", out, spec.num_query_heads),
        ("k", k, spec.num_kv_heads),
        ("v", v, spec.num_kv_heads),
    ):
        if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (heads, spec.head_size):
            raise ValueError(
                f"{name} must have packed [tokens, heads, head_size] shape"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if q.shape[0] != out.shape[0] or k.shape[0] != v.shape[0]:
        raise ValueError("packed output/value extents must match query/key extents")
    for name, tensor, extent, tile, maximum, allow_empty in (
        ("cu_seqlens_q", cu_seqlens_q, q.shape[0], spec.block_m, spec.seqlen_q, True),
        (
            "cu_seqlens_kv",
            cu_seqlens_kv,
            k.shape[0],
            spec.block_n,
            spec.seqlen_kv,
            False,
        ),
    ):
        if str(tensor.dtype) != "torch.int32" or tuple(tensor.shape) != (
            spec.batch + 1,
        ):
            raise ValueError(f"{name} must be int32 [batch+1]")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        # The existing packed body has no per-sequence load-tail handling.
        # This enabled-path content check synchronizes device offsets to host.
        offsets = tensor.tolist()
        if offsets[0] != 0 or offsets[-1] != extent:
            raise ValueError(f"{name} endpoints must span its packed tensor")
        for start, stop in zip(offsets, offsets[1:]):
            length = stop - start
            if length < 0 or length > maximum or length % tile:
                raise ValueError(
                    f"{name} lengths must be monotone, tile-aligned, and within spec"
                )
            if not allow_empty and length == 0:
                raise ValueError("LSE-enabled packed KV sequences must be nonempty")


Stride = Union[int, Value]
ExtraValid = Callable[[IRBuilder, RowEpilogue], Optional[Value]]


def _scaled(b: IRBuilder, idx: Value, stride: Stride) -> Value:
    # A literal stride of 1 is the common contiguous case; skip the multiply.
    if isinstance(stride, int) and stride == 1:
        return idx
    return b.mul(idx, as_i32(b, stride))


def lse_row_value(b: IRBuilder, e: RowEpilogue, valid: Value) -> Value:
    """FP32 natural-log LSE of the row, ``-inf`` where ``valid`` is false."""
    return b.select(
        valid,
        natural_lse_from_log2_stats(b, e.m_log2, e.l),
        b.const_f32(float("-inf")),
    )


def make_lse_epilogue(
    b: IRBuilder,
    lse: Value,
    *,
    head_idx: Value,
    row_stride: Stride,
    head_stride: Stride,
    max_slots: int,
    row_base: Optional[Value] = None,
    batch_idx: Optional[Value] = None,
    batch_stride: Optional[Stride] = None,
    extra_valid: Optional[ExtraValid] = None,
) -> Callable[[IRBuilder, RowEpilogue], Optional[Value]]:
    """Return an epilogue hook that stores the natural-log LSE.

    ``lse`` is the FP32 output pointer.  With ``batch_idx`` and
    ``batch_stride`` (elements) the pointer is rebased once, here, with a
    64-bit byte offset, so the per-row index stays 32-bit.  ``extra_valid``
    runs once per row slot and may return an i1 that is ANDed into the row
    validity; the same i1 is returned to the body so the O row is zeroed
    consistently.  Slots at or beyond ``max_slots`` are rejected at build time.
    """
    if (batch_idx is None) != (batch_stride is None):
        raise ValueError("batch_idx and batch_stride must be given together")
    base = lse
    if batch_idx is not None:
        off = b.mul(b.zext(batch_idx, I64), b.zext(as_i32(b, batch_stride), I64))
        base = b.global_ptr_add(lse, b.mul(off, b.const_i64(4)))

    def hook(bb: IRBuilder, e: RowEpilogue) -> Optional[Value]:
        if not 0 <= e.slot < max_slots:
            raise ValueError(
                f"row slot {e.slot} outside 0..{max_slots - 1} for this body"
            )
        extra = extra_valid(bb, e) if extra_valid is not None else None
        valid = e.row_valid if extra is None else bb.land(e.row_valid, extra)
        row = e.q_row if row_base is None else bb.add(row_base, e.q_row)
        idx = bb.add(_scaled(bb, row, row_stride), _scaled(bb, head_idx, head_stride))
        store_ok = e.is_row_leader
        if e.in_range is not None:
            store_ok = bb.land(store_ok, e.in_range)
        value = lse_row_value(bb, e, valid)
        with bb.scf_if(store_ok):
            bb.global_store(base, idx, value, align=4)
        return extra

    return hook


def make_mfma_lse_epilogue(b: IRBuilder, lse: Value, **layout):
    """LSE epilogue for the MFMA body (4 row slots per lane)."""
    return make_lse_epilogue(b, lse, max_slots=LSE_ROWS_PER_LANE["mfma"], **layout)


def make_wmma_lse_epilogue(b: IRBuilder, lse: Value, **layout):
    """LSE epilogue for the WMMA body (8 row slots per lane)."""
    return make_lse_epilogue(b, lse, max_slots=LSE_ROWS_PER_LANE["wmma"], **layout)


def validate_unified_lse(
    problem,
    lse,
    *,
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    sinks=None,
    alibi_slopes=None,
    qq_bias=None,
    additive_bias=None,
):
    """Caller-owned LSE contract for unified attention: FP32 [total_q, Hq, 1].

    The row extent is ``problem.total_q`` -- the reduce grid's x extent -- so the
    buffer is tied to the rows the kernel writes, not to the caller's q view."""
    if not problem.emit_lse:
        if lse is not None:
            raise ValueError("lse requires problem.emit_lse=True")
        return
    if lse is None:
        raise ValueError("emit_lse=True requires a caller-owned lse buffer")
    _check_lse_buffer(
        lse,
        q,
        (problem.total_q, problem.num_query_heads, 1),
        {
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_k": seqused_k,
            "block_table": block_table,
            "sinks": sinks,
            "alibi_slopes": alibi_slopes,
            "qq_bias": qq_bias,
            "additive_bias": additive_bias,
        },
    )
