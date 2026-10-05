# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Natural-log LSE output from an existing base-2 softmax state.

Shared by the dense prefill kernels: the IR helpers emit the
optional LSE store and empty-row output masking; ``validate_dense_lse`` is the
host-side contract for the caller-owned LSE buffer.
"""

from rocke.core.ir import I16, I32, IRBuilder, Value, VectorType

_LN2 = 0.6931471805599453


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
    value = b.fmul(b.fadd(safe_m, b.log2(safe_l)), b.const_f32(_LN2))
    value = b.select(row_has_mass, value, b.const_f32(float("-inf")))
    b.global_store(dst, element_offset, value, align=4)


def dense_row_has_key(b, query_row, kv_length, *, causal, diagonal, window):
    """Key existence is independent of the finite score-mask sentinel."""
    last_key = b.sub(kv_length, b.const_i32(1))
    shifted_q = b.add(query_row, b.const_i32(diagonal)) if diagonal else query_row
    upper = b.smin(last_key, shifted_q) if causal else last_key
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

    import torch

    if not isinstance(lse, torch.Tensor):
        raise ValueError("lse must be a torch tensor")
    expected = (
        (q.shape[0], spec.num_query_heads, 1)
        if spec.varlen
        else (spec.batch, spec.num_query_heads, spec.seqlen_q, 1)
    )
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
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("out", out),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_kv", cu_seqlens_kv),
        ("block_tables", block_tables),
        ("kv_lens", kv_lens),
        ("sinks", sinks),
    ):
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
        if tensor.dtype != torch.int32 or tuple(tensor.shape) != (spec.batch + 1,):
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
