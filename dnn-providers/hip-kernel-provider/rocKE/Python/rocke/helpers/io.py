# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dtype-dispatched I/O helpers.

This module deduplicates the small ``_io_type`` / ``_load_vec`` /
``_store_vec`` / ``_global_load_scalar`` blocks that every small-op
file (elementwise, layernorm, rmsnorm, reduce, transpose) was copying
verbatim. The helpers are dtype-string-tolerant: ``"f16"``, ``"fp16"``,
and ``"bf16"`` all resolve to the canonical IR type, so ports of CK
Tile reference kernels (which use ``"fp16"``) don't have to translate.

The convention matches CK Tile's ``type_convert<DstT, SrcT>`` family in
``include/ck_tile/core/numeric/type_convert.hpp``: I/O is always at the
native storage dtype (f16/bf16); compute is in f32 (the
``ComputeDataType`` alias in every CK Tile op's TypeConfig). The
:func:`load_vec_as_f32` / :func:`load_scalar_as_f32` convenience wrap
the most common dtype-promote-on-read pattern.
"""

from __future__ import annotations

from typing import Literal

from ..core.ir import BF16, F16, IRBuilder, Type, Value


__all__ = [
    "DType",
    "io_ir_type",
    "load_lane_slice_f32",
    "load_scalar",
    "load_scalar_as_f32",
    "load_vec",
    "load_vec_as_f32",
    "pack_f32_to",
    "pack_quant_chunk_f32",
    "store_packed_chunk",
    "store_scalar",
    "store_scalar_from_f32",
    "store_vec",
    "vector_row_copy",
]


# Aliases the small ops use. ``"fp16"`` is the CK Tile / ``UniversalGemm``
# spelling; we accept it to make port snippets read literally.
DType = Literal["f16", "fp16", "bf16"]


def io_ir_type(dtype: str) -> Type:
    """Map a dtype string to the canonical IR type object.

    Accepts ``"f16"``, ``"fp16"`` (alias), or ``"bf16"``. Raises
    :class:`ValueError` for any other value -- f8 and i8 paths go
    through their own helpers because their compute dtype isn't f32.
    """
    if dtype in ("f16", "fp16"):
        return F16
    if dtype == "bf16":
        return BF16
    raise ValueError(f"unsupported I/O dtype {dtype!r}; expected f16/fp16/bf16")


def load_scalar(b: IRBuilder, ptr: Value, idx: Value, *, dtype: str) -> Value:
    """One scalar global load. Returns a value in the native dtype.

    Caller is responsible for any subsequent ``cast_to_f32`` if compute
    needs f32. Use :func:`load_scalar_as_f32` for the common
    "load + promote" pattern.
    """
    if dtype in ("f16", "fp16"):
        return b.global_load_f16(ptr, idx)
    if dtype == "bf16":
        return b.global_load_bf16(ptr, idx)
    raise ValueError(f"unsupported I/O dtype {dtype!r}")


def load_scalar_as_f32(b: IRBuilder, ptr: Value, idx: Value, *, dtype: str) -> Value:
    """One scalar global load promoted to f32.

    Equivalent to ``cast_to_f32(load_scalar(...))``; the helper exists
    because every norm / reduce / elementwise tail path does exactly
    this two-step.
    """
    return b.cast_to_f32(load_scalar(b, ptr, idx, dtype=dtype))


def load_vec(b: IRBuilder, ptr: Value, idx: Value, *, dtype: str, n: int) -> Value:
    """Vectorised global load of ``n`` consecutive elements.

    Supports ``n in {2, 4, 8}`` for f16/bf16; ``n=1`` is rejected
    because the IR distinguishes scalar vs vector loads and the n=1
    case would silently lose vectorisation. Use :func:`load_scalar`
    when a scalar is what you want.
    """
    if n not in (2, 4, 8):
        raise ValueError(f"load_vec n must be 2/4/8 (got {n}); use load_scalar for n=1")
    ty = io_ir_type(dtype)
    if dtype in ("f16", "fp16"):
        return b.global_load_vN_f16(ptr, idx, n)
    return b.global_load_vN(ptr, idx, ty, n)


def load_vec_as_f32(
    b: IRBuilder, ptr: Value, idx: Value, *, dtype: str, n: int
) -> list[Value]:
    """Vectorised load + per-lane f32 promotion.

    Returns a list of ``n`` scalar f32 :class:`Value`\\s, one per element
    of the loaded vector. This is the canonical "ingest into f32 compute
    registers" pattern used by every norm/reduce kernel.
    """
    v = load_vec(b, ptr, idx, dtype=dtype, n=n)
    return [b.cast_to_f32(b.vec_extract(v, i)) for i in range(n)]


# Power-of-two vector widths the DSL's ``global_load_vN`` covers. EPT
# values outside this set (1 / 3 for head_size 64 / 192) fall back to
# per-element scalar loads so every supported head size keeps working.
_VEC_WIDTHS = (2, 4, 8)


def load_lane_slice_f32(
    b: IRBuilder,
    ptr: Value,
    row_base: Value,
    lane_d_base: Value,
    *,
    dtype: str,
    ept: int,
) -> list[Value]:
    """Load this lane's ``EPT`` consecutive elements as a list of f32.

    Uses one vectorised ``global_load_vN`` + ``vec_extract`` chain when
    ``EPT`` is a supported vector width (2 / 4 / 8 -- one VMEM
    transaction per call). Falls back to per-element scalar loads for
    the ``EPT == 1`` (head_size=64) and ``EPT == 3`` (head_size=192,
    not a power of two) corner cases so every supported head size keeps
    working.

    Promoted from the AST-identical local ``_load_lane_slice_f32`` in
    :mod:`rocke.instances.common.fmha_bwd` and
    :mod:`rocke.instances.common.fmha_splitkv_decode`. Matches the
    per-warp K/V/Q load pattern used by CK Tile's ``BlockFmhaBwd*``
    register-tile loads (``load_tile`` over a distributed
    ``rt<bf16, ..., row_l, rt_16x32_s>`` tensor) and AITER's varlen bwd
    ``tl.load(ptr + offs, ...)`` 8-element vector loads.
    """
    if ept in _VEC_WIDTHS:
        return load_vec_as_f32(b, ptr, b.add(row_base, lane_d_base), dtype=dtype, n=ept)
    return [
        load_scalar_as_f32(
            b,
            ptr,
            b.add(row_base, b.add(lane_d_base, b.const_i32(k))),
            dtype=dtype,
        )
        for k in range(ept)
    ]


def store_scalar(
    b: IRBuilder, ptr: Value, idx: Value, value: Value, *, dtype: str
) -> None:
    """One scalar global store.

    ``value`` must be in the native dtype already. Use
    :func:`store_scalar_from_f32` to handle the f32 -> {f16, bf16}
    cast for you.
    """
    if dtype not in ("f16", "fp16", "bf16"):
        raise ValueError(f"unsupported I/O dtype {dtype!r}")
    b.global_store(ptr, idx, value)


def store_scalar_from_f32(
    b: IRBuilder, ptr: Value, idx: Value, value_f32: Value, *, dtype: str
) -> None:
    """Trunc an f32 value to ``dtype`` and store it scalar.

    Most norm/reduce kernels keep the per-element accumulator in f32
    and want to fuse the trunc + store. This is the helper for that
    last mile.
    """
    target = io_ir_type(dtype)
    b.global_store(ptr, idx, b.cast_f32_to(value_f32, target))


def store_vec(b: IRBuilder, ptr: Value, idx: Value, value: Value, *, n: int) -> None:
    """Vectorised global store. ``value`` must already be a ``<n x T>``
    vector in the target dtype (use :func:`pack_f32_to` to assemble
    one from a list of f32 scalars).
    """
    if n not in (2, 4, 8):
        raise ValueError(
            f"store_vec n must be 2/4/8 (got {n}); use store_scalar for n=1"
        )
    b.global_store_vN(ptr, idx, value, n)


def pack_f32_to(b: IRBuilder, scalars_f32: list[Value], *, dtype: str) -> Value:
    """Trunc a list of f32 scalars to ``dtype`` and pack into a vector.

    The dual of :func:`load_vec_as_f32`: every norm / elementwise
    kernel finishes with this exact pattern before writing back to
    global. Returns a ``<len(scalars_f32) x dtype>`` vector.
    """
    target = io_ir_type(dtype)
    casts = [b.cast_f32_to(v, target) for v in scalars_f32]
    return b.vec_pack(casts, target)


def vector_row_copy(
    b: IRBuilder,
    *,
    src: Value,
    dst: Value,
    src_base: Value,
    dst_base: Value,
    H: int,
    dtype: str,
    vec_bytes: int = 16,
) -> None:
    """Vectorised row copy along a head / hidden dimension.

    Promotes the inline ``_copy_row_vec`` from
    :mod:`rocke.instances.common.fmha_appendkv` into a shared helper so the
    same 16-byte-vector pattern can be reused by ``moe_gather`` and
    ``fmha_bwd`` postlude.

    The copy issues ``H // (vec_bytes / sizeof(elem))`` aligned vmem
    transactions plus a scalar tail for any residual elements; the
    AMDGPU backend coalesces the vector form into a single
    ``buffer_load_dwordx4`` + ``buffer_store_dwordx4`` pair when
    pointer alignment supports it. Matches CK Tile's
    ``store_tile(k_dram_block_window, knew_tile)`` semantics in
    ``include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_appendkv_pipeline.hpp``.
    """
    ty = io_ir_type(dtype)
    elem_bytes = 2  # f16 / bf16 storage width
    vec = vec_bytes // elem_bytes
    if vec not in (2, 4, 8):
        raise ValueError(
            f"vector_row_copy: vec_bytes {vec_bytes} maps to vec={vec}; "
            "expected 4/8/16-byte aligned"
        )
    n_chunks = H // vec
    for c in range(n_chunks):
        d = c * vec
        src_addr = b.add(src_base, b.const_i32(d))
        dst_addr = b.add(dst_base, b.const_i32(d))
        v = b.global_load_vN(src, src_addr, ty, vec, align=vec * elem_bytes)
        b.global_store_vN(dst, dst_addr, v, vec, align=vec * elem_bytes)
    for d in range(n_chunks * vec, H):
        s = load_scalar_as_f32(b, src, b.add(src_base, b.const_i32(d)), dtype=dtype)
        store_scalar_from_f32(b, dst, b.add(dst_base, b.const_i32(d)), s, dtype=dtype)


def pack_quant_chunk_f32(
    b: IRBuilder,
    chunk_f32: list[Value],
    *,
    qdtype: str,
) -> Value:
    """Pack a list of 4 f32 scalars into a single packed quant vector.

    Promotes the inline ``_pack_quant_chunk_f32`` from
    :mod:`rocke.instances.common.smoothquant` (and the duplicate copy in
    :mod:`rocke.instances.common.moe_smoothquant`) into a shared helper.

    Routes through the matching packed cvt primitive on the IR:

    * ``"fp8e4m3"`` / ``"fp8"`` — :meth:`IRBuilder.cvt_pk_fp8_f32x4`
    * ``"bf8e5m2"`` / ``"bf8"`` — :meth:`IRBuilder.cvt_pk_bf8_f32x4`
    * ``"i8"`` — :meth:`IRBuilder.cvt_pk_i8_f32x4` (P09)

    The caller still owns the scale-multiply + clamp prelude (see
    :func:`rocke.helpers.quant.quantize_scalar_f32`); this helper
    only does the packed cvt over a chunk of 4 f32 scalars.
    """
    if len(chunk_f32) != 4:
        raise ValueError(
            f"pack_quant_chunk_f32 expects exactly 4 f32 scalars, got {len(chunk_f32)}"
        )
    from ..core.ir import F32

    vec_in = b.vec_pack(chunk_f32, F32)
    if qdtype in ("fp8", "fp8e4m3"):
        return b.cvt_pk_fp8_f32x4(vec_in)
    if qdtype in ("bf8", "bf8e5m2"):
        return b.cvt_pk_bf8_f32x4(vec_in)
    if qdtype == "i8":
        return b.cvt_pk_i8_f32x4(vec_in)
    raise ValueError(
        f"pack_quant_chunk_f32: unsupported qdtype {qdtype!r}; expected fp8 / bf8 / i8"
    )


def store_packed_chunk(
    b: IRBuilder,
    *,
    dst: Value,
    dst_base: Value,
    packed_vec: Value,
) -> None:
    """Store a 4-byte packed quant vector to global memory.

    Companion to :func:`pack_quant_chunk_f32`: takes the ``<4 x i8>``
    /``<4 x fp8e4m3>`` / ``<4 x bf8e5m2>`` produced by the cvt and
    emits a single 4-byte global store. AMDGPU's backend coalesces
    adjacent lanes' chunks into one ``global_store_dwordx<N>``
    transaction when alignments line up.
    """
    b.global_store_vN(dst, dst_base, packed_vec, 4, align=4)
