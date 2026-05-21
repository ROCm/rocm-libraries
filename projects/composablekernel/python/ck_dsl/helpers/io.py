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
    "load_scalar",
    "load_scalar_as_f32",
    "load_vec",
    "load_vec_as_f32",
    "store_scalar",
    "store_scalar_from_f32",
    "store_vec",
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
