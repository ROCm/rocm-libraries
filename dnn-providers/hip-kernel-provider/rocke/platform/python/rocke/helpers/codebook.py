# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Codebook dequantisation: int4 / int8 -> {fp8e4m3, bf8e5m2}.

The Sage attention int variants (i8 / i4 paths in CK Tile
``49_sageattention``) store K and V in 8-bit or 4-bit signed integers
plus a per-tensor or per-block scale. The inner attention loop
re-materialises the fp8 mantissa via a codebook lookup:

 int8 -> fp8e4m3 : table[i + 128]
 int4 -> fp8e4m3 : table[i + 8]
 int4 -> bf8e5m2 : table_bf8[i + 8]

The codebook tables are 256-entry (i8) or 16-entry (i4) f32 arrays
that the host prebuilds once and the kernel reads via a single global
load + ``cvt_f32_to_fp8`` / ``cvt_f32_to_bf8`` per element.
"""

from __future__ import annotations

from typing import Tuple

from ..core.ir import F32, IRBuilder, PtrType, Value
from .i4_dequant import unpack_i4_byte_to_pair_i32


__all__ = [
    "apply_per_tensor_scale",
    "codebook_lookup_i4_pair_to_bf8",
    "codebook_lookup_i4_pair_to_f32",
    "codebook_lookup_i4_pair_to_fp8",
    "codebook_lookup_i8_to_bf8",
    "codebook_lookup_i8_to_f32",
    "codebook_lookup_i8_to_fp8",
]


def _validate_codebook_ptr(codebook_ptr: Value, *, expected_entries: int) -> None:
    if not isinstance(codebook_ptr.type, PtrType):
        raise ValueError("codebook pointer must be a typed pointer")
    if codebook_ptr.type.pointee != F32:
        raise ValueError(
            f"codebook must be ptr<f32>, got ptr<{codebook_ptr.type.pointee.name}>"
        )
    if expected_entries not in (16, 256):
        raise ValueError(
            f"unsupported codebook entry count {expected_entries}; expected 16 or 256"
        )


def _codebook_i8_f32(
    b: IRBuilder,
    codebook_ptr: Value,
    i8_value_i32: Value,
    *,
    per_tensor_scale: Value | None = None,
) -> Value:
    """i8 -> f32 codebook core: ``idx = i + 128; load; optional scale``.

    Shared body of the i8 lookups; the public ``_to_f32`` / ``_to_fp8``
    / ``_to_bf8`` wrappers add only the trailing ``cvt`` (if any). Emits
    the identical op stream the wrappers used to emit inline.
    """
    _validate_codebook_ptr(codebook_ptr, expected_entries=256)
    if i8_value_i32.type.name != "i32":
        raise ValueError(
            f"codebook_lookup_i8 expects i32 (sign-extended i8) input, "
            f"got {i8_value_i32.type.name}"
        )
    idx = b.add(i8_value_i32, b.const_i32(128))
    f32_v = b.global_load_f32(codebook_ptr, idx)
    if per_tensor_scale is not None:
        f32_v = b.fmul(f32_v, per_tensor_scale)
    return f32_v


def _codebook_i4_pair_f32(
    b: IRBuilder,
    codebook_ptr: Value,
    packed_byte_i8: Value,
    *,
    per_tensor_scale: Value | None = None,
) -> Tuple[Value, Value]:
    """i4-pair -> (f32, f32) codebook core: unpack + 2 loads + optional scale.

    Shared body of the i4-pair lookups; the public ``_to_f32`` /
    ``_to_fp8`` / ``_to_bf8`` wrappers add only the trailing ``cvt``
    (if any). Emits the identical op stream the wrappers used inline.
    """
    _validate_codebook_ptr(codebook_ptr, expected_entries=16)
    if packed_byte_i8.type.name != "i8":
        raise ValueError(
            f"codebook_lookup_i4_pair expects i8 input "
            f"(load the packed byte via global_load(..., I8) first), "
            f"got {packed_byte_i8.type.name}"
        )
    lo_i32, hi_i32 = unpack_i4_byte_to_pair_i32(b, packed_byte_i8)
    lo_idx = b.add(lo_i32, b.const_i32(8))
    hi_idx = b.add(hi_i32, b.const_i32(8))
    lo_f = b.global_load_f32(codebook_ptr, lo_idx)
    hi_f = b.global_load_f32(codebook_ptr, hi_idx)
    if per_tensor_scale is not None:
        lo_f = b.fmul(lo_f, per_tensor_scale)
        hi_f = b.fmul(hi_f, per_tensor_scale)
    return lo_f, hi_f


def codebook_lookup_i8_to_f32(
    b: IRBuilder,
    codebook_ptr: Value,
    i8_value_i32: Value,
    *,
    per_tensor_scale: Value | None = None,
) -> Value:
    """Direct i8 -> f32 codebook lookup (skips the ``cvt_f32_to_fp8``
    round-trip the existing :func:`codebook_lookup_i8_to_fp8` does).

    The native MFMA i8/i4 attention path (P30) consumes f32 partials
    directly; the round-trip through fp8 is unnecessary there. This
    helper saves the ``cvt.f32.fp8`` per element while keeping the
    same codebook semantics.
    """
    return _codebook_i8_f32(
        b, codebook_ptr, i8_value_i32, per_tensor_scale=per_tensor_scale
    )


def codebook_lookup_i4_pair_to_f32(
    b: IRBuilder,
    codebook_ptr: Value,
    packed_byte_i8: Value,
    *,
    per_tensor_scale: Value | None = None,
) -> Tuple[Value, Value]:
    """Direct i4-pair -> f32 codebook lookup (skips the fp8 round-trip).

    Companion to :func:`codebook_lookup_i8_to_f32` for the 4-bit
    variant of the native MFMA i4 attention path (P30).
    """
    return _codebook_i4_pair_f32(
        b, codebook_ptr, packed_byte_i8, per_tensor_scale=per_tensor_scale
    )


def codebook_lookup_i8_to_fp8(
    b: IRBuilder,
    codebook_ptr: Value,
    i8_value_i32: Value,
    *,
    per_tensor_scale: Value | None = None,
) -> Value:
    """One i8 -> f32 (codebook) -> fp8e4m3 lookup.

    Args:
    codebook_ptr: ``ptr<f32>`` -- the 256-entry codebook
    (indexed by ``i + 128``).
    i8_value_i32: ``i32`` -- the sign-extended i8 input.
    per_tensor_scale: optional f32 -- if set, multiplies the
    f32 codebook output before the fp8 round.
    """
    f32_v = _codebook_i8_f32(
        b, codebook_ptr, i8_value_i32, per_tensor_scale=per_tensor_scale
    )
    return b.cvt_f32_to_fp8(f32_v)


def codebook_lookup_i8_to_bf8(
    b: IRBuilder,
    codebook_ptr: Value,
    i8_value_i32: Value,
    *,
    per_tensor_scale: Value | None = None,
) -> Value:
    """Bf8e5m2 sibling of :func:`codebook_lookup_i8_to_fp8`."""
    f32_v = _codebook_i8_f32(
        b, codebook_ptr, i8_value_i32, per_tensor_scale=per_tensor_scale
    )
    return b.cvt_f32_to_bf8(f32_v)


def codebook_lookup_i4_pair_to_fp8(
    b: IRBuilder,
    codebook_ptr: Value,
    packed_byte_i8: Value,
    *,
    per_tensor_scale: Value | None = None,
) -> Tuple[Value, Value]:
    """One packed-i4 byte -> two fp8e4m3 elements via codebook.

    Args:
    codebook_ptr: ``ptr<f32>`` -- 16-entry codebook (indexed by ``i + 8``).
    packed_byte_i8: ``i8`` -- the byte holding two i4 nibbles.
    per_tensor_scale: optional f32 -- applied before the fp8 round.
    """
    lo_f, hi_f = _codebook_i4_pair_f32(
        b, codebook_ptr, packed_byte_i8, per_tensor_scale=per_tensor_scale
    )
    return b.cvt_f32_to_fp8(lo_f), b.cvt_f32_to_fp8(hi_f)


def codebook_lookup_i4_pair_to_bf8(
    b: IRBuilder,
    codebook_ptr: Value,
    packed_byte_i8: Value,
    *,
    per_tensor_scale: Value | None = None,
) -> Tuple[Value, Value]:
    """Bf8e5m2 sibling of :func:`codebook_lookup_i4_pair_to_fp8`."""
    lo_f, hi_f = _codebook_i4_pair_f32(
        b, codebook_ptr, packed_byte_i8, per_tensor_scale=per_tensor_scale
    )
    return b.cvt_f32_to_bf8(lo_f), b.cvt_f32_to_bf8(hi_f)


def apply_per_tensor_scale(b: IRBuilder, value_f32: Value, scale: Value) -> Value:
    """``value_f32 * scale`` -- one ``v_mul_f32`` on AMDGPU.

    Exposed as a discrete helper so the kernel author can defer the
    scale application until after a chain of fused operations.
    """
    if value_f32.type.name != "f32" or scale.type.name != "f32":
        raise ValueError(
            f"apply_per_tensor_scale expects f32 inputs, got "
            f"{value_f32.type.name} / {scale.type.name}"
        )
    return b.fmul(value_f32, scale)
