# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""i4-packed buffer load + dequant helper.

CK Tile's i4-quant weight path (``38_block_scale_gemm`` i4-fp8 variants,
the fused-MoE w4a8 path, and the bf8 / fp8 ``preshuffleb`` columns of
the same example) stores B-matrix weights as 4-bit signed integers
packed two-per-byte. The dequant pipeline is the same in every variant::

 1. Load a packed byte buffer (each byte holds two i4 values).
 2. Unpack into a pair of i8 values (sign-extending the high nibble
 from 4 bits to 8 bits).
 3. Cast to f32 via ``sitofp``.
 4. Multiply by the per-block scale (fp32 or fp8 depending on the
 variant).
 5. Cast back to the MFMA input dtype (fp8e4m3 or bf8e5m2 for the
 FP8 MFMA path).

This module exposes the unpack + scale-apply chain as one helper so the
fused-MoE / block-scale-GEMM call sites don't have to re-derive the
nibble-shift + sext sequence each time. The actual per-block scale
load lives in :mod:`ck_dsl.helpers.mx_scale` (when the scale buffer is
shared-exponent format) or is a plain per-row scalar load (when the
scale is fp32 per-tile).

What v1 ships:

* :func:`unpack_i4_byte_to_pair_f32`: one packed byte -> two f32 values.
* :func:`dequant_i4_byte_to_fp8_pair`: full chain to two
 ``fp8e4m3`` lanes (the common FP8-MFMA prep).
* :func:`dequant_i4_byte_to_bf8_pair`: bf8e5m2 sibling.

The helpers operate on single bytes; vector forms are composed from
these in the kernel author's K-loop body so the per-thread unroll
stays explicit.
"""

from __future__ import annotations

from typing import Tuple

from ..core.ir import I32, IRBuilder, Value
from .quant import quantize_scalar_f32


__all__ = [
    "dequant_i4_byte_to_bf8_pair",
    "dequant_i4_byte_to_fp8_pair",
    "unpack_i4_byte_to_pair_f32",
    "unpack_i4_byte_to_pair_i8",
]


def unpack_i4_byte_to_pair_i32(b: IRBuilder, packed_byte: Value) -> Tuple[Value, Value]:
    """One byte of packed i4 weights -> two sign-extended i32 values.

    Layout assumption (matches CK Tile's i4 weight packer):

    .. code-block:: text

    byte = (high_nibble << 4) | (low_nibble & 0x0F)
    low = sign-extend(byte & 0x0F, 4 -> 32) # via ``(byte << 28) >> 28``
    high = sign-extend(byte >> 4, 4 -> 32) # via ``(byte << 24) >> 28``

    Outputs are i32 in the canonical i4 signed range ``[-8, 7]``.

    Why i32 (not i8): the IRBuilder's ``arith.shl`` / ``arith.div``
    pair lowers to a single ``v_bfe_i32`` (bit-field-extract signed)
    on the AMDGPU backend, so producing i32 directly avoids an extra
    sign-extend round-trip downstream of the sitofp. Callers that
    want to store the unpacked values back as i8 should ``trunc`` the
    result via a separate IR op.
    """
    if packed_byte.type.name != "i8":
        raise ValueError(
            f"unpack_i4_byte_to_pair_i32 expects i8 input, got {packed_byte.type.name}"
        )
    byte_i32 = b.sext(packed_byte, I32)
    # Sign-extend each nibble via mask + conditional subtract. The
    # earlier ``(x << 28) >> 28`` idiom can't use ``arith.div`` because
    # signed division truncates toward zero while arithmetic shift
    # floors; mask + branchless subtract is the correct + portable
    # equivalent that the AMDGPU backend folds to ``v_bfe_i32``.
    mask_lo = b.const_i32(0x0F)
    c8 = b.const_i32(8)
    c16 = b.const_i32(16)
    low_unsigned = b.land(byte_i32, mask_lo)
    high_unsigned = b.land(b.lshr(byte_i32, b.const_i32(4)), mask_lo)
    low_signed = b.select(
        b.cmp_ge(low_unsigned, c8),
        b.sub(low_unsigned, c16),
        low_unsigned,
    )
    high_signed = b.select(
        b.cmp_ge(high_unsigned, c8),
        b.sub(high_unsigned, c16),
        high_unsigned,
    )
    return low_signed, high_signed


def unpack_i4_byte_to_pair_i8(b: IRBuilder, packed_byte: Value) -> Tuple[Value, Value]:
    """One byte of packed i4 -> two signed i32 values.

    Returns ``i32`` per element today (see
    :func:`unpack_i4_byte_to_pair_i32` rationale). The name is kept
    for parity with the conceptual i8 view; downstream paths that
    want a literal i8 type can ``trunc`` once explicitly.
    """
    return unpack_i4_byte_to_pair_i32(b, packed_byte)


def unpack_i4_byte_to_pair_f32(b: IRBuilder, packed_byte: Value) -> Tuple[Value, Value]:
    """One byte of packed i4 -> two f32 values (signed, in ``[-8, 7]``)."""
    low_i32, high_i32 = unpack_i4_byte_to_pair_i32(b, packed_byte)
    return b.sitofp_f32(low_i32), b.sitofp_f32(high_i32)


def dequant_i4_byte_to_fp8_pair(
    b: IRBuilder,
    packed_byte: Value,
    *,
    inv_scale: Value,
) -> Tuple[Value, Value]:
    """Full i4 -> fp8e4m3 dequant for one packed byte.

    Pipeline (per element)::

    i32 = sext(sign_extract(byte, 4-bit field))
    f32 = sitofp(i32)
    fp8 = cvt_f32_to_fp8(clamp(f32 * inv_scale, -448, 448))

    Returns ``(low_fp8, high_fp8)``. Composes the dequant + quant
    pipeline via :func:`ck_dsl.helpers.quantize_scalar_f32` so the
    clamp constants stay aligned with the rest of the FP8 quant family.
    """
    low_f32, high_f32 = unpack_i4_byte_to_pair_f32(b, packed_byte)
    low_fp8 = quantize_scalar_f32(b, low_f32, inv_scale=inv_scale, qdtype="fp8e4m3")
    high_fp8 = quantize_scalar_f32(b, high_f32, inv_scale=inv_scale, qdtype="fp8e4m3")
    return low_fp8, high_fp8


def dequant_i4_byte_to_bf8_pair(
    b: IRBuilder,
    packed_byte: Value,
    *,
    inv_scale: Value,
) -> Tuple[Value, Value]:
    """bf8e5m2 sibling of :func:`dequant_i4_byte_to_fp8_pair`."""
    low_f32, high_f32 = unpack_i4_byte_to_pair_f32(b, packed_byte)
    low_bf8 = quantize_scalar_f32(b, low_f32, inv_scale=inv_scale, qdtype="bf8e5m2")
    high_bf8 = quantize_scalar_f32(b, high_f32, inv_scale=inv_scale, qdtype="bf8e5m2")
    return low_bf8, high_bf8
