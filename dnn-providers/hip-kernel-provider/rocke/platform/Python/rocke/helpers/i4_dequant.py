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
load lives in :mod:`rocke.helpers.mx_scale` (when the scale buffer is
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
    "dequant_i4_byte_to_f16_pair",
    "dequant_i4_byte_to_fp8_pair",
    "unpack_fp4_byte_to_pair_f32",
    "unpack_fp6_bytes_to_quad_f32",
    "unpack_i4_byte_to_pair_f16",
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


def unpack_i4_byte_to_pair_f16(b: IRBuilder, packed_byte: Value) -> Tuple[Value, Value]:
    """One byte of packed i4 -> two f16 values (signed, in ``[-8, 7]``).

    f16 sibling of :func:`unpack_i4_byte_to_pair_f32`. The i4 range is
    exactly representable in f16, so this is a lossless ``sitofp`` to f32
    followed by a ``trunc`` to f16 (the AMDGPU backend has no direct
    int->f16 convert for the WMMA input dtype).
    """
    low_f32, high_f32 = unpack_i4_byte_to_pair_f32(b, packed_byte)
    return b.trunc_f32_to_f16(low_f32), b.trunc_f32_to_f16(high_f32)


def dequant_i4_byte_to_f16_pair(
    b: IRBuilder,
    packed_byte: Value,
    *,
    scale: Value,
) -> Tuple[Value, Value]:
    """Full i4 -> f16 dequant for one packed byte (WMMA prep).

    Pipeline (per element)::

        i32 = sext(sign_extract(byte, 4-bit field))
        f32 = sitofp(i32)
        f16 = trunc_f32_to_f16(f32 * scale)

    Returns ``(low_f16, high_f16)`` ready to feed ``wmma_f32_16x16x16_f16``.
    Unlike the fp8 / bf8 siblings this multiplies by the *forward* per-group
    ``scale`` (not an inverse) because the i4 weights are being reconstructed
    to their original magnitude, not re-quantised. The multiply runs in f32
    and the result is truncated once to f16 so the WMMA atom sees its native
    input dtype.
    """
    low_f32, high_f32 = unpack_i4_byte_to_pair_f32(b, packed_byte)
    low_f16 = b.trunc_f32_to_f16(b.fmul(low_f32, scale))
    high_f16 = b.trunc_f32_to_f16(b.fmul(high_f32, scale))
    return low_f16, high_f16


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
    pipeline via :func:`rocke.helpers.quantize_scalar_f32` so the
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


# ---------------------------------------------------------------------
# OCP MX fp4 / fp6 unpack (P54)
# ---------------------------------------------------------------------
#
# fp4 layout per OCP MX spec: 1 sign bit, 2 exponent bits (bias 1), 1
# mantissa bit. ``00 -> +0``, ``08 -> -0`` (treated as 0 in ML), denormal
# encoding ``01 -> 0.5``. We implement the canonical 16-entry codebook
# and lookup directly: simpler + faster than reconstructing the bit
# fields and matches OCP-spec semantics exactly.
#
# fp6 layout per OCP MX spec: 1 sign bit, 3 exponent bits (bias 3), 2
# mantissa bits. 64-entry codebook keyed by the raw 6-bit value;
# denormals at exponent 0 become 0.0 (ML convention).


_FP4_CODEBOOK = (
    +0.0,  # 0000
    +0.5,  # 0001
    +1.0,  # 0010
    +1.5,  # 0011
    +2.0,  # 0100
    +3.0,  # 0101
    +4.0,  # 0110
    +6.0,  # 0111
    -0.0,  # 1000
    -0.5,  # 1001
    -1.0,  # 1010
    -1.5,  # 1011
    -2.0,  # 1100
    -3.0,  # 1101
    -4.0,  # 1110
    -6.0,  # 1111
)


def _select_fp4_value(b: IRBuilder, code_i32: Value) -> Value:
    """Static-codebook select chain for one fp4 nibble.

    Generates a ternary chain of ``b.select`` calls keyed on
    ``code_i32 == k`` for ``k in 0..15``. Results in 16 selects;
    AMDGPU's MachineCombiner folds the chain into ``v_cmp_eq`` +
    ``v_cndmask`` per element which matches the codebook-lookup
    pattern used by AITER's i4 dequant path.
    """
    out = b.const_f32(0.0)
    for k, fv in enumerate(_FP4_CODEBOOK):
        is_k = b.cmp_eq(code_i32, b.const_i32(k))
        out = b.select(is_k, b.const_f32(float(fv)), out)
    return out


def unpack_fp4_byte_to_pair_f32(
    b: IRBuilder, packed_byte: Value
) -> Tuple[Value, Value]:
    """One packed fp4 byte -> two f32 values via the OCP MX codebook.

    Reference: OCP "MX FP4" spec — 1 sign bit + 2 exponent bits + 1
    mantissa bit. Denormals at exponent 0 are treated as 0.0; the
    codebook above lists every representable value.

    Layout assumption (matches the i4 unpack convention):
    ``low = byte & 0xF`` (bits 0..3), ``high = (byte >> 4) & 0xF``
    (bits 4..7).
    """
    if packed_byte.type.name != "i8":
        raise ValueError(
            f"unpack_fp4_byte_to_pair_f32 expects i8 packed byte, got "
            f"{packed_byte.type.name}"
        )
    byte_i32 = b.zext(packed_byte, I32)
    low_code = b.land(byte_i32, b.const_i32(0xF))
    high_code = b.land(b.lshr(byte_i32, b.const_i32(4)), b.const_i32(0xF))
    return _select_fp4_value(b, low_code), _select_fp4_value(b, high_code)


# 64-entry fp6 codebook per OCP MX spec.
_FP6_CODEBOOK: Tuple[float, ...] = tuple(
    (
        # Positive sign (bit 5 = 0).
        # Exponent = 0 → all denormals collapse to 0.0 (ML convention).
        # Exponent in 1..7: value = 2^(e-3) × (1 + m/4).
        (1.0 if (i & 0b011111) == 0 else 0.0)
        if ((i >> 2) & 0b111) == 0
        else (2.0 ** (((i >> 2) & 0b111) - 3) * (1.0 + (i & 0b11) / 4.0))
    )
    * (1.0 if (i >> 5) == 0 else -1.0)
    for i in range(64)
)


def _select_fp6_value(b: IRBuilder, code_i32: Value) -> Value:
    """Static-codebook select chain for one fp6 6-bit value.

    Same shape as :func:`_select_fp4_value` but 64 entries deep. The
    ternary chain is unrolled at Python time; at LLVM the AMDGPU
    backend folds it into ``v_cndmask`` chains. For perf-critical
    paths a future optimisation is to load the codebook from
    constant memory (``addrspace(4)``) via P17 and replace the
    chain with one ``s_load`` + ``v_lshrrev`` index — keeping the
    helper signature stable.
    """
    out = b.const_f32(0.0)
    for k, fv in enumerate(_FP6_CODEBOOK):
        is_k = b.cmp_eq(code_i32, b.const_i32(k))
        out = b.select(is_k, b.const_f32(float(fv)), out)
    return out


def unpack_fp6_bytes_to_quad_f32(
    b: IRBuilder, packed_lo: Value, packed_hi: Value
) -> Tuple[Value, Value, Value, Value]:
    """3-byte (= 4 × fp6) load -> four f32 values.

    OCP MX fp6 packs 4 values into 24 bits. The caller passes two
    bytes (``packed_lo`` = bits 0-15, ``packed_hi`` = bits 16-23)
    so the helper can treat them as ``i32`` material with no
    cross-byte shuffling at the call site.

    Returns ``(v0, v1, v2, v3)`` as f32 scalars.
    """
    if packed_lo.type.name != "i8" or packed_hi.type.name != "i8":
        raise ValueError(
            "unpack_fp6_bytes_to_quad_f32 expects i8 packed bytes; got "
            f"({packed_lo.type.name}, {packed_hi.type.name})"
        )
    lo_i32 = b.zext(packed_lo, I32)
    hi_i32 = b.zext(packed_hi, I32)
    word = b.lor(lo_i32, b.shl(hi_i32, b.const_i32(8)))
    mask = b.const_i32(0x3F)
    v0 = _select_fp6_value(b, b.land(word, mask))
    v1 = _select_fp6_value(b, b.land(b.lshr(word, b.const_i32(6)), mask))
    v2 = _select_fp6_value(b, b.land(b.lshr(word, b.const_i32(12)), mask))
    v3 = _select_fp6_value(b, b.land(b.lshr(word, b.const_i32(18)), mask))
    return v0, v1, v2, v3
