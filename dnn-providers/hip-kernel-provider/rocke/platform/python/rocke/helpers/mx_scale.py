# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MX (microscaling) shared-exponent helpers.

The OCP "MX" formats (mxFP8, mxFP6, mxFP4) trade a per-element scale
for a *shared-exponent* (E8M0) scale that's broadcast across a block
of 32 mantissa elements. The dequantisation pipeline is uniform across
the three mantissa types and across the CK Tile ``42_mx_gemm`` /
``38_block_scale_gemm`` mx variants::

 1. Load the 8-bit E8M0 scale (one byte per 32-element mantissa
 block). The byte is an unsigned biased exponent: no sign, no mantissa.
 2. Decode the scale into an f32 multiplier, as the OCP MX spec
 defines it: codes 0..254 are exactly ``2^(code - 127)`` (code 0 is
 ``2^-127``, an f32 subnormal) and code 255 is NaN.
 3. Apply the scale to the dequantised mantissa value via a single
 ``fmul`` before it enters the accumulator.

The actual MX-aware MFMA intrinsics
(``llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4`` family on gfx950+) take
the scale operand directly and apply step 3 in-instruction; this
module is for the **non-MFMA** code path (e.g., the f32 reference for
a parity test, or the per-warp f32 prologue when emitting MX-mantissa
loads through the existing FP8 MFMA pipeline).

What v1 ships:

* :func:`decode_mx_scale_e8m0`: ``i8 e8m0 -> f32 multiplier`` via
 ``exp2(code - 127)``, NaN for code 255.
* :func:`apply_mx_scale`: scalar ``fmul`` with a dequantised value.
* :func:`load_and_decode_mx_scale_byte`: convenience wrapper that
 reads one byte from a scale buffer and decodes it.

The decoded scale is f32; downstream callers can ``cast_f32_to(scale,
f16/bf16)`` if the consumer wants the scale in a narrower dtype.
"""

from __future__ import annotations

import math

from ..core.ir import I8, I32, IRBuilder, PtrType, Value


__all__ = [
    "apply_mx_scale",
    "decode_mx_scale_e8m0",
    "load_and_decode_mx_scale_byte",
    "load_and_decode_mx_scales_wave",
]


def decode_mx_scale_e8m0(b: IRBuilder, e8m0: Value) -> Value:
    """Decode an 8-bit E8M0 scale code into its f32 multiplier, per the
    OCP MX spec.

    The byte is an unsigned biased exponent (bias 127, no sign, no
    mantissa):

    * ``0 <= code <= 254`` -> exactly ``2^(code - 127)``. Code 0 is
      ``2^-127``, an f32 subnormal; it is not zero.
    * ``code == 255`` -> NaN.

    Lowering: ``zext`` the byte to i32 (the byte is read unsigned, so
    codes 128..255 keep their value; the result is 0..255, so the signed
    ``sitofp`` converts it exactly), subtract 127
    (exact in f32), ``exp2``, then a ``select`` puts a quiet NaN on
    code 255. The ``exp2`` lowering returns exact powers of two for the
    integer exponents -127..127, including the subnormal for code 0; the
    decoder test checks all 256 codes on the GPU.
    """
    if e8m0.type.name != "i8":
        raise ValueError(f"decode_mx_scale_e8m0 expects i8 input, got {e8m0.type.name}")
    code = b.zext(e8m0, I32)
    is_nan_code = b.cmp_eq(code, b.const_i32(255))
    exponent = b.fsub(b.sitofp_f32(code), b.const_f32(127.0))
    scale = b.exp2(exponent)
    return b.select(is_nan_code, b.const_f32(math.nan), scale)


def apply_mx_scale(b: IRBuilder, value_f32: Value, scale_f32: Value) -> Value:
    """``value_f32 * scale_f32`` -- one fmul. The single-line wrapper
    exists so call sites stay clearly tagged as "MX scale apply"
    rather than as a generic fmul.

    Kernel authors typically pre-decode the scale once per 32-element
    block (via :func:`decode_mx_scale_e8m0`) and apply it to every
    dequantised element in that block via this helper.
    """
    if value_f32.type.name != "f32" or scale_f32.type.name != "f32":
        raise ValueError(
            f"apply_mx_scale expects f32 inputs, got "
            f"value={value_f32.type.name}, scale={scale_f32.type.name}"
        )
    return b.fmul(value_f32, scale_f32)


def load_and_decode_mx_scale_byte(
    b: IRBuilder,
    scale_ptr: Value,
    scale_idx: Value,
) -> Value:
    """Convenience: ``decode_mx_scale_e8m0(load_i8(scale_ptr[scale_idx]))``.

    ``scale_ptr`` must be a global ``ptr<i8>``. The helper does one
    typed scalar load + the decode chain, returning the f32
    multiplier ready for :func:`apply_mx_scale`.
    """
    if not isinstance(scale_ptr.type, PtrType):
        raise ValueError(
            f"load_and_decode_mx_scale_byte expects a pointer, got {scale_ptr.type!r}"
        )
    if scale_ptr.type.pointee.name != "i8":
        raise ValueError(
            f"load_and_decode_mx_scale_byte expects ptr<i8>, "
            f"got ptr<{scale_ptr.type.pointee.name}>"
        )
    e8m0 = b.global_load_i8(scale_ptr, scale_idx)
    return decode_mx_scale_e8m0(b, e8m0)


def load_and_decode_mx_scales_wave(
    b: IRBuilder,
    *,
    scale_ptr: Value,
    base_idx: Value,
    lane: Value,
    src_lane_fn,
    is_row_lead_fn,
) -> Value:
    """Wave-broadcast MX scale load + decode (P51).

    In MX MFMA, A's scale only varies over ``m_in_atom = lane % 16``
    (and B's scale over ``n_in_atom``). With the standard
    ``BlockGemmMxARegBSmemCRegV1`` distribution, lanes 0/16/32/48
    (different ``k_blk``, same ``m_in_atom``) all redundantly load
    the same byte — 4× redundancy per wave per ``kg``.

    This helper has only the row-lead lanes (``is_row_lead_fn`` →
    True for lanes 0..15 on the A side, lanes 0..15 / 0..15
    + ``[0, 16, 32, 48]`` for B depending on the layout) issue the
    scale byte load + decode; the other lanes pull the f32 scale via
    ``ds_bpermute``.

    ``src_lane_fn(b, lane) -> i32`` returns the source lane each
    consumer should read from (typically ``lane & 0xF`` for A's
    16-row-grouped scale broadcast). ``is_row_lead_fn(b, lane) -> i1``
    selects which lanes do the actual load.

    Returns a per-lane f32 scale value — the same scalar every member
    of the row-group sees.
    """
    is_lead = is_row_lead_fn(b, lane)
    # Lead lanes load + decode; non-lead lanes get a placeholder 0
    # (their value is overwritten by the broadcast below).
    lead_idx = b.add(base_idx, lane)
    leader_scale = load_and_decode_mx_scale_byte(b, scale_ptr, lead_idx)
    raw = b.select(is_lead, leader_scale, b.const_f32(0.0))

    # Bitcast to i32 for the cross-lane broadcast (ds_bpermute is i32-
    # native), broadcast the lead's value, bitcast back to f32.
    raw_i32 = b.bitcast(raw, I32)
    src_lane = src_lane_fn(b, lane)
    addr = b.mul(src_lane, b.const_i32(4))  # i32 lane index in bits[7:2]
    bcast_i32 = b.ds_bpermute(addr, raw_i32)
    from ..core.ir import F32

    return b.bitcast(bcast_i32, F32)
