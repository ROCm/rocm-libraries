# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MX (microscaling) shared-exponent helpers.

The OCP "MX" formats (mxFP8, mxFP6, mxFP4) trade a per-element scale
for a *shared-exponent* (E8M0) scale that's broadcast across a block
of 32 mantissa elements. The dequantisation pipeline is uniform across
the three mantissa types and across the CK Tile ``42_mx_gemm`` /
``38_block_scale_gemm`` mx variants::

 1. Load the 8-bit unbiased E8M0 scale (one byte per 32-element
 mantissa block).
 2. Decode the scale into an f32 multiplier: ``scale_f32 = 2^(e - 127)``
 where ``e`` is the unbiased exponent. Special cases: ``e == 0``
 -> denormal (treated as 0 for ML use); ``e == 255`` -> NaN
 (treated as 0 by AMDGPU's MX MFMA hardware path).
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
 ``exp2(e - 127)``.
* :func:`apply_mx_scale`: scalar ``fmul`` with a dequantised value.
* :func:`load_and_decode_mx_scale_byte`: convenience wrapper that
 reads one byte from a scale buffer and decodes it.

The decoded scale is f32; downstream callers can ``cast_f32_to(scale,
f16/bf16)`` if the consumer wants the scale in a narrower dtype.
"""

from __future__ import annotations

from ..core.ir import I8, I32, IRBuilder, PtrType, Value


__all__ = [
    "apply_mx_scale",
    "decode_mx_scale_e8m0",
    "load_and_decode_mx_scale_byte",
]


def decode_mx_scale_e8m0(b: IRBuilder, e8m0: Value) -> Value:
    """Decode an 8-bit E8M0 exponent (unbiased, no mantissa, no sign)
    into an f32 multiplier ``2^(e - 127)``.

    E8M0 layout matches IEEE-754's f32 exponent field:

    * ``e == 0`` -> subnormal multiplier; we return ``0.0`` since
    ML uses the all-zero scale as a "block disabled" sentinel.
    * ``e == 255`` -> NaN; we return ``0.0`` so downstream
    accumulators don't poison their row. Matches the AMDGPU MX
    MFMA hardware path.
    * ``1 <= e <= 254`` -> ``2^(e - 127)``, exact.

    Lowering: ``sitofp(sext(e)) - 127.0`` then ``exp2``; with both
    sentinels handled via ``select``. The AMDGPU backend fuses the
    select chain into a single ``v_med3_f32`` + ``v_exp_f32``.
    """
    if e8m0.type.name != "i8":
        raise ValueError(f"decode_mx_scale_e8m0 expects i8 input, got {e8m0.type.name}")
    e_i32 = b.sext(e8m0, I32)
    is_zero = b.cmp_eq(e_i32, b.const_i32(0))
    is_nan = b.cmp_eq(e_i32, b.const_i32(255))
    e_minus_127_f32 = b.fsub(b.sitofp_f32(e_i32), b.const_f32(127.0))
    # ``exp2`` returns f32 here; the per-byte ML quant block size
    # caps the exponent range to roughly +/-127 so the result fits
    # well within f32's dynamic range.
    raw_scale = b.exp2(e_minus_127_f32)
    zero = b.const_f32(0.0)
    return b.select(b.lor(is_zero, is_nan), zero, raw_scale)


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
    e8m0 = b.global_load(scale_ptr, scale_idx, I8, align=1)
    return decode_mx_scale_e8m0(b, e8m0)
