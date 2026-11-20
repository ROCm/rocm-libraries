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

The MX-aware MFMA intrinsics (the ``llvm.amdgcn.mfma.scale.f32.*.f8f6f4``
family on gfx950) take the scale operand directly and apply step 3
in-instruction. This module serves the code paths that apply the scales
themselves (a parity reference, or scaling the dot product of an unscaled
fp8 MFMA), and bounds the codes the scaled MFMA may take directly
(:func:`largest_safe_mx_scale_code`).

What v1 ships:

* :func:`decode_mx_scale_e8m0`: ``i8 e8m0 -> f32 multiplier`` via
 ``exp2(code - 127)``, NaN for code 255.
* :func:`add_mx_scaled_dot`: ``acc + dot * 2^(ea + eb)`` element by element
 for the A and B scale bytes of each element, with one ``ldexp`` per element
 instead of a product of decoded scales (which can overflow or flush to zero
 in f32), and NaN for the NaN code.
* :func:`largest_safe_mx_scale_code`: the largest code an A and a B scale may
 both have before a sum of scaled products can overflow f32, for a matrix
 instruction that applies the scales itself.
* :func:`apply_mx_scale`: scalar ``fmul`` with a dequantised value.
* :func:`load_and_decode_mx_scale_byte`: convenience wrapper that
 reads one byte from a scale buffer and decodes it.

The decoded scale is f32; downstream callers can ``cast_f32_to(scale,
f16/bf16)`` if the consumer wants the scale in a narrower dtype.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..core.ir import F32, I32, IRBuilder, PtrType, Value, VectorType


__all__ = [
    "BF8_E5M2_FNUZ",
    "BF8_E5M2_OCP",
    "FP8_E4M3_FNUZ",
    "FP8_E4M3_OCP",
    "MxElementFormat",
    "add_mx_scaled_dot",
    "apply_mx_scale",
    "decode_mx_scale_e8m0",
    "largest_safe_mx_scale_code",
    "load_and_decode_mx_scale_byte",
    "load_and_decode_mx_scales_wave",
]


_E8M0_BIAS = 127
_E8M0_NAN_CODE = 255
_E8M0_LARGEST_FINITE_CODE = 254
# The f32 overflow threshold is 2^128 (less half an ulp); a sum bounded by
# 2^127 stays a full binade below it.
_F32_SAFE_SUM_EXPONENT = 127


@dataclass(frozen=True)
class MxElementFormat:
    """An element format's largest finite magnitude, which bounds the products
    of an MX group (:func:`largest_safe_mx_scale_code`). ``name`` says which
    encoding it is (for example ``"E4M3 FNUZ"``).
    """

    name: str
    largest_value: float


# The fp8 and bf8 encodings matrix cores read. OCP: E4M3 (bias 7, largest 448)
# and E5M2 (bias 15, largest 57344). FNUZ: the same bit layouts with the bias
# one higher and no infinities or negative zero (largest 240 and 57344).
FP8_E4M3_OCP = MxElementFormat("OCP E4M3", 448.0)
BF8_E5M2_OCP = MxElementFormat("OCP E5M2", 57344.0)
FP8_E4M3_FNUZ = MxElementFormat("E4M3 FNUZ", 240.0)
BF8_E5M2_FNUZ = MxElementFormat("E5M2 FNUZ", 57344.0)


def largest_safe_mx_scale_code(
    a_format: MxElementFormat, b_format: MxElementFormat, products: int
) -> int:
    """The largest E8M0 code an A and a B scale may both have so that a sum of
    ``products`` scaled A x B products stays a binade below the f32 overflow
    threshold.

    A matrix instruction that applies the scales itself (the gfx950 scaled
    MFMA) multiplies every product by ``2^(ea + eb)`` before it sums them. With
    both codes at most the returned ``c``, every partial sum is bounded by
    ``products * a_max * b_max * 2^(2c - 254) <= 2^127``. Above it a finite
    result can come out Inf or NaN: two scaled products of opposite sign that
    each overflow give NaN although their sum is an ordinary f32. A caller
    whose codes exceed the bound scales each group's dot product with
    :func:`add_mx_scaled_dot` instead, whose scaling never overflows on its
    own: its term overflows only when the scaled dot product itself is beyond
    the f32 range.
    """
    if products <= 0:
        raise ValueError(
            f"largest_safe_mx_scale_code: products must be positive, got {products}"
        )
    bound = products * a_format.largest_value * b_format.largest_value
    top = math.ceil(math.log2(bound))  # bound <= 2^top
    # 2c - 2 * bias + top <= 127
    return min(
        _E8M0_LARGEST_FINITE_CODE,
        (_F32_SAFE_SUM_EXPONENT + 2 * _E8M0_BIAS - top) // 2,
    )


def _read_e8m0_code(caller: str, b: IRBuilder, e8m0: Value) -> tuple[Value, Value]:
    """Read an E8M0 scale byte as ``(code, is_nan)``.

    The byte is an unsigned biased exponent, so it is zero-extended to an
    i32 code 0..255; ``is_nan`` is true for the NaN code 255.
    """
    if e8m0.type.name != "i8":
        raise ValueError(f"{caller} expects i8 input, got {e8m0.type.name}")
    code = b.zext(e8m0, I32)
    return code, b.cmp_eq(code, b.const_i32(_E8M0_NAN_CODE))


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
    code, is_nan_code = _read_e8m0_code("decode_mx_scale_e8m0", b, e8m0)
    exponent = b.fsub(b.sitofp_f32(code), b.const_f32(127.0))
    scale = b.exp2(exponent)
    return b.select(is_nan_code, b.const_f32(math.nan), scale)


def add_mx_scaled_dot(
    b: IRBuilder,
    acc: Value,
    dot: Value,
    a_codes: Sequence[Value],
    b_codes: Sequence[Value],
) -> Value:
    """``acc + dot * 2^(ea + eb)``, element by element.

    ``acc`` and ``dot`` are matching ``<n x f32>`` vectors; ``a_codes`` and
    ``b_codes`` hold the n A and B E8M0 scale bytes (i8) of the n elements, so
    ``ea`` and ``eb`` are ``code - 127``. Pass the same byte several times when
    elements share a scale. If either code of an element is the NaN code 255,
    that element's term is NaN.

    The term is ``ldexp(dot, ea + eb)``: one rounding, and correct for every
    pair of codes. The product of the two decoded scales spans
    ``2^-254..2^254`` and would overflow to Inf or flush to zero in f32 even
    when the term is an ordinary f32; ldexp never forms it. A term beyond the
    f32 range is the infinity or the zero of its sign. The sum with ``acc`` is
    an ordinary f32 add: once a term or the running sum overflows, the result
    is Inf, or NaN where infinities of both signs meet.
    """
    if not isinstance(dot.type, VectorType) or dot.type.elem.name != "f32":
        raise ValueError(
            f"add_mx_scaled_dot expects an f32 vector dot, got {dot.type.name}"
        )
    if acc.type != dot.type:
        raise ValueError(
            f"add_mx_scaled_dot expects the accumulator type to match the dot "
            f"type {dot.type.name}, got {acc.type.name}"
        )
    count = dot.type.count
    if len(a_codes) != count or len(b_codes) != count:
        raise ValueError(
            f"add_mx_scaled_dot expects {count} A and {count} B scale bytes, got "
            f"{len(a_codes)} and {len(b_codes)}"
        )
    read: dict[int, tuple[Value, Value]] = {}

    def code_of(e8m0: Value) -> tuple[Value, Value]:
        # Read each distinct byte once: (code, is_nan).
        if id(e8m0) not in read:
            read[id(e8m0)] = _read_e8m0_code("add_mx_scaled_dot", b, e8m0)
        return read[id(e8m0)]

    nan = b.const_f32(math.nan)
    two_biases = b.const_i32(2 * _E8M0_BIAS)
    terms = []
    for i in range(count):
        a_code, a_is_nan = code_of(a_codes[i])
        b_code, b_is_nan = code_of(b_codes[i])
        exponent = b.sub(b.add(a_code, b_code), two_biases)
        term = b.ldexp(b.vec_extract(dot, i), exponent)
        term = b.select(b.lor(a_is_nan, b_is_nan), nan, term)
        terms.append(b.fadd(b.vec_extract(acc, i), term))
    return b.vec_pack(terms, F32)


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
    return b.bitcast(bcast_i32, F32)
