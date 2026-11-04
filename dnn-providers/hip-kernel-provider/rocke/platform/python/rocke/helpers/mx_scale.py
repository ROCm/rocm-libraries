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
* :func:`fma_with_mx_scale_pair`: ``value * 2^(ea + eb) + addend`` for an
 A and a B scale byte, rounded once, without forming the product of the two
 scales (which can overflow or flush to zero in f32). It takes the
 :class:`MxScaleSplit` that :func:`exact_mx_scale_split` derives from the
 :class:`MxElementFormat` of the A and B elements.
* :func:`apply_mx_scale`: scalar ``fmul`` with a dequantised value.
* :func:`load_and_decode_mx_scale_byte`: convenience wrapper that
 reads one byte from a scale buffer and decodes it.

The decoded scale is f32; downstream callers can ``cast_f32_to(scale,
f16/bf16)`` if the consumer wants the scale in a narrower dtype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.ir import I8, I32, IRBuilder, PtrType, Value, VectorType


__all__ = [
    "BF8_E5M2_FNUZ",
    "BF8_E5M2_OCP",
    "FP8_E4M3_FNUZ",
    "FP8_E4M3_OCP",
    "MxElementFormat",
    "MxScaleSplit",
    "apply_mx_scale",
    "decode_mx_scale_e8m0",
    "exact_mx_scale_split",
    "fma_with_mx_scale_pair",
    "load_and_decode_mx_scale_byte",
    "load_and_decode_mx_scales_wave",
]


_E8M0_BIAS = 127
_E8M0_NAN_CODE = 255
# ea + eb for two finite E8M0 codes (0..254) spans -254..254.
_SMALLEST_EXPONENT_SUM = -2 * _E8M0_BIAS
_LARGEST_EXPONENT_SUM = 2 * _E8M0_BIAS
# The f32 powers of two: 2^-149 (the smallest subnormal) .. 2^127.
_F32_SMALLEST_EXPONENT = -149
_F32_LARGEST_EXPONENT = 127
_F32_SIGNIFICAND_BITS = 24


@dataclass(frozen=True)
class MxElementFormat:
    """The values an element format can hold, as far as an MX group's dot
    product depends on them.

    ``smallest_exponent``: the smallest positive value is
    ``2^smallest_exponent`` (the smallest subnormal), and every value is a
    multiple of it. ``largest_value``: the largest finite magnitude.
    ``name`` appears in error messages.
    """

    name: str
    smallest_exponent: int
    largest_value: float


# The fp8 and bf8 encodings matrix cores read. OCP: E4M3 (bias 7, largest 448)
# and E5M2 (bias 15, largest 57344). FNUZ: the same bit layouts with the bias
# one higher and no infinities or negative zero (largest 240 and 57344).
FP8_E4M3_OCP = MxElementFormat("OCP E4M3", -9, 448.0)
BF8_E5M2_OCP = MxElementFormat("OCP E5M2", -16, 57344.0)
FP8_E4M3_FNUZ = MxElementFormat("E4M3 FNUZ", -10, 240.0)
BF8_E5M2_FNUZ = MxElementFormat("E5M2 FNUZ", -17, 57344.0)


@dataclass(frozen=True)
class MxScaleSplit:
    """How :func:`fma_with_mx_scale_pair` splits ``2^(ea + eb)`` into two f32
    factors; derive it with :func:`exact_mx_scale_split`.

    ``ea + eb`` is split as ``pre + fused`` with ``fused`` clamped to
    ``fused_min..fused_max`` and ``pre`` the rest, then ``pre`` is saturated
    at :attr:`pre_max`. As ``exponent_cap`` is at least ``fused_max``, that
    is the split of ``ea + eb`` saturated at ``exponent_cap``.
    """

    exponent_cap: int
    fused_min: int
    fused_max: int

    @property
    def pre_max(self) -> int:
        """The largest ``pre``: ``exponent_cap - fused_max``. ``ea + eb``
        above ``exponent_cap`` has ``fused = fused_max`` and ``pre`` beyond
        this."""
        return self.exponent_cap - self.fused_max


def _f32_top_exponent(magnitude: float) -> int:
    """``e`` with ``2^e <= y < 2^(e + 1)``, for ``y`` the smallest f32 that is
    at least ``magnitude`` (a positive float that holds its exact value)."""
    fraction, exponent = math.frexp(magnitude)  # magnitude = fraction * 2^exponent
    top = exponent - 1
    # Above the largest 24-bit significand at 2^top, an f32 rounds up to 2^(top + 1).
    if magnitude > (2.0 - 2.0 ** (1 - _F32_SIGNIFICAND_BITS)) * 2.0**top:
        top += 1
    return top


def exact_mx_scale_split(
    a_format: MxElementFormat, b_format: MxElementFormat, group_size: int
) -> MxScaleSplit:
    """The split of ``2^(ea + eb)`` that makes :func:`fma_with_mx_scale_pair`
    exact for the dot product of ``group_size`` A and B elements.

    Exact means ``addend + dot * 2^(ea + eb)`` rounded once to f32, bit for
    bit, with IEEE special values and signed zeros, for every ``dot`` such a
    group can produce: +-0, +-Inf, NaN and every finite f32 the dot product
    of ``group_size`` products of an ``a_format`` and a ``b_format`` value
    can round to. The dot product is bounded, and the bounds give the split:

    * Every product is a multiple of ``2^low``, ``low`` the sum of the two
      formats' smallest exponents, and so is the dot product: an f32 rounding
      of such a sum is a multiple of ``2^low`` too.
    * ``|dot| < 2^(top + 1)``, ``2^top`` the top bit of the smallest f32 at
      least ``group_size * a_format.largest_value * b_format.largest_value``.

    So ``dot * 2^pre`` is an f32 multiply with an exact result for every
    ``pre`` in ``-149 - low .. 127 - top``. The fused factor takes every f32
    power of two, ``fused_min = -149`` and ``fused_max = 127``, and
    ``exponent_cap = 127 + (127 - top)`` is the largest ``ea + eb`` whose
    ``pre`` is still in range: ``pre_max = 127 - top``, and the cap is at
    least ``fused_max`` as ``top`` is at most 127. Then
    ``fma(dot * 2^pre, 2^fused, addend)``
    rounds once: the fma's product is exact and so is its first factor.
    Above the cap a nonzero ``dot`` times ``2^exponent_cap`` has magnitude at
    least ``2^(low + exponent_cap)``, and the exact product is larger still.
    If that is at least ``2^129``, which no finite addend can pull back below
    the f32 overflow threshold, the saturated update gives what the exact one
    gives: the infinity of the product's sign for a finite addend, the addend
    for an infinite one, and ``addend + 0`` for a zero ``dot``.

    Raises ``ValueError`` when the formats' bounds admit no such split: a
    dot product beyond the f32 range, products so small that
    ``dot * 2^pre`` cannot stay exact at the smallest scales
    (``ea + eb = -254`` needs ``pre <= -254 + 149``), or a dot product range
    so wide that saturating ``ea + eb`` would change a result.
    """
    low = a_format.smallest_exponent + b_format.smallest_exponent
    top = _f32_top_exponent(
        group_size * a_format.largest_value * b_format.largest_value
    )
    pair = f"{group_size} products of {a_format.name} and {b_format.name} values"
    if top > _F32_LARGEST_EXPONENT:
        raise ValueError(
            f"exact_mx_scale_split: the dot product of {pair} can reach 2^{top}, "
            f"beyond the f32 range"
        )
    pre_min = _F32_SMALLEST_EXPONENT - low
    if _SMALLEST_EXPONENT_SUM - _F32_SMALLEST_EXPONENT < pre_min:
        raise ValueError(
            f"exact_mx_scale_split: the dot product of {pair} can be as small as "
            f"2^{low}; at the smallest scales (ea + eb = {_SMALLEST_EXPONENT_SUM}) "
            f"it needs a factor of 2^{_SMALLEST_EXPONENT_SUM - _F32_SMALLEST_EXPONENT} "
            f"before the fma, which would round it"
        )
    exponent_cap = 2 * _F32_LARGEST_EXPONENT - top
    if (
        exponent_cap < _LARGEST_EXPONENT_SUM
        and low + exponent_cap < _F32_LARGEST_EXPONENT + 2
    ):
        raise ValueError(
            f"exact_mx_scale_split: the dot product of {pair} spans 2^{low} to "
            f"2^{top + 1}, too wide to saturate ea + eb at {exponent_cap} without "
            f"changing a result"
        )
    return MxScaleSplit(
        exponent_cap=exponent_cap,
        fused_min=_F32_SMALLEST_EXPONENT,
        fused_max=_F32_LARGEST_EXPONENT,
    )


def _read_e8m0_code(caller: str, b: IRBuilder, e8m0: Value):
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


def fma_with_mx_scale_pair(
    b: IRBuilder,
    value: Value,
    a_e8m0: Value,
    b_e8m0: Value,
    addend: Value,
    *,
    split: MxScaleSplit,
) -> Value:
    """``addend + value * 2^(ea + eb)`` for two E8M0 scale bytes, rounded once.

    ``ea`` and ``eb`` are the unbiased exponents of the scale codes ``a_e8m0``
    and ``b_e8m0`` (i8 each), so the product of the two decoded scales is
    ``2^(ea + eb)``. ``value`` and ``addend`` are matching ``<n x f32>``
    vectors; the result has the same type. If either code is the NaN code
    255 the result is NaN.

    ``value`` is a group's dot product, and ``split`` is
    :func:`exact_mx_scale_split` of the formats of its A and B elements. For
    every ``value`` such a group can produce, the result is ``addend + value
    * 2^(ea + eb)`` rounded once to f32, bit for bit, with IEEE special values
    and signed zeros; :func:`exact_mx_scale_split` says why.

    Why not multiply the two decoded scales: ``ea + eb`` spans -254..254 and
    an f32 holds only 2^-149..2^127, so the product of two decoded scales
    overflows to Inf for large codes (and a zero ``value`` then gives
    ``0 * Inf = NaN``) and flushes to zero for small ones, even when the
    true result is an ordinary f32.

    Instead the exponents are added as integers and split as ``pre + fused``
    with ``fused`` clamped to ``split.fused_min..split.fused_max`` and
    ``pre`` the rest, saturated at ``split.pre_max``: the split of ``ea +
    eb`` saturated at ``split.exponent_cap``. The result is ``fma(value *
    2^pre, 2^fused, addend)``: ``value * 2^pre`` is exact for the values the
    split was derived for, and the fma rounds once.
    """
    if not isinstance(split, MxScaleSplit):
        raise ValueError(
            f"fma_with_mx_scale_pair expects an MxScaleSplit, got {split!r}"
        )
    if not isinstance(value.type, VectorType) or value.type.elem.name != "f32":
        raise ValueError(
            f"fma_with_mx_scale_pair expects an f32 vector value, got {value.type.name}"
        )
    if addend.type != value.type:
        raise ValueError(
            f"fma_with_mx_scale_pair expects the addend type to match the value "
            f"type {value.type.name}, got {addend.type.name}"
        )
    a_code, a_is_nan = _read_e8m0_code("fma_with_mx_scale_pair", b, a_e8m0)
    b_code, b_is_nan = _read_e8m0_code("fma_with_mx_scale_pair", b, b_e8m0)
    exponent = b.sub(b.add(a_code, b_code), b.const_i32(2 * _E8M0_BIAS))
    fused = b.smin(
        b.smax(exponent, b.const_i32(split.fused_min)), b.const_i32(split.fused_max)
    )
    pre = b.smin(b.sub(exponent, fused), b.const_i32(split.pre_max))
    pre_scale = b.exp2(b.sitofp_f32(pre))
    fused_scale = b.select(
        b.lor(a_is_nan, b_is_nan),
        b.const_f32(math.nan),
        b.exp2(b.sitofp_f32(fused)),
    )
    lanes = value.type.count
    pre_scaled = b.vector_mul(value, b.vector_splat(pre_scale, lanes))
    return b.vector_fma(pre_scaled, b.vector_splat(fused_scale, lanes), addend)


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
