# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MXFP8 (E4M3 + one E8M0 scale per 32 values) activation quantizer.

Inline emitters for use inside a larger :class:`~rocke.core.ir.IRBuilder`
kernel, plus a numpy reference. The rule is the decode-GEMV activation rule of
the MXFP4-weight x MXFP8-activation scaled-MFMA path::

    amax  = max |x| over the 32 values of a block
    ratio = fl(amax * fl(1/448))                      # fl(1/448) = 0x3B124925
    E     = expbits(ratio) + (mantissa(ratio) != 0), clamped to [1, 254]
    q     = e4m3fn_rne(x / 2^(E - 127))               # never saturates for finite x

``E`` is the E8M0 scale byte. ``q`` is produced by the gfx950 scaled convert
``v_cvt_scalef32_pk_fp8_f32`` with the f32 scale ``2^(E - 127)``, two values per
instruction, so a dword of four E4M3 bytes takes two converts.

Emitters (``b`` is the IRBuilder; every value is a per-lane f32 / i32):

* :func:`e8m0_fp8` ``(b, amax) -> (E, scale)``.
* :func:`pack_fp8x4` ``(b, v0, v1, v2, v3, scale) -> i32`` (``v0`` in byte 0).
* :func:`block_amax` ``(b, values, lanes) -> f32``: in-lane max of ``|v|`` then a
  butterfly over ``lanes`` aligned consecutive lanes.
* :func:`quantize_mxfp8` ``(b, values, lanes) -> (words, E)``: the whole rule
  for one 32-value block spread over ``lanes`` lanes x ``len(values)`` values.

Knobs (performance levers; every value is bitwise-identical in output). Every
op this module emits is mirrored by both engines (``tile.inline_asm`` and the
DPP / ``ds_swizzle`` shuffles):

* ``cvt="asm"`` (default) emits the converts through ``tile.inline_asm``. A
  callable ``cvt(b, old, lo, hi_v, scale, *, hi) -> i32`` replaces the convert
  (``old`` is ``None`` for the low half); the Python-only intrinsic variant is
  such a callable and lives in the library family
  (``kernels/gfx950/mx_quant.py``, ``cvt="intrinsic"``), not here.
* ``reduce`` picks the in-row (``lanes <= 16``) butterfly steps:
  ``"dpp"`` (default) one register-only DPP move per step (masks 1, 2, 7, 15:
  ``quad_perm``, ``row_half_mirror``, ``row_mirror``); ``"swizzle"``
  ``ds_swizzle`` (LDS unit) for every step; ``"dpp_rowshift"`` the reference
  FlyDSL kernels' ``xshfl`` (masks 1, 2, 4, 8; 4 and 8 as a ``row_shr`` /
  ``row_shl`` pair of bank-masked ``v_mov_b32_dpp``, in inline asm).
  ``max`` is exact and order-free, so the block maximum does not depend on
  the reduction order.
* ``xor16`` picks the 16-lane xor step of a 32-lane block (``lanes == 32``):
  ``"swizzle"`` (default) ``ds_swizzle``; ``"permlane"`` the VALU
  ``v_permlane16_swap_b32`` (inline asm), as the reference kernels' ``xred``.
  The DPP and permlane inline asm carry an ``s_nop 1`` ahead of the shuffle:
  a VALU write followed by a DPP or permlane-swap read of that VGPR needs two
  wait states on gfx950, and the compiler cannot see into inline asm.
* ``nan="propagate"`` (default) takes the maximum with ``v_maximum3_f32``
  (inline asm, ``|x|`` source modifiers), so a NaN anywhere in a block makes
  ``amax`` NaN and ``E = 254``, as the NaN-propagating maximum of the
  reference FlyDSL kernels does. ``nan="ignore"`` uses ``llvm.maxnum``
  (compiler-visible ``v_max3_f32``), which drops NaN operands; both are
  bitwise identical on NaN-free blocks.

The numpy reference (:func:`ref_quantize_mxfp8`, :func:`ref_e8m0_fp8`,
:func:`ref_fp8e4m3_rne`, :func:`ref_dequant_mxfp8`) is defined for finite inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..core.ir import F32, I32, IRBuilder, Value

#: fl(1/448): the f32 nearest to 1/448 (slightly above it).
RATIO_BITS = 0x3B124925
#: E8M0 exponent clamp of the rule.
E8M0_MIN = 1
E8M0_MAX = 254
#: E4M3 (fn) largest finite magnitude.
FP8_MAX = 448.0

CVT_MODES = ("asm",)
REDUCE_MODES = ("dpp", "swizzle", "dpp_rowshift")
XOR16_MODES = ("swizzle", "permlane")
NAN_MODES = ("propagate", "ignore")

# Butterfly masks covering ``lanes`` aligned consecutive lanes with the
# gfx9 DPP xor table (1, 2, 3, 7, 8, 15 inside a 16-lane row). After the
# 1 and 2 steps every quad holds its max; 7 (half-row mirror) joins the two
# quads of an 8-lane group and 15 (row mirror) the two halves of a row.
_DPP_MASKS = {1: (), 2: (1,), 4: (1, 2), 8: (1, 2, 7), 16: (1, 2, 7, 15)}
_SWIZZLE_MASKS = {1: (), 2: (1,), 4: (1, 2), 8: (1, 2, 4), 16: (1, 2, 4, 8)}
_MASKS = {"dpp": _DPP_MASKS, "swizzle": _SWIZZLE_MASKS, "dpp_rowshift": _SWIZZLE_MASKS}
# ``xshfl`` of the reference kernels for xor 4 / 8: ``row_shr`` into the upper
# bank of each pair, then ``row_shl`` into the lower one.
_ROWSHIFT_ASM = {
    4: ("row_shr:4 row_mask:0xf bank_mask:0xa", "row_shl:4 row_mask:0xf bank_mask:0x5"),
    8: ("row_shr:8 row_mask:0xf bank_mask:0xc", "row_shl:8 row_mask:0xf bank_mask:0x3"),
}


# ----------------------------------------------------------------- device


def e8m0_fp8(b: IRBuilder, amax: Value) -> tuple[Value, Value]:
    """``(E, scale)`` of a block with absolute maximum ``amax`` (f32).

    ``E`` is the i32 E8M0 byte in ``[1, 254]``; ``scale`` is the f32
    ``2^(E - 127)`` the scaled convert takes.
    """
    if amax.type != F32:
        raise ValueError(f"e8m0_fp8 expects f32 amax, got {amax.type.name}")
    ratio = b.fmul(amax, b.bitcast(b.const_i32(RATIO_BITS), F32))
    bits = b.bitcast(ratio, I32)
    expo = b.land(b.lshr(bits, b.const_i32(23)), b.const_i32(0xFF))
    mant_nz = b.cmp_ne(b.land(bits, b.const_i32(0x7FFFFF)), b.const_i32(0))
    expo = b.add(expo, b.select(mant_nz, b.const_i32(1), b.const_i32(0)))
    expo = b.smin(b.smax(expo, b.const_i32(E8M0_MIN)), b.const_i32(E8M0_MAX))
    scale = b.bitcast(b.shl(expo, b.const_i32(23)), F32)
    return expo, scale


def _check_cvt(cvt) -> None:
    if not callable(cvt) and cvt not in CVT_MODES:
        raise ValueError(f"cvt must be one of {CVT_MODES} or a callable, got {cvt!r}")


def _cvt_pair(b, old, lo, hi_v, scale, *, hi: bool, cvt):
    if callable(cvt):
        return cvt(b, old, lo, hi_v, scale, hi=hi)
    if not hi:
        # The low half is written; the high half of the result register is
        # whatever it held and is overwritten by the high convert.
        return b.inline_asm(
            "v_cvt_scalef32_pk_fp8_f32 $0, $1, $2, $3",
            "=v,v,v,v",
            [lo, hi_v, scale],
            I32,
            sideeffect=False,
            result_name_hint="q8",
        )
    return b.inline_asm(
        "v_cvt_scalef32_pk_fp8_f32 $0, $2, $3, $4 op_sel:[0,0,0,1]",
        "=v,0,v,v,v",
        [old, lo, hi_v, scale],
        I32,
        sideeffect=False,
        result_name_hint="q8",
    )


def pack_fp8x4(
    b: IRBuilder,
    v0: Value,
    v1: Value,
    v2: Value,
    v3: Value,
    scale: Value,
    *,
    cvt="asm",
) -> Value:
    """Four f32 -> one i32 of E4M3 bytes ``v0..v3`` (``v0`` in the low byte)."""
    _check_cvt(cvt)
    for v in (v0, v1, v2, v3, scale):
        if v.type != F32:
            raise ValueError(f"pack_fp8x4 expects f32 operands, got {v.type.name}")
    lo = _cvt_pair(b, None, v0, v1, scale, hi=False, cvt=cvt)
    return _cvt_pair(b, lo, v2, v3, scale, hi=True, cvt=cvt)


def _maximum3_abs(b: IRBuilder, x: Value, y: Value, z: Value) -> Value:
    """NaN-propagating ``max(|x|, |y|, |z|)`` (gfx950 ``v_maximum3_f32``)."""
    return b.inline_asm(
        "v_maximum3_f32 $0, |$1|, |$2|, |$3|",
        "=v,v,v,v",
        [x, y, z],
        F32,
        sideeffect=False,
        result_name_hint="amax",
    )


def _max(b: IRBuilder, x: Value, y: Value, nan: str) -> Value:
    return _maximum3_abs(b, x, y, y) if nan == "propagate" else b.fmax(x, y)


def _rowshift_xor(b: IRBuilder, x: Value, mask: int) -> Value:
    """Lane ``l`` gets ``x`` of lane ``l ^ mask`` (4 or 8) by two bank-masked
    DPP row shifts; the early-clobber output keeps the source intact."""
    shr, shl = _ROWSHIFT_ASM[mask]
    return b.inline_asm(
        f"s_nop 1\n\tv_mov_b32_dpp $0, $1 {shr}\n\tv_mov_b32_dpp $0, $1 {shl}",
        "=&v,v",
        [x],
        F32,
        sideeffect=False,
        result_name_hint="xshfl",
    )


def _permlane16_max(b: IRBuilder, x: Value, nan: str) -> Value:
    """``max(x, x of lane l ^ 16)`` through ``v_permlane16_swap_b32``.

    With both registers holding ``x`` the swap leaves rows ``(0, 0, 2, 2)`` in
    one and ``(1, 1, 3, 3)`` in the other, so their maximum is the xor-16
    reduction in every lane."""
    lo, hi = b.inline_asm_multi(
        "s_nop 1\n\tv_permlane16_swap_b32 $0, $1",
        "=v,=v,0,1",
        [x, x],
        result_types=[F32, F32],
        sideeffect=False,
        result_name_hint="pl16",
    )
    return _max(b, lo, hi, nan)


def _lane_max(
    b: IRBuilder, x: Value, mask: int, reduce: str, nan: str, xor16: str = "swizzle"
) -> Value:
    if mask == 16 and xor16 == "permlane":
        return _permlane16_max(b, x, nan)
    if mask >= 16 or reduce == "swizzle":
        other = b.warp_shuffle_xor(x, mask)
    elif reduce == "dpp_rowshift" and mask in _ROWSHIFT_ASM:
        other = _rowshift_xor(b, x, mask)
    else:
        other = b.warp_shuffle_xor_dpp(x, mask)
    return _max(b, x, other, nan)


def block_amax(
    b: IRBuilder,
    values: Sequence[Value],
    lanes: int,
    *,
    reduce: str = "dpp",
    nan: str = "propagate",
    xor16: str = "swizzle",
) -> Value:
    """``max |v|`` over ``values`` of this lane and ``lanes - 1`` neighbours.

    The ``lanes`` lanes of a block are consecutive and aligned (lane ``l``
    belongs to block ``l // lanes``). Every lane of the wave must be active.
    """
    if reduce not in REDUCE_MODES:
        raise ValueError(f"reduce must be one of {REDUCE_MODES}, got {reduce!r}")
    if nan not in NAN_MODES:
        raise ValueError(f"nan must be one of {NAN_MODES}, got {nan!r}")
    if xor16 not in XOR16_MODES:
        raise ValueError(f"xor16 must be one of {XOR16_MODES}, got {xor16!r}")
    if lanes not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"lanes must be a power of two in 1..32, got {lanes}")
    if not values:
        raise ValueError("block_amax needs at least one value")
    if nan == "propagate":
        rest = list(values)
        amax = rest.pop(0)
        if not rest:
            amax = _maximum3_abs(b, amax, amax, amax)
        while rest:
            y = rest.pop(0)
            z = rest.pop(0) if rest else y
            amax = _maximum3_abs(b, amax, y, z)
    else:
        mags = [b.fabs(v) for v in values]
        amax = mags[0]
        rest = mags[1:]
        while len(rest) >= 2:
            amax = b.fmax3(amax, rest[0], rest[1])
            rest = rest[2:]
        if rest:
            amax = b.fmax(amax, rest[0])
    for mask in _MASKS[reduce][min(lanes, 16)]:
        amax = _lane_max(b, amax, mask, reduce, nan)
    if lanes == 32:
        amax = _lane_max(b, amax, 16, reduce, nan, xor16)
    return amax


def quantize_mxfp8(
    b: IRBuilder,
    values: Sequence[Value],
    lanes: int,
    *,
    cvt="asm",
    reduce: str = "dpp",
    nan: str = "propagate",
    xor16: str = "swizzle",
) -> tuple[list[Value], Value]:
    """Quantize one 32-value block held as ``len(values)`` f32 per lane over
    ``lanes`` aligned lanes (``len(values) * lanes == 32``, ``len(values)`` a
    multiple of 4).

    Returns ``(words, E)``: ``len(values) // 4`` i32 words of E4M3 bytes in
    ``values`` order and the block's i32 E8M0 byte (equal in every lane of the
    block).
    """
    n = len(values)
    if n % 4 or n * lanes != 32:
        raise ValueError(
            f"need len(values) a multiple of 4 and len(values) * lanes == 32, "
            f"got {n} values x {lanes} lanes"
        )
    _check_cvt(cvt)
    amax = block_amax(b, values, lanes, reduce=reduce, nan=nan, xor16=xor16)
    expo, scale = e8m0_fp8(b, amax)
    words = [pack_fp8x4(b, *values[i : i + 4], scale, cvt=cvt) for i in range(0, n, 4)]
    return words, expo


# ----------------------------------------------------------------- host reference


def ref_e8m0_fp8(amax) -> np.ndarray:
    """E8M0 byte (uint8) per block maximum (f32, finite, >= 0)."""
    amax = np.asarray(amax, dtype=np.float32)
    ratio_c = np.array([RATIO_BITS], dtype=np.uint32).view(np.float32)[0]
    with np.errstate(over="ignore", under="ignore"):
        ratio = (amax * ratio_c).astype(np.float32)
    bits = ratio.view(np.uint32).astype(np.int64)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0)
    return np.clip(e, E8M0_MIN, E8M0_MAX).astype(np.uint8)


def ref_fp8e4m3_rne(y) -> np.ndarray:
    """E4M3 (fn) codes of ``y`` (finite, ``|y| <= 464``), round to nearest even."""
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise ValueError("ref_fp8e4m3_rne needs finite input")
    mag = np.abs(y)
    if np.any(mag >= 464.0):
        raise ValueError("value out of the non-saturating E4M3 range")
    sign = np.signbit(y).astype(np.uint8) << 7
    sub = mag < 2.0**-6
    code = np.zeros(y.shape, dtype=np.int64)
    # Subnormals: step 2^-9; a round-up to 8 steps is the smallest normal (code 8).
    code[sub] = np.rint(mag[sub] * 2.0**9).astype(np.int64)
    m = mag[~sub]
    e = np.floor(np.log2(m)).astype(np.int64)
    # log2 can be off by one at exact powers of two; correct it.
    e = np.where(m >= 2.0 ** (e + 1), e + 1, e)
    e = np.where(m < 2.0**e, e - 1, e)
    frac = np.rint((m / 2.0**e - 1.0) * 8.0).astype(np.int64)
    e = np.where(frac == 8, e + 1, e)
    frac = np.where(frac == 8, 0, frac)
    code[~sub] = ((e + 7) << 3) | frac
    if np.any(code > 0x7E):
        raise AssertionError("E4M3 overflow in reference")
    return (code.astype(np.uint8) | sign).astype(np.uint8)


def ref_quantize_mxfp8(x) -> tuple[np.ndarray, np.ndarray]:
    """``(codes [..., 32] uint8, E [...] uint8)`` for f32 blocks ``x[..., 32]``."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] != 32:
        raise ValueError("last axis must be the 32-value block")
    if not np.all(np.isfinite(x)):
        raise ValueError("ref_quantize_mxfp8 needs finite input")
    amax = np.max(np.abs(x), axis=-1)
    e = ref_e8m0_fp8(amax)
    y = x.astype(np.float64) * np.exp2(127.0 - e.astype(np.float64))[..., None]
    return ref_fp8e4m3_rne(y), e


def ref_fp8e4m3_value(codes) -> np.ndarray:
    """float64 value of E4M3 (fn) codes (0x7F / 0xFF decode to NaN)."""
    c = np.asarray(codes, dtype=np.uint8).astype(np.int64)
    sign = np.where(c & 0x80, -1.0, 1.0)
    e = (c >> 3) & 0xF
    f = c & 7
    val = np.where(e == 0, f * 2.0**-9, (1.0 + f / 8.0) * np.exp2(e - 7.0))
    val = np.where((c & 0x7F) == 0x7F, np.nan, val)
    return sign * val


def ref_dequant_mxfp8(codes, e) -> np.ndarray:
    """float64 values of quantized blocks (``codes [..., 32]``, ``E [...]``)."""
    scale = np.exp2(np.asarray(e, dtype=np.float64) - 127.0)
    return ref_fp8e4m3_value(codes) * scale[..., None]


__all__ = [
    "CVT_MODES",
    "E8M0_MAX",
    "E8M0_MIN",
    "FP8_MAX",
    "NAN_MODES",
    "RATIO_BITS",
    "REDUCE_MODES",
    "XOR16_MODES",
    "block_amax",
    "e8m0_fp8",
    "pack_fp8x4",
    "quantize_mxfp8",
    "ref_dequant_mxfp8",
    "ref_e8m0_fp8",
    "ref_fp8e4m3_rne",
    "ref_fp8e4m3_value",
    "ref_quantize_mxfp8",
]
