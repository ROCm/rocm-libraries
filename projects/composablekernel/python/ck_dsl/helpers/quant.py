# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Quantisation helpers (f32 -> {i8, fp8e4m3, bf8e5m2}).

CK Tile's quantisation kernels (SmoothQuant, RDQuant, MoE-Quant,
block-scaled GEMM epilogues) all share the same compute kernel
underneath:

    q = sat_round(x * inv_scale)

with three knobs:

* The output type (``i8`` for the SmoothQuant default; ``fp8e4m3`` for
  the FP8-output variant; ``bf8e5m2`` for the e5m2-output variant).
* The clamp range per output type (``[-127, 127]``, ``[-448, 448]``,
  ``[-57344, 57344]``).
* Whether the rounding is round-to-nearest-even (default) or stochastic
  (a v2 follow-on).

This module exposes the small set of helpers every quantised op needs:

* :class:`QDType` — the output dtype literal alias used in spec dataclasses.
* :data:`QUANT_MAX_ABS` — clamp bound per output type (matches CK Tile's
  ``ComputeDataType{127.0f}`` / ``448.0f`` / ``57344.0f`` literals).
* :func:`quant_ir_type` — map a ``QDType`` to the IR :class:`Type`.
* :func:`quantize_scalar_f32` — the one-element quant fast path
  (``cvt_f32_to_<qd>(clamp(x * inv_scale, -max, max))``).
* :func:`dequantize_scalar_to_f32` — the dual; auto-dispatches on the
  input value's IR type.

Authoring style mirrors :mod:`ck_dsl.helpers.io`: every helper takes
the dtype as a *string* alias so call sites read naturally
(``quantize_scalar_f32(b, v, inv_scale=inv, qdtype="i8")``), with
:func:`quant_ir_type` handling the canonicalisation.
"""

from __future__ import annotations

from typing import Literal

from ..core.ir import BF8E5M2, FP8E4M3, I8, I32, IRBuilder, Type, Value


__all__ = [
    "QDType",
    "QUANT_MAX_ABS",
    "dequantize_scalar_to_f32",
    "ir_to_qdtype",
    "quant_ir_type",
    "quant_max_abs",
    "quantize_scalar_f32",
]


QDType = Literal["i8", "fp8e4m3", "bf8e5m2"]


# Per-dtype clamp magnitude (the largest representable absolute value).
# CK Tile's ``SmoothquantPipeline`` uses 127.0f for i8 and the
# ``ck_tile::numeric<Q>::max()`` constants for fp8/bf8; we match
# those exactly so a port of any CK Tile reference yields bit-identical
# quantised tensors after rounding.
QUANT_MAX_ABS = {
    "i8": 127.0,
    "fp8e4m3": 448.0,
    "bf8e5m2": 57344.0,
}


# Common aliases the kernel-author surface accepts. ``"int8"`` and
# ``"fp8"`` / ``"bf8"`` show up in CK Tile reference code and in
# Triton-ported kernels, so we normalise them in one place.
_QDTYPE_ALIAS = {
    "i8": "i8",
    "int8": "i8",
    "fp8e4m3": "fp8e4m3",
    "fp8": "fp8e4m3",
    "fp8_e4m3": "fp8e4m3",
    "bf8e5m2": "bf8e5m2",
    "bf8": "bf8e5m2",
    "fp8_e5m2": "bf8e5m2",
}


_IR_TO_QDTYPE = {
    "i8": "i8",
    "fp8e4m3": "fp8e4m3",
    "bf8e5m2": "bf8e5m2",
}


def _canon(qdtype: str) -> QDType:
    if qdtype not in _QDTYPE_ALIAS:
        raise ValueError(
            f"unsupported quant dtype {qdtype!r}; expected one of "
            f"{sorted(_QDTYPE_ALIAS)}"
        )
    return _QDTYPE_ALIAS[qdtype]  # type: ignore[return-value]


def quant_ir_type(qdtype: str) -> Type:
    """Map a quant-dtype alias string to the canonical IR :class:`Type`.

    Accepts ``"i8"`` / ``"int8"`` / ``"fp8e4m3"`` / ``"fp8"`` /
    ``"bf8e5m2"`` / ``"bf8"``. Raises :class:`ValueError` for anything
    else; this is the single point of truth for the alias map so
    everywhere in the codebase agrees on the canonical names.
    """
    canon = _canon(qdtype)
    if canon == "i8":
        return I8
    if canon == "fp8e4m3":
        return FP8E4M3
    if canon == "bf8e5m2":
        return BF8E5M2
    raise ValueError(f"unreachable: canon={canon!r}")


def quant_max_abs(qdtype: str) -> float:
    """Saturating clamp magnitude (positive) for ``qdtype``.

    Returns the value used in
    ``quantize_scalar_f32(x, inv_scale, qdtype)`` as the upper / lower
    clamp bound: ``127`` for i8, ``448`` for fp8e4m3, ``57344`` for
    bf8e5m2. Matches CK Tile's ``numeric<DType>::max()`` literals.
    """
    canon = _canon(qdtype)
    return QUANT_MAX_ABS[canon]


def ir_to_qdtype(t: Type) -> QDType:
    """Inverse of :func:`quant_ir_type`. Rejects non-quant types."""
    if t.name in _IR_TO_QDTYPE:
        return _IR_TO_QDTYPE[t.name]  # type: ignore[return-value]
    raise ValueError(f"type {t.name!r} is not a quant dtype")


def quantize_scalar_f32(
    b: IRBuilder,
    x_f32: Value,
    *,
    inv_scale: Value,
    qdtype: str,
) -> Value:
    """Quantise one f32 scalar to ``qdtype``.

    Pipeline (single hardware path per output dtype):

    .. code-block:: text

        scaled  = x_f32 * inv_scale
        clamped = clamp_f32(scaled, -QUANT_MAX_ABS, +QUANT_MAX_ABS)
        result  = cvt_f32_to_<qdtype>(clamped)

    The clamp is redundant for fp8/bf8 (the AMDGPU hardware already
    saturates on conversion), but it's a cheap two-op ``v_med3_f32``
    that keeps the f32 path interpretable in IR dumps and matches the
    CK Tile reference's saturation semantics exactly for the i8 path
    (where ``v_cvt_pk_i16_i32`` does *not* saturate without the
    explicit clamp).

    ``inv_scale`` is in the *natural* direction (``inv_scale = 1 / scale``
    where ``scale = amax / quant_max``). Pre-computing the reciprocal
    outside the inner loop is the same trick CK Tile uses; it amortises
    one ``v_rcp_f32`` over the whole row.

    Returns a value whose IR type is the quant dtype (i8 / fp8e4m3 /
    bf8e5m2). Caller stores it via the standard ``global_store``.
    """
    if x_f32.type.name != "f32":
        raise ValueError(
            f"quantize_scalar_f32 expects f32 input, got {x_f32.type.name}"
        )
    if inv_scale.type.name != "f32":
        raise ValueError(
            f"quantize_scalar_f32 expects f32 inv_scale, got {inv_scale.type.name}"
        )
    canon = _canon(qdtype)
    qmax = quant_max_abs(canon)
    c_pos = b.const_f32(qmax)
    c_neg = b.const_f32(-qmax)
    scaled = b.fmul(x_f32, inv_scale)
    clamped = b.clamp_f32(scaled, c_neg, c_pos)
    if canon == "i8":
        return b.cvt_f32_to_i8_sat(clamped)
    if canon == "fp8e4m3":
        return b.cvt_f32_to_fp8(clamped)
    if canon == "bf8e5m2":
        return b.cvt_f32_to_bf8(clamped)
    raise ValueError(f"unreachable canon={canon!r}")


def dequantize_scalar_to_f32(
    b: IRBuilder,
    x_q: Value,
    *,
    scale: Value,
) -> Value:
    """Dequantise one i8 / fp8e4m3 / bf8e5m2 scalar to f32.

    Pipeline:

    .. code-block:: text

        as_f32 = cvt_<input>_to_f32(x_q)
        return  as_f32 * scale

    ``scale`` is the forward-direction scale used by the corresponding
    :func:`quantize_scalar_f32` (i.e. ``amax / quant_max``); pass
    ``inv_scale = 1 / scale`` to ``quantize_scalar_f32`` to keep the
    two operations symmetric.

    Auto-dispatches on the input value's IR type; raises for non-quant
    input dtypes.
    """
    if scale.type.name != "f32":
        raise ValueError(
            f"dequantize_scalar_to_f32 expects f32 scale, got {scale.type.name}"
        )
    ty = x_q.type.name
    if ty == "i8":
        # i8 -> i32 -> f32 via sext + sitofp. The two ops fold into
        # one ``v_cvt_f32_i32`` on AMDGPU when the i8 lives in the low
        # byte of a register (which it does after a byte load).
        as_f32 = b.sitofp_f32(b.sext(x_q, I32))
    elif ty == "fp8e4m3":
        as_f32 = b.cvt_fp8_to_f32(x_q)
    elif ty == "bf8e5m2":
        as_f32 = b.cvt_bf8_to_f32(x_q)
    else:
        raise ValueError(f"dequantize_scalar_to_f32 unsupported input type {ty!r}")
    return b.fmul(as_f32, scale)
