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

Authoring style mirrors :mod:`rocke.helpers.io`: every helper takes
the dtype as a *string* alias so call sites read naturally
(``quantize_scalar_f32(b, v, inv_scale=inv, qdtype="i8")``), with
:func:`quant_ir_type` handling the canonicalisation.
"""

from __future__ import annotations

from typing import List, Literal

from ..core.ir import BF8E5M2, F32, FP8E4M3, I8, I32, I64, IRBuilder, Type, Value


__all__ = [
    "QDType",
    "QUANT_MAX_ABS",
    "dequantize_scalar_to_f32",
    "ir_to_qdtype",
    "pack_quant_chunk_local_f32",
    "quant_ir_type",
    "quant_max_abs",
    "quantize_scalar_f32",
    "store_packed_chunk_local",
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
    skip_clamp_on_pack: bool = False,
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
    # P55: when ``skip_clamp_on_pack`` is set and the target is fp8 /
    # bf8, drop the explicit ``v_med3_f32`` clamp because the AMDGPU
    # ``cvt.pk.fp8.f32`` / ``cvt.pk.bf8.f32`` instructions already
    # saturate the input on conversion. For ``i8`` we keep the clamp:
    # ``v_cvt_pk_i16_i32`` does not saturate so the explicit
    # ``smax`` / ``smin`` chain is required for correctness.
    if skip_clamp_on_pack and canon in ("fp8e4m3", "bf8e5m2"):
        if canon == "fp8e4m3":
            return b.cvt_f32_to_fp8(scaled)
        return b.cvt_f32_to_bf8(scaled)
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


def pack_quant_chunk_local_f32(
    b: IRBuilder,
    scaled_f32: List[Value],
    *,
    q_ty: Type,
    out_dtype: QDType,
) -> Value:
    """Quantise ``len(scaled_f32)`` f32 scalars and pack them into a
    ``<N x q_ty>`` vector, reproducing the *local* pass-2 packed-store
    emission shared by the SmoothQuant / add_rmsnorm2d_rdquant /
    MoE-SmoothQuant instances.

    ``scaled_f32`` is the chunk of values **already multiplied by the
    inverse quant scale** — i.e. ready for the dtype-specific saturating
    cast. The packing routes are:

    * For ``fp8e4m3`` / ``bf8e5m2`` with a 4-wide chunk we issue one
      packed ``v_cvt_pk_fp8_f32`` (resp. ``v_cvt_pk_bf8_f32``) via
      :func:`IRBuilder.cvt_pk_fp8_f32x4` / ``cvt_pk_bf8_f32x4``,
      saving 3 scalar ``v_cvt_*_f32`` instructions and the redundant
      ``clamp_f32`` (the hardware saturates on conversion).
    * For ``i8`` (no packed cvt today) and ``VEC == 2`` chunks (no
      2-wide packed cvt), we emit per-element scalar
      ``cvt_f32_to_i8_sat`` (with the explicit ``[-127, 127]``
      :func:`clamp_f32`) / ``cvt_f32_to_fp8`` / ``cvt_f32_to_bf8`` and
      pack via :func:`IRBuilder.vec_pack`.

    For ``len(scaled_f32) == 8`` two 4-wide cvt results are stitched
    with :func:`IRBuilder.vec_concat` into a ``<8 x q_ty>`` that lowers
    to a single 8-byte VMEM store.

    NOTE: this deliberately reproduces the local scalar-cvt + ``vec_pack``
    + bitcast-store op stream, which is a *different* op stream from the
    packed ``cvt_pk_i8_f32x4`` / ``global_store_vN`` path in
    :func:`rocke.helpers.io.pack_quant_chunk_f32`. The two are not
    interchangeable; this one preserves the instance-local IR.
    """
    n = len(scaled_f32)
    if n not in (2, 4, 8):
        raise ValueError(f"pack_quant_chunk_local_f32 expects n in {{2,4,8}}, got {n}")

    if out_dtype in ("fp8e4m3", "bf8e5m2") and n % 4 == 0:
        cvt = b.cvt_pk_fp8_f32x4 if out_dtype == "fp8e4m3" else b.cvt_pk_bf8_f32x4
        packed_chunks: List[Value] = []
        for off in range(0, n, 4):
            quad = b.vec_pack(scaled_f32[off : off + 4], F32)
            packed_chunks.append(cvt(quad))
        if len(packed_chunks) == 1:
            return packed_chunks[0]
        out = packed_chunks[0]
        for chunk in packed_chunks[1:]:
            out = b.vec_concat(out, chunk)
        return out

    # i8 path (or VEC=2 fp8/bf8): per-element saturating cast + vec_pack.
    qs: List[Value] = []
    for sf in scaled_f32:
        if out_dtype == "i8":
            qs.append(
                b.cvt_f32_to_i8_sat(
                    b.clamp_f32(sf, b.const_f32(-127.0), b.const_f32(127.0))
                )
            )
        elif out_dtype == "fp8e4m3":
            qs.append(b.cvt_f32_to_fp8(sf))
        elif out_dtype == "bf8e5m2":
            qs.append(b.cvt_f32_to_bf8(sf))
        else:
            raise ValueError(f"unsupported out_dtype {out_dtype!r}")
    return b.vec_pack(qs, q_ty)


def store_packed_chunk_local(
    b: IRBuilder,
    qy_ptr: Value,
    byte_off: Value,
    packed: Value,
    *,
    n: int,
) -> None:
    """Bitcast a ``<n x q_ty>`` (q_ty is an 8-bit dtype) to an integer
    and emit a single coalesced global store, reproducing the *local*
    bitcast-store emission shared by the SmoothQuant /
    add_rmsnorm2d_rdquant / MoE-SmoothQuant instances.

    * ``n == 4`` -> ``i32``, one ``buffer_store_dword``.
    * ``n == 8`` -> ``i64``, one ``buffer_store_dwordx2``.

    ``byte_off`` is the absolute byte offset within the QY buffer; the
    integer GEP stride for the chosen integer type is ``n`` bytes, so the
    helper divides ``byte_off`` by ``n`` via a logical right shift. Both
    ``byte_off`` and ``n`` are guaranteed multiples of ``n`` by spec
    validation (``N % (BS * VEC) == 0``).

    ``n == 2`` is not supported here: there is no ``I16`` IR type exposed
    today, and the AMDGPU backend already coalesces adjacent lanes'
    single-byte stores into a wave-wide dword in the scalar fall-back
    path.
    """
    if n == 4:
        as_int = b.bitcast(packed, I32)
        idx = b.lshr(byte_off, b.const_i32(2))
        b.global_store(qy_ptr, idx, as_int, align=4)
    elif n == 8:
        as_int = b.bitcast(packed, I64)
        idx = b.lshr(byte_off, b.const_i32(3))
        b.global_store(qy_ptr, idx, as_int, align=8)
    else:
        raise ValueError(f"store_packed_chunk_local supports n in {{4, 8}}, got {n}")
