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
* The clamp range per output type: ``[-127, 127]`` for i8, and for fp8 /
  bf8 the largest finite value of the encoding the target converts to
  (:func:`~rocke.helpers.fp8_encoding.fp8_element_format`: 240 / 57344 for
  gfx942's FNUZ, 448 / 57344 for OCP).
* Whether the rounding is round-to-nearest-even (default) or stochastic
  (a v2 follow-on).

This module exposes the small set of helpers every quantised op needs:

* :class:`QDType` — the output dtype literal alias used in spec dataclasses.
* :class:`LogicalQDType` — recognized types, including FP4/FP6 without scalar conversion.
* :func:`quant_ir_type` — map a recognized dtype string to the IR :class:`Type`.
* :func:`quant_max_abs` — that clamp bound for a dtype on a target.
* :func:`validate_quant_out_dtype` — whether a quantizing kernel can write a
  dtype on a target.
* :func:`quantize_scalar_f32` — the one-element quant fast path
  (``cvt_f32_to_<qd>(clamp(x * inv_scale, -max, max))``, NaN kept for
  fp8 / bf8).
* :func:`quantize_prescaled_f32` — the same clamp and conversion for a value
  the caller has already scaled.
* :func:`dequantize_scalar_to_f32` — the dual; auto-dispatches on the
  input value's IR type.

Dynamic scales: a kernel that derives its scale from the data does not bound
an infinite scaled value (``x * smscale`` in SmoothQuant and MoE-SmoothQuant,
``x * gamma`` in add + RMSNorm + RDQuant, the f32 Hidden in the fused fp8
MoE), whatever the clamp does:

* SmoothQuant, MoE-SmoothQuant and the fused fp8 MoE: the ``amax`` is +inf,
  so the stored scale is +inf and ``1 / scale`` is 0. The inf element becomes
  ``inf * 0`` = NaN (-127 for i8) and every finite element 0, over the row
  (the fused MoE: over the 128-column block, for every row of the M tile).
  Dequantizing with the stored +inf scale gives NaN for all of them.
* add + RMSNorm + RDQuant: ``1 / rms`` is 0, which multiplies every element
  of the row, so again the inf element becomes NaN (-127 for i8) and the
  finite ones 0, as long as ``1 / (eps_q / quant_max)`` is finite (an
  ``eps_q / quant_max`` so small that its reciprocal is +inf makes every
  element of the row NaN).

``platform/dsl_docs/numerics/`` describes each dtype's NaN and infinity
handling.

Authoring style mirrors :mod:`rocke.helpers.io`: every helper takes
the dtype as a *string* alias so call sites read naturally
(``quantize_scalar_f32(b, v, inv_scale=inv, qdtype="i8", arch=arch)``), with
:func:`quant_ir_type` handling the canonicalisation.
"""

from __future__ import annotations

from typing import List, Literal, TypeAlias, get_args

from ..core.arch import ArchTarget
from ..core.ir import (
    BF8E5M2,
    F32,
    FP8E4M3,
    I8,
    I32,
    I64,
    IRBuilder,
    Type,
    Value,
    dtype_to_ir_type,
)
from .fp8_encoding import fp8_element_format


__all__ = [
    "LogicalQDType",
    "QDType",
    "dequantize_scalar_to_f32",
    "ir_to_qdtype",
    "pack_quant_chunk_local_f32",
    "quant_ir_type",
    "quant_max_abs",
    "quantize_prescaled_f32",
    "quantize_scalar_f32",
    "store_packed_chunk_local",
    "validate_quant_out_dtype",
]


QDType: TypeAlias = Literal["i8", "fp8e4m3", "bf8e5m2"]
LogicalQDType: TypeAlias = Literal[QDType, "fp4e2m1", "fp6e2m3", "fp6e3m2"]


# The i8 clamp magnitude: the symmetric ``[-127, 127]`` of CK Tile's
# ``SmoothquantPipeline`` (``ComputeDataType{127.0f}``), not the i8 type's -128.
_I8_MAX_ABS = 127.0


# Common aliases the kernel-author surface accepts. ``"int8"`` and
# ``"fp8"`` / ``"bf8"`` show up in CK Tile reference code and in
# Triton-ported kernels, so we normalise them in one place.
_QDTYPE_ALIAS = {
    "i8": "i8",
    "int8": "i8",
    "fp8e4m3": "fp8e4m3",
    "fp8": "fp8e4m3",
    "fp8_e4m3": "fp8e4m3",
    "e4m3": "fp8e4m3",
    "bf8e5m2": "bf8e5m2",
    "bf8": "bf8e5m2",
    "fp8_e5m2": "bf8e5m2",
}


# Type recognition includes formats without scalar quantization support.
_QUANT_TYPE_ALIAS = _QDTYPE_ALIAS | {
    "fp4": "fp4e2m1",
    "fp4e2m1": "fp4e2m1",
    "fp6": "fp6e2m3",
    "fp6e2m3": "fp6e2m3",
    "bf6": "fp6e3m2",
    "fp6e3m2": "fp6e3m2",
}


_IR_TO_QDTYPE = {
    "i8": "i8",
    "fp8e4m3": "fp8e4m3",
    "bf8e5m2": "bf8e5m2",
    "fp4e2m1": "fp4e2m1",
    "fp6e2m3": "fp6e2m3",
    "fp6e3m2": "fp6e3m2",
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

    Accepts the existing i8/FP8/BF8 aliases and the FP4/FP6 aliases from
    the common dtype resolver. Type recognition does not enable scalar
    quantization or dequantization for the low-bit formats.
    """
    if qdtype not in _QUANT_TYPE_ALIAS:
        raise ValueError(
            f"unsupported quant dtype {qdtype!r}; expected one of "
            f"{sorted(_QUANT_TYPE_ALIAS)}"
        )
    return dtype_to_ir_type(_QUANT_TYPE_ALIAS[qdtype])


def quant_max_abs(qdtype: str, *, arch: str) -> float:
    """Clamp magnitude (positive) of :func:`quantize_scalar_f32` and
    :func:`quantize_prescaled_f32` for ``qdtype`` on ``arch``.

    ``127`` for i8 (CK Tile's symmetric range). For fp8e4m3 / bf8e5m2, the
    largest finite value of the encoding ``arch`` converts to
    (:func:`~rocke.helpers.fp8_encoding.fp8_element_format`): 240 / 57344 for the
    FNUZ encodings of gfx942, 448 / 57344 for the OCP ones of gfx950 and
    gfx1250. Raises ``ValueError`` for an fp8 / bf8 dtype on a target with no
    known encoding.
    """
    canon = _canon(qdtype)
    if canon == "i8":
        return _I8_MAX_ABS
    return fp8_element_format(canon, arch).largest_value


def validate_quant_out_dtype(out_dtype: str, arch: str) -> tuple[bool, str]:
    """Whether a quantizing kernel can write ``out_dtype`` on ``arch``:
    ``(True, "")``, else ``(False, reason)``.

    ``out_dtype`` must be a canonical :data:`QDType` name (no aliases). An
    fp8e4m3 / bf8e5m2 output needs the CDNA-only ``v_cvt_pk_{fp8,bf8}_f32``
    conversion (an RDNA target has none, and selecting it would abort with an
    uncatchable ``LLVM ERROR: Cannot select``) and a target with a known
    encoding (:func:`quant_max_abs`). ``arch`` must be a known target. The C++
    engine's twin is ``rocke_validate_quant_out_dtype``.
    """
    if out_dtype not in get_args(QDType):
        return False, f"unsupported out_dtype {out_dtype!r}"
    if out_dtype == "i8":
        return True, ""
    family = ArchTarget.from_gfx(arch).family
    if family != "cdna":
        return False, (
            f"out_dtype {out_dtype!r} needs the CDNA-only "
            f"v_cvt_pk_{{fp8,bf8}}_f32 conversion; {arch} (family "
            f"{family!r}) has no fp8/bf8 pack op -- use out_dtype='i8'"
        )
    try:
        quant_max_abs(out_dtype, arch=arch)
    except ValueError as error:
        return False, str(error)
    return True, ""


def ir_to_qdtype(t: Type) -> LogicalQDType:
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
    arch: str,
) -> Value:
    """Quantise one f32 scalar to ``qdtype`` as ``arch`` encodes it:
    ``quantize_prescaled_f32(b, x_f32 * inv_scale, qdtype=qdtype, arch=arch)``.

    ``inv_scale`` is in the *natural* direction (``inv_scale = 1 / scale``
    where ``scale = amax / quant_max``). Pre-computing the reciprocal
    outside the inner loop is the same trick CK Tile uses; it amortises
    one ``v_rcp_f32`` over the whole row.

    A scale the kernel derives from the data does not bound an infinite
    scaled value (see "Dynamic scales" in the module docstring).

    Returns a value whose IR type is the quant dtype (i8 / fp8e4m3 /
    bf8e5m2). Caller stores it via the standard ``global_store``. Raises
    ``ValueError`` for an fp8 / bf8 dtype on a target with no known
    encoding (:func:`quant_max_abs`).
    """
    if x_f32.type.name != "f32":
        raise ValueError(
            f"quantize_scalar_f32 expects f32 input, got {x_f32.type.name}"
        )
    if inv_scale.type.name != "f32":
        raise ValueError(
            f"quantize_scalar_f32 expects f32 inv_scale, got {inv_scale.type.name}"
        )
    # The bounds come before the multiply: IR value names follow op order,
    # and both engines' quantizing kernels are blessed with this order.
    canon, c_neg, c_pos = _quant_bounds(b, qdtype, arch=arch)
    scaled = b.fmul(x_f32, inv_scale)
    return _saturate_and_convert(b, scaled, canon=canon, c_neg=c_neg, c_pos=c_pos)


def quantize_prescaled_f32(
    b: IRBuilder,
    scaled_f32: Value,
    *,
    qdtype: str,
    arch: str,
) -> Value:
    """Convert one f32 scalar that is already scaled to ``qdtype`` as
    ``arch`` encodes it, clamped to the dtype's range (NaN kept for fp8 /
    bf8).

    Pipeline (single hardware path per output dtype):

    .. code-block:: text

        clamped = clamp_f32(scaled_f32, -quant_max_abs, +quant_max_abs)
        result  = cvt_f32_to_<qdtype>(scaled_f32 if isnan(scaled_f32) else clamped)

    For fp8 / bf8 the clamp is what makes the conversion saturate: the
    hardware conversion does not, so a value past the encoding's largest
    finite value (and +-inf) would become NaN, or +-inf for OCP bf8. The
    bound is that of ``arch``'s encoding (:func:`quant_max_abs`), so finite
    values and +-inf convert to finite values. A NaN is converted as it is
    and stays NaN (the encoding's NaN code); ``clamp_f32`` alone would turn
    it into the negative bound.

    For i8 the clamp sets the output range: ``cvt_f32_to_i8_sat`` alone
    saturates to ``[-128, 127]``, and the clamp narrows that to the
    symmetric ``[-127, 127]`` of the CK Tile reference. i8 has no NaN; a NaN
    input gives -127. (The i8 path does not use ``v_cvt_pk_i16_i32``; that
    instruction does saturate, but to the i16 range, so it would not bound
    an i8 result either.)

    Use it where the kernel scales the value some other way than
    :func:`quantize_scalar_f32`'s multiply, for example by dividing by the
    scale. Returns a value whose IR type is the quant dtype. Raises
    ``ValueError`` for an fp8 / bf8 dtype on a target with no known
    encoding (:func:`quant_max_abs`).
    """
    if scaled_f32.type.name != "f32":
        raise ValueError(
            f"quantize_prescaled_f32 expects f32 input, got {scaled_f32.type.name}"
        )
    canon, c_neg, c_pos = _quant_bounds(b, qdtype, arch=arch)
    return _saturate_and_convert(b, scaled_f32, canon=canon, c_neg=c_neg, c_pos=c_pos)


def _quant_bounds(
    b: IRBuilder, qdtype: str, *, arch: str
) -> tuple[QDType, Value, Value]:
    """``(canon, -bound, +bound)``: the canonical name of ``qdtype`` and the
    clamp bounds of :func:`quant_max_abs` as f32 constants (``+bound``
    emitted first)."""
    canon = _canon(qdtype)
    qmax = quant_max_abs(canon, arch=arch)
    c_pos = b.const_f32(qmax)
    c_neg = b.const_f32(-qmax)
    return canon, c_neg, c_pos


def _saturate_and_convert(
    b: IRBuilder, scaled_f32: Value, *, canon: QDType, c_neg: Value, c_pos: Value
) -> Value:
    """The rule both quantize helpers share: clamp ``scaled_f32`` to
    ``[c_neg, c_pos]`` and convert it to ``canon``, keeping a NaN for fp8 /
    bf8 (see :func:`quantize_prescaled_f32`). The caller builds the bounds,
    so it decides where they sit in the IR."""
    clamped = b.clamp_f32(scaled_f32, c_neg, c_pos)
    if canon == "i8":
        return b.cvt_f32_to_i8_sat(clamped)
    saturated = b.select(b.fcmp("uno", scaled_f32, scaled_f32), scaled_f32, clamped)
    if canon == "fp8e4m3":
        return b.cvt_f32_to_fp8(saturated)
    if canon == "bf8e5m2":
        return b.cvt_f32_to_bf8(saturated)
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
    inverse quant scale**. The fp8 / bf8 routes apply no clamp, and the
    conversion does not saturate (a value past the encoding's largest finite
    value becomes NaN, or +-inf for OCP bf8), so the caller's scale must keep
    the values within ``quant_max_abs(out_dtype, arch=...)``: the instances
    derive it as ``amax / quant_max_abs``. The packing routes are:

    * For ``fp8e4m3`` / ``bf8e5m2`` with a 4-wide chunk we issue one
      packed ``v_cvt_pk_fp8_f32`` (resp. ``v_cvt_pk_bf8_f32``) via
      :func:`IRBuilder.cvt_pk_fp8_f32x4` / ``cvt_pk_bf8_f32x4``,
      saving 3 scalar ``v_cvt_*_f32`` instructions.
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

    # i8 path (or VEC=2 fp8/bf8): per-element cast + vec_pack.
    qs: List[Value] = []
    for sf in scaled_f32:
        if out_dtype == "i8":
            qs.append(
                b.cvt_f32_to_i8_sat(
                    b.clamp_f32(sf, b.const_f32(-_I8_MAX_ABS), b.const_f32(_I8_MAX_ABS))
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
