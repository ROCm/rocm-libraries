# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 MXFP8 activation quantizer knobs that need a Python-only op.

:mod:`rocke.helpers.mx_quant` emits only ops both engines lower. The
``cvt="intrinsic"`` value of the convert knob uses
:meth:`~rocke.core.ir.IRBuilder.cvt_scalef32_pk_fp8_f32`
(``arith.cvt_scalef32_pk_fp8_f32``, in ``rocke.core.ir.PYTHON_ONLY_OPCODES``),
so per the placement policy it lives here, in the library family, and reaches
the helper as its ``cvt`` callable.

==============  =========  ===================================================
``cvt``         engines    convert route
==============  =========  ===================================================
``asm``         both       ``tile.inline_asm`` (the helper's own default)
``intrinsic``   python     ``llvm.amdgcn.cvt.scalef32.pk.fp8.f32``
==============  =========  ===================================================

A kernel built with ``cvt="intrinsic"`` compiles with ``backend="python"``
(:func:`backend_for`), carries the ``_pyonly`` name suffix
(:func:`kernel_name_suffix`) and is excluded from the C++ byte-identity gate.
Every other knob (``reduce``, ``nan``, ``xor16``) passes through to the helper.
"""

from __future__ import annotations

from collections.abc import Sequence

from rocke.core.ir import IRBuilder, Value
from rocke.helpers import mx_quant as _mq

QUANT_CVT_MODES = ("asm", "intrinsic")
DEFAULT_QUANT_CVT = "asm"


def intrinsic_cvt(b: IRBuilder, old, lo: Value, hi_v: Value, scale: Value, *, hi: bool):
    """The helper's ``cvt`` callable for the ``intrinsic`` route."""
    if old is None:
        old = b.const_i32(0)
    return b.cvt_scalef32_pk_fp8_f32(old, lo, hi_v, scale, hi=hi)


def cvt_impl(cvt: str):
    """The helper ``cvt`` argument for a knob value."""
    if cvt == "asm":
        return "asm"
    if cvt == "intrinsic":
        return intrinsic_cvt
    raise ValueError(f"cvt must be one of {QUANT_CVT_MODES}, got {cvt!r}")


def backend_for(cvt: str) -> str | None:
    """``compile_kernel`` backend a build with ``cvt`` must use."""
    cvt_impl(cvt)
    return "python" if cvt == "intrinsic" else None


def kernel_name_suffix(cvt: str) -> str:
    """Kernel-name suffix marking builds outside the C++ byte-identity gate."""
    return "_pyonly" if backend_for(cvt) == "python" else ""


def pack_fp8x4(b: IRBuilder, v0, v1, v2, v3, scale, *, cvt: str = DEFAULT_QUANT_CVT):
    """:func:`rocke.helpers.mx_quant.pack_fp8x4` with the ``cvt`` knob."""
    return _mq.pack_fp8x4(b, v0, v1, v2, v3, scale, cvt=cvt_impl(cvt))


def quantize_mxfp8(
    b: IRBuilder,
    values: Sequence[Value],
    lanes: int,
    *,
    cvt: str = DEFAULT_QUANT_CVT,
    reduce: str = "dpp",
    nan: str = "propagate",
    xor16: str = "swizzle",
):
    """:func:`rocke.helpers.mx_quant.quantize_mxfp8` with the ``cvt`` knob."""
    return _mq.quantize_mxfp8(
        b, values, lanes, cvt=cvt_impl(cvt), reduce=reduce, nan=nan, xor16=xor16
    )


__all__ = [
    "DEFAULT_QUANT_CVT",
    "QUANT_CVT_MODES",
    "backend_for",
    "cvt_impl",
    "intrinsic_cvt",
    "kernel_name_suffix",
    "pack_fp8x4",
    "quantize_mxfp8",
]
