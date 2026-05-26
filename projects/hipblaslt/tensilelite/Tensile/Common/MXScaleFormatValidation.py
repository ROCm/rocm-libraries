################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
"""MX scale-format combination validation (Python mirror of the C++ header
``Tensile/MXScaleFormatValidation.hpp``).

The gfx1250 ISA constrains the legal
(matrix_a_fmt, matrix_a_scale_fmt, matrix_b_fmt, matrix_b_scale_fmt)
tuples accepted by ``v_wmma_scale_f32_16x16x128_f8f6f4``. The AMDGPU
assembler currently does not enforce these joint constraints (see
ROCm/llvm-project#2634), so Tensilelite must validate them on the host
before generating kernels that would otherwise be silently miscompiled.

Per the ISA:

* matrix class FP8 / BF8 / FP6 / BF6 must pair with scale E8 (UE8M0).
* matrix class FP4 may pair with E8 (UE8M0), E5M3, or E4M3 (= FP8 OCP).
* when both A and B are FP4, the two scales must match.

In Tensilelite's enum vocabulary:

* ``E8`` scale          -> :class:`DataTypeEnum.E8`
* ``E5M3`` scale        -> :class:`DataTypeEnum.E5M3`
* ``E4M3`` scale        -> :class:`DataTypeEnum.Float8` (OCP FP8 E4M3)

The fnuz (NANOO) FP8/BF8 variants are gfx940/gfx950 dtypes and are *not*
governed by the gfx1250 f8f6f4 rules; they are deliberately excluded from
:func:`isMXMatrixDataType`.

The public API takes :class:`Tensile.Common.DataType.DataType` objects
(not raw enums) so callers can pass what ``Problem.py`` already holds.
"""

from rocisa.enum import DataTypeEnum

from Tensile.Common.DataType import DataType


__all__ = [
    "isMXMatrixDataType",
    "isFP4MatrixDataType",
    "isValidMXScaleFormatForDataType",
    "isValidMXScaleFormatCombination",
    "formatMXScaleFormatCombination",
    "mxScaleFormatCombinationError",
]


def isMXMatrixDataType(dt: DataType) -> bool:
    """True iff ``dt`` is an MX matrix class (FP8, BF8, FP6, BF6, FP4)
    governed by the gfx1250 ``v_wmma_scale_f32_16x16x128_f8f6f4`` rules.

    The NANOO (fnuz) FP8/BF8 variants live on gfx940/gfx950 and are
    intentionally excluded; the per-side scale-format constraint does
    not apply to them.
    """
    if not isinstance(dt, DataType):
        return False
    return (dt.isFloat8()
            or dt.isBFloat8()
            or dt.isFloat6()
            or dt.isBFloat6()
            or dt.isFloat4())


def isFP4MatrixDataType(dt: DataType) -> bool:
    """True iff ``dt`` is the FP4 matrix class.

    FP4 is the only MX matrix class that admits more than one legal
    scale format, so callers must single it out when enforcing the
    FP4xFP4 "scales must match" rule.
    """
    if not isinstance(dt, DataType):
        return False
    return dt.isFloat4()


def isValidMXScaleFormatForDataType(matrixDt: DataType, scaleDt: DataType) -> bool:
    """Per-side rule for one (matrix dtype, scale dtype) pair.

    * FP8 / BF8 / FP6 / BF6  require ``E8``.
    * FP4 accepts ``E8``, ``E5M3``, or ``E4M3`` (= ``DataTypeEnum.Float8``).
    * For any non-MX matrix dtype this returns ``True`` (the rule does
      not apply; the host stack uses other paths for scaling there).
    """
    if not isMXMatrixDataType(matrixDt):
        return True

    if isFP4MatrixDataType(matrixDt):
        return (scaleDt.value == DataTypeEnum.E8.value
                or scaleDt.value == DataTypeEnum.E5M3.value
                or scaleDt.value == DataTypeEnum.Float8.value)

    return scaleDt.value == DataTypeEnum.E8.value


def isValidMXScaleFormatCombination(aType: DataType,
                                    scaleAType: DataType,
                                    bType: DataType,
                                    scaleBType: DataType) -> bool:
    """True iff the joint (A, scaleA, B, scaleB) tuple is legal for
    ``v_wmma_scale_f32_16x16x128_f8f6f4`` on gfx1250.

    The per-side rules of :func:`isValidMXScaleFormatForDataType` apply
    independently, and FP4 x FP4 additionally requires the two scale
    formats to match. (FP6/FP8/BF6/BF8 each already pin their scale to
    E8, so a mixed-class problem cannot have mismatching scales except
    via the FP4-only rule.)
    """
    if not isValidMXScaleFormatForDataType(aType, scaleAType):
        return False
    if not isValidMXScaleFormatForDataType(bType, scaleBType):
        return False

    if isFP4MatrixDataType(aType) and isFP4MatrixDataType(bType):
        if scaleAType.value != scaleBType.value:
            return False

    return True


def _mxScaleLabel(dt: DataType) -> str:
    """Stable label that maps tensilelite's enum back to the spelling
    used in the ISA / valid-combinations table."""
    if dt.value == DataTypeEnum.E8.value:
        return "E8"
    if dt.value == DataTypeEnum.E5M3.value:
        return "E5M3"
    if dt.value == DataTypeEnum.Float8.value:
        return "E4M3"
    return dt.toName()


def _mxMatrixLabel(dt: DataType) -> str:
    """Stable label for the MX matrix class."""
    if dt.value == DataTypeEnum.Float8.value:
        return "FP8"
    if dt.value == DataTypeEnum.BFloat8.value:
        return "BF8"
    if dt.value == DataTypeEnum.Float6.value:
        return "FP6"
    if dt.value == DataTypeEnum.BFloat6.value:
        return "BF6"
    if dt.value == DataTypeEnum.Float4.value:
        return "FP4"
    return dt.toName()


def formatMXScaleFormatCombination(aType: DataType,
                                   scaleAType: DataType,
                                   bType: DataType,
                                   scaleBType: DataType) -> str:
    """Human-readable rendering of the (A, scaleA, B, scaleB) tuple,
    e.g. ``"(A=FP4, AScale=E5M3, B=FP4, BScale=E4M3)"``.

    Matrix classes are spelled ``FP8`` / ``BF8`` / ``FP6`` / ``BF6`` /
    ``FP4``; scale classes are spelled ``E8`` / ``E5M3`` / ``E4M3``.
    Anything outside that set falls back to the raw enum name so log
    output remains unambiguous.
    """
    return "(A={}, AScale={}, B={}, BScale={})".format(
        _mxMatrixLabel(aType),
        _mxScaleLabel(scaleAType),
        _mxMatrixLabel(bType),
        _mxScaleLabel(scaleBType),
    )


def mxScaleFormatCombinationError(aType: DataType,
                                  scaleAType: DataType,
                                  bType: DataType,
                                  scaleBType: DataType) -> str:
    """Return ``""`` if the combination is legal, otherwise a diagnostic
    string explaining which side(s) violate which rule and pointing at
    the upstream bug.

    The wording matches the C++ implementation in
    ``src/MXScaleFormatValidation.cpp`` so the codebase speaks with one
    voice across host and library layers.
    """
    if isValidMXScaleFormatCombination(aType, scaleAType, bType, scaleBType):
        return ""

    parts = []
    parts.append("Invalid MX scale-format combination ")
    parts.append(formatMXScaleFormatCombination(aType, scaleAType, bType, scaleBType))
    parts.append(": ")

    aSideValid = isValidMXScaleFormatForDataType(aType, scaleAType)
    bSideValid = isValidMXScaleFormatForDataType(bType, scaleBType)

    if not aSideValid:
        parts.append("matrix A class {} does not accept scale format {}; ".format(
            _mxMatrixLabel(aType), _mxScaleLabel(scaleAType)))
    if not bSideValid:
        parts.append("matrix B class {} does not accept scale format {}; ".format(
            _mxMatrixLabel(bType), _mxScaleLabel(scaleBType)))

    if (isFP4MatrixDataType(aType)
            and isFP4MatrixDataType(bType)
            and scaleAType.value != scaleBType.value
            and aSideValid
            and bSideValid):
        parts.append("FP4 x FP4 requires AScale ({}) == BScale ({}); ".format(
            _mxScaleLabel(scaleAType), _mxScaleLabel(scaleBType)))

    parts.append("see table-valid-combinations.txt / ROCm/llvm-project#2634.")
    return "".join(parts)
