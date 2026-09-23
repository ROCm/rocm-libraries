################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""MX scale-format combination validator for gfx1250's
``v_wmma_scale_f32_16x16x128_f8f6f4`` MX path.

Lives under ``Tensile/SolutionStructs/Validators/`` so other Solution-level
validators can join it.

Public entry point: :func:`validateMXScaleFormatCombination`.
"""

from rocisa.enum import DataTypeEnum

from ..Utilities import reject


# ============================================================================
# MX scale-format combination validation (gfx1250 v_wmma_scale rules).
#
# gfx1250's v_wmma_scale_f32_16x16x128_f8f6f4 only accepts a fixed set of
# (matrix-A class, A-scale fmt, matrix-B class, B-scale fmt) tuples. The
# AMDGPU assembler does not enforce these joint constraints (see
# ROCm/llvm-project#2634), so the kernel generator has to reject candidates
# that would otherwise codegen into illegal encodings.
#
# Tensilelite enum spellings used below:
#   - matrix FP8        -> DataTypeEnum.Float8           (also FP8 _fnuz)
#   - matrix BF8        -> DataTypeEnum.BFloat8          (also BF8 _fnuz)
#   - matrix FP6 / BF6  -> DataTypeEnum.Float6 / BFloat6
#   - matrix FP4        -> DataTypeEnum.Float4
#   - scale  E8         -> DataTypeEnum.E8     (UE8M0)
#   - scale  E5M3       -> DataTypeEnum.E5M3
#   - scale  E4M3       -> DataTypeEnum.Float8 (same byte as OCP FP8)
# ============================================================================

# Matrix classes governed by the gfx1250 f8f6f4 MX rules.
_MX_FP8_LIKE = frozenset({
    DataTypeEnum.Float8.value,
    DataTypeEnum.Float8_fnuz.value,
})
_MX_BF8_LIKE = frozenset({
    DataTypeEnum.BFloat8.value,
    DataTypeEnum.BFloat8_fnuz.value,
})
_MX_F6_LIKE  = frozenset({
    DataTypeEnum.Float6.value,
    DataTypeEnum.BFloat6.value,
})
_MX_FP4      = frozenset({DataTypeEnum.Float4.value})
_MX_ALL      = _MX_FP8_LIKE | _MX_BF8_LIKE | _MX_F6_LIKE | _MX_FP4

# Legal scale formats per matrix class (FP4 has 3, everyone else needs E8).
_E8_ONLY      = frozenset({DataTypeEnum.E8.value})
_FP4_SCALES   = frozenset({
    DataTypeEnum.E8.value,
    DataTypeEnum.E5M3.value,
    DataTypeEnum.Float8.value,  # E4M3 byte
})


def _mxMatrixLabel(dtValue):
    """ISA-spec spelling for the matrix class enum value."""
    if dtValue in _MX_FP8_LIKE:
        return "FP8"
    if dtValue in _MX_BF8_LIKE:
        return "BF8"
    if dtValue == DataTypeEnum.Float6.value:
        return "FP6"
    if dtValue == DataTypeEnum.BFloat6.value:
        return "BF6"
    if dtValue == DataTypeEnum.Float4.value:
        return "FP4"
    return str(dtValue)


def _mxScaleLabel(dtValue):
    """ISA-spec spelling for an MX scale enum value."""
    if dtValue == DataTypeEnum.E8.value:
        return "E8"
    if dtValue == DataTypeEnum.E5M3.value:
        return "E5M3"
    if dtValue == DataTypeEnum.Float8.value:
        return "E4M3"
    return str(dtValue)


def _mxEnumValue(field):
    """Resolve a ProblemType field that may be a ``DataType`` instance, a raw
    ``DataTypeEnum``, or ``None`` to its underlying integer enum value (or
    ``None``).

    Two shapes are produced upstream depending on where in the pipeline
    Solution.py is called from: ``DataType`` wrappers (typical during
    ``assignDerivedParameters``) and raw ``DataTypeEnum`` values (e.g.,
    states that have already been serialized/flattened by callers that
    unwrap to the enum). Both must be accepted."""
    if field is None:
        return None
    # ``DataType.value`` is the underlying *int* (see Tensile.Common.DataType);
    # ``DataTypeEnum.X.value`` is also an int. So the first getattr unwraps
    # DataType -> int, and the second is a no-op for ints / unwraps a raw
    # DataTypeEnum -> int.
    value = getattr(field, "value", field)
    return getattr(value, "value", value)


def _isLegalMXScaleForMatrix(matrixVal, scaleVal):
    """Per-side rule: does the matrix class accept this scale dtype?"""
    if matrixVal not in _MX_ALL:
        return True  # Non-MX matrix class - the joint MX rules don't apply.
    if matrixVal in _MX_FP4:
        return scaleVal in _FP4_SCALES
    return scaleVal in _E8_ONLY  # FP8/BF8/FP6/BF6


def validateMXScaleFormatCombination(state, asmCaps, printRejectionReason):
    """Reject candidate solutions whose joint MX scale-format tuple is not
    legal on gfx1250's ``v_wmma_scale_f32_16x16x128_f8f6f4``.

    The rules below describe the gfx1250 ``HasWMMA_V3`` MX path only. Other
    architectures use different MX instructions whose joint constraints are
    not the same, so the helper short-circuits to ``True`` whenever
    ``asmCaps["HasWMMA_V3"]`` is false.

    Rules enforced (per the ISA / table-valid-combinations.txt):

    * FP8 / BF8 / FP6 / BF6 (incl. _fnuz variants) must pair with E8 (UE8M0)
      scale.
    * FP4 accepts E8, E5M3, or E4M3 scale.
    * When both A and B are FP4 the two scales must match.

    Sides whose ``MXBlock`` is 0 carry no MX scale and are skipped: the
    helper also short-circuits to ``True`` when neither side has MX scaling.

    Args:
        state: Solution state dict. Reads
            ``state["ProblemType"]["DataType{A,B}"]`` (matrix dtype),
            ``state["ProblemType"]["DataTypeMXS{A,B}"]`` (scale dtype),
            ``state["ProblemType"]["MXBlock{A,B}"]`` (int).
        asmCaps: Mapping with at least ``"HasWMMA_V3"`` (bool). Used to
            gate the gfx1250-specific rule set; non-gfx1250 candidates
            are passed through untouched.
        printRejectionReason: Forwarded to :func:`reject`.

    Returns:
        ``True`` if the state is valid (no reject fired); ``False`` if a
        reject was emitted, in which case ``state["Valid"]`` is also
        ``False`` and the caller should propagate the early return upstream.
    """
    # gfx1250 WMMA_V3 MX path only. On other arches the joint constraints
    # come from different instructions (MFMA on gfx9*, older WMMA on
    # gfx10/11/12) and over-rejecting here would discard legitimate kernels.
    if not asmCaps.get("HasWMMA_V3", False):
        return True

    pt = state["ProblemType"]
    mxBlockA = pt.get("MXBlockA", 0)
    mxBlockB = pt.get("MXBlockB", 0)
    if not mxBlockA and not mxBlockB:
        return True  # No MX scaling on either side - rules don't apply.

    # Resolve enum values. A side without MX scaling is reported as (None,
    # None) so the FP4xFP4 joint rule cannot fire spuriously across a
    # mixed MX / non-MX problem and the per-side rule short-circuits.
    if mxBlockA:
        aMatrix = _mxEnumValue(pt.get("DataTypeA"))
        aScale  = _mxEnumValue(pt.get("DataTypeMXSA"))
    else:
        aMatrix, aScale = None, None
    if mxBlockB:
        bMatrix = _mxEnumValue(pt.get("DataTypeB"))
        bScale  = _mxEnumValue(pt.get("DataTypeMXSB"))
    else:
        bMatrix, bScale = None, None

    reasons = []
    if mxBlockA and not _isLegalMXScaleForMatrix(aMatrix, aScale):
        reasons.append(
            "matrix A class %s does not accept scale format %s"
            % (_mxMatrixLabel(aMatrix), _mxScaleLabel(aScale)))
    if mxBlockB and not _isLegalMXScaleForMatrix(bMatrix, bScale):
        reasons.append(
            "matrix B class %s does not accept scale format %s"
            % (_mxMatrixLabel(bMatrix), _mxScaleLabel(bScale)))
    # FP4 x FP4 -> scales must match. (FP6/FP8/BF6/BF8 each already pin scale
    # to E8, so a mixed-class problem cannot have mismatching scales except
    # via the FP4-only rule.)
    if (mxBlockA and mxBlockB
        and aMatrix in _MX_FP4 and bMatrix in _MX_FP4
        and _isLegalMXScaleForMatrix(aMatrix, aScale)
        and _isLegalMXScaleForMatrix(bMatrix, bScale)
        and aScale != bScale):
        reasons.append(
            "FP4 x FP4 requires AScale (%s) == BScale (%s)"
            % (_mxScaleLabel(aScale), _mxScaleLabel(bScale)))

    if not reasons:
        return True

    tuple_str = (
        "(A=%s, AScale=%s, B=%s, BScale=%s)"
        % (_mxMatrixLabel(aMatrix) if aMatrix is not None else "None",
           _mxScaleLabel(aScale)   if aScale  is not None else "None",
           _mxMatrixLabel(bMatrix) if bMatrix is not None else "None",
           _mxScaleLabel(bScale)   if bScale  is not None else "None"))
    reject(state, printRejectionReason,
           "Invalid MX scale-format combination %s: %s; "
           "see table-valid-combinations.txt / ROCm/llvm-project#2634."
           % (tuple_str, "; ".join(reasons)))
    return False
