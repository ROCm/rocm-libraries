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
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for :mod:`Tensile.Common.MXScaleFormatValidation`.

Covers the per-side and joint rules that gate gfx1250's
``v_wmma_scale_f32_16x16x128_f8f6f4`` (see ROCm/llvm-project#2634), the
human-readable formatter, the error builder, and the
``ProblemType``-level integration that converts an illegal YAML
combination into an ``Exception`` at config-load time.
"""

import pytest

from rocisa.enum import DataTypeEnum

from Tensile.Common.DataType import DataType
from Tensile.Common.MXScaleFormatValidation import (
    formatMXScaleFormatCombination,
    isFP4MatrixDataType,
    isMXMatrixDataType,
    isValidMXScaleFormatCombination,
    isValidMXScaleFormatForDataType,
    mxScaleFormatCombinationError,
)


# ---------------------------------------------------------------------------
# Local builders / vocabularies
# ---------------------------------------------------------------------------

def _dt(enum_member):
    """Build a DataType from a DataTypeEnum member."""
    return DataType(enum_member)


# Matrix-class shorthand -> DataType (gfx1250 MX classes).
_MATRIX_CLASS = {
    "FP8": _dt(DataTypeEnum.Float8),
    "BF8": _dt(DataTypeEnum.BFloat8),
    "FP6": _dt(DataTypeEnum.Float6),
    "BF6": _dt(DataTypeEnum.BFloat6),
    "FP4": _dt(DataTypeEnum.Float4),
}

# Scale-class shorthand -> DataType. "E4M3" is OCP FP8, which is
# DataTypeEnum.Float8 in tensilelite's vocabulary.
_SCALE_CLASS = {
    "E8":   _dt(DataTypeEnum.E8),
    "E5M3": _dt(DataTypeEnum.E5M3),
    "E4M3": _dt(DataTypeEnum.Float8),
}

# YAML spellings (case-insensitive via DataType.lookup).
_MATRIX_YAML = {
    "FP8": "F8",
    "BF8": "B8",
    "FP6": "F6",
    "BF6": "B6",
    "FP4": "F4",
}
_SCALE_YAML = {
    "E8":   "E8",
    "E5M3": "E5M3",
    "E4M3": "f8",   # E4M3 == OCP FP8
}


def _valid_combinations():
    """Enumerate every legal (A, AScale, B, BScale) tuple per the table
    and the BF8/BF6 mirror cases.

    Rules:
      * FP8 / BF8 / FP6 / BF6 require scale E8.
      * FP4 accepts E8, E5M3, or E4M3.
      * FP4 x FP4 requires the two scales to match.

    Yields tuples of shorthand strings: (aClass, aScale, bClass, bScale).
    """
    nonFp4 = ["FP8", "BF8", "FP6", "BF6"]
    fp4Scales = ["E8", "E5M3", "E4M3"]
    combos = []

    for a in nonFp4:
        for b in nonFp4:
            combos.append((a, "E8", b, "E8"))                          # 4 x 4 = 16
    for a in nonFp4:
        for s in fp4Scales:
            combos.append((a, "E8", "FP4", s))                         # 4 x 3 = 12
    for b in nonFp4:
        for s in fp4Scales:
            combos.append(("FP4", s, b, "E8"))                         # 3 x 4 = 12
    for s in fp4Scales:
        combos.append(("FP4", s, "FP4", s))                            # 3
    return combos


VALID_COMBINATIONS = _valid_combinations()


# ---------------------------------------------------------------------------
# isMXMatrixDataType
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("enum_member,expected", [
    (DataTypeEnum.Float8,         True),
    (DataTypeEnum.BFloat8,        True),
    (DataTypeEnum.Float6,         True),
    (DataTypeEnum.BFloat6,        True),
    (DataTypeEnum.Float4,         True),
    (DataTypeEnum.Float,          False),
    (DataTypeEnum.Half,           False),
    (DataTypeEnum.BFloat16,       False),
    (DataTypeEnum.Int8,           False),
    (DataTypeEnum.E8,             False),
    (DataTypeEnum.E5M3,           False),
    (DataTypeEnum.Float8_fnuz,    False),
    (DataTypeEnum.BFloat8_fnuz,   False),
])
def test_isMXMatrixDataType(enum_member, expected):
    assert isMXMatrixDataType(_dt(enum_member)) is expected


def test_isMXMatrixDataType_rejects_non_DataType():
    # Defensive: callers should pass DataType, not raw enums.
    assert isMXMatrixDataType(DataTypeEnum.Float4) is False
    assert isMXMatrixDataType(None) is False


# ---------------------------------------------------------------------------
# isFP4MatrixDataType
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("enum_member,expected", [
    (DataTypeEnum.Float4,         True),
    (DataTypeEnum.Float8,         False),
    (DataTypeEnum.BFloat8,        False),
    (DataTypeEnum.Float6,         False),
    (DataTypeEnum.BFloat6,        False),
    (DataTypeEnum.Half,           False),
    (DataTypeEnum.E8,             False),
    (DataTypeEnum.E5M3,           False),
])
def test_isFP4MatrixDataType(enum_member, expected):
    assert isFP4MatrixDataType(_dt(enum_member)) is expected


# ---------------------------------------------------------------------------
# isValidMXScaleFormatForDataType  (per-side rule)
# ---------------------------------------------------------------------------

# FP8/BF8/FP6/BF6 require E8 (and only E8).
_PER_SIDE_NON_FP4 = []
for _m in ("FP8", "BF8", "FP6", "BF6"):
    _PER_SIDE_NON_FP4.append((_m, "E8",   True))
    _PER_SIDE_NON_FP4.append((_m, "E5M3", False))
    _PER_SIDE_NON_FP4.append((_m, "E4M3", False))
    # BF8 is a legal *matrix* class but never a legal *scale*; cover that.
    _PER_SIDE_NON_FP4.append((_m, "_BF8_AS_SCALE_", False))

@pytest.mark.parametrize("matrixShort,scaleShort,expected", _PER_SIDE_NON_FP4)
def test_isValidMXScaleFormatForDataType_non_fp4(matrixShort, scaleShort, expected):
    matrixDt = _MATRIX_CLASS[matrixShort]
    if scaleShort == "_BF8_AS_SCALE_":
        scaleDt = _dt(DataTypeEnum.BFloat8)
    else:
        scaleDt = _SCALE_CLASS[scaleShort]
    assert isValidMXScaleFormatForDataType(matrixDt, scaleDt) is expected


# FP4 accepts E8 / E5M3 / E4M3, rejects everything else.
@pytest.mark.parametrize("scale_member,expected", [
    (DataTypeEnum.E8,       True),
    (DataTypeEnum.E5M3,     True),
    (DataTypeEnum.Float8,   True),    # E4M3
    (DataTypeEnum.Half,     False),
    (DataTypeEnum.BFloat8,  False),
    (DataTypeEnum.Float,    False),
    (DataTypeEnum.BFloat16, False),
])
def test_isValidMXScaleFormatForDataType_fp4(scale_member, expected):
    assert isValidMXScaleFormatForDataType(
        _MATRIX_CLASS["FP4"], _dt(scale_member)) is expected


# Non-MX matrix dtypes: the rule does not apply -> always True.
@pytest.mark.parametrize("matrix_member", [
    DataTypeEnum.Float,
    DataTypeEnum.Half,
    DataTypeEnum.BFloat16,
    DataTypeEnum.Int8,
    DataTypeEnum.Float8_fnuz,
    DataTypeEnum.BFloat8_fnuz,
])
@pytest.mark.parametrize("scale_member", [
    DataTypeEnum.E8,
    DataTypeEnum.E5M3,
    DataTypeEnum.Float8,
    DataTypeEnum.Half,
    DataTypeEnum.BFloat8,
])
def test_isValidMXScaleFormatForDataType_non_mx_matrix_always_true(
        matrix_member, scale_member):
    assert isValidMXScaleFormatForDataType(
        _dt(matrix_member), _dt(scale_member)) is True


# ---------------------------------------------------------------------------
# isValidMXScaleFormatCombination -- exhaustive valid table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aClass,aScale,bClass,bScale", VALID_COMBINATIONS,
                         ids=[f"{a}_{sa}__{b}_{sb}" for a, sa, b, sb in VALID_COMBINATIONS])
def test_valid_combinations_all_accepted(aClass, aScale, bClass, bScale):
    assert isValidMXScaleFormatCombination(
        _MATRIX_CLASS[aClass],
        _SCALE_CLASS[aScale],
        _MATRIX_CLASS[bClass],
        _SCALE_CLASS[bScale],
    ) is True


def test_valid_combinations_count():
    # Sanity guard: 4*4 + 4*3 + 3*4 + 3 == 43. If the table changes,
    # this test will alert reviewers to revisit the validator semantics.
    assert len(VALID_COMBINATIONS) == 43


# ---------------------------------------------------------------------------
# isValidMXScaleFormatCombination -- curated invalid table
# ---------------------------------------------------------------------------

INVALID_COMBINATIONS = [
    # FP8 with non-E8 scale on the A side
    ("FP8", "E5M3", "FP8", "E8"),
    ("FP8", "E4M3", "FP8", "E8"),
    ("FP8", "E5M3", "FP4", "E8"),
    # FP6 with non-E8 scale on the A side
    ("FP6", "E4M3", "FP8", "E8"),
    ("FP6", "E5M3", "FP4", "E5M3"),
    # BF8 / BF6 must also use E8
    ("BF8", "E5M3", "BF8", "E8"),
    ("BF6", "E4M3", "BF6", "E8"),
    # FP4 x FP4 with mismatched scales
    ("FP4", "E8",   "FP4", "E5M3"),
    ("FP4", "E5M3", "FP4", "E4M3"),
    ("FP4", "E4M3", "FP4", "E8"),
    # FP4 + non-FP4 where the non-FP4 side has an illegal scale
    ("FP4", "E5M3", "FP8", "E5M3"),
    ("FP4", "E4M3", "FP6", "E4M3"),
    # Both sides illegal
    ("FP8", "E5M3", "FP6", "E4M3"),
]


@pytest.mark.parametrize("aClass,aScale,bClass,bScale", INVALID_COMBINATIONS,
                         ids=[f"{a}_{sa}__{b}_{sb}" for a, sa, b, sb in INVALID_COMBINATIONS])
def test_invalid_combinations_rejected(aClass, aScale, bClass, bScale):
    assert isValidMXScaleFormatCombination(
        _MATRIX_CLASS[aClass],
        _SCALE_CLASS[aScale],
        _MATRIX_CLASS[bClass],
        _SCALE_CLASS[bScale],
    ) is False


# ---------------------------------------------------------------------------
# formatMXScaleFormatCombination
# ---------------------------------------------------------------------------

def test_formatMXScaleFormatCombination_canonical():
    s = formatMXScaleFormatCombination(
        _MATRIX_CLASS["FP4"],
        _SCALE_CLASS["E5M3"],
        _MATRIX_CLASS["FP4"],
        _SCALE_CLASS["E4M3"],
    )
    assert s == "(A=FP4, AScale=E5M3, B=FP4, BScale=E4M3)"


def test_formatMXScaleFormatCombination_all_matrix_classes_render_short():
    rendered = formatMXScaleFormatCombination(
        _MATRIX_CLASS["BF8"],
        _SCALE_CLASS["E8"],
        _MATRIX_CLASS["BF6"],
        _SCALE_CLASS["E8"],
    )
    assert rendered == "(A=BF8, AScale=E8, B=BF6, BScale=E8)"


def test_formatMXScaleFormatCombination_unknown_falls_back_to_enum_name():
    # Half is neither a matrix nor a scale class for MX, so the helper
    # must fall back to the enum name rather than emit a misleading label.
    rendered = formatMXScaleFormatCombination(
        _dt(DataTypeEnum.Half),
        _dt(DataTypeEnum.BFloat8),
        _MATRIX_CLASS["FP4"],
        _SCALE_CLASS["E8"],
    )
    assert "Half" in rendered
    assert "BFloat8" in rendered
    assert "B=FP4" in rendered
    assert "BScale=E8" in rendered


# ---------------------------------------------------------------------------
# mxScaleFormatCombinationError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aClass,aScale,bClass,bScale", VALID_COMBINATIONS[:8],
                         ids=[f"{a}_{sa}__{b}_{sb}" for a, sa, b, sb in VALID_COMBINATIONS[:8]])
def test_mxScaleFormatCombinationError_empty_for_valid(aClass, aScale, bClass, bScale):
    assert mxScaleFormatCombinationError(
        _MATRIX_CLASS[aClass],
        _SCALE_CLASS[aScale],
        _MATRIX_CLASS[bClass],
        _SCALE_CLASS[bScale],
    ) == ""


def test_mxScaleFormatCombinationError_fp4_mismatch_wording():
    err = mxScaleFormatCombinationError(
        _MATRIX_CLASS["FP4"],
        _SCALE_CLASS["E5M3"],
        _MATRIX_CLASS["FP4"],
        _SCALE_CLASS["E4M3"],
    )
    assert err.startswith("Invalid MX scale-format combination ")
    assert "(A=FP4, AScale=E5M3, B=FP4, BScale=E4M3)" in err
    assert "FP4 x FP4 requires AScale (E5M3) == BScale (E4M3)" in err
    assert "ROCm/llvm-project#2634" in err
    assert "table-valid-combinations.txt" in err


def test_mxScaleFormatCombinationError_a_side_violation_wording():
    err = mxScaleFormatCombinationError(
        _MATRIX_CLASS["FP8"],
        _SCALE_CLASS["E5M3"],
        _MATRIX_CLASS["FP8"],
        _SCALE_CLASS["E8"],
    )
    assert "matrix A class FP8 does not accept scale format E5M3" in err
    assert "matrix B" not in err.split("matrix A", 1)[1]
    assert "ROCm/llvm-project#2634" in err


def test_mxScaleFormatCombinationError_b_side_violation_wording():
    err = mxScaleFormatCombinationError(
        _MATRIX_CLASS["FP4"],
        _SCALE_CLASS["E5M3"],
        _MATRIX_CLASS["FP8"],
        _SCALE_CLASS["E5M3"],
    )
    assert "matrix B class FP8 does not accept scale format E5M3" in err
    assert "matrix A" not in err.split(":", 1)[1]
    assert "ROCm/llvm-project#2634" in err


def test_mxScaleFormatCombinationError_both_sides_violation_wording():
    err = mxScaleFormatCombinationError(
        _MATRIX_CLASS["FP8"],
        _SCALE_CLASS["E5M3"],
        _MATRIX_CLASS["FP6"],
        _SCALE_CLASS["E4M3"],
    )
    assert "matrix A class FP8 does not accept scale format E5M3" in err
    assert "matrix B class FP6 does not accept scale format E4M3" in err
    # FP8/FP6 sides already failed; the FP4xFP4 clause should not appear.
    assert "FP4 x FP4" not in err


# ---------------------------------------------------------------------------
# ProblemType integration
# ---------------------------------------------------------------------------

def _problem_type_config(*, dataType="F4", mxsA="E5M3", mxsB="E5M3",
                         mxBlockA=32, mxBlockB=32):
    return {
        "OperationType":          "GEMM",
        "DataType":               dataType,
        "DestDataType":           "s",
        "ComputeDataType":        "s",
        "TransposeA":             True,
        "TransposeB":             False,
        "UseBeta":                True,
        "Batched":                True,
        "HighPrecisionAccumulate": True,
        "MXBlockA":               mxBlockA,
        "MXBlockB":               mxBlockB,
        "DataTypeMXSA":           mxsA,
        "DataTypeMXSB":           mxsB,
    }


def test_problem_type_accepts_valid_fp4_combination():
    from Tensile.SolutionStructs.Problem import ProblemType
    # Should not raise.
    ProblemType(_problem_type_config(mxsA="E5M3", mxsB="E5M3"), False)


def test_problem_type_rejects_fp4_scale_mismatch():
    from Tensile.SolutionStructs.Problem import ProblemType
    # FP4 x FP4 with E5M3 vs E4M3 (encoded in YAML as 'f8') — the
    # FP4-only "scales must match" rule fires.
    with pytest.raises(Exception) as exc_info:
        ProblemType(
            _problem_type_config(mxsA="E5M3", mxsB="f8"),
            False,
        )
    assert "Invalid MX scale-format combination" in str(exc_info.value)
    assert "ROCm/llvm-project#2634" in str(exc_info.value)


def test_problem_type_no_mx_block_skips_validation():
    from Tensile.SolutionStructs.Problem import ProblemType
    # When both MX blocks are zero the validator must not fire even if
    # the MXS dtypes look nonsensical (they get carried but unused).
    ProblemType(
        _problem_type_config(
            dataType="s", mxsA="E5M3", mxsB="E5M3",
            mxBlockA=0, mxBlockB=0,
        ),
        False,
    )


# ---------------------------------------------------------------------------
# Cross-validation: YAML-spelling round trip through the validator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aClass,aScale,bClass,bScale", VALID_COMBINATIONS,
                         ids=[f"{a}_{sa}__{b}_{sb}" for a, sa, b, sb in VALID_COMBINATIONS])
def test_yaml_spellings_round_trip_valid(aClass, aScale, bClass, bScale):
    """Every legal tuple must remain legal when each dtype is built from
    its YAML spelling instead of from a DataTypeEnum constant. This
    guards against drift between the table-valid-combinations.txt
    encoding and the validator's enum view."""
    aDt      = DataType(_MATRIX_YAML[aClass])
    aScaleDt = DataType(_SCALE_YAML[aScale])
    bDt      = DataType(_MATRIX_YAML[bClass])
    bScaleDt = DataType(_SCALE_YAML[bScale])
    assert isValidMXScaleFormatCombination(aDt, aScaleDt, bDt, bScaleDt) is True
