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

"""Characterization tests for
``Validators.MXScaleFormat.validateMXScaleFormatCombination`` and its
private label/legality helpers.

These tests PIN CURRENT BEHAVIOUR with syrupy snapshots: each case builds a
minimal Solution state, calls the public validator (or a helper directly),
and snapshots the structured result. The validator is a pure dict-in /
bool-out function whose only side effect (`reject`) is run with
``printRejectionReason=False`` so it silently sets ``state["Valid"]`` and
emits no stdout — deterministic, no GPU, no client.

Snapshot shape for the public fn: ``{"returned": bool, "valid": <Valid or
"unset">}`` — pins both the return value and the state mutation.
"""

import pytest

from rocisa.enum import DataTypeEnum
from Tensile.SolutionStructs.Validators.MXScaleFormat import (
    validateMXScaleFormatCombination,
    _mxMatrixLabel,
    _mxScaleLabel,
    _mxEnumValue,
    _isLegalMXScaleForMatrix,
)

pytestmark = pytest.mark.unit

# ISA-spec shorthands (tensilelite enum spellings, see module docstring).
FP8, FP8_FNUZ = DataTypeEnum.Float8, DataTypeEnum.Float8_fnuz
BF8, BF8_FNUZ = DataTypeEnum.BFloat8, DataTypeEnum.BFloat8_fnuz
FP6, BF6 = DataTypeEnum.Float6, DataTypeEnum.BFloat6
FP4 = DataTypeEnum.Float4
E8, E5M3, E4M3 = DataTypeEnum.E8, DataTypeEnum.E5M3, DataTypeEnum.Float8
NONMX = DataTypeEnum.BFloat16  # a matrix class outside the MX f8f6f4 rules


def _state(aMat, aScale, bMat, bScale, mxA=32, mxB=32):
    return {
        "ProblemType": {
            "DataTypeA": aMat, "DataTypeMXSA": aScale, "MXBlockA": mxA,
            "DataTypeB": bMat, "DataTypeMXSB": bScale, "MXBlockB": mxB,
        }
    }


def _run(state, hasWmmaV3=True):
    caps = {"HasWMMA_V3": hasWmmaV3}
    returned = validateMXScaleFormatCombination(state, caps, printRejectionReason=False)
    return {"returned": returned, "valid": state.get("Valid", "unset")}


# --- public validator: representative legal / illegal combinations ----------

@pytest.mark.parametrize("name,aMat,aScale,bMat,bScale", [
    ("fp8_e8__fp8_e8",   FP8, E8,   FP8, E8),    # FP8/BF8/FP6/BF6 must pair with E8
    ("bf8_e8__fp6_e8",   BF8, E8,   FP6, E8),
    ("bf6_e8__fp8fnuz",  BF6, E8,   FP8_FNUZ, E8),
    ("fp4_e8__fp4_e8",   FP4, E8,   FP4, E8),    # FP4 accepts E8/E5M3/E4M3
    ("fp4_e5m3_match",   FP4, E5M3, FP4, E5M3),
    ("fp4_e4m3_match",   FP4, E4M3, FP4, E4M3),
])
def test_legal_combinations(name, aMat, aScale, bMat, bScale, snapshot):
    assert _run(_state(aMat, aScale, bMat, bScale)) == snapshot(name=name)


@pytest.mark.parametrize("name,aMat,aScale,bMat,bScale", [
    ("fp8_with_e5m3_illegal",  FP8, E5M3, FP8, E8),   # FP8 needs E8
    ("bf8_with_e4m3_illegal",  BF8, E4M3, BF8, E8),   # BF8 needs E8
    ("fp6_with_e5m3_illegal",  FP6, E5M3, FP6, E8),
    ("fp4xfp4_scale_mismatch", FP4, E5M3, FP4, E4M3), # FP4xFP4 scales must match
    ("matrixB_illegal",        FP4, E8,   BF8, E5M3),
])
def test_illegal_combinations(name, aMat, aScale, bMat, bScale, snapshot):
    assert _run(_state(aMat, aScale, bMat, bScale)) == snapshot(name=name)


# --- short-circuit paths ----------------------------------------------------

def test_non_gfx1250_passes_through(snapshot):
    # HasWMMA_V3 False -> rule set does not apply, always valid.
    assert _run(_state(FP8, E5M3, FP8, E5M3), hasWmmaV3=False) == snapshot


def test_no_mx_scaling_on_either_side(snapshot):
    assert _run(_state(FP8, E8, FP8, E8, mxA=0, mxB=0)) == snapshot


def test_one_sided_mx_a_only(snapshot):
    # B side has no MX block -> (None, None); only A is checked.
    assert _run(_state(FP4, E5M3, FP8, E8, mxB=0)) == snapshot


def test_one_sided_mx_b_only(snapshot):
    # A side has no MX block (mxBlockA falsy branch); only B is checked.
    assert _run(_state(FP8, E8, FP4, E5M3, mxA=0)) == snapshot


def test_none_scale_resolves_to_none(snapshot):
    # MXBlockA set but scale dtype None -> _mxEnumValue(None) path (line ~120)
    # and _mxScaleLabel(None) str-fallback (line ~106); illegal -> rejected.
    assert _run(_state(FP8, None, FP8, E8)) == snapshot


def test_non_mx_matrix_class_is_legal(snapshot):
    # Matrix class outside _MX_ALL with MX block set -> _isLegalMXScaleForMatrix
    # returns True early (line ~132); no rejection from that side.
    assert _run(_state(NONMX, E8, FP8, E8)) == snapshot


# --- private helpers, pinned directly --------------------------------------

@pytest.mark.parametrize("name,val", [
    ("fp8", FP8.value), ("fp8_fnuz", FP8_FNUZ.value),
    ("bf8", BF8.value), ("bf8_fnuz", BF8_FNUZ.value),
    ("fp6", FP6.value), ("bf6", BF6.value), ("fp4", FP4.value),
])
def test_mx_matrix_label(name, val, snapshot):
    assert _mxMatrixLabel(val) == snapshot(name=name)


@pytest.mark.parametrize("name,val", [
    ("e8", E8.value), ("e5m3", E5M3.value), ("e4m3", E4M3.value),
])
def test_mx_scale_label(name, val, snapshot):
    assert _mxScaleLabel(val) == snapshot(name=name)


def test_mx_enum_value_variants(snapshot):
    # Accepts raw DataTypeEnum, its int .value, and None.
    out = {
        "from_enum": _mxEnumValue(FP8),
        "from_int": _mxEnumValue(FP8.value),
        "from_none": _mxEnumValue(None),
    }
    assert out == snapshot


@pytest.mark.parametrize("name,mat,scale", [
    ("fp8_e8_ok",      FP8.value, E8.value),
    ("fp8_e5m3_bad",   FP8.value, E5M3.value),
    ("fp4_e4m3_ok",    FP4.value, E4M3.value),
    ("nonmx_anything", NONMX.value, E5M3.value),
])
def test_is_legal_mx_scale_for_matrix(name, mat, scale, snapshot):
    assert _isLegalMXScaleForMatrix(mat, scale) == snapshot(name=name)
