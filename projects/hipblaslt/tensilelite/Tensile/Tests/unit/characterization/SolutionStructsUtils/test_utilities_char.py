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

"""Characterization tests for ``Tensile.SolutionStructs.Utilities`` — the small
pure helpers: ``getMiInputType`` (the 3 MI-operand-type branches), ``reject``
(the rejection-state machine), ``pvar``, ``roundupRatio``, and the
``.value``-based ``getRealDataType{A,B}`` mix-type mappers.
"""

import pytest
from rocisa.enum import DataTypeEnum

from Tensile.Common.DataType import DataType
import Tensile.SolutionStructs.Utilities as U

pytestmark = pytest.mark.unit


# ===========================================================================
# getMiInputType — 3 branches
# ===========================================================================

def test_mi_input_type_xf32_emulation(snapshot):
    # EnableF32XdlMathOp + UseF32XEmulation -> BFloat16.
    kernel = {
        "EnableF32XdlMathOp": True,
        "UseF32XEmulation": True,
        "ProblemType": {"F32XdlMathOp": DataType("x"), "DataType": DataType("s")},
    }
    assert U.getMiInputType(kernel).toName() == snapshot


def test_mi_input_type_native_xf32(snapshot):
    # EnableF32XdlMathOp only -> ProblemType["F32XdlMathOp"].
    kernel = {
        "EnableF32XdlMathOp": True,
        "UseF32XEmulation": False,
        "ProblemType": {"F32XdlMathOp": DataType("x"), "DataType": DataType("s")},
    }
    assert U.getMiInputType(kernel).toName() == snapshot


def test_mi_input_type_plain(snapshot):
    # Neither -> ProblemType["DataType"].
    kernel = {
        "EnableF32XdlMathOp": False,
        "UseF32XEmulation": False,
        "ProblemType": {"F32XdlMathOp": DataType("x"), "DataType": DataType("h")},
    }
    assert U.getMiInputType(kernel).toName() == snapshot


# ===========================================================================
# reject — the rejection-state machine
# ===========================================================================

def test_reject_no_reject_flag(snapshot):
    # NoReject set -> always returns False, state untouched.
    state = {"NoReject": True, "Valid": True}
    result = U.reject(state, False, "some reason")
    assert {"result": result, "valid": state["Valid"]} == snapshot


def test_reject_quiet(snapshot):
    # printSolutionRejectionReason=False -> sets Valid=False, returns True.
    state = {"Valid": True}
    result = U.reject(state, False, "bad tile")
    assert {"result": result, "valid": state["Valid"]} == snapshot


def test_reject_none_state(snapshot):
    # state=None -> falls through to implicit None.
    assert U.reject(None, False) == snapshot


def test_reject_print_no_solution_index(snapshot, capsys):
    # printReason=True with no/sentinel SolutionIndex -> prints, no raise, then
    # Valid=False / True.
    state = {"Valid": True, "SolutionIndex": -1}
    result = U.reject(state, True, "reason-a", "reason-b")
    assert {"result": result, "valid": state["Valid"]} == snapshot


def test_reject_valid_solution_index_raises(snapshot):
    # printReason=True with a valid SolutionIndex -> Exception (a rejection of a
    # LibraryLogic solution is unexpected).
    state = {"SolutionIndex": 5, "SolutionNameMin": "Sol_gfx942_x"}
    with pytest.raises(Exception) as excinfo:
        U.reject(state, True, "unexpected")
    assert str(excinfo.value) == snapshot


def test_reject_valid_index_problemtype_name_fallback(snapshot):
    # No SolutionNameMin -> the message falls back to str(state["ProblemType"]).
    state = {"SolutionIndex": 7, "ProblemType": "PT_placeholder"}
    with pytest.raises(Exception) as excinfo:
        U.reject(state, True)
    assert str(excinfo.value) == snapshot


# ===========================================================================
# pvar / roundupRatio
# ===========================================================================

def test_pvar(snapshot):
    assert U.pvar({"DepthU": 16}, "DepthU") == snapshot


@pytest.mark.parametrize(
    "dividend,divisor",
    [(10, 3), (9, 3), (0, 5), (1, 1), (7, 2)],
    ids=["remainder", "exact", "zero", "one", "half"],
)
def test_roundup_ratio(dividend, divisor, snapshot):
    assert U.roundupRatio(dividend, divisor) == snapshot


# ===========================================================================
# getRealDataTypeA / getRealDataTypeB — .value-based mix-type mapping
# ===========================================================================

_DT_CASES = [
    "Float8BFloat8",
    "BFloat8Float8",
    "Float8BFloat8_fnuz",
    "BFloat8Float8_fnuz",
    "Float",   # passthrough (else)
    "Half",    # passthrough
]


@pytest.mark.parametrize("name", _DT_CASES, ids=_DT_CASES)
def test_get_real_data_type_a(name, snapshot):
    assert U.getRealDataTypeA(DataType(getattr(DataTypeEnum, name))).toName() == snapshot


@pytest.mark.parametrize("name", _DT_CASES, ids=_DT_CASES)
def test_get_real_data_type_b(name, snapshot):
    assert U.getRealDataTypeB(DataType(getattr(DataTypeEnum, name))).toName() == snapshot
