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

"""Characterization tests for the pure serialisation primitives in
``Tensile.LibraryIO``: the hand-rolled fast YAML emitter
(``_fast_yaml_scalar`` / ``_fast_yaml_str`` / ``_fast_yaml_flow_list`` /
``fast_yaml_dump``), the dtype mappers (``getRealDataTypeA`` /
``getRealDataTypeB``), and the ``LibraryLogic`` NamedTuple. All Tier A: pure
functions, snapshot the returned string / value / dict.
"""

import io

import pytest
from rocisa.enum import DataTypeEnum

import Tensile.LibraryIO as L

pytestmark = pytest.mark.unit


# ===========================================================================
# _fast_yaml_scalar — one snapshot per Python type branch
# ===========================================================================

@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        1,
        -42,
        3.5,
        -0.0,
        float("inf"),
        "plain",
        "",
        [1, 2, 3],
        [],
        {"a": 1},  # dict falls through to repr() — documented limitation
    ],
    ids=[
        "none", "true", "false", "int0", "int1", "negint", "float", "negzero",
        "inf", "str", "emptystr", "list", "emptylist", "dict_fallthrough",
    ],
)
def test_fast_yaml_scalar(value, snapshot):
    assert L._fast_yaml_scalar(value) == snapshot


# ===========================================================================
# _fast_yaml_str — quoting edge cases
# ===========================================================================

@pytest.mark.parametrize(
    "s",
    [
        "hello",
        "",                 # empty -> quoted
        "true",             # bool keyword -> quoted
        "false",
        "yes",
        "null",             # null keyword -> quoted
        "~",
        "it's",             # embedded single quote -> escaped+quoted
        "- dash",           # special start char
        " leading",         # leading space
        "trailing ",        # trailing space
        "key: value",       # ': ' -> quoted
        "trailing colon:",  # endswith ':' -> quoted
        "has #comment",     # ' #' -> quoted
        "line\nbreak",      # newline -> quoted
        "123",              # looks like number -> quoted
        "+1.5",             # signed number-like -> quoted
        "-x",               # '-' start but not a number -> bare
        "1abc",             # digit start, not float -> bare
        ".name",            # '.' start, not float
        "1.2.3",            # digit start, float() fails -> bare
    ],
    ids=[
        "plain", "empty", "true", "false", "yes", "null", "tilde", "apostrophe",
        "dashstart", "leadingspace", "trailingspace", "colonspace", "endcolon",
        "hashcomment", "newline", "numberlike", "signedfloat", "dash_notnum",
        "digit_notfloat", "dot_notfloat", "dotted_notfloat",
    ],
)
def test_fast_yaml_str(s, snapshot):
    assert L._fast_yaml_str(s) == snapshot


# ===========================================================================
# _fast_yaml_flow_list
# ===========================================================================

@pytest.mark.parametrize(
    "lst",
    [[], [1], [1, 2, 3], ["a", "b"], [True, None, 1.5], [[1, 2], [3]]],
    ids=["empty", "single", "ints", "strs", "mixed", "nested"],
)
def test_fast_yaml_flow_list(lst, snapshot):
    assert L._fast_yaml_flow_list(lst) == snapshot


# ===========================================================================
# fast_yaml_dump — full emitter over a list of solution-shaped dicts
# ===========================================================================

def _dump(states):
    buf = io.StringIO()
    L.fast_yaml_dump(states, buf)
    return buf.getvalue()


def test_fast_yaml_dump_flat(snapshot):
    states = [{"B": 2, "A": 1, "Name": "sol0"}]  # keys are sorted on output
    assert _dump(states) == snapshot


def test_fast_yaml_dump_nested_dict(snapshot):
    # One level of dict nesting is supported (keys sorted, 4-space indent).
    states = [{"ProblemType": {"OperationType": "GEMM", "DataType": 4}, "Idx": 0}]
    assert _dump(states) == snapshot


def test_fast_yaml_dump_multiple_solutions(snapshot):
    states = [{"A": 1}, {"A": 2, "B": [1, 2]}]
    assert _dump(states) == snapshot


def test_fast_yaml_dump_empty(snapshot):
    assert _dump([]) == snapshot


# ===========================================================================
# getRealDataTypeA / getRealDataTypeB — pure dtype mapping
# ===========================================================================

_DT_CASES = {
    "Float8BFloat8": DataTypeEnum.Float8BFloat8.value,
    "BFloat8Float8": DataTypeEnum.BFloat8Float8.value,
    "Float8BFloat8_fnuz": DataTypeEnum.Float8BFloat8_fnuz.value,
    "BFloat8Float8_fnuz": DataTypeEnum.BFloat8Float8_fnuz.value,
    "Half_passthrough": DataTypeEnum.Half.value,
}


@pytest.mark.parametrize("name", list(_DT_CASES), ids=list(_DT_CASES))
def test_get_real_data_type_a(name, snapshot):
    assert L.getRealDataTypeA(_DT_CASES[name]) == snapshot


@pytest.mark.parametrize("name", list(_DT_CASES), ids=list(_DT_CASES))
def test_get_real_data_type_b(name, snapshot):
    assert L.getRealDataTypeB(_DT_CASES[name]) == snapshot


# ===========================================================================
# LibraryLogic NamedTuple
# ===========================================================================

def test_library_logic_defaults(snapshot):
    # typeMismatches defaults to {}; construct with the 6 required fields.
    lg = L.LibraryLogic(
        schedule="aldebaran",
        architecture="gfx942",
        problemType="<ProblemType>",
        solutions=["s0", "s1"],
        exactLogic=[["k", "v"]],
        library="<MasterSolutionLibrary>",
    )
    assert lg._asdict() == snapshot


def test_library_logic_with_mismatches(snapshot):
    lg = L.LibraryLogic(
        "sched", "gfx950", "<PT>", [], None, "<lib>",
        {"DataType": ("bf16", "f16")},
    )
    assert lg._asdict() == snapshot
