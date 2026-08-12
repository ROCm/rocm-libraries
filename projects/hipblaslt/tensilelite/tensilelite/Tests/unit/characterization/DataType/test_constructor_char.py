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

"""Characterization tests for ``Tensile.Common.DataType.DataType`` construction
and the value-converter / string-builder surface: ``__init__`` (the four
accepted input forms plus the invalid-type ``RuntimeError``), the
``to*`` converters, ``toDevice`` (HIP path + the non-HIP ``assert``), and
``zeroString``. All pure: snapshot the returned value / raised error.
"""

import pytest
from rocisa.enum import DataTypeEnum

from Tensile.Common.DataType import DataType

pytestmark = pytest.mark.unit


# The table is the single source of truth for the dtype set; iterate it so the
# suite tracks any future additions automatically.
_ENUMS = [p["enum"] for p in DataType.properties]
_NAMES = [e.name for e in _ENUMS]


# ===========================================================================
# __init__ — the four accepted input forms + the invalid-type else branch
# ===========================================================================

def test_init_from_enum(snapshot):
    # Every enum form maps to the same row index its value names.
    result = {e.name: DataType(e).value for e in _ENUMS}
    assert result == snapshot


def test_init_from_int(snapshot):
    # int is stored verbatim as the row index.
    result = {i: DataType(i).value for i in range(len(_ENUMS))}
    assert result == snapshot


def test_init_from_str_by_name(snapshot):
    # str form looks up the (lower-cased) enum *name* in DataType.lookup.
    result = {name: DataType(name.lower()).value for name in _NAMES}
    assert result == snapshot


def test_init_from_str_by_char(snapshot):
    # str form also resolves the (lower-cased) 'char' abbreviation.
    result = {p["char"]: DataType(p["char"].lower()).value for p in DataType.properties}
    assert result == snapshot


def test_init_from_datatype_identity(snapshot):
    # Wrapping an existing DataType copies its .value.
    src = DataType(DataTypeEnum.Half)
    assert DataType(src).value == snapshot


@pytest.mark.parametrize(
    "bad",
    [3.5, None, (1,), [0], 2.0],
    ids=["float", "none", "tuple", "list", "floatint"],
)
def test_init_invalid_type_raises(bad, snapshot):
    with pytest.raises(RuntimeError) as excinfo:
        DataType(bad)
    assert str(excinfo.value) == snapshot


def test_init_unknown_str_key_raises(snapshot):
    # A string not in the lookup table raises KeyError from the dict access.
    with pytest.raises(KeyError) as excinfo:
        DataType("not_a_real_dtype")
    assert repr(excinfo.value) == snapshot


# ===========================================================================
# to* converters — one snapshot per method, across every dtype
# ===========================================================================

def test_to_char(snapshot):
    assert {e.name: DataType(e).toChar() for e in _ENUMS} == snapshot


def test_to_name(snapshot):
    assert {e.name: DataType(e).toName() for e in _ENUMS} == snapshot


def test_to_name_abbrev(snapshot):
    assert {e.name: DataType(e).toNameAbbrev() for e in _ENUMS} == snapshot


def test_to_enum(snapshot):
    assert {e.name: DataType(e).toEnum().name for e in _ENUMS} == snapshot


# ===========================================================================
# toDevice — HIP path returns the 'hip' name; any other language asserts
# ===========================================================================

def test_to_device_hip(snapshot):
    assert {e.name: DataType(e).toDevice("HIP") for e in _ENUMS} == snapshot


@pytest.mark.parametrize("language", ["HLSL", "OCL", "", "hip"], ids=["hlsl", "ocl", "empty", "lowerhip"])
def test_to_device_non_hip_asserts(language):
    # The else branch is a bare `assert 0` -> AssertionError, no message.
    with pytest.raises(AssertionError):
        DataType(DataTypeEnum.Float).toDevice(language)


# ===========================================================================
# zeroString — vectorWidth 1 (no width suffix) and >1 (width suffix)
# ===========================================================================

@pytest.mark.parametrize("vector_width", [1, 2, 4], ids=["vw1", "vw2", "vw4"])
def test_zero_string(vector_width, snapshot):
    result = {e.name: DataType(e).zeroString("HIP", vector_width) for e in _ENUMS}
    assert result == snapshot


def test_zero_string_non_hip_asserts():
    # zeroString delegates language handling to toDevice -> assert for non-HIP.
    with pytest.raises(AssertionError):
        DataType(DataTypeEnum.Float).zeroString("HLSL", 1)
