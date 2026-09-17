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

"""Characterization tests for the module-level helpers of
``Tensile.SolutionStructs.Problem`` (the ProblemType slice): the mix-type dtype
mappers ``getRealDataTypeA`` / ``getRealDataTypeB``, the in-place enum converter
``problemTypeToEnum`` (incl. the ``DataTypeMetadata`` / ``DataTypeMXSA`` /
``DataTypeMXSB`` present/absent branches), the type checker
``validateProblemTypeParameterTypes`` (which writes into Solution's shared
collector), ``getBiasDataTypeListDefault``, and the ``_defaultProblemType``
registry + its derived ``_expectedProblemTypeParamTypes``.
"""

import pytest
from rocisa.enum import DataTypeEnum

from Tensile.Activation import ActivationType
from Tensile.Common.DataType import DataType
import Tensile.SolutionStructs.Problem as P
from Tensile.SolutionStructs.Problem import (
    getRealDataTypeA,
    getRealDataTypeB,
    problemTypeToEnum,
    validateProblemTypeParameterTypes,
    getBiasDataTypeListDefault,
    ProblemType,
    _defaultProblemType,
    _expectedProblemTypeParamTypes,
)

pytestmark = pytest.mark.unit


def _render(v):
    """Deterministic rendering for snapshots: enums -> name, DataType -> name."""
    if isinstance(v, DataType):
        return f"<DataType {v.toName()}>"
    if isinstance(v, DataTypeEnum):
        return f"<DataTypeEnum {v.name}>"
    if isinstance(v, list):
        return [_render(x) for x in v]
    return v


# ===========================================================================
# getRealDataTypeA / getRealDataTypeB — mix-type mapping
# ===========================================================================

_DT_CASES = [
    "Float8BFloat8",
    "BFloat8Float8",
    "Float8BFloat8_fnuz",
    "BFloat8Float8_fnuz",
    "Float",   # passthrough (else branch)
    "Half",    # passthrough
]


@pytest.mark.parametrize("name", _DT_CASES, ids=_DT_CASES)
def test_get_real_data_type_a(name, snapshot):
    dt = DataType(getattr(DataTypeEnum, name))
    assert getRealDataTypeA(dt).toName() == snapshot


@pytest.mark.parametrize("name", _DT_CASES, ids=_DT_CASES)
def test_get_real_data_type_b(name, snapshot):
    dt = DataType(getattr(DataTypeEnum, name))
    assert getRealDataTypeB(dt).toName() == snapshot


# ===========================================================================
# problemTypeToEnum — in-place conversion of DataType-valued fields -> .value
# ===========================================================================

def _enum_input_base():
    s = DataType("s")
    return {
        "DataType": DataType("s"),
        "MacDataTypeA": DataType("s"),
        "MacDataTypeB": DataType("s"),
        "DataTypeA": DataType("s"),
        "DataTypeB": DataType("s"),
        "DataTypeE": DataType("s"),
        "DataTypeAmaxD": DataType("s"),
        "DestDataType": DataType("s"),
        "ComputeDataType": DataType("s"),
        "BiasDataTypeList": [DataType("s"), DataType("h")],
        "ActivationComputeDataType": DataType("s"),
        "ActivationType": ActivationType("none"),
        "F32XdlMathOp": DataType("s"),
    }


def test_problem_type_to_enum_with_mx_present(snapshot):
    # MXSA/MXSB present (-> .value), DataTypeMetadata absent (the False arm).
    d = _enum_input_base()
    d["DataTypeMXSA"] = DataType(DataTypeEnum.E8)
    d["DataTypeMXSB"] = DataType(DataTypeEnum.E8)
    problemTypeToEnum(d)
    assert {k: _render(v) for k, v in sorted(d.items())} == snapshot


def test_problem_type_to_enum_with_metadata_and_no_mx(snapshot):
    # DataTypeMetadata present (-> .value); MXSA/MXSB absent (the else arms that
    # default them to DataTypeEnum.E8 — note: set to the *enum member*).
    d = _enum_input_base()
    d["DataTypeMetadata"] = DataType("I8")
    problemTypeToEnum(d)
    assert {k: _render(v) for k, v in sorted(d.items())} == snapshot


# ===========================================================================
# validateProblemTypeParameterTypes — clean + bool/int mismatch
# ===========================================================================

def _run_validate_capture(state, srcFile="fixture.yaml"):
    """Run the validator against an isolated copy of Solution's shared collector
    and return the delta it produced (then restore the collector)."""
    from Tensile.SolutionStructs.Solution import _typeMismatchCollector

    saved = {k: dict(v) for k, v in _typeMismatchCollector.items()}
    _typeMismatchCollector.clear()
    try:
        validateProblemTypeParameterTypes(state, srcFile=srcFile, raiseOnMismatch=False)
        delta = {
            "|".join(k): {
                "count": v["count"],
                "values": sorted(v["values"]),
                "files": sorted(v["files"]),
            }
            for k, v in _typeMismatchCollector.items()
        }
    finally:
        _typeMismatchCollector.clear()
        _typeMismatchCollector.update(saved)
    return delta


def test_validate_clean_state_no_mismatch(snapshot):
    # Feeding the defaults back in -> every type matches -> empty delta.
    delta = _run_validate_capture(dict(_defaultProblemType))
    assert delta == snapshot


def test_validate_bool_where_int_mismatch(snapshot):
    # UseBias defaults to int(0); a bool there is the classic mismatch.
    state = dict(_defaultProblemType)
    state["UseBias"] = True
    delta = _run_validate_capture(state)
    assert delta == snapshot


def test_validate_unknown_key_skipped(snapshot):
    # A key not in _expectedProblemTypeParamTypes is ignored (continue arm).
    delta = _run_validate_capture({"TotallyMadeUpKey": 123})
    assert delta == snapshot


def test_validate_mismatch_no_srcfile(snapshot):
    # srcFile="" -> the `if srcFile:` false arm (no file recorded in the entry).
    state = dict(_defaultProblemType)
    state["UseBias"] = True
    delta = _run_validate_capture(state, srcFile="")
    assert delta == snapshot


# ===========================================================================
# getBiasDataTypeListDefault — filters dtypes with numBytes > 1
# ===========================================================================

def test_get_bias_data_type_list_default_single(snapshot):
    pt = ProblemType({"DataType": 0}, False)  # all S -> 4 bytes
    result = [d.toName() for d in getBiasDataTypeListDefault(pt)]
    assert result == snapshot


def test_get_bias_data_type_list_default_int8(snapshot):
    # I8 in/out (1 byte) is filtered out; compute (I/Int32, 4 bytes) stays.
    cfg = {"DataType": "I8", "DestDataType": "I8", "ComputeDataType": "I"}
    pt = ProblemType(cfg, False)
    result = [d.toName() for d in getBiasDataTypeListDefault(pt)]
    assert result == snapshot


# ===========================================================================
# _defaultProblemType registry + derived expected-types map
# ===========================================================================

def test_default_problem_type_roster(snapshot):
    assert sorted(_defaultProblemType.keys()) == snapshot


def test_expected_param_types(snapshot):
    rendered = {
        k: sorted(t.__name__ for t in types)
        for k, types in sorted(_expectedProblemTypeParamTypes.items())
    }
    assert rendered == snapshot
