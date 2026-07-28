import importlib

import pytest

from Tensile.Common.DataType import DataType, DataTypeEnum

S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit

def _state(enum):
    return {"ProblemType": {"DataType": DataType(enum)}}

def test_single_dtype_supported_kills_and_of_single_double():

    dt = DataType(DataTypeEnum.Float)
    assert dt.isSingle() is True and dt.isDouble() is False and dt.isComplex() is False
    assert S.Solution.isDirectToVgprSupportDataType(_state(DataTypeEnum.Float)) is True

def test_double_dtype_supported_kills_and_of_double_complex():

    dt = DataType(DataTypeEnum.Double)
    assert dt.isDouble() is True and dt.isComplex() is False and dt.isSingle() is False
    assert S.Solution.isDirectToVgprSupportDataType(_state(DataTypeEnum.Double)) is True

def test_complex_dtype_supported():
    assert S.Solution.isDirectToVgprSupportDataType(_state(DataTypeEnum.ComplexFloat)) is True

def test_half_and_bf16_and_int8_and_fp8_supported():
    assert S.Solution.isDirectToVgprSupportDataType(_state(DataTypeEnum.Half)) is True
    assert S.Solution.isDirectToVgprSupportDataType(_state(DataTypeEnum.BFloat16)) is True
    assert S.Solution.isDirectToVgprSupportDataType(_state(DataTypeEnum.Int8)) is True
    assert S.Solution.isDirectToVgprSupportDataType(_state(DataTypeEnum.Float8)) is True

def test_unsupported_dtype_returns_false():

    assert S.Solution.isDirectToVgprSupportDataType(_state(DataTypeEnum.Int32)) is False
