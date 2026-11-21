# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MAC data types of mixed MX problem types.

DataType seeds both MacDataTypeA and MacDataTypeB, so a mixed problem such as
DataType F4 with DataTypeB F8 used to get MacDataTypeB F4. The kernel then
treated B as converted from F8 to F4, a conversion it does not implement, and
loaded only B's scalar scale. An operand whose storage and MAC types are both
floats of 8 bits or fewer now takes its storage type as its MAC type.
"""

import pytest

from tensilelite.Common.DataType import DataType
from tensilelite.SolutionStructs.Problem import ProblemType, _isFloatWithAtMostBits

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "typeChar, nBits, expected",
    [
        ("F8", 8, True), ("B8", 8, True), ("F8N", 8, True), ("F6", 8, True),
        ("B6", 8, True), ("F4", 8, True), ("H", 8, False), ("I8", 8, False),
        ("I4", 8, False), ("H", 16, True), ("B", 16, True), ("S", 16, False),
    ],
)
def test_float_with_at_most_bits(typeChar, nBits, expected):
    assert _isFloatWithAtMostBits(DataType(typeChar), nBits) == expected


def _problemType(**overrides):
    config = {
        "OperationType": "GEMM",
        "DestDataType": "s",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
    }
    config.update(overrides)
    return ProblemType(config, False)


def _chars(pt, *keys):
    return tuple(pt[k].toChar() for k in keys)


@pytest.mark.parametrize(
    "config, expectedMac",
    [
        ({"DataType": "F4", "DataTypeB": "F8"}, ("F4", "F8")),
        ({"DataType": "F4", "DataTypeA": "F4", "DataTypeB": "F8"}, ("F4", "F8")),
        ({"DataType": "F4", "DataTypeA": "F8"}, ("F8", "F4")),
        ({"DataType": "F8", "DataTypeA": "F8", "DataTypeB": "F4"}, ("F8", "F4")),
        ({"DataType": "B6", "DataTypeA": "B6", "DataTypeB": "F8"}, ("B6", "F8")),
        ({"DataType": "F6", "DataTypeA": "F6", "DataTypeB": "B6"}, ("F6", "B6")),
    ],
    ids=["F4_B_F8", "F4_A_F4_B_F8", "F4_A_F8", "F8_A_F8_B_F4", "B6_A_B6_B_F8", "F6_A_F6_B_B6"],
)
def test_mixed_mx_mac_follows_operand_type(config, expectedMac):
    pt = _problemType(**config)
    assert _chars(pt, "MacDataTypeA", "MacDataTypeB") == expectedMac
    assert _chars(pt, "DataTypeA", "DataTypeB") == expectedMac
    assert pt["isMixMode"]


def test_mixed_mx_name_has_no_conversion_suffix():
    implicit = _problemType(DataType="F4", DataTypeB="F8")
    explicit = _problemType(DataType="F4", DataTypeB="F8", MacDataTypeB="F8")
    assert str(implicit) == str(explicit)
    assert str(implicit) == "Cijk_Alik_Bljk_F4F8SS_BH_UserArgs"


@pytest.mark.parametrize(
    "config, expectedMac",
    [
        ({"DataType": "h", "DataTypeB": "F8", "DestDataType": "h"}, ("H", "H")),
        ({"DataType": "h", "DataTypeA": "F8N", "DestDataType": "h"}, ("H", "H")),
        ({"DataType": "F8", "DataTypeA": "h", "DestDataType": "h"}, ("F8", "F8")),
        ({"DataType": "F8B8", "DataTypeA": "h", "DataTypeB": "h", "DestDataType": "h"}, ("F8", "B8")),
    ],
    ids=["H_B_F8", "H_A_F8N", "F8_A_H", "F8B8_AB_H"],
)
def test_input_conversion_keeps_data_type_mac(config, expectedMac):
    pt = _problemType(**config)
    assert _chars(pt, "MacDataTypeA", "MacDataTypeB") == expectedMac


def test_explicit_unsupported_mx_conversion_raises():
    with pytest.raises(Exception, match=r"DataTypeB \(F8\) -> MacDataTypeB \(F4\)"):
        _problemType(DataType="F4", DataTypeB="F8", MacDataTypeB="F4")
