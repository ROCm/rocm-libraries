# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""deepcopy of a DataType re-shares the class-level properties row."""

import copy

import pytest

from Tensile.Common.DataType import DataType

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", range(len(DataType.properties)))
def test_deepcopy_shares_properties_row(value):
    original = DataType(value)
    copied = copy.deepcopy(original)

    assert copied is not original
    assert copied == original
    assert copied.properties is DataType.properties[value]


def test_deepcopy_keeps_aliasing_within_one_copy():
    shared = DataType("h")
    copied = copy.deepcopy({"A": shared, "B": shared, "C": DataType("s")})

    assert copied["A"] is copied["B"]
    assert copied["A"] is not shared
    assert copied["A"].properties is DataType.properties[shared.value]
    assert copied["C"].properties is DataType.properties[DataType("s").value]
