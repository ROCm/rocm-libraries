# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from Tensile.Common.DataType import DataType
from Tensile.KernelWriter import _registerCount


def test_register_pool_counts_are_converted_to_integers():
    accumulator_count = 8 * 8 * DataType("half").numRegisters()
    register_count = _registerCount(accumulator_count)

    assert accumulator_count == 32.0
    assert register_count == 32
    assert isinstance(register_count, int)
