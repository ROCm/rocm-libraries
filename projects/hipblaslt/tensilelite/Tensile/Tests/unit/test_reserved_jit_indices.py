# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

pytestmark = pytest.mark.unit

from Tensile.TensileCreateLibrary.Run import RESERVED_JIT_INDEX, checkReservedJitIndices


def test_indices_below_the_range_are_accepted():
    checkReservedJitIndices(0)
    checkReservedJitIndices(RESERVED_JIT_INDEX)


def test_an_index_in_the_range_is_rejected():
    with pytest.raises(RuntimeError, match="reserves for JIT solutions"):
        checkReservedJitIndices(RESERVED_JIT_INDEX + 1)
