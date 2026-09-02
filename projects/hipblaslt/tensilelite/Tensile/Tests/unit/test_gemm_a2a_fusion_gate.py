# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.TensileCreateLibrary.Run import _includeGemmA2AFusionProblemType


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "problem_type,enabled,expected",
    [
        ({"FusedGemmA2A": False}, False, True),
        ({"FusedGemmA2A": True}, False, False),
        ({"FusedGemmA2A": True}, True, True),
    ],
)
def test_gemm_a2a_logic_requires_explicit_enable(problem_type, enabled, expected):
    assert _includeGemmA2AFusionProblemType(problem_type, enabled) is expected
