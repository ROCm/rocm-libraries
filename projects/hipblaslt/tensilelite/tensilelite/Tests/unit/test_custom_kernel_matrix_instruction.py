# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Custom kernels skip matrix-instruction validation.

Prebuilt assembly does not carry EnableMatrixInstruction or the derived
MI keys. validateMIParameters returns before those checks, so an empty
ISA map must not be consulted.
"""

import pytest

from tensilelite.SolutionStructs.Validators.MatrixInstruction import validateMIParameters

pytestmark = pytest.mark.unit


def test_custom_kernel_skips_matrix_instruction_validation():
    solution = {"CustomKernel": {"name": "rocroller"}}
    assert validateMIParameters(solution, {}) is True
