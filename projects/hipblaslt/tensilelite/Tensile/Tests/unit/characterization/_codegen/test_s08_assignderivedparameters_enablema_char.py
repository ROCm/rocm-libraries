################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S08 -- assignDerivedParameters EnableMatrixInstruction reject cluster (gfx942).

Reachable-invalid (category A): the DGEMM (double) TN config forks two valid
f64 MFMA MatrixInstruction shapes that pass the earlier validateMIParameters
gate and reach the type/MI reject cluster inside
Tensile/SolutionStructs/Solution.py:assignDerivedParameters, where each fork
trips a distinct reject branch and early-returns. No valid solution survives.

Solution.py lines that fire during the rejected derivation (probe-confirmed):
  1968, 1970 : Variant MI [16,16,4,...,3,1] -> waves=3 -> MIWaveGroup=[3,1];
               the non-power-of-two MIWaveGroup guard in the LraTileAssignment
               vectorStaticRemainder path rejects.
  2015, 2016 : Variant MI [4,4,4,4] + ComputeDataType double + ISA (9,4,x)
               (!= IsaVersion(9,0,10)) + ScheduleIterAlg==3 -> "[4,4,4,4] is
               disabled" reject.

Both forks reject during derivation, so ``len(solutions_from_config(...)) == 0``
pins the reachable-invalid reject (category A). CPU-only; no GPU, no compile.
"""

import os

import pytest

from config_harness import solutions_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "s08_assignderivedparameters_enablema.yaml",
)


def test_s08_assignderivedparameters_enablema_derives_reject():
    """All forks reject during assignDerivedParameters -> 0 valid solutions.

    Both MI forks reach the Solution.py type/MI reject cluster (lines
    1968/1970 for the non-power-of-two MIWaveGroup fork and 2015/2016 for the
    [4,4,4,4] disabled fork) and early-return, so no valid solution survives.
    """
    sols = solutions_from_config(_CONFIG, arch=_ARCH)
    assert len(sols) == 0, (
        f"Expected 0 surviving solutions (all forks reachable-invalid), "
        f"got {len(sols)}"
    )


def test_s08_assignderivedparameters_enablema_golden(snapshot):
    """S08 golden: surviving-solution count pins the reachable-invalid reject."""
    assert len(solutions_from_config(_CONFIG, arch=_ARCH)) == snapshot
