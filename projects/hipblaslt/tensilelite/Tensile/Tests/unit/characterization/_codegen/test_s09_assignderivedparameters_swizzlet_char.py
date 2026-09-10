################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S09 -- assignDerivedParameters SwizzleTensorA/B reject cluster (gfx942).

Reachable-invalid (category A): the half TN config sets SwizzleTensorA and
SwizzleTensorB and forks the DirectToVgprA/B permutations. Derivation runs
Tensile/SolutionStructs/Solution.py:assignDerivedParameters, reaching the
SwizzleTensor derivation + reject block, where each fork trips a distinct reject
branch and early-returns. No valid solution survives.

Solution.py lines that fire during the rejected derivation (probe-confirmed):
  3742       SwizzleTensor pack-K / GlobalReadVectorWidth derivation arm.
  3789       SwizzleTensor GRVW derivation follow-on.
  3856       SwizzleTensorA requires DirectToVgprA -> reject (DirectToVgprA=0).
  3861-3863  SwizzleTensorB requires DirectToVgprB -> reject (DirectToVgprB=0).
  3867       SwizzleTensorB TN-only combined transpose check (runs whenever
             SwizzleTensorB is set).

Every fork rejects during derivation, so
``len(solutions_from_config(...)) == 0`` pins the reachable-invalid reject
(category A). CPU-only; no GPU, no compile. pytestmark = pytest.mark.unit.
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
    "s09_assignderivedparameters_swizzlet.yaml",
)


def test_s09_assignderivedparameters_swizzlet_derives_reject():
    """All forks reject during assignDerivedParameters -> 0 valid solutions.

    The SwizzleTensorA/B derivation reaches the Solution.py swizzle block
    (lines 3742, 3789, 3856, 3861-3863, 3867); the DirectToVgprA/B forks each
    trip the SwizzleTensorA/B-requires-DirectToVgpr reject and early-return, so
    no valid solution survives.
    """
    sols = solutions_from_config(_CONFIG, arch=_ARCH)
    assert len(sols) == 0, (
        f"Expected 0 surviving solutions (all forks reachable-invalid), "
        f"got {len(sols)}"
    )


def test_s09_assignderivedparameters_swizzlet_golden(snapshot):
    """S09 golden: surviving-solution count pins the reachable-invalid reject."""
    assert len(solutions_from_config(_CONFIG, arch=_ARCH)) == snapshot
