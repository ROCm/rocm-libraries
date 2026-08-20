################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S08 -- calLRVW HasWMMA_V3 explicit-LRVW reject cluster (gfx1250).

Reachable-invalid (category A): the FP16 NN config forks a single gfx1250
HasWMMA_V3 solution with TransposeLDS=1 and explicit
LocalReadVectorWidthA/B=4 (non-max; maxLRVW=8 for FP16 under the 16-byte
MAX_NUM_DS_LOAD_BYTES cap). Derivation runs assignDerivedParameters, which
calls calLRVW in Tensile/SolutionStructs/Solution.py; the explicit-LRVW !=
maxLRVW guard rejects. No valid solution survives.

Solution.py lines that fire during the rejected derivation (probe-confirmed):
  3523 : HasWMMA_V3 + LocalReadVectorWidthA (4) != maxLRVWA (8) + TransposeLDS
         -> reject "gfx1250 requires lrvwA == {maxLRVWA}".
  3558 : same B-side check (the A reject does not return, so the B guard also
         runs and rejects).

The feature combination is rejected during derivation, so
``len(solutions_from_config(...)) == 0`` pins the reachable-invalid reject
(category A). CPU-only; no GPU, no compile.
"""

import os

import pytest

from config_harness import solutions_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "s08_depthuiteration_callrvw_lrvw_aut.yaml",
)


def test_s08_depthuiteration_callrvw_lrvw_aut_derives_reject():
    """calLRVW rejects during assignDerivedParameters -> 0 valid solutions.

    The gfx1250 HasWMMA_V3 fork reaches calLRVW in Solution.py, where the
    explicit-LRVW != maxLRVW guards (lines 3523 A-side, 3558 B-side) fire and
    reject, so no valid solution survives.
    """
    sols = solutions_from_config(_CONFIG, arch=_ARCH)
    assert len(sols) == 0, (
        f"Expected 0 surviving solutions (fork reachable-invalid), "
        f"got {len(sols)}"
    )


def test_s08_depthuiteration_callrvw_lrvw_aut_golden(snapshot):
    """S08 golden: surviving-solution count pins the reachable-invalid reject."""
    assert len(solutions_from_config(_CONFIG, arch=_ARCH)) == snapshot
