################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S09 -- assignDerivedParameters SwizzleTensorA/B reject cluster (gfx942).

Reachable-invalid (category A): the half TN config sets SwizzleTensorA and
SwizzleTensorB and forks the DirectToVgprA/B permutations. Derivation runs
tensilelite/SolutionStructs/Solution.py:assignDerivedParameters, reaching the
SwizzleTensor derivation + reject block, where each fork trips a distinct reject
branch and early-returns. No valid solution survives.

Solution.py lines that fire during the rejected derivation (probe-confirmed):
  3742       SwizzleTensor pack-K / GlobalReadVectorWidth derivation arm.
  3789       SwizzleTensor GRVW derivation follow-on.
  3867       SwizzleTensorB TN-only combined transpose check (runs whenever
             SwizzleTensorB is set).

Swizzle no longer requires DirectToVgpr: a fork without it is LDS-staged, which
is only enabled on gfx11, so the LDS-staging guard rejects it here (A is checked
first, so the both-off fork reports A only).

Every fork rejects during derivation, so
``len(solutions_from_config(...)) == 0`` pins the reachable-invalid reject
(category A). CPU-only; no GPU, no compile. pytestmark = pytest.mark.unit.
"""

import os

import pytest

from config_harness import assert_config_rejects

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


def test_s09_assignderivedparameters_swizzlet_rejects_with_reason(monkeypatch, capsys):
    """Every fork reports the expected tensor-swizzle rejection."""
    assert_config_rejects(
        _CONFIG,
        _ARCH,
        monkeypatch,
        capsys,
        {
            "reject: LDS-staged SwizzleTensorA is only validated on gfx11": 2,
            "reject: LDS-staged SwizzleTensorB is only validated on gfx11": 1,
            "reject: DirectToVgprA + DirectToVgprB disabled": 1,
        },
    )
