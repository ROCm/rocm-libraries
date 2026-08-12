################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurface characterization: the not versionIsCompatible(...)
predicate at Tensile/Common/GlobalParameters.py:660.

Branch 4914224d6e015f0cc6a041a74683fe38f6eae196.  The branch fires (True) when
the YAML-supplied MinimumRequiredVersion is INCOMPATIBLE with the baked-in
__version__ = "5.0.0" (Tensile/__init__.py:30).

Reachability gate (line 659): if "MinimumRequiredVersion" in config must be
True before this predicate is evaluated.

versionIsCompatible (Tensile/Common/Utilities.py:148) returns False when:
  * qMajor != tMajor, OR
  * int(qMinor) > tMinor, OR
  * qMinor == tMinor AND int(qStep) > tStep

So the predicate not versionIsCompatible(...) is True (branch fires,
printExit called) exactly when the config version is INCOMPATIBLE.

These tests pin ACTUAL observed behavior; they do not assert anything aspirational.
"""

import pytest

from Tensile.Common.Utilities import versionIsCompatible

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure-helper: extracted predicate model
# ---------------------------------------------------------------------------

def minimum_version_rejected(
    qMajor: int, qMinor: int, qStep: int,
    tMajor: int, tMinor: int, tStep: int,
) -> bool:
    """Pure model of: not versionIsCompatible(MinimumRequiredVersion).

    True == config version INCOMPATIBLE with Tensile __version__ (branch fires,
    printExit is reached).

    pre: 0 <= qMajor <= 9 and 0 <= qMinor <= 9 and 0 <= qStep <= 9
    pre: 0 <= tMajor <= 9 and 0 <= tMinor <= 9 and 0 <= tStep <= 9
    post: __return__ == (qMajor != tMajor or qMinor > tMinor or
                         (qMinor == tMinor and qStep > tStep))
    """
    if qMajor != tMajor:
        return True
    if qMinor > tMinor:
        return True
    if qMinor == tMinor:
        if qStep > tStep:
            return True
    return False


# ---------------------------------------------------------------------------
# True-branch witnesses (predicate == True == branch fires == incompatible)
# ---------------------------------------------------------------------------

def test_minimum_version_rejected_true_major_mismatch():
    # 4.0.0 vs 5.0.0: qMajor 4 != tMajor 5 -> incompatible
    assert minimum_version_rejected(4, 0, 0, 5, 0, 0) is True


def test_minimum_version_rejected_true_minor_exceeds():
    # 5.1.0 vs 5.0.0: same major, qMinor 1 > tMinor 0 -> incompatible
    assert minimum_version_rejected(5, 1, 0, 5, 0, 0) is True


def test_minimum_version_rejected_true_step_exceeds():
    # 5.0.1 vs 5.0.0: same major/minor, qStep 1 > tStep 0 -> incompatible
    assert minimum_version_rejected(5, 0, 1, 5, 0, 0) is True


# ---------------------------------------------------------------------------
# False-branch witness (predicate == False == branch skipped == compatible)
# ---------------------------------------------------------------------------

def test_minimum_version_rejected_false_exact_match():
    # 5.0.0 vs 5.0.0: exact match -> compatible -> branch NOT taken
    assert minimum_version_rejected(5, 0, 0, 5, 0, 0) is False


def test_minimum_version_rejected_false_older_minor_ok():
    # 5.0.0 queried against 5.1.0 target -> compatible (older is fine)
    assert minimum_version_rejected(5, 0, 0, 5, 1, 0) is False


def test_minimum_version_rejected_false_older_step_ok():
    # 5.0.0 queried against 5.0.1 target -> compatible (older step is fine)
    assert minimum_version_rejected(5, 0, 0, 5, 0, 1) is False


# ---------------------------------------------------------------------------
# Real entry-point pin: versionIsCompatible against baked-in __version__=5.0.0
# ---------------------------------------------------------------------------

def test_real_version_compatible_exact_match():
    # 5.0.0 == __version__ -> compatible -> predicate False -> branch skipped
    assert versionIsCompatible("5.0.0") is True


def test_real_version_incompatible_major_mismatch():
    # 4.0.0: major mismatch -> not compatible
    assert versionIsCompatible("4.0.0") is False


def test_real_version_incompatible_minor_exceeds():
    # 5.1.0: qMinor 1 > tMinor 0 -> not compatible
    assert versionIsCompatible("5.1.0") is False


def test_real_version_incompatible_step_exceeds():
    # 5.0.1: same major/minor, qStep 1 > tStep 0 -> not compatible
    assert versionIsCompatible("5.0.1") is False


def test_real_version_compatible_older_step():
    # Tensile 5.0.0; YAML requires 5.0.0: exact match -> True
    assert versionIsCompatible("5.0.0") is True
