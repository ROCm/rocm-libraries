################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.versionIsCompatible``.

The function compares a query version string ``"major.minor.step"`` against the
package ``__version__`` (currently ``5.0.0``):
  - major must match exactly,
  - a higher query minor is incompatible,
  - when minors are equal, a higher query step is incompatible.

These tests pin the ACTUAL current behavior so they pass on clean source and
fail under the surviving mutants 12, 13 and 17.
"""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")
versionIsCompatible = U.versionIsCompatible

pytestmark = pytest.mark.unit


def test_higher_query_minor_is_incompatible():
    # Kills mutmut_12: `if int(qMinor) > int(tMinor): return False` -> return True.
    # __version__ is 5.0.0, so query minor 1 > target minor 0 with matching major.
    # Original returns False; mutant would return True.
    assert versionIsCompatible("5.1.0") is False


def test_equal_minor_branch_is_entered_for_step_check():
    # Kills mutmut_13: `if qMinor == tMinor:` -> `if qMinor != tMinor:`.
    # With equal minors ("5.0.x") the step-check branch MUST be entered.
    # qStep 1 > target step 0 -> original returns False. Under the mutant the
    # branch is skipped (0 != 0 is False) and it would return True.
    assert versionIsCompatible("5.0.1") is False


def test_higher_step_when_minor_equal_is_incompatible():
    # Kills mutmut_17: inner `return False` -> `return True` when qStep > tStep.
    # Equal minor (0 == 0), query step 1 > target step 0.
    # Original returns False; mutant would return True.
    assert versionIsCompatible("5.0.1") is False
