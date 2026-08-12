################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.state_key_ordering``.

``state_key_ordering`` is a class decorator that installs ``__lt__`` and
``__eq__`` based on a tuple of the attributes named in ``cls.StateKeys`` and
then applies ``functools.total_ordering``.

These tests pin the ACTUAL current behavior so they pass on clean source and
fail under the surviving mutant 7, which changes the strict ``<`` in ``__lt__``
to ``<=``.
"""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")
state_key_ordering = U.state_key_ordering

pytestmark = pytest.mark.unit


def _make_ordered_cls():
    @state_key_ordering
    class Keyed:
        StateKeys = ["a", "b"]

        def __init__(self, a, b):
            self.a = a
            self.b = b

    return Keyed


def test_lt_is_strict_for_equal_objects():
    # Kills mutmut_7: `tup(a) < tup(b)` -> `tup(a) <= tup(b)`.
    # Two instances with identical StateKey values compare equal, so a strict
    # less-than MUST be False. Under the mutant `<=` would return True.
    Keyed = _make_ordered_cls()
    x = Keyed(1, 2)
    y = Keyed(1, 2)
    assert (x < y) is False


def test_lt_true_for_strictly_smaller_object():
    # Sanity-pins the surviving (non-equal) branch: a genuinely smaller object
    # is still less-than under both original and mutant, confirming the test
    # above isolates the equal-tuple distinguishing case.
    Keyed = _make_ordered_cls()
    smaller = Keyed(1, 2)
    larger = Keyed(1, 3)
    assert (smaller < larger) is True
