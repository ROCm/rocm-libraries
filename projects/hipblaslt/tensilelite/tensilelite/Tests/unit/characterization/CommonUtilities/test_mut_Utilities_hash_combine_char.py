################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.hash_combine``.

``hash_combine`` folds an iterable of integers into a single value::

    rv = next(it)
    for value in it:
        rv = (rv << shift) ^ value

where ``shift`` defaults to 1 but can be overridden via the ``shift`` keyword.
A single non-iterable positional argument is returned unchanged (the ``iter``
raises ``TypeError`` which is swallowed). An empty iterable returns the initial
``rv`` (currently ``0``) via the swallowed ``StopIteration``.

These tests pin the ACTUAL current behavior so they pass on clean source and
fail under the surviving mutants 3, 4, 6, 7, 8, 13 and 14.
"""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")
hash_combine = U.hash_combine

pytestmark = pytest.mark.unit


def test_shift_kwarg_is_honored():
    # Kills mutmut_3 ("shift" -> "XXshiftXX") and mutmut_4 ("shift" -> "SHIFT")
    # in the `if "shift" in kwargs` guard: under those mutants the guard never
    # fires, shift stays at the default 1, and the result would be (1<<1)^4 = 6.
    # Original honors shift=2: (1<<2)^4 = 0.
    assert hash_combine([1, 4], shift=2) == 0


def test_shift_kwarg_lookup_uses_correct_key():
    # Kills mutmut_7 (kwargs["XXshiftXX"]) and mutmut_8 (kwargs["SHIFT"]):
    # the guard fires (real "shift" key present) but the body looks up a
    # non-existent key, raising KeyError. Original returns (1<<2)^4 = 0.
    assert hash_combine([1, 4], shift=2) == 0


def test_shift_value_must_be_the_passed_integer():
    # Kills mutmut_6 (shift = None): with shift None the `rv << None`
    # raises TypeError, which is swallowed and `objs` ([1, 4]) is returned.
    # Original returns the integer 0, not the list.
    result = hash_combine([1, 4], shift=2)
    assert result == 0
    assert isinstance(result, int)
    assert result != [1, 4]


def test_default_shift_is_one():
    # Pins the default shift (1) folding behavior: (1<<1)^4 = 6.
    # Distinguishes the default path from the shift-kwarg path.
    assert hash_combine([1, 4]) == 6


def test_empty_iterable_returns_zero():
    # Kills mutmut_13 (rv = 0 -> rv = None) and mutmut_14 (rv = 0 -> rv = 1).
    # With an empty iterable, next(it) raises StopIteration (swallowed) and the
    # initial rv is returned unchanged. Original returns 0; mutants return
    # None / 1 respectively.
    result = hash_combine([])
    assert result == 0
    assert result is not None
