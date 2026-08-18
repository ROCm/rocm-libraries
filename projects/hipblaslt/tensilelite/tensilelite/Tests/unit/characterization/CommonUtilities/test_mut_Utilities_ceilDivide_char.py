################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.ceilDivide``.

``ceilDivide(numerator, denominator)`` computes ``ceil(numerator/denominator)``
via integer floor division of ``(numerator + denominator - 1) // denominator``.
It guards two error paths, each returning 0:
  - a negative ``numerator`` OR negative ``denominator`` raises ``ValueError``
    (caught) and prints a "negative register value" error,
  - a zero ``denominator`` reaches the division, raises ``ZeroDivisionError``
    (caught) and prints a "Divide by 0" error.

These tests pin the ACTUAL current behavior (return value and which guard
branch is taken) so they pass on clean source and fail under the surviving
boundary/arithmetic mutants. The exact wording of the error strings is NOT
pinned here (those mutants are pure logging noise handled via pragma); instead
the tests assert which BRANCH executed by checking for the presence/absence of
the distinguishing path and the empty-output happy path.
"""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")
ceilDivide = U.ceilDivide

pytestmark = pytest.mark.unit


def test_negative_numerator_returns_zero():
    # Kills mutmut_1: `numerator < 0 or denominator < 0` -> `... and ...`.
    # With numerator=-1, denominator=1 only ONE operand is negative.
    # Original: the OR triggers the ValueError guard -> returns 0.
    # Mutant (AND): guard not taken -> computes (-1+1-1)//1 == -1.
    assert ceilDivide(-1, 1) == 0


def test_zero_numerator_does_not_take_negative_guard(capsys):
    # Kills mutmut_2 (`numerator < 0` -> `numerator <= 0`) and
    # mutmut_3 (`numerator < 0` -> `numerator < 1`).
    # For numerator=0 the original does NOT enter the negative guard: it
    # computes (0+1-1)//1 == 0 and prints nothing. Both mutants would treat 0
    # as negative, enter the guard, and print the negative-register error.
    result = ceilDivide(0, 1)
    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == ""


def test_zero_denominator_takes_divide_by_zero_branch(capsys):
    # Kills mutmut_4 (`denominator < 0` -> `denominator <= 0`) and
    # mutmut_5 (`denominator < 0` -> `denominator < 1`).
    # For denominator=0 the original does NOT enter the negative guard; it
    # reaches the division, hits ZeroDivisionError, and prints a divide-by-0
    # message (NOT the negative-register message). Both mutants would treat 0
    # as negative and instead print the negative-register error.
    result = ceilDivide(1, 0)
    captured = capsys.readouterr()
    assert result == 0
    assert "negative" not in captured.out


def test_floor_division_not_float_division():
    # Kills mutmut_13: `(...) // denominator` -> `(...) / denominator`.
    # Float division of large ints loses precision; int() then truncates to a
    # different value. Original (true integer floor) yields the exact quotient.
    assert ceilDivide(99999999999999999, 3) == 33333333333333333


def test_minus_one_offset_in_ceiling_formula():
    # Kills mutmut_14: `(numerator+denominator-1)` -> `(numerator+denominator+1)`.
    # ceilDivide(1, 1): original (1+1-1)//1 == 1; mutant (1+1+1)//1 == 3.
    assert ceilDivide(1, 1) == 1
    # An additional exact-multiple case to further constrain the formula:
    # ceilDivide(7, 3): original (7+3-1)//3 == 3; mutant (7+3+1)//3 == 3 would
    # actually also be 3, so the (1,1) case above is the discriminating one.
    assert ceilDivide(7, 3) == 3
