################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-killing characterization tests for
``Tensile.Common.Utilities.ceilDivide``.

``ceilDivide(numerator, denominator)`` computes ``ceil(numerator/denominator)``
via integer floor division of ``(numerator + denominator - 1) // denominator``.
It has two error branches, each printing a fixed message and returning ``0``:

  - the ValueError guard: when ``numerator < 0 or denominator < 0`` it raises
    (then catches) ``ValueError`` and prints
    ``"ERROR: Can't have a negative register value"``;
  - the ZeroDivisionError guard: a zero ``denominator`` (that was not caught by
    the negative guard) reaches the division, raises (then catches)
    ``ZeroDivisionError`` and prints ``"ERROR: Divide by 0"``.

These tests pin the ACTUAL current behavior (return value AND exact stdout of
each branch) against clean source, verified by running the function's logic:

    ceilDivide(-1, 1)                -> 0, "ERROR: Can't have a negative register value\\n"
    ceilDivide(1, 0)                 -> 0, "ERROR: Divide by 0\\n"
    ceilDivide(0, 1)                 -> 0, ""
    ceilDivide(1, 1)                 -> 1, ""
    ceilDivide(7, 3)                 -> 3, ""
    ceilDivide(99999999999999999, 3) -> 33333333333333333, ""

The eight primary target survivors are string/argument mutations of the two
``print`` calls; they are killed ONLY by exact-text stdout assertions:

  ValueError-branch message (kills mutmut_6/7/8/9):
    _6  print(None)                                    -> "None\\n"
    _7  print("XXERROR: ...valueXX")                   -> "XX...XX\\n"
    _8  print("error: can't have a negative ...")      -> lowercased
    _9  print("ERROR: CAN'T HAVE A NEGATIVE ...")      -> uppercased
  ZeroDivisionError-branch message (kills mutmut_17/18/19/20):
    _17 print(None)                                    -> "None\\n"
    _18 print("XXERROR: Divide by 0XX")                -> "XX...XX\\n"
    _19 print("error: divide by 0")                    -> lowercased
    _20 print("ERROR: DIVIDE BY 0")                    -> uppercased

Boundary/arithmetic survivors (mutmut_1-5, 10-16, 21) are also constrained by
the same tests so the module stands alone as a full characterization.
"""

import pytest

from Tensile.Common.Utilities import ceilDivide

pytestmark = pytest.mark.unit


def test_negative_value_branch_exact_message(capsys):
    """ValueError guard: exact return value and exact stdout.

    Primary targets killed here: mutmut_6/7/8/9 (each rewrites the printed
    negative-register message; any deviation from the exact text fails the
    stdout assertion). Also kills mutmut_1 (``or`` -> ``and``: with a single
    negative operand the guard is skipped, so the function computes
    ``(-1+1-1)//1 == -1`` and prints nothing) and mutmut_10 (``return 1``).
    """
    result = ceilDivide(-1, 1)
    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "ERROR: Can't have a negative register value\n"


def test_zero_numerator_is_not_treated_as_negative(capsys):
    """A zero numerator must NOT enter the negative guard.

    Kills mutmut_2 (``numerator < 0`` -> ``numerator <= 0``) and mutmut_3
    (``numerator < 0`` -> ``numerator < 1``): both would classify ``0`` as
    negative, print the negative-register message, and return 0. The original
    computes ``(0+1-1)//1 == 0`` silently.
    """
    result = ceilDivide(0, 1)
    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == ""


def test_divide_by_zero_branch_exact_message(capsys):
    """ZeroDivisionError guard: exact return value and exact stdout.

    Primary targets killed here: mutmut_17/18/19/20 (each rewrites the printed
    divide-by-zero message). Also kills mutmut_4 (``denominator < 0`` ->
    ``<= 0``) and mutmut_5 (``< 0`` -> ``< 1``): both would treat ``0`` as
    negative and print the negative-register message instead of reaching the
    division, and mutmut_21 (``return 1`` in this branch).
    """
    result = ceilDivide(1, 0)
    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "ERROR: Divide by 0\n"


def test_ceiling_formula_offset_and_floor_semantics(capsys):
    """The happy path: exact ceiling arithmetic with no error output.

    Kills mutmut_14 (``-1`` -> ``+1``: ceilDivide(1,1) would be
    ``(1+1+1)//1 == 3``), mutmut_15 (``numerator+denominator`` ->
    ``numerator-denominator``: ceilDivide(7,3) would be ``(7-3-1)//3 == 1``),
    mutmut_16 (``-1`` -> ``-2``: ceilDivide(7,3) would be ``(7+3-2)//3 == 2``),
    mutmut_11 (``div = None`` -> returns None) and mutmut_12 (``div =
    int(None)`` -> raises TypeError).
    """
    assert ceilDivide(1, 1) == 1
    assert ceilDivide(7, 3) == 3
    assert ceilDivide(2, 3) == 1
    assert capsys.readouterr().out == ""


def test_floor_division_not_float_division():
    """Kills mutmut_13 (``//`` -> ``/``).

    Float division of this large integer loses precision, and ``int()`` then
    truncates to a different value; true integer floor division is exact.
    """
    assert ceilDivide(99999999999999999, 3) == 33333333333333333
