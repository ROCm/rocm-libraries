################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.ProgressBar.update``.

These pin the exact arithmetic of the fraction computation and the strict
``>`` tick-advance comparison, so each test passes on clean source and fails
when the corresponding survivor mutant is applied."""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")

pytestmark = pytest.mark.unit


def test_update_fraction_is_value_div_maxvalue():
    """Pins ``currentFraction = 1.0 * value / self.maxValue``.

    With maxValue=100, value=50 the original fraction is exactly 0.5.
    - mutmut_2 (value * maxValue) -> 5000.0
    - mutmut_3 (1.0 / value / maxValue) -> 0.0002
    - mutmut_4 (2.0 * value / maxValue) -> 1.0
    All differ from 0.5, so this assertion distinguishes all three.
    """
    bar = U.ProgressBar(100)
    bar.update(50)
    assert bar.fraction == 0.5


def test_update_numticks_matches_value_div_maxvalue():
    """Pins the tick count derived from the original fraction.

    maxTicks = width - 7 = 73 (default width 80). For value=50, maxValue=100
    the original numTicks = int(0.5 * 73) = 36. The arithmetic mutants
    (mutmut_2/_3/_4) all yield a different fraction and therefore a different
    numTicks, so this is a second independent kill for them.
    """
    bar = U.ProgressBar(100)
    assert bar.maxTicks == 73
    bar.update(50)
    assert bar.numTicks == 36


def test_update_no_advance_when_ticks_equal(capsys):
    """Pins the strict ``>`` comparison (kills mutmut_8: ``>=``).

    On a fresh bar numTicks == 0. Calling update(0) yields currentNumTicks == 0,
    so the original ``0 > 0`` is False: no printStatus, fraction stays 0, and
    nothing is written to stdout. The ``>=`` mutant would take the branch,
    set fraction, and print.
    """
    bar = U.ProgressBar(100)
    bar.update(0)
    captured = capsys.readouterr()
    assert bar.fraction == 0
    assert bar.numTicks == 0
    assert captured.out == ""
