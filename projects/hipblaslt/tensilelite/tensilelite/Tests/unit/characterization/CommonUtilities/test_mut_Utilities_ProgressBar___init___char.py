################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.ProgressBar.__init__``.

These pin the exact attribute initialization performed by the constructor so
that survivor mutants flipping individual assignments are detected. They assert
ACTUAL current behavior (characterization), not an idealized contract."""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")

pytestmark = pytest.mark.unit


def test_init_default_width_and_maxticks():
    """Default width is 80 and maxTicks = width - 7.

    Kills mutmut_1 (width default 80->81), mutmut_7 (width-7 -> width+7),
    mutmut_8 (width-7 -> width-8).
    """
    bar = U.ProgressBar(100)
    assert bar.width == 80
    assert bar.maxTicks == 73  # 80 - 7


def test_init_maxticks_explicit_width():
    """maxTicks is computed as width - 7 for an explicit width.

    Reinforces mutmut_7 (width+7 -> 27) and mutmut_8 (width-8 -> 12) kills with
    an independent width value.
    """
    bar = U.ProgressBar(100, width=20)
    assert bar.width == 20
    assert bar.maxTicks == 13  # 20 - 7


def test_init_char_is_pipe():
    """The tick char is the single pipe character. Kills mutmut_3."""
    bar = U.ProgressBar(100)
    assert bar.char == "|"


def test_init_priorvalue_zero():
    """priorValue starts at 0. Kills mutmut_10 (0 -> 1)."""
    bar = U.ProgressBar(100)
    assert bar.priorValue == 0


def test_init_fraction_zero():
    """fraction starts at integer 0. Kills mutmut_11 (0 -> None) and
    mutmut_12 (0 -> 1)."""
    bar = U.ProgressBar(100)
    assert bar.fraction == 0
    assert bar.fraction is not None


def test_init_numticks_zero():
    """numTicks starts at 0. Kills mutmut_14 (0 -> 1)."""
    bar = U.ProgressBar(100)
    assert bar.numTicks == 0
