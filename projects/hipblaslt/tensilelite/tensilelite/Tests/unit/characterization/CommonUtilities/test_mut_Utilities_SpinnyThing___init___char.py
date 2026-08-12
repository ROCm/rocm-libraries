################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.SpinnyThing.__init__``.

These pin the exact attribute initialization performed by the constructor so
that survivor mutants altering the spinner-char list or the initial index are
detected. They assert ACTUAL current behavior (characterization), not an
idealized contract."""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")

pytestmark = pytest.mark.unit


def test_init_chars_list_exact():
    """The spinner chars are exactly the four single-character glyphs.

    Kills mutmut_2 (chars[0] "|"->"XX|XX"), mutmut_3 (chars[1] "/"->"XX/XX"),
    mutmut_4 (chars[2] "-"->"XX-XX"), mutmut_5 (chars[3] "\\"->"XX\\XX").
    """
    spinner = U.SpinnyThing()
    assert spinner.chars == ["|", "/", "-", "\\"]


def test_init_index_zero():
    """The initial spinner index is integer 0. Kills mutmut_7 (0 -> 1)."""
    spinner = U.SpinnyThing()
    assert spinner.index == 0
