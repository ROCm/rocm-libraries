################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.SpinnyThing.increment``.

Pins the index-advance behavior of ``increment``: each call advances
``self.index`` forward by exactly one, modulo ``len(self.chars)`` (4). This
distinguishes the original ``(index + 1) % len`` from arithmetic mutants that
step backward (``index - 1``) or step by two (``index + 2``).
"""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")

pytestmark = pytest.mark.unit


def test_increment_advances_index_forward_by_one():
    # Fresh instance: index starts at 0, chars has length 4.
    spinner = U.SpinnyThing()
    assert spinner.index == 0
    assert len(spinner.chars) == 4

    # One forward step: (0 + 1) % 4 == 1.
    # Kills mutmut_7 ((0 - 1) % 4 == 3) and mutmut_8 ((0 + 2) % 4 == 2).
    spinner.increment()
    assert spinner.index == 1

    # Second forward step: (1 + 1) % 4 == 2.
    spinner.increment()
    assert spinner.index == 2


def test_increment_wraps_modulo_chars_length():
    # Pin the full forward cycle so a backward step is unambiguously excluded:
    # forward stepping yields 1, 2, 3, 0 over four calls.
    spinner = U.SpinnyThing()
    observed = []
    for _ in range(len(spinner.chars)):
        spinner.increment()
        observed.append(spinner.index)
    assert observed == [1, 2, 3, 0]
