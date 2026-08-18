################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for ``Tensile.Common.Utilities.print1``.

print1 prints the message (and flushes) iff ``getVerbosity() >= 1``. These tests
pin the verbosity boundary at exactly 1 and the exact message that is printed.
"""

import contextlib
import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")

pytestmark = pytest.mark.unit


@contextlib.contextmanager
def _verbosity(v):
    saved = U.getVerbosity()
    U.setVerbosity(v)
    try:
        yield
    finally:
        U.setVerbosity(saved)


def test_print1_prints_at_verbosity_one(capsys):
    # Boundary: at verbosity == 1 the message MUST be printed.
    # Kills `>= 1` -> `> 1` (mutmut_1) and `>= 1` -> `>= 2` (mutmut_2),
    # both of which would suppress output at verbosity 1.
    with _verbosity(1):
        U.print1("boundary-msg")
        assert capsys.readouterr().out == "boundary-msg\n"


def test_print1_prints_exact_message(capsys):
    # The printed text is the message itself, not a placeholder.
    # Kills `print(message)` -> `print(None)` (mutmut_3): mutant emits "None\n".
    with _verbosity(1):
        U.print1("exact-payload")
        out = capsys.readouterr().out
    assert out == "exact-payload\n"
    assert "None" not in out
