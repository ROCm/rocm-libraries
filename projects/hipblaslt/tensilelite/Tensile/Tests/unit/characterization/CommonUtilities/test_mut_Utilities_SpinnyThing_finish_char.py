################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.SpinnyThing.finish``.

``finish`` writes exactly ``"\\b*\\n"`` to stdout (then flushes): a backspace to
erase the last spinner char, a ``*`` to mark completion, and a newline. These
tests pin that exact written payload so a mutant that pads the string with
sentinel text (e.g. ``"XX\\b*\\nXX"``) is killed.
"""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")

pytestmark = pytest.mark.unit


def test_finish_writes_exact_payload(capsys):
    # finish() must emit precisely "\b*\n" -- backspace, star, newline.
    # Kills mutmut_2 which writes "XX\b*\nXX" instead.
    spinner = U.SpinnyThing()
    spinner.finish()
    out = capsys.readouterr().out
    assert out == "\b*\n"
    assert "X" not in out
