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

"""Mutation-killing characterization tests for ``_report_xcc_failure``.

``_report_xcc_failure`` has no return value and mutates the module-global
``_xcc_failures_by_file`` map identically on every call, so the *only*
observable, branch-dependent behaviour is what it prints to stdout:

  * call #1 for a file (count == 1): full ``Error: ...`` line incl. the
    solution's ``SolutionIndex`` (or ``'?'`` when absent),
  * call #2 (count == 2): the ``... (more solutions in this file)`` line,
  * call #3+ : silence.

These tests capture stdout (``capsys``) to pin that exact contract, which the
state-only snapshot suite in ``test_validworkgroupmappingxcc_char.py`` cannot
distinguish. Every test resets the global map first so order never matters.
"""

from pathlib import Path

import pytest

import Tensile.TensileLogic.ValidWorkGroupMappingXCC as xcc_mod
from Tensile.TensileLogic.ValidWorkGroupMappingXCC import (
    _report_xcc_failure,
    reset_reported_failures,
)

pytestmark = pytest.mark.unit

_FILE = Path("logic/aquavanjaram/gfx942_38cu/Equality/logic.yaml")
_DETAIL = "WorkGroupMappingXCC must be -1 or positive (WorkGroupMappingXCC=0)"


# --- count == 1 branch (full message) : mutmut_12, 13, 14 -------------------

def test_first_failure_prints_full_error_line(capsys):
    """count==1 -> full ``Error: ...`` line is printed on the FIRST call.

    Kills mutmut_12 (``count != 1``: silent on 1st call) and mutmut_13
    (``count == 2``: silent on 1st call).
    """
    reset_reported_failures()
    _report_xcc_failure(_FILE, {"SolutionIndex": 7}, _DETAIL)
    out = capsys.readouterr().out
    assert out == f"Error: {_DETAIL} (file: {_FILE}, index: 7)\n"


def test_first_failure_message_is_not_literal_none(capsys):
    """count==1 prints the formatted f-string, not ``None``.

    Kills mutmut_14 (``print(None)``).
    """
    reset_reported_failures()
    _report_xcc_failure(_FILE, {"SolutionIndex": 7}, _DETAIL)
    out = capsys.readouterr().out
    assert out != "None\n"
    assert _DETAIL in out


# --- SolutionIndex lookup (present) : mutmut_15, 19, 20, 21 -----------------

def test_present_solution_index_is_emitted(capsys):
    """When ``SolutionIndex`` is present, its value appears as ``index: 7``.

    A correct lookup must read the real key; mutants that look up ``None``
    (mutmut_15) or a mangled key (mutmut_19/20/21) fall back to ``'?'``.
    Using a value (7) distinct from ``'?'`` distinguishes them.
    """
    reset_reported_failures()
    _report_xcc_failure(_FILE, {"SolutionIndex": 7}, _DETAIL)
    out = capsys.readouterr().out
    assert "index: 7)" in out
    assert "index: ?)" not in out


# --- SolutionIndex lookup (absent) : mutmut_16, 17, 18, 22 -----------------

def test_absent_solution_index_falls_back_to_question_mark(capsys):
    """When ``SolutionIndex`` is absent, the default ``'?'`` is printed.

    Kills mutmut_16/17/18 (default becomes ``None`` -> ``index: None``) and
    mutmut_22 (default literal becomes ``'XX?XX'``).
    """
    reset_reported_failures()
    _report_xcc_failure(_FILE, {}, _DETAIL)
    out = capsys.readouterr().out
    assert "index: ?)" in out
    assert "index: None)" not in out
    assert "XX?XX" not in out


# --- count == 2 branch ("...more") : mutmut_13(elif), 23, 24, 25 -----------

def test_second_failure_prints_more_line(capsys):
    """count==2 -> the ``... (more solutions in this file)`` line.

    Kills mutmut_24 (``count == 3``: silent on the 2nd call) and is part of
    pinning mutmut_23 / mutmut_25.
    """
    reset_reported_failures()
    _report_xcc_failure(_FILE, {"SolutionIndex": 0}, _DETAIL)  # count 1
    capsys.readouterr()  # discard the first (full) line
    _report_xcc_failure(_FILE, {"SolutionIndex": 1}, _DETAIL)  # count 2
    out = capsys.readouterr().out
    assert out == "  ... (more solutions in this file)\n"


def test_second_failure_more_line_is_not_literal_none(capsys):
    """count==2 prints the ``...more`` text, not ``None``.

    Kills mutmut_25 (``print(None)``).
    """
    reset_reported_failures()
    _report_xcc_failure(_FILE, {"SolutionIndex": 0}, _DETAIL)  # count 1
    capsys.readouterr()
    _report_xcc_failure(_FILE, {"SolutionIndex": 1}, _DETAIL)  # count 2
    out = capsys.readouterr().out
    assert out != "None\n"
    assert "more solutions in this file" in out


def test_third_failure_is_silent(capsys):
    """count==3 prints nothing (only count 1 and 2 emit).

    Kills mutmut_23 (``count != 2``: the ``...more`` line would re-fire on the
    3rd call).
    """
    reset_reported_failures()
    for idx in range(2):
        _report_xcc_failure(_FILE, {"SolutionIndex": idx}, _DETAIL)
    capsys.readouterr()  # discard call 1 + 2 output
    _report_xcc_failure(_FILE, {"SolutionIndex": 2}, _DETAIL)  # count 3
    out = capsys.readouterr().out
    assert out == ""
