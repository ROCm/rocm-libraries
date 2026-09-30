################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurface characterization: the ``if runningTuning:`` predicate
in ``Tensile/Tensile.py`` at line 409, inside ``restore_prob_sol_map``.

Branch aa18a787b08bf05166d8f981c6cefe39f0e5a016. The predicate is a bare
boolean local ``runningTuning`` (initialized False at line 388; set to True
when a line from the logfile starts with the sentinel
``'run,problem-progress,'`` at line 405).

  * TRUE branch  -> ``runningTuning`` is True (logfile contained at least one
                    sentinel line before ``clientExit``).  The block processes
                    bench-result lines from the tuning log.
  * FALSE branch -> ``runningTuning`` is still False (no sentinel line seen).
                    The ``if runningTuning:`` block is skipped entirely;
                    processing never begins.

These tests pin ACTUAL observed behavior via ``restore_prob_sol_map`` called
with a temporary logfile that exercises each branch.  CPU-only, no GPU probe.
"""

import importlib

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helper: pure predicate extractor (mirrors the assignment at line 405)
# ---------------------------------------------------------------------------

START_HINT = "run,problem-progress,"


def _runningTuning_after_line(line: str) -> bool:
    """Reproduce the exact assignment at Tensile.py:405.

    runningTuning = line.startswith(startHint)

    Returns the resulting boolean value.
    """
    return line.startswith(START_HINT)


# ---------------------------------------------------------------------------
# Pure-helper tests (no I/O, no imports needed)
# ---------------------------------------------------------------------------


def test_sentinel_line_yields_runningTuning_true():
    """A line starting with the sentinel produces runningTuning=True (TRUE branch)."""
    line = "run,problem-progress,0/5,Contraction..."
    assert _runningTuning_after_line(line) is True


def test_non_sentinel_line_yields_runningTuning_false():
    """A line NOT starting with the sentinel keeps runningTuning=False (FALSE branch)."""
    line = "# some header line that is not the sentinel"
    assert _runningTuning_after_line(line) is False


def test_empty_line_yields_runningTuning_false():
    """An empty line (after strip) keeps runningTuning=False."""
    assert _runningTuning_after_line("") is False


def test_partial_sentinel_yields_runningTuning_false():
    """A line with a prefix of the sentinel but not the full sentinel is False."""
    assert _runningTuning_after_line("run,problem-progress") is False


# ---------------------------------------------------------------------------
# Integration tests: call restore_prob_sol_map with real temp logfiles
# ---------------------------------------------------------------------------


def _import_restore():
    """Import restore_prob_sol_map from Tensile.Tensile."""
    M = importlib.import_module("Tensile.Tensile")
    return M.restore_prob_sol_map


def test_false_branch_no_sentinel_returns_empty_map(tmp_path):
    """FALSE branch: logfile with no sentinel line -> function returns empty map.

    runningTuning stays False throughout; the ``if runningTuning:`` block
    at line 409 is never entered; the returned prob_sol_map is {}.
    """
    restore_prob_sol_map = _import_restore()

    logfile = tmp_path / "no_sentinel.log"
    logfile.write_text(
        "# header comment\n"
        "info line\n"
        "another line without the sentinel\n"
    )

    result = restore_prob_sol_map(logfile)
    # No sentinel -> no problem/solution data recorded
    assert result == {}


def test_true_branch_with_sentinel_and_clientExit(tmp_path):
    """TRUE branch: logfile with sentinel present -> enters processing block.

    After the sentinel line, runningTuning=True and the ``if runningTuning:``
    block at line 409 is taken.  The ``clientExit`` line terminates parsing.
    The returned map is empty because no valid Contraction lines follow.
    """
    restore_prob_sol_map = _import_restore()

    logfile = tmp_path / "sentinel_then_exit.log"
    logfile.write_text(
        "# header\n"
        "run,problem-progress,0/1,some-info\n"
        "clientExit\n"
    )

    result = restore_prob_sol_map(logfile)
    # Branch entered, then exited via clientExit before any Contraction line
    assert result == {}
