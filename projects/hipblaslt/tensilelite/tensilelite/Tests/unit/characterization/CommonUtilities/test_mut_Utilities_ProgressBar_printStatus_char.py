################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.ProgressBar.printStatus``.

``printStatus`` writes a progress line directly to ``sys.stdout`` using a fixed
format. These tests pin the EXACT current output bytes so that mutations to the
literals, the percentage arithmetic, the completion branch condition, and the
elapsed-time arithmetic are all observable via ``capsys``.

The tests drive ``printStatus`` by constructing a ``ProgressBar`` and setting
its instance fields directly, so output is fully deterministic.
"""

import importlib

import pytest

U = importlib.import_module("Tensile.Common.Utilities")

pytestmark = pytest.mark.unit


def _make_bar(*, maxTicks, char, numTicks, fraction, createTime):
    """Build a ProgressBar with deterministic printStatus inputs."""
    bar = U.ProgressBar(maxValue=1, width=80)
    bar.maxTicks = maxTicks
    bar.char = char
    bar.numTicks = numTicks
    bar.fraction = fraction
    bar.createTime = createTime
    return bar


def test_printstatus_incomplete_line_exact_output(capsys):
    """Pin the exact in-progress line (numTicks < maxTicks).

    Kills:
      - mutmut_2:  "\\r"            -> "XX\\rXX"   (leading CR prefix)
      - mutmut_5:  "[%-*s] %3d%%"  -> "XX...%%XX" (bar-line literal)
      - mutmut_8:  fraction * 100  -> fraction / 100  (percentage value)
      - mutmut_9:  fraction * 100  -> fraction * 101  (percentage value)
    The bar is incomplete, so no "secs elapsed" suffix is written.
    """
    bar = _make_bar(maxTicks=10, char="|", numTicks=3, fraction=1.0, createTime=0.0)
    bar.printStatus()
    out = capsys.readouterr().out
    # Original: "\r" then "[|||       ] 100%"; fraction*100 == 100 (not 101, not 0.01).
    assert out == "\r[|||       ] 100%"


def test_printstatus_percentage_uses_times_100_not_divide(capsys):
    """A fractional value pins fraction*100 specifically.

    fraction=0.5 -> "%3d" of 50.0 == " 50"; the /100 mutant (mutmut_8) would
    yield 0.005 -> "  0", and *101 (mutmut_9) -> 50.5 -> still " 50" but the
    100% case in the prior test separates them. This test reinforces mutmut_8.
    """
    bar = _make_bar(maxTicks=10, char="|", numTicks=5, fraction=0.5, createTime=0.0)
    bar.printStatus()
    out = capsys.readouterr().out
    assert out == "\r[|||||     ]  50%"


def test_printstatus_incomplete_has_no_elapsed_suffix(capsys):
    """When numTicks != maxTicks the completion branch must NOT fire.

    Kills mutmut_10: `if self.numTicks == self.maxTicks` -> `!=`. Under the
    mutant, an incomplete bar would incorrectly append the elapsed-time suffix.
    """
    bar = _make_bar(maxTicks=10, char="|", numTicks=3, fraction=1.0, createTime=0.0)
    bar.printStatus()
    out = capsys.readouterr().out
    assert "secs elapsed" not in out
    assert out == "\r[|||       ] 100%"


def test_printstatus_complete_line_exact_output(capsys):
    """Pin the completed line incl. the elapsed-time suffix (numTicks==maxTicks).

    Kills:
      - mutmut_10: `==` -> `!=` (suffix would be dropped on completion)
      - mutmut_14: " (%-.1f secs elapsed)\\n" -> "XX (...)\\nXX" (suffix literal)
      - mutmut_15: "secs elapsed" -> "SECS ELAPSED" (suffix casing)
      - mutmut_16: (stopTime - createTime) -> (stopTime + createTime)

    createTime is set to 0.0 and time.time() is large/positive, so subtraction
    yields a positive elapsed value while addition yields the same magnitude;
    they coincide numerically here, so mutmut_16 is pinned by the dedicated
    test below with a negative createTime. This test fixes the literal/casing.
    """
    bar = _make_bar(maxTicks=4, char="|", numTicks=4, fraction=1.0, createTime=0.0)
    bar.printStatus()
    out = capsys.readouterr().out
    assert out.startswith("\r[||||] 100%")
    assert " secs elapsed)\n" in out
    assert out.endswith(")\n")
    # Casing must be lowercase exactly as the original literal.
    assert "secs elapsed" in out
    assert "SECS ELAPSED" not in out
    # No mutated wrapper bytes around the suffix.
    assert "XX" not in out


def test_printstatus_elapsed_is_difference_not_sum(capsys, monkeypatch):
    """Pin elapsed = stopTime - createTime (mutmut_16: '-' -> '+').

    With createTime far in the past, the difference is a small positive number
    while the sum would be ~2x the epoch (a huge value). Freeze time.time().
    """
    fixed_now = 1000.0
    monkeypatch.setattr(U.time, "time", lambda: fixed_now)
    bar = _make_bar(maxTicks=4, char="|", numTicks=4, fraction=1.0, createTime=995.0)
    bar.printStatus()
    out = capsys.readouterr().out
    # Original elapsed = 1000.0 - 995.0 = 5.0 -> " (5.0 secs elapsed)\n".
    # Mutant (sum) = 1995.0 -> " (1995.0 secs elapsed)\n".
    assert out == "\r[||||] 100% (5.0 secs elapsed)\n"
