# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Test for evaluateEnableESM2TrackValuVsrc() in Solution.py.

The ESM2 VALU-src VA_VDST stamp (EnableESM2TrackValuVsrc) is on for every kernel.
It was previously derived from the Sparse problem type.
"""

from pathlib import Path

_SOLUTION_PY = Path(__file__).resolve().parents[2] / "SolutionStructs" / "Solution.py"


def _func_body() -> str:
    source = _SOLUTION_PY.read_text(encoding="utf-8")
    start = source.find("def evaluateEnableESM2TrackValuVsrc()")
    assert start != -1, "evaluateEnableESM2TrackValuVsrc not found in Solution.py"
    return source[start : start + 500]


def test_enabled_unconditionally():
    """The flag must be on regardless of problem type."""
    body = _func_body()
    assert "return True" in body
    assert 'state["ProblemType"]["Sparse"]' not in body


def test_state_key_assigned():
    """state["EnableESM2TrackValuVsrc"] must be assigned from the evaluator."""
    source = _SOLUTION_PY.read_text(encoding="utf-8")
    assert 'state["EnableESM2TrackValuVsrc"] = evaluateEnableESM2TrackValuVsrc()' in source
