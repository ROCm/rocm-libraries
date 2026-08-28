# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Test for evaluateEnableESM2TrackValuVsrc() in Solution.py.

The ESM2 VALU-src VA_VDST stamp (EnableESM2TrackValuVsrc) is derived from the
Sparse problem type: on for sparse, off otherwise.
"""

from pathlib import Path

_SOLUTION_PY = Path(__file__).resolve().parents[2] / "SolutionStructs" / "Solution.py"


def _func_body() -> str:
    source = _SOLUTION_PY.read_text(encoding="utf-8")
    start = source.find("def evaluateEnableESM2TrackValuVsrc()")
    assert start != -1, "evaluateEnableESM2TrackValuVsrc not found in Solution.py"
    return source[start : start + 500]


def test_derives_from_sparse():
    """The flag must be derived from state["ProblemType"]["Sparse"]."""
    assert 'state["ProblemType"]["Sparse"]' in _func_body()


def test_state_key_assigned():
    """state["EnableESM2TrackValuVsrc"] must be assigned from the evaluator."""
    source = _SOLUTION_PY.read_text(encoding="utf-8")
    assert 'state["EnableESM2TrackValuVsrc"] = evaluateEnableESM2TrackValuVsrc()' in source
