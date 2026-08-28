################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
################################################################################
"""
Regression test for the sparse guard in evaluateStinkyTofuESM2().

Ensures that Solution.py's evaluateStinkyTofuESM2() contains an early-return
for sparse problem types, preventing Expert Scheduling Mode 2 from being
enabled on sparse kernels (which caused non-deterministic correctness failures
on gfx1250 due to missing s_wait_alu coverage in the StinkyTofu pipeline).
"""

from pathlib import Path
import ast


_SOLUTION_PY = (
    Path(__file__).resolve().parents[2] / "SolutionStructs" / "Solution.py"
)


def _get_source() -> str:
    return _SOLUTION_PY.read_text(encoding="utf-8")


def test_solution_py_exists():
    assert _SOLUTION_PY.exists(), f"Solution.py not found at {_SOLUTION_PY}"


def test_evaluateStinkyTofuESM2_has_sparse_guard():
    """evaluateStinkyTofuESM2 must reject sparse problem types (Sparse != 0).

    Without this guard, gfx1250 sparse kernels enable ESM2 which disables
    hardware VGPR interlock and relies on StinkyTofu to insert s_wait_alu.
    The StinkyTofu pipeline currently has a gap for the metadata-pack -> SWMMAC
    dependency, leading to intermittent correctness failures (~30-57% fail rate).
    """
    source = _get_source()
    func_start = source.find("def evaluateStinkyTofuESM2()")
    assert func_start != -1, "evaluateStinkyTofuESM2 not found in Solution.py"

    # Extract the function body up to the next top-level def/class at same indent
    func_body = source[func_start : func_start + 1000]

    assert 'state["ProblemType"]["Sparse"]' in func_body, (
        'evaluateStinkyTofuESM2 must check state["ProblemType"]["Sparse"] '
        "to avoid enabling ESM2 on sparse kernels"
    )
    assert "return False" in func_body, (
        "evaluateStinkyTofuESM2 must have a return False path for sparse kernels"
    )


def test_sparse_guard_appears_before_dtype_checks():
    """The sparse guard should appear early in evaluateStinkyTofuESM2,
    before the dtype (f64) guards, to short-circuit as soon as possible."""
    source = _get_source()
    func_start = source.find("def evaluateStinkyTofuESM2()")
    assert func_start != -1

    func_body = source[func_start : func_start + 1000]

    sparse_pos = func_body.find('state["ProblemType"]["Sparse"]')
    f64_pos = func_body.find("isDouble()")

    assert sparse_pos != -1, "Sparse guard not found"
    assert f64_pos != -1, "f64 guard not found"
    assert sparse_pos < f64_pos, (
        "Sparse guard should appear before the f64 dtype guard in evaluateStinkyTofuESM2"
    )
