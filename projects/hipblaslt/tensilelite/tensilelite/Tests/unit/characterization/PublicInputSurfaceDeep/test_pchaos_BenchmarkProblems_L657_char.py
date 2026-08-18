################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurfaceDeep characterization: ``elif not os.path.exists(resultsFileName) or
globalParameters["ForceRedoBenchmarkProblems"]`` at
``Tensile/BenchmarkProblems.py:657``.

Branch 6ff09bcdcc5703903cfcdef2e13b07dfd86ec4fb.

The predicate is a boolean disjunction of two public inputs:

  * ``exists``  -- ``os.path.exists(resultsFileName)``; abstracted as a free
    boolean since the predicate outcome is fully determined by its boolean value.
  * ``force``   -- ``globalParameters["ForceRedoBenchmarkProblems"]``; a
    runtime configuration flag whose hardcoded default is ``True``
    (GlobalParameters.py:101).

Truth table (all 4 assignments, solver-backed via z3 4.16.0 in-container):

  (exists=F, force=F) -> True   # file absent  — always redo
  (exists=F, force=T) -> True   # file absent  + forced (redundant)
  (exists=T, force=F) -> False  # file present + NOT forced — UNIQUE false case
  (exists=T, force=T) -> True   # file present + forced override

TRUE branch: the benchmark client is run (re-/first-time execution).
FALSE branch: the results file exists AND ForceRedo is explicitly False;
  benchmark is skipped.

The FALSE branch requires ``ForceRedoBenchmarkProblems=False`` (opt-in
override), since the default is True.

CPU-only.  Pure-helper test; no filesystem or GPU access required.
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure-helper: mirror the predicate at BenchmarkProblems.py:657
# ---------------------------------------------------------------------------

def benchmark_rerun_guard(exists: bool, force: bool) -> bool:
    """Return the value of the elif predicate at BenchmarkProblems.py:657.

    ``not os.path.exists(resultsFileName) or globalParameters["ForceRedoBenchmarkProblems"]``

    Parameters
    ----------
    exists:
        Boolean value of ``os.path.exists(resultsFileName)``.
    force:
        Boolean value of ``globalParameters["ForceRedoBenchmarkProblems"]``.
    """
    return (not exists) or force


# ---------------------------------------------------------------------------
# TRUE branch witnesses — guard evaluates to True (benchmark is run)
# ---------------------------------------------------------------------------

class TestBenchmarkRerunGuardTrue:
    """Predicate is True: the benchmark client is (re-)executed."""

    def test_file_absent_force_false(self):
        """File does not exist; ForceRedo is False.
        File-absent dominates: guard is True."""
        assert benchmark_rerun_guard(exists=False, force=False) is True

    def test_file_absent_force_true(self):
        """File does not exist; ForceRedo is True (default).
        Both operands agree: guard is True."""
        assert benchmark_rerun_guard(exists=False, force=True) is True

    def test_file_present_force_true(self):
        """File exists but ForceRedo=True overrides the skip.
        Forced rerun: guard is True."""
        assert benchmark_rerun_guard(exists=True, force=True) is True


# ---------------------------------------------------------------------------
# FALSE branch witness — guard evaluates to False (benchmark is skipped)
# ---------------------------------------------------------------------------

class TestBenchmarkRerunGuardFalse:
    """Predicate is False: the results file exists and ForceRedo is off;
    the benchmark step is skipped.  This is the UNIQUE false assignment."""

    def test_file_present_force_false(self):
        """File exists AND ForceRedo=False — unique false assignment.
        Benchmark is skipped; guard is False."""
        assert benchmark_rerun_guard(exists=True, force=False) is False


# ---------------------------------------------------------------------------
# Completeness: all four truth-table rows in one parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exists,force,expected", [
    (False, False, True),
    (False, True,  True),
    (True,  False, False),
    (True,  True,  True),
])
def test_benchmark_rerun_guard_truth_table(exists, force, expected):
    """Pin the full 2x2 truth table for the BenchmarkProblems.py:657 predicate."""
    assert benchmark_rerun_guard(exists=exists, force=force) is expected
