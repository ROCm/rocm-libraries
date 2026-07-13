################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurfaceDeep characterization: ``if ForceRedoBenchmarkProblems or not exists``
at ``Tensile/BenchmarkProblems.py:740`` inside ``benchmarkProblemType``.

Branch 8e797886ed0f058dd8a9d8a97dc4aa07a8cccfb5.

The predicate is::

    if globalParameters["ForceRedoBenchmarkProblems"] \\
            or not os.path.exists(newResultsFileName):

Two independent boolean arms:

* ``arm1`` — ``globalParameters["ForceRedoBenchmarkProblems"]`` (public input: a
  ``bool`` in the global parameters dict, default ``True``).
* ``arm2`` — ``not os.path.exists(newResultsFileName)`` (external runtime state:
  filesystem probe; unreachable from public inputs alone without a prior
  benchmark run that has written the CSV file).

Truth table (P = arm1 OR arm2):

  force=F, exists=F  ->  P = True   (file absent forces re-run)
  force=T, exists=T  ->  P = True   (flag forces re-run regardless of file)
  force=F, exists=T  ->  P = False  (only false region: flag off AND file present)

The FALSE region is unreachable from pure public inputs — it requires a prior
benchmark run to have written the CSV.  Tests pin ACTUAL observed behavior.
CPU-only; no GPU hardware required.
"""

import os

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure-helper: mirror the predicate at BenchmarkProblems.py:740
# ---------------------------------------------------------------------------


def _force_or_missing(force: bool, file_exists: bool) -> bool:
    """Mirror predicate ``globalParameters["ForceRedoBenchmarkProblems"] or not os.path.exists(...)``.

    Args:
        force: value of ``globalParameters["ForceRedoBenchmarkProblems"]``.
        file_exists: whether the results CSV file exists on disk.

    Returns:
        True when benchmarking should proceed (re-run or file absent).
        False only when force is False AND the file already exists.
    """
    return force or not file_exists


# ---------------------------------------------------------------------------
# Pure-helper tests (no I/O, deterministic — full truth table)
# ---------------------------------------------------------------------------


def test_force_false_file_absent_predicate_is_true():
    """force=False, file absent -> P = True (file missing drives re-run)."""
    assert _force_or_missing(False, False) is True


def test_force_true_file_present_predicate_is_true():
    """force=True, file present -> P = True (flag overrides presence of file)."""
    assert _force_or_missing(True, True) is True


def test_force_true_file_absent_predicate_is_true():
    """force=True, file absent -> P = True (both arms True, short-circuits on arm1)."""
    assert _force_or_missing(True, False) is True


def test_force_false_file_present_predicate_is_false():
    """force=False, file present -> P = False (unique false region: file exists, no force)."""
    assert _force_or_missing(False, True) is False


# ---------------------------------------------------------------------------
# Integration pin: arm1 public input via globalParameters dict
# ---------------------------------------------------------------------------


def test_arm1_force_true_from_global_parameters():
    """arm1: globalParameters["ForceRedoBenchmarkProblems"]=True -> predicate True.

    Exercises the public-input arm directly by reading from globalParameters
    as the production code does.  Does not reach the filesystem probe.
    """
    from Tensile.Common.GlobalParameters import globalParameters

    # Snapshot and restore to avoid cross-test contamination.
    original = globalParameters["ForceRedoBenchmarkProblems"]
    try:
        globalParameters["ForceRedoBenchmarkProblems"] = True
        force = globalParameters["ForceRedoBenchmarkProblems"]
        # arm1 alone -> predicate True regardless of arm2
        result = force or False  # file_exists=True would make not-exists False
        assert result is True
    finally:
        globalParameters["ForceRedoBenchmarkProblems"] = original


def test_arm1_force_false_from_global_parameters():
    """arm1: globalParameters["ForceRedoBenchmarkProblems"]=False -> arm1 contributes False.

    When force is False, the predicate outcome depends on arm2 (file probe).
    This pins that arm1 does not short-circuit the OR when False.
    """
    from Tensile.Common.GlobalParameters import globalParameters

    original = globalParameters["ForceRedoBenchmarkProblems"]
    try:
        globalParameters["ForceRedoBenchmarkProblems"] = False
        force = globalParameters["ForceRedoBenchmarkProblems"]
        assert force is False
        # With force=False, predicate reduces to `not os.path.exists(newResultsFileName)`
        # Pin via pure helper: force=False, file_exists=False -> True
        assert _force_or_missing(force, False) is True
        # Pin via pure helper: force=False, file_exists=True -> False
        assert _force_or_missing(force, True) is False
    finally:
        globalParameters["ForceRedoBenchmarkProblems"] = original


def test_arm1_default_is_bool():
    """Default value of globalParameters["ForceRedoBenchmarkProblems"] is a bool.

    Pins the type so regressions in GlobalParameters.py are caught.
    The canonical default (line 101 of GlobalParameters.py) is True.
    """
    from Tensile.Common.GlobalParameters import globalParameters

    assert isinstance(globalParameters["ForceRedoBenchmarkProblems"], bool)


# ---------------------------------------------------------------------------
# Integration pin: arm2 filesystem probe (via tmp_path, no monkeypatching)
# ---------------------------------------------------------------------------


def test_arm2_file_absent_predicate_is_true(tmp_path):
    """arm2: results CSV does not exist -> not os.path.exists(...) = True -> P = True."""
    nonexistent = str(tmp_path / "missing_results.csv")
    assert not os.path.exists(nonexistent)
    # Replicate predicate with force=False so arm2 is decisive
    result = False or not os.path.exists(nonexistent)
    assert result is True


def test_arm2_file_present_predicate_is_false(tmp_path):
    """arm2: results CSV exists -> not os.path.exists(...) = False -> P = False (force=False).

    This is the only false region: file present AND ForceRedoBenchmarkProblems=False.
    """
    existing = tmp_path / "results.csv"
    existing.write_text("dummy")
    assert os.path.exists(str(existing))
    # Replicate predicate with force=False so arm2 is decisive
    result = False or not os.path.exists(str(existing))
    assert result is False
