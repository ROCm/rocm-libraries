################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurfaceDeep characterization: ``if solution["ProblemType"] != problemType`` at
``Tensile/BenchmarkProblems.py:302`` (line numbering in this source tree; the
enclosing function is ``_generateCustomKernelSolutions``).

Branch 0643ca620d99a48ea2d2d60b273d12754ef5c5b9.

The predicate compares two ``ProblemType`` state dicts:

  * ``solution["ProblemType"].state`` — the ProblemType embedded in the custom
    kernel ``.s`` file, read from disk.
  * ``problemType.state``             — the ProblemType declared in the YAML
    config, passed in from ``_benchmarkProblemType``.

``ProblemType.__ne__`` (SolutionStructs/Problem.py:1356) is:

    not (isinstance(other, ProblemType) and self.getAttributes() == other.getAttributes())

At this callsite ``problemType`` is always a ``ProblemType`` instance
(constructed unconditionally at ``BenchmarkStructs.py:83``), so the
``isinstance`` guard is statically True and drops out.  The effective predicate
reduces to::

    solution["ProblemType"].state != problemType.state

**ActivationType normalisation (line 301):** The line immediately preceding the
check writes::

    solution["ProblemType"]["ActivationType"] = problemType["ActivationType"]

This means the kernel's ``ActivationType`` is overwritten with the config value
*before* the comparison, so ``ActivationType`` can never contribute to a
mismatch.  Only other state fields (``DataType``, ``OperationType``,
``UseBias``, etc.) determine the predicate outcome.

Both branches are solver-confirmed reachable (z3 4.16.0 in-container, SAT for
both assignments).  CrossHair 0.0.106 found no counterexample to the
ActivationType-invariance property within a 20 s/condition budget.

CPU-only.  Pure-helper test; no filesystem or GPU access required.
"""

from typing import Any, Dict

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper: model of BenchmarkProblems.py:302 predicate
# ---------------------------------------------------------------------------

def problemtype_mismatch(kernel_state: Dict[str, Any], config_state: Dict[str, Any]) -> bool:
    """Pure model of ``solution["ProblemType"] != problemType`` at BenchmarkProblems.py:302.

    ``ProblemType.__ne__`` reduces to a structural dict-inequality at this
    callsite (``isinstance`` is statically True).

    ActivationType is normalised away on the preceding line (line 301:
    ``solution["ProblemType"]["ActivationType"] = problemType["ActivationType"]``),
    so it can never cause a mismatch; we apply that normalisation here.

    Parameters
    ----------
    kernel_state:
        The ``state`` dict of the kernel-embedded ProblemType (from the ``.s``
        file), *before* the ActivationType normalisation.
    config_state:
        The ``state`` dict of the config-declared ProblemType (from the YAML).

    Returns
    -------
    bool
        True  -> kernel is rejected (or triggers printExit if failOnMismatch).
        False -> kernel is accepted (ProblemType matches the config).
    """
    k = dict(kernel_state)
    # Mirror line 301: overwrite kernel ActivationType with config value.
    if "ActivationType" in config_state:
        k["ActivationType"] = config_state["ActivationType"]
    elif "ActivationType" in k:
        del k["ActivationType"]
    return k != config_state


# ---------------------------------------------------------------------------
# TRUE branch witnesses — predicate is True (kernel rejected)
# ---------------------------------------------------------------------------

class TestProblemtypeMismatchTrue:
    """Predicate evaluates to True: kernel ProblemType differs from config
    after ActivationType normalisation -> kernel is rejected (or printExit)."""

    def test_datatype_mismatch_s_vs_h(self):
        """Kernel has DataType=h, config has DataType=s.
        After ActivationType normalisation both dicts are otherwise equal, but
        DataType differs -> mismatch is True (kernel rejected).
        This is the TRUE witness from the Solve fragment."""
        config_state = {"DataType": "s", "OperationType": "GEMM", "ActivationType": "none"}
        kernel_state = {"DataType": "h", "OperationType": "GEMM", "ActivationType": "none"}
        assert problemtype_mismatch(kernel_state, config_state) is True

    def test_operationtype_mismatch(self):
        """Kernel has a different OperationType than config -> mismatch True."""
        config_state = {"DataType": "s", "OperationType": "GEMM", "ActivationType": "none"}
        kernel_state = {"DataType": "s", "OperationType": "GEMM_GRAD_A", "ActivationType": "none"}
        assert problemtype_mismatch(kernel_state, config_state) is True

    def test_extra_field_in_kernel_causes_mismatch(self):
        """Kernel state has an additional field not present in config state.
        Structural dict-equality fails -> mismatch True."""
        config_state = {"DataType": "s", "OperationType": "GEMM", "ActivationType": "none"}
        kernel_state = {"DataType": "s", "OperationType": "GEMM", "ActivationType": "relu",
                        "UseBias": 1}
        assert problemtype_mismatch(kernel_state, config_state) is True


# ---------------------------------------------------------------------------
# FALSE branch witnesses — predicate is False (kernel accepted)
# ---------------------------------------------------------------------------

class TestProblemtypeMismatchFalse:
    """Predicate evaluates to False: kernel ProblemType matches config after
    ActivationType normalisation -> kernel is accepted."""

    def test_activation_only_diff_normalised_away(self):
        """Kernel has ActivationType=relu, config has ActivationType=none.
        Line 301 normalises kernel ActivationType to 'none' before the check,
        making the two dicts equal -> predicate is False (kernel accepted).
        This is the FALSE witness from the Solve fragment."""
        config_state = {"DataType": "s", "OperationType": "GEMM", "ActivationType": "none"}
        kernel_state = {"DataType": "s", "OperationType": "GEMM", "ActivationType": "relu"}
        assert problemtype_mismatch(kernel_state, config_state) is False

    def test_identical_states(self):
        """Kernel and config state dicts are identical -> predicate is False."""
        config_state = {"DataType": "s", "OperationType": "GEMM", "ActivationType": "none"}
        kernel_state = {"DataType": "s", "OperationType": "GEMM", "ActivationType": "none"}
        assert problemtype_mismatch(kernel_state, config_state) is False

    def test_activation_all_normalised_away(self):
        """Kernel ActivationType=all is overwritten by config's 'none'; other
        fields match -> predicate is False (kernel accepted)."""
        config_state = {"DataType": "h", "OperationType": "GEMM", "ActivationType": "none"}
        kernel_state = {"DataType": "h", "OperationType": "GEMM", "ActivationType": "all"}
        assert problemtype_mismatch(kernel_state, config_state) is False


# ---------------------------------------------------------------------------
# Completeness: parametrized truth-table covering both branches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel_state,config_state,expected", [
    # TRUE: DataType mismatch (s vs h) — solver TRUE witness
    (
        {"DataType": "h", "OperationType": "GEMM", "ActivationType": "none"},
        {"DataType": "s", "OperationType": "GEMM", "ActivationType": "none"},
        True,
    ),
    # TRUE: OperationType mismatch
    (
        {"DataType": "s", "OperationType": "GEMM_GRAD_A", "ActivationType": "none"},
        {"DataType": "s", "OperationType": "GEMM",        "ActivationType": "none"},
        True,
    ),
    # FALSE: ActivationType-only diff normalised away — solver FALSE witness
    (
        {"DataType": "s", "OperationType": "GEMM", "ActivationType": "relu"},
        {"DataType": "s", "OperationType": "GEMM", "ActivationType": "none"},
        False,
    ),
    # FALSE: exact match
    (
        {"DataType": "s", "OperationType": "GEMM", "ActivationType": "none"},
        {"DataType": "s", "OperationType": "GEMM", "ActivationType": "none"},
        False,
    ),
])
def test_problemtype_mismatch_truth_table(kernel_state, config_state, expected):
    """Pin both branches of the BenchmarkProblems.py:302 predicate
    across representative input assignments."""
    assert problemtype_mismatch(kernel_state, config_state) is expected
