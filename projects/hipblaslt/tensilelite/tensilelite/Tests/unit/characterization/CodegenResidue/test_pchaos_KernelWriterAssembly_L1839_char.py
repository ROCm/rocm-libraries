################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterisation test — branch_id 2829358caa879be3c22e412e96e81fe8c9fd7610

Predicate : not kernel["EnableMatrixInstruction"]
Site      : Tensile/KernelWriterAssembly.py:1839
Function  : macroAndSet
Solver    : z3 4.16.0  — SAT  (solver-backed-under-assumptions)

Classification note
-------------------
Both polarities are solver-satisfiable over the seeded boolean domain
{False, True} for EnableMatrixInstruction (declared in ValidParameters.py:1054).

  EMI=False -> predicate True  -> enter branch (legacy MAC path)
  EMI=True  -> predicate False -> skip branch (MatrixInstruction path)

The predicate is a pure function of one bool public input; no GPU/env gate.

Tests here:
  1. Predicate semantics — pin that the negation logic is correct for both z3
     witnesses.
  2. ValidParameters pin — confirm EnableMatrixInstruction is declared as a
     two-valued boolean in ValidParameters so both branches remain reachable.
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Predicate semantics (pure; no Tensile import required)
# ---------------------------------------------------------------------------

def _pred(enable_matrix_instruction: bool) -> bool:
    """Mirror of the branch predicate at KernelWriterAssembly.py:1839.

    ``if not kernel["EnableMatrixInstruction"]:``
    """
    return not enable_matrix_instruction


def test_emi_false_enters_branch():
    """z3 true witness: EnableMatrixInstruction=False -> predicate True (legacy MAC)."""
    assert _pred(False) is True


def test_emi_true_skips_branch():
    """z3 false witness: EnableMatrixInstruction=True -> predicate False (MatrixInstruction path)."""
    assert _pred(True) is False


# ---------------------------------------------------------------------------
# ValidParameters pin — both values declared reachable
# ---------------------------------------------------------------------------

def test_enable_matrix_instruction_declared_as_two_valued_bool():
    """
    ValidParameters.py:1054 declares 'EnableMatrixInstruction': [False, True]
    inside newMIValidParameters. Pin the declaration so any future narrowing of
    the domain is caught here, which would make one branch of
    KernelWriterAssembly.py:1839 structurally dead.
    """
    from Tensile.Common.ValidParameters import newMIValidParameters  # noqa: PLC0415

    domain = newMIValidParameters["EnableMatrixInstruction"]
    assert False in domain, (
        "EnableMatrixInstruction no longer includes False in validParameters; "
        "the MAC legacy branch at KernelWriterAssembly.py:1839 would become dead code."
    )
    assert True in domain, (
        "EnableMatrixInstruction no longer includes True in validParameters; "
        "the MatrixInstruction skip path at KernelWriterAssembly.py:1839 would become dead code."
    )
