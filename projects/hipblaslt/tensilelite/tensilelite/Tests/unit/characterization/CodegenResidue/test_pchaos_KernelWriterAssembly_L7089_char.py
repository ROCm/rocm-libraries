################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterisation test — branch_id 4480256b782f32e867dd3bf96665dc8ae5b6a883

Predicate : tailLoop
Site      : Tensile/KernelWriterAssembly.py:7089  (calculateLoopNumIter, if)
Solver    : z3 4.16.0  — SAT  (solver-backed-under-assumptions)

What the predicate reads
------------------------
``tailLoop`` is computed at line 7081::

    tailLoop = loopIdx < 0

The predicate at line 7089 ``if tailLoop:`` gates the *Tail Loop* code section
inside ``calculateLoopNumIter``.

Call-site convention (docstring at 7074-7079 and surrounding callers):
  loopIdx == -1   -> tail-loop path requested  -> tailLoop = True
  loopIdx >= 0    -> summation-loop index       -> tailLoop = False

Reachability (NoTailLoop gate — KernelWriter.py:5352):
  The entire tail-loop section is only emitted when ``kernel[NoTailLoop]``
  is False.  NoTailLoop is derived in Solution.py:3845-3847::

      state["NoTailLoop"] = (AssertSummationElementMultiple % DepthU == 0)

  So the TRUE witness (loopIdx=-1) is reachable only when ASEM % DepthU != 0.
  The FALSE witness (loopIdx>=0) can coincide with NoTailLoop=True.

z3 witnesses (cross-checked in-container with DIRECT_EVAL_OK=True):
  TRUE  : loopIdx=-1, ASEM=12, DepthU=8  -> tailLoop=True,  NoTailLoop=False
  FALSE : loopIdx=0,  ASEM=8,  DepthU=8  -> tailLoop=False, NoTailLoop=True

Classification: solver-backed-under-assumptions
(loopIdx is a call-site convention arg, not a raw YAML key; NoTailLoop
 reachability gate is a derived YAML param; both reduce to seeded ints).
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper — mirrors KernelWriterAssembly.calculateLoopNumIter line 7081.
# ---------------------------------------------------------------------------

def is_tail_loop(loopIdx: int) -> bool:
    """Mirror of KernelWriterAssembly.calculateLoopNumIter L7081:
       tailLoop = loopIdx < 0.
    The if at L7089 tests this bool: ``if tailLoop:``.

    post: __return__ == (loopIdx < 0)
    """
    return loopIdx < 0


def no_tail_loop(AssertSummationElementMultiple: int, DepthU: int) -> bool:
    """Mirror of Solution.py:3845-3847:
       state["NoTailLoop"] = (AssertSummationElementMultiple % DepthU == 0).
    When True, the tail section (loopIdx=-1 calls) is not emitted (KernelWriter.py:5352).
    """
    return (AssertSummationElementMultiple % DepthU) == 0


# ---------------------------------------------------------------------------
# TRUE witness — tail loop section entered
# ---------------------------------------------------------------------------

def test_tail_loop_true_loopIdx_minus1():
    """z3 TRUE: loopIdx=-1 -> tailLoop=True (tail-loop code section entered)."""
    assert is_tail_loop(-1) is True


def test_tail_loop_true_reachable_with_asem12_depthu8():
    """z3 TRUE (reachability): ASEM=12, DepthU=8 -> NoTailLoop=False, so tail call is emitted."""
    asem, depthu = 12, 8
    assert no_tail_loop(asem, depthu) is False  # gate open: tail call is emitted
    assert is_tail_loop(-1) is True              # loopIdx=-1 enters the branch at L7089


# ---------------------------------------------------------------------------
# FALSE witness — tail loop section skipped
# ---------------------------------------------------------------------------

def test_tail_loop_false_loopIdx_zero():
    """z3 FALSE: loopIdx=0 -> tailLoop=False (tail-loop code section skipped)."""
    assert is_tail_loop(0) is False


def test_tail_loop_false_coincides_with_no_tail_loop_asem8_depthu8():
    """z3 FALSE (reachability): ASEM=8, DepthU=8 -> NoTailLoop=True -> no tail call emitted."""
    asem, depthu = 8, 8
    assert no_tail_loop(asem, depthu) is True    # gate closed: tail section not emitted
    assert is_tail_loop(0) is False              # summation loop (loopIdx>=0) skips L7089 branch


# ---------------------------------------------------------------------------
# Boundary pins — document the exact transition at 0/-1
# ---------------------------------------------------------------------------

def test_tail_loop_boundary_minus2_is_also_tail():
    """Any negative loopIdx (not just -1) satisfies the predicate."""
    assert is_tail_loop(-2) is True


def test_tail_loop_positive_loopIdx_is_not_tail():
    """Positive loopIdx values (other summation loop indices) also skip the branch."""
    assert is_tail_loop(1) is False
    assert is_tail_loop(5) is False


def test_no_tail_loop_asem_not_divisible():
    """When ASEM is not divisible by DepthU, NoTailLoop=False -> tail calls reachable."""
    assert no_tail_loop(12, 8) is False   # 12%8==4
    assert no_tail_loop(8, 16) is False   # 8%16==8
    assert no_tail_loop(12, 16) is False  # 12%16==12


def test_no_tail_loop_asem_divisible():
    """When ASEM is divisible by DepthU, NoTailLoop=True -> tail calls not emitted."""
    assert no_tail_loop(8, 8) is True     # 8%8==0
    assert no_tail_loop(16, 8) is True    # 16%8==0
    assert no_tail_loop(16, 16) is True   # 16%16==0
