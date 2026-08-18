################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterisation test -- branch_id 4108a06779b564c9652562b2d8b1fd277e5ad963

Predicate : doReadB   (local variable at entry to `if doReadB:`)
Site      : Tensile/KernelWriter.py:4145  (inside _loopBody, inner iui loop)
Solver    : z3 4.16.0 -- SAT (solver-backed-under-assumptions)

Assignment chain (KernelWriter.py lines 4023-4063)
---------------------------------------------------
  4023: hasLiveLdsData = kernel["PrefetchGlobalRead"]          # != 0 means live
  4029: doReadB = (u < kernel["LoopIters"]/numIterPerCoalescedReadB
                     - numItersPLR)                             # true-div threshold
  4036: if ForceUnrollSubIter: doReadB = 1 if u==0 else 0      # override
  4046: doReadB = doReadB or (hasLiveLdsData and doNext)        # next-loop OR arm
  4063: doReadB = doReadB and                                   # inner-unroll gate
              iui*numReadsIterCoalescedB < kernel["InnerUnroll"]

Predicate at 4145:
  if doReadB:     # True -> enter local-read-B block; False -> skip

Witnesses (5/5 verified in-container; all 5 match _eval_doReadB):
  true[0]  : u=1, iui=0, LoopIters=2, InnerUnroll=1, PGR=1, FUSI=False, doNext=False
             -> threshold arm: 1 < 2/1 - 0 = 2.0, gate: 0*1 < 1  => doReadB=True
  true[1]  : same but FUSI=True, doNext=True
             -> FUSI zeroes base (u!=0), next-loop OR arm (PGR!=0 and doNext=True) fires
             -> gate: 0*1 < 1  => doReadB=True
  false[0] : u=1, iui=1, FUSI=True, doNext=False
             -> base zeroed by FUSI+u!=0, OR arm False; gate 1*1 < 1 also False
  false[1] : u=1, iui=1, FUSI=False, doNext=False
             -> threshold True BUT gate 1*1 < 1 is False => doReadB=False
  false[2] : same as false[0] -- FUSI+u!=0 primary false (harmless duplicate)

Classification: solver-backed-under-assumptions
  numIterPerCoalescedReadB, numReadsIterCoalescedB, numItersPLR, ForceUnrollSubIter, doNext
  are derived kernel-setup state; witnesses realise them via:
    UnrollMajorLDSB=False, PrefetchLocalRead=0, HalfPLR=0 => nIPCRB=nRICB=1, numItersPLR=0
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper -- faithful Python reimplementation of the 4029/4036/4046/4063
# assignment chain.  No GPU.  No rocisa.
# ---------------------------------------------------------------------------

def _eval_doReadB(
    u: int,
    iui: int,
    LoopIters: int,
    InnerUnroll: int,
    PrefetchGlobalRead: int,
    numIterPerCoalescedReadB: int,
    numItersPLR: int,
    numReadsIterCoalescedB: int,
    ForceUnrollSubIter: bool,
    doNext: bool,
) -> bool:
    """
    Mirror of KernelWriter.py:4029-4063 assignment chain for doReadB.

    Parameters mirror the z3 declarations from the Solve fragment.
    All derived params are passed in directly (solver-backed-under-assumptions).
    """
    # 4023
    hasLiveLdsData = PrefetchGlobalRead  # truthy when != 0

    # 4029 (Python 3 true-division)
    doReadB = u < LoopIters / numIterPerCoalescedReadB - numItersPLR

    # 4036
    if ForceUnrollSubIter:
        doReadB = 1 if u == 0 else 0

    # 4046
    doReadB = doReadB or (hasLiveLdsData and doNext)

    # 4063 (inner-unroll gate applied per iui)
    doReadB = doReadB and iui * numReadsIterCoalescedB < InnerUnroll

    return bool(doReadB)


# ---------------------------------------------------------------------------
# TRUE witnesses
# ---------------------------------------------------------------------------

def test_doReadB_true_normal_threshold_arm():
    """
    True witness[0]: u=1 passes the threshold compare; inner-unroll gate also passes.
    FUSI=False so the ForceUnrollSubIter override is not taken.
    doNext=False so the OR arm is irrelevant -- carried by the 4029 threshold alone.
    Branch TAKEN.
    """
    result = _eval_doReadB(
        u=1, iui=0,
        LoopIters=2, InnerUnroll=1,
        PrefetchGlobalRead=1,
        numIterPerCoalescedReadB=1,
        numItersPLR=0,
        numReadsIterCoalescedB=1,
        ForceUnrollSubIter=False,
        doNext=False,
    )
    assert result is True


def test_doReadB_true_nextloop_or_arm():
    """
    True witness[1]: ForceUnrollSubIter=True and u=1 zeroes the base (doReadB=0),
    but hasLiveLdsData(PGR=1) and doNext=True fires the OR arm at line 4046.
    Inner-unroll gate (iui=0) passes.  Branch TAKEN via next-loop prefetch arm.

    Load-bearing sensitivity: same inputs with doNext=False -> False.
    """
    result = _eval_doReadB(
        u=1, iui=0,
        LoopIters=2, InnerUnroll=1,
        PrefetchGlobalRead=1,
        numIterPerCoalescedReadB=1,
        numItersPLR=0,
        numReadsIterCoalescedB=1,
        ForceUnrollSubIter=True,
        doNext=True,
    )
    assert result is True


def test_doReadB_true_nextloop_or_arm_sensitivity_doNext_false():
    """
    Sensitivity check for witness[1]: flipping doNext to False turns the result False,
    confirming the OR arm (hasLiveLdsData and doNext) is the sole True carrier
    when ForceUnrollSubIter=True and u!=0.
    """
    result = _eval_doReadB(
        u=1, iui=0,
        LoopIters=2, InnerUnroll=1,
        PrefetchGlobalRead=1,
        numIterPerCoalescedReadB=1,
        numItersPLR=0,
        numReadsIterCoalescedB=1,
        ForceUnrollSubIter=True,
        doNext=False,       # <-- flipped from witness[1]
    )
    assert result is False


# ---------------------------------------------------------------------------
# FALSE witnesses
# ---------------------------------------------------------------------------

def test_doReadB_false_inner_unroll_gate_discriminator():
    """
    False witness[1]: threshold arm is True (u=1 < 2.0), FUSI=False so no override,
    but inner-unroll gate at line 4063 -- iui=1 * numReadsIterCoalescedB=1 < InnerUnroll=1
    evaluates to 1 < 1 = False, zeroing doReadB.  Branch SKIPPED.

    This isolates the 4063 gate as a load-bearing clause.
    """
    result = _eval_doReadB(
        u=1, iui=1,
        LoopIters=2, InnerUnroll=1,
        PrefetchGlobalRead=1,
        numIterPerCoalescedReadB=1,
        numItersPLR=0,
        numReadsIterCoalescedB=1,
        ForceUnrollSubIter=False,
        doNext=False,
    )
    assert result is False


def test_doReadB_false_sensitivity_iui_flip():
    """
    Sensitivity check for witness[1]: same inputs but iui=0 -> gate 0*1 < 1 = True.
    Result flips to True, confirming iui is the discriminating variable for this witness.
    """
    result = _eval_doReadB(
        u=1, iui=0,      # <-- iui=0 instead of 1
        LoopIters=2, InnerUnroll=1,
        PrefetchGlobalRead=1,
        numIterPerCoalescedReadB=1,
        numItersPLR=0,
        numReadsIterCoalescedB=1,
        ForceUnrollSubIter=False,
        doNext=False,
    )
    assert result is True


def test_doReadB_false_forceunrollsubiter_and_no_next():
    """
    False witness[0/2]: FUSI=True and u=1 != 0 zeroes the base at 4036.
    No OR arm (doNext=False).  Gate also fails (iui=1 * 1 < 1 is False).
    Branch SKIPPED by both paths simultaneously.
    """
    result = _eval_doReadB(
        u=1, iui=1,
        LoopIters=2, InnerUnroll=1,
        PrefetchGlobalRead=1,
        numIterPerCoalescedReadB=1,
        numItersPLR=0,
        numReadsIterCoalescedB=1,
        ForceUnrollSubIter=True,
        doNext=False,
    )
    assert result is False


# ---------------------------------------------------------------------------
# Multi-clause interaction table
# Pins the five z3-verified witnesses as a parametrized truth table.
# ---------------------------------------------------------------------------

_WITNESSES = [
    # (u, iui, LI, IU, PGR, nIPCRB, nItersPLR, nRICB, FUSI,  doNext, expected, label)
    (1, 0, 2, 1, 1, 1, 0, 1, False, False, True,  "true[0]:normal_threshold"),
    (1, 0, 2, 1, 1, 1, 0, 1, True,  True,  True,  "true[1]:nextloop_or_arm"),
    (1, 1, 2, 1, 1, 1, 0, 1, True,  False, False, "false[0]:fusi_no_next"),
    (1, 1, 2, 1, 1, 1, 0, 1, False, False, False, "false[1]:inner_unroll_gate"),
    (1, 1, 2, 1, 1, 1, 0, 1, True,  False, False, "false[2]:fusi_dup"),
]


@pytest.mark.parametrize(
    "u,iui,LI,IU,PGR,nIPCRB,nItersPLR,nRICB,FUSI,doNext,expected,label",
    _WITNESSES,
    ids=[w[-1] for w in _WITNESSES],
)
def test_doReadB_witness_table(u, iui, LI, IU, PGR, nIPCRB, nItersPLR, nRICB, FUSI, doNext, expected, label):
    """
    Pin all five z3-verified witnesses from the Solve fragment (branch_id
    4108a06779b564c9652562b2d8b1fd277e5ad963).  5/5 verified in-container.
    """
    result = _eval_doReadB(
        u=u, iui=iui,
        LoopIters=LI, InnerUnroll=IU,
        PrefetchGlobalRead=PGR,
        numIterPerCoalescedReadB=nIPCRB,
        numItersPLR=nItersPLR,
        numReadsIterCoalescedB=nRICB,
        ForceUnrollSubIter=FUSI,
        doNext=doNext,
    )
    assert result is expected, (
        f"Witness {label!r}: expected doReadB={expected}, got {result}. "
        "Re-verify the assignment chain at KernelWriter.py:4029-4063 "
        "(branch_id 4108a06779b564c9652562b2d8b1fd277e5ad963)."
    )
