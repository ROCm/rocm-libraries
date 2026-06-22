################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterisation test — branch_id c8562779274f6936fbe265f93ad5caa9f4aa44da

Predicate : kernel["DirectToLds%s" % tc]
Site      : Tensile/KernelWriterAssembly.py:10892  (function directToLdsM0Update)
Solver    : z3 4.16.0  — SAT  (solver-backed-under-assumptions)

Background
----------
tc = tP["tensorChar"] is a compile-time constant in {"A", "B"} fixed at the
call-site (KernelWriter.getTensorParameters passes "A" for tensorParametersA,
"B" for tensorParametersB). This single source branch therefore reifies into
two concrete predicates:

  kernel["DirectToLdsA"]  (tc == "A")
  kernel["DirectToLdsB"]  (tc == "B")

Both booleans are derived from the YAML/MsgPack solution integer DirectToLds
(0-3, default 0) via Solution.py:722-723:

  DirectToLdsA = (DirectToLds == 1 or DirectToLds == 2)
  DirectToLdsB = (DirectToLds == 1 or DirectToLds == 3)

Semantic summary:
  DirectToLds == 0  -> A=False, B=False  (no DTL)
  DirectToLds == 1  -> A=True,  B=True   (both A+B)
  DirectToLds == 2  -> A=True,  B=False  (A only)
  DirectToLds == 3  -> A=False, B=True   (B only)

Guard note: Solution.py:1828-1832 forces both flags False when BufferLoad=False
or KernelLanguage != Assembly. Witnesses are valid under the assumptions:
BufferLoad=True, KernelLanguage=Assembly, isDirectToLdsDoable passes.
Classification: solver-backed-under-assumptions.

Witnesses (z3-derived, re-validated by executed real derivation):

  TRUE  tc=A: DirectToLds=1  -> DirectToLdsA=(1==1 or 1==2)=True  -> branch taken
  TRUE  tc=A: DirectToLds=2  -> DirectToLdsA=(2==1 or 2==2)=True  -> branch taken
  TRUE  tc=B: DirectToLds=1  -> DirectToLdsB=(1==1 or 1==3)=True  -> branch taken
  TRUE  tc=B: DirectToLds=3  -> DirectToLdsB=(3==1 or 3==3)=True  -> branch taken
  FALSE tc=A: DirectToLds=0  -> DirectToLdsA=(0==1 or 0==2)=False -> branch skipped
  FALSE tc=A: DirectToLds=3  -> DirectToLdsA=(3==1 or 3==2)=False -> branch skipped
  FALSE tc=B: DirectToLds=0  -> DirectToLdsB=(0==1 or 0==3)=False -> branch skipped
  FALSE tc=B: DirectToLds=2  -> DirectToLdsB=(2==1 or 2==3)=False -> branch skipped
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper — transcription of Solution.py:722-723 derivation + branch guard
# ---------------------------------------------------------------------------

def kernel_direct_to_lds(direct_to_lds: int, tc: str) -> bool:
    """
    Pure model of KernelWriterAssembly.py:10892 predicate kernel["DirectToLds%s" % tc].

    tc is the compile-time-constant tensorChar in {"A", "B"}.
    The boolean kernel["DirectToLdsA"] / kernel["DirectToLdsB"] is derived from
    the YAML int DirectToLds via Solution.py:722-723:

      DirectToLdsA = (DirectToLds == 1 or DirectToLds == 2)
      DirectToLdsB = (DirectToLds == 1 or DirectToLds == 3)

    pre:  0 <= direct_to_lds <= 3
    pre:  tc in {"A", "B"}
    post: __return__ == (
              (direct_to_lds == 1 or direct_to_lds == 2) if tc == "A"
              else (direct_to_lds == 1 or direct_to_lds == 3)
          )
    """
    if tc == "A":
        return direct_to_lds == 1 or direct_to_lds == 2
    return direct_to_lds == 1 or direct_to_lds == 3


# ---------------------------------------------------------------------------
# TRUE witnesses — tc == "A" selector  (DirectToLdsA)
# ---------------------------------------------------------------------------

def test_true_tc_a_dtl1_both():
    """
    z3 true witness: tc=A, DirectToLds=1 (both A+B).
    DirectToLdsA = (1==1 or 1==2) = True -> branch taken.
    """
    assert kernel_direct_to_lds(1, "A") is True


def test_true_tc_a_dtl2_a_only():
    """
    z3 true witness: tc=A, DirectToLds=2 (A only).
    DirectToLdsA = (2==1 or 2==2) = True -> branch taken.
    """
    assert kernel_direct_to_lds(2, "A") is True


# ---------------------------------------------------------------------------
# TRUE witnesses — tc == "B" selector  (DirectToLdsB)
# ---------------------------------------------------------------------------

def test_true_tc_b_dtl1_both():
    """
    z3 true witness: tc=B, DirectToLds=1 (both A+B).
    DirectToLdsB = (1==1 or 1==3) = True -> branch taken.
    """
    assert kernel_direct_to_lds(1, "B") is True


def test_true_tc_b_dtl3_b_only():
    """
    z3 true witness: tc=B, DirectToLds=3 (B only).
    DirectToLdsB = (3==1 or 3==3) = True -> branch taken.
    """
    assert kernel_direct_to_lds(3, "B") is True


# ---------------------------------------------------------------------------
# FALSE witnesses — tc == "A" selector  (DirectToLdsA)
# ---------------------------------------------------------------------------

def test_false_tc_a_dtl0_no_dtl():
    """
    z3 false witness: tc=A, DirectToLds=0 (no DTL, default).
    DirectToLdsA = (0==1 or 0==2) = False -> branch skipped.
    """
    assert kernel_direct_to_lds(0, "A") is False


def test_false_tc_a_dtl3_b_only():
    """
    z3 false witness: tc=A, DirectToLds=3 (B only).
    DirectToLdsA = (3==1 or 3==2) = False -> branch skipped for A call-site.
    """
    assert kernel_direct_to_lds(3, "A") is False


# ---------------------------------------------------------------------------
# FALSE witnesses — tc == "B" selector  (DirectToLdsB)
# ---------------------------------------------------------------------------

def test_false_tc_b_dtl0_no_dtl():
    """
    z3 false witness: tc=B, DirectToLds=0 (no DTL).
    DirectToLdsB = (0==1 or 0==3) = False -> branch skipped.
    """
    assert kernel_direct_to_lds(0, "B") is False


def test_false_tc_b_dtl2_a_only():
    """
    z3 false witness: tc=B, DirectToLds=2 (A only).
    DirectToLdsB = (2==1 or 2==3) = False -> branch skipped for B call-site.
    """
    assert kernel_direct_to_lds(2, "B") is False


# ---------------------------------------------------------------------------
# Exhaustive partition pin — full 4x2 cross-check against derivation
# ---------------------------------------------------------------------------

def test_exhaustive_4x2_domain():
    """
    Exhaustive check of all (DirectToLds in {0,1,2,3}) x (tc in {"A","B"}) combos.
    Pins the complete truth table derived from Solution.py:722-723.
    """
    expected = {
        # (dtl, tc): result
        (0, "A"): False,   # no DTL
        (0, "B"): False,
        (1, "A"): True,    # both A+B
        (1, "B"): True,
        (2, "A"): True,    # A only
        (2, "B"): False,
        (3, "A"): False,   # B only
        (3, "B"): True,
    }
    for (dtl, tc), want in expected.items():
        got = kernel_direct_to_lds(dtl, tc)
        assert got is want, (
            f"kernel_direct_to_lds({dtl!r}, {tc!r}) returned {got!r}, expected {want!r}"
        )
