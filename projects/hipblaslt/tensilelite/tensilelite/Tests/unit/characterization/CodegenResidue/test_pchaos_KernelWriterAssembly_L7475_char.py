################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterisation test — branch_id e85e407ea3082d913528ced06c6d634c43e5ef22

Predicate : tailLoop
Site      : Tensile/KernelWriterAssembly.py:7475  (function closeLoop)
Solver    : z3 4.16.0 — SAT (fully-static)

Background
----------
Inside closeLoop() (def at line 7458), loopIdx is a plain integer parameter
(never derived from kernel[] within the function body):

  line 7472: tailLoop = loopIdx < 0
  line 7475: if tailLoop:

loopIdx values observed at all call-sites:
  -2  — tailloopInNll path: KernelWriter.py:3532, gated by
         kernel["TailloopInNll"] / useTailloopInNll
  -1  — normal tail-loop path: KernelWriter.py:5818/5820/5856, gated by
         `if not kernel["NoTailLoop"]`
   0  — normal unroll close: KernelWriter.py:4399/4416/4420/4424,
         loopIdx = self.states.unrollIdx (always >= 0)
  >=1 — multi-summation unroll close (unrollIdx > 0)

The predicate reduces to a pure integer comparison:
  tailLoop == (loopIdx < 0)

Classification: fully-static. Both polarities confirmed SAT by z3 and re-verified
manually: loopIdx=-2 -> True, loopIdx=-1 -> True, loopIdx=0 -> False, loopIdx=1 -> False.

Witnesses
---------
  TRUE   loopIdx=-2   tailloopInNll call-site (TailloopInNll/NoTailLoop=True)
  TRUE   loopIdx=-1   tail-loop call-site     (NoTailLoop=False)
  FALSE  loopIdx= 0   unroll close            (unrollIdx=0, single summation)
  FALSE  loopIdx= 1   unroll close            (unrollIdx=1, multi-summation)
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper — transcription of KernelWriterAssembly.py:7472 predicate
# ---------------------------------------------------------------------------

def tail_loop_taken(loopIdx: int) -> bool:
    """
    Pure model of KernelWriterAssembly.py:7475 predicate inside closeLoop.

    tailLoop = (loopIdx < 0)

    post: __return__ == (loopIdx < 0)
    """
    tailLoop = loopIdx < 0
    return tailLoop


# ---------------------------------------------------------------------------
# TRUE witnesses
# ---------------------------------------------------------------------------

def test_true_loopidx_neg2_tailloop_in_nll():
    """
    z3 true witness: loopIdx=-2  (tailloopInNll call-site).
    KernelWriter.py:3532 passes -2 when useTailloopInNll/TailloopInNll is set.
    tailLoop = (-2 < 0) = True -> branch at line 7475 taken.
    """
    assert tail_loop_taken(-2) is True


def test_true_loopidx_neg1_normal_tail():
    """
    z3 true witness: loopIdx=-1  (normal tail-loop call-site).
    KernelWriter.py:5818/5820/5856 passes -1 when kernel["NoTailLoop"] is False.
    tailLoop = (-1 < 0) = True -> branch at line 7475 taken.
    """
    assert tail_loop_taken(-1) is True


# ---------------------------------------------------------------------------
# FALSE witnesses
# ---------------------------------------------------------------------------

def test_false_loopidx_0_unroll_close():
    """
    z3 false witness: loopIdx=0  (normal unroll close, single summation).
    self.states.unrollIdx >= 0 at all unroll close call-sites.
    tailLoop = (0 < 0) = False -> branch at line 7475 skipped.
    """
    assert tail_loop_taken(0) is False


def test_false_loopidx_1_multi_summation():
    """
    z3 false witness: loopIdx=1  (unroll close, multi-summation, unrollIdx > 0).
    tailLoop = (1 < 0) = False -> branch at line 7475 skipped.
    """
    assert tail_loop_taken(1) is False


# ---------------------------------------------------------------------------
# Extended domain pin — full integer sign partition
# ---------------------------------------------------------------------------

def test_all_negative_indices_true():
    """
    All negative loopIdx values produce tailLoop=True.
    Covers the full true-branch domain.
    """
    for idx in [-10, -3, -2, -1]:
        assert tail_loop_taken(idx) is True, f"tail_loop_taken({idx}) should be True"


def test_all_nonnegative_indices_false():
    """
    All non-negative loopIdx values produce tailLoop=False.
    Covers the full false-branch domain (unrollIdx is always >= 0).
    """
    for idx in [0, 1, 2, 5, 10]:
        assert tail_loop_taken(idx) is False, f"tail_loop_taken({idx}) should be False"


# ---------------------------------------------------------------------------
# Composition model: YAML gates -> call-site loopIdx -> tailLoop branch
# ---------------------------------------------------------------------------

def loopidx_to_tail(is_normal_unroll_close: bool, tailloop_in_nll: bool, no_tail_loop: bool) -> bool:
    """
    Composition: YAML kernel[] gates -> call-site loopIdx -> tailLoop branch result.

    Mirrors the call-site selection logic in KernelWriter.py:

      is_normal_unroll_close=True  -> loopIdx = self.states.unrollIdx (>=0) -> False
      tailloop_in_nll=True         -> loopIdx = -2                          -> True
      not no_tail_loop             -> loopIdx = -1                           -> True
      (else)                       -> loopIdx = 0 (fallback)                 -> False

    post: __return__ == ((not is_normal_unroll_close) and
                         (tailloop_in_nll or (not no_tail_loop)))
    """
    if is_normal_unroll_close:
        loopIdx = 0
    elif tailloop_in_nll:
        loopIdx = -2
    elif not no_tail_loop:
        loopIdx = -1
    else:
        loopIdx = 0
    return loopIdx < 0


def test_composition_tailloop_in_nll_path():
    """TailloopInNll path: is_normal_unroll=False, tailloop_in_nll=True -> True."""
    assert loopidx_to_tail(False, True, True) is True


def test_composition_normal_tail_path():
    """Normal tail path: is_normal_unroll=False, tailloop_in_nll=False, no_tail=False -> True."""
    assert loopidx_to_tail(False, False, False) is True


def test_composition_unroll_close_path():
    """Normal unroll close: is_normal_unroll=True -> always False."""
    assert loopidx_to_tail(True, False, False) is False
    assert loopidx_to_tail(True, True, True) is False


def test_composition_no_tail_loop_skips():
    """NoTailLoop=True and not tailloopInNll and not unroll: loopIdx=0 -> False."""
    assert loopidx_to_tail(False, False, True) is False
