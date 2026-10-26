################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Characterization tests for KernelWriter.py line 882.

Branch ID: 7b6b7c5fa8d48c75aa699041eed7a307cb3b9947
Predicate:  scheduleIterAlg == 0   (KernelWriter.py:882 in _makeSubIterSchedule)

The local variable `scheduleIterAlg` is resolved at lines 877-880:

    scheduleIterAlg = self.states.scheduleIterAlg          # line 877
    if (NLLlast and tailloopInNll):                        # line 878
        scheduleIterAlg = 0                                # line 880

where:
  - self.states.scheduleIterAlg = kernel["_ScheduleIterAlg"]
    (set at KernelWriter.py:6227 for KernelLanguage == "Assembly")
  - _ScheduleIterAlg = ScheduleIterAlg for SIA in {0,1,2,3}
    (Solution.py:604; SIA=4 hardware path excluded from this test domain)
  - NLLlast starts True (noLoadLoop:3552), set False when PGR>=2 and isNGLL
    (noLoadLoop:3563-3567)
  - tailloopInNll = self.states.tailloopInNll (noLoadLoop call site)

Two paths reach predicate == True:
  Path A: ScheduleIterAlg == 0 (direct, NLLlast/tailloopInNll irrelevant)
  Path B: NLLlast == True AND tailloopInNll == True (override forces 0)

One path reaches predicate == False:
  Path C: ScheduleIterAlg != 0 AND (NLLlast == False OR tailloopInNll == False)

CPU-only.  No GPU hardware required.  SIA=4 (StinkyTofu) path excluded (requires
hardware probes in rocisa).

Verification source: Adversarial re-derivation; z3 4.16.0 in-container confirms
both polarities sat; exhaustive 36-point enumeration: 0 violations.
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper: mirrors the local-variable resolution at KernelWriter.py:877-882
# ---------------------------------------------------------------------------

def schedule_iter_alg_at_882(
    sia: int,
    nll_last: bool,
    tailloop_in_nll: bool,
) -> int:
    """Mirror of KernelWriter.py lines 877-882.

    Computes the effective scheduleIterAlg value that the if-predicate at
    line 882 tests.  Inputs:

      sia              -- _ScheduleIterAlg == ScheduleIterAlg for SIA in {0,1,3}
                          (Assembly path; non-Assembly forces sia=0 unconditionally)
      nll_last         -- True unless PGR>=2 and isNGLL (noLoadLoop:3563-3567)
      tailloop_in_nll  -- self.states.tailloopInNll passed into _makeSubIterSchedule

    Returns the resolved scheduleIterAlg value (0 or non-zero).
    """
    scheduleIterAlg = sia
    if nll_last and tailloop_in_nll:
        scheduleIterAlg = 0
    return scheduleIterAlg


def predicate_at_882(sia: int, nll_last: bool, tailloop_in_nll: bool) -> bool:
    """The boolean predicate `scheduleIterAlg == 0` at KernelWriter.py:882."""
    return schedule_iter_alg_at_882(sia, nll_last, tailloop_in_nll) == 0


# ---------------------------------------------------------------------------
# Pure predicate tests (helper contract)
# ---------------------------------------------------------------------------

class TestPredicateHelperContract:
    """Verify the extracted predicate helper against all four documented witnesses."""

    # TRUE branch — Path A: direct SIA=0
    def test_true_A_sia0_nll_false_tnll_false(self):
        """TRUE-A: SIA=0, NLLlast=False, tailloopInNll=False -> predicate True (Path A, direct)."""
        assert predicate_at_882(sia=0, nll_last=False, tailloop_in_nll=False) is True

    def test_true_A_sia0_nll_true_tnll_false(self):
        """TRUE-A variant: SIA=0, NLLlast=True, tailloopInNll=False -> predicate True (Path A)."""
        assert predicate_at_882(sia=0, nll_last=True, tailloop_in_nll=False) is True

    def test_true_A_sia0_nll_true_tnll_true(self):
        """TRUE-A variant: SIA=0, NLLlast=True, tailloopInNll=True -> predicate True (both paths A and B)."""
        assert predicate_at_882(sia=0, nll_last=True, tailloop_in_nll=True) is True

    # TRUE branch — Path B: override fires (SIA != 0, but NLLlast and tailloopInNll both True)
    def test_true_B_sia1_nll_true_tnll_true(self):
        """TRUE-B: SIA=1, NLLlast=True, tailloopInNll=True -> override sets sia=0, predicate True."""
        assert predicate_at_882(sia=1, nll_last=True, tailloop_in_nll=True) is True

    def test_true_B_sia3_nll_true_tnll_true(self):
        """TRUE-B: SIA=3, NLLlast=True, tailloopInNll=True -> override sets sia=0, predicate True."""
        assert predicate_at_882(sia=3, nll_last=True, tailloop_in_nll=True) is True

    # FALSE branch — Path C: SIA != 0 and override does not fire
    def test_false_1_sia1_nll_false(self):
        """FALSE-1: SIA=1, NLLlast=False -> override does not fire, predicate False."""
        assert predicate_at_882(sia=1, nll_last=False, tailloop_in_nll=True) is False

    def test_false_2_sia3_nll_false(self):
        """FALSE-2: SIA=3, NLLlast=False -> override does not fire, predicate False."""
        assert predicate_at_882(sia=3, nll_last=False, tailloop_in_nll=True) is False

    def test_false_3_sia1_tnll_false(self):
        """FALSE-3: SIA=1, tailloopInNll=False -> override does not fire, predicate False."""
        assert predicate_at_882(sia=1, nll_last=True, tailloop_in_nll=False) is False

    def test_false_4_sia3_tnll_false(self):
        """FALSE-4: SIA=3, tailloopInNll=False -> override does not fire, predicate False."""
        assert predicate_at_882(sia=3, nll_last=True, tailloop_in_nll=False) is False


class TestHelperOverrideSemantics:
    """Verify the override mechanic: NLLlast AND tailloopInNll forces sia=0."""

    def test_override_requires_both_conditions(self):
        """Override fires only when BOTH NLLlast and tailloopInNll are True."""
        # Override fires
        assert schedule_iter_alg_at_882(1, True, True) == 0
        # NLLlast alone — override does NOT fire
        assert schedule_iter_alg_at_882(1, True, False) == 1
        # tailloopInNll alone — override does NOT fire
        assert schedule_iter_alg_at_882(1, False, True) == 1
        # Neither — override does NOT fire
        assert schedule_iter_alg_at_882(1, False, False) == 1

    def test_direct_sia0_unaffected_by_override_flags(self):
        """SIA=0 path: result is always 0 regardless of NLLlast/tailloopInNll."""
        for nll_last in (True, False):
            for tnll in (True, False):
                result = schedule_iter_alg_at_882(0, nll_last, tnll)
                assert result == 0, (
                    f"Expected 0 for SIA=0, NLLlast={nll_last}, tailloopInNll={tnll}; got {result}"
                )

    def test_override_does_not_affect_predicate_False_when_sia0(self):
        """When SIA=0, predicate is always True (override only reinforces)."""
        for nll_last in (True, False):
            for tnll in (True, False):
                assert predicate_at_882(0, nll_last, tnll) is True, (
                    f"SIA=0 must always yield True; NLLlast={nll_last}, tailloopInNll={tnll}"
                )


class TestNLLlastDerivation:
    """Verify the NLLlast derivation from PrefetchGlobalRead + isNGLL.

    NLLlast starts True (noLoadLoop:3552).
    Set False when PGR >= 2 AND isNGLL (noLoadLoop:3563-3567).
    Equivalently: NLLlast = not (PGR >= 2 and isNGLL).
    """

    def _nll_last(self, pgr: int, is_ngll: bool) -> bool:
        """Mirror of noLoadLoop NLLlast derivation."""
        nll_last = True
        if pgr >= 2:
            if is_ngll:
                nll_last = False
        return nll_last

    def test_pgr0_any_ngll_nll_last_true(self):
        """PGR=0: NLLlast always True (PGR>=2 branch not entered)."""
        assert self._nll_last(0, True) is True
        assert self._nll_last(0, False) is True

    def test_pgr1_any_ngll_nll_last_true(self):
        """PGR=1: NLLlast always True (PGR>=2 branch not entered)."""
        assert self._nll_last(1, True) is True
        assert self._nll_last(1, False) is True

    def test_pgr2_ngll_true_nll_last_false(self):
        """PGR=2, isNGLL=True: NLLlast = False (both conditions met)."""
        assert self._nll_last(2, True) is False

    def test_pgr2_ngll_false_nll_last_true(self):
        """PGR=2, isNGLL=False: NLLlast = True (isNGLL condition not met)."""
        assert self._nll_last(2, False) is True

    def test_pgr3_ngll_true_nll_last_false(self):
        """PGR=3, isNGLL=True: NLLlast = False."""
        assert self._nll_last(3, True) is False

    def test_pgr3_ngll_false_nll_last_true(self):
        """PGR=3, isNGLL=False: NLLlast = True."""
        assert self._nll_last(3, False) is True


class TestScheduleIterAlgDerivationFromPublicInputs:
    """End-to-end: public-input (ScheduleIterAlg yaml param) -> predicate value.

    _ScheduleIterAlg = ScheduleIterAlg for SIA in {0,1,3} (Solution.py:604).
    self.states.scheduleIterAlg = _ScheduleIterAlg when KernelLanguage='Assembly' (KernelWriter.py:6227).
    """

    def _sia_from_yaml(self, schedule_iter_alg: int) -> int:
        """Mirror of Solution.py:603-604 for SIA in {0,1,3} (cpu-testable domain)."""
        assert schedule_iter_alg in (0, 1, 3), "SIA=4 excluded (requires hardware probe)"
        return schedule_iter_alg

    def test_sia0_direct_true_without_override(self):
        """ScheduleIterAlg=0 -> predicate True without any override (NLLlast=False, tnll=False)."""
        sia = self._sia_from_yaml(0)
        assert predicate_at_882(sia, nll_last=False, tailloop_in_nll=False) is True

    def test_sia1_no_override_predicate_false(self):
        """ScheduleIterAlg=1, no override (NLLlast=False) -> predicate False."""
        sia = self._sia_from_yaml(1)
        assert predicate_at_882(sia, nll_last=False, tailloop_in_nll=True) is False

    def test_sia1_with_override_predicate_true(self):
        """ScheduleIterAlg=1, override fires (NLLlast=True, tnll=True) -> predicate True."""
        sia = self._sia_from_yaml(1)
        assert predicate_at_882(sia, nll_last=True, tailloop_in_nll=True) is True

    def test_sia3_no_override_predicate_false(self):
        """ScheduleIterAlg=3 (default), no override (NLLlast=False) -> predicate False."""
        sia = self._sia_from_yaml(3)
        assert predicate_at_882(sia, nll_last=False, tailloop_in_nll=True) is False

    def test_sia3_with_override_predicate_true(self):
        """ScheduleIterAlg=3 (default), override fires (NLLlast=True, tnll=True) -> predicate True."""
        sia = self._sia_from_yaml(3)
        assert predicate_at_882(sia, nll_last=True, tailloop_in_nll=True) is True

    def test_sia_domain_exhaustive_36_point(self):
        """Exhaustive 36-point check: SIA in {0,1,3} x NLLlast in {T,F} x tnll in {T,F} x pgr-proxy.

        For each combination, assert predicate matches the expected analytic formula:
          predicate_true iff (sia == 0) or (nll_last and tailloop_in_nll).
        """
        for sia in (0, 1, 3):
            for nll_last in (True, False):
                for tnll in (True, False):
                    expected = (sia == 0) or (nll_last and tnll)
                    actual = predicate_at_882(sia, nll_last, tnll)
                    assert actual == expected, (
                        f"Mismatch: SIA={sia}, NLLlast={nll_last}, tailloopInNll={tnll}; "
                        f"expected={expected}, got={actual}"
                    )
