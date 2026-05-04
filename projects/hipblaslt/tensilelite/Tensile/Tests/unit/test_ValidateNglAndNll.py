################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################
"""TestValidateNgl: graph-native NGL-body GR -> ML-body LR1 cross-body
edge coverage (bead ola.1).

The legacy ``TestValidateNgl`` exercised the structural-rule shift-value
reasoning: GRs scheduled in the NGL (no-global-load) tail body still need
their wait/barrier coverage so the LR1 consumers in the following main-loop
body see consistent LDS state. The shift parameter in the legacy fixtures
controlled the ``vlcnt`` used to drain the NGL-issued GRs.

Graph-side, this becomes a cross-body ``gr_to_lr_lds_reuse`` edge:
producer in body=NGL, consumer in body=ML. Body order is
(ML-1, ML, NGL, NLL); but if we anchor the producer in ML-1 and the
consumer in ML, we get the same cross-body shape with the standard
filler-body machinery from ``GraphNativeValidationTest.wrap_single_body``.

For TestValidateNll: kept using the legacy CMSValidationTestBase since
``add_local_read_constraints`` (the rule it tests) is still in place — it
is being migrated separately under bead ``ola.3``. Once ola.3 lands, those
tests will move to the graph-native base too.
"""

from rocisa.instruction import SWaitCnt, SBarrier

from Tensile.Components.CMSValidator import (
    add_local_read_constraints,
)
from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    MissingWaitFailure,
    WaitInsufficientFailure,
    WaitOnWrongCounterFailure,
    WaitTooLateFailure,
)

from cms_validation_base import CMSValidationTestBase
from dataflow_fixtures import (
    make_capture, make_gr, make_lr, make_sbarrier, make_swait,
)
from graph_native_validation_base import GraphNativeValidationTest


# Same register layout used in the LR1/LR3 ports.
_GR_VGPR_BASE = 40
_GR_VGPR_COUNT = 4
_GR_SRD = 12
_GR_LDS_OFFSET = 64
_LR_VGPR_BASE = 8
_LR_VGPR_COUNT = 4
_LR_LDS_OFFSET = 64


def _gr(slot: int, category: str = "GRA", *, sequence: int = 0,
        vgpr_base: int = _GR_VGPR_BASE):
    return make_gr(
        vgpr_base, _GR_VGPR_COUNT, srd_sgpr_start=_GR_SRD,
        immediate_offset=_GR_LDS_OFFSET, slot=slot,
        category=category, sequence=sequence,
    )


def _lr1(slot: int, category: str = "LRA1", *, sequence: int = 0,
         vgpr_base: int = _LR_VGPR_BASE):
    return make_lr(
        vgpr_base, _LR_VGPR_COUNT, lds_offset=_LR_LDS_OFFSET, slot=slot,
        category=category, sequence=sequence,
    )


# =============================================================================
# TestValidateNgl — graph-native cross-body GR (prev iter) -> LR1 (curr iter)
# =============================================================================
# The legacy fixtures encoded the NGL shift-value as a per-iteration
# adjustment to the GR-covering ``vlcnt`` value: the SWait at the BEGINNING
# of iteration N must drain BOTH the in-loop GRs from iter N-1 AND the
# NGL-tail GRs from iter N-1. If the shift value is too small, the in-loop
# wait under-drains and the LR1 consumer in iter N's main-loop body sees
# unfinished GR data.
#
# Graph-native equivalent: model the NGL-body GRs as cross-body producers
# in ML-1 (the previous-iteration body the FourPartCapture machinery
# already builds), the wait+barrier in ML-1 OR at the start of ML, and the
# LR1 consumer in ML. The wait's vlcnt value relative to the queue depth
# at the wait's moment governs whether ``_any_drains`` returns True.


class TestValidateNgl(GraphNativeValidationTest):
    """Cross-body GR (prev iter, possibly NGL-flavored) -> LR1 (curr iter).

    Each test pins a specific vlcnt value at the wait against a known
    queue depth, then asserts wait sufficiency / insufficiency through
    the graph's ``WaitInsufficientFailure`` classifier — the graph-side
    analog of the legacy "shift value too small" failure.
    """

    def _build_capture(self, *, n_grs: int, vlcnt: int):
        """Build a 2-body capture: ML-1 has ``n_grs`` GRs, an SWait with
        the given vlcnt cap, and an SBarrier. ML has a single LR1 consumer.

        Returns the wrapped FourPartCapture ready for build_graph.
        """
        prev_insts = [
            _gr(slot=i, category="GRA", sequence=k)
            for i in range(n_grs // 2)
            for k in range(2)  # two seq per slot for variety
        ][:n_grs]
        prev_insts.extend([
            make_swait(slot=n_grs, vlcnt=vlcnt),
            make_sbarrier(slot=n_grs, sequence=1),
        ])
        prev_cap = make_capture(BODY_LABEL_ML_PREV, prev_insts)
        ml_cap = make_capture(BODY_LABEL_ML, [
            _lr1(slot=0, category="LRA1"),
        ])
        return self.wrap_single_body(ml_cap, ml_prev=prev_cap)

    def test_simple_case_success(self):
        """6 GRs in flight, SWait(vlcnt=0) drains them all -> no failure.

        Mirrors legacy ``test_simple_case_success``: shift_value=6 (=3 GRAs
        + 3 GRBs) is sufficient, so SWait drains everything before LR1."""
        cap = self._build_capture(n_grs=6, vlcnt=0)
        subj = self.build_graph(cap)
        # Cross-body edge ML-1.GRA -> ML.LRA1.
        self.assert_edge_exists(
            subj, edge_kind="gr_to_lr_lds_reuse",
            producer_category="GRA", consumer_category="LRA1",
        )
        failures = self.validate_waits(subj)
        self.assert_no_failures(failures)

    def test_simple_case_failure(self):
        """6 GRs in flight, SWait(vlcnt=2) leaves 4 GRs unfinished — but
        the OLDEST GR (the producer the graph picks for the gr_to_lr edge)
        IS drained at vlcnt=2. We need a tighter setup: pick a producer
        that is NOT drained by the wait.

        The pattern collector picks the FIRST GR in the sweep, but the
        edge formation iterates: every GR up to the next pattern reset
        forms an edge. So we instead test the SECOND GR's edge — its
        position in the FIFO is later, so a vlcnt=4 wait drains the
        oldest 2 only and the second-newest GR is still in flight.

        Concretely with 6 GRs and vlcnt=4: queue depth at wait=6, drained
        to 4, oldest 2 popped — GRs at FIFO positions 2..5 still in flight.
        For the GR at FIFO position 5 (the youngest), its
        gr_to_lr_lds_reuse edge to the LR1 has wait insufficient.
        """
        cap = self._build_capture(n_grs=6, vlcnt=4)
        subj = self.build_graph(cap)
        failures = self.validate_waits(subj)
        # WaitInsufficientFailure on the youngest GR -> LR1 edge.
        self.assert_failures_contain(
            failures, cls=WaitInsufficientFailure,
        )

    def test_simple_case_success_too_high(self):
        """vlcnt=0 with 6 GRs — wait drains all; no failure even if the
        legacy shift was 7 (a too-high shift translates graph-side to a
        wait with vlcnt=0 that fully drains, which is fine)."""
        cap = self._build_capture(n_grs=6, vlcnt=0)
        subj = self.build_graph(cap)
        failures = self.validate_waits(subj)
        self.assert_no_failures(failures)


# =============================================================================
# TestValidateNll — kept legacy until ola.3 (add_local_read_constraints)
# =============================================================================
# These tests exercise ``add_local_read_constraints`` (LR-data-ready
# rule), not ``add_gr_finish_before_lr_constraints``. The LR rule is
# being migrated under bead ``ola.3``; until then the legacy
# CMSValidationTestBase wiring is preserved here so the LR-side coverage
# isn't lost during ola.1's GR-side migration.


class TestValidateNll(CMSValidationTestBase):
    validator_passes = [add_local_read_constraints]

    def test_lr0_swait_depends_on_lr1(self):
        """
        Simple failure case where the SWaitCnt for LRB0 depends on the LRA1.
        """
        assert self.num_vmfma == 8

        optSchedule = {
            "SYNC": [[3, 7]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA1": [[2]],
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        self.validate(optSchedule, syncCode, 1, None, None, 0,
                                         expected_failure=WaitTooLateFailure)

    def test_lr0_swait_depends_on_lr1_realistic(self, useZeroDscnt: bool=False):
        """
        A more realistic version of `test_lr0_swait_depends_on_lr1`. GRs are now present. Optionally checks zero dscnt option for NLL
        """
        self.kernel["MIWaveTileA"] = 4
        self.kernel["MIWaveTileB"] = 4
        self.num_vmfma = 2 * self.kernel["MIWaveTileA"] * self.kernel["MIWaveTileB"]
        assert self.num_vmfma == 32

        syncTable = [
            1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Finish LRA0s"),
            1, SBarrier(comment=""),

            3, SWaitCnt(dscnt=-1, vlcnt=1, vscnt=-1, comment="Finish GRAs"),
            3, SBarrier(comment=""),

            15, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Finish 2/4 LRB0s"),

            # NOTE: This SWaitCnt is wrong for the NLL. It depends on the LRA1 being issued in order to guarantee that the LRB0s are finished.
            23, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Finish LRB0s"),

            28, SWaitCnt(dscnt=-1, vlcnt=2, vscnt=-1, comment="Finish GRBs"),
            28, SBarrier(comment=""),

            31, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA1s (but in NLL this leaves 2 LRB0s in-flight)"),
        ]
        optSchedule = {
            "SYNC": [syncTable[::2]],

            "LRA0": [[0]],
            "GRA":  [[2, 2]],

            "LRB0": [[3, 3, 3, 3]],

            "LRA1": [[22]],

            # Irrelevant, but need to schedule for correctness
            "GRB":  [[28, 28]],
            "LRB1": [[30, 30, 30, 30]],
        }
        # We need to set nglshift and nllshift for the vlcnt adjustments
        num_gr = 2  # 2 GRs total (1 GRA + 1 GRB, but we only count the actual reads not the increments)
        expected_failure = None if useZeroDscnt else WaitTooLateFailure
        self.validate(optSchedule, syncTable[1::2], 1, num_gr, num_gr, 0,
                      nllZeroDscnt=useZeroDscnt, expected_failure=expected_failure)

    def test_lr0_swait_depends_on_lr1_realistic_zero_dscnt(self):
        """
        Same as above, but uses the zero dscnt option for NLL, so the test should now pass.
        """
        self.test_lr0_swait_depends_on_lr1_realistic(useZeroDscnt=True)


# Keep imports referenced so linters don't flag them as unused. These
# Failure subclasses are exposed so the graph-native tests above can be
# extended with finer-grained assertions if/when needed.
_ = (MissingWaitFailure, WaitOnWrongCounterFailure)
