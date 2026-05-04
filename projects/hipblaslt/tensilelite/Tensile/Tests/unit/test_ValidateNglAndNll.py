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
"""TestValidateNgl + TestValidateNll: graph-native cross-body
GR -> LR1 / LR0 wait coverage (beads ola.1 + ola.3).

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

The legacy ``TestValidateNll`` exercised
``add_local_read_constraints``: SWaitCnts whose dscnt cap accidentally
includes a later LR1 leave earlier LR0s in flight when a consumer MFMA
fires. Bead ola.3 phase-2 deleted that rule; the equivalent coverage
graph-side is the LR0 -> MFMA RAW edge classified by
``validate_edge_wait_coverage`` (``MissingWaitFailure(dscnt)`` /
``WaitInsufficientFailure(dscnt)``).
"""

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    MissingWaitFailure,
    WaitInsufficientFailure,
    WaitOnWrongCounterFailure,
    WaitTooLateFailure,
)

from dataflow_fixtures import (
    make_capture, make_gr, make_lr, make_mfma, make_sbarrier, make_swait,
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
# TestValidateNll — graph-native LR-data-ready coverage (bead ola.3 phase-2)
# =============================================================================
# Legacy ``TestValidateNll`` exercised ``add_local_read_constraints``:
# the NLL-tail SWaitCnt's dscnt cap was set assuming an LRA1 had been
# issued, but in NLL no LRA1 actually exists (hence the "depends on
# LRA1" failure mode). With the cap miscounted, the LRB0 fed to the
# consumer MFMA never gets fully drained.
#
# Graph-native equivalent: model LRA0 + LRB0 (the producers the legacy
# rule complained about), then put the consumer MFMA in the ML body
# reading from the LRA0/LRB0 vgprs. Failure modes:
#
#   * No SWait between LR and consumer MFMA -> MissingWaitFailure(dscnt)
#   * SWait with insufficient dscnt cap (e.g. dscnt=1 leaves 1 LR in
#     flight) -> WaitInsufficientFailure(dscnt) on the older LR's edge


class TestValidateNll(GraphNativeValidationTest):
    """LR0 -> MFMA wait-coverage with mis-capped or absent SWaitCnts.

    Mirrors the legacy three failure shapes:

      1. Single LRA0 + single LRB0 sharing one consumer MFMA, no SWait
         in window -> MissingWait on the older LR's edge.
      2. Multi-LR shape: 4 LRB0s in flight, SWait(dscnt=1) drains 3 ->
         the oldest LRB0 -> MFMA edge is uncovered.
      3. Pass case: SWait(dscnt=0) covers everything.
    """

    def test_lr0_swait_depends_on_lr1(self):
        """Two LRs (LRA0 @ slot 0, LRB0 @ slot 0) feed one MFMA @ slot 4
        reading both vgprs. SWait(dscnt=1) at slot 3 leaves 1 LR in
        flight -> the OLDER LR's edge is uncovered.

        Mirrors legacy ``test_lr0_swait_depends_on_lr1`` where the
        NLL SWait's dscnt was set assuming LRA1 had been issued, but
        without LRA1 in the window, the cap was off-by-one and the
        consumer MFMA fired before the older LR was drained.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(dst_vgpr_start=8, dst_vgpr_count=2, lds_offset=64,
                    slot=0, category="LRA0"),
            make_lr(dst_vgpr_start=10, dst_vgpr_count=2, lds_offset=128,
                    slot=0, category="LRB0", sequence=1),
            make_swait(slot=3, dscnt=1),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=10,
                      slot=4, a_src_count=1, b_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        # The dscnt=1 cap leaves 1 LR pending. Either the older LR's
        # edge is reported as MissingWait (no qualifying drain) or
        # WaitInsufficient (drain present but cap leaves it in flight).
        # Both are equivalent coverage signals here.
        dscnt_failures = [
            f for f in failures
            if isinstance(f, (MissingWaitFailure, WaitInsufficientFailure))
        ]
        assert dscnt_failures, (
            f"Expected dscnt-related failure on the older LR's edge. "
            f"Got: {[type(f).__name__ for f in failures]}"
        )

    def test_lr0_swait_depends_on_lr1_realistic(self):
        """4 LRB0s in flight, SWait(dscnt=1) leaves 1 LR pending. The
        consumer MFMA reads from the YOUNGEST (sequence=3) LR's vgpr —
        which is the one dscnt=1 leaves un-drained -> WaitInsufficient
        on the youngest LR's edge.

        Mirrors the legacy ``test_lr0_swait_depends_on_lr1_realistic``
        shape: the NLL's SWait(dscnt=1) was meant to cover LRB0s but
        accidentally counted LRA1 as already-issued; without the LRA1
        in the queue the cap is wrong and the MFMA's data is unreliable.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(dst_vgpr_start=8, dst_vgpr_count=2, lds_offset=64,
                    slot=0, category="LRB0", sequence=0),
            make_lr(dst_vgpr_start=10, dst_vgpr_count=2, lds_offset=128,
                    slot=0, category="LRB0", sequence=1),
            make_lr(dst_vgpr_start=12, dst_vgpr_count=2, lds_offset=192,
                    slot=0, category="LRB0", sequence=2),
            make_lr(dst_vgpr_start=14, dst_vgpr_count=2, lds_offset=256,
                    slot=0, category="LRB0", sequence=3),
            # dscnt=1 leaves 1 LR pending — the YOUNGEST (sequence=3)
            # in dscnt's LIFO interpretation. MFMA reads v14 (that LR's
            # dst), so the LR -> MFMA edge fails coverage.
            make_swait(slot=3, dscnt=1),
            make_mfma(c_dst_start=0, a_src_start=14, b_src_start=32,
                      slot=4, a_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        dscnt_failures = [
            f for f in failures
            if isinstance(f, (MissingWaitFailure, WaitInsufficientFailure))
        ]
        assert dscnt_failures, (
            f"Expected dscnt-related failure. Got: "
            f"{[type(f).__name__ for f in failures]}"
        )

    def test_lr0_swait_depends_on_lr1_realistic_zero_dscnt(self):
        """Pass case: same shape but SWait(dscnt=0) drains everything
        before the consumer MFMA fires. No failure.

        Mirrors the legacy ``test_lr0_swait_depends_on_lr1_realistic_zero_dscnt``
        which used the ``nllZeroDscnt`` option to force the NLL SWait
        to dscnt=0 and verified the validator passed.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(dst_vgpr_start=8, dst_vgpr_count=2, lds_offset=64,
                    slot=0, category="LRB0", sequence=0),
            make_lr(dst_vgpr_start=10, dst_vgpr_count=2, lds_offset=128,
                    slot=0, category="LRB0", sequence=1),
            make_lr(dst_vgpr_start=12, dst_vgpr_count=2, lds_offset=192,
                    slot=0, category="LRB0", sequence=2),
            make_lr(dst_vgpr_start=14, dst_vgpr_count=2, lds_offset=256,
                    slot=0, category="LRB0", sequence=3),
            make_swait(slot=3, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=14, b_src_start=32,
                      slot=4, a_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        self.assert_no_failures(failures)


# Keep imports referenced so linters don't flag them as unused. These
# Failure subclasses are exposed so the graph-native tests above can be
# extended with finer-grained assertions if/when needed.
_ = (WaitOnWrongCounterFailure, WaitTooLateFailure)
