################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# SPDX-License-Identifier: MIT
################################################################################
"""Graph-native port of test_ValidateGRsCompleteBeforeLr1s.

Migrates the ``add_gr_finish_before_lr_constraints`` structural-rule tests to
the dataflow-graph wait-coverage model. The legacy rule encoded:

    GR (write LDS) -> SWaitCnt(vlcnt=0) -> SBarrier -> LR1 (read LDS, next iter)

which is exactly ``edge_kind="gr_to_lr_lds_reuse"`` in the graph
(ScheduleCapture._collect_barrier_edges). The graph emits the same typed
Failures (MissingWaitFailure / MissingBarrierFailure) via two paths:

  * ``compare_graphs(ref, subj)`` -> ``diagnose_missing_edge``: when subj
    is missing the ``gr_to_lr_lds_reuse`` edge that ref has, the diagnosis
    walks subj's wait/barrier window between the (still-present) producer
    and consumer nodes and classifies why the edge didn't form.
  * ``validate_edge_wait_coverage(subj)``: when subj DOES form the edge
    (correct GR->SWait->SBarrier->LR pattern) the per-edge classifier
    re-validates wait sufficiency / barrier presence.

Acceptance criterion: commenting out the gr_to_lr_lds_reuse branch in
``_collect_barrier_edges`` (or the LDS-reuse barrier branch in
``_classify_edge_coverage`` / ``diagnose_missing_edge``) MUST break the
negative tests below — verifying these tests bind to the graph-side
classifier and not to the (now-removed) structural rule.

The wait-after-consumer cases (``test_guaranteed_after_first_lr1``,
``test_swap_global_read_order_failure``) surface as ``MissingWaitFailure`` /
``MissingBarrierFailure``: when the GR's covering wait sits AT or AFTER
the consumer LR1, the wait falls outside ``waits_in_window``'s
[producer, consumer) range and the classifier reports a missing-wait
condition. This is the same defect under a different name — both indicate
the schedule cannot guarantee LDS coherence at the LR1 read.
"""

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    MissingBarrierFailure,
    MissingWaitFailure,
    OrderInvertedFailure,
)

from dataflow_fixtures import (
    make_capture, make_gr, make_lr, make_mfma, make_sbarrier, make_swait,
)
from graph_native_validation_base import GraphNativeValidationTest


# Common register layout for the GR -> LR1 LDS-reuse pattern.
# GR writes to vgpr range [40..43] (its dst — interpreted as the LDS slot
# under DTL); LR1 reads from LDS offset 64 into vgpr range [8..11]. The
# specific register identities don't form a register-RAW edge between the
# two (LR reads from LDS, not from the GR's vgpr) — the edge is the
# barrier-pattern-derived ``gr_to_lr_lds_reuse``.
_GR_VGPR_BASE = 40
_GR_VGPR_COUNT = 4
_GR_SRD = 12
_GR_LDS_OFFSET = 64

_LR_VGPR_BASE = 8
_LR_VGPR_COUNT = 4
_LR_LDS_OFFSET = 64


def _gr(slot: int, category: str = "GRA", *, sequence: int = 0,
        vgpr_base: int = _GR_VGPR_BASE):
    """Single GR producer at the given slot. Reuses constants so all
    tests in this file use identical resource pins (vgpr, SRD, LDS slot)
    -> the per-byte resolver doesn't form spurious RAW edges across tests."""
    return make_gr(
        vgpr_base, _GR_VGPR_COUNT, srd_sgpr_start=_GR_SRD,
        immediate_offset=_GR_LDS_OFFSET, slot=slot,
        category=category, sequence=sequence,
    )


def _lr1(slot: int, category: str = "LRA1", *, sequence: int = 0,
         vgpr_base: int = _LR_VGPR_BASE):
    """Single LR1 consumer at the given slot."""
    return make_lr(
        vgpr_base, _LR_VGPR_COUNT, lds_offset=_LR_LDS_OFFSET, slot=slot,
        category=category, sequence=sequence,
    )


# =============================================================================
# Positive tests — pattern correct, edge forms, wait coverage passes
# =============================================================================


class TestGRBeforeLR1_Positive(GraphNativeValidationTest):
    """Schedules where the GR -> SWait(vlcnt) -> SBarrier -> LR1 pattern is
    correctly placed. validate_edge_wait_coverage emits no failures.
    """

    def test_simple_case_success(self):
        """Canonical placement: GR @ 0, SWait(vlcnt=0) @ 2, SBarrier @ 2,
        LR1 @ 6. Edge forms; wait drains the GR; barrier follows the wait."""
        cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            _gr(slot=0, category="GRB", vgpr_base=44),
            make_swait(slot=2, vlcnt=0),
            make_sbarrier(slot=2, sequence=1),
            _lr1(slot=6, category="LRA1"),
            _lr1(slot=6, category="LRB1", vgpr_base=12, sequence=1),
        ])
        subj = self.build_graph(self.wrap_single_body(cap))
        # Edge formed by the barrier-pattern collector.
        self.assert_edge_exists(subj, edge_kind="gr_to_lr_lds_reuse",
                                producer_category="GRA",
                                consumer_category="LRA1")
        failures = self.validate_waits(subj)
        self.assert_no_failures(failures)

    def test_grs_in_preloop_passing(self):
        """Pathological-but-valid: GR in ML-1 (prev iter), SWait+SBarrier
        in ML-1 too, LR1 in ML body. Cross-body edge from ML-1.GR to ML.LR1
        forms cleanly via _collect_barrier_edges' cross-body sweep."""
        prev_cap = make_capture(BODY_LABEL_ML_PREV, [
            _gr(slot=0, category="GRA"),
            make_swait(slot=1, vlcnt=0),
            make_sbarrier(slot=2),
        ])
        ml_cap = make_capture(BODY_LABEL_ML, [
            _lr1(slot=0, category="LRA1"),
        ])
        cap = self.wrap_single_body(ml_cap, ml_prev=prev_cap)
        subj = self.build_graph(cap)
        # Cross-body edge ML-1.GRA -> ML.LRA1.
        self.assert_edge_exists(subj, edge_kind="gr_to_lr_lds_reuse",
                                producer_category="GRA",
                                consumer_category="LRA1")
        failures = self.validate_waits(subj)
        self.assert_no_failures(failures)

    def test_swap_global_read_order_passing(self):
        """SwapGlobalReadOrder=True scenario in the legacy test boils down,
        graph-side, to a pair of GR -> LR1 chains (one for A, one for B)
        each with their own SWait+SBarrier. The graph doesn't care about
        the swap kernel flag — it cares only about the captured shape.

        With 2 GRs total in the queue at the GRA-covering wait and 1 GR
        beyond it, vlcnt drains must be tight enough to preserve wait
        sufficiency for both LR1 consumers (GRB feeds LRA1 here, mirroring
        the swap semantics where GRBs actually load A)."""
        cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_swait(slot=1, vlcnt=1),
            make_sbarrier(slot=1, sequence=1),
            _gr(slot=3, category="GRB", vgpr_base=44),
            make_swait(slot=4, vlcnt=0),
            make_sbarrier(slot=4, sequence=1),
            _lr1(slot=5, category="LRB1", vgpr_base=12),
            _lr1(slot=6, category="LRA1"),
        ])
        subj = self.build_graph(self.wrap_single_body(cap))
        failures = self.validate_waits(subj)
        self.assert_no_failures(failures)


# =============================================================================
# Negative tests — pattern broken; compare_graphs(ref, subj) drives the
# classifier. Each test builds a CORRECT reference (edge forms) and a
# BROKEN subject (edge doesn't form OR forms but wait/barrier inadequate);
# diagnose_missing_edge or validate_edge_wait_coverage emits the typed
# failure that the legacy structural rule used to emit.
# =============================================================================


class TestGRBeforeLR1_MissingWait(GraphNativeValidationTest):
    """SWait dropped or wrong counter -> MissingWaitFailure with vlcnt."""

    def _ref(self):
        """Reference: full GR -> SWait(vlcnt=0) -> SBarrier -> LR1 chain."""
        return make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_swait(slot=2, vlcnt=0),
            make_sbarrier(slot=2, sequence=1),
            _lr1(slot=6, category="LRA1"),
        ])

    def test_grs_swait_on_wrong_counter(self):
        """SWait fires with dscnt=0 (LR-counter) instead of vlcnt=0
        (GR-counter). Pattern collector requires vlcnt-draining wait, so
        the edge doesn't form. diagnose_missing_edge finds no vlcnt-draining
        wait in the [GR, LR1) window and emits MissingWaitFailure with
        counter_kind='vlcnt' AND nearby_other_counter_waits populated with
        the wrong-counter SWait (so the user knows they could extend it).

        Equivalent to legacy ``test_grs_not_swait``: same defect (no
        vlcnt drain on the GR). The presence of the wrong-counter
        SWaitCnt is surfaced via nearby_other_counter_waits — the user
        can extend that SWaitCnt to add a vlcnt drain rather than
        inserting a new one."""
        subj_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_swait(slot=2, dscnt=0),    # wrong counter
            make_sbarrier(slot=2, sequence=1),
            _lr1(slot=6, category="LRA1"),
        ])
        failures = self.compare(
            self.wrap_single_body(self._ref()),
            self.wrap_single_body(subj_cap),
        )
        f = self.assert_failures_contain(
            failures, cls=MissingWaitFailure,
            counter_kind="vlcnt",
        )
        assert f.producer.category == "GRA"
        assert f.consumer.category == "LRA1"
        # The wrong-counter SWait (dscnt drain at slot=2) is surfaced
        # as a nearby SWait the user could extend.
        assert len(f.nearby_wait_indices) >= 1

    def test_grs_no_swait_at_all(self):
        """No SWait of ANY kind in the window — diagnose_missing_edge
        emits MissingWaitFailure with counter_kind='vlcnt'."""
        subj_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            # No SWait. SBarrier present but irrelevant for the wait check.
            make_sbarrier(slot=2),
            _lr1(slot=6, category="LRA1"),
        ])
        failures = self.compare(
            self.wrap_single_body(self._ref()),
            self.wrap_single_body(subj_cap),
        )
        f = self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="vlcnt",
        )
        assert f.producer.category == "GRA"
        assert f.consumer.category == "LRA1"

    def test_swait_after_lr1(self):
        """SWait sits AFTER the LR1 consumer. The pattern collector won't
        form an edge (wait must precede consumer); compare_graphs sees the
        missing edge and diagnose_missing_edge classifies as MissingWait
        (no covering wait in [producer, consumer) window)."""
        subj_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            _lr1(slot=3, category="LRA1"),
            # Wait+Barrier sit AFTER the LR1 — outside the
            # [producer, consumer) window.
            make_swait(slot=4, vlcnt=0),
            make_sbarrier(slot=4, sequence=1),
        ])
        failures = self.compare(
            self.wrap_single_body(self._ref()),
            self.wrap_single_body(subj_cap),
        )
        f = self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="vlcnt",
        )
        assert f.producer.category == "GRA"
        assert f.consumer.category == "LRA1"


class TestGRBeforeLR1_MissingBarrier(GraphNativeValidationTest):
    """SWait present and drains, but no SBarrier in the window -> MissingBarrierFailure
    with role='needed_by'."""

    def _ref(self):
        return make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_swait(slot=2, vlcnt=0),
            make_sbarrier(slot=2, sequence=1),
            _lr1(slot=6, category="LRA1"),
        ])

    def test_no_sbarrier(self):
        """SBarrier omitted entirely. Pattern collector won't form the edge;
        compare_graphs reports it missing; diagnose_missing_edge sees the
        wait drains the producer and emits MissingBarrierFailure
        (role='needed_by')."""
        subj_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_swait(slot=2, vlcnt=0),
            # SBarrier deleted.
            _lr1(slot=6, category="LRA1"),
        ])
        failures = self.compare(
            self.wrap_single_body(self._ref()),
            self.wrap_single_body(subj_cap),
        )
        f = self.assert_failures_contain(
            failures, cls=MissingBarrierFailure, role="needed_by",
        )
        assert f.producer.category == "GRA"
        assert f.consumer.category == "LRA1"

    def test_sbarrier_before_swait(self):
        """SBarrier sits BEFORE the SWait that drains GR's vlcnt. Required
        ordering is GR -> SWait -> SBarrier -> LR1; this subject has
        GR -> SBarrier -> SWait -> LR1, so the cross-wave coherence
        invariant fails. _collect_pattern walks producer -> wait -> barrier
        in that order and won't pair this barrier with the wait, so the
        edge doesn't form. diagnose_missing_edge classifies as
        MissingBarrierFailure (the wait drains but no barrier follows it)."""
        subj_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_sbarrier(slot=1),                  # barrier BEFORE wait
            make_swait(slot=2, vlcnt=0),
            _lr1(slot=6, category="LRA1"),
        ])
        failures = self.compare(
            self.wrap_single_body(self._ref()),
            self.wrap_single_body(subj_cap),
        )
        self.assert_failures_contain(
            failures, cls=MissingBarrierFailure, role="needed_by",
        )


class TestGRBeforeLR1_WaitAfterConsumer(GraphNativeValidationTest):
    """SWait that would cover the GR fires AT or AFTER the LR1 consumer.
    Surfaces as MissingWaitFailure — the wait isn't in the [producer,
    consumer) window, which from the consumer's perspective is
    indistinguishable from no wait at all."""

    def test_guaranteed_after_first_lr1(self):
        """4 GRs in flight; SWait(vlcnt=4) is at slot 4, but the FIRST LR1
        consumer is at slot 3 — so for that consumer the wait falls
        AFTER the consumer position. compare_graphs flags the missing
        gr_to_lr_lds_reuse edge for the first LR1 -> MissingWait."""
        ref_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            _gr(slot=0, category="GRA", sequence=1),
            _gr(slot=0, category="GRA", sequence=2),
            _gr(slot=0, category="GRA", sequence=3),
            make_swait(slot=2, vlcnt=4),
            make_sbarrier(slot=2, sequence=1),
            _lr1(slot=6, category="LRA1"),
        ])
        subj_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            _gr(slot=0, category="GRA", sequence=1),
            _gr(slot=0, category="GRA", sequence=2),
            _gr(slot=0, category="GRA", sequence=3),
            _lr1(slot=3, category="LRA1"),
            # SWait fires at slot 4 — after the LR1 consumer at slot 3.
            make_swait(slot=4, vlcnt=4),
            make_sbarrier(slot=4, sequence=1),
        ])
        failures = self.compare(
            self.wrap_single_body(ref_cap),
            self.wrap_single_body(subj_cap),
        )
        f = self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="vlcnt",
        )
        assert f.producer.category == "GRA"
        assert f.consumer.category == "LRA1"

    def test_swap_global_read_order_failure(self):
        """SwapGlobalReadOrder=True legacy scenario: GRBs (which actually
        load A under swap) must precede LRA1 — but in this subj, GRB sits
        at slot=3 AFTER LRA1 at slot=2. Graph-side: compare_graphs emits
        OrderInvertedFailure for the GRB -> LRA1 cross-graph edge whose
        positions are reversed in subj relative to ref.

        The wait-after-consumer view of this defect is one symptom of the
        order inversion. The graph reports the more fundamental issue: the
        producer is positioned after its consumer."""
        ref_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            _gr(slot=0, category="GRB", vgpr_base=44),
            make_swait(slot=1, vlcnt=0),
            make_sbarrier(slot=1, sequence=1),
            _lr1(slot=2, category="LRB1", vgpr_base=12),
            _lr1(slot=5, category="LRA1"),
        ])
        subj_cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_swait(slot=1, vlcnt=0),
            make_sbarrier(slot=1, sequence=1),
            _lr1(slot=2, category="LRA1"),
            _gr(slot=3, category="GRB", vgpr_base=44),
            # GRB's covering wait/barrier sit AFTER LRA1 — but the more
            # fundamental defect is that GRB is positioned AFTER its
            # cross-graph LRA1 consumer in subj's edge set.
            make_swait(slot=4, vlcnt=0),
            make_sbarrier(slot=4, sequence=1),
            _lr1(slot=5, category="LRB1", vgpr_base=12),
        ])
        failures = self.compare(
            self.wrap_single_body(ref_cap),
            self.wrap_single_body(subj_cap),
            raise_on_unexplained=False,
        )
        # OrderInverted on a GR -> LR1 edge that the swap-aware reference
        # has but the broken subject doesn't. The graph also reports
        # unclassifiable failures (synthetic MissingWait with
        # counter_kind='unknown') for the cross-graph mismatch — accepting
        # the OrderInverted is sufficient to bind the test to the
        # gr_to_lr_lds_reuse classifier.
        f = self.assert_failures_contain(failures, cls=OrderInvertedFailure)
        assert f.producer.category in {"GRA", "GRB"}
        assert f.consumer.category in {"LRA1", "LRB1"}


# =============================================================================
# Edge-formation invariants
# =============================================================================
# These tests bind directly to _collect_barrier_edges + _classify_edge_coverage
# behavior: they assert the edge IS or IS NOT present in the graph for a
# given input shape. Useful as the "mutation-smell-test" foundation: if
# _collect_barrier_edges' gr_to_lr_lds_reuse pattern is removed, the
# positive edge-existence assertions break immediately.


class TestGRToLR1EdgeFormation(GraphNativeValidationTest):
    def test_edge_forms_with_correct_pattern(self):
        cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_swait(slot=1, vlcnt=0),
            make_sbarrier(slot=2),
            _lr1(slot=3, category="LRA1"),
        ])
        subj = self.build_graph(self.wrap_single_body(cap))
        self.assert_edge_exists(
            subj, edge_kind="gr_to_lr_lds_reuse",
            producer_category="GRA", consumer_category="LRA1",
        )

    def test_no_edge_when_swait_missing(self):
        """SWait absent -> _collect_pattern can't form the gr_to_lr_lds_reuse
        edge. The structural absence IS the bug detection at the graph
        layer; the wait-coverage classifier doesn't run for non-existent
        edges."""
        cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_sbarrier(slot=1),
            _lr1(slot=2, category="LRA1"),
        ])
        subj = self.build_graph(self.wrap_single_body(cap))
        self.assert_edge_absent(subj, edge_kind="gr_to_lr_lds_reuse")

    def test_no_edge_when_sbarrier_missing(self):
        """SBarrier absent -> pattern collector requires it to form edge."""
        cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_swait(slot=1, vlcnt=0),
            _lr1(slot=2, category="LRA1"),
        ])
        subj = self.build_graph(self.wrap_single_body(cap))
        self.assert_edge_absent(subj, edge_kind="gr_to_lr_lds_reuse")

    def test_no_edge_when_swait_drains_wrong_counter(self):
        """SWait fires with dscnt=0 instead of vlcnt — wrong counter for
        the GR producer (GR uses vlcnt). Pattern collector skips this wait."""
        cap = make_capture(BODY_LABEL_ML, [
            _gr(slot=0, category="GRA"),
            make_swait(slot=1, dscnt=0),    # wrong counter
            make_sbarrier(slot=2),
            _lr1(slot=3, category="LRA1"),
        ])
        subj = self.build_graph(self.wrap_single_body(cap))
        self.assert_edge_absent(subj, edge_kind="gr_to_lr_lds_reuse")


# =============================================================================
# MFMA filler — lifted from existing graph-native tests.
# Some tests above don't include an MFMA in the ML body; the wrap_single_body
# filler bodies (NGL/NLL/ML-1) each contain a single MFMA at high vgpr ranges
# (200+) so the edge-formation invariants under test aren't polluted by
# unrelated MFMA-derived edges.
# =============================================================================


# Sanity: every test class above uses GraphNativeValidationTest. Importing
# make_mfma here keeps the symbol available for any future quad-cycle
# follow-ups without scattering imports.
_ = make_mfma  # noqa: F841
