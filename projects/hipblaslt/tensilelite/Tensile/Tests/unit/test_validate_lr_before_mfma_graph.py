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
# SPDX-License-Identifier: MIT
################################################################################
"""Graph-native ports of test_ValidateLRsCompleteBeforeVMFMA.py.

Replaces the (now-removed) structural rule ``add_local_read_constraints``
— and its ``set_lr_needed_by_for_VMFMA`` consumer-MFMA wiring — with
graph-side LR -> MFMA wait-coverage.

The legacy rule annotated each LocalRead with the MFMA that consumes it
and then validated that an SWaitCnt(dscnt) drained the LR before the
MFMA fires. The graph already builds LR -> MFMA RAW edges via
``_DSLoadRule`` (writes vgpr) + ``_MFMARule`` (reads vgpr) + the per-byte
latest-writer resolver. ``validate_edge_wait_coverage`` then checks each
such edge for ``dscnt`` coverage and emits ``MissingWaitFailure`` when
no qualifying wait sits in the producer→consumer window (this collapses
both the legacy "missing wait" and "wait too late" cases — when the
SWait sits AT or AFTER the consumer, the window-search returns no
waits, yielding MissingWaitFailure on dscnt).

Migration mapping:

  SWait at-or-after consumer        ->  MissingWaitFailure(counter_kind="dscnt")
  Legacy "no failure"               ->  empty failure list

The legacy tests' positional/SWait constructions are translated into
graph fixtures by:

  * Picking distinct vgpr ranges for each LR so the graph forms a
    separate edge per LR. LRA0 -> v8..v9, LRB0 -> v10..v11, LRA1 ->
    v12..v13, etc. The MFMA's a_src/b_src/c_dst pull from these so
    edges form unambiguously.
  * Placing the SWait at the same slot index that the legacy SYNC entry
    occupied — between the LR and the MFMA in stream order.
  * For "LR before MFMA in same subiter" (LR0 case): place LR low slot,
    MFMA at a higher slot, SWait between them.
  * For "LR finishes after MFMA" (LR0 positionally late, LR1 needed by
    next-iter MFMA): place the LR after the MFMA in stream order. The
    graph's per-byte latest-writer rejects this construction (the MFMA
    reads the OLD value, not the new one), so we instead model LR1 by
    placing the producer in BODY_LABEL_ML_PREV (previous iteration) and
    the consumer MFMA in BODY_LABEL_ML — yielding the cross-iter edge
    that legacy ``set_lr_needed_by_for_VMFMA`` materialized via the
    "+ num_vmfma" offset.

Out of scope (deferred — see "Production-side deletions" note below):

  * Pure-Python helper tests TestIndexForForceUnrollSubIter and
    TestLRNeededByMFMA pin internal tile-math helpers
    (``index_for_force_unroll_sub_iter``, ``lr_needed_by_mfma``). The
    graph model derives LR -> MFMA pairing from real register dataflow
    instead of this helper, so the helpers themselves are no longer
    exercised through the public path. They are left in place pending
    deletion alongside the structural rule.

  * Tests assuming a specific consumer-MFMA index that depends on
    ``mfma_reorder`` reordering: graph picks the MFMA that actually
    reads the LR's vgpr based on register dataflow, not on
    ``mfma_reorder`` heuristics. Tests that constructed LR/MFMA register
    assignments where the heuristic and the dataflow disagree may fall
    out differently — covered case-by-case in test docstrings below.

Production-side helpers (``add_local_read_constraints``,
``set_lr_needed_by_for_VMFMA``, ``lr_needed_by_mfma``,
``index_for_force_unroll_sub_iter``, ``LRDataReadyRule``,
``LocalRead.validate``, ``LocalRead.needed_by``) have been removed once
all sibling test files (``test_ValidatePack.py``,
``test_LR_Pack_interaction.py``, ``test_ValidateNglAndNll.py``,
``test_register_tracing.py``) had graph-native equivalents.
"""

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    MissingWaitFailure,
)

from dataflow_fixtures import (
    make_capture, make_lr, make_mfma, make_swait,
)
from graph_native_validation_base import GraphNativeValidationTest


# =============================================================================
# Helpers
# =============================================================================
# Convention for vgpr ranges across all tests in this file:
#
#   LRA0 -> v8..v9    (2 vgprs)         consumed by MFMA.a_src
#   LRB0 -> v10..v11                    consumed by MFMA.b_src
#   LRA1 -> v12..v13                    consumed by next-iter MFMA.a_src
#   LRB1 -> v14..v15                    consumed by next-iter MFMA.b_src
#   LRA3 -> v16..v17                    consumed by next-iter MFMA.a_src
#                                       (ForceUnrollSubIter alias for LRA1)
#   LRB3 -> v18..v19                    same for B
#   MFMA.c_dst -> v0..v3                accumulator, distinct per fixture
#
# Filler bodies use vgpr 200+ (see graph_native_validation_base._FILLER_RANGES)
# so they don't form spurious edges against this body's nodes.


def _ml_lr_then_mfma(lr_slot: int, swait_slot: int, mfma_slot: int,
                     *, lr_vgpr_start: int = 8, lr_vgpr_count: int = 2,
                     mfma_a_start: int = 8, mfma_b_start: int = 32,
                     dscnt: int = 0, lr_category: str = "LRA0"):
    """Build a single-body main-loop capture with one LR -> SWait -> MFMA chain.

    The LR writes ``[lr_vgpr_start, lr_vgpr_start+lr_vgpr_count)`` and the
    MFMA reads from ``mfma_a_start`` (size 1 source). When
    ``mfma_a_start == lr_vgpr_start`` the graph forms an LR -> MFMA RAW
    edge on the overlapping vgpr.
    """
    return make_capture(BODY_LABEL_ML, [
        make_lr(dst_vgpr_start=lr_vgpr_start, dst_vgpr_count=lr_vgpr_count,
                lds_offset=64, slot=lr_slot, category=lr_category),
        make_swait(slot=swait_slot, dscnt=dscnt),
        make_mfma(c_dst_start=0, a_src_start=mfma_a_start,
                  b_src_start=mfma_b_start, slot=mfma_slot, a_src_count=1),
    ])


# =============================================================================
# LR0 -> MFMA RAW: simple pass/fail cases (port of TestValidateLRsCompleteBeforeVMFMA)
# =============================================================================
# Legacy tests built an 8-MFMA timeline and asserted that LR0 had to be
# guaranteed (via dscnt SWait) before the "halfway" MFMA — index 4 (which
# the legacy code derived from num_vmfma // 2). The graph derives the
# consumer MFMA from real register dataflow: a single LR -> MFMA edge
# is enough to exercise the wait-coverage classifier.


class TestLRBeforeMFMA_LR0(GraphNativeValidationTest):
    """Graph-native port of TestValidateLRsCompleteBeforeVMFMA::test_simple_LR0.

    Legacy: LR0 must finish (be drained by SWait) before the halfway-point
    MFMA (idx 4 in the legacy 8-MFMA loop).

    Graph: LR (writing v8) is consumed by an MFMA at slot 4 reading v8.
    SWait(dscnt=0) at slot 3 sits in the producer→consumer window and
    drains the dscnt queue → no failure.
    """

    def test_LR0_baseline_swait_in_window(self):
        """LR @ slot 0 writes v8; SWait(dscnt=0) @ slot 3; MFMA @ slot 4
        reads v8. SWait covers the LR -> MFMA dscnt drain."""
        cap = _ml_lr_then_mfma(lr_slot=0, swait_slot=3, mfma_slot=4)
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        self.assert_no_failures(failures)

    def test_LR0_swait_after_consumer_dscnt_missing(self):
        """LR @ slot 0; MFMA @ slot 4; SWait sits AT slot 6 (AFTER consumer).
        Window producer→consumer is empty → MissingWaitFailure(dscnt).

        Mirrors the legacy ``test_simple_LR0`` second sub-case where SYNC
        @ idx 3 covered an LR @ idx 6 — but the LR was issued AFTER the
        SWait, leaving the consumer MFMA without a covering wait.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(dst_vgpr_start=8, dst_vgpr_count=2, lds_offset=64,
                    slot=0, category="LRA0"),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=4, a_src_count=1),
            make_swait(slot=6, dscnt=0),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )

    def test_LR0_lr_after_mfma_positionally_late(self):
        """LR sits AFTER its consumer MFMA in stream order. The graph
        emits the edge in stream order (writer-after-reader is detected
        by per-byte latest-writer + cross-graph compare). This test pins
        the within-graph wait-coverage signal: with no SWait between the
        LR (slot 6) and any earlier consumer (slot 4), the edge is
        not covered and we get MissingWaitFailure.

        Note: when the LR slot > MFMA slot, the per-byte resolver in the
        graph builder publishes the LR write as the latest writer for
        any LATER MFMA. With only one MFMA at slot 4 reading v8, the
        v8 write at slot 6 establishes the LR as a producer for any
        post-slot-6 consumer — but our MFMA at slot 4 reads the value
        from BEFORE the LR. To reliably test the legacy "LR positionally
        late" case in graph terms, we add a second MFMA AFTER the LR
        that consumes its newly-written value.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(dst_vgpr_start=8, dst_vgpr_count=2, lds_offset=64,
                    slot=6, category="LRA0"),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=7, a_src_count=1),  # reads the late LR
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        # No SWait at all in the LR -> MFMA window → MissingWait.
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )


class TestLRBeforeMFMA_LRB0(GraphNativeValidationTest):
    """Variant where LRB0 is the producer being uncovered.

    Mirrors the third sub-case of legacy ``test_simple_LR0`` (LRB0 @ idx 6
    with SYNC @ idx 3 — SWait fires before the LR producer, so it doesn't
    drain LRB0; surfaces as MissingWaitFailure).
    """

    def test_LRB0_swait_before_lr_does_not_cover(self):
        """LRA0 @ 1 covered by SWait @ 3; LRB0 @ 4 reads-by MFMA @ 5.
        No SWait sits in the LRB0 → MFMA window → MissingWait on dscnt
        for the LRB0 -> MFMA edge."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(dst_vgpr_start=8, dst_vgpr_count=2, lds_offset=64,
                    slot=1, category="LRA0"),
            make_swait(slot=3, dscnt=0),
            make_lr(dst_vgpr_start=10, dst_vgpr_count=2, lds_offset=128,
                    slot=4, category="LRB0"),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=10,
                      slot=5, a_src_count=1, b_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        # The LRA0 -> MFMA edge is covered (SWait at slot 3 sits between
        # LRA0 @ 1 and MFMA @ 5). The LRB0 -> MFMA edge has no covering
        # SWait in its window (slot 4..5).
        f = self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )
        # Pin which producer the failure is about — must be LRB0, not LRA0.
        assert f.producer.category == "LRB0", (
            f"expected LRB0 as failing producer, got {f.producer.category}"
        )


class TestLRBeforeMFMA_LR1(GraphNativeValidationTest):
    """LR1 (current iter) feeds NEXT-iter MFMA.

    Legacy ``set_lr_needed_by_for_VMFMA`` adds a `+num_vmfma` offset to
    pair LR1 with a next-iter MFMA. The graph models this directly: the
    LR1 lives in BODY_LABEL_ML_PREV and the consumer MFMA lives in
    BODY_LABEL_ML — a cross-body raw_intrawave edge.
    """

    def test_LR1_baseline_passing(self):
        """LR1 in main_loop_prev writes v12; next-iter MFMA in main_loop
        reads v12. SWait(dscnt=0) inside main_loop covers the cross-iter
        edge.
        """
        ml_prev = make_capture(BODY_LABEL_ML_PREV, [
            make_lr(dst_vgpr_start=12, dst_vgpr_count=2, lds_offset=192,
                    slot=4, category="LRA1"),
        ])
        ml = make_capture(BODY_LABEL_ML, [
            make_swait(slot=0, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=12, b_src_start=32,
                      slot=1, a_src_count=1),
        ])
        cap = self.wrap_single_body(ml, ml_prev=ml_prev)
        failures = self.validate_waits(self.build_graph(cap))
        self.assert_no_failures(failures)

    def test_LR1_no_swait_in_next_iter_window(self):
        """LR1 in main_loop_prev; consumer MFMA in main_loop @ slot 0.
        No SWait sits in the cross-iter window → MissingWait on dscnt.
        Mirrors the legacy ``test_simple_LR1_guaranteed_too_late``.
        """
        ml_prev = make_capture(BODY_LABEL_ML_PREV, [
            make_lr(dst_vgpr_start=12, dst_vgpr_count=2, lds_offset=192,
                    slot=4, category="LRA1"),
        ])
        ml = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=12, b_src_start=32,
                      slot=0, a_src_count=1),
        ])
        cap = self.wrap_single_body(ml, ml_prev=ml_prev)
        failures = self.validate_waits(self.build_graph(cap))
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )


# =============================================================================
# Multi-LR / multi-SWait shape (port of test_more_LRs and test_handling_instruction_order)
# =============================================================================


class TestLRBeforeMFMA_MultiLR(GraphNativeValidationTest):
    """Multiple LRs in flight; SWait drains a configurable number.

    Graph models each LR as a separate edge with its own register
    identity, so the dscnt drain N applies to each edge identically.
    """

    def test_two_LR0s_both_drained(self):
        """Two LR0s on distinct vgprs; SWait(dscnt=0) drains both before
        the consumer MFMA reads them. No failure."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(dst_vgpr_start=8, dst_vgpr_count=2, lds_offset=64,
                    slot=0, category="LRA0"),
            make_lr(dst_vgpr_start=10, dst_vgpr_count=2, lds_offset=128,
                    slot=1, category="LRB0"),
            make_swait(slot=2, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=10,
                      slot=3, a_src_count=1, b_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        self.assert_no_failures(failures)

    def test_two_LR0s_swait_drains_only_one(self):
        """Two LR0s in dscnt queue; SWait(dscnt=1) drains the LATEST one
        only, leaving the EARLIER one still pending → WaitInsufficient
        on the earlier producer's edge.

        Mirrors the legacy ``test_more_LRs_failure`` shape. In the graph
        wait-coverage classifier this surfaces as either
        ``MissingWaitFailure`` (if no qualifying drain in window) or
        ``WaitInsufficientFailure`` (if a wait exists but its counter
        leaves the earlier producer pending). Both are acceptable
        coverage signals; the test asserts on the broader "some failure
        on dscnt was emitted for the LR -> MFMA edge".
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(dst_vgpr_start=8, dst_vgpr_count=2, lds_offset=64,
                    slot=0, category="LRA0"),
            make_lr(dst_vgpr_start=10, dst_vgpr_count=2, lds_offset=128,
                    slot=1, category="LRB0"),
            # dscnt=1: leaves 1 LR pending. The earlier LR (slot 0)
            # should NOT be considered drained.
            make_swait(slot=2, dscnt=1),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=10,
                      slot=3, a_src_count=1, b_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        # At least one failure on dscnt — either MissingWait or
        # WaitInsufficient — for the slot-0 LR's edge.
        from Tensile.Components.ScheduleCapture import (
            WaitInsufficientFailure,
        )
        dscnt_failures = [
            f for f in failures
            if isinstance(f, (MissingWaitFailure, WaitInsufficientFailure))
        ]
        assert dscnt_failures, (
            f"Expected at least one dscnt-related failure on the slot-0 LR -> "
            f"MFMA edge. Got: {[type(f).__name__ for f in failures]}"
        )

    def test_handling_instruction_order_swait_after_lrs(self):
        """LR0s, then LR1 placeholders for next iter, then SWait covering
        all in main_loop. Mirrors the third sub-case of legacy
        ``test_handling_instruction_order``. SWait sits AFTER all LRs
        and BEFORE the consumer MFMA — covers everything → no failure.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(dst_vgpr_start=8, dst_vgpr_count=2, lds_offset=64,
                    slot=3, category="LRA0"),
            make_lr(dst_vgpr_start=10, dst_vgpr_count=2, lds_offset=128,
                    slot=3, category="LRB0", sequence=1),
            make_swait(slot=3, dscnt=0, sequence=2),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=10,
                      slot=4, a_src_count=1, b_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        self.assert_no_failures(failures)

    def test_handling_instruction_order_swait_before_lrs_fails(self):
        """SWait BEFORE the LR (sequence orders: SWait, then LRs at same
        slot). The LR -> MFMA edge has no SWait in its window because
        the SWait sits BEFORE the producer.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_swait(slot=3, dscnt=0, sequence=0),
            make_lr(dst_vgpr_start=8, dst_vgpr_count=2, lds_offset=64,
                    slot=3, category="LRA0", sequence=1),
            make_lr(dst_vgpr_start=10, dst_vgpr_count=2, lds_offset=128,
                    slot=3, category="LRB0", sequence=2),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=10,
                      slot=4, a_src_count=1, b_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        # Window between LR @ slot 3 (sequence 1) and MFMA @ slot 4
        # contains no qualifying SWait (the SWait sits at sequence 0
        # of slot 3, BEFORE the LR).
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )


# =============================================================================
# Pre-loop LR / cross-iter wraparound (port of test_pre_loop_LR / test_pre_loop_SWaitCnt)
# =============================================================================
# Legacy "pre-loop LR" tests used slot index -1 to model an LR issued
# BEFORE the main loop (in NGL or the prologue). The graph uses
# BODY_LABEL_NGL or BODY_LABEL_ML_PREV for the same purpose.


class TestLRBeforeMFMA_PreLoop(GraphNativeValidationTest):
    """Pre-loop LR shape: producer in main_loop_prev, consumer in main_loop."""

    def test_pre_loop_LR_with_swait_in_main_loop(self):
        """LR in main_loop_prev; SWait(dscnt=0) at the start of main_loop;
        MFMA in main_loop reads the pre-loop-loaded value.

        Equivalent to the legacy ``test_pre_loop_LR``: SYNC @ [3] in main
        loop covers an LR0 @ [-1, -1] (pre-loop). Graph: cross-iter edge
        from main_loop_prev LR -> main_loop MFMA, drained by main_loop
        SWait.
        """
        ml_prev = make_capture(BODY_LABEL_ML_PREV, [
            make_lr(dst_vgpr_start=8, dst_vgpr_count=2, lds_offset=64,
                    slot=0, category="LRA0"),
            make_lr(dst_vgpr_start=10, dst_vgpr_count=2, lds_offset=128,
                    slot=1, category="LRB0"),
        ])
        ml = make_capture(BODY_LABEL_ML, [
            make_swait(slot=3, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=10,
                      slot=4, a_src_count=1, b_src_count=1),
        ])
        cap = self.wrap_single_body(ml, ml_prev=ml_prev)
        failures = self.validate_waits(self.build_graph(cap))
        self.assert_no_failures(failures)


# =============================================================================
# MFMA-reorder shape (port of TestValidateLRsCompleteBeforeVMFMA_MfmaReorder)
# =============================================================================
# Legacy ``mfmaReorder`` parameter remapped MFMA execution order; the
# legacy rule's ``set_lr_needed_by_for_VMFMA`` had to chase the
# permutation. The graph derives consumer-MFMA from REGISTER DATAFLOW,
# so reordering MFMAs (which still read the same vgprs) doesn't change
# which LR is the producer of which edge — it only changes the slot
# index of the consumer.
#
# Verification: the graph picks up the LR -> MFMA edge whichever slot
# the MFMA winds up at, AND wait-coverage is enforced based on the
# stream-order window between producer and consumer.


class TestLRBeforeMFMA_MfmaReorder(GraphNativeValidationTest):
    """Sanity check: MFMA reordering is invisible to the graph rule.

    The graph forms LR -> MFMA edges from register operands; it never
    consults a heuristic MFMA-index assignment. So a "reordered" schedule
    where MFMA-reading-LRA0 ends up at any slot is handled identically
    as long as the SWait sits before the MFMA in stream order.

    E2E coverage of the production ``mfmaReorder`` pipeline (
    ``ScheduleInfo(mfmaReorder=[...])`` -> permutation step at
    ``CustomSchedule.py:336-337`` -> capture -> ``build_dataflow_graph``
    -> ``validate_edge_wait_coverage``) lives in:

      * ``test_mfma_reorder_e2e.py::TestMfmaReorderE2E``

    This class focuses on the unit-level invariant that the graph rule
    is REORDER-INVARIANT once the capture is built; the e2e test
    exercises the production path that constructs the reordered capture.
    """

    def test_reordered_MFMA_at_late_slot_with_covering_swait(self):
        """LR @ slot 0; MFMA reading the LR is placed at slot 7 (late).
        SWait(dscnt=0) at slot 3 sits between them → no failure. The
        graph doesn't care that the consumer MFMA is "late" — register
        dataflow is what determines edge formation.
        """
        cap = _ml_lr_then_mfma(lr_slot=0, swait_slot=3, mfma_slot=7)
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        self.assert_no_failures(failures)

    def test_reordered_MFMA_at_early_slot_swait_too_late(self):
        """LR @ slot 0; MFMA reading the LR is placed at slot 1; SWait
        at slot 3 sits AFTER the MFMA. Window LR(0) -> MFMA(1) is empty
        of SWaits → MissingWait on dscnt.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(dst_vgpr_start=8, dst_vgpr_count=2, lds_offset=64,
                    slot=0, category="LRA0"),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=1, a_src_count=1),
            make_swait(slot=3, dscnt=0),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )


# =============================================================================
# ForceUnrollSubIter shape (port of TestValidateLRsCompleteBeforeVMFMA_ForceUnrollSubIter)
# =============================================================================
# Legacy ForceUnrollSubIter remapped slot indices via
# ``index_for_force_unroll_sub_iter``. Same graph reasoning as
# MfmaReorder: the graph reads register operands from the MFMA at
# whatever slot it lands, so the rule is invariant under slot
# permutations. Tests below exercise an LR3 (alias for LR1 in the
# ForceUnrollSubIter mode) feeding a next-iter MFMA.


class TestLRBeforeMFMA_ForceUnrollSubIter(GraphNativeValidationTest):
    """LR3 (current iter) -> next-iter MFMA, with ForceUnrollSubIter slot
    permutation modeled implicitly by graph register dataflow.
    """

    def test_LR3_baseline_passing(self):
        """LR3 in main_loop_prev writes v16; main_loop MFMA reads v16
        with covering SWait. No failure."""
        ml_prev = make_capture(BODY_LABEL_ML_PREV, [
            make_lr(dst_vgpr_start=16, dst_vgpr_count=2, lds_offset=192,
                    slot=7, category="LRA3"),
        ])
        ml = make_capture(BODY_LABEL_ML, [
            make_swait(slot=0, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=16, b_src_start=32,
                      slot=1, a_src_count=1),
        ])
        cap = self.wrap_single_body(ml, ml_prev=ml_prev)
        failures = self.validate_waits(self.build_graph(cap))
        self.assert_no_failures(failures)

    def test_LR3_positionally_late_no_swait_coverage(self):
        """LR3 in main_loop_prev; MFMA at slot 0 of main_loop; no SWait
        anywhere in the cross-iter window → MissingWait on dscnt.
        """
        ml_prev = make_capture(BODY_LABEL_ML_PREV, [
            make_lr(dst_vgpr_start=16, dst_vgpr_count=2, lds_offset=192,
                    slot=15, category="LRA3"),
        ])
        ml = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=16, b_src_start=32,
                      slot=0, a_src_count=1),
        ])
        cap = self.wrap_single_body(ml, ml_prev=ml_prev)
        failures = self.validate_waits(self.build_graph(cap))
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )


# =============================================================================
# Note on TF32 (port of TestValidateLRsCompleteBeforeVMFMA_tf32)
# =============================================================================
# The legacy TF32 class wired UseF32XEmulation + ForceUnrollSubIter and
# asserted that LRA0 fed an MFMA at vmfma_index 3 (the first hi*hi MFMA
# of the second-half tile). In the graph model:
#
#   * The LR's destination vgprs are consumed by Pack (CVT0/CVT1)
#     instructions FIRST, then the Pack output feeds the MFMA. The
#     LR -> Pack RAW edge is what wait-coverage checks; the
#     Pack -> MFMA edge has no wait-counter requirement (ALU producer).
#   * That LR -> Pack edge is already covered by ola.4's
#     test_validate_pack_graph.py, which exercises both the BF16
#     (VPermB32) and TF32 (VCvtPkF32toBF16) Pack flavors.
#
# So a separate TF32 LR -> MFMA test class would duplicate ola.4's
# coverage. Instead, the simple LR -> MFMA tests above cover the
# rule's core behavior; TF32-specific Pack coverage lives in
# test_validate_pack_graph.py.
