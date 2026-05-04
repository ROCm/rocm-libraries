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
"""Graph-native ports of test_ValidateGlobalReadsNotTooEarly.py — bead ola.2.

Sub-task ola.2 of the CMS validation migration epic (`br show
rocm-libraries-ola`): replace structural rule `add_gr_not_too_early_constraints`
(CMSValidator.py:3113) — and its `GR.must_start_after` constraint annotations —
with graph-native edge-coverage checks.

Rule analysis:
  * The structural rule encodes:
        LR0 (read LDS) -> SWait(dscnt=0) -> SBarrier -> GR (write LDS)
    which is exactly `edge_kind="lr_to_gr_lds_reuse"` in the dataflow graph
    (ScheduleCapture.py `_collect_pattern`).
  * `validate_edge_wait_coverage` already emits the expected Failure types
    (MissingWait / WaitInsufficient / MissingBarrier); `compare_graphs` ->
    `diagnose_missing_edge` covers the cases where the edge is absent from
    the subject graph entirely (e.g., SBarrier-before-SWait, GR before any
    barrier, etc.).
  * DtlPlusLdsBuf cross-iteration variant: GR in iter N depends on LR0 in
    iter N-1. `FourPartCapture.main_loop_prev` already models the prev-
    iteration body and `_collect_pattern` walks across bodies in execution
    order, so cross-body edges form naturally.

Mutation-smell-test (acceptance criterion from r4o, recorded on the parent
bead): commenting out the `lr_to_gr_lds_reuse` branch in
`_classify_edge_coverage` (ScheduleCapture.py, currently lines 4001-4014)
must break at least one test in this file. `test_missing_barrier_lr_drained`
satisfies this — it asserts a `MissingBarrierFailure` driven through the
`validate_edge_wait_coverage` path (subject graph has a covering wait but
no barrier between drain and consumer).

Production-side deletions (bead `ola.2` phase 2, landed):
  * `add_gr_not_too_early_constraints`, `set_gr_must_start_after_from_lr0s`,
    `set_gr_must_start_after_from_grinc`, `apply_must_start_after_barriers`,
    `_apply_must_start_after_barriers_single`, `GRAfterLRRule`, the
    `GR.must_start_after` / `GR.must_start_after_barriered_at` fields, and
    `GlobalRead._validate_must_start_after` are ALL deleted from
    CMSValidator.py.
  * The legacy `test_ValidateGlobalReadsNotTooEarly.py` is also deleted —
    its coverage now lives in this file.
  * The GRInc -> GR ordering arc previously consumed by
    `test_GRMustStartAfterGRInc.py` is now graph-native via the SRD sgpr
    RAW edge (GRInc's SAddU32 writes the SRD sgpr that GR's BufferLoad
    reads); see the rewritten `test_GRMustStartAfterGRInc.py`.
"""

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    MissingBarrierFailure,
    MissingWaitFailure,
    OrderInvertedFailure,
    SLOT_KIND_MFMA,
    SlotKey,
    TaggedInstruction,
    WaitInsufficientFailure,
)

from dataflow_fixtures import (
    make_capture, make_dtl_buffer_load, make_gr, make_lr, make_lw,
    make_sbarrier, make_swait,
)
from graph_native_validation_base import GraphNativeValidationTest


# =============================================================================
# Helpers
# =============================================================================
# All graph-native fixtures keep producer/consumer vgpr ranges below 200 so
# they don't collide with the filler-MFMA ranges used by `wrap_single_body`
# for unused bodies (see graph_native_validation_base._FILLER_RANGES).


# Distinct LDS slots for A/B sides — keeps the visual mapping to the legacy
# tests obvious. The graph builder is content-blind for LDS-reuse pattern
# matching, so the actual offsets don't change edge formation.
_LDS_A = 64
_LDS_B = 128

# Distinct vgpr ranges for LRA0/LRB0/GRA/GRB. Each LR writes a 4-vgpr range,
# each GR writes a different 4-vgpr range. They DON'T overlap because the
# LDS-reuse edge collector pattern-matches on category, not on register.
_LRA0_DST = 8     # v8..v11
_LRB0_DST = 16    # v16..v19
_GRA_DST = 40     # v40..v43
_GRB_DST = 48     # v48..v51


def _lra0(slot, *, sequence=0, dst_start=_LRA0_DST, lds_offset=_LDS_A):
    return make_lr(dst_start, 4, lds_offset, slot=slot,
                   category="LRA0", sequence=sequence)


def _lrb0(slot, *, sequence=0, dst_start=_LRB0_DST, lds_offset=_LDS_B):
    return make_lr(dst_start, 4, lds_offset, slot=slot,
                   category="LRB0", sequence=sequence)


def _gra(slot, *, sequence=0):
    return make_gr(_GRA_DST, 4, srd_sgpr_start=12,
                   immediate_offset=_LDS_A, slot=slot,
                   category="GRA", sequence=sequence)


def _grb(slot, *, sequence=0):
    return make_gr(_GRB_DST, 4, srd_sgpr_start=20,
                   immediate_offset=_LDS_B, slot=slot,
                   category="GRB", sequence=sequence)


# =============================================================================
# Standard (non-DtlPlusLdsBuf) tests
# =============================================================================
# Port of TestValidateGlobalReadsNotTooEarly. Each test builds a single ML
# body and runs validate_edge_wait_coverage (or compare_graphs against a
# canonical reference when the subject lacks the edge entirely).


class TestGRNotTooEarlyGraph(GraphNativeValidationTest):
    """Graph-native port of TestValidateGlobalReadsNotTooEarly.

    Maps the legacy `add_gr_not_too_early_constraints` Failure shapes onto
    graph-native equivalents:

      ConstraintViolationFailure (LR not guaranteed by SWait)
        -> MissingWaitFailure / WaitInsufficientFailure
      MissingBarrierFailure (legacy: SBarrier missing in window)
        -> MissingBarrierFailure (same class; either via wait-coverage
           when the edge IS formed but no barrier follows the drain, or
           via compare_graphs when the edge is missing entirely).
    """

    # -------------------------------------------------------------------------
    # Positive tests
    # -------------------------------------------------------------------------

    def test_basic_passing(self):
        """Canonical placement: LR0s -> SWait(dscnt=0) -> SBarrier -> GRs."""
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            _lrb0(slot=1),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
            _gra(slot=5),
            _grb(slot=6),
        ]))
        failures = self.validate_waits(self.build_graph(cap))
        self.assert_no_failures(failures)
        # Pin: at least one lr_to_gr_lds_reuse edge formed.
        self.assert_edge_exists(self.build_graph(cap),
                                edge_kind="lr_to_gr_lds_reuse")

    def test_two_pair_drains_passing(self):
        """Two full SWait(dscnt=0)+SBarrier pairs. Both fully drain all
        outstanding LR0s before each GR — passes.

        Note: unlike the legacy structural rule, the graph collector pairs
        ALL prior LR0s with each downstream GR (it's category-based, not
        operand-aware). So a partial drain (dscnt=K with K>0) leaves edges
        with insufficient coverage. Only full drains (dscnt=0) work in the
        graph model when multiple LR0 categories are present.
        """
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            _lrb0(slot=1),
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            _gra(slot=4),
            make_swait(slot=5, dscnt=0),
            make_sbarrier(slot=6),
            _grb(slot=7),
        ]))
        failures = self.validate_waits(self.build_graph(cap))
        self.assert_no_failures(failures)

    def test_multiple_lr0s_full_drain_passing(self):
        """Multiple LR0s, two full-drain SWait+SBarrier pairs. All edges
        properly covered."""
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            _lrb0(slot=1),
            _lra0(slot=2, sequence=1),
            _lrb0(slot=3, sequence=1),
            make_swait(slot=4, dscnt=0),
            make_sbarrier(slot=4, sequence=1),
            _gra(slot=5),
            make_swait(slot=6, dscnt=0),
            make_sbarrier(slot=6, sequence=1),
            _grb(slot=7),
        ]))
        failures = self.validate_waits(self.build_graph(cap))
        self.assert_no_failures(failures)

    def test_swap_global_read_order_passing(self):
        """SwapGlobalReadOrder is purely a kernel-side semantic flag — in
        graph-native terms the LDS-reuse edge collector pattern-matches on
        LR/GR categories, not on A vs B identity. So a swap test is just
        a different placement of the same LR0/GR pair. Verifies that the
        graph correctly forms edges regardless of operand identity."""
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            make_swait(slot=1, dscnt=0),
            make_sbarrier(slot=2),
            _grb(slot=3),  # GRB issued first under swap
            _lrb0(slot=4),
            make_swait(slot=5, dscnt=0),
            make_sbarrier(slot=6),
            _gra(slot=7),
        ]))
        failures = self.validate_waits(self.build_graph(cap))
        self.assert_no_failures(failures)

    # -------------------------------------------------------------------------
    # Negative tests — wait insufficient (legacy ConstraintViolationFailure)
    # -------------------------------------------------------------------------

    def test_negative_wait_insufficient(self):
        """SWaitCnt(dscnt=1) leaves 1 LR0 outstanding when the GR fires.
        The edge IS formed (SBarrier present in the right window). The
        graph collector pairs ALL prior LRs with the GR — so the more
        recently-issued LR0 (LRB0) remains in flight at the wait, and
        the WaitInsufficient branch of `_classify_edge_coverage` fires.

        Legacy: ConstraintViolationFailure (LR not guaranteed).
        Graph: WaitInsufficientFailure (more specific diagnostic — names
        the under-drained wait by name and reports the queue depth).
        """
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            _lrb0(slot=1),
            make_swait(slot=3, dscnt=1),  # leaves 1 outstanding
            make_sbarrier(slot=4),
            _gra(slot=5),
            _grb(slot=6),
        ]))
        failures = self.validate_waits(self.build_graph(cap))
        self.assert_failures_contain(
            failures, cls=WaitInsufficientFailure,
        )

    def test_negative_wait_too_late(self):
        """LR0 -> GR with SWait+SBarrier sequence sitting AFTER the GR. In
        subject the LR0 -> SWait -> SBarrier -> GR pattern doesn't match
        because the GR comes BEFORE the wait. Reference has the GR after
        the SBarrier; compare_graphs sees the missing edge and fires
        MissingWaitFailure (no wait in window between LR and GR)."""
        ref_cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            _gra(slot=5),
        ]))
        subj_cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            _gra(slot=1),  # too early: before any wait/barrier
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
        ]))
        failures = self.compare(ref_cap, subj_cap)
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )

    # -------------------------------------------------------------------------
    # Negative tests — barrier coverage
    # -------------------------------------------------------------------------
    # MUTATION-SMELL-TEST anchor: this test specifically requires the
    # `lr_to_gr_lds_reuse` branch in `_classify_edge_coverage`
    # (ScheduleCapture.py validate_edge_wait_coverage helper) to fire.
    # Commenting out that branch must break this test. Verified by
    # running the test against a locally-mutated ScheduleCapture.

    def test_missing_barrier_lr_drained(self):
        """MUTATION-SMELL-TEST anchor: this test specifically exercises
        the `lr_to_gr_lds_reuse` barrier branch in
        `_classify_edge_coverage` (ScheduleCapture.py
        validate_edge_wait_coverage helper, currently lines 4001-4014).
        Commenting out that branch must break this test (no
        MissingBarrierFailure emitted).

        Setup:
          LRA0 @ 0, SWait(dscnt=0) @ 1, SBarrier @ 2,
          SWait(dscnt=0) @ 4, GRA @ 5

        The state machine in `_collect_pattern` matches LR -> first SWait
        -> first SBarrier -> consumers; it forms an edge from LRA0(0) to
        GRA(5). Wait coverage walks every wait in the
        (LR.position, GR.position) window — sees both SWaits, both drain
        the LR. `last_drain` becomes the LATER SWait (slot 4). The
        barrier check then runs barriers_in_window(start=slot4, end=slot5)
        — empty (the SBarrier sits at slot 2, BEFORE last_drain). The
        LDS-reuse barrier branch fires MissingBarrierFailure.

        If the branch is commented out: the edge still passes
        `_classify_edge_coverage` (no barrier check) and no failure is
        emitted — test breaks.
        """
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            make_swait(slot=1, dscnt=0),
            make_sbarrier(slot=2),
            make_swait(slot=4, dscnt=0),  # later drain
            _gra(slot=5),
        ]))
        graph = self.build_graph(cap)
        # Verify the edge exists (state machine matched the first
        # SWait+SBarrier pair).
        self.assert_edge_exists(graph, edge_kind="lr_to_gr_lds_reuse")
        failures = self.validate_waits(graph)
        f = self.assert_failures_contain(failures, cls=MissingBarrierFailure)
        assert f.role == "must_start_after"

    def test_negative_barrier_before_swait(self):
        """SBarrier at 2 comes BEFORE SWait(dscnt=0) at 3. The state
        machine in `_collect_pattern` requires SWait first then SBarrier,
        so subject doesn't form the edge — compare_graphs against the
        canonical reference fires MissingBarrierFailure (the last_drain
        is the SWait at 3, but no SBarrier follows it before the GR at 5)."""
        ref_cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            _gra(slot=5),
        ]))
        subj_cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            make_sbarrier(slot=2),
            make_swait(slot=3, dscnt=0),
            _gra(slot=5),
        ]))
        failures = self.compare(ref_cap, subj_cap)
        self.assert_failures_contain(failures, cls=MissingBarrierFailure)

    def test_negative_barrier_too_late_after_gr(self):
        """SBarrier sits AFTER the GR — edge is not formed in subject.
        Legacy: MissingBarrierFailure. Graph: same, via compare_graphs."""
        ref_cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            _gra(slot=5),
        ]))
        subj_cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            make_swait(slot=2, dscnt=0),
            _gra(slot=5),
            make_sbarrier(slot=6),  # Too late for the GR
        ]))
        failures = self.compare(ref_cap, subj_cap)
        self.assert_failures_contain(failures, cls=MissingBarrierFailure)

    def test_negative_one_gr_too_early(self):
        """Two GRs: the EARLY GR is misplaced before any drain. The graph
        identity for a GR is `(class_tag, loop_idx, canonical_render)` —
        the render encodes the destination vgpr range — so each GR needs
        a DISTINCT vgpr range to be a distinct node identity in the
        compare_graphs diff. We use two GRA-categorized GRs with non-
        overlapping dst ranges (v40 and v44).

        compare_graphs against a reference where both GRs sit after the
        SBarrier emits MissingWaitFailure for the early GR's missing edge.
        """
        # Two distinct GR identities: dst v40..v43 and dst v44..v47.
        gra_early_kwargs = dict(srd_sgpr_start=12, immediate_offset=_LDS_A,
                                category="GRA")
        ref_cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            make_gr(40, 4, slot=4, **gra_early_kwargs),
            make_gr(44, 4, slot=5, **gra_early_kwargs),
        ]))
        subj_cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            make_gr(40, 4, slot=1, **gra_early_kwargs),  # too early
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            make_gr(44, 4, slot=5, **gra_early_kwargs),  # safe
        ]))
        failures = self.compare(ref_cap, subj_cap)
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )


# =============================================================================
# DtlPlusLdsBuf cross-iteration tests
# =============================================================================
# Port of TestValidateGlobalReadsNotTooEarlyDtlPlusLdsBuf. With DtlPlusLdsBuf,
# GR in iter N depends on LR0 in iter N-1 (cross-iteration). The graph models
# this naturally: put LR0 in `main_loop_prev`, GR in `main_loop`. The
# `_collect_pattern` collector walks bodies in execution order
# (ML_PREV -> ML -> NGL -> NLL) so the cross-body LR -> GR edge forms.


class TestGRNotTooEarlyDtlPlusLdsBufGraph(GraphNativeValidationTest):
    """Graph-native port of TestValidateGlobalReadsNotTooEarlyDtlPlusLdsBuf.

    Cross-iteration LR0 -> GR dependency. The legacy structural rule
    distinguishes prev-iter from same-iter via SchedulePosition comparison
    over a Timeline; the graph models the same shape via main_loop_prev
    body wiring + the per-body SchedulePosition.loop_index ordering.
    """

    def test_basic_passing(self):
        """LR0 in ML-1 with proper SWait+SBarrier in ML-1, GR in ML."""
        ml_prev = make_capture(BODY_LABEL_ML_PREV, [
            _lra0(slot=0),
            _lrb0(slot=1),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
        ])
        ml = make_capture(BODY_LABEL_ML, [
            # Same-iter LR0 also present (DtlPlusLdsBuf doesn't FORBID
            # same-iter reads, only their dependency on the GR is no-op).
            _lra0(slot=0),
            _lrb0(slot=1),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
            _gra(slot=5),
            _grb(slot=6),
        ])
        cap = self.wrap_single_body(ml, ml_prev=ml_prev)
        failures = self.validate_waits(self.build_graph(cap))
        self.assert_no_failures(failures)

    def test_gr_before_same_iter_sync_safe(self):
        """GR at slot 2 in ML, but ML-1 had its own complete drain pattern.
        The cross-body edge (ML-1 LR0 -> ML GR) is covered by ML-1's
        SWait+SBarrier (which precede the ML body in stream order)."""
        ml_prev = make_capture(BODY_LABEL_ML_PREV, [
            _lra0(slot=0),
            _lrb0(slot=1),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
        ])
        ml = make_capture(BODY_LABEL_ML, [
            _gra(slot=2),  # GR early in ML — covered by ML-1's drain
            _lra0(slot=3),
            _lrb0(slot=3, sequence=1),
            make_swait(slot=4, dscnt=0),
            make_sbarrier(slot=5),
            _grb(slot=6),
        ])
        cap = self.wrap_single_body(ml, ml_prev=ml_prev)
        failures = self.validate_waits(self.build_graph(cap))
        self.assert_no_failures(failures)

    def test_negative_prev_iter_lr0_not_drained(self):
        """ML-1's LR0 is misplaced (slot 5, AFTER ML-1's drain). The
        cross-body edge (ML-1 LR0 @ slot 5 -> ML GR @ slot 2) is missing
        from the subject because no SWait+SBarrier sequence sits between
        them. compare_graphs against a canonical reference (LR0 @ slot 0
        in ML-1, properly drained) emits MissingWaitFailure for the
        missing edge."""
        ref_ml_prev = make_capture(BODY_LABEL_ML_PREV, [
            _lra0(slot=0),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
        ])
        ref_ml = make_capture(BODY_LABEL_ML, [
            _gra(slot=2),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
            _grb(slot=6),
        ])
        ref_cap = self.wrap_single_body(ref_ml, ml_prev=ref_ml_prev)

        subj_ml_prev = make_capture(BODY_LABEL_ML_PREV, [
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
            _lra0(slot=5),  # too late — after ML-1's drain
        ])
        subj_ml = make_capture(BODY_LABEL_ML, [
            _gra(slot=2),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
            _grb(slot=6),
        ])
        subj_cap = self.wrap_single_body(subj_ml, ml_prev=subj_ml_prev)
        failures = self.compare(ref_cap, subj_cap)
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )

    def test_multiple_prev_iter_lr0s_passing(self):
        """Multiple LR0s in ML-1, all drained by ML-1's SWait. GR in ML
        at slot 5 well past ML-1's drain. Should pass."""
        ml_prev = make_capture(BODY_LABEL_ML_PREV, [
            _lra0(slot=0),
            _lra0(slot=2, sequence=1),
            _lrb0(slot=1),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
        ])
        ml = make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            _lrb0(slot=1),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
            _gra(slot=5),
            _grb(slot=6),
        ])
        cap = self.wrap_single_body(ml, ml_prev=ml_prev)
        failures = self.validate_waits(self.build_graph(cap))
        self.assert_no_failures(failures)

    def test_negative_one_prev_iter_lr0_not_drained(self):
        """Two LR0s in ML-1: one at slot 0 (drained by ML-1's SWait at 3),
        one at slot 5 (NOT drained — placed after the drain). The
        not-drained one creates an LR0 with no covering wait/barrier
        sequence before the ML GR. compare_graphs against a reference
        where both LR0s sit before the drain emits MissingWaitFailure
        for the missing edge."""
        ref_ml_prev = make_capture(BODY_LABEL_ML_PREV, [
            _lra0(slot=0),
            _lra0(slot=2, sequence=1),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
        ])
        ref_ml = make_capture(BODY_LABEL_ML, [
            _gra(slot=2),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
            _grb(slot=6),
        ])
        ref_cap = self.wrap_single_body(ref_ml, ml_prev=ref_ml_prev)

        subj_ml_prev = make_capture(BODY_LABEL_ML_PREV, [
            _lra0(slot=0),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
            _lra0(slot=5, sequence=1),  # not drained in ML-1
        ])
        subj_ml = make_capture(BODY_LABEL_ML, [
            _gra(slot=2),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
            _grb(slot=6),
        ])
        subj_cap = self.wrap_single_body(subj_ml, ml_prev=subj_ml_prev)
        failures = self.compare(ref_cap, subj_cap)
        self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )


# =============================================================================
# DTL vs non-DTL path coverage — bead wa6
# =============================================================================
# Pin the two distinct production code paths feeding the LR0 -> GR LDS-reuse
# invariant:
#
#   - DTL path (DirectToLds=1): BufferLoad has dst=None, writes directly to
#     LDS via the m0 register. _DTLBufferLoadRule publishes m0 as a read for
#     such loads. The LR0 -> GR cross-iter chain is collected by the
#     `lr_to_gr_lds_reuse` barrier-pattern collector regardless of DTL vs not.
#
#   - Non-DTL path: BufferLoad has a vgpr dst; the chain is GR -> LW (vgpr
#     RAW edge, vlcnt counter) -> next-iter LR. The GR -> LW edge is a
#     `raw_intrawave` edge from the per-byte resolver; the LR0 -> GR cross-
#     iter LDS-reuse handoff is the SAME barrier-pattern edge as the DTL path
#     (collector is category-based, not instance-based).
#
# `TestDTLPathCoverage` and `TestNonDTLPathCoverage` exercise each path
# independently with discriminating mutation-smell-tests. `TestCrossModeParity`
# proves the LR0 -> GR LDS-reuse Failure shape is identical across modes for
# the analogous violation, so a future regression that breaks the lr_to_gr
# branch in one mode but not the other is impossible without breaking both
# tests in lockstep.
#
# Helpers below mirror the wx9.7 DTL m0 tests' shape (test_dataflow_graph_
# register_gaps.TestDTLm0Tracking): the m0 setter is a real rocisa
# SMovB32 to mgpr(0), tagged with category="GRA" so the per-byte resolver
# routes it via `_GenericALURule` and the m0 RAW edge to the DTL BufferLoad
# forms via the m0 register identity.


# Distinct LDS slot/vgpr ranges for the DTL/non-DTL chains. The non-DTL LW
# reads the same vgpr the GR wrote (that is the `raw_intrawave` edge under
# test); cross-iter tests share the same LDS slot between prev-iter LRA0
# and curr-iter GR.
_NDLDS = 64           # LDS offset shared across the chain
_NDVGPR = 60          # vgpr range carrying GR.dst -> LW.src (non-DTL)
_NDVGPR_COUNT = 4
_DTL_VADDR = 40       # vaddr vgpr for the DTL BufferLoad
_DTL_SRD = 12         # SRD sgpr base for both DTL and non-DTL


def _smov_m0_set(slot, *, sequence=0, category="GRA"):
    """Build a real rocisa SMovB32 to mgpr(0) wrapped in a TaggedInstruction.

    Mirrors the m0-setter shape emitted by KWA at lines 10049-10072 (DTL
    default path: SMovB32(dst=mgpr(0), src=sgpr("LocalWriteAddrA"))). Used
    by the DTL path tests to publish a producer for the m0 read that
    `_DTLBufferLoadRule` exposes on the following BufferLoad.

    Tagged with category="GRA" because in production the m0 setter is
    emitted within the GRA emission group; the per-byte resolver doesn't
    care about category for register tracking, only for the
    `_collect_pattern` LR/GR pattern matching (which pairs by category).
    """
    # Lazy imports — keep the rocisa cost out of fake-only test runs.
    from rocisa.container import sgpr, mgpr
    from rocisa.instruction import SMovB32
    inst = SMovB32(dst=mgpr(0), src=sgpr("LocalWriteAddrA", 1))
    return TaggedInstruction(
        inst=inst,
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def _nd_gr(slot, *, sequence=0, category="GRA"):
    """Non-DTL BufferLoad: writes vgpr (_NDVGPR..+_NDVGPR_COUNT)."""
    return make_gr(
        _NDVGPR, _NDVGPR_COUNT, srd_sgpr_start=_DTL_SRD,
        immediate_offset=_NDLDS, slot=slot, category=category,
        sequence=sequence,
    )


def _nd_lw(slot, *, sequence=0):
    """Non-DTL LW: reads the same vgpr the non-DTL GR wrote."""
    return make_lw(
        _NDVGPR, _NDVGPR_COUNT, lds_offset=_NDLDS, slot=slot,
        category="LWA", sequence=sequence,
    )


def _dtl_gr(slot, *, sequence=0, category="GRA"):
    """DTL BufferLoad (dst=None) — implicitly reads m0 via _DTLBufferLoadRule."""
    return make_dtl_buffer_load(
        vaddr_vgpr_start=_DTL_VADDR, srd_sgpr_start=_DTL_SRD,
        slot=slot, category=category, immediate_offset=_NDLDS,
        sequence=sequence,
    )


def _prev_lra0_drained(*, lra0_dst_start=8):
    """ML-1 body: LRA0 then full dscnt drain + barrier. Used as the
    canonical reference for DTL/non-DTL cross-iter tests."""
    return make_capture(BODY_LABEL_ML_PREV, [
        make_lr(lra0_dst_start, 4, lds_offset=_NDLDS, slot=0, category="LRA0"),
        make_swait(slot=2, dscnt=0),
        make_sbarrier(slot=3),
    ])


def _prev_lra0_misplaced(*, lra0_dst_start=8):
    """ML-1 body: dscnt drain + barrier emitted FIRST, then LRA0 issued
    AFTER — the LRA0 has no covering wait/barrier before the next-iter GR."""
    return make_capture(BODY_LABEL_ML_PREV, [
        make_swait(slot=2, dscnt=0),
        make_sbarrier(slot=3),
        make_lr(lra0_dst_start, 4, lds_offset=_NDLDS, slot=5, category="LRA0"),
    ])


# -----------------------------------------------------------------------------
# DTL path coverage
# -----------------------------------------------------------------------------


class TestDTLPathCoverage(GraphNativeValidationTest):
    """Pin the DTL BufferLoad code path: dst=None + implicit m0 read.

    These tests use `make_dtl_buffer_load` so the BufferLoad goes through
    `_DTLBufferLoadRule` (publishes `reads=(m0, srd)`) rather than the
    non-DTL `_BufferLoadRule` (publishes `writes=(vgpr_dst,)`).

    Mutation-smell-tests pinned by this class:
      * Comment out `_DTLBufferLoadRule.extract`'s m0 publish (or the
        `applies` predicate so DTL BufferLoads route to `_BufferLoadRule`):
        `test_dtl_m0_edge_present_when_setter_present` no longer finds the
        m0 RAW edge and fails.
      * Comment out the `lr_to_gr_lds_reuse` branch in
        `_classify_edge_coverage`: `test_dtl_gr_before_prev_lr0_dscnt_drain`
        no longer emits MissingWaitFailure(counter_kind='dscnt') on the
        cross-iter LRA0 -> GRA edge.
    """

    def test_dtl_gr_before_prev_lr0_dscnt_drain(self):
        """DTL BufferLoad (dst=None, reads m0) in current iter consumes the
        LDS slot the previous iter's LRA0 read. The cross-iter
        `lr_to_gr_lds_reuse` edge requires a dscnt drain + barrier between
        the prev LRA0 and the curr DTL load.

        REF: prev LRA0 properly drained in ML-1, current DTL BufferLoad in ML.
        SUBJ: the dscnt drain + barrier is emitted in ML-1 BEFORE the LRA0,
        so the LRA0 is not covered before the next-iter GR.

        Asserts MissingWaitFailure(counter_kind='dscnt') on the cross-iter
        LRA0 -> GRA edge.
        """
        ref_cap = self.wrap_single_body(
            make_capture(BODY_LABEL_ML, [_dtl_gr(slot=0)]),
            ml_prev=_prev_lra0_drained(),
        )
        subj_cap = self.wrap_single_body(
            make_capture(BODY_LABEL_ML, [_dtl_gr(slot=0)]),
            ml_prev=_prev_lra0_misplaced(),
        )
        failures = self.compare(ref_cap, subj_cap)
        f = self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )
        assert f.producer.category == "LRA0"
        assert f.consumer.category == "GRA"

    def test_dtl_m0_edge_present_when_setter_present(self):
        """Positive structural pin: when an SMovB32 to mgpr(0) precedes
        the DTL BufferLoad, `_DTLBufferLoadRule` publishes `reads=(m0,...)`
        and the per-byte resolver forms a `raw_intrawave` edge from the
        SMov to the BufferLoad with `resource.regType == 'm'`.

        Mutation pin: commenting out the `mgpr(0)` publish in
        `_DTLBufferLoadRule.extract` removes the m0 from the wrapper's
        reads, and this assertion fails (no edge with `regType == 'm'`).
        """
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _smov_m0_set(slot=0),
            _dtl_gr(slot=1),
        ]))
        graph = self.build_graph(cap)
        # Find the m0 RAW edge from the SMov to the DTL BufferLoad.
        m0_edges = [
            e for e in graph.edges
            if e.edge_kind == "raw_intrawave"
            and getattr(e.resource, "regType", None) == "m"
        ]
        assert len(m0_edges) >= 1, (
            f"Expected at least one m0 raw_intrawave edge from the SMov "
            f"to the DTL BufferLoad. Edges in graph: "
            f"{[(e.producer.category, e.consumer.category, e.edge_kind, getattr(e.resource, 'regType', None)) for e in graph.edges]}"
        )

    def test_dtl_m0_setter_missing_before_buffer_load(self):
        """Negative structural pin: with NO m0 setter in the stream, the
        DTL BufferLoad's m0 read has no producer — the per-byte resolver
        emits no `raw_intrawave` edge with `regType == 'm'` into the load.

        This test asserts the edge ABSENCE explicitly. Complementary to
        `test_dtl_m0_edge_present_when_setter_present`: the m0 read is
        published by `_DTLBufferLoadRule` regardless, but with no producer
        in the latest_writer map for the m0 byte-key, no edge forms —
        modelling the production hazard where a missing m0 update lets
        the BufferLoad write to the wrong LDS slot.
        """
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _dtl_gr(slot=0),  # No SMov m0 setter precedes this
        ]))
        graph = self.build_graph(cap)
        m0_edges = [
            e for e in graph.edges
            if e.edge_kind == "raw_intrawave"
            and getattr(e.resource, "regType", None) == "m"
        ]
        assert m0_edges == [], (
            f"Expected NO m0 RAW edge into the DTL BufferLoad when no m0 "
            f"setter precedes it; got: {m0_edges!r}"
        )

    def test_dtl_cross_iter_full_chain_passing(self):
        """Full positive: DTL chain across `main_loop_prev -> main_loop`.

        ML-1: LRA0 -> SWaitCnt(dscnt=0) -> SBarrier
        ML  : SMov m0 -> DTL BufferLoad (reads m0)

        Both the cross-iter `lr_to_gr_lds_reuse` edge (LRA0 -> GRA) and the
        m0 `raw_intrawave` edge (SMov -> DTL BufferLoad) form. Wait
        coverage emits no failures.
        """
        prev = _prev_lra0_drained()
        ml = make_capture(BODY_LABEL_ML, [
            _smov_m0_set(slot=0),
            _dtl_gr(slot=1),
        ])
        cap = self.wrap_single_body(ml, ml_prev=prev)
        graph = self.build_graph(cap)
        # Pin both edge surfaces.
        self.assert_edge_exists(graph, edge_kind="lr_to_gr_lds_reuse",
                                producer_category="LRA0",
                                consumer_category="GRA")
        m0_edges = [
            e for e in graph.edges
            if e.edge_kind == "raw_intrawave"
            and getattr(e.resource, "regType", None) == "m"
        ]
        assert m0_edges, "Expected at least one m0 RAW edge in the DTL chain"
        # Wait coverage clean.
        failures = self.validate_waits(graph)
        self.assert_no_failures(failures)


# -----------------------------------------------------------------------------
# Non-DTL path coverage
# -----------------------------------------------------------------------------


class TestNonDTLPathCoverage(GraphNativeValidationTest):
    """Pin the non-DTL BufferLoad code path: GR (vgpr dst) -> LW chain.

    The non-DTL BufferLoad has a vgpr dst, so the per-byte resolver forms a
    `raw_intrawave` edge from GR.dst to the LW that reads the vgpr. The
    counter for a GR producer is `vlcnt`, so a missing vlcnt drain in the
    GR -> LW window emits MissingWaitFailure(counter_kind='vlcnt').

    The next-iter LR0 reusing the LDS slot is captured by the SAME
    `lr_to_gr_lds_reuse` cross-iter pattern as the DTL path — the
    barrier-pattern collector is category-based, so the dscnt drain
    requirement applies to both modes uniformly. The DSCNT-drain test
    here therefore exercises the non-DTL BufferLoad as the GR consumer
    of the LR0->GR pattern.

    Mutation-smell-tests pinned by this class:
      * Comment out the GR vlcnt-counter mapping (the GR* branch in
        `counter_for`): `test_nondtl_gr_to_lw_vlcnt_drain_missing` would
        either crash with CaptureUnknownInstructionError or, if the branch
        defaults to dscnt, the explicit `counter_kind='vlcnt'` assertion
        fails.
      * Comment out the `lr_to_gr_lds_reuse` branch in
        `_classify_edge_coverage` / `diagnose_missing_edge`:
        `test_nondtl_lr0_to_gr_dscnt_drain_missing` no longer emits
        MissingWait on the LR0 -> GR cross-iter edge.
      * Comment out OrderInverted's Phase-1 check in `diagnose_missing
        _edge`: `test_nondtl_lw_after_gr_order_inverted` no longer surfaces
        the GR -> LW reorder.
    """

    def test_nondtl_gr_to_lw_vlcnt_drain_missing(self):
        """Non-DTL GR (writes vgpr v60..v63) -> LW (reads v60..v63). With no
        SWaitCnt(vlcnt=0) in the [GR, LW) window, the GR's vector-memory
        load hasn't necessarily completed when the LW issues — the LW
        sources potentially stale data.

        validate_edge_wait_coverage on a single graph emits MissingWait
        Failure(counter_kind='vlcnt') on the `raw_intrawave` GRA -> LWA
        edge.
        """
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _nd_gr(slot=0),
            # No SWaitCnt(vlcnt=0) here — the LW reads the vgpr while the
            # buffer load may still be in flight.
            _nd_lw(slot=2),
        ]))
        graph = self.build_graph(cap)
        failures = self.validate_waits(graph)
        f = self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="vlcnt",
        )
        assert f.producer.category == "GRA"
        assert f.consumer.category == "LWA"

    def test_nondtl_lr0_to_gr_dscnt_drain_missing(self):
        """Non-DTL cross-iter chain analogue of
        `TestDTLPathCoverage.test_dtl_gr_before_prev_lr0_dscnt_drain`.

        The LR0 -> GR cross-iter `lr_to_gr_lds_reuse` edge is the SAME
        pattern-collector edge for both modes (collector is category-based,
        not instance-based). For the non-DTL path the GR is a plain
        BufferLoad with a vgpr dst (uses `make_gr` / `_BufferLoadRule`)
        instead of `_DTLBufferLoadRule`. The dscnt-drain requirement is
        identical — pinned here for the non-DTL leaf so a future regression
        that special-cases DTL vs non-DTL in the lr_to_gr branch fails
        BOTH tests in lockstep.
        """
        ref_cap = self.wrap_single_body(
            make_capture(BODY_LABEL_ML, [_nd_gr(slot=0)]),
            ml_prev=_prev_lra0_drained(),
        )
        subj_cap = self.wrap_single_body(
            make_capture(BODY_LABEL_ML, [_nd_gr(slot=0)]),
            ml_prev=_prev_lra0_misplaced(),
        )
        failures = self.compare(ref_cap, subj_cap)
        f = self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )
        assert f.producer.category == "LRA0"
        assert f.consumer.category == "GRA"

    def test_nondtl_lw_after_gr_order_inverted(self):
        """Non-DTL: LW reordered to issue BEFORE the GR that produces the
        vgpr it reads. Reference order is GR (writes v60..v63) -> LW
        (reads v60..v63); subject inverts this to LW -> GR.

        diagnose_missing_edge's Phase-1 OrderInvertedFailure detection
        flags the inversion (default schedule has producer<consumer in
        stream order, subject has producer>consumer).

        Note: the bead text says "LW emitted AFTER the next-iter LR that
        consumes it"; the more direct OrderInverted surface for the
        non-DTL chain is the GR->LW vgpr RAW edge in the SAME body, since
        the LW->LR LDS handoff isn't tracked register-side (LDS is not in
        the per-byte resolver's address space). Same Failure type, same
        Phase-1 detection branch — the inversion of the GR->LW edge
        captures the equivalent "data flowing backward through the chain"
        defect.
        """
        ref_cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _nd_gr(slot=0),
            make_swait(slot=1, vlcnt=0),
            _nd_lw(slot=2),
        ]))
        subj_cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _nd_lw(slot=0),  # LW issues BEFORE the GR — reads stale vgpr
            _nd_gr(slot=1),
            make_swait(slot=2, vlcnt=0),
        ]))
        failures = self.compare(ref_cap, subj_cap, raise_on_unexplained=False)
        f = self.assert_failures_contain(failures, cls=OrderInvertedFailure)
        assert f.producer.category == "GRA"
        assert f.consumer.category == "LWA"


# -----------------------------------------------------------------------------
# Cross-mode parity
# -----------------------------------------------------------------------------


class TestCrossModeParity(GraphNativeValidationTest):
    """Lockstep parity between DTL and non-DTL modes.

    For the analogous logical violation — "previous-iter LR0 has no covering
    dscnt drain before the current-iter GR consumes the LDS slot" — both
    DTL and non-DTL fixtures must emit a Failure of identical SHAPE
    (same class, same counter_kind, same producer/consumer category strings).

    This guards against a future divergence where one path's edge-formation
    logic changes and silently drops a failure on its branch — the lockstep
    assertion fails immediately with both fixtures' failure lists side-by-
    side.
    """

    def _build_pair(self, gr_factory):
        """Build (REF, SUBJ) FourPartCaptures for the LRA0(prev) misplacement
        violation, parameterised by the GR factory (DTL vs non-DTL).
        """
        ref = self.wrap_single_body(
            make_capture(BODY_LABEL_ML, [gr_factory(slot=0)]),
            ml_prev=_prev_lra0_drained(),
        )
        subj = self.wrap_single_body(
            make_capture(BODY_LABEL_ML, [gr_factory(slot=0)]),
            ml_prev=_prev_lra0_misplaced(),
        )
        return ref, subj

    def test_dtl_and_nondtl_same_violation_same_failure_shape(self):
        # DTL fixture
        dtl_ref, dtl_subj = self._build_pair(_dtl_gr)
        dtl_failures = self.compare(dtl_ref, dtl_subj)
        dtl_f = self.assert_failures_contain(
            dtl_failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )

        # Non-DTL fixture
        nd_ref, nd_subj = self._build_pair(_nd_gr)
        nd_failures = self.compare(nd_ref, nd_subj)
        nd_f = self.assert_failures_contain(
            nd_failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )

        # Lockstep parity: same Failure class, same counter, same producer
        # category, same consumer category. The producer/consumer NODE
        # identities differ (different rocisa instances) — that's expected;
        # what we pin is the diagnostic SHAPE, not the per-instance fields.
        assert type(dtl_f) is type(nd_f), (
            f"Failure class divergence: DTL emits {type(dtl_f).__name__}, "
            f"non-DTL emits {type(nd_f).__name__}"
        )
        assert dtl_f.counter_kind == nd_f.counter_kind == "dscnt"
        assert dtl_f.producer.category == nd_f.producer.category == "LRA0"
        assert dtl_f.consumer.category == nd_f.consumer.category == "GRA"
