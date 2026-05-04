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

Production-side deletions deferred:
  * `add_gr_not_too_early_constraints` is still used by
    `test_GRMustStartAfterGRInc.py` for the GRInc -> GR ordering arc; that
    arc should migrate via the SCC sentinel resource (mrj epic, complete)
    in a follow-up bead. Until then we KEEP the legacy rule and the legacy
    `test_ValidateGlobalReadsNotTooEarly.py` to preserve coverage.
  * `set_gr_must_start_after_from_lr0s`, `apply_must_start_after_barriers`,
    `GRAfterLRRule`, and the `must_start_after` / `must_start_after_barriered_at`
    fields on `GlobalRead` will be deleted in the same follow-up bead, once
    `test_GRMustStartAfterGRInc.py` is migrated and the legacy test file
    can be removed.

This mirrors the pattern set by ola.4 (graph-native pack tests live in
`test_validate_pack_graph.py`; legacy `test_ValidatePack.py` retained).
"""

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    MissingBarrierFailure,
    MissingWaitFailure,
    WaitInsufficientFailure,
)

from dataflow_fixtures import (
    make_capture, make_gr, make_lr, make_sbarrier, make_swait,
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
