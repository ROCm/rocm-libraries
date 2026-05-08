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
"""Edge formation + wait coverage in build_dataflow_graph.

Two-phase model:

1. EDGE FORMATION resolves resources in stream-position order. For each
   consumer's read, prior writers (those with SchedulePosition < consumer's)
   whose written resource overlaps the read are yielded as producers.
   Whether an SWaitCnt sits between producer and consumer in the captured
   stream does NOT affect edge formation — that's the wait-coverage pass.

2. WAIT COVERAGE is a separate validation pass, exposed as
   validate_edge_wait_coverage(graph). For each edge, it walks the
   captured stream and checks whether an SWaitCnt of the right counter
   drains the producer's queue slot before the consumer reads. It
   emits typed Failures (MissingWait, WaitOnWrongCounter, WaitInsufficient,
   MissingBarrier) — same Failure types the cross-graph
   diagnose_missing_edge classifier emits.

Tests below organized accordingly: TestBasicDataflow asserts on edge
SHAPE (which producer for which consumer's read); TestWaitCoverage
asserts on the lint surfaced by validate_edge_wait_coverage.
"""

import pytest

from Tensile.Components.ScheduleCapture import (
    FourPartCapture,
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    CaptureUnknownInstructionError,
    CaptureEmptyBodyError,
)
from Tensile.Components.CMSValidator import (
    DataflowGraph,
    GraphNode,
    MissingWaitFailure,
    WaitInsufficientFailure,
    OrderInvertedFailure,
    build_dataflow_graph,
    validate_edge_wait_coverage,
    _DEFAULT_CDNA4_ARCH_PROFILE,
)

from dataflow_fixtures import (
    make_lr, make_lw, make_gr, make_mfma, make_swait, make_sbarrier,
    make_capture,
)


# =============================================================================
# Helpers
# =============================================================================


def _wrap(ml_capture, *, ml_prev=None, ngl=None, nll=None):
    """Wrap a single body capture as a FourPartCapture for build_dataflow_graph.

    Bodies that are None get a degenerate single-MFMA capture using register
    indices in the high range (200+) so the filler doesn't accidentally form
    edges against producers in `ml_capture`. The test under examination must
    not place producers/consumers in the high-register range.
    """
    # Filler MFMA per body uses a distinct high-vgpr range so the
    # cross-body MFMA acc-chain edges (added when MFMARule started
    # writing acc) don't pollute the test's own edge set.
    _FILLER_RANGES = {
        BODY_LABEL_ML_PREV: (200, 204, 208),
        BODY_LABEL_NGL:     (220, 224, 228),
        BODY_LABEL_NLL:     (240, 244, 248),
    }

    def _filler(label):
        c, a, b = _FILLER_RANGES[label]
        return make_capture(label, [make_mfma(
            c_dst_start=c, a_src_start=a, b_src_start=b, slot=0,
        )])
    return FourPartCapture(
        main_loop={0: ml_capture},
        main_loop_prev={0: ml_prev if ml_prev is not None else _filler(BODY_LABEL_ML_PREV)},
        n_gl={0: ngl if ngl is not None else _filler(BODY_LABEL_NGL)},
        n_ll={0: nll if nll is not None else _filler(BODY_LABEL_NLL)},
        num_mfma=1, num_codepaths=1, source="cms",
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
    )


def _edges_to_pairs(graph):
    """Return list of (producer.category, consumer.category, register tuple)."""
    return [
        (e.producer.category, e.consumer.category,
         (e.resource.regType, e.resource.regIdx, e.resource.regNum))
        for e in graph.edges
    ]


def _has_edge(graph, p_cat, c_cat, reg_start=None):
    for e in graph.edges:
        if e.producer.category != p_cat or e.consumer.category != c_cat:
            continue
        if reg_start is None or e.resource.regIdx == reg_start:
            return True
    return False


# =============================================================================
# Positive — basic dataflow
# =============================================================================


class TestBasicDataflow:
    def test_lr_then_swait_then_consumer_forms_edge(self):
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32, slot=2),
        ])
        g = build_dataflow_graph(_wrap(cap))
        edges = [e for e in g.edges if e.edge_kind == "raw_intrawave"]
        assert len(edges) == 1
        assert edges[0].producer.category == "LRA0"
        assert edges[0].consumer.category == "MFMA"
        assert edges[0].resource.regIdx == 8

    def test_two_lrs_one_consumer_resolves_each_read(self):
        """MFMA reads v[8:11] (LR_a) AND v[12:15] (LR_b) — register-name
        resolution emits one edge per read."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_lr(12, 4, 80, slot=1, category="LRA0"),
            make_swait(slot=2, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=3, a_src_count=8),
        ])
        g = build_dataflow_graph(_wrap(cap))
        edges = [e for e in g.edges if e.edge_kind == "raw_intrawave"]
        producers = sorted(e.resource.regIdx for e in edges)
        assert producers == [8, 12]

    def test_each_consumer_resolves_to_its_register_writer(self):
        """Four LRs each writing different sub-ranges; two MFMAs each
        reading a specific sub-range — register resolution emits the
        edge to the WRITER of that range, regardless of FIFO state.
        Whether the SWait drained the producer is irrelevant to edge
        formation (that's wait-coverage, validated separately).
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),     # LR_a v[8:11]
            make_lr(12, 4, 80, slot=1, category="LRA0"),    # LR_b v[12:15]
            make_lr(16, 4, 96, slot=2, category="LRA0"),    # LR_c v[16:19]
            make_lr(20, 4, 112, slot=3, category="LRA0"),   # LR_d v[20:23]
            make_swait(slot=4, dscnt=2),                    # not relevant to edges
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=5, a_src_count=4),               # MFMA1 reads v[8:11]
            make_mfma(c_dst_start=4, a_src_start=20, b_src_start=32,
                      slot=6, a_src_count=4),               # MFMA2 reads v[20:23]
        ])
        g = build_dataflow_graph(_wrap(cap))
        # Both MFMAs get their producer edge by register resolution.
        producer_regs = {e.resource.regIdx for e in g.edges
                         if e.edge_kind == "raw_intrawave"}
        assert producer_regs == {8, 20}

    def test_dscnt_and_vlcnt_consumers_resolve_independently(self):
        """LR (writes v8) and GR (writes v40) both produce the same MFMA's
        reads — register resolution finds each writer independently."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=0,
                    slot=1, category="GRA"),
            make_swait(slot=2, dscnt=0, vlcnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=40,
                      slot=3),
        ])
        g = build_dataflow_graph(_wrap(cap))
        cats = sorted(e.producer.category for e in g.edges
                      if e.edge_kind == "raw_intrawave")
        assert "LRA0" in cats
        assert "GRA" in cats

    def test_three_lrs_consumer_resolves_to_register_writer(self):
        """Three LRs, MFMA reads v[12:15] which only LR_b writes — edge
        attributes to LR_b regardless of FIFO state across the SWaits."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),     # LR_a v[8:11]
            make_lr(12, 4, 80, slot=1, category="LRA0"),    # LR_b v[12:15]
            make_swait(slot=2, dscnt=1),
            make_lr(16, 4, 96, slot=3, category="LRA0"),    # LR_c v[16:19]
            make_swait(slot=4, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=12, b_src_start=32,
                      slot=5, a_src_count=4),               # reads v[12:15]
        ])
        g = build_dataflow_graph(_wrap(cap))
        # Edge LR_b -> MFMA exists by register resolution.
        assert _has_edge(g, "LRA0", "MFMA", reg_start=12)

    def test_two_consumers_share_producer_two_edges(self):
        """Two MFMAs both read v[8:11] (written by LR_a) — register
        resolution emits one edge per consumer to the same producer."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=1, a_src_count=4),
            make_swait(slot=2, dscnt=0),
            make_mfma(c_dst_start=4, a_src_start=8, b_src_start=32,
                      slot=3, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        edges = [e for e in g.edges if e.edge_kind == "raw_intrawave"
                 and e.resource.regIdx == 8]
        # Both MFMAs (at vmfma_index 1 and 3) get an edge from LR_a.
        consumer_slots = sorted(e.consumer.position.vmfma_index for e in edges)
        assert consumer_slots == [1, 3]
        # All edges attribute to the same producer (LR_a).
        assert all(e.producer.position.vmfma_index == 0 for e in edges)

    def test_no_producer_no_edge(self):
        """SWait alone with no producer in the graph: MFMA reads have no
        candidate writer, so no edges form. Register resolution requires
        a producer; SWait alone doesn't synthesize one."""
        cap = make_capture(BODY_LABEL_ML, [
            make_swait(slot=0, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32, slot=1),
        ])
        g = build_dataflow_graph(_wrap(cap))
        # No LR/LW/GR in the graph — MFMA reads can't resolve to a producer
        # within this body. (Cross-body resolution against the filler ML-1
        # in _wrap may form edges to the filler MFMA's writes, but those
        # aren't producers either.)
        assert all(e.edge_kind != "raw_intrawave" for e in g.edges)


# =============================================================================
# Wait coverage — validate_edge_wait_coverage classifier
# =============================================================================
# After register-name resolution forms the edge, a separate validator
# (validate_edge_wait_coverage) walks the captured stream and reports
# typed Failures for edges that aren't covered by an SWaitCnt that
# drains the producer.
#
# The pattern in every test below: edge IS formed (register resolution
# is unconditional) AND validate_edge_wait_coverage emits a specific
# Failure type describing what's missing in the schedule.


class TestWaitCoverage:
    def test_no_swait_emits_missing_wait_failure(self):
        """LR -> MFMA with NO SWait between them: edge forms; validator
        reports MissingWaitFailure (no SWait of any kind in the window
        that could cover the producer)."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=1, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        # Edge IS formed by register resolution.
        assert _has_edge(g, "LRA0", "MFMA", reg_start=8)
        # But validator flags the missing wait.
        failures = validate_edge_wait_coverage(g)
        assert any(isinstance(f, MissingWaitFailure)
                   and f.producer.category == "LRA0"
                   and f.consumer.category == "MFMA"
                   for f in failures)

    def test_swait_after_consumer_emits_missing_wait_failure(self):
        """SWait sits after the MFMA — too late; not in (producer, consumer)
        window. Edge forms (register resolution doesn't care). Validator
        reports MissingWaitFailure."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=1, a_src_count=4),
            make_swait(slot=2, dscnt=0),
        ])
        g = build_dataflow_graph(_wrap(cap))
        assert _has_edge(g, "LRA0", "MFMA", reg_start=8)
        failures = validate_edge_wait_coverage(g)
        assert any(isinstance(f, MissingWaitFailure) for f in failures)

    def test_swait_insufficient_count_emits_wait_insufficient_failure(self):
        """3 LRs in queue + SWait(dscnt=2) leaves the YOUNGEST (LR_c) NOT
        drained. Edge LR_c -> MFMA forms by register resolution. Validator
        reports WaitInsufficientFailure for that edge — the SWait covers
        the window but its cap value is too lax to drain LR_c's slot."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_lr(12, 4, 80, slot=1, category="LRA0"),
            make_lr(16, 4, 96, slot=2, category="LRA0"),
            make_swait(slot=3, dscnt=2),
            make_mfma(c_dst_start=0, a_src_start=16, b_src_start=32,
                      slot=4, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        assert _has_edge(g, "LRA0", "MFMA", reg_start=16)
        failures = validate_edge_wait_coverage(g)
        assert any(isinstance(f, WaitInsufficientFailure)
                   and f.producer.position.vmfma_index == 2  # LR_c
                   and f.consumer.position.vmfma_index == 4
                   for f in failures)

    def test_swait_on_wrong_counter_emits_missing_wait_with_nearby_hint(self):
        """SWait drains vlcnt; LR needs dscnt. Edge forms by register
        resolution. Validator reports MissingWaitFailure with
        nearby_other_counter_waits populated (the wrong-counter SWait at
        slot=1 is surfaced so the user can extend it). The former
        WaitOnWrongCounterFailure was collapsed into MissingWaitFailure
        — see bead `hof`."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, vlcnt=0, dscnt=-1),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        assert _has_edge(g, "LRA0", "MFMA", reg_start=8)
        failures = validate_edge_wait_coverage(g)
        miss = [f for f in failures if isinstance(f, MissingWaitFailure)]
        assert miss, f"Expected MissingWaitFailure, got: {[type(f).__name__ for f in failures]}"
        assert miss[0].counter_kind == "dscnt"
        assert len(miss[0].nearby_wait_indices) >= 1

    def test_swait_dscnt_minus_one_does_not_drain_lr_producer(self):
        """SWaitCnt with dscnt=-1 means 'no constraint' (counter ignored) —
        it does NOT drain any LDS-load producer. Pins the semantic that
        `_swait_drains` returns None for v < 0, so the LRA0 in the queue
        is NOT considered guaranteed at the consumer. validate_edge_wait_coverage
        must emit MissingWaitFailure on the LRA0 -> MFMA edge.

        Regression guard: if a future refactor flips -1 to mean 'drain
        everything', this test fails and the misread is caught. See
        WaitInsufficientFailure._format_canonical for the matching
        comment ('Counter value of -1 means "no constraint" ... -1 does
        NOT satisfy')."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=-1, vlcnt=-1, vscnt=-1),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        # Edge IS formed by register resolution.
        assert _has_edge(g, "LRA0", "MFMA", reg_start=8)
        # The dscnt=-1 SWait does NOT drain the LR producer, so the
        # validator must report MissingWaitFailure (no SWait in the
        # window constrains dscnt for this edge).
        failures = validate_edge_wait_coverage(g)
        assert any(isinstance(f, MissingWaitFailure)
                   and f.producer.category == "LRA0"
                   and f.consumer.category == "MFMA"
                   for f in failures), \
            (f"Expected MissingWaitFailure for LRA0->MFMA edge — dscnt=-1 "
             f"must not drain the producer. Got: "
             f"{[type(f).__name__ for f in failures]}")

    def test_swait_vlcnt_minus_one_does_not_drain_gr_producer(self):
        """Symmetric to the dscnt case: vlcnt=-1 means 'no constraint',
        so a GR (vector-load) producer is NOT drained. Validator must
        emit MissingWaitFailure on the GRA -> MFMA edge."""
        cap = make_capture(BODY_LABEL_ML, [
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=0,
                    slot=0, category="GRA"),
            make_swait(slot=1, dscnt=-1, vlcnt=-1, vscnt=-1),
            make_mfma(c_dst_start=0, a_src_start=40, b_src_start=32,
                      slot=2, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        # Edge IS formed by register resolution.
        assert _has_edge(g, "GRA", "MFMA", reg_start=40)
        failures = validate_edge_wait_coverage(g)
        assert any(isinstance(f, MissingWaitFailure)
                   and f.producer.category == "GRA"
                   and f.consumer.category == "MFMA"
                   for f in failures), \
            (f"Expected MissingWaitFailure for GRA->MFMA edge — vlcnt=-1 "
             f"must not drain the producer. Got: "
             f"{[type(f).__name__ for f in failures]}")

    def test_consumer_without_producer_no_edge(self):
        """SWait + MFMA with no producer in the body: register resolution
        finds no writer for MFMA's reads, so no edge forms in this body."""
        cap = make_capture(BODY_LABEL_ML, [
            make_swait(slot=0, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=1, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        # No producer for v[8:11] in this body — no edge formed by
        # resolution (the filler MFMAs in ML-1/NGL/NLL don't write v[8:11]).
        assert all(e.edge_kind != "raw_intrawave"
                   or e.consumer.body_label != BODY_LABEL_ML
                   for e in g.edges)


# =============================================================================
# Cross-body queue persistence
# =============================================================================


class TestCrossBodyQueueState:
    def test_cross_body_queue_persists_positive(self):
        """Hardware preserves SWaitCnt state across labels — the queue from
        ML-1 carries into ML, so a wait at the start of ML can drain a
        producer left pending at the end of ML-1."""
        prev_cap = make_capture(BODY_LABEL_ML_PREV, [
            make_lr(8, 4, 64, slot=99, category="LRA0"),
        ])
        ml_cap = make_capture(BODY_LABEL_ML, [
            make_swait(slot=0, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=1, a_src_count=4),
        ])
        cap = _wrap(ml_cap, ml_prev=prev_cap)
        g = build_dataflow_graph(cap)
        # Edge from ML-1.LRA0 to ML.MFMA on register v8.
        cross_body = [e for e in g.edges
                      if e.edge_kind == "raw_intrawave"
                      and e.producer.body_label == BODY_LABEL_ML_PREV
                      and e.consumer.body_label == BODY_LABEL_ML]
        assert len(cross_body) == 1
        assert cross_body[0].resource.regIdx == 8

    def test_cross_body_no_wait_emits_missing_wait_failure(self):
        """Producer in ML-1 with consumer in ML and NO SWait at the start
        of ML: the cross-body edge STILL forms (register resolution finds
        the producer regardless of where waits sit), but
        validate_edge_wait_coverage flags it as MissingWaitFailure since
        no SWait drains the producer in the (producer, consumer) window."""
        prev_cap = make_capture(BODY_LABEL_ML_PREV, [
            make_lr(8, 4, 64, slot=99, category="LRA0"),
        ])
        ml_cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=0, a_src_count=4),
        ])
        cap = _wrap(ml_cap, ml_prev=prev_cap)
        g = build_dataflow_graph(cap)
        # Cross-body edge IS formed by register resolution.
        cross_body = [e for e in g.edges
                      if e.edge_kind == "raw_intrawave"
                      and e.producer.body_label == BODY_LABEL_ML_PREV
                      and e.consumer.body_label == BODY_LABEL_ML]
        assert len(cross_body) == 1
        # But validator flags the missing wait.
        failures = validate_edge_wait_coverage(g)
        assert any(isinstance(f, MissingWaitFailure)
                   and f.producer.body_label == BODY_LABEL_ML_PREV
                   and f.consumer.body_label == BODY_LABEL_ML
                   for f in failures)


# =============================================================================
# Sanity / structural
# =============================================================================


class TestStructuralProperties:
    def test_unified_4_body_graph_holds_all_bodies(self):
        cap = FourPartCapture(
            main_loop={0: make_capture(BODY_LABEL_ML, [
                make_mfma(0, 8, 32, slot=0)])},
            main_loop_prev={0: make_capture(BODY_LABEL_ML_PREV, [
                make_mfma(0, 8, 32, slot=0)])},
            n_gl={0: make_capture(BODY_LABEL_NGL, [
                make_mfma(0, 8, 32, slot=0)])},
            n_ll={0: make_capture(BODY_LABEL_NLL, [
                make_mfma(0, 8, 32, slot=0)])},
            num_mfma=1, num_codepaths=1, source="cms",
            arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
        )
        g = build_dataflow_graph(cap)
        assert set(g.captures.keys()) == {
            BODY_LABEL_ML_PREV, BODY_LABEL_ML, BODY_LABEL_NGL, BODY_LABEL_NLL
        }

    def test_identity_stable_across_reorderings(self):
        """The same producer at different stream positions gets the same
        identity. Identity is reorder-stable; tests the interface, not the
        internal tuple shape."""
        cap_a = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=5, category="LRA0"),
            make_swait(slot=6, dscnt=0),
            make_mfma(0, 8, 32, slot=7, a_src_count=4),
        ])
        cap_b = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=12, category="LRA0"),
            make_swait(slot=13, dscnt=0),
            make_mfma(0, 8, 32, slot=14, a_src_count=4),
        ])
        g_a = build_dataflow_graph(_wrap(cap_a))
        g_b = build_dataflow_graph(_wrap(cap_b))
        # Both graphs should have at least one node tagged 'LRA0'; their
        # identities should be equal (signature is content-based).
        ids_a = [n.identity for n in g_a.nodes.values() if n.category == "LRA0"]
        ids_b = [n.identity for n in g_b.nodes.values() if n.category == "LRA0"]
        assert len(ids_a) == 1
        assert len(ids_b) == 1
        assert ids_a[0] == ids_b[0]

    def test_unknown_instruction_raises_always(self):
        """An instruction whose category resolves to no recognized scheduler-
        role tag AND whose Python class isn't one of LR/LW/GR/MFMA/SWait/
        SBarrier MUST raise — the missed-instruction guard. There is no
        lenient mode: silently skipping unmodeled instructions hid
        capture-pipeline gaps.
        """
        from dataclasses import dataclass

        @dataclass
        class _UnknownInst:
            pass

        from Tensile.Components.ScheduleCapture import (
            TaggedInstruction, WrappedInstruction, SlotKey, SLOT_KIND_MFMA,
        )
        ti = TaggedInstruction(
            wrapped=WrappedInstruction(_UnknownInst()),
            category="WHATEVER",  # not a recognized prefix or exact match
            slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                         mfma_index=0, sequence=0),
        )
        cap = make_capture(BODY_LABEL_ML, [ti])
        with pytest.raises(CaptureUnknownInstructionError):
            build_dataflow_graph(_wrap(cap))

    def test_known_category_with_unmodeled_class_becomes_node(self):
        """An instruction with a recognized category (e.g. PackA0) but a
        Python class the FIFO classifier doesn't model (e.g. a stand-in
        for VPermB32) MUST become a graph node — node-only, no FIFO/edge
        action. This is the case that motivates 'every non-sync instruction
        is a node': pack-VPerms must show up so cross-graph topology
        comparison sees them.
        """
        from dataclasses import dataclass

        @dataclass
        class _PackInst:
            def __str__(self):
                return "v_perm_b32 v[0:0], v[1:1], v[2:2], s[3:3]"

        from Tensile.Components.ScheduleCapture import (
            TaggedInstruction, WrappedInstruction, SlotKey, SLOT_KIND_MFMA,
        )
        # Need at least one MFMA so the body satisfies non-empty constraints
        # downstream.
        ti_pack = TaggedInstruction(
            wrapped=WrappedInstruction(_PackInst()),
            category="PackA0",
            slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                         mfma_index=0, sequence=0),
        )
        cap = make_capture(BODY_LABEL_ML, [
            ti_pack,
            make_mfma(0, 8, 32, slot=1, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        pack_nodes = [n for n in g.nodes.values() if n.category == "PackA0"]
        assert len(pack_nodes) == 1
        assert pack_nodes[0].identity[0] == "PACK"

    def test_empty_body_raises(self):
        """A captured body with zero TaggedInstructions is a capture-pipeline
        bug — bodies always contain at least the MFMA loop."""
        cap = FourPartCapture(
            main_loop={0: make_capture(BODY_LABEL_ML, [])},
            main_loop_prev={0: make_capture(BODY_LABEL_ML_PREV, [
                make_mfma(0, 8, 32, slot=0)])},
            n_gl={0: make_capture(BODY_LABEL_NGL, [
                make_mfma(0, 8, 32, slot=0)])},
            n_ll={0: make_capture(BODY_LABEL_NLL, [
                make_mfma(0, 8, 32, slot=0)])},
            num_mfma=1, num_codepaths=1, source="cms",
            arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
        )
        with pytest.raises(CaptureEmptyBodyError):
            build_dataflow_graph(cap)


# =============================================================================
# Edge.resource precision (Part 2 of resolver wrinkle)
# =============================================================================
# `_resolve_producers` yields (producer, overlap_reg) where
# overlap_reg is the *intersection* of the consumer's read with the
# producer's write — not the producer's full write. Previously the full
# write was emitted, overstating consumption when the read was narrower.


class TestEdgeRegisterIntersection:
    def test_wide_write_narrow_read_yields_intersection(self):
        """LR writes v[8:11] (4 vgprs); MFMA reads v[8:9] (2 vgprs).
        edge.resource should be v[8:9], not v[8:11]."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),       # writes v[8:11]
            make_swait(slot=1, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, a_src_count=2),                  # reads v[8:9]
        ])
        g = build_dataflow_graph(_wrap(cap))
        edges = [e for e in g.edges if e.edge_kind == "raw_intrawave"
                 and e.producer.category == "LRA0"]
        assert len(edges) == 1
        e = edges[0]
        assert e.resource.regIdx == 8, "intersection starts at consumer's read base"
        assert e.resource.regNum == 2, (
            f"intersection width should be 2 (consumer's narrower read), "
            f"got {e.resource.regNum} (would be 4 under old wreg-as-edge.resource)"
        )

    def test_narrow_write_wide_read_yields_intersection(self):
        """LR writes v[10:13] (4 vgprs); MFMA reads v[8:15] (8 vgprs).
        Intersection is v[10:13] — the narrower of the two."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(10, 4, 64, slot=0, category="LRA0"),      # writes v[10:13]
            make_swait(slot=1, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, a_src_count=8),                  # reads v[8:15]
        ])
        g = build_dataflow_graph(_wrap(cap))
        edges = [e for e in g.edges if e.edge_kind == "raw_intrawave"
                 and e.producer.category == "LRA0"]
        assert len(edges) == 1
        assert edges[0].resource.regIdx == 10
        assert edges[0].resource.regNum == 4


# =============================================================================
# Multi-write producer (Part 1 of resolver wrinkle)
# =============================================================================
# `_resolve_producers` previously had a `break` after the first
# matching write per producer, so a producer with multiple writes could
# only emit one edge per consumer-read. After the fix, each overlapping
# write yields its own edge. Today no production extractor returns
# multiple writes from `_writes(inst)`, so we exercise the resolver
# directly with a synthesized producers_by_kind map.


class TestMultiWriteProducer:
    def test_multi_write_producer_yields_one_edge_per_overlapping_write(self):
        """Synthesize a producer node whose written_regs list has TWO
        non-overlapping entries that both fall within the consumer's
        wide read. The resolver should yield two (producer, overlap)
        pairs — one per matching write — instead of breaking after the
        first match."""
        from Tensile.Components.ScheduleCapture import (
            _resolve_producers, SchedulePosition,
        )
        from rocisa.container import vgpr

        # Build minimal nodes by hand. The resolver only reads node.position.
        producer = GraphNode(
            identity=("LR", 1, ("v", 8, 4), 64),
            position=SchedulePosition(loop_index=1, vmfma_index=0, sub_index=0),
            category="LRA0", rocisa_inst=None, tagged_inst=None,
            body_label=BODY_LABEL_ML, name="multi-write-LR",
        )
        consumer = GraphNode(
            identity=("MFMA", 1, ("v", 8, 8)),
            position=SchedulePosition(loop_index=1, vmfma_index=2, sub_index=0),
            category="MFMA", rocisa_inst=None, tagged_inst=None,
            body_label=BODY_LABEL_ML, name="wide-read-MFMA",
        )

        # Producer writes v[8:11] AND v[12:15]. (No real instruction yields
        # two separate writes today, but the resolver must handle this for
        # the future MFMA-acc-write / VAddCO-dst+VCC cases.)
        wreg_a = vgpr(8, 4)
        wreg_b = vgpr(12, 4)
        # Per-byte latest-writer map: v8..v11 from wreg_a (write-slot 0),
        # v12..v15 from wreg_b (write-slot 1). The third tuple element is
        # the producer's positional write-slot (rocm-libraries-wx9.3
        # phase 3, memo §6.1 step 1) — threaded through the resolver into
        # the cross-graph edge identity via `DataflowEdge.src_operand_slot`.
        latest_writer = {}
        for i in range(4):
            latest_writer[("v", 8 + i)] = (producer, wreg_a, 0)
            latest_writer[("v", 12 + i)] = (producer, wreg_b, 1)

        # Consumer reads v[8:15] — overlaps both writes.
        read_reg = vgpr(8, 8)
        results = list(_resolve_producers(
            read_reg, consumer, latest_writer,
        ))

        # Should yield TWO 4-tuples, both attributing to `producer` but with
        # different overlap registers (v[8:11] and v[12:15]) and different
        # write-slots (0 and 1). Per rocm-libraries-wx9.3 phase 3, the
        # resolver yields
        # `(producer, overlap, intra_operand_byte_offsets, src_operand_slot)`.
        assert len(results) == 2, (
            f"multi-write producer should yield one edge per overlapping "
            f"write; got {len(results)} 4-tuple(s)"
        )
        assert all(p is producer for p, _, _, _ in results)
        overlap_starts = sorted(reg.regIdx for _, reg, _, _ in results)
        assert overlap_starts == [8, 12]
        overlap_widths = sorted(reg.regNum for _, reg, _, _ in results)
        assert overlap_widths == [4, 4]
        # Intra-operand offsets within the v[8:15] consumer read:
        # bytes 0..3 satisfied by wreg_a (v[8:11]), bytes 4..7 by wreg_b.
        offsets_by_start = {reg.regIdx: offs for _, reg, offs, _ in results}
        assert offsets_by_start[8] == (0, 1, 2, 3)
        assert offsets_by_start[12] == (4, 5, 6, 7)
        # Producer's write-slot rides alongside each yielded edge: wreg_a
        # was published at write-slot 0, wreg_b at write-slot 1.
        slots_by_start = {reg.regIdx: slot for _, reg, _, slot in results}
        assert slots_by_start[8] == 0
        assert slots_by_start[12] == 1

    def test_multi_write_producer_only_one_overlap_yields_one_edge(self):
        """Producer writes v[8:11] AND v[100:103]. Consumer reads v[8:9].
        Only the first write overlaps; only one edge should yield. Confirms
        the dropped-break doesn't over-yield when only one write matches."""
        from Tensile.Components.ScheduleCapture import (
            _resolve_producers, SchedulePosition,
        )
        from rocisa.container import vgpr

        producer = GraphNode(
            identity=("LR", 1, ("v", 8, 4), 64),
            position=SchedulePosition(loop_index=1, vmfma_index=0, sub_index=0),
            category="LRA0", rocisa_inst=None, tagged_inst=None,
            body_label=BODY_LABEL_ML, name="multi-write-LR",
        )
        consumer = GraphNode(
            identity=("MFMA", 1, ("v", 8, 2)),
            position=SchedulePosition(loop_index=1, vmfma_index=2, sub_index=0),
            category="MFMA", rocisa_inst=None, tagged_inst=None,
            body_label=BODY_LABEL_ML, name="narrow-read-MFMA",
        )

        # Per-byte map: v8..v11 from wreg_a (write-slot 0); v100..v103 from
        # wreg_b (write-slot 1). Consumer reads v8..v9 — only v8/v9 are in
        # latest_writer for the read. The third tuple element is the
        # producer's positional write-slot (rocm-libraries-wx9.3 phase 3).
        wreg_a = vgpr(8, 4)
        wreg_b = vgpr(100, 4)
        latest_writer = {}
        for i in range(4):
            latest_writer[("v", 8 + i)] = (producer, wreg_a, 0)
            latest_writer[("v", 100 + i)] = (producer, wreg_b, 1)

        results = list(_resolve_producers(
            vgpr(8, 2), consumer, latest_writer,
        ))
        assert len(results) == 1
        assert results[0][1].regIdx == 8
        assert results[0][1].regNum == 2  # intersection of v[8:11] and v[8:9]
        # Intra-operand offsets within the consumer's 2-byte read: 0 and 1.
        assert results[0][2] == (0, 1)
        # Write-slot rides through the yield (wreg_a was published at
        # write-slot 0).
        assert results[0][3] == 0


# =============================================================================
# Register intersection through the production `_intersection` dispatcher
# =============================================================================
# The register-vs-register path is implemented inside `_intersection`, which
# wraps inputs via `Register.from_rocisa`, intersects via
# `Register.intersection`, and materializes the result back into a fresh
# `RegisterContainer` for downstream consumers. These tests exercise that
# end-to-end path on the rocisa-typed inputs the production resolver sees.


class TestRegIntersection:
    def test_numeric_full_overlap(self):
        from Tensile.Components.ScheduleCapture import _intersection
        from rocisa.container import vgpr
        r = _intersection(vgpr(8, 4), vgpr(8, 4))
        assert r is not None
        assert (r.regType, r.regIdx, r.regNum) == ("v", 8, 4)

    def test_numeric_partial_overlap(self):
        from Tensile.Components.ScheduleCapture import _intersection
        from rocisa.container import vgpr
        # v[8:11] intersected with v[10:13] -> v[10:11]
        r = _intersection(vgpr(8, 4), vgpr(10, 4))
        assert r is not None
        assert (r.regIdx, r.regNum) == (10, 2)

    def test_numeric_no_overlap_returns_none(self):
        from Tensile.Components.ScheduleCapture import _intersection
        from rocisa.container import vgpr
        assert _intersection(vgpr(8, 4), vgpr(20, 4)) is None

    def test_cross_type_returns_none(self):
        from Tensile.Components.ScheduleCapture import _intersection
        from rocisa.container import vgpr, sgpr
        assert _intersection(vgpr(8, 4), sgpr(8, 4)) is None

    def test_none_inputs_return_none(self):
        from Tensile.Components.ScheduleCapture import _intersection
        from rocisa.container import vgpr
        assert _intersection(None, vgpr(8, 4)) is None
        assert _intersection(vgpr(8, 4), None) is None

    def test_symbolic_full_overlap(self):
        from Tensile.Components.ScheduleCapture import _intersection
        from rocisa.container import vgpr
        # vgpr("ValuA", 4) intersected with itself -> same
        a = vgpr("ValuA", 4)
        b = vgpr("ValuA", 4)
        r = _intersection(a, b)
        assert r is not None
        assert r.regName.name == "ValuA"
        assert r.regNum == 4

    def test_symbolic_partial_overlap(self):
        from Tensile.Components.ScheduleCapture import _intersection
        from rocisa.container import vgpr
        # ValuA+0..3 intersected with ValuA+2..5 -> ValuA+2..3 (regNum=2)
        a = vgpr("ValuA", 4)             # offset 0, count 4
        b = vgpr("ValuA+2", 4)           # offset 2, count 4
        r = _intersection(a, b)
        assert r is not None
        assert r.regName.name == "ValuA"
        assert r.regName.getTotalOffsets() == 2
        assert r.regNum == 2

    def test_symbolic_different_names_no_overlap(self):
        from Tensile.Components.ScheduleCapture import _intersection
        from rocisa.container import vgpr
        assert _intersection(vgpr("ValuA", 4), vgpr("ValuB", 4)) is None

    def test_mixed_symbolic_numeric_returns_none(self):
        from Tensile.Components.ScheduleCapture import _intersection
        from rocisa.container import vgpr
        assert _intersection(vgpr("ValuA", 4), vgpr(8, 4)) is None

    def test_intersection_is_hashable_and_dedups(self):
        """Two intersections produced from equivalent inputs must compare
        equal so DataflowGraph.edge_keys()'s set dedup works."""
        from Tensile.Components.ScheduleCapture import _intersection
        from rocisa.container import vgpr
        r1 = _intersection(vgpr(8, 4), vgpr(10, 4))
        r2 = _intersection(vgpr(8, 4), vgpr(10, 4))
        assert r1 == r2
        assert hash(r1) == hash(r2)
        assert {r1, r2} == {r1}
