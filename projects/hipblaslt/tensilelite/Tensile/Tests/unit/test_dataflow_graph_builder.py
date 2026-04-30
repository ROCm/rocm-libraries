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

1. EDGE FORMATION is reorder-invariant: edges are derived purely from
   register-name resolution. For each consumer's read, the producer is
   the unique writer of that register, looked up by name (via
   _reg_overlaps), with logical_position-based ordering (body, iter,
   kind_rank, intra_seq) for tiebreaking. Whether an SWaitCnt sits
   between producer and consumer in the captured stream does NOT
   affect edge formation. Two captures of the same instruction set
   in different schedules produce identical edges by construction.

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
    DataflowGraph,
    GraphNode,
    build_dataflow_graph,
    validate_edge_wait_coverage,
    MissingWaitFailure,
    WaitOnWrongCounterFailure,
    WaitInsufficientFailure,
    OrderInvertedFailure,
    CaptureUnknownInstructionError,
    CaptureEmptyBodyError,
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
    def _filler(label):
        # Filler MFMA reads from v[200:203] — high enough that real test
        # producers won't overlap.
        return make_capture(label, [make_mfma(
            c_dst_start=200, a_src_start=204, b_src_start=208, slot=0,
        )])
    return FourPartCapture(
        main_loop={0: ml_capture},
        main_loop_prev={0: ml_prev if ml_prev is not None else _filler(BODY_LABEL_ML_PREV)},
        n_gl={0: ngl if ngl is not None else _filler(BODY_LABEL_NGL)},
        n_ll={0: nll if nll is not None else _filler(BODY_LABEL_NLL)},
        num_mfma=1, num_codepaths=1, source="cms",
    )


def _edges_to_pairs(graph):
    """Return list of (producer.category, consumer.category, register tuple)."""
    return [
        (e.producer.category, e.consumer.category,
         (e.register.regType, e.register.regIdx, e.register.regNum))
        for e in graph.edges
    ]


def _has_edge(graph, p_cat, c_cat, reg_start=None):
    for e in graph.edges:
        if e.producer.category != p_cat or e.consumer.category != c_cat:
            continue
        if reg_start is None or e.register.regIdx == reg_start:
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
        assert edges[0].register.regIdx == 8

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
        producers = sorted(e.register.regIdx for e in edges)
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
        producer_regs = {e.register.regIdx for e in g.edges
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
                 and e.register.regIdx == 8]
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

    def test_swait_on_wrong_counter_emits_wait_on_wrong_counter_failure(self):
        """SWait drains vlcnt; LR needs dscnt. Edge forms by register
        resolution. Validator reports WaitOnWrongCounterFailure."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, vlcnt=0, dscnt=-1),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        assert _has_edge(g, "LRA0", "MFMA", reg_start=8)
        failures = validate_edge_wait_coverage(g)
        assert any(isinstance(f, WaitOnWrongCounterFailure) for f in failures)

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
        assert cross_body[0].register.regIdx == 8

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
            TaggedInstruction, SlotKey, SLOT_KIND_MFMA,
        )
        ti = TaggedInstruction(
            inst=_UnknownInst(),
            category="WHATEVER",  # not a recognized prefix or exact match
            slot=SlotKey(iteration=0, slot_kind=SLOT_KIND_MFMA,
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
            TaggedInstruction, SlotKey, SLOT_KIND_MFMA,
        )
        # Need at least one MFMA so the body satisfies non-empty constraints
        # downstream.
        ti_pack = TaggedInstruction(
            inst=_PackInst(),
            category="PackA0",
            slot=SlotKey(iteration=0, slot_kind=SLOT_KIND_MFMA,
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
        )
        with pytest.raises(CaptureEmptyBodyError):
            build_dataflow_graph(cap)
