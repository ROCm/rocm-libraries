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
"""SWait queue semantics in build_dataflow_graph.

Each test exercises a single drain semantic in isolation. Graph-builder
correctness for raw_intrawave edges depends entirely on the FIFO drain
model (s_waitcnt CAPS outstanding ops at N — does NOT pop N entries).
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

    Bodies that are None get a single-MFMA capture so the builder doesn't
    barf on empty bodies. ml_prev defaults to the same single-MFMA filler.
    """
    def _filler():
        return make_capture(BODY_LABEL_ML, [make_mfma(0, 8, 32, slot=0)])
    return FourPartCapture(
        main_loop={0: ml_capture},
        main_loop_prev={0: ml_prev if ml_prev is not None else _filler()},
        n_gl={0: ngl if ngl is not None else _filler()},
        n_ll={0: nll if nll is not None else _filler()},
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

    def test_two_lrs_then_swait_zero_drains_both_lrs(self):
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_lr(12, 4, 80, slot=1, category="LRA0"),
            make_swait(slot=2, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=3, a_src_count=8),
        ])
        g = build_dataflow_graph(_wrap(cap))
        # MFMA reads v[8:11] (covered by LR_a) and v[12:15] (covered by LR_b).
        edges = [e for e in g.edges if e.edge_kind == "raw_intrawave"]
        producers = sorted(e.register.regIdx for e in edges)
        assert producers == [8, 12]

    def test_swait_dscnt_two_leaves_two_pending(self):
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),     # LR_a
            make_lr(12, 4, 80, slot=1, category="LRA0"),    # LR_b
            make_lr(16, 4, 96, slot=2, category="LRA0"),    # LR_c
            make_lr(20, 4, 112, slot=3, category="LRA0"),   # LR_d
            make_swait(slot=4, dscnt=2),
            # MFMA1 reads ONLY LR_a's regs — should form an edge.
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=5, a_src_count=4),
            # MFMA2 reads ONLY LR_d's regs — should NOT form an edge.
            make_mfma(c_dst_start=4, a_src_start=20, b_src_start=32,
                      slot=6, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        # After SWait(dscnt=2), the OLDEST 2 (LR_a, LR_b) drain — they're ready.
        # LR_c and LR_d remain pending.
        producer_regs = {e.register.regIdx for e in g.edges
                         if e.edge_kind == "raw_intrawave"}
        assert 8 in producer_regs    # LR_a -> MFMA1 formed
        assert 20 not in producer_regs  # LR_d still pending; no MFMA2 edge

    def test_dscnt_and_vlcnt_routed_to_separate_queues(self):
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=0,
                    slot=1, category="GRA"),
            make_swait(slot=2, dscnt=0, vlcnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=40,
                      slot=3),
        ])
        g = build_dataflow_graph(_wrap(cap))
        # Both producers should drain because both counters are at zero.
        cats = sorted(e.producer.category for e in g.edges
                      if e.edge_kind == "raw_intrawave")
        assert "LRA0" in cats
        assert "GRA" in cats

    def test_subsequent_swaits_drain_remaining_queue(self):
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),     # LR_a
            make_lr(12, 4, 80, slot=1, category="LRA0"),    # LR_b
            make_swait(slot=2, dscnt=1),                    # drains LR_a only
            make_lr(16, 4, 96, slot=3, category="LRA0"),    # LR_c
            make_swait(slot=4, dscnt=0),                    # drains LR_b and LR_c
            make_mfma(c_dst_start=0, a_src_start=12, b_src_start=32,
                      slot=5, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        # LR_b -> MFMA edge should exist via the second SWait.
        assert _has_edge(g, "LRA0", "MFMA", reg_start=12)

    def test_swait_between_two_consumers_only_late_consumer_gets_edge(self):
        """MFMA1 reads BEFORE the SWait — must NOT form an edge.
           MFMA2 reads AFTER the SWait — MUST form an edge.
        Catches the implementation defect 'drain on consumer' instead of 'drain on SWait'.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=1, a_src_count=4),  # MFMA1 — too early
            make_swait(slot=2, dscnt=0),
            make_mfma(c_dst_start=4, a_src_start=8, b_src_start=32,
                      slot=3, a_src_count=4),  # MFMA2 — covered by SWait
        ])
        g = build_dataflow_graph(_wrap(cap))
        edges = [e for e in g.edges if e.edge_kind == "raw_intrawave"]
        consumer_slots = sorted(e.consumer.position.vmfma_index for e in edges)
        # Only MFMA2 (at slot 3) should be in the edges, NOT MFMA1 (at slot 1).
        assert consumer_slots == [3]

    def test_empty_queue_swait_dscnt_zero_is_noop(self):
        cap = make_capture(BODY_LABEL_ML, [
            make_swait(slot=0, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32, slot=1),
        ])
        g = build_dataflow_graph(_wrap(cap))
        # SWait fires on empty queue — natural no-op; no edge formed.
        assert all(e.edge_kind != "raw_intrawave" for e in g.edges)


# =============================================================================
# Negative — no edge formed when guarantee absent
# =============================================================================


class TestNoEdgeWhenGuaranteeAbsent:
    def test_no_swait_means_no_edge(self):
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=1, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        assert all(e.edge_kind != "raw_intrawave" for e in g.edges)

    def test_swait_after_consumer_means_no_edge(self):
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=1, a_src_count=4),
            make_swait(slot=2, dscnt=0),
        ])
        g = build_dataflow_graph(_wrap(cap))
        assert all(e.edge_kind != "raw_intrawave" for e in g.edges)

    def test_swait_insufficient_count_means_no_edge_for_pending(self):
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),     # LR_a
            make_lr(12, 4, 80, slot=1, category="LRA0"),    # LR_b
            make_lr(16, 4, 96, slot=2, category="LRA0"),    # LR_c
            make_swait(slot=3, dscnt=2),
            # MFMA reads only LR_c's regs — but LR_c is still pending.
            make_mfma(c_dst_start=0, a_src_start=16, b_src_start=32,
                      slot=4, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        # LR_c (regIdx=16) must NOT have a raw_intrawave edge.
        assert not _has_edge(g, "LRA0", "MFMA", reg_start=16)

    def test_swait_on_wrong_counter_means_no_edge(self):
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            # Wait drains vlcnt; LR is on dscnt — useless for this LR.
            make_swait(slot=1, vlcnt=0, dscnt=-1),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=2, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        assert all(e.edge_kind != "raw_intrawave" for e in g.edges)

    def test_consumer_without_producer_no_edge(self):
        cap = make_capture(BODY_LABEL_ML, [
            make_swait(slot=0, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=1, a_src_count=4),
        ])
        g = build_dataflow_graph(_wrap(cap))
        assert all(e.edge_kind != "raw_intrawave" for e in g.edges)


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

    def test_cross_body_queue_persists_negative(self):
        """Without a wait at the start of ML, the producer in ML-1 stays
        pending and no cross-body edge forms."""
        prev_cap = make_capture(BODY_LABEL_ML_PREV, [
            make_lr(8, 4, 64, slot=99, category="LRA0"),
        ])
        ml_cap = make_capture(BODY_LABEL_ML, [
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=0, a_src_count=4),
        ])
        cap = _wrap(ml_cap, ml_prev=prev_cap)
        g = build_dataflow_graph(cap)
        cross_body = [e for e in g.edges
                      if e.edge_kind == "raw_intrawave"
                      and e.producer.body_label == BODY_LABEL_ML_PREV
                      and e.consumer.body_label == BODY_LABEL_ML]
        assert cross_body == []


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

    def test_unknown_instruction_class_raises(self):
        """An instruction whose rocisa class is none of LR/LW/GR/MFMA/SWait/
        SBarrier should cause build_dataflow_graph to raise."""
        from dataclasses import dataclass

        @dataclass
        class _UnknownInst:
            pass

        from Tensile.Components.ScheduleCapture import (
            TaggedInstruction, SlotKey, SLOT_KIND_MFMA,
        )
        ti = TaggedInstruction(
            inst=_UnknownInst(),
            category="WHATEVER",
            slot=SlotKey(iteration=0, slot_kind=SLOT_KIND_MFMA,
                         mfma_index=0, sequence=0),
        )
        cap = make_capture(BODY_LABEL_ML, [ti])
        with pytest.raises(CaptureUnknownInstructionError):
            build_dataflow_graph(_wrap(cap))

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
