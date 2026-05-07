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
"""SBarrier edge collectors in build_dataflow_graph.

Tests the strict ordering invariant: producer -> SWait -> SBarrier -> consumer.
Edges form ONLY when the order is exactly right (and the SWait drains the
correct counter).

Two patterns:
  LR0 -> SWait(dscnt=0) -> SBarrier -> GR    (lr_to_gr_lds_reuse, must_start_after)
  GR  -> SWait(vlcnt=0) -> SBarrier -> LR1   (gr_to_lr_lds_reuse, needed_by)
"""

import pytest

from Tensile.Components.ScheduleCapture import (
    FourPartCapture,
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
)
from Tensile.Components.CMSValidator import (
    build_dataflow_graph,
    _DEFAULT_CDNA4_ARCH_PROFILE,
)

from dataflow_fixtures import (
    make_lr, make_gr, make_mfma, make_swait, make_sbarrier, make_capture,
)


# =============================================================================
# Helpers
# =============================================================================
# Same _wrap pattern as test_dataflow_graph_builder.py — fillers use the
# high register range so they don't form unrelated edges.


def _wrap(ml_capture, *, ml_prev=None, ngl=None, nll=None):
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


def _has_edge_kind(graph, kind):
    return any(e.edge_kind == kind for e in graph.edges)


def _edges_of_kind(graph, kind):
    return [e for e in graph.edges if e.edge_kind == kind]


# =============================================================================
# Positive — barrier edges form when ordering is correct
# =============================================================================


class TestBarrierEdgeFormation:
    def test_lr_swait_sbarrier_gr_forms_must_start_after_edge(self):
        """LR0 -> SWait(dscnt=0) -> SBarrier -> GR (LDS-reuse-write).
        GR overwrites the LDS slot LR0 read; barrier ensures all waves
        finished reading."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_sbarrier(slot=2),
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=3, category="GRA"),
        ])
        g = build_dataflow_graph(_wrap(cap))
        edges = _edges_of_kind(g, "lr_to_gr_lds_reuse")
        assert len(edges) == 1
        assert edges[0].producer.category == "LRA0"
        assert edges[0].consumer.category == "GRA"

    def test_gr_swait_sbarrier_lr1_forms_needed_by_edge(self):
        """GR -> SWait(vlcnt=0) -> SBarrier -> LR1 (LDS-reuse-read).
        LR1 reads the LDS slot GR wrote; barrier ensures all waves
        finished writing."""
        cap = make_capture(BODY_LABEL_ML, [
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=0, category="GRA"),
            make_swait(slot=1, vlcnt=0),
            make_sbarrier(slot=2),
            make_lr(8, 4, 64, slot=3, category="LRA1"),
        ])
        g = build_dataflow_graph(_wrap(cap))
        edges = _edges_of_kind(g, "gr_to_lr_lds_reuse")
        assert len(edges) == 1
        assert edges[0].producer.category == "GRA"
        assert edges[0].consumer.category == "LRA1"

    def test_multiple_lrs_then_swait_then_barrier_then_gr(self):
        """All LR0s before the SWait become barrier-covered for the GR."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_lr(12, 4, 80, slot=1, category="LRA0"),
            make_lr(16, 4, 96, slot=2, category="LRA0"),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=5, category="GRA"),
        ])
        g = build_dataflow_graph(_wrap(cap))
        edges = _edges_of_kind(g, "lr_to_gr_lds_reuse")
        # All 3 LR0s become barrier-covered for the single GR.
        assert len(edges) == 3


# =============================================================================
# Negative — barrier edge does NOT form when ordering is wrong
# =============================================================================


class TestNoBarrierEdgeWhenWrong:
    def test_no_swait_means_no_barrier_edge(self):
        """Barrier without SWait — in-wave dscnt counter never drained."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_sbarrier(slot=1),
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=2, category="GRA"),
        ])
        g = build_dataflow_graph(_wrap(cap))
        assert not _has_edge_kind(g, "lr_to_gr_lds_reuse")

    def test_no_sbarrier_means_no_barrier_edge(self):
        """SWait without SBarrier — cross-wave coherence not established."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, dscnt=0),
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=2, category="GRA"),
        ])
        g = build_dataflow_graph(_wrap(cap))
        assert not _has_edge_kind(g, "lr_to_gr_lds_reuse")

    def test_sbarrier_before_swait_means_no_barrier_edge(self):
        """Wrong order: barrier-then-wait. Other waves barriered while their
        data wasn't yet ready locally."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_sbarrier(slot=1),
            make_swait(slot=2, dscnt=0),
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=3, category="GRA"),
        ])
        g = build_dataflow_graph(_wrap(cap))
        assert not _has_edge_kind(g, "lr_to_gr_lds_reuse")

    def test_swait_after_gr_means_no_barrier_edge(self):
        """SWait fires after GR — by the time it fires, GR has already
        executed and the barrier window after the wait can't precede GR."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_sbarrier(slot=1),
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=2, category="GRA"),
            make_swait(slot=3, dscnt=0),
        ])
        g = build_dataflow_graph(_wrap(cap))
        assert not _has_edge_kind(g, "lr_to_gr_lds_reuse")

    def test_swait_on_wrong_counter_means_no_barrier_edge(self):
        """vlcnt instead of dscnt — wait doesn't drain the LR queue."""
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(8, 4, 64, slot=0, category="LRA0"),
            make_swait(slot=1, vlcnt=0),    # vlcnt not dscnt
            make_sbarrier(slot=2),
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=3, category="GRA"),
        ])
        g = build_dataflow_graph(_wrap(cap))
        assert not _has_edge_kind(g, "lr_to_gr_lds_reuse")


# =============================================================================
# Cross-iteration (DTL+LdsBuf): cross-body barrier edges
# =============================================================================


class TestCrossBodyBarrierEdges:
    def test_dtl_lds_buf_cross_iteration_barrier_edge(self):
        """ML-1 ends with [LR0, SWait(dscnt=0), SBarrier]; ML starts with
        a GR writing the LDS slot LR0 read. Cross-body barrier edge from
        ML-1.LR0 to ML.GR."""
        prev_cap = make_capture(BODY_LABEL_ML_PREV, [
            make_lr(8, 4, 64, slot=99, category="LRA0"),
            make_swait(slot=100, dscnt=0),
            make_sbarrier(slot=101),
        ])
        ml_cap = make_capture(BODY_LABEL_ML, [
            make_gr(40, 4, srd_sgpr_start=12, immediate_offset=64,
                    slot=0, category="GRA"),
        ])
        cap = _wrap(ml_cap, ml_prev=prev_cap)
        g = build_dataflow_graph(cap)
        cross_body = [e for e in _edges_of_kind(g, "lr_to_gr_lds_reuse")
                      if e.producer.body_label == BODY_LABEL_ML_PREV
                      and e.consumer.body_label == BODY_LABEL_ML]
        assert len(cross_body) == 1
