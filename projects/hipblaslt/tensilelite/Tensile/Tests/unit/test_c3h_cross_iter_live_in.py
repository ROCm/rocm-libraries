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
"""C3h (si5f): cross-iter and cross-body live-in unit tests.

Tests added by rocm-libraries-si5f (C3h).

TestCrossIterLiveIn
    Focuses on the COMPARISON CONTRACT for a graph that contains a genuine
    cross-iter (ML_PREV → ML) edge:

      (a) compare_graphs(g, g) == [] — cross-iter edges cancel in set-diff
          when the graph is compared against itself.
      (b) No OrderInvertedFailure is emitted in the above call — i.e. the
          cross-iter edge is not incorrectly classified as an ordering
          violation when ref and subj are the same graph.

    Note: edge-field-level assertions (producer_iter_index, consumer_iter_index,
    body labels) are already pinned in
    test_dataflow_graph_builder.py::test_cross_iter_edge_carries_diagnostic_annotations.
    This class covers only the comparison-contract angle.

TestCrossBodyLiveIn
    Analogous test for a PRO → ML cross-body edge built with _wrap_with_pro.
    Asserts that the edge exists (producer_body_label == BODY_LABEL_PROLOGUE)
    and that compare_graphs(g, g) == [] for a graph containing that edge.
"""

import pytest

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    BODY_LABEL_PROLOGUE,
    FourPartCapture,
)
from Tensile.Components.CMSValidator import (
    OrderInvertedFailure,
    build_dataflow_graph,
    compare_graphs,
    _DEFAULT_CDNA4_ARCH_PROFILE,
)
from dataflow_fixtures import (
    make_capture,
    make_lr,
    make_mfma,
    make_swait,
)


# =============================================================================
# Helpers
# =============================================================================


def _wrap_ml_prev_and_ml(ml_prev_cap, ml_cap):
    """Build a FourPartCapture with explicit ML_PREV + ML bodies and filler
    bodies in NGL / NLL.  The filler MFMA register ranges (200+) are chosen
    so they don't form edges against the payload registers used by the tests.
    """
    return FourPartCapture(
        main_loop_prev={0: ml_prev_cap},
        main_loop={0: ml_cap},
        n_gl={0: make_capture(BODY_LABEL_NGL, [
            make_mfma(220, 224, 228, slot=0, a_src_count=1)])},
        n_ll={0: make_capture(BODY_LABEL_NLL, [
            make_mfma(240, 244, 248, slot=0, a_src_count=1)])},
        num_mfma=1, num_codepaths=1, source="cms",
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
    )


def _wrap_with_pro(prologue_cap, ml_cap):
    """Build a FourPartCapture with a non-empty prologue + ML body and
    filler MFMAs in ML-1 / NGL / NLL so build_dataflow_graph accepts
    every required body.  Matches the same helper in test_dataflow_graph_hdem.py.
    """
    return FourPartCapture(
        prologue=prologue_cap,
        main_loop_prev={0: make_capture(BODY_LABEL_ML_PREV, [
            make_mfma(0, 200, 232, slot=0, a_src_count=1)])},
        main_loop={0: ml_cap},
        n_gl={0: make_capture(BODY_LABEL_NGL, [
            make_mfma(0, 208, 240, slot=0, a_src_count=1)])},
        n_ll={0: make_capture(BODY_LABEL_NLL, [
            make_mfma(0, 216, 248, slot=0, a_src_count=1)])},
        num_mfma=1, num_codepaths=1, source="cms",
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
    )


def _build_cross_iter_graph():
    """Build a DataflowGraph that contains exactly one cross-iter edge:
    ML_PREV's LR writes v[8:12), ML's MFMA reads v[8:12).

    The vgpr range [8..12) is disjoint from the filler ranges used by the
    NGL/NLL bodies (200+).
    """
    ml_prev_cap = make_capture(BODY_LABEL_ML_PREV, [
        make_lr(8, 4, 64, slot=0, category="LRA0"),
    ])
    ml_cap = make_capture(BODY_LABEL_ML, [
        make_swait(slot=0, dscnt=0),
        make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                  slot=1, a_src_count=4),
    ])
    fpc = _wrap_ml_prev_and_ml(ml_prev_cap, ml_cap)
    return build_dataflow_graph(fpc)


def _build_cross_body_graph():
    """Build a DataflowGraph that contains a PRO → ML cross-body edge:
    PRO's LR writes v[16:20), ML's MFMA reads v[16:20).

    The vgpr range [16..20) is disjoint from the filler ranges (200+).
    """
    pro_cap = make_capture(BODY_LABEL_PROLOGUE, [
        make_lr(16, 4, 128, slot=0, category="LRA0"),
    ])
    ml_cap = make_capture(BODY_LABEL_ML, [
        make_swait(slot=0, dscnt=0),
        make_mfma(c_dst_start=4, a_src_start=16, b_src_start=40,
                  slot=1, a_src_count=4),
    ])
    fpc = _wrap_with_pro(pro_cap, ml_cap)
    return build_dataflow_graph(fpc)


# =============================================================================
# TestCrossIterLiveIn
# =============================================================================


class TestCrossIterLiveIn:
    """Cross-iter (ML_PREV → ML) live-in: comparison contract.

    The graph built here has a genuine cross-iter edge (LR in ML_PREV writes
    the same vgpr bytes that MFMA in ML reads).  The tests pin the COMPARISON
    CONTRACT, not the edge-field annotations (those are already pinned in
    test_dataflow_graph_builder.py::test_cross_iter_edge_carries_diagnostic_annotations).
    """

    def test_cross_iter_edge_present(self):
        """Sanity: the graph actually contains a cross-iter edge so that the
        comparison-contract tests below are non-trivial."""
        g = _build_cross_iter_graph()
        cross_iter = [
            e for e in g.edges
            if e.edge_kind == "raw_intrawave"
            and e.producer.body_label == BODY_LABEL_ML_PREV
            and e.consumer.body_label == BODY_LABEL_ML
        ]
        assert len(cross_iter) == 1, (
            f"Expected 1 cross-iter ML_PREV→ML edge; got {len(cross_iter)}"
        )

    def test_compare_graphs_self_returns_empty(self):
        """compare_graphs(g, g) must return [] for a graph that contains a
        cross-iter edge.

        When ref == subj (same object), every edge_key in ref is also in subj
        and vice versa — the set-diff is empty in both directions.  Cross-iter
        edges do NOT escape set-diff cancellation.  This is a property of set
        arithmetic, not of hardware or schedule details.
        """
        g = _build_cross_iter_graph()
        failures = compare_graphs(g, g)
        assert failures == [], (
            f"compare_graphs(g, g) must be [] for a graph with cross-iter edges; "
            f"got {failures!r}"
        )

    def test_no_order_inverted_failure_on_self_compare(self):
        """No OrderInvertedFailure is emitted when a cross-iter graph is compared
        against itself.

        OrderInvertedFailure would indicate the comparator is treating the
        cross-iter edge as a reorder violation.  When ref == subj the missing_keys
        set is empty — diagnose_missing_edge is never called — so no
        OrderInvertedFailure can be emitted.
        """
        g = _build_cross_iter_graph()
        failures = compare_graphs(g, g)
        order_violations = [f for f in failures if isinstance(f, OrderInvertedFailure)]
        assert order_violations == [], (
            f"No OrderInvertedFailure expected on self-compare; "
            f"got {order_violations!r}"
        )


# =============================================================================
# TestCrossBodyLiveIn
# =============================================================================


class TestCrossBodyLiveIn:
    """Cross-body (PRO → ML) live-in: edge presence + comparison contract.

    The graph built here has a cross-body edge: PRO's LR writes the vgpr bytes
    that ML's MFMA reads.  PRO appears first in the unrolled stream, so
    latest_writer resolves the LR as the producer for those byte-keys.
    """

    def test_cross_body_pro_to_ml_edge_present(self):
        """A PRO → ML live-in edge is formed when a PRO LR writes vgpr bytes
        that an ML MFMA reads."""
        g = _build_cross_body_graph()
        pro_to_ml = [
            e for e in g.edges
            if e.producer.body_label == BODY_LABEL_PROLOGUE
            and e.consumer.body_label == BODY_LABEL_ML
        ]
        assert len(pro_to_ml) >= 1, (
            f"Expected at least 1 PRO→ML cross-body edge; "
            f"got {len(pro_to_ml)}: {pro_to_ml!r}"
        )

    def test_cross_body_producer_iter_index_is_zero(self):
        """PRO is a non-ML body; its iter_index must be 0 in the unrolled stream
        (non-ML bodies carry iter_index=0 per UnrolledIterRecord)."""
        g = _build_cross_body_graph()
        pro_to_ml = [
            e for e in g.edges
            if e.producer.body_label == BODY_LABEL_PROLOGUE
            and e.consumer.body_label == BODY_LABEL_ML
        ]
        assert len(pro_to_ml) >= 1
        for e in pro_to_ml:
            assert e.producer_iter_index == 0, (
                f"PRO node must carry iter_index=0; got {e.producer_iter_index}"
            )

    def test_cross_body_compare_graphs_self_returns_empty(self):
        """compare_graphs(g, g) returns [] even when g contains a PRO→ML
        cross-body edge."""
        g = _build_cross_body_graph()
        failures = compare_graphs(g, g)
        assert failures == [], (
            f"compare_graphs(g, g) must be [] for a graph with cross-body edges; "
            f"got {failures!r}"
        )
