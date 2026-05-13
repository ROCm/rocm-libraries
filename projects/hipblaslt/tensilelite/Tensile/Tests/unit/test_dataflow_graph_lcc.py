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
"""LCC (loop-counter code) graph-node coverage.

Per `LCC_AUDIT.md` the loop-counter code emitted by `closeLoop(...,
finalLoop=False)` and routed into the CMS macro under category "LCC"
consists of exactly two SALU instructions per codepath:

    SSubU32(dst=loopCounter, src0=loopCounter, src1=1)
    SCmpEQI32(src0=loopCounter, src1=hex(endCounter))

Each is 1 quad-cycle. These instructions are full participants in the
dataflow graph: nodes appear in `nodes_by_identity` with
`category == "LCC"` and `issue_cycles == 1`, and they contribute to
`cumulative_issue_cycles` walks.

The `LCC not in nll_categories` invariant in `test_ScheduleCapture.py`
is preserved because `n_ll` has `\\useLoop == 0` — the macro itself
suppresses LCC instructions there, regardless of any graph-level filter.
"""

import pytest

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    FourPartCapture,
)
from Tensile.Components.CMSValidator import (
    build_dataflow_graph,
    cumulative_issue_cycles,
    _DEFAULT_CDNA4_ARCH_PROFILE,
)

from dataflow_fixtures import (
    make_capture,
    make_lcc_pair,
    make_lr,
    make_mfma,
)


# =============================================================================
# Helpers
# =============================================================================


def _wrap(ml_capture):
    """Wrap a single main-loop capture into a FourPartCapture, filling the
    other three bodies with a no-op MFMA so build_dataflow_graph's
    non-empty-body precondition is satisfied. Mirrors the pattern in
    test_dataflow_graph_register_gaps.py."""
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
        main_loop_prev={0: _filler(BODY_LABEL_ML_PREV)},
        n_gl={0: _filler(BODY_LABEL_NGL)},
        n_ll={0: _filler(BODY_LABEL_NLL)},
        num_mfma=1, num_codepaths=1, source="cms",
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
    )


# =============================================================================
# Phase 1 coverage — LCC nodes appear in the graph
# =============================================================================


class TestLCCInGraph:
    def test_lcc_nodes_present_in_graph(self):
        """The two LCC instructions become full graph nodes with
        `category == "LCC"`, the right `body_label`, and
        `issue_cycles == 1` apiece. The LCC pair is placed at the same
        `mfma_index` as the leading MFMA but with sequence values 1/2 so
        stream ordering reflects "MFMA first, then LCC"."""
        sub_ti, cmp_ti = make_lcc_pair(slot=0, sequence_start=1)
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(0, 8, 32, slot=0, a_src_count=4, sequence=0),
            sub_ti,
            cmp_ti,
        ])
        graph = build_dataflow_graph(_wrap(cap))

        lcc_nodes = [n for n in graph.nodes.values()
                     if n.body_label == BODY_LABEL_ML and n.category == "LCC"]
        assert len(lcc_nodes) == 2, (
            f"Expected 2 LCC nodes in ML body, got {len(lcc_nodes)}: "
            f"{[type(n.rocisa_inst).__name__ for n in lcc_nodes]}"
        )
        cls_names = sorted(type(n.rocisa_inst).__name__ for n in lcc_nodes)
        assert cls_names == ["SCmpEQI32", "SSubU32"], cls_names
        for node in lcc_nodes:
            # Per EMISSION_ORDINAL_DESIGN.md §4.5 + ORAM1
            # (rocm-libraries-hdem) Approach A, the identity tuple is now
            # `(canonical_render, emission_ordinal)` (body-blind); LCC's
            # rocisa classes (SSubU32 / SCmpEQI32) lack a registered category
            # in `_CLASS_NAME_TO_CATEGORY`, so the rocisa-derived classifier
            # returns None for them. The CMS-shaped role string lives on
            # `node.category`; pin it there.
            assert node.category == "LCC", node.category
            assert node.body_label == BODY_LABEL_ML, node.body_label
            assert node.issue_cycles == 1, (
                f"LCC node {type(node.rocisa_inst).__name__} expected 1 "
                f"quad-cycle, got {node.issue_cycles}"
            )

    def test_lcc_nodes_appear_in_stream_order(self):
        """The per-body sidecar `_graph_nodes` carries LCC nodes in the
        order they were emitted (SSubU32 before SCmpEQI32)."""
        sub_ti, cmp_ti = make_lcc_pair(slot=0, sequence_start=1)
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(0, 8, 32, slot=0, a_src_count=4, sequence=0),
            sub_ti,
            cmp_ti,
        ])
        graph = build_dataflow_graph(_wrap(cap))

        ml_body = graph.captures[BODY_LABEL_ML]
        sidecar = ml_body._graph_nodes
        lcc_in_order = [n for n in sidecar if n.category == "LCC"]
        assert len(lcc_in_order) == 2
        assert type(lcc_in_order[0].rocisa_inst).__name__ == "SSubU32"
        assert type(lcc_in_order[1].rocisa_inst).__name__ == "SCmpEQI32"


# =============================================================================
# `cumulative_issue_cycles` accounting includes the LCC contribution
# =============================================================================


class TestLCCIssueCycleAccounting:
    """`cumulative_issue_cycles` walks the per-body instruction stream and
    sums per-instruction issue costs between producer and consumer.
    LCC's two 1-quad-cycle instructions sitting between an MFMA producer
    and a non-MFMA consumer must show up in the gap.

    Note: the gap returned is `c_issue - p_issue - 1`. Two extra
    1-quad-cycle instructions shift the consumer's issue start by +2,
    so the reported gap also grows by 2.

    Why a non-MFMA consumer: when both producer and consumer are MFMAs,
    `mfma_free_at = current_issue + 1 + finish_cycles` clamps the
    consumer's start, masking small per-instruction issue-cycle deltas.
    Using an LR consumer (no MFMA-blocking arithmetic on it) makes the
    LCC contribution observable directly.
    """

    def _build_capture(self, *, with_lcc: bool):
        # Producer MFMA at mfma_index=0; consumer LR at mfma_index=1.
        # When with_lcc=True, two LCC instructions sit between them at
        # mfma_index=0, sequence=1/2.
        instructions = [
            make_mfma(c_dst_start=100, a_src_start=8, b_src_start=32,
                      slot=0, a_src_count=4, sequence=0),
        ]
        if with_lcc:
            sub_ti, cmp_ti = make_lcc_pair(slot=0, sequence_start=1)
            instructions.extend([sub_ti, cmp_ti])
        instructions.append(
            make_lr(dst_vgpr_start=300, dst_vgpr_count=4,
                    lds_offset=0, slot=1, category="LRA0", sequence=0),
        )
        return make_capture(BODY_LABEL_ML, instructions)

    def test_cumulative_issue_cycles_includes_lcc_contribution(self):
        cap_with = self._build_capture(with_lcc=True)
        cap_without = self._build_capture(with_lcc=False)
        g_with = build_dataflow_graph(_wrap(cap_with))
        g_without = build_dataflow_graph(_wrap(cap_without))

        def _ml_node(graph, *, category):
            matches = [n for n in graph.nodes.values()
                       if n.body_label == BODY_LABEL_ML
                       and n.category == category]
            assert len(matches) == 1, (category, matches)
            return matches[0]

        prod_with = _ml_node(g_with, category="MFMA")
        cons_with = _ml_node(g_with, category="LRA0")
        prod_without = _ml_node(g_without, category="MFMA")
        cons_without = _ml_node(g_without, category="LRA0")

        gap_with = cumulative_issue_cycles(g_with, prod_with, cons_with)
        gap_without = cumulative_issue_cycles(
            g_without, prod_without, cons_without)

        # Each LCC instruction adds 1 quad-cycle; two of them -> +2.
        assert gap_with - gap_without == 2, (
            f"Adding the LCC pair should shift the producer-to-consumer "
            f"gap by exactly 2 quad-cycles. Got with={gap_with}, "
            f"without={gap_without}, delta={gap_with - gap_without}."
        )


# =============================================================================
# Regression guard — LCC stays out of NLL
# =============================================================================


class TestLCCExclusionGuards:
    """`Tests/unit/test_ScheduleCapture.py:748-754` already asserts
    `"LCC" not in nll_categories`. That invariant is enforced by the
    macro (`\\useLoop == 0` strips LCC out of n_ll bodies, per the audit
    Step 4) — NOT by the graph-level filter that was dropped in 2bu.2.
    The presence of this test in this file makes the dependency
    explicit: dropping the graph-level filter does not regress the
    body-level guarantee.
    """

    def test_n_ll_filler_has_no_lcc_nodes(self):
        sub_ti, cmp_ti = make_lcc_pair(slot=0, sequence_start=1)
        cap = make_capture(BODY_LABEL_ML, [
            make_mfma(0, 8, 32, slot=0, a_src_count=4, sequence=0),
            sub_ti,
            cmp_ti,
        ])
        graph = build_dataflow_graph(_wrap(cap))

        # The n_ll filler in `_wrap` is a single MFMA — no LCC instructions
        # were ever in that body, so no LCC node should appear there.
        n_ll_lcc = [n for n in graph.nodes.values()
                    if n.body_label == BODY_LABEL_NLL and n.category == "LCC"]
        assert n_ll_lcc == [], (
            f"n_ll body should never carry LCC nodes (macro guards them "
            f"with \\useLoop==0); got {n_ll_lcc}"
        )
