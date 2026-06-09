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
"""Minimal reproducer for the cross-subiter Pack -> MFMA dataflow-graph artifact.

Companion test for `Tensile/Components/CROSS_SUBITER_ALU_FP_INVESTIGATION.md`
(bead `rocm-libraries-bwfr`) and the visual memo
`Tensile/Components/CROSS_SUBITER_ALU_FP_MINIMAL_REPRO.md`.

The artifact: when two Pack instructions in different subiters write the
same physical scratch vgpr (the "v133 pattern" — see
`ScheduleCapture.py:1550`), the per-byte `latest_writer` resolver in
`build_dataflow_graph` overwrites the writer entry on each Pack. In the
default schedule (all Packs emitted before all MFMAs) this means the
MFMA's read of the scratch vgpr resolves to the *later* Pack as its
producer — even though that Pack semantically belongs to a different
subiter. CMS pipelines the Pack-MFMA pairs so the resolver sees the
correct per-subiter writer.

This file pins three pieces of the artifact at the smallest possible
fixture scale (real `VCvtPkF32toBF16` Pack instances, two subiters, one
MFMA):

  1. The default-side graph contains the artifactual `PackA1 -> MFMA`
     edge (PackA1 was the last writer in stream order).
  2. The CMS-side graph contains the semantically correct
     `PackA0 -> MFMA` edge.
  3. Under the unrolled walk, `compare_graphs(ref, subj)` returns zero
     failures — not because of a carve-out (the cross-graph exemption in
     `diagnose_missing_edge` was deleted in C1 / rocm-libraries-5tf9),
     but because the byte-key edge keys differ by `emission_ordinal`
     (Option E / rocm-libraries-56e3) and `diagnose_missing_edge` Phase 2
     routes the ALU->MFMA synthetic-fixture edge through the unconditional
     passthrough fallback (`num_mfma_per_subiter=0` in this fixture).
     The artifact is real at the individual-graph level but harmless in
     the cross-graph comparison.

The carve-out neutralization probe (item 4 from the original list) was
deleted in C4 (rocm-libraries-5ryl): it monkeypatched `GraphNode.subiter`
to expose the missing edge as an `OrderInvertedFailure`, but the synthetic
fixture's `num_mfma_per_subiter=0` causes the `nmps=0` gate in
`_evaluate_gap_rule_condition` to short-circuit before `subiter()` is ever
called — making the probe doubly dead (also, the cross-graph exemption it
originally targeted no longer exists post-C1).
"""

from rocisa.container import vgpr
from rocisa.instruction import VCvtPkF32toBF16

from Tensile.Components.CMSValidator import (
    _DEFAULT_CDNA4_ARCH_PROFILE,
    build_dataflow_graph,
    compare_graphs,
)
from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    FourPartCapture,
    SLOT_KIND_MFMA,
    SlotKey,
    TaggedInstruction,
    WrappedInstruction,
)

from dataflow_fixtures import make_capture, make_mfma


# =============================================================================
# Fixture builders — 3 instructions: 2 Packs (different subiters) + 1 MFMA
# =============================================================================
# Both Packs write to the SAME physical vgpr (v133), the canonical
# scratch-reuse pattern from `ScheduleCapture.py:1550`. Each Pack reads
# distinct sources (v8/v9 vs v10/v11) so the rocisa render-strings differ
# and the captures yield two distinct identity tuples — render-string
# identity collisions would silently mask the bug.


_SCRATCH_VGPR = 133


def _tag_pack(inst, *, category, mfma_index, sequence):
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=mfma_index, sequence=sequence),
    )


def _make_pack(category, mfma_index, sequence, *, src0_idx, src1_idx):
    inst = VCvtPkF32toBF16(
        dst=vgpr(_SCRATCH_VGPR, 1),
        src0=vgpr(src0_idx, 1),
        src1=vgpr(src1_idx, 1),
    )
    return _tag_pack(inst, category=category,
                     mfma_index=mfma_index, sequence=sequence)


def _wrap(ml_capture):
    """Wrap a single ML capture in a FourPartCapture with filler bodies.

    Mirrors the standard `_wrap` helper from
    `test_dataflow_graph_comparison.py`. Filler MFMAs use vgpr ranges
    well above the scratch register so they don't alias.
    """
    def _filler(label, c, a, b):
        return make_capture(label, [
            make_mfma(c_dst_start=c, a_src_start=a, b_src_start=b, slot=0),
        ])
    return FourPartCapture(
        main_loop={0: ml_capture},
        main_loop_prev={0: _filler(BODY_LABEL_ML_PREV, 200, 204, 208)},
        n_gl={0: _filler(BODY_LABEL_NGL, 220, 224, 228)},
        n_ll={0: _filler(BODY_LABEL_NLL, 240, 244, 248)},
        num_mfma=1, num_codepaths=1, source="cms",
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
    )


def _build_default_capture():
    """Default schedule: PackA0, PackA1, MFMA — all Packs before MFMA.

    Matches what the default SIA scheduler emits within a body. The
    per-byte latest-writer for v133 ends up pointing at PackA1 by the
    time MFMA's read is resolved.
    """
    return make_capture(BODY_LABEL_ML, [
        _make_pack("PackA0", mfma_index=0, sequence=0, src0_idx=8, src1_idx=9),
        _make_pack("PackA1", mfma_index=0, sequence=1, src0_idx=10, src1_idx=11),
        make_mfma(c_dst_start=200, a_src_start=_SCRATCH_VGPR,
                  b_src_start=140, slot=1, a_src_count=1, b_src_count=1),
    ])


def _build_cms_capture():
    """CMS schedule: PackA0, MFMA, PackA1 — pipelined so the MFMA reads
    v133 between the two Pack writes. The resolver attributes MFMA's
    read of v133 to PackA0, the semantically correct producer.
    """
    return make_capture(BODY_LABEL_ML, [
        _make_pack("PackA0", mfma_index=0, sequence=0, src0_idx=8, src1_idx=9),
        make_mfma(c_dst_start=200, a_src_start=_SCRATCH_VGPR,
                  b_src_start=140, slot=1, a_src_count=1, b_src_count=1),
        _make_pack("PackA1", mfma_index=2, sequence=0, src0_idx=10, src1_idx=11),
    ])


def _v133_edges(graph):
    """Return the list of edges in `graph` whose resource is v133."""
    out = []
    for e in graph.edges:
        res = e.resource
        if (getattr(res, "regType", None) == "v"
                and getattr(res, "regIdx", None) == _SCRATCH_VGPR):
            out.append(e)
    return out


# =============================================================================
# Tests
# =============================================================================


class TestCrossSubiterPackArtifact:
    """Three positive assertions that demonstrate the artifact and why it is harmless
    in cross-graph comparison under the unrolled walk."""

    def test_artifact_present_in_default_graph(self):
        """Default schedule produces the artifactual `PackA1 -> MFMA` edge.

        PackA1 is a *later* writer in stream order than PackA0; the
        per-byte latest-writer resolver overwrites v133's entry on the
        PackA1 write, so MFMA's read of v133 sees PackA1 as its
        producer. This edge is an artifact of emission order interacting
        with a destructive last-writer-wins resolver — the kernel writer
        intended PackA0 to be the real producer for this MFMA's
        subiter-0 work.
        """
        g_default = build_dataflow_graph(_wrap(_build_default_capture()))
        v133_edges = _v133_edges(g_default)
        assert len(v133_edges) == 1, (
            f"Expected exactly one v133 edge in default graph, got "
            f"{len(v133_edges)}: {[e.producer.category for e in v133_edges]}"
        )
        producer = v133_edges[0].producer
        consumer = v133_edges[0].consumer
        assert producer.category == "PackA1", (
            f"Default graph must surface the artifactual edge with PackA1 "
            f"(the LAST stream-order writer of v133) as producer; got "
            f"{producer.category}."
        )
        assert consumer.category == "MFMA"
        # Producer is positionally before consumer in default's linear emission.
        # Under the unrolled walk, `position` maps to `unrolled_position`.
        # PackA1 is at unrolled_position=1; MFMA is at unrolled_position=2 → True.
        assert producer.position < consumer.position

    def test_correct_edge_present_in_cms_graph(self):
        """CMS pipelining produces the semantically correct `PackA0 -> MFMA`
        edge. The MFMA's read of v133 happens between PackA0's write and
        PackA1's write, so the resolver attributes the read to PackA0.
        """
        g_cms = build_dataflow_graph(_wrap(_build_cms_capture()))
        v133_edges = _v133_edges(g_cms)
        assert len(v133_edges) == 1
        producer = v133_edges[0].producer
        assert producer.category == "PackA0", (
            f"CMS graph must surface the semantically correct edge with "
            f"PackA0 (the subiter that logically owns this MFMA's input) "
            f"as producer; got {producer.category}."
        )

    def test_compare_graphs_returns_no_failures_for_cross_subiter_artifact(self):
        """End-to-end: `compare_graphs` reports zero failures even though the
        default-side graph carries the artifactual `PackA1 -> MFMA` edge
        that does not exist in the CMS-side graph.

        As of C3 (rocm-libraries-si5f), the reason is NOT a carve-out in
        `diagnose_missing_edge` (that code was deleted in C1 /
        rocm-libraries-5tf9).  The zero-failure result arises because
        `diagnose_missing_edge` Phase 2 dispatches the ALU->MFMA edge
        through the unconditional passthrough fallback: the synthetic
        fixture's `num_mfma_per_subiter=0` causes the `nmps=0` gate in
        `_evaluate_gap_rule_condition` to return False for all
        condition-gated rules, so the `unconditional_passthrough` rule
        fires and returns `_PASSTHROUGH` → `[]`.  No exemption is
        involved; the passthrough is the principled fallback for
        `nmps=0` synthetic fixtures.
        """
        g_default = build_dataflow_graph(_wrap(_build_default_capture()))
        g_cms = build_dataflow_graph(_wrap(_build_cms_capture()))

        failures = compare_graphs(g_default, g_cms)
        assert failures == [], (
            f"Expected zero failures for the cross-subiter artifact under the "
            f"unrolled walk (ALU->MFMA unconditional passthrough for nmps=0); "
            f"got {len(failures)} failure(s): "
            f"{[type(f).__name__ for f in failures]}"
        )
