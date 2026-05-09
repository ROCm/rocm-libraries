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
  3. With the section-7.3 carve-out engaged (production behavior),
     `compare_graphs` returns zero failures despite the edge mismatch,
     because the carve-out classifies cross-subiter ALU-producer
     order inversions as legitimate pipelining.
  4. With the carve-out neutralized (probe via monkey-patched
     `_node_subiter`), the missing edge surfaces as one
     `OrderInvertedFailure` — confirming the carve-out is the only
     mechanism suppressing it.

This is the synthetic equivalent of the 768-edge failure reported in
`test_ScheduleCapture.py::TestRealKernelCapture::test_tf32_4x4_tn_capture_shape`
when the carve-out is neutralized.
"""

from rocisa.container import vgpr
from rocisa.instruction import VCvtPkF32toBF16

from Tensile.Components import CMSValidator
from Tensile.Components.CMSValidator import (
    OrderInvertedFailure,
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
    """Three positive assertions that demonstrate the artifact + suppression."""

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
        # Producer is positionally AFTER consumer in stream order would be
        # the inversion case — here we instead have producer-before-consumer
        # in default's linear emission, which is exactly what the carve-out
        # site (CMSValidator.py:2584) gates on as the inversion direction.
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

    def test_carveout_suppresses_artifact_and_neutralization_surfaces_it(self):
        """End-to-end: with the section-7.3 carve-out engaged (production
        behavior), `compare_graphs` reports zero failures even though the
        default-side graph carries the artifactual `PackA1 -> MFMA` edge
        that does not exist in the CMS-side graph.

        Neutralizing the carve-out via a monkey-patch on `_node_subiter`
        (the same probe technique used in
        `Tests/scratch/run_with_carveout_off.py` per bwfr §3.2) flips the
        suppression off and surfaces the artifact as exactly one
        `OrderInvertedFailure`.
        """
        g_default = build_dataflow_graph(_wrap(_build_default_capture()))
        g_cms = build_dataflow_graph(_wrap(_build_cms_capture()))

        # Production behavior: carve-out absorbs the diff -> 0 failures.
        failures_with_carveout = compare_graphs(g_default, g_cms)
        assert failures_with_carveout == [], (
            f"Expected the section-7.3 carve-out to suppress the artifact; "
            f"got {len(failures_with_carveout)} failure(s): "
            f"{[type(f).__name__ for f in failures_with_carveout]}"
        )

        # Probe: neutralize the carve-out's subiter predicate by forcing
        # both producer and consumer to look like subiter 0. The
        # `_node_subiter(p) != _node_subiter(c)` check fails, the carve-out
        # branch is skipped, and the OrderInvertedFailure path fires.
        original_node_subiter = CMSValidator._node_subiter
        CMSValidator._node_subiter = lambda n, nmps: 0
        try:
            failures_neutralized = compare_graphs(g_default, g_cms)
        finally:
            CMSValidator._node_subiter = original_node_subiter

        assert len(failures_neutralized) == 1, (
            f"With the carve-out neutralized, the artifact must surface as "
            f"exactly one OrderInvertedFailure; got {len(failures_neutralized)}."
        )
        assert isinstance(failures_neutralized[0], OrderInvertedFailure)
