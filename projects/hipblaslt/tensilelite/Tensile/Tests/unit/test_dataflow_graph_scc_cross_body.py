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
"""SCC body-boundary clearing in build_dataflow_graph
(rocm-libraries-theq).

SCC is a single-bit hardware status register that is NOT preserved
across loop iterations by any compiler convention. The per-byte
latest-writer resolver treats SCC like any 1-byte numeric register
(byte_key ``("scc", 0)``); without intervention, an SCC writer in body
N would silently source an SCC reader in body N+1 because
``latest_writer`` carries entries across the unified position-sorted
node stream.

The fix in `build_dataflow_graph` clears all SCC-typed entries from
``latest_writer`` whenever the iteration moves from one ``body_label``
to the next. The downstream cross-body suppression that used to live
in ``diagnose_missing_edge`` (CMSValidator.py:2584-2585, removed in
the same commit) was working in the wrong layer; this regression test
pins the correct, source-layer behavior.

See `Tensile/Components/SCC_CROSS_BODY_INVESTIGATION.md` (commit
cab6f57471) for the full investigation.
"""

from rocisa.container import sgpr
from rocisa.instruction import (
    SCmpEQU32,
    SCSelectB32,
    SAddU32,
    SAddCU32,
)

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML_PREV,
    BODY_LABEL_ML,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    FourPartCapture,
    SLOT_KIND_MFMA,
    SlotKey,
    TaggedInstruction,
    WrappedInstruction,
)
from Tensile.Components.CMSValidator import (
    _DEFAULT_CDNA4_ARCH_PROFILE,
    build_dataflow_graph,
)

from dataflow_fixtures import make_capture, make_mfma


# =============================================================================
# Helpers
# =============================================================================


def _tag(inst, *, slot_idx: int, sequence: int, category: str = "GRIncA"
         ) -> TaggedInstruction:
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot_idx, sequence=sequence),
    )


def _filler_capture(label: str, c: int, a: int, b: int):
    """A no-op MFMA so a body is non-empty enough for build_dataflow_graph."""
    return make_capture(label, [
        make_mfma(c_dst_start=c, a_src_start=a, b_src_start=b, slot=0),
    ])


def _scc_edges(graph):
    return [e for e in graph.edges
            if getattr(e.resource, "regType", None) == "scc"]


# =============================================================================
# Body-boundary SCC clearing
# =============================================================================


class TestSCCBodyBoundaryClear:
    """build_dataflow_graph must NOT emit SCC edges that cross body
    boundaries (ML-1 -> ML, ML -> NGL, NGL -> NLL). The per-byte
    resolver clears SCC entries from `latest_writer` at every body-
    label transition in the position-sorted node stream.

    Within a single body, SCC dataflow is unaffected — same-body SCC
    edges still form (covered in test_dataflow_graph_scc.py)."""

    def _build_two_body_capture(self, writer_label, reader_label):
        """Construct a FourPartCapture with an SCC writer in
        ``writer_label`` and an SCC reader in ``reader_label``. The
        other two bodies hold no-op MFMA fillers so the
        build_dataflow_graph precondition is met.
        """
        scc_writer = SCmpEQU32(sgpr(50, 1), sgpr(51, 1))
        scc_reader = SCSelectB32(dst=sgpr(100, 1),
                                 src0=sgpr(50, 1),
                                 src1=sgpr(51, 1))

        body_caps = {}
        body_caps[writer_label] = make_capture(writer_label, [
            _tag(scc_writer, slot_idx=0, sequence=0),
        ])
        body_caps[reader_label] = make_capture(reader_label, [
            _tag(scc_reader, slot_idx=0, sequence=0),
        ])

        # Filler ranges chosen disjoint from the writer/reader sgpr
        # operands so they cannot accidentally form the very edge under
        # test.
        filler_ranges = {
            BODY_LABEL_ML_PREV: (200, 204, 208),
            BODY_LABEL_ML:      (260, 264, 268),
            BODY_LABEL_NGL:     (220, 224, 228),
            BODY_LABEL_NLL:     (240, 244, 248),
        }
        for label, (c, a, b) in filler_ranges.items():
            if label not in body_caps:
                body_caps[label] = _filler_capture(label, c, a, b)

        return FourPartCapture(
            main_loop={0: body_caps[BODY_LABEL_ML]},
            main_loop_prev={0: body_caps[BODY_LABEL_ML_PREV]},
            n_gl={0: body_caps[BODY_LABEL_NGL]},
            n_ll={0: body_caps[BODY_LABEL_NLL]},
            num_mfma=1, num_codepaths=1, source="cms",
            arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
        )

    def _assert_no_cross_body_scc_edge(self, graph, writer_label, reader_label):
        cross = [
            e for e in _scc_edges(graph)
            if e.producer.body_label == writer_label
            and e.consumer.body_label == reader_label
        ]
        assert cross == [], (
            f"Expected NO cross-body SCC edge from {writer_label} to "
            f"{reader_label}; build_dataflow_graph must clear SCC "
            f"latest_writer entries at body boundaries. Got: "
            f"{[(e.producer.body_label, e.consumer.body_label, type(e.producer.rocisa_inst).__name__, type(e.consumer.rocisa_inst).__name__) for e in cross]}"
        )

    def test_no_scc_edge_ml_prev_to_ml(self):
        """ML-1 SCC writer must not source an ML SCC reader."""
        cap = self._build_two_body_capture(BODY_LABEL_ML_PREV, BODY_LABEL_ML)
        graph = build_dataflow_graph(cap)
        self._assert_no_cross_body_scc_edge(graph, BODY_LABEL_ML_PREV,
                                            BODY_LABEL_ML)

    def test_no_scc_edge_ml_to_ngl(self):
        """ML SCC writer must not source an NGL SCC reader."""
        cap = self._build_two_body_capture(BODY_LABEL_ML, BODY_LABEL_NGL)
        graph = build_dataflow_graph(cap)
        self._assert_no_cross_body_scc_edge(graph, BODY_LABEL_ML,
                                            BODY_LABEL_NGL)

    def test_no_scc_edge_ngl_to_nll(self):
        """NGL SCC writer must not source an NLL SCC reader."""
        cap = self._build_two_body_capture(BODY_LABEL_NGL, BODY_LABEL_NLL)
        graph = build_dataflow_graph(cap)
        self._assert_no_cross_body_scc_edge(graph, BODY_LABEL_NGL,
                                            BODY_LABEL_NLL)

    def test_no_scc_edge_skips_intermediate_body(self):
        """ML-1 SCC writer must not source an NGL SCC reader even when
        the intermediate ML body has no SCC traffic. The boundary
        clear must fire on every transition, not just the
        immediately-preceding-body case."""
        cap = self._build_two_body_capture(BODY_LABEL_ML_PREV, BODY_LABEL_NGL)
        graph = build_dataflow_graph(cap)
        self._assert_no_cross_body_scc_edge(graph, BODY_LABEL_ML_PREV,
                                            BODY_LABEL_NGL)

    def test_same_body_scc_edge_still_forms(self):
        """Sanity: clearing SCC at body boundaries must NOT affect
        intra-body SCC dataflow. SAddU32 -> SAddCU32 in ML still
        forms its carry-chain SCC edge."""
        producer = SAddU32(dst=sgpr(10, 1), src0=sgpr(10, 1),
                           src1=sgpr(100, 1))
        consumer = SAddCU32(dst=sgpr(11, 1), src0=sgpr(11, 1),
                            src1=sgpr(101, 1))

        ml_cap = make_capture(BODY_LABEL_ML, [
            _tag(producer, slot_idx=0, sequence=0),
            _tag(consumer, slot_idx=0, sequence=1),
        ])
        cap = FourPartCapture(
            main_loop={0: ml_cap},
            main_loop_prev={0: _filler_capture(BODY_LABEL_ML_PREV, 200, 204, 208)},
            n_gl={0: _filler_capture(BODY_LABEL_NGL, 220, 224, 228)},
            n_ll={0: _filler_capture(BODY_LABEL_NLL, 240, 244, 248)},
            num_mfma=1, num_codepaths=1, source="cms",
            arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
        )
        graph = build_dataflow_graph(cap)

        same_body = [
            e for e in _scc_edges(graph)
            if e.producer.body_label == BODY_LABEL_ML
            and e.consumer.body_label == BODY_LABEL_ML
        ]
        assert same_body, (
            "Expected same-body SAddU32 -> SAddCU32 SCC carry edge in "
            f"ML; got SCC edges: {[(e.producer.body_label, e.consumer.body_label) for e in _scc_edges(graph)]}"
        )
