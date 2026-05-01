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
"""SCC sentinel + _SCCRule edge formation (bead mrj.1).

Sub-task 1 of the SCC migration epic (`br show rocm-libraries-mrj`):
verify that build_dataflow_graph emits SCC RAW edges between SCC
producers and SCC consumers AFTER the rule attaches the SCC sentinel
to per-opcode reads/writes.

NO failure-wiring assertions live here — that's bead mrj.2. These
tests only assert that the edge SHAPE is correct; the SCC clobber
diagnostic in `diagnose_missing_edge` is built on top of these edges
in the next sub-task.

Per-opcode flag-table coverage is also exercised here so any future
addition / removal in `_SCC_OPCODE_FLAGS` is caught by a unit test
before it hits integration.
"""

from rocisa.container import sgpr
from rocisa.instruction import (
    SAddU32, SAddCU32, SSubU32, SSubBU32,
    SCmpEQU32, SCmpLgU32, SCSelectB32,
)

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    SLOT_KIND_MFMA,
    SlotKey,
    SCCConflictFailure,
    TaggedInstruction,
    build_dataflow_graph,
    compare_graphs,
    _SCC_OPCODE_FLAGS,
    _get_scc_sentinel,
    _populate_wrapper,
    WrappedInstruction,
)

from dataflow_fixtures import make_capture


# =============================================================================
# Helpers
# =============================================================================


def _tag(inst, *, slot_idx, sequence, category="GRIncA"):
    """Wrap a rocisa instruction in a TaggedInstruction at the given slot."""
    return TaggedInstruction(
        inst=inst,
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot_idx, sequence=sequence),
    )


def _wrap_single_body(ml_capture):
    """Wrap a single ML body capture so build_dataflow_graph accepts it.

    Imports kept local so the test module's top-level imports stay narrow.
    """
    from Tensile.Components.ScheduleCapture import (
        FourPartCapture,
        BODY_LABEL_ML_PREV,
        BODY_LABEL_NGL,
        BODY_LABEL_NLL,
    )
    from dataflow_fixtures import make_mfma

    def _filler(label, c, a, b):
        return make_capture(label, [make_mfma(
            c_dst_start=c, a_src_start=a, b_src_start=b, slot=0,
        )])

    return FourPartCapture(
        main_loop={0: ml_capture},
        main_loop_prev={0: _filler(BODY_LABEL_ML_PREV, 200, 204, 208)},
        n_gl={0: _filler(BODY_LABEL_NGL, 220, 224, 228)},
        n_ll={0: _filler(BODY_LABEL_NLL, 240, 244, 248)},
        num_mfma=1, num_codepaths=1, source="cms",
    )


def _scc_edges(graph):
    """Return only edges whose resource is the SCC sentinel."""
    out = []
    for e in graph.edges:
        res = e.resource
        if getattr(res, "regType", None) == "scc":
            out.append(e)
    return out


# =============================================================================
# Sentinel identity
# =============================================================================


class TestSCCSentinel:
    def test_singleton_identity(self):
        """Repeated calls return the same object — required so
        _NUMERIC_REG_FACTORIES['scc'] yields a stable hashable container
        that the per-byte latest-writer map can dedup on."""
        a = _get_scc_sentinel()
        b = _get_scc_sentinel()
        assert a is b

    def test_sentinel_shape(self):
        s = _get_scc_sentinel()
        assert s.regType == "scc"
        assert s.regIdx == 0
        assert s.regNum == 1

    def test_sentinel_equals_and_hashes(self):
        """The container must be usable as a dict key — _byte_keys_for_resource
        keys it as ('scc', 0) and the resolver dedups by (writer, write_res)."""
        a = _get_scc_sentinel()
        b = _get_scc_sentinel()
        assert a == b
        assert hash(a) == hash(b)


# =============================================================================
# Per-opcode flag table — published reads/writes match the table
# =============================================================================


class TestSCCRuleExtract:
    """Build a wrapped instruction and verify _populate_wrapper attaches
    the SCC sentinel to reads/writes per the per-opcode flag table.

    These tests are the source-of-truth assertions for the table's
    semantics; if a flag flips, the test catches it. Sibling agents
    (wx9.8 VSwap, wx9.9 VCC) running in parallel against the same
    `_OPERAND_RULES` registry can re-run these to confirm SCC behavior
    is unchanged after their merges.
    """

    def _wrap_and_populate(self, inst):
        w = WrappedInstruction(inst)
        _populate_wrapper(w)
        return w

    def _has_scc(self, regs):
        return any(getattr(r, "regType", None) == "scc" for r in regs)

    def test_scmp_writes_scc_does_not_read(self):
        # SCmpEQU32 has dst=nullptr, srcs=[s50, s51]. _GenericALURule
        # would misclassify s50 as a write at params[0]; _SCCRule must
        # claim it first and treat both srcs as reads only.
        inst = SCmpEQU32(sgpr(50, 1), sgpr(51, 1))
        w = self._wrap_and_populate(inst)
        assert self._has_scc(w.writes), "SCmpEQU32 must publish SCC write"
        assert not self._has_scc(w.reads), "SCmpEQU32 must NOT read SCC"
        # No sgpr write — fixes the false-write quirk.
        sgpr_writes = [r for r in w.writes if getattr(r, "regType", None) == "s"]
        assert sgpr_writes == [], (
            f"SCmpEQU32 should not have sgpr writes (no dst), got {sgpr_writes}"
        )
        # Both srcs land as reads.
        sgpr_reads = sorted(r.regIdx for r in w.reads
                            if getattr(r, "regType", None) == "s")
        assert sgpr_reads == [50, 51]

    def test_scmplgu32_writes_scc(self):
        """SCmpLgU32 added to the table beyond the 5 the bead lists by
        name — covered to catch flag-table drift."""
        inst = SCmpLgU32(sgpr(60, 1), sgpr(61, 1))
        w = self._wrap_and_populate(inst)
        assert self._has_scc(w.writes)
        assert not self._has_scc(w.reads)

    def test_saddu32_writes_scc_does_not_read(self):
        inst = SAddU32(dst=sgpr(10, 1), src0=sgpr(10, 1), src1=sgpr(20, 1))
        w = self._wrap_and_populate(inst)
        assert self._has_scc(w.writes), "SAddU32 must publish SCC write"
        assert not self._has_scc(w.reads), "SAddU32 must NOT read SCC"
        # sgpr dst preserved.
        sgpr_writes = [r for r in w.writes if getattr(r, "regType", None) == "s"]
        assert len(sgpr_writes) == 1 and sgpr_writes[0].regIdx == 10

    def test_saddcu32_reads_and_writes_scc(self):
        inst = SAddCU32(dst=sgpr(11, 1), src0=sgpr(11, 1), src1=sgpr(21, 1))
        w = self._wrap_and_populate(inst)
        assert self._has_scc(w.reads), "SAddCU32 must read SCC (carry-in)"
        assert self._has_scc(w.writes), "SAddCU32 must write SCC (carry-out)"

    def test_ssubbu32_reads_and_writes_scc(self):
        inst = SSubBU32(dst=sgpr(12, 1), src0=sgpr(12, 1), src1=sgpr(22, 1))
        w = self._wrap_and_populate(inst)
        assert self._has_scc(w.reads)
        assert self._has_scc(w.writes)

    def test_scselectb32_reads_scc_only(self):
        inst = SCSelectB32(dst=sgpr(100, 1), src0=sgpr(50, 1), src1=sgpr(51, 1))
        w = self._wrap_and_populate(inst)
        assert self._has_scc(w.reads), "SCSelectB32 must read SCC"
        assert not self._has_scc(w.writes), "SCSelectB32 must NOT write SCC"
        # Normal sgpr operands preserved.
        sgpr_writes = [r for r in w.writes if getattr(r, "regType", None) == "s"]
        sgpr_reads = sorted(r.regIdx for r in w.reads
                            if getattr(r, "regType", None) == "s")
        assert len(sgpr_writes) == 1 and sgpr_writes[0].regIdx == 100
        assert sgpr_reads == [50, 51]

    def test_table_covers_all_five_canonical_opcodes(self):
        """The bead spec calls out a minimum 5-opcode coverage. Pin it."""
        for cls in ("SCmpEQU32", "SAddU32", "SSubU32",
                    "SAddCU32", "SSubBU32", "SCSelectB32"):
            assert cls in _SCC_OPCODE_FLAGS, (
                f"{cls} missing from _SCC_OPCODE_FLAGS"
            )


# =============================================================================
# End-to-end edge formation in build_dataflow_graph
# =============================================================================


class TestSCCEdgeFormation:
    """The mrj.1 acceptance test: an SCmpEQU32 -> SCSelectB32 sequence
    forms an SCC RAW edge in the built dataflow graph.

    This is the smallest possible chain — a producer-only opcode followed
    by a consumer-only opcode — so any failure pinpoints the rule wiring
    rather than confounding it with the carry-chain (SAdd+SAddC) which
    has both reads and writes on the consumer side.
    """

    def test_scmp_then_cselect_forms_scc_edge(self):
        """SCmpEQU32 writes SCC; SCSelectB32 reads SCC — one edge.

        Body shape: a single SCC producer at slot 0, single SCC consumer
        at slot 1, identical category tag so they live in the same
        captured stream. No MFMA/LR/LW between them — only the SCC
        sentinel resource is in play.
        """
        producer = SCmpEQU32(sgpr(50, 1), sgpr(51, 1))
        consumer = SCSelectB32(dst=sgpr(100, 1), src0=sgpr(50, 1), src1=sgpr(51, 1))

        cap = make_capture(BODY_LABEL_ML, [
            _tag(producer, slot_idx=0, sequence=0, category="GRIncA"),
            _tag(consumer, slot_idx=0, sequence=1, category="GRIncA"),
        ])
        graph = build_dataflow_graph(_wrap_single_body(cap))

        scc_edges = _scc_edges(graph)
        assert len(scc_edges) >= 1, (
            "Expected at least one SCC edge between SCmpEQU32 (writer) "
            f"and SCSelectB32 (reader); got edges={[ (e.producer.category, e.consumer.category, getattr(e.resource, 'regType', None)) for e in graph.edges ]}"
        )
        # Verify the producer/consumer identities on at least one SCC edge.
        matched = [
            e for e in scc_edges
            if type(e.producer.rocisa_inst).__name__ == "SCmpEQU32"
            and type(e.consumer.rocisa_inst).__name__ == "SCSelectB32"
        ]
        assert matched, (
            "SCC edge present but did not connect SCmpEQU32 -> SCSelectB32. "
            f"Got: {[(type(e.producer.rocisa_inst).__name__, type(e.consumer.rocisa_inst).__name__) for e in scc_edges]}"
        )
        # Edge resource must be the SCC sentinel.
        assert matched[0].resource.regType == "scc"
        assert matched[0].resource.regIdx == 0

    def test_saddu32_then_saddcu32_forms_scc_edge(self):
        """The canonical GRInc carry chain: SAddU32 (writes SCC) feeds
        SAddCU32 (reads SCC carry-in). One SCC edge between them, plus
        the existing sgpr edges from operand sharing — but we only
        assert on the SCC edge here.
        """
        producer = SAddU32(dst=sgpr(10, 1), src0=sgpr(10, 1), src1=sgpr(100, 1))
        consumer = SAddCU32(dst=sgpr(11, 1), src0=sgpr(11, 1), src1=sgpr(101, 1))

        cap = make_capture(BODY_LABEL_ML, [
            _tag(producer, slot_idx=0, sequence=0, category="GRIncA"),
            _tag(consumer, slot_idx=0, sequence=1, category="GRIncA"),
        ])
        graph = build_dataflow_graph(_wrap_single_body(cap))

        scc_edges = _scc_edges(graph)
        scc_chain = [
            e for e in scc_edges
            if type(e.producer.rocisa_inst).__name__ == "SAddU32"
            and type(e.consumer.rocisa_inst).__name__ == "SAddCU32"
        ]
        assert scc_chain, (
            "Expected an SCC edge SAddU32 -> SAddCU32 (carry chain); "
            f"got SCC edges: {[(type(e.producer.rocisa_inst).__name__, type(e.consumer.rocisa_inst).__name__) for e in scc_edges]}"
        )

    def test_intervening_clobber_breaks_direct_scc_edge(self):
        """Per-byte latest-writer model: if an unrelated SCC writer
        sits between producer and consumer, the consumer's SCC read
        resolves to the CLOSER (clobbering) writer, not the original
        producer.

        This is the property that mrj.2 will turn into the SCC clobber
        failure diagnostic. Here we only assert on the structural edge
        shape — no failure assertions.
        """
        original_producer = SCmpEQU32(sgpr(50, 1), sgpr(51, 1))
        # An unrelated SCC writer (different sgpr operands) intervenes.
        intervening_clobber = SAddU32(
            dst=sgpr(80, 1), src0=sgpr(80, 1), src1=sgpr(81, 1)
        )
        consumer = SCSelectB32(dst=sgpr(100, 1), src0=sgpr(50, 1), src1=sgpr(51, 1))

        cap = make_capture(BODY_LABEL_ML, [
            _tag(original_producer, slot_idx=0, sequence=0, category="GRIncA"),
            _tag(intervening_clobber, slot_idx=0, sequence=1, category="GRIncA"),
            _tag(consumer, slot_idx=0, sequence=2, category="GRIncA"),
        ])
        graph = build_dataflow_graph(_wrap_single_body(cap))

        scc_edges = _scc_edges(graph)
        # Edge consumer->producer pairs landing on the SCSelectB32 read.
        consumer_inputs = [
            type(e.producer.rocisa_inst).__name__
            for e in scc_edges
            if type(e.consumer.rocisa_inst).__name__ == "SCSelectB32"
        ]
        # The consumer's SCC read must resolve to the intervening clobber
        # (the closer writer), NOT the original producer. This is the
        # property that defines "an SCC clobber happened".
        assert "SAddU32" in consumer_inputs, (
            "SCSelectB32's SCC read should resolve to the intervening "
            f"SAddU32 (latest writer); got inputs: {consumer_inputs}"
        )
        assert "SCmpEQU32" not in consumer_inputs, (
            "SCmpEQU32's SCC write should have been clobbered by the "
            f"intervening SAddU32; got inputs: {consumer_inputs}"
        )


# =============================================================================
# Failure-shape wiring (bead mrj.2): compare_graphs(default, subj) classifies
# an SCC clobber as SCCConflictFailure with producer/consumer/intervening_writer.
# =============================================================================


class TestSCCClobberFailure:
    """Build matched default + subject graphs differing only in an
    intervening SCC writer; verify diagnose_missing_edge classifies the
    missing producer->consumer SCC edge as SCCConflictFailure carrying
    the intervening clobber as `intervening_writer`.

    The default graph has NO clobber: producer SCmpEQU32 writes SCC,
    consumer SCSelectB32 reads it. The subject graph inserts an
    intervening SAddU32 between them — the per-byte latest-writer
    resolver re-routes the consumer's SCC read to SAddU32, breaking the
    producer->consumer edge.
    """

    def _build_pair(self, with_clobber):
        """Return (default_graph, subject_graph) for a producer/consumer
        SCC chain, optionally with an intervening clobber in the subject.

        Both graphs share the same SCmpEQU32 (producer) and SCSelectB32
        (consumer) sgpr operands so their identity tuples match across
        graphs (compare_graphs only requires LR/LW/GR/MFMA identity
        coverage; SCC/ALU instructions are unconstrained, but matching
        identities are needed for the edge keys to align).
        """
        producer = SCmpEQU32(sgpr(50, 1), sgpr(51, 1))
        consumer = SCSelectB32(dst=sgpr(100, 1), src0=sgpr(50, 1), src1=sgpr(51, 1))

        default_cap = make_capture(BODY_LABEL_ML, [
            _tag(producer, slot_idx=0, sequence=0, category="GRIncA"),
            _tag(consumer, slot_idx=0, sequence=1, category="GRIncA"),
        ])
        default = build_dataflow_graph(_wrap_single_body(default_cap))

        if with_clobber:
            # Same producer + consumer instances; insert clobber between.
            producer2 = SCmpEQU32(sgpr(50, 1), sgpr(51, 1))
            consumer2 = SCSelectB32(dst=sgpr(100, 1), src0=sgpr(50, 1),
                                    src1=sgpr(51, 1))
            clobber = SAddU32(dst=sgpr(80, 1), src0=sgpr(80, 1),
                              src1=sgpr(81, 1))
            subj_cap = make_capture(BODY_LABEL_ML, [
                _tag(producer2, slot_idx=0, sequence=0, category="GRIncA"),
                _tag(clobber, slot_idx=0, sequence=1, category="GRIncA"),
                _tag(consumer2, slot_idx=0, sequence=2, category="GRIncA"),
            ])
        else:
            producer2 = SCmpEQU32(sgpr(50, 1), sgpr(51, 1))
            consumer2 = SCSelectB32(dst=sgpr(100, 1), src0=sgpr(50, 1),
                                    src1=sgpr(51, 1))
            subj_cap = make_capture(BODY_LABEL_ML, [
                _tag(producer2, slot_idx=0, sequence=0, category="GRIncA"),
                _tag(consumer2, slot_idx=0, sequence=1, category="GRIncA"),
            ])
        subj = build_dataflow_graph(_wrap_single_body(subj_cap))
        return default, subj

    def test_clobber_yields_scc_conflict_failure(self):
        """Negative path: with an intervening SAddU32 in subj, the
        SCC edge SCmpEQU32 -> SCSelectB32 is missing, and
        diagnose_missing_edge surfaces SCCConflictFailure with
        intervening_writer pointing at SAddU32."""
        default, subj = self._build_pair(with_clobber=True)
        failures = compare_graphs(default, subj, raise_on_unexplained=False)
        scc_failures = [f for f in failures if isinstance(f, SCCConflictFailure)]
        assert len(scc_failures) >= 1, (
            f"Expected at least one SCCConflictFailure; got "
            f"{[type(f).__name__ for f in failures]}"
        )
        # All graph-shape SCCConflictFailures must carry populated nodes.
        f = scc_failures[0]
        assert f.producer is not None
        assert f.consumer is not None
        assert f.intervening_writer is not None
        # Class identities of the wrapped rocisa instructions.
        assert type(f.producer.rocisa_inst).__name__ == "SCmpEQU32"
        assert type(f.consumer.rocisa_inst).__name__ == "SCSelectB32"
        assert type(f.intervening_writer.rocisa_inst).__name__ == "SAddU32"
        # Format must mention the intervening writer.
        msg = f.format(capture=None)
        assert "Intervening SCC writer" in msg
        assert "SAddU32" in msg

    def test_no_clobber_yields_no_scc_conflict_failure(self):
        """Positive path: without an intervening writer, the SCC edge is
        present in both graphs — compare_graphs must NOT emit any
        SCCConflictFailure."""
        default, subj = self._build_pair(with_clobber=False)
        failures = compare_graphs(default, subj, raise_on_unexplained=False)
        scc_failures = [f for f in failures if isinstance(f, SCCConflictFailure)]
        assert scc_failures == [], (
            f"Expected no SCCConflictFailure on the no-clobber path; got "
            f"{scc_failures}"
        )
