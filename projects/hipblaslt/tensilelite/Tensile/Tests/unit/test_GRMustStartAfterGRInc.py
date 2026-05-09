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
"""Graph-native ports of the GRInc -> GR ordering tests.

The legacy structural rule `add_gr_not_too_early_constraints` encoded two
distinct invariants in a single pass:

  1. LR0 -> SWait(dscnt=0) -> SBarrier -> GR    (lr_to_gr_lds_reuse, cross-wave
     LDS sync). Migrated to graph-native in
     `test_validate_gr_not_too_early_graph.py`.

  2. last GRInc<X> -> first GR<X>                (intra-wave SRD ordering;
     GRInc writes the SRD sgprs that GR reads). This file covers the
     graph-native port: the dataflow graph forms a real RAW edge from the
     GRInc's SAddU32 (which writes the SRD sgpr) to the GR's BufferLoad
     (which reads it), and `compare_graphs` flips a reversed order into
     `OrderInvertedFailure`.

Note on the SCC sentinel: the SCC machinery detects an UNRELATED SCC
writer landing between an SCC producer/consumer pair (clobber). That is
a DIFFERENT invariant from GRInc->GR ordering — the GRInc SAddU32 writes
both the SRD sgpr AND SCC; the GR's BufferLoad reads the SRD sgpr but
NOT SCC. The ordering arc this file pins is the SRD RAW edge (not an
SCC edge).

Mutation-smell-test (acceptance criterion): commenting out the SCC clobber
branch in `diagnose_missing_edge` (ScheduleCapture.py) does NOT break
tests in this file (correct — the GRInc->GR arc is sgpr-RAW, not SCC).
But commenting out the `_GenericALURule` path (which publishes the
SAddU32 sgpr write) DOES break this file's negative tests, because it's
that rule that lets the dataflow graph see the GRInc -> GR sgpr RAW edge
in the first place.
"""

from rocisa.container import sgpr
from rocisa.instruction import SAddU32

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    SLOT_KIND_MFMA,
    SlotKey,
    TaggedInstruction,
    WrappedInstruction,
)
from Tensile.Components.CMSValidator import (
    MissingBarrierFailure,
    OrderInvertedFailure,
)

from dataflow_fixtures import make_capture, make_gr, make_lr, make_sbarrier, make_swait
from graph_native_validation_base import GraphNativeValidationTest


# =============================================================================
# Helpers
# =============================================================================
# SRD sgpr ranges: GRA reads SRD at s12..s15, GRB reads SRD at s20..s23.
# A GRInc<X> is modeled as an SAddU32 that writes the FIRST sgpr of the
# SRD range (the standard incLower update pattern from incrementSrd in
# KernelWriterAssembly.py — the carry-out via SAddCU32 to s13 isn't needed
# here because the per-byte latest-writer resolver only requires the dst
# overlap to form the RAW edge to the GR's read).

_SRD_A = 12   # s12..s15 — GRA SRD
_SRD_B = 20   # s20..s23 — GRB SRD
_LDS_A = 64
_LDS_B = 128

# vgpr ranges. Stay below 200 so they don't collide with filler-MFMA ranges
# in `wrap_single_body` (see graph_native_validation_base._FILLER_RANGES).
_LRA0_DST = 8     # v8..v11
_LRB0_DST = 16    # v16..v19
_GRA_DST = 40     # v40..v43
_GRB_DST = 48     # v48..v51


def _make_grinc(srd_sgpr: int, slot: int, *, sequence: int = 0,
                category: str = "GRIncA") -> TaggedInstruction:
    """Wrap an SAddU32 that writes the SRD sgpr as a tagged GRInc instruction.

    SAddU32 dst=s<srd_sgpr> src0=s<srd_sgpr> src1=s100 mirrors the structure
    of the real GRInc inLower update (KernelWriterAssembly.incrementSrd):
    self-add of the lower 32 bits of the SRD by the per-iteration stride.
    The dst overlap with GR's SRD read is the load-bearing property —
    `_GenericALURule` publishes the sgpr write, and the per-byte resolver
    forms a RAW edge to any later reader of that sgpr.
    """
    inst = SAddU32(dst=sgpr(srd_sgpr, 1), src0=sgpr(srd_sgpr, 1),
                   src1=sgpr(100, 1))
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst), category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=slot, sequence=sequence),
    )


def _grinc_a(slot: int, *, sequence: int = 0) -> TaggedInstruction:
    return _make_grinc(_SRD_A, slot, sequence=sequence, category="GRIncA")


def _grinc_b(slot: int, *, sequence: int = 0) -> TaggedInstruction:
    return _make_grinc(_SRD_B, slot, sequence=sequence, category="GRIncB")


def _lra0(slot: int, *, sequence: int = 0) -> TaggedInstruction:
    return make_lr(_LRA0_DST, 4, _LDS_A, slot=slot,
                   category="LRA0", sequence=sequence)


def _lrb0(slot: int, *, sequence: int = 0) -> TaggedInstruction:
    return make_lr(_LRB0_DST, 4, _LDS_B, slot=slot,
                   category="LRB0", sequence=sequence)


def _gra(slot: int, *, sequence: int = 0,
         srd_sgpr_start: int = _SRD_A) -> TaggedInstruction:
    return make_gr(_GRA_DST, 4, srd_sgpr_start=srd_sgpr_start,
                   immediate_offset=_LDS_A, slot=slot,
                   category="GRA", sequence=sequence)


def _grb(slot: int, *, sequence: int = 0,
         srd_sgpr_start: int = _SRD_B) -> TaggedInstruction:
    return make_gr(_GRB_DST, 4, srd_sgpr_start=srd_sgpr_start,
                   immediate_offset=_LDS_B, slot=slot,
                   category="GRB", sequence=sequence)


# =============================================================================
# Tests
# =============================================================================


def _assert_inverted(failure, *, producer_category, subj_producer_idx,
                     consumer_category, subj_consumer_idx):
    """Inline assertion for graph-native OrderInvertedFailure shape.

    Graph nodes carry names like 'GRIncA@5.0' that include the slot, so
    the legacy `assert_order_inverted` helper (which compares against a
    bare 'GRIncA') doesn't match. We pin on `category` (the schedule
    category — 'GRIncA' / 'GRA') plus the SUBJECT-side
    `producer.position.vmfma_index` / `consumer.position.vmfma_index`,
    which is where the order inversion was actually observed (the legacy
    structural-rule assertion semantics). After the rocm-libraries-5v4u
    SchedulePosition collapse, `position` is a `_PositionStr` that parses
    the kernel-writer slot id back out of the rendered `@ idx=N` string —
    the value is sourced from `tagged_inst.slot.mfma_index` at label
    construction time, so identity is preserved.
    """
    assert failure.producer.category == producer_category, (
        f"producer.category: expected {producer_category!r}, got "
        f"{failure.producer.category!r}"
    )
    assert failure.consumer.category == consumer_category, (
        f"consumer.category: expected {consumer_category!r}, got "
        f"{failure.consumer.category!r}"
    )
    assert failure.producer.position.vmfma_index == subj_producer_idx, (
        f"producer.position.vmfma_index (subject): expected "
        f"{subj_producer_idx}, got "
        f"{failure.producer.position.vmfma_index}"
    )
    assert failure.consumer.position.vmfma_index == subj_consumer_idx, (
        f"consumer.position.vmfma_index (subject): expected "
        f"{subj_consumer_idx}, got "
        f"{failure.consumer.position.vmfma_index}"
    )


class TestGRMustStartAfterGRIncGraph(GraphNativeValidationTest):
    """Graph-native port of the legacy structural-rule tests.

    The legacy `set_gr_must_start_after_from_grinc` rule asserted that
    GR<X> must be issued after the last GRInc<X> in the same iteration.
    Graph-native equivalent: GRInc writes SRD sgprs, GR reads them; the
    sgpr RAW edge produced by `_GenericALURule` + the per-byte
    latest-writer resolver naturally encodes the ordering, and
    `compare_graphs` flips a reversed-order subject into
    `OrderInvertedFailure`.

    The LR0-barrier arc (legacy MissingBarrierFailure) is also still
    exercised here for the test cases that combine both invariants —
    those failures route through `validate_edge_wait_coverage` on the
    `lr_to_gr_lds_reuse` edge, identical to the phase-1 graph-native
    tests in `test_validate_gr_not_too_early_graph.py`.
    """

    # -------------------------------------------------------------------------
    # Positive paths — GRInc precedes GR (SRD RAW edge in canonical order)
    # -------------------------------------------------------------------------

    def test_basic_grinc_before_gr(self):
        """GRIncA at slot 1 finishes before GRA at slot 5. LR0+barrier
        present. Pass.

        Mirrors legacy `test_basic_grinc_before_gr`. The graph forms one
        SRD RAW edge (GRIncA s12 -> GRA s12) and one lr_to_gr_lds_reuse
        edge (LRA0 -> GRA covered by the SWait+SBarrier pair); both are
        well-formed.
        """
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            _lrb0(slot=1),
            _grinc_a(slot=1, sequence=1),
            _grinc_b(slot=1, sequence=2),
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            _gra(slot=5),
            _grb(slot=6),
        ]))
        graph = self.build_graph(cap)
        self.assert_no_failures(self.validate_waits(graph))
        # Pin: the SRD RAW edge GRIncA -> GRA exists in canonical order.
        self.assert_edge_exists(graph, producer_category="GRIncA",
                                consumer_category="GRA")

    def test_both_operands(self):
        """GRIncA/GRA and GRIncB/GRB pairs both present and ordered. Pass.
        Mirrors legacy `test_both_operands`."""
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            _lrb0(slot=1),
            _grinc_a(slot=1, sequence=1),
            _grinc_b(slot=1, sequence=2),
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            _gra(slot=5),
            _grb(slot=5, sequence=1),
        ]))
        graph = self.build_graph(cap)
        self.assert_no_failures(self.validate_waits(graph))
        self.assert_edge_exists(graph, producer_category="GRIncA",
                                consumer_category="GRA")
        self.assert_edge_exists(graph, producer_category="GRIncB",
                                consumer_category="GRB")

    def test_no_grinc_in_schedule(self):
        """GR present but no GRInc — SRD RAW arc is inactive. Only the
        LR0-barrier path is exercised. Mirrors legacy
        `test_no_grinc_in_schedule`."""
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            _lrb0(slot=1),
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            _gra(slot=5),
            _grb(slot=6),
        ]))
        graph = self.build_graph(cap)
        self.assert_no_failures(self.validate_waits(graph))

    def test_multiple_grinc_instructions(self):
        """Multiple GRIncA at distinct slots, all before GRA at slot 7.
        The per-byte resolver picks the LATEST GRInc as the producer of
        the SRD edge. Mirrors legacy `test_multiple_grinc_instructions`.
        """
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            _lrb0(slot=1),
            _grinc_a(slot=1, sequence=1),
            _grinc_a(slot=2, sequence=0),
            _grinc_b(slot=2, sequence=1),
            make_swait(slot=3, dscnt=0),
            make_sbarrier(slot=4),
            _gra(slot=7),
            _grb(slot=7, sequence=1),
        ]))
        graph = self.build_graph(cap)
        self.assert_no_failures(self.validate_waits(graph))

    # -------------------------------------------------------------------------
    # Positive — Swap pairing
    # -------------------------------------------------------------------------
    # SwapGlobalReadOrder semantics: GRA loads B (uses GRIncB's SRD),
    # GRB loads A (uses GRIncA's SRD). In graph-native terms this is a
    # crossover at the SRD-sgpr level: GRA's `srd_sgpr_start` points at
    # the sgpr that GRIncB writes (and vice versa). No special "swap"
    # flag is needed — the dataflow graph follows the registers.

    def test_swap_grinc_before_gr(self):
        """Swap: GRA reads GRIncB's SRD sgpr (s20). Both pairs ordered. Pass.
        Mirrors legacy `test_swap_grinc_before_gr`."""
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            _lrb0(slot=1),
            _grinc_b(slot=1, sequence=1),  # writes _SRD_B (s20) — GRA reads
            _grinc_a(slot=1, sequence=2),  # writes _SRD_A (s12) — GRB reads
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            # GRA reads SRD_B, GRB reads SRD_A under swap. Both placed AFTER
            # the SBarrier so the lr_to_gr_lds_reuse coverage is satisfied.
            _grb(slot=5, srd_sgpr_start=_SRD_A),
            _gra(slot=6, srd_sgpr_start=_SRD_B),
        ]))
        graph = self.build_graph(cap)
        self.assert_no_failures(self.validate_waits(graph))
        # Cross-pair: GRIncB's sgpr feeds GRA; GRIncA's sgpr feeds GRB.
        self.assert_edge_exists(graph, producer_category="GRIncB",
                                consumer_category="GRA")
        self.assert_edge_exists(graph, producer_category="GRIncA",
                                consumer_category="GRB")

    def test_swap_both_operands_cross_pairing(self):
        """Verify GRIncA->GRB and GRIncB->GRA cross-pairing via SRD sgprs.
        Both ordered, no failures. Mirrors legacy
        `test_swap_both_operands_cross_pairing`."""
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            _lrb0(slot=1),
            _grinc_a(slot=1, sequence=1),
            _grinc_b(slot=1, sequence=2),
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            _grb(slot=5, srd_sgpr_start=_SRD_A),  # cross-paired
            _gra(slot=6, srd_sgpr_start=_SRD_B),  # cross-paired
        ]))
        graph = self.build_graph(cap)
        self.assert_no_failures(self.validate_waits(graph))

    def test_grinc_and_gr_same_slot_grinc_first(self):
        """GRIncA and GRA at the same vmfma_index but with GRInc emitted
        FIRST in the captured stream (lower sub_index). The per-byte
        resolver pairs them by stream order, RAW edge canonical. Mirrors
        legacy `test_grinc_and_gr_same_index_grinc_declared_first`."""
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            _lrb0(slot=1),
            _grinc_a(slot=1, sequence=1),
            _grinc_b(slot=1, sequence=2),
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            # GRIncA at slot 5 sequence=0, GRA at slot 5 sequence=1 — same
            # vmfma_index but GRIncA emitted first.
            _make_grinc(_SRD_A, slot=5, sequence=0, category="GRIncA"),
            _gra(slot=5, sequence=1),
            _grb(slot=6),
        ]))
        graph = self.build_graph(cap)
        self.assert_no_failures(self.validate_waits(graph))

    # -------------------------------------------------------------------------
    # Negative paths — reversed GRInc/GR forms an inverted SRD RAW edge
    # -------------------------------------------------------------------------
    # In each negative case below, the REFERENCE places GRInc before GR
    # (canonical). The SUBJECT swaps the order. compare_graphs sees the
    # inverted RAW edge and emits OrderInvertedFailure with the GRInc as
    # producer and GR as consumer.

    def _ref_grinc_then_gr(self, *, gr_slot: int, grinc_slot: int,
                           gr_factory=_gra,
                           grinc_factory=_grinc_a):
        """Build a reference body where GRInc precedes GR with a clean
        LR0+barrier+drain pattern."""
        return self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            grinc_factory(slot=grinc_slot, sequence=0),
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            gr_factory(slot=gr_slot),
        ]))

    def _subj_gr_then_grinc(self, *, gr_slot: int, grinc_slot: int,
                            gr_factory=_gra,
                            grinc_factory=_grinc_a):
        """Build a subject body where GR precedes GRInc — the SRD RAW
        edge is inverted vs. the reference."""
        return self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            gr_factory(slot=gr_slot),
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            grinc_factory(slot=grinc_slot, sequence=0),
        ]))

    def test_negative_gr_before_grinc(self):
        """GRA at slot 4 issued BEFORE GRIncA at slot 5. The SRD RAW edge
        from GRIncA (sgpr write) to GRA (sgpr read) exists in the reference
        (slot 5 -> slot 4 + N) but inverted in subject. compare_graphs
        emits OrderInvertedFailure. Mirrors legacy
        `test_negative_gr_before_grinc`."""
        ref = self._ref_grinc_then_gr(gr_slot=6, grinc_slot=5)
        subj = self._subj_gr_then_grinc(gr_slot=4, grinc_slot=5)
        failures = self.compare(ref, subj)
        f = self.assert_failures_contain(
            failures, cls=OrderInvertedFailure,
        )
        _assert_inverted(
            f, producer_category="GRIncA", subj_producer_idx=5,
            consumer_category="GRA", subj_consumer_idx=4,
        )

    def test_negative_b_before_grinc_b(self):
        """GRB issued before GRIncB finishes. Same shape as the A-side,
        but exercising GRIncB -> GRB SRD pairing. Mirrors legacy
        `test_negative_b_before_grinc_b`."""
        ref = self._ref_grinc_then_gr(
            gr_slot=7, grinc_slot=6,
            gr_factory=_grb, grinc_factory=_grinc_b,
        )
        subj = self._subj_gr_then_grinc(
            gr_slot=5, grinc_slot=6,
            gr_factory=_grb, grinc_factory=_grinc_b,
        )
        failures = self.compare(ref, subj)
        f = self.assert_failures_contain(
            failures, cls=OrderInvertedFailure,
        )
        _assert_inverted(
            f, producer_category="GRIncB", subj_producer_idx=6,
            consumer_category="GRB", subj_consumer_idx=5,
        )

    def test_negative_swap_wrong_pairing(self):
        """Swap: GRA loads B so it reads GRIncB's SRD (s20). Subject puts
        GRA before GRIncB — RAW edge inverted. Mirrors legacy
        `test_negative_swap_wrong_pairing`.

        Reference: GRIncB at slot 5 writes s20; GRA at slot 6 reads s20.
        Subject: GRA at slot 4 reads s20 BEFORE GRIncB at slot 5 writes
        it. compare_graphs flags GRIncB -> GRA as inverted.
        """
        ref = self._ref_grinc_then_gr(
            gr_slot=6, grinc_slot=5,
            gr_factory=lambda slot, sequence=0: _gra(
                slot=slot, sequence=sequence, srd_sgpr_start=_SRD_B),
            grinc_factory=_grinc_b,
        )
        subj = self._subj_gr_then_grinc(
            gr_slot=4, grinc_slot=5,
            gr_factory=lambda slot, sequence=0: _gra(
                slot=slot, sequence=sequence, srd_sgpr_start=_SRD_B),
            grinc_factory=_grinc_b,
        )
        failures = self.compare(ref, subj)
        f = self.assert_failures_contain(
            failures, cls=OrderInvertedFailure,
        )
        _assert_inverted(
            f, producer_category="GRIncB", subj_producer_idx=5,
            consumer_category="GRA", subj_consumer_idx=4,
        )

    def test_negative_grinc_and_gr_same_slot_gr_first(self):
        """Same vmfma_index, GR emitted FIRST in the stream (GR sequence=0,
        GRInc sequence=1). Subject's stream-order put GR before GRInc at
        the same slot. Reference has GRInc first (canonical). compare_graphs
        flags the inverted RAW edge.

        Mirrors legacy
        `test_negative_grinc_and_gr_same_index_grinc_declared_after`.
        """
        ref = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            _make_grinc(_SRD_A, slot=5, sequence=0, category="GRIncA"),
            _gra(slot=5, sequence=1),
        ]))
        subj = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            make_swait(slot=2, dscnt=0),
            make_sbarrier(slot=3),
            _gra(slot=5, sequence=0),  # GR first at the slot
            _make_grinc(_SRD_A, slot=5, sequence=1, category="GRIncA"),
        ]))
        failures = self.compare(ref, subj)
        f = self.assert_failures_contain(
            failures, cls=OrderInvertedFailure,
        )
        _assert_inverted(
            f, producer_category="GRIncA", subj_producer_idx=5,
            consumer_category="GRA", subj_consumer_idx=5,
        )

    # -------------------------------------------------------------------------
    # Negative — LR0 barrier-coverage failure (lr_to_gr_lds_reuse arc)
    # -------------------------------------------------------------------------
    # The legacy rule's MissingBarrierFailure path was already migrated in
    # phase 1 (`test_validate_gr_not_too_early_graph.py`). One smoke test
    # is kept here to pin that the barrier check still fires when the
    # GRInc ordering passes — i.e. the two failure paths don't shadow
    # each other.

    def test_negative_grinc_ok_but_no_barrier(self):
        """GRInc ordering passes (GRIncA at slot 5 < GRA at slot 7), but
        the LR0 barrier window is missing — i.e. there's an early
        SWait+SBarrier pair (which forms the lr_to_gr_lds_reuse edge via
        `_collect_pattern`'s state machine), then a LATER SWait with no
        SBarrier following it before the GR. The wait-coverage check
        walks every wait in the (LR.position, GR.position) window —
        sees the later SWait (slot 6), promotes `last_drain` to it, and
        finds no SBarrier between slot 6 and the GR at slot 7.

        Mirrors legacy `test_negative_grinc_tighter_but_no_barrier` and
        the phase-1 graph-native `test_missing_barrier_lr_drained`.
        Verifies the barrier branch in `_classify_edge_coverage` for the
        `lr_to_gr_lds_reuse` edge still fires, even when the GRInc->GR
        SRD RAW arc is well-formed.
        """
        cap = self.wrap_single_body(make_capture(BODY_LABEL_ML, [
            _lra0(slot=0),
            _grinc_a(slot=1, sequence=0),
            make_swait(slot=2, dscnt=0),       # first drain — pattern matches
            make_sbarrier(slot=3),             # forms lr_to_gr_lds_reuse edge
            _make_grinc(_SRD_A, slot=4, sequence=0, category="GRIncA"),
            make_swait(slot=6, dscnt=0),       # later drain, no SBarrier after
            _gra(slot=7),
        ]))
        graph = self.build_graph(cap)
        # Pin: the lr_to_gr_lds_reuse edge IS formed (state machine
        # matched the slot-2/slot-3 SWait/SBarrier pair).
        self.assert_edge_exists(graph, edge_kind="lr_to_gr_lds_reuse")
        failures = self.validate_waits(graph)
        f = self.assert_failures_contain(failures, cls=MissingBarrierFailure)
        assert f.role == "must_start_after"
