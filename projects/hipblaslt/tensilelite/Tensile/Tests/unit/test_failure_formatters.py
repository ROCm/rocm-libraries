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
"""Pinning tests for Failure.format() output wording.

This is the only place wording is asserted. Detection-site rules and tests
assert on Failure type and field, not on string content. If a formatter's
wording legitimately changes, only this file's tests update.
"""

from dataclasses import dataclass

import pytest

from Tensile.Components.ScheduleCapture import (
    Failure,
    OrderInvertedFailure,
    MissingWaitFailure,
    WaitOnWrongCounterFailure,
    WaitTooLateFailure,
    WaitInsufficientFailure,
    MissingBarrierFailure,
    WrongInterleavingFailure,
    TimingTooCloseFailure,
    InvalidCounterValueFailure,
    SCCConflictFailure,
    SWaitCountExceedsOutstandingFailure,
    OutOfOrderSequenceFailure,
    LoopBodyCapture,
    _ordinal,
    format_position,
)


# =============================================================================
# Minimal synthetic stand-ins
# =============================================================================
# Real GraphNode/SchedulePosition aren't constructed here on purpose: format()
# only reads attributes, so duck-typed stand-ins suffice and keep the tests
# isolated from production capture-pipeline plumbing.


@dataclass
class _Pos:
    vmfma_index: int


@dataclass
class _Node:
    category: str
    name: str
    position: _Pos
    tagged_inst: object = None
    body_label: str = "ML"


def _make_node(category, name, vmfma_index, tagged_inst=None, body_label="ML"):
    return _Node(
        category=category,
        name=name,
        position=_Pos(vmfma_index=vmfma_index),
        tagged_inst=tagged_inst,
        body_label=body_label,
    )


def _capture_with(*tagged_instructions):
    """Build a LoopBodyCapture whose .instructions list is exactly the given list.

    The instructions need only support `==` against themselves (default object
    identity is fine), since format_position uses list.index().
    """
    return LoopBodyCapture(instructions=list(tagged_instructions))


# =============================================================================
# _ordinal
# =============================================================================


@pytest.mark.parametrize(
    "n,expected",
    [
        (1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"),
        (10, "10th"), (11, "11th"), (12, "12th"), (13, "13th"),
        (21, "21st"), (22, "22nd"), (23, "23rd"),
        (101, "101st"), (102, "102nd"), (111, "111th"), (113, "113th"),
    ],
)
def test_ordinal(n, expected):
    assert _ordinal(n) == expected


# =============================================================================
# format_position
# =============================================================================


def test_format_position_no_capture_returns_idx_only():
    node = _make_node("LRA0", "LRA0[0]", 7)
    assert format_position(node, capture=None) == "@ idx=7"


def test_format_position_excludes_list_suffix_for_plain_mfma():
    """MFMAs are fixed by the underlying instruction loop — no list suffix."""
    other = object()
    node = _make_node("MFMA", "MFMA", 7, tagged_inst=object())
    capture = _capture_with(other, node.tagged_inst, other)
    rendered = format_position(node, capture)
    assert "(2nd entry in list)" not in rendered
    assert rendered == "@ idx=7"


def test_format_position_includes_list_suffix_for_lr():
    node = _make_node("LRA0", "LRA0[0]", 7, tagged_inst=object())
    other_a, other_b, other_c = object(), object(), object()
    capture = _capture_with(other_a, node.tagged_inst, other_b, other_c)
    assert format_position(node, capture) == "@ idx=7 (2nd entry in list)"


def test_format_position_includes_list_suffix_for_mfmapack():
    """MFMAPack inherits from both Pack and MFMA; the category-tag discriminator
    routes it to the user-scheduled branch (list-position included). Regression
    guard against accidentally checking isinstance(MFMAInstruction) instead of
    the category tag."""
    node = _make_node("PackB1", "PackB1[3]", 12, tagged_inst=object())
    other_a, other_b = object(), object()
    capture = _capture_with(other_a, node.tagged_inst, other_b)
    assert "(2nd entry in list)" in format_position(node, capture)
    assert "@ idx=12" in format_position(node, capture)


# =============================================================================
# Per-Failure-class formatter tests
# =============================================================================


def test_order_inverted_failure_format():
    """Pinning fixture mirrors a REAL OrderInvertedFailure shape captured from
    test_ValidateGRsCompleteBeforeLr1s.py::test_swap_global_read_order_failure:
    GRB producer (CMS idx=3, default idx=0) and LRA1 consumer (CMS idx=2,
    default idx=5). Both are non-MFMA categories, so both can shift across
    schedules — CMS placed GRB later than default and LRA1 earlier, inverting
    the producer-before-consumer order.

    Capture has a sibling GRB earlier in the stream so the producer renders
    as GRB[1] (second GRB in its category-stream); LRA1 is the only LRA1 so
    it renders as LRA1[0]."""
    from Tensile.Components.ScheduleCapture import SchedulePosition

    @dataclass(eq=False)
    class _TaggedStub:
        category: str

    earlier_grb = _TaggedStub(category="GRB")
    producer_tagged = _TaggedStub(category="GRB")
    consumer_tagged = _TaggedStub(category="LRA1")

    producer = _make_node("GRB", "GRB", 3, tagged_inst=producer_tagged)
    consumer = _make_node("LRA1", "LRA1", 2, tagged_inst=consumer_tagged)
    capture = _capture_with(earlier_grb, consumer_tagged, producer_tagged)

    failure = OrderInvertedFailure(
        producer=producer,
        consumer=consumer,
        default_producer_position=SchedulePosition(loop_index=1, vmfma_index=0, sub_index=0),
        default_consumer_position=SchedulePosition(loop_index=1, vmfma_index=5, sub_index=0),
    )
    msg = failure.format(capture=capture)
    assert "GRB[1] @ idx=3" in msg              # second GRB in its category-stream
    assert "LRA1[0] @ idx=2" in msg             # first (and only) LRA1
    assert "is issued after its consumer" in msg
    assert "first consumer" not in msg          # only one consumer per failure; "first" dropped
    assert "Default schedule" not in msg        # default-side prose removed
    assert "CMS reverses" not in msg            # default-side prose removed
    assert "must complete before" not in msg    # generic prose dropped earlier


def test_order_inverted_failure_format_no_capture_falls_back():
    """capture=None path: per-category [N] omitted, just the bare category."""
    from Tensile.Components.ScheduleCapture import SchedulePosition
    producer = _make_node("GRB", "GRB", 3)
    consumer = _make_node("LRA1", "LRA1", 2)
    failure = OrderInvertedFailure(
        producer=producer,
        consumer=consumer,
        default_producer_position=SchedulePosition(loop_index=1, vmfma_index=0, sub_index=0),
        default_consumer_position=SchedulePosition(loop_index=1, vmfma_index=5, sub_index=0),
    )
    msg = failure.format(capture=None)
    assert "GRB @ idx=3" in msg
    assert "LRA1 @ idx=2" in msg
    assert "GRB[" not in msg     # no per-category index without a capture
    assert "LRA1[" not in msg


def test_order_inverted_failure_format_mfma_consumer_omits_bracket():
    """Consumer is plain MFMA (category='MFMA'): the [N] suffix is omitted
    even when a capture is provided, because vmfma_index is the canonical
    identity. PackMFMAs (category 'PackA*'/'PackB*') would still get [N]."""
    from Tensile.Components.ScheduleCapture import SchedulePosition

    @dataclass(eq=False)
    class _TaggedStub:
        category: str

    producer_tagged = _TaggedStub(category="LRA0")
    consumer_tagged = _TaggedStub(category="MFMA")
    producer = _make_node("LRA0", "LRA0", 5, tagged_inst=producer_tagged)
    consumer = _make_node("MFMA", "MFMA", 3, tagged_inst=consumer_tagged)
    capture = _capture_with(producer_tagged, consumer_tagged)
    failure = OrderInvertedFailure(
        producer=producer,
        consumer=consumer,
        default_producer_position=SchedulePosition(loop_index=1, vmfma_index=2, sub_index=0),
        default_consumer_position=SchedulePosition(loop_index=1, vmfma_index=3, sub_index=0),
    )
    msg = failure.format(capture=capture)
    assert "LRA0[0] @ idx=5" in msg
    assert "MFMA @ idx=3" in msg
    assert "MFMA[" not in msg     # plain MFMA omits [N]


def test_missing_wait_failure_format():
    """Plain same-iteration scenario: producer LRA0 in ML, consumer MFMA in
    ML, no SWaitCnt(dscnt) between. With a capture provided, producer renders
    as LRA0[0] (first LRA0 in stream); consumer is plain MFMA so no [N]."""
    @dataclass(eq=False)
    class _TaggedStub:
        category: str
    producer_tagged = _TaggedStub(category="LRA0")
    consumer_tagged = _TaggedStub(category="MFMA")
    producer = _make_node("LRA0", "LRA0", 0, tagged_inst=producer_tagged)
    consumer = _make_node("MFMA", "MFMA", 2, tagged_inst=consumer_tagged)
    capture = _capture_with(producer_tagged, consumer_tagged)
    failure = MissingWaitFailure(
        producer=producer, consumer=consumer, counter_kind="dscnt"
    )
    msg = failure.format(capture=capture)
    assert msg == "SWaitCnt(dscnt) missing between LRA0[0] @ idx=0 and MFMA @ idx=2."


def test_missing_wait_failure_format_cross_iteration():
    """Cross-iteration LDS-reuse: producer LRA0 in ML-1, consumer GRB in ML.
    Real-data fixture from test_validate_gr_not_too_early_graph.py::
    test_negative_prev_iter_lr0_not_drained — producer LRA0 @ idx=5 in ML-1,
    consumer GRB @ idx=6 in ML. The "(of next iteration)" suffix appends to
    the consumer's position because ML-1 -> ML IS the cross-loop-iteration
    transition in the captured 4-body model."""
    @dataclass(eq=False)
    class _TaggedStub:
        category: str
    producer_tagged = _TaggedStub(category="LRA0")
    consumer_tagged = _TaggedStub(category="GRB")
    producer = _make_node("LRA0", "LRA0", 5, tagged_inst=producer_tagged, body_label="ML-1")
    consumer = _make_node("GRB", "GRB", 6, tagged_inst=consumer_tagged, body_label="ML")
    # The capture passed in is the body the consumer lives in; fixture uses
    # the consumer's body to keep the per-category index meaningful for it.
    capture = _capture_with(consumer_tagged)
    failure = MissingWaitFailure(
        producer=producer, consumer=consumer, counter_kind="dscnt"
    )
    msg = failure.format(capture=capture)
    assert msg == "SWaitCnt(dscnt) missing between LRA0 @ idx=5 and GRB[0] @ idx=6 (of next iteration)."


def test_missing_wait_failure_format_no_capture():
    """capture=None: per-category [N] omitted; iteration note still works
    when body_labels are populated on the nodes."""
    producer = _make_node("LRA0", "LRA0", 0)
    consumer = _make_node("MFMA", "MFMA", 2)
    failure = MissingWaitFailure(
        producer=producer, consumer=consumer, counter_kind="vlcnt"
    )
    msg = failure.format(capture=None)
    assert msg == "SWaitCnt(vlcnt) missing between LRA0 @ idx=0 and MFMA @ idx=2."


def test_wait_on_wrong_counter_failure_format():
    producer = _make_node("LRA0", "LRA0[0]", 5)
    consumer = _make_node("MFMA", "MFMA", 10)
    wrong_wait = _make_node("SYNC", "SWaitCnt", 7)
    failure = WaitOnWrongCounterFailure(
        producer=producer,
        consumer=consumer,
        expected_counter="dscnt",
        wrong_counter_waits=[wrong_wait],
    )
    msg = failure.format(capture=None)
    assert "dscnt" in msg
    assert "Did you mean dscnt?" in msg
    assert "@ idx=7" in msg


def test_wait_too_late_failure_format():
    producer = _make_node("LRA0", "LRA0[0]", 5)
    consumer = _make_node("MFMA", "MFMA", 10)
    failure = WaitTooLateFailure(
        producer=producer, consumer=consumer, wait_position=_Pos(vmfma_index=12)
    )
    msg = failure.format(capture=None)
    assert "fires at or after the consumer" in msg
    assert "@ idx=12" in msg
    assert "Move the wait earlier" in msg


def test_wait_insufficient_failure_format():
    producer = _make_node("LRA0", "LRA0[0]", 5)
    consumer = _make_node("MFMA", "MFMA", 10)
    wait = _make_node("SYNC", "SWaitCnt", 7)
    failure = WaitInsufficientFailure(
        producer=producer,
        consumer=consumer,
        wait=wait,
        queue_depth_at_wait=5,
        counter_value=2,
    )
    msg = failure.format(capture=None)
    assert "queue depth at wait = 5" in msg
    assert "counter value (2)" in msg
    assert "Tighten the wait" in msg


def test_missing_barrier_failure_must_start_after_format():
    producer = _make_node("LRA0", "LRA0[0]", 8)
    consumer = _make_node("GRA", "GRA[0]", 12)
    failure = MissingBarrierFailure(
        producer=producer, consumer=consumer, role="must_start_after"
    )
    msg = failure.format(capture=None)
    assert "LRA0 -> SWaitCnt(dscnt=0) -> SBarrier -> GRA" in msg
    assert "GRA overwrites the LDS slot read by LRA0" in msg


def test_missing_barrier_failure_needed_by_format():
    producer = _make_node("GRA", "GRA[0]", 8)
    consumer = _make_node("LRA1", "LRA1[0]", 22)
    failure = MissingBarrierFailure(
        producer=producer, consumer=consumer, role="needed_by"
    )
    msg = failure.format(capture=None)
    assert "GRA -> SWaitCnt(vlcnt=0) -> SBarrier -> LRA1" in msg
    assert "LRA1 reads the LDS slot written by GRA" in msg


def test_wrong_interleaving_failure_format():
    pack = _make_node("PackA0", "MiddlePack_a", 0)
    pack.issued_at = _Pos(vmfma_index=10)
    expected = _make_node("PackA0", "MiddlePack_b", 0)
    expected.issued_at = _Pos(vmfma_index=11)
    actual = _make_node("PackA0", "MiddlePack_c", 0)
    actual.issued_at = _Pos(vmfma_index=12)
    failure = WrongInterleavingFailure(
        pack=pack, expected_next=expected, actual_next=actual
    )
    msg = failure.format(capture=None)
    assert "wrong interleaving" in msg
    assert "MiddlePack_a" in msg
    assert "MiddlePack_b" in msg
    assert "MiddlePack_c" in msg


def test_timing_too_close_failure_format():
    producer = _make_node("PackA0", "CVT0_a", 0)
    producer.issued_at = _Pos(vmfma_index=5)
    consumer = _make_node("MFMA", "MFMA", 0)
    consumer.issued_at = _Pos(vmfma_index=6)
    failure = TimingTooCloseFailure(
        producer=producer,
        consumer=consumer,
        expected_quad_cycles=2,
        actual_quad_cycles=1,
    )
    msg = failure.format(capture=None)
    assert "too little gap" in msg
    assert "Expected at least 2 quad-cycles" in msg
    assert "only 1 passed" in msg


def test_invalid_counter_value_failure_format():
    swait = _make_node("SYNC", "SWaitCnt", 0)
    swait.issued_at = _Pos(vmfma_index=4)
    failure = InvalidCounterValueFailure(
        swait=swait, dscnt=-2, vlcnt=0, vscnt=-1
    )
    msg = failure.format(capture=None)
    assert "SWaitCnt @ idx=4 is invalid" in msg
    assert "dscnt=-2" in msg
    assert "vlcnt=0" in msg
    assert ">= -1" in msg


def test_scc_conflict_failure_format_graph_shape():
    """Graph-native shape (mrj.2) — populated by diagnose_missing_edge
    when an SCC reference edge is missing from the subject graph due to
    an intervening SCC writer."""
    class _FakeInst:
        pass
    producer_inst = type("SCmpEQU32", (_FakeInst,), {})()
    intervening_inst = type("SAddU32", (_FakeInst,), {})()
    consumer_inst = type("SCSelectB32", (_FakeInst,), {})()
    producer = _make_node("GRIncA", "scc_producer", 4)
    producer.rocisa_inst = producer_inst
    intervening = _make_node("GRIncA", "scc_clobber", 5)
    intervening.rocisa_inst = intervening_inst
    consumer = _make_node("GRIncA", "scc_consumer", 6)
    consumer.rocisa_inst = consumer_inst
    failure = SCCConflictFailure(
        producer=producer,
        consumer=consumer,
        intervening_writer=intervening,
    )
    msg = failure.format(capture=None)
    assert "GRIncA[SCSelectB32]" in msg
    assert "@ idx=6" in msg
    assert "GRIncA[SCmpEQU32]" in msg
    assert "@ idx=4" in msg
    assert "Intervening SCC writer" in msg
    assert "GRIncA[SAddU32]" in msg
    assert "@ idx=5" in msg


def test_scc_conflict_failure_format_graph_shape_no_clobber():
    """Graph-native shape with no identifiable intervening writer — should
    still render producer/consumer info without an Intervening clause."""
    producer = _make_node("GRIncA", "scc_producer", 4)
    producer.rocisa_inst = type("SCmpEQU32", (), {})()
    consumer = _make_node("GRIncA", "scc_consumer", 6)
    consumer.rocisa_inst = type("SCSelectB32", (), {})()
    failure = SCCConflictFailure(
        producer=producer,
        consumer=consumer,
        intervening_writer=None,
    )
    msg = failure.format(capture=None)
    assert "Intervening SCC writer" not in msg
    assert "GRIncA[SCmpEQU32]" in msg
    assert "GRIncA[SCSelectB32]" in msg


def test_swait_count_exceeds_outstanding_failure_format_dscnt():
    swait = _make_node("SYNC", "SWaitCnt", 0)
    swait.issued_at = _Pos(vmfma_index=8)
    failure = SWaitCountExceedsOutstandingFailure(
        swait=swait, counter_kind="dscnt", counter_value=3, outstanding=1
    )
    msg = failure.format(capture=None)
    assert "SWaitCnt @ idx=8" in msg
    assert "dscnt=3" in msg
    assert "1 DS loads" in msg


def test_swait_count_exceeds_outstanding_failure_format_vlcnt():
    swait = _make_node("SYNC", "SWaitCnt", 0)
    swait.issued_at = _Pos(vmfma_index=8)
    failure = SWaitCountExceedsOutstandingFailure(
        swait=swait, counter_kind="vlcnt", counter_value=4, outstanding=2
    )
    msg = failure.format(capture=None)
    assert "vlcnt=4" in msg
    assert "2 VM loads" in msg


def test_out_of_order_sequence_failure_sequence_format():
    failure = OutOfOrderSequenceFailure(
        kind="sequence",
        schedule_key="GRIncA",
        sequence=[0, 1, 1, 0],
        bad_value=0,
        bad_index=3,
        prev_value=1,
    )
    msg = failure.format(capture=None)
    assert "Non-descending-order rule failed" in msg
    assert "schedule key 'GRIncA'" in msg
    assert "sequence [0, 1, 1, 0]" in msg
    assert "value 0 at index 3 is less than 1 at index 2" in msg


def test_out_of_order_sequence_failure_cvt_pair_format():
    failure = OutOfOrderSequenceFailure(
        kind="cvt_pair",
        schedule_key="PackA0 group 3",
        sequence=("CVT0", "CVT1"),
        bad_value=12,
        bad_index=0,
        prev_value=15,
    )
    msg = failure.format(capture=None)
    assert "CVT pair ordering violated" in msg
    assert "PackA0 group 3" in msg
    assert "CVT0 ends at issue_index 15" in msg
    assert "CVT1 starts at issue_index 12" in msg


# =============================================================================
# Base-class contract
# =============================================================================


def test_failure_base_format_raises():
    """Subclasses must override format(); the base raises NotImplementedError."""
    f = Failure()
    with pytest.raises(NotImplementedError):
        f.format(capture=None)


def test_format_works_without_reference_in_scope():
    """Structural test: build a Failure inside a function; in the caller's scope
    where any reference data is unreachable, format() must still succeed.

    Replaces the brittle 'no field references the reference graph' assertion.
    If the Failure tried to dereference unrelated data, the format call would
    raise AttributeError or KeyError.
    """
    def _build():
        return MissingWaitFailure(
            producer=_make_node("LRA0", "LRA0[0]", 5),
            consumer=_make_node("MFMA", "MFMA", 10),
            counter_kind="dscnt",
        )

    failure = _build()
    # In this scope, the function's locals are gone — the format must work
    # purely from data on the Failure.
    msg = failure.format(capture=None)
    assert isinstance(msg, str) and msg
