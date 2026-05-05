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

import pytest

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_TO_LOOP_INDEX,
    Failure,
    GraphNode,
    OrderInvertedFailure,
    MissingWaitFailure,
    WaitTooLateFailure,
    WaitInsufficientFailure,
    MissingBarrierFailure,
    WrongInterleavingFailure,
    TimingTooCloseFailure,
    InvalidCounterValueFailure,
    SchedulePosition,
    SCCConflictFailure,
    SlotKey,
    SWaitCountExceedsOutstandingFailure,
    OutOfOrderSequenceFailure,
    LoopBodyCapture,
    TaggedInstruction,
    _ordinal,
    format_position,
)


def _make_node(category, name, vmfma_index, tagged_inst=None, body_label="ML"):
    return GraphNode(
        identity=(category, BODY_LABEL_TO_LOOP_INDEX[body_label], (name,)),
        position=SchedulePosition(
            loop_index=BODY_LABEL_TO_LOOP_INDEX[body_label],
            vmfma_index=vmfma_index,
            sub_index=0,
        ),
        category=category,
        rocisa_inst=None,
        tagged_inst=tagged_inst,
        body_label=body_label,
        name=name,
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


def test_format_position_empty_capture_returns_idx_only():
    """tagged_inst not in (empty) capture.instructions -> bare '@ idx=N'.
    Same fallback path that legacy callers triggered with capture=None;
    the fallback now routes through the tagged_inst-not-found branch
    instead of a removed top-level short-circuit."""
    node = _make_node("LRA0", "LRA0[0]", 7)
    assert format_position(node, LoopBodyCapture(instructions=[])) == "@ idx=7"


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
    earlier_grb = TaggedInstruction(inst=object(), category="GRB", slot=SlotKey(0, "ml", 0, 0))
    producer_tagged = TaggedInstruction(inst=object(), category="GRB", slot=SlotKey(0, "ml", 0, 1))
    consumer_tagged = TaggedInstruction(inst=object(), category="LRA1", slot=SlotKey(0, "ml", 0, 0))

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


def test_order_inverted_failure_format_mfma_consumer_omits_bracket():
    """Consumer is plain MFMA (category='MFMA'): the [N] suffix is omitted
    even when a capture is provided, because vmfma_index is the canonical
    identity. PackMFMAs (category 'PackA*'/'PackB*') would still get [N]."""
    producer_tagged = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml", 0, 0))
    consumer_tagged = TaggedInstruction(inst=object(), category="MFMA", slot=SlotKey(0, "ml", 0, 0))
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
    producer_tagged = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml", 0, 0))
    consumer_tagged = TaggedInstruction(inst=object(), category="MFMA", slot=SlotKey(0, "ml", 0, 0))
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
    producer_tagged = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml-1", 0, 0))
    consumer_tagged = TaggedInstruction(inst=object(), category="GRB", slot=SlotKey(0, "ml", 0, 0))
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


def test_missing_wait_failure_format_with_nearby_wrong_counter_hint():
    """When the window contains a wrong-counter SWaitCnt (former
    WaitOnWrongCounterFailure case), MissingWaitFailure surfaces it via
    nearby_other_counter_waits + appends a hint to the message."""
    producer = _make_node("LRA0", "LRA0", 5)
    consumer = _make_node("MFMA", "MFMA", 10)
    wrong_wait = _make_node("SYNC", "SWaitCnt", 7)
    failure = MissingWaitFailure(
        producer=producer,
        consumer=consumer,
        counter_kind="dscnt",
        nearby_other_counter_waits=[wrong_wait],
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert msg == (
        "SWaitCnt(dscnt) missing between LRA0 @ idx=5 and MFMA @ idx=10 "
        "(existing SWaitCnts at idx=7 drain other counters)."
    )


def test_wait_too_late_failure_format():
    producer = _make_node("LRA0", "LRA0[0]", 5)
    consumer = _make_node("MFMA", "MFMA", 10)
    failure = WaitTooLateFailure(
        producer=producer, consumer=consumer, wait_position=SchedulePosition(loop_index=1, vmfma_index=12, sub_index=0)
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert "fires at or after the consumer" in msg
    assert "@ idx=12" in msg
    assert "Move the wait earlier" in msg
    assert "(of next iteration)" not in msg


def test_wait_too_late_failure_format_with_capture_brackets():
    """capture=given: consumer renders as 'category[N] @ idx=M' with the
    per-category-stream index. Plain MFMA omits the bracket."""
    other_lra0 = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml", 0, 0))
    consumer_tagged = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml", 0, 1))
    producer = _make_node("MFMA", "MFMA", 5)
    consumer = _make_node("LRA0", "LRA0", 10, tagged_inst=consumer_tagged)
    capture = _capture_with(other_lra0, consumer_tagged)
    failure = WaitTooLateFailure(
        producer=producer, consumer=consumer,
        wait_position=SchedulePosition(loop_index=1, vmfma_index=12, sub_index=0),
    )
    msg = failure.format(capture=capture)
    assert "LRA0[1] @ idx=10" in msg     # second LRA0 in its category-stream


def test_wait_too_late_failure_format_cross_iteration():
    """Producer in loop i, consumer in loop i+1 -> '(of next iteration)' suffix."""
    producer = _make_node("LRA0", "LRA0[0]", 5, body_label="ML-1")
    consumer = _make_node("MFMA", "MFMA", 10, body_label="ML")
    failure = WaitTooLateFailure(
        producer=producer, consumer=consumer,
        wait_position=SchedulePosition(loop_index=1, vmfma_index=12, sub_index=0),
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert "@ idx=10 (of next iteration) is guaranteed" in msg


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
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert "queue depth at wait = 5" in msg
    assert "counter value (2)" in msg
    assert "Tighten the wait" in msg
    assert "(of next iteration)" not in msg


def test_wait_insufficient_failure_format_with_capture_brackets():
    """capture=given: producer + consumer get per-category [N] index. Wait
    stays bare (the message already says 'SWaitCnt')."""
    older_lra0 = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml", 0, 0))
    producer_tagged = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml", 0, 1))
    consumer_tagged = TaggedInstruction(inst=object(), category="MFMA", slot=SlotKey(0, "ml", 0, 0))
    producer = _make_node("LRA0", "LRA0", 5, tagged_inst=producer_tagged)
    consumer = _make_node("MFMA", "MFMA", 10, tagged_inst=consumer_tagged)
    wait = _make_node("SYNC", "SWaitCnt", 7)
    capture = _capture_with(older_lra0, producer_tagged, consumer_tagged)
    failure = WaitInsufficientFailure(
        producer=producer, consumer=consumer, wait=wait,
        queue_depth_at_wait=5, counter_value=2,
    )
    msg = failure.format(capture=capture)
    assert "MFMA @ idx=10" in msg          # plain MFMA — no bracket
    assert "MFMA[" not in msg
    assert "LRA0[1] @ idx=5" in msg        # producer is second LRA0
    assert "SWaitCnt @ idx=7" in msg       # wait stays bare


def test_wait_insufficient_failure_format_cross_iteration():
    """Producer in loop i, consumer in loop i+1 -> '(of next iteration)' suffix."""
    producer = _make_node("LRA0", "LRA0[0]", 5, body_label="ML-1")
    consumer = _make_node("MFMA", "MFMA", 10, body_label="ML")
    wait = _make_node("SYNC", "SWaitCnt", 7, body_label="ML")
    failure = WaitInsufficientFailure(
        producer=producer, consumer=consumer, wait=wait,
        queue_depth_at_wait=5, counter_value=2,
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert "@ idx=10 (of next iteration)'s producer" in msg


def test_missing_barrier_failure_must_start_after_format():
    producer = _make_node("LRA0", "LRA0[0]", 8)
    consumer = _make_node("GRA", "GRA[0]", 12)
    failure = MissingBarrierFailure(
        producer=producer, consumer=consumer, role="must_start_after"
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert "LRA0 -> SWaitCnt(dscnt=0) -> SBarrier -> GRA" in msg
    assert "GRA overwrites the LDS slot read by LRA0" in msg
    assert "(of next iteration)" not in msg


def test_missing_barrier_failure_needed_by_format():
    producer = _make_node("GRA", "GRA[0]", 8)
    consumer = _make_node("LRA1", "LRA1[0]", 22)
    failure = MissingBarrierFailure(
        producer=producer, consumer=consumer, role="needed_by"
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert "GRA -> SWaitCnt(vlcnt=0) -> SBarrier -> LRA1" in msg
    assert "LRA1 reads the LDS slot written by GRA" in msg
    assert "(of next iteration)" not in msg


def test_missing_barrier_failure_format_with_capture_brackets():
    """capture=given: consumer gets per-category [N] index in the trailing
    reference."""
    older_gra = TaggedInstruction(inst=object(), category="GRA", slot=SlotKey(0, "ml", 0, 0))
    consumer_tagged = TaggedInstruction(inst=object(), category="GRA", slot=SlotKey(0, "ml", 0, 1))
    producer = _make_node("LRA0", "LRA0[0]", 8)
    consumer = _make_node("GRA", "GRA[0]", 12, tagged_inst=consumer_tagged)
    capture = _capture_with(older_gra, consumer_tagged)
    failure = MissingBarrierFailure(
        producer=producer, consumer=consumer, role="must_start_after",
    )
    msg = failure.format(capture=capture)
    assert msg.endswith("GRA[1] @ idx=12."), msg


def test_missing_barrier_failure_format_cross_iteration():
    """Producer in loop i, consumer in loop i+1 -> '(of next iteration)' suffix
    on the trailing consumer reference. Mirrors the canonical lr_to_gr_lds_reuse
    cross-body case (LR0 in ML-1, GR in ML)."""
    producer = _make_node("LRA0", "LRA0[0]", 8, body_label="ML-1")
    consumer = _make_node("GRA", "GRA[0]", 2, body_label="ML")
    failure = MissingBarrierFailure(
        producer=producer, consumer=consumer, role="must_start_after"
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert msg.endswith("GRA @ idx=2 (of next iteration)."), msg


def test_wrong_interleaving_failure_format():
    """Empty capture: bare 'PackA0 @ idx=N' for each of pack /
    expected_next / actual_next (no per-category-stream [N] without a
    capture containing the tagged_inst)."""
    pack = _make_node("PackA0", "PackA0", 10)
    expected = _make_node("PackA0", "PackA0", 11)
    actual = _make_node("PackA0", "PackA0", 12)
    failure = WrongInterleavingFailure(
        pack=pack, expected_next=expected, actual_next=actual
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert msg == (
        "PackA0 @ idx=10 has wrong interleaving. Should have been "
        "followed by PackA0 @ idx=11 but was followed by PackA0 @ idx=12."
    )


def test_wrong_interleaving_failure_format_with_capture_brackets():
    """Capture given: pack / expected_next / actual_next get per-category
    [N] index — disambiguates the three same-category Packs."""
    pack_tagged = TaggedInstruction(inst=object(), category="PackA0", slot=SlotKey(0, "ml", 0, 0))
    expected_tagged = TaggedInstruction(inst=object(), category="PackA0", slot=SlotKey(0, "ml", 0, 1))
    actual_tagged = TaggedInstruction(inst=object(), category="PackA0", slot=SlotKey(0, "ml", 0, 2))
    pack = _make_node("PackA0", "PackA0", 10, tagged_inst=pack_tagged)
    expected = _make_node("PackA0", "PackA0", 11, tagged_inst=expected_tagged)
    actual = _make_node("PackA0", "PackA0", 12, tagged_inst=actual_tagged)
    capture = _capture_with(pack_tagged, expected_tagged, actual_tagged)
    failure = WrongInterleavingFailure(
        pack=pack, expected_next=expected, actual_next=actual
    )
    msg = failure.format(capture=capture)
    assert "PackA0[0] @ idx=10" in msg
    assert "PackA0[1] @ idx=11" in msg
    assert "PackA0[2] @ idx=12" in msg


def test_timing_too_close_failure_format():
    """Empty capture: bare 'PackA0 @ idx=N' / 'MFMA @ idx=N'. Plain MFMA
    omits [N] either way per _node_label's MFMA discriminator."""
    producer = _make_node("PackA0", "PackA0", 5)
    consumer = _make_node("MFMA", "MFMA", 6)
    failure = TimingTooCloseFailure(
        producer=producer, consumer=consumer,
        expected_quad_cycles=2, actual_quad_cycles=1,
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert msg == (
        "PackA0 @ idx=5 has too little gap between it and MFMA @ idx=6. "
        "Expected at least 2 quad-cycles but only 1 passed."
    )


def test_timing_too_close_failure_format_with_capture_brackets():
    """Capture given: producer Pack gets [N]; plain-MFMA consumer omits."""
    producer_tagged = TaggedInstruction(inst=object(), category="PackA0", slot=SlotKey(0, "ml", 0, 0))
    consumer_tagged = TaggedInstruction(inst=object(), category="MFMA", slot=SlotKey(0, "ml", 0, 0))
    producer = _make_node("PackA0", "PackA0", 5, tagged_inst=producer_tagged)
    consumer = _make_node("MFMA", "MFMA", 6, tagged_inst=consumer_tagged)
    capture = _capture_with(producer_tagged, consumer_tagged)
    failure = TimingTooCloseFailure(
        producer=producer, consumer=consumer,
        expected_quad_cycles=2, actual_quad_cycles=1,
    )
    msg = failure.format(capture=capture)
    assert "PackA0[0] @ idx=5" in msg
    assert "MFMA @ idx=6" in msg
    assert "MFMA[" not in msg


def test_invalid_counter_value_failure_format():
    swait = _make_node("SYNC", "SWaitCnt", 0)
    swait.issued_at = SchedulePosition(loop_index=1, vmfma_index=4, sub_index=0)
    failure = InvalidCounterValueFailure(
        swait=swait, dscnt=-2, vlcnt=0, vscnt=-1
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert "SWaitCnt @ idx=4 is invalid" in msg
    assert "dscnt=-2" in msg
    assert "vlcnt=0" in msg
    assert ">= -1" in msg


def test_scc_conflict_failure_format_graph_shape():
    """Graph-native shape (mrj.2) — populated by diagnose_missing_edge
    when an SCC reference edge is missing from the subject graph due to
    an intervening SCC writer. Empty capture: bare 'category @ idx=N' for
    every node; brackets only appear when the capture contains the
    relevant tagged_inst."""
    producer = _make_node("GRIncA", "scc_producer", 4)
    intervening = _make_node("GRIncA", "scc_clobber", 5)
    consumer = _make_node("GRIncA", "scc_consumer", 6)
    failure = SCCConflictFailure(
        producer=producer, consumer=consumer, intervening_writer=intervening,
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert "GRIncA @ idx=6's SCC read should resolve" in msg
    assert "producer GRIncA @ idx=4" in msg
    assert "Intervening SCC writer GRIncA @ idx=5" in msg


def test_scc_conflict_failure_format_with_capture_brackets():
    """capture=given: producer / consumer / intervening_writer get the
    per-category-stream [N] index. The user schedules SCC instructions by
    position, so the index is the right discriminator."""
    other_grinca = TaggedInstruction(inst=object(), category="GRIncA", slot=SlotKey(0, "ml", 0, 0))
    producer_tagged = TaggedInstruction(inst=object(), category="GRIncA", slot=SlotKey(0, "ml", 0, 1))
    intervening_tagged = TaggedInstruction(inst=object(), category="GRIncA", slot=SlotKey(0, "ml", 0, 2))
    consumer_tagged = TaggedInstruction(inst=object(), category="GRIncA", slot=SlotKey(0, "ml", 0, 3))
    producer = _make_node("GRIncA", "scc_producer", 4, tagged_inst=producer_tagged)
    intervening = _make_node("GRIncA", "scc_clobber", 5, tagged_inst=intervening_tagged)
    consumer = _make_node("GRIncA", "scc_consumer", 6, tagged_inst=consumer_tagged)
    capture = _capture_with(other_grinca, producer_tagged, intervening_tagged, consumer_tagged)
    failure = SCCConflictFailure(
        producer=producer, consumer=consumer, intervening_writer=intervening,
    )
    msg = failure.format(capture=capture)
    assert "GRIncA[1] @ idx=4" in msg     # producer is 2nd GRIncA in stream
    assert "GRIncA[2] @ idx=5" in msg     # intervening is 3rd
    assert "GRIncA[3] @ idx=6" in msg     # consumer is 4th


def test_scc_conflict_failure_format_graph_shape_no_clobber():
    """Graph-native shape with no identifiable intervening writer — should
    still render producer/consumer info with [N] index when a capture is
    given, and without an Intervening clause."""
    older_grinca = TaggedInstruction(inst=object(), category="GRIncA", slot=SlotKey(0, "ml", 0, 0))
    producer_tagged = TaggedInstruction(inst=object(), category="GRIncA", slot=SlotKey(0, "ml", 0, 1))
    consumer_tagged = TaggedInstruction(inst=object(), category="GRIncA", slot=SlotKey(0, "ml", 0, 2))
    producer = _make_node("GRIncA", "scc_producer", 4, tagged_inst=producer_tagged)
    consumer = _make_node("GRIncA", "scc_consumer", 6, tagged_inst=consumer_tagged)
    capture = _capture_with(older_grinca, producer_tagged, consumer_tagged)
    failure = SCCConflictFailure(
        producer=producer, consumer=consumer, intervening_writer=None,
    )
    msg = failure.format(capture=capture)
    assert "Intervening SCC writer" not in msg
    assert "GRIncA[1] @ idx=4" in msg     # producer is 2nd GRIncA
    assert "GRIncA[2] @ idx=6" in msg     # consumer is 3rd


def test_swait_count_exceeds_outstanding_failure_format_dscnt():
    swait = _make_node("SYNC", "SWaitCnt", 0)
    swait.issued_at = SchedulePosition(loop_index=1, vmfma_index=8, sub_index=0)
    failure = SWaitCountExceedsOutstandingFailure(
        swait=swait, counter_kind="dscnt", counter_value=3, outstanding=1
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert "SWaitCnt @ idx=8" in msg
    assert "dscnt=3" in msg
    assert "1 DS loads" in msg


def test_swait_count_exceeds_outstanding_failure_format_vlcnt():
    swait = _make_node("SYNC", "SWaitCnt", 0)
    swait.issued_at = SchedulePosition(loop_index=1, vmfma_index=8, sub_index=0)
    failure = SWaitCountExceedsOutstandingFailure(
        swait=swait, counter_kind="vlcnt", counter_value=4, outstanding=2
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
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
    msg = failure.format(LoopBodyCapture(instructions=[]))
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
    msg = failure.format(LoopBodyCapture(instructions=[]))
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
        f.format(LoopBodyCapture(instructions=[]))


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
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert isinstance(msg, str) and msg
