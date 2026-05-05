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

from typing import Optional

import pytest

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_TO_LOOP_INDEX,
    Failure,
    GraphNode,
    OrderInvertedFailure,
    MissingWaitFailure,
    WaitInsufficientFailure,
    MissingBarrierFailure,
    OverriddenInputFailure,
    TimingTooCloseFailure,
    InvalidCounterValueFailure,
    SchedulePosition,
    SlotKey,
    LoopBodyCapture,
    TaggedInstruction,
    _ordinal,
)


def _make_node(
    category: str,
    name: str,
    vmfma_index: int,
    tagged_inst: Optional[TaggedInstruction] = None,
    body_label: str = "ML",
) -> GraphNode:
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


def _capture_with(*tagged_instructions: TaggedInstruction) -> LoopBodyCapture:
    """Build a LoopBodyCapture whose .instructions list is exactly the given list.

    The instructions need only support `==` against themselves (default object
    identity is fine), since `_node_label` uses list.index() for the
    per-category-stream [N] lookup.
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
    transition in the captured 4-body model.

    Both producer and consumer tagged_insts go in the capture so the strict
    [N] lookup succeeds; in production a Failure with nodes from multiple
    bodies is rendered against a unified capture or the per-body capture
    that contains every referenced node's tagged_inst."""
    producer_tagged = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml-1", 0, 0))
    consumer_tagged = TaggedInstruction(inst=object(), category="GRB", slot=SlotKey(0, "ml", 0, 0))
    producer = _make_node("LRA0", "LRA0", 5, tagged_inst=producer_tagged, body_label="ML-1")
    consumer = _make_node("GRB", "GRB", 6, tagged_inst=consumer_tagged, body_label="ML")
    capture = _capture_with(producer_tagged, consumer_tagged)
    failure = MissingWaitFailure(
        producer=producer, consumer=consumer, counter_kind="dscnt"
    )
    msg = failure.format(capture=capture)
    assert msg == "SWaitCnt(dscnt) missing between LRA0[0] @ idx=5 and GRB[0] @ idx=6 (of next iteration)."


def test_missing_wait_failure_format_with_nearby_wrong_counter_hint():
    """When the window contains a wrong-counter SWaitCnt (former
    WaitOnWrongCounterFailure case), MissingWaitFailure surfaces it via
    nearby_other_counter_waits + appends a hint to the message."""
    producer_tagged = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml", 0, 0))
    producer = _make_node("LRA0", "LRA0", 5, tagged_inst=producer_tagged)
    consumer = _make_node("MFMA", "MFMA", 10)        # MFMA exempt from [N]
    wrong_wait = _make_node("SYNC", "SWaitCnt", 7)   # SYNC not labeled, only its idx is read
    capture = _capture_with(producer_tagged)
    failure = MissingWaitFailure(
        producer=producer,
        consumer=consumer,
        counter_kind="dscnt",
        nearby_other_counter_waits=[wrong_wait],
    )
    msg = failure.format(capture)
    assert msg == (
        "SWaitCnt(dscnt) missing between LRA0[0] @ idx=5 and MFMA @ idx=10 "
        "(existing SWaitCnts at idx=7 drain other counters)."
    )


def test_wait_insufficient_failure_format_with_capture_brackets():
    """SWaitCnt-as-subject wording: dscnt too high to drain producer.
    Producer + consumer get per-category [N] index; wait stays bare."""
    older_lra0 = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml", 0, 0))
    producer_tagged = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml", 0, 1))
    consumer_tagged = TaggedInstruction(inst=object(), category="MFMA", slot=SlotKey(0, "ml", 0, 0))
    producer = _make_node("LRA0", "LRA0", 5, tagged_inst=producer_tagged)
    consumer = _make_node("MFMA", "MFMA", 10, tagged_inst=consumer_tagged)
    wait = _make_node("SYNC", "SWaitCnt", 7)
    capture = _capture_with(older_lra0, producer_tagged, consumer_tagged)
    # Producer at position 0, queue_depth 5 -> max acceptable counter = 4.
    failure = WaitInsufficientFailure(
        producer=producer, consumer=consumer, wait=wait,
        counter_kind="dscnt", counter_value=4,
        queue_depth_at_wait=5, producer_position=0,
    )
    msg = failure.format(capture=capture)
    assert msg == (
        "SWaitCnt @ idx=7 has a dscnt that's too high to guarantee producer "
        "LRA0[1] @ idx=5 for consumer MFMA @ idx=10. "
        "Current value of 4 must be in range [0, 4]."
    )


def test_wait_insufficient_failure_format_must_be_zero():
    """Y=0 special case: producer is the most-recent op in the queue, so
    the only acceptable counter value is 0 (drain everything)."""
    producer_tagged = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml", 0, 0))
    consumer_tagged = TaggedInstruction(inst=object(), category="MFMA", slot=SlotKey(0, "ml", 0, 0))
    producer = _make_node("LRA0", "LRA0", 5, tagged_inst=producer_tagged)
    consumer = _make_node("MFMA", "MFMA", 10, tagged_inst=consumer_tagged)
    wait = _make_node("SYNC", "SWaitCnt", 7)
    capture = _capture_with(producer_tagged, consumer_tagged)
    # Producer at position 4 (last in queue), queue_depth 5 -> max acceptable = 0.
    failure = WaitInsufficientFailure(
        producer=producer, consumer=consumer, wait=wait,
        counter_kind="dscnt", counter_value=2,
        queue_depth_at_wait=5, producer_position=4,
    )
    msg = failure.format(capture=capture)
    assert msg == (
        "SWaitCnt @ idx=7 has a dscnt that's too high to guarantee producer "
        "LRA0[0] @ idx=5 for consumer MFMA @ idx=10. "
        "Current value of 2 must be 0."
    )


def test_wait_insufficient_failure_format_cross_iteration():
    """Producer in loop i, consumer in loop i+1 -> '(of next iteration)' suffix
    on the consumer rendering."""
    producer_tagged = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml-1", 0, 0))
    producer = _make_node("LRA0", "LRA0[0]", 5, tagged_inst=producer_tagged, body_label="ML-1")
    consumer = _make_node("MFMA", "MFMA", 10, body_label="ML")
    wait = _make_node("SYNC", "SWaitCnt", 7, body_label="ML")
    capture = _capture_with(producer_tagged)
    failure = WaitInsufficientFailure(
        producer=producer, consumer=consumer, wait=wait,
        counter_kind="dscnt", counter_value=2,
        queue_depth_at_wait=5, producer_position=0,
    )
    msg = failure.format(capture)
    assert "consumer MFMA @ idx=10 (of next iteration)" in msg


def test_missing_barrier_failure_must_start_after_format():
    """Pins the LR0 -> GR LDS-write barrier message wording."""
    producer_tagged = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml", 0, 0))
    consumer_tagged = TaggedInstruction(inst=object(), category="GRA", slot=SlotKey(0, "ml", 0, 0))
    producer = _make_node("LRA0", "LRA0[0]", 8, tagged_inst=producer_tagged)
    consumer = _make_node("GRA", "GRA[0]", 12, tagged_inst=consumer_tagged)
    wait = _make_node("SYNC", "SWaitCnt", 10)
    capture = _capture_with(producer_tagged, consumer_tagged)
    failure = MissingBarrierFailure(
        producer=producer, consumer=consumer, wait=wait,
    )
    msg = failure.format(capture)
    assert msg == (
        "SBarrier missing between SWaitCnt @ idx=10 and consumer "
        "GRA[0] @ idx=12, needed for producer LRA0[0] @ idx=8."
    )


def test_missing_barrier_failure_needed_by_format():
    """GR -> LR1 LDS-read scenario uses the same compact wording as
    the LR -> GR LDS-write scenario; producer/consumer categories make
    the direction obvious."""
    producer_tagged = TaggedInstruction(inst=object(), category="GRA", slot=SlotKey(0, "ml", 0, 0))
    consumer_tagged = TaggedInstruction(inst=object(), category="LRA1", slot=SlotKey(0, "ml", 0, 0))
    producer = _make_node("GRA", "GRA[0]", 8, tagged_inst=producer_tagged)
    consumer = _make_node("LRA1", "LRA1[0]", 22, tagged_inst=consumer_tagged)
    wait = _make_node("SYNC", "SWaitCnt", 18)
    capture = _capture_with(producer_tagged, consumer_tagged)
    failure = MissingBarrierFailure(
        producer=producer, consumer=consumer, wait=wait,
    )
    msg = failure.format(capture)
    assert msg == (
        "SBarrier missing between SWaitCnt @ idx=18 and consumer "
        "LRA1[0] @ idx=22, needed for producer GRA[0] @ idx=8."
    )


def test_missing_barrier_failure_format_with_capture_brackets():
    """capture=given: consumer gets per-category [N] index."""
    older_gra = TaggedInstruction(inst=object(), category="GRA", slot=SlotKey(0, "ml", 0, 0))
    consumer_tagged = TaggedInstruction(inst=object(), category="GRA", slot=SlotKey(0, "ml", 0, 1))
    producer_tagged = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml", 0, 0))
    producer = _make_node("LRA0", "LRA0[0]", 8, tagged_inst=producer_tagged)
    consumer = _make_node("GRA", "GRA[0]", 12, tagged_inst=consumer_tagged)
    wait = _make_node("SYNC", "SWaitCnt", 10)
    capture = _capture_with(producer_tagged, older_gra, consumer_tagged)
    failure = MissingBarrierFailure(
        producer=producer, consumer=consumer, wait=wait,
    )
    msg = failure.format(capture=capture)
    assert "consumer GRA[1] @ idx=12" in msg


def test_missing_barrier_failure_format_cross_iteration():
    """Producer in loop i, consumer in loop i+1 -> '(of next iteration)' suffix
    on the consumer rendering."""
    consumer_tagged = TaggedInstruction(inst=object(), category="GRA", slot=SlotKey(0, "ml", 0, 0))
    producer_tagged = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml-1", 0, 0))
    producer = _make_node("LRA0", "LRA0[0]", 8, tagged_inst=producer_tagged, body_label="ML-1")
    consumer = _make_node("GRA", "GRA[0]", 2, tagged_inst=consumer_tagged, body_label="ML")
    wait = _make_node("SYNC", "SWaitCnt", 0, body_label="ML")
    capture = _capture_with(producer_tagged, consumer_tagged)
    failure = MissingBarrierFailure(
        producer=producer, consumer=consumer, wait=wait,
    )
    msg = failure.format(capture)
    assert "consumer GRA[0] @ idx=2 (of next iteration)" in msg


def test_overridden_input_failure_format_pack_pair():
    """Pack pair-leader's vgpr clobbered by an intervening pair-leader of
    the same Pack category."""
    pack_tagged = TaggedInstruction(inst=object(), category="PackA0", slot=SlotKey(0, "ml", 0, 0))
    expected_tagged = TaggedInstruction(inst=object(), category="PackA0", slot=SlotKey(0, "ml", 0, 1))
    actual_tagged = TaggedInstruction(inst=object(), category="PackA0", slot=SlotKey(0, "ml", 0, 2))
    pack = _make_node("PackA0", "PackA0", 10, tagged_inst=pack_tagged)
    expected = _make_node("PackA0", "PackA0", 11, tagged_inst=expected_tagged)
    actual = _make_node("PackA0", "PackA0", 12, tagged_inst=actual_tagged)
    capture = _capture_with(pack_tagged, expected_tagged, actual_tagged)
    failure = OverriddenInputFailure(
        producer=pack, consumer=expected,
        resource="vgpr",
        intervening_writer=actual,
    )
    msg = failure.format(capture=capture)
    assert msg == (
        "PackA0[2] @ idx=12 is incorrectly scheduled between producer "
        "PackA0[0] @ idx=10 and consumer PackA0[1] @ idx=11, clobbering "
        "the vgpr the consumer needs."
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
    assert "(of next iteration)" not in msg


def test_timing_too_close_failure_format_cross_iteration():
    """Cross-iteration Pack->MFMA timing: producer Pack in ML-1, consumer
    MFMA in ML. Mirrors the cross-body shape now reachable in production
    (post-2bu.3/4/5 cumulative_issue_cycles): the formatter must attach
    the "(of next iteration)" suffix to the consumer rendering, like
    MissingWaitFailure / WaitInsufficientFailure / MissingBarrierFailure
    already do."""
    producer_tagged = TaggedInstruction(inst=object(), category="PackA0", slot=SlotKey(0, "ml-1", 0, 0))
    consumer_tagged = TaggedInstruction(inst=object(), category="MFMA", slot=SlotKey(0, "ml", 0, 0))
    producer = _make_node("PackA0", "PackA0", 7, tagged_inst=producer_tagged, body_label="ML-1")
    consumer = _make_node("MFMA", "MFMA", 0, tagged_inst=consumer_tagged, body_label="ML")
    capture = _capture_with(producer_tagged, consumer_tagged)
    failure = TimingTooCloseFailure(
        producer=producer, consumer=consumer,
        expected_quad_cycles=2, actual_quad_cycles=1,
    )
    msg = failure.format(capture=capture)
    assert "PackA0[0] @ idx=7" in msg
    assert "MFMA @ idx=0 (of next iteration)" in msg


def test_invalid_counter_value_failure_format_single_bad():
    """Only the field below -1 appears in the message; valid fields (>= -1)
    are omitted so the user sees just what's wrong."""
    swait = _make_node("SYNC", "SWaitCnt", 0)
    swait.issued_at = SchedulePosition(loop_index=1, vmfma_index=4, sub_index=0)
    failure = InvalidCounterValueFailure(
        swait=swait, dscnt=-2, vlcnt=0, vscnt=-1
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert msg == (
        "SWaitCnt @ idx=4 is invalid: dscnt=-2. "
        "All counter fields must be >= -1."
    )


def test_invalid_counter_value_failure_format_multiple_bad():
    """Two fields below -1 -> both listed, comma-separated."""
    swait = _make_node("SYNC", "SWaitCnt", 0)
    swait.issued_at = SchedulePosition(loop_index=1, vmfma_index=7, sub_index=0)
    failure = InvalidCounterValueFailure(
        swait=swait, dscnt=-2, vlcnt=-1, vscnt=-3
    )
    msg = failure.format(LoopBodyCapture(instructions=[]))
    assert msg == (
        "SWaitCnt @ idx=7 is invalid: dscnt=-2, vscnt=-3. "
        "All counter fields must be >= -1."
    )


def test_overridden_input_failure_format_scc_clobber():
    """SCC carry-chain clobber: GRIncA[2] writes SCC between GRIncA[1]
    (producer) and GRIncA[3] (consumer)."""
    other_grinca = TaggedInstruction(inst=object(), category="GRIncA", slot=SlotKey(0, "ml", 0, 0))
    producer_tagged = TaggedInstruction(inst=object(), category="GRIncA", slot=SlotKey(0, "ml", 0, 1))
    intervening_tagged = TaggedInstruction(inst=object(), category="GRIncA", slot=SlotKey(0, "ml", 0, 2))
    consumer_tagged = TaggedInstruction(inst=object(), category="GRIncA", slot=SlotKey(0, "ml", 0, 3))
    producer = _make_node("GRIncA", "scc_producer", 4, tagged_inst=producer_tagged)
    intervening = _make_node("GRIncA", "scc_clobber", 5, tagged_inst=intervening_tagged)
    consumer = _make_node("GRIncA", "scc_consumer", 6, tagged_inst=consumer_tagged)
    capture = _capture_with(other_grinca, producer_tagged, intervening_tagged, consumer_tagged)
    failure = OverriddenInputFailure(
        producer=producer, consumer=consumer, resource="SCC",
        intervening_writer=intervening,
    )
    msg = failure.format(capture=capture)
    assert msg == (
        "GRIncA[2] @ idx=5 is incorrectly scheduled between producer "
        "GRIncA[1] @ idx=4 and consumer GRIncA[3] @ idx=6, clobbering "
        "the SCC the consumer needs."
    )




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

    capture is constructed in the same scope as the failure (the producer
    needs tagged_inst -> capture lookup to succeed under strict mode); the
    test demonstrates that format() reads only the failure + capture, not
    any other data in the calling scope.
    """
    def _build():
        producer_tagged = TaggedInstruction(inst=object(), category="LRA0", slot=SlotKey(0, "ml", 0, 0))
        return (
            MissingWaitFailure(
                producer=_make_node("LRA0", "LRA0[0]", 5, tagged_inst=producer_tagged),
                consumer=_make_node("MFMA", "MFMA", 10),
                counter_kind="dscnt",
            ),
            _capture_with(producer_tagged),
        )

    failure, capture = _build()
    # Function's locals are gone except for what we returned; format() must
    # work purely from the failure + capture handed back.
    msg = failure.format(capture)
    assert isinstance(msg, str) and msg
