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

Post-g4w: Failures consume eager source-aware `FailureNodeLabel` instances
constructed at Failure-construction time by source-aware label providers
(CMS-side via `cms_node_label`, asm-side via a future provider). The
formatter is pure string composition over `(primary, position)` pairs +
scalar fields stored on the Failure itself, so fixtures construct
FailureNodeLabel directly with literal strings.

Wording is preserved BIT-EXACTLY against the pre-g4w baseline (hof-era
pinnings); these tests are the canonical source of truth for that.
"""

import pytest

from Tensile.Components.ScheduleCapture import (
    SchedulePosition,
)
from Tensile.Components.CMSValidator import (
    Failure,
    FailureNodeLabel,
    OrderInvertedFailure,
    MissingWaitFailure,
    WaitInsufficientFailure,
    MissingBarrierFailure,
    OverriddenInputFailure,
    TimingTooCloseFailure,
    InvalidCounterValueFailure,
    _ordinal,
)


def _cms_label(category: str, idx: int, *, name_idx: int = None) -> FailureNodeLabel:
    """Build a CMS-style FailureNodeLabel.

    `category` is the bare schedule-stream tag (e.g. 'LRA0', 'PackA0', 'MFMA').
    Plain MFMA renders as bare 'MFMA' (no [N]); every other category renders
    as 'category[name_idx]' where name_idx is the per-category-stream index.
    `idx` is the vmfma_index that the position string carries.
    """
    if category == "MFMA" and name_idx is None:
        primary = category
    else:
        primary = f"{category}[{name_idx if name_idx is not None else 0}]"
    return FailureNodeLabel(
        primary=primary,
        position=f"@ idx={idx}",
        category=category,
    )


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
    schedules.

    Producer renders as GRB[1] (second GRB in its category-stream); LRA1 is
    the only LRA1 so it renders as LRA1[0]. Labels are pre-rendered by
    `cms_node_label` at construction time."""
    failure = OrderInvertedFailure(
        producer=_cms_label("GRB", 3, name_idx=1),
        consumer=_cms_label("LRA1", 2, name_idx=0),
        default_producer_position=SchedulePosition(loop_index=1, stream_index=0),
        default_consumer_position=SchedulePosition(loop_index=1, stream_index=5),
    )
    msg = failure.format()
    assert "GRB[1] @ idx=3" in msg              # second GRB in its category-stream
    assert "LRA1[0] @ idx=2" in msg             # first (and only) LRA1
    assert "is issued after consumer" in msg
    assert msg.startswith("Producer ")
    assert "first consumer" not in msg          # only one consumer per failure; "first" dropped
    assert "Default schedule" not in msg        # default-side prose removed
    assert "CMS reverses" not in msg            # default-side prose removed
    assert "must complete before" not in msg    # generic prose dropped earlier


def test_order_inverted_failure_format_mfma_consumer_omits_bracket():
    """Consumer is plain MFMA (category='MFMA'): the [N] suffix is omitted
    even when the per-category index is implicit, because vmfma_index is the
    canonical identity. PackMFMAs (category 'PackA*'/'PackB*') would still
    get [N]."""
    failure = OrderInvertedFailure(
        producer=_cms_label("LRA0", 5, name_idx=0),
        consumer=_cms_label("MFMA", 3),
        default_producer_position=SchedulePosition(loop_index=1, stream_index=2),
        default_consumer_position=SchedulePosition(loop_index=1, stream_index=3),
    )
    msg = failure.format()
    assert "LRA0[0] @ idx=5" in msg
    assert "MFMA @ idx=3" in msg
    assert "MFMA[" not in msg     # plain MFMA omits [N]


def test_missing_wait_failure_format():
    """Plain same-iteration scenario: producer LRA0, consumer MFMA, no
    SWaitCnt(dscnt) between. Producer renders as LRA0[0]; consumer is plain
    MFMA so no [N]. iter_delta=0 so no '(of next iteration)' suffix."""
    failure = MissingWaitFailure(
        producer=_cms_label("LRA0", 0, name_idx=0),
        consumer=_cms_label("MFMA", 2),
        counter_kind="dscnt",
    )
    msg = failure.format()
    assert msg == "SWaitCnt(dscnt) missing between producer LRA0[0] @ idx=0 and consumer MFMA @ idx=2."


def test_missing_wait_failure_format_cross_iteration():
    """Cross-iteration GR -> LR1 dataflow: producer GRA loads to LDS in ML-1,
    consumer LRA1 reads the same LDS slot in ML. iter_delta=1 selects the
    "(of next iteration)" suffix.

    Realistic shape: GR -> LR1 is the canonical cross-iteration dataflow
    (DTL=1: BufferLoad m0-routed to LDS, then LRA1 reads from LDS); the
    relevant counter is vlcnt (drains the global load).
    """
    failure = MissingWaitFailure(
        producer=_cms_label("GRA", 2, name_idx=0),
        consumer=_cms_label("LRA1", 4, name_idx=0),
        iter_delta=1,
        counter_kind="vlcnt",
    )
    msg = failure.format()
    assert msg == "SWaitCnt(vlcnt) missing between producer GRA[0] @ idx=2 and consumer LRA1[0] @ idx=4 (of next iteration)."


def test_missing_wait_failure_format_with_nearby_wrong_counter_hint():
    """When the window contains a wrong-counter SWaitCnt (former
    WaitOnWrongCounterFailure case), MissingWaitFailure surfaces it via
    nearby_wait_positions (a tuple of pre-rendered source-aware position
    strings, e.g. CMS form `"@ idx=7"`) + appends a hint to the message.

    Post-3dy: positions are rendered via the source-aware
    `TaggedInstructionLike.render_position()` Protocol at Failure-
    construction time; the formatter strips the leading "@ " for the
    inner comma-joined phrasing."""
    failure = MissingWaitFailure(
        producer=_cms_label("LRA0", 5, name_idx=0),
        consumer=_cms_label("MFMA", 10),
        counter_kind="dscnt",
        nearby_wait_positions=("@ idx=7",),
    )
    msg = failure.format()
    assert msg == (
        "SWaitCnt(dscnt) missing between producer LRA0[0] @ idx=5 and consumer MFMA @ idx=10 "
        "(existing SWaitCnts at idx=7 drain other counters)."
    )


def test_wait_insufficient_failure_format_dscnt_range():
    """dscnt-as-subject wording with range bound. Fixture: producer is at
    FIFO position 2 in a queue of depth 5, so max acceptable counter value
    is 5-2-1=2. Current value 4 is OUT of range [0, 2] -> failure.

    Producer is LRA0[1] (second LRA0 in its category-stream — there's an
    older LRA0 at the head of the queue)."""
    failure = WaitInsufficientFailure(
        producer=_cms_label("LRA0", 5, name_idx=1),
        consumer=_cms_label("MFMA", 10),
        wait_position="@ idx=7",
        counter_kind="dscnt", counter_value=4,
        queue_depth_at_wait=5, producer_position=2,
    )
    msg = failure.format()
    assert msg == (
        "dscnt for SWaitCnt @ idx=7 is too high to guarantee producer "
        "LRA0[1] @ idx=5 for consumer MFMA @ idx=10. "
        "Current value of 4 must be in range [0, 2]."
    )


def test_wait_insufficient_failure_format_vlcnt_range():
    """vlcnt variant: GR producer in vlcnt FIFO, LR1 consumer waits on the
    drain. Fixture: producer is at FIFO position 1 in queue depth 4,
    max acceptable = 4-1-1 = 2. Current value 3 is OUT of range [0, 2]."""
    failure = WaitInsufficientFailure(
        producer=_cms_label("GRA", 3, name_idx=0),
        consumer=_cms_label("LRA1", 8, name_idx=0),
        wait_position="@ idx=6",
        counter_kind="vlcnt", counter_value=3,
        queue_depth_at_wait=4, producer_position=1,
    )
    msg = failure.format()
    assert msg == (
        "vlcnt for SWaitCnt @ idx=6 is too high to guarantee producer "
        "GRA[0] @ idx=3 for consumer LRA1[0] @ idx=8. "
        "Current value of 3 must be in range [0, 2]."
    )


def test_wait_insufficient_failure_format_must_be_zero():
    """Y=0 special case: producer is the most-recent op in the queue (last
    to drain), so the only acceptable counter value is 0 (drain everything).
    Fixture: producer at position 4 in queue depth 5, max acceptable = 0."""
    failure = WaitInsufficientFailure(
        producer=_cms_label("LRA0", 5, name_idx=0),
        consumer=_cms_label("MFMA", 10),
        wait_position="@ idx=7",
        counter_kind="dscnt", counter_value=2,
        queue_depth_at_wait=5, producer_position=4,
    )
    msg = failure.format()
    assert msg == (
        "dscnt for SWaitCnt @ idx=7 is too high to guarantee producer "
        "LRA0[0] @ idx=5 for consumer MFMA @ idx=10. "
        "Current value of 2 must be 0."
    )


def test_wait_insufficient_failure_format_cross_iteration():
    """Producer in loop i, consumer in loop i+1 -> '(of next iteration)' suffix
    on the consumer rendering. Fixture: producer at position 2 in queue depth
    5, max acceptable = 2; current value 4 fails."""
    failure = WaitInsufficientFailure(
        producer=_cms_label("LRA0", 5, name_idx=0),
        consumer=_cms_label("MFMA", 10),
        iter_delta=1,
        wait_position="@ idx=7",
        counter_kind="dscnt", counter_value=4,
        queue_depth_at_wait=5, producer_position=2,
    )
    msg = failure.format()
    assert "consumer MFMA @ idx=10 (of next iteration)" in msg
    assert "Current value of 4 must be in range [0, 2]." in msg


def test_missing_barrier_failure_must_start_after_format():
    """Pins the LR0 -> GR LDS-write barrier message wording."""
    failure = MissingBarrierFailure(
        producer=_cms_label("LRA0", 8, name_idx=0),
        consumer=_cms_label("GRA", 12, name_idx=0),
        wait_position="@ idx=10",
    )
    msg = failure.format()
    assert msg == (
        "SBarrier missing between SWaitCnt @ idx=10 and consumer "
        "GRA[0] @ idx=12, needed for producer LRA0[0] @ idx=8."
    )


def test_missing_barrier_failure_needed_by_format():
    """GR -> LR1 LDS-read scenario uses the same compact wording as
    the LR -> GR LDS-write scenario; producer/consumer categories make
    the direction obvious."""
    failure = MissingBarrierFailure(
        producer=_cms_label("GRA", 8, name_idx=0),
        consumer=_cms_label("LRA1", 22, name_idx=0),
        wait_position="@ idx=18",
    )
    msg = failure.format()
    assert msg == (
        "SBarrier missing between SWaitCnt @ idx=18 and consumer "
        "LRA1[0] @ idx=22, needed for producer GRA[0] @ idx=8."
    )


def test_missing_barrier_failure_format_with_capture_brackets():
    """Consumer renders with per-category [N] suffix from its label
    (GRA[1] = second GRA in its category-stream)."""
    failure = MissingBarrierFailure(
        producer=_cms_label("LRA0", 8, name_idx=0),
        consumer=_cms_label("GRA", 12, name_idx=1),
        wait_position="@ idx=10",
    )
    msg = failure.format()
    assert "consumer GRA[1] @ idx=12" in msg


def test_missing_barrier_failure_format_cross_iteration():
    """Producer in loop i, consumer in loop i+1 -> '(of next iteration)' suffix
    on the consumer rendering."""
    failure = MissingBarrierFailure(
        producer=_cms_label("LRA0", 8, name_idx=0),
        consumer=_cms_label("GRA", 2, name_idx=0),
        iter_delta=1,
        wait_position="@ idx=0",
    )
    msg = failure.format()
    assert "consumer GRA[0] @ idx=2 (of next iteration)" in msg


def test_overridden_input_failure_format_pack_pair():
    """Pack pair-leader's vgpr clobbered by an intervening pair-leader of
    the same Pack category."""
    failure = OverriddenInputFailure(
        producer=_cms_label("PackA0", 10, name_idx=0),
        consumer=_cms_label("PackA0", 11, name_idx=1),
        resource="vgpr",
        intervening_writer=_cms_label("PackA0", 12, name_idx=2),
    )
    msg = failure.format()
    assert msg == (
        "PackA0[2] @ idx=12 is incorrectly scheduled between producer "
        "PackA0[0] @ idx=10 and consumer PackA0[1] @ idx=11, clobbering "
        "the vgpr that the consumer needs."
    )


def test_timing_too_close_failure_format_with_capture_brackets():
    """Capture given: producer Pack gets [N]; plain-MFMA consumer omits."""
    failure = TimingTooCloseFailure(
        producer=_cms_label("PackA0", 5, name_idx=0),
        consumer=_cms_label("MFMA", 6),
        expected_quad_cycles=2, actual_quad_cycles=1,
    )
    msg = failure.format()
    assert "PackA0[0] @ idx=5" in msg
    assert "MFMA @ idx=6" in msg
    assert "MFMA[" not in msg
    assert "(of next iteration)" not in msg


def test_timing_too_close_failure_format_cross_iteration():
    """Cross-iteration Pack->MFMA timing: producer Pack in ML-1, consumer
    MFMA in ML. iter_delta=1 selects the "(of next iteration)" suffix."""
    failure = TimingTooCloseFailure(
        producer=_cms_label("PackA0", 7, name_idx=0),
        consumer=_cms_label("MFMA", 0),
        iter_delta=1,
        expected_quad_cycles=2, actual_quad_cycles=1,
    )
    msg = failure.format()
    assert "PackA0[0] @ idx=7" in msg
    assert "MFMA @ idx=0 (of next iteration)" in msg


def test_invalid_counter_value_failure_format_single_bad():
    """Only the field below -1 appears in the message; valid fields (>= -1)
    are omitted so the user sees just what's wrong."""
    failure = InvalidCounterValueFailure(
        swait_position="@ idx=4", dscnt=-2, vlcnt=0, vscnt=-1
    )
    msg = failure.format()
    assert msg == (
        "SWaitCnt @ idx=4 is invalid: dscnt=-2. "
        "All counter fields must be >= -1."
    )


def test_invalid_counter_value_failure_format_multiple_bad():
    """Two fields below -1 -> both listed, comma-separated."""
    failure = InvalidCounterValueFailure(
        swait_position="@ idx=7", dscnt=-2, vlcnt=-1, vscnt=-3
    )
    msg = failure.format()
    assert msg == (
        "SWaitCnt @ idx=7 is invalid: dscnt=-2, vscnt=-3. "
        "All counter fields must be >= -1."
    )


def test_overridden_input_failure_format_scc_clobber():
    """SCC carry-chain clobber: GRIncA[2] writes SCC between GRIncA[1]
    (producer) and GRIncA[3] (consumer)."""
    failure = OverriddenInputFailure(
        producer=_cms_label("GRIncA", 4, name_idx=1),
        consumer=_cms_label("GRIncA", 6, name_idx=3),
        resource="SCC",
        intervening_writer=_cms_label("GRIncA", 5, name_idx=2),
    )
    msg = failure.format()
    assert msg == (
        "GRIncA[2] @ idx=5 is incorrectly scheduled between producer "
        "GRIncA[1] @ idx=4 and consumer GRIncA[3] @ idx=6, clobbering "
        "the SCC that the consumer needs."
    )


# =============================================================================
# Base-class contract
# =============================================================================


def test_failure_base_format_raises():
    """Subclasses must override _format_canonical(); the base raises
    NotImplementedError."""
    f = Failure()
    with pytest.raises(NotImplementedError):
        f.format()


def test_format_works_without_reference_in_scope():
    """Structural test: build a Failure inside a function; in the caller's
    scope where any reference data is unreachable, format() must still
    succeed.

    Post-g4w: format() takes no argument and reads only fields stored on
    the Failure (FailureNodeLabel + scalar fields), so it cannot reach
    out to graphs/captures by construction.
    """
    def _build():
        return MissingWaitFailure(
            producer=_cms_label("LRA0", 5, name_idx=0),
            consumer=_cms_label("MFMA", 10),
            counter_kind="dscnt",
        )

    failure = _build()
    msg = failure.format()
    assert isinstance(msg, str) and msg


# =============================================================================
# New tests covering source-aware label decoupling (g4w)
# =============================================================================


def test_ngl_body_failure_renders_cleanly():
    """An NGL-body failure: previously triggered ValueError in `_node_label`
    when a Failure carrying a node from the NGL body was rendered against
    the ML body capture. Post-g4w: labels are computed eagerly by the
    source-aware provider against the RIGHT body capture, so the rendering
    layer can't touch the wrong body and there's nothing to crash on.

    Witnessed (pre-g4w): gfx950 BBS Ailk MT192x256x64 (ot2.B Run 7).
    """
    failure = MissingWaitFailure(
        # Producer's primary carries the per-category-stream index resolved
        # against the NGL body's instruction list, NOT the ML body's.
        producer=FailureNodeLabel(
            primary="LRA0[0]", position="@ idx=3",
            category="LRA0", body_label="NGL",
        ),
        consumer=FailureNodeLabel(
            primary="MFMA", position="@ idx=5",
            category="MFMA", body_label="NGL",
        ),
        counter_kind="dscnt",
    )
    msg = failure.format()
    # No exception; message is well-formed even though NGL body is involved.
    assert msg == (
        "SWaitCnt(dscnt) missing between producer LRA0[0] @ idx=3 "
        "and consumer MFMA @ idx=5."
    )


def test_cross_body_failure_producer_in_ml_consumer_in_ngl():
    """Cross-body failure: producer in ML, consumer in NGL — both render
    cleanly with their own `primary` strings (each resolved against its
    own body capture by the source-aware provider). Mirrors the production
    shape that `_node_label` could not handle (one body's tagged_inst not
    present in the other's instruction stream)."""
    failure = OrderInvertedFailure(
        producer=FailureNodeLabel(
            primary="LRA0[2]", position="@ idx=8",
            category="LRA0", body_label="ML",
        ),
        consumer=FailureNodeLabel(
            primary="MFMA", position="@ idx=4",
            category="MFMA", body_label="NGL",
        ),
        default_producer_position=SchedulePosition(loop_index=1, stream_index=2),
        default_consumer_position=SchedulePosition(loop_index=2, stream_index=4),
    )
    msg = failure.format()
    assert msg == "Producer LRA0[2] @ idx=8 is issued after consumer MFMA @ idx=4."


def test_non_cms_label_renders_raw_asm_style():
    """Non-CMS source: a future SIA0 (raw asm) label provider would emit
    `(primary='ds_load_b128', position='@ stream_pos=42')`. The format API
    is source-agnostic — it cares only that primary + position are
    strings; the formatter inserts them verbatim into the canonical
    template. This is the SIA0-prerequisite half of the bead's design."""
    failure = MissingWaitFailure(
        producer=FailureNodeLabel(primary="ds_load_b128", position="@ stream_pos=42"),
        consumer=FailureNodeLabel(primary="v_mfma_f32_16x16x16", position="@ stream_pos=58"),
        counter_kind="dscnt",
    )
    msg = failure.format()
    assert msg == (
        "SWaitCnt(dscnt) missing between producer ds_load_b128 @ stream_pos=42 "
        "and consumer v_mfma_f32_16x16x16 @ stream_pos=58."
    )


def test_iter_delta_two_renders_general_suffix():
    """iter_delta > 1: exercises the general 'consumer is N iterations
    after producer' branch of `_iter_suffix` (delta=0 -> empty, delta=1 ->
    '(of next iteration)', else -> general phrasing). The validator's
    compare/validate code currently only produces delta in {0, 1}, but the
    rendering layer is ready for the general case."""
    failure = MissingWaitFailure(
        producer=_cms_label("LRA0", 5, name_idx=0),
        consumer=_cms_label("MFMA", 10),
        iter_delta=2,
        counter_kind="dscnt",
    )
    msg = failure.format()
    assert msg == (
        "SWaitCnt(dscnt) missing between producer LRA0[0] @ idx=5 and consumer "
        "MFMA @ idx=10 (consumer is 2 iterations after producer)."
    )
