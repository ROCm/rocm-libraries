################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Scaffold-level fixture for Timeline / TimelineEvent (rocm-libraries-3g4).

These tests demonstrate the Timeline shape works end-to-end as a passive
container: a Timeline can be constructed with TimelineBody children whose
events carry the existing CMS `TaggedInstruction` (which structurally
satisfies the `TaggedInstructionLike` Protocol once `render()` lands in
sub-bead `rocm-libraries-3dy`).

These tests intentionally do NOT exercise the validator — no caller is wired
through Timeline yet (that is sub-bead `rocm-libraries-iig`). They confirm
only the dataclass shape, field accessibility, and the source-agnostic
ordering invariant.
"""

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    SchedulePosition,
    make_position,
)
from Tensile.Components.Timeline import (
    Timeline,
    TimelineBody,
    TimelineEvent,
)

from Tensile.Tests.unit.dataflow_fixtures import make_capture, make_mfma


# A minimal stand-in implementing the Protocol surface — used to confirm the
# Protocol accepts arbitrary structural conformance, not only the CMS
# `TaggedInstruction` shape.
class _StubTagged:
    def render(self) -> str:
        return "stub"


def _build_event_from_tagged(tagged_inst, body_label: str, stream_index: int
                             ) -> TimelineEvent:
    """Lift a CMS `TaggedInstruction` into a `TimelineEvent`.

    This is the minimal field-by-field copy the real
    `cms_capture_to_timeline` bridge (sub-bead `rocm-libraries-xe5`) will
    perform — kept in-test here so this scaffold doesn't pre-empt the
    bridge bead.
    """
    wrapped = tagged_inst.wrapped
    return TimelineEvent(
        rocisa_inst=wrapped.rocisa_inst,
        reads=tuple(wrapped.reads),
        writes=tuple(wrapped.writes),
        read_slots=tuple(wrapped.read_slots),
        write_slots=tuple(wrapped.write_slots),
        category=tagged_inst.category,
        position=make_position(body_label, stream_index),
        body_label=body_label,
        tagged_inst=tagged_inst,
    )


def test_timeline_scaffold_round_trip_one_body():
    """Construct a one-body Timeline from CMS TaggedInstructions and confirm
    every consumer-visible field round-trips through the dataclass shape."""
    mfma1 = make_mfma(c_dst_start=200, a_src_start=204, b_src_start=208,
                      slot=0, sequence=0)
    mfma2 = make_mfma(c_dst_start=300, a_src_start=304, b_src_start=308,
                      slot=1, sequence=0)

    # Run the CMS-side `make_capture` so `_populate_wrapper` fills out the
    # WrappedInstruction reads/writes/slot tuples — that is the producer-side
    # state the bridge will copy across.
    cap = make_capture(BODY_LABEL_ML, [mfma1, mfma2])

    events = [
        _build_event_from_tagged(ti, BODY_LABEL_ML, idx)
        for idx, ti in enumerate(cap.instructions)
    ]
    body = TimelineBody(events=events)
    timeline = Timeline(
        events_by_body={BODY_LABEL_ML: body},
        arch_profile=None,
        num_mfma_per_subiter=0,
    )

    # Top-level shape.
    assert list(timeline.events_by_body.keys()) == [BODY_LABEL_ML]
    assert timeline.arch_profile is None
    assert timeline.num_mfma_per_subiter == 0

    # Per-body shape.
    body_round_trip = timeline.events_by_body[BODY_LABEL_ML]
    assert isinstance(body_round_trip, TimelineBody)
    assert body_round_trip.name_to_idx == {}
    assert len(body_round_trip.events) == 2

    # Per-event field round-trip — every field from the spike's §1.7
    # catalogue is reachable, has the expected type, and matches the
    # producer-side TaggedInstruction.
    for event, source_ti in zip(body_round_trip.events, cap.instructions):
        assert isinstance(event, TimelineEvent)
        assert event.rocisa_inst is source_ti.wrapped.rocisa_inst
        assert event.category == source_ti.category
        assert event.body_label == BODY_LABEL_ML
        assert event.tagged_inst is source_ti
        # reads/writes/slots are tuples (not lists) — required by the
        # validator's edge-formation pipeline.
        assert isinstance(event.reads, tuple)
        assert isinstance(event.writes, tuple)
        assert isinstance(event.read_slots, tuple)
        assert isinstance(event.write_slots, tuple)
        assert isinstance(event.position, SchedulePosition)


def test_timeline_event_satisfies_protocol_via_tagged_instruction():
    """The CMS `TaggedInstruction` will satisfy `TaggedInstructionLike` once
    `rocm-libraries-3dy` lands `render()`. Until then we confirm Protocol
    structural typing accepts any object with a `render()` callable so the
    bridge has the freedom to pick its carrier."""
    stub = _StubTagged()
    # Protocols don't enforce anything at construction time — the field is
    # just stored. The runtime guarantee is: anything passed in here that
    # has `render()` will work for downstream callers.
    event = TimelineEvent(
        rocisa_inst=object(),
        reads=(),
        writes=(),
        read_slots=(),
        write_slots=(),
        category="MFMA",
        position=SchedulePosition(loop_index=1, stream_index=0),
        body_label=BODY_LABEL_ML,
        tagged_inst=stub,
    )
    assert event.tagged_inst.render() == "stub"
    # And the Protocol matches structurally — `isinstance` against a
    # non-`runtime_checkable` Protocol is a no-go, but the type checker
    # call site is what matters; we exercise the surface here.
    assert hasattr(event.tagged_inst, "render")


def test_timeline_position_lex_order_across_bodies():
    """`SchedulePosition` lex-sort over `(loop_index, stream_index)` gives
    global stream order across bodies. The Timeline shape relies on this for
    the graph builder's cross-body walks."""
    pos_ml_prev_first = make_position(BODY_LABEL_ML_PREV, 0)
    pos_ml_first = make_position(BODY_LABEL_ML, 0)
    pos_ml_second = make_position(BODY_LABEL_ML, 1)

    assert pos_ml_prev_first < pos_ml_first
    assert pos_ml_first < pos_ml_second
    # And constructable both ways without surprises.
    assert pos_ml_first == SchedulePosition(loop_index=1, stream_index=0)


def test_timeline_body_default_name_to_idx_is_empty_dict():
    """`name_to_idx` defaults to `{}` so asm-source bridges (which leave it
    empty) don't have to construct a placeholder."""
    body = TimelineBody(events=[])
    assert body.name_to_idx == {}
    # Independent default per-instance — `field(default_factory=dict)` not a
    # shared mutable.
    other = TimelineBody(events=[])
    other.name_to_idx["foo"] = 1
    assert body.name_to_idx == {}
