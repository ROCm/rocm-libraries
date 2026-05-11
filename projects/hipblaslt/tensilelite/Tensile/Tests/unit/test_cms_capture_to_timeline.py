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

"""Equivalence tests for `cms_capture_to_timeline` (rocm-libraries-xe5).

The bridge takes a CMS `FourPartCapture` and emits a source-agnostic
`Timeline` per the shape from `rocm-libraries-3g4`. Until
`rocm-libraries-iig` reroutes the validator entry point onto `Timeline`,
the validator does NOT consume `Timeline` — so equivalence is necessarily
indirect: we round-trip the bridge against the original CMS capture and
assert every consumer-visible field matches the source.

The round-trip pins the bridge as a faithful translator: anything the
validator's graph builder + edge-coverage logic would read off the
TaggedInstruction, the bridge must place on the corresponding
TimelineEvent at the same value.
"""

from typing import List, Tuple

import pytest

from Tensile.Components.cms_to_timeline import cms_capture_to_timeline
from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    BODY_LABEL_TO_LOOP_INDEX,
    FourPartCapture,
    LoopBodyCapture,
    SchedulePosition,
    TaggedInstruction,
    assign_stream_indices_for_body,
)
from Tensile.Components.Timeline import (
    Timeline,
    TimelineBody,
    TimelineEvent,
)

from Tensile.Tests.unit.dataflow_fixtures import (
    make_capture,
    make_lr,
    make_lcc_pair,
    make_mfma,
    make_sbarrier,
    make_snop,
    make_swait,
)


# =============================================================================
# FourPartCapture builders — mirror the in-test `_wrap` helpers used by the
# existing dataflow-graph fixture suites.
# =============================================================================
# Filler MFMAs use distinct high-vgpr ranges per body so they cannot
# collide with the test's own register ranges. The values here are
# arbitrary-but-distinct; the bridge does not interpret them.
_FILLER_RANGES = {
    BODY_LABEL_ML_PREV: (200, 204, 208),
    BODY_LABEL_ML:      (210, 214, 218),
    BODY_LABEL_NGL:     (220, 224, 228),
    BODY_LABEL_NLL:     (240, 244, 248),
}


def _filler(body_label: str) -> LoopBodyCapture:
    c, a, b = _FILLER_RANGES[body_label]
    return make_capture(body_label, [make_mfma(
        c_dst_start=c, a_src_start=a, b_src_start=b, slot=0,
    )])


def _wrap_four(ml_capture: LoopBodyCapture, *,
               ml_prev: LoopBodyCapture = None,
               ngl: LoopBodyCapture = None,
               nll: LoopBodyCapture = None,
               num_mfma_per_subiter: int = 0,
               arch_profile: object = None,
               num_mfma: int = 1,
               num_codepaths: int = 1,
               source: str = "cms") -> FourPartCapture:
    return FourPartCapture(
        main_loop={0: ml_capture},
        main_loop_prev={0: ml_prev if ml_prev is not None else _filler(BODY_LABEL_ML_PREV)},
        n_gl={0: ngl if ngl is not None else _filler(BODY_LABEL_NGL)},
        n_ll={0: nll if nll is not None else _filler(BODY_LABEL_NLL)},
        num_mfma=num_mfma,
        num_codepaths=num_codepaths,
        source=source,
        num_mfma_per_subiter=num_mfma_per_subiter,
        arch_profile=arch_profile,
    )


# =============================================================================
# Round-trip equivalence checker
# =============================================================================


def _all_bodies_in_capture(capture: FourPartCapture) -> List[Tuple[str, LoopBodyCapture]]:
    """Return `(body_label, LoopBodyCapture)` pairs for every body present
    in `capture` at codepath 0, in `_BODY_BUILD_ORDER`."""
    sources = (
        (BODY_LABEL_ML_PREV, capture.main_loop_prev),
        (BODY_LABEL_ML,      capture.main_loop),
        (BODY_LABEL_NGL,     capture.n_gl),
        (BODY_LABEL_NLL,     capture.n_ll),
    )
    out: List[Tuple[str, LoopBodyCapture]] = []
    for label, by_cp in sources:
        if 0 in by_cp:
            out.append((label, by_cp[0]))
    return out


def _assert_equivalent(capture: FourPartCapture, timeline: Timeline) -> None:
    """Assert that every consumer-visible field carried by the capture is
    reachable verbatim on the timeline."""
    # Top-level kernel-wide knobs.
    assert timeline.arch_profile is capture.arch_profile, (
        "arch_profile must round-trip identity-equal — the validator reads "
        "it via getattr(four_part_capture, 'arch_profile', None) and threads "
        "it into DataflowGraph.arch_profile."
    )
    assert timeline.num_mfma_per_subiter == capture.num_mfma_per_subiter

    # Body set: same labels, same order via the shared _BODY_BUILD_ORDER.
    bodies = _all_bodies_in_capture(capture)
    assert list(timeline.events_by_body.keys()) == [label for label, _ in bodies]

    # Per-body / per-event equivalence.
    for body_label, loop_body in bodies:
        tb = timeline.events_by_body[body_label]
        assert isinstance(tb, TimelineBody)
        assert tb.name_to_idx == loop_body.name_to_idx
        # name_to_idx is shallow-copied — mutating the timeline's copy must
        # not bleed back into the source.
        if loop_body.name_to_idx:
            assert tb.name_to_idx is not loop_body.name_to_idx

        # Stream-index assignment: must match the canonical
        # `assign_stream_indices_for_body` derivation.
        expected_stream = assign_stream_indices_for_body(loop_body.instructions)

        # The events list is in stream-index order (== natural append order
        # for production-built bodies). Build a parallel TaggedInstruction
        # sequence sorted by stream_index for comparison.
        expected_tis = sorted(
            loop_body.instructions,
            key=lambda ti: expected_stream[id(ti)],
        )
        assert len(tb.events) == len(expected_tis), (
            f"Body {body_label!r} event count mismatch: "
            f"{len(tb.events)} != {len(expected_tis)}"
        )

        for event, ti in zip(tb.events, expected_tis):
            _assert_event_matches_tagged(event, ti, body_label,
                                         expected_stream[id(ti)])


def _assert_event_matches_tagged(event: TimelineEvent,
                                 ti: TaggedInstruction,
                                 body_label: str,
                                 stream_index: int) -> None:
    from Tensile.Components.cms_to_timeline import CmsLabelRenderer
    assert isinstance(event, TimelineEvent)
    # Identity preservation — the validator currently reaches the rocisa
    # instance via `node.tagged_inst.wrapped.rocisa_inst`; the bridge
    # promises the timeline's `event.rocisa_inst` IS that same instance.
    assert event.rocisa_inst is ti.wrapped.rocisa_inst
    # Post-3dy: the bridge wraps the original CMS TaggedInstruction in a
    # CmsLabelRenderer that satisfies `TaggedInstructionLike` with the
    # body-context-derived `[N]` rendering. The original is preserved as
    # the wrapper's `tagged_inst` field (so downstream consumers can still
    # reach `wrapped.rocisa_inst`/`slot`/etc. through it).
    assert isinstance(event.tagged_inst, CmsLabelRenderer)
    assert event.tagged_inst.tagged_inst is ti

    # Operand tuples — must be tuples (not lists) per Timeline contract.
    assert isinstance(event.reads, tuple)
    assert isinstance(event.writes, tuple)
    assert isinstance(event.read_slots, tuple)
    assert isinstance(event.write_slots, tuple)
    # Element-wise equality with the source wrapper. Resources are typically
    # RegisterContainer / MemoryRegion instances; identity is the strict
    # contract since the bridge does not deep-copy.
    assert event.reads == tuple(ti.wrapped.reads)
    assert event.writes == tuple(ti.wrapped.writes)
    assert event.read_slots == tuple(ti.wrapped.read_slots)
    assert event.write_slots == tuple(ti.wrapped.write_slots)

    # Categorical & positional fields.
    assert event.category == ti.category
    assert event.body_label == body_label
    assert isinstance(event.position, SchedulePosition)
    assert event.position.loop_index == BODY_LABEL_TO_LOOP_INDEX[body_label]
    assert event.position.stream_index == stream_index

    # CmsLabelRenderer satisfies the TaggedInstructionLike Protocol with
    # both rendering surfaces. `render()` returns `category[N]` (or bare
    # `MFMA` for plain MFMA); `render_position()` returns
    # `@ idx={vmfma_index}`.
    assert hasattr(event.tagged_inst, "render")
    assert hasattr(event.tagged_inst, "render_position")
    rendered = event.tagged_inst.render()
    assert isinstance(rendered, str)
    if ti.category == "MFMA":
        assert rendered == "MFMA"
    else:
        assert rendered == f"{ti.category}[{event.tagged_inst.name_idx}]"
    rendered_pos = event.tagged_inst.render_position()
    assert isinstance(rendered_pos, str)
    assert rendered_pos == f"@ idx={ti.slot.mfma_index}"


# =============================================================================
# Equivalence tests across a representative set of CMS-fixture shapes
# =============================================================================


def test_round_trip_single_mfma_main_loop():
    """Minimal capture: one MFMA in ML, fillers everywhere else."""
    ml = make_capture(BODY_LABEL_ML, [
        make_mfma(c_dst_start=0, a_src_start=4, b_src_start=8, slot=0),
    ])
    cap = _wrap_four(ml)
    timeline = cms_capture_to_timeline(cap)
    _assert_equivalent(cap, timeline)


def test_round_trip_lr_then_mfma_pair():
    """LR-then-MFMA: the canonical producer-consumer pair the validator
    forms a Phase-2 read edge for. Stream order must be preserved."""
    ml = make_capture(BODY_LABEL_ML, [
        make_lr(dst_vgpr_start=0, dst_vgpr_count=4, lds_offset=0, slot=0,
                category="LRA0", sequence=0),
        make_mfma(c_dst_start=100, a_src_start=0, b_src_start=8, slot=0,
                  sequence=1),
    ])
    cap = _wrap_four(ml)
    timeline = cms_capture_to_timeline(cap)
    _assert_equivalent(cap, timeline)

    # Sanity: stream_index runs 0, 1 in append order.
    events = timeline.events_by_body[BODY_LABEL_ML].events
    assert [e.position.stream_index for e in events] == [0, 1]
    assert [e.category for e in events] == ["LRA0", "MFMA"]


def test_round_trip_lcc_pair_in_main_loop():
    """LCC pair — two TaggedInstructions at the same `mfma_index` with
    consecutive `sequence` values. Tie-breaking is folded into
    stream_index by the bridge, so the two events get distinct
    `stream_index` 0 and 1."""
    lcc_a, lcc_b = make_lcc_pair(slot=0)
    ml = make_capture(BODY_LABEL_ML, [lcc_a, lcc_b])
    cap = _wrap_four(ml)
    timeline = cms_capture_to_timeline(cap)
    _assert_equivalent(cap, timeline)

    events = timeline.events_by_body[BODY_LABEL_ML].events
    assert len(events) == 2
    assert events[0].position.stream_index == 0
    assert events[1].position.stream_index == 1
    assert events[0].position.loop_index == events[1].position.loop_index


def test_round_trip_swait_and_sbarrier_carried_through():
    """SWait/SBarrier/SNop instructions are scheduler-choice (not graph
    nodes) but still appear in `LoopBodyCapture.instructions` — the
    bridge must carry them through verbatim so cross-body wait/barrier
    walkers can find them in stream order."""
    ml = make_capture(BODY_LABEL_ML, [
        make_lr(dst_vgpr_start=0, dst_vgpr_count=4, lds_offset=0, slot=0,
                sequence=0),
        make_swait(slot=0, vlcnt=0, sequence=1),
        make_sbarrier(slot=0, sequence=2),
        make_snop(slot=0, wait_state=2, sequence=3),
        make_mfma(c_dst_start=100, a_src_start=0, b_src_start=8, slot=0,
                  sequence=4),
    ])
    cap = _wrap_four(ml)
    timeline = cms_capture_to_timeline(cap)
    _assert_equivalent(cap, timeline)

    cats = [e.category for e in timeline.events_by_body[BODY_LABEL_ML].events]
    assert cats == ["LRA0", "SYNC", "BARRIER", "SNOP", "MFMA"]


def test_round_trip_all_four_bodies_populated():
    """All four bodies populated with non-filler captures — confirms the
    bridge walks every body and assigns body-local stream indices
    independently."""
    ml_prev = make_capture(BODY_LABEL_ML_PREV, [
        make_mfma(c_dst_start=0, a_src_start=4, b_src_start=8, slot=0),
    ])
    ml = make_capture(BODY_LABEL_ML, [
        make_lr(dst_vgpr_start=20, dst_vgpr_count=4, lds_offset=0, slot=0,
                sequence=0),
        make_mfma(c_dst_start=100, a_src_start=20, b_src_start=24, slot=0,
                  sequence=1),
    ])
    ngl = make_capture(BODY_LABEL_NGL, [
        make_mfma(c_dst_start=120, a_src_start=124, b_src_start=128, slot=0),
    ])
    nll = make_capture(BODY_LABEL_NLL, [
        make_mfma(c_dst_start=140, a_src_start=144, b_src_start=148, slot=0),
    ])
    cap = _wrap_four(ml, ml_prev=ml_prev, ngl=ngl, nll=nll)
    timeline = cms_capture_to_timeline(cap)
    _assert_equivalent(cap, timeline)

    # All four bodies present, in canonical order.
    assert list(timeline.events_by_body.keys()) == [
        BODY_LABEL_ML_PREV, BODY_LABEL_ML, BODY_LABEL_NGL, BODY_LABEL_NLL,
    ]
    # Cross-body lex order: ML-1's stream_index 0 < ML's stream_index 0 etc.
    pos_ml_prev = timeline.events_by_body[BODY_LABEL_ML_PREV].events[0].position
    pos_ml = timeline.events_by_body[BODY_LABEL_ML].events[0].position
    pos_ngl = timeline.events_by_body[BODY_LABEL_NGL].events[0].position
    pos_nll = timeline.events_by_body[BODY_LABEL_NLL].events[0].position
    assert pos_ml_prev < pos_ml < pos_ngl < pos_nll


def test_round_trip_arch_profile_and_num_mfma_per_subiter():
    """Kernel-wide knobs propagate to the Timeline."""
    ml = make_capture(BODY_LABEL_ML, [
        make_mfma(c_dst_start=0, a_src_start=4, b_src_start=8, slot=0),
    ])
    sentinel = object()
    cap = _wrap_four(ml, num_mfma_per_subiter=4, arch_profile=sentinel)
    timeline = cms_capture_to_timeline(cap)
    _assert_equivalent(cap, timeline)
    assert timeline.arch_profile is sentinel
    assert timeline.num_mfma_per_subiter == 4


def test_round_trip_name_to_idx_carried_and_independent():
    """`LoopBodyCapture.name_to_idx` round-trips via shallow-copy: equal
    contents, but mutating the timeline's copy doesn't bleed back."""
    ml = make_capture(BODY_LABEL_ML, [
        make_mfma(c_dst_start=0, a_src_start=4, b_src_start=8, slot=0),
    ])
    ml.name_to_idx = {"ValuA_X0_I0": 76, "ValuB_X0_I0": 80}
    cap = _wrap_four(ml)
    timeline = cms_capture_to_timeline(cap)
    _assert_equivalent(cap, timeline)

    tb = timeline.events_by_body[BODY_LABEL_ML]
    tb.name_to_idx["scratch"] = 999
    assert "scratch" not in ml.name_to_idx


def test_round_trip_absent_body_skipped():
    """A FourPartCapture missing codepath 0 on one body — the bridge
    skips it, matching `build_dataflow_graph`'s tolerance for partial
    captures."""
    ml = make_capture(BODY_LABEL_ML, [
        make_mfma(c_dst_start=0, a_src_start=4, b_src_start=8, slot=0),
    ])
    cap = _wrap_four(ml)
    # Drop ML-1.
    cap.main_loop_prev = {}
    timeline = cms_capture_to_timeline(cap)
    _assert_equivalent(cap, timeline)
    assert BODY_LABEL_ML_PREV not in timeline.events_by_body
    assert BODY_LABEL_ML in timeline.events_by_body


def test_tagged_instruction_render_method_exists_and_is_string():
    """The bridge's Protocol contract requires `tagged_inst.render() ->
    str`. The CMS-side TaggedInstruction now implements render() returning
    its category — verify it works on a freshly built TaggedInstruction."""
    ti = make_mfma(c_dst_start=0, a_src_start=4, b_src_start=8, slot=0,
                   category="MFMA")
    rendered = ti.render()
    assert isinstance(rendered, str)
    assert rendered == "MFMA"


def test_timeline_module_does_not_import_cms_specific_shapes():
    """`Tensile.Components.Timeline` must NOT import LoopBodyCapture,
    FourPartCapture, or TaggedInstruction. Only the bridge module is
    allowed to bridge CMS-specific types into the timeline path."""
    import Tensile.Components.Timeline as tl_mod
    # SchedulePosition is generic-position infra (not CMS-source-specific)
    # and is allowed.
    forbidden = ("LoopBodyCapture", "FourPartCapture", "TaggedInstruction",
                 "WrappedInstruction", "SlotKey")
    for name in forbidden:
        assert not hasattr(tl_mod, name), (
            f"Timeline.py must not expose CMS-specific name {name!r}; "
            "only the bridge module (cms_to_timeline.py) is allowed to "
            "import CMS-specific shapes."
        )
