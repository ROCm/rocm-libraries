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

"""CMS -> Timeline bridge (rocm-libraries-xe5, sub-bead of 5gd Part A).

`cms_capture_to_timeline` walks a `FourPartCapture` and emits a
source-agnostic `Timeline` per the shape defined in
`Tensile.Components.Timeline` (rocm-libraries-3g4) and the field-by-field
catalogue in `TIMELINE_INTERFACE_SPIKE.md` §4.2.

This module is the ONLY place in the timeline path that imports
CMS-specific shapes (`FourPartCapture`, `LoopBodyCapture`,
`TaggedInstruction`). `Timeline.py` itself stays source-agnostic; the
asm-side bridge (`assembly_to_timeline`, sub-bead `rocm-libraries-x6s`)
will live in its own module symmetrically.

Discipline (from the parent bead):
  * No fallback paths. Required fields are required.
  * No parallel APIs. One bridge function.
  * No re-computation of producer-side state — the bridge copies
    pre-populated `WrappedInstruction.{reads, writes, read_slots,
    write_slots}` tuples and `assign_stream_indices_for_body`'s stream
    indices verbatim.
"""

from __future__ import annotations

from typing import Dict, List

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    BODY_LABEL_ML_PREV,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
    FourPartCapture,
    LoopBodyCapture,
    TaggedInstruction,
    assign_stream_indices_for_body,
    make_position,
)
from Tensile.Components.Timeline import (
    Timeline,
    TimelineBody,
    TimelineEvent,
)


# Canonical body build order, matching CMSValidator._BODY_BUILD_ORDER. Kept
# inline here (not imported) so this module has zero dependency on
# CMSValidator — the bridge sits BELOW the validator in the layering.
_BODY_BUILD_ORDER = (
    BODY_LABEL_ML_PREV,
    BODY_LABEL_ML,
    BODY_LABEL_NGL,
    BODY_LABEL_NLL,
)


def _body_to_timeline_body(
    loop_body: LoopBodyCapture,
    body_label: str,
) -> TimelineBody:
    """Lift one CMS `LoopBodyCapture` into a `TimelineBody`.

    Stream indices are assigned via `assign_stream_indices_for_body`,
    which lex-sorts the body's TaggedInstructions on
    `(slot.mfma_index, slot.sequence)` and assigns 0, 1, 2, ... in that
    order. The TimelineEvents are emitted in the same order — i.e. the
    body's natural `instructions` order is preserved when (as in
    production) it already matches the lex sort. Test fixtures with
    out-of-order `instructions` get the canonical assignment.
    """
    stream_index_by_id = assign_stream_indices_for_body(loop_body.instructions)
    events: List[TimelineEvent] = []
    for tagged_inst in loop_body.instructions:
        wrapped = tagged_inst.wrapped
        stream_index = stream_index_by_id[id(tagged_inst)]
        events.append(TimelineEvent(
            rocisa_inst=wrapped.rocisa_inst,
            reads=tuple(wrapped.reads),
            writes=tuple(wrapped.writes),
            read_slots=tuple(wrapped.read_slots),
            write_slots=tuple(wrapped.write_slots),
            category=tagged_inst.category,
            position=make_position(body_label, stream_index),
            body_label=body_label,
            tagged_inst=tagged_inst,
        ))
    return TimelineBody(
        events=events,
        name_to_idx=dict(loop_body.name_to_idx),
    )


def cms_capture_to_timeline(capture: FourPartCapture) -> Timeline:
    """Bridge a CMS `FourPartCapture` into a source-agnostic `Timeline`.

    Walks the four bodies (ML-1, ML, NGL, NLL) in `_BODY_BUILD_ORDER`,
    pulling each body's `{0: LoopBodyCapture}` codepath-0 entry (the
    consumer never reads other codepaths — see
    `TIMELINE_INTERFACE_SPIKE.md` §1.1). Each body becomes a
    `TimelineBody`; absent bodies (no codepath 0) are skipped, matching
    `build_dataflow_graph`'s tolerance for partial captures
    (CMSValidator.py:941).

    Per-event field mapping (per spike §4.2):
      * `rocisa_inst`     <- `tagged_inst.wrapped.rocisa_inst`
      * `reads`/`writes`  <- `tagged_inst.wrapped.reads`/`.writes` (as tuples)
      * `read_slots`      <- `tagged_inst.wrapped.read_slots` (as tuple)
      * `write_slots`     <- `tagged_inst.wrapped.write_slots` (as tuple)
      * `category`        <- `tagged_inst.category`
      * `position`        <- `make_position(body_label, stream_index)`,
                             where `stream_index` comes from
                             `assign_stream_indices_for_body` for the
                             body
      * `body_label`      <- the body's label key (lifted onto the event,
                             implicit on `LoopBodyCapture` today)
      * `tagged_inst`     <- the original CMS `TaggedInstruction`
                             (satisfies `TaggedInstructionLike` via its
                             newly-added `render()` method)

    Per-body field mapping:
      * `events`          <- per-event list above, in
                             `assign_stream_indices_for_body` order
      * `name_to_idx`     <- shallow-copied from `LoopBodyCapture`

    Per-Timeline field mapping:
      * `events_by_body`  <- `{body_label: TimelineBody, ...}` for every
                             body present in `capture`
      * `arch_profile`    <- `capture.arch_profile`
      * `num_mfma_per_subiter` <- `capture.num_mfma_per_subiter`
    """
    body_sources = (
        (BODY_LABEL_ML_PREV, capture.main_loop_prev),
        (BODY_LABEL_ML,      capture.main_loop),
        (BODY_LABEL_NGL,     capture.n_gl),
        (BODY_LABEL_NLL,     capture.n_ll),
    )
    bodies: Dict[str, TimelineBody] = {}
    for body_label, by_cp in body_sources:
        if 0 not in by_cp:
            continue
        bodies[body_label] = _body_to_timeline_body(by_cp[0], body_label)
    return Timeline(
        events_by_body=bodies,
        arch_profile=capture.arch_profile,
        num_mfma_per_subiter=capture.num_mfma_per_subiter,
    )


__all__ = ["cms_capture_to_timeline"]
