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

from dataclasses import dataclass
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


# =============================================================================
# CmsLabelRenderer — CMS-side `TaggedInstructionLike` implementation
# =============================================================================
# Wraps a CMS `TaggedInstruction` plus the body-context-derived per-category
# stream index N so `render()` can return `category[N]` without needing to
# re-scan the body at render time. Body context is captured exactly once at
# wrap time (either by the bridge here, or by `cms_node_label` on the
# legacy graph-only path) and frozen onto the wrapper.
#
# The wrapper holds a reference to the original `TaggedInstruction` so
# downstream consumers that need the raw underlying state (e.g.
# `wrapped.rocisa_inst`, `slot`) can still reach it via `tagged_inst`.
@dataclass(frozen=True)
class CmsLabelRenderer:
    """CMS-side renderer satisfying `TaggedInstructionLike`.

    Holds a reference to the underlying `TaggedInstruction` plus the
    pre-computed per-category-stream index `name_idx` (the `[N]` in
    `LRA0[N]`). Both rendering surfaces are pure reads off these fields
    plus `tagged_inst.slot.mfma_index`.

    Constructed at exactly two sites:
      * `cms_capture_to_timeline._body_to_timeline_body` — for events
        emitted into a `TimelineBody` (the new path).
      * `CMSValidator.cms_node_label` — for `FailureNodeLabel` construction
        on the legacy `(GraphNode, LoopBodyCapture)` path until that path
        is rerouted onto Timeline by sub-bead `rocm-libraries-iig`.
    Both sites compute `name_idx` by scanning the body's instructions for
    same-category entries and finding this `tagged_inst`'s position.

    `render()` returns:
      * `category[name_idx]` for any non-MFMA category.
      * Bare `category` (no `[N]`) for plain MFMA — preserves the existing
        CMS rendering convention (vmfma_index is the canonical identity for
        plain MFMA, so `[N]` is redundant).

    `render_position()` returns `@ idx={tagged_inst.slot.mfma_index}` —
    the kernel-writer's MFMA-slot id (NOT the bridge-collapsed
    `position.stream_index`). This preserves byte-identical output of every
    existing CMS pinning test.
    """
    tagged_inst: TaggedInstruction
    name_idx: int

    def render(self) -> str:
        cat = self.tagged_inst.category
        if cat == "MFMA":
            return cat
        return f"{cat}[{self.name_idx}]"

    def render_position(self) -> str:
        slot = self.tagged_inst.slot
        return f"@ idx={slot.mfma_index}"


def _name_idx_for(
    tagged_inst: TaggedInstruction,
    body_instructions: list,
) -> int:
    """Per-category-stream index of `tagged_inst` within `body_instructions`.

    The `[N]` in `LRA0[N]`: count of same-category TaggedInstructions
    appearing earlier in the body, returned as the 0-based index.

    `tagged_inst` is required to appear in `body_instructions`; an absence
    would indicate a capture-pipeline bug and is asserted (matches the
    existing `cms_node_label` invariant — every GraphNode is constructed
    from the body it indexes).
    """
    cat = tagged_inst.category
    same_cat = [t for t in body_instructions if getattr(t, "category", None) == cat]
    assert tagged_inst in same_cat, (
        f"_name_idx_for: tagged_inst not found in body for category {cat!r}. "
        f"Every event/node must originate from the body it indexes."
    )
    return same_cat.index(tagged_inst)


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
        # Wrap in CmsLabelRenderer so the event's `tagged_inst` field
        # satisfies `TaggedInstructionLike` with body-context-derived
        # `[N]` rendering. The renderer holds a back-reference to the
        # original TaggedInstruction so downstream consumers can still
        # reach `wrapped.rocisa_inst`/`slot`/etc. through it.
        name_idx = _name_idx_for(tagged_inst, loop_body.instructions)
        events.append(TimelineEvent(
            rocisa_inst=wrapped.rocisa_inst,
            reads=tuple(wrapped.reads),
            writes=tuple(wrapped.writes),
            read_slots=tuple(wrapped.read_slots),
            write_slots=tuple(wrapped.write_slots),
            category=tagged_inst.category,
            position=make_position(body_label, stream_index),
            body_label=body_label,
            tagged_inst=CmsLabelRenderer(tagged_inst=tagged_inst, name_idx=name_idx),
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


__all__ = ["cms_capture_to_timeline", "CmsLabelRenderer", "_name_idx_for"]
