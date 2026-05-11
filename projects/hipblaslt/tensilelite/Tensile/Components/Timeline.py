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

"""
Source-agnostic Timeline / TimelineEvent input shape for the validator
(rocm-libraries-3g4, 5gd Part A scaffold).

This module is the consumer-facing boundary of the generalized validator. It
defines exactly the shape that `build_dataflow_graph` and the edge-coverage
logic consume — no more, no less. Source-side bridges populate it:

  * `cms_capture_to_timeline` (sub-bead `rocm-libraries-xe5`) walks a
    `FourPartCapture` and emits a `Timeline`.
  * `assembly_to_timeline` (sub-bead `rocm-libraries-x6s`) walks a rocisa
    asm dump and emits a `Timeline`.
  * The validator entry point is rerouted onto `Timeline` by sub-bead
    `rocm-libraries-iig`. Until that bead lands, this module is unwired:
    it sits alongside the existing `FourPartCapture` path, not in front
    of it.

Field shape and rationale are catalogued in
`TIMELINE_INTERFACE_SPIKE.md` §2 and §3 (cil's spike, with the user's
`SchedulePosition` collapse from `rocm-libraries-5v4u` already applied).
This module is the implementation of those sections; it deliberately does
NOT reproduce the catalogue here.

Discipline (from the parent bead):
  * No fallback paths. Every consumed field per the spike's §1.7 catalogue
    is required — no `Optional` defaults except where the spike itself
    documented one.
  * No parallel APIs. Each Timeline-related symbol has one home (this
    file). Bridges import; they do not redeclare.
  * The producer-side `WrappedInstruction` proxy collapses at the consumer
    boundary: `TimelineEvent` owns `rocisa_inst` / `reads` / `writes` /
    `read_slots` / `write_slots` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Tuple

from Tensile.Components.ScheduleCapture import SchedulePosition


# =============================================================================
# TaggedInstructionLike — Protocol for the source-aware label carrier
# =============================================================================
# Both the existing CMS `TaggedInstruction` (from ScheduleCapture) and a
# future asm-side wrapper must satisfy this Protocol structurally. The only
# guaranteed surface today is `render() -> str`: rendering the instruction
# for a failure label. The CMS-side `render()` is implemented in sub-bead
# `rocm-libraries-3dy` (5gd.B.1, source-aware label dispatch); the asm-side
# wrapper supplies its own.
#
# Why a Protocol and not a base class:
#   * The CMS `TaggedInstruction` is a `@dataclass` with structural fields
#     (`wrapped`, `category`, `slot`) that the asm-side wrapper has no
#     reason to mirror — only the `render()` callable is shared.
#   * `typing.Protocol` does structural-subtype checking, so neither side
#     has to inherit from this interface.
#
# Note: `render()` is the only method the validator's failure-rendering
# pipeline calls on the carrier today. Adding more requirements here is a
# decision for the bead that introduces the new caller, not a speculative
# extension here.
class TaggedInstructionLike(Protocol):
    """Source-agnostic carrier for the per-event source-aware label.

    Both `Tensile.Components.ScheduleCapture.TaggedInstruction` (CMS source,
    via `Tensile.Components.cms_to_timeline.CmsLabelRenderer` wrapper that
    captures body context) and the asm-side wrapper
    (`Tensile.Components.asm_to_timeline_renderers.AsmLabelRenderer`)
    satisfy this Protocol structurally.

    `render()` and `render_position()` are the two and only two surfaces the
    failure-rendering pipeline calls on the carrier. They return strings;
    the formatter is pure string composition over those returned strings
    plus per-Failure scalar fields. By construction the formatter has no
    knowledge of which source emitted the timeline — a CMS-source carrier
    and an asm-source carrier produce the same `(primary, position)` pair
    shape via different rendering bodies.

    Why both surfaces live on the same carrier (vs. a separate
    `PositionRenderer` object): both rendering surfaces are functions of
    source-side state that the carrier already owns (CMS slot.mfma_index;
    asm-stream line number). Splitting them across two objects forces the
    caller to thread parallel objects per event and re-establish their
    pairing — pure overhead with no decoupling benefit, since neither
    renderer makes sense in isolation from the other.
    """

    def render(self) -> str:
        """Render the underlying instruction for a failure label.

        CMS source: returns `category[N]` where N is the per-category-stream
        index in the body that emitted this event (e.g. `LRA0[3]`). Plain
        `MFMA` omits the `[N]` suffix.

        Asm source: returns `mnemonic + operands`, the rocisa-canonical
        render string (e.g. `ds_load_b128 v[0:3], v255 offset:0`).
        """
        ...

    def render_position(self) -> str:
        """Render the source-native position string for this event.

        CMS source: returns `@ idx={vmfma_index}` where vmfma_index is the
        kernel-writer's MFMA-slot id (recovered from
        `tagged_inst.slot.mfma_index`). Existing CMS pinning tests pin this
        wording byte-identically.

        Asm source: returns `@ asm_line={line_number}` (or whichever
        asm-native location the asm bridge supplies). Asm has no MFMA-slot
        concept; faking `@ idx=N` for asm would lie in failure messages,
        so the asm carrier returns its native position shape and the
        formatter inserts whatever the carrier produced.
        """
        ...


# =============================================================================
# TimelineEvent — one scheduled instruction, source-agnostic
# =============================================================================
# Field catalogue and rationale: TIMELINE_INTERFACE_SPIKE.md §2.
#
# All four operand-tuple fields are tuples (not lists): the validator's
# byte-key resolver and edge-formation pipeline iterate them and never
# mutate them, and the producer side already produces tuples
# (`WrappedInstruction.{reads, writes, read_slots, write_slots}`).
#
# `rocisa_inst` is `object` because the validator only inspects it via
# `_class_tag` / `_canonical_render` / a few rocisa `getattr` probes —
# the type is not statically checkable here without dragging in a hard
# rocisa import for every Timeline consumer.
@dataclass
class TimelineEvent:
    """One scheduled instruction in source-agnostic form.

    Carries exactly what the validator's graph builder + edge-coverage logic
    consumes today (TIMELINE_INTERFACE_SPIKE.md §1.7). No fields beyond
    that catalogue.

    Producer-side state (`SlotKey.subiter`, `SlotKey.slot_kind`,
    `WrappedInstruction` proxy, `FourPartCapture.num_mfma` /
    `num_codepaths` / `source`) does NOT appear here — those fields are
    confirmed unread by the consumer per the spike's grep audit.
    """
    # Underlying rocisa Instruction. Opaque to the validator except for
    # `_class_tag`, `_canonical_render`, and the `getattr` probes inside
    # `_swait_drains` / `_collect_pattern`. For an asm-source Timeline
    # this is still a rocisa Instruction — `assembly_to_timeline`
    # (rocm-libraries-x6s) is responsible for materializing one. Keeping
    # the field type identical across sources avoids a discriminator
    # branch in `_class_tag`.
    rocisa_inst: object

    # Pre-populated reads / writes + parallel positional operand-slot
    # indices. Same shape and semantics as
    # `WrappedInstruction.{reads, writes, read_slots, write_slots}` today.
    # Element type is `RegisterContainer | MemoryRegion | None` for
    # reads/writes; small-int for the slot tuples. The asm bridge populates
    # these from the rocisa instance via the existing `_populate_wrapper`
    # rule registry (rocm-libraries-7qm).
    reads: Tuple[object, ...]
    writes: Tuple[object, ...]
    read_slots: Tuple[int, ...]
    write_slots: Tuple[int, ...]

    # Scheduler-role tag. CMS supplies categories like "LRA0", "PackB1",
    # "MFMA", "SYNC". An asm-source Timeline supplies tags from a
    # source-aware classifier (rocm-libraries-3dy). The string itself
    # remains the discriminator for `_class_tag_from_category`,
    # `_is_alu_producer`, etc.; the asm tag schema must therefore preserve
    # the prefixes those predicates pattern-match on (LR, LW, GR, MFMA,
    # Pack, SYNC, BARRIER, SNOP, SSETPRIO, LCC).
    category: str

    # Stream-position coordinates within the body. After
    # rocm-libraries-5v4u, `SchedulePosition` is `(loop_index,
    # stream_index)` — a tuple-style lex-comparable key. `loop_index`
    # encodes the body (via `BODY_LABEL_TO_LOOP_INDEX`) so cross-body
    # sort over `(loop_index, stream_index)` gives global stream order.
    position: SchedulePosition

    # Body the event lives on. Today this is implicit (bodies are dict
    # values; `body_label` is set on `GraphNode` by `_make_node` from the
    # for-loop iterator over `_BODY_BUILD_ORDER`). Lifting it onto the
    # event makes the per-event shape closed: the consumer never has to
    # ask "which container did I come from?".
    body_label: str

    # Source-aware label carrier. Carries a `render()` method per the
    # `TaggedInstructionLike` Protocol above. CMS bridges populate this
    # with the existing `TaggedInstruction`; asm bridges populate it with
    # the asm-side wrapper. Kept as a back-reference (not flattened into
    # event fields) because the failure-rendering pipeline needs the
    # per-source rendering hook, not just its rendered string —
    # `cms_node_label` (rocm-libraries-3dy) computes the
    # per-category-stream `[N]` index lazily from the body's instruction
    # stream, which only the carrier has the context for.
    tagged_inst: TaggedInstructionLike


# =============================================================================
# TimelineBody — per-body event stream + body-local maps
# =============================================================================
# Field catalogue and rationale: TIMELINE_INTERFACE_SPIKE.md §3.4.
#
# Why a wrapper class (vs. a parallel `Dict[str, dict]`):
#   * The graph builder writes a sidecar (`_graph_nodes` today) onto each
#     body for cross-body stream walks (CMSValidator.py:1003). A
#     `TimelineBody` is the natural home for that sidecar slot — sub-bead
#     `rocm-libraries-iig` will add it as a typed field when wiring the
#     validator entry. We do NOT pre-add it here per the parent bead's
#     "no Optional defaults unless the spike called for them" directive.
#   * `name_to_idx` is per-body and lives next to `events`; a parallel
#     dict layout would force every Timeline consumer to thread the body
#     label through twice.
@dataclass
class TimelineBody:
    """The events of one loop body plus its body-local maps.

    `name_to_idx` is the CMS symbolic-vgpr-name → numeric-base map
    (rocm-libraries-bb34). The CMS bridge populates it from the writer's
    RegSet directives. The asm bridge leaves it empty (asm dumps already
    carry resolved numeric registers) — the byte-key resolver tolerates
    an empty map.
    """
    events: List[TimelineEvent]
    name_to_idx: dict = field(default_factory=dict)


# =============================================================================
# Timeline — kernel-wide source-agnostic input
# =============================================================================
# Field catalogue and rationale: TIMELINE_INTERFACE_SPIKE.md §3.1, §3.3.
#
# `events_by_body` is keyed by string body label (today: one of "ML-1",
# "ML", "NGL", "NLL"; see `BODY_LABEL_*` in ScheduleCapture). An
# asm-source Timeline can populate fewer than four bodies — the graph
# builder's existing per-body iteration tolerates absences
# (CMSValidator.py:965).
#
# `arch_profile` and `num_mfma_per_subiter` are kernel-wide knobs the
# validator reads off the input today (CMSValidator.py:951, :956). Both
# are required to match the §1.7 catalogue. `arch_profile` is `object` to
# avoid a hard import of `ArchProfile` (which lives in CMSValidator and
# would form a cycle with this module's bridge consumers).
@dataclass
class Timeline:
    """Source-agnostic input to the validator.

    Holds the per-body event streams plus the kernel-wide knobs the
    validator needs.

    No callers are wired through this shape yet — sub-bead
    `rocm-libraries-iig` (5gd.A.4) reroutes `build_dataflow_graph` onto
    `Timeline`. Until then this dataclass sits alongside the existing
    `FourPartCapture` path, not in front of it.
    """
    events_by_body: Dict[str, TimelineBody]
    arch_profile: object
    num_mfma_per_subiter: int
