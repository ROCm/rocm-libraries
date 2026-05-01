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
Schedule capture and comparison for default-vs-CMS validation.

Builds a uniform 4-entry tagged-instruction-stream representation for both the
default scheduler (SIA3) and the CMS scheduler so the two can be diffed for
discrepancies. See plans/then-let-s-work-on-jaunty-reddy.md for design context.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Capture-pipeline exceptions
# =============================================================================
# These are raised (not asserted) so they survive `python -O`. They indicate
# capture/validator-pipeline bugs, NOT user-actionable schedule defects.

class CaptureWiringError(Exception):
    """A TaggedInstruction has rocisa_inst=None when wiring required it."""


class CaptureSMEMError(Exception):
    """A captured body contains SMEM ops; the dscnt FIFO model can't handle them."""


class CaptureFlatError(Exception):
    """A captured body contains flat ops; they decrement two counters at once."""


class CaptureStoreError(Exception):
    """A captured body contains a vector-memory store; vscnt is not modeled."""


class CaptureMfmaCodeShapeError(Exception):
    """The kernel's mfma_code shape doesn't match the expected MFMA pattern."""


class CaptureConsistencyError(Exception):
    """Default and CMS captures disagree on which instructions exist."""


class CaptureIdmapMismatchError(Exception):
    """idMap and capture disagree on instruction count for some category."""


class CaptureUnknownInstructionError(Exception):
    """build_dataflow_graph encountered an instruction class it can't classify."""


class CaptureEmptyBodyError(Exception):
    """A captured body has zero TaggedInstructions; the body should always be populated."""


class UnexplainedMissingEdgeError(Exception):
    """diagnose_missing_edge couldn't classify a missing edge — classifier or pipeline bug."""


SLOT_KIND_PRE_LOOP = "pre_loop"
SLOT_KIND_MFMA = "mfma"
SLOT_KIND_POST_LOOP = "post_loop"


@dataclass(frozen=True)
class SlotKey:
    """Canonical ordering position for a captured instruction.

    Multiple instructions can share the same (iteration, slot_kind, mfma_index);
    `sequence` disambiguates them in emission order.
    """
    iteration: int
    slot_kind: str
    mfma_index: int
    sequence: int


@dataclass
class TaggedInstruction:
    """A rocisa Instruction with its CMS-style id_map category and slot position.

    `category` is determined at emission time by the source list/module the
    instruction was popped from (not by inspecting the instruction's class).
    The same rocisa class (e.g. VFmaMixF32) may carry different categories
    depending on which bucket emitted it.
    """
    inst: object
    category: str
    slot: SlotKey


@dataclass
class LoopBodyCapture:
    """One scheduled loop body as a flat ordered stream of tagged instructions."""
    instructions: list


@dataclass
class CaptureContext:
    """Per-kernel capture lifecycle for the dataflow-graph comparison.

    Replaces the five capture-state attributes that used to live scattered
    across writer (`_last_default_capture`, `_last_cms_capture`,
    `_last_default_main_capture`) and writer.states (`_defaultNGLCapture`,
    `_defaultNLLCapture`), plus the lifecycle helpers that go alongside
    (`_defaultCaptureBuilder`, `_prefetchPackCode{A,B}`).

    `default` and `cms` are the consumer-facing artifacts (read by tests
    after the build); they survive across kernels until overwritten by
    the next build. Everything else is per-kernel scratch state cleared by
    `reset()` from kernelBody's finally block.
    """
    # Final captures (consumer-facing — read by tests, CMSValidator, etc.)
    default: object = None       # FourPartCapture | None
    cms: object = None           # FourPartCapture | None
    # Per-kernel scratch state (cleared by reset()).
    default_main: object = None  # LoopBodyCapture | None
    default_n_gl: object = None  # LoopBodyCapture | None
    default_n_ll: object = None  # LoopBodyCapture | None
    builder: object = None       # LoopBodyCaptureBuilder | None
    # Snapshots of prefetch pack code per plrIdx, used by _loopBody to tag
    # leaves consumed at iter 0 (which never enter PackCodeAAllIters).
    prefetch_pack_a: list = field(default_factory=list)
    prefetch_pack_b: list = field(default_factory=list)

    def reset(self):
        """Clear all per-kernel scratch state. `default` and `cms` are
        intentionally preserved — they're the consumer-facing artifacts.
        """
        self.default_main = None
        self.default_n_gl = None
        self.default_n_ll = None
        self.builder = None
        self.prefetch_pack_a = []
        self.prefetch_pack_b = []


@dataclass
class FourPartCapture:
    """Symmetric per-codepath dict shape for all four loop-body entries.

    main_loop and main_loop_prev are keyed by codepath (under CMS, len ==
    num_codepaths; under default-side capture, always {0: body}).
    n_gl and n_ll are always {0: body} since CMS hard-codes \\ID=0 for the
    tail loops at KernelWriter.py:2858-2866.

    num_mfma is a kernel-level invariant (all four bodies share the same MFMA
    sequence; only non-MFMA categories are stripped/added between them) and is
    promoted here rather than carried per-body.
    """
    main_loop: dict
    main_loop_prev: dict
    n_gl: dict
    n_ll: dict
    num_mfma: int
    num_codepaths: int
    source: str  # 'cms' or 'default-sia3'
    # numMfmaPerIter — used by build_dataflow_graph to derive per-MFMA
    # iteration index (mfma_index // num_mfma_per_iter). Both default-side
    # and CMS-side construction sites should pass writer.states.numMfmaPerIter
    # so the two graphs derive matching logical positions for the same MFMA.
    # Defaults to 0 ("don't split MFMAs by iter"); the resolver then keeps
    # all MFMAs at iter 0, which loses cross-iter PLR dataflow edges. Test
    # fixtures may safely leave it unset.
    num_mfma_per_iter: int = 0


# =============================================================================
# Body labels — one entry per loop body in a FourPartCapture
# =============================================================================
# Stable string labels used in GraphNode.body_label and as keys in
# DataflowGraph.captures. The numeric loop_index inside SchedulePosition is
# derived via BODY_LABEL_TO_LOOP_INDEX so cross-body order is well-defined
# (ML-1 < ML < NGL < NLL by construction).

BODY_LABEL_ML_PREV = "ML-1"
BODY_LABEL_ML = "ML"
BODY_LABEL_NGL = "NGL"
BODY_LABEL_NLL = "NLL"

BODY_LABEL_TO_LOOP_INDEX = {
    BODY_LABEL_ML_PREV: 0,
    BODY_LABEL_ML: 1,
    BODY_LABEL_NGL: 2,
    BODY_LABEL_NLL: 3,
}


@dataclass(frozen=True)
class GraphPosition:
    """Like CMSValidator.SchedulePosition but defined here to keep the graph
    builder free of a hard CMSValidator import.

    Fields ordered for tuple-style comparison (loop_index, vmfma_index, sub_index).
    """
    loop_index: int
    vmfma_index: int
    sub_index: int

    def __lt__(self, other) -> bool:
        return (self.loop_index, self.vmfma_index, self.sub_index) < \
               (other.loop_index, other.vmfma_index, other.sub_index)

    def __le__(self, other) -> bool:
        return (self.loop_index, self.vmfma_index, self.sub_index) <= \
               (other.loop_index, other.vmfma_index, other.sub_index)

    def __gt__(self, other) -> bool:
        return (self.loop_index, self.vmfma_index, self.sub_index) > \
               (other.loop_index, other.vmfma_index, other.sub_index)

    def __ge__(self, other) -> bool:
        return (self.loop_index, self.vmfma_index, self.sub_index) >= \
               (other.loop_index, other.vmfma_index, other.sub_index)


def make_position(body_label, slot) -> GraphPosition:
    """Construct a GraphPosition from a TaggedInstruction.slot SlotKey.

    The body_label maps to loop_index via BODY_LABEL_TO_LOOP_INDEX so cross-body
    ordering is well-defined.
    """
    return GraphPosition(
        loop_index=BODY_LABEL_TO_LOOP_INDEX[body_label],
        vmfma_index=slot.mfma_index,
        sub_index=slot.sequence,
    )


@dataclass
class GraphNode:
    """A node in the unified 4-body dataflow graph.

    identity is the canonical key for cross-graph comparison: position-independent
    (survives CMS reordering) and content-based (same producer in default and CMS
    captures gets the same identity even if its stream position differs).

    position lives in graph-builder space (loop_index spans bodies); the
    underlying TaggedInstruction.slot is preserved on tagged_inst.
    """
    identity: tuple                     # (rocisa_class_name, loop_index, signature_tuple)
    position: GraphPosition
    category: str                       # propagated from TaggedInstruction
    rocisa_inst: object                 # back-reference to the rocisa instruction
    tagged_inst: TaggedInstruction      # back-reference for stream-position lookup
    body_label: str                     # 'ML-1' | 'ML' | 'NGL' | 'NLL'
    name: str = ""                      # human-readable label (e.g. 'LRA0[2]')


@dataclass
class DataflowEdge:
    """A register-flow edge in the dataflow graph.

    edge_kind discriminates the three kinds of dataflow this graph models:
      raw_intrawave        — producer SWait drains the in-wave counter
      lr_to_gr_lds_reuse   — LR0 -> SWait -> SBarrier -> GR (write reuses LDS slot)
      gr_to_lr_lds_reuse   — GR -> SWait -> SBarrier -> LR1 (read of just-written LDS)
    """
    producer: GraphNode
    consumer: GraphNode
    register: object                    # RegisterContainer (opaque to avoid hard rocisa import)
    edge_kind: str                      # 'raw_intrawave' | 'lr_to_gr_lds_reuse' | 'gr_to_lr_lds_reuse'


@dataclass
class DataflowGraph:
    """Unified graph spanning all 4 captured bodies.

    Single graph (not one per body) so cross-body edges (e.g. DTL+LdsBuf
    previous-iteration LR0 -> current GR) are represented natively as edges
    between nodes whose body_labels differ.

    nodes is keyed by identity; the comparison rule iterates the top-level
    edges list. Per-node adjacency is intentionally NOT stored — the
    diagnostic classifier walks captures[body_label].instructions instead.
    """
    nodes: dict                         # identity -> GraphNode
    edges: list                         # list[DataflowEdge]
    captures: dict                      # body_label -> LoopBodyCapture

    def edge_keys(self):
        """Edge-equality keys for cross-graph diff: (p_id, c_id, register, kind)."""
        return {(e.producer.identity, e.consumer.identity, e.register, e.edge_kind)
                for e in self.edges}


# =============================================================================
# Failure hierarchy — typed scheduling defects with polymorphic formatters
# =============================================================================
#
# Single base class; each concrete subclass owns its formatter via format(capture).
# Tests assert on type and field, not on string content. The only place wording
# is asserted is in Tensile/Tests/unit/test_failure_formatters.py.
#
# Each Failure carries CMS-side state only — never a reference to the
# default-side schedule. The user fixes the CMS schedule from the data on
# the Failure.

def _ordinal(n: int) -> str:
    """Return '1st', '2nd', '3rd', '4th', ..., 'Nth' for any positive n."""
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def format_position(node, capture=None) -> str:
    """Render a node's schedule position with optional list-position suffix.

    MFMAs are excluded from the list-position suffix because they aren't
    user-scheduled (their order is fixed by the underlying instruction loop).
    Everything else, including MFMAPack (category 'PackA*'/'PackB*'), gets
    the (Nth entry in list) suffix because the CMS user controls placement.

    The discriminator is the category tag, NOT isinstance — MFMAPack's
    multiple inheritance from both Pack and MFMA would confuse isinstance.

    If the node's tagged_inst isn't in the given capture (e.g. when the
    node is from a different body in the unified 4-body graph), the
    list-position suffix is silently omitted.
    """
    base = f"@ idx={node.position.vmfma_index}"
    if capture is None or node.category == "MFMA":
        return base
    if node.tagged_inst is None:
        return base
    try:
        list_pos = capture.instructions.index(node.tagged_inst)
    except ValueError:
        return base
    return f"{base} ({_ordinal(list_pos + 1)} entry in list)"


@dataclass
class Failure:
    """Common base for all reported scheduling problems.

    No body_label field on the base — every concrete subclass carries
    producer/consumer GraphNode references (or equivalent), and
    GraphNode.body_label is the source of truth.
    """

    def format(self, capture=None) -> str:
        """Stable boundary method. Delegates to the subclass canonical
        formatter."""
        return self._format_canonical(capture)

    def _format_canonical(self, capture=None) -> str:
        raise NotImplementedError("subclasses must implement _format_canonical()")


# ----------------------------------------------------------------------------
# 1. OrderInvertedFailure — producer issued after consumer (same body only).
#    Replaces: GR _validate_must_start_after early-issue branch, Pack early/late.
#    Emitted by: rules + dataflow comparison classifier.
# ----------------------------------------------------------------------------
@dataclass
class OrderInvertedFailure(Failure):
    producer: object  # GraphNode or ValidatorInstruction
    consumer: object

    def _format_canonical(self, capture=None) -> str:
        producer_pos = format_position(self.producer, capture)
        consumer_pos = format_position(self.consumer, capture)
        return (
            f"{self.producer.category}[{getattr(self.producer, 'name', '')}] "
            f"{producer_pos} is issued after its consumer "
            f"{self.consumer.category}[{getattr(self.consumer, 'name', '')}] "
            f"{consumer_pos}. The producer must complete before the consumer can use it."
        )


# ----------------------------------------------------------------------------
# 2. MissingWaitFailure — no SWaitCnt covers the producer at all.
# ----------------------------------------------------------------------------
@dataclass
class MissingWaitFailure(Failure):
    producer: object
    consumer: object
    counter_kind: str  # 'dscnt' / 'vlcnt' / 'vscnt'

    def _format_canonical(self, capture=None) -> str:
        return (
            f"{self.consumer.category}[{getattr(self.consumer, 'name', '')}] "
            f"{format_position(self.consumer, capture)} is not guaranteed by any "
            f"SWaitCnt before its producer {self.producer.category} "
            f"{format_position(self.producer, capture)}. No SWaitCnt with "
            f"{self.counter_kind} drain appears between them in the schedule."
        )


# ----------------------------------------------------------------------------
# 3. WaitOnWrongCounterFailure — SWait exists but on wrong counter.
# ----------------------------------------------------------------------------
@dataclass
class WaitOnWrongCounterFailure(Failure):
    producer: object
    consumer: object
    expected_counter: str
    wrong_counter_waits: list  # list[GraphNode], in stream order

    def _format_canonical(self, capture=None) -> str:
        wait_descriptions = ", ".join(
            f"SWaitCnt {format_position(w, capture)}"
            for w in self.wrong_counter_waits
        )
        return (
            f"{self.consumer.category}[{getattr(self.consumer, 'name', '')}] "
            f"{format_position(self.consumer, capture)}'s producer "
            f"{self.producer.category} {format_position(self.producer, capture)} "
            f"requires an SWaitCnt with {self.expected_counter} drain. "
            f"Existing SWaitCnts in the window drain other counters: {wait_descriptions}. "
            f"Did you mean {self.expected_counter}?"
        )


# ----------------------------------------------------------------------------
# 4. WaitTooLateFailure — SWait fires at/after the consumer.
# ----------------------------------------------------------------------------
@dataclass
class WaitTooLateFailure(Failure):
    producer: object
    consumer: object
    wait_position: object  # SchedulePosition

    def _format_canonical(self, capture=None) -> str:
        return (
            f"{self.consumer.category}[{getattr(self.consumer, 'name', '')}] "
            f"{format_position(self.consumer, capture)} is guaranteed by an "
            f"SWaitCnt @ idx={self.wait_position.vmfma_index} which fires at "
            f"or after the consumer position. Move the wait earlier in the schedule."
        )


# ----------------------------------------------------------------------------
# 5. WaitInsufficientFailure — wait at correct position but counter value too lax.
# ----------------------------------------------------------------------------
@dataclass
class WaitInsufficientFailure(Failure):
    producer: object
    consumer: object
    wait: object  # GraphNode
    queue_depth_at_wait: int
    counter_value: int

    def _format_canonical(self, capture=None) -> str:
        return (
            f"{self.consumer.category}[{getattr(self.consumer, 'name', '')}] "
            f"{format_position(self.consumer, capture)}'s producer "
            f"{self.producer.category} {format_position(self.producer, capture)} "
            f"is guaranteed by SWaitCnt {format_position(self.wait, capture)}, "
            f"but the counter value ({self.counter_value}) leaves "
            f"{self.queue_depth_at_wait - self.counter_value} ops still pending "
            f"(queue depth at wait = {self.queue_depth_at_wait}). "
            f"Tighten the wait's counter value."
        )


# ----------------------------------------------------------------------------
# 6. MissingBarrierFailure — cross-wave LDS-reuse needs barrier in the window.
# ----------------------------------------------------------------------------
@dataclass
class MissingBarrierFailure(Failure):
    producer: object
    consumer: object
    role: str  # 'must_start_after' | 'needed_by'

    def _format_canonical(self, capture=None) -> str:
        if self.role == "must_start_after":
            order = (
                f"{self.producer.category} -> SWaitCnt(dscnt=0) -> SBarrier "
                f"-> {self.consumer.category}"
            )
            why = (
                f"{self.consumer.category} overwrites the LDS slot read by "
                f"{self.producer.category}. The barrier ensures all waves "
                f"finished reading before the write."
            )
        else:
            order = (
                f"{self.producer.category} -> SWaitCnt(vlcnt=0) -> SBarrier "
                f"-> {self.consumer.category}"
            )
            why = (
                f"{self.consumer.category} reads the LDS slot written by "
                f"{self.producer.category}. The barrier ensures all waves "
                f"finished writing before the read."
            )
        return (
            f"{why} Required ordering is {order}. SWaitCnt is present but no "
            f"SBarrier appears between the SWaitCnt and "
            f"{self.consumer.category} {format_position(self.consumer, capture)}."
        )


# ----------------------------------------------------------------------------
# 7. WrongInterleavingFailure — MiddlePack pair-consumer ordering wrong.
# ----------------------------------------------------------------------------
@dataclass
class WrongInterleavingFailure(Failure):
    pack: object  # MiddlePack
    expected_next: object  # MiddlePack (pair_consumer)
    actual_next: object  # MiddlePack (next_scheduled_middle_16)

    def _format_canonical(self, capture=None) -> str:
        return (
            f"{self.pack.name} @ idx={self.pack.issued_at.vmfma_index} has wrong "
            f"interleaving. Should have been followed by "
            f"{self.expected_next.name} @ idx={self.expected_next.issued_at.vmfma_index} "
            f"but was followed by "
            f"{self.actual_next.name} @ idx={self.actual_next.issued_at.vmfma_index}."
        )


# ----------------------------------------------------------------------------
# 8. TimingTooCloseFailure — quad-cycle gap too small (Pack timing).
# ----------------------------------------------------------------------------
@dataclass
class TimingTooCloseFailure(Failure):
    producer: object  # Pack
    consumer: object  # Pack/MFMA
    expected_quad_cycles: int
    actual_quad_cycles: int

    def _format_canonical(self, capture=None) -> str:
        return (
            f"{self.producer.name} @ idx={self.producer.issued_at.vmfma_index} has "
            f"too little gap between it and {self.consumer.name} @ idx="
            f"{self.consumer.issued_at.vmfma_index}. Expected at least "
            f"{self.expected_quad_cycles} quad-cycles but only "
            f"{self.actual_quad_cycles} passed."
        )


# ----------------------------------------------------------------------------
# 9. InvalidCounterValueFailure — SWait field range check.
# ----------------------------------------------------------------------------
@dataclass
class InvalidCounterValueFailure(Failure):
    swait: object  # SWait validator instruction
    dscnt: int
    vlcnt: int
    vscnt: int

    def _format_canonical(self, capture=None) -> str:
        return (
            f"SWaitCnt @ idx={self.swait.issued_at.vmfma_index} is invalid: "
            f"dscnt={self.dscnt}, vlcnt={self.vlcnt}, vscnt={self.vscnt}. "
            f"All counter fields must be >= -1 (with -1 meaning 'don't care')."
        )


# ----------------------------------------------------------------------------
# 10. SCCConflictFailure — GRInc SCC overlap window.
# ----------------------------------------------------------------------------
@dataclass
class SCCConflictFailure(Failure):
    conflicting_name: str
    grinc_name: str
    conflicting_index: int
    interval_start: int
    interval_end: int

    def _format_canonical(self, capture=None) -> str:
        return (
            f"{self.conflicting_name} at index {self.conflicting_index} can't be "
            f"between {self.grinc_name} {self.interval_start}-{self.interval_end} "
            f"due to SCC usage."
        )


# ----------------------------------------------------------------------------
# 11. SWaitCountExceedsOutstandingFailure — SWait references too many ops.
# ----------------------------------------------------------------------------
@dataclass
class SWaitCountExceedsOutstandingFailure(Failure):
    swait: object
    counter_kind: str  # 'dscnt' | 'vlcnt'
    counter_value: int
    outstanding: int

    def _format_canonical(self, capture=None) -> str:
        load_kind = "DS loads" if self.counter_kind == "dscnt" else "VM loads"
        return (
            f"SWaitCnt @ idx={self.swait.issued_at.vmfma_index} has "
            f"{self.counter_kind}={self.counter_value} but only {self.outstanding} "
            f"{load_kind} are outstanding."
        )


# ----------------------------------------------------------------------------
# 12. OutOfOrderSequenceFailure — instruction A came after B that should follow it.
#     Two detection sites with the same shape: schedule-sequence ordering
#     (verify_ascending_order) and CVT0/CVT1 pair ordering (CMSValidator.py:1642).
# ----------------------------------------------------------------------------
@dataclass
class OutOfOrderSequenceFailure(Failure):
    kind: str  # 'sequence' | 'cvt_pair'
    schedule_key: str  # category name (e.g. 'GRIncA') or pair description
    sequence: object  # the offending sequence (list) or pair (tuple)
    bad_value: int
    bad_index: int
    prev_value: int

    def _format_canonical(self, capture=None) -> str:
        if self.kind == "sequence":
            return (
                f"Non-descending-order rule failed, schedule key "
                f"'{self.schedule_key}', sequence {self.sequence}: value "
                f"{self.bad_value} at index {self.bad_index} is less than "
                f"{self.prev_value} at index {self.bad_index - 1}."
            )
        else:
            # cvt_pair
            return (
                f"CVT pair ordering violated for {self.schedule_key}: CVT0 ends "
                f"at issue_index {self.prev_value} but CVT1 starts at issue_index "
                f"{self.bad_value}. CVT0 must precede CVT1 in the pack chain."
            )


# Class-name lists for finalize() guards. Class-name matching (not isinstance)
# keeps this module free of hard rocisa imports — the same names work for
# real rocisa classes and for synthetic test stand-ins.

_SMEM_CLASS_NAMES = {
    "SLoadB32", "SLoadB64", "SLoadB128", "SLoadB256", "SLoadB512",
    "SStoreB32", "SStoreB64", "SStoreB128",
    "SMemLoadInstruction", "SMemStoreInstruction",
}

_FLAT_CLASS_NAMES = {
    "FlatLoadB8", "FlatLoadB16", "FlatLoadB32", "FlatLoadB64", "FlatLoadB128",
    "FlatStoreB8", "FlatStoreB16", "FlatStoreB32", "FlatStoreB64", "FlatStoreB128",
    "FLATReadInstruction", "FLATStoreInstruction",
}

_VECTOR_STORE_CLASS_NAMES = {
    "BufferStoreB32", "BufferStoreB64", "BufferStoreB128",
    "GlobalStoreB32", "GlobalStoreB64", "GlobalStoreB128",
    "BufferStoreInstruction", "GlobalStoreInstruction",
}


class LoopBodyCaptureBuilder:
    """Accumulates TaggedInstructions across multiple emission calls.

    Owns a `sequence` counter that increments per-append within the same
    (iteration, slot_kind, mfma_index) triple to produce deterministic SlotKeys.

    finalize() runs capture-pipeline guards before returning the capture:
      - rocisa wiring: every TaggedInstruction.inst is non-None
      - SMEM guard:    no SLoad*/SStore* in the body (would desync dscnt FIFO)
      - flat guard:    no Flat* in the body (decrement two counters at once)
      - store guard:   no vector-memory stores (vscnt is not modeled)

    These checks raise named exceptions (NOT bare assert) so they survive
    `python -O` and propagate diagnostic context.
    """

    def __init__(self):
        self._instructions = []
        self._seq_counter = {}  # (iteration, slot_kind, mfma_index) -> next sequence

    def append(self, inst, category, iteration, slot_kind=SLOT_KIND_MFMA, mfma_index=-1):
        key = (iteration, slot_kind, mfma_index)
        seq = self._seq_counter.get(key, 0)
        self._seq_counter[key] = seq + 1
        slot = SlotKey(
            iteration=iteration,
            slot_kind=slot_kind,
            mfma_index=mfma_index,
            sequence=seq,
        )
        self._instructions.append(TaggedInstruction(inst=inst, category=category, slot=slot))

    def finalize(self):
        for ti in self._instructions:
            inst = ti.inst
            if inst is None:
                raise CaptureWiringError(
                    f"LoopBodyCaptureBuilder.finalize: TaggedInstruction "
                    f"in category {ti.category!r} at slot "
                    f"mfma_index={ti.slot.mfma_index} has inst=None. "
                    f"rocisa wiring failed during capture."
                )
            cls_name = type(inst).__name__
            if cls_name in _SMEM_CLASS_NAMES:
                raise CaptureSMEMError(
                    f"LoopBodyCaptureBuilder.finalize: SMEM op "
                    f"{cls_name} in category {ti.category!r}. "
                    f"SMEM also decrements dscnt and would desync the "
                    f"per-counter FIFO model used by build_dataflow_graph."
                )
            if cls_name in _FLAT_CLASS_NAMES:
                raise CaptureFlatError(
                    f"LoopBodyCaptureBuilder.finalize: flat op "
                    f"{cls_name} in category {ti.category!r}. "
                    f"Flat ops decrement both vmcnt and dscnt simultaneously, "
                    f"which the per-counter queue model doesn't handle."
                )
            if cls_name in _VECTOR_STORE_CLASS_NAMES:
                raise CaptureStoreError(
                    f"LoopBodyCaptureBuilder.finalize: vector-memory store "
                    f"{cls_name} in category {ti.category!r}. "
                    f"vscnt is not tracked; no current CMS body emits stores."
                )
        return LoopBodyCapture(instructions=list(self._instructions))


# =============================================================================
# --- CMS category schema ---
# =============================================================================
# Single source of truth for which CMS categories exist and which source modules
# each draws from. Both customMainLoopSchedule (for idMap consumed by
# cmsv.isValid) and the SIA3 default-side capture path (for tag_by_origin_id)
# call build_idmap; the latter then inverts via invert_idmap_to_id_to_category
# to feed its {id(inst): category} tag map. Adding a new category means adding
# it here and only here.


def split_for_plr(module):
    """Split a single per-loop module into 2 halves for numIterPLR=0 mode.

    Mirrors customMainLoopSchedule's nested splitForPLR. Both call sites
    (CustomSchedule and the default-side capture builder) must use this
    helper; otherwise CMS and default-side capture see different
    per-iteration shapes and their idMaps diverge.

    Returns [second_half, first_half] — second half is iter 0; first
    half is iter 1. Items are NOT cloned; the returned lists share
    Python identity with the source module's items.
    """
    items = module.flatitems()
    n = len(items)
    return [items[n // 2:], items[:n // 2]]


def build_idmap(*, num_loop_iter,
                LRCodeA, PackCodeA, LRCodeB, PackCodeB,
                globalReadA, globalReadB,
                globalReadIncACode, globalReadIncBCode,
                localWriteA, localWriteB,
                LRSwapA, LRSwapB, LWSwapA, LWSwapB,
                loopCounterCode, syncCode, snopCode):
    """Build the canonical {category: source-module-or-list} dict.

    This is the SINGLE definition of which categories exist and what
    source each draws from. Both consumers call this:
      - customMainLoopSchedule (CustomSchedule.py): builds idMap for
        cmsv.isValid() and stashes on writer._last_id_map.
      - _loopBody capture site (KernelWriter.py): inverts via
        invert_idmap_to_id_to_category() to feed the SIA3 default-side
        capture's tag map.

    Adding a new category means adding it here and only here.
    """
    idmap = {
        'GRIncA': globalReadIncACode,
        'GRIncB': globalReadIncBCode,
        'GRA':    globalReadA,
        'GRB':    globalReadB,
        'LWA':    localWriteA,
        'LWB':    localWriteB,
        'LRSA':   LRSwapA,
        'LRSB':   LRSwapB,
        'LWSA':   LWSwapA,
        'LWSB':   LWSwapB,
        'LCC':    loopCounterCode,
    }
    for u in range(num_loop_iter):
        idmap[f"LRA{u}"]   = LRCodeA[u]
        idmap[f"LRB{u}"]   = LRCodeB[u]
        idmap[f"PackA{u}"] = PackCodeA[u]
        idmap[f"PackB{u}"] = PackCodeB[u]
    idmap['SYNC'] = syncCode
    idmap['SNOP'] = snopCode
    return idmap


def invert_idmap_to_id_to_category(idmap):
    """Invert {category: items_or_module} to {id(instruction): category}.

    Handles values that are either Modules (with .flatitems()) or plain
    lists of instructions (the form CustomSchedule's removeComments()
    produces).

    Raises ValueError if the same instruction id appears under two
    categories — this is a schema bug (a leaf shared between two
    categories), not a normal case. Last-wins would silently corrupt
    categorization downstream.
    """
    out = {}
    for cat, val in idmap.items():
        if val is None:
            continue
        items = val.flatitems() if hasattr(val, 'flatitems') else val
        for item in items:
            key = id(item)
            if key in out and out[key] != cat:
                raise ValueError(
                    f"Instruction id {key} appears under both "
                    f"categories {out[key]!r} and {cat!r}; idMap schema bug."
                )
            out[key] = cat
    return out


def structural_clone(item):
    """Recursively clone rocisa Module wrappers; share leaf references.

    SIA3 calls popFirstItem/popFirstNItems on input modules — that mutates
    the Module's child list. To isolate a callee from that mutation while
    preserving leaf-instruction Python identity (so id-based categorization
    survives across the boundary), we clone every Module wrapper in the
    tree but reuse the same Python objects for non-Module children
    (Instruction, TextBlock, Label, ValueIf, etc.).

    The reused leaves have stable id() values, so any pre-built
    {id(item) -> category} dict (e.g. from CMS's idMap) continues to
    work for lookups against the cloned tree's contents.

    Module attribute audit (Step 0 of the structural-clone work):
      - name: constructor arg.
      - setNoOpt/isNoOpt: local boolean — preserved via setNoOpt(...).
      - parent: re-assigned when the cloned child is add()-ed to its
        cloned parent in the recursion above.
      - kernel, vgprIdx, vgprMsb, archCaps/asmCaps/asmBugs/regCaps:
        read-only properties derived from kernel context, no local
        state.
      - setInlineAsmPrintMode: no getter exposed (invisible state); only
        set in Activation.py, far from any capture-input module.
      - addTempVgpr: method exists but no Python callers.
    """
    from rocisa.code import Module
    if not isinstance(item, Module):
        return item  # leaf: share reference (Instruction, TextBlock, Label, ValueIf, etc.)
    # Pass empty string explicitly rather than relying on a truthy check —
    # an unnamed Module's name should round-trip to an unnamed clone.
    new_mod = Module(item.name or "")
    if item.isNoOpt():
        new_mod.setNoOpt(True)
    for child in item.items():
        new_mod.add(structural_clone(child))
    return new_mod


def build_id_to_category_per_iter(*, iteration, localReadCode, localWriteCode,
                                  globalReadCode, packCode, packPreCode,
                                  globalReadA=None, globalReadB=None,
                                  globalReadIncACode=None, globalReadIncBCode=None,
                                  inner_unroll_max=8):
    """Build {id(item) -> category} for one SIA3 iteration's combined modules.

    Companion to build_idmap. Two factories, one schema, two input shapes:

    - build_idmap + invert_idmap_to_id_to_category: when the caller has
      per-category source modules (LRCodeAAllIters[u] etc.). Used by the
      main-loop capture path in _loopBody.
    - build_id_to_category_per_iter: when the caller has per-iteration
      combined modules (localReads, pack[packIdx], packPre[packPreIdx])
      with named A/B/MXSA/MXSB/Metadata sub-modules. Used by the NLL/NGL
      capture path in _noLoadLoopBodyDefault.

    Both produce the same {id(item) -> category} mapping that
    _captureSubIterToBuilder consumes.

    Walk strategy (most-specific tag wins, populated first):
      - globalReadIncACode / globalReadIncBCode (optional): items tagged
        GRIncA / GRIncB. Same scheme CMS's idMap uses, so cross-graph
        comparison agrees on the GR-inc identities instead of seeing
        them as generic 'GR' on one side.
      - globalReadA / globalReadB (optional): items tagged GRA / GRB.
        Splits the buffer-load instructions by tensor side.
      - localReadCode: per-tensor (A/B/MXSA/MXSB/Metadata) sub-modules
        named LocalReadDo{Tensor}_I{iui}; tag as LR{Tensor}{iteration}.
      - packCode/packPreCode: per-side (A/B) sub-modules named
        pack{A,B}_I{iui} (and "pack{A,B}_I{iui} Pre"); tag as
        Pack{A,B}{iteration}.
      - globalReadCode: fallback generic 'GR' for items not already
        tagged (covers anything the per-side modules above missed).
      - localWriteCode: tagged generically as 'LW'.
      - First-tag-wins under DirectToLds=1 where globalRead instructions
        ARE the local writes (same Instruction objects); the GR tag wins.
    """
    id_to_category = {}

    def tag_module(mod, category):
        if mod is None:
            return
        for item in mod.flatitems():
            id_to_category[id(item)] = category

    def tag_module_setdefault(mod, category):
        if mod is None:
            return
        for item in mod.flatitems():
            id_to_category.setdefault(id(item), category)

    # Most-specific GR-related tags first so they win over the generic
    # 'GR' fallback below. The globalReadIncCode and globalReadA/B leaves
    # are SHARED with globalReadCode (perIterGlobalRead) leaves via
    # SIA.py:732 (extends GR-load list with GR-inc list and distributes
    # across iters); tagging them by id() here pre-empts the fallback.
    if globalReadIncACode is not None:
        tag_module(globalReadIncACode, "GRIncA")
    if globalReadIncBCode is not None:
        tag_module(globalReadIncBCode, "GRIncB")
    if globalReadA is not None:
        tag_module_setdefault(globalReadA, "GRA")
    if globalReadB is not None:
        tag_module_setdefault(globalReadB, "GRB")

    if localReadCode is not None:
        for sub_name, cat_template in (
            ("LocalReadDoA",        "LRA{}"),
            ("LocalReadDoB",        "LRB{}"),
            ("LocalReadDoMXSA",     "LRMXSA{}"),
            ("LocalReadDoMXSB",     "LRMXSB{}"),
            ("LocalReadDoMetadata", "LRMetadata{}"),
        ):
            cat = cat_template.format(iteration)
            for iui in range(inner_unroll_max):
                sub = localReadCode.findNamedItem(f"{sub_name}_I{iui}")
                if sub is not None:
                    tag_module(sub, cat)

    if globalReadCode is not None:
        for item in globalReadCode.flatitems():
            id_to_category.setdefault(id(item), "GR")

    if localWriteCode is not None:
        for item in localWriteCode.flatitems():
            id_to_category.setdefault(id(item), "LW")

    for pack_mod in (packCode, packPreCode):
        if pack_mod is None:
            continue
        for iui in range(inner_unroll_max):
            for prefix, side in (("packA", "A"), ("packB", "B")):
                sub = pack_mod.findNamedItem(f"{prefix}_I{iui}")
                if sub is not None:
                    tag_module(sub, f"Pack{side}{iteration}")
                sub_pre = pack_mod.findNamedItem(f"{prefix}_I{iui} Pre")
                if sub_pre is not None:
                    tag_module(sub_pre, f"Pack{side}{iteration}")

    return id_to_category


def assert_idmap_completeness(idmap, capture):
    """Verify per-category instruction counts match between idMap and capture.

    Excludes SYNC and SNOP categories: CMS lets the user specify arbitrary
    numbers of waits and snops, so count parity isn't a coverage property.

    Raises CaptureIdmapMismatchError if any other category's count differs.
    """
    captured_by_cat = {}
    for ti in capture.instructions:
        captured_by_cat.setdefault(ti.category, []).append(ti)
    for cat, idmap_insts in idmap.items():
        if cat in ("SYNC", "SNOP"):
            continue
        captured = captured_by_cat.get(cat, [])
        if len(idmap_insts) != len(captured):
            raise CaptureIdmapMismatchError(
                f"Category {cat}: idMap declares {len(idmap_insts)} "
                f"instructions, capture has {len(captured)}."
            )


# =============================================================================
# Module-level helpers used by the graph builder, edge collectors, and classifier
# =============================================================================
# Defined as free functions (not methods on DataflowGraph or LoopBodyCapture)
# so they're testable in isolation with synthetic fixtures.


PRODUCER_CATEGORIES_LDS = ("LRA0", "LRA1", "LRA3", "LRB0", "LRB1", "LRB3",
                           "LWA", "LWB", "LW")
PRODUCER_CATEGORIES_GLOBAL = ("GRA", "GRB", "GR")
SWAIT_CATEGORY = "SYNC"
SBARRIER_CATEGORY = "BARRIER"


def counter_for(node_or_category) -> str:
    """Return the SWaitCnt counter that gates the given producer.

    'dscnt' for LR/LW (LDS ops); 'vlcnt' for GR (vector-memory loads).

    Raises CaptureUnknownInstructionError if asked about a category that
    isn't one of the recognized producer kinds — graph builder should
    never have created a node whose category is unknown.
    """
    cat = node_or_category if isinstance(node_or_category, str) else node_or_category.category
    if cat in PRODUCER_CATEGORIES_LDS:
        return "dscnt"
    if cat in PRODUCER_CATEGORIES_GLOBAL:
        return "vlcnt"
    raise CaptureUnknownInstructionError(
        f"counter_for: category {cat!r} is not a recognized producer kind. "
        f"Expected one of LR*/LW* (dscnt) or GR* (vlcnt)."
    )


def _swait_drains(swait_node, counter: str):
    """Return the counter value the SWait imposes on `counter`, or None if it
    doesn't constrain that counter.

    A SWaitCnt's field is set to -1 when the counter is unconstrained
    ('don't care'); a value >= 0 caps outstanding ops at that count.
    """
    inst = swait_node.rocisa_inst
    if inst is None:
        return None
    if counter == "dscnt":
        v = getattr(inst, "dscnt", -1)
    elif counter == "vlcnt":
        v = getattr(inst, "vlcnt", -1)
    elif counter == "vscnt":
        v = getattr(inst, "vscnt", -1)
    else:
        return None
    if v is None or v < 0:
        return None
    return v


def _all_nodes_in_order(subj_graph):
    """Yield every node in execution order across all bodies.

    Used by the wait/barrier helpers below to walk cross-body windows
    (e.g. producer in body=ML-1, consumer in body=ML). Per-body
    `_graph_nodes` is already in stream order; bodies are enumerated in
    `_BODY_BUILD_ORDER` which matches GraphPosition.loop_index ordering,
    so concatenating yields a globally-correct stream.
    """
    for label in _BODY_BUILD_ORDER:
        cap = subj_graph.captures.get(label) if subj_graph is not None else None
        if cap is None or not hasattr(cap, '_graph_nodes'):
            continue
        for node in cap._graph_nodes:
            yield node


def waits_in_window(subj_graph, start: GraphPosition, end: GraphPosition,
                    *, counter=None, exclude_counter=None):
    """Return SWaitCnt nodes (as GraphNodes) whose position is in [start, end)
    and whose counter field constrains the requested counter.

    Walks across bodies via `subj_graph.captures` so cross-body windows
    (producer in body=ML-1, consumer in body=ML) include SWaits from
    every body that overlaps the window.

    Either `counter` or `exclude_counter` may be passed, not both. If both
    are None, returns all SWaits in the window regardless of counter.
    """
    if counter is not None and exclude_counter is not None:
        raise ValueError("counter and exclude_counter are mutually exclusive")
    out = []
    for node in _all_nodes_in_order(subj_graph):
        if node.category != SWAIT_CATEGORY:
            continue
        if not (start <= node.position < end):
            continue
        if counter is not None:
            if _swait_drains(node, counter) is None:
                continue
        if exclude_counter is not None:
            if _swait_drains(node, exclude_counter) is not None:
                # The wait DOES constrain the excluded counter — skip.
                continue
        out.append(node)
    return out


def barriers_in_window(subj_graph, start: GraphPosition, end: GraphPosition):
    """Return SBarrier nodes whose position is in (start, end) — exclusive on
    both ends. A barrier at the producer's position doesn't cover the producer;
    a barrier at the consumer's position doesn't precede the consumer.

    Walks across bodies for the same reason as waits_in_window.
    """
    out = []
    for node in _all_nodes_in_order(subj_graph):
        if node.category != SBARRIER_CATEGORY:
            continue
        if start < node.position < end:
            out.append(node)
    return out


# =============================================================================
# Per-instruction shape extractors
# =============================================================================
# Each function returns enough state for the graph builder to:
#   1. Build a content-based identity (for cross-graph diff)
#   2. Discover what registers the instruction reads/writes (for edges)
#
# Detection by class-name string keeps this module free of hard rocisa
# imports — tests use _Fake* stand-ins; production passes real rocisa
# classes. The class name is the same in both worlds.

# Class names (as returned by type(inst).__name__) recognized by the builder.
_LR_CLASS_NAMES = {
    "_FakeLR",
    # Real rocisa LR classes: DSLoadB32 / DSLoadB64 / DSLoadB128 / DSLoadB256
    "DSLoadB32", "DSLoadB64", "DSLoadB128", "DSLoadB256",
    # Generic class umbrella (for isinstance fallback if needed)
    "DSLoadInstruction",
}
_LW_CLASS_NAMES = {
    "_FakeLW",
    "DSStoreB8", "DSStoreB16", "DSStoreB32", "DSStoreB64", "DSStoreB128",
    "DSStoreInstruction",
}
_GR_CLASS_NAMES = {
    "_FakeGR",
    # rocisa BufferLoad classes
    "BufferLoadB32", "BufferLoadB64", "BufferLoadB128",
    "GlobalLoadB32", "GlobalLoadB64", "GlobalLoadB128",
    "BufferLoadInstruction", "GlobalLoadInstruction",
    "GlobalReadInstruction",
}
_MFMA_CLASS_NAMES = {
    "_FakeMFMA",
    "MFMAInstruction",
}
_SWAIT_CLASS_NAMES = {
    "_FakeSWait",
    "SWaitCnt",
}
_SBARRIER_CLASS_NAMES = {
    "_FakeSBarrier",
    "SBarrier",
}
_SNOP_CLASS_NAMES = {
    "_FakeSNop",
    "SNop",
}


def _is_lr(inst):
    return type(inst).__name__ in _LR_CLASS_NAMES


def _is_lw(inst):
    return type(inst).__name__ in _LW_CLASS_NAMES


def _is_gr(inst):
    return type(inst).__name__ in _GR_CLASS_NAMES


def _is_mfma(inst):
    return type(inst).__name__ in _MFMA_CLASS_NAMES


def _is_swait(inst):
    return type(inst).__name__ in _SWAIT_CLASS_NAMES


def _is_sbarrier(inst):
    return type(inst).__name__ in _SBARRIER_CLASS_NAMES


def _is_snop(inst):
    return type(inst).__name__ in _SNOP_CLASS_NAMES


def _reg_signature(reg) -> tuple:
    """Stable hashable tuple summary of a RegisterContainer.

    Real CMS schedules use SYMBOLIC register names (regIdx=-1, regName
    holds the actual identity like 'ValuA_X0_I0'). Including only
    (regType, regIdx, regNum) would collapse all symbolic registers to
    the same signature and corrupt graph identity. Include the regName's
    name + offsets when present so symbolic-named registers are
    distinguishable.
    """
    if reg is None:
        return ()
    name_sig = ()
    rname = getattr(reg, "regName", None)
    if rname is not None:
        try:
            offs = tuple(rname.getOffsets()) if hasattr(rname, "getOffsets") else ()
        except Exception:
            offs = ()
        name_sig = (getattr(rname, "name", None), offs)
    return (getattr(reg, "regType", None),
            getattr(reg, "regIdx", None),
            getattr(reg, "regNum", None),
            name_sig)


def _get_param(inst, idx, default=None):
    """Read positional constructor param `idx` from a rocisa instruction.

    Real rocisa instances expose constructor args via getParams() (an
    InstructionInputVector) rather than as named attributes. This helper
    bridges the synthetic-fixture and real-rocisa shapes.
    """
    if not hasattr(inst, "getParams"):
        return default
    try:
        params = inst.getParams()
    except Exception:
        return default
    try:
        return params[idx]
    except (IndexError, TypeError):
        return default


def _inst_dst(inst):
    """Return the destination register for LR/LW/GR — try named attr first,
    then positional constructor param 0 (matches DSLoad*/BufferLoad*/
    GlobalLoad*/DSStore* constructors)."""
    dst = getattr(inst, "dst", None)
    if dst is not None:
        return dst
    return _get_param(inst, 0)


def _inst_lds_offset(inst):
    """Return the LDS offset for LR/LW. Synthetic fixture exposes
    `lds_offset`; real rocisa stores it inside DSModifiers (the 3rd
    constructor arg for DSLoad*). For the identity tuple we use
    str(modifier) so two LDS-ops with identical offsets get equal sigs."""
    off = getattr(inst, "lds_offset", None)
    if off is not None:
        return off
    ds_mods = _get_param(inst, 2)
    if ds_mods is None:
        return None
    return str(ds_mods)


def _inst_buffer_srd(inst):
    """Return the SRD sgpr for a BufferLoad — synthetic 'srd' attr or
    constructor param 2 (saddr)."""
    srd = getattr(inst, "srd", None)
    if srd is not None:
        return srd
    return _get_param(inst, 2)


def _inst_buffer_offset(inst):
    """Return the immediate offset for a BufferLoad — synthetic
    'immediate_offset' attr or constructor param 3 (soffset)."""
    off = getattr(inst, "immediate_offset", None)
    if off is not None:
        return off
    return _get_param(inst, 3)


def _inst_mfma_acc(inst):
    """Return MFMA accumulator/c_dst register. Synthetic uses 'c_dst';
    real rocisa MFMA's getParams() returns [acc, a, b, acc_or_d, comment]
    — acc is at index 0, NOT 4 (the param-list contains the input/output
    registers, not the full constructor args)."""
    for attr in ("c_dst", "acc"):
        v = getattr(inst, attr, None)
        if v is not None:
            return v
    return _get_param(inst, 0)


def _inst_mfma_a(inst):
    for attr in ("a_src", "a"):
        v = getattr(inst, attr, None)
        if v is not None:
            return v
    return _get_param(inst, 1)


def _inst_mfma_b(inst):
    for attr in ("b_src", "b"):
        v = getattr(inst, attr, None)
        if v is not None:
            return v
    return _get_param(inst, 2)


def _inst_dsstore_src(inst):
    """LW (DSStore) source — synthetic 'src' or constructor param 1
    (DSStore signature is (dst_lds, src_vgpr, ds_mods, comment))."""
    src = getattr(inst, "src", None)
    if src is not None:
        return src
    return _get_param(inst, 1)


_COMMENT_STRIP_RE = None  # lazy-compiled


def _canonical_render(inst) -> str:
    """Return a normalized render-string for an instruction.

    Used as the identity-defining payload so the comparison is robust to
    register-naming differences (symbolic / numeric / mixed). Two
    instructions producing the same canonical render-string represent
    the same GPU operation, regardless of how their constructor was
    invoked.

    Normalizations applied:
      - strip trailing comment ('// ...')
      - strip leading/trailing whitespace
      - collapse runs of whitespace to a single space

    Symbolic registers (vgpr("ValuA_X0_I0", 4)) render as
    'v[vgprValuA_X0_I0:vgprValuA_X0_I0+3]'; numeric registers
    (vgpr(8, 4)) render as 'v[8:11]'. The same instruction emitted by
    two different code paths in the SAME kernel writer build will have
    identical renders because both paths consume the same writer state.
    Cross-kernel comparison would still differ on numeric allocation
    but that's not a use case here (compare_graphs operates within one
    build).
    """
    global _COMMENT_STRIP_RE
    if _COMMENT_STRIP_RE is None:
        import re as _re
        _COMMENT_STRIP_RE = _re.compile(r"//.*$", _re.MULTILINE)
    s = str(inst)
    s = _COMMENT_STRIP_RE.sub("", s).strip()
    # Collapse internal whitespace to single spaces for consistent matching
    return " ".join(s.split())


def _class_tag(inst) -> str:
    """Return the stable class tag (LR/LW/GR/MFMA/SWAIT/SBARRIER) for an
    instruction. Used as the first element of the identity tuple so
    diagnostic categorization works without parsing the render-string.
    """
    if _is_lr(inst):
        return "LR"
    if _is_lw(inst):
        return "LW"
    if _is_gr(inst):
        return "GR"
    if _is_mfma(inst):
        return "MFMA"
    if _is_swait(inst):
        return "SWAIT"
    if _is_sbarrier(inst):
        return "SBARRIER"
    raise CaptureUnknownInstructionError(
        f"_class_tag: cannot classify instruction class "
        f"{type(inst).__name__!r}."
    )


def _class_tag_from_category(category, inst) -> str:
    """Like _class_tag(inst) but consults TaggedInstruction.category first.

    The pure isinstance path is wrong for instructions whose Python class
    doesn't reflect their scheduler role: F32X TF32 emulation MFMAs in the
    pack path are real MFMAInstruction objects but are categorized as
    PackA{u}/PackB{u}. Treating them as cls='MFMA' in the identity tuple
    causes them to appear as missing main-loop MFMAs in compare_graphs
    when the two captures see different counts of pack-MFMAs.

    Maps categories to scheduler-role tags so cross-capture comparison
    discriminates pack-MFMAs from real MFMAs.

    Falls back to _class_tag(inst) when category is None or unrecognized
    so test sites that pass bare insts (no TaggedInstruction wrapping)
    keep working.
    """
    if category is None:
        return _class_tag(inst)
    # Per-tensor / per-iteration suffixes -> scheduler-role tag.
    if category.startswith(("LRA", "LRB", "LRMXSA", "LRMXSB", "LRMetadata")):
        return "LR"
    if category.startswith("LRS"):
        return "LRS"
    if category.startswith("LWS"):
        return "LWS"
    if category.startswith("LW"):
        return "LW"
    if category.startswith("GRInc"):
        return "GRINC"
    if category.startswith("GR"):
        return "GR"
    if category.startswith("Pack"):
        return "PACK"
    if category == "LCC":
        return "LCC"
    if category == "SYNC":
        # _captureSubIterToBuilder lumps SWaitCnt AND SBarrier into category
        # "SYNC", so we must distinguish them here by class. Without this,
        # an SBarrier would render as cls='SWAIT' and never match a real
        # SBARRIER identity in the other graph.
        return _class_tag(inst)
    if category == "SNOP":
        return "SNOP"
    if category == "BARRIER":
        return "SBARRIER"
    if category == "MFMA":
        return "MFMA"
    # Unrecognized category (e.g. UNKNOWN) -> fall back to isinstance.
    return _class_tag(inst)


# =============================================================================
# --- Reorder-invariant logical position ---
# =============================================================================
# Edge formation must be reorder-invariant: two captures of the same instruction
# set should produce the same edges, regardless of how each scheduler placed
# the instructions in the stream. The current FIFO-simulation model isn't
# invariant — ready_writer's last-write-wins is sensitive to the order in
# which producers drain. Instead, derive a "logical position" per node from
# (body, iter, kind, sequence) where:
#
#   - body comes from BODY_LABEL_TO_LOOP_INDEX.
#   - iter comes from the category suffix (LRA0 -> 0, PackB3 -> 3) for
#     LR/Pack/MX*; from mfma_index // numMfmaPerIter for MFMA; 0 otherwise.
#   - kind_rank is a small finite map of category-base -> integer.
#   - sequence is the node's index within its (body, category) cohort —
#     used as a tiebreaker; mostly irrelevant when each producer writes a
#     unique register sub-range (the typical Tensile pattern).
#
# Every component is preserved across schedulers (they all derive from the
# kernel writer's idMap/category structure, NOT from stream position).

import re as _re
_TRAILING_DIGITS_RE = _re.compile(r"^(.*?)(\d*)$")

# kind_rank assigns every category-base to a small integer. Order matters
# only insofar as it determines which producer is "more recent" when two
# share the same iter — typically Pack > LR is the natural pipeline order.
# MFMA (rank 7) is encoded specially: its iter comes from mfma_index, not
# from a category suffix.
_KIND_RANK = {
    # local reads (iter-suffixed)
    "LRA":         0,
    "LRB":         0,
    "LRMXSA":      0,
    "LRMXSB":      0,
    "LRMetadata":  0,
    # local read pointer ops (no iter suffix)
    "LRSA":        1,
    "LRSB":        1,
    "LRS":         1,
    # local writes (no iter suffix)
    "LWA":         2,
    "LWB":         2,
    # local write pointer ops
    "LWSA":        3,
    "LWSB":        3,
    "LWS":         3,
    # global reads (no iter suffix)
    "GRA":         4,
    "GRB":         4,
    "GR":          4,
    "GRIncA":      4,
    "GRIncB":      4,
    # pack ops (iter-suffixed)
    "PackA":       5,
    "PackB":       5,
    # MFMA handled specially in _logical_position; rank 6 used for it.
    # SYNC / SBARRIER / SNOP / LCC don't get logical positions (they're
    # excluded from the cross-graph identity set, hence excluded from
    # edges).
}

_MFMA_KIND_RANK = 6


def _split_category_iter(category):
    """Split 'LRA0' -> ('LRA', 0), 'PackB3' -> ('PackB', 3), 'GRA' -> ('GRA', 0).

    Trailing digits become the iteration index; everything before is the
    base category name. Categories with no trailing digits (e.g. GRA, LWA)
    return iter=0.
    """
    m = _TRAILING_DIGITS_RE.match(category)
    base, suffix = m.group(1), m.group(2)
    return base, (int(suffix) if suffix else 0)


def _logical_position(node, num_mfma_per_iter):
    """Return the reorder-invariant logical position for `node`.

    Format: (body_loop_index, iter, kind_rank, intra_seq).

    intra_seq is the node's offset within its body's _graph_nodes list —
    used only as a tiebreaker for the (typically rare) case where two
    producers of the same kind write overlapping registers in the same
    iteration. For most Tensile patterns each producer writes a unique
    register sub-range, so resolution succeeds before intra_seq matters.

    Raises if the node's category doesn't map to a known kind. SYNC /
    SBARRIER / SNOP / LCC are excluded from edge formation upstream so
    this function isn't called for them.
    """
    body_idx = node.position.loop_index
    cat = node.category
    if cat == "MFMA":
        idx = node.position.vmfma_index
        if num_mfma_per_iter and num_mfma_per_iter > 0:
            iter_ = idx // num_mfma_per_iter
            within = idx % num_mfma_per_iter
        else:
            iter_, within = 0, idx
        return (body_idx, iter_, _MFMA_KIND_RANK, within)
    base, iter_ = _split_category_iter(cat)
    rank = _KIND_RANK.get(base)
    if rank is None:
        raise CaptureUnknownInstructionError(
            f"_logical_position: category {cat!r} (base {base!r}) has no "
            f"kind_rank entry; node={node.name!r} body={node.body_label!r}."
        )
    # intra_seq is the node's offset in its body's _graph_nodes list,
    # filled in by the caller (build_dataflow_graph) since it's body-relative.
    return (body_idx, iter_, rank, getattr(node, '_intra_seq', 0))


def _resolve_register_producers(read_reg, consumer, producers_by_kind, num_mfma_per_iter):
    """Resolve every producer of `read_reg` for `consumer`.

    Yields (producer_node, written_reg) pairs for each producer whose
    logical_position is strictly before consumer's AND whose written
    register overlaps `read_reg`. A wide read (e.g. MFMA reading
    v[8:15]) covered by two narrower writes (LR_a writes v[8:11],
    LR_b writes v[12:15]) yields BOTH edges — both producers
    genuinely contribute data the consumer reads.

    Reorder-invariant: each yielded pair depends only on register
    identity + category + iteration, all preserved across schedulers.
    Both schedulers see the same instruction set so both yield the
    same set of (producer, register) pairs.
    """
    consumer_lp = _logical_position(consumer, num_mfma_per_iter)
    for kind, prod_list in producers_by_kind.items():
        for lp, prod_node, written_regs in prod_list:
            if lp >= consumer_lp:
                # prod_list is sorted ascending; nothing later will be < consumer_lp.
                break
            for wreg in written_regs:
                if _reg_overlaps(read_reg, wreg):
                    yield (prod_node, wreg)
                    break  # this producer matched once; one edge per producer per read


def _identity_for(inst, body_label: str, category=None) -> tuple:
    """Build a content-based identity tuple for an instruction.

    Format: (class_tag, loop_index, canonical_render).

    Render-string identity (rather than a per-class structured signature
    of register fields) makes the comparison robust to register-naming
    variations: an MFMA emitted as
        v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+0:...], v[74:75], ...
    has a stable identity regardless of whether the schedulers happen
    to spell its inputs symbolically, numerically, or mixed — the
    rendered assembly is what the GPU sees, and that's what we compare
    on.

    class_tag (LR/LW/GR/MFMA/SWAIT/SBARRIER/PACK/...) is preserved as the
    first element so the identity-mismatch diagnostic in compare_graphs can
    still categorize differences by kind.

    `category` (TaggedInstruction.category) is consulted first when
    provided; this prevents pack-MFMAs (TF32 emulation MFMAInstruction
    objects categorized as PackA{u}/PackB{u}) from masquerading as
    main-loop MFMAs in the identity tuple. When omitted, falls back to
    pure isinstance-based classification — preserves existing test sites
    that synthesize bare insts.

    Raises CaptureUnknownInstructionError when an instruction class
    isn't one of the recognized kinds AND category is None.
    """
    loop_idx = BODY_LABEL_TO_LOOP_INDEX[body_label]
    cls_tag = _class_tag_from_category(category, inst)
    return (cls_tag, loop_idx, _canonical_render(inst))


def _writes(inst):
    """Registers written by this instruction (returned as a list of RegisterContainers)."""
    if _is_lr(inst):
        dst = _inst_dst(inst)
        return [dst] if dst is not None else []
    if _is_gr(inst):
        dst = _inst_dst(inst)
        return [dst] if dst is not None else []
    return []


def _reads(inst):
    """Registers read by this instruction."""
    if _is_lw(inst):
        src = _inst_dsstore_src(inst)
        return [src] if src is not None else []
    if _is_mfma(inst):
        out = []
        for v in (_inst_mfma_a(inst), _inst_mfma_b(inst), _inst_mfma_acc(inst)):
            if v is not None:
                out.append(v)
        return out
    return []


def _reg_overlaps(read_reg, written_reg) -> bool:
    """True if `read_reg` overlaps with `written_reg` (vgpr ranges intersect).

    Handles three cases:
      - Numeric registers (regIdx >= 0): overlap by numeric range.
      - Symbolic registers (regIdx == -1, regName set): overlap requires
        same regName.name AND overlapping range using getTotalIdx() (the
        offset within the named region).
      - Mixed numeric + symbolic: NEVER overlap — different naming
        conventions can't be compared without a name-resolution table.
    """
    if read_reg is None or written_reg is None:
        return False
    if read_reg.regType != written_reg.regType:
        return False

    a_named = read_reg.regIdx == -1 and getattr(read_reg, "regName", None) is not None
    b_named = written_reg.regIdx == -1 and getattr(written_reg, "regName", None) is not None

    if a_named != b_named:
        # Mixed naming convention; can't compare safely.
        return False

    if a_named:
        # Both symbolic — must share the same name root, then compare offsets.
        if read_reg.regName.name != written_reg.regName.name:
            return False
        a_off = read_reg.regName.getTotalOffsets() if hasattr(read_reg.regName, "getTotalOffsets") else 0
        b_off = written_reg.regName.getTotalOffsets() if hasattr(written_reg.regName, "getTotalOffsets") else 0
        a_lo = a_off
        a_hi = a_lo + (read_reg.regNum or 1)
        b_lo = b_off
        b_hi = b_lo + (written_reg.regNum or 1)
        return a_lo < b_hi and b_lo < a_hi

    # Both numeric.
    a_lo = read_reg.regIdx
    a_hi = a_lo + (read_reg.regNum or 1)
    b_lo = written_reg.regIdx
    b_hi = b_lo + (written_reg.regNum or 1)
    return a_lo < b_hi and b_lo < a_hi


def _make_node(tagged_inst, body_label: str) -> GraphNode:
    inst = tagged_inst.inst
    identity = _identity_for(inst, body_label, category=tagged_inst.category)
    position = make_position(body_label, tagged_inst.slot)
    name = f"{tagged_inst.category}@{position.vmfma_index}.{position.sub_index}"
    return GraphNode(
        identity=identity,
        position=position,
        category=tagged_inst.category,
        rocisa_inst=inst,
        tagged_inst=tagged_inst,
        body_label=body_label,
        name=name,
    )


# Body order for graph construction. Cross-body queue state persists in the
# order ML-1 -> ML -> NGL -> NLL (matching hardware execution order).
_BODY_BUILD_ORDER = (BODY_LABEL_ML_PREV, BODY_LABEL_ML, BODY_LABEL_NGL, BODY_LABEL_NLL)


def build_dataflow_graph(four_part_capture):
    """Build the unified 4-body register dataflow graph from a FourPartCapture.

    Two phases:

    Phase 1 — node construction. Walks bodies in execution order
    (ML-1 -> ML -> NGL -> NLL). Every captured instruction becomes a
    node EXCEPT SWait/SBarrier/SNop (scheduler-choice; sidecar only)
    and LCC (out-of-scope: default emits it outside _loopBody, CMS
    bakes it into the macro). Per-body sidecar `_graph_nodes` is
    attached so wait/barrier helpers can find sync ops in stream order.

    Phase 2 — edge formation by REGISTER-NAME RESOLUTION (reorder-
    invariant). For each consumer's read register R, find the unique
    producer P that writes R. Producer is the latest in LOGICAL ORDER
    (body, iter, kind_rank, intra_seq) whose written register overlaps
    R. Logical order is derived from category + mfma_index — preserved
    by both schedulers — so two captures of the same instruction set
    in different schedules produce IDENTICAL edges.

    A separate barrier-edge collector (`_collect_barrier_edges`) emits
    LDS-reuse edges (lr_to_gr_lds_reuse, gr_to_lr_lds_reuse) over the
    unified node stream — same as before; that collector is already
    pattern-based and reorder-invariant.

    Wait-coverage validation lives elsewhere (see
    `validate_edge_wait_coverage` and `diagnose_missing_edge`) — those
    take the constructed graph and check, per-edge, whether CMS's
    stream has a covering SWaitCnt that drains the producer.

    Missed-instruction guard: an instruction whose category resolves to
    no recognized scheduler-role tag AND whose Python class isn't in
    LR/LW/GR/MFMA/SWait/SBarrier raises CaptureUnknownInstructionError.

    Always raises CaptureEmptyBodyError if any body has zero
    instructions.
    """
    captures = {}
    if four_part_capture is None:
        return DataflowGraph(nodes={}, edges=[], captures=captures)

    # Seed captures dict and validate bodies are non-empty.
    body_sources = (
        (BODY_LABEL_ML_PREV, four_part_capture.main_loop_prev),
        (BODY_LABEL_ML, four_part_capture.main_loop),
        (BODY_LABEL_NGL, four_part_capture.n_gl),
        (BODY_LABEL_NLL, four_part_capture.n_ll),
    )
    for label, by_cp in body_sources:
        if 0 not in by_cp:
            continue
        body = by_cp[0]
        if not body.instructions:
            raise CaptureEmptyBodyError(
                f"Body {label!r} has zero captured instructions; "
                f"bodies always contain at least the MFMA loop."
            )
        captures[label] = body

    num_mfma_per_iter = getattr(four_part_capture, 'num_mfma_per_iter', 0) or 0

    nodes_by_identity = {}
    nodes_per_body = {label: [] for label in _BODY_BUILD_ORDER}

    # ---------------------------------------------------------------------
    # Phase 1 — node construction + sidecar.
    # ---------------------------------------------------------------------
    # Per-body intra_seq counters: keyed on category, used for the
    # logical-position tiebreaker. The same item-within-category order is
    # preserved by both schedulers because they consume from the same idMap
    # source modules; intra_seq based on per-body cohort index works.
    for label in _BODY_BUILD_ORDER:
        if label not in captures:
            continue
        body = captures[label]
        intra_seq_counter = {}  # category -> next sequence

        for tagged_inst in body.instructions:
            inst = tagged_inst.inst
            try:
                node = _make_node(tagged_inst, label)
            except CaptureUnknownInstructionError as e:
                raise CaptureUnknownInstructionError(
                    f"build_dataflow_graph: cannot classify instruction "
                    f"{type(inst).__name__!r} (category={tagged_inst.category!r}) "
                    f"in body {label!r}. The capture pipeline must assign a "
                    f"recognized category, or the instruction's class must be "
                    f"one of LR/LW/GR/MFMA/SWait/SBarrier. Inner: {e}"
                ) from e

            # Stamp intra_seq for logical-position tiebreaking.
            cat = tagged_inst.category
            seq = intra_seq_counter.get(cat, 0)
            intra_seq_counter[cat] = seq + 1
            node._intra_seq = seq

            # Per-body sidecar: every node lives here, including SWait/
            # SBarrier/SNop, so waits_in_window/barriers_in_window can
            # find them.
            nodes_per_body[label].append(node)

            # Cross-graph identity set: only "real" instructions
            # participate (excludes scheduler-choice SWait/SBarrier/SNop
            # and out-of-scope LCC).
            if (not (_is_swait(inst) or _is_sbarrier(inst) or _is_snop(inst))
                    and node.identity[0] != "LCC"):
                nodes_by_identity[node.identity] = node

        # Stash per-body GraphNodes on the LoopBodyCapture for the helpers.
        body._graph_nodes = nodes_per_body[label]

    # ---------------------------------------------------------------------
    # Phase 2 — edge formation by register-name resolution.
    # ---------------------------------------------------------------------
    # Collect every producer node (anything _writes() returns regs for)
    # bucketed by kind_rank, sorted by logical_position. Then walk every
    # consumer node, resolve each read register to its unique producer
    # via _resolve_register_producer, emit the edge.
    edges = []

    # Skip when nothing was captured — e.g., the no-op build_dataflow_graph(None)
    # contract holds but here we have an empty captures map after seeding.
    if nodes_by_identity:
        producers_by_kind = {}  # kind_rank -> list of (lp, node, written_regs)
        for node in nodes_by_identity.values():
            written = _writes(node.rocisa_inst)
            if not written:
                continue
            try:
                lp = _logical_position(node, num_mfma_per_iter)
            except CaptureUnknownInstructionError:
                # Producer's category not in _KIND_RANK — skip (can't
                # participate in dataflow without a logical position).
                continue
            producers_by_kind.setdefault(lp[2], []).append((lp, node, written))

        # Sort each kind's producer list by logical_position ascending so
        # _resolve_register_producer can walk-from-end to find the latest.
        for kind in producers_by_kind:
            producers_by_kind[kind].sort(key=lambda triple: triple[0])

        # Resolve each consumer's reads.
        for node in nodes_by_identity.values():
            reads = _reads(node.rocisa_inst)
            if not reads:
                continue
            try:
                # _logical_position is required for the resolver's
                # ordering check — skip if consumer's kind isn't ranked.
                _logical_position(node, num_mfma_per_iter)
            except CaptureUnknownInstructionError:
                continue
            for read_reg in reads:
                if read_reg is None:
                    continue
                for producer, written_reg in _resolve_register_producers(
                    read_reg, node, producers_by_kind, num_mfma_per_iter,
                ):
                    edges.append(DataflowEdge(
                        producer=producer,
                        consumer=node,
                        register=written_reg,
                        edge_kind="raw_intrawave",
                    ))

    # =========================================================================
    # SBarrier-edge collectors (cross-wave LDS-reuse)
    # =========================================================================
    # Two patterns mirror CMSValidator.apply_must_start_after_barriers (line 1216)
    # and apply_barriers (line 1195):
    #
    #   lr_to_gr_lds_reuse  (must_start_after):
    #     Producer LR0/LR1 -> SWaitCnt(dscnt drain) -> SBarrier -> Consumer GR
    #
    #   gr_to_lr_lds_reuse  (needed_by):
    #     Producer GR -> SWaitCnt(vlcnt drain) -> SBarrier -> Consumer LR1/LR3
    #
    # Both demand strict ordering: the SWait must precede the SBarrier; SWait
    # alone (no barrier) means cross-wave coherence isn't established; SBarrier
    # alone (no wait) means the in-wave counter never drained.
    #
    # We collect across the unified node stream (all bodies in execution order)
    # so cross-body patterns (DTL+LdsBuf: LR0 in ML-1 + GR in ML) form
    # naturally — the producer's body_label and consumer's body_label may
    # differ on the resulting DataflowEdge.

    all_nodes_in_order = []
    for label in _BODY_BUILD_ORDER:
        all_nodes_in_order.extend(nodes_per_body[label])

    barrier_edges = _collect_barrier_edges(all_nodes_in_order)
    edges.extend(barrier_edges)

    return DataflowGraph(nodes=nodes_by_identity, edges=edges, captures=captures)


def _collect_barrier_edges(nodes_in_order):
    """Walk the unified node stream once and emit SBarrier-pattern edges.

    Returns a list of DataflowEdges with edge_kind in
    {'lr_to_gr_lds_reuse', 'gr_to_lr_lds_reuse'}.

    Algorithm: for each pair (producer_kind, counter, consumer_kind, edge_kind):
      For each producer node (in stream order):
        Walk forward looking for SWaitCnt that drains `counter`.
        Once found, walk further forward looking for SBarrier.
        Once both found in correct order, every consumer node of `consumer_kind`
        appearing after the SBarrier (until a NEW producer of producer_kind is
        seen, which restarts the pattern) becomes the edge target.
    """
    out = []

    # Build per-kind node lists.
    lr_categories = {"LRA0", "LRA1", "LRA3", "LRB0", "LRB1", "LRB3"}
    gr_categories = {"GRA", "GRB", "GR"}

    # ---------- Pattern 1: LR -> SWait(dscnt) -> SBarrier -> GR ----------
    out.extend(_collect_pattern(
        nodes_in_order,
        producer_categories=lr_categories,
        consumer_categories=gr_categories,
        counter="dscnt",
        edge_kind="lr_to_gr_lds_reuse",
    ))

    # ---------- Pattern 2: GR -> SWait(vlcnt) -> SBarrier -> LR ----------
    # Consumer is LR (any LR* category — typically LR1 or LR3 in CMS).
    out.extend(_collect_pattern(
        nodes_in_order,
        producer_categories=gr_categories,
        consumer_categories=lr_categories,
        counter="vlcnt",
        edge_kind="gr_to_lr_lds_reuse",
    ))

    return out


def _collect_pattern(nodes_in_order, *, producer_categories, consumer_categories,
                     counter, edge_kind):
    """Sweep nodes_in_order and emit edges where the producer/SWait/SBarrier/
    consumer ordering invariant holds.

    State machine per producer:
      0. Producer seen -> remember it.
      1. Find SWaitCnt with `counter` drain after the producer.
      2. Find SBarrier strictly after the SWait.
      3. Every consumer of `consumer_categories` strictly after the SBarrier
         becomes an edge target — until either:
           - a new producer of `producer_categories` resets the pattern
             (its own pending edges will be collected on the next iteration),
           - or stream ends.
    """
    edges = []

    # We do an O(N^2) sweep — for each producer, scan forward. Body sizes are
    # at most a few hundred instructions; this is comfortably fast.
    for i, producer in enumerate(nodes_in_order):
        if producer.category not in producer_categories:
            continue

        wait_idx = None
        barrier_idx = None
        for j in range(i + 1, len(nodes_in_order)):
            node = nodes_in_order[j]

            # If we hit another producer of the same kind before completing
            # the pattern, this producer's pattern remains unfinished — but
            # the new producer's pattern will be collected on its own iteration.
            # We don't break (the new producer can still share the wait/barrier
            # if they appear after both producers).

            if wait_idx is None:
                if node.category == SWAIT_CATEGORY and \
                        _swait_drains(node, counter) is not None:
                    wait_idx = j
                continue

            if barrier_idx is None:
                if node.category == SBARRIER_CATEGORY:
                    barrier_idx = j
                # If a new SWait appears, prefer the latest (more aggressive
                # drain). Don't change wait_idx because a later wait still
                # drains earlier producers — but the FIRST wait/barrier pair
                # already establishes the invariant.
                continue

            # We have both wait_idx and barrier_idx. Now any consumer of
            # consumer_categories at j > barrier_idx becomes an edge.
            if node.category in consumer_categories:
                # Determine which register the producer "passed" to the consumer.
                # For LDS-reuse patterns, the resource is an LDS slot; we
                # represent it via the producer's written register signature
                # (or the GR's destination, which IS the LDS slot under DTL).
                if edge_kind == "lr_to_gr_lds_reuse":
                    # Producer LR -> destination vgpr; consumer GR -> destination
                    # vgpr (under DTL=1, that vgpr is bound to the same LDS slot).
                    # We tag the edge with the producer's destination register
                    # since that's the resource pin.
                    register = getattr(producer.rocisa_inst, "dst", None)
                else:  # gr_to_lr_lds_reuse
                    register = getattr(producer.rocisa_inst, "dst", None)

                edges.append(DataflowEdge(
                    producer=producer,
                    consumer=node,
                    register=register,
                    edge_kind=edge_kind,
                ))

            # Pattern reset: a NEW producer of producer_categories ends this
            # producer's "passing window". The new producer starts fresh.
            if node.category in producer_categories:
                break

    return edges


# =============================================================================
# Cross-graph comparison + diagnostic classifier
# =============================================================================
#
# compare_graphs takes the default-side (reference) and CMS-side (subject)
# graphs and returns a list of typed Failures explaining every reference
# edge that's missing from the subject graph.
#
# diagnose_missing_edge classifies a single missing edge into one of:
#   OrderInvertedFailure       — same-body producer position > consumer position
#   MissingWaitFailure         — no SWait on the right counter in the window
#   WaitOnWrongCounterFailure  — SWait exists but drains the wrong counter
#   WaitInsufficientFailure    — SWait counter value too lax
#   MissingBarrierFailure      — wait covers but no barrier in window (LDS-reuse only)
#
# The classifier emits Failures the user fixes by editing their CMS schedule;
# capture-pipeline bugs (missing nodes, identity mismatches) are caught as
# CaptureConsistencyError BEFORE comparison runs.


def compare_graphs(reference: DataflowGraph, subject: DataflowGraph,
                   *, raise_on_unexplained=True) -> list:
    """Compare two dataflow graphs as edge sets keyed on
    (producer.identity, consumer.identity, register, edge_kind).

    Returns a list of Failure objects — one or more per missing edge,
    routed through diagnose_missing_edge.

    Raises CaptureConsistencyError BEFORE comparison if the two graphs'
    DATA-FLOW node identity sets differ — a capture-pipeline bug, not a
    CMS schedule defect.

    `raise_on_unexplained` propagates to diagnose_missing_edge — soft mode
    (False) is intended for production observability so unclassified
    misses don't crash the build.
    """
    # Identity-coverage check at entry, restricted to DATA-FLOW nodes
    # (LR/LW/GR/MFMA). CMS legitimately adds/removes scheduling control
    # flow (SWait, SBarrier, SNop) — those identity differences are NOT
    # capture-pipeline bugs. The check guards against the only true
    # capture-pipeline failure mode: a producer or consumer present in
    # one capture but missing from the other.
    _DATA_FLOW_KINDS = ("LR", "LW", "GR", "MFMA")

    def _data_flow_ids(graph):
        return {k for k in graph.nodes.keys() if k and k[0] in _DATA_FLOW_KINDS}

    ref_ids = _data_flow_ids(reference)
    subj_ids = _data_flow_ids(subject)
    if ref_ids != subj_ids:
        only_ref = ref_ids - subj_ids
        only_subj = subj_ids - ref_ids
        # Categorize the diff by class_tag (LR/LW/GR/MFMA) to make the
        # error actionable. The full identity tuple list is too long for
        # a single error string when 16+ identities differ.
        def _summary_by_class(ids):
            counts = {}
            for ident in ids:
                cls_tag = ident[0] if ident else "?"
                counts[cls_tag] = counts.get(cls_tag, 0) + 1
            return counts
        msg_parts = []
        if only_ref:
            counts = _summary_by_class(only_ref)
            msg_parts.append(
                f"in reference but not subject: {len(only_ref)} identities "
                f"({counts}); first 3: {sorted(only_ref)[:3]}"
            )
        if only_subj:
            counts = _summary_by_class(only_subj)
            msg_parts.append(
                f"in subject but not reference: {len(only_subj)} identities "
                f"({counts}); first 3: {sorted(only_subj)[:3]}"
            )
        raise CaptureConsistencyError(
            "compare_graphs: data-flow node identity sets differ. "
            + "; ".join(msg_parts)
        )

    ref_keys = reference.edge_keys()
    subj_keys = subject.edge_keys()
    missing_keys = ref_keys - subj_keys

    # Map missing keys back to reference edge objects for diagnosis.
    failures = []
    ref_edges_by_key = {
        (e.producer.identity, e.consumer.identity, e.register, e.edge_kind): e
        for e in reference.edges
    }
    for key in missing_keys:
        ref_edge = ref_edges_by_key[key]
        failures.extend(diagnose_missing_edge(
            ref_edge, subject, raise_on_unexplained=raise_on_unexplained,
        ))
    return failures


def diagnose_missing_edge(ref_edge: DataflowEdge, subj_graph: DataflowGraph,
                          *, raise_on_unexplained=True) -> list:
    """Classify why a reference edge is absent from the CMS subject graph.

    See plan §"Comparison and diagnosis" for the phased classifier:
      Phase 0: identity lookup (gating — missing nodes raise).
      Phase 1: OrderInvertedFailure (same-body only — gating for Phase 2).
      Phase 2: MissingWaitFailure / WaitOnWrongCounterFailure /
               WaitInsufficientFailure (mutually exclusive); plus
               MissingBarrierFailure when a wait covers but no barrier
               sits in the post-wait window (LDS-reuse edges only).

    `raise_on_unexplained=True` (default) raises UnexplainedMissingEdgeError
    when the classifier reaches a fall-through — used in unit tests to
    catch classifier regressions. `raise_on_unexplained=False` is used
    by production observability paths that prefer a soft Failure return
    over a hard exception.
    """
    p_id = ref_edge.producer.identity
    c_id = ref_edge.consumer.identity
    p_node = subj_graph.nodes.get(p_id)
    c_node = subj_graph.nodes.get(c_id)

    # Phase 0 — gating. Missing nodes: raise (not assert; survive python -O).
    if p_node is None or c_node is None:
        raise CaptureConsistencyError(
            f"diagnose_missing_edge invoked with missing node — "
            f"identity-coverage check at compare_graphs entry was bypassed. "
            f"p_id={p_id} (found={p_node is not None}), "
            f"c_id={c_id} (found={c_node is not None})."
        )

    # Phase 1 — gating: order check (same-body only).
    if p_node.body_label == c_node.body_label and p_node.position > c_node.position:
        return [OrderInvertedFailure(producer=p_node, consumer=c_node)]

    # Phase 2 — independent checks. Run all; collect failures.
    # All wait/barrier helpers walk subj_graph cross-body: producer in
    # body=ML-1 with consumer in body=ML must see the FIFO state from
    # body=ML-1 forward. Passing only the consumer's body capture would
    # exclude the producer from the simulated queue and mis-classify
    # cross-body edges.
    failures: list = []

    # Determine the counter that the producer requires.
    expected_counter = counter_for(p_node)

    # Look at SWaits in the window between producer.position and consumer.position.
    waits = waits_in_window(subj_graph, p_node.position, c_node.position,
                            counter=expected_counter)
    waits_other = waits_in_window(subj_graph, p_node.position, c_node.position,
                                  exclude_counter=expected_counter)

    wait_failure_emitted = False

    if not waits:
        # No SWait on the expected counter at all in the window.
        if waits_other:
            failures.append(WaitOnWrongCounterFailure(
                producer=p_node,
                consumer=c_node,
                expected_counter=expected_counter,
                wrong_counter_waits=waits_other,
            ))
        else:
            failures.append(MissingWaitFailure(
                producer=p_node,
                consumer=c_node,
                counter_kind=expected_counter,
            ))
        wait_failure_emitted = True
    else:
        # At least one wait on the right counter. Check if any drains the producer.
        if not _any_drains(waits, p_node, subj_graph):
            insufficient = _first_insufficient(waits, p_node, subj_graph)
            if insufficient is not None:
                # Compute queue depth at the wait's position for diagnostic.
                depth = _queue_depth_at(insufficient, p_node, subj_graph)
                cv = _swait_drains(insufficient, expected_counter)
                failures.append(WaitInsufficientFailure(
                    producer=p_node,
                    consumer=c_node,
                    wait=insufficient,
                    queue_depth_at_wait=depth,
                    counter_value=cv if cv is not None else 0,
                ))
                wait_failure_emitted = True
            else:
                # waits exist on the right counter but none drains the producer.
                # Treat as MissingWait — every wait fired before the producer
                # entered the queue (or the producer is positioned after every
                # wait we found).
                failures.append(MissingWaitFailure(
                    producer=p_node,
                    consumer=c_node,
                    counter_kind=expected_counter,
                ))
                wait_failure_emitted = True

    # Barrier check is meaningful ONLY when a covering wait actually drains
    # the producer. If wait_failure_emitted, suppress MissingBarrier — the
    # user's wait fix will cascade-restore barrier semantics on the next build.
    if (ref_edge.edge_kind in ("lr_to_gr_lds_reuse", "gr_to_lr_lds_reuse")
            and not wait_failure_emitted):
        last_drain = _last_drain(waits, p_node, subj_graph)
        if last_drain is not None:
            barriers = barriers_in_window(subj_graph,
                                          start=last_drain.position,
                                          end=c_node.position)
            if not barriers:
                role = ("must_start_after"
                        if ref_edge.edge_kind == "lr_to_gr_lds_reuse"
                        else "needed_by")
                failures.append(MissingBarrierFailure(
                    producer=p_node, consumer=c_node, role=role,
                ))

    if not failures:
        # Couldn't classify — capture pipeline bug or classifier bug.
        msg = (
            f"diagnose_missing_edge could not classify missing edge "
            f"{p_id} -> {c_id} (kind={ref_edge.edge_kind}). "
            f"This indicates either a classifier bug or a capture-pipeline "
            f"bug that bypassed earlier sanity checks."
        )
        if raise_on_unexplained:
            raise UnexplainedMissingEdgeError(msg)
        # Soft-mode: return a synthetic Failure so production observability
        # logs the issue without crashing the build.
        unexplained = MissingWaitFailure(
            producer=p_node, consumer=c_node, counter_kind="unknown",
        ).with_legacy_msg(f"[unclassified] {msg}")
        return [unexplained]
    return failures


def _queue_depth_at(wait_node, producer, subj_graph) -> int:
    """Replay the per-counter FIFO from start of the graph to wait_node.position
    and return the queue depth at the wait's moment for the producer's counter.

    Walks across all bodies in execution order so cross-body queue state
    is preserved (matches build_dataflow_graph's persistent-queue model).
    """
    counter = counter_for(producer)
    depth = 0
    for n in _all_nodes_in_order(subj_graph):
        if n.position >= wait_node.position:
            break
        if counter == "dscnt" and n.category in PRODUCER_CATEGORIES_LDS:
            depth += 1
        elif counter == "vlcnt" and n.category in PRODUCER_CATEGORIES_GLOBAL:
            depth += 1
        elif n.category == SWAIT_CATEGORY:
            cap_value = _swait_drains(n, counter)
            if cap_value is not None and depth > cap_value:
                # Drain to cap; same as build_dataflow_graph semantics.
                depth = cap_value
    return depth


def _producer_queue_position(producer, subj_graph) -> int:
    """Return the producer's position in the per-counter FIFO at the moment
    it joined (zero-indexed from the queue head AT THAT MOMENT). Cross-body
    aware via _all_nodes_in_order."""
    counter = counter_for(producer)
    queue_size = 0
    for n in _all_nodes_in_order(subj_graph):
        if n is producer:
            return queue_size  # producer enters at this index
        if counter == "dscnt" and n.category in PRODUCER_CATEGORIES_LDS:
            queue_size += 1
        elif counter == "vlcnt" and n.category in PRODUCER_CATEGORIES_GLOBAL:
            queue_size += 1
        elif n.category == SWAIT_CATEGORY:
            cap_value = _swait_drains(n, counter)
            if cap_value is not None and queue_size > cap_value:
                queue_size = cap_value
    return queue_size


def _wait_drains_producer(wait_node, producer, subj_graph) -> bool:
    """True if `wait_node` drains `producer` — i.e. the wait's counter cap
    is low enough that the producer's slot in the FIFO falls inside the
    drained range at the wait's moment.

    Walks the WHOLE-graph stream (cross-body) so a producer in body=ML-1
    and a wait in body=ML correctly see each other in the simulation —
    same persistent-queue model as build_dataflow_graph.
    """
    counter = counter_for(producer)
    cap_value = _swait_drains(wait_node, counter)
    if cap_value is None:
        return False
    queue = []        # list of producer GraphNodes
    drained_ids = set()
    target_id = id(producer)
    for n in _all_nodes_in_order(subj_graph):
        if n.position > wait_node.position:
            break
        if counter == "dscnt" and n.category in PRODUCER_CATEGORIES_LDS:
            queue.append(n)
        elif counter == "vlcnt" and n.category in PRODUCER_CATEGORIES_GLOBAL:
            queue.append(n)
        elif n.category == SWAIT_CATEGORY:
            cv = _swait_drains(n, counter)
            if cv is not None:
                while len(queue) > cv:
                    drained_ids.add(id(queue.pop(0)))
    return target_id in drained_ids


def _any_drains(waits, producer, subj_graph) -> bool:
    return any(_wait_drains_producer(w, producer, subj_graph) for w in waits)


def _first_insufficient(waits, producer, subj_graph):
    """Return the first wait (in stream order) that does NOT drain the producer
    despite drainable counter. None if every wait drains, or no wait applies."""
    for w in waits:
        if not _wait_drains_producer(w, producer, subj_graph):
            return w
    return None


def _last_drain(waits, producer, subj_graph):
    """Return the latest wait that drained the producer, else None."""
    drainers = [w for w in waits if _wait_drains_producer(w, producer, subj_graph)]
    if not drainers:
        return None
    return max(drainers, key=lambda w: w.position)


# =============================================================================
# Self-validation: per-edge wait coverage
# =============================================================================
# Edges are now reorder-invariant (register-name resolution). The
# scheduler-correctness check — does the schedule have a covering
# s_waitcnt that drains each producer before its consumer reads? —
# is a SEPARATE pass over the graph + the captured stream.
#
# This is the lint that replaces:
#   - LRDataReadyRule              (CMSValidator.py:3461)
#   - GRAfterLRRule                (CMSValidator.py:3470)  [LDS-reuse barrier-edges]
#   - GRBeforeLRRule               (CMSValidator.py:3480)  [LDS-reuse barrier-edges]
#   - PackDataReadyRule (ordering) (CMSValidator.py:3464)
#
# Same Failure types the cross-graph diagnose_missing_edge classifier
# emits — the wiring is just driven differently. Instead of "for each
# missing edge, classify why subject lacks it", it's "for each edge in
# the (single) graph, classify whether the schedule covers it".


def validate_edge_wait_coverage(graph, *, raise_on_unexplained=False):
    """Validate that every dataflow edge has a covering wait/barrier in
    the captured stream.

    For each `raw_intrawave` edge: walk the captured stream between
    producer.position and consumer.position; require an SWaitCnt on the
    producer's counter (`dscnt` for LR/LW, `vlcnt` for GR) that drains
    the producer's queue slot. Emits MissingWaitFailure /
    WaitOnWrongCounterFailure / WaitInsufficientFailure as appropriate.

    For each `lr_to_gr_lds_reuse` / `gr_to_lr_lds_reuse` edge: the wait
    check above plus a barrier-coverage check (mirrors the LDS-reuse
    barrier requirement); emits MissingBarrierFailure when the wait
    covers but no barrier follows.

    Same-body OrderInverted (producer.position > consumer.position
    within the same body) is reported here as well — it indicates the
    schedule placed the producer after its consumer, which the wait
    machinery can't recover from.

    `raise_on_unexplained`: if True, raise UnexplainedMissingEdgeError
    when an edge falls through every classifier branch (defensive —
    means the classifier missed a case). Default False (production
    observability prefers a soft synthetic Failure).

    Returns a list of Failure objects. Empty list means "every edge in
    the graph has a covering wait/barrier in the captured stream".
    """
    failures = []
    for edge in graph.edges:
        edge_failures = _classify_edge_coverage(edge, graph,
                                                raise_on_unexplained=raise_on_unexplained)
        failures.extend(edge_failures)
    return failures


def _classify_edge_coverage(edge, subj_graph, *, raise_on_unexplained=False):
    """Per-edge coverage classifier — same logic diagnose_missing_edge
    runs in compare_graphs, but driven from a single graph rather than
    a missing-edge diff.
    """
    p_node = edge.producer
    c_node = edge.consumer

    # Phase 1 — same-body order check.
    if p_node.body_label == c_node.body_label and p_node.position > c_node.position:
        return [OrderInvertedFailure(producer=p_node, consumer=c_node)]

    # Phase 2 — wait coverage.
    expected_counter = counter_for(p_node)
    waits = waits_in_window(subj_graph, p_node.position, c_node.position,
                            counter=expected_counter)
    waits_other = waits_in_window(subj_graph, p_node.position, c_node.position,
                                  exclude_counter=expected_counter)

    failures = []
    wait_failure_emitted = False

    if not waits:
        if waits_other:
            failures.append(WaitOnWrongCounterFailure(
                producer=p_node, consumer=c_node,
                expected_counter=expected_counter,
                wrong_counter_waits=waits_other,
            ))
        else:
            failures.append(MissingWaitFailure(
                producer=p_node, consumer=c_node,
                counter_kind=expected_counter,
            ))
        wait_failure_emitted = True
    else:
        if not _any_drains(waits, p_node, subj_graph):
            insufficient = _first_insufficient(waits, p_node, subj_graph)
            if insufficient is not None:
                depth = _queue_depth_at(insufficient, p_node, subj_graph)
                cv = _swait_drains(insufficient, expected_counter)
                failures.append(WaitInsufficientFailure(
                    producer=p_node, consumer=c_node,
                    wait=insufficient,
                    queue_depth_at_wait=depth,
                    counter_value=cv if cv is not None else 0,
                ))
            else:
                failures.append(MissingWaitFailure(
                    producer=p_node, consumer=c_node,
                    counter_kind=expected_counter,
                ))
            wait_failure_emitted = True

    # Barrier check for LDS-reuse edges only.
    if (edge.edge_kind in ("lr_to_gr_lds_reuse", "gr_to_lr_lds_reuse")
            and not wait_failure_emitted):
        last_drain = _last_drain(waits, p_node, subj_graph)
        if last_drain is not None:
            barriers = barriers_in_window(subj_graph,
                                          start=last_drain.position,
                                          end=c_node.position)
            if not barriers:
                role = ("must_start_after"
                        if edge.edge_kind == "lr_to_gr_lds_reuse"
                        else "needed_by")
                failures.append(MissingBarrierFailure(
                    producer=p_node, consumer=c_node, role=role,
                ))

    if not failures and raise_on_unexplained:
        raise UnexplainedMissingEdgeError(
            f"validate_edge_wait_coverage: edge {p_node.identity} -> "
            f"{c_node.identity} (kind={edge.edge_kind}) wasn't classified "
            f"by any branch. Classifier bug."
        )
    return failures


def clone_loop_body(body):
    """Verbatim deepcopy of a LoopBodyCapture.

    Used to construct main_loop_prev[cp] from main_loop[cp]. The deepcopy is
    intentional: prev needs distinct TaggedInstruction objects so cross-iteration
    edges in the dataflow graph point at the prev copy, not the body's original.
    The underlying rocisa Instruction objects are deepcopied too; this is fine
    because RegisterContainer hashing/equality is value-based (verified against
    rocisa container.cpp:560-566).
    """
    return deepcopy(body)


# =============================================================================
# CMS macro walker
# =============================================================================

def evaluate_guard(guard_str, flags):
    """Evaluate a macro guard string against substituted flag values.

    Grammar observed in CustomSchedule.py:
        '\\<var> == <int>'
        '\\<var> == <int> && \\<var> == <int>'
    Multiple clauses are AND-joined.

    flags: dict mapping '\\ID', '\\useGR', '\\usePLR', '\\useGRInc', '\\useLoop'
    to int values.
    """
    clauses = [c.strip() for c in guard_str.split("&&")]
    for clause in clauses:
        if "==" not in clause:
            raise ValueError(f"Unsupported guard clause (no =='): {clause!r}")
        lhs, rhs = (s.strip() for s in clause.split("==", 1))
        if lhs not in flags:
            raise ValueError(f"Unknown guard variable {lhs!r} in {guard_str!r}")
        try:
            rhs_int = int(rhs)
        except ValueError:
            raise ValueError(f"Non-integer RHS {rhs!r} in clause {clause!r}")
        if flags[lhs] != rhs_int:
            return False
    return True


def _value_if_expr(item):
    """Extract the guard expression from a ValueIf or ValueElseIf node.

    The rocisa Python bindings don't expose `value` as an attribute — they only
    expose `__str__()` which returns `.if <expr>\\n` or `.elseif <expr>\\n`.
    The fake ValueIf classes used in tests have a plain `.value` attribute, so
    we try that first.
    """
    if hasattr(item, "value"):
        return item.value
    s = str(item)
    if s.startswith(".if "):
        return s[len(".if "):].rstrip("\n")
    if s.startswith(".elseif "):
        return s[len(".elseif "):].rstrip("\n")
    raise ValueError(f"Cannot extract guard expression from {s!r}")


def expand_cms_macro(macro, id_value, useGR, usePLR, useGRInc, useLoop,
                     tag_by_origin_id, sync_class=None, snop_class=None,
                     mfma_classes=()):
    """Walk a CMS MAINLOOP macro and produce a LoopBodyCapture for the given flags.

    Evaluates ValueIf/ValueElseIf/ValueEndif chains against the flag values and
    emits only the active-branch instructions. Tracks the "current" MFMA index
    by counting MFMA-class instructions encountered.

    Tag recovery priority:
        1. tag_by_origin_id[id(inst)] if present (original instructions, populated
           by the modified emit_instructions in CustomSchedule.py).
        2. Class-based fallback for SWaitCnt deepcopies emitted by
           nllvmcntHandling (those carry SYNC tag), MFMAs added directly via
           macro.add(mfmaCode[miIndex]) (carry MFMA tag), and SNop synthetic
           instructions.
        3. 'UNKNOWN' as last resort — should never happen for a well-formed macro.

    sync_class, snop_class, mfma_classes: rocisa types passed in to keep this
    module free of hard rocisa imports. The caller (CustomSchedule integration)
    supplies them.
    """
    flags = {
        "\\ID": id_value,
        "\\useGR": useGR,
        "\\usePLR": usePLR,
        "\\useGRInc": useGRInc,
        "\\useLoop": useLoop,
    }
    builder = LoopBodyCaptureBuilder()

    items = macro.items() if hasattr(macro, "items") else macro.itemList
    if callable(items):
        items = items()

    # Stack of {'had_match': bool, 'active': bool} per enclosing ValueIf chain.
    stack = []
    current_mfma_idx = -1

    def is_active():
        return all(frame["active"] for frame in stack)

    for item in items:
        cls_name = type(item).__name__

        if cls_name == "ValueIf":
            guard_active = is_active() and evaluate_guard(_value_if_expr(item), flags)
            stack.append({"had_match": guard_active, "active": guard_active})
            continue
        if cls_name == "ValueElseIf":
            if not stack:
                raise ValueError("ValueElseIf without enclosing ValueIf")
            frame = stack[-1]
            if frame["had_match"]:
                frame["active"] = False
            else:
                # Re-evaluate against the parent's active context (frame above
                # us in the stack); we already know parent context held when
                # the chain started, otherwise had_match would be irrelevant
                # since the chain would have been entered as inactive.
                parent_active = all(f["active"] for f in stack[:-1])
                guard_active = parent_active and evaluate_guard(_value_if_expr(item), flags)
                frame["active"] = guard_active
                frame["had_match"] = guard_active
            continue
        if cls_name == "ValueEndif":
            if not stack:
                raise ValueError("ValueEndif without enclosing ValueIf")
            stack.pop()
            continue
        if cls_name == "TextBlock":
            # Comments and raw text — skip; not part of the schedule semantics.
            continue

        if not is_active():
            continue

        # Resolve category.
        category = tag_by_origin_id.get(id(item))
        if category is None:
            if sync_class is not None and isinstance(item, sync_class):
                category = "SYNC"
            elif snop_class is not None and isinstance(item, snop_class):
                category = "SNOP"
            elif mfma_classes and isinstance(item, mfma_classes):
                category = "MFMA"
            else:
                category = "UNKNOWN"

        if category == "MFMA":
            current_mfma_idx += 1
            slot_kind = SLOT_KIND_MFMA
            mfma_index = current_mfma_idx
        elif current_mfma_idx == -1:
            slot_kind = SLOT_KIND_PRE_LOOP
            mfma_index = -1
        else:
            slot_kind = SLOT_KIND_MFMA
            mfma_index = current_mfma_idx

        builder.append(
            inst=item,
            category=category,
            iteration=0,  # CMS macro is iteration-flattened; iteration is encoded in category (e.g. LRA0 vs LRA1)
            slot_kind=slot_kind,
            mfma_index=mfma_index,
        )

    return builder.finalize()


def build_cms_four_part_capture(macro, num_codepaths, tag_by_origin_id,
                                  sync_class, snop_class, mfma_classes,
                                  num_mfma_per_iter=0):
    """Expand a CMS MAINLOOP macro four ways and assemble a FourPartCapture.

    main_loop[cp] expands with all flags=1 and \\ID=cp for each cp.
    main_loop_prev[cp] is a verbatim clone of main_loop[cp].
    n_gl[0] expands with useGR=0, usePLR=1, useGRInc=1, useLoop=0, \\ID=0.
    n_ll[0] expands with useGR=0, usePLR=0, useGRInc=0, useLoop=0, \\ID=0.

    These flag assignments mirror the CMS dispatch sites at:
      - simdSpecDispatch (KernelWriterAssembly.py:16175) for main_loop
      - noLoadLoopBody CMS shortcut (KernelWriter.py:2858-2866) for n_gl/n_ll
    """
    main_loop = {}
    main_loop_prev = {}
    for cp in range(num_codepaths):
        body = expand_cms_macro(
            macro, id_value=cp,
            useGR=1, usePLR=1, useGRInc=1, useLoop=1,
            tag_by_origin_id=tag_by_origin_id,
            sync_class=sync_class, snop_class=snop_class,
            mfma_classes=mfma_classes,
        )
        main_loop[cp] = body
        main_loop_prev[cp] = clone_loop_body(body)

    n_gl_body = expand_cms_macro(
        macro, id_value=0,
        useGR=0, usePLR=1, useGRInc=1, useLoop=0,
        tag_by_origin_id=tag_by_origin_id,
        sync_class=sync_class, snop_class=snop_class,
        mfma_classes=mfma_classes,
    )
    n_ll_body = expand_cms_macro(
        macro, id_value=0,
        useGR=0, usePLR=0, useGRInc=0, useLoop=0,
        tag_by_origin_id=tag_by_origin_id,
        sync_class=sync_class, snop_class=snop_class,
        mfma_classes=mfma_classes,
    )

    num_mfma = sum(1 for ti in main_loop[0].instructions if ti.category == "MFMA")

    return FourPartCapture(
        main_loop=main_loop,
        main_loop_prev=main_loop_prev,
        n_gl={0: n_gl_body},
        n_ll={0: n_ll_body},
        num_mfma=num_mfma,
        num_codepaths=num_codepaths,
        source="cms",
        num_mfma_per_iter=num_mfma_per_iter,
    )
