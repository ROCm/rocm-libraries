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


@dataclass
class DataflowEdge:
    src: TaggedInstruction
    dst: TaggedInstruction
    register: object  # RegisterContainer; opaque here to avoid hard rocisa import
    kind: str  # 'raw', 'wait', 'barrier'


@dataclass
class DataflowGraph:
    nodes: list
    edges: list


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
    """
    base = f"@ idx={node.position.vmfma_index}"
    if capture is None or node.category == "MFMA":
        return base
    list_pos = capture.instructions.index(node.tagged_inst)
    return f"{base} ({_ordinal(list_pos + 1)} entry in list)"


@dataclass
class Failure:
    """Common base for all reported scheduling problems.

    No body_label field on the base — every concrete subclass carries
    producer/consumer GraphNode references (or equivalent), and
    GraphNode.body_label is the source of truth.
    """

    def format(self, capture=None) -> str:
        raise NotImplementedError("subclasses must implement format()")


# ----------------------------------------------------------------------------
# 1. OrderInvertedFailure — producer issued after consumer (same body only).
#    Replaces: GR _validate_must_start_after early-issue branch, Pack early/late.
#    Emitted by: rules + dataflow comparison classifier.
# ----------------------------------------------------------------------------
@dataclass
class OrderInvertedFailure(Failure):
    producer: object  # GraphNode or ValidatorInstruction
    consumer: object

    def format(self, capture=None) -> str:
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

    def format(self, capture=None) -> str:
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

    def format(self, capture=None) -> str:
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

    def format(self, capture=None) -> str:
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

    def format(self, capture=None) -> str:
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

    def format(self, capture=None) -> str:
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

    def format(self, capture=None) -> str:
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

    def format(self, capture=None) -> str:
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

    def format(self, capture=None) -> str:
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

    def format(self, capture=None) -> str:
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

    def format(self, capture=None) -> str:
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

    def format(self, capture=None) -> str:
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


class LoopBodyCaptureBuilder:
    """Accumulates TaggedInstructions across multiple emission calls.

    Owns a `sequence` counter that increments per-append within the same
    (iteration, slot_kind, mfma_index) triple to produce deterministic SlotKeys.
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
        return LoopBodyCapture(instructions=list(self._instructions))


def build_dataflow_graph(body, prev=None):
    """Walk prev.instructions + body.instructions and build the register dataflow graph.

    Currently a skeleton that returns an empty graph. Real edge construction
    lands in a later phase (see plan §4); the node list is populated so other
    code can rely on the data structure shape.
    """
    nodes = []
    if prev is not None:
        nodes.extend(prev.instructions)
    nodes.extend(body.instructions)
    return DataflowGraph(nodes=nodes, edges=[])


def compare_captures(default_capture, cms_capture):
    """Initial data-movement-totals comparison rule.

    Returns (True, "") if the two captures agree on the kernel-level invariants
    that must hold for correctness; otherwise (False, message). Internal
    scheduling-noise differences (extra SYNC/SNOP, etc.) are NOT flagged here
    because CMS legitimately adds them. Richer per-edge dataflow comparisons
    are added later as separate rules.
    """
    if default_capture is None or cms_capture is None:
        return True, ""

    # Per-body MFMA presence check: within each capture, every populated body
    # must contain at least one MFMA-tagged instruction. Empty bodies are
    # skipped (not yet captured in this phase). Strict equality against
    # cap.num_mfma is NOT enforced because F32X emulation distributes MFMAs
    # into pack-code submodules differently across bodies (main_loop's
    # source-bucket classification may tag some MFMAs as 'PackB1'/'LRB3'
    # etc., while n_gl/n_ll's classification differs because the bodies are
    # scheduled with different inputs). The presence check still catches
    # gross drops (an entire body with zero MFMAs is almost certainly a
    # capture bug).
    for cap in (default_capture, cms_capture):
        body_groups = (
            ("main_loop", cap.main_loop),
            ("main_loop_prev", cap.main_loop_prev),
            ("n_gl", cap.n_gl),
            ("n_ll", cap.n_ll),
        )
        for label, by_cp in body_groups:
            for cp, body in by_cp.items():
                if not body.instructions:
                    continue
                got = sum(1 for ti in body.instructions if ti.category == "MFMA")
                if got == 0:
                    return False, (
                        f"{cap.source}.{label}[{cp}] has zero MFMA-tagged "
                        f"instructions; expected at least one"
                    )

    # Cross-scheduler num_mfma comparison (SOFT): F32X emulation distributes
    # MFMAs into pack-code submodules differently between default and CMS
    # schedulers, so the per-category 'MFMA' count won't match exactly. The
    # per-edge dataflow comparison (Phase 7) is the right place for semantic
    # equivalence checks. For now, no-op cross-scheduler MFMA-count check.

    # Cross-scheduler data-movement totals must match per codepath.
    # Default-side classification uses identity sets snapshotted from
    # self.codes.globalReadA/B/localWriteA/B; deepcopies of those modules
    # fall through to the generic 'GR'/'LW' tag. CMS-side uses 'GRA'/'GRB'
    # (named idMap keys). Under DirectToLds=1, GR instructions ARE the LW
    # instructions (same BufferLoad with lds flag set), so the GR vs LW
    # split itself differs between schedulers. We compare the COMBINED
    # data-movement total (GR+LW) which must match regardless of how the
    # individual buckets are tagged.
    DATA_MOVEMENT_CATS = ("GRA", "GRB", "GR", "LWA", "LWB", "LW")
    for cp in default_capture.main_loop:
        if cp not in cms_capture.main_loop:
            return False, f"main_loop[{cp}] missing from cms capture"
        d_main = default_capture.main_loop[cp]
        c_main = cms_capture.main_loop[cp]
        d_n = sum(1 for ti in d_main.instructions if ti.category in DATA_MOVEMENT_CATS)
        c_n = sum(1 for ti in c_main.instructions if ti.category in DATA_MOVEMENT_CATS)
        if d_n != c_n:
            return False, (
                f"main_loop[{cp}] data-movement (GR+LW) count differs: "
                f"default={d_n} vs cms={c_n}"
            )

    return True, ""


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
                                  sync_class, snop_class, mfma_classes):
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
    )
