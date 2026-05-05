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

import functools
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

    `subiter` is the inner-unroll subiteration (0..LoopIters-1) the
    instruction belongs to. Multiple instructions can share the same
    (slot_kind, mfma_index); `sequence` disambiguates them in stream-
    emission order, continuing across subiters within the same bucket
    so that SchedulePosition's (loop_index, vmfma_index, sub_index) tuple
    encodes stream order without needing the subiter field.
    """
    subiter: int
    slot_kind: str
    mfma_index: int
    sequence: int


# =============================================================================
# WrappedInstruction + MemoryRegion — unified resource model
# =============================================================================
# Real rocisa-bound classes refuse setattr (no __dict__; nanobind-locked) and
# can't be weakref'd, so any per-instance metadata must live somewhere we own.
# WrappedInstruction is a thin proxy: forwards attribute access to the
# underlying rocisa instance via __getattr__, holds wrapper-native fields
# (`reads`, `writes`) populated once at capture time by `_populate`. Callers
# read those fields directly afterwards — no merge of "implicit" vs "explicit".
#
# MemoryRegion lets non-register resources (LDS bytes today; scratch / GDS /
# global later) participate in the same edge-formation pipeline as
# RegisterContainer. The graph builder's `_intersection` is type-dispatched
# so a heterogeneous reads/writes list "just works": a read of an LDS region
# matches a write of the same region; a read of a vgpr never matches an LDS
# region (returns None).


class WrappedInstruction:
    """Thin proxy around a rocisa instruction.

    Forwards attribute access to the underlying rocisa instance via
    __getattr__; owns mutable wrapper-native fields (`reads`, `writes`) for
    the metadata the dataflow graph needs.

    `_rocisa_inst` is exposed as a property `rocisa_inst` so existing
    `type(node.rocisa_inst).__name__` callers keep working unchanged.
    `__str__` delegates to the underlying instance so `_canonical_render`
    sees the same render-string whether or not the inst is wrapped.
    """
    __slots__ = ("_rocisa_inst", "reads", "writes")

    def __init__(self, rocisa_inst):
        self._rocisa_inst = rocisa_inst
        self.reads = ()
        self.writes = ()

    @property
    def rocisa_inst(self):
        return self._rocisa_inst

    def __getattr__(self, name):
        # __slots__ guarantees _rocisa_inst/reads/writes are found via normal
        # lookup; we only land here for attrs the wrapper doesn't own.
        # Don't forward dunders — deepcopy/copy/pickle look up special
        # methods (__deepcopy__, __reduce_ex__, __getstate__, ...) via
        # getattr, and forwarding them to the rocisa inst would cause those
        # protocols to copy the inst directly and lose the wrapper. The
        # explicit __deepcopy__/__copy__ below handle the relevant cases.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._rocisa_inst, name)

    def __str__(self):
        return str(self._rocisa_inst)

    def __repr__(self):
        return f"WrappedInstruction({self._rocisa_inst!r})"

    def __deepcopy__(self, memo):
        from copy import deepcopy
        new = WrappedInstruction(deepcopy(self._rocisa_inst, memo))
        new.reads = deepcopy(self.reads, memo)
        new.writes = deepcopy(self.writes, memo)
        memo[id(self)] = new
        return new

    def __copy__(self):
        new = WrappedInstruction(self._rocisa_inst)
        new.reads = self.reads
        new.writes = self.writes
        return new


@dataclass(frozen=True)
class MemoryRegion:
    """A byte-addressed region of a memory space.

    `space` discriminates LDS / scratch / GDS / global; `buffer_id` is the
    symbolic root of the address (e.g. "LocalReadAddrA", "m0"); `offset` and
    `byte_count` describe the slice within that buffer. `frozen=True` makes
    the type hashable for free, so it can sit alongside RegisterContainer in
    `DataflowGraph.edge_keys()` set-dedup.

    Symbolic `buffer_id` is acceptable for the validator's cross-graph
    comparison because both schedulers consume the same kernel-writer state
    and produce the same address-vgpr names — over-claim and under-claim
    errors cancel out across reference and subject. Numeric resolution
    (interpreting SXorB32/VXorB32/SAddU32 over LDS-address vgprs to a
    concrete byte offset) is a follow-up for absolute LDS dataflow tracking,
    not required for graph comparison.
    """
    space: str           # "lds" today; "scratch" / "gds" / "global" forward-compatible
    buffer_id: str       # symbolic root of the address vgpr
    offset: int          # bytes from buffer base
    byte_count: int      # extent


@dataclass
class TaggedInstruction:
    """A rocisa Instruction with its CMS-style id_map category and slot position.

    `category` is determined at emission time by the source list/module the
    instruction was popped from (not by inspecting the instruction's class).
    The same rocisa class (e.g. VFmaMixF32) may carry different categories
    depending on which bucket emitted it.

    `wrapped` is a WrappedInstruction holding pre-populated `reads`/`writes`
    tuples for dataflow-graph edge formation. Production callers
    (LoopBodyCaptureBuilder.append) populate it at construction; the
    `_ensure_wrapped` helper lazily populates it for callers that build
    TaggedInstructions directly (test fixtures).
    """
    inst: object
    category: str
    slot: SlotKey
    wrapped: object = None  # WrappedInstruction; lazily populated


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
    tail loops in _emitNoLoadLoopBodyCMSMacro (intentional — code-paths
    differ only in main-loop scheduling order, all SIMDs converge to
    identical architectural state at the loop boundary; see that method's
    docstring and bead rocm-libraries-9sh).

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
    # Inner-unroll subiterations per MFMA group — used by build_dataflow_graph
    # to derive each MFMA's subiter index (vmfma_index // num_mfma_per_subiter).
    # Both default-side and CMS-side construction sites should pass
    # writer.states.numMfmaPerIter (upstream Tensile naming retains "Iter"
    # but it refers to the inner unroll subiteration here).
    # Defaults to 0 ("don't split MFMAs by subiter"); the resolver then keeps
    # all MFMAs at subiter 0, which loses cross-subiter PLR dataflow edges.
    # Test fixtures may safely leave it unset.
    num_mfma_per_subiter: int = 0


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


@functools.total_ordering
@dataclass(frozen=True)
class SchedulePosition:
    """Position in the instruction schedule. Fields ordered for tuple-style comparison."""
    # Which loop iteration this instruction belongs (larger index means later iteration)
    loop_index: int
    # Which VMFMA slot within the loop
    #   * 0 to num_vmfma-1 for normal positions
    #   * -1 for wrap-around between iterations
    #     (occurs before the first VMFMA in this loop but after the last VMFMA of the previous loop)
    vmfma_index: int
    # Ordering among instructions issued at the same (loop_index, vmfma_index).
    # Multiple instructions can share a VMFMA slot; this field breaks ties.
    sub_index: int

    def __lt__(self, other: 'SchedulePosition') -> bool:
        if self.loop_index == other.loop_index:
            if self.vmfma_index == other.vmfma_index:
                return self.sub_index < other.sub_index
            else:
                return self.vmfma_index < other.vmfma_index
        else:
            return self.loop_index < other.loop_index


def make_position(body_label, slot) -> SchedulePosition:
    """Construct a SchedulePosition from a TaggedInstruction.slot SlotKey.

    The body_label maps to loop_index via BODY_LABEL_TO_LOOP_INDEX so cross-body
    ordering is well-defined.
    """
    return SchedulePosition(
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

    issue_cycles is the per-instruction quad-cycle issue cost (bead `nk0`):
    mirrors `ValidatorInstruction.min_issue_quad_cycles()` (CMSValidator.py:327
    + 645). Default 1; SNop-shaped instructions return `1 + wait_state`.
    Populated by `_make_node` from a class-dispatch table so the graph-side
    `cumulative_issue_cycles` helper can simulate per-instruction issue costs
    cycle-exactly without re-importing CMSValidator (which would create an
    import cycle — CMSValidator imports from this module). After bead `8nz`
    the structural-side simulators (`precompute_issue_times`,
    `estimate_quad_cycles_precomputed`, `estimate_quad_cycles`) are deleted
    and `cumulative_issue_cycles` is the single source of truth.
    """
    identity: tuple                     # (rocisa_class_name, loop_index, signature_tuple)
    position: SchedulePosition
    category: str                       # propagated from TaggedInstruction
    rocisa_inst: object                 # back-reference to the rocisa instruction
    tagged_inst: TaggedInstruction      # back-reference for stream-position lookup
    body_label: str                     # 'ML-1' | 'ML' | 'NGL' | 'NLL'
    name: str = ""                      # human-readable label (e.g. 'LRA0[2]')
    issue_cycles: int = 1               # bead `nk0`: per-instruction quad-cycle cost


@dataclass
class DataflowEdge:
    """A dataflow edge — register or memory-region flow.

    `resource` (formerly `register`) holds either a RegisterContainer or a
    MemoryRegion: the unified `_intersection` is type-dispatched so both
    can flow through the same edge-formation pipeline.

    edge_kind discriminates the three kinds of dataflow this graph models:
      raw_intrawave        — producer SWait drains the in-wave counter
      lr_to_gr_lds_reuse   — LR0 -> SWait -> SBarrier -> GR (write reuses LDS slot)
      gr_to_lr_lds_reuse   — GR -> SWait -> SBarrier -> LR1 (read of just-written LDS)
    """
    producer: GraphNode
    consumer: GraphNode
    resource: object                    # RegisterContainer | MemoryRegion (opaque)
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

    `num_mfma_per_subiter` is copied from FourPartCapture so the
    OrderInverted classifier can derive an MFMA node's inner-unroll
    subiteration (`vmfma_index // n`) without re-plumbing it through every
    classifier call. Non-MFMA nodes get subiter from their category trailing
    digits (`PackA0` → 0).
    """
    nodes: dict                            # identity -> GraphNode
    edges: list                            # list[DataflowEdge]
    captures: dict                         # body_label -> LoopBodyCapture
    num_mfma_per_subiter: int = 0          # copied from FourPartCapture; 0 ⇒ all-subiter-0

    def edge_keys(self):
        """Edge-equality keys for cross-graph diff: (p_id, c_id, resource, kind)."""
        return {(e.producer.identity, e.consumer.identity, e.resource, e.edge_kind)
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


def _node_label(node, capture) -> str:
    """Render a node as 'category[N]' where N is its 0-based position within
    its category's emit stream in `capture`. Plain MFMAs (category=='MFMA')
    omit the [N] because vmfma_index already serves as their identity. Pack
    categories ('PackA0', 'PackB1', ...) keep [N] because CMS reschedules
    them.

    Strict: every non-MFMA node MUST carry a `tagged_inst` AND that
    `tagged_inst` MUST be present in `capture.instructions`. The fallback
    paths for "no tagged_inst" or "lookup miss" were removed — any caller
    that can't satisfy this contract is using the formatter incorrectly
    and should thread a real capture / populate tagged_inst before
    calling.
    """
    cat = node.category
    if cat == "MFMA":
        return cat
    tagged = getattr(node, "tagged_inst", None)
    if tagged is None:
        raise ValueError(
            f"_node_label: node with category={cat!r} has no tagged_inst; "
            f"every non-MFMA node passed to a Failure formatter must carry "
            f"tagged_inst so the per-category-stream [N] index can be computed"
        )
    same_cat = [
        t for t in capture.instructions
        if getattr(t, "category", None) == cat
    ]
    try:
        idx = same_cat.index(tagged)
    except ValueError as exc:
        raise ValueError(
            f"_node_label: node tagged_inst={tagged!r} (category={cat!r}) not "
            f"found in capture's same-category stream "
            f"({len(same_cat)} {cat} instructions in capture); "
            f"the caller passed a capture that doesn't contain the node"
        ) from exc
    return f"{cat}[{idx}]"


def _node_with_pos(node, capture) -> str:
    """Render a node as 'category[N] @ idx=M' — the canonical per-Failure
    node reference. Combines `_node_label` (per-category-stream index, MFMA
    omits brackets) with the bare vmfma_index. Used by every Failure
    formatter that names a producer/consumer node.

    Accepts GraphNode (`position`) or ValidatorInstruction (`issued_at`).
    """
    pos = getattr(node, 'position', None) or node.issued_at
    return f"{_node_label(node, capture)} @ idx={pos.vmfma_index}"


def _iter_note(producer, consumer) -> str:
    """Return ' (of next iteration)' when consumer is exactly one loop_index
    past producer (i -> i+1). SchedulePosition.loop_index is the canonical
    cross-body iteration counter, so the numeric +1 test captures every
    next-iteration crossing without hardcoding body labels. Empty string
    otherwise.

    Accepts GraphNode (`position`) or ValidatorInstruction (`issued_at`).
    """
    p_pos = getattr(producer, 'position', None) or producer.issued_at
    c_pos = getattr(consumer, 'position', None) or consumer.issued_at
    if c_pos.loop_index == p_pos.loop_index + 1:
        return " (of next iteration)"
    return ""


@dataclass
class Failure:
    """Common base for all reported scheduling problems.

    No body_label field on the base — every concrete subclass carries
    producer/consumer GraphNode references (or equivalent), and
    GraphNode.body_label is the source of truth.
    """

    def format(self, capture) -> str:
        """Stable boundary method. Delegates to the subclass canonical
        formatter. `capture` is mandatory — pass an explicit
        `LoopBodyCapture(instructions=[])` if the calling context lacks one."""
        return self._format_canonical(capture)

    def _format_canonical(self, capture) -> str:
        raise NotImplementedError("subclasses must implement _format_canonical()")


# ----------------------------------------------------------------------------
# 1. OrderInvertedFailure — cross-graph reorder detection.
#    Subject reverses the producer/consumer order that the default schedule
#    established. The default schedule IS the canonical reference; if subj
#    emits the producer at a later stream position than the consumer while
#    default emitted them in the opposite order, the subject violates a
#    real dataflow dependency.
#    Emitted exclusively by diagnose_missing_edge (compare_graphs).
# ----------------------------------------------------------------------------
@dataclass
class OrderInvertedFailure(Failure):
    producer: object  # GraphNode (subject-side)
    consumer: object  # GraphNode (subject-side)
    default_producer_position: object  # SchedulePosition (default-side, for diagnostics)
    default_consumer_position: object  # SchedulePosition (default-side, for diagnostics)

    def _format_canonical(self, capture) -> str:
        return (
            f"{_node_with_pos(self.producer, capture)} is issued after its consumer "
            f"{_node_with_pos(self.consumer, capture)}."
        )


# ----------------------------------------------------------------------------
# 2. MissingWaitFailure — no SWaitCnt drains the expected counter in the
#    window between producer and consumer. If other-counter SWaitCnts ARE
#    in the window, they're surfaced via `nearby_other_counter_waits` so
#    the user knows they could extend an existing SWaitCnt rather than
#    insert a new one. (Bead `hof` collapsed the former
#    WaitOnWrongCounterFailure into this single type — the user-facing
#    fix is the same in both cases.)
# ----------------------------------------------------------------------------
@dataclass
class MissingWaitFailure(Failure):
    producer: object
    consumer: object
    counter_kind: str  # 'dscnt' / 'vlcnt' / 'vscnt'
    nearby_other_counter_waits: list = field(default_factory=list)
    # ^ list[GraphNode] — SWaitCnts present in the window but draining other
    # counters. Empty when no SWaitCnts are in the window at all.

    def _format_canonical(self, capture) -> str:
        # Optional hint when other-counter SWaitCnts exist in the window:
        # the user could extend one of them rather than insert a new SWaitCnt.
        hint = ""
        if self.nearby_other_counter_waits:
            indices = ", ".join(
                f"idx={(getattr(w, 'position', None) or w.issued_at).vmfma_index}"
                for w in self.nearby_other_counter_waits
            )
            hint = f" (existing SWaitCnts at {indices} drain other counters)"
        return (
            f"SWaitCnt({self.counter_kind}) missing between "
            f"{_node_with_pos(self.producer, capture)} and "
            f"{_node_with_pos(self.consumer, capture)}"
            f"{_iter_note(self.producer, self.consumer)}{hint}."
        )


# ----------------------------------------------------------------------------
# 4. WaitTooLateFailure — SWait fires at/after the consumer.
# ----------------------------------------------------------------------------
@dataclass
class WaitTooLateFailure(Failure):
    producer: object
    consumer: object
    wait_position: SchedulePosition

    def _format_canonical(self, capture) -> str:
        return (
            f"{_node_with_pos(self.consumer, capture)}"
            f"{_iter_note(self.producer, self.consumer)} is guaranteed by an "
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

    def _format_canonical(self, capture) -> str:
        wait_pos = getattr(self.wait, 'position', None) or self.wait.issued_at
        return (
            f"{_node_with_pos(self.consumer, capture)}"
            f"{_iter_note(self.producer, self.consumer)}'s producer "
            f"{_node_with_pos(self.producer, capture)} "
            f"is guaranteed by SWaitCnt @ idx={wait_pos.vmfma_index}, "
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

    def _format_canonical(self, capture) -> str:
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
            f"{_node_with_pos(self.consumer, capture)}"
            f"{_iter_note(self.producer, self.consumer)}."
        )


# ----------------------------------------------------------------------------
# 7. WrongInterleavingFailure — MiddlePack pair-consumer ordering wrong.
# ----------------------------------------------------------------------------
@dataclass
class WrongInterleavingFailure(Failure):
    pack: object  # MiddlePack — validator-side ValidatorInstruction OR graph-side GraphNode
    expected_next: object  # MiddlePack (pair_consumer) — same shape choice
    actual_next: object  # MiddlePack (next_scheduled_middle_16) — same shape choice

    def _format_canonical(self, capture) -> str:
        return (
            f"{_node_with_pos(self.pack, capture)} has wrong "
            f"interleaving. Should have been followed by "
            f"{_node_with_pos(self.expected_next, capture)} "
            f"but was followed by "
            f"{_node_with_pos(self.actual_next, capture)}."
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

    def _format_canonical(self, capture) -> str:
        return (
            f"{_node_with_pos(self.producer, capture)} has "
            f"too little gap between it and "
            f"{_node_with_pos(self.consumer, capture)}. Expected at least "
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

    def _format_canonical(self, capture) -> str:
        bad = [(name, val) for name, val in
               (("dscnt", self.dscnt), ("vlcnt", self.vlcnt), ("vscnt", self.vscnt))
               if val < -1]
        bad_str = ", ".join(f"{name}={val}" for name, val in bad)
        return (
            f"SWaitCnt @ idx={self.swait.issued_at.vmfma_index} is invalid: "
            f"{bad_str}. All counter fields must be >= -1."
        )


# ----------------------------------------------------------------------------
# 10. SCCConflictFailure — SCC clobber window.
#     An SCC writer (`intervening_writer`) sits between an SCC producer and
#     SCC consumer, displacing the producer's value before the consumer can
#     read it.
#
#     Graph-native shape (mrj.2): producer/consumer/intervening_writer are
#     GraphNode references emitted by `diagnose_missing_edge` when an
#     SCC-typed reference edge is missing from the subject graph.
#
#     SCCConflictFailure is reserved for the CLOBBER case. Pure SCC reorder
#     (no intervening writer, just consumer issued before producer) is
#     surfaced by `OrderInvertedFailure` via the existing Phase-1 order
#     check — SCC reads/writes are tracked since mrj.1 so the same machinery
#     covers SCC operands.
# ----------------------------------------------------------------------------
@dataclass
class SCCConflictFailure(Failure):
    producer: object = None             # GraphNode (subject-side SCC writer the consumer SHOULD have read)
    consumer: object = None             # GraphNode (subject-side SCC reader)
    intervening_writer: object = None   # GraphNode (subject-side SCC writer that clobbered the producer)

    def _format_canonical(self, capture) -> str:
        inter_desc = ""
        if self.intervening_writer is not None:
            inter_desc = (
                f" Intervening SCC writer "
                f"{_node_with_pos(self.intervening_writer, capture)} "
                f"clobbered the producer's SCC value."
            )
        return (
            f"{_node_with_pos(self.consumer, capture)}'s SCC read should "
            f"resolve to producer {_node_with_pos(self.producer, capture)}, "
            f"but the CMS schedule routes it elsewhere.{inter_desc}"
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
    (slot_kind, mfma_index) bucket. The counter is SHARED across subiters
    because SchedulePosition's tuple — (loop_index, vmfma_index, sub_index) —
    drops `subiter`; sub_index must therefore continue across subiters
    so positions encode stream-emission order without collisions.

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
        self._seq_counter = {}  # (slot_kind, mfma_index) -> next sequence

    def append(self, inst, category, subiter, slot_kind=SLOT_KIND_MFMA, mfma_index=-1):
        key = (slot_kind, mfma_index)
        seq = self._seq_counter.get(key, 0)
        self._seq_counter[key] = seq + 1
        slot = SlotKey(
            subiter=subiter,
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
            # Wrap and populate reads/writes so build_dataflow_graph picks
            # up dataflow without needing per-call extraction. The wrapper
            # is the single source of (reads, writes) — DTL m0 reads,
            # MFMA acc writes, LDS-address vgpr reads etc. all flow
            # through `_populate_wrapper`'s rule registry.
            ti.wrapped = WrappedInstruction(inst)
            _populate_wrapper(ti.wrapped, category=ti.category)
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


def build_id_to_category_per_iter(*, subiter, localReadCode, localWriteCode,
                                  globalReadCode, packCode, packPreCode,
                                  globalReadA=None, globalReadB=None,
                                  globalReadIncACode=None, globalReadIncBCode=None,
                                  inner_unroll_max=8):
    """Build {id(item) -> category} for one SIA3 subiter's combined modules.

    Companion to build_idmap. Two factories, one schema, two input shapes:

    - build_idmap + invert_idmap_to_id_to_category: when the caller has
      per-category source modules (LRCodeAAllIters[u] etc.). Used by the
      main-loop capture path in _loopBody.
    - build_id_to_category_per_iter: when the caller has per-subiter
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
        named LocalReadDo{Tensor}_I{iui}; tag as LR{Tensor}{subiter}.
      - packCode/packPreCode: per-side (A/B) sub-modules named
        pack{A,B}_I{iui} (and "pack{A,B}_I{iui} Pre"); tag as
        Pack{A,B}{subiter}.
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
            cat = cat_template.format(subiter)
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
                    tag_module(sub, f"Pack{side}{subiter}")
                sub_pre = pack_mod.findNamedItem(f"{prefix}_I{iui} Pre")
                if sub_pre is not None:
                    tag_module(sub_pre, f"Pack{side}{subiter}")

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
    `_BODY_BUILD_ORDER` which matches SchedulePosition.loop_index ordering,
    so concatenating yields a globally-correct stream.
    """
    for label in _BODY_BUILD_ORDER:
        cap = subj_graph.captures.get(label) if subj_graph is not None else None
        if cap is None or not hasattr(cap, '_graph_nodes'):
            continue
        for node in cap._graph_nodes:
            yield node


def waits_in_window(subj_graph, start: SchedulePosition, end: SchedulePosition,
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


def barriers_in_window(subj_graph, start: SchedulePosition, end: SchedulePosition):
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
# CVT-pack rocisa classes (TF32 emulation: v_cvt_pk_bf16_f32 and friends).
# Used by `_is_cvt_pack` (Sub-B of bead `35z`) to identify CVTPack producers
# whose results feed downstream MFMAs and need the 2-quad-cycle settle window
# (`_QUAD_CYCLES_CVT_BEFORE_MFMA` in this file). Mirrors the
# class-name-set lookup pattern used for MFMA / LR / LW / GR; the production
# rocisa class is `VCvtPkF32toBF16` (CMSValidator.py:676 PACK_TYPE_MAP entry
# binds it to the `CVTPack` validator dataclass). Test fixtures use plain
# `_FakeCVTPack` if and when they need to exercise this branch with a
# non-rocisa stub; today the production class is the only entry.
_CVT_PACK_CLASS_NAMES = {
    "VCvtPkF32toBF16",
}

# MiddlePack rocisa classes (TF32 emulation: the 16 instructions in the middle
# of each 24-pack group that compute the bf16 error terms via paired
# v_cvt_f32_bf16 + v_sub_f32 instructions). Used by `_is_middle_pack` and
# `validate_middle_pack_pair_interleaving` (bead `dpi`) to identify the
# pair-leader / pair-consumer relationships independently of the structural
# `MiddlePack` validator-dataclass binding (CMSValidator.py:627-630
# PACK_TYPE_MAP entries). The pairing semantics: middle-16 packs in each
# category (e.g. PackA0) are paired adjacently in stream order — pair (0,1),
# pair (2,3), etc. — and each pair's two halves share a temporary VGPR, so no
# OTHER middle-16 pack (from any category, even another one in this category)
# may appear between them in the global stream.
_MIDDLE_PACK_CLASS_NAMES = {
    "PVCvtBF16toFP32",
    "VCvtBF16toFP32",
    "VSubF32",
    "VDot2CF32BF16",
}


def _is_lr(inst):
    return type(inst).__name__ in _LR_CLASS_NAMES


def _is_lw(inst):
    return type(inst).__name__ in _LW_CLASS_NAMES


def _is_gr(inst):
    return type(inst).__name__ in _GR_CLASS_NAMES


def _is_mfma(inst):
    return type(inst).__name__ in _MFMA_CLASS_NAMES


def _is_middle_pack(inst):
    """True for MiddlePack rocisa instances (TF32 middle-16 v_cvt_f32_bf16
    + v_sub_f32 / PVCvtBF16toFP32 / VDot2CF32BF16 family).

    These are the 16 instructions in each 24-pack group that compute the
    bf16 error terms. Per CMSValidator.py PACK_TYPE_MAP, all of them bind
    to the `MiddlePack` validator dataclass which carries the pair-consumer
    interleaving invariant. The graph-side classifier in
    `validate_middle_pack_pair_interleaving` (bead `dpi`) uses this
    discriminator to identify pair leaders / consumers from the GraphNode
    stream without re-importing the validator dataclass (which would create
    an import cycle). Test fixtures may use any of the four production
    rocisa classes, or a stub class whose `type(...).__name__` matches one
    of the names in `_MIDDLE_PACK_CLASS_NAMES`.
    """
    return type(inst).__name__ in _MIDDLE_PACK_CLASS_NAMES


def _is_cvt_pack(inst):
    """True for CVT-pack rocisa instances (`v_cvt_pk_bf16_f32` family).

    These are the TF32 CVT0/CVT1 packs that bind to the validator-side
    `CVTPack` dataclass via PACK_TYPE_MAP (CMSValidator.py:676). When such
    an instruction writes a vgpr that a downstream MFMA reads, the CDNA 4
    ISA (section 7.6) requires 2 quad-cycles between them
    (`_QUAD_CYCLES_CVT_BEFORE_MFMA` in this file). The graph-side
    enforcement of this rule (Sub-B of bead `35z`) routes CVTPack producers
    through `_cvt_to_mfma_gap_ok` instead of the ALU-immediate exemption.
    """
    return type(inst).__name__ in _CVT_PACK_CLASS_NAMES


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
# --- Stream-position ordering ---
# =============================================================================
# The resolver walks producers in stream-position order. SchedulePosition
# (loop_index, vmfma_index, sub_index) lex-sorts to actual stream order by
# construction (see SlotKey docstring + commit f06ffc4770), so node.position
# is the canonical ordering key — no synthetic kind_rank table needed.
#
# Two "iter" axes exist in this codebase; do not conflate them:
#   1. Outer iteration / body — which body the node belongs to
#      (ml_prev, ml, ngl, nll). Encoded in SchedulePosition.loop_index
#      and node.body_label. Cross-body comparison happens naturally via
#      stream-position lex sort (loop_index is the first component).
#   2. Inner-unroll subiteration ("subiter") — which inner unroll iteration
#      within a single body. Encoded in category trailing digits (LRA0,
#      PackB3) for non-MFMA, or in vmfma_index // num_mfma_per_subiter
#      for MFMA. Computed by _node_subiter.
#
# _node_subiter is used by the within-graph same-subiter gate in
# _classify_edge_coverage (which has no default reference). Both subiter
# derivations are schedule-invariant per identity (categories and
# vmfma_indices are kernel-writer-set, identical across captures).

import re as _re
_TRAILING_DIGITS_RE = _re.compile(r"^(.*?)(\d*)$")


def _split_category_iter(category):
    """Split 'LRA0' -> ('LRA', 0), 'PackB3' -> ('PackB', 3), 'GRA' -> ('GRA', 0).

    Trailing digits become the iteration index; everything before is the
    base category name. Categories with no trailing digits (e.g. GRA, LWA)
    return iter=0.
    """
    m = _TRAILING_DIGITS_RE.match(category)
    base, suffix = m.group(1), m.group(2)
    return base, (int(suffix) if suffix else 0)


def _node_subiter(node, num_mfma_per_subiter) -> int:
    """Inner-unroll subiteration index for a graph node.

    "Subiter" = which inner unroll subiteration this node belongs to within
    its body. NOT the outer loop iteration (those are encoded in
    SchedulePosition.loop_index and the body label ml_prev / ml / ngl / nll).

    For non-MFMA categories, parsed from the category trailing digits
    (`PackA0` ⇒ 0). For MFMA, derived from
    `vmfma_index // num_mfma_per_subiter`. When `num_mfma_per_subiter` is 0
    (test fixtures that don't set it), MFMA subiter collapses to 0 — the
    OrderInverted gate then degenerates to "fire on any same-body
    stream-position inversion".
    """
    if node.category == "MFMA":
        return node.position.vmfma_index // num_mfma_per_subiter if num_mfma_per_subiter else 0
    return _split_category_iter(node.category)[1]


def _byte_keys_for_resource(resource):
    """Enumerate byte-grain keys covered by `resource`.

    Two resources that overlap return overlapping key sets; resources
    that don't overlap have disjoint key sets. The keys are used as
    indices into the per-byte latest-writer map maintained by
    build_dataflow_graph Phase 2.

    Numeric registers: one key per register index in the range
        (regType, regIdx)
    Symbolic registers: keyed by name root + offset within the named region
        (regType, name, offset)
    MemoryRegion: one key per byte in the slice
        ("mem", space, buffer_id, offset_byte)

    Returns an empty tuple for unrecognized resource shapes.
    """
    from rocisa.container import RegisterContainer
    if isinstance(resource, MemoryRegion):
        return tuple(
            ("mem", resource.space, resource.buffer_id, resource.offset + i)
            for i in range(resource.byte_count)
        )
    if not isinstance(resource, RegisterContainer):
        return ()
    rt = resource.regType
    count = resource.regNum or 1
    if resource.regIdx >= 0:
        return tuple((rt, resource.regIdx + i) for i in range(count))
    name_obj = getattr(resource, "regName", None)
    if name_obj is None:
        return ()
    name = name_obj.name
    base = name_obj.getTotalOffsets() if hasattr(name_obj, "getTotalOffsets") else 0
    return tuple((rt, name, base + i) for i in range(count))


def _resolve_producers(read_resource, consumer, latest_writer):
    """Yield (producer_node, overlap) pairs for `consumer`'s read of `read_resource`.

    `latest_writer` is the per-byte map maintained by build_dataflow_graph
    Phase 2: byte_key -> (writer_node, write_resource). For each byte the
    read covers, look up the latest writer; group bytes by
    (writer_node, write_resource) so distinct writes (e.g., a multi-write
    producer feeding a wide read) each emit one edge.

    The yielded `overlap` is the intersection of read_resource with the
    writer's actual write_resource — same precision the old resolver
    yielded, so diagnostic formatters and edge-set dedup still work.
    """
    writer_groups = {}  # (id(writer), id(write_res)) -> (writer_node, write_res)
    for bk in _byte_keys_for_resource(read_resource):
        entry = latest_writer.get(bk)
        if entry is None:
            continue
        writer_node, write_res = entry
        key = (id(writer_node), id(write_res))
        if key not in writer_groups:
            writer_groups[key] = (writer_node, write_res)

    for writer_node, write_res in writer_groups.values():
        overlap = _intersection(read_resource, write_res)
        if overlap is not None:
            yield (writer_node, overlap)


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


# =============================================================================
# Operand rule registry — single source of per-class semantic knowledge
# =============================================================================
# Each rocisa instruction class exposes its read/write operands differently.
# A rule encapsulates: which classes it applies to (`applies(inst)`), and
# what those operands are (`extract(inst) -> (reads_tuple, writes_tuple)`).
# Rules are tried in order and their outputs accumulated into the wrapper.
#
# This collapses the prior `_writes` / `_reads` if/elif chain (which only
# handled LR/LW/GR/MFMA) plus the separate implicit-operand tagging
# mechanism (which existed solely to plumb m0 reads through DTL BufferLoads)
# into one extensible pipeline. New classes just add a rule.


def _is_dtl_buffer_load(inst) -> bool:
    """A BufferLoad whose dst is None is a DTL-mode load (kernel writer's
    `dst = None if lds else vgpr(...)` at KWA:14608). Such loads write to
    LDS rather than a vgpr and implicitly read m0 for the LDS destination.

    Real rocisa BufferLoad classes don't expose `dst` as a Python attribute
    at all — `hasattr(inst, "dst")` returns False even for non-DTL loads.
    The actual dst lives at `getParams()[0]`, which is None for DTL and a
    RegisterContainer otherwise. We use that as the structural signal."""
    if not _is_gr(inst):
        return False
    return _inst_dst(inst) is None


def _is_register(x) -> bool:
    """True if x walks like a RegisterContainer (has regType + regIdx).

    Used by the GenericALURule fallback to filter getParams() entries that
    aren't registers (modifiers, ints, comments, None)."""
    return x is not None and hasattr(x, "regType") and hasattr(x, "regIdx")


def _is_vcc(x) -> bool:
    """True if x is a rocisa VCC sentinel (the carry-out / carry-in / cmp-out
    operand class).

    VCC instances have no regType/regIdx (verified empirically in bead wx9.9
    inspection — only `__class__` distinguishes them), so `_is_register`
    rejects them. The dedicated `_VCCRule` recognizes them via this helper
    and substitutes the synthetic `_VCC_RESOURCE` singleton so per-byte
    tracking can flow through the standard resolver.
    """
    return type(x).__name__ == "VCC"


# Synthetic singleton resource representing the VCC 64-bit register pair
# (vcc_lo, vcc_hi). Built on first use to avoid a top-level rocisa import.
# Shape: regType="vcc", regIdx=0, regNum=2 — `_byte_keys_for_resource`
# returns (("vcc", 0), ("vcc", 1)) and `_reg_intersection` reuses the
# numeric-range path (same regType always overlaps for the singleton).
_VCC_RESOURCE = None


def _vcc_resource():
    global _VCC_RESOURCE
    if _VCC_RESOURCE is None:
        from rocisa.container import RegisterContainer, RegName
        _VCC_RESOURCE = RegisterContainer("vcc", RegName("vcc", []), 0, 2)
    return _VCC_RESOURCE


# Class names whose instructions write VCC at slot 1 (carry-out dst1).
# Detected by name to avoid hard-importing rocisa at module load. All such
# classes have getParams shape `[vgpr_dst, VCC_dst1, src0, src1, ...]`.
_VCC_DST1_CARRY_OUT_CLASSES = frozenset({
    "VAddCOU32",   # 64-bit add lower half: writes vgpr + carry-out VCC
    "VAddCCOU32",  # 64-bit add upper half: writes vgpr + carry-out VCC, reads carry-in VCC
    "VSubCoU32",   # 64-bit sub lower half: writes vgpr + borrow-out VCC
})


class _DSLoadRule:
    """DSLoadB* — dst (vgpr) and the LDS-address vgpr (`LocalReadAddr{tc}`).

    Real rocisa: getParams() is `[dst, src_lds_addr, ds_modifiers, ...]`.
    Synthetic _FakeLR: only exposes `dst`, no LDS-address vgpr in the fake.
    """
    def applies(self, inst, category=None): return _is_lr(inst)
    def extract(self, inst, category=None):
        dst = _inst_dst(inst)
        lds_addr = _get_param(inst, 1)  # None for fakes (no getParams)
        writes = (dst,) if dst is not None else ()
        reads = (lds_addr,) if _is_register(lds_addr) else ()
        return reads, writes


class _DSStoreRule:
    """DSStoreB* — reads the LDS-address vgpr (slot 0) and the data src
    vgpr (slot 1). Writes nothing register-wise (the bytes go to LDS;
    register-side dataflow has no write).

    Real rocisa: getParams() is `[dstAddr, src_data, ds_modifiers, comment]`.
    Synthetic _FakeLW: only exposes `src` for data; no LDS-address vgpr.
    """
    def applies(self, inst, category=None): return _is_lw(inst)
    def extract(self, inst, category=None):
        lds_addr = _get_param(inst, 0)
        src_data = _inst_dsstore_src(inst)
        reads = tuple(r for r in (lds_addr, src_data) if _is_register(r))
        return reads, ()


class _DTLBufferLoadRule:
    """DTL BufferLoad (dst=None) — implicit m0 read for LDS destination,
    plus the SRD sgpr. No register-side write (data goes straight to LDS).
    """
    def applies(self, inst, category=None): return _is_dtl_buffer_load(inst)
    def extract(self, inst, category=None):
        from rocisa.container import mgpr
        srd = _inst_buffer_srd(inst)
        reads = [mgpr(0)]
        if _is_register(srd):
            reads.append(srd)
        return tuple(reads), ()


class _BufferLoadRule:
    """Non-DTL BufferLoad — writes dst, reads SRD."""
    def applies(self, inst, category=None):
        return _is_gr(inst) and not _is_dtl_buffer_load(inst)
    def extract(self, inst, category=None):
        dst = _inst_dst(inst)
        srd = _inst_buffer_srd(inst)
        writes = (dst,) if dst is not None else ()
        reads = (srd,) if _is_register(srd) else ()
        return reads, writes


class _MFMARule:
    """MFMA — accumulator is read-modify-write; a/b are reads.

    Excludes Pack-categorized MFMAInstructions (TF32 emulation pattern):
    those are syntactically MFMAInstruction but semantically Pack
    operations producing values for downstream MFMAs in the SAME or NEXT
    inner-unroll subiteration. Treating them as MFMA producers creates
    cross-subiter edges that the legacy OrderInverted classifier couldn't
    distinguish from same-subiter reorders. After 9.6.1 (default-as-canonical
    OrderInverted) the legacy gate no longer applies; PackMFMA inclusion can
    be revisited as a follow-up.
    """
    def applies(self, inst, category=None):
        if not _is_mfma(inst):
            return False
        if category is not None and category.startswith("Pack"):
            return False
        return True

    def extract(self, inst, category=None):
        a = _inst_mfma_a(inst)
        b = _inst_mfma_b(inst)
        acc = _inst_mfma_acc(inst)
        reads = tuple(r for r in (a, b, acc) if r is not None)
        writes = (acc,) if acc is not None else ()
        return reads, writes


class _NoDataflowRule:
    """SWaitCnt / SBarrier / SNop — pure scheduling control; no dataflow."""
    def applies(self, inst, category=None):
        return _is_swait(inst) or _is_sbarrier(inst) or _is_snop(inst)
    def extract(self, inst, category=None):
        return (), ()


class _VSwapRule:
    """VSwapB32 (and any v_swap_* variant) — symmetric R+W on both operands.

    `v_swap_b32 dst, src` exchanges the two registers: BOTH are read AND
    BOTH are written. Modelling this with the asymmetric
    `_GenericALURule` shape (`writes=(params[0],)`, `reads=params[1:]`)
    drops one of the four edge classes. Concretely, given:

        sw1: VSwap(v0, v1)   # ref position
        sw2: VSwap(v1, v2)

    under the asymmetric model sw1 publishes write=v0/read=v1 and sw2
    publishes write=v1/read=v2 — they share v1 only as
    sw1.read + sw2.write, which is a WAR edge sw1->sw2. Reverse the pair
    and the edge becomes RAW sw2->sw1: the SUBJECT graph gains a NEW
    edge that REF lacks. Because `compare_graphs`
    (ScheduleCapture.py:~2598, `missing = ref - subj`) is one-directional,
    the reorder is invisible.

    Symmetric model: `reads = writes = (regs...)`. Now BOTH orderings of
    the pair carry both a WAR edge (sw_first.read + sw_second.write on
    the shared reg) AND a WAW edge (both write the shared reg) AND a
    RAW edge (sw_first.write + sw_second.read). Swapping the pair flips
    which instruction is producer/consumer on each of those edges, so
    the edge KEY (producer.identity, consumer.identity, register, kind)
    differs between REF and SUBJ — `missing = ref - subj` finds at least
    one such key and `compare_graphs` flags the reorder.

    Order: MUST come before `_GenericALURule` so VSwap is claimed by this
    rule and not the asymmetric fallback.

    Bead: rocm-libraries-wx9.8.
    """
    def applies(self, inst, category=None):
        # Match rocisa VSwapB32 today and any future v_swap_* width
        # variant (e.g. VSwapB16) by class-name prefix. Keeps us decoupled
        # from a hardcoded opcode allowlist that needs maintenance per
        # ISA addition.
        return type(inst).__name__.startswith("VSwap")

    def extract(self, inst, category=None):
        try:
            params = list(inst.getParams())
        except Exception:
            return (), ()
        # First two positional params are the swapped operands. Filter
        # non-registers defensively (modifiers, comments) — a well-formed
        # VSwap will have two register params, but we don't want a stray
        # int/None to crash the rule.
        regs = tuple(p for p in params[:2] if _is_register(p))
        # Symmetric: both operands are read AND both are written.
        return regs, regs


class _VCCRule:
    """Track VCC carry / borrow / cmp-out as a first-class resource.

    rocisa exposes VCC as an opaque sentinel class with no regType/regIdx,
    so the generic resolver filters it out of every rule's reads/writes.
    That leaves VAddCO/VAddCCO/VSubCo carry chains and VCmp* / SOrSaveExec*
    VCC writes invisible to the dataflow graph — reorders that break a
    64-bit add silently slip past `compare_graphs`.

    This rule fires on any instruction whose `getParams()` contains at
    least one VCC sentinel and rewrites the (reads, writes) lists,
    substituting `_vcc_resource()` for each VCC sentinel.

    Position semantics:
      slot 0 : write (the dst — vgpr OR VCC for VCmp*, SOrSaveExec*)
      slot 1 : write IF the class has a `dst1` carry-out slot
               (VAddCOU32 / VAddCCOU32 / VSubCoU32 — see
               `_VCC_DST1_CARRY_OUT_CLASSES`)
      else   : read (covers VAddCCOU32 src2 = VCC carry-in)

    Order: this rule MUST come before `_GenericALURule`. The generic rule
    misclassifies VAddCCOU32's slot-1 VCC dst1 as a read (everything past
    slot 0 is a read) and drops the VCC carry-in read at slot 4. This rule
    claims VCC-bearing instructions and emits the correct shape.

    Coverage (bead wx9.9):
      - VAddCOU32 / VAddCCOU32 — 64-bit add carry chain (globalReadIncrement
        non-buffer path, KernelWriterAssembly.py:9120-9146)
      - VSubCoU32 — 64-bit sub borrow chain
      - VCmpEQU32 and other VCmp* — VCC-as-comparison-output
      - SOrSaveExecBX with dst=VCC() — VCC-as-saved-EXEC

    NOT addressed here (out of bead scope):
      - sgpr-pair-as-VCC: real CMS sometimes uses an sgpr pair as the
        carry destination (e.g. `dst1=sgpr("VCC", 2)`) instead of the
        VCC sentinel. Those flow through `_GenericALURule` already; the
        wx9.9 fix only addresses the VCC-sentinel form.
    """
    def applies(self, inst, category=None):
        if not hasattr(inst, "getParams"):
            return False
        try:
            params = inst.getParams()
        except Exception:
            return False
        return any(_is_vcc(p) for p in params)

    def extract(self, inst, category=None):
        try:
            params = list(inst.getParams())
        except Exception:
            return (), ()
        if not params:
            return (), ()
        cls_name = type(inst).__name__
        has_dst1 = cls_name in _VCC_DST1_CARRY_OUT_CLASSES

        def _resolve(p):
            return _vcc_resource() if _is_vcc(p) else p

        # Slot 0 — dst (vgpr or VCC for VCmp* etc.).
        writes = []
        if _is_register(params[0]) or _is_vcc(params[0]):
            writes.append(_resolve(params[0]))

        reads_start = 1
        if has_dst1 and len(params) > 1 and _is_vcc(params[1]):
            writes.append(_resolve(params[1]))
            reads_start = 2

        reads = tuple(
            _resolve(p) for p in params[reads_start:]
            if _is_register(p) or _is_vcc(p)
        )
        return reads, tuple(writes)


# =============================================================================
# SCC sentinel resource (bead mrj.1)
# =============================================================================
# SCC is a single-bit hardware status register, written implicitly by most
# scalar ALU and compare ops and read by SCSelect/SCMov/SCBranchSCC*. To
# model it as a first-class graph resource we use a module-level singleton
# RegisterContainer with `regType="scc"` (a fresh regType, distinct from
# "v"/"s"/"acc"/"m"). This rides on existing register machinery:
#
#   * `_is_register` accepts it (has regType + regIdx).
#   * `_reg_intersection` short-circuits on `regType` mismatch, so SCC
#     never accidentally aliases vgpr/sgpr.
#   * For two SCC instances `_reg_intersection` falls into the numeric
#     branch and looks up `_NUMERIC_REG_FACTORIES["scc"]`, which we
#     register below as a constant-returning factory (always returns the
#     singleton — SCC is a single bit, regIdx=0, regNum=1).
#   * `_byte_keys_for_resource` keys it as `("scc", 0)`, giving the
#     per-byte latest-writer resolver one slot to track.
#
# Lazy initialization avoids a top-level rocisa import (the module's
# convention; the numeric factories follow the same pattern).
_SCC_SENTINEL = None


def _get_scc_sentinel():
    """Return the module-level SCC RegisterContainer singleton.

    Constructed by mutating a fresh `vgpr(0)` because the nanobind binding
    for `RegisterContainer.__init__` requires a non-None `regName: RegName`
    even though the underlying C++ field is `std::optional<RegName>`. The
    `vgpr` factory bottoms out in the C++ `std::nullopt` overload and
    leaves `regName` as None on the Python side; mutating `regType` after
    construction is safe (the field is `def_rw`).
    """
    global _SCC_SENTINEL
    if _SCC_SENTINEL is None:
        from rocisa.container import vgpr
        sentinel = vgpr(0)
        sentinel.regType = "scc"
        _SCC_SENTINEL = sentinel
    return _SCC_SENTINEL


# Source of truth for which scalar opcodes touch SCC. Class-name keyed
# (matches `type(inst).__name__`) -> (reads_scc, writes_scc).
#
# Hand-curated from KernelWriterAssembly.py emissions and the rocisa
# rocisa/include/instruction/{cmp,common,branch}.hpp class definitions.
# The companion epic doc (`br show rocm-libraries-mrj`) traces these
# back to the gfxIsa.inc IF_ImplicitReadSCC/IF_ImplicitWriteSCC flags.
#
# Three shape categories:
#   - "no_dst": SCmp* / SCBranchSCC* — no sgpr dst, all register params
#     are reads (override of _GenericALURule's params[0]=write default).
#   - "dst_then_srcs": SAdd/SSub/SCSelect/SCMov/SAnd/SOr/etc — write at
#     params[0], reads in params[1:] (same shape as _GenericALURule, but
#     this rule gets first dibs so it can attach SCC reads/writes too).
_SCC_OPCODE_FLAGS = {
    # writes-only: implicit-write SOPC/SOP2 family.
    # SOPC compare (no sgpr dst).
    "SCmpEQU32":   ("no_dst",        False, True),
    "SCmpEQU64":   ("no_dst",        False, True),
    "SCmpEQI32":   ("no_dst",        False, True),
    "SCmpGeU32":   ("no_dst",        False, True),
    "SCmpGeI32":   ("no_dst",        False, True),
    "SCmpGtU32":   ("no_dst",        False, True),
    "SCmpGtI32":   ("no_dst",        False, True),
    "SCmpLeU32":   ("no_dst",        False, True),
    "SCmpLeI32":   ("no_dst",        False, True),
    "SCmpLgU32":   ("no_dst",        False, True),
    "SCmpLtU32":   ("no_dst",        False, True),
    "SCmpLtI32":   ("no_dst",        False, True),
    # SOPK compare (no sgpr dst, K is an immediate baked into the opcode).
    "SCmpKEQU32":  ("no_dst",        False, True),
    "SCmpKGeU32":  ("no_dst",        False, True),
    "SCmpKGtU32":  ("no_dst",        False, True),
    "SCmpKLGU32":  ("no_dst",        False, True),
    # SOPC bit test (no sgpr dst, writes SCC = ((ssrc0 >> ssrc1) & 1)).
    # Same shape footgun as SCmp*: rocisa constructs it with `dst=nullptr`,
    # so `getParams()` returns just `[src0, src1]` and the generic rule
    # would misclassify src0 as a write. Currently emitted only OUTSIDE the
    # captured CMS body (KernelWriterAssembly.py:8813 in openSumAtLeastUnroll;
    # 16979/17009/17061 in TDM setup), but claimed here for defense-in-depth
    # so a future move into the captured region cannot silently corrupt the
    # graph. Audit trail: bead rocm-libraries-ckj.
    "SBitcmp1B32": ("no_dst",        False, True),
    # SOP2 with sgpr dst, implicit write SCC.
    "SAddU32":             ("dst_then_srcs", False, True),
    "SAddI32":             ("dst_then_srcs", False, True),
    "SSubU32":             ("dst_then_srcs", False, True),
    "SSubI32":             ("dst_then_srcs", False, True),
    "SAndB32":             ("dst_then_srcs", False, True),
    "SAndB64":             ("dst_then_srcs", False, True),
    "SAndN2B32":           ("dst_then_srcs", False, True),
    "SOrB32":              ("dst_then_srcs", False, True),
    "SXorB32":             ("dst_then_srcs", False, True),
    "SAbsI32":             ("dst_then_srcs", False, True),
    "SAShiftRightI32":     ("dst_then_srcs", False, True),
    "SLShiftLeftB32":      ("dst_then_srcs", False, True),
    "SLShiftLeftB64":      ("dst_then_srcs", False, True),
    "SLShiftRightB32":     ("dst_then_srcs", False, True),
    "SLShiftRightB64":     ("dst_then_srcs", False, True),
    "SLShiftLeft2AddU32":  ("dst_then_srcs", False, True),
    # read+write: carry/borrow chain ops.
    "SAddCU32":            ("dst_then_srcs", True,  True),
    "SSubBU32":            ("dst_then_srcs", True,  True),
    # SAndSaveExec/SOrSaveExec write SCC (per gfx ISA: condition produced
    # from the resulting EXEC mask) and consume the implicit EXEC src,
    # but importantly do NOT read SCC — leave reads_scc=False. Kept here
    # so they're claimed by _SCCRule for the SCC-write modeling.
    "SAndSaveExecB32":     ("dst_then_srcs", False, True),
    "SAndSaveExecB64":     ("dst_then_srcs", False, True),
    "SOrSaveExecB32":      ("dst_then_srcs", False, True),
    "SOrSaveExecB64":      ("dst_then_srcs", False, True),
    # reads-only: SCC consumers.
    "SCSelectB32":         ("dst_then_srcs", True,  False),
    "SCMovB32":            ("dst_then_srcs", True,  False),
    "SCBranchSCC0":        ("no_dst",        True,  False),
    "SCBranchSCC1":        ("no_dst",        True,  False),
}


class _SCCRule:
    """Per-opcode SCC read/write publisher (bead mrj.1).

    Placed BEFORE `_GenericALURule` in `_OPERAND_RULES`: claims every
    SCC-touching scalar opcode (per `_SCC_OPCODE_FLAGS`) and emits its
    register reads/writes plus the SCC sentinel where appropriate.

    Two extract shapes drive the register-side handling:

      * "dst_then_srcs" — same convention as `_GenericALURule`
        (params[0]=write, params[1:]=reads). Used for SAdd/SSub/SCSelect/
        SAndSaveExec/etc.

      * "no_dst" — all register params are reads, no register write.
        Used for SCmp* (which have `dst=nullptr` in rocisa, so
        `getParams()` returns just `[src0, src1]`) and SCBranchSCC*
        (label-only). This avoids the `_GenericALURule` quirk where
        SCmp's src0 would land at params[0] and be misclassified as a
        write.

    For SCC itself, the sentinel singleton is appended to reads/writes
    per `(reads_scc, writes_scc)` — the per-byte latest-writer resolver
    (Phase 2 of build_dataflow_graph) then naturally emits SCC RAW edges
    between producers and consumers, and an intervening SCC clobber
    becomes the new latest writer that breaks the producer's edge to the
    later consumer.

    This sub-task (mrj.1) ONLY publishes the edges. Failure-shape
    wiring (turning the missing SCC edge into a typed Failure via
    `diagnose_missing_edge`) is sub-task mrj.2.
    """

    def applies(self, inst, category=None):
        return type(inst).__name__ in _SCC_OPCODE_FLAGS

    def extract(self, inst, category=None):
        cls = type(inst).__name__
        shape, reads_scc, writes_scc = _SCC_OPCODE_FLAGS[cls]
        try:
            params = list(inst.getParams())
        except Exception:
            params = []

        if shape == "no_dst":
            reg_reads = tuple(p for p in params if _is_register(p))
            reg_writes = ()
        else:  # "dst_then_srcs"
            if params and _is_register(params[0]):
                reg_writes = (params[0],)
                reg_reads = tuple(p for p in params[1:] if _is_register(p))
            else:
                reg_writes = ()
                reg_reads = tuple(p for p in params if _is_register(p))

        if reads_scc:
            reg_reads = reg_reads + (_get_scc_sentinel(),)
        if writes_scc:
            reg_writes = reg_writes + (_get_scc_sentinel(),)
        return reg_reads, reg_writes


class _GenericALURule:
    """Catch-all for vgpr/sgpr ALU instructions not absorbed by an earlier
    rule.

    Covers the P1 register-coverage gaps the prior fallback `_writes(inst)
    == [] / _reads(inst) == []` left wide open: Pack (VCvt*, VPack*, VLShift*,
    VPerm*), VSwap, VAddCO/VAddCCO carry chains, VXor (LRS/LWS pointer-flip),
    SAdd/SSub on sgpr/SRD (GRInc), SMov to m0 (DTL m0 setter), and any
    other CommonInstruction-shaped ALU op.

    Convention (matches Pack rocisa shape: dst at slot 0, srcs follow):
      writes = (params[0],) iff params[0] walks like a register
      reads  = tuple(p for p in params[1:] if _is_register(p))

    Non-register positional params (modifiers, ints, comments, VCC, labels)
    are filtered out by `_is_register`. Branch instructions whose only
    param is a label string therefore yield (reads, writes) == ((), ()) —
    no dataflow contribution, which is the desired behavior for control
    flow.

    Order: this rule MUST be last in `_OPERAND_RULES`. The earlier rules
    (DSLoad/DSStore/DTL/BufferLoad/MFMA/NoDataflow) claim their classes;
    everything left over with a `getParams()` shape lands here.

    Pre-Sub-B this rule was deferred because the legacy resolver yielded
    every prior writer for each reader of a scratch vgpr — Pack scratch
    reuse (v133 across PackA/PackB) blew up to 24,688 false-positive
    cross-side OrderInverteds. Sub-B's per-byte latest-writer resolver
    eliminates that, so the rule is now safe to publish reads/writes for
    every Pack-shaped instruction.

    Scalar-ALU coverage (bead wx9.10): SCSelectB32, SAddU32, SAddCU32,
    SSubU32, SSubBU32, SCmpEQU32 and similar SOP1/SOP2 instructions land
    here. Their sgpr dst (params[0]) and sgpr srcs are tracked, so a
    reversed GRInc-style chain forms RAW edges that `compare_graphs`
    flips into `OrderInvertedFailure`. This is what allowed
    `verify_ascending_order` (CMSValidator.py) to be retired in bead
    wx9.11. Coverage proven by
    `test_dataflow_graph_comparison.py::TestGRIncReorderDetection` and
    `::TestVgprChainReorderDetection`.

    NOT covered here (deliberate scope cut, see downstream beads):
      - VCC carry-out / carry-in (regType=None) — handled by `_VCCRule`
        (bead wx9.9), which precedes this rule in `_OPERAND_RULES`.
      - VSwap symmetric R+W (both operands read AND written) — handled by
        `_VSwapRule` (bead wx9.8), which precedes this rule.

    Cleaned up by `_SCCRule` (mrj.1), which also precedes this rule:
      - SCC implicit read/write for SOPC/SOP1/SOP2/SOPK/branch ops. The
        SCC sentinel resource is published in reads/writes per opcode.
      - SCmp* false-write quirk: SCmp* has no sgpr dst (`dst=nullptr` in
        rocisa) but `getParams()` skips the absent dst and returns just
        `[src0, src1]`, so the generic rule would misclassify `src0` as a
        write at params[0]. `_SCCRule` claims SCmp* opcodes BEFORE the
        generic rule and treats every register-shaped param as a read.
      - SBitcmp1B32 (bead ckj): same `dst=nullptr` shape; claimed by
        `_SCCRule` with shape="no_dst" so `params[0]=ssrc0` is correctly
        treated as a read. Currently dormant in CMS captures (emitted in
        TDM setup + openSumAtLeastUnroll, both outside the captured body)
        but claimed for defense-in-depth.

    DANGER ZONE for new rocisa classes (audit checklist for future PRs):
      Any new CommonInstruction subclass whose constructor passes
      `nullptr` as the `dst` argument (look for the second positional arg
      in the `CommonInstruction(InstType::..., <dst>, ...)` super-call)
      will hit this rule with the SAME footgun: `getParams()` skips the
      absent dst and returns `[src0, ...]`, so `params[0]` is a SOURCE
      that this rule will misclassify as a write. If the new class also
      lands in a captured CMS body region (per `LoopBodyCaptureBuilder`),
      that false write will form phantom RAW edges in the dataflow graph.
      Mitigation: add the class to `_SCC_OPCODE_FLAGS` with shape="no_dst"
      and the appropriate `(reads_scc, writes_scc)` flags. The SCmp*,
      SCmpK*, SCBranchSCC*, and SBitcmp1B32 entries are the existing
      precedents. Audit was bead rocm-libraries-ckj — see commit msg for
      the full grep methodology and the `dst=nullptr` survey of
      `rocisa/rocisa/include/instruction/`.
    """
    def applies(self, inst, category=None):
        # Only real rocisa CommonInstruction-shaped objects expose
        # getParams(). Synthetic _FakeInstBase subclasses in test fixtures
        # do not — they fall through to the empty (reads, writes) default
        # and rely on their own bespoke rules (e.g. _FakePackRule injected
        # by tests via using_pack_rule()).
        return hasattr(inst, "getParams")

    def extract(self, inst, category=None):
        try:
            params = list(inst.getParams())
        except Exception:
            return (), ()
        if not params:
            return (), ()
        writes = (params[0],) if _is_register(params[0]) else ()
        reads = tuple(p for p in params[1:] if _is_register(p))
        return reads, writes


# Order matters: more specific rules first; _GenericALURule is the
# catch-all and MUST come last so earlier rules claim their classes.
# `_SCCRule` (mrj.1) sits BEFORE `_GenericALURule` so it claims SCC-touching
# scalar opcodes and attaches the SCC sentinel to their reads/writes — also
# fixes the SCmp* false-write quirk noted in `_GenericALURule`'s docstring.
_OPERAND_RULES = (
    _DSLoadRule(),
    _DSStoreRule(),
    _DTLBufferLoadRule(),
    _BufferLoadRule(),
    _MFMARule(),
    _NoDataflowRule(),
    _VSwapRule(),       # bead wx9.8 — symmetric R+W on the two operands
    _VCCRule(),         # bead wx9.9 — claims VCC-bearing classes before generic
    _SCCRule(),         # bead mrj.1 — claims SCC-touching scalar opcodes
    _GenericALURule(),
)


def _populate_wrapper(wrapper, category=None) -> None:
    """Run the operand rules over wrapper._rocisa_inst, accumulating
    reads and writes into wrapper.reads / wrapper.writes.

    `category` (the TaggedInstruction.category) lets a rule discriminate
    on emission-time bucket — e.g. MFMARule excludes Pack-categorized
    MFMAInstructions (TF32 emulation pattern) so they're not treated as
    main-loop MFMA producers.

    Only the FIRST matching rule contributes; this prevents e.g. an MFMA
    from being processed by multiple rules.

    Idempotent: rules are pure functions of (inst, category).
    """
    inst = wrapper._rocisa_inst
    for rule in _OPERAND_RULES:
        if rule.applies(inst, category):
            reads, writes = rule.extract(inst, category)
            wrapper.reads = tuple(reads)
            wrapper.writes = tuple(writes)
            return
    wrapper.reads = ()
    wrapper.writes = ()


def _ensure_wrapped(tagged_inst):
    """Return the WrappedInstruction for `tagged_inst`, populating it
    on first access. Used by build_dataflow_graph for tagged_insts that
    were constructed directly (test fixtures) without going through
    LoopBodyCaptureBuilder.append.
    """
    if tagged_inst.wrapped is None:
        tagged_inst.wrapped = WrappedInstruction(tagged_inst.inst)
        _populate_wrapper(tagged_inst.wrapped, category=tagged_inst.category)
    return tagged_inst.wrapped


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


# Lazy-populated factory map: regType character -> RegisterContainer factory
# function (idx, count) -> RegisterContainer. Populated on first call to
# _reg_intersection so this module stays free of a top-level rocisa import.
# vgpr/sgpr/accvgpr factories produce the canonical regName=None numeric
# form that compares equal under value-based __eq__/__hash__; that property
# is what makes the set-based dedup in DataflowGraph.edge_keys() work after
# we start emitting intersection-precise registers on edges.
_NUMERIC_REG_FACTORIES = None


def _ensure_numeric_factories():
    global _NUMERIC_REG_FACTORIES
    if _NUMERIC_REG_FACTORIES is not None:
        return
    from rocisa.container import vgpr, sgpr, mgpr, accvgpr, RegisterContainer, RegName
    _NUMERIC_REG_FACTORIES = {
        "v": vgpr,
        "s": sgpr,
        "acc": accvgpr,
        # mgpr's count is fixed at 1 in practice (m0). Wrap for uniform call shape.
        "m": lambda idx, count=1: mgpr(idx),
        # VCC factory (bead wx9.9): synthetic RegisterContainer for the VCC
        # 64-bit register pair. The _VCCRule emits resources of regType="vcc"
        # so the byte-key resolver and _reg_intersection can both handle them.
        "vcc": lambda idx, count=1: RegisterContainer("vcc", RegName("vcc", []), idx, count),
        # SCC sentinel (bead mrj.1): single-bit hardware status register,
        # modeled as a singleton RegisterContainer with regType="scc". The
        # factory always returns the singleton so equality/hashing across
        # producer-write and consumer-read containers stays stable.
        "scc": lambda idx, count=1: _get_scc_sentinel(),
    }


def _reg_intersection(read_reg, written_reg):
    """Return a new RegisterContainer covering the overlap between
    `read_reg` and `written_reg`, or None if they don't overlap or have
    incompatible naming.

    Mirrors the structure of `_reg_overlaps` but constructs the precise
    overlap subrange. Two reasons this matters:

      1. Diagnostic precision — a downstream Failure formatter that prints
         `edge.resource` shows the actual sub-range the consumer reads,
         not the producer's full write. Example: LR writes v[8:11], MFMA
         reads v[8:9] — edge.resource == v[8:9].

      2. Set-based edge dedup in `DataflowGraph.edge_keys()`. When a
         producer with multiple writes feeds a consumer with multiple
         reads, the per-(producer,consumer,register) tuple needs the
         register field to reflect the actual overlap to remain a stable
         hashable identity. Numeric containers built via vgpr/sgpr/...
         factories use regName=None and compare equal to other
         factory-built containers with the same (regIdx, regNum); that's
         what set dedup relies on.

    Symbolic intersections preserve the regName.name root and adjust the
    offset list to point at the intersection's start.
    """
    if read_reg is None or written_reg is None:
        return None
    if read_reg.regType != written_reg.regType:
        return None

    a_named = read_reg.regIdx == -1 and getattr(read_reg, "regName", None) is not None
    b_named = written_reg.regIdx == -1 and getattr(written_reg, "regName", None) is not None

    if a_named != b_named:
        return None

    if a_named:
        if read_reg.regName.name != written_reg.regName.name:
            return None
        a_off = read_reg.regName.getTotalOffsets() if hasattr(read_reg.regName, "getTotalOffsets") else 0
        b_off = written_reg.regName.getTotalOffsets() if hasattr(written_reg.regName, "getTotalOffsets") else 0
        a_lo, a_hi = a_off, a_off + (read_reg.regNum or 1)
        b_lo, b_hi = b_off, b_off + (written_reg.regNum or 1)
        lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
        if lo >= hi:
            return None
        # Symbolic: same name root, offset = lo, count = hi-lo.
        from rocisa.container import RegisterContainer, RegName
        return RegisterContainer(
            read_reg.regType,
            RegName(read_reg.regName.name, [lo]),
            -1,
            hi - lo,
        )

    # Numeric.
    a_lo = read_reg.regIdx
    a_hi = a_lo + (read_reg.regNum or 1)
    b_lo = written_reg.regIdx
    b_hi = b_lo + (written_reg.regNum or 1)
    lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
    if lo >= hi:
        return None
    _ensure_numeric_factories()
    factory = _NUMERIC_REG_FACTORIES.get(read_reg.regType)
    if factory is None:
        # Unknown regType (shouldn't happen for v/s/m/acc). Return None so
        # the resolver skips this edge rather than emitting a bogus container.
        return None
    return factory(lo, hi - lo)


def _memory_intersection(a, b):
    """Return the overlap MemoryRegion between `a` and `b`, or None.

    Two MemoryRegions overlap when they're in the same `space` AND share
    the same `buffer_id` (symbolic root) AND their `[offset, offset+byte_count)`
    byte ranges intersect. Different buffers with the same space don't
    overlap (e.g., LocalReadAddrA vs LocalReadAddrB are distinct LDS
    halves under symbolic resolution).
    """
    if a.space != b.space or a.buffer_id != b.buffer_id:
        return None
    lo = max(a.offset, b.offset)
    hi = min(a.offset + a.byte_count, b.offset + b.byte_count)
    if lo >= hi:
        return None
    return MemoryRegion(space=a.space, buffer_id=a.buffer_id,
                        offset=lo, byte_count=hi - lo)


def _intersection(a, b):
    """Type-dispatched resource intersection. Returns the overlap (subresource
    of the same type) or None if no overlap (including heterogeneous types
    — a register and a MemoryRegion never overlap).
    """
    if isinstance(a, MemoryRegion) and isinstance(b, MemoryRegion):
        return _memory_intersection(a, b)
    if isinstance(a, MemoryRegion) or isinstance(b, MemoryRegion):
        return None
    # Both are RegisterContainers (or duck-types). Reuse the register path.
    return _reg_intersection(a, b)


def _min_issue_quad_cycles_for(rocisa_inst) -> int:
    """Return the per-instruction quad-cycle issue cost (bead `nk0`).

    Mirrors `ValidatorInstruction.min_issue_quad_cycles()` from CMSValidator.py:
        - Default `min_issue_quad_cycles_base = 1` (CMSValidator.py:298, 327-328).
        - `SNop.min_issue_quad_cycles` adds `wait_state` (CMSValidator.py:645-647).
    Every other validator dataclass keeps the base cost of 1.

    Why duplicated here. CMSValidator imports from ScheduleCapture for typed
    Failure formatters (search 'from Tensile.Components.ScheduleCapture' in
    CMSValidator.py); importing CMSValidator from here would close the cycle.
    Post-`8nz` this helper is the canonical per-instruction cost table — the
    structural-side simulators are deleted.
    """
    if rocisa_inst is None:
        return 1
    if _is_snop(rocisa_inst):
        # Test-fixture path: _FakeSNop exposes `wait_state` directly.
        wait_state = getattr(rocisa_inst, "wait_state", None)
        if wait_state is not None:
            return 1 + int(wait_state)
        # Production rocisa path: SNop stores wait_state as the first param
        # (matches CMSValidator.py:1058-1060: `snop.getParams()[0]`).
        get_params = getattr(rocisa_inst, "getParams", None)
        if callable(get_params):
            try:
                params = get_params()
                if params:
                    return 1 + int(params[0])
            except Exception:
                pass
        return 1
    return 1


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
        issue_cycles=_min_issue_quad_cycles_for(inst),
    )


# Body order for graph construction. Cross-body queue state persists in the
# order ML-1 -> ML -> NGL -> NLL (matching hardware execution order).
_BODY_BUILD_ORDER = (BODY_LABEL_ML_PREV, BODY_LABEL_ML, BODY_LABEL_NGL, BODY_LABEL_NLL)


def build_dataflow_graph(four_part_capture):
    """Build the unified 4-body register dataflow graph from a FourPartCapture.

    Two phases:

    Phase 1 — node construction. Walks bodies in execution order
    (ML-1 -> ML -> NGL -> NLL). Every captured instruction becomes a
    node EXCEPT SWait/SBarrier/SNop (scheduler-choice; sidecar only).
    LCC instructions (SSubU32 + SCmpEQI32, per LCC_AUDIT.md from
    bead 2bu.1) ARE nodes — their per-instruction issue cycles
    contribute to `cumulative_issue_cycles` walks. Cross-body cycle
    counting (sibling beads 2bu.3/4/5) depends on this.
    Per-body sidecar `_graph_nodes` is attached so wait/barrier
    helpers can find sync ops in stream order.

    Phase 2 — edge formation by RESOURCE RESOLUTION. For each consumer's
    read resource R, walk producers in stream-position order
    (SchedulePosition: loop_index, vmfma_index, sub_index) and yield
    every prior writer whose written resource overlaps R. The current
    resolver yields ALL prior overlapping writers (the per-byte
    latest-writer rewrite is tracked as wx9.4.2 / Sub-B).

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

    num_mfma_per_subiter = getattr(four_part_capture, 'num_mfma_per_subiter', 0) or 0

    nodes_by_identity = {}
    nodes_per_body = {label: [] for label in _BODY_BUILD_ORDER}

    # ---------------------------------------------------------------------
    # Phase 1 — node construction + sidecar.
    # ---------------------------------------------------------------------
    for label in _BODY_BUILD_ORDER:
        if label not in captures:
            continue
        body = captures[label]

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

            # Per-body sidecar: every node lives here, including SWait/
            # SBarrier/SNop, so waits_in_window/barriers_in_window can
            # find them.
            nodes_per_body[label].append(node)

            # Cross-graph identity set: only "real" instructions
            # participate (excludes scheduler-choice SWait/SBarrier/SNop).
            # LCC instructions (SSubU32 + SCmpEQI32) ARE included — their
            # per-instruction issue cycles contribute to
            # `cumulative_issue_cycles` walks; cross-body cycle counting
            # (sibling beads 2bu.3/4/5) depends on it.
            if not (_is_swait(inst) or _is_sbarrier(inst) or _is_snop(inst)):
                nodes_by_identity[node.identity] = node

        # Stash per-body GraphNodes on the LoopBodyCapture for the helpers.
        body._graph_nodes = nodes_per_body[label]

    # ---------------------------------------------------------------------
    # Phase 2 — edge formation by resource-name resolution.
    # ---------------------------------------------------------------------
    # Each node's wrapper carries the precomputed (reads, writes) tuples
    # populated by `_populate_wrapper` (rule registry). Resources may be
    # RegisterContainers or MemoryRegions; the type-dispatched
    # `_intersection` handles both.
    edges = []

    # Skip when nothing was captured — e.g., the no-op build_dataflow_graph(None)
    # contract holds but here we have an empty captures map after seeding.
    if nodes_by_identity:
        # Per-byte latest-writer construction. Walk all data-flow nodes in
        # ascending stream-position order; for each node, first emit edges
        # for its reads (consulting the current latest_writer state), then
        # update latest_writer for its writes. A new write to a byte_key
        # OVERWRITES the previous writer — exactly what "latest writer"
        # means, and what kills the phantom-edge bug from scratch-vgpr
        # reuse.
        #
        # NO subiter scoping. A vgpr is one physical register; whoever
        # wrote it most recently in stream order is what every subsequent
        # read sees, regardless of which subiter logically "owns" it.
        # If a kernel writer mis-pipelines a prefetch (e.g., PackA1
        # writes v133 before PackA0's subiter-0 consumer reads it), the
        # resolver faithfully reports PackA1 as the producer — the same
        # garbage value the GPU will read. compare_graphs then surfaces
        # the divergence. Adding per-subiter scoping would HIDE such
        # scheduling bugs to make diagnostics look cleaner — the wrong
        # tradeoff.
        latest_writer = {}  # byte_key -> (writer_node, write_resource)
        sorted_nodes = sorted(nodes_by_identity.values(), key=lambda n: n.position)

        for node in sorted_nodes:
            wrapped = _ensure_wrapped(node.tagged_inst)

            # Phase 2a — reads first: emit one edge per distinct
            # (writer, write_resource) that contributes any byte of any
            # read of this node.
            for read_resource in wrapped.reads:
                if read_resource is None:
                    continue
                for producer, overlap in _resolve_producers(
                    read_resource, node, latest_writer,
                ):
                    is_memory = isinstance(overlap, MemoryRegion)
                    edges.append(DataflowEdge(
                        producer=producer,
                        consumer=node,
                        resource=overlap,
                        edge_kind=("lds_raw_intrawave" if is_memory
                                   else "raw_intrawave"),
                    ))

            # Phase 2b — writes second: update latest_writer for every
            # byte this node covers. Done AFTER reads so a single
            # instruction reading and writing the same register sees its
            # PREVIOUS writer, not itself.
            for write_resource in wrapped.writes:
                if write_resource is None:
                    continue
                for bk in _byte_keys_for_resource(write_resource):
                    latest_writer[bk] = (node, write_resource)

    # =========================================================================
    # SBarrier-edge collectors (cross-wave LDS-reuse)
    # =========================================================================
    # Two patterns mirror the legacy CMSValidator structural rules
    # `apply_must_start_after_barriers` and `apply_barriers` (the former was
    # deleted in bead `ola.2` phase 2; this collector is now the sole source
    # of LR0 -> GR LDS-reuse coverage):
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

    return DataflowGraph(nodes=nodes_by_identity, edges=edges, captures=captures,
                         num_mfma_per_subiter=num_mfma_per_subiter)


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
                    resource = getattr(producer.rocisa_inst, "dst", None)
                else:  # gr_to_lr_lds_reuse
                    resource = getattr(producer.rocisa_inst, "dst", None)

                edges.append(DataflowEdge(
                    producer=producer,
                    consumer=node,
                    resource=resource,
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
#   OrderInvertedFailure       — subject reverses producer/consumer order
#                                 from default schedule (default is canonical)
#   MissingWaitFailure         — no SWait on the right counter in the window
#                                 (carries `nearby_other_counter_waits` when
#                                 wrong-counter SWaits sit in the window;
#                                 the former WaitOnWrongCounterFailure was
#                                 collapsed into this type by bead `hof`)
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
        (e.producer.identity, e.consumer.identity, e.resource, e.edge_kind): e
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
      Phase 2: MissingWaitFailure / WaitInsufficientFailure (mutually
               exclusive); plus MissingBarrierFailure when a wait covers
               but no barrier sits in the post-wait window (LDS-reuse
               edges only). MissingWaitFailure carries
               `nearby_other_counter_waits` populated when SWaitCnts on
               other counters sit in the window — replaces the former
               WaitOnWrongCounterFailure.

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

    # Phase 1 — gating: order check, default schedule as canonical reference.
    # The default schedule IS the canonical order. If default emitted the
    # producer before the consumer and subject emitted them in the opposite
    # relative order, the subject reordered a real dataflow dependency past
    # its producer. Cross-body edges are skipped (different stream-position
    # spaces — can't compare directly).
    ref_p = ref_edge.producer
    ref_c = ref_edge.consumer
    if p_node.body_label == c_node.body_label:
        default_p_before_c = ref_p.position < ref_c.position
        subj_p_before_c = p_node.position < c_node.position
        if default_p_before_c and not subj_p_before_c:
            # Cross-subiter ALU-producer edges are a known false-positive
            # source: a PackA3 (subiter 3) writes a symbolic vgpr that an
            # earlier-subiter MFMA reads under the same symbolic name. The
            # default schedule emits all Packs before all MFMAs (linear
            # within-body); CMS pipelines so subiter-N+1's Pack issues after
            # subiter-N's MFMA — the order inversion across subiters is
            # legitimate pipelining, not a real reorder of a same-subiter
            # dependency. Mirrors the same-subiter gate
            # _classify_edge_coverage uses in within-graph mode.
            nmps = subj_graph.num_mfma_per_subiter
            if (_is_alu_producer(p_node)
                    and _node_subiter(p_node, nmps)
                        != _node_subiter(c_node, nmps)):
                return []  # cross-subiter pipelined dependency — legitimate
            return [OrderInvertedFailure(
                producer=p_node,
                consumer=c_node,
                default_producer_position=ref_p.position,
                default_consumer_position=ref_c.position,
            )]
        if default_p_before_c and subj_p_before_c:
            # Order preserved — fall through to wait/barrier coverage checks.
            pass
        # default has producer at-or-after consumer (e.g., kind_rank-induced
        # edge from default's resolver). Don't flag — subj's order can't be
        # judged against an artifactual default ordering. Falls through to
        # Phase 2 wait coverage if applicable.

    # SCC-typed missing edge (bead mrj.2): if the reference edge's resource
    # is the SCC sentinel and Phase-1's order check passed, the most likely
    # cause is a CLOBBER — an unrelated SCC writer issued between the
    # producer and consumer in the subject schedule, displacing the
    # producer's SCC value. Find that intervening writer in the subject
    # graph (the new SCC producer the consumer pairs with) and emit a
    # typed SCCConflictFailure carrying the producer/consumer/clobber
    # triple.
    #
    # If no intervening SCC writer exists in the subject graph (e.g. the
    # consumer simply lost its SCC edge to the producer for an unrelated
    # reason), fall through to the ALU-producer early-return below — the
    # missing edge is a non-clobber phenomenon that this branch can't
    # classify, and a soft return matches the prior behavior for
    # ALU-immediate producers.
    ref_resource = ref_edge.resource
    if getattr(ref_resource, "regType", None) == "scc":
        # Same-body only. SCC is a single-bit hw status register that is
        # NOT preserved across loop iterations by any compiler convention,
        # so a cross-body SCC edge in the default graph is an aliasing
        # artifact of the per-byte latest-writer resolver running over the
        # SCC sentinel — not a real dataflow dependency. Suppress to
        # avoid false-positive failures on cross-body SCC handoffs.
        if p_node.body_label != c_node.body_label:
            return []
        intervening_writer = None
        for e in subj_graph.edges:
            if (e.consumer.identity == c_id
                    and getattr(e.resource, "regType", None) == "scc"
                    and e.producer.identity != p_id):
                intervening_writer = e.producer
                break
        if intervening_writer is not None:
            return [SCCConflictFailure(
                producer=p_node,
                consumer=c_node,
                intervening_writer=intervening_writer,
            )]
        # No intervening SCC writer found — the consumer's SCC slot is
        # simply unsourced in subj. Fall through to the generic ALU early
        # return so we don't double-emit on a non-clobber miss.

    # 4x4 PackMFMA-as-producer feeding CVTPack-as-consumer: 5-quad-cycle
    # settle window (CDNA 4 ISA 7.6, `_QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1`).
    # Sub-C of bead `or9` migrated this to the graph side. Must run BEFORE
    # the generic `_is_mfma_producer` branch below: `_is_mfma_producer`
    # claims PackMFMA producers (post-`e7w`) and would otherwise route
    # this pair through `_quad_cycle_gap_ok`, which uses
    # `_mfma_finish_cycles_for(producer) == 1` for 4x4 PackMFMAs — too
    # weak by 4 quad-cycles versus the 5-cycle CVT1 visibility window.
    # Bead `8nz` deleted the structural-side mirror
    # (`_handle_min_pack_quad_cycles` /
    # `MFMAPack.min_quad_cycles_before_result_used`); this dispatch is
    # now the only enforcement path.
    if (_is_mfma_pack_producer(p_node)
            and _is_cvt_pack_producer(c_node)):
        ok, expected, actual = _mfma_pack_to_cvt_gap_ok(
            p_node, c_node, subj_graph)
        if not ok:
            return [TimingTooCloseFailure(
                producer=p_node,
                consumer=c_node,
                expected_quad_cycles=expected,
                actual_quad_cycles=actual,
            )]
        return []

    # MFMA-as-producer: governed by quad-cycle issue-timing constraints,
    # not by the dscnt/vlcnt FIFO. Sub-C (wx9.4.3) replaces the prior
    # blanket exemption with an explicit gap check; an MFMA producer whose
    # consumer fires too soon after it is a TimingTooClose violation. Sub-A
    # of bead `e7w` widens the dispatch from `category == "MFMA"` to
    # `_is_mfma_producer` so 4x4 PackMFMAs (categorized Pack* but
    # syntactically MFMAInstruction) are routed here BEFORE the ALU
    # exemption claims them.
    if _is_mfma_producer(p_node):
        nmps = subj_graph.num_mfma_per_subiter
        ok, expected, actual = _quad_cycle_gap_ok(p_node, c_node, nmps, graph=subj_graph)
        if not ok:
            return [TimingTooCloseFailure(
                producer=p_node,
                consumer=c_node,
                expected_quad_cycles=expected,
                actual_quad_cycles=actual,
            )]
        return []

    # CVTPack-as-producer feeding MFMA-as-consumer: 2-quad-cycle settle
    # window (CDNA 4 ISA 7.6, `_QUAD_CYCLES_CVT_BEFORE_MFMA`). Sub-B of
    # bead `35z` migrated this to the graph side; bead `8nz` deleted the
    # structural-side mirror so this dispatch is now the only enforcement
    # path. Must precede the ALU-immediate exemption below (same
    # dispatch-order constraint as the MFMA branch above) so CVTPacks
    # don't get silently waved through. Non-MFMA consumers fall through
    # to the ALU exemption — only the CVT->MFMA edge carries the
    # quad-cycle constraint.
    if _is_cvt_pack_producer(p_node) and _is_mfma_producer(c_node):
        ok, expected, actual = _cvt_to_mfma_gap_ok(p_node, c_node, subj_graph)
        if not ok:
            return [TimingTooCloseFailure(
                producer=p_node,
                consumer=c_node,
                expected_quad_cycles=expected,
                actual_quad_cycles=actual,
            )]
        return []

    # ALU-as-producer (scalar/vector ALU, GRInc, m0 setters): result is
    # immediately visible to the next issued instruction; no SWaitCnt drain
    # applies. Phase 1 already classified any order inversion; nothing else
    # to verify.
    if _is_alu_producer(p_node):
        return []

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
        # No SWait on the expected counter at all in the window. If other-
        # counter SWaits exist, surface them as nearby_other_counter_waits
        # so the user can extend one of them rather than insert a new
        # SWaitCnt; the underlying fix is the same either way.
        failures.append(MissingWaitFailure(
            producer=p_node,
            consumer=c_node,
            counter_kind=expected_counter,
            nearby_other_counter_waits=waits_other,
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
        # Cross-body edges where waits exist that DO drain the producer:
        # this is a loop-carried dataflow handoff — the captured stream
        # has the producer at body N's end, the consumer at body N+1's
        # start, and the SWaitCnt that bridges them drains the producer's
        # counter. The edge is "missing" from subj only because the
        # symbolic register name is reused across iterations and the
        # subj graph paired this consumer with a different (closer)
        # producer. No real classifier bug; suppress.
        if (p_node.body_label != c_node.body_label
                and waits and _any_drains(waits, p_node, subj_graph)):
            return []

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
        # logs the issue without crashing the build. (The historic
        # `.with_legacy_msg(...)` chained call referenced a setter that
        # was planned but never implemented; the bare MissingWaitFailure
        # carries enough info to be actionable.)
        return [MissingWaitFailure(
            producer=p_node, consumer=c_node, counter_kind="unknown",
        )]
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


# --- Quad-cycle constants (single source of truth, post-`8nz`) ---------------
# CDNA 4 ISA section 7.6. These constants used to be mirrored from
# CMSValidator.py; bead `8nz` deleted the structural-side `QUAD_CYCLES_*` /
# `MFMA_TYPE_SWITCH_THRESHOLD_*` constants together with the simulator that
# consumed them (`estimate_quad_cycles`). The values below are now the only
# place these magic numbers live in the code base.
_QUAD_CYCLES_STANDARD_MFMA_FINISH = 3   # Standard MFMA: 3 quad-cycles to finish.
_QUAD_CYCLES_MFMA_4X4_FINISH = 1        # 4x4 (Pack-flavored) MFMA: 1 quad-cycle.
_QUAD_CYCLES_CVT_BEFORE_MFMA = 2        # CVT pack -> MFMA settle window.
_QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1 = 5   # 4x4 PackMFMA -> CVT1 settle window.

# --- MFMA Type-Switch Thresholds (single source of truth, post-`8nz`) --------
# When `cumulative_issue_cycles` observes consecutive MFMAs of DIFFERENT
# classes whose issue gap is below the producer's threshold, it injects a +1
# quad-cycle stall — the consumer is forced to issue one cycle later. The
# thresholds depend on the PRODUCER class: 5 quad-cycles required after a
# standard MFMA, 3 after a 4x4 PackMFMA.
_MFMA_TYPE_SWITCH_THRESHOLD_FROM_STANDARD = 5
_MFMA_TYPE_SWITCH_THRESHOLD_FROM_4X4 = 3


def _mfma_finish_cycles_for(rocisa_inst) -> int:
    """Classify an MFMA-shaped rocisa instruction as standard (3 quad-cycles)
    or 4x4 PackMFMA (1 quad-cycle).

    The rocisa `MFMAInstruction` C++ class accepts a `variant` list at
    construction (`[M, N, K, blk]`, e.g. `[4, 4, 4, 16]` for the 4x4 PackMFMA
    family) but does NOT expose that field as a readable Python attribute via
    the nanobind binding. The rendered assembly string IS canonical and
    stable — every MFMA family renders as `..._<M>x<N>x<K>_<dtype>...`. We
    discriminate the 4x4 family by parsing for the `_4x4x` substring.

    Test fixtures (`_FakeMFMA`) expose a `variant` Python attribute directly
    (default `[32, 32]` for standard MFMAs; tests pass `[4, 4, 4, ...]` to
    model PackMFMAs). The attribute path is checked first so fixtures don't
    have to roundtrip through `str()`.

    Single source of truth for per-MFMA finish cycles (post-`8nz`): the
    structural-side `MFMA.mfma_finish_cycles` / `MFMAPack.mfma_finish_cycles`
    ClassVars were deleted along with the simulator that consumed them.
    Returns the standard value when the instance lacks both signals
    (defensive fallback).
    """
    if rocisa_inst is None:
        return _QUAD_CYCLES_STANDARD_MFMA_FINISH
    # Fast path: test fixtures expose `variant` directly.
    variant = getattr(rocisa_inst, "variant", None)
    if variant is not None:
        try:
            m, n = variant[0], variant[1]
        except (IndexError, TypeError):
            m = n = None
        if m == 4 and n == 4:
            return _QUAD_CYCLES_MFMA_4X4_FINISH
        if m is not None:
            return _QUAD_CYCLES_STANDARD_MFMA_FINISH
    # Production rocisa MFMAInstruction does not expose `variant` as an
    # attribute — discriminate by parsing the rendered assembly form.
    try:
        rendered = str(rocisa_inst)
    except Exception:
        return _QUAD_CYCLES_STANDARD_MFMA_FINISH
    if "_4x4x" in rendered:
        return _QUAD_CYCLES_MFMA_4X4_FINISH
    return _QUAD_CYCLES_STANDARD_MFMA_FINISH


def _is_mfma_pack_producer(producer) -> bool:
    """Return True for a 4x4 PackMFMA producer.

    PackMFMAs (TF32 4x4 emulation) are syntactically `MFMAInstruction` rocisa
    objects but are categorized as `PackA*` / `PackB*` because the macro
    classifier groups them with the surrounding CVT pack chain. Discrimination:
    `category.startswith("Pack")` AND `rocisa_inst` is an MFMA-shaped class.

    Used by `_is_mfma_producer` so PackMFMAs route to the quad-cycle gap
    branch rather than the ALU-immediate-visibility branch — without this,
    the original pre-e7w bug let PackMFMA producers skip timing checks
    entirely (the ALU-producer exemption fired first).
    """
    if not getattr(producer, "category", "").startswith("Pack"):
        return False
    inst = getattr(producer, "rocisa_inst", None)
    if inst is None:
        return False
    return _is_mfma(inst)


def _is_mfma_producer(producer) -> bool:
    """True for any producer subject to MFMA quad-cycle finish-time gating.

    Two shapes:
      - `category == "MFMA"` — the standard MFMA path (everything but the
        TF32 4x4 emulation pack chain).
      - PackMFMA — `category.startswith("Pack")` with an MFMA-shaped rocisa
        instance. Sub-A of bead `e7w` added this branch so the dispatch in
        `_classify_edge_coverage` and `diagnose_missing_edge` claims pack-
        MFMA producers BEFORE the ALU-producer exemption fires.
    """
    if getattr(producer, "category", None) == "MFMA":
        return True
    return _is_mfma_pack_producer(producer)


def _is_cvt_pack_producer(producer) -> bool:
    """True for a CVTPack producer (TF32 v_cvt_pk_bf16_f32 family).

    CVTPacks are categorized `Pack*` (PackA0/PackA1/PackB0/PackB1/PackA3/
    PackB3 depending on the surrounding LR group); discrimination here is
    `category.startswith("Pack")` AND `rocisa_inst` is the
    `VCvtPkF32toBF16` rocisa class. Used by `_classify_edge_coverage` and
    `diagnose_missing_edge` (Sub-B of bead `35z`) so CVTPack-feeding-MFMA
    edges are routed to `_cvt_to_mfma_gap_ok` BEFORE the ALU-immediate
    exemption claims them — same shape as the e7w PackMFMA carve-out, but
    with the CVT class set in place of the MFMA class set.

    Bead `8nz` deleted the structural-side mirror (`_handle_min_pack_quad_cycles`,
    `Pack.min_quad_cycles_before_result_used`, `Pack.estimated_quad_cycles_before_result_used`,
    `estimate_quad_cycles`); the graph-side dispatch in `_classify_edge_coverage`
    is now the only enforcement path for the CVT -> MFMA settle window.
    """
    if not getattr(producer, "category", "").startswith("Pack"):
        return False
    inst = getattr(producer, "rocisa_inst", None)
    if inst is None:
        return False
    return _is_cvt_pack(inst)
def cumulative_issue_cycles(graph, producer, consumer) -> int:
    """Return the exact number of quad-cycles between producer's issue
    completion and consumer's issue start (bead `nk0`).

    Replaces the slot-delta + subiter approximation with cycle-exact
    arithmetic. Originally derived from the (since-deleted, beads `arv` /
    `8nz`) structural-side `precompute_issue_times` /
    `estimate_quad_cycles_precomputed` / `estimate_quad_cycles` simulators
    in CMSValidator.py; this helper is now the canonical implementation.
    Walks the captured body's instruction stream from the producer up to
    (and excluding) the consumer, simulating per-instruction issue
    accumulation including:

    1. Per-instruction issue cost (`node.issue_cycles`, populated by
       `_make_node` from `_min_issue_quad_cycles_for`). Default 1; SNop adds
       wait_state.
    2. MFMA-only contention: each MFMA's `mfma_free_at = current_issue + 1
       + finish_cycles` blocks the next MFMA's issue start.
    3. MFMA type-switch +1 stall: when consecutive MFMAs differ in class
       (standard vs 4x4 Pack-MFMA) AND the gap-since-last-MFMA is below the
       producer-class threshold (FROM_STANDARD=5 / FROM_4X4=3), the consumer
       is delayed by one quad-cycle.

    Returns the gap as `consumer_issue_start - producer_issue_start - 1`
    (the convention previously used by the deleted
    `estimate_quad_cycles_precomputed`).

    Cross-body (bead 2bu.5): when producer and consumer live in different
    captured bodies, the simulator continues across body boundaries in
    `_BODY_BUILD_ORDER` (ML-1 → ML → NGL → NLL — hardware execution
    order). Simulator state (current_issue, mfma_free_at, last_mfma_class,
    last_mfma_issue) persists across boundaries because the bodies issue
    back-to-back; no extra "body boundary" stall is injected. The
    cross-body gap is therefore the cumulative sum of intervening
    instruction issue costs, exactly the same arithmetic the same-body
    walk uses. This is the unified single source of truth — the
    cross-iteration distinction is a red herring; the graph lays out
    instructions in execution order regardless of body membership.

    Falls back to `0` if the body or the producer/consumer cannot be
    located in the captured stream (defensive — should not happen in
    well-formed graphs but keeps unit-test scaffolding resilient).
    """
    captures = getattr(graph, "captures", None)
    if not captures:
        return 0

    # Producer must always be strictly before consumer in stream order. The
    # SchedulePosition `__lt__` compares (loop_index, vmfma_index, sub_index)
    # so this single check covers same-body and cross-body cases uniformly.
    if not (producer.position < consumer.position):
        return 0

    # Build the list of bodies to traverse, starting from the producer's
    # body and continuing forward through `_BODY_BUILD_ORDER` until (and
    # including) the consumer's body. Cross-body unification (bead 2bu.5):
    # the simulator state — `current_issue`, `mfma_free_at`,
    # `last_mfma_class`, `last_mfma_issue` — is preserved across body
    # boundaries because the captured bodies issue back-to-back in
    # hardware execution order. There is no extra "body boundary" stall
    # injected; every per-instruction cost is already accounted for as we
    # walk each body's instructions, so the cross-body gap is just the
    # sum of intervening instruction issue costs.
    p_body_idx = None
    c_body_idx = None
    for i, label in enumerate(_BODY_BUILD_ORDER):
        if label == producer.body_label:
            p_body_idx = i
        if label == consumer.body_label:
            c_body_idx = i
    if p_body_idx is None or c_body_idx is None or p_body_idx > c_body_idx:
        return 0

    p_ti = getattr(producer, "tagged_inst", None)
    c_ti = getattr(consumer, "tagged_inst", None)
    p_key = (producer.position.vmfma_index, producer.position.sub_index)
    c_key = (consumer.position.vmfma_index, consumer.position.sub_index)

    # Walk bodies in execution order. Simulator state persists across
    # boundaries (single source of truth for cycle gaps regardless of
    # body membership — bead 2bu.5).
    mfma_free_at = 0
    current_issue = 0
    last_mfma_class = None
    last_mfma_issue = -1
    p_issue_start = None
    c_issue_start = None
    found_producer = False

    for body_i in range(p_body_idx, c_body_idx + 1):
        label = _BODY_BUILD_ORDER[body_i]
        body = captures.get(label)
        if body is None:
            continue
        instructions = getattr(body, "instructions", None)
        if not instructions:
            continue

        # In producer's body: locate producer and start the walk at it.
        # In subsequent bodies: walk from the start. Consumer may live in
        # any body from producer's onward.
        start_idx = 0
        if not found_producer:
            for i, ti in enumerate(instructions):
                if ti is p_ti or (
                        p_ti is None
                        and getattr(ti, "slot", None) is not None
                        and (getattr(ti.slot, "mfma_index", None),
                             getattr(ti.slot, "sequence", None)) == p_key):
                    start_idx = i
                    found_producer = True
                    break
            if not found_producer:
                # Producer not in this body — defensive; should not happen.
                return 0

        # End_idx: where (if at all) the consumer lives in this body.
        end_idx = len(instructions) - 1
        consumer_idx_in_body = None
        if label == consumer.body_label:
            for i in range(start_idx, len(instructions)):
                ti = instructions[i]
                if ti is c_ti or (
                        c_ti is None
                        and getattr(ti, "slot", None) is not None
                        and (getattr(ti.slot, "mfma_index", None),
                             getattr(ti.slot, "sequence", None)) == c_key):
                    consumer_idx_in_body = i
                    end_idx = i
                    break
            if consumer_idx_in_body is None:
                # Consumer expected in this body but not found.
                return 0

        # Walk start_idx..end_idx with the canonical issue-time simulator.
        for i in range(start_idx, end_idx + 1):
            ti = instructions[i]
            inst = getattr(ti, "inst", None)
            is_mfma = inst is not None and _is_mfma(inst)
            if is_mfma:
                current_issue = max(current_issue, mfma_free_at)
                current_mfma_class = _mfma_finish_cycles_for(inst)
                if last_mfma_class is not None and current_mfma_class != last_mfma_class:
                    gap = current_issue - last_mfma_issue
                    # Threshold is producer-class-keyed: from a 4x4 producer use
                    # FROM_4X4=3, otherwise FROM_STANDARD=5. We discriminate by
                    # the previous MFMA's finish cycles (1 → 4x4 family).
                    threshold = (_MFMA_TYPE_SWITCH_THRESHOLD_FROM_4X4
                                 if last_mfma_class == _QUAD_CYCLES_MFMA_4X4_FINISH
                                 else _MFMA_TYPE_SWITCH_THRESHOLD_FROM_STANDARD)
                    if gap < threshold:
                        current_issue += 1
                mfma_free_at = current_issue + 1 + current_mfma_class
                last_mfma_issue = current_issue
                last_mfma_class = current_mfma_class

            if p_issue_start is None and i == start_idx and label == producer.body_label:
                p_issue_start = current_issue
            if consumer_idx_in_body is not None and i == consumer_idx_in_body:
                c_issue_start = current_issue
                break
            # Per-instruction issue cost. Skip lookup for SWait/SBarrier/SNop
            # whose rocisa instances are not graph nodes — read their cost
            # directly from `_min_issue_quad_cycles_for`. For graph-tracked
            # nodes the cost is identical (default base 1) so either path is
            # cycle-exact.
            current_issue += _min_issue_quad_cycles_for(inst)

        if c_issue_start is not None:
            break

    if p_issue_start is None or c_issue_start is None:
        return 0
    return c_issue_start - p_issue_start - 1


def _quad_cycle_gap_ok(producer, consumer, num_mfma_per_subiter=0, graph=None):
    """Verify that enough quad-cycles separate an MFMA producer from its
    consumer for the producer's result to be visible.

    Returns (ok, expected_quad_cycles, actual_quad_cycles).

    Bead `2bu.3` unification: same-body and cross-body share ONE code
    path that delegates to `cumulative_issue_cycles`. The hardware MFMA
    pipeline does not reset at body boundaries — `mfma_free_at` and the
    type-switch stall carry through — so the cross-body cycle gap is
    just the same simulator extended over the body boundary. The previous
    `body_delta * 1000` cross-body placeholder (which always returned
    ok=True regardless of how tight the boundary actually was) is gone;
    `cumulative_issue_cycles` now walks the unified instruction stream
    across all bodies in `_BODY_BUILD_ORDER` (extended by bead `2bu.5`).

    `num_mfma_per_subiter` is retained as a positional parameter for
    backward compatibility with existing call sites and tests but is no
    longer consulted (the helper has the body context). `graph` is the
    DataflowGraph the producer/consumer belong to; when omitted (or when
    the body can't be located) we degrade gracefully by reporting an
    `actual` of 0 — strictly conservative, will fail the gap check.
    """
    finish = _mfma_finish_cycles_for(getattr(producer, "rocisa_inst", None))
    expected = finish

    if graph is None:
        # No graph passed (degenerate test path): treat as zero-gap. Strict
        # callers always pass `graph=subj_graph`.
        return False, expected, 0

    actual = cumulative_issue_cycles(graph, producer, consumer)
    return actual >= expected, expected, actual


def _cvt_to_mfma_gap_ok(producer, consumer, subj_graph):
    """Verify that enough quad-cycles separate a CVTPack producer from its
    downstream MFMA consumer for the CVT result to be visible.

    Sub-B of bead `35z`. The threshold is fixed at
    `_QUAD_CYCLES_CVT_BEFORE_MFMA == 2` (CDNA 4 ISA 7.6).

    Returns `(ok, expected, actual)` — same triple shape as
    `_quad_cycle_gap_ok` so callers can wrap a single
    `TimingTooCloseFailure(expected, actual)` regardless of the gap kind.

    Bead `2bu.4` unification: same-body and cross-body share ONE code
    path that delegates to `cumulative_issue_cycles`. The previous
    slot-delta formula (`slot_delta * (1 + finish) - 1 + intervening`)
    was DOUBLE-COUNTING — it charged 1 cycle per slot-INDEX gap AND
    +intervening for actual instructions in those slots. The cycle-exact
    walk only counts actual instructions, producing a smaller (more
    conservative) `actual` for densely-populated streams. The previous
    `body_delta * 1000` cross-body placeholder is also gone;
    `cumulative_issue_cycles` walks the unified instruction stream
    across all bodies in `_BODY_BUILD_ORDER` (extended by bead `2bu.5`).
    """
    expected = _QUAD_CYCLES_CVT_BEFORE_MFMA

    if subj_graph is None:
        return False, expected, 0  # Strict: no graph -> conservative fail.

    actual = cumulative_issue_cycles(subj_graph, producer, consumer)
    return actual >= expected, expected, actual


def _mfma_pack_to_cvt_gap_ok(producer, consumer, subj_graph):
    """Verify that enough quad-cycles separate a 4x4 PackMFMA producer from
    its downstream CVTPack (CVT1) consumer for the accumulator to settle.

    Sub-C of bead `or9` (parent epic `w7f`). The threshold is fixed at
    `_QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1 == 5` (CDNA 4 ISA 7.6). This is
    the LARGEST gap among the four section-7.6 quad-cycle constants; the
    4x4 MFMA finish-cycle (1) is shorter than the 5-cycle visibility
    window the CVT1 needs, so this helper enforces a larger min-gap on
    PackMFMA->CVTPack edges than the bare finish would suggest.

    Returns `(ok, expected, actual)` — same triple shape as
    `_quad_cycle_gap_ok` and `_cvt_to_mfma_gap_ok` so callers can wrap a
    single `TimingTooCloseFailure(expected, actual)` regardless of the
    gap kind.

    Approach: CYCLE-EXACT via `cumulative_issue_cycles` (bead `nk0`), the
    same simulator `_quad_cycle_gap_ok` uses. The helper walks the
    captured stream from the producer to the consumer, accumulating
    per-instruction issue costs plus MFMA-specific finish-time and
    type-switch stalls. Bead `8nz` deleted the structural-side
    `precompute_issue_times` simulator that this helper originally
    mirrored; the graph-side path is now the single source of truth for
    the PackMFMA -> CVT settle window.

    Bead `2bu.5`: same-body and cross-body share the SAME code path. The
    cross-iteration distinction is a red herring — the graph has all
    instructions laid out in execution order regardless of which body
    they belong to, so `cumulative_issue_cycles` (extended to walk
    across body boundaries in `_BODY_BUILD_ORDER`) is THE function that
    computes the actual cycle gap. The previous `body_delta * 1000`
    placeholder always-true short-circuit is gone; cross-body PackMFMA
    -> CVT1 edges are now enforced with the same 5-quad-cycle threshold
    as same-body edges.
    """
    expected = _QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1

    if subj_graph is None:
        # Strict: no graph -> conservative fail (cannot compute gap).
        return False, expected, 0

    actual = cumulative_issue_cycles(subj_graph, producer, consumer)
    return actual >= expected, expected, actual


def _is_alu_producer(producer):
    """Producers whose results are immediately visible (no SWaitCnt drain
    required, no quad-cycle gap modeled). Includes scalar/vector ALU,
    GRInc (SAdd-family on SRD), and m0 setters.

    LR/LW (LDS) and GR (vector-memory) producers are NOT ALU — they have
    real wait counters and live outside this set.

    Two category-vs-instance mismatches exist after wx9.4.4 added the
    `_GenericALURule` catch-all:
      - DTL m0 setter: category "GRA"/"GRB" (lives in the GRA emission
        group) but the rocisa class is SMov/SAddU32 — a scalar ALU op
        with no vlcnt to drain. Promote to ALU.
      - TF32 Pack-MFMA: category "PackA0".."PackB3" but the rocisa class
        is MFMAInstruction. _MFMARule excludes Pack-categorized MFMAs
        from main-loop MFMA semantics, and _GenericALURule then publishes
        their reads/writes; the producer behaves as ALU (immediate
        visibility), not as a 4-cycle-finish main MFMA.

    So: classify by category_first (Pack* / PackMFMA → ALU), then by
    instance class for the GR-categorized m0 setter, finally fall back
    to category for cases where rocisa_inst is None (test fixtures).

    Sub-A of bead `e7w` carved out a special case for the TF32 4x4 PackMFMA:
    those are categorized `Pack*` but the rocisa class is `MFMAInstruction`,
    so they DO need the quad-cycle finish-time gap modelled (1 quad-cycle
    for v_mfma_f32_4x4x4_*). Without the carve-out the ALU-immediate
    exemption fired first, the quad-cycle branch never ran for PackMFMA
    producers, and a same-slot PackMFMA->MFMA acc chain silently slipped
    past the timing check. PackMFMAs now route to the MFMA branch via
    `_is_mfma_pack_producer`; the rest of the Pack* category (CVT0/CVT1/
    middle packs / SwapPacks) stays on the ALU exemption.
    """
    cat = producer.category
    if cat.startswith("Pack"):
        # PackMFMA carve-out: pack-categorized but real MFMA → quad-cycle
        # finish gating, not ALU-immediate. Other Pack* (CVT/Middle/Swap)
        # behave as ALU.
        if _is_mfma_pack_producer(producer):
            return False
        return True
    if cat == "MFMA":
        return False
    inst = getattr(producer, "rocisa_inst", None)
    if inst is not None:
        if _is_lr(inst) or _is_lw(inst) or _is_gr(inst) or _is_mfma(inst):
            return False
        # Real ALU instance regardless of category bucket.
        return True
    if cat in PRODUCER_CATEGORIES_LDS or cat in PRODUCER_CATEGORIES_GLOBAL:
        return False
    return True


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
#   - GRAfterLRRule                (deleted in bead ola.2 phase 2)  [LDS-reuse barrier-edges]
#   - GRBeforeLRRule               (deleted in bead ola.1)          [LDS-reuse barrier-edges]
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
    WaitInsufficientFailure as appropriate. (When other-counter SWaits
    sit in the window, MissingWaitFailure carries them in
    `nearby_other_counter_waits`.)

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
    # Bead `dpi`: MiddlePack pair-interleaving is a stream-shape invariant,
    # not an edge-shape invariant, so it's a sibling pass driven from the
    # same entry point (rather than per-edge inside _classify_edge_coverage).
    # Returns the same WrongInterleavingFailure shape the structural-side
    # MiddlePack.validate emits in CMSValidator.py:420-433.
    failures.extend(validate_middle_pack_pair_interleaving(graph))
    return failures


def _classify_edge_coverage(edge, subj_graph, *, raise_on_unexplained=False):
    """Per-edge coverage classifier — same logic diagnose_missing_edge
    runs in compare_graphs, but driven from a single graph rather than
    a missing-edge diff.
    """
    p_node = edge.producer
    c_node = edge.consumer

    # Phase 1 — same-body order check is no longer needed here: Sub-B's
    # per-byte latest-writer resolver only emits edges where producer is
    # before consumer in stream order. Within-graph OrderInverted detection
    # is therefore impossible by construction; the cross-graph classifier
    # in diagnose_missing_edge owns OrderInverted detection (with default
    # positions for diagnostics).

    # 4x4 PackMFMA-as-producer feeding CVTPack-as-consumer: 5-quad-cycle
    # settle window (CDNA 4 ISA 7.6, `_QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1`).
    # Sub-C of bead `or9` migrated this rule to the graph side; bead `8nz`
    # deleted the structural-side mirror so this dispatch is now the only
    # enforcement path. Must run BEFORE the generic `_is_mfma_producer`
    # branch below: PackMFMA producers are claimed by `_is_mfma_producer`
    # (post-`e7w`) and would otherwise route through `_quad_cycle_gap_ok`,
    # whose threshold for 4x4 PackMFMAs (`_mfma_finish_cycles_for == 1`)
    # is too weak by 4 quad-cycles versus the 5-cycle CVT1 visibility
    # window.
    if (_is_mfma_pack_producer(p_node)
            and _is_cvt_pack_producer(c_node)):
        ok, expected, actual = _mfma_pack_to_cvt_gap_ok(
            p_node, c_node, subj_graph)
        if not ok:
            return [TimingTooCloseFailure(
                producer=p_node,
                consumer=c_node,
                expected_quad_cycles=expected,
                actual_quad_cycles=actual,
            )]
        return []

    # MFMA-as-producer: gated by quad-cycle issue timing rather than
    # SWaitCnt (see diagnose_missing_edge). Sub-C (wx9.4.3) replaces the
    # blanket exemption with a quad-cycle gap check. Sub-A of bead `e7w`
    # widens the dispatch from `category == "MFMA"` to `_is_mfma_producer`
    # so PackMFMAs (categorized Pack* but rocisa MFMAInstruction) reach
    # this branch instead of getting silently exempted by `_is_alu_producer`
    # below.
    if _is_mfma_producer(p_node):
        nmps = subj_graph.num_mfma_per_subiter
        ok, expected, actual = _quad_cycle_gap_ok(p_node, c_node, nmps, graph=subj_graph)
        if not ok:
            return [TimingTooCloseFailure(
                producer=p_node,
                consumer=c_node,
                expected_quad_cycles=expected,
                actual_quad_cycles=actual,
            )]
        return []

    # CVTPack-as-producer feeding MFMA-as-consumer: governed by the
    # `_QUAD_CYCLES_CVT_BEFORE_MFMA` (= 2) settle window from CDNA 4 ISA
    # section 7.6. Sub-B of bead `35z` migrated this rule to the graph
    # side; bead `8nz` deleted the structural-side mirror so this dispatch
    # is now the only enforcement path. Must run BEFORE the ALU-immediate
    # exemption
    # below — CVTPacks are categorized `Pack*` and `_is_alu_producer`
    # would otherwise silently absorb them and skip the gap check entirely
    # (this is the same dispatch-order bug Sub-A fixed for PackMFMAs).
    # Restricted to MFMA consumers: a CVTPack feeding a non-MFMA consumer
    # (e.g. another Pack or VXor that uses the converted result for
    # something other than an MFMA operand load) carries no quad-cycle
    # constraint — fall through to the ALU exemption in that case.
    if _is_cvt_pack_producer(p_node) and _is_mfma_producer(c_node):
        ok, expected, actual = _cvt_to_mfma_gap_ok(p_node, c_node, subj_graph)
        if not ok:
            return [TimingTooCloseFailure(
                producer=p_node,
                consumer=c_node,
                expected_quad_cycles=expected,
                actual_quad_cycles=actual,
            )]
        return []

    # ALU-as-producer: results are immediately visible; no wait counter
    # applies. Within-graph order inversions were already handled above.
    if _is_alu_producer(p_node):
        return []

    # Phase 2 — wait coverage.
    expected_counter = counter_for(p_node)
    waits = waits_in_window(subj_graph, p_node.position, c_node.position,
                            counter=expected_counter)
    waits_other = waits_in_window(subj_graph, p_node.position, c_node.position,
                                  exclude_counter=expected_counter)

    failures = []
    wait_failure_emitted = False

    if not waits:
        # See note in _classify_edge_coverage's MissingWaitFailure emit.
        failures.append(MissingWaitFailure(
            producer=p_node, consumer=c_node,
            counter_kind=expected_counter,
            nearby_other_counter_waits=waits_other,
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


# =============================================================================
# MiddlePack pair-interleaving classifier (bead `dpi`)
# =============================================================================
#
# TF32 emulation packs come in groups of 24, the middle 16 of which compute
# bf16 error terms via paired (v_cvt_f32_bf16, v_sub_f32) instructions
# bound to the `MiddlePack` validator dataclass (CMSValidator.py:415,
# PACK_TYPE_MAP entries lines 627-630). Each pair shares a temporary VGPR,
# which the validator-side rule `MiddlePack.validate` enforces by requiring
# the two halves of every pair appear back-to-back in stream order: no
# OTHER MiddlePack (from any category, including the same one) may sit
# between a pair's leader (even index in its category's middle-16 list) and
# its consumer (the next, odd index).
#
# The structural rule (`_hook_up_middle_16_pairs` / `_hook_up_packs_f32`
# in CMSValidator.py) wires `pack.pair_consumer` and `pack.next_scheduled_
# middle_16` then asserts identity at `MiddlePack.validate` time. The
# graph-side equivalent here works directly off the GraphNode stream,
# without any cross-reference to validator-side dataclasses (avoids an
# import cycle: CMSValidator imports ScheduleCapture, not the other way).
#
# Pair-detection algorithm:
#   1. Walk every node in stream order.
#   2. Filter to MiddlePack-class nodes (rocisa class in
#      `_MIDDLE_PACK_CLASS_NAMES`). Group by `category` (PackA0, PackB0,
#      etc.) — categories partition the pack stream.
#   3. Within each category, pair MiddlePacks by adjacency in the
#      category-local stream order: pair (0, 1), (2, 3), (4, 5), ...
#      Each pair's first element is the LEADER; the second is the
#      CONSUMER. (Mirrors `_hook_up_middle_16_pairs` line 1944.)
#   4. For each LEADER, scan the GLOBAL MiddlePack stream and find the
#      first MiddlePack node positioned strictly after the leader. If it
#      isn't the leader's CONSUMER, emit `WrongInterleavingFailure`.
#      (Mirrors `MiddlePack.validate` line 426-433.)


def validate_middle_pack_pair_interleaving(graph):
    """Bead `dpi`: graph-side MiddlePack pair-interleaving check.

    Walks the unified node stream once and enforces the
    pair-leader/pair-consumer adjacency invariant for TF32 middle-16 packs.
    Emits zero-or-more `WrongInterleavingFailure` for each violating pair
    (one failure per pair, never per intervening node).

    Returns a list of `WrongInterleavingFailure` instances. Empty list
    means every middle-16 pair was emitted contiguously in the global
    stream order.

    Coverage parity with the structural rule
    (CMSValidator.py:`MiddlePack.validate` + `_hook_up_middle_16_pairs`):
    same pair-detection (adjacency within category-local middle-16 list),
    same successor scan (next MiddlePack in global stream), same failure
    shape (`pack` / `expected_next` / `actual_next`).
    """
    # 1. Collect MiddlePacks in global stream order.
    middle_packs_global = []
    for node in _all_nodes_in_order(graph):
        inst = getattr(node, "rocisa_inst", None)
        if inst is None:
            continue
        if not _is_middle_pack(inst):
            continue
        # Defensive: only honor nodes whose category looks like a Pack* tag
        # (production resolves PackA0/PackB0/... via PACK_TYPE_MAP). A
        # MiddlePack rocisa class with a non-Pack category would mean the
        # category resolver mis-tagged the instruction; ignore rather than
        # blow up — the resolver is exercised by test_ScheduleCapture.py.
        if not getattr(node, "category", "").startswith("Pack"):
            continue
        middle_packs_global.append(node)

    if not middle_packs_global:
        return []

    # 2. Bucket by category and determine pair-leader/consumer relationships
    #    using category-local adjacency.
    by_category: dict = {}
    for node in middle_packs_global:
        by_category.setdefault(node.category, []).append(node)

    # leader_to_consumer[leader_node] = consumer_node
    leader_to_consumer: dict = {}
    for cat_nodes in by_category.values():
        # Pair (0,1), (2,3), ... — same convention as
        # `_hook_up_middle_16_pairs`. A trailing unpaired leader (odd-length
        # category list) is ignored: the structural rule never sets
        # `pair_consumer` for it (and the production middle-16 always comes
        # in even multiples of 8 per group), so no invariant applies.
        for i in range(0, len(cat_nodes) - 1, 2):
            leader_to_consumer[id(cat_nodes[i])] = cat_nodes[i + 1]

    # 3. For each leader, find the next MiddlePack in global stream order
    #    and compare to expected consumer.
    failures = []
    for global_idx, node in enumerate(middle_packs_global):
        consumer = leader_to_consumer.get(id(node))
        if consumer is None:
            continue
        # Find the next MiddlePack (any category) strictly after this leader
        # in the global stream.
        if global_idx + 1 >= len(middle_packs_global):
            # Leader at end of stream with no follower — the structural
            # rule's `next_scheduled_middle_16` indexing would IndexError
            # here too; treat as missing data, not a failure.
            continue
        actual_next = middle_packs_global[global_idx + 1]
        if actual_next is consumer:
            continue
        failures.append(WrongInterleavingFailure(
            pack=node,
            expected_next=consumer,
            actual_next=actual_next,
        ))
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
            subiter=0,  # CMS macro is subiter-flattened; subiter is encoded in category (e.g. LRA0 vs LRA1)
            slot_kind=slot_kind,
            mfma_index=mfma_index,
        )

    return builder.finalize()


def build_cms_four_part_capture(macro, num_codepaths, tag_by_origin_id,
                                  sync_class, snop_class, mfma_classes,
                                  num_mfma_per_subiter=0):
    """Expand a CMS MAINLOOP macro four ways and assemble a FourPartCapture.

    main_loop[cp] expands with all flags=1 and \\ID=cp for each cp.
    main_loop_prev[cp] is a verbatim clone of main_loop[cp].
    n_gl[0] expands with useGR=0, usePLR=1, useGRInc=1, useLoop=0, \\ID=0.
    n_ll[0] expands with useGR=0, usePLR=0, useGRInc=0, useLoop=0, \\ID=0.

    These flag assignments mirror the CMS dispatch sites at:
      - simdSpecDispatch (KernelWriterAssembly.py, simdSpecDispatch) for main_loop
      - _emitNoLoadLoopBodyCMSMacro (KernelWriter.py) for n_gl/n_ll

    n_gl/n_ll are keyed only at {0: body} because the CMS tail-loop emission
    hard-codes \\ID=0 (see _emitNoLoadLoopBodyCMSMacro docstring for the
    correctness rationale and bead rocm-libraries-9sh for the audit trail).
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
        num_mfma_per_subiter=num_mfma_per_subiter,
    )
