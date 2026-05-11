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

Module dependency edge: this module is the *upstream* leaf of the validator
stack — it never imports `CMSValidator` at runtime. `CMSValidator` imports
from here eagerly. Only string-typed `TYPE_CHECKING` references reach back
into `CMSValidator`, so importing this module never triggers `CMSValidator`
load.
"""

from __future__ import annotations

import functools
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from Tensile.Components.register import Register

if TYPE_CHECKING:
    # Imported only for type hints — CMSValidator imports from this module at
    # runtime, so a hard import here would create a cycle. PEP 563
    # (from __future__ import annotations) keeps these names as strings at
    # runtime; resolution is on-demand for static analyzers only.
    # Only `GraphNode` survives br4.10 cleanup — it's the sole symbol still
    # consumed by an actual annotation in this file (`_resolve_producers`).
    from Tensile.Components.CMSValidator import GraphNode


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
    """Default and CMS captures disagree on which instructions exist.

    Raised by `compare_graphs` when the two graphs' node-identity sets
    diverge — see `CMSValidator.py`. The capture pipeline is the single
    source of truth for which bodies are present; absent bodies are
    treated as "this body was not emitted" rather than as an error.
    """


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
    emission order, continuing across subiters within the same bucket so
    that the bridge's `(slot.mfma_index, slot.sequence)` lex sort yields
    a collision-free per-body monotonic `stream_index` for SchedulePosition.
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
    __slots__ = ("_rocisa_inst", "reads", "writes",
                 "read_slots", "write_slots")

    def __init__(self, rocisa_inst):
        self._rocisa_inst = rocisa_inst
        self.reads = ()
        self.writes = ()
        # Positional operand-slot indices, parallel to `reads` / `writes`.
        # Each slot is the 0-based position of the operand in the producing
        # rocisa instruction's per-side accessor (`getDstParams()` for
        # writes, `getSrcParams()` for reads). Allocation-invariant by
        # construction (small integer, not a register reference).
        # Threaded into the cross-graph edge-identity tuple via
        # `DataflowEdge.src_operand_slot` / `sink_operand_slot` (see
        # rocm-libraries-wx9.3 phase 3, memo §6.1 step 1).
        self.read_slots = ()
        self.write_slots = ()

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
        new.read_slots = self.read_slots
        new.write_slots = self.write_slots
        memo[id(self)] = new
        return new

    def __copy__(self):
        new = WrappedInstruction(self._rocisa_inst)
        new.reads = self.reads
        new.writes = self.writes
        new.read_slots = self.read_slots
        new.write_slots = self.write_slots
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

    `wrapped` is a mandatory WrappedInstruction holding the underlying rocisa
    instruction and its `reads`/`writes` tuples for dataflow-graph edge
    formation. Production callers (LoopBodyCaptureBuilder.append) construct
    the wrapper at append time and finalize() then populates reads/writes via
    `_populate_wrapper`. Test fixtures must construct
    `TaggedInstruction(wrapped=WrappedInstruction(inst), ...)` directly;
    `make_capture` in dataflow_fixtures populates reads/writes for them.

    The underlying rocisa instruction is reachable via `wrapped.rocisa_inst`.
    """
    wrapped: "WrappedInstruction"
    category: str
    slot: SlotKey

    def render(self) -> str:
        """Render this instruction for a failure label.

        Minimal CMS-side implementation of the `TaggedInstructionLike`
        Protocol declared in `Tensile.Components.Timeline`. Returns the
        `category[N]`-style label shape that `cms_node_label` produces
        today; the per-stream `[N]` index requires body context that this
        method does not have, so a bare `category` is returned. Sub-bead
        `rocm-libraries-3dy` (5gd.B.1) refines this to the source-aware
        per-category-stream `[N]` rendering.
        """
        return self.category


@dataclass
class LoopBodyCapture:
    """One scheduled loop body as a flat ordered stream of tagged instructions.

    `name_to_idx` is a body-local symbolic-name -> base register index map,
    populated from the writer's RegSet directives during capture (see
    `collect_regset_stream` and `expand_cms_macro` for the producer side).
    The map is consumed by `_byte_keys_for_resource` in
    `build_dataflow_graph` Phase 2: a symbolic operand whose bare name is in
    the table is resolved to a numeric byte-key, ensuring two operands
    referring to the same physical register under different syntactic forms
    (one symbolic, one numeric) collapse to a single key. Symbolic refs
    whose name is NOT in the table fall through to the legacy
    symbolic-keying behavior unchanged.

    Each FourPartCapture body has its own `name_to_idx` so cross-body name
    collisions (e.g., the same vgpr name rebound by an `undefineSgpr` +
    redefine pair on a different body) cannot bleed across bodies.
    """
    instructions: list
    name_to_idx: dict = field(default_factory=dict)


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
    docstring for the audit trail).

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
    # to derive each MFMA's subiter index from `slot.mfma_index //
    # num_mfma_per_subiter` (read from `node.tagged_inst.slot.mfma_index`,
    # NOT from SchedulePosition.stream_index — stream_index is a
    # bridge-collapsed sort key, not the kernel-writer's MFMA slot id).
    # Both default-side and CMS-side construction sites should pass
    # writer.states.numMfmaPerIter (upstream Tensile naming retains "Iter"
    # but it refers to the inner unroll subiteration here).
    # Defaults to 0 ("don't split MFMAs by subiter"); the resolver then keeps
    # all MFMAs at subiter 0, which loses cross-subiter PLR dataflow edges.
    # Test fixtures may safely leave it unset.
    num_mfma_per_subiter: int = 0
    # Per-architecture timing profile resolved via
    # `_resolve_arch_profile_for_isa(kernel["ISA"])`. `None` means
    # "no profile registered for this kernel's ISA; timing-related
    # validation is skipped." Production callers must resolve and pass
    # explicitly; test fixtures must either pass an explicit profile
    # (e.g. `_DEFAULT_CDNA4_ARCH_PROFILE`) or accept that timing checks
    # will be skipped. Tracked: `rocm-libraries-zkzw`.
    arch_profile: Optional[ArchProfile] = None


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
    """Position in the instruction schedule. Fields ordered for tuple-style comparison.

    Collapsed shape (rocm-libraries-5v4u): the historical
    `(loop_index, vmfma_index, sub_index)` triple is replaced by
    `(loop_index, stream_index)`. The CMS bridge (`make_position` /
    `assign_stream_indices_for_body`) computes a single monotonic
    `stream_index` per body by walking events in the existing
    `(slot.mfma_index, slot.sequence)` lex order, folding tie-breaking into
    the index. Source-aware display values (CMS `mfma_index`, asm
    `stream_pos`) live on `tagged_inst`, NOT on SchedulePosition.

    `loop_index` keeps its name (rather than `body_index`) because the
    project will eventually model pre-loop / post-loop / nested loops, where
    the field truly indexes loop scopes.
    """
    # Which loop iteration this instruction belongs (larger index means later iteration)
    loop_index: int
    # Monotonic per-body stream-emission index. Computed at the bridge by
    # lex-sorting body events on `(slot.mfma_index, slot.sequence)` and
    # assigning 0, 1, 2, ... in that order. Lex sort over
    # `(loop_index, stream_index)` therefore continues to give global
    # stream order across bodies.
    stream_index: int

    def __lt__(self, other: 'SchedulePosition') -> bool:
        if self.loop_index == other.loop_index:
            return self.stream_index < other.stream_index
        return self.loop_index < other.loop_index


def assign_stream_indices_for_body(instructions: list) -> Dict[int, int]:
    """Walk a body's TaggedInstruction list and return `{id(ti): stream_index}`.

    Stream order is the lex sort of `(slot.mfma_index, slot.sequence)` over
    all events in the body. By construction `LoopBodyCaptureBuilder.append`
    issues `slot.sequence` per `(slot_kind, mfma_index)` bucket starting at
    0, with the sequence counter shared across subiters — so the natural
    list-append order matches the lex order in production. This helper
    re-derives stream_index from the slot tuple anyway so callers that
    construct synthetic captures with out-of-order `instructions` lists
    still get a canonical assignment.
    """
    sorted_tis = sorted(
        instructions,
        key=lambda ti: (ti.slot.mfma_index, ti.slot.sequence),
    )
    return {id(ti): i for i, ti in enumerate(sorted_tis)}


def make_position(body_label: str, stream_index: int) -> SchedulePosition:
    """Construct a SchedulePosition from a body label and a precomputed
    per-body `stream_index`.

    The body_label maps to loop_index via BODY_LABEL_TO_LOOP_INDEX so
    cross-body ordering is well-defined. `stream_index` is computed by the
    caller (typically via `assign_stream_indices_for_body`) by walking
    body events in their natural slot lex order.
    """
    return SchedulePosition(
        loop_index=BODY_LABEL_TO_LOOP_INDEX[body_label],
        stream_index=stream_index,
    )


# Failure hierarchy + FailureNodeLabel + CMS-side label/iter-delta helpers
# moved to Tensile.Components.CMSValidator (br4.4). All functions that
# consumed them (compare_graphs, diagnose_missing_edge,
# validate_edge_wait_coverage, _classify_edge_coverage,
# validate_middle_pack_pair_interleaving) have since been moved to
# CMSValidator as well (sub-beads 5-9). The dependency edge is now one-way:
# CMSValidator imports from ScheduleCapture; ScheduleCapture has zero
# reverse-imports back.


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
    because the bridge (`assign_stream_indices_for_body`) lex-sorts events
    on `(slot.mfma_index, slot.sequence)` to assign monotonic
    `stream_index` values; if `sequence` reset per-subiter, two events in
    different subiters could share a `(mfma_index, sequence)` pair and
    collide.

    finalize() runs capture-pipeline guards before returning the capture:
      - rocisa wiring: every TaggedInstruction.wrapped.rocisa_inst is non-None
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
        # Wrap eagerly: WrappedInstruction is now mandatory on TaggedInstruction.
        # finalize() populates reads/writes via _populate_wrapper for survivors.
        self._instructions.append(TaggedInstruction(
            wrapped=WrappedInstruction(inst), category=category, slot=slot,
        ))

    def finalize(self):
        for ti in self._instructions:
            inst = ti.wrapped.rocisa_inst
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
            # Populate reads/writes so build_dataflow_graph picks up dataflow
            # without needing per-call extraction. The wrapper is the single
            # source of (reads, writes) — DTL m0 reads, MFMA acc writes,
            # LDS-address vgpr reads etc. all flow through `_populate_wrapper`'s
            # rule registry. The WrappedInstruction itself was constructed in
            # append(); we only need to populate it now that the instruction
            # has cleared the SMEM/flat/store guards.
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
#
# br4.6 moved the FIFO simulator (counter_for, _swait_drains,
# _all_nodes_in_order, waits_in_window, barriers_in_window, _queue_depth_at,
# _producer_queue_position, _wait_drains_producer, _any_drains,
# _first_insufficient, _last_drain), the producer-classifier helpers
# (_is_mfma_pack_producer, _is_mfma_producer, _is_cvt_pack_producer,
# _is_alu_producer), the cycle-gap helpers (_mfma_finish_cycles_for,
# cumulative_issue_cycles, _quad_cycle_gap_ok, _cvt_to_mfma_gap_ok,
# _mfma_pack_to_cvt_gap_ok), the identity / class-tag helpers (_class_tag,
# _class_tag_from_category, _split_category_iter, _node_subiter,
# _identity_for, _min_issue_quad_cycles_for, _make_node), and the related
# constants (PRODUCER_CATEGORIES_*, SWAIT_CATEGORY, SBARRIER_CATEGORY,
# _BODY_BUILD_ORDER) into CMSValidator.py. The remaining graph-builder
# entry points below import them via narrow lazy imports inside their
# function bodies.


# =============================================================================
# Per-instruction shape extractors
# =============================================================================
# Each function returns enough state for the graph builder to:
#   1. Build a content-based identity (for cross-graph diff)
#   2. Discover what registers the instruction reads/writes (for edges)
#
# Detection by class-name string keeps this module free of hard rocisa
# imports — both tests and production wire real rocisa instances.

# Class names (as returned by type(inst).__name__) recognized by the builder.
_LR_CLASS_NAMES = {
    # Real rocisa LR classes: DSLoadB32 / DSLoadB64 / DSLoadB128 / DSLoadB256
    "DSLoadB32", "DSLoadB64", "DSLoadB128", "DSLoadB256",
    # Generic class umbrella (for isinstance fallback if needed)
    "DSLoadInstruction",
}
_LW_CLASS_NAMES = {
    "DSStoreB8", "DSStoreB16", "DSStoreB32", "DSStoreB64", "DSStoreB128",
    # Wider DSStore variants from rocisa/include/instruction/mem.hpp. All
    # are subclasses of DSStoreInstruction with the same Python ctor
    # signature `(dstAddr, src, ds, comment)`. _DSStoreRule already
    # handles them correctly via `getParams()` slot 0 (lds_addr) + slot 1
    # (src_data). Listed here defensively so the type-name dispatch in
    # `_is_lw` does not misclassify them as UNKNOWN if a CMS kernel
    # emits them.
    "DSStoreU16", "DSStoreB96", "DSStoreB192", "DSStoreB256",
    # D16HI / B8HID16 partial-half-word LDS stores. These are real
    # DSStore subclasses (rocisa/include/instruction/mem.hpp) with the
    # same Python constructor signature (dstAddr, src, ds, comment) as
    # DSStoreB16. The "D16HI"/"B8HID16" semantic only changes which
    # 16/8 bits of the LDS word are written — it does NOT change
    # register-side dataflow: the source vgpr is read in full (not as a
    # partial read; the upper or lower bits are selected by the LDS
    # write itself, not by the register read), and no register is
    # written. So the existing _DSStoreRule
    # (reads = (lds_addr, src_data); writes = ()) is correct as-is.
    "DSStoreD16HIB16", "DSStoreB8HID16",
    # DSStore2B32 — `ds_store2_b32` writes TWO independent 32-bit values
    # to LDS in a single instruction. Python ctor signature is
    # `(dstAddr, src0, src1, ds, comment)` (rocisa/src/instruction/mem.cpp
    # bindings; rocisa/include/instruction/mem.hpp::DSStore2B32). Unlike
    # the single-src DSStore* family, both src0 AND src1 are real register
    # operands. No special-case extract logic is needed because
    # `DSStoreInstruction::getSrcParams` (mem.hpp:883) already returns
    # `{dstAddr, src0, src1}` for ALL DSStoreInstruction subclasses; the
    # generic `_DSStoreRule.extract` routes that through
    # `_operands_with_slots` and `Register.is_register`, which correctly
    # captures both src registers for DSStore2B32 (and filters out the
    # None src1 hole for single-src variants like DSStoreB32).
    "DSStore2B32",
    "DSStoreInstruction",
}
_GR_CLASS_NAMES = {
    # rocisa BufferLoad classes
    "BufferLoadB32", "BufferLoadB64", "BufferLoadB128",
    "GlobalLoadB32", "GlobalLoadB64", "GlobalLoadB128",
    "BufferLoadInstruction", "GlobalLoadInstruction",
    "GlobalReadInstruction",
}
_MFMA_CLASS_NAMES = {
    "MFMAInstruction",
}
_SWAIT_CLASS_NAMES = {
    "SWaitCnt",
}
_SBARRIER_CLASS_NAMES = {
    "SBarrier",
}
_SNOP_CLASS_NAMES = {
    "SNop",
}
# SSetPrior (s_setprio) — sets the wave priority. Pure scheduling-control
# scalar instruction with NO register dataflow: the only constructor param
# is an `int prior` (rocisa/include/instruction/common.hpp::SSetPrior), and
# `getParams()` returns `{prior}` — no RegisterContainer reads or writes.
# Treated like SNop/SBarrier/SWaitCnt for validator purposes:
#   - claimed by `_NoDataflowRule` (no reads/writes contributed),
#   - excluded from the cross-graph data-flow node identity set
#     (`build_dataflow_graph` Phase 1, around `:2949`),
#   - issue cost is the default 1 quad-cycle (NO wait_state add — unlike
#     SNop which encodes a wait_state in its first param).
# Witnessed under gfx950 HSS MT256x256x64 #2 and BBS Range MT256x256x64
# under CMS=1+PGR=2+PLR=1+DTL=T/T.
_SSETPRIO_CLASS_NAMES = {
    "SSetPrior",
}
# CVT-pack rocisa classes (TF32 emulation: v_cvt_pk_bf16_f32 and friends).
# Used by `_is_cvt_pack` to identify CVTPack producers whose results feed
# downstream MFMAs and need the 2-quad-cycle settle window
# (`_QUAD_CYCLES_CVT_BEFORE_MFMA` in this file). Mirrors the class-name-set
# lookup pattern used for MFMA / LR / LW / GR; the production rocisa class
# is `VCvtPkF32toBF16` (CMSValidator.py:676 PACK_TYPE_MAP entry binds it to
# the `CVTPack` validator dataclass). Test fixtures use plain
# `_FakeCVTPack` if and when they need to exercise this branch with a
# non-rocisa stub; today the production class is the only entry.
_CVT_PACK_CLASS_NAMES = {
    "VCvtPkF32toBF16",
}

# MiddlePack rocisa classes (TF32 emulation: the 16 instructions in the middle
# of each 24-pack group that compute the bf16 error terms via paired
# v_cvt_f32_bf16 + v_sub_f32 instructions). Used by `_is_middle_pack` and
# `validate_middle_pack_pair_interleaving` to identify the pair-leader /
# pair-consumer relationships independently of the structural `MiddlePack`
# validator-dataclass binding (CMSValidator.py:627-630 PACK_TYPE_MAP entries).
# The pairing semantics: middle-16 packs in each category (e.g. PackA0) are
# paired adjacently in stream order — pair (0,1), pair (2,3), etc. — and each
# pair's two halves share a temporary VGPR, so no OTHER middle-16 pack (from
# any category, even another one in this category) may appear between them in
# the global stream.
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
    `validate_middle_pack_pair_interleaving` uses this discriminator to
    identify pair leaders / consumers from the GraphNode stream without
    re-importing the validator dataclass (which would create an import
    cycle). Test fixtures may use any of the four production rocisa
    classes, or a stub class whose `type(...).__name__` matches one of
    the names in `_MIDDLE_PACK_CLASS_NAMES`.
    """
    return type(inst).__name__ in _MIDDLE_PACK_CLASS_NAMES


def _is_cvt_pack(inst):
    """True for CVT-pack rocisa instances (`v_cvt_pk_bf16_f32` family).

    These are the TF32 CVT0/CVT1 packs that bind to the validator-side
    `CVTPack` dataclass via PACK_TYPE_MAP (CMSValidator.py:676). When such
    an instruction writes a vgpr that a downstream MFMA reads, the CDNA 4
    ISA (section 7.6) requires 2 quad-cycles between them
    (`_QUAD_CYCLES_CVT_BEFORE_MFMA` in this file). The graph-side
    enforcement of this rule routes CVTPack producers through
    `_cvt_to_mfma_gap_ok` instead of the ALU-immediate exemption.
    """
    return type(inst).__name__ in _CVT_PACK_CLASS_NAMES


def _is_swait(inst):
    return type(inst).__name__ in _SWAIT_CLASS_NAMES


def _is_sbarrier(inst):
    return type(inst).__name__ in _SBARRIER_CLASS_NAMES


def _is_snop(inst):
    return type(inst).__name__ in _SNOP_CLASS_NAMES


def _is_ssetprio(inst):
    """SSetPrior — wave-priority scalar op, no register dataflow. See
    `_SSETPRIO_CLASS_NAMES` for the rationale."""
    return type(inst).__name__ in _SSETPRIO_CLASS_NAMES


# Stable hashable signatures for RegisterContainers are obtained via
# `Register.from_rocisa(rc).signature()` — see Tensile/Components/register.py.

# Per-shape positional `getParams()` extractors removed in q9j: each
# rule now reads `inst.getSrcParams()` / `inst.getDstParams()` directly
# (rocisa nanobind-bound) and filters through `Register.is_register`.


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


# Stream-position ordering / identity helpers (_class_tag,
# _class_tag_from_category, _split_category_iter, _node_subiter,
# _TRAILING_DIGITS_RE) moved to CMSValidator.py in br4.6.


def _byte_keys_for_resource(resource, name_to_idx=None):
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

    `name_to_idx` is the body-local symbolic-name -> numeric-base lookup
    populated from the writer's RegSet directives (see
    `LoopBodyCapture.name_to_idx`). When supplied, a symbolic operand
    whose bare name appears in the table is resolved to a NUMERIC byte-key
    (same shape as the `regIdx >= 0` branch), so two operands referring
    to the same physical register under different syntactic forms
    (symbolic vs. numeric) collapse to a single byte-key. Symbolic refs
    whose name is NOT in the table fall through to the legacy
    symbolic-keying behavior unchanged — preserving correctness for
    registers genuinely emitted symbolic-only.
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
    if name_to_idx:
        bare = name[4:] if name.startswith(("vgpr", "sgpr")) else name
        resolved = name_to_idx.get(bare)
        if resolved is not None:
            return tuple((rt, resolved + base + i) for i in range(count))
    return tuple((rt, name, base + i) for i in range(count))


def _resolve_producers(read_resource: Any, consumer: GraphNode, latest_writer: Dict[Any, Tuple[GraphNode, Any, int]], name_to_idx=None):
    """Yield (producer_node, overlap, intra_operand_byte_offsets,
    src_operand_slot) tuples for `consumer`'s read of `read_resource`.

    `latest_writer` is the per-byte map maintained by build_dataflow_graph
    Phase 2: byte_key -> (writer_node, write_resource, write_slot). For
    each byte the read covers, look up the latest writer; group bytes by
    (writer_node, write_resource, write_slot) so distinct writes (e.g., a
    multi-write producer feeding a wide read) each emit one edge — and
    so a writer that emitted the SAME register from two different
    positional slots (the symmetric VSwap case: dst at slot 0, src at
    slot 1, both naming the same physical reg in the swap-with-self
    edge) yields two distinct producer entries.

    The yielded `overlap` is the intersection of read_resource with the
    writer's actual write_resource — same precision the old resolver
    yielded, so diagnostic formatters still work.

    The yielded `intra_operand_byte_offsets` is the sorted tuple of byte
    POSITIONS WITHIN `read_resource` (0..N-1) that this writer satisfies.
    Allocation-invariant by construction: the position is relative to the
    read operand's start, not an absolute physical-register byte-key.

    The yielded `src_operand_slot` is the writer's POSITIONAL operand
    index — the slot at which the producer's `getDstParams()` (or
    legacy positional emit) emitted this write. Allocation-invariant by
    construction (small integer, not a register reference). This — paired
    with the consumer's read-side slot, threaded into the cross-graph
    edge identity by `build_dataflow_graph` — is what makes a within-
    graph reorder of two instructions sharing a register detectable
    while keeping across-graph register-renames equal. See
    `DataflowEdge.src_operand_slot` / `sink_operand_slot` and
    rocm-libraries-wx9.3 phase 3, memo §6.1 step 1.

    `name_to_idx` is forwarded to `_byte_keys_for_resource` so symbolic
    reads resolve to the same numeric byte-keys as the corresponding
    numeric writes (and vice-versa); see `_byte_keys_for_resource` for
    semantics.
    """
    # Group key includes write_slot so a producer that wrote the SAME
    # physical register from two distinct positional slots yields two
    # producer entries (carries the slot through the edge identity).
    writer_groups = {}
    for i, bk in enumerate(_byte_keys_for_resource(read_resource, name_to_idx=name_to_idx)):
        entry = latest_writer.get(bk)
        if entry is None:
            continue
        writer_node, write_res, write_slot = entry
        key = (id(writer_node), id(write_res), write_slot)
        if key not in writer_groups:
            writer_groups[key] = (writer_node, write_res, write_slot, [i])
        else:
            writer_groups[key][3].append(i)

    for writer_node, write_res, write_slot, offsets in writer_groups.values():
        overlap = _intersection(read_resource, write_res)
        if overlap is not None:
            yield (writer_node, overlap, tuple(offsets), write_slot)


# _identity_for moved to CMSValidator.py in br4.6.


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


# -----------------------------------------------------------------------------
# Operand-rule helpers
# -----------------------------------------------------------------------------
# Most rules below collapse to "read getSrcParams(), write getDstParams(),
# filter through Register.is_register". The rocisa C++ side encodes per-shape
# semantics directly (see Q9J_SCOPE_REASSESSMENT.md §2):
#
#   * DSLoadInstruction::getDstParams = {dst}, getSrcParams = {srcs}
#   * DSStoreInstruction::getDstParams = {}, getSrcParams = {dstAddr, src0, src1}
#   * MUBUFReadInstruction::getDstParams = {dst}, getSrcParams = {vaddr, saddr, soffset}
#   * MFMAInstruction::getDstParams = {acc}, getSrcParams = {a, b, acc2}
#     (acc2 defaults to acc for in-place RMW — matches the validator's
#      prior `reads = (a, b, acc)` synthesis for that case, and is
#      strictly more correct for the out-of-place case.)
#   * VSwapB32::getDstParams = getSrcParams = {dst, src} — symmetric R+W
#     already encoded in C++ (rocisa/include/instruction/common.hpp:5179-5194).
#
# Synthetic test fixtures (`_Fake*` in `Tensile/Tests/unit/dataflow_fixtures.py`)
# implement matching `getDstParams` / `getSrcParams` methods so the rules
# can call them uniformly on real and fake instructions.


def _operands_via_accessors(inst):
    """Return `(reads, writes)` from `inst.getSrcParams()` /
    `inst.getDstParams()` filtered through `Register.is_register`.

    The single-source-of-truth pattern shared by `_DSLoadRule`,
    `_DSStoreRule`, `_BufferLoadRule`, `_MFMARule`, `_VSwapRule`, and
    `_GenericALURule`. Returns `((), ())` if the instance lacks either
    accessor (e.g. a stray non-rocisa duck-typed object).
    """
    reads, _, writes, _ = _operands_with_slots(inst)
    return reads, writes


def _operands_with_slots(inst):
    """Return `(reads, read_slots, writes, write_slots)` from
    `inst.getSrcParams()` / `inst.getDstParams()` filtered through
    `Register.is_register`.

    The slot index is the 0-based POSITION of the operand within the
    per-side accessor (`getSrcParams()` for reads, `getDstParams()` for
    writes). Slots are preserved through `Register.is_register` filtering
    so non-register positional params (modifiers, ints, None) cause holes
    in the slot sequence rather than re-indexing — that preserves the
    positional contract the cross-graph edge identity depends on
    (rocm-libraries-wx9.3 phase 3, memo §6.1 step 1).

    Convention: for `MFMA(acc, a, b, [acc2])`, write-slot 0 = acc,
    read-slot 0 = a, read-slot 1 = b, read-slot 2 = acc2. For
    `VSwap(dst, src)` (symmetric R+W), write-slot 0 = dst, write-slot 1 =
    src, read-slot 0 = dst, read-slot 1 = src — same operand has the
    same slot on both sides because `getDstParams() == getSrcParams()`.
    """
    if not (hasattr(inst, "getSrcParams") and hasattr(inst, "getDstParams")):
        return (), (), (), ()
    try:
        srcs = inst.getSrcParams()
        dsts = inst.getDstParams()
    except Exception:
        return (), (), (), ()
    read_pairs = [(i, r) for i, r in enumerate(srcs) if Register.is_register(r)]
    write_pairs = [(i, w) for i, w in enumerate(dsts) if Register.is_register(w)]
    reads = tuple(r for _, r in read_pairs)
    read_slots = tuple(i for i, _ in read_pairs)
    writes = tuple(w for _, w in write_pairs)
    write_slots = tuple(i for i, _ in write_pairs)
    return reads, read_slots, writes, write_slots


class _DSLoadRule:
    """DSLoadB* — `getDstParams() = {dst}`, `getSrcParams() = {src_lds_addr, ...}`.
    Both filtered through `Register.is_register` (modifiers/None drop out).
    """
    def applies(self, inst, category=None): return _is_lr(inst)
    def extract(self, inst, category=None):
        return _operands_with_slots(inst)


class _DSStoreRule:
    """DSStoreB* — `getDstParams() = {}` (LDS write has no register dst);
    `getSrcParams() = {dstAddr, src0, src1}` (None src1 filtered out).
    """
    def applies(self, inst, category=None): return _is_lw(inst)
    def extract(self, inst, category=None):
        return _operands_with_slots(inst)


class _BufferLoadRule:
    """BufferLoad — `getDstParams() = {dst}`,
    `getSrcParams() = {vaddr, saddr, soffset}`. Reads pick up vaddr
    AND saddr (SRD) as registers; soffset filters out as an int.
    Slight coverage expansion over the prior srd-only model — see
    `Q9J_SCOPE_REASSESSMENT.md` §2 row `_BufferLoadRule`.

    For DirectToLds (DTL) loads (dst=None, mubuf->lds=True), additionally
    records an implicit read of m0 for the LDS destination address; the
    kernel writer constructs these as `dst=None if lds else vgpr(...)`
    at KWA:14608.

    Pre-bead-dzl this was two rules (`_DTLBufferLoadRule` +
    `_BufferLoadRule`) discriminated by a `_is_dtl_buffer_load` heuristic
    that checked `_inst_dst(inst) is None`. Now `MUBUFReadInstruction`
    carries a native `is_dtl` flag set in its C++ constructor from
    `mubuf->lds`, so the discriminator is just an attribute lookup and
    the m0 read is published via the rocisa-supplied `m0_resource()`
    singleton instead of reconstructing `mgpr(0)` per call.
    """
    def applies(self, inst, category=None):
        return _is_gr(inst)
    def extract(self, inst, category=None):
        reads, read_slots, writes, write_slots = _operands_with_slots(inst)
        # DTL-mode loads (dst=None, mubuf->lds=True) implicitly read m0.
        # `is_dtl` is set in the rocisa MUBUFReadInstruction constructor
        # from `mubuf->lds`. All instances reaching this point are real
        # rocisa BufferLoad subclasses (test fixtures construct real
        # BufferLoadB128 too).
        # The implicit m0 read gets a synthetic slot index past the last
        # explicit src slot — preserves slot uniqueness within the
        # instruction's read sequence without overlapping any positional
        # accessor index.
        if inst.is_dtl:
            from rocisa.instruction import m0_resource
            implicit_slot = (max(read_slots) + 1) if read_slots else 0
            reads = reads + (m0_resource(),)
            read_slots = read_slots + (implicit_slot,)
        return reads, read_slots, writes, write_slots


class _MFMARule:
    """MFMA — `getDstParams() = {acc}`, `getSrcParams() = {a, b, acc2}`.

    `acc2` defaults to `acc` for the in-place RMW case (the dominant
    shape), so the rule's reads include `acc` automatically — matches
    the prior `(a, b, acc)` synthesis. For the out-of-place case
    (`acc2 != acc`) the new accessor returns the actual hardware-read
    register, which the prior synthesis got wrong (it assumed acc was
    always the read). See `Q9J_SCOPE_REASSESSMENT.md` §2 row `_MFMARule`.

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
        return _operands_with_slots(inst)


class _NoDataflowRule:
    """SWaitCnt / SBarrier / SNop / SSetPrior — pure scheduling control;
    no register dataflow. SSetPrior takes only an `int prior`
    (`getParams() -> {prior}`); it has no register reads or writes.

    Real rocisa: `getDstParams() = getSrcParams() = {}` for all four;
    `_operands_via_accessors` would return `((), ())` correctly. The
    explicit empty return below short-circuits the accessor lookup so
    this rule remains well-defined for any class-name-only stand-ins
    that may surface in ad-hoc tests.
    """
    def applies(self, inst, category=None):
        return (
            _is_swait(inst)
            or _is_sbarrier(inst)
            or _is_snop(inst)
            or _is_ssetprio(inst)
        )
    def extract(self, inst, category=None):
        return (), (), (), ()


class _VSwapRule:
    """VSwapB32 (and any v_swap_* variant) — symmetric R+W on both operands.

    The symmetric semantic is **already encoded in C++**:
    `VSwapB32::getDstParams()` and `VSwapB32::getSrcParams()` BOTH return
    the two operands (`rocisa/include/instruction/common.hpp:5179-5194`),
    so the validator's `_operands_via_accessors` path produces
    `reads = writes = (op0, op1)` for free.

    Why symmetric R+W matters: `v_swap_b32 dst, src` exchanges the two
    registers — BOTH are read AND BOTH are written. Modelling this with
    the asymmetric `_GenericALURule` shape (`writes=(dst,)`,
    `reads=(src,)`) drops one of the four edge classes. Concretely:

        sw1: VSwap(v0, v1)   # ref position
        sw2: VSwap(v1, v2)

    Asymmetric: sw1 publishes write=v0/read=v1, sw2 publishes
    write=v1/read=v2 — they share v1 only as sw1.read + sw2.write, which
    is a WAR edge sw1->sw2. Reverse the pair and the edge becomes RAW
    sw2->sw1: the SUBJECT graph gains a NEW edge that REF lacks. Because
    `compare_graphs` (ScheduleCapture.py: `missing = ref - subj`) is
    one-directional, the reorder is invisible.

    Symmetric: BOTH orderings carry WAR + WAW + RAW edges on the shared
    register. Swapping the pair flips producer/consumer on each edge, so
    the edge KEY (producer.identity, consumer.identity, register, kind)
    differs between REF and SUBJ — `missing = ref - subj` finds at least
    one such key and `compare_graphs` flags the reorder.

    Order: MUST come before `_GenericALURule` so VSwap is claimed by this
    rule (its dst/src appear in BOTH lists) and not the asymmetric fallback.
    """
    def applies(self, inst, category=None):
        # Match rocisa VSwapB32 today and any future v_swap_* width
        # variant (e.g. VSwapB16) by class-name prefix. Keeps us decoupled
        # from a hardcoded opcode allowlist that needs maintenance per
        # ISA addition.
        return type(inst).__name__.startswith("VSwap")

    def extract(self, inst, category=None):
        return _operands_with_slots(inst)


# =============================================================================
# SCC implicit-operand rule
# =============================================================================
# SCC is a single-bit hardware status register, written implicitly by most
# scalar ALU and compare ops and read by SCSelect/SCMov/SCBranchSCC*. The
# rocisa C++ classes carry `reads_scc` / `writes_scc` flags set in their
# constructors (see bead rocm-libraries-dzl); the validator queries those
# flags directly rather than maintaining a parallel class-name table. The
# SCC RegisterContainer singleton is supplied by `rocisa.instruction
# .scc_resource()` so equality/hashing across producer-write and
# consumer-read containers stays stable.
#
# This singleton rides on existing register machinery:
#   * `Register.is_register` accepts it (has regType + regIdx).
#   * `Register.intersection` short-circuits on `reg_type` mismatch, so SCC
#     never accidentally aliases vgpr/sgpr.
#   * For two SCC instances the `_intersection` materializer looks up
#     `_NUMERIC_REG_FACTORIES["scc"]`, which we register below as a
#     constant-returning factory.
#   * `_byte_keys_for_resource` keys it as `("scc", 0)`, giving the
#     per-byte latest-writer resolver one slot to track.


class _SCCRule:
    """Per-opcode SCC read/write publisher.

    Placed BEFORE `_GenericALURule` in `_OPERAND_RULES`: claims every
    SCC-touching scalar opcode (per the rocisa-supplied `reads_scc` /
    `writes_scc` flags on the instruction class) and emits its register
    reads/writes plus the SCC singleton from `scc_resource()` where
    appropriate.

    Two extract shapes drive the register-side handling:

      * "dst" present (`inst.dst is not None`) — same convention as
        `_GenericALURule` (params[0]=write, params[1:]=reads). Used for
        SAdd/SSub/SCSelect/SAndSaveExec/etc.

      * "no dst" (`inst.dst is None`, or no `dst` attribute as for
        `BranchInstruction`) — all register params are reads, no
        register write. Used for SCmp* (which have `dst=nullptr` in
        rocisa, so `getParams()` returns just `[src0, src1]`) and
        SCBranchSCC* (label-only). This avoids the `_GenericALURule`
        quirk where SCmp's src0 would land at params[0] and be
        misclassified as a write.

    For SCC itself, the singleton from `scc_resource()` is appended to
    reads/writes per the `reads_scc` / `writes_scc` flags — the per-byte
    latest-writer resolver (Phase 2 of build_dataflow_graph) then
    naturally emits SCC RAW edges between producers and consumers, and an
    intervening SCC clobber becomes the new latest writer that breaks the
    producer's edge to the later consumer. Failure-shape wiring (turning
    the missing SCC edge into a typed Failure) lives in
    `diagnose_missing_edge`.
    """

    def applies(self, inst, category=None):
        # An instruction is SCC-relevant iff it sets either rocisa flag,
        # OR its `dst is None` shape needs the no-dst override (the
        # SCmp*/SBitcmp1B32 false-write quirk handled below). The flag
        # check is sufficient for current rocisa; the no-dst SCC opcodes
        # (SCmp*/SCBranchSCC*/SBitcmp1B32) all have one of the flags set.
        # Direct attribute access: every rocisa Instruction-derived class
        # carries reads_scc/writes_scc as bound C++ fields. The
        # LoopBodyCaptureBuilder.append `inst=` contract guarantees only
        # real rocisa Instruction instances reach here (including
        # bookkeeping tests, which use cheap stand-ins like SNop).
        return inst.reads_scc or inst.writes_scc

    def extract(self, inst, category=None):
        from rocisa.instruction import scc_resource
        try:
            params = list(inst.getParams())
        except Exception:
            params = []

        # Shape inference: instructions whose dst is null in C++ surface as
        # `inst.dst is None` (CommonInstruction.dst is bound def_rw). For
        # those the entire param list is reads. Branch instructions have
        # no `dst` attribute at all and getParams() returns the label
        # string only — also no register dst, no register srcs.
        # Slot indices are positional within `params` (preserved through
        # the Register.is_register filter so non-register positional args
        # leave holes rather than re-pack).
        has_dst = getattr(inst, "dst", None) is not None
        if not has_dst:
            read_pairs = [(i, p) for i, p in enumerate(params)
                          if Register.is_register(p)]
            write_pairs = []
        else:
            write_pairs = []
            if params and Register.is_register(params[0]):
                write_pairs.append((0, params[0]))
                read_pairs = [(i + 1, p)
                              for i, p in enumerate(params[1:])
                              if Register.is_register(p)]
            else:
                read_pairs = [(i, p) for i, p in enumerate(params)
                              if Register.is_register(p)]

        reg_reads = tuple(p for _, p in read_pairs)
        read_slots = tuple(i for i, _ in read_pairs)
        reg_writes = tuple(p for _, p in write_pairs)
        write_slots = tuple(i for i, _ in write_pairs)

        # Direct attribute access: only SCC-relevant rocisa instances
        # reach `extract()` (gated by `applies()` above). The flags are
        # bound on every rocisa instruction class in the same bead, so
        # they're guaranteed to exist here. Implicit SCC operands get
        # synthetic slot indices past the explicit positional sequence.
        if inst.reads_scc:
            scc_slot = (max(read_slots) + 1) if read_slots else 0
            reg_reads = reg_reads + (scc_resource(),)
            read_slots = read_slots + (scc_slot,)
        if inst.writes_scc:
            scc_slot = (max(write_slots) + 1) if write_slots else 0
            reg_writes = reg_writes + (scc_resource(),)
            write_slots = write_slots + (scc_slot,)
        return reg_reads, read_slots, reg_writes, write_slots


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
      reads  = tuple(p for p in params[1:] if Register.is_register(p))

    Non-register positional params (modifiers, ints, comments, VCC, labels)
    are filtered out by `Register.is_register`. Branch instructions whose only
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

    Scalar-ALU coverage: SCSelectB32, SAddU32, SAddCU32, SSubU32,
    SSubBU32, SCmpEQU32 and similar SOP1/SOP2 instructions land here.
    Their sgpr dst (params[0]) and sgpr srcs are tracked, so a reversed
    GRInc-style chain forms RAW edges that `compare_graphs` flips into
    `OrderInvertedFailure`. This is what allowed `verify_ascending_order`
    (CMSValidator.py) to be retired. Coverage proven by
    `test_dataflow_graph_comparison.py::TestGRIncReorderDetection` and
    `::TestVgprChainReorderDetection`.

    NOT covered here (deliberate scope cut):
      - VCC dataflow (carry-out / carry-in / cmp-out) is intentionally
        not tracked by the validator (permanent design choice — see
        `CMSValidator_LIMITATIONS.md` §"VCC dataflow tracking is
        intentionally not provided"). VCC sentinels have no
        regType/regIdx so this rule's `Register.is_register` filter drops them
        from reads/writes; VCC RAW edges are not formed.
      - VSwap symmetric R+W (both operands read AND written) — handled by
        `_VSwapRule`, which precedes this rule.

    Cleaned up by `_SCCRule`, which also precedes this rule:
      - SCC implicit read/write for SOPC/SOP1/SOP2/SOPK/branch ops. The
        SCC singleton from `rocisa.instruction.scc_resource()` is
        published in reads/writes per the rocisa-supplied
        `inst.reads_scc` / `inst.writes_scc` flags.
      - SCmp* false-write quirk: SCmp* has no sgpr dst (`dst=nullptr` in
        rocisa) but `getParams()` skips the absent dst and returns just
        `[src0, src1]`, so the generic rule would misclassify `src0` as a
        write at params[0]. `_SCCRule` claims any SCC-flagged opcode
        BEFORE the generic rule and treats every register-shaped param
        as a read when `inst.dst is None`.
      - SBitcmp1B32: same `dst=nullptr` shape; flagged `writes_scc=true`
        in rocisa so `_SCCRule` claims it and `params[0]=ssrc0` is
        correctly treated as a read.

    DANGER ZONE for new rocisa classes (audit checklist for future PRs):
      Any new CommonInstruction subclass whose constructor passes
      `nullptr` as the `dst` argument (look for the second positional arg
      in the `CommonInstruction(InstType::..., <dst>, ...)` super-call)
      AND does NOT touch SCC will hit this rule with the SAME footgun:
      `getParams()` skips the absent dst and returns `[src0, ...]`, so
      `params[0]` is a SOURCE that this rule will misclassify as a
      write. Mitigation: any SCC-touching no-dst class is already
      handled — set `reads_scc`/`writes_scc` in its constructor and
      `_SCCRule` claims it first. For non-SCC no-dst classes, add an
      explicit rule before `_GenericALURule`.
    """
    def applies(self, inst, category=None):
        # Only real rocisa Instruction-derived objects expose the
        # getDstParams/getSrcParams pair. Ad-hoc test stand-ins that
        # don't expose the pair fall through to the empty (reads, writes)
        # default and rely on their own bespoke rules (e.g. _FakePackRule
        # injected by tests via using_pack_rule()).
        return hasattr(inst, "getSrcParams") and hasattr(inst, "getDstParams")

    def extract(self, inst, category=None):
        return _operands_with_slots(inst)


# Order matters: more specific rules first; _GenericALURule is the
# catch-all and MUST come last so earlier rules claim their classes.
# `_SCCRule` sits BEFORE `_GenericALURule` so it claims SCC-touching
# scalar opcodes and attaches the SCC sentinel to their reads/writes — also
# fixes the SCmp* false-write quirk noted in `_GenericALURule`'s docstring.
_OPERAND_RULES = (
    _DSLoadRule(),
    _DSStoreRule(),
    _BufferLoadRule(),  # handles DTL via inst.is_dtl; no separate DTL rule
    _MFMARule(),
    _NoDataflowRule(),
    _VSwapRule(),       # symmetric R+W on the two operands
    _SCCRule(),         # claims SCC-touching scalar opcodes
    _GenericALURule(),
)


def _populate_wrapper(wrapper, category=None) -> None:
    """Run the operand rules over wrapper._rocisa_inst, accumulating
    reads and writes into wrapper.reads / wrapper.writes (and, in
    parallel, wrapper.read_slots / wrapper.write_slots).

    `category` (the TaggedInstruction.category) lets a rule discriminate
    on emission-time bucket — e.g. MFMARule excludes Pack-categorized
    MFMAInstructions (TF32 emulation pattern) so they're not treated as
    main-loop MFMA producers.

    Only the FIRST matching rule contributes; this prevents e.g. an MFMA
    from being processed by multiple rules.

    Rules return a 4-tuple `(reads, read_slots, writes, write_slots)`.
    Every production rule and test fixture conforms to this shape; the
    dispatch forbids fallback paths.

    Idempotent: rules are pure functions of (inst, category).
    """
    inst = wrapper._rocisa_inst
    # Every wrapped instruction reaching this point is a real rocisa
    # Instruction subclass (LoopBodyCaptureBuilder.append's `inst=`
    # contract; capture-pipeline tests use real rocisa stand-ins like
    # SNop for bookkeeping coverage). The operand rules (notably
    # _SCCRule) do direct attribute access on rocisa-bound flags like
    # reads_scc / writes_scc; those flags are guaranteed bound on every
    # rocisa Instruction class, so no shape gate is needed.
    for rule in _OPERAND_RULES:
        if rule.applies(inst, category):
            reads, read_slots, writes, write_slots = rule.extract(inst, category)
            wrapper.reads = tuple(reads)
            wrapper.writes = tuple(writes)
            wrapper.read_slots = tuple(read_slots)
            wrapper.write_slots = tuple(write_slots)
            return
    wrapper.reads = ()
    wrapper.writes = ()
    wrapper.read_slots = ()
    wrapper.write_slots = ()


# Lazy-populated factory map: regType character -> RegisterContainer factory
# function (idx, count) -> RegisterContainer. Populated on first invocation so
# this module stays free of a top-level rocisa import.
# vgpr/sgpr/accvgpr factories produce the canonical regName=None numeric
# form that compares equal under value-based __eq__/__hash__; that property
# is what makes the set-based dedup in DataflowGraph.edge_keys() work after
# we start emitting intersection-precise registers on edges.
_NUMERIC_REG_FACTORIES = None


def _ensure_numeric_factories():
    global _NUMERIC_REG_FACTORIES
    if _NUMERIC_REG_FACTORIES is not None:
        return
    from rocisa.container import vgpr, sgpr, mgpr, accvgpr
    from rocisa.instruction import scc_resource
    _NUMERIC_REG_FACTORIES = {
        "v": vgpr,
        "s": sgpr,
        "acc": accvgpr,
        # mgpr's count is fixed at 1 in practice (m0). Wrap for uniform call shape.
        "m": lambda idx, count=1: mgpr(idx),
        # SCC: single-bit hardware status register, modeled by the rocisa-
        # supplied singleton (regType="scc"). The factory always returns
        # that singleton so equality/hashing across producer-write and
        # consumer-read containers stays stable.
        "scc": lambda idx, count=1: scc_resource(),
    }


def _materialize_register(reg):
    """Build a rocisa ``RegisterContainer`` from a ``Register``.

    Numeric ``Register``s use the per-regType factory (vgpr/sgpr/accvgpr/...)
    so the resulting container has the canonical ``regName=None`` form that
    set-based edge dedup relies on. Symbolic ``Register``s rebuild a fresh
    ``RegisterContainer(regType, RegName(name, [base]), -1, count)``.

    Returns ``None`` for unknown ``reg_type`` — callers (the per-byte edge
    resolver) skip the edge rather than emit a bogus container.
    """
    if reg.is_symbolic():
        from rocisa.container import RegisterContainer, RegName
        return RegisterContainer(
            reg.reg_type,
            RegName(reg.name, [reg.base]),
            -1,
            reg.count,
        )
    _ensure_numeric_factories()
    factory = _NUMERIC_REG_FACTORIES.get(reg.reg_type)
    if factory is None:
        return None
    return factory(reg.base, reg.count)


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

    Register-typed inputs are dispatched through the ``Register`` abstraction:
    inputs are wrapped via :meth:`Register.from_rocisa`, intersected via
    :meth:`Register.intersection`, and the result is materialized back into
    a fresh ``RegisterContainer`` for downstream consumers (edge formatters,
    set-based edge dedup).
    """
    if isinstance(a, MemoryRegion) and isinstance(b, MemoryRegion):
        return _memory_intersection(a, b)
    if isinstance(a, MemoryRegion) or isinstance(b, MemoryRegion):
        return None
    # Both are RegisterContainers (or duck-types). Wrap and intersect.
    if a is None or b is None:
        return None
    try:
        ra = Register.from_rocisa(a)
        rb = Register.from_rocisa(b)
    except ValueError:
        return None
    overlap = ra.intersection(rb)
    if overlap is None:
        return None
    return _materialize_register(overlap)


# _min_issue_quad_cycles_for, _make_node, and _BODY_BUILD_ORDER moved to
# CMSValidator.py in br4.6.


# build_dataflow_graph, _collect_barrier_edges, and _collect_pattern moved to
# CMSValidator.py in br4.7.


# compare_graphs and diagnose_missing_edge moved to CMSValidator.py in br4.8.


# validate_edge_wait_coverage, _classify_edge_coverage, and
# validate_middle_pack_pair_interleaving moved to CMSValidator.py in br4.9.
# After this move ScheduleCapture has zero reverse-imports back to
# CMSValidator: the dependency edge is one-way (CMSValidator → ScheduleCapture).


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


def collect_regset_stream(writer):
    """Harvest the symbolic-name -> numeric-base map from a KernelWriter's
    RegSet directives.

    The writer emits `RegSet` directives into separate scratch modules
    (`writer.moduleVgprMacro*` for vgpr macros, `writer.module` for the
    kernel body) that BYPASS the schedule streams (`LRCodeA`, `PackCodeA`,
    `globalReadA`, ...) which feed `customMainLoopSchedule`. So the macro
    walked by `expand_cms_macro` never sees a `RegSet` to consume; this
    helper provides a parallel collection mechanism.

    For vgprs we replay the `value`/`ref`/`offset` chain in three passes
    (mirroring the rocisa C++ `RegSet::setIdx` semantics — value-form is
    `value + offset`, ref-form is `name_to_idx[ref] + offset`). Three
    passes are enough because RegSet refs only chain one or two levels
    deep in production. For sgprs we consume the writer's symbolic
    `sgprs` pool directly (the rocisa per-thread vgpr-name singleton is
    cleared post-codegen, but `writer.sgprs` survives — see
    `INTRA_GRAPH_7A_REGSET_INVESTIGATION.md` Q4 for the empirical audit).

    Returns a flat `name -> idx` dict where `name` is the bare name
    (without the `vgpr` / `sgpr` prefix) and `idx` is the numeric base
    register index. The same map is used to seed each
    `LoopBodyCapture.name_to_idx` (every body of the four-part capture
    sees the same writer-level symbols by construction).
    """
    name_to_idx: Dict[str, int] = {}

    # 1. Walk writer's modules and collect every RegSet directive in
    #    emission order.
    regsets = []

    def _walk(node):
        cls = type(node).__name__
        if cls == "RegSet":
            regsets.append(node)
            return
        for getter in ("flatitems", "items"):
            if hasattr(node, getter):
                try:
                    kids = getattr(node, getter)()
                except Exception:
                    continue
                for k in kids:
                    _walk(k)
                return
        if hasattr(node, "itemList"):
            for k in node.itemList:
                _walk(k)

    cand_attrs = [a for a in dir(writer)
                  if (a.startswith("module") or a.startswith("kernelBody"))
                  and not a.startswith("__")]
    for nm in cand_attrs:
        try:
            mod = getattr(writer, nm)
        except Exception:
            continue
        if mod is None or callable(mod):
            continue
        _walk(mod)

    # 2. Seed anchors that ref-form RegSets dereference.
    #    `vgprMXSBase` is bound numerically by KernelWriterAssembly.py:865
    #    (`module.add(RegSet("v", "vgprMXSBase", 0))`); we'll catch it in the
    #    pass below. The MXS valu-base is set imperatively on the writer
    #    (see `writer.states.mxsa.startVgprValu`), not via RegSet, so the
    #    ref-chain replay needs it pre-seeded.
    mxsa = getattr(getattr(writer, "states", None), "mxsa", None)
    if mxsa is not None:
        base = getattr(mxsa, "startVgprValu", None)
        if base is not None:
            name_to_idx["Base"] = base

    # 3. Three passes over the collected RegSets — value-form binds
    #    immediately; ref-form binds once the referenced name is known.
    for _ in range(3):
        for rs in regsets:
            name = getattr(rs, "name", None)
            if not name:
                continue
            if name.startswith("vgpr"):
                bare = name[4:]
            elif name.startswith("sgpr"):
                # sgprs are seeded from writer.sgprs (step 4); skip the
                # RegSet path because some sgpr RegSets use the bare-name
                # ref form ("sgprtdmBGroup0" -> "sgprtdmAGroup0") which
                # would shadow the writer.sgprs binding.
                continue
            else:
                bare = name
            value = getattr(rs, "value", None)
            ref = getattr(rs, "ref", None)
            offset = getattr(rs, "offset", 0) or 0
            if value is not None and bare not in name_to_idx:
                name_to_idx[bare] = int(value) + offset
            elif ref:
                ref_bare = ref[4:] if ref.startswith(("vgpr", "sgpr")) else ref
                base = name_to_idx.get(ref_bare)
                if base is not None and bare not in name_to_idx:
                    name_to_idx[bare] = base + offset

    # 4. Sgpr pool — `writer.sgprs` is a name -> idx mapping that survives
    #    `_getKernelSource`. Names here are bare (no "sgpr" prefix), which
    #    matches the convention used by `_byte_keys_for_resource`'s
    #    symbolic lookup.
    sgprs = getattr(writer, "sgprs", None)
    if sgprs is not None:
        try:
            for nm, idx in dict(sgprs).items():
                if nm not in name_to_idx:
                    name_to_idx[nm] = int(idx)
        except Exception:
            pass

    return name_to_idx


def expand_cms_macro(macro, id_value, useGR, usePLR, useGRInc, useLoop,
                     tag_by_origin_id, sync_class=None, snop_class=None,
                     mfma_classes=(), name_to_idx=None):
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

    body = builder.finalize()
    if name_to_idx:
        # Body-local copy: callers may mutate per-body without poisoning the
        # shared writer-level map. Empirically RegSet directives are emitted
        # before any body's instruction stream begins (see
        # INTRA_GRAPH_7A_REGSET_INVESTIGATION.md Q1), so seeding the body's
        # map up-front is equivalent to interleaving RegSet consumption with
        # instruction emission for current production kernels.
        body.name_to_idx = dict(name_to_idx)
    return body


# =============================================================================
# Body-emission control for FourPartCapture builders (PGR/PLR)
# =============================================================================
# Historically (pre rocm-libraries-dj1g) a pair of Python-side predicates
# re-derived from kernel config whether `KernelWriter.kernelBody` would
# emit NGL/NLL bodies, and a separate consistency-check helper cross-
# checked the captured-body presence against those predicates. Both were
# dropped: the capture pipeline is now the single source of truth for the
# default-side capture, and the CMS-side capture mirrors the default-side
# capture's body shape by construction.
#
# `build_cms_four_part_capture` consumes the default-side capture and
# expands the CMS macro four ways, populating `n_gl` / `n_ll` only when
# the corresponding default-side dict is non-empty. There is no
# Python-side prediction of what the kernel will emit; the capture
# pipeline observes what was actually emitted, and the CMS expander
# observes that observation.


@dataclass
class CmsCaptureInputs:
    """Inputs needed to expand a CMS MAINLOOP macro into a FourPartCapture.

    `customMainLoopSchedule` runs before the default-side capture exists
    (see KernelWriter.kernelBody flow: customMainLoopSchedule produces
    the macro and this stash; later, noLoadLoop populates
    ctx.default_n_gl / ctx.default_n_ll; later still, kernelBody assembles
    the default-side FourPartCapture). Expansion is therefore deferred
    until the default-side capture is available, at which point the
    deferred call to `build_cms_four_part_capture` mirrors the default
    side's body shape.

    `regset_stream` is the symbolic-name -> numeric-base lookup harvested
    from the writer's `moduleVgprMacro*` modules and `module` (kernel
    body) at the same scope as the schedule streams (see
    `collect_regset_stream`). It feeds `LoopBodyCapture.name_to_idx` so
    the in-pipeline graph builder can resolve symbolic operand references
    to numeric byte-keys when forming dataflow edges.
    """
    macro: object
    num_codepaths: int
    tag_by_origin_id: dict
    sync_class: type
    snop_class: type
    mfma_classes: tuple
    num_mfma_per_subiter: int = 0
    regset_stream: dict = field(default_factory=dict)


def build_cms_four_part_capture(macro, num_codepaths, tag_by_origin_id,
                                  sync_class, snop_class, mfma_classes,
                                  default_capture: 'FourPartCapture',
                                  num_mfma_per_subiter: int = 0,
                                  regset_stream=None):
    """Expand a CMS MAINLOOP macro four ways and assemble a FourPartCapture.

    main_loop[cp] expands with all flags=1 and \\ID=cp for each cp.
    main_loop_prev[cp] is a verbatim clone of main_loop[cp].
    n_gl[0] expands with useGR=0, usePLR=1, useGRInc=1, useLoop=0, \\ID=0
        — only when ``default_capture.n_gl`` is non-empty.
    n_ll[0] expands with useGR=0, usePLR=0, useGRInc=0, useLoop=0, \\ID=0
        — only when ``default_capture.n_ll`` is non-empty.

    These flag assignments mirror the CMS dispatch sites at:
      - simdSpecDispatch (KernelWriterAssembly.py, simdSpecDispatch) for main_loop
      - _emitNoLoadLoopBodyCMSMacro (KernelWriter.py) for n_gl/n_ll

    n_gl/n_ll are keyed only at {0: body} because the CMS tail-loop emission
    hard-codes \\ID=0 (see _emitNoLoadLoopBodyCMSMacro docstring for the
    correctness rationale).

    ``default_capture`` is the already-built default-side FourPartCapture;
    its `.n_gl` / `.n_ll` dict-presence is the single source of truth for
    whether the corresponding CMS-side body is expanded. By construction
    the CMS-side capture's body shape matches the default-side's — there
    is no separate Python-side predicate, no consistency check.
    """
    default_has_n_gl = bool(default_capture.n_gl)
    default_has_n_ll = bool(default_capture.n_ll)

    main_loop = {}
    main_loop_prev = {}
    for cp in range(num_codepaths):
        body = expand_cms_macro(
            macro, id_value=cp,
            useGR=1, usePLR=1, useGRInc=1, useLoop=1,
            tag_by_origin_id=tag_by_origin_id,
            sync_class=sync_class, snop_class=snop_class,
            mfma_classes=mfma_classes,
            name_to_idx=regset_stream,
        )
        main_loop[cp] = body
        main_loop_prev[cp] = clone_loop_body(body)

    if default_has_n_gl:
        n_gl_body = expand_cms_macro(
            macro, id_value=0,
            useGR=0, usePLR=1, useGRInc=1, useLoop=0,
            tag_by_origin_id=tag_by_origin_id,
            sync_class=sync_class, snop_class=snop_class,
            mfma_classes=mfma_classes,
            name_to_idx=regset_stream,
        )
        n_gl_dict = {0: n_gl_body}
    else:
        n_gl_dict = {}

    if default_has_n_ll:
        n_ll_body = expand_cms_macro(
            macro, id_value=0,
            useGR=0, usePLR=0, useGRInc=0, useLoop=0,
            tag_by_origin_id=tag_by_origin_id,
            sync_class=sync_class, snop_class=snop_class,
            mfma_classes=mfma_classes,
            name_to_idx=regset_stream,
        )
        n_ll_dict = {0: n_ll_body}
    else:
        n_ll_dict = {}

    num_mfma = sum(1 for ti in main_loop[0].instructions if ti.category == "MFMA")

    return FourPartCapture(
        main_loop=main_loop,
        main_loop_prev=main_loop_prev,
        n_gl=n_gl_dict,
        n_ll=n_ll_dict,
        num_mfma=num_mfma,
        num_codepaths=num_codepaths,
        source="cms",
        num_mfma_per_subiter=num_mfma_per_subiter,
    )
