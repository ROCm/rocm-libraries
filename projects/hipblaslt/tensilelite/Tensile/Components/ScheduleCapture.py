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

    Also raised by `assert_capture_body_consistency` when a captured
    `FourPartCapture`'s body presence (key 0 in `n_gl` / `n_ll`) does not
    match the kernel-config-derived predicates `kernel_emits_n_gl` /
    `kernel_emits_n_ll`. That signals either a production-gate change in
    `KernelWriter.kernelBody` (KernelWriter.py:5118 and :5141-5142) that
    was not mirrored in the predicates, or a capture-pipeline bug that
    populated/omitted a body when the predicates said otherwise.
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
    # to derive each MFMA's subiter index (vmfma_index // num_mfma_per_subiter).
    # Both default-side and CMS-side construction sites should pass
    # writer.states.numMfmaPerIter (upstream Tensile naming retains "Iter"
    # but it refers to the inner unroll subiteration here).
    # Defaults to 0 ("don't split MFMAs by subiter"); the resolver then keeps
    # all MFMAs at subiter 0, which loses cross-subiter PLR dataflow edges.
    # Test fixtures may safely leave it unset.
    num_mfma_per_subiter: int = 0
    # Per-architecture timing profile. None = default CDNA 4. Production
    # callers pass the profile resolved via `_resolve_arch_profile_for_isa`
    # from `kernel["ISA"]`; test fixtures may leave it unset to keep the
    # historical code path bit-identical.
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


def make_position(body_label: str, slot: SlotKey) -> SchedulePosition:
    """Construct a SchedulePosition from a TaggedInstruction.slot SlotKey.

    The body_label maps to loop_index via BODY_LABEL_TO_LOOP_INDEX so cross-body
    ordering is well-defined.
    """
    return SchedulePosition(
        loop_index=BODY_LABEL_TO_LOOP_INDEX[body_label],
        vmfma_index=slot.mfma_index,
        sub_index=slot.sequence,
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
    because SchedulePosition's tuple — (loop_index, vmfma_index, sub_index) —
    drops `subiter`; sub_index must therefore continue across subiters
    so positions encode stream-emission order without collisions.

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


# Stream-position ordering / identity helpers (_class_tag,
# _class_tag_from_category, _split_category_iter, _node_subiter,
# _TRAILING_DIGITS_RE) moved to CMSValidator.py in br4.6.


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


def _resolve_producers(read_resource: Any, consumer: GraphNode, latest_writer: Dict[Any, Tuple[GraphNode, Any]]):
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

    VCC instances have no regType/regIdx (only `__class__` distinguishes
    them), so `_is_register` rejects them. The dedicated `_VCCRule`
    recognizes them via this helper and substitutes the synthetic
    `_VCC_RESOURCE` singleton so per-byte tracking can flow through the
    standard resolver.
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
    """SWaitCnt / SBarrier / SNop / SSetPrior — pure scheduling control;
    no register dataflow. SSetPrior takes only an `int prior`
    (`getParams() -> {prior}`); it has no register reads or writes."""
    def applies(self, inst, category=None):
        return (
            _is_swait(inst)
            or _is_sbarrier(inst)
            or _is_snop(inst)
            or _is_ssetprio(inst)
        )
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

    REMOVAL TARGET — bead `rocm-libraries-uraq`. This rule and its
    supporting helpers (`_is_vcc`, `_vcc_resource`, `_VCC_RESOURCE`,
    `_VCC_DST1_CARRY_OUT_CLASSES`) are scheduled for permanent
    deletion. **No replacement is planned.** VCC dataflow tracking
    is being removed from the validator's scope as a permanent
    design choice. See `CMSValidator_LIMITATIONS.md` §"VCC dataflow
    tracking is intentionally not provided" for the resulting
    limitation.

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

    Coverage:
      - VAddCOU32 / VAddCCOU32 — 64-bit add carry chain (globalReadIncrement
        non-buffer path, KernelWriterAssembly.py:9120-9146)
      - VSubCoU32 — 64-bit sub borrow chain
      - VCmpEQU32 and other VCmp* — VCC-as-comparison-output
      - SOrSaveExecBX with dst=VCC() — VCC-as-saved-EXEC

    NOT addressed here:
      - sgpr-pair-as-VCC: real CMS sometimes uses an sgpr pair as the
        carry destination (e.g. `dst1=sgpr("VCC", 2)`) instead of the
        VCC sentinel. Those flow through `_GenericALURule` already; this
        rule only addresses the VCC-sentinel form.
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
# SCC sentinel resource
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
# Each entry traces back to the gfxIsa.inc IF_ImplicitReadSCC /
# IF_ImplicitWriteSCC flags.
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
    # graph.
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
    """Per-opcode SCC read/write publisher.

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
    later consumer. Failure-shape wiring (turning the missing SCC edge
    into a typed Failure) lives in `diagnose_missing_edge`.
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

    Scalar-ALU coverage: SCSelectB32, SAddU32, SAddCU32, SSubU32,
    SSubBU32, SCmpEQU32 and similar SOP1/SOP2 instructions land here.
    Their sgpr dst (params[0]) and sgpr srcs are tracked, so a reversed
    GRInc-style chain forms RAW edges that `compare_graphs` flips into
    `OrderInvertedFailure`. This is what allowed `verify_ascending_order`
    (CMSValidator.py) to be retired. Coverage proven by
    `test_dataflow_graph_comparison.py::TestGRIncReorderDetection` and
    `::TestVgprChainReorderDetection`.

    NOT covered here (deliberate scope cut):
      - VCC carry-out / carry-in (regType=None) — handled by `_VCCRule`,
        which precedes this rule in `_OPERAND_RULES`.
      - VSwap symmetric R+W (both operands read AND written) — handled by
        `_VSwapRule`, which precedes this rule.

    Cleaned up by `_SCCRule`, which also precedes this rule:
      - SCC implicit read/write for SOPC/SOP1/SOP2/SOPK/branch ops. The
        SCC sentinel resource is published in reads/writes per opcode.
      - SCmp* false-write quirk: SCmp* has no sgpr dst (`dst=nullptr` in
        rocisa) but `getParams()` skips the absent dst and returns just
        `[src0, src1]`, so the generic rule would misclassify `src0` as a
        write at params[0]. `_SCCRule` claims SCmp* opcodes BEFORE the
        generic rule and treats every register-shaped param as a read.
      - SBitcmp1B32: same `dst=nullptr` shape; claimed by `_SCCRule` with
        shape="no_dst" so `params[0]=ssrc0` is correctly treated as a
        read. Currently dormant in CMS captures (emitted in TDM setup +
        openSumAtLeastUnroll, both outside the captured body) but claimed
        for defense-in-depth.

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
      precedents.
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
# `_SCCRule` sits BEFORE `_GenericALURule` so it claims SCC-touching
# scalar opcodes and attaches the SCC sentinel to their reads/writes — also
# fixes the SCmp* false-write quirk noted in `_GenericALURule`'s docstring.
_OPERAND_RULES = (
    _DSLoadRule(),
    _DSStoreRule(),
    _DTLBufferLoadRule(),
    _BufferLoadRule(),
    _MFMARule(),
    _NoDataflowRule(),
    _VSwapRule(),       # symmetric R+W on the two operands
    _VCCRule(),         # claims VCC-bearing classes before generic
    _SCCRule(),         # claims SCC-touching scalar opcodes
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
        # VCC factory: synthetic RegisterContainer for the VCC 64-bit
        # register pair. The _VCCRule emits resources of regType="vcc" so
        # the byte-key resolver and _reg_intersection can both handle them.
        "vcc": lambda idx, count=1: RegisterContainer("vcc", RegName("vcc", []), idx, count),
        # SCC sentinel: single-bit hardware status register, modeled as a
        # singleton RegisterContainer with regType="scc". The factory
        # always returns the singleton so equality/hashing across
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


# =============================================================================
# Body-emission predicates and consistency helper (PGR/PLR Phase C)
# =============================================================================
# Mirror the production NGLL/NLL emission gates in `KernelWriter.kernelBody`:
#   - NGLL is emitted iff `for remainPgr in range(PGR-1, 0, -1)` runs at
#     least once, i.e. PGR >= 2 (KernelWriter.py:5118).
#   - NLL is emitted iff `if kernel["PrefetchGlobalRead"] and not
#     kernel["SuppressNoLoadLoop"]:` is true, i.e. PGR >= 1 and the
#     suppress-flag is off (KernelWriter.py:5141-5142).
#
# Validator-side and CMS-side constructors of FourPartCapture both consult
# these predicates to decide whether to populate `n_gl[0]` / `n_ll[0]` or
# leave the dict empty. Single source of truth — if the production gates
# ever shift, update these helpers and `assert_capture_body_consistency`
# will surface any caller that still synthesizes a phantom body.

def kernel_emits_n_gl(kernel) -> bool:
    """True iff `KernelWriter.kernelBody` will emit at least one NGLL.

    Mirrors the gate at `KernelWriter.py:5118`:
        for remainPgr in range(kernel["PrefetchGlobalRead"]-1, 0, -1):
    The loop body runs iff PGR >= 2.
    """
    return kernel["PrefetchGlobalRead"] >= 2


def kernel_emits_n_ll(kernel) -> bool:
    """True iff `KernelWriter.kernelBody` will emit a (non-Opt) NLL.

    Mirrors the gate at `KernelWriter.py:5141-5142`:
        if kernel["PrefetchGlobalRead"]:
          if not kernel["SuppressNoLoadLoop"]:
    NLL emission requires PGR >= 1 and SuppressNoLoadLoop=False.

    Note: this predicate does NOT model OptNLL (see ot2.C design 5.3).
    `_noLoadLoopBodyDefault` is guarded by `not isOptNLL`
    (KernelWriter.py:3703); OptNLL kernels currently leave default_n_ll
    None even when the predicate says yes. The consistency check will
    fire loudly on the first such kernel, at which point this predicate
    needs an `OptNoLoadLoop` clause.
    """
    return bool(kernel["PrefetchGlobalRead"]) and not kernel["SuppressNoLoadLoop"]


def assert_capture_body_consistency(four_part_capture, kernel) -> None:
    """Cross-check a FourPartCapture's body presence against the
    kernel-config-derived predicates `kernel_emits_n_gl` / `kernel_emits_n_ll`.

    Raises `CaptureConsistencyError` on mismatch. Catches:
      - A future production-side gate change that quietly stops emitting
        a body the predicate still says should exist (silent under-validation).
      - A capture-pipeline bug that synthesizes a phantom body the
        production side cannot match (silent over-validation).
      - Either side accidentally diverging from the agreed predicate so
        compare_graphs would walk asymmetric body label sets.

    Called in `KernelWriter.kernelBody` immediately after constructing
    `ctx.default` and `ctx.cms`, before `build_dataflow_graph`.
    """
    if four_part_capture is None:
        return
    expected_n_gl = kernel_emits_n_gl(kernel)
    expected_n_ll = kernel_emits_n_ll(kernel)
    actual_n_gl = 0 in four_part_capture.n_gl
    actual_n_ll = 0 in four_part_capture.n_ll
    if actual_n_gl != expected_n_gl or actual_n_ll != expected_n_ll:
        raise CaptureConsistencyError(
            f"FourPartCapture body presence does not match kernel-config "
            f"predicates (source={four_part_capture.source!r}). "
            f"PGR={kernel['PrefetchGlobalRead']!r}, "
            f"SuppressNoLoadLoop={kernel.get('SuppressNoLoadLoop')!r}. "
            f"Expected n_gl present={expected_n_gl}, got {actual_n_gl}. "
            f"Expected n_ll present={expected_n_ll}, got {actual_n_ll}. "
            f"Either the production gates at KernelWriter.py:5118/:5141-5142 "
            f"changed without updating kernel_emits_n_gl/n_ll, or a "
            f"FourPartCapture constructor synthesized a body the predicate "
            f"says should not exist."
        )


def build_cms_four_part_capture(macro, num_codepaths, tag_by_origin_id,
                                  sync_class, snop_class, mfma_classes,
                                  num_mfma_per_subiter=0,
                                  emit_n_gl=True, emit_n_ll=True):
    """Expand a CMS MAINLOOP macro four ways and assemble a FourPartCapture.

    main_loop[cp] expands with all flags=1 and \\ID=cp for each cp.
    main_loop_prev[cp] is a verbatim clone of main_loop[cp].
    n_gl[0] expands with useGR=0, usePLR=1, useGRInc=1, useLoop=0, \\ID=0
        — only when ``emit_n_gl`` is true.
    n_ll[0] expands with useGR=0, usePLR=0, useGRInc=0, useLoop=0, \\ID=0
        — only when ``emit_n_ll`` is true.

    These flag assignments mirror the CMS dispatch sites at:
      - simdSpecDispatch (KernelWriterAssembly.py, simdSpecDispatch) for main_loop
      - _emitNoLoadLoopBodyCMSMacro (KernelWriter.py) for n_gl/n_ll

    n_gl/n_ll are keyed only at {0: body} because the CMS tail-loop emission
    hard-codes \\ID=0 (see _emitNoLoadLoopBodyCMSMacro docstring for the
    correctness rationale).

    ``emit_n_gl`` / ``emit_n_ll`` (default True for backwards compat with
    direct test fixtures) gate whether each macro is expanded at all; when
    false the corresponding dict is left empty so the CMS-side capture
    matches the default-side capture's structural absence under
    PGR/SuppressNoLoadLoop combinations that legitimately suppress those
    bodies. Production callers in `CustomSchedule.py` derive the flags
    from `kernel_emits_n_gl(kernel)` / `kernel_emits_n_ll(kernel)`.
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

    if emit_n_gl:
        n_gl_body = expand_cms_macro(
            macro, id_value=0,
            useGR=0, usePLR=1, useGRInc=1, useLoop=0,
            tag_by_origin_id=tag_by_origin_id,
            sync_class=sync_class, snop_class=snop_class,
            mfma_classes=mfma_classes,
        )
        n_gl_dict = {0: n_gl_body}
    else:
        n_gl_dict = {}

    if emit_n_ll:
        n_ll_body = expand_cms_macro(
            macro, id_value=0,
            useGR=0, usePLR=0, useGRInc=0, useLoop=0,
            tag_by_origin_id=tag_by_origin_id,
            sync_class=sync_class, snop_class=snop_class,
            mfma_classes=mfma_classes,
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
