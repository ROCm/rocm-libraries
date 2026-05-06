################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Tuple, Union

from rocisa.instruction import (
    SWaitCnt, SBarrier,
    MFMAInstruction, SMovB32, SAddU32,
    VPermB32, VOrB32, VLShiftLeftOrB32, VSwapB32,
    VCvtPkF32toBF16, PVCvtBF16toFP32, VCvtBF16toFP32, VSubF32, VDot2CF32BF16,
    BufferLoadB128, BufferLoadB64, BufferLoadB32,
)
from Tensile.Common.Utilities import printWarning
# Eager imports from ScheduleCapture are safe: ScheduleCapture only imports
# from this module under TYPE_CHECKING (no eager reverse-import), so loading
# CMSValidator triggers a complete ScheduleCapture load, after which all of
# ScheduleCapture's public symbols are available. The reverse direction is the
# same — ScheduleCapture loads independently, then CMSValidator can be loaded
# on demand. (br4.6 added the helper-related symbols below.)
from Tensile.Components.ScheduleCapture import (
    SchedulePosition,
    BODY_LABEL_ML_PREV, BODY_LABEL_ML, BODY_LABEL_NGL, BODY_LABEL_NLL,
    BODY_LABEL_TO_LOOP_INDEX,
    CaptureConsistencyError,
    CaptureEmptyBodyError,
    CaptureUnknownInstructionError,
    UnexplainedMissingEdgeError,
    MemoryRegion,
    make_position,
    _canonical_render,
    _is_lr, _is_lw, _is_gr, _is_mfma, _is_swait, _is_sbarrier, _is_snop,
    _is_ssetprio, _is_cvt_pack,
    _byte_keys_for_resource, _resolve_producers,
)

if TYPE_CHECKING:
    # Annotations only — kept as strings at runtime by `from __future__ import
    # annotations`. Importing these eagerly would form a hard cycle
    # (ScheduleCapture imports from this module at runtime).
    from Tensile.Components.ScheduleCapture import (
        LoopBodyCapture, TaggedInstruction,
    )


# Sentinel values for "infinitely far" positions. Values chosen to be well beyond
# any realistic schedule size (num_vmfma is typically ~48-200).
POSITION_INF = SchedulePosition(loop_index=9_999, vmfma_index=9_999, sub_index=9_999)
POSITION_NEG_INF = SchedulePosition(loop_index=-9_999, vmfma_index=-9_999, sub_index=-9_999)

class ValidationConcern(Enum):
    """Abstract correctness properties the validator must check.

    Each concern is a unit of coverage tracking — if isValid returns True,
    every required concern was checked by at least one rule.
    """
    # Universal (any ISA)
    INSTRUCTION_ORDERING = auto()
    SCHEDULE_COMPLETENESS = auto()
    # Data readiness
    LR_DATA_READY = auto()
    PACK_DATA_READY = auto()
    # LDS coherence
    LDS_WRITE_AFTER_READ = auto()
    LDS_READ_AFTER_WRITE = auto()
    # DTL=0 specific
    LW_ORDERING = auto()
    GR_VGPR_READY = auto()
    # Scalar safety
    SCALAR_REGISTER_SAFETY = auto()
    # Timing
    QUAD_CYCLE_TIMING = auto()


# All concerns that kernels on a given ISA can require.
ISA_CONCERN_CATALOG: dict[tuple, set[ValidationConcern]] = {
    (9, 5, 0): {    # CDNA 4 (gfx950)
        ValidationConcern.INSTRUCTION_ORDERING,
        ValidationConcern.SCHEDULE_COMPLETENESS,
        ValidationConcern.LR_DATA_READY,
        ValidationConcern.PACK_DATA_READY,
        ValidationConcern.LDS_WRITE_AFTER_READ,
        ValidationConcern.LDS_READ_AFTER_WRITE,
        ValidationConcern.SCALAR_REGISTER_SAFETY,
        ValidationConcern.QUAD_CYCLE_TIMING,
    },
}


def active_concerns(kernel: dict, idmap: dict) -> set[ValidationConcern]:
    """Determine which concerns must be covered for this specific kernel.

    Intersects kernel-derived requirements with the ISA's concern catalog.
    Returns an empty set if the ISA is not in the catalog (graceful fallback
    for ISAs that haven't been characterized yet).
    """
    if "ISA" not in kernel:
        return set()  # No ISA specified — no concern-based validation
    isa = tuple(kernel["ISA"])
    if isa not in ISA_CONCERN_CATALOG:
        return set()  # Unknown ISA — no concern-based validation
    isa_concerns = ISA_CONCERN_CATALOG[isa]

    active = {
        ValidationConcern.INSTRUCTION_ORDERING,
        ValidationConcern.SCHEDULE_COMPLETENESS,
    }

    has_local_reads = any(k.startswith("LR") and not k.startswith("LRS") for k in idmap)
    if has_local_reads:
        active.add(ValidationConcern.LR_DATA_READY)

    has_packs = any(k.startswith("Pack") for k in idmap)
    if has_packs:
        active.add(ValidationConcern.PACK_DATA_READY)

    dtl = kernel.get("DirectToLds", 0)
    if dtl:
        active.add(ValidationConcern.LDS_WRITE_AFTER_READ)
        active.add(ValidationConcern.LDS_READ_AFTER_WRITE)
    else:
        active.add(ValidationConcern.LW_ORDERING)
        active.add(ValidationConcern.GR_VGPR_READY)
        active.add(ValidationConcern.LDS_WRITE_AFTER_READ)
        active.add(ValidationConcern.LDS_READ_AFTER_WRITE)

    has_grinc = any(k.startswith("GRInc") for k in idmap)
    if has_grinc:
        active.add(ValidationConcern.SCALAR_REGISTER_SAFETY)

    if has_packs and kernel.get("UseF32XEmulation", False):
        active.add(ValidationConcern.QUAD_CYCLE_TIMING)

    return active & isa_concerns


# Pack-related invariants live graph-side:
#   * LR -> Pack RAW (dscnt) and Pack -> MFMA RAW: `validate_edge_wait_coverage`
#     + `compare_graphs` over register dataflow from `_GenericALURule`.
#   * MiddlePack pair-interleaving: `validate_middle_pack_pair_interleaving`
#     (emits `OverriddenInputFailure`).
#   * Quad-cycle visibility: `_quad_cycle_gap_ok` / `_cvt_to_mfma_gap_ok` /
#     `_mfma_pack_to_cvt_gap_ok` (emit `TimingTooCloseFailure`).
# Migrated coverage: `test_validate_pack_graph.py`,
# `test_dataflow_graph_register_gaps.py`.

# --- Loop Names ---
MAIN_LOOP_PREV = "ML-1"
MAIN_LOOP = "ML"
NO_GLOBAL_LOAD_LOOP = "NGL"
NO_LOCAL_LOAD_LOOP = "NLL"

# --- Pack Group Sizes ---
PACK_GROUP_SIZE_TF32 = 24        # 4 CVT0 + 16 middle + 4 CVT1
PACK_GROUP_SIZE_TF32_4X4 = 10    # 4 CVT0 + 2 MFMA + 4 CVT1

# --- Quad-Cycle Timing (CDNA 4 ISA section 7.6) ---
# Quad-cycle visibility verdicts live graph-side; the legacy module-scope
# alias constants (`_QUAD_CYCLES_*`, `_MFMA_TYPE_SWITCH_THRESHOLD_*`) live
# in `Tensile/Components/ScheduleCapture.py` alongside the
# `_quad_cycle_gap_ok` / `_cvt_to_mfma_gap_ok` /
# `_mfma_pack_to_cvt_gap_ok` helpers that consume them. The per-arch
# `ArchProfile` dataclass and resolvers are defined below so the validator
# can resolve a profile from `kernel["ISA"]` without a back-import into
# ScheduleCapture.

# --- VGPRs ---
VGPRS_PER_CONVERSION_GROUP = 8   # 8 VGPRs per conversion group in TF32 emulation


# =============================================================================
# ArchProfile — per-architecture quad-cycle and issue-cycle constants
# =============================================================================
# Quad-cycle finish-time and settle-window magic numbers used to live as
# module-scope literals derived from CDNA 4 (gfx950) ISA section 7.6. The
# quad-cycle simulator (`cumulative_issue_cycles`) and the four pair-specific
# helpers (`_quad_cycle_gap_ok`, `_cvt_to_mfma_gap_ok`,
# `_mfma_pack_to_cvt_gap_ok`, `_mfma_finish_cycles_for`) plus the
# per-instruction issue-cost helper (`_min_issue_quad_cycles_for`) now read
# every magic number from a per-arch `ArchProfile`. The default profile keeps
# the historical CDNA 4 values bit-identical, so kernels (and the entire
# unit-test suite) that don't carry an ISA tag still see the original
# behavior. New archs construct a fresh `ArchProfile` and register it in
# `_ARCH_PROFILES_BY_ISA`.

@dataclass(frozen=True)
class ArchProfile:
    """Per-architecture timing constants consumed by the quad-cycle simulator.

    Frozen so the singleton `_DEFAULT_CDNA4_ARCH_PROFILE` is safe to share
    across the entire test suite without any caller mutating it. New archs
    construct a fresh `ArchProfile(...)` rather than copying-and-mutating.

    Fields:
      name: Human-readable architecture tag (e.g. "CDNA4").
      isa: ISA tuple identifying the arch (e.g. (9, 5, 0) for gfx950).
           Used by `_resolve_arch_profile_for_isa` to look up the profile
           from a `kernel["ISA"]` value.

      Quad-cycle finish times for matrix-mul ops:
        standard_mfma_finish_cycles   — full-tile MFMA finish window.
        mfma_4x4_finish_cycles        — 4x4 PackMFMA finish window.

      Pair-specific settle windows:
        cvt_before_mfma_quad_cycles      — CVTPack -> MFMA visibility.
        mfma_4x4_before_cvt_quad_cycles  — 4x4 PackMFMA -> CVT1 visibility.

      MFMA type-switch thresholds:
        mfma_type_switch_threshold_from_standard
        mfma_type_switch_threshold_from_4x4
            When two consecutive MFMAs differ in class, the consumer takes
            an extra +1 quad-cycle stall if the gap is below the
            producer-class threshold.

      Per-instruction issue cost:
        default_issue_quad_cycles  — base issue cost for non-SNop
            instructions (CDNA 4 = 1; arch-specific overrides go here).

    Adding a new arch: instantiate ArchProfile and register it in
    `_ARCH_PROFILES_BY_ISA`. Tests that exercise the new arch attach the
    profile to the FourPartCapture / DataflowGraph via the `arch_profile`
    field so the resolver picks it up.
    """
    name: str
    isa: Tuple[int, int, int]
    standard_mfma_finish_cycles: int
    mfma_4x4_finish_cycles: int
    cvt_before_mfma_quad_cycles: int
    mfma_4x4_before_cvt_quad_cycles: int
    mfma_type_switch_threshold_from_standard: int
    mfma_type_switch_threshold_from_4x4: int
    default_issue_quad_cycles: int = 1


# CDNA 4 (gfx950) — sourced from ISA section 7.6. These are the values that
# lived as module-scope literals in this file before the ArchProfile
# refactor; they remain the default everywhere a profile isn't explicitly
# attached to keep the existing unit-test suite bit-identical.
_DEFAULT_CDNA4_ARCH_PROFILE = ArchProfile(
    name="CDNA4",
    isa=(9, 5, 0),
    standard_mfma_finish_cycles=3,         # Full-tile MFMA: 3 quad-cycles to finish.
    mfma_4x4_finish_cycles=1,              # 4x4 PackMFMA: 1 quad-cycle.
    cvt_before_mfma_quad_cycles=2,         # CVT pack -> MFMA settle window.
    mfma_4x4_before_cvt_quad_cycles=5,     # 4x4 PackMFMA -> CVT1 settle window.
    mfma_type_switch_threshold_from_standard=5,
    mfma_type_switch_threshold_from_4x4=3,
    default_issue_quad_cycles=1,
)


# Lookup table — extend with new archs as their profiles are characterized.
_ARCH_PROFILES_BY_ISA: Dict[Tuple[int, int, int], ArchProfile] = {
    _DEFAULT_CDNA4_ARCH_PROFILE.isa: _DEFAULT_CDNA4_ARCH_PROFILE,
}


def _resolve_arch_profile_for_isa(isa: Optional[Tuple[int, int, int]]) -> ArchProfile:
    """Return the ArchProfile for an ISA tuple. Default = CDNA 4.

    Unknown / missing ISAs intentionally fall back to CDNA 4 rather than
    raising — the validator's pre-existing behavior was hardcoded CDNA 4,
    so this keeps every uncharacterized arch on the historical code path.
    """
    if isa is None:
        return _DEFAULT_CDNA4_ARCH_PROFILE
    return _ARCH_PROFILES_BY_ISA.get(tuple(isa), _DEFAULT_CDNA4_ARCH_PROFILE)


def _resolve_arch_profile(carrier: object) -> ArchProfile:
    """Return the ArchProfile attached to a DataflowGraph or FourPartCapture.

    `carrier` may be:
      - DataflowGraph with `.arch_profile` set,
      - FourPartCapture with `.arch_profile` set,
      - GraphNode (resolves via its captured graph back-ref, if any),
      - None (degenerate test path).

    Defaults to `_DEFAULT_CDNA4_ARCH_PROFILE` when no profile is attached
    so the historical code path stays bit-identical for callers that don't
    plumb arch info through.
    """
    if carrier is None:
        return _DEFAULT_CDNA4_ARCH_PROFILE
    profile = getattr(carrier, "arch_profile", None)
    if isinstance(profile, ArchProfile):
        return profile
    return _DEFAULT_CDNA4_ARCH_PROFILE


# --- Quad-cycle constants (default-CDNA 4 aliases) ---------------------------
# These module-scope names are RETAINED for two reasons:
#   1. Backward compatibility for callers (and tests) that import them as
#      module attributes. They now resolve from `_DEFAULT_CDNA4_ARCH_PROFILE`
#      so any future tweak to the CDNA 4 numbers happens in exactly one
#      place — the profile constructor.
#   2. The classification helpers `_mfma_finish_cycles_for` /
#      `_quad_cycle_gap_ok` use the standard / 4x4 finish-cycle values as
#      DISCRIMINATOR sentinels (e.g. "is the previous MFMA's class the
#      4x4 family?"). Those discriminators are class identity, not
#      arch-specific timing — so they remain pinned to the CDNA 4 default.
#      Per-arch overrides for the actual finish window come from the
#      resolved profile inside each helper.
# Source: CDNA 4 ISA section 7.6.
_QUAD_CYCLES_STANDARD_MFMA_FINISH = _DEFAULT_CDNA4_ARCH_PROFILE.standard_mfma_finish_cycles
_QUAD_CYCLES_MFMA_4X4_FINISH = _DEFAULT_CDNA4_ARCH_PROFILE.mfma_4x4_finish_cycles
_QUAD_CYCLES_CVT_BEFORE_MFMA = _DEFAULT_CDNA4_ARCH_PROFILE.cvt_before_mfma_quad_cycles
_QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1 = _DEFAULT_CDNA4_ARCH_PROFILE.mfma_4x4_before_cvt_quad_cycles

# --- MFMA Type-Switch Thresholds (default-CDNA 4 aliases) --------------------
# When `cumulative_issue_cycles` observes consecutive MFMAs of DIFFERENT
# classes whose issue gap is below the producer's threshold, it injects a +1
# quad-cycle stall — the consumer is forced to issue one cycle later. The
# thresholds depend on the PRODUCER class. Resolved from the CDNA 4 profile
# for the legacy module-scope path; the cycle simulator inside
# `cumulative_issue_cycles` reads the actual values from the resolved
# profile so per-arch overrides apply.
_MFMA_TYPE_SWITCH_THRESHOLD_FROM_STANDARD = (
    _DEFAULT_CDNA4_ARCH_PROFILE.mfma_type_switch_threshold_from_standard
)
_MFMA_TYPE_SWITCH_THRESHOLD_FROM_4X4 = (
    _DEFAULT_CDNA4_ARCH_PROFILE.mfma_type_switch_threshold_from_4x4
)


# =============================================================================
# Dataflow graph — GraphNode / DataflowEdge / DataflowGraph
# =============================================================================
#
# Lives here (not in ScheduleCapture) so the Failure formatters and the
# `cms_node_label` / `_node_position` / `_body_for_node` helpers can reference
# `GraphNode`-shaped objects natively without re-importing across the module
# boundary. ScheduleCapture's `build_dataflow_graph` and `_make_node` instantiate
# these via narrow lazy imports inside their function bodies (the cycle is
# unavoidable: ScheduleCapture imports `SchedulePosition` from CMSValidator at
# runtime — actually the reverse direction; eager imports either way would
# deadlock module init).

# Type alias: every Failure-formatter parameter accepts either a graph-side
# GraphNode (carries `position`, `category`, `tagged_inst`) or a structural-side
# ValidatorInstruction (carries `issued_at` and exposes `category` via property).
# Both shapes are exercised across the test suite; `_node_position` /
# `cms_node_label` discriminate via getattr probes.
NodeLike = Union["GraphNode", "ValidatorInstruction"]


@dataclass
class GraphNode:
    """A node in the unified 4-body dataflow graph.

    identity is the canonical key for cross-graph comparison: position-independent
    (survives CMS reordering) and content-based (same producer in default and CMS
    captures gets the same identity even if its stream position differs).

    position lives in graph-builder space (loop_index spans bodies); the
    underlying TaggedInstruction.slot is preserved on tagged_inst.

    issue_cycles is the per-instruction quad-cycle issue cost: mirrors
    `ValidatorInstruction.min_issue_quad_cycles()` (CMSValidator.py:327 +
    645). Default 1; SNop-shaped instructions return `1 + wait_state`.
    Populated by `_make_node` from a class-dispatch table so the graph-side
    `cumulative_issue_cycles` helper can simulate per-instruction issue costs
    cycle-exactly without re-importing CMSValidator (which would create an
    import cycle — CMSValidator imports from this module). The structural-side
    simulators (`precompute_issue_times`, `estimate_quad_cycles_precomputed`,
    `estimate_quad_cycles`) have been removed; `cumulative_issue_cycles` is
    the single source of truth.
    """
    identity: tuple                     # (rocisa_class_name, loop_index, signature_tuple)
    position: SchedulePosition
    category: str                       # propagated from TaggedInstruction
    rocisa_inst: object                 # back-reference to the rocisa instruction
    tagged_inst: "TaggedInstruction"    # back-reference for stream-position lookup
    body_label: str                     # 'ML-1' | 'ML' | 'NGL' | 'NLL'
    name: str = ""                      # human-readable label (e.g. 'LRA0[2]')
    issue_cycles: int = 1               # per-instruction quad-cycle cost


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
    # Per-architecture timing profile. None = default CDNA 4. Copied from
    # FourPartCapture by `build_dataflow_graph` so the four pair-specific
    # quad-cycle helpers can resolve arch-specific constants from the graph.
    arch_profile: Optional[ArchProfile] = None

    def edge_keys(self):
        """Edge-equality keys for cross-graph diff: (p_id, c_id, resource, kind)."""
        return {(e.producer.identity, e.consumer.identity, e.resource, e.edge_kind)
                for e in self.edges}


# =============================================================================
# Graph walkers + FIFO simulator + identity helpers (br4.6)
# =============================================================================
#
# These helpers operate on `GraphNode`-shaped data and used to live in
# ScheduleCapture.py. They were moved here so the helper bodies can reference
# `GraphNode`, `DataflowGraph`, `_DEFAULT_CDNA4_ARCH_PROFILE`, and
# `_resolve_arch_profile` directly (no lazy reverse-imports). The remaining
# graph-builder / validator entry points in ScheduleCapture.py (notably
# `build_dataflow_graph`, `_collect_pattern`, `diagnose_missing_edge`,
# `_classify_edge_coverage`, `validate_middle_pack_pair_interleaving`) consume
# these helpers via narrow lazy imports inside their function bodies; br4.7,
# br4.8, and br4.9 will move those entry points and eliminate the lazy
# imports.


PRODUCER_CATEGORIES_LDS = ("LRA0", "LRA1", "LRA3", "LRB0", "LRB1", "LRB3",
                           "LWA", "LWB", "LW")
PRODUCER_CATEGORIES_GLOBAL = ("GRA", "GRB", "GR")
SWAIT_CATEGORY = "SYNC"
SBARRIER_CATEGORY = "BARRIER"


def counter_for(node_or_category: Union[str, GraphNode, "ValidatorInstruction"]) -> str:
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


def _swait_drains(swait_node: GraphNode, counter: str) -> Optional[int]:
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


def _all_nodes_in_order(subj_graph: Optional[DataflowGraph]):
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


def waits_in_window(
    subj_graph: DataflowGraph,
    start: SchedulePosition,
    end: SchedulePosition,
    *,
    counter: Optional[str] = None,
    exclude_counter: Optional[str] = None,
) -> List[GraphNode]:
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


def barriers_in_window(
    subj_graph: DataflowGraph, start: SchedulePosition, end: SchedulePosition
) -> List[GraphNode]:
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


def _class_tag(inst) -> str:
    """Return the stable class tag
    (LR/LW/GR/MFMA/SWAIT/SBARRIER/SNOP/SSETPRIO) for an instruction.
    Used as the first element of the identity tuple so diagnostic
    categorization works without parsing the render-string.

    SNop and SSetPrior are recognized here so that the production capture
    path — which may end up assigning category="UNKNOWN" (via
    `_captureSubIterToBuilder`'s fallback) when an instruction is neither
    in the id-map nor matched by the explicit isinstance branches — still
    falls through `_class_tag_from_category(category="UNKNOWN", inst)` to
    a recognized tag rather than raising
    `CaptureUnknownInstructionError`. These tags are excluded from the
    cross-graph data-flow identity set (`build_dataflow_graph` Phase 1)
    just like SWait/SBarrier.
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
    if _is_snop(inst):
        return "SNOP"
    if _is_ssetprio(inst):
        return "SSETPRIO"
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
    if category == "SSETPRIO":
        return "SSETPRIO"
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


def _node_subiter(node: GraphNode, num_mfma_per_subiter: int) -> int:
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


def _min_issue_quad_cycles_for(rocisa_inst, profile: Optional[ArchProfile] = None) -> int:
    """Return the per-instruction quad-cycle issue cost.

    Mirrors `ValidatorInstruction.min_issue_quad_cycles()` from CMSValidator.py:
        - Default `min_issue_quad_cycles_base = 1` (CMSValidator.py:298, 327-328).
        - `SNop.min_issue_quad_cycles` adds `wait_state` (CMSValidator.py:645-647).
    Every other validator dataclass keeps the base cost of 1.

    `profile` defaults to the CDNA 4 arch profile (base = 1). Per-arch
    overrides for the default issue cost come from
    `profile.default_issue_quad_cycles`. The SNop wait-state add is
    arch-independent (the wait_state is encoded in the SNop instruction
    itself, not the arch).

    With the structural-side simulators removed, this helper is the
    canonical per-instruction cost table.
    """
    if profile is None:
        p = _DEFAULT_CDNA4_ARCH_PROFILE
    else:
        p = profile
    base = p.default_issue_quad_cycles
    if rocisa_inst is None:
        return base
    if _is_snop(rocisa_inst):
        # Test-fixture path: _FakeSNop exposes `wait_state` directly.
        wait_state = getattr(rocisa_inst, "wait_state", None)
        if wait_state is not None:
            return base + int(wait_state)
        # Production rocisa path: SNop stores wait_state as the first param
        # (matches CMSValidator.py:1058-1060: `snop.getParams()[0]`).
        get_params = getattr(rocisa_inst, "getParams", None)
        if callable(get_params):
            try:
                params = get_params()
                if params:
                    return base + int(params[0])
            except Exception:
                pass
        return base
    return base


def _make_node(
    tagged_inst: "TaggedInstruction",
    body_label: str,
    profile: Optional[ArchProfile] = None,
) -> GraphNode:
    inst = tagged_inst.wrapped.rocisa_inst
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
        issue_cycles=_min_issue_quad_cycles_for(inst, profile),
    )


# Body order for graph construction. Cross-body queue state persists in the
# order ML-1 -> ML -> NGL -> NLL (matching hardware execution order).
_BODY_BUILD_ORDER = (BODY_LABEL_ML_PREV, BODY_LABEL_ML, BODY_LABEL_NGL, BODY_LABEL_NLL)


# =============================================================================
# Graph builder: build_dataflow_graph + barrier-pattern collectors (br4.7)
# =============================================================================
#
# These three functions used to live in ScheduleCapture.py. They were moved
# here in br4.7 so the bodies can reference DataflowGraph / DataflowEdge /
# _make_node / _BODY_BUILD_ORDER / SWAIT_CATEGORY / SBARRIER_CATEGORY /
# _swait_drains directly (no lazy reverse-imports). The intersection
# primitives they consume (_byte_keys_for_resource, _resolve_producers,
# MemoryRegion) STAY in ScheduleCapture per the long-term-plan audit and
# are imported eagerly at the top of this module.


def build_dataflow_graph(four_part_capture):
    """Build the unified 4-body register dataflow graph from a FourPartCapture.

    Two phases:

    Phase 1 — node construction. Walks bodies in execution order
    (ML-1 -> ML -> NGL -> NLL). Every captured instruction becomes a
    node EXCEPT SWait/SBarrier/SNop (scheduler-choice; sidecar only).
    LCC instructions (SSubU32 + SCmpEQI32, per LCC_AUDIT.md) ARE
    nodes — their per-instruction issue cycles contribute to
    `cumulative_issue_cycles` walks; cross-body cycle counting depends
    on this. Per-body sidecar `_graph_nodes` is attached so
    wait/barrier helpers can find sync ops in stream order.

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

    An entirely-omitted body (key 0 absent from the body's dict) is
    treated as "this body was not emitted by either scheduler", NOT as
    an error. PGR/SuppressNoLoadLoop combinations that legitimately do
    not emit NGL or NLL (see `kernel_emits_n_gl` / `kernel_emits_n_ll`)
    leave the corresponding dict empty; the loop below skips that body
    cleanly and the validator runs against the remaining bodies.

    The empty-instruction-list guard above (`{0: empty_body}`) is
    deliberately distinct from the absent-key path: a present-but-empty
    body indicates the capture pipeline lost data for a body that
    *should* have content, which is a real bug worth raising on. Do not
    "fix" the absence path by synthesizing a `LoopBodyCapture(instructions=[])`
    — that conflates the two error modes; use dict-omission instead.
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
    # Forward the FourPartCapture's arch_profile (or None) to the graph so
    # the four pair-specific quad-cycle helpers can resolve the per-arch
    # constants. None => default CDNA 4 (historical bit-identical path).
    arch_profile = getattr(four_part_capture, 'arch_profile', None)

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
            inst = tagged_inst.wrapped.rocisa_inst
            try:
                node = _make_node(tagged_inst, label, arch_profile)
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
            # participate (excludes scheduler-choice SWait/SBarrier/SNop/
            # SSetPrior). SSetPrior is a wave-priority scalar op with no
            # register dataflow (rocisa SSetPrior takes only `int prior`),
            # mirrored by `_NoDataflowRule`.
            # LCC instructions (SSubU32 + SCmpEQI32) ARE included — their
            # per-instruction issue cycles contribute to
            # `cumulative_issue_cycles` walks; cross-body cycle counting
            # depends on it.
            if not (
                _is_swait(inst) or _is_sbarrier(inst)
                or _is_snop(inst) or _is_ssetprio(inst)
            ):
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
            wrapped = node.tagged_inst.wrapped

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
    # This collector is the sole source of LR0 -> GR LDS-reuse coverage.
    # Two patterns:
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
                         num_mfma_per_subiter=num_mfma_per_subiter,
                         arch_profile=arch_profile)


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


# -----------------------------------------------------------------------------
# FIFO simulator + producer-classifier helpers
# -----------------------------------------------------------------------------


def _queue_depth_at(wait_node: GraphNode, producer: GraphNode, subj_graph: DataflowGraph) -> int:
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


def _producer_queue_position(producer: GraphNode, subj_graph: DataflowGraph) -> int:
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


def _wait_drains_producer(wait_node: GraphNode, producer: GraphNode, subj_graph: DataflowGraph) -> bool:
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


def _any_drains(waits: List[GraphNode], producer: GraphNode, subj_graph: DataflowGraph) -> bool:
    return any(_wait_drains_producer(w, producer, subj_graph) for w in waits)


def _mfma_finish_cycles_for(rocisa_inst, profile: Optional[ArchProfile] = None) -> int:
    """Classify an MFMA-shaped rocisa instruction as standard or 4x4 PackMFMA
    and return the per-arch finish-cycle count.

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

    `profile` defaults to the CDNA 4 arch profile so the historical call
    sites (which pass only the rocisa instance) keep returning identical
    values. Per-arch overrides come from `profile.standard_mfma_finish_cycles`
    and `profile.mfma_4x4_finish_cycles`.
    """
    if profile is None:
        p = _DEFAULT_CDNA4_ARCH_PROFILE
    else:
        p = profile
    if rocisa_inst is None:
        return p.standard_mfma_finish_cycles
    # Fast path: test fixtures expose `variant` directly.
    variant = getattr(rocisa_inst, "variant", None)
    if variant is not None:
        try:
            m, n = variant[0], variant[1]
        except (IndexError, TypeError):
            m = n = None
        if m == 4 and n == 4:
            return p.mfma_4x4_finish_cycles
        if m is not None:
            return p.standard_mfma_finish_cycles
    # Production rocisa MFMAInstruction does not expose `variant` as an
    # attribute — discriminate by parsing the rendered assembly form.
    try:
        rendered = str(rocisa_inst)
    except Exception:
        return p.standard_mfma_finish_cycles
    if "_4x4x" in rendered:
        return p.mfma_4x4_finish_cycles
    return p.standard_mfma_finish_cycles


def _is_mfma_pack_producer(producer: GraphNode) -> bool:
    """Return True for a 4x4 PackMFMA producer.

    PackMFMAs (TF32 4x4 emulation) are syntactically `MFMAInstruction` rocisa
    objects but are categorized as `PackA*` / `PackB*` because the macro
    classifier groups them with the surrounding CVT pack chain. Discrimination:
    `category.startswith("Pack")` AND `rocisa_inst` is an MFMA-shaped class.

    Used by `_is_mfma_producer` so PackMFMAs route to the quad-cycle gap
    branch rather than the ALU-immediate-visibility branch — without this
    discriminator the ALU-producer exemption would fire first and PackMFMA
    producers would skip timing checks entirely.
    """
    if not getattr(producer, "category", "").startswith("Pack"):
        return False
    inst = getattr(producer, "rocisa_inst", None)
    if inst is None:
        return False
    return _is_mfma(inst)


def _is_mfma_producer(producer: GraphNode) -> bool:
    """True for any producer subject to MFMA quad-cycle finish-time gating.

    Two shapes:
      - `category == "MFMA"` — the standard MFMA path (everything but the
        TF32 4x4 emulation pack chain).
      - PackMFMA — `category.startswith("Pack")` with an MFMA-shaped rocisa
        instance. The dispatch in `_classify_edge_coverage` and
        `diagnose_missing_edge` claims pack-MFMA producers BEFORE the
        ALU-producer exemption fires.
    """
    if getattr(producer, "category", None) == "MFMA":
        return True
    return _is_mfma_pack_producer(producer)


def _is_cvt_pack_producer(producer: GraphNode) -> bool:
    """True for a CVTPack producer (TF32 v_cvt_pk_bf16_f32 family).

    CVTPacks are categorized `Pack*` (PackA0/PackA1/PackB0/PackB1/PackA3/
    PackB3 depending on the surrounding LR group); discrimination here is
    `category.startswith("Pack")` AND `rocisa_inst` is the
    `VCvtPkF32toBF16` rocisa class. Used by `_classify_edge_coverage` and
    `diagnose_missing_edge` so CVTPack-feeding-MFMA edges are routed to
    `_cvt_to_mfma_gap_ok` BEFORE the ALU-immediate exemption claims them
    — same shape as the PackMFMA carve-out, but with the CVT class set
    in place of the MFMA class set.

    The structural-side mirror (`_handle_min_pack_quad_cycles`,
    `Pack.min_quad_cycles_before_result_used`,
    `Pack.estimated_quad_cycles_before_result_used`, `estimate_quad_cycles`)
    has been removed; the graph-side dispatch in
    `_classify_edge_coverage` is now the only enforcement path for the
    CVT -> MFMA settle window.
    """
    if not getattr(producer, "category", "").startswith("Pack"):
        return False
    inst = getattr(producer, "rocisa_inst", None)
    if inst is None:
        return False
    return _is_cvt_pack(inst)


def cumulative_issue_cycles(graph: DataflowGraph, producer: GraphNode, consumer: GraphNode) -> int:
    """Return the exact number of quad-cycles between producer's issue
    completion and consumer's issue start.

    Replaces the slot-delta + subiter approximation with cycle-exact
    arithmetic. Originally derived from the (now-removed) structural-side
    `precompute_issue_times` / `estimate_quad_cycles_precomputed` /
    `estimate_quad_cycles` simulators in CMSValidator.py; this helper is
    now the canonical implementation.
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

    Cross-body: when producer and consumer live in different captured
    bodies, the simulator continues across body boundaries in
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

    # Resolve the per-arch profile from the graph; default = CDNA 4 so
    # callers that haven't plumbed arch info through get the historical
    # bit-identical numbers. The simulator below uses
    # `profile.mfma_4x4_finish_cycles` for MFMA-class discrimination
    # (rather than the legacy module-scope alias) so per-arch overrides
    # to the 4x4 finish window don't decouple from the discriminator.
    profile = _resolve_arch_profile(graph)

    # Producer must always be strictly before consumer in stream order. The
    # SchedulePosition `__lt__` compares (loop_index, vmfma_index, sub_index)
    # so this single check covers same-body and cross-body cases uniformly.
    if not (producer.position < consumer.position):
        return 0

    # Build the list of bodies to traverse, starting from the producer's
    # body and continuing forward through `_BODY_BUILD_ORDER` until (and
    # including) the consumer's body. The simulator state —
    # `current_issue`, `mfma_free_at`, `last_mfma_class`,
    # `last_mfma_issue` — is preserved across body boundaries because the
    # captured bodies issue back-to-back in hardware execution order.
    # There is no extra "body boundary" stall injected; every
    # per-instruction cost is already accounted for as we walk each body's
    # instructions, so the cross-body gap is just the sum of intervening
    # instruction issue costs.
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
    # body membership).
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
            wrapped = getattr(ti, "wrapped", None)
            inst = getattr(wrapped, "rocisa_inst", None) if wrapped is not None else None
            is_mfma = inst is not None and _is_mfma(inst)
            if is_mfma:
                current_issue = max(current_issue, mfma_free_at)
                current_mfma_class = _mfma_finish_cycles_for(inst, profile)
                if last_mfma_class is not None and current_mfma_class != last_mfma_class:
                    gap = current_issue - last_mfma_issue
                    # Threshold is producer-class-keyed; thresholds come from
                    # the resolved arch profile. Discrimination uses the
                    # profile's own 4x4 finish-cycle value so per-arch
                    # overrides don't decouple discriminator from threshold.
                    threshold = (
                        profile.mfma_type_switch_threshold_from_4x4
                        if last_mfma_class == profile.mfma_4x4_finish_cycles
                        else profile.mfma_type_switch_threshold_from_standard
                    )
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
            current_issue += _min_issue_quad_cycles_for(inst, profile)

        if c_issue_start is not None:
            break

    if p_issue_start is None or c_issue_start is None:
        return 0
    return c_issue_start - p_issue_start - 1


def _quad_cycle_gap_ok(
    producer: GraphNode,
    consumer: GraphNode,
    num_mfma_per_subiter: int = 0,
    graph: Optional[DataflowGraph] = None,
) -> Tuple[bool, int, int]:
    """Verify that enough quad-cycles separate an MFMA producer from its
    consumer for the producer's result to be visible.

    Returns (ok, expected_quad_cycles, actual_quad_cycles).

    Same-body and cross-body share ONE code path that delegates to
    `cumulative_issue_cycles`. The hardware MFMA pipeline does not reset
    at body boundaries — `mfma_free_at` and the type-switch stall carry
    through — so the cross-body cycle gap is just the same simulator
    extended over the body boundary. (A previous `body_delta * 1000`
    cross-body placeholder always returned ok=True regardless of how
    tight the boundary actually was; do not reintroduce it.)
    `cumulative_issue_cycles` walks the unified instruction stream
    across all bodies in `_BODY_BUILD_ORDER`.

    `num_mfma_per_subiter` is retained as a positional parameter for
    backward compatibility with existing call sites and tests but is no
    longer consulted (the helper has the body context). `graph` is the
    DataflowGraph the producer/consumer belong to; when omitted (or when
    the body can't be located) we degrade gracefully by reporting an
    `actual` of 0 — strictly conservative, will fail the gap check.
    """
    profile = _resolve_arch_profile(graph)
    finish = _mfma_finish_cycles_for(getattr(producer, "rocisa_inst", None), profile)
    expected = finish

    if graph is None:
        # No graph passed (degenerate test path): treat as zero-gap. Strict
        # callers always pass `graph=subj_graph`.
        return False, expected, 0

    actual = cumulative_issue_cycles(graph, producer, consumer)
    return actual >= expected, expected, actual


def _cvt_to_mfma_gap_ok(
    producer: GraphNode, consumer: GraphNode, subj_graph: Optional[DataflowGraph]
) -> Tuple[bool, int, int]:
    """Verify that enough quad-cycles separate a CVTPack producer from its
    downstream MFMA consumer for the CVT result to be visible.

    The threshold is fixed at `_QUAD_CYCLES_CVT_BEFORE_MFMA == 2`
    (CDNA 4 ISA 7.6).

    Returns `(ok, expected, actual)` — same triple shape as
    `_quad_cycle_gap_ok` so callers can wrap a single
    `TimingTooCloseFailure(expected, actual)` regardless of the gap kind.

    Same-body and cross-body share ONE code path that delegates to
    `cumulative_issue_cycles`. WARNING for future reverts: the previous
    slot-delta formula (`slot_delta * (1 + finish) - 1 + intervening`)
    was DOUBLE-COUNTING — it charged 1 cycle per slot-INDEX gap AND
    +intervening for actual instructions in those slots. The cycle-exact
    walk only counts actual instructions, producing a smaller (more
    conservative) `actual` for densely-populated streams. A previous
    `body_delta * 1000` cross-body placeholder is also gone;
    `cumulative_issue_cycles` walks the unified instruction stream
    across all bodies in `_BODY_BUILD_ORDER`.
    """
    profile = _resolve_arch_profile(subj_graph)
    expected = profile.cvt_before_mfma_quad_cycles

    if subj_graph is None:
        return False, expected, 0  # Strict: no graph -> conservative fail.

    actual = cumulative_issue_cycles(subj_graph, producer, consumer)
    return actual >= expected, expected, actual


def _mfma_pack_to_cvt_gap_ok(
    producer: GraphNode, consumer: GraphNode, subj_graph: Optional[DataflowGraph]
) -> Tuple[bool, int, int]:
    """Verify that enough quad-cycles separate a 4x4 PackMFMA producer from
    its downstream CVTPack (CVT1) consumer for the accumulator to settle.

    The threshold is fixed at `_QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1 == 5`
    (CDNA 4 ISA 7.6). This is the LARGEST gap among the four
    section-7.6 quad-cycle constants; the 4x4 MFMA finish-cycle (1) is
    shorter than the 5-cycle visibility window the CVT1 needs, so this
    helper enforces a larger min-gap on PackMFMA->CVTPack edges than
    the bare finish would suggest.

    Returns `(ok, expected, actual)` — same triple shape as
    `_quad_cycle_gap_ok` and `_cvt_to_mfma_gap_ok` so callers can wrap a
    single `TimingTooCloseFailure(expected, actual)` regardless of the
    gap kind.

    Approach: CYCLE-EXACT via `cumulative_issue_cycles`, the same
    simulator `_quad_cycle_gap_ok` uses. The helper walks the captured
    stream from the producer to the consumer, accumulating
    per-instruction issue costs plus MFMA-specific finish-time and
    type-switch stalls. The structural-side `precompute_issue_times`
    simulator that this helper originally mirrored has been removed;
    the graph-side path is now the single source of truth for the
    PackMFMA -> CVT settle window.

    Same-body and cross-body share the SAME code path. The
    cross-iteration distinction is a red herring — the graph has all
    instructions laid out in execution order regardless of which body
    they belong to, so `cumulative_issue_cycles` (extended to walk
    across body boundaries in `_BODY_BUILD_ORDER`) is THE function that
    computes the actual cycle gap. (A previous `body_delta * 1000`
    always-true placeholder is gone; do not reintroduce it.) Cross-body
    PackMFMA -> CVT1 edges are enforced with the same 5-quad-cycle
    threshold as same-body edges.
    """
    profile = _resolve_arch_profile(subj_graph)
    expected = profile.mfma_4x4_before_cvt_quad_cycles

    if subj_graph is None:
        # Strict: no graph -> conservative fail (cannot compute gap).
        return False, expected, 0

    actual = cumulative_issue_cycles(subj_graph, producer, consumer)
    return actual >= expected, expected, actual


def _is_alu_producer(producer: GraphNode) -> bool:
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

    A special case carves out the TF32 4x4 PackMFMA: those are
    categorized `Pack*` but the rocisa class is `MFMAInstruction`, so they
    DO need the quad-cycle finish-time gap modelled (1 quad-cycle for
    v_mfma_f32_4x4x4_*). Without the carve-out the ALU-immediate
    exemption would fire first, the quad-cycle branch would never run
    for PackMFMA producers, and a same-slot PackMFMA->MFMA acc chain
    would silently slip past the timing check. PackMFMAs route to the
    MFMA branch via `_is_mfma_pack_producer`; the rest of the Pack*
    category (CVT0/CVT1/middle packs / SwapPacks) stays on the ALU
    exemption.
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


def _first_insufficient(
    waits: List[GraphNode], producer: GraphNode, subj_graph: DataflowGraph
) -> Optional[GraphNode]:
    """Return the first wait (in stream order) that does NOT drain the producer
    despite drainable counter. None if every wait drains, or no wait applies."""
    for w in waits:
        if not _wait_drains_producer(w, producer, subj_graph):
            return w
    return None


def _last_drain(
    waits: List[GraphNode], producer: GraphNode, subj_graph: DataflowGraph
) -> Optional[GraphNode]:
    """Return the latest wait that drained the producer, else None."""
    drainers = [w for w in waits if _wait_drains_producer(w, producer, subj_graph)]
    if not drainers:
        return None
    return max(drainers, key=lambda w: w.position)


# =============================================================================
# Failure hierarchy — typed scheduling defects with polymorphic formatters
# =============================================================================
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


class _PositionStr(str):
    """A `str` subclass for FailureNodeLabel.position that also exposes
    `vmfma_index` (parsed from the trailing '@ idx=N' segment for CMS-side
    labels, '@ stream_pos=N' for asm-side labels).

    Subclassing `str` keeps the formatter side trivial — `f"{label.position}"`
    yields the literal string — while letting structured callers
    (test assertions, diagnostics tooling) recover the integer position
    without parsing the string themselves. Returns -1 if the position
    string carries no integer suffix.
    """

    __slots__ = ()

    @property
    def vmfma_index(self) -> int:
        eq = self.rfind("=")
        if eq == -1:
            return -1
        try:
            return int(self[eq + 1:])
        except ValueError:
            return -1


def _node_position(node: NodeLike) -> SchedulePosition:
    """Resolve the SchedulePosition for a NodeLike (GraphNode or
    ValidatorInstruction). GraphNode has `position`; ValidatorInstruction
    has `issued_at`. Mirrors the getattr probe used by the old free
    helpers."""
    return getattr(node, "position", None) or node.issued_at


@dataclass(frozen=True)
class FailureNodeLabel:
    """Pre-rendered identification of a node in a Failure message.

    Computed by source-aware label providers (e.g. CMS-side `cms_node_label`)
    at Failure-construction time so the Failure carries no back-reference to
    captures, graphs, or rocisa instances. The rendering layer is therefore
    source-agnostic: a non-CMS provider (e.g. raw asm) supplies the same
    `(primary, position)` pair shape with its own naming convention.

    Conventions used today:
      - CMS-side primary:   'LRA0[3]', 'PackA0[2]', 'MFMA' (no [N] for plain MFMA)
      - CMS-side position:  '@ idx=5'  (vmfma_index)
      - Asm-side primary:   'ds_load_b128' (rocisa class name)
      - Asm-side position:  '@ stream_pos=42'

    Optional `category` and `body_label` carry source-side metadata that's
    useful for non-rendering inspections (test assertions, structured
    diagnostics). They are NOT consumed by `_format_canonical` — that
    method reads only `primary` + `position`. Sources that don't carry
    these concepts (raw asm) leave them unset.
    """
    primary: str
    position: str
    # Optional source-side metadata. Reading-only — formatters never touch.
    category: Optional[str] = None
    body_label: Optional[str] = None


@dataclass
class Failure:
    """Common base for all reported scheduling problems.

    `iter_delta` is the canonical loop-offset between consumer and producer
    (consumer.loop_index - producer.loop_index). It's stored on every Failure
    so the rendering layer can decide whether to append the "(of next
    iteration)" suffix without re-deriving it from positions stored elsewhere.
    InvalidCounterValueFailure carries 0 (vestigial — it has no
    producer/consumer pair).
    """

    iter_delta: int = 0

    def _iter_suffix(self) -> str:
        """Render the cross-iteration suffix from `iter_delta`.

        delta=0  ->  ''                                   (same iteration)
        delta=1  ->  ' (of next iteration)'               (i -> i+1, dominant case)
        else    ->  ' (consumer is N iterations after producer)'
        """
        if self.iter_delta == 1:
            return " (of next iteration)"
        if self.iter_delta == 0:
            return ""
        return f" (consumer is {self.iter_delta} iterations after producer)"

    def format(self) -> str:
        """Stable boundary method. Delegates to the subclass canonical
        formatter. No argument: every Failure carries pre-rendered
        `FailureNodeLabel`s plus scalar fields, so formatting is pure
        string composition."""
        return self._format_canonical()

    def _format_canonical(self) -> str:
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
    producer: FailureNodeLabel = None
    consumer: FailureNodeLabel = None
    default_producer_position: Optional[SchedulePosition] = None  # default-side, for diagnostics
    default_consumer_position: Optional[SchedulePosition] = None  # default-side, for diagnostics

    def _format_canonical(self) -> str:
        return (
            f"Producer {self.producer.primary} {self.producer.position} "
            f"is issued after consumer {self.consumer.primary} "
            f"{self.consumer.position}{self._iter_suffix()}."
        )


# ----------------------------------------------------------------------------
# 2. MissingWaitFailure — no SWaitCnt drains the expected counter in the
#    window between producer and consumer. If other-counter SWaitCnts ARE
#    in the window, they're surfaced via `nearby_wait_indices` so
#    the user knows they could extend an existing SWaitCnt rather than
#    insert a new one. (Bead `hof` collapsed the former
#    WaitOnWrongCounterFailure into this single type — the user-facing
#    fix is the same in both cases.)
# ----------------------------------------------------------------------------
@dataclass
class MissingWaitFailure(Failure):
    producer: FailureNodeLabel = None
    consumer: FailureNodeLabel = None
    counter_kind: str = ""  # 'dscnt' / 'vlcnt' / 'vscnt'
    # Pre-extracted vmfma indices of nearby SWaitCnts that drain OTHER counters
    # (formerly stored as full GraphNode list under
    # `nearby_other_counter_waits`; reduced to scalars so the Failure carries
    # no graph back-references).
    nearby_wait_indices: Tuple[int, ...] = ()

    def _format_canonical(self) -> str:
        # Optional hint when other-counter SWaitCnts exist in the window:
        # the user could extend one of them rather than insert a new SWaitCnt.
        hint = ""
        if self.nearby_wait_indices:
            indices = ", ".join(f"idx={i}" for i in self.nearby_wait_indices)
            hint = f" (existing SWaitCnts at {indices} drain other counters)"
        return (
            f"SWaitCnt({self.counter_kind}) missing between producer "
            f"{self.producer.primary} {self.producer.position} and consumer "
            f"{self.consumer.primary} {self.consumer.position}"
            f"{self._iter_suffix()}{hint}."
        )


# ----------------------------------------------------------------------------
# 4. WaitInsufficientFailure — wait at correct position but counter value too lax.
# ----------------------------------------------------------------------------
@dataclass
class WaitInsufficientFailure(Failure):
    producer: FailureNodeLabel = None
    consumer: FailureNodeLabel = None
    wait_idx: int = 0  # vmfma_index of the failing SWaitCnt
    counter_kind: str = ""  # 'dscnt' / 'vlcnt' / 'vscnt'
    counter_value: int = 0
    queue_depth_at_wait: int = 0
    producer_position: int = 0  # 0-indexed FIFO position when producer entered the queue

    def _format_canonical(self) -> str:
        # max acceptable counter value = queue_depth - producer_position - 1
        # (drain enough ops that producer's slot falls inside the drained range).
        # Counter value of -1 means "no constraint" (counter ignored), so the
        # acceptable range is [0, max_acceptable] — -1 does NOT satisfy.
        max_acceptable = self.queue_depth_at_wait - self.producer_position - 1
        if max_acceptable <= 0:
            bound = "must be 0"
        else:
            bound = f"must be in range [0, {max_acceptable}]"
        return (
            f"{self.counter_kind} for SWaitCnt @ idx={self.wait_idx} "
            f"is too high to guarantee producer "
            f"{self.producer.primary} {self.producer.position} for consumer "
            f"{self.consumer.primary} {self.consumer.position}"
            f"{self._iter_suffix()}. "
            f"Current value of {self.counter_value} {bound}."
        )


# ----------------------------------------------------------------------------
# 6. MissingBarrierFailure — cross-wave LDS-reuse needs barrier in the window.
# ----------------------------------------------------------------------------
@dataclass
class MissingBarrierFailure(Failure):
    producer: FailureNodeLabel = None
    consumer: FailureNodeLabel = None
    # role distinguishes the LDS-coherence direction:
    #   'must_start_after' - producer is LR (read), consumer is GR (overwrite)
    #   'needed_by'        - producer is GR (write), consumer is LR (read)
    # Used by tests to assert which scenario triggered; not rendered in the
    # user-facing message (producer/consumer categories make the direction obvious).
    role: str = "must_start_after"
    wait_idx: Optional[int] = None  # vmfma_index of the SWaitCnt that drained the producer

    def _format_canonical(self) -> str:
        if self.wait_idx is not None:
            wait_part = f"between SWaitCnt @ idx={self.wait_idx} and consumer"
        else:
            wait_part = "between covering SWaitCnt and consumer"
        return (
            f"SBarrier missing {wait_part} "
            f"{self.consumer.primary} {self.consumer.position}"
            f"{self._iter_suffix()}, needed for producer "
            f"{self.producer.primary} {self.producer.position}."
        )


# ----------------------------------------------------------------------------
# 8. TimingTooCloseFailure — quad-cycle gap too small (Pack timing).
# ----------------------------------------------------------------------------
@dataclass
class TimingTooCloseFailure(Failure):
    producer: FailureNodeLabel = None  # Pack
    consumer: FailureNodeLabel = None  # Pack/MFMA
    expected_quad_cycles: int = 0
    actual_quad_cycles: int = 0

    def _format_canonical(self) -> str:
        return (
            f"Not enough quad-cycles between producer "
            f"{self.producer.primary} {self.producer.position} and consumer "
            f"{self.consumer.primary} {self.consumer.position}"
            f"{self._iter_suffix()}. "
            f"Need at least {self.expected_quad_cycles} quad-cycles "
            f"but only {self.actual_quad_cycles} guaranteed."
        )


# ----------------------------------------------------------------------------
# 9. InvalidCounterValueFailure — SWait field range check.
# ----------------------------------------------------------------------------
@dataclass
class InvalidCounterValueFailure(Failure):
    # Pre-extracted vmfma index of the offending SWaitCnt. Single-instruction
    # structural failure — no producer/consumer pair, so iter_delta stays at
    # the base default (0).
    swait_idx: int = 0
    dscnt: int = 0
    vlcnt: int = 0
    vscnt: int = 0

    def _format_canonical(self) -> str:
        bad = [(name, val) for name, val in
               (("dscnt", self.dscnt), ("vlcnt", self.vlcnt), ("vscnt", self.vscnt))
               if val < -1]
        bad_str = ", ".join(f"{name}={val}" for name, val in bad)
        return (
            f"SWaitCnt @ idx={self.swait_idx} is invalid: "
            f"{bad_str}. All counter fields must be >= -1."
        )


# ----------------------------------------------------------------------------
# 8. OverriddenInputFailure — intervening write clobbers the resource the
#     consumer needed before the consumer reads it.
#
#     Unifies the bug shape that previously had two failure classes:
#       - SCC carry-chain clobber (intervening SCC writer between two
#         GRInc-style ops that share the SCC carry).
#       - Pack pair-leader vgpr clobber (next pack-pair-leader scheduled
#         between a pair-leader and its pair-consumer, overwriting the
#         vgpr the pair-consumer needs to read).
#     Same shape, different resource.
#
#     intervening_writer may be None when the graph-side classifier knows
#     the resource was clobbered but cannot pinpoint the writer.
# ----------------------------------------------------------------------------
@dataclass
class OverriddenInputFailure(Failure):
    producer: FailureNodeLabel = None            # wrote the value the consumer needs
    consumer: FailureNodeLabel = None            # needed to read producer's value
    resource: str = ""                           # e.g. "SCC", "vgpr"
    intervening_writer: FailureNodeLabel = None  # overwrote the resource between producer and consumer

    def _format_canonical(self) -> str:
        return (
            f"{self.intervening_writer.primary} {self.intervening_writer.position} is "
            f"incorrectly scheduled between producer "
            f"{self.producer.primary} {self.producer.position} and consumer "
            f"{self.consumer.primary} {self.consumer.position}, clobbering the "
            f"{self.resource} that the consumer needs."
        )


# =============================================================================
# CMS-side FailureNodeLabel provider
# =============================================================================
#
# Mirrors the wording previously produced by the now-removed `_node_label`
# / `_node_with_pos` helpers. Source-aware: takes a (node, body_capture)
# pair and renders the per-category-stream `[N]` index against THAT body's
# instruction list. The compare/validate code that emits Failures looks
# up `node.body_label` against the FourPartCapture's per-body captures and
# calls this helper for each producer/consumer node in scope.
#
# A non-CMS source (e.g. raw asm) would supply its own label provider
# producing `FailureNodeLabel` instances of the same shape but with the
# source's own naming convention (see FailureNodeLabel docstring).


def cms_node_label(
    node: NodeLike,
    body_capture: Optional["LoopBodyCapture"],
) -> FailureNodeLabel:
    """Construct a FailureNodeLabel for a CMS-side node.

    Wording is identical to the old `_node_with_pos`:
      - primary: 'category[N]' (per-category-stream 0-based index in the
        node's body capture); plain MFMA omits '[N]'.
      - position: '@ idx={vmfma_index}'.

    `body_capture` is the LoopBodyCapture for the body that emitted this
    node (resolved by the caller from `node.body_label`). When
    `body_capture` is None or the node has no `tagged_inst` recorded in
    that body, the helper falls back to a bare `category` primary —
    important for SWaitCnt / SBarrier / SNop nodes (no tagged_inst by
    construction) and for cross-body callsites that don't index every
    body's tagged_inst stream.
    """
    cat = node.category
    if cat == "MFMA":
        primary = cat
    else:
        tagged = getattr(node, "tagged_inst", None)
        primary = cat
        if body_capture is not None and tagged is not None:
            same_cat = [
                t for t in body_capture.instructions
                if getattr(t, "category", None) == cat
            ]
            try:
                idx = same_cat.index(tagged)
                primary = f"{cat}[{idx}]"
            except ValueError:
                # Lookup miss (cross-body callsite, or a body capture that
                # genuinely doesn't contain this node) — degrade to bare
                # category. The new label API tolerates this gracefully
                # because labels are computed eagerly; the previous
                # `_node_label` raised here, which is exactly the source
                # of the body-mismatch ValueError this bead resolves.
                pass
    pos = _node_position(node)
    return FailureNodeLabel(
        primary=primary,
        position=_PositionStr(f"@ idx={pos.vmfma_index}"),
        category=cat,
        body_label=getattr(node, "body_label", None),
    )


def _cms_iter_delta(producer: NodeLike, consumer: NodeLike) -> int:
    """Compute the canonical loop-offset between consumer and producer.

    Mirrors the old `_iter_note`'s loop_index arithmetic. Returns the raw
    integer delta (consumer.loop_index - producer.loop_index) so the
    Failure base's `_iter_suffix` can pick the appropriate suffix.
    """
    p_pos = _node_position(producer)
    c_pos = _node_position(consumer)
    return c_pos.loop_index - p_pos.loop_index


def _body_for_node(graph: "DataflowGraph", node: NodeLike) -> Optional["LoopBodyCapture"]:
    """Look up the LoopBodyCapture that emitted `node`.

    Uses `node.body_label` against `graph.captures`. Returns None for
    NodeLike values without a body_label (e.g. some test stubs) or when
    the body isn't present in the graph (kernel didn't emit it).
    """
    label = getattr(node, "body_label", None)
    if label is None or graph is None:
        return None
    return graph.captures.get(label)


# =============================================================================
# Cross-graph comparison: compare_graphs + diagnose_missing_edge
# =============================================================================
# compare_graphs takes the default-side (reference) and CMS-side (subject)
# DataflowGraphs and emits Failure objects for every reference edge missing
# from the subject graph, routed through diagnose_missing_edge.
#
# diagnose_missing_edge classifies a single missing edge into one of:
#   OrderInvertedFailure, MissingWaitFailure, WaitInsufficientFailure,
#   MissingBarrierFailure, OverriddenInputFailure, TimingTooCloseFailure.


def compare_graphs(
    reference: DataflowGraph,
    subject: DataflowGraph,
    *,
    raise_on_unexplained: bool = True,
) -> List["Failure"]:
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


def diagnose_missing_edge(
    ref_edge: DataflowEdge,
    subj_graph: DataflowGraph,
    *,
    raise_on_unexplained: bool = True,
) -> List["Failure"]:
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
                producer=cms_node_label(p_node, _body_for_node(subj_graph, p_node)),
                consumer=cms_node_label(c_node, _body_for_node(subj_graph, c_node)),
                iter_delta=_cms_iter_delta(p_node, c_node),
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

    # SCC-typed missing edge: if the reference edge's resource is the SCC
    # sentinel and Phase-1's order check passed, the most likely cause is
    # a CLOBBER — an unrelated SCC writer issued between the producer and
    # consumer in the subject schedule, displacing the producer's SCC
    # value. Find that intervening writer in the subject graph (the new
    # SCC producer the consumer pairs with) and emit a typed
    # OverriddenInputFailure carrying the producer/consumer/clobber triple.
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
            return [OverriddenInputFailure(
                producer=cms_node_label(p_node, _body_for_node(subj_graph, p_node)),
                consumer=cms_node_label(c_node, _body_for_node(subj_graph, c_node)),
                iter_delta=_cms_iter_delta(p_node, c_node),
                resource="SCC",
                intervening_writer=cms_node_label(
                    intervening_writer,
                    _body_for_node(subj_graph, intervening_writer),
                ),
            )]
        # No intervening SCC writer found — the consumer's SCC slot is
        # simply unsourced in subj. Fall through to the generic ALU early
        # return so we don't double-emit on a non-clobber miss.

    # 4x4 PackMFMA-as-producer feeding CVTPack-as-consumer: 5-quad-cycle
    # settle window (CDNA 4 ISA 7.6, `_QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1`).
    # Must run BEFORE the generic `_is_mfma_producer` branch below:
    # `_is_mfma_producer` claims PackMFMA producers and would otherwise
    # route this pair through `_quad_cycle_gap_ok`, which uses
    # `_mfma_finish_cycles_for(producer) == 1` for 4x4 PackMFMAs — too
    # weak by 4 quad-cycles versus the 5-cycle CVT1 visibility window.
    # The structural-side mirror (`_handle_min_pack_quad_cycles` /
    # `MFMAPack.min_quad_cycles_before_result_used`) was removed; this
    # dispatch is now the only enforcement path.
    if (_is_mfma_pack_producer(p_node)
            and _is_cvt_pack_producer(c_node)):
        ok, expected, actual = _mfma_pack_to_cvt_gap_ok(
            p_node, c_node, subj_graph)
        if not ok:
            return [TimingTooCloseFailure(
                producer=cms_node_label(p_node, _body_for_node(subj_graph, p_node)),
                consumer=cms_node_label(c_node, _body_for_node(subj_graph, c_node)),
                iter_delta=_cms_iter_delta(p_node, c_node),
                expected_quad_cycles=expected,
                actual_quad_cycles=actual,
            )]
        return []

    # MFMA-as-producer: governed by quad-cycle issue-timing constraints,
    # not by the dscnt/vlcnt FIFO. An explicit gap check fires; an MFMA
    # producer whose consumer fires too soon after it is a TimingTooClose
    # violation. Dispatch is via `_is_mfma_producer` (rather than `category
    # == "MFMA"`) so 4x4 PackMFMAs (categorized Pack* but syntactically
    # MFMAInstruction) are routed here BEFORE the ALU exemption claims
    # them.
    if _is_mfma_producer(p_node):
        nmps = subj_graph.num_mfma_per_subiter
        ok, expected, actual = _quad_cycle_gap_ok(p_node, c_node, nmps, graph=subj_graph)
        if not ok:
            return [TimingTooCloseFailure(
                producer=cms_node_label(p_node, _body_for_node(subj_graph, p_node)),
                consumer=cms_node_label(c_node, _body_for_node(subj_graph, c_node)),
                iter_delta=_cms_iter_delta(p_node, c_node),
                expected_quad_cycles=expected,
                actual_quad_cycles=actual,
            )]
        return []

    # CVTPack-as-producer feeding MFMA-as-consumer: 2-quad-cycle settle
    # window (CDNA 4 ISA 7.6, `_QUAD_CYCLES_CVT_BEFORE_MFMA`). The
    # structural-side mirror was removed; this dispatch is now the only
    # enforcement path. Must precede the ALU-immediate exemption below
    # (same dispatch-order constraint as the MFMA branch above) so
    # CVTPacks don't get silently waved through. Non-MFMA consumers fall
    # through to the ALU exemption — only the CVT->MFMA edge carries the
    # quad-cycle constraint.
    if _is_cvt_pack_producer(p_node) and _is_mfma_producer(c_node):
        ok, expected, actual = _cvt_to_mfma_gap_ok(p_node, c_node, subj_graph)
        if not ok:
            return [TimingTooCloseFailure(
                producer=cms_node_label(p_node, _body_for_node(subj_graph, p_node)),
                consumer=cms_node_label(c_node, _body_for_node(subj_graph, c_node)),
                iter_delta=_cms_iter_delta(p_node, c_node),
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

    p_label = cms_node_label(p_node, _body_for_node(subj_graph, p_node))
    c_label = cms_node_label(c_node, _body_for_node(subj_graph, c_node))
    iter_delta = _cms_iter_delta(p_node, c_node)

    if not waits:
        # No SWait on the expected counter at all in the window. If other-
        # counter SWaits exist, surface their vmfma indices via
        # `nearby_wait_indices` so the user can extend one of them rather
        # than insert a new SWaitCnt; the underlying fix is the same either
        # way.
        nearby_indices = tuple(_node_position(w).vmfma_index for w in waits_other)
        failures.append(MissingWaitFailure(
            producer=p_label,
            consumer=c_label,
            iter_delta=iter_delta,
            counter_kind=expected_counter,
            nearby_wait_indices=nearby_indices,
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
                pos = _producer_queue_position(p_node, subj_graph)
                failures.append(WaitInsufficientFailure(
                    producer=p_label,
                    consumer=c_label,
                    iter_delta=iter_delta,
                    wait_idx=_node_position(insufficient).vmfma_index,
                    counter_kind=expected_counter,
                    counter_value=cv if cv is not None else 0,
                    queue_depth_at_wait=depth,
                    producer_position=pos,
                ))
                wait_failure_emitted = True
            else:
                # waits exist on the right counter but none drains the producer.
                # Treat as MissingWait — every wait fired before the producer
                # entered the queue (or the producer is positioned after every
                # wait we found).
                failures.append(MissingWaitFailure(
                    producer=p_label,
                    consumer=c_label,
                    iter_delta=iter_delta,
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
                    producer=p_label,
                    consumer=c_label,
                    iter_delta=iter_delta,
                    role=role,
                    wait_idx=_node_position(last_drain).vmfma_index,
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
            producer=cms_node_label(p_node, _body_for_node(subj_graph, p_node)),
            consumer=cms_node_label(c_node, _body_for_node(subj_graph, c_node)),
            iter_delta=_cms_iter_delta(p_node, c_node),
            counter_kind="unknown",
        )]
    return failures


@dataclass
class ValidatorInstruction(ABC):
    """Abstract base for all validator instructions."""
    name: str
    issued_at: SchedulePosition
    # The minimum number of quad-cycles that this instruction takes to issue.
    min_issue_quad_cycles_base: ClassVar[int] = 1
    # Reference to the rocisa instruction object this was created from.
    # None when constructed from mock/test data without real instructions.
    # init=False: set via _insert after construction to avoid dataclass
    # ordering issues (non-default following default in subclasses).
    rocisa_inst: object = field(default=None, init=False, repr=False, compare=False)

    @property
    def category(self) -> str:
        """Alias for `name`. Lets the typed-Failure canonical formatters in
        ScheduleCapture.py — which were written against GraphNode (which has
        both `category` and `name`) — accept ValidatorInstruction objects
        without crashing. ValidatorInstruction's `name` already carries the
        per-tensor / per-iter tag (e.g. 'LRA0', 'PackB1'), matching what
        GraphNode.category holds."""
        return self.name

    @abstractmethod
    def validate(self) -> Optional[str]:
        ...

    def done_idx(self) -> SchedulePosition:
        """Position after which this instruction is done for scheduling purposes.

        Default: instruction is done at its issue position.
        Override in subclasses where completion depends on an SWaitCnt (LocalRead, GlobalRead).
        """
        return self.issued_at

    def min_issue_quad_cycles(self) -> int:
        return self.min_issue_quad_cycles_base

@dataclass
class LocalRead(ValidatorInstruction):
    # The index in the list of Local Read instructions provided by a CMS schedule.
    # Needed to properly calculate must_start_after for Packs.
    issue_index: int
    # `guaranteed_by` is set by ``apply_swaits`` to the SWaitCnt position
    # that drains this LR; consumed by Pack-side dependency derivation
    # (``derive_pack_must_start_after``) via ``done_idx()``.
    guaranteed_by: SchedulePosition = field(default_factory=lambda: POSITION_INF)

    def done_idx(self) -> SchedulePosition:
        return self.guaranteed_by

    def validate(self) -> Optional[str]:
        """No-op. LR-data-ready (LR -> MFMA dscnt drain) lives graph-side
        via ``validate_edge_wait_coverage`` over LR -> MFMA RAW edges.
        The ``ValidatorInstruction.validate`` abstract method still
        requires an override on every subclass; this no-op preserves the
        contract without re-introducing the deleted structural rule.
        """
        return None

@dataclass
class MFMA(ValidatorInstruction):
    def validate(self) -> Optional[str]:
        return None

@dataclass
class Pack(ValidatorInstruction):
    """BF16 pack instructions (v_perm). Base class for all pack types.

    The structural-side ordering rule (`Pack.validate`,
    `Pack.must_start_after`, `Pack.needed_by`, plus all `_hook_up_packs_*`
    plumbing) was removed. Pack-related ordering invariants (LR -> Pack
    RAW dscnt drain, Pack -> MFMA RAW order, Pack -> Pack WAR ordering)
    are now enforced graph-side by `validate_edge_wait_coverage` and
    `compare_graphs` over real register dataflow edges produced by
    `_GenericALURule` (ScheduleCapture.py). See migrated coverage in
    `Tensile/Tests/unit/test_validate_pack_graph.py`.
    """
    # The index in the list of Pack instructions provided by a CMS schedule.
    # Set at construction time (the `_insert` call paths in `Timeline`).
    # Retained as a construction-time tag; the producing call site assigns
    # it unconditionally.
    issue_index: int
    # Which tile/group this pack belongs to, computed at construction time.
    # Only meaningful for TF32 subclasses (CVTPack, MiddlePack, MFMAPack); None for BF16 packs.
    group_index: Optional[int] = None

    def validate(self) -> Optional[str]:
        """No-op. Pack-ordering invariants live graph-side; the
        ValidatorInstruction abstract method still requires an override on
        every subclass, so this preserves the contract.
        """
        return None

@dataclass
class CVTPack(Pack):
    """TF32 CVT0/CVT1 packs (v_cvt_pk_bf16_f32). Type marker for isinstance dispatch."""
    pass

@dataclass
class MiddlePack(Pack):
    """Middle-16 packs in TF32 groups of 24.

    The structural-side `MiddlePack.validate`, `pair_consumer`, and
    `next_scheduled_middle_16` helpers were removed. The pair-interleaving
    invariant (`OverriddenInputFailure` on a non-pair-consumer between
    pair-leader and pair-consumer) is now enforced graph-side via
    `validate_middle_pack_pair_interleaving`; see
    `test_validate_pack_graph.py::TestMiddlePackPairInterleavingGraph`.
    """
    pass

@dataclass
class SwapPack(Pack):
    """VSwapB32 instructions that transpose registers after wider local reads (VW > 1).

    Generated by transposeLRVregs() in LocalRead.py. Count per pack group: 4 * (vw - 1).
    Always appear at the beginning of a pack sequence before CVT0/MFMA/CVT1 groups.
    """
    pass

@dataclass
class MFMAPack(Pack, MFMA):
    """A v_mfma_f32_4x4x4_16b_bf16 instruction used in TF32 4x4 emulation pack groups.

    Identified from idMap via resolve_pack_type() (isinstance MFMAInstruction).
    They are real MFMA instructions but participate in the pack dependency
    chain (CVT0 -> MFMAPack -> CVT1).

    Inherits from both Pack and MFMA:
    - isinstance(x, Pack) is True — works with pack gathering, filtering, type hints
    - isinstance(x, MFMA) is True — captures "this IS an MFMA" semantics
    """


@dataclass
class GlobalRead(ValidatorInstruction):
    swap_global_read_order: bool
    needed_by: ValidatorInstruction = field(default_factory=lambda: MFMA(name="MFMA", issued_at=POSITION_INF))
    guaranteed_by: SchedulePosition = field(default_factory=lambda: POSITION_INF)
    barriered_at: list[SchedulePosition] = field(default_factory=list)

    def done_idx(self) -> SchedulePosition:
        return self.guaranteed_by

    def validate(self):
        """Stack 1.3: returns typed Failure or None (legacy str also accepted
        for the residual defensive-fallback branches)."""
        # Check needed_by constraint (GR must finish before LR1/3)
        needed_by_error = self._validate_needed_by()
        if needed_by_error:
            return needed_by_error

        return None

    def _validate_needed_by(self):
        """Validate: GR -> SWait -> SBarrier -> LR1. Returns Failure or None."""
        # If needed_by is at inf, the constraint is not active (e.g. no LR1s).
        if self.needed_by.issued_at == POSITION_INF:
            return None

        if self.issued_at < self.guaranteed_by < self.needed_by.issued_at:
            if any(self.guaranteed_by < barriered_at < self.needed_by.issued_at for barriered_at in self.barriered_at):
                    return None

        issued_at = self.issued_at.vmfma_index
        needed_by = self.needed_by.issued_at.vmfma_index

        name = self._name()

        # Eager source-aware labels. ValidatorInstruction has no body capture
        # context (no tagged_inst, no body_label), so cms_node_label falls
        # through to bare-category primary — matches the pre-g4w wording for
        # this path (which previously raised under strict mode anyway).
        producer_label = cms_node_label(self, None)
        consumer_label = cms_node_label(self.needed_by, None)
        iter_delta = _cms_iter_delta(self, self.needed_by)

        # 1. No SWait
        if self.guaranteed_by == POSITION_INF:
            return MissingWaitFailure(
                producer=producer_label, consumer=consumer_label,
                iter_delta=iter_delta, counter_kind="vlcnt",
            )

        # NOTE: Must do it after the check above to guard against infinity.
        guaranteed_by = self.guaranteed_by.vmfma_index

        # 2. No Barrier
        if len(self.barriered_at) == 0:
            return MissingBarrierFailure(
                producer=producer_label, consumer=consumer_label,
                iter_delta=iter_delta, role="needed_by",
            )

        # 3. Guaranteed after needed — semantically equivalent to no wait from the
        # consumer's perspective, so surface as MissingWaitFailure.
        if self.guaranteed_by > self.needed_by.issued_at:
            return MissingWaitFailure(
                producer=producer_label, consumer=consumer_label,
                iter_delta=iter_delta, counter_kind="vlcnt",
            )

        # 4. No Barrier between SWait and LR1
        if not any(self.guaranteed_by < barriered_at < self.needed_by.issued_at for barriered_at in self.barriered_at):
            return MissingBarrierFailure(
                producer=producer_label, consumer=consumer_label,
                iter_delta=iter_delta, role="needed_by",
            )

        # Defensive fallback — string return; should not fire on valid logic.
        return (f"{name} @ idx={issued_at} is not valid. issued @ idx={issued_at}, "
                f"guaranteed @ idx={guaranteed_by}, "
                f"barriered @ idx={[b.vmfma_index for b in self.barriered_at]}, "
                f"needed @ idx={needed_by} is not valid.")

    def _name(self) -> str:
        name = self.name
        if not self.swap_global_read_order:
            return name

        if name.startswith("GRA"):
            return name + " (Swapped, loading B)"
        elif name.startswith("GRB"):
            return name + " (Swapped, loading A)"
        else:
            raise ValueError(f"Unexpected global read name: {name}")

@dataclass
class SWait(ValidatorInstruction):
    dscnt: int
    vlcnt: int
    vscnt: int
    comment: str

    def _is_valid(self) -> bool:
        return self.dscnt >= -1 and self.vlcnt >= -1 and self.vscnt >= -1 and self.issued_at.vmfma_index >= -1

    def validate(self):
        """Stack 1 of jaunty-reddy plan: returns typed Failure or None."""
        if self._is_valid():
            return None
        return InvalidCounterValueFailure(
            swait_idx=self.issued_at.vmfma_index,
            dscnt=self.dscnt,
            vlcnt=self.vlcnt,
            vscnt=self.vscnt,
        )

@dataclass
class Barrier(ValidatorInstruction):
    comment: str

    def validate(self) -> Optional[str]:
        # Sentinel range check stays as a defensive guard. No user-facing
        # Failure type — this fires only on malformed test fixtures.
        # Returning a string keeps the legacy boundary working.
        if self.issued_at.vmfma_index < -1:
            return f"Barrier at index {self.issued_at.vmfma_index} is not valid. Must be >= -1."
        return None

@dataclass
class SNop(ValidatorInstruction):
    wait_state: int

    def min_issue_quad_cycles(self) -> int:
        # Base instruction quad-cycles plus wait_state additional cycles
        return self.min_issue_quad_cycles_base + self.wait_state

    def validate(self) -> Optional[str]:
        return None

@dataclass
class GRInc(ValidatorInstruction):
    """Scalar pointer-increment instructions (GRIncA/GRIncB) that advance the
    global memory address before the next buffer_load."""

    def validate(self) -> Optional[str]:
        return None

ALL_INSTRUCTION_NAMES = [
    "LRA0", "LRB0", "LRA1", "LRB1", "LRA3", "LRB3",
    "GRA", "GRB",
    "GRIncA", "GRIncB",
    "PackA0", "PackB0", "PackA1", "PackB1", "PackA3", "PackB3",
    "SYNC", "SNOP",
]


# --- Type Resolution for idMap-driven instruction classification ---
# Maps rocisa instruction types to (ValidatorInstruction subclass, assembly label).
# Lookup: exact type match first, then isinstance fallback for subclass chains.
PACK_TYPE_MAP: dict[type, tuple[type, str]] = {
    VPermB32:          (Pack,       "v_perm_b32"),
    VOrB32:            (Pack,       "v_or_b32"),
    VLShiftLeftOrB32:  (Pack,       "v_lshlrev_or_b32"),
    VCvtPkF32toBF16:   (CVTPack,    "v_cvt_pk_bf16_f32"),
    PVCvtBF16toFP32:   (MiddlePack, "p_v_cvt_f32_bf16"),
    VCvtBF16toFP32:    (MiddlePack, "v_cvt_f32_bf16"),
    VSubF32:           (MiddlePack, "v_sub_f32"),
    VDot2CF32BF16:     (MiddlePack, "v_dot2c_f32_bf16"),
    VSwapB32:          (SwapPack,   "v_swap_b32"),
    MFMAInstruction:   (MFMAPack,   "v_mfma_*"),
}

# Maps rocisa instruction types to (is_gr_load: bool).
# Used to distinguish actual GR loads from m0 pointer writes in DTL sequences.
GR_TYPE_MAP: dict[type, bool] = {
    SMovB32:        False,  # m0 pointer setup
    SAddU32:        False,  # m0 pointer increment
    BufferLoadB128: True,   # actual GR load
    BufferLoadB64:  True,
    BufferLoadB32:  True,
}


def resolve_pack_type(rocisa_inst: object) -> tuple[type, str]:
    """Resolve a rocisa instruction to its ValidatorInstruction class and assembly label.

    Args:
        rocisa_inst: A rocisa instruction object from the idMap.

    Returns:
        Tuple of (ValidatorInstruction subclass, assembly label string).

    Raises:
        ValueError: If the instruction type is not recognized.
    """
    inst_type = type(rocisa_inst)
    # Exact match first
    if inst_type in PACK_TYPE_MAP:
        return PACK_TYPE_MAP[inst_type]
    # isinstance fallback for subclass chains (e.g. MFMAInstruction subclasses)
    for base_type, result in PACK_TYPE_MAP.items():
        if isinstance(rocisa_inst, base_type):
            return result
    raise ValueError(f"Unknown pack instruction type: {inst_type.__name__} ({rocisa_inst})")


def is_gr_load(rocisa_inst: object) -> bool:
    """Return True if the rocisa instruction is an actual GR load, False if it's a pointer operation.

    Raises:
        ValueError: If the instruction type is not recognized.
    """
    inst_type = type(rocisa_inst)
    if inst_type in GR_TYPE_MAP:
        return GR_TYPE_MAP[inst_type]
    for base_type, result in GR_TYPE_MAP.items():
        if isinstance(rocisa_inst, base_type):
            return result
    raise ValueError(f"Unknown GR instruction type: {inst_type.__name__} ({rocisa_inst})")


def detect_pack_groups(idmap_items: list) -> list[dict]:
    """Walk the idMap entries for a pack name and identify group boundaries from the type pattern.

    Returns a list of group dicts, each containing:
      - 'group_index': int (or None for ungrouped SwapPacks)
      - 'entries': list of (index, validator_cls, asm_label) tuples

    Group detection rules:
      - SwapPack instructions at the start are ungrouped (group_index=None).
      - If no CVTPack/MiddlePack/MFMAPack types appear, all entries are plain Pack (BF16, no grouping needed).
      - Otherwise, determine the group size from the type pattern (presence of MiddlePack → 24, MFMAPack → 10)
        and split the non-swap entries into fixed-size groups.
    """
    if not idmap_items:
        return []

    entries = []
    for idx, inst in enumerate(idmap_items):
        cls, label = resolve_pack_type(inst)
        entries.append((idx, cls, label))

    # Check if there are any TF32-related types
    has_middle = any(cls is MiddlePack for _, cls, _ in entries)
    has_mfma_pack = any(cls is MFMAPack for _, cls, _ in entries)
    has_cvt = any(cls is CVTPack for _, cls, _ in entries)

    if not (has_middle or has_mfma_pack or has_cvt):
        # BF16: all plain Pack, single group
        return [{"group_index": 0, "entries": entries}]

    groups = []
    # Separate leading SwapPacks
    n_swaps = 0
    for idx, cls, label in entries:
        if cls is SwapPack:
            n_swaps += 1
        else:
            break

    if n_swaps > 0:
        groups.append({"group_index": None, "entries": entries[:n_swaps]})

    remaining = entries[n_swaps:]
    if not remaining:
        return groups

    # Determine group size from the type pattern
    if has_middle:
        group_size = PACK_GROUP_SIZE_TF32       # 24: CVT0×4 + Middle×16 + CVT1×4
    elif has_mfma_pack:
        group_size = PACK_GROUP_SIZE_TF32_4X4   # 10: CVT0×4 + MFMAPack×2 + CVT1×4
    else:
        # CVTPack only (no middle or mfma) — shouldn't happen in practice, but handle gracefully
        group_size = len(remaining)

    # Split remaining entries into fixed-size groups
    for i in range(0, len(remaining), group_size):
        group_entries = remaining[i:i + group_size]
        group_idx = i // group_size
        groups.append({"group_index": group_idx, "entries": group_entries})

    return groups


def create_unified_timeline(
    schedule_info: 'ScheduleInfo',
    kernel: 'Solution',
    code_path: int,
    id_map: dict,
    mfma_code: list,
) -> 'Timeline':
    """Create a single Timeline with all instruction types.

    Args:
        schedule_info:  The schedule to validate.
        kernel:         Kernel configuration dict.
        code_path:      Which code path to build the timeline for.
        id_map:         Maps schedule keys to lists of rocisa instruction objects.
        mfma_code:      Flat list of MFMA rocisa instruction objects in execution order
                        (already reordered by mfmaReorder).
    """
    available_names = set(schedule_info.optSchedule.keys())
    names_to_add = [n for n in ALL_INSTRUCTION_NAMES if n in available_names]
    return Timeline(names_to_add, code_path, schedule_info, kernel, id_map=id_map, mfma_code=mfma_code)


class Timeline:
    def __init__(self, instruction_names_to_add: list[str], code_path: int, schedule_info: 'ScheduleInfo', kernel: 'Solution',
                 id_map: dict, mfma_code: list):
        """Build a Timeline restricted to the categories in
        ``instruction_names_to_add``. Only one ``code_path`` is materialized
        per Timeline — multi-codepath validation builds N Timelines so
        per-codepath schedule disagreements are isolated.

        Asserts the per-iteration suffix scheme (LRA0/LRA1/... or 0..3 for
        ForceUnrollSubIter) matches what DepthU + matrixInstK derive. A
        mismatched schedule key here means upstream drift between the CMS
        author's iteration count and the kernel's, which would otherwise
        surface as a confusing IndexError deep in the validators."""
        
        available_keys = schedule_info.optSchedule.keys()
        has_lr1s = "LRA1" in available_keys or "LRB1" in available_keys
        has_lr3s = "LRA3" in available_keys or "LRB3" in available_keys
        assert not (has_lr1s and has_lr3s), "Can't mix LR1s and LR3s."

        # Validate that sub-iteration suffixes are consistent with the kernel configuration.
        # The valid suffixes depend on how numLoopIter is determined:
        # - ForceUnrollSubIter=True: numLoopIter = numSubTiles² = 4 (KernelWriter.py:4592)
        # - DepthU == matrixInstK (n_sub_iters == 1): split to numLoopIter = 2 (CustomSchedule.py:317)
        # - DepthU > matrixInstK: numLoopIter = DepthU / matrixInstK
        if "DepthU" in kernel and "MatrixInstruction" in kernel:
            force_unroll = kernel.get("ForceUnrollSubIter", False)
            if force_unroll:
                valid_suffixes = {0, 1, 2, 3}
            else:
                n_sub_iters = kernel["DepthU"] // kernel["MatrixInstruction"][2]
                if n_sub_iters == 1:
                    valid_suffixes = {0, 1}
                else:
                    valid_suffixes = set(range(n_sub_iters))
            for key in available_keys:
                for prefix in ("LRA", "LRB", "PackA", "PackB"):
                    if key.startswith(prefix):
                        suffix_str = key[len(prefix):]
                        if suffix_str.isdigit():
                            suffix = int(suffix_str)
                            assert suffix in valid_suffixes, (
                                f"Schedule key '{key}' has sub-iteration index {suffix}, "
                                f"but with DepthU={kernel['DepthU']} and matrixInstK={kernel['MatrixInstruction'][2]}, "
                                f"valid sub-iteration indices are {sorted(valid_suffixes)}."
                            )
                        break

        self.num_vmfma = schedule_info.numMfma
        self.id_map = id_map
        self.mfma_code = mfma_code
        self.vlcnt_shift = defaultdict(int)
        self.vlcnt_shift[NO_GLOBAL_LOAD_LOOP] = schedule_info.nglshift
        self.vlcnt_shift[NO_LOCAL_LOAD_LOOP] = schedule_info.nllshift
        self.nll_zero_dscnt = schedule_info.nllZeroDscnt

        self.loops = [MAIN_LOOP_PREV, MAIN_LOOP, NO_GLOBAL_LOAD_LOOP, NO_LOCAL_LOAD_LOOP]
        # NOTE: num_vmfma + 1 to account for special idx=-1.
        #       idx=-1 is special case that occurs BEFORE the first VMFMA but AFTER the last VMFMA.
        #       Instructions at idx=-1 happen after all instructions at idx=num_vmfma-1 and BEFORE all instructions (including the VMFMA) at idx=0.
        self._instructions_at_index: dict[str, list[list[ValidatorInstruction]]] = {loop: [[] for _ in range(self.num_vmfma+1)] for loop in self.loops}
        
        # Linear timelines for each loop.
        self._timelines: dict[str, list[ValidatorInstruction]] = {loop: [] for loop in self.loops}
        # One linear timeline that spans all loops.
        self.combined_timeline: list[ValidatorInstruction] = []

        # Lookup for all instructions in a given loop for a given name.
        # First key is the loop name, second key is the instruction name (e.g. "GRA").
        # Value is a list of tuples of (index, instruction) for the given name in the given loop.
        # Index is the index of the instruction in the loop, index in [0, len(self._timelines[loop])-1]
        self._instructions_for_name: dict[str, dict[str, list[tuple[int, ValidatorInstruction]]]] = {loop: defaultdict(list) for loop in self.loops}
        # Same as above, except for all instructions across all loops.
        # Only index by instruction name.
        # Index is the index of the instruction in the combined timeline. index in [0, len(self.combined_timeline)-1]
        self._instructions_for_name_combined: dict[str, list[tuple[int, ValidatorInstruction]]] = defaultdict(list)

        # Track which validation passes have already been applied to this timeline to avoid applying them multiple times.
        self._applied_passes: set[Callable[['Timeline', 'ValidationContext'], None]] = set()

        # Populate the timeline with instructions
        self._populate_instructions(instruction_names_to_add, code_path, schedule_info, kernel)
        self._linearize_timeline()
    
    def _populate_instructions(self, instruction_names_to_add: list[str], code_path: int, schedule_info: 'ScheduleInfo', kernel: 'Solution') -> None:
        assert kernel["DirectToLds"], "Only DirectToLds cases are supported by validator."
        assert kernel.get("LocalSplitU", 1) == 1, "Only LocalSplitU=1 cases are supported by validator."

        swap_global_read_order = kernel["SwapGlobalReadOrder"]
        is_tf32_emulation = kernel.get("UseF32XEmulation", False)
        is_4x4mfma_tf32 = kernel.get("UseMFMAF32XEmulation", False)

        # Explicitly add MFMAs to timeline.
        # Do at the top here so they are the first ones scheduled at each vmfma index.
        for i_vmfma in range(self.num_vmfma):
            # mfmaReorder[new_pos] = original_pos. i_vmfma is the new (execution) position.
            vmfma_slot = i_vmfma
            if schedule_info.mfmaReorder:
                vmfma_slot = schedule_info.mfmaReorder[i_vmfma]

            mfma_kwargs = {"name": "MFMA"}
            # mfma_code is indexed by new position (already reordered in CustomSchedule.py)
            if self.mfma_code and i_vmfma < len(self.mfma_code):
                mfma_kwargs["rocisa_inst"] = self.mfma_code[i_vmfma]
            self._insert(vmfma_slot, MFMA, mfma_kwargs, kernel)

        def _get_rocisa(name: str, idx: int) -> object:
            """Look up the rocisa instruction from id_map, or None if unavailable."""
            if self.id_map and name in self.id_map:
                items = self.id_map[name]
                if idx < len(items):
                    return items[idx]
            return None

        # NOTE: Relative ordering of instructions must be preserved.
        #       Order dictates the order in which instructions are scheduled if they are scheduled at the same vmfmaindex.
        for name in schedule_info.optSchedule.keys():
            if name not in instruction_names_to_add:
                continue

            if name == "SYNC":
                for idx_sync, (idx_vmfma, sync) in enumerate(zip(schedule_get(name, code_path, schedule_info), schedule_info.syncCode)):
                    assert idx_vmfma >= -1, f"Code path {code_path}: SWaitCnt at index {idx_sync} is not valid. Must be >= -1."
                    ri = _get_rocisa("SYNC", idx_sync)
                    if isinstance(sync, SWaitCnt):
                        self._insert(idx_vmfma, SWait, {"name": "SWaitCnt", "dscnt": sync.dscnt, "vlcnt": sync.vlcnt, "vscnt": sync.vscnt, "comment": sync.comment, "rocisa_inst": ri}, kernel)
                    elif isinstance(sync, SBarrier):
                        self._insert(idx_vmfma, Barrier, {"name": "SBarrier", "comment": sync.comment, "rocisa_inst": ri}, kernel)
                    else:
                        raise ValueError(f"Unexpected sync instruction type: {type(sync)}")
            elif name == "SNOP":
                for idx_snop, (idx_vmfma, snop) in enumerate(zip(schedule_get(name, code_path, schedule_info), schedule_info.snopCode)):
                    assert idx_vmfma >= -1, f"Code path {code_path}: SNop at index {idx_snop} is not valid. Must be >= -1."
                    # The waitState is stored as the first parameter in the rocisa SNop instruction
                    wait_state = snop.getParams()[0]
                    self._insert(idx_vmfma, SNop, {"name": "SNop", "wait_state": wait_state, "rocisa_inst": _get_rocisa("SNOP", idx_snop)}, kernel)
            elif name.startswith("LRA") or name.startswith("LRB"):
                for idx_LR, idx_vmfma in enumerate(schedule_get(name, code_path, schedule_info)):
                    assert idx_vmfma >= -1, f"Code path {code_path}: LocalRead {name} at index {idx_LR} is not valid. Must be >= -1."

                    # TODO: For ForceUnrollSubIter, need to account for register reuse and the fact that the LR0/LR1/LR3s must start after a certain point in the iteration.
                    self._insert(idx_vmfma, LocalRead, {"name": name, "issue_index": idx_LR, "rocisa_inst": _get_rocisa(name, idx_LR)}, kernel)
            elif name.startswith("GRInc"):
                grincs = schedule_get(name, code_path, schedule_info)
                for idx_grinc, idx_vmfma in enumerate(grincs):
                    assert idx_vmfma >= -1, f"Code path {code_path}: GRInc {name} at index {idx_grinc} is not valid. Must be >= -1."
                    self._insert(idx_vmfma, GRInc, {"name": name, "rocisa_inst": _get_rocisa(name, idx_grinc)}, kernel)
            elif name.startswith("GRA") or name.startswith("GRB"):
                global_reads = schedule_get(name, code_path, schedule_info)
                idmap_items = self.id_map[name]
                if kernel["DirectToLds"]:
                    assert len(global_reads) % 2 == 0, f"Code path {code_path}: {name} has an odd number of indices. Must be even if DirectToLds is True."
                for idx_GR, idx_vmfma in enumerate(global_reads):
                    assert idx_vmfma >= -1, f"Code path {code_path}: GlobalRead {name} at index {idx_GR} is not valid. Must be >= -1."
                    ri = idmap_items[idx_GR] if idx_GR < len(idmap_items) else None
                    if ri is not None and not is_gr_load(ri):
                        continue  # pointer setup (SMovB32/SAddU32), not an actual load
                    self._insert(idx_vmfma, GlobalRead, {"name": name, "swap_global_read_order": swap_global_read_order, "rocisa_inst": ri}, kernel)
            elif name.startswith("Pack"):
                packs = schedule_get(name, code_path, schedule_info)
                idmap_items = self.id_map[name]
                groups = detect_pack_groups(idmap_items)
                idx_to_cls: dict[int, tuple[type, Optional[int]]] = {}
                for group in groups:
                    gidx = group["group_index"]
                    for entry_idx, cls, label in group["entries"]:
                        idx_to_cls[entry_idx] = (cls, gidx)

                for idx_pack, idx_vmfma in enumerate(packs):
                    assert idx_vmfma >= -1, f"Code path {code_path}: Pack {name} at index {idx_pack} is not valid. Must be >= -1."
                    ri = idmap_items[idx_pack] if idx_pack < len(idmap_items) else None
                    pack_cls, group_idx = idx_to_cls.get(idx_pack, (Pack, None))
                    pack_kwargs = {"name": name, "issue_index": idx_pack, "rocisa_inst": ri}
                    if group_idx is not None:
                        pack_kwargs["group_index"] = group_idx
                    self._insert(idx_vmfma, pack_cls, pack_kwargs, kernel)
            else:
                raise NotImplementedError(f"Instruction {name} not implemented")

    def _insert(self, vmfma_index: int, cls: type[ValidatorInstruction], kwargs: dict, kernel: 'Solution') -> None:
        """
        Construct and add an instruction to the timeline at a given VMFMA index.
        A fresh instance is created for each applicable loop directly from
        *cls* and *kwargs*, so no copying is needed.

        Args:
            vmfma_index:  The VMFMA slot to place the instruction at.
            cls:          The instruction class to instantiate (e.g. LocalRead, MFMA).
            kwargs:       Constructor keyword arguments **excluding** ``issued_at``
                          (which is set per-loop by this method).
                          ``rocisa_inst`` is extracted and set post-construction
                          (it uses init=False on the dataclass).
            kernel:       The kernel configuration dict.
        """
        # Extract rocisa_inst before passing kwargs to constructor (init=False field).
        rocisa_inst = kwargs.pop("rocisa_inst", None)

        for loop in self.loops:
            if self._should_add(cls, kwargs.get("name", ""), loop, kernel):
                loop_index = self.loops.index(loop)
                sub_index = len(self._instructions_at_index[loop][vmfma_index + 1])
                kwargs["issued_at"] = SchedulePosition(loop_index=loop_index, vmfma_index=vmfma_index, sub_index=sub_index)

                _instruction = cls(**kwargs)
                _instruction.rocisa_inst = rocisa_inst

                # Adjust for NLL/NGL shifts.
                if isinstance(_instruction, SWait):
                    if _instruction.vlcnt != -1:
                        vlcnt = max(0, _instruction.vlcnt - self.vlcnt_shift[loop])
                        _instruction.vlcnt = vlcnt
                    if _instruction.dscnt != -1 and self.nll_zero_dscnt \
                       and loop in [NO_LOCAL_LOAD_LOOP]:
                        _instruction.dscnt = 0

                self._instructions_at_index[loop][vmfma_index+1].append(_instruction)

    @staticmethod
    def _should_add(cls: type[ValidatorInstruction], name: str, loop: str, kernel: 'Solution') -> bool:
        """Per-loop instruction filter: GR/GRInc only fire in main loops;
        only LR0 / Pack0 reach NLL (the rest of LR/Pack live in the
        main-loop bodies). Encodes the CMS pipeline-stage contract."""
        if issubclass(cls, GlobalRead):
            # No GRs issued in NGL or NLL
            return loop == MAIN_LOOP or loop == MAIN_LOOP_PREV
        elif issubclass(cls, GRInc):
            return loop == MAIN_LOOP or loop == MAIN_LOOP_PREV
        elif issubclass(cls, LocalRead):
            # Only LR0s are issued in the NLL
            if loop == NO_LOCAL_LOAD_LOOP:
                return name == "LRA0" or name == "LRB0"
            return True
        elif issubclass(cls, Pack):
            if kernel.get("UsePLRPack", False):
                # Packs1/3s correspond to the LR1/3s of this iteration.
                if loop == NO_LOCAL_LOAD_LOOP:
                    return name == "PackA0" or name == "PackB0"
            return True
        else:
            return True
   
    def __len__(self):
        return len(self._timelines)

    def __getitem__(self, index: int) -> ValidatorInstruction:
        return self._timelines[index]

    def get_instruction_names(self) -> list[str]:
        return list(self._instructions_for_name_combined.keys())

    def get_instructions(self, name: str, loop: str) -> list[tuple[int, ValidatorInstruction]]:
        return self._instructions_for_name[loop][name]

    def get_instructions_combined(self, name: str) -> list[tuple[int, ValidatorInstruction]]:
        return self._instructions_for_name_combined[name]

    def get_instructions_at(self, index: int, loop: str) -> list[ValidatorInstruction]:
        return self._instructions_at_index[loop][index+1]

    def _linearize_timeline(self) -> None:
        """Materialize the per-loop and combined linear timelines plus the
        per-name lookup tables that ``get_instructions*`` indexes into."""
        self.combined_timeline.clear()
        self._instructions_for_name_combined.clear()
        i_combined = 0
        for loop_name, loop_instructions in self._instructions_at_index.items():
            i_loop = 0
            self._timelines[loop_name].clear()
            self._instructions_for_name[loop_name].clear()

            for instructions in loop_instructions:
                for instruction in instructions:
                    self._timelines[loop_name].append(instruction)
                    self._instructions_for_name[loop_name][instruction.name].append((i_loop, instruction))
                    self._instructions_for_name_combined[instruction.name].append((i_combined, instruction))
                    i_loop += 1
                    i_combined += 1
            
            self.combined_timeline.extend(self._timelines[loop_name])


def _compute_swap_register_pairs(vw: int, total_regs: int) -> list[tuple[int, int]]:
    """Compute the (src_reg, dst_reg) pairs for each VSwapB32 instruction, in issue order.

    Replicates the iteration logic of transposeLRVregs() in LocalRead.py.
    Uses the same conversion tables from getTransposeIndex() (LocalRead.py:319-320).

    The conversion tables map each register index within a block of size
    MIInputPerThread(8) * VW to its transposed position. transposeLRVregs
    iterates indices 1..totalRegs-2 (skipping first and last), and for each:
      - Looks up the target position via the conversion table
      - If neither position has been visited and they differ, emits a VSwapB32
      - Marks both positions as visited

    Args:
        vw: The vector width (lrvwTile). Must be 2 or 4.
        total_regs: Total number of registers being transposed within one block
                    (VGPRS_PER_CONVERSION_GROUP * vw).

    Returns:
        List of (src_reg, dst_reg) tuples, one per swap, in the order they are issued.
    """
    if vw <= 1:
        return []
    _CONV_TABLE = {
        2: [0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15],
        4: [0, 8, 16, 24, 4, 12, 20, 28, 1, 9, 17, 25, 5, 13, 21, 29,
            2, 10, 18, 26, 6, 14, 22, 30, 3, 11, 19, 27, 7, 15, 23, 31],
    }
    conv = _CONV_TABLE[vw]
    block_size = 8 * vw  # MIInputPerThread * lrvwTile
    start, last = 0, total_regs - 1
    done = [start, last]
    pairs = []
    for idx in range(start + 1, last):
        block_idx = idx // block_size
        new_idx = conv[idx % block_size] + block_idx * block_size
        if idx in done or idx == new_idx:
            done.append(idx)
            continue
        pairs.append((idx, new_idx))
        done.append(idx)
        done.append(new_idx)
    return pairs


def _failure_to_string(result: object) -> Optional[str]:
    """Boundary helper: a rule's validate() may return either a legacy
    string OR a typed Failure. Normalize to string for the existing
    isValid contract.

    Stack 1 of plans/then-let-s-work-on-jaunty-reddy.md migrates rules
    one at a time. Until every rule emits Failures, this helper supports
    both shapes so the boundary stays stable.

    Failure.format requires a capture; structural rules don't have one
    in scope, so pass an empty LoopBodyCapture. Formatters that read
    capture for [N] index lookup will fall through to bare category for
    these legacy rule paths — an honest acknowledgement that the
    structural-rule emitter has no per-body context.
    """
    if result is None:
        return None
    if isinstance(result, str):
        return result
    # Typed Failure (from Tensile.Components.ScheduleCapture).
    from Tensile.Components.ScheduleCapture import LoopBodyCapture
    return result.format(LoopBodyCapture(instructions=[]))


def validate_timeline(timeline: Timeline) -> Optional[str]:
    """Iterate every instruction in stream order; the first one whose
    ``validate()`` returns non-None ends the walk.

    Side effect: stashes the typed Failure (when one is returned) on
    ``timeline._last_failure`` so test infrastructure can assert on type +
    fields without parsing message text. Production callers consume only
    the returned string. Rules still on the legacy str path leave
    ``_last_failure`` None."""
    timeline._last_failure = None
    for loop in timeline.loops:
        for instruction in timeline._timelines[loop]:
            result = instruction.validate()
            message = _failure_to_string(result)
            if message is not None:
                # Surface the typed Failure to test introspection. Rules
                # that still return raw strings during migration leave
                # _last_failure as None.
                timeline._last_failure = result if not isinstance(result, str) else None
                if loop in [NO_GLOBAL_LOAD_LOOP, NO_LOCAL_LOAD_LOOP]:
                    message = f"Loop {loop}: {message}"
                return message
    return None


def schedule_get(name: str, code_path: int, schedule_info: 'ScheduleInfo') -> list[list[int]]:
    """When the schedule has only one code path, return it (broadcast for
    all callers regardless of ``code_path``); otherwise return the slice
    for ``code_path``. Lets multi-codepath rules iterate ``code_path`` in
    a loop without special-casing single-codepath schedules."""
    assert code_path >= 0, f"Code path {code_path} is not valid. Must be >= 0."
    schedules = schedule_info.optSchedule[name]
    return schedules[0] if len(schedules) == 1 else schedules[code_path]


@dataclass
class ValidationContext:
    """Typed context for CMS validation — replaces the raw context dict and ValidatorPassContext."""
    kernel: dict
    id_map: dict
    mfma_code: list
    mfma_reorder: list[int] = field(default_factory=list)
    # Phase 6 of plans/then-let-s-work-on-jaunty-reddy.md: schedule captures
    # for default-vs-CMS comparison rules. Both fields are optional and
    # populated by customMainLoopSchedule when capture is enabled. None means
    # the comparison rule is skipped (existing behavior preserved).
    default_capture: object = None
    cms_capture: object = None
    # Side-channel for typed Failure objects emitted by structural rules
    # (e.g. verify_scc_overlap). These rules return (False, str) for backward
    # compatibility but stash the Failure here so test infrastructure can
    # introspect typed fields. Reset to None at the top of each rule iteration
    # in isValid to prevent cross-rule leakage.
    _last_failure: Optional[object] = None

    @property
    def swap_global_read_order(self) -> bool:
        return self.kernel.get("SwapGlobalReadOrder", False)

    @property
    def direct_to_lds(self) -> bool:
        return self.kernel.get("DirectToLds", False)

    @property
    def use_f32x_emulation(self) -> bool:
        return self.kernel.get("UseF32XEmulation", False)

    @property
    def use_4x4mfma_tf32(self) -> bool:
        return self.kernel.get("UseMFMAF32XEmulation", False)

    @property
    def use_direct_32x_emulation(self) -> bool:
        return self.kernel.get("UseDirect32XEmulation", False)

    @property
    def use_plr_pack(self) -> bool:
        return self.kernel.get("UsePLRPack", False)

    @property
    def force_unroll_sub_iter(self) -> bool:
        return self.kernel.get("ForceUnrollSubIter", False)

    @property
    def n_tiles_a(self) -> int:
        return self.kernel["MIWaveTileA"]

    @property
    def n_tiles_b(self) -> int:
        return self.kernel["MIWaveTileB"]

    @property
    def dtl_plus_lds_buf(self) -> bool:
        return self.kernel.get("DtlPlusLdsBuf", False)

    @property
    def use_shadow_limit(self) -> bool:
        return self.kernel.get("Use64bShadowLimit", True)


def isValid(scheduleInfo: 'ScheduleInfo', context: 'ValidationContext') -> tuple[bool, str]:
    """
    Return True if all the validation rules pass, False otherwise.
    If validation fails, a string containing the reason is returned.

    Note 1: If True is returned, this is not proof that this schedule
    is valid. It may be a false negative.

    Note 2: if False is returned, this is not proof that the schedule
    is invalid. It may be a false positive.

    Rule dispatch uses the concern-based framework: active_concerns()
    determines which ValidationConcern values apply to this kernel,
    and only rules whose concerns intersect are executed.
    """
    kernel = context.kernel

    context.mfma_reorder = scheduleInfo.mfmaReorder or []

    # Determine required concerns from kernel config + ISA catalog.
    required = active_concerns(kernel, context.id_map)

    # Unknown/missing ISA (not in catalog) → run all rules (legacy behavior).
    # This ensures tests and kernels without ISA metadata get full validation.
    if not required:
        required = set(ValidationConcern)

    # Cross-scheduler comparison rule. Only fires when both default and CMS
    # captures are present in the context (i.e. capture was enabled when
    # this kernel was scheduled). Validates the CMS schedule against the
    # default via dataflow-graph equality + per-edge wait-coverage.
    if context.default_capture is not None and context.cms_capture is not None:
        # br4.7 moved build_dataflow_graph; br4.8 moved compare_graphs into
        # this module. validate_edge_wait_coverage still lives in
        # ScheduleCapture (br4.9 territory) — pulled in lazily to avoid the
        # remaining cycle.
        from Tensile.Components.ScheduleCapture import validate_edge_wait_coverage
        # Attach the per-arch quad-cycle profile derived from this kernel's
        # ISA tuple so the four pair-specific gap helpers consult arch-
        # appropriate finish-cycle / settle-window values. Unknown ISAs
        # fall back to the CDNA 4 default for historical compatibility.
        arch_profile = _resolve_arch_profile_for_isa(
            tuple(kernel["ISA"]) if "ISA" in kernel else None
        )
        if context.default_capture.arch_profile is None:
            context.default_capture.arch_profile = arch_profile
        if context.cms_capture.arch_profile is None:
            context.cms_capture.arch_profile = arch_profile
        ref_graph = build_dataflow_graph(context.default_capture)
        subj_graph = build_dataflow_graph(context.cms_capture)
        graph_failures = compare_graphs(
            ref_graph, subj_graph, raise_on_unexplained=False,
        )
        if graph_failures:
            summary = "\n  ".join(
                f.format()
                for f in graph_failures
            )
            return False, (
                f"Dataflow graph comparison failed: "
                f"{len(graph_failures)} edge difference(s):\n  {summary}"
            )
        wait_failures = validate_edge_wait_coverage(
            subj_graph, raise_on_unexplained=False,
        )
        if wait_failures:
            summary = "\n  ".join(
                f.format()
                for f in wait_failures
            )
            return False, (
                f"Wait-coverage validation failed: "
                f"{len(wait_failures)} failure(s):\n  {summary}"
            )

    return True, ""


def findValidPositions(
    scheduleInfo: 'ScheduleInfo',
    context: 'ValidationContext',
    inst_name: str,
    inst_issue_idx: int,
) -> list[tuple[int, int]]:
    """Find all vmfma indices where an instruction can be moved while keeping the schedule valid.

    For each candidate vmfma index in [-1, numMfma-1], mutates the target
    instruction's entry in scheduleInfo.optSchedule in-place, calls isValid(),
    and restores. Collects all positions that pass and merges adjacent indices
    into ranges.

    Args:
        scheduleInfo:   The schedule to query.
        context:        Validation context dict (must contain "kernel", optionally "idMap"/"mfmaCode").
        inst_name:      The schedule key of the instruction to move (e.g. "LRA0", "GRA", "PackA0").
        inst_issue_idx: The index within that instruction's schedule list (0-based).

    Returns:
        A list of (start, end) tuples representing inclusive ranges of valid vmfma indices.
        E.g. [(-1, 2), (5, 8)] means indices -1, 0, 1, 2, 5, 6, 7, 8 are valid.
        An empty list means the instruction cannot be moved anywhere.
    """
    assert inst_name in scheduleInfo.optSchedule, f"Unknown instruction: {inst_name}"

    num_code_paths = len(scheduleInfo.optSchedule[inst_name])

    # Save original values across all code paths.
    original_values = []
    for cp in range(num_code_paths):
        original_values.append(scheduleInfo.optSchedule[inst_name][cp][inst_issue_idx])

    valid_indices = []
    for candidate in range(-1, scheduleInfo.numMfma):
        # Mutate in-place.
        for cp in range(num_code_paths):
            scheduleInfo.optSchedule[inst_name][cp][inst_issue_idx] = candidate

        valid, _ = isValid(scheduleInfo, context)
        if valid:
            valid_indices.append(candidate)

    # Restore original values.
    for cp in range(num_code_paths):
        scheduleInfo.optSchedule[inst_name][cp][inst_issue_idx] = original_values[cp]

    # Merge adjacent indices into ranges.
    ranges = []
    for idx in valid_indices:
        if ranges and idx == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], idx)
        else:
            ranges.append((idx, idx))

    return ranges

