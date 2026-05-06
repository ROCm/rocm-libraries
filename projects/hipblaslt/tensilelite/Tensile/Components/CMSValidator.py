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
from typing import TYPE_CHECKING, ClassVar, Dict, Optional, Tuple, Union

from rocisa.instruction import (
    SWaitCnt, SBarrier,
    MFMAInstruction, SMovB32, SAddU32,
    VPermB32, VOrB32, VLShiftLeftOrB32, VSwapB32,
    VCvtPkF32toBF16, PVCvtBF16toFP32, VCvtBF16toFP32, VSubF32, VDot2CF32BF16,
    BufferLoadB128, BufferLoadB64, BufferLoadB32,
)
from Tensile.Common.Utilities import printWarning
from Tensile.Components.ScheduleCapture import SchedulePosition

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
        from Tensile.Components.ScheduleCapture import (
            build_dataflow_graph, compare_graphs, validate_edge_wait_coverage,
        )
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

