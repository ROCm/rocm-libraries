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

"""
CMS scheduler validator: graph construction, dataflow rules, FIFO simulator,
edge-coverage analysis, timing checks, and Failure formatting.

After br4 (CMS validator consolidation epic), this module is the *downstream*
side of a strict one-way dependency edge: CMSValidator imports from
ScheduleCapture eagerly; ScheduleCapture references CMSValidator symbols only
under `TYPE_CHECKING` (string-typed). No runtime cycle exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from rocisa.instruction import SWaitCnt
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
    WrappedInstruction,
    make_position,
    assign_stream_indices_for_body,
    _byte_keys_for_resource, _resolve_producers,
)
# rocm-libraries-009 (re-scoped): the 13 `_*_CLASS_NAMES` sets and 10
# `_is_*` discriminator predicates that used to live in ScheduleCapture
# have been collapsed into a single class-name -> InstructionCategory
# registry in `Tensile/Components/InstructionCategory.py`. CMSValidator
# call sites compare directly against `InstructionCategory.X` rather than
# routing through per-bucket boolean predicates.
from Tensile.Components.InstructionCategory import (
    InstructionCategory,
    category as _category,
)

if TYPE_CHECKING:
    # Annotations only — kept as strings at runtime by `from __future__ import
    # annotations`. Importing these eagerly would form a hard cycle
    # (ScheduleCapture imports from this module at runtime).
    from Tensile.Components.ScheduleCapture import (
        LoopBodyCapture, TaggedInstruction,
    )


# --- Quad-Cycle Timing (CDNA 4 ISA section 7.6) ---
# Per-arch profiles (`ArchProfile`) defined below resolve from `kernel["ISA"]`.
# The `_quad_cycle_gap_ok` / `_cvt_to_mfma_gap_ok` / `_mfma_pack_to_cvt_gap_ok`
# helpers that consume them are defined later in this same file.

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

    @classmethod
    def for_isa(
        cls, isa: Optional[Tuple[int, int, int]],
    ) -> Optional["ArchProfile"]:
        """Return the ArchProfile for an ISA tuple, or `None` when unknown.

        Behavior:
          - `isa is None`: returns the CDNA 4 default profile (legacy callers
            that haven't plumbed ISA through still get the historical
            CDNA 4 path — we can't tell "no ISA available" apart from the
            historical "didn't bother" case here).
          - `isa` registered in `_ARCH_PROFILES_BY_ISA`: returns that profile.
          - `isa` NOT registered: emits a `printWarning(...)` naming the ISA
            tuple verbatim (grep-able), then returns `None`. There is NO
            silent fallback to CDNA 4 — callers that receive `None` must
            skip timing-related validation for this kernel (cross-graph
            diff, wait coverage, SCC, MiddlePack interleaving still run).

        Every call with an unknown ISA emits the warning — there is no
        process-wide de-dup. A build that schedules N kernels for the same
        uncharacterized arch fires N warnings. Production callers gate the
        warning surface by calling the resolver once per kernel resolution.

        Tracked: `rocm-libraries-zkzw`. The previous behavior silently
        substituted CDNA 4 timing constants for any uncharacterized arch,
        producing wrong timing answers without any signal to the user.
        """
        if isa is None:
            return _DEFAULT_CDNA4_ARCH_PROFILE
        isa_key = tuple(isa)
        profile = _ARCH_PROFILES_BY_ISA.get(isa_key)
        if profile is not None:
            return profile
        printWarning(
            f"CMSValidator: no ArchProfile registered for ISA {isa_key!r}; "
            f"skipping timing-related validation."
        )
        return None

    @classmethod
    def from_carrier(cls, carrier: object) -> Optional["ArchProfile"]:
        """Return the ArchProfile attached to a DataflowGraph or FourPartCapture.

        `carrier` may be:
          - DataflowGraph with `.arch_profile` set,
          - FourPartCapture with `.arch_profile` set,
          - None (degenerate test path).

        Returns:
          - The attached `ArchProfile` if one is set.
          - `None` when `carrier.arch_profile is None` (unknown-ISA case;
            the resolver returned None and the timing-related validation
            is skipped downstream).
        """
        if carrier is None:
            return None
        profile = getattr(carrier, "arch_profile", None)
        if isinstance(profile, cls):
            return profile
        return None

    def min_issue_quad_cycles_for(self, rocisa_inst) -> int:
        """Return the per-instruction quad-cycle issue cost.

        Default issue cost is `default_issue_quad_cycles`; SNop adds
        `wait_state` (read off the rocisa instruction's first parameter).
        Every other instruction shape keeps the base cost.

        With the structural-side simulators removed, this method is the
        canonical per-instruction cost table.
        """
        base = self.default_issue_quad_cycles
        if rocisa_inst is None:
            return base
        if WrappedInstruction(rocisa_inst).is_snop:
            # SNop stores wait_state as the first param of getParams()
            # (matches CMSValidator.py SNop branches: `snop.getParams()[0]`).
            try:
                params = rocisa_inst.getParams()
                if params:
                    return base + int(params[0])
            except Exception:
                pass
            return base
        return base

    def mfma_finish_cycles_for(self, rocisa_inst) -> int:
        """Classify an MFMA-shaped rocisa instruction as standard or 4x4 PackMFMA
        and return the per-arch finish-cycle count.

        The rocisa `MFMAInstruction` C++ class accepts a `variant` list at
        construction (`[M, N, K, blk]`, e.g. `[4, 4, 4, 16]` for the 4x4 PackMFMA
        family) but does NOT expose that field as a readable Python attribute via
        the nanobind binding. The rendered assembly string IS canonical and
        stable — every MFMA family renders as `..._<M>x<N>x<K>_<dtype>...`. We
        discriminate the 4x4 family by parsing for the `_4x4x` substring.
        """
        if rocisa_inst is None:
            return self.standard_mfma_finish_cycles
        try:
            rendered = str(rocisa_inst)
        except Exception:
            return self.standard_mfma_finish_cycles
        if "_4x4x" in rendered:
            return self.mfma_4x4_finish_cycles
        return self.standard_mfma_finish_cycles

    def quad_cycle_gap_ok(
        self,
        producer: "GraphNode",
        consumer: "GraphNode",
        graph: "DataflowGraph",
    ) -> "TimingCheck":
        """Verify that enough quad-cycles separate an MFMA producer from its
        consumer for the producer's result to be visible.

        Returns a `TimingCheck` (result, observed, required). Same-body and
        cross-body share ONE code path that delegates to
        `cumulative_issue_cycles`. The hardware MFMA pipeline does not reset
        at body boundaries — `mfma_free_at` and the type-switch stall carry
        through.
        """
        finish = self.mfma_finish_cycles_for(getattr(producer, "rocisa_inst", None))
        required = finish
        observed = cumulative_issue_cycles(graph, producer, consumer)
        if observed >= required:
            return TimingCheck.passing(observed, required)
        return TimingCheck.failing(observed, required)


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


# `_resolve_arch_profile_for_isa(isa)` is now `ArchProfile.for_isa(isa)`.
# `_resolve_arch_profile(carrier)` is now `ArchProfile.from_carrier(carrier)`.


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

@dataclass
class GraphNode:
    """A node in the unified 4-body dataflow graph.

    identity is the canonical key for cross-graph comparison: position-independent
    (survives CMS reordering) and content-based (same producer in default and CMS
    captures gets the same identity even if its stream position differs).

    position lives in graph-builder space (loop_index spans bodies); the
    underlying TaggedInstruction.slot is preserved on tagged_inst.

    issue_cycles is the per-instruction quad-cycle issue cost: default 1.
    SNop-shaped instructions get `1 + wait_state` (computed by
    `_min_issue_quad_cycles_for`). Populated by `_make_node` so the graph-side
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

    @property
    def canonical_position(self) -> SchedulePosition:
        """SchedulePosition for this node — the canonical ordering key.

        Returns `self.position`; named for explicitness at call sites that
        once went through a free `_node_position(node)` helper. New code can
        read `self.position` directly.
        """
        return self.position

    def subiter(self, num_mfma_per_subiter: int) -> int:
        """Inner-unroll subiteration index for this node.

        "Subiter" = which inner unroll subiteration this node belongs to within
        its body. NOT the outer loop iteration (those are encoded in
        SchedulePosition.loop_index and the body label ml_prev / ml / ngl / nll).

        For non-MFMA categories, parsed from the category trailing digits
        (`PackA0` ⇒ 0). For MFMA, derived from
        `tagged_inst.slot.mfma_index // num_mfma_per_subiter` — sourced from
        the kernel-writer's MFMA slot id on the TaggedInstruction's SlotKey
        (NOT from the bridge-collapsed `SchedulePosition.stream_index`, which
        is a per-body monotonic sort key with no MFMA-slot semantics). When
        `num_mfma_per_subiter` is 0 (test fixtures that don't set it), or
        when `tagged_inst` is absent, MFMA subiter collapses to 0 — the
        OrderInverted gate then degenerates to "fire on any same-body
        stream-position inversion".
        """
        if self.category == "MFMA":
            if not num_mfma_per_subiter:
                return 0
            ti = getattr(self, "tagged_inst", None)
            if ti is None or getattr(ti, "slot", None) is None:
                return 0
            return ti.slot.mfma_index // num_mfma_per_subiter
        return _split_category_iter(self.category)[1]

    @staticmethod
    def is_mfma_pack_producer(producer) -> bool:
        """True for a 4x4 PackMFMA producer.

        PackMFMAs (TF32 4x4 emulation) are syntactically `MFMAInstruction` rocisa
        objects but are categorized as `PackA*` / `PackB*` because the macro
        classifier groups them with the surrounding CVT pack chain. Discrimination:
        `category.startswith("Pack")` AND `rocisa_inst` is an MFMA-shaped class.

        Used by `is_mfma_producer` so PackMFMAs route to the quad-cycle gap
        branch rather than the ALU-immediate-visibility branch — without this
        discriminator the ALU-producer exemption would fire first and PackMFMA
        producers would skip timing checks entirely.

        Static so it accepts both `GraphNode` instances and duck-typed stubs
        (test fixtures synthesize `_StubNode(category, rocisa_inst)` shapes).
        """
        if not getattr(producer, "category", "").startswith("Pack"):
            return False
        inst = getattr(producer, "rocisa_inst", None)
        if inst is None:
            return False
        return WrappedInstruction(inst).is_mfma

    @staticmethod
    def is_mfma_producer(producer) -> bool:
        """True for any producer subject to MFMA quad-cycle finish-time gating.

        Two shapes:
          - `category == "MFMA"` — the standard MFMA path (everything but the
            TF32 4x4 emulation pack chain).
          - PackMFMA — `category.startswith("Pack")` with an MFMA-shaped rocisa
            instance. The dispatch in `_classify_edge_coverage` and
            `diagnose_missing_edge` claims pack-MFMA producers BEFORE the
            ALU-producer exemption fires.

        Static so it accepts duck-typed stubs alongside `GraphNode` instances.
        """
        if getattr(producer, "category", None) == "MFMA":
            return True
        return GraphNode.is_mfma_pack_producer(producer)

    @staticmethod
    def is_cvt_pack_producer(producer) -> bool:
        """True for a CVTPack producer (TF32 v_cvt_pk_bf16_f32 family).

        CVTPacks are categorized `Pack*` (PackA0/PackA1/PackB0/PackB1/PackA3/
        PackB3 depending on the surrounding LR group); discrimination here is
        `category.startswith("Pack")` AND `rocisa_inst` is the
        `VCvtPkF32toBF16` rocisa class. Used by `_classify_edge_coverage` and
        `diagnose_missing_edge` so CVTPack-feeding-MFMA edges are routed to
        `_cvt_to_mfma_gap_ok` BEFORE the ALU-immediate exemption claims them
        — same shape as the PackMFMA carve-out, but with the CVT class set
        in place of the MFMA class set.

        Static so it accepts duck-typed stubs alongside `GraphNode` instances.
        """
        if not getattr(producer, "category", "").startswith("Pack"):
            return False
        inst = getattr(producer, "rocisa_inst", None)
        if inst is None:
            return False
        return WrappedInstruction(inst).is_cvt_pack

    def iter_delta_to(self, other: "GraphNode") -> int:
        """Compute the canonical loop-offset between `other` (consumer) and
        `self` (producer): `other.position.loop_index - self.position.loop_index`.

        Mirrors the legacy `_cms_iter_delta`'s arithmetic. The Failure base's
        `_iter_suffix` consumes the result to pick the appropriate suffix.
        """
        return other.position.loop_index - self.position.loop_index


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

    `intra_operand_byte_offset` is the allocation-invariant tuple of byte
    positions WITHIN the connected operand satisfied by this edge. For an
    LR's 4-byte destination feeding the four bytes of an MFMA's `a` input,
    the tuple is `(0, 1, 2, 3)`. For a single-byte SCC RAW edge the tuple
    is `(0,)`. NOT an absolute physical-register byte-key — that would
    break allocation invariance (`vgpr(8,4)` vs `vgpr(12,4)` for the same
    logical operand would collide on absolute keys but match on intra-
    operand offsets). See rocm-libraries-wx9.3 memo §6.1.

    `src_operand_slot` is the producer's POSITIONAL operand-slot index
    (the position within `inst.getDstParams()` for the writer, or the
    legacy positional emit-order for test-fixture rules). For
    `MFMA(acc, a, b, [acc2])` the writer has `acc` at slot 0. For
    `VSwap(dst, src)` (symmetric R+W) `dst` is at slot 0 and `src` at
    slot 1 — the same operand has the same slot on both sides because
    `getDstParams() == getSrcParams()`.

    `sink_operand_slot` is the consumer's POSITIONAL operand-slot index
    (the position within `inst.getSrcParams()` for the reader). For an
    `MFMA(acc, a, b, acc2)` consuming `a`, sink slot is 0; consuming
    `b`, sink slot is 1.

    Both slot fields are ALLOCATION-INVARIANT by construction: they are
    small integers describing positional structure, not register
    references. They are ORDER-SENSITIVE: when two instructions sharing
    a register are reordered, the shared register lands at different
    operand-slots on each end of the resulting edge — so the within-
    graph reorder shows up as a different edge identity (memo §6.1
    step 1). This is what makes `compare_graphs` simultaneously
    register-rename-equal across graphs and reorder-detecting within
    one graph.
    """
    producer: GraphNode
    consumer: GraphNode
    resource: object                    # RegisterContainer | MemoryRegion (opaque)
    edge_kind: str                      # 'raw_intrawave' | 'lr_to_gr_lds_reuse' | 'gr_to_lr_lds_reuse'
    intra_operand_byte_offset: tuple = ()  # allocation-invariant byte positions
    src_operand_slot: int = 0           # positional operand-slot of the producer's write
    sink_operand_slot: int = 0          # positional operand-slot of the consumer's read


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
    # Per-architecture timing profile copied from FourPartCapture by
    # `build_dataflow_graph`. `None` means "no profile registered for
    # this kernel's ISA; the four pair-specific timing helpers and any
    # other timing-related validation are skipped." Tracked:
    # `rocm-libraries-zkzw`.
    arch_profile: Optional[ArchProfile] = None

    def edge_keys(self):
        """Edge-equality keys for cross-graph diff.

        Per rocm-libraries-wx9.3 phase 3 (memo §6.1 step 1), the edge
        identity tuple is:

            (src_role, src_position, src_operand_slot,
             sink_role, sink_position, sink_operand_slot,
             edge_kind, intra_operand_byte_offset)

        * `src_role` / `sink_role` are the producer/consumer scheduler-role
          tags (`identity[0]` — LR/LW/GR/MFMA/PACK/...). Allocation-invariant.
        * `src_position` / `sink_position` are SchedulePositions. Position-
          sensitive so cross-body register reuse remains distinct; legitimate
          CMS reorders are absorbed by `diagnose_missing_edge`'s pipeline
          branch when both endpoints exist with preserved relative order.
        * `src_operand_slot` / `sink_operand_slot` are positional integer
          indices (0, 1, 2, ...) describing WHICH positional operand of
          the producer was written and WHICH positional operand of the
          consumer was read for this edge. Allocation-invariant by
          construction (small integer, not a register reference) AND
          order-sensitive (it flips when two instructions sharing a
          register are reordered, because the shared register lands in
          different operand positions on each end). This pair is what
          simultaneously yields across-graph allocation invariance and
          within-graph reorder detection — see `DataflowEdge.src_operand_slot`
          for the convention and the worked VSwap-pair example.
        * `intra_operand_byte_offset` is the tuple of byte positions WITHIN
          the connected operand (0..N-1) — allocation-invariant. NOT the
          absolute physical-register byte-key.

        The render-string (`identity[2]`) and the `resource` object stay on
        DataflowEdge for human-facing `Failure` rendering only — they drop
        out of the matching path entirely.
        """
        return {(e.producer.identity[0], e.producer.position, e.src_operand_slot,
                 e.consumer.identity[0], e.consumer.position, e.sink_operand_slot,
                 e.edge_kind, e.intra_operand_byte_offset)
                for e in self.edges}

    @property
    def all_nodes_in_order(self):
        """Yield every node in execution order across all bodies.

        Used by the wait/barrier helpers below to walk cross-body windows
        (e.g. producer in body=ML-1, consumer in body=ML). Per-body
        `_graph_nodes` is already in stream order; bodies are enumerated in
        `_BODY_BUILD_ORDER` which matches SchedulePosition.loop_index ordering,
        so concatenating yields a globally-correct stream.
        """
        for label in _BODY_BUILD_ORDER:
            cap = self.captures.get(label)
            if cap is None or not hasattr(cap, '_graph_nodes'):
                continue
            for node in cap._graph_nodes:
                yield node

    def body_for(self, node: GraphNode) -> "LoopBodyCapture":
        """Look up the LoopBodyCapture that emitted `node`.

        Uses `node.body_label` against `self.captures`. Both arguments are
        required: every caller in CMSValidator passes a constructed
        DataflowGraph (the wa57 cleanup removed the last None-graph caller),
        and every GraphNode was built from a body that lives in
        `self.captures` (see `build_dataflow_graph` Phase 1, where each
        `_make_node` is invoked under `for label in _BODY_BUILD_ORDER` with
        `label in captures`). Subscripts directly so a future capture-
        pipeline regression that produces a node with an unknown body_label
        raises KeyError instead of silently degrading downstream labels.
        """
        return self.captures[node.body_label]

    def queue_depth_at(self, wait_node: GraphNode, producer: GraphNode) -> int:
        """Replay the per-counter FIFO from start of the graph to wait_node.position
        and return the queue depth at the wait's moment for the producer's counter.

        Walks across all bodies in execution order so cross-body queue state
        is preserved (matches build_dataflow_graph's persistent-queue model).
        """
        counter = counter_for(producer)
        depth = 0
        for n in self.all_nodes_in_order:
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

    def producer_queue_position(self, producer: GraphNode) -> int:
        """Return the producer's position in the per-counter FIFO at the moment
        it joined (zero-indexed from the queue head AT THAT MOMENT). Cross-body
        aware via `all_nodes_in_order`."""
        counter = counter_for(producer)
        queue_size = 0
        for n in self.all_nodes_in_order:
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

    def wait_drains_producer(self, wait_node: GraphNode, producer: GraphNode) -> bool:
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
        for n in self.all_nodes_in_order:
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

    def any_drains(self, waits: List[GraphNode], producer: GraphNode) -> bool:
        return any(self.wait_drains_producer(w, producer) for w in waits)


# =============================================================================
# Graph walkers + FIFO simulator + identity helpers (br4.6)
# =============================================================================
#
# These helpers operate on `GraphNode`-shaped data. They reference
# `GraphNode`, `DataflowGraph`, `_DEFAULT_CDNA4_ARCH_PROFILE`, and
# `_resolve_arch_profile` directly — all defined in this file — so no lazy
# reverse-imports are needed. The graph-builder / validator entry points
# (`build_dataflow_graph`, `_collect_pattern`, `diagnose_missing_edge`,
# `_classify_edge_coverage`, `validate_middle_pack_pair_interleaving`) live
# in this file too as of br4.9.


PRODUCER_CATEGORIES_LDS = ("LRA0", "LRA1", "LRA3", "LRB0", "LRB1", "LRB3",
                           "LWA", "LWB", "LW")
PRODUCER_CATEGORIES_GLOBAL = ("GRA", "GRB", "GR")
SWAIT_CATEGORY = "SYNC"
SBARRIER_CATEGORY = "BARRIER"


def counter_for(node_or_category: Union[str, GraphNode]) -> str:
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

    Left as a free function (not migrated to `GraphNode`) per nn0:
    semantically only meaningful for SWait nodes — the SYNC bucket lumps
    SWaitCnt and SBarrier together, and `swait_drains` on an SBarrier
    returns `None` immediately. A method on `GraphNode` would have to
    advertise that asymmetry; keeping it free with this `isinstance`
    early-out keeps the call sites obvious.
    """
    inst = swait_node.rocisa_inst
    if inst is None:
        return None
    # Production lumps SWaitCnt and SBarrier into the same "SYNC" category
    # (see _class_tag_from_category), so this helper is sometimes called
    # with an SBarrier node — those have no counter fields and contribute
    # nothing to drain semantics. Skip them.
    if not isinstance(inst, SWaitCnt):
        return None
    # Direct attribute access: every SWaitCnt rocisa instance carries
    # dscnt/vlcnt/vscnt as bound C++ fields. (Pre-vvcm this was wrapped
    # in `getattr(..., -1)` to tolerate the `_FakeSWait` test impostor;
    # tests now build real SWaitCnt instances.)
    if counter == "dscnt":
        v = inst.dscnt
    elif counter == "vlcnt":
        v = inst.vlcnt
    elif counter == "vscnt":
        v = inst.vscnt
    else:
        return None
    if v is None or v < 0:
        return None
    return v


# `_all_nodes_in_order(graph)` is now `graph.all_nodes_in_order` (property on
# `DataflowGraph`). The new property requires a non-None graph, matching the
# wa57 cleanup that removed the last None-graph call site.


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
    for node in subj_graph.all_nodes_in_order:
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
    for node in subj_graph.all_nodes_in_order:
        if node.category != SBARRIER_CATEGORY:
            continue
        if start < node.position < end:
            out.append(node)
    return out


# Categories that have real wait-counter or finish-time semantics — i.e.
# producers whose results are NOT immediately visible. The complement is
# the ALU set (`_is_alu_producer`); see that function's docstring for the
# DTL-m0 / PackMFMA carve-outs that the surrounding code applies on top.
_NON_ALU_CATEGORIES = frozenset({
    InstructionCategory.LR,
    InstructionCategory.LW,
    InstructionCategory.GR,
    InstructionCategory.MFMA,
})

# Categories whose instances participate in the cross-graph data-flow
# identity set. Excludes the four scheduler-control categories
# (SWAIT / SBARRIER / SNOP / SSETPRIO) — they're emitted by scheduler
# choice, not by user-program semantics, and would create spurious
# identity collisions across captures that picked different wait/nop
# placement. Used by `build_dataflow_graph` Phase 1 (see the
# `nodes_by_identity` accumulation below).
_NO_DATAFLOW_IDENTITY_CATEGORIES = frozenset({
    InstructionCategory.SWAIT,
    InstructionCategory.SBARRIER,
    InstructionCategory.SNOP,
    InstructionCategory.SSETPRIO,
})


# `_class_tag(inst)` and `_class_tag_from_category(category, inst)` are now
# `WrappedInstruction.class_tag(inst)` and
# `WrappedInstruction.class_tag_for_category(category, inst)` (static methods on
# the wrapper) — see ScheduleCapture.py. The 009-introduced
# `_CLASS_TAG_CATEGORIES` frozenset is gone with them: WrappedInstruction.class_tag
# enumerates the per-bucket properties directly (is_lr / is_lw / ...).


# =============================================================================
# --- Stream-position ordering ---
# =============================================================================
# The resolver walks producers in stream-position order. SchedulePosition
# (loop_index, stream_index) — collapsed at rocm-libraries-5v4u from the
# historical (loop_index, vmfma_index, sub_index) triple — lex-sorts to
# actual stream order by construction (the bridge `make_position` /
# `assign_stream_indices_for_body` walks events in `(slot.mfma_index,
# slot.sequence)` order and assigns 0,1,2,... per body). `node.position`
# is the canonical ordering key — no synthetic kind_rank table needed.
#
# Two "iter" axes exist in this codebase; do not conflate them:
#   1. Outer iteration / body — which body the node belongs to
#      (ml_prev, ml, ngl, nll). Encoded in SchedulePosition.loop_index
#      and node.body_label. Cross-body comparison happens naturally via
#      stream-position lex sort (loop_index is the first component).
#   2. Inner-unroll subiteration ("subiter") — which inner unroll iteration
#      within a single body. Encoded in category trailing digits (LRA0,
#      PackB3) for non-MFMA, or in
#      `tagged_inst.slot.mfma_index // num_mfma_per_subiter` for MFMA
#      (sourced from the kernel-writer's slot, NOT from the bridge-collapsed
#      `stream_index`). Computed by _node_subiter.
#
# _node_subiter is used by the within-graph same-subiter gate in
# _classify_edge_coverage (which has no default reference). Both subiter
# derivations are schedule-invariant per identity (categories and slot
# mfma_indices are kernel-writer-set, identical across captures).

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


# `_node_subiter(node, n)` is now `node.subiter(n)` (method on `GraphNode`).


# `_identity_for(inst, body_label, category=None)` is now
# `TaggedInstruction.identity_for(body_label)` (instance method on
# TaggedInstruction in ScheduleCapture.py). Callers that have a bare
# rocisa instance and a category must construct a TaggedInstruction
# (via `WrappedInstruction(inst)` plus a category) before calling.
# Removed in the nn0 follow-up (2026-05-08): the free-function form was
# a parallel API that the user has rejected.


# `_min_issue_quad_cycles_for(rocisa_inst, profile)` is now
# `profile.min_issue_quad_cycles_for(rocisa_inst)` (method on `ArchProfile`).
# Production callers always have a non-None profile; the method documents the
# requirement instead of raising.


def _make_node(
    tagged_inst: "TaggedInstruction",
    body_label: str,
    stream_index: int,
    profile: Optional[ArchProfile] = None,
) -> GraphNode:
    inst = tagged_inst.wrapped.rocisa_inst
    identity = tagged_inst.identity_for(body_label)
    position = make_position(body_label, stream_index)
    # Node name continues to reference the kernel-writer's slot fields
    # (`mfma_index.sequence`) — sourced from `tagged_inst.slot` because
    # SchedulePosition no longer carries vmfma_index / sub_index after the
    # 5v4u collapse. Names remain stable for any test that grep'd them.
    slot = tagged_inst.slot
    name = f"{tagged_inst.category}@{slot.mfma_index}.{slot.sequence}"
    # `issue_cycles` is only meaningful when an arch profile is available
    # (the field is never consumed downstream — `cumulative_issue_cycles`
    # calls `_min_issue_quad_cycles_for` directly). Skip the helper call
    # entirely on the unknown-ISA path so the helper can require a
    # non-None profile.
    issue_cycles = profile.min_issue_quad_cycles_for(inst) if profile is not None else 0
    return GraphNode(
        identity=identity,
        position=position,
        category=tagged_inst.category,
        rocisa_inst=inst,
        tagged_inst=tagged_inst,
        body_label=body_label,
        name=name,
        issue_cycles=issue_cycles,
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
    (SchedulePosition: loop_index, stream_index) and yield
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
    not emit NGL or NLL leave the corresponding dict empty (the capture
    pipeline observes whether the kernel actually emitted the body, and
    the CMS-side capture mirrors that shape — see rocm-libraries-dj1g);
    the loop below skips absent bodies cleanly and the validator runs
    against the remaining bodies.

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
    # constants. `None` means "no profile registered for this kernel's
    # ISA; timing checks are skipped." Tracked: `rocm-libraries-zkzw`.
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

        # Per-body stream_index assignment (the CMS bridge: collapses
        # `(slot.mfma_index, slot.sequence)` lex order into a single
        # monotonic int per body). See SchedulePosition / make_position
        # docstrings.
        stream_idx_by_id = assign_stream_indices_for_body(body.instructions)

        for tagged_inst in body.instructions:
            inst = tagged_inst.wrapped.rocisa_inst
            try:
                node = _make_node(
                    tagged_inst,
                    label,
                    stream_idx_by_id[id(tagged_inst)],
                    arch_profile,
                )
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
            if _category(inst) not in _NO_DATAFLOW_IDENTITY_CATEGORIES:
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
        # latest_writer entries now carry the producer's positional
        # write-slot so the cross-graph edge identity can encode WHICH
        # operand-slot of the producer published this byte
        # (rocm-libraries-wx9.3 phase 3, memo §6.1 step 1).
        latest_writer = {}  # byte_key -> (writer_node, write_resource, write_slot)
        sorted_nodes = sorted(nodes_by_identity.values(), key=lambda n: n.position)

        # Track the body_label of the previously processed node so we can
        # detect body-boundary transitions (ML-1 -> ML, ML -> NGL, NGL ->
        # NLL) within the unified position-sorted stream. SCC is a single-
        # bit hardware status register that is NOT preserved across loop
        # iterations by any compiler convention, so an SCC writer in body
        # N must NOT be visible to an SCC reader in body N+1. Clearing
        # the SCC entries from latest_writer at the boundary stops the
        # per-byte resolver from emitting cross-body SCC edges in the
        # first place — fixing the artifact at its source rather than
        # absorbing it later in the failure-classification layer
        # (rocm-libraries-theq, see SCC_CROSS_BODY_INVESTIGATION.md §5).
        # NOTE: only SCC keys are cleared. Non-SCC dataflow (LR/LW/GR/
        # MFMA on vgpr/sgpr/memory) IS legitimately preserved across body
        # boundaries — that cross-body dataflow is the whole reason the
        # graph walks bodies as a single stream.
        prev_body_label = None

        for node in sorted_nodes:
            if prev_body_label is not None and node.body_label != prev_body_label:
                # Body-boundary transition. Drop SCC byte_keys
                # (first tuple element == "scc") so SCC writers in
                # the previous body cannot source SCC reads in the
                # next body.
                latest_writer = {
                    bk: v for bk, v in latest_writer.items()
                    if not (bk and bk[0] == "scc")
                }
            prev_body_label = node.body_label

            wrapped = node.tagged_inst.wrapped
            # Per-body symbolic-name -> numeric-base lookup, populated
            # from the writer's RegSet directives during capture (see
            # `LoopBodyCapture.name_to_idx` and `collect_regset_stream`).
            # Forwarding it to `_byte_keys_for_resource` makes symbolic
            # operands resolve to the same numeric byte-keys as the
            # corresponding numeric writes, so cross-form references
            # (e.g. consumer reads `vgprValuA_T0_I0`, producer writes
            # `v[76:79]` for the same physical reg) collapse to a single
            # latest-writer entry. rocm-libraries-bb34.
            body_capture = captures.get(node.body_label)
            n2i = getattr(body_capture, "name_to_idx", None) if body_capture is not None else None

            # Phase 2a — reads first: emit one edge per distinct
            # (writer, write_resource, write_slot) that contributes any
            # byte of any read of this node. The reader's positional
            # read-slot rides alongside the resource via wrapped.read_slots.
            read_slots = getattr(wrapped, "read_slots", None) or tuple(
                range(len(wrapped.reads))
            )
            for read_idx, read_resource in enumerate(wrapped.reads):
                if read_resource is None:
                    continue
                sink_slot = (read_slots[read_idx] if read_idx < len(read_slots)
                             else read_idx)
                for producer, overlap, intra_offsets, src_slot in _resolve_producers(
                    read_resource, node, latest_writer, name_to_idx=n2i,
                ):
                    is_memory = isinstance(overlap, MemoryRegion)
                    edges.append(DataflowEdge(
                        producer=producer,
                        consumer=node,
                        resource=overlap,
                        edge_kind=("lds_raw_intrawave" if is_memory
                                   else "raw_intrawave"),
                        intra_operand_byte_offset=intra_offsets,
                        src_operand_slot=src_slot,
                        sink_operand_slot=sink_slot,
                    ))

            # Phase 2b — writes second: update latest_writer for every
            # byte this node covers. Done AFTER reads so a single
            # instruction reading and writing the same register sees its
            # PREVIOUS writer, not itself.
            write_slots = getattr(wrapped, "write_slots", None) or tuple(
                range(len(wrapped.writes))
            )
            for write_idx, write_resource in enumerate(wrapped.writes):
                if write_resource is None:
                    continue
                w_slot = (write_slots[write_idx] if write_idx < len(write_slots)
                          else write_idx)
                for bk in _byte_keys_for_resource(write_resource, name_to_idx=n2i):
                    latest_writer[bk] = (node, write_resource, w_slot)

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

                # Intra-operand byte offset for LDS-reuse edges: the
                # resource is the producer's dst (an LDS slot pin). The
                # offset tuple covers all bytes of that resource — the
                # whole slot participates in the cross-wave handoff. This
                # is allocation-invariant by construction because the
                # tuple is intra-operand byte indices (0..N-1), not
                # absolute register byte-keys.
                bks = _byte_keys_for_resource(resource) if resource is not None else ()
                intra_offsets = tuple(range(len(bks)))

                # SBarrier-mediated LDS-reuse edges: the producer's
                # written resource is dst (write-slot 0); the consumer
                # reads its dst (read-slot 0). Both slot indices are
                # fixed at 0 — there's no other positional choice for
                # these patterns. Required for the cross-graph edge
                # identity (wx9.3 phase 3).
                edges.append(DataflowEdge(
                    producer=producer,
                    consumer=node,
                    resource=resource,
                    edge_kind=edge_kind,
                    intra_operand_byte_offset=intra_offsets,
                    src_operand_slot=0,
                    sink_operand_slot=0,
                ))

            # Pattern reset: a NEW producer of producer_categories ends this
            # producer's "passing window". The new producer starts fresh.
            if node.category in producer_categories:
                break

    return edges


# -----------------------------------------------------------------------------
# FIFO simulator + producer-classifier helpers
# -----------------------------------------------------------------------------


# `_queue_depth_at(wait, producer, graph)` is now
# `graph.queue_depth_at(wait, producer)`.
# `_producer_queue_position(producer, graph)` is now
# `graph.producer_queue_position(producer)`.
# `_wait_drains_producer(wait, producer, graph)` is now
# `graph.wait_drains_producer(wait, producer)`.
# `_any_drains(waits, producer, graph)` is now `graph.any_drains(waits, producer)`.


# `_mfma_finish_cycles_for(rocisa_inst, profile)` is now
# `profile.mfma_finish_cycles_for(rocisa_inst)` (method on `ArchProfile`).
# `_is_mfma_pack_producer(producer)`, `_is_mfma_producer(producer)`, and
# `_is_cvt_pack_producer(producer)` are now staticmethods on `GraphNode`:
# `GraphNode.is_mfma_pack_producer(producer)`,
# `GraphNode.is_mfma_producer(producer)`,
# `GraphNode.is_cvt_pack_producer(producer)`. They are static rather than
# instance methods so duck-typed test stubs (which lack the full GraphNode
# shape) can still dispatch through them.




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

    # Resolve the per-arch profile from the graph. The simulator below uses
    # `profile.mfma_4x4_finish_cycles` for MFMA-class discrimination
    # (rather than the legacy module-scope alias) so per-arch overrides
    # to the 4x4 finish window don't decouple from the discriminator.
    # All callers (the four pair-specific timing helpers) short-circuit
    # on `graph.arch_profile is None` before invoking this function;
    # production code paths therefore never reach here with a None
    # profile. Test fixtures that want timing checks must pass an
    # explicit profile (e.g. `_DEFAULT_CDNA4_ARCH_PROFILE`).
    profile = ArchProfile.from_carrier(graph)

    # Producer must always be strictly before consumer in stream order. The
    # SchedulePosition `__lt__` compares (loop_index, stream_index)
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
    # Fallback slot keys for the by-slot lookup below: SchedulePosition no
    # longer carries (vmfma_index, sub_index) after the 5v4u collapse, so
    # source the kernel-writer slot tuple directly from the node's own
    # `tagged_inst.slot`. The `(mfma_index, sequence)` pair is the same
    # tuple we used pre-collapse — only its provenance changes.
    def _slot_key(node):
        ti = getattr(node, "tagged_inst", None)
        if ti is None or getattr(ti, "slot", None) is None:
            return None
        return (ti.slot.mfma_index, ti.slot.sequence)
    p_key = _slot_key(producer)
    c_key = _slot_key(consumer)

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
            is_mfma = inst is not None and _category(inst) is InstructionCategory.MFMA
            if is_mfma:
                current_issue = max(current_issue, mfma_free_at)
                current_mfma_class = profile.mfma_finish_cycles_for(inst)
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
            current_issue += profile.min_issue_quad_cycles_for(inst)

        if c_issue_start is not None:
            break

    if p_issue_start is None or c_issue_start is None:
        return 0
    return c_issue_start - p_issue_start - 1


class TimingResult(Enum):
    """Outcome of a per-edge timing check.

    PASS                 — observed cycle gap >= required threshold.
    FAIL                 — observed gap is strictly less than required;
                           caller emits a `TimingTooCloseFailure`.
    ARCH_NOT_SUPPORTED   — the kernel's ISA had no `ArchProfile` registered
                           in `_ARCH_PROFILES_BY_ISA`. The check is
                           short-circuited (no timing failure emitted) and
                           the warning was already fired at resolve time.
    """
    PASS = "pass"
    FAIL = "fail"
    ARCH_NOT_SUPPORTED = "arch_not_supported"


@dataclass(frozen=True)
class TimingCheck:
    """Return value of the four pair-specific timing helpers
    (`_quad_cycle_gap_ok`, `_cvt_to_mfma_gap_ok`,
    `_mfma_pack_to_cvt_gap_ok`).

    Fields:
      result   — `TimingResult` enum (PASS / FAIL / ARCH_NOT_SUPPORTED).
      observed — observed quad-cycles between producer and consumer
                 (0 when ARCH_NOT_SUPPORTED or graph absent).
      required — per-arch threshold the gap must meet for PASS
                 (0 when ARCH_NOT_SUPPORTED).
    """
    result: TimingResult
    observed: int
    required: int

    @classmethod
    def arch_not_supported(cls) -> "TimingCheck":
        return cls(result=TimingResult.ARCH_NOT_SUPPORTED, observed=0, required=0)

    @classmethod
    def passing(cls, observed: int, required: int) -> "TimingCheck":
        return cls(result=TimingResult.PASS, observed=observed, required=required)

    @classmethod
    def failing(cls, observed: int, required: int) -> "TimingCheck":
        return cls(result=TimingResult.FAIL, observed=observed, required=required)


# `_quad_cycle_gap_ok(producer, consumer, num_mfma_per_subiter, graph)` is now
# `graph.arch_profile.quad_cycle_gap_ok(producer, consumer, graph)` with the
# unknown-ISA short-circuit handled at the call site (see
# `_classify_edge_coverage` / `diagnose_missing_edge`):
#
#     if graph.arch_profile is None:
#         check = TimingCheck.arch_not_supported()
#     else:
#         check = graph.arch_profile.quad_cycle_gap_ok(p, c, graph)
#
# (The defensive `graph is None` guard from the legacy free function is gone:
# production callers always pass a real graph, and that path was an artifact
# of unit-test scaffolding.)


def _cvt_to_mfma_gap_ok(
    producer: GraphNode, consumer: GraphNode, subj_graph: DataflowGraph
) -> "TimingCheck":
    """Verify that enough quad-cycles separate a CVTPack producer from its
    downstream MFMA consumer for the CVT result to be visible.

    The threshold is fixed at `_QUAD_CYCLES_CVT_BEFORE_MFMA == 2`
    (CDNA 4 ISA 7.6).

    Returns a `TimingCheck` — same shape as `ArchProfile.quad_cycle_gap_ok`
    so callers can wrap a single `TimingTooCloseFailure(required, observed)`
    regardless of the gap kind.

    Same-body and cross-body share ONE code path that delegates to
    `cumulative_issue_cycles`. WARNING for future reverts: the previous
    slot-delta formula (`slot_delta * (1 + finish) - 1 + intervening`)
    was DOUBLE-COUNTING — it charged 1 cycle per slot-INDEX gap AND
    +intervening for actual instructions in those slots. The cycle-exact
    walk only counts actual instructions, producing a smaller (more
    conservative) `observed` for densely-populated streams. A previous
    `body_delta * 1000` cross-body placeholder is also gone;
    `cumulative_issue_cycles` walks the unified instruction stream
    across all bodies in `_BODY_BUILD_ORDER`.
    """
    # Unknown-ISA short-circuit (rocm-libraries-zkzw).
    if subj_graph.arch_profile is None:
        return TimingCheck.arch_not_supported()

    profile = subj_graph.arch_profile
    required = profile.cvt_before_mfma_quad_cycles

    observed = cumulative_issue_cycles(subj_graph, producer, consumer)
    if observed >= required:
        return TimingCheck.passing(observed, required)
    return TimingCheck.failing(observed, required)


def _mfma_pack_to_cvt_gap_ok(
    producer: GraphNode, consumer: GraphNode, subj_graph: DataflowGraph
) -> "TimingCheck":
    """Verify that enough quad-cycles separate a 4x4 PackMFMA producer from
    its downstream CVTPack (CVT1) consumer for the accumulator to settle.

    The threshold is fixed at `_QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1 == 5`
    (CDNA 4 ISA 7.6). This is the LARGEST gap among the four
    section-7.6 quad-cycle constants; the 4x4 MFMA finish-cycle (1) is
    shorter than the 5-cycle visibility window the CVT1 needs, so this
    helper enforces a larger min-gap on PackMFMA->CVTPack edges than
    the bare finish would suggest.

    Returns a `TimingCheck` — same shape as `ArchProfile.quad_cycle_gap_ok`
    and `_cvt_to_mfma_gap_ok` so callers can wrap a single
    `TimingTooCloseFailure(required, observed)` regardless of the gap
    kind.

    Approach: CYCLE-EXACT via `cumulative_issue_cycles`, the same
    simulator `ArchProfile.quad_cycle_gap_ok` uses. The helper walks
    the captured stream from the producer to the consumer, accumulating
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
    # Unknown-ISA short-circuit (rocm-libraries-zkzw).
    if subj_graph.arch_profile is None:
        return TimingCheck.arch_not_supported()

    profile = subj_graph.arch_profile
    required = profile.mfma_4x4_before_cvt_quad_cycles

    observed = cumulative_issue_cycles(subj_graph, producer, consumer)
    if observed >= required:
        return TimingCheck.passing(observed, required)
    return TimingCheck.failing(observed, required)


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
        if GraphNode.is_mfma_pack_producer(producer):
            return False
        return True
    if cat == "MFMA":
        return False
    inst = getattr(producer, "rocisa_inst", None)
    if inst is not None:
        if _category(inst) in _NON_ALU_CATEGORIES:
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
        if not subj_graph.wait_drains_producer(w, producer):
            return w
    return None


def _last_drain(
    waits: List[GraphNode], producer: GraphNode, subj_graph: DataflowGraph
) -> Optional[GraphNode]:
    """Return the latest wait that drained the producer, else None."""
    drainers = [w for w in waits if subj_graph.wait_drains_producer(w, producer)]
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


# `_node_position(node)` is now `node.canonical_position` (property on
# `GraphNode`). New code reads `node.position` directly.


def _node_display_idx(node: NodeLike) -> int:
    """Source-aware "user-facing N" for failure rendering.

    For CMS-side nodes: recover the kernel-writer's `slot.mfma_index` from
    `tagged_inst.slot` so existing pinning tests that match `@ idx=N` keep
    seeing the kernel-writer's MFMA-slot id. For nodes without a tagged
    instruction (SWaitCnt / SBarrier / SNop in the sidecar; ValidatorInstruction
    on the structural side), fall back to `position.stream_index` — these
    nodes never carried a meaningful CMS `vmfma_index` either, so the
    bridge-collapsed sort key is the best we can do without the
    source-aware label dispatch tracked under rocm-libraries-3dy.
    """
    ti = getattr(node, "tagged_inst", None)
    if ti is not None and getattr(ti, "slot", None) is not None:
        return ti.slot.mfma_index
    return node.position.stream_index


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


class ValidationError(Exception):
    """Raised by ``isValid`` on the first validation failure.

    Carries the typed ``Failure`` describing the problem; ``str(exc)``
    delegates to ``failure.format()`` so the formatted message is the
    natural rendering. Callers that want structured access read
    ``exc.failure``.

    The exception model replaces the old ``tuple[bool, str]`` return
    contract: ignoring a failure is no longer possible without an
    explicit ``try/except ValidationError`` at the call site.

    A ``message`` may be supplied directly (without a Failure) for
    legacy non-typed failure sites that build prose strings; in that
    case ``failure`` is None and ``str(exc)`` returns the prose.
    """

    def __init__(self, failure: 'Optional[Failure]' = None, message: Optional[str] = None):
        if failure is None and message is None:
            raise ValueError("ValidationError requires either a failure or a message")
        self.failure = failure
        self._message = message
        super().__init__(message if message is not None else (failure.format() if failure is not None else ""))

    def __str__(self) -> str:
        if self.failure is not None:
            return self.failure.format()
        return self._message or ""


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
    node: GraphNode,
    body_capture: "LoopBodyCapture",
) -> FailureNodeLabel:
    """Construct a FailureNodeLabel for a CMS-side GraphNode.

    Wording is identical to the old `_node_with_pos`:
      - primary: 'category[N]' (per-category-stream 0-based index in the
        node's body capture); plain MFMA omits '[N]'.
      - position: '@ idx={vmfma_index}'.

    `body_capture` is the LoopBodyCapture for the body that emitted this
    node (resolved by the caller from `node.body_label`, typically via
    `_body_for_node`). It is required and non-None: every GraphNode is
    constructed from a body that lives in `graph.captures`, so the lookup
    cannot miss. The same invariant guarantees `node.tagged_inst` appears
    in `body_capture.instructions`, so the per-category index lookup also
    cannot miss; the `same_cat.index` call uses an explicit assertion to
    surface any future violation rather than silently degrading.
    """
    cat = node.category
    if cat == "MFMA":
        primary = cat
    else:
        same_cat = [
            t for t in body_capture.instructions
            if getattr(t, "category", None) == cat
        ]
        # node.tagged_inst is guaranteed to be in body_capture.instructions
        # (the node was constructed from this body in build_dataflow_graph),
        # so it must appear in same_cat. Any miss indicates a capture-
        # pipeline bug — surface it loudly rather than degrade the label.
        assert node.tagged_inst in same_cat, (
            f"cms_node_label: node.tagged_inst not found in body_capture "
            f"instructions for category {cat!r} (body_label={node.body_label!r}). "
            f"Every GraphNode must be constructed from the body it indexes."
        )
        idx = same_cat.index(node.tagged_inst)
        primary = f"{cat}[{idx}]"
    return FailureNodeLabel(
        primary=primary,
        position=_PositionStr(f"@ idx={_node_display_idx(node)}"),
        category=cat,
        body_label=node.body_label,
    )


# `_cms_iter_delta(producer, consumer)` is now `producer.iter_delta_to(consumer)`
# (method on `GraphNode`).


# `_body_for_node(graph, node)` is now `graph.body_for(node)` (method on
# `DataflowGraph`).


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
) -> List["Failure"]:
    """Compare two dataflow graphs as edge sets keyed on
    (producer.identity, consumer.identity, register, edge_kind).

    Returns a list of Failure objects — one or more per missing edge,
    routed through diagnose_missing_edge.

    Raises CaptureConsistencyError BEFORE comparison if the two graphs'
    DATA-FLOW node identity sets differ — a capture-pipeline bug, not a
    CMS schedule defect.

    Unclassified missing edges raise UnexplainedMissingEdgeError
    unconditionally (via diagnose_missing_edge): a fall-through is a
    validator bug, not a soft observability event. There is no
    silent-Failure path — production observes the raise the same way
    tests do.
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
    # Same edge-key shape as DataflowGraph.edge_keys() (memo §6.1 step 1,
    # wx9.3 phase 3):
    # (src_role, src_position, src_operand_slot,
    #  sink_role, sink_position, sink_operand_slot,
    #  edge_kind, intra_operand_byte_offset).
    failures = []
    ref_edges_by_key = {
        (e.producer.identity[0], e.producer.position, e.src_operand_slot,
         e.consumer.identity[0], e.consumer.position, e.sink_operand_slot,
         e.edge_kind, e.intra_operand_byte_offset): e
        for e in reference.edges
    }
    for key in missing_keys:
        ref_edge = ref_edges_by_key[key]
        failures.extend(diagnose_missing_edge(ref_edge, subject))
    return failures


def diagnose_missing_edge(
    ref_edge: DataflowEdge,
    subj_graph: DataflowGraph,
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

    A fall-through (no classifier branch claimed the edge) raises
    UnexplainedMissingEdgeError unconditionally — it indicates a
    classifier or capture-pipeline bug. Production observes the same
    raise as unit tests; there is no soft synthetic-Failure path. This
    matches the validator's no-silent-ignore contract: the validator
    either knows or admits it doesn't know, by raising.
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

    # Legitimate-CMS-reorder branch (rocm-libraries-wx9.3 memo §6.1).
    # The new edge identity tuple includes the producer/consumer
    # SchedulePositions (memo §6.1, point 1), so a CMS reorder that pushes
    # an edge's endpoints to different stream slots in subj surfaces here
    # as a "missing" key even when the same logical producer-consumer
    # dataflow exists in subj. The legitimate case: subj contains an edge
    # with the same (producer.identity, consumer.identity, edge_kind,
    # intra_operand_byte_offset). Edge formation only emits an edge when
    # the writer was seen before the reader in stream order
    # (build_dataflow_graph Phase 2's latest_writer is updated AFTER
    # reads), so the existence of such an edge in subj already implies
    # the relative order is preserved — no inversion. Return [] without
    # firing any Failure. This is a classifier-pipeline tweak; the
    # `Failure` hierarchy is unchanged (memo §6.1, point 2).
    edge_kind = ref_edge.edge_kind
    intra = ref_edge.intra_operand_byte_offset
    src_slot = ref_edge.src_operand_slot
    sink_slot = ref_edge.sink_operand_slot
    for e in subj_graph.edges:
        if (e.producer.identity == p_id
                and e.consumer.identity == c_id
                and e.edge_kind == edge_kind
                and e.intra_operand_byte_offset == intra
                and e.src_operand_slot == src_slot
                and e.sink_operand_slot == sink_slot):
            return []

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
                    and p_node.subiter(nmps) != c_node.subiter(nmps)):
                return []  # cross-subiter pipelined dependency — legitimate
            return [OrderInvertedFailure(
                producer=cms_node_label(p_node, subj_graph.body_for(p_node)),
                consumer=cms_node_label(c_node, subj_graph.body_for(c_node)),
                iter_delta=p_node.iter_delta_to(c_node),
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
        # Cross-body SCC edges no longer reach this branch: the per-byte
        # latest-writer resolver in build_dataflow_graph clears SCC
        # entries at body boundaries (rocm-libraries-theq), so any SCC
        # ref_edge here connects a producer and consumer in the SAME
        # body. The downstream cross-body suppression that used to live
        # here was absorbing artifacts the resolver should never have
        # emitted — see SCC_CROSS_BODY_INVESTIGATION.md §5 for the
        # rationale.
        intervening_writer = None
        for e in subj_graph.edges:
            if (e.consumer.identity == c_id
                    and getattr(e.resource, "regType", None) == "scc"
                    and e.producer.identity != p_id):
                intervening_writer = e.producer
                break
        if intervening_writer is not None:
            return [OverriddenInputFailure(
                producer=cms_node_label(p_node, subj_graph.body_for(p_node)),
                consumer=cms_node_label(c_node, subj_graph.body_for(c_node)),
                iter_delta=p_node.iter_delta_to(c_node),
                resource="SCC",
                intervening_writer=cms_node_label(
                    intervening_writer,
                    subj_graph.body_for(intervening_writer),
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
    if (GraphNode.is_mfma_pack_producer(p_node)
            and GraphNode.is_cvt_pack_producer(c_node)):
        check = _mfma_pack_to_cvt_gap_ok(
            p_node, c_node, subj_graph)
        if check.result == TimingResult.FAIL:
            return [TimingTooCloseFailure(
                producer=cms_node_label(p_node, subj_graph.body_for(p_node)),
                consumer=cms_node_label(c_node, subj_graph.body_for(c_node)),
                iter_delta=p_node.iter_delta_to(c_node),
                expected_quad_cycles=check.required,
                actual_quad_cycles=check.observed,
            )]
        return []

    # MFMA-as-producer: governed by quad-cycle issue-timing constraints,
    # not by the dscnt/vlcnt FIFO. An explicit gap check fires; an MFMA
    # producer whose consumer fires too soon after it is a TimingTooClose
    # violation. Dispatch is via `_is_mfma_producer` (rather than `category
    # == "MFMA"`) so 4x4 PackMFMAs (categorized Pack* but syntactically
    # MFMAInstruction) are routed here BEFORE the ALU exemption claims
    # them.
    if GraphNode.is_mfma_producer(p_node):
        if subj_graph.arch_profile is None:
            check = TimingCheck.arch_not_supported()
        else:
            check = subj_graph.arch_profile.quad_cycle_gap_ok(p_node, c_node, subj_graph)
        if check.result == TimingResult.FAIL:
            return [TimingTooCloseFailure(
                producer=cms_node_label(p_node, subj_graph.body_for(p_node)),
                consumer=cms_node_label(c_node, subj_graph.body_for(c_node)),
                iter_delta=p_node.iter_delta_to(c_node),
                expected_quad_cycles=check.required,
                actual_quad_cycles=check.observed,
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
    if GraphNode.is_cvt_pack_producer(p_node) and GraphNode.is_mfma_producer(c_node):
        check = _cvt_to_mfma_gap_ok(p_node, c_node, subj_graph)
        if check.result == TimingResult.FAIL:
            return [TimingTooCloseFailure(
                producer=cms_node_label(p_node, subj_graph.body_for(p_node)),
                consumer=cms_node_label(c_node, subj_graph.body_for(c_node)),
                iter_delta=p_node.iter_delta_to(c_node),
                expected_quad_cycles=check.required,
                actual_quad_cycles=check.observed,
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

    p_label = cms_node_label(p_node, subj_graph.body_for(p_node))
    c_label = cms_node_label(c_node, subj_graph.body_for(c_node))
    iter_delta = p_node.iter_delta_to(c_node)

    if not waits:
        # No SWait on the expected counter at all in the window. If other-
        # counter SWaits exist, surface their vmfma indices via
        # `nearby_wait_indices` so the user can extend one of them rather
        # than insert a new SWaitCnt; the underlying fix is the same either
        # way.
        nearby_indices = tuple(_node_display_idx(w) for w in waits_other)
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
        if not subj_graph.any_drains(waits, p_node):
            insufficient = _first_insufficient(waits, p_node, subj_graph)
            if insufficient is not None:
                # Compute queue depth at the wait's position for diagnostic.
                depth = subj_graph.queue_depth_at(insufficient, p_node)
                cv = _swait_drains(insufficient, expected_counter)
                pos = subj_graph.producer_queue_position(p_node)
                failures.append(WaitInsufficientFailure(
                    producer=p_label,
                    consumer=c_label,
                    iter_delta=iter_delta,
                    wait_idx=_node_display_idx(insufficient),
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
                    wait_idx=_node_display_idx(last_drain),
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
                and waits and subj_graph.any_drains(waits, p_node)):
            return []

        # Couldn't classify — capture pipeline bug or classifier bug.
        # Always raise: an unclassified edge means the validator doesn't
        # know what to make of something the graph builder produced. A
        # silent soft-Failure here would be the silent-miss pattern the
        # validator is explicitly designed to eliminate.
        raise UnexplainedMissingEdgeError(
            f"diagnose_missing_edge could not classify missing edge "
            f"{p_id} -> {c_id} (kind={ref_edge.edge_kind}). "
            f"This indicates either a classifier bug or a capture-pipeline "
            f"bug that bypassed earlier sanity checks."
        )
    return failures


# =============================================================================
# Self-validation: per-edge wait coverage
# =============================================================================
# Edges are reorder-invariant (register-name resolution). The
# scheduler-correctness check — does the schedule have a covering
# s_waitcnt that drains each producer before its consumer reads? —
# is a SEPARATE pass over the graph + the captured stream.
#
# Same Failure types the cross-graph diagnose_missing_edge classifier
# emits — the wiring is just driven differently. Instead of "for each
# missing edge, classify why subject lacks it", it's "for each edge in
# the (single) graph, classify whether the schedule covers it".
#
# br4.9 moved these from ScheduleCapture.py. They sit next to
# diagnose_missing_edge (the cross-graph counterpart) so the per-edge
# classifier dispatch can be inspected side-by-side.


def validate_edge_wait_coverage(
    graph: "DataflowGraph",
) -> List["Failure"]:
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

    Every edge classifies positively — no unexplained fall-through
    path: typed early-return branches (MFMA/CVT/ALU exemptions) cover
    structural cases, and Phase-2 wait coverage either appends a
    failure or confirms the edge is properly covered. (The historic
    `raise_on_unexplained` parameter and its synthetic Failure soft
    path were dead code: no caller ever passed `True`, and an empty
    failure list at the bottom of `_classify_edge_coverage` is a
    positive classification, not a miss.)

    Returns a list of Failure objects. Empty list means "every edge in
    the graph has a covering wait/barrier in the captured stream".
    """
    failures = []
    for edge in graph.edges:
        edge_failures = _classify_edge_coverage(edge, graph)
        failures.extend(edge_failures)
    # MiddlePack pair-interleaving is a stream-shape invariant, not an
    # edge-shape invariant, so it's a sibling pass driven from the same
    # entry point (rather than per-edge inside _classify_edge_coverage).
    failures.extend(validate_middle_pack_pair_interleaving(graph))
    return failures


def _classify_edge_coverage(
    edge: "DataflowEdge", subj_graph: "DataflowGraph",
) -> List["Failure"]:
    """Per-edge coverage classifier — same logic diagnose_missing_edge
    runs in compare_graphs, but driven from a single graph rather than
    a missing-edge diff.

    Every reachable end-state classifies: typed early-return branches
    (MFMA/CVT/ALU producer exemptions) or Phase-2 wait coverage. There
    is no unexplained fall-through — if Phase-2 cannot find a draining
    wait it appends a MissingWait/WaitInsufficient failure; if it does,
    the edge is positively classified as covered.
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
    # The structural-side mirror was removed; this dispatch is now the
    # only enforcement path. Must run BEFORE the generic
    # `_is_mfma_producer` branch below: PackMFMA producers are claimed by
    # `_is_mfma_producer` and would otherwise route through
    # `_quad_cycle_gap_ok`, whose threshold for 4x4 PackMFMAs
    # (`_mfma_finish_cycles_for == 1`) is too weak by 4 quad-cycles
    # versus the 5-cycle CVT1 visibility window.
    p_label = cms_node_label(p_node, subj_graph.body_for(p_node))
    c_label = cms_node_label(c_node, subj_graph.body_for(c_node))
    iter_delta = p_node.iter_delta_to(c_node)

    if (GraphNode.is_mfma_pack_producer(p_node)
            and GraphNode.is_cvt_pack_producer(c_node)):
        check = _mfma_pack_to_cvt_gap_ok(
            p_node, c_node, subj_graph)
        if check.result == TimingResult.FAIL:
            return [TimingTooCloseFailure(
                producer=p_label,
                consumer=c_label,
                iter_delta=iter_delta,
                expected_quad_cycles=check.required,
                actual_quad_cycles=check.observed,
            )]
        return []

    # MFMA-as-producer: gated by quad-cycle issue timing rather than
    # SWaitCnt (see diagnose_missing_edge). An explicit gap check fires.
    # Dispatch is via `_is_mfma_producer` (rather than `category == "MFMA"`)
    # so PackMFMAs (categorized Pack* but rocisa MFMAInstruction) reach
    # this branch instead of getting silently exempted by
    # `_is_alu_producer` below.
    if GraphNode.is_mfma_producer(p_node):
        if subj_graph.arch_profile is None:
            check = TimingCheck.arch_not_supported()
        else:
            check = subj_graph.arch_profile.quad_cycle_gap_ok(p_node, c_node, subj_graph)
        if check.result == TimingResult.FAIL:
            return [TimingTooCloseFailure(
                producer=p_label,
                consumer=c_label,
                iter_delta=iter_delta,
                expected_quad_cycles=check.required,
                actual_quad_cycles=check.observed,
            )]
        return []

    # CVTPack-as-producer feeding MFMA-as-consumer: governed by the
    # `_QUAD_CYCLES_CVT_BEFORE_MFMA` (= 2) settle window from CDNA 4 ISA
    # section 7.6. The structural-side mirror was removed; this dispatch
    # is now the only enforcement path. Must run BEFORE the ALU-immediate
    # exemption below — CVTPacks are categorized `Pack*` and
    # `_is_alu_producer` would otherwise silently absorb them and skip
    # the gap check entirely (same dispatch-order constraint as the
    # PackMFMA carve-out above). Restricted to MFMA consumers: a CVTPack
    # feeding a non-MFMA consumer (e.g. another Pack or VXor that uses
    # the converted result for something other than an MFMA operand load)
    # carries no quad-cycle constraint — fall through to the ALU
    # exemption in that case.
    if GraphNode.is_cvt_pack_producer(p_node) and GraphNode.is_mfma_producer(c_node):
        check = _cvt_to_mfma_gap_ok(p_node, c_node, subj_graph)
        if check.result == TimingResult.FAIL:
            return [TimingTooCloseFailure(
                producer=p_label,
                consumer=c_label,
                iter_delta=iter_delta,
                expected_quad_cycles=check.required,
                actual_quad_cycles=check.observed,
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
        nearby_indices = tuple(_node_display_idx(w) for w in waits_other)
        failures.append(MissingWaitFailure(
            producer=p_label, consumer=c_label,
            iter_delta=iter_delta,
            counter_kind=expected_counter,
            nearby_wait_indices=nearby_indices,
        ))
        wait_failure_emitted = True
    else:
        if not subj_graph.any_drains(waits, p_node):
            insufficient = _first_insufficient(waits, p_node, subj_graph)
            if insufficient is not None:
                depth = subj_graph.queue_depth_at(insufficient, p_node)
                cv = _swait_drains(insufficient, expected_counter)
                pos = subj_graph.producer_queue_position(p_node)
                failures.append(WaitInsufficientFailure(
                    producer=p_label, consumer=c_label,
                    iter_delta=iter_delta,
                    wait_idx=_node_display_idx(insufficient),
                    counter_kind=expected_counter,
                    counter_value=cv if cv is not None else 0,
                    queue_depth_at_wait=depth,
                    producer_position=pos,
                ))
            else:
                failures.append(MissingWaitFailure(
                    producer=p_label, consumer=c_label,
                    iter_delta=iter_delta,
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
                    producer=p_label, consumer=c_label,
                    iter_delta=iter_delta,
                    role=role,
                    wait_idx=_node_display_idx(last_drain),
                ))

    # No fall-through case here: every reachable end-state of this
    # function either returned early from a typed branch (MFMA/CVT/ALU
    # producer exemptions) or classified the edge against Phase-2 wait
    # coverage. `failures == []` at this point means "edge is properly
    # covered by an SWaitCnt that drains the producer (and a barrier if
    # required for LDS reuse)" — that is a positive classification, not
    # an unexplained miss. The historic `raise_on_unexplained=True`
    # branch that fired here was dead code; if ever exercised, it would
    # have raised on every clean edge. (Compare with
    # `diagnose_missing_edge`, which runs only on edges already known to
    # be missing from the subject — an unclassifiable miss there is
    # genuinely a validator bug and does raise.)
    return failures


# =============================================================================
# MiddlePack pair-interleaving classifier
# =============================================================================
#
# TF32 emulation packs come in groups of 24, the middle 16 of which compute
# bf16 error terms via paired (v_cvt_f32_bf16, v_sub_f32) instructions
# bound to the `MiddlePack` validator dataclass. Each pair shares a
# temporary VGPR, so the two halves of every pair must appear back-to-back
# in stream order: no OTHER MiddlePack (from any category) may sit between
# a pair's leader (even index in its category's middle-16 list) and its
# consumer (the next, odd index). This works directly off the GraphNode
# stream — no cross-reference to validator-side dataclasses (would form an
# import cycle: CMSValidator imports ScheduleCapture, not the other way).
#
# Pair-detection algorithm:
#   1. Walk every node in stream order.
#   2. Filter to MiddlePack-class nodes (rocisa class registered as
#      `InstructionCategory.MIDDLE_PACK`). Group by `category` (PackA0,
#      PackB0, etc.) — categories partition the pack stream.
#   3. Within each category, pair MiddlePacks by adjacency in the
#      category-local stream order: pair (0, 1), (2, 3), (4, 5), ...
#      Each pair's first element is the LEADER; the second is the
#      CONSUMER.
#   4. For each LEADER, scan the GLOBAL MiddlePack stream and find the
#      first MiddlePack node positioned strictly after the leader. If it
#      isn't the leader's CONSUMER, emit `OverriddenInputFailure`.


def validate_middle_pack_pair_interleaving(graph: "DataflowGraph") -> List["OverriddenInputFailure"]:
    """Graph-side MiddlePack pair-interleaving check.

    Walks the unified node stream once and enforces the
    pair-leader/pair-consumer adjacency invariant for TF32 middle-16 packs.
    Emits zero-or-more `OverriddenInputFailure` for each violating pair
    (one failure per pair, never per intervening node).

    Returns a list of `OverriddenInputFailure` instances. Empty list
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
    for node in graph.all_nodes_in_order:
        inst = getattr(node, "rocisa_inst", None)
        if inst is None:
            continue
        if _category(inst) is not InstructionCategory.MIDDLE_PACK:
            continue
        # Defensive: only honor nodes whose category looks like a Pack* tag.
        # A MiddlePack rocisa class with a non-Pack category would mean the
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
        failures.append(OverriddenInputFailure(
            producer=cms_node_label(node, graph.body_for(node)),
            consumer=cms_node_label(consumer, graph.body_for(consumer)),
            iter_delta=node.iter_delta_to(consumer),
            resource="vgpr",
            intervening_writer=cms_node_label(
                actual_next, graph.body_for(actual_next),
            ),
        ))
    return failures


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


def isValid(scheduleInfo: 'ScheduleInfo', context: 'ValidationContext') -> None:
    """Validate the schedule. Returns ``None`` on success.

    Raises ``ValidationError`` on the first validation failure. The
    exception carries the typed ``Failure`` describing what went wrong;
    ``str(exc)`` yields the formatted human-readable message.

    Callers that want to handle failures gracefully wrap the call in
    ``try/except ValidationError``. Callers that do not catch propagate
    the exception upward — making it impossible to silently ignore a
    validation failure (the previous ``tuple[bool, str]`` contract
    allowed a caller to drop the bool check and miss real defects).

    Note 1: A successful return (no exception) is not proof that the
    schedule is valid. It may be a false negative.

    Note 2: A raised ``ValidationError`` is not proof that the schedule
    is invalid. It may be a false positive.
    """
    kernel = context.kernel

    context.mfma_reorder = scheduleInfo.mfmaReorder or []

    # Cross-scheduler comparison rule. Only fires when both default and CMS
    # captures are present in the context (i.e. capture was enabled when
    # this kernel was scheduled). Validates the CMS schedule against the
    # default via dataflow-graph equality + per-edge wait-coverage.
    if context.default_capture is not None and context.cms_capture is not None:
        # br4.7 moved build_dataflow_graph; br4.8 moved compare_graphs;
        # br4.9 moved validate_edge_wait_coverage into this module. All three
        # are now same-file calls — no lazy import needed.
        # Attach the per-arch quad-cycle profile derived from this kernel's
        # ISA tuple so the four pair-specific gap helpers consult arch-
        # appropriate finish-cycle / settle-window values. Unknown ISAs
        # (no ArchProfile registered in `_ARCH_PROFILES_BY_ISA`) get a
        # `None` profile back from the resolver — the resolver emits a
        # warning every time it's called with an unknown ISA; this single
        # call site keeps the warning surface to once-per-kernel-resolution.
        # `arch_profile=None` propagates through both captures and graphs;
        # the four pair-specific timing helpers short-circuit when they
        # see it (rocm-libraries-zkzw). Cross-graph diff, wait coverage,
        # SCC, and MiddlePack interleaving still run for unknown ISAs;
        # only the four pair-specific timing checks are suppressed.
        isa_tuple = tuple(kernel["ISA"]) if "ISA" in kernel else None
        arch_profile = ArchProfile.for_isa(isa_tuple)
        if context.default_capture.arch_profile is None:
            context.default_capture.arch_profile = arch_profile
        if context.cms_capture.arch_profile is None:
            context.cms_capture.arch_profile = arch_profile
        ref_graph = build_dataflow_graph(context.default_capture)
        subj_graph = build_dataflow_graph(context.cms_capture)
        graph_failures = compare_graphs(ref_graph, subj_graph)
        if graph_failures:
            summary = "\n  ".join(
                f.format()
                for f in graph_failures
            )
            raise ValidationError(message=(
                f"Dataflow graph comparison failed: "
                f"{len(graph_failures)} edge difference(s):\n  {summary}"
            ))
        wait_failures = validate_edge_wait_coverage(subj_graph)
        if wait_failures:
            summary = "\n  ".join(
                f.format()
                for f in wait_failures
            )
            raise ValidationError(message=(
                f"Wait-coverage validation failed: "
                f"{len(wait_failures)} failure(s):\n  {summary}"
            ))

    return None


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

        # findValidPositions probes a search space; an invalid candidate
        # is normal here (it just means "skip this index"), so we trap
        # the ValidationError rather than letting it propagate.
        try:
            isValid(scheduleInfo, context)
        except ValidationError:
            continue
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

