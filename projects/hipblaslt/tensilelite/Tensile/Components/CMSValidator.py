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

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum, auto
from typing import ClassVar, Optional

from rocisa.instruction import (
    SWaitCnt, SBarrier,
    MFMAInstruction, SMovB32, SAddU32,
    VPermB32, VOrB32, VLShiftLeftOrB32, VSwapB32,
    VCvtPkF32toBF16, PVCvtBF16toFP32, VCvtBF16toFP32, VSubF32, VDot2CF32BF16,
    BufferLoadB128, BufferLoadB64, BufferLoadB32,
)
from Tensile.Common.Utilities import printWarning
from Tensile.Components.ScheduleCapture import SchedulePosition


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
    (11, 5, 1): {   # RDNA 3.5 (gfx1151)
        ValidationConcern.INSTRUCTION_ORDERING,
        # Concerns below are declared but have no rules yet — they will be added in stage 12
        # (gfx1151 rules) when LWAfterLRRule, LWBeforeLRRule, GRVgprReadyRule are implemented.
        # Until then, only INSTRUCTION_ORDERING is actively validated for gfx1151.
        #
        # NOT YET ACTIVE (no rules):
        #   SCHEDULE_COMPLETENESS — CMS counts authored for CDNA wave64, not calibrated for wave32
        #   LR_DATA_READY — needs WMMA-specific LR→WMMA tracing
        #   LDS_WRITE_AFTER_READ — needs DTL=0 LW-based rule
        #   LDS_READ_AFTER_WRITE — needs DTL=0 LW-based rule
        #   LW_ORDERING — needs new LWAfterLRRule
        #   GR_VGPR_READY — needs new GRVgprReadyRule
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


@dataclass(frozen=True)
class PipelineStage:
    """Describes one loop stage in the CMS pipeline.

    Replaces the hardcoded [ML-1, ML, NGL, NLL] loop list with a model
    that generates the correct number of stages based on PGR.

    PGR=1: [ML, NLL] (2 stages)
    PGR=2: [ML-1, ML, NGL, NLL] (4 stages — same as today)
    PGR=3: [ML-2, ML-1, ML, NGL-2, NGL-1, NLL] (6 stages)
    """
    name: str
    has_global_reads: bool
    has_global_read_incs: bool
    has_local_reads_lr0_only: bool
    vlcnt_shift: int


def build_pipeline_stages(pgr: int, nglshift: int, nllshift: int) -> list[PipelineStage]:
    """Build the pipeline stage list for a given PGR value.

    Args:
        pgr:      PrefetchGlobalRead value (1, 2, or higher)
        nglshift: Total vlcnt shift for no-global-load stages
        nllshift: vlcnt shift for the no-local-load stage
    """
    stages = []
    # N main loop copies
    for i in range(pgr - 1, -1, -1):
        stages.append(PipelineStage(
            name=f"ML-{i}" if i > 0 else "ML",
            has_global_reads=True, has_global_read_incs=True,
            has_local_reads_lr0_only=False, vlcnt_shift=0))
    # N-1 NGLL drain stages
    for k in range(pgr - 1, 0, -1):
        shift = nglshift * (pgr - k) // (pgr - 1) if pgr > 1 else 0
        stages.append(PipelineStage(
            name=f"NGL-{k}" if pgr > 2 else "NGL",
            has_global_reads=False, has_global_read_incs=False,
            has_local_reads_lr0_only=False, vlcnt_shift=shift))
    # NLL
    stages.append(PipelineStage(
        name="NLL", has_global_reads=False, has_global_read_incs=False,
        has_local_reads_lr0_only=True, vlcnt_shift=nllshift))
    return stages


class ValidationRule(ABC):
    """A composable validation rule that operates on a Timeline.

    Each rule declares which ValidationConcern(s) it covers. isValid uses
    active_concerns() to decide which rules to run for a given kernel.

    The run() method may return either:
      - None  (validation passed)
      - str   (legacy: pre-migration rules emit string messages directly)
      - Failure subclass instance (migrated rules; isValid converts via
        Failure.format(capture=None) at the boundary)

    Stack 1 of plans/then-let-s-work-on-jaunty-reddy.md migrates rules
    one at a time; the union return type accommodates the transition.
    """

    @abstractmethod
    def concerns(self) -> set[ValidationConcern]:
        """Which concerns this rule covers."""
        ...

    @abstractmethod
    def run(self, timeline: 'Timeline', ctx: 'ValidationContext'):
        """Execute the rule. Return None if valid, str OR Failure if not."""
        ...


class StructuralRule(ABC):
    """A composable validation rule that operates on raw schedule data (no Timeline).

    The run() method returns a (bool, str) tuple. Migrated structural rules
    construct typed Failures internally and call .format(capture=None) to
    produce the str — keeping the tuple shape stable for the existing
    isValid contract.
    """

    @abstractmethod
    def concerns(self) -> set[ValidationConcern]:
        """Which concerns this rule covers."""
        ...

    @abstractmethod
    def run(self, schedule_info: 'ScheduleInfo', context: 'ValidationContext',
            code_path: int) -> tuple[bool, str]:
        """Execute the rule. Return (True, '') if valid, (False, message) if not."""
        ...




# Bead `ola.4` phase-2: deleted the structural-side Pack rule machinery
# (`add_pack_constraints` + `hook_up_packs` + `_hook_up_packs_*` family +
# `derive_pack_must_start_after` + `set_*_needed_by_from_mfma_operands` +
# `_set_pack_needed_by` + `find_earliest_mfma_execution` + `apply_swaits`
# + `_get_lrs_for_pack` + `_hook_up_middle_16_pairs` + `_build_reg_to_lr_map`
# + `invert_mfma_reorder` + `PackDataReadyRule`). All Pack-related
# invariants moved graph-side: LR -> Pack RAW (dscnt drain) and Pack ->
# MFMA RAW are enforced by `validate_edge_wait_coverage` +
# `compare_graphs` over real register dataflow edges produced by
# `_GenericALURule`; MiddlePack pair-interleaving (`WrongInterleavingFailure`)
# is enforced by `validate_middle_pack_pair_interleaving` (bead `dpi`);
# quad-cycle visibility (`TimingTooCloseFailure`) by `_quad_cycle_gap_ok` /
# `_cvt_to_mfma_gap_ok` / `_mfma_pack_to_cvt_gap_ok` (beads `nk0`, `35z`,
# `or9`). See migrated coverage in
# `Tensile/Tests/unit/test_validate_pack_graph.py` and
# `Tensile/Tests/unit/test_dataflow_graph_register_gaps.py`.

# --- Loop Names ---
MAIN_LOOP_PREV = "ML-1"
MAIN_LOOP = "ML"
NO_GLOBAL_LOAD_LOOP = "NGL"
NO_LOCAL_LOAD_LOOP = "NLL"

# --- Pack Group Sizes ---
PACK_GROUP_SIZE_TF32 = 24        # 4 CVT0 + 16 middle + 4 CVT1
PACK_GROUP_SIZE_TF32_4X4 = 10    # 4 CVT0 + 2 MFMA + 4 CVT1

    # TF32 Pack Index Range constants removed — type resolution via idMap replaces positional arithmetic

# --- Quad-Cycle Timing (CDNA 4 ISA section 7.6) ---
# The structural-side `estimate_quad_cycles` orchestrator and its
# accompanying constants were deleted by bead `8nz`. The graph-side
# `_quad_cycle_gap_ok` / `_cvt_to_mfma_gap_ok` / `_mfma_pack_to_cvt_gap_ok`
# helpers in `Tensile/Components/ScheduleCapture.py` are now the single
# source of truth for MFMA quad-cycle visibility verdicts; their constants
# (`_QUAD_CYCLES_*`, `_MFMA_TYPE_SWITCH_THRESHOLD_*`) live alongside them
# in ScheduleCapture.py.

# --- TF32 Emulation ---
MFMAS_PER_TILE_TF32 = 3   # 3 MFMAs per tile pair in TF32 emulation
MFMAS_PER_TILE_BF16 = 1   # 1 MFMA per tile pair in BF16

# --- VGPRs ---
VGPRS_PER_CONVERSION_GROUP = 8   # 8 VGPRs per conversion group in TF32 emulation


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

    def validate(self):
        """Bead ``ola.3`` phase-2: the LR-data-ready rule
        (``add_local_read_constraints`` + ``set_lr_needed_by_for_VMFMA``)
        was deleted. LR -> MFMA wait-coverage now lives graph-side via
        ``validate_edge_wait_coverage`` (LR -> MFMA RAW edges). The
        ``ValidatorInstruction.validate`` abstract method still requires
        an override on every subclass; this no-op preserves the contract
        without re-introducing the structural rule.
        """
        return None

@dataclass
class MFMA(ValidatorInstruction):
    # Bead `8nz`: `mfma_finish_cycles` was the only field consumed by
    # the deleted `estimate_quad_cycles` simulator. The graph-side
    # `_mfma_finish_cycles_for(rocisa_inst)` (ScheduleCapture.py:3543)
    # is now the single source of truth for per-MFMA finish-cycle costs.

    def validate(self) -> Optional[str]:
        return None

@dataclass
class Pack(ValidatorInstruction):
    """BF16 pack instructions (v_perm). Base class for all pack types.

    Bead `ola.4` phase-2: deleted the structural-side ordering rule.
    `Pack.validate` (the must_start_after/needed_by ordering check),
    `Pack.must_start_after`, `Pack.needed_by`, plus all `_hook_up_packs_*`
    plumbing were removed. Pack-related ordering invariants (LR -> Pack
    RAW dscnt drain, Pack -> MFMA RAW order, Pack -> Pack WAR ordering)
    are now enforced graph-side by `validate_edge_wait_coverage` and
    `compare_graphs` over real register dataflow edges produced by
    `_GenericALURule` (ScheduleCapture.py). See migrated coverage in
    `Tensile/Tests/unit/test_validate_pack_graph.py`.
    """
    # The index in the list of Pack instructions provided by a CMS schedule.
    # Set at construction time (the `_insert` call paths in `Timeline`). No
    # longer consumed after ola.4 phase-2 deletion of the structural rule;
    # retained as a harmless construction-time tag because the producing
    # call site assigns it unconditionally.
    issue_index: int
    # Which tile/group this pack belongs to, computed at construction time.
    # Only meaningful for TF32 subclasses (CVTPack, MiddlePack, MFMAPack); None for BF16 packs.
    group_index: Optional[int] = None

    def validate(self):
        """No-op after bead `ola.4` phase-2. The Pack-ordering rule moved
        graph-side; the ValidatorInstruction abstract method still requires
        an override on every subclass, so this preserves the contract.
        """
        return None

@dataclass
class CVTPack(Pack):
    """TF32 CVT0/CVT1 packs (v_cvt_pk_bf16_f32). Type marker for isinstance dispatch."""
    pass

@dataclass
class MiddlePack(Pack):
    """Middle-16 packs in TF32 groups of 24.

    Bead `ola.4` phase-2: `MiddlePack.validate`, `pair_consumer`, and
    `next_scheduled_middle_16` were deleted. The pair-interleaving
    invariant (`WrongInterleavingFailure` on a non-pair-consumer between
    pair-leader and pair-consumer) is now enforced graph-side via
    `validate_middle_pack_pair_interleaving` (bead `dpi`); see
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
    # Bead `8nz`: deleted the `mfma_finish_cycles` ClassVar override
    # (consumed only by the deleted `estimate_quad_cycles` simulator).
    # Graph-side `_mfma_finish_cycles_for(rocisa_inst)` distinguishes
    # 4x4 PackMFMAs from standard MFMAs by rocisa class name.


@dataclass
class GlobalRead(ValidatorInstruction):
    swap_global_read_order: bool
    needed_by: ValidatorInstruction = field(default_factory=lambda: MFMA(name="MFMA", issued_at=POSITION_INF))
    guaranteed_by: SchedulePosition = field(default_factory=lambda: POSITION_INF)
    barriered_at: list[SchedulePosition] = field(default_factory=list)
    # Bead `ola.2` phase 2: deleted `must_start_after` and
    # `must_start_after_barriered_at` fields. The LR0 -> SWait -> SBarrier ->
    # GR (LDS-reuse) ordering invariant is now enforced graph-side via
    # `lr_to_gr_lds_reuse` edges in ScheduleCapture.py
    # (`_collect_barrier_edges` + `validate_edge_wait_coverage`); the GRInc
    # -> GR (SRD ordering) invariant is enforced graph-side via the SRD
    # sgpr RAW edge formed by `_GenericALURule` and surfaced as
    # `OrderInvertedFailure` by `compare_graphs`. See migrated tests at
    # Tensile/Tests/unit/test_validate_gr_not_too_early_graph.py and
    # Tensile/Tests/unit/test_GRMustStartAfterGRInc.py.

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
        from Tensile.Components.ScheduleCapture import (
            MissingWaitFailure, MissingBarrierFailure, WaitTooLateFailure,
        )

        # If needed_by is at inf, the constraint is not active (e.g. no LR1s).
        if self.needed_by.issued_at == POSITION_INF:
            return None

        if self.issued_at < self.guaranteed_by < self.needed_by.issued_at:
            if any(self.guaranteed_by < barriered_at < self.needed_by.issued_at for barriered_at in self.barriered_at):
                    return None

        issued_at = self.issued_at.vmfma_index
        needed_by = self.needed_by.issued_at.vmfma_index

        name = self._name()

        # 1. No SWait
        if self.guaranteed_by == POSITION_INF:
            return MissingWaitFailure(
                producer=self, consumer=self.needed_by, counter_kind="vlcnt",
            )

        # NOTE: Must do it after the check above to guard against infinity.
        guaranteed_by = self.guaranteed_by.vmfma_index

        # 2. No Barrier
        if len(self.barriered_at) == 0:
            return MissingBarrierFailure(
                producer=self, consumer=self.needed_by, role="needed_by",
            )

        # 3. Guaranteed after needed
        if self.guaranteed_by > self.needed_by.issued_at:
            return WaitTooLateFailure(
                producer=self, consumer=self.needed_by, wait_position=self.guaranteed_by,
            )

        # 4. No Barrier between SWait and LR1
        if not any(self.guaranteed_by < barriered_at < self.needed_by.issued_at for barriered_at in self.barriered_at):
            return MissingBarrierFailure(
                producer=self, consumer=self.needed_by, role="needed_by",
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
        from Tensile.Components.ScheduleCapture import InvalidCounterValueFailure
        return InvalidCounterValueFailure(
            swait=self,
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


# --- Register Utilities for rocisa-based dependency analysis ---

def _parse_reg_name(reg_name_str: str) -> tuple[str, int]:
    """Parse a register name string like 'ValuA_X0_I0+12' into (base_name, offset).

    Returns:
        Tuple of (base_name, offset). If no '+' offset, offset is 0.
        E.g. 'ValuA_X0_I0+12' → ('ValuA_X0_I0', 12)
             'ValuA_X0_I0' → ('ValuA_X0_I0', 0)
    """
    parts = str(reg_name_str).split('+')
    base = parts[0]
    offset = int(parts[1]) if len(parts) > 1 else 0
    return base, offset


def get_reg_range(rc) -> Optional[tuple[str, int, int]]:
    """Extract the register identity range from a RegisterContainer.

    Returns:
        Tuple of (base_name, start_offset, end_offset_exclusive), or None if
        the container is not a VGPR register or cannot be resolved.

        For numeric registers (regIdx >= 0): base_name is the regType string,
        start/end are the numeric indices.

        For named registers (regName set): base_name is the symbolic name,
        start/end are the offsets within that name.

    Examples:
        v[10:13] → ('v', 10, 14)
        v[vgprValuA_X0_I0+12:+15] → ('ValuA_X0_I0', 12, 16)
    """
    if rc is None:
        return None

    reg_num = int(rc.regNum)
    if reg_num <= 0:
        return None

    if rc.regIdx >= 0:
        # Numeric register
        return (rc.regType, rc.regIdx, rc.regIdx + reg_num)

    if rc.regName is not None:
        # Named register with symbolic name
        base, offset = _parse_reg_name(rc.regName)
        return (base, offset, offset + reg_num)

    return None


def reg_ranges_overlap(a: tuple[str, int, int], b: tuple[str, int, int]) -> bool:
    """Check if two register ranges overlap.

    Both ranges must have the same base name (register file) to overlap.
    Ranges are [start, end) — half-open intervals.
    """
    if a[0] != b[0]:
        return False
    return a[1] < b[2] and b[1] < a[2]


def get_dst_range(rocisa_inst) -> Optional[tuple[str, int, int]]:
    """Extract the destination register range from a rocisa instruction.

    Handles CommonInstruction (.dst), MFMAInstruction (.acc), and
    CompositeInstruction (.dst) via attribute inspection.
    """
    # MFMAInstruction: destination is the accumulator
    if hasattr(rocisa_inst, 'acc') and rocisa_inst.acc is not None:
        return get_reg_range(rocisa_inst.acc)
    # CommonInstruction and others
    if hasattr(rocisa_inst, 'dst') and rocisa_inst.dst is not None:
        return get_reg_range(rocisa_inst.dst)
    return None


def get_src_ranges(rocisa_inst) -> list[tuple[str, int, int]]:
    """Extract all source register ranges from a rocisa instruction.

    Returns a list of register ranges for all source operands that are
    RegisterContainer objects (skips immediates and string operands).
    """
    ranges = []
    if hasattr(rocisa_inst, 'srcs'):
        for src in rocisa_inst.srcs:
            rng = get_reg_range(src)
            if rng is not None:
                ranges.append(rng)
    # MFMAInstruction: .a and .b are sources
    if hasattr(rocisa_inst, 'a') and rocisa_inst.a is not None:
        rng = get_reg_range(rocisa_inst.a)
        if rng:
            ranges.append(rng)
    if hasattr(rocisa_inst, 'b') and rocisa_inst.b is not None:
        rng = get_reg_range(rocisa_inst.b)
        if rng:
            ranges.append(rng)
    return ranges


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
        """
        Create a timeline from the provided schedule_info which contains only the instructions inside `instruction_names_to_add`.

        Args:
            instruction_names_to_add:   The list of instruction names to add to the timeline.
            code_path:                  The code path to create a timeline out of.
            schedule_info:              The schedule information to add to the timeline.
            kernel:                     The kernel to add to the timeline.
            id_map:                     Maps schedule keys to lists of rocisa instruction objects.
            mfma_code:                  Flat list of MFMA rocisa instructions in execution order.
        """
        
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
        """
        Populates all timelines with instructions from schedule_info.
        """
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
        """
        Determine if an instruction should be added to a given loop.
        """
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
        """
        Return the names of all instructions scheduled in the timeline.
        """
        return list(self._instructions_for_name_combined.keys())

    def get_instructions(self, name: str, loop: str) -> list[tuple[int, ValidatorInstruction]]:
        """
        Return the instructions scheduled with a given name (e.g. "GRA").
        """
        return self._instructions_for_name[loop][name]
    
    def get_instructions_combined(self, name: str) -> list[tuple[int, ValidatorInstruction]]:
        """
        Return the instructions scheduled with a given name (e.g. "GRA") across all loops.
        """
        return self._instructions_for_name_combined[name]

    def get_instructions_at(self, index: int, loop: str) -> list[ValidatorInstruction]:
        """
        Return the instructions scheduled at a given VMFMA index.
        """
        return self._instructions_at_index[loop][index+1]

    def _linearize_timeline(self) -> None:
        """
        Generate the linear timelines and the lookup tables for instructions by name.
        """
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


def applies_only_once(func):
    """Decorator: skips the function if it has already been applied to this timeline."""
    @functools.wraps(func)
    def wrapper(timeline, *args, **kwargs):
        if func in timeline._applied_passes:
            return
        result = func(timeline, *args, **kwargs)
        timeline._applied_passes.add(func)
        return result
    return wrapper


# `apply_barriers` was deleted as a follow-up cleanup after ola.4 phase 2.
# Its only consumers were `add_gr_finish_before_lr_constraints` and
# `add_pack_constraints` (both deleted in ola.1 / ola.4). The barriered_at
# bookkeeping it produced is now derived graph-side via `_collect_barrier_edges`
# in ScheduleCapture.py (`gr_to_lr_lds_reuse` edge_kind).


# `apply_must_start_after_barriers` and its `_apply_must_start_after_barriers_single`
# helper were deleted in bead `ola.2` phase 2. Their only consumer was
# `add_gr_not_too_early_constraints` (also deleted), which encoded the
# LR0 -> SWait(dscnt=0) -> SBarrier -> GR LDS-reuse invariant. That
# invariant is now enforced graph-side via `lr_to_gr_lds_reuse` edges in
# ScheduleCapture.py (`_collect_barrier_edges` + `validate_edge_wait_coverage`).




# `set_lr_needed_by_for_VMFMA` was deleted in bead `ola.3` phase-2.
# It was the consumer-MFMA wiring helper for the now-deleted
# `add_local_read_constraints` rule. LR -> MFMA wait-coverage is
# enforced graph-side: `_DSLoadRule` (writes vgpr) + `_MFMARule`
# (reads vgpr) + per-byte latest-writer resolver build LR -> MFMA
# RAW edges; `validate_edge_wait_coverage` then emits
# `MissingWaitFailure(counter_kind='dscnt')` /
# `WaitInsufficientFailure(counter_kind='dscnt')` for any uncovered
# edge. See `Tensile/Tests/unit/test_validate_lr_before_mfma_graph.py`
# for the migrated coverage.


# `set_gr_needed_by_from_lrs` was removed in bead `ola.1`. Its only
# caller was `add_gr_finish_before_lr_constraints`, also removed. The
# GR -> LR1/3 LDS-reuse coverage is now graph-side; see
# `_collect_barrier_edges` and `validate_edge_wait_coverage` in
# ScheduleCapture.py.

# `set_gr_must_start_after_from_lr0s` and `set_gr_must_start_after_from_grinc`
# were deleted in bead `ola.2` phase 2. Their only consumer was
# `add_gr_not_too_early_constraints` (also deleted). The LR0 -> GR LDS-reuse
# invariant is now graph-side via `lr_to_gr_lds_reuse` edges
# (validate_edge_wait_coverage). The GRInc -> GR SRD ordering invariant is
# graph-side via the SRD sgpr RAW edge from the GRInc's SAddU32 to the GR's
# BufferLoad — `compare_graphs` flips a reversed-order subject into
# OrderInvertedFailure (proven by test_GRMustStartAfterGRInc.py).




# Bead `8nz`: `_handle_min_pack_quad_cycles` was deleted. It only set
# `Pack.min_quad_cycles_before_result_used`, which itself was only read
# by the deleted `estimate_quad_cycles` simulator and the deleted
# timing-check block in `Pack.validate`. Quad-cycle visibility is now
# enforced by the graph-side helpers in ScheduleCapture.py.

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















# Bead `8nz`: `estimate_quad_cycles` (the structural-side issue-time
# simulator) was deleted. Quad-cycle visibility (CDNA 4 ISA section 7.6)
# is now enforced exclusively by the graph-side helpers in
# Tensile/Components/ScheduleCapture.py:
#   * `cumulative_issue_cycles(graph, producer, consumer)` — cycle-exact
#     equivalent of the deleted simulator (bead `nk0`).
#   * `_quad_cycle_gap_ok` / `_cvt_to_mfma_gap_ok` /
#     `_mfma_pack_to_cvt_gap_ok` — pair-specific dispatch, all routed
#     through `_classify_edge_coverage`, all emitting `TimingTooCloseFailure`.
# See `Tensile/Tests/unit/test_dataflow_graph_register_gaps.py` for the
# graph-native coverage of the rules previously pinned by the (now
# removed) `TimingTooClose*` tests in `test_ValidatePack.py`.

def _failure_to_string(result) -> Optional[str]:
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
    """
    Validate the timeline by calling the validate method of each instruction.

    Side effect: stashes the raw Failure object on `timeline._last_failure`
    (or None on success) so test infrastructure can assert on type + fields
    without parsing the formatted string. Production callers consume only
    the returned string.

    Args:
        timeline: The Timeline object to validate.

    Returns:
        Error message if validation fails, None if validation passes.
    """
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
    """
    Helper function to get the schedule for a given instruction name and code path.
    When multiple code paths are provided, return the schedule for the given code path.
    If only one code path is implemented, return that schedule.

    Args:
        name: The name of the instruction to get the schedule for (e.g. "LRA0", "LRB0", "SYNC")
        code_path: The code path to get the schedule for (0-indexed)
        schedule_info: The schedule information (ScheduleInfo object)

    Returns:
        The schedule for the given instruction name and code path.
    """
    assert code_path >= 0, f"Code path {code_path} is not valid. Must be >= 0."
    schedules = schedule_info.optSchedule[name]
    return schedules[0] if len(schedules) == 1 else schedules[code_path]


# `_transform_index_with_force_unroll_sub_iter`,
# `_transform_index_standard`, and `lr_needed_by_mfma` were deleted in
# bead `ola.3` phase-2 alongside `set_lr_needed_by_for_VMFMA` /
# `add_local_read_constraints` / `index_for_force_unroll_sub_iter`.
# These were the positional tile-index -> MFMA-index helpers used by
# the old structural rule. Graph-side LR -> MFMA edges are derived
# from real register dataflow (`_DSLoadRule` writes vgpr; `_MFMARule`
# reads vgpr), so no positional heuristic is needed.


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


# Keep backward-compatible alias during transition
ValidatorPassContext = ValidationContext


# `add_local_read_constraints` was deleted in bead `ola.3` phase-2.
# It was the structural rule that wired LR.needed_by from a positional
# tile-index heuristic (`set_lr_needed_by_for_VMFMA`) and then drove
# `LocalRead.validate` through the now-deleted `apply_swaits` + `apply_barriers`.
# The graph-side replacement is `validate_edge_wait_coverage` over
# LR -> MFMA RAW edges (`_DSLoadRule` writes vgpr; `_MFMARule` reads
# vgpr). Migrated coverage lives in
# `Tensile/Tests/unit/test_validate_lr_before_mfma_graph.py`.




# `add_gr_not_too_early_constraints` was deleted in bead `ola.2` phase 2.
# It encoded two invariants:
#   * LR0 -> SWait(dscnt=0) -> SBarrier -> GR (LDS-reuse) — now graph-side
#     via `lr_to_gr_lds_reuse` edges (validate_edge_wait_coverage). Phase 1
#     of ola.2 migrated parallel coverage in test_validate_gr_not_too_early_graph.py.
#   * last GRInc<X> -> first GR<X> (intra-wave SRD ordering) — now graph-side
#     via the SRD sgpr RAW edge published by `_GenericALURule`; reversed
#     subjects surface as `OrderInvertedFailure` in `compare_graphs`. See
#     Tensile/Tests/unit/test_GRMustStartAfterGRInc.py for the migrated
#     coverage.


# `add_gr_finish_before_lr_constraints` was removed in bead `ola.1`.
# The legacy rule encoded GR -> SWait(vlcnt) -> SBarrier -> LR1/3
# LDS-reuse coverage; this is now graph-side (`gr_to_lr_lds_reuse`
# edges in ScheduleCapture._collect_barrier_edges +
# validate_edge_wait_coverage). See migrated tests at
# Tensile/Tests/unit/test_ValidateGRsCompleteBeforeLr{1,3}s.py.


# `index_for_force_unroll_sub_iter` was deleted in bead `ola.3` phase-2.
# It mapped column-major tile indices into the ForceUnrollSubIter
# block-permuted scheme used by the deleted `lr_needed_by_mfma`
# helper. Graph-side LR -> MFMA edges read from real register dataflow
# and are invariant under MFMA-slot permutations (whether by
# ForceUnrollSubIter or mfma_reorder), so no positional remapping is
# needed.


# `verify_correct_number_of_instructions` + `InstructionCountRule` were
# deleted in bead `4tw`. The rule guarded against
# `len(optSchedule[K]) != len(id_map[K])` drift, which would have caused
# the (now-empty) TIMELINE_RULES path to silently truncate instructions
# via `min(len_a, len_b)`. With ola.1-4 closing out every Timeline
# consumer, the drift modes the rule guarded are now caught upstream
# and downstream of `isValid()`:
#   * `optSchedule[K]` references too-large index → `IndexError` at
#     `CustomSchedule.scheduleInst()` (CustomSchedule.py:410) with a
#     clear "stream X[i] references instruction index N but only M
#     instructions exist" message.
#   * `optSchedule[K]` shorter than `id_map[K]` → fewer rocisa
#     instructions emitted into the CMS macro → `compare_graphs` Phase 0
#     in `KernelWriter.py:5225` raises `CaptureConsistencyError` on the
#     resulting data-flow node identity mismatch between default and CMS.
# See `Tensile/Tests/unit/test_ValidateInstructionCount.py` (deleted in
# the same bead) for the full coverage matrix the old rule pinned.


# `LRDataReadyRule` was deleted in bead `ola.3` phase-2. The
# LR-data-ready concern (LR -> MFMA dscnt drain) is enforced graph-side
# by `validate_edge_wait_coverage` over LR -> MFMA RAW edges. The
# `ValidationConcern.LR_DATA_READY` enum value is preserved for
# downstream compatibility but no rule currently dispatches on it.




# `GRAfterLRRule` and its underlying `add_gr_not_too_early_constraints`
# were deleted in bead `ola.2` phase 2. The rule's two invariants
# (LR0 -> SWait -> SBarrier -> GR LDS-reuse coverage AND last GRInc<X>
# before first GR<X> SRD ordering) are now both enforced graph-side; see
# the comment block above `add_gr_not_too_early_constraints`'s former
# location and the migrated tests at
# Tensile/Tests/unit/test_validate_gr_not_too_early_graph.py and
# Tensile/Tests/unit/test_GRMustStartAfterGRInc.py.


# `GRBeforeLRRule` and its underlying `add_gr_finish_before_lr_constraints` /
# `set_gr_needed_by_from_lrs` were removed in bead `ola.1`. The rule's
# semantics (GR -> SWait(vlcnt) -> SBarrier -> LR1/3 LDS-reuse coverage)
# are now enforced graph-side by `_collect_barrier_edges` (edge_kind
# `gr_to_lr_lds_reuse`) + `validate_edge_wait_coverage` /
# `diagnose_missing_edge` in ScheduleCapture.py. See test_ValidateGRsComplete
# BeforeLr1s.py / Lr3s.py / test_ValidateNglAndNll.py for the migrated
# coverage.


TIMELINE_RULES: list[ValidationRule] = [
    # `LRDataReadyRule()` removed in bead `ola.3` phase-2 — see comment above.
    # `PackDataReadyRule()` removed in bead `ola.4` phase-2; the Pack ordering
    # invariants moved graph-side. The list is now empty; isValid still
    # iterates it as a no-op so the dispatch contract stays stable.
    # `GRAfterLRRule()` deleted in bead `ola.2` phase 2 — see comment block above.
]

STRUCTURAL_RULES: list[StructuralRule] = [
    # `InstructionCountRule()` deleted in bead `4tw` — see comment block
    # above the former `verify_correct_number_of_instructions` location.
    # The list is now empty; isValid still iterates it as a no-op so the
    # dispatch contract stays stable.
]


def format_kernel_string(kernel: 'Solution') -> str:
    """Format a human-readable description of the kernel's tile dimensions and transpose modes."""
    mt0 = kernel.get("MacroTile0", "?")
    mt1 = kernel.get("MacroTile1", "?")
    du = kernel.get("DepthU", "?")
    transA = "T" if kernel.get("TransA") else "N"
    transB = "T" if kernel.get("TransB") else "N"
    return f"MT0xMT1xDepthU = {mt0}x{mt1}x{du} {transA}{transB}"


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

    # Set mfma_reorder from schedule info
    context.mfma_reorder = scheduleInfo.mfmaReorder or []

    # Determine required concerns from kernel config + ISA catalog.
    required = active_concerns(kernel, context.id_map)

    # Unknown/missing ISA (not in catalog) → run all rules (legacy behavior).
    # This ensures tests and kernels without ISA metadata get full validation.
    if not required:
        required = set(ValidationConcern)

    for code_path in range(scheduleInfo.numCodePaths):
        # === Structural rules (no Timeline needed) ===
        for rule in STRUCTURAL_RULES:
            if not (rule.concerns() & required):
                continue
            # Reset typed-Failure side-channel before each rule so a previous
            # rule's stale Failure can't leak into this iteration's diagnostics.
            context._last_failure = None
            status, message = rule.run(scheduleInfo, context, code_path)
            if not status:
                scheduleInfo.pretty_print()
                return False, f"Code path {code_path}: {message}"

        # === Timeline rules ===
        # Skip timeline construction entirely when no timeline rule is needed.
        # This avoids hitting CDNA-4 layout assertions for ISAs (e.g. gfx1151)
        # whose timeline concerns are not yet covered by any rule.
        timeline_needed = any(r.concerns() & required for r in TIMELINE_RULES)
        if not timeline_needed:
            continue

        timeline = create_unified_timeline(
            scheduleInfo, kernel, code_path,
            id_map=context.id_map,
            mfma_code=context.mfma_code,
        )

        for rule in TIMELINE_RULES:
            if not (rule.concerns() & required):
                continue
            result = rule.run(timeline, context)
            error = _failure_to_string(result)
            if error:
                scheduleInfo.pretty_print()
                return False, f"Code path {code_path}: {error}"

    # Cross-scheduler comparison rule. Only fires when both default and CMS
    # captures are present in the context (i.e. capture was enabled when
    # this kernel was scheduled). Validates the CMS schedule against the
    # default via dataflow-graph equality + per-edge wait-coverage.
    if context.default_capture is not None and context.cms_capture is not None:
        from Tensile.Components.ScheduleCapture import (
            build_dataflow_graph, compare_graphs, validate_edge_wait_coverage,
        )
        ref_graph = build_dataflow_graph(context.default_capture)
        subj_graph = build_dataflow_graph(context.cms_capture)
        graph_failures = compare_graphs(
            ref_graph, subj_graph, raise_on_unexplained=False,
        )
        if graph_failures:
            summary = "\n  ".join(
                f.format(context.cms_capture.main_loop.get(0))
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
                f.format(context.cms_capture.main_loop.get(0))
                for f in wait_failures
            )
            return False, (
                f"Wait-coverage validation failed: "
                f"{len(wait_failures)} failure(s):\n  {summary}"
            )

    # All rules passed, considered valid.
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

