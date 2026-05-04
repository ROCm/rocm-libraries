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


def invert_mfma_reorder(mfma_reorder: list[int]) -> dict[int, int]:
    """
    Compute the inverse mapping of mfmaReorder.
    
    The mfmaReorder array has semantics: mfmaReorder[new_position] = original_position.
    This means the MFMA that was originally at index `original_position` will be
    executed at `new_position` after reordering.
    
    This function returns the inverse: original_position -> new_position (execution index).
    Use this when you have an original/logical MFMA index and need to find when it executes.
    
    Args:
        mfma_reorder: List where mfma_reorder[new_pos] = original_pos
        
    Returns:
        Dictionary mapping original_position -> new_position (execution index)
    """
    return {orig: new_pos for new_pos, orig in enumerate(mfma_reorder)}


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
    """BF16 pack instructions (v_perm). Base class for all pack types."""
    # The index in the list of Pack instructions provided by a CMS schedule.
    # Needed to properly calculate needed_by and must_start_after.
    issue_index: int
    # Which tile/group this pack belongs to, computed at construction time.
    # Only meaningful for TF32 subclasses (CVTPack, MiddlePack, MFMAPack); None for BF16 packs.
    group_index: Optional[int] = None
    needed_by: ValidatorInstruction = field(default_factory=lambda: MFMA(name="MFMA", issued_at=POSITION_INF))
    must_start_after: list[ValidatorInstruction] = field(default_factory=list)

    # Bead `8nz`: `min_quad_cycles_before_result_used` /
    # `estimated_quad_cycles_before_result_used` (and the timing-check
    # block below) were deleted. Quad-cycle visibility is now enforced
    # exclusively by the graph-side `_quad_cycle_gap_ok` /
    # `_cvt_to_mfma_gap_ok` / `_mfma_pack_to_cvt_gap_ok` helpers in
    # ScheduleCapture.py, which emit the same `TimingTooCloseFailure`
    # type via `_classify_edge_coverage`.

    def validate(self):
        """Stack 1.3: returns typed Failure or None."""
        from Tensile.Components.ScheduleCapture import ConstraintViolationFailure
        issued_at = self.issued_at.vmfma_index

        # Collapse must_start_after list to the single latest constraint
        effective_must_start_after = max(
            self.must_start_after, key=lambda c: c.done_idx()
        ) if self.must_start_after else MFMA(name="MFMA", issued_at=POSITION_NEG_INF)

        if effective_must_start_after.done_idx() < self.issued_at < self.needed_by.done_idx():
            return None  # Ordering checks passed; quad-cycle timing now graph-side.
        if self.issued_at < effective_must_start_after.done_idx():
            # Issued too early
            return ConstraintViolationFailure(
                producer=effective_must_start_after, consumer=self,
            )
        if self.issued_at >= self.needed_by.issued_at:
            # Issued too late
            return ConstraintViolationFailure(
                producer=self, consumer=self.needed_by,
            )
        # Generic fallback — defensive guard, should not fire on valid
        # rule logic. Stays as a string return (no semantic Failure type).
        return f"{self.name} at index {issued_at} is not valid."

@dataclass
class CVTPack(Pack):
    """TF32 CVT0/CVT1 packs (v_cvt_pk_bf16_f32). Type marker for isinstance dispatch."""
    pass

@dataclass
class MiddlePack(Pack):
    """Middle-16 packs in TF32 groups of 24. Have pair constraints for shared temp VGPR."""
    pair_consumer: Optional['MiddlePack'] = None
    next_scheduled_middle_16: Optional['MiddlePack'] = None

    def validate(self):
        """Stack 1.3: returns typed Failure or None."""
        from Tensile.Components.ScheduleCapture import WrongInterleavingFailure
        error = super().validate()
        if error:
            return error
        if self.pair_consumer:
            assert self.next_scheduled_middle_16, "Pair leader must have a next_middle_16_in_schedule."
            if not (self.next_scheduled_middle_16 is self.pair_consumer):
                return WrongInterleavingFailure(
                    pack=self,
                    expected_next=self.pair_consumer,
                    actual_next=self.next_scheduled_middle_16,
                )
        return None

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


@applies_only_once
def apply_barriers(timeline: Timeline) -> None:
    """
    Apply the effect of SBarriers to the GlobalReads in the timeline by updating the barriered_at field of GlobalReads.
    Timeline is modified in place.
    
    Args:
        timeline: The Timeline object containing the instructions.
    """
    for i_barrier, barrier in timeline.get_instructions_combined("SBarrier"):
        for i_inst in range(i_barrier-1, -1, -1):
            instruction = timeline.combined_timeline[i_inst]
            if not isinstance(instruction, GlobalRead):
                continue
            if instruction.barriered_at and barrier.issued_at >= instruction.needed_by.issued_at:
                # Note: Cannot break since we can't say anything about the relationship 
                #       of `GR.needed_by` between GRs based on the order they're encountered.
                continue
            instruction.barriered_at.append(barrier.issued_at)


# `apply_must_start_after_barriers` and its `_apply_must_start_after_barriers_single`
# helper were deleted in bead `ola.2` phase 2. Their only consumer was
# `add_gr_not_too_early_constraints` (also deleted), which encoded the
# LR0 -> SWait(dscnt=0) -> SBarrier -> GR LDS-reuse invariant. That
# invariant is now enforced graph-side via `lr_to_gr_lds_reuse` edges in
# ScheduleCapture.py (`_collect_barrier_edges` + `validate_edge_wait_coverage`).


@applies_only_once
def apply_swaits(timeline: Timeline) -> None:
    """
    Apply the effect of SWaitCnts to the timeline by updating the guaranteed_by field of LocalReads and GlobalReads.
    Timeline is modified in place.
    
    Args:
        timeline: The Timeline object containing the instructions.
    """
    def apply(timeline_list: list[ValidatorInstruction], swait: SWait, ReadClazz: type, num_left_in_flight: int) -> None:
        for instruction in timeline_list:
            if not isinstance(instruction, ReadClazz):
                continue
            if num_left_in_flight > 0:
                num_left_in_flight -= 1
                continue
            if swait.issued_at >= instruction.guaranteed_by:
                # If this SWaitCnt is already guaranteed, then all earlier LRs/GRs before it are also guaranteed by here.
                break
            instruction.guaranteed_by = swait.issued_at
    
    for i_swait, swait in timeline.get_instructions_combined("SWaitCnt"):
        if i_swait == 0:
            # This is an SWaitCnt issued first thing in a schedule, there are no instructions before it in this iteration.
            # Next iteration, this same SWaitCnt will have LRs/GRs to act on.
            continue
        if swait.dscnt != -1:
            apply(timeline.combined_timeline[i_swait-1::-1], swait, LocalRead, swait.dscnt)
        if swait.vlcnt != -1:
            apply(timeline.combined_timeline[i_swait-1::-1], swait, GlobalRead, swait.vlcnt)


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


def find_earliest_mfma_execution(
    is_pack_B: bool,
    tile_index: int,
    mfma_in_tile: int,
    base_offset: int,
    n_a_tiles: int,
    n_b_tiles: int,
    mfma_reorder: list[int],
    mfmas_per_tile: int = 3,
) -> int:
    """
    Find the earliest MFMA execution index that uses a Pack's output.
    
    MFMAs form a 2D grid of (a_tile, b_tile) pairs, stored column-major (A contiguous).
    Each tile pair may have multiple MFMAs (3 for TF32, 1 for BF16).
    With MFMA reordering, a Pack's data may be used by multiple MFMAs (one per opposite tile),
    interleaved in complex ways.
    This function finds the one that executes first.
    
    Args:
        is_pack_B: True if this is a PackB, False for PackA.
        tile_index: Which tile this Pack prepares data for (B tile if is_pack_B, else A tile).
        mfma_in_tile: Which MFMA within the tile group (0 for BF16; 0, 1, or 2 for TF32).
        base_offset: Base MFMA index offset (e.g., for iteration quarter or half).
        num_a_tiles: Number of A tiles.
        num_b_tiles: Number of B tiles.
        mfma_reorder: MFMA reordering list where mfma_reorder[new_pos] = original_pos, or empty if no reordering.
        mfmas_per_tile: Number of MFMAs per tile pair (1 for BF16, 3 for TF32). Defaults to 3.
    
    Returns:
        The earliest execution index among all MFMAs that use this Pack's output.
    """
    # Column-major layout: A tiles are contiguous, B tiles are strided
    a_tile_stride = mfmas_per_tile
    b_tile_stride = n_a_tiles * mfmas_per_tile
    
    def tile_to_logical_mfma(a_tile: int, b_tile: int) -> int:
        """Convert (a_tile, b_tile) to logical MFMA index."""
        return base_offset + a_tile * a_tile_stride + b_tile * b_tile_stride + mfma_in_tile
    
    # Without MFMA reordering, logical index == execution index.
    # The first MFMA in the tile is always the earliest consumer.
    if not mfma_reorder:
        if is_pack_B:
            return tile_to_logical_mfma(a_tile=0, b_tile=tile_index)
        else:
            return tile_to_logical_mfma(a_tile=tile_index, b_tile=0)
    
    # With reordering, search all MFMAs that use this Pack's output to find the earliest.
    # mfma_reorder[new_pos] = original_pos, so we need the inverse to find execution position.
    inverse = invert_mfma_reorder(mfma_reorder)
    if is_pack_B:
        # PackB prepares B tile data, used by MFMAs: (A0, Bi), (A1, Bi), ... for all A tiles
        return min(
            inverse[tile_to_logical_mfma(a_tile, tile_index)]
            for a_tile in range(n_a_tiles)
        )
    else:
        # PackA prepares A tile data, used by MFMAs: (Ai, B0), (Ai, B1), ... for all B tiles
        return min(
            inverse[tile_to_logical_mfma(tile_index, b_tile)]
            for b_tile in range(n_b_tiles)
        )

def _set_pack_needed_by(packs: list[Pack], pack_name: str, i_loop: int, mfma_reorder: list[int], mfma_for_linear_index: dict[int, MFMA], num_vmfma: int, kernel: 'Solution') -> None:
    """
    Set the needed_by field for Pack instructions.
    This function handles all cases (BF16 and TF32).
    
    For BF16:
        - The packs are only ever needed by the VMFMA instructions.
    For regular TF32:
        - The first and last 4 packs are needed by the VMFMA instructions.
          There is a minimum number of quad-cycle restriction on the spacing between these packs and their VMFMAs.
        - The middle-16 packs are handled implicitly.
    For 4x4 MFMA TF32: 
        - The first 4 packs are needed by the 5th and 6th packs (which are VMFMAs) as well as the regular VMFMs.
          Both must be accounted, and both are subject to a minimum number of quad-cycle spacing restrictions.
        - The 5th and 6th packs (middle 2) are needed by the last 4 packs.
          These are subject to a minimum number of quad-cycle spacing restrictions.
        - The last 4 packs are needed by regular VMFMs.
          These are subject to a minimum number of quad-cycle spacing restrictions.
    
    Args:
        packs: List of Pack instructions to set needed_by for.
        pack_name: The name of the pack (e.g., "PackA0", "PackB1").
        i_loop: The loop index (0 for MAIN_LOOP_PREV, 1 for MAIN_LOOP, etc.).
        mfma_reorder: The reordering mapping for MFMA indices.
        mfma_for_linear_index: Dictionary mapping linear MFMA indices to MFMA instructions.
        num_vmfma: The number of MFMAs per iteration (not total across loops).
        kernel: The kernel class containing metadata.
    """
    # SwapPacks don't have a meaningful needed_by MFMA (they feed into CVT0, not directly into MFMAs).
    packs = [p for p in packs if not isinstance(p, SwapPack)]
    if not packs:
        return

    force_unroll_sub_iter = kernel.get("ForceUnrollSubIter", False)
    is_tf32_emulation = kernel.get("UseF32XEmulation", False)
    is_4x4mfma_tf32 = kernel.get("UseMFMAF32XEmulation", False)
    is_pack_B = pack_name.startswith("PackB")
    use_plr_pack = kernel.get("UsePLRPack", False)
    n_tiles_a = kernel["MIWaveTileA"]
    n_tiles_b = kernel["MIWaveTileB"]
    
    # Calculate needed_by_offset based on pack type and configuration
    pack_0 = pack_name.endswith("0")
    needed_by_offset = num_vmfma * i_loop
    if force_unroll_sub_iter:
        if pack_0:
            if pack_name.startswith("PackA"):
                # Needed for 2nd quarter
                needed_by_offset += num_vmfma // 4
            else:
                # Needed for 3rd quarter
                needed_by_offset += num_vmfma // 2
        else:  # Pack3
            # Both A and B are needed for 1st quarter, the flag impacts whether it's this iteration's or next iteration's 1st quarter.
            if use_plr_pack:
                needed_by_offset += num_vmfma
    else:
        if pack_0:
            needed_by_offset += num_vmfma // 2
        else:
            if use_plr_pack:
                needed_by_offset += num_vmfma
    
    # Extract iteration offset from needed_by_offset to apply mfma_reorder correctly
    # mfma_reorder only applies within a single iteration
    iteration_offset = (needed_by_offset // num_vmfma) * num_vmfma
    base_offset = needed_by_offset % num_vmfma
    
    if not is_tf32_emulation:
        # BF16 case: 1 MFMA per tile pair
        # Calculate packs_per_tile dynamically based on actual pack count
        n_tiles = n_tiles_b if is_pack_B else n_tiles_a
        packs_per_tile = len(packs) // n_tiles
        
        for pack in packs:
            # Determine which tile this pack belongs to
            tile_index = pack.issue_index // packs_per_tile
            
            execution_index = find_earliest_mfma_execution(
                is_pack_B=is_pack_B,
                tile_index=tile_index,
                mfma_in_tile=0,  # BF16 has only 1 MFMA per tile
                base_offset=base_offset,
                n_a_tiles=n_tiles_a,
                n_b_tiles=n_tiles_b,
                mfma_reorder=mfma_reorder,
                mfmas_per_tile=MFMAS_PER_TILE_BF16,  # BF16: 1 MFMA per tile pair
            )
            
            # Add iteration offset to get final position
            needed_by = iteration_offset + execution_index
            pack.needed_by = mfma_for_linear_index[needed_by]
        return

    if is_4x4mfma_tf32:
        # TF32 4x4 MFMA: Packs come in groups of 10
        # CVT0 packs feed into MFMAPacks, MFMAPacks feed into CVT1 packs
        # CVT0 and CVT1 packs also feed into external MFMAs

        # Half tile count since each quarter uses half of the A tiles and half of the B tiles.
        n_tiles_a //= 2
        n_tiles_b //= 2

        packs = sorted(packs, key=lambda x: x.issue_index)
        # Group packs by group_index (computed at construction time)
        groups: dict[int, list[Pack]] = defaultdict(list)
        for pack in packs:
            groups[pack.group_index].append(pack)

        for group_index, group_packs in sorted(groups.items()):
            # Separate by type within each group
            cvt_packs = [p for p in group_packs if isinstance(p, CVTPack)]
            mfma_packs = [p for p in group_packs if isinstance(p, MFMAPack)]
            assert len(cvt_packs) == 8, f"{packs[0].name}: Expected 8 CVT packs per group, got {len(cvt_packs)}"
            assert len(mfma_packs) == 2, f"{packs[0].name}: Expected 2 MFMA packs per group, got {len(mfma_packs)}"
            # CVT0 come before CVT1 by construction order (sorted by issue_index)
            cvt0 = cvt_packs[:4]
            cvt1 = cvt_packs[4:]
            assert cvt0[-1].issue_index < cvt1[0].issue_index, f"{packs[0].name}: CVT0 packs must have lower issue_index than CVT1 packs"

            # CVT0 → MFMAPack inter-pack dependencies
            # Packs 0 and 1 are needed by first 4x4 MFMA
            # Packs 2 and 3 are needed by second 4x4 MFMA
            cvt0[0].needed_by = mfma_packs[0]
            cvt0[1].needed_by = mfma_packs[0]
            cvt0[2].needed_by = mfma_packs[1]
            cvt0[3].needed_by = mfma_packs[1]

            # MFMAPack → CVT1 inter-pack dependencies
            mfma_packs[0].needed_by = cvt1[2]
            mfma_packs[1].needed_by = cvt1[0]

            # External MFMA needed_by for CVT0 packs (all share the same MFMA target)
            cvt0_earliest = find_earliest_mfma_execution(
                is_pack_B=is_pack_B,
                tile_index=group_index,
                mfma_in_tile=0,  # CVT0 feeds into 1st MFMA (bf16*bf16)
                base_offset=base_offset,
                n_a_tiles=n_tiles_a,
                n_b_tiles=n_tiles_b,
                mfma_reorder=mfma_reorder,
            )
            cvt0_mfma_needed_by = mfma_for_linear_index[iteration_offset + cvt0_earliest]
            for pack in cvt0:
                # CVT0 packs have both inter-pack and MFMA needed_by; take the earlier one
                if pack.needed_by.issued_at > cvt0_mfma_needed_by.issued_at:
                    pack.needed_by = cvt0_mfma_needed_by

            # External MFMA needed_by for CVT1 packs (all share the same MFMA target)
            cvt1_earliest = find_earliest_mfma_execution(
                is_pack_B=is_pack_B,
                tile_index=group_index,
                mfma_in_tile=2 if is_pack_B else 1,
                base_offset=base_offset,
                n_a_tiles=n_tiles_a,
                n_b_tiles=n_tiles_b,
                mfma_reorder=mfma_reorder,
            )
            cvt1_mfma_needed_by = mfma_for_linear_index[iteration_offset + cvt1_earliest]
            for pack in cvt1:
                if pack.needed_by.issued_at > cvt1_mfma_needed_by.issued_at:
                    pack.needed_by = cvt1_mfma_needed_by
    else:
        # Regular TF32: Packs come in groups of 24
        # Half tile count since each quarter uses half of the A tiles and half of the B tiles.
        n_tiles_a //= 2
        n_tiles_b //= 2

        # Group packs by group_index (computed at construction time)
        groups: dict[int, list[Pack]] = defaultdict(list)
        for pack in packs:
            groups[pack.group_index].append(pack)

        for group_index, group_packs in sorted(groups.items()):
            # MiddlePacks don't need needed_by set (handled implicitly)
            cvt_packs = [p for p in group_packs if isinstance(p, CVTPack)]
            assert len(cvt_packs) == 8, f"{packs[0].name}: Expected 8 CVT packs per group, got {len(cvt_packs)}"
            # CVT0 come before CVT1 by construction order (sorted by issue_index)
            cvt0 = cvt_packs[:4]
            cvt1 = cvt_packs[4:]
            assert cvt0[-1].issue_index < cvt1[0].issue_index, "CVT0 packs must have lower issue_index than CVT1 packs"

            # CVT0 packs (bf16 approximations) are used by MFMA 0 (bf16*bf16)
            cvt0_earliest = find_earliest_mfma_execution(
                is_pack_B=is_pack_B,
                tile_index=group_index,
                mfma_in_tile=0,
                base_offset=base_offset,
                n_a_tiles=n_tiles_a,
                n_b_tiles=n_tiles_b,
                mfma_reorder=mfma_reorder,
            )
            cvt0_needed_by = mfma_for_linear_index[iteration_offset + cvt0_earliest]
            for pack in cvt0:
                pack.needed_by = cvt0_needed_by

            # CVT1 packs (error terms): A_error -> 2nd MFMA, B_error -> 3rd MFMA
            cvt1_earliest = find_earliest_mfma_execution(
                is_pack_B=is_pack_B,
                tile_index=group_index,
                mfma_in_tile=2 if is_pack_B else 1,
                base_offset=base_offset,
                n_a_tiles=n_tiles_a,
                n_b_tiles=n_tiles_b,
                mfma_reorder=mfma_reorder,
            )
            cvt1_needed_by = mfma_for_linear_index[iteration_offset + cvt1_earliest]
            for pack in cvt1:
                pack.needed_by = cvt1_needed_by
       

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

def _build_reg_to_lr_map(vw: int, n_lrs: int) -> dict[int, int]:
    """Build a map from logical register index to LR index.

    Both swap VGPRs and LR destination VGPRs are resolved through
    TXInterleaveLayoutIdx (LocalRead.py:211), which splits registers into
    T (idx%8 < 4) and X (idx%8 >= 4) arrays with physical offsets.

    But LR destinations ALSO go through dsReadConvTable (LocalRead.py:242-243)
    first, which reorders which LR writes to which physical position. The swap
    VGPRs skip dsReadConvTable (transposeLRVregs passes lrvwTile=1).

    This function builds the reverse map by:
    1. Computing each LR's physical VGPR range via dsReadConvTable + TXInterleave
    2. Computing each swap register's physical VGPR via TXInterleave alone
    3. Matching swap VGPRs to LR ranges
    """
    _DS_READ_CONV_TABLE = {
        2: [0, 8, 2, 10, 4, 12, 6, 14],
        4: [0, 8, 16, 24, 4, 12, 20, 28],
    }
    conv = _DS_READ_CONV_TABLE[vw]

    def _tx_interleave(idx):
        """Replicate TXInterleaveLayoutIdx: returns (is_t_array, physical_offset)."""
        if idx % 8 < 4:
            return (True, (idx // 8) * 4 + idx % 4)
        return (False, idx)

    # Step 1: For each LR[i], compute the physical VGPR range it writes to.
    # LR[i] goes through dsReadConvTable[i] → TXInterleave → loads `vw` consecutive VGPRs.
    phys_to_lr: dict[tuple[bool, int], int] = {}
    for lr_idx in range(n_lrs):
        is_t, start_offset = _tx_interleave(conv[lr_idx])
        for j in range(vw):
            phys_to_lr[(is_t, start_offset + j)] = lr_idx

    # Step 2: For each logical register, find which LR loaded its physical VGPR.
    # Swap VGPRs go through TXInterleave only (no dsReadConvTable).
    total_regs = 8 * vw  # one transpose block
    reg_to_lr: dict[int, int] = {}
    for reg in range(total_regs):
        phys = _tx_interleave(reg)
        reg_to_lr[reg] = phys_to_lr[phys]

    return reg_to_lr

def _hook_up_packs_bf16(packs: list[Pack], local_reads: list[LocalRead]) -> None:
    """
    For BF16/Half: each Pack uses the result of 2 consecutive LRs.
    Pack ordering follows the v_perm loop in LocalRead.py:
        for vectorIdx in range(0, 2):        # V0, V1
            for elementIdx in range(0, num_element_pairs):
                pack uses D[elementIdx*2] and D[elementIdx*2+1]
    
    So element_idx = pack_position % num_element_pairs
    And LR indices are: elementIdx*2 and elementIdx*2+1
    
    This function sets the must_start_after field based on LR dependencies.
    The needed_by field is set separately by _set_pack_needed_by.
    """
    num_element_pairs = len(local_reads) // 2
    
    # Re-order local_reads by their index in the list of Local Read instructions, rather than by the mfma index they were issued at.
    # It is this order that's needed to properly calculate must_start_after for Packs.
    local_reads.sort(key=lambda lr: lr.issue_index)

    # Calculate must_start_after
    for pack in packs:
        # Determine which element pair this pack uses
        element_idx = pack.issue_index % num_element_pairs
        lr_idx_0 = element_idx * 2
        lr_idx_1 = element_idx * 2 + 1                    
        pack_to_lrs = [local_reads[lr_idx_0], local_reads[lr_idx_1]]

        # Max is most restrictive since `guaranteed_by` is a lower bound on issued_at.
        latest_lr = max(pack_to_lrs, key=lambda lr: lr.done_idx())
        pack.must_start_after.append(latest_lr)

def _hook_up_packs_f32(packs: list[Pack], all_middle_16_packs: list['MiddlePack'], local_reads: list[LocalRead]) -> None:
    """
    For TF32 emulation, data is loaded as fp32 and converted into pairs of bf16 values.
    Each fp32 value is converted into a bf16 approximation and an error term.

    Conversion happens in groups of 8 VGPRs (32*8 = 256 bytes).
    Input is 8 VGPRs, each holding one fp32 value.
    Output is 8 VGPRs, all holding packed bf16 values.
    The first 4 output registers hold the bf16 approximations (packed in pairs).
    The second 4 output registers hold the error terms (packed in pairs).

    Pack instructions in order (24 instructions total):
    - 4 `v_cvt_pk_bf16_f32` to calculate and pack the bf16 approximations.
    - 8 pairs of (`v_cvt_f32_bf16`, `v_sub_f32`) to calculate the error terms.
    - 4 `v_cvt_pk_bf16_f32` to pack the error terms into final registers.

    This function sets the must_start_after field based on LR and inter-pack dependencies,
    and handles pair constraints for middle-16 packs.
    The needed_by field is set separately by _set_pack_needed_by.
    """
    if not local_reads or not packs:
        return

    # Sort by index in the list of pack instructions rather than by the mfma_index they are placed at.
    # This is necessary to handle inter-pack dependencies.
    packs = sorted(packs, key=lambda x: x.issue_index)

    # Group packs by group_index (computed at construction time)
    pack_groups: dict[int, list[Pack]] = defaultdict(list)
    for pack in packs:
        pack_groups[pack.group_index].append(pack)
    n_pack_groups = len(pack_groups)

    assert len(local_reads) % n_pack_groups == 0, "Case not supported: Different number of LRs for each Pack group."
    n_lrs_per_group = len(local_reads) // n_pack_groups

    # NOTE: Assuming that all LRs are of the same width.
    vgprs_per_local_read = VGPRS_PER_CONVERSION_GROUP // n_lrs_per_group

    # Partial Pack->Pack dependency graph within a group of 24.
    # Key: pack index (0-23), Value: list of pack indices it depends on.
    # Empty list means it depends on local reads only (CVT0 packs).
    # NOTE: This is only a partial graph. It does not account for use of the temporary register by the middle 16 packs.
    #       That interaction is handled separately at the end of this function.
    # CVT1 deps (indices 20-23) reflect WAR hazards from the reverse-write
    # pattern, same as TF32 4x4. Pack 23 has WAR on pack 21 (CVT1[1] reads
    # the register that CVT1[3] overwrites), not on pack 22.
    pack_dependencies: dict[int, list[int]] = {
        # First 4 packs (v_cvt_pk_bf16_f32) depend on local reads only, and are not included
        0: [], 1: [], 2: [], 3: [],
        # Middle 16 packs (v_cvt_f32_bf16 + v_sub_f32 pairs) - error term calculation
         4: [0],  5: [ 4],  6: [0],  7: [ 6],
         8: [1],  9: [ 8], 10: [1], 11: [10],
        12: [2], 13: [12], 14: [2], 15: [14],
        16: [3], 17: [16], 18: [3], 19: [18],
        # Final 4 packs (v_cvt_pk_bf16_f32) - pack error terms
        # WAR hazard chain from reverse-write pattern.
        20: [17, 19],
        21: [13, 15, 20],
        22: [ 9, 11, 21],
        23: [ 5,  7, 21],
    }

    for group_idx in sorted(pack_groups.keys()):
        start = group_idx * n_lrs_per_group
        end = start + n_lrs_per_group
        local_reads_for_group = local_reads[start:end]

        pack_group = pack_groups[group_idx]

        # Set must_start_after
        for leader_idx, pack in enumerate(pack_group):
            dependencies = pack_dependencies[leader_idx]
            if not dependencies:
                # CVT0 packs: depend only on local reads.
                first_lr = (leader_idx * 2) // vgprs_per_local_read
                last_lr = (leader_idx * 2 + 1) // vgprs_per_local_read
                pack_lrs = local_reads_for_group[first_lr:last_lr + 1]
                latest_lr = max(pack_lrs, key=lambda lr: lr.done_idx())
                pack.must_start_after.append(latest_lr)
            else:
                # MiddlePack and CVT1: depend on other packs (via pack_dependencies).
                latest_dep = max((pack_group[d] for d in dependencies), key=lambda p: p.done_idx())
                pack.must_start_after.append(latest_dep)

    # For the middle-16 packs, hook up the consumer Pack to the producer Pack to handle temporary register re-use.
    # The middle 16 packs are scheduled sequentially in pairs, and no other middle-16 pack
    # (even from other groups) can be scheduled between a pair.
    for group_idx in sorted(pack_groups.keys()):
        middle_packs = [p for p in pack_groups[group_idx] if isinstance(p, MiddlePack)]
        for i in range(0, len(middle_packs), 2):
            middle_packs[i].pair_consumer = middle_packs[i + 1]

    # Hook up the producer Pack in each pair to the middle-16 Pack scheduled immediately after it.
    # Only modify the packs that were passed in, rather than all packs in all_middle_16_packs.
    for pack in packs:
        if not isinstance(pack, MiddlePack):
            continue
        if pack.pair_consumer is None:  # Not a producer (pair_consumer set above)
            continue
        pack.next_scheduled_middle_16 = all_middle_16_packs[all_middle_16_packs.index(pack) + 1]

def _hook_up_packs_f32_mfma(packs: list[Pack], local_reads: list[LocalRead], vw: int) -> None:
    """
    For TF32 emulation, data is loaded as fp32 and converted into pairs of bf16 values.
    Each fp32 value is converted into a bf16 approximation and an error term.

    Conversion happens in groups of 8 VGPRs (32*8 = 256 bytes).
    Input is 8 VGPRs, each holding one fp32 value.
    Output is 8 VGPRs, all holding packed bf16 values.
    The first 4 output registers hold the bf16 approximations (packed in pairs).
    The second 4 output registers hold the error terms (packed in pairs).

    Pack instructions in order (10 instructions total):
    - 4 `v_cvt_pk_bf16_f32` to calculate and pack the bf16 approximations.
    - 2 `v_mfma_f32_4x4x4_16b_bf16` to calculate the error terms.
    - 4 `v_cvt_pk_bf16_f32` to pack the error terms into final registers.

    When VectorWidth > 1, SwapPack instructions (VSwapB32) appear before the regular
    pack groups. These transpose registers after wider local reads. Each swap depends
    only on the 2 specific LRs that loaded its register pair, and each CVT0 pack depends
    on the specific swaps (or LRs) that produced its input registers.
    """
    if not local_reads or not packs:
        return

    # Sort by index in the list of pack instructions rather than by the mfma_index they are placed at.
    # This is necessary to handle inter-pack dependencies.
    packs = sorted(packs, key=lambda x: x.issue_index)

    # Separate SwapPacks from regular packs
    swap_packs = [p for p in packs if isinstance(p, SwapPack)]
    regular_packs = [p for p in packs if not isinstance(p, SwapPack)]

    # Group regular packs by group_index (computed at construction time)
    pack_groups_map: dict[int, list[Pack]] = defaultdict(list)
    for pack in regular_packs:
        pack_groups_map[pack.group_index].append(pack)
    n_pack_groups = len(pack_groups_map)
    if n_pack_groups == 0:
        return

    # Determine the register-to-LR mapping.
    # With VW > 1 (swap packs present), TF32EmuInterleaveTreg is active and registers
    # alternate between T and X VGPR arrays. The dsReadConvTable further reorders
    # which LR writes to which physical VGPR position. Use _build_reg_to_lr_map.
    # With VW = 1 (no swaps), registers are contiguous. Use simple linear mapping.
    n_lrs = len(local_reads)
    if swap_packs:
        reg_to_lr_map = _build_reg_to_lr_map(vw, n_lrs)
        reg_to_lr = lambda reg: reg_to_lr_map[reg]
    else:
        vgprs_per_local_read = VGPRS_PER_CONVERSION_GROUP * n_pack_groups // n_lrs if n_lrs > 0 else 1
        if vgprs_per_local_read == 0:
            vgprs_per_local_read = 1
        reg_to_lr = lambda reg: reg // vgprs_per_local_read

    # Build fine-grained swap dependencies
    swap_for_reg: dict[int, SwapPack] = {}
    if swap_packs:
        total_regs = VGPRS_PER_CONVERSION_GROUP * vw
        swap_reg_pairs = _compute_swap_register_pairs(vw, total_regs)

        # Each SwapPack depends on the 2 LRs that loaded its register pair.
        for sp, (reg_src, reg_dst) in zip(swap_packs, swap_reg_pairs):
            lr_a = local_reads[reg_to_lr(reg_src)]
            lr_b = local_reads[reg_to_lr(reg_dst)]
            sp.must_start_after.append(lr_a)
            if lr_a is not lr_b:
                sp.must_start_after.append(lr_b)

        # Build reg -> swap lookup for CVT0 dependencies
        for sp, (reg_src, reg_dst) in zip(swap_packs, swap_reg_pairs):
            swap_for_reg[reg_src] = sp
            swap_for_reg[reg_dst] = sp

    # Partial Pack->Pack dependency graph within a group of 10.
    # Key: pack index (0-9), Value: list of pack indices it depends on.
    # Empty list means it depends on local reads only (CVT0 packs).
    # NOTE: Does not handle the quad-cycle spacing dependencies between packs and MFMAs.
    #
    # CVT1 deps (indices 6-9) reflect WAR (write-after-read) hazards from
    # the reverse-write pattern: CVT1 writes dst registers in descending
    # order (7d, 6d, 5d, 4d) and each overwrites a register that a prior
    # CVT1 read. Specifically:
    #   CVT1[0] reads v6d (which CVT1[1] will overwrite) → 7 depends on 6
    #   CVT1[1] reads v4d,v5d (which CVT1[2],CVT1[3] overwrite) → 8,9 depend on 7
    # Pack 9 has no WAR on pack 8 because pack 9 writes v4d which only
    # pack 7 (CVT1[1]) previously read, not pack 8.
    pack_dependencies: dict[int, list[int]] = {
        # First 4 packs only depend on local reads.
        0: [], 1: [], 2: [], 3: [],
        # Middle 2 Packs are vmfma and depend on the previous 4 packs.
        4: [0, 1],
        5: [2, 3],
        # CVT1 packs: WAR hazard chain from reverse-write pattern.
        6: [5],
        7: [5, 6],
        8: [4, 7],
        9: [4, 7],
    }

    for group_idx in sorted(pack_groups_map.keys()):
        pack_group = pack_groups_map[group_idx]

        # Set must_start_after
        for pack_idx, pack in enumerate(pack_group):
            dependencies = pack_dependencies[pack_idx]
            if not dependencies:
                # CVT0 packs: depend on swaps (for swapped regs) or LRs (for non-swapped regs).
                first_reg = group_idx * VGPRS_PER_CONVERSION_GROUP + pack_idx * 2
                last_reg = first_reg + 1
                for reg in (first_reg, last_reg):
                    if reg in swap_for_reg:
                        pack.must_start_after.append(swap_for_reg[reg])
                    else:
                        pack.must_start_after.append(local_reads[reg_to_lr(reg)])
            else:
                # MFMAPack and CVT1: depend on other packs (via pack_dependencies).
                latest_dep = max((pack_group[d] for d in dependencies), key=lambda p: p.done_idx())
                pack.must_start_after.append(latest_dep)

def _get_lrs_for_pack(timeline: Timeline, use_plr_pack: bool, pack_name: str, loop: str) -> list[LocalRead]:
    """
    For a given Pack instruction, get all the LocalRead instructions it depends on.
    If use_plr_pack==True:
        - All Pack instructions load data from LRs issued in this iteration (including Pack0).
    
    If use_plr_pack==False:
        - The Pack1/3 instructions pack data loaded by LRs issued in the previous iteration.
          - If it's the first loop for Pack1/3, we don't have LRs to hook up to.
          - The same insturctions will be handled in the next loop.
        - The Pack0 instructions pack data loaded by LRs issued in the current iteration.

    Args:
        timeline: The Timeline object to get the LRs from.
        use_plr_pack: Whether to the UserPLRPack flag is set.
        pack_name: The name of the pack to get the LRs for.
        loop: The name of the loop to get the LRs for.

    Returns:
        A list of LocalRead objects.
    """
    pack_1_or_3 = not pack_name.endswith("0")
    if pack_1_or_3 and loop == timeline.loops[0]:
        return []

    lr_names = pack_name.replace("Pack", "LR")
    if use_plr_pack:
        return [lr for _,lr in timeline.get_instructions(lr_names, loop)]

    i_loop = timeline.loops.index(loop)
    loop_to_use = timeline.loops[i_loop - 1] if pack_1_or_3 else loop
    local_reads = timeline.get_instructions(lr_names, loop_to_use)
    return [lr for _,lr in local_reads]

def _hook_up_middle_16_pairs(packs: list[Pack], all_middle_16_packs: list['MiddlePack']) -> None:
    """Set pair_consumer and next_scheduled_middle_16 for middle-16 packs.

    Middle-16 packs (v_cvt_f32_bf16 + v_sub_f32 pairs) share a temporary register.
    Each pair must be scheduled adjacently — no other middle-16 pack (even from
    other groups) may be scheduled between a pair's producer and consumer.
    """
    pack_groups: dict[int, list[Pack]] = defaultdict(list)
    for pack in packs:
        if isinstance(pack, MiddlePack):
            pack_groups[pack.group_index].append(pack)

    for group_idx in sorted(pack_groups.keys()):
        middle_packs = pack_groups[group_idx]
        for i in range(0, len(middle_packs), 2):
            middle_packs[i].pair_consumer = middle_packs[i + 1]

    for pack in packs:
        if not isinstance(pack, MiddlePack):
            continue
        if pack.pair_consumer is None:
            continue
        pack.next_scheduled_middle_16 = all_middle_16_packs[all_middle_16_packs.index(pack) + 1]


@applies_only_once
def hook_up_packs(timeline: Timeline, kernel: 'Solution', mfma_reorder: list[int]) -> None:
    """
    Set the needed_by and must_start_after fields of Packs based on the LR(s) they depend on.

    Args:
        timeline:       The Timeline object containing the instructions.
        kernel:         Solution object containing the kernel metadata.
        mfma_reorder:   Mapping between the index of a default-scheduled MFMA and its new custom assigned index.
    """
    if mfma_reorder and len(mfma_reorder) != timeline.num_vmfma:
        raise ValueError(f"Incorrect number of VMFMA indices in mfmaReorder. Expected {timeline.num_vmfma}, given {len(mfma_reorder)}.")
    

    is_tf32_emulation = kernel.get("UseF32XEmulation", False)
    is_4x4mfma_tf32 = kernel.get("UseMFMAF32XEmulation", False)
    is_direct_32x_emulation = kernel.get("UseDirect32XEmulation", False)

    if is_tf32_emulation and not is_direct_32x_emulation:
        raise ValueError("UseDirect32XEmulation is False, case not supported.")

    mfma_for_linear_index: dict[int, MFMA] = {
        mfma.issued_at.loop_index * timeline.num_vmfma + mfma.issued_at.vmfma_index: mfma
        for _, mfma in timeline.get_instructions_combined("MFMA")
    }

    use_plr_pack = kernel.get("UsePLRPack", False)
    for i_loop, loop in enumerate(timeline.loops):
        # 1. Gather all Packs in the current loop.
        packs_by_name: dict[str, list[Pack]] = {}
        for pack_name in timeline.get_instruction_names():
            if not pack_name.startswith("Pack"):
                continue
            packs_and_indices = timeline.get_instructions(pack_name, loop)
            if not packs_and_indices:
                continue
            packs_by_name[pack_name] = [pack for _, pack in packs_and_indices]

        # 2. Gather all middle-16 packs in the current loop.
        if is_tf32_emulation and not is_4x4mfma_tf32:
            all_middle_16_packs = []
            for packs in packs_by_name.values():
                for pack in packs:
                    if isinstance(pack, MiddlePack):
                        all_middle_16_packs.append(pack)
            all_middle_16_packs.sort(key=lambda p: p.issued_at)

        # Compute the loop-scoped register-traced needed_by map once per loop.
        all_packs_in_loop = [p for plist in packs_by_name.values() for p in plist]
        real_mfmas_in_loop = [m for _, m in timeline.get_instructions("MFMA", loop)]
        needed_by_map = set_pack_needed_by_from_mfma_operands(all_packs_in_loop, real_mfmas_in_loop)

        # 3. Hook up the needed_by and must_start_after fields
        for pack_name, packs in packs_by_name.items():
            local_reads = _get_lrs_for_pack(timeline, use_plr_pack, pack_name, loop)
            if not local_reads:
                continue

            # Set must_start_after from register operand tracing (RAW + WAR).
            reg_deps = derive_pack_must_start_after(packs, local_reads)
            for pack in packs:
                if pack.issue_index in reg_deps and reg_deps[pack.issue_index]:
                    pack.must_start_after = reg_deps[pack.issue_index]

            # TF32-specific constraints
            # Bead `8nz`: removed `_handle_min_pack_quad_cycles(packs)` —
            # quad-cycle visibility is now graph-side.
            if is_tf32_emulation:
                if not is_4x4mfma_tf32:
                    _hook_up_middle_16_pairs(packs, all_middle_16_packs)

            # Set needed_by: prefer the new register-traced path when every
            # eligible pack in this pack-name is covered; otherwise fall back
            # wholesale to _set_pack_needed_by. Per-pack mixing would let one
            # CVT0 in a TF32 4x4 group point at the new path's MFMAPack while
            # a sibling points at the old path's external MFMA.
            per_name_map = needed_by_map.get(pack_name, {})
            eligible = [p for p in packs if not isinstance(p, SwapPack)
                        and p.rocisa_inst is not None]
            if eligible and all(p.issue_index in per_name_map for p in eligible):
                for pack in eligible:
                    pack.needed_by = per_name_map[pack.issue_index]
            else:
                _set_pack_needed_by(packs, pack_name, i_loop, mfma_reorder, mfma_for_linear_index, timeline.num_vmfma, kernel)

def derive_pack_must_start_after(
    packs: list[Pack],
    local_reads: list[LocalRead],
) -> dict[int, list[ValidatorInstruction]]:
    """Derive must_start_after constraints for Packs from register operands.

    Traces register dependencies: for each Pack, finds which prior instruction
    must complete before this pack can start. Captures two types of hazards:

    - RAW (read-after-write): this pack reads a register that a prior instruction
      wrote. The prior instruction must finish writing before this pack reads.
    - WAR (write-after-read): this pack writes a register that a prior instruction
      read. The prior instruction must finish reading before this pack overwrites.
      WAR tracking is skipped for MiddlePack — those ordering constraints are
      handled separately by the pair_consumer/next_scheduled_middle_16 mechanism.

    Args:
        packs:        Pack instructions (must have rocisa_inst set).
        local_reads:  LocalRead instructions (must have rocisa_inst set).

    Returns:
        Dict mapping issue_index -> list of producer ValidatorInstructions
        (reduced to the single latest dependency).
        Empty dict if any instruction lacks rocisa_inst.
    """
    # All instructions must have rocisa_inst (id_map is mandatory since stage 07)
    missing_packs = [p for p in packs if p.rocisa_inst is None]
    if missing_packs:
        raise ValueError(
            f"Pack(s) missing rocisa_inst: {[f'{p.name}[{p.issue_index}]' for p in missing_packs]}. "
            f"id_map is mandatory — all instructions must have rocisa_inst."
        )
    missing_lrs = [lr for lr in local_reads if lr.rocisa_inst is None]
    if missing_lrs:
        raise ValueError(
            f"LocalRead(s) missing rocisa_inst: {[f'{lr.name}[{lr.issue_index}]' for lr in missing_lrs]}. "
            f"id_map is mandatory — all instructions must have rocisa_inst."
        )

    # Producer map: register -> instruction that last WROTE this register
    producers: dict[tuple[str, int], ValidatorInstruction] = {}
    # Consumer map: register -> instruction that last READ this register
    consumers: dict[tuple[str, int], ValidatorInstruction] = {}

    # Pre-populate producers with LR destinations
    for lr in local_reads:
        dst = get_dst_range(lr.rocisa_inst)
        if dst:
            base, start, end = dst
            for off in range(start, end):
                producers[(base, off)] = lr

    # Walk packs in issue_index order
    sorted_packs = sorted(packs, key=lambda p: p.issue_index)
    result: dict[int, list[ValidatorInstruction]] = {}

    for pack in sorted_packs:
        # Use dict keyed by id() since ValidatorInstruction is unhashable (mutable dataclass)
        deps: dict[int, ValidatorInstruction] = {}

        # RAW: find which producers wrote to this pack's source registers
        src_ranges = get_src_ranges(pack.rocisa_inst)
        for base, start, end in src_ranges:
            for off in range(start, end):
                key = (base, off)
                if key in producers and producers[key] is not pack:
                    dep = producers[key]
                    deps[id(dep)] = dep

        # WAR: find which consumers read from this pack's destination registers.
        # This pack will overwrite those registers, so it must wait for the
        # reader to finish. This captures the TF32 CVT1 reverse-write pattern
        # where CVT1 packs write dst registers in descending order (7d,6d,5d,4d)
        # and each overwrites a register a prior CVT1 read. The last CVT1 in
        # a group depends on CVT1[1] (which read the register), not on CVT1[2]
        # (no shared register). See pack_dependencies comments in
        # _hook_up_packs_f32_mfma and _hook_up_packs_f32.
        # Skip for MiddlePack — pair ordering is handled by
        # pair_consumer/next_scheduled_middle_16.
        if not isinstance(pack, MiddlePack):
            dst = get_dst_range(pack.rocisa_inst)
            if dst:
                base, start, end = dst
                for off in range(start, end):
                    key = (base, off)
                    if key in consumers and consumers[key] is not pack:
                        dep = consumers[key]
                        deps[id(dep)] = dep

        # Reduce to only the latest dependency — only the latest constraint
        # matters for scheduling (Pack.validate() already takes max anyway).
        if deps:
            latest = max(deps.values(), key=lambda d: d.done_idx())
            result[pack.issue_index] = [latest]
        else:
            result[pack.issue_index] = []

        # Update consumer map with this pack's source registers
        for base, start, end in src_ranges:
            for off in range(start, end):
                consumers[(base, off)] = pack

        # Update producer map with this pack's destination
        dst = get_dst_range(pack.rocisa_inst)
        if dst:
            base, start, end = dst
            for off in range(start, end):
                producers[(base, off)] = pack

        # SwapPack special case: v_swap_b32 writes to BOTH src and dst registers
        # (it swaps their contents). The rocisa binding only tracks .dst as the
        # write destination. We must also mark the .src registers as written by
        # this swap so that later packs reading from those registers find the
        # swap as the producer.
        if isinstance(pack, SwapPack):
            for base, start, end in src_ranges:
                for off in range(start, end):
                    producers[(base, off)] = pack

    return result


def set_lr_needed_by_from_mfma_operands(
    lrs: list[LocalRead],
    packs: list[Pack],
    mfmas: list[MFMA],
) -> dict[str, dict[int, MFMA]]:
    """Derive LR.needed_by from register operands.

    For each LR, finds the earliest real MFMA that (transitively, through
    pack chains) consumes the LR's destination register. The chain follow
    handles arbitrary depth: BF16 (LR→Pack→MFMA), regular TF32
    (LR→CVT0→MiddlePack→CVT1→MFMA), and TF32 4x4
    (LR→CVT0→MFMAPack→CVT1→MFMA).

    Inputs MUST be loop-scoped: caller passes LRs and the candidate
    consumers (packs + real MFMAs) from a single loop replication. For
    LR0, consumers are same-loop. For LR1/LR3, consumers are next-loop
    (positional baseline shifts +num_vmfma at :2775-2776).

    The function is self-contained: it computes the pack-chain consumer
    table inline by calling set_pack_needed_by_from_mfma_operands rather
    than reading pack.needed_by. This eliminates dependence on the
    validator pass order (add_pack_constraints need not run before
    add_local_read_constraints) and on test mocks initializing
    pack.needed_by.

    The "issued strictly after" filter handles same-loop register reuse
    (kernel prefetch pattern): an LR cannot pick a candidate issued at
    or before itself, mirroring set_pack_needed_by_from_mfma_operands.

    Returns dict lr_name → issue_index → real-MFMA consumer. LRs whose
    chain does not terminate at a real MFMA (e.g. SwapPack-only
    consumer in TF32 4x4 VW>1, missing rocisa_inst on a chain link)
    are omitted; caller treats absence as "fall back to positional path."
    """
    if not lrs:
        return {}

    # Step 1: Compute the pack-chain consumer table inline. Independent
    # of pack.needed_by field state.
    pack_chain = set_pack_needed_by_from_mfma_operands(packs, mfmas)

    # Build id-keyed lookup for chain-follow.
    next_consumer: dict[int, ValidatorInstruction] = {}
    for pack in packs:
        per_name = pack_chain.get(pack.name, {})
        if pack.issue_index in per_name:
            next_consumer[id(pack)] = per_name[pack.issue_index]

    # Step 2: Build candidates index (real MFMAs + non-Swap packs),
    # sorted by issued_at per register. Mirrors Pack-side at :2459-2471.
    candidates: list[ValidatorInstruction] = [
        m for m in mfmas if m.rocisa_inst is not None
    ]
    candidates.extend(
        p for p in packs
        if p.rocisa_inst is not None and not isinstance(p, SwapPack)
    )
    by_reg: dict[tuple[str, int], list[ValidatorInstruction]] = {}
    for inst in candidates:
        for base, start, end in get_src_ranges(inst.rocisa_inst):
            for off in range(start, end):
                by_reg.setdefault((base, off), []).append(inst)
    for lst in by_reg.values():
        lst.sort(key=lambda i: i.issued_at)

    # Step 3: For each LR, find immediate consumer + follow chain to a
    # real MFMA.
    result: dict[str, dict[int, MFMA]] = {}
    for lr in lrs:
        if lr.rocisa_inst is None:
            continue
        dst = get_dst_range(lr.rocisa_inst)
        if dst is None:
            continue
        base, start, end = dst

        # Earliest immediate consumer with strict < filter. Mirrors
        # Pack-side at :2474-2493.
        immediate: Optional[ValidatorInstruction] = None
        for off in range(start, end):
            lst = by_reg.get((base, off))
            if not lst:
                continue
            for cand in lst:
                if not (lr.issued_at < cand.issued_at):
                    continue
                if immediate is None or cand.issued_at < immediate.issued_at:
                    immediate = cand
                break  # list sorted by issued_at; first valid is earliest for this off

        if immediate is None:
            continue

        # Step 4: Follow the chain. Unified loop — handles plain Pack,
        # CVTPack, MiddlePack, MFMAPack uniformly. MFMAPack inherits Pack,
        # so the loop continues through it; the acceptance check below
        # rejects MFMAPack as a final answer.
        target: Optional[ValidatorInstruction] = immediate
        seen: set[int] = set()
        while isinstance(target, Pack) and id(target) not in seen:
            seen.add(id(target))
            nxt = next_consumer.get(id(target))
            if nxt is None:
                target = None  # Chain dies; e.g. SwapPack with no successor
                break
            target = nxt

        # Accept only real MFMAs (not MFMAPack, not the placeholder
        # default at :333/:378/:476 which has rocisa_inst=None).
        if (target is not None
                and isinstance(target, MFMA)
                and not isinstance(target, MFMAPack)
                and target.rocisa_inst is not None):
            result.setdefault(lr.name, {})[lr.issue_index] = target

    return result


def set_pack_needed_by_from_mfma_operands(
    packs: list[Pack],
    mfmas: list[MFMA],
) -> dict[str, dict[int, ValidatorInstruction]]:
    """Derive Pack.needed_by from register operands.

    For each non-SwapPack, finds the earliest instruction (real MFMA or
    another Pack) whose source register range overlaps the pack's
    destination register range AND that is issued strictly after the pack.

    The TF32 4x4 chain CVT0.dst → MFMAPack.b/acc → CVT1.src/dst →
    real_MFMA.a/b emerges naturally because MFMAPacks are included in
    `packs` (they are Pack subclass instances).

    Inputs MUST be loop-scoped: pass packs and MFMAs from a single
    loop replication. Physical VGPRs are reused across loop iterations,
    so a cross-loop input would assign cross-loop needed_by and break
    Pack.validate()'s pack.issued_at < pack.needed_by.issued_at check
    (SchedulePosition orders by loop_index first; see CMSValidator.py:56-63).

    The "issued strictly after" filter handles same-loop register reuse:
    e.g. CVT0[3] reads from a register that an earlier-iteration CVT1[1]
    wrote — the same physical register; in this loop's instruction list
    CVT0[3] is issued before CVT1[1], and we must not assign
    CVT1[1].needed_by = CVT0[3].

    Returns dict pack_name → issue_index → consumer ValidatorInstruction.
    Empty dict if no MFMAs or no eligible packs.
    """
    if not mfmas and not packs:
        return {}

    # Candidates: real MFMAs + non-SwapPack packs with a rocisa_inst.
    candidates: list[ValidatorInstruction] = [m for m in mfmas if m.rocisa_inst is not None]
    candidates.extend(p for p in packs
                      if p.rocisa_inst is not None and not isinstance(p, SwapPack))

    # Index candidates by source register, sorted by issued_at.
    by_reg: dict[tuple[str, int], list[ValidatorInstruction]] = {}
    for inst in candidates:
        for base, start, end in get_src_ranges(inst.rocisa_inst):
            for off in range(start, end):
                by_reg.setdefault((base, off), []).append(inst)
    for lst in by_reg.values():
        lst.sort(key=lambda i: i.issued_at)

    result: dict[str, dict[int, ValidatorInstruction]] = {}
    for pack in packs:
        if pack.rocisa_inst is None or isinstance(pack, SwapPack):
            continue
        dst = get_dst_range(pack.rocisa_inst)
        if dst is None:
            continue
        base, start, end = dst
        earliest: Optional[ValidatorInstruction] = None
        for off in range(start, end):
            lst = by_reg.get((base, off))
            if not lst:
                continue
            for cand in lst:
                if cand is pack:
                    continue
                if not (pack.issued_at < cand.issued_at):
                    continue  # cand fires at/before pack — reads prior iteration
                if earliest is None or cand.issued_at < earliest.issued_at:
                    earliest = cand
                break  # list sorted by issued_at; first valid is earliest for this off
        if earliest is not None:
            result.setdefault(pack.name, {})[pack.issue_index] = earliest

    return result


def verify_swaitcnt_counters(timeline: 'Timeline') -> Optional[str]:
    """Verify that SWaitCnt counter values match the actual number of outstanding memory operations.

    Counts DSLoad* instructions for dscnt and BufferLoad*/GlobalLoad* for vlcnt,
    then checks each SWaitCnt's counter value against the actual count.

    Returns an error message if a mismatch is found, None if all counters are correct.
    Only runs if rocisa_inst is available on instructions (requires idMap).
    """
    from rocisa.instruction import DSLoadInstruction, GlobalReadInstruction

    # Check if rocisa_inst is available
    has_rocisa = False
    for inst in timeline.combined_timeline:
        if inst.rocisa_inst is not None:
            has_rocisa = True
            break
    if not has_rocisa:
        return None

    # Count outstanding memory operations and verify at each SWaitCnt
    outstanding_ds = 0  # lgkmcnt / dscnt counter
    outstanding_vm = 0  # vmcnt / vlcnt counter

    for inst in timeline.combined_timeline:
        if isinstance(inst, LocalRead) and inst.rocisa_inst is not None:
            if isinstance(inst.rocisa_inst, DSLoadInstruction):
                outstanding_ds += 1
        elif isinstance(inst, GlobalRead) and inst.rocisa_inst is not None:
            if isinstance(inst.rocisa_inst, GlobalReadInstruction):
                outstanding_vm += 1
        elif isinstance(inst, SWait):
            from Tensile.Components.ScheduleCapture import SWaitCountExceedsOutstandingFailure
            if inst.dscnt >= 0:
                if inst.dscnt > outstanding_ds:
                    return SWaitCountExceedsOutstandingFailure(
                        swait=inst,
                        counter_kind="dscnt",
                        counter_value=inst.dscnt,
                        outstanding=outstanding_ds,
                    ).format(capture=None)
                outstanding_ds = inst.dscnt
            if inst.vlcnt >= 0:
                if inst.vlcnt > outstanding_vm:
                    return SWaitCountExceedsOutstandingFailure(
                        swait=inst,
                        counter_kind="vlcnt",
                        counter_value=inst.vlcnt,
                        outstanding=outstanding_vm,
                    ).format(capture=None)
                outstanding_vm = inst.vlcnt

    return None


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
    """
    if result is None:
        return None
    if isinstance(result, str):
        return result
    # Typed Failure (from Tensile.Components.ScheduleCapture).
    return result.format(capture=None)


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
# `LocalRead.validate` through `apply_swaits` + `apply_barriers`.
# The graph-side replacement is `validate_edge_wait_coverage` over
# LR -> MFMA RAW edges (`_DSLoadRule` writes vgpr; `_MFMARule` reads
# vgpr). Migrated coverage lives in
# `Tensile/Tests/unit/test_validate_lr_before_mfma_graph.py`.


def add_pack_constraints(timeline: Timeline, ctx: 'ValidationContext') -> None:
    """
    Ensure that the Packs start and end at the correct indices.
    The pack commands take the data loaded into registers by LR commands and manipulate it in various ways to prepare it for the VMFMA instructions.

    There are several restrictions placed on Pack instructions:
    1. For all gemm types (tf32, bf16, etc.) the Pack instructions must be issued after the data is guaranteed to be loaded into the registers (guaranteed by SWaitCnt instructions). And they must finish before the first VMFMA that uses their results.
    2. For fp32 GEMMs, there are additional ordering restrictions on the Pack instructions.
    The CDNA 4 ISA section 7.6 quad-cycle visibility restrictions for fp32
    Pack instructions used to be enforced here via `estimate_quad_cycles`
    plus a per-Pack timing check; bead `8nz` migrated those to the graph
    side (`_quad_cycle_gap_ok` / `_cvt_to_mfma_gap_ok` /
    `_mfma_pack_to_cvt_gap_ok` in ScheduleCapture.py, all routed through
    `_classify_edge_coverage`).
    """
    if ctx.kernel.get("UseF32XEmulation", False) and not ctx.kernel.get("UseDirect32XEmulation", False):
        printWarning("UseF32XEmulation is set to True but UseDirect32XEmulation is not set to True. Skipping CMS validation for packs.")
        return
    apply_swaits(timeline)
    hook_up_packs(timeline, ctx.kernel, ctx.mfma_reorder)


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


def verify_correct_number_of_instructions(schedule_info: 'ScheduleInfo', context: 'ValidationContext', code_path: int) -> tuple[bool, str]:
    """
    Verify that the number of instructions in the schedule is correct for a single code path.
    """
    for instruction_name in schedule_info.optSchedule.keys():
        schedule = schedule_get(instruction_name, code_path, schedule_info)

        len_actual = len(schedule)
        len_expected = len(context.id_map[instruction_name])
        if len_actual != len_expected:
            return False, f"{instruction_name} has {len_actual} instructions, but {len_expected} instructions are required."
    return True, ""


# ---------------------------------------------------------------------------
# Concern-based rule wrappers (stage 11 → isValid wiring)
# ---------------------------------------------------------------------------

class InstructionCountRule(StructuralRule):
    def concerns(self) -> set[ValidationConcern]:
        return {ValidationConcern.SCHEDULE_COMPLETENESS}

    def run(self, schedule_info, context, code_path):
        return verify_correct_number_of_instructions(schedule_info, context, code_path)


# `LRDataReadyRule` was deleted in bead `ola.3` phase-2. The
# LR-data-ready concern (LR -> MFMA dscnt drain) is enforced graph-side
# by `validate_edge_wait_coverage` over LR -> MFMA RAW edges. The
# `ValidationConcern.LR_DATA_READY` enum value is preserved for
# downstream compatibility but no rule currently dispatches on it.


class PackDataReadyRule(ValidationRule):
    def concerns(self) -> set[ValidationConcern]:
        return {ValidationConcern.PACK_DATA_READY, ValidationConcern.QUAD_CYCLE_TIMING}

    def run(self, timeline, ctx):
        add_pack_constraints(timeline, ctx)
        return validate_timeline(timeline)


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
    PackDataReadyRule(),
    # `GRAfterLRRule()` deleted in bead `ola.2` phase 2 — see comment block above.
]

STRUCTURAL_RULES: list[StructuralRule] = [
    InstructionCountRule(),
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

