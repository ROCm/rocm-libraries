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
from copy import deepcopy
from enum import Enum, auto
from typing import ClassVar, Optional

from rocisa.instruction import SWaitCnt, SBarrier
from Tensile.Common.Utilities import printWarning
from Tensile.Components.CMSValidatorDialect import (
    CDNA4_DIALECT,
    ValidatorDialect,
    resolve_dialect,
)


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

# Sentinel values for "infinitely far" positions. Values chosen to be well beyond
# any realistic schedule size (num_vmfma is typically ~48-200).
POSITION_INF = SchedulePosition(loop_index=9_999, vmfma_index=9_999, sub_index=9_999)
POSITION_NEG_INF = SchedulePosition(loop_index=-9_999, vmfma_index=-9_999, sub_index=-9_999)

class ValidatorPass(Enum):
    # Structural checks
    VERIFY_CORRECT_NUMBER_OF_INSTRUCTIONS = auto()
    VERIFY_ASCENDING_ORDER = auto()
    VERIFY_SCC_OVERLAP = auto()
    # Numerical-correctness checks (statically detectable bug patterns
    # that produce wrong answers but compile cleanly):
    #   * VERIFY_NO_LR_LW_LDS_RACE: For LDSB=1 (single LDS buffer),
    #     LWs must not begin until the LAST LR of the same iter has
    #     issued. Otherwise iter K-1's data is partially overwritten
    #     by iter K's LWs while iter K-1's LRs are still reading.
    #     For LDSB=0 (double-buffered), the schedule must contain
    #     LWSA/LWSB swaps that flip the write address before LWs.
    VERIFY_NO_LR_LW_LDS_RACE = auto()
    # Timeline passes
    ADD_LOCAL_READ_CONSTRAINTS = auto()
    ADD_PACK_CONSTRAINTS = auto()
    ADD_GR_NOT_TOO_EARLY_CONSTRAINTS = auto()
    ADD_GR_FINISH_BEFORE_LR_CONSTRAINTS = auto()


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

# --- TF32 Pack Index Ranges (within a group) ---
# Regular TF32 (groups of 24)
TF32_CVT0_END = 4                # Indices 0..3 are CVT0
TF32_MIDDLE_16_START = 4         # Indices 4..19 are middle-16
TF32_MIDDLE_16_END = 20          # (exclusive)
# TF32_CVT1 occupies indices 20..23

# 4x4 MFMA TF32 (groups of 10)
TF32_4X4_MFMA_START = 4          # Indices 4..5 are 4x4 MFMAs
TF32_4X4_MFMA_END = 6            # (exclusive)
# CVT0: 0..3, CVT1: 6..9

# --- Quad-Cycle Timing (CDNA 4 ISA section 7.6) ---
QUAD_CYCLES_CVT_BEFORE_MFMA = 2          # CVT packs need 2 quad-cycles before MFMA can use result
QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1 = 5     # 4x4 MFMA needs 5 quad-cycles before CVT1 can use result
QUAD_CYCLES_STANDARD_MFMA_FINISH = 3      # Standard MFMA takes 3 quad-cycles to finish after issue
QUAD_CYCLES_MFMA_4X4_FINISH = 1           # 4x4 MFMA takes 1 quad-cycle to finish after issue

# --- MFMA Type-Switch Thresholds ---
MFMA_TYPE_SWITCH_THRESHOLD_FROM_STANDARD = 5  # Min gap before type switch from standard MFMA
MFMA_TYPE_SWITCH_THRESHOLD_FROM_4X4 = 3       # Min gap before type switch from 4x4 MFMA

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
    needed_by: ValidatorInstruction = field(default_factory=lambda: MFMA(name="MFMA", issued_at=POSITION_INF))
    guaranteed_by: SchedulePosition = field(default_factory=lambda: POSITION_INF)

    def done_idx(self) -> SchedulePosition:
        return self.guaranteed_by

    def validate(self) -> Optional[str]:
        # For when local reads are not being guaranteed by a particular pass.
        if self.needed_by.issued_at == POSITION_INF:
            return None

        # Needs to be guaranteed BEFORE the index at which it's needed since the
        # SWaitCnt is issued AFTER the vmfma.
        if self.guaranteed_by < self.needed_by.issued_at:
            return None

        issued_at = self.issued_at.vmfma_index
        needed_by = self.needed_by.issued_at.vmfma_index
        if self.guaranteed_by == POSITION_INF:
            return f"{self.name} @ idx={issued_at} is not valid. There are no guarantees on when it will be done."

        guaranteed_by = self.guaranteed_by.vmfma_index

        context_str = ""
        if self.needed_by.issued_at.loop_index > self.issued_at.loop_index:
            context_str = " (of next iteration)"

        return f"{self.name} @ idx={issued_at} issued too late, must be guaranteed before {self.needed_by.name} @ idx={needed_by}{context_str} but only guaranteed @ idx={guaranteed_by}."

@dataclass
class MatrixInst(ValidatorInstruction):
    """Architecture-agnostic matrix-instruction placeholder.

    Used to mark matrix-instruction slots (MFMA on CDNA, WMMA on RDNA 3.5)
    in the timeline. The class-level ``mfma_finish_cycles`` is the CDNA 4
    default (ISA §7.6 -> 3 quad-cycles). Architecture-specific timing is
    carried by the validator's active ``ValidatorDialect`` for dialect-aware
    passes; this ClassVar survives for the existing CDNA-4 code paths that
    read it directly off the instruction object.
    """
    mfma_finish_cycles: ClassVar[int] = QUAD_CYCLES_STANDARD_MFMA_FINISH

    def validate(self) -> Optional[str]:
        return None


# Back-compat alias. The legacy name ``MFMA`` is kept for one release so
# external callers (tests, diagnostics that match on ``instruction.name``)
# keep working while the codebase transitions to ``MatrixInst``.
MFMA = MatrixInst

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

    def validate(self) -> Optional[str]:
        issued_at = self.issued_at.vmfma_index

        # Collapse must_start_after list to the single latest constraint
        effective_must_start_after = max(
            self.must_start_after, key=lambda c: c.done_idx()
        ) if self.must_start_after else MFMA(name="MFMA", issued_at=POSITION_NEG_INF)

        if effective_must_start_after.done_idx() < self.issued_at < self.needed_by.done_idx():
            return None

        # Issued too early
        if self.issued_at < effective_must_start_after.done_idx():
            must_start_after_at = effective_must_start_after.done_idx().vmfma_index
            must_start_after_issued_at = effective_must_start_after.issued_at.vmfma_index
            return f"{self.name} @ idx={issued_at} issued too early, must be issued after idx={must_start_after_at} (because of {effective_must_start_after.name} issued @ idx={must_start_after_issued_at})."

        # Issued too late
        if self.issued_at >= self.needed_by.issued_at:
            needed_by_at = self.needed_by.issued_at.vmfma_index
            return f"{self.name} @ idx={issued_at} issued too late, must be issued before {self.needed_by.name} @ idx={needed_by_at}."

        return f"{self.name} at index {issued_at} is not valid."

@dataclass
class TimedPack(Pack):
    """Pack with quad-cycle timing constraints (TF32 CVT and MFMA packs)."""
    # The minimum number of quad-cycles that must pass before the result of this pack is used.
    # Measure from the point that this Pack is finished being issued.
    # See section 7.6 of the CDNA 4 ISA
    min_quad_cycles_before_result_used: int = 0
    # The estimated number of quad-cycles that passed between the pack being issued and the result being used.
    # This is a lower bound estimate (does not account for most stalls and such).
    estimated_quad_cycles_before_result_used: int = 0

    def validate(self) -> Optional[str]:
        error = super().validate()
        if error:
            return error
        if self.estimated_quad_cycles_before_result_used < self.min_quad_cycles_before_result_used:
            issued_at = self.issued_at.vmfma_index
            needed_by_at = self.needed_by.issued_at.vmfma_index
            return f"{self.name} @ idx={issued_at} has too little gap between it and {self.needed_by.name} @ idx={needed_by_at}. Expected at least {self.min_quad_cycles_before_result_used} quad-cycles but only {self.estimated_quad_cycles_before_result_used} passed."
        return None

@dataclass
class CVTPack(TimedPack):
    """TF32 CVT0/CVT1 packs (v_cvt_pk_bf16_f32). Type marker for isinstance dispatch."""
    pass

@dataclass
class MiddlePack(Pack):
    """Middle-16 packs in TF32 groups of 24. Have pair constraints for shared temp VGPR."""
    pair_consumer: Optional['MiddlePack'] = None
    next_scheduled_middle_16: Optional['MiddlePack'] = None

    def validate(self) -> Optional[str]:
        error = super().validate()
        if error:
            return error
        if self.pair_consumer:
            assert self.next_scheduled_middle_16, "Pair leader must have a next_middle_16_in_schedule."
            if not (self.next_scheduled_middle_16 is self.pair_consumer):
                issued_at = self.issued_at.vmfma_index
                next_issued_at = self.next_scheduled_middle_16.issued_at.vmfma_index
                pair_issued_at = self.pair_consumer.issued_at.vmfma_index
                return f"{self.name} @ idx={issued_at} has wrong interleaving. Should have been followed by {self.pair_consumer.name} @ idx={pair_issued_at} but was followed by {self.next_scheduled_middle_16.name} @ idx={next_issued_at}."
        return None

@dataclass
class MFMAPack(TimedPack, MFMA):
    """A v_mfma_f32_4x4x4_16b_bf16 instruction used in TF32 4x4 emulation pack groups.

    These appear at indices TF32_4X4_MFMA_START..TF32_4X4_MFMA_END within each group
    of PACK_GROUP_SIZE_TF32_4X4. They are real MFMA instructions but participate in
    the pack dependency chain (CVT0 -> MFMAPack -> CVT1).

    Inherits from both TimedPack and MFMA:
    - isinstance(x, Pack) is True — works with pack gathering, filtering, type hints
    - isinstance(x, TimedPack) is True — has quad-cycle timing constraints
    - isinstance(x, MFMA) is True — captures "this IS an MFMA" semantics
    """
    # Override MFMA's finish cycles for 4x4 timing
    mfma_finish_cycles: ClassVar[int] = QUAD_CYCLES_MFMA_4X4_FINISH

    # NOTE: min_quad_cycles_before_result_used is NOT overridden here.
    # It keeps TimedPack's default (0) and is set by _handle_min_pack_quad_cycles
    # only when the constraint is active (when local reads exist).
    #
    # NOTE: validate() is NOT overridden here. The MRO chain
    # (TimedPack.validate → Pack.validate) handles MFMAPack correctly.


@dataclass
class GlobalRead(ValidatorInstruction):
    swap_global_read_order: bool
    needed_by: ValidatorInstruction = field(default_factory=lambda: MFMA(name="MFMA", issued_at=POSITION_INF))
    guaranteed_by: SchedulePosition = field(default_factory=lambda: POSITION_INF)
    barriered_at: list[SchedulePosition] = field(default_factory=list)
    must_start_after: list[ValidatorInstruction] = field(default_factory=list)
    must_start_after_barriered_at: list[SchedulePosition] = field(default_factory=list)

    def done_idx(self) -> SchedulePosition:
        return self.guaranteed_by

    def validate(self) -> Optional[str]:
        # Check must_start_after constraint (GR must start after LR0s are done)
        must_start_after_error = self._validate_must_start_after()
        if must_start_after_error:
            return must_start_after_error

        # Check needed_by constraint (GR must finish before LR1/3)
        needed_by_error = self._validate_needed_by()
        if needed_by_error:
            return needed_by_error

        return None

    def _validate_must_start_after(self) -> Optional[str]:
        """Validate all must_start_after constraints."""
        for constraint in self.must_start_after:
            if constraint.done_idx() == POSITION_NEG_INF:
                continue

            name = self._name()
            issued_at = self.issued_at.vmfma_index
            constraint_done = constraint.done_idx()

            # 1. Check ordering: GR must be issued after constraint is done
            if self.issued_at <= constraint_done:
                context_str = ""
                if constraint_done.loop_index > self.issued_at.loop_index:
                    context_str = " (of next iteration)"
                return (
                    f"{name} @ idx={issued_at} is issued too early. "
                    f"Must be issued after idx={constraint_done.vmfma_index}{context_str}, "
                    f"which is when {constraint.name} is guaranteed done."
                )

            # 2. LocalRead constraints require an SBarrier (cross-wave LDS sync)
            if isinstance(constraint, LocalRead):
                if not any(constraint_done < b < self.issued_at
                           for b in self.must_start_after_barriered_at):
                    return (
                        f"There is an SBarrier missing between the SWaitCnt "
                        f"@ idx={constraint_done.vmfma_index} (which guarantees "
                        f"{constraint.name} from idx={constraint.issued_at.vmfma_index} "
                        f"to done) and the {name} @ idx={issued_at}. "
                        f"Order must be {constraint.name} -> SWait -> SBarrier -> {name}."
                    )

        return None

    def _validate_needed_by(self) -> Optional[str]:
        """Validate: GR -> SWait -> SBarrier -> LR1"""
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
            return f"{name} @ idx={issued_at} is not valid. There are no guarantees on when it will be done."

        # NOTE: Must do it after the check above to guard against infinity.
        guaranteed_by = self.guaranteed_by.vmfma_index

        # 2. No Barrier
        if len(self.barriered_at) == 0:
            return f"{name} @ idx={issued_at} is not valid. There is no SBarrier acting on it."

        # 3. Guaranteed after needed
        if self.guaranteed_by > self.needed_by.issued_at:
            return f"{name} @ idx={issued_at} is not valid. It is guaranteed by the SWait @ idx={guaranteed_by} which is after the first corresponding {self.needed_by.name} @ idx={needed_by}. Order must be {name} -> SWait -> SBarrier -> {self.needed_by.name}."

        # 4. No Barrier between SWait and LR1
        if not any(self.guaranteed_by < barriered_at < self.needed_by.issued_at for barriered_at in self.barriered_at):
            return f"{name} @ idx={issued_at} is not valid. No SBarrier between SWait @ idx={guaranteed_by} and {self.needed_by.name} @ idx={needed_by}. Order must be {name} -> SWait -> SBarrier -> {self.needed_by.name}."

        # TODO: Did we miss a case and will we ever end up here?
        return f"{name} @ idx={issued_at} is not valid. issued @ idx={issued_at}, guaranteed @ idx={guaranteed_by}, barriered @ idx={[b.vmfma_index for b in self.barriered_at]}, needed @ idx={needed_by} is not valid."

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

    def validate(self) -> Optional[str]:
        if self._is_valid():
            return None
        return f"SWait at index {self.issued_at.vmfma_index} is invalid: dscnt={self.dscnt}, vlcnt={self.vlcnt}, vscnt={self.vscnt}, issued_at={self.issued_at.vmfma_index}."

@dataclass
class Barrier(ValidatorInstruction):
    comment: str

    def validate(self) -> Optional[str]:
        return f"Barrier at index {self.issued_at.vmfma_index} is not valid. Must be >= -1." if self.issued_at.vmfma_index < -1 else None

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


def create_unified_timeline(
    schedule_info: 'ScheduleInfo',
    kernel: 'Solution',
    code_path: int,
    dialect: ValidatorDialect = CDNA4_DIALECT,
) -> 'Timeline':
    """Create a single Timeline with all instruction types.

    If the dialect provides a ``timeline_factory``, it is used to construct
    the Timeline. Otherwise the default ``Timeline`` class (CDNA-4 DTL=1
    layout) is used, which is the behavior the CDNA 4 dialect relies on.
    The RDNA 3.5 dialect supplies its own factory that yields an
    ``RDNA35WMMATimeline`` configured for DTL=0 single-stream GR.
    """
    available_names = set(schedule_info.optSchedule.keys())
    names_to_add = [n for n in ALL_INSTRUCTION_NAMES if n in available_names]
    factory = dialect.timeline_factory
    if factory is None:
        return Timeline(names_to_add, code_path, schedule_info, kernel, dialect)
    return factory(names_to_add, code_path, schedule_info, kernel, dialect)


class Timeline:
    """Base timeline.

    Policy knobs (override in subclasses):
      ``REQUIRES_DIRECT_TO_LDS``: when True, the timeline asserts
          ``kernel['DirectToLds']`` during population. CDNA-4 kernels use
          DTL=1 so the assert is active there; RDNA 3.5 WMMA kernels use
          DTL=0 so the subclass turns this off.
      ``GR_HAS_M0_POINTER_UPDATES``: when True, treat every even index in
          ``GRA``/``GRB`` streams as an ``m0`` pointer update (skip it and
          require an even stream length). On DTL=1 CDNA-4 each GR is
          preceded by an SMEM m0 write; on DTL=0 RDNA 3.5 GRs are plain
          VMEM loads with no m0 interleave.
      ``WAVEFRONT_SIZE``: informational; structural checks are currently
          independent of wave size, but stream-length auditing uses it to
          express the wave32/wave64 ratio authored schedules must match.

    The default values preserve historical CDNA-4 behavior for every
    test and caller of the bare ``Timeline`` class.
    """

    REQUIRES_DIRECT_TO_LDS: ClassVar[bool] = True
    GR_HAS_M0_POINTER_UPDATES: ClassVar[bool] = True
    WAVEFRONT_SIZE: ClassVar[int] = 64
    # When True, the CDNA-4 "can't mix LR1s and LR3s" assert is enforced.
    # CDNA-4 kernels either use a two-sub-iter (LR0/LR1) or a four-sub-iter
    # layout (LR0..LR3) layout but never mix both LR1 and LR3. RDNA 3.5
    # WMMA kernels with DepthU > 2 * matrixInstK use LR0..LR(numSubIter-1)
    # including both LR1 and LR3, so the subclass disables this check.
    ENFORCE_LR1_LR3_EXCLUSIVITY: ClassVar[bool] = True

    def __init__(self, instruction_names_to_add: list[str], code_path: int, schedule_info: 'ScheduleInfo', kernel: 'Solution', dialect: ValidatorDialect = CDNA4_DIALECT):
        """
        Create a timeline from the provided schedule_info which contains only the instructions inside `instruction_names_to_add`.
        Organized as a list of lists indexed by vmfma_index + 1.

        The +1 is required in order to handle the special case of idx=-1, which is at timeline[0].
        idx=-1 is special case that occurs BEFORE the first VMFMA but AFTER the last VMFMA.

        Multiple timelines are created under the hood:
        1. The previous main loop iteration (iteration N-1).
        2. The main loop (iteration N).
        3. The No Global load loop (iteration N+1)
        4. The No Local load loop (iteration N+2)

        Two main loop iterations are created to properly validate cross-iteration effects within the mainloop, especially GRs which start in one iteration and complete in another.

        Args:
            instruction_names_to_add:   The list of instruction names to add to the timeline.
            code_path:                  The code path to create a timeline out of.
            schedule_info:              The schedule information to add to the timeline.
            kernel:                     The kernel to add to the timeline.
            num_iterations:             Number of iterations to consider for cross-iteration effects (default 2).
        """
        
        available_keys = schedule_info.optSchedule.keys()
        if self.ENFORCE_LR1_LR3_EXCLUSIVITY:
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
        self._applied_passes: set[Callable[['Timeline', 'ValidatorPassContext'], None]] = set()

        # Architecture dialect (drives pack-group layout, matrix-instruction
        # timing, SCC cluster shape). Defaults to CDNA4 so existing
        # call-sites keep working.
        self.dialect: ValidatorDialect = dialect

        # Populate the timeline with instructions
        self._populate_instructions(instruction_names_to_add, code_path, schedule_info, kernel)
        self._linearize_timeline()
    
    def _populate_instructions(self, instruction_names_to_add: list[str], code_path: int, schedule_info: 'ScheduleInfo', kernel: 'Solution') -> None:
        """
        Populates all timelines with deep copies of the instructions from schedule_info.
        """
        if self.REQUIRES_DIRECT_TO_LDS:
            assert kernel["DirectToLds"], "Only DirectToLds cases are supported by validator."

        swap_global_read_order = kernel["SwapGlobalReadOrder"]
        is_tf32_emulation = kernel.get("UseF32XEmulation", False)
        is_4x4mfma_tf32 = kernel.get("UseMFMAF32XEmulation", False)

        # Explicitly add MFMAs to timeline.
        # Do at the top here so they are the first ones scheduled at each vmfma index.
        for i_vmfma in range(self.num_vmfma):
            if schedule_info.mfmaReorder:
                i_vmfma = schedule_info.mfmaReorder[i_vmfma]
                
            mfma = MFMA(name="MFMA", issued_at=POSITION_NEG_INF)
            self._insert(i_vmfma, mfma, kernel)

        # NOTE: Relative ordering of instructions must be preserved.
        #       Order dictates the order in which instructions are scheduled if they are scheduled at the same vmfmaindex.
        #
        # Dialect override: when ``sync_insert_last`` is set (RDNA 3.5 WMMA),
        # the SYNC key is processed AFTER every other instruction stream so
        # that SWaitCnt/SBarrier instructions end up at the highest
        # ``sub_index`` of their ``vmfma_index`` bucket. This makes
        # ``apply_swaits``'s backward walk from a SWaitCnt reach LRs/GRs
        # authored at the same boundary, which matches the gfx1151 CMS
        # author's intent (SWaitCnt at vmfma=N covers all same-iter LDS
        # traffic at vmfma<=N). Codegen is unaffected -- this only reorders
        # the validator's in-memory timeline model.
        schedule_keys = list(schedule_info.optSchedule.keys())
        if self.dialect.sync_insert_last and "SYNC" in schedule_keys:
            schedule_keys.remove("SYNC")
            schedule_keys.append("SYNC")
        for name in schedule_keys:
            if name not in instruction_names_to_add:
                continue

            if name == "SYNC":
                for idx_sync, (idx_vmfma, sync) in enumerate(zip(schedule_get(name, code_path, schedule_info), schedule_info.syncCode)):
                    assert idx_vmfma >= -1, f"Code path {code_path}: SWaitCnt at index {idx_sync} is not valid. Must be >= -1."
                    
                    if isinstance(sync, SWaitCnt):
                        sync_instruction = SWait(name="SWaitCnt", issued_at=POSITION_NEG_INF, dscnt=sync.dscnt, vlcnt=sync.vlcnt, vscnt=sync.vscnt, comment=sync.comment)
                    elif isinstance(sync, SBarrier):
                        sync_instruction = Barrier(name="SBarrier", issued_at=POSITION_NEG_INF, comment=sync.comment)
                    else:
                        raise ValueError(f"Unexpected sync instruction type: {type(sync)}")
                    
                    self._insert(idx_vmfma, sync_instruction, kernel)
            elif name == "SNOP":
                for idx_snop, (idx_vmfma, snop) in enumerate(zip(schedule_get(name, code_path, schedule_info), schedule_info.snopCode)):
                    assert idx_vmfma >= -1, f"Code path {code_path}: SNop at index {idx_snop} is not valid. Must be >= -1."
                    # The waitState is stored as the first parameter in the rocisa SNop instruction
                    wait_state = snop.getParams()[0]
                    snop_instruction = SNop(name="SNop", issued_at=POSITION_NEG_INF, wait_state=wait_state)
                    self._insert(idx_vmfma, snop_instruction, kernel)
            elif name.startswith("LRA") or name.startswith("LRB"):
                for idx_LR, idx_vmfma in enumerate(schedule_get(name, code_path, schedule_info)):
                    assert idx_vmfma >= -1, f"Code path {code_path}: LocalRead {name} at index {idx_LR} is not valid. Must be >= -1."

                    # TODO: For ForceUnrollSubIter, need to account for register reuse and the fact that the LR0/LR1/LR3s must start after a certain point in the iteration.
                    local_read = LocalRead(name=name, issued_at=POSITION_NEG_INF, issue_index=idx_LR)
                    self._insert(idx_vmfma, local_read, kernel)
            elif name.startswith("GRInc"):
                grincs = schedule_get(name, code_path, schedule_info)
                for idx_grinc, idx_vmfma in enumerate(grincs):
                    assert idx_vmfma >= -1, f"Code path {code_path}: GRInc {name} at index {idx_grinc} is not valid. Must be >= -1."
                    grinc = GRInc(name=name, issued_at=POSITION_NEG_INF)
                    self._insert(idx_vmfma, grinc, kernel)
            elif name.startswith("GRA") or name.startswith("GRB"):
                global_reads = schedule_get(name, code_path, schedule_info)
                if self.GR_HAS_M0_POINTER_UPDATES:
                    # CDNA 4 DTL=1: every even index is an m0 pointer update,
                    # every odd index is the real buffer_load. Stream must be
                    # even-length to pair them up.
                    assert len(global_reads) % 2 == 0, f"Code path {code_path}: {name} has an odd number of indices. Must be even if DirectToLds is True."

                for idx_GR, idx_vmfma in enumerate(global_reads):
                    assert idx_vmfma >= -1, f"Code path {code_path}: GlobalRead {name} at index {idx_GR} is not valid. Must be >= -1."

                    # DTL=1 only: skip the m0 pointer update, keep the real
                    # buffer_load that follows it. On RDNA 3.5 DTL=0 the
                    # subclass sets ``GR_HAS_M0_POINTER_UPDATES = False`` and
                    # every index is a real GR.
                    if self.GR_HAS_M0_POINTER_UPDATES and idx_GR % 2 == 0:
                        continue

                    global_read = GlobalRead(name=name, issued_at=POSITION_NEG_INF, swap_global_read_order=swap_global_read_order)
                    self._insert(idx_vmfma, global_read, kernel)
            elif name.startswith("PackA") or name.startswith("PackB"):
                packs = schedule_get(name, code_path, schedule_info)

                for idx_pack, idx_vmfma in enumerate(packs):
                    assert idx_vmfma >= -1, f"Code path {code_path}: Pack {name} at index {idx_pack} is not valid. Must be >= -1."
                    pg = self.dialect.pack_graph
                    if is_4x4mfma_tf32:
                        # Dialect-driven constants (CDNA4: 10-wide, MFMAs at 4..6).
                        idx_in_group = idx_pack % pg.group_size_tf32_4x4
                        group_idx = idx_pack // pg.group_size_tf32_4x4
                        if pg.tf32_4x4_mfma_start <= idx_in_group < pg.tf32_4x4_mfma_end:
                            pack = MFMAPack(name=name, issued_at=POSITION_NEG_INF, issue_index=idx_pack, group_index=group_idx)
                        else:
                            pack = CVTPack(name=name, issued_at=POSITION_NEG_INF, issue_index=idx_pack, group_index=group_idx)
                    elif is_tf32_emulation:
                        # Dialect-driven constants (CDNA4: 24-wide, middle-16 at 4..20).
                        idx_in_group = idx_pack % pg.group_size_tf32
                        group_idx = idx_pack // pg.group_size_tf32
                        if pg.tf32_middle_16_start <= idx_in_group < pg.tf32_middle_16_end:
                            pack = MiddlePack(name=name, issued_at=POSITION_NEG_INF, issue_index=idx_pack, group_index=group_idx)
                        else:
                            pack = CVTPack(name=name, issued_at=POSITION_NEG_INF, issue_index=idx_pack, group_index=group_idx)
                    else:
                        pack = Pack(name=name, issued_at=POSITION_NEG_INF, issue_index=idx_pack)
                    self._insert(idx_vmfma, pack, kernel)
            else:
                raise NotImplementedError(f"Instruction {name} not implemented")
    
    def _insert(self, vmfma_index: int, instruction: ValidatorInstruction, kernel: 'Solution') -> None:
        """
        Add an instruction to the timeline at a given VMFMA index.
        Adds it to all relevant loops.
        Internal method used during initialization - does not re-linearize.
        """
        for loop in self.loops:
            if self._should_add(instruction, loop, kernel):
                _instruction = deepcopy(instruction)

                loop_index = self.loops.index(loop)
                sub_index = len(self._instructions_at_index[loop][vmfma_index + 1])
                _instruction.issued_at = SchedulePosition(loop_index=loop_index, vmfma_index=vmfma_index, sub_index=sub_index)

                # Adjust for NLL/NGL shifts.
                if isinstance(_instruction, SWait):
                    if _instruction.vlcnt != -1:
                        vlcnt = max(0, _instruction.vlcnt - self.vlcnt_shift[loop])
                        _instruction.vlcnt = vlcnt
                    if _instruction.dscnt != -1 and self.nll_zero_dscnt \
                       and loop in [NO_LOCAL_LOAD_LOOP]:
                        _instruction.dscnt = 0

                self._instructions_at_index[loop][vmfma_index+1].append(_instruction)

    def _should_add(self, instruction: ValidatorInstruction, loop: str, kernel: 'Solution') -> bool:
        """
        Determine if an instruction should be added to a given loop.
        """
        assert loop in self.loops, f"Invalid loop: {loop}"
        if isinstance(instruction, GlobalRead):
            # No GRs issued in NGL or NLL
            return loop == MAIN_LOOP or loop == MAIN_LOOP_PREV
        elif isinstance(instruction, GRInc):
            return loop == MAIN_LOOP or loop == MAIN_LOOP_PREV
        elif isinstance(instruction, LocalRead):
            # Only LR0s are issued in the NLL
            if loop == NO_LOCAL_LOAD_LOOP:
                return instruction.name == "LRA0" or instruction.name == "LRB0"
            return True
        elif isinstance(instruction, Pack):
            if kernel.get("UsePLRPack", False):
                # Packs1/3s correspond to the LR1/3s of this iteration.
                if loop == NO_LOCAL_LOAD_LOOP:
                    return instruction.name == "PackA0" or instruction.name == "PackB0"
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


class CDNA4DTLTimeline(Timeline):
    """Timeline specialized for CDNA 4 MFMA kernels with DirectToLds=1.

    This is the historical layout and is what the base ``Timeline`` class
    already produces (all policy knobs are at their CDNA-4 defaults). The
    explicit subclass exists so that ``CMSValidatorDialect.CDNA4_DIALECT``
    can name it as its ``timeline_factory`` without coupling to the base
    class identity.
    """

    REQUIRES_DIRECT_TO_LDS: ClassVar[bool] = True
    GR_HAS_M0_POINTER_UPDATES: ClassVar[bool] = True
    WAVEFRONT_SIZE: ClassVar[int] = 64


class RDNA35WMMATimeline(Timeline):
    """Timeline specialized for RDNA 3.5 WMMA kernels with DirectToLds=0.

    Differences from CDNA 4:
      * DTL=0: no ``assert kernel['DirectToLds']`` at construction.
      * GR is a plain wave32 ``buffer_load`` / VMEM op. There is no m0
        pointer-update interleave, so every index in ``GRA``/``GRB`` is
        a real GlobalRead and the stream length is NOT required to be
        even.
      * Wave32: ``WAVEFRONT_SIZE = 32``. Structural checks do not yet key
        off this value directly; it is consulted by the stream-length
        auditing helpers when comparing authored schedule lengths against
        the wave32 ``idMap``.

    The waitcnt counter-name handling in ``_insert`` / ``apply_swaits``
    currently reuses the CDNA 4 model. The ``VMcnt`` / ``VScnt`` /
    ``LGKMcnt`` / ``EXPcnt`` split for RDNA 3.5 can be specialized here
    later if a pass needs per-counter precision that the shared model
    cannot express.
    """

    REQUIRES_DIRECT_TO_LDS: ClassVar[bool] = False
    GR_HAS_M0_POINTER_UPDATES: ClassVar[bool] = False
    WAVEFRONT_SIZE: ClassVar[int] = 32
    # RDNA 3.5 WMMA CMS schedules (see _get_schedule_*_gfx1151 in
    # CustomSchedule.py) routinely use LR0..LR7 when DepthU/matrixInstK
    # is 8, so LR1 and LR3 legitimately coexist in the same schedule.
    ENFORCE_LR1_LR3_EXCLUSIVITY: ClassVar[bool] = False


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


@applies_only_once
def apply_must_start_after_barriers(timeline: Timeline) -> None:
    """
    Apply the effect of SBarriers to the must_start_after_barriered_at field of GlobalReads.
    For each GlobalRead, finds SBarrier instructions that occur between must_start_after.done_idx()
    and the GlobalRead's issued_at. These barriers ensure all waves have completed the LR0s.
    Timeline is modified in place.

    Args:
        timeline: The Timeline object containing the instructions.
    """
    for i_gr, gr in timeline.get_instructions_combined("GRA"):
        _apply_must_start_after_barriers_single(timeline, gr, i_gr)
    for i_gr, gr in timeline.get_instructions_combined("GRB"):
        _apply_must_start_after_barriers_single(timeline, gr, i_gr)


def _apply_must_start_after_barriers_single(timeline: Timeline, gr: GlobalRead, i_gr: int) -> None:
    """Apply must_start_after barriers for a single GlobalRead instruction."""
    lr_constraints = [c for c in gr.must_start_after
                      if isinstance(c, LocalRead)
                      and c.done_idx() != POSITION_NEG_INF]
    if not lr_constraints:
        return

    # Use min to search the widest window for barrier candidates;
    # _validate_must_start_after does per-constraint filtering afterwards.
    earliest_done = min(c.done_idx() for c in lr_constraints)

    for i_inst in range(i_gr - 1, -1, -1):
        instruction = timeline.combined_timeline[i_inst]
        if not isinstance(instruction, Barrier):
            continue
        if earliest_done < instruction.issued_at < gr.issued_at:
            gr.must_start_after_barriered_at.append(instruction.issued_at)


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


@applies_only_once
def set_lr_needed_by_for_VMFMA(timeline: Timeline, kernel: 'Solution', mfma_reorder: list[int], dialect: ValidatorDialect = CDNA4_DIALECT) -> None:
    """
    Set the needed_by field of LocalReads based on the VMFMA index they are required for.
    Timeline is modified in place.
    
    For LRA0/LRB0, the data is needed at a VMFMA index offset by num_vmfma // 2 (halfway point).
    For LRA1/LRB1, the data is needed at a VMFMA index offset by num_vmfma (next iteration).
    
    Args:
        timeline:       The Timeline object containing the instructions.
        kernel:         Solution object containing the kernel metadata.
        mfma_reorder:   Mapping between the index of a default-scheduled MFMA and its new custom assigned index.
    """

    if mfma_reorder and len(mfma_reorder) != timeline.num_vmfma:
        raise ValueError(f"Incorrect number of VMFMA indices in mfmaReorder. Expected {timeline.num_vmfma}, given {len(mfma_reorder)}.")

    n_tiles_a = kernel["MIWaveTileA"]
    n_tiles_b = kernel["MIWaveTileB"]

    n_local_reads_a = len(timeline.get_instructions("LRA0", MAIN_LOOP))
    n_local_reads_b = len(timeline.get_instructions("LRB0", MAIN_LOOP))

    mfma_for_linear_index: dict[int, MFMA] = {
        mfma.issued_at.loop_index * timeline.num_vmfma + mfma.issued_at.vmfma_index: mfma
        for _, mfma in timeline.get_instructions_combined("MFMA")
    }

    for i_loop, loop in enumerate(timeline.loops):
        loop_offset = timeline.num_vmfma * i_loop
        for instruction_name in timeline.get_instruction_names():
            if not instruction_name.startswith("LRA") and not instruction_name.startswith("LRB"):
                continue
            local_reads = timeline.get_instructions(instruction_name, loop)
            for lr_idx, (_, lr) in enumerate(local_reads):
                needed_by = lr_needed_by_mfma(
                    local_read_name=lr.name,
                    lr_idx=lr_idx,
                    num_vmfma=timeline.num_vmfma,
                    mfma_reorder=mfma_reorder,
                    n_tiles_a=n_tiles_a, n_tiles_b=n_tiles_b,
                    n_local_reads_a=n_local_reads_a,
                    n_local_reads_b=n_local_reads_b,
                    force_unroll_sub_iter=kernel.get("ForceUnrollSubIter", False),
                    use_f32x_emulation=kernel.get("UseF32XEmulation", False),
                    dialect=dialect)
                lr.needed_by = mfma_for_linear_index[needed_by + loop_offset]


@applies_only_once
def set_gr_needed_by_from_lrs(timeline: Timeline, swap_global_read_order: bool) -> None:
    """
    Set the needed_by field of GlobalReads based on the LR1/3 instructions.
    If GRA or GRB is missing, this function will NOT error out.
    If either GRA or GRB is present, the corresponding LR1/3 instruction must be present.
    
    Args:
        timeline: The Timeline object containing the instructions.
        swap_global_read_order: Whether global read order is swapped.
    """
    # If the global read order is swapped, we need to swap the target indices since GRAs actually load B and GRBs actually load A.
    target_names = {"GRA": "LRA1", "GRB": "LRB1"}
    
    if "LRA1" not in timeline.get_instruction_names():
        assert "LRA3" in timeline.get_instruction_names(), "LRA3 must be present if LRA1 is not"
        target_names["GRA"] = "LRA3"
        target_names["GRB"] = "LRB3"
    
    if swap_global_read_order:
        target_names["GRA"], target_names["GRB"] = target_names["GRB"], target_names["GRA"]

    for i_loop, loop in enumerate(timeline.loops):
        for gr_name, target_name in target_names.items():
            # NOTE: For the NGL and NLL loops, we don't have any GRs being issued at all.
            #       Also, for testing purposes we may ommit GRAs or LRA1s to improve readability.
            #       Another validator pass will ensure that they are present if they are needed.
            grs = timeline.get_instructions(gr_name, loop)
            if not grs:
                continue

            # NOTE: Can't index out of bounds since NGL and NLL loops don't issue GRs, check above would fail.
            target = timeline.get_instructions(target_name, timeline.loops[i_loop + 1])
            if len(target) == 0:
                raise ValueError(f"No {target_name} instructions found in schedule.")
            
            _, LR_target = target[0]
            for _, gr in grs:
                gr.needed_by = LR_target

@applies_only_once
def set_gr_must_start_after_from_lr0s(timeline: Timeline, swap_global_read_order: bool, dtl_plus_lds_buf: bool = False) -> None:
    """
    Set the must_start_after field of GlobalReads based on the last LR0 that shares their LDS block.

    Standard case (dtl_plus_lds_buf=False):
        GRs in iteration N write (DDR->LDS) to the same LDS block that LR0s of iteration N read from.
        Each GR must start after the last same-iteration LR0 is guaranteed done.

    DtlPlusLdsBuf case (dtl_plus_lds_buf=True):
        GRs in iteration N write to a different LDS block than same-iteration LR0s read from,
        so there is no same-iteration dependency. However, GRs in iteration N write to the LDS
        block that LR0s from iteration N-1 were reading from, creating a cross-iteration dependency.
        Each GR must start after the last previous-iteration LR0 is guaranteed done.

    If SwapGlobalReadOrder is True, GRA loads B so the first GRA must start after the last LRB0,
    and the first GRB must start after the last LRA0.

    The LR0's done_idx() is its guaranteed_by (set by apply_swaits), which is the SWaitCnt index.

    Args:
        timeline: The Timeline object containing the instructions.
        swap_global_read_order: Whether global read order is swapped.
        dtl_plus_lds_buf: Whether DtlPlusLdsBuf is enabled (cross-iteration dependency).
    """
    target_names = {"GRA": "LRA0", "GRB": "LRB0"}

    if swap_global_read_order:
        target_names["GRA"], target_names["GRB"] = target_names["GRB"], target_names["GRA"]

    for i_loop, loop in enumerate(timeline.loops):
        for gr_name, lr0_name in target_names.items():
            grs = timeline.get_instructions(gr_name, loop)
            if not grs:
                continue

            if dtl_plus_lds_buf:
                # GRs write to a different LDS block than same-iteration LR0s.
                # The dependency is against the previous iteration's LR0s instead.
                if i_loop == 0:
                    continue  # No previous iteration available (ML-1)
                lr0s = timeline.get_instructions(lr0_name, timeline.loops[i_loop - 1])
            else:
                lr0s = timeline.get_instructions(lr0_name, loop)

            if not lr0s:
                continue

            # Pick the LR0 that finishes last (highest guaranteed_by)
            last_lr0 = max((lr0 for _, lr0 in lr0s), key=lambda lr0: lr0.guaranteed_by)
            for _, gr in grs:
                gr.must_start_after.append(last_lr0)

@applies_only_once
def set_gr_must_start_after_from_grinc(timeline: Timeline, swap_global_read_order: bool) -> None:
    """
    Set the must_start_after constraint of GlobalReads based on the last GRInc
    that increments their address pointer.

    GRIncA always increments A's pointer, GRIncB always increments B's pointer.
    With SwapGlobalReadOrder: GRA loads B (uses GRIncB), GRB loads A (uses GRIncA).

    This is an ordering-only constraint (no SBarrier needed) since GRInc and GR
    are scalar/VMEM instructions within the same wave.
    """
    target_names = {"GRA": "GRIncA", "GRB": "GRIncB"}

    if swap_global_read_order:
        target_names["GRA"], target_names["GRB"] = target_names["GRB"], target_names["GRA"]

    for loop in timeline.loops:
        for gr_name, grinc_name in target_names.items():
            grs = timeline.get_instructions(gr_name, loop)
            if not grs:
                continue

            grincs = timeline.get_instructions(grinc_name, loop)
            if not grincs:
                continue

            # Pick the GRInc that finishes last (highest issued_at)
            last_grinc = max((grinc for _, grinc in grincs), key=lambda g: g.done_idx())

            for _, gr in grs:
                gr.must_start_after.append(last_grinc)


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
            assert len(cvt_packs) == 8, f"Expected 8 CVT packs per group, got {len(cvt_packs)}"
            assert len(mfma_packs) == 2, f"Expected 2 MFMA packs per group, got {len(mfma_packs)}"
            # CVT0 come before CVT1 by construction order (sorted by issue_index)
            cvt0 = cvt_packs[:4]
            cvt1 = cvt_packs[4:]
            assert cvt0[-1].issue_index < cvt1[0].issue_index, "CVT0 packs must have lower issue_index than CVT1 packs"

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
            assert len(cvt_packs) == 8, f"Expected 8 CVT packs per group, got {len(cvt_packs)}"
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
       

def _handle_min_pack_quad_cycles(packs: list[Pack], dialect: ValidatorDialect = CDNA4_DIALECT) -> None:
    """
    Set the min_quad_cycles_before_result_used field for TimedPack instructions.
    This is used to enforce timing constraints for TF32 emulation modes.
    Only TimedPack subclasses (CVTPack, MFMAPack) have timing fields;
    MiddlePack and plain Pack are skipped.

    Args:
        packs: List of Pack instructions to set minimum quad-cycles for.
        dialect: Architecture dialect that owns the quad-cycle constants.
            Defaults to the CDNA 4 dialect, whose values equal the historical
            module-level constants byte-for-byte.
    """
    for pack in packs:
        if isinstance(pack, MFMAPack):
            # 4x4 MFMAs need 5 quad-cycles before CVT1 can use result (CDNA 4 ISA §7.6)
            pack.min_quad_cycles_before_result_used = dialect.timing.mfma_4x4_before_cvt1
        elif isinstance(pack, CVTPack):
            # CVT packs need 2 quad-cycles before MFMAs can use their results (CDNA 4 ISA §7.6)
            pack.min_quad_cycles_before_result_used = dialect.timing.cvt_before_mfma
        # All other packs have no timing constraints

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
    pack_dependencies: dict[int, list[int]] = {
        # First 4 packs (v_cvt_pk_bf16_f32) depend on local reads only, and are not included
        0: [], 1: [], 2: [], 3: [],
        # Middle 16 packs (v_cvt_f32_bf16 + v_sub_f32 pairs) - error term calculation
         4: [0],  5: [ 4],  6: [0],  7: [ 6],
         8: [1],  9: [ 8], 10: [1], 11: [10],
        12: [2], 13: [12], 14: [2], 15: [14],
        16: [3], 17: [16], 18: [3], 19: [18],
        # Final 4 packs (v_cvt_pk_bf16_f32) - pack error terms
        20: [17, 19],
        21: [13, 15, 20],
        22: [ 9, 11, 21],
        23: [ 5,  7, 22],
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

def _hook_up_packs_f32_mfma(packs: list[Pack], local_reads: list[LocalRead]) -> None:
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
    """
    # Sort by index in the list of pack instructions rather than by the mfma_index they are placed at.
    # This is necessary to handle inter-pack dependencies.
    packs = sorted(packs, key=lambda x: x.issue_index)

    # Group packs by group_index (computed at construction time)
    pack_groups_map: dict[int, list[Pack]] = defaultdict(list)
    for pack in packs:
        pack_groups_map[pack.group_index].append(pack)
    n_pack_groups = len(pack_groups_map)

    assert len(local_reads) % n_pack_groups == 0, "Case not supported: Different number of LRs for each Pack group."
    n_lrs_per_group = len(local_reads) // n_pack_groups

    # NOTE: Assuming that all LRs are of the same width.
    vgprs_per_local_read = VGPRS_PER_CONVERSION_GROUP // n_lrs_per_group

    # Partial Pack->Pack dependency graph within a group of 10.
    # Key: pack index (0-9), Value: list of pack indices it depends on.
    # Empty list means it depends on local reads only (CVT0 packs).
    # NOTE: Does not handle the quad-cycle spacing dependencies between packs and MFMAs.
    pack_dependencies: dict[int, list[int]] = {
        # First 4 packs only depend on local reads.
        0: [], 1: [], 2: [], 3: [],
        # Middle 2 Packs are vmfma and depend on the previous 4 packs.
        4: [0, 1],
        5: [2, 3],
        # Last 2 packs are vmfma and depend on the previous 2 packs.
        6: [5],
        7: [5, 6],
        8: [4, 7],
        9: [4, 8],
    }

    for group_idx in sorted(pack_groups_map.keys()):
        start = group_idx * n_lrs_per_group
        end = start + n_lrs_per_group
        local_reads_for_group = local_reads[start:end]

        pack_group = pack_groups_map[group_idx]

        # Set must_start_after
        for pack_idx, pack in enumerate(pack_group):
            dependencies = pack_dependencies[pack_idx]
            if not dependencies:
                # CVT0 packs: depend only on local reads.
                first_lr = (pack_idx * 2) // vgprs_per_local_read
                last_lr = (pack_idx * 2 + 1) // vgprs_per_local_read
                pack_lrs = local_reads_for_group[first_lr:last_lr + 1]
                latest_lr = max(pack_lrs, key=lambda lr: lr.done_idx())
                pack.must_start_after.append(latest_lr)
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

@applies_only_once
def hook_up_packs(timeline: Timeline, kernel: 'Solution', mfma_reorder: list[int], dialect: ValidatorDialect = CDNA4_DIALECT) -> None:
    """
    Set the needed_by fields 
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

        # 3. Hook up the needed_by and must_start_after fields
        for pack_name, packs in packs_by_name.items():
            local_reads = _get_lrs_for_pack(timeline, use_plr_pack, pack_name, loop)
            if not local_reads:
                continue

            if is_tf32_emulation:
                if is_4x4mfma_tf32:
                    _hook_up_packs_f32_mfma(packs, local_reads)
                else:
                    _hook_up_packs_f32(packs, all_middle_16_packs, local_reads)
                _handle_min_pack_quad_cycles(packs, dialect)
            else:
                _hook_up_packs_bf16(packs, local_reads)
            
            _set_pack_needed_by(packs, pack_name, i_loop, mfma_reorder, mfma_for_linear_index, timeline.num_vmfma, kernel)

def precompute_issue_times(instructions: list[ValidatorInstruction], dialect: ValidatorDialect = CDNA4_DIALECT) -> list[int]:
    """
    Returns a list where issue_times[i] represents the quad-cycle when instruction i starts issuing.

    Args:
        instructions: List of ValidatorInstruction objects in execution order.
        dialect: Architecture dialect that owns the type-switch thresholds.
            Defaults to the CDNA 4 dialect, whose values equal the historical
            module-level constants byte-for-byte.
    """
    mfma_free_at = 0
    current_issue = 0
    last_mfma_class: Optional[type] = None
    last_mfma_issue = -1

    issue_times = []
    for instruction in instructions:
        if isinstance(instruction, MFMA):
            # MFMAs must wait for previous MFMA to finish
            current_issue = max(current_issue, mfma_free_at)

            # MFMA type switch penalty
            current_mfma_class = type(instruction)
            if last_mfma_class and current_mfma_class != last_mfma_class:
                gap = current_issue - last_mfma_issue
                threshold = dialect.timing.type_switch_threshold_from_4x4 \
                            if last_mfma_class is MFMAPack \
                            else dialect.timing.type_switch_threshold_from_standard
                if gap < threshold:
                    current_issue += 1

            # Matrix-instruction "finish" latency is dialect-dependent.
            # CDNA 4 uses ISA section 7.6 quad-cycles (3 standard, 1 for
            # 4x4 MFMA). RDNA 3.5 uses the LLVM GFX11SpeedModel binding
            # (Write32Bit=5 cycles = 2 quad-cycles; no 4x4 WMMA variant).
            finish_cycles = (
                dialect.timing.mfma_4x4_finish
                if current_mfma_class is MFMAPack
                else dialect.timing.standard_mfma_finish
            )
            mfma_free_at = current_issue + 1 + finish_cycles  # 1 to issue + finish_cycles to complete

            last_mfma_issue = current_issue
            last_mfma_class = current_mfma_class

        issue_times.append(current_issue)
        current_issue = current_issue + instruction.min_issue_quad_cycles()

    return issue_times

def estimate_quad_cycles_precomputed(i_start: int, i_end: int, issue_times: list[int]) -> int:
    """
    Calculates the number of quad-cycles between when the instruction at i_start HAS BEEN issued
    and when the instruction at i_end STARTS being issued.
    
    issue_times[i_end] is when i_end starts issuing
    issue_times[i_start] is when i_start starts issuing
    After i_start finishes issuing (1 cycle later), we're at issue_times[i_start] + 1
    
    Args:
        i_start: Index of the starting instruction (already issued).
        i_end: Index of the ending instruction (about to start issuing).
        issue_times: Pre-computed list of issue times from precompute_issue_times.
    
    Returns:
        Number of quad-cycles between the two instructions.
    """
    return issue_times[i_end] - issue_times[i_start] - 1

@applies_only_once
def estimate_quad_cycles(timeline: Timeline, kernel: 'Solution', dialect: ValidatorDialect = CDNA4_DIALECT) -> int:
    """
    Perform a rough estimate on the number of quad-cycles that pass between when an instruction is issued and when its result is used.
    Needed to ensure the restrictions laid out in section 7.6 of the CDNA 4 ISA are met. Failing to meet these restrictions will result in deterministic errors.
    
    E.g. for the 4x4 MFMA TF32 route the 6th and 7th pack instructions map to:
    v_mfma_f32_4x4x4_16b_bf16 v[0:3], ..., ..., ...
    v_cvt_pk_bf16_f32 v[3], v[2], v[3]

    As listed above, the sequence of instructions is incorrect since (they reference the same VGPRs and) there must be a minimum of 5 quad-cycles between when v_mfma_f32_4x4x4_16b_bf16 has been issued and when v_cvt_pk_bf16_f32 starts issuing. As written there is a 0 quad-cycle gap (the v_cvt issues and completes in parallel with the v_mfma completing.) One way to write a correct sequency would be:
    v_mfma_f32_4x4x4_16b_bf16 v[0:3], ..., ..., ...
    s_nop 4
    v_cvt_pk_bf16_f32 v[3], v[2], v[3]

    Only operates on instructions which have a set needed_by field and a set min_quad_cycles_before_result_used field.

    All instructions take 1 quad-cycle to issue minimum.
    Swaits will stall everything else for 1 + wait_state number of quad-cycles.
    SWait is assumed to be only 1 quad-cycle, have no easy way to determine stalls.
    SBarrier is assumed to be only 1 quad-cycle, have no easy way to determine stalls.
    MFMAs take a different number of quad-cycles to finish. Currently assumed that it's 4 quad-cycles (1 issue + 3 finish).
    Packs take a different number of quad-cycles to finish (since some are actually MFMAs).
        - Specifically the 5th and 6th pack for 4x4MFMA TF32 approximation, which will take 2 quad-cycles (1 issue + 1 finish).

    During the finish cycles of an MFMA we can issue other instructions.
    E.g.: MFMA, SNop(2)
    There will have an execution time of 4 quad-cycles.
    The SNop(2) which takes 3 quad-cycles (1 issue + 2 finish) will be executed in parallel with the MFMA finishing and fit entirely behind the 3 cycles the mfma takes to finish.
    """
    if not kernel.get("UseF32XEmulation", False):
        # Only F32 emulation issues instructions (Packs) which need estimation of quad-cycles for correctness.
        return

    if not kernel.get("UseDirect32XEmulation", False):
        raise ValueError("UseDirect32XEmulation is False, case not supported.")

    # Build helper lookup
    index_for_inst_id = {id(inst): i for i, inst in enumerate(timeline.combined_timeline)}

    # Precompute issue times using dialect-driven type-switch thresholds
    issue_times = precompute_issue_times(timeline.combined_timeline, dialect)
        
    # Estimate number of quad-cycles between being issued and result being used
    for i_instruction, instruction in enumerate(timeline.combined_timeline):
        if not isinstance(instruction, TimedPack) or instruction.min_quad_cycles_before_result_used == 0:
            continue

        needed_by = instruction.needed_by
        if needed_by is None:
            continue
        if not isinstance(needed_by, ValidatorInstruction):
            continue
        if needed_by.issued_at == POSITION_INF:
            continue

        i_needed_by = index_for_inst_id.get(id(needed_by))
        estimate = estimate_quad_cycles_precomputed(i_instruction, i_needed_by, issue_times)
        instruction.estimated_quad_cycles_before_result_used = estimate

def validate_timeline(timeline: Timeline) -> Optional[str]:
    """
    Validate the timeline by calling the validate method of each instruction.
    
    Args:
        timeline: The Timeline object to validate.
    
    Returns:
        Error message if validation fails, None if validation passes.
    """
    for loop in timeline.loops:
        for instruction in timeline._timelines[loop]:
            message = instruction.validate()
            if message is not None:
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


def _transform_index_with_force_unroll_sub_iter(
    linear_index: int,
    is_lr0: bool,
    is_lra: bool,
    n_tiles_a: int,
    n_tiles_b: int,
    use_f32x_emulation: bool,
    mfma_reorder: list[int],
    num_vmfma: int,
    dialect: ValidatorDialect = CDNA4_DIALECT,
) -> int:
    """
    Convert column-major linear index into needed_by mfma index when ForceUnrollSubIter is enabled.
    
    LR data is consumed by multiple MFMAs (one for each tile in the opposite dimension).
    With MFMA reordering, we find the earliest consumer.
    """
    mfmas_per_tile = MFMAS_PER_TILE_TF32 if use_f32x_emulation else MFMAS_PER_TILE_BF16
    
    # Determine the tile coordinate for this LR
    # For LRA: linear_index is the A tile index
    # For LRB: linear_index is n_tiles_a * B tile index, so extract B tile
    if is_lra:
        a_tile = linear_index
        if is_lr0:
            a_tile += n_tiles_a // 2  # Second half of A tiles
    else:
        b_tile = linear_index // n_tiles_a
        if is_lr0:
            b_tile += n_tiles_b // 2  # Second half of B tiles
    
    def compute_consumer_mfma_index(a: int, b: int) -> int:
        """Compute MFMA index for tile (a, b) after ForceUnrollSubIter permutation."""
        # Column-major tile index
        col_major_idx = a + b * n_tiles_a
        # Apply ForceUnrollSubIter permutation
        permuted = index_for_force_unroll_sub_iter(col_major_idx, n_tiles_a, n_tiles_b)
        # Convert to MFMA index (multiply by 3 for TF32)
        return permuted * mfmas_per_tile
    
    if mfma_reorder:
        # Find earliest consumer across all tiles in the opposite dimension.
        # mfma_reorder[new_pos] = original_pos, so we need the inverse to find execution position.
        inverse = invert_mfma_reorder(mfma_reorder)
        if is_lra:
            # LRA's A tile is consumed by MFMAs at (a_tile, b) for all b tiles
            needed_by = min(
                inverse[compute_consumer_mfma_index(a_tile, b)]
                for b in range(n_tiles_b)
            )
        else:
            # LRB's B tile is consumed by MFMAs at (a, b_tile) for all a tiles
            needed_by = min(
                inverse[compute_consumer_mfma_index(a, b_tile)]
                for a in range(n_tiles_a)
            )
    else:
        # Without reorder, the first consumer (in permuted order) is always earliest
        if is_lra:
            needed_by = compute_consumer_mfma_index(a_tile, 0)
        else:
            needed_by = compute_consumer_mfma_index(0, b_tile)
    
    if not is_lr0:  # LR1/LR3 reads data for next iteration.
        # Force-unroll-sub-iter schedules are CDNA 4 only in the current
        # CustomSchedule.py corpus, so the LR1 offset follows the CDNA 4
        # "first half of NEXT iteration" convention. The dialect-supplied
        # offset (``lr1_consumer_half_offset`` * num_vmfma//2) is applied
        # in place of the old hard-coded ``+ num_vmfma`` to keep parity
        # with the non-force-unroll path.
        needed_by += dialect.lr1_consumer_half_offset * (num_vmfma // 2)

    return needed_by


def _transform_index_standard(
    linear_index: int,
    is_lr0: bool,
    is_lra: bool,
    n_tiles_a: int,
    n_tiles_b: int,
    use_f32x_emulation: bool,
    mfma_reorder: list[int],
    num_vmfma: int,
    dialect: ValidatorDialect = CDNA4_DIALECT,
) -> int:
    """
    Convert column-major linear index into needed_by mfma index when ForceUnrollSubIter is disabled.

    LR data is consumed by multiple MFMAs (one for each tile in the opposite dimension).
    With MFMA reordering, we find the earliest consumer.

    The per-LR consumer-phase offset (which "half" of the schedule the LR
    feeds) is taken from ``dialect.lr0_consumer_half_offset`` /
    ``lr1_consumer_half_offset`` so CDNA 4 and RDNA 3.5 WMMA can coexist.
    """
    mfmas_per_tile = MFMAS_PER_TILE_TF32 if use_f32x_emulation else MFMAS_PER_TILE_BF16

    needed_by = linear_index * mfmas_per_tile

    half_step = num_vmfma // 2
    if is_lr0:
        needed_by += dialect.lr0_consumer_half_offset * half_step

    if mfma_reorder:
        inverse = invert_mfma_reorder(mfma_reorder)
        if is_lra:
            needed_by = min(
                inverse[needed_by + b * n_tiles_a * mfmas_per_tile]
                for b in range(n_tiles_b)
            )
        else:
            needed_by = min(
                inverse[needed_by + a * mfmas_per_tile]
                for a in range(n_tiles_a)
            )

    if not is_lr0:
        needed_by += dialect.lr1_consumer_half_offset * half_step

    return needed_by


def lr_needed_by_mfma(
    local_read_name: str,
    lr_idx: int,
    num_vmfma: int,
    mfma_reorder: list[int],
    n_tiles_a: int,
    n_tiles_b: int,
    n_local_reads_a: int,
    n_local_reads_b: int,
    force_unroll_sub_iter: bool,
    use_f32x_emulation: bool,
    dialect: ValidatorDialect = CDNA4_DIALECT,
    ) -> int:
    """
    Helper function to calculate the index of the MFMA at which the given LRA/LRB will be needed by.

    Args:
        local_read_name: The name of the local read to calculate the needed_by index for.
        lr_idx: The index of the LRA/LRB in the list of LRAs/LRBs for the given code path.
        num_vmfma: The number of MFMA indices.
        mfma_reorder: The reordering mapping for MFMA indices.
        n_tiles_a: The number of tiles in the A dimension.
        n_tiles_b: The number of tiles in the B dimension.
        n_local_reads_a: The number of local reads in the A dimension.
        n_local_reads_b: The number of local reads in the B dimension.
        force_unroll_sub_iter: Whether to force unroll the sub-iter.
        use_f32x_emulation: Whether TF32 emulation is enabled (3 MFMAs per tile).

    Returns:
        The index of the MFMA at which the given LRA/LRB will be needed by.
    """
    # How many MFMA worth of data is loaded by each LRA/LRB
    n_tiles_per_lra = n_tiles_a / n_local_reads_a
    n_tiles_per_lrb = n_tiles_b / n_local_reads_b

    mfma_per_tile = 3 if use_f32x_emulation else 1
    single_sub_iter = num_vmfma == n_tiles_a * n_tiles_b * mfma_per_tile
    if force_unroll_sub_iter or single_sub_iter:
        # Without the unroll, the LRs are for half of the vmfmas.
        # But the number of vmfmas == 2 * n_tiles_a * n_tiles_b.
        # So each LR loads n_tiles tiles.
        # For force_unroll_sub_iter (and single-sub-iter schedules), there are only
        # n_tiles_a * n_tiles_b vmfmas. So each LR only loads half as many tiles.
        n_tiles_per_lra /= 2
        n_tiles_per_lrb /= 2

    # NOTE: This is based on the current bahaviour where we iterate through MFMAs in column-major order (A faster than B).
    def index_lra_needed_by_mfma(lra_idx: int) -> int:
        return int(lra_idx * n_tiles_per_lra)
    def index_lrb_needed_by_mfma(lrb_idx: int) -> int:
        return n_tiles_a * int(lrb_idx * n_tiles_per_lrb)

    # Calculate base tile index in column-major order
    is_lra = local_read_name.startswith("LRA")
    if is_lra:
        linear_index = index_lra_needed_by_mfma(lr_idx)
    else:
        linear_index = index_lrb_needed_by_mfma(lr_idx)
    
    # Apply transformations based on scheduling mode
    is_lr0 = local_read_name == "LRA0" or local_read_name == "LRB0"
    
    transform_function = _transform_index_standard
    if force_unroll_sub_iter:
        transform_function = _transform_index_with_force_unroll_sub_iter        
    needed_by = transform_function(
        linear_index, is_lr0, is_lra, n_tiles_a, n_tiles_b,
        use_f32x_emulation, mfma_reorder, num_vmfma, dialect
    )
    
    return needed_by


@dataclass
class GRIncData:
    """
    Data structure representing GRInc-related information.
    """
    name: list[int]
    intervals: list[tuple[int, int]]
    insts: list[int]

def verify_scc_overlap(scheduleInfo, context: dict, code_path: int) -> tuple[bool, str]:
    """SCC data-flow integrity check for a single code path.

    Guarantees that no unrelated SCC-writing scalar op is scheduled between
    the producer (``s_add_u32`` / ``s_cmp_eq_u32``) and consumer
    (``s_addc_u32`` / ``s_cselect_b32``) of each GRInc cluster. Otherwise
    the consumer would observe the wrong SCC value and compute a corrupt
    address.

    Interaction with the dialect:

    * ``dialect.scc_cluster.interval_sizes_shadow_limit`` /
      ``interval_sizes_no_shadow_limit`` supply the cluster shape (SALU
      op counts per cluster). CDNA 4 and RDNA 3.5 both use a standard
      64-bit buffer-address increment with identical SALU mnemonics, so
      the interval shapes are also identical.
    * ``dialect.scc_cluster.check_gr_m0_updates_when_dtl`` selects whether
      GRA/GRB streams are also inspected. CDNA 4 DTL=1 embeds an
      ``s_mov_b32 m0, ...`` + m0-pointer update per GR, so the pass must
      confirm that no SCC writer lands between those m0 writes and the
      GR buffer-load. RDNA 3.5 DTL=0 has no such m0 update, so the GR
      streams are skipped entirely.

    Semantic note on RDNA 3.5: per ISA section 5.6 (lines 2171-2172),
    ``S_NOP`` is NOT required between dependent scalar ops on RDNA 3.5
    for correctness -- so this pass is a data-flow integrity check on
    that architecture, not a hardware-hazard check. The CDNA 4 rationale
    (hardware hazard that requires an ``s_nop 1``) is subsumed.

    Historical shapes (preserved for CDNA 4 via ``CDNA4_DIALECT``):

    * Shadow limit (``Use64bShadowLimit=1``):
        - s_cmp_eq_u32, s_cselect_b32, s_cselect_b32 (3)
        - s_add_u32,    s_addc_u32                   (2)
        - s_sub_u32,    s_subb_u32                   (2)
        - s_cmp_eq_u32, s_cselect_b32                (2)
    * No shadow limit:
        - s_cmp_eq_u32, s_cselect_b32, s_cselect_b32 (3)
        - s_add_u32,    s_addc_u32                   (2)
        - s_sub_u32                                  (2)

    This function checks no unrelated scalar op writes SCC inside those
    intervals.
    """
    kernel = context["kernel"]
    DTL = kernel["DirectToLds"]
    ShadowLimit = kernel["Use64bShadowLimit"]

    # Dialect-driven cluster shape. CDNA4_DIALECT mirrors the historical
    # `[3,2,2,2]` / `[3,2,1]` values exactly; RDNA dialects are free to
    # specify different interval templates.
    dialect = context.get("dialect", CDNA4_DIALECT)
    scc = dialect.scc_cluster
    intervalSize = list(scc.interval_sizes_shadow_limit) if ShadowLimit \
                   else list(scc.interval_sizes_no_shadow_limit)
    numElements = sum(intervalSize)

    # Gets intervals from GRInc indices based on the above `intervalSize` value
    def getIntervals(indices):
        output = []
        current_start = 0
        for size in intervalSize:
            current_end = current_start + size
            min_val = indices[current_start]
            max_val = indices[current_end - 1]
            output.append([min_val, max_val])
            current_start = current_end
        return output

    # Checks value is in [interval[0],interval[1]].
    # if lhsGt : ]interval[0],interval[1]] else  [interval[0],interval[1][
    def inInterval(value: int, interval: list[int], lhsGt: bool):
        if lhsGt:
            return value>interval[0] and value<=interval[1]
        else:
            return value>=interval[0] and value<interval[1]

    def getDeclarationIndex(name):
        return list(scheduleInfo.optSchedule).index(name)

    GRIncNames = ["GRIncA", "GRIncB"]
    names = ["LWSA", "LWSB"]
    # We only care about GRA/B when DTL is activated (m0 usage).
    # On RDNA 3.5 DTL=0, GR is a plain VMEM op with no SCC writer, so the
    # dialect switches this off via ``check_gr_m0_updates_when_dtl=False``.
    if DTL and scc.check_gr_m0_updates_when_dtl:
        names += ["GRA", "GRB"]

    def verifyIndices(grIncData: GRIncData, name: str, indices: list[int]) -> Optional[str]:
        dclIndex = getDeclarationIndex(name)
        dclIndexGrInc = getDeclarationIndex(grIncData.name)
        for v in indices:
            for interval in grIncData.intervals:
                if inInterval(v,interval, dclIndex<dclIndexGrInc):
                    return f"{name} at index {v} can't be between {grIncData.name} {interval[0]}-{interval[1]} due to SCC usage."
        return None

    # SCC-overlap validation is only meaningful when the schedule emits GRIncA and GRIncB.
    # Some schedule variants (e.g. wave32 PGR=2 schedules that bake the pointer increment
    # into GRA/GRB themselves) do not emit separate GRInc groups; there is nothing to check
    # in that case.
    missing_grincs = [n for n in GRIncNames if n not in scheduleInfo.optSchedule]
    if missing_grincs:
        return True, ""

    GRIncs = []
    for GRIncName in GRIncNames:
        GRInc = schedule_get(GRIncName, code_path, scheduleInfo)
        assert numElements==len(GRInc), f"{GRIncName} expected size if {numElements}, given {len(GRInc)}."
        GRIncs.append(GRIncData(name = GRIncName, insts = GRInc, intervals = getIntervals(GRInc)))

    # First check GRIncA&B together
    errorMessage = verifyIndices(GRIncs[0],GRIncs[1].name, GRIncs[1].insts)
    if errorMessage:
        return False, errorMessage

    # Then, check GR and LW on all GRIncs. Skip names that are not emitted by this
    # schedule variant: for example, schedules that use PGR=2 buffer alternation
    # may not emit LWSA/LWSB swap instructions.
    for grIncData in GRIncs:
        for name in names:
            if name not in scheduleInfo.optSchedule:
                continue
            insts = schedule_get(name, code_path, scheduleInfo)
            # In case of GRA/GRB, just take m0 updates indices
            if name.startswith("GR"):
                insts = insts[0::2]
            errorMessage = verifyIndices(grIncData, name, insts)
            if errorMessage:
                return False, errorMessage

    return True, ""


@dataclass
class ValidatorPassContext:
    """Context object containing all values needed by validator passes."""
    kernel: 'Solution'
    mfma_reorder: list[int]
    swap_global_read_order: bool
    # Architecture dialect (CDNA 4 MFMA, RDNA 3.5 WMMA, ...).
    # Defaults to CDNA4 so existing callers/tests that construct
    # ValidatorPassContext by hand keep working byte-identically.
    # ``CDNA4_DIALECT`` is a frozen dataclass, so a plain default is
    # safe and avoids the per-instance lambda allocation.
    dialect: ValidatorDialect = CDNA4_DIALECT


def add_local_read_constraints(timeline: Timeline, ctx: ValidatorPassContext) -> None:
    """Add LR.needed_by and LR.guaranteed_by constraints to the provided timeline."""
    set_lr_needed_by_for_VMFMA(timeline, ctx.kernel, ctx.mfma_reorder, ctx.dialect)
    apply_swaits(timeline)
    apply_barriers(timeline)


def add_pack_constraints(timeline: Timeline, ctx: ValidatorPassContext) -> None:
    """
    Ensure that the Packs start and end at the correct indices.
    The pack commands take the data loaded into registers by LR commands and manipulate it in various ways to prepare it for the VMFMA instructions.

    There are several restrictions placed on Pack instructions:
    1. For all gemm types (tf32, bf16, etc.) the Pack instructions must be issued after the data is guaranteed to be loaded into the registers (guaranteed by SWaitCnt instructions). And they must finish before the first VMFMA that uses their results.
    2. For fp32 GEMMs, there are additional restrictions on:
        1. The ordering of the Pack instructions.
        2. The minimum number of quad-cycles that must pass between issuing certain pack instructions and when their results get used. These restrictions are defined in section 7.6 of the CDNA 4 ISA.
    """
    if ctx.kernel.get("UseF32XEmulation", False) and not ctx.kernel.get("UseDirect32XEmulation", False):
        printWarning("UseF32XEmulation is set to True but UseDirect32XEmulation is not set to True. Skipping CMS validation for packs.")
        return
    apply_swaits(timeline)
    hook_up_packs(timeline, ctx.kernel, ctx.mfma_reorder, ctx.dialect)
    estimate_quad_cycles(timeline, ctx.kernel, ctx.dialect)


def add_gr_not_too_early_constraints(timeline: Timeline, ctx: ValidatorPassContext) -> None:
    """
    Ensure that GlobalReads are not issued before the corresponding LR0s are guaranteed complete.

    Standard case (DtlPlusLdsBuf=False):
        Same-iteration dependency. GRs write to the same LDS block that LR0s read from.
        Required ordering per operand:
            last LR0 -> SWaitCnt -> SBarrier -> first GR (within same iteration)

    DtlPlusLdsBuf case (DtlPlusLdsBuf=True):
        Cross-iteration dependency. GRs write to a different LDS block than same-iteration LR0s,
        but to the same block that previous-iteration LR0s were reading from.
        Required ordering per operand:
            last LR0 (iter N-1) -> SWaitCnt -> SBarrier -> first GR (iter N)

    GRA writes (DDR->LDS) to the LDS that LRA0 reads from (LDS->VGPR).
    We conservatively assume GRA always writes everywhere that a thread in the workgroup is reading from in LRA0.
    Thus we must ensure that every thread in every wave in the workgroup has finished all of its LRA0 instructions
    before GRA is issued. Same logic applies for B. No cross-operand constraints (LRA0 vs GRB are independent).
    """
    dtl_plus_lds_buf = ctx.kernel.get("DtlPlusLdsBuf", False)

    # apply_swaits must run first so that LR0.guaranteed_by (done_idx) is set before must_start_after hookup.
    apply_swaits(timeline)
    # ``set_gr_must_start_after_from_lr0s`` encodes the CDNA 4 DTL=1 same-
    # LDS-block reuse hazard: GR writes (DDR->LDS) land in the same LDS
    # block LR0 was just reading from, so each GR must be issued after the
    # last same-iteration LR0 is guaranteed done. RDNA 3.5 DTL=0 kernels
    # perform GR as a plain ``buffer_load`` to VGPRs (LDS traffic goes
    # through separate LocalWrites), so there is no same-block LDS
    # dependency between LR0 and GR. The dialect flag
    # ``gr_must_start_after_lr0s`` selects.
    if ctx.dialect.gr_must_start_after_lr0s:
        set_gr_must_start_after_from_lr0s(timeline, ctx.swap_global_read_order, dtl_plus_lds_buf)
    # ``set_gr_must_start_after_from_grinc`` encodes the CDNA 4 DTL=1
    # convention that each loop iteration's GRInc stream must complete
    # before that same iteration's GR stream issues (because GRInc writes
    # ``m0`` and the following buffer-load reads it). RDNA 3.5 WMMA uses
    # DTL=0 with a PGR=2 schedule where iteration N's GRs read scalar
    # addresses prepared by iteration N-1's GRInc, so the same-loop
    # ordering requirement does not apply there. The dialect flag
    # ``gr_must_follow_grinc_in_same_loop`` selects.
    if ctx.dialect.gr_must_follow_grinc_in_same_loop:
        set_gr_must_start_after_from_grinc(timeline, ctx.swap_global_read_order)
    apply_must_start_after_barriers(timeline)


def add_gr_finish_before_lr_constraints(timeline: Timeline, ctx: ValidatorPassContext) -> None:
    """Add GR.needed_by and GR.barriered_at constraints.

    This pass encodes the CDNA 4 DTL=1 invariant "GR -> SWait -> SBarrier
    -> next-iter LR1". Under DTL=1 the GlobalRead writes directly to the
    LDS block that the NEXT iteration's LR1 will read, so a barrier must
    sit between the GR's SWait and that LR1. On RDNA 3.5 WMMA (DTL=0)
    the GR is a plain ``buffer_load`` whose destination is a VGPR pool;
    the LDS fill happens through a separate ``LocalWrite`` stream that
    has its own wait/barrier handshake. The ``needed_by`` chain between
    GR and next-iter LR1 therefore does not exist on this dialect, and
    the constraint is skipped.
    """
    if not ctx.dialect.gr_finish_before_lr:
        return
    apply_swaits(timeline)
    set_gr_needed_by_from_lrs(timeline, ctx.swap_global_read_order)
    apply_barriers(timeline)


TIMELINE_PASSES: dict[ValidatorPass, Callable[['Timeline', 'ValidatorPassContext'], None]] = {
    ValidatorPass.ADD_LOCAL_READ_CONSTRAINTS: add_local_read_constraints,
    ValidatorPass.ADD_PACK_CONSTRAINTS: add_pack_constraints,
    ValidatorPass.ADD_GR_NOT_TOO_EARLY_CONSTRAINTS: add_gr_not_too_early_constraints,
    ValidatorPass.ADD_GR_FINISH_BEFORE_LR_CONSTRAINTS: add_gr_finish_before_lr_constraints,
}


def index_for_force_unroll_sub_iter(original_idx: int, M: int, N: int) -> int:
    """
    Map original column-major index to index scheme used by force unroll sub-iter:
    Split the tile for each wave into 4 blocks, each indexed in column-major order.
        -------
        | 0| 2|
        |--|--|
        | 1| 3|
        -------
    Then, within each block, index within column-major order.
    For a 4x4 tile, the index changes as follows:
        |  0  4  8 12 |  ->  |  0  2  8 10 |
        |  1  5  9 13 |  ->  |  1  3  9 11 |
        |  2  6 10 14 |  ->  |  4  6 12 14 |
        |  3  7 11 15 |  ->  |  5  7 13 15 |
    
    Args:
        original_idx: The original column-major index
        M: Number of rows in the matrix
        N: Number of columns in the matrix
    
    Returns:
        The permuted index
    """
    # Block dimensions
    block_rows = M // 2
    block_cols = N // 2
    block_size = block_rows * block_cols
    
    # Convert linear index to 2D coordinates (column-major)
    row = original_idx % M
    col = original_idx // M
    
    # Determine which block (0-3) in column-major order
    block_row = row // block_rows  # 0 or 1
    block_col = col // block_cols  # 0 or 1
    block_idx = block_col * 2 + block_row  # Column-major block indexing
    
    # Position within the block
    local_row = row % block_rows
    local_col = col % block_cols
    local_idx = local_col * block_rows + local_row
    
    return block_idx * block_size + local_idx


def verify_correct_number_of_instructions(schedule_info: 'ScheduleInfo', context: dict, code_path: int) -> tuple[bool, str]:
    """
    Verify that the authored CMS stream lengths are consistent with the
    wave-specific ``idMap`` the kernel writer emits, for a single code path.

    The invariant is dialect-aware:

    * CDNA 4 (DTL=1, wave64): strict equality ``len(authored) == len(idMap)``.
      Each authored slot maps 1:1 onto exactly one physically-issued
      instruction, so strict equality is the correct check.
    * RDNA 3.5 (DTL=0, wave32): divisibility
      ``len(idMap) % len(authored) == 0 and len(authored) > 0``. Authored
      load/store slots represent bundles of N physically-issued ops where
      N is a structural function of ``(MIWaveTile, LocalReadVectorWidth,
      bpe)``.

    The choice is driven by ``dialect.stream_length_strict_equality``
    which is set to True on ``CDNA4_DIALECT`` and False on
    ``RDNA35_WMMA_DIALECT``.
    """
    if "idMap" not in context:
        # NOTE: Only skipping because the idMap is hard to construct in testing, but will always be present
        #       when actually generating the CMS kernel.
        printWarning("idMap not found in context. Skipping CMS validation for correct number of instructions.")
        return True, ""

    # Resolve the invariant: strict equality vs divisibility. Tests that
    # build a context without a dialect get CDNA-4 semantics (the
    # historical default).
    dialect = context.get("dialect")
    strict_equality = getattr(dialect, "stream_length_strict_equality", True)

    for instruction_name in schedule_info.optSchedule.keys():
        schedule = schedule_get(instruction_name, code_path, schedule_info)

        len_actual = len(schedule)
        len_expected = len(context["idMap"][instruction_name])

        if strict_equality:
            if len_actual != len_expected:
                return False, f"{instruction_name} has {len_actual} instructions, but {len_expected} instructions are required."
        else:
            if len_actual == 0:
                return False, (
                    f"{instruction_name} has 0 authored slots but "
                    f"idMap expects {len_expected} emitted instructions."
                )
            if len_expected % len_actual != 0:
                return False, (
                    f"{instruction_name} has {len_actual} authored slots, which does not evenly divide "
                    f"the {len_expected} emitted instructions expected by idMap "
                    f"(remainder {len_expected % len_actual}). RDNA 3.5 DTL=0 requires "
                    f"authored_len | idmap_len so each authored slot represents a uniform pack of "
                    f"{len_expected // max(len_actual, 1)} ops."
                )
    return True, ""


def verify_ascending_order(scheduleInfo, context: dict, code_path: int) -> tuple[bool, str]:
    """
    Ensure that all sequences of scheduleInfo.optSchedule are non-decreasing for a single code path.

    Context and example: There will be a sequence of N 'GRIncA' instructions
    for incrementing the memory address that the A macro tile is read from.
    The CMS developer has the freedom to insert these N instructions into
    'vmfmaIndices' of their choice. A vmfma_index is a sequence of instructions between
    2 consecutive mfma instructions. Example: 'GRIncA' : [[0,1,1,3]] would
    mean that the N=4 instructions to increment the pointer appear as follows:

    instruction 1    : between mfma 0 and mfma 1.
    instructions 2,3 : between mfma 1 and mfma 2.
    instruction 4    : between mfma 3 and mfma 4.

    However, there is a correctness requirement that the N vmfmaIndices for these
    instructions are non-decreasing. This rule is true for all groups of instructions,
    not just the 'GRIncA' instructions.
    """
    # TODO: Move this validation into each instructions's validation to allow for custom ordering.
    for k in scheduleInfo.optSchedule.keys():
        if k.startswith("Pack"):
            # Packs have their own validation for ordering.
            continue
        seq = schedule_get(k, code_path, scheduleInfo)
        for i in range(1, len(seq)):
            if seq[i] < seq[i - 1]:
                return (
                    False,
                    f"Non-descending-order rule failed, "
                    f"schedule key '{k}', sequence {seq}: "
                    f"value {seq[i]} at index {i} is less than "
                    f"{seq[i-1]} at index {i-1}."
                )
    return True, ""


def verify_no_lr_lw_lds_race(scheduleInfo, context: dict, code_path: int) -> tuple[bool, str]:
    """
    Detect statically-visible LR-vs-LW LDS races in the schedule.

    Why this matters
    ----------------
    The CMS framework lets a schedule freely interleave LRX (local read)
    and LW (local write) instructions across mfma slots. But the
    HARDWARE rule is that LWs of iter K and LRs of iter K must access
    the LDS in a coherent order:

      * LDSB=1 (single LDS buffer): iter K LWs OVERWRITE the same bytes
        that iter K LRs read. Therefore every LR of iter K must finish
        BEFORE any LW of iter K starts. In schedule slot terms:
            max(slot(LRX_*)) < min(slot(LWA_*) ∪ slot(LWB_*))
        (with `slot=-1` treated as iter-prologue, i.e. before slot 0.)

      * LDSB=0 (double-buffered): iter K LWs write to the OTHER LDS
        half so the same-iter race doesn't apply, but the schedule MUST
        contain LWSA/LWSB swap entries (so iter K's LW addr lands in
        the half iter K's LRs are NOT reading), and matching LRSA/LRSB
        swap entries (so iter K+1's LR addr flips to where iter K just
        wrote). A schedule that omits these swaps reads stale data.

    This pass catches the pattern that broke sol[12]/sol[17]/sol[22]
    CMS in the gfx1151 Equality library — those schedules placed LR3
    overlapping or after sub2/sub3 LWs with LDSB=1, producing NaN
    outputs that compiled successfully.

    Implementation notes
    --------------------
    * 'GRX' (global reads) are unrelated — they target G2L vgprs, not
      LDS.
    * 'LCC' (loop counter) is irrelevant.
    * SYNC entries between LR and LW *can* legalise an interleaved
      pattern (lgkmcnt(0) drains DS before the next LW), so we look
      for an explicit dscnt-draining SWaitCnt at slot in
      [max_lr_slot, min_lw_slot]. If found, we accept the schedule.
    """
    kernel = context.get("kernel", {})
    lds_buffer = kernel.get("1LDSBuffer", 1)

    sched = scheduleInfo.optSchedule

    def slots_for_prefix(prefix: str) -> list[int]:
        """Collect all schedule slots for keys starting with prefix."""
        out = []
        for key, indexLists in sched.items():
            if not key.startswith(prefix):
                continue
            seq = schedule_get(key, code_path, scheduleInfo)
            out.extend(seq)
        return out

    # Local Reads: LRA0/LRA1/.../LRB0/.../LRSA/LRSB
    lr_slots = [s for s in slots_for_prefix("LRA") + slots_for_prefix("LRB")]
    # Drop swap entries (LRSA/LRSB) — these are addr swaps, not LDS reads
    lrs_swap = sched.get("LRSA", [[]])[0] + sched.get("LRSB", [[]])[0] \
               if "LRSA" in sched or "LRSB" in sched else []

    lw_slots = sched.get("LWA", [[]])[0] + sched.get("LWB", [[]])[0]

    if not lr_slots or not lw_slots:
        return True, ""  # nothing to race

    max_lr = max(lr_slots)
    min_lw = min(lw_slots)

    if lds_buffer == 1:
        # LDSB=1: same-buffer race. All LRs must finish before any LW.
        if min_lw <= max_lr:
            # Look for a dscnt-draining sync between min_lw and the LR
            # that overlaps with it. The sync must drain DS counter
            # (dscnt=0) at a slot in [max_lr_overlap_slot, min_lw-1].
            # If present, the LRs are done before the LW fires.
            sync_slots = sched.get("SYNC", [[]])[0]
            sync_codes = scheduleInfo.syncCode
            # Find any SWaitCnt with dscnt=0 (or ==0 explicitly) at a
            # slot >= max_lr that overlaps with min_lw (i.e. fires
            # before the first LW). Ascending ordering is guaranteed
            # by VERIFY_ASCENDING_ORDER.
            for slot, code in zip(sync_slots, sync_codes):
                if not isinstance(code, SWaitCnt):
                    continue
                if code.dscnt != 0:
                    continue
                # Sync drains DS at this slot. If slot >= max_lr and
                # slot <= min_lw, it legalises the LR-then-LW order.
                # We require slot in [max_lr_overlap, min_lw] — the
                # tightest possible window.
                if slot >= max_lr and slot <= min_lw:
                    return True, ""
            return False, (
                f"LDSB=1 LR-vs-LW race: LRs span up to slot {max_lr} but "
                f"LWs begin at slot {min_lw}. With LDSB=1 the iter-K LWs "
                f"overwrite the same LDS bytes that iter-K LRs read, so "
                f"all LRs must complete before any LW. Either move the "
                f"LR3 group earlier (before any LW slot) or add an "
                f"`SWaitCnt(dscnt=0)` SYNC entry in [{max_lr}, {min_lw}]."
            )
    elif lds_buffer == 0:
        # LDSB=0: double-buffered. Schedule must contain LWS/LRS swaps.
        has_lws = bool(sched.get("LWSA", [[]])[0]) or bool(sched.get("LWSB", [[]])[0])
        has_lrs = bool(sched.get("LRSA", [[]])[0]) or bool(sched.get("LRSB", [[]])[0])
        if not has_lws or not has_lrs:
            return False, (
                f"LDSB=0 missing buffer swaps: LDSB=0 is a double-buffered "
                f"LDS layout requiring iter-end LWSA/LWSB (LW addr swap) "
                f"and pre-next-iter LRSA/LRSB (LR addr swap) entries to "
                f"flip between the two halves. Without these every iter "
                f"writes to and reads from the same half, racing with "
                f"itself. Found LWSA/LWSB={has_lws}, LRSA/LRSB={has_lrs}."
            )

    return True, ""


STRUCTURAL_CHECKS: dict[ValidatorPass, Callable] = {
    ValidatorPass.VERIFY_CORRECT_NUMBER_OF_INSTRUCTIONS: verify_correct_number_of_instructions,
    ValidatorPass.VERIFY_ASCENDING_ORDER: verify_ascending_order,
    ValidatorPass.VERIFY_SCC_OVERLAP: verify_scc_overlap,
    ValidatorPass.VERIFY_NO_LR_LW_LDS_RACE: verify_no_lr_lw_lds_race,
}


def format_kernel_string(kernel: 'Solution') -> str:
    """Format a human-readable description of the kernel's tile dimensions and transpose modes."""
    mt0 = kernel.get("MacroTile0", "?")
    mt1 = kernel.get("MacroTile1", "?")
    du = kernel.get("DepthU", "?")
    transA = "T" if kernel.get("TransA") else "N"
    transB = "T" if kernel.get("TransB") else "N"
    return f"MT0xMT1xDepthU = {mt0}x{mt1}x{du} {transA}{transB}"


def isValid(scheduleInfo: 'ScheduleInfo', context: dict) -> tuple[bool, str]:
    """
    Return True if all the validation rules pass, False otherwise.
    If validation fails, a string containing the reason is returned.

    Note 1: If True is returned, this is not proof that this schedule
    is valid. It may be a false negative.

    Note 2: if False is returned, this is not proof that the schedule
    is invalid. It may be a false positive.
    """
    kernel = context["kernel"]

    # Resolve the architecture dialect once and propagate it to every pass
    # via the context dict (structural checks) and ValidatorPassContext
    # (timeline passes). Tests may override by setting ``context["dialect"]``
    # directly; otherwise we ask ``resolve_dialect`` to pick based on kernel
    # fields.
    dialect: ValidatorDialect = context.get("dialect") or resolve_dialect(kernel)
    context["dialect"] = dialect

    # Log disabled passes once, before iterating over code paths.
    kernel_desc = format_kernel_string(kernel)

    # Check if ALL passes are disabled — single warning + early return
    all_disabled_reasons = {p: scheduleInfo.reasonForDisablingValidationPass(p) for p in ValidatorPass}
    if all(all_disabled_reasons.values()):
        reasons = set(all_disabled_reasons.values())
        reason_str = "; ".join(reasons)
        printWarning(f"All validation passes disabled on {kernel_desc}: {reason_str}")
        return True, ""

    disabled_structural = {}
    for pass_id in STRUCTURAL_CHECKS:
        if reason := scheduleInfo.reasonForDisablingValidationPass(pass_id):
            disabled_structural[pass_id] = reason
            printWarning(f"Skipping {pass_id.name} on {kernel_desc}: {reason}")

    disabled_timeline = {}
    for pass_id in TIMELINE_PASSES:
        if reason := scheduleInfo.reasonForDisablingValidationPass(pass_id):
            disabled_timeline[pass_id] = reason
            printWarning(f"Skipping {pass_id.name} on {kernel_desc}: {reason}")

    for code_path in range(scheduleInfo.numCodePaths):
        # === Structural checks (no Timeline needed) ===
        for pass_id, check in STRUCTURAL_CHECKS.items():
            if pass_id in disabled_structural:
                continue
            status, message = check(scheduleInfo, context, code_path)
            if not status:
                return False, f"Code path {code_path}: {message}"

        # === Timeline-based checks ===
        # The per-dialect Timeline subclass enforces architecture-specific
        # layout invariants (CDNA-4 DTL=1 vs RDNA 3.5 DTL=0) during
        # population, so Timeline construction is safe even if every
        # timeline pass is disabled.
        ctx = ValidatorPassContext(
            kernel=kernel,
            mfma_reorder=scheduleInfo.mfmaReorder or [],
            swap_global_read_order=kernel.get("SwapGlobalReadOrder", False),
            dialect=dialect,
        )

        timeline = create_unified_timeline(scheduleInfo, kernel, code_path, dialect)

        for pass_id, add_constraints in TIMELINE_PASSES.items():
            if pass_id in disabled_timeline:
                continue
            add_constraints(timeline, ctx)
            if error := validate_timeline(timeline):
                return False, f"Code path {code_path}: {error}"

    # All rules passed, considered valid.
    return True, ""

