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

from rocisa.instruction import SWaitCnt, SBarrier
from Tensile.Common.Utilities import printWarning


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
class MFMA(ValidatorInstruction):
    mfma_finish_cycles: ClassVar[int] = QUAD_CYCLES_STANDARD_MFMA_FINISH

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

    # The minimum number of quad-cycles that must pass before the result of this pack is used.
    # Measure from the point that this Pack is finished being issued.
    # See section 7.6 of the CDNA 4 ISA.
    # Default 0 = no timing constraint. Set by _handle_min_pack_quad_cycles for CVTPack and MFMAPack.
    min_quad_cycles_before_result_used: int = 0
    # The estimated number of quad-cycles that passed between the pack being issued and the result being used.
    # This is a lower bound estimate (does not account for most stalls and such).
    estimated_quad_cycles_before_result_used: int = 0

    def validate(self) -> Optional[str]:
        issued_at = self.issued_at.vmfma_index

        # Collapse must_start_after list to the single latest constraint
        effective_must_start_after = max(
            self.must_start_after, key=lambda c: c.done_idx()
        ) if self.must_start_after else MFMA(name="MFMA", issued_at=POSITION_NEG_INF)

        if effective_must_start_after.done_idx() < self.issued_at < self.needed_by.done_idx():
            pass  # Ordering checks passed, fall through to timing check
        elif self.issued_at < effective_must_start_after.done_idx():
            # Issued too early
            must_start_after_at = effective_must_start_after.done_idx().vmfma_index
            must_start_after_issued_at = effective_must_start_after.issued_at.vmfma_index
            return f"{self.name} @ idx={issued_at} issued too early, must be issued after idx={must_start_after_at} (because of {effective_must_start_after.name} issued @ idx={must_start_after_issued_at})."
        elif self.issued_at >= self.needed_by.issued_at:
            # Issued too late
            needed_by_at = self.needed_by.issued_at.vmfma_index
            return f"{self.name} @ idx={issued_at} issued too late, must be issued before {self.needed_by.name} @ idx={needed_by_at}."
        else:
            return f"{self.name} at index {issued_at} is not valid."

        # Timing check (only fires when min > 0, i.e. when _handle_min_pack_quad_cycles has set a constraint)
        if self.min_quad_cycles_before_result_used > 0:
            if self.estimated_quad_cycles_before_result_used < self.min_quad_cycles_before_result_used:
                needed_by_at = self.needed_by.issued_at.vmfma_index
                return f"{self.name} @ idx={issued_at} has too little gap between it and {self.needed_by.name} @ idx={needed_by_at}. Expected at least {self.min_quad_cycles_before_result_used} quad-cycles but only {self.estimated_quad_cycles_before_result_used} passed."

        return None

@dataclass
class CVTPack(Pack):
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
class SwapPack(Pack):
    """VSwapB32 instructions that transpose registers after wider local reads (VW > 1).

    Generated by transposeLRVregs() in LocalRead.py. Count per pack group: 4 * (vw - 1).
    Always appear at the beginning of a pack sequence before CVT0/MFMA/CVT1 groups.
    """
    pass

@dataclass
class MFMAPack(Pack, MFMA):
    """A v_mfma_f32_4x4x4_16b_bf16 instruction used in TF32 4x4 emulation pack groups.

    These appear at indices TF32_4X4_MFMA_START..TF32_4X4_MFMA_END within each group
    of PACK_GROUP_SIZE_TF32_4X4. They are real MFMA instructions but participate in
    the pack dependency chain (CVT0 -> MFMAPack -> CVT1).

    Inherits from both Pack and MFMA:
    - isinstance(x, Pack) is True — works with pack gathering, filtering, type hints
    - isinstance(x, MFMA) is True — captures "this IS an MFMA" semantics
    """
    # Override MFMA's finish cycles for 4x4 timing
    mfma_finish_cycles: ClassVar[int] = QUAD_CYCLES_MFMA_4X4_FINISH

    # NOTE: min_quad_cycles_before_result_used is NOT overridden here.
    # It keeps Pack's default (0) and is set by _handle_min_pack_quad_cycles
    # only when the constraint is active (when local reads exist).


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
    code_path: int
) -> 'Timeline':
    """Create a single Timeline with all instruction types."""
    available_names = set(schedule_info.optSchedule.keys())
    names_to_add = [n for n in ALL_INSTRUCTION_NAMES if n in available_names]
    return Timeline(names_to_add, code_path, schedule_info, kernel)


class Timeline:
    def __init__(self, instruction_names_to_add: list[str], code_path: int, schedule_info: 'ScheduleInfo', kernel: 'Solution'):
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
            if schedule_info.mfmaReorder:
                i_vmfma = schedule_info.mfmaReorder[i_vmfma]

            self._insert(i_vmfma, MFMA, {"name": "MFMA"}, kernel)

        # NOTE: Relative ordering of instructions must be preserved.
        #       Order dictates the order in which instructions are scheduled if they are scheduled at the same vmfmaindex.
        for name in schedule_info.optSchedule.keys():
            if name not in instruction_names_to_add:
                continue

            if name == "SYNC":
                for idx_sync, (idx_vmfma, sync) in enumerate(zip(schedule_get(name, code_path, schedule_info), schedule_info.syncCode)):
                    assert idx_vmfma >= -1, f"Code path {code_path}: SWaitCnt at index {idx_sync} is not valid. Must be >= -1."

                    if isinstance(sync, SWaitCnt):
                        self._insert(idx_vmfma, SWait, {"name": "SWaitCnt", "dscnt": sync.dscnt, "vlcnt": sync.vlcnt, "vscnt": sync.vscnt, "comment": sync.comment}, kernel)
                    elif isinstance(sync, SBarrier):
                        self._insert(idx_vmfma, Barrier, {"name": "SBarrier", "comment": sync.comment}, kernel)
                    else:
                        raise ValueError(f"Unexpected sync instruction type: {type(sync)}")
            elif name == "SNOP":
                for idx_snop, (idx_vmfma, snop) in enumerate(zip(schedule_get(name, code_path, schedule_info), schedule_info.snopCode)):
                    assert idx_vmfma >= -1, f"Code path {code_path}: SNop at index {idx_snop} is not valid. Must be >= -1."
                    # The waitState is stored as the first parameter in the rocisa SNop instruction
                    wait_state = snop.getParams()[0]
                    self._insert(idx_vmfma, SNop, {"name": "SNop", "wait_state": wait_state}, kernel)
            elif name.startswith("LRA") or name.startswith("LRB"):
                for idx_LR, idx_vmfma in enumerate(schedule_get(name, code_path, schedule_info)):
                    assert idx_vmfma >= -1, f"Code path {code_path}: LocalRead {name} at index {idx_LR} is not valid. Must be >= -1."

                    # TODO: For ForceUnrollSubIter, need to account for register reuse and the fact that the LR0/LR1/LR3s must start after a certain point in the iteration.
                    self._insert(idx_vmfma, LocalRead, {"name": name, "issue_index": idx_LR}, kernel)
            elif name.startswith("GRInc"):
                grincs = schedule_get(name, code_path, schedule_info)
                for idx_grinc, idx_vmfma in enumerate(grincs):
                    assert idx_vmfma >= -1, f"Code path {code_path}: GRInc {name} at index {idx_grinc} is not valid. Must be >= -1."
                    self._insert(idx_vmfma, GRInc, {"name": name}, kernel)
            elif name.startswith("GRA") or name.startswith("GRB"):
                global_reads = schedule_get(name, code_path, schedule_info)
                assert len(global_reads) % 2 == 0, f"Code path {code_path}: {name} has an odd number of indices. Must be even if DirectToLds is True."

                for idx_GR, idx_vmfma in enumerate(global_reads):
                    assert idx_vmfma >= -1, f"Code path {code_path}: GlobalRead {name} at index {idx_GR} is not valid. Must be >= -1."

                    # If using DirectToLds, only every other index (starting at index=1) is an actual GR, the others are increments to a pointer.
                    if idx_GR % 2 == 0:
                        continue

                    self._insert(idx_vmfma, GlobalRead, {"name": name, "swap_global_read_order": swap_global_read_order}, kernel)
            elif name.startswith("Pack"):
                packs = schedule_get(name, code_path, schedule_info)
                n_swaps = _compute_swap_pack_count(kernel, name) if is_4x4mfma_tf32 else 0

                for idx_pack, idx_vmfma in enumerate(packs):
                    assert idx_vmfma >= -1, f"Code path {code_path}: Pack {name} at index {idx_pack} is not valid. Must be >= -1."
                    if is_4x4mfma_tf32:
                        if idx_pack < n_swaps:
                            self._insert(idx_vmfma, SwapPack, {"name": name, "issue_index": idx_pack, "group_index": None}, kernel)
                        else:
                            adjusted_idx = idx_pack - n_swaps
                            # Construction-time constants: PACK_GROUP_SIZE_TF32_4X4, TF32_4X4_MFMA_START/END
                            idx_in_group = adjusted_idx % PACK_GROUP_SIZE_TF32_4X4
                            group_idx = adjusted_idx // PACK_GROUP_SIZE_TF32_4X4
                            if TF32_4X4_MFMA_START <= idx_in_group < TF32_4X4_MFMA_END:
                                self._insert(idx_vmfma, MFMAPack, {"name": name, "issue_index": idx_pack, "group_index": group_idx}, kernel)
                            else:
                                self._insert(idx_vmfma, CVTPack, {"name": name, "issue_index": idx_pack, "group_index": group_idx}, kernel)
                    elif is_tf32_emulation:
                        # Construction-time constants: PACK_GROUP_SIZE_TF32, TF32_MIDDLE_16_START/END
                        idx_in_group = idx_pack % PACK_GROUP_SIZE_TF32
                        group_idx = idx_pack // PACK_GROUP_SIZE_TF32
                        if TF32_MIDDLE_16_START <= idx_in_group < TF32_MIDDLE_16_END:
                            self._insert(idx_vmfma, MiddlePack, {"name": name, "issue_index": idx_pack, "group_index": group_idx}, kernel)
                        else:
                            self._insert(idx_vmfma, CVTPack, {"name": name, "issue_index": idx_pack, "group_index": group_idx}, kernel)
                    else:
                        self._insert(idx_vmfma, Pack, {"name": name, "issue_index": idx_pack}, kernel)
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
            kernel:       The kernel configuration dict.
        """
        for loop in self.loops:
            if self._should_add(cls, kwargs.get("name", ""), loop, kernel):
                loop_index = self.loops.index(loop)
                sub_index = len(self._instructions_at_index[loop][vmfma_index + 1])
                kwargs["issued_at"] = SchedulePosition(loop_index=loop_index, vmfma_index=vmfma_index, sub_index=sub_index)

                _instruction = cls(**kwargs)

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
def set_lr_needed_by_for_VMFMA(timeline: Timeline, kernel: 'Solution', mfma_reorder: list[int]) -> None:
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
                    use_f32x_emulation=kernel.get("UseF32XEmulation", False))
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
       

def _handle_min_pack_quad_cycles(packs: list[Pack]) -> None:
    """
    Set the min_quad_cycles_before_result_used field for Pack instructions that need timing constraints.
    This is used to enforce timing constraints for TF32 emulation modes.
    Only CVTPack and MFMAPack get non-zero values;
    MiddlePack, SwapPack, and plain Pack keep the default of 0.

    Args:
        packs: List of Pack instructions to set minimum quad-cycles for.
    """
    for pack in packs:
        if isinstance(pack, MFMAPack):
            # 4x4 MFMAs need 5 quad-cycles before CVT1 can use result
            pack.min_quad_cycles_before_result_used = QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1
        elif isinstance(pack, CVTPack):
            # CVT packs need 2 quad-cycles before MFMAs can use their results
            pack.min_quad_cycles_before_result_used = QUAD_CYCLES_CVT_BEFORE_MFMA
        # All other packs have no timing constraints

def _compute_swap_pack_count(kernel: 'Solution', pack_name: str) -> int:
    """
    Return number of swap packs for this side. Formula: 4 * (vw - 1).

    SwapPacks (VSwapB32) are only emitted by transposeLRVregs when the
    operand needs LDS transpose (TLUA for A, TLUB for B). Without
    transpose, wider local reads don't need register swapping regardless
    of VectorWidth.
    """
    is_a = pack_name.startswith("PackA")
    tlu_key = "TLUA" if is_a else "TLUB"
    needs_transpose = kernel["ProblemType"][tlu_key]
    if not needs_transpose:
        return 0
    vw = kernel["VectorWidthA" if is_a else "VectorWidthB"]
    assert vw in (1, 2, 4), f"Unsupported VectorWidth {vw} for {pack_name}. Must be 1, 2, or 4."
    return 4 * (vw - 1)

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
        vgprs_per_local_read = VGPRS_PER_CONVERSION_GROUP * n_pack_groups // n_lrs
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

        # 3. Hook up the needed_by and must_start_after fields
        for pack_name, packs in packs_by_name.items():
            local_reads = _get_lrs_for_pack(timeline, use_plr_pack, pack_name, loop)
            if not local_reads:
                continue

            if is_tf32_emulation:
                if is_4x4mfma_tf32:
                    is_a = pack_name.startswith("PackA")
                    vw = kernel["VectorWidthA" if is_a else "VectorWidthB"]
                    _hook_up_packs_f32_mfma(packs, local_reads, vw)
                else:
                    _hook_up_packs_f32(packs, all_middle_16_packs, local_reads)
                _handle_min_pack_quad_cycles(packs)
            else:
                _hook_up_packs_bf16(packs, local_reads)
            
            _set_pack_needed_by(packs, pack_name, i_loop, mfma_reorder, mfma_for_linear_index, timeline.num_vmfma, kernel)

def precompute_issue_times(instructions: list[ValidatorInstruction]) -> list[int]:
    """
    Returns a list where issue_times[i] represents the quad-cycle when instruction i starts issuing.

    Args:
        instructions: List of ValidatorInstruction objects in execution order.
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
                threshold = MFMA_TYPE_SWITCH_THRESHOLD_FROM_4X4 if last_mfma_class is MFMAPack else MFMA_TYPE_SWITCH_THRESHOLD_FROM_STANDARD
                if gap < threshold:
                    current_issue += 1

            mfma_free_at = current_issue + 1 + instruction.mfma_finish_cycles  # 1 to issue + finish_cycles to complete

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
def estimate_quad_cycles(timeline: Timeline, kernel: 'Solution') -> int:
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

    # Precompute issue times
    issue_times = precompute_issue_times(timeline.combined_timeline)
        
    # Estimate number of quad-cycles between being issued and result being used
    for i_instruction, instruction in enumerate(timeline.combined_timeline):
        if not isinstance(instruction, Pack) or instruction.min_quad_cycles_before_result_used == 0:
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
        needed_by += num_vmfma
    
    return needed_by


def _transform_index_standard(
    linear_index: int,
    is_lr0: bool,
    is_lra: bool,
    n_tiles_a: int,
    n_tiles_b: int,
    use_f32x_emulation: bool,
    mfma_reorder: list[int],
    num_vmfma: int
) -> int:
    """
    Convert column-major linear index into needed_by mfma index when ForceUnrollSubIter is disabled.
    
    LR data is consumed by multiple MFMAs (one for each tile in the opposite dimension).
    With MFMA reordering, we find the earliest consumer.
    """
    mfmas_per_tile = MFMAS_PER_TILE_TF32 if use_f32x_emulation else MFMAS_PER_TILE_BF16
    
    # Convert linear index to MFMA base index
    needed_by = linear_index * mfmas_per_tile
    
    if is_lr0:  # LR0 reads data for 2nd half of this iteration (when present)
        needed_by += num_vmfma // 2
    
    if mfma_reorder:
        # With reordering, we need to find the earliest consumer across all tiles in the opposite dimension.
        # LR data is used by multiple MFMAs - one for each tile in the opposite dimension.
        # mfma_reorder[new_pos] = original_pos, so we need the inverse to find execution position.
        inverse = invert_mfma_reorder(mfma_reorder)
        if is_lra:
            # LRA's A tile is consumed by MFMAs at (a, b) for all b tiles
            # In column-major layout: index = base + b * n_tiles_a * mfmas_per_tile
            needed_by = min(
                inverse[needed_by + b * n_tiles_a * mfmas_per_tile]
                for b in range(n_tiles_b)
            )
        else:
            # LRB's B tile is consumed by MFMAs at (a, b) for all a tiles
            # In column-major layout: index = base + a * mfmas_per_tile
            needed_by = min(
                inverse[needed_by + a * mfmas_per_tile]
                for a in range(n_tiles_a)
            )
    
    if not is_lr0:  # LR1/LR3 reads data for first half of next iteration
        needed_by += num_vmfma
    
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
        use_f32x_emulation, mfma_reorder, num_vmfma
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
    """
    Ensure we don't overlap scalar instructions modifying SCC for a single code path.
    This can happen:
        - between GRIncA and GRIncB
        - between GRInc and GR when DLT is activated
        - between GRInc and LWS

    By default, GRInc instructions can be split into 3 distinct intervals where we shouldn't touch SCC
      - s_cmp_eq_u32, s_cselect_b32,s_cselect_b32 (3)
      - s_add_u32, s_addc_u32 (2)
      - s_sub_u32, s_subb_u32 (2)
      - s_cmp_eq_u32, s_cselect_b32 (2)
    With ShadowLimit disabled (`Use64bShadowLimit`):
      - s_cmp_eq_u32, s_cselect_b32,s_cselect_b32 (3)
      - s_add_u32, s_addc_u32 (2)
      - s_sub_u32  (2)

        This function checks no other scalar instructions is inside the above intervals.
    """
    kernel = context["kernel"]
    DTL = kernel["DirectToLds"]
    ShadowLimit = kernel["Use64bShadowLimit"]

    intervalSize = [3,2,2,2] if ShadowLimit else [3,2,1] # Values explained above
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
    # We only care about GRA/B when DTL is activated (m0 usage)
    if DTL:
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


def add_local_read_constraints(timeline: Timeline, ctx: ValidatorPassContext) -> None:
    """Add LR.needed_by and LR.guaranteed_by constraints to the provided timeline."""
    set_lr_needed_by_for_VMFMA(timeline, ctx.kernel, ctx.mfma_reorder)
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
    hook_up_packs(timeline, ctx.kernel, ctx.mfma_reorder)
    estimate_quad_cycles(timeline, ctx.kernel)


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
    set_gr_must_start_after_from_lr0s(timeline, ctx.swap_global_read_order, dtl_plus_lds_buf)
    set_gr_must_start_after_from_grinc(timeline, ctx.swap_global_read_order)
    apply_must_start_after_barriers(timeline)


def add_gr_finish_before_lr_constraints(timeline: Timeline, ctx: ValidatorPassContext) -> None:
    """Add GR.needed_by and GR.barriered_at constraints."""
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
    Verify that the number of instructions in the schedule is correct for a single code path.
    """
    if "idMap" not in context:
        # NOTE: Only skipping because the idMap is hard to construct in testing, but will always be present
        #       when actually generating the CMS kernel.
        printWarning("idMap not found in context. Skipping CMS validation for correct number of instructions.")
        return True, ""

    for instruction_name in schedule_info.optSchedule.keys():
        schedule = schedule_get(instruction_name, code_path, schedule_info)

        len_actual = len(schedule)
        len_expected = len(context["idMap"][instruction_name])
        if len_actual != len_expected:
            return False, f"{instruction_name} has {len_actual} instructions, but {len_expected} instructions are required."
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


STRUCTURAL_CHECKS: dict[ValidatorPass, Callable] = {
    ValidatorPass.VERIFY_CORRECT_NUMBER_OF_INSTRUCTIONS: verify_correct_number_of_instructions,
    ValidatorPass.VERIFY_ASCENDING_ORDER: verify_ascending_order,
    ValidatorPass.VERIFY_SCC_OVERLAP: verify_scc_overlap,
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

    # If every timeline pass is disabled, skip timeline construction entirely.
    # create_unified_timeline currently asserts CDNA-4 layout invariants (DTL=1,
    # LR suffix range) which do not hold for schedules that opted out of all
    # timeline checks (e.g. gfx1151 WMMA schedules). This lets those schedules
    # still exercise the ISA-agnostic structural passes above.
    all_timeline_disabled = len(disabled_timeline) == len(TIMELINE_PASSES)

    for code_path in range(scheduleInfo.numCodePaths):
        # === Structural checks (no Timeline needed) ===
        for pass_id, check in STRUCTURAL_CHECKS.items():
            if pass_id in disabled_structural:
                continue
            status, message = check(scheduleInfo, context, code_path)
            if not status:
                scheduleInfo.pretty_print()
                return False, f"Code path {code_path}: {message}"

        if all_timeline_disabled:
            continue

        # === Timeline-based checks ===
        ctx = ValidatorPassContext(
            kernel=kernel,
            mfma_reorder=scheduleInfo.mfmaReorder or [],
            swap_global_read_order=kernel.get("SwapGlobalReadOrder", False),
        )

        timeline = create_unified_timeline(scheduleInfo, kernel, code_path)

        for pass_id, add_constraints in TIMELINE_PASSES.items():
            if pass_id in disabled_timeline:
                continue
            add_constraints(timeline, ctx)
            if error := validate_timeline(timeline):
                scheduleInfo.pretty_print()
                return False, f"Code path {code_path}: {error}"

    # All rules passed, considered valid.
    return True, ""

