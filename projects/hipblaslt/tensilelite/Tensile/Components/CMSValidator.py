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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy
from math import floor

from rocisa.instruction import SWaitCnt, SBarrier
from Tensile.Common.Utilities import printWarning


def get_most_recent_local_reads(
    vmfmas: list[int],
    counts: list[int],
    history: list[tuple[int, list[str]]],
) -> list[dict[str, int]]:
    """
    For each waitcnt instruction located at `vmfmas[i]`, compute how many
    of the most recent `counts[i]` local reads come from each of the 4 local read
    types (LRA0, LRA1, LRB0, LRB1). This is computed by walking backwards through
    historical local read positions.

    Args:
        vmfmas:
            List of vmfma instruction indices where s_waitcnt appears.
        counts:
            Number of outstanding operations each waitcnt must account for.
        history:
            A list containing all local read events for every vmfma index where
            at least one local read occured.

    Returns:
        A list of dicts, of the form:
            [{"LRA0": int, "LRB0": int, "LRA1": int, "LRB1": int}, ...]

        where each entry corresponds to the number of most-recent local reads
        attributed to different local read types for that waitcnt.
    """

    assert len(counts) == len(
        vmfmas
    ), "Counts and vmfmas must match in length."

    results: list[dict[str, int]] = []
    for count, vmfma in zip(counts, vmfmas):
        idx = 0
        while idx + 1 < len(history) and history[idx + 1][0] <= vmfma:
            idx += 1
        result = {"LRA0": 0, "LRB0": 0, "LRA1": 0, "LRB1": 0}
        remaining = count
        while idx >= 0 and remaining > 0:
            for operand in history[idx][1][::-1]:
                result[operand] += 1
                remaining -= 1
                if remaining == 0:
                    break
            idx -= 1
        results.append(result)

    return results


def verify_global_reads_not_too_early_single_code_path(
    globalReadA: list[int],
    globalReadB: list[int],
    localReadA0: list[int],
    localReadB0: list[int],
    localReadA1: list[int],
    localReadB1: list[int],
    syncIndices: list[int],
    syncCodes: list,
    context: dict,
    positions: dict,
):
    """
    Verifies that:
      1. All local reads complete before global reads.
      2. Completion is confirmed by:
         - s_waitcnt (for wave)
         - s_barrier (for workgroup)

    Returns:
        (status: bool, message: str)
    """
    assert len(syncIndices) == len(
        syncCodes
    ), "Mismatch between number of SYNC vmfmaIndices and number of SYNC codes."

    # indices of waits:
    vmfmaIndices = []

    # number outstanding of the waits:
    counts = []

    # indices of the waits with the list of sync codes:
    indicesOfWaitsInSyncCode = []

    syncCodeIndex = 0
    for syncIndex, syncCode in zip(syncIndices, syncCodes):
        if isinstance(syncCode, SWaitCnt):
            vmfmaIndices.append(syncIndex)
            counts.append(syncCode.dscnt)
            indicesOfWaitsInSyncCode.append(syncCodeIndex)
        syncCodeIndex += 1


    # Construct, for every index where a local read occured, the sequence of
    # local reads that occured, in chronological order.
    localReads = [
        ("LRA0", localReadA0),
        ("LRB0", localReadB0),
        ("LRA1", localReadA1),
        ("LRB1", localReadB1),
    ]
    localReads.sort(key=lambda x: positions[x[0]])

    history = {}
    for symbol, values in localReads:
        for v in values:
            if v not in history:
                history[v] = []
            history[v].append(symbol)
    history = sorted(history.items(), key=lambda t: t[0])

    preceding = get_most_recent_local_reads(
        vmfmaIndices,
        counts,
        history)
    for l, g, operand in [
        (localReadA0, globalReadA, "A"),
        (localReadB0, globalReadB, "B"),
    ]:
        if len(l) == 0 or len(g) == 0:
            continue

        lastLocal = l[-1]
        firstGlobal = g[0]

        # Find the first index after the last local read that this wave
        # completes all the local reads for this operand (if one exists)
        LRX = "LRA0" if operand == "A" else "LRB0"
        GRX = "GRA" if operand == "A" else "GRB"
        local_before_sync = positions[LRX] < positions["SYNC"]
        global_before_sync = positions[GRX] < positions["SYNC"]
        closedLowerBound = lastLocal + 1 - local_before_sync
        openUpperBound = firstGlobal + 1 - global_before_sync

        outstandings = []
        waveCompleteIndex = None
        # From the start of the schedule which waitcnt is the current one?
        waitIndex = 0
        for syncIndex, precede in zip(vmfmaIndices, preceding):
            if closedLowerBound <= syncIndex and syncIndex < openUpperBound:
                count = precede[LRX]
                outstandings.append(count)
                if waveCompleteIndex is None and count == 0:
                    waveCompleteIndex = syncIndex
                    break
            waitIndex += 1

        if waveCompleteIndex is None:
            return False, (
                f"Failed to verify that all local reads for {operand} ({LRX}) are complete "
                f"before the first global read for {operand} is issued. "
                f"Last local read for {operand} issued at vmfma_index:{lastLocal}. "
                f"First global read for {operand} issued at vmfma_index:{firstGlobal}. "
                f"{len(outstandings)} waitcnt operation(s) in [{closedLowerBound}, {openUpperBound}) "
                f"provide upper bounds on the number of outstanding {LRX} operations: "
                f"{outstandings} <-- none of these is 0."
            )

        # Check that there is a sync after the wave completion index
        barrierFound = False
        syncCodeIndex = indicesOfWaitsInSyncCode[waitIndex]
        while (
            syncCodeIndex + 1 < len(syncCodes)
            and syncIndices[syncCodeIndex + 1] < openUpperBound
        ):
            if isinstance(syncCodes[syncCodeIndex + 1], SBarrier):
                barrierFound = True
                break
            syncCodeIndex += 1

        if not barrierFound:
            return False, (
                f"Failed to verify that a barrier (to sync waves) exists between completion of "
                f"local reads for {operand} and the first global read for {operand}. "
                f"Last local read of {operand} issued at vmfma_index {lastLocal}, "
                f"first global read of {operand} issued at vmfma_index {firstGlobal}, "
                f"wave completion at vmfma_index {waveCompleteIndex}. Expected a barrier "
                f"in the range [{waveCompleteIndex}, {openUpperBound})."
            )

    return True, ""


def verify_global_reads_not_too_early(scheduleInfo, context: dict, code_path: int) -> tuple[bool, str]:
    """
    We require the sequence of instructions to be of the form for a single code path:

    last LRA0 instruction
    [...]
    SWaitCnt to ensure that all LRA0s are done for this wave
    [...]
    SBarrier to ensure that all LRA0s are done for this workgroup
    [...]
    First GRA instruction

    Why?

    GRA writes (DDR->LDS) to the LDS that LRA0 reads from (LDS->VGPR).

    We don't know where in LDS GRA writes from. In theory we could determine
    where exactly each GRA instruction writes to, but currently this is too
    complicated and so we conservatively assume it always writes everywhere that
    a thread in the workgroup is reading from in LRA0. Thus we must ensure
    that every thread in every wave in the workgroup has finished all of its
    LRA0 instructions before GRA is issued.

    Exactly the same logic applies for the B operand. Note that there are no
    constraints between LRA0 and GRB, or between LRB0 and GRA, because the LDS
    used for A and B are completely separate.
    """

    # Get the relative order of the relevant operations within a vmfma index.
    positions = {
        "SYNC": -1,
        "LRA0": -1,
        "LRB0": -1,
        "GRA": -1,
        "GRB": -1,
        "LRA1": -1,
        "LRB1": -1,
    }
    index = 0
    for n in scheduleInfo.optSchedule.keys():
        positions[n] = index
        index += 1

    def get(name, codePath):
        l = scheduleInfo.optSchedule.get(name, [[]])
        if len(l) == 1:
            return l[0]
        return l[codePath]

    globalReadA = get("GRA", code_path)
    globalReadB = get("GRB", code_path)
    if context.get("kernel", {}).get("SwapGlobalReadOrder", False):
        globalReadA, globalReadB = globalReadB, globalReadA

    kernel = context.get("kernel", {})
    aDirect = kernel.get("DirectToLdsA", False) or kernel.get("DirectToLds", False)
    bDirect = kernel.get("DirectToLdsB", False) or kernel.get("DirectToLds", False)

    # The actual buffer loads correspond to the indices of the lists
    # if using DirectToLDS.
    if aDirect:
        globalReadA = globalReadA[1::2]

    if bDirect:
        globalReadB = globalReadB[1::2]

    localReadA0 = get("LRA0", code_path)
    localReadB0 = get("LRB0", code_path)
    localReadA1 = get("LRA1", code_path)
    localReadB1 = get("LRB1", code_path)
    syncIndices = get("SYNC", code_path)
    syncCodes = scheduleInfo.syncCode

    status, message = verify_global_reads_not_too_early_single_code_path(
        globalReadA,
        globalReadB,
        localReadA0,
        localReadB0,
        localReadA1,
        localReadB1,
        syncIndices,
        syncCodes,
        context,
        positions,
    )
    return status, message

class ValidatorInstruction(ABC):
    """
    Abstract class with no method just for type hinting purposes.
    """
    name: str
    issued_at: float

    @abstractmethod
    def validate(self) -> str | None:
        ...

@dataclass
class LocalRead(ValidatorInstruction):
    name: str
    num_vmfma: int
    issued_at: int | float
    needed_by: int | float = float('inf')
    guaranteed_by: int | float = float('inf')

    def validate(self) -> str | None:
        # For when local reads are not being guaranteed by a particular pass.
        if self.needed_by == float('inf'):
            return None

        # Needs to be guaranteed BEFORE the index at which it's needed since the
        # SWaitCnt is issued AFTER the vmfma.
        if self.guaranteed_by < self.needed_by:
            return None

        guaranteed_by = self.guaranteed_by
        # Modulo for LRs that finish in next iteration.
        needed_by = self.needed_by % self.num_vmfma
        issued_at = floor(self.issued_at) % self.num_vmfma
        if guaranteed_by == float('inf'):
            message = f"{self.name} at index {issued_at} is not valid. " + \
                        "There are no guarantees on when it will be done."
        else:
            if self.num_vmfma - 1 + 0.5 <= (guaranteed_by % self.num_vmfma) < self.num_vmfma:
                # Special case to handle idx=-1 which is in the range of [numVMFMA + 0.5, numVMFMA)
                guaranteed_by = -1
            else:
                guaranteed_by = floor(guaranteed_by) % self.num_vmfma
            
            context_str = ""
            if self.needed_by > self.num_vmfma:
                context_str = " (of next iteration)"

            message = f"{self.name} at index {issued_at} is not valid. " + \
                        f"Needed before index {needed_by}{context_str}, but only guaranteed at index {guaranteed_by}."
        return message

@dataclass
class GlobalRead(ValidatorInstruction):
    name: str
    num_vmfma: int
    issued_at: int | float
    swap_global_read_order: bool
    needed_by: float = float('inf')
    guaranteed_by: int | float = float('inf')
    barriered_at: list[int | float] = field(default_factory=list)

    def validate(self) -> str | None:
        if self.issued_at < self.guaranteed_by < self.needed_by:
            if any(self.guaranteed_by < barriered_at < self.needed_by for barriered_at in self.barriered_at):
                    return None

        issued_at = floor(self.issued_at) % self.num_vmfma
        needed_by = floor(self.needed_by) % self.num_vmfma

        name = self._name()

        # 1. No SWait
        if self.guaranteed_by == float('inf'):
            return f"{name} at index {issued_at} is not valid. There are no guarantees on when it will be done."

        # NOTE: Must do it after the check above to guard against infinity.
        guaranteed_by = floor(self.guaranteed_by) % self.num_vmfma

        # 2. No Barrier
        if len(self.barriered_at) == 0:
            return f"{name} at index {issued_at} is not valid. There is no SBarrier acting on it."

        # 3. Guaranteed after needed
        if self.guaranteed_by > self.needed_by:
            return f"{name} at index {issued_at} is not valid. It is guaranteed by the SWait @ idx={guaranteed_by} which is after the first corresponding LR1 @ idx={needed_by}. Order must be GR -> SWait -> SBarrier -> LR1."

        # 4. No Barrier between SWait and LR1
        if not any(self.guaranteed_by < barriered_at < self.needed_by for barriered_at in self.barriered_at):
            return f"{name} at index {issued_at} is not valid. No SBarrier between SWait @ idx={guaranteed_by} and LR1 @ idx={needed_by}. Order must be GR -> SWait -> SBarrier -> LR1."

        # TODO: Did we miss a case and will we ever end up here?
        return f"{name} at index {issued_at} is not valid. issued @ idx={issued_at}, guaranteed @ idx={guaranteed_by}, barriered @ idx={[floor(i) for i in self.barriered_at]}, needed @ idx={needed_by} is not valid."

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
    issued_at: int | float
    dscnt: int
    vlcnt: int
    vscnt: int
    comment: str
    name: str = "SWaitCnt"

    def _is_valid(self) -> bool:
        return self.dscnt >= -1 and self.vlcnt >= -1 and self.vscnt >= -1 and self.issued_at >= -1

    def validate(self) -> str | None:
        if self._is_valid():
            return None
        return f"SWait at index {floor(self.issued_at)} is invalid: dscnt={self.dscnt}, vlcnt={self.vlcnt}, vscnt={self.vscnt}, issued_at={floor(self.issued_at)}."

@dataclass
class Barrier(ValidatorInstruction):
    issued_at: int | float
    comment: str
    name: str = "SBarrier"

    def validate(self) -> str | None:
        return f"Barrier at index {floor(self.issued_at)} is not valid. Must be >= -1." if self.issued_at < -1 else None

MAIN_LOOP_PREV = "ML-1"
MAIN_LOOP = "ML"
NO_GLOBAL_LOAD_LOOP = "NGL"
NO_LOCAL_LOAD_LOOP = "NLL"

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

        # Populate the timeline with instructions
        self._populate_instructions(instruction_names_to_add, code_path, schedule_info, kernel)
        self._resolve_issued_at_indices()
        self._linearize_timeline()
    
    def _populate_instructions(self, instruction_names_to_add: list[str], code_path: int, schedule_info: 'ScheduleInfo', kernel: 'Solution') -> None:
        """
        Populates all timelines with deep copies of the instructions from schedule_info.
        """
        assert kernel["DirectToLds"], "Only DirectToLds cases are supported by validator."

        halfway_point = self.num_vmfma // 2
        swap_global_read_order = kernel["SwapGlobalReadOrder"]
        
        # NOTE: Relative ordering of instructions must be preserved.
        #       Order dictates the order in which instructions are scheduled if they are scheduled at the same vmfmaindex.
        for name in schedule_info.optSchedule.keys():
            if name not in instruction_names_to_add:
                continue

            if name == "SYNC":
                for idx_sync, (idx_vmfma, sync) in enumerate(zip(schedule_get(name, code_path, schedule_info), schedule_info.syncCode)):
                    assert idx_vmfma >= -1, f"Code path {code_path}: SWaitCnt at index {idx_sync} is not valid. Must be >= -1."
                    
                    if isinstance(sync, SWaitCnt):
                        sync_instruction = SWait(issued_at=idx_vmfma, dscnt=sync.dscnt, vlcnt=sync.vlcnt, vscnt=sync.vscnt, comment=sync.comment)
                    elif isinstance(sync, SBarrier):
                        sync_instruction = Barrier(issued_at=idx_vmfma, comment=sync.comment)
                    else:
                        raise ValueError(f"Unexpected sync instruction type: {type(sync)}")
                    
                    self._insert(idx_vmfma, sync_instruction)
            elif name.startswith("LRA") or name.startswith("LRB"):
                for idx_LR, idx_vmfma in enumerate(schedule_get(name, code_path, schedule_info)):
                    assert idx_vmfma >= -1, f"Code path {code_path}: LocalRead {name} at index {idx_LR} is not valid. Must be >= -1."

                    local_read = LocalRead(name=name, num_vmfma=self.num_vmfma, issued_at=idx_vmfma)
                    self._insert(idx_vmfma, local_read)
            elif name.startswith("GRA") or name.startswith("GRB"):
                global_reads = schedule_get(name, code_path, schedule_info)
                assert len(global_reads) % 2 == 0, f"Code path {code_path}: {name} has an odd number of indices. Must be even if DirectToLds is True."
                
                for idx_GR, idx_vmfma in enumerate(global_reads):
                    assert idx_vmfma >= -1, f"Code path {code_path}: GlobalRead {name} at index {idx_GR} is not valid. Must be >= -1."

                    # If using DirectToLds, only every other index (starting at index=1) is an actual GR, the others are increments to a pointer.
                    if idx_GR % 2 == 0:
                        continue

                    global_read = GlobalRead(name=name, num_vmfma=self.num_vmfma, issued_at=idx_vmfma, swap_global_read_order=swap_global_read_order)
                    self._insert(idx_vmfma, global_read)
            else:
                raise NotImplementedError(f"Instruction {name} not implemented")
    
    def _resolve_issued_at_indices(self) -> None:
        """
        Resolve issued_at index to a sub-index resolution.
        E.g. if issuing [GRA, SWaitCnt, SBarrier, LRA1] at index 5, the issued_at indices for each would be [5, 5.25, 5.5, 5.75].
        I.e. vmfma_index + i/len(instructions @ vmfma_index) 
        NOTE: Because idx=-1 is special and occurs AFTER the instructions scheduled at idx=numVMFMA-1 but BEFORE the VMFMA at idx=0,
              we need to adjust the index so that the instructions at idx=(numVMFMA-1) have [0, 0.5) and the instructions at idx=0 have [0.5, 1).
        """
        for loop in self.loops:
            for i_vmfma in range(-1, self.num_vmfma):
                instructions = self.get_instructions_at(i_vmfma, loop)
                for i_instruction, instruction in enumerate(instructions):
                    divisor = len(instructions)
                    if i_vmfma == -1:
                        divisor *= 2
                        instruction.issued_at += 0.5
                    if i_vmfma == self.num_vmfma - 1:
                        divisor *= 2
                    instruction.issued_at += i_instruction / divisor
    
    def _insert(self, vmfma_index: int, instruction: ValidatorInstruction) -> None:
        """
        Add an instruction to the timeline at a given VMFMA index.
        Adds it to all relevant loops.
        Internal method used during initialization - does not re-linearize.
        """
        for loop in self.loops:
            if self._should_add(instruction, loop):
                _instruction = deepcopy(instruction)

                adjust = self.num_vmfma * self.loops.index(loop)
                _instruction.issued_at += adjust

                if isinstance(_instruction, SWait):
                    if _instruction.vlcnt != -1:
                        vlcnt = max(0, _instruction.vlcnt - self.vlcnt_shift[loop])
                        _instruction.vlcnt = vlcnt
                    if _instruction.dscnt != -1 and self.nll_zero_dscnt \
                       and loop in [NO_LOCAL_LOAD_LOOP]:
                        _instruction.dscnt = 0

                self._instructions_at_index[loop][vmfma_index+1].append(_instruction)

    def _should_add(self, instruction: ValidatorInstruction, loop: str) -> bool:
        """
        Determine if an instruction should be added to a given loop.
        """
        assert loop in self.loops, f"Invalid loop: {loop}"
        if isinstance(instruction, GlobalRead):
            # No GRs issued in NGL or NLL
            return loop == MAIN_LOOP or loop == MAIN_LOOP_PREV
        elif isinstance(instruction, LocalRead):
            # Only LR0s are issued in the NLL
            if loop == NO_LOCAL_LOAD_LOOP:
                return instruction.name == "LRA0" or instruction.name == "LRB0"
            return True
        else:  # TODO: Handle Pack commands in NLL and NGL
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
            if instruction.barriered_at and barrier.issued_at >= instruction.needed_by:
                # Note: Cannot break since we can't say anything about the relationship 
                #       of `GR.needed_by` between GRs based on the order they're encountered.
                continue
            instruction.barriered_at.append(barrier.issued_at)


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


def set_lr_needed_by_for_VMFMA(timeline: Timeline, kernel: 'Solution') -> None:
    """
    Set the needed_by field of LocalReads based on the VMFMA index they are required for.
    Timeline is modified in place.
    
    For LRA0/LRB0, the data is needed at a VMFMA index offset by num_vmfma // 2 (halfway point).
    For LRA1/LRB1, the data is needed at a VMFMA index offset by num_vmfma (next iteration).
    
    Args:
        timeline: The Timeline object containing the instructions.
    """
    for i_loop, loop in enumerate(timeline.loops):
        loop_offset = timeline.num_vmfma * i_loop
        for instruction_name in timeline.get_instruction_names():
            if not instruction_name.startswith("LRA") and not instruction_name.startswith("LRB"):
                continue
            local_reads = timeline.get_instructions(instruction_name, loop)
            for lr_idx, (_, lr) in enumerate(local_reads):
                base_needed_by = lr_needed_by_mfma(lr, lr_idx, timeline, kernel)
                lr.needed_by = base_needed_by + loop_offset


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
                gr.needed_by = LR_target.issued_at


def validate_timeline(timeline: Timeline) -> str | None:
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


def lr_needed_by_mfma(local_read: LocalRead, lr_idx: int, timeline: Timeline, kernel: 'Solution') -> int:
    """
    Helper function to calculate the index of the MFMA at which the given LRA/LRB will be needed by.

    Args:
        lr_idx: The index of the LRA/LRB in the list of LRAs/LRBs for the given code path.
        offset: The offset from the halfway point of the main loop.
        schedule_info: The schedule information (ScheduleInfo object)
        kernel: The kernel (Solution object)

    Returns:
        The index of the MFMA at which the given LRA/LRB will be needed by.
    """
    offset = local_read.num_vmfma
    if "0" in local_read.name:
        offset //= 2

    n_tiles_a = kernel['MIWaveTileA']
    n_tiles_b = kernel['MIWaveTileB']
    # NOTE: This calculation will produce incorrect results if the user provided the wrong number of LRs.
    n_lr_a = len(timeline.get_instructions("LRA0", MAIN_LOOP))
    n_lr_b = len(timeline.get_instructions("LRB0", MAIN_LOOP))

    # How many MFMA worth of data is loaded by each LRA/LRB
    n_tiles_per_lra = n_tiles_a / n_lr_a
    n_tiles_per_lrb = n_tiles_b / n_lr_b

    # NOTE: This is based on the current bahaviour where we iterate through A faster than B.
    def index_lra_needed_by_mfma(lra_idx: int, offset: int) -> int:
        """
        Calculate the index of the MFMA at which the given LRA will be needed by.
        """
        return int(lra_idx * n_tiles_per_lra) + offset

    def index_lrb_needed_by_mfma(lrb_idx: int, offset: int) -> int:
        """
        Calculate the index of the MFMA at which the given LRB will be needed by.
        """
        return n_tiles_a * int(lrb_idx * n_tiles_per_lrb) + offset

    if local_read.name.startswith("LRA"):
        return index_lra_needed_by_mfma(lr_idx, offset)
    else:
        return index_lrb_needed_by_mfma(lr_idx, offset)


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
    def inInterval(value : int, interval : list[int], lhsGt : bool):
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

    def verifyIndices(grIncData : GRIncData, name : str, indices : list[int]) -> str | None:
        dclIndex = getDeclarationIndex(name)
        dclIndexGrInc = getDeclarationIndex(grIncData.name)
        for v in indices:
            for interval in grIncData.intervals:
                if inInterval(v,interval, dclIndex<dclIndexGrInc):
                    return f"{name} at index {v} can't be between {grIncData.name} {interval[0]}-{interval[1]} due to SCC usage."
        return None

    GRIncs = []
    for GRIncName in GRIncNames:
        GRInc = schedule_get(GRIncName, code_path, scheduleInfo)
        assert numElements==len(GRInc), f"{GRIncName} expected size if {numElements}, given {len(GRInc)}."
        GRIncs.append(GRIncData(name = GRIncName, insts = GRInc, intervals = getIntervals(GRInc)))

    # First check GRIncA&B together
    errorMessage = verifyIndices(GRIncs[0],GRIncs[1].name, GRIncs[1].insts)
    if errorMessage:
        return False, errorMessage

    # Then, check GR and LW on all GRIncs
    for grIncData in GRIncs:
        for name in names:
            insts = schedule_get(name, code_path, scheduleInfo)
            # In case of GRA/GRB, just take m0 updates indices
            if name.startswith("GR"):
                insts = insts[0::2]
            errorMessage = verifyIndices(grIncData, name, insts)
            if errorMessage:
                return False, errorMessage

    return True, ""


def verify_gr_inc_order(scheduleInfo, context: dict, code_path: int) -> tuple[bool, str]:
    """
    Ensure GRInc A and B are done before GR A & B for a single code path.
    When using `SwapGlobalReadOrder=True`, one should check GRIncB is done before GRA (and GRIncA before GRB)
    """
    SwapGR = context["kernel"]["SwapGlobalReadOrder"]

    def getDeclarationIndex(name):
        return list(scheduleInfo.optSchedule).index(name)

    GRIncNames = ["GRIncA", "GRIncB"]
    GRNames = ["GRA", "GRB"] if not SwapGR else ["GRB", "GRA"]

    for [grIncName, grName] in zip(GRIncNames, GRNames):
        grInc = schedule_get(grIncName, code_path, scheduleInfo)
        gr = schedule_get(grName, code_path, scheduleInfo)[1::2] # ignore m0
        grIncDclAfter = getDeclarationIndex(grIncName)>getDeclarationIndex(grName)
        # Fails if GrInc is after Gr or if same index but grInc is declared after.
        if max(grInc)>min(gr) or (grIncDclAfter and max(grInc) == min(gr)):
             return False, f"{grIncName} finishes after {grName} starts ({max(grInc)} vs {min(gr)})"

    return True, ""

def verify_grs_finish_before_lrs(schedule_info: 'ScheduleInfo', context: dict, code_path: int) -> tuple[bool, str]:
    """
    Ensure that the GlobalReads issued in the previous iteration are guaranteed to be complete before the first corresponding LR1/3 of this iteration.
    """
    relevant_names = ["GRA", "GRB", "LRA1", "LRB1", "LRA3", "LRB3", "SYNC"]
    kernel = context["kernel"]
    timeline = Timeline(relevant_names, code_path, schedule_info, kernel)
    
    # Apply standalone functions to populate timeline fields
    set_gr_needed_by_from_lrs(timeline, kernel["SwapGlobalReadOrder"])
    apply_swaits(timeline)
    apply_barriers(timeline)

    message = validate_timeline(timeline)
    if message:
        return False, message
    return True, ""


def verify_lrs_finished_before_vmfma(schedule_info: 'ScheduleInfo', context: dict, code_path: int) -> tuple[bool, str]:
    """
    Ensure that the LocalReads are guaranteed to be complete before the first VMFMA that uses their data.
    """
    if context.get("kernel", {}).get("UseF32XEmulation", False) or context.get("kernel", {}).get("ForceUnrollSubIter", False):
        message = "LR completion before vmfma validation is disabled for F32X emulation or ForceUnrollSubIter"
        printWarning(f"{message}")
        return True, message
    
    if len(schedule_info.mfmaReorder) != 0:
        printWarning("CMS Validation does not currently support mfmaReorder, cannot guarantee that LRs will be correct.")
        return True, ""

    relevant_names = ["LRA0", "LRB0", "LRA1", "LRB1", "SYNC"]
    kernel = context["kernel"]
    timeline = Timeline(relevant_names, code_path, schedule_info, kernel)

    set_lr_needed_by_for_VMFMA(timeline, kernel)
    apply_swaits(timeline)
    apply_barriers(timeline)

    message = validate_timeline(timeline)
    if message:
        return False, message
    return True, ""

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
    for k in scheduleInfo.optSchedule.keys():
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


def isValid(scheduleInfo: 'ScheduleInfo', context: dict) -> tuple[bool, str]:
    """
    Return True if all the validation rules pass, False otherwise.
    If validation fails, a string containing the reason is returned.

    Note 1: If True is returned, this is not proof that this schedule
    is valid. It may be a false negative.

    Note 2: if False is returned, this is not proof that the schedule
    is invalid. It may be a false positive.
    """
    # Case where there was an explicit request to skip validation.
    if scheduleInfo.__skipValidation__:
        mt0 = context.get("kernel", {}).get("MacroTile0", "?")
        mt1 = context.get("kernel", {}).get("MacroTile1", "?")
        du = context.get("kernel", {}).get("DepthU", "?")
        transA = context.get("kernel", {}).get("TransA")
        transB = context.get("kernel", {}).get("TransB")
        transA = "T" if transA else "N"
        transB = "T" if transB else "N"
        message = f"CMS validation explicitly disabled. Running on kernel with MT0xMT1xDepthU = {mt0}x{mt1}x{du} {transA}{transB}"
        printWarning(f"{message}")

        # All rules bypassed, considered valid.
        return True, message

    # The set of validation rules to run (code-path aware)
    rules = [
        verify_correct_number_of_instructions,
        verify_ascending_order,
        verify_lrs_finished_before_vmfma,
        verify_global_reads_not_too_early,
        verify_grs_finish_before_lrs,
        verify_scc_overlap,
        verify_gr_inc_order
    ]

    for code_path in range(scheduleInfo.numCodePaths):
        for rule in rules:
            status, message = rule(scheduleInfo, context, code_path)
            if not status:
                return False, f"Code path {code_path}: {message}"

    # All rules passed, considered valid.
    return True, ""

