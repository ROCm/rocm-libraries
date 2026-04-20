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

from dataclasses import dataclass, field, asdict
from rocisa.code import KernelBody, Label, Macro, Module, RegSet, SrdUpperValue, \
                        StructuredModule, TextBlock, ValueEndif, ValueIf, ValueElseIf, ValueSet, SignatureBase
from rocisa.container import vgpr, sgpr, SMEMModifiers, replaceHolder, EXEC,\
    VOP3PModifiers, ContinuousRegister
from rocisa.instruction import BufferLoadB128, BufferLoadB32, BufferLoadB64, \
  BufferLoadD16B16, BufferLoadD16U8, DSLoad2B32, DSLoad2B64, DSLoadB128, \
  DSLoadB32, DSLoadB64, DSLoadB64TrB16, DSLoadInstruction, DSLoadU16, \
  DSLoadU8, DSStore2B32, DSStore2B64, DSStoreB128, DSStoreB16, DSStoreB256, \
  DSStoreB32, DSStoreB64, DSStoreB8, DSStoreInstruction, FlatLoadB128, FlatLoadB32, \
  FlatLoadB64, FlatStoreB128, FlatStoreB32, FlatStoreB64, Instruction, \
  MFMAInstruction, SBarrier, SBranch, SCBranchSCC0, SCBranchSCC1, SCmpLeU32, \
  SMovB32, SMFMAInstruction, SNop, SSetPrior, SSetRegIMM32B32, SSubU32, SWaitCnt, SWaitAlu, \
  SLongBranchPositive, VFmaMixF32, VMadMixF32, VMovB32
from rocisa.instruction import SAddU32, SAddCU32, SCmpEQU32, SCSelectB32, SSubBU32
from Tensile.Common import IsaVersion
from Tensile.Common.Utilities import printWarning
from Tensile.Utilities.Decorators.Shared import CallableGuard

from copy import deepcopy
from typing import Callable, Optional, Union, Tuple
from enum import Enum, auto
import Tensile.Components.CMSValidator as cmsv
from typing import Callable
from itertools import product

# Enum to distinguish between different schedule matching outcomes
class ScheduleMatchStatus(Enum):
    FOUND = auto()                  # Schedule found and supported
    NO_MATCH = auto()               # Criteria don't match, continue searching
    UNSUPPORTED_VARIANT = auto()    # Criteria match but variant unsupported, stop searching

# Global registry for schedule functions
_SCHEDULE_REGISTRY = []

_SCHEDULE_METADATA: list["CMSKernelInfo"] = []

# Map dtype predicate functions to human-readable names
_DTYPE_PREDICATE_NAMES: dict[Callable, str] = {}

def _register_dtype_name(func: Callable, name: str) -> Callable:
    """Helper to register a dtype predicate name mapping."""
    _DTYPE_PREDICATE_NAMES[func] = name
    return func

@dataclass(frozen=True)
class CMSKernelInfo:
    """
    Metadata about registered CMS kernels 
    Contains the minimum combination of parameters needed to use the CMS kernel.
    Important Note:
    If you are adding new parameters to this list (of params use in CMS kernels), please make sure those names match Tensile names.
    These names will be used by caller/tuning codes to set correct parameter/values.
    """
    name: str
    dtype: str
    MacroTile0: int
    MacroTile1: int
    DepthU: int
    PrefetchGlobalRead: int
    PrefetchLocalRead: int
    DirectToLds: bool
    DtlPlusLdsBuf: int
    WaveSeparateGlobalReadA: int
    WaveSeparateGlobalReadB: int
    GlobalReadVectorWidthA: int
    GlobalReadVectorWidthB: int
    LocalReadVectorWidth: int
    MatrixInstruction: list[int]
    MIWaveGroup: list[int]
    LDSTrInst: bool
    TransposeLDS: int
    TransposeA: bool
    TransposeB: bool

    def matches(self, dtype: Optional[str] = None, layout: Optional[str] = None) -> bool:
        """Check if this kernel info matches the given dtype and/or layout filter.

        Args:
            dtype:  Data type filter string (e.g. "16bit", "8bit", "TF32"), or None for any.
            layout: Layout filter string (e.g. "TN", "NT", "NN", "TT"), or None for any.

        Returns:
            True if the kernel matches all provided filters.
        """
        if dtype is not None and self.dtype.lower() != dtype.lower():
            return False
        if layout is not None:
            layout = layout.upper()
            if self.TransposeA != (layout[0] == "T") or self.TransposeB != (layout[1] == "T"):
                return False
        return True



@dataclass
class SyncSchedule:
    schedule: list[tuple[int, Union[SWaitCnt, SBarrier]]] = field(default_factory=list)

    def add(self, idx: int, dscnt: int = -1, vlcnt: int = -1, vscnt: int = -1, comment: str = "", barrier: bool = False, barrier_idx: Optional[int] = None, barrier_comment: str = ""):
        """ Add a SWaitCnt (and optionally a SBarrier) to the schedule at the given index.
        Args:
            idx:             The index at which to add the SWaitCnt.
            dscnt:           The dscnt value for the SWaitCnt.
            vlcnt:           The vlcnt value for the SWaitCnt.
            vscnt:           The vscnt value for the SWaitCnt.
            comment:         An optional comment for the SWaitCnt.
            barrier:         If True, also add a SBarrier.
            barrier_idx:     The index at which to add the SBarrier. If None, uses idx.
            barrier_comment: An optional comment for the SBarrier.

        Example:
            wait.add(2, dscnt=3)                                   adds SWaitCnt at index 2 with dscnt=3
            wait.add(5, dscnt=0, sbarrier=True)                    adds SWaitCnt at index 5 with dscnt=0 and a SBarrier at the same index
            wait.add(5, dscnt=0, sbarrier=True, barrier_idx=6)     adds SWaitCnt at index 5 with dscnt=0 and a SBarrier at index 6
        """
        self.schedule.append( (idx, SWaitCnt(dscnt=dscnt, vlcnt=vlcnt, vscnt=vscnt, comment=comment)) )
        if barrier:
            barrier_idx = barrier_idx if barrier_idx is not None else idx
            self.schedule.append( (barrier_idx, SBarrier(comment=barrier_comment)) )

    def get_indicies(self):
        return [item[0] for item in self.schedule]
    def get_code(self):
        return [item[1] for item in self.schedule]

def create_range(min_val: int, num: int, max_val: int = -1, step: int = 1, repeat: int = 2) -> list[int]:
    """
    Generate a list where each value in range(min_val, min_val+num, step) is repeated 'repeat' times.
    Value is clamped to max_val
    
    Args:
        min_val: Starting value (inclusive)
        num: Number of values
        step: Step between values
        max_val: Maximum value (clamp)
        repeat: Number of times to repeat each value
    
    Example:
        create_range(100, 5,200, 1, 2) => [100, 100, 101, 101, 102, 102, 103, 103, 104, 104]
        create_range(0, 5, 10, 2, 3) => [0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8]
        create_range(0, 5, 6, 2, 3) => [0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6, 6, 6, 6]
    """
    if max_val == -1:
        max_val = min_val + step*num
    return [min(val, max_val) for val in range(min_val, min_val + step*num, step) for _ in range(repeat)]

def inflight(lst, index):
    """
    Return number of inflight loads in a given list of instructions at a specified index
    """
    return sum(val < (index) for val in lst)

def duplicate_list_items(input_list: list, repeat_count: int, step: int = 0) -> list:
    """
    Duplicate each item in input_list repeat_count times. Optionally duplicate with a step

    Example:
        duplicate_list_items([1, 2, 3], 3)    => [1,1,1, 2,2,2, 3,3,3]
        duplicate_list_items([1, 2, 3], 3, 1) => [1,2,3, 2,3,4, 3,4,5]
    """
    return [item + step * j for item in input_list for j in range(repeat_count)]

def count_items(input_list: list[int], sv: Optional[int] = None, ev: Optional[int] = None):
    """
    Count how many items in the list are between start value `sv` (inclusive) and end value `ev` (exclusive)

    Example:
        count_items([1,2,3,4,5], sv=2, ev=5) => 3 (2,3,4)
        count_items([1,2,3,4,5], sv=3)        => 3 (3,4,5)
        count_items([1,2,3,4,5], ev=4)        => 3 (1,2,3)
    """
    count = 0
    sv = sv if sv is not None else input_list[0]
    ev = ev if ev is not None else input_list[-1]
    for item in input_list:
        if sv <= item < ev:
            count += 1
    return count

def switch_A_B_schedule(optSchedule):
    # Swap A and B entries in the schedule
    # Only replace A/B if it's the last or second-last character
    swappedSchedule = dict()
    for key, value in optSchedule.items():
        # Check if A or B is in the last or second-last position
        if len(key) >= 1 and key[-1] in ('A', 'B'):
            # Last character is A or B
            new_key = key[:-1] + ('B' if key[-1] == 'A' else 'A')
        elif len(key) >= 2 and key[-2] in ('A', 'B'):
            # Second-last character is A or B
            new_key = key[:-2] + ('B' if key[-2] == 'A' else 'A') + key[-1]
        else:
            # No A or B in last or second-last position, keep unchanged
            new_key = key
        swappedSchedule[new_key] = value
    return swappedSchedule

class ScheduleInfo:
    def __init__(
        self,
        numCodePaths: int,
        numMfma: int,
        optSchedule: dict[str, list[list[int]]],
        syncCode: list[Union[SWaitCnt, SBarrier]],
        nglshift: int,
        nllshift: int,
        nllZeroDscnt: bool = False,
        mfmaReorder = [],
        snopCode: list[SNop] = [],
    ):
        self.numCodePaths = numCodePaths
        self.numMfma = numMfma
        self.optSchedule = optSchedule
        self.syncCode = syncCode
        self.nglshift = nglshift  # vmcnt shift for noglobalload loop
        self.nllshift = nllshift  # vmcnt shift for nolocalload loop
        self.nllZeroDscnt = nllZeroDscnt
        self.mfmaReorder = mfmaReorder
        self.snopCode = snopCode
        self._disabledPasses: dict[cmsv.ValidatorPass, str] = {}

    def disableValidationPass(self, pass_id: cmsv.ValidatorPass, reason: str) -> None:
        """Disable a specific validator pass for this schedule.

        Args:
            pass_id: The ValidatorPass enum member to disable.
            reason:  Mandatory explanation of why this pass is being disabled.
        """
        if not isinstance(pass_id, cmsv.ValidatorPass):
            raise TypeError(f"pass_id must be a ValidatorPass enum member, got {type(pass_id).__name__}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Reason for disabling pass must be a non-empty string")
        self._disabledPasses[pass_id] = reason

    def disableValidation(self, reason: str) -> None:
        """Disable all validator passes for this schedule."""
        for pass_id in cmsv.ValidatorPass:
            self.disableValidationPass(pass_id, reason)

    def reasonForDisablingValidationPass(self, pass_id: cmsv.ValidatorPass) -> Optional[str]:
        """Return the reason this pass was disabled, or None if it is enabled.

        Raises TypeError if pass_id is not a ValidatorPass enum member.
        """
        if not isinstance(pass_id, cmsv.ValidatorPass):
            raise TypeError(f"pass_id must be a ValidatorPass enum member, got {type(pass_id).__name__}")
        return self._disabledPasses.get(pass_id)

    def pretty_print(self):
        print("{")
        keys = list(self.optSchedule.keys())
        maxKeyLen = max(len(k) for k in keys) if keys else 0
        for i, k in enumerate(keys):
            v = self.optSchedule[k]
            comma = "," if i < len(keys) - 1 else ""
            pad = " " * (maxKeyLen - len(k))
            if len(v) == 1:
                print(f"    '{k}':{pad} [{v[0]}]{comma}")
            else:
                # Align continuation rows after the opening bracket
                bracketCol = 8 + maxKeyLen
                indent = " " * (bracketCol + 1)
                print(f"    '{k}':{pad} [")
                for j, row in enumerate(v):
                    row_comma = "," if j < len(v) - 1 else ""
                    print(f"{indent}{row}{row_comma}")
                print(f"{' ' * bracketCol}]{comma}")
        print("}")

        if snops := self.optSchedule.get('SNOP', []):
            print("---- SNOP code ----")
            for idx, code in zip(snops[0], self.snopCode):
                print(f"{idx:>2}: {str(code).strip()}")

        if syncs := self.optSchedule.get('SYNC', []):
            print("---- SYNC code ----")
            for idx, code in zip(syncs[0], self.syncCode):
                print(f"{idx:>2}: {str(code).strip()}")

def removeComments(module):
    retModule = Module()
    for i in module.flatitems():
        if type(i) != TextBlock and not isinstance(i, (SCBranchSCC1, SNop)):
            retModule.add(i)
    return retModule.flatitems()


def customMainLoopSchedule(writer, kernel, tensorParametersA, tensorParametersB, \
                      globalReadIncACode, globalReadIncBCode, \
                      LRCodeA, PackCodeA, LRCodeB, PackCodeB,\
                      LRSwapA, LRSwapB, \
                      globalReadA, globalReadB, \
                      LWSwapA, LWSwapB, \
                      mfmaCode, loopCounterCode, \
                      ):

    module = Module()

    globalReadIncACode = removeComments(globalReadIncACode)
    globalReadIncBCode = removeComments(globalReadIncBCode)
    numLoopIter = kernel["LoopIters"]
    ph = -2 # placeholder index

    if numLoopIter > 1:
        for uIdx in range(0, numLoopIter):
            LRCodeA[uIdx] = removeComments(LRCodeA[uIdx])
            LRCodeB[uIdx] = removeComments(LRCodeB[uIdx])
            PackCodeA[uIdx] = removeComments(PackCodeA[uIdx])
            PackCodeB[uIdx] = removeComments(PackCodeB[uIdx])
    else: # numIterPLR = 0 case
        numLoopIter = 2
        # Split instruction stream for numiterPLR=0
        # First half is done in subiter 1, second half is done in subiter 0
        def splitForPLR(oldModule):
            newList0 = []
            newList1 = []
            numInst = len(oldModule.flatitems())
            for ii in range(numInst):
                if ii < numInst // 2:
                    newList1.append(oldModule.flatitems()[ii])
                else:
                    newList0.append(oldModule.flatitems()[ii])
            return [newList0, newList1]

        LRCodeA = splitForPLR(LRCodeA[0])
        LRCodeB = splitForPLR(LRCodeB[0])
        PackCodeA = splitForPLR(PackCodeA[0])
        PackCodeB = splitForPLR(PackCodeB[0])


    LRSwapA = removeComments(LRSwapA)
    LRSwapB = removeComments(LRSwapB)
    globalReadA = removeComments(globalReadA)
    globalReadB = removeComments(globalReadB)
    localWriteA = removeComments(writer.codes.localWriteA)
    localWriteB = removeComments(writer.codes.localWriteB)
    LWSwapA = removeComments(LWSwapA)
    LWSwapB = removeComments(LWSwapB)
    loopCounterCode = removeComments(loopCounterCode)
    mfmaCode = removeComments(mfmaCode)

    _, opt1 = hasCustomSchedule(kernel)
    numCodePath = opt1.numCodePaths
    assert opt1.numMfma == len(mfmaCode)


    for _, indexList in opt1.optSchedule.items():
        assert len(indexList) <= opt1.numCodePaths

    if len(opt1.mfmaReorder) > 0:
        mfmaCode = [mfmaCode[x] for x in opt1.mfmaReorder]

    idMap = {
        'GRIncA' : globalReadIncACode,
        'GRIncB' : globalReadIncBCode,
        'GRA'    : globalReadA,
        'GRB'    : globalReadB,
        'LWA'    : localWriteA,
        'LWB'    : localWriteB,
        'LRSA'   : LRSwapA,
        'LRSB'   : LRSwapB,
        'LWSA'   : LWSwapA,
        'LWSB'   : LWSwapB,
        'LCC'    : loopCounterCode,
    }


    for uIdx in range(0, numLoopIter):
        idMap["LRA%u"%uIdx] = LRCodeA[uIdx]
        idMap["LRB%u"%uIdx] = LRCodeB[uIdx]
        idMap["PackA%u"%uIdx] = PackCodeA[uIdx]
        idMap["PackB%u"%uIdx] = PackCodeB[uIdx]

    idMap['SYNC'] = opt1.syncCode
    idMap['SNOP'] = opt1.snopCode

    status, message = cmsv.isValid(opt1, {'kernel' : kernel, "idMap": idMap})
    # create the case str (TN, NT, TT, or NN)
    if isTN(kernel):
        case_str = "TN"
    elif isNT(kernel):
        case_str = "NT"
    elif isTT(kernel):
        case_str = "TT"
    elif isNN(kernel):
        case_str = "NN"
    else:
        case_str = "Unknown"
    assert status is True, f"CMS validation failed for kernel {kernel['MacroTile0']}x{kernel['MacroTile1']}x{kernel['DepthU']} {case_str}: {message}"

    InstStreams = {key: [stream, idMap[key]] for key, stream in opt1.optSchedule.items()}

    macro = Macro("MAINLOOP", ["ID", "useGR=1", "usePLR=1", "useGRInc=1", "useLoop=1"])

    lastIter = numLoopIter - 1

    for miIndex in range(-1, len(mfmaCode)):
        if miIndex >= 0:
            macro.addComment0("mfmaIndex:%u"%(miIndex))
            macro.add(mfmaCode[miIndex])

        def scheduleInst(indexList, instructionList):
            ret = [None]*len(indexList)
            totalNumInst = len(instructionList)
            for i in range(len(indexList)):
                if indexList[i]: # For specific codepath
                    cc = 0
                    # Add slower, but allow reordering of instructions
                    while miIndex in indexList[i]:
                        ind = indexList[i].index(miIndex)
                        if cc == 0:
                            ret[i] = Module()
                        ret[i].add(instructionList[ind])
                        cc += 1
                        indexList[i][ind] = ph
            if ret.count(None) == len(ret):
                return [None]
            else:
                return ret

        ToSched = {k: scheduleInst(stream[0], stream[1]) for k, stream in InstStreams.items()}

        def nllvmcntHandling(inst, shift0, shift1):
            if isinstance(inst, SWaitCnt) and (inst.vlcnt != -1 or (inst.dscnt != -1 and opt1.nllZeroDscnt)):
                macro.add(ValueIf("\\useGR == 1 && \\usePLR == 1")) # in main loop
                macro.addComment0("vmcnt used in main loop")
                macro.add(inst)
                macro.add(ValueElseIf("\\useGR == 0 && \\usePLR == 1")) # in NGL
                instModified = deepcopy(inst)
                if inst.vlcnt != -1:
                    macro.addComment0("vmcnt used in ngl, applying %u shift"%shift0)
                    instModified.vlcnt = max(0, instModified.vlcnt - shift0)
                macro.add(instModified)
                macro.add(ValueElseIf("\\useGR == 0 && \\usePLR == 0")) # in NLL
                instModified = deepcopy(inst)
                if inst.vlcnt != -1:
                    macro.addComment0("vmcnt used in nll, applying %u shift"%shift1)
                    instModified.vlcnt = max(0, instModified.vlcnt - shift1)
                if (inst.dscnt != -1 and opt1.nllZeroDscnt):
                    macro.addComment0("setting dscnt = 0 for NLL")
                    instModified.dscnt = 0
                macro.add(instModified)
                macro.add(ValueEndif())
            else:
                macro.add(inst)

        def get_macro_guard(key):
            """Determine the macro guard for a given instruction key."""
            if key in ['GRIncA', 'GRIncB']:
                return "\\useGRInc == 1"
            elif key in ['GRA', 'GRB', 'LWSA', 'LWSB']:
                return "\\useGR == 1"
            elif key in ['LRA%u' % lastIter, 'LRB%u' % lastIter, 'LRSA', 'LRSB']:
                return "\\usePLR == 1"
            elif key in ['LCC']:
                return "\\useLoop == 1"
            return ""

        def emit_instructions(instModule, macroGuard: str):
            """Emit instructions from a module with optional macro guard."""
            if instModule is not None:
                for inst in instModule.flatitems():
                    if isinstance(inst, SWaitCnt):
                        nllvmcntHandling(inst, opt1.nglshift, opt1.nllshift)
                    else:
                        if macroGuard:
                            macro.add(ValueIf(macroGuard))
                        macro.add(inst)
                        if macroGuard:
                            macro.add(ValueEndif(comment="EndIf %s" % macroGuard))

        for k, ts in ToSched.items():
            macroGuard = get_macro_guard(k)

            if len(ts) == 1:
                emit_instructions(ts[0], macroGuard)
            elif len(ts) == numCodePath:
                # Multi codepath - emit inside ID conditionals
                for codepath in range(numCodePath):
                    if codepath == 0:
                        macro.add(ValueIf("\\ID == %u" % codepath))
                    else:
                        macro.add(ValueElseIf("\\ID == %u" % codepath))
                    emit_instructions(ts[codepath], macroGuard)
                macro.add(ValueEndif(comment="EndIf \\ID checks"))
            else:
                raise ValueError(f"Invalid number of instructions for {k}: {len(ts)}")
 
    module.add(macro)
    return module, numCodePath


def hasCustomSchedule(kernel):
    """
    Trampoline function that checks if a custom schedule is available.
    Iterates through registered schedule functions and returns the first match.
    """
    if not kernel["UseCustomMainLoopSchedule"]:
        return False, None
    if not kernel["EnableMatrixInstruction"]:
        return False, None
    if not kernel["ISA"] == IsaVersion(9,5,0):
        return False, None
    if isMixed(kernel):
        return False, None

    useLDSTr = kernel["LDSTrInst"]
    TLDS = kernel["TransposeLDS"]
    
    for schedule_func in _SCHEDULE_REGISTRY:
        status, schedule = schedule_func(kernel, useLDSTr, TLDS)
        if status == ScheduleMatchStatus.FOUND:
            return True, schedule
        elif status == ScheduleMatchStatus.UNSUPPORTED_VARIANT:
            # Criteria matched but variant unsupported - stop searching
            return False, None
        # status == NO_MATCH: continue to next schedule
    
    return False, None


def query_cms_kernels(dtype: Optional[str] = None, layout: Optional[str] = None) -> list[dict]:
    """Query for available CMS kernels matching the given data type and/or layout.

    This function searches the CMS kernel registry and returns the minimum
    combination of parameters needed for each matching CMS kernel.

    Args:
        dtype:  Data type filter (case-insensitive).
                Accepted values: "16bit", "8bit", "TF32", or None for all.
        layout: Layout / transpose e.g. ("TN", "NT", "NN", "TT", or None for all)
                

    Returns:
        A list of dicts, each containing the minimum parameter combination
        needed for a matching CMS kernel. Each dict includes the minimal parameters/values combinations needed for using each CMS kernel.

    """
    results = []
    for info in _SCHEDULE_METADATA:
        if info.matches(dtype=dtype, layout=layout):
            results.append(asdict(info))
    return results


def get_cms_kernel_info_objects(dtype: Optional[str] = None, layout: Optional[str] = None) -> list[CMSKernelInfo]:
    """Query for available CMS kernels and return CMSKernelInfo objects.

    Same filtering as :func:`query_cms_kernels` but returns the raw
    ``CMSKernelInfo`` dataclass instances instead of dicts.

    Args:
        dtype:  Data type filter (case-insensitive), or None for all.
        layout: Layout filter (case-insensitive), or None for all.

    Returns:
        A list of CMSKernelInfo objects matching the filters.
    """
    return [info for info in _SCHEDULE_METADATA if info.matches(dtype=dtype, layout=layout)]


def get_available_dtypes() -> set[str]:
    """Return set of all data type strings that have at least one CMS kernel."""
    return {info.dtype for info in _SCHEDULE_METADATA}


def get_available_layouts(dtype: Optional[str] = None) -> set[str]:
    """Return a set of all layout strings available for the given data type.

    Args:
        dtype: Optional data type filter, or None for all data types.

    Returns:
        Sorted list of unique layout strings (e.g. ["NN", "NT", "TN", "TT"]).
    """
    def as_str(transpose: bool) -> str:
        return "T" if transpose else "N"

    layouts: set[str] = set()
    for info in _SCHEDULE_METADATA:
        if dtype is None or info.dtype.lower() == dtype.lower():
            layouts.add(as_str(info.TransposeA) + as_str(info.TransposeB))
    return layouts

@CallableGuard
def isNN(kernel):
    return not kernel["ProblemType"]["TransposeA"] and not kernel["ProblemType"]["TransposeB"]

@CallableGuard
def isNT(kernel):
    return not kernel["ProblemType"]["TransposeA"] and kernel["ProblemType"]["TransposeB"]

@CallableGuard
def isTT(kernel):
    return kernel["ProblemType"]["TransposeA"] and kernel["ProblemType"]["TransposeB"]

@CallableGuard
def isTN(kernel):
    return kernel["ProblemType"]["TransposeA"] and not kernel["ProblemType"]["TransposeB"]

@CallableGuard
def is16bit(kernel):
    return kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()
_register_dtype_name(is16bit, "16bit")

@CallableGuard
def is8bit(kernel):
    return kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat()
_register_dtype_name(is8bit, "8bit")

@CallableGuard
def isMixed(kernel):
    return kernel["ProblemType"]["DataTypeA"].numBytes() != kernel["ProblemType"]["DataTypeB"].numBytes()

@CallableGuard
def isTF32(kernel):
    return kernel["UseF32XEmulation"]
_register_dtype_name(isTF32, "TF32")

@dataclass(frozen=True)
class TileConfig:
    macro_tile_size_0: int
    macro_tile_size_1: int
    depth_u: int
    prefetch_global_read: int
    prefetch_local_read: int
    direct_to_lds: int
    dtl_plus_lds_buf: bool
    wave_separate_global_read_a: int
    wave_separate_global_read_b: int

class _ProbeDataType:
    """Minimal DataType stub for layout probing at registration time."""
    def isHalf(self): return False
    def isBFloat16(self): return False
    def isInt8(self): return False
    def is8bitFloat(self): return False
    def numBytes(self): return 2

class RegisterSchedule:
    """
    Decorator that registers a schedule function with its matching criteria.
    The function is wrapped with logic that checks if the kernel matches the criteria.
    Supported layouts are auto-detected by probing the inner function at registration time.

    Usage:
        @RegisterSchedule(
            tile_config=TileConfig(256, 96, 64, 2, 1, 1, False, 0, 0),
            dtype_predicate=is16bit,
            vector_widths=[8, 8, 8],
            matrix_inst=[16, 16, 32, 1],
            mfma_wave_group=[2, 2]
        )
        def _get_schedule_256x96x64_16bit(kernel, useLDSTr, TLDS):
            ...
    """

    def __init__(self, tile_config: TileConfig, dtype_predicate: Callable, vector_widths: list[int], matrix_inst: list[int], mfma_wave_group: list[int]):
        """
        Initialize the registration decorator with matching criteria.

        Args:
            tile_config:        TileConfig object
            dtype_predicate:    Callable that takes kernel and returns True if dtype matches
            vector_widths:      List of [GRVWA, GRVWB, LRVW]
            matrix_inst:        List [M, N, K, B] for MI
            mfma_wave_group:    List [rows, cols] for MIWG
        """
        self.tile_config = tile_config
        self.dtype_predicate = dtype_predicate
        self.vector_widths = vector_widths
        self.matrix_inst = matrix_inst
        self.mfma_wave_group = mfma_wave_group

    def _make_probe_kernel(self, transA: bool, transB: bool, useLDSTr: bool, TLDS: int, vectorWidthA: int, vectorWidthB: int) -> dict:
        """Build a synthetic kernel dict for probing layout support."""
        tc = self.tile_config
        mi = self.matrix_inst
        miwg = self.mfma_wave_group
        probe_dtype = _ProbeDataType()
        return {
            "ProblemType": {
                "DataType": probe_dtype,
                "DataTypeA": probe_dtype,
                "DataTypeB": probe_dtype,
                "TransposeA": transA,
                "TransposeB": transB,
            },
            "MacroTile0": tc.macro_tile_size_0,
            "MacroTile1": tc.macro_tile_size_1,
            "DepthU": tc.depth_u,
            "PrefetchGlobalRead": tc.prefetch_global_read,
            "PrefetchLocalRead": tc.prefetch_local_read,
            "DirectToLds": tc.direct_to_lds,
            "WaveSeparateGlobalReadA": tc.wave_separate_global_read_a,
            "WaveSeparateGlobalReadB": tc.wave_separate_global_read_b,
            "GlobalReadVectorWidthA": self.vector_widths[0],
            "GlobalReadVectorWidthB": self.vector_widths[1],
            "VectorWidthA": vectorWidthA,
            "VectorWidthB": vectorWidthB,
            "LocalReadVectorWidth": self.vector_widths[2],
            "MatrixInstruction": list(self.matrix_inst),
            "MIWaveGroup": list(self.mfma_wave_group),
            "LDSTrInst": useLDSTr,
            "TransposeLDS": TLDS,
            "MIWaveTileA": tc.macro_tile_size_0 // (mi[0] * miwg[0]),
            "MIWaveTileB": tc.macro_tile_size_1 // (mi[1] * miwg[1]),
            # Standard flags that inner functions may read/write
            "UseCustomMainLoopSchedule": True,
            "EnableMatrixInstruction": True,
            "UnrollLoopSwapGlobalReadOrder": False,
            "ISA": IsaVersion(9, 5, 0),
            "WavefrontSize": 64,
            "Use64bShadowLimit": 1,
            "ForceUnrollSubIter": False,
            "SwapGlobalReadOrder": False,
            "UsePLRPack": False,
            "UseF32XEmulation": False,
            "UseDirect32XEmulation": False,
            "MfmaInitCVgprs": False,
        }

    def _detect_supported_layouts(self, func: Callable) -> list[Tuple[bool, bool, bool, int]]:
        """Probe the inner function to discover which layouts it actually handles."""
        def as_str(transpose: bool) -> str:
            return "T" if transpose else "N"
        
        valid_vector_widths = [1, 2, 3, 4, 6, 8]
        detected = set()
        for transA, transB in product([True, False], repeat=2):
            for useLDSTr, TLDS in product([True, False], [1, 0]):
                for vwA, vwB in product(valid_vector_widths, repeat=2):
                    probe = self._make_probe_kernel(transA, transB, useLDSTr, TLDS, vwA, vwB)
                    try:
                        found, _ = func(probe, useLDSTr, TLDS)
                        if found:
                            detected_info_tuple = (transA, transB, useLDSTr, TLDS)
                            detected.add(detected_info_tuple)
                    except (ValueError, KeyError) as e:
                        layout = as_str(transA) + as_str(transB)
                        printWarning(
                            f"Layout probe failed for func '{func.__name__}' "
                            f"with layout={layout}, useLDSTr={useLDSTr}, TLDS={TLDS}, "
                            f"VectorWidthA={vwA}, VectorWidthB={vwB}\n"
                            f"  Kernel: {probe['MacroTile0']}x{probe['MacroTile1']}x{probe['DepthU']} {layout}\n"
                            f"  Error: {e}"
                        )
                        continue

        return list(detected)

    def __call__(self, func: Callable) -> Callable:
        """Wrap the function with matching logic and register it."""
        def wrapped_func(kernel: dict, useLDSTr: bool, TLDS: int) -> tuple[ScheduleMatchStatus, Optional[ScheduleInfo]]:
            # TODO: Currently ULSGRO not checked for in CMS, disabled for now
            if kernel["UnrollLoopSwapGlobalReadOrder"]:
                return ScheduleMatchStatus.NO_MATCH, None

            if not self.dtype_predicate(kernel):
                return ScheduleMatchStatus.NO_MATCH, None

            MT0, MT1, DU = kernel["MacroTile0"], kernel["MacroTile1"], kernel["DepthU"]
            PGR, PLR, DTL, DPLB = kernel["PrefetchGlobalRead"], kernel["PrefetchLocalRead"], kernel["DirectToLds"], kernel["DtlPlusLdsBuf"]
            WSGRA, WSGRB = kernel["WaveSeparateGlobalReadA"], kernel["WaveSeparateGlobalReadB"]
            kernel_tile_config = TileConfig(MT0, MT1, DU, PGR, PLR, DTL, DPLB, WSGRA, WSGRB)
            if self.tile_config != kernel_tile_config:
                return ScheduleMatchStatus.NO_MATCH, None

            GRVWA, GRVWB = kernel["GlobalReadVectorWidthA"], kernel["GlobalReadVectorWidthB"]
            LRVWA, LRVWB = kernel["LocalReadVectorWidthA"], kernel["LocalReadVectorWidthB"]
            kernel_vector_widths = [GRVWA, GRVWB, LRVWA, LRVWB]
            # WA: if need to support different LRVW for A and B, add a new parameter to vector_widths
            extended_vector_widths = self.vector_widths + [self.vector_widths[2]]
            if extended_vector_widths != kernel_vector_widths:
                return ScheduleMatchStatus.NO_MATCH, None
            
            if self.matrix_inst != kernel["MatrixInstruction"]:
                return ScheduleMatchStatus.NO_MATCH, None
            
            if self.mfma_wave_group != kernel["MIWaveGroup"]:
                return ScheduleMatchStatus.NO_MATCH, None
            
            # All wrapper criteria matched - call inner function
            match, schedule = func(kernel, useLDSTr, TLDS)
            
            if match:
                return ScheduleMatchStatus.FOUND, schedule
            # Inner function returned False - variant unsupported, stop searching
            return ScheduleMatchStatus.UNSUPPORTED_VARIANT, None
               
        _SCHEDULE_REGISTRY.append(wrapped_func)

        # Auto-detect supported layouts by probing the inner function
        detected_infos = self._detect_supported_layouts(func)

        # Store metadata for query API
        dtype_name = _DTYPE_PREDICATE_NAMES.get(self.dtype_predicate, str(self.dtype_predicate))
        tc = self.tile_config
        for detected_info in detected_infos:
            _transA, _transB, _useLDSTr, _TLDS = detected_info
            _SCHEDULE_METADATA.append(CMSKernelInfo(
                name=func.__name__,
                dtype=dtype_name,
                TransposeA=_transA,
                TransposeB=_transB,
                MacroTile0=tc.macro_tile_size_0,
                MacroTile1=tc.macro_tile_size_1,
                DepthU=tc.depth_u,
                PrefetchGlobalRead=tc.prefetch_global_read,
                PrefetchLocalRead=tc.prefetch_local_read,
                DirectToLds=tc.direct_to_lds,
                DtlPlusLdsBuf=tc.dtl_plus_lds_buf,
                WaveSeparateGlobalReadA=tc.wave_separate_global_read_a,
                WaveSeparateGlobalReadB=tc.wave_separate_global_read_b,
                GlobalReadVectorWidthA=self.vector_widths[0],
                GlobalReadVectorWidthB=self.vector_widths[1],
                LocalReadVectorWidth=self.vector_widths[2],
                MatrixInstruction=list(self.matrix_inst),
                MIWaveGroup=list(self.mfma_wave_group),
                LDSTrInst=_useLDSTr,
                TransposeLDS=_TLDS,
            ))
        
        # Return original function unchanged (so it can still be called directly)
        return func

@RegisterSchedule(
    tile_config=TileConfig(256, 96, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x96x64_16bit(kernel, useLDSTr, TLDS):

    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll

    if isTN(kernel) and TLDS == 1:

        nglshift = nllshift = 11
        syncTable = [
                    -1, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Finish all LRA1 and 1/3 LRB1"),
                    7, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="Finish 2/3 LRB1"),

                    15, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="All LRB1 and LRA0 done"),
                    15, SBarrier(comment=""),

                    23, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="1/3 LRB0 done"),

                    29, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="All LRB0 done"),
                    29, SBarrier(comment=""),

                    35, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="All GRA launched, 3 prev GRB."),
                    35, SBarrier(comment=""),

                    42, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="Only global reads for this iter"),
                    42, SBarrier(comment="")]

        syncCode = syncTable[1::2]
        optSchedule = {
            'SYNC'   : [syncTable[::2]],
            'GRIncA' : [[1,1,2,2,3,3,3,4,4]],
            'GRIncB' : [[5,5,6,6,6,7,7,8,8]],
            'LRA0'   : [[1,2,3,4,5,6,  8,10],
                        [1,2,3,4,5,6,  9,11]],
            'LRB0'   : [[12,16,18],
                        [13,17,19]],
            'GRB'    : [[36,36,38,38,40,40],
                        [37,37,39,39,41,41]],
            'GRA'    : [[16,16,18,18,20,20,22,22,24,24,26,26,28,28,30,30],
                        [17,17,19,19,21,21,23,23,25,25,27,27,29,29,31,31]],
            'LRA1'   : [[36,37,38,39,40,41,42,43]],
            'LRB1'   : [[44,45,46]],
            'LRSA'   : [[30]], # this must come before next reads of A X0 - so the LRA1
            'LRSB'   : [[31]], # this must come before next reads of A X0 - so the LRB1
            'LWSA'   : [[32]],  # swap after last gr a
            'LWSB'   : [[42]],  # swap after last gr b
            'LCC'   : [[47, 47]],
        }
    elif isNN(kernel) and useLDSTr and TLDS == 1:

        nglshift = nllshift = 11

        syncTable = [
            -1, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for LRB1 in prev iteration"),
            
            7, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for prior 5 LRA0"),
            20, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="All LRA0 is launched"),
            20, SBarrier(comment=""),
            
            21, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="All LRB0 launched"),
            21, SBarrier(comment=""),

            36, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="All GRA launched"),
            36, SBarrier(comment=""),

            43, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="All GRB launched"),
            43, SBarrier(comment=""),
        ]
        
        syncCode = syncTable[1::2]
        optSchedule = {
            'SYNC'   : [syncTable[::2]],
            
            'GRIncA' : [[1,1,1, 2,2,2, 3,3,3]],
            'GRIncB' : [[4,4,4, 5,5,5, 6,6,6]],
            
            'LRA0'   : [[1, 3,3, 5,5,   7,7, 9,9, 11,11, 13,13, 15,15, 17],
                        [2, 4,4, 6,6,   8,8, 10,10, 12,12, 14,14, 16,16, 18]],
            'LRB0'   : [[13, 15, 17],
                        [14, 16, 18]],

            'GRA'    : [[21,21, 23,23, 25,25, 27,27, 29,29, 31,31, 33,33, 35,35],
                        [20,20, 22,22, 24,24, 26,26, 28,28, 30,30, 32,32, 34,34]],
            'GRB'    : [[37,37, 39,39, 41,41],
                        [38,38, 40,40, 42,42]],

            'LRSA'   : [[30]],
            'LRSB'   : [[31]],

            'LWSA'   : [[36]],
            'LWSB'   : [[43]],

            'LRA1'   : [[36,36, 37,37, 38,38, 39,39, 40,40, 41,41, 42,42, 43,43]],
            'LRB1'   : [[43, 44, 45]],

            'LCC'    : [[47, 47]],
        }
    else:
        return False, None

    numMfma = 48
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 96, 64, 2, 1, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x96x64_16bit_DPLB(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll

    if isNT(kernel) and useLDSTr and TLDS == 0:
        syncTable = [
            -1, SWaitCnt(dscnt= 4, vlcnt=-1, vscnt=-1, comment="wait for all LRA1 and one LRB1"),
             7, SWaitCnt(dscnt= 6, vlcnt=-1, vscnt=-1, comment="wait the rest of LRB1"),
            23, SWaitCnt(dscnt= 0, vlcnt=-1, vscnt=-1, comment="wait for all LR0 before starting 2nd sub-iteration"),
            27, SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="wait for GRAs before starting LRA1"),
            27, SBarrier(comment=""),
            41, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="wait for GRBs before starting LRB1"),
            41, SBarrier(comment=""),
        ]

        optSchedule = {
            'SYNC': [syncTable[::2]],

            'GRIncA': [[0,1,1, 1,2,3, 3,3,4],
                       [0,0,0, 1,2,2, 2,3,4]],
            'GRIncB': [[18,18,18, 21,21,21, 22,22,22]],

            'LRA0': [[0,0, 2,2, 4,4, 6,6, 8,8, 10,10, 12,12, 14,14],
                     [1,1, 3,3, 5,5, 7,7, 9,9, 11,11, 13,13, 15,15]],
            'LRB0': [[9,11, 13,15, 16,16],
                     [10,12, 14,16, 17,17]],

            'GRA': [[5,5, 5,6, 7, 9, 11,11, 15,16, 19,20, 24,25, 26,27],
                    [4,5, 6,6, 7,10, 11,12, 15,16, 19,20, 24,25, 26,28]],
            'GRB': [[31,31, 35,35, 39,39],
                    [32,32, 36,36, 40,40]],

            'LRA1': [[28,28, 30,30, 32,32, 34,34, 36,36, 37,37, 38,38, 40,40],
                     [29,29, 31,31, 33,33, 35,35, 37,37, 39,39, 41,41, 42,42]],
            'LRB1': [[42,42, 43,43, 45,45],
                     [43,43, 44,44, 46,46]],

            'LRSA': [[26,26,26,27]],
            'LRSB': [[27]],
            'LWSA': [[44,44,44],
                     [44,45,45]],
            'LWSB': [[]],
            'LCC': [[46,46],
                    [45,46]],
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 11
    else:
        return False, None

    numMfma = 48
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    opt1.disableValidationPass(cmsv.ValidatorPass.ADD_GR_NOT_TOO_EARLY_CONSTRAINTS, "GR validation is not yet supported for DtlPlusLdsBuf")
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(192, 256, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_192x256x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isNN(kernel) and useLDSTr and TLDS==1:
        # TODO: This schedule can be improved when BC are resolved for MT192
        # Note: A/B Global read orders are swapped
        # i.e. GRA contains GR for B
        kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            'SYNC'    : [[12,13, 47,48,49,50,51, 52,53, 56,56, 95]],
            'GRIncB' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncA' : [[42,42,43,43,44,44,45,45,46]],
            'LRB0'   : [[0,0,1,1,2,2,6,8],
                        [3,3,4,4,5,5,7,9]],
            # These local reads have BC
            'LRA0'   : [[10, 15,17,19,21,23, 25,27,29,33,37,39],
                        [11, 14,16,18,20,22, 24,26,28,32,36,38]],
            'GRA'    : [[14,14, 16,16, 18,18, 20,20, 22,22, 34,34,36,36,38,38],
                        [15,15, 17,17, 19,19, 21,21, 23,23, 35,35,37,37,39,39]],
            'GRB'    : [[54,54, 56,56, 58,58, 60,60, 62,62, 64,64],
                        [55,55, 57,57, 59,59, 61,61, 63,63, 65,65]],
            'LRSA'   : [[40]],
            'LRSB'   : [[40]],
            'LWSB'   : [[41]], # For B
            'LWSA'   : [[66]], # For A
            'LRB1'   : [[57,57,59,59,61,61,63,65],
                        [58,58,60,60,62,62,64,64]],
            'LRA1'   : [[67,71,73,75,77,79,81,85,87,89,91,93],
                        [68,72,74,76,78,80,82,86,88,90,92,94]],
            'LCC'    : [[95, 95]],
        }
        syncCode = [SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=10, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SWaitCnt(dscnt=8, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=9, vscnt=-1, comment="Wait for GRA & GRB to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 & LRB1 to complete"),]
        nglshift = nllshift = 14 # vmcnt shift for ngl and nll
    elif isTN(kernel) and not useLDSTr and TLDS == 1:
        #index and code pair
        syncTable = [-1, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="for LRB1-0"),
                      5, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="for LRB1"),
                     14, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="for LRA0 complete"),
                     14, SBarrier(comment="for GRA start"),
                     46, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="for LRB0"),
                     46, SBarrier(comment="for GRB start"),
                     50, SWaitCnt(dscnt=-1, vlcnt=14+1, vscnt=-1, comment="for LRA1"),
                     50, SBarrier(comment="for LRA1 start"),
                     65, SWaitCnt(dscnt=-1, vlcnt=6+5, vscnt=-1, comment="for LRB0"),
                     65, SBarrier(comment="for LRB1 start"),]
        optSchedule = {
                'SYNC'  : [syncTable[::2]],
                'GRIncA': [[6,6,7,7,8,8,9,9,9]],
                'GRIncB': [[33,34,35,36,37,38,39,40,41]],

                'LRA0'  : [[0, 1, 2, 3, 4, 5],
                           [-1, 0, 1, 2, 3, 4]],
                'LRB0'  : [[7, 9, 11, 13, 15, 17, 19, 21],
                           [8, 10, 12, 13, 16, 18, 20, 22]],
                'GRA'   : [[14,14, 16,16, 18,18, 20,20, 25,25, 31,31],
                           [15,15, 17,17, 19,19, 21,21, 26,26, 32,32]],

                'GRB'   : [[46,46, 50,50, 54,54, 58,58, 62,62, 66,66, 70,70, 76,76],
                           [47,47, 51,51, 55,55, 59,59, 63,63, 67,67, 71,71, 77,77]],
                'LRA1'  : [[50, 52, 56, 58, 60, 62],
                            [51, 53, 57, 59, 61, 63]],
                'LRB1'  : [[65, 67, 69, 71, 73, 75, 77, 79],
                           [66, 68, 70, 72, 74, 76, 78, 80]],

                'LRSA'  : [[47]],
                'LRSB'  : [[47]],
                'LWSA'  : [[47]],
                'LWSB'  : [[80]],
                'LCC'   : [[95, 95]],
            }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 14 # vmcnt shift for ngl and nll
    elif isNT(kernel) and not useLDSTr and TLDS == 0:
        optSchedule = {
            'SYNC'  : [[-1, 25, 25, 46, 46, 55, 55, 72, 72]],
            'GRIncA': [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
            'GRIncB': [[3, 3, 3, 4, 4, 4, 5, 5, 5]],
            'LRA0'  : [[0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20, 22, 22],
                       [1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15, 17, 17, 19, 19, 21, 21, 23, 23]],
            'LRB0'  : [[24, 24, 26, 26, 28, 28, 30 ,30],
                       [25, 25, 27, 27, 29, 29, 31, 31]],
            'GRA'   : [[25, 25, 27, 27, 29, 29, 31, 31, 33, 33, 35, 35],
                       [26, 26, 28, 28, 30, 30, 32, 32, 34, 34, 36, 36]],
            'GRB'   : [[47, 47, 49, 49, 51, 51, 53, 53, 64, 64, 66, 66, 68, 68, 70, 70],
                       [48, 48, 50, 50, 52, 52, 54, 54, 65, 65, 67, 67, 69, 69, 71, 71]],
            'LRA1'  : [[55, 55, 57, 57, 59, 59, 61, 61, 63, 63, 65, 65, 67, 67, 69, 69, 87, 87, 89, 89, 91, 91, 93, 93],
                       [56, 56, 58, 58, 60, 60, 62, 62, 64, 64, 66, 66, 68, 68, 70, 70, 88, 88, 90, 90, 92, 92, 94, 94]],
            'LRB1'  : [[72, 74, 76, 78, 80, 82, 84, 86],
                       [73, 75, 77, 79, 81, 83, 85, 87]],
            'LRSA'  : [[37]],
            'LRSB'  : [[37]],
            'LWSA'  : [[71]],
            'LWSB'  : [[71]],
            'PackB1': [[-1, -1, -1, -1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]],
            'PackA1': [[-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]],
            'PackB0': [[47, 47, 47, 47, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53]],
            'PackA0': [[47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50]],
            'LCC'   : [[95, 95]],
        }

        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 to complete") ,
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete") ,
                    SBarrier(comment="") ,
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete") ,
                    SBarrier(comment="") ,
                    SWaitCnt(dscnt=-1, vlcnt=14+4, vscnt=-1, comment="Wait for global reads to complete") ,
                    SBarrier(comment="") ,
                    SWaitCnt(dscnt=-1, vlcnt=14, vscnt=-1, comment="Wait for global reads to complete") ,
                    SBarrier(comment="") ,
        ]
        nglshift = nllshift = 14
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 96
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 192, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x192x64_16bit(kernel, useLDSTr, TLDS):
    numMfma = 96
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and not useLDSTr and TLDS == 1:
        #index and code pair
        syncTable = [-1, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="wait for LRB1-0"),
                     7, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="wait for LRB1"),
                     10, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="wait for LRA0"),
                     10, SBarrier(comment="for GRA"),

                     47, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0-0"),
                     50, SWaitCnt(dscnt=-1, vlcnt=14, vscnt=-1, comment="for previous GRA"),
                     50, SBarrier(comment="for GRA"),

                     70, SWaitCnt(dscnt=-1, vlcnt=12, vscnt=-1, comment="for previous GRB"),
                     70, SBarrier(comment="for GRB"),
                     ]
        optSchedule = {
                'SYNC'  : [syncTable[::2]],
                'GRIncA': [[0,1,2,3,4,5,6,7,8]],
                'GRIncB': [[37,37,38,38,39,39,40,40,41]],
                'LRA0': [[-1, 0, 1, 2, 3, 4, 5, 6],
                         [0, 1, 2, 3, 4, 5, 6, 7]],
                 #interleave LRB0 , GRA
                'LRB0': [[7, 9, 11, 13, 15, 17],
                        [8, 10, 12, 14, 16, 18]],
                'GRA': [[10,10, 12,12, 14,14, 16,16, 20,20, 31,31, 33,33, 35,35],
                        [11,11, 13,13, 15,15, 17,17, 21,21, 32,32, 34,34, 36,36]],
                 #interleave GRB, LRB1
                'GRB': [[51,51, 55,55, 59,59, 63,63, 83,83, 85,85],
                        [52,52, 56,56, 60,60, 64,64, 84,84, 86,86]],
                'LRA1': [[50, 52, 57, 60, 62, 64, 66, 68],
                         [51, 53, 58, 61, 63, 65, 67, 69]],

                'LRB1': [[70, 72, 74, 76, 78, 79],
                         [71, 73, 75, 77, 79, 80]],
                'LRSA': [[20]],
                'LRSB': [[64]],
                'LWSA': [[41]],
                'LWSB': [[90]],
                'LCC' : [[95, 95]],
            }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 14 # vmcnt shift for ngl and nll
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNT(kernel) and useLDSTr and TLDS == 0:
        kernel["SwapGlobalReadOrder"] = True
        #index and code pair
        syncTable = [-1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="for LRB1"),
                     29, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="wait for LRB0. For code path 0, this is actually wait for LRB0 + 1/16 LRA0"),
                     29, SBarrier(comment="for GRA"),
                     47, SWaitCnt(dscnt=0, vlcnt=14, vscnt=-1, comment="wait for previous GRB and LRA0"),
                     47, SBarrier(comment="for GRB"),
                     70, SWaitCnt(dscnt=-1, vlcnt=14-3, vscnt=-1, comment="wait for previous GRA"),
                     70, SBarrier(comment="for GRB"),
                     ]
        optSchedule = {
                'SYNC'  : [syncTable[::2]],
                'GRIncA': [[18, 19,20,21,22,23,24,25,26]],
                'GRIncB': [[9,10,11,12,13,14,15,16,17]],

                'LRB0': [[-1,1, 3,5, 7,9, 11,13, 15,17, 19,21],
                         [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]],
                'LRA0': [[23, 24, 25, 26, 27, 28, 29, 30,31, 32,33, 34,35, 36,37, 38],
                         [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]],
                'GRA': [[29,29, 31,31, 33,33, 38,38, 40,40, 41,41],
                         [30, 30, 32, 32, 34, 34, 39, 39, 41, 41, 42, 42]],

                'GRB': [[57,57, 59,59, 61,61, 63,63, 65,65, 70,70, 75,75, 80,80],
                        [58, 58, 60, 60, 62, 62, 64, 64, 66, 66, 71, 71, 76, 76, 81, 81]],
                'LRB1': [[47,48, 50,52, 54,56, 58,60, 62,64, 66,68],
                         [48, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69]],
                'LRA1': [[70,71, 72,73, 74,75, 76,77, 78,79, 80,81, 82,83, 84,85],
                         [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86]],

                'LRSB': [[39]],
                'LRSA': [[46]],
                'LWSB': [[78]],
                'LWSA': [[95]],
                'LCC' : [[95, 95]],}
        syncCode = syncTable[1::2]
        nglshift = nllshift = 14 # vmcnt shift for ngl and nll
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNN(kernel) and useLDSTr and TLDS == 1:
        kernel["SwapGlobalReadOrder"] = True
        #index and code pair
        syncTable = [-1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRA1"),
                     15, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="wait for LRB0"),
                     15, SBarrier(comment=""),
                     46, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
                     51, SWaitCnt(dscnt=-1, vlcnt=14, vscnt=-1, comment="wait for previous set of global reads"),
                     51, SBarrier(comment=""),
                     63, SWaitCnt(dscnt=-1, vlcnt=14-4, vscnt=-1, comment="wait for previous set of global reads"),
                     63, SBarrier(comment=""),
                    ]
        optSchedule = {
                    'SYNC'  : [syncTable[::2]],
                    'GRIncA': [[35,36,37,38,39,40,41,42,43]],
                    'GRIncB': [[0,1,2,3,4,5,6,7,8]],

                    'LRB0': [[-1, 0, 1, 2, 3, 4],
                             [0, 1, 2, 3, 4, 5]],
                    'LRA0': [[6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 27, 29, 31, 33, 35],
                             [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 28, 30, 32, 34, 36]],
                    'GRA': [[15,15, 17,17, 27,27, 29,29, 31,31, 33,33],
                            [16, 16, 18, 18, 28, 28, 30, 30, 32, 32, 34, 34]],

                    'GRB': [[51,51, 53,53, 55,55, 57,57, 67,67, 69,69, 71,71, 73,73],
                            [52,52, 54,54, 56,56, 58,58, 68,68, 70,70, 72,72, 74,74]],
                    'LRB1': [[51, 53, 55, 57, 59, 61],
                             [52, 54, 56, 58, 60, 62]],
                    'LRA1': [[63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93],
                             [64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94]],

                    'LRSB': [[14]],
                    'LRSA': [[45]],
                    'LWSB': [[94]],
                    'LWSA': [[94]],
                    'LCC' : [[95, 95]],
                }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 14 # vmcnt shift for ngl and nll
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 256, 128, 2, 0, 1, False, 0, 0),
    dtype_predicate=is8bit,
    vector_widths=[16, 16, 16],
    matrix_inst=[16, 16, 128, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x256x128_8bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []

    plr = 3 if kernel["ForceUnrollSubIter"] else 1
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and TLDS == 1:
        optSchedule = {
            'SYNC'      : [[6,7, 20,21, 46,47, 61]],
            'GRIncA'    : [[0,1,2,3,4,4,4,4,4]],
            'GRIncB'    : [[5,5,5,5,5,6,6,6,6]],
            'LRA0'      : [[0,0, 1,1, 2,2, 3,3]],
            'GRA'       : [[8,8,9,9,10,10,11,11,12,12, 23,23,24,24,25,25]],
            'LRB0'      : [[13,13,14,14,15,15,16,16]],
            'LRA%u'%plr : [[48,48,49,49,50,50,51,51]],
            'LRB%u'%plr : [[52,52,54,54,55,55,56,56]],
            'GRB'       : [[26,26,27,27, 39,39,40,40,41,41,42,42,43,43, 53,53]],
            'LCC'       : [[60, 60]],
            'LRSA'      : [[17]],
            'LRSB'      : [[17]],
            'LWSA'      : [[57]],
            'LWSB'      : [[57]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0/LRB0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0/LRB0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=15, vscnt=-1, comment="Wait for GRA to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for PLR to complete")]
        nglshift = nllshift = 16
    else:
        return False, None

    numMfma = 64
    # B0A0, B0A1, B1A0, B1A1
    mfmaReorder = []
    kernel["MfmaInitCVgprs"] = True
    if not kernel["ForceUnrollSubIter"]:
        mfmaReorder = [0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27,
                       4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31,
                       32,33,34,35, 40,41,42,43, 48,49,50,51, 56,57,58,59,
                       36,37,38,39, 44,45,46,47, 52,53,54,55, 60,61,62,63]
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift, mfmaReorder)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 256, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x256x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []

    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and TLDS == 1:
        optSchedule = {
            'SYNC'   : [[19,20, 50,51, 67,68, 104, 105, 127]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[9,10,11,12,13,14,15,16,17]],
            'LRA0'   : [[0,2,4,6,8,10,12,14],
                        [1,3,5,7,9,11,13,15]],
            'LRB0'   : [[24,27,30,33,36,38,40,42],
                        [22,25,28,31,34,37,39,41]],
            'GRA'    : [[21,22, 23,25, 26,28, 29,31, 32,34, 35,52, 53,55, 56,58],
                        [21,23, 24,26, 27,29, 30,32, 33,35, 36,53, 54,56, 57,59]],
            'GRB'    : [[59,61, 62,64, 65,85, 86,87, 88,89, 94,96, 98,100, 102,124],
                        [60,62, 63,65, 66,84, 85,86, 87,88, 93,95, 97,99, 103,123]],
            'LRA1'   : [[69,71,73,75,77,79,81,83],
                        [70,72,74,76,78,80,82,90]],
            'LRB1'   : [[106,108,110,112,114,116,118,120],
                        [107,109,111,113,115,117,119,121]],
            'LRSA'   : [[16]],
            'LRSB'   : [[83]],
            'LWSA'   : [[125]],
            'LWSB'   : [[125]],
            'LCC'   : [[126, 126]],
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=(2 + 8 + 8), vscnt=-1, comment="Wait for previous GRA to completely"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=15, vscnt=-1, comment="Wait for previous GRB to completely"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 and 3/8 LRB1 to complete")]
        nglshift = nllshift = 16
    elif isNT(kernel) and not useLDSTr and TLDS == 0:
        kernel["UsePLRPack"] = True

        optSchedule = {
            'SYNC'   : [[12,13, 36,44, 56,59, 66,68, 73,92]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[28,29,30,31,32,33,34,35,36]],
            'LRA0'   : [[0,0,2,2,4,4,6,6],
                        [1,1,3,3,5,5,7,7]],
            'LRB0'   : [[8,8,10,10,15,15,18,18],
                        [9,9,11,11,14,14,17,17]],
            'GRA'    : [[14,14, 17,17, 20,20, 23,23, 26,26,   45,45, 48,48, 51,51],
                        [15,15, 18,18, 21,21, 24,24, 27,27,   46,46, 49,49, 52,52]],
            'GRB'    : [[54,54, 57,57, 87,87,90,90,93,93,96,96,99,99, 123,123],
                        [55,55, 58,58, 88,88,91,91,94,94,97,97,100,100, 124,124]],
            'LRA1'   : [[60,60,62,62,64,64,66,66],
                        [61,61,63,63,65,65,67,67]],
            'LRB1'   : [[69,69,71,71,73,73,75,75],
                        [70,70,72,72,74,74,76,76]],
            'LRSA'   : [[59]],
            'LRSB'   : [[59]],
            'LWSA'   : [[125]],
            'LWSB'   : [[125]],
            'LCC'    : [[126, 126]],
            'PackA0' : [[16,16, 19,19, 21,21, 22,22, 24,24, 25,25, 27,27, 28,28, 29,29, 30,30, 31,31, 32,32, 33,33, 34,34, 35,35, 36,36],
                        [16,16, 19,19, 20,20, 22,22, 23,23, 25,25, 26,26, 28,28, 29,29, 30,30, 31,31, 32,32, 33,33, 34,34, 35,35, 36,36]],
            'PackB0' : [[37,37, 38,38, 39,39, 40,40, 41,41, 42,42, 43,43, 46,46, 47,47, 49,49, 50,50, 52,52, 53,53, 55,55, 56,56, 58,58],
                        [37,37, 38,38, 39,39, 40,40, 41,41, 42,42, 43,43, 45,45, 47,47, 48,48, 50,50, 51,51, 53,53, 54,54, 56,56, 57,57]],
            'PackA1' : [[74,74, 76,76, 77,77, 78,78, 79,79, 80,80, 81,81, 82,82, 83,83, 84,84, 85,85, 86,86, 88,88, 89,89, 91,91, 92,92],
                        [75,75, 77,77, 78,78, 79,79, 80,80, 81,81, 82,82, 83,83, 84,84, 85,85, 86,86, 87,87, 89,89, 90,90, 92,92, 93,93]],
            'PackB1' : [[94,94, 95,95, 97,97, 98,98, 100,100, 101,101, 102,102, 103,103, 104,104, 105,105, 106,106, 107,107, 108,108, 109,109, 110,110, 111,111],
                        [95,95, 96,96, 98,98, 99,99, 101,101, 102,102, 103,103, 104,104, 105,105, 106,106, 107,107, 108,108, 109,109, 110,110, 111,111, 112,112]],
        }
        syncCode = [SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=17, vscnt=-1, comment="Wait for GRA to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=9, vscnt=-1, comment="Wait for GRB to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 to complete"),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB1 to complete")]
        nglshift = nllshift = 16
    elif (isNN(kernel) or isTT(kernel)) and not useLDSTr and TLDS == 1:
        kernel["UsePLRPack"] = True

        optSchedule = {
            'SYNC'   : [[8, 12,13, 36,44, 56,59, 66,68, 74, 127]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[28,29,30,31,32,33,34,35,36]],
            'LRA0'   : [[0,0,2,2,4,4,6,6],
                        [1,1,3,3,5,5,7,7]],
            'LRB0'   : [[9,11, 15,18,21,24,27,30],
                        [10,12, 14,17,20,23,26,29]],
            'GRA'    : [[14,14, 17,17, 20,20, 23,23, 26,26,   45,45, 48,48, 51,51],
                        [15,15, 18,18, 21,21, 24,24, 27,27,   46,46, 49,49, 52,52]],
            'GRB'    : [[54,54, 57,57, 87,87,90,90,93,93,96,96,99,99, 123,123],
                        [55,55, 58,58, 88,88,91,91,94,94,97,97,100,100, 124,124]],
            'LRA1'   : [[60,60,62,62,64,64,66,66],
                        [61,61,63,63,65,65,67,67]],
            'LRB1'   : [[68,70,72,74,76,78,80,82],
                        [69,71,73,75,77,79,81,83]],
            'LRSA'   : [[59]],
            'LRSB'   : [[59]],
            'LWSA'   : [[125]],
            'LWSB'   : [[125]],
            'LCC'    : [[126, 126]],
            'PackA0' : [[8,8, 16,16, 19,19, 22,22, 25,25, 28,28, 29,29, 31,31, 32,32, 33,33, 34,34, 35,35, 36,36, 37,37, 38,38, 39,39]],
            'PackA1' : [[75,75, 77,77, 79,79, 81,81, 83,83, 84,84, 85,85, 86,86, 88,88, 89,89, 91,91, 92,92, 94,94, 95,95, 97,97, 98,98],
                        [74,74, 76,76, 78,78, 80,80, 82,82, 84,84, 85,85, 86,86, 87,87, 89,89, 90,90, 92,92, 93,93, 95,95, 96,96, 98,98]],
        }
        syncCode = [SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 first half to complete"),
                    SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=17, vscnt=-1, comment="Wait for GRA to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=9, vscnt=-1, comment="Wait for GRB to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 to complete"),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB1 to complete")]
        if isTT(kernel):
            kernel["SwapGlobalReadOrder"] = True

            optSchedule['GRIncA'], optSchedule['GRIncB'] = optSchedule['GRIncB'], optSchedule['GRIncA']
            optSchedule['LRA0'], optSchedule['LRB0'] = optSchedule['LRB0'], optSchedule['LRA0']
            optSchedule['LRA1'], optSchedule['LRB1'] = optSchedule['LRB1'], optSchedule['LRA1']
            optSchedule['PackB0'] = optSchedule['PackA0']
            optSchedule['PackB1'] = optSchedule['PackA1']
            del optSchedule['PackA0'], optSchedule['PackA1']
        nglshift = nllshift = 16
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 128
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(160, 256, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_160x256x64_16bit(kernel, useLDSTr, TLDS):
    numMfma = 80
    optSchedule = dict()
    syncCode = []

    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and TLDS==1:
        optSchedule = {

            'SYNC'   : [[-1, 4, 13,13, 38,39, 42,43, 70,70]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[29,30,31,32,33,34,35,36,37]],

            'LRA0'   : [[0,2,3,4,5]],  ## -2 is place holder

            'LRB0'   : [[13,15,18,21,24,26,28,30],   ## After LRA0, we can mix LRB0 and GRA
                        [14,16,19,22,25,27,29,31]],
            ## GRA should start after LRA0 is done.
            'GRA'    : [[11,14, 17,17, 20,20, 23,23, 26,27],
                        [12,15, 18,18, 21,21, 24,24, 27,28]],

            ## GRB should start after LRB0 is done
            'GRB'    : [[40,40, 43,43, 46,46, 49,49, 59,59, 62,62, 65,65, 67,68],  # m0 inc is part of GRA/GRB
                        [41,41, 44,44, 47,47, 57,57, 60,60, 63,63, 66,66, 68,69]],
            'LRA1'   : [[44, 47, 53, 58, 63],
                        [45, 48, 54, 59, 64]],

            #After GRB is done.
            'LRB1'   : [[70,71,72,73,75,76,77,78]],

            'LRSA'   : [[33]], # after LRA0 and before LRA1
            'LRSB'   : [[33]], # after LRB0 and before LRB2
            'LWSA'   : [[74]], # For A
            'LWSB'   : [[76]],

            'LCC'    : [[79, 79]],
        }
        # note: syncCode needs to be
        syncCode = [SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="Wait for necessary prior LRA1/LRB1 before starting main loop"),
                    SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 for the remaining main loop"),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete to start GRA"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete to start GRB"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=(13 + 1), vscnt=-1, comment="Wait for GRA to complete to start LRA1"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=13, vscnt=-1, comment="Wait for GRB to complete to start LRB1"),
                    SBarrier(comment=""),
                   ]
        nglshift = nllshift = 13 # vmcnt shift for ngl and nll
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNN(kernel) and useLDSTr and TLDS==1:
        kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            'SYNC'   : [[-1,
            12, 12, # Wait for B
            24, 24, # wait LRB0.
            41, 41,
            61, 61  # wait GRA.
            ]],
            # Addr. update (be done before GRA/GRB).
            'GRIncA' : [[5,6,7,8,9,10,11,12,13]],
            'GRIncB' : [[0,0,1,1,2,2,3,3,4]],
            # Current iteration.
            'LRA0'   : [[8,9,10,11,12,13,14,15,16,17]],
            'LRB0'   : [[0,1,2,3,4,5,6,7]],
            # Buffer loads.
            'GRB'    : [[51,51, 55,55, 59,61, 76,77, 78,78]],
            'GRA'    : [[11,12, 16,16, 20,20, 24,24, 28, 28, 32,32, 36, 36, 40, 40]],
            # Prefetch next iteration.
            'LRA1'   : [[62,63,64,65,66,67,68,69,70,71]],
            'LRB1'   : [[41,42,43,44,45,46,47,49]],
            'LRSA'   : [[39]],
            'LRSB'   : [[39]],
            'LWSA'   : [[60]],
            'LWSB'   : [[60]],
            'LCC'   : [[79, 79]], # Loop control.
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 LRB1"),
                    SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRB0"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=13, vscnt=-1, comment="Wait for previous GRA(B) to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="Wait for previous GRB(A) to complete"),
                    SBarrier(comment="")]
        nglshift = nllshift = 13
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNT(kernel) and useLDSTr and TLDS==0:
        optSchedule = {
            'SYNC': [[-1,17,17,57,57]],
            'GRA': [[16,17,20,20,24,24,28,28,31,31]],
            'GRB': [[35,35,39,39,68,68,70,70,71,71,76,76,77,77,78,78]],
            'GRIncA': [[0,0,1,1,2,2,3,3,4]],
            'GRIncB': [[4,5,5,13,13,13,14,14,14]],
            'LCC': [[79,79]],
            'LRA0': [[0,1,1,2,2,3,3,4,4,5]],
            'LRA1': [[58,60,62,64,66,67,67,68,68,69]],
            'LRB0': [[0,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12]],
            'LRB1': [[59,61,63,65,67,71,72,72,73,73,74,74,75,75,76,76]],
            'LRSA': [[38]],
            'LRSB': [[38]],
            'LWSA': [[61]],
            'LWSB': [[61]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=7, vscnt=-1, comment="wait for previous set of global reads"),
            SBarrier(comment="")
        ]
        nglshift = nllshift = 13
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(96, 256, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_96x256x64_16bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True

    numMfma = 48
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 11

    if isTN(kernel) and TLDS==1:
        kernel["SwapGlobalReadOrder"] = True

        syncTable = [
            7, SWaitCnt(dscnt=8, vlcnt=-1, vscnt=-1, comment="Finish all LRA1s and LRB1s"),

            9, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="All LRB0 done"),
            9, SBarrier(comment=""),

            23, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="All LRA0 done"),

            35, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="Wait for prev GRBs"),
            35, SBarrier(comment=""),

            43, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="Wait for prev GRA"),
            43, SBarrier(comment=""),

            47, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Finish all LRB1 and 1/3 LRA1"),
        ]

        syncCode = syncTable[1::2]
        optSchedule = {
            'GRIncA' : [[0,0,0,    2,2,2, 3, 4,4],
                        [-1,-1,-1, 1,1,1, 3, 3,3]],
            'GRIncB' : [[4, 5,5,5, 6,6,6, 7,7],
                        [4, 5,5,5, 6,6,6, 7,7]],

            'LRB0'   : [[-1,-1,-1, 1,1,1, 3,3],
                        [0,0,0,    2,2,2, 4,4]],
            'LRSB'   : [[8], [9]],
            
            'SYNC'   : [syncTable[::2]],

            # Actually loads GRB
            'GRA'    : [[8,9,  11,11, 13,13, 15,15,  20,20, 22,22, 24,24, 26,26],
                        [8,10, 12,12, 14,14, 16,16,  21,21, 23,23, 25,25, 27,27]],
            'LWSB'   : [[32]],

            'LRA0'   : [[17, 17, 17],
                        [15, 15, 15]],
            'LRSA'   : [[19]],
            
            # Actually loads GRA
            'GRB'    : [[36,36, 38,38, 40,40],
                        [37,37, 39,39, 41,41]],
            'LWSA'   : [[45]],
            
            'LRB1'   : [[35,35, 37,37, 39,39, 41,41],
                        [36,36, 38,38, 40,40, 42,42]],
            'LRA1'   : [[43,44,46],
                        [43,45,46]],
            
            'LCC'   : [[47, 47]],
        }

        # Reorder to basically create the 256x96 case
        mfmaReorder = [
             0,  3,  6,  9, 12, 15, 18, 21,
             1,  4,  7, 10, 13, 16, 19, 22,
             2,  5,  8, 11, 14, 17, 20, 23,
              # Second half
             24, 27, 30, 33, 36, 39, 42, 45,
             25, 28, 31, 34, 37, 40, 43, 46,
             26, 29, 32, 35, 38, 41, 44, 47
        ]

        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift, mfmaReorder=mfmaReorder)

    elif isNT(kernel) and useLDSTr and TLDS == 0:
        # A: MIWaveTileA=3 => 2*3 = 6 local/global reads
        # B: MIWaveTileB=8 => 2*8 = 16 local/global reads
        syncTable = [
            -1, SWaitCnt(dscnt=12, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LR/LW for iteration == 0"),
             5, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LR/LW for iteration == 0"),
            12, SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="wait for LRA0 to complete before GRA"),
            12, SBarrier(comment="barrier after LRA0, before GRA"),
            23, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0 to complete before GRB"),
            23, SBarrier(comment="barrier after LRB0, before GRB"),

            25, SWaitCnt(dscnt=-1, vlcnt=11+1, vscnt=-1, comment="wait for global reads before next-tile LR"),
            25, SBarrier(comment="final barrier before LRA1/LRB1"),
            36, SWaitCnt(dscnt=-1, vlcnt=11-2, vscnt=-1, comment="wait for global reads before next-tile LR"),
            36, SBarrier(comment="final barrier before LRA1/LRB1"),
        ]
        optSchedule = {
            'SYNC'   : [syncTable[::2]],

            # Address increments
            'GRIncA' : [[0, 0, 1, 1, 2, 2, 3, 3, 4]],
            'GRIncB' : [[4, 5, 5, 6, 6, 7, 7, 8, 8]],

            # Current-iteration local reads
            'LRA0'   : [[0, 1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 4, 6]],
            # Ensure LRB0 completes early enough for upcoming MFMA consumers.
            'LRB0'   : [[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                        [7, 8, 9, 10, 11, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]],

            # Global reads (DirectToLds). Issue after LRA0 barrier, while finishing LRB0.
            # Do not interleave GRA between GRIncB indices (SCC hazard).
            'GRA'    : [[12, 12, 18, 18, 21, 21],
                        [13, 13, 19, 19, 22, 22]],
            # Start GRB after the LRB0 barrier. Use stride=2 starting at 24.
            'GRB'    : [[24, 24, 26, 26, 28, 28, 30, 30, 32, 32, 34, 34, 38, 38, 40, 40],
                        [24, 24, 27, 27, 29, 29, 31, 31, 33, 33, 35, 35, 37, 37, 39, 39]],

            # Next-iteration local reads
            'LRA1'   : [[25, 26, 27, 28, 29, 30],
                        [26, 27, 28, 29, 30, 31]],
            # Keep within the mfma window; repeated indices are allowed (multiple instructions scheduled at same mfma slot).
            'LRB1'   : [[36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 43, 45, 46, 47],
                        [37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 44, 46, 47, 47]],

            # Epilogue-related
            'LRSA'   : [[22]],
            'LRSB'   : [[23]],
            'LWSA'   : [[43]],
            'LWSB'   : [[44]],
            'LCC'    : [[45, 46]],
        }

        syncCode = syncTable[1::2]
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNN(kernel) and not useLDSTr and TLDS==1:
        snops = []
        syncs = SyncSchedule()
        syncs.add(-1, dscnt=4, comment="Wait for prior local read local write")
        syncs.add(2, dscnt=3, comment="Wait for prior local read local write")
        syncs.add(8, dscnt=6, comment="Wait for partial LRA0")
        syncs.add(16, dscnt=0, barrier=True, comment="Wait for LRA0 to complete before starting LRB0+GRA")
        syncs.add(23, dscnt=0, vlcnt=3, barrier=True, comment="Wait for LRB0+GRA")

        snopIdxs = [1, 25]
        snops = [[x, SNop(1, comment="")] for x in snopIdxs]

        lra0 = [0,0,1,1,2,2,3,3,4,5,5,6,6,7,7,8,8,9,10,11,12,13,14,15]

        # Issue LRB0+GRA after LRA0 completes.
        lrb0 = [16, 17, 18, 19, 20, 21, 22, 22]
        grA =  [18, 18, 20, 20, 21, 21]

        # Issue LRA1+GRB after LRB0+GRA complete.
        lra1 = [24,24, 25,25, 26,26, 27,27, 28, 29,29, 30,30, 31,31, 32,32, 33,34,35,36,37,38,39]
        grB  = [24,24,26,26,28,28,30,30,32,32,36,36,38,38,40,40]
        lrb1 = [39,40,41,42,43,44,45,46]

        packA1 = [
            -1,-1,-1,-1,-1,-1,
            0,0,0,0,
            1,1,
        ]
        packA0 = [
            23,23,23,23,23,23,
            24,24,24,24,
            25,25,
        ]

        # GRIncs should be ordered AFTER LRs.
        grIncA = [0,1,2,3,4,5,6,7,8]
        grIncB = [9,10,11,12,13,14,15,16,17]

        lwsa = [46]
        lwsb = [46]
        lrsa = [22]
        lrsb = [22]
        num_gr = (len(grA) + len(grB)) // 2
        optSchedule = {
            'SYNC'   : [syncs.get_indicies()],
            'LRA0'   : [lra0],
            'LRA1'   : [lra1],
            'PackA0' : [packA0],
            'PackA1' : [packA1],
            'LRB0'   : [lrb0],
            'LRB1'   : [lrb1],
            'GRIncA' : [grIncA],
            'GRIncB' : [grIncB],
            'GRA'    : [grA],
            'GRB'    : [grB],
            'LRSA'   : [lrsa],
            'LRSB'   : [lrsb],
            'LWSA'   : [lwsa],
            'LWSB'   : [lwsb],
            'LCC'    : [[47, 47]],
        }
        nllshift = nglshift = num_gr
        if snops:
            optSchedule['SNOP'] = [[s[0] for s in snops]]
            snopCode = [s[1] for s in snops]

        opt1 = ScheduleInfo(1, 48, optSchedule=optSchedule, syncCode=syncs.get_code(), nglshift=nglshift, nllshift=nllshift, snopCode=snopCode)
    else:
        return False, None

    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 160, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x160x64_16bit(kernel, useLDSTr, TLDS):
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    numMfma = 80
    if isNN(kernel) and useLDSTr and TLDS==1:
        kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            'SYNC'   : [[-1,
            12,12, # Wait for LRB0
            24, 24,# Wait LRA0.
            41,41,
            61, 61 # Wait previous GR.
            ]],
            # Addr. update (be done before GRA/GRB).
            'GRIncA' : [[21,21,21,22,22,22,23,23,23]],
            'GRIncB' : [[0,1,2,3,4,5,6,7,8]],
            # Current iteration.
            'LRA0'   : [[5,5,7,7,9,9,11,11,13,13,15,15,17,18,19,20]],
            'LRB0'   : [[0,0,1,2,3]],
            # Buffer loads.
            'GRB'    : [[30,30, 33,33, 36,36, 52,52, 56,56, 60,61, 76,77, 78,78]],
            'GRA'    : [[11,12, 16,16, 20,20, 25,25, 26, 28]],
            # Prefetch next iteration.
            'LRA1'   : [[62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77]],
            'LRB1'   : [[41,42,43,44,45]],
            'LRSA'   : [[39]],
            'LRSB'   : [[39]],
            'LWSA'   : [[60]],
            'LWSB'   : [[60]],
            'LCC'   : [[79, 79]], # Loop control
        }
        syncCode = [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 LRB1"),
                    SWaitCnt(dscnt=8, vlcnt=-1, vscnt=-1, comment="Wait for LRB0"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for previous GRB to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=(13+8-5), vscnt=-1, comment="Wait for previous GRB to complete"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=(13+10-13), vscnt=-1, comment="Wait for previous GRA to complete"),
                    SBarrier(comment="")]
        nglshift = nllshift = 13
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isTN(kernel) and (not useLDSTr) and TLDS==1:
        syncTable = [
            -1, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1 (partial) before starting main loop"),
             4, SWaitCnt(dscnt=0+2, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1 (complete) for the remaining main loop"),
            14, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete to start GRB"),
            14, SBarrier(comment=""),
            # Must be dscnt=0 here: validator requires proving all LRA0 are complete
            # before the first GRA is issued (vmfma_index window [31,41)).
            39, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 / ensure LRA0 complete before starting GRB/GRA"),
            39, SBarrier(comment=""),
            45, SWaitCnt(dscnt=-1, vlcnt=13+2, vscnt=-1, comment="Wait for GRB to complete before LRB1"),
            45, SBarrier(comment=""),
            69, SWaitCnt(dscnt=-1, vlcnt=13, vscnt=-1, comment="Wait for GRA to complete before LRA1"),
            69, SBarrier(comment=""),
        ]
        optSchedule = {
            'SYNC'   : [syncTable[::2]],
            'GRIncA' : [[29,30,31,32,33,34,35,36,37]],
            'GRIncB' : [[0,1,2,3,4,5,6,7,8]],

            # Current iteration.
            'LRB0'   : [[0,2,3,4,5],
                        [1,3,4,5,6]],
            'LRA0'   : [[13,15,18,21,24,26,28,30],
                        [13,16,19,22,25,27,29,31]],

            # GRB must not start before the SYNC at idx 15 (LRB0 completion).
            'GRB'    : [[14,14, 17,17, 20,20, 23,23, 26,26],
                        [15,15, 18,18, 21,21, 24,24, 27,27]],
            # Buffer loads.
            'GRA'    : [[40,40, 43,43, 46,46, 49,49, 59,59, 62,62, 65,65, 67,67],
                        [41,41, 44,44, 47,47, 57,57, 60,60, 63,63, 66,66, 68,68]],
            # Prefetch next iteration.
            # Need 5 local reads for B (MIWaveTileB=5).
            'LRB1'   : [[45,46,47,48,49],
                        [46,47,48,49,50]],
            # Need 8 local reads for A (MIWaveTileA=8) in each code path.
            # Path1 LRA1 must be earlier than path0 (validator requirement).
            'LRA1'   : [[69, 70, 71, 72, 73, 74, 75, 76],
                        [70, 71, 72, 73, 74, 75, 76, 77]],
            'LRSA'   : [[32]],
            'LRSB'   : [[33]],
            'LWSA'   : [[74]],
            'LWSB'   : [[76]],
            'LCC'    : [[77, 78]],
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 13
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNT(kernel) and useLDSTr and TLDS==0:
        nglshift = nllshift = 0
        kernel["SwapGlobalReadOrder"] = True
        optSchedule = {
            'SYNC': [[-1,17,17,57,57]],
            'GRA': [[16,17,20,20,24,24,28,28,31,31]],
            'GRB': [[35,35,39,39,68,68,70,70,71,71,76,76,77,77,78,78]],
            'GRIncA': [[0,0,1,1,2,2,3,3,4]],
            'GRIncB': [[4,5,5,13,13,13,14,14,14]],
            'LCC': [[79,79]],
            'LRA0': [[0,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12]],
            'LRB0': [[0,1,1,2,2,3,3,4,4,5]],
            'LRA1': [[59,61,63,65,67,71,72,72,73,73,74,74,75,75,76,76]],
            'LRB1': [[58,60,62,64,66,67,67,68,68,69]],
            'LRSA': [[38]],
            'LRSB': [[38]],
            'LWSA': [[61]],
            'LWSB': [[61]]
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=7, vscnt=-1, comment="wait for previous set of global reads"),
            SBarrier(comment="")
        ]
        nglshift = nllshift = 13
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 240, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 2, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[4, 1]
)
def _get_schedule_256x240x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0
    if isTN(kernel) and TLDS==1:
        optSchedule = {
            'GRIncA': [[0, 0, 1, 1, 2, 2, 3, 3, 4]],
            'GRIncB': [[30, 30, 31, 31, 32, 32, 33, 33, 34]],
            'LRA0': [[0, 1, 1, 2]],
            'LRB0': [[3, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]],
            'GRA': [[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]],
            'GRB': [[35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 89, 90, 90, 91, 91]],
            'LRA1': [[93, 94, 95, 96]],
            'LRB1': [[97, 98, 99, 100, 102, 104, 106, 108, 110, 112, 114, 116, 116, 116, 116]],
            'LRSA': [[59]],
            'LRSB': [[59]],
            'LWSA': [[91]],
            'LWSB': [[91]],
            'LCC': [[119, 119]],
            'SYNC': [[-1, -1, 4, 4, 33, 33, 92, 92]],
        }
        nglshift = 38
        nllshift = 38
        syncCode = [
            SBarrier(comment="wavefront sync at loop start"),
            SWaitCnt(dscnt=13, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LR/LW"),
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="wait for LRA0 to complete before GRA DirectToLds"),
            SBarrier(comment="barrier after LRA0 (idx 3), before GRA starts (idx 5)"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0 to complete before GRB DirectToLds"),
            SBarrier(comment="barrier after LRB0 (idx 29), before GRB starts (idx 35)"),
            SWaitCnt(dscnt=-1, vlcnt=38, vscnt=-1, comment="wait for global reads before using data"),
            SBarrier(comment="earlier final barrier to reduce idle time"),
        ]
    elif isNT(kernel) and TLDS==0 and useLDSTr:
        optSchedule = {
            'LRA0': [[0, 1, 1, 2, 2, 3, 3, 4]],
            'LRB0': [[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]],
            'LRA1': [[98, 99, 99, 100, 100, 101, 101, 102]],
            'LRB1': [[98, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119]],
            'GRA': [[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]],
            'GRB': [[40, 41, 43, 44, 46, 47, 49, 50, 52, 53, 55, 56, 58, 59, 61, 62, 64, 65, 67, 68, 70, 71, 73, 74, 76, 77, 79, 80, 82, 83, 85, 86, 88, 89, 91, 92, 94, 95, 97, 98, 100, 101, 103, 104, 106, 107, 109, 110, 112, 113, 115, 116, 118, 119, 119, 119, 119, 119, 119, 119]],
            'GRIncA': [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
            'GRIncB': [[3, 3, 3, 4, 4, 4, 5, 5, 5]],
            'LCC': [[119, 119]],
            'LRSA': [[57]],
            'LRSB': [[57]],
            'LWSA': [[96]],
            'LWSB': [[96]],
            'SYNC': [[-1, 6, 6, 38, 38, 96, 96]],
        }
        nglshift = 38
        nllshift = 38
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LR/LW"),
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="wait for LRA0 to complete before GRA DirectToLds"),
            SBarrier(comment="barrier after LRA0 (idx 4), before GRA starts (idx 8)"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0 to complete before GRB DirectToLds"),
            SBarrier(comment="barrier after LRB0 (idx 33), before GRB starts (idx 40)"),
            SWaitCnt(dscnt=-1, vlcnt=27, vscnt=-1, comment="wait for 54 global reads before idx 96 (16 GRA + 38 GRB). vlcnt = 38 - 11 = 27"),
            SBarrier(comment="barrier at idx 96 - before LRA1/LRB1 start at 98"),
        ]
    elif isNN(kernel) and TLDS==1 and useLDSTr:
        optSchedule = {
                'GRIncA': [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
                'GRIncB': [[3, 3, 3, 4, 4, 4, 5, 5, 5]],
                'LRA0': [[0, 1, 1, 2, 2, 3, 3, 4]],
                'LRB0': [[1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]],
                'GRA': [[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]],
                'GRB': [[26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45, 47, 48, 50, 51, 53, 54, 56, 57, 59, 60, 62, 63, 65, 66, 68, 69, 71, 72, 74, 75, 77, 78, 80, 81, 83, 84, 86, 87, 89, 90, 92, 93, 95, 96, 98, 99, 101, 102, 104, 105, 107, 108, 110, 111, 113, 114]],
                'LRA1': [[93, 95, 95, 96, 96, 97, 97, 98]],
                'LRB1': [[94, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]],
                'LRSA': [[57]],
                'LRSB': [[57]],
                'LWSA': [[91]],
                'LWSB': [[91]],
                'LCC': [[119, 119]],
                'SYNC': [[-1, 6, 6, 26, 26, 90, 90]],
            }
        nglshift = 38
        nllshift = 38
        syncCode = [
            SWaitCnt(dscnt=13, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LR/LW"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRA0 to complete before GRA DirectToLds"),
            SBarrier(comment="barrier after LRA0 (idx 4), before GRA starts (idx 8)"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0 to complete before GRB DirectToLds"),
            SBarrier(comment="barrier after LRB0 (idx 19), before GRB starts (idx 26)"),
            SWaitCnt(dscnt=-1, vlcnt=30, vscnt=-1, comment="wait for 59 global reads before idx 90 (16 GRA + 43 GRB). vlcnt = 38 - 8 = 30"),
            SBarrier(comment="barrier at idx 90 - before LRA1/LRB1 start at 93/94"),
        ]
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 120  # Must match actual MFMA count for 256x240x64 tile
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 208, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 2, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[4, 1]
)
def _get_schedule_256x208x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0
    if isTN(kernel) and TLDS==1:
        optSchedule = {
            'SYNC': [[-1, 3, 23, 23, 35, 35, 81, 81]],
            'LRA0': [[0, 1, 2, 4]],
            'LRB0': [[5, 6, 8, 9, 11, 14, 16, 19, 21, 24, 26, 29, 31]],
            'GRIncA': [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
            'GRIncB': [[3, 3, 3, 4, 4, 4, 5, 5, 5]],
            'GRA': [[23, 23, 23, 23, 25, 25, 28, 28, 29, 29, 31, 31, 33, 33, 35, 35]],
            'GRB': [[36, 36, 37, 37, 39, 39, 41, 41, 42, 42, 44, 44, 47, 47, 48, 48, 50, 50, 52, 52, 54, 54, 55, 55, 57, 57, 59, 59, 60, 60, 62, 62, 64, 64, 66, 66, 67, 67, 69, 69, 71, 71, 73, 73, 74, 74, 76, 76, 78, 78, 79, 79]],
            'LRSA': [[50]],
            'LRSB': [[50]],
            'LWSA': [[79]],
            'LWSB': [[80]],
            'LRA1': [[82, 84, 87, 90]],
            'LRB1': [[83, 85, 86, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98]],
            'LCC': [[98, 98]],
        }

        syncCode = [
            SWaitCnt(dscnt=8, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LRB1-4"),
            SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LRB1-remaining"),
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="wait for LRA0 to complete before GRA DirectToLds"),
            SBarrier(comment="barrier after LRA0, before GRA starts"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0 to complete before GRB DirectToLds"),
            SBarrier(comment="barrier after LRB0, before GRB starts"),
            SWaitCnt(dscnt=-1, vlcnt=34, vscnt=-1, comment="wait for all GRA/GRB before next-tile LDS reads"),
            SBarrier(comment="final barrier before LRA1/LRB1"),
        ]

        nglshift = nllshift = 34

    elif isNN(kernel) and useLDSTr and TLDS==1:
        kernel["SwapGlobalReadOrder"] = True
        nglshift = nllshift = 0

        optSchedule = {
            # last index of producer <SYNC> first index of consumer
            # SYNC[0] = -1 to align all waves at the start of the loop
            # A fence at 23, annd B fence at 36, final vmem fence at 81
            'SYNC': [[-1, 3, 7, 11, 23, 36, 36, 81, 81]],

            # Avoid interleaving of LRA0 and LRB0
            # LRA0: tightly packed at the beginning
            'LRA0': [[0, 1, 2, 3, 4, 5, 6, 7]],
            # LRB0 scheduled after the A fence, overlapping with GRA/GRB
            'LRB0': [[24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 35]],

            # Address increments for GR
            'GRIncA': [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
            'GRIncB': [[3, 3, 3, 4, 4, 4, 5, 5, 5]],

            # first stream (A side in NN+swapped)
            'GRB': [[23, 23, 24, 24,
                    26, 26, 28, 28,
                    29, 29, 31, 31,
                    33, 33, 35, 35]],

            # second stream (B side in NN+swapped)
            'GRA': [[36, 36, 38, 38, 40, 40, 41, 41,
                    43, 43, 45, 45, 47, 47, 48, 48,
                    50, 50, 52, 52, 54, 54, 55, 55,
                    57, 57, 59, 59, 60, 60, 62, 62,
                    64, 64, 66, 66, 67, 67, 69, 69,
                    71, 71, 73, 73, 74, 74, 76, 76,
                    78, 78, 79, 79]],

            # from epilogue in the default schedule
            # these are not updated in the updated schedule
            'LRSA': [[50]],
            'LRSB': [[50]],
            'LWSA': [[80]],
            'LWSB': [[80]],
            'LRA1': [[82, 84, 84, 85, 85, 86, 86, 87]], # 8
            'LRB1': [[83, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]], # 13
            'LCC':  [[100, 100]],
        }

        syncCode = [
            SWaitCnt(dscnt=12, vlcnt=-1, vscnt=-1, comment="wait for prior iteration LR/LW"),
            SWaitCnt(dscnt=11, vlcnt=-1, vscnt=-1, comment="ensure all previous LRA1/LRB1 done before early MFMA use"),
            SWaitCnt(dscnt=10, vlcnt=-1, vscnt=-1, comment="ensure all previous LRA1/LRB1 done before early MFMA use"),


            # A fence: all LRA0 are done before DTL writes from the first GR stream startign at 23 (swapped)
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRA0 to complete before first DTL writes"),
            # barrier after LRA0, before first global DTL phase at 23
            SBarrier(comment="barrier after LRA0 , before GR at 23"),

            # B fence : all LRB0 are done before second DTL stream
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0 to complete before second DTL writes"),
            # barrier after LRB0 before second global stream at 36
            SBarrier(comment="barrier after LRB0, before GR at 36"),

            # final vmem fence: ensure all GRA/GRB are done before next tile LDS reads LRA1/LRB1
            SWaitCnt(dscnt=-1, vlcnt=34, vscnt=-1, comment="wait for all GRA/GRB before next-tile LDS reads"),
            # final barrier : all waves; make next-tile LDS visible to LRA1/LRB1
            SBarrier(comment="final barrier before LRA1/LRB1 (at 83)"),
        ]

        nglshift = nllshift = 34
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 104
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
   tile_config=TileConfig(192, 128, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_192x128x64_16bit(kernel, useLDSTr, TLDS):
    """192x128x64 TN schedule (BF16/FP16)."""
    kernel["MfmaInitCVgprs"] = True

    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll

    # 192 = 16 * MIWaveTileA(6) * MIWaveGroup0(2)
    # 128 = 16 * MIWaveTileB(4) * MIWaveGroup1(2)
    if isTN(kernel) and (not useLDSTr) and TLDS == 1:
        numMfma = 2 * kernel["MIWaveTileA"] * kernel["MIWaveTileB"]
        # Number of global reads per iter (A:6, B:4) = 10
        nglshift = nllshift = 10

        # Use syncTable format (idx, wait/barrier, idx, wait/barrier, ...)
        syncTable = [
            # Loop start: must guarantee prior-iteration LR1 completion for the next iteration's early MFMA use.
            -1, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 before starting main loop"),
            5,  SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 for the remaining main loop"),

            14,  SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete to start GRA"),
            14,  SBarrier(comment=""),

            23, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete to start GRB"),
            23, SBarrier(comment=""),

            # GR -> SWait(vmcnt=0) -> SBarrier -> LR1 ordering (global-read to LDS must be visible before LR1)
            28, SWaitCnt(dscnt=-1, vlcnt=10+1, vscnt=-1, comment="Wait for all GRs to complete before starting LR1"),
            28, SBarrier(comment=""),

            44, SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="Wait for all GRs to complete before starting LRB1"),
            44, SBarrier(comment=""),
        ]

        optSchedule = {
            'SYNC'   : [syncTable[::2]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[9,9,10,10,11,11,12,13,13]],

            'LRA0'   : [[0,1,2,3,4,5]],
            'LRB0'   : [[6,9,12,15]],

            'GRA'    : [[14,14, 16,16, 18,18, 20,20, 22,22, 24,24],
                        [15,15, 17,17, 19,19, 21,21, 23,23, 25,25]],
            'GRB'    : [[26,26, 30,30, 37,37, 43,43]],

            # Prefetch next iteration
            'LRA1'   : [[28,30,32,34,36,38],
                        [29,31,33,35,37,39]],
            'LRB1'   : [[44,45,46,47]],

            'LRSA'   : [[21]],
            'LRSB'   : [[21]],
            'LWSA'   : [[46]],
            'LWSB'   : [[46]],
            'LCC'    : [[47, 47]],
        }

        syncCode = syncTable[1::2]
    else:
        return False, None

    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(224, 128, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_224x128x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    numMfma = 56
    numCodePaths = 1

    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and not useLDSTr and TLDS==1:
        nglshift = nllshift = 11 # vmcnt shift for ngl and nll
        optSchedule = {
            'SYNC': [[-1, 6, 14, 14, 27,27, 47, 47]],
            'LRA0': [[0,1, 2,3,4,5,5]],
            'GRIncA': [[0, 0, 1, 1, 2,2 , 3,3, 4]],
            'LRB0': [[9, 11,13, 19]],
            'GRIncB': [[ 6,6,7,7,8,8,9,9,10]],
            'GRA': [[14, 14, 16,16,18,18,20,20,23,23, 26,26, 27, 27]],
            'LRSA': [[26]],
            'LRSB': [[26]],
            'GRB': [[33,34,36,38,38,42,42,46]],
            'LWSA': [[54]],
            'LWSB': [[54]],
            'LRA1': [[30,35,44, 45, 46, 48,51]],
            'LRB1': [[47,52,54,55]],
            'LCC': [[55, 55]],
        }

        syncCode = [
            SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="wait for all of LRA1 and the first instance of LRB1"),
            SWaitCnt(dscnt=8, vlcnt=-1, vscnt=-1, comment="wait for the second instance of LRB1"),
            SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="wait for all LRA0 to complete so GRA could begin. Makes sure LRB1 is completed so no need for a barrier at 21"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=10, vscnt=-1, comment="wait for all LR. All of previous GRB (4) and current GRA (6), total of 10 can be outstanding"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="Outstanding LR are all LRA so no need to wait. All of GR from previous iteration must be done."),
            SBarrier(comment="")
        ]
    elif isNN(kernel) and useLDSTr and TLDS==1:
        optSchedule = {

            'SYNC'   : [[-1,3, 16,16, 27, 35,35, 48,48]],
            'GRIncA' : [[1,2,3,4,4,5,5,6,6]],
            'GRIncB' : [[7,7,8,8,9,9,10,11,11]],

            'LRA0'   : [[0,1,1,2,2,3,4,5,6,7,8,9,10,11]],
            'LRB0'   : [[17,18,19,20]],   ## After LRA0, we can mix LRB0 and GRA

            ## GRA should start after LRA0 is done.
            'GRA'    : [[15,16, 19,19, 22,22, 25,25, 28,28, 31,31, 34,34]],
            ## GRB should start after LRB0 is done.
            'GRB'    : [[42,42, 45,45, 48,48, 51,51]],

            #After previous GRA is done.
            'LRA1'   : [[35,36,37,38,39,40,41,42,43,44,45,46,47,48]],
            #After previous GRB is done.
            'LRB1'   : [[49,50, 52,53]],

            'LRSA'   : [[24]], # after LRA0 and before LRA1
            'LRSB'   : [[24]], # after LRB0 and before LRB2
            'LWSA'   : [[53]], # For A
            'LWSB'   : [[54]],

            'LCC'    : [[54, 55]],
        }
        # note: syncCode needs to be
        syncCode = [SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for necessary prior LRA1/LRB1 before starting main loop"),
                    SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 for the remaining main loop"),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete to start GRA"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
                    SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="Wait for previous GRA to complete to start LRA1"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=9, vscnt=-1, comment="Wait for previous GRB to complete to start LRB1"),
                    SBarrier(comment=""),
                   ]
        nglshift = nllshift = 11 # vmcnt shift for ngl and nll
    elif isNT(kernel) and useLDSTr and TLDS == 0:
        # Global read scheduling:
        # Each GR has two instructions (addr update + buffer_load), so we list them explicitly as
        # two adjacent MFMA indices per GR.
        kernel["SwapGlobalReadOrder"] = True
        numCodePaths = 2

        syncTable = [
            # Loop start:
            # - LRB1 waits previous-iter GRB
            # - LRA1 waits previous-iter GRA
            -1, SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 before starting main loop"),

            # After early MFMAs (keep prior-iter LR fully fenced)
            3,  SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 for the remaining main loop"),

            # GRB must wait for LRB0 (interleave LRA0 + GRB safely)
            15, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete to start GRB"),
            15, SBarrier(comment=""),

            # GRA must wait for LRA0; LRB1 can be interleaved with GRA after this fence
            27, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0/GRB to complete to start GRA/LRB1"),
            27, SBarrier(comment=""),

            # Mid-loop global-read safety fence (GR-to-LDS)
            30, SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="Mid-loop fence (wait for outstanding GR-to-LDS)"),
            30, SBarrier(comment=""),

            # Ensure all GR-to-LDS are complete before LRA1 (next-iter A LDS reads)
            43, SWaitCnt(dscnt=-1, vlcnt=11-2, vscnt=-1, comment="Wait for all GR-to-LDS to complete before LRA1"),
            43, SBarrier(comment=""),
        ]

        optSchedule = {
            'SYNC'   : [syncTable[::2]],
            # Swap A/B increments
            'GRIncB' : [[0, 0, 1, 1, 2, 2, 4, 4, 5]],
            'GRIncA' : [[5, 6, 6, 7, 7, 8, 8, 9, 9]],

            'LRB0'   : [[0, 1, 2, 3, 4, 5, 6, 7],
                        [1, 2, 3, 4, 5, 6, 7, 8]],
            'LRA0'   : [[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]],

            'GRA'    : [[14,15, 17,18, 20,21, 23,24],
                        [15,16, 18,19, 21,22, 24,25]],
            'GRB'    : [[28,29, 31,32, 34,35, 37,38, 40,41, 43,44, 46,47],
                        [29,30, 32,33, 35,36, 38,39, 41,42, 44,45, 45,46]],

            'LRB1'   : [[30, 31, 32, 33, 34, 35, 36, 37],
                        [31, 32, 33, 34, 35, 36, 37, 38]],
            'LRA1'   : [[43, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]],

            'LRSA'   : [[24]],
            'LRSB'   : [[25]],
            'LWSA'   : [[48]],
            'LWSB'   : [[49]],
            'LCC'    : [[53, 54]],
        }

        syncCode = syncTable[1::2]
        nglshift = nllshift = 11
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(numCodePaths, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(224, 256, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_224x256x64_16bit(kernel, useLDSTr, TLDS):
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    optSchedule = dict()
    syncCode = []
    if isTN(kernel) and TLDS==1:
        optSchedule = {
            'SYNC'   : [[ -1,  18,  18,  51,  51,  90,  90]],
            'GRIncA' : [[  1,   1,   3,   3,   5,   5,   7,   7,   9],
                        [  0,   0,   2,   2,   4,   4,   6,   6,   8]],
            'GRIncB' : [[  9,  11,  11,  13,  13,  15,  15,  17,  17],
                        [  8,  10,  10,  12,  12,  14,  14,  16,  16]],
            'LRA0'   : [[  0,   2,   4,   6,   8,  10,  12],
                        [  1,   3,   5,   7,   9,  11,  13]],

            'LRB0'   : [[ 14,  16,      19,  21,  23,  25,  27,      29],
                        [ 15,  17,      20,  22,  24,  26,  28,      30]],
            'GRA'    : [[ 19,  20,  21,  22,  23,  24,  25,  26,  27,  28,     46,  47,  48,  49],
                        [ 20,  21,  22,  23,  24,  25,  26,  27,  28,  29,     47,  48,  49,  50]],

            'LRA1'   : [[ 56,  58,       77,  79,  81,  83,  85],
                        [ 57,  59,       78,  80,  82,  84,  86]],
            'GRB'    : [[ 52,  53,  54,  55,  56,  57,      77,  78,  79,  80,  81,  82,  83,  84,  85,  86],
                        [ 53,  54,  55,  56,  57,  58,      78,  79,  80,  81,  82,  83,  84,  85,  86,  87]],

            'LRB1'   : [[ 91,  93,  95,  97,  99, 101, 103, 105],
                        [ 92,  94,  96,  98, 100, 102, 104, 106]],
            'LRSA'   : [[ 50], [52]],
            'LRSB'   : [[ 50], [52]],
            'LWSA'   : [[108]],
            'LWSB'   : [[109]],
            'LCC'    : [[110, 111]]
        }
        syncCode = [
            SWaitCnt(dscnt= 0, vlcnt=-1, vscnt=-1, comment="Wait for LRBs"),
            SWaitCnt(dscnt= 2, vlcnt=-1, vscnt=-1, comment="Wait for LRAs"),
            SBarrier(comment=""),
            SWaitCnt(dscnt= 0, vlcnt=15, vscnt=-1, comment="Wait for LRBs and previous set of GRs"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=15, vscnt=-1, comment="Wait for previous set of GRs"),
            SBarrier(comment=""),
        ]
        nglshift = nllshift = 15
    elif isNT(kernel) and useLDSTr and TLDS == 0:
        optSchedule = {
            'SYNC'   : [[-1, 21, 21, 51, 51, 79, 79]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[9,10,11,12,13,14,15,16,17]],

            'LRA0'   : [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]],
            'LRB0'   : [[14, 17, 20, 23, 26, 29, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]],

            'GRA'    : [[22,22, 26,26, 30,30, 34,34, 38,38, 42,42, 46,46]],
            'GRB'    : [[52,53, 56,57, 60,61, 64,65, 68,69, 71,72, 74,75, 77,78]],

            'LRA1'   : [[79,80, 81,82, 83,84, 85,86, 87,88, 89,90, 91,92]],
            'LRB1'   : [[93,94, 95,96, 97,98, 99,100, 101,102, 103,104, 105,106, 107,108]],
            'LRSA'   : [[54]],
            'LRSB'   : [[54]],
            'LWSA'   : [[91]],
            'LWSB'   : [[91]],
            'LCC'    : [[111, 111]]
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=0 newLW=0 newLR=0"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=15, vscnt=-1, comment="wait for previous set of global reads"),
            SBarrier(comment=""),
        ]
        nglshift = nllshift = 15
    elif isNN(kernel) and useLDSTr and TLDS == 1:
        optSchedule = {
            'SYNC': [[  -1,6,
                        23,23,
                        55,55,
                        94,
                    ]],
            'GRIncA': [[0,1,2,3,4,5,6,7,8]],
            'GRIncB': [[9,10,11,12,13,14,15,16,17]],
            'LRA0': [[0,1,2,3,4,5,6,7,8,9,10,11,12,13]],
            'GRA': [[23,23,27,27,32,32,37,37,42,42,46,46,51,51],
                    [25,25,29,29,34,34,39,39,44,44,48,48,53,53]],
            'LRB0': [[23,27,32,37,42,46,51,54],
                     [25,29,34,39,44,48,53,54]],
            
            'LWSA': [[87],[86]],
            'LWSB': [[90],[89]],
            'LRA1': [[57,57,61,61,65,65,75,75,81,81,87,90,96,96],
                     [55,55,59,59,63,63,69,69,79,79,86,89,94,94]],                     
            'GRB': [[56,56, 60,60, 70,70, 80,80, 90,92, 100,100, 106,106, 109,109],
                    [58,58, 62,62, 72,72, 82,82, 91,93, 101,101, 107,107, 110,110]],                    
            'LRB1': [[91,92,97,100,102,103,106,109]],
            'LRSA': [[54]],
            'LRSB': [[54]],
            'LCC': [[109,110]],
        }

        syncCode = [
            SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="wait for 8-5 local reads // oldleft=8, completed=3"),
            SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="wait for 11-6 local reads // oldleft=5, new=6, completed=5"),
            SWaitCnt(dscnt=0, vlcnt=8, vscnt=-1,  comment="wait for prior global reads and local reads // oldleft=6, new=8, completed=14"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=7, vscnt=-1,  comment="wait for prior global reads and local reads // oldleft=0, new=8, completed=8"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="wait for 14-5 local reads // oldleft=0, new=14, completed=9"),
        ]

        nglshift = nllshift = 15    
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 112
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(192, 320, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_192x320x64_16bit(kernel, useLDSTr, TLDS):
    numMfma = 120
    nllZeroDscnt = False
    syncs = SyncSchedule()
    gr_inc_step = 0

    if isNN(kernel) and useLDSTr and TLDS==1:
        syncs.add(-1, dscnt=9, comment="wait for all LRA1 and one item from LRB1 before starting the sub-iteration")
        lra0   = [0,1,2,3,4,6,8,10,12,14,16,18]

        syncs.add(5, dscnt=5, comment="wait for the rest of LRB1 to complete")
        grinca = [7,7,7,9,9,9,11,11,11]
        grincb = [13,13,13,15,15,15,17,17,17]
        lrb0   = [20,22,24,25,27,29,31,33,35,37]

        syncs.add(26, dscnt=4, barrier=True, comment="wait for all LRA0 to complete before GRA start")
        gra    = [28,30,32,36,39,42] # one index for two instructions
        syncs.add(44, dscnt=0, barrier=True, comment="wait for all LRB0 to complete before GRB start")
        grb    = [53,58,63,67,72,77,82,86,91,96] # one index for two instructions
        num_gr = len(gra) + len(grb)

        lrsa   = [58]
        lrsb   = [58]

        syncs.add(71, vlcnt=10, barrier=True, comment="wait for previous set of global reads")
        lra1   = [72,74,76,78,80,82,84,87,90,92,98,100]
        lrb1   = [99,106,107,108,109,110,111,112,113,114]
        lwsa   = [95]
        lwsb   = [95]

    elif isTN(kernel) and not useLDSTr and TLDS==1:
        syncs.add(-1, dscnt=9, comment="wait for all LRA1 and one item from LRB1 before starting the sub-iteration")
        lra0   = [0,1,2,3,4,6]

        syncs.add(5, dscnt=5, comment="wait for the rest of LRB1 to complete")
        grinca = [0,1,2,3,4,5,6,7,8]
        grincb = [9,10,12,13,14,15,16,17,18]
        lrb0   = [8,10,12,14,16,18,20,22,24,27]

        syncs.add(26, dscnt=9, barrier=True, comment="wait for all LRA0 to complete before GRA start")
        gra    = [28,30,32,36,39,42] # one index for two instructions
        syncs.add(44, dscnt=0, barrier=True, comment="wait for all LRB0 to complete before GRB start")
        grb    = [53,58,63,67,72,77,82,86,91,96] # one index for two instructions
        num_gr = len(gra) + len(grb)

        lrsa   = [58]
        lrsb   = [58]
        syncs.add(71, vlcnt=10, barrier=True, comment="wait for previous set of global reads")

        lra1   = [72,76,80,84,90,98]
        lrb1   = [99,106,107,108,109,110,111,112,113,114]
        lwsa   = [95]
        lwsb   = [95]

        gr_inc_step = 1

    elif isNT(kernel) and useLDSTr and TLDS == 0:
        lra0   = [0,1,3,5,7, 9,10,12,14,16, 18,19] # 12 loads
        lrb0   = [21,23,25,27,28, 30,32,34,36,37, 39,41,43,45,46, 48,50,52,54,55] # 20 loads
        # need two LRB1 items because a single LRB read gets only half of the data needed for MFMA
        # note, max dscnt value is 15, so in this case 19 will be maxed at 15, thus we will wait more than needed :(
        syncs.add(-1, dscnt=len(lrb0)-2, comment="wait for all LRA1 and two items from LRB1 before starting the sub-iteration")

        i = 5 # next LRB1 is needed at index 6, so insert wait at 5
        syncs.add(i, dscnt=count_items(lra0,ev=i), comment="wait for the rest of LRB1 to complete")
        grinca = [0,1,2,3,4,5,6,7,8]
        grincb = [9,10,12,13,14,15,16,17,18]

        i = 26
        syncs.add(i, dscnt=count_items(lrb0,ev=i), barrier=True, comment="wait for all LRA0 to complete before GRA start")
        gra    = [28,30,32,36,39,42] # one index for two instructions

        lrsa   = [57]
        lrsb   = [58]

        # This wait serves dual purpose
        syncs.add(59, dscnt=len(lrb0)-2, vlcnt=len(gra), barrier=True,
                  comment="wait for the first LRB0 to complete to start 2nd batch of MFMAs and also make GRs from the previous iteration is done before LRA1 starts")
        lra1   = [61,62,63,64,65, 66,67,69,71,73, 75,76] # 12 loads

        i = 65 # next LRB0 is needed at index 66, so insert wait at 65
        syncs.add(i, dscnt=count_items(lra1,ev=i), barrier=True,
                  comment="wait for the rest of LRB0 to complete across all waves before GRB start")
        grb    = [66,70,75,79,84, 88,93,97,102,106] # one index for two instructions
        num_gr = len(gra) + len(grb)

        lwsa   = [68]
        lwsb   = [77]
        lrb1   = [78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116] # 20 loads

        gr_inc_step = 1
        nllZeroDscnt = True
    else:
        return False, None

    optSchedule = {
        'SYNC':   [syncs.get_indicies()],
        'LRA0':   [lra0],
        'GRIncA': [grinca],
        'LRB0':   [lrb0],
        'GRIncB': [grincb],
        # Note: each GRA/GRB item corresponds to two instructions (addr increment and read). So duplicate each item twice.
        'GRA':    [duplicate_list_items(gra, 2, gr_inc_step)],
        'GRB':    [duplicate_list_items(grb, 2, gr_inc_step)],
        'LRSA':   [lrsa],
        'LRSB':   [lrsb],
        'LWSA':   [lwsa],
        'LWSB':   [lwsb],
        'LRA1':   [lra1],
        'LRB1':   [lrb1],
        'LCC':    [[numMfma-2, numMfma-1]],
    }

    kernel["MfmaInitCVgprs"] = True
    kernel["SwapGlobalReadOrder"] = False
    syncCode = syncs.get_code()
    nglshift = nllshift = num_gr
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift, nllZeroDscnt)

    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 224, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x224x64_16bit(kernel, useLDSTr, TLDS):
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    optSchedule = dict()
    syncCode = []
    if isTN(kernel) and TLDS==1:
        optSchedule = {
            'SYNC'   : [[ -1,  18,  18,  51,  51,  90,  90]],
            'GRIncA' : [[  1,   1,   3,   3,   5,   5,   7,   7,   9],
                        [  0,   0,   2,   2,   4,   4,   6,   6,   8]],
            'GRIncB' : [[  9,  11,  11,  13,  13,  15,  15,  17,  17],
                        [  8,  10,  10,  12,  12,  14,  14,  16,  16]],
            'LRA0'   : [[  0,   2,   4,   6,   8,  10,  12,  14],
                        [  1,   3,   5,   7,   9,  11,  13,  15]],
            # schduling GRIncA/B and LRA0 as follow,
            # SIMD 0 | ... | MFMA | GRInc  | GRInc  | MFMA | LDS Load            | MFMA | GRInc  | GRInc  | MFMA | ...
            # SIMD 1 | ... | MFMA | LDS Load        | MFMA | GRInc  | GRInc      | MFMA | LDS Load        | MFMA | ...

            'LRB0'   : [[ 16,      19,  21,  23,  25,  27,      29],
                        [ 17,      20,  22,  24,  26,  28,      30]],
            'GRA'    : [[ 19,  20,  21,  22,  23,  24,  25,  26,  27,  28,     46,  47,  48,  49,  52,  53],
                        [ 20,  21,  22,  23,  24,  25,  26,  27,  28,  29,     47,  48,  49,  50,  53,  54]],

            'LRA1'   : [[ 52,  56,  58,       77,  79,  81,  83,  85],
                        [ 53,  57,  59,       78,  80,  82,  84,  86]],
            'GRB'    : [[ 54,  55,  56,  57,      77,  78,  79,  80,  81,  82,  83,  84,  85,  86],
                        [ 55,  56,  57,  58,      78,  79,  80,  81,  82,  83,  84,  85,  86,  87]],

            'LRB1'   : [[ 91,  93,  95,  97,  99, 101, 103],
                        [ 92,  94,  96,  98, 100, 102, 104]],
            'LRSA'   : [[ 50], [52]],
            'LRSB'   : [[ 50], [52]],
            'LWSA'   : [[108]],
            'LWSB'   : [[109]],
            'LCC'    : [[110, 111]]
        }
        syncCode = [
            SWaitCnt(dscnt= 0, vlcnt=-1, vscnt=-1, comment="Wait for LRBs"),
            SWaitCnt(dscnt= 1, vlcnt=-1, vscnt=-1, comment="Wait for LRAs"),
            SBarrier(comment=""),
            SWaitCnt(dscnt= 0, vlcnt=14, vscnt=-1, comment="Wait for LRBs and previous set of GRAs"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=15, vscnt=-1, comment="Wait for previous set of GRBs"),
            SBarrier(comment=""),
        ]
        nglshift = nllshift = 15
    elif isNT(kernel) and useLDSTr and TLDS == 0:
        optSchedule = {
            'SYNC': [[-1,6,
                       21,21,55,55,60,60]],
            'GRIncA': [[0,1,2,3,4,5,6,7,8]],
            'LRA0': [[0,0,2,2,4,4,6,6,8,8,10,10,12,12,14,14],
                     [1,1,3,3,5,5,7,7,9,9,11,11,13,13,15,15]],
            
            'GRIncB': [[9,10,11,12,13,14,15,16,17]],
            
            'GRA': [[21,22, 25,26, 30,31, 35,36, 40,41, 45,46, 50,51, 53,54]],
            'LRB0': [[21,22, 25,26, 30,31, 35,36, 40,41, 45,46, 50,51]],

            'GRB': [[61,61, 63,63, 65,65, 79,79, 85,85, 95,95, 100,100],
                    [62,62, 64,64, 66,66, 80,80, 91,91, 96,96, 101,101]],
            'LWSA': [[93],[99]],
            'LWSB': [[91],[87]],
            'LRA1': [[61,61, 63,63, 65,65, 79,79, 85,85, 93,93, 100,100, 104,104],
                    [62,62, 64,64, 66,66, 80,80, 91,91, 96,96, 101,101, 106,106]],
            'LRB1': [[91,91,95,95,98,98,100,100,110,110,110,111,111,111],
                     [87,87,94,94,99,99,101,101,103,103,105,105,107,107]],

            'LRSA': [[54]],
            'LRSB': [[54]],
            'LCC': [[110,111]],
        }

        syncCode = [
            SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="wait for prior local read"),
            SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="wait for prior local read"),
            SWaitCnt(dscnt=0, vlcnt=7, vscnt=-1, comment="wait for previous set of global reads and Local Reads"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior local read"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=8, vscnt=-1, comment="wait for previous set of global reads"),
            SBarrier(comment="")
        ]
        nglshift = nllshift = 15

    elif isNN(kernel) and useLDSTr and TLDS == 1:
        kernel["SwapGlobalReadOrder"] = True
        #index and code pair
        syncTable = [-1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRA1"),
                     17, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="wait for LRB0"),
                     17, SBarrier(comment=""),
                    #  46, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
                     51, SWaitCnt(dscnt=0, vlcnt=15, vscnt=-1, comment="wait for previous set of global reads"),
                     51, SBarrier(comment=""),
                     65, SWaitCnt(dscnt=-1, vlcnt=15-4, vscnt=-1, comment="wait for previous set of global reads"),
                     65, SBarrier(comment=""),
                    ]
        optSchedule = {
                    'SYNC'  : [syncTable[::2]],
                    'GRIncA': [[37,38,39,40,41,42,43,44,45]],
                    'GRIncB': [[0,1,2,3,4,5,6,7,8]],

                    'LRB0': [[-1, 0, 1, 2, 3, 4, 5],
                             [ 0, 1, 2, 3, 4, 5, 6]],
                    'LRA0': [[8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 27, 29, 31, 33, 35, 37],
                             [9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 28, 30, 32, 34, 36, 38]],
                    'GRA': [[17,17, 19,19, 26,26, 28,28, 30,30, 32,32, 34,34],
                            [18,18, 20,20, 27,27, 29,29, 31,31, 33,33, 35,35]],

                    'GRB': [[52,52, 54,54, 56,56, 58,58, 66,66, 68,68, 70,70, 72,72],
                            [53,53, 55,55, 57,57, 59,59, 67,67, 69,69, 71,71, 73,73]],
                    'LRB1': [[51, 53, 55, 57, 59, 61, 63],
                             [52, 54, 56, 58, 60, 62, 64]],
                    'LRA1': [[65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95],
                             [66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96]],

                    'LRSB': [[14]],
                    'LRSA': [[45]],
                    'LWSB': [[97]],
                    'LWSA': [[97]],
                    'LCC' : [[110, 110]],
                }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 15 # vmcnt shift for ngl and nll

    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 112
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(320, 192, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_320x192x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll

    if isNN(kernel) and useLDSTr and TLDS == 1:
        kernel["SwapGlobalReadOrder"] = True
        syncTable = [
            -1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRA1 "),
            19, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="before DirectToLds load, ensure LRB0 have finished"),
            19, SBarrier(comment=""),
            52, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRA0 finish"),
            52, SBarrier(comment=""),
            53, SWaitCnt(dscnt=-1, vlcnt=16+1, vscnt=-1, comment="wait for previous GRB finish"),
            53, SBarrier(comment=""),
            71, SWaitCnt(dscnt=-1, vlcnt=16-8, vscnt=-1, comment="wait for previous GRA finish"),
            71, SBarrier(comment=""),
        ]

        optSchedule = {
            'SYNC': [syncTable[::2]],
            'GRIncA': [[0,1,2,3,4,5,6,7,8]],
            'GRIncB': [[9,10,11,12,13,14,16,16,16]],

            'LRB0': [[0, 1, 2, 3, 4, 5],
                     [1, 2, 3, 4, 5, 6]],
            'LRA0': [[6, 8, 10, 12, 14, 16, 18,  20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44],
                     [7, 8, 11, 13, 15, 17, 18,  21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45]],
            'GRA': [[19,19, 21,21, 28,28, 31,31, 33,33, 41,41],
                    [20,20, 22,22, 29,29, 32,32, 34,34, 42,42]],
            'GRB': [[52,52, 64,64, 74,74, 78,78, 83,83, 88,88, 93,93, 98,98, 105,105, 109,109],
                    [52,52, 65,65, 75,75, 79,79, 84,84, 89,89, 94,94, 99,99, 106,106, 110,110]],
            'LRB1': [[53, 55, 57, 59, 63, 67],
                     [54, 56, 58, 60, 64, 68]],
            'LRA1': [[71,73, 75,77, 79,81, 83,85, 87,89, 91,93, 95,97, 99,101, 103,105, 107,109],
                     [72,74, 76,78, 80,82, 84,86, 88,90, 92,94, 96,98, 100,102, 104,106, 108,110]],
            'LRSB': [[16]],
            'LRSA': [[48]],
            'LWSB': [[48]],
            'LWSA': [[112]],
            'LCC': [[119, 119]],
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 16
    elif isTN(kernel) and TLDS==1:
        kernel["SwapGlobalReadOrder"] = True
        # Note: A/B Global read orders are swapped
        # i.e. GRA contains GR for B
        optSchedule = {
            'SYNC'  : [[-1, 4, 16, 16, 50, 50, 54, 55, 84, 85]],
            'GRIncA': [[0,  1,  2,  3,  4,  5,  6,  7,  8]],
            'GRIncB': [[9, 10, 11, 12, 13, 14, 16, 16, 16]],
            'LRB0'  : [[0, 2, 4, 6, 8, 10],
                       [1, 3, 5, 7, 9, 11]],
            'LRA0'  : [[12, 14, 24, 26, 28, 30, 32, 34, 36, 38],
                       [13, 15, 25, 27, 29, 31, 33, 35, 37, 39]],
            'GRA'   : [[16, 16, 18, 18, 20, 20, 22, 22, 46, 46, 48, 48],
                       [17, 17, 19, 19, 21, 21, 23, 23, 47, 47, 49, 49]],
            'GRB'   : [[50, 50, 52, 52, 74, 75, 78, 78, 80, 80, 82, 82, 106, 106, 108, 108, 110, 110, 112, 112],
                       [51, 51, 53, 53, 76, 77, 79, 79, 81, 81, 83, 83, 107, 107, 109, 109, 111, 111, 113, 113]],
            'LRB1'  : [[56, 58, 60, 62, 64, 66],
                       [57, 59, 61, 63, 65, 67]],
            'LRA1'  : [[86, 88, 90, 92, 94, 96, 98, 100, 102, 104],
                       [87, 89, 91, 93, 95, 97, 99, 101, 103, 105]],
            'LRSA'  : [[44]],
            'LRSB'  : [[45]],
            'LWSA'  : [[115]],
            'LWSB'  : [[117]],
            'LCC'   : [[119, 119]],
        }

        syncCode = [
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for prior local read. Relax a bit to dscnt=4 to reduce latency") ,
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for all prior local read requested as the input for MFMA") ,
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for prior LRB0 before GRA (GR for MatrixB). Skip LRA0*2") ,
            SBarrier(comment="") ,
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA0 before GRB (GR for MatrixA)") ,
            SBarrier(comment="") ,
            SWaitCnt(dscnt=-1, vlcnt=18, vscnt=-1, comment="Wait for GRA (GR for MatrixB) from previous iteration before LRB1. Skip GRB*10 from last iter and GRA*6 + GRB*2 from this iter.") ,
            SBarrier(comment="") ,
            SWaitCnt(dscnt=-1, vlcnt=12, vscnt=-1, comment="Wait for GRB (GR for MatrixA) from previous iteration before LRA1. Skip GRA*6 + GRB*6 from this iter.") ,
            SBarrier(comment="") ,
        ]
        nglshift = nllshift = 16
    elif isNT(kernel) and useLDSTr and TLDS == 0:
        kernel["SwapGlobalReadOrder"] = True
        # Note: A/B Global read orders are swapped
        # i.e. GRA contains GR for B
        optSchedule = {
            'SYNC'  : [[-1, 7, 17, 17, 49, 49, 59, 59]],
            'GRIncA': [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
            'GRIncB': [[3, 3, 3, 4, 4, 4, 5, 5, 5]],
            'LRB0'  : [[0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10],
                       [1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11]],
            'LRA0'  : [[11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 31, 33, 33, 35, 37, 39, 41, 43, 45],
                       [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 36, 38, 38, 40, 42, 44, 46]],
            'GRA'   : [[18, 18, 20, 20, 22, 22, 24, 24, 26, 26, 28, 28],
                       [19, 19, 21, 21, 23, 23, 25, 25, 27, 27, 29, 29]],
            'GRB'   : [[49, 49, 51, 51, 53, 53, 55, 55, 57, 57, 89, 89, 91, 91, 93, 93, 95, 95, 97, 97],
                       [50, 50, 52, 52, 54, 54, 56, 56, 58, 58, 90, 90, 92, 92, 94, 94, 96, 96, 98, 98]],
            'LRB1'  : [[60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82],
                       [61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83]],
            'LRA1'  : [[85, 87, 89, 91, 93, 95, 97,  99, 101, 103, 103, 105, 105, 107, 107, 109, 111, 113, 115, 117],
                       [86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 106, 108, 108, 110, 110, 112, 114, 116, 118],],
            'LRSA'  : [[58]],
            'LRSB'  : [[58]],
            'LWSA'  : [[99]],
            'LWSB'  : [[99]],
            'LCC'   : [[119, 119]],
        }

        syncCode = [
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for prior local read. Relax a bit to dscnt=4 to reduce latency") ,
            SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for remaining LRA1.") ,
            SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for all LRB0 prior to  LRA0*3") ,
            SBarrier(comment="") ,
            SWaitCnt(dscnt=0,  vlcnt=-1, vscnt=-1, comment="Wait for prior local read") ,
            SBarrier(comment="") ,
            SWaitCnt(dscnt=-1, vlcnt=11, vscnt=-1, comment="Wait for prior GRA*6 + GRB*5 = 11 global reads") ,
            SBarrier(comment="") ,
        ]
        nglshift = nllshift = 16
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 120
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(352, 192, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_352x192x64_16bit(kernel, useLDSTr, TLDS):
    numMfma = 132
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    
    if isTN(kernel) and TLDS==1:
        syncTable = [
            -1, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=5 newLW=0 newLR=5 for iteration == 0"),
            14, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="wait for 5 LRA0s to complete"),
            25, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="wait for next 4 LRA0s to complete"),
            
            34, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for all LRA0s to complete"),
            34, SBarrier(comment="Barrier before GRA"),
            
            46, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="wait for 3 LRB0s to complete"),
            
            87, SWaitCnt(dscnt=0, vlcnt=17, vscnt=-1, comment="wait for previous GRA to complete"),
            87, SBarrier(comment="Barrier before LRA1"),
            118, SWaitCnt(dscnt=-1, vlcnt=12, vscnt=-1, comment="wait for previous GRB to complete"),
            118, SBarrier(comment="Barrier before LRB1"),
        ]
        
        optSchedule = {
            'SYNC': [syncTable[::2]], # 6
            
            'GRIncA': [[0, 0, 1, 1, 2, 2, 3, 3, 4]], # 9
            'GRIncB': [[4, 5, 5, 6, 6, 7, 7, 8, 8]], # 9
            
            'LRA0': [[0, 2, 4, 6, 8 , 10, 14, 18, 20, 24, 26],
                     [1, 3, 5, 7, 9, 11, 15, 17, 21, 25, 27]], # 11
            
            'LRB0': [[31, 33, 36, 39, 41, 44],
                     [32, 34, 37, 40, 42, 45]], # 6

            'GRA': [[35, 35, 40, 40, 46, 46, 51, 51, 56, 56, 61, 61, 67, 67, 72, 72, 77, 77, 82, 82, 86, 86]], # 22
            'GRB': [[88, 88, 93, 93, 98, 98, 103, 103, 109, 109, 114, 114]], # 12

            'LRSA': [[64]], # 1
            'LRSB': [[64]], # 1
            
            'LWSA': [[109]], # 1
            'LWSB': [[109]], # 1
            'LRA1': [[87, 90, 92, 94, 96, 98, 100, 102, 104, 106, 114]], # 11
            'LRB1': [[118, 120, 122, 124, 127, 130]], # 6 
 
            'LCC': [[numMfma-3, numMfma-3]], # 2
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = len(optSchedule["GRA"][0])/2 + len(optSchedule["GRB"][0])/2
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
        
@RegisterSchedule(
    tile_config=TileConfig(240, 256, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[2, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[1, 4]
)
def _get_schedule_240x256x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    if isTN(kernel) and TLDS==1:
        kernel["SwapGlobalReadOrder"] = False
        optSchedule = {
            'SYNC': [[-1,
                      14,
                      26,26,
                      59,
                      69,69,
                      98,98]],
            'LRA0': [[0,2,3,4,5,6,8,10,12,14, 16,18,20,22,24]],
            'GRIncA': [[0,0,0,1,1,1,2,2,2]],
            'GRIncB': [[3,3,3,4,4,4,5,5,5]],
            'LRB0': [[28,30,36,38]],
            'GRA': [[26,26,27,27,29,29,31,31,33,33,35,35,37,37,39,39,41,41,42,42,44,44,46,46,48,48,50,50,52,52,54,54,56,56,58,58,59,59,61,61,63,63,65,65,67,67,69,69,71,71,73,73,75,75,76,76,78,78,80,80]],
            'GRB': [[82,82,84,84,86,86,88,88,90,90,92,92,94,94,96,96],
                    [81,81,83,83,85,85,87,87,91,91,93,93,95,95,97,97]],
            'LRSA': [[58]],
            'LRSB': [[58]],
            'LWSA': [[98]],
            'LWSB': [[98]],
            'LRA1': [[70,72,74,76,77,79,80,82,84,86,88,90,92,94,96]],
            'LRB1': [[99,114,115,116]],
            'LCC': [[119, 119]]
        }

        syncCode = [
            SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=3 newLW=0 newLR=3 for iteration == 0"),
            SWaitCnt(dscnt=9, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=3 newLW=0 newLR=3 for iteration == 0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRA0"),
            SBarrier(comment=""),

            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LRB0"),

            SWaitCnt(dscnt=-1, vlcnt=23+8, vscnt=-1, comment="wait for previous set of GRA"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=-1, vlcnt=38, vscnt=-1, comment="wait for previous set of GRB"),
            SBarrier(comment="")
        ]
        numMfma = 120
        nglshift = nllshift = len(optSchedule["GRA"][0])/2 + len(optSchedule["GRB"][0])/2
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNT(kernel) and TLDS==0:
        optSchedule = {
            'SYNC': [[-1,24,24,59,59]],
            'LRA0': [[0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15]],
            'LRB0': [[24,24,26,26,28,28,30,30]],
            'GRA': [[24,24,25,25,27,27,29,29,31,31,33,33,35,35,37,37,39,39,41,41,43,43,45,45,47,47,49,49,50,50,52,52,54,54,56,56,58,58,60,60,61,61,62,62,63,63,64,64, 65,65,66,66,67,67,68,68,69,69,70,70]],
            'GRB': [[75,75,76,76,77,77,78,78,  89,89,91,91,93,93,95,95]],
            'LRA1': [[60,60,61,61,62,62,63,63,64,64, 65,65,66,66,67,67,68,68,69,69,70,70, 75,75,76,76,77,77,78,78]],
            'LRB1': [[89,89,91,91,93,93,95,95]],
            'GRIncA': [[0,0,0,1,1,1,2,2,2]],
            'GRIncB': [[3,3,3,4,4,4,5,5,5]],
            'LRSA': [[58]],
            'LRSB': [[58]],
            'LWSA': [[95]],
            'LWSB': [[95]],
            'LCC': [[119,119]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=0 newLW=0 newLR=0 for iteration == 0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=19, vscnt=-1, comment="wait for prior local read local write old=0, new=0 newLW=0 newLR=0, wait for previous set of global reads"),
            SBarrier(comment="")
        ]
        numMfma = 120
        nglshift = nllshift = len(optSchedule["GRA"][0])/2 + len(optSchedule["GRB"][0])/2
        opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift)
    elif isNN(kernel) and useLDSTr and TLDS==1:
        optSchedule = {
            'SYNC': [[-1,
                      26,26,
                      59,59
                    ]],
            'LRA0': [[0,2,2,3,3,4,4,5,5,6,6,7,7,9,9,11,11,13,13,15,15,17,17,19,19,21,21,23,23,24],
                     [0,2,2,3,3,4,4,5,5,6,6,7,7,8,8,10,10,12,12,14,14,16,16,18,18,20,20,22,22,25]],
            'LRB0': [[26,27,28,29]],
            'GRA': [[26,26,27,27,29,29,31,31,33,33,35,35,37,37,39,39,41,41,42,42,44,44,46,46,48,48,50,50,52,52,54,54,56,56,58,58,59,59,61,61,63,63,65,65,67,67,69,69,71,71,73,73,75,75,76,76,78,78,80,80]],
            'GRB': [[82,82,84,84,86,86,88,88,90,90,92,92,93,93,96,96],
                    [83,83,85,85,87,87,89,89,91,91,94,94,99,99,103,103]],
            'LRA1': [[59,59,61,61,63,63,65,65,67,67,69,69,71,71,73,73,75,75,76,76,78,78,80,80,82,82,84,84, 86,86]],
            'LRB1': [[88,90,92,94]],
            'LRSA': [[58]],
            'LRSB': [[58]],
            'LWSA': [[95]],
            'LWSB': [[95]],
            'GRIncA': [[1,1,1,17,17,17,18,18,18]],
            'GRIncB': [[19,19,19,20,20,20,21,21,21]],
            'LCC': [[119,119]],
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=3 newLW=0 newLR=3 for iteration == 0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SBarrier(comment=""),
            SWaitCnt(dscnt=0, vlcnt=18, vscnt=-1, comment="wait for prior local read local write old=0, new=0 newLW=0 newLR=0"),
            SBarrier(comment=""),
        ]
        numMfma = 120
        nglshift = nllshift = len(optSchedule["GRA"][0])/2 + len(optSchedule["GRB"][0])/2
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(208, 256, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[2, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[1, 4]
)
def _get_schedule_208x256x64_16bit(kernel, useLDSTr, TLDS):
    numMfma = 104
    syncs = SyncSchedule()

    if isTN(kernel) and not useLDSTr and TLDS==1:
        syncs.add(-1, dscnt=3, comment="wait for all LRA1 and one item from LRB1 before starting the sub-iteration")
        lra0   = [0,1,2,3,5,7,9,11,13,15,17,19,20]

        syncs.add(12, dscnt=8, comment="wait for the rest of LRB1 to complete")
        grinca = [4,4,4,10,10,10,14,14,14]
        lrb0   = [22,24,26,28]
        grincb = [16,16,16,18,18,18,21,21,21]

        syncs.add(29, dscnt=4, barrier=True, comment="wait for all LRA0 to complete before GRA start")
        # one index for two instructions
        gra    = [30,31,32,33,34,35,36,37,38,39,41,43,45,46,48,50,52,53,55,57,59,60,62,64,66,67]

        syncs.add(40, dscnt=0, barrier=True, comment="wait for all LRB0 to complete before GRB start")
        # one index for two instructions
        grb    = [69,71,73,77,81,85,89,93]
        num_gr = len(gra) + len(grb)
        lrsa   = [50]
        lrsb   = [50]
        lwsa   = [70]
        lwsb   = [70]

        syncs.add(72, vlcnt=num_gr-6, barrier=True, comment="wait for previous set of global reads")
        lra1   = [73,74,75,76,78,82,84,86,88,90,92,94,96]
        lrb1   = [80,98,99,100]

    elif isNN(kernel) and useLDSTr and TLDS==1:
        syncs.add(-1, dscnt=3, comment="wait for all LRA1 and one item from LRB1 before starting the sub-iteration")
        grinca = [8,9,10,11,13,14,15,16,17]
        grincb = [19,20,21,22,23,24,25,26,27]

        syncs.add(12, dscnt=12, comment="wait for the rest of LRB1 to complete")
        lra0   = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] # 26 loads

        syncs.add(30, dscnt=2, barrier=True, comment="wait for all LRA0 to complete before GRA start")
        lrb0   = [26,28,30,33]

        syncs.add(38, dscnt=0, barrier=True, comment="wait for all LRB0 to complete before GRB start")
        # one index for two instructions
        gra    = [31,32,34,36,37,39,41,43,45,46,48,50,52,53,55,57,59,60,62,64,66,67,68,69,70,71] # 26 loads
        grb    = [73,74,81,83,87,91,92,93]
        num_gr = len(gra) + len(grb)

        syncs.add(72, vlcnt=num_gr-8, barrier=True, comment="wait for previous set of global reads")
        lra1   = [73,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99] # 26 loads
        lrb1   = [74,100,101,102]

        lrsa   = [49]
        lrsb   = [51]
        lwsa   = [84]
        lwsb   = [85]

    elif isNT(kernel) and useLDSTr and TLDS==0:
        syncs.add(-1, dscnt=3, comment="wait for all LRA1 and one item from LRB1 before starting the sub-iteration")
        grinca = [8,9,10,11,13,14,15,16,17]
        grincb = [19,20,21,22,23,24,25,26,27]

        syncs.add(12, dscnt=12, comment="wait for the rest of LRB1 to complete")
        lra0   = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] # 26 loads
        lrb0   = [27,29,31,33,35,38,40,42] # 8 loads
        syncs.add(30, dscnt=2, barrier=True, comment="wait for all LRA0 to complete before GRA start")

        syncs.add(49, dscnt=0, barrier=True, comment="wait for all LRB0 to complete before GRB start")
        # one index for two instructions
        gra    = [31,32,34,36,37,39,41,43,45,46,48,50,52,53,55,57,59,60,62,64,66,67,68,69,70,71] # 26 loads
        grb    = [73,74,81,83,87,91,92,93] # 8 loads
        num_gr = len(gra) + len(grb)

        # 8 GRBs from previous iteration + 16 GRAs from current iteration (indices 31-57) can be still pending
        syncs.add(58, vlcnt=(8+16), barrier=True, comment="wait for previous set of GRA")
        lra1   = [59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84] # 26 loads

        # 20 GRAs (indices 31-64) from current iteration can be still pending
        syncs.add(65,vlcnt=20, barrier=True, comment="wait for previous set of GRB")
        lrb1   = [66,85,86,88,90,92,94,96] # 8 loads

        lrsa   = [49]
        lrsb   = [51]
        lwsa   = [84]
        lwsb   = [85]
    else:
        return False, None

    def duplicate_list_items(input_list, repeat_count):
        """Example: duplicate_list_items([1, 2, 3], 3) => [1,1,1, 2,2,2, 3,3,3]"""
        return [item for item in input_list for _ in range(repeat_count)]

    optSchedule = {
        'SYNC':   [syncs.get_indicies()],
        'LRA0':   [lra0],
        'GRIncA': [grinca],
        'LRB0':   [lrb0],
        'GRIncB': [grincb],
        # Note: each GRA/GRB item corresponds to two instructions (addr increment and read). So duplicate each item twice.
        'GRA':    [duplicate_list_items(gra, 2)],
        'GRB':    [duplicate_list_items(grb, 2)],
        'LRSA':   [lrsa],
        'LRSB':   [lrsb],
        'LWSA':   [lwsa],
        'LWSB':   [lwsb],
        'LRA1':   [lra1],
        'LRB1':   [lrb1],
        'LCC':    [[numMfma-2, numMfma-1]],
    }
    syncCode = syncs.get_code()
    nglshift = nllshift = num_gr

    kernel["MfmaInitCVgprs"] = True
    kernel["SwapGlobalReadOrder"] = False
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(128, 224, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x224x64_16bit(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []

    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and not useLDSTr and TLDS==1:
        optSchedule = {

            'SYNC'   : [[-1,3, 10,10, 26,27, 45,45]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[22,22,24,24,25,25,26,27,27]],

            'LRA0'   : [[0,2,3,4]],  ## -2 is place holder

            'LRB0'   : [[8,11,13,15,17,19,21],   ## After LRA0, we can mix LRB0 and GRA
                        [9,12,14,16,18,20,22]],
            ## GRA should start after LRA0 is done.
            'GRA'    : [[10,11, 14,14, 17,17, 20,20],
                        [11,12, 15,15, 18,18, 21,21]],

            ## GRB should start after LRB0 is done
            'GRB'    : [[28,28, 31,31, 34,34, 37,37, 40,40, 43,43, 46,46],  # m0 inc is part of GRA/GRB
                        [29,29, 32,32, 35,35, 38,38, 41,41, 44,44, 47,47]],
            'LRA1'   : [[29, 32, 35, 38],
                        [30, 33, 36, 39]],

            #After GRB is done.
            'LRB1'   : [[45,46,47,48,49,50,51]],

            'LRSA'   : [[23]], # after LRA0 and before LRA1
            'LRSB'   : [[23]], # after LRB0 and before LRB2
            'LWSA'   : [[54]], # For A
            'LWSB'   : [[54]],

            'LCC'    : [[55, 55]],
        }
        # note: syncCode needs to be
        syncCode = [SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for necessary prior LRA1/LRB1 before starting main loop"),
                    SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 for the remaining main loop"),
                    SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete to start GRA"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=11, vscnt=-1, comment="Wait for LRB0/GRA to complete to start GRB/LRA1"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="Wait for GRB to complete to start LRB1"),
                    SBarrier(comment=""),
                   ]
        nglshift = nllshift = 11 # vmcnt shift for ngl and nll
    elif isNN(kernel) and useLDSTr and TLDS==1:
        optSchedule = {

            'SYNC'   : [[-1,3, 12,12, 26,27, 45,45]],
            'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
            'GRIncB' : [[22,22,24,24,25,25,26,27,27]],

            'LRA0'   : [[0,0,2,2,3,3,5,6]],  ## -2 is place holder

            'LRB0'   : [[8,10,13,15,17,19,21],   ## After LRA0, we can mix LRB0 and GRA
                        [9,11,14,16,18,20,22]],
            ## GRA should start after LRA0 is done.
            'GRA'    : [[10,12, 14,14, 17,17, 20,20],
                        [11,13, 15,15, 18,18, 21,21]],

            ## GRB should start after LRB0 is done
            'GRB'    : [[28,28, 31,31, 34,34, 37,37, 40,40, 43,43, 46,46],  # m0 inc is part of GRA/GRB
                        [29,29, 32,32, 35,35, 38,38, 41,41, 44,44, 47,47]],
            'LRA1'   : [[29,30, 32,33, 35,36, 38,39],
                        [30,31, 33,34, 36,37, 39,40]],

            #After GRB is done.
            'LRB1'   : [[45,46,47,48,49,50,51]],

            'LRSA'   : [[23]], # after LRA0 and before LRA1
            'LRSB'   : [[23]], # after LRB0 and before LRB2
            'LWSA'   : [[54]], # For A
            'LWSB'   : [[54]],

            'LCC'    : [[55, 55]],
        }
        # note: syncCode needs to be
        syncCode = [SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for necessary prior LRA1/LRB1 before starting main loop"),
                    SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for prior LRA1/LRB1 for the remaining main loop"),
                    SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete to start GRA"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=0, vlcnt=11, vscnt=-1, comment="Wait for LRB0/GRA to complete to start GRB/LRA1"),
                    SBarrier(comment=""),
                    SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="Wait for GRB to complete to start LRB1"),
                    SBarrier(comment=""),
                   ]
        nglshift = nllshift = 11 # vmcnt shift for ngl and nll
    elif isNT(kernel) and useLDSTr and TLDS == 0:
        valid, opt = _get_schedule_224x128x64_16bit(kernel, useLDSTr, TLDS)
        if not valid:
            return False, None
        optSchedule = switch_A_B_schedule(opt.optSchedule)
        return True, ScheduleInfo(opt.numCodePaths, opt.numMfma, optSchedule, opt.syncCode, opt.nglshift, opt.nllshift)
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 56
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(128, 192, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x192x64_16bit(kernel, useLDSTr, TLDS):
    """128x192x64 TN schedule (BF16/FP16)."""
    kernel["MfmaInitCVgprs"] = True

    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    numMfma = 2 * kernel["MIWaveTileA"] * kernel["MIWaveTileB"] # 48

    syncs = SyncSchedule()
    gr_inc_step = 0

    if isTN(kernel) and not useLDSTr and TLDS==1:
        grinca = [0,1,2, 3,4,5, 6,7,7]
        grincb = [7,8,9, 9,9,10, 10,10,11]

        syncs.add(-1, dscnt=5, comment="wait for all LRA1 and one item from LRB1 before starting the sub-iteration")
        lra0   = [0,1,2,3]
        syncs.add(      3, dscnt=3, comment="wait for the rest of LRB1 to complete")
        lrb0   = [       4,5,6,8,11,  14]

        syncs.add(                 12, dscnt=5, barrier=True, comment="wait for LRA0 before GRA start")
        gra    = [                   13,15,17,19] # one index for two instructions
        
        syncs.add(                             21, dscnt=0, vlcnt=4+6, barrier=True, comment="wait for LRB0 before GRB start + wait for previous GRAs before LRA1")
        grb    = [                              21,24,    27,31,34,37] # one index for two instructions
        lra1   = [                                22,25,26,29]
        syncs.add(                                                35, vlcnt=4+5, barrier=True, comment="wait for previous GRBs to complete before LRB1")
        lrb1   = [                                                35,38,40,42,44,46]

        num_gr = len(gra) + len(grb)
        lrsa   = [18]
        lrsb   = [18]
        lwsa   = [30]
        lwsb   = [30]

    else:
        return False, None

    optSchedule = {
        'SYNC':   [syncs.get_indicies()],
        'LRA0':   [lra0],
        'GRIncA': [grinca],
        'LRB0':   [lrb0],
        'GRIncB': [grincb],
        # Note: each GRA/GRB item corresponds to two instructions (addr increment and read). So duplicate each item twice.
        'GRA':    [duplicate_list_items(gra, 2, gr_inc_step)],
        'GRB':    [duplicate_list_items(grb, 2, gr_inc_step)],
        'LRSA':   [lrsa],
        'LRSB':   [lrsb],
        'LWSA':   [lwsa],
        'LWSB':   [lwsb],
        'LRA1':   [lra1],
        'LRB1':   [lrb1],
        'LCC':    [[numMfma-3, numMfma-2]],
    }
    syncCode = syncs.get_code()
    nglshift = nllshift = num_gr
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift)

    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(128, 192, 32, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x192x32_TF32(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isNN(kernel) and not useLDSTr and TLDS==1:
        # TODO: Add NN schedule in upcoming PR
        return False, None
    elif isTN(kernel) and not useLDSTr and TLDS==1:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = False
        kernel["UseDot2F32XEmulation"] = False
        syncTable = [
            5,  SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Before PackA0. Wait for all LRA0. Skip 1*LRB0.") ,
            17, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Before PackB0. Wait for all prior LRB0 for PackB0.") ,
            17, SBarrier(comment="GRA") ,
            32, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Before GRB. Wait for all prior LRB0.") ,
            32, SBarrier(comment="GRB") ,
            35, SWaitCnt(dscnt=-1, vlcnt=6, vscnt=-1, comment="Before LRB3. Wait for GRB from previous iter. Skip 4*GRA + 2*GRB") ,
            35, SBarrier(comment="LRB") ,
            44, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Before PackB3. Wait for all prior LRB3.") ,
            53, SWaitCnt(dscnt=0, vlcnt=10, vscnt=-1, comment="Before LRA3. Wait for GRA from previous iter. Skip 4*GRA + 6*GRB") ,
            53, SBarrier(comment="LRA") ,
            63, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Before PackA3. Wait for all prior LRA3.") ,
        ]
        optSchedule = {
            'SYNC'  : [syncTable[::2]],
            'GRIncA': [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
            'GRIncB': [[3, 3, 3, 4, 4, 4, 5, 5, 5]],
            'LRA0'  : [[0, 1, 2, 3]],
            'LRB0'  : [[4, 6, 8, 10, 12, 14],
                       [4, 7, 9, 11, 13, 15]],
            'PackA0': [create_range(5,12,17, 1, 4)],
            'PackB0': [create_range(18,18,34, 1, 4)],
            'GRA'   : [[17,17, 18,18, 19,19, 20,20]],
            'GRB'   : [[33, 33, 34, 34, 41, 42, 43, 44, 51, 51, 52, 52]],
            'LRB3'  : [[36, 37, 38, 39, 40, 41]],
            'LRA3'  : [[53, 55, 57, 59],
                       [54, 56, 58, 60]],
            'PackB3': [create_range(44,9,52, 1, 8)],
            'PackA3': [create_range(63,8,71, 1, 6)],
            'LRSA'  : [[16]],
            'LRSB'  : [[16]],
            'LWSA'  : [[61]],
            'LWSB'  : [[62]],
            'LCC'   : [[71, 71]],
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 10
    elif isNT(kernel) and useLDSTr and TLDS==0:
        # TODO: Add NT schedule in upcoming PR
        return False, None
    else:
        return False, None
    
    kernel["MfmaInitCVgprs"] = True
    numMfma = 72
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(192, 256, 32, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_192x256x32_TF32(kernel, useLDSTr, TLDS):
    numMfma = 144
    optSchedule = dict()
    syncCode = []
    mfmaReorder = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and not useLDSTr and TLDS==1:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UseDot2F32XEmulation"] = False

        # Used the following constrains to create schedule
        #  - LRA0 + PACKA0 needs to be done before 1/4 MFMAs
        #  - LBR0 + PACKB0 needs to be done before 2/4 MFMAs
        #  - LRB3 + PACKB3 needs to start after 2/4 MFMAs
        #  - LRA3 + PACKA3 needs to start after 3/4 MFMAs

        # LRA0 + GRIncA
        lra0 = create_range(min_val = 0, num = 6, step = 1, repeat = 1)
        grIncA = create_range(min_val = max(lra0)+1, num = 3, step = 1, repeat = 3)
        # Hide LRA0 latency behind GRIncA
        waitLRA0 = max(grIncA)+5
        startPACKA0 = waitLRA0

        # Reordering of packA instructions.
        # 4 CVT + 2 4x4x4_16B MFMAs + 4 CVTs
        # we interleave the 3 blocks together to avoid :
        # - having a 5 state wait after each 4x4x4_16B MFMA
        # - having extra latency when switching between MFMA types
        packAOffset = [ 
                   0, 0, 1, 1, 
                   6, 6,
                   7, 7, 8, 8,

                   2, 2, 3, 3, 
                   6, 6,
                   9, 9, 10, 11,

                   4, 4, 5, 5, 
                   6, 6,
                   12, 12, 13, 13,
                   ]


        packA0 = [x + startPACKA0 for x in packAOffset]
        packA0Done = max(packA0)

        # Sanity check
        assert packA0Done < numMfma//4

        # LRB0 + GRIncB
        lrb0 = create_range(min_val = max(packA0)+1, num = 8, step = 1, repeat = 1)
        grIncB = create_range(max(lrb0)+1,3,max(lrb0)+4,1,3)
        waitLRB0 = max(grIncB)+6
        startPACKB0 = waitLRB0
        packBOffset = [ 
            0, 0, 1, 1, 
            8, 8,
            9, 9, 10, 11,

            2, 2, 3, 3, 
            8, 8,
            12, 12, 13, 13,

            4, 4, 5, 5, 
            8, 8,
            14, 14, 15, 15,

            6, 6, 7, 7, 
            8, 8,
            16, 16, 17, 17,
            ]

        packB0 = [x + startPACKB0 for x in packBOffset]

        # GRA                
        grA = [create_range(min_val = max(packB0)+1, num = 6, step = 2,repeat = 2),
               create_range(min_val = max(packB0)+2, num = 6, step = 2,repeat = 2)]

        halfMFMA = numMfma//2
        assert max(packB0) < halfMFMA

        # LR3
        startLRB3 = halfMFMA
        lrb3 = create_range(min_val = startLRB3, num = 2, step = 1, repeat = 2)
        lrb3 += create_range(min_val = max(lrb3)+6,num = 2, step = 1, repeat = 2)

        # GRB (split in two blocks)
        grB = create_range(min_val = max(lrb3)+1,num = 4,step = 2, repeat = 2)
        waitLRB3 = max(grB)+1 
        grB += create_range(min_val = max(grB)+47,num = 4,step = 2, repeat = 2)

        # PackB3 (starts after 1st GRB block)
        packB3 = [x + waitLRB3 for x in packBOffset]

        # LRA3 + PACKA3
        startLRA3 = (3*numMfma)//4 # Can't start before 3/4 MFMAs
        lra3 = create_range(min_val = startLRA3,num=6,step=1,repeat=1)
        waitLRA3 = max(lra3) + 8 
        packA3 = [x + waitLRA3 for x in packAOffset]

        syncTable = [                    
                    waitLRA0, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    waitLRB0, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),

                    max(packB0)+1, SBarrier(comment="Barrier before GRA&GRB"),

                    startLRB3-1,SWaitCnt(dscnt=-1, vlcnt=5, vscnt=-1, comment="Wait for previous GRA&B"),
                    startLRB3-1,SBarrier(comment=""),

                    waitLRB3,SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB3 to complete"),
                    waitLRA3, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA3 to complete"),                    
                    ]

        syncCode = syncTable[1::2]
        optSchedule = {
            'SYNC': [syncTable[::2]],

            'GRIncA': [grIncA],
            'GRIncB': [grIncB],
            'LRA0': [lra0],
            'LRB0': [lrb0],
            'PackA0' : [packA0],
            'PackB0' : [packB0],

            'GRA': [*grA],
            'GRB': [grB],              
            'LRSA': [[max(grIncB)+1]],
            'LRSB': [[max(grIncB)+2]],
            'LWSA': [[142]],
            'LWSB': [[142]],
            'LCC': [[143, 143]],
            'LRA3': [lra3],
            'LRB3': [lrb3],
            'PackB3' : [packB3],
            'PackA3' : [packA3],
        }

        nglshift = nllshift = 14 # vmcnt shift for ngl and nll
    elif isNN(kernel) and TLDS==1 and kernel["VectorWidthA"] == 1:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UseDot2F32XEmulation"] = False

        numLrReadA = 24 
        numLrReadB = 8

        # A is 24 instructions as current codegen can't generate ds_read_b128 in NN case.
        # Instead of reading A first, this schedule re-orders the mfma instructions and reads B first (less instructions)
        # Before :
        #  B0 - A0
        #  B0 - A1
        #  B1 - A0
        #  B1 - A1
        # Now :
        #  B0 - A0
        #  B1 - A0
        #  B0 - A1
        #  B1 - A1

        # mfma Reordering
        mfmaReorder = [i for i in range(0,numMfma//4)] + [i for i in range(numMfma//2,3*numMfma//4)]+[i for i in range(numMfma//4,numMfma//2)]+[i for i in range(3*numMfma//4,numMfma)]

        # Interleaving of LBR0 and GRINCB to hide LRB0 latency
        lrb0 = create_range(min_val = 0, num = 6, step = 1, repeat = 1)
        grIncB = create_range(min_val = max(lrb0)+1, num = 3, step = 1, repeat = 3)
        lrb0 += create_range(min_val = max(grIncB)+1, num = 2, step = 1, repeat = 1)
        grIncA = create_range(min_val = max(lrb0)+1, num = 3, step = 1, repeat = 3)
        waitLRB0 = max(grIncA)+4

        # PackB0 using mfma4x4x4_16b
        startPACKB0 = waitLRB0
        packBOffset = [ 
            0, 0, 1, 1, 
            8, 8,
            9, 9, 10, 10,

            2, 2, 3, 3, 
            8, 8,
            11, 11, 12, 12,

            4, 4, 5, 5, 
            8, 8,
            13, 13, 14, 14,

            6, 6, 7, 7, 
            8, 8,
            15, 15, 16, 16,
            ]

        packB0 = [x + startPACKB0 for x in packBOffset]
        packB0Done = max(packB0)

        # Sanity check
        assert packB0Done < numMfma//4

        # GRB (1st block) interleaved with LRA0
        grB = [create_range(min_val = packB0Done+2,num = 4,step = 4, repeat = 2),
               create_range(min_val = packB0Done+1,num = 4,step = 4, repeat = 2)]
       
        # LRA0 
        lra0 = [create_range(min_val = max(packB0)+1, num = numLrReadA // 2, step = 2, repeat = 2),
                create_range(min_val = max(packB0)+2, num = numLrReadA // 2, step = 2, repeat = 2)]
       
        # PackA0
        waitLRA0 = max(lra0[1])+2
        startPACKA0 = waitLRA0
        packAOffset = [ 
                   0, 0, 1, 1, 
                   6, 6,
                   7, 7, 8, 8,

                   2, 2, 3, 3, 
                   6, 6,
                   9, 9, 10, 10,

                   4, 4, 5, 5, 
                   6, 6,
                   11, 11, 12, 12,
                   ]

        packA0 = [x + startPACKA0 for x in packAOffset]

        halfMFMA = numMfma//2
        assert max(packA0) < halfMFMA

        # LRA3 interleaved with GRB (2nd half)
        startLRA3 = halfMFMA
        lra3 = [create_range(min_val = startLRA3, num = numLrReadA // 2, step = 2, repeat = 2),
                create_range(min_val = startLRA3+1, num = numLrReadA // 2, step = 2, repeat = 2)]
        grB[0] += create_range(min_val = startLRA3+1,num = 4,step = 2, repeat = 2)
        grB[1] += create_range(min_val = startLRA3,num = 4,step = 2, repeat = 2)
        waitLRA3 = max(lra3[0])+4  
        # LRB3 + PACKA3 & PACKB3
        startLRB3 = (3*numMfma)//4 - 4 # Starts 4 indexes before 3/4 MFMAs to accommodate LRB3 latency
        lrb3 = create_range(min_val = startLRB3,num=numLrReadB - 2,step=1,repeat=1)
        grA = [create_range(min_val = min(lrb3)+1, num = 8, step = 1,repeat = 1),
               create_range(min_val = min(lrb3)+3, num = 8, step = 1,repeat = 1)]
        lrb3 += create_range(min_val = max(lrb3)+3,num=2,step=1,repeat=1)
        waitLRB3 = max(lrb3) + 6 

        # Grouping segment of 4x4x4_16B MFMAs together for PackB3 & PackA3 (reduce MFMA type switching cost)
        packB3 = [x + waitLRB3 for x in packBOffset]
        start_4x4x4 = packB3[4] # 5th index is start of 4x4x4_16B MFMA for PackB3
        packA3 = [ 
                   *create_range(min_val = waitLRA3, num = 2, step = 1, repeat = 2),
                   start_4x4x4,start_4x4x4,
                   *create_range(min_val = max(packB3)+1, num = 2, step = 1, repeat = 2),

                   *create_range(min_val = waitLRA3+2, num = 2, step = 1, repeat = 2),
                   start_4x4x4,start_4x4x4,
                   *create_range(min_val = max(packB3)+3, num = 2, step = 1, repeat = 2),

                   *create_range(min_val = waitLRA3+4, num = 2, step = 1, repeat = 2),
                   start_4x4x4,start_4x4x4,
                   *create_range(min_val = max(packB3)+5, num = 2, step = 1, repeat = 2),
                   ]

        # GRA 2nd half
        grA[0] += create_range(min_val = max(packB3)+1, num = 2, step = 1,repeat = 2)
        grA[1] += create_range(min_val = max(packB3)+1, num = 2, step = 1,repeat = 2)

        syncTable = [                                      
                    waitLRB0, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for 4/8 LRB0 to complete"),
                    waitLRB0+4, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRB0 to complete"),
                    waitLRB0+4, SBarrier(comment="Barrier before GRB"), #Barrier can be after CVT

                    # dscnt has a max value of 15
                    waitLRA0, SWaitCnt(dscnt=min(15,numLrReadA-4), vlcnt=-1, vscnt=-1, comment="Wait for 4 LRA0 to complete"),
                    waitLRA0+1, SWaitCnt(dscnt=min(15,numLrReadA-8), vlcnt=-1, vscnt=-1, comment="Wait for 8 LRA0 to complete"),
                    waitLRA0+2, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),

                    startLRA3-1,SWaitCnt(dscnt=-1, vlcnt=4, vscnt=-1, comment="Wait for previous GRA&B"),
                    startLRA3-1,SBarrier(comment="Sync before GRA, LRA3 & LRB3"),

                    # incremental wait on LRA3
                    waitLRA3, SWaitCnt(dscnt=min(15,numLrReadA-4), vlcnt=-1, vscnt=-1, comment="Wait for 4 LRA3 to complete"),                    
                    waitLRA3+1, SWaitCnt(dscnt=min(15,numLrReadA-8), vlcnt=-1, vscnt=-1, comment="Wait for 8 LRA3 to complete"), 
                    waitLRA3+2, SWaitCnt(dscnt=min(15,numLrReadA-12), vlcnt=-1, vscnt=-1, comment="Wait for 12 LRA3 to complete"), 
                    waitLRA3+3, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA3 to complete"),                    

                    # incremental wait on LRB3
                    waitLRB3, SWaitCnt(dscnt=(numLrReadB-1), vlcnt=-1, vscnt=-1, comment="Wait for 1st LRB3 to complete"),
                    waitLRB3+1, SWaitCnt(dscnt=(numLrReadB-2), vlcnt=-1, vscnt=-1, comment="Wait for 2nd LRB3 to complete"),
                    waitLRB3+2, SWaitCnt(dscnt=(numLrReadB-3), vlcnt=-1, vscnt=-1, comment="Wait for 3rd LRB3 to complete"),
                    waitLRB3+3, SWaitCnt(dscnt=(numLrReadB-4), vlcnt=-1, vscnt=-1, comment="Wait for 4th LRB3 to complete"),
                    waitLRB3+4, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRB3 to complete"),

                    ]

        syncCode = syncTable[1::2]
        optSchedule = {

            'SYNC': [syncTable[::2]],

            'GRIncA': [grIncA],
            'GRIncB': [grIncB],
            'LRA0': [*lra0],
            'LRB0': [lrb0],
            'PackA0' : [packA0],
            'PackB0' : [packB0],
            'GRA': [*grA],
            'GRB': [*grB],              
            'LRSA': [[max(lra0[1])+1]],
            'LRSB': [[max(lra0[1])+1]],
            'LWSA': [[numMfma-2]],
            'LWSB': [[numMfma-2]],
            'LCC': [[numMfma-1, numMfma-1]],
            'LRA3': [*lra3],
            'LRB3': [lrb3],
            'PackB3' : [packB3],
            'PackA3' : [packA3],

        }

        nglshift = nllshift = 14
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift, mfmaReorder=mfmaReorder)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 192, 32, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x192x32_TF32(kernel, useLDSTr, TLDS):
    numMfma = 144
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and not useLDSTr and TLDS == 1:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = False
        kernel["UseDot2F32XEmulation"] = False
        numPackInstr = 24 
        numPackIndices = numPackInstr // 2 # Assign 2 pack instructions per mfma index
        
        # LRA0 + PACKA0 - done before 1/4 MFMAs - index 36
        lrA0 = [0,0, 1,1, 2,2, 3,3]
        waitLRA0 = max(lrA0) + 2
        startPACKA0 = waitLRA0
        packA0 = create_range(startPACKA0, (len(lrA0)//2)*numPackIndices, numMfma//4-1)
        
         # LBR0 + PACKB0 - done before 2/4 MFMAs - index 72
        lrB0 = [7,7, 11,11, 15,15]
        waitLRB0 = max(lrB0) + 2
        startPACKB0 = max(waitLRB0,max(packA0)) # Starts after waitLRB0 and packA0
        packB0 = create_range(startPACKB0, (len(lrB0)//2)*numPackIndices, numMfma//2-1)
        
        # LBR3 + PACKB3 - start after 2/4 MFMAs - index 72
        halfMFMA = numMfma//2
        startLRB3 = halfMFMA
        lrB3 = create_range(startLRB3, 1, numMfma-1)
        lrB3 += create_range(max(lrB3)+6, 2, numMfma-1)
        waitLRB3 = startLRB3 + 6
        packB3 = create_range(waitLRB3, (len(lrB3)//2)*numPackIndices, numMfma-1)
        
        # LRA3 + PACKA3 - start after 3/4 MFMAs - index 108
        startLRA3 = (3*numMfma)//4
        lrA3 = create_range(startLRA3, 4, numMfma-1)
        waitLRA3 = startLRA3 + 5
        packA3 = create_range(waitLRA3, (len(lrA3)//2)*numPackIndices, numMfma-1)
        
        syncTable = [
            waitLRA0, SWaitCnt(dscnt=inflight(lrA0, waitLRA0)-2, vlcnt=-1, vscnt=-1, comment="wait for 1st 2 LRA0 to complete"),
            waitLRA0+numPackIndices, SWaitCnt(dscnt=inflight(lrB0, waitLRA0+numPackIndices), vlcnt=-1, vscnt=-1, comment="wait for all LRA0 to complete"),
            
            waitLRB0, SWaitCnt(dscnt=inflight(lrB0, waitLRB0)-2, vlcnt=-1, vscnt=-1, comment="wait for 1st 2 LRB0 to complete"),
            waitLRB0+numPackIndices, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for all LRB0 to complete"),
            waitLRB0+numPackIndices, SBarrier(comment="Barrier before GRA&GRB"),
            
            startLRB3-1, SWaitCnt(dscnt=-1, vlcnt=6, vscnt=-1, comment="Wait for prev GRA&GRB"),
            startLRB3-1, SBarrier(comment=""),
            
            waitLRB3,SWaitCnt(dscnt=inflight(lrB3, waitLRB3)-2, vlcnt=-1, vscnt=-1, comment="Wait for 1st 2 LRB3 to complete"),
            waitLRB3+numPackIndices,SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRB3 to complete"),

            waitLRA3, SWaitCnt(dscnt=inflight(lrA3,waitLRA3)-2, vlcnt=-1, vscnt=-1, comment="Wait for 1st 2 LRA3 to complete"),
            waitLRA3+numPackIndices, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRA3 to complete")
        ]

        optSchedule = {
            'SYNC'   : [syncTable[::2]],
            
            'GRIncA' : [[0,0,0, 1,1,1, 2,2,2]],
            'GRIncB' : [[3,3,3, 4,4,4, 5,5,5]], 

            'LRA0'   : [lrA0],
            'PackA0' : [packA0],
            
            'LRB0'   : [lrB0],
            'PackB0' : [packB0],
            
            'GRA'    : [[72,72, 74,74, 76,76, 100,100, 102,102, 104,104, 106,106, 108,108],
                        [73,73, 75,75, 77,77, 101,101, 103,103, 105,105, 107,107, 109,109]],
            'GRB'    : [[38,38, 40,40, 42,42, 44,44, 46,46, 48,48],
                        [39,39, 41,41, 43,43, 45,45, 47,47, 49,49]],
            
            'LRA3'   : [lrA3],
            'PackA3' : [packA3],
            
            'LRB3'   : [lrB3],
            'PackB3' : [packB3],

            'LRSA'   : [[35]],
            'LRSB'   : [[35]],

            'LWSA'   : [[107]],
            'LWSB'   : [[107]],

            'LCC'    : [[143, 143]],
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 14 # vmcnt shift for ngl and nll
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)

    elif isNN(kernel) and TLDS==1 and kernel["VectorWidthA"] == 1:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = True
        
        numLrReadA = 32 
        numLrReadB = 6

        # mfma Reordering
        mfmaReorder = [i for i in range(0,numMfma//4)] + [i for i in range(numMfma//2,3*numMfma//4)]+[i for i in range(numMfma//4,numMfma//2)]+[i for i in range(3*numMfma//4,numMfma)]

        # Interleave LBR0 and GRINCB
        lrb0 = create_range(min_val = 0, num = 6, step = 1, repeat = 1)
        grIncB = create_range(min_val = 0, num = 6, step = 1, repeat = 1)
        grIncB += create_range(min_val = max(lrb0)+1, num = 1, step = 1, repeat = 3)
        
        # Interleave GRINCA and PACKB0
        grIncA = create_range(min_val = max(grIncB)+1, num = 2, step = 1, repeat = 3)
        waitLRB0 = max(grIncA)
        grIncA += create_range(min_val = max(grIncA)+2, num = 3, step = 1, repeat = 1)

        startPACKB0 = waitLRB0
        packBOffset = [ 
                   0, 0, 1, 1, 
                   6, 6,
                   7, 7, 8, 8,

                   2, 2, 3, 3, 
                   6, 6,
                   9, 9, 10, 10,

                   4, 4, 5, 5, 
                   6, 6,
                   11, 11, 12, 12,
                   ]

        packB0 = [x + startPACKB0 for x in packBOffset]
        packB0Done = max(packB0)

        # Sanity check
        assert packB0Done < numMfma//4, f"packB0Done {packB0Done} >= numMfma//4 {numMfma//4}"

        # GRB (1st block) interleaved with LRA0
        grB = [create_range(min_val = packB0Done+2,num = 3,step = 4, repeat = 2),
               create_range(min_val = packB0Done+1,num = 3,step = 4, repeat = 2)]
       
        # LRA0 
        lra0 = [create_range(min_val = max(packB0)+1, num = numLrReadA // 2, step = 2, repeat = 2),
                create_range(min_val = max(packB0)+2, num = numLrReadA // 2, step = 2, repeat = 2)]
       
        # PackA0
        waitLRA0 = max(lra0[1])+1
        startPACKA0 = waitLRA0

        packAOffset = [ 
            0, 0, 1, 1, 
            8, 8,
            9, 9, 10, 10,

            2, 2, 3, 3, 
            8, 8,
            11, 11, 12, 12,

            4, 4, 5, 5, 
            8, 8,
            13, 13, 14, 14,

            6, 6, 7, 7, 
            8, 8,
            15, 15, 16, 16,
            ]
        packA0 = [x + startPACKA0 for x in packAOffset]

        halfMFMA = numMfma//2
        assert max(packA0) < halfMFMA, f"max(packA0) {max(packA0)} >= halfMFMA {halfMFMA}"

        # LRA3 interleaved with GRB (2nd half)
        startLRA3 = halfMFMA
        lra3 = [create_range(min_val = startLRA3+1, num = numLrReadA // 2, step = 2, repeat = 2),
                create_range(min_val = startLRA3, num = numLrReadA // 2, step = 2, repeat = 2)]

        # M0 update before barrier to prevent bad interleaving between the 2 codepaths
        grB[0] += [startLRA3-2,startLRA3]
        grB[1] += [startLRA3-2,startLRA3+1]

        grB[0] += create_range(min_val = startLRA3+2,num = 2,step = 2, repeat = 2)
        grB[1] += create_range(min_val = startLRA3+3,num = 2,step = 2, repeat = 2)
        
        # LRB3 + PACKA3 & PACKB3
        startLRB3 = (3*numMfma)//4 - 4 # Starts 4 indexes before 3/4 MFMAs to accommodate LRB3 latency
        lrb3 = create_range(min_val = startLRB3,num=numLrReadB - 2,step=1,repeat=1)
        
        grA = [create_range(min_val = max(lra3[0])+1, num = 8, step = 1,repeat = 1),
               create_range(min_val = max(lra3[1])+1, num = 8, step = 1,repeat = 1)]
        lrb3 += create_range(min_val = max(lrb3)+3,num=2,step=1,repeat=1)
        
        waitLRB3 = max(lrb3) + 9 

        # Grouping segment of 4x4x4_16B MFMAs together for PackB3 & PackA3 (reduce MFMA type switching cost)
        packB3 = [x + waitLRB3 for x in packBOffset]
        start_4x4x4 = packB3[4] # 5th index is start of 4x4x4_16B MFMA for PackB3
        waitLRA3 = max(lrb3)+1
        packA3 = [ 
                   *create_range(min_val = waitLRA3, num = 2, step = 1, repeat = 2),
                   start_4x4x4, start_4x4x4,
                   *create_range(min_val = max(packB3)+1, num = 2, step = 1, repeat = 2),

                   *create_range(min_val = waitLRA3+2, num = 2, step = 1, repeat = 2),
                   start_4x4x4, start_4x4x4,
                   *create_range(min_val = max(packB3)+3, num = 2, step = 1, repeat = 2),

                   *create_range(min_val = waitLRA3+4, num = 2, step = 1, repeat = 2),
                   start_4x4x4, start_4x4x4,
                   *create_range(min_val = max(packB3)+5, num = 2, step = 1, repeat = 2),

                   *create_range(min_val = waitLRA3+6, num = 2, step = 1, repeat = 2),
                   start_4x4x4, start_4x4x4,
                   *create_range(min_val = max(packB3)+7, num = 2, step = 1, repeat = 2),
                   ]

        # GRA 2nd half
        grA[0] += create_range(min_val = max(packB3)+1, num = 4, step = 1,repeat = 2)
        grA[1] += create_range(min_val = max(packB3)+1, num = 4, step = 1,repeat = 2)

        syncTable = [                                      
                    waitLRB0, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for 1/6 LRB0 to complete"),
                    waitLRB0+1, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for 2/6 LRB0 to complete"),
                    waitLRB0+2, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for 3/6 LRB0 to complete"),
                    waitLRB0+3, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for 4/6 LRB0 to complete"),
                    waitLRB0+4, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for 5/6 LRB0 to complete"),
                    waitLRB0+5, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for 6/6 LRB0 to complete"),
                    waitLRB0+5, SBarrier(comment="Barrier before GRB"), 

                    # incremental wait on LRA0
                    waitLRA0, SWaitCnt(dscnt=min(15,numLrReadA-4), vlcnt=-1, vscnt=-1, comment="Wait for 4 LRA0 to complete"),
                    waitLRA0+4, SWaitCnt(dscnt=numLrReadA-20, vlcnt=-1, vscnt=-1, comment="Wait for 20 LRA0 to complete"),
                    waitLRA0+5, SWaitCnt(dscnt=numLrReadA-24, vlcnt=-1, vscnt=-1, comment="Wait for 24 LRA0 to complete"),
                    waitLRA0+6, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),

                    startLRA3-1,SWaitCnt(dscnt=-1, vlcnt=3, vscnt=-1, comment="Wait for previous GRA&B"),
                    startLRA3-1,SBarrier(comment="Sync before GRA, LRA3 & LRB3"),

                    # incremental wait on LRA3 & LRB3
                    waitLRA3, SWaitCnt(dscnt=15, vlcnt=-1, vscnt=-1, comment="Wait for 17/32 LRA3 to complete"),         
                    waitLRA3+4, SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for LRA3 to complete"),                    
                    waitLRA3+7, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB3 to complete"),                    

                    ]

        syncCode = syncTable[1::2]
        optSchedule = {

            'SYNC': [syncTable[::2]],

            'GRIncA': [grIncA],
            'GRIncB': [grIncB],
            'LRA0': [*lra0],
            'LRB0': [lrb0],
            'LRSA': [[packA0[4]]],#Use slot between MFMA 16x16x32 & 4x4x4 for LRSA
            'LRSB': [[packA0[4]]],
            'PackA0' : [packA0],
            'PackB0' : [packB0],
            'GRA': [*grA],
            'GRB': [*grB],              
            'LWSA': [[numMfma-2]],
            'LWSB': [[numMfma-2]],
            'LCC': [[numMfma-1, numMfma-1]],
            'LRA3': [*lra3],
            'LRB3': [lrb3],
            'PackB3' : [packB3],
            'PackA3' : [packA3],

        }

        nglshift = nllshift = 14
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift, mfmaReorder=mfmaReorder)
    
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 256, 32, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x256x32_TF32(kernel, useLDSTr, TLDS):
    numMfma = 192
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0
    disable_validation = False
    if isTN(kernel) and not useLDSTr and TLDS==1:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UseDot2F32XEmulation"] = False
        # This schedule follows similar pattern as 192x256x32 TF32 schedule

        # LRA0 + GRIncA
        lra0 = create_range(min_val = 0, num = 4, step = 1, repeat = 1)
        lra0 += create_range(min_val = max(lra0)+4, num = 4, step = 1, repeat = 1)

        grIncA = create_range(min_val = max(lra0)+1, num = 3, step = 1, repeat = 3)
        waitLRA0 = max(grIncA)+5
        startPACKA0 = waitLRA0

        # Use a common packOffset re-ordering for both A and B
        packOffset = [ 
            0, 0, 1, 1, 
            8, 8,
            9, 9, 10, 11,

            2, 2, 3, 3, 
            8, 8,
            12, 12, 13, 13,

            4, 4, 5, 5, 
            8, 8,
            14, 14, 15, 15,

            6, 6, 7, 7, 
            8, 8,
            16, 16, 17, 17,
            ]   


        packA0 = [x + startPACKA0 for x in packOffset]
        packA0Done = max(packA0)

        # Sanity check
        assert packA0Done < numMfma//4

        # LRB0 + GRIncB (Split LRB0 into two halves to hide latency)
        lrb0 = create_range(min_val = max(packA0)+1, num = 4, step = 1, repeat = 1)
        lrb0 += create_range(min_val = max(lrb0)+4, num = 4, step = 1, repeat = 1)
        grIncB = create_range(max(lrb0)+1,3,max(lrb0)+4,1,3)
        waitLRB0 = max(grIncB)+6
        startPACKB0 = waitLRB0

        packB0 = [x + startPACKB0 for x in packOffset]

        # GRA - 1st half (4 reads)                
        grA = create_range(min_val = max(packB0)+1, num = 4, step = 2,repeat = 2)

        halfMFMA = numMfma//2
        assert max(packB0) < halfMFMA

        # LR3
        startLRB3 = halfMFMA
        lrb3 = create_range(min_val = startLRB3, num = 4, step = 1, repeat = 1)
        lrb3 += create_range(min_val = max(lrb3)+4, num = 4, step = 1, repeat = 1)

        # GRA - 2nd half (4 reads)   
        grA += create_range(min_val = max(lrb3)+1, num = 4, step = 2,repeat = 2)
        waitLRB3 = max(grA)+1 

        # PackB3
        packB3 = [x + waitLRB3 for x in packOffset]

        # GRB - 1st half (4 reads) 
        grB = create_range(min_val = max(packB3)+1,num = 4,step = 2, repeat = 2)

        # LRA3 + PACKA3
        startLRA3 = (3*numMfma)//4 
        lra3 = create_range(min_val = startLRA3,num=4,step=1,repeat=1)
        lra3 += create_range(min_val = max(lra3)+4,num=4,step=1,repeat=1)
        waitLRA3 = max(lra3) + 10 
        packA3 = [x + waitLRA3 for x in packOffset]

        # GRB - 2nd half (4 reads) 
        grB += create_range(min_val = max(packA3)+1,num = 4,step = 2, repeat = 2)

        syncTable = [                    
                    waitLRA0, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
                    waitLRB0, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),

                    max(packB0)+1, SBarrier(comment="Barrier before GRA&GRB"),

                    startLRB3-1,SWaitCnt(dscnt=-1, vlcnt=4, vscnt=-1, comment="Wait for previous GRA&B"),
                    startLRB3-1,SBarrier(comment=""),

                    waitLRB3,SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB3 to complete"),
                    waitLRA3, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA3 to complete"),                    
                    ]

        syncCode = syncTable[1::2]
        optSchedule = {

            'SYNC': [syncTable[::2]],

            'GRIncA': [grIncA],
            'GRIncB': [grIncB],
            'LRA0': [lra0],
            'LRB0': [lrb0],
            'PackA0' : [packA0],
            'PackB0' : [packB0],

            'GRA': [grA],
            'GRB': [grB],              
            'LRSA': [[max(grIncB)+1]],
            'LRSB': [[max(grIncB)+2]],
            'LWSA': [[numMfma-2]],
            'LWSB': [[numMfma-2]],
            'LCC': [[numMfma-1, numMfma-1]],
            'LRA3': [lra3],
            'LRB3': [lrb3],
            'PackB3' : [packB3],
            'PackA3' : [packA3],

        }

        nglshift = nllshift = 16 # vmcnt shift for ngl and nll
    elif isNT(kernel) and not useLDSTr and TLDS==0 and kernel["VectorWidthA"] == 4 and kernel["VectorWidthB"] == 4:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UseDot2F32XEmulation"] = False
        swap_idx =   [1,2,3, # depend on DS1 
                        7,8, # depend on DS2
                         11, # depend on DS3
                      4,5,6, # depend on DS5
                       9,10, # depend on DS6
                         12, # depend on DS7
                        ]
        optSchedule = {
            'SYNC': [[9, 14, 38, 43, 73, 95, 95, 106, 111, 157, 162]],
            'GRIncA': [[0,1,2,3, 5,6,7,8, 11]],
            'GRIncB': [[38, 39, 40, 41, 45, 46, 47, 50, 51]],
            'LRA0': [[0,1,2,3, 5,6,7,8]],
            'LRB0': [[28, 30, 31, 32, 34, 35, 36, 37]],

            'PackA0': [[*[x + 8 for x in swap_idx],
                        20,20,21,21, 28,28, 29,29,30,30,
                        22,22,23,23, 28,28, 31,31,32,32, 
                        24,24,25,25, 28,28, 33,33,34,34, 
                        26,26,27,27, 28,28, 34,35,36,36]],
            'PackB0': [[*[x + 37 for x in swap_idx],
                        55,55,56,56, 63,63, 64,64,65,65,
                        57,57,58,58, 63,63, 66,66,67,67,
                        59,59,60,60, 63,63, 68,68,69,69,
                        61,61,62,62, 63,63, 70,70,71,71]],
            'LRSA': [[50]],
            'LRSB': [[51]],
            'LRB3': [[96,96,98,98, 103,103,105,105],
                    [97,97,99,99,  102,102,104,104]],                    
            'GRA': [[75,75,80,80,85,85,90,90, 107,107,109,109,111,111,113,113],
                    [77,77,82,82,87,87,92,92, 108,108,110,110,112,112,114,114]],
            'LRA3': [[138, 138, 144, 144, 150, 150, 154, 154],
                     [139, 139, 146, 146, 152, 152, 156, 156]],
            'GRB': [[130,130,135,135,140,140,145,145, 165,165,170,170,175,175,180,180],
                    [132,132,137,137,142,142,147,147, 167,167,172,172,177,177,182,182]],
            
            'PackB3': [[*[x + 105 for x in swap_idx],
                        119,119,120,120, 128,128, 129,129,130,130,
                        121,121,122,122, 128,128, 131,131,132,132,
                        123,123,124,124, 128,128, 133,133,134,134,
                        125,125,126,126, 128,128, 135,135,136,136]], 
            'PackA3': [[*[x + 156 for x in swap_idx],
                        169,169,170,170, 177,177, 178,178,179,179,
                        171,171,172,172, 177,177, 180,180,181,181,
                        173,173,174,174, 177,177, 182,182,183,183,
                        175,175,176,176, 177,177, 184,184,185,185]],
            'LWSA': [[187]],
            'LWSB': [[188]],
            'LCC': [[189, 190]],
        }

        syncCode = [                    
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
            SBarrier(comment="Barrier before GRA&GRB"),
            SWaitCnt(dscnt=-1, vlcnt=4, vscnt=-1, comment="Wait for previous GRA&B"),
            SBarrier(comment=""),
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRA3 to complete"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA3 to complete"),
            SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRB3 to complete"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB3 to complete"),
        ]
        nglshift = nllshift = 16 # vmcnt shift for ngl and nll
        # disable the validation until 4x4MFMA with wider loads is supported by validator
        disable_validation = True
    else:
        return False, None
        
    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    if disable_validation:
        opt1.disableValidation("4x4MFMA with wider loads is not yet supported by validator")
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(192, 128, 32, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_192x128x32_TF32(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    if isTN(kernel) and useLDSTr and TLDS==1:

        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = False
        kernel["UseDot2F32XEmulation"] = False
        # Used the following constrains to create schedule
        #  - LRA0 + PACKA0 needs to be done before 1/4 MFMAs - index 18
        #  - LBR0 + PACKB0 needs to be done before 2/4 MFMAs - index 36
        #  - LRB3 + PACKB3 needs to start after 2/4 MFMAs - index 36
        #  - LRA3 + PACKA3 needs to start after 3/4 MFMAs - index 54
        syncTable = [
                    -1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),

                    6, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="wait for first 2 LRA0"),
                    11, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for all LRA0"),

                    18, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="wait for first 2 LRB0"),
                    23, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for all LRB0"),
                    23, SBarrier(comment="Barrier before GRA&GRB"),

                    44, SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="Wait for prev GRA&GRB"),
                    44, SBarrier(comment=""),

                    49, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="wait for first 2 LRB3"),
                    56, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for all LRB3"),


                    57, SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="Wait for prev GRA&GRB"),
                    57, SBarrier(comment=""),

                    61, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="wait for first 2 LRA3"),
                    66, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for all LRA3"),

                    ]

        syncCode = syncTable[1::2]

        optSchedule = {
            'SYNC'  : [syncTable[::2]],
            'GRIncA': [[0,0,0,1,1,1,2,2,2]],
            'GRIncB': [[3,3,3,4,4,4,5,5,5]],
            #  - LRA0 + PACKA0 needs to be done before 1/4 MFMAs - index 18
            'LRA0': [[0,1,2,3,4,5]],
            'PackA0' : [[7]*4 + [8]*10 + [9]*10 + [12]*4 + [13]*20 + [14]*8 + [15]*8 + [16]*8],
            #  - LBR0 + PACKB0 needs to be done before 2/4 MFMAs - index 36
            'LRB0': [[12,13,14,15]],
             # First two LRB0 need to be done at 18, all LRB0 done by 23
            'PackB0' : [create_range(19,4,22, repeat=3) +  create_range(24,12,35, repeat=3) ],
            'GRB': [[36,36,38,38,40,40,42,42],
                    [37,37,39,39,41,41,43,43]],
            'GRA': [[45,45,47,47,49,49,51,51,53,53,55,55],
                    [46,46,48,48,50,50,52,52,54,54,56,56]],
            'LRSA': [[36]],
            'LRSB': [[36]],
            'LWSA': [[52]],
            'LWSB': [[52]],
            'LCC': [[71, 71]],
            #  - LRB3 + PACKB3 needs to start after 2/4 MFMAs - index 36
            'LRB3': [[45, 46, 47, 48]],
             # First two LRB3 need to be done by 43, all LRB3 done at 48 
            'PackB3' : [create_range(50,12,71, repeat=4)],
            #  - LRA3 + PACKA3 needs to start after 3/4 MFMAs - index 54
            'LRA3': [[58,59,60,62,63,64]],
             # First two LRA3 need to be done by 61. All LRA0 done by 66.
            'PackA3' : [create_range(62,12,71, repeat=6)],

        }

        nglshift = nllshift = 10 # vmcnt shift for ngl and nll
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    numMfma = 72
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(128, 128, 32, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x128x32_TF32(kernel, useLDSTr, TLDS):
    n_mfma = 4 * 4 * 3    # 128 MT0 / 2 WT0 / 16 mfma dim  * 128/2/16 * 3 bf16 MFMAs per tf32 mfma

    optSchedule = dict()
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    syncs = SyncSchedule()
    syncCode = []   
    snops: list[tuple[int, SNop]] = []
    snopCode = []

    if isTN(kernel) and not useLDSTr and TLDS==1:
        kernel["UseMFMAF32XEmulation"] = True

        lra0 = [0,0,1,1]
        syncs.add( 3, dscnt=2, comment="Wait for the first 2 LRA0 to complete before pack")
        syncs.add( 5, dscnt=0, comment="Wait for the rest    LRA0 to complete before pack")
        pack_a0 = [3,3,4,4, 7,7, 8,8,9,9, 5,5,6,6, 7,7, 10,10,11,11]
        pack_b0 = [12,12,13,13, 16,16, 17,17,18,18,  14,14,15,15, 16,16,  19,19,20,20]

        lrb0 = [6,6,7,7]
        syncs.add(11, dscnt=0, comment="Wait for LRB0 to complete before pack",
                  barrier=True, barrier_comment="Wait for all waves to finish LRs before GRs")
        grinca = [0,1,2, 2,2,2, 2,4,5]
        grincb = [6,8,9, 10,11,12, 13,14,15]
        lrsa = [13]
        lrsb = [14]
        lwsa = [45]
        lwsb = [45]
        
        gra = [15,17, 18,19, 20,21, 25,26]
        grb = [27,28, 31,33, 36,37, 39,40]
        num_gr = (len(gra[1::2]) + len(grb[1::2]))
        
        gr_wait = 23
        v = count_items(gra[1::2]+grb[1::2], ev=gr_wait)
        syncs.add(gr_wait, vlcnt=v, barrier=True, comment = "Wait for previous GRA&B")

        lrb3 = [24,24,25,25]
        syncs.add( 28, dscnt=2, comment="Wait for the first 2 LRB3 to complete")
        syncs.add( 30, dscnt=0, comment="Wait for the rest    LRB3 to complete")
        pack_b3 = [28,28,29,29, 32,32,  33,33,34,34,  30,30,31,31, 32,32,  35,35,36,36]
        
        lra3 = [36,36,37,37]
        syncs.add(39, dscnt=2, comment="Wait for the first 2 LRA3 to complete")
        syncs.add(41, dscnt=0, comment="Wait for the rest    LRA3 to complete")
        pack_a3 = [39,39,40,40, 43,43, 44,44,45,45, 41,41,42,42, 43,43, 46,46,47,47]

    else:
        return False, None

    optSchedule = {
        'SYNC':   [syncs.get_indicies()],
        'GRIncA': [grinca],
        'GRIncB': [grincb],
        'LRA0':   [lra0],
        'LRB0':   [lrb0],
        'PackA0': [pack_a0],
        'PackB0': [pack_b0],
        'GRA':    [gra],
        'GRB':    [grb],
        'LRSA':   [lrsa],
        'LRSB':   [lrsb],
        'LWSA':   [lwsa],
        'LWSB':   [lwsb],
        'LRA3':   [lra3],
        'LRB3':   [lrb3],
        'PackB3': [pack_b3],
        'PackA3': [pack_a3],
        'LCC':    [[n_mfma-2, n_mfma-1]],
    }

    syncCode = syncs.get_code()
    nglshift = nllshift = num_gr
    if snops:
        optSchedule['SNOP'] = [ [s[0] for s in snops] ]
        snopCode = [s[1] for s in snops]
 
    kernel["MfmaInitCVgprs"] = True
    kernel["UsePLRPack"] = True
    opt1 = ScheduleInfo(1, n_mfma, optSchedule, syncCode, nglshift, nllshift, snopCode=snopCode)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(128, 128, 32, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[32, 32, 16, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x128x32_TF32_plr1(kernel, useLDSTr, TLDS):
    n_mfma = 128//2//32 * 128//2//32 * 3 * 2    # 128 MT0 / 2 WT0 / 32 mfma dim  * 128/2/32 * 3 bf16 MFMAs per tf32 mfma * 2 PLR=1

    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    syncs = SyncSchedule()
    gr_inc_step = 0
    disable_validation = False
    num_code_paths = 1

    if isTN(kernel) and not useLDSTr and TLDS==1:
        lra0   = [0,1,2,3]
        lrb0   = [       4,5,6,7]
        #                wait then read
        syncs.add(       4, dscnt=2, comment="wait for the first 2 LRAs before packing")
        syncs.add(         5, dscnt=1, comment="wait for the rest of LRAs before packing them")
        pack_a0 = [      4,4,4,4, 6,6, 7,7,7,7,
                           5,5,5,5, 6,6, 8,8,8,8]
        # because of GR starting at 10, we need barrier at 9, will use that for sync too.
        syncs.add(                               9, dscnt=0, comment="wait for LRBs before the packing them",
                                                 barrier=True, barrier_comment="make sure all LRs are done before starting GR")
        pack_b0= [                               9,9,9,9, 10,10, 11,11,11,11,
                                                 9,9,9,9, 10,10, 11,11,11,11]

        grinca = [0,0,0,1,1,1,2,2,2]
        grincb = [3,3,3,6,6,6,6,6,6]
        lrsa   = [10]
        lrsb   = [10]    
        
        gra    = [                                 10,10,11,11] # one index for two instructions
        grb    = [                                              13,13,14,14] # one index for two instructions
        num_gr = len(gra) + len(grb)
        syncs.add(                                             12, vlcnt=8, barrier=True, comment="wait for the previous GRAs")

        lra1   = [                                             12,12,13,14] # twice on 12 since we are waiting for GRA anyway at 12
        lrb1   = [                                                        15,16,16,17]
        #                                                                 wait then read
        syncs.add(                                                        15, dscnt=2, vlcnt=8, comment="wait for the first 2 LRAs before packing. Also wait for GRBs",
                                                                              barrier=True, barrier_comment="make sure GRBs are done before starting LRBs"  )
        syncs.add(                                                            17, dscnt=3, comment="wait for the rest of LRAs before packing them")
        pack_a1 = [                                                          16,16,16,16, 20,20, 21,21,21,21,
                                                                              17,17,17,17, 20,20, 21,21,21,21]
        syncs.add(                                                              18, dscnt=2, comment="wait for 2 LRBs before the packing them")
        syncs.add(                                                               19, dscnt=0, comment="wait for the rest of LRBs before the packing them")
        pack_b1= [                                                              18,18,18,18, 20,20, 22,22,22,22,
                                                                                 19,19,19,19, 20,20, 22,22,23,23]
        lwsa   = [                                                                          20] # use delay before mfma4x4x4
        lwsb   = [                                                                          20]
        
    elif isNN(kernel) and TLDS==1  and kernel["VectorWidthA"] == 2:
        disable_validation = True # swap instructions included in pack are not supported yet

        lra0   = [0,0,0,0,
                    1,1,1,1]
        lrb0   = [     3,  4,6,6]
        #                wait then read
        syncs.add(     3, dscnt=4, comment="wait for the first 2x2 LRAs before packing")
        syncs.add(         4, dscnt=1, comment="wait for the rest of LRAs")
        pack_a0 = [    3,3,4,4, # swap instructions, must come after LR and before other packs
                             4,5,5,5, 6,6, 7,7,7,7, 
                             5,5,6,6, 6,6, 8,8,8,8]
        # because of GR starting at 10, we need barrier at 9, will use that for sync too.
        syncs.add(                               9, dscnt=0, comment="wait for LRBs before the packing them",
                                                 barrier=True, barrier_comment="make sure all LRs are done before starting GR")
        pack_b0= [                               10,10,10,10, 10,10, 11,11,11,11,
                                                 9,9,9,9,     10,10, 11,11,11,11]
        grinca = [0,0,1, 1,2,2, 2,2,2]
        grincb = [2,2,6, 7,7,7, 8,8,8]
        lrsa   = [10]
        lrsb   = [10]   
        
        num_code_paths = 2
        gra   = [                                9,9,   11,11]
        gra2  = [                                 10,10,11,11]
        grb    = [                                              13,        14,14,17] # one index for two instructions
        grb2   = [                                              13,         15,15,17] # one index for two instructions
        num_gr = len(gra) + len(grb)
        syncs.add(                                             12, vlcnt=8, barrier=True, comment="wait for the previous GRAs")

        lra1   = [                                             12,12,12,12,
                                                                13,13,13,13]
        syncs.add(                                                         14, vlcnt=4+1, barrier=True, barrier_comment="make sure GRBs are done before starting LRBs"  )
        lrb1   = [                                                         14,15,16,16]
        syncs.add(                                                            15, dscnt=1, comment="wait for LRAs")
        pack_a1 =[                                                            15,15,16,16, # swap instructions, must come after LR and before other packs
                                                                                17,17,17,17, 20,20, 21,21,21,21,
                                                                                 18,18,18,18, 20,20, 21,21,21,21]
        syncs.add(                                                                19, dscnt=2, comment="wait for the first 2 LRBs before the packing them")
        syncs.add(                                                                 20, dscnt=0, comment="wait for the rest of LRBs")
        pack_b1= [                                                                19,19,19,19, 20,20, 22,22,22,22,
                                                                                   20,20,20,20, 20,20, 22,22,22,22]
        lwsa   = [                                                                            20] # use delay before mfma4x4x4
        lwsb   = [                                                                            20]    
    
    elif isNT(kernel) and useLDSTr and TLDS==0  and kernel["VectorWidthA"] == 2 and kernel["VectorWidthB"] == 2:
        disable_validation = True # swap instructions included in pack are not supported yet

        lra0   = [0,0,0,0,
                    1,1,1,1]
        lrb0   = [     3,3,4,4,
                                 6,6,6,6]
        #              wait then read
        syncs.add(     3, dscnt=4, comment="wait for the first 2x2 LRAs before packing")
        syncs.add(         4, dscnt=2, comment="wait for the rest of LRAs")
        pack_a0 = [    3,3,4,4, # swap instructions, must come after LR and before other packs
                             4,5,5,5, 6,6, 7,7,7,7, 
                             5,5,6,6, 6,6, 8,8,8,8]
        # because of GR starting at 10, we need barrier at 9, will use that for sync too.
        syncs.add(                               9, dscnt=0, comment="wait for LRBs",
                                                 barrier=True, barrier_comment="make sure all LRs are done before starting GR")
        pack_b0= [                               9,9, 9,9, # swap instructions, must come after LR and before other packs
                                                 10,10,10,10, 10,10, 11,11,11,11,
                                                 9,9,9,9,     10,10, 11,11,11,11]
        grinca = [0,0,1, 1,2,2, 2,2,2]
        grincb = [2,2,6, 7,7,7, 8,8,8]
        lrsa   = [10]
        lrsb   = [10]   
        
        num_code_paths = 2
        gra   = [                                9,9,   11,11]
        gra2  = [                                 10,10,11,11]
        grb    = [                                              13,        14,14,17] # one index for two instructions
        grb2   = [                                              13,         15,15,17] # one index for two instructions
        num_gr = len(gra) + len(grb)
        syncs.add(                                             12, vlcnt=8, barrier=True, comment="wait for the previous GRAs")

        lra1   = [                                             12,12,12,12,
                                                                13,13,13,13]
        syncs.add(                                                         14, vlcnt=4+1, barrier=True, barrier_comment="make sure GRBs are done before starting LRBs"  )
        lrb1   = [                                                         14,14,15,15,
                                                                             16,16,16,16]
        syncs.add(                                                            15, dscnt=2, comment="wait for LRAs")
        pack_a1 =[                                                            15,15,16,16, # swap instructions, must come after LR and before other packs
                                                                                17,17,17,17, 20,20, 21,21,21,21,
                                                                                 18,18,18,18, 20,20, 21,21,21,21]
        syncs.add(                                                                19, dscnt=0, comment="wait for LRBs")
        pack_b1= [                                                                19,19,19,19, # swap instructions, must come after LR and before other packs
                                                                                  19,19,19,19, 20,20, 22,22,22,22,
                                                                                   20,20,20,20, 20,20, 22,22,22,22]
        lwsa   = [                                                                            20] # use delay before mfma4x4x4
        lwsb   = [                                                                            20]    

    else:
        return False, None  
    
    final_gra = [duplicate_list_items(gra, 2, gr_inc_step)]
    if num_code_paths == 2:
        final_gra += [duplicate_list_items(gra2, 2, gr_inc_step)]
    
    final_grb = [duplicate_list_items(grb, 2, gr_inc_step)]
    if num_code_paths == 2:
        final_grb += [duplicate_list_items(grb2, 2, gr_inc_step)]

    optSchedule = {
        'SYNC':   [syncs.get_indicies()],
        'GRIncA': [grinca],
        'GRIncB': [grincb],
        'LRA0':   [lra0],
        'LRB0':   [lrb0],
        'GRA':    final_gra,
        'GRB':    final_grb,
        'LRSA':   [lrsa],
        'LRSB':   [lrsb],
        'LWSA':   [lwsa],
        'LWSB':   [lwsb],
        'PackA0': [pack_a0],
        'PackB0': [pack_b0],
        'LRA1':   [lra1],
        'LRB1':   [lrb1],
        'PackB1': [pack_b1],
        'PackA1': [pack_a1],
        'LCC':    [[n_mfma-1, n_mfma-1]],
    }

    syncCode = syncs.get_code()
    nglshift = nllshift = num_gr

    kernel["MfmaInitCVgprs"] = True
    kernel["UsePLRPack"] = True
    kernel["UseMFMAF32XEmulation"] = True
    kernel["UseDot2F32XEmulation"] = False
    opt1 = ScheduleInfo(num_code_paths, n_mfma, optSchedule, syncCode, nglshift, nllshift)
    if disable_validation:
        opt1.disableValidationPass(cmsv.ValidatorPass.ADD_PACK_CONSTRAINTS, "swap instructions included in pack are not supported yet")
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(128, 128, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x128x64_TF32(kernel, useLDSTr, TLDS):
    n_mfma = 96
    optSchedule = dict()
    nglshift = nllshift = 0

    syncs = SyncSchedule()
    syncCode = []   
    gr_inc_step = 1
    disable_validation = False

    if isTN(kernel) and not useLDSTr and TLDS==1:
        offset=[0,0,1,1, 8,8,  9, 9,10,10, 
                2,2,3,3, 8,8, 11,11,12,12,
                4,4,5,5, 8,8, 13,13,14,14, 
                6,6,7,7, 8,8, 15,15,16,16]
        
        lra0   = [ 0,1,2,3,4,5,6,7]
        lrb0   = [   8,9,10,11,12,13,14,15]
        #                wait then read
        syncs.add(       10, dscnt=6, comment="wait for the first 2 LRAs before packing")
        syncs.add(       14, dscnt=6, comment="wait for the rest of LRAs before packing them")
        pack_a0= [          i+11 for i in offset] # last at 27
        # because of GR starting at 22, we need barrier at 21, will use that for sync too.
        syncs.add(                          21, dscnt=0, comment="wait for LRBs before the packing them",
                                            barrier=True, barrier_comment="barrier before GR")
        pack_b0= [                                i+28 for i in offset] # last at 44

        grinca = [0,0,1,1,2,2,3,3,4]
        grincb = [5,5,6,6,7,7,8,8,9]
        lrsa   = [45]
        lrsb   = [45]
        lwsa   = [72]
        lwsb   = [72]        
        
        gra    = [                            22,25,29,33, 37,41,45,49] # one index for two instructions
        grb    = [                                                    53,57,61, 64,69,75,79,84] # one index for two instructions
        num_gr = len(gra) + len(grb)

        syncs.add(                                                 48, vlcnt=7, barrier=True, comment="wait for the previous GRs")

        lra1   =[[                                                 48,49,50,51,52,53,54,55]]
        lrb1   = [                                                    56,57,  59,60,61,62,63,64]
        syncs.add(                                                          58, dscnt=6, comment="wait for the first two LRAs before packing")
        syncs.add(                                                          63, dscnt=6, comment="wait for the rest of LRAs before packing them")
        pack_a1 = [                                                           i+59 for i in offset] # last at 75
        syncs.add(                                                                                 76, dscnt=0, comment="wait for LRBs before the packing them")
        pack_b1 =[                                                                                 i+77 for i in offset] # last at 93

    elif isNN(kernel) and TLDS==1 and kernel["VectorWidthA"] == 4:
        disable_validation = True

        offset=[0,0,1,1, 8,8,  9, 9,10,10, 
                2,2,3,3, 8,8, 11,11,12,12,
                4,4,5,5, 8,8, 13,13,14,14, 
                6,6,7,7, 8,8, 15,15,16,16]
        
        lra0   = [ 0,0,2,2,4,4,6,6]
        lrb0   = [                                       11,11,13,13,15,15,17,17]
        #                wait then read
        syncs.add(             6, dscnt=2, comment="wait for the first 4 LRAs before swapping/packing")
        syncs.add(                     9, dscnt=0, comment="wait for the rest of LRAs before swapping/packing them")
        pack_a0= [              7,7,7, 9,9,9, 8,8, 10,10, 8, 10] # swap instructions
        pack_a0+=[                                       i+11 for i in offset] # last at 27
        # because of GR starting at 22, we need barrier at 21, will use that for sync too.
        syncs.add(                                                                21, dscnt=0, comment="wait for LRBs before the packing them",
                                                                                  barrier=True, barrier_comment="barrier before GR")
        pack_b0= [                                                                  i+28 for i in offset] # last at 44

        grinca = [0,1,1,1,2,3,3,3,4]
        grincb = [5,5,5,7,8,9,10,19,19]
        lrsa   = [45]
        lrsb   = [45]
        lwsa   = [72]
        lwsb   = [72]        
        
        gra    = [                            22,25,29,33, 37,41,45,49] # one index for two instructions
        grb    = [                                                    53,57,61, 64,69,75,79,84] # one index for two instructions
        num_gr = len(gra) + len(grb)

        syncs.add(                                                 48, vlcnt=7, barrier=True, comment="wait for the previous GRs")
        lra1   =[[                                                 48,48,50,50,52,52,54,54],
                                                                   [49,49,51,51,53,53,54,54]]
        lrb1   = [                                                                                  59,59,  61,61,63,63,65,65]
        syncs.add(                                                                   54, dscnt=2, comment="wait for the first two LRAs before packing")
        syncs.add(                                                                             57, dscnt=0, comment="wait for the rest of LRAs before packing them")
        pack_a1 =[                                                                    55,55,55, 57,57,57, 55,55, 57,57, 56, 58] # swap instructions
        pack_a1+= [                                                                                 i+59 for i in offset] # last at 75
        syncs.add(                                                                                     76, dscnt=0, comment="wait for LRBs before the packing them")
        pack_b1 =[                                                                                     i+77 for i in offset] # last at 93

    else:
        return False, None  
    
    optSchedule = {
        'SYNC':   [syncs.get_indicies()],
        'GRIncA': [grinca],
        'GRIncB': [grincb],
        'LRA0':   [lra0],
        'LRB0':   [lrb0],
        'GRA':    [duplicate_list_items(gra,                2, gr_inc_step),
                   duplicate_list_items([i+1 for i in gra], 2, gr_inc_step)],
        'GRB':    [duplicate_list_items(grb,                2, gr_inc_step),
                   duplicate_list_items([i+1 for i in grb], 2, gr_inc_step)],
        'LRSA':   [lrsa],
        'LRSB':   [lrsb],
        'LWSA':   [lwsa],
        'LWSB':   [lwsb],
        'PackA0': [pack_a0],
        'PackB0': [pack_b0],
        'LRA1':   lra1,
        'LRB1':   [lrb1],
        'PackB1': [pack_b1],
        'PackA1': [pack_a1],
        'LCC':    [[n_mfma-2, n_mfma-2]],
    }

    syncCode = syncs.get_code()
    nglshift = nllshift = num_gr

    kernel["MfmaInitCVgprs"] = True
    kernel["UseMFMAF32XEmulation"] = True
    kernel["UseDot2F32XEmulation"] = False
    kernel["UsePLRPack"] = True
    opt1 = ScheduleInfo(2, n_mfma, optSchedule, syncCode, nglshift, nllshift)
    if disable_validation:
        opt1.disableValidationPass(cmsv.ValidatorPass.ADD_PACK_CONSTRAINTS, "Pack validation for NN transpose is not yet supported by validator")
    return True, opt1


@RegisterSchedule(
    tile_config=TileConfig(128, 256, 32, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x256x32_TF32(kernel, useLDSTr, TLDS):
    numMfma = 96
    optSchedule = dict()
    syncCode = []
    mfmaReorder = []
    nglshift = nllshift = 0
    if isTN(kernel) and not useLDSTr and TLDS==1:
        kernel["UsePLRPack"] = True
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UseDot2F32XEmulation"] = False

        # LRA0 + GRIncA
        lra0 = create_range(min_val = 0, num = 4, step = 1, repeat = 1)
        grIncA = create_range(min_val = max(lra0)+1, num = 3, step = 1, repeat = 3)

        waitLRA0 = max(grIncA)+2
        startPACKA0 = waitLRA0

        packAOffset = [
            0, 0, 1, 1,
            4, 4,
            5, 5, 6, 6,

            2, 2, 3, 3,
            4, 4,
            7, 7, 8, 8,
        ]
        packA0 = [x + startPACKA0 for x in packAOffset]
        packA0Done = max(packA0)

        # Sanity check
        assert packA0Done < numMfma//4 , f"packA0Done {packA0Done} >= {numMfma//4}"

        # 1st part of LRB0 + GRIncB
        lrb0 = create_range(min_val = max(packA0)+1, num = 6, step = 1, repeat = 1)
        grIncB = create_range(min_val = max(packA0)+1, num = 4, step = 1, repeat= 2)
        grIncB += [max(grIncB)+1]

        # GRA  + 2nd part of LBR0
        grA = create_range(min_val = max(lrb0)+1, num = 4, step = 2,repeat = 2)
        lrb0 += create_range(min_val = max(lrb0)+4, num = 2, step = 1, repeat = 1)

        waitLRB0 = max(lrb0)+2
        startPACKB0 = waitLRB0

        packBOffset = [
            0, 0, 1, 1,
            8, 8,
            9, 9, 10, 10,

            2, 2, 3, 3,
            8, 8,
            11, 11, 12, 12,

            4, 4, 5, 5,
            8, 8,
            13, 13, 14, 14,

            6, 6, 7, 7,
            8, 8,
            15, 15, 16, 16,
        ]
        packB0 = [x + startPACKB0 for x in packBOffset]

        halfMFMA = numMfma//2
        assert max(packB0) < halfMFMA, f"max(packB0) {max(packB0)} >= halfMFMA {halfMFMA}"

        # LR3
        startLRB3 = halfMFMA
        # Interleave GRA and LBR3
        lrb3 = [create_range(min_val = startLRB3, num = 3, step = 2, repeat = 2),
                create_range(min_val = startLRB3+1, num = 3, step = 2, repeat = 2)]

        # Splitting LRB3 to avoid LDS Issue latency
        lrb3[0]+= create_range(min_val = max(lrb3[0])+5, num = 1, step = 2, repeat = 2)
        lrb3[1]+= create_range(min_val = max(lrb3[1])+5, num = 1, step = 2, repeat = 2)

        grB = [create_range(min_val = startLRB3+1,num = 4,step = 2, repeat = 2),
               create_range(min_val = startLRB3,num = 4,step = 2, repeat = 2)]

        waitLRB3 = max(lrb3[1])+2 

        # Use different PackBOffset to shift last 5 CVTs iterations after GRB/LRA3
        packB3Offset = [
            0, 0, 1, 1,
            8, 8,
            9, 9, 10, 10,

            2, 2, 3, 3,
            8, 8,
            11, 11, 19, 19,

            4, 4, 5, 5,
            8, 8,
            20, 20, 21, 21,

            6, 6, 7, 7,
            8, 8,
            22, 22, 23, 23,
        ]   

        # PackB3
        packB3 = [x + waitLRB3 for x in packB3Offset]

        startLRA3 = (3*numMfma)//4 
        # GRB + LRA3 (interleaved)
        grB[0] += create_range(min_val = startLRA3,num = 4,step = 2, repeat = 2)
        grB[1] += create_range(min_val = startLRA3+1,num = 4,step = 2, repeat = 2)

        lra3 = [create_range(min_val = startLRA3+1,num=4,step=2,repeat=1),
                create_range(min_val = startLRA3,num=4,step=2,repeat=1)]

        waitLRA3 = max(lra3[0]) + 6 
        packA3 = [x + waitLRA3 for x in packAOffset]

        syncTable = [
            -1, SBarrier(comment="Sync codepath"),
            waitLRA0, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for 1st LRA0 to complete"),
            waitLRA0+1, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for 2nd LRA0 to complete"),
            waitLRA0+2, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for 3rd LRA0 to complete"),
            waitLRA0+3, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRA0 to complete"),
            min(grIncB), SBarrier(comment="Barrier before GRA"),

            waitLRB0, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="Wait for 1/8 LRB0 to complete"),
            waitLRB0+1, SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for 2/8 LRB0 to complete"),
            waitLRB0+2, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for 3/8 LRB0 to complete"),
            waitLRB0+3, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for 4/8 LRB0 to complete"),
            waitLRB0+4, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for 5/8 LRB0 to complete"),
            waitLRB0+5, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for 6/8 LRB0 to complete"),
            waitLRB0+6, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for 7/8 LRB0 to complete"),
            waitLRB0+7, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for 8/8 LRB0 to complete"),

            startLRB3-1, SWaitCnt(dscnt=-1, vlcnt=4, vscnt=-1, comment="Wait for previous GRA&B"),
            startLRB3-1, SBarrier(comment="Barrier before GRB and before LRBA3/LBRB3"),

            waitLRB3, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="Wait for 1/8 LRB3 to complete"),
            waitLRB3+1, SWaitCnt(dscnt=6, vlcnt=-1, vscnt=-1, comment="Wait for 2/8 LRB3 to complete"),
            waitLRB3+2, SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for 3/8 LRB3 to complete"),
            waitLRB3+3, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for 4/8 LRB3 to complete"),
            waitLRB3+4, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for 5/8 LRB3 to complete"),
            waitLRB3+5, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for 6/8 LRB3 to complete"),
            waitLRB3+6, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for 7/8 LRB3 to complete"),
            waitLRB3+7, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for 8/8 LRB3 to complete"),

            waitLRA3, SWaitCnt(dscnt=3, vlcnt=-1, vscnt=-1, comment="Wait for 1/4 LRA3 to complete"),                    
            waitLRA3+1, SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="Wait for 2/4 LRA3 to complete"),                    
            waitLRA3+2, SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="Wait for 3/4 LRA3 to complete"),                    
            waitLRA3+3, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for 4/4 LRA3 to complete"),                    
        ]

        syncCode = syncTable[1::2]
        optSchedule = {
            'SYNC': [syncTable[::2]],

            'GRIncA': [grIncA],
            'GRIncB': [grIncB],
            'LRA0': [lra0],
            'LRB0': [lrb0],
            'PackA0' : [packA0],
            'PackB0' : [packB0],

            'GRA': [grA],
            'GRB': [*grB],              
            'LRSA': [[max(lrb0)+1]],
            'LRSB': [[max(lrb0)+1]],
            'LWSA': [[numMfma-2]],
            'LWSB': [[numMfma-2]],
            'LCC': [[numMfma-1, numMfma-1]],
            'LRA3': [*lra3],
            'LRB3': [*lrb3],
            'PackB3' : [packB3],
            'PackA3' : [packA3],
        }
        nglshift = nllshift = 12 # vmcnt shift for ngl and nll
    elif isNN(kernel) and TLDS==1:
        return False, None
        # kernel["UsePLRPack"] = True
        # kernel["UseMFMAF32XEmulation"] = True
        # kernel["UseDot2F32XEmulation"] = False

        # numLrReadA = 16
        # numLrReadB = 8
        # # mfma Reordering
        # mfmaReorder = [i for i in range(0, numMfma//4)] + [i for i in range(numMfma//2, 3*numMfma//4)] + [i for i in range(numMfma//4, numMfma//2)] + [i for i in range(3*numMfma//4, numMfma)]

        # # LBR0
        # lrb0 = create_range(min_val = 0, num = numLrReadB//2, step = 1, repeat = 2)
        # grIncB = create_range(min_val = 0, num = 3, step = 1, repeat = 3)
        # grIncA = create_range(min_val = max(grIncB)+1, num = 3, step = 1, repeat = 3)
        # waitLRB0 = max(lrb0)
        # # PackB0 using mfma4x4x4_16b
        # startPACKB0 = waitLRB0 + 4
        # packBOffset = [ 
        #     0, 0, 1, 1, 
        #     8, 8,
        #     9, 9, 10, 10,

        #     2, 2, 3, 3, 
        #     8, 8,
        #     11, 11, 12, 12,

        #     4, 4, 5, 5, 
        #     8, 8,
        #     13, 13, 14, 14,

        #     6, 6, 7, 7, 
        #     8, 8,
        #     15, 15, 16, 16,
        # ]
        # packB0 = [x + startPACKB0 for x in packBOffset]
        # packB0Done = max(packB0)
        # assert packB0Done < numMfma//4

        # # LRA0 
        # lra0 = create_range(min_val = waitLRB0+4, num = numLrReadA, step = 1, repeat = 1)
        # grA = create_range(min_val = packB0Done+6, num = 8, step = 2, repeat = 1)
        # waitLRA0 = max(lra0)
        # startPACKA0 = waitLRA0 + 2
        # packAOffset = [
        #     0, 0, 1, 1,
        #     4, 4,
        #     5, 5, 6, 6,

        #     2, 2, 3, 3,
        #     4, 4,
        #     7, 7, 8, 8,
        # ]
        # packA0 = [x + startPACKA0 for x in packAOffset]
        # halfMFMA = numMfma//2
        # assert max(packA0) < halfMFMA

        # # LRA3
        # startLRA3 = halfMFMA
        # lra3 = create_range(min_val = startLRA3, num = numLrReadA, step = 1, repeat = 1)
        # grB = create_range(min_val = startLRA3+1, num = 4, step = 2, repeat = 2)
        # grB += create_range(min_val = max(lra3)+1, num = 4, step = 2, repeat = 2)
        # waitLRA3 = max(lra3)
        # startPACKA3 = waitLRA3 + 4

        # # LRB3
        # startLRB3 = (3*numMfma)//4 - 4 # Starts 4 indexes before 3/4 MFMAs to accommodate LRB3 latency
        # lrb3 = create_range(min_val = startLRB3, num = numLrReadB//2, step = 1, repeat = 2)
        # waitLRB3 = max(lrb3)
        # startPACKB3 = waitLRB3 + 4

        # # Grouping segment of 4x4x4_16B MFMAs together for PackB3 & PackA3 (reduce MFMA type switching cost)
        # packB3 = [x + startPACKB3 for x in packBOffset]
        # start_4x4x4 = packB3[4] # 5th index is start of 4x4x4_16B MFMA for PackB3
        # packA3 = [
        #     *create_range(min_val = startPACKA3, num = 2, step = 1, repeat = 2),
        #     start_4x4x4, start_4x4x4,
        #     *create_range(min_val = max(packB3)+1, num = 2, step = 1, repeat = 2),

        #     *create_range(min_val = startPACKA3+2, num = 2, step = 1, repeat = 2),
        #     start_4x4x4, start_4x4x4,
        #     *create_range(min_val = max(packB3)+3, num = 2, step = 1, repeat = 2),
        # ]

        # syncTable = [
        #     waitLRB0, SWaitCnt(dscnt=min(15,numLrReadA-4), vlcnt=-1, vscnt=-1, comment="Wait for 4 LRA0 to complete"),
        #     waitLRB0+4, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 to complete"),
        #     waitLRB0+4, SBarrier(comment=""),

        #     waitLRA0, SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for 4/8 LRB0 to complete"),
        #     waitLRA0+4, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRB0 to complete"),
        #     waitLRA0+4, SBarrier(comment=""),

        #     startLRA3-1, SWaitCnt(dscnt=-1, vlcnt=6, vscnt=-1, comment="Wait for previous GRA&GRB"),
        #     startLRA3-1, SBarrier(comment=""),

        #     waitLRA3, SWaitCnt(dscnt=(numLrReadB-1), vlcnt=-1, vscnt=-1, comment="Wait for 1st LRB3 to complete"),
        #     waitLRA3+4, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRB3 to complete"),

        #     startLRB3-1, SWaitCnt(dscnt=-1, vlcnt=6, vscnt=-1, comment="Wait for previous GRA&GRB"),
        #     startLRB3-1, SBarrier(comment=""),

        #     waitLRB3, SWaitCnt(dscnt=min(15,numLrReadA-4), vlcnt=-1, vscnt=-1, comment="Wait for 4 LRA3 to complete"),
        #     waitLRB3+4, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA3 to complete"),
        # ]

        # optSchedule = {
        #     'SYNC': [syncTable[::2]],
        #     'GRIncA': [grIncA],
        #     'GRIncB': [grIncB],
        #     'LRA0': [lra0],
        #     'LRB0': [lrb0],
        #     'PackA0' : [packA0],
        #     'PackB0' : [packB0],
        #     'GRA': [grA],
        #     'GRB': [grB],
        #     'LRSA': [[max(lra0)+1]],
        #     'LRSB': [[max(lra0)+1]],
        #     'LWSA': [[max(lrb3)+1]],
        #     'LWSB': [[max(lrb3)+1]],
        #     'LCC': [[numMfma-1, numMfma-1]],
        #     'LRA3': [lra3],
        #     'LRB3': [lrb3],
        #     'PackA3' : [packA3],
        #     'PackB3' : [packB3],
        # }
        # syncCode = syncTable[1::2]
        # nglshift = nllshift = 12
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift, mfmaReorder=mfmaReorder)
    return True, opt1


@RegisterSchedule(
    tile_config=TileConfig(128, 160, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x160x64_TF32(kernel, useLDSTr, TLDS):
    n_mfma = 120
    optSchedule = dict()
    nglshift = nllshift = 0

    syncs = SyncSchedule()
    syncCode = []
    gr_inc_step = 0

    if isTN(kernel) and not useLDSTr and TLDS==1:
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UsePLRPack"] = True

        grinca = [0,0,1,1,2,2,3,3,4]
        grincb = [4,5,5,6,6,7,7,8,8]
        lrsa   = [58]
        lrsb   = [59]
        lwsa   = [118]
        lwsb   = [118]

        pack_a = [0,0,1,1, 8,8, 9,9,10,10,
                  2,2,3,3, 8,8, 11,11,12,12,
                  4,4,5,5, 8,8, 13,13,14,14,
                  6,6,7,7, 8,8, 15,15,16,16
                  ]
        pack_b = [0,0,1,1, 10,10, 11,11,12,12,
                  2,2,3,3, 10,10, 13,13,14,14,
                  4,4,5,5, 10,10, 15,15,16,16,
                  6,6,7,7, 10,10, 17,17,18,18,
                  8,8,9,9, 10,10, 19,19,20,20
                  ]
        lra0   = [0,1,2,3,4,5,6,7]
        syncs.add(                 12, dscnt=4, barrier=True, comment="wait for LRA0 before pack to complete + barrier for GRA")
        pack_a0 = [                i+13 for i in pack_a]  ## last element = 13 + 16 = 29

        lrb0   = [               8,9,10,11, 13,14,15,16, 18,19]
        syncs.add(                                               24, dscnt=0, comment="wait for LRB0 before pack to complete")
        pack_b0 = [                                                  i+30 for i in pack_b]  ## last element = 30 + 20 = 50

        gra    = [                                    17,22,27,32, 42,47,52,57] # one index for two instructions
        grb    = [                                                               67,71,75,79, 89,93,97,101, 112,116] # one index for two instructions
        num_gr = len(gra) + len(grb)

        syncs.add(                                                            59, vlcnt=8, barrier=True, comment="wait for previous set of global reads + barrier for GRB")

        lra1   = [60,61,62,63,64,65,66,67]
        syncs.add(                          72, dscnt=4, comment="wait for LRA1 before pack to complete")
        pack_a1 = [                         i+73 for i in pack_a]  ## last element = 73 + 16 = 89

        lrb1   = [                        68,69,70,71, 73,74,75,76, 78,79]
        syncs.add(                                                            85, dscnt=0, comment="wait for LRB1 before pack to complete")
        pack_b1 = [                                                           i+90 for i in pack_b]  ## last element = 90 + 20 = 110

        optSchedule = {
            'SYNC':   [syncs.get_indicies()],
            'GRIncA': [grinca],
            'GRIncB': [grincb],
            'LRA0':   [lra0],
            'LRB0':   [lrb0],
            'PackA0': [pack_a0],
            'PackB0': [pack_b0],
            'GRA':    [duplicate_list_items(gra, 2, gr_inc_step),
                       duplicate_list_items([x+1 for x in gra], 2, gr_inc_step)],
            'GRB':    [duplicate_list_items(grb, 2, gr_inc_step),
                       duplicate_list_items([x+1 for x in grb], 2, gr_inc_step)],
            'LRSA':   [lrsa],
            'LRSB':   [lrsb],
            'LWSA':   [lwsa],
            'LWSB':   [lwsb],
            'LRA1':   [lra1],
            'LRB1':   [lrb1],
            'PackB1': [pack_b1],
            'PackA1': [pack_a1],
            'LCC':    [[n_mfma-2, n_mfma-1]],
        }

        syncCode = syncs.get_code()
        nglshift = nllshift = num_gr
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(2, n_mfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1


@RegisterSchedule(
    tile_config=TileConfig(256, 128, 32, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x128x32_TF32(kernel, useLDSTr, TLDS):
    numMfma = 96
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll

    if isTN(kernel) and useLDSTr and TLDS==1:
        kernel["UseMFMAF32XEmulation"] = False
        kernel["UseDot2F32XEmulation"] = False
        kernel["UsePLRPack"] = True
        numPackInstr = 24 
        numPackIndices = numPackInstr // 2 # Assign 2 pack instructions per mfma index

        # LRA0 + PACKA0 - done before 1/4 MFMAs - index 24
        lrA0 = [0,0, 1,1, 2,2, 3,3]
        waitLRA0 = max(lrA0) + 2
        startPACKA0 = waitLRA0
        packA0 = create_range(startPACKA0, (len(lrA0)//2)*numPackIndices, numMfma//4-1)

         # LBR0 + PACKB0 - done before 2/4 MFMAs - index 48
        lrB0 = [7,7, 15,15]
        waitLRB0 = max(lrB0) + 2
        startPACKB0 = max(waitLRB0,max(packA0)) # Starts after waitLRB0 and packA0
        packB0 = create_range(startPACKB0, (len(lrB0)//2)*numPackIndices, numMfma//2-1)

        # LRB3 + PACKB3 - start after 2/4 MFMAs - index 48
        halfMFMA = numMfma//2
        startLRB3 = halfMFMA
        lrB3 = create_range(startLRB3, 1, numMfma-1)
        lrB3 += create_range(max(lrB3)+6, 1, numMfma-1)
        waitLRB3 = startLRB3 + 4
        packB3 = create_range(waitLRB3, (len(lrB3)//2)*numPackIndices, numMfma-1)

        # LRA3 + PACKA3 - start after 3/4 MFMAs - index 72
        startLRA3 = (3*numMfma)//4
        lrA3 = create_range(startLRA3, 4, numMfma-1)
        waitLRA3 = startLRA3 + 4
        packA3 = create_range(waitLRA3, (len(lrA3)//2)*numPackIndices, numMfma-1)

        syncTable = [
            waitLRA0, SWaitCnt(dscnt=inflight(lrA0, waitLRA0)-2, vlcnt=-1, vscnt=-1, comment="wait for 1st 2 LRA0 to complete"),
            waitLRA0+numPackIndices, SWaitCnt(dscnt=inflight(lrA0, waitLRA0+numPackIndices), vlcnt=-1, vscnt=-1, comment="wait for all LRA0 to complete"),

            waitLRB0, SWaitCnt(dscnt=inflight(lrB0, waitLRB0)-2, vlcnt=-1, vscnt=-1, comment="wait for 1st 2 LRB0 to complete"),
            waitLRB0+numPackIndices, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for all LRB0 to complete"),
            waitLRB0+numPackIndices, SBarrier(comment="Barrier before GRA&GRB"),

            startLRB3-1, SWaitCnt(dscnt=-1, vlcnt=6, vscnt=-1, comment="Wait for prev GRA&GRB"),
            startLRB3-1, SBarrier(comment=""),

            waitLRB3,SWaitCnt(dscnt=inflight(lrB3, waitLRB3)-2, vlcnt=-1, vscnt=-1, comment="Wait for 1st 2 LRB3 to complete"),
            waitLRB3+numPackIndices,SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRB3 to complete"),

            startLRA3, SWaitCnt(dscnt=-1, vlcnt=6, vscnt=-1, comment="Wait for prev GRA&GRB"),
            startLRA3, SBarrier(comment=""),

            waitLRA3, SWaitCnt(dscnt=inflight(lrA3,waitLRA3)-2, vlcnt=-1, vscnt=-1, comment="Wait for 1st 2 LRA3 to complete"),
            waitLRA3+numPackIndices, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all LRA3 to complete")
        ]

        optSchedule = {
            'SYNC'   : [syncTable[::2]],
            'GRIncA' : [[0,0,0, 1,1,1, 2,2,2]],
            'GRIncB' : [[3,3,3, 4,4,4, 5,5,5]],

            'LRA0'   : [lrA0],
            'PackA0' : [packA0],
            'LRB0'   : [lrB0],
            'PackB0' : [packB0],

            'GRA': [[48, 48, 50, 50, 52, 52, 54, 54, 66, 66, 68, 68, 70, 70, 72, 72]],
            'GRB': [[30, 32, 34, 36, 40, 42, 44, 46]],

            'LRA3'   : [lrA3],
            'PackA3' : [packA3],
            'LRB3'   : [lrB3],
            'PackB3' : [packB3],

            'LRSA': [[22]],
            'LRSB': [[22]],
            'LWSA': [[70]],
            'LWSB': [[70]],
            'LCC': [[95, 95]],
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 12 # vmcnt shift for ngl and nll
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(64, 128, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_64x128x64_TF32(kernel, useLDSTr, TLDS):
    n_mfma = 48
    optSchedule = dict()
    nglshift = nllshift = 0

    syncs = SyncSchedule()
    syncCode = []
    gr_inc_step = 0

    if isTN(kernel) and not useLDSTr and TLDS==1:
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UsePLRPack"] = True

        grinca = [0,0,1,1,2,2,3,3,4]
        grincb = [4,5,5,6,6,7,7,8,8]
        lrsa   = [23]
        lrsb   = [23]
        lwsa   = [47]
        lwsb   = [47]

        pack_a_offset = [0,0,0,0, 2,2, 3,3,3,3,
                         1,1,1,1, 2,2, 4,4,4,4
                        ]
        pack_b_offset = [0,0,0,0, 4,4, 5,5,5,5,
                         1,1,1,1, 4,4, 6,6,6,6,
                         2,2,2,2, 4,4, 7,7,7,7,
                         3,3,3,3, 4,4, 8,8,8,8
                        ]
        lra0   = [0,1,2,3]
        syncs.add(                8, dscnt=5, comment="wait for necessary LRA0 before pack to start")
        syncs.add(                10, dscnt=4, barrier=True, comment="wait for remaining LRA0 before pack to complete + barrier for GRA")
        pack_a0 = [                i+9 for i in pack_a_offset]  ## last index = 9 + 4 = 13

        lrb0   = [        4,5, 7, 9,10,11,12,13]
        syncs.add(                           14, dscnt=4, comment="wait for necessary LRB0 before pack to start")
        syncs.add(                           16, dscnt=0, comment="wait for remaining LRB0 before pack to complete")
        pack_b0 = [                          i+14 for i in pack_b_offset]  ## last index = 14 + 8 = 22

        gra    = [                 10,13,17,21] # one index for two instructions
        grb    = [                            24,28,32,35,38,41,43,45] # one index for two instructions
        num_gr = len(gra) + len(grb)

        syncs.add(                            24, vlcnt=4, barrier=True, comment="wait for previous set of global reads + barrier for GRB")

        lra1   = [24,25,26,27]
        syncs.add(                       32, dscnt=5, comment="wait for necessary LRA1 before pack to start")
        syncs.add(                       34, dscnt=4, comment="wait for remaining LRA1 before pack to complete")
        pack_a1 = [                         i+33 for i in pack_a_offset]  ## last index = 33 + 4 = 37

        lrb1   = [           28,29, 31,32, 34,35,36,37]
        syncs.add(                                     38, dscnt=4, comment="wait for necessary LRB1 before pack to start")
        syncs.add(                                     40, dscnt=0, comment="wait for remaining LRB1 before pack to complete")
        pack_b1 = [                                    i+38 for i in pack_b_offset]  ## last index = 38 + 8 = 46

        optSchedule = {
            'SYNC':   [syncs.get_indicies()],
            'GRIncA': [grinca],
            'GRIncB': [grincb],
            'LRA0':   [lra0],
            'LRB0':   [lrb0],
            'PackA0': [pack_a0],
            'PackB0': [pack_b0],
            'GRA':    [duplicate_list_items(gra, 2, gr_inc_step)],
            'GRB':    [duplicate_list_items(grb, 2, gr_inc_step)],
            'LRSA':   [lrsa],
            'LRSB':   [lrsb],
            'LWSA':   [lwsa],
            'LWSB':   [lwsb],
            'LRA1':   [lra1],
            'LRB1':   [lrb1],
            'PackB1': [pack_b1],
            'PackA1': [pack_a1],
            'LCC':    [[n_mfma-1, n_mfma-1]],
        }

        syncCode = syncs.get_code()
        nglshift = nllshift = num_gr
    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    opt1 = ScheduleInfo(1, n_mfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1


@RegisterSchedule(
    tile_config=TileConfig(128, 64, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x64x64_TF32(kernel, useLDSTr, TLDS):
    valid, opt = _get_schedule_64x128x64_TF32(kernel, useLDSTr, TLDS)
    if not valid:
        return False, None

    optSchedule = switch_A_B_schedule(opt.optSchedule)
    return True, ScheduleInfo(opt.numCodePaths, opt.numMfma, optSchedule, opt.syncCode, opt.nglshift, opt.nllshift)

@RegisterSchedule(
    tile_config=TileConfig(160, 128, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=isTF32,
    vector_widths=[4, 4, 4],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_160x128x64_TF32(kernel, useLDSTr, TLDS):
    n_mfma = 120
    optSchedule = dict()
    nglshift = nllshift = 0

    syncs = SyncSchedule()
    syncCode = []

    if isNN(kernel) and useLDSTr and TLDS==1:
        kernel["UseMFMAF32XEmulation"] = True
        kernel["UsePLRPack"] = True
        syncs.add(11, dscnt=8, comment="wait for LRB0 before pack to complete")
        syncs.add(16, dscnt=8, barrier=True, comment="wait for LRB0 before pack to complete", barrier_comment="barrier for GRA")
        syncs.add(33, dscnt=0, comment="wait for LRA0 before pack to complete")
        syncs.add(59, vlcnt=8, barrier=True, comment="wait for previous set of global reads", barrier_comment="barrier for GRB")
        syncs.add(73, dscnt=8, comment="wait for LRB1 before pack to complete")
        syncs.add(78, dscnt=8, comment="wait for LRB1 before pack to complete")
        syncs.add(94, dscnt=0, comment="wait for LRA1 before pack to complete")

        optSchedule = {
            'SYNC': [syncs.get_indicies()],

            'GRIncB': [[0, 1, 2, 3, 4, 6, 7, 8, 9]],
            'LRB0'  : [[0,2,
                        4,5, 
                        6,7, 
                        9,11],
                                [0,1,
                                 3,5, 
                                 6,8,
                                 10,12]],

            'GRIncA': [[10, 11, 12, 13, 14, 15, 15, 16, 16]],

            'PackB0': [[13,13,14,14, 21,21, 22,22,23,23, 
                        15,15,16,16, 21,21, 24,24,25,25, 
                        17,17,18,18, 21,21, 26,26,27,27, 
                        19,19,20,20, 21,21, 28,28,29,29]],

            'LRA0'  : [[0, 0, 2, 2, 4, 4, 5, 5,
                        7, 7, 8, 8, 10,10,12,12,
                        13,13,13,13,14,14,14,14,
                        16,16,16,16,18,18,18,18,
                        20,20,20,20,22,24,26,28],
                                                [0, 0, 1, 1, 3, 3, 5, 5,
                                                 7, 7, 9, 9, 11,11,12,12, 
                                                 13,13,13,13,15,15,15,15,
                                                 17,17,17,17,19,19,19,19,
                                                 21,21,21,21,23,23,23,23]],

            'PackA0': [[30,30,31,31, 40,40, 41,41,42,42, 
                        32,32,33,33, 40,40, 43,43,44,44, 
                        34,34,35,35, 40,40, 45,45,46,46, 
                        36,36,37,37, 40,40, 47,47,48,48,
                        38,38,39,39, 40,40, 49,49,50,50]],

            'GRA'   : [[60, 60, 62, 62, 64, 64, 66, 66, 68, 68,  
                        81, 81, 83, 83, 85, 85, 87, 87, 90, 90]],

            'GRB'   : [[17, 17, 22, 22, 27, 27, 32, 32, 42, 42, 47, 47, 52, 52, 57, 57]],

            'LRSA'  : [[58]], 'LRSB'  : [[59]],'LWSA'  : [[118]], 'LWSB'  : [[118]],

            'LRB1'  : [[59,61,
                        63,65,
                        67,69,
                        71,73],
                                    [60,62,
                                     64,66,
                                     68,70,
                                     72,74]],

            'PackB1': [[73,73,74,74, 81,81, 82,82,83,83, 
                        75,75,76,76, 81,81, 84,84,85,85, 
                        77,77,78,78, 81,81, 86,86,87,87, 
                        79,79,80,80, 81,81, 88,88,89,89]],

            'LRA1'  : [[59,59,61,61, 63,63,65,65,
                        67,67,69,69, 71,71,73,73,
                        74,74,74,74, 75,75,75,75,
                        77,77,77,77, 79,79,79,79,
                        81,81,81,81, 83,83,83,83],
                                                    [60,60,62,62, 64,64,65,65,
                                                     66,66,68,68, 70,70,72,72,
                                                     74,74,74,74, 76,76,76,76,
                                                     78,78,78,78, 80,80,80,80,
                                                     81,81,81,81, 82,82,82,82]],

            'PackA1': [[90,90,91,91,            100,100, 101,101,102,102, 
                        92,92,93,93,            100,100, 103,103,104,104, 
                        94,94,95,95,            100,100, 105,105,106,106, 
                        96,96,97,97,            100,100, 107,107,108,108, 
                        98,98,99,99,            100,100, 109,109,110,110]],

            'LCC': [[118, 119]]
        }

        syncCode = syncs.get_code()
        nglshift = nllshift = len(optSchedule["GRA"][0])/2 + len(optSchedule["GRB"][0])/2

        opt1 = ScheduleInfo(2, n_mfma, optSchedule, syncCode, nglshift, nllshift)

    elif isTN(kernel) and not useLDSTr and TLDS==1:
        valid, opt = _get_schedule_128x160x64_TF32(kernel, useLDSTr, TLDS)
        if not valid:
            return False, None
        optSchedule = switch_A_B_schedule(opt.optSchedule)
        opt1 = ScheduleInfo(opt.numCodePaths, opt.numMfma, optSchedule, opt.syncCode, opt.nglshift, opt.nllshift)

    else:
        return False, None

    kernel["MfmaInitCVgprs"] = True
    return True, opt1

        
@RegisterSchedule(
    tile_config=TileConfig(128, 256, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x256x64_16bit(kernel, useLDSTr, TLDS):
    numMfma = 64
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0
    if isNN(kernel) and useLDSTr and TLDS == 1:
        lra0 = [create_range(min_val = 1, num = 4, step = 2, repeat = 2),
                create_range(min_val = 0, num = 4, step = 2, repeat = 2)]

        GRIncA = [create_range(min_val = 2, num = 3, step = 2, repeat = 3),
                  create_range(min_val = 1, num = 3, step = 2, repeat = 3)]

        waitLRA0 = max(lra0[1])+5
        gra = create_range(min_val = waitLRA0+1, num = 4, step = 2, repeat = 2)
        lrb0 = create_range(min_val = max(gra)+1, num = 8, step = 1, repeat = 1)
        GRIncB = create_range(min_val = max(gra)+1, num = 9, step = 1, repeat = 1)

        assert max(lrb0) < numMfma // 2, "lrb0 max {} numMfma/2 {}".format(max(lrb0), numMfma//2)

        startGRB = max(lrb0) + 5

        assert startGRB < numMfma // 2, "startGRB {} numMfma/2 {}".format(startGRB, numMfma//2)
        grb = create_range(min_val = startGRB, num = 4, step = 2, repeat = 2)
        startLRA1 = max(grb) + 3

        lra1 = create_range(min_val = startLRA1, num = 8, step = 1, repeat = 1)
        startLRB1 = max(lra1) + 1
        grb += create_range(min_val = startLRB1, num = 4, step = 2, repeat = 2)
        lrb1 = create_range(min_val = startLRB1+1, num = 4, step = 2, repeat = 1)
        lrb1 += create_range(min_val = max(lrb1)+2, num = 4, step = 1, repeat = 1)
        syncTable = [
            -1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0 & LRB0"),
            waitLRA0,  SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRA0"),
            waitLRA0, SBarrier(comment=""),

            startGRB-1, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0"),
            startGRB-1, SBarrier(comment=""),
            startLRA1-1, SWaitCnt(dscnt=-1, vlcnt=16, vscnt=-1, comment="wait for previous GRA & GRB"),
            startLRA1-1, SBarrier(comment=""),

            startLRB1-1, SWaitCnt(dscnt=-1, vlcnt=8, vscnt=-1, comment="wait for previous GRA & GRB"),
            startLRB1-1, SBarrier(comment="")
        ]

        optSchedule = {
            'GRA': [gra],
            'GRB': [grb],
            'GRIncA': [*GRIncA],
            'GRIncB': [GRIncB],
            'LCC': [[numMfma-2,numMfma-2]],
            'LRA0': [*lra0],
            'LRA1': [lra1],
            'LRB0': [lrb0],
            'LRB1': [lrb1],
            'LRSA': [[startGRB-1]],
            'LRSB': [[startGRB-1]],
            'LWSA': [[numMfma-3]],
            'LWSB': [[numMfma-3]],
            'SYNC': [syncTable[::2]],
        }

        syncCode = syncTable[1::2]
        nglshift = nllshift = 12 
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    else:
        # No matching variant found
        return False, None

    kernel["MfmaInitCVgprs"] = True
    return True, opt1


@RegisterSchedule(
    tile_config=TileConfig(224, 320, 64, 2, 1, 1, False, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_224x320x64_16bit(kernel, useLDSTr, TLDS):
    numMfma = 140
    optSchedule = dict()
    syncCode = []
    nglshift = nllshift = 0 # vmcnt shift for ngl and nll
    kernel["MfmaInitCVgprs"] = True
    kernel["SwapGlobalReadOrder"] = False

    if isTN(kernel) and useLDSTr and TLDS==1:
        syncTable = [
            -1, SWaitCnt(dscnt=9, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write old=0, new=9 newLW=0 newLR=9 for iteration == 0"),
            6, SWaitCnt(dscnt=7, vlcnt=-1, vscnt=-1, comment="wait for prior local read local write"),
            25, SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="wait for LR0 before DTL"),
            25, SBarrier(comment=""),
            78, SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="wait for prev iter GR before LRA1 and LRB1"),
            78, SBarrier(comment=""),
        ]
        optSchedule = {
            'SYNC': [syncTable[::2]],
            'GRIncA': [[0, 1, 2, 3, 4, 5, 6, 7, 8]], # 9
            'GRIncB': [[9, 10, 11, 12, 13, 14, 15, 16, 17]], # 9

            'LRA0': [[0, 3, 6, 9, 12, 15, 18]], # 7
            'LRB0': [[1, 2, 4, 5, 7, 8, 10, 11, 13, 17]], # 10

            'GRA': [[25,25, 30,30, 36,36, 42,42, 48,48, 54,54, 60,60]], # 14
            'GRB': [[62,62, 67,67, 72,72, 77,77, 88,88, 94,94, 100,100, 106,106, 112,112, 118,118]], # 20

            'LRA1': [[79, 81, 82, 83, 84, 85, 86]], # 7
            'LRB1': [[120, 121, 122, 123, 124, 125, 126, 127, 128, 129]], # 10

            'LRSA': [[66]], # 1
            'LRSB': [[66]], # 1
            'LWSA': [[118]], # 1
            'LWSB': [[118]], # 1
            'LCC': [[138, 138]], # 2
        }
        syncCode = syncTable[1::2]
        nglshift = nllshift = 17

    else:
        return False, None

    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
