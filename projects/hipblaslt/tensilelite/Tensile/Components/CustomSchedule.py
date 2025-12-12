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

from dataclasses import dataclass, field
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
from Tensile.Utilities.Decorators.Shared import CallableGuard

from copy import deepcopy
from typing import Callable
from enum import Enum, auto
import Tensile.Components.CMSValidator as cmsv

# Enum to distinguish between different schedule matching outcomes
class ScheduleMatchStatus(Enum):
    FOUND = auto()                  # Schedule found and supported
    NO_MATCH = auto()               # Criteria don't match, continue searching
    UNSUPPORTED_VARIANT = auto()    # Criteria match but variant unsupported, stop searching

# Global registry for schedule functions
_SCHEDULE_REGISTRY = []

@dataclass
class SyncSchedule:
    schedule : list[tuple[int, SWaitCnt | SBarrier]] = field(default_factory=list)

    def add(self, idx:int, dscnt:int=-1, vlcnt:int=-1, vscnt:int=-1, comment:str="", barrier:bool=False, barrier_idx:int|None=None, barrier_comment:str=""):
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

def duplicate_list_items(input_list: list, repeat_count: int, step:int=0) -> list:
    """
    Duplicate each item in input_list repeat_count times. Optionally duplicate with a step

    Example:
        duplicate_list_items([1, 2, 3], 3)    => [1,1,1, 2,2,2, 3,3,3]
        duplicate_list_items([1, 2, 3], 3, 1) => [1,2,3, 2,3,4, 3,4,5]
    """
    return [item + step * j for item in input_list for j in range(repeat_count)]

def count_items(input_list: list[int], sv:int|None = None, ev:int|None = None):
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


class ScheduleInfo:
    numCodePaths: int
    numMfma: int
    __skipValidation__: bool

    def __init__(
        self,
        numCodePaths,
        numMfma,
        optSchedule,
        syncCode,
        nglshift,
        nllshift,
        nllZeroDscnt=False,
        mfmaReorder=[],
    ):
        self.numCodePaths = numCodePaths
        self.numMfma = numMfma
        self.optSchedule = optSchedule
        self.syncCode = syncCode
        self.nglshift = nglshift  # vmcnt shift for noglobalload loop
        self.nllshift = nllshift  # vmcnt shift for nolocalload loop
        self.nllZeroDscnt = nllZeroDscnt
        self.mfmaReorder = mfmaReorder
        self.__skipValidation__ = False

    def disableValidation(self):
        self.__skipValidation__ = True


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
    assert status is True, f"Custom mainloop schedule validation failed for kernel {kernel['MacroTile0']}x{kernel['MacroTile1']}x{kernel['DepthU']} {case_str}: {message}"

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

        needIfMacro = False
        ToSched = dict()
        for k, stream in InstStreams.items():
            ToSched[k] = scheduleInst(stream[0], stream[1])
            if len(ToSched[k]) > 1:
                needIfMacro = True

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

        def scheduleInst1(instList, macroGuard=""):
            if len(instList) == 1:
                if instList[0] != None:
                    for inst in instList[0].flatitems():
                        if isinstance(inst, SWaitCnt):
                            nllvmcntHandling(inst, opt1.nglshift, opt1.nllshift)
                        else:
                            if macroGuard != "":
                                macro.add(ValueIf(macroGuard))
                            macro.add(inst)
                            if macroGuard != "":
                                macro.add(ValueEndif(comment="EndIf %s"%(macroGuard)))

        for k,ts in ToSched.items():
            if k in ['GRIncA', 'GRIncB']: # check for global read inc
                scheduleInst1(ts, "\\useGRInc == 1")
            elif k in ['GRA', 'GRB', 'LWSA', 'LWSB']: # check for global reads
                scheduleInst1(ts, "\\useGR == 1")
            elif k in ['LRA%u'%lastIter, 'LRB%u'%lastIter, 'LRSA', 'LRSB']: # check for next prefetch
                scheduleInst1(ts, "\\usePLR == 1")
            elif k in ['LCC']: # check for next prefetch
                scheduleInst1(ts, "\\useLoop == 1")
            else:
                scheduleInst1(ts)

        if needIfMacro:
            for codepath in range(numCodePath):
                if codepath == 0:
                    macro.add(ValueIf("\\ID == %u"%codepath))
                else:
                    macro.add(ValueElseIf("\\ID == %u\n"%codepath))

                def scheduleInst2(instList, macroGuard=""):
                    if len(instList) == numCodePath:
                        if instList[codepath] != None:
                            for inst in instList[codepath].flatitems():
                                if isinstance(inst, SWaitCnt):
                                    nllvmcntHandling(inst, opt1.nglshift, opt1.nllshift)
                                else:
                                    if macroGuard != "":
                                        macro.add(ValueIf(macroGuard))
                                    macro.add(inst)
                                    if macroGuard != "":
                                        macro.add(ValueEndif(comment="EndIf %s"%(macroGuard)))

                for k,ts in ToSched.items():
                    if k in ['GRIncA', 'GRIncB']: # check for global read inc
                        scheduleInst2(ts, "\\useGRInc == 1\n")
                    elif k in ['GRA', 'GRB', 'LWSA', 'LWSB']: # check for global reads
                        scheduleInst2(ts, "\\useGR == 1\n")
                    elif k in ['LRA%u'%lastIter, 'LRB%u'%lastIter, 'LRSA', 'LRSB']: # check for next prefetch
                        scheduleInst2(ts, "\\usePLR == 1\n")
                    elif k in ['LCC']: # check for next prefetch
                        scheduleInst2(ts, "\\useLoop == 1\n")
                    else:
                        scheduleInst2(ts)

                if codepath == numCodePath - 1:
                    macro.add(ValueEndif(comment="EndIf \\ID checks"))

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

@CallableGuard
def is8bit(kernel):
    return kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat()

@CallableGuard
def isMixed(kernel):
    return kernel["ProblemType"]["DataTypeA"].numBytes() != kernel["ProblemType"]["DataTypeB"].numBytes()

@dataclass(frozen=True)
class TileConfig:
    macro_tile_size_0: int
    macro_tile_size_1: int
    depth_u: int
    prefetch_global_read: int
    prefetch_local_read: int
    direct_to_lds: bool
    wave_separate_global_read_a: int
    wave_separate_global_read_b: int

class RegisterSchedule:
    """
    Decorator that registers a schedule function with its matching criteria.
    The function is wrapped with logic that checks if the kernel matches the criteria.
    
    Usage:
        @RegisterSchedule(
            tile_config=TileConfig(256, 96, 64, 2, 1, True, 0, 0),
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
    
    def __call__(self, func: Callable) -> Callable:
        """Wrap the function with matching logic and register it."""
        def wrapped_func(kernel: dict, useLDSTr: bool, TLDS: int) -> tuple[ScheduleMatchStatus, ScheduleInfo | None]:
            # TODO: Currently ULSGRO not checked for in CMS, disabled for now
            if kernel["UnrollLoopSwapGlobalReadOrder"]:
                return ScheduleMatchStatus.NO_MATCH, None

            if not self.dtype_predicate(kernel):
                return ScheduleMatchStatus.NO_MATCH, None

            MT0, MT1, DU = kernel["MacroTile0"], kernel["MacroTile1"], kernel["DepthU"]
            PGR, PLR, DTL = kernel["PrefetchGlobalRead"], kernel["PrefetchLocalRead"], kernel["DirectToLds"]
            WSGRA, WSGRB = kernel["WaveSeparateGlobalReadA"], kernel["WaveSeparateGlobalReadB"]
            kernel_tile_config = TileConfig(MT0, MT1, DU, PGR, PLR, DTL, WSGRA, WSGRB)
            if self.tile_config != kernel_tile_config:
                return ScheduleMatchStatus.NO_MATCH, None

            GRVWA, GRVWB = kernel["GlobalReadVectorWidthA"], kernel["GlobalReadVectorWidthB"]
            LRVW = kernel["LocalReadVectorWidth"]
            kernel_vector_widths = [GRVWA, GRVWB, LRVW]            
            if self.vector_widths != kernel_vector_widths:
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
        
        # Return original function unchanged (so it can still be called directly)
        return func


@RegisterSchedule(
    tile_config=TileConfig(256, 96, 64, 2, 1, True, 0, 0),
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

    else:
        return False, None


    numMfma = 48
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(192, 256, 64, 2, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_192x256x64_16bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True

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

        kernel["UsePLRPack"] = True

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

    numMfma = 96
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 192, 64, 2, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x192x64_16bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True
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

    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 256, 128, 2, 0, True, 0, 0),
    dtype_predicate=is8bit,
    vector_widths=[16, 16, 16],
    matrix_inst=[16, 16, 128, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x256x128_8bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True

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
    if not kernel["ForceUnrollSubIter"]:
        mfmaReorder = [0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27,
                       4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31,
                       32,33,34,35, 40,41,42,43, 48,49,50,51, 56,57,58,59,
                       36,37,38,39, 44,45,46,47, 52,53,54,55, 60,61,62,63]
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift, mfmaReorder)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 256, 64, 2, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x256x64_16bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True

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
            'SYNC'   : [[8, 12,13, 36,44, 56,59, 66,68, 74, 85, 127]],
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
                    SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for 6/8 LRA1 to complete"),
                    SWaitCnt(dscnt=8, vlcnt=-1, vscnt=-1, comment="Wait for all LRA1 to complete"),
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

    numMfma = 128
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(160, 256, 64, 2, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_160x256x64_16bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True
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


    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 160, 64, 2, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x160x64_16bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True
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

    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 240, 64, 2, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 2, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[4, 1]
)
def _get_schedule_256x240x64_16bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True
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


    numMfma = 120  # Must match actual MFMA count for 256x240x64 tile
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 208, 64, 2, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 2, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[4, 1]
)
def _get_schedule_256x208x64_16bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True
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

    numMfma = 104
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(224, 256, 64, 2, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_224x256x64_16bit(kernel, userLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True
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
    else:
        return False, None
    numMfma = 112
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(192, 320, 64, 2, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_192x320x64_16bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True
    kernel["SwapGlobalReadOrder"] = False
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
    syncCode = syncs.get_code()
    nglshift = nllshift = num_gr
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift, nllZeroDscnt)

    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(256, 224, 64, 2, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_256x224x64_16bit(kernel, userLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True
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
    else:
        return False, None
    numMfma = 112
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(320, 192, 64, 2, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_320x192x64_16bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True
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

    numMfma = 120
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(240, 256, 64, 2, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[2, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[1, 4]
)
def _get_schedule_240x256x64_16bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True
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
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(208, 256, 64, 2, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[2, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[1, 4]
)
def _get_schedule_208x256x64_16bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True
    kernel["SwapGlobalReadOrder"] = False
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
    opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1

@RegisterSchedule(
    tile_config=TileConfig(128, 224, 64, 2, 1, True, 0, 0),
    dtype_predicate=is16bit,
    vector_widths=[8, 8, 8],
    matrix_inst=[16, 16, 32, 1],
    mfma_wave_group=[2, 2]
)
def _get_schedule_128x224x64_16bit(kernel, useLDSTr, TLDS):
    kernel["MfmaInitCVgprs"] = True

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
    else:
        return False, None
    numMfma = 56
    opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode, nglshift, nllshift)
    return True, opt1
