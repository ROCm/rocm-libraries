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

from rocisa.code import Module, TextBlock, StructuredModule, KernelBody
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

class ScheduleInfo:
    numCodePaths: int
    numMfma: int
    def __init__(self, numCodePaths, numMfma, optSchedule, syncCode, mfmaReorder = []):
        self.numCodePaths = numCodePaths
        self.numMfma = numMfma
        self.optSchedule = optSchedule
        self.syncCode = syncCode
        self.mfmaReorder = mfmaReorder

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

    def convOptToStream(sched):
        InstStreams = dict()
        def addToStream(key, indexList, InstructionList):
            if indexList:
                for l in indexList:
                    assert len(l) == len(InstructionList), \
                        "Index list length (%u) for %s does not match instruction list length (%u)"%(len(l), key, len(InstructionList))
            InstStreams[key] = [indexList, InstructionList]

        for key,stream in opt1.optSchedule.items():
            addToStream(key, stream, idMap[key])

        return InstStreams

    InstStreams = convOptToStream(opt1)


    module.add(TextBlock(".macro MAINLOOP ID useGR=1 usePLR=1 useGRInc=1 useLoop=1\n"))
    #module.add(SBarrier(comment="debug"))

    lastIter = numLoopIter - 1

    for miIndex in range(-1, len(mfmaCode)):
        if miIndex >= 0:
            module.addComment0("mfmaIndex:%u"%(miIndex))
            module.add(mfmaCode[miIndex])

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

        def scheduleInst1(instList, macroGuard=""):
            if len(instList) == 1:
                if instList[0] != None:
                    if macroGuard != "":
                        module.add(TextBlock(macroGuard))
                    module.add(instList[0])
                    if macroGuard != "":
                        module.add(TextBlock(".endif\n"))

        for k,ts in ToSched.items():
            if k in ['GRIncA', 'GRIncB']: # check for global read inc
                scheduleInst1(ts, ".if     \\useGRInc == 1\n")
            elif k in ['GRA', 'GRB', 'LWSA', 'LWSB']: # check for global reads
                scheduleInst1(ts, ".if     \\useGR == 1\n")
            elif k in ['LRA%u'%lastIter, 'LRB%u'%lastIter, 'LRSA', 'LRSB']: # check for next prefetch
                scheduleInst1(ts, ".if     \\usePLR == 1\n")
            elif k in ['LCC']: # check for next prefetch
                scheduleInst1(ts, ".if     \\useLoop == 1\n")
            else:
                scheduleInst1(ts)

        if needIfMacro:
            for codepath in range(numCodePath):
                if codepath == 0:
                    module.add(TextBlock(".if     \\ID == %u\n"%codepath))
                else:
                    module.add(TextBlock(".elseif \\ID == %u\n"%codepath))

                def scheduleInst2(instList, macroGuard=""):
                    if len(instList) == numCodePath:
                        if instList[codepath] != None:
                            if macroGuard != "":
                                module.add(TextBlock(macroGuard))
                            module.add(instList[codepath])
                            if macroGuard != "":
                                module.add(TextBlock(".endif\n"))

                for k,ts in ToSched.items():
                    if k in ['GRIncA', 'GRIncB']: # check for global read inc
                        scheduleInst2(ts, ".if     \\useGRInc == 1\n")
                    elif k in ['GRA', 'GRB', 'LWSA', 'LWSB']: # check for global reads
                        scheduleInst2(ts, ".if     \\useGR == 1\n")
                    elif k in ['LRA%u'%lastIter, 'LRB%u'%lastIter, 'LRSA', 'LRSB']: # check for next prefetch
                        scheduleInst2(ts, ".if     \\usePLR == 1\n")
                    elif k in ['LCC']: # check for next prefetch
                        scheduleInst2(ts, ".if     \\useLoop == 1\n")
                    else:
                        scheduleInst2(ts)

                if codepath == numCodePath - 1:
                    module.add(TextBlock(".endif\n"))

    module.add(TextBlock(".endm\n"))
    return module, numCodePath


def hasCustomSchedule(kernel):

    if not kernel["UseCustomMainLoopSchedule"]:
        return False, None
    # Only support kernels using matrix instructions for now
    if not kernel["EnableMatrixInstruction"]:
        return False, None
    if not kernel["ISA"] == IsaVersion(9,5,0):
        return False, None

    is16bit = kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()
    is8bit = kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat()
    isMixed = kernel["ProblemType"]["DataTypeA"].numBytes() != kernel["ProblemType"]["DataTypeB"].numBytes()

    MT0, MT1, DU, PGR, PLR, DTL = kernel["MacroTile0"], kernel["MacroTile1"], kernel["DepthU"], kernel["PrefetchGlobalRead"], kernel["PrefetchLocalRead"], kernel["DirectToLds"]
    GRVWA, GRVWB = kernel["GlobalReadVectorWidthA"], kernel["GlobalReadVectorWidthB"]
    LRVW = kernel["LocalReadVectorWidth"]
    MI = kernel["MatrixInstruction"]
    MIWG = kernel["MIWaveGroup"]
    useLDSTr = kernel["LDSTrInst"]
    TLDS = kernel["TransposeLDS"]

    is256x256x64DTL  = [MT0, MT1, DU, PGR, PLR, DTL] == [256, 256, 64, 2, 1, True]
    is192x256x64DTL  = [MT0, MT1, DU, PGR, PLR, DTL] == [192, 256, 64, 2, 1, True]
    is256x256x128DTL = [MT0, MT1, DU, PGR, PLR, DTL] == [256, 256, 128, 2, 0, True]


    transA = kernel["ProblemType"]["TransposeA"]
    transB = kernel["ProblemType"]["TransposeB"]

    isNN = transA == False and transB == False
    isNT = transA == False and transB == True
    isTT = transA == True and transB == True
    isTN = transA == True and transB == False

    # Custom main loop scheduling for 256x256x64 16bit
    if is256x256x64DTL and is16bit and not isMixed and ([GRVWA, GRVWB, LRVW] == [8,8,8]) and MI == [16,16,32,1] and MIWG == [2,2]:

        kernel["MfmaInitCVgprs"] = True

        optSchedule = dict()
        syncCode = []

        if isTN and TLDS == 1:
            optSchedule = {
                'SYNC'   : [[19,20, 50,51, 67,68, 104, 105]],
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
                        SWaitCnt(dscnt=-1, vlcnt=15, vscnt=-1, comment="Wait for previous GRA to completely"),
                        SBarrier(comment="")]
        elif isNT and not useLDSTr and TLDS == 0:
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
        elif (isNN or isTT) and not useLDSTr and TLDS == 1:
            kernel["UsePLRPack"] = True

            optSchedule = {
                'SYNC'   : [[8, 12,13, 36,44, 56,59, 66,68, 73]],
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
                        SWaitCnt(dscnt=4, vlcnt=-1, vscnt=-1, comment="Wait for LRA1 to complete")]
            if isTT:
                kernel["SwapGlobalReadOrder"] = True

                optSchedule['GRIncA'], optSchedule['GRIncB'] = optSchedule['GRIncB'], optSchedule['GRIncA']
                optSchedule['LRA0'], optSchedule['LRB0'] = optSchedule['LRB0'], optSchedule['LRA0']
                optSchedule['LRA1'], optSchedule['LRB1'] = optSchedule['LRB1'], optSchedule['LRA1']
                optSchedule['PackB0'] = optSchedule['PackA0']
                optSchedule['PackB1'] = optSchedule['PackA1']
                del optSchedule['PackA0'], optSchedule['PackA1']
        else:
            return False, None


        numMfma = 128
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode)
        return True, opt1
    elif is256x256x128DTL and is8bit and not isMixed and ([GRVWA, GRVWB, LRVW] == [16, 16, 16]) and MI == [16,16,128,1] and MIWG == [2,2]:

        kernel["MfmaInitCVgprs"] = True

        optSchedule = dict()
        syncCode = []

        plr = 3 if kernel["ForceUnrollSubIter"] else 1

        if isTN and TLDS == 1:
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
        opt1 = ScheduleInfo(1, numMfma, optSchedule, syncCode, mfmaReorder)
        return True, opt1
    elif is192x256x64DTL and is16bit and not isMixed and ([GRVWA, GRVWB, LRVW] == [8, 8, 8]) and MI == [16,16,32,1] and MIWG == [2,2]:

        kernel["MfmaInitCVgprs"] = True

        optSchedule = dict()
        syncCode = []
        if isNN and useLDSTr and TLDS==1:
            # Note: A/B Global read orders are swapped
            # i.e. GRA contains GR for B
            optSchedule = {
                'SYNC'    : [[20,21,23,25,27,29,31,33,46,57,58,94],
                             [20,21,24,26,28,30,32,34,47,58,58,94]],
                'GRIncA' : [[0,1,2,3,4,5,6,7,8]],
                'GRIncB' : [[9,10,11,12,13,14,15,16,17]],
                'LRB0'    : [[0,0,1,1,2,2,6,8],
                             [3,3,4,4,5,5,7,9]],
                'LRA0'    : [[10,12,14,16,18,23,35,37,39,41,43,45],
                             [11,13,15,17,19,22,36,38,40,42,44,46]],
                'LWA'     : [[23,25,27,29,31,33],
                             [24,26,28,30,32,34]],
                'GRA'     : [[22,22,24,24,26,26,28,28,30,30, 42,42,43,43,45,45],
                             [23,23,25,25,27,27,29,29,31,31, 43,43,44,44,46,46]],
                'GRB'     : [[54, 56, 58, 60, 62, 64],
                             [55, 57, 59, 61, 63, 65]],
                'LRSA'   : [[47]],
                'LRSB'   : [[37]],
                'LWSB'   : [[47]], # For B
                'LWSA'   : [[52]], # For A
                'LRB1'    : [[59,59,61,61,63,63,65,67],
                             [60,60,62,62,64,64,66,68]],
                'LRA1'    : [[69,71,73,75,77,79,81,83,85,85,87,87],
                             [70,72,74,76,78,80,82,84,86,86,88,88]],
                'LCC'    : [[95, 95]],
            }
            syncCode = [SWaitCnt(dscnt=5, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
                        SBarrier(comment=""),
                        SWaitCnt(dscnt=-1, vlcnt=5, vscnt=-1, comment="Wait for LRB0 to complete"),
                        SWaitCnt(dscnt=-1, vlcnt=5, vscnt=-1, comment="Wait for LRB0 to complete"),
                        SWaitCnt(dscnt=-1, vlcnt=5, vscnt=-1, comment="Wait for LRB0 to complete"),
                        SWaitCnt(dscnt=-1, vlcnt=5, vscnt=-1, comment="Wait for LRB0 to complete"),
                        SWaitCnt(dscnt=-1, vlcnt=5, vscnt=-1, comment="Wait for LRB0 to complete"),
                        SWaitCnt(dscnt=-1, vlcnt=5, vscnt=-1, comment="Wait for LRB0 to complete"),
                        SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),
                        SWaitCnt(dscnt=-1, vlcnt=10, vscnt=-1, comment="Wait for LRB0 to complete"),
                        SBarrier(comment=""),
                        SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LRB0 to complete"),]

        else:
            return False, None

        numMfma = 96
        opt1 = ScheduleInfo(2, numMfma, optSchedule, syncCode)
        return True, opt1

    return False, None
