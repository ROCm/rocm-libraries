################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

from rocisa import countInstruction, countGlobalRead, countLocalWrite, \
                   countDSStoreB192, countDSStoreB128, countVMovB32
from rocisa.base import Item
from rocisa.code import Module, TextBlock
from rocisa.container import DSModifiers, HolderContainer, replaceHolder

from rocisa.instruction import SWaitCnt, SWaitAlu, DSStoreB128, DSStoreB64, DSStoreB32, TensorLoadToLds

from ..Common import roundUp, print2
from ..Component import SIA

from copy import deepcopy
from typing import Tuple
PRECISION = 100
class SIA3(SIA):
    kernel = {"ScheduleIterAlg": 3}
    def __call__(self):
        assert(0)

    def schedIntoIteration(self, writer, kernel, tensorParametersA, tensorParametersB, localWriteEndIter, firstIter, lastLoop, lastLc, globalReadIncACode, globalReadIncBCode, isNGLL):
        # Get schedule information
        numMfmaBetweenLWandBarrier, latencyLeft = getLocalWriteMFMAEnd(writer, kernel, tensorParametersA, tensorParametersB)
        #########
        # Internally assign an optimized LWPM value for PGR2
        #########
        # strategy is to distribute LW/GR as wide as possible to avoid hitting vmem FIFO
        # LWPM = (LW_End - LW_Start) / numLW
        if kernel["LocalWritePerMfma"] == -1:
            lwStartMfmaIndex = getLocalWriteMFMAStart(writer, kernel, tensorParametersA, tensorParametersB, latencyLeft)
            numLocalWriteModPerMfma = getNumLocalWritePerMfma(writer, kernel, lwStartMfmaIndex)
        else:
            numLocalWriteModPerMfma = roundUp(kernel["LocalWritePerMfma"]*PRECISION)
            # safe guard
            # should not be smaller than the value calculated with lwStartMfmaIndex=0
            numLocalWriteModPerMfmaMin = getNumLocalWritePerMfma(writer, kernel, 0)
            numLocalWriteModPerMfma = max(numLocalWriteModPerMfma, numLocalWriteModPerMfmaMin)

        writer.states.numGlobalReadInsPerMfma, writer.states.numLocalWriteModPerMfma = calculateGRPMandLWPM(writer, kernel, numLocalWriteModPerMfma)
        localWriteEndIter = fixLocalWriteEndMfmaIndex(writer, kernel, tensorParametersA, tensorParametersB, \
            globalReadIncACode, globalReadIncBCode, numMfmaBetweenLWandBarrier, lastLoop)
        numGlobalReadInsPerIter, numLocalWriteModPerIter, numEmptyGlobalReadIncCode = getScheduleParamMfma(writer)
        numLocalWritesPerSched = writer.states.numLocalWriteModPerMfma
        # Schedule global read
        if not writer.states.scheduleGlobalRead:
            itemsGRToSchedLater, lastLoadIter = noSchedGlobalRead(writer, kernel, globalReadIncACode, globalReadIncBCode)
            writer.states.grEndMfmaIndex = 0
        else:
            itemsGRToSched, itemsGRToSchedLater = prepareGRInstToSched(writer, kernel, isNGLL)
            itemsGRIncToSched = appendInstToSchedSIA3(writer, kernel, numEmptyGlobalReadIncCode, globalReadIncACode, globalReadIncBCode)
            schedNumForIter0, endIter = getSchedNumForIter0SIA3(writer, kernel, itemsGRToSched, itemsGRIncToSched, numGlobalReadInsPerIter)
            lastLoadIter = schedGlobalRead(writer, itemsGRToSched, itemsGRIncToSched, numGlobalReadInsPerIter, schedNumForIter0, endIter)
        # Schedule local write
        if not writer.states.scheduleLocalWrite:
            noSchedLocalWrite(writer, kernel, tensorParametersA, tensorParametersB, localWriteEndIter)
            writer.states.lwStartMfmaIndex = writer.states.lwEndMfmaIndex
        else:
            itemsLWToSched, numWritesToSched = prepareLWInstToSched(writer, kernel, numLocalWritesPerSched, isNGLL=isNGLL)
            startIter = assignLWSchedIndexSIA3(writer, kernel, numLocalWritesPerSched, localWriteEndIter, numWritesToSched)
            readsToWait, readsToWaitNGLL = getReadsToWait(writer, kernel)
            # add waitcnt for DirectToVgpr + PGR1. Delaying wait for DirectToVgpr global read
            if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and kernel["PrefetchGlobalRead"] == 1:
              # DirectToVgpr + swapGlobalRead case, actual DTV load is in self.globalReadBCode (due to swap).
              # Need to check self.globalReadBCode
              readsToWaitDTV = len(list(writer.codes.globalReadB.middle.items()))
              readsToWait += readsToWaitDTV
              readsToWaitNGLL += readsToWaitDTV
            # make sure numLocalWriteModPerIter is enough to schedule localwrite
            startIterItem = numLocalWriteModPerIter - (writer.states.lwStartMfmaIndex % writer.states.numMfmaPerIter) * numLocalWritesPerSched
            schedLocalWrite(writer, kernel, numLocalWriteModPerIter, numLocalWritesPerSched, localWriteEndIter, \
              itemsGRToSchedLater, itemsLWToSched, startIter, readsToWait, readsToWaitNGLL, \
              firstIter, lastLc, isNGLL, startIterItem)

class SIA2(SIA):
    kernel = {"ScheduleIterAlg": 2}
    def __call__(self):
        assert(0)

    def schedIntoIteration(self, writer, kernel, tensorParametersA, tensorParametersB, localWriteEndIter, firstIter, lastLoop, lastLc, globalReadIncACode, globalReadIncBCode, isNGLL):
        # Get schedule information
        numGlobalReadInsPerIter, numLocalWriteModPerIter, numEmptyGlobalReadIncCode = getScheduleParams(kernel)
        numLocalWritesPerSched = numLocalWriteModPerIter
        # Schedule global read
        if not writer.states.scheduleGlobalRead:
            itemsGRToSchedLater, lastLoadIter = noSchedGlobalRead(writer, kernel, globalReadIncACode, globalReadIncBCode)
        else:
            itemsGRToSched, itemsGRToSchedLater = prepareGRInstToSched(writer, kernel, isNGLL)
            itemsGRIncToSched = appendInstToSchedDefault(numEmptyGlobalReadIncCode, globalReadIncACode, globalReadIncBCode)
            schedNumForIter0, endIter = getSchedNumForIter0Default(itemsGRToSched, itemsGRIncToSched, numGlobalReadInsPerIter, localWriteEndIter)
            lastLoadIter = schedGlobalRead(writer, itemsGRToSched, itemsGRIncToSched, numGlobalReadInsPerIter, schedNumForIter0, endIter)
        # Schedule local write
        if not writer.states.scheduleLocalWrite:
            noSchedLocalWrite(writer, kernel, tensorParametersA, tensorParametersB, localWriteEndIter)
        else:
            itemsLWToSched, numWritesToSched = prepareLWInstToSched(writer, kernel, numLocalWritesPerSched)
            startIter = assignLWSchedIndexDefault(writer, kernel, numLocalWritesPerSched, localWriteEndIter, lastLoadIter, numWritesToSched)
            readsToWait, readsToWaitNGLL = getReadsToWait(writer, kernel)
            schedLocalWrite(writer, kernel, numLocalWriteModPerIter, numLocalWritesPerSched, localWriteEndIter, \
              itemsGRToSchedLater, itemsLWToSched, startIter, readsToWait, readsToWaitNGLL, \
              firstIter, lastLc, isNGLL)

class SIA1(SIA):
    kernel = {"ScheduleIterAlg": 1}
    def __call__(self):
        assert(0)

    def schedIntoIteration(self, writer, kernel, tensorParametersA, tensorParametersB, localWriteEndIter, firstIter, lastLoop, lastLc, globalReadIncACode, globalReadIncBCode, isNGLL):
        # Get schedule information
        numGlobalReadInsPerIter, numLocalWriteModPerIter, numEmptyGlobalReadIncCode = getScheduleParams(kernel)
        numLocalWritesPerSched = numLocalWriteModPerIter
        # Schedule global read
        if not writer.states.scheduleGlobalRead:
            itemsGRToSchedLater, lastLoadIter = noSchedGlobalRead(writer, kernel, globalReadIncACode, globalReadIncBCode)
        else:
            itemsGRToSched, itemsGRToSchedLater = prepareGRInstToSched(writer, kernel, isNGLL)
            itemsGRIncToSched = appendInstToSchedDefault(numEmptyGlobalReadIncCode, globalReadIncACode, globalReadIncBCode)
            schedNumForIter0, endIter = getSchedNumForIter0Default(itemsGRToSched, itemsGRIncToSched, numGlobalReadInsPerIter, localWriteEndIter)
            lastLoadIter = schedGlobalRead(writer, itemsGRToSched, itemsGRIncToSched, numGlobalReadInsPerIter, schedNumForIter0, endIter)
        # Schedule local write
        if not writer.states.scheduleLocalWrite:
            noSchedLocalWrite(writer, kernel, tensorParametersA, tensorParametersB, localWriteEndIter)
        else:
            itemsLWToSched, numWritesToSched = prepareLWInstToSched(writer, kernel, numLocalWritesPerSched)
            startIter = assignLWSchedIndexDefault(writer, kernel, numLocalWritesPerSched, localWriteEndIter, lastLoadIter, numWritesToSched)
            readsToWait, readsToWaitNGLL = getReadsToWait(writer, kernel)
            schedLocalWrite(writer, kernel, numLocalWriteModPerIter, numLocalWritesPerSched, localWriteEndIter, \
              itemsGRToSchedLater, itemsLWToSched, startIter, readsToWait, readsToWaitNGLL, \
              firstIter, lastLc, isNGLL)

class SIA0(SIA):
    kernel = {"ScheduleIterAlg": 0}
    def __call__(self):
        assert False

    def schedIntoIteration(self, writer, kernel, tensorParametersA, tensorParametersB, localWriteEndIter, firstIter, lastLoop, lastLc, globalReadIncACode, globalReadIncBCode, isNGLL):
        # Get schedule information
        numGlobalReadInsPerIter, numLocalWriteModPerIter, numEmptyGlobalReadIncCode = getScheduleParams(kernel)
        numLocalWritesPerSched = numLocalWriteModPerIter
        itemsGRToSchedLater, lastLoadIter = noSchedGlobalRead(writer, kernel, globalReadIncACode, globalReadIncBCode)
        # Schedule local write
        noSchedLocalWrite(writer, kernel, tensorParametersA, tensorParametersB, localWriteEndIter)

################################################################################
################################################################################
###
###   Schedule Parameters
###
################################################################################
################################################################################

def checkLocalReadFIFO(localReadFIFO, miLatency, numWaves, numMFMA, blockWidth):
    # Add space to avoid LR FIFO stall
    # lrStallLatencyBuffer:
    # 40 quad-cycle - 4 x miLatency for b128
    # 20 quad-cycle - 4 x miLatency for b64 (equal to one miLatency)
    # 10 quad-cycle - 4 x miLatency for b32 (no stall)
    # so no stall happen for b64/b32/b16
    if blockWidth < 4:
        return False
    # The FIFO length is 16 so that each wave has 16/numWaves buffer.
    lrStallLatencyBuffer = roundUp(blockWidth) * 10 - ((16 / numWaves) * miLatency)
    if len(localReadFIFO) < (16 / numWaves):
        localReadFIFO.append(numMFMA)
    else:
        oldNumMFMA = localReadFIFO[0]
        if (numMFMA - oldNumMFMA) * miLatency >= lrStallLatencyBuffer:
            localReadFIFO.pop(0)
            localReadFIFO.append(numMFMA)
        else:
            # FIFO is full
            return True
    return False

def getScheduleParams(kernel):
    numGlobalReadInsPerIter = roundUp(kernel["GlobalReadPerMfma"] * PRECISION) if kernel["GlobalReadPerMfma"] > 0 else PRECISION
    numLocalWriteModPerIter = roundUp(kernel["LocalWritePerMfma"] * PRECISION) if kernel["LocalWritePerMfma"] > 0 else PRECISION
    numEmptyGlobalReadIncCode = numGlobalReadInsPerIter - 1
    return numGlobalReadInsPerIter, numLocalWriteModPerIter, numEmptyGlobalReadIncCode

def getLocalWriteMFMAEnd(writer, kernel, tensorParametersA, tensorParametersB):
    numMfmaPerIter = writer.states.numMfmaPerIter
    #########
    # Get localWriteEnd
    #########
    # assign parameter
    # 1. we calculate number of mfma to prefetch localReads for next loop
    # 2. we put a barrier before the last mfma
    # 3. we put last localWrite before 2~3 mfma, then the barrier
    # localReads followed following sequence to be scheduled
    # ds_read[A][0], ds_read[B][0], ds_read[A][1:], ds_read[B][1:]
    # NOTE: we need this sequence for new feature "breaking waitcnt"
    # TODO: breaking waitcnt
    writer.states.numMfmaForLR = 1
    latencyLeft = writer.states.miLatencyLeft
    miLatencyLeft = writer.states.miLatencyLeft
    tPM = tensorParametersA["tpsMetadata"] if tensorParametersA["is_sparse"] else tensorParametersB["tpsMetadata"]

    # we can skip some LR waitcnt
    # Since the first mfma only use B[:1], so we only wait for B[0]
    isLocalReadsOpt = False
    tmpLatencyLeft  = 0
    tmpNumMfmaForLR = 0
    localReadsNotWaited = writer.states.numReadsPerIterB // kernel["InnerUnroll"] - writer.states.numReadsPerUnrollB
    if kernel["UnrollMajorLDSB"] and localReadsNotWaited > 0:
        isLocalReadsOpt = True

    # check instruction FIFO
    localReadFIFO = []
    numWaves      = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1] * kernel["LocalSplitU"]

    def calculateLatencyLeft(numReads, localReadBlockWidth, localReadLatency):
        nonlocal localReadFIFO, numWaves, latencyLeft, miLatencyLeft
        for _ in range(numReads):
            while(checkLocalReadFIFO(localReadFIFO, writer.states.miLatency, numWaves, writer.states.numMfmaForLR, localReadBlockWidth)):
                writer.states.numMfmaForLR += 1
            latencyLeft -= localReadLatency*2
            if latencyLeft < 0:
                writer.states.numMfmaForLR += 1
                latencyLeft = max(miLatencyLeft - localReadLatency*2,0)

    for iui in range(kernel["InnerUnroll"]):
        # ds_read[A][0]
        calculateLatencyLeft(writer.states.numReadsPerUnrollA, tensorParametersA["localReadInstruction"].blockWidth, tensorParametersA["localReadInstruction"].issueLatency)
        # ds_read[MXSA][0]
        if kernel["ProblemType"]["MXBlockA"]:
            calculateLatencyLeft(writer.states.numReadsPerUnrollMXSA, tensorParametersA["MX"]["localReadInstruction"].blockWidth, tensorParametersA["MX"]["localReadInstruction"].issueLatency)
        # ds_read[MXSB][0]
        if kernel["ProblemType"]["MXBlockB"]:
            calculateLatencyLeft(writer.states.numReadsPerUnrollMXSB, tensorParametersB["MX"]["localReadInstruction"].blockWidth, tensorParametersB["MX"]["localReadInstruction"].issueLatency)
        # ds_read[M][0]
        if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            calculateLatencyLeft(writer.states.numReadsPerUnrollMetadata, tPM["localReadInstruction"].blockWidth, tPM["localReadInstruction"].issueLatency)
        # ds_read[B][0]
        calculateLatencyLeft(writer.states.numReadsPerUnrollB, tensorParametersB["localReadInstruction"].blockWidth, tensorParametersB["localReadInstruction"].issueLatency)

        # ds_read[A][1:]
        calculateLatencyLeft((writer.states.numReadsPerIterA//kernel["InnerUnroll"] - writer.states.numReadsPerUnrollA), tensorParametersA["localReadInstruction"].blockWidth, tensorParametersA["localReadInstruction"].issueLatency)
        # ds_read[MXSA][1:]
        if kernel["ProblemType"]["MXBlockA"]:
            calculateLatencyLeft(writer.states.numReadsPerIterMXSA//kernel["InnerUnroll"] - writer.states.numReadsPerUnrollMXSA, tensorParametersA["MX"]["localReadInstruction"].blockWidth, tensorParametersA["MX"]["localReadInstruction"].issueLatency)
        # ds_read[M][1:]
        if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            calculateLatencyLeft((writer.states.numReadsPerIterMetadata//kernel["InnerUnroll"] - writer.states.numReadsPerUnrollMetadata), tPM["localReadInstruction"].blockWidth, tPM["localReadInstruction"].issueLatency)
        # ds_read[MXSB][1:]
        if kernel["ProblemType"]["MXBlockB"]:
            calculateLatencyLeft((writer.states.numReadsPerIterMXSB//kernel["InnerUnroll"] - writer.states.numReadsPerUnrollMXSB), tensorParametersB["MX"]["localReadInstruction"].blockWidth, tensorParametersB["MX"]["localReadInstruction"].issueLatency)
        # get the latency before B[:1]
        if isLocalReadsOpt:
            tmpLatencyLeft = latencyLeft
            tmpNumMfmaForLR = writer.states.numMfmaForLR
        # ds_read[B][1:]
        calculateLatencyLeft((writer.states.numReadsPerIterB//kernel["InnerUnroll"] - writer.states.numReadsPerUnrollB), tensorParametersB["localReadInstruction"].blockWidth, tensorParametersB["localReadInstruction"].issueLatency)

    # to calculate number of mfma we need to wait before data arrive from lds to vgpr.
    # latency: 40 quad-cycle for 4 word, 20 quad-cycle for 2 word, 10 quad-cycle for 1 word / half word
    writer.states.numMfmaForNextLoopLR = writer.states.numMfmaForLR
    latencyForLR = roundUp(tensorParametersB["localReadInstruction"].blockWidth)*10
    if isLocalReadsOpt:
        latencyForLR = min(latencyForLR, (writer.states.numMfmaForLR - tmpNumMfmaForLR) * writer.states.miLatency)
        latencyForLR -= max(tmpLatencyLeft,0) # remaining latency in mfma
    else:
        latencyForLR -= max(latencyLeft,0) # remaining latency in mfma
    latencyForLR -= writer.states.miLatency # last LR will have 1 mfma latency
    # add extra latency
    latencyForLR += kernel["ExtraLatencyForLR"]
    while latencyForLR > 0:
        latencyForLR -= writer.states.miLatency
        writer.states.numMfmaForNextLoopLR += 1
    # final index definition
    writer.states.numMfmaForNextLoopLR = min(writer.states.numMfmaForNextLoopLR,numMfmaPerIter-1)
    writer.states.syncPlrMfmaIndex = numMfmaPerIter*(kernel["LoopIters"]-writer.states.numItersPLR+1) - writer.states.numMfmaForNextLoopLR - 1 if writer.states.numItersPLR else 0
    if writer.states.doFullPackCodePrefetch and kernel["ForceUnrollSubIter"]:
        # move syncPlrMfmaIndex one iteration ahead for doFullPackCodePrefetch and subIter
        writer.states.syncPlrMfmaIndex = max(0, writer.states.syncPlrMfmaIndex - numMfmaPerIter)
    elif writer.states.doPackPreSchedulingNextLoop:
        # doPackPreSchedulingNextLoop only case (not doFullPackCodePrefetch), move syncPlrMfmaIndex to the top of
        writer.states.syncPlrMfmaIndex = numMfmaPerIter*(kernel["LoopIters"]-writer.states.numItersPLR)

    numMfmaBetweenLWandBarrier = 2 if kernel["MatrixInstM"] == 32 else 3
    if kernel["NoLdsWriteCode"]:
        # no interval needed in no ds_write case
        numMfmaBetweenLWandBarrier = 0
    if writer.states.scheduleGROverBarrier or writer.states.numItersPLR == 0:
        writer.states.lwEndMfmaIndex = writer.states.numMfmaPerIter*kernel["LoopIters"] - 1
    else:
        writer.states.lwEndMfmaIndex = max(writer.states.syncPlrMfmaIndex - numMfmaBetweenLWandBarrier,0)
    return numMfmaBetweenLWandBarrier, latencyLeft

def getLocalWriteMFMAStart(writer, kernel, tensorParametersA, tensorParametersB, latencyLeft):
    numMfmaPerIter = writer.states.numMfmaPerIter

    tPM = tensorParametersA["tpsMetadata"] if tensorParametersA["is_sparse"] else tensorParametersB["tpsMetadata"]
    #########
    # Get localWriteStart
    #########
    if not writer.states.oneBufferScheduling:
        # TODO: replace here for real number of globalReadIncInst
        # numGRIncInst = 18 # Always on. Original logic: 12 if not kernel["StaggerU"] else 18
        numGRIncInst = 12 if not writer.states.staggerUCode else 18
        if kernel["ProblemType"]["MXBlockA"] and kernel["ProblemType"]["MXBlockB"]:
            # MXSA+MXSB case, double the number
            numGRIncInst *= 2
        numInstPerMfma = max(roundUp(writer.states.miLatencyLeft/2),1)
        numMfmaToSched = roundUp(numGRIncInst/numInstPerMfma)
        lwStartMfmaIndex = 1 + numMfmaToSched
    else:
        # check instruction FIFO
        localReadFIFO = []
        numWaves      = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1] * kernel["LocalSplitU"]
        # for 1LDSB, we have to issue localwrites after localreads
        if kernel["ClusterLocalRead"]:
            # we have enough vgprBuffer to schedule localReads in the front of loop
            numMfmaForCurrentLoopLR = 1
            latencyLeft = writer.states.miLatencyLeft
            subIter = kernel["ForceUnrollSubIter"]
            # subIter case, LoopIters is increased to 4, but we should use 1 for latency calculation
            lookRange = 1 if subIter else kernel["LoopIters"] - writer.states.numItersPLR
            for u in range(lookRange):
                doReadA = ((u < kernel["LoopIters"] // writer.states.numIterPerCoalescedReadA - writer.states.numItersPLR) and not kernel["DirectToVgprA"]) or subIter
                doReadMXSA = False
                if kernel["ProblemType"]["MXBlockA"] > 0:
                    doReadMXSA = ((u < kernel["LoopIters"] // writer.states.numIterPerCoalescedReadMXSA - writer.states.numItersPLR) and not kernel["DirectToVgprMXSA"]) or subIter
                doReadB = ((u < kernel["LoopIters"] // writer.states.numIterPerCoalescedReadB - writer.states.numItersPLR) and not kernel["DirectToVgprB"]) or subIter
                doReadMXSB = False
                if kernel["ProblemType"]["MXBlockB"] > 0:
                    doReadMXSB = ((u < kernel["LoopIters"] // writer.states.numIterPerCoalescedReadMXSB - writer.states.numItersPLR) and not kernel["DirectToVgprMXSB"]) or subIter
                doReadM = ((u < kernel["LoopIters"] // writer.states.numIterPerCoalescedReadMetadata - writer.states.numItersPLR))
                doReadMXSA = doReadMXSA and kernel["ProblemType"]["MXBlockA"] and (not kernel["DirectToVgprMXSA"])
                doReadMXSB = doReadMXSB and kernel["ProblemType"]["MXBlockB"] and (not kernel["DirectToVgprMXSB"])
                doReadM = doReadM and (kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"])

                def calculateLatencyLeft(numReads, localReadBlockWidth, localReadLatency):
                    nonlocal localReadFIFO, numWaves, latencyLeft, numMfmaForCurrentLoopLR
                    for _ in range(numReads):
                        while(checkLocalReadFIFO(localReadFIFO, writer.states.miLatency, numWaves, numMfmaForCurrentLoopLR, localReadBlockWidth)):
                            numMfmaForCurrentLoopLR += 1
                        latencyLeft -= localReadLatency*2
                        if latencyLeft < 0:
                            numMfmaForCurrentLoopLR += 1
                            latencyLeft = max(writer.states.miLatencyLeft - localReadLatency*2,0)

                for iui in range(kernel["InnerUnroll"]):
                    # ds_read[A][0]
                    calculateLatencyLeft(writer.states.numReadsPerUnrollA * doReadA, tensorParametersA["localReadInstruction"].blockWidth, tensorParametersA["localReadInstruction"].issueLatency)
                    # ds_read[MXSA][0]
                    if doReadMXSA:
                        calculateLatencyLeft(writer.states.numReadsPerUnrollMXSA * doReadMXSA, tensorParametersA["MX"]["localReadInstruction"].blockWidth, tensorParametersA["MX"]["localReadInstruction"].issueLatency)
                    # ds_read[M][0]
                    if doReadM:
                        calculateLatencyLeft(writer.states.numReadsPerUnrollMetadata * doReadM, tPM["localReadInstruction"].blockWidth, tPM["localReadInstruction"].issueLatency)
                    # ds_read[MXSB][0]
                    if doReadMXSB:
                        calculateLatencyLeft(writer.states.numReadsPerUnrollMXSB * doReadMXSB, tensorParametersB["MX"]["localReadInstruction"].blockWidth, tensorParametersB["MX"]["localReadInstruction"].issueLatency)
                    # ds_read[B][0]
                    calculateLatencyLeft(writer.states.numReadsPerUnrollB * doReadB, tensorParametersB["localReadInstruction"].blockWidth, tensorParametersB["localReadInstruction"].issueLatency)

                    # ds_read[A][1:]
                    calculateLatencyLeft((writer.states.numReadsPerIterA//kernel["InnerUnroll"] - writer.states.numReadsPerUnrollA) * doReadA, tensorParametersA["localReadInstruction"].blockWidth, tensorParametersA["localReadInstruction"].issueLatency)
                    # ds_read[MXSA][1:]
                    if doReadMXSA:
                        calculateLatencyLeft((writer.states.numReadsPerIterMXSA//kernel["InnerUnroll"] - writer.states.numReadsPerUnrollMXSA) * doReadMXSA, tensorParametersA["MX"]["localReadInstruction"].blockWidth, tensorParametersA["MX"]["localReadInstruction"].issueLatency)
                    # ds_read[M][1:]
                    if doReadM:
                        calculateLatencyLeft((writer.states.numReadsPerIterMetadata//kernel["InnerUnroll"] - writer.states.numReadsPerUnrollMetadata) * doReadM, tPM["localReadInstruction"].blockWidth, tPM["localReadInstruction"].issueLatency)
                    # ds_read[MXSB][1:]
                    if doReadMXSB:
                        calculateLatencyLeft((writer.states.numReadsPerIterMXSB//kernel["InnerUnroll"] - writer.states.numReadsPerUnrollMXSB) * doReadMXSB, tensorParametersB["MX"]["localReadInstruction"].blockWidth, tensorParametersB["MX"]["localReadInstruction"].issueLatency)
                    # ds_read[B][1:]
                    calculateLatencyLeft((writer.states.numReadsPerIterB//kernel["InnerUnroll"] - writer.states.numReadsPerUnrollB) * doReadB, tensorParametersB["localReadInstruction"].blockWidth, tensorParametersB["localReadInstruction"].issueLatency)
            lwStartMfmaIndex = numMfmaForCurrentLoopLR
        else:
            lwStartMfmaIndex = numMfmaPerIter * (kernel["LoopIters"] - 1 - writer.states.numItersPLR) + writer.states.numMfmaForLR
        # to calculate number of mfma we need to wait before data arrive from lds to vgpr.
        # latency: 40 quad-cycle for 4 word, 20 quad-cycle for 2 word, 10 quad-cycle for 1 word / half word
        if writer.states.numIterPerCoalescedReadB > writer.states.numIterPerCoalescedReadA:
            latencyForLR = roundUp(tensorParametersA["localReadInstruction"].blockWidth) * 10
        else:
            latencyForLR = roundUp(tensorParametersB["localReadInstruction"].blockWidth) * 10
        latencyForLR -= max(latencyLeft,0) # remaining latency in mfma
        while latencyForLR > 0:
            latencyForLR -= writer.states.miLatency
            lwStartMfmaIndex += 1

    if lwStartMfmaIndex > writer.states.lwEndMfmaIndex:
        lwStartMfmaIndex = writer.states.lwEndMfmaIndex
    return lwStartMfmaIndex

def getNumLocalWritePerMfma(writer, kernel, lwStartMfmaIndex):
    #########
    # Get LocalWritePerMfma
    #########
    numMfmaCanSched = writer.states.lwEndMfmaIndex - lwStartMfmaIndex + 1
    numLoadsA = kernel["DepthU"]*kernel["MacroTileA"]//kernel["GlobalReadVectorWidthA"]//kernel["NumThreads"]
    if kernel["ProblemType"]["MXBlockA"]:
        numLoadsA += kernel["_DepthUMXSA"]*kernel["MacroTileA"]//kernel["GlobalReadVectorWidthMXSA"]//kernel["NumThreads"]
    numLoadsB = kernel["DepthU"]*kernel["MacroTileB"]//kernel["GlobalReadVectorWidthB"]//kernel["NumThreads"]
    if kernel["ProblemType"]["MXBlockB"]:
        numLoadsB += kernel["_DepthUMXSB"]*kernel["MacroTileB"]//kernel["GlobalReadVectorWidthMXSB"]//kernel["NumThreads"]
    if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
        macroTile = kernel["MacroTileB"] if kernel["ProblemType"]["Sparse"] == 2 else kernel["MacroTileA"]
        numLoadsM = kernel["DepthU"]*macroTile//kernel["GlobalReadVectorWidthMetadata"]//kernel["NumThreads"]
    else:
        numLoadsM = 0
    writesToSched = (numLoadsA + numLoadsB + numLoadsM- 1) * PRECISION
    oldValue = 0
    newValue = PRECISION
    loop = 0
    #   1. number of padded writesToSched is (numWrites - 1) * 100 + 1
    #     LW ---99--- LW ---99--- LW
    #   2. we need to pad it to multiple of LWPM
    #     LW ---99--- LW ---99--- LW --?--
    #     | ------- multiple of LWPM ---- |
    #   3. if LWPM is not multiple of 100, we need extra empty instructions to schedule GR for PGR2
    #     LW ---99--- LW ---99--- LW --?-- --?--
    #     | ------- multiple of LWPM ---- |-LWPM-|
    #   4. then we put GR into padded writesToSched
    #       put GR after LW + LWPM of empty inst, so that we can offset GR 1 mfma with LW if possible
    #     Ex. LWPM = 0.25
    #         LW --24- GR ------74------ LW --24- GR ------74------ LW --24- GR --24-
    #     mfma--24-mfma--24-mfma--24-mfma--24-mfma--24-mfma--24-mfma--24-mfma--24-mfma
    # we need LWPM to get precise LWPM
    # so we iterate formula 10 times to get LWPM
    while oldValue != newValue and newValue > 0 and loop < 10:
        loop += 1
        oldValue = newValue
        newValue = roundUp((writesToSched+1 + (oldValue - (writesToSched+1) % oldValue) + oldValue%PRECISION) / numMfmaCanSched)
    return newValue

def calculateGRPMandLWPM(writer, kernel, numLocalWriteModPerMfma) -> Tuple[int, int]:
    #####
    # Assign GRPM and LWPM
    #####
    # HOW THIS WORK
    # padding each globalReadInstruction to 100 with empty instruction,
    # each mfma will schedule intructions GRPM*100 times from padded globalReadInstruction.
    #   Ex. GRPM = 0.5
    #        GR ---------99--------- GR --------99---------- GR
    #   mfma --49-- mfma --49-- mfma --49-- mfma --49-- mfma --49--
    numGlobalReadInsPerMfma = roundUp(kernel["GlobalReadPerMfma"]*PRECISION)
    # HOW THIS WORK
    # padding each globalReadInstruction to 100 with empty instruction,
    # each mfma will schedule intructions GRPM*100 times from padded globalReadInstruction.
    #   Ex. LWPM = 0.5
    #        LW ---------99--------- LW --------99---------- LW
    #   mfma --49-- mfma --49-- mfma --49-- mfma --49-- mfma --49--
    if kernel["PrefetchGlobalRead"] == 1:
        # In PGR1:
        #   Larger LWPM can provide more latency to hide global read
        #   However, larger LWPM may cause mfma bubbles
        #   we set LWPM to 1 unless it requires larger LWPM to enable 1LDSB
        if kernel["1LDSBuffer"]:
            numLocalWriteModPerMfma = max(numLocalWriteModPerMfma, PRECISION)
        else:
            numLocalWriteModPerMfma = PRECISION

    return numGlobalReadInsPerMfma, numLocalWriteModPerMfma

def getScheduleParamMfma(writer):
    numMfmaPerIter = writer.states.numMfmaPerIter
    numGlobalReadInsPerIter = numMfmaPerIter * writer.states.numGlobalReadInsPerMfma
    numLocalWriteModPerIter = numMfmaPerIter * writer.states.numLocalWriteModPerMfma
    # if numGlobalReadInsPerMfma>1, we still want to schedule only 1 GlobalReadIncCode per mfma
    # inserting empty CodeModule so that generator will schedule 1 GlobalReadIncCode 1 empty CodeModule if numGlobalReadInsPerMfma=2
    numEmptyGlobalReadIncCode = writer.states.numGlobalReadInsPerMfma - 1
    return numGlobalReadInsPerIter, numLocalWriteModPerIter, numEmptyGlobalReadIncCode

def fixLocalWriteEndMfmaIndex(writer, kernel, tPA, tPB, globalReadIncACode, globalReadIncBCode, numMfmaBetweenLWandBarrier, lastLoop):
    numMfmaPerIter = writer.states.numMfmaPerIter
    # If numLocalWriteModPerMfma is not multiple of 100,
    # last globalread will be scheduled at lwEndMfmaIndex,
    # and last localwrite will be scheduled at lwEndMfmaIndex - 1
    # so we offset lwEndMfmaIndex by 1 mfma
    if kernel["PrefetchGlobalRead"] >= 2 and writer.states.numLocalWriteModPerMfma % PRECISION != 0 and not kernel["NoLdsWriteCode"]:
        numMfmaBetweenLWandBarrier -= 1

    if writer.states.scheduleGROverBarrier or writer.states.numItersPLR == 0:
        writer.states.lwEndMfmaIndex = numMfmaPerIter*kernel["LoopIters"] - 1
    else:
        writer.states.lwEndMfmaIndex = max(writer.states.syncPlrMfmaIndex - numMfmaBetweenLWandBarrier,0)
    localWriteEndIter = writer.states.lwEndMfmaIndex//numMfmaPerIter
    localWriteEndIter = min(kernel["LoopIters"] - 1, localWriteEndIter)
    assert localWriteEndIter < kernel["LoopIters"]
    assert writer.states.lwEndMfmaIndex < numMfmaPerIter*kernel["LoopIters"]
    return localWriteEndIter

################################################################################
################################################################################
###
###   Schedule Global Read
###
################################################################################
################################################################################

def _splitTdmLoad(grCode):
    """Split a globalRead module into (non-TDM items module, TDM-load-only module)."""
    tdmLoadMod = Module("deferredTdmLoad")
    nonTdmMod = Module()
    grAItems = grCode.items() if hasattr(grCode, 'items') else []
    for item in grAItems:
        hasTdm = isinstance(item, TensorLoadToLds) or \
                 (hasattr(item, 'flatitems') and any(isinstance(f, TensorLoadToLds) for f in item.flatitems()))
        if hasTdm:
            tdmLoadMod.add(item)
        else:
            nonTdmMod.add(item)
    return nonTdmMod, tdmLoadMod


def noSchedGlobalRead(writer, kernel, globalReadIncACode, globalReadIncBCode):
    # For TDM+SIA0: defer tensor_load_to_lds to after the LDS swap to avoid writing
    # into the LDS buffer still being read by ds_loads
    tdmDeferLoad = kernel["enableTDMA"] and kernel["enableTDMB"]
    localWriteEndIter = kernel["LoopIters"] - writer.states.numItersPLR - 1
    tdmLoadIter = min(localWriteEndIter + 1, kernel["LoopIters"] - 1)

    if kernel["PrefetchGlobalRead"] == 2:
        imod = writer.codes.perIterGlobalRead[0].add(Module())
        imod.addComment1("Global Read IncA")
        imod.add(globalReadIncACode)
        imod.addComment1("Global Read IncB")
        imod.add(globalReadIncBCode)
        if tdmDeferLoad:
            imod.addComment1("Global Read A")
            imod.add(writer.codes.dtlsM0UpdateA)
            nonTdmMod, tdmLoadModA = _splitTdmLoad(writer.codes.globalReadA)
            imod.add(nonTdmMod)
            imod.addComment1("Global Read MXSA")
            imod.add(writer.codes.dtlsM0UpdateMXSA)
            imod.add(writer.codes.globalReadMXSA)
            imod.addComment1("Global Read MXSB")
            imod.add(writer.codes.dtlsM0UpdateMXSB)
            imod.add(writer.codes.globalReadMXSB)
            imod.addComment1("Global Read B")
            imod.add(writer.codes.dtlsM0UpdateB)
            nonTdmModB, tdmLoadModB = _splitTdmLoad(writer.codes.globalReadB)
            imod.add(nonTdmModB)
            # Defer both A and B tensor_load_to_lds to after the LDS swap (at tdmLoadIter).
            # This ensures both loads go into the post-swap buffer, matching the ds_load reads.
            deferMod = writer.codes.perIterGlobalRead[tdmLoadIter].add(Module())
            if tdmLoadModA.itemsSize() > 0:
                deferMod.addComment1("Global Read A (TDM deferred after LDS swap)")
                deferMod.add(tdmLoadModA)
            if tdmLoadModB.itemsSize() > 0:
                # TODO: For the 1-wave case, schedule B's TDM load independently for better tensor load balance.
                deferMod.addComment1("Global Read B (TDM deferred after LDS swap)")
                deferMod.add(tdmLoadModB)
            # TODO: Once metadata TDM issueLoad is embedded inside globalReadDo(), globalReadMetadata
            # will be empty and this block can be removed. Until then, apply _splitTdmLoad so the
            # metadata TensorLoadToLds is deferred to tdmLoadIter together with A and B.
            if kernel["ProblemType"]["Sparse"]:
                imod.addComment1("Global Read Metadata")
                imod.add(writer.codes.globalReadMetadata)
        else:
            imod.addComment1("Global Read A")
            imod.add(writer.codes.dtlsM0UpdateA)
            imod.add(writer.codes.globalReadA)
            imod.addComment1("Global Read MXSA")
            imod.add(writer.codes.dtlsM0UpdateMXSA)
            imod.add(writer.codes.globalReadMXSA)
            imod.addComment1("Global Read MXSB")
            imod.add(writer.codes.dtlsM0UpdateMXSB)
            imod.add(writer.codes.globalReadMXSB)
            imod.addComment1("Global Read B")
            imod.add(writer.codes.dtlsM0UpdateB)
            imod.add(writer.codes.globalReadB)
            if kernel["ProblemType"]["Sparse"]:
                imod.addComment1("Global Read Metadata")
                imod.add(writer.codes.globalReadMetadata)
    else:
        # put everything in the header (original behavior for PGR=0/1):
        writer.codes.unrollLoopHeader.add(writer.codes.dtlsM0UpdateA)
        writer.codes.unrollLoopHeader.add(writer.codes.globalReadA)
        writer.codes.unrollLoopHeader.add(writer.codes.dtlsM0UpdateMXSA)
        writer.codes.unrollLoopHeader.add(writer.codes.globalReadMXSA)
        writer.codes.unrollLoopHeader.add(writer.codes.dtlsM0UpdateMXSB)
        writer.codes.unrollLoopHeader.add(writer.codes.globalReadMXSB)
        writer.codes.unrollLoopHeader.add(writer.codes.dtlsM0UpdateB)
        writer.codes.unrollLoopHeader.add(writer.codes.globalReadB)
        writer.codes.unrollLoopHeader.add(writer.codes.globalReadMetadata) if kernel["ProblemType"]["Sparse"] else None
        writer.codes.unrollLoopHeader.add(globalReadIncACode)
        writer.codes.unrollLoopHeader.add(globalReadIncBCode)
    # Dummy
    itemsGRToSchedLater = []
    lastLoadIter = 0
    return itemsGRToSchedLater, lastLoadIter

def prepareGRInstToSched(writer, kernel, isNGLL):
    writer.codes.unrollLoopHeader.add(writer.codes.globalReadA.header)
    writer.codes.unrollLoopHeader.add(writer.codes.globalReadMXSA.header)
    writer.codes.unrollLoopHeader.add(writer.codes.globalReadMXSB.header)
    writer.codes.unrollLoopHeader.add(writer.codes.globalReadB.header)
    writer.codes.unrollLoopHeader.add(writer.codes.globalReadMetadata.header) if kernel["ProblemType"]["Sparse"] else None

    # Add all loads from middle as individual schedulable items
    # when using PGR2, put global read instruction right after corresponding localWrite instruction
    if isNGLL and (kernel["UnrollLoopSwapGlobalReadOrder"] == 1 and not (kernel["DirectToLdsA"] and kernel["DirectToLdsB"])):
        itemsGRToSched =  []
        itemsGRToSchedLater = []
    elif kernel["PrefetchGlobalRead"] >= 2:
        itemsGRToSched =  []
        itemsGRToSchedLater = list(writer.codes.globalReadA.middle.items()) + \
                         list(writer.codes.globalReadMXSA.middle.items()) + \
                         list(writer.codes.globalReadMXSB.middle.items()) + \
                         list(writer.codes.globalReadB.middle.items()) + \
                         list(writer.codes.globalReadMetadata.middle.items())
    else:
        itemsGRToSched =  list(writer.codes.globalReadA.middle.items()) + \
                        list(writer.codes.globalReadMXSA.middle.items()) + \
                        list(writer.codes.globalReadMXSB.middle.items()) + \
                        list(writer.codes.globalReadB.middle.items()) + \
                        list(writer.codes.globalReadMetadata.middle.items())
        itemsGRToSchedLater = []

    itemsGRToSchedTemp = []
    for i in range(len(itemsGRToSched)):
        itemsGRToSchedTemp.append(itemsGRToSched.pop(0))
        for j in range(PRECISION-1):
            itemsGRToSchedTemp.append(Module())
    itemsGRToSched = itemsGRToSchedTemp
    return itemsGRToSched, itemsGRToSchedLater


def appendInstToSchedSIA3(writer, kernel, numEmptyGlobalReadIncCode, globalReadIncACode, globalReadIncBCode):
    itemsGRIncToSched = []
    # for SIA3, we can break GlobalReadIncCode to avoid mfma bubbles
    if kernel["PrefetchGlobalRead"] >= 2:
    # skip to schedule global read for PGR2 first mfma
        for i in range(numEmptyGlobalReadIncCode+1):
            imod = Module()
            itemsGRIncToSched.append(imod)
    numInst = countInstruction(globalReadIncACode) + countInstruction(globalReadIncBCode)
    numInstPerMfma = writer.states.numInstPerMfma

    globalReadIncItems = globalReadIncACode.flatitems() + globalReadIncBCode.flatitems()
    numMfmaToSched = roundUp(numInst/numInstPerMfma)
    for j in range(numMfmaToSched):
        imod = Module()
        count = 0
        while globalReadIncItems and count < numInstPerMfma:
            tempInst = globalReadIncItems.pop(0)
            imod.add(tempInst)
            if countInstruction(tempInst):
                count += 1
        itemsGRIncToSched.append(imod)
        for i in range(numEmptyGlobalReadIncCode):
            imod = Module()
            itemsGRIncToSched.append(imod)
    return itemsGRIncToSched

def appendInstToSchedDefault(numEmptyGlobalReadIncCode, globalReadIncACode, globalReadIncBCode):
    itemsGRIncToSched = []
    itemsGRIncToSched.append(globalReadIncACode)
    for i in range(numEmptyGlobalReadIncCode):
        imod = Module()
        itemsGRIncToSched.append(imod)
    itemsGRIncToSched.append(globalReadIncBCode)
    for i in range(numEmptyGlobalReadIncCode):
        imod = Module()
        itemsGRIncToSched.append(imod)
    return itemsGRIncToSched

def getSchedNumForIter0SIA3(writer, kernel, itemsGRToSched, itemsGRIncToSched, numGlobalReadInsPerIter):
    numMfmaPerIter = writer.states.numMfmaPerIter
    # Loop in PGR1: GlobalRead -> GlobalReadInc -> LocalWrite
    # but GlobalReadInc shouldn't block LocalWrite so we count them out
    # Loop in PGR2: GlobalReadInc -> LocalWrite/GlobalRead pair
    # since LocalWrite/GlobalRead pair depends on GlobalReadInc, we count in only GlobalReadInc
    if kernel["PrefetchGlobalRead"] >= 2:
        loadsToSched = len(itemsGRIncToSched)
    else:
        loadsToSched = len(itemsGRToSched)

    # Here is to adjust scheduling silently in order to have validation pass.
    # Better way is to use larger globalReadPerMfma.
    ## schedule more instructions at first iteration if no enough mfma to schedule globalRead
    writer.states.grEndMfmaIndex = max(0, roundUp(loadsToSched/writer.states.numGlobalReadInsPerMfma) - 1)
    endIndex = writer.states.lwEndMfmaIndex
    if writer.states.scheduleGROverBarrier:
        # scheduleGROverBarrier case, lwEnd can be after barrier sync
        # grEnd must be before barrier sync in that case
        endIndex = min(writer.states.lwEndMfmaIndex, writer.states.syncPlrMfmaIndex)
    if writer.states.IncLdsBufSwitch:
      # IncLdsBufSwitch case, we use s operation for local read swap and need to finish GR Inc before that
      if endIndex > writer.states.numMfmaPerIter - 1:
          # adjust endIndex
          endIndex = writer.states.numMfmaPerIter - 1
    if writer.states.grEndMfmaIndex > endIndex:
        schedNumForIter0 = numGlobalReadInsPerIter + (writer.states.grEndMfmaIndex - endIndex) * writer.states.numGlobalReadInsPerMfma
        writer.states.grEndMfmaIndex = endIndex
    else:
        schedNumForIter0 = numGlobalReadInsPerIter
    if kernel["PrefetchGlobalRead"] == 1:
        globalReadIncEndMfmaIndex = writer.states.grEndMfmaIndex + roundUp(len(itemsGRIncToSched)/writer.states.numGlobalReadInsPerMfma)
        endIter = roundUp((globalReadIncEndMfmaIndex+1)/numMfmaPerIter)
    else:
        endIter = roundUp((writer.states.grEndMfmaIndex+1)/numMfmaPerIter)
    ## schedule more instructions at first iteration if no enough mfma to schedule globalRead + globalReadInc
    if endIter > kernel["LoopIters"]:
        endIter = kernel["LoopIters"]
        if kernel["PrefetchGlobalRead"] == 1:
            schedNumForIter0 += (globalReadIncEndMfmaIndex+1 - kernel["LoopIters"]*numMfmaPerIter) * writer.states.numGlobalReadInsPerMfma
    return schedNumForIter0, endIter

# schedNumForIter0 SIA 1 or 2
# distribute the instructions in itemsGRToSched evenly as possible to iterations: perIterGlobalReadCode[0,endIter)
# last one is perIterGlobalReadCode[endIter-1],
# Ideally:     endIter <= localWriteEndIter,
#              then put M0 updateCode (if any) and first 'schedNumForIter0' GR-inst in perIterGlobalReadCode[0]
#              put every numGlobalReadInsPerIter GR-insts in perIterGlobalReadCode[1]~[endIter-1]
# corner case: endIter > localWriteEndIter, set endIter = localWriteEndIter,in this case, schedNumForIter0 will > 1
#              and perIterGlobalReadCode[0] would need to schedule more instructions
def getSchedNumForIter0Default(itemsGRToSched, itemsGRIncToSched, numGlobalReadInsPerIter, localWriteEndIter):
    # reads and incs are scheduled in iters range(0..endIter)
    endIter = roundUp((len(itemsGRToSched) + len(itemsGRIncToSched)) / numGlobalReadInsPerIter)
    # FIXME:
    # above formula precisely count number of GR + GRInc
    # however it has regression issue with tuned yaml with default GRPM.
    # below formula follows old logic to add 2 to the instruction count, so it may has larger schedNumForIter0
    # we should use above formula with GRPM tuning for better performance
    # NOTE: both formula pass validation test
    endIter = roundUp((len(itemsGRToSched) + len(itemsGRIncToSched) + 2*PRECISION) / numGlobalReadInsPerIter)
    if endIter > localWriteEndIter:
        # Front-load some of the buffer loads if we don't have enough loop iters:
        # could use a different/smarter algorithm to space out the loads?
        schedNumForIter0 = (endIter-(localWriteEndIter) + 1) * numGlobalReadInsPerIter
        endIter = localWriteEndIter
    else:
        # schedule b2b for readCnt > 2 (True for bigger TT)
        schedNumForIter0 = numGlobalReadInsPerIter
    return schedNumForIter0, endIter

def schedGlobalRead(writer, itemsGRToSched, itemsGRIncToSched, numGlobalReadInsPerIter, schedNumForIter0, endIter):
    # insert dtlsM0UpdateACode dtlsM0UpdateBCode code
    if writer.codes.globalReadA.middle.items():
        writer.codes.globalReadA.middle.getItem(0).add(writer.codes.dtlsM0UpdateA, 0)
    if writer.codes.globalReadMXSA.middle.items():
        writer.codes.globalReadMXSA.middle.getItem(0).add(writer.codes.dtlsM0UpdateMXSA, 0)
    if writer.codes.globalReadMXSB.middle.items():
        writer.codes.globalReadMXSB.middle.getItem(0).add(writer.codes.dtlsM0UpdateMXSB, 0)
    if writer.codes.globalReadB.middle.items():
        writer.codes.globalReadB.middle.getItem(0).add(writer.codes.dtlsM0UpdateB, 0)
    # TODO: if metadata supports directToLds
    # if writer.codes.globalReadMetadata.middle.items():
    #     writer.codes.globalReadMetadata.middle.getItem(0).add(writer.codes.dtlsM0UpdateMetadata, 0)

    itemsGRToSched.extend(itemsGRIncToSched)
    # append 'n' global load at a time
    # append global load(S) first 'number of global load(s)' determined by schedNumForIter0
    for item in itemsGRToSched[:schedNumForIter0]:
        writer.codes.perIterGlobalRead[0].add(item)
    itemsGRToSched = itemsGRToSched[schedNumForIter0:] # trim the scheduled GRs, do the rest in the following loop

    lastLoadIter = 0
    for u in range(1, endIter):
        # append itemPerIter GR for each iteration,
        # and trim the scheduled ones at the end of loop
        itemPerIter = 1 * numGlobalReadInsPerIter
        try:
            for item in itemsGRToSched[:itemPerIter]:
                writer.codes.perIterGlobalRead[u].add(item)
                lastLoadIter = u
            itemsGRToSched = itemsGRToSched[itemPerIter:]
        except IndexError:
            break # itemsGRToSched is 0-length, no code left to schedule

    assert not itemsGRToSched # should have scheduled everything already, itemsGRToSched should be empty

    writer.codes.perIterGlobalRead[endIter-1].add(writer.codes.globalReadA.footer)
    writer.codes.perIterGlobalRead[endIter-1].add(writer.codes.globalReadMXSA.footer)
    writer.codes.perIterGlobalRead[endIter-1].add(writer.codes.globalReadMXSB.footer)
    writer.codes.perIterGlobalRead[endIter-1].add(writer.codes.globalReadB.footer)
    writer.codes.perIterGlobalRead[endIter-1].add(writer.codes.globalReadMetadata.footer)
    return lastLoadIter

################################################################################
################################################################################
###
###   Schedule Local Write
###
################################################################################
################################################################################

def noSchedLocalWrite(writer, kernel, tensorParametersA, tensorParametersB, localWriteEndIter):
    # if no scheduleLocalWrite - just add writes to localWritelocalWriteEndIter
    # If PGR=0, writes have to be done immediately following the loads - no opportunity to schedule
    #   so don't add to schedule, these will be added separately and before the first iter
    if kernel["PrefetchGlobalRead"] and not kernel["NoLdsWriteCode"]:
        # do we need a module here? That would prevent these from being scheduled
        imod = writer.codes.perIterLocalWrite[localWriteEndIter][1].add(Module())
        imod.add(
            writer._wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, \
            "1wait for global read"))
        imod.addComment1("local write A")
        imod.add(writer.codes.localWriteA)
        imod.addComment1("local write MXSA")
        imod.add(writer.codes.localWriteMXSA)
        imod.addComment1("local write MXSB")
        imod.add(writer.codes.localWriteMXSB)
        imod.addComment1("local write B")
        imod.add(writer.codes.localWriteB)

        if kernel["PrefetchGlobalRead"] == 2 and writer.codes.perIterLocalWriteCodeNGLL is not None: # TODO: check condition
            # do we need a module here? That would prevent these from being scheduled
            imod = writer.codes.perIterLocalWriteCodeNGLL[localWriteEndIter][1].add(Module())
            imod.add(
                writer._wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, \
                "1wait for global read"))
            imod.addComment1("local write A")
            imod.add(writer.codes.localWriteA)
            imod.addComment1("local write MXSA")
            imod.add(writer.codes.localWriteMXSA)
            imod.addComment1("local write MXSB")
            imod.add(writer.codes.localWriteMXSB)
            imod.addComment1("local write B")
            imod.add(writer.codes.localWriteB)

def prepareLWInstToSched(writer, kernel, numLocalWritesPerSched, isNGLL=False):
    #################
    # create a plan #
    #################
    itemsLWToSched = list(writer.codes.localWriteA.items()) + \
                     list(writer.codes.localWriteMXSA.items()) + \
                     list(writer.codes.localWriteMXSB.items()) + \
                     list(writer.codes.localWriteB.items())
    numDummy = 0
    insertDummyTop = False
    if kernel["PrefetchGlobalRead"] >= 2:
        # PrefetchGlobalRead + DirectToLds/DirectToVgpr case, need to add dummy list to insert global read
        lenA    = len(list(writer.codes.globalReadA.middle.items()))
        lenMXSA = len(list(writer.codes.globalReadMXSA.middle.items()))
        lenMXSB = len(list(writer.codes.globalReadMXSB.middle.items()))
        lenB    = len(list(writer.codes.globalReadB.middle.items()))

        lenAFooter    = len(list(writer.codes.globalReadA.footer.items()))
        lenMXSAFooter = len(list(writer.codes.globalReadMXSA.footer.items()))
        lenMXSBFooter = len(list(writer.codes.globalReadMXSB.footer.items()))
        lenBFooter    = len(list(writer.codes.globalReadB.footer.items()))

        # A/B swap check for DTV. NGLL case, no swap
        swapped = writer.isSwapGlobalReadOrderForDtvOrDtl(kernel) and (not isNGLL)
        insertDummyTop = True
        if swapped:
          # swap A and B (SwapGlobalReadOrder case, the actual content is swapped (B is in globalReadACode). Need adjustment)
          lenA, lenB = lenB, lenA
          lenMXSA, lenMXSB = lenMXSB, lenMXSA

        if kernel["DirectToLdsA"] or kernel["DirectToVgprA"]:
            if kernel["DirectToLdsA"]:
              # PGR2 + DTLcase, footer code is added in middle. Need to subtract 1 (for footer inst)
              lenA -= lenAFooter
            numDummy += lenA
            insertDummyTop = (not swapped)
        if kernel["ProblemType"]["MXBlockA"] and (kernel["DirectToLdsMXSA"] or kernel["DirectToVgprMXSA"]):
            if kernel["DirectToLdsMXSA"]:
              # PGR2 + DTLcase, footer code is added in middle. Need to subtract 1 (for footer inst)
              lenMXSA -= lenMXSAFooter
            numDummy += lenMXSA
            insertDummyTop = (not swapped)
        if kernel["ProblemType"]["MXBlockB"] and (kernel["DirectToLdsMXSB"] or kernel["DirectToVgprMXSB"]):
            if kernel["DirectToLdsMXSB"]:
              # PGR2 + DTLcase, footer code is added in middle. Need to subtract 1 (for footer inst)
              lenMXSB -= lenMXSBFooter
            numDummy += lenMXSB
            insertDummyTop = swapped
        if kernel["DirectToLdsB"] or kernel["DirectToVgprB"]:
            if kernel["DirectToLdsB"]:
              # PGR2 + DTLcase, footer code is added in middle. Need to subtract 1 (for footer inst)
              lenB -= lenBFooter
            numDummy += lenB
            insertDummyTop = swapped
    # extend localWrite by inserting empty Module
    # See getNumLocalWritePerMfma for how this work
    itemsLWToSchedTemp = []
    counter = 0
    itemsLWToSchedLength_1 = len(itemsLWToSched) - 1
    for i in range(itemsLWToSchedLength_1 + numDummy):
        if insertDummyTop:
            if i < numDummy:
                item = None
            else:
                item = itemsLWToSched.pop(0)
                itemsLWToSchedTemp.append([counter, item])
        else:
            if i < itemsLWToSchedLength_1:
                item = itemsLWToSched.pop(0)
                itemsLWToSchedTemp.append([counter, item])
            else:
                item = None
        counter += 1
        skip = kernel["PrefetchGlobalRead"] >= 2 and kernel["ProblemType"]["Sparse"] and kernel["DirectToVgprSparseMetadata"] \
           and item.name.startswith("MetadataWrite") and countVMovB32(item)
        if not skip:
           for _ in range(PRECISION-1):
               counter += 1
    if itemsLWToSched or (numDummy and itemsLWToSchedLength_1 == -1):
        if itemsLWToSched:
            itemsLWToSchedTemp.append([counter, itemsLWToSched.pop(0)])
        counter += 1
        for i in range(numLocalWritesPerSched + numLocalWritesPerSched % PRECISION - counter % numLocalWritesPerSched):
            counter += 1
    itemsLWToSched = itemsLWToSchedTemp
    if not itemsLWToSched:
        itemsLWToSched.append([max(0, counter - 1), None])
    elif itemsLWToSched[-1][0] != (counter - 1):
        itemsLWToSched.append([counter - 1, None])  # end of the list if not equal to counter
    # This counts the number of modules which contain a ds_write
    # Scheduler below keeps all writes in the same module in same iteration
    # so this is better match to what it is trying to do
    # numWritesToSched = sum(1 for item in itemsLWToSched if countLocalWrite(item)
    numWritesToSched = itemsLWToSched[-1][0]
    return itemsLWToSched, numWritesToSched

def assignLWSchedIndexSIA3(writer, kernel, numLocalWritesPerSched, localWriteEndIter, numWritesToSched):
    numMfmaPerIter = writer.states.numMfmaPerIter
    writer.states.lwStartMfmaIndex = writer.states.lwEndMfmaIndex - max(1,roundUp(numWritesToSched/numLocalWritesPerSched)) + 1
    if writer.states.lwStartMfmaIndex < writer.states.grEndMfmaIndex:
          # adjust lwStartMfmaIndex for PGR1
          writer.states.lwStartMfmaIndex = writer.states.grEndMfmaIndex
    if kernel["1LDSBuffer"] or kernel["DirectToLds"]:
        writer.states.sync1LdsMfmaIndex = max(writer.states.lwStartMfmaIndex - 1, 0)
    startIter = writer.states.lwStartMfmaIndex//numMfmaPerIter
    assert startIter < localWriteEndIter+1 # startIter should be at or before the endIter
    return startIter

def assignLWSchedIndexDefault(writer, kernel, numLocalWritesPerSched, localWriteEndIter, lastLoadIter, numWritesToSched):
    startIter = localWriteEndIter - roundUp(numWritesToSched/numLocalWritesPerSched) + 1
    # - can't move a write past the load it depends on
    #   as a simplification, don't move writes past any loads
    if startIter < lastLoadIter:
        startIter = lastLoadIter
    return startIter

def getReadsToWait(writer, kernel):
    lwAcnt = len(list(writer.codes.localWriteA.items())) + \
             countDSStoreB192(writer.codes.localWriteA)
    lwBcnt = len(list(writer.codes.localWriteB.items())) + \
             countDSStoreB192(writer.codes.localWriteB)
    lwMXAcnt = len(list(writer.codes.localWriteMXSA.items()))
    lwMXBcnt = len(list(writer.codes.localWriteMXSB.items()))
    readsToWait = lwAcnt + lwBcnt + lwMXAcnt + lwMXBcnt
    readsToWaitNGLL = readsToWait
    return readsToWait, readsToWaitNGLL

def schedLocalWrite(writer, kernel, numLocalWriteModPerIter, numLocalWritesPerSched, localWriteEndIter, \
  itemsGRToSchedLater, itemsLWToSched, startIter, readsToWait, readsToWaitNGLL, \
  firstIter, lastLc, isNGLL, startIterItem = None):
    # schedule here
    localwriteCnt        = 0
    globalReadInstOffset = 0
    additionalIndexList  = {}
    skip = 0
    if itemsLWToSched and itemsLWToSched[0][0] == 0 and itemsLWToSched[0][1] is None:
        itemsLWToSched.pop(0) # remove the dummy item

    itemsLWToSchedIndexLast = 0
    for u in range(startIter, localWriteEndIter+1):
        itemsLWToSchedLength = itemsLWToSched[-1][0] if itemsLWToSched else 0
        if u == localWriteEndIter:
            itemPerIter = itemsLWToSchedLength # schedule all remaining activity
        else:
            itemPerIter = itemsLWToSchedIndexLast + numLocalWriteModPerIter - 1
            # if localwrite is not multiple of numLocalWriteModPerIter, fill last iteration first.
            # make sure numLocalWriteModPerIter is enough to schedule localwrite
            # TODO: if numLocalWriteModPerIter is not enough to schedule localwrite, need smarter way to distribute localWrite
            if u == startIter and startIterItem:
                itemPerIter = startIterItem - 1
        # Convert timeline index to map index
        if itemsLWToSched:
            foundIndex = -1
            skipInsert = False
            for index, [itemIndex, _] in enumerate(itemsLWToSched):
                if itemPerIter == itemIndex:
                    skipInsert = True
                    foundIndex = index
                    break
                if itemPerIter < itemIndex:
                    foundIndex = index
                    break
            if not skipInsert: # Insert scheduling point if needed
                if foundIndex != -1:
                    itemsLWToSched.insert(foundIndex, [itemPerIter, None])
                    itemPerIter = foundIndex + 1
                else:
                    itemsLWToSched.append([itemPerIter, None])
                    itemPerIter = len(itemsLWToSched)
                    itemsLWToSchedLength = itemPerIter
            else:
                itemPerIter = foundIndex + 1

        perIterLocalWriteCodeCounter = 0
        perIterLocalWriteCodeNGLLCounter = 0

        itemsLWToSchedLengthLeft = itemsLWToSchedLength - itemsLWToSchedIndexLast
        itemsLWToSchedIndexForSplitDS = 0
        for itemsLWToSchedIndex, item in itemsLWToSched[:itemPerIter]:
            tmp = itemsLWToSchedIndexLast + numLocalWriteModPerIter
            for gapIndex in range(itemsLWToSchedIndexLast, itemsLWToSchedIndex + 1):
                imodList = []
                imodNGLLList = []
                # Use a module to ensure these pieces stay together in the sub-iter scheduler
                if gapIndex == itemsLWToSchedIndex:
                    if item:
                        writesPerItem = countLocalWrite(item)
                        if kernel["ProblemType"]["Sparse"] and not writesPerItem:
                            writesPerItem = item.name.startswith("MetadataWrite") and countVMovB32(item)
                        if writesPerItem:
                            writesPerItem = countLocalWrite(item)
                            if kernel["ProblemType"]["Sparse"] and not writesPerItem:
                                writesPerItem = item.name.startswith("MetadataWrite") and countVMovB32(item)
                            # Split into several dsStore32
                            syncEndExpandedNumIndex = itemsLWToSchedIndexForSplitDS

                            if writer.states.numMfmaPerIter and u == (writer.states.lwEndMfmaIndex // writer.states.numMfmaPerIter):
                                syncEndExpandedNumIndex = numLocalWriteModPerIter
                                syncEndExpandedNumIndex *= ((writer.states.syncPlrMfmaIndex % writer.states.numMfmaPerIter) / writer.states.numMfmaPerIter)
                                syncEndExpandedNumIndex = roundUp(syncEndExpandedNumIndex)

                            itemNew, numItemNew, globalReadInstOffset = splitDSInstructionIntoSmaller(writer, kernel, item, numLocalWritesPerSched, syncEndExpandedNumIndex, itemsLWToSchedIndexForSplitDS) if writer.do["AutoSplitDsWrite"] else (None, 0, 0)
                            if itemsLWToSchedIndexForSplitDS + globalReadInstOffset <= itemsLWToSchedLengthLeft:
                                additionalIndexList.clear()
                                for i in range(numItemNew):
                                    additionalIndexList[int(i * numLocalWritesPerSched + itemsLWToSchedIndexForSplitDS)] = itemNew[i]
                            else:
                                globalReadInstOffset = 0

                            imodList.append(TextBlock("/* sched write - iter %u writesPerItem=%u */\n"%(u,writesPerItem)))
                            imodNGLLList.append(TextBlock("/* sched write - iter %u writesPerItem=%u */\n"%(u,writesPerItem)))
                            # if writesPerItem>1 this indicates multiple LocalWrites in the same module
                            # this happens in some transpose cases.  Here the first write needs to wait
                            # for the associated global read to finish, then the remaining writes can flow
                            # TODO - can schedule these writes across iters, should figure this out above
                            # A ds_store_b128 + ds_store_b64 pair (counted as one B192-equivalent by
                            # countDSStoreB192) corresponds to 2 physical VMEM loads. Each extra physical
                            # load beyond the first must also decrement readsToWait so that the generated
                            # s_wait_loadcnt value correctly tracks the outstanding load count.
                            numPhysLoads = 1 + countDSStoreB192(item)
                            readsToWait = readsToWait - numPhysLoads
                            readsToWaitNGLL = readsToWaitNGLL - numPhysLoads
                            imodList.append(SWaitCnt(vlcnt=readsToWait, \
                                comment="wait for global read before writing to local"))
                            imodNGLLList.append(SWaitCnt(vlcnt=readsToWaitNGLL, \
                                comment="wait for global read before writing to local"))
                        # PK and StoreCUnroll is removed so you cannot find any HolderContainer in s_waitcnt
                        if kernel["PrefetchGlobalRead"]>=2:
                            hasHolder, wcList = hasHolderInWaitCnt(item)
                            if hasHolder:
                                readsToWaitAdjust = readsToWait
                                if kernel["NoLdsWriteCode"]:
                                    # DirectToLds for both A and B case, use  the number of global read for both A and B as vmcnt (only for PGR=1)
                                    readsToWaitAdjust = len(list(writer.codes.globalReadA.middle.items())) + \
                                                        len(list(writer.codes.globalReadMXSA.middle.items())) + \
                                                        len(list(writer.codes.globalReadMXSB.middle.items())) + \
                                                        len(list(writer.codes.globalReadB.middle.items()))
                                for wc in wcList:
                                    replaceHolder(wc, (readsToWaitAdjust))
                if itemsLWToSchedIndexForSplitDS in additionalIndexList:
                    imodList.append(additionalIndexList[itemsLWToSchedIndexForSplitDS])
                    additionalIndexList.pop(itemsLWToSchedIndexForSplitDS)
                elif gapIndex < itemsLWToSchedIndex or (not item):
                    pass
                else:
                    imodList.append(item)
                # schedule global instruction that need to be scheduled later
                numGlobalReadA = kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"]
                numGlobalReadB = kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"]
                dtvReadNum = numGlobalReadA if kernel["DirectToVgprA"] else numGlobalReadB
                totalNumGR = numGlobalReadA + numGlobalReadB
                nondtvReadNum = totalNumGR - dtvReadNum

                readCntA = 1
                readCntB = 1
                readCntA = 2 if kernel["DirectToVgprA"] and kernel["reorderGRInstForDTVA"] and \
                                kernel["NumLoadsCoalescedA"] % 2 == 0 else 1
                readCntB = 2 if kernel["DirectToVgprB"] and kernel["reorderGRInstForDTVB"] and \
                                kernel["NumLoadsCoalescedB"] % 2 == 0 else 1

                if kernel["DirectToVgprA"]:  # In loop, load A first
                    readCnt = readCntA if (len(itemsGRToSchedLater) > nondtvReadNum) or isNGLL else readCntB
                elif kernel["DirectToVgprB"]:  # In loop, load B first
                    readCnt = readCntB if (len(itemsGRToSchedLater) > nondtvReadNum) or isNGLL else readCntA
                else:  # not kernel["DirectToVgprA"] and not kernel["DirectToVgprB"]
                    readCnt = 1

                if localwriteCnt % PRECISION == ((numLocalWritesPerSched % PRECISION) + globalReadInstOffset):
                    if not skip:
                        globalReadInstOffset = 0
                        reads = 0
                        while itemsGRToSchedLater:
                            itemGR = itemsGRToSchedLater[0]
                            readsInc = countGlobalRead(itemGR)
                            reads = reads + readsInc
                            if reads > readCnt * readsInc:
                                break
                            if kernel["ExpertSchedulingMode"] > 0:
                                imodList.append(SWaitAlu(vm_vsrc=0, comment="wait for local read to vgpr complete"))
                            # PK and StoreCUnroll is removed so you cannot find any HolderContainer in s_waitcnt
                            hasHolder, wcList = hasHolderInWaitCnt(itemGR)
                            if hasHolder:
                                for wc in wcList:
                                    replaceHolder(wc, (readsToWait))
                                imodList.append(itemGR)
                            else:
                                imodList.append(itemGR)
                            readsToWait = readsToWait + readsInc # GR instruction increments vlcnt
                            itemsGRToSchedLater.pop(0)

                    if readCnt == 2:
                        skip = skip ^ 1
                    else:
                        skip = 0
                localwriteCnt += 1
                if gapIndex < itemsLWToSchedIndex or (not item):
                    pass
                else:
                    imodNGLLList.append(deepcopy(item))

                perIterLocalWriteCodeCounter += 1
                perIterLocalWriteCodeNGLLCounter += 1
                if imodList:
                    imod = Module("LocalWriteMod%u"%u)
                    imod.addItems(imodList)
                    writer.codes.perIterLocalWrite[u][0].append(perIterLocalWriteCodeCounter)
                    writer.codes.perIterLocalWrite[u][1].add(imod)
                if lastLc:
                    # local write code for NGLL should be updated at the last lc
                    # in init acc opt case, the last inner loop generated is not for the last lc.
                    # in that case, local write code for NGLL is not as expected.
                    if imodNGLLList:
                        imodNGLL = Module("LocalWriteModNGLL%u"%u)
                        imodNGLL.addItems(imodNGLLList)
                        writer.codes.perIterLocalWriteCodeNGLL[u][0].append(perIterLocalWriteCodeNGLLCounter)
                        writer.codes.perIterLocalWriteCodeNGLL[u][1].add(imodNGLL)
                itemsLWToSchedIndexForSplitDS += 1
            itemsLWToSchedIndexLast = itemsLWToSchedIndex + 1
        if writer.codes.perIterLocalWrite[u][0] and writer.codes.perIterLocalWrite[u][0][-1] != perIterLocalWriteCodeCounter:
            writer.codes.perIterLocalWrite[u][0].append(perIterLocalWriteCodeCounter)
        if lastLc and writer.codes.perIterLocalWriteCodeNGLL[u][0] and writer.codes.perIterLocalWriteCodeNGLL[u][0][-1] != perIterLocalWriteCodeNGLLCounter:
            writer.codes.perIterLocalWriteCodeNGLL[u][0].append(perIterLocalWriteCodeNGLLCounter)
        itemsLWToSched = itemsLWToSched[itemPerIter:]

    # should never run out of items to schedule
    assert not itemsLWToSched # should have scheduled everthing already

    #For the sparse case, GR and LW are not in paired.
    #Hence, we must add all the remaining GRs into imod at the end.
    if kernel["ProblemType"]["Sparse"]:
        while itemsGRToSchedLater:
            itemGR = itemsGRToSchedLater[0]
            imod.add(itemGR)
            itemsGRToSchedLater.pop(0)

def splitDSInstructionIntoSmaller(writer, kernel, item, numLocalWritesPerSched, lenOfItems, currentModIdx):
    if not item:
        return None, 0, 0
    if countDSStoreB128(item) != 1 or countInstruction(item) != 1:
        # only support one b128
        return None, 0, 0

    itemList = item.flatitems()
    instruction = next(filter(lambda inst: isinstance(inst, DSStoreB128), itemList))
    miLatency = writer.states.miLatency
    div       = 1
    dsOffset  = 0

    LocalWriteX = DSStoreB128
    if DSStoreB128.issueLatency() < (miLatency - 1):
        # no need to split
        return None, 0, 0
    elif DSStoreB64.issueLatency() < (miLatency - 1):
        LocalWriteX = DSStoreB64
        dsOffset = 8
        div = 2
    elif DSStoreB32.issueLatency() < (miLatency - 1):
        LocalWriteX = DSStoreB32
        dsOffset = 4
        div = 4
    else:
        # miLatency is not enough
        return None, 0, 0

    if numLocalWritesPerSched * div >= PRECISION:
        # no enough mfma to split
        return None, 0, 0

    if (currentModIdx + numLocalWritesPerSched * div) >= lenOfItems:
        # no enough modules to schedule
        return None, 0, 0

    # LW b32 4-way bank conflict latency ~ 108 cycles
    # round up with quad-cycle
    finalLWCycles  = roundUp(108 / 4)

    # How many mfmas between 2 LWs
    # the MFMA buffer must be larger then the latency
    numMfmaBetweenLW = PRECISION // numLocalWritesPerSched
    mfmaBuffer = numMfmaBetweenLW * miLatency
    if mfmaBuffer <= finalLWCycles:
        # no enough cycles between LWs
        return None, 0, 0

    extraSched = roundUp(finalLWCycles / miLatency)
    if (currentModIdx + numLocalWritesPerSched * (div - 1 + extraSched)) >= lenOfItems:
        # no enough cycles before barrier
        return None, 0, 0

    addr = instruction.getParams()[0]
    srcr = instruction.getParams()[1]
    offs = instruction.getParams()[2]
    ds   = instruction.ds
    writeInst = []
    for d in range(div):
        ds1 = DSModifiers(na=1, offset=ds.offset + dsOffset * d)
        r1  = deepcopy(srcr)
        r1.regNum //= div
        r1.regName.addOffset(4 // div * d)
        writeInst.append(LocalWriteX(dstAddr=addr, src=r1, ds=ds1, comment=instruction.comment + " splitted"))

    print2(f"Split ds_write_b128 to 4xds_write_b32 for {str(instruction)}")

    return writeInst, len(writeInst), numLocalWritesPerSched * (div - 1)


################################################################################
################################################################################
###
###   Helper function
###
################################################################################
################################################################################

def hasHolderInWaitCnt(module: Item):
    wcList = []
    hasHolder = False
    if isinstance(module, Module):
        for item in module.items():
            tmpHasHolder, tmpList = hasHolderInWaitCnt(item)
            hasHolder = hasHolder or tmpHasHolder
            wcList.extend(tmpList)
    elif isinstance(module, SWaitCnt):
        wcList.append(module)
        if isinstance(module.dscnt, HolderContainer) or \
           isinstance(module.kmcnt, HolderContainer) or \
           isinstance(module.vlcnt, HolderContainer) or \
           isinstance(module.vscnt, HolderContainer):
           hasHolder = True
    return hasHolder, wcList
