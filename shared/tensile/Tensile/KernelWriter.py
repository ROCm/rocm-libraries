################################################################################
#
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from . import Code
from . import Common
from .Common import globalParameters, CHeader, roundUp, Backup, tPrint
from .ReplacementKernels import ReplacementKernels
from .CustomKernels import isCustomKernelConfig
from .SolutionStructs import Solution

import abc
from collections.abc import Sequence
import os
import shutil
import subprocess
import copy
from math import ceil

################################################################################
# Kernel Writer
################################################################################
class KernelWriter(metaclass=abc.ABCMeta):
  #__metaclass__=abc.ABCMeta

  ##############################################################################
  # Init
  ##############################################################################
  def __init__( self, kernelMinNaming, kernelSerialNaming, removeTemporaries=True ):
    self.kernelMinNaming = kernelMinNaming
    self.kernelSerialNaming = kernelSerialNaming
    self.overflowedResources = 0
    self.removeTemporaries = removeTemporaries

  @property
  def asmCaps(self):
    """
    Assembler capabilities for the current ISA version.
    """
    return globalParameters["AsmCaps"][self.version]

  @property
  def archCaps(self):
    """
    Architectural capabilities for the current ISA version.
    """
    return globalParameters["ArchCaps"][self.version]

  @property
  def globalParams(self):
    """
    Global parameters for current configuration.
    """
    return globalParameters

  ##############################################################################
  # returns number of Local Read included in current loop iteration
  ##############################################################################
  def countNumMfmaForCurrentOrNextLoopLR(self, kernel, tensorParametersA, tensorParametersB, curr=True):
    numMfmaForLRA = 0
    numMfmaForLRB = 0
    latencyLeft = self.miLatencyLeft
    issueLatencyA = min(self.miLatencyLeft, tensorParametersA["localReadInstruction"].IssueLatency*2)
    issueLatencyB = min(self.miLatencyLeft, tensorParametersB["localReadInstruction"].IssueLatency*2)
    # loop iteration:
    #   current loop: kernel["LoopIters"] - self.numItersPLR
    #   next loop   : 1 (prefetch for the next loop iter
    loopIter = kernel["LoopIters"] - self.numItersPLR if curr else 1
    for u in range(loopIter):
      doReadA = (not curr) or (u < kernel["LoopIters"] // self.numIterPerCoalescedReadA - self.numItersPLR)
      doReadB = (not curr) or (u < kernel["LoopIters"] // self.numIterPerCoalescedReadB - self.numItersPLR)
      # count the number of LocalRead even for DirectToVgpr
      # (performance is better with this)
      # ds_read[A][0]
      for i in range(self.numReadPerVectorA * doReadA):
        latencyLeft -= issueLatencyA
        if latencyLeft < 0:
          numMfmaForLRA += 1
          latencyLeft = self.miLatencyLeft - issueLatencyA
      # ds_read[B][0]
      for i in range(self.numReadPerVectorB * doReadB):
        latencyLeft -= issueLatencyB
        if latencyLeft < 0:
          numMfmaForLRB += 1
          latencyLeft = self.miLatencyLeft - issueLatencyB
      # ds_read[A][1:]
      for i in range((self.numReadsPerIterA - self.numReadPerVectorA) * doReadA):
        latencyLeft -= issueLatencyA
        if latencyLeft < 0:
          numMfmaForLRA += 1
          latencyLeft = self.miLatencyLeft - issueLatencyA
      # ds_read[B][1:]
      for i in range((self.numReadsPerIterB - self.numReadPerVectorB) * doReadB):
        latencyLeft -= issueLatencyB
        if latencyLeft < 0:
          numMfmaForLRB += 1
          latencyLeft = self.miLatencyLeft - issueLatencyB

    # initial value(1) + A + B
    numMfmaForLR = 1 + numMfmaForLRA + numMfmaForLRB

    latencyForLRCount = 0
    # no need to count if DTVA and DTVB are both true (no local read code)
    if not (kernel["DirectToVgprA"] and kernel["DirectToVgprB"]):
      # to calculate number of mfma we need to wait before data arrive from lds to vgpr.
      # latency: 40 quad-cycle for 4 word, 20 quad-cycle for 2 word, 10 quad-cycle for 1 word / half word
      latencyForLRA = roundUp(tensorParametersA["localReadInstruction"].blockWidth) * 10
      latencyForLRB = roundUp(tensorParametersB["localReadInstruction"].blockWidth) * 10

      if kernel["DirectToVgprA"]:
        # DTVA case, use B
        latencyForLRA = latencyForLRB
      elif kernel["DirectToVgprB"]:
        # DTVB case, use A
        latencyForLRB = latencyForLRA
      if not curr:
        latencyForLR = latencyForLRB
      else:
        #latencyForLR = max(latencyForLRA, latencyForLRB)
        latencyForLR = latencyForLRA if self.numIterPerCoalescedReadB > self.numIterPerCoalescedReadA else latencyForLRB

      latencyForLR -= max(latencyLeft,0) # remaining latency in mfma
      if not curr:
        latencyForLR -= self.miLatency # last LR will have 1 mfma latency
      # add extra latency
      latencyForLR += kernel["ExtraLatencyForLR"]
      while latencyForLR > 0:
        latencyForLR -= self.miLatency
        latencyForLRCount += 1

      if kernel["1LDSBuffer"] and curr and latencyForLRCount == 0:
         # 1LDS buffer case, we need at least 1 MFMA between end of local read and start of local write (fur current local read only)
         # otherwise, it results in overflowedResources = 5 error
        latencyForLRCount = 1

    return numMfmaForLR, latencyForLRCount, latencyLeft

  ##############################################################################
  # set and adjust lwEndMfmaIndex
  ##############################################################################
  def setAndAdjustLwEndMfmaIndex(self, kernel, tensorParametersA, tensorParametersB, numMfmaBetweenLWandBarrier, lastLoop):
    numMfmaPerIter = self.numMfmaPerIter
    self.lwEndMfmaIndex = max(self.barrierMfmaIndex - numMfmaBetweenLWandBarrier,0) if self.numItersPLR else numMfmaPerIter*kernel["LoopIters"] - 1
    # adjust lwEndMfmaIndex for the following cases
    #  1) PGR=2 + DirectToVgpr(DTV)
    #  2) last loop and StoreCInUnrollPostLoop enabled case
    # In these cases, lwEndMfmaIndex needs to be < numMfmaPerIter * (kernel["LoopIters"] - 1)
    # to schedule global read for DTV after lwEndMfmaIndex or execute PostLoop after StoreC in NoLoadLoop
    # kernel["LoopIters"]  has to be > 1 to make this logic work.
    if kernel["LoopIters"] > 1 and \
       ((kernel["PrefetchGlobalRead"] == 2 and (kernel["DirectToVgprA"] or kernel["DirectToVgprB"])) or \
        (lastLoop and kernel["StoreCInUnrollPostLoop"])):
      self.lwEndMfmaIndex = min(self.lwEndMfmaIndex, numMfmaPerIter * (kernel["LoopIters"] - 1) - 1)
    # another adjustment
    if (kernel["1LDSBuffer"] or kernel["DirectToLdsA"] or kernel["DirectToLdsB"]) and kernel["PrefetchGlobalRead"] == 2:
      # (1LDSBuffer of DirectToLds) + PGR=2 case, lwEndMfmaIndex must be after the end of local read (excluding local reads for next iter)
      numMfmaForCurrentLoopLR, latencyForLRCount, _ = self.countNumMfmaForCurrentOrNextLoopLR(kernel, tensorParametersA, tensorParametersB)
      numMfmaForCurrentLoopLR += latencyForLRCount
      lrEnd = min(self.barrierMfmaIndex - 1, numMfmaForCurrentLoopLR)
      if self.lwEndMfmaIndex < lrEnd:
        self.lwEndMfmaIndex = lrEnd

  ##############################################################################
  # makeSchedule:  Schedule work into interations.

  # Tensile uses a two-level scheduler.  This the first-level, which
  # schedules global reads, global incs, and local writes into iteration.
  # Then makeSubIterSchedule schedules the instructions within the iteration.
  #
  # Inputs:
  #   localWriteEndIter: loop iteration where last writes should be inserted
  #      If scheduleLocalWrite=0, all writes will be be placed in this iteration.
  #      If scheduleLocalWrite=1, the scheduler will work backwards from this
  #      iteration.
  #
  # Outputs:
  #   self.unrollLoopHeaderCode:
  #      - Code module that should be added into the unroll loop header
  #        In unscheduled code this contains global loads and global address increment
  #   self.perIterGlobalReadCode[], self.perIterLocalWriteCode[]
  #      - List indexed by unroll iteration.
  #        Each entry in the list is a code module that should be added into that iteration.
  #        May be None, indicating no extra code for that iteration
  #   self.grEndMfmaIndex
  #   self.lwStartMfmaIndex
  #   self.lwEndMfmaIndex
  #   self.barrierMfmaIndex
  #   self.numMfmaForNextLoopLR
  # This routine is responsible for setting the schedule including determining
  # that all necessary dependency are met.  The driver code in kernelBody
  # blindly follows the plan set in unrollLoopHeaderCode and perIterCode
  ##############################################################################
  def makeSchedule(self, kernel, tensorParametersA, tensorParametersB, localWriteEndIter, uDu=0, skipGlobalReadInc=False, firstIter=False, lastLoop=False, lastLc=False):

    currentIsa = globalParameters["CurrentISA"]
    maxVmcnt = globalParameters["AsmCaps"][currentIsa]["MaxVmcnt"]

    self.unrollLoopHeaderCode = Code.Module()
    # schedule of work for each local_read iteration:
    self.perIterGlobalReadCode = [ Code.Module() for i in range (kernel["LoopIters"]) ]
    self.perIterLocalWriteCode = [ Code.Module() for i in range (kernel["LoopIters"]) ]
    if lastLc:
      self.perIterLocalWriteCodeNGLL = [ Code.Module() for i in range (kernel["LoopIters"]) ]
    self.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
    self.perIterGlobalReadCodeDTV = [ Code.Module() for i in range (kernel["LoopIters"]) ] # global read for DirectToVgpr
    assert([item.name for item in self.globalReadIncrements.itemList] == ['globalReadIncrementA', 'globalReadIncrementB'])

    globalReadIncACode  = self.globalReadIncrements.findNamedItem("globalReadIncrementA")
    globalReadIncBCode  = self.globalReadIncrements.findNamedItem("globalReadIncrementB")

    if uDu < kernel["DepthULdsDivisor"] - 1 and kernel.enabledSplitLDS and kernel["PrefetchGlobalRead"] \
       or skipGlobalReadInc:
      globalReadIncACode  = Code.Module()
      globalReadIncBCode  = Code.Module()

    grBackup = None
    if uDu != kernel["DepthULdsDivisor"] - 2 and kernel.enabledSplitLDS:
      # hack RAII object for auto restore
      # withhold issuing global read codes until in the 2nd last subloop, meaning we empty the code
      # modules in other subloops.
      grBackup = Backup(self, globalReadACode = self.globalReadACode, globalReadBCode = self.globalReadBCode)
      self.globalReadACode = Code.StructuredModule() # empty
      self.globalReadBCode = Code.StructuredModule() # empty

    numGlobalReadC = self.getNumberOfLoadCInForLoadC(kernel)

    lastLoadIter = 0
    # PRECISION for scheduling
    # use 2 times value to make the calculation more accurate
    # using larger PRECISION makes scheduling more accurate but it increses Tensile execution time
    #PRECISION = 100
    PRECISION = 100 * 2
    self.PRECISION = PRECISION
    if kernel["EnableMatrixInstruction"] and kernel["ScheduleIterAlg"] == 3:
      numMfmaPerIter = self.numMfmaPerIter
      #########
      # Get localWriteEnd
      #########
      # assign parameter
      # 1. we calculate number of mfma to prefetch localReads for next loop
      # 2. we put barrier 1 mfma ahead that
      # 3. we put last localWrite 1~2 mfma ahead barrier
      # localReads followed following sequence to be scheduled
      # ds_read[A][0], ds_read[B][0], ds_read[A][1:], ds_read[B][1:]
      # NOTE: we need this sequence for new feature "breaking waitcnt"
      # TODO: breaking waitcnt
      # number of local read for the next loop iteration
      self.numMfmaForLR, latencyForLRCount, latencyLeft = self.countNumMfmaForCurrentOrNextLoopLR(kernel, tensorParametersA, tensorParametersB, curr=False)
      self.numMfmaForNextLoopLR = self.numMfmaForLR + latencyForLRCount
      # final index definition
      self.numMfmaForNextLoopLR = min(self.numMfmaForNextLoopLR,numMfmaPerIter-1)
      self.barrierMfmaIndex = numMfmaPerIter*(kernel["LoopIters"]-self.numItersPLR+1) - self.numMfmaForNextLoopLR - 1 if self.numItersPLR else 0
      numMfmaBetweenLWandBarrier = 2 if kernel["MatrixInstM"] == 32 else 3
      if self.miLatency <= 4 and kernel["LoopIters"] >= 4:
        # low latency MFMA and enough number of loop iteration case, we double numMfmaBetweenLWandBarrier
        numMfmaBetweenLWandBarrier *= 2
      # set and adjust lwEndMfmaIndex
      self.setAndAdjustLwEndMfmaIndex(kernel, tensorParametersA, tensorParametersB, numMfmaBetweenLWandBarrier, lastLoop)

      #########
      # Internally assign an optimized LWPM value for PGR2
      #########
      # strategy is to distribute LW/GR as wide as possible to avoid hitting vmem FIFO
      # LWPM = (LW_End - LW_Start) / numLW
      # calculate numLocalWriteModPerMfma for all LocalWritePerMfma cases (use this for parameter check in LocalWritePerMfma!=1 case)
      #if kernel["LocalWritePerMfma"] == -1:
      if True:
        #########
        # Get localWriteStart
        #########
        if not (kernel["1LDSBuffer"] or (kernel["DirectToLdsA"] or kernel["DirectToLdsB"])):
          # TODO: replace here for real number of globalReadIncInst
          numGRIncInst = 12 if not kernel["StaggerU"] else 18
          numInstPerMfma = max(roundUp(self.miLatencyLeft/2),1)
          numMfmaToSched = roundUp(numGRIncInst/numInstPerMfma)
          lwStartMfmaIndex = 1 + numMfmaToSched
          latencyForLRCount = 0
        else:
          # for 1LDSB or DTL, we have to issue localwrites after localreads
          numMfmaForLRCurr, latencyForLRCount, latencyLeft = self.countNumMfmaForCurrentOrNextLoopLR(kernel, tensorParametersA, tensorParametersB)
          if self.numVgprBuffer == kernel["LoopIters"]:
            if ((not kernel["DirectToVgprA"]) and self.numReadPerVectorA > 1 or (not kernel["DirectToVgprB"]) and self.numReadPerVectorB > 1) and \
               (not kernel["VgprForLocalReadPacking"]):
            # no VgprForLocalReadPacking only
            # fp16 or bf16, we read 1 element to vgprBuffer the other element to tempVgpr.
            # since each iteration shares same tempVgpr, only read-to-vgprBuffer can
            # be scheduled in the front of loop.
              # localwrite have to start after last read-to-tempVgpr.
              numHalfReads = (self.numReadPerVectorA//2)*kernel["InnerUnroll"]*kernel["MIWaveTileA"] + (self.numReadPerVectorB//2)*kernel["InnerUnroll"]*kernel["MIWaveTileB"]
              numMfmaForHalfRead = 1
              latencyLeft = self.miLatencyLeft
              sub2 = min(2, self.miLatencyLeft) # value for subtraction should not exceed self.miLatencyLeft
              for i in range(numHalfReads):
                latencyLeft -= sub2
                if latencyLeft < 0:
                  numMfmaForHalfRead += 1
                  latencyLeft = self.miLatencyLeft - sub2
              lwStartMfmaIndex = numMfmaPerIter * (kernel["LoopIters"] - 1 - self.numItersPLR) + numMfmaForHalfRead
            else:
            # we have enough vgprBuffer to schedule localReads in the front of loop
              lwStartMfmaIndex = numMfmaForLRCurr
          else:
            lwStartMfmaIndex = numMfmaPerIter * (kernel["LoopIters"] - 1 - self.numItersPLR) + self.numMfmaForLR
          # add latency count for local read (applicable for all one buffer scheduling cases)
          lwStartMfmaIndex += latencyForLRCount
        #########
        # Get LocalWritePerMfma
        #########
        if lwStartMfmaIndex > self.lwEndMfmaIndex:
          lwStartMfmaIndex = self.lwEndMfmaIndex
        numMfmaCanSched = self.lwEndMfmaIndex - lwStartMfmaIndex + 1
        numLoadsA = kernel["DepthU"]*kernel["MacroTileA"]//int(kernel["GlobalLoadVectorWidthA"]*kernel["NumThreads"])
        numLoadsB = kernel["DepthU"]*kernel["MacroTileB"]//int(kernel["GlobalLoadVectorWidthB"]*kernel["NumThreads"])
        # PGR2 + DirectToVgpr case, no need to schedule local write. Change numLoads to 0
        if kernel["PrefetchGlobalRead"] == 2:
          if kernel["DirectToVgprA"]:
            numLoadsA = 0
          if kernel["DirectToVgprB"]:
            numLoadsB = 0
        writesToSched = (numLoadsA + numLoadsB - 1) * PRECISION
        # In StoreCInUnroll case, add StoreC code related code to writesToSched
        if kernel["StoreCInUnroll"]:
          numStoreCUnrollCode = len(list(self.StoreCUnrollCode.items()))
          writesToSched += numStoreCUnrollCode * PRECISION
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
        while oldValue != newValue and loop < 10:
          loop += 1
          oldValue = newValue
          newValue = roundUp((writesToSched+1 + (oldValue - (writesToSched+1) % oldValue) + oldValue%PRECISION) / numMfmaCanSched)
          newValue = max(1, newValue) # minimum 1 to avoid 0 division
        numLocalWriteModPerMfma = newValue

      #####
      # Assign GRPM and LWPM
      #####
      # HOW THIS WORK
      # padding each globalReadInstruction to 100 with empty instruction,
      # each mfma will schedule intructions GRPM*100 times from padded globalReadInstruction.
      #   Ex. GRPM = 0.5
      #        GR ---------99--------- GR --------99---------- GR
      #   mfma --49-- mfma --49-- mfma --49-- mfma --49-- mfma --49--
      self.numGlobalReadInsPerMfma = roundUp(kernel["GlobalReadPerMfma"]*PRECISION)

      # HOW THIS WORK
      # padding each globalReadInstruction to 100 with empty instruction,
      # each mfma will schedule intructions GRPM*100 times from padded globalReadInstruction.
      #   Ex. LWPM = 0.5
      #        LW ---------99--------- LW --------99---------- LW
      #   mfma --49-- mfma --49-- mfma --49-- mfma --49-- mfma --49--
      # calculate numLocalWriteModPerMfma for all LocalWritePerMfma cases (use this for parameter check in LocalWritePerMfma!=1 case)
      if kernel["PrefetchGlobalRead"] == 1:
        # In PGR1:
        #   Larger LWPM can provide more latency to hide global read
        #   However, larger LWPM may cause mfma bubbles
        #   we set LWPM to 1 unless it requires larger LWPM to enable 1LDSB
        if kernel["1LDSBuffer"]:
          self.numLocalWriteModPerMfma = max(numLocalWriteModPerMfma,PRECISION)
        else:
          self.numLocalWriteModPerMfma = PRECISION
      else:
        self.numLocalWriteModPerMfma = numLocalWriteModPerMfma
      if kernel["EnableMatrixInstruction"] and kernel["ScheduleIterAlg"] == 3 and kernel["PrefetchGlobalRead"] == 2:
        if kernel["LocalWritePerMfma"] != -1:
          valueFromParam = roundUp(kernel["LocalWritePerMfma"]*PRECISION)
          # parameter check (to avoid mismatch due to overwrapping)
          if (valueFromParam < self.numLocalWriteModPerMfma):
            tPrint(3, "LocalWritePerMfma (%f) is too small. Auto-adjusted." % kernel["LocalWritePerMfma"])
            valueFromParam = self.numLocalWriteModPerMfma
          self.numLocalWriteModPerMfma = valueFromParam
        elif (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]):
          # DTV + PGR2 case, DTV load is not a part of LocalWrite scheduing and NumLocalWritePerMfma can be very small
          # we set minimum value of NumLocalWritePerMfma for DTV + PGR2
          # currently, minimum is set as 1/6 (means 1 local write per 6 MFMA)
          minNumLocalWritePerMfma = int((1/6) * PRECISION)
          self.numLocalWriteModPerMfma = max(self.numLocalWriteModPerMfma, minNumLocalWritePerMfma)


      ##################################
      numGlobalReadInsPerIter = numMfmaPerIter * self.numGlobalReadInsPerMfma
      numLocalWriteModPerIter = numMfmaPerIter * self.numLocalWriteModPerMfma
      # if numGlobalReadInsPerMfma>1, we still want to schedule only 1 GlobalReadIncCode per mfma
      # inserting empty CodeModule so that generator will schedule 1 GlobalReadIncCode 1 empty CodeModule if numGlobalReadInsPerMfma=2
      numEmptyGlobalReadIncCode = self.numGlobalReadInsPerMfma - 1

      # If numLocalWriteModPerMfma is not multiple of 100,
      # last globalread will be scheduled at lwEndMfmaIndex,
      # and last localwrite will be scheduled at lwEndMfmaIndex - 1
      # so we offset lwEndMfmaIndex by 1 mfma
      if kernel["PrefetchGlobalRead"] == 2 and self.numLocalWriteModPerMfma % PRECISION != 0:
        numMfmaBetweenLWandBarrier -= 1
      def assignParamSplitLds(numMfmaBetweenLWandBarrier):
        if not kernel.enabledSplitLDS:
          return numMfmaBetweenLWandBarrier
        # how many local reads in terms of mfma indices (height)
        # total number of instructions (total) minus the instructions prefetched outside of loop (spent), divided by mfma bubble (width)
        issueLatency = max(self.localReadInstructionA.IssueLatency, self.localReadInstructionB.IssueLatency) * 2
        width = self.miLatencyLeft // issueLatency
        width = max(width, 1)
        spent = self.numItersPLR * (self.numReadsPerIterA + self.numReadsPerIterB)
        total = kernel["LoopIters"]//self.numIterPerCoalescedReadA*self.numReadsPerIterA + \
                kernel["LoopIters"]//self.numIterPerCoalescedReadB*self.numReadsPerIterB
        height = int(ceil((total-spent)/width))
        # how many local writes
        localWritesToSched = self.localWriteACode.countType(Code.LocalWriteInst) + \
                             self.localWriteBCode.countType(Code.LocalWriteInst)
        if kernel["StoreCInUnroll"]:
          # in StoreCInUnroll case, add number of storeC related code here
          # add store C related code to itemsLWToSched
          numStoreCUnrollCode = len(list(self.StoreCUnrollCode.items()))
          localWritesToSched += numStoreCUnrollCode
        localWritesPerMfma = self.numLocalWriteModPerMfma / PRECISION # was scaled by PRECISION
        # _numMfmaBetweenLastLWandBarrier: a function of 'spacing', which is num of mfma instructions until local write starts
        _numMfmaBetweenLastLWandBarrier = lambda spacing : self.barrierMfmaIndex + 1 - ceil(localWritesToSched/localWritesPerMfma) - spacing
        addrIncToSched = sum(1 for codemod in [globalReadIncACode, globalReadIncBCode] if len(codemod.items()))
        if uDu < kernel["DepthULdsDivisor"] - 1:
          if kernel["1LDSBuffer"] and kernel["PrefetchLocalRead"] > 1:
            # space the stream of local writes so that 1st local write is scheduled after last local read,
            # but give it 2 mfma's worth of headroom
            spacing = 2 + height
          else:
            # can start ds_write/buffer_load as soon as loop starts, but give it 1 mfma's worth of headroom
            spacing = 1
        else:
          # query how much spacing we have by calling lambda(0), minus the original 'numMfmaBetweenLWandBarrier'
          # we get the spacing that results in exactly 'numMfmaBetweenLWandBarrier' between last write and barrier
          spacing = _numMfmaBetweenLastLWandBarrier(0) - numMfmaBetweenLWandBarrier + addrIncToSched - 1
        return max(0, _numMfmaBetweenLastLWandBarrier(spacing))

      numMfmaBetweenLWandBarrier = assignParamSplitLds(numMfmaBetweenLWandBarrier)
      # In StoreCInUnroll + num of store > 1 case, reduce numMfmaBetweenLWandBarrier to 1
      # because interval between local write and read is already added by StoreCInUnroll code
      if kernel["StoreCInUnroll"] and self.getNumberOfStoreCInTemplate(kernel) > 1:
        numMfmaBetweenLWandBarrier = min(numMfmaBetweenLWandBarrier, 1)

      # set and adjust lwEndMfmaIndex
      self.setAndAdjustLwEndMfmaIndex(kernel, tensorParametersA, tensorParametersB, numMfmaBetweenLWandBarrier, lastLoop)

      localWriteEndIter = self.lwEndMfmaIndex//numMfmaPerIter
      localWriteEndIter = min(kernel["LoopIters"] - 1, localWriteEndIter)
      assert localWriteEndIter < kernel["LoopIters"]
      assert self.lwEndMfmaIndex < numMfmaPerIter*kernel["LoopIters"]
    else:
      numGlobalReadInsPerIter = roundUp(kernel["GlobalReadPerMfma"] * PRECISION) if kernel["GlobalReadPerMfma"] > 0 else PRECISION
      numLocalWriteModPerIter = roundUp(kernel["LocalWritePerMfma"] * PRECISION) if kernel["LocalWritePerMfma"] > 0 else PRECISION
      numEmptyGlobalReadIncCode = numGlobalReadInsPerIter - 1

    numLocalWritesPerSched = numLocalWriteModPerIter if kernel["ScheduleIterAlg"] != 3 else self.numLocalWriteModPerMfma

    if not self.scheduleGlobalRead:
      # put everything in the header:
      self.unrollLoopHeaderCode.addCode(self.dtlsM0UpdateACode)
      self.unrollLoopHeaderCode.addCode(self.globalReadACode)
      self.unrollLoopHeaderCode.addCode(self.dtlsM0UpdateBCode)
      self.unrollLoopHeaderCode.addCode(self.globalReadBCode)
      self.unrollLoopHeaderCode.addCode(globalReadIncACode)
      self.unrollLoopHeaderCode.addCode(globalReadIncBCode)
      if kernel["EnableMatrixInstruction"] and kernel["ScheduleIterAlg"] == 3:
        self.grEndMfmaIndex = 0
        itemsGRToSchedLater = []
    else:
      self.unrollLoopHeaderCode.addCode(self.globalReadACode.header)
      self.unrollLoopHeaderCode.addCode(self.globalReadBCode.header)

      # Add all loads from middle as individual schedulable items
      # when using PGR2, put global read instruction right after corresponding localWrite instruction
      if kernel["PrefetchGlobalRead"] == 2 or kernel.enabledSplitLDS:
        itemsGRToSched =  []
        itemsGRToSchedLater = list(self.globalReadACode.middle.items()) + \
                         list(self.globalReadBCode.middle.items())
        itemsGRToSchedLaterDTV = []
        # PGR2 and DirectToVgpr case, schedule global read for DirectToVgpr separately after registers are used for mfma
        if kernel["EnableMatrixInstruction"]:
          if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
            itemsGRToSchedLater = list(self.globalReadACode.middle.items())      # not DirectToVgpr (A has non-DirectToVgpr load)
            itemsGRToSchedLaterDTV = list(self.globalReadBCode.middle.items()) # DirectToVgpr (B has DirectToVgpr load)
          # add to self.perIterGlobalReadCodeDTV to schedule DirectToVgpr
          while itemsGRToSchedLaterDTV:
            itemGR = itemsGRToSchedLaterDTV.pop(0)
            self.perIterGlobalReadCodeDTV[kernel["LoopIters"] - 1].addCode(itemGR)
        if kernel.enabledSetPrioSplitLDS and itemsGRToSchedLater:
          itemsGRToSchedLater.insert(1, Code.Inst("s_setprio", "3", "top priority for load"))
          itemsGRToSchedLater.insert(len(itemsGRToSchedLater), Code.Inst("s_setprio", "0", ""))
      else:
        itemsGRToSched =  list(self.globalReadACode.middle.items()) + \
                        list(self.globalReadBCode.middle.items())
        itemsGRToSchedLater = []
        if kernel["StoreCInUnroll"]:
          # in StoreCInUnroll case, add loadC code here (self.LoadCTemplate is empty for no loadC required case)
          # The location to insert LoadC is decided based on DirectToLds and DirectToVgpr setting
          # 1) No Lds write case (Both DirectToLds or DirectToVgpr enabled), insert Load C before Load A and B
          if kernel["NoLdsWriteCode"]:
            itemsGRToSched =  list(list(self.LoadCUnrollCode.itemList) + self.globalReadACode.middle.items()) +\
                            list(self.globalReadBCode.middle.items())
          # 2) DirectToVgpr only enabled case, insert Load C before Load for DirectToVgpr
          elif kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
            itemsGRToSched =  list(self.globalReadACode.middle.items()) + list(self.LoadCUnrollCode.itemList) +\
                            list(self.globalReadBCode.middle.items())
          # 3) no DirectToVgpr/Lds case, insert Load C after Load A,B
          else:
            itemsGRToSched += list(self.LoadCUnrollCode.itemList)

      itemsGRToSchedTemp = []
      for i in range(len(itemsGRToSched)):
        itemsGRToSchedTemp.append(itemsGRToSched.pop(0))
        for j in range(PRECISION-1):
          itemsGRToSchedTemp.append(Code.Module())
      itemsGRToSched = itemsGRToSchedTemp

      itemsGRIncToSched = []
      if kernel["EnableMatrixInstruction"] and kernel["ScheduleIterAlg"] == 3:
        # for SIA3, we can break GlobalReadIncCode to avoid mfma bubbles
        if kernel["PrefetchGlobalRead"] == 2:
          # skip to schedule global read for PGR2 first mfma
          for i in range(numEmptyGlobalReadIncCode+1):
            imod = Code.Module()
            itemsGRIncToSched.append(imod)
        numInst = globalReadIncACode.countType(Code.Inst) + globalReadIncBCode.countType(Code.Inst)
        numInstPerMfma = max(roundUp(self.miLatencyLeft/2),1)

        globalReadInc1 = globalReadIncACode.flatitems()
        globalReadInc2 = globalReadIncBCode.flatitems()
        if self.isSwapGlobalReadOrderForDtvOrDtl(kernel):
          # swap the order of readInc for DTV
          globalReadInc1, globalReadInc2 = globalReadInc2, globalReadInc1
        globalReadIncItems = globalReadInc1 + globalReadInc2
        if kernel["StoreCInUnroll"] and  kernel["PrefetchGlobalRead"] == 2:
          # PGR=2 + StoreCInUnroll case, add first LoadC after IncA, second LoadC (if exist) after IncB
          tmpList = list(self.LoadCUnrollCode.itemList)
          dummyList = [ Code.Module() for i in range (numInstPerMfma - 1) ]
          if len(tmpList) > 0:
            # first LoadC
            globalReadIncItems = globalReadInc1 + tmpList[0:1] + dummyList + globalReadInc2
          if len(tmpList) > 1:
            # second LoadC
            globalReadIncItems += tmpList[1:]
          # add len(LoadCUnrollCode.itemList) to numInst
          numInst += len(tmpList)
        numMfmaToSched = roundUp(numInst/numInstPerMfma)
        for j in range(numMfmaToSched):
          imod = Code.Module()
          count = 0
          while globalReadIncItems and count < numInstPerMfma:
            tempInst = globalReadIncItems.pop(0)
            imod.addCode(tempInst)
            if tempInst.countType(Code.Inst):
              count += 1
          itemsGRIncToSched.append(imod)
          for i in range(numEmptyGlobalReadIncCode):
            imod = Code.Module()
            itemsGRIncToSched.append(imod)
      else:
        itemsGRIncToSched.append(globalReadIncACode)
        for i in range(numEmptyGlobalReadIncCode):
          imod = Code.Module()
          itemsGRIncToSched.append(imod)
        itemsGRIncToSched.append(globalReadIncBCode)
        for i in range(numEmptyGlobalReadIncCode):
          imod = Code.Module()
          itemsGRIncToSched.append(imod)

      if kernel["EnableMatrixInstruction"] and kernel["ScheduleIterAlg"] == 3:
        # Loop in PGR1: GlobalRead -> GlobalReadInc -> LocalWrite
        # but GlobalReadInc shouldn't block LocalWrite so we count them out
        # Loop in PGR2: GlobalReadInc -> LocalWrite/GlobalRead pair
        # since LocalWrite/GlobalRead pair depends on GlobalReadInc, we count in only GlobalReadInc
        if kernel["PrefetchGlobalRead"] == 2:
          loadsToSched = len(itemsGRIncToSched)
        else:
          loadsToSched = len(itemsGRToSched)

        # Here is to adjust scheduling silently in order to have validation pass.
        # Better way is to use larger globalReadPerMfma.
        ## schedule more instructions at first iteration if no enough mfma to schedule globalRead
        self.grEndMfmaIndex = max(0, roundUp(loadsToSched/self.numGlobalReadInsPerMfma) - 1)
        if self.grEndMfmaIndex > self.lwEndMfmaIndex:
          schedNumForIter0 = numGlobalReadInsPerIter + (self.grEndMfmaIndex - self.lwEndMfmaIndex) * self.numGlobalReadInsPerMfma
          self.grEndMfmaIndex = self.lwEndMfmaIndex
        else:
          schedNumForIter0 = numGlobalReadInsPerIter
        if kernel["PrefetchGlobalRead"] == 1:
          globalReadIncEndMfmaIndex = self.grEndMfmaIndex + roundUp(len(itemsGRIncToSched)/self.numGlobalReadInsPerMfma)
          endIter = roundUp((globalReadIncEndMfmaIndex+1)/numMfmaPerIter)
        else:
          endIter = roundUp((self.grEndMfmaIndex+1)/numMfmaPerIter)
        ## schedule more instructions at first iteration if no enough mfma to schedule globalRead + globalReadInc
        if endIter > kernel["LoopIters"]:
          endIter = kernel["LoopIters"]
          if kernel["PrefetchGlobalRead"] == 1:
            schedNumForIter0 += (globalReadIncEndMfmaIndex+1 - kernel["LoopIters"]*numMfmaPerIter) * self.numGlobalReadInsPerMfma

      # SIA 1 or 2
      # distribute the instructions in itemsGRToSched evenly as possible to iterations: perIterGlobalReadCode[0,endIter)
      # last one is perIterGlobalReadCode[endIter-1],
      # Ideally:     endIter <= localWriteEndIter,
      #              then put M0 updateCode (if any) and first 'schedNumForIter0' GR-inst in perIterGlobalReadCode[0]
      #              put every numGlobalReadInsPerIter GR-insts in perIterGlobalReadCode[1]~[endIter-1]
      # corner case: endIter > localWriteEndIter, set endIter = localWriteEndIter,in this case, schedNumForIter0 will > 1
      #              and perIterGlobalReadCode[0] would need to schedule more instructions
      else:
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

      # insert dtlsM0UpdateACode dtlsM0UpdateBCode code
      if self.globalReadACode.middle.items():
        self.globalReadACode.middle.items()[0].items().insert(0,self.dtlsM0UpdateACode)
      if self.globalReadBCode.middle.items():
        self.globalReadBCode.middle.items()[0].items().insert(0,self.dtlsM0UpdateBCode)

      itemsGRToSched.extend(itemsGRIncToSched)
      # append 'n' global load at a time
      # append global load(S) first 'number of global load(s)' determined by schedNumForIter0
      for item in itemsGRToSched[:schedNumForIter0]:
        self.perIterGlobalReadCode[0].addCode(item)
      itemsGRToSched = itemsGRToSched[schedNumForIter0:] # trim the scheduled GRs, do the rest in the following loop

      for u in range(1, endIter):
        # append itemPerIter GR for each iteration,
        # and trim the scheduled ones at the end of loop
        itemPerIter = 1 * numGlobalReadInsPerIter
        try:
          for item in itemsGRToSched[:itemPerIter]:
            self.perIterGlobalReadCode[u].addCode(item)
            lastLoadIter = u
          itemsGRToSched = itemsGRToSched[itemPerIter:]
        except IndexError:
          break # itemsGRToSched is 0-length, no code left to schedule

      assert not itemsGRToSched # should have scheduled everything already, itemsGRToSched should be empty

      # adjustment for StoreCInUnroll
      # lastLoop case, make the last perIterGlobalReadCode[] (LoopIters-1) empty
      # otherwise, mixing global read inc code and StoreCInUnroll post code could cause memory access issue
      if kernel["StoreCInUnroll"] and lastLoop:
        lastIter = kernel["LoopIters"] - 1
        prevLastIter = max(0, lastIter - 1)
        if prevLastIter < lastIter:
          while self.perIterGlobalReadCode[lastIter].items():
            self.perIterGlobalReadCode[prevLastIter].addCode(self.perIterGlobalReadCode[lastIter].items().pop(0))

      self.perIterGlobalReadCode[endIter-1].addCode(self.globalReadACode.footer)
      self.perIterGlobalReadCode[endIter-1].addCode(self.globalReadBCode.footer)

    # Now schedule the writes:
    if not self.scheduleLocalWrite:
      # if no scheduleLocalWrite - just add writes to localWritelocalWriteEndIter
      # If PGR=0, writes have to be done immediately following the loads - no opportunity to schedule
      #   so don't add to schedule, these will be added separately and before the first iter
      if kernel["PrefetchGlobalRead"]:
        # do we need a module here? That would prevent these from being scheduled
        imod = self.perIterLocalWriteCode[localWriteEndIter].addCode(Code.Module())
        if self.enable["Wait"]:
          imod.addCode(
              self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, \
              "1wait for global read"))
        imod.addComment1("local write A")
        imod.addCode(self.localWriteACode)
        imod.addComment1("local write B")
        imod.addCode(self.localWriteBCode)
      if kernel["EnableMatrixInstruction"] and kernel["ScheduleIterAlg"] == 3:
        self.lwStartMfmaIndex = self.lwEndMfmaIndex
    else:
      #################
      # create a plan #
      #################
      itemsLWToSched = list(self.localWriteACode.items()) + list(self.localWriteBCode.items())
      if kernel["PrefetchGlobalRead"] == 2:
        # PrefetchGlobalRead + DirectToLds case, need to add dummy list to insert global read
        tmpList = []
        numDummy = 0
        lenA = len(list(self.globalReadACode.middle.items()))
        lenB = len(list(self.globalReadBCode.middle.items()))
        if self.isSwapGlobalReadOrderForDtvOrDtl(kernel):
          # swap A and B (SwapGlobalReadOrder case, the actual content is swapped (B is in globalReadACode). Need adjustment)
          lenA, lenB = lenB, lenA
        if kernel["DirectToLdsA"]:
          if not kernel["StoreCInUnroll"]:
            # PGR2 + DTL (and not SCIU) case, footer code is added in middle. Need to subtract 1 (for footer inst)
            lenA -= 1
          numDummy += lenA
        if kernel["DirectToLdsB"]:
          if not kernel["StoreCInUnroll"]:
            # PGR2 + DTL (and not SCIU) case, footer code is added in middle. Need to subtract 1 (for footer inst)
            lenB -= 1
          numDummy += lenB
        for i in range(numDummy):
          tmpList.append(Code.Module())
        # add dummy at the top of the list
        itemsLWToSched = tmpList + itemsLWToSched
      if kernel["StoreCInUnroll"]:
        # in StoreCInUnroll case, add storeC related code here
        # add store C related code to itemsLWToSched
        tmpList = list(self.StoreCUnrollCode.items())
        itemsLWToSched += tmpList
      # extend localWrite by inserting empty Module
      itemsLWToSchedTemp = []
      for i in range(len(itemsLWToSched)-1):
        itemsLWToSchedTemp.append(itemsLWToSched.pop(0))
        for j in range(PRECISION-1):
          itemsLWToSchedTemp.append(Code.Module())
      if itemsLWToSched:
        itemsLWToSchedTemp.append(itemsLWToSched.pop(0))
        for i in range(numLocalWritesPerSched + numLocalWritesPerSched % PRECISION - len(itemsLWToSchedTemp) % numLocalWritesPerSched):
          itemsLWToSchedTemp.append(Code.Module())
      itemsLWToSched = itemsLWToSchedTemp
      # This counts the number of modules which contain a ds_write
      # Scheduler below keeps all writes in the same module in same iteration
      # so this is better match to what it is trying to do
      # writesToSched = sum(1 for item in itemsLWToSched if item.countType(Code.LocalWriteInst))
      writesToSched = len(itemsLWToSched)
      # assign schedule index
      if kernel["EnableMatrixInstruction"] and kernel["ScheduleIterAlg"] == 3:
        self.lwStartMfmaIndex = self.lwEndMfmaIndex - max(1,roundUp(writesToSched/numLocalWritesPerSched)) + 1
        if self.lwStartMfmaIndex < self.grEndMfmaIndex:
          self.lwStartMfmaIndex = self.grEndMfmaIndex
        if kernel["PrefetchGlobalRead"]==2:
          # adjustment for PGR2
          # new lwStartMfmaIndex value calculated with numLocalWritesPerSched should not be smaller than pre-calculated lwStartMfmaIndex
          # (which can cause overwrapping)
          if self.lwStartMfmaIndex < lwStartMfmaIndex:
            self.lwStartMfmaIndex = lwStartMfmaIndex
        # after adjusting lwStartMfmaIndex, lwStartMfmaIndex should not be larger than lwEndMfmaIndex
        if self.lwStartMfmaIndex > self.lwEndMfmaIndex:
          self.lwStartMfmaIndex = self.lwEndMfmaIndex
        startIter = self.lwStartMfmaIndex//numMfmaPerIter
        assert startIter < localWriteEndIter+1 # startIter should be at or before the endIter
      else:
        startIter = localWriteEndIter - roundUp(writesToSched/numLocalWritesPerSched) + 1
        # - can't move a write past the load it depends on
        #   as a simplification, don't move writes past any loads
        if startIter < lastLoadIter:
          startIter = lastLoadIter

      readsToWait = len(list(self.localWriteACode.items())) + len(list(self.localWriteBCode.items()))
      readsToWaitDTV = 0
      # add waitcnt for DirectToVgpr. Delaying wait for DirectToVgpr global read
      if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
        # DirectToVgpr + swapGlobalRead case, actual DTV load is in self.globalReadBCode (due to swap).
        # Need to check self.globalReadBCode
        readsToWaitDTV += len(list(self.globalReadBCode.middle.items()))
      # add waitcnt for StoreCInUnroll. Delaying wait for Load C
      readsToWait += numGlobalReadC

      readsToWaitNGLL = readsToWait

      localwriteCnt = 0
      for u in range(startIter, localWriteEndIter+1):
        if u==(localWriteEndIter):
          itemPerIter = len(itemsLWToSched) # schedule all remaining activity
        else:
          itemPerIter = numLocalWriteModPerIter
          # if localwrite is not multiple of numLocalWriteModPerIter, fill last iteration first.
          # make sure numLocalWriteModPerIter is enough to schedule localwrite
          # TODO: if numLocalWriteModPerIter is not enough to schedule localwrite, need smarter way to distribute localWrite
          if u == startIter and kernel["ScheduleIterAlg"] == 3:
            itemPerIter = numLocalWriteModPerIter - (self.lwStartMfmaIndex % numMfmaPerIter) * numLocalWritesPerSched

        for item in itemsLWToSched[:itemPerIter]:
          # Use a module to ensure these pieces stay together in the sub-iter scheduler
          imod = Code.Module("LocalWriteMod%u"%u)
          imodNGLL = Code.Module("LocalWriteMod%u"%u)
          writesPerItem = item.countType(Code.LocalWriteInst)
          readsToWaitAdjustForStoreC = 0
          if kernel["StoreCInUnroll"] and not firstIter and kernel["PrefetchGlobalRead"]==2:
            # get number of StoreC in template
            readsToWaitAdjustForStoreC += self.getNumberOfStoreCInTemplate(kernel)
          if writesPerItem:
            imod.addComment0("sched write - iter %u writesPerItem=%u"%(u,writesPerItem))
            imodNGLL.addComment0("sched write - iter %u writesPerItem=%u"%(u,writesPerItem))
            # if writesPerItem>1 this indicates multiple LocalWrites in the same module
            # this happens in some transpose cases.  Here the first write needs to wait
            # for the associated global read to finish, then the remaining writes can flow
            # TODO - can schedule these writes across iters, should figure this out above
            readsToWait = readsToWait - 1
            readsToWaitNGLL = readsToWaitNGLL - 1
            if uDu < kernel["DepthULdsDivisor"]-1:
              imod.addComment0("no wait vmcnt except for in the last subLdsLoop")
            else:
              imod.addCode(Code.WaitCnt(self.version, -1, min(maxVmcnt, readsToWait + readsToWaitDTV + readsToWaitAdjustForStoreC), \
                "wait for global read before writing to local"))
              imodNGLL.addCode(Code.WaitCnt(self.version, -1, min(maxVmcnt, readsToWaitNGLL  + readsToWaitDTV + readsToWaitAdjustForStoreC), \
                "wait for global read before writing to local"))
          if kernel["StoreCInUnroll"] or kernel["PrefetchGlobalRead"]==2:
            if "s_waitcnt" in str(item) and "__placeholder__" in str(item):
              # waitcnt adjustment for StoreCInUnroll
              readsToWaitAdjust = readsToWait + readsToWaitDTV - numGlobalReadC
              if kernel["PrefetchGlobalRead"]==2:
                # PGR=2 special cases
                if (kernel["AtomicAddC"] or not kernel["ProblemType"]["UseBeta"]):
                  # no Load C case
                  if not firstIter:
                    # PGR=2 and not firstIter case, __placeholder__ includes num of storeC from previous Iter
                    readsToWaitAdjust += readsToWaitAdjustForStoreC
                else:
                  # Load C case
                  # adjustment for waitcnt for loadC
                  if kernel["StoreCInUnroll"] and self.StoreCUnrollLoadCWaitComment in str(item):
                    # readsToWaitDTV should not be added for loadC waitcnt
                    readsToWaitAdjust -= readsToWaitDTV
              if kernel["NoLdsWriteCode"] and kernel["PrefetchGlobalRead"]!=2:
                # DirectToLds or DirectToVgpr for both A and B case, use  the number of global read for both A and B as vmcnt (only for PGR=1)
                readsToWaitAdjust = len(list(self.globalReadACode.middle.items())) + len(list(self.globalReadBCode.middle.items()))
              item = str(item).replace("__placeholder__", str(readsToWaitAdjust))

          imod.addCode(item)
          # schedule global instruction that need to be scheduled later
          if localwriteCnt % PRECISION == (numLocalWritesPerSched % PRECISION):
            reads = 0
            while itemsGRToSchedLater:
              itemGR = itemsGRToSchedLater[0]
              readsInc = itemGR.countType(Code.GlobalReadInst)
              if kernel["StoreCInUnroll"] and readsInc == 0:
                # adjustment for StoreCInUnroll
                # count buffer_load if it exist but not counted
                readsInc += str(itemGR).count("_buffer_load")
              reads = reads + readsInc
              if reads > 1:
                break
              if "s_waitcnt" in str(itemGR) and "__placeholder__" in str(itemGR):
                itemGR2 = (str(itemGR).replace("__placeholder__", str(readsToWait)))
                imod.addText(itemGR2)
              else:
                imod.addCode(itemGR)
              readsToWait = readsToWait + readsInc # GR instruction increments vmcnt
              itemsGRToSchedLater.pop(0)
          localwriteCnt += 1
          self.perIterLocalWriteCode[u].addCode(imod)

          imodNGLL.addCode(copy.deepcopy(item))
          if lastLc:
            # local write code for NGLL should be updated at the last lc
            # in init acc opt case, the last inner loop generated is not for the last lc.
            # in that case, local write code for NGLL is not as expected.
            self.perIterLocalWriteCodeNGLL[u].addCode(imodNGLL)

        itemsLWToSched = itemsLWToSched[itemPerIter:]

      # should never run out of items to schedule
      assert not itemsLWToSched # should have scheduled everthing already

    if grBackup is not None:
      del grBackup

  ##############################################################################
  # Schedule work into the each unroll loop iteration
  # localReadCode is the local reads for this loop iteration
  #  (returned by localReadDo). The instructions in localReadCode
  #  will retain their relative order, but may be interleaved
  #  with instructions from otherCode.

  # globalReadCode is the 'other' buffer loads and addr increments
  # localWriteCode is the 'other' local writes
  #  to schedule in with the ds reads.  The instructions
  #  will retain their relative order, but may be interleaved
  #  with instructions from localReadCode.

  # pointerCode contains local pointer changes (if needed)
  # waitCode contains s_waitcnt before macs.
  #   - Cannot be "" or None
  #   - may be empty Module if not waiting is desired (perhaps for debug)
  #   - may be multiple instructions (ConservativeWaitCnt)
  #   - typically is a single Code.WaitCnt.  This routine will
  #     modify the lgkmcnt to account for any scheduling decisions.
  #     If this is not desired, add the waitCnt to pointerCode and
  #     set waitCode to an empty module
  # macIterCode contains the mac iters.  May be a macro call.
  #
  # returns: a Module with the combined, optimally scheduled
  #  localReadCode + otherCode
  ##############################################################################
  def makeSubIterSchedule(self, kernel, localReadCode, iteration, pointerLWCode, pointerLRCode, waitCode, macIterCode, \
      waitLWCode = Code.Module(), syncCode = Code.Module(), packCode = Code.Module(), isDTVodd = False, NLLlast = False):

    iterCode = Code.Module()
    globalReadCode = copy.deepcopy(self.perIterGlobalReadCode[iteration])
    globalReadCodeDTV = self.perIterGlobalReadCodeDTV[iteration]
    origLenGlobalReadCodeDTV = len(list(self.perIterGlobalReadCodeDTV[iteration].items()))
    localWriteCode = self.perIterLocalWriteCode[iteration]
    isBarrier = kernel["LoopIters"] - self.numItersPLR
    hasLocalRead = localReadCode.countType(Code.LocalReadInst)
    # Default schedule is other, local reads, then local writes:
    if self.scheduleIterAlg==0:
      # simple schedule, just add the modules in-order
      iterCode.addCode(globalReadCode)
      iterCode.addCode(globalReadCodeDTV)
      # pop out all items
      while len(list(globalReadCodeDTV.items())):
        globalReadCodeDTV.items().pop(0)
      iterCode.addCode(waitLWCode)
      iterCode.addCode(syncCode)
      iterCode.addCode(localReadCode)
      iterCode.addCode(localWriteCode)
      iterCode.addCode(pointerLWCode)
      iterCode.addCode(pointerLRCode)
      iterCode.addCode(waitCode)
      iterCode.addCode(packCode)
      iterCode.addCode(macIterCode)
    elif self.scheduleIterAlg == 1:
      iterCode.addCode(waitLWCode)
      iterCode.addCode(syncCode)
      #import pdb
      #pdb.set_trace()
      # simple algorithm - do half the reads first:
      readsToSchedule = localReadCode.countType(Code.LocalReadInst) / 2
      #localReadCode.prettyPrint()
      readItems = localReadCode.flatitems()
      while readItems:
        item = readItems.pop(0)
        #print "readsToSchedule=", readsToSchedule, "item=", item
        iterCode.addCode(item)
        readsThisItem = item.countType(Code.LocalReadInst)
        if readsThisItem:
          assert readsThisItem==1, "Scheduler assumes 1 read per item"
          readsToSchedule = readsToSchedule - 1
          if readsToSchedule == 0:
            break

      iterCode.addCode(globalReadCode)
      iterCode.addCode(globalReadCodeDTV)
      # pop out all items
      while len(list(globalReadCodeDTV.items())):
        globalReadCodeDTV.items().pop(0)

      # add rest of the reads here
      for item in readItems:
        iterCode.addCode(item)

      #move down write to be the last
      iterCode.addCode(localWriteCode)
      # tack on the pointer and mac code:
      iterCode.addCode(pointerLWCode)
      iterCode.addCode(pointerLRCode)
      iterCode.addCode(waitCode)
      iterCode.addCode(packCode)
      iterCode.addCode(macIterCode)
    elif self.scheduleIterAlg == 2:
    # SIA2 use only 1 iteration and separate compute and fetch by raising compute priority
    # 2 workgroup interleave, while WG0/WG1 doing compute, WG1/WG0 doing fetch
    # EPS need to be 1, or valu instruction will break interleave
      iterCode.addCode(globalReadCode)
      iterCode.addCode(globalReadCodeDTV)
      # pop out all items
      while len(list(globalReadCodeDTV.items())):
        globalReadCodeDTV.items().pop(0)
      iterCode.addCode(waitLWCode)
      iterCode.addCode(syncCode)
      iterCode.addCode(localReadCode)
      iterCode.addCode(waitCode)

      # interleave pack code
      # BF16 or FP16: each packCode is for one 32-bit reg,  1 packing inst: half-to-single x1
      # INT8        : each packCode is for one 32-bit regs, 3 packing inst: byte-to-half x2 + half-to-single x1
      if self.archCaps["HasEccHalf"]:
          instPerRegPack = 1 / kernel["ProblemType"]["DataType"].numRegisters() - 1
      else:
          instPerRegPack = 1 if (kernel["ProblemType"]["DataType"].numRegisters() == 0.25) else 0
      instPerPack    = int(kernel["MIInputPerThread"] * kernel["ProblemType"]["DataType"].numRegisters() * instPerRegPack)
      packItems = []
      for iui in range(kernel["InnerUnroll"]):
        packINtems = [ [] for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)) ]
        packA = packCode.findNamedItem("packA_I%s"%(iui))
        packB = packCode.findNamedItem("packB_I%s"%(iui))
        # In case localReadDo not generate pack Module
        # and findNamedItem will return None type
        # TODO: let all type have pack Module
        if not packA:
          packA = Code.Module()
        packAItems = packA.flatitems()
        if not packB:
          packB = Code.Module()
        packBItems = packB.flatitems()
        if packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        if packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        while packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        while packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)):
          packItems += packINtems.pop(0)

      macIterItem = macIterCode.flatitems()
      # pop the first code which is s_nop 1 for packing
      item = macIterItem.pop(0)

      numMfmaPerIter = self.numMfmaPerIter
      curPackIdx = 0
      packAIdx = 0
      packBIdx = 0

      for i in range(numMfmaPerIter):
        if packItems:
          # how many pack have to be done
          # calculate the data index of this mfma used for A and B
          # if i // kernel["MIWaveTile"][0]==0, mfma will use new A (need to take iu into account)
          # if i % kernel["MIWaveTile"][0]==0, mfma will use new B
          packAIdx += instPerPack if i//(kernel["MIWaveTileA"]+kernel["MIWaveTileA"]*kernel["MIWaveTileB"]*(i//(kernel["MIWaveTileA"]*kernel["MIWaveTileB"]))) == 0 else 0
          packBIdx += instPerPack if i % kernel["MIWaveTileA"] == 0 else 0
          packAIdx = packAIdx if self.needPackA else 0
          packBIdx = packBIdx if self.needPackB else 0
          numPack = (packAIdx + packBIdx)
          iterCode.addComment0("pack scheduling: packAIdx:%u, packBIdx:%u" %(packAIdx,packBIdx))
          # we put 2 pack in each mfma, "2" means A & B
          if packItems:
            for j in range(instPerPack):
              iterCode.addCode(packItems.pop(0))
              curPackIdx += 1
          if packItems:
            for j in range(instPerPack):
              iterCode.addCode(packItems.pop(0))
              curPackIdx += 1
          # since packed register need to wait 2 quad cycle to finish packing
          # we insert pack instruction if we can, or s_nop
          while curPackIdx < numPack+2:
            if packItems:
              for j in range(instPerPack):
                iterCode.addCode(packItems.pop(0))
                curPackIdx += 1
            else:
              iterCode.addInst("s_nop ","0","VALU packing writes to be consumed by matrix instruction")
              curPackIdx += 1
        if i == 0:
          if not packItems:
            tmpVgpr = self.vgprPool.checkOut(1)
            iterCode.addInst("v_mov_b32 ","v%u"%(tmpVgpr),"0x0","valu operation to have different priority")
            self.vgprPool.checkIn(tmpVgpr)
          iterCode.addInst("s_setprio ","3","Raise priority while processing macs")
        item = macIterItem.pop(0)
        iterCode.addCode(item)

      iterCode.addInst("s_setprio ","1","Raise priority while processing macs")
      if kernel["1LDSBuffer"]:
        barrier = Code.Module()
        barrier.addComment0("1 LDS buffer: read-sync-write")
        barrier.addInst("s_waitcnt lgkmcnt(0)","")
        barrier.addInst("s_barrier","")
        iterCode.addCode(barrier)
      iterCode.addCode(localWriteCode)
      iterCode.addCode(pointerLWCode)
      iterCode.addCode(pointerLRCode)
      iterCode.addInst("s_setprio ","2","Raise priority while processing macs")
      pass
    elif self.scheduleIterAlg == 3:
      iterCode.addComment0(" grEndMfmaIndex:%u, lwStartMfmaIndex:%u, lwEndMfmaIndex:%u " %(self.grEndMfmaIndex,self.lwStartMfmaIndex,self.lwEndMfmaIndex))
      printLocalWritePerMfma = kernel["PrefetchGlobalRead"] == 2 and kernel["ScheduleIterAlg"] == 3
      strForLocalWritePerMfma = ", LocalWritePerMfma:%.3f"%(self.numLocalWriteModPerMfma/self.PRECISION) if printLocalWritePerMfma else ""
      iterCode.addComment0(" numMfmaForLR:%u, barrierMfmaIndex:%u%s" %(self.numMfmaForNextLoopLR,self.barrierMfmaIndex,strForLocalWritePerMfma))
      #####
      # Prepare and Assign parameter
      ####
      if iteration == 0:
        self.localReadsVacancy = []
        self.localReadsWait = [ [] for j in range(kernel["LoopIters"])]
      self.localReadsWait[iteration] = waitCode
      numMfmaPerIter = self.numMfmaPerIter
      isBarrier = kernel["LoopIters"] - self.numItersPLR
      writeItems = list(localWriteCode.items())
      macIterItems = macIterCode.flatitems()
      skipLocalWriteWaitcnt = 0
      localReadsWaitcnt = 0
      curPackIdx = 0
      packAIdx = 0
      packBIdx = 0

      #####
      # Prepare localReadCode
      ####
      localReadCodeAB = Code.Module()
      for iui in range(kernel["InnerUnroll"]):
        localReadCodeA = localReadCode.findNamedItem("LocalReadDoA_I%s"%(iui))
        localReadCodeB = localReadCode.findNamedItem("LocalReadDoB_I%s"%(iui))
        # In case localReadDo not generate localReadCode Module
        # and findNamedItem will return None type
        # TODO: findNamedItem return Code.Module() if not found
        if not localReadCodeA:
          localReadCodeA = Code.Module()
        if not localReadCodeB:
          localReadCodeB = Code.Module()
        if localReadCodeA.items():
          localReadCodeAB.addCode(localReadCodeA.items().pop(0))
        if localReadCodeB.items():
          localReadCodeAB.addCode(localReadCodeB.items().pop(0))
        while localReadCodeA.items():
          localReadCodeAB.addCode(localReadCodeA.items().pop(0))
        while localReadCodeB.items():
          localReadCodeAB.addCode(localReadCodeB.items().pop(0))
      localReadItems = localReadCodeAB.flatitems()
      localReadItemsThisLoop = localReadItems if iteration < isBarrier else []
      localReadItemsNextLoop = localReadItems if iteration >= isBarrier else []

      #####
      # Prepare pack Code                for B:
      # since the mfma reuse B first =>    for A: mfma[A][B]
      # we need 1 vector A and 1 vector B for first mfma
      # then we prepare remaining A, then remaining B
      # BF16 or FP16: each packCode is for one 32-bit reg,  1 packing inst: half-to-single x1
      # INT8        : each packCode is for one 32-bit regs, 3 packing inst: byte-to-half x2 + half-to-single x1
      ####
      if self.archCaps["HasEccHalf"]:
          instPerRegPack = 1 / kernel["ProblemType"]["DataType"].numRegisters() - 1
      else:
          instPerRegPack = 1 if (kernel["ProblemType"]["DataType"].numRegisters() == 0.25) else 0
      instPerPack    = int(kernel["MIInputPerThread"] * kernel["ProblemType"]["DataType"].numRegisters() * instPerRegPack)
      packItems = []
      for iui in range(kernel["InnerUnroll"]):
        packINtems = [ [] for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)) ]
        packA = packCode.findNamedItem("packA_I%s"%(iui))
        packB = packCode.findNamedItem("packB_I%s"%(iui))
        # In case localReadDo not generate pack Module
        # and findNamedItem will return None type
        # TODO: let all type have pack Module
        if not packA:
          packA = Code.Module()
        packAItems = packA.flatitems()
        if not packB:
          packB = Code.Module()
        packBItems = packB.flatitems()
        if packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        if packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        while packAItems:
          for j in range(self.numReadsIterCoalescedA):
            for n in range(instPerPack):
              packINtems[j].append(packAItems.pop(0))
        while packBItems:
          for j in range(self.numReadsIterCoalescedB):
            for n in range(instPerPack):
              packINtems[j].append(packBItems.pop(0))
        for j in range(max(self.numReadsIterCoalescedA,self.numReadsIterCoalescedB)):
          packItems += packINtems.pop(0)

      # remove s_nop for packing
      # we will add s_nop if needed
      if macIterItems:
        macIterItems.pop(0)

      ####
      # scheduled local read to previous iterations
      ####
      if self.numVgprBuffer >= kernel["LoopIters"]:
        for vacancy in self.localReadsVacancy:
          # {"items","latencyLeft","atIter","atMfmaIndex","noReadsAtThisIter"}
          for localRead in list(localReadItemsThisLoop):
            # issueLatency should not exceed miLatencyLeft
            issueLatency = min(localRead.IssueLatency * 2, self.miLatencyLeft)
            if vacancy["latencyLeft"] >= issueLatency:
              if not localRead.readToTempVgpr:
                vacancy["latencyLeft"] -= issueLatency
                vacancy["items"].addCode(localRead)
                localReadItemsThisLoop.remove(localRead)
                if vacancy["atMfmaIndex"] > self.lwStartMfmaIndex - 1 and kernel["1LDSBuffer"]:
                  self.overflowedResources = 5
                # update waitCnt
                if self.numItersPLR:
                  for readsIter in range(vacancy["atIter"], iteration + self.numItersPLR):
                    if (vacancy["atMfmaIndex"] % numMfmaPerIter == 0 or readsIter != vacancy["atIter"]) and \
                        (vacancy["noReadsAtThisIter"] or readsIter <= vacancy["atIter"] + self.numItersPLR):
                      if isinstance(self.localReadsWait[readsIter], Code.WaitCnt):
                        self.localReadsWait[readsIter].lgkmcnt += 1
            else:
              # make sure the localread sequence remain the same
              vacancy["latencyLeft"] = 0
      numReadsInst = len(localReadItemsThisLoop) if iteration < isBarrier else len(localReadItemsNextLoop)

      oneBufferScheduling = kernel["1LDSBuffer"] or kernel["DirectToLdsA"] or kernel["DirectToLdsB"]

      for i in range(numMfmaPerIter):
        mfmaIndex = iteration * numMfmaPerIter + i
        lastMfmaIndex = kernel["LoopIters"] * numMfmaPerIter - 1
        iterCode.addComment0(" mfmaIndex:%u " %(mfmaIndex))

        ####
        # scheduled local read
        ####
        readLeft = numReadsInst
        latencyLeft = self.miLatencyLeft
        # with PrefetchLocalRead, localreads can interleave with mfma
        if self.numItersPLR and iteration < isBarrier:
          # take ds_write into account to schedule ds_read, assume A and B localwrite have same width (TLDS=1)
          if (mfmaIndex >= self.lwStartMfmaIndex) and not globalReadCode.countType(Code.GlobalReadInst):
            for j in range(min(len(writeItems),self.numLocalWriteModPerMfma)):
              if writeItems[j].countType(Code.LocalWriteInst):
                # issueLatency should not exceed miLatencyLeft
                latencyLeft -= min((self.tPA["localWriteInstruction"].IssueLatency*2), self.miLatencyLeft)
          readLeftLROPT = 0
          for j in range(len(localReadItemsThisLoop)):
            # issueLatency should not exceed miLatencyLeft
            issueLatency = min(localReadItemsThisLoop[j].IssueLatency*2, self.miLatencyLeft)
            latencyLeft -= issueLatency
            readLeftLROPT += 1 if latencyLeft >= 0 else 0
          # at least 1 instruction
          readLeftLROPT = max(readLeftLROPT,1)
          # evenly schedule localread with each mfma
          readLeftLREven = numReadsInst // numMfmaPerIter
          if (numReadsInst % (numMfmaPerIter)) > i:
            readLeftLREven += 1
          # we want no localreads at first mfma
          if (iteration == 0) and numMfmaPerIter != 1:
            numMfmaForLR = numMfmaPerIter - 1
            if i < numMfmaPerIter - numMfmaForLR:
              readLeftLREven = 0
              readLeftLROPT = 0
            # rest mfma help to schedule those localReads
            else:
              readLeftLREven = numReadsInst // (numMfmaPerIter-1)
              if (numReadsInst % (numMfmaPerIter-1)) >= i:
                readLeftLREven += 1
          # if there are too many localreads, change strategy to even.
          readLeft = max(readLeftLREven,readLeftLROPT)
        if not self.numItersPLR and iteration < isBarrier:
          for j in range(len(localReadItemsThisLoop)):
            # issueLatency should not exceed miLatencyLeft
            issueLatency = min(localReadItemsThisLoop[j].IssueLatency*2, self.miLatencyLeft)
            latencyLeft -= issueLatency
        # if start to schedule localwrite, but still have localreads not scheduled yet,
        # reject to use 1LDSB, since it will write and read same lds buffer at same time.
        # apply same logic to DirectToLds (need to schedule local read as 1LDS buffer)
        # TODO: force to schedule all remaining localreads before start to schedule localwrite.
        if mfmaIndex >= self.lwStartMfmaIndex and mfmaIndex <= max(self.lwEndMfmaIndex,self.barrierMfmaIndex) and \
          localReadItemsThisLoop and localWriteCode.countType(Code.LocalWriteInst) and oneBufferScheduling:
          self.overflowedResources = 5
        # DirectToVgpr case, localReadItemsThisLoop and localWriteCode.countType(Code.LocalWriteInst) do not satisfy at the same time.
        # However, it is still invaid if localReadItemsThisLoop exists when mfmaIndex > lwStartMfmaIndex
        elif (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and \
          mfmaIndex > self.lwStartMfmaIndex and mfmaIndex <= max(self.lwEndMfmaIndex,self.barrierMfmaIndex) and \
          localReadItemsThisLoop and oneBufferScheduling:
          self.overflowedResources = 5
        for j in range(readLeft):
          if localReadItemsThisLoop:
            item = localReadItemsThisLoop.pop(0)
            iterCode.addCode(item)
            if (i == 0):
              localReadsWaitcnt += 1
        if not localReadItemsThisLoop and latencyLeft > 0 and iteration < isBarrier and \
            not(mfmaIndex > self.lwStartMfmaIndex and kernel["1LDSBuffer"]):
          item = Code.Module()
          item.addComment0("localReadsVacancy: latencyLeft %d"%(latencyLeft))
          iterCode.addCode(item)
          self.localReadsVacancy.append({ "items": item, \
                                          "latencyLeft": latencyLeft, \
                                          "atIter": iteration, \
                                          "atMfmaIndex": mfmaIndex, \
                                          "noReadsAtThisIter": numReadsInst == 0, \
                                        })

        ####
        # scheduled global read
        ####
        for j in range(self.numGlobalReadInsPerMfma):
          if globalReadCode.items():
            loadText = str(globalReadCode.items().pop(0))
            if isDTVodd:
              # need to swap Vgpr set for odd code
              loadText = self.flipVregSetForDirectToVgprInGlobalRead(kernel, loadText)
            iterCode.addText(loadText)
        # schedule remaining globalReadInst
        if mfmaIndex == self.grEndMfmaIndex:
          while globalReadCode.items() and \
              (globalReadCode.countType(Code.GlobalReadInst) or kernel["PrefetchGlobalRead"] == 2):
            loadText = str(globalReadCode.items().pop(0))
            if isDTVodd:
              # need to swap Vgpr set for odd code
              loadText = self.flipVregSetForDirectToVgprInGlobalRead(kernel, loadText)
            iterCode.addText(loadText)
        # schedule remaining globalReadIncInst
        if i == numMfmaPerIter - 1:
          while globalReadCode.items():
            loadText = str(globalReadCode.items().pop(0))
            if isDTVodd:
              # need to swap Vgpr set for odd code
              loadText = self.flipVregSetForDirectToVgprInGlobalRead(kernel, loadText)
            iterCode.addText(loadText)

        ####
        # scheduled local write
        ####
        if kernel["1LDSBuffer"] and mfmaIndex == self.lwStartMfmaIndex - 1:
          barrier = Code.Module()
          barrier.addComment0("1 LDS buffer: read-sync-write")
          barrier.addInst("s_waitcnt lgkmcnt(0)","")
          barrier.addInst("s_barrier","")
          iterCode.addCode(barrier)

        if kernel["StorePriorityOpt"]:
          flagInsert = False
          if kernel["PrefetchGlobalRead"] == 2:
            lwStartOffset = 0
            if (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]):
              lwStartOffset = 2
            #  if (mfmaIndex == self.lwStartMfmaIndex or mfmaIndex == self.barrierMfmaIndex+2):
            if (mfmaIndex == self.lwStartMfmaIndex + lwStartOffset or mfmaIndex == self.barrierMfmaIndex+1) :
              flagInsert = True
          elif kernel["PrefetchGlobalRead"] == 1 and numMfmaPerIter >= 4:
            # this setting is good for fixed clock, but not good for auto clock
            #if (mfmaIndex == self.grEndMfmaIndex or mfmaIndex == self.barrierMfmaIndex+1) :
            withGL = ((not NLLlast) or (self.prefetchAcrossPersistent and kernel["PrefetchAcrossPersistentMode"] == 1))
            withDTLload = (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]) and withGL
            startIndex = 0 if withDTLload else 1
            if (mfmaIndex == startIndex or withGL and mfmaIndex == self.barrierMfmaIndex+1):
              flagInsert = True
          if flagInsert:
            iterCode.addInst("s_setprio 3","store optimization")

        if (mfmaIndex >= self.lwStartMfmaIndex):
          for j in range(self.numLocalWriteModPerMfma):
            # in case there are localWrite and globalread in same iteration
            # we need to make sure globalRead before localWrite
            if writeItems and not globalReadCode.countType(Code.GlobalReadInst):
              writeItem = writeItems.pop(0)
              # check StoreCInUnrollLoopCodeStart
              if kernel["StoreCInUnroll"]:
                if self.StoreCUnrollStartComment in str(writeItem):
                  self.StoreCUnrollLoopCodeStarted = 1 # mark as started
                if self.StoreCUnrollStoreStartComment in str(writeItem):
                  # generate all remaining pre code before the first Store C
                  while(len(self.StoreCUnrollPreCode.items()) > 0):
                    iterCode.addCode(self.StoreCUnrollPreCode.items().pop(0))
              iterCode.addCode(writeItem)
              # if there is localWrite at first mfma, need to skip it in waitcnt.
              if i == 0:
                skipLocalWriteWaitcnt += writeItem.countType(Code.LocalWriteInst)
              if not localReadItemsThisLoop:
                self.perIterLocalWriteCanSkip[iteration] += writeItem.countType(Code.LocalWriteInst)
        if mfmaIndex == self.lwEndMfmaIndex:
          while writeItems:
            writeItem = writeItems.pop(0)
            # generate all remaining pre code before the first Store C
            if kernel["StoreCInUnroll"]:
              if self.StoreCUnrollStoreStartComment in str(writeItem):
                while(len(self.StoreCUnrollPreCode.items()) > 0):
                  iterCode.addCode(self.StoreCUnrollPreCode.items().pop(0))
            iterCode.addCode(writeItem)
            if i == 0:
              skipLocalWriteWaitcnt += writeItem.countType(Code.LocalWriteInst)
            if not localReadItemsThisLoop:
              self.perIterLocalWriteCanSkip[iteration] += writeItem.countType(Code.LocalWriteInst)

        ####
        # scheduled pointer
        ####
        if mfmaIndex == self.lwEndMfmaIndex:
          iterCode.addCode(pointerLWCode)
        if i == numMfmaPerIter - 1:
          iterCode.addCode(pointerLRCode)

        ####
        # scheduled sync
        ####
        if mfmaIndex == self.barrierMfmaIndex and self.numItersPLR:
          iterCode.addCode(waitLWCode)
          iterCode.addCode(syncCode)

        ####
        # scheduled local read for next loop
        # localReads for next loop should after barrier
        ####
        latencyLeft = self.miLatencyLeft
        if self.numItersPLR and iteration >= isBarrier:
          readLeftLROPT = 0
          for j in range(len(localReadItemsNextLoop)):
            # issueLatency should not exceed miLatencyLeft
            issueLatency = min(localReadItemsNextLoop[j].IssueLatency*2, self.miLatencyLeft)
            latencyLeft -= issueLatency
            readLeftLROPT += 1 if latencyLeft >= 0 else 0
          # at least 1 instruction
          readLeftLROPT = max(readLeftLROPT,1)
          # evenly schedule localread with each mfma
          readLeftLREven = numReadsInst // numMfmaPerIter
          if (numReadsInst % (numMfmaPerIter)) > i:
            readLeftLREven += 1
          # we want no localreads at barrier mfma
          if (iteration == isBarrier) and numMfmaPerIter != 1:
            numMfmaForLR = self.numMfmaForNextLoopLR
            if i < numMfmaPerIter - numMfmaForLR:
              readLeftLREven = 0
              readLeftLROPT = 0
            # rest mfma help to schedule those localReads
            else:
              readLeftLREven = numReadsInst // (numMfmaPerIter-1)
              if (numReadsInst % (numMfmaPerIter-1)) >= i:
                readLeftLREven += 1
          # if there are too many localreads, change strategy to even.
          readLeft = max(readLeftLREven,readLeftLROPT)
        for j in range(readLeft):
          if localReadItemsNextLoop:
            item = localReadItemsNextLoop.pop(0)
            iterCode.addCode(item)
            if (i == 0):
              localReadsWaitcnt += 1

        ####
        # scheduled wait localReads
        ####
        if i == 0:
          iterCode.addCode(waitCode)

        ####
        # scheduled pack
        ####
        if packItems:
          # how many pack have to be done
          # calculate the data index of this mfma used for A and B
          # if i // kernel["MIWaveTile"][0]==0, mfma will use new A (need to take iu into account)
          # if i % kernel["MIWaveTile"][0]==0, mfma will use new B
          packAIdx += instPerPack if i//(kernel["MIWaveTileA"]+kernel["MIWaveTileA"]*kernel["MIWaveTileB"]*(i//(kernel["MIWaveTileA"]*kernel["MIWaveTileB"]))) == 0 else 0
          packBIdx += instPerPack if i % kernel["MIWaveTileA"] == 0 else 0
          packAIdx = packAIdx if self.needPackA else 0
          packBIdx = packBIdx if self.needPackB else 0
          numPack = (packAIdx + packBIdx)
          iterCode.addComment0("pack scheduling: packAIdx:%u, packBIdx:%u" %(packAIdx,packBIdx))
          # we put 2 pack in each mfma
          if packItems:
            for j in range(instPerPack):
              iterCode.addCode(packItems.pop(0))
              curPackIdx += 1
          # insert second packing code only if miLatencyLeft is large enough
          if packItems and self.miLatencyLeft > 2:
            for j in range(instPerPack):
              iterCode.addCode(packItems.pop(0))
              curPackIdx += 1
          # since packed register need to wait 2 quad cycle to finish packing
          # we insert pack instruction if we can, or s_nop
          count = 0 # count number of cycle for nop to insert
          while curPackIdx < numPack+2:
            if packItems:
              for j in range(instPerPack):
                iterCode.addCode(packItems.pop(0))
                curPackIdx += 1
            else:
              count += 1
              curPackIdx += 1
          if count:
            # insert 1 nop instruction
            iterCode.addInst("s_nop ",str(count - 1),"VALU packing writes to be consumed by matrix instruction")
        # the final mfma or DirectToVgpr + PGR2 case, we need to generate all remaining pack instructions
        # otherwise, DTV global load can overwrite pack register values
        if i == numMfmaPerIter - 1 or len(list(globalReadCodeDTV.items())) > 0:
          while packItems:
            iterCode.addCode(packItems.pop(0))

        ####
        # scheduled StoreCInUnrollPreProcess
        ####
        if kernel["StoreCInUnroll"]:
          if self.StoreCUnrollLoopCodeStarted and len(list(self.StoreCUnrollPreCode.items())) > 0:
            iterCode.addCode(self.StoreCUnrollPreCode.items().pop(0))

        ####
        # scheduled mfma
        ####
        iterCode.addCode(macIterItems.pop(0) if macIterItems else Code.Module())

        ####
        # scheduled global read for DirectToVgpr (PGR=2 only)
        ####
        numLoadVgpr = len(list(globalReadCodeDTV.items()))
        if numLoadVgpr > 0:
          interval = roundUp(numMfmaPerIter / origLenGlobalReadCodeDTV)
          if kernel["ProblemType"]["DataType"].isComplex():
            # adjustment for complex
            # limit the max of interval up to 4
            interval = min(4, interval)
          else:
            # not complex case
            # adjust(swap) the inner and outer loop index in mfmaIter to make interval as large as possible (swap for DTVA only)
            # interval can be maximum of kernel["MIWaveTile"][innerLoopIndex]
            innerLoopIndex = 1 if self.swapMfmaInnerLoop else 0
            interval = min(kernel["MIWaveTile"][innerLoopIndex], interval)
          # if number of mfma after self.grEndMfmaIndex is smaller than numMfmaPerIter, we need to use smaller interval to insert DTV load.
          # this is to ensure DTV load is generated after lwStartMfmaIndex
          intervalAfterGrEnd = kernel["LoopIters"] * numMfmaPerIter - self.lwStartMfmaIndex
          intervalMfma = min(numMfmaPerIter, intervalAfterGrEnd)
          numInstToInsert = roundUp(origLenGlobalReadCodeDTV / intervalMfma)
          remainingTimesToInsert = roundUp(numLoadVgpr / numInstToInsert)
          insertMfmaIndex = kernel["LoopIters"] * numMfmaPerIter - 1 - interval * (remainingTimesToInsert - 1)
          # avoid insertMfmaIndex getting smaller than (kernel["LoopIters"] - 1) * numMfmaPerIter
          insertMfmaIndex = max(insertMfmaIndex, (kernel["LoopIters"] - 1) * numMfmaPerIter)
          # avoid insertMfmaIndex getting smaller than lwEndMfmaIndex (DTV loads must be generated after non DTV loads)
          insertMfmaIndex = max(insertMfmaIndex, self.lwEndMfmaIndex)
          # if mfmaIndex is the last index, insert all DTV loads
          if mfmaIndex == lastMfmaIndex:
            insertMfmaIndex = mfmaIndex
            numInstToInsert = numLoadVgpr
          if mfmaIndex == insertMfmaIndex:
            for i in range(min(numLoadVgpr, numInstToInsert)):
              loadDTVText = str(globalReadCodeDTV.items().pop(0))
              if isDTVodd:
                # need to swap Vgpr set for odd code
                loadDTVText = self.flipVregSetForDirectToVgprInGlobalRead(kernel, loadDTVText)
              iterCode.addText(loadDTVText)

        ####
        # scheduled StoreCInUnrollPostProcess
        ####
        if kernel["StoreCInUnroll"]:
          numItems = len(self.StoreCUnrollPostCode.items())
          # need to make sure all global read inc is already generated
          # (iteration should be the last one)
          if numItems > 0 and iteration == kernel["LoopIters"] - 1 and len(globalReadCode.items()) == 0:
            totalMfma = kernel["LoopIters"] * numMfmaPerIter
            interval = 1
            numInstToInsert = roundUp(numItems / (totalMfma - mfmaIndex))
            remainingTimesToInsert = roundUp(numItems / numInstToInsert)
            insertMfmaIndex = totalMfma - 2 - interval * (remainingTimesToInsert - 1)
            if mfmaIndex >= insertMfmaIndex:
              for i in range(numInstToInsert):
                iterCode.addCode(self.StoreCUnrollPostCode.items().pop(0))

        if kernel["StorePriorityOpt"]:
          flagInsert = False
          if kernel["PrefetchGlobalRead"] == 2:
            #  if (mfmaIndex == self.barrierMfmaIndex or mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)):
            if (mfmaIndex == self.barrierMfmaIndex - 1 or (not NLLlast) and mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)) :
                flagInsert = True
          elif kernel["PrefetchGlobalRead"] == 1 and numMfmaPerIter >= 4:
            # this setting is good for fixed clock, but not good for auto clock
            #if (mfmaIndex == mfmaIndex == self.barrierMfmaIndex - 1 or mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)) :
            insertPos1 = self.grEndMfmaIndex
            if not kernel["NoLdsWriteCode"]:
              insertPos1 = self.lwStartMfmaIndex - 1
            withGL = ((not NLLlast) or (self.prefetchAcrossPersistent and kernel["PrefetchAcrossPersistentMode"] == 1))
            if withGL and (mfmaIndex == insertPos1 or (not NLLlast) and mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter - 1)) or \
               (not withGL) and mfmaIndex == (kernel["LoopIters"] * numMfmaPerIter // 2 - 1):
              flagInsert = True
          if flagInsert:
            iterCode.addInst("s_setprio 0","store optimization")
    else:
      assert 0, "Unsupported scheduleIterAlg=%u"%self.scheduleIterAlg

    if isinstance(waitCode, Code.WaitCnt):

      # Set the waitCount, based on the new iter schedule
      lgkmcnt = waitCode.lgkmcnt
      localReads = 0
      localWrites = 0
      if kernel["EnableMatrixInstruction"]:
        # dataAtIter      : the data we wait is read at which iteration
        # numReadsIter    : in this loop, number of iteration we have read (data used in current loop)
        dataAtIterA = iteration//self.numIterPerCoalescedReadA - self.numItersPLR
        dataAtIterB = iteration//self.numIterPerCoalescedReadB - self.numItersPLR
        numReadsIterA = min(iteration+1, kernel["LoopIters"]//self.numIterPerCoalescedReadA - self.numItersPLR)
        numReadsIterB = min(iteration+1, kernel["LoopIters"]//self.numIterPerCoalescedReadB - self.numItersPLR)
        skipReadsIterA = numReadsIterA - dataAtIterA - 1 if not dataAtIterA < max(dataAtIterA,dataAtIterB) else 0
        skipReadsIterB = numReadsIterB - dataAtIterB - 1 if not dataAtIterB < max(dataAtIterA,dataAtIterB) else 0
        # numPrefetchIter : in this loop, number of prefetch iteration we have read (data used in next loop)
        # currently we have localReadA and localReadB if iteration >= isBarrier
        # some case will not have localReads if PGR=0 or NoLoadLoop
        # known bug: wider localread + numItersPLR>1 may have chance to fail.
        numPrefetchIter = (iteration//(kernel["LoopIters"]-self.numItersPLR))*((iteration+1)-(kernel["LoopIters"]-self.numItersPLR)) if kernel["PrefetchGlobalRead"] else 0
        numPrefetchIter = 0 if iteration >= isBarrier and not hasLocalRead else numPrefetchIter
        skipReadsIterA += numPrefetchIter
        skipReadsIterB += numPrefetchIter
        # here the reads are prefetches so can skip them in the waitcnt
        # how many localreads can skip is based on how many iterations we prefetch.
        localReadsA = 0 if kernel["DirectToVgprA"] else self.numReadsPerIterA * skipReadsIterA
        localReadsB = 0 if kernel["DirectToVgprB"] else self.numReadsPerIterB * skipReadsIterB
        localReads += localReadsA + localReadsB
        # some of localReads is interleaved after waitcnt in SIA3
        if kernel["ScheduleIterAlg"] == 3 and self.numItersPLR and\
          (iteration < numReadsIterA or iteration < numReadsIterB or numPrefetchIter) and \
          self.enable["LocalRead"]:
          if ((iteration//self.numIterPerCoalescedReadA < numReadsIterA and not dataAtIterA < max(dataAtIterA,dataAtIterB)) or numPrefetchIter) and (not kernel["DirectToVgprA"]):
            localReads -= self.numReadsPerIterA
          if ((iteration//self.numIterPerCoalescedReadB < numReadsIterB and not dataAtIterB < max(dataAtIterA,dataAtIterB)) or numPrefetchIter) and (not kernel["DirectToVgprB"]):
            localReads -= self.numReadsPerIterB
          localReads += localReadsWaitcnt
        lgkmcnt += localReads
        iterCode.addComment0("numPrefetchIter=%u" % numPrefetchIter)
        iterCode.addComment0("dataAtIterA=%u numReadsIterA=%u skipReadsIterA=%u readsPerIterA=%u" % (dataAtIterA, numReadsIterA, skipReadsIterA, self.numReadsPerIterA))
        iterCode.addComment0("dataAtIterB=%u numReadsIterB=%u skipReadsIterB=%u readsPerIterB=%u" % (dataAtIterB, numReadsIterB, skipReadsIterB, self.numReadsPerIterB))
        if kernel["ScheduleIterAlg"] == 0 or kernel["ScheduleIterAlg"] == 1:
          # adjust the initial value of loop counter for DirectToVgpr
          adj = 1 if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) else 0
          for i in range (max(dataAtIterA,dataAtIterB)+adj,iteration+1):
            localWrites += self.perIterLocalWriteCode[i].countType(Code.LocalWriteInst)
        # ScheduleIterAlg=2, localwrite is after waitCnt, no need to count it's current iteration.
        if kernel["ScheduleIterAlg"] == 3:
          for i in range (max(dataAtIterA,dataAtIterB)+1,iteration):
            localWrites += self.perIterLocalWriteCode[i].countType(Code.LocalWriteInst)
          if kernel["ScheduleLocalWrite"] > 0:
            # current iteration localWrite count
            localWrites += skipLocalWriteWaitcnt
            # dataAtIter iteration localWrite count
            if self.numItersPLR:
              skipPreIterLW = self.perIterLocalWriteCanSkip[max(dataAtIterA,dataAtIterB)]
              if kernel["PrefetchGlobalRead"] == 2 and oneBufferScheduling:
                # PGR==2 and oneBufferScheduling case, count local write before max(dataAtIterA,dataAtIterB)
                # NOTE: This logic assumes that local write is scheduled after local read.
                for up in range(max(dataAtIterA,dataAtIterB)):
                  skipPreIterLW += self.perIterLocalWriteCanSkip[up]
              localWrites += skipPreIterLW
        lgkmcnt += localWrites
      else:
        for item in list(iterCode.items()):
          localReads  = item.countType(Code.LocalReadInst)
          localWrites = item.countType(Code.LocalWriteInst)
          if self.numVgprBuffer:
            # SQ: If PrefetchLocalRead = 1 and DepthU == LocalSplitU, then there is no double
            #  buffering and we must wait for all localReads but not localWrites.
            #  In that case, LoopIters == 1:
            if kernel["LoopIters"] > 1:
              # here the reads are prefetches so can skip them in the waitcnt
              lgkmcnt += localReads
            # and the writes are targetting another section of LDS and are
            # synchronized through a different waitcnt than this one
            # (which is always just before the macs)
            lgkmcnt += localWrites
          else:
            # if UnrollLoopEfficiencyEnable == True  use waitCode passed lgkmCnt
            # else:
            # we need to wait for all preceding reads before the macs
            # so only opportunity for optimization is if the writes are at the end
            if globalParameters["UnrollLoopEfficiencyEnable"]:
              lgkmcnt = waitCode.lgkmcnt
            else:
              if localReads:
                lgkmcnt = 0 # reset to wait for all reads
              else:
                lgkmcnt = localWrites  # this only survives if writes are at the end

      waitCode.comment += " old=%u, new=%u newLW=%u newLR=%u" % (waitCode.lgkmcnt, lgkmcnt,localWrites,localReads)
      waitCode.lgkmcnt = lgkmcnt

    return iterCode

  ##############################################################################
  # returns list of modules or text
  # papIter indicates this is the setup for the "prefetchAcrossPersistent"
  # (aka pap) iteration
  ##############################################################################
  def setupNewTile(self, kernel, tensorParametersA, tensorParametersB, isPap, isOptNLL=False, forceNoTileCode=False, forceNoGRCode=False):
    kl = []
    kl_LW = [] # generate tile assignment code + local write code separately for init code optimization

    if self.enable["PreLoop"]:
      ####################################
      # Global Read Addresses
      ####################################
      kl.append(self.comment3("Begin setupNewTile, isPap=%s") % isPap)

      # work-group assignments
      kl.append(self.comment("global read addresses: work-group"))
      if not forceNoTileCode:
        kl.append(self.graWorkGroup(kernel, isPap))

      needShift = False
      if (kernel["EdgeType"] == "ShiftPtr") and \
         (not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialA"])) or \
         (not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialB"])):
        needShift = True

      # some case (PAP), we don't have to append the code for duplicated calculation
      # only those calculation related to WorkGroupID need to be generated. otherwise it's just redundant
      # default dontAppendCode = False, means need to append code
      self.dontAppendCode = False

      # 1. during isPap, this is actually no needed, so we can skip this.
      #    but since there are some vgpr value is used in the later lwaFirstOffset (when not OptNLL, such as "lwoT")
      #    so we still do this part when "isPap & not OptNLL"
      # 2. if tile edge, then we still need to add all these codes even isPap

      self.dontAppendCode = isPap and kernel["PrefetchAcrossPersistentMode"] == 1 and ((not needShift) or self.useGlobalReadTileVgpr)
      self.dontAppendCode = self.dontAppendCode or forceNoTileCode
      # tile assignments
      kl_LW.append(self.comment("global read addresses: tile offset assignment a"))
      kl_LW.append(self.graTileAssignment(kernel, tensorParametersA))
      kl_LW.append(self.comment("global read addresses: tile offset assignment b"))
      kl_LW.append(self.graTileAssignment(kernel, tensorParametersB))
      # init code optimization
      # not init code opt case, add tile assignments code here
      # init code opt case, not insert tile assignments code here and return tile assignments code separately for replacement
      if not self.isInitCodeOptLW:
        kl += kl_LW
        kl_LW = []

      self.dontAppendCode = isPap and (not needShift)
      self.dontAppendCode = self.dontAppendCode or forceNoTileCode
      # unroll assignments
      kl.append(self.comment("global read addresses: unroll assignment a"))
      kl.append(self.graUnrollAssignment(kernel, tensorParametersA))
      kl.append(self.comment("global read addresses: unroll assignment b"))
      kl.append(self.graUnrollAssignment(kernel, tensorParametersB))
      self.dontAppendCode = False
      self.dontAppendCode = self.dontAppendCode or forceNoTileCode

      # other free indices
      if kernel["ProblemType"]["NumIndicesC"] > 2:
        kl.append(self.comment("global read addresses: other free assignments"))
        kl.append(self.graOtherFreeAssignments(kernel))

      # other summation indices
      if self.otherSummations:
        kl.append(self.comment("global read addresses: other summation assignments"))
        kl.append(self.graOtherSummationAssignments(kernel))

      self.dontAppendCode = isPap and ((not needShift) or self.useGlobalReadTileVgpr)
      self.dontAppendCode = self.dontAppendCode or forceNoTileCode
      # tile offsets
      kl.append(self.comment("global read addresses: tile offsets a"))
      kl.append(self.graTileOffsets(kernel, tensorParametersA))
      kl.append(self.comment("global read addresses: tile offsets b"))
      kl.append(self.graTileOffsets(kernel, tensorParametersB))

      # unroll offsets
      kl.append(self.comment("global read addresses: unroll offsets a"))
      kl.append(self.graUnrollOffsets(kernel, tensorParametersA))
      kl.append(self.comment("global read addresses: unroll offsets b"))
      kl.append(self.graUnrollOffsets(kernel, tensorParametersB))
      self.dontAppendCode = False
      self.dontAppendCode = self.dontAppendCode or forceNoTileCode

      # tile edges
      if kernel["EdgeType"] == "ShiftPtr":
        # Shift here has two purposes:
        #  1. Ensure the loads are in-bounds to prevent fault.
        #     BufferLoad uses the buffer limit hardware and does not require bounds checking for this case
        #  2. Shift-left a wide vector load to ensure it is completely in-bounds.
        #     If this occurs we need to 'unshift' the C values (see shiftVectorComponents)
        #     BufferLoad does support this shifting, but if GuaranteeNoPartial=1 then
        #     it can be guaranteed that no shifting is required.
        if not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialA"]) and not forceNoTileCode:
          kl.append(self.comment("global read addresses: shift a"))
          kl.append(self.graShift(kernel, tensorParametersA))
        if not (kernel["BufferLoad"] and  kernel["GuaranteeNoPartialB"]) and not forceNoTileCode:
          kl.append(self.comment("global read addresses: shift b"))
          kl.append(self.graShift(kernel, tensorParametersB))
      elif kernel["EdgeType"] == "Branch":
        kl.append(self.comment("global read addresses: branch a"))
        kl.append(self.graBranch(kernel, tensorParametersA))
        kl.append(self.comment("global read addresses: branch b"))
        kl.append(self.graBranch(kernel, tensorParametersB))

      self.dontAppendCode = isPap and (not needShift)
      self.dontAppendCode = self.dontAppendCode or forceNoTileCode
      # final offsets
      kl.append(self.comment("global read addresses: final offsets a"))
      kl.append(self.graFinalOffsets(kernel, tensorParametersA))
      kl.append(self.comment("global read addresses: final offsets b"))
      kl.append(self.graFinalOffsets(kernel, tensorParametersB))
      self.dontAppendCode = False
      self.dontAppendCode = self.dontAppendCode or forceNoTileCode

      # addresses
      if not forceNoTileCode:
        kl.append(self.comment("global read addresses: addresses a"))
        kl.append(self.graAddresses(kernel, tensorParametersA, isPap))
        kl.append(self.comment("global read addresses: addresses b"))
        kl.append(self.graAddresses(kernel, tensorParametersB, isPap))

      self.dontAppendCode = isPap
      self.dontAppendCode = self.dontAppendCode or forceNoTileCode
      # increments
      kl.append(self.comment("global read addresses: increments a"))
      for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
        kl.append(self.graIncrements(kernel, i, tensorParametersA))
      kl.append(self.comment("global read addresses: increments b"))
      for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
        kl.append(self.graIncrements(kernel, i, tensorParametersB))
      self.dontAppendCode = False
      self.dontAppendCode = self.dontAppendCode or forceNoTileCode

      ####################################
      # Local Write Addresses
      ####################################
      kl_LW.append(self.comment3("Local Write Addresses"))

      # tile assignments
      kl_LW.append(self.lwaTileAssignment(kernel, tensorParametersA))
      kl_LW.append(self.lwaTileAssignment(kernel, tensorParametersB))

      # unroll assignments
      kl_LW.append(self.lwaUnrollAssignment(kernel, tensorParametersA))
      kl_LW.append(self.lwaUnrollAssignment(kernel, tensorParametersB))

      # if PAP, no need to reset LWA, but if not OptNLL, we still do this (due to TailLoop)

      self.dontAppendCode = isPap and kernel["PrefetchAcrossPersistentMode"] == 1
      self.dontAppendCode = self.dontAppendCode or forceNoTileCode
      # first offsets
      kl_LW.append(self.comment("local write addresses: first offset a"))
      kl_LW.append(self.lwaFirstOffset(kernel, tensorParametersA))
      kl_LW.append(self.comment("local write addresses: first offset b"))
      kl_LW.append(self.lwaFirstOffset(kernel, tensorParametersB))
      self.dontAppendCode = False
      self.dontAppendCode = self.dontAppendCode or forceNoTileCode

      # final offsets
      kl_LW.append(self.lwaFinalOffsets(kernel, tensorParametersA))
      kl_LW.append(self.lwaFinalOffsets(kernel, tensorParametersB))

      # declare addresses
      kl_LW.append(self.lwaDeclareAddresses(kernel, tensorParametersA))
      kl_LW.append(self.lwaDeclareAddresses(kernel, tensorParametersB))

      # init pointers
      kl_LW.append(self.localWriteInitPointers(kernel, tensorParametersA))
      kl_LW.append(self.localWriteInitPointers(kernel, tensorParametersB))

      # init code optimization
      if self.isInitCodeOptLW:
        # init code optimization case, release lwaVgpr here after all lwa code is generated
        # (to avoid lwa vgpr overwritten by remaining lwa code)
        self.lwaReleaseTileVgpr(kernel, tensorParametersA)
        self.lwaReleaseTileVgpr(kernel, tensorParametersB)
      else:
        # not init code opt case, add local write code here
        # init code opt case, not insert local write code here and return local write code separately for replacement
        kl += kl_LW
        kl_LW = []

    ###########################################################################
    # summations loops: open
    ###########################################################################

    # declare loop num iter
    if not forceNoTileCode:
      kl.append(self.comment1("declare loop num iterations"))
      kl.append(self.declareLoopNumIter(kernel))

    # perform initC in the shadow of the prefetch
    # Prefetch occurs at start of unroll loop
    # If we have multiple summation indices (otherSummationLoops>0),
    # we can't init in shadow of this prefetch
    # since that would initC inside the other summation loops

    if self.doShadowInit != 2:
      kl.append(self.initC(kernel))

    # open non-unrolled summation loops
    if not forceNoTileCode:
      for i in range(kernel["ProblemType"]["NumIndicesSummation"]-1):
        kl.append(self.comment("summation loop %u"%i))
        kl.append(self.calculateLoopNumIter(kernel, i, isPap))
        if self.actualSummationLoops>1:
          kl.append(self.openLoop(kernel, i))
      kl.append(self.calculateLoopNumIter(kernel, self.unrollIdx, isPap))

    if not forceNoTileCode:
      if self.staggerU:
        kl.append(self.declareStaggerParms(kernel))
        kl.append(self.calculateStagger(kernel, tensorParametersA))
        kl.append(self.calculateStagger(kernel, tensorParametersB))

    # isPap don't init the read pointers - we want to continue to use the double-buffer
    # LRO and LWA as assigned
    if self.enable["PreLoop"] and not isPap:
      # init lds read pointers before each unrolled loop
      kl.append(self.comment1("local read addresses: init pointers a"))
      kl.append(self.localReadInitPointers(kernel, tensorParametersA))
      kl.append(self.comment1("local read addresses: init pointers b"))
      kl.append(self.localReadInitPointers(kernel, tensorParametersB))

    if isPap and not isOptNLL:
      # init lds read pointers before each unrolled loop
      kl.append(self.comment1("local read addresses: reset offset a"))
      kl.append(self.localReadResetOffsets(kernel, tensorParametersA))
      kl.append(self.comment1("local read addresses: reset offset b"))
      kl.append(self.localReadResetOffsets(kernel, tensorParametersB))

    ####################################
    # prefetch: unrolled loop prefix
    ####################################
    if kernel["PrefetchGlobalRead"]:
      pfi = 1
      kl.append(self.comment("prefetch: global -> local"))
      kl.append(self.openSumAtLeastUnroll(kernel, prefetch=True, isOptNLL=isOptNLL, isPap=isPap))
      if isPap and isOptNLL:
        # forceNoGRCode case, reset and not generate global read A/B code
        if self.enable["GlobalRead"]  and (not forceNoGRCode):
          # if DirectToVgpr is enabled and swapGlobalRead is true, swap the order of global read (B->A)
          tensorParameters1st = tensorParametersA
          tensorParameters2nd = tensorParametersB
          if self.isSwapGlobalReadOrderForDtvOrDtl(kernel):
            tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
          self.dtlsM0UpdateACode = self.directToLdsM0Update(kernel, 0, tensorParameters1st, usePlaceHolder=isPap)
          self.globalReadACode = self.globalReadDo(kernel, 0, tensorParameters1st, 0)
          self.dtlsM0UpdateBCode = self.directToLdsM0Update(kernel, 0, tensorParameters2nd, usePlaceHolder=isPap)
          self.globalReadBCode = self.globalReadDo(kernel, 0, tensorParameters2nd, 0)
        else:
          self.dtlsM0UpdateACode = Code.StructuredModule()
          self.globalReadACode = Code.StructuredModule() # empty
          self.dtlsM0UpdateBCode = Code.StructuredModule()
          self.globalReadBCode = Code.StructuredModule() # empty

        if self.enable["GlobalReadInc"]:
          self.globalReadIncrements = self.globalReadIncrementAB(kernel, self.unrollIdx, pfi)
        else:
          self.globalReadIncrements = Code.Module()
          self.globalReadIncrements.addCode(Code.Module("globalReadIncrementA"))
          self.globalReadIncrements.addCode(Code.Module("globalReadIncrementB"))

      else:
        if self.enable["GlobalRead"]:
          # if DirectToVgpr is enabled and swapGlobalRead is true, swap the order of global read (B->A)
          tensorParameters1st = tensorParametersA
          tensorParameters2nd = tensorParametersB
          if self.isSwapGlobalReadOrderForDtvOrDtl(kernel):
            tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
          tmpStr = str(self.directToLdsM0Update(kernel, 0, tensorParameters1st, usePlaceHolder=isPap))
          tmpStr = tmpStr.replace("__placeholder__", str(0))
          kl.append(tmpStr)
          kl.append(str(self.globalReadDo(kernel, 0, tensorParameters1st, 0)))
          tmpStr = str(self.directToLdsM0Update(kernel, 0, tensorParameters2nd, usePlaceHolder=isPap))
          tmpStr = tmpStr.replace("__placeholder__", str(0))
          kl.append(tmpStr)
          kl.append(str(self.globalReadDo(kernel, 0, tensorParameters2nd, 0)))
        if self.enable["GlobalReadInc"]:
          kl.append(self.globalReadIncrementAB(kernel, self.unrollIdx, pfi))

    kl.append(self.comment3("End setupNewTile, isPap=%s") % isPap)

    return kl, kl_LW


  ##############################################################################
  # get conditions to skip local read write wait
  ##############################################################################
  def getConditionToSkipLocalReadWriteWait( self, kernel , isPap, u, lastU):
    # not generate wait code here if u == 0 u != lastU and DirectToVgpr + DirectToLds is enabled
    # (to remove redundant wait. isPap case only)
    # exception is PGR=2. wait is necessary for u = 0 in PGR=2 case
    cond1 = not (isPap and u == 0 and u != lastU and kernel["PrefetchLocalRead"] != 0 and \
       (kernel["DirectToVgprA"] and kernel["DirectToLdsB"] or kernel["DirectToVgprB"] and kernel["DirectToLdsA"])) \
      or kernel["PrefetchGlobalRead"]==2
    # no need local read wait if LocalReadVectorWidth>1 and u%numReadsIterCoalescedB is not 0
    # In that case, Prefetch local read covers both u = 0 and 1 (limit to MFMA+double+DirectToVgpr only)
    # (The other side of numReadsIterCoalesced must be 0 to skip local read wait)
    condSkip = (u%self.numReadsIterCoalescedB != 0) and kernel["EnableMatrixInstruction"] and \
              ((kernel["DirectToVgprA"] and (not kernel["ProblemType"]["TLUB"])) or \
               (kernel["DirectToVgprB"] and (not kernel["ProblemType"]["TLUA"])))
    # another skip condition
    # skip wait for SIA=3 and 1LDSBuffer and PLR > LoopIters and u > localWriteStartIter
    # in this case, all local read is executed before 1LDSBuffer sync and no need to wait for local read
    if (kernel["ScheduleIterAlg"] == 3 and kernel["1LDSBuffer"] and kernel["PrefetchLocalRead"] > kernel["LoopIters"]):
      localWriteStartIter = self.lwStartMfmaIndex//self.numMfmaPerIter
      if u > localWriteStartIter:
        condSkip = True
    # no local write wait is necessary in DirectToVgprA + DirectToVgprB case
    cond2 = not (kernel["DirectToVgprA"] and kernel["DirectToVgprB"])
    return cond1 and (not condSkip) and cond2

  ##############################################################################
  # No Load Loop Body
  ##############################################################################
  def noLoadLoopBody( self, kernel, tensorParametersA, tensorParametersB, kl, pack, isOptNLL, isPap, isNGLL, NLLfirst, NLLlast, isDTVodd=False):
    expand = kernel["ExpandPointerSwap"]
    lastuIdx = False
    pflr     = self.numItersPLR
    localWriteEndIter = kernel["LoopIters"] - self.numItersPLR - 1
    # noTailLoop optimization
    # if this is the last NoLoadLoop(NLLlast) and self.tailLoopInNLL case, set tail=True for mfmaIter
    needTailCode = NLLlast and self.tailLoopInNLL

    # vregSetIdx for DTV
    # NGLL case, use first set. NLL case, use second set
    vregSetIdxMFMA = 0 if isNGLL else 1
    # flip vregSetIdx if isDTVodd is True
    if isDTVodd:
      vregSetIdxMFMA = 1 - vregSetIdxMFMA

    for uIdx in range(0, kernel["LoopIters"]*kernel["DepthULdsDivisor"]):
      u = uIdx % kernel["LoopIters"]    #   u: index in compute loop (in contrast to the notion of global read loop)
      uDu = uIdx // kernel["LoopIters"] # uDu: index of compute loop
      isLastLoop = (uDu == kernel["DepthULdsDivisor"] -1 ) and not isNGLL
      if u == 0:
        if uDu > 0:
          if self.enable["GlobalRead"]:
            assert len(self.globalReadACode.items()) > 0 and len(self.globalReadBCode.items()) > 0 # already issued in first uDu
            self.globalReadACode = Code.StructuredModule() # empty
            self.globalReadBCode = Code.StructuredModule() # empty
          if self.enable["GlobalReadInc"]:
            self.globalReadIncrements = Code.Module() # empty
            self.globalReadIncrements.addCode(Code.Module("globalReadIncrementA"))
            self.globalReadIncrements.addCode(Code.Module("globalReadIncrementB"))
        if not isLastLoop:
          self.localWriteACode = self.localWriteDo(kernel, tensorParametersA, (uDu+1)%kernel["DepthULdsDivisor"])  # local write in loopcnt N targets data for loopcnt N+1
          self.localWriteBCode = self.localWriteDo(kernel, tensorParametersB, (uDu+1)%kernel["DepthULdsDivisor"])
        else:
          self.localWriteACode = Code.Module()
          self.localWriteBCode = Code.Module()

        # TODO schedule waitcnt/barrier in makeSubIterSchedule()
        if kernel["PrefetchGlobalRead"] and kernel["LoopIters"] in [1, 2] and uDu > 0:
          if self.enable["Wait"]:
            kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "wait for local write"))
          if self.enable["Sync"]:
            kl.append(self.syncThreads(kernel, "sync for local read after write"))

        if not isNGLL or isPap:
          # PAP would have GlobalRead and GlobalInc, but no localWrite
          # Get the perIterGlobalReadCode code for PAP (if PAP=On), else would be empty
          # NGLL (PGR=2) and isPap case, we do not need globalInc code. Set skip flag in that case
          skipGlobalReadInc = isNGLL and isPap
          self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, uDu, skipGlobalReadInc=skipGlobalReadInc, lastLoop=NLLlast)
          kl.append(str(self.unrollLoopHeaderCode))

      # which loop iteration to reset the LRO,
      # note if PLR=0, isResetLroIter is False for all u
      isResetLroIter = (u == localWriteEndIter)
      isSwapAndResetLwoIter = isResetLroIter
      isSwapLroIter = isResetLroIter
      if kernel["ScheduleIterAlg"] == 3:
          isSwapAndResetLwoIter = (u == self.lwEndMfmaIndex//(self.numMfmaPerIter))

      extraComment = ""
      if isLastLoop:
        extraComment += " (last unrolled loop)"
      else:
        if kernel.enabledSplitLDS:
            extraComment += f" (uDu={uDu}) "
        if isResetLroIter:
            extraComment += " (reset local read pointers iteration) "
        if isSwapAndResetLwoIter:
            extraComment += " (swap and reset local write pointers iteration) "
        if isSwapLroIter:
            extraComment += " (swap local read pointers iteration) "

      kl.append(self.comment("iter %u%s"%(u,extraComment)))
      plrIdx = ((u+pflr) % (self.numVgprBuffer+1)) % kernel["LoopIters"]
      plrIdxDTV = (u+pflr) % kernel["LoopIters"]
      localReads = Code.Module()

      pointerLWCode = Code.Module()
      pointerLRCode = Code.Module()
      waitCode = Code.Module()  # may be overwritten (not added to) below
      macIterCode = Code.Module()
      waitLWCode = Code.Module()
      syncCode = Code.Module()

      # vregSetIdx for DTV
      vregSetIdxLR = vregSetIdxMFMA
      if kernel["LoopIters"] > 1 and u == kernel["LoopIters"] - 1 and \
         not ( isNGLL and (not isDTVodd) and self.generateDTVodd):
        # use next vregSet for the last loopIter
        # exception for isNGLL and not isDTVodd and odd code enabled case
        # in this case, we need to flip twice (for NGLL to NLL, even to odd)
        # then, we do not need to flip vregSetIdxLR
        vregSetIdxLR = 1 - vregSetIdxLR

      if self.enable["LocalRead"]:
        hasLiveLdsData = kernel["PrefetchGlobalRead"] or (uDu < kernel["DepthULdsDivisor"]-1)
        hasLiveLdsData = hasLiveLdsData and not isLastLoop
        # for DirectToVgpr + DTVodd
        # need to call localReadDo to allocate tmpVgpr for the next DTVeven case (no actual asm code generated for DTV)
        needExtraLocalReadDo = (NLLlast and isDTVodd and u > localWriteEndIter)
        # DTV pack case, generate local read code for DTVodd case
        hasLiveLdsData = hasLiveLdsData or needExtraLocalReadDo
        # reads for current loop are done in previous iteration because of wider local read
        doReadA = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadA - self.numItersPLR)
        doReadB = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadB - self.numItersPLR)
        # reads for next loop
        doReadA = doReadA or (hasLiveLdsData and u > localWriteEndIter)
        doReadB = doReadB or (hasLiveLdsData and u > localWriteEndIter)
        for iui in range(0,kernel["InnerUnroll"]):
          doReadA = doReadA and iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]
          doReadB = doReadB and iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]
          if doReadA:
            # needExtraLocalReadDo only case, no need to generate actual code
            # just need to call localReadDo to allocate tmpVgpr
            if not needExtraLocalReadDo:
              localReads.addText(self.comment("local read a"))
            bufferIdx = plrIdx*self.numIterPerCoalescedReadA
            if self.packDTVA:
              # DTV + pack case, offset bufferIdx for local read packing instructions
              bufferIdx = plrIdxDTV*self.numIterPerCoalescedReadA + vregSetIdxLR * kernel["LoopIters"]
            localReadCodeA, packCodeA = self.localReadDo(kernel, bufferIdx, iui*self.numReadsIterCoalescedA, 0, tensorParametersA)
            if not needExtraLocalReadDo:
              localReads.addCode(localReadCodeA)
              pack[plrIdx*self.numIterPerCoalescedReadA].addCode(packCodeA)
            if needExtraLocalReadDo and (self.packDTVA or self.packDTVB):
              # packDTV + needExtraLocalReadDo, need to update pack code to flip vregSetIdx
              pack[plrIdx*self.numIterPerCoalescedReadA].addCode(packCodeA)
          if doReadB:
            # needExtraLocalReadDo only case, no need to generate actual code
            # just need to call localReadDo to allocate tmpVgpr
            if not needExtraLocalReadDo:
              localReads.addText(self.comment("local read b"))
            bufferIdx = plrIdx*self.numIterPerCoalescedReadB
            if self.packDTVB:
              # DTV + pack case, offset bufferIdx for local read packing instructions
              bufferIdx = plrIdxDTV*self.numIterPerCoalescedReadB + vregSetIdxLR * kernel["LoopIters"]
            localReadCodeB, packCodeB = self.localReadDo(kernel, bufferIdx, iui*self.numReadsIterCoalescedB, 0, tensorParametersB)
            if not needExtraLocalReadDo:
              localReads.addCode(localReadCodeB)
              pack[plrIdx*self.numIterPerCoalescedReadB].addCode(packCodeB)
            if needExtraLocalReadDo and (self.packDTVA or self.packDTVB):
              # packDTV + needExtraLocalReadDo, need to update pack code to flip vregSetIdx
              pack[plrIdx*self.numIterPerCoalescedReadB].addCode(packCodeB)
          if (not isResetLroIter or iui != kernel["InnerUnroll"]-1):
            if doReadA:
              localReads.addText(self.comment("local read increment a"))
              localReads.addText(self.localReadInc(kernel, iui, tensorParametersA))
            if doReadB:
              localReads.addText(self.comment("local read increment b"))
              localReads.addText(self.localReadInc(kernel, iui, tensorParametersB))

      if not isLastLoop:
        if kernel["PrefetchGlobalRead"]:
          # put barrier at localWriteEndIter+1
          if u == localWriteEndIter+1 or (u == (localWriteEndIter+1)%kernel["LoopIters"] and kernel["ScheduleIterAlg"] == 2):
            if self.enable["Wait"]:
              # skip local write wait if DirectToVgpr + DirectToLds is enabled
              if not kernel["NoLdsWriteCode"]:
                waitLWCode.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "3wait for local write"))
              if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]):
                # DirectToVgpr + DirectToLds case, add waitcnt vmcnt before s_barrier
                # Except for PGR=2 and Load C (StoreCInUnroll) case. In that case, Load C is executed after necessary Load A and B.
                # Wait for Load C is already done here in PGR=2 case.
                needLoadC = kernel["StoreCInUnroll"] and (not kernel["AtomicAddC"]) and kernel["ProblemType"]["UseBeta"]
                if not (kernel["PrefetchGlobalRead"]==2 and needLoadC):
                  retStr = self.getWaitcntCodeForDirectToVgpr(kernel, localWriteEndIter, u, firstIter=False, beforeBarrier=True)
                  waitLWCode.addCode(retStr)
            if self.enable["Sync"]:
              if kernel["PrefetchGlobalRead"]==2 and (kernel["DirectToLdsA"] and kernel["DirectToLdsB"]):
                # PGR=2 and DTLA+B case, wait for global read needs to be added (wait is not generated with local write)
                syncCode.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "wait for global read with lds"))
              syncCode.addCode(self.syncThreads(kernel))

          if isSwapAndResetLwoIter: # ResetLroIter
            if self.enable["LocalWrite"]:
              # local write for next iter, used to have local writes here
              pointerLWCode.addText(self.comment("local write swap offsets a"))
              pointerLWCode.addText(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
              pointerLWCode.addText(self.comment("local write swap offsets b"))
              pointerLWCode.addText(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))
              pointerLWCode.addText(self.localWriteInitPointers(kernel, tensorParametersA))
              pointerLWCode.addText(self.localWriteInitPointers(kernel, tensorParametersB))

          if isSwapLroIter: # ResetLroIter
            if self.enable["LocalRead"]:
              # Swap, reset, or increment the LRO:
              # force internalPointerSwap = False in NGLL case
              internalPointerSwap = expand and not isNGLL
              pointerLRCode.addText(self.comment("local read swap offsets a"))
              pointerLRCode.addText(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersA))
              pointerLRCode.addText(self.comment("local read swap offsets b"))
              pointerLRCode.addText(self.localReadSwapOffsets(kernel, internalPointerSwap, tensorParametersB))

        if isResetLroIter: # ResetLroIter
          if self.enable["LocalRead"]:
            pointerLRCode.addText(self.comment("local read init pointers a"))
            pointerLRCode.addText(self.localReadInitPointers(kernel, tensorParametersA))
            pointerLRCode.addText(self.comment("local read init pointers b"))
            pointerLRCode.addText(self.localReadInitPointers(kernel, tensorParametersB))

      # we initiate lgkmcnt to 0, then assigning it correct value in makeSubIterSchedule()
      if self.enable["Wait"]:
        if self.getConditionToSkipLocalReadWriteWait(kernel, isPap, u, kernel["LoopIters"] - 1):
          waitCode = self.wait(kernel, tensorParametersA, tensorParametersB, \
              -1, 0, 0, \
              "wait for prior local read local write")
        # DirectToVgpr case, wait for global read as well as local read/write
        if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
          # not generate wait here
          #  1) local write code in previous u (u-1) has waitcnt vmcnt
          prevVmcnt = False
          prevLocalWrite = ""
          if (u > 0):
            prevLocalWrite = ' '.join([str(x) for x in self.perIterLocalWriteCode[u-1].flatitems()])
            prevVmcnt = "vmcnt" in prevLocalWrite
          if not prevVmcnt:
            retStr = self.getWaitcntCodeForDirectToVgpr(kernel, localWriteEndIter, u, False, isPap or isNGLL, NLLlast=NLLlast)
            kl.append(retStr)

      # generate StoreCInUnroll post loop if it is enabled (only PAP case (but not NGLL))
      if u == localWriteEndIter+1:
        if kernel["StoreCInUnrollPostLoop"] and isPap and not isNGLL:
          kl.append(self.generateStoreInUnrollPostLoop(kernel, isOptNLL, isDTVodd))

      luIdx = (u) % (self.numVgprBuffer+1) # local to use for MACs
      if self.enable["MAC"]:
        if kernel["EnableMatrixInstruction"]:
          if ((uIdx+1) == kernel["LoopIters"]*kernel["DepthULdsDivisor"]) and \
              (kernel["StoreCInUnroll"]):
            lastuIdx = (isOptNLL or self.enableSingleNLLOpt) and not isNGLL # do not apply lastuIdx for not isOptNLL case
          macIterCode.addCode(self.mfmaIter(kernel, u, kernel["InnerUnroll"], vregSetIdxMFMA,lastuIdx,tail=needTailCode))
        else:
          macIterCode.addCode(self.macIter(kernel, luIdx, kernel["InnerUnroll"], True ))

      subIterCode = self.makeSubIterSchedule(kernel, localReads, \
                      u, pointerLWCode, pointerLRCode, waitCode, macIterCode, waitLWCode, syncCode, pack[luIdx], isDTVodd, NLLlast)
      kl.append(subIterCode)
      # vgpr.checkin for all the checked-out vgpr in LocalRead
      for item in list(pack[luIdx].items()):
        if item.tempVgpr != None:
          self.vgprPool.checkIn(item.tempVgpr)
          item.tempVgpr = None
      pack[luIdx] = Code.Module()

      # tail loop in NoLoadLoop case, generate close loop code for TailLoop here (except for last loop iteration)
      if needTailCode:
        finalLoop = (u == kernel["LoopIters"] - 1)
        skipJump = finalLoop or self.noEarlyExitForTailLoopInNLL
        kl.append(self.closeLoop(kernel, -1, finalLoop, 1, oddLabel=isDTVodd, skipCondJumpCounter=u, isOptNLL=isOptNLL, skipJump=skipJump))

  ##############################################################################
  # noLoadLoop
  # Create the no load loop (NLL)
  #
  # isOptNLL : the NLL is to be optimized for the alpha=1 and non-edge case
  ##############################################################################
  def noLoadLoop( self, kernel, tensorParametersA, tensorParametersB, isOptNLL, isPap, isNGLL, pack ):
    kl = []
    if isNGLL:
      LoopNameComment = "NoGlobalLoadLoop"
    else:
      LoopNameComment = "NoLoadLoop"
    if isOptNLL:
      PAPcomment = "Opt. %s %s PAP - Begin " % (LoopNameComment, "With" if isPap else "Without")
    else:
      PAPcomment = "Ord. %s - Begin " % (LoopNameComment)
    kl.append(self.comment3("%s")%PAPcomment)
    NLLfirst = True
    NLLlast = True
    if kernel["PrefetchGlobalRead"] == 2:
      # PGR=2 case NoLoadLoop(NLL) is generated twice
      # we need to distinguish them to generate proper code at each NLL
      if isNGLL:
        NLLlast = False
      else:
        # PGR=2 and not isNGLL means second NoLoadLoop for PGR2.
        # Need to avoid generating duplicated code which is already generated in NGLL(first NoLoadLoop for PGR=2)
        NLLfirst = False
    if isNGLL:
      self.perIterLocalWriteCode = self.perIterLocalWriteCodeNGLL
      self.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
    #else:
    if not isNGLL or isPap:
      self.dtlsM0UpdateACode = Code.StructuredModule()
      self.globalReadACode = Code.StructuredModule() # empty
      self.dtlsM0UpdateBCode = Code.StructuredModule()
      self.globalReadBCode = Code.StructuredModule() # empty
      self.globalReadIncrements = Code.Module()
      self.globalReadIncrements.addCode(Code.Module("globalReadIncrementA"))
      self.globalReadIncrements.addCode(Code.Module("globalReadIncrementB"))
      self.localWriteACode = Code.Module()
      self.localWriteBCode = Code.Module()

    # the scheduled GlobalRead,Inc code of PAP is inside openSumAtLeastUnroll (if PAP=on)
    isPapTmp = isPap
    if kernel["PrefetchGlobalRead"]==2:
      # PGR=2 case, set isPap only if isNGLL is True. This is to generate NewTile code at NGLL in PAP + PGR=2 case
      isPapTmp = isPap and not isNGLL
    kStrOpenSum = self.openSumAtLeastUnroll(kernel, prefetch=False, isOptNLL=isOptNLL, isPap=isPapTmp)

    #if self.prefetchAcrossPersistent and kernel["PrefetchAcrossPersistentMode"] == 1 and isPap:
    if self.prefetchAcrossPersistent and isPap:
    #if self.prefetchAcrossPersistent and isPap \
    #   and (kernel["PrefetchAcrossPersistentMode"] == 0 or isOptNLL):
      kStr = ""
      #kStr += str(self.openPrefetchAcrossPersistent(kernel, isOptNLL=False, useBufferOOB=True))
      # For PAPMode 1, using isOptNLL true to generate prefetch code

      if kernel["PrefetchAcrossPersistentMode"] == 0:
        # generate openSumAtLeastUnroll code here
        kStr += kStrOpenSum
        kStrOpenSum = "" # empty OpenSum str to avoid inserting it again

      # isPap and kernel["PrefetchAcrossPersistentMode"] == 1 and isOptNLL==False,
      # no need to append NewTile code because it is already generated in OptNLL code
      # also, NGLL second NoLoadLoop case, we do not append code for NewTile
      forceNoTileCode = False
      if (isOptNLL==False or (not NLLfirst)):
        forceNoTileCode = True
      # PGR=2 and last loop case, we do not need GlobalRead code
      forceNoGRCode = False
      if kernel["PrefetchGlobalRead"] == 2 and NLLlast:
        forceNoGRCode = True

      newTileCodes, _ = self.setupNewTile(kernel, self.tPA, self.tPB, isPap=True, isOptNLL=True, \
                                       forceNoTileCode=forceNoTileCode, forceNoGRCode = forceNoGRCode)
      codes = '\n'.join([str(x) for x in newTileCodes])
      kStr += codes
      # openPrefetchAcrossPersistent should be after newTileCodes to set correct values to ShadowLimit
      # also, NGLL second NoLoadLoop case, we do not append code for Open/Close PAP
      if isOptNLL:
        if NLLfirst:
          kStr += str(self.openPrefetchAcrossPersistent(kernel, isOptNLL=False, useBufferOOB=True))
          kStr += str(self.closePrefetchAcrossPersistent(kernel, isOptNLL=False, useBufferOOB=True))
      kl.append(kStr)

    # skip generating OpenSum code here for SingleNLLOpt
    if not (isOptNLL and self.enableSingleNLLOpt):
      kl.append(kStrOpenSum)
      kStrOpenSum = "" # empty OpenSum str to avoid inserting it again

    # re-calculate loop counter for tailLoopInNLL (without noEarlyExitForTailLoopInNLL)
    if NLLlast and self.tailLoopInNLL and not self.noEarlyExitForTailLoopInNLL:
      kl.append(self.calculateLoopNumIter(kernel, -1, False))

    if not self.numItersPLR:
      if self.enable["Wait"]:
        if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "10wait for global read"))
        # TODO: need to check if we correctly checked-in the temp VGPR used for Int8 LocalWrite (uDu, PGR=2)
        kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "4wait for local write"))
      if self.enable["Sync"]:
        kl.append(self.syncThreads(kernel))

    # if DirectToVgpr and  ASEM/GSU is not multiple of DepthU*2, generate noLoadLoopBody twice for odd and even exit separately
    if self.generateDTVodd:
      # generate additional No Load Loop Body code for odd case (to use the other Vreg set for DirectToVgpr)
      # 1. generate odd check
      name = ""
      if isNGLL:
        name += "NoGlobalLoadLoop"
      else:
        name += "NoLoadLoop"
      if isOptNLL:
        name += "Opt"
      else:
        name += "Ord"
      kl.append(self.openOddNoLoadLoopForDTV(kernel, isNGLL, name))
      # 2. generate  no Load Loop Body code for odd
      # backup
      self.saveLocalPointers(kernel)
      # copy pack
      if isNGLL or (self.packDTVA or self.packDTVB):
        # NGLL or pack DTV case, no deep copy for pack
        # pack code for local prefetch is generated in noLoadLoopBody and used for DTV even
        deepCopyPack = pack
      else:
        # deepCopy packCode for OptNLL noLoadLoop
        deepCopyPack = copy.deepcopy(pack)
      # keep StoreCInUnroll related code for the next noLoadLoop
      if kernel["StoreCInUnroll"]:
        self.backupStoreCInUnrollRelatedCode()
      self.noLoadLoopBody(kernel, tensorParametersA, tensorParametersB, kl, deepCopyPack, isOptNLL, isPap, isNGLL, NLLfirst, NLLlast, isDTVodd=True)
      # restore
      self.restoreLocalPointers(kernel)
      # restore StoreCInUnroll related code
      if kernel["StoreCInUnroll"]:
        self.restoreStoreCInUnrollRelatedCode()
      # 3. PAP enabled and isLast and odd code case, the last global load for DirectToVgpr is the seconde reg set.
      #    Need to copy to the first set for the next PK loop
      if isPap and NLLlast:
        kl.append(self.getWaitcntCodeForDirectToVgpr(kernel, 0, 0, False, oddLast=True))
        kl.append(self.generateOddEndVgprCopyForDTV(kernel))
      # 4. generate even start label
      kl.append(self.closeOddNoLoadLoopForDTV(kernel, isNGLL, name))
      # 5. generate  no Load Loop Body code for odd
      # need to re-initialize perIterLocalWriteCanSkip to avoid having incorrect lgkmcnt
      self.perIterLocalWriteCanSkip = [ 0 for i in range (kernel["LoopIters"]) ]
      self.noLoadLoopBody(kernel, tensorParametersA, tensorParametersB, kl, pack, isOptNLL, isPap, isNGLL, NLLfirst, NLLlast)
      # 6. generate even end label
      kl.append(self.generateEvenEndLabeNoLoadLoopForDTV(kernel, isNGLL, name))
    else:
      # generate no Load Loop Body code
      self.noLoadLoopBody(kernel, tensorParametersA, tensorParametersB, kl, pack, isOptNLL, isPap, isNGLL, NLLfirst, NLLlast)

    # tail loop in NLL and early exit case, need to wait for all prefetch local read (and global read for DirectToVgpr)
    if NLLlast and self.tailLoopInNLL and (not self.noEarlyExitForTailLoopInNLL):
      if self.enable["Wait"]:
        kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "13wait for remaining local read for tail loop in NLL"))
        if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "14wait for remaining DirectToVgpr global read for tail loop in NLL"))

    if NLLlast and isPap:
      # reset or swap local write offset
      # If DirectToLds is True, first LDS buffer is already used by lds global read and offset already points first one
      # Swap/Reset is not necessary
      # If DirectToLds is False, first LDS buffer is no used yet, need reset.
      if kernel["ExpandPointerSwap"]:
        if not kernel["DirectToLdsA"]:
          kl.append(self.comment("local write reset offsets a"))
          kl.append(self.localWriteResetOffsets(kernel,  False, tensorParametersA))
        if not kernel["DirectToLdsB"]:
          kl.append(self.comment("local write reset offsets b"))
          kl.append(self.localWriteResetOffsets(kernel,  False, tensorParametersB))
        kl.append(self.localReadResetOffsets(kernel,  tensorParametersA))
        kl.append(self.localReadResetOffsets(kernel,  tensorParametersB))

    # add OpenSum code here if it is not empty
    if kStrOpenSum != "":
      kl.append(kStrOpenSum)

    # Close code is necessary for both first and last (NGLL case(=NLLfirst) needs label)
    kl.append(self.closeSumAtLeastUnroll(kernel, prefetch=False, isOptNLL=isOptNLL, isPap=isPap, isNGLL=isNGLL))

    return kl

  ##############################################################################
  # Loop Body
  ##############################################################################
  def loopBody( self, kernel, tensorParametersA, tensorParametersB, kl, pack, lc, loopCopies, finalLoop, firstIter=False ):
    expand = kernel["ExpandPointerSwap"]
    # firstIter flag for waitcnt
    # if useInitAccVgprOpt is not used, waitcnt of the first iteration should be same as firstIter
    # (does not include storeC from the previous PK loop)
    firstIterForWait = firstIter or ((not self.useInitAccVgprOpt) and lc == 0)

    # generate storeC code for StoreCInUnroll (need to call for not StoreCInUnroll case as well)
    self.generateStoreCCodeInUnrollLoop(kernel, lc & 1)

    # not generate openLoop for firstIter
    if not firstIter:
      kl.append(self.comment3("Unrolled Loop %u/%u - Begin" % (lc+1, loopCopies)))
      kl.append(self.openLoopCopy(kernel, lc))
    if kernel["PrefetchGlobalRead"] and not self.numItersPLR and not kernel["ScheduleIterAlg"] == 2:
      if self.enable["Wait"]:
        if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "11wait for global read"))
        kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "1wait for local write"))
      if self.enable["Sync"]:
        kl.append(self.syncThreads(kernel, "4sync for global read"))

    kl.append(self.comment("Begin Each Unroll: Check VGPR.checkin for INT8 LW"))

    if self.enable["GlobalRead"]:
      # if DirectToVgpr is enabled and swapGlobalRead is true, swap the order of global read (B->A)
      tensorParameters1st = tensorParametersA
      tensorParameters2nd = tensorParametersB
      tc1 = 'A'
      tc2 = 'B'
      if self.isSwapGlobalReadOrderForDtvOrDtl(kernel):
        tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
        tc1, tc2 = tc2, tc1
      # unrolled loop: global read A, B
      # M0 update for directToLds
      vregSetIdxGR = 0
      if (kernel["DirectToVgpr%s"%tc1]):
        vregSetIdxGR = (kernel["PrefetchGlobalRead"] + lc ) % 2 # toggle vreg set for DirectToVgpr.
      self.dtlsM0UpdateACode = self.directToLdsM0Update(kernel, 1, tensorParameters1st, usePlaceHolder=True)
      self.globalReadACode  = self.globalReadDo(kernel, 1, tensorParameters1st, vregSetIdxGR)
      vregSetIdxGR = 0
      if (kernel["DirectToVgpr%s"%tc2]):
        vregSetIdxGR = (kernel["PrefetchGlobalRead"] + lc ) % 2 # toggle vreg set for DirectToVgpr.
      self.dtlsM0UpdateBCode = self.directToLdsM0Update(kernel, 1, tensorParameters2nd, usePlaceHolder=True)
      self.globalReadBCode = self.globalReadDo(kernel, 1, tensorParameters2nd, vregSetIdxGR)
    else:
      self.dtlsM0UpdateACode = Code.StructuredModule()
      self.globalReadACode = Code.StructuredModule() # empty
      self.dtlsM0UpdateBCode = Code.StructuredModule()
      self.globalReadBCode = Code.StructuredModule() # empty

    if self.enable["GlobalReadInc"]:
      # unrolled loop: increment global read addresses
      self.globalReadIncrements = self.globalReadIncrementAB(kernel, self.unrollIdx, 0)
    else:
      self.globalReadIncrements = Code.Module()
      self.globalReadIncrements.addCode(Code.Module("globalReadIncrementA"))
      self.globalReadIncrements.addCode(Code.Module("globalReadIncrementB"))

    if self.enable["LocalWrite"] and not kernel["NoLdsWriteCode"]:
      self.localWriteACode = self.localWriteDo(kernel, tensorParametersA)
      self.localWriteBCode = self.localWriteDo(kernel, tensorParametersB)
    else:
      self.localWriteACode = Code.Module()
      self.localWriteBCode = Code.Module()

    # localWriteEndIter is used to determine which iteration to put sync
    # if PGR=0, GR,LW,sync,LR will put at front of loop.
    localWriteEndIter = kernel["LoopIters"] - self.numItersPLR - 1

    # Schedule the global read, global read inc, and writes:
    unrollLoopHeaderCodeScheduled = False
    if not kernel["PrefetchGlobalRead"]:
      unrollLoopHeaderCodeScheduled = True
      self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, firstIter=firstIter)
      kl.append(str(self.unrollLoopHeaderCode))

    # if not prefetch global, localWrite before mac's
    if not kernel["PrefetchGlobalRead"]:
      # unrolled loop: local write A, B
      if self.enable["Wait"]:
        kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "5wait for global read"))
      if self.enable["Sync"]:
        kl.append(self.syncThreads(kernel, "PGR=0, prior iter done reading lds"))
      if self.enable["LocalWrite"] and not kernel["NoLdsWriteCode"]:
        kl.append(self.comment("local write a"))
        tempLWCodeModA = self.localWriteDo(kernel, tensorParametersA)
        kl.append(tempLWCodeModA)
        kl.append(self.comment("local write b"))
        tempLWCodeModB = self.localWriteDo(kernel, tensorParametersB)
        kl.append(tempLWCodeModB)
      if self.enable["Wait"]:
        kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "2prefetch wait for local write"))
      if self.enable["Sync"]:
        kl.append(self.syncThreads(kernel))
        # debug Local state
        """
        kl.append("    /* print Local state */" + self.endLine)
        kl.append("    for (unsigned int i = serial; i < LDS_NUM_ELEMENTS; i+=NUM_THREADS) {%s" % self.endLine)
        kl.append("      printf(\\\"localMemory[%%06u] = %%.0f\\\\n\\\", i, localMemory[i]);%s" )
            % self.endLine
        kl.append("    }" + self.endLine)
        """

    # unrolled loop: prefetch local
    if self.numItersPLR and not kernel["PrefetchGlobalRead"]:
      if self.enable["LocalRead"]:
        for plrIdx in range(0, self.numItersPLR):
          pack[plrIdx] = Code.Module()
          for iui in range(0,kernel["InnerUnroll"]):
            if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]:
              kl.append(self.comment("prefetch local a"))
              localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadA, iui*self.numReadsIterCoalescedA, 0, tensorParametersA)
              kl.append(localReadCodeA)
              pack[plrIdx].addCode(packCodeA)
            if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]:
              kl.append(self.comment("prefetch local b"))
              localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadB, iui*self.numReadsIterCoalescedB, 0, tensorParametersB)
              kl.append(localReadCodeB)
              pack[plrIdx].addCode(packCodeB)
            if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]:
              kl.append(self.comment1("local read increment a"))
              kl.append(self.localReadInc(kernel, iui, tensorParametersA))
            if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]:
              kl.append(self.comment1("local read increment b"))
              kl.append(self.localReadInc(kernel, iui, tensorParametersB))

    pflr     = self.numItersPLR  # how many pf already done above
    # vregSetIdx for DTV
    vregSetIdxMFMA = lc

    ############################################################################
    # unrolled loop: mac iterations
    ############################################################################

    # double/quadruple the number of compute loop for each DepthU's worth of data read
    for uIdx in range(0, kernel["LoopIters"]*kernel["DepthULdsDivisor"]):
      u = uIdx % kernel["LoopIters"]    #   u: index in compute loop (in contrast to the notion of global read loop)
      uDu = uIdx // kernel["LoopIters"] # uDu: index of compute loop
      if u==0: # if at start of subloop...
        # ...update local write code
        if self.enable["LocalWrite"] and not kernel["NoLdsWriteCode"]:
          self.localWriteACode = self.localWriteDo(kernel, tensorParametersA, (uDu+1)%kernel["DepthULdsDivisor"])  # local write in loopcnt N targets data for loopcnt N+1
          self.localWriteBCode = self.localWriteDo(kernel, tensorParametersB, (uDu+1)%kernel["DepthULdsDivisor"])
        else:
          self.localWriteACode = Code.Module()
          self.localWriteBCode = Code.Module()

        # TODO schedule waitcnt/barrier in makeSubIterSchedule()
        if kernel["PrefetchGlobalRead"] and kernel["LoopIters"] in [1, 2] and uDu > 0:
          if self.enable["Wait"]:
            kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 1, 0, -1, "wait for local write"))
          if self.enable["Sync"]:
            kl.append(self.syncThreads(kernel, "sync for local read after write"))

        if not unrollLoopHeaderCodeScheduled:
          self.makeSchedule(kernel, tensorParametersA, tensorParametersB, localWriteEndIter, uDu, firstIter=firstIter, lastLoop=False, lastLc=(lc==loopCopies-1))
          kl.append(str(self.unrollLoopHeaderCode))

      # for PGR=0 where generator can't schedule the instructions (yet),
      # we duplicate the local write codegen and append to string list directly
      if not kernel["PrefetchGlobalRead"]:
        doWrite = False
        if uDu<kernel["DepthULdsDivisor"]-1 and u==kernel["LoopIters"]-self.numItersPLR:
          doWrite = True
          writeForNextLoop = 1
        if uDu>0 and self.numItersPLR==0 and u==0:
          assert doWrite==False # should be exclusive with the previous condition
          doWrite = True
          writeForNextLoop = 0
        # unrolled loop: local write A, B
        if doWrite:
          if self.enable["Wait"]:
            kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "5wait for local read"))
          if self.enable["Sync"]:
            kl.append(self.syncThreads(kernel, "PGR=0, prior iter done reading lds"))
          if self.enable["LocalWrite"] and not kernel["NoLdsWriteCode"]:
            kl.append(self.comment("local write a"))
            tempLWCodeModA = self.localWriteDo(kernel, tensorParametersA, (uDu+writeForNextLoop)%kernel["DepthULdsDivisor"])
            kl.append(tempLWCodeModA)
            kl.append(self.comment("local write b"))
            tempLWCodeModB = self.localWriteDo(kernel, tensorParametersB, (uDu+writeForNextLoop)%kernel["DepthULdsDivisor"])
            kl.append(tempLWCodeModB)
          if self.enable["Wait"]:
            kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "2prefetch wait for local write"))
          if self.enable["Sync"]:
            kl.append(self.syncThreads(kernel))

      # which loop iteration to reset the LRO,
      # note if PLR=0, isResetLroIter is False for all u
      isResetLroIter = (u == localWriteEndIter)
      isSwapAndResetLwoIter = isResetLroIter
      isSwapLroIter = isResetLroIter
      if kernel["ScheduleIterAlg"] == 3:
        isSwapAndResetLwoIter = (u == self.lwEndMfmaIndex//(self.numMfmaPerIter))
      extraComment = ""
      if kernel.enabledSplitLDS:
        extraComment += f" (uDu={uDu}) "
      if isResetLroIter:
        extraComment += " (reset local read pointers iteration) "
      if isSwapAndResetLwoIter:
        extraComment += " (swap and reset local write pointers iteration) "
      if isSwapLroIter:
        extraComment += " (swap local read pointers iteration) "

      kl.append(self.comment("iter %u%s"%(u,extraComment)))
      plrIdx = ((u+pflr) % (self.numVgprBuffer+1)) % kernel["LoopIters"]
      plrIdxDTV = (u+pflr) % kernel["LoopIters"]

      localReads = Code.Module()
      localReadsA = Code.Module()
      localReadsB = Code.Module()

      pointerLWCode = Code.Module()
      pointerLRCode = Code.Module()
      waitCode = Code.Module()  # may be overwritten (not added to) below
      macIterCode = Code.Module()
      waitLWCode = Code.Module()
      syncCode = Code.Module()

      # vregSetIdx for DTV
      vregSetIdxLR = vregSetIdxMFMA
      if kernel["LoopIters"] > 1 and u == kernel["LoopIters"] - 1:
        # use next vregSet for the last loopIter (exception: LoopIters==1)
        # LoopIters==1 case, local read is for the current iteration and not for the next iteration
        vregSetIdxLR = (vregSetIdxLR + 1) % loopCopies
        # final loop case, use vregSetIdx for noLoadLoop (NGLL(PGR2) or NLL(PGR1))
        if finalLoop:
          # finalLoop case, this is for NoLoadLoop (NGLL(PGR2) or NLL(PGR1))
          # PGR2 case, next is NGLL. Use first set
          # PGR1 case, next is NLL. Use second set
          vregSetIdxLR = 0 if kernel["PrefetchGlobalRead"] == 2 else 1
          # flip vregSetIdx if isDTVodd is True
          if self.generateDTVodd:
            vregSetIdxLR = 1 - vregSetIdxLR

      if self.enable["LocalRead"]:
        hasLiveLdsData = kernel["PrefetchGlobalRead"] or (uDu < kernel["DepthULdsDivisor"]-1)
        # reads for current loop are done in previous iteration because of wider local read
        doReadA = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadA - self.numItersPLR)
        doReadB = (u < kernel["LoopIters"]/self.numIterPerCoalescedReadB - self.numItersPLR)
        # reads for next loop
        doReadA = doReadA or (hasLiveLdsData and u > localWriteEndIter)
        doReadB = doReadB or (hasLiveLdsData and u > localWriteEndIter)
        # disable LocalRead if DirectToVgpr is enabled
        for iui in range(0,kernel["InnerUnroll"]):
          doReadA = doReadA and iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]
          doReadB = doReadB and iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]
          if doReadA:
            localReads.addText(self.comment("local read a"))
            bufferIdx = plrIdx*self.numIterPerCoalescedReadA
            if self.packDTVA:
              # DTV + pack case, offset bufferIdx for local read packing instructions
              bufferIdx = plrIdxDTV*self.numIterPerCoalescedReadA + vregSetIdxLR * kernel["LoopIters"]
            localReadCodeA, packCodeA = self.localReadDo(kernel, bufferIdx, iui*self.numReadsIterCoalescedA, 0, tensorParametersA)
            localReads.addCode(localReadCodeA)
            localReadsA.addCode(localReadCodeA)
            pack[plrIdx*self.numIterPerCoalescedReadA].addCode(packCodeA)
          if doReadB:
            localReads.addText(self.comment("local read b"))
            bufferIdx = plrIdx*self.numIterPerCoalescedReadB
            if self.packDTVB:
              # DTV + pack case, offset bufferIdx for local read packing instructions
              bufferIdx = plrIdxDTV*self.numIterPerCoalescedReadB + vregSetIdxLR * kernel["LoopIters"]
            localReadCodeB, packCodeB = self.localReadDo(kernel, bufferIdx, iui*self.numReadsIterCoalescedB, 0, tensorParametersB)
            localReads.addCode(localReadCodeB)
            localReadsB.addCode(localReadCodeB)
            pack[plrIdx*self.numIterPerCoalescedReadB].addCode(packCodeB)
          # Don't increment the LRO if we are going to reset them below:
          if not isResetLroIter or iui != kernel["InnerUnroll"]-1:
            if doReadA:
              localReads.addText(self.comment("local read increment a"))
              localReads.addText(self.localReadInc(kernel, iui, tensorParametersA))
            if doReadB:
              localReads.addText(self.comment("local read increment b"))
              localReads.addText(self.localReadInc(kernel, iui, tensorParametersB))

      if kernel["PrefetchGlobalRead"]:
        # wait code for DirectToVgpr
        if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
          if self.enable["Wait"]:
            # not generate wait here
            #  1) for the first unroll with self.canOptimizePreLoopLWVmcnt = True
            #  2) local write code in previous u (u-1) has waitcnt vmcnt
            prevVmcnt = False
            prevLocalWrite = ""
            if (u > 0 and kernel["ScheduleIterAlg"] == 3):
              for up in range(u):
                prevLocalWrite += ' '.join([str(x) for x in self.perIterLocalWriteCode[up].flatitems()])
              prevVmcnt = "vmcnt" in prevLocalWrite
            if not (firstIter and u == 0 and self.canOptimizePreLoopLWVmcnt) and not prevVmcnt:
              retStr = self.getWaitcntCodeForDirectToVgpr(kernel, localWriteEndIter, u, firstIterForWait)
              kl.append(retStr)
        # put barrier at localWriteEndIter+1
        if u == localWriteEndIter+1 or (u == (localWriteEndIter+1)%kernel["LoopIters"] and kernel["ScheduleIterAlg"] == 2):
          if self.enable["Wait"]:
            if kernel["DirectToLdsA"] or kernel["DirectToLdsB"]:
              # skip generating wait for global read again here in DirectToVgpr case or no DirectToVgpr + PGR=2
              # no DTV and PGR=2 case, wait is generated at sync (barrier), which is before next local read
              if not(kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) and not kernel["PrefetchGlobalRead"]==2:
                kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "12wait for global read"))
              else:
                # DirectToVgpr + DirectToLds case, add waitcnt vmcnt before s_barrier
                # Except for PGR=2 and Load C case. In that case, Load C is executed after necessary Load A and B.
                # Wait for Load C is already done here in PGR=2 case.
                needLoadC = kernel["StoreCInUnroll"] and (not kernel["AtomicAddC"]) and kernel["ProblemType"]["UseBeta"]
                if not (kernel["PrefetchGlobalRead"]==2 and needLoadC):
                  retStr = self.getWaitcntCodeForDirectToVgpr(kernel, localWriteEndIter, u, firstIterForWait, beforeBarrier=True)
                  waitLWCode.addCode(retStr)
            # skip local write wait if DirectToVgpr + DirectToLds is enabled
            # (no local write code. Global read wait for DirectToLds is already done)
            if not kernel["NoLdsWriteCode"]:
              waitLWCode.addCode(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "3wait for local write"))
          if self.enable["Sync"]:
            if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
              if not (kernel["DirectToVgprA"] and kernel["DirectToVgprB"]):
                # put only barrier for DirectToVgpr (to avoid generating waitcnt for global read)
                # barrier is not necessary if both DirectToVgprA and B are enabled
                syncCode.addCode("s_barrier" + self.endLine)
            else:
              syncCode.addCode(self.syncThreads(kernel))

        if isSwapAndResetLwoIter: # ResetLroIter
          if self.enable["LocalWrite"]:
            # local write for next iter, used to have local writes here
            pointerLWCode.addText(self.comment("local write swap offsets a"))
            pointerLWCode.addText(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
            pointerLWCode.addText(self.comment("local write swap offsets b"))
            pointerLWCode.addText(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))
            pointerLWCode.addText(self.localWriteInitPointers(kernel, tensorParametersA))
            pointerLWCode.addText(self.localWriteInitPointers(kernel, tensorParametersB))

        if isSwapLroIter: # ResetLroIter
          if self.enable["LocalRead"]:
            # Swap, reset, or increment the LRO:
            pointerLRCode.addText(self.comment("local read swap offsets a"))
            pointerLRCode.addText(self.localReadSwapOffsets(kernel, expand, tensorParametersA))
            pointerLRCode.addText(self.comment("local read swap offsets b"))
            pointerLRCode.addText(self.localReadSwapOffsets(kernel, expand, tensorParametersB))

      if isResetLroIter: # ResetLroIter
        if self.enable["LocalRead"]:
          pointerLRCode.addText(self.comment("local read init pointers a"))
          pointerLRCode.addText(self.localReadInitPointers(kernel, tensorParametersA))
          pointerLRCode.addText(self.comment("local read init pointers b"))
          pointerLRCode.addText(self.localReadInitPointers(kernel, tensorParametersB))

      # we initiate lgkmcnt to 0, then assigning it correct value in makeSubIterSchedule()
      if self.enable["Wait"]:
        if self.getConditionToSkipLocalReadWriteWait(kernel, True, u, kernel["LoopIters"] - 1):
          waitCode = self.wait(kernel, tensorParametersA, tensorParametersB, \
              -1, 0, 0, \
              "wait for prior local read local write")

      luIdx = (u) % (self.numVgprBuffer+1) # local to use for MACs
      if self.enable["MAC"]:
        if kernel["EnableMatrixInstruction"]:
          macIterCode.addCode(self.mfmaIter(kernel, u, kernel["InnerUnroll"], vregSetIdxMFMA, firstIter=firstIter and u == 0))
        else:
          macIterCode.addCode(self.macIter(kernel, luIdx, kernel["InnerUnroll"], True ))

      ###### unroll loop efficiency implementation######################################
      # unroll loop efficiency implementation
      ## split A&B fetch&MAC code into multiple groups
      ## splitting strategy   based on TT size
      ## 6x4 -> split  MAC blob(s) into group of 8(s) and 16 FMA instructions.
      ##        LDS fetch(es) into group of A{1-2)B(0) , A(3),B(1) (not implemented yet)
      ## 4x6 -> split  MAC blob(s) into group of 8(s) and 16 FMA instructions.
      ##        LDS fetch(es) into group of B{1-2)A(0) , B(3),A(1)
      ## 4x4 -> split into group of 8 and 8  MAC(s)
      ## 6x6 -> split into group of 12 MAC(s)
      ## 8x4/4x8 -> split into group of 16 and 16  MAC(s)
      ## 8x8 -> split into group of 16 MAC(s)
      ## supports only PLR=0
      ###############################################################################
      if self.numItersPLR or (not globalParameters["UnrollLoopEfficiencyEnable"]):
        subIterCode = self.makeSubIterSchedule(kernel, localReads, \
                        u, pointerLWCode, pointerLRCode, waitCode, macIterCode, waitLWCode, syncCode, pack[luIdx])
        kl.append(subIterCode) # add scheduled "other", local reads, local writes
        for item in list(pack[luIdx].items()):
          if item.tempVgpr != None:
            self.vgprPool.checkIn(item.tempVgpr)
            item.tempVgpr = None
        pack[luIdx] = Code.Module()
      else:
        macIterCode = Code.Module()
        MacitemsReorder = []
        if self.enable["MAC"]:
          luIdx = (u) % (self.numVgprBuffer+1) # local to use for MACs
          macIterCode.addCode(self.macCode(kernel, luIdx, kernel["InnerUnroll"] ))
        MacIteritems = macIterCode.flatitems()
        #remove last and second entry from list if AggressiveMode is set
        # re-insert them back later
        if (kernel["AggressivePerfMode"]):
          MacIteritems = MacIteritems[:-1]
          MacIteritems.pop(1)
        #print("number MacItems\n",len(MacIteritems))
        blockWidth = tensorParametersA["localReadInstruction"].blockWidth
        numVectorsPerTileA = (kernel["ThreadTile%u"%tensorParametersA["tensorIdx"]]/kernel["VectorWidth"])
        numReadsPerVectorA = (kernel["VectorWidth"] * tensorParametersA["bpe"] ) / (blockWidth*4)
        numVectorsPerTileB = (kernel["ThreadTile%u"%tensorParametersB["tensorIdx"]]/kernel["VectorWidth"])
        TotalnumLdsFetches = numVectorsPerTileA*numReadsPerVectorA + numVectorsPerTileB*numReadsPerVectorA
        ## Rules for applying kernel["UnrollLoopEfficiencyEnable"]
        ## if A+B fetches <= 3 no split approach
        if not TotalnumLdsFetches > 3:
          subIterCode = self.makeSubIterSchedule(kernel, localReads, \
                       u, pointerLWCode, pointerLRCode, waitCode, macIterCode)
          kl.append(subIterCode) # add scheduled "other", local reads, local writes
        else:
          if ((kernel["ThreadTile0"] == 6 and kernel["ThreadTile1"] == 4) or
             (kernel["ThreadTile0"] == 4 and kernel["ThreadTile1"] == 6)):
            numGroups = 2   #group0 = 8 MAC(s)  #group1 = 16 MAC(s) (6x4 - 4x2)
            # ldsItems for splitting lds(s)
            ldsItems = ([[4,2],[2,2]]) if kernel["ThreadTile0"] == 6 else ([[2,4],[2,2]])
            macItems = [8,16]
            waitCntItems = [0,0]
          elif (kernel["ThreadTile0"] == 4 and kernel["ThreadTile1"] == 4):
            numGroups = 2   #group0 = 8 MAC(s)  #group1 = 8  MAC(s) 2)
            ldsItems = ([[4,2],[0,2]])
            macItems = [8,8]
            waitCntItems = [0,0]
          elif (kernel["ThreadTile0"] == 6 and kernel["ThreadTile1"] == 6):
            numGroups = 2   #group0 = 8 MAC(s)  #group1 = 16 MAC(s) 2)
            ldsItems = ([[4,4],[2,2]])
            macItems = [16,20]
            waitCntItems = [0,0]
          elif ((kernel["ThreadTile0"] == 8 and kernel["ThreadTile1"] == 4) or
             (kernel["ThreadTile0"] == 4 and kernel["ThreadTile1"] == 8)):
            numGroups = 2   #group0 = 16 MAC(s)  #group1 = 16 MAC(s) 2)
            ldsItems = ([[4,4],[4,0]]) if kernel["ThreadTile0"] == 8 else ([[4,4],[0,4]])
            macItems = [16,16]
            waitCntItems = [0,0]
          elif (kernel["ThreadTile0"] == 8 and kernel["ThreadTile1"] == 8):
            numGroups = 2   #group0 = 8 MAC(s)  #group1 = 8 MAC(s) 2)
            #ldsItems = ([[4,4],[4,4]])
            macItems = [16,48]
            waitCntItems = [0,0]
          AitemsToReorder = localReadsA.flatitems()
          BitemsToReorder = localReadsB.flatitems()
          ##reorder code?? based on LDS fetch
          ## works for 2 groups.. needs fix for more than 2 groups
          for iter in range(0,numGroups):
            endIdx   = ldsItems[iter][0] if iter == 0 else kernel["ThreadTile%u"%tensorParametersA["tensorIdx"]]
            startIdx = 0  if iter == 0 else ldsItems[iter-1][1]
            for Bitems in range(startIdx, startIdx+ldsItems[iter][1]):
              for Aitems in range(0, endIdx):
                idx = Aitems+(kernel["ThreadTile%u"%tensorParametersA["tensorIdx"]]*Bitems)
                MacitemsReorder.append(MacIteritems[idx])
              if (iter != 0):
                for Bitems in range(0, ldsItems[iter-1][1]):
                  for Aitems in range(ldsItems[iter-1][0], kernel["ThreadTile%u"%tensorParametersA["tensorIdx"]]):
                     MacitemsReorder.append(MacIteritems[Aitems+((kernel["ThreadTile%u"%tensorParametersA["tensorIdx"]])*Bitems)])
          #print("Total number mac items A(%u)\n",len(MacitemsReorder))
          #print("Total number ds items A(%u)\n"%(TotalnumLdsFetches))
          #print("number ds items A_B(%u .. %u)\n"%(len(AitemsToReorder),len(BitemsToReorder)))
          #reorder LDS fetches so order in which A+B fetches matches MAC blob
          #e.g 8x4 original order in DGEMM case A[0-1]A[2-3]A[4-5]A[6-7]B[0-1][2-3]
          #we want to re-order them into A[0-1][2-3]B[0-1]B[2-3];  In all other except
          #DGEMM type, number of LDS fetches <=4 so no need for LDS re-order
          if self.enable["LocalRead"] and TotalnumLdsFetches > 4:
            localReads = Code.Module()
            for iter in range(0,numGroups):
              if len(AitemsToReorder):
                localReads.addText(self.comment("local read a"))
                numLocalReads = roundUp((ldsItems[iter][0])/kernel["VectorWidth"])
                ##print("number ds items A(%u..%u)\n"%(iter,numLocalReads))
                for idx in range(0,numLocalReads):
                  localReads.addCode(AitemsToReorder[0])
                  AitemsToReorder = AitemsToReorder[1:]
              if len(BitemsToReorder):
                numLocalReads = roundUp(ldsItems[iter][1]/kernel["VectorWidth"])
                ##print("number ds items B(%u..%u)\n"%(iter,numLocalReads))
                localReads.addText(self.comment("local read b"))
                for items in range(0,numLocalReads):
                  localReads.addCode(BitemsToReorder[0])
                  BitemsToReorder = BitemsToReorder[1:]
              if iter == 0:
                waitCntItems[iter] = TotalnumLdsFetches - ((ldsItems[iter][0])/kernel["VectorWidth"] + (ldsItems[iter][1])/kernel["VectorWidth"])
              elif iter+1 != numGroups:
                waitCntItems[iter] = TotalnumLdsFetches - ((ldsItems[iter][0])/kernel["VectorWidth"] + (ldsItems[iter][1])/kernel["VectorWidth"] + waitCntItems[iter-1])
              else:
                waitCntItems[iter] = 0
              #print("Waitcnt(%u..%u)\n"%(iter,waitCntItems[iter]))
          for iter in range(0,numGroups):
            #Mac Code
            #place holder for future work Instruction class for generting MAC instruction
            #FMAInstruction = MacInstruction(globalParameters["CurrentISA"])
            subIterCode = Code.Module()
            waitCode = Code.Module()
            macIterCodeGrp = Code.Module()
            doOnce = False
            if self.enable["MAC"]:
              numMacItems = macItems[iter]
              for Items in range(0,numMacItems):
                macItem = MacitemsReorder.pop(0)
                macIterCodeGrp.addCode(macItem)
                ## add s_setprio 1 when AggressivePerfMode ==1 as second instruction for second-last blob macCode
                if (kernel["AggressivePerfMode"] and not doOnce):
                    macIterCodeGrp.addInst("s_setprio ","1","Raise priority while processing macs")
                    doOnce = True
              ## add s_setprio 0 when AggressivePerfMode ==1 as last instruction
              if (kernel["AggressivePerfMode"]):
                macIterCodeGrp.addInst("s_setprio ","0","Reset priority after macs")
            #print("ReadWaitcnt(%u..%u)\n"%(iter,waitCntItems[iter]))
            #print("WriteCodeCount(%d..%u)\n",u,self.perIterLocalWriteCode[u].count())
            if (iter == 0):
              if self.enable["Wait"]:
                #calculate lgkmcnt value including read+write for first iteration
                waitCntVal = waitCntItems[iter] + 1 if (self.perIterLocalWriteCode[u].count()>0) else waitCntItems[iter]
                # read + write instructions lgkmcnt (1=> for write)
                # build waitCnt using new lgkmcnt
                waitCode = Code.WaitCnt(self.version, waitCntVal,-1,"wait for prior local read")
              subIterCode = self.makeSubIterSchedule(kernel, localReads, \
                       u, pointerLWCode, pointerLRCode, waitCode, macIterCodeGrp)
            else:
                #last group only pointer + localWrite Code
              if self.enable["Wait"]:
                waitCode = Code.WaitCnt(self.version, waitCntItems[iter],-1,"wait for prior local read & local writes")
              subIterCode.addCode(waitCode)
              subIterCode.addCode(macIterCodeGrp)
            kl.append(subIterCode) # add scheduled "other", local reads, local writes

    # close unrolled loop
    if expand:
      if not finalLoop:
        kl.append(self.comment3("Unrolled Loop - End %u/%u"%(lc+1, loopCopies)))
      else:
        kl.append(self.comment3("Unrolled Loop - End %u/%u (final)"%(lc+1, loopCopies)))

        # add wait for global read here canOptimizePreLoopLWVmcnt is true and DirectToVgpr is true
        # StoreCInUnroll does not require this wait because wait code is generated at the top of inner loop
        if kernel["PrefetchGlobalRead"] and self.canOptimizePreLoopLWVmcnt and (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]) \
          and (not kernel["StoreCInUnroll"]):
          retStr = self.getWaitcntCodeForDirectToVgpr(kernel, localWriteEndIter, u=0, firstIter=False)
          kl.append(retStr)

    else:
      kl.append(self.comment3("Unrolled Loop - End"))

    oddLabel = lc == 0
    kl.append(self.closeLoop(kernel, self.unrollIdx, finalLoop, loopCopies, oddLabel=oddLabel))

  ##############################################################################
  # Kernel Body
  ##############################################################################
  def kernelBody( self, kernel, tensorParametersA, tensorParametersB ):
    expand = kernel["ExpandPointerSwap"]

    ####################################
    # Begin String
    kl = []

    ####################################
    # Function Prefix
    kl.append(self.comment3("Function Prefix"))
    kl.append(self.functionPrefix(kernel))

    ####################################
    # Function Signature
    ####################################
    kl.append(self.comment3("Begin Kernel"))
    kl.append(self.functionSignaturePrefix(kernel))

    beforeFunctionSignature = '\n'.join([str(x) for x in kl])
    kl = []

    kl.append(self.functionSignatureSuffix(kernel))
    kl.append(self.functionBegin(kernel))

    # init code optimization: generate local read address code before wait for kernel arg load (in allocateResources())
    klLR = []
    self.isLocalReadAddressCodeGenerated = False

    if self.enable["PreLoop"]:
      if self.isInitCodeOptLR:
        ####################################
        # Local Read Addresses
        ####################################
        klLR += self.generateLocalReadAddressCode(kernel, tensorParametersA, tensorParametersB)

    # init code optimization : allocate resource
    self.lwaInitOptAllocate()

    lraCode=None
    placeholderInitCodeOpt=None
    if self.isInitCodeOptLW:
      lraCode = ""
      placeholderInitCodeOpt = "__placeholderInitCodeOpt__" # placeholder for local write code (for future replacement)
    if self.isInitCodeOptLR:
      # string for local read code
      lraCode = '\n'.join([str(x) for x in klLR])
    # local write code is generated later. Here, just add placeholder to replace with local write code
    if self.isInitCodeOptLW:
      lraCode += '\n' + placeholderInitCodeOpt + '\n'

    kl.append(self.comment3("Allocate Resources"))
    kl.append(self.allocateResources(kernel, lraCode))

    if (not kernel["PreloadKernelArguments"]):
      # not PreloadKernelArguments case (local read code is delayed in PreloadKernelArguments case)
      # add local read address code to kl after allocateResources()
      # generate Local Read Addresses for not isInitCodeOptLR case
      kl += self.generateLocalReadAddressCode(kernel, tensorParametersA, tensorParametersB)

    # doShadowInit performs initialization in the 'shadow' of the global mem prefetch
    self.doShadowInit = 0
    if kernel["PrefetchGlobalRead"]:
      if self.actualSummationLoops == 1:
        self.doShadowInit = 2 # 2 is both store setup and initC
      else:
        # can't do shadow initC with multiple summation since this resets the ValuC counters
        # on each unroll iteration.
        self.doShadowInit = 1 # 1 is just store setup

    if self.prefetchAcrossPersistent:
      # SrdC/D init before persistent loop
      kl.append(self.globalWriteWorkGroupInitBeforePersistentLoop(kernel))

      # init code for StoreCInUnroll (only once before persistent kernel loop)
      # SrdC/D init has to be done beforehand
      if self.storeCInUnroll:
        kl.append(self.initStoreCInUnroll(kernel))

      # first prefetch is outside persistent loop, subsequent prefetch will
      # be integrated into no-load-loop
      kl_NT, kl_LW = self.setupNewTile(kernel, tensorParametersA, tensorParametersB, isPap=False, isOptNLL=False)
      kl += kl_NT
      kl.append(self.openPersistentLoop(kernel))
    else:
      # prefetch is inside persistent loop
      kl.append(self.openPersistentLoop(kernel))
      kl_NT, kl_LW = self.setupNewTile(kernel, tensorParametersA, tensorParametersB, isPap=False, isOptNLL=False)
      kl += kl_NT

    # init code optimization : release resource
    self.lwaInitOptRelease()

    pack = [ Code.Module() for i in range (self.numVgprBuffer+1) ]
    self.preLoopLocalWriteCode = None

    # add local read address code if it is not generated (PreloadKernelArguments case)
    kl += self.generateLocalReadAddressCode(kernel, tensorParametersA, tensorParametersB)

    if kernel["PrefetchGlobalRead"]:
      if self.doShadowInit:
        kl.append(self.openShadowInit(kernel))
        # init code for StoreCInUnroll per each persistent kernel loop iteration
        # before generate new srdC/D (in globalWriteWorkGroupInit())
        if self.storeCInUnroll:
          kl.append(self.initStoreCInUnrollPerPersistentLoop(kernel))
        kl.append(self.globalWriteWorkGroupInit(kernel))
        # after genarating new SrdC,D, swap with backup values so that previous srdC,D is used in unroll loop for StoreCInUnroll
        if self.storeCInUnroll:
          kl.append(self.swapSrdCDandBackup(kernel))
        if self.doShadowInit == 2:
          kl.append(self.initC(kernel)) # initC while waiting for global reads
        kl.append(self.closeShadowInit(kernel))

      if self.enable["Wait"] and not self.canOptimizePreLoopLWVmcnt:
        kl.append(self.getWaitcntCodeForPGR(kernel))
        # These cases loop back and run the prefetch loop again
        # we need an extra barrier to ensure that the ds_reads (either for SR or MFMA) from previous iteration
        # have finished before we generate the prefetch for the next summation index.
        if kernel["PersistentKernel"] or kernel["StreamK"] > 0 or self.actualSummationLoops>1:
          kl.append( self.indent + self.syncStr + "// for PersistentKernel / StreamK " + self.endLine )

      if self.enable["LocalWrite"]:
        # local write
        self.preLoopLocalWriteCode = self.preLoopLocalWriteDo(kernel, tensorParametersA, tensorParametersB)
        kl.append(self.preLoopLocalWriteCode)
        # swap local ptrs
        kl.append(self.comment("local write swap a"))
        kl.append(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
        kl.append(self.comment("local write swap b"))
        kl.append(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))
        kl.append(self.localWriteInitPointers(kernel, tensorParametersA))
        kl.append(self.localWriteInitPointers(kernel, tensorParametersB))

      if kernel["PrefetchGlobalRead"] == 2:
        kl.append(self.openPrefetchGlobalRead2(kernel))
        if self.enable["GlobalRead"]:
          # if DirectToVgpr is enabled and swapGlobalRoad is true, swap the order of global read (B->A)
          tensorParameters1st = tensorParametersA
          tensorParameters2nd = tensorParametersB
          if self.isSwapGlobalReadOrderForDtvOrDtl(kernel):
            tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
          kl.append(str(self.directToLdsM0Update(kernel, 1, tensorParameters1st)))
          kl.append(str(self.globalReadDo(kernel, 0, tensorParameters1st, 1)))
          kl.append(str(self.directToLdsM0Update(kernel, 1, tensorParameters2nd)))
          kl.append(str(self.globalReadDo(kernel, 0, tensorParameters2nd, 1)))

          # swap local ptrs again if DirectToLds is enabled
          if kernel["DirectToLdsA"]:
            kl.append(self.comment("local write swap a"))
            kl.append(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
            kl.append(self.localWriteInitPointers(kernel, tensorParametersA))
          if kernel["DirectToLdsB"]:
            kl.append(self.comment("local write swap b"))
            kl.append(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))
            kl.append(self.localWriteInitPointers(kernel, tensorParametersB))

        kl.append(self.closePrefetchGlobalRead2(kernel))

      # prefetch-local
      if self.numItersPLR:
        # not generate wait for local write if LDS write code is not generated
        if self.enable["Wait"] and not kernel["NoLdsWriteCode"]:
          # TODO: need to check if we correctly checked-in the temp VGPR used for Int8 LocalWrite (uDu, PGR=2)
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "0prefetch wait for local write"))
        if self.enable["Sync"]:
          kl.append(self.syncThreads(kernel))

        # in some cases need an extra copy of the LDS read with appropriate double buffer offsets
        if self.enable["LocalRead"]:
          for plrIdx in range(0, self.numItersPLR):
            pack[plrIdx] = Code.Module()
            # no matter EPS or PAP, only prefect local once per plrIdx
            # for espi in range(0, (self.prefetchAcrossPersistent and kernel["ExpandPointerSwap"])+1):
            for espi in range(0, 1):
              for iui in range(0,kernel["InnerUnroll"]):
                if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]:
                  kl.append(self.comment("local read prefetch a"))
                  localReadCodeA, packCodeA = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadA, iui*self.numReadsIterCoalescedA, espi, tensorParametersA)
                  kl.append(localReadCodeA)
                  pack[plrIdx].addCode(packCodeA)
                if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]:
                  kl.append(self.comment("local read prefetch b"))
                  localReadCodeB, packCodeB = self.localReadDo(kernel, plrIdx*self.numIterPerCoalescedReadB, iui*self.numReadsIterCoalescedB, espi, tensorParametersB)
                  kl.append(localReadCodeB)
                  pack[plrIdx].addCode(packCodeB)
                if iui*self.numReadsIterCoalescedA < kernel["InnerUnroll"]:
                  kl.append(self.comment("local read inc a"))
                  kl.append(self.localReadInc(kernel, iui, tensorParametersA))
                if iui*self.numReadsIterCoalescedB < kernel["InnerUnroll"]:
                  kl.append(self.comment("local read inc b"))
                  kl.append(self.localReadInc(kernel, iui, tensorParametersB))
      kl.append(self.closeSumAtLeastUnroll(kernel, prefetch=True, isOptNLL=False, isPap=False, isNGLL=False))

    loopCopies = 2 if expand else 1

    if self.useInitAccVgprOpt:
      # generate first iteration code for init accvgpr opt
      kl.append(self.comment3("First Unrolled Iter for InitAccVgprOpt - Begin"))
      # open loop without Label
      kl.append(self.openLoop(kernel, self.unrollIdx, noLabelGen=True))
      self.loopBody( kernel, tensorParametersA, tensorParametersB, kl, pack, 0, loopCopies, False, firstIter=True )

    # open unrolled summation loop
    kl.append(self.comment3("Unrolled Loop(s) - Begin"))
    # In StoreCInUnroll case, LoopCounter check code is already generated. We need only LoopBeginLabel
    beginLabelOnly = kernel["StoreCInUnroll"]
    kl.append(self.openLoop(kernel, self.unrollIdx, beginLabelOnly=beginLabelOnly))

    lcStart = 0
    if self.useInitAccVgprOpt:
      lcStart = 1 if loopCopies == 2 else 0
    for lc in range(0, loopCopies):
      loopIndex = lcStart + lc
      if loopIndex >= loopCopies:
        loopIndex -= loopCopies
      # loop body code generation
      finalLoop = lc == loopCopies - 1
      self.loopBody( kernel, tensorParametersA, tensorParametersB, kl, pack, loopIndex, loopCopies, finalLoop )

    kl.append(self.comment("Before NLL: Check VGPR.checkin for INT8 LW"))

    # swap local write, read again before noLoadLoop if PrefetchGlobalRead and DirectToLds is enabled
    # In DirectToLds enabled case, local write address is necessary for prefetch global read (for m0).
    # However, even exit with DirectToLds will not pass with this code (limitation).
    # So far, this code is to make odd exit case (i.e. k is multiple of 2*depthU) pass for DirectToVgpr
    if not self.useInitAccVgprOpt and kernel["PrefetchGlobalRead"] and self.enable["LocalWrite"] and kernel["ExpandPointerSwap"]:
      # local write for next iter, used to have local writes here
      if(kernel["DirectToLdsA"]):
        kl.append(self.comment("local write swap offsets a"))
        kl.append(self.localWriteSwapOffsets(kernel, expand, tensorParametersA))
      if(kernel["DirectToLdsB"]):
        kl.append(self.comment("local write swap offsets b"))
        kl.append(self.localWriteSwapOffsets(kernel, expand, tensorParametersB))
    # swap local read point for self.useInitAccVgprOpt
    if self.useInitAccVgprOpt and kernel["ExpandPointerSwap"]:
      if self.enable["LocalRead"]:
        kl.append(self.comment("local read swap offsets a"))
        kl.append(self.localReadSwapOffsets(kernel, expand, tensorParametersA))
        kl.append(self.comment("local read swap offsets b"))
        kl.append(self.localReadSwapOffsets(kernel, expand, tensorParametersB))

    if kernel["PrefetchGlobalRead"] == 2:
      # re-generate store code for StoreCInUnroll (odd=0,isLast=False))
      self.generateStoreCCodeInUnrollLoop(kernel, 0, isLast=False)
      isOptNLL=False
      isPap=False
      if self.prefetchAcrossPersistent:
        isOptNLL = True
        isPap = True
      kl += self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=isOptNLL, isPap=isPap, isNGLL=True, pack=pack)

    # re-generate store code for StoreCInUnroll (no increment code (isLast=True))
    # this should be after NGLL code for PGR=2
    odd = 1
    self.generateStoreCCodeInUnrollLoop(kernel, odd, isLast=True)

    # This "NoLoad" loop is a copy of the unroll loop but with global loads + LDS writes removed
    # doShadowInit is required since this pushes up the store SRD initialization before the NLL
    # OptNLL only allowed for single summation index  - for multiple summation we (currently)
    # execute the NLL inside each unroll iteration not just once at the end.
    if kernel["PrefetchGlobalRead"]:
      if not kernel["SuppressNoLoadLoop"]:

        firstNLLgenerated = False
        if kernel["KernelLanguage"] == "Assembly" and kernel["OptNoLoadLoop"] and \
           kernel["BufferLoad"] and kernel["BufferStore"] and self.doShadowInit and \
           kernel["LocalSplitU"]==1 and kernel["GlobalSplitU"] == 1 and \
           self.actualSummationLoops==1:

          firstNLLgenerated = True

          # two different noLoadLoops:
          # 1. OptNLL & PAP global-read interleaved (only for PAP=ON)
          # (2. OptNLL : No PAP global-read (For PAP=OFF, or PAP=ON but the last tile))
          #  -> this is unified with 1. global-read is invalidated at the last tile.
          # 3. OrdinaryNLL (Not Opt.)
          self.saveLocalPointers(kernel)
          # deepCopy packCode for OptNLL noLoadLoop
          deepCopyPack = copy.deepcopy(pack)
          # keep StoreCInUnroll related code for the next noLoadLoop
          if kernel["StoreCInUnroll"]:
            self.backupStoreCInUnrollRelatedCode()
          isPap = self.prefetchAcrossPersistent
          kl += self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=True, isPap=isPap, isNGLL=False, pack=deepCopyPack)
          self.restoreLocalPointers(kernel)
          # restore StoreCInUnroll related code
          if kernel["StoreCInUnroll"]:
            self.restoreStoreCInUnrollRelatedCode()

        # skip second NLL code if enableSingleNLLOpt
        if not (self.enableSingleNLLOpt and firstNLLgenerated):
          papMode = self.prefetchAcrossPersistent and kernel["PrefetchAcrossPersistentMode"] == 1
          kl += self.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=False, isPap=papMode, isNGLL=False, pack=pack)

        else:
          # generate PrefetchGlobalLastIterEnd label
          kl.append(self.closeSumAtLeastUnroll(kernel, prefetch=False, isOptNLL=False, isPap=False, isNGLL=False))

        if kernel["StoreCInUnroll"]:
          # end process for StoreCInUnroll per PersistentLoop (NoOptNLL)
          kl.append(self.endProcessPersistentLoopforStoreCInUnrollNoOptNLL(kernel))

      # if PGR, last few iterations will have PLR,
      # and those PLR will not be used(register not checkIn) if without NoLoadLoop
      else:
        for i in range(self.numVgprBuffer):
          for item in list(pack[i].items()):
            if item.tempVgpr != None:
              self.vgprPool.checkIn(item.tempVgpr)
              item.tempVgpr = None

    if self.staggerU and self.actualSummationLoops>1:
      kl.append(self.comment("remove stagger offsets"))
      kl.append(self.removeStagger(kernel, tensorParametersA))
      kl.append(self.removeStagger(kernel, tensorParametersB))

    if not self.noTailLoop:
      ########################################
      # Tail Loop
      # PackSummationDims=1 requires that the tile slice does not cross DepthU
      # which means tail loop not needed.
      ########################################
      self.inTailLoop = True
      if kernel["LoopTail"] and not kernel["PackSummationDims"]:
        kl.append(self.comment3("Tail Loop"))

        # Update local write pointers in case the upcoming global reads are writing directly to LDS:
        if self.enable["LocalWrite"]:
          if kernel["PrefetchGlobalRead"]:
            kl.append(self.comment("local write reset offsets a"))
            kl.append(self.localWriteResetOffsets(kernel,  kernel["ExpandPointerSwap"], tensorParametersA))
            if kernel["ExpandPointerSwap"]:
              # reset local write offset in asm code as well
              kl.append(self.localWriteResetOffsets(kernel, False, tensorParametersA))
            kl.append(self.comment("local write reset offsets b"))
            kl.append(self.localWriteResetOffsets(kernel,  kernel["ExpandPointerSwap"], tensorParametersB))
            if kernel["ExpandPointerSwap"]:
              # reset local write offset in asm code as well
              kl.append(self.localWriteResetOffsets(kernel, False, tensorParametersB))

        if self.enable["GlobalRead"]:
          # tail: global read
          kl.append(self.calculateLoopNumIter(kernel, -1, False))
          if self.staggerU and self.actualSummationLoops==1:
            kl.append(self.comment("remove stagger offsets for tail loop"))
            kl.append(self.removeStagger(kernel, tensorParametersA))
            kl.append(self.removeStagger(kernel, tensorParametersB))

          # if DirectToVgpr is enabled and swapGlobalRoad is true, swap the order of global read (B->A)
          tensorParameters1st = tensorParametersA
          tensorParameters2nd = tensorParametersB
          tc1 = 'a'
          tc2 = 'b'
          if self.isSwapGlobalReadOrderForDtvOrDtl(kernel):
            tensorParameters1st, tensorParameters2nd = tensorParameters2nd, tensorParameters1st
            tc1, tc2 = tc2, tc1
          kl.append(self.comment("Update M0 for DTLDS"))
          tmpStr = str(self.directToLdsM0Update(kernel, 1, tensorParameters1st))
          tmpStr = tmpStr.replace("__placeholder__", str(0))
          kl.append(tmpStr)
          kl.append(self.comment("global read %s"%tc1))
          vregSetIdx = 0
          kl.append(str(self.globalReadDo(kernel, 2, tensorParameters1st, vregSetIdx)))
          kl.append(self.comment("Update M0 for DTLDS"))
          tmpStr = str(self.directToLdsM0Update(kernel, 1, tensorParameters2nd))
          tmpStr = tmpStr.replace("__placeholder__", str(0))
          kl.append(tmpStr)
          kl.append(self.comment("global read %s"%tc2))
          vregSetIdx = 0
          kl.append(str(self.globalReadDo(kernel, 2, tensorParameters2nd, vregSetIdx)))
        if self.enable["Wait"]:
          kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, 0, -1, -1, "2wait for global read"))
        if self.enable["Sync"]:
          kl.append(self.syncThreads(kernel))

        kl.append(self.doneGlobalABReads(kernel))

        # the following read/write addresses could be modified in recalcLocal(Read|Write)Addresses due to policy change
        self.oriLraA = None # back up original local read address vgpr
        self.oriLraB = None
        self.oriLwaA = None # back up original local write address vgpr
        self.oriLwaB = None
        for uDu in range(0, kernel["DepthULdsDivisor"]):
          # change local write policy from interleave-K to fractional as tail loop
          # iterate LDS read address one unit of K at a time
          # skip recalcLocalWriteAddresses if DirectToVgpr is enabled
          if kernel.enabledSplitLDS and not (kernel["DirectToVgprA"]):
            kl.append(self.comment("Recalc local write offset A"))
            kl.append(self.recalcLocalWriteAddresses(kernel, tensorParametersA, uDu))
          if kernel.enabledSplitLDS and not (kernel["DirectToVgprB"]):
            kl.append(self.comment("Recalc local write offset B"))
            kl.append(self.recalcLocalWriteAddresses(kernel, tensorParametersB, uDu))
          if self.enable["Sync"]:
            if uDu > 0:
              kl.append(self.comment("sync before local write"))
              kl.append(self.syncThreads(kernel))
          if self.enable["LocalWrite"] and not kernel["NoLdsWriteCode"]:
            # tail: local write
            kl.append(self.localWriteInitPointers(kernel, tensorParametersA))
            kl.append(self.localWriteInitPointers(kernel, tensorParametersB))
            kl.append(self.comment("local write a"))
            tempLWCodeModA = self.localWriteDo(kernel, tensorParametersA, None)
            kl.append(tempLWCodeModA)
            kl.append(self.comment("local write b"))
            tempLWCodeModB = self.localWriteDo(kernel, tensorParametersB, None)
            kl.append(tempLWCodeModB)
          # change local read policy from wider local read to one unit of K at a time
          # DirectToVgpr case, use original wider local read instead of recalculating local read address
          if not (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]):
            kl.append(self.comment("Recalc local read offsets"))
            kl.append(self.recalcLocalReadAddressesAB(kernel))
          if self.enable["Wait"]:
            # TODO: need to check if we correctly checked-in the temp VGPR used for Int8 LocalWrite (uDu, PGR=2)
            kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, 0, -1, "5wait for local write"))
          if self.enable["Sync"]:
            kl.append(self.syncThreads(kernel))
          #kl.append(self.dumpLds(kernel, 0, 8))

          # tail: re-init local read addresses
          if kernel["PrefetchGlobalRead"]:
            kl.append(self.comment("local read reset offsets a"))
            kl.append(self.localReadResetOffsets(kernel, tensorParametersA))
            kl.append(self.comment("local read reset offsets b"))
            kl.append(self.localReadResetOffsets(kernel, tensorParametersB))
            kl.append(self.comment("local read init pointers a"))
            kl.append(self.localReadInitPointers(kernel, tensorParametersA))
            kl.append(self.comment("local read init pointers b"))
            kl.append(self.localReadInitPointers(kernel, tensorParametersB))
          # tail: macs
          kl.append(self.comment("tail loop: macs"))
          kl.append(self.openLoop(kernel, -1, uDu if kernel.enabledSplitLDS else None))

          # Try to use InnerUnroll in the tail loop if allowed:
          KinInnerUnroll = kernel["InnerUnroll"]
          if kernel["EnableMatrixInstruction"]:
            KinInnerUnroll *= kernel["MatrixInstK"]

          tailLoopInnerUnroll = 1
          asem = kernel["AssertSummationElementMultiple"]
          gsu = kernel["GlobalSplitU"]
          if ((asem%gsu == 0) and (asem//gsu) % KinInnerUnroll == 0):
            tailLoopInnerUnroll = kernel["InnerUnroll"]
          elif (kernel["LocalDotLayout"] > 1) and (kernel["InnerUnroll"] == kernel["LocalDotLayout"]):
            tailLoopInnerUnroll = kernel["InnerUnroll"]
          # need to unroll tail loop for the following cases
          mEnd = 1
          if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"] or kernel["DirectToLdsA"] or kernel["DirectToLdsB"]) \
             and kernel["EnableMatrixInstruction"]:
            mEnd = kernel["_DepthULds"]//(kernel["MatrixInstK"]*kernel["LocalSplitU"])
          elif kernel["EnableMatrixInstruction"] and \
            ((kernel["LdsPadA"] and kernel["LdsBlockSizePerPadA"]) or (kernel["LdsPadB"] and kernel["LdsBlockSizePerPadB"])):
            # LdsPad + LBSPP case, address increment is not distributed uniformly. So, we need to unroll tail loop
            mEnd = kernel["_DepthULds"]//(kernel["MatrixInstK"]*kernel["LocalSplitU"])

          for mValue in range(mEnd):
            if mEnd > 1:
              # print tail loop counter if mEnd>1 (means do tail loop unroll)
              kl.append(self.comment("tail loop unroll iter %u"%(mValue)))
            pack[0] = Code.Module()
            # always use vregSetIdx=0 for DirectToVgpr + tail loop
            vregSetIdxMFMA = 0
            for iui in range(0, tailLoopInnerUnroll):
              if self.enable["LocalRead"]:
                # local read buffer id. No prefetch in tail loop case.
                bufIdx = (mValue % (self.numVgprBuffer+1)) % kernel["LoopIters"]
                # DTVpack case, use different bufIdx for all loop iterations
                bufIdxDTV = mValue % kernel["LoopIters"]
                bufIdxA = bufIdxDTV if self.packDTVA else bufIdx
                bufIdxB = bufIdxDTV if self.packDTVB else bufIdx
                if mValue*self.numReadsIterCoalescedA < mEnd:
                  # Reading 16-bit data from LDS requires packing when ECC enabled
                  kl.append(self.comment("local read a"))
                  localReadCodeA, packCodeA = self.localReadDo(kernel, bufIdxA*self.numIterPerCoalescedReadA, iui*self.numIterPerCoalescedReadA, 0, tensorParametersA)
                  kl.append(localReadCodeA)
                  pack[0].addCode(packCodeA)
                if mValue*self.numReadsIterCoalescedB < mEnd:
                  kl.append(self.comment("local read b"))
                  localReadCodeB, packCodeB = self.localReadDo(kernel, bufIdxB*self.numIterPerCoalescedReadB, iui*self.numIterPerCoalescedReadB, 0, tensorParametersB)
                  kl.append(localReadCodeB)
                  pack[0].addCode(packCodeB)
                # adjustment for DirectToLds case
                iuiParam = iui + tailLoopInnerUnroll * mValue
                kl.append(self.comment("local read inc a"))
                kl.append(self.localReadInc(kernel, iuiParam, tensorParametersA))
                kl.append(self.comment("local read inc b"))
                kl.append(self.localReadInc(kernel, iuiParam, tensorParametersB))
            if self.enable["Wait"]:
              kl.append(self.wait(kernel, tensorParametersA, tensorParametersB, -1, -1, 0, "4wait for local read"))

            if kernel["EnableMatrixInstruction"]:
              kl.append(pack[0])
              # vgpr.checkin for all the checked-out vgpr in LocalRead
              for item in list(pack[0].items()):
                if item.tempVgpr != None:
                  self.vgprPool.checkIn(item.tempVgpr)
                  item.tempVgpr = None
              pack[0] = Code.Module()

            if self.enable["MAC"]:
              if kernel["EnableMatrixInstruction"]:
                # always use vregSetIdx=0 for DirectToVgpr + tail loop
                vregSetIdxMFMA = 0
                kl.append(self.mfmaIter(kernel, mValue, tailLoopInnerUnroll, vregSetIdxMFMA, False, True))
              else:
                kl.append(self.macIter(kernel, mValue, tailLoopInnerUnroll, True, True))

            finalLoop = mValue == mEnd - 1
            kl.append(self.closeLoop(kernel, -1, finalLoop, loopCopies, uDu if kernel.enabledSplitLDS else None, skipCondJumpCounter=mValue))
      # always emit the skip-tail-loop label
      kl.append(self.closeLoop(kernel, -1, None, loopCopies, emitEndLabelOnly=True))
      # tail: close
      self.inTailLoop = False

    # extra summation loops: global increment and close
    for i in reversed(range(self.otherSummationLoops)):
      kl.append(self.comment("global read inc AB"))
      kl.append(self.globalReadIncrementAB(kernel, i, 0))
      kl.append(self.closeLoop(kernel, i, True, loopCopies))

    if self.prefetchAcrossPersistent and kernel["PrefetchAcrossPersistentMode"] != 1:
      kl.append(str(self.openPrefetchAcrossPersistent(kernel, isOptNLL=False)))
      kl_NT, _ = self.setupNewTile(kernel, self.tPA, self.tPB, isPap=True, isOptNLL=False)
      kl += kl_NT
      kl.append(str(self.closePrefetchAcrossPersistent(kernel, isOptNLL=False)))

    kl.append(self.endSummation(kernel))
    if self.enable["PostLoop"]:
      if not self.doShadowInit:
        kl.append(self.globalWriteWorkGroupInit(kernel))

      ####################################
      # Shift Vector Components
      ####################################
      if kernel["EdgeType"] == "ShiftPtr":
        # GuaranteeNoPartial means each component in the vector loads is always valid.  In this case we
        # don't need the unshift code

        # shift vector components d0
        if not kernel["GuaranteeNoPartialA"] and self.readTileDimVectorA:
          kl.append(self.comment("shift vector components d0"))
          kl.append(self.shiftVectorComponents(kernel, tensorParametersA))

        # shift vector components d1, for MFMA version, B never entered this
        if not kernel["GuaranteeNoPartialB"] and self.readTileDimVectorB:
          kl.append(self.comment("shift vector components d1"))
          kl.append(self.shiftVectorComponents(kernel, tensorParametersB))

      # complex declare tmp registers
      kl.append(self.complexDeclareTmpRegisters(kernel))

      ####################################
      # LocalSplitU reduction
      ####################################
      #if kernel["NumThreads"]%kernel["MacroTile0"] == 0:
      if kernel["LocalSplitU"] > 1:
        kl.append(self.comment3("LocalSplitU Reduction"))
        if self.enable["Sync"]:
          # not generate sync code when both DirectToVgprA and B are enabled
          # in this case, LDS is not used and no need to sync to use LDS for LocalSplitU
          if not (kernel["DirectToVgprA"] and kernel["DirectToVgprB"]):
            kl.append(self.syncThreads(kernel))

        # LocalSplitU: local write
        kl.append(self.comment("LocalSplitU: local write"))
        kl.append(self.localSplitULocalWrite(kernel))

        # LocalSplitU: local read
        kl.append(self.comment("LocalSplitU: local read"))
        kl.append(self.localSplitULocalRead(kernel))

        # LocalSplitU: local read
        kl.append(self.comment("LocalSplitU: reduction"))
        kl.append(self.localSplitUReduction(kernel))

        # LocalSplitU: global write indices
        kl.append(self.comment("LocalSplitU: global write indices"))
        kl.append(self.localSplitUGlobalWriteIndices(kernel))

        # LocalSplitU: global write
        kl.append(self.comment("LocalSplitU: global write"))
        kl.append(self.localSplitUGlobalWrite(kernel))

      else:
        ####################################
        # NOT LocalSplitU
        ####################################

        # global write indices
        kl.append(self.comment("not-LocalSplitU: global write indices"))
        kl.append(self.notLocalSplitUGlobalWriteIndices(kernel))

        # global write
        kl.append(self.comment("not-LocalSplitU: global write"))
        kl.append(self.notLocalSplitUGlobalWrite(kernel))

    # After we know the #-of-globalwrite instructions, we can go back to replace the pre-loop LW vmcnt
    # Note that currently, this code-replacement occurs only when PrefetchAcrossPersistent=True,
    # otherwise, nothing is changed
    if self.preLoopLocalWriteCode != None:
      self.replacePreLoopLWVmcnt(kernel)

    # function suffix
    kl.append(self.functionEnd(kernel, True))
    kl.append(self.functionSuffix(kernel))

    kStr = '\n'.join([str(x) for x in kl])
    # init code opt
    if placeholderInitCodeOpt != None:
      # replace placeholder with localWriteCode
      kStrLW = '\n'.join([str(x) for x in kl_LW])
      kStr = kStr.replace(placeholderInitCodeOpt, kStrLW)
    afterFunctionSignature = kStr

    error = self.overflowedResources

    # function signature last since it needs to know how many gprs were actually used
    kStr = beforeFunctionSignature + self.functionSignature(kernel) + afterFunctionSignature
    return (error,kStr)

  ##############################################################################
  # Kernel Body
  ##############################################################################
  def generateLocalReadAddressCode( self, kernel, tensorParametersA, tensorParametersB ):
    kl = []
    # if code is already generated, just return
    if self.isLocalReadAddressCodeGenerated:
      return kl
    ####################################
    # Local Read Addresses
    ####################################
    kl.append(self.comment3("Local Read Addresses"))

    # tile assignments
    kl.append(self.comment("local read addresses: tile assignments a/b"))
    kl.append(self.lraTileAssignment(kernel, tensorParametersA, tensorParametersB))

    # final offsets
    kl.append(self.comment("local read addresses: final offsets a"))
    kl.append(self.lraFinalOffset(kernel, tensorParametersA))
    kl.append(self.comment("local read addresses: final offsets b"))
    kl.append(self.lraFinalOffset(kernel, tensorParametersB))

    # declare addresses
    kl.append(self.comment("local read addresses: declare addresses a"))
    kl.append(self.lraDeclareAddresses(kernel, tensorParametersA))
    kl.append(self.comment("local read addresses: declare addresses b"))
    kl.append(self.lraDeclareAddresses(kernel, tensorParametersB))

    # set flag
    self.isLocalReadAddressCodeGenerated = True

    return kl

  ##############################################################################
  #
  #   Functions to Write Kernel Segments
  #
  ##############################################################################

  def comment1(self, text):
    """
    single line comment
    """

    s = ""
    s += self.indent
    s += self.commentPrefix
    s += " %s " % text
    s += self.commentSuffix
    s += self.endLine
    return s

  def comment(self, text):
    """
    comment with prior newline
    """

    s = ""
    s += self.endLine
    s += self.comment1(text)
    return s

  def comment3(self, text):
    """
    3-line comment
    """

    s = ""
    s += self.endLine
    s += self.indent
    s += self.commentPrefix
    s += self.commentHR
    s += self.commentSuffix
    s += self.endLine

    for line in text.split("\n"):
      s += self.indent
      s += self.commentPrefix
      s += " %-38s " % line
      s += self.commentSuffix
      s += self.endLine

    s += self.indent
    s += self.commentPrefix
    s += self.commentHR
    s += self.commentSuffix
    s += self.endLine
    return s

  ##############################################################################
  # Init Kernel
  ##############################################################################
  @abc.abstractmethod
  def initKernel(self, kernel, tensorParametersA, tensorParametersB ):

    self.staggerU = kernel["StaggerU"]
    if self.staggerU:
      assert (kernel["KernelLanguage"]=="Source" or kernel["BufferLoad"])
    self.tPA = tensorParametersA
    self.tPB = tensorParametersB

    # Only assembly supports scheduling
    self.canSchedule = (kernel["KernelLanguage"] == "Assembly")

    if self.canSchedule:
      self.scheduleGlobalRead = kernel["ScheduleGlobalRead"] \
          and kernel["PrefetchGlobalRead"]
    else:
      self.scheduleGlobalRead = 0

    if self.canSchedule:
      self.scheduleLocalWrite = kernel["ScheduleLocalWrite"] \
          and kernel["PrefetchGlobalRead"]
    else:
      self.scheduleLocalWrite = 0

    if self.canSchedule:
      self.scheduleIterAlg = kernel["ScheduleIterAlg"]
    else:
      self.scheduleIterAlg = 0

    self.prefetchAcrossPersistent = kernel["PrefetchAcrossPersistent"]

    self.storeCInUnroll = kernel["StoreCInUnroll"]

    # NoTailLoop optimization
    # Case 1 (NoTailLoop = 1): just remove TailLoop
    #   -  ASEM%GSU=0 and ASEM/GSU is multiple of DepthU. TailLoop code will not be used in this case.
    # Case 2(NoTailLoop = 2): generate TailLoop code in NoLoadLoop (last loop code) and remove TailLoop
    #      all of the following conditions should be true
    #   - BufferLoad = True
    #   - SuppressNoLoadLoop = False
    #   - MatrixInstruction + MatrixInstK > 1
    #   - global read width for TailLoop decided by assert is multiple of GlobalLoadVectorWidthA/B
    #     (this is necessary to use prefetch global read fot tail loop without out of range access at the edge)
    #   - GlobalSplitU = 1
    #     GSU>1 case, remaining K is distributed unevenly and does not work with tailLoop in noLoadLoop
    #   - PersistentKernel = 0
    #   - StreamK = 0
    #   - DepthULdsDivisor = 1
    #   - StaggerU = 0
    #     StaggerU=0 case, we can exit NoLoadLoop earlier when whole K range is processed
    #   - InnerUnroll = 1
    #     K mask part does not work properly with InnerUnroll>1
    # Case 3 (NoTailLoop = 3): Case 2 + StaggerU != 0 + (NT + BufferLoad (except for DirectToLds)
    #   - StaggerU>0 and NT(+BufferLoad)
    #     if StaggerU>0, the partial K part can be in unroll and K mask cannot be handled in NoLoadLoop
    #     If NT and BufferLoad, global load for out of range K is always 0 because out of range K address
    #     (This is not true in DirectToLds case)
    #     is always out of array load (means load 0)
    #     If StaggerU is enabled, cannot exit unless whole code in NoLoadLoop is done
    #
    # Reject the following cases if noTailLoop is not enabled
    #  - PrefetchAcrossPersistent and PrefetchAcrossPersistentMode
    #    PrefetchAcrossPersistentMode does not support TailLoop (TLU is necessary for NoTailLoop)
    #  - DirectToLds + TLU + NumLoadsCoalesced > 1 (special local read offset conversion is not implemented in tail loop code)
    #  - DirectToLds + LRVW > 1

    # global load width for tail loop (based on AssertFree0, 1 or AssertSummationElementMultiple)
    asem = kernel["AssertSummationElementMultiple"]
    # need to adjust asem for GSU
    gsu = kernel["GlobalSplitU"]
    asemDivGSU = 1 if asem%gsu !=0 else asem//gsu
    # A
    tluA = kernel["ProblemType"]["TLUA"]
    glvwA = kernel["GlobalLoadVectorWidthA"]
    afem = kernel["AssertFree0ElementMultiple"]
    tailLoopLoadWidthA = afem if tluA else asem
    # B
    tluB = kernel["ProblemType"]["TLUB"]
    glvwB = kernel["GlobalLoadVectorWidthB"]
    afem = kernel["AssertFree1ElementMultiple"]
    tailLoopLoadWidthB = afem if tluB else asem
    # if glvw is not power of 2, use 1
    if glvwA <= 1 or (glvwA & (glvwA - 1)):
      tailLoopLoadWidthA = 1
    if glvwB <= 1 or (glvwB & (glvwB - 1)):
      tailLoopLoadWidthB = 1

    noTailLoop = 0
    if (asemDivGSU % kernel["DepthU"] == 0):
      noTailLoop = 1
    elif kernel["BufferLoad"] and (not kernel["SuppressNoLoadLoop"]) and \
         kernel["EnableMatrixInstruction"] and kernel["MatrixInstK"] > 1 and \
         (glvwA <= 1 or (tailLoopLoadWidthA % glvwA == 0)) and (glvwB <= 1 or (tailLoopLoadWidthB % glvwB == 0)) and \
         gsu == 1 and kernel["PersistentKernel"] == 0 and kernel["StreamK"] == 0 and kernel["DepthULdsDivisor"] == 1 and \
         kernel["InnerUnroll"] == 1:
      if kernel["StaggerU"] == 0:
        noTailLoop = 2
      elif (tluA and (not kernel["DirectToLdsA"]) and tluB and (not kernel["DirectToLdsB"])):
        noTailLoop = 3

    # no tail loop optimization setting
    # noTailLoop=1: remove TailLoop
    # noTailLoop=2: remove TailLoop and generate TailLoop in NoLoadLoop with early exit
    # noTailLoop=3: remove TailLoop and generate TailLoop in NoLoadLoop without early exit
    self.noTailLoop = noTailLoop > 0
    self.tailLoopInNLL = noTailLoop >= 2
    self.noEarlyExitForTailLoopInNLL = noTailLoop == 3

    # DTV odd check
    # need DTV odd code if (asem//GSU) is not multiple of DepthU*2
    self.generateDTVodd = False
    if ( kernel["DirectToVgprA"] or  kernel["DirectToVgprB"]) and (asemDivGSU % (kernel["DepthU"] * 2) != 0):
      self.generateDTVodd = True

    self.actualSummationLoops = 1 if kernel["PackSummationDims"] else kernel["ProblemType"]["NumIndicesSummation"]
    self.otherSummationLoops  = self.actualSummationLoops-1
    self.otherSummations      = kernel["ProblemType"]["NumIndicesSummation"]-1 # not loops but summations vars

    # If 0, unroll loop is decremented by 1 each iteration and scaled by DEPTHU when number of summation elements
    # is required.
    # if 1, unroll loop starts at 0 and increments by DEPTHU.  No scaling is required.  This mode is required
    # for pack summation dims, but can also be used independently and this is useful for isolation and testing.
    self.unrollIncIsDepthU = kernel["UnrollIncIsDepthU"] or kernel["PackSummationDims"] \
                             or bool(kernel["ProblemType"]["ZeroPadA"]) or bool(kernel["ProblemType"]["ZeroPadB"])

    # turn on parts of prefetchAcrossPersistent code for testing
    self.prefetchAcrossPersistent0 = 0 or self.prefetchAcrossPersistent
    self.canOptimizePreLoopLWVmcnt = kernel["OptPreLoopVmcnt"]

    self.enable = {}
    dkp = kernel["DisableKernelPieces"]
    # Can locally overrid these by changing True to False or
    # use the DisableKernelPieces for a quick search (see Common.py)
    self.enable["PreLoop"]        = True and not (dkp>0 and dkp >= 7) and not dkp == -7
    self.enable["GlobalRead"]     = True and not (dkp>0 and dkp >= 2) and not dkp == -2
    self.enable["GlobalReadInc"]  = True and not (dkp>0 and dkp >= 7) and not dkp == -7
    self.enable["LocalWrite"]     = True and not (dkp>0 and dkp >= 3) and not dkp == -3
    self.enable["LocalRead"]      = True and not (dkp>0 and dkp >= 4) and not dkp == -4
    self.enable["Wait"]           = True and not (dkp>0 and dkp >= 5) and not dkp == -5
    self.enable["Sync"]           = True and not (dkp>0 and dkp >= 5) and not dkp == -5
    self.enable["MAC"]            = True and not (dkp>0 and dkp >= 6) and not dkp == -6
    self.enable["PostLoop"]       = True and not (dkp>0 and dkp >= 1) and not dkp == -1
    self.enable["InvalidLocalReadA"] = dkp == -10 or dkp == -12
    self.enable["InvalidLocalReadB"] = dkp == -11 or dkp == -12
    self.enable["InvalidLocalWriteA"] = dkp == -13 or dkp == -15
    self.enable["InvalidLocalWriteB"] = dkp == -14 or dkp == -15
    self.enable["InvalidGlobalReadA"] = (dkp == -16 or dkp == -18) and kernel["BufferLoad"]
    self.enable["InvalidGlobalReadB"] = (dkp == -17 or dkp == -18) and kernel["BufferLoad"]

    #if dkp:
    #  print "\nKernelWriter enable:", self.enable

    if kernel["KernelLanguage"] == "Source":
      self.language = globalParameters["RuntimeLanguage"]
    else:
      self.language = "ASM"
    self.indexChars = []
    for i in range(0, len(globalParameters["IndexChars"])):
      self.indexChars.append(globalParameters["IndexChars"][i])
    self.indexChars[kernel["ProblemType"]["Index0"]] \
        = "0" + self.indexChars[kernel["ProblemType"]["Index0"]]
    self.indexChars[kernel["ProblemType"]["Index1"]] \
        = "1" + self.indexChars[kernel["ProblemType"]["Index1"]]
    self.unrollIdx = kernel["ProblemType"]["NumIndicesSummation"]-1
    self.unrollChar = \
        self.indexChars[kernel["ProblemType"]["IndicesSummation"][\
        self.unrollIdx]]
    self.tileChar0 = self.indexChars[kernel["ProblemType"]["Index0"]]
    self.tileChar1 = self.indexChars[kernel["ProblemType"]["Index1"]]
    self.tileCharA = self.tileChar0 if (kernel["ProblemType"]["Tensor0"]==0) \
        else self.tileChar1
    self.tileCharB = self.tileChar0 if (kernel["ProblemType"]["Tensor0"]==1) \
        else self.tileChar1

    """
    if kernel["ProblemType"]["Tensor0"]==0:
      kernel["ThreadTileA"] = kernel["ThreadTile0"]
      kernel["ThreadTileB"] = kernel["ThreadTile1"]
      kernel["SubGroupA"] = kernel["SubGroup0"]
      kernel["SubGroupB"] = kernel["SubGroup1"]
      kernel["MacroTileA"] = kernel["MacroTile0"]
      kernel["MacroTileB"] = kernel["MacroTile1"]
    else:
      kernel["ThreadTileB"] = kernel["ThreadTile0"]
      kernel["ThreadTileA"] = kernel["ThreadTile1"]
      kernel["SubGroupB"] = kernel["SubGroup0"]
      kernel["SubGroupA"] = kernel["SubGroup1"]
      kernel["MacroTileB"] = kernel["MacroTile0"]
      kernel["MacroTileA"] = kernel["MacroTile1"]
    """

    ########################################
    # derrive global-read-coalesce-group from local in config
    """
    if kernel["ProblemType"]["TLUA"]:
      self.globalReadCoalesceGroupA = kernel["LocalWriteCoalesceGroupA"]
    else:
      self.globalReadCoalesceGroupA = not kernel["LocalWriteCoalesceGroupA"]
    if kernel["ProblemType"]["TLUB"]:
      self.globalReadCoalesceGroupB = kernel["LocalWriteCoalesceGroupB"]
    else:
      self.globalReadCoalesceGroupB = not kernel["LocalWriteCoalesceGroupB"]
    """
    self.globalReadCoalesceGroupA = kernel["GlobalReadCoalesceGroupA"]
    self.globalReadCoalesceGroupB = kernel["GlobalReadCoalesceGroupB"]
    """
    # original parameters
    NumLoadsCoalesced -> NumLoadsPerpendicular
    # new intermediate parameters
    numReadsTile # nrt
    numReadsUnroll # nru
    numReadsTileVecComp # nrvt
    numReadsUnrollVecComp # nrvu
    numWritesCoal # nwc
    numWritesPerp # nwp
    numWritesCoalVecComp # nwvc
    numWritesPerpVecComp # nwvp
    readTileComponents (based on grcv)
    readTileVector
    """

    # TODO load sub-vector
    vwa = kernel["GlobalLoadVectorWidthA"]
    vwb = kernel["GlobalLoadVectorWidthB"]

    # lrvwTileA,B
    # lrvwTileA,B > 1 is to use wider local read + v_perm
    # MIInputPerThread > 1 case, we need MIInputPerThread continuous K elements
    # lrvwTileA,B should not exceed MIInputPerThread
    # = 2 means, load 2 continuous M or N and do swap
    #     (M,K) = (0,0) (1,0)
    #             (0,1) (1,1)
    #      to
    #     (M,K) = (0,0) (0,1)
    #             (1,0) (1,1)
    # = 4 means, load 4 continuous M or N and do swap
    #     (M,K) = (0,0) (1,0) (2,0) (3,0)
    #             (0,1) (1,1) (2,1) (3,1)
    #             (0,2) (1,2) (2,2) (3,2)
    #             (0,3) (1,3) (2,3) (3,3)
    #      to
    #     (M,K) = (0,0) (0,1) (0,2) (0,3)
    #             (1,0) (1,1) (1,2) (1,3)
    #             (2,0) (2,1) (2,2) (2,3)
    #             (3,0) (3,1) (3,2) (3,3)
    # it requires
    #   EnableMatrixInstruction + MIInputPerThread > 1
    #   SourceSwap only (TODO: non SourceSwap)
    #   VgprForLocalReadPacking  (need dedicated vpgr for packing)
    #   ClusterLocalRead
    #   not UnrollMajorLDS
    #   VectorWidthA,B > 1
    self.lrvwTileA = 1
    self.lrvwTileB = 1
    self.useWiderLocalReadB = False
    if kernel["EnableMatrixInstruction"] and kernel["MIInputPerThread"] > 1 and \
       kernel["VgprForLocalReadPacking"] and kernel["ClusterLocalRead"]:
      if (not kernel["UnrollMajorLDSA"]):
        self.lrvwTileA = min(kernel["MIInputPerThread"], kernel["VectorWidthA"]) # should not exceed MIInputPerThread
      if (not kernel["UnrollMajorLDSB"]):
        self.lrvwTileB = min(kernel["MIInputPerThread"], kernel["VectorWidthB"]) # should not exceed MIInputPerThread
    # DirectToVgpr + pack (v_perm)
    self.packDTVA = kernel["DirectToVgprA"] and self.lrvwTileA > 1
    self.packDTVB = kernel["DirectToVgprB"] and self.lrvwTileB > 1

    self.numItersPLR = kernel["PrefetchLocalRead"]%kernel["LoopIters"]
    self.numVgprBuffer = kernel["LoopIters"] if kernel["PrefetchLocalRead"] > kernel["LoopIters"] else kernel["PrefetchLocalRead"]
    if kernel["UnrollMajorLDSA"]:
      self.lrvwA = kernel["LocalReadVectorWidth"]
    else:
      if kernel["EnableMatrixInstruction"]:
        # MI + UMLDS, we need minimum of MIInputPerThread for lrvw
        self.lrvwA = kernel["MIInputPerThread"]
        if kernel["DirectToVgprA"]:
          # DirectToVgprA case, ignore LocalReadVectorWidth and use GlobalLoadVectorWidth instead.
          self.lrvwA = vwa
      else:
        self.lrvwA = 1
    if kernel["UnrollMajorLDSB"]:
      self.lrvwB = kernel["LocalReadVectorWidth"]
    else:
      if kernel["EnableMatrixInstruction"]:
        # MI + UMLDS, we need minimum of MIInputPerThread for lrvw
        self.lrvwB = kernel["MIInputPerThread"]
        if kernel["DirectToVgprB"]:
          # DirectToVgprB case, ignore LocalReadVectorWidth and use GlobalLoadVectorWidth instead.
          self.lrvwB = vwb
        elif kernel["DirectToVgprA"] and kernel["ProblemType"]["TLUA"]:
          # MI + UMLDS, we need minimum of MIInputPerThread for lrvw
          # DirectToVgprA + TLUA + UnrollMajorLDSB=False case, allow wider LocalReadVectorWidth
          self.lrvwB = kernel["LocalReadVectorWidth"]
          self.useWiderLocalReadB = self.lrvwB > kernel["MIInputPerThread"]
      else:
        self.lrvwB = 1

    # Wider LocalRead
    if kernel["EnableMatrixInstruction"]:
      self.numReadsIterCoalescedA = ceil(self.lrvwA / kernel["MIInputPerThread"]) if kernel["UnrollMajorLDSA"] else 1
      self.numReadsIterCoalescedB = ceil(self.lrvwB / kernel["MIInputPerThread"]) if kernel["UnrollMajorLDSB"] else 1
    else:
      self.numReadsIterCoalescedA  = 1
      self.numReadsIterCoalescedB  = 1
    self.numIterPerCoalescedReadA = max(1,self.numReadsIterCoalescedA//kernel["InnerUnroll"])
    self.numIterPerCoalescedReadB = max(1,self.numReadsIterCoalescedB//kernel["InnerUnroll"])

    if kernel["ScheduleIterAlg"] == 3 or kernel["ScheduleIterAlg"] == 2:
      self.numMfmaPerIter = kernel["MIWaveTile"][0] * kernel["MIWaveTile"][1] * kernel["InnerUnroll"]
      if kernel["ProblemType"]["DataType"].isComplex(): self.numMfmaPerIter *= 4

    ########################################
    # read vectors or vector components
    ########################################
    if kernel["ProblemType"]["TLUA"]: # NT no transpose
      self.numReadsTileA = kernel["NumLoadsCoalescedA"]
      self.numReadsUnrollA = kernel["NumLoadsPerpendicularA"]
      if kernel["GlobalReadCoalesceVectorA"]: # read vectors
        self.readTileDimComponentsA = False # Vector
        self.readTileDimVectorA = True # Vector
        self.readUnrollDimComponentsA = False # Scalar
        self.readUnrollDimVectorA = False # Scalar
        self.numReadsTileVecCompA = vwa
        self.numReadsUnrollVecCompA = 1
      else: # read components, write components
        self.readTileDimComponentsA = False # Scalar
        self.readTileDimVectorA = False # Scalar
        self.readUnrollDimComponentsA = kernel["VectorWidthA"] > 1 # Components
        self.readUnrollDimVectorA = False # Components
        self.numReadsTileVecCompA = 1
        self.numReadsUnrollVecCompA = vwa
    else: # TN yes transpose
      self.numReadsTileA = kernel["NumLoadsPerpendicularA"]
      self.numReadsUnrollA = kernel["NumLoadsCoalescedA"]
      if kernel["GlobalReadCoalesceVectorA"]: # read vector
        self.readTileDimComponentsA = False # Scalar
        self.readTileDimVectorA = False # Scalar
        self.readUnrollDimComponentsA = False # Vector
        self.readUnrollDimVectorA = True # Vector
        self.numReadsUnrollVecCompA = vwa
        self.numReadsTileVecCompA = 1
      else: # read components, write vectors
        self.readTileDimComponentsA = kernel["VectorWidthA"] > 1 # Components
        self.readTileDimVectorA = False # Components
        self.readUnrollDimComponentsA = False # Scalar
        self.readUnrollDimVectorA = False # Scalar
        # NEW
        self.numReadsUnrollVecCompA = 1
        self.numReadsTileVecCompA = vwa

    ########################################
    # write vectors or vector components
    ########################################
    if kernel["ProblemType"]["TLUA"] != kernel["UnrollMajorLDSA"]: # NT no transpose
      self.numWritesCoalA = kernel["NumLoadsCoalescedA"]
      if kernel["GlobalReadCoalesceVectorA"]: # read vectors, write vectors
        self.writeUnrollDimComponentsA = False # Scalar
        if kernel["LocalDotLayout"]>1:
          self.writeTileDimComponentsA = kernel["GlobalLoadVectorWidthA"] > 1 # Components
          writeCoal = False
        else:
          self.writeTileDimComponentsA = False # Vector
          writeCoal = True
      else: # read components, write components
        self.writeTileDimComponentsA = False # Scalar
        self.writeUnrollDimComponentsA = kernel["GlobalLoadVectorWidthA"] > 1 # Components
        writeCoal = False
    else: # TN yes transpose
      self.numWritesCoalA = kernel["NumLoadsPerpendicularA"]
      if kernel["GlobalReadCoalesceVectorA"]: # read vector, write components
        self.writeUnrollDimComponentsA = False # Scalar
        if kernel["LocalDotLayout"]>1:
          self.writeTileDimComponentsA = kernel["GlobalLoadVectorWidthA"] > 1 # Components
          # LDS writes with LDL>1 will never be coalesced
          writeCoal = False
        else:
          self.writeTileDimComponentsA = kernel["GlobalLoadVectorWidthA"] > 1 # Components
          writeCoal = False
      else: # read components, write vectors
        self.writeTileDimComponentsA = False # Vector
        self.writeUnrollDimComponentsA = False # Scalar
        writeCoal = True

    # writeCoal indicates writes should be done in the coal dim
    # else in perp
    if writeCoal:
      self.numWritesCoalVecCompA = vwa / kernel["DepthULdsDivisor"] if vwa < 1 else vwa // kernel["DepthULdsDivisor"]
      self.numWritesPerpVecCompA = 1
    else:
      self.numWritesCoalVecCompA = 1
      self.numWritesPerpVecCompA = vwa
    del writeCoal

    self.numReadVectorComponentsA = kernel["GlobalLoadVectorWidthA"] \
        if (self.readTileDimComponentsA \
        or self.readUnrollDimComponentsA) else 1
    # self.numWriteVectorComponentsA = kernel["GlobalLoadVectorWidthA"] \
    #     if (self.writeTileDimComponentsA \
    #     or self.writeUnrollDimComponentsA) else 1
    # self.numReadTileVectorComponentsA = kernel["GlobalLoadVectorWidthA"] \
    #     if self.readTileDimComponentsA else 1 # for branches
    # convert tile/unroll to para/perp
    if kernel["ProblemType"]["TLUA"]:
      self.numReadsCoalVecCompA = self.numReadsTileVecCompA
      self.numReadsPerpVecCompA = self.numReadsUnrollVecCompA
      # for asm
      self.readCoalescedComponentsA  = self.readTileDimComponentsA
      # self.readCoalescedVectorA      = self.readTileDimVectorA  # Not Used
      self.readPerpendicularComponentsA  = self.readUnrollDimComponentsA
      # self.readPerpendicularVectorA      = self.readUnrollDimVectorA  # Not Used
    else:
      self.numReadsCoalVecCompA = self.numReadsUnrollVecCompA
      self.numReadsPerpVecCompA = self.numReadsTileVecCompA
      # for asm
      self.readCoalescedComponentsA  = self.readUnrollDimComponentsA
      # self.readCoalescedVectorA      = self.readUnrollDimVectorA  # Not Used
      self.readPerpendicularComponentsA  = self.readTileDimComponentsA
      # self.readPerpendicularVectorA      = self.readTileDimVectorA  # Not Used

    ####################################
    # read vectors or vector components b
    ####################################
    if kernel["ProblemType"]["TLUB"]: # NT no transpose
      self.numReadsTileB = kernel["NumLoadsCoalescedB"]
      self.numReadsUnrollB = kernel["NumLoadsPerpendicularB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.readTileDimComponentsB = False # Vector
        self.readTileDimVectorB = True # Vector
        self.readUnrollDimComponentsB = False # Scalar
        self.readUnrollDimVectorB = False # Scalar
        self.numReadsTileVecCompB = vwb
        self.numReadsUnrollVecCompB = 1
      else:
        self.readTileDimComponentsB = False # Scalar
        self.readTileDimVectorB = False # Scalar
        self.readUnrollDimComponentsB = kernel["VectorWidthB"] > 1 # Components
        self.readUnrollDimVectorB = False # Components
        # NEW
        self.numReadsTileVecCompB = 1
        self.numReadsUnrollVecCompB = vwb
    else: # TN yes transpose
      self.numReadsTileB = kernel["NumLoadsPerpendicularB"]
      self.numReadsUnrollB = kernel["NumLoadsCoalescedB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.readTileDimComponentsB = False # Scalar
        self.readTileDimVectorB = False # Scalar
        self.readUnrollDimComponentsB = False # Vector
        self.readUnrollDimVectorB = True # Vector
        self.numReadsUnrollVecCompB = vwb
        self.numReadsTileVecCompB = 1
      else:
        self.readTileDimComponentsB = kernel["VectorWidthB"] > 1 # Components
        self.readTileDimVectorB = False # Components
        self.readUnrollDimComponentsB = False # Scalar
        self.readUnrollDimVectorB = False # Scalar
        # NEW
        self.numReadsUnrollVecCompB = 1
        self.numReadsTileVecCompB = vwb

    ####################################
    # write vectors or vector components b
    ####################################
    if kernel["ProblemType"]["TLUB"] != kernel["UnrollMajorLDSB"]: # NT no transpose
      self.numWritesCoalB = kernel["NumLoadsCoalescedB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.writeUnrollDimComponentsB = False # Vector
        if kernel["LocalDotLayout"]>1:
          self.writeTileDimComponentsB = kernel["GlobalLoadVectorWidthB"] > 1 # Components
          writeCoal = False
        else:
          self.writeTileDimComponentsB = False # Vector
          writeCoal = True
      else:
        self.writeTileDimComponentsB = False # Scalar
        self.writeUnrollDimComponentsB = kernel["GlobalLoadVectorWidthB"] > 1 # Components
        # NEW
        self.numWritesCoalVecCompB = 1
        self.numWritesPerpVecCompB = vwb
    else: # TN yes transpose
      self.numWritesCoalB = kernel["NumLoadsPerpendicularB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        self.writeUnrollDimComponentsB = False
        if kernel["LocalDotLayout"]>1:
          self.writeTileDimComponentsB = kernel["GlobalLoadVectorWidthB"] > 1 # Components
          # LDS writes with LDL>1 will never be coalesced
          writeCoal = False
        else:
          self.writeTileDimComponentsB = kernel["GlobalLoadVectorWidthB"] > 1 # Components
          writeCoal = False
      else:
        self.writeTileDimComponentsB = False # Vector
        self.writeUnrollDimComponentsB = False # Scalar
        # NEW
        self.numWritesCoalVecCompB = vwb
        self.numWritesPerpVecCompB = 1

    # writeCoal indicates writes should be done in the coal dim
    # else in perp
    if writeCoal:
      self.numWritesCoalVecCompB = vwb / kernel["DepthULdsDivisor"] if vwb < 1 else vwb // kernel["DepthULdsDivisor"]
      self.numWritesPerpVecCompB = 1
    else:
      self.numWritesCoalVecCompB = 1
      self.numWritesPerpVecCompB = vwb
    del writeCoal

    # numReadVectorComponentsB is refers to global reads
    self.numReadVectorComponentsB = kernel["GlobalLoadVectorWidthB"] \
        if (self.readTileDimComponentsB \
        or self.readUnrollDimComponentsB) else 1
    # self.numWriteVectorComponentsB = kernel["GlobalLoadVectorWidthB"] \
    #     if (self.writeTileDimComponentsB \
    #     or self.writeUnrollDimComponentsB) else 1
    # self.numReadTileVectorComponentsB = kernel["GlobalLoadVectorWidthB"] \
    #     if self.readTileDimComponentsB else 1 # for branches
    # convert tile/unroll to para/perp
    if kernel["ProblemType"]["TLUB"]:
      self.numReadsCoalVecCompB = self.numReadsTileVecCompB
      self.numReadsPerpVecCompB = self.numReadsUnrollVecCompB
      # for asm
      self.readCoalescedComponentsB  = self.readTileDimComponentsB
      # self.readCoalescedVectorB      = self.readTileDimVectorB  # Not Used
      self.readPerpendicularComponentsB  = self.readUnrollDimComponentsB
      # self.readPerpendicularVectorB      = self.readUnrollDimVectorB  # Not Used
    else:
      self.numReadsCoalVecCompB = self.numReadsUnrollVecCompB
      self.numReadsPerpVecCompB = self.numReadsTileVecCompB
      # for asm
      self.readCoalescedComponentsB  = self.readUnrollDimComponentsB
      # self.readCoalescedVectorB      = self.readUnrollDimVectorB  # Not Used
      self.readPerpendicularComponentsB  = self.readTileDimComponentsB
      # self.readPerpendicularVectorB      = self.readTileDimVectorB  # Not Used

    ####################################
    # load sizes
    """
    if kernel["ProblemType"]["TLUA"]:
      kernel["LSCA"] = kernel["MacroTileA"] \
          / kernel["NumLoadsCoalescedA"]
      kernel["LSPA"] = kernel["DepthU"] / kernel["NumLoadsPerpendicularA"]
    else:
      kernel["LSCA"] = kernel["DepthU"] / kernel["NumLoadsCoalescedA"]
      kernel["LSPA"] = kernel["MacroTileA"] \
          / kernel["NumLoadsPerpendicularA"]
    if kernel["ProblemType"]["TLUB"]:
      kernel["LSCB"] = kernel["MacroTileB"] \
          / kernel["NumLoadsCoalescedB"]
      kernel["LSPB"] = kernel["DepthU"] / kernel["NumLoadsPerpendicularB"]
    else:
      kernel["LSCB"] = kernel["DepthU"] / kernel["NumLoadsCoalescedB"]
      kernel["LSPB"] = kernel["MacroTileB"] \
          / kernel["NumLoadsPerpendicularB"]
    kernel["LVCA"] = kernel["LSCA"] / kernel["GlobalLoadVectorWidthA"]
    kernel["LVCB"] = kernel["LSCB"] / kernel["GlobalLoadVectorWidthB"]
    kernel["LVPA"] = kernel["LSPA"] / kernel["GlobalLoadVectorWidthA"]
    kernel["LVPB"] = kernel["LSPB"] / kernel["GlobalLoadVectorWidthB"]
    """

    self.getTensorParameters(tensorParametersA, kernel, True)
    self.getTensorParameters(tensorParametersB, kernel, False)

    tensorParametersA["PackBatchDims"] = kernel["PackBatchDims"] if kernel["PackBatchDims"] & 0x1 else 0
    tensorParametersB["PackBatchDims"] = kernel["PackBatchDims"] if kernel["PackBatchDims"] & 0x2 else 0
    tensorParametersA["PackedIndices"] = kernel["PackedC%uIndicesX"%self.tPA["tile01Idx"]]
    tensorParametersB["PackedIndices"] = kernel["PackedC%uIndicesX"%self.tPB["tile01Idx"]]

    # condition(s) to enable init accvgpr opt (use const "0" as an operand instead of initializing whole accvgpr)
    self.useInitAccVgprOpt = False
    # enable for the following conditions
    if kernel["EnableMatrixInstruction"] and (kernel["PrefetchGlobalRead"] == 1 or kernel["PrefetchGlobalRead"] == 2) \
       and globalParameters["AsmCaps"][globalParameters["CurrentISA"]]["HasMFMA_constSrc"] \
       and kernel["StreamK"] == 0:
      self.useInitAccVgprOpt = True
    # force to disable for the following conditions
    if self.useInitAccVgprOpt:
      asgt3 = 0 if not (3 in kernel["AssertSizeGreaterThan"].keys()) else kernel["AssertSizeGreaterThan"][3]
      gsu = kernel["GlobalSplitU"]
      # PGR=1 case, K > DepthU * 1 is necessary ( if not noTailLoop, need > DepthU * 2)
      # (kernel["AssertSizeGreaterThan"][3] > DepthU * GSU * 1 (or 2)
      # PGR=2 case, K > DepthU * 2 is necessary ( if not noTailLoop, need > DepthU * 3)
      # (kernel["AssertSizeGreaterThan"][3] > DepthU * GSU * 2 (or 3)
      minDUnum = kernel["PrefetchGlobalRead"]
      if not self.noTailLoop:
        minDUnum += 1
      if not (asgt3 >= kernel["DepthU"] * gsu * minDUnum):
        tPrint(3, "InitAccVgprOpt is disabled because AssertSizeGreaterThan for K is not greater than DepthU * GSU * %u"%minDUnum)
        self.useInitAccVgprOpt = False

    # condition(s) to enable singleNLL opt
    self.enableSingleNLLOpt = False
    if self.noTailLoop and not (self.prefetchAcrossPersistent and kernel["PrefetchAcrossPersistentMode"] == 0):
      if kernel["StoreCInUnroll"]:
        self.enableSingleNLLOpt = True
      # so far, not enabled for DirectToVgpr
      # Performance is better with Tensile, but does not perform better with HPL
      #if kernel["DirectToVgprA"] or kernel["DirectToVgprB"]:
      #  self.enableSingleNLLOpt = True

    # condition(s) for swapMfmaInnerLoop
    # swap inner and outer loop for mfmaIter
    self.swapMfmaInnerLoop = False
    if kernel["EnableMatrixInstruction"]:
      if kernel["ProblemType"]["DataType"].isComplex() and tensorParametersB["tile01Idx"]:
        # complex case, swap inner loop and outer loop so that idxA comes outer
        # this is to re-use same tmp vgpr to nagate ai or ar
        self.swapMfmaInnerLoop = True
      if (not kernel["ProblemType"]["DataType"].isComplex()) and kernel["DirectToVgprA"] and kernel["PrefetchGlobalRead"] == 2:
        # not Complex and DTVA + PGR2 case, use B for inner loop to schedule more mfma between DTVA global read instructions
        # exception: both A and B pack code case
        needPackAB = kernel["EnableMatrixInstruction"] and kernel["MIInputPerThread"] > 1 and \
                   ((not kernel["UnrollMajorLDSA"]) and (not kernel["UnrollMajorLDSB"]))
        if not needPackAB:
          self.swapMfmaInnerLoop = True

    # init code optimization
    # generate local read/write address code and global read tile offset code before wait for kernel arg load (if applicable)
    # set False here for source kernel (enable only for Asm)
    self.isInitCodeOptLR = False
    self.isInitCodeOptLW = False

  @staticmethod
  def zpForSumIdx(sumIdx, zeroPad):
     """ Returns zero-pad for specified sumIdx if it matches or None if not """
     return next((zpi for zpi in zeroPad if zpi[1] == sumIdx), None)


  ##############################################################################
  # Function Prefix
  ##############################################################################
  @abc.abstractmethod
  def functionPrefix(self, kernel):
    return ""

  ##############################################################################
  # Function Signature Prefix
  ##############################################################################
  @abc.abstractmethod
  def functionSignaturePrefix(self, kernel):
    return ""

  ##############################################################################
  # Function Signature
  ##############################################################################
  @abc.abstractmethod
  def functionSignature(self, kernel ):
    return ""

  ##############################################################################
  # Function Signature Suffix
  ##############################################################################
  @abc.abstractmethod
  def functionSignatureSuffix(self, kernel):
    return ""

  ##############################################################################
  # Function Begin
  ##############################################################################
  @abc.abstractmethod
  def functionBegin(self, kernel):
    return ""

  ##############################################################################
  # Allocate Resources
  ##############################################################################
  @abc.abstractmethod
  def allocateResources(self, kernel, lraCode=None):
    return ""


  ##############################################################################
  # Open Persistent Loop
  # init iteration counter, define loop target
  ##############################################################################
  @abc.abstractmethod
  def openPersistentLoop(self, kernel):
    return ""


  ##############################################################################
  # Global Read Addresses: Work-Group
  ##############################################################################
  @abc.abstractmethod
  def graWorkGroup(self, kernel, isPap):
    return ""

  ##############################################################################
  # Get Params For Tensor A/B
  ##############################################################################
  def getTensorParameters(self, tP, kernel, tA):
    tP["mirror"] = bool(kernel["ProblemType"]["MirrorDims%s" % ("A" if tA else "B")])
    if tA: # A
      tP["isA"] = True                                      # is this tensor A
      tP["isB"] = False                                     # is this tensor B
      tP["bpe"] = int(4*kernel["ProblemType"]["DataType"].numRegisters())
      tP["tensorChar"] = "A"                                # tensor character A/B
      tP["tensorIdx"] = 0                                   # tensor index A=0, B=1
      tP["tileChar"] = self.tileCharA                       # tile char I0 or J1
      tP["tileIdx"] = kernel["ProblemType"]["Index01A"]     # is the tile dimension of A the 0th or 1th index, i.e. Aki, tileIdx=0
      tP["tile01Idx"] = 1 if tP["tileIdx"] else 0
      tP["lsc"] = "LSCA"                                    # load size coalesced A, number of elements that get loaded along coalesced dimension with each load
      tP["lsp"] = "LSPA"                                    # load size perpendicular A, number of elements that get loaded along non-coalesced dimension with each load
      tP["lvc"] = "LVCA"                                    # "load size" in terms of number of short-vectors and not elements
      tP["lvp"] = "LVPA"                                    # "load size" in terms of number of short-vectors and not elements
      tP["rtv"] = self.readTileDimVectorA                   # bool in the tile dimension, reads will read vectors
      tP["rtc"] = self.readTileDimComponentsA               # bool in the tile dimension, reads will read vector components
      #tP["ruv"] = self.readUnrollDimVectorA
      #tP["nlvc"] = self.numReadVectorComponentsA
      #tP["nwvc"] = self.numWriteVectorComponentsA
      tP["wg"] = "WorkGroup%u" % (tP["tile01Idx"])# these are storing the actual strong to lookup the number from kernel dictionary
      tP["prevWg"] = "PrevWorkGroup0"                       # used for prefetch-across-persistent #NHWC TO-do
      tP["sg"] = "SubGroup%u" % (tP["tile01Idx"])
      tP["tt"] = "ThreadTile%u" % (tP["tile01Idx"])
      tP["mt"] = "MacroTile%u" % (tP["tile01Idx"])
      tP["grcg"] = self.globalReadCoalesceGroupA            # global reads are coalesced along threads
      tP["grcv"] = kernel["GlobalReadCoalesceVectorA"]      # global reads are vector reads, and lds writes will be components if transposing
      tP["tlu"] = kernel["ProblemType"]["TLUA"]             # thread stride is less than unroll stride, i.e., not transposing matrix
      tP["ia"] = kernel["ProblemType"]["IndexAssignmentsA"] # array of index assignments
      #tP["nlc"] = kernel["NumLoadsCoalescedA"]
      #tP["nlp"] = kernel["NumLoadsPerpendicularA"]
      #tP["nlcv"] = self.numReadsCoalVecCompA
      tP["nlpv"] = self.numReadsPerpVecCompA                # num vector components perpendicular to coalesced; =1 or VW
      # NEW
      tP["nrt"] = self.numReadsTileA                        # number of reads along tile dimension
      tP["nrtv"] = self.numReadsTileVecCompA                # number of vector components along tile dimension; =1 or VW
      tP["nru"] = self.numReadsUnrollA                      # number of reads along unroll dimension
      tP["nruv"] = self.numReadsUnrollVecCompA              # number of vector components along unroll dimension; =1 or VW
      tP["nrc"] = kernel["NumLoadsCoalescedA"]              # number of reads along coalesced dimension
      tP["nrcv"] = self.numReadsCoalVecCompA                # number of vector components along coalesced dimension
      tP["nrp"] = kernel["NumLoadsPerpendicularA"]          # number of reads along perpendicular dimension
      tP["nrpv"] = self.numReadsPerpVecCompA                # number of vector components along perpendicular dimension
      tP["nwcv"] = self.numWritesCoalVecCompA               # number of vector component writes along coalesced dimension
      tP["nwpv"] = self.numWritesPerpVecCompA               # number of vector component writes along perpendicular dimension
      tP["glvw"] = kernel["GlobalLoadVectorWidthA"]
      # asm
      tP["rcc"] = self.readCoalescedComponentsA             # read vector components along coalesced dimensions
      # tP["rcv"] = self.readCoalescedVectorA                 # read vector along coalesced dimension
      tP["rpc"] = self.readPerpendicularComponentsA         # read vector components along perpendicular dimension
      # tP["rpv"] = self.readPerpendicularVectorA             # read vector along perpendicular dimension
      tP["ruc"] = self.readUnrollDimComponentsA             # read vector components along unroll dimension
      tP["wtc"] = self.writeTileDimComponentsA              # write vector components along tile dimension
      tP["wuc"] = self.writeUnrollDimComponentsA            # write vector components along unroll dimension
      tP["idx"] = kernel["ProblemType"]["Index0"]           # index 0 is tile dimension belonging to A. Note 'idx' may not be in tP['ia'].
      tP["rc"] = kernel["ProblemType"]["IndexAssignmentsA"][0] \
          in [kernel["ProblemType"]["Index01A"], \
          kernel["ProblemType"]["IndexUnroll"]]             # can read coalesced
      tP["NonTemporal"] = kernel["NonTemporalA"]            # non-temporal read type
    else: # B
      tP["isA"] = False
      tP["isB"] = True
      tP["bpe"] = int(4*kernel["ProblemType"]["DataType"].numRegisters())
      tP["tensorChar"] = "B"
      tP["tensorIdx"] = 1
      tP["tileChar"] = self.tileCharB
      tP["tileIdx"] = kernel["ProblemType"]["Index01B"]
      tP["tile01Idx"] = 1 if tP["tileIdx"] else 0
      tP["lsc"] = "LSCB"
      tP["lsp"] = "LSPB"
      tP["lvc"] = "LVCB"
      tP["lvp"] = "LVPB"
      tP["rtv"] = self.readTileDimVectorB
      tP["rtc"] = self.readTileDimComponentsB
      #tP["ruv"] = self.readUnrollDimVectorB
      #tP["nlvc"] = self.numReadVectorComponentsB
      #tP["nwvc"] = self.numWriteVectorComponentsB
      tP["wg"] = "WorkGroup%u" % (tP["tile01Idx"])
      tP["prevWg"] = "PrevWorkGroup1"
      tP["sg"] = "SubGroup%u" % (tP["tile01Idx"])
      tP["tt"] = "ThreadTile%u" % (tP["tile01Idx"])
      tP["mt"] = "MacroTile%u" % (tP["tile01Idx"])
      tP["grcg"] = self.globalReadCoalesceGroupB
      tP["grcv"] = kernel["GlobalReadCoalesceVectorB"]
      tP["tlu"] = kernel["ProblemType"]["TLUB"]
      tP["ia"] = kernel["ProblemType"]["IndexAssignmentsB"]
      # NEW
      tP["nrt"] = self.numReadsTileB
      tP["nrtv"] = self.numReadsTileVecCompB
      tP["nru"] = self.numReadsUnrollB
      tP["nruv"] = self.numReadsUnrollVecCompB
      tP["nrc"] = kernel["NumLoadsCoalescedB"]
      tP["nrcv"] = self.numReadsCoalVecCompB
      tP["nrp"] = kernel["NumLoadsPerpendicularB"]
      tP["nrpv"] = self.numReadsPerpVecCompB
      tP["nwcv"] = self.numWritesCoalVecCompB
      tP["nwpv"] = self.numWritesPerpVecCompB
      tP["glvw"] = kernel["GlobalLoadVectorWidthB"]
      # asm
      tP["rcc"] = self.readCoalescedComponentsB
      # tP["rcv"] = self.readCoalescedVectorB
      tP["rpc"] = self.readPerpendicularComponentsB
      # tP["rpv"] = self.readPerpendicularVectorB
      tP["ruc"] = self.readUnrollDimComponentsB
      tP["wtc"] = self.writeTileDimComponentsB
      tP["wuc"] = self.writeUnrollDimComponentsB
      tP["idx"] = kernel["ProblemType"]["Index1"]
      tP["rc"] = kernel["ProblemType"]["IndexAssignmentsB"][0] \
          in [kernel["ProblemType"]["Index01B"], \
          kernel["ProblemType"]["IndexUnroll"]] # can read coalesced
      tP["NonTemporal"] = kernel["NonTemporalB"]

  ##############################################################################
  # Global Read Addresses: Tile Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def graTileAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Unroll Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def graUnrollAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Other Free Assignments
  ##############################################################################
  @abc.abstractmethod
  def graOtherFreeAssignments(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments
  ##############################################################################
  @abc.abstractmethod
  def graOtherSummationAssignments(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Tile Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graTileOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graUnrollOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Branch A/B
  ##############################################################################
  @abc.abstractmethod
  def graBranch(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Shift A/B
  ##############################################################################
  @abc.abstractmethod
  def graShift(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def graFinalOffsets(self, kernel, tP, releaseResource=False):
    return ""

  ##############################################################################
  # Global Read Addresses: Addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def graAddresses(self, kernel, tP, isPap=False):
    return ""

  ##############################################################################
  # Global Read Addresses: Increments A/B
  # This function declares the increments
  ##############################################################################
  @abc.abstractmethod
  def graIncrements(self, kernel, loopIdx, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Tile Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaTileAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaUnrollAssignment(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Release tile related vgpr
  ##############################################################################
  @abc.abstractmethod
  def lwaReleaseTileVgpr(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: First Offset A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaFirstOffset(self, kernel, tP, uDu=0):
    return ""

  ##############################################################################
  # Local Write Addresses: Final Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaFinalOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Declare Addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def lwaDeclareAddresses(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write Addresses: Allocate tmpSgpr for initOpt
  ##############################################################################
  @abc.abstractmethod
  def lwaInitOptAllocate(self):
    return ""

  ##############################################################################
  # Local Write Addresses: Release tmpSgpr for initOpt
  ##############################################################################
  @abc.abstractmethod
  def lwaInitOptRelease(self):
    return ""

  ##############################################################################
  # Local Read Addresses: Tile Assignment
  ##############################################################################
  @abc.abstractmethod
  def lraTileAssignment(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # Local Read Addresses: Final Offset A/B
  ##############################################################################
  @abc.abstractmethod
  def lraFinalOffset(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read Addresses offset conversion for DTL + NLC > 1
  ##############################################################################
  @abc.abstractmethod
  def lraOffsetConversionForDTLandNLC(self, kernel, tP, offset_val, generateAsm=False, \
                                      finalVgpr=None, tmp1=None, tmp2=None):
    return ""

  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def lraDeclareAddresses(self, kernel, tP):
    return ""

  ##############################################################################
  # Recalculate local read addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def recalcLocalReadAddressesAB(self, kernel):
    return ""

  ##############################################################################
  # Recalculate local write addresses A/B
  ##############################################################################
  @abc.abstractmethod
  def recalcLocalWriteAddresses(self, kernel, tP, uDu):
    return ""

  ##############################################################################
  # Declare Loop Num Iterations
  ##############################################################################
  @abc.abstractmethod
  def declareLoopNumIter(self, kernel):
    return ""

  ##############################################################################
  # Define stagger parms that will be used in calculateStagger
  ##############################################################################
  @abc.abstractmethod
  def declareStaggerParms(self, kernel):
    return ""


  ##############################################################################
  # Calculate and apply stagger offsets and edge
  ##############################################################################
  @abc.abstractmethod
  def calculateStagger(self, kernel, loopIdx):
    return ""

  ##############################################################################
  # Remove stagger offset (before tail loop)
  ##############################################################################
  @abc.abstractmethod
  def removeStagger(self, kernel):
    return ""

  ##############################################################################
  # Calculate Loop Num Iter
  ##############################################################################
  @abc.abstractmethod
  def calculateLoopNumIter(self, kernel, loopIdx, isPap):
    return ""


  ##############################################################################
  # openShadowInit:
  # Top of shadow init code
  ##############################################################################
  @abc.abstractmethod
  def openShadowInit(self, kernel):
    return ""

  ##############################################################################
  # closeShadowInit:
  # Top of shadow init code
  ##############################################################################
  @abc.abstractmethod
  def closeShadowInit(self, kernel):
    return ""

  ##############################################################################
  # Initialize C
  ##############################################################################
  @abc.abstractmethod
  def initC(self, kernel):
    return ""

  ##############################################################################
  # Open Loop
  # loopIdx<0 : tail loop
  ##############################################################################
  @abc.abstractmethod
  def openLoop(self, kernel, loopIdx, uDu, noLabelGen, beginLabelOnly):
    return ""

  ##############################################################################
  # Close Loop
  ##############################################################################
  @abc.abstractmethod
  def closeLoop(self, kernel, loopIdx, finalLoop, loopCopies, uDu=None, emitEndLabelOnly=False, oddLabel=False, \
                skipCondJumpCounter=-1, isOptNLL=False, skipJump=False):
    return ""

  ##############################################################################
  # Open Loop Copy
  ##############################################################################
  @abc.abstractmethod
  def openLoopCopy(self, kernel, lc):
      return ""

  ##############################################################################
  # End Summation
  ##############################################################################
  @abc.abstractmethod
  def endSummation(self, kernel, label = None, isOptNLL = False):
    return ""

  ##############################################################################
  # MAC Iteration
  # useMacro : if true, call the MAC* macro. If False, inline the MACs
  ##############################################################################
  @abc.abstractmethod
  def macIter(self, kernel, bufferIdx, iuiCount, useMacro, isTail=False):
    return ""

  ##############################################################################
  # At Least 1 Unroll
  ##############################################################################
  @abc.abstractmethod
  def openSumAtLeastUnroll(self, kernel, prefetch, isOptNLL, isPap):
    return ""

  @abc.abstractmethod
  def closeSumAtLeastUnroll(self, kernel, prefetch, isOptNLL, isPap, isNGLL):
    return ""

  ##############################################################################
  # Global Read: Increment A/B
  ##############################################################################
  @abc.abstractmethod
  def globalReadIncrementAB(self, kernel, loopIdx, prefetchIndex, incs=1):
    return ""

  ##############################################################################
  # Global Read: Do It A/B
  # mode: 0=prefetch, 1=unroll loop, 2=guardK
  ##############################################################################
  @abc.abstractmethod
  def globalReadDo(self, kernel, mode, tP, vregSetIdx=0):
    return ""

  ##############################################################################
  # Global Read A/B completed
  ##############################################################################
  @abc.abstractmethod
  def doneGlobalABReads(self, kernel):
    return ""

  ##############################################################################
  # directToLds m0 update: Do It A/B
  # mode: 0=prefetch, 1=unroll loop, 2=guardK
  ##############################################################################
  @abc.abstractmethod
  def directToLdsM0Update(self, kernel, mode, tP, usePlaceHolder=False):
    return ""

  ##############################################################################
  # Local Write: Swap Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteSwapOffsets(self, kernel, internalPointerSwap, tP):
    return ""

  ##############################################################################
  # Local Write: Reset Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteResetOffsets(self, kernel, internalPointerSwap, tP):
    return ""

  ##############################################################################
  # Local Write: Init Pointers A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteInitPointers(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write in Prefetch Pass (PreLoop): Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def preLoopLocalWriteDo(self, kernel, tPA, tPB):
    return ""

  ##############################################################################
  # Replace the determined vmcnt in PreLoop LocalWrite
  ##############################################################################
  @abc.abstractmethod
  def replacePreLoopLWVmcnt(self, kernel):
    return ""

  ##############################################################################
  # Local Write: Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def localWriteDo(self, kernel, tP, uDu):
    return ""

  ##############################################################################
  # Local Read: Swap Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadSwapOffsets(self, kernel, internalPointerSwap, tP):
    return ""

  ##############################################################################
  # Local Read: Reset Offsets A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadResetOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Init Pointers A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadInitPointers(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Increment A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadInc(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Read: Do It A/B
  ##############################################################################
  @abc.abstractmethod
  def localReadDo(self, kernel, bufferIdx, innerUnrollIndex, epsi, tP):
    return ""

  ##############################################################################
  # Shift Vector Components d0/1
  ##############################################################################
  @abc.abstractmethod
  def shiftVectorComponents(self, kernel, tP):
    return ""

  ##############################################################################
  # Complex Declare Tmp Registers
  ##############################################################################
  @abc.abstractmethod
  def complexDeclareTmpRegisters(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Local Write
  ##############################################################################
  @abc.abstractmethod
  def localSplitULocalWrite(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Local Read
  ##############################################################################
  @abc.abstractmethod
  def localSplitULocalRead(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Reduction
  ##############################################################################
  @abc.abstractmethod
  def localSplitUReduction(self, kernel):
    return ""

  ##############################################################################
  # globalWriteWorkGroupInitBeforePersistentLoop:
  ##############################################################################
  @abc.abstractmethod
  def globalWriteWorkGroupInitBeforePersistentLoop(self, kernel):
    return ""

  ##############################################################################
  # globalWriteWorkGroupInit:
  # Perform work-group granularity init
  ##############################################################################
  @abc.abstractmethod
  def globalWriteWorkGroupInit(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  @abc.abstractmethod
  def localSplitUGlobalWriteIndices(self, kernel):
    return ""

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  @abc.abstractmethod
  def localSplitUGlobalWrite(self, kernel):
    return ""

  ##############################################################################
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  @abc.abstractmethod
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    return ""

  ##############################################################################
  # Not LocalSplitU: Global Write
  ##############################################################################
  @abc.abstractmethod
  def notLocalSplitUGlobalWrite(self, kernel):
    return ""

  @abc.abstractmethod
  def openPrefetchAcrossPersistent(self, kernel, isOptNLL=False, useBufferOOB=False):
    return ""

  @abc.abstractmethod
  def closePrefetchAcrossPersistent(self, kernel, isOptNLL=False, useBufferOOB=False):
    return ""

  ##############################################################################
  # init for StoreCInUnroll
  ##############################################################################
  @abc.abstractmethod
  def initStoreCInUnroll(self, kernel):
    return ""

  ##############################################################################
  # init for StoreCInUnroll per Persistent Loop
  ##############################################################################
  @abc.abstractmethod
  def initStoreCInUnrollPerPersistentLoop(self, kernel):
    return ""

  ##############################################################################
  # init for StoreCInUnroll per Unroll Loop
  ##############################################################################
  @abc.abstractmethod
  def initStoreCInUnrollPerUnrollLoop(self, kernel, needInit):
    return ""

  ##############################################################################
  # swap SrdC and SrdCbackup, SrdD and SrdDbackup
  ##############################################################################
  @abc.abstractmethod
  def swapSrdCDandBackup(self, kernel):
    return ""

  ##############################################################################
  # C/D address increment value for StoreCInUnroll
  ##############################################################################
  @abc.abstractmethod
  def generateCorDaddrIncrementForStoreCInUnroll(self, kernel, CorD, odd, tmpSgprWork):
    return ""

  ##############################################################################
  # get address/gpr index increment frequency for StoreCInUnroll
  ##############################################################################
  @abc.abstractmethod
  def getAddrGprIdxIncrementFrequencyForStoreCInUnroll(self, kernel):
    return ""

  ##############################################################################
  # generate post process for StoreCInUnroll loop
  ##############################################################################
  @abc.abstractmethod
  def generatePostProcessForStoreCInUnrollLoop(self, kernel, needPost):
    return ""

  ##############################################################################
  # restore SrdCbackup and SrdDbackup
  ##############################################################################
  @abc.abstractmethod
  def restoreSrdCandDBackup(self, kernel):
    return ""

  ##############################################################################
  # reset storeC sync objects
  ##############################################################################
  @abc.abstractmethod
  def resetStoreCsyncObject(self, kernel):
    return ""

  ##############################################################################
  # set storeC sync objects
  ##############################################################################
  @abc.abstractmethod
  def setStoreCsyncObject(self, kernel):
    return ""

  ##############################################################################
  # end process for StoreCInUnroll per PersistentLoop (OptNLL)
  ##############################################################################
  @abc.abstractmethod
  def endProcessPersistentLoopforStoreCInUnrollOptNLL(self, kernel):
    return ""

  ##############################################################################
  # end process for StoreCInUnroll per PersistentLoop (NoOptNLL)
  ##############################################################################
  @abc.abstractmethod
  def endProcessPersistentLoopforStoreCInUnrollNoOptNLL(self, kernel):
    return ""

  ##############################################################################
  # number of storeC code in template for StoreCInUnroll
  ##############################################################################
  @abc.abstractmethod
  def getNumberOfStoreCInTemplate(self, kernel):
    return ""

  ##############################################################################
  # number of LoadC code in template for StoreCInUnroll
  ##############################################################################
  @abc.abstractmethod
  def getNumberOfLoadCInForLoadC(self, kernel):
    return ""

  ##############################################################################
  # generate storeCInUnroll post loop code
  ##############################################################################
  @abc.abstractmethod
  def generateStoreInUnrollPostLoop(self, kernel, isOptNLL, isDTVodd):
    return ""

  ##############################################################################
  # openOddNoLoadLoopForDTV
  # generate open code for DirectToVgpr + odd exit case in noLoadLoop code
  ##############################################################################
  @abc.abstractmethod
  def openOddNoLoadLoopForDTV(self, kernel, isNGLL, name):
    return ""

  ##############################################################################
  # closeOddNoLoadLoopForDTV
  # generate close code for DirectToVgpr + odd exit case in noLoadLoop code
  ##############################################################################
  @abc.abstractmethod
  def closeOddNoLoadLoopForDTV(self, kernel, isNGLL, name):
    return ""

  ##############################################################################
  # generateEvenEndLabeNoLoadLoopForDTV
  # generate even end label for DirectToVgpr
  ##############################################################################
  @abc.abstractmethod
  def generateEvenEndLabeNoLoadLoopForDTV(self, kernel, isNGLL, name):
    return ""

  ##############################################################################
  # generateOddEndVgprCopyForDTV
  # generate odd end vgpr copy for DirectToVgpr
  ##############################################################################
  @abc.abstractmethod
  def generateOddEndVgprCopyForDTV(self, kernel):
    return ""

  ##############################################################################
  # isSwapGlobalReadOrderForDtvOrDtl
  ##############################################################################
  @abc.abstractmethod
  def isSwapGlobalReadOrderForDtvOrDtl(self, kernel):
    return ""

  ##############################################################################
  # PrefetchGlobalRead2
  ##############################################################################
  @abc.abstractmethod
  def openPrefetchGlobalRead2(self, kernel):
    return ""

  @abc.abstractmethod
  def closePrefetchGlobalRead2(self, kernel):
    return ""

  ##############################################################################
  # Function End
  ##############################################################################
  @abc.abstractmethod
  def functionEnd(self, kernel, addLabel=True):
    return ""

  ##############################################################################
  # Function Suffix
  ##############################################################################
  @abc.abstractmethod
  def functionSuffix(self, kernel):
    return ""

  ##############################################################################
  # Kernel Body Prefix
  ##############################################################################
  @abc.abstractmethod
  def kernelBodyPrefix(self, kernel, tPA, tPB ):
    return ""


  ##############################################################################
  # Kernel Body Suffix
  ##############################################################################
  @abc.abstractmethod
  def kernelBodySuffix(self, kernel, tPA, tPB ):
    return ""

  ##############################################################################
  # WaitCnt
  ##############################################################################
  @abc.abstractmethod
  def wait(self, kernel, tPA, tPB, globalRead, localWrite, localRead, comment):
    return ""

  ##############################################################################
  # waitcnt code for DirectToVgpr
  ##############################################################################
  @abc.abstractmethod
  def getWaitcntCodeForDirectToVgpr(self, kernel, localWriteEndIter, u, firstIter, isPap=True, beforeBarrier=False, NLLlast=False, oddLast=False):
    return ""

  ##############################################################################
  # waitcnt code for PrefetchGlobalRead
  ##############################################################################
  @abc.abstractmethod
  def getWaitcntCodeForPGR(self, kernel):
    return ""

  ##############################################################################
  # SyncThreads
  ##############################################################################
  @abc.abstractmethod
  def syncThreads(self, kernel):
    return self.indent + self.syncStr + self.endLine

  ##############################################################################
  # MapAcctoArch
  ##############################################################################
  @abc.abstractmethod
  def MapAcctoArchRegs(self, kernel, option):
    return ""

  ##############################################################################
  # openmovaccVgpr
  ##############################################################################
  @abc.abstractmethod
  def openmovaccVgpr(self, kernel, backupSgpr):
    return ""

  ##############################################################################
  # getAccVgprCode
  ##############################################################################
  @abc.abstractmethod
  def getAccVgprCode(self,kernel,odd):
    return ""

  ##############################################################################
  # closemovaccVgpr
  ##############################################################################
  @abc.abstractmethod
  def closemovaccVgpr(self, kernel, backupSgpr):
    return ""


  ##############################################################################
  #
  #   Entry Functions
  #
  ##############################################################################


  ##############################################################################
  # get kernel name
  ##############################################################################
  def getKernelFileBase(self, kernel):
    if isCustomKernelConfig(kernel):
      fileBase = kernel["CustomKernelName"]
    elif globalParameters["ShortNames"]:
      fileBase = Solution.getNameSerial(kernel, self.kernelSerialNaming)
    else:
      fileBase = self.shortenFileBase(kernel)
    return fileBase

  def getKernelName(self, kernel):
    kernelName = Solution.getNameMin(kernel, self.kernelMinNaming)
    return kernelName

  def getKernelSource(self, kernel):
    """
    Returns the source of the kernel, either C++ or assembly.
    """


    fileString = ""
    self.tPA = tensorParametersA = {}
    self.tPB = tensorParametersB = {}
    self.initKernel(kernel, tensorParametersA, tensorParametersB )
    fileString += self.kernelBodyPrefix( kernel, tensorParametersA, \
        tensorParametersB )
    self.stringIdx = 0
    (error, kb) = self.kernelBody( kernel, tensorParametersA, tensorParametersB)

    fileString += kb
    fileString += self.kernelBodySuffix( kernel, tensorParametersA, \
        tensorParametersB )

    if error != 0:
      if globalParameters["ForceGenerateKernel"]:
        print ("warning: Generating kernel source resulted in error {}, but ForceGenerateKernel=1 so saving source".format(error))
      else:
        raise RuntimeError("Generating kernel source resulted in error {}".format(error))

    return fileString

  def getAssemblyDirectory(self):
      return Common.ensurePath(os.path.join(globalParameters["WorkingPath"], "assembly"))

  def byteArrayScriptSource(self):
    return """
#!/usr/bin/env python

fileString = ""
fileString += "/*******************************************************************************\\n"
fileString += "* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.\\n"
fileString += "*\\n"
fileString += "* Permission is hereby granted, free of charge, to any person obtaining a copy\\n"
fileString += '* of this software and associated documentation files (the \"Software\"), to deal\\n'
fileString += "* in the Software without restriction, including without limitation the rights\\n"
fileString += "* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-\\n"
fileString += "* ies of the Software, and to permit persons to whom the Software is furnished\\n"
fileString += "* to do so, subject to the following conditions:\\n"
fileString += "*\\n"
fileString += "* The above copyright notice and this permission notice shall be included in all\\n"
fileString += "* copies or substantial portions of the Software.\\n"
fileString += "*\\n"
fileString += '* THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-\\n'
fileString += "* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS\\n"
fileString += "* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR\\n"
fileString += "* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER\\n"
fileString += "* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-\\n"
fileString += "* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\\n"
fileString += "*******************************************************************************/\\n\\n"
fileString += "/**************************************************\\n"
fileString += "* This file was generated by Tensile:             *\\n"
fileString += "* https://github.com/ROCmSoftwarePlatform/Tensile *\\n"
fileString += "**************************************************/\\n\\n\\n"
import os.path
fileString += '#include "Kernels.h"\\n\\n'
fileString += "/* code object byte array */\\n\\n"
codeObjectFileNames = [f for f in os.listdir(".") if (os.path.isfile(f) and f.endswith(".co"))]
for codeObjectFileName in codeObjectFileNames:
  print codeObjectFileName
  print "\\n"
  kernelName=os.path.splitext(codeObjectFileName)[0]
  codeObjectFile = open(codeObjectFileName, "r")
  codeObjectByteArray = bytearray(codeObjectFile.read())
  codeObjectFile.close()
# write code object byte array for asm
  fileString += "const unsigned char %s_coba[%u] = {\\n" % (kernelName, len(codeObjectByteArray))
  for byteIdx in range(0, len(codeObjectByteArray)):
    byte = codeObjectByteArray[byteIdx]
    fileString += "0x%02x" % byte
    if byteIdx < len(codeObjectByteArray)-1:
      fileString += ","
    else:
      fileString += "};\\n"
    if byteIdx % 16 == 15:
      fileString += "\\n"
  text_file = open("Kernels.cpp", "w")
  text_file.write("%s" % fileString)
  text_file.close()
"""

  def writeByteArrayScript(self):
    asmPath = self.getAssemblyDirectory()

    bytearrayFileName = os.path.join(asmPath,"insert_byte_array.py")
    if not os.path.isfile(bytearrayFileName):
      with open(bytearrayFileName, 'w') as bytearrayFile:
        bytearrayFile.write(self.byteArrayScriptSource())
      os.chmod(bytearrayFileName, 0o777)
    return bytearrayFileName

  def getReplacementKernelPath(self, kernel):
    if not kernel["ReplacementKernel"] and not isCustomKernelConfig(kernel): #kernel["CustomKernelName"]:
      return None

    kernelName = self.getKernelName(kernel)

    if isCustomKernelConfig(kernel):
      return os.path.join(globalParameters["CustomKernelDirectory"], (kernelName + ".s"))
    else: # Replacement kernel
      return ReplacementKernels.Get(kernelName)

  def shortenFileBase(self, kernel):
    base = self.getKernelName(kernel)
    if len(base) <= globalParameters["MaxFileName"]:
      return base

    import hashlib
    import base64

    pivot = globalParameters["MaxFileName"] * 3 // 4
    firstPart = base[:pivot]
    secondPart = base[pivot:]

    secondHash = hashlib.sha256(secondPart.encode()).digest()
    secondPart = base64.b64encode(secondHash, b'_-').decode()

    return firstPart + secondPart

  def getKernelObjectAssemblyFile(self, kernel):
    asmPath = self.getAssemblyDirectory()
    # write assembly file to assembly directory
    kernelName = self.getKernelFileBase(kernel)
    fileBase = os.path.join(asmPath, kernelName )
    assemblyFileName = "%s.s" % fileBase

    replacementKernel = self.getReplacementKernelPath(kernel)

    if replacementKernel is not None:
      self.tPA = tensorParametersA = {}
      self.tPB = tensorParametersB = {}
      if isCustomKernelConfig(kernel):
        kernelFoundMessage = "Custom kernel filename "
        # ISA version, such as 803
        self.kernel = kernel
        self.language = "ASM"
        self.version = globalParameters["CurrentISA"]
        if "ISA" in kernel:
          self.version = tuple(kernel["ISA"])
        if not globalParameters["AsmCaps"][self.version]["SupportedISA"]:
          defaultIsa = (9,0,0)
          print("warning: ISA:", self.version, " is not supported; overriding with ", defaultIsa)
          self.version = defaultIsa
      else:
        kernelFoundMessage = "replacement_assemblyFilename "
        self.initKernel(kernel, tensorParametersA, tensorParametersB )

      shutil.copyfile(replacementKernel, assemblyFileName)
      if globalParameters["PrintLevel"] >= 1:
        print(kernelFoundMessage + assemblyFileName)
    else:
      kernelSource = self.getKernelSource(kernel)

      if globalParameters["PrintLevel"] >= 3:
        print("write_assemblyFilename %s" % assemblyFileName)

      with open(assemblyFileName, 'w') as assemblyFile:
        assemblyFile.write(kernelSource)

    return assemblyFileName

  def getAssembledKernelObjectFile(self, kernel):
    assemblyFileName = self.getKernelObjectAssemblyFile(kernel)

    base, ext = os.path.splitext(assemblyFileName)
    objectFileName = base + '.o'

    args = self.getCompileArgs(assemblyFileName, objectFileName)
    tPrint(2, "Assemble command: " + " ".join(args) + " && ")

    # change to use  check_output to force windows cmd block util command finish
    try:
      out = subprocess.check_output(args, stderr=subprocess.STDOUT, cwd=self.getAssemblyDirectory())
      tPrint(3, out)
    except subprocess.CalledProcessError as err:
      print(err.output)
      raise
    if self.removeTemporaries:
        os.remove(assemblyFileName)

    return objectFileName

  def getSingleCodeObjectFile(self, kernel):
    objectFileName = self.getAssembledKernelObjectFile(kernel)

    base, ext = os.path.splitext(objectFileName)
    coFileName = base + '.co'

    args = self.getLinkCodeObjectArgs([objectFileName], coFileName)

    tPrint(2, 'Single Code Object File: ' + ' '.join(args))

    # change to use  check_output to force windows cmd block util command finish
    try:
      out = subprocess.check_output(args, stderr=subprocess.STDOUT, cwd=self.getAssemblyDirectory())
      tPrint(3, out)
    except subprocess.CalledProcessError as err:
      print(err.output)
      raise

    return coFileName

  def getByteArrayCobaDefinition(self, varName, byteArray):
    s = self.comment("code object byte array")
    s += "const unsigned char %s_coba[%u] = {\n" % (varName, len(byteArray))

    if len(byteArray) != 0:
      s += "0x%02x" % byteArray[0]
      for byteIdx, byte in enumerate(byteArray[1:]):
        if byteIdx % 16 == 15:
          s += ",\n0x%02x" % byte
        else:
          s += ",0x%02x" % byte

    s += '};\n'
    return s

  def getFileCobaDefinition(self, varName, fileName):
    with open(fileName, 'rb') as f:
      byteArray = bytearray(f.read())
    return self.getByteArrayCobaDefinition(varName, byteArray)

  ##############################################################################
  def getSourceFileString(self, kernel):
    """
    Returns a string suitable for placing in Kernels.cpp.  This means the actual kernel source in the case
    of a source kernel, or an assembled code object byte array definition in the case of an assembly kernel,
    or an empty string in the case that CodeFromFiles is true.

    In the case of an assembly kernel, this function has the side effect of creating the following files:
     * An assembly source file
     * An object file
     * A code object file
     * A Python script which can create byte array variable definitions.
    """

    try:
      if kernel["KernelLanguage"] == "Assembly":
        # asmPath = self.getAssemblyDirectory()
        # kernelName = self.getKernelName(kernel)

        # Skip if .o files will have already been built for this file
        # @TODO remove need for this with better code organization
        if kernel.duplicate:
          self.language = "ASM"
          return (0, "")
        if globalParameters["GenerateSourcesAndExit"]:
          # only create the assembly file.
          self.getKernelObjectAssemblyFile(kernel)
          return (0, "")
        else:
          self.writeByteArrayScript()
          self.getSingleCodeObjectFile(kernel)

          # I guess in this case we are making sure that the code object file exists by executing the code
          # above but we aren't placing it into the source.
          return (0, "")

          # Old client debug option
          # return (0, self.getFileCobaDefinition(kernelName, os.path.join(asmPath, coFile)))

      else:
        return (0, self.getKernelSource(kernel))

    except subprocess.CalledProcessError as exc:
      if isinstance(exc.cmd, Sequence):
        print("Command: ")
        print(' '.join(exc.cmd))
        print("returned non-zero exit status ", exc.returncode)
      else:
        print(exc)
      return (-1, "")
    except RuntimeError as exc:
      if globalParameters["PrintSolutionRejectionReason"]:
        print(exc)
      return (-2, "")

  ##############################################################################
  # header file string
  ##############################################################################
  def getHeaderFileString(self, kernel):
    kernelName = self.getKernelName(kernel)
    fileString = "" # CHeader
    if self.language == "HIP":
      if not globalParameters["MergeFiles"]:
        fileString += CHeader
        fileString += "#pragma once\n\n"
        fileString += "#include <hip/hip_runtime.h>\n"
        fileString += "#include <hip/hip_fp16.h>\n"
        fileString += "#include <KernelHeader.h>\n"
        fileString += "\n"
      fileString += self.functionSignature(kernel)
      fileString += ";\n"
    else:
      if not globalParameters["MergeFiles"] or globalParameters["NumMergedFiles"] > 1:
        fileString += "#pragma once\n\n"
      if not globalParameters["CodeFromFiles"]:
        fileString += "extern const unsigned char %s_coba[]; // code object byte array\n" % kernelName

    return fileString

  ##############################################################################
  # flip Vreg set for DirectToVgpr in global read
  ##############################################################################
  def flipVregSetForDirectToVgprInGlobalRead(self, kernel, itemStr):
    # need to swap VGPR register set for odd code
    baseName = "G2LA" if kernel["DirectToVgprA"] else "G2LB" # only one of them is enabled
    set0 = baseName + "0"
    set1 = baseName + "1"
    if set0 in itemStr:
      # replace set0 with set1
      itemStr = itemStr.replace(set0, set1)
    elif set1 in itemStr:
      # replace set1 with set0
      itemStr = itemStr.replace(set1, set0)
    return itemStr

  ##############################################################################
  # Backup StoreCInUnroll related code
  ##############################################################################
  def backupStoreCInUnrollRelatedCode(self):
    # keep StoreCInUnrollPreCode, StoreCUnrollPostCode for the next noLoadLoop
    self.StoreCUnrollPreCodeBackup = copy.deepcopy(self.StoreCUnrollPreCode)
    self.StoreCUnrollPostCodeBackup = copy.deepcopy(self.StoreCUnrollPostCode)

  ##############################################################################
  # Restore StoreCInUnroll related code
  ##############################################################################
  def restoreStoreCInUnrollRelatedCode(self):
    self.StoreCUnrollPreCode = self.StoreCUnrollPreCodeBackup
    self.StoreCUnrollPostCode = self.StoreCUnrollPostCodeBackup
    self.StoreCUnrollLoopCodeStarted = 0

  ##############################################################################
  # generate storeC code in UnrollLoop
  ##############################################################################
  def generateStoreCCodeInUnrollLoop(self, kernel, odd, isLast=False):
    self.LoadCUnrollCode = Code.Module()
    self.StoreCUnrollCode = Code.Module()
    self.StoreCUnrollPreCode = Code.Module()
    self.StoreCUnrollPostCode = Code.Module()
    self.numItemsBeforeStoreC = 0
    self.StoreCUnrollStartComment ="Start of StoreCInUnroll code"
    self.StoreCUnrollStoreStartComment ="Start of StoreCInUnroll Store code"
    self.StoreCUnrollLoopCodeStarted = 0  # 0:not StoreC code started, 1: started
    if kernel["StoreCInUnroll"]:
      needInit = not odd
      needPost = odd
      needInc  = (not isLast) or kernel["StoreCInUnrollPostLoop"]
      backupSgpr = self.getTmpSgpr(2).idx()  # allocate all tmp register here
      tmpSgprWork = backupSgpr + 1
      needAddrC = (not kernel["AssertCEqualsD"]) and kernel["ProblemType"]["UseBeta"]

      # init/inc code is necessary if inc frequency is 1
      needInit = needInit or (self.getAddrGprIdxIncrementFrequencyForStoreCInUnroll(kernel) == 1)
      needPost = needPost or (self.getAddrGprIdxIncrementFrequencyForStoreCInUnroll(kernel) == 1)

      # generate init code for StoreCInUnroll per Unroll Loop
      initPerUnrollCode = self.initStoreCInUnrollPerUnrollLoop(kernel, needInit)

      # loadC
      for x in self.LoadCTemplate.items():
        # Load C case, insert Init per unroll code before Load C (setup vgpr offset for loadC and StoreC)
        s = initPerUnrollCode + str(x)
        initPerUnrollCode = "" # reset initPerUnrollCode so that it is not inserted again
        self.LoadCUnrollCode.addText(s)
      # Addr C increment code (no increment for isLast (and not PostLoop))
      if needInc and needAddrC:
        oddParam = needPost
        kStr = self.generateCorDaddrIncrementForStoreCInUnroll(kernel, "C", oddParam, tmpSgprWork)
        self.LoadCUnrollCode.addText(kStr)

      if initPerUnrollCode != "":
        # If init code is not inserted (no Load C case), insert it to the top of StoreC list (setup vgpr offset for StoreC)
        self.StoreCUnrollPreCode.addText(initPerUnrollCode)
        initPerUnrollCode = "" # reset initPerUnrollCode so that it is not inserted again

      # these 3 items need to be in the same set
      #  open gpr indexing
      #  accVgpr (need gpr indexing)
      #  close gpr indexing
      kStr = self.openmovaccVgpr(kernel, backupSgpr)
      # odd case, use + (1 iteration) for gpr index, but not necessary if index frequency is 1
      oddGprIndex = odd and (self.getAddrGprIdxIncrementFrequencyForStoreCInUnroll(kernel) > 1)
      kStr += self.getAccVgprCode(kernel, oddGprIndex)
      first, second = self.closemovaccVgpr(kernel, backupSgpr)
      kStr += first
      self.StoreCUnrollPreCode.addText(kStr)
      # put second part of close gpr indexing separately (for better scheduling)
      self.StoreCUnrollPreCode.addText(second)
      # Alpha
      kStr = ""
      for x in self.AlphaOpTemplate.items():
        kStr += str(x)

      if kStr != "":
        self.StoreCUnrollPreCode.addText(kStr)

      # count the number of items before StoreC (before beta)
      self.numItemsBeforeStoreC = len(list(self.StoreCUnrollPreCode.items()))

      # StoreC

      # put marker comment to recognize start point of StoreC code
      # this must be the first item in self.StoreCUnrollCode.
      self.StoreCUnrollCode.addComment0(self.StoreCUnrollStartComment)
      # add necessary dummy based on number of mfma instructions between local write items
      # put enough interval (=3) for LocalWritePerMfma == -1 case
      numMfma = 3 if kernel["LocalWritePerMfma"] == -1 else roundUp(1/kernel["LocalWritePerMfma"])
      n = self.numItemsBeforeStoreC - numMfma # first numMfma items are inserted at the start comment and following mfmas
      while n >= numMfma:
        self.StoreCUnrollCode.addText("")
        n -= numMfma

      # insert items in postProcessList between StoreC/AtomicAdd (StoreVectorWidth=1 only)
      imod = Code.Module()
      imod.addComment0(self.StoreCUnrollStoreStartComment)
      StartComment = str(imod)

      # Beta
      kStrBeta = ""
      for x in self.BetaOpTemplate.items():
        kStrBeta += str(x)
      # double complex case or num of store == 1 case, put beta instruction separately
      if kStrBeta != "" and (kernel["ProblemType"]["DestDataType"].isDoubleComplex() or self.getNumberOfStoreCInTemplate(kernel) == 1):
        # combine beta code with first StoreC comment to avoid generating beta before alpha
        self.StoreCUnrollCode.addText(kStrBeta + StartComment)
        kStrBeta = ""
        StartComment = ""

      # number of instructions(items) of increment code between MFMAs
      putCount =  1
      postProcessListIndex = 0
      # generate post process for StoreCInUnroll loop
      # 1) increment gpr indexing (new value in tmp). Put this as separate item in StoreCUnrollCode
      # 2-1) increment StoreC address  (new value in tmp)
      # 2-2) check enable count and apply new values when necessary
      postProcessList = []
      finalAddrIncList = []
      if needInc:
        postProcessList, finalAddrIncList = self.generatePostProcessForStoreCInUnrollLoop(kernel, needPost)

      for x in self.StoreCTemplate.items():
        kStr = ""
        if x == self.StoreCTemplate.items()[0]:
          kStr += kStrBeta + StartComment # combine beta code with first StoreC. first item case, add marker comment
          StartComment = ""
        strX = str(x)
        kStr += strX
        if x != self.StoreCTemplate.items()[-1]:
          # not the last StoreC
          # add postprocess code or empty between StoreC
          self.StoreCUnrollCode.addCode(kStr)
          end = kernel["StoreCInUnrollInterval"] - 1
          for i in range(end):
            if postProcessListIndex < len(postProcessList):
              self.StoreCUnrollCode.addText(postProcessList[postProcessListIndex])
              postProcessListIndex += 1
            else:
              self.StoreCUnrollCode.addText("") # add empty str to add interval between Store codes
        else:
          # last StoreC
          if not (kernel["StoreCInUnrollPostLoop"] and isLast):
            # last element and not StoreCInUnrollPostLoop+isLast case
            self.StoreCUnrollCode.addCode(kStr)
            # add remaining postprocess, finalAddrInc code in StoreCUnrollPostCode
            count = 0
            kStr = ""
            for i in range(postProcessListIndex, len(postProcessList)):
              kStr += postProcessList[i]
              count+=1
              if count == putCount:
                self.StoreCUnrollPostCode.addText(kStr)
                count = 0
                kStr = ""
            for i in range(len(finalAddrIncList)):
              kStr += finalAddrIncList[i]
              count+=1
              if count == putCount:
                self.StoreCUnrollPostCode.addText(kStr)
                count = 0
                kStr = ""
            if count > 0:
              self.StoreCUnrollPostCode.addText(kStr)
          else:
            # not last element or StoreCInUnrollPostLoop+isLast
            # add all remaining items in postProcessList and finalAddrInc code after the last StoreC (in the same item)
            for item in (postProcessList[postProcessListIndex:] + finalAddrIncList):
              kStr += item
            self.StoreCUnrollCode.addCode(kStr)
