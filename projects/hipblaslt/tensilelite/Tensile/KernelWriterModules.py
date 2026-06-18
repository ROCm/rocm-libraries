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

from rocisa.code import Label, Module
from rocisa.container import vgpr, sgpr, accvgpr, Holder, MemTokenData
from rocisa.instruction import SBarrier, SBranch, SMovB32, SMovB64, SWaitCnt, SWaitTensorcnt,\
  VAccvgprReadB32, VAccvgprWriteB32, VFmaF32, VFmaF64, VLShiftLeftB64, VMovB32, \
  VMovRelsD2B32, VMulF32, VMulF64, VMulLOU32, VMulPKF16
from rocisa.functions import BranchIfNotZero

from Tensile.Common.DataType import DataType

def tdmWait(states, kernel, tPA, tPB, tensorcnt: int, comment: str) -> Module:
  #TODO: refactor this
  skipGR = tensorcnt > -1
  vmcnt = 0 if skipGR else -1
  mod = Module()
  if skipGR:
    #TODO: remove
    # numMXSA = kernel["NumLoadsPerpendicularMXSA"] * kernel["NumLoadsCoalescedMXSA"] if kernel["ProblemType"]["MXBlockA"] else 0
    # numMXSB = kernel["NumLoadsPerpendicularMXSB"] * kernel["NumLoadsCoalescedMXSB"] if kernel["ProblemType"]["MXBlockB"] else 0
    # numM = 0
    # if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
    #   numM = kernel["NumLoadsPerpendicularMetadata"] * kernel["NumLoadsCoalescedMetadata"]
    # numGR = 0
    # if tensorcnt > -1:
    #   numGR += tensorcnt * (numMXSA + numMXSB + numM)
    # vmcnt += numGR
    mod.add(SWaitCnt(vlcnt=vmcnt))
  mod.add(SWaitTensorcnt(tensorcnt=tensorcnt, comment=comment))
  return mod

##############################################################################
# WaitCnt
# 3 components can contribute to the waitcnt:
#   - Pending global reads.  (skipGlobalRead)
#   - Pending local write.  (skipLocalWrite)
#   - Pending local reads (skipLocalRead)
# specify global read in inst unit (for DirectToVgpr. Optional):
#   - Pending global reads in inst unit.
# If a skip* arg is -1, the associated component does not contribute to
# the expected dscnt or vlcnt
##############################################################################
def wait(states, kernel, tPA, tPB, skipGlobalRead, skipLocalWrite, \
    skipLocalRead, conservativeWaitCnt: int, comment, skipGlobalReadInst=-1):
    # skip = -1 -> ignore
    # skip =  n -> waitcnt(n*num)

    dscnt = 0 if skipLocalWrite > -1 or skipLocalRead > -1 else -1

    if skipLocalWrite > -1 or skipLocalRead > -1:
        if skipLocalWrite > -1:
            numA = 0 if (kernel["DirectToLdsA"] or  kernel["DirectToVgprA"]) \
                   else tPA["nrp"]*tPA["nrc"]*max(tPA["nwcv"],tPA["nwpv"])//tPA["nwcvpi"]
            numB = 0 if (kernel["DirectToLdsB"] or  kernel["DirectToVgprB"]) \
                   else tPB["nrp"]*tPB["nrc"]*max(tPB["nwcv"],tPB["nwpv"])//tPB["nwcvpi"]
            numMXSA = 0
            numMXSB = 0
            if kernel["ProblemType"]["MXBlockA"]:
                numMXSA = 0 if (kernel["DirectToLdsA"] or kernel["DirectToVgprA"]) \
                       else tPA["MX"]["nrp"]*tPA["MX"]["nrc"]*max(tPA["MX"]["nwcv"],tPA["MX"]["nwpv"])//tPA["MX"]["nwcvpi"]
            if kernel["ProblemType"]["MXBlockB"]:
                numMXSB = 0 if (kernel["DirectToLdsB"] or kernel["DirectToVgprB"]) \
                       else tPB["MX"]["nrp"]*tPB["MX"]["nrc"]*max(tPB["MX"]["nwcv"],tPB["MX"]["nwpv"])//tPB["MX"]["nwcvpi"]
            numM = 0
            if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
              tPM = tPA["tpsMetadata"] if tPA["is_sparse"] else tPB["tpsMetadata"]
              numM = tPM["nrp"]*tPM["nrc"]*max(tPM["nwcv"],tPM["nwpv"])//tPM["nwcvpi"]
            dscnt += skipLocalWrite * (numA + numB + numM + numMXSA + numMXSB)
        if skipLocalRead > -1:
            numInstPerReadA  = 2 if (tPA["localReadInstruction"].blockWidth == 6) else 1
            numInstPerReadB  = 2 if (tPB["localReadInstruction"].blockWidth == 6) else 1
            numReadsPerIterA = 0 if kernel["DirectToVgprA"] else states.numReadsPerIterA * numInstPerReadA
            numReadsPerIterB = 0 if kernel["DirectToVgprB"] else states.numReadsPerIterB * numInstPerReadB
            numReadsPerIterMXSA = states.numReadsPerIterMXSA if (kernel["ProblemType"]["MXBlockA"] and (not kernel["DirectToVgprMXSA"])) else 0
            numReadsPerIterMXSB = states.numReadsPerIterMXSB if (kernel["ProblemType"]["MXBlockB"] and (not kernel["DirectToVgprMXSB"])) else 0
            readsPerIter = numReadsPerIterA + numReadsPerIterMXSA + numReadsPerIterB + numReadsPerIterMXSB + states.numReadsPerIterMetadata
            dscnt += skipLocalRead * readsPerIter

    skipGR = skipGlobalRead > -1 or skipGlobalReadInst > -1
    vlcnt = 0 if skipGR else -1
    if skipGR:
        numA = kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"]
        numB = kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"]
        numMXSA = kernel["NumLoadsPerpendicularMXSA"] * kernel["NumLoadsCoalescedMXSA"] if kernel["ProblemType"]["MXBlockA"] else 0
        numMXSB = kernel["NumLoadsPerpendicularMXSB"] * kernel["NumLoadsCoalescedMXSB"] if kernel["ProblemType"]["MXBlockB"] else 0
        numM = 0
        if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
          numM = kernel["NumLoadsPerpendicularMetadata"] * kernel["NumLoadsCoalescedMetadata"]
        numGR = 0
        if skipGlobalRead > -1:
          numGR += skipGlobalRead * (numA + numB + numMXSA + numMXSB + numM)
        if skipGlobalReadInst > -1:
          numGR += skipGlobalReadInst
        vlcnt += numGR

        # Unlike flat loads, BufferLoad do not increment the outstanding
        # dscnt
        if dscnt > -1 and not kernel["BufferLoad"]:
            dscnt += numGR

    if (conservativeWaitCnt & 0x2) and skipGR or \
       (conservativeWaitCnt & 0x4) and skipLocalWrite != -1 or \
       (conservativeWaitCnt & 0x8) and skipLocalRead  != -1:
        imod = Module("ConservativeWaitCnt")
        imod.add(SWaitCnt(dscnt=0, vlcnt=0, vscnt=0, comment="debug %s"%comment))
        imod.add(SBarrier(comment="debug"))
        return imod

    if dscnt >= 0 and vlcnt >= 0:
        vlcnt = -1 # preserve prior behavior of removing vlcnt here?

    waitcnt = SWaitCnt(dscnt=dscnt, vlcnt=vlcnt, comment=comment)
    return waitcnt

##############################################################################
# SyncThreads
##############################################################################
def syncThreads(kernel, archCaps, asmCaps, comment="", skipForceWaitcnt0=False, memoryToken=None):
    imod = Module("syncThreads")
    if kernel["NumThreads"] > kernel["WavefrontSize"]:
        if asmCaps["SeparateVscnt"]:
            imod.add(SWaitCnt(dscnt=0, comment="extra navi wait"))
        elif kernel["_ScheduleIterAlg"] == 2 \
          or kernel["PrefetchGlobalRead"] >= 2 \
          or skipForceWaitcnt0:
            imod.addComment("Skip force waitcnt0")
        elif archCaps["Waitcnt0Disabled"]:
            imod.add(SWaitCnt(dscnt=0, vlcnt=0, vscnt=0, comment="force waitcnt0"))

        _barrier = SBarrier(comment=comment)
        if memoryToken is not None:
            _barrier.setMemToken(MemTokenData(memoryToken))
        imod.add(_barrier)
    else:
        imod.addComment("Skip barrier: NumThreads=%s"%(kernel["NumThreads"]) + \
                comment)
    return imod

def _getAccToArchInfo(kernel):
  matrixInstM  = (kernel["MatrixInstM"] * kernel["MatrixInstBM"]) if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
  matrixInstN  = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]
  matrixInstBM = 1                                                if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBM"]
  matrixInstBN = 1                                                if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstBN"]

  OutputsPerMFMA1B = matrixInstM * matrixInstN // kernel["WavefrontSize"]
  VectorWidth0     = kernel["VectorWidthA"]
  outerTT0         = kernel["MIWaveTile"][0] // VectorWidth0
  VectorWidth1     = kernel["VectorWidthB"]
  outerTT1         = kernel["MIWaveTile"][1] // VectorWidth1
  return matrixInstBM, matrixInstBN, OutputsPerMFMA1B, VectorWidth0, VectorWidth1, outerTT0, outerTT1

def getAccToArchLen(kernel):
  matrixInstBM, matrixInstBN, OutputsPerMFMA1B, VectorWidth0, VectorWidth1, outerTT0, outerTT1 = _getAccToArchInfo(kernel)
  return (outerTT1 * outerTT0 * matrixInstBN * matrixInstBM * OutputsPerMFMA1B * VectorWidth0 * VectorWidth1)

##############################################################################
# accToArchMapper
# Provides forward (acc2arch) and backward (arch2acc) index transformation
#  - Forward transformation is currently used for acc->vgpr copying
#  - Backward transformation is used in ShiftVectorComponent() to map logical
#    C-tile index back to original acc index
##############################################################################
def accToArchMapper(kernel):
  acc2arch = dict()
  arch2acc = dict()

  matrixInstBM, matrixInstBN, OutputsPerMFMA1B, VectorWidth0, VectorWidth1, outerTT0, outerTT1 = _getAccToArchInfo(kernel)

  for wgIdx1 in range(0, outerTT1):
    for wgIdx0 in range(0, outerTT0):
      for bIdx1 in range(0, matrixInstBN):
        for bIdx0 in range(0, matrixInstBM):
          for tIdx in range(0, OutputsPerMFMA1B):
            for vw1 in range(0, VectorWidth1):
              for vw0 in range(0, VectorWidth0):
                src, dst = 0, 0
                if kernel["SourceSwap"]:
                  src = tIdx + OutputsPerMFMA1B * (bIdx0 + matrixInstBM * (bIdx1 + matrixInstBN * (vw0 + VectorWidth0 * (wgIdx0 + outerTT0 * (vw1 + VectorWidth1 * (wgIdx1))))))
                  dst = vw0 + VectorWidth0 * (bIdx0 + matrixInstBM * (wgIdx0 + outerTT0 * (vw1 + VectorWidth1 * (tIdx + OutputsPerMFMA1B * (bIdx1 + matrixInstBN * (wgIdx1))))))
                else:
                  src = tIdx + OutputsPerMFMA1B * (bIdx1 + matrixInstBN * (bIdx0 + matrixInstBM * (vw0 + VectorWidth0 * (wgIdx0 + outerTT0 * (vw1 + VectorWidth1 * (wgIdx1))))))
                  dst = vw0 + VectorWidth0 * (tIdx + OutputsPerMFMA1B * (bIdx0 + matrixInstBM * (wgIdx0 + outerTT0 * (vw1 + VectorWidth1 * (bIdx1 + matrixInstBN * (wgIdx1))))))
                acc2arch[src] = dst
                arch2acc[dst] = src
  return acc2arch, arch2acc

def accVgprImagNumOffset(kernel):
  acc2arch, _ = accToArchMapper(kernel)
  return len(acc2arch) * kernel["MIRegPerOut"]

##############################################################################
# MapAcctoArch
# function to map MFMA Acc  Registers to Arch VGPR register
##############################################################################
def mapAcctoArchRegs(kernel, maxAgpr=256, write=False, spilledVgprBase=None):
  acc2arch, _ = accToArchMapper(kernel)

  complexMultiplier = 2 if kernel["ProblemType"]["DataType"].isComplex() else 1
  itemList = [None] * kernel["MIRegPerOut"] * complexMultiplier * len(acc2arch)
  accImOffset = accVgprImagNumOffset(kernel)
  for i in range(len(acc2arch)):
    for cm in range(complexMultiplier):
      for r in range(kernel["MIRegPerOut"]):
        destIdx = (acc2arch[i]*complexMultiplier + cm) * kernel["MIRegPerOut"] + r
        srcIdx = ((i * kernel["MIRegPerOut"] + r) + (cm*accImOffset))
        if not kernel["MIArchVgpr"]:
          def gprfunc(idx):
            if idx >= maxAgpr:
              return vgpr(idx-maxAgpr)
            else:
              return accvgpr(idx)
          accStr = gprfunc(srcIdx)
          if srcIdx >= maxAgpr:
            # Spilled accumulator: lives in an arch vgpr, not an accvgpr.
            # For subtile kernels the spilled D-tile vgprs are allocated from
            # the pool at spilledVgprBase (not at ValuC+N), so reference them
            # directly.  For non-subtile kernels spilledVgprBase is None and
            # the legacy "ValuC+N" addressing is used (vgprValuC == 0 there).
            spill_offset = srcIdx - maxAgpr
            if spilledVgprBase is not None:
              spilledVgpr = vgpr(spilledVgprBase + spill_offset)
            else:
              spilledVgpr = vgpr("ValuC+%u" % spill_offset)
            if write:
              itemList[destIdx] = VMovB32(dst=spilledVgpr,
                                             src=vgpr(Holder(name="ValuC")),
                                             comment="copy vreg[%u] to MI out reg" % destIdx)
            else:
              itemList[destIdx] = VMovB32(dst=vgpr(Holder(name="ValuC")),
                                              src=spilledVgpr,
                                              comment="copy MI out reg to vreg[%u]" % destIdx)
          else:
            if write:
              itemList[destIdx] = VAccvgprWriteB32(dst=accStr,
                                                        src=vgpr(Holder(name="ValuC")),
                                                        comment="copy vreg[%u] to acc" % destIdx)
            else:
              itemList[destIdx] = VAccvgprReadB32(dst=vgpr(Holder(name="ValuC")),
                                                      src=accStr,
                                                      comment="copy acc to vreg[%u]" % destIdx)
        else:
          if write:
            itemList[destIdx] = VMovB32(dst=vgpr("ValuC+%u"%srcIdx),
                                             src=vgpr(Holder(name="ValuC")),
                                             comment="copy vreg[%u] to MI out reg" % destIdx)
          elif kernel.get("CompactLoopStore", False):
            # CompactLoopStore: use v_movrelsd_2_b32 so the dst VGPR index is offset
            # by M0 at runtime. The CLS countdown loop (later commit) drives M0 per
            # iter so one "copy MI out reg" body covers multiple MI accumulator
            # slices. Non-CLS keeps v_mov_b32 verbatim.
            itemList[destIdx] = VMovRelsD2B32(dst=vgpr(Holder(name="ValuC")),
                                             src=vgpr("ValuC+%u"%srcIdx),
                                             comment="copy MI out reg to vreg[%u]" % destIdx)
          else:
            itemList[destIdx] = VMovB32(dst=vgpr(Holder(name="ValuC")),
                                             src=vgpr("ValuC+%u"%srcIdx),
                                             comment="copy MI out reg to vreg[%u]" % destIdx)
  imod = Module("AccVgpr{}".format("Write" if write else "Read"))
  imod.setItems(itemList)
  return imod

##############################################################################
# MulMIoutAlphaToArch
# function to handle MFMA alpha*MIout to Arch VGPR register
##############################################################################
def mulMIoutAlphaToArch(kernel, startVgprAlphaTmp):
  acc2arch, _ = accToArchMapper(kernel)
  itemList = [None] * len(acc2arch)
  for i in range(len(acc2arch)):
    destIdx = acc2arch[i]
    srcIdx  = i * kernel["MIRegPerOut"]
    # TODO: Add conversion support for different compute data types
    if kernel["ProblemType"]["ComputeDataType"].isDouble():
      itemList[destIdx] = VMulF64(dst=vgpr(Holder(name="ValuC"),2),
                                                    src0=sgpr("Alpha",2), src1=vgpr("ValuC+%u"%srcIdx,2),
                                                    comment="Multiply MI out reg with alpha")
    elif kernel["ProblemType"]["ComputeDataType"].isSingle() or \
        (kernel["ProblemType"]["ComputeDataType"].isHalf() and kernel["ProblemType"]["HighPrecisionAccumulate"]):
      itemList[destIdx] = VMulF32(dst=vgpr(Holder(name="ValuC")),
                                                    src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%srcIdx),
                                                    comment="Multiply MI out reg with alpha")
    elif (kernel["ProblemType"]["ComputeDataType"].isHalf() and not kernel["ProblemType"]["HighPrecisionAccumulate"]):
      itemList[destIdx] = VMulPKF16(dst=vgpr(Holder(name="ValuC")),
                                                       src0=sgpr("Alpha"),
                                                       src1=vgpr("ValuC+%u"%srcIdx), comment="Multiply MI out reg with alpha")
    elif kernel["ProblemType"]["ComputeDataType"].isInt32():
      itemList[destIdx] = VMulLOU32(dst=vgpr(Holder(name="ValuC")),
                                                      src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%srcIdx),
                                                       comment="Multiply MI out reg with alpha")
    elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
        accImOffset = accVgprImagNumOffset(kernel)
        cimod = Module()
        # cannot use tmp vgpr for write batch, use allocated vgpr instead
        vtmp1 = startVgprAlphaTmp
        vtmp2 = vtmp1 + 1
        # tmp1 = a.real * b.real
        cimod.add(VMulF32(dst=vgpr(vtmp1), src0=sgpr("Alpha+0"), src1=vgpr("ValuC+%u"%srcIdx), comment="tmp1 = a.real * b.real"))
        # tmp2 = a.imag * b.real
        cimod.add(VMulF32(dst=vgpr(vtmp2), src0=sgpr("Alpha+1"), src1=vgpr("ValuC+%u"%srcIdx), comment="tmp2 = a.imag * b.real"))
        # c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
        cimod.add(VFmaF32(dst=vgpr(Holder(name="ValuC")), src0=sgpr("Alpha+1"), src1=vgpr("ValuC+%u"%(srcIdx+accImOffset)).getMinus(), src2=vgpr(vtmp1), comment="c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag"))
        # c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
        cimod.add(VFmaF32(dst=vgpr(Holder(name="ValuC+1")), src0=sgpr("Alpha+0"), src1=vgpr("ValuC+%u"%(srcIdx+accImOffset)), src2=vgpr(vtmp2), comment="c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2"))
        itemList[destIdx] = cimod
    elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
      accImOffset = accVgprImagNumOffset(kernel)
      cimod = Module()
      # cannot use tmp vgpr for write batch, use allocated vgpr instead
      vtmp1 = startVgprAlphaTmp
      vtmp2 = vtmp1 + 2
      # tmp1 = a.real * b.real
      cimod.add(VMulF64(dst=vgpr(vtmp1,2), src0=sgpr("Alpha+0",2), src1=vgpr("ValuC+%u"%srcIdx,2)))
      # tmp2 = a.imag * b.real
      cimod.add(VMulF64(dst=vgpr(vtmp2,2), src0=sgpr("Alpha+2",2), src1=vgpr("ValuC+%u"%srcIdx,2)))
      # c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
      cimod.add(VFmaF64(dst=vgpr(Holder(name="ValuC"),2), src0=sgpr("Alpha+2",2), src1=vgpr("ValuC+%u"%(srcIdx+accImOffset),2).getMinus(), src2=vgpr(vtmp1,2)))
      # c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
      cimod.add(VFmaF64(dst=vgpr(Holder(name="ValuC+2"),2), src0=sgpr("Alpha+0",2), src1=vgpr("ValuC+%u"%(srcIdx+accImOffset),2), src2=vgpr(vtmp2,2)))
      itemList[destIdx] = cimod

  imod = Module("MulAlpha")
  imod.setItems(itemList)
  return imod

  ##############################################################################
  # MoveMIoutToArch
  # function to handle MFMA MIout to Arch VGPR register
  ##############################################################################
def moveMIoutToArch(kernel, startVgprAlphaTmp):
  acc2arch, _ = accToArchMapper(kernel)
  itemList = [None] * len(acc2arch)
  for i in range(len(acc2arch)):
    destIdx = acc2arch[i]
    srcIdx  = i * kernel["MIRegPerOut"]
    if kernel["ProblemType"]["ComputeDataType"].isDouble():
      itemList[destIdx] = VLShiftLeftB64(dst=vgpr(Holder(name="ValuC"), 2),
                                                     shiftHex=0,
                                                     src=vgpr("ValuC+%u"%srcIdx,2), comment="Rearrange MI out reg")
    elif kernel["ProblemType"]["ComputeDataType"].isSingle() or \
        (kernel["ProblemType"]["ComputeDataType"].isHalf() and kernel["ProblemType"]["HighPrecisionAccumulate"]):
      itemList[destIdx] = VMovB32(dst=vgpr(Holder(name="ValuC")),
                                                     src=vgpr("ValuC+%u"%srcIdx), comment="Rearrange MI out reg")
    elif (kernel["ProblemType"]["ComputeDataType"].isHalf() and not kernel["ProblemType"]["HighPrecisionAccumulate"]):
      itemList[destIdx] = VMovB32(dst=vgpr(Holder(name="ValuC")),
                                                     src=vgpr("ValuC+%u"%srcIdx), comment="Rearrange MI out reg")
    elif kernel["ProblemType"]["ComputeDataType"].isInt32():
      itemList[destIdx] = VMovB32(dst=vgpr(Holder(name="ValuC")),
                                                     src=vgpr("ValuC+%u"%srcIdx), comment="Rearrange MI out reg")
    elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
        accImOffset = accVgprImagNumOffset(kernel)
        cimod = Module()
        cimod.add(VMovB32(dst=vgpr(Holder(name="ValuC")), src=vgpr("ValuC+%u"%srcIdx), comment="Rearrange MI out reg"))
        cimod.add(VMovB32(dst=vgpr(Holder(name="ValuC+1")), src=vgpr("ValuC+%u"%(srcIdx+accImOffset)), comment="Rearrange MI out reg"))
        itemList[destIdx] = cimod
    elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
      accImOffset = accVgprImagNumOffset(kernel)
      cimod = Module()
      # tmp1 = a.real * b.real
      cimod.add(VLShiftLeftB64(dst=vgpr(Holder(name="ValuC"), 2), shiftHex=0, src=vgpr("ValuC+%u"%srcIdx,2), comment="Rearrange MI out reg"))
      # tmp2 = a.imag * b.real
      cimod.add(VLShiftLeftB64(dst=vgpr(Holder(name="ValuC+2"), 2), shiftHex=0, src=vgpr("ValuC+%u"%(srcIdx+accImOffset),2), comment="Rearrange MI out reg"))
      itemList[destIdx] = cimod

  imod = Module("MulAlpha")
  imod.setItems(itemList)
  return imod
