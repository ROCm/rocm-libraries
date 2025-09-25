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


from rocisa.code import Module, Label, ValueSet
from rocisa.container import vgpr, sgpr, SMEMModifiers, replaceHolder, EXEC,\
    VOP3PModifiers, ContinuousRegister
from rocisa.instruction import SAbsI32, SAddCU32, SAddI32, SAddU32, SAndB32, SBarrier, \
    SBranch, SBfmB32, SCBranchSCC0, SCBranchSCC1, SCMovB32, SCSelectB32, SCmpEQU32, SCmpEQU64, \
    SCmpGeI32, SCmpGeU32, SCmpGtI32, SCmpGtU32, SCmpLeU32, SCmpLtU32, SFf1B32, SFlbitI32B32, \
    SLShiftLeftB32, SLShiftLeftB64, SLShiftRightB32, SLoadB32, \
    SMaxU32, SMinU32, SMovB32, SMovB64, SMulI32, SNop, SSExtI16toI32, SSleep, SStoreB32, SSubU32, \
    SWaitCnt, VAddF32, VAddF64, VAddPKF16, VAddU32, VLShiftRightB32, VMovB32, \
    VReadfirstlaneB32, VReadlaneB32, VCvtBF16toFP32
from rocisa.functions import scalarStaticDivideAndRemainder, sMagicDiv2, \
    vectorStaticMultiply, BranchIfNotZero, scalarUInt32DivideAndRemainder, \
    vectorUInt32CeilDivideAndRemainder

from Tensile.Common import roundUp, log2, ceilDivide


def wgmXCC(writer, kernel, tmpSgprNumWorkGroups):
    module = Module("WGMXCC")
    module.addComment1("remap workgroup to XCCs")

    sgprWGM = "WGM"
    label_skipWGMXCC = Label(label="skip_WGMXCC", comment="skip WGMXCC if no enough WGs to remap")
    wgmDispatchMask = writer.states.WGMDispatchMask

    # Add option to skip XCC reorder
    if kernel["SpaceFillingAlgo"] > 0:
      sgprTmp = writer.sgprPool.checkOut(1)
      module.add(SAndB32(dst=sgpr(sgprTmp), src0=sgpr(sgprWGM), src1=hex(wgmDispatchMask), comment="Get XCC Reorder flag value"))
      module.add(SCmpEQU32(src0=sgpr(sgprTmp), src1=hex(wgmDispatchMask), comment="Check if general WGM is being requested"))
      module.add(SCBranchSCC1(labelName=label_skipWGMXCC.getLabelName(), comment=""))
      writer.sgprPool.checkIn(sgprTmp)

    with writer.allocTmpSgpr(6, 2) as tmpSgprRes:
      tmpSgpr      = tmpSgprRes.idx
      tmpSgpr0     = tmpSgpr+1
      tmpSgpr1     = tmpSgpr0+1
      tmpSgpr2     = tmpSgpr1+1
      WGMXCCSgpr   = tmpSgpr2+1
      CU_CountSgpr = WGMXCCSgpr+1

      module.add(SLShiftRightB32(dst=sgpr(WGMXCCSgpr), shiftHex=hex(16), src=sgpr(sgprWGM), comment="Get WGMXCC"))
      module.add(SFf1B32(dst=sgpr(WGMXCCSgpr), src=sgpr(WGMXCCSgpr), comment="Get log(WGMXCC)"))
      module.add(SLShiftRightB32(dst=sgpr(CU_CountSgpr), shiftHex=hex(22), src=sgpr(sgprWGM), comment="Get CU_Count"))

      module.addComment0("remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 )")
      module.add(SCmpGtI32(src0=sgpr(WGMXCCSgpr), src1=0))
      module.add(SCBranchSCC0(label_skipWGMXCC.getLabelName()))

      module.addComment0("only remap WGs in the range")
      tmpVgpr     = writer.vgprPool.checkOut(2)
      tmpVgprRes  = ContinuousRegister(tmpVgpr, 2)
      module.add(SLShiftRightB32(dst=sgpr(tmpSgpr0), shiftHex=sgpr(WGMXCCSgpr), src=sgpr(tmpSgprNumWorkGroups)))
      module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr0), shiftHex=sgpr(WGMXCCSgpr), src=sgpr(tmpSgpr0)))
      module.add(SCmpGeU32(src0=sgpr("WorkGroup0"), src1=sgpr(tmpSgpr0)))
      module.add(SCBranchSCC1(label_skipWGMXCC.getLabelName()))

      label_XCCG_nonzero = Label(label="XCCG_nonzero", comment="")
      module.add(SCmpEQU32(src0=sgpr(CU_CountSgpr), src1=0, comment="CU_Count == 0 ?"))
      module.add(SCBranchSCC0(label_XCCG_nonzero.getLabelName()))

      # CU_count == 0
      module.add(SLShiftRightB32(dst=sgpr(tmpSgpr0), shiftHex=sgpr(WGMXCCSgpr), src=sgpr("WorkGroup0")))
      module.add(SBfmB32(dst=sgpr(tmpSgpr1), src0=sgpr(WGMXCCSgpr), src1=0))
      module.add(SAndB32(dst=sgpr(tmpSgpr1), src0=sgpr("WorkGroup0"), src1=sgpr(tmpSgpr1)))
      module.add(SLShiftRightB32(dst=sgpr(tmpSgpr2), shiftHex=sgpr(WGMXCCSgpr), src=sgpr(tmpSgprNumWorkGroups)))
      module.add(SMulI32(dst=sgpr(tmpSgpr1), src0=sgpr(tmpSgpr1), src1=sgpr(tmpSgpr2)))
      module.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr(tmpSgpr0), src1=sgpr(tmpSgpr1)))
      module.add(SBranch(label_skipWGMXCC.getLabelName()))

      # CU_count > 0
      module.add(label_XCCG_nonzero)
      module.addComment0("temp0 = (wg//CU_Count)*CU_Count")
      module.add(scalarUInt32DivideAndRemainder(qReg=tmpSgpr0, dReg="WorkGroup0", divReg=CU_CountSgpr, rReg=tmpSgpr1, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True, comment="wg//CU_Count"))
      module.add(SMulI32(dst=sgpr(tmpSgpr0), src0=sgpr(tmpSgpr0), src1=sgpr(CU_CountSgpr)))
      module.addComment0("temp1 = (wg%CU_Count)//WGMXCC")
      module.add(SLShiftRightB32(dst=sgpr(tmpSgpr1), shiftHex=sgpr(WGMXCCSgpr), src=sgpr(tmpSgpr1)))
      module.addComment0("temp0 = temp0 + temp1")
      module.add(SAddU32(dst=sgpr(tmpSgpr0), src0=sgpr(tmpSgpr0), src1=sgpr(tmpSgpr1)))
      module.addComment0("temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC")
      module.add(scalarUInt32DivideAndRemainder(qReg=tmpSgpr1, dReg=tmpSgprNumWorkGroups, divReg=CU_CountSgpr, rReg=-1, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=False, comment="WGs//CU_Count"))
      module.add(SMulI32(dst=sgpr(tmpSgpr1), src0=sgpr(tmpSgpr1), src1=sgpr(CU_CountSgpr)))
      module.add(SSubU32(dst=sgpr(tmpSgpr2), src0=sgpr(tmpSgprNumWorkGroups), src1=sgpr(tmpSgpr1)))
      module.add(SCmpGtU32(src0=sgpr("WorkGroup0"), src1=sgpr(tmpSgpr1)))
      module.add(SCSelectB32(dst=sgpr(tmpSgpr1), src0=sgpr(tmpSgpr2), src1=sgpr(CU_CountSgpr)))
      module.add(SLShiftRightB32(dst=sgpr(tmpSgpr1), shiftHex=sgpr(WGMXCCSgpr), src=sgpr(tmpSgpr1)))
      module.add(SBfmB32(dst=sgpr(tmpSgpr2), src0=sgpr(WGMXCCSgpr), src1=0))
      module.add(SAndB32(dst=sgpr(tmpSgpr2), src0=sgpr("WorkGroup0"), src1=sgpr(tmpSgpr2)))
      writer.vgprPool.checkIn(tmpVgpr)
      module.add(SMulI32(dst=sgpr(tmpSgpr1), src0=sgpr(tmpSgpr1), src1=sgpr(tmpSgpr2)))
      module.addComment0("WorkGroup0 = temp0 + temp1")
      module.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr(tmpSgpr0), src1=sgpr(tmpSgpr1)))

      module.add(label_skipWGMXCC)

    return module


def DefaultWGM(writer, kernel, sgprWGM):
    module = Module("graWGMCalc")
    module.addComment0("WGM Calculation")
    # Restore WGM

    # We allocate a temp sgpr and keep sgpr[WGM] untouched.
    tmpWGM = writer.sgprPool.checkOut(1)

    module.add(SMovB32(dst=sgpr(tmpWGM), src=sgpr(sgprWGM), comment="Restore WGM"))
    module.add(SSExtI16toI32(dst=sgpr(tmpWGM), src=sgpr(tmpWGM), comment="Restore WGM"))

    wgmLabel         = Label(label=writer.labels.getNameInc("WGM"), comment="")
    wgmLabelPositive = Label(label=writer.labels.getNameInc("WGMPositive"), comment="")
    module.add(SCmpGtI32(src0=sgpr(tmpWGM), src1=1, comment="WGM > 1 ?"))
    module.add(SCBranchSCC1(labelName=wgmLabelPositive.getLabelName(), comment="branch if WGM > 1"))
    with writer.allocTmpSgprList(nums=[2,1,1]) as tmpSgprInfoList:
      wgmDivisor = tmpSgprInfoList[0].idx
      wgmDivisor2 = tmpSgprInfoList[0].idx + 1
      blockId2 = tmpSgprInfoList[1].idx
      wgSerial2 = tmpSgprInfoList[2].idx
      wgmDivisorMagicNumber = tmpSgprInfoList[0].idx + 1
      wgmAbs = tmpWGM
      tmpVgpr = writer.vgprPool.checkOut(2, "div")
      tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)

      # TODO: Unify this when sgpr is enough
      for wgmType in [True, False]: # Negative/Positive
        if wgmType:
          workgroupFirst = "WorkGroup1"
          workgroupSecond = "WorkGroup0"
          numWorkgroupsFirst = "NumWorkGroups1"
          numWorkgroupsSecond = "NumWorkGroups0"
        else:
          workgroupFirst = "WorkGroup0"
          workgroupSecond = "WorkGroup1"
          numWorkgroupsFirst = "NumWorkGroups0"
          numWorkgroupsSecond = "NumWorkGroups1"

        if not wgmType:
          module.add(wgmLabelPositive)
          module.add(SMovB32(dst=sgpr(wgmAbs), src=sgpr(tmpWGM), comment="WGM"))
        else:
          module.add(SCmpGeI32(src0=sgpr(tmpWGM), src1=0, comment="WGM >= 0 ?"))
          module.add(SCBranchSCC1(labelName=wgmLabel.getLabelName(), comment="branch if WGM >= 0"))
          module.add(SAbsI32(dst=sgpr(wgmAbs), src=sgpr(tmpWGM), comment="abs(WGM)"))
        # note this overwrites blockId2+1
        module.add(scalarUInt32DivideAndRemainder(qReg=blockId2, dReg=workgroupSecond, divReg=wgmAbs, rReg=wgSerial2, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=False, comment="WGM"))
        module.add(SMulI32(dst=sgpr(wgSerial2), src0=sgpr(blockId2), src1=sgpr(wgmAbs), comment="quotient * non-magic divisor"))
        module.add(SSubU32(dst=sgpr(wgSerial2), src0=sgpr(workgroupSecond), src1=sgpr(wgSerial2), comment="%s=remainder"%workgroupSecond))
        module.add(SMulI32(dst=sgpr(wgSerial2), src0=sgpr(wgSerial2), src1=sgpr(numWorkgroupsFirst), comment="(wg1 %% WGM)*%s"%numWorkgroupsFirst))
        module.add(SAddU32(dst=sgpr(wgSerial2), src0=sgpr(wgSerial2), src1=sgpr(workgroupFirst), comment="wgSerial = wg0 + (wg1 %% WGM)*%s"%numWorkgroupsFirst))

        module.add(scalarUInt32DivideAndRemainder(qReg=wgmDivisor, dReg=numWorkgroupsSecond, divReg=wgmAbs, rReg=wgSerial2, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=False, comment="WGM"))
        module.add(SMulI32(dst=sgpr(wgmDivisor2), src0=sgpr(wgmAbs), src1=sgpr(wgmDivisor), comment="quotient * non-magic divisor"))
        module.add(SSubU32(dst=sgpr(wgmDivisorMagicNumber), src0=sgpr(numWorkgroupsSecond), src1=sgpr(wgmDivisor2), comment="%s=remainder"%numWorkgroupsSecond))
        module.add(SCmpEQU32(src0=sgpr(wgmDivisorMagicNumber), src1=0, comment="remainder == 0 ?"))
        module.add(SCMovB32(dst=sgpr(wgmDivisorMagicNumber), src=sgpr(wgmAbs), comment="remainder = WGM if remainder == 0"))

        module.add(SCmpGeU32(src0=sgpr(blockId2), src1=sgpr(wgmDivisor), comment="blockId >= numFullBlocks ?"))
        module.add(SCSelectB32(dst=sgpr(wgmDivisor), src0=sgpr(wgmDivisorMagicNumber), src1=sgpr(wgmAbs)))

        # For WGM >= 1
        # WorkGroup0 = wgSerial2 / wgmDivisor
        # WorkGroup1 = wgSerial2 - (wgmDivisor * WorkGroup0)
        module.add(scalarUInt32DivideAndRemainder(qReg=workgroupFirst, dReg=wgSerial2, divReg=wgmDivisor, rReg=workgroupSecond, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"]))
        module.add(SMulI32(dst=sgpr(workgroupSecond), src0=sgpr(workgroupFirst), src1=sgpr(wgmDivisor), comment="quotient * non-magic divisor"))
        module.add(SSubU32(dst=sgpr(workgroupSecond), src0=sgpr(wgSerial2), src1=sgpr(workgroupSecond), comment="%s=remainder"%workgroupSecond))
        module.add(SMulI32(dst=sgpr(blockId2), src0=sgpr(blockId2), src1=sgpr(wgmAbs), comment="blockId * WGM"))
        module.add(SAddU32(dst=sgpr(workgroupSecond), src0=sgpr(workgroupSecond), src1=sgpr(blockId2), comment="wg1 += blockId * WGM"))
        if wgmType:
          module.add(SBranch(wgmLabel.getLabelName()))

    module.add(wgmLabel)

    writer.sgprPool.checkIn(tmpWGM)
    tmpVgprRes = None
    writer.vgprPool.checkIn(tmpVgpr)

    return module


def SpaceFillingCurveWalk(writer, kernel, sgprWGM):
    module = Module("remapSpaceFillingCurveWalk")

    # TODO: Query arch specific values instead of hard code
    numTotalCU = 256
    numXCC = 8
    sgprWGID = "WorkGroup0"

    if not kernel["StreamK"]:
      sgprTmp = writer.sgprPool.checkOut(1)
      # Recompute the 1D ID from 2D IDs
      module.add(SMulI32(dst=sgpr(sgprTmp), src0=sgpr("WorkGroup1"), src1=sgpr("NumWorkGroups0"), comment=""))
      module.add(SAddU32(dst=sgpr(sgprWGID), src0=sgpr("WorkGroup0"), src1=sgpr(sgprTmp), comment=""))
      writer.sgprPool.checkIn(sgprTmp)
    else:
      module.add(SMovB32(dst=sgpr(sgprWGID), src=sgpr("StreamKTileID"), comment=""))

    # Num WGs in M, N directions
    sgprNumTilesM = writer.sgprPool.checkOut(1)
    sgprNumTilesN = writer.sgprPool.checkOut(1)

    module.add(SMovB32(dst=sgpr(sgprNumTilesM), src=sgpr("NumWorkGroups0"), comment=""))
    module.add(SMovB32(dst=sgpr(sgprNumTilesN), src=sgpr("NumWorkGroups1"), comment=""))

    # Apply XCC remap
    useXCCRemap = kernel["SpaceFillingAlgo"] > 0

    if useXCCRemap:
      tmpSgpr = []
      numTmpSgpr = 3 + 2 + 1 + 2
      for i in range(0, numTmpSgpr):
          tmpSgpr.append(writer.sgprPool.checkOut(1))

      sgprXcc = writer.sgprPool.checkOut(1)

      module.add(SAndB32(dst=sgpr(sgprXcc), src0=sgpr(sgprWGM), src1="0x00010000", comment="Get XCC Reorder flag value"))
      labelDbg = Label(writer.labels.getUniqueNamePrefix("xccdbg"), comment="")
      module.add(labelDbg)
      module.add(SCmpEQU32(src0=sgpr(sgprXcc), src1=0, comment="sgprXCC == 0?"))
      labelSkip = Label(writer.labels.getUniqueNamePrefix("SkipXCCReorder"), comment="")
      module.add(SCBranchSCC1(labelName=labelSkip.getLabelName(), comment=""))

      module.addComment0("Remap 1D based on XCC")
      numWG = tmpSgpr[0] # alias
      module.add(SMulI32(dst=sgpr(numWG), src0=sgpr("NumWorkGroups0"), src1=sgpr("NumWorkGroups1")))
      module.add(SAddU32(dst=sgpr(tmpSgpr[1]), src0=sgpr(numWG), src1=(numTotalCU - 1)))
      numRd = tmpSgpr[1] # alias, number of rounds
      curRd = tmpSgpr[2] # alias, current round
    # Need to use this for non power of 2 total CUs
   #module.add(scalarUInt32DivideAndRemainder(qReg=tmpSgpr.idx, dReg="WorkGroup0", divReg=tmpSgpr.idx, rReg=tmpSgpr.idx+1,\
     #                                     tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=False))
      module.add(SLShiftRightB32(dst=sgpr(numRd), src=sgpr(numRd), shiftHex=log2(numTotalCU), comment="Calc number rounds"))
      module.add(SLShiftRightB32(dst=sgpr(curRd), src=sgpr(sgprWGID), shiftHex=log2(numTotalCU), comment="Calc current rounds"))
      module.addComment0("")
      # tmp sgpr 3, 4, 5
      module.add(SMinU32(dst=sgpr(tmpSgpr[3]), src0=sgpr(numWG), src1=numTotalCU, comment="min(numwg, %u)"%(numTotalCU) ))
      module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr[4]), src=sgpr(curRd), shiftHex=log2(numTotalCU), comment=" current round * %u"%(numTotalCU)))
      module.add(SSubU32(dst=sgpr(tmpSgpr[4]), src0=sgpr(numWG), src1=sgpr(tmpSgpr[4]), comment="numwg - cr * %u"%(numTotalCU)))
      module.addComment0("nwg = cr < nr - 1 ? std::min<int>(%u, nwg) : nwg - (cr) * %u"%(numTotalCU, numTotalCU))
      module.add(SSubU32(dst=sgpr(tmpSgpr[5]), src0=sgpr(numRd), src1=1, comment="num rounds - 1"))
      module.add(SCmpLtU32(sgpr(curRd), sgpr(tmpSgpr[5]), comment="current round < num rounds - 1"))
      module.add(SCSelectB32(dst=sgpr(numWG), src0=sgpr(tmpSgpr[3]), src1=sgpr(tmpSgpr[4])))
      module.addComment0("")
      # tmp sgpr 3, 4, 5
      module.add(SAndB32(dst=sgpr(sgprWGID), src0=sgpr(sgprWGID), src1=(numTotalCU - 1)))
      module.add(SLShiftRightB32(dst=sgpr(tmpSgpr[3]), src=sgpr(numWG), shiftHex=log2(numXCC), comment="num wg per xcc"))
      module.add(SAndB32(dst=sgpr(tmpSgpr[4]), src0=sgpr(numWG), src1=(numXCC - 1), comment="num xcc with extra wg"))
      module.addComment0("")
      module.add(SAndB32(dst=sgpr(tmpSgpr[5]), src0=sgpr(sgprWGID), src1=(numXCC - 1), comment="logical xcc id"))
      numWGPerXCC = tmpSgpr[3] # alias
      numXCCExtraWG = tmpSgpr[4] # alias
      logicalXCCID = tmpSgpr[5] # alias
      # tmp sgpr 0, 1, 6, 7 numWG, numRd not needed anymore
      module.add(SMinU32(dst=sgpr(tmpSgpr[1]), src0=sgpr(logicalXCCID), src1=sgpr(numXCCExtraWG), comment="min(cutoff, xccid)" ))
      module.add(SSubU32(dst=sgpr(tmpSgpr[6]), src0=sgpr(logicalXCCID), src1=sgpr(tmpSgpr[1]), comment="xccid - min(cutoff, xccid)"))
      module.addComment0("")
      module.add(SMulI32(dst=sgpr(tmpSgpr[7]), src0=sgpr(tmpSgpr[6]), src1=sgpr(numWGPerXCC)))
      module.add(SAddU32(dst=sgpr(numWGPerXCC), src0=sgpr(numWGPerXCC), src1=1))
      module.add(SMulI32(dst=sgpr(tmpSgpr[0]), src0=sgpr(tmpSgpr[1]), src1=sgpr(numWGPerXCC)))
      module.add(SAddU32(dst=sgpr(tmpSgpr[0]), src0=sgpr(tmpSgpr[0]), src1=sgpr(tmpSgpr[7])))
      module.addComment0("")
      module.add(SLShiftRightB32(dst=sgpr(numWGPerXCC), src=sgpr(sgprWGID), shiftHex=log2(numXCC), comment=""))
      module.add(SAddU32(dst=sgpr(sgprWGID), src0=sgpr(tmpSgpr[0]), src1=sgpr(numWGPerXCC)))

      module.add(SLShiftLeftB32(dst=sgpr(curRd), src=sgpr(curRd), shiftHex=log2(numTotalCU), comment=""))
      module.add(SAddU32(dst=sgpr(sgprWGID), src0=sgpr(sgprWGID), src1=sgpr(curRd)))
      for i in range(0, numTmpSgpr):
        writer.sgprPool.checkIn(tmpSgpr[i])
      writer.sgprPool.checkIn(sgprXcc)
      module.add(labelSkip)

    # Starting (M,N) offset
    sgprXOffset       = writer.sgprPool.checkOut(1)
    sgprYOffset       = writer.sgprPool.checkOut(1)

    # Default offsets
    module.add(SMovB32(dst=sgpr(sgprXOffset), src=0))
    module.add(SMovB32(dst=sgpr(sgprYOffset), src=0))

    # Use space-filling curve to generate new WG IDs
    module.add(SpaceFillCurveSimpleImpl(writer, kernel, sgprWGM, sgprWGID, sgprNumTilesM, sgprNumTilesN, sgprXOffset, sgprYOffset))

    writer.sgprPool.checkIn(sgprXOffset)
    writer.sgprPool.checkIn(sgprYOffset)

    writer.sgprPool.checkIn(sgprNumTilesM)
    writer.sgprPool.checkIn(sgprNumTilesN)

    return module

def SpaceFillCurveSimpleImpl(writer, kernel, sgprWGM, sgprWGID, sgprNumTilesM, sgprNumTilesN, sgprXOffset, sgprYOffset):

    defaultBlkM = 8
    defaultBlkN = 4

    module = Module()

    tmpSgprBlockM       = writer.sgprPool.checkOut(1)
    tmpSgprBlockN       = writer.sgprPool.checkOut(1)
    tmpSgprBlockSz      = writer.sgprPool.checkOut(1)
    tmpSgprCurDir       = writer.sgprPool.checkOut(1)


    sgprNumTilesM1       = writer.sgprPool.checkOut(1)
    sgprNumTilesN1       = writer.sgprPool.checkOut(1)
    sgprNumTilesM2       = writer.sgprPool.checkOut(1)
    sgprNumTilesN2       = writer.sgprPool.checkOut(1)

    block0 = [sgprNumTilesM1, sgprNumTilesN1]
    block1 = [sgprNumTilesM2, sgprNumTilesN1]
    block2 = [sgprNumTilesM1, sgprNumTilesN2]
    block3 = [sgprNumTilesM2, sgprNumTilesN2]

    def blockXYOffset(block):
      if block == block0:
        return [0,0]
      elif block == block1:
        return [sgprNumTilesM1, 0]
      elif block == block2:
        return [0, sgprNumTilesN1]
      elif block == block3:
        return [sgprNumTilesM1, sgprNumTilesN1]

    directions = []
    directionBlocks = []
    directionNewDir = []
    directionLabels = []

    if kernel["SpaceFillingAlgo"] == 1 or kernel["SpaceFillingAlgo"] == 5:
      directions = [
        "HilbertWalkNCC",
        "HilbertWalkN",
        "HilbertWalkSCC",
        "HilbertWalkS",
        "HilbertWalkECC",
        "HilbertWalkE",
        "HilbertWalkWCC",
        "HilbertWalkW",
      ]
      directionBlocks = [
        [block0, block1, block3, block2], # NCC
        [block2, block3, block1, block0], # N
        [block3, block2, block0, block1], # SCC
        [block1, block0, block2, block3], # S
        [block2, block0, block1, block3], # ECC
        [block3, block1, block0, block2], # E
        [block0, block1, block3, block2], # WCC
        [block1, block3, block2, block0], # W
      ]
      directionNewDir = [
        [directions[7], directions[0], directions[0], directions[5]], #NCC
        [directions[4], directions[1], directions[1], directions[6]], #N
        [directions[6], directions[2], directions[2], directions[4]], #SCC
        [directions[5], directions[3], directions[3], directions[7]], #S
        [directions[1], directions[4], directions[4], directions[3]], #ECC
        [directions[2], directions[5], directions[5], directions[0]], #E
        [directions[3], directions[6], directions[6], directions[1]], #WCC
        [directions[0], directions[7], directions[7], directions[2]], #W
      ]
    elif kernel["SpaceFillingAlgo"] == 2:
      # Z-Walk
      directions = ["MortonZ"]
      directionBlocks = [
        [block0, block2, block1, block3],
      ]
    elif kernel["SpaceFillingAlgo"] == 3:
      # ReverseN-Walk
      directions = ["MortonRN"]
      directionBlocks = [
        [block0, block1, block2, block3],
      ]
    elif kernel["SpaceFillingAlgo"] == 4:
      # U-Walk
      directions = ["MortonU"]
      directionBlocks = [
        [block0, block1, block3, block2],
      ]

    for i in range(0, len(directions)):
      directionLabels.append(Label((writer.labels.getUniqueNamePrefix(directions[i])), comment=""))

    for i in range(0, len(directions)):
      module.add(ValueSet(directions[i], i, format=1))

    tmpSgpr = []
    numTmpSgpr = 3
    for i in range(0, numTmpSgpr):
      tmpSgpr.append(writer.sgprPool.checkOut(1))

    module.add(SAndB32(dst=sgpr(tmpSgprBlockM), src0=sgpr(sgprWGM), src1="0x000000ff", comment="Get BLKM value"))
    module.add(SAndB32(dst=sgpr(tmpSgprBlockN), src0=sgpr(sgprWGM), src1="0x0000ff00", comment="Get BLKM value"))
    module.add(SLShiftRightB32(dst=sgpr(tmpSgprBlockN), src=sgpr(tmpSgprBlockN), shiftHex=8, comment="Get BLKM value"))

    module.add(SCmpEQU32(src0=sgpr(tmpSgprBlockM), src1=0, comment="blkM == 0?"))
    module.add(SCSelectB32(dst=sgpr(tmpSgprBlockM), src0=defaultBlkM , src1=sgpr(tmpSgprBlockM), comment=""))
    module.add(SCmpEQU32(src0=sgpr(tmpSgprBlockN), src1=0, comment="blkN == 0?"))
    module.add(SCSelectB32(dst=sgpr(tmpSgprBlockN), src0=defaultBlkN , src1=sgpr(tmpSgprBlockN), comment=""))

    if len(directions) > 1:
      module.add(SMovB32(dst=sgpr(tmpSgprCurDir), src=directions[0]))
    module.add(SMulI32(dst=sgpr(tmpSgprBlockSz), src0=sgpr(tmpSgprBlockM), src1=sgpr(tmpSgprBlockN)))

    # Begin SFC

    labelStartWhile = Label(writer.labels.getUniqueNamePrefix("SpaceFillingCurveWalkStartWhile"), comment="")
    labelEndWhile = Label(writer.labels.getUniqueNamePrefix("SpaceFillingCurveWalkEndWhile"), comment="")

    module.add(labelStartWhile)
    #module.addComment0("start while")
    module.add(SMulI32(dst=sgpr(tmpSgpr[0]), src0=sgpr(sgprNumTilesM), src1=sgpr(sgprNumTilesN)))
    module.add(SCmpGtU32(src0=sgpr(tmpSgpr[0]), src1=sgpr(tmpSgprBlockSz), comment="M * N > bkM * bkN"))
    module.add(SCBranchSCC0(labelName=labelEndWhile.getLabelName(), comment=""))

    # body of loop
    module.add(SFlbitI32B32(dst=sgpr(tmpSgpr[0]), src=sgpr(sgprNumTilesM), comment=""))
    module.add(SFlbitI32B32(dst=sgpr(tmpSgpr[1]), src=sgpr(sgprNumTilesN), comment=""))
    module.add(SSubU32(dst=sgpr(tmpSgpr[0]), src0=31, src1=sgpr(tmpSgpr[0]), comment=""))
    module.add(SSubU32(dst=sgpr(tmpSgpr[1]), src0=31, src1=sgpr(tmpSgpr[1]), comment=""))
    module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr[0]), src=1, shiftHex=sgpr(tmpSgpr[0]), comment=""))
    module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr[1]), src=1, shiftHex=sgpr(tmpSgpr[1]), comment=""))

    module.add(SLShiftRightB32(dst=sgpr(sgprNumTilesM1), src=sgpr(sgprNumTilesM), shiftHex=1, comment="M1 = M / 2"))
    module.add(SLShiftRightB32(dst=sgpr(sgprNumTilesN1), src=sgpr(sgprNumTilesN), shiftHex=1, comment="N1 = N / 2"))
    module.add(SCmpEQU32(src0=sgpr(tmpSgpr[0]), src1=sgpr(sgprNumTilesM), comment=""))
    module.add(SCSelectB32(dst=sgpr(sgprNumTilesM1), src0=sgpr(sgprNumTilesM1), src1=sgpr(tmpSgpr[0])))
    module.add(SCmpEQU32(src0=sgpr(tmpSgpr[1]), src1=sgpr(sgprNumTilesN), comment=""))
    module.add(SCSelectB32(dst=sgpr(sgprNumTilesN1), src0=sgpr(sgprNumTilesN1), src1=sgpr(tmpSgpr[1])))

    module.add(SMinU32(dst=sgpr(sgprNumTilesM2), src0=sgpr(sgprNumTilesM), src1=sgpr(tmpSgprBlockM), comment="" ))
    module.add(SMinU32(dst=sgpr(sgprNumTilesN2), src0=sgpr(sgprNumTilesN), src1=sgpr(tmpSgprBlockN), comment="" ))
    module.add(SMaxU32(dst=sgpr(sgprNumTilesM1), src0=sgpr(sgprNumTilesM1), src1=sgpr(sgprNumTilesM2), comment="" ))
    module.add(SMaxU32(dst=sgpr(sgprNumTilesN1), src0=sgpr(sgprNumTilesN1), src1=sgpr(sgprNumTilesN2), comment="" ))

    module.add(SSubU32(dst=sgpr(sgprNumTilesM2), src0=sgpr(sgprNumTilesM), src1=sgpr(sgprNumTilesM1), comment=""))
    module.add(SSubU32(dst=sgpr(sgprNumTilesN2), src0=sgpr(sgprNumTilesN), src1=sgpr(sgprNumTilesN1), comment=""))

    if len(directions) > 1:
      for i in range(0, len(directions)):
        module.add(SCmpEQU32(src0=sgpr(tmpSgprCurDir), src1=directions[i], comment=""))
        module.add(SCBranchSCC1(labelName=directionLabels[i].getLabelName()))

    for i in range(0, len(directions)):
      if len(directions) > 1:
        module.add(directionLabels[i])
      block = directionBlocks[i]
      module.add(SMulI32(dst=sgpr(tmpSgpr[0]), src0=sgpr(block[0][0]), src1=sgpr(block[0][1])))
      module.add(SMulI32(dst=sgpr(tmpSgpr[1]), src0=sgpr(block[1][0]), src1=sgpr(block[1][1])))
      module.add(SMulI32(dst=sgpr(tmpSgpr[2]), src0=sgpr(block[2][0]), src1=sgpr(block[2][1])))
      module.add(SAddU32(dst=sgpr(tmpSgpr[1]), src0=sgpr(tmpSgpr[1]), src1=sgpr(tmpSgpr[0])))
      module.add(SAddU32(dst=sgpr(tmpSgpr[2]), src0=sgpr(tmpSgpr[2]), src1=sgpr(tmpSgpr[1])))
      for j in range(0, 4):
        label = Label((writer.labels.getUniqueNamePrefix(directions[i]+"_block%u"%(j+1))), comment="")
        if j < 3:
          module.add(SCmpLtU32(sgpr(sgprWGID), sgpr(tmpSgpr[j]), comment=""))
          module.add(SCBranchSCC0(labelName=label.getLabelName(), comment=""))
        module.add(SMovB32(dst=sgpr(sgprNumTilesM), src=sgpr(block[j][0]), comment="Update M"))
        module.add(SMovB32(dst=sgpr(sgprNumTilesN), src=sgpr(block[j][1]), comment="Update N"))
        curBlockOffset = blockXYOffset(directionBlocks[i][j])
        if curBlockOffset[0] != 0:
          module.add(SAddU32(dst=sgpr(sgprXOffset), src0=sgpr(sgprXOffset), src1=sgpr(curBlockOffset[0]), comment="Update idX"))
        if curBlockOffset[1] != 0:
          module.add(SAddU32(dst=sgpr(sgprYOffset), src0=sgpr(sgprYOffset), src1=sgpr(curBlockOffset[1]), comment="Update idY"))
        if len(directions) > 1:
          module.add(SMovB32(dst=sgpr(tmpSgprCurDir), src=directionNewDir[i][j], comment="Update direction"))
        if j > 0:
          module.add(SSubU32(dst=sgpr(sgprWGID), src0=sgpr(sgprWGID), src1=sgpr(tmpSgpr[j-1]), comment="Update serial idx"))
        module.add(SBranch(labelName=labelStartWhile.getLabelName()))
        if j < 3:
          module.add(label)


    #module.addComment0("end while")
    module.add(labelEndWhile)
    tmpVgpr = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr")
    tmpVgprRes = ContinuousRegister(tmpVgpr, 2)
    module.add(scalarUInt32DivideAndRemainder(qReg=tmpSgpr[0], dReg=sgprWGID, divReg=sgprNumTilesN, rReg=tmpSgpr[1], tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True, comment=""))
    module.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr(sgprXOffset), src1=sgpr(tmpSgpr[0]), comment=""))
    module.add(SAddU32(dst=sgpr("WorkGroup1"), src0=sgpr(sgprYOffset), src1=sgpr(tmpSgpr[1]), comment=""))

    writer.vgprPool.checkIn(tmpVgpr)
    for i in range(0, numTmpSgpr):
      writer.sgprPool.checkIn(tmpSgpr[i])

    writer.sgprPool.checkIn(tmpSgprBlockM)
    writer.sgprPool.checkIn(tmpSgprBlockN)
    writer.sgprPool.checkIn(tmpSgprBlockSz)
    writer.sgprPool.checkIn(tmpSgprCurDir)
    writer.sgprPool.checkIn(sgprNumTilesM1)
    writer.sgprPool.checkIn(sgprNumTilesN1)
    writer.sgprPool.checkIn(sgprNumTilesM2)
    writer.sgprPool.checkIn(sgprNumTilesN2)

    return module
