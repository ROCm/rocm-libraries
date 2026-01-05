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
    SBranch, SBfmB32, SCBranchSCC0, SCBranchSCC1, SCMovB32, SCSelectB32, SCSelectB64, SCmpEQU32, SCmpEQU64, \
    SCmpGeI32, SCmpGeU32, SCmpGtI32, SCmpGtU32, SCmpLeU32, SCmpLtU32, SFf1B32, SFlbitI32B32, \
    SLShiftLeftB32, SLShiftLeftB64, SLShiftRightB32, SLoadB32, \
    SMaxU32, SMinU32, SMovB32, SMovB64, SMulI32, SNop, SOrB32, SSExtI16toI32, SSleep, SStoreB32, SSubU32, \
    SWaitCnt, VAddF32, VAddF64, VAddPKF16, VAddU32, VLShiftRightB32, VMovB32, VMovB64, VMulF32, VMulF64, \
    VReadfirstlaneB32, VReadlaneB32, VWritelaneB32, VCvtBF16toFP32, VCvtU32toF32, VCvtU32toF64, VRcpIFlagF32, \
    VRcpF64, VCvtF32toU32, VCvtF64toU32, VMulU32U24, VMulLOU32, VSubU32, VCmpXGeU32, VCmpXGtU32, VCmpXEqU32
from rocisa.functions import scalarStaticDivideAndRemainder, sMagicDiv2, \
    vectorStaticMultiply, BranchIfNotZero, scalarUInt24DivideAndRemainder, \
    vectorUInt32CeilDivideAndRemainder

from Tensile.Common import roundUp, log2, ceilDivide

def scalarUInt24DivideAndRemainderPair(qReg, dReg, divReg, rReg, tmpVgprRes, wavewidth, doRemainder=True, doQuotient=True):

    module = Module()
    tmpVgpr = tmpVgprRes.idx
    tmpVgpr1 = tmpVgprRes.idx+2
    divRegVgpr = tmpVgprRes.idx+4
    dRegVgpr = tmpVgprRes.idx+5
    module.addComment0("Start of pair-wise integer division. [s%s / s%s, s%s / s%s]"%(str(dReg[0]), str(divReg[0]), str(dReg[1]), str(divReg[1])))
    module.add(VMovB32(dst=vgpr(tmpVgpr), src=sgpr(divReg[0]), comment=""))
    module.add(VWritelaneB32(dst=vgpr(tmpVgpr), src0=sgpr(divReg[1]), src1=1, comment=""))
    module.add(VMovB32(dst=vgpr(tmpVgpr1), src=sgpr(dReg[0]), comment=""))
    module.add(VWritelaneB32(dst=vgpr(tmpVgpr1), src0=sgpr(dReg[1]), src1=1, comment=""))

    module.add(VMovB32(dst=vgpr(divRegVgpr), src=vgpr(tmpVgpr), comment=""))
    module.add(VMovB32(dst=vgpr(dRegVgpr), src=vgpr(tmpVgpr1), comment=""))

    module.add(VCvtU32toF64(dst=vgpr(tmpVgpr,2), src=vgpr(tmpVgpr)))
    module.add(VRcpF64(dst=vgpr(tmpVgpr,2), src=vgpr(tmpVgpr,2)))
    module.add(VCvtU32toF64(dst=vgpr(tmpVgpr1,2), src=vgpr(tmpVgpr1)))
    module.add(VMulF64(dst=vgpr(tmpVgpr,2), src0=vgpr(tmpVgpr,2), src1=vgpr(tmpVgpr1,2)))
    module.add(VCvtF64toU32(dst=vgpr(tmpVgpr), src=vgpr(tmpVgpr,2)))

    module.add(VMulLOU32(dst=vgpr(tmpVgprRes.idx+1), src0=vgpr(tmpVgpr), src1=vgpr(divRegVgpr), comment=""))
    module.add(VSubU32(dst=vgpr(tmpVgpr1), src0=vgpr(dRegVgpr), src1=vgpr(tmpVgprRes.idx+1)))
    module.add(VCmpXGeU32(EXEC(), vgpr(tmpVgpr1), vgpr(divRegVgpr)))
    module.add(VAddU32(dst=vgpr(tmpVgpr), src0=vgpr(tmpVgpr), src1=1))

    if wavewidth == 64:
        module.add(SMovB64(dst=EXEC(), src=-1, comment="reset mask"))
    else:
        module.add(SMovB32(dst=EXEC(), src=-1, comment="reset mask"))

    if doRemainder:
        module.add(VMulLOU32(dst=vgpr(tmpVgprRes.idx+1), src0=vgpr(tmpVgpr), src1=vgpr(divRegVgpr), comment=""))
        module.add(VSubU32(dst=vgpr(tmpVgpr1), src0=vgpr(dRegVgpr), src1=vgpr(tmpVgprRes.idx+1)))

    if doQuotient:
        module.addComment0("Store quotients to [s%u, s%u]"%(qReg[0], qReg[1]))
        module.add(VReadlaneB32(dst=sgpr(qReg[0]), src0=vgpr(tmpVgpr), src1=0, comment=""))
        module.add(VReadlaneB32(dst=sgpr(qReg[1]), src0=vgpr(tmpVgpr), src1=1, comment=""))
    else:
        module.add(SNop(waitState=0, comment=""))

    if doRemainder:
        module.addComment0("Store remainders to [s%u, s%u]"%(rReg[0], rReg[1]))
        module.add(VReadlaneB32(dst=sgpr(rReg[0]), src0=vgpr(tmpVgpr1), src1=0, comment=""))
        module.add(VReadlaneB32(dst=sgpr(rReg[1]), src0=vgpr(tmpVgpr1), src1=1, comment=""))

    module.addComment0("End of pair-wise integer division. [s%s / s%s, s%s / s%s]"%(str(dReg[0]), str(divReg[0]), str(dReg[1]), str(divReg[1])))
    return module

def wgmXCC(writer, kernel, tmpSgprNumWorkGroups):
    module = Module("WGMXCC")
    module.addComment1("remap workgroup to XCCs")

    sgprWGM = "WGM"
    label_skipWGMXCC = Label(label="skip_WGMXCC", comment="skip WGMXCC if no enough WGs to remap")

    if(kernel["StreamK"] != 0 and kernel["WorkGroupMappingXCC"] == -1):
        """
        Formula:
        x, g   = divmod(old_wg, WGMXCC)
        q, r   = divmod(WG, WGMXCC)
        group  = q + (1 if g < r else 0)
        offset = 0 if g < r else r
        new    = x + offset + g * group
        """
        with writer.allocTmpSgpr(6, 2) as tmpSgprRes:
            SgprWGMXCC = tmpSgprRes.idx
            SgprX      = tmpSgprRes.idx + 1
            SgprG      = tmpSgprRes.idx + 2
            SgprQ      = tmpSgprRes.idx + 3
            SgprR      = tmpSgprRes.idx + 4
            SgprO      = tmpSgprRes.idx + 5
            # Reuse some sgprs
            tmpSgpr = SgprWGMXCC
            group   = SgprQ
            # offset  = SgprR
            
            tmpVgpr     = writer.vgprPool.checkOutAligned(4,2)
            tmpVgprRes  = ContinuousRegister(tmpVgpr, 4)

            module.add(SLShiftRightB32(dst=sgpr(SgprWGMXCC), shiftHex=hex(16), src=sgpr(sgprWGM), comment="Get WGMXCC"))
            module.add(SAndB32(dst=sgpr(SgprWGMXCC), src0=sgpr(SgprWGMXCC), src1=hex(63), comment="Get WGMXCC"))
            module.addComment0("remap WGs if WGMXCC > 1")
            module.add(SCmpGtU32(src0=sgpr(SgprWGMXCC), src1=1))
            module.add(SCBranchSCC0(label_skipWGMXCC.getLabelName()))
            module.addComment0("divmod(old_wg, WGMXCC)")
            module.add(scalarUInt24DivideAndRemainder(qReg=SgprX, rReg=SgprG, dReg="WorkGroup0", divReg=SgprWGMXCC, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True))
            module.add(SWaitCnt(kmcnt=0, comment="wait for args to load"))
            module.addComment0("divmod(WG, WGMXCC)")
            module.add(scalarUInt24DivideAndRemainder(qReg=SgprQ, rReg=SgprR, dReg="skGrid", divReg=SgprWGMXCC, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True))            
            writer.vgprPool.checkIn(tmpVgpr)
            # Check if current group requires a remainder WG or not
            module.add(SCmpLtU32(src0=sgpr(SgprG), src1=sgpr(SgprR)))
            module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=hex(1), src1=hex(0), comment="Select multiplier"))
            module.add(SCSelectB32(dst=sgpr(SgprO), src0=hex(0), src1=sgpr(SgprR), comment="Select remainder"))
            module.add(SAddU32(dst=sgpr(group), src0=sgpr(SgprQ), src1=sgpr(tmpSgpr), comment="Adjust multiplier"))
            # Assemble everything
            module.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr(SgprX), src1=sgpr(SgprO)))
            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(SgprG), src1=sgpr(group)))
            module.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(tmpSgpr)))

            module.add(label_skipWGMXCC)
    else:
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
            tmpVgpr     = writer.vgprPool.checkOutAligned(4,2)
            tmpVgprRes  = ContinuousRegister(tmpVgpr, 4)
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
            module.add(scalarUInt24DivideAndRemainder(qReg=tmpSgpr0, dReg="WorkGroup0", divReg=CU_CountSgpr, rReg=tmpSgpr1, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True))
            module.add(SMulI32(dst=sgpr(tmpSgpr0), src0=sgpr(tmpSgpr0), src1=sgpr(CU_CountSgpr)))
            module.addComment0("temp1 = (wg%CU_Count)//WGMXCC")
            module.add(SLShiftRightB32(dst=sgpr(tmpSgpr1), shiftHex=sgpr(WGMXCCSgpr), src=sgpr(tmpSgpr1)))
            module.addComment0("temp0 = temp0 + temp1")
            module.add(SAddU32(dst=sgpr(tmpSgpr0), src0=sgpr(tmpSgpr0), src1=sgpr(tmpSgpr1)))
            module.addComment0("temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC")
            module.add(scalarUInt24DivideAndRemainder(qReg=tmpSgpr1, dReg=tmpSgprNumWorkGroups, divReg=CU_CountSgpr, rReg=-1, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=False))
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
        tmpVgpr = writer.vgprPool.checkOutAligned(4, 2, "div")
        tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=4)

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
            module.add(scalarUInt24DivideAndRemainder(qReg=blockId2, dReg=workgroupSecond, divReg=wgmAbs, rReg=wgSerial2, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=False))
            module.add(SMulI32(dst=sgpr(wgSerial2), src0=sgpr(blockId2), src1=sgpr(wgmAbs), comment="quotient * non-magic divisor"))
            module.add(SSubU32(dst=sgpr(wgSerial2), src0=sgpr(workgroupSecond), src1=sgpr(wgSerial2), comment="%s=remainder"%workgroupSecond))
            module.add(SMulI32(dst=sgpr(wgSerial2), src0=sgpr(wgSerial2), src1=sgpr(numWorkgroupsFirst), comment="(wg1 %% WGM)*%s"%numWorkgroupsFirst))
            module.add(SAddU32(dst=sgpr(wgSerial2), src0=sgpr(wgSerial2), src1=sgpr(workgroupFirst), comment="wgSerial = wg0 + (wg1 %% WGM)*%s"%numWorkgroupsFirst))

            module.add(scalarUInt24DivideAndRemainder(qReg=wgmDivisor, dReg=numWorkgroupsSecond, divReg=wgmAbs, rReg=wgSerial2, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=False))
            module.add(SMulI32(dst=sgpr(wgmDivisor2), src0=sgpr(wgmAbs), src1=sgpr(wgmDivisor), comment="quotient * non-magic divisor"))
            module.add(SSubU32(dst=sgpr(wgmDivisorMagicNumber), src0=sgpr(numWorkgroupsSecond), src1=sgpr(wgmDivisor2), comment="%s=remainder"%numWorkgroupsSecond))
            module.add(SCmpEQU32(src0=sgpr(wgmDivisorMagicNumber), src1=0, comment="remainder == 0 ?"))
            module.add(SCMovB32(dst=sgpr(wgmDivisorMagicNumber), src=sgpr(wgmAbs), comment="remainder = WGM if remainder == 0"))

            module.add(SCmpGeU32(src0=sgpr(blockId2), src1=sgpr(wgmDivisor), comment="blockId >= numFullBlocks ?"))
            module.add(SCSelectB32(dst=sgpr(wgmDivisor), src0=sgpr(wgmDivisorMagicNumber), src1=sgpr(wgmAbs)))

            # For WGM >= 1
            # WorkGroup0 = wgSerial2 / wgmDivisor
            # WorkGroup1 = wgSerial2 - (wgmDivisor * WorkGroup0)
            module.add(scalarUInt24DivideAndRemainder(qReg=workgroupFirst, dReg=wgSerial2, divReg=wgmDivisor, rReg=workgroupSecond, tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True))
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

# Remap 1D workgroup ID
def chiplet_transform(writer, kernel, sgprIndex, sgprNumTilesM, sgprNumTilesN):

    numXCC = 8
    module = Module()

    sgprXCC = writer.sgprPool.checkOut(1)
    sgprPOSINXCC = writer.sgprPool.checkOut(1)
    sgprNumXCC = writer.sgprPool.checkOut(1)

    sgprMinPerXCC = writer.sgprPool.checkOut(1)
    sgprExtraWG = writer.sgprPool.checkOut(1)
    sgprTmp = writer.sgprPool.checkOut(1)

    tmpVgpr = writer.vgprPool.checkOutAligned(6,2,"tmpVgpr")
    tmpVgprRes = ContinuousRegister(tmpVgpr, 6)

    module.add(SMovB32(dst=sgpr(sgprNumXCC), src=numXCC, comment=""))
    module.add(SMulI32(dst=sgpr(sgprTmp), src0=sgpr(sgprNumTilesM), src1=sgpr(sgprNumTilesN), comment="Compute total number of tiles"))

    # POSINXCC = Index // NumXCC
    # - ID in XCC
    # XCC = Index % NumXCC
    # - XCC ID
    #
    # MinPerXCC = TotalNumTiles // NumXCC
    # ExtraWG = TotalNumTiles % NumXCC
    module.add(scalarUInt24DivideAndRemainderPair(qReg=[sgprPOSINXCC,sgprMinPerXCC], dReg=[sgprIndex,sgprTmp], \
                                                  divReg=[sgprNumXCC,sgprNumXCC], rReg=[sgprXCC,sgprExtraWG], tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"]))

    module.add(SMulI32(dst=sgpr(sgprIndex), src0=sgpr(sgprXCC), src1=sgpr(sgprMinPerXCC), comment=""))
    module.add(SCmpLtU32(sgpr(sgprXCC), sgpr(sgprExtraWG), comment=""))
    module.add(SCSelectB32(dst=sgpr(sgprTmp), src0=sgpr(sgprXCC), src1=sgpr(sgprExtraWG)))
    module.add(SAddU32(dst=sgpr(sgprIndex), src0=sgpr(sgprIndex), src1=sgpr(sgprTmp), comment=""))
    module.add(SAddU32(dst=sgpr(sgprIndex), src0=sgpr(sgprIndex), src1=sgpr(sgprPOSINXCC), comment=""))

    writer.sgprPool.checkIn(sgprXCC)
    writer.sgprPool.checkIn(sgprPOSINXCC)
    writer.sgprPool.checkIn(sgprNumXCC)
    writer.sgprPool.checkIn(sgprMinPerXCC)
    writer.sgprPool.checkIn(sgprExtraWG)
    writer.sgprPool.checkIn(sgprTmp)
    writer.vgprPool.checkIn(tmpVgpr)

    return module

###################################################################################################
###################################################################################################
# SpaceFillingCurveWalk:
# Main entry point for space-fill walk
#
# Inputs: sgprWGM, sgprIndex
# sgprWGM: Stores num tiles in M/N directions for each level.
#   Format: (msb) [GridDimN_L2, GridDimM_L2, GridDimN_L1, GridDimM_L1] (lsb)
#       GridDim{M,N}_L{1,2} - Number of tiles in the M,N dims for each level. 8bit values each.
# sgprIndex: 1D global index [0, numTotalTiles]
#
# Conventions:
#      M,Y are used to reference rows of C matrix.
#      N,X are used to reference cols of C matrix.
#
###################################################################################################
###################################################################################################
def SpaceFillingCurveWalk(writer, kernel, sgprWGM):
    module = Module("remapSpaceFillingCurveWalk")

    # Currently disable check, since we ensure correct values are passed host-side
    checkWGMForZeroes = False
    if checkWGMForZeroes:
        # if gridDim is zero in wgm, set to 1.
        ts = writer.sgprPool.checkOut(1)
        for i in range(4):
            module.add(SAndB32(dst=sgpr(ts), src0=sgpr(sgprWGM), src1=hex(0x000000ff << (i * 8)), comment=""))
            module.add(SCmpEQU32(src0=sgpr(ts), src1=0, comment=""))
            module.add(SCSelectB32(dst=sgpr(ts), src0=hex(0x00000001 << (i * 8)), src1=sgpr(ts)))
            module.add(SOrB32(dst=sgpr(sgprWGM), src0=sgpr(sgprWGM), src1=sgpr(ts)))
        writer.sgprPool.checkIn(ts)

    # For non-streamK kernels we need to recompute the 1D global tile index
    # For StreamK we store the global tile index when its computed in skIndexToWG(...) in StreamK.py
    sgprIndex = "WorkGroup0"
    if not kernel["StreamK"]:
        sgprTmp = writer.sgprPool.checkOut(1)
        # Recompute the 1D ID from 2D IDs
        module.add(SMulI32(dst=sgpr(sgprTmp), src0=sgpr("WorkGroup1"), src1=sgpr("NumWorkGroups0"), comment=""))
        module.add(SAddU32(dst=sgpr(sgprIndex), src0=sgpr("WorkGroup0"), src1=sgpr(sgprTmp), comment=""))
        writer.sgprPool.checkIn(sgprTmp)
    else:
        module.add(SMovB32(dst=sgpr(sgprIndex), src=sgpr("StreamKTileID"), comment=""))

    # Global number of WGs in M, N directions.
    # sgprNumTiles{M,N} may be overwritten so we store NumWorkGroups{0,1} values in them.
    sgprNumTilesM = writer.sgprPool.checkOut(1)
    sgprNumTilesN = writer.sgprPool.checkOut(1)
    module.add(SMovB32(dst=sgpr(sgprNumTilesM), src=sgpr("NumWorkGroups0")))
    module.add(SMovB32(dst=sgpr(sgprNumTilesN), src=sgpr("NumWorkGroups1")))

    # Store sgprs in sgprsToStore to a single vgpr. This allows them to be
    # reused as tmps sgprs. The values are restored at the end of this subroutine (SpaceFillingCurveWalk)
    # If sgprsToStore is set to empty array, no sgprs are stored in a vgpr.
    sgprsToStore = ["NumWorkGroups0", "NumWorkGroups1", "Alpha", "Beta",\
                    "WorkGroup2", "StreamKLocalStart", "StreamKIterEnd", \
                    "LoopCounterL", "OrigLoopCounterL", "SizesSum"]
    if len(sgprsToStore):
        vgprSgprPool = writer.vgprPool.checkOut(1)
    for sgprVarsId in range(len(sgprsToStore)):
        sgprVars = sgprsToStore[sgprVarsId]
        if sgprVars not in writer.sgprs:
            continue
        module.add(VWritelaneB32(dst=vgpr(vgprSgprPool), src0=sgpr(sgprVars), src1=sgprVarsId, comment="Storing %s to vgpr"%sgprVars))
        writer.addSgprVarToPool(sgprVars)

    # Apply XCC remap
    # Apply a permtutation of all sgprIndex values. This will overwrite sgprIndex.
    useXCCRemap = len(kernel["SpaceFillingAlgo"])
    if useXCCRemap:
      module.add(chiplet_transform(writer, kernel, sgprIndex, sgprNumTilesM, sgprNumTilesN))


    # Starting (M,N) tile offset
    sgprGlobal       = writer.sgprPool.checkOutAligned(2,2)
    sgprGlobalY      = sgprGlobal
    sgprGlobalX      = sgprGlobal+1

    # Init X,Y tile offets to zero, these will be iteratively updated at each level to
    # point to a specific tile of C
    module.add(SMovB64(dst=sgpr(sgprGlobal,2), src=0))

    numLevels = writer.states.WGMTransformLevels
    if numLevels > 1:

    ##############################################################################################
    #
    # Partition C with MxN tiles to 4 blocks as follows
    #
    # | B0 | B3 |
    # | B1 | B2 |
    #
    #  Set M = M1 + M2, N = N1 + N2
    #
    #  B0 - Subpartition with M1xN1 tiles
    #    - The multi-level walks are applied to this block
    #
    #  Single level walks are applied to the remaining blocks, these contain the edge tiles
    #  B1 - Subpartition with M2xN1 tiles
    #  B2 - Subpartition with M2xN2 tiles
    #  B3 - Subpartition with M1xN2 tiles
    #
    #  M1 = RoundDown(GlobalGridM, GridM_L1 * GridM_L2), M1 is a multiple of GridM_L1 * GridM_L2
    #  N1 = RoundDown(GlobalGridN, GridN_L1 * GridN_L2), N1 is a multiple of GridN_L1 * GridN_L2
    #
    ##############################################################################################

        # Allocate as 64b aligned to use b64 SALUs when possible
        sgprNumTilesMN1 = writer.sgprPool.checkOutAligned(2,2)
        sgprNumTilesMN2 = writer.sgprPool.checkOutAligned(2,2)
        sgprNumTilesM1 = sgprNumTilesMN1
        sgprNumTilesN1 = sgprNumTilesMN1+1
        sgprNumTilesM2 = sgprNumTilesMN2
        sgprNumTilesN2 = sgprNumTilesMN2+1

        sgprTemp = writer.sgprPool.checkOut(1)
        sgprTemp2 = writer.sgprPool.checkOut(1)

        tmpVgpr = writer.vgprPool.checkOutAligned(6,2,"tmpVgpr")
        tmpVgprRes = ContinuousRegister(tmpVgpr, 6)
        # Fetch GridM_L1, GridM_L2 and calculate GridM_L1 * GridM_L2
        module.add(SAndB32(dst=sgpr(sgprNumTilesM1), src0=sgpr(sgprWGM), src1="0x000000ff"))
        if numLevels == 3:
            module.add(SAndB32(dst=sgpr(sgprTemp), src0=sgpr(sgprWGM), src1="0x00ff0000"))
            module.add(SLShiftRightB32(dst=sgpr(sgprTemp), src=sgpr(sgprTemp), shiftHex=16, comment=""))
            module.add(SMulI32(dst=sgpr(sgprNumTilesM1), src0=sgpr(sgprTemp), src1=sgpr(sgprNumTilesM1)))
        # Fetch GridN_L1, GridN_L2 and calculate GridN_L1 * GridN_L2
        module.add(SAndB32(dst=sgpr(sgprNumTilesN1), src0=sgpr(sgprWGM), src1="0x0000ff00"))
        module.add(SLShiftRightB32(dst=sgpr(sgprNumTilesN1), src=sgpr(sgprNumTilesN1), shiftHex=8, comment=""))
        if numLevels == 3:
            module.add(SAndB32(dst=sgpr(sgprTemp), src0=sgpr(sgprWGM), src1="0xff000000"))
            module.add(SLShiftRightB32(dst=sgpr(sgprTemp), src=sgpr(sgprTemp), shiftHex=24, comment=""))
            module.add(SMulI32(dst=sgpr(sgprNumTilesN1), src0=sgpr(sgprTemp), src1=sgpr(sgprNumTilesN1)))

        # Calculate M2,N2
        module.add(scalarUInt24DivideAndRemainderPair(qReg=[], dReg=[sgprNumTilesM,sgprNumTilesN], \
                                                      divReg=[sgprNumTilesM1,sgprNumTilesN1], rReg=[sgprNumTilesM2,sgprNumTilesN2], \
                                                      tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True, doQuotient=False))
        writer.vgprPool.checkIn(tmpVgpr)

        labelTransformN = Label(writer.labels.getUniqueNamePrefix("TransformNLevels"), comment="")
        labelTransform1 = Label(writer.labels.getUniqueNamePrefix("Transform1Level"), comment="")
        labelTransformsEnd = Label(writer.labels.getUniqueNamePrefix("TransformsEnd"), comment="")

        # Check if sgprIndex points to a tile in Block0 (M1xN1)
        module.add(SSubU32(dst=sgpr(sgprNumTilesN1), src0=sgpr(sgprNumTilesN), src1=sgpr(sgprNumTilesN2), comment=""))
        module.add(SSubU32(dst=sgpr(sgprNumTilesM1), src0=sgpr(sgprNumTilesM), src1=sgpr(sgprNumTilesM2), comment=""))
        module.add(SMulI32(dst=sgpr(sgprTemp), src0=sgpr(sgprNumTilesN1), src1=sgpr(sgprNumTilesM1)))
        module.add(SCmpLtU32(sgpr(sgprIndex), sgpr(sgprTemp), comment=""))
        module.add(SCBranchSCC1(labelName=labelTransformN.getLabelName(), comment=""))


        # Check if sgprIndex points to a tile in Block1 (M2xN1)
        # Global offsets start at tile (M1, 0)
        module.add(SSubU32(dst=sgpr(sgprIndex), src0=sgpr(sgprIndex), src1=sgpr(sgprTemp), comment=""))
        module.add(SMulI32(dst=sgpr(sgprTemp), src0=sgpr(sgprNumTilesM2), src1=sgpr(sgprNumTilesN1)))
        module.add(SCmpLtU32(sgpr(sgprIndex), sgpr(sgprTemp), comment=""))
        module.add(SCSelectB32(dst=sgpr(sgprGlobalY), src0=sgpr(sgprNumTilesM1), src1=sgpr(sgprGlobalY)))
        module.add(SCSelectB32(dst=sgpr(sgprNumTilesM), src0=sgpr(sgprNumTilesM2), src1=sgpr(sgprNumTilesM)))
        module.add(SCSelectB32(dst=sgpr(sgprNumTilesN), src0=sgpr(sgprNumTilesN1), src1=sgpr(sgprNumTilesN)))
        module.add(SCBranchSCC1(labelName=labelTransform1.getLabelName(), comment=""))

        # Check if sgprIndex points to a tile in Block2 (M2xN2)
        # Global offsets start at tile (M1, N1)
        module.add(SSubU32(dst=sgpr(sgprIndex), src0=sgpr(sgprIndex), src1=sgpr(sgprTemp), comment=""))
        module.add(SMulI32(dst=sgpr(sgprTemp), src0=sgpr(sgprNumTilesM2), src1=sgpr(sgprNumTilesN2)))
        module.add(SCmpLtU32(sgpr(sgprIndex), sgpr(sgprTemp), comment=""))
        module.add(SCSelectB64(dst=sgpr(sgprGlobalY,2), src0=sgpr(sgprNumTilesM1,2), src1=sgpr(sgprGlobalY,2)))
        module.add(SCSelectB64(dst=sgpr(sgprNumTilesM,2), src0=sgpr(sgprNumTilesM2,2), src1=sgpr(sgprNumTilesM,2)))
        module.add(SCBranchSCC1(labelName=labelTransform1.getLabelName(), comment=""))

        # Check if sgprIndex points to a tile in Block3 (M1xN2)
        # Global offsets start at tile (0, N1)
        module.add(SSubU32(dst=sgpr(sgprIndex), src0=sgpr(sgprIndex), src1=sgpr(sgprTemp), comment=""))
        module.add(SMulI32(dst=sgpr(sgprTemp), src0=sgpr(sgprNumTilesM1), src1=sgpr(sgprNumTilesN2)))
        module.add(SCmpLtU32(sgpr(sgprIndex), sgpr(sgprTemp), comment=""))
        module.add(SCSelectB32(dst=sgpr(sgprGlobalX), src0=sgpr(sgprNumTilesN1), src1=sgpr(sgprGlobalX)))
        module.add(SCSelectB32(dst=sgpr(sgprNumTilesM), src0=sgpr(sgprNumTilesM1), src1=sgpr(sgprNumTilesM)))
        module.add(SCSelectB32(dst=sgpr(sgprNumTilesN), src0=sgpr(sgprNumTilesN2), src1=sgpr(sgprNumTilesN)))
        module.add(SCBranchSCC1(labelName=labelTransform1.getLabelName(), comment=""))

        # Start of multi-level walk code-path
        module.add(labelTransformN)
        module.add(TransformNLevels(writer, kernel, sgprIndex, sgprNumTilesM1, sgprNumTilesN1, sgprWGM, sgprGlobalY, sgprGlobalX, numLevels))
        module.add(SBranch(labelName=labelTransformsEnd.getLabelName()))

        # Start of single-level walk code-path, to handle edge tiles
        module.add(labelTransform1)
        module.add(TransformNLevels(writer, kernel, sgprIndex, sgprNumTilesM, sgprNumTilesN, sgprWGM, sgprGlobalY, sgprGlobalX, 1))

        module.add(labelTransformsEnd)

        writer.sgprPool.checkIn(sgprNumTilesMN1)
        writer.sgprPool.checkIn(sgprNumTilesMN2)
        writer.sgprPool.checkIn(sgprTemp)
        writer.sgprPool.checkIn(sgprTemp2)

    else:
        module.add(TransformNLevels(writer, kernel, sgprIndex, sgprNumTilesM, sgprNumTilesN, sgprWGM, sgprGlobalY, sgprGlobalX, 1))

    writer.sgprPool.checkIn(sgprGlobal)
    writer.sgprPool.checkIn(sgprNumTilesM)
    writer.sgprPool.checkIn(sgprNumTilesN)

    # Restore sgprs listed in sgprsToStore to their original sgprs
    for sgprVarsId in range(len(sgprsToStore)):
        sgprVars = sgprsToStore[sgprVarsId]
        if sgprVars not in writer.sgprs:
            continue
        module.add(VReadlaneB32(dst=sgpr(sgprVars), src0=vgpr(vgprSgprPool), src1=sgprVarsId, comment="Restoring %s to sgpr"%sgprVars))
        writer.removeSgprVarFromPool(sgprVars)

    if len(sgprsToStore):
        module.add(SNop(waitState=1, comment=""))
        writer.vgprPool.checkIn(vgprSgprPool)

    return module

###################################################################################################
# TransformNLevels:
# Main driver of nested orderings, only supports 1,2,3 Levels. This applies ordering to a subpartition
# of C matrix
#
# Inputs:
#   - sgprIndex: 1D tile index for subpartition ranges [0, numTilesM * numTilesN - 1]
#   - sgprNumTiles{M,N}: Number of C-matrix tiles in subpartition in {M,N} dim
#   - sgprWGM: Stores num tiles in M/N directions for each level.
#      Format: (msb) [GridDimN_L2, GridDimM_L2, GridDimN_L1, GridDimM_L1] (lsb)
#   - sgprGlobal{Y,X}: Stores global tile offset. Initial offsets are set to upper left corner of
#        the subpartition. This will be updated with local index computations in this subroutine
#
# Level-1
#   - First level has local grid size of (GridDimM_L1 x GridDimN_L1)
# Level-2
#   - Second level has local grid size of (GridDimM_L2 x GridDimN_L2)
# Level-3
#   - Third level has local grid size of GM x GN where
#      G{M,N} = floor(NumTiles{M,N} // (GridDim{M,N}_L1 * GridDim{M,N}_L2)) * (GridDim{M,N}_L1 * GridDim{M,N}_L2)
#      The grid sizes at this level is not explicitly specified, it is computed using grid sizes from previous
#      levels and the global number of M/N tiles in the subpartition
###################################################################################################
def TransformNLevels(writer, kernel, sgprIndex, sgprNumTilesM, sgprNumTilesN, sgprWGM, sgprGlobalY, sgprGlobalX, numLevels):

    module = Module()
    numLevels = min(numLevels, writer.states.WGMTransformLevels)

    sgprCumulativeDenominator = writer.sgprPool.checkOut(1)

    if numLevels > 1:
        sgprCumulativeX = writer.sgprPool.checkOut(1)
        sgprCumulativeY = writer.sgprPool.checkOut(1)
        sgprBlockM       = writer.sgprPool.checkOut(1)
        sgprBlockN       = writer.sgprPool.checkOut(1)

    sgprLocalXY      = writer.sgprPool.checkOutAligned(2,2)
    sgprLocalY       = sgprLocalXY
    sgprLocalX       = sgprLocalXY + 1
    sgprTemp         = writer.sgprPool.checkOut(1)

    SFCO = kernel["SpaceFillingAlgo"]

    # Default cumulative values
    module.add(SMovB32(dst=sgpr(sgprCumulativeDenominator), src=1))
    if numLevels > 1:
        module.add(SMovB32(dst=sgpr(sgprCumulativeX), src=1))
        module.add(SMovB32(dst=sgpr(sgprCumulativeY), src=1))
        module.add(SMovB32(dst=sgpr(sgprTemp), src=sgpr(sgprWGM)))

    if numLevels > 1:
        # Multi-level code-path
        for lvl in range(numLevels):
            module.addComment0("Transform Level-%u"%lvl)

            if lvl != numLevels - 1:
                # Read grid dims for WGM if not last level
                module.add(SAndB32(dst=sgpr(sgprBlockM), src0=sgpr(sgprTemp), src1="0x000000ff"))
                module.add(SAndB32(dst=sgpr(sgprBlockN), src0=sgpr(sgprTemp), src1="0x0000ff00"))
                module.add(SLShiftRightB32(dst=sgpr(sgprBlockN), src=sgpr(sgprBlockN), shiftHex=8, comment=""))
            else:
                # Compute grid dims from current level using dims from previous levels
                tmpVgpr = writer.vgprPool.checkOutAligned(6,2,"tmpVgpr")
                tmpVgprRes = ContinuousRegister(tmpVgpr, 6)
                # Fetch numTilesM for the particular level
                module.add(SAndB32(dst=sgpr(sgprBlockM), src0=sgpr(sgprWGM), src1="0x000000ff"))
                if numLevels == 3:
                    module.add(SAndB32(dst=sgpr(sgprTemp), src0=sgpr(sgprWGM), src1="0x00ff0000"))
                    module.add(SLShiftRightB32(dst=sgpr(sgprTemp), src=sgpr(sgprTemp), shiftHex=16, comment=""))
                    module.add(SMulI32(dst=sgpr(sgprBlockM), src0=sgpr(sgprTemp), src1=sgpr(sgprBlockM)))
                # Fetch numTilesN for the particular level
                module.add(SAndB32(dst=sgpr(sgprBlockN), src0=sgpr(sgprWGM), src1="0x0000ff00"))
                module.add(SLShiftRightB32(dst=sgpr(sgprBlockN), src=sgpr(sgprBlockN), shiftHex=8, comment=""))
                if numLevels == 3:
                    module.add(SAndB32(dst=sgpr(sgprTemp), src0=sgpr(sgprWGM), src1="0xff000000"))
                    module.add(SLShiftRightB32(dst=sgpr(sgprTemp), src=sgpr(sgprTemp), shiftHex=24, comment=""))
                    module.add(SMulI32(dst=sgpr(sgprBlockN), src0=sgpr(sgprTemp), src1=sgpr(sgprBlockN)))

                module.add(scalarUInt24DivideAndRemainderPair(qReg=[sgprBlockM,sgprBlockN], dReg=[sgprNumTilesM,sgprNumTilesN], \
                                                              divReg=[sgprBlockM,sgprBlockN], rReg=[-1,-1], tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=False))
                writer.vgprPool.checkIn(tmpVgpr)

            ##############################################
            # Generate ordering requested. This is build-time generated. No runtime dispatch
            if SFCO[lvl] == 0: # ColMajorOrder
                module.add(ColRowMajor(writer, kernel, sgprIndex, sgprBlockM, sgprBlockN, sgprLocalY, sgprLocalX, sgprCumulativeDenominator, lvl, True))
            elif SFCO[lvl] == 1: # RowMajorOrder
                module.add(ColRowMajor(writer, kernel, sgprIndex, sgprBlockM, sgprBlockN, sgprLocalY, sgprLocalX, sgprCumulativeDenominator, lvl, False))
            elif SFCO[lvl] in range(2,6): # HilbertOrderSimple
                orderID = SFCO[lvl]
                module.add(SpaceFillCurveSimpleImpl(writer, kernel, sgprIndex, sgprBlockM, sgprBlockN, sgprLocalY, sgprLocalX, sgprCumulativeDenominator, lvl, orderID))

            # Update global indices with contributions from local indices
            module.add(SMulI32(dst=sgpr(sgprLocalX), src0=sgpr(sgprLocalX), src1=sgpr(sgprCumulativeX)))
            module.add(SMulI32(dst=sgpr(sgprLocalY), src0=sgpr(sgprLocalY), src1=sgpr(sgprCumulativeY)))
            module.addComment0("Updating global offsets with local offsets")
            module.add(SAddU32(dst=sgpr(sgprGlobalX), src0=sgpr(sgprLocalX), src1=sgpr(sgprGlobalX)))
            module.add(SAddU32(dst=sgpr(sgprGlobalY), src0=sgpr(sgprLocalY), src1=sgpr(sgprGlobalY)))

            if lvl != numLevels - 1:
                module.add(SMulI32(dst=sgpr(sgprCumulativeX), src0=sgpr(sgprBlockN), src1=sgpr(sgprCumulativeX)))
                module.add(SMulI32(dst=sgpr(sgprCumulativeY), src0=sgpr(sgprBlockM), src1=sgpr(sgprCumulativeY)))
                module.add(SLShiftRightB32(dst=sgpr(sgprTemp), src=sgpr(sgprTemp), shiftHex=16, comment=""))
    else:
        # Single-level code-path
        lvl = 0
        sgprBlockM = sgprNumTilesM
        sgprBlockN = sgprNumTilesN
        if SFCO[lvl] == 0: # ColMajorOrder
            module.add(ColRowMajor(writer, kernel, sgprIndex, sgprBlockM, sgprBlockN, sgprLocalY, sgprLocalX, sgprCumulativeDenominator, lvl, True))
        elif SFCO[lvl] == 1: # RowMajorOrder
            module.add(ColRowMajor(writer, kernel, sgprIndex, sgprBlockM, sgprBlockN, sgprLocalY, sgprLocalX, sgprCumulativeDenominator, lvl, False))
        elif SFCO[lvl] in range(2,6): # HilbertOrderSimple
            orderID = SFCO[0]
            module.add(SpaceFillCurveSimpleImpl(writer, kernel, sgprIndex, sgprBlockM, sgprBlockN, sgprLocalY, sgprLocalX, sgprCumulativeDenominator, lvl, orderID))
        module.addComment0("Updating global offsets with local offsets")
        module.add(SAddU32(dst=sgpr(sgprGlobalX), src0=sgpr(sgprLocalX), src1=sgpr(sgprGlobalX)))
        module.add(SAddU32(dst=sgpr(sgprGlobalY), src0=sgpr(sgprLocalY), src1=sgpr(sgprGlobalY)))

    # Global offset computation done. Store values
    module.addComment0("Done with tile transforms. Storing globalY/X offsets to WG Ids")
    module.add(SMovB32(dst=sgpr("WorkGroup0"), src=sgpr(sgprGlobalY), comment="Store globalOffsetY"))
    module.add(SMovB32(dst=sgpr("WorkGroup1"), src=sgpr(sgprGlobalX), comment="Store globalOffsetX"))

    writer.sgprPool.checkIn(sgprCumulativeDenominator)
    if numLevels > 1:
        writer.sgprPool.checkIn(sgprCumulativeX)
        writer.sgprPool.checkIn(sgprCumulativeY)
        writer.sgprPool.checkIn(sgprBlockM)
        writer.sgprPool.checkIn(sgprBlockN)
    writer.sgprPool.checkIn(sgprLocalXY)
    writer.sgprPool.checkIn(sgprTemp)

    return module

###############################################################################
###############################################################################
#
# Implementations of various orderings
#
###############################################################################
###############################################################################


###################################################################################################
# OrderingPreamble:
# Shared logic common across all ordering impls which computes local indices and tile offsets
# given grid sizes and cumulative denominator and current level
#
###################################################################################################
def OrderingPreamble(writer, kernel, sgprIndex, sgprGridY, sgprGridX, sgprCumulativeDenominator, sgprLocalIndex, lvl):
    module = Module()

    sgprTemp = writer.sgprPool.checkOut(1)
    tmpVgpr = writer.vgprPool.checkOutAligned(4,2,"tmpVgpr")
    tmpVgprRes = ContinuousRegister(tmpVgpr, 4)
    module.addComment0("Computing local indices for level-%u"%lvl)
    # index // cumulativeDenom
    if lvl == 0:
        # For lowest level, cumulativeDenom will always be 1, so skip the division
        module.add(SMovB32(dst=sgpr(sgprLocalIndex), src=sgpr(sgprIndex)))
    else:
        # For higher levels, we compute "global index" for this level. sgprIndex = sgprIndex // sgprCumulativeDenominator
        module.add(scalarUInt24DivideAndRemainder(qReg=sgprLocalIndex, dReg=sgprIndex, divReg=sgprCumulativeDenominator, rReg=-1, \
                                                  tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=False))

    module.add(SMulI32(dst=sgpr(sgprTemp), src0=sgpr(sgprGridX), src1=sgpr(sgprGridY)))
    # Computes final local index from "global index" of this level,
    # sgprLocalIndex = (sgprIndex // sgprCumulativeDenominator) % (gridX * gridY), qReg not used
    module.add(scalarUInt24DivideAndRemainder(qReg=-1, dReg=sgprLocalIndex, divReg=sgprTemp, rReg=sgprLocalIndex, \
                                              tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True, doQuotient=False))

    module.add(SMulI32(dst=sgpr(sgprCumulativeDenominator), src0=sgpr(sgprCumulativeDenominator), src1=sgpr(sgprTemp)))
    module.addComment0("Done computing local indices for level-%u. Index store in s%u"%(lvl, sgprLocalIndex))

    writer.sgprPool.checkIn(sgprTemp)
    writer.vgprPool.checkIn(tmpVgpr)

    return module


###################################################################################################
# ColRowMajor:
# Computes column or row major ordering for a subparition of C
#
###################################################################################################
def ColRowMajor(writer, kernel, sgprIndex, sgprGridY, sgprGridX, sgprLocalY, sgprLocalX, sgprCumulativeDenominator, lvl, isCol):
    module = Module("RowMajor")

    sgprLocalIndex   = writer.sgprPool.checkOut(1)

    module.addComment0("Start of Col/Row Major ordering transform")
    # Calculate local index for this order, and update cumulativedenom
    module.add(OrderingPreamble(writer, kernel, sgprIndex, sgprGridY, sgprGridX, sgprCumulativeDenominator, sgprLocalIndex, lvl))

    # Temp vpgr for div, rem calcs
    tmpVgpr = writer.vgprPool.checkOutAligned(4,2,"tmpVgpr")
    tmpVgprRes = ContinuousRegister(tmpVgpr, 4)

    qReg   = sgprLocalX if isCol else sgprLocalY
    divReg = sgprGridY  if isCol else sgprGridX
    rReg   = sgprLocalY if isCol else sgprLocalX

    module.add(scalarUInt24DivideAndRemainder(qReg=sgprLocalX, dReg=sgprLocalIndex, divReg=sgprGridY, rReg=sgprLocalY, \
                                              tmpVgprRes=tmpVgprRes, wavewidth=kernel["WavefrontSize"], doRemainder=True))

    module.addComment0("End of Col/Row Major ordering transform")

    writer.vgprPool.checkIn(tmpVgpr)
    writer.sgprPool.checkIn(sgprLocalIndex)

    return module

###################################################################################################
# SpaceFillCurvesSimpleImpl:
# Computes Hilbert/Morton ordering for a sub partition of C
#
###################################################################################################
def SpaceFillCurveSimpleImpl(writer, kernel, sgprIndex, sgprGridY, sgprGridX, sgprLocalY, sgprLocalX, sgprCumulativeDenominator, lvl, orderID):

    module = Module()

    sgprLocalIndex   = writer.sgprPool.checkOut(1)

    # Stores GridY/GridX values, since tmp vgprs, since these will need to be updated.
    sgprNumTilesMN    = writer.sgprPool.checkOutAligned(2,2)
    sgprNumTilesM     = sgprNumTilesMN
    sgprNumTilesN     = sgprNumTilesMN + 1
    module.add(SMovB64(dst=sgpr(sgprNumTilesM,2), src=sgpr(sgprGridY,2)))

    # Calculate local index for this order, and update cumulativedenom
    module.add(OrderingPreamble(writer, kernel, sgprIndex, sgprNumTilesM, sgprNumTilesN, sgprCumulativeDenominator, sgprLocalIndex, lvl))
    # Init local XY offsets to 0
    module.add(SMovB64(dst=sgpr(sgprLocalY,2), src=0))

    tmpSgprCurDir        = writer.sgprPool.checkOut(1)
    sgprNumTilesMN1      = writer.sgprPool.checkOutAligned(2,2)
    sgprNumTilesM1      = sgprNumTilesMN1
    sgprNumTilesN1      = sgprNumTilesMN1 + 1
    sgprNumTilesM2       = writer.sgprPool.checkOut(1)
    sgprNumTilesN2       = writer.sgprPool.checkOut(1)

    tmpSgpr = writer.sgprPool.checkOut(1)

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

    if orderID == 2:
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
    elif orderID == 3:
      # Z-Walk
      directions = ["MortonZ"]
      directionBlocks = [
        [block0, block2, block1, block3],
      ]
    elif orderID == 4:
      # ReverseN-Walk
      directions = ["MortonRN"]
      directionBlocks = [
        [block0, block1, block2, block3],
      ]
    elif orderID == 5:
      # U-Walk
      directions = ["MortonU"]
      directionBlocks = [
        [block0, block1, block3, block2],
      ]

    # Generate labels for each direction
    for i in range(0, len(directions)):
      directionLabels.append(Label((writer.labels.getUniqueNamePrefix(directions[i])), comment=""))

    # assign numeric values for each directions
    for i in range(0, len(directions)):
      module.add(ValueSet(directions[i], i, format=1))

    if len(directions) > 1:
      module.add(SMovB32(dst=sgpr(tmpSgprCurDir), src=directions[0]))

    # Begin SFC

    labelStartWhile = Label(writer.labels.getUniqueNamePrefix("SpaceFillingCurveWalkStartWhile"), comment="")
    labelEndWhile = Label(writer.labels.getUniqueNamePrefix("SpaceFillingCurveWalkEndWhile"), comment="")

    module.add(labelStartWhile)
    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(sgprNumTilesM), src1=sgpr(sgprNumTilesN)))
    module.add(SCmpGtU32(src0=sgpr(tmpSgpr), src1=1, comment="M * N > bkM * bkN"))
    module.add(SCBranchSCC0(labelName=labelEndWhile.getLabelName(), comment=""))

    ###########################################
    # Body of loop
    ###########################################

    # Compute M = M1 + M2, N = N1 + N2, the dims of the four subparitions
    sgprNumTilesL  = [sgprNumTilesM, sgprNumTilesN]
    sgprNumTilesL1 = [sgprNumTilesM1, sgprNumTilesN1]
    sgprNumTilesL2 = [sgprNumTilesM2, sgprNumTilesN2]
    for i in range(2):
        # For each dim compute:
        #   L1 = L == bit_floor(L) ? L / 2 : bit_floor(L)
        #   L2 = L - L1
        dc = 'M' if i == 0 else 'N'
        module.addComment0("Computing %s1, %s2: %s1 + %s2 = %s"%(dc,dc,dc,dc,dc))
        module.add(SFlbitI32B32(dst=sgpr(tmpSgpr), src=sgpr(sgprNumTilesL[i]), comment="bit_floor(%s)"%dc))
        module.add(SSubU32(dst=sgpr(tmpSgpr), src0=31, src1=sgpr(tmpSgpr), comment="bit_floor(%s)"%dc))
        module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr), src=1, shiftHex=sgpr(tmpSgpr), comment="bit_floor(%s)"%dc))

        module.add(SLShiftRightB32(dst=sgpr(sgprNumTilesL1[i]), src=sgpr(sgprNumTilesL[i]), shiftHex=1, comment="%s1 = %s / 2"%(dc,dc)))

        module.add(SCmpEQU32(src0=sgpr(tmpSgpr), src1=sgpr(sgprNumTilesL[i]), comment="%s == bit_floor(%s) ?"%(dc, dc)))
        module.add(SCSelectB32(dst=sgpr(sgprNumTilesL1[i]), src0=sgpr(sgprNumTilesL1[i]), src1=sgpr(tmpSgpr), comment="%s == bit_floor(%s) ? %s / 2 : bit_floor(%s)"%(dc,dc,dc,dc)))
        module.add(SSubU32(dst=sgpr(sgprNumTilesL2[i]), src0=sgpr(sgprNumTilesL[i]), src1=sgpr(sgprNumTilesL1[i]), comment="%s2 = %s - %s1"%(dc,dc,dc)))

    if len(directions) > 1:
        for i in range(0, len(directions)):
            module.add(SCmpEQU32(src0=sgpr(tmpSgprCurDir), src1=directions[i], comment=""))
            module.add(SCBranchSCC1(labelName=directionLabels[i].getLabelName()))

    for i in range(0, len(directions)):
        if len(directions) > 1:
            module.addComment0("Code-branch for %s"%directions[i])
            module.add(directionLabels[i])
        block = directionBlocks[i]
        # check which subpartition sgprLocalIndex belongs to and dispatch to that block.
        for j in range(0, 4):
            # Label to next block partition
            label = Label((writer.labels.getUniqueNamePrefix(directions[i]+"_block%u"%(j+1))), comment="")
            # Update sgprLocalIndex, by subtracting tile partition sizes
            if j > 0:
                module.add(SSubU32(dst=sgpr(sgprLocalIndex), src0=sgpr(sgprLocalIndex), src1=sgpr(tmpSgpr), comment="Update serial idx: subtract block-%u"%j))
            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(block[j][0]), src1=sgpr(block[j][1])))
            if j < 3:
                module.add(SCmpGeU32(sgpr(sgprLocalIndex), sgpr(tmpSgpr), comment=""))
                module.add(SCBranchSCC1(labelName=label.getLabelName(), comment=""))
            if block[j][1] == block[j][0] + 1 and block[j][0] % 2 == 0:
                module.add(SMovB64(dst=sgpr(sgprNumTilesM,2), src=sgpr(block[j][0],2), comment="Update M/N"))
            else:
                module.add(SMovB32(dst=sgpr(sgprNumTilesM), src=sgpr(block[j][0]), comment="Update M"))
                module.add(SMovB32(dst=sgpr(sgprNumTilesN), src=sgpr(block[j][1]), comment="Update N"))
            curBlockOffset = blockXYOffset(directionBlocks[i][j])
            if curBlockOffset[0] != 0:
                module.add(SAddU32(dst=sgpr(sgprLocalY), src0=sgpr(sgprLocalY), src1=sgpr(curBlockOffset[0]), comment="Update idX"))
            if curBlockOffset[1] != 0:
                module.add(SAddU32(dst=sgpr(sgprLocalX), src0=sgpr(sgprLocalX), src1=sgpr(curBlockOffset[1]), comment="Update idY"))
            if len(directions) > 1:
                module.add(SMovB32(dst=sgpr(tmpSgprCurDir), src=directionNewDir[i][j], comment="Update direction"))
            module.add(SBranch(labelName=labelStartWhile.getLabelName()))
            if j < 3:
                module.add(label)

    module.add(labelEndWhile)


    writer.sgprPool.checkIn(tmpSgpr)
    writer.sgprPool.checkIn(tmpSgprCurDir)
    writer.sgprPool.checkIn(sgprNumTilesMN)
    writer.sgprPool.checkIn(sgprNumTilesMN1)
    writer.sgprPool.checkIn(sgprNumTilesM2)
    writer.sgprPool.checkIn(sgprNumTilesN2)
    writer.sgprPool.checkIn(sgprLocalIndex)

    return module
