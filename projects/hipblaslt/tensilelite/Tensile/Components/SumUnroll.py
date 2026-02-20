################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

from rocisa.code import RegSet, Module
from rocisa.container import EXEC, vgpr, sgpr, SDWAModifiers, DSModifiers, ContinuousRegister
from rocisa.enum import SelectBit
from rocisa.instruction import DSStoreB16, DSStoreB32, DSStoreB64, SBarrier, \
    SMovB32, SSetMask, VAddF32, VAddU32, VCmpXEqU32, VCvtPkBF8toF32, VCvtPkFP8toF32, \
    VDot2F32BF16, VDot2F32F16, VLShiftLeftB32, VMovB32, VCvtBF16toFP32
from rocisa.functions import vectorStaticDivide, vectorStaticRemainder, vectorStaticMultiply
from ..Component import SumUnroll
from ..Common import printExit, log2

class SumUnrollMfma(SumUnroll):
    kernel = {"EnableMatrixInstruction": True}

    """
    Sum unroll for reduction
    Use the same pattern as mfma
    """
    def __call__(self):
        assert(0)

    def initSumUnroll(self, writer, kernel):
        imod = Module("initSumUnroll")
        idx = 0
        while idx < writer.states.bias.numVgprValu:
            imod.add(VMovB32(dst=vgpr("ValuSum+%d"%idx), src=0, comment="reset 0"))
            idx += 1

        # Init sum unroll, create pack for dot if needed
        # Unregister is in storeSumLDS
        if kernel["ProblemType"]["DataType"].numRegisters() < 1:
            if kernel["ProblemType"]["DataType"].isHalf():
                writer.defineSgpr("SumUnrollConstOne", 1)
                imod.add(RegSet("s", "sgprSumUnrollConstOne", writer.sgprs["SumUnrollConstOne"]))
                imod.add(SMovB32(dst=sgpr("SumUnrollConstOne"), src=hex(0x3c003c00), comment="packed 1.0"))
            elif kernel["ProblemType"]["DataType"].isBFloat16() and writer.states.asmCaps['v_dot2_f32_bf16']:
                writer.defineSgpr("SumUnrollConstOne", 1)
                imod.add(RegSet("s", "sgprSumUnrollConstOne", writer.sgprs["SumUnrollConstOne"]))
                imod.add(SMovB32(dst=sgpr("SumUnrollConstOne"), src=hex(0x3F803F80), comment="packed 1.0"))
            else:
                assert "[initSumUnroll] Unsupported data type"
        return imod

    def loopSum(self, writer, kernel, tP, u, innerUnroll):
        tc   = tP["tensorChar"]
        imod = Module("SumUnroll%s_I%s" % (tc, innerUnroll))

        m = (u) % (writer.states.numVgprBuffer) # local to use for MACs

        # calculate constant
        numRegistersIn   = kernel["ProblemType"]["DataType"].numRegisters()
        numMIInput       = kernel["MIInputPerThread%s"%tc]
        vgprPerInput     = int(numMIInput * numRegistersIn)

        if tc == "A":
            waveTile = kernel["MIWaveTile"][0]
            numIterPerCoalescedRead = writer.states.numIterPerCoalescedReadA
            numReadsIterCoalesced = writer.states.numReadsIterCoalescedA
        elif tc == "B":
            waveTile = kernel["MIWaveTile"][1]
            numIterPerCoalescedRead = writer.states.numIterPerCoalescedReadB
            numReadsIterCoalesced = writer.states.numReadsIterCoalescedB
        else:
            printExit("Unsupported tc %s"%tc)
        # here we remap index to where it read for wider local read
        # ex. if we read 2 iteration at a time,
        #   original   : _ds_load_b64  valuA_X0_I0
        #   read 2 iter: _ds_load_b128 valuA_X0_I0 (we read valuA_X0_I0 and valuA_X1_I0)
        # instead of using valuA_X1_I0, we use valuA_X0_I0+2 as mfma input

        vgprBuffer_new = (m//numIterPerCoalescedRead)*numIterPerCoalescedRead
        vgprBuffer_new_offset = m%numIterPerCoalescedRead*kernel["InnerUnroll"]*vgprPerInput

        if kernel["ProblemType"]["DataType"].isBFloat16():
            hiBitsMaskVgpr = writer.vgprPool.checkOut(1)
            imod.add(VMovB32(dst=vgpr(hiBitsMaskVgpr), src=hex(0xffff0000), comment="mask 0xffff0000 for pack two bfloat16 element to 32bit"))

        for iui in range(0, innerUnroll):
            iui_new = (iui//numReadsIterCoalesced)*numReadsIterCoalesced
            iui_new_offset = iui%numReadsIterCoalesced*vgprPerInput
            for idx in range(0, waveTile):
                new     = idx*vgprPerInput*numReadsIterCoalesced
                # valuStr = "Valu%s_X%u_I%u+%u+%u+%u" % (tc, vgprBuffer_new, iui_new, new, vgprBuffer_new_offset, iui_new_offset)
                valuStr    = "Valu%s_X%u_I%u+%u+%u" % (tc, vgprBuffer_new, iui_new, new, vgprBuffer_new_offset)
                valuSumStr = "ValuSum+%u"%idx
                # If direct ot vgpr, use "G2LA+%u+%u+%u", currently not supported
                if kernel["ProblemType"]["DataType"].isHalf():
                    # First version only supports mfma with K > 1
                    if vgprPerInput > 1 and (vgprPerInput % 2 == 0):
                        for inputIdx in range(0, vgprPerInput):
                            imod.add(VDot2F32F16(dst=vgpr(valuSumStr), src0=vgpr("%s+%s"%(valuStr, iui_new_offset + inputIdx)), src1=sgpr("SumUnrollConstOne"), src2=vgpr(valuSumStr), comment="sum K"))
                    else:
                        printExit("Currently unsupported vgprPerInput %u"%vgprPerInput)
                elif kernel["ProblemType"]["DataType"].isSingle():
                    inputIdx = 0
                    while inputIdx < vgprPerInput:
                        imod.add(VAddF32(dst=vgpr(valuSumStr), src0=vgpr("%s+%s"%(valuStr, iui_new_offset + inputIdx)), src1=vgpr(valuSumStr), comment="sum K"))
                        inputIdx += 1
                elif kernel["ProblemType"]["DataType"].isBFloat16():
                    # BF16 BiasSrcA,B
                    tmpVgpr = writer.vgprPool.checkOutAligned(2,2)
                    if vgprPerInput > 1 and (vgprPerInput % 2 == 0):
                        for inputIdx in range(0, vgprPerInput):
                            if writer.states.asmCaps['v_dot2_f32_bf16']:
                                imod.add(VDot2F32BF16(dst=vgpr(valuSumStr), src0=vgpr("%s+%s"%(valuStr, iui_new_offset + inputIdx)), src1=sgpr("SumUnrollConstOne"), src2=vgpr(valuSumStr), comment="sum K"))
                            else:
                                imod.add(VCvtBF16toFP32(dst=vgpr(tmpVgpr), src=vgpr("%s+%s"%(valuStr, iui_new_offset + inputIdx)), vgprMask=None, vi=0))
                                imod.add(VCvtBF16toFP32(dst=vgpr(tmpVgpr+1), src=vgpr("%s+%s"%(valuStr, iui_new_offset + inputIdx)), vgprMask=vgpr(hiBitsMaskVgpr), vi=1))
                                imod.add(VAddF32(dst=vgpr(valuSumStr), src0=vgpr(tmpVgpr), src1=vgpr(valuSumStr), comment="sum K"))
                                imod.add(VAddF32(dst=vgpr(valuSumStr), src0=vgpr(tmpVgpr+1), src1=vgpr(valuSumStr), comment="sum K"))
                    else:
                        printExit("Currently unsupported vgprPerInput %u"%vgprPerInput)
                    writer.vgprPool.checkIn(tmpVgpr)
                elif (kernel["ProblemType"]["DataType"].isAnyFloat8A() and tc == "A") or \
                     (kernel["ProblemType"]["DataType"].isAnyFloat8B() and tc == "B"):
                    #FP8
                    tmpVgpr = writer.vgprPool.checkOutAligned(4,2)
                    if vgprPerInput > 1 and (vgprPerInput % 2 == 0):
                        for inputIdx in range(0, vgprPerInput):
                            sdwa = SDWAModifiers(src0_sel=SelectBit.WORD_0)
                            imod.add(VCvtPkFP8toF32(dst=vgpr(tmpVgpr,2), src=vgpr("%s+%s"%(valuStr, iui_new_offset + inputIdx)), sdwa=sdwa, comment="convert to FP32"))
                            sdwa = SDWAModifiers(src0_sel=SelectBit.WORD_1)
                            imod.add(VCvtPkFP8toF32(dst=vgpr(tmpVgpr+2,2), src=vgpr("%s+%s"%(valuStr, iui_new_offset + inputIdx)), sdwa=sdwa, comment="convert to FP32"))
                            imod.add(VAddF32(dst=vgpr(valuSumStr), src0=vgpr(tmpVgpr), src1=vgpr(valuSumStr), comment="sum K"))
                            imod.add(VAddF32(dst=vgpr(valuSumStr), src0=vgpr(tmpVgpr+1), src1=vgpr(valuSumStr), comment="sum K"))
                            imod.add(VAddF32(dst=vgpr(valuSumStr), src0=vgpr(tmpVgpr+2), src1=vgpr(valuSumStr), comment="sum K"))
                            imod.add(VAddF32(dst=vgpr(valuSumStr), src0=vgpr(tmpVgpr+3), src1=vgpr(valuSumStr), comment="sum K"))
                    else:
                        printExit("Currently unsupported vgprPerInput %u"%vgprPerInput)
                    writer.vgprPool.checkIn(tmpVgpr)
                elif (kernel["ProblemType"]["DataType"].isAnyBFloat8A() and tc == "A") or \
                     (kernel["ProblemType"]["DataType"].isAnyBFloat8B() and tc == "B"):
                    #BF8
                    tmpVgpr = writer.vgprPool.checkOutAligned(4,2)
                    if vgprPerInput > 1 and (vgprPerInput % 2 == 0):
                        for inputIdx in range(0, vgprPerInput):
                            sdwa = SDWAModifiers(src0_sel=SelectBit.WORD_0)
                            imod.add(VCvtPkBF8toF32(dst=vgpr(tmpVgpr,2), src=vgpr("%s+%s"%(valuStr, iui_new_offset + inputIdx)), sdwa=sdwa, comment="convert to FP32"))
                            sdwa = SDWAModifiers(src0_sel=SelectBit.WORD_1)
                            imod.add(VCvtPkBF8toF32(dst=vgpr(tmpVgpr+2,2), src=vgpr("%s+%s"%(valuStr, iui_new_offset + inputIdx)), sdwa=sdwa, comment="convert to FP32"))
                            imod.add(VAddF32(dst=vgpr(valuSumStr), src0=vgpr(tmpVgpr), src1=vgpr(valuSumStr), comment="sum K"))
                            imod.add(VAddF32(dst=vgpr(valuSumStr), src0=vgpr(tmpVgpr+1), src1=vgpr(valuSumStr), comment="sum K"))
                            imod.add(VAddF32(dst=vgpr(valuSumStr), src0=vgpr(tmpVgpr+2), src1=vgpr(valuSumStr), comment="sum K"))
                            imod.add(VAddF32(dst=vgpr(valuSumStr), src0=vgpr(tmpVgpr+3), src1=vgpr(valuSumStr), comment="sum K"))
                    else:
                        printExit("Currently unsupported vgprPerInput %u"%vgprPerInput)
                    writer.vgprPool.checkIn(tmpVgpr)
                else:
                    printExit("Currently unsupported data type")

        if kernel["ProblemType"]["DataType"].isBFloat16():
            writer.vgprPool.checkIn(hiBitsMaskVgpr)

        return imod

    """
    Store sum to LDS
    Use the same pattern as local read, except the leading dimension is the sum index instead of free indices
    maxKId is calculated to find out the length of the sum index.
    """
    def storeSumLDS(self, writer, kernel, tP):
        imod = Module("StoreSumLDS")
        # Unregister defined sgpr
        if kernel["ProblemType"]["DataType"].numRegisters() < 1:
            if kernel["ProblemType"]["DataType"].isHalf():
                writer.undefineSgpr("SumUnrollConstOne")

        # bias data type
        diasBpe        = kernel["ProblemType"]["ComputeDataType"].numBytes()
        # get constant parameter
        tile01         = tP["tile01Idx"]
        waveWidth      = writer.states.kernel["WavefrontSize"]

        wReg    = writer.vgprPool.checkOut(1,"wReg") # quotient
        tReg    = writer.vgprPool.checkOut(1,"tReg") # remainder
        kReg    = writer.vgprPool.checkOut(1,"kReg") # remainder
        tmpVgpr = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr")
        tmpVgprRes = ContinuousRegister(tmpVgpr, 2)
        ldsVgpr = writer.vgprPool.checkOut(1,"ldsVgpr")
        ldsVgpr1 = writer.vgprPool.checkOut(1,"ldsVgpr1")
        dummy   = writer.vgprPool.checkOut(1,"dummy")

        # parameter for get each type index
        tile10           = 1 - tile01
        # Must init LraTileProperties in LraTileAssignment
        assert writer.states.lraTileProperties[tile01] and writer.states.lraTileProperties[tile10]

        dividendForKId   = writer.states.lraTileProperties[tile01].dividendForKId
        num1DBlocks      = writer.states.lraTileProperties[tile01].num1DBlocks
        num1DBlocks1     = writer.states.lraTileProperties[tile10].num1DBlocks
        num1DWaves       = writer.states.lraTileProperties[tile01].num1DWaves
        dividedForBlkId  = writer.states.lraTileProperties[tile01].dividedForBlkId
        dividedForBlkId1 = writer.states.lraTileProperties[tile10].dividedForBlkId
        dividedForWaveId = writer.states.lraTileProperties[tile01].dividedForWaveId
        vectorWidth      = writer.states.lraTileProperties[tile01].vectorWidth
        maxKId           = writer.states.lraTileProperties[tile01].maxKId

        strideTile       = maxKId
        strideBlock      = kernel["MatrixInstM"] * strideTile
        strideWave       = kernel["MatrixInstM"] * num1DBlocks * strideTile * vectorWidth

        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            # tile offset
            imod.add(vectorStaticRemainder(dummy, kReg, "Serial", waveWidth, tmpVgprRes, tmpSgprInfo, \
                "0. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth))
            imod.add(vectorStaticRemainder(dummy, tReg, kReg, kernel["MatrixInstN"], tmpVgprRes, tmpSgprInfo, \
                "1. N offset: nIdx = wtid %% MI_N(%u)" % kernel["MatrixInstN"]))
            imod.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), strideTile, tmpSgprInfo, \
                "1. N offset: nOffset = nIdx * nStride(%u)" % strideTile))
            # block offset
            # Here we calculate the coordinate of the block offset, and remove the duplicated blocks.
            # For example we have block = 2x2, source swap = False:
            # 0 1
            # 2 3
            # For tile01 = 0 we should only use 0, 2. For tile01 = 1 we should only use 0, 1.
            imod.add(vectorStaticDivide(wReg, kReg, dividedForBlkId1, tmpVgprRes, \
                "2-1. block offset: bnIdx = wtid / dividedForBlkId1(%u)" % dividedForBlkId1))
            imod.add(vectorStaticRemainder(dummy, wReg, wReg, num1DBlocks1, tmpVgprRes, tmpSgprInfo, \
                "2-1. block offset: bnIdx = bnIdx %% num1DBlocks1(%u)" % num1DBlocks1))
            imod.add(VCmpXEqU32(dst=EXEC(), src0=vgpr(wReg), src1=0, comment="2-1. True if ans = 0"))
            imod.add(vectorStaticDivide(wReg, kReg, dividedForBlkId, tmpVgprRes, \
                "2. block offset: bnIdx = wtid / dividedForBlkId(%u)" % dividedForBlkId))
            imod.add(vectorStaticRemainder(dummy, wReg, wReg, num1DBlocks, tmpVgprRes, tmpSgprInfo, \
                "2. block offset: bnIdx = bnIdx %% num1DBlocks(%u)" % num1DBlocks))
            imod.add(vectorStaticMultiply(vgpr(wReg), vgpr(wReg), strideBlock, tmpSgprInfo, \
                "2. block offset: bnOffset = bnIdx * strideBlock(%u)" % strideBlock))
            imod.add(VAddU32(dst=vgpr(tReg), src0=vgpr(wReg), src1=vgpr(tReg), \
                comment="3. add N and block offset: bnOffset = block and N offset"))
            imod.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), vectorWidth, tmpSgprInfo, \
                "3. apply VectorWidth: bnOffset = bnOffset * vw(%u)" % vectorWidth))
            # unroll offset
            imod.add(vectorStaticDivide(kReg, kReg, dividendForKId, tmpVgprRes, \
                "4. K offset: kIdx = wtid / (MIN(%u) * MIBB(%u))" % (kernel["MatrixInstN"], kernel["MatrixInstB"])))
            imod.add(vectorStaticMultiply(vgpr(kReg), vgpr(kReg), 1, tmpSgprInfo, \
                "4. K offset: lrKOffset = kIdx * mStride(1)"))

            imod.add(VAddU32(dst=vgpr(tReg), src0=vgpr(kReg), src1=vgpr(tReg), \
                comment="5. offset in wave: lrOffset = bnOffset + lrKOffset"))
            # wave offset
            if num1DWaves > 1:
            # FIXME: Should be two cases tile01 == 0 and tile01 == 1
                imod.add(vectorStaticDivide(wReg, "Serial", dividedForWaveId, tmpVgprRes, \
                    "6. wave offset in N dimen: wtid = tid / dividedForWaveId(%u)" % dividedForWaveId))
                # Here we calculate the coordinate of the block offset, and remove the duplicated wave.
                if tile01 == 0:
                    imod.add(vectorStaticDivide(dummy, wReg, num1DWaves, tmpVgprRes, \
                    "6-1.mask duplicated: wtid0 = wtid / num1DWaves(%u)" % num1DWaves))
                else:
                    imod.add(vectorStaticDivide(dummy, "Serial", waveWidth, tmpVgprRes, \
                        "6-1. wave offset in N dimen (waveWidth): wtid_1 = tid / waveWidth(%u)" % waveWidth))
                    imod.add(vectorStaticRemainder(dummy, dummy, dummy, kernel["MIWaveGroup"][0], tmpVgprRes, tmpSgprInfo, \
                        "6-1. wave offset in M dimen (MIWaveGroup0): wtid0 = wtid_1 %% MIWaveGroup0(%u)" % kernel["MIWaveGroup"][0]))
                imod.add(VCmpXEqU32(dst=EXEC(), src0=vgpr(dummy), src1=0, comment="6-1. True if ans = 0"))
                imod.add(vectorStaticRemainder(dummy, wReg, wReg, num1DWaves, tmpVgprRes, tmpSgprInfo, \
                    "6. wave offset in M dimen: wtid0 = wtid %% num1DWaves(%u)" % num1DWaves))
                imod.add(vectorStaticMultiply(vgpr(wReg), vgpr(wReg), strideWave, tmpSgprInfo, \
                    "6. wave offset in M dimen: wOffset = wtid0 * W0Stride(%u)" % strideWave))
                imod.add(VAddU32(dst=vgpr(tReg), src0=vgpr(wReg), src1=vgpr(tReg), \
                    comment="7. final local read offset: flrOffset = lrOffset + WOffset"))
            imod.add(VLShiftLeftB32(dst=vgpr(tReg), src=vgpr(tReg), shiftHex=hex(log2(diasBpe)), \
                comment="offset = offset*bpe" ))
        # release register
        writer.vgprPool.checkIn(wReg)
        writer.vgprPool.checkIn(kReg)
        writer.vgprPool.checkIn(tmpVgpr)
        writer.vgprPool.checkIn(ldsVgpr)
        writer.vgprPool.checkIn(ldsVgpr1)
        writer.vgprPool.checkIn(dummy)

        # Add barrier here to avoid race condition if lds offset starts from 0
        if kernel["LdsOffsetBias"] == 0:  # Compatible for both GSU1 and GSUM
          imod.add(SBarrier(comment="Wait for all wavefronts"))

        MIWaveGroupShape = [ kernel["MatrixInstM"] * kernel["MatrixInstBM"] * kernel["MIWaveGroup"][0] * kernel["VectorWidthA"], \
                            kernel["MatrixInstN"] * kernel["MatrixInstBN"] * kernel["MIWaveGroup"][1] * kernel["VectorWidthB"]]
        numReadPerTileVector = vectorWidth
        numVectorsPerTile    = kernel["MIWaveTile"][tile01] // numReadPerTileVector
        idx = 0
        for vIdx in range(0, numVectorsPerTile):
            for eIdx in range(0, numReadPerTileVector):
                # normal case
                offset_val = (eIdx + vIdx * MIWaveGroupShape[tile01]) * maxKId * kernel["ProblemType"]["ComputeDataType"].numBytes()
                bps = kernel["ProblemType"]["ComputeDataType"].numBytes()
                ds  = DSModifiers(offset=offset_val)
                dst = vgpr(tReg)
                vgprStr = "ValuSum+%u"%idx
                if bps==2:
                    imod.add(DSStoreB16(dstAddr=dst, src=vgpr(vgprStr), ds=ds, comment="local store bias"))
                elif bps==4:
                    imod.add(DSStoreB32(dstAddr=dst, src=vgpr(vgprStr), ds=ds, comment="local store bias"))
                elif bps==8:
                    imod.add(DSStoreB64(dstAddr=dst, src=vgpr(vgprStr, 2), ds=ds, comment="local store bias"))
                else:
                    assert 0
                idx += 1
        # tReg
        writer.vgprPool.checkIn(tReg)
        imod.add(SSetMask(dst=EXEC(), src=-1, comment="reset mask" ))
        # For later write in Global Write Batch
        return imod
