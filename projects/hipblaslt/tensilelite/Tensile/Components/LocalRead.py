################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

from rocisa.code import Module
from rocisa.container import DSModifiers, vgpr, sgpr, SDWAModifiers, VOP3PModifiers, ContinuousRegister
from rocisa.enum import HighBitSel, SelectBit, InstType
from rocisa.instruction import SMovB32, SWaitCnt, VOrB32, VPermB32, VLShiftLeftOrB32, \
                            VMovB32, VMovB64, VLShiftRightB32, VCvtFP8toF16, VCvtScalePkFP8toF16, VCvtFP8toF32, VCvtScaleFP8toF16, VCvtScalePkFP8toF16, \
                            VCvtPkF32toBF16, VCvtBF16toFP32, PVCvtBF16toFP32, VDot2CF32BF16, SNop, VSubF32, VSwapB32, MFMAInstruction, \
                            ECvtPkFP8toF32, ECvtF32toF16

from ..Component import LocalRead

from math import ceil

class LocalReadVALU(LocalRead):
    kernel = {"EnableMatrixInstruction": False}

    """
    Local Read: Do It A/B
    iui = Inner Unroll Idx
    epsi = expand pointer swap index. Only used for PAP
    """
    def __call__(self, writer, kernel, bufferIdx, iui, epsi, tP):
        tc = tP["tensorChar"]

        if tc == "A":
            writer.states.localReadDoCntA += 1
        elif tc == "Metadata":
            writer.states.localReadDoCntMetadata += 1
        else:
            writer.states.localReadDoCntB += 1

        tile01            = tP["tile01Idx"]
        imod              = Module("LocalReadDo%s_I%s"%(tc,iui))
        pack              = Module("pack%s_I%s"%(tc,iui))
        instruction       = tP["localReadInstruction"]
        numOffsets        = instruction.numOffsets
        blockWidth        = instruction.blockWidth
        offsetMultiplier  = 1 # instruction.offsetMultiplier
        valuIdx           = 0
        # dot2: currently only support unroll major LDS
        if kernel["UseDotInstruction"]:
            numVectorsPerTile = kernel["ThreadTile%u"%tile01]
            numReadsPerVector = ceil((writer.states.lrvwUnrollA * tP["bpe"]) / (blockWidth*4)) # bytes/register
            LdsPad            = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
            tileStride        = kernel["_DepthU%s"%tc] + LdsPad if kernel["UnrollMajorLDS%s" % tP["tensorChar"]] else 1
        else:
            numVectorsPerTile = (kernel["ThreadTile%u"%tile01]//kernel["VectorWidthA"])
            numReadsPerVector = ceil((kernel["VectorWidthA"] * tP["bpe"]) / (blockWidth*4)) # bytes/register

        for vIdx in range(0, numVectorsPerTile):
            for rIdx in range(0, int(numReadsPerVector)):
                localReadCode = imod.add(Module("LocalRead%s Valu%u"%(tc,valuIdx)))
                paramList     = []
                destVgpr      = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuIdx), blockWidth)

                # paramList.append(destVgpr)
                # paramList.append(vgpr("LocalReadAddr%s"%tc))

                for oIdx in range(0, numOffsets):
                    # dot2
                    if kernel["UseDotInstruction"]:
                        paramList.append(int(((rIdx*blockWidth + kernel["SubGroup%u"%tile01] * (vIdx*numOffsets+oIdx) * tileStride \
                            + tP["localReadOffset"]) * tP["bpe"] + tP["localReadSwapByteOffset"]) // offsetMultiplier))
                    else:
                        paramList.append(int(((rIdx*blockWidth + kernel["SubGroup%u"%tile01] * (vIdx*numOffsets+oIdx)*kernel["VectorWidthA"] \
                            + tP["localReadOffset"]) * tP["bpe"] + tP["localReadSwapByteOffset"]) // offsetMultiplier))
                    # print("Debug: Matrix{}, rIdx offset {}, vIdx offset {}, bpe {}, net offset {}".format( \
                    #     tP["tensorChar"], \
                    #     rIdx * blockWidth, \
                    #     kernel["SubGroup%u" % tP["tensorIdx"]] * (vIdx * numOffsets + oIdx) * kernel["VectorWidth"] + tP["localReadOffset"], \
                    #     tP["bpe"], \
                    #     paramList[-1]))
                # paramTuple = tuple(paramList)
                num = paramList[0] //65536
                paramList[0] = paramList[0] - num * 65536
                srcVgpr=vgpr("LocalReadAddr%s+%d"%(tc,num))

                if numOffsets == 1:
                    addrIdx = paramList[0] // 65536
                    srcAddr=vgpr("LocalReadAddr%s+%u"%(tc, addrIdx))
                    paramList[0] -= addrIdx * 65536
                    ds = DSModifiers(na=1, offset=paramList[0])
                if numOffsets == 2:
                    ds = DSModifiers(na=2, offset0=paramList[0], offset1=paramList[1])
                    srcAddr=vgpr("LocalReadAddr%s"%tc)
                LocalReadX = instruction.getInst()
                self._emitLdsRead(writer, kernel, tP, LocalReadX, dst=destVgpr, src=srcAddr, ds=ds, module=localReadCode)
                valuIdx += blockWidth

                # TODO - handle vector-load
                with writer.allocTmpSgpr(1) as tmpSgprInfo:
                    tmpSgpr = tmpSgprInfo.idx
                    if writer.db["CheckValue1%s" % tc]:
                        dbgVgpr = destVgpr
                        dbgVgprList = destVgpr.split("v[")
                        if len(dbgVgprList) == 1: # vIdx, no []
                            dbgVgpr = dbgVgprList[0]
                        else:
                            # We only check the first one now
                            # TODO: Handle vector, but need to take care the last one
                            dbgVgprList = (dbgVgprList[1].split("]")[0]).split(':')
                            dbgVgpr = "v[%s]"%dbgVgprList[0]

                        # localReadCode.addInst("s_waitcnt lgkmcnt(0)", "CheckValue1 wait for LDS read")
                        localReadCode.add(SWaitCnt(dscnt=0, comment="CheckValue1 wait for lds read"))
                        if writer.asmCaps["SeparateVscnt"]:
                            # localReadCode.addInst( "s_waitcnt_vscnt", -2, "0", "")
                            localReadCode.add(SWaitCnt(vlcnt=0, vscnt=0))

                        if kernel["ProblemType"]["DataType"].isHalf():
                            localReadCode.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(0x3c003c00), comment="CheckValue1: FP16")) # packed 1s
                            localReadCode.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                        elif kernel["ProblemType"]["DataType"].isBFloat16():
                            localReadCode.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(0x3f803f80), comment="CheckValue1: BF16")) # packed 1s
                            localReadCode.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                        # TODO - Check if this works
                        if kernel["ProblemType"]["DataType"].isInt8():
                            localReadCode.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(0x01010101), comment="CheckValue1: INT8")) # packed 1s
                            localReadCode.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                        # TODO - Check if this works
                        elif kernel["ProblemType"]["DataType"].isInt8x4():
                            localReadCode.add(writer.assert_eq( dbgVgpr, 1))

                        elif kernel["ProblemType"]["DataType"].isSingle():
                            localReadCode.add(writer.assert_eq( dbgVgpr, 1.0) )

        return imod, pack, Module()

class LocalReadMFMA(LocalRead):
    kernel = {"EnableMatrixInstruction": True}

    # LDS size is increased on gfx950. const offset is still 16-bit.
    # this function handles both LDS size < 64K and LDS size >= 64K
    def cal_offset_srcAddr(self, maxLDSConstOffset, tc, offset):
        num = offset // maxLDSConstOffset
        offset_val = offset - num * maxLDSConstOffset
        srcAddr = vgpr("LocalReadAddr%s+%u" %(tc, num))
        return offset_val, srcAddr

    # Vreg Value layout (assuming MIInputPerThread = 8)
    # (1) local read dst
    #       T: index transpose case (lrvwTile>1 and not transposeCode)
    #       X: lrvwTile == 1 or not UseDirect32XEmulation
    #       interleave T reg case, (2)
    #  local read + lrvwTile == 2
    #   valOffset  0 -  1 (+16): X/T reg,  0 -  1 (+16)
    #   valOffset  2 -  3 (+16): X/T reg,  8 -  9 (+16)
    #   valOffset  4 -  5 (+16): X/T reg,  2 -  3 (+16)
    #   valOffset  6 -  7 (+16): X/T reg, 10 - 11 (+16)
    #   valOffset  8 -  9 (+16): X/T reg,  4 -  5 (+16)
    #   valOffset 10 - 11 (+16): X/T reg, 12 - 13 (+16)
    #   valOffset 12 - 13 (+16): X/T reg,  6 -  7 (+16)
    #   valOffset 14 - 15 (+16): X/T reg, 14 - 15 (+16)
    #  local read + lrvwTile == 4
    #   valOffset  0 -  3: X/T reg,  0 -  3
    #   valOffset  4 -  7: X/T reg,  8 - 11
    #   valOffset  8 - 11: X/T reg, 16 - 19
    #   valOffset 12 - 15: X/T reg, 24 - 27
    #   valOffset 16 - 19: X/T reg,  4 -  7
    #   valOffset 20 - 23: X/T reg, 12 - 15
    #   valOffset 24 - 27: X/T reg, 20 - 23
    #   valOffset 28 - 31: X/T reg, 28 - 31
    # (2) interleave T reg (src of high value cvt, local read dest)
    #     For local read dst, apply this logic on top of above (1)
    #   valOffset  0 -  3: T reg,  0 -  3
    #   valOffset  4 -  7: X reg,  4 -  7
    #   valOffset  8 - 11: T reg,  4 -  7
    #   valOffset 12 - 15: X reg, 12 - 15
    #   valOffset 16 - 19: T reg,  8 - 11
    #   valOffset 20 - 23: X reg, 20 - 23
    #   valOffset 24 - 27: T reg, 12 - 15
    #   valOffset 28 - 31: X reg, 28 - 31
    # (3) src of high value
    # (3-1) not UseDirect32XEmulation case
    #   valOffset  0 - 31: X reg,  0 - 31
    # (3-2) (not 3-1) and interleave T reg
    #   same as above (2)
    # (3-3) (not 3-1) and not (interleave T reg) and transpose
    #   valOffset  0 - 31: T reg,  0 - 31
    # (4) dst of cvt (high, low)
    # (4-1) not (interleave T reg) and transpose
    #   valOffset  0 - 31: T reg,  0 - 31
    # (4-2) (not 4-1)
    #   valOffset  0 - 31: X reg,  0 - 31
    # Non transpose case
    #  src (dst = False)
    #  dst (dst = True)
    #   valOffset  0 - 31: X reg,  0 - 31
    # Do index transpose if transpose is true
    def TXInterleaveLayoutIdx(self, idx, miInputPerThread=8):
        halfM = miInputPerThread // 2
        retIdx = idx
        XTchar = "X"
        if idx % miInputPerThread < halfM:
            XTchar = "T"
            retIdxUpper = (idx // miInputPerThread) * halfM
            retIdxLower = idx % halfM
            retIdx = retIdxUpper + retIdxLower
        return retIdx, XTchar

    @staticmethod
    def _genDsReadConvTable(miInputPerThread, lrvwTile):
        halfM = miInputPerThread // 2
        numRows = lrvwTile
        colsPerRow = halfM // lrvwTile
        table = []
        for group in range(2):
            for col in range(colsPerRow):
                for row in range(numRows):
                    table.append(row * miInputPerThread + col * lrvwTile + group * halfM)
        return table

    def getTransposeXorTIndex(self, writer, kernel, idx, tc, lrvwTile, dst, isNext, localRead=False):
        mipt = kernel["MIInputPerThread"]
        assert lrvwTile <= 4, "lrvwTile is %d"%lrvwTile
        abmatrixinfo = writer.states.a if tc == 'A' else writer.states.b
        if isNext:
            useTransposeCode = abmatrixinfo.useTransposeCodeNext
            useDirect32XEmulation = abmatrixinfo.useDirect32XEmulationNext
        else:
            useTransposeCode = abmatrixinfo.useTransposeCodeThis
            useDirect32XEmulation = abmatrixinfo.useDirect32XEmulationThis
        TF32EmuInterleaveTreg = abmatrixinfo.TF32EmuInterleaveTreg
        swapBlockSizeSub = abmatrixinfo.swapBlockSizeSub
        blockH = mipt
        blockW = lrvwTile
        blockSize = blockH * blockW
        # default: no change
        XTchar = "X"
        if localRead:
            if lrvwTile > 1:
                dsReadConvTable = self._genDsReadConvTable(mipt, lrvwTile)
                blockIdx = idx // blockSize
                idxInBlk = idx % blockSize
                tableIdx = idxInBlk // blockW
                idx = dsReadConvTable[tableIdx] + blockIdx * blockSize
            if TF32EmuInterleaveTreg:
                if writer.states.doFullPackCodePrefetch:
                    idx, XTchar = self.TXInterleaveLayoutIdx(idx, kernel["MIInputPerThread"])
                else:
                    withinGroup = idx % 8
                    if withinGroup < 4:
                        XTchar = "T"
                        idx = (idx // 8) * 4 + withinGroup
            elif writer.states.doFullPackCodePrefetch:
                # full pack code prefetch case, dst of local read is always T reg
                # use T as destination
                XTchar = "T"
                # always use same swapBlockSizeSub(=0)
                idx = idx % swapBlockSizeSub
            return idx, XTchar
        if not dst:
            # src case
            if (not useDirect32XEmulation):
                # temp reg case (both lrvwTile==1 and >1)
                # no register conversion (use X 0-31)
                pass
            elif TF32EmuInterleaveTreg:
                if writer.states.doFullPackCodePrefetch:
                    idx, XTchar = self.TXInterleaveLayoutIdx(idx, kernel["MIInputPerThread"])
                else:
                    withinGroup = idx % 8
                    if withinGroup < 4:
                        XTchar = "T"
                        idx = (idx // 8) * 4 + withinGroup
            elif writer.states.doFullPackCodePrefetch:
                # full pack code prefetch case, dst of local read is always T reg
                XTchar = "T" # use T reg for wider local read + transpose code
                # always use same swapBlockSizeSub(=0)
                idx = idx % swapBlockSizeSub
                if lrvwTile > 1 and not useTransposeCode:
                    # do index transpose for src only
                    idx = self.getTransposeIndex(kernel, idx, lrvwTile)
        else:
            # dst case
            # always "X"
            pass
        return idx, XTchar

    # transpose vgprs for lrvwTile > 1
    # From
    #  lrvwTile==2
    #   reg  0 reg  1 reg 16 reg 17
    #   reg  8 reg  9 reg 24 reg 25
    #   reg  2 reg  3 reg 18 reg 19
    #   reg 10 reg 11 reg 26 reg 27
    #   reg  4 reg  5 reg 20 reg 21
    #   reg 12 reg 13 reg 28 reg 29
    #   reg  6 reg  7 reg 22 reg 23
    #   reg 14 reg 15 reg 30 reg 31
    #  lrvwTile==4
    #   reg  0 reg  1 reg  2 reg  3
    #   reg  8 reg  9 reg 10 reg 11
    #   reg 16 reg 17 reg 18 reg 19
    #   reg 24 reg 25 reg 26 reg 27
    #   reg  4 reg  5 reg  6 reg  7
    #   reg 12 reg 13 reg 14 reg 15
    #   reg 20 reg 21 reg 22 reg 23
    #   reg 28 reg 29 reg 30 reg 31
    # To (common for lrvwTile==2 and 4)
    #   reg 0 reg  8 reg 16 reg 24
    #   reg 1 reg  9 reg 17 reg 25
    #   reg 2 reg 10 reg 18 reg 26
    #   reg 3 reg 11 reg 19 reg 27
    #   reg 4 reg 12 reg 20 reg 28
    #   reg 5 reg 13 reg 21 reg 29
    #   reg 6 reg 14 reg 22 reg 30
    #   reg 7 reg 15 reg 23 reg 31
    def getTransposeIndex(self, kernel, idx, lrvwTile):
        if lrvwTile == 1:
            return idx
        mipt = kernel["MIInputPerThread"]
        assert lrvwTile <= 4, "lrvwTile is %d"%lrvwTile
        blockSize = mipt * lrvwTile
        blockIdx = idx // blockSize
        idxInBlk = idx % blockSize
        dsTable = self._genDsReadConvTable(mipt, lrvwTile)
        swappedIdx = dsTable[idxInBlk % mipt] + (idxInBlk // mipt) + blockIdx * blockSize
        return swappedIdx

    # if prefetch is for NextLoop or not
    def getIsNext(self, writer, kernel, tc, bufferIdx):
        numIterPerCoalescedRead = writer.states.numIterPerCoalescedReadA if tc == "A" else writer.states.numIterPerCoalescedReadB
        isNext = bufferIdx < writer.states.numItersPLR * numIterPerCoalescedRead
        # subIter case
        if kernel["ForceUnrollSubIter"]:
            isNext = writer.states.SubTileIdx == 0
        return isNext

    # Get vgpr string for Emu from mfmaIter
    def getVgprStrForEmuMfma(self, writer, kernel, tc, bufferIdx, iui, valOffset, lrvwTile, u, dst=False, localRead=False):
        # subIter case
        if kernel["ForceUnrollSubIter"]:
            # isNext means using prefetch local read for next iter
            # MFMA case, isNext means smaller vreg index
            if tc == "A":
                isNext = (u & 1) == 0
            else:
                isNext = u < 2
        else:
            isNext = self.getIsNext(writer, kernel, tc, bufferIdx)
        v0Idx, v0XTchar = self.getTransposeXorTIndex(writer, kernel, valOffset, tc, lrvwTile, dst, isNext, localRead=localRead)
        return "Valu%s_%s%u_I%u+%u"%(tc, v0XTchar, bufferIdx, iui, v0Idx)
    # Get vgpr string for Emu
    def getVgprStrForEmu(self, writer, kernel, tc, bufferIdx, iui, valOffset, lrvwTile, dst=False, localRead=False):
        isNext = self.getIsNext(writer, kernel, tc, bufferIdx)
        v0Idx, v0XTchar = self.getTransposeXorTIndex(writer, kernel, valOffset, tc, lrvwTile, dst, isNext, localRead=localRead)
        return "Valu%s_%s%u_I%u+%u"%(tc, v0XTchar, bufferIdx, iui, v0Idx)
    # Get vgpr for Emu
    def getVgprForEmu(self, writer, kernel, tc, bufferIdx, iui, valOffset, lrvwTile, vgprLen=1, dst=False, localRead=False):
        v0 = vgpr(self.getVgprStrForEmu(writer, kernel, tc, bufferIdx, iui, valOffset, lrvwTile, dst=dst, localRead=localRead), vgprLen)
        return v0
    def get2VgprForEmu(self, writer, kernel, tc, bufferIdx, valOffset, iui, lrvwTile, vgprLen=1, dst=False, localRead=False):
        # create 4 continuous vgpr set
        v0 = self.getVgprForEmu(writer, kernel, tc, bufferIdx, iui, valOffset + 0 * vgprLen, lrvwTile, vgprLen=vgprLen, dst=dst, localRead=localRead)
        v1 = self.getVgprForEmu(writer, kernel, tc, bufferIdx, iui, valOffset + 1 * vgprLen, lrvwTile, vgprLen=vgprLen, dst=dst, localRead=localRead)
        return v0, v1
    def get4VgprForEmu(self, writer, kernel, tc, bufferIdx, valOffset, iui, lrvwTile, vgprLen=1, dst=False, localRead=False):
        # create 4 continuous vgpr set
        v0 = self.getVgprForEmu(writer, kernel, tc, bufferIdx, iui, valOffset + 0 * vgprLen, lrvwTile, vgprLen=vgprLen, dst=dst, localRead=localRead)
        v1 = self.getVgprForEmu(writer, kernel, tc, bufferIdx, iui, valOffset + 1 * vgprLen, lrvwTile, vgprLen=vgprLen, dst=dst, localRead=localRead)
        v2 = self.getVgprForEmu(writer, kernel, tc, bufferIdx, iui, valOffset + 2 * vgprLen, lrvwTile, vgprLen=vgprLen, dst=dst, localRead=localRead)
        v3 = self.getVgprForEmu(writer, kernel, tc, bufferIdx, iui, valOffset + 3 * vgprLen, lrvwTile, vgprLen=vgprLen, dst=dst, localRead=localRead)
        return v0, v1, v2, v3

    # do transpose with v_swap
    def transposeLRVregs(self, kernel, module, tc, bufferIdx, iui, writer, lrvwTile, totalRegs, subTileIdx):
        start = subTileIdx * totalRegs
        last = start + totalRegs - 1
        if lrvwTile == 1:
            return module
        done = [start, last]
        for idx in range(start + 1,last):
            vSrc = self.getVgprForEmu(writer, kernel, tc, bufferIdx, iui, idx, 1, localRead=True)
            newIdx = self.getTransposeIndex(kernel, idx, lrvwTile)
            if idx in done or idx == newIdx:
                # Already done or same vreg. No need to swap
                done.append(idx)
                continue
            vDst = self.getVgprForEmu(writer, kernel, tc, bufferIdx, iui, newIdx, 1, localRead=True)
            commentStr = ""
            if idx == last - 1:
                commentStr = "__PACK_PRE_" + tc + "_I%d__"%(iui)
            module.add(VSwapB32(dst=vDst, src=vSrc, comment="swap %d and %d"%(newIdx, idx) + commentStr))
            done.append(idx)
            done.append(newIdx)

    def initTmpVregForPack(self, kernel, writer, tct, index, bufferIdx, baseValuiIdx, iui, module, lrvwTile, tmpvgpr, useDirect32XEmulation):
        valOffset = baseValuiIdx + index
        src0, src1 = self.get2VgprForEmu(writer, kernel, tct, bufferIdx, valOffset, iui, lrvwTile, vgprLen=2, dst=False)
        if index % 8 == 0: # First half of registers use tmp registers
            if (not useDirect32XEmulation):
                # First half of the registers will be overwritten. Store FP32 values in tmp
                overlap = True
                while overlap == True:
                    val1 = writer.vgprPool.checkOutAligned(2, 2, "x32f temp val1")
                    val2 = writer.vgprPool.checkOutAligned(2, 2, "x32f temp val2")
                    overlap = False
                    if baseValuiIdx not in writer.states.tmpvgpr:
                        writer.states.tmpvgpr[baseValuiIdx] = []
                    if tct == "B":
                        overlap |= val1 in writer.states.tmpvgpr[baseValuiIdx]
                        overlap |= val2 in writer.states.tmpvgpr[baseValuiIdx]
                    tmpvgpr.append(val1)
                    tmpvgpr.append(val2)
                module.add(VMovB64(dst=vgpr(val1, 2), src=src0))
                module.add(VMovB64(dst=vgpr(val2, 2), src=src1))

    def releaseTmpVregForPack(self, kernel, writer, tc, baseValuiIdx, tmpvgprFP32, useDirect32XEmulation):
        if not (kernel["MatrixInstM"] == 16 and kernel["MatrixInstK"] == 16):
            if not useDirect32XEmulation:
                while len(tmpvgprFP32):
                    tmp = tmpvgprFP32.pop()
                    writer.vgprPool.checkIn(tmp)
                # if A, write tmp vgpr to writer state
                if tc == "A":
                    if baseValuiIdx not in writer.states.tmpvgpr:
                        writer.states.tmpvgpr[baseValuiIdx] = []
                    if tmp not in writer.states.tmpvgpr[baseValuiIdx]:
                        writer.states.tmpvgpr[baseValuiIdx].append(tmp)
                # if B, free tmp vgpr from writer state
                elif tc == "B":
                    if baseValuiIdx in writer.states.tmpvgpr:
                        writer.states.tmpvgpr[baseValuiIdx] = []

    # 8 registers were read in fp32. Write to 2 of 4 registers with high bits
    # Input: vgprValuA/B_X/T_ i ... i + 7 -> FP32
    # Output: vgprValuA/B_X_ i .. i + 3 -> BF16 High bits
    def pack4HiBits(self, kernel, writer, tct, index, bufferIdx, baseValuiIdx, iui, module, lrvwTile, commentForSchedule1, useDirect32XEmulation):
        valOffset = baseValuiIdx + index
        dstValOffset = baseValuiIdx + index // 2
        v0, v1, v2, v3 = self.get4VgprForEmu(writer, kernel, tct, bufferIdx, valOffset, iui, lrvwTile, dst=False)
        dst0, dst1 = self.get2VgprForEmu(writer, kernel, tct, bufferIdx, dstValOffset, iui, lrvwTile, dst=True)
        module.add(VCvtPkF32toBF16(dst=dst0, src0=v0, src1=v1))
        commentStr = commentForSchedule1
        # do not put comment for scheduling in the following cases
        # - UseMFMAF32XEmulation
        #   Delay the comment to the 2nd mfma
        # - index % 8 == 0
        #   we need to put schedule comment at index % 8 == 4
        # - not useDirect32XEmulation
        #   tmp vreg case, tmp vreg can be same between A and B
        #   put schedule comment only at final pack
        #if kernel["UseMFMAF32XEmulation"] or (index % 8 == 0) or (not useDirect32XEmulation):
        #    commentStr = ""
        module.add(VCvtPkF32toBF16(dst=dst1, src0=v2, src1=v3, comment=commentStr))

    def pack4LowBitsStep1(self, kernel, writer, tc, valuiIdx, bufferIdx, iui, packCodeT, lrvwTile, tmpvgprFP32, commentForSchedule1, useDirect32XEmulation):
        baseValuiIdx = valuiIdx - valuiIdx % 8
        if (valuiIdx % 8) == 0 and not useDirect32XEmulation:
            # get 4 tmp vgprs
            tmpIdx = len(tmpvgprFP32) - 2
            vTBase = tmpvgprFP32[tmpIdx + 0] # str(v0t.regName) does not work for tmp vreg. Put vgpr index directly here
            v0t = vgpr(tmpvgprFP32[tmpIdx + 0])
            v1t = vgpr(tmpvgprFP32[tmpIdx] + 1)
            v2t = vgpr(tmpvgprFP32[tmpIdx + 1])
            v3t = vgpr(tmpvgprFP32[tmpIdx + 1] + 1)
        else:
            v0t, v1t, v2t, v3t = self.get4VgprForEmu(writer, kernel, tc, bufferIdx, valuiIdx, iui, lrvwTile)
            vTBase = str(v0t.regName)
        srcVIdxHi = (valuiIdx-baseValuiIdx)//2 + baseValuiIdx
        # vHi0, vHi1 is dst for Upper cvt
        vHi0, vHi1 = self.get2VgprForEmu(writer, kernel, tc, bufferIdx, srcVIdxHi, iui, lrvwTile, dst=True)
        # interleaveTreg case (means dst and src are same (dot2 or mfma))
        if kernel["UseMFMAF32XEmulation"]:
            vHiBase = str(vHi0.regName)
            idMat = vgpr(writer.states.startVgprIdentityMatrix,2)
            # We use a single MFMA 4x4x4_16b to perform 4 `vT - vHi` operations.
            # - A is set to negative identity matrix
            # - no need for DPP as B has the same layout as C & D
            commentStr = ""
            if commentForSchedule1 != "":
                # add comment for scheduling (__TF32_1_)
                # UseMFMAF32XEmulation case, put the mark at second mfma4x4. Skip here
                commentStr = commentForSchedule1 + ": "
            packCodeT.add(MFMAInstruction(instType=InstType.INST_BF16, accType=InstType.INST_F32, variant=[4,4,4,16], mfma1k=False,acc=vgpr(vTBase,4), a=idMat, b=vgpr(vHiBase,2), acc2=vgpr(vTBase,4),
            comment="Calculate low bits for TF32 emulation%s"%commentStr))
        else:
            tmp = writer.vgprPool.checkOut(1, "x32f tmp mod 4")
            # Compute low bits = fp32(highBF16(A/B)) - fp32(A/B)
            if kernel["UseDot2F32XEmulation"]:
                packCodeT.add(VDot2CF32BF16(dst=v0t, src0=hex(0x8000bf80), src1=vHi0))
                packCodeT.add(VDot2CF32BF16(dst=v1t, src0=hex(0xbf800000), src1=vHi0))
            else:
                packCodeT.add(PVCvtBF16toFP32(dst=vgpr(tmp), src=vHi0, comment="begin"+str(valuiIdx)))
                packCodeT.add(VSubF32(dst=v0t, src0=v0t, src1=vgpr(tmp)))
                packCodeT.add(VCvtBF16toFP32(dst=vgpr(tmp), src=vHi0, vgprMask=None, vi=1))
                packCodeT.add(VSubF32(dst=v1t, src0=v1t, src1=vgpr(tmp)))
            if kernel["UseDot2F32XEmulation"] and (valuiIdx % 8) == 0:
                packCodeT.add(VDot2CF32BF16(dst=v2t, src0=hex(0x8000bf80), src1=vHi1))
                packCodeT.add(VDot2CF32BF16(dst=v3t, src0=hex(0xbf800000), src1=vHi1))
            else:
                # We use cvt+sub pair since dot2 requires adding 4 wait states.
                packCodeT.add(PVCvtBF16toFP32(dst=vgpr(tmp), src=vHi1))
                packCodeT.add(VSubF32(dst=v2t, src0=v2t, src1=vgpr(tmp)))
                packCodeT.add(VCvtBF16toFP32(dst=vgpr(tmp), src=vHi1, vgprMask=None, vi=1))
                packCodeT.add(VSubF32(dst=v3t, src0=v3t, src1=vgpr(tmp), comment="end"))
            writer.vgprPool.checkIn(tmp)

    def pack4LowBitsFinal(self, kernel, writer, tc, valuiIdx, bufferIdx, iui, packCodeT, lrvwTile, tmpvgprFP32, useDirect32XEmulation, noComment=False):
        baseValuiIdx = valuiIdx - valuiIdx % 8
        # on last iteration, store lower bits in last 4 registers
        if not (kernel["MatrixInstM"] == 16 and kernel["MatrixInstK"] == 16):
            if useDirect32XEmulation:
                v0, v1, v2, v3 = self.get4VgprForEmu(writer, kernel, tc, bufferIdx, baseValuiIdx, iui, lrvwTile)
            else:
                #v0, v1, v2, v3 = self.get4TmpVgprForEmu(writer, kernel, tc, bufferIdx, baseValuiIdx, iui)
                tmpIdx = len(tmpvgprFP32) - 2
                v0 = vgpr(tmpvgprFP32[tmpIdx + 0])
                v1 = vgpr(tmpvgprFP32[tmpIdx + 0] + 1)
                v2 = vgpr(tmpvgprFP32[tmpIdx + 1])
                v3 = vgpr(tmpvgprFP32[tmpIdx + 1] + 1)
            v4, v5, v6, v7 = self.get4VgprForEmu(writer, kernel, tc, bufferIdx, baseValuiIdx + 4, iui, lrvwTile)
            # dst regs
            # v4 and v4d should be different in useTransposeCode=False case
            v4d, v5d, v6d, v7d = self.get4VgprForEmu(writer, kernel, tc, bufferIdx, baseValuiIdx + 4, iui, lrvwTile, dst=True)
            # remove s_nop by scheduling mfma4x4 right after first 4 cvt for high
            # we will have mfma hi * hi + 4 final cvt A between mfma4x4 B and final conv B
            # (more instructions between mfma4x4 B and final conv A)
            # however, we still need nop in tail loop case (A, B scheduling is not interleaved in tail loop) and CMS
            # we still need s_nop in main loop, NGLL, NLL if number MIWaveTileA/B are different.
            # s_nop insertion is handled in _interleavePackAB
            packCodeT.add(VCvtPkF32toBF16(dst=v7d, src0=v6, src1=v7, comment="pack final begin"))
            packCodeT.add(VCvtPkF32toBF16(dst=v6d, src0=v4, src1=v5))
            packCodeT.add(VCvtPkF32toBF16(dst=v5d, src0=v2, src1=v3))
            commentStr = "" if noComment else "__TF32_2_" + tc + "_%d pack final end"%(baseValuiIdx//8)
            packCodeT.add(VCvtPkF32toBF16(dst=v4d, src0=v0, src1=v1, comment=commentStr))
    """
    Local Read: Do It A/B
    iui = Inner Unroll Idx
    epsi = expand pointer swap index. Only used for PAP
    """
    def __call__(self, writer, kernel, bufferIdx, iui, epsi, tP):
        imod = Module("LocalReadDo%s_I%s" % (tP["tensorChar"],iui))
        subTileIdx = writer.states.SubTileIdx if kernel["ForceUnrollSubIter"] else 0 # use SubTileIdx only for ForceUnrollSubIter

        tc = tP["tensorChar"]
        if tc == "A":
            writer.states.localReadDoCntA += 1
        elif tc == "MXSA":
            writer.states.localReadDoCntMXSA += 1
        elif tc == "Metadata":
            writer.states.localReadDoCntMetadata += 1
        elif tc == "B":
            writer.states.localReadDoCntB += 1
        elif tc == "MXSB":
            writer.states.localReadDoCntMXSB += 1
        else:
            raise Exception(f"unsupport tc %s{tc}")

        isgfx950 = kernel["ISA"][:2] == (9, 5)
        isgfx950mx = isgfx950 and ("MXS" in tc)
        MacDataType      = f"MacDataType{tc}" if(tc=="A" or tc=="B") else "DataType"
        tile01           = tP["tile01Idx"]
        instruction      = tP["localReadInstruction"]
        bpr              = 4 # bytes/register

        numOffsets       = instruction.numOffsets
        blockWidth       = instruction.blockWidth
        unrollBlockWidth = instruction.blockWidth if kernel["UnrollMajorLDS%s"%tc] else tP["bpeDS"]/4
        tileBlockWidth   = tP["bpeDS"]/4 if kernel["UnrollMajorLDS%s"%tc] else instruction.blockWidth

        vectorWidth  = kernel["VectorWidth%s"%tc]

        numSubTiles = kernel["numSubTiles"]
        MIWaveGroupShape = [ kernel["MatrixInstM"] * kernel["MatrixInstBM"] * kernel["MIWaveGroup"][0] * kernel["VectorWidthA"], \
                            kernel["MatrixInstN"] * kernel["MatrixInstBN"] * kernel["MIWaveGroup"][1] * kernel["VectorWidthB"]]

        LdsPad           = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
        tileStride       = 1
        UnrollStride     = kernel["MacroTile%s" % tP["tensorChar"]] + LdsPad
        if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
            tileStride   = kernel["_DepthU%s"%tc] + LdsPad
            UnrollStride = 1

        enableLDSTr = tP["enableLDSTr"]
        matrixInstT = kernel["MatrixInstM"] if (tile01 == 0) else kernel["MatrixInstN"]
        matrixInstTO = min(kernel["MatrixInstM"], kernel["MatrixInstN"])
        matrixInstTO = matrixInstT if ("MXS" in tc) else matrixInstTO
        numTilePerInst = matrixInstT // matrixInstTO
        MIInputPerThUnroll = kernel["MIInputPerThread%s"%tc] // numTilePerInst
        numVectorsPerTile = kernel["MIWaveTile"][tile01] // vectorWidth
        numReadsPerVector = int((vectorWidth * tP["bpeDS"]) / (tileBlockWidth * bpr))
        # overloading numReadsPerUnroll for DirectToLds x2/x4 case when blockWidth of instruction < LocalReadVectorWidth
        # fp64 TLU=1 reading 0.5element/lane/read..
        # for TLU=0 case, blockWidth and LRVW should match
        numReadsPerUnroll = ceil(tP["bpeDS"] * MIInputPerThUnroll / (unrollBlockWidth * bpr))
        numVgpr  = int(ceil(blockWidth))
        tmpvgprFP32 = []

        if tc == 'A':
            lrvwTile = writer.states.lrvwTileA
            abmatrixinfo = writer.states.a
        elif tc == "MXSA":
            lrvwTile = writer.states.lrvwTileMXSA
            abmatrixinfo = writer.states.mxsa
        elif tc == 'B':
            lrvwTile = writer.states.lrvwTileB
            abmatrixinfo = writer.states.b
        elif tc == "MXSB":
            lrvwTile = writer.states.lrvwTileMXSB
            abmatrixinfo = writer.states.mxsb
        elif tc == "Metadata":
            lrvwTile = writer.states.lrvwTileMetadata
            abmatrixinfo = writer.states.m
        else:
            raise Exception(f"unsupport tc %s{tc}")

        numElementPerRead = 1 if kernel["ConvertAfterDS"] and not kernel["UseF32XEmulation"] else int(int(blockWidth * bpr) // tP['bpe'] // lrvwTile)
        perpStride = abmatrixinfo.gNLCPerpStride

        # pack register
        if writer.states.archCaps["HasEccHalf"] or not writer.states.asmCaps["HasWMMA_V1"]:
            needPack = tP["bpeDS"] < 4 and not kernel["UnrollMajorLDS%s"%tc] and not tP["isM"]
            # specify I8 for the case that input number is equal to the localread blockwidth but need to split low and high bytes to different vgprs.
            needPackMetadata = tP["isM"] and ((MIInputPerThUnroll * tP["bpeDS"] / (blockWidth * 4) > 1) or (kernel["ProblemType"][MacDataType].numBytes() == 1 and writer.states.lrvwTileMetadata > 1))
            needPack |= needPackMetadata
        else:
            needPack = blockWidth == 0.25
        needPack |= (kernel["ConvertAfterDS"] and (tP["bpe"] != tP["bpeDS"]))
        needPack |= kernel["UseF32XEmulation"]
        if isgfx950mx:
            # Keep the gfx950 workaround, but allow normal MX packing on gfx1250.
            needPack = False

        pack     = Module("pack%s_I%s"%(tc,iui))
        packPre = Module("pack%s_I%s Pre"%(tc,iui))

        isNext = self.getIsNext(writer, kernel, tc, bufferIdx)
        if isNext:
            useTransposeCode = writer.states.a.useTransposeCodeNext if tc == "A" else writer.states.b.useTransposeCodeNext
            useDirect32XEmulation = writer.states.a.useDirect32XEmulationNext if tc == "A" else writer.states.b.useDirect32XEmulationNext
        else:
            useTransposeCode = writer.states.a.useTransposeCodeThis if tc == "A" else writer.states.b.useTransposeCodeThis
            useDirect32XEmulation = writer.states.a.useDirect32XEmulationThis if tc == "A" else writer.states.b.useDirect32XEmulationThis
        indexTranpose = lrvwTile > 1 and (not useTransposeCode)

        numSplitMetadata = max(ceil((blockWidth * 4) // tP["bpeDS"]) - 1, 0) if tP["isM"] else 0

        # caculate SMFMA layout
        blocksPerTGroupSMFMA = 1
        elementsPerBlockSMFMA = 1
        blockOffsetSMFMA = 1
        if kernel["ProblemType"]["Sparse"] != 0:
            if kernel["MIInputPerThread"] * kernel["ProblemType"]["DataTypeB"].numBytes() > 16: # double K
                isSparseTrack = (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]) or (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or tP["isM"]
                # gfx950 sparse track only has one block for each thread group.
                # TODO adjust this value for other arch.
                blocksPerTGroupSMFMA = 1 if isSparseTrack else 2
                if writer.states.asmCaps["HasSWMMAC_gfx1250"] and not tP["isM"]: blocksPerTGroupSMFMA = 2
                if blocksPerTGroupSMFMA > 1:
                    threadGroups = kernel["WavefrontSize"] // matrixInstTO
                    elementsPerBlockSMFMA = MIInputPerThUnroll // blocksPerTGroupSMFMA  # need adjust if blocks > 1 and is sparse track.
                    blockStride = elementsPerBlockSMFMA * threadGroups
                    blockOffsetSMFMA = blockStride - elementsPerBlockSMFMA

        maxLDSConstOffset = writer.states.regCaps["maxLDSConstOffset"]

        subIterLoadCount = 0
        valufIdx = 0
        if enableLDSTr:
            numberMTilesPerWave = kernel["MIWaveTile"][tile01]
            if writer.states.asmCaps["HasWMMA_V3"]:
                if tP["bpeDS"] == 0.5:
                    LocalReadX = instruction.getInst(0)
                    wtRegStride = int(MIInputPerThUnroll * tP["bpeDS"] // bpr)
                    outerUnrolledIncrements = 64
                    innerUnrolledIncrements = 16
                    vwTrLoad = 16

                    for tIdx in range(numberMTilesPerWave):
                        for ti in range(0, numTilePerInst):
                            constOffset = int((tP["localReadOffset"] + matrixInstTO * ti + MIWaveGroupShape[tile01] * tIdx) * tP["bpeDS"])
                            for outerIdx in range(MIInputPerThUnroll//kernel[f"LocalReadVectorWidth{tc if('MXS' not in tc) else 'MXS'}"]):
                                for innerIdx in range(kernel[f"LocalReadVectorWidth{tc if('MXS' not in tc) else 'MXS'}"]//vwTrLoad):
                                    paddedOffset = constOffset
                                    paddedOffset += int((innerIdx * innerUnrolledIncrements + outerIdx * outerUnrolledIncrements) * UnrollStride * tP["bpeDS"])
                                    if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                                        paddedOffset += int((paddedOffset // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeDS"])
                                    ds = DSModifiers(na=1, offset=paddedOffset)
                                    destVgpr = vgpr("Valu%s_X%u_I%u+%u+%u"%(tc, bufferIdx, iui, wtRegStride * (tIdx * numTilePerInst+ti), 2 * (innerIdx + 2 * outerIdx)), blockWidth)
                                    localReadCode = imod.add(Module("LocalRead%s Valu%u"%(tc, int(valufIdx))))
                                    self._emitLdsRead(writer, kernel, tP, LocalReadX, dst=destVgpr, src=vgpr("LocalReadAddr%s"%tc), ds=ds, module=localReadCode, comment="LDS Transpose")
                elif tP["bpeDS"] == 0.75:
                    LocalReadX = instruction.getInst(0)
                    wtRegStride = (int(MIInputPerThUnroll * tP["bpeDS"] // bpr) + 15) // 16 * 16
                    outerUnrolledIncrements = 64
                    innerUnrolledIncrements = 16
                    vwTrLoad = 16
                    numVgprsPerLoad = 4 #use 3 for upcoming compiler change

                    for tIdx in range(numberMTilesPerWave):
                        constOffset = int((tP["localReadOffset"] + MIWaveGroupShape[tile01] * tIdx) * tP["bpeDS"])
                        for outerIdx in range(MIInputPerThUnroll//kernel[f"LocalReadVectorWidth{tc if('MXS' not in tc) else 'MXS'}"]):
                            for innerIdx in range(kernel[f"LocalReadVectorWidth{tc if('MXS' not in tc) else 'MXS'}"]//vwTrLoad):
                                paddedOffset = constOffset
                                paddedOffset += int((innerIdx * innerUnrolledIncrements + outerIdx * outerUnrolledIncrements) * UnrollStride * tP["bpeDS"])
                                if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                                    paddedOffset += int((paddedOffset // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeDS"])
                                ds = DSModifiers(na=1, offset=paddedOffset)
                                vgprOffset = numVgprsPerLoad * (innerIdx + 2 * outerIdx)
                                destVgpr = vgpr("Valu%s_X%u_I%u+%u+%u"%(tc, bufferIdx, iui, wtRegStride*tIdx, vgprOffset), blockWidth)
                                localReadCode = imod.add(Module("LocalRead%s Valu%u"%(tc, int(valufIdx))))
                                self._emitLdsRead(writer, kernel, tP, LocalReadX, dst=destVgpr, src=vgpr("LocalReadAddr%s"%tc), ds=ds, module=localReadCode, comment="LDS Transpose")

                elif tP["bpeDS"] == 1:
                    LocalReadX = instruction.getInst(0)
                    wtRegStride = MIInputPerThUnroll * tP["bpeDS"] // bpr
                    numUnrolledIncrements = 32
                    vwTrLoad = 8
                    numberLRVWPerMIInput = MIInputPerThUnroll // kernel[f"LocalReadVectorWidth{tc if('MXS' not in tc) else 'MXS'}"]
                    for tIdx in range(numberMTilesPerWave):
                        offset = int((tP["localReadOffset"] + MIWaveGroupShape[tile01] * tIdx) * tP["bpeDS"])
                        if tP["isM"]:
                            numLoadTrPerMetadata = max(MIInputPerThUnroll // vwTrLoad, 1)
                            for v in range(numLoadTrPerMetadata):
                                incrementBytes = int((v * vwTrLoad ) * tP["bpeDS"] * UnrollStride)
                                paddedOffset = offset + incrementBytes
                                if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                                    paddedOffset += int((paddedOffset // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeDS"])
                                paddedOffset += tP["localReadSwapByteOffset"]
                                addrIdx = paddedOffset // 65536
                                srcAddr=vgpr("LocalReadAddr%s+%u"%(tc, addrIdx))
                                paddedOffset -= addrIdx * 65536
                                ds = DSModifiers(na=1, offset=paddedOffset)
                                wtRegStride = 2 if wtRegStride < 2 else wtRegStride # wtRegStride at least need to be 2.
                                destVgpr = vgpr("Valu%s_X%u_I%u+%u+%u"%(tc, bufferIdx, iui, wtRegStride*tIdx, 2*v), blockWidth)
                                localReadCode: Module = imod.add(Module("LocalRead%s Valu%u"%(tc, int(valufIdx))))
                                self._emitLdsRead(writer, kernel, tP, LocalReadX, dst=destVgpr, src=srcAddr, ds=ds, module=localReadCode, comment="LDS Transpose")
                        else:
                            for i in range(MIInputPerThUnroll//kernel[f"LocalReadVectorWidth{tc if('MXS' not in tc) else 'MXS'}"]):
                                for v in range(kernel[f"LocalReadVectorWidth{tc if('MXS' not in tc) else 'MXS'}"]//vwTrLoad):
                                    incrementBytes = int((v * vwTrLoad + i * numUnrolledIncrements) * tP["bpeDS"] * UnrollStride)
                                    sparseDenseOffset = 0
                                    if numberLRVWPerMIInput == 4 and kernel["ProblemType"]["Sparse"] != 0:
                                        sparseDenseOffset = int((numUnrolledIncrements) * tP["bpeDS"] * UnrollStride) // 2
                                        if i%2 == 1:
                                            # For sparse, interleaved so that the 2nd part will have an offset.
                                            paddedOffset = offset + incrementBytes - sparseDenseOffset
                                        else:
                                            paddedOffset = offset + incrementBytes
                                    else:
                                        paddedOffset = offset + incrementBytes

                                    if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                                        paddedOffset += int((paddedOffset // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeDS"])
                                    ds = DSModifiers(na=1, offset=paddedOffset)
                                    destVgpr = vgpr("Valu%s_X%u_I%u+%u+%u"%(tc, bufferIdx, iui, wtRegStride*tIdx, 2*v+4*i), blockWidth)
                                    localReadCode: Module = imod.add(Module("LocalRead%s Valu%u"%(tc, int(valufIdx))))
                                    self._emitLdsRead(writer, kernel, tP, LocalReadX, dst=destVgpr, src=vgpr("LocalReadAddr%s"%tc), ds=ds, module=localReadCode, comment="LDS Transpose")
                elif tP["bpeDS"] == 2:
                    numberLRVWPerMIInput = MIInputPerThUnroll // kernel[f"LocalReadVectorWidth{tc if('MXS' not in tc) else 'MXS'}"]
                    for tIdx in range(0, numberMTilesPerWave):
                        offset_val = int((tP["localReadOffset"]+MIWaveGroupShape[tile01]*tIdx) * tP["bpeDS"])
                        unpaddedOffset = offset_val
                        if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                            offset_val += int((offset_val // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeDS"])
                        ds = DSModifiers(na=1, offset=offset_val)
                        LocalReadX = instruction.getInst(0)
                        wtRegStride = int(MIInputPerThUnroll * tP["bpeDS"] // bpr)
                        destVgpr = vgpr("Valu%s_X%u_I%u+%u+0"%(tc,bufferIdx,iui, wtRegStride*tIdx), blockWidth)
                        valuiIdx = int(valufIdx)
                        localReadCode = imod.add(Module("LocalRead%s Valu%u"%(tc,valuiIdx)))
                        self._emitLdsRead(writer, kernel, tP, LocalReadX, dst=destVgpr, src=vgpr("LocalReadAddr%s"%tc), ds=ds, module=localReadCode, comment="LDS Transpose")
                        destVgpr = vgpr("Valu%s_X%u_I%u+%u+%u"%(tc,bufferIdx,iui, wtRegStride*tIdx, blockWidth), blockWidth)
                        incrementBytes = int(numberLRVWPerMIInput*UnrollStride*kernel[f"LocalReadVectorWidth{tc if('MXS' not in tc) else 'MXS'}"]*tP["bpeDS"])

                        sparseDenseOffset = 0
                        if numberLRVWPerMIInput == 4:
                            # generally numberLRVWPerMIInput should be 2, for the dense matrix of sparse cases, it will be 4
                            incrementBytes = incrementBytes // 2
                            sparseDenseOffset = incrementBytes // 2  # for sparse dense matrix, we read the 2nd half of the data seperately.
                        offset_val = unpaddedOffset + incrementBytes - sparseDenseOffset
                        if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                            offset_val += int((offset_val // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeDS"])
                        ds = DSModifiers(na=1, offset=offset_val)
                        self._emitLdsRead(writer, kernel, tP, LocalReadX, dst=destVgpr, src=vgpr("LocalReadAddr%s"%tc), ds=ds, module=localReadCode, comment="LDS Transpose")
                        if numberLRVWPerMIInput == 4:
                            # for the dense case when sparse.
                            destVgpr = vgpr("Valu%s_X%u_I%u+%u+%u"%(tc,bufferIdx,iui, wtRegStride*tIdx, blockWidth * 2), blockWidth)
                            offset_val = unpaddedOffset + incrementBytes * 2
                            if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                                offset_val += int((offset_val // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeDS"])
                            ds = DSModifiers(na=1, offset=offset_val)
                            self._emitLdsRead(writer, kernel, tP, LocalReadX, dst=destVgpr, src=vgpr("LocalReadAddr%s"%tc), ds=ds, module=localReadCode, comment="LDS Transpose")
                            destVgpr = vgpr("Valu%s_X%u_I%u+%u+%u"%(tc,bufferIdx,iui, wtRegStride*tIdx, blockWidth * 3), blockWidth)
                            offset_val = unpaddedOffset + incrementBytes * 2 + sparseDenseOffset
                            if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                                offset_val += int((offset_val // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeDS"])
                            ds = DSModifiers(na=1, offset=offset_val)
                            self._emitLdsRead(writer, kernel, tP, LocalReadX, dst=destVgpr, src=vgpr("LocalReadAddr%s"%tc), ds=ds, module=localReadCode, comment="LDS Transpose")
                else:
                    assert False, f"Unhandled bpeDS: {tP['bpeDS']}"
            else:
                numOffsetsPerLoad = 2 * blocksPerTGroupSMFMA
                numberMTilesPerWave = kernel["MIWaveTile"][tile01]
                highBits = 0
                totalLoads = numberMTilesPerWave * numOffsetsPerLoad
                for tIdx in range(0, numberMTilesPerWave):
                    valuiIdx = int(valufIdx)
                    LocalReadX = instruction.getInst(highBits)

                    offset_val = (tP["localReadOffset"]+MIWaveGroupShape[tile01]*tIdx) * tP["bpeDS"] + tP["localReadSwapByteOffset"]

                    def applyPad(offset_val):
                        if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                            offset_val = offset_val + (offset_val // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeDS"]
                        return offset_val

                    for oIdx in range(0,numOffsetsPerLoad):
                        if blocksPerTGroupSMFMA > 1 and oIdx % blocksPerTGroupSMFMA == 0:
                            offset_val += (kernel["MacroTile%s"%tc] * blockOffsetSMFMA) * tP["bpeDS"] * (oIdx // blocksPerTGroupSMFMA)

                        offset, srcAddr = self.cal_offset_srcAddr(maxLDSConstOffset, tc, offset_val)
                        offset = int(applyPad(offset))
                        ds = DSModifiers(na=1, offset=offset)
                        destVgpr = vgpr("Valu%s_X%u_I%u+%u+%u"%(tc,bufferIdx,iui, 4*tIdx*blocksPerTGroupSMFMA, oIdx * 2), 2)
                        localReadCode = Module("LocalRead%s Valu%u"%(tc,valuiIdx))
                        self._emitLdsRead(writer, kernel, tP, LocalReadX, dst=destVgpr, src=srcAddr, ds=ds, module=localReadCode, comment="LDS Transpose")
                        if perpStride == 1:
                            inputPerThread = kernel[f"LocalReadVectorWidth{tc if('MXS' not in tc) else 'MXS'}"] if not writer.states.inTailLoop else MIInputPerThUnroll
                            offset_val += (UnrollStride*inputPerThread) // (blocksPerTGroupSMFMA if writer.states.inTailLoop else 1)
                        else:
                            permBlock = kernel["MatrixInstK"]
                            perpStrideInv = permBlock // perpStride
                            inv4K = perpStrideInv * (4 % perpStride) + 4 // perpStride
                            offset_val += inv4K * kernel["MacroTile%s"%tc] * tP["bpeDS"]
                        if ((subTileIdx == 0 and subIterLoadCount < totalLoads // numSubTiles) \
                            or (subTileIdx == 1 and subIterLoadCount >= totalLoads // numSubTiles) \
                            or numSubTiles == 1) or writer.states.inTailLoop:
                            imod.add(localReadCode)
                        subIterLoadCount += 1
        # Without enableLDSTr
        else:
            totalLoads = numVectorsPerTile * numReadsPerVector * numReadsPerUnroll
            swapBlockSizeSub = (totalLoads * blockWidth)
            if not writer.states.inTailLoop:
                # divided by numSubTiles for non TailLoop case
                # TailLoop case, code is not devided by numSubTiles
                swapBlockSizeSub //= numSubTiles
            # save the value for index transpose
            abmatrixinfo.swapBlockSizeSub = swapBlockSizeSub
            for vIdx in range(0, numVectorsPerTile):
                for eIdx in range(0, numReadsPerVector):
                    valuiIdx = int(valufIdx)
                    baseValuiIdx = valuiIdx
                    localReadCode = imod.add(Module("LocalRead%s Valu%u"%(tc,valuiIdx)))
                    if needPack or numSplitMetadata:
                        packCode = pack.add(Module("packCode"))
                        packCodePre = packPre.add(Module("packCodePre"))

                    tmpvgpr = []
                    is_wmma_v3 = writer.states.asmCaps.get("HasWMMA_V3", False)
                    multiGroupXF32 = kernel["UseF32XEmulation"] and is_wmma_v3 and numVgpr * numReadsPerUnroll > 8
                    outerBaseValuiIdx = baseValuiIdx
                    for tiIdx in range(0, numTilePerInst):
                        for rIdx in range(0, numReadsPerUnroll):
                            valuiIdx = int(valufIdx)
                            baseValuiIdx = valuiIdx - (valuiIdx%8) # use multiple of 8
                            baseLRVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx), numVgpr)
                            destVgpr = baseLRVgpr
                            highBitsForHalf = (blockWidth == 0.5) and ((rIdx % 2) == 1) # rIdx = 1
                            isHigh16Bits = (blockWidth == 0.25) and ( ((rIdx % 4) //2) == 1) # 2,3

                            packCodeT = Module() # Allocate temporary module for pack code
                            packCodePreT = Module() # Allocate temporary module for pack code Pre
                            localReadCodeT = Module()

                            if needPack or numSplitMetadata:
                                if kernel["UseF32XEmulation"]:
                                    # Pack data 0-7 with layout:
                                    # Val+0: bf16 high (0,1)
                                    # Val+1: bf16 high (2,3)
                                    # Val+2: bf16 high (4,5)
                                    # Val+3: bf16 high (6,7)
                                    # Val+4: bf16 low  (0,1)
                                    # Val+5: bf16 low  (2,3)
                                    # Val+6: bf16 low  (4,5)
                                    # Val+7: bf16 low  (6,7)

                                    if (valuiIdx % swapBlockSizeSub) == 0:
                                        allPack4HiDone = False
                                        allPack4LoDone = False
                                        if useTransposeCode:
                                            # generate Tranpose code (with v_swap) for wider local read + useTransposeCode
                                            self.transposeLRVregs(kernel, packCodePreT, tc, bufferIdx, iui, writer, lrvwTile, swapBlockSizeSub, subTileIdx)
                                        if writer.states.doFullPackCodePrefetch and useDirect32XEmulation:
                                            # index transpose + doFullPackCodePrefetch case, finish all first conversion (high value) at once
                                            allPack4HiDone = True
                                            for idx in range(0, swapBlockSizeSub, 4):
                                                newBaseValuiIdx = baseValuiIdx + idx - (idx % 8)
                                                newIdx = (idx % 8)
                                                # add comment at last one only
                                                commentForSchedule1 = "__TF32_1_" + tc + "_%d"%(newBaseValuiIdx//8)
                                                noComment = idx < swapBlockSizeSub - 4
                                                commentForPack = "" if noComment else commentForSchedule1
                                                self.pack4HiBits(kernel, writer, tc, newIdx, bufferIdx, newBaseValuiIdx, iui, packCodeT, lrvwTile, commentForPack, useDirect32XEmulation)
                                            # do all low cvt
                                            if useDirect32XEmulation:
                                                allPack4LoDone = True
                                                # allocate tmp vgpr first
                                                self.initTmpVregForPack(kernel, writer, tc, 0, bufferIdx, baseValuiIdx, iui, packCodeT, lrvwTile, tmpvgprFP32, useDirect32XEmulation)
                                                # low pack 1st stage
                                                for idx in range(0, swapBlockSizeSub, 4):
                                                    # add comment at last one only
                                                    commentForSchedule1 = "__TF32_1_" + tc + "_%d"%(newBaseValuiIdx//8)
                                                    noComment = idx < swapBlockSizeSub - 4
                                                    commentForPack = "" if noComment else commentForSchedule1
                                                    self.pack4LowBitsStep1(kernel, writer, tc, baseValuiIdx + idx, bufferIdx, iui, packCodeT, lrvwTile, tmpvgprFP32, commentForPack, useDirect32XEmulation)
                                                # low pack final
                                                for idx in range(0, swapBlockSizeSub, 4):
                                                    # on last iteration, store lower bits in last 4 registers
                                                    if idx % 8 == 4:
                                                        self.pack4LowBitsFinal(kernel, writer, tc, baseValuiIdx + idx, bufferIdx, iui, packCodeT, lrvwTile, tmpvgprFP32, useDirect32XEmulation, noComment=idx < swapBlockSizeSub - 4)
                                                # release tmp regs
                                                self.releaseTmpVregForPack(kernel, writer, tc, baseValuiIdx, tmpvgprFP32, useDirect32XEmulation)

                                    # For every 8 read vgprs of fp32, pack high bits of bf16 into first 4 vgprs
                                    if valuiIdx % 8 == 0 and not allPack4LoDone:
                                        commentForSchedule1 = "__TF32_1_" + tc + "_%d"%(baseValuiIdx//8)
                                        # allocate tmp vgpr first
                                        self.initTmpVregForPack(kernel, writer, tc, 0, bufferIdx, baseValuiIdx, iui, packCodeT, lrvwTile, tmpvgprFP32, useDirect32XEmulation)
                                        if not allPack4HiDone:
                                            commentForPack = ""
                                            self.pack4HiBits(kernel, writer, tc, 0, bufferIdx, baseValuiIdx, iui, packCodeT, lrvwTile, commentForPack, useDirect32XEmulation)
                                            commentForPack = commentForSchedule1
                                            if kernel["UseMFMAF32XEmulation"] or (not useDirect32XEmulation):
                                                commentForPack = ""
                                            self.pack4HiBits(kernel, writer, tc, 4, bufferIdx, baseValuiIdx, iui, packCodeT, lrvwTile, commentForPack, useDirect32XEmulation)

                                    do8PackAtOnce = indexTranpose and not allPack4HiDone
                                    if (valuiIdx % 8) == 4 and do8PackAtOnce:
                                        # index transpose  case
                                        # we need to keep both original values and transpose values
                                        # do "Compute low bits" for 0-3 and 4-7 + final pack here
                                        # do all at (valuiIdx % 8) == 4
                                        tmp = writer.vgprPool.checkOut(1, "x32f tmp")
                                        valuiIdx0 = valuiIdx - 4 # for 0-3
                                        valuiIdx1 = valuiIdx     # for 4-7
                                        # src (original value)
                                        v0, v1, v2, v3 = self.get4VgprForEmu(writer, kernel, tc, bufferIdx, valuiIdx0, iui, lrvwTile)
                                        v4, v5, v6, v7 = self.get4VgprForEmu(writer, kernel, tc, bufferIdx, valuiIdx1, iui, lrvwTile)
                                        # dst (high)
                                        v0t, v1t, v2t, v3t = self.get4VgprForEmu(writer, kernel, tc, bufferIdx, valuiIdx0, iui, lrvwTile, dst=True)
                                        v4t, v5t, v6t, v7t = self.get4VgprForEmu(writer, kernel, tc, bufferIdx, valuiIdx1, iui, lrvwTile, dst=True)
                                        # low bits 0
                                        packCodeT.add(PVCvtBF16toFP32(dst=vgpr(tmp), src=v0t, comment="begin"+str(valuiIdx0)))
                                        packCodeT.add(VSubF32(dst=v4t, src0=v0, src1=vgpr(tmp)))
                                        # low bits 1
                                        packCodeT.add(VCvtBF16toFP32(dst=vgpr(tmp), src=v0t, vgprMask=None, vi=1))
                                        packCodeT.add(VSubF32(dst=vgpr(tmp), src0=v1, src1=vgpr(tmp)))
                                        # final 4
                                        packCodeT.add(VCvtPkF32toBF16(dst=v4t, src0=v4t, src1=vgpr(tmp), comment="pack final begin"))
                                        # low bits 2
                                        packCodeT.add(PVCvtBF16toFP32(dst=vgpr(tmp), src=v1t, comment="begin"+str(valuiIdx0+2)))
                                        packCodeT.add(VSubF32(dst=v5t, src0=v2, src1=vgpr(tmp)))
                                        # low bits 3
                                        packCodeT.add(VCvtBF16toFP32(dst=vgpr(tmp), src=v1t, vgprMask=None, vi=1))
                                        packCodeT.add(VSubF32(dst=vgpr(tmp), src0=v3, src1=vgpr(tmp)))
                                        # final 5
                                        packCodeT.add(VCvtPkF32toBF16(dst=v5t, src0=v5t, src1=vgpr(tmp)))
                                        # low bits 4
                                        packCodeT.add(PVCvtBF16toFP32(dst=vgpr(tmp), src=v2t, comment="begin"+str(valuiIdx0+4)))
                                        packCodeT.add(VSubF32(dst=v6t, src0=v4, src1=vgpr(tmp)))
                                        # low bits 5
                                        packCodeT.add(VCvtBF16toFP32(dst=vgpr(tmp), src=v2t, vgprMask=None, vi=1))
                                        packCodeT.add(VSubF32(dst=vgpr(tmp), src0=v5, src1=vgpr(tmp)))
                                        # final 6
                                        packCodeT.add(VCvtPkF32toBF16(dst=v6t, src0=v6t, src1=vgpr(tmp)))
                                        # low bits 6
                                        packCodeT.add(PVCvtBF16toFP32(dst=vgpr(tmp), src=v3t, comment="begin"+str(valuiIdx0+6)))
                                        packCodeT.add(VSubF32(dst=v7t, src0=v6, src1=vgpr(tmp)))
                                        # low bits 7
                                        packCodeT.add(VCvtBF16toFP32(dst=vgpr(tmp), src=v3t, vgprMask=None, vi=1))
                                        packCodeT.add(VSubF32(dst=vgpr(tmp), src0=v7, src1=vgpr(tmp)))
                                        # final 7
                                        commentStr = "__TF32_2_" + tc + "_%d pack final end"%(baseValuiIdx//8)
                                        packCodeT.add(VCvtPkF32toBF16(dst=v7t, src0=v7t, src1=vgpr(tmp), comment=commentStr))
                                        writer.vgprPool.checkIn(tmp)
                                    elif valuiIdx % 4 == 0 and (not do8PackAtOnce) and (not allPack4LoDone):
                                        noComment = valuiIdx % 8 == 0
                                        commentForPack = "" if noComment else commentForSchedule1
                                        self.pack4LowBitsStep1(kernel, writer, tc, valuiIdx, bufferIdx, iui, packCodeT, lrvwTile, tmpvgprFP32, commentForPack, useDirect32XEmulation)
                                        # on last iteration, store lower bits in last 4 registers
                                        if valuiIdx % 8 == 4:
                                            self.pack4LowBitsFinal(kernel, writer, tc, valuiIdx, bufferIdx, iui, packCodeT, lrvwTile, tmpvgprFP32, useDirect32XEmulation)
                                    if valuiIdx % 8 == 4 and (not allPack4LoDone):
                                        self.releaseTmpVregForPack(kernel, writer, tc, baseValuiIdx, tmpvgprFP32, useDirect32XEmulation)
                                    # For WMMA V3 multigroup XF32: rearrange packed BF16 data so all
                                    # hi values are contiguous (X0+0..7) followed by lo values (X0+8..15).
                                    # Must happen here in pack code (not MAC code) to avoid the scheduler
                                    # interleaving the swap with residual/TF32_2 packing operations.
                                    if multiGroupXF32 and rIdx == numReadsPerUnroll - 1:
                                        halfGroup = 4
                                        for i in range(halfGroup):
                                            swapIdx1 = outerBaseValuiIdx + halfGroup + i
                                            swapIdx2 = outerBaseValuiIdx + halfGroup * 2 + i
                                            swapVgpr1 = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, swapIdx1))
                                            swapVgpr2 = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, swapIdx2))
                                            packCodeT.add(VSwapB32(dst=swapVgpr1, src=swapVgpr2,
                                                comment="XF32 pack rearrange %s: swap [%d] <-> [%d]"%(tc, swapIdx1, swapIdx2)))

                                if kernel["ConvertAfterDS"] and (tP["bpe"] != tP["bpeDS"]):
                                    if tP["bpe"] == 2 and tP["bpeDS"] == 4:
                                        assert 0 # Doesn't support ConvertAfterDS
                                    else:
                                        highBitsForHalf = False
                                        isHigh16Bits = False
                                        if kernel["UnrollMajorLDS%s"%tc]:
                                            cvtTimes = int(blockWidth * writer.states.bpr // tP["bpeDS"]) // MIInputPerThUnroll
                                            for i in range(0, cvtTimes):
                                                offset = cvtTimes - i - 1
                                                if writer.states.asmCaps["Hascvtf16_fp8_sf32"]:
                                                    packCodeT.add(VCvtScalePkFP8toF16(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+1+offset*2)),\
                                                                                src=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+offset)), scale=0x3f800000,\
                                                                                vop3=VOP3PModifiers(op_sel=[1,0,0,0]), comment="convert fp8 to f16"))
                                                    packCodeT.add(VCvtScalePkFP8toF16(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+0+offset*2)),\
                                                                                src=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+offset)), scale=0x3f800000,\
                                                                                vop3=VOP3PModifiers(op_sel=[0,0,0,0]), comment="convert fp8 to f16"))
                                                elif writer.states.asmCaps["HasCvtFP8toF16"]:
                                                    packCode.add(VCvtScalePkFP8toF16(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+1+offset*2)),\
                                                                                src=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+offset)),\
                                                                                vop3=VOP3PModifiers(op_sel=[1,0]),comment="convert F8 to F16"))
                                                    packCode.add(VCvtScalePkFP8toF16(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+0+offset*2)),\
                                                                                src=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+offset)),\
                                                                                vop3=VOP3PModifiers(op_sel=[0,0]), comment="convert fp8 to f16"))
                                                else:
                                                    packCodeT.add(ECvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+offset)),\
                                                                                sel=HighBitSel.HIGH, comment="convert to F32"))
                                                    packCodeT.add(ECvtF32toF16(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+1+offset*2)), src=vgpr("CvtTemp+0"),\
                                                                            sel=HighBitSel.LOW, comment="Convert to FP16"))
                                                    packCodeT.add(ECvtF32toF16(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+1+offset*2)), src=vgpr("CvtTemp+1"),\
                                                                            sel=HighBitSel.HIGH, comment="Convert to FP16"))
                                                    packCodeT.add(ECvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+offset)),\
                                                                                sel=HighBitSel.LOW, comment="convert to F32"))
                                                    packCodeT.add(ECvtF32toF16(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+0+offset*2)), src=vgpr("CvtTemp+0"),\
                                                                            sel=HighBitSel.LOW, comment="Convert to FP16"))
                                                    packCodeT.add(ECvtF32toF16(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+0+offset*2)), src=vgpr("CvtTemp+1"),\
                                                                            sel=HighBitSel.HIGH, comment="Convert to FP16"))
                                        elif (writer.states.lrvwTileA == 1 and tc == 'A') or (writer.states.lrvwTileB == 1 and tc == 'B'):
                                            destVgpr   = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, int(valufIdx*2)), numVgpr)
                                            CvtDstVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, int(valufIdx*2)), numVgpr)
                                            if needPack and (rIdx%4) != 0:
                                                if (rIdx % 4) != 0:
                                                    destVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%4, valuiIdx), numVgpr)
                                                else:
                                                    destVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx), numVgpr)
                                            if writer.states.asmCaps["Hascvtf16_fp8_sf32"]:
                                                sel = [0,0,0,0] if (rIdx % 2 == 0)  else [0,0,1,0]
                                                packCodeT.add(VCvtScaleFP8toF16(dst=CvtDstVgpr, src=destVgpr, scale=0x3f800000, vop3=VOP3PModifiers(op_sel=sel), comment="convert fp8 to f16"))
                                            elif writer.states.asmCaps["HasCvtFP8toF16"]:
                                                sel = (rIdx % 2 == 0)
                                                packCode.add(VCvtFP8toF16(dst=CvtDstVgpr, src=destVgpr, vop3=VOP3PModifiers(byte_sel=[sel]),comment="convert F8 to F16"))
                                            else:
                                                packCodeT.add(VCvtFP8toF32(dst=destVgpr, src=destVgpr, sdwa=SDWAModifiers(src0_sel=SelectBit.BYTE_0)))
                                                packCodeT.add(ECvtF32toF16(dst=CvtDstVgpr, src=destVgpr, sel=HighBitSel.LOW if rIdx % 2 == 0 else HighBitSel.HIGH, comment="Convert to FP16"))
                                        elif (writer.states.lrvwTileA == 2 and tc == 'A') or (writer.states.lrvwTileB == 2 and tc == 'B'):
                                            if needPack or numSplitMetadata:
                                                destVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(MIInputPerThUnroll), vIdx*numVgpr), numVgpr)
                                                for i in range(0, numVgpr):
                                                    cvtDstVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(MIInputPerThUnroll), vIdx*numVgpr), numVgpr)
                                                    if writer.states.asmCaps["Hascvtf16_fp8_sf32"]:
                                                        packCodeT.add(VCvtScalePkFP8toF16(dst=destVgpr, src=destVgpr,scale=0x3f800000,vop3=VOP3PModifiers(op_sel=[0,0,0,0]),comment="convert F8 to F16"))
                                                    elif writer.states.asmCaps["HasCvtFP8toF16"]:
                                                        packCode.add(VCvtScalePkFP8toF16(dst=destVgpr, src=destVgpr, vop3=VOP3PModifiers(op_sel=[0]),comment="convert F8 to F16"))
                                                    else:
                                                        packCodeT.add(ECvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=destVgpr, sel=HighBitSel.LOW, comment="convert to F32"))
                                                        packCodeT.add(ECvtF32toF16(dst=destVgpr, src=vgpr("CvtTemp+0"), sel=HighBitSel.LOW, comment="Convert to FP16"))
                                                        packCodeT.add(ECvtF32toF16(dst=destVgpr, src=vgpr("CvtTemp+1"), sel=HighBitSel.HIGH, comment="Convert to FP16"))

                                                if rIdx == numReadsPerUnroll-1:
                                                    for i in range(0, numVgpr):
                                                        vgprIdx = int((vIdx * numVgpr + i) * tP["bpe"] * MIInputPerThUnroll // writer.states.bpr * min(writer.states.bpr // tP["bpe"], vectorWidth))
                                                        vgprOffset = 0
                                                        for vectorIdx in range(0, 2):
                                                            for elementIdx in range(0, int(tP["bpe"]*MIInputPerThUnroll//writer.states.bpr)):
                                                                packCodeT.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+vgprOffset)), \
                                                                                    src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*2+1, i+vIdx*numVgpr)), \
                                                                                    src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*2, i+vIdx*numVgpr)), \
                                                                                    src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                                    comment="select K=%u%u for vector=%u"%(elementIdx*2,  elementIdx*2+1, vectorIdx)))
                                                                vgprOffset += 1
                                        elif (writer.states.lrvwTileA == 4 and tc == 'A') or (writer.states.lrvwTileB == 4 and tc == 'B'):
                                            if needPack or numSplitMetadata:
                                                destVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(MIInputPerThUnroll), 2 * vIdx * numVgpr), numVgpr)
                                                cvtDestVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(MIInputPerThUnroll), 2 * vIdx * numVgpr + 1), numVgpr)
                                                packCodeT.add(VLShiftRightB32(dst=cvtDestVgpr, shiftHex=16, src=destVgpr, comment="shift 2 element to vgpr+1"))
                                                if writer.states.asmCaps["Hascvtf16_fp8_sf32"]:
                                                    packCodeT.add(VCvtScalePkFP8toF16(dst=destVgpr, src=destVgpr,scale=0x3f800000, comment="convert F8 to F16"))
                                                    packCodeT.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr, src=cvtDestVgpr,scale=0x3f800000, comment="convert F8 to F16"))
                                                elif writer.states.asmCaps["HasCvtFP8toF16"]:
                                                    packCode.add(VCvtScalePkFP8toF16(dst=destVgpr, src=destVgpr, vop3=VOP3PModifiers(op_sel=[0]),comment="convert F8 to F16"))
                                                    packCode.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr, src=cvtDestVgpr, vop3=VOP3PModifiers(op_sel=[0]),comment="convert F8 to F16"))
                                                else:
                                                    packCodeT.add(ECvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=destVgpr, sel=HighBitSel.LOW, comment="convert to F32"))
                                                    packCodeT.add(ECvtF32toF16(dst=destVgpr, src=vgpr("CvtTemp+0"), sel=HighBitSel.LOW, comment="Convert to FP16"))
                                                    packCodeT.add(ECvtF32toF16(dst=destVgpr, src=vgpr("CvtTemp+1"), sel=HighBitSel.HIGH, comment="Convert to FP16"))
                                                    packCodeT.add(ECvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=cvtDestVgpr, sel=HighBitSel.LOW, comment="convert to F32"))
                                                    packCodeT.add(ECvtF32toF16(dst=cvtDestVgpr, src=vgpr("CvtTemp+0"), sel=HighBitSel.LOW, comment="Convert to FP16"))
                                                    packCodeT.add(ECvtF32toF16(dst=cvtDestVgpr, src=vgpr("CvtTemp+1"), sel=HighBitSel.HIGH, comment="Convert to FP16"))

                                                if rIdx == numReadsPerUnroll-1:
                                                    for i in range(0, numVgpr*2):
                                                        vgprIdx = int((2 * vIdx * numVgpr + i) * tP["bpe"] * MIInputPerThUnroll // writer.states.bpr * min(writer.states.bpr // tP["bpe"], vectorWidth))
                                                        vgprOffset = 0
                                                        for vectorIdx in range(0, 2):
                                                            for elementIdx in range(0, int(tP["bpe"]*MIInputPerThUnroll//writer.states.bpr)):
                                                                packCodeT.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+vgprOffset)), \
                                                                                    src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*2+1, i+2*vIdx*numVgpr)), \
                                                                                    src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*2, i+2*vIdx*numVgpr)), \
                                                                                    src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                                    comment="select K=%u%u for vector=%u"%(elementIdx*2,  elementIdx*2+1, vectorIdx)))
                                                                vgprOffset += 1
                                        elif (writer.states.lrvwTileA == 8 and tc == 'A') or (writer.states.lrvwTileB == 8 and tc == 'B'):
                                            if needPack or numSplitMetadata:
                                                destVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(MIInputPerThUnroll), 2*vIdx*numVgpr), numVgpr)
                                                cvtDestVgpr0 = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(MIInputPerThUnroll), 2*vIdx*numVgpr+0), 1)
                                                cvtDestVgpr1 = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(MIInputPerThUnroll), 2*vIdx*numVgpr+1), 1)
                                                cvtDestVgpr2 = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(MIInputPerThUnroll), 2*vIdx*numVgpr+2), 1)
                                                cvtDestVgpr3 = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(MIInputPerThUnroll), 2*vIdx*numVgpr+3), 1)
                                                packCodeT.add(VLShiftRightB32(dst=cvtDestVgpr3, shiftHex=16, src=cvtDestVgpr1, comment="shift 2 element to vgpr+3"))

                                                packCodeT.add(VMovB32(dst=cvtDestVgpr2, src=cvtDestVgpr1))
                                                packCodeT.add(VLShiftRightB32(dst=cvtDestVgpr1, shiftHex=16, src=cvtDestVgpr0, comment="shift 2 element to vgpr+1"))
                                                if writer.states.asmCaps["Hascvtf16_fp8_sf32"]:
                                                    packCodeT.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr0, src=cvtDestVgpr0,scale=0x3f800000, comment="convert F8 to F16"))
                                                    packCodeT.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr1, src=cvtDestVgpr1,scale=0x3f800000, comment="convert F8 to F16"))
                                                    packCodeT.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr2, src=cvtDestVgpr2,scale=0x3f800000, comment="convert F8 to F16"))
                                                    packCodeT.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr3, src=cvtDestVgpr3,scale=0x3f800000, comment="convert F8 to F16"))
                                                elif writer.states.asmCaps["HasCvtFP8toF16"]:
                                                    packCode.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr0, src=cvtDestVgpr0, vop3=VOP3PModifiers(op_sel=[0]),comment="convert F8 to F16"))
                                                    packCode.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr1, src=cvtDestVgpr1, vop3=VOP3PModifiers(op_sel=[0]),comment="convert F8 to F16"))
                                                    packCode.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr2, src=cvtDestVgpr2, vop3=VOP3PModifiers(op_sel=[0]),comment="convert F8 to F16"))
                                                    packCode.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr3, src=cvtDestVgpr3, vop3=VOP3PModifiers(op_sel=[0]),comment="convert F8 to F16"))
                                                else:
                                                    packCodeT.add(ECvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=cvtDestVgpr0, sel=HighBitSel.LOW, comment="convert to F32"))
                                                    packCodeT.add(ECvtF32toF16(dst=cvtDestVgpr0, src=vgpr("CvtTemp+0"), sel=HighBitSel.LOW, comment="Convert to FP16"))
                                                    packCodeT.add(ECvtF32toF16(dst=cvtDestVgpr0, src=vgpr("CvtTemp+1"), sel=HighBitSel.HIGH, comment="Convert to FP16"))
                                                    packCodeT.add(ECvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=cvtDestVgpr1, sel=HighBitSel.LOW, comment="convert to F32"))
                                                    packCodeT.add(ECvtF32toF16(dst=cvtDestVgpr1, src=vgpr("CvtTemp+0"), sel=HighBitSel.LOW, comment="Convert to FP16"))
                                                    packCodeT.add(ECvtF32toF16(dst=cvtDestVgpr1, src=vgpr("CvtTemp+1"), sel=HighBitSel.HIGH, comment="Convert to FP16"))
                                                    packCodeT.add(ECvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=cvtDestVgpr2, sel=HighBitSel.LOW, comment="convert to F32"))
                                                    packCodeT.add(ECvtF32toF16(dst=cvtDestVgpr2, src=vgpr("CvtTemp+0"), sel=HighBitSel.LOW, comment="Convert to FP16"))
                                                    packCodeT.add(ECvtF32toF16(dst=cvtDestVgpr2, src=vgpr("CvtTemp+1"), sel=HighBitSel.HIGH, comment="Convert to FP16"))
                                                    packCodeT.add(ECvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=cvtDestVgpr3, sel=HighBitSel.LOW, comment="convert to F32"))
                                                    packCodeT.add(ECvtF32toF16(dst=cvtDestVgpr3, src=vgpr("CvtTemp+0"), sel=HighBitSel.LOW, comment="Convert to FP16"))
                                                    packCodeT.add(ECvtF32toF16(dst=cvtDestVgpr3, src=vgpr("CvtTemp+1"), sel=HighBitSel.HIGH, comment="Convert to FP16"))

                                                if rIdx == numReadsPerUnroll-1:
                                                    for i in range(0, numVgpr*2):
                                                        vgprIdx = int((2 * vIdx * numVgpr + i) * tP["bpe"] * MIInputPerThUnroll // writer.states.bpr * min(writer.states.bpr // tP["bpe"], vectorWidth))
                                                        vgprOffset = 0
                                                        for vectorIdx in range(0, 2):
                                                            for elementIdx in range(0, int(tP["bpe"]*MIInputPerThUnroll//writer.states.bpr)):
                                                                packCodeT.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+vgprOffset)), \
                                                                                    src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*2+1, i+2*vIdx*numVgpr)), \
                                                                                    src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*2, i+2*vIdx*numVgpr)), \
                                                                                    src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                                    comment="select K=%u%u for vector=%u"%(elementIdx*2,  elementIdx*2+1, vectorIdx)))
                                                                vgprOffset += 1
                                        else:
                                            pass
                                elif lrvwTile > 1 and not kernel["UseF32XEmulation"]:
                                    highBitsForHalf = 0
                                    isHigh8Bits = 0
                                    isHigh16Bits = 0
                                    numElementPerReg = int(writer.states.bpr//tP["bpe"])

                                    needPackK16  = False
                                    needPackK8Lw = False
                                    if kernel["ProblemType"][MacDataType].isHalf() or kernel["ProblemType"][MacDataType].isBFloat16():
                                        if writer.states.lrvwTileA > 1 or writer.states.lrvwTileB > 1:
                                            needPackK16 = True
                                        if writer.states.lrvwTileMetadata > 1:
                                            needPackK8Lw = True

                                    tPackM = "M" if needPackK16 and needPackK8Lw else ""

                                    if needPack or numSplitMetadata:
                                        destVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(MIInputPerThUnroll), vIdx*numVgpr), numVgpr)
                                    if rIdx == numReadsPerUnroll-1:
                                        for i in range(0, numVgpr):
                                            # convert from [tile][MiInputPerThread][vector] to [tile][vector][MiInputPerThread]
                                            vgprIdx = int((vIdx*numVgpr+i)*tP["bpeDS"]*MIInputPerThUnroll//writer.states.bpr*min(writer.states.bpr//tP["bpeDS"],vectorWidth))
                                            if numSplitMetadata:
                                                vgprIdx = (vIdx*numVgpr+i)*ceil(tP["bpeDS"]*MIInputPerThUnroll / writer.states.bpr)*min(writer.states.bpr//tP["bpeDS"],vectorWidth)
                                                if MIInputPerThUnroll == 8:
                                                    vgprOffset = 0
                                                    for elementIdx in range(0, numSplitMetadata+1):
                                                        if elementIdx >= writer.states.bpr:
                                                            break
                                                        packCode.add(VPermB32(
                                                            dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx + elementIdx * 2)),
                                                            src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 1, i+vIdx*numVgpr)),
                                                            src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 0, i+vIdx*numVgpr)),
                                                            src2=sgpr("PackKFor%sV%u"%(tPackM, vgprOffset)),
                                                            comment="1 select K=%u,%u for vector=%u"%(0, 1, vgprOffset)
                                                        ))
                                                        packCode.add(VPermB32(
                                                            dst=vgpr("PackTemp"),
                                                            src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 3, i+vIdx*numVgpr)),
                                                            src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 2, i+vIdx*numVgpr)),
                                                            src2=sgpr("PackKFor%sV%u"%(tPackM, vgprOffset)),
                                                            comment="1 select K=%u, %u for vector=%u"%(2, 3, vgprOffset)
                                                        ))
                                                        packCode.add(VLShiftLeftOrB32(
                                                            dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx + elementIdx * 2)),
                                                            src0=vgpr("PackTemp"),
                                                            shiftHex=16,
                                                            src1=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx + elementIdx * 2)),
                                                            comment="pack two half Vgpr to one Vgpr"
                                                            ))
                                                        packCode.add(VPermB32(
                                                            dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx + elementIdx * 2 + 1)),
                                                            src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 5, i+vIdx*numVgpr)),
                                                            src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 4, i+vIdx*numVgpr)),
                                                            src2=sgpr("PackKFor%sV%u"%(tPackM, vgprOffset)),
                                                            comment="2 select K=%u,%u for vector=%u"%(4, 5, vgprOffset)
                                                        ))
                                                        packCode.add(VPermB32(
                                                            dst=vgpr("PackTemp"),
                                                            src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 7, i+vIdx*numVgpr)),
                                                            src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 6, i+vIdx*numVgpr)),
                                                            src2=sgpr("PackKFor%sV%u"%(tPackM, vgprOffset)),
                                                            comment="2 select K=%u,%u for vector=%u"%(6, 7, vgprOffset)
                                                        ))
                                                        packCode.add(VLShiftLeftOrB32(
                                                            dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx + elementIdx * 2 + 1)),
                                                            src0=vgpr("PackTemp"),
                                                            shiftHex=16,
                                                            src1=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx + elementIdx * 2 + 1)),
                                                            comment="pack two half Vgpr to one Vgpr"
                                                            ))
                                                        vgprOffset += 1
                                                elif MIInputPerThUnroll == 4:
                                                    vgprOffset = 0
                                                    for elementIdx in range(0, numSplitMetadata+1):
                                                        if elementIdx >= writer.states.bpr:
                                                            break
                                                        # since the number of input thread is 4, so will alwasy be D0, D1, D2, D3
                                                        packCode.add(VPermB32(
                                                            dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx)),
                                                            src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 1, i+vIdx*numVgpr)),
                                                            src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 0, i+vIdx*numVgpr)),
                                                            src2=sgpr("PackKFor%sV%u"%(tPackM, vgprOffset)),
                                                            comment="1 select K=%u%u for vector=%u"%(0, 1, vgprOffset)
                                                            ))
                                                        packCode.add(VPermB32(
                                                            dst=vgpr("PackTemp"),
                                                            src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 3, i+vIdx*numVgpr)),
                                                            src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 2, i+vIdx*numVgpr)),
                                                            src2=sgpr("PackKFor%sV%u"%(tPackM, vgprOffset)),
                                                            comment="1 select K=%u%u for vector=%u"%(2, 3, vgprOffset)
                                                            ))
                                                        packCode.add(VLShiftLeftOrB32(
                                                            dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx)),
                                                            src0=vgpr("PackTemp"),
                                                            shiftHex=16,
                                                            src1=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx)),
                                                            comment="pack two half Vgpr to one Vgpr"
                                                            ))
                                                        vgprOffset += 1
                                                elif MIInputPerThUnroll == 2:
                                                    vgprOffset = 0
                                                    for elementIdx in range(0, numSplitMetadata+1):
                                                        if elementIdx >= writer.states.bpr:
                                                            break
                                                        # since the number of input thread is 2, so will alwasy be D0 and D1
                                                        packCodeT.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx)), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 1, i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 0, i+vIdx*numVgpr)), src2=sgpr("PackKFor%sV%u"%(tPackM, vgprOffset)), \
                                                                            comment="select K=%u%u for vector=%u"%(0, 1, vgprOffset)))
                                                        vgprOffset += 1
                                                elif MIInputPerThUnroll == 1:
                                                    destVgpr_ = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(MIInputPerThUnroll), vIdx*numVgpr + i))
                                                    bitShift = 0
                                                    for elementIdx in range(0, numSplitMetadata+1):
                                                        # go to next vgpr
                                                        if elementIdx >= writer.states.bpr:
                                                            break
                                                        comment_ = "another VGPR storing lshr %d-bit value %d %d" %(bitShift, vgprIdx, elementIdx) if bitShift != 0 else ""
                                                        packCodeT.add(VMovB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx)), src=destVgpr_, comment=comment_))
                                                        if bitShift != 0:
                                                            packCodeT.add(VLShiftRightB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx)), shiftHex=hex(bitShift), src=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx)), comment="ValuMetadata Vpgr >> %d" % bitShift))
                                                        bitShift += 8
                                                else:
                                                    assert False
                                            elif tP["isM"]:
                                                vgprOffset = 0
                                                for elementIdx in range(0, MIInputPerThUnroll):
                                                    packCodeT.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx+vIdx*2)), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, vgprOffset*2 + 1 , i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, vgprOffset*2, i+vIdx*numVgpr)), src2=sgpr("PackKForV%u"%elementIdx), \
                                                                        comment="select K=%u%u for vector=%u"%(vgprOffset*2+1, vgprOffset*2, elementIdx)))
                                                    vgprOffset += (1 if elementIdx % 2 == 1 else 0)
                                            elif kernel["ProblemType"][MacDataType].isHalf() or kernel["MFMA_BF16_1K"] or kernel["ProblemType"][MacDataType].isBFloat16():
                                                vgprOffset = 0
                                                for vectorIdx in range(0, numElementPerReg):
                                                    for elementIdx in range(0, int(tP["bpe"]*MIInputPerThUnroll//writer.states.bpr)):
                                                        packCodeT.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+vgprOffset)), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*numElementPerReg+1, i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*numElementPerReg, i+vIdx*numVgpr)), src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                            comment="select K=%u%u for vector=%u"%(elementIdx*numElementPerReg,  elementIdx*numElementPerReg+1, vectorIdx)))
                                                        vgprOffset += 1
                                            elif kernel["ProblemType"][MacDataType].isInt8() or kernel["ProblemType"][MacDataType].is8bitFloat():
                                                vgprOffset = 0
                                                # vertorIdx 2,3 is for the case vectorWidth > 2
                                                for vectorIdx in range(0, numElementPerReg):
                                                    if vectorWidth <= 2 and vectorIdx > 1:
                                                        break
                                                    for elementIdx in range(0, int(tP["bpe"]*MIInputPerThUnroll//writer.states.bpr)):
                                                        packCodeT.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+vgprOffset)), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*numElementPerReg+1, i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*numElementPerReg, i+vIdx*numVgpr)), src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                            comment="select K=%u%u for vector=%u"%(elementIdx*4,  elementIdx*4+1, vectorIdx)))
                                                        packCodeT.add(VPermB32(dst=vgpr("PackTemp"), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*numElementPerReg+3, i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*numElementPerReg+2, i+vIdx*numVgpr)), src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                            comment="select K=%u%u for vector=%u"%(elementIdx*4+2,  elementIdx*4+3, vectorIdx)))
                                                        packCodeT.add(VLShiftLeftOrB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx + vgprOffset)), src0=vgpr("PackTemp"), shiftHex=16, src1=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx + vgprOffset)), comment="pack two half Vgpr to one Vgpr"))
                                                        vgprOffset += 1

                                else:
                                    isHigh8Bits  = (blockWidth == 0.25) and ( ((rIdx % 4) % 2) == 1) # 1,3
                                    # pack for blockWidth 0.5 type
                                    if tP["isM"]:
                                        elementsPerPack = kernel["MIInputPerThreadMetadata"]
                                        vgprsPerPack = ceil(kernel["MIInputPerThreadMetadata"] / writer.states.bpr)
                                        dstVgpr  = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valufIdx//elementsPerPack*vgprsPerPack), numVgpr)
                                        if (blockWidth == 0.25):
                                            if ((rIdx % elementsPerPack) == (elementsPerPack-1)):  #rIdx = 1, 3,...
                                                for rIdx_ in range(0, ceil(rIdx/2)):
                                                    if rIdx_:
                                                        lowVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx_*2, valufIdx//elementsPerPack), numVgpr)
                                                    else:
                                                        lowVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valufIdx//elementsPerPack*vgprsPerPack), numVgpr)
                                                    highVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx_*2 + 1, valufIdx//elementsPerPack), numVgpr)
                                                    packCode.add(VLShiftLeftOrB32(dst=lowVgpr, src0=highVgpr, shiftHex=8, src1=lowVgpr, comment="pack two int8 Vgpr to one half Vgpr"))
                                                if elementsPerPack >= 4:
                                                    for rIdx_ in range(0, ceil(rIdx/4)):
                                                        if rIdx_:
                                                            lowVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx_*4, valufIdx//elementsPerPack), numVgpr)
                                                        else:
                                                            lowVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valufIdx//elementsPerPack*vgprsPerPack), numVgpr)
                                                        highVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx_*4 + 2, valufIdx//elementsPerPack), numVgpr)
                                                        dstVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valufIdx//elementsPerPack*vgprsPerPack+rIdx_), numVgpr)
                                                        packCode.add(VLShiftLeftOrB32(dst=dstVgpr, src0=highVgpr, shiftHex=hex(0x10), src1=lowVgpr, comment="pack two int8x2 Vgpr to one Vgpr"))
                                            if needPackMetadata:
                                                destVgpr = dstVgpr
                                                if rIdx % elementsPerPack:
                                                    destVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx % elementsPerPack, valufIdx//elementsPerPack), numVgpr)
                                        #don't use the ds read high
                                        isHigh8Bits = False
                                        isHigh16Bits = False
                                    elif writer.states.archCaps["HasEccHalf"] or not writer.states.asmCaps["HasWMMA_V1"]: # ECC pack
                                        if highBitsForHalf:
                                            highVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%2, valuiIdx), numVgpr)
                                            if writer.states.archCaps["DSLow16NotPreserve"]:
                                              packCodeT.add(VLShiftLeftOrB32(dst=baseLRVgpr, src0=highVgpr, shiftHex=hex(0x10), src1=baseLRVgpr, comment="pack two half Vgpr to one Vgpr"))
                                            else:
                                              packCodeT.add(VOrB32(dst=baseLRVgpr, src0=baseLRVgpr, src1=highVgpr, comment="pack two half Vgpr to one Vgpr"))
                                            destVgpr = highVgpr
                                        # pack for blockWidth 0.25 type
                                        if rIdx != 0:
                                            if isHigh8Bits or isHigh16Bits:
                                                highVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%4, valuiIdx), numVgpr)
                                                destVgpr = highVgpr
                                            if isHigh8Bits:
                                                lowVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, (rIdx%4)-1, valuiIdx), numVgpr) if isHigh16Bits else baseLRVgpr
                                                packCodeT.add(VLShiftLeftOrB32(dst=lowVgpr, src0=highVgpr, shiftHex=8, src1=lowVgpr, comment="pack two int8 Vgpr to one half Vgpr"))
                                                if isHigh16Bits:
                                                    if writer.states.archCaps["DSLow16NotPreserve"]:
                                                      packCodeT.add(VLShiftLeftOrB32(dst=baseLRVgpr, src0=lowVgpr, shiftHex=hex(0x10), src1=baseLRVgpr, comment="pack two half Vgpr to one Vgpr"))
                                                    else:
                                                      packCodeT.add(VOrB32(dst=baseLRVgpr, src0=baseLRVgpr, src1=lowVgpr, comment="pack two half Vgpr to one Vgpr"))
                                    else: # no ECC pack
                                    # pack for No ECC blockwidth 0.25 type
                                        if rIdx != 0:
                                            if isHigh8Bits:
                                                highVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%2, valuiIdx), numVgpr)
                                                destVgpr = highVgpr
                                            if isHigh8Bits and isHigh16Bits:
                                                packCodeT.add(VLShiftLeftOrB32(dst=baseLRVgpr, src0=highVgpr, shiftHex=hex(0x8), src1=baseLRVgpr, comment="pack two int8x2 Vgpr to one Vgpr"))

                            if kernel["ConvertAfterDS"] and kernel["UnrollMajorLDS%s"%tc]:
                                valufIdx += blockWidth * (tP["bpe"] // tP["bpeDS"]) if (not tP["isM"]) else 1
                            # workaround for gfx950 MX
                            # need to increment valufIdx by 1
                            elif isgfx950mx:
                                valufIdx += 1
                            else:
                                valufIdx += blockWidth if (not tP["isM"]) else (numVgpr if writer.states.asmCaps["HasSWMMAC_gfx1250"] else 1)

                            # load read instrution
                            paramList = []

                            # gfx1250 LDS offset formula shared by XF32 and BF16/Half/FP8/etc paths.
                            # The WMMA V3 LDS layout uses a *2 factor on the unroll stride.
                            def calcGfx1250LdsOffset():
                                if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
                                    incOffset = rIdx * numElementPerRead * UnrollStride * 2
                                    incOffset += tiIdx * matrixInstTO * vectorWidth * tileStride
                                else:
                                    vw = kernel[f"LocalReadVectorWidth{tc if('MXS' not in tc) else 'MXS'}"]
                                    incOffset = (rIdx // vw) * UnrollStride * vw
                                    incOffset += rIdx * numElementPerRead * UnrollStride
                                return int((incOffset + offset_val + tP["localReadOffset"]) * tP["bpeDS"])

                            for oIdx in range(0, numOffsets):
                                if perpStride > 1 and kernel["ProblemType"]["TLU%s"%tc] == 0:
                                    permBlock = kernel["MatrixInstK"] if kernel["ProblemType"]["TLU%s"%tc] == 1 else kernel["VectorWidth%s"%tc] * kernel["MatrixInstM"]
                                    perpStrideInv = permBlock // perpStride
                                    offset_val = (eIdx * (perpStrideInv) + ((vIdx) * numOffsets+oIdx) * MIWaveGroupShape[tile01]) * tileStride
                                else:
                                    offset_val = (eIdx + (vIdx * numOffsets + oIdx) * MIWaveGroupShape[tile01]) * tileStride

                                if kernel["ProblemType"]["Sparse"] != 0:
                                    if blocksPerTGroupSMFMA > 1:
                                        blockId = (rIdx * numElementPerRead) // elementsPerBlockSMFMA  # block 0 or block 1
                                        if kernel["UnrollMajorLDS%s"%(tc)]:
                                            offset_val = offset_val + (blockOffsetSMFMA * blockId)
                                        else:
                                            offset_val = offset_val + (blockOffsetSMFMA * blockId) * UnrollStride
                                    offset_val = int((rIdx * numElementPerRead * UnrollStride + offset_val + tP["localReadOffset"]) * tP["bpeDS"])
                                elif writer.states.asmCaps["HasMFMA_f8f6f4"] and kernel["ProblemType"][MacDataType].is8bitFloat() and kernel["MatrixInstK"] > 32 and not isgfx950mx:
                                    incOffset = 0
                                    midIdx = numReadsPerUnroll // 2
                                    if rIdx >= midIdx:
                                        if kernel["UnrollMajorLDS%s" % tP["tensorChar"]] == False:
                                            # TODO: why are these the offsets???
                                            if kernel["MatrixInstM"] == 32:
                                                 incOffset = midIdx * numElementPerRead * UnrollStride
                                            elif kernel["MatrixInstM"] == 16:
                                                incOffset = 3 * midIdx * numElementPerRead * UnrollStride
                                        else:
                                            if kernel["MatrixInstM"] == 32:
                                                incOffset = 16
                                            elif kernel["MatrixInstM"] == 16:
                                                incOffset = 48
                                    incOffset = rIdx * numElementPerRead * UnrollStride + incOffset
                                    offset_val = (incOffset + offset_val + tP["localReadOffset"]) * tP["bpeDS"]
                                elif kernel["UseF32XEmulation"]:
                                    # Previously a single ds_read could be used to load all inputs for mfma
                                    # For emulated TF32, 2x ds_read is required along with a different mfma layout
                                    # so we need to adjust the offsets accordingly for the second ds_read.
                                    if tuple(kernel["ISA"][:2]) == (12, 5):
                                        # gfx1250 WMMA V3: shared offset formula with BF16/Half (see calcGfx1250LdsOffset)
                                        offset_val = calcGfx1250LdsOffset()
                                    else:
                                        # Numbers here are specific to the mfma layout (gfx950)
                                        incOffset = 0
                                        midIdx = numReadsPerUnroll // 2
                                        if rIdx >= midIdx:
                                            if kernel["UnrollMajorLDS%s" % tP["tensorChar"]] == False:
                                                if kernel["MatrixInstM"] == 32:
                                                    incOffset = midIdx * numElementPerRead * UnrollStride
                                                elif kernel["MatrixInstM"] == 16:
                                                    incOffset = 3 * midIdx * numElementPerRead * UnrollStride
                                            else:
                                                if kernel["MatrixInstM"] == 32 and kernel["MatrixInstK"] == 16:
                                                    incOffset = 4
                                                elif kernel["MatrixInstM"] == 16 and kernel["MatrixInstK"] == 32:
                                                    incOffset = 12
                                        incOffset = rIdx * numElementPerRead * UnrollStride + incOffset
                                        offset_val = (incOffset + offset_val + tP["localReadOffset"]) * tP["bpeDS"]
                                # For wmma_v3, the maximum number of bytes per read is 16 bytes in 4 vgprs, which happens in the case of fp16/bf16/fp8/bf8/fp6/bf6/f4.
                                elif tuple(kernel["ISA"][:2]) == (12, 5) \
                                        and (kernel["ProblemType"][MacDataType].is8bitFloat() or kernel["ProblemType"][MacDataType].isBFloat16() \
                                            or kernel["ProblemType"][MacDataType].isHalf() or kernel["ProblemType"][MacDataType].isFloat4() \
                                                or kernel["ProblemType"][MacDataType].is6bitFloat() or kernel["ProblemType"][MacDataType].isInt8()):
                                    offset_val = calcGfx1250LdsOffset()
                                else:
                                    offset_val = int((rIdx * numElementPerRead * UnrollStride + offset_val + tP["localReadOffset"]) * tP["bpeDS"])

                                if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                                    offset_val = int(offset_val + (offset_val // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeDS"])
                                offset_val = offset_val + tP["localReadSwapByteOffset"]
                                # TODO: Add NLC>1 offset calcs here? 
                                if (kernel["DirectToLds%s" % tc] and  \
                                    kernel["GlobalReadVectorWidth%s"%tc] * tP["bpeDS"] > 4) and not kernel["UseGeneralizedNLCOne%s"%tc]:
                                  # another address conversion for DirectToLds + NumLoadsCoalesced > 1
                                  dummy, offset_val = writer.lraOffsetConversionForDTLandNLC(kernel, tP, offset_val)

                                paramList.append(int(offset_val))

                                comment = "L -> Reg lro=%d swapByteOffset=%u ti=%u vIdx=%u eIdx=%u rIdx=%u oIdx=%u buffer=%u iui=%u" \
                                    % (tP["localReadOffset"], tP["localReadSwapByteOffset"], MIWaveGroupShape[tile01], vIdx, eIdx, rIdx, oIdx, bufferIdx, iui)

                            highBits = 0 if writer.states.archCaps["DSLow16NotPreserve"] else highBitsForHalf or isHigh16Bits

                            addrIdx = paramList[0] // 65536
                            srcAddr=vgpr("LocalReadAddr%s+%u"%(tc, addrIdx))
                            paramList[0] -= addrIdx * 65536

                            if numOffsets == 1:
                                ds = DSModifiers(na=1, offset=paramList[0])
                            else:
                                ds = DSModifiers(na=2, offset0=paramList[0], offset1=paramList[1])
                            LocalReadX = instruction.getInst(highBits)
                            if kernel["UseF32XEmulation"]:
                                index = valuiIdx
                                # UseF32XEmulation case, convert index in transpose case.
                                # getVgprForEmu handles both transose and non-transpose cases.
                                # dest of local read is source of TF32 conv. Need to specify dest=False for getVgprForEmu
                                # indexTranpose case, disable index conversion for local read
                                destVgpr = self.getVgprForEmu(writer, kernel, tc, bufferIdx, iui, index, lrvwTile, vgprLen=numVgpr, dst=False, localRead=True)

                            self._emitLdsRead(writer, kernel, tP, LocalReadX, dst=destVgpr, src=srcAddr, ds=ds, module=localReadCodeT, comment=comment)
                            # TODO - handle vector-load
                            with writer.allocTmpSgpr(1) as tmpSgprInfo:
                                tmpSgpr = tmpSgprInfo.idx
                                if writer.db["CheckValue1%s"%tc] and not writer.inTailLoop:

                                    dbgVgpr = destVgpr
                                    dbgVgprList = destVgpr.split("v[")
                                    if len(dbgVgprList) == 1: # vIdx, no []
                                        dbgVgpr = dbgVgprList[0]
                                    else:
                                        # We only check the first one now
                                        # TODO: Handle vector, but need to take care the last one
                                        dbgVgprList = (dbgVgprList[1].split("]")[0]).split(':')
                                        dbgVgpr = "v[%s]"%dbgVgprList[0]
                                    localReadCodeT.add(SWaitCnt(dscnt=0, vscnt=0, comment="CheckValue1 wait for LDS read"))

                                    if kernel["ProblemType"][MacDataType].isHalf():
                                        hexValue = hex(0x3c003c00)     # packed 1s
                                        if needPack:
                                            hexValue = hex(0x3c000000) if highBitsForHalf else hex(0x00003c00)
                                        localReadCodeT.add(SMovB32(dst=sgpr(tmpSgpr), src=hexValue, comment="CheckValue1: FP16"))
                                        localReadCodeT.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                                    elif kernel["ProblemType"][MacDataType].isBFloat16():
                                        hexValue = hex(0x3f803f80)     # packed 1s
                                        if needPack:
                                            hexValue = hex(0x3f800000) if highBitsForHalf else hex(0x00003f80)
                                        localReadCodeT.add(SMovB32(dst=sgpr(tmpSgpr), src=hexValue, comment="CheckValue1: BF16"))
                                        localReadCodeT.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                                    if kernel["ProblemType"][MacDataType].isInt8():
                                        if needPack:
                                            hexValue = hex(0x00010000) if isHigh16Bits else hex(0x00000001)
                                            localReadCodeT.add(SMovB32(dst=sgpr(tmpSgpr), src=hexValue, comment="CheckValue1: INT8"))
                                            localReadCodeT.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                                    # TODO - Check if this works. But need this? MFMA would use INT8
                                    elif kernel["ProblemType"][MacDataType].isInt8x4():
                                        localReadCodeT.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(0x01010101), comment="CheckValue1: INT8x4"))
                                        localReadCodeT.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                                    elif kernel["ProblemType"][MacDataType].isSingle():
                                        localReadCodeT.add(writer.assert_eq( dbgVgpr, 1.0) )

                            addPackLR = False
                            if ((subTileIdx == 0 and subIterLoadCount < totalLoads // numSubTiles) \
                               or (subTileIdx == 1 and subIterLoadCount >= totalLoads // numSubTiles) \
                               or numSubTiles == 1) or writer.states.inTailLoop:
                                addPackLR = True

                            if addPackLR:
                                if needPack or numSplitMetadata:
                                    packCode.add(packCodeT)
                                    packCodePre.add(packCodePreT)
                                localReadCode.add(localReadCodeT)

                            subIterLoadCount += 1
                    # End of loop3
                    if needPack:
                        if tP["isA"]:
                            writer.states.a.numPackCvt = len(packCode.flatitems())
                        elif tP["isB"]:
                            writer.states.b.numPackCvt = len(packCode.flatitems())
                # End of loop2
            # End of loop1
        # DTV case, do not return local read code.
        if (tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc]:
            imod = Module("LocalReadDo%s_I%s (Empty)" % (tP["tensorChar"],iui))

        # DTV and Tr Load case, do not return pack code
        if (tP["isA"] or tP["isB"]) and kernel["enableGLTr%s"%tc]:
            pack = Module("Pack%s_I%s (Empty)" % (tP["tensorChar"],iui))

        # free any remaining tmp vgprs from emulation
        if useDirect32XEmulation:
            while len(tmpvgprFP32):
                tmp = tmpvgprFP32.pop()
                writer.vgprPool.checkIn(tmp)

        return imod, pack, packPre
