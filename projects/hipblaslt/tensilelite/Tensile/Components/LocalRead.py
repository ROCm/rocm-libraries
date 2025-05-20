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

from rocisa.code import Module
from rocisa.container import DSModifiers, vgpr, sgpr, SDWAModifiers, VOP3PModifiers
from rocisa.enum import SelectBit
from rocisa.instruction import SMovB32, SWaitCnt, VOrB32, VPermB32, VLShiftLeftOrB32, \
                            VMovB32, VLShiftRightB32, VCvtPkFP8toF32, VCvtF32toF16, VCvtFP8toF32,VCvtScaleFP8toF16,VCvtScalePkFP8toF16

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
            numReadsPerVector = (writer.states.lrvwUnrollA * tP["bpe"]) // (blockWidth*4) # bytes/register
            LdsPad            = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
            tileStride        = kernel["_DepthU%s"%tc] + LdsPad if kernel["UnrollMajorLDS%s" % tP["tensorChar"]] else 1
        else:
            numVectorsPerTile = (kernel["ThreadTile%u"%tile01]//kernel["VectorWidthA"])
            numReadsPerVector = (kernel["VectorWidthA"] * tP["bpe"]) // (blockWidth*4) # bytes/register

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
                        paramList.append(((rIdx*blockWidth + kernel["SubGroup%u"%tile01] * (vIdx*numOffsets+oIdx) * tileStride \
                            + tP["localReadOffset"]) * tP["bpe"] + tP["localReadSwapByteOffset"]) // offsetMultiplier)
                    else:
                        paramList.append(((rIdx*blockWidth + kernel["SubGroup%u"%tile01] * (vIdx*numOffsets+oIdx)*kernel["VectorWidthA"] \
                            + tP["localReadOffset"]) * tP["bpe"] + tP["localReadSwapByteOffset"]) // offsetMultiplier)
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
                    ds = DSModifiers(na=1, offset=paramList[0])
                if numOffsets == 2:
                    ds = DSModifiers(na=2, offset0=paramList[0], offset1=paramList[1])
                LocalReadX = instruction.getInst()
                localReadCode.add(LocalReadX(dst=destVgpr, src=srcVgpr, ds=ds))
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
                        localReadCode.add(SWaitCnt(lgkmcnt=0, comment="CheckValue1 wait for lds read"))
                        if writer.archCaps["SeparateVscnt"]:
                            # localReadCode.addInst( "s_waitcnt_vscnt", -2, "0", "")
                            localReadCode.add(SWaitCnt(vmcnt=0, vscnt=0))

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

        return imod, pack

class LocalReadMFMA(LocalRead):
    kernel = {"EnableMatrixInstruction": True}

    """
    Local Read: Do It A/B
    iui = Inner Unroll Idx
    epsi = expand pointer swap index. Only used for PAP
    """
    def __call__(self, writer, kernel, bufferIdx, iui, epsi, tP):
        imod = Module("LocalReadDo%s_I%s" % (tP["tensorChar"],iui))

        tc = tP["tensorChar"]
        if tc == "A":
            writer.states.localReadDoCntA += 1
        elif tc == "Metadata":
            writer.states.localReadDoCntMetadata += 1
        else:
            writer.states.localReadDoCntB += 1
        tile01           = tP["tile01Idx"]
        instruction      = tP["localReadInstruction"]
        bpr              = 4 # bytes/register

        numOffsets       = instruction.numOffsets
        blockWidth       = instruction.blockWidth
        unrollBlockWidth = instruction.blockWidth if kernel["UnrollMajorLDS%s"%tc] else tP["bpeDS"]/4
        tileBlockWidth   = tP["bpeDS"]/4 if kernel["UnrollMajorLDS%s"%tc] else instruction.blockWidth

        vectorWidth  = kernel["VectorWidth%s"%tc]

        MIWaveGroupShape = [ kernel["MatrixInstM"] * kernel["MatrixInstBM"] * kernel["MIWaveGroup"][0] * kernel["VectorWidthA"], \
                            kernel["MatrixInstN"] * kernel["MatrixInstBN"] * kernel["MIWaveGroup"][1] * kernel["VectorWidthB"]]

        LdsPad           = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
        tileStride       = 1
        UnrollStride     = kernel["MacroTile%s" % tP["tensorChar"]] + LdsPad
        if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
            tileStride   = kernel["_DepthU%s"%tc] + LdsPad
            UnrollStride = 1

        numVectorsPerTile = kernel["MIWaveTile"][tile01] // vectorWidth
        numReadsPerVector = (vectorWidth * tP["bpeDS"]) // int(tileBlockWidth * bpr)
        # overloading numReadsPerUnroll for DirectToLds x2/x4 case when blockWidth of instruction < LocalReadVectorWidth
        # fp64 TLU=1 reading 0.5element/lane/read..
        # for TLU=0 case, blockWidth and LRVW should match
        enableLDSTr = tP["enableLDSTr"]
        numReadsPerUnroll = ceil(tP["bpeDS"] * kernel["MIInputPerThread%s"%tc] / int(unrollBlockWidth * bpr))
        numVgpr  = int(ceil(blockWidth))
        if tc == 'A':
            lrvwTile = writer.states.lrvwTileA
        elif tc == 'B':
            lrvwTile = writer.states.lrvwTileB
        elif tc == "Metadata":
            lrvwTile = writer.states.lrvwTileMetadata
        else:
            lrvwTile = 1
        numElementPerRead = 1 if kernel["ConvertAfterDS"] else (int(blockWidth * bpr) // tP['bpe'] // lrvwTile)
        inputPerThread   = kernel["LocalReadVectorWidth"] if not writer.states.inTailLoop else kernel["MIInputPerThread%s"%tc]

        # pack register
        if writer.states.archCaps["HasEccHalf"] or not writer.states.asmCaps["HasWMMA_V1"]:
            needPack = tP["bpeDS"] < 4 and not kernel["UnrollMajorLDS%s"%tc] and not tP["isM"]
            # specify I8 for the case that input number is equal to the localread blockwidth but need to split low and high bytes to different vgprs.
            needPackMetadata = tP["isM"] and ((kernel["MIInputPerThread%s"%tc] * tP["bpeDS"] / (blockWidth * 4) > 1) or (kernel["ProblemType"]["DataType"].numBytes() == 1 and writer.states.lrvwTileMetadata > 1))
            needPack |= needPackMetadata
        else:
            needPack = blockWidth == 0.25
        needPack |= (kernel["ConvertAfterDS"] and (tP["bpe"] != tP["bpeDS"]))
        pack     = Module("pack%s_I%s"%(tc,iui))

        # split Metadata when localread width > mi input
        numSplitMetadata = max(ceil((blockWidth * 4) // tP["bpeDS"]) - 1, 0) if tP["isM"] else 0
        valufIdx = 0
        if enableLDSTr:
            numberMTilesPerWave = kernel["MIWaveTile"][tile01]
            highBits = 0
            for tIdx in range(0, numberMTilesPerWave):
                offset_val = (tP["localReadOffset"]+MIWaveGroupShape[tile01]*tIdx) * tP["bpeDS"]
                if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                    offset_val = offset_val + (offset_val // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeDS"]
                paramList = []
                paramList.append(int(offset_val))
                ds = DSModifiers(na=1, offset=paramList[0])
                LocalReadX = instruction.getInst(highBits)
                destVgpr = vgpr("Valu%s_X%u_I%u+%u+0"%(tc,bufferIdx,iui, 4*tIdx), 2)
                comment = "LDS Transpose"
                valuiIdx = int(valufIdx)
                localReadCode = imod.add(Module("LocalRead%s Valu%u"%(tc,valuiIdx)))
                localReadCode.add(LocalReadX(dst=destVgpr, src=vgpr("LocalReadAddr%s"%tc), ds=ds, comment=comment))
                destVgpr = vgpr("Valu%s_X%u_I%u+%u+2"%(tc,bufferIdx,iui,4*tIdx), 2)
                offset_val = paramList[0]+UnrollStride*inputPerThread;
                paramList.append(int(offset_val))
                ds = DSModifiers(na=1, offset=paramList[1])
                localReadCode.add(LocalReadX(dst=destVgpr, src=vgpr("LocalReadAddr%s"%tc), ds=ds, comment=comment))
        else:
            for vIdx in range(0, numVectorsPerTile):
                for eIdx in range(0, numReadsPerVector):
                    valuiIdx = int(valufIdx)
                    localReadCode = imod.add(Module("LocalRead%s Valu%u"%(tc,valuiIdx)))
                    if needPack or numSplitMetadata:
                        packCode = pack.add(Module("packCode"))
                    for rIdx in range(0, numReadsPerUnroll):
                        valuiIdx = int(valufIdx)
                        baseLRVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx), numVgpr)
                        destVgpr = baseLRVgpr
                        highBitsForHalf = (blockWidth == 0.5) and ((rIdx % 2) == 1) # rIdx = 1
                        isHigh16Bits = (blockWidth == 0.25) and ( ((rIdx % 4) //2) == 1) # 2,3
    
                        if needPack or numSplitMetadata:
                            if kernel["ConvertAfterDS"] and (tP["bpe"] != tP["bpeDS"]):
                                highBitsForHalf = False
                                isHigh16Bits = False
                                #Case A
                                if kernel["UnrollMajorLDS%s"%tc]:
                                    cvtTimes = (blockWidth * writer.states.bpr // tP["bpeDS"]) // kernel["MIInputPerThread%s"%tc]
                                    for i in range(0, cvtTimes):
                                        offset = cvtTimes - i - 1
                                        if writer.states.asmCaps["Hascvtf16_fp8"]:
                                            packCode.add(VCvtScalePkFP8toF16(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+1+offset*2)),\
                                                                         src=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+offset)), scale=0x3f800000,\
                                                                         vop3=VOP3PModifiers(op_sel=[1,0,0,0]), comment="convert fp8 to f16"))
                                            packCode.add(VCvtScalePkFP8toF16(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+0+offset*2)),\
                                                                         src=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+offset)), scale=0x3f800000,\
                                                                        vop3=VOP3PModifiers(op_sel=[0,0,0,0]), comment="convert fp8 to f16"))
                                        else:
                                            packCode.add(VCvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+offset)),\
                                                                         sdwa=SDWAModifiers(src0_sel=SelectBit.WORD_1), comment="convert to F32"))
                                            packCode.add(VCvtF32toF16(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+1+offset*2)), src=vgpr("CvtTemp+0"),\
                                                                       sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0), comment="Convert to FP16"))
                                            packCode.add(VCvtF32toF16(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+1+offset*2)), src=vgpr("CvtTemp+1"),\
                                                                       sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
                                            packCode.add(VCvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+offset)),\
                                                                         sdwa=SDWAModifiers(src0_sel=SelectBit.WORD_0), comment="convert to F32"))
                                            packCode.add(VCvtF32toF16(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+0+offset*2)), src=vgpr("CvtTemp+0"),\
                                                                       sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0), comment="Convert to FP16"))
                                            packCode.add(VCvtF32toF16(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx+0+offset*2)), src=vgpr("CvtTemp+1"),\
                                                                       sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
                                #Case B
                                elif (writer.states.lrvwTileA == 1 and tc == 'A') or (writer.states.lrvwTileB == 1 and tc == 'B'):
                                    sdwa = SDWAModifiers(dst_sel=SelectBit.WORD_0) if (rIdx % 2 == 0) else SDWAModifiers(dst_sel=SelectBit.WORD_1)
                                    destVgpr   = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, int(valufIdx*2)), numVgpr)
                                    CvtDstVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, int(valufIdx*2)), numVgpr)
                                    if needPack and rIdx != 0:
                                        destVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%4, valuiIdx), numVgpr)
                                    if writer.states.asmCaps["Hascvtf16_fp8"]:
                                        sel = [0,0,0,0] if (rIdx % 2 == 0)  else [0,0,1,0]
                                        packCode.add(VCvtScaleFP8toF16(dst=CvtDstVgpr, src=destVgpr, scale=0x3f800000, vop3=VOP3PModifiers(op_sel=sel), comment="convert fp8 to f16"))
                                    else:
                                        packCode.add(VCvtFP8toF32(dst=destVgpr, src=destVgpr, sdwa=SDWAModifiers(src0_sel=SelectBit.BYTE_0)))
                                        packCode.add(VCvtF32toF16(dst=CvtDstVgpr, src=destVgpr, sdwa=sdwa, comment="Convert to FP16"))
                                #Case C
                                elif (writer.states.lrvwTileA == 2 and tc == 'A') or (writer.states.lrvwTileB == 2 and tc == 'B'):
                                    if needPack or numSplitMetadata:
                                        destVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(kernel["MIInputPerThread%s"%tc]), vIdx*numVgpr), numVgpr)
                                        for i in range(0, numVgpr):
                                            cvtDstVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(kernel["MIInputPerThread%s"%tc]), vIdx*numVgpr), numVgpr)
                                            if writer.states.asmCaps["Hascvtf16_fp8"]:
                                                packCode.add(VCvtScalePkFP8toF16(dst=destVgpr, src=destVgpr,scale=0x3f800000,vop3=VOP3PModifiers(op_sel=[0,0,0,0]),comment="convert F8 to F16"))
                                            else:
                                                packCode.add(VCvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=destVgpr, sdwa=SDWAModifiers(src0_sel=SelectBit.WORD_0), comment="convert to F32"))
                                                packCode.add(VCvtF32toF16(dst=destVgpr, src=vgpr("CvtTemp+0"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0), comment="Convert to FP16"))
                                                packCode.add(VCvtF32toF16(dst=destVgpr, src=vgpr("CvtTemp+1"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
    
                                        if rIdx == numReadsPerUnroll-1:
                                            for i in range(0, numVgpr):
                                                vgprIdx = (vIdx * numVgpr + i) * tP["bpe"] * kernel["MIInputPerThread%s"%tc] // writer.states.bpr * min(writer.states.bpr // tP["bpe"], vectorWidth)
                                                vgprOffset = 0
                                                for vectorIdx in range(0, 2):
                                                    for elementIdx in range(0, tP["bpe"]*kernel["MIInputPerThread%s"%tc]//writer.states.bpr):
                                                        packCode.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+vgprOffset)), \
                                                                            src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*2+1, i+vIdx*numVgpr)), \
                                                                            src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*2, i+vIdx*numVgpr)), \
                                                                            src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                            comment="select K=%u%u for vector=%u"%(elementIdx*2,  elementIdx*2+1, vectorIdx)))
                                                        vgprOffset += 1
                                #Case D
                                elif (writer.states.lrvwTileA == 4 and tc == 'A') or (writer.states.lrvwTileB == 4 and tc == 'B'):
                                    if needPack or numSplitMetadata:
                                        destVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(kernel["MIInputPerThread%s"%tc]), 2 * vIdx * numVgpr), numVgpr)
                                        cvtDestVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(kernel["MIInputPerThread%s"%tc]), 2 * vIdx * numVgpr + 1), numVgpr)
                                        packCode.add(VLShiftRightB32(dst=cvtDestVgpr, shiftHex=16, src=destVgpr, comment="shift 2 element to vgpr+1"))
                                        if writer.states.asmCaps["Hascvtf16_fp8"]:
                                            packCode.add(VCvtScalePkFP8toF16(dst=destVgpr, src=destVgpr,scale=0x3f800000, comment="convert F8 to F16"))
                                            packCode.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr, src=cvtDestVgpr,scale=0x3f800000, comment="convert F8 to F16"))
                                        else:
                                            packCode.add(VCvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=destVgpr, sdwa=SDWAModifiers(src0_sel=SelectBit.WORD_0), comment="convert to F32"))
                                            packCode.add(VCvtF32toF16(dst=destVgpr, src=vgpr("CvtTemp+0"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0), comment="Convert to FP16"))
                                            packCode.add(VCvtF32toF16(dst=destVgpr, src=vgpr("CvtTemp+1"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
                                            packCode.add(VCvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=cvtDestVgpr, sdwa=SDWAModifiers(src0_sel=SelectBit.WORD_0), comment="convert to F32"))
                                            packCode.add(VCvtF32toF16(dst=cvtDestVgpr, src=vgpr("CvtTemp+0"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0), comment="Convert to FP16"))
                                            packCode.add(VCvtF32toF16(dst=cvtDestVgpr, src=vgpr("CvtTemp+1"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
    
                                        if rIdx == numReadsPerUnroll-1:
                                            for i in range(0, numVgpr*2):
                                                vgprIdx = (2 * vIdx * numVgpr + i) * tP["bpe"] * kernel["MIInputPerThread%s"%tc] // writer.states.bpr * min(writer.states.bpr // tP["bpe"], vectorWidth)
                                                vgprOffset = 0
                                                for vectorIdx in range(0, 2):
                                                    for elementIdx in range(0, tP["bpe"]*kernel["MIInputPerThread%s"%tc]//writer.states.bpr):
                                                        packCode.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+vgprOffset)), \
                                                                            src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*2+1, i+2*vIdx*numVgpr)), \
                                                                            src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*2, i+2*vIdx*numVgpr)), \
                                                                            src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                            comment="select K=%u%u for vector=%u"%(elementIdx*2,  elementIdx*2+1, vectorIdx)))
                                                        vgprOffset += 1
                                #Case E
                                elif (writer.states.lrvwTileA == 8 and tc == 'A') or (writer.states.lrvwTileB == 8 and tc == 'B'):
                                    if needPack or numSplitMetadata:
                                        destVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(kernel["MIInputPerThread%s"%tc]), 2*vIdx*numVgpr), numVgpr)
                                        cvtDestVgpr0 = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(kernel["MIInputPerThread%s"%tc]), 2*vIdx*numVgpr+0), 1)
                                        cvtDestVgpr1 = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(kernel["MIInputPerThread%s"%tc]), 2*vIdx*numVgpr+1), 1)
                                        cvtDestVgpr2 = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(kernel["MIInputPerThread%s"%tc]), 2*vIdx*numVgpr+2), 1)
                                        cvtDestVgpr3 = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(kernel["MIInputPerThread%s"%tc]), 2*vIdx*numVgpr+3), 1)
                                        packCode.add(VLShiftRightB32(dst=cvtDestVgpr3, shiftHex=16, src=cvtDestVgpr1, comment="shift 2 element to vgpr+3"))
    
                                        packCode.add(VMovB32(dst=cvtDestVgpr2, src=cvtDestVgpr1))
                                        packCode.add(VLShiftRightB32(dst=cvtDestVgpr1, shiftHex=16, src=cvtDestVgpr0, comment="shift 2 element to vgpr+1"))
                                        if writer.states.asmCaps["Hascvtf16_fp8"]:
                                            packCode.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr0, src=cvtDestVgpr0,scale=0x3f800000, comment="convert F8 to F16"))
                                            packCode.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr1, src=cvtDestVgpr1,scale=0x3f800000, comment="convert F8 to F16"))
                                            packCode.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr2, src=cvtDestVgpr2,scale=0x3f800000, comment="convert F8 to F16"))
                                            packCode.add(VCvtScalePkFP8toF16(dst=cvtDestVgpr3, src=cvtDestVgpr3,scale=0x3f800000, comment="convert F8 to F16"))
                                        else:
                                            packCode.add(VCvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=cvtDestVgpr0, sdwa=SDWAModifiers(src0_sel=SelectBit.WORD_0), comment="convert to F32"))
                                            packCode.add(VCvtF32toF16(dst=cvtDestVgpr0, src=vgpr("CvtTemp+0"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0), comment="Convert to FP16"))
                                            packCode.add(VCvtF32toF16(dst=cvtDestVgpr0, src=vgpr("CvtTemp+1"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
                                            packCode.add(VCvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=cvtDestVgpr1, sdwa=SDWAModifiers(src0_sel=SelectBit.WORD_0), comment="convert to F32"))
                                            packCode.add(VCvtF32toF16(dst=cvtDestVgpr1, src=vgpr("CvtTemp+0"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0), comment="Convert to FP16"))
                                            packCode.add(VCvtF32toF16(dst=cvtDestVgpr1, src=vgpr("CvtTemp+1"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
                                            packCode.add(VCvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=cvtDestVgpr2, sdwa=SDWAModifiers(src0_sel=SelectBit.WORD_0), comment="convert to F32"))
                                            packCode.add(VCvtF32toF16(dst=cvtDestVgpr2, src=vgpr("CvtTemp+0"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0), comment="Convert to FP16"))
                                            packCode.add(VCvtF32toF16(dst=cvtDestVgpr2, src=vgpr("CvtTemp+1"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
                                            packCode.add(VCvtPkFP8toF32(dst=vgpr("CvtTemp", 2), src=cvtDestVgpr3, sdwa=SDWAModifiers(src0_sel=SelectBit.WORD_0), comment="convert to F32"))
                                            packCode.add(VCvtF32toF16(dst=cvtDestVgpr3, src=vgpr("CvtTemp+0"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0), comment="Convert to FP16"))
                                            packCode.add(VCvtF32toF16(dst=cvtDestVgpr3, src=vgpr("CvtTemp+1"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1), comment="Convert to FP16"))
    
                                        if rIdx == numReadsPerUnroll-1:
                                            for i in range(0, numVgpr*2):
                                                vgprIdx = (2 * vIdx * numVgpr + i) * tP["bpe"] * kernel["MIInputPerThread%s"%tc] // writer.states.bpr * min(writer.states.bpr // tP["bpe"], vectorWidth)
                                                vgprOffset = 0
                                                for vectorIdx in range(0, 2):
                                                    for elementIdx in range(0, tP["bpe"]*kernel["MIInputPerThread%s"%tc]//writer.states.bpr):
                                                        packCode.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+vgprOffset)), \
                                                                            src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*2+1, i+2*vIdx*numVgpr)), \
                                                                            src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*2, i+2*vIdx*numVgpr)), \
                                                                            src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                            comment="select K=%u%u for vector=%u"%(elementIdx*2,  elementIdx*2+1, vectorIdx)))
                                                        vgprOffset += 1
                                else:
                                    pass
                            elif lrvwTile > 1:
                                highBitsForHalf = 0
                                isHigh8Bits = 0
                                isHigh16Bits = 0
                                numElementPerReg = writer.states.bpr//tP["bpe"]

                                needPackK16  = False
                                needPackK8Lw = False
                                if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
                                    if writer.states.lrvwTileA > 1 or writer.states.lrvwTileB > 1:
                                        needPackK16 = True
                                    if writer.states.lrvwTileMetadata > 1:
                                        needPackK8Lw = True

                                tPackM = "M" if needPackK16 and needPackK8Lw else ""

                                if needPack or numSplitMetadata:
                                    destVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(kernel["MIInputPerThread%s"%tc]), vIdx*numVgpr), numVgpr)
                                if rIdx == numReadsPerUnroll-1:
                                    for i in range(0, numVgpr):
                                        # convert from [tile][MiInputPerThread][vector] to [tile][vector][MiInputPerThread]
                                        vgprIdx = (vIdx*numVgpr+i)*tP["bpeDS"]*kernel["MIInputPerThread%s"%tc]//writer.states.bpr*min(writer.states.bpr//tP["bpeDS"],vectorWidth)
                                        if numSplitMetadata:
                                            vgprIdx = (vIdx*numVgpr+i)*ceil(tP["bpeDS"]*kernel["MIInputPerThread%s"%tc] / writer.states.bpr)*min(writer.states.bpr//tP["bpeDS"],vectorWidth)
                                            if kernel["MIInputPerThread%s"%tc] == 4:
                                                vgprOffset = 0
                                                for elementIdx in range(0, numSplitMetadata+1):
                                                    if elementIdx >= writer.states.bpr:
                                                        break
                                                    # since the number of input thread is 4, so will alwasy be D0, D1, D2, D3
                                                    packCode.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx)), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 1, i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 0, i+vIdx*numVgpr)), src2=sgpr("PackKFor%sV%u"%(tPackM, vgprOffset)), \
                                                                       comment="1 select K=%u%u for vector=%u"%(0, 1, vgprOffset)))
                                                    packCode.add(VPermB32(dst=vgpr("PackTemp"), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 3, i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 2, i+vIdx*numVgpr)), src2=sgpr("PackKFor%sV%u"%(tPackM, vgprOffset)), \
                                                                       comment="1 select K=%u%u for vector=%u"%(2, 3, vgprOffset)))
                                                    packCode.add(VLShiftLeftOrB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx)), src0=vgpr("PackTemp"), shiftHex=16, src1=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx)), comment="pack two half Vgpr to one Vgpr"))
                                                    vgprOffset += 1
                                            elif kernel["MIInputPerThread%s"%tc] == 2:
                                                vgprOffset = 0
                                                for elementIdx in range(0, numSplitMetadata+1):
                                                    if elementIdx >= writer.states.bpr:
                                                        break
                                                    # since the number of input thread is 2, so will alwasy be D0 and D1
                                                    packCode.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx)), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 1, i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, 0, i+vIdx*numVgpr)), src2=sgpr("PackKFor%sV%u"%(tPackM, vgprOffset)), \
                                                                        comment="select K=%u%u for vector=%u"%(0, 1, vgprOffset)))
                                                    vgprOffset += 1
                                            elif kernel["MIInputPerThread%s"%tc] == 1:
                                                destVgpr_ = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(kernel["MIInputPerThread%s"%tc]), vIdx*numVgpr + i))
                                                bitShift = 0
                                                for elementIdx in range(0, numSplitMetadata+1):
                                                    # go to next vgpr 
                                                    if elementIdx >= writer.states.bpr:
                                                        break
                                                    comment_ = "another VGPR storing lshr %d-bit value %d %d" %(bitShift, vgprIdx, elementIdx) if bitShift != 0 else ""
                                                    packCode.add(VMovB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx)), src=destVgpr_, comment=comment_))
                                                    if bitShift != 0:
                                                        packCode.add(VLShiftRightB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx)), shiftHex=hex(bitShift), src=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx)), comment="ValuMetadata Vpgr >> %d" % bitShift))
                                                    bitShift += 8
                                            else:
                                                assert False
                                        elif tP["isM"]:
                                            vgprOffset = 0
                                            for elementIdx in range(0, kernel["MIInputPerThread%s"%tc]):
                                                packCode.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+elementIdx+vIdx*2)), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, vgprOffset*2 + 1 , i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, vgprOffset*2, i+vIdx*numVgpr)), src2=sgpr("PackKForV%u"%elementIdx), \
                                                                    comment="select K=%u%u for vector=%u"%(vgprOffset*2+1, vgprOffset*2, elementIdx)))
                                                vgprOffset += (1 if elementIdx % 2 == 1 else 0)
                                        elif kernel["ProblemType"]["DataType"].isHalf() or kernel["MFMA_BF16_1K"] or kernel["ProblemType"]["DataType"].isBFloat16():
                                            vgprOffset = 0
                                            for vectorIdx in range(0, numElementPerReg):
                                                for elementIdx in range(0, tP["bpe"]*kernel["MIInputPerThread%s"%tc]//writer.states.bpr):
                                                    packCode.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+vgprOffset)), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*numElementPerReg+1, i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*numElementPerReg, i+vIdx*numVgpr)), src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                        comment="select K=%u%u for vector=%u"%(elementIdx*numElementPerReg,  elementIdx*numElementPerReg+1, vectorIdx)))
                                                    vgprOffset += 1
                                        elif kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat():
                                            vgprOffset = 0
                                            # vertorIdx 2,3 is for the case vectorWidth > 2
                                            for vectorIdx in range(0, numElementPerReg):
                                                if vectorWidth <= 2 and vectorIdx > 1:
                                                    break
                                                for elementIdx in range(0, tP["bpe"]*kernel["MIInputPerThread%s"%tc]//writer.states.bpr):
                                                    packCode.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+vgprOffset)), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*numElementPerReg+1, i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*numElementPerReg, i+vIdx*numVgpr)), src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                        comment="select K=%u%u for vector=%u"%(elementIdx*4,  elementIdx*4+1, vectorIdx)))
                                                    packCode.add(VPermB32(dst=vgpr("PackTemp"), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*numElementPerReg+3, i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*numElementPerReg+2, i+vIdx*numVgpr)), src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                        comment="select K=%u%u for vector=%u"%(elementIdx*4+2,  elementIdx*4+3, vectorIdx)))
                                                    packCode.add(VLShiftLeftOrB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx + vgprOffset)), src0=vgpr("PackTemp"), shiftHex=16, src1=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx + vgprOffset)), comment="pack two half Vgpr to one Vgpr"))
                                                    vgprOffset += 1
    
                            else:
                                isHigh8Bits  = (blockWidth == 0.25) and ( ((rIdx % 4) % 2) == 1) # 1,3
                                # pack for blockWidth 0.5 type
                                if tP["isM"]:
                                    isHigh8Bits  = (blockWidth == 0.25) and ( (rIdx % 2) == 1) # rIdx = 1
                                    isHigh16Bits = (blockWidth == 0.25) and ( (rIdx % 4) == 3) if kernel["MIInputPerThread%s"%tc] == 4 else False # rIdx = 3
                                    if isHigh8Bits:
                                        dstVgpr  = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valufIdx/2), numVgpr)
                                        lowVgpr  = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valufIdx - 1), numVgpr)
                                        highVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valufIdx), numVgpr)
                                        packCode.add(VLShiftLeftOrB32(dst=dstVgpr, src0=highVgpr, shiftHex=8, src1=lowVgpr, comment="pack two int8 Vgpr to one half Vgpr"))
                                    if isHigh16Bits:
                                        # every 4 metadatas will be packed into one vgpr, so divide 4 to let dstVgrp be 0,1,2,...
                                        dstVgpr  = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valufIdx/4), numVgpr)
                                        lowVgpr  = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valufIdx/2 - 1), numVgpr)
                                        highVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valufIdx/2), numVgpr)
                                        packCode.add(VLShiftLeftOrB32(dst=dstVgpr, src0=highVgpr, shiftHex=hex(0x10), src1=lowVgpr, comment="pack two int8x2 Vgpr to one Vgpr"))
                                    # Metadata only use one vgpr in current SMFMA instructions, so doesn't need these two flags at localread (gfx94x, gfx95x).
                                    isHigh16Bits = False
                                    isHigh8Bits = False
                                elif writer.states.archCaps["HasEccHalf"] or not writer.states.asmCaps["HasWMMA_V1"]: # ECC pack
                                    if highBitsForHalf:
                                        highVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%2, valuiIdx), numVgpr)
                                        if writer.states.archCaps["DSLow16NotPreserve"]:
                                          packCode.add(VLShiftLeftOrB32(dst=baseLRVgpr, src0=highVgpr, shiftHex=hex(0x10), src1=baseLRVgpr, comment="pack two half Vgpr to one Vgpr"))
                                        else:
                                          packCode.add(VOrB32(dst=baseLRVgpr, src0=baseLRVgpr, src1=highVgpr, comment="pack two half Vgpr to one Vgpr"))
                                        destVgpr = highVgpr
                                    # pack for blockWidth 0.25 type
                                    if rIdx != 0:
                                        if isHigh8Bits or isHigh16Bits:
                                            highVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%4, valuiIdx), numVgpr)
                                            destVgpr = highVgpr
                                        if isHigh8Bits:
                                            lowVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, (rIdx%4)-1, valuiIdx), numVgpr) if isHigh16Bits else baseLRVgpr
                                            packCode.add(VLShiftLeftOrB32(dst=lowVgpr, src0=highVgpr, shiftHex=8, src1=lowVgpr, comment="pack two int8 Vgpr to one half Vgpr"))
                                            if isHigh16Bits:
                                                if writer.states.archCaps["DSLow16NotPreserve"]:
                                                  packCode.add(VLShiftLeftOrB32(dst=baseLRVgpr, src0=lowVgpr, shiftHex=hex(0x10), src1=baseLRVgpr, comment="pack two half Vgpr to one Vgpr"))
                                                else:
                                                  packCode.add(VOrB32(dst=baseLRVgpr, src0=baseLRVgpr, src1=lowVgpr, comment="pack two half Vgpr to one Vgpr"))
                                else: # no ECC pack
                                # pack for No ECC blockwidth 0.25 type
                                    if rIdx != 0:
                                        if isHigh8Bits:
                                            highVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%2, valuiIdx), numVgpr)
                                            destVgpr = highVgpr
                                        if isHigh8Bits and isHigh16Bits:
                                            packCode.add(VLShiftLeftOrB32(dst=baseLRVgpr, src0=highVgpr, shiftHex=hex(0x8), src1=baseLRVgpr, comment="pack two int8x2 Vgpr to one Vgpr"))
    
                        if kernel["ConvertAfterDS"] and kernel["UnrollMajorLDS%s"%tc]:
                            valufIdx += blockWidth * (tP["bpe"] // tP["bpeDS"]) if (not tP["isM"]) else 1
                        else:
                            valufIdx += blockWidth if (not tP["isM"]) else 1
    
                        # load read instrution
                        paramList = []
    
                        for oIdx in range(0, numOffsets):
                            offset_val = (eIdx + (vIdx * numOffsets+oIdx) * MIWaveGroupShape[tile01]) * tileStride
    
                            strideK = 1
                            # FIXME gfx950 sparse
                            if kernel["ProblemType"]["Sparse"] != 0:
                                if kernel["MIInputPerThread"] * kernel["ProblemType"]["DataTypeB"].numBytes() > 16: # double K
                                  is8bits = kernel["ProblemType"]["DataType"].numBytes() == 1
                                  isSparseTrack = (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]) or (kernel["ProblemType"]["Sparse"] == 2 and  tP["isB"]) or tP["isM"]
                                  if not isSparseTrack:
                                    if kernel["UnrollMajorLDS%s"%(tc)]:
                                        divider = 64 if is8bits else 32
                                        strideK = max(1, kernel["MatrixInstK"] // divider * numReadsPerUnroll)
                                    else:
                                        divider = 32
                                        elementsPerBlock = 16 // kernel["ProblemType"]["DataTypeB"].numBytes() # 16 bytes, 4 vgprs per block for gfx95x
                                        blockId = rIdx // elementsPerBlock
                                        offsetPerBlock = kernel["MatrixInstK"] // divider * 16 * blockId
                                        offset_val = offset_val + (offsetPerBlock - blockId * elementsPerBlock) * kernel["MacroTile%s"%tc]
                                offset_val = (strideK * rIdx * numElementPerRead * UnrollStride + offset_val + tP["localReadOffset"]) * tP["bpeDS"]
                            elif kernel["ProblemType"]["DataType"].is8bitFloat() and kernel["MatrixInstK"] > 32:
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
                            else:
                                offset_val = (rIdx * numElementPerRead * UnrollStride + offset_val + tP["localReadOffset"]) * tP["bpeDS"]
    
                            if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                                offset_val = offset_val + (offset_val // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpeDS"]
                            offset_val = offset_val + tP["localReadSwapByteOffset"]
                            if (kernel["DirectToLds%s" % tc] and  \
                                kernel["GlobalReadVectorWidth%c"%tc] * tP["bpeDS"] > 4):
                              # another address conversion for DirectToLds + NumLoadsCoalesced > 1
                              dummy, offset_val = writer.lraOffsetConversionForDTLandNLC(kernel, tP, offset_val)
    
                            paramList.append(int(offset_val))
    
                        comment = "L -> Reg lro=%d swapByteOffset=%u ti=%u vIdx=%u eIdx=%u rIdx=%u oIdx=%u buffer=%u iui=%u" \
                                % (tP["localReadOffset"], tP["localReadSwapByteOffset"], MIWaveGroupShape[tile01], vIdx, eIdx, rIdx, oIdx, bufferIdx, iui)
    
                        highBits = 0 if writer.states.archCaps["DSLow16NotPreserve"] else highBitsForHalf or isHigh16Bits
    
    
                        if(paramList[0] >=131072):
                            paramList[0] = paramList[0] -131072
                            srcAddr=vgpr("LocalReadAddr%s+2"%tc)
                        elif (paramList[0] >=65536):
                            paramList[0] = paramList[0] -65536
                            srcAddr=vgpr("LocalReadAddr%s+1"%tc)
                        else:
                            srcAddr=vgpr("LocalReadAddr%s"%tc)
    
                        if numOffsets == 1:
                            ds = DSModifiers(na=1, offset=paramList[0])
                        else:
                            ds = DSModifiers(na=2, offset0=paramList[0], offset1=paramList[1])
                        LocalReadX = instruction.getInst(highBits)
                        localReadCode.add(LocalReadX(dst=destVgpr, src=srcAddr, ds=ds, comment=comment))
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
    
                                localReadCode.add(SWaitCnt(lgkmcnt=0, vscnt=0, comment="CheckValue1 wait for LDS read"))
    
                                if kernel["ProblemType"]["DataType"].isHalf():
                                    hexValue = hex(0x3c003c00)     # packed 1s
                                    if needPack:
                                        hexValue = hex(0x3c000000) if highBitsForHalf else hex(0x00003c00)
                                    localReadCode.add(SMovB32(dst=sgpr(tmpSgpr), src=hexValue, comment="CheckValue1: FP16"))
                                    localReadCode.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))
    
                                elif kernel["ProblemType"]["DataType"].isBFloat16():
                                    hexValue = hex(0x3f803f80)     # packed 1s
                                    if needPack:
                                        hexValue = hex(0x3f800000) if highBitsForHalf else hex(0x00003f80)
                                    localReadCode.add(SMovB32(dst=sgpr(tmpSgpr), src=hexValue, comment="CheckValue1: BF16"))
                                    localReadCode.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))
    
                                if kernel["ProblemType"]["DataType"].isInt8():
                                    if needPack:
                                        hexValue = hex(0x00010000) if isHigh16Bits else hex(0x00000001)
                                        localReadCode.add(SMovB32(dst=sgpr(tmpSgpr), src=hexValue, comment="CheckValue1: INT8"))
                                        localReadCode.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))
    
                                # TODO - Check if this works. But need this? MFMA would use INT8
                                elif kernel["ProblemType"]["DataType"].isInt8x4():
                                    localReadCode.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(0x01010101), comment="CheckValue1: INT8x4"))
                                    localReadCode.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))
    
                                elif kernel["ProblemType"]["DataType"].isSingle():
                                    localReadCode.add(writer.assert_eq( dbgVgpr, 1.0) )

        # DTV case, do not return local read code. Return pack code only.
        if (tP["isA"] or tP["isB"]) and kernel["DirectToVgpr%s"%tc]:
          imod = Module("LocalReadDo%s_I%s (Empty)" % (tP["tensorChar"],iui))
        
        if kernel["UseF32XEmulation"] and kernel["EnableF32XEmulationLds"] and tP["isA"] and tc == "A":
            tf32mod = Module()
            tf32mod.add(TextBlock("/*TF32 Emulation read lds*/\n"))
            tf32mod.add(LocalReadX(dst=vgpr("Cvt+0"), src=vgpr("LocalReadAddrA"), ds=DSModifiers(na=1, offset=0)))
            tf32mod.add(LocalReadX(dst=vgpr("Cvt+1"), src=vgpr("LocalReadAddrA"), ds=DSModifiers(na=1, offset=256)))
            tf32mod.add(LocalReadX(dst=vgpr("Cvt+2"), src=vgpr("LocalReadAddrA"), ds=DSModifiers(na=1, offset=512)))
            tf32mod.add(LocalReadX(dst=vgpr("Cvt+3"), src=vgpr("LocalReadAddrA"), ds=DSModifiers(na=1, offset=768)))
            tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+0"), src=vgpr("Cvt+0"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1, src0_sel=SelectBit.WORD_1)))
            tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+1"), src=vgpr("Cvt+2"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1, src0_sel=SelectBit.WORD_1)))
            tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+2"), src=vgpr("Cvt+0"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1, src0_sel=SelectBit.WORD_0)))
            tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+3"), src=vgpr("Cvt+2"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1, src0_sel=SelectBit.WORD_0)))
            tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+0"), src=vgpr("Cvt+1"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0, src0_sel=SelectBit.WORD_1)))
            tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+1"), src=vgpr("Cvt+3"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0, src0_sel=SelectBit.WORD_1)))
            tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+2"), src=vgpr("Cvt+1"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0, src0_sel=SelectBit.WORD_0)))
            tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+3"), src=vgpr("Cvt+3"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0, src0_sel=SelectBit.WORD_0)))
            localReadCode.add(tf32mod)

        return imod, pack
