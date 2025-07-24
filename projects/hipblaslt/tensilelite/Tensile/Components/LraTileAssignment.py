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
from rocisa.container import vgpr, ContinuousRegister
from rocisa.instruction import VAddU32
from rocisa.functions import vectorStaticRemainder, \
    vectorStaticDivideAndRemainder, vectorStaticDivide, vectorStaticMultiply, \
    vectorStaticMultiplyAdd

from ..Component import LraTileAssignment, LraTileProperties
from dataclasses import dataclass

@dataclass
class LraTilePropertiesMFMA(LraTileProperties):
   dividendForKId: int
   num1DBlocks: int
   num1DWaves: int
   dividedForBlkId: int
   dividedForWaveId: int
   vectorWidth: int
   maxKId: int

class LraTileAssignmentVALU(LraTileAssignment):
    kernel = {"EnableMatrixInstruction": False}

    """
    Local Read Addresses: Tile Assignment
    """
    def __call__(self, writer, kernel, tP):
        module = Module("LraTileAssignmentVALU")

        # allocate resources
        qReg    = writer.vgprPool.checkOut(1,"qReg") # quotient
        rReg    = writer.vgprPool.checkOut(1,"rReg") # remainder
        # dot2: currently only support unroll major LDS
        tc               = tP["tensorChar"]
        umlds            = kernel["UnrollMajorLDS%s" % tc]
        LdsPad           = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0
        strideTile       = kernel["_DepthU%s"%tc] + LdsPad if umlds else 1
        tmpVgpr          = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr")
        tmpVgprRes       = ContinuousRegister(tmpVgpr, 2)

        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            if tP["tileIdx"] == 0:
                # kStr += "%slr%s = serial %% SG%s%s%s" \
                #         % (writer.commentPrefix, tP["tileChar"], tP["tileChar"], \
                #         writer.commentSuffix, writer.endLine)

                # constant
                dividendReg = "Serial" # local serial
                divisor = kernel["SubGroup0"]
                # dot2: waveSplitK
                if kernel["UseDotInstruction"]:
                    if kernel["NumWaveSplitK"] > 1:
                        newSerial = writer.vgprPool.checkOut(1,"newSerial")
                        module.add(vectorStaticDivide(newSerial, dividendReg, kernel["NumWaveSplitK"], tmpVgprRes, \
                        "Divided by NumWaveSplitK(%u)" % kernel["NumWaveSplitK"]))
                        # generate instruction
                        module.add(vectorStaticDivideAndRemainder(qReg, rReg, newSerial, divisor, tmpVgprRes))
                        # tile offset
                        module.add(vectorStaticMultiply(vgpr(rReg), vgpr(rReg), strideTile, tmpSgprInfo, \
                        "1. M offset: mOffset = mIdx * mStride(%u)" % strideTile))
                        writer.vgprPool.checkIn(newSerial)
                    else:
                        module.add(vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpVgprRes))
                        # tile offset
                        module.add(vectorStaticMultiply(vgpr(rReg), vgpr(rReg), strideTile, tmpSgprInfo, \
                        "1. M offset: mOffset = mIdx * mStride(%u)" % strideTile))
                else:
                    module.add(vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpVgprRes))

                # release and return resource
                tP["gpr"]["lro"] = rReg
                writer.tmplro = qReg
            else:
                # kStr += "%slr%s = (serial / SG%s) %% SG%s%s%s" \
                #         % (writer.commentPrefix, tP["tileChar"], tP["tileChar"], \
                #         tP["tileChar"], writer.commentSuffix, writer.endLine)

                # constant
                divisor = kernel["SubGroup1"]
                dividendReg = writer.tmplro
                # generate instruction
                module.add(vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpVgprRes))

                if kernel["UseDotInstruction"]:
                    # tile offset
                    module.add(vectorStaticMultiply(vgpr(rReg), vgpr(rReg), strideTile, tmpSgprInfo, \
                    "1. N offset: nOffset = nIdx * nStride(%u)" % strideTile))

                # release and return resource
                tP["gpr"]["lro"] = rReg

                writer.vgprPool.checkIn(writer.tmplro) # old
                writer.vgprPool.checkIn(qReg)

        writer.vgprPool.checkIn(tmpVgpr)

        return module

class LraTileAssignmentMFMA(LraTileAssignment):
    kernel = {"EnableMatrixInstruction": True}

    """
    Local Read Addresses: Tile Assignment A/B
    """
    def __call__(self, writer, kernel, tP):
        module = Module("LraTileAssignmentMFMA")
        module.addComment0("lr%s" % tP["tileChar"])
        # alloc vgpr
        tReg    = writer.vgprPool.checkOut(1,"tReg") # remainder
        kReg    = writer.vgprPool.checkOut(1,"kReg") # remainder
        tmpVgpr = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr")
        tmpVgprRes = ContinuousRegister(tmpVgpr, 2)

        module.add(self.LraTileAssignmentCode(writer, kernel, tP, tReg, kReg, tmpVgprRes))

        # release register
        tP["gpr"]["lro"] = tReg
        writer.vgprPool.checkIn(kReg)
        writer.vgprPool.checkIn(tmpVgpr)

        return module

    def LraTileAssignmentCode(self, writer, kernel, tP, tReg, kReg, tmpVgprRes, dividendReg="Serial", isDTVAB=False):
        module = Module("LraTileAssignmentCode")

        # alloc vgpr
        enableLDSTr = tP["enableLDSTr"]
        dummy   = writer.vgprPool.checkOut(1,"dummy")
        if enableLDSTr:
           sReg    = writer.vgprPool.checkOut(1,"sReg") # remainder
           mReg    = writer.vgprPool.checkOut(1,"mReg") # remainder

        isWmma_v1 = writer.states.asmCaps["HasWMMA_V1"]

        # get constant parameter
        tc               = tP["tensorChar"]
        tile01           = tP["tile01Idx"]
        waveWidth        = writer.states.kernel["WavefrontSize"]
        inputPerThread   = kernel["LocalReadVectorWidth"] if not writer.states.inTailLoop else kernel["MIInputPerThread%s"%tc]
        if kernel["ProblemType"]["Sparse"]:
          if (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and  tP["isA"]):
            inputPerThread = inputPerThread // 2
          elif tP["isM"]:
            inputPerThread = inputPerThread // 8
        LdsPad           = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0

        # parameter for get each type index
        dividendForKId   = kernel["MatrixInstM"] * kernel["MatrixInstB"]
        num1DBlocks      = kernel["MatrixInstBM"] if (tile01 == 0) else kernel["MatrixInstBN"]
        num1DWaves       = kernel["MIWaveGroup"][0] if (tile01 == 0) else kernel["MIWaveGroup"][1]
        if kernel["SourceSwap"]:
            dividedForBlkId  = kernel["MatrixInstM"] if (tile01 == 0) else (kernel["MatrixInstM"] * kernel["MatrixInstBM"])
        else:
            dividedForBlkId  = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (tile01 == 0) else kernel["MatrixInstN"]
        dividedForWaveId = waveWidth if (tile01 == 0) else (waveWidth * kernel["MIWaveGroup"][0])
        vectorWidth      = kernel["VectorWidth%s"%tc]
        if isDTVAB:
            if tP["tlu"]:
                # DTV + TLU case, glvw and vw are applied to the same direction. No need to apply both.
                # non TLU case, glvw and vw are applied to the different direction. We need to apply vw here.
                vectorWidth = 1
        maxKId = waveWidth // ((kernel["MatrixInstM"] if (tile01 == 0) else kernel["MatrixInstN"]) * kernel["MatrixInstB"])
        writer.states.lraTileProperties[tile01] = LraTilePropertiesMFMA(dividendForKId=dividendForKId, \
                                                                        num1DBlocks=num1DBlocks, \
                                                                        num1DWaves=num1DWaves, \
                                                                        dividedForBlkId=dividedForBlkId, \
                                                                        dividedForWaveId = dividedForWaveId, \
                                                                        vectorWidth=vectorWidth, \
                                                                        maxKId=maxKId)

        # strider for each type of index
        umlds            = kernel["UnrollMajorLDS%s" % tc]
        mt               = kernel["MacroTile%u" % tile01]
        if enableLDSTr:
           strideTile = 4
        else:
           strideTile       = kernel["_DepthU%s"%tc] + LdsPad if umlds else 1
        if isDTVAB:
          strideTile  = 1 # DTV case. Actual stride will be applied later.

        strideK          = inputPerThread if umlds else (mt + LdsPad) * inputPerThread
        if enableLDSTr:
           strideK1 = mt+LdsPad

        # FIXME SPARSE
        if kernel["ProblemType"]["Sparse"] != 0:
            if kernel["MIInputPerThread"] * kernel["ProblemType"]["DataType"].numBytes() > 16:
              isSparseTrack = (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]) or tP["isM"]
              strideK      = (inputPerThread if umlds else (mt + LdsPad) * inputPerThread) * (2 if isSparseTrack and kernel["MIInputPerThread%s"%tc] >  inputPerThread else 1)
        #special case for new F8 MFMA
        elif  kernel["ProblemType"]["DataType"].is8bitFloat() and kernel["MatrixInstK"] > 32:
            if umlds:
                strideK = 16
            else:
                strideK = (mt + LdsPad) * 16
        elif kernel["UseF32XEmulation"] and not (kernel["MatrixInstM"] == 16 and kernel["MatrixInstK"] == 16):
            if umlds:
                strideK = 4
            else:
                strideK = (mt + LdsPad) * 4

        strideBlock      = kernel["MatrixInstM"] * strideTile
        if enableLDSTr:
           strideWave = kernel["MatrixInstM"] * vectorWidth
        else:
           strideWave       = kernel["MatrixInstM"] * num1DBlocks * strideTile * vectorWidth

        lsu              = kernel["LocalSplitU"]

        if isDTVAB:
          strideTile  = 1 # DTV case. Actual stride will be applied later.


        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            # tile offset
            module.add(vectorStaticRemainder(dummy, kReg, dividendReg, waveWidth, tmpVgprRes, tmpSgprInfo, \
                "0. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth))
            if enableLDSTr:
               module.add(vectorStaticRemainder(dummy, tReg, kReg, 4, tmpVgprRes, tmpSgprInfo, \
                                                "1. N offset: nIdx = wtid %% 4"))
               module.add(vectorStaticRemainder(dummy, sReg, kReg, dividendForKId, tmpVgprRes, tmpSgprInfo, \
                                                "1. N offset: nIdx = wtid %% MI_M(%d)"%dividendForKId))
               module.add(vectorStaticDivide(sReg, sReg, 16, tmpVgprRes, \
                                                "1. thread id in wave: k1Idx = mtid // 4"))
               module.add(vectorStaticMultiply(vgpr(sReg), vgpr(sReg), 16, tmpSgprInfo, \
                                         "1. K1 offset: lrK1Offset = k1Idx * mStride(%u)" % (strideK1)))

            else:
               module.add(vectorStaticRemainder(dummy, tReg, kReg, kernel["MatrixInstN"], tmpVgprRes, tmpSgprInfo, \
                                             "1. N offset: nIdx = wtid %% MI_N(%u)" % kernel["MatrixInstN"]))
            module.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), strideTile, tmpSgprInfo, \
                "1. N offset: nOffset = nIdx * nStride(%u)" % strideTile))
            if enableLDSTr:
                module.add(VAddU32(dst=vgpr(tReg), src0=vgpr(sReg), src1=vgpr(tReg), \
                           comment="1. offset in wave: lrOffset = bnOffset + lrKOffset"))
            # block offset
            if num1DBlocks > 1:
                module.add(vectorStaticDivide(dummy, kReg, dividedForBlkId, tmpVgprRes, \
                    "2. block offset: bnIdx = wtid / dividedForBlkId(%u)" % dividedForBlkId))
                module.add(vectorStaticRemainder(dummy, dummy, dummy, num1DBlocks, tmpVgprRes, tmpSgprInfo, \
                    "2. block offset: bnIdx = bnIdx %% num1DBlocks(%u)" % num1DBlocks))
                module.add(vectorStaticMultiplyAdd(vgpr(tReg), vgpr(dummy), strideBlock, vgpr(tReg), tmpSgprInfo, \
                    "2. block offset: bnOffset = bnIdx * strideBlock(%u); 3. add N and block offset: bnOffset = block and N offset" % strideBlock))
            else:
                module.addComment0("Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1")

            module.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), vectorWidth, tmpSgprInfo, \
                "4. apply VectorWidth: bnOffset = bnOffset * vw(%u)" % vectorWidth))

            # unroll offset
            #if isMfma and (dividendForKId != waveWidth):
            if not isWmma_v1:
                if (dividendForKId != waveWidth) and (not isDTVAB):
                    if enableLDSTr:
                        module.add(vectorStaticRemainder(dummy, mReg, kReg, 16, tmpVgprRes, tmpSgprInfo, \
                                                        "5.1 thread id in wave: mtid = wtid %% 16"))
                        module.add(vectorStaticDivide(mReg, mReg, 4, tmpVgprRes, \
                                                     "5.2 thread id in wave: k1Idx = mtid // 4"))
                if (dividendForKId != waveWidth) or isDTVAB:
                  # DTVAB case, add this regardless of dividendForKId != waveWidth
                    module.add(vectorStaticDivide(kReg, kReg, dividendForKId, tmpVgprRes, \
                        "5. K offset: kIdx = wtid / (MIN(%u) * MIBB(%u))" % (kernel["MatrixInstN"], kernel["MatrixInstB"])))
                if (dividendForKId != waveWidth) and (not isDTVAB):
                    if enableLDSTr:
                        module.add(vectorStaticMultiply(vgpr(kReg), vgpr(kReg), strideK, tmpSgprInfo, \
                                                 "5. K offset: lrKOffset = kIdx * mStride(%u)" % (strideK)))
                        module.add(vectorStaticMultiply(vgpr(mReg), vgpr(mReg), strideK1, tmpSgprInfo, \
                                                 "5.1 K1 offset: lrK1Offset = k1Idx * mStride(%u)" % (strideK1)))
                        module.add(VAddU32(dst=vgpr(kReg), src0=vgpr(mReg), src1=vgpr(kReg), \
                                          comment="5.2 offset in wave: lrOffset = bnOffset + lrKOffset"))
                        module.add(VAddU32(dst=vgpr(tReg), src0=vgpr(kReg), src1=vgpr(tReg), \
                                          comment="6. offset in wave: lrOffset = bnOffset + lrKOffset"))
                    else:
                        module.add(vectorStaticMultiplyAdd(vgpr(tReg), vgpr(kReg), strideK, vgpr(tReg), tmpSgprInfo, \
                                                    "5. K offset: lrKOffset = kIdx * mStride(%u); 6. offset in wave: lrOffset = bnOffset + lrKOffset" % (strideK)))

            # wave offset
            if num1DWaves > 1:
                module.add(vectorStaticDivide(dummy, dividendReg, dividedForWaveId, tmpVgprRes, \
                    "7. wave offset in N dimen: wtid = tid / dividedForWaveId(%u)" % dividedForWaveId))
                module.add(vectorStaticRemainder(dummy, dummy, dummy, num1DWaves, tmpVgprRes, tmpSgprInfo, \
                    "7. wave offset in M dimen: wtid0 = wtid / num1DWaves(%u)" % num1DWaves))
                module.add(vectorStaticMultiplyAdd(vgpr(tReg), vgpr(dummy), strideWave, vgpr(tReg), tmpSgprInfo, \
                                             "7. wave offset in M dimen: wOffset = wtid0 * W0Stride(%u); 7. final local read offset: flrOffset = lrOffset + WOffset" % strideWave))

        # release register
        writer.vgprPool.checkIn(dummy)
        if enableLDSTr:
           writer.vgprPool.checkIn(sReg)
           writer.vgprPool.checkIn(mReg)

        return module
