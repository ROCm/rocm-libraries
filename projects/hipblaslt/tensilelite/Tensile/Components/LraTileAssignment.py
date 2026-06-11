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

from rocisa.code import Module, Label
from rocisa.container import sgpr, vgpr, ContinuousRegister
from rocisa.instruction import SMovB32, SMovB64, SNop, VAddU32, VAndB32, VMovB32, VLShiftLeftB32, VLShiftRightB32
from rocisa.functions import vectorStaticRemainder, \
    vectorStaticDivideAndRemainder, vectorStaticDivide, vectorStaticMultiply, \
    vectorStaticMultiplyAdd

from ..Component import LraTileAssignment, LraTileProperties
from ..Common import roundUp, log2, ceilDivide
from ..Common.DataType import DataType
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
        qReg    = writer.vgprPool.checkOut(1, tag="LraTileAssignmentVALU_qReg") # quotient
        rReg    = writer.vgprPool.checkOut(1, tag="LraTileAssignmentVALU_rReg") # remainder
        # dot2: currently only support unroll major LDS
        tc               = tP["tensorChar"]
        umlds            = kernel["UnrollMajorLDS%s" % tc]
        LdsPad           = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0
        strideTile       = kernel["_DepthU%s"%tc] + LdsPad if umlds else 1
        tmpVgpr          = writer.vgprPool.checkOutAligned(2,2, tag="LraTileAssignmentVALU_tmpVgpr")
        tmpVgprRes       = ContinuousRegister(tmpVgpr, 2)

        with writer.allocTmpSgpr(1, tag="LraTileAssignmentVALU_tmpSgprInfo") as tmpSgprInfo:
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

class LraTileAssignmentTransposedMFMA(LraTileAssignment):
    kernel = {"EnableMatrixInstruction": True,
              "DirectToVgprA": False,
              "DirectToVgprB": False,
              "ProblemType": {
                  "DataType": DataType("b")
              }}
    asmCaps = {
        "HasLDSTrB128B16": True
    }

    NUM_CONT_READ_ELEMENTS = 8
    NUM_READ_ELEMENT_PER_THREAD = 16
    NUM_UNROLLED_STRIDE_ELEMENTS = NUM_CONT_READ_ELEMENTS * 2

    def __call__(self, writer, kernel, tP):
        if not tP["enableLDSTr"]:
            comp = LraTileAssignmentMFMA()
            return comp(writer, kernel, tP)
        elif tP["isM"]:
            comp = LraTileAssignmentTransposedMFMAB8()
            return comp(writer, kernel, tP)

        dividendReg = "Serial"
        module = Module("LraTileAssignmentTransposedMFMA")
        module.addComment0("lr%s" % tP["tileChar"])
        # alloc vgpr
        tReg    = writer.vgprPool.checkOut(1,"tReg") # remainder
        kReg    = writer.vgprPool.checkOut(1,"kReg") # remainder
        tmpVgpr = writer.vgprPool.checkOutAligned(2, 2, tag="LraTileAssignmentTransposedMFMA_tmpVgpr")
        tmpVgprRes = ContinuousRegister(tmpVgpr, 2)

        # alloc vgpr
        dummy   = writer.vgprPool.checkOut(1, "dummy")
        sReg    = writer.vgprPool.checkOut(1, "sReg") # remainder
        mReg    = writer.vgprPool.checkOut(1, "mReg") # remainder

        # get constant parameter
        tc               = tP["tensorChar"]
        tile01           = tP["tile01Idx"]
        waveWidth        = writer.states.kernel["WavefrontSize"]
        #FIXME: tail loop with transposed load b128

        isSparseDenseMatrix = False
        if kernel["ProblemType"]["Sparse"]:
          if (kernel["ProblemType"]["Sparse"] == 1 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 2 and  tP["isA"]):
            isSparseDenseMatrix = True
        ldsPad           = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0

        # parameter for get each type index
        matrixInstT      = min(kernel["MatrixInstM"], kernel["MatrixInstN"])
        numTileInInst    = (kernel["MatrixInstM"] if (tile01 == 0) else kernel["MatrixInstN"]) // matrixInstT
        dividendForKId   = matrixInstT * kernel["MatrixInstB"]
        num1DBlocks      = kernel["MatrixInstBM"] if (tile01 == 0) else kernel["MatrixInstBN"]
        num1DWaves       = kernel["MIWaveGroup"][0] if (tile01 == 0) else kernel["MIWaveGroup"][1]
        if kernel["SourceSwap"]:
            dividedForBlkId  = matrixInstT if (tile01 == 0) else (matrixInstT * kernel["MatrixInstBM"])
        else:
            dividedForBlkId  = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (tile01 == 0) else kernel["MatrixInstN"]

        dividedForWaveId = waveWidth if (tile01 == 0) else (waveWidth * kernel["MIWaveGroup"][0])
        vectorWidth      = kernel["VectorWidth%s"%tc]
        maxKId = waveWidth // ((matrixInstT if (tile01 == 0) else kernel["MatrixInstN"]) * kernel["MatrixInstB"])
        writer.states.lraTileProperties[tile01] = LraTilePropertiesMFMA(dividendForKId=dividendForKId, \
                                                                        num1DBlocks=num1DBlocks, \
                                                                        num1DWaves=num1DWaves, \
                                                                        dividedForBlkId=dividedForBlkId, \
                                                                        dividedForWaveId = dividedForWaveId, \
                                                                        vectorWidth=vectorWidth, \
                                                                        maxKId=maxKId)

        # strider for each type of index
        mt           = kernel["MacroTile%u" % tile01]
        strideTile   = int(int(tP["localReadInstruction"].blockWidth * writer.states.bpr) // tP["bpeDS"])
        strideUnroll = mt + ldsPad
        strideWave   = numTileInInst * matrixInstT * vectorWidth

        with writer.allocTmpSgpr(1, tag="LraTileAssignmentTransposedMFMA_tmpSgprInfo") as tmpSgprInfo:
            # tile offset = (wtId%16)//8*8
            module.add(vectorStaticRemainder(dummy, kReg, dividendReg, waveWidth, tmpVgprRes, tmpSgprInfo, \
                "0. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth))
            #FIXME: calculate this
            module.add(vectorStaticRemainder(dummy, tReg, kReg, self.NUM_READ_ELEMENT_PER_THREAD, tmpVgprRes, tmpSgprInfo, "tileOffset=wtId%16"))
            module.add(vectorStaticDivide(tReg, tReg, self.NUM_CONT_READ_ELEMENTS, tmpVgprRes, f"tileOffset//={self.NUM_CONT_READ_ELEMENTS}"))
            module.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), strideTile, tmpSgprInfo, \
                "tileOffset*=strideTile(%u)" % strideTile))
            # block offset
            # removed, wmma has only single block

            #FIXME: should apply along k-dir?
            module.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), vectorWidth, tmpSgprInfo, \
                "4. apply VectorWidth: bnOffset = bnOffset * vw(%u)" % vectorWidth))

            # unroll offset
            module.add(vectorStaticRemainder(dummy, kReg, dividendReg, waveWidth, tmpVgprRes, tmpSgprInfo, "wtId=tid%wavelen"))
            module.add(vectorStaticDivide(kReg, kReg, self.NUM_UNROLLED_STRIDE_ELEMENTS, tmpSgprInfo, f"kOffset=wtId//{self.NUM_UNROLLED_STRIDE_ELEMENTS}"))
            if isSparseDenseMatrix:
                # Dense matrix of SPARSE are interleaved format, so we need to multiply by 2
                module.add(vectorStaticMultiply(vgpr(mReg), vgpr(kReg), self.NUM_CONT_READ_ELEMENTS * 2, tmpSgprInfo, f"kOffset*=2x{self.NUM_CONT_READ_ELEMENTS}"))
            else:
                module.add(vectorStaticMultiply(vgpr(mReg), vgpr(kReg), self.NUM_CONT_READ_ELEMENTS, tmpSgprInfo, f"kOffset*={self.NUM_CONT_READ_ELEMENTS}"))
            module.add(vectorStaticRemainder(dummy, kReg, dividendReg, waveWidth, tmpVgprRes, tmpSgprInfo, "wtId=tid%wavelen"))
            module.add(vectorStaticRemainder(dummy, kReg, kReg, self.NUM_CONT_READ_ELEMENTS, tmpVgprRes, tmpSgprInfo, f"ktOffset=wtid%{self.NUM_CONT_READ_ELEMENTS}"))
            module.add(VAddU32(vgpr(mReg), vgpr(mReg), vgpr(kReg), "kOffset+=ktOffset"))
            module.add(vectorStaticMultiply(vgpr(mReg), vgpr(mReg), strideUnroll, tmpSgprInfo, "kOffset*=stride"))
            module.add(VAddU32(vgpr(tReg), vgpr(tReg), vgpr(mReg), "lrOffset = kOffset + tileOffset"))

            # wave offset
            if num1DWaves > 1:
                module.add(vectorStaticDivide(dummy, dividendReg, dividedForWaveId, tmpVgprRes, \
                    "7. wave offset in N dimen: wtid = tid / dividedForWaveId(%u)" % dividedForWaveId))
                module.add(vectorStaticRemainder(dummy, dummy, dummy, num1DWaves, tmpVgprRes, tmpSgprInfo, \
                    "7. wave offset in M dimen: wtid0 = wtid / num1DWaves(%u)" % num1DWaves))
                module.add(vectorStaticMultiplyAdd(vgpr(tReg), vgpr(dummy), strideWave, vgpr(tReg), tmpSgprInfo, \
                                             "7. wave offset in M dimen: wOffset = wtid0 * W0Stride(%u); 7. final local read offset: flrOffset = lrOffset + WOffset" % strideWave))

        tP["gpr"]["lro"] = tReg
        # release register
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(sReg)
        writer.vgprPool.checkIn(mReg)
        writer.vgprPool.checkIn(kReg)
        writer.vgprPool.checkIn(tmpVgpr)

        return module

class LraTileAssignmentTransposedMFMAFP32(LraTileAssignmentTransposedMFMA):
    kernel = {"EnableMatrixInstruction": True,
              "DirectToVgprA": False,
              "DirectToVgprB": False,
              "ProblemType": {
                  "DataType": DataType("s")
              }}

class LraTileAssignmentTransposedMFMAFP16(LraTileAssignmentTransposedMFMA):
    kernel = {"EnableMatrixInstruction": True,
              "DirectToVgprA": False,
              "DirectToVgprB": False,
              "ProblemType": {
                  "DataType": DataType("h")
              }}
    asmCaps = {
        "HasLDSTrB128B16": True
    }

class LraTileAssignmentTransposedMFMAB8(LraTileAssignmentTransposedMFMA):
    kernel = {"EnableMatrixInstruction": True,
              "DirectToVgprA": False,
              "DirectToVgprB": False,
              "ProblemType": {
                  "DataType": DataType("I8")
              }}
    asmCaps = {
        "HasLDSTrB64B8": True
    }
    NUM_CONT_READ_ELEMENTS = 4
    NUM_READ_ELEMENT_PER_THREAD = 8
    NUM_UNROLLED_STRIDE_ELEMENTS = 16

    def __call__(self, writer, kernel, tP):
        if not tP["enableLDSTr"]:
            comp = LraTileAssignmentMFMA()
            return comp(writer, kernel, tP)

        dividendReg = "Serial"
        module = Module("LraTileAssignmentTransposedMFMA")
        module.addComment0("lr%s" % tP["tileChar"])
        # alloc vgpr
        tReg    = writer.vgprPool.checkOut(1,"tReg") # remainder
        kReg    = writer.vgprPool.checkOut(1,"kReg") # remainder
        tmpVgpr = writer.vgprPool.checkOutAligned(2, 2, tag="LraTileAssignmentTransposedMFMAB8_tmpVgpr")
        tmpVgprRes = ContinuousRegister(tmpVgpr, 2)

        # alloc vgpr
        dummy   = writer.vgprPool.checkOut(1, "dummy")
        sReg    = writer.vgprPool.checkOut(1, "sReg") # remainder
        mReg    = writer.vgprPool.checkOut(1, "mReg") # remainder

        # get constant parameter
        tc               = tP["tensorChar"]
        tile01           = tP["tile01Idx"]
        waveWidth        = writer.states.kernel["WavefrontSize"]
        #FIXME: tail loop with transposed load b128

        isSparseDenseMatrix = False
        if kernel["ProblemType"]["Sparse"]:
          if (kernel["ProblemType"]["Sparse"] == 1 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 2 and  tP["isA"]):
            isSparseDenseMatrix = True
        ldsPad           = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0

        # parameter for get each type index
        matrixInstT      = min(kernel["MatrixInstM"], kernel["MatrixInstN"])
        numTileInInst    = (kernel["MatrixInstM"] if (tile01 == 0) else kernel["MatrixInstN"]) // matrixInstT
        dividendForKId   = matrixInstT * kernel["MatrixInstB"]
        num1DBlocks      = kernel["MatrixInstBM"] if (tile01 == 0) else kernel["MatrixInstBN"]
        num1DWaves       = kernel["MIWaveGroup"][0] if (tile01 == 0) else kernel["MIWaveGroup"][1]
        if kernel["SourceSwap"]:
            dividedForBlkId  = matrixInstT if (tile01 == 0) else (matrixInstT * kernel["MatrixInstBM"])
        else:
            dividedForBlkId  = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (tile01 == 0) else kernel["MatrixInstN"]

        dividedForWaveId = waveWidth if (tile01 == 0) else (waveWidth * kernel["MIWaveGroup"][0])
        vectorWidth      = kernel["VectorWidth%s"%tc]
        maxKId = waveWidth // ((matrixInstT if (tile01 == 0) else kernel["MatrixInstN"]) * kernel["MatrixInstB"])
        writer.states.lraTileProperties[tile01] = LraTilePropertiesMFMA(dividendForKId=dividendForKId, \
                                                                        num1DBlocks=num1DBlocks, \
                                                                        num1DWaves=num1DWaves, \
                                                                        dividedForBlkId=dividedForBlkId, \
                                                                        dividedForWaveId = dividedForWaveId, \
                                                                        vectorWidth=vectorWidth, \
                                                                        maxKId=maxKId)

        # strider for each type of index
        mt           = kernel["MacroTile%u" % tile01]
        strideTile   = int(int(tP["localReadInstruction"].blockWidth * writer.states.bpr) // tP["bpeDS"])
        strideUnroll = mt + ldsPad
        strideWave   = numTileInInst * matrixInstT * vectorWidth

        with writer.allocTmpSgpr(1, tag="LraTileAssignmentTransposedMFMAB8_tmpSgprInfo") as tmpSgprInfo:
            # tile offset = (wtId%8)//4*8
            module.add(vectorStaticRemainder(dummy, kReg, dividendReg, waveWidth, tmpVgprRes, tmpSgprInfo, \
                "0. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth))
            #FIXME: calculate this
            module.add(vectorStaticRemainder(dummy, tReg, kReg, self.NUM_READ_ELEMENT_PER_THREAD, tmpVgprRes, tmpSgprInfo, "tileOffset=wtId%8"))
            module.add(vectorStaticDivide(tReg, tReg, self.NUM_CONT_READ_ELEMENTS, tmpSgprInfo, f"tileOffset//={self.NUM_CONT_READ_ELEMENTS}"))
            module.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), strideTile, tmpSgprInfo, \
                "tileOffset*=strideTile(%u)" % strideTile))
            # block offset
            # removed, wmma has only single block

            #FIXME: should apply along k-dir?
            module.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), vectorWidth, tmpSgprInfo, \
                "4. apply VectorWidth: bnOffset = bnOffset * vw(%u)" % vectorWidth))

            if tP["isM"]:
                # multiplyBy:
                # 2 : t8-t15 will read the same block as the t0-t7. t0-t7 reads 0-63
                # 1 : t8-t15 will read the next block. t0-t7 reads 0-63 and t8-t15 read 64-127
                # TODO Should have wider local read to let t8-t15 read the next iter's data and remove the muliplyBy=1 case.
                multiplyBy = 1 if kernel["MIInputPerThread%s"%tc] * tP["bpeDS"] // writer.states.bpr == 2 else 2
                strideK = self.NUM_UNROLLED_STRIDE_ELEMENTS // self.NUM_READ_ELEMENT_PER_THREAD * self.NUM_CONT_READ_ELEMENTS * multiplyBy
                module.add(vectorStaticRemainder(dummy, sReg, kReg, self.NUM_CONT_READ_ELEMENTS, tmpVgprRes, tmpSgprInfo, f"5.1 kOffset = wtId % {self.NUM_CONT_READ_ELEMENTS}"))
                module.add(vectorStaticDivide(mReg, kReg, strideK , tmpSgprInfo, f"5. blockOffset = wtId / strideK({strideK})"))

                module.add(vectorStaticMultiplyAdd(vgpr(kReg), vgpr(mReg), self.NUM_CONT_READ_ELEMENTS, vgpr(sReg), tmpSgprInfo,f"5.3 kOffset = kOffset + blockOffset * {self.NUM_CONT_READ_ELEMENTS}"))
                module.add(vectorStaticMultiply(vgpr(kReg), vgpr(kReg), strideUnroll, tmpSgprInfo, f"6.1 lrOffset = kOffset * strideUnroll({strideUnroll})"))
                module.add(VAddU32(vgpr(tReg), vgpr(tReg), vgpr(kReg), "6.2 lrOffset = lrOffset + tileOffset"))
            else:
                # unroll offset
                module.add(vectorStaticRemainder(dummy, kReg, dividendReg, waveWidth, tmpVgprRes, tmpSgprInfo, "wtId=tid%wavelen"))
                module.add(vectorStaticDivide(kReg, kReg, self.NUM_UNROLLED_STRIDE_ELEMENTS, tmpSgprInfo, f"kOffset=wtId//{self.NUM_UNROLLED_STRIDE_ELEMENTS}"))
                if isSparseDenseMatrix:
                    # Dense matrix of SPARSE are interleaved format, so we need to multiply by 2
                    module.add(vectorStaticMultiply(vgpr(mReg), vgpr(kReg), self.NUM_UNROLLED_STRIDE_ELEMENTS*2, tmpSgprInfo, f"kOffset*={self.NUM_UNROLLED_STRIDE_ELEMENTS}x2"))
                else:
                    module.add(vectorStaticMultiply(vgpr(mReg), vgpr(kReg), self.NUM_UNROLLED_STRIDE_ELEMENTS, tmpSgprInfo, f"kOffset*={self.NUM_UNROLLED_STRIDE_ELEMENTS}"))
                module.add(vectorStaticRemainder(dummy, kReg, dividendReg, waveWidth, tmpVgprRes, tmpSgprInfo, "wtId=tid%wavelen"))
                module.add(vectorStaticRemainder(dummy, kReg, dividendReg, self.NUM_UNROLLED_STRIDE_ELEMENTS, tmpVgprRes, tmpSgprInfo, "ktOffset=wtId%16"))
                module.add(vectorStaticDivide(kReg, kReg, self.NUM_READ_ELEMENT_PER_THREAD, tmpSgprInfo, f"kOffset=ktOffset//{self.NUM_READ_ELEMENT_PER_THREAD}"))
                module.add(vectorStaticMultiply(vgpr(kReg), vgpr(kReg), self.NUM_CONT_READ_ELEMENTS, tmpSgprInfo, f"ktOffset*={self.NUM_CONT_READ_ELEMENTS}"))
                module.add(VAddU32(vgpr(mReg), vgpr(mReg), vgpr(kReg), "kOffset+=ktOffset"))
                module.add(vectorStaticRemainder(dummy, kReg, dividendReg, waveWidth, tmpVgprRes, tmpSgprInfo, "wtId=tid%wavelen"))
                module.add(vectorStaticRemainder(dummy, kReg, kReg, self.NUM_CONT_READ_ELEMENTS, tmpVgprRes, tmpSgprInfo, f"ktOffset=wtid%{self.NUM_CONT_READ_ELEMENTS}"))
                module.add(VAddU32(vgpr(mReg), vgpr(mReg), vgpr(kReg), "kOffset+=ktOffset"))
                module.add(vectorStaticMultiply(vgpr(mReg), vgpr(mReg), strideUnroll, tmpSgprInfo, "kOffset*=stride"))
                module.add(VAddU32(vgpr(tReg), vgpr(tReg), vgpr(mReg), "lrOffset = kOffset + tileOffset"))

            # wave offset
            if num1DWaves > 1:
                module.add(vectorStaticDivide(dummy, dividendReg, dividedForWaveId, tmpVgprRes, \
                    "7. wave offset in N dimen: wtid = tid / dividedForWaveId(%u)" % dividedForWaveId))
                module.add(vectorStaticRemainder(dummy, dummy, dummy, num1DWaves, tmpVgprRes, tmpSgprInfo, \
                    "7. wave offset in M dimen: wtid0 = wtid / num1DWaves(%u)" % num1DWaves))
                module.add(vectorStaticMultiplyAdd(vgpr(tReg), vgpr(dummy), strideWave, vgpr(tReg), tmpSgprInfo, \
                                             "7. wave offset in M dimen: wOffset = wtid0 * W0Stride(%u); 7. final local read offset: flrOffset = lrOffset + WOffset" % strideWave))

        tP["gpr"]["lro"] = tReg
        # release register
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(sReg)
        writer.vgprPool.checkIn(mReg)
        writer.vgprPool.checkIn(kReg)
        writer.vgprPool.checkIn(tmpVgpr)

        return module

class LraTileAssignmentTransposedMFMA_FP8(LraTileAssignmentTransposedMFMAB8):
    kernel = {"EnableMatrixInstruction": True,
              "DirectToVgprA": False,
              "DirectToVgprB": False,
              "ProblemType": {
                  "DataType": DataType("F8"),
                  "isMixMode": False,
              }}
    asmCaps = {
        "HasLDSTrB64B8": True
    }

class LraTileAssignmentTransposedMFMA_BF8(LraTileAssignmentTransposedMFMA_FP8):
    kernel = {"EnableMatrixInstruction": True,
              "DirectToVgprA": False,
              "DirectToVgprB": False,
              "ProblemType": {
                  "DataType": DataType("B8"),
                  "isMixMode": False,
              }}
    asmCaps = {
        "HasLDSTrB64B8": True
    }

class LraTileAssignmentTransposedMFMA_FP8BF8(LraTileAssignmentTransposedMFMA_FP8):
    kernel = {"EnableMatrixInstruction": True,
              "DirectToVgprA": False,
              "DirectToVgprB": False,
              "ProblemType": {
                  "DataType": DataType("F8B8"),
                  "isMixMode": False,
              }}
    asmCaps = {
        "HasLDSTrB64B8": True
    }

class LraTileAssignmentTransposedMFMA_BF8FP8(LraTileAssignmentTransposedMFMA_FP8):
    kernel = {"EnableMatrixInstruction": True,
              "DirectToVgprA": False,
              "DirectToVgprB": False,
              "ProblemType": {
                  "DataType": DataType("B8F8"),
                  "isMixMode": False,
              }}
    asmCaps = {
        "HasLDSTrB64B8": True
    }
      
class LraTileAssignmentTransposedMFMAMixMode(LraTileAssignmentTransposedMFMAB8):
    kernel = {"EnableMatrixInstruction": True,
              "DirectToVgprA": False,
              "DirectToVgprB": False,
              "ProblemType": {
                  "isMixMode": True,
              }}
    asmCaps = {
        "HasLDSTrB64B8": True,
        "HasLDSTrB64B4": True,
        "HasLDSTrB96B6": True,
    }
    def __call__(self, writer, kernel, tP):
        # TODO: check correctness of this condition
        tc = tP["tensorChar"]
        MacDataType = f"MacDataType{tc}" if(tc=="A" or tc=="B") else "DataType"
        if not tP["enableLDSTr"]:
            comp = LraTileAssignmentMFMA()
            return comp(writer, kernel, tP)
        if kernel["ProblemType"][MacDataType].numBytes() == 0.5:
            comp = LraTileAssignmentTransposedMFMAF4()
            return comp(writer, kernel, tP)
        if kernel["ProblemType"][MacDataType].numBytes() == 0.75:
            comp = LraTileAssignmentTransposedMFMAF6()
            return comp(writer, kernel, tP)
        if kernel["ProblemType"][MacDataType].numBytes() == 1:
            comp = LraTileAssignmentTransposedMFMAB8()
            return comp(writer, kernel, tP)
        comp = LraTileAssignmentTransposedMFMAB8()
        return comp(writer, kernel, tP)

class LraTileAssignmentTransposedMFMAF4(LraTileAssignmentTransposedMFMA):
    kernel = {"EnableMatrixInstruction": True,
              "DirectToVgprA": False,
              "DirectToVgprB": False,
              "ProblemType": {
                  "DataType": DataType("F4"),
                  "isMixMode": False,
              }}
    asmCaps = {
        "HasLDSTrB64B4": True
    }

    NUM_CONT_READ_ELEMENTS = 8
    NUM_READ_ELEMENT_PER_THREAD = 16
    NUM_UNROLLED_STRIDE_ELEMENTS = 32

    def __call__(self, writer, kernel, tP):
        if not tP["enableLDSTr"]:
            comp = LraTileAssignmentMFMA()
            return comp(writer, kernel, tP)

        dividendReg = "Serial"
        module = Module("LraTileAssignmentTransposedMFMA")
        module.addComment0("lr%s" % tP["tileChar"])
        # alloc vgpr
        tReg    = writer.vgprPool.checkOut(1,"tReg") # remainder
        kReg    = writer.vgprPool.checkOut(1,"kReg") # remainder
        tmpVgpr = writer.vgprPool.checkOutAligned(2, 2, tag="LraTileAssignmentTransposedMFMAF4_tmpVgpr")
        tmpVgprRes = ContinuousRegister(tmpVgpr, 2)

        # alloc vgpr
        dummy   = writer.vgprPool.checkOut(1, "dummy")
        sReg    = writer.vgprPool.checkOut(1, "sReg") # remainder
        mReg    = writer.vgprPool.checkOut(1, "mReg") # remainder

        # get constant parameter
        tc               = tP["tensorChar"]
        tile01           = tP["tile01Idx"]
        waveWidth        = writer.states.kernel["WavefrontSize"]
        #FIXME: tail loop with transposed load b128

        ldsPad           = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0

        # parameter for get each type index
        matrixInstT      = (kernel["MatrixInstM"] if (tile01 == 0) else kernel["MatrixInstN"])
        matrixInstTO     = min(kernel["MatrixInstM"], kernel["MatrixInstN"])
        matrixInstTO     = matrixInstT if ("MXS" in tc) else matrixInstTO
        numTileInInst    = matrixInstT // matrixInstTO

        dividendForKId   = kernel["MatrixInstM"] * kernel["MatrixInstB"]
        num1DBlocks      = kernel["MatrixInstBM"] if (tile01 == 0) else kernel["MatrixInstBN"]
        num1DWaves       = kernel["MIWaveGroup"][0] if (tile01 == 0) else kernel["MIWaveGroup"][1]
        if kernel["SourceSwap"]:
            dividedForBlkId  = matrixInstTO if (tile01 == 0) else (matrixInstTO * kernel["MatrixInstBM"])
        else:
            dividedForBlkId  = (matrixInstTO * kernel["MatrixInstBN"]) if (tile01 == 0) else matrixInstTO
        dividedForWaveId = waveWidth if (tile01 == 0) else (waveWidth * kernel["MIWaveGroup"][0])
        vectorWidth      = kernel["VectorWidth%s"%tc]
        maxKId = waveWidth // (matrixInstTO * kernel["MatrixInstB"])
        writer.states.lraTileProperties[tile01] = LraTilePropertiesMFMA(dividendForKId=dividendForKId, \
                                                                        num1DBlocks=num1DBlocks, \
                                                                        num1DWaves=num1DWaves, \
                                                                        dividedForBlkId=dividedForBlkId, \
                                                                        dividedForWaveId = dividedForWaveId, \
                                                                        vectorWidth=vectorWidth, \
                                                                        maxKId=maxKId)

        # strider for each type of index
        mt           = kernel["MacroTile%u" % tile01]
        strideUnroll = mt + ldsPad
        strideWave   = matrixInstT * vectorWidth

        with writer.allocTmpSgpr(1, tag="LraTileAssignmentTransposedMFMAF4_tmpSgprInfo") as tmpSgprInfo:
            module.add(vectorStaticRemainder(dummy, kReg, dividendReg, waveWidth, tmpVgprRes, tmpSgprInfo, "wtId=tid%wavelen"))
            # calc col index
            module.add(vectorStaticDivide(sReg, kReg, self.NUM_READ_ELEMENT_PER_THREAD, tmpVgprRes, f"s=wtid//{self.NUM_READ_ELEMENT_PER_THREAD}"))
            module.add(vectorStaticMultiply(vgpr(sReg), vgpr(sReg), self.NUM_CONT_READ_ELEMENTS, tmpSgprInfo, comment=f"s*={self.NUM_CONT_READ_ELEMENTS}"))
            module.add(vectorStaticRemainder(dummy, mReg, kReg, self.NUM_READ_ELEMENT_PER_THREAD, tmpVgprRes, tmpSgprInfo, comment=f"m=wtid%{self.NUM_READ_ELEMENT_PER_THREAD}"))
            module.add(vectorStaticDivide(mReg, mReg, self.NUM_CONT_READ_ELEMENTS, tmpVgprRes, comment=f"m/={self.NUM_CONT_READ_ELEMENTS}"))
            module.add(vectorStaticMultiply(vgpr(mReg), vgpr(mReg), self.NUM_UNROLLED_STRIDE_ELEMENTS, tmpSgprInfo, comment=f"m*={self.NUM_UNROLLED_STRIDE_ELEMENTS}"))
            module.add(VAddU32(vgpr(tReg), vgpr(sReg), vgpr(mReg), "t=s+m"))
            module.add(vectorStaticRemainder(dummy, kReg, kReg, self.NUM_CONT_READ_ELEMENTS, tmpVgprRes, tmpSgprInfo, comment=f"k=k%{self.NUM_CONT_READ_ELEMENTS}"))
            module.add(VAddU32(vgpr(tReg), vgpr(tReg), vgpr(kReg), "t+=k"))
            module.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), strideUnroll, tmpSgprInfo, f"t*=strideUnroll({strideUnroll})"))

            # wave offset
            if num1DWaves > 1:
                module.add(vectorStaticDivide(dummy, dividendReg, dividedForWaveId, tmpVgprRes, \
                    "7. wave offset in N dimen: wtid = tid / dividedForWaveId(%u)" % dividedForWaveId))
                module.add(vectorStaticRemainder(dummy, dummy, dummy, num1DWaves, tmpVgprRes, tmpSgprInfo, \
                    "7. wave offset in M dimen: wtid0 = wtid / num1DWaves(%u)" % num1DWaves))
                module.add(vectorStaticMultiplyAdd(vgpr(tReg), vgpr(dummy), strideWave, vgpr(tReg), tmpSgprInfo, \
                                             "7. wave offset in M dimen: wOffset = wtid0 * W0Stride(%u); 7. final local read offset: flrOffset = lrOffset + WOffset" % strideWave))

        tP["gpr"]["lro"] = tReg
        # release register
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(sReg)
        writer.vgprPool.checkIn(mReg)
        writer.vgprPool.checkIn(kReg)
        writer.vgprPool.checkIn(tmpVgpr)

        return module

class LraTileAssignmentTransposedMFMAF6(LraTileAssignmentTransposedMFMA):
    kernel = {"EnableMatrixInstruction": True,
              "DirectToVgprA": False,
              "DirectToVgprB": False,
              "ProblemType": {
                  "DataType": DataType("F6"),
                  "isMixMode": False,
              }}
    asmCaps = {
        "HasLDSTrB96B6": True
    }

    NUM_CONT_READ_ELEMENTS = 8
    NUM_READ_ELEMENT_PER_THREAD = 4
    NUM_UNROLLED_STRIDE_ELEMENTS = 32

    def __call__(self, writer, kernel, tP):
        if not tP["enableLDSTr"]:
            comp = LraTileAssignmentMFMA()
            return comp(writer, kernel, tP)

        dividendReg = "Serial"
        module = Module("LraTileAssignmentTransposedMFMA")
        module.addComment0("lr%s" % tP["tileChar"])
        # alloc vgpr
        tReg    = writer.vgprPool.checkOut(1,"tReg") # remainder
        kReg    = writer.vgprPool.checkOut(1,"kReg") # remainder
        tmpVgpr = writer.vgprPool.checkOutAligned(2, 2, tag="LraTileAssignmentTransposedMFMAF6_tmpVgpr")
        tmpVgprRes = ContinuousRegister(tmpVgpr, 2)

        # alloc vgpr
        dummy   = writer.vgprPool.checkOut(1, "dummy")
        sReg    = writer.vgprPool.checkOut(1, "sReg") # remainder
        mReg    = writer.vgprPool.checkOut(1, "mReg") # remainder

        # get constant parameter
        tc               = tP["tensorChar"]
        tile01           = tP["tile01Idx"]
        waveWidth        = writer.states.kernel["WavefrontSize"]
        #FIXME: tail loop with transposed load b128

        ldsPad           = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0

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
        maxKId = waveWidth // ((kernel["MatrixInstM"] if (tile01 == 0) else kernel["MatrixInstN"]) * kernel["MatrixInstB"])
        writer.states.lraTileProperties[tile01] = LraTilePropertiesMFMA(dividendForKId=dividendForKId, \
                                                                        num1DBlocks=num1DBlocks, \
                                                                        num1DWaves=num1DWaves, \
                                                                        dividedForBlkId=dividedForBlkId, \
                                                                        dividedForWaveId = dividedForWaveId, \
                                                                        vectorWidth=vectorWidth, \
                                                                        maxKId=maxKId)

        # strider for each type of index
        mt           = kernel["MacroTile%u" % tile01]
        strideUnroll = mt + ldsPad
        strideWave   = kernel["MatrixInstM"] * vectorWidth

        with writer.allocTmpSgpr(1, tag="LraTileAssignmentTransposedMFMAF6_tmpSgprInfo") as tmpSgprInfo:
            module.add(vectorStaticRemainder(dummy, kReg, dividendReg, waveWidth, tmpVgprRes, tmpSgprInfo, "wtId(k)=tid%wavelen"))
            # calc col index
            # col = (wtId % 8) // 4 * 32 + (wtId // 8) * 4 + wtId % 4
            module.add(vectorStaticRemainder(dummy, sReg, kReg, self.NUM_CONT_READ_ELEMENTS, tmpVgprRes, tmpSgprInfo, comment=f"s=wtId%{self.NUM_CONT_READ_ELEMENTS}"))
            module.add(vectorStaticDivide(sReg, sReg, self.NUM_READ_ELEMENT_PER_THREAD, tmpVgprRes, f"s//={self.NUM_READ_ELEMENT_PER_THREAD}"))
            module.add(vectorStaticMultiply(vgpr(tReg), vgpr(sReg), self.NUM_UNROLLED_STRIDE_ELEMENTS, tmpSgprInfo, comment=f"t=s*{self.NUM_UNROLLED_STRIDE_ELEMENTS}"))

            module.add(vectorStaticDivide(sReg, kReg, self.NUM_CONT_READ_ELEMENTS, tmpVgprRes, comment=f"s=k//{self.NUM_CONT_READ_ELEMENTS}"))
            module.add(vectorStaticMultiply(vgpr(sReg), vgpr(sReg), self.NUM_READ_ELEMENT_PER_THREAD, tmpSgprInfo, comment=f"s*={self.NUM_READ_ELEMENT_PER_THREAD}"))
            module.add(VAddU32(vgpr(tReg), vgpr(tReg), vgpr(sReg), "t+=s"))
            module.add(vectorStaticRemainder(dummy, kReg, kReg, self.NUM_READ_ELEMENT_PER_THREAD, tmpVgprRes, tmpSgprInfo, comment=f"k=k%{self.NUM_READ_ELEMENT_PER_THREAD}"))
            module.add(VAddU32(vgpr(tReg), vgpr(tReg), vgpr(kReg), "t+=k"))
            module.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), strideUnroll, tmpSgprInfo, f"t*=strideUnroll({strideUnroll})"))

            # wave offset
            if num1DWaves > 1:
                module.add(vectorStaticDivide(dummy, dividendReg, dividedForWaveId, tmpVgprRes, \
                    "7. wave offset in N dimen: wtid = tid / dividedForWaveId(%u)" % dividedForWaveId))
                module.add(vectorStaticRemainder(dummy, dummy, dummy, num1DWaves, tmpVgprRes, tmpSgprInfo, \
                    "7. wave offset in M dimen: wtid0 = wtid / num1DWaves(%u)" % num1DWaves))
                module.add(vectorStaticMultiplyAdd(vgpr(tReg), vgpr(dummy), strideWave, vgpr(tReg), tmpSgprInfo, \
                                             "7. wave offset in M dimen: wOffset = wtid0 * W0Stride(%u); 7. final local read offset: flrOffset = lrOffset + WOffset" % strideWave))

        tP["gpr"]["lro"] = tReg
        # release register
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(sReg)
        writer.vgprPool.checkIn(mReg)
        writer.vgprPool.checkIn(kReg)
        writer.vgprPool.checkIn(tmpVgpr)

        return module

class LraTileAssignmentTransposedMFMAB6(LraTileAssignmentTransposedMFMAF6):
    kernel = {"EnableMatrixInstruction": True,
              "DirectToVgprA": False,
              "DirectToVgprB": False,
              "ProblemType": {
                  "DataType": DataType("B6"),
                  "isMixMode": False,
              }}
    asmCaps = {
        "HasLDSTrB96B6": True
    }

class LraTileAssignmentMFMA(LraTileAssignment):
    kernel = {"EnableMatrixInstruction": True, }
    asmCaps = {
        "HasLDSTrB128B16": False
    }

    """
    Local Read Addresses: Tile Assignment A/B
    """
    def __call__(self, writer, kernel, tP):
        module = Module("LraTileAssignmentMFMA")
        module.addComment0("lr%s" % tP["tileChar"])
        # alloc vgpr
        tReg    = writer.vgprPool.checkOut(1,"tReg") # remainder
        kReg    = writer.vgprPool.checkOut(1,"kReg") # remainder
        tmpVgpr = writer.vgprPool.checkOutAligned(2,2, tag="LraTileAssignmentMFMA_tmpVgpr")
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
        dummy = writer.vgprPool.checkOut(1,"dummy")
        mReg  = writer.vgprPool.checkOut(1,"mReg") # remainder
        if enableLDSTr:
           sReg = writer.vgprPool.checkOut(1,"sReg") # remainder

        # get constant parameter
        tc               = tP["tensorChar"]
        tile01           = tP["tile01Idx"]
        waveWidth        = writer.states.kernel["WavefrontSize"]

        noUnrollOffset = writer.states.asmCaps["HasWMMA_V1"] or ("MXS" in tc)
        isgfx950 = kernel["ISA"][:2] == (9, 5)
        isgfx950mx = isgfx950 and ("MXS" in tc)
        # workaround for gfx950
        # force noUnrollOffset=False for MX
        if isgfx950:
            noUnrollOffset = False

        lrvw             = kernel["LocalReadVectorWidthMXS"] if ("MXS" in tc) else kernel[f"LocalReadVectorWidth{tc}"]
        inputPerThread   = lrvw if not writer.states.inTailLoop else kernel["MIInputPerThread%s"%tc]
        # workaround for gfx950 + fp4
        # use MIInputPerThread if lrvw < MIInputPerThread
        if isgfx950 and tP["bpeDS"] == 0.5 and lrvw < kernel["MIInputPerThread%s"%tc]:
            inputPerThread = kernel["MIInputPerThread%s"%tc]
        LdsPad           = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0

        # parameter for get each type index
        matrixInstT      = (kernel["MatrixInstM"] if (tile01 == 0) else kernel["MatrixInstN"])
        matrixInstTO     = min(kernel["MatrixInstM"], kernel["MatrixInstN"])
        matrixInstTO     = matrixInstT if ("MXS" in tc) else matrixInstTO
        numTileInInst    = matrixInstT // matrixInstTO

        dividendForKId   = matrixInstTO * kernel["MatrixInstB"]
        num1DBlocks      = kernel["MatrixInstBM"] if (tile01 == 0) else kernel["MatrixInstBN"]
        num1DWaves       = kernel["MIWaveGroup"][0] if (tile01 == 0) else kernel["MIWaveGroup"][1]
        if kernel["SourceSwap"]:
            dividedForBlkId  = matrixInstTO if (tile01 == 0) else (matrixInstTO * kernel["MatrixInstBM"])
        else:
            dividedForBlkId  = (matrixInstTO * kernel["MatrixInstBN"]) if (tile01 == 0) else matrixInstTO
        dividedForWaveId = waveWidth if (tile01 == 0) else (waveWidth * kernel["MIWaveGroup"][0])
        vectorWidth      = kernel["VectorWidth%s"%tc]
        if isDTVAB:
            if tP["tlu"]:
                # DTV + TLU case, glvw and vw are applied to the same direction. No need to apply both.
                # non TLU case, glvw and vw are applied to the different direction. We need to apply vw here.
                vectorWidth = 1
        maxKId = waveWidth // (matrixInstTO * kernel["MatrixInstB"])
        writer.states.lraTileProperties[tile01] = LraTilePropertiesMFMA(dividendForKId=dividendForKId, \
                                                                        num1DBlocks=num1DBlocks, \
                                                                        num1DWaves=num1DWaves, \
                                                                        dividedForBlkId=dividedForBlkId, \
                                                                        dividedForWaveId = dividedForWaveId, \
                                                                        vectorWidth=vectorWidth, \
                                                                        maxKId=maxKId)

        abmatrixinfo = writer.states.a if tc == 'A' else writer.states.b
        perpStride = abmatrixinfo.gNLCPerpStride
        permBlock  = abmatrixinfo.gNLCPermBlock
        perpBlockSize  = abmatrixinfo.gRDtlSwizzlePerpBlockSize

        # strider for each type of index
        umlds            = kernel["UnrollMajorLDS%s" % tc]
        mt               = kernel["MacroTile%u" % tile01]
        if ("MXS" in tc):
           subTc = tc[3]
           # MX scale LDS tile-stride, gated by MXScaleFormat:
           #   - Swizzled (HostPreSwizzle/InMemorySwizzle): MatrixInstK/MXBlock (= MX-unit)
           #   - NoSwizzle (canonical):                      _DepthU_MXS (= K-scales per M)
           mxScaleFormat = kernel.get("MXScaleFormat", "NoSwizzle")
           isMxSwizzled  = mxScaleFormat in ("InMemorySwizzle", "HostPreSwizzle")
           if isMxSwizzled:
              strideTile = kernel["MatrixInstK"] // kernel["ProblemType"][f"MXBlock{subTc}"]
           else:
              strideTile = kernel["_DepthU%s"%tc] + LdsPad if umlds else 1
        elif enableLDSTr:
           strideTile = 4
        else:
           strideTile = kernel["_DepthU%s"%tc] + LdsPad if umlds else 1
        if isDTVAB:
          strideTile = 1 # DTV case. Actual stride will be applied later.

        strideK = inputPerThread if umlds else (mt + LdsPad) * inputPerThread
        if enableLDSTr:
           if kernel["UseGeneralizedNLCOne%s"%tc] and perpStride > 1:
              strideK  = 8
           strideK1 = mt+LdsPad
        elif kernel["ProblemType"]["Sparse"]:
            if tP["isM"]:
                strideK //= 4
        # special case for new F8 MFMA, need to exclude wmma_v3
        elif kernel["ProblemType"]["DataType"].is8bitFloat() and kernel["MatrixInstK"] > 32 and (not writer.states.asmCaps["HasWMMA_V3"]) and (not isgfx950mx):
            if umlds:
                strideK = 16
            else:
                strideK = (mt + LdsPad) * 16
        elif kernel["UseF32XEmulation"] and not (kernel["MatrixInstM"] == 16 and kernel["MatrixInstK"] == 16):
            if umlds:
                strideK = 4
            else:
                strideK = (mt + LdsPad) * 4
        # sparse
        if writer.states.asmCaps["HasSWMMAC"] and writer.states.asmCaps["HasSWMMAC_gfx1250"]:
            if (kernel["ProblemType"]["Sparse"] == 1 and tP["isB"]) or (kernel["ProblemType"]["Sparse"] == 2 and tP["isA"]) or tP["isM"]:
                strideK *= 2

        strideBlock = matrixInstT * strideTile
        if ("MXS" in tc):
           strideWave = matrixInstT * num1DBlocks * strideTile * vectorWidth
        elif enableLDSTr:
           strideWave = matrixInstT * vectorWidth
        else:
           strideWave = matrixInstT * num1DBlocks * strideTile * vectorWidth

        lsu              = kernel["LocalSplitU"]

        if isDTVAB:
          strideTile  = 1 # DTV case. Actual stride will be applied later.

        def perpPerm(vgprReg):
           reMap0 = writer.vgprPool.checkOut(1, tag="perpPerm_reMap0")
           reMap1 = writer.vgprPool.checkOut(1, tag="perpPerm_reMap1")
           perpStrideInv = permBlock // perpStride

           module.addComment0("Computing strided(%u) perp indicies"%perpStrideInv)
           module.add(VAndB32(dst=vgpr(reMap0), src0=(permBlock // perpStrideInv - 1), src1=vgpr(vgprReg), comment="r0 = I %% (%u // %u)"%(permBlock, perpStrideInv)))
           module.add(VLShiftLeftB32(dst=vgpr(reMap0), shiftHex=log2(perpStrideInv), src=vgpr(reMap0), comment="r0 = %u * r0"%(perpStrideInv)))
           module.addComment0("Computing r1 = (I %% %u) // (%u // %u)"%(permBlock, permBlock, perpStrideInv))
           module.add(VAndB32(dst=vgpr(reMap1), src0=(permBlock - 1), src1=vgpr(vgprReg), comment="r1 = I %% (%u)"%(permBlock)))
           module.add(VLShiftRightB32(dst=vgpr(reMap1), shiftHex=log2(permBlock // perpStrideInv), src=vgpr(reMap1), comment="r1 = (r1) // (%u // %u)"%(permBlock, perpStrideInv)))
           module.add(VAddU32(dst=vgpr(reMap0), src0=vgpr(reMap0), src1=vgpr(reMap1), comment="r0 = r0 + r1" ))

           module.add(VLShiftRightB32(dst=vgpr(reMap1), shiftHex=log2(permBlock), src=vgpr(vgprReg), comment="r1 = I // %u"%(permBlock)))
           module.add(vectorStaticMultiplyAdd(vgpr(vgprReg), vgpr(reMap1), permBlock, vgpr(reMap0), None))

           module.addComment0("Done computing strided(%u) perp indices"%perpStrideInv)
           writer.vgprPool.checkIn(reMap0)
           writer.vgprPool.checkIn(reMap1)

        with writer.allocTmpSgpr(1, tag="LraTileAssignmentMFMA_tmpSgprInfo") as tmpSgprInfo:

            if perpBlockSize > 0:
               rotVgpr = writer.vgprPool.checkOut(1, tag="perpPerm_rotVgpr") # remainder

            # tile offset
            module.add(vectorStaticRemainder(dummy, kReg, dividendReg, waveWidth, tmpVgprRes, tmpSgprInfo, \
                "0. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth))
            if enableLDSTr:
               module.add(vectorStaticRemainder(dummy, tReg, kReg, 4, tmpVgprRes, tmpSgprInfo, \
                                                "1. N offset: nIdx = wtid %% 4"))
               module.add(vectorStaticRemainder(dummy, sReg, kReg, dividendForKId, tmpVgprRes, tmpSgprInfo, \
                                                "1. N offset: nIdx = wtid %% MI_M(%d)"%dividendForKId))
               module.add(vectorStaticDivide(sReg, sReg, 16, tmpVgprRes, \
                                                "1. thread id in wave: k1Idx = mtid // 16"))
               module.add(vectorStaticMultiply(vgpr(sReg), vgpr(sReg), 16, tmpSgprInfo, \
                                         "1. K1 offset: lrK1Offset = k1Idx * mStride(%u)" % (strideK1)))

            else:
               module.add(vectorStaticRemainder(dummy, tReg, kReg, matrixInstTO, tmpVgprRes, tmpSgprInfo, \
                                             "1. N offset: nIdx = wtid %% MI_N(%u)" % matrixInstTO))

            applyVWCalcEarly = perpStride > 1 and kernel["ProblemType"]["TLU%s"%tc] == 0 and kernel["ProblemType"]["DataType"].numBytes() != 2
            if applyVWCalcEarly:
               # Apply vector width calc before we apply permutation to perp dim
               module.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), vectorWidth, tmpSgprInfo, \
                                               "1. apply VectorWidth: bnOffset = bnOffset * vw(%u)" % vectorWidth))
               perpPerm(tReg)

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

            if not applyVWCalcEarly:
                module.add(vectorStaticMultiply(vgpr(tReg), vgpr(tReg), vectorWidth, tmpSgprInfo, \
                                                "4. apply VectorWidth: bnOffset = bnOffset * vw(%u)" % vectorWidth))

            if perpBlockSize > 0:
               # Moved here since we need to wait until vectorwidth calculation is applied
               module.add(vectorStaticDivide(rotVgpr, tReg, perpBlockSize * strideTile, tmpVgprRes, \
                                             "Test rotating"))

            # unroll offset
            #if isMfma and (dividendForKId != waveWidth):
            if not noUnrollOffset:
                if (dividendForKId != waveWidth) and (not isDTVAB):
                    if enableLDSTr:
                        module.add(vectorStaticRemainder(dummy, mReg, kReg, 16, tmpVgprRes, tmpSgprInfo, \
                                                        "5.1 thread id in wave: mtid = wtid %% 16"))
                        module.add(vectorStaticDivide(mReg, mReg, 4, tmpVgprRes, \
                                                     "5.2 thread id in wave: k1Idx = mtid // 4"))
                if (dividendForKId != waveWidth) or isDTVAB:
                  # DTVAB case, add this regardless of dividendForKId != waveWidth
                    module.add(vectorStaticDivide(kReg, kReg, dividendForKId, tmpVgprRes, \
                        "5. K offset: kIdx = wtid / (MIN(%u) * MIBB(%u))" % (matrixInstTO, kernel["MatrixInstB"])))

                if perpBlockSize > 0:
                      module.add(VAddU32(dst=vgpr(kReg), src0=vgpr(kReg), src1=vgpr(rotVgpr), \
                                         comment="rotate"))
                      module.add(VAndB32(dst=vgpr(kReg), src0=(abmatrixinfo.gRDtlSwizzleParaBlockSize - 1), src1=vgpr(kReg), \
                                         comment="rotate: numThreadsCoalesced: %u"%(abmatrixinfo.gRDtlSwizzleParaBlockSize)))
                if (dividendForKId != waveWidth) and (not isDTVAB):

                    if enableLDSTr:
                        module.add(vectorStaticMultiply(vgpr(kReg), vgpr(kReg), strideK, tmpSgprInfo, \
                                                 "5. K offset: lrKOffset = kIdx * mStride(%u)" % (strideK)))

                        if perpStride == 1:
                           module.add(vectorStaticMultiply(vgpr(mReg), vgpr(mReg), strideK1, tmpSgprInfo, \
                                                           "5.1 K1 offset: lrK1Offset = k1Idx * mStride(%u)" % (strideK1)))
                           module.add(VAddU32(dst=vgpr(kReg), src0=vgpr(mReg), src1=vgpr(kReg), \
                                              comment="5.1 offset in wave: lrOffset = bnOffset + lrKOffset"))
                        else:
                           module.add(VAddU32(dst=vgpr(kReg), src0=vgpr(mReg), src1=vgpr(kReg), \
                                              comment="5.1 offset in wave: lrOffset = bnOffset + lrKOffset"))
                           # Apply permutation to perpendicular dim
                           if perpStride > 1:
                              perpPerm(kReg)
                           module.add(vectorStaticMultiply(vgpr(kReg), vgpr(kReg), strideK1, tmpSgprInfo, \
                                                           "5.2 K1 offset: lrK1Offset = k1Idx * mStride(%u)" % (strideK1)))
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
            if perpBlockSize > 0:
               writer.vgprPool.checkIn(rotVgpr)

        # release register
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(mReg)
        if enableLDSTr:
           writer.vgprPool.checkIn(sReg)

        return module
