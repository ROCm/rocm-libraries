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
from rocisa.container import ContinuousRegister, vgpr, sgpr
from rocisa.instruction import SMulI32, VAddLShiftLeftU32, VAddU32, VMovB32, VMulLOU32
from rocisa.functions import vectorStaticRemainder, vectorStaticDivideAndRemainder, vectorStaticDivide, vectorStaticMultiply

from ..Component import ComputeStoreVgprs
from ..Common import DataDirection, log2

class ComputeStoreVgprsVALU(ComputeStoreVgprs):
    kernel = {"EnableMatrixInstruction": False,
              "SourceSwap": False}

    """
    computeStoreVgprs
    Compute workitem/TT offsets in VGPRS
    and coord0/coord1
    tid0Scale specifies the number of output elements in 0/coalesced dim
    that should be written by each work-item in each batch element.
    """
    def __call__(self, writer, kernel, divisor, tid0Scale, tid1Scale):

        module = Module("ComputeStoreVgprsVALU")

        # tmpS0 = writer.getTmpSgpr(3).idx()
        # tmpS1 = tmpS0+1
        # wgMT1 = tmpS0+2

        # if writer.prefetchAcrossPersistent:
        #     wg0="PrevWorkGroup0"
        #     wg1="PrevWorkGroup1"
        # else:
        wg0="WorkGroup0"
        wg1="WorkGroup1"

        # tid0, tid1: element offsets from the start of macroTile in 0 and 1 direction
        # These will live for entire GlobalWrite loop - allocate before tmps
        # to avoid fragmentation
        tid0 = writer.vgprPool.checkOut(1, "tid0")
        tid1 = writer.vgprPool.checkOut(1, "tid1")

        packedC1 = kernel["PackedC1IndicesX"]

        if kernel["BufferStore"]:
            writer.vgprs.cinRowPtr  = writer.vgprPool.checkOut(1, "cinRowPtr")
            writer.vgprs.coutRowPtrD = writer.vgprPool.checkOut(1, "coutRowPtrD")

        with writer.allocTmpSgpr(3) as tmpSgprInfo:
            tmpS0 = tmpSgprInfo.idx
            tmpS1 = tmpS0+1
            wgMT1 = tmpS0+2
            tmpVgpr = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr")
            tmpVgprRes = ContinuousRegister(tmpVgpr, 2)
            # dot2: consecutive NumWaveSplitK threads compute the same element, divide it first before computing tile indices
            if kernel["NumWaveSplitK"] > 1:
                newSerial = writer.vgprPool.checkOut(1, "newSerial")
                module.add(vectorStaticDivide(newSerial, "Serial", kernel["NumWaveSplitK"], tmpVgprRes, comment="Divided by NumWaveSplitK"))
                module.add(vectorStaticDivideAndRemainder(tid1, tid0, newSerial, divisor, tmpVgprRes))
                writer.vgprPool.checkIn(newSerial)
            else:
                module.add(vectorStaticDivideAndRemainder(tid1, tid0, "Serial", divisor, tmpVgprRes))

            writer.vgprPool.checkIn(tmpVgpr)
            tmpS1Res = ContinuousRegister(tmpS1, 1)
            module.add(vectorStaticMultiply(vgpr(tid0), vgpr(tid0), tid0Scale, tmpS1Res))
            if tid1Scale != 1:
                module.add(vectorStaticMultiply(vgpr(tid1), vgpr(tid1), tid1Scale, tmpS1Res))

            if kernel["BufferStore"]:
                # compute rowStart- this is just tid1 scaled by appropriate stride.
                # rowPtr is offset from the beginning of the tile/SRD not the tensor base
                # when incremented, it moves in units of (col) Stride to get to a new row
                # it is used for address computation, not element range detection.
                # rowPtr is in the element space and must be scaled by bpe if bytes are required.
                # Do this before code below which overwries the tid1:
                # TODO-packed
                # Eventually need to modify if supporting packed coord1, to start just assert if that case is detected
                #--
                strideC1 = "StrideC%s" % (writer.states.indexChars[packedC1[0]])
                module.add(VMulLOU32(dst=vgpr(writer.vgprs.cinRowPtr), src0=vgpr(tid1), src1=sgpr(strideC1), comment="rowStart vgpr"))
                strideD1 = "StrideD%s" % (writer.states.indexChars[packedC1[0]])
                module.add(VMulLOU32(dst=vgpr(writer.vgprs.coutRowPtrD), src0=vgpr(tid1), src1=sgpr(strideD1), comment="rowStart vgpr"))
                module.addSpaceLine()

            # Compute coord0 and coord1
            # These are element offsets from the beginning of the tensor.
            # These are 'flattened' meaning they span packed tensor dims.
            # They need to be preserved so can use in comparisons against
            # product-of-packed sizes to determine OOB cases. (for Edge tiles only)
            module.add(SMulI32(dst=sgpr(tmpS0), src0=hex(kernel["MacroTile0"]), src1=sgpr(wg0), comment="%s = wg0*MT0"%sgpr(tmpS0)))

            # coord = tid*VW + workgroup offset
            module.add(VAddU32(dst=vgpr(tid0), src0=sgpr(tmpS0), src1=vgpr(tid0), comment="coord0 = tid0*VW + wg0*MT0"))
            module.add(SMulI32(dst=sgpr(wgMT1), src0=hex(kernel["MacroTile1"]), src1=sgpr(wg1), comment="<- wg1*MT1"))
            module.add(VAddU32(dst=vgpr(tid1), src0=sgpr(wgMT1), src1=vgpr(tid1), comment="coord1 = tid1*VW + wg1*MT1"))

            if len(packedC1) > 1:
                module.add(writer.extractPackedCoord1ToRowStart(kernel, packedC1, tid1, 'D'))

            writer.vgprs.coord0 = tid0
            writer.vgprs.coord1 = tid1

        return module

class ComputeStoreVgprsMFMA(ComputeStoreVgprs):
    kernel = {"EnableMatrixInstruction": True,
              "SourceSwap": False}

    """
    computeStoreVgprs
    Compute workitem/TT offsets in VGPRS
    and coord0/coord1
    tid0Scale specifies the number of output elements in 0/coalesced dim
    that should be written by each work-item in each batch element.
    """
    def __call__(self, writer, kernel):

        # writer.coord0
        # writer.coord1
        # writer.vgprs.cinRowPtr  : C buffer coulmn offset
        # writer.vgprs.coutRowPtrD : D buffer coulmn offset

        # alloc resources
        tid0 = writer.vgprPool.checkOut(1, "coord0")
        tid1 = writer.vgprPool.checkOut(1, "coord1")
        if kernel["BufferStore"]:
            writer.vgprs.cinRowPtr   = writer.vgprPool.checkOut(1, "cinRowPtr")
            writer.vgprs.coutRowPtrD = writer.vgprPool.checkOut(1, "coutRowPtrD")
            if kernel["ProblemType"]["UseE"]:
                writer.vgprs.coutRowPtrE = writer.vgprPool.checkOut(1, "coutRowPtrE")
            if writer.states.useBias == DataDirection.WRITE and (not kernel["WorkGroupReduction"]) and kernel["ProblemType"]["BiasSrc"] == "D":
                writer.vgprs.coutRowPtrBias = writer.vgprPool.checkOut(1, "coutRowPtrBias")
        if kernel["LocalSplitU"] > 1:
            writer.vgprs.coord0InMT = writer.vgprPool.checkOut(1, "coord0InMT")
            writer.vgprs.coord1InMT = writer.vgprPool.checkOut(1, "coord1InMT")
        else:
            writer.vgprs.coord0InMT = tid0
            writer.vgprs.coord1InMT = tid1
        lsuTid0 = writer.vgprs.coord0InMT
        lsuTid1 = writer.vgprs.coord1InMT

        wave_id = writer.vgprPool.checkOut(1, "tmpWaveID")

        tmpVgpr0 = writer.vgprPool.checkOut(1,"tmpVgpr0")
        tmpVgpr1 = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr1")
        tmpVgpr0Res = ContinuousRegister(tmpVgpr0, 1)
        tmpVgpr1Res = ContinuousRegister(tmpVgpr1, 2)
        dummy    = writer.vgprPool.checkOut(1,"dummy")

        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            tmpSgpr = tmpSgprInfo.idx

            # constant
            MIBShape0 = kernel["MatrixInstM"] * kernel["MatrixInstBM"]
            MIBShape1 = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

            matrixInstM = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
            matrixInstN = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]

            module = Module("ComputeStoreVgprsMFMA")

            # coord 1 : wave part
            module.add(vectorStaticDivide(wave_id, "Serial", writer.states.kernel["WavefrontSize"], tmpVgpr1Res))
            module.add(vectorStaticDivide(tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1Res))
            if kernel["LocalSplitU"] > 1:
                module.add(vectorStaticRemainder(dummy, tmpVgpr0, tmpVgpr0, kernel["MIWaveGroup"][1], tmpVgpr1Res, tmpSgprInfo))
            module.add(VMulLOU32(dst=vgpr(tid1), src0=hex(MIBShape1), src1=vgpr(tmpVgpr0), comment="wave coordination offset 1"))

            # coord 1 : thread part
            module.add(vectorStaticRemainder(dummy, tmpVgpr0, "Serial", matrixInstN, tmpVgpr1Res, tmpSgprInfo))
            module.add(VAddLShiftLeftU32(dst=vgpr(lsuTid1), src0=vgpr(tmpVgpr0), src1=vgpr(tid1), shiftHex=log2(kernel["VectorWidthB"]), comment="coordination 1 = vwB *(wave_id1 + tid1)"))

            # coord 1 : offset part
            packedC1 = kernel["PackedC1IndicesX"]
            strideC1 = "StrideC%s" % (writer.states.indexChars[packedC1[0]])
            strideD1 = "StrideD%s" % (writer.states.indexChars[packedC1[0]])
            module.add(VMulLOU32(dst=vgpr(writer.vgprs.cinRowPtr), src0=vgpr(lsuTid1), src1=sgpr(strideC1), comment=" offset 1"))
            module.add(VMulLOU32(dst=vgpr(writer.vgprs.coutRowPtrD), src0=vgpr(lsuTid1), src1=sgpr(strideD1), comment=" offset 1"))
            if kernel["ProblemType"]["UseE"] and (kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1):
                module.add(VMovB32(dst=vgpr(writer.vgprs.coutRowPtrE), src=vgpr(lsuTid1), comment=" save offset 1 for E"))
            if writer.vgprs.coutRowPtrBias != -1:
                index = packedC1[0] - 1
                strideW1 = "Size%s" % "I" if index == 0 else ("J" if index == 1 else (writer.states.indexChars[index]))
                module.add(VMulLOU32(dst=vgpr(writer.vgprs.coutRowPtrBias), src0=vgpr(lsuTid1), src1=sgpr(strideW1), comment=" offset 1"))

            # coord 0 : wave part
            module.add(vectorStaticRemainder(dummy, tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1Res, tmpSgprInfo))
            module.add(VMulLOU32(dst=vgpr(tmpVgpr0), src0=hex(MIBShape0), src1=vgpr(tmpVgpr0), comment="wave coordination offset 0"))

            # coord 0 : thread part
            module.add(vectorStaticRemainder(dummy, tid0, "Serial", writer.states.kernel["WavefrontSize"], tmpVgpr1Res, tmpSgprInfo))
            module.add(vectorStaticDivide(tid0, tid0, matrixInstN, tmpVgpr1Res))
            module.add(vectorStaticMultiply(vgpr(tid0), vgpr(tid0), kernel["MIOutputVectorWidth"], tmpSgprInfo, "thread0 * continuous_output"))
            module.add(VAddLShiftLeftU32(dst=vgpr(lsuTid0), src0=vgpr(tmpVgpr0), src1=vgpr(tid0), shiftHex=log2(kernel["VectorWidthA"]), comment="coordination 0 = vwA *(wave_id0 + tid0)"))

            wg0="WorkGroup0"
            wg1="WorkGroup1"

            # macro tile 0 part
            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile0"], src1=sgpr(wg0), comment="wgp0 * MT0"))
            module.add(VAddU32(dst=vgpr(tid0), src0=sgpr(tmpSgpr), src1=vgpr(lsuTid0), comment="coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0"))

            # macro tile 1 part
            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile1"], src1=sgpr(wg1), comment="wgp1 * MT1"))
            module.add(VAddU32(dst=vgpr(tid1), src0=sgpr(tmpSgpr), src1=vgpr(lsuTid1), comment="coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1"))

        # release resource
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(tmpVgpr1)
        writer.vgprPool.checkIn(tmpVgpr0)
        writer.vgprPool.checkIn(wave_id)

        # StoreRemap: calculate
        # 1. local read address
        # 2. local write address
        # 3. global write coord0 and coord1
        if kernel["StoreRemapVectorWidth"]:
            module.add(writer.storeRemapComputeStoreVgprs(kernel))

        writer.vgprs.coord0 = tid0
        writer.vgprs.coord1 = tid1

        return module

class ComputeStoreVgprsMFMASwap(ComputeStoreVgprs):
    kernel = {"EnableMatrixInstruction": True,
              "SourceSwap": True}

    """
    computeStoreVgprs
    Compute workitem/TT offsets in VGPRS
    and coord0/coord1
    tid0Scale specifies the number of output elements in 0/coalesced dim
    that should be written by each work-item in each batch element.
    """
    def __call__(self, writer, kernel):

        # writer.coord0
        # writer.coord1
        # writer.vgprs.cinRowPtr  : C buffer coulmn offset
        # writer.vgprs.coutRowPtrD : D buffer coulmn offset

        # alloc resources
        tid0 = writer.vgprPool.checkOut(1, "coord0")
        tid1 = writer.vgprPool.checkOut(1, "coord1")
        if kernel["BufferStore"]:
            writer.vgprs.cinRowPtr  = writer.vgprPool.checkOut(1, "cinRowPtr")
            writer.vgprs.coutRowPtrD = writer.vgprPool.checkOut(1, "coutRowPtrD")
            if kernel["ProblemType"]["UseE"]:
                writer.vgprs.coutRowPtrE = writer.vgprPool.checkOut(1, "coutRowPtrE")
            if writer.states.useBias == DataDirection.WRITE and (not kernel["WorkGroupReduction"]) and kernel["ProblemType"]["BiasSrc"] == "D":
                writer.vgprs.coutRowPtrBias = writer.vgprPool.checkOut(1, "coutRowPtrBias")
        if kernel["LocalSplitU"] > 1:
            writer.vgprs.coord0InMT = writer.vgprPool.checkOut(1, "coord0InMT")
            writer.vgprs.coord1InMT = writer.vgprPool.checkOut(1, "coord1InMT")
        else:
            writer.vgprs.coord0InMT = tid0
            writer.vgprs.coord1InMT = tid1
        lsuTid0 = writer.vgprs.coord0InMT
        lsuTid1 = writer.vgprs.coord1InMT

        wave_id = writer.vgprPool.checkOut(1, "tmpWaveID")

        tmpVgpr0 = writer.vgprPool.checkOut(1,"tmpVgpr0")
        tmpVgpr1 = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr1")
        #lsu_id   = tmpVgpr1
        tmpVgpr1Res = ContinuousRegister(tmpVgpr1, 2)
        dummy    = writer.vgprPool.checkOut(1,"dummy")

        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            tmpSgpr = tmpSgprInfo.idx

            # constant
            MIBShape0 = kernel["MatrixInstM"] * kernel["MatrixInstBM"]
            MIBShape1 = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

            matrixInstM = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
            matrixInstN = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]

            module = Module("ComputeStoreVgprsMFMASwap")


            # coord 1 : wave part
            module.add(vectorStaticDivide(wave_id, "Serial", writer.states.kernel["WavefrontSize"], tmpVgpr1Res))
            module.add(vectorStaticDivide(tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1Res))
            if kernel["LocalSplitU"] > 1:
                module.add(vectorStaticRemainder(dummy, tmpVgpr0, tmpVgpr0, kernel["MIWaveGroup"][1], tmpVgpr1Res, tmpSgprInfo))
            module.add(VMulLOU32(dst=vgpr(tmpVgpr0), src0=hex(MIBShape1), src1=vgpr(tmpVgpr0), comment="wave coordination offset 1"))

            # coord 1 : thread part
            module.add(vectorStaticRemainder(dummy, tid1, "Serial", writer.states.kernel["WavefrontSize"], tmpVgpr1Res, tmpSgprInfo))
            module.add(vectorStaticDivide(tid1, tid1, matrixInstM, tmpVgpr1Res))
            module.add(vectorStaticMultiply(vgpr(tid1), vgpr(tid1), kernel["MIOutputVectorWidth"], tmpSgprInfo, "thread0 * continuous_output"))
            module.add(VAddLShiftLeftU32(dst=vgpr(lsuTid1), src0=vgpr(tmpVgpr0), src1=vgpr(tid1), shiftHex=log2(kernel["VectorWidthB"]), comment="coordination 1 = vwB *(wave_id1 + tid1)"))

            # coord 1 : offset part
            packedC1 = kernel["PackedC1IndicesX"]
            strideC1 = "StrideC%s" % (writer.states.indexChars[packedC1[0]])
            strideD1 = "StrideD%s" % (writer.states.indexChars[packedC1[0]])
            module.add(VMulLOU32(dst=vgpr(writer.vgprs.cinRowPtr), src0=vgpr(lsuTid1), src1=sgpr(strideC1), comment=" offset 1"))
            module.add(VMulLOU32(dst=vgpr(writer.vgprs.coutRowPtrD), src0=vgpr(lsuTid1), src1=sgpr(strideD1), comment=" offset 1"))
            if kernel["ProblemType"]["UseE"] and (kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1):
                module.add(VMovB32(dst=vgpr(writer.vgprs.coutRowPtrE), src=vgpr(lsuTid1), comment=" save offset 1 for E"))
            if writer.vgprs.coutRowPtrBias != -1:
                index = packedC1[0] - 1
                strideW1 = "Size%s" % "I" if index == 0 else ("J" if index == 1 else (writer.states.indexChars[index]))
                module.add(VMulLOU32(dst=vgpr(writer.vgprs.coutRowPtrBias), src0=vgpr(lsuTid1), src1=sgpr(strideW1), comment=" offset 1"))

            # coord 0 : wave part
            module.add(vectorStaticRemainder(dummy, tid0, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1Res, tmpSgprInfo))
            module.add(VMulLOU32(dst=vgpr(tid0), src0=hex(MIBShape0), src1=vgpr(tid0), comment="wave coordination offset 0"))

            # coord 0 : thread part
            module.add(vectorStaticRemainder(dummy, tmpVgpr0, "Serial", matrixInstM, tmpVgpr1Res, tmpSgprInfo))
            module.add(VAddLShiftLeftU32(dst=vgpr(lsuTid0), src0=vgpr(tmpVgpr0), src1=vgpr(tid0), shiftHex=log2(kernel["VectorWidthA"]), comment="coordination 0 = vwA * (wave_id0 + tid0)"))

            wg0="WorkGroup0"
            wg1="WorkGroup1"

            # macro tile 0 part
            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile0"], src1=sgpr(wg0), comment="wgp0 * MT0"))
            module.add(VAddU32(dst=vgpr(tid0), src0=sgpr(tmpSgpr), src1=vgpr(lsuTid0), comment="coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0"))

            # macro tile 1 part
            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile1"], src1=sgpr(wg1), comment="wgp1 * MT1"))
            module.add(VAddU32(dst=vgpr(tid1), src0=sgpr(tmpSgpr), src1=vgpr(lsuTid1), comment="coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1"))

        # release resource
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(tmpVgpr1)
        writer.vgprPool.checkIn(tmpVgpr0)
        writer.vgprPool.checkIn(wave_id)

        # StoreRemap: calculate
        # 1. local read address
        # 2. local write address
        # 3. global write coord0 and coord1
        if kernel["StoreRemapVectorWidth"]:
            module.add(writer.storeRemapComputeStoreVgprs(kernel))

        writer.vgprs.coord0 = tid0
        writer.vgprs.coord1 = tid1

        return module
