################################################################################
#
# Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

from ..Component import ComputeStoreVgprs
from ..AsmUtils import log2, vectorStaticDivideAndRemainder, staticMultiply, vgpr, sgpr, inst, vectorStaticDivide, vectorStaticRemainder

class ComputeStoreVgprsVALU(ComputeStoreVgprs):
    kernel = {"EnableMatrixInstructionStore": False}

    """
    computeStoreVgprs
    Compute workitem/TT offsets in VGPRS
    and coord0/coord1
    tid0Scale specifies the number of output elements in 0/coalesced dim
    that should be written by each work-item in each batch element.
    """
    def __call__(self, writer, kernel, divisor, tid0Scale, tid1Scale):

        kStr = ""

        tmpS0 = writer.getTmpSgpr(3).idx()
        tmpS1 = tmpS0+1
        wgMT1 = tmpS0+2

        if writer.prefetchAcrossPersistent:
            wg0="PrevWorkGroup0"
            wg1="PrevWorkGroup1"
        else:
            wg0="WorkGroup0"
            wg1="WorkGroup1"

        # tid0, tid1: element offsets from the start of macroTile in 0 and 1 direction
        # These will live for entire GlobalWrite loop - allocate before tmps
        # to avoid fragmentation
        tid0 = writer.vgprPool.checkOut(1, "tid0")
        tid1 = writer.vgprPool.checkOut(1, "tid1")

        packedC1 = kernel["PackedC1IndicesX"]

        if kernel["BufferStore"]:
            writer.cinRowPtr    = writer.vgprPool.checkOut(1, "cinRowPtr")
            writer.coutRowPtr = writer.vgprPool.checkOut(1, "coutRowPtr")

        kStr += vectorStaticDivideAndRemainder(tid1, tid0, "Serial", divisor, tmpS0)
        kStr += staticMultiply(vgpr(tid0), vgpr(tid0), tid0Scale, sgpr(tmpS1))
        if tid1Scale != 1:
            kStr += staticMultiply(vgpr(tid1), vgpr(tid1), tid1Scale, sgpr(tmpS1))

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
            strideC1 = "StrideC%s" % (writer.indexChars[packedC1[0]])
            kStr += inst("v_mul_lo_u32", vgpr(writer.cinRowPtr),
                         vgpr(tid1), sgpr(strideC1), \
                         "rowStart vgpr")
            strideD1 = "StrideD%s" % (writer.indexChars[packedC1[0]])
            kStr += inst("v_mul_lo_u32", vgpr(writer.coutRowPtr),
                         vgpr(tid1), sgpr(strideD1), \
                         "rowStart vgpr")
            kStr += "\n"

            #kStr += writer.assert_ne(sgpr("WorkGroup1"),1)

        # Compute coord0 and coord1
        # These are element offsets from the beginning of the tensor.
        # These are 'flattened' meaning they span packed tensor dims.
        # They need to be preserved so can use in comparisons against
        # product-of-packed sizes to determine OOB cases. (for Edge tiles only)
        kStr += inst("s_mul_i32", \
                sgpr(tmpS0), \
                hex(kernel["MacroTile0"]), \
                sgpr(wg0), \
                "%s = wg0*MT0"%sgpr(tmpS0))

        # coord = tid*VW + workgroup offset
        kStr += inst("_v_add_co_u32", \
                vgpr(tid0), \
                writer.vcc, \
                sgpr(tmpS0), \
                vgpr(tid0), \
                "coord0 = tid0*VW + wg0*MT0")
        kStr += inst("s_mul_i32", \
                sgpr(wgMT1), \
                hex(kernel["MacroTile1"]), \
                sgpr(wg1), \
                "<- wg1*MT1")
        kStr += inst("_v_add_co_u32", \
                vgpr(tid1), \
                writer.vcc, \
                sgpr(wgMT1), \
                vgpr(tid1), \
                "coord1 = tid1*VW + wg1*MT1")

        if len(packedC1) > 1:
          kStr += writer.extractPackedCoord1ToRowStart(kernel, packedC1, tid1, 'D')

        writer.coord0 = tid0
        writer.coord1 = tid1

        return kStr

class ComputeStoreVgprsMFMA(ComputeStoreVgprs):
    kernel = {"EnableMatrixInstructionStore": True,
              "SourceSwap": False}

    """
    computeStoreVgprs
    Compute workitem/TT offsets in VGPRS
    and coord0/coord1
    tid0Scale specifies the number of output elements in 0/coalesced dim
    that should be written by each work-item in each batch element.
    """
    def __call__(self, writer, kernel, divisor, tid0Scale, tid1Scale):

        # writer.coord0
        # writer.coord1
        # writer.cinRowPtr  : C buffer column offset
        # writer.coutRowPtr : D buffer column offset

        # alloc resources
        tid0 = writer.vgprPool.checkOut(1)
        tid1 = writer.vgprPool.checkOut(1)
        if kernel["BufferStore"]:
            writer.cinRowPtr    = writer.vgprPool.checkOut(1, "cinRowPtr")
            writer.coutRowPtr = writer.vgprPool.checkOut(1, "coutRowPtr")

        wave_id = writer.vgprPool.checkOut(1)

        tmpVgpr0 = writer.vgprPool.checkOut(1,"tmpVgpr0")
        tmpSgpr  = writer.getTmpSgpr(1).idx()

        # constant
        MIBShape0 = kernel["MatrixInstM"] * kernel["MatrixInstBM"]
        MIBShape1 = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

        # matrixInstM = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
        matrixInstN = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]

        kStr = ""

        # coord 1 : wave part
        kStr += vectorStaticDivide(wave_id, "Serial", writer.kernel["WavefrontSize"], tmpSgpr)
        kStr += vectorStaticDivide(tid1, wave_id, kernel["MIWaveGroup"][0], tmpSgpr)
        kStr += inst("v_mul_lo_u32", vgpr(tid1), hex(MIBShape1), vgpr(tid1), "wave coordination offset 1")

        # coord 1 : thread part
        kStr += vectorStaticRemainder(tmpVgpr0, "Serial", matrixInstN, tmpSgpr)
        kStr += inst("_v_add_lshl_u32", vgpr(tid1), vgpr(tmpVgpr0), vgpr(tid1), log2(kernel["VectorWidthB"]), "coordination 1 = vwb *(wave_id1 + tid1)")

        if kernel["BufferStore"]:
          # coord 1 : offset part
          packedC1 = kernel["PackedC1IndicesX"]
          strideC1 = "StrideC%s" % (writer.indexChars[packedC1[0]])
          strideD1 = "StrideD%s" % (writer.indexChars[packedC1[0]])
          kStr += inst("v_mul_lo_u32", vgpr(writer.cinRowPtr), vgpr(tid1), sgpr(strideC1), " offset 1")
          kStr += inst("v_mul_lo_u32", vgpr(writer.coutRowPtr), vgpr(tid1), sgpr(strideD1), " offset 1")

        # coord 0 : thread part
        kStr += vectorStaticRemainder(tid0, "Serial", writer.kernel["WavefrontSize"], tmpSgpr)
        kStr += vectorStaticDivide(tid0, tid0, matrixInstN, tmpSgpr)
        kStr += staticMultiply(vgpr(tid0), vgpr(tid0), kernel["MIOutputVectorWidth"], sgpr(tmpSgpr), "thread0 * continuous_output")
        if kernel["MatrixInstBM"] > 1 and kernel["MatrixInstN"] == 4 and (kernel["MatrixInstM"] > kernel["MIOutputVectorWidth"]):
          # conversion for MI4x4 + MIBM>1
          # tid0 = (tid0/MIBM) + (tid0%MIBM)*(MIM//MIOVW)
          kStr += vectorStaticDivideAndRemainder(tmpVgpr0, tid0, tid0, kernel["MatrixInstBM"], tmpSgpr) # using rReg=dReg (assuming MIBM is power of 2)
          kStr += staticMultiply(vgpr(tid0), vgpr(tid0), kernel["MatrixInstM"]//kernel["MIOutputVectorWidth"], sgpr(tmpSgpr), "(tid1%MIBM)*(MIM//MIOVW)")
          kStr += inst("_v_add_u32", vgpr(tid0), vgpr(tmpVgpr0), vgpr(tid0), "tid0 = (tid0/MIBM) + (tid0%MIBM)*MIM")
        # coord 0 : wave part
        kStr += vectorStaticRemainder(tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpSgpr)
        kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr0), hex(MIBShape0), vgpr(tmpVgpr0), "wave coordination offset 0")

        kStr += inst("_v_add_lshl_u32", vgpr(tid0), vgpr(tmpVgpr0), vgpr(tid0), log2(kernel["VectorWidthA"]), "coordination 0 = vwa *(wave_id0 + tid0)")

        if writer.prefetchAcrossPersistent:
            wg0="PrevWorkGroup0"
            wg1="PrevWorkGroup1"
        else:
            wg0="WorkGroup0"
            wg1="WorkGroup1"

        # macro tile 0 part
        kStr += inst("s_mul_i32", sgpr(tmpSgpr), kernel["MacroTile0"], sgpr(wg0), "wgp0 * MT0")
        kStr += inst("_v_add_u32", vgpr(tid0), sgpr(tmpSgpr), vgpr(tid0), "coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0")

        # macro tile 1 part
        kStr += inst("s_mul_i32", sgpr(tmpSgpr), kernel["MacroTile1"], sgpr(wg1), "wgp1 * MT1")
        kStr += inst("_v_add_u32", vgpr(tid1), sgpr(tmpSgpr), vgpr(tid1), "coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1")

        # extract packed rowStart vgpr
        if kernel["BufferStore"] and len(packedC1) > 1:
          kStr += writer.extractPackedCoord1ToRowStart(kernel, packedC1, tid1, 'D')

        # release resource
        writer.vgprPool.checkIn(tmpVgpr0)
        writer.vgprPool.checkIn(wave_id)

        # StoreRemap: calculate
        # 1. local read address
        # 2. local write address
        # 3. global write coord0 and coord1
        if kernel["StoreRemapVectorWidth"]:
            kStr += writer.storeRemapComputeStoreVgprs(kernel)

        writer.coord0 = tid0
        writer.coord1 = tid1

        return kStr

class ComputeStoreVgprsMFMASwap(ComputeStoreVgprs):
    kernel = {"EnableMatrixInstructionStore": True,
              "SourceSwap": True}

    """
    computeStoreVgprs
    Compute workitem/TT offsets in VGPRS
    and coord0/coord1
    tid0Scale specifies the number of output elements in 0/coalesced dim
    that should be written by each work-item in each batch element.
    """
    def __call__(self, writer, kernel, divisor, tid0Scale, tid1Scale):

        # writer.coord0
        # writer.coord1
        # writer.cinRowPtr  : C buffer column offset
        # writer.coutRowPtr : D buffer column offset

        # alloc resources
        tid0 = writer.vgprPool.checkOut(1)
        tid1 = writer.vgprPool.checkOut(1)
        if kernel["BufferStore"]:
            writer.cinRowPtr    = writer.vgprPool.checkOut(1, "cinRowPtr")
            writer.coutRowPtr = writer.vgprPool.checkOut(1, "coutRowPtr")

        wave_id = writer.vgprPool.checkOut(1)

        tmpVgpr0 = writer.vgprPool.checkOut(1,"tmpVgpr0")
        tmpSgpr  = writer.getTmpSgpr(1).idx()

        # constant
        MIBShape0 = kernel["MatrixInstM"] * kernel["MatrixInstBM"]
        MIBShape1 = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

        matrixInstM = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
        # matrixInstN = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]

        kStr = ""

        kStr += vectorStaticDivide(wave_id, "Serial", writer.kernel["WavefrontSize"], tmpSgpr)


        # coord 1 : thread part
        kStr += vectorStaticRemainder(tid1, "Serial", writer.kernel["WavefrontSize"], tmpSgpr)
        kStr += vectorStaticDivide(tid1, tid1, matrixInstM, tmpSgpr)
        kStr += staticMultiply(vgpr(tid1), vgpr(tid1), kernel["MIOutputVectorWidth"], sgpr(tmpSgpr), "thread0 * continuous_output")
        if kernel["MatrixInstBN"] > 1 and kernel["MatrixInstM"] == 4 and (kernel["MatrixInstN"] > kernel["MIOutputVectorWidth"]):
          # conversion for MI4x4 + MIBN>1
          # tid1 = (tid1/MIBN) + (tid1%MIBN)*(MIN//MIOVW)
          kStr += vectorStaticDivideAndRemainder(tmpVgpr0, tid1, tid1, kernel["MatrixInstBN"], tmpSgpr) # using rReg=dReg (assuming MIBN is power of 2)
          kStr += staticMultiply(vgpr(tid1), vgpr(tid1), kernel["MatrixInstN"]//kernel["MIOutputVectorWidth"], sgpr(tmpSgpr), "(tid1%MIBN)*(MIN//MIOVW)")
          kStr += inst("_v_add_u32", vgpr(tid1), vgpr(tmpVgpr0), vgpr(tid1), "tid1 = (tid1/MIBN) + (tid1%MIBN)*MIN")
        # coord 1 : wave part
        kStr += vectorStaticDivide(tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpSgpr)
        kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr0), hex(MIBShape1), vgpr(tmpVgpr0), "wave coordination offset 1")
        kStr += inst("_v_add_lshl_u32", vgpr(tid1), vgpr(tmpVgpr0), vgpr(tid1), log2(kernel["VectorWidthB"]), "coordination 1 = vwb *(wave_id1 + tid1)")

        if kernel["BufferStore"]:
          # coord 1 : offset part
          packedC1 = kernel["PackedC1IndicesX"]
          strideC1 = "StrideC%s" % (writer.indexChars[packedC1[0]])
          strideD1 = "StrideD%s" % (writer.indexChars[packedC1[0]])
          kStr += inst("v_mul_lo_u32", vgpr(writer.cinRowPtr), vgpr(tid1), sgpr(strideC1), " offset 1")
          kStr += inst("v_mul_lo_u32", vgpr(writer.coutRowPtr), vgpr(tid1), sgpr(strideD1), " offset 1")

        # coord 0 : wave part
        kStr += vectorStaticRemainder(tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpSgpr)
        if kernel["MIWaveGroup"][0] > 1:
          kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr0), hex(MIBShape0), vgpr(tmpVgpr0), "wave coordination offset 0")

        # coord 0 : thread part
        kStr += vectorStaticRemainder(tid0, "Serial", matrixInstM, tmpSgpr)
        kStr += inst("_v_add_lshl_u32", vgpr(tid0), vgpr(tmpVgpr0), vgpr(tid0), log2(kernel["VectorWidthA"]), "coordination 0 = vwa *(wave_id0 + tid0)")

        if writer.prefetchAcrossPersistent:
            wg0="PrevWorkGroup0"
            wg1="PrevWorkGroup1"
        else:
            wg0="WorkGroup0"
            wg1="WorkGroup1"

        # macro tile 0 part
        kStr += inst("s_mul_i32", sgpr(tmpSgpr), kernel["MacroTile0"], sgpr(wg0), "wgp0 * MT0")
        kStr += inst("_v_add_u32", vgpr(tid0), sgpr(tmpSgpr), vgpr(tid0), "coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0")

        # macro tile 1 part
        kStr += inst("s_mul_i32", sgpr(tmpSgpr), kernel["MacroTile1"], sgpr(wg1), "wgp1 * MT1")
        kStr += inst("_v_add_u32", vgpr(tid1), sgpr(tmpSgpr), vgpr(tid1), "coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1")

        # release resource
        writer.vgprPool.checkIn(tmpVgpr0)
        writer.vgprPool.checkIn(wave_id)

        # StoreRemap: calculate
        # 1. local read address
        # 2. local write address
        # 3. global write coord0 and coord1
        if kernel["StoreRemapVectorWidth"]:
            kStr += writer.storeRemapComputeStoreVgprs(kernel)

        writer.coord0 = tid0
        writer.coord1 = tid1

        return kStr
