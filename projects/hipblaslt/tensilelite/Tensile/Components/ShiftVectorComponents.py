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

from rocisa.code import Label, Module
from rocisa.container import VCC, DSModifiers, vgpr, sgpr, ContinuousRegister
from rocisa.instruction import DSBPermuteB32, SBranch, SCBranchVCCNZ, \
                                SMovB32, SMovB64, SNop, \
                                SOrSaveExecB32, SOrSaveExecB64, SWaitCnt, \
                                VAddCOU32, \
                                VAndB32, VCmpEQU32, VCmpLtU32, VCmpXEqU32, \
                                VCndMaskB32, VMovB32, VMulI32I24, VLShiftLeftB32, \
                                VLShiftRightB32, VSubU32, VMinI32
from rocisa.functions import vectorStaticRemainder, vectorStaticDivide, vectorStaticMultiply
from ..Common import log2
from ..Component import ShiftVectorComponents
from ..KernelWriterModules import *

class ShiftVectorComponentsVALU(ShiftVectorComponents):
    kernel = {"EnableMatrixInstruction": False}

    """
    Shift Vector Components d0,1
    """
    def __call__(self, writer, kernel, tP):
        # kStr = ""
        module = Module("ShiftVectorComponentsVALU")

        # common parameter
        tc              = tP["tensorChar"]
        glvw            = tP["glvw"]
        numThreadInWave = writer.states.kernel["WavefrontSize"]
        vectorWidth     = kernel["VectorWidth%s"%tc]
        numVectors = kernel[tP["tt"]]//glvw

        # labels
        svrLabels = []
        sviLabels = []
        for i in range(0, glvw):
            r = (i+1) % glvw
            # label = writer.getLabelNum("ShiftVectorComponents%u_R%u"%(tP["idx"], r) )
            label = Label(writer.labels.getName("ShiftVectorComponents%u_R%u"%(tP["idx"], r) ), "")
            svrLabels.append(label)
            tmpLabels = []
            for v in range(0, numVectors):
                # label = writer.getLabelNum("ShiftVectorComponents%u_R%u_V%u"%(tP["idx"], r, v) )
                label = Label(writer.labels.getName("ShiftVectorComponents%u_R%u_V%u"%(tP["idx"], r, v) ), "")
                tmpLabels.append(label)
            sviLabels.append(tmpLabels)

        with writer.allocTmpSgpr(writer.states.laneSGPRCount) as tmpSgprInfo:
            # wgMT value
            wg = tP["wg"]
            tmpSgpr = tmpSgprInfo.idx
            dummy   = writer.vgprPool.checkOut(1)
            tmpVgpr = writer.vgprPool.checkOut(1,"tmpVgpr")
            wgMT = writer.vgprPool.checkOut(1,"wgMT")
            tmpVgprRes = ContinuousRegister(tmpVgpr, 2)
            module.add(VMovB32(dst=vgpr(wgMT), src=sgpr(wg), comment=""))
            module.add(VMulI32I24(dst=vgpr(wgMT), src0=hex(-kernel[tP["mt"]]), src1=vgpr(wgMT), comment="wg*MT"))
            module.add(VAddCOU32(dst=vgpr(wgMT), dst1=VCC(), src0=sgpr("SizesFree+%u"%tP["idx"]), src1=vgpr(wgMT), comment="wgMT = Size - wg*MT"))
            module.add(VMovB32(dst=vgpr(tmpVgpr), src=hex(kernel[tP["mt"]]), comment="MT"))
            module.add(VMinI32(dst=vgpr(wgMT), src0=vgpr(tmpVgpr), src1=vgpr(wgMT), comment="wgMT = (wgMT < MT) ? wgMT : MT"))
            #module.add(VCndMaskB32(dst=vgpr(wgMT), src0=vgpr(mtReg), src1=vgpr(wgMT), src2=sgpr(tmpSgpr,writer.states.laneSGPRCount), comment="wgMT = (wgMT < MT) ? wgMT : MT" ))
            writer.vgprPool.checkIn(tmpVgpr)

            # qReg
            qReg = writer.vgprPool.checkOut(1,"qReg")
            module.add(vectorStaticDivide(qReg, wgMT, vectorWidth, tmpVgprRes))

            # rReg
            rReg = writer.vgprPool.checkOut(1,"rReg")
            module.add(vectorStaticRemainder(dummy, rReg, wgMT, glvw, tmpVgprRes, tmpSgprInfo))

            # qReg %/ SG
            eReg = writer.vgprPool.checkOut(1,"eReg")
            divisor = kernel[tP["sg"]]
            module.add(vectorStaticRemainder(dummy, eReg, qReg, divisor, tmpVgprRes, tmpSgprInfo))

            if tP["isA"]:
                # thread = serial % SG0
                thread = writer.vgprPool.checkOut(1,"thread")
                divisor = kernel["SubGroup0"]
                module.add(vectorStaticRemainder(dummy, thread, "Serial", divisor, tmpVgprRes, tmpSgprInfo))
            else:
                # thread = (serial / SG0) % SG1
                sd0 = writer.vgprPool.checkOut(1,"sd0")
                divisor = kernel["SubGroup0"]
                module.add(vectorStaticDivide(sd0, "Serial", divisor, tmpVgprRes))
                divisor = kernel["SubGroup1"]
                thread = writer.vgprPool.checkOut(1,"thread")
                module.add(vectorStaticRemainder(dummy, thread, sd0, divisor, tmpVgprRes, tmpSgprInfo))
                writer.vgprPool.checkIn(sd0)

            # which glvw vector of thread to shift? wgMT / (SG0*VW) -> (wgMT%VW) / glvw
            # (wgMT/(WG0*VW))*(VW/glvw) + (wgMT%VW) / glvw
            if True:#tP["tensorIdx"] > kernel["VectorWidth"]:
                mvReg = writer.vgprPool.checkOut(1,"mvReg")
                divisor = kernel[tP["sg"]]*vectorWidth
                module.add(vectorStaticDivide(mvReg, wgMT, divisor, tmpVgprRes))
                if glvw < vectorWidth:
                    module.add(VLShiftRightB32(dst=vgpr(mvReg), shiftHex=hex(log2(vectorWidth//glvw)), src=vgpr(mvReg), comment="vId *= GLVW/glvw"))

            vReg = writer.vgprPool.checkOut(1,"vReg")
            divisor = vectorWidth
            module.add(vectorStaticRemainder(dummy, vReg, wgMT, divisor, tmpVgprRes, tmpSgprInfo))
            vRegD = writer.vgprPool.checkOut(1,"vRegD")
            module.add(VMovB32(dst=vgpr(vRegD), src=vgpr(vReg), comment="duplicate"))
            divisor = glvw
            module.add(vectorStaticDivide(vReg, vRegD, divisor, tmpVgprRes))

            if True:#tP["tensorIdx"] > kernel["VectorWidth"]:
                module.add(VAddCOU32(dst=vgpr(vReg), dst1=VCC(), src0=vgpr(mvReg), src1=vgpr(vReg), comment="vId = 2 components"))
                writer.vgprPool.checkIn(mvReg)
                writer.vgprPool.checkIn(vRegD)

            # for each remainder, jump
            for r in range(1, glvw):
                module.add(VCmpEQU32(dst=VCC(), src0=vgpr(rReg), src1=hex(r), comment="wgMT%%GLVW == %u"%r ))
                module.add(SCBranchVCCNZ(labelName=svrLabels[(r-1)%glvw].getLabelName(), comment="shift d%u r=%u"%(tP["idx"], r)))
            module.add(SBranch(labelName=svrLabels[glvw-1].getLabelName(), comment="no shifting" ))

            # code blocks for shifting
            for r in range(1, glvw):
                module.addComment2("shift d%u r=%u"%(tP["idx"], r))
                module.add(svrLabels[r-1])

                # for each vector index, jump
                for vectorIdx in range(0, numVectors):
                    module.add(VCmpEQU32(dst=VCC(), src0=vgpr(vReg), src1=hex(vectorIdx), comment="wgMT/(SG*VW) == %u"%vectorIdx ))
                    module.add(SCBranchVCCNZ(labelName=sviLabels[(r-1)%glvw][vectorIdx].getLabelName(), comment="shift d%u, r=%u, v=%u"%(tP["idx"], r, vectorIdx)))

                # code blocks for shifting vector
                for vectorIdx in range(0, numVectors):
                    module.addComment1("shift d%u r=%u v=%u"%(tP["idx"], r, vectorIdx))
                    module.add(sviLabels[r-1][vectorIdx])

                    # mask if last thread in thread#-tile column
                    module.add(VCmpXEqU32(dst=sgpr(tmpSgpr, writer.states.laneSGPRCount), src0=vgpr(thread), src1=vgpr(eReg), comment="serial % SG == (wgMT/VECTOR_WIDTH)%SG" ))
                    tto = kernel["ThreadTile%u"%((tP["idx"]+1)%2)] # thread tile orthogonal
                    for tt in range(0, tto):
                        for s in range(0, r):
                            comment = ""
                            if tP["isA"]: # shift d0
                                dst = (s) \
                                    + vectorIdx * glvw + tt * kernel["ThreadTile0"]
                                src = (s+glvw-r) \
                                    + vectorIdx * glvw + tt * kernel["ThreadTile0"]
                            else: # shift d1
                                dst = (tt) \
                                    + vectorIdx*glvw*kernel["ThreadTile0"] + s * kernel["ThreadTile0"]
                                src = (tt) \
                                    + vectorIdx * glvw*kernel["ThreadTile0"] + (s+glvw-r) * kernel["ThreadTile0"]

                            module.addComment("src=%u, dst=%u" % (src,dst))

                            # f32, f16 isn't be supported
                            for i in range(0, writer.states.bpeCinternal//writer.states.bpr):
                                module.add(VMovB32(dst=vgpr(writer.states.c.startVgprValu+dst*writer.states.bpeCinternal//writer.states.bpr+i), \
                                                   src=vgpr(writer.states.c.startVgprValu+src*writer.states.bpeCinternal//writer.states.bpr+i), \
                                                   comment=comment))

                    # end shift reset mask and jump out
                    all1mask = "0xFFFFFFFF" if (kernel["WavefrontSize"] == 32) else "0xFFFFFFFFFFFFFFFF"
                    SOrSaveExecBX = SOrSaveExecB64 if kernel["WavefrontSize"] == 64 else SOrSaveExecB32
                    module.add(SMovB32(sgpr(tmpSgpr, writer.states.laneSGPRCount), all1mask, "to restore all threads active"))
                    module.add(SOrSaveExecBX(dst=VCC(), src=sgpr(tmpSgpr, writer.states.laneSGPRCount), comment="all threads active"))
                    module.add(SBranch(labelName=svrLabels[glvw-1].getLabelName(), comment="done shifting" ))
            module.add(svrLabels[glvw-1])

        # checkin scratch vgprs
        writer.vgprPool.checkIn(wgMT)
        writer.vgprPool.checkIn(qReg)
        writer.vgprPool.checkIn(rReg)
        writer.vgprPool.checkIn(eReg)
        writer.vgprPool.checkIn(thread)
        writer.vgprPool.checkIn(vReg)
        writer.vgprPool.checkIn(tmpVgpr)
        writer.vgprPool.checkIn(dummy)
        return module

class ShiftVectorComponentsMFMA(ShiftVectorComponents):
    kernel = {"EnableMatrixInstruction": True}

    """
    Shift Vector Components d0,1
    """
    def __call__(self, writer, kernel, tP):
        """
            if (glvw > vectorwidth * continuousOutput * threadsInCoal)
                use all thread algorithm
            else
                use partial thread algorithm
        """
        # common parameter
        tc              = tP["tensorChar"]
        glvw            = tP["glvw"]
        numThreadInWave = writer.states.kernel["WavefrontSize"]
        vectorWidth     = kernel["VectorWidth%s"%tc]

        # use to handle MatrixInst 4x4
        matrixInstM     = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
        matrixInstN     = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]

        # unify process for dimension M/N
        matrixInstCoal  = matrixInstM if tP["isA"] else matrixInstN
        matrixInstPrep  = matrixInstN if tP["isA"] else matrixInstM

        # unify process for SourceSwap and non-SourceSwap
        conThInProcDim  = kernel["SourceSwap"] ^ tP["isB"] # continuous threads in processed dimension(Coalesced dimension)
        numThreadInCoal = matrixInstCoal if conThInProcDim else (numThreadInWave // matrixInstPrep)
        #numContOutCoal  = 1 if conThInProcDim else kernel["MIOutputVectorWidth"]
        #allContOutCoal  = numContOutCoal * vectorWidth
        numContOutCoal  = vectorWidth if conThInProcDim else kernel["MIOutputVectorWidth"] * vectorWidth
        allContOutCoal  = numContOutCoal

        if (glvw > (allContOutCoal * numThreadInCoal)):
           return self.ShiftVectorComponentsMFMAAllThread(writer, kernel, tP)
        else:
           return self.ShiftVectorComponentsMFMAPartialThread (writer, kernel, tP)


    def ShiftVectorComponentsMFMAPartialThread(self, writer, kernel, tP):

        """ when we enable shift ptr with vectorwidth(2), we shift global read on edge block when size % vectorwidth != 0.
            For example if M size == 3 vector width == 2, we want to do global read for [0-1] and [2-3].
            But 3 is not in memory object, so we shift to do global read [0-1] and [1-2].
            So buffer become [0, 1, 1, 2], assume result in register is same as input [0, 1, 1, 2]
            We need to shift it back to [0, 1, 2].

            In MFMA outputs, We have numContinuousOutput(4) for each thread.
            We have numThreadInWave(64) threads.
            number of thread in N is sames as kernel["MatrixInstN"] (32)
            number of thread in M is numThreadInWave/numOutputThreads1 = 2
            stride of continuous output for each thread (numSubOutputPerWave0) is numOutputThreads0 * numContinuousOutput, (8).
            we have numSubOutputGroupsPerWave0 which is 4 (kernel[tP["mt"]](64) // numSubOutputPerWave0(8))

            So we do shift back by below algorithm.
            1. check if M_size % GlobalReadVectorWidth != 0, return if == 0
            2. decide which subgroup we need to shift, M_size(3) means 3/8 = group 0
            3. decide which thread we need to shift, we have different groups of thread, (0-31) for first group, (32-63) for second group.
            4. decide which shift block (subTile1) we want to shift. for ex [0-1], [1-2], we want to shift second subtile
        """

        # common parameter
        tc              = tP["tensorChar"]
        regPerElem      = kernel["MIRegPerOut"]
        glvw            = tP["glvw"]
        numThreadInWave = writer.states.kernel["WavefrontSize"]
        accImOffset     = accVgprImagNumOffset(kernel)
        vectorWidth     = kernel["VectorWidth%s"%tc]

        # use to handle MatrixInst 4x4
        matrixInstM     = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
        matrixInstN     = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]
        matrixInstBM    = 1 if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBM"]
        matrixInstBN    = 1 if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstBN"]

        # unify process for dimension M/N
        matrixInstCoal  = matrixInstM              if tP["isA"] else matrixInstN
        matrixInstPrep  = matrixInstN              if tP["isA"] else matrixInstM
        matrixInstBCoal = matrixInstBM             if tP["isA"] else matrixInstBN
        matrixInstBPrep = matrixInstBN             if tP["isA"] else matrixInstBM
        miWaveGroupCoal = kernel["MIWaveGroup"][0] if tP["isA"] else kernel["MIWaveGroup"][1]
        miWGIdStride    = numThreadInWave          if tP["isA"] else (numThreadInWave * kernel["MIWaveGroup"][0])
        miWaveTitleCoal = kernel["MIWaveTile"][0]  if tP["isA"] else kernel["MIWaveTile"][1]
        miWaveTitlePrep = kernel["MIWaveTile"][1]  if tP["isA"] else kernel["MIWaveTile"][0]

        # unify process for SourceSwap and non-SourceSwap
        conThInProcDim  = kernel["SourceSwap"] ^ tP["isB"] # continuous threads in processed dimension(Coalesced dimension)

        threadInterval  = 1 if conThInProcDim else matrixInstPrep
        numThreadInCoal = matrixInstCoal if conThInProcDim else (numThreadInWave // matrixInstPrep)

        numContOutCoal  = vectorWidth if conThInProcDim else kernel["MIOutputVectorWidth"] * vectorWidth
        allContOutCoal  = numContOutCoal

        OutBlocksInMI   = (vectorWidth * matrixInstCoal * matrixInstPrep) // numThreadInWave // numContOutCoal
        OutBlocksInMI   = 1 if conThInProcDim else OutBlocksInMI

        subMBShapeCoal  = (matrixInstCoal * vectorWidth) if conThInProcDim else ((numThreadInWave // matrixInstPrep) * numContOutCoal)
        MBShapeCoal     = subMBShapeCoal * OutBlocksInMI
        MIBShapeCoal    = MBShapeCoal * matrixInstBCoal
        WGShapeCoal     = MIBShapeCoal * miWaveGroupCoal
        miOuterTTCoal   = miWaveTitleCoal // vectorWidth

        numOutputsPrep  = (matrixInstCoal * matrixInstPrep // numThreadInWave) if conThInProcDim else 1
        numOutputsPrep  = numOutputsPrep * matrixInstBPrep * miWaveTitlePrep
        complexMultiplier = 2 if kernel["ProblemType"]["DataType"].isComplex() else 1

        # unify process for dimension M/N
        regStrideCoal = 1                                                                if tP["isA"] else numOutputsPrep
        regStridePrep = miOuterTTCoal * matrixInstBCoal * OutBlocksInMI * allContOutCoal if tP["isA"] else 1


        # labels for shiftptr
        glvwLabels = []
        MBblockLabels = []
        VWBlockLabels = []
        for i in range(0, glvw): # grvw block
            r = (i+1) % glvw    # r = [1,2,3,...,glvw-1, 0], the last one glvwLabels[glvw-1] stores for r=0 -> no shift
            comment = "end shift0" if i == glvw-1 else ""
            label = Label(writer.labels.getName("ShiftVectorComponents%u_GLVW%u" % (tP["idx"], r) ), comment)
            glvwLabels.append(label)
            subMBLabels = []
            subVWBlockLabels = []
            for mb in range(0, OutBlocksInMI * matrixInstBCoal * miOuterTTCoal): # unit block of each thread
                label = Label(writer.labels.getName("ShiftVectorComponents%u_GLVW%u_BM%u" % (tP["idx"], r, mb)), "")
                subMBLabels.append(label)
                sub2VWBlockLabels = []
                for vw in range(0, max(1, allContOutCoal//glvw)): # vw block of glvw
                    label = Label(writer.labels.getName("ShiftVectorComponents%u_GLVW%u_BM%u_VW%u" % (tP["idx"], r, mb, vw)), "")
                    sub2VWBlockLabels.append(label)
                subVWBlockLabels.append(sub2VWBlockLabels)
            MBblockLabels.append(subMBLabels)
            VWBlockLabels.append(subVWBlockLabels)

        with writer.allocTmpSgpr(writer.states.laneSGPRCount) as tmpSgprInfo:
            # wgMT value
            tmpSgpr = tmpSgprInfo.idx
            tmpVgpr = writer.vgprPool.checkOutAligned(2,2)
            tmpVgprRes = ContinuousRegister(tmpVgpr, 2)
            dummy   = writer.vgprPool.checkOut(1)
            wgMT    = writer.vgprPool.checkOut(1)
            wg      = tP["wg"]

            module = Module("ShiftVectorComponentsMFMA")

            # get M size of edge block
            mtReg = writer.vgprPool.checkOut(1)
            module.add(VMovB32(dst=vgpr(wgMT), src=sgpr(wg)))
            module.add(VMulI32I24(dst=vgpr(wgMT), src0=hex(-kernel[tP["mt"]]), src1=vgpr(wgMT), comment="wg*MT"))
            module.add(VAddCOU32(dst=vgpr(wgMT), dst1=VCC(), src0=sgpr("SizesFree+%u"%tP["idx"]), src1=vgpr(wgMT), comment="wgMT = Size - wg*MT"))
            module.add(VMovB32(dst=vgpr(mtReg), src=hex(kernel[tP["mt"]]), comment="MT"))
            module.add(VCmpLtU32(dst=sgpr(tmpSgpr,writer.states.laneSGPRCount), src0=vgpr(wgMT), src1=vgpr(mtReg), comment="wgMT < MT"))
            module.add(VCndMaskB32(dst=vgpr(wgMT), src0=vgpr(mtReg), src1=vgpr(wgMT), src2=sgpr(tmpSgpr,writer.states.laneSGPRCount), comment="wgMT = (wgMT < MT) ? wgMT : MT" ))

            # identify which wave have to process
            wReg = writer.vgprPool.checkOut(1)
            sReg = writer.vgprPool.checkOut(1)
            module.add(vectorStaticDivide(wReg, "Serial", miWGIdStride, tmpVgprRes))
            module.add(vectorStaticRemainder(dummy, wReg, wReg, miWaveGroupCoal, tmpVgprRes, tmpSgprInfo))
            module.add(vectorStaticDivide(sReg, wgMT, MIBShapeCoal, tmpVgprRes))
            module.add(vectorStaticRemainder(dummy, sReg, sReg, miWaveGroupCoal, tmpVgprRes, tmpSgprInfo))
            module.add(VCmpEQU32(dst=sgpr(tmpSgpr,writer.states.laneSGPRCount), src0=vgpr(sReg), src1=vgpr(wReg), comment="wave_id == block_belong_to_wave?" ))
            module.add(VCndMaskB32(dst=vgpr(wgMT), src0=vgpr(mtReg), src1=vgpr(wgMT), src2=sgpr(tmpSgpr,writer.states.laneSGPRCount), comment="wgMT = (wgMT < MT) ? wgMT : MT" ))
            writer.vgprPool.checkIn(mtReg)
            writer.vgprPool.checkIn(sReg)

            # mbReg: which mb block meed to shift, mb(matrixInstM*VectorWidth)
            module.addComment1("mbReg: which mb block need to shift, mb(matrixInstCoal(%u) * VectorWidth(%u))" % (matrixInstCoal, vectorWidth))
            mbReg = writer.vgprPool.checkOut(1)
            tReg  = writer.vgprPool.checkOut(1)
            module.add(vectorStaticDivide(mbReg, wgMT, subMBShapeCoal, tmpVgprRes))
            module.add(vectorStaticMultiply(vgpr(tReg), vgpr(wReg), (matrixInstBCoal * OutBlocksInMI), tmpSgprInfo))
            module.add(VSubU32(dst=vgpr(mbReg), src0=vgpr(mbReg), src1=vgpr(tReg)))
            writer.vgprPool.checkIn(tReg)

            # gbReg: glvw block id
            module.addComment1("gbReg: glvw block id")
            gbReg = writer.vgprPool.checkOut(1)
            module.add(vectorStaticDivide(gbReg, wgMT, glvw, tmpVgprRes))

            # tgbReg: thread in glvw block
            module.addComment1("tgbReg: glvw block id")
            tgbReg = writer.vgprPool.checkOut(1)
            module.add(vectorStaticDivide(tgbReg, "Serial", threadInterval, tmpVgprRes))
            module.add(vectorStaticRemainder(dummy, tgbReg, tgbReg, numThreadInCoal, tmpVgprRes, tmpSgprInfo))
            module.add(vectorStaticMultiply(vgpr(tgbReg), vgpr(tgbReg), allContOutCoal, tmpSgprInfo))
            module.add(vectorStaticDivide(tgbReg, tgbReg, glvw, tmpVgprRes))
            module.add(vectorStaticMultiply(vgpr(wReg), vgpr(wReg), MIBShapeCoal//glvw, tmpSgprInfo))
            module.add(VAddCOU32(dst=vgpr(tgbReg), dst1=VCC(), src0=vgpr(wReg), src1=vgpr(tgbReg), comment="tgbReg = (tid_coal * continOut) / GLVW"))
            module.add(VSubU32(dst=vgpr(gbReg), src0=vgpr(gbReg), src1=vgpr(tgbReg)))
            writer.vgprPool.checkIn(wReg)
            writer.vgprPool.checkIn(tgbReg)

            # vw block of glvw
            module.addComment1("vwReg: glvw in which vw block?")
            vwReg = writer.vgprPool.checkOut(1)
            module.add(VAndB32(dst=vgpr(vwReg), src0=allContOutCoal-1, src1=vgpr(wgMT), comment="permute register between threads"))
            module.add(VLShiftRightB32(dst=vgpr(vwReg), shiftHex=log2(glvw), src=vgpr(vwReg), comment="permute register between threads"))

            shiftVectorComponentsPlaceHolders   = []
            shiftVectorComponentsPlaceTargets    = []
            shiftVectorComponentsPlaceOperations = []
            # rReg : reminder of M_size % vectorwidth
            # decide to jump to block which handle this case, M_size % vector width
            module.addComment1("rReg : reminder of M_size % GlobalReadVectorWidth")
            rReg = writer.vgprPool.checkOut(1)
            module.add(vectorStaticRemainder(dummy, rReg, wgMT, glvw, tmpVgprRes, tmpSgprInfo))
            for r in range(1, glvw):
                module.add(VCmpEQU32(dst=VCC(), src0=vgpr(rReg), src1=hex(r), comment="wgMT%%VW == %u"%r ))
                module.add(SCBranchVCCNZ(labelName=glvwLabels[(r-1)].getLabelName(), comment="branch to shift d%u r=%u"%(tP["idx"], r)))

            label_name = glvwLabels[glvw-1].getLabelName().replace("label_", "")
            place_holder = "%d_%s_placeholder"%(len(shiftVectorComponentsPlaceHolders), label_name)
            shiftVectorComponentsPlaceHolders.append(place_holder)
            shiftVectorComponentsPlaceTargets.append(label_name)
            shiftVectorComponentsPlaceOperations.append("SBranch")
            shiftVectorComponentsModel = Module(place_holder)
            shiftVectorComponentsModel.addComment1("no shifting")
            module.add(shiftVectorComponentsModel)
            writer.vgprPool.checkIn(rReg)

            _, arch2acc = accToArchMapper(kernel)

            # blocks for handle M_size % vector width
            for r in range(1, glvw):
                module.addComment2("shift d%u r=%u"%(tP["idx"], r))
                module.add(glvwLabels[r-1])
                for tt in range(0, miOuterTTCoal):
                    for bm in range(0, matrixInstBCoal):
                        for ob in range(0, OutBlocksInMI):
                            label  = ob + OutBlocksInMI * (bm + matrixInstBCoal * tt)
                            target = ob + OutBlocksInMI * (bm + matrixInstBCoal * miWaveGroupCoal * tt)
                            if target > 64:
                                src1 = sgpr(tmpSgprInfo.idx)
                                module.add(SMovB32(dst=src1, src=hex(target), comment="large mb"))
                            else:
                                src1 = hex(target)
                            module.add(VCmpEQU32(dst=VCC(), src0=vgpr(mbReg), src1=src1))
                            label_name = MBblockLabels[r-1][label].getLabelName().replace("label_", "")
                            place_holder = "%d_%s_placeholder"%(len(shiftVectorComponentsPlaceHolders), label_name)
                            shiftVectorComponentsPlaceHolders.append(place_holder)
                            shiftVectorComponentsPlaceTargets.append(label_name)
                            shiftVectorComponentsPlaceOperations.append("SCBranchVCCNZ")
                            shiftVectorComponentsModel = Module(place_holder)
                            shiftVectorComponentsModel.addComment1("branch to shift d%u r%u mb%u" % (tP["idx"], r, label))
                            module.add(shiftVectorComponentsModel)

            for r in range(1, glvw):
                for mb in range(0, miOuterTTCoal * matrixInstBCoal * OutBlocksInMI):
                    module.addComment2("shift d%u r=%u mb=%u"%(tP["idx"], r, mb))
                    MBblockLabels[r-1][mb].comment = "r%u mb%u"%(r, mb)
                    module.add(MBblockLabels[r-1][mb])
                    for vw in range(0, max(1, allContOutCoal//glvw)):
                        module.add(VCmpEQU32(dst=VCC(), src0=vgpr(vwReg), src1=hex(vw)))
                        label_name = VWBlockLabels[r-1][mb][vw].getLabelName().replace("label_", "")
                        place_holder = "%d_%s_placeholder"%(len(shiftVectorComponentsPlaceHolders), label_name)
                        shiftVectorComponentsPlaceHolders.append(place_holder)
                        shiftVectorComponentsPlaceTargets.append(label_name)
                        shiftVectorComponentsPlaceOperations.append("SCBranchVCCNZ")
                        shiftVectorComponentsModel = Module(place_holder)
                        shiftVectorComponentsModel.addComment1("branch to shift d%u r%u mb%u vw%u" % (tP["idx"], r, mb, vw))
                        module.add(shiftVectorComponentsModel)

            # blocks for handle M_size % vector width
            tReg  = writer.vgprPool.checkOut(min(glvw, allContOutCoal))
            for r in range(1, glvw):
                for tt in range(0, miOuterTTCoal):
                    for bm in range(0, matrixInstBCoal):
                        for ob in range(0, OutBlocksInMI):
                            mb = ob + OutBlocksInMI * (bm + matrixInstBCoal * tt)
                            for vw in range(0, max(1, allContOutCoal//glvw)):
                                module.addComment2("shift d%u r=%u mb=%u vw%d"%(tP["idx"], r, mb, vw))
                                VWBlockLabels[r-1][mb][vw].comment = "r%u mb%u vw%u"%(r, mb, vw)
                                module.add(VWBlockLabels[r-1][mb][vw])
                                module.add(SMovB32(dst=sgpr(tmpSgpr), src=(((ob*subMBShapeCoal + bm*MBShapeCoal + tt*WGShapeCoal) // glvw) + vw)))
                                module.add(VCmpXEqU32(dst=sgpr(tmpSgpr, writer.states.laneSGPRCount), src0=vgpr(gbReg), src1=sgpr(tmpSgpr), comment="is thread in edge glvw region" ))
                                module.add(VAndB32(dst=vgpr(tmpVgpr), src0=kernel["WavefrontSize"]-1, src1=vgpr("Serial"), comment="permute register between threads"))
                                module.add(VLShiftLeftB32(dst=vgpr(tmpVgpr), shiftHex=log2(writer.states.bpr), src=vgpr(tmpVgpr), comment="permute register between threads"))

                                for ot in range(numOutputsPrep):
                                    for c  in range(complexMultiplier):
                                        for nr in range(regPerElem):
                                            vgprOffsetForSCIU = 0
                                            for e in range(min(r, allContOutCoal)):
                                                src = (e+(glvw-r)) % allContOutCoal
                                                srcVgpr = (src + (vw * glvw) + allContOutCoal * mb) * regStrideCoal
                                                srcVgpr = srcVgpr + ot * regStridePrep
                                                srcVgpr = arch2acc[srcVgpr] * regPerElem + nr + c * accImOffset + vgprOffsetForSCIU
                                                srcVal  = writer.accVgprReadWriteIndex(kernel, srcVgpr)
                                                copyInst = writer.accVgprReadWriteFunction(kernel, srcVgpr, True)
                                                module.add(copyInst(dst=vgpr(tReg+e), src=srcVal, comment="glvw %u mb %u tt1 %u r %u" % (r, mb, ot, nr)))

                                            if not kernel["MIArchVgpr"]:
                                                module.add(SNop(waitState=1, comment="v_accvgpr read vgpr after write vgpr: 2 wait states"))

                                            needWait = False
                                            for e in range(min(r, allContOutCoal)):
                                                crossThread = (e+(glvw-r)) // allContOutCoal
                                                if crossThread != 0:
                                                    ds = DSModifiers(na=1, offset=crossThread*threadInterval*4)
                                                    module.add(DSBPermuteB32(dst=vgpr(tReg+e), src0=vgpr(tmpVgpr), src1=vgpr(tReg+e), ds=ds, comment="permute edge values"))
                                                    needWait = True

                                            if needWait:
                                                module.add(SWaitCnt(waitAll=True, comment="wait for swizzle operation"))

                                            for e in range(min(r, allContOutCoal)):
                                                dstVgpr = (e + (vw * glvw) + allContOutCoal * mb) * regStrideCoal
                                                dstVgpr = dstVgpr + ot * regStridePrep
                                                dstVgpr = arch2acc[dstVgpr] * regPerElem + nr + c * accImOffset + vgprOffsetForSCIU
                                                dstStr = writer.accVgprReadWriteIndex(kernel, dstVgpr)
                                                copyInst = writer.accVgprReadWriteFunction(kernel, dstVgpr, False)
                                                module.add(copyInst(dst=dstStr, src=vgpr(tReg+e)))

                                # end shift reset mask and jump out
                                all1mask = "0xFFFFFFFF" if (kernel["WavefrontSize"] == 32) else "0xFFFFFFFFFFFFFFFF"
                                SMovBX = SMovB64 if kernel["WavefrontSize"] == 64 else SMovB32
                                SOrSaveExecBX = SOrSaveExecB64 if kernel["WavefrontSize"] == 64 else SOrSaveExecB32
                                module.add(SMovBX(dst=sgpr(tmpSgpr, writer.states.laneSGPRCount), src=all1mask, comment="to restore all threads active"))
                                module.add(SOrSaveExecBX(dst=VCC(), src=sgpr(tmpSgpr,writer.states.laneSGPRCount), comment="all threads active"))
                                label_name = glvwLabels[glvw-1].getLabelName().replace("label_", "")
                                place_holder = "%d_%s_placeholder"%(len(shiftVectorComponentsPlaceHolders), label_name)
                                shiftVectorComponentsPlaceHolders.append(place_holder)
                                shiftVectorComponentsPlaceTargets.append(label_name)
                                shiftVectorComponentsPlaceOperations.append("SBranch")
                                shiftVectorComponentsModel = Module(place_holder)
                                shiftVectorComponentsModel.addComment1("no shifting")
                                module.add(shiftVectorComponentsModel)
                                module.addSpaceLine()

            module.add(glvwLabels[glvw-1])
            writer.vgprPool.checkIn(tReg)

            # checkin scratch vgprs
            writer.vgprPool.checkIn(tmpVgpr)
            writer.vgprPool.checkIn(wgMT)
            writer.vgprPool.checkIn(dummy)
            writer.vgprPool.checkIn(gbReg)
            writer.vgprPool.checkIn(vwReg)
            writer.vgprPool.checkIn(mbReg)
            writer.updateBranchPlaceHolder(module, shiftVectorComponentsPlaceHolders, shiftVectorComponentsPlaceTargets, shiftVectorComponentsPlaceOperations)
        return module

    def ShiftVectorComponentsMFMAAllThread(self, writer, kernel, tP):
        """
        """
        # common parameter
        tc              = tP["tensorChar"]
        glvw            = tP["glvw"]
        numThreadInWave = kernel["WavefrontSize"]
        vectorWidth     = kernel["VectorWidth%s"%tc]

        # use to handle MatrixInst 4x4
        matrixInstM     = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
        matrixInstN     = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]
        matrixInstBM    = 1 if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBM"]
        matrixInstBN    = 1 if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstBN"]

        # unify process for dimension M/N
        matrixInstCoal  = matrixInstM              if tP["isA"] else matrixInstN
        matrixInstPrep  = matrixInstN              if tP["isA"] else matrixInstM
        matrixInstBCoal = matrixInstBM             if tP["isA"] else matrixInstBN
        matrixInstBPrep = matrixInstBN             if tP["isA"] else matrixInstBM
        miWaveGroupCoal = kernel["MIWaveGroup"][0] if tP["isA"] else kernel["MIWaveGroup"][1]
        miWGIdStride    = numThreadInWave          if tP["isA"] else (numThreadInWave * kernel["MIWaveGroup"][0])
        miWaveTitleCoal = kernel["MIWaveTile"][0]  if tP["isA"] else kernel["MIWaveTile"][1]
        miWaveTitlePrep = kernel["MIWaveTile"][1]  if tP["isA"] else kernel["MIWaveTile"][0]

        # unify process for SourceSwap and non-SourceSwap
        conThInProcDim  = kernel["SourceSwap"] ^ tP["isB"] # continuous threads in processed dimension(Coalesced dimension)

        threadInterval  = 1 if conThInProcDim else matrixInstPrep
        numThreadInCoal = matrixInstCoal if conThInProcDim else (numThreadInWave // matrixInstPrep)

        #numContOutCoal  = 1 if conThInProcDim else kernel["MIOutputVectorWidth"]
        #allContOutCoal  = numContOutCoal * vectorWidth
        numContOutCoal  = vectorWidth if conThInProcDim else kernel["MIOutputVectorWidth"] * vectorWidth
        allContOutCoal  = numContOutCoal

        #OutBlocksInMI   = (matrixInstCoal * matrixInstPrep) // numThreadInWave // numContOutCoal
        OutBlocksInMI   = (vectorWidth * matrixInstCoal * matrixInstPrep) // numThreadInWave // numContOutCoal
        OutBlocksInMI   = 1 if conThInProcDim else OutBlocksInMI

        subMBShapeCoal  = (matrixInstCoal * vectorWidth) if conThInProcDim else ((numThreadInWave // matrixInstPrep) * numContOutCoal)
        MBShapeCoal     = subMBShapeCoal * OutBlocksInMI
        MIBShapeCoal    = MBShapeCoal * matrixInstBCoal
        miOuterTTCoal   = miWaveTitleCoal // vectorWidth

        numOutputsPrep  = (matrixInstCoal * matrixInstPrep // numThreadInWave) if conThInProcDim else 1
        numOutputsPrep  = numOutputsPrep * matrixInstBPrep * miWaveTitlePrep

        # unify process for dimension M/N
        regStrideCoal = 1                                                                if tP["isA"] else numOutputsPrep
        regStridePrep = miOuterTTCoal * matrixInstBCoal * OutBlocksInMI * allContOutCoal if tP["isA"] else 1

        # labels for shiftptr
        shiftLabels = []
        GLVWBLKLabels = []
        for i in range(0, glvw): # grvw block
            r = (i+1) % glvw    # r = [1,2,3,...,glvw-1, 0], the last one shiftLabels[glvw-1] stores for r=0 -> no shift
            label = Label(writer.labels.getName("ShiftVectorComponents%u_shift%u" % (tP["idx"], r) ), "")
            shiftLabels.append(label)
            subGLVWBLKLabels = []
            for glvwBlk in range(0, (MIBShapeCoal // glvw) * miOuterTTCoal):
                label = Label(writer.labels.getName("ShiftVectorComponents%u_shift%u_glvwblk%u" % (tP["idx"], r, glvwBlk)), "")
                subGLVWBLKLabels.append(label)
            GLVWBLKLabels.append(subGLVWBLKLabels)

        with writer.allocTmpSgpr(writer.states.laneSGPRCount) as tmpSgprInfo:
            # wgMT value
            tmpSgpr = tmpSgprInfo.idx
            tmpVgpr = writer.vgprPool.checkOutAligned(2,2)
            tmpVgprRes = ContinuousRegister(tmpVgpr, 2)
            dummy   = writer.vgprPool.checkOut(1)
            wgMT    = writer.vgprPool.checkOut(1)
            wg      = tP["wg"]

            module = Module("ShiftVectorComponentsMFMA")

            # get M size of edge block
            module.addComment1("check which macro tile need to shift")
            mtReg = writer.vgprPool.checkOut(1)
            module.add(VMovB32(dst=vgpr(wgMT), src=sgpr(wg)))
            module.add(VMulI32I24(dst=vgpr(wgMT), src0=hex(-kernel[tP["mt"]]), src1=vgpr(wgMT), comment="wg*MT"))
            module.add(VAddCOU32(dst=vgpr(wgMT), dst1=VCC(), src0=sgpr("SizesFree+%u"%tP["idx"]), src1=vgpr(wgMT), comment="wgMT = Size - wg*MT"))
            module.add(VMovB32(dst=vgpr(mtReg), src=hex(kernel[tP["mt"]]), comment="MT"))
            module.add(VCmpLtU32(dst=sgpr(tmpSgpr,writer.states.laneSGPRCount), src0=vgpr(wgMT), src1=vgpr(mtReg), comment="wgMT < MT"))
            module.add(VCndMaskB32(dst=vgpr(wgMT), src0=vgpr(mtReg), src1=vgpr(wgMT), src2=sgpr(tmpSgpr,writer.states.laneSGPRCount), comment="wgMT = (wgMT < MT) ? wgMT : MT" ))
            module.addSpaceLine()

            # identify which wave have to process
            module.addComment1("check which wave need to shift")
            wReg = writer.vgprPool.checkOut(1)
            sReg = writer.vgprPool.checkOut(1)
            module.add(vectorStaticDivide(wReg, "Serial", miWGIdStride, tmpVgprRes))
            module.add(vectorStaticRemainder(dummy, wReg, wReg, miWaveGroupCoal, tmpVgprRes, tmpSgprInfo))
            module.add(vectorStaticDivide(sReg, wgMT, MIBShapeCoal, tmpVgprRes))
            module.add(vectorStaticRemainder(dummy, sReg, sReg, miWaveGroupCoal, tmpVgprRes, tmpSgprInfo))
            module.add(VCmpEQU32(dst=sgpr(tmpSgpr,writer.states.laneSGPRCount), src0=vgpr(sReg), src1=vgpr(wReg), comment="wave_id == block_belong_to_wave?" ))
            module.add(VCndMaskB32(dst=vgpr(wgMT), src0=vgpr(mtReg), src1=vgpr(wgMT), src2=sgpr(tmpSgpr,writer.states.laneSGPRCount), comment="wgMT = (wgMT < MT) ? wgMT : MT" ))
            module.addSpaceLine()

            # glveblkid
            module.addComment1("get id of which glvw block need to shift")
            glvwblkidReg = writer.vgprPool.checkOut(1)
            module.add(VMulI32I24(dst=vgpr(glvwblkidReg), src0=hex(-MIBShapeCoal), src1=vgpr(wReg), comment="wg * MIB"))
            module.add(VAddCOU32(dst=vgpr(glvwblkidReg), dst1=VCC(), src0=vgpr(glvwblkidReg), src1=vgpr(wgMT), comment="wgMT = Size - wg*MIB"))
            module.add(vectorStaticDivide(glvwblkidReg, glvwblkidReg, glvw, tmpVgprRes, "glvw block id"))
            module.addSpaceLine()

            # rReg : reminder of M_size % vectorwidth
            # decide to jump to block which handle this case, M_size % vector width
            module.addComment1("dispatch to differet shift block for shift")
            rReg = writer.vgprPool.checkOut(1)
            module.add(vectorStaticRemainder(dummy, rReg, wgMT, glvw, tmpVgprRes, tmpSgprInfo))
            for r in range(1, glvw):
                module.add(VCmpEQU32(dst=VCC(), src0=vgpr(rReg), src1=hex(r), comment="wgMT%%VW == %u"%r ))
                module.add(SCBranchVCCNZ(labelName=shiftLabels[(r-1)].getLabelName(), comment="branch to shift d%u r=%u"%(tP["idx"], r)))
            module.add(SBranch(labelName=shiftLabels[(glvw-1)].getLabelName(), comment="no shifting"))
            writer.vgprPool.checkIn(rReg)

        # blocks for handle M_size % vector width
            glvwBlkInMIB = MIBShapeCoal // glvw
            numRegInGlvwblkCoal = glvw // numThreadInCoal
            numRegInMIBCoal = MIBShapeCoal // numThreadInCoal
            for r in range(1, glvw):
                module.addComment2("shift d%u shift=%u"%(tP["idx"], r))
                module.add(shiftLabels[r-1])
                for tt in range(0, miOuterTTCoal):
                    for glvwBlk in range(0, glvwBlkInMIB):
                        label  = glvwBlk + tt * glvwBlkInMIB
                        target = glvwBlk + tt * glvwBlkInMIB * miWaveGroupCoal
                        module.add(VCmpEQU32(dst=VCC(), src0=vgpr(glvwblkidReg), src1=hex(target)))
                        module.add(SCBranchVCCNZ(labelName=GLVWBLKLabels[r-1][label].getLabelName(), comment="branch to shift d%u shift%u glvwblk%u" % (tP["idx"], r, label)))

            _, arch2acc = accToArchMapper(kernel)

            permuteIndexReg = writer.vgprPool.checkOut(1)
            threadIdInCoalReg = writer.vgprPool.checkOut(1)
            movReg = writer.vgprPool.checkOut(numContOutCoal*numOutputsPrep)
            module.addComment2("Reg %d-%d"%(movReg, movReg+15))
            for shift in range(1, glvw):
                for tt in range(0, miOuterTTCoal):
                    for glvwBlk in range(0, glvwBlkInMIB):
                        label = glvwBlk + tt * glvwBlkInMIB
                        module.addComment2("shift d%u shift=%u glvwblk=%u"%(tP["idx"], shift, glvwBlk))
                        module.add(GLVWBLKLabels[shift-1][label])
                        module.add(VAndB32(dst=vgpr(permuteIndexReg), src0=kernel["WavefrontSize"]-1, src1=vgpr("Serial"), comment="permute register between threads"))
                        module.add(vectorStaticMultiply(vgpr(permuteIndexReg), vgpr(permuteIndexReg), writer.states.bpr, tmpSgprInfo, comment="permute register between threads"))
                        module.add(vectorStaticDivide(threadIdInCoalReg, "Serial", threadInterval, tmpVgprRes))
                        module.add(vectorStaticRemainder(dummy, threadIdInCoalReg, threadIdInCoalReg, numThreadInCoal, tmpVgprRes, tmpSgprInfo))

                        for dstMbblkId in range(glvw//(numContOutCoal*numThreadInCoal)):
                            for dstThreadId in range(numThreadInCoal):
                                skip = True
                                for dstContId in range(numContOutCoal):
                                    dst = dstContId + dstThreadId * numContOutCoal + dstMbblkId * numThreadInCoal * numContOutCoal
                                    src = dst + (glvw - shift)
                                    if (src < glvw):
                                        skip = False
                                        srcContId   = src % numContOutCoal
                                        srcThreadId = (src // numContOutCoal) % numThreadInCoal
                                        srcMbblkId  = src // (numContOutCoal * numThreadInCoal)
                                        for ot in range(numOutputsPrep):
                                            movRegId    = movReg + dstContId + ot * numContOutCoal
                                            srcGpr      = srcContId + srcMbblkId * numContOutCoal + glvwBlk * numRegInGlvwblkCoal + tt * numRegInMIBCoal
                                            srcGpr      = srcGpr * regStrideCoal + ot * regStridePrep
                                            srcGpr      = arch2acc[srcGpr]
                                            srcGprStr   = writer.accVgprReadWriteIndex(kernel, srcGpr)
                                            copyInst = copyInst = writer.accVgprReadWriteFunction(kernel, srcGpr, True)
                                            module.add(copyInst(dst=vgpr(movRegId), src=srcGprStr, comment=""))

                                if not skip:
                                     module.add(SNop(waitState=1, comment="v_accvgpr read vgpr after write vgpr: 2 wait states"))

                                needWait = False
                                for dstContId in range(numContOutCoal):
                                    dst = dstContId + dstThreadId * numContOutCoal + dstMbblkId * numThreadInCoal * numContOutCoal
                                    src = dst + (glvw - shift)
                                    if (src < glvw):
                                        srcContId   = src % numContOutCoal
                                        srcThreadId = (src // numContOutCoal) % numThreadInCoal
                                        srcMbblkId  = src // (numContOutCoal * numThreadInCoal)
                                        srcGpr      = srcContId + srcMbblkId * numContOutCoal
                                        if dstThreadId != srcThreadId:
                                            needWait = True
                                            permuteOffset = (((srcThreadId - dstThreadId) * threadInterval) % kernel["WavefrontSize"]) * 4
                                            for ot in range(numOutputsPrep):
                                                movRegId    = movReg + dstContId + ot * numContOutCoal
                                                ds = DSModifiers(na=1, offset=permuteOffset)
                                                module.add(DSBPermuteB32(dst=vgpr(movRegId), src0=vgpr(permuteIndexReg), src1=vgpr(movRegId), ds=ds, comment="permute edge values"))

                                if needWait:
                                    module.add(SWaitCnt(dscnt=0, comment="wait for swizzle operation"))

                                if not skip:
                                    module.add(SMovB32(dst=sgpr(tmpSgpr), src=dstThreadId, comment="which thread need to shfit in this block"))
                                    module.add(VCmpXEqU32(dst=sgpr(tmpSgpr, writer.states.laneSGPRCount), src0=vgpr(threadIdInCoalReg), src1=sgpr(tmpSgpr), comment="is thread in edge glvw region"))
                                    module.add(SNop(waitState=3, comment="wait for exec mask"))

                                for dstContId in range(numContOutCoal):
                                    dst = dstContId + dstThreadId * numContOutCoal + dstMbblkId * numThreadInCoal * numContOutCoal
                                    src = dst + (glvw - shift)
                                    if (src < glvw):
                                        for ot in range(numOutputsPrep):
                                            movRegId    = movReg + dstContId + ot * numContOutCoal
                                            dstGpr      = dstContId + dstMbblkId * numContOutCoal + glvwBlk * numRegInGlvwblkCoal + tt * numRegInMIBCoal
                                            dstGpr      = dstGpr * regStrideCoal + ot * regStridePrep
                                            dstGpr      = arch2acc[dstGpr]
                                            dstGprStr   = writer.accVgprReadWriteIndex(kernel, dstGpr)
                                            copyInst = writer.accVgprReadWriteFunction(kernel, dstGpr, False)
                                            module.add(copyInst(dst=dstGprStr, src=vgpr(movRegId), comment=""))

                                if not skip:
                                    all1mask = "0xFFFFFFFF" if (kernel["WavefrontSize"] == 32) else "0xFFFFFFFFFFFFFFFF"
                                    SMovBX = SMovB64 if kernel["WavefrontSize"] == 64 else SMovB32
                                    SOrSaveExecBX = SOrSaveExecB64 if kernel["WavefrontSize"] == 64 else SOrSaveExecB32
                                    module.add(SMovBX(dst=sgpr(tmpSgpr, writer.states.laneSGPRCount), src=all1mask, comment="to restore all threads active"))
                                    module.add(SOrSaveExecBX(dst=VCC(), src=sgpr(tmpSgpr,writer.states.laneSGPRCount), comment="all threads active"))
                                    module.add(SNop(waitState=3, comment="wait for exec mask"))

                        module.add(SBranch(labelName=shiftLabels[glvw-1].getLabelName(), comment="done"))

            module.add(shiftLabels[glvw-1])

            writer.vgprPool.checkIn(movReg)
            writer.vgprPool.checkIn(threadIdInCoalReg)
            writer.vgprPool.checkIn(permuteIndexReg)

            writer.vgprPool.checkIn(glvwblkidReg)
            writer.vgprPool.checkIn(sReg)
            writer.vgprPool.checkIn(wReg)
            writer.vgprPool.checkIn(mtReg)

            # checkin scratch vgprs
            writer.vgprPool.checkIn(tmpVgpr)
            writer.vgprPool.checkIn(dummy)
            writer.vgprPool.checkIn(wgMT)

        return module
