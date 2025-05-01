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

from ..Component import ShiftVectorComponents
from ..AsmUtils import inst, vgpr, sgpr, accvgpr, staticMultiply, vectorStaticDivide, vectorStaticRemainder, log2

class ShiftVectorComponentsVALU(ShiftVectorComponents):
    kernel = {"EnableMatrixInstruction": False}

    """
    Shift Vector Components d0,1
    """
    def __call__(self, writer, kernel, tP):
        kStr = ""

        # glvw
        vw = tP["glvw"]
        numVectors = kernel[tP["tt"]]//vw
        # labels
        svrLabels = []
        sviLabels = []
        for i in range(0, vw):
            r = (i+1) % vw
            label = writer.getLabelNum("ShiftVectorComponents%u_R%u"%(tP["idx"], r) )
            svrLabels.append(label)
            tmpLabels = []
            for v in range(0, numVectors):
                label = writer.getLabelNum("ShiftVectorComponents%u_R%u_V%u"%(tP["idx"], r, v) )
                tmpLabels.append(label)
            sviLabels.append(tmpLabels)

        wg = tP["prevWg"] if writer.prefetchAcrossPersistent else tP["wg"]
        # wgMT value
        tmpSgpr = writer.getTmpSgpr(writer.laneSGPRCount).idx()
        tmpVgpr = writer.vgprPool.checkOut(1,"tmpVgpr")
        wgMT = writer.vgprPool.checkOut(1,"wgMT")
        kStr += inst("v_mov_b32", vgpr(wgMT), sgpr(wg), "")
        kStr += inst("v_mul_i32_i24", vgpr(wgMT), hex(-kernel[tP["mt"]]), vgpr(wgMT), \
                "wg*MT")
        kStr += inst("_v_add_co_u32", vgpr(wgMT), writer.vcc, sgpr("SizesFree+%u"%tP["idx"]), \
                vgpr(wgMT), "wgMT = Size - wg*MT")
        kStr += inst("v_mov_b32", vgpr(tmpVgpr), hex(kernel[tP["mt"]]), "MT")
        kStr += inst("v_min_u32"    , vgpr(wgMT), vgpr(tmpVgpr), vgpr(wgMT), "wgMT = (wgMT < MT) ? wgMT : MT" )
        writer.vgprPool.checkIn(tmpVgpr)

        # qReg
        qReg = writer.vgprPool.checkOut(1,"qReg")
        divisor = kernel["VectorWidth"] # vw
        kStr += vectorStaticDivide(qReg, wgMT, divisor, tmpSgpr)

        # rReg
        rReg = writer.vgprPool.checkOut(1,"rReg")
        divisor = vw
        kStr += vectorStaticRemainder(rReg, wgMT, divisor, tmpSgpr)

        # qReg %/ SG
        eReg = writer.vgprPool.checkOut(1,"eReg")
        divisor = kernel[tP["sg"]]
        kStr += vectorStaticRemainder(eReg, qReg, divisor, tmpSgpr)

        if tP["isA"]:
            # thread = serial % SG0
            thread = writer.vgprPool.checkOut(1,"thread")
            divisor = kernel["SubGroup0"]
            kStr += vectorStaticRemainder(thread, "Serial", divisor, tmpSgpr)
            #kStr += dump(vgpr(thread))
            #kStr += dump(vgpr(thread))
        else:
            # thread = (serial / SG0) % SG1
            sd0 = writer.vgprPool.checkOut(1,"sd0")
            divisor = kernel["SubGroup0"]
            kStr += vectorStaticDivide(sd0, "Serial", divisor, tmpSgpr) # thread = serial / SG0
            divisor = kernel["SubGroup1"]
            thread = writer.vgprPool.checkOut(1,"thread")
            kStr += vectorStaticRemainder(thread, sd0, divisor, tmpSgpr) # thread = (serial / SG0) % SG1
            writer.vgprPool.checkIn(sd0)

        # which glvw vector of thread to shift? wgMT / (SG0*VW) -> (wgMT%VW) / glvw
        # (wgMT/(WG0*VW))*(VW/glvw) + (wgMT%VW) / glvw
        if True:#tP["tensorIdx"] > kernel["VectorWidth"]:
            mvReg = writer.vgprPool.checkOut(1,"mvReg")
            divisor = kernel[tP["sg"]]*kernel["VectorWidth"]
            kStr += vectorStaticDivide(mvReg, wgMT, divisor, tmpSgpr)
            if vw < kernel["VectorWidth"]:
                kStr += inst("v_lshlrev_b32", vgpr(mvReg), hex(log2(kernel["VectorWidth"]//vw)), vgpr(mvReg), "vId *= VW/glvw")
        #kStr += dump(vgpr(mvReg))

        vReg = writer.vgprPool.checkOut(1,"vReg")
        divisor = kernel["VectorWidth"]
        kStr += vectorStaticRemainder(vReg, wgMT, divisor, tmpSgpr)
        vRegD = writer.vgprPool.checkOut(1,"vRegD")
        kStr += inst("v_mov_b32", vgpr(vRegD), vgpr(vReg), "duplicate")
        divisor = vw
        kStr += vectorStaticDivide(vReg, vRegD, divisor, tmpSgpr)
        #kStr += dump(vgpr(vReg))

        if True:#tP["tensorIdx"] > kernel["VectorWidth"]:
            kStr += inst("_v_add_co_u32", vgpr(vReg), writer.vcc, vgpr(mvReg), vgpr(vReg), "vId = 2 components")
            writer.vgprPool.checkIn(mvReg)
            writer.vgprPool.checkIn(vRegD)

        # for each remainder, jump
        for r in range(1, vw):
            kStr += inst("v_cmp_eq_u32", writer.vcc, vgpr(rReg), \
                    hex(r), "wgMT%%VW == %u"%r )
            kStr += inst("s_cbranch_vccnz label_%04u"\
                    % svrLabels[(r-1)%vw], \
                    "shift d%u r=%u"%(tP["idx"], r))
            #kStr += inst("s_mov_b32", sgpr(sgprLoc), hex(location), \
            #        "location=%u"%location) location *= 2
            #kStr += inst("v_or_b32", vgpr(vgprPath), sgpr(sgprLoc), \
            #        vgpr(vgprPath), "path+=location")
        kStr += inst("s_branch label_%04u"%svrLabels[vw-1], \
                "no shifting" )

        # code blocks for shifting
        for r in range(1, vw):
            kStr += writer.comment3("shift d%u r=%u"%(tP["idx"], r))
            kStr += "label_%04u:%s" % (svrLabels[r-1], writer.endLine)
            #if r==3 and tP["isA"]:
            #    kStr += dump(vgpr("Serial"))

            # for each vector index, jump
            for vectorIdx in range(0, numVectors):
                kStr += inst("v_cmp_eq_u32", writer.vcc, vgpr(vReg), \
                        hex(vectorIdx), "wgMT/(SG*VW) == %u"%vectorIdx )
                kStr += inst("s_cbranch_vccnz label_%04u"\
                        % sviLabels[(r-1)%vw][vectorIdx], \
                        "shift d%u, r=%u, v=%u"%(tP["idx"], r, vectorIdx))

            # code blocks for shifting vector
            for vectorIdx in range(0, numVectors):
                kStr += writer.comment("shift d%u r=%u v=%u"%(tP["idx"], r, vectorIdx))
                kStr += "label_%04u:%s" % (sviLabels[r-1][vectorIdx], writer.endLine)
                # mask if last thread in thread#-tile column
                kStr += inst("_v_cmpx_eq_u32", sgpr(tmpSgpr,writer.laneSGPRCount), vgpr(thread), \
                    vgpr(eReg), "serial % SG == (wgMT/VECTOR_WIDTH)%SG" )
                tto = kernel["ThreadTile%u"%((tP["idx"]+1)%2)] # thread tile orthogonal
                for tt in range(0, tto):
                    for s in range(0, r):
                        if tP["isA"]: # shift d0
                            dst = (s) \
                                + vectorIdx * vw + tt * kernel["ThreadTile0"]
                            src = (s+vw-r) \
                                + vectorIdx * vw + tt * kernel["ThreadTile0"]
                            comment = "rC[%u+%u*VW+%u*TT%s] = rC[%u+%u*VW+%u*TT%s]" \
                                % (s, vectorIdx, tt, writer.tileChar0, \
                                s+vw-r, vectorIdx, tt, writer.tileChar0 )
                        else: # shift d1
                            dst = (tt) \
                                + vectorIdx*vw*kernel["ThreadTile0"] + s * kernel["ThreadTile0"]
                            src = (tt) \
                                + vectorIdx * vw*kernel["ThreadTile0"] + (s+vw-r) * kernel["ThreadTile0"]
                            comment = "rC[%u+%u*TT%s*VW+%u*TT%s] = rC[%u+%u*TT%s*VW+%u*TT%s]" \
                                % (tt, vectorIdx, writer.tileChar0, s, writer.tileChar0, \
                                tt, vectorIdx, writer.tileChar0, \
                                s+vw-r, writer.tileChar0)

                        kStr += "// src=%u, dst=%u\n" % (src,dst)

                        # f16
                        #jgolds I think this should be bpeCinternal
                        if writer.bpeCinternal == 2:
                            srcVgpr = writer.startVgprValuC+src*writer.bpeCinternal//writer.bpr
                            dstVgpr = writer.startVgprValuC+dst*writer.bpeCinternal//writer.bpr
                            kStr += "// %u, %u, %u, %u, %u, %u\n" % (r, vectorIdx, tt, s, dst, src)
                            if tP["isA"]: # f16 d0
                                if r % 2 == 0: # even shift can use mov_b32
                                    if s % 2 == 0:
                                        kStr += inst("v_mov_b32", vgpr(dstVgpr), \
                                                vgpr(srcVgpr), comment)
                                    else:
                                        pass # above command performs two moves
                                else: # odd shift
                                    srcLo = src % 2 == 0 # even
                                    dstLo = dst % 2 == 0 # even
                                    kStr += "// srcLo=%u, dstLo=%u\n" % (srcLo,dstLo)
                                    if dstLo: # hi src to lo dst; can clobber hi bits
                                        kStr += inst("v_lshrrev_b32", vgpr(dstVgpr), \
                                                hex(16), vgpr(srcVgpr), "hi16 -> lo16")
                                    else: # dstHi; cannot clobber lo bits
                                        tmpSrcVgpr = writer.vgprPool.checkOut(1,"tmpSrcVgpr")
                                        # zero out dst hi bits -> dst
                                        kStr += inst("v_and_b32", vgpr(dstVgpr), \
                                                "0x0000FFFF", vgpr(dstVgpr), "zero out dst hi16")
                                        if srcLo: # lo src to hi dst
                                            # left shift src 16 bits -> tmpSrc
                                            kStr += inst("v_lshlrev_b32", vgpr(tmpSrcVgpr), \
                                                    hex(16), vgpr(srcVgpr), "left shift src 16 bits")
                                        else: # hi src to hi dst
                                            # zero out src lo bits -> tmpSrc
                                            kStr += inst("v_and_b32", vgpr(srcVgpr), \
                                                    "0xFFFF0000", vgpr(tmpSrcVgpr), "zero out src lo16")
                                        # dst = tmpSrc | dst
                                        kStr += inst("v_or_b32", vgpr(dstVgpr), \
                                                vgpr(tmpSrcVgpr), vgpr(dstVgpr), "dst = tmpSrc | dst")
                                        writer.vgprPool.checkIn(tmpSrcVgpr)

                            else: # f16 d1
                                if tt%2==0:
                                    kStr += inst("v_mov_b32", vgpr(dstVgpr), \
                                            vgpr(srcVgpr), comment)
                                else:
                                    pass # above shift moves two f16

                        # f32 or larger
                        else:
                            for i in range(0, writer.bpeCinternal//writer.bpr):
                                kStr += inst("v_mov_b32", vgpr(writer.startVgprValuC+dst*writer.bpeCinternal//writer.bpr+i), \
                                        vgpr(writer.startVgprValuC+src*writer.bpeCinternal//writer.bpr+i), comment)

                # end shift reset mask and jump out
                all1mask = "0xFFFFFFFF" if (kernel["WavefrontSize"] == 32) else "0xFFFFFFFFFFFFFFFF"
                kStr += inst("s_mov_b{}".format(kernel["WavefrontSize"]), sgpr(tmpSgpr,writer.laneSGPRCount), \
                        all1mask, "to restore all threads active")
                kStr += inst("s_or_saveexec_b{}".format(kernel["WavefrontSize"]), writer.vcc, sgpr(tmpSgpr,writer.laneSGPRCount), \
                        "all threads active")
                kStr += inst("s_branch label_%04u"%svrLabels[vw-1], \
                        "done shifting" )
        kStr += "label_%04u: // end shift0%s" % (svrLabels[vw-1], writer.endLine)

        # checkin scratch vgprs
        writer.vgprPool.checkIn(wgMT)
        writer.vgprPool.checkIn(qReg)
        writer.vgprPool.checkIn(rReg)
        writer.vgprPool.checkIn(eReg)
        writer.vgprPool.checkIn(thread)
        writer.vgprPool.checkIn(vReg)
        return kStr


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
        numThreadInWave = writer.kernel["WavefrontSize"]
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
        numContOutCoal  = vectorWidth if conThInProcDim else kernel["MIOutputVectorWidth"] * vectorWidth
        allContOutCoal  = numContOutCoal

        if (glvw > (allContOutCoal * numThreadInCoal)):
           return self.ShiftVectorComponentsMFMAAllThread(writer, kernel, tP, vectorWidth)
        else:
           return self.ShiftVectorComponentsMFMAPartialThread (writer, kernel, tP, vectorWidth)


    def ShiftVectorComponentsMFMAPartialThread(self, writer, kernel, tP, vectorWidth):
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
            1. check if M_size % GlobalLoadVectorWidth != 0, return if == 0
            2. decide which subgroup we need to shift, M_size(3) means 3/8 = group 0
            3. decide which thread we need to shift, we have different groups of thread, (0-31) for first group, (32-63) for second group.
            4. decide which shift block (subTile1) we want to shift. for ex [0-1], [1-2], we want to shift second subtile
        """

        kStr = ""

        # common parameter
        regPerElem      = kernel["MIRegPerOut"]
        glvw            = tP["glvw"]
        numThreadInWave = writer.kernel["WavefrontSize"]
        accImOffset     = writer.AccVgprImagNumOffset(kernel)

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
            label = writer.getLabelNum("ShiftVectorComponents%u_GLVW%u" % (tP["idx"], r) )
            glvwLabels.append(label)
            subMBLabels = []
            subVWBlockLabels = []
            for mb in range(0, OutBlocksInMI * matrixInstBCoal * miOuterTTCoal): # unit block of each thread
                label = writer.getLabelNum("ShiftVectorComponents%u_GLVW%u_BM%u" % (tP["idx"], r, mb))
                subMBLabels.append(label)
                sub2VWBlockLabels = []
                for vw in range(0, max(1, allContOutCoal//glvw)): # vw block of glvw
                    label = writer.getLabelNum("ShiftVectorComponents%u_GLVW%u_BM%u_VW%u" % (tP["idx"], r, mb, vw))
                    sub2VWBlockLabels.append(label)
                subVWBlockLabels.append(sub2VWBlockLabels)
            MBblockLabels.append(subMBLabels)
            VWBlockLabels.append(subVWBlockLabels)

        # wgMT value
        tmpSgpr = writer.getTmpSgpr(writer.laneSGPRCount).idx()
        tmpVgpr = writer.vgprPool.checkOut(1)
        wgMT    = writer.vgprPool.checkOut(1)
        wg      = tP["prevWg"] if writer.prefetchAcrossPersistent else tP["wg"]

        # get M size of edge block
        mtReg = writer.vgprPool.checkOut(1)
        kStr += inst("v_mov_b32"    , vgpr(wgMT), sgpr(wg), "")
        kStr += inst("v_mul_i32_i24", vgpr(wgMT), hex(-kernel[tP["mt"]]), vgpr(wgMT), "wg*MT")
        kStr += inst("_v_add_co_u32", vgpr(wgMT), writer.vcc, sgpr("SizesFree+%u"%tP["idx"]), vgpr(wgMT), "wgMT = Size - wg*MT")
        kStr += inst("v_mov_b32"    , vgpr(mtReg), hex(kernel[tP["mt"]]), "MT")
        kStr += inst("v_min_u32"    , vgpr(wgMT), vgpr(mtReg), vgpr(wgMT), "wgMT = (wgMT < MT) ? wgMT : MT" )

        # identify which wave have to process
        wReg = writer.vgprPool.checkOut(1)
        sReg = writer.vgprPool.checkOut(1)
        kStr += vectorStaticDivide(tmpVgpr, "Serial", miWGIdStride, tmpSgpr)
        kStr += vectorStaticRemainder(wReg, tmpVgpr, miWaveGroupCoal, tmpSgpr)
        kStr += vectorStaticDivide(tmpVgpr, wgMT, MIBShapeCoal, tmpSgpr)
        kStr += vectorStaticRemainder(sReg, tmpVgpr, miWaveGroupCoal, tmpSgpr)
        kStr += inst("v_cmp_eq_u32" , sgpr(tmpSgpr,writer.laneSGPRCount), vgpr(sReg), vgpr(wReg), "wave_id == block_belong_to_wave?" )
        kStr += inst("v_cndmask_b32", vgpr(wgMT), vgpr(mtReg), vgpr(wgMT), sgpr(tmpSgpr,writer.laneSGPRCount), "wgMT = (wgMT < MT) ? wgMT : MT" )
        writer.vgprPool.checkIn(mtReg)
        writer.vgprPool.checkIn(sReg)

        # mbReg: which mb block meed to shift, mb(matrixInstM*VectorWidth)
        kStr += writer.comment("mbReg: which mb block need to shift, mb(matrixInstCoal(%u) * VectorWidth(%u))" % (matrixInstCoal, vectorWidth))
        mbReg = writer.vgprPool.checkOut(1)
        tReg  = writer.vgprPool.checkOut(1)
        kStr += vectorStaticDivide(mbReg, wgMT, subMBShapeCoal, tmpSgpr)
        kStr += staticMultiply(vgpr(tReg), vgpr(wReg), (matrixInstBCoal * OutBlocksInMI), sgpr(tmpSgpr))
        kStr += inst("_v_sub_u32", vgpr(mbReg), vgpr(mbReg), vgpr(tReg), "")
        writer.vgprPool.checkIn(tReg)

        # gbReg: glvw block id
        kStr += writer.comment("gbReg: glvw block id")
        gbReg = writer.vgprPool.checkOut(1)
        kStr += vectorStaticDivide(gbReg, wgMT, glvw, tmpSgpr)

        # tgbReg: thread in glvw block
        kStr += writer.comment("tgbReg: glvw block id")
        tgbReg = writer.vgprPool.checkOut(1)
        kStr += vectorStaticDivide(tmpVgpr, "Serial", threadInterval, tmpSgpr)
        kStr += vectorStaticRemainder(tgbReg, tmpVgpr, numThreadInCoal, tmpSgpr)
        kStr += staticMultiply(vgpr(tgbReg), vgpr(tgbReg), allContOutCoal, sgpr(tmpSgpr))
        kStr += vectorStaticDivide(tgbReg, tgbReg, glvw, tmpSgpr)
        kStr += staticMultiply(vgpr(wReg), vgpr(wReg), MIBShapeCoal//glvw, sgpr(tmpSgpr))
        kStr += inst("_v_add_co_u32", vgpr(tgbReg), writer.vcc, vgpr(wReg), vgpr(tgbReg), "tgbReg = (tid_coal * continOut) / GLVW")
        kStr += inst("_v_sub_u32", vgpr(gbReg), vgpr(gbReg), vgpr(tgbReg), "")
        writer.vgprPool.checkIn(wReg)
        writer.vgprPool.checkIn(tgbReg)

        # vw block of glvw
        kStr += writer.comment("vwReg: glvw in which vw block?")
        vwReg = writer.vgprPool.checkOut(1)
        kStr += inst("v_and_b32", vgpr(vwReg), allContOutCoal-1, vgpr(wgMT), "permute register between threads")
        kStr += inst("v_lshrrev_b32", vgpr(vwReg), log2(glvw), vgpr(vwReg), "permute register between threads")

        # rReg : reminder of M_size % vectorwidth
        # decide to jump to block which handle this case, M_size % vector width
        kStr += writer.comment("rReg : reminder of M_size % GlobalLoadVectorWidth")
        rReg = writer.vgprPool.checkOut(1)
        kStr += vectorStaticRemainder(rReg, wgMT, glvw, tmpSgpr)
        for r in range(1, glvw):
            kStr += inst("v_cmp_eq_u32", writer.vcc, vgpr(rReg), hex(r), "wgMT%%VW == %u"%r )
            kStr += inst("s_cbranch_vccnz label_%04u" % glvwLabels[(r-1)], "branch to shift d%u r=%u"%(tP["idx"], r))
        kStr += inst("s_branch label_%04u"%glvwLabels[glvw-1], "no shifting" )
        writer.vgprPool.checkIn(rReg)

        _, arch2acc = writer.AccToArchMapper(kernel)

        # blocks for handle M_size % vector width
        for r in range(1, glvw):
            kStr += writer.comment3("shift d%u r=%u"%(tP["idx"], r))
            kStr += "label_%04u:%s" % (glvwLabels[r-1], writer.endLine)
            for tt in range(0, miOuterTTCoal):
                for bm in range(0, matrixInstBCoal):
                    for ob in range(0, OutBlocksInMI):
                        label  = ob + OutBlocksInMI * (bm + matrixInstBCoal * tt)
                        target = ob + OutBlocksInMI * (bm + matrixInstBCoal * miWaveGroupCoal * tt)
                        kStr += inst("v_cmp_eq_u32", writer.vcc, vgpr(mbReg), hex(target), "")
                        kStr += inst("s_cbranch_vccnz label_%04u" % MBblockLabels[r-1][label], "branch to shift d%u r%u mb%u" % (tP["idx"], r, label))

        for r in range(1, glvw):
            for mb in range(0, miOuterTTCoal * matrixInstBCoal * OutBlocksInMI):
                kStr += writer.comment3("shift d%u r=%u mb=%u"%(tP["idx"], r, mb))
                kStr += "label_%04u: // r%u mb%u %s" % (MBblockLabels[r-1][mb], r, mb, writer.endLine)
                for vw in range(0, max(1, allContOutCoal//glvw)):
                    kStr += inst("v_cmp_eq_u32", writer.vcc, vgpr(vwReg), hex(vw), "")
                    kStr += inst("s_cbranch_vccnz label_%04u" % VWBlockLabels[r-1][mb][vw], "branch to shift d%u r%u mb%u vw%u" % (tP["idx"], r, mb, vw))

        # blocks for handle M_size % vector width
        # no need to allocate tReg for the following scenarios
        #   - allContOutCoal==1 and kernel["MIArchVgpr"]
        #   - glvw <= allContOutCoal and glvw<=2 and kernel["MIArchVgpr"]
        needTReg = not (allContOutCoal==1 and kernel["MIArchVgpr"] or \
                        glvw <= allContOutCoal and glvw<=2 and kernel["MIArchVgpr"])
        tReg = None
        if needTReg:
          tReg  = writer.vgprPool.checkOut(min(glvw, allContOutCoal))
        for r in range(1, glvw):
            for tt in range(0, miOuterTTCoal):
                for bm in range(0, matrixInstBCoal):
                    for ob in range(0, OutBlocksInMI):
                        mb = ob + OutBlocksInMI * (bm + matrixInstBCoal * tt)
                        for vw in range(0, max(1, allContOutCoal//glvw)):
                            kStr += writer.comment3("shift d%u r=%u mb=%u vw%d"%(tP["idx"], r, mb, vw))
                            kStr += "label_%04u: // r%u mb%u vw%u %s" % (VWBlockLabels[r-1][mb][vw], r, mb, vw, writer.endLine)
                            kStr += inst("s_mov_b32", sgpr(tmpSgpr), (((ob*subMBShapeCoal + bm*MBShapeCoal + tt*WGShapeCoal) // glvw) + vw), "")
                            kStr += inst("_v_cmpx_eq_u32", sgpr(tmpSgpr, writer.laneSGPRCount), vgpr(gbReg), sgpr(tmpSgpr), "is thread in edge glvw region" )
                            kStr += inst("v_and_b32", vgpr(tmpVgpr), kernel["WavefrontSize"]-1, vgpr("Serial"), "permute register between threads")
                            kStr += inst("v_lshlrev_b32", vgpr(tmpVgpr), log2(writer.bpr), vgpr(tmpVgpr), "permute register between threads")

                            for ot in range(numOutputsPrep):
                                for c  in range(complexMultiplier):
                                    for nr in range(regPerElem):
                                        vgprOffsetForSCIU = 0
                                        if kernel["StoreCInUnroll"] and writer.enableSingleNLLOpt:
                                          # single NLL opt case, use second acc register set
                                          vgprOffsetForSCIU += writer.startaccValuC1
                                        if allContOutCoal==1 and kernel["MIArchVgpr"]:
                                          # if allContOutCoal==1, src and dest are always same
                                          # MIArchVgpr case, move is unnecessary and we can directly update srcVgpr with ds_bpermute_b32
                                          srcVgpr = ((vw * glvw) + allContOutCoal * mb) * regStrideCoal
                                          srcVgpr = srcVgpr + ot * regStridePrep
                                          srcVgpr = arch2acc[srcVgpr] * regPerElem + nr + c * accImOffset + vgprOffsetForSCIU
                                          # allContOutCoal==1 case, crossThread is always non 0 (= glvw-r)
                                          # ds_bpermute_b32 is always necessary
                                          crossThread = (glvw-r) // allContOutCoal
                                          kStr += inst("ds_bpermute_b32", vgpr(srcVgpr), vgpr(tmpVgpr), vgpr(srcVgpr), "offset:{}".format(crossThread*threadInterval*4), "permute edge values")
                                          kStr += inst("s_waitcnt", "0", "wait for swizzle operation")
                                        elif glvw <= allContOutCoal and glvw<=2 and kernel["MIArchVgpr"]:
                                          # if glvw <= allContOutCoal,
                                          #   -> r < glvw <= allContOutCoal, then, min(r, allContOutCoal) = r
                                          #   -> (e+(glvw-r)) <= r-1+(glvw-r) = glvw-1 < allContOutCoal (because glvw<=allContOutCoal)
                                          #   -> crossThread = (e+(glvw-r)) // allContOutCoal = 0 and we do not need ds_bpermute_b32
                                          # MIArchVgpr case, we do not need to move src to tReg.
                                          # Instead, we can directly move src to dst and reduce the amount of v_mov
                                          # To safely reduce v_mov, the range of e should be range(1). Otherwise, mov to dst might overwrite unused src regs
                                          #   -> glvw<=2 ensures the range of e to be range(1)
                                          copyInstStr = "v_mov_b32"
                                          for e in range(min(r, allContOutCoal)):
                                              src = (e+(glvw-r)) % allContOutCoal
                                              srcVgpr = (src + (vw * glvw) + allContOutCoal * mb) * regStrideCoal
                                              srcVgpr = srcVgpr + ot * regStridePrep
                                              srcVgpr = arch2acc[srcVgpr] * regPerElem + nr + c * accImOffset + vgprOffsetForSCIU
                                              srcStr = vgpr(srcVgpr)
                                              dstVgpr = (e + (vw * glvw) + allContOutCoal * mb) * regStrideCoal
                                              dstVgpr = dstVgpr + ot * regStridePrep
                                              dstVgpr = arch2acc[dstVgpr] * regPerElem + nr + c * accImOffset + vgprOffsetForSCIU
                                              dstStr = vgpr(dstVgpr)
                                              kStr += inst(copyInstStr, dstStr, srcStr, "glvw %u mb %u tt1 %u r %u" % (r, mb, ot, nr))
                                        else:
                                          copyInstStr = "v_accvgpr_read_b32" if not kernel["MIArchVgpr"] else "v_mov_b32"
                                          for e in range(min(r, allContOutCoal)):
                                              src = (e+(glvw-r)) % allContOutCoal
                                              srcVgpr = (src + (vw * glvw) + allContOutCoal * mb) * regStrideCoal
                                              srcVgpr = srcVgpr + ot * regStridePrep
                                              srcVgpr = arch2acc[srcVgpr] * regPerElem + nr + c * accImOffset + vgprOffsetForSCIU
                                              srcStr = accvgpr(srcVgpr) if not kernel["MIArchVgpr"] else vgpr(srcVgpr)
                                              kStr += inst(copyInstStr, vgpr(tReg+e), srcStr, "glvw %u mb %u tt1 %u r %u" % (r, mb, ot, nr))

                                          if not kernel["MIArchVgpr"]:
                                              kStr += inst("s_nop", "1", "v_accvgpr read vgpr after write vgpr: 2 wait states")

                                          needWait = False
                                          for e in range(min(r, allContOutCoal)):
                                              crossThread = (e+(glvw-r)) // allContOutCoal
                                              if crossThread != 0:
                                                  kStr += inst("ds_bpermute_b32", vgpr(tReg+e), vgpr(tmpVgpr), vgpr(tReg+e), "offset:{}".format(crossThread*threadInterval*4), "permute edge values")
                                                  needWait = True

                                          if needWait:
                                              kStr += inst("s_waitcnt", "0", "wait for swizzle operation")

                                          copyInstStr = "v_accvgpr_write_b32" if not kernel["MIArchVgpr"] else "v_mov_b32"
                                          for e in range(min(r, allContOutCoal)):
                                              dstVgpr = (e + (vw * glvw) + allContOutCoal * mb) * regStrideCoal
                                              dstVgpr = dstVgpr + ot * regStridePrep
                                              dstVgpr = arch2acc[dstVgpr] * regPerElem + nr + c * accImOffset + vgprOffsetForSCIU
                                              dstStr = accvgpr(dstVgpr) if not kernel["MIArchVgpr"] else vgpr(dstVgpr)
                                              kStr += inst(copyInstStr, dstStr, vgpr(tReg+e), "")

                            # end shift reset mask and jump out
                            all1mask = "0xFFFFFFFF" if (kernel["WavefrontSize"] == 32) else "0xFFFFFFFFFFFFFFFF"
                            kStr += inst("s_mov_b{}".format(kernel["WavefrontSize"]), sgpr(tmpSgpr, writer.laneSGPRCount), all1mask, "to restore all threads active")
                            kStr += inst("s_or_saveexec_b{}".format(kernel["WavefrontSize"]), writer.vcc, sgpr(tmpSgpr,writer.laneSGPRCount), "all threads active")
                            kStr += inst("s_branch label_%04u" % glvwLabels[glvw-1], "done shifting" )
                            kStr += writer.endLine

        kStr += "label_%04u: // end shift0%s" % (glvwLabels[glvw-1], writer.endLine)
        if tReg!= None:
          writer.vgprPool.checkIn(tReg)

        # checkin scratch vgprs
        writer.vgprPool.checkIn(tmpVgpr)
        writer.vgprPool.checkIn(wgMT)
        writer.vgprPool.checkIn(gbReg)
        writer.vgprPool.checkIn(vwReg)
        writer.vgprPool.checkIn(mbReg)

        return kStr


    def ShiftVectorComponentsMFMAAllThread(self, writer, kernel, tP, vectorWidth):
        """
        """

        kStr = ""

        # common parameter
        glvw            = tP["glvw"]
        numThreadInWave = writer.kernel["WavefrontSize"]

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
            label = writer.getLabelNum("ShiftVectorComponents%u_shift%u" % (tP["idx"], r) )
            shiftLabels.append(label)
            subGLVWBLKLabels = []
            for glvwBlk in range(0, (MIBShapeCoal // glvw) * miOuterTTCoal):
                label = writer.getLabelNum("ShiftVectorComponents%u_shift%u_glvwblk%u" % (tP["idx"], r, glvwBlk))
                subGLVWBLKLabels.append(label)
            GLVWBLKLabels.append(subGLVWBLKLabels)

        # wgMT value
        wg      = tP["prevWg"] if writer.prefetchAcrossPersistent else tP["wg"]
        tmpSgpr = writer.getTmpSgpr(writer.laneSGPRCount).idx()

        tmpVgpr = writer.vgprPool.checkOut(1)
        wgMT    = writer.vgprPool.checkOut(1)
        mtReg   = writer.vgprPool.checkOut(1)

        # get M size of edge block
        kStr += writer.comment1("check which macro tile need to shift")
        kStr += inst("v_mov_b32"    , vgpr(wgMT), sgpr(wg), "")
        kStr += inst("v_mul_i32_i24", vgpr(wgMT), hex(-kernel[tP["mt"]]), vgpr(wgMT), "wg*MT")
        kStr += inst("_v_add_co_u32", vgpr(wgMT), writer.vcc, sgpr("SizesFree+%u"%tP["idx"]), vgpr(wgMT), "wgMT = Size - wg*MT")
        kStr += inst("v_mov_b32"    , vgpr(mtReg), hex(kernel[tP["mt"]]), "MT")
        kStr += inst("v_min_u32"    , vgpr(wgMT), vgpr(mtReg), vgpr(wgMT), "wgMT = (wgMT < MT) ? wgMT : MT" )
        kStr += writer.endLine

        # identify which wave have to process
        kStr += writer.comment1("check which wave need to shift")
        wReg = writer.vgprPool.checkOut(1)
        sReg = writer.vgprPool.checkOut(1)
        kStr += vectorStaticDivide(tmpVgpr, "Serial", miWGIdStride, tmpSgpr)
        kStr += vectorStaticRemainder(wReg, tmpVgpr, miWaveGroupCoal, tmpSgpr)
        kStr += vectorStaticDivide(tmpVgpr, wgMT, MIBShapeCoal, tmpSgpr)
        kStr += vectorStaticRemainder(sReg, tmpVgpr, miWaveGroupCoal, tmpSgpr)
        kStr += inst("v_cmp_eq_u32" , sgpr(tmpSgpr,writer.laneSGPRCount), vgpr(sReg), vgpr(wReg), "wave_id == block_belong_to_wave?" )
        kStr += inst("v_cndmask_b32", vgpr(wgMT), vgpr(mtReg), vgpr(wgMT), sgpr(tmpSgpr,writer.laneSGPRCount), "wgMT = (wave_id == block_belong_to_wave) ? wgMT : MT" )
        kStr += writer.endLine

        # glveblkid
        kStr += writer.comment1("get id of which glvw block need to shift")
        glvwblkidReg = writer.vgprPool.checkOut(1)
        kStr += inst("v_mul_i32_i24", vgpr(glvwblkidReg), hex(-MIBShapeCoal), vgpr(wReg), "wg * MIB")
        kStr += inst("_v_add_co_u32", vgpr(glvwblkidReg), writer.vcc, vgpr(glvwblkidReg), vgpr(wgMT), "wgMT = Size - wg*MIB")
        kStr += vectorStaticDivide(glvwblkidReg, glvwblkidReg, glvw, tmpSgpr, "glvw block id")
        kStr += writer.endLine

        # rReg : reminder of M_size % vectorwidth
        # decide to jump to block which handle this case, M_size % vector width
        kStr += writer.comment1("dispatch to different shift block for shift")
        rReg = writer.vgprPool.checkOut(1)
        kStr += vectorStaticRemainder(rReg, wgMT, glvw, tmpSgpr)
        for r in range(1, glvw):
            kStr += inst("v_cmp_eq_u32", writer.vcc, vgpr(rReg), hex(r), "wgMT%%GLVW == %u"%r )
            kStr += inst("s_cbranch_vccnz label_%04u" % shiftLabels[(r-1)], "branch to shift d%u r=%u"%(tP["idx"], r))
        kStr += inst("s_branch label_%04u" % shiftLabels[glvw-1], "no shifting" )
        writer.vgprPool.checkIn(rReg)

        # blocks for handle M_size % vector width
        glvwBlkInMIB = MIBShapeCoal // glvw
        numRegInGlvwblkCoal = glvw // numThreadInCoal
        numRegInMIBCoal = MIBShapeCoal // numThreadInCoal
        for r in range(1, glvw):
            kStr += writer.comment3("shift d%u shift=%u"%(tP["idx"], r))
            kStr += "label_%04u:%s" % (shiftLabels[r-1], writer.endLine)
            for tt in range(0, miOuterTTCoal):
                for glvwBlk in range(0, glvwBlkInMIB):
                    label  = glvwBlk + tt * glvwBlkInMIB
                    target = glvwBlk + tt * glvwBlkInMIB * miWaveGroupCoal
                    kStr += inst("v_cmp_eq_u32", writer.vcc, vgpr(glvwblkidReg), hex(target), "")
                    kStr += inst("s_cbranch_vccnz label_%04u" % GLVWBLKLabels[r-1][label], "branch to shift d%u shift%u glvwblk%u" % (tP["idx"], r, target))

        _, arch2acc = writer.AccToArchMapper(kernel)

        permuteIndexReg = writer.vgprPool.checkOut(1)
        threadIdInCoalReg = writer.vgprPool.checkOut(1)
        movReg = writer.vgprPool.checkOut(numContOutCoal*numOutputsPrep)
        kStr += writer.comment3("Tony Reg %d-%d"%(movReg, movReg+15))
        for shift in range(1, glvw):
            for tt in range(0, miOuterTTCoal):
                for glvwBlk in range(0, glvwBlkInMIB):
                    label = glvwBlk + tt * glvwBlkInMIB
                    kStr += writer.comment3("shift d%u shift=%u glvwblk=%u"%(tP["idx"], shift, glvwBlk))
                    kStr += "label_%04u:%s" % (GLVWBLKLabels[shift-1][label], writer.endLine)
                    kStr += inst("v_and_b32", vgpr(permuteIndexReg), kernel["WavefrontSize"]-1, vgpr("Serial"), "permute register between threads")
                    kStr += staticMultiply(vgpr(permuteIndexReg), vgpr(permuteIndexReg), writer.bpr, sgpr(tmpSgpr), "permute register between threads")

                    kStr += vectorStaticDivide(tmpVgpr, "Serial", threadInterval, tmpSgpr)
                    kStr += vectorStaticRemainder(threadIdInCoalReg, tmpVgpr, numThreadInCoal, tmpSgpr)

                    for dstMbblkId in range(glvw//(numContOutCoal*numThreadInCoal)):
                        for dstThreadId in range(numThreadInCoal):
                            skip = True
                            copyInstStr = "v_accvgpr_read_b32" if not kernel["MIArchVgpr"] else "v_mov_b32"
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
                                      srcGprStr   = accvgpr(srcGpr) if not kernel["MIArchVgpr"] else vgpr(srcGpr)
                                      kStr += inst(copyInstStr, vgpr(movRegId), srcGprStr, "")

                            if not skip:
                                kStr += inst("s_nop", "1", "v_accvgpr read vgpr after write vgpr: 2 wait states")

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
                                            kStr += inst("ds_bpermute_b32", vgpr(movRegId), vgpr(permuteIndexReg), vgpr(movRegId), f"offset:{permuteOffset}", "permute edge values")

                            if needWait:
                                kStr += inst("s_waitcnt", "lgkmcnt(0)", "wait for swizzle operation")

                            if not skip:
                                kStr += inst("s_mov_b32", sgpr(tmpSgpr), dstThreadId, "which thread need to shfit in this block")
                                kStr += inst("_v_cmpx_eq_u32", sgpr(tmpSgpr, writer.laneSGPRCount), vgpr(threadIdInCoalReg), sgpr(tmpSgpr), "is thread in edge glvw region" )
                                kStr += inst("s_nop", "3", "wait for exec mask")

                            copyInstStr = "v_accvgpr_write_b32" if not kernel["MIArchVgpr"] else "v_mov_b32"
                            for dstContId in range(numContOutCoal):
                                dst = dstContId + dstThreadId * numContOutCoal + dstMbblkId * numThreadInCoal * numContOutCoal
                                src = dst + (glvw - shift)
                                if (src < glvw):
                                    for ot in range(numOutputsPrep):
                                        movRegId    = movReg + dstContId + ot * numContOutCoal
                                        dstGpr      = dstContId + dstMbblkId * numContOutCoal + glvwBlk * numRegInGlvwblkCoal + tt * numRegInMIBCoal
                                        dstGpr      = dstGpr * regStrideCoal + ot * regStridePrep
                                        dstGpr      = arch2acc[dstGpr]
                                        dstGprStr   = accvgpr(dstGpr) if not kernel["MIArchVgpr"] else vgpr(dstGpr)
                                        kStr += inst(copyInstStr, dstGprStr, vgpr(movRegId), "")

                            if not skip:
                                all1mask = "0xFFFFFFFF" if (kernel["WavefrontSize"] == 32) else "0xFFFFFFFFFFFFFFFF"
                                kStr += inst("s_mov_b{}".format(kernel["WavefrontSize"]), sgpr(tmpSgpr, writer.laneSGPRCount), all1mask, "to restore all threads active")
                                kStr += inst("s_or_saveexec_b{}".format(kernel["WavefrontSize"]), writer.vcc, sgpr(tmpSgpr,writer.laneSGPRCount), "all threads active")
                                kStr += inst("s_nop", "3", "wait for exec mask")

                    kStr += inst("s_branch label_%04u" % shiftLabels[glvw-1], "done" )

        kStr += "label_%04u: // end shift0%s" % (shiftLabels[glvw-1], writer.endLine)

        writer.vgprPool.checkIn(movReg)
        writer.vgprPool.checkIn(threadIdInCoalReg)
        writer.vgprPool.checkIn(permuteIndexReg)

        writer.vgprPool.checkIn(glvwblkidReg)
        writer.vgprPool.checkIn(sReg)
        writer.vgprPool.checkIn(wReg)
        writer.vgprPool.checkIn(mtReg)

        # checkin scratch vgprs
        writer.vgprPool.checkIn(tmpVgpr)
        writer.vgprPool.checkIn(wgMT)

        return kStr
