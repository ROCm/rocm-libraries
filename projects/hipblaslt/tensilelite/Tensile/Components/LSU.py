################################################################################
#
# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

from rocisa.code import Module, Label, RegSet
from rocisa.container import DSModifiers, ContinuousRegister
from rocisa.instruction import DSLoadB128, DSLoadB32, DSLoadB64, DSStoreB128, \
    DSStoreB32, DSStoreB64, SAndB32, SCBranchSCC0, SCmpEQU32, SMovB32, SWaitCnt, \
    VAddF32, VAddI32, VAddU32, VAndB32, VLShiftLeftAddU32, VMovB32, VMulLOU32
from rocisa.functions import vectorStaticDivide
from copy import deepcopy
from ..Common import log2, ceilDivide
from ..Component import Component
from ..KernelWriterModules import *
from ..AsmStoreState import StoreState, VectorDataTypes
#import abc

class LSU(Component):
    """
    LSU block.
    """

class LSUOff(LSU):
    kernel = {"LocalSplitU": 1}

class LSUOn(LSU):

    @classmethod
    def matches(cls, writer, debug=False):
        return writer.states.kernel["LocalSplitU"] > 1

    def __call__(self):
        assert(0)

    def splitOutputData(self, writer, kernel):
        self.LSUelemCoord0 = []
        self.LSUelemCoord1 = []
        self.LSUelements   = []
        self.LSUfullVw     = []
        (vwdummy, eledummy, self.LSUfullVw, self.LSUelements) = writer.notLocalFullTileElements(kernel, False)
        storevw = self.LSUfullVw
        atomic = False # atomic is for GSU > 1
        beta = True
        vectorDataTypes = VectorDataTypes()
        ss = StoreState(writer, kernel, storevw, False, beta, atomic, self.LSUelements, vectorDataTypes, dim=0)
        self.LSUelemCoord0, self.LSUelemCoord1 = ss.getStoreElementsInfoForBatch(kernel, self.LSUelements)

        # search for valid lsu wave offset
        maxtt1 = 0
        maxtt0 = 0
        maxvc1 = 0
        maxvc0 = 0
        validOffset  = -1
        validOffset0 = -1
        validOffset1 = -1
        self.LSUelementsArchIdx      = [[] for i in range(4)]
        self.LSUelementsPerLSUWave   = []
        self.LSUelemCoord0PerLSUWave = []
        self.LSUelemCoord1PerLSUWave = []

        # Check valid LSU/VW combination
        if len(self.LSUelements) >= kernel["LocalSplitU"]:
            if kernel["LocalSplitU"] == 4:
                idxGrp = 1
                for idxGrp in range(1, len(self.LSUelements)//4 + 1):
                    for i in range(idxGrp):
                        i0 = i
                        i1 = i + 1 * idxGrp
                        i2 = i + 2 * idxGrp
                        i3 = i + 3 * idxGrp
                        offset0 = self.LSUelemCoord0[i0] + self.LSUelemCoord1[i0] * kernel["MacroTile0"]
                        offset1 = self.LSUelemCoord0[i1] + self.LSUelemCoord1[i1] * kernel["MacroTile0"]
                        offset2 = self.LSUelemCoord0[i2] + self.LSUelemCoord1[i2] * kernel["MacroTile0"]
                        offset3 = self.LSUelemCoord0[i3] + self.LSUelemCoord1[i3] * kernel["MacroTile0"]
                        if (offset3 - offset2 == offset2 - offset1) and (offset2 - offset1 == offset1 - offset0):
                            validOffset0 = self.LSUelemCoord0[i1] - self.LSUelemCoord0[i0]
                            validOffset1 = self.LSUelemCoord1[i1] - self.LSUelemCoord1[i0]
                        if self.LSUelemCoord0[i2] - self.LSUelemCoord0[i1] == validOffset0 \
                            and self.LSUelemCoord0[i3] - self.LSUelemCoord0[i2] == validOffset0 \
                            and self.LSUelemCoord1[i2] - self.LSUelemCoord1[i1] == validOffset1 \
                            and self.LSUelemCoord1[i3] - self.LSUelemCoord1[i2] == validOffset1:
                            validOffset  = offset1 - offset0
                            break
                    if validOffset != -1:
                        break
                for idx in range(0, len(self.LSUelements), 4*idxGrp):
                    for idx2 in range(idxGrp):
                        self.LSUelementsArchIdx[0].append(self.LSUfullVw*(idx + idx2))
                        self.LSUelementsArchIdx[1].append(self.LSUfullVw*(idx + 1*idxGrp + idx2))
                        self.LSUelementsArchIdx[2].append(self.LSUfullVw*(idx + 2*idxGrp + idx2))
                        self.LSUelementsArchIdx[3].append(self.LSUfullVw*(idx + 3*idxGrp + idx2))
                        self.LSUelementsPerLSUWave.append(self.LSUelements[idx + idx2])
                        self.LSUelemCoord0PerLSUWave.append(self.LSUelemCoord0[idx + idx2])
                        self.LSUelemCoord1PerLSUWave.append(self.LSUelemCoord1[idx + idx2])
            elif kernel["LocalSplitU"] == 2:
                i = 0
                offset0      = self.LSUelemCoord0[i] + self.LSUelemCoord1[i] * kernel["MacroTile0"]
                offset1      = self.LSUelemCoord0[i + 1] + self.LSUelemCoord1[i + 1] * kernel["MacroTile0"]
                validOffset  = offset1 - offset0
                validOffset0 = self.LSUelemCoord0[i + 1] - self.LSUelemCoord0[i]
                validOffset1 = self.LSUelemCoord1[i + 1] - self.LSUelemCoord1[i]
                for idx in range(0, len(self.LSUelements), 2):
                    self.LSUelementsArchIdx[0].append(self.LSUfullVw*idx)
                    self.LSUelementsArchIdx[1].append(self.LSUfullVw*(idx+1))
                    self.LSUelementsPerLSUWave.append(self.LSUelements[idx])
                    self.LSUelemCoord0PerLSUWave.append(self.LSUelemCoord0[idx])
                    self.LSUelemCoord1PerLSUWave.append(self.LSUelemCoord1[idx])
            else:
                assert 0, "No valid LSU offset found."

        if validOffset == -1:
            assert 0, "No valid LSU offset found."
        self.LSUValidOffset0 = validOffset0
        self.LSUValidOffset1 = validOffset1
        return validOffset

    ##############################################################################
    # LocalSplitU: Local Write, Read, and Reduction
    ##############################################################################
    def writeReadReduction(self, writer, kernel):
        module = Module("localSplitULocalWriteAndRead")
        module.addComment2("LocalSplitU Reduction")
        module.add(writer._syncThreads(kernel))
        module.add(Label("localSplitULocalWriteAndRead", ""))

        acc2arch, arch2acc = accToArchMapper(kernel)

        # prepare the data that is to be Reduction in this wave
        # the output LSUelementsArchIdx has all arch-indices.
        validOffset = self.splitOutputData(writer, kernel)

        numAccIdx    = len(self.LSUelementsArchIdx[0])
        numSetAccIdx = ceilDivide(numAccIdx, kernel["LocalSplitUReuseLDS"])
        maxLDSConstOffset = writer.states.regCaps["maxLDSConstOffset"]
        # computeStoreVgprs
        if kernel["EnableMatrixInstruction"]:
            module.add(writer.computeStoreVgprs(kernel))
        else:
            # new method. output self.vgprs.coord0InMT/coord1InMT
            module.add(writer.computeStoreVgprs(kernel, \
                                                divisor = kernel["MacroTile0"] // kernel["GlobalWriteVectorWidth"], \
                                                tid0Scale = kernel["GlobalWriteVectorWidth"], \
                                                tid1Scale = 1))

        # Checkout local read resource
        bpr            = 4 #bytes per register
        bytesPerElem   = kernel["ProblemType"]["ComputeDataType"].numBytes()
        bytesPerVector = self.LSUfullVw * bytesPerElem
        numWaves       = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
        regsPerStep = int((bytesPerVector+3)//4)
        elementStep = bytesPerVector // bytesPerElem
        numTotalAccVgprLdsReduction = len(self.LSUelements)*regsPerStep*(self.LSUfullVw//elementStep)
        assert (numTotalAccVgprLdsReduction%kernel["LocalSplitU"]) == 0, "AccVgprLdsRedcution % LSU != 0"
        numTotalAccVgprLdsReduction = numTotalAccVgprLdsReduction // kernel["LocalSplitU"]
        self.accVgprLdsReduction    = writer.vgprPool.checkOutAligned(numTotalAccVgprLdsReduction, 4, "LsuReduction")
        module.add(RegSet("v", "vgprLsuReduction", self.accVgprLdsReduction))
        module.addComment0("Size of vgprLsuReduction is %u"%(numTotalAccVgprLdsReduction))
        writer.states.c.startVgprValu = self.accVgprLdsReduction

        # Local Read VGPR idx
        localReadVgprIdx = 0

        lsu_id = writer.vgprPool.checkOut(1,"lsu_id")
        wave_id = writer.vgprPool.checkOut(1,"wave_id")
        tmpVgpr = writer.vgprPool.checkOutAligned(2, 2, "tmpVgpr")
        tmpVgprRes = ContinuousRegister(tmpVgpr, 2)

        module.add(vectorStaticDivide(wave_id, "Serial", \
            kernel["WavefrontSize"], tmpVgprRes))
        module.add(vectorStaticDivide(lsu_id, wave_id, numWaves, tmpVgprRes, \
            comment="Get LSU wave ID"))
        module.add(VAndB32(vgpr(wave_id), hex(numWaves - 1), vgpr(wave_id), \
            comment="Get wave ID"))

        # accVgprs read in GSUAMBSK
        codeAccVgprRead = []
        if kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel":
            codeAccVgprRead = deepcopy(writer.codes.accVgprRead) if writer.states.serializedStore else None

        for reUseIdx in range(kernel["LocalSplitUReuseLDS"]):
            module.addComment1("LocalSplitU: local write %d/%d"%(reUseIdx+1,kernel["LocalSplitUReuseLDS"]))
            module.add(Label("localSplitULocalWriteAndRead_%d"%(reUseIdx+1), ""))

            startLSUaccIdxSet = reUseIdx * numSetAccIdx
            endLSUaccIdxSet   = min(numAccIdx, startLSUaccIdxSet + numSetAccIdx)

            #scan the needed accVGPRIdx
            neededAccVGPRIdx = [[] for i in range(kernel["LocalSplitU"])]
            numAccVgpr = 0
            for lsu in range(kernel["LocalSplitU"]):
                for i in range(startLSUaccIdxSet, endLSUaccIdxSet):
                    for j in range(self.LSUfullVw):
                        accIdx = arch2acc[self.LSUelementsArchIdx[lsu][i] + j]
                        neededAccVGPRIdx[lsu].append(accIdx)
                        numAccVgpr += 1

            # lsuProcessOffset is used when local read
            numVgprPerLSU    = len(neededAccVGPRIdx[0])
            lsuProcessOffset = numVgprPerLSU * kernel["WavefrontSize"] * 4

            # break here instead of assert to release resource and do a post-rejection
            # caused by range(startLSUaccIdxSet, endLSUaccIdxSet) = range(1,1)
            if numAccVgpr <= 0:
                writer.states.invalidLSUCode = True
                break

            assert numAccVgpr > 0,"startLSUaccIdxSet=%u,endLSUaccIdxSet=%u,numAccIdx=%u"%(startLSUaccIdxSet,endLSUaccIdxSet,numAccIdx)
            accVgprRes = writer.vgprPool.checkOutAligned(numAccVgpr, 4, "accLSUVgprRes")

            destIdx = 0
            for lsu in range(kernel["LocalSplitU"]):
                for i in range(numVgprPerLSU):
                    srcIdx   = neededAccVGPRIdx[lsu][i]
                    readInst = writer.accVgprReadWriteFunction(kernel, srcIdx, True)
                    srcVgpr  = writer.accVgprReadWriteIndex(kernel, srcIdx)
                    module.add(readInst(dst=vgpr(accVgprRes+destIdx),
                                        src=srcVgpr,
                                        comment="copy acc[%u] to vreg[%u], LSU%u will process" % (srcIdx,destIdx,lsu)))
                    destIdx += 1

            dataPerWave = numAccVgpr * kernel["WavefrontSize"] * 4
            ldsStride   = dataPerWave * numWaves
            # Prepare Write/Read instruction info
            if bytesPerVector % 16 == 0:
                DSStoreBX    = DSStoreB128
                DSLoadBX     = DSLoadB128
                numInstPerVW = bytesPerVector // 16
                regsPerStore = 4
            elif bytesPerVector % 8 == 0:
                DSStoreBX    = DSStoreB64
                DSLoadBX     = DSLoadB64
                numInstPerVW = bytesPerVector // 8
                regsPerStore = 2
            else:
                DSStoreBX    = DSStoreB32
                DSLoadBX     = DSLoadB32
                numInstPerVW = bytesPerVector // 4
                regsPerStore = 1

            maxOffset = (kernel["LocalSplitU"] -1) * ldsStride + ((numVgprPerLSU // self.LSUfullVw -1) * numInstPerVW + (numInstPerVW -1)) * regsPerStore * (bpr * kernel["WavefrontSize"])
            numAddr = maxOffset // maxLDSConstOffset + 1
            addr = writer.vgprPool.checkOut(numAddr,"addr")
            with writer.allocTmpSgpr(1) as tmpSgprInfo:
                tmpSgpr = tmpSgprInfo.idx
                module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(dataPerWave), \
                    comment="dataPerWave (%d)"%dataPerWave))
                module.add(VAndB32(vgpr(addr), hex(kernel["WavefrontSize"]-1), vgpr("Serial"), \
                    comment="initial addr"))
                module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=sgpr(tmpSgpr), src1=vgpr(wave_id), \
                    comment="tmp = waveId * dataPerWave"))
                module.add(VLShiftLeftAddU32(dst=vgpr(addr), shiftHex=log2(regsPerStore * bpr), src0=vgpr(addr), src1=vgpr(tmpVgpr), \
                    comment="addr = initial addr + tmp"))
                module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(ldsStride), \
                    comment="ldsStride = waveNum * dataPerWave (%d)"%ldsStride))
                module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=sgpr(tmpSgpr), src1=vgpr(lsu_id), \
                    comment="tmp = (waveNum * dataPerWave) * lsu_id"))
                module.add(VAddU32(vgpr(addr), vgpr(tmpVgpr), vgpr(addr), \
                    comment="addr += tmp"))

            module.add(SWaitCnt(dscnt=0, comment="wait for all writes"))
            module.add(writer._syncThreads(kernel, "pre-lsu local write"))

            module.add(Label("localSplitULocalWrite_%d"%(reUseIdx+1), ""))

            # Do Local Write
            for i in range(0, numAccVgpr // self.LSUfullVw):
                for v in range(numInstPerVW):
                    regIdx = (i * numInstPerVW + v) * regsPerStore
                    module.add(DSStoreBX(dstAddr=vgpr(addr), src=vgpr(accVgprRes+regIdx, regsPerStore), \
                            ds=DSModifiers(offset=(regIdx * (bpr * kernel["WavefrontSize"]))), \
                            comment="arch[%d]"%(i * numInstPerVW + v)))

            # Release local write resource
            writer.vgprPool.checkIn(accVgprRes)

            module.addComment1("LocalSplitU: local read %d/%d"%(reUseIdx+1,kernel["LocalSplitUReuseLDS"]))

            # Calculate offset for wave id and lsu id
            with writer.allocTmpSgpr(1) as tmpSgprInfo:
                tmpSgpr = tmpSgprInfo.idx
                module.add(VAndB32(vgpr(addr), hex(kernel["WavefrontSize"]-1), vgpr("Serial"), \
                    comment="initial addr"))
                module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(dataPerWave), \
                    comment="wave offset (%d)"%dataPerWave))
                module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=sgpr(tmpSgpr), src1=vgpr(wave_id), \
                    comment="wave offset = wave_id * wave offset"))
                module.add(VLShiftLeftAddU32(dst=vgpr(addr), shiftHex=log2(regsPerStore * bpr), src0=vgpr(addr), src1=vgpr(tmpVgpr), \
                    comment="addr = initial addr + wave offset"))
                module.add(SMovB32(dst=sgpr(tmpSgpr), \
                    src=hex(lsuProcessOffset), comment="LSU Process Offset %d"%(lsuProcessOffset)))
                module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=sgpr(tmpSgpr), src1=vgpr(lsu_id), \
                    comment="lsu offset = lsu_id * LSU Process Offset"))
                module.add(VAddU32(dst=vgpr(addr), src0=vgpr(addr), src1=vgpr(tmpVgpr), \
                    comment="addr += lsu offset"))
                for i in range(1,numAddr):
                    module.add(VAddU32(vgpr(addr+i), maxLDSConstOffset*i, vgpr(addr), \
                    comment="addr += maxLDSConstOffset*%u"%(i)))

            module.add(SWaitCnt(dscnt=0, comment="wait for all writes"))
            module.add(writer._syncThreads(kernel, "post-lsu local write"))
            module.add(Label("localSplitULocalRead_%d"%(reUseIdx+1), ""))

            moduleReduction = Module("LocalSplitU_Reduction")
            inLoopTmpVgpr   = writer.vgprPool.checkOutAligned(numVgprPerLSU*(kernel["LocalSplitU"]-1), 4, "TempLsuReduction")

            # Do Local Read
            for i in range(0, numVgprPerLSU // self.LSUfullVw):
                for v in range(numInstPerVW):
                    for r in range(0, kernel["LocalSplitU"]):
                        regIdx = (i * numInstPerVW + v) * regsPerStore
                        offset = r * ldsStride + regIdx * (bpr * kernel["WavefrontSize"])
                        num = offset // maxLDSConstOffset
                        offset -= num * maxLDSConstOffset
                        srcvgpr = vgpr(addr+num)
                        if r == 0:
                            vgprStr = "LsuReduction+%u"%(localReadVgprIdx)
                        else:
                            vgprStr = inLoopTmpVgpr + (numVgprPerLSU * (r - 1) + regIdx)
                        module.add(DSLoadBX(dst=vgpr(vgprStr, regsPerStore), src=srcvgpr, \
                                    ds=DSModifiers(offset=(offset)), \
                                    comment="r=%u i=%u, from acc[%d]"%(r, (i * numInstPerVW + v), neededAccVGPRIdx[0][(i * numInstPerVW + v)])))
                        # Generate Reduction code at the same time.
                        if r == 0:
                            # Insert waitcnt code here
                            numTotalInst  = numVgprPerLSU // self.LSUfullVw * numInstPerVW * kernel["LocalSplitU"]
                            numPassedInst = (i * numInstPerVW + (v + 1)) * kernel["LocalSplitU"]
                            numLRWaitCnt = numTotalInst - numPassedInst
                            moduleReduction.add(SWaitCnt(dscnt=numLRWaitCnt, comment="wait count is (%u-%u)"%(numTotalInst, numPassedInst)))
                        if r > 0:
                            for regToAdd in range(regsPerStore):
                                if kernel["ProblemType"]["ComputeDataType"].isSingle():
                                    moduleReduction.add(VAddF32(dst=vgpr("LsuReduction+%u"%(localReadVgprIdx+regToAdd)), src0=vgpr(vgprStr+regToAdd), \
                                                src1=vgpr("LsuReduction+%u"%(localReadVgprIdx+regToAdd)), comment=""))
                                elif kernel["ProblemType"]["ComputeDataType"].isInt32():
                                    moduleReduction.add(VAddI32(dst=vgpr("LsuReduction+%u"%(localReadVgprIdx+regToAdd)), src0=vgpr(vgprStr+regToAdd), \
                                                src1=vgpr("LsuReduction+%u"%(localReadVgprIdx+regToAdd)), comment=""))
                                else:
                                # TODO: hpa_half, int8
                                    assert(0) # unsupported data type, need to modify here and LSU write/read code
                    
                    if kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel":
                        if not kernel["MIArchVgpr"]:
                            # write to accvgpr
                            for regToAdd in range(regsPerStore):
                                codeAccVgprReadInst = codeAccVgprRead.popFirstItem() # v_accvgpr_read_b32
                                accIdx = codeAccVgprReadInst.srcs[0].regIdx
                                writeInst = writer.accVgprReadWriteFunction(kernel, accIdx, False)
                                dstVgpr  = writer.accVgprReadWriteIndex(kernel, accIdx)
                                moduleReduction.add(writeInst(dst=dstVgpr, src=vgpr("LsuReduction+%u"%(localReadVgprIdx+regToAdd)),
                                                    comment="copy vreg[%u] to acc[%u]" % (localReadVgprIdx+regToAdd, accIdx)))
                        else:
                            # write to vgpr
                            for regToAdd in range(regsPerStore):
                                codeAccVgprReadInst = codeAccVgprRead.popFirstItem() # v_mov_b32
                                dstVgpr = codeAccVgprReadInst.srcs[0]
                                moduleReduction.add(VMovB32(dst=dstVgpr, src=vgpr("LsuReduction+%u"%(localReadVgprIdx+regToAdd)),
                                                            comment="copy lsu results to vreg"))
                    localReadVgprIdx += regsPerStore

            # Release write/read resource
            writer.vgprPool.checkIn(addr)

            # Do Reduction
            module.add(moduleReduction)

            # Release reduction resource
            writer.vgprPool.checkIn(inLoopTmpVgpr)

        # Release all resource
        writer.vgprPool.checkIn(lsu_id)
        writer.vgprPool.checkIn(wave_id)
        writer.vgprPool.checkIn(tmpVgpr)
        # reset vgprValuC register
        module.add(RegSet("v", "vgprValuC", self.accVgprLdsReduction))

        return module

    ##############################################################################
    # LocalSplitU: Global Write Indices
    ##############################################################################
    def globalWriteIndices(self, writer, kernel):
        module = Module("localSplitUGlobalWriteIndices")

        # Add LSU Offset back
        packedC1 = kernel["PackedC1IndicesX"]
        strideC1 = "StrideC%s" % (writer.states.indexChars[packedC1[0]])
        strideD1 = "StrideD%s" % (writer.states.indexChars[packedC1[0]])
        wave_id = writer.vgprPool.checkOut(1, "tmpWaveID")
        tmpVgpr1 = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr1")
        tmpVgpr1Res = ContinuousRegister(tmpVgpr1, 2)
        module.add(vectorStaticDivide(wave_id, "Serial", kernel["WavefrontSize"], tmpVgpr1Res))
        numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]
        module.add(vectorStaticDivide(wave_id, wave_id, numWaves, tmpVgpr1Res))

        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            tmpSgpr = tmpSgprInfo.idx
            if self.LSUValidOffset0 > 0:
                module.add(SMovB32(dst=sgpr(tmpSgpr), \
                        src=hex(self.LSUValidOffset0), comment="a valid offset"))
                module.add(VMulLOU32(dst=vgpr(tmpVgpr1), src0=vgpr(wave_id), src1=sgpr(tmpSgpr), comment="wave LSU offset"))
                module.add(VAddU32(dst=vgpr(writer.vgprs.coord0), src0=vgpr(tmpVgpr1), src1=vgpr(writer.vgprs.coord0), comment="coord0 += LSU offset0"))
            else:
                module.addComment0("valid offset coord0 is zero.")

            if self.LSUValidOffset1 > 0:
                module.add(SMovB32(dst=sgpr(tmpSgpr), \
                        src=hex(self.LSUValidOffset1), comment="a valid offset"))
                module.add(VMulLOU32(dst=vgpr(tmpVgpr1), src0=vgpr(wave_id), src1=sgpr(tmpSgpr), comment="wave LSU offset"))
                module.add(VAddU32(dst=vgpr(writer.vgprs.coord1), src0=vgpr(tmpVgpr1), src1=vgpr(writer.vgprs.coord1), comment="coord1 += LSU offset1"))
                module.add(VAddU32(dst=vgpr(writer.vgprs.coord1InMT), src0=vgpr(tmpVgpr1), src1=vgpr(writer.vgprs.coord1InMT), comment="coord1InMT += LSU offset1"))

                # this code is from CouputeStoreVgprs. coord 1 : offset part
                packedC1 = kernel["PackedC1IndicesX"]
                strideC1 = "StrideC%s" % (writer.states.indexChars[packedC1[0]])
                strideD1 = "StrideD%s" % (writer.states.indexChars[packedC1[0]])
                module.add(VMulLOU32(dst=vgpr(writer.vgprs.cinRowPtr), src0=vgpr(writer.vgprs.coord1InMT), src1=sgpr(strideC1), comment=" offset 1"))
                module.add(VMulLOU32(dst=vgpr(writer.vgprs.coutRowPtrD), src0=vgpr(writer.vgprs.coord1InMT), src1=sgpr(strideD1), comment=" offset 1"))
                if kernel["ProblemType"]["UseE"] and (kernel["GlobalSplitU"] == 1 or kernel["GlobalSplitU"] == -1):
                        module.add(VMovB32(dst=vgpr(writer.vgprs.coutRowPtrE), src=vgpr(writer.vgprs.coord1InMT), comment=" save offset 1 for E"))
                if writer.vgprs.coutRowPtrBias != -1:
                        index = packedC1[0] - 1
                        strideW1 = "Size%s" % "I" if index == 0 else ("J" if index == 1 else (writer.states.indexChars[index]))
                        module.add(VMulLOU32(dst=vgpr(writer.vgprs.coutRowPtrBias), src0=vgpr(writer.vgprs.coord1InMT), src1=sgpr(strideW1), comment=" offset 1"))
            else:
                module.addComment0("valid offset coord1 is zero.")

        writer.vgprPool.checkIn(tmpVgpr1)
        writer.vgprPool.checkIn(wave_id)
        writer.vgprPool.checkIn(writer.vgprs.coord0InMT)
        writer.vgprPool.checkIn(writer.vgprs.coord1InMT)

        if kernel["BufferStore"]:
            #print "----AddressC-LocalSplitU"
            #print self.vgprPool.state()
            writer.vgprs.addrE             = -1
            writer.vgprs.addrD             = -1
            writer.vgprs.addrC             = -1
            writer.vgprs.addrBias          = -1
            writer.vgprs.addrScaleAVec     = -1
            writer.vgprs.addrScaleBVec     = -1
            writer.vgprs.addrScaleAlphaVec = -1
        else:
            writer.vgprs.addrD = writer.vgprPool.checkOut(2)
            module.add(VMovB32(
                    dst=vgpr(writer.vgprs.addrD+0), \
                    src=sgpr("AddressD+0"), \
                    comment="sgpr -> vgpr"))
            module.add(VMovB32(
                    dst=vgpr(writer.vgprs.addrD+1), \
                    src=sgpr("AddressD+1"), \
                    comment="sgpr -> vgpr"))
            writer.vgprs.addrC = writer.vgprPool.checkOut(2)
            module.add(VMovB32(
                    dst=vgpr(writer.vgprs.addrC+0), \
                    src=sgpr("AddressC+0"), \
                    comment="sgpr -> vgpr"))
            module.add(VMovB32(
                    dst=vgpr(writer.vgprs.addrC+1), \
                    src=sgpr("AddressC+1"), \
                    comment="sgpr -> vgpr"))

            if kernel["GlobalSplitU"] != 0:
                gsuLabel = Label(label=writer.labels.getNameInc("GSU"), comment="")
                with writer.allocTmpSgpr(1) as tmpSgprGSU:
                    module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
                    module.add(SCmpEQU32(src0=sgpr(tmpSgprGSU.idx), src1=1, comment="GSU == 1 ?"))
                module.add(SCBranchSCC0(labelName=gsuLabel.getLabelName(), comment="branch if GSU != 1"))
            if kernel["ProblemType"]["UseE"]:
                writer.vgprs.addrE = writer.vgprPool.checkOut(2, 'addrE')
                module.add(VMovB32( \
                        dst=vgpr(writer.vgprs.addrE+0), \
                        src=sgpr("AddressE+0"), \
                        comment="sgpr -> vgpr"))
                module.add(VMovB32( \
                        dst=vgpr(writer.vgprs.addrE+1), \
                        src=sgpr("AddressE+1"), \
                        comment="sgpr -> vgpr"))
            if writer.states.useBias == DataDirection.READ:
                writer.vgprs.addrBias = writer.vgprPool.checkOut(2, 'addrBias')
                module.add(VMovB32( \
                        dst=vgpr(writer.vgprs.addrBias+0), \
                        src=sgpr("AddressBias+0"), \
                        comment="sgpr -> vgpr"))
                module.add(VMovB32( \
                        dst=vgpr(writer.vgprs.addrBias+1), \
                        src=sgpr("AddressBias+1"), \
                        comment="sgpr -> vgpr"))
            if (kernel["ProblemType"]["UseScaleAB"] == "Vector"):
                writer.vgprs.addrScaleAVec = writer.vgprPool.checkOut(2, 'addrScaleAVec')
                module.add(VMovB32( \
                        dst=vgpr(writer.vgprs.addrScaleAVec+0), \
                        src=sgpr("AddressScaleA+0"), \
                        comment="sgpr -> vgpr"))
                module.add(VMovB32( \
                        dst=vgpr(writer.vgprs.addrScaleAVec+1), \
                        src=sgpr("AddressScaleA+1"), \
                        comment="sgpr -> vgpr"))
                writer.vgprs.addrScaleBVec = writer.vgprPool.checkOut(2, 'addrScaleVVec')
                module.add(VMovB32( \
                        dst=vgpr(writer.vgprs.addrScaleBVec+0), \
                        src=sgpr("AddressScaleB+0"), \
                        comment="sgpr -> vgpr"))
                module.add(VMovB32( \
                        dst=vgpr(writer.vgprs.addrScaleBVec+1), \
                        src=sgpr("AddressScaleB+1"), \
                        comment="sgpr -> vgpr"))
            if kernel["ProblemType"]["UseScaleAlphaVec"]:
                writer.vgprs.addrScaleAlphaVec = writer.vgprPool.checkOut(2, 'addrScaleAlphaVec')
                module.add(VMovB32( \
                        dst=vgpr(self.vgprs.addrScaleAlphaVec+0), \
                        src=sgpr("AddressScaleAlphaVec+0"), \
                        comment="sgpr -> vgpr"))
                module.add(VMovB32( \
                        dst=vgpr(self.vgprs.addrScaleAlphaVec+1), \
                        src=sgpr("AddressScaleAlphaVec+1"), \
                        comment="sgpr -> vgpr"))
            if kernel["GlobalSplitU"] != 0:
                module.add(gsuLabel)

        return module


    ##############################################################################
    # LocalSplitU: Global Write
    ##############################################################################
    def globalWrite(self, writer, kernel, tPA, tPB):
        if not writer.do["PostLoop"]: return ""

        elements_0 = [[] for y in range(2)]
        elements_1 = [[] for y in range(2)]
        elements_f0    = [[] for y in range(2)]
        elements_f1    = [[] for y in range(2)]
        (fullVw, elements_0[False], fullVw_1, elements_1[False]) = writer.notLocalFullTileElements(kernel, False)
        (edgeVw, elements_0[True], edgeVw_1, elements_1[True] )    = writer.notLocalFullTileElements(kernel, True)
        edgeScaled_0 = len(elements_0[True]) // len(elements_1[False])
        edgeScaled_1 = len(elements_1[True]) // len(elements_1[False])
        noEgScaled_0 = len(elements_0[False]) // len(elements_1[False])

        for i in range(0, len(elements_1[False])):
            element = elements_1[False][i]
            if element in self.LSUelementsPerLSUWave:
                elements_f1[False].append(element)
                for j in range(0, edgeScaled_0):
                    # in general, edge will affect vc0 dimension.
                    element = elements_0[True][i*edgeScaled_0+j]
                    elements_f0[True].append(element)
                for j in range(0, edgeScaled_1):
                    # in general, edge will affect vc0 dimension.
                    element = elements_1[True][i*edgeScaled_1+j]
                    elements_f1[True].append(element)
                for j in range(0, noEgScaled_0):
                    # in general, edge will affect vc0 dimension.
                    element = elements_0[False][i*noEgScaled_0+j]
                    elements_f0[False].append(element)

        vectorWidths     = [fullVw, edgeVw]
        vectorWidths_1 = [fullVw_1, edgeVw_1]

        noGSUBranch = (kernel["GlobalSplitU"] == 0 and kernel["StreamK"] != 3)
        module = Module("localSplitUGlobalWrite")
        module.add(writer.globalWriteElements(kernel, tPA, tPB, vectorWidths, vectorWidths_1, elements_f0, elements_f1, noGSUBranch=noGSUBranch))
        writer.cleanupGlobalWrite(kernel)
        writer.vgprPool.checkIn(self.accVgprLdsReduction)
        return module
