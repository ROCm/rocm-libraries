################################################################################
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

from rocisa import countInstruction
from rocisa.code import Module, Label, RegSet, ValueSet
from rocisa.container import ContinuousRegister, EXEC, SMEMModifiers, MUBUFModifiers, FLATModifiers, vgpr, sgpr, replaceHolder
from rocisa.enum import CacheScope
from rocisa.instruction import SAddCU32, SAddU32, SAndB32, SLoadB32, SStoreB32, SBranch, \
    SCBranchSCC0, SCBranchSCC1, SCMovB32, SCSelectB32, SCmpEQU32, SCmpLgU32, SCmpLtU32, SCmpGtI32, \
    SLShiftLeftB64, SLShiftRightB32, SMovB32, SMovB64, SMulI32, SSubU32, SCmpEQI32, SEndpgm, \
    SCmpLeI32, VCmpGEI32, SSubI32, SCBranchSCC0, VMovB32, SLShiftLeftB32, SWaitCnt, SWaitXCnt, SBarrier, \
    SNop, SSleep, VAddF32, VAddI32, VReadfirstlaneB32, SMulHIU32, VAddPKF32, VCndMaskB32, SAtomicDec, \
    SCmpEQU64, BufferStoreB32, BufferLoadB32, VMovB64, FlatAtomicDecU32
from rocisa.functions import scalarStaticMultiply64, scalarUInt32DivideAndRemainder, vectorStaticMultiply

from ..Common import ceilDivide, log2, print2, INDEX_CHARS
from ..Component import Component
from ..AsmStoreState import StoreState, VectorDataTypes
from ..AsmAddressCalculation import AddrCalculation
import abc
from copy import deepcopy
from math import ceil
from ..KernelWriterModules import mapAcctoArchRegs

class GSU(Component):
    """
    GSU block.
    """
    @abc.abstractmethod
    def graWorkGroup(self, writer, kernel):
        pass

    @abc.abstractmethod
    def computeLoadSrd(self, writer, kernel, tP, stmp, tileStart):
        pass

    @abc.abstractmethod
    def graIncrements(self, writer, kernel, loopIdx, tP):
        pass

    @abc.abstractmethod
    def graIncrementsRestore(self, writer, kernel, loopCounterName):
        pass

    def graIncrementsCommon(self, writer, loopIdx, tc, stride, m):
        module = Module("GSU Common graIncrements")

        # multiply by stride, optimizing if unit stride
        if writer.isConstUnitStride(stride):
            if tc == "A":
                abinfo = writer.states.a
            elif tc == "B":
                abinfo = writer.states.b
            elif tc == "MXSA":
                abinfo = writer.states.mxsa
            elif tc == "MXSB":
                abinfo = writer.states.mxsb
            else:
                abinfo = None
            if abinfo != None and abinfo.useConstSgprGlobalReadIncs:
                # useConstSgprGlobalReadIncs case, define value set for GlobalReadIncs here instead of initializing sgpr
                module.add(ValueSet("GlobalReadIncs%s"%tc, m))
            else:
                module.add(SMovB32(dst=sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx)), src=m, \
                    comment="incr%s (unrollIdx)"%(tc) ))
        else:
            module.add(SMulI32(dst=sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx)), \
                src0=m, src1=stride, \
                comment="incr%s unrollIdx)"%(tc) ))

        return module

    @abc.abstractmethod
    def calculateLoopNumIter(self, writer, kernel, loopCounterName, tmpSgprInfo):
        pass

    @abc.abstractmethod
    def calculateIncrementMetadata(self, writer, kernel, sgprOut):
        pass

    @abc.abstractmethod
    def computeStoreSrdStart(self, writer, kernel):
        pass
    
    def computeStoreSrdStartCommon(self, writer, kernel):
        module = Module("GSU Common computeStoreSrdStart")
        
        indices = list(range(0, kernel["ProblemType"]["NumIndicesC"]))
        numDim = len(indices)

        if kernel["_GlobalAccumulation"] == "MultipleBuffer" or kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
            gsuLabel = Label(label=writer.labels.getNameInc("GSU"), comment="")
            with writer.allocTmpSgpr(1, tag="computeStoreSrdStart_tmpSgprGSU") as tmpSgprGSU:
                module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
                module.add(SCmpEQU32(src0=sgpr(tmpSgprGSU.idx), src1=1, comment="GSU == 1 ?"))
                module.add(SCBranchSCC1(labelName=gsuLabel.getLabelName(), comment="branch if GSU == 1"))
            # GSU algorithm 2: adjust output buffer address to per GSU buffer
            with writer.allocTmpSgpr(4, alignment=1, tag="computeStoreSrdStart_tmpSgprInfo") as tmpSgprInfo:
                if tmpSgprInfo.idx % 2 == 0:
                    tmpSgprX2 = tmpSgprInfo.idx+0
                    tmpSgpr0 = tmpSgprInfo.idx+0
                    tmpSgpr1 = tmpSgprInfo.idx+1
                    tmpSgpr2 = tmpSgprInfo.idx+2
                    tmpSgpr3 = tmpSgprInfo.idx+3
                else:
                    tmpSgprX2 = tmpSgprInfo.idx+1
                    tmpSgpr0 = tmpSgprInfo.idx+1
                    tmpSgpr1 = tmpSgprInfo.idx+2
                    tmpSgpr2 = tmpSgprInfo.idx+0
                    tmpSgpr3 = tmpSgprInfo.idx+3
                module.addComment("GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s")
                module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgpr0), sgpr(tmpSgpr1), sgpr("SizesFree+0"), sgpr("GSUSumIdx"), comment="Free0"))
                for i in range(1, numDim):
                    module.add(SSubU32(dst=sgpr(tmpSgpr2), src0=sgpr("SizesFree+%u"%i), src1=1, comment="Free%u" % i))
                    module.add(SMulI32(dst=sgpr(tmpSgpr2), src0=sgpr(tmpSgpr2), src1=sgpr("GSUSumIdx"), comment="Free%u" % i))
                    module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgpr2), sgpr(tmpSgpr3), sgpr(tmpSgpr2), sgpr("StrideC%s"%writer.states.indexChars[i]), comment="Free%u" % i))
                    module.add(SAddU32(dst=sgpr(tmpSgpr0), src0=sgpr(tmpSgpr0), src1=sgpr(tmpSgpr2), comment="Free%u" % i))
                    module.add(SAddCU32(dst=sgpr(tmpSgpr1), src0=sgpr(tmpSgpr1), src1=sgpr(tmpSgpr3), comment="Free%u" % i))
                module.add(SLShiftLeftB64(dst=sgpr(tmpSgprX2,2), src=sgpr(tmpSgprX2,2), shiftHex=log2(writer.states.bpeCinternal), comment="scale by bpe"))
                module.add(SAddU32(dst=sgpr("SrdD+0"), src0=sgpr("SrdD+0"), src1=sgpr(tmpSgprX2), comment="add lo GSU offset to SRD"))
                module.add(SAddCU32(dst=sgpr("SrdD+1"), src0=sgpr("SrdD+1"), src1=sgpr(tmpSgpr1), comment="add hi GSU offset to SRD"))
            module.add(gsuLabel)

        return module

    @abc.abstractmethod
    def noLoadLoop(self, writer, kernel, tensorParametersA, tensorParametersB, pack, packPre):
        pass

    @abc.abstractmethod
    def tailLoopNumIter(self, writer, kernel, loopCounter):
        pass

    @abc.abstractmethod
    def setupNewTile(self, writer, kernel, tensorParametersA, tensorParametersB, tPM):
        pass

    def graIncrementsAB(self, writer, kernel, tensorParametersA, tensorParametersB, tPM):
        module = Module("GSU Common graIncrementsAB")
        module.addComment1("global read addresses: increments a")
        for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
            module.add(writer.graIncrements(kernel, i, tensorParametersA))
        if kernel["ProblemType"]["MXBlockA"]:
          module.addComment1("global read addresses: increments mxsa")
          for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
              module.add(writer.graIncrements(kernel, i, tensorParametersA["MX"]))
        if kernel["ProblemType"]["MXBlockB"]:
          module.addComment1("global read addresses: increments mxsb")
          for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
              module.add(writer.graIncrements(kernel, i, tensorParametersB["MX"]))
        if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            module.addComment1("global read addresses: increments metadata")
            for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
                module.add(writer.graIncrements(kernel, i, tPM))
        module.addComment1("global read addresses: increments b")
        for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
            module.add(writer.graIncrements(kernel, i, tensorParametersB))

        return module

    @abc.abstractmethod
    def globalWriteBatchProlog(self, writer, kernel, tmpVgpr, tmpVgprSize, tmpVgprDynamic, \
                               batchIdx, ss, gwvw, batchElements, \
                               beta, edge, sumIdxGSUSYNC, addrCalc):
        pass

    @abc.abstractmethod
    def defineAndResources(self, writer, kernel, tmpSgpr0, tmpSgprM, tmpSgprN, tmpSgprNumWG0, tmpSgprAccumTiles):
        pass

    @abc.abstractmethod
    def writeBiasToGlobal(self, writer, kernel, biasDataType, tP, tmpSgprRes, biasBpe):
        pass

    @abc.abstractmethod
    def reductionBranches(self, writer, kernel, tPB, vectorWidths, elements, tmpVgpr, cvtVgprStruct, vectorDataTypes, factorDims, reductionEndLabel, endLabel):
        pass
    
    @abc.abstractmethod
    def initializeSrd(self, writer, ArgTypeCheckLabel, GeneralBatchedGemmSrdInitiation_End, kernel, ch):
        pass

    @abc.abstractmethod
    def routeToGeneralBatchedOrStridedBatched(self, stridedBatchedGemmLoad, multipleBufferChecks, generalBatchedGemmLoad, mat, kernel, tmpS1):
        pass

class GSUOff(GSU):
    kernel = {"GlobalSplitU": 0}

    def __call__(self):
        assert(0)

    def graWorkGroup(self, writer, kernel):
        module = Module("GSU Off graWorkGroup")
        return module

    def computeLoadSrd(self, writer, kernel, tP, stmp, tileStart):
        module = Module("GSU Off computeLoadSrd")
        return module

    def graIncrements(self, writer, kernel, loopIdx, tP):
        module = Module("GSU Off graIncrements")

        tc = tP["tensorChar"]
        tIdx: int = tP["idx"]
        dimIdx = kernel["ProblemType"]["IndicesSummation"][loopIdx] # dimension index
        loopChar = writer.states.indexChars[dimIdx]
        stride = writer.strideRef(tc, dimIdx)
        isMirrorIdx = dimIdx in kernel["ProblemType"]["MirrorDims%s"%tc]
        m = int(kernel["_DepthU%s"%tc] * tP["bpeGR"])
        if isMirrorIdx:
          m = -m

        if writer.states.globalReadIncsUseVgpr:
            with writer.allocTmpSgpr(2, tag="GSUOff_graIncrements_tmpSgprInfo") as tmpSgprInfo:
                tmpSgpr = tmpSgprInfo.idx
                module.add(SMovB32(dst=sgpr(tmpSgpr+0), src="DepthU*%d"%(tP["bpeGR"]), comment="DepthU*Bpe"))
                module.add(SMulI32(dst=sgpr(tmpSgpr+0), src0=sgpr(tmpSgpr+0), src1=stride, \
                    comment="incr%s%s = %s*DepthU*bpeGR (unrollIdx)"%(tc, loopChar, stride) ))
                # TODO - this should be mul-H??
                module.add(SMovB32(dst=sgpr(tmpSgpr+1), src=hex(0), comment="(carry)"))
                module.add(VMovB32(dst=vgpr("GlobalReadIncs%s+%u+0"%(tc, 2*loopIdx)), src=sgpr(tmpSgpr+0)))
                module.add(VMovB32(dst=vgpr("GlobalReadIncs%s+%u+1"%(tc, 2*loopIdx)), src=sgpr(tmpSgpr+1)))
        else:
            # MX scale unroll-step SRD increment, gated by MXScaleFormat:
            #   - Swizzled (HostPreSwizzle/InMemorySwizzle): each K-block of
            #     scales (DepthU/MXBlock K-scales per M) is laid out as a
            #     contiguous (M, K_inner) block, so a K-step of DepthU advances
            #     the SRD by Size{tile} * (DepthU/MXBlock * bpe).
            #   - NoSwizzle: canonical row/column layout with K-axis stride 1;
            #     the K-step of DepthU is just DepthU/MXBlock * bpe and goes
            #     through the standard graIncrementsCommon path.
            mxFmt = kernel.get("MXScaleFormat", "NoSwizzle")
            isMxSwizzledScale = ('MXS' in tc) and mxFmt in ("InMemorySwizzle", "HostPreSwizzle")
            if isMxSwizzledScale and writer.isConstUnitStride(stride):
                module.add(SMulI32(dst=sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx)), \
                    src0=sgpr("Size%s"%INDEX_CHARS[tIdx]), src1=m, \
                    comment="incr%s = Size%s*DepthU*Bpe (unrollIdx, swizzled MX scale layout)"%(tc, INDEX_CHARS[tIdx])))
            else:
                module.add(self.graIncrementsCommon(writer, loopIdx, tc, stride, m))

        return module

    def graIncrementsRestore(self, writer, kernel, loopCounterName):
        module = Module("GSU Off graIncrementsRestore")
        return module

    def calculateLoopNumIter(self, writer, kernel, loopCounterName, tmpSgprInfo):
        module = Module("GSU Off calculateLoopNumIter")
        return module

    def calculateIncrementMetadata(self, writer, kernel, sgprOut):
        module = Module("GSU Off calculateLoopNumIter")
        module.add(SMovB32(dst=sgpr(sgprOut), src=kernel["DepthU"], comment="IncsMetadata = DepthU if GSUC == 1"))
        module.add(SLShiftRightB32(dst=sgpr(sgprOut), shiftHex=hex(log2(8)), src=sgpr(sgprOut)))
        return module

    def computeStoreSrdStart(self, writer, kernel):
        module = Module("GSU Off computeStoreSrdStart")
        return module

    def noLoadLoop(self, writer, kernel, tensorParametersA, tensorParametersB, pack, packPre):
        module = Module("GSU Off noLoadLoop")
        return module

    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("GSU Off tailLoopNumIter")
        return module

    def setupNewTile(self, writer, kernel, tensorParametersA, tensorParametersB, tPM):
        module = Module("GSU Off setupNewTile")
        module.add(self.graIncrementsAB(writer, kernel, tensorParametersA, tensorParametersB, tPM))
        return module

    def reductionBranches(self, writer, kernel, tPB, vectorWidths, elements, tmpVgpr, cvtVgprStruct, vectorDataTypes, factorDims, reductionEndLabel, endLabel):
        module = Module("GSU Off reductionBranches")
        return module

    def globalWriteBatchProlog(self, writer, kernel, tmpVgpr, tmpVgprSize, tmpVgprDynamic, \
                               batchIdx, ss, gwvw, batchElements, \
                               beta, edge, sumIdxGSUSYNC, addrCalc):
        module = Module("GSU Off globalWriteBatchProlog")
        return module

    def defineAndResources(self, writer, kernel, tmpSgpr0, tmpSgprM, tmpSgprN, tmpSgprNumWG0, tmpSgprAccumTiles):
        module = Module("GSU Off defineAndResources")
        return module

    def writeBiasToGlobal(self, writer, kernel, biasDataType, tP, tmpSgprRes, biasBpe):
        module = Module("GSU Off writeBiasToGlobal")
        return module

    def initializeSrd(self, writer, ArgTypeCheckLabel, GeneralBatchedGemmSrdInitiation_End, kernel, ch):
        module = Module("GSU Off initializeSrd")
        return module

    def routeToGeneralBatchedOrStridedBatched(self, stridedBatchedGemmLoad, multipleBufferChecks, generalBatchedGemmLoad, mat, kernel, tmpS1):
        module = Module("GSU Off routeToGeneralBatchedOrStridedBatched")
        return module
        
class GSUOn(GSU):
    kernel = {"GlobalSplitUAlgorithm": ["MultipleBuffer", "MultipleBufferSingleKernel"]}
    # if GSU <= gsuThreshold, last wg does the reduction and no R/W to WS
    # else, atomic_dec chooses the wg to do the reduction
    gsuThreshold = 2

    @classmethod
    def matches(cls, writer, debug=False):
        return writer.states.kernel["GlobalSplitU"] > 0 or writer.states.kernel["GlobalSplitU"] == -1

    def __call__(self):
        assert(0)

    def graWorkGroup(self, writer, kernel):
        module = Module("GSU On graWorkGroup")

        gsuLabel    = Label(label=writer.labels.getNameInc("GSU"), comment="")
        gsuLabelEnd = Label(label=writer.labels.getNameInc("GSU_End"), comment="")
        with writer.allocTmpSgpr(1, tag="GSU On graWorkGroup_tmpSgprGSU") as tmpSgprGSU:
            module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
            module.add(SCmpEQU32(src0=sgpr(tmpSgprGSU.idx), src1=1, comment="GSU == 1 ?"))
            module.add(SCBranchSCC1(labelName=gsuLabel.getLabelName(), comment="branch if GSU == 1"))

        if (kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel" or kernel["AdaptiveGemmGSUA"] == 1):
            extReadEpilogueLabeltmp    = Label(label=writer.labels.getNameInc("LoadExternalEpilogueStruct"), comment="")
            module.addComment0("Check if custom structure pointer is null")
            if kernel["ProblemType"]["SupportUserArgs"]:
                module.add(SCmpEQU32(src0=sgpr("ArgType"), src1=2, comment="ArgType == 2 ?"))
                module.add(SCBranchSCC0(labelName=extReadEpilogueLabeltmp.getLabelName()))

            with writer.allocTmpSgpr(2,2, tag="GSU On graWorkGroup_tmpSgprD") as tmpSgprD:
                module.add(SMovB64(dst=sgpr(tmpSgprD.idx,2), src=sgpr("AddressD",2), comment="tmp=Output"))
                module.add(SMovB64(dst=sgpr("AddressD",2), src=sgpr("AddressTD",2), comment="D=Workspace"))
                module.add(SMovB64(dst=sgpr("AddressTD",2), src=sgpr(tmpSgprD.idx,2), comment="TD=Output"))
            module.add(extReadEpilogueLabeltmp)

        module.addComment("GSU-not-WGMapRR :nwg1 = (size%s + MT%s - 1) / MT%s;" \
            % (writer.states.tileChar1, writer.states.tileChar1, writer.states.tileChar1))

        tmpVgpr = writer.vgprPool.checkOut(2, tag="GSUOn_graWorkGroup_tmpVgpr")
        tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
        gsuwgmrrLabel    = Label(label=writer.labels.getNameInc("GSUWGMRR"), comment="")
        gsuwgmrrLabelEnd = Label(label=writer.labels.getNameInc("GSUWGMRR_End"), comment="")
        with writer.allocTmpSgpr(1, tag="GSU On graWorkGroup_tmpSgprInfo") as tmpSgprInfo:
            module.add(SAndB32(dst=sgpr(tmpSgprInfo.idx), src0=sgpr("GSU"), src1=hex(0x4000), comment="SCC = (GSUWGMRR == 1) ?"))
            module.add(SCBranchSCC1(labelName=gsuwgmrrLabel.getLabelName(), comment="branch if GSUWGMRR == 1"))
            # wg1       = wg1 / GSU
            # gsuSumIdx = wg1 % GSU
            module.add(SAndB32(dst=sgpr(tmpSgprInfo.idx), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
            module.add(scalarUInt32DivideAndRemainder("WorkGroup1", "WorkGroup1", tmpSgprInfo.idx, "GSUSumIdx", tmpVgprRes, kernel["WavefrontSize"]))
            module.add(SBranch(gsuwgmrrLabelEnd.getLabelName()))
            module.add(gsuwgmrrLabel)
            # gsuSumIdx = wg1 / numWg1
            # wg1       = wg1 % numWg1
            module.add(scalarUInt32DivideAndRemainder("GSUSumIdx", "WorkGroup1", "NumWorkGroups1", "WorkGroup1", tmpVgprRes, kernel["WavefrontSize"]))
            module.add(gsuwgmrrLabelEnd)
        writer.vgprPool.checkIn(tmpVgpr)
        module.add(SMovB32(dst=sgpr("GSULog2BpeC"), src=log2(int(writer.states.bpr * kernel["ProblemType"]["DestDataType"].numRegisters()))))
        module.add(SMovB32(dst=sgpr("GSULog2BpeD"), src=log2(writer.states.bpeCinternal)))

        module.add(SBranch(gsuLabelEnd.getLabelName()))
        module.add(gsuLabel)
        module.add(SMovB64(dst=sgpr("GSUSumIdx", 2), src=0, comment="Set GSUSumIdx to 0"))
        module.add(SMovB32(dst=sgpr("GSULog2BpeC"), src=log2(writer.states.bpeCexternalGSU1)))
        module.add(SMovB32(dst=sgpr("GSULog2BpeD"), src=log2(writer.states.bpeCexternalGSU1)))
        module.add(gsuLabelEnd)

        return module

    def computeLoadSrd(self, writer, kernel, tP, stmp, tileStart):
        module = Module("GSU On computeLoadSrd")

        tc = tP["tensorChar"]
        isgfx950 = kernel["ISA"][:2] == (9, 5)
        # MX scales in a swizzled layout (HostPreSwizzle / InMemorySwizzle)
        # use the per-tensor _DepthU{tc} sequencing because their effective
        # K stride differs from A/B; NoSwizzle MX scales share _DepthU with A/B.
        mxScaleFormat = kernel.get("MXScaleFormat", "NoSwizzle")
        isMxSwizzledScaleLayout = ("MXS" in tc) and mxScaleFormat in ("InMemorySwizzle", "HostPreSwizzle")
        depthU = kernel["DepthU"]
        depthUDiv = kernel["_DepthU%s"%tc] if isMxSwizzledScaleLayout else kernel["_DepthU"]
        # swizzle
        if (tP["isSwizzled"] and tc == 'A'):
            depthUDiv = kernel["DepthU"] * kernel["MatrixInstM"]
        elif (tP["isSwizzled"] and tc == 'B'):
            depthUDiv = kernel["DepthU"] * kernel["MatrixInstN"]

        gsuOffsetStr = "gsuOffset = DepthU*bpeGR*GSUSumIdx"
        divider = 1
        if kernel["ProblemType"]["Sparse"]:
            if (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or \
                (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]) :
                divider = 2
            elif tP["isM"]:
                divider = 8
            if divider != 1:
                depthUDiv = depthU // divider
                gsuOffsetStr = "gsuOffset = DepthU/%s*bpeGR*GSUSumIdx"%(divider)

        gsucLabel    = Label(label=writer.labels.getNameInc(f"GSUC_{tc}"), comment="")
        gsucLabelEnd = Label(label=writer.labels.getNameInc(f"GSUC_{tc}_End"), comment="")
        module.add(SAndB32(dst=sgpr(stmp), src0=sgpr("GSU"), src1=hex(0x8000), comment="SCC = (GSUC == 1) ?"))
        module.add(SCBranchSCC1(labelName=gsucLabel.getLabelName(), comment="branch if GSUC == 1"))
        gsuOffsetStr = "gsuOffset = DepthU*GSUSumIdx"
        module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), depthUDiv, sgpr("GSUSumIdx"), comment=gsuOffsetStr))
        module.add(SBranch(gsucLabelEnd.getLabelName()))
        module.add(gsucLabel)
        gsuOffsetStr = "gsuOffset = DepthU*accumulatedNumOfLoopCounterL"
        loopCounterName = writer.loopCounterName(kernel, writer.states.unrollIdx)
        module.add(SLShiftRightB32(dst=sgpr(loopCounterName), src=sgpr("SizesSum"), shiftHex=log2(depthU), \
                                    comment="s[%s] = s[sgprSizesSum] / %s"%(loopCounterName, depthU)))
        tmpSgprInfo = ContinuousRegister(idx=stmp, size=2)
        module.add(writer.calculateLoopNumIterOffsetGsu(kernel, loopCounterName, tmpSgprInfo))
        module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), sgpr(stmp+0), depthUDiv, comment=gsuOffsetStr))
        module.add(gsucLabelEnd)

        unrollSummation = [ i for i in tP["ia"] if i in kernel["ProblemType"]["IndicesSummation"] ]
        stride = writer.strideRef(tc, unrollSummation[-1])
        if tP["tlu"] and not writer.isConstUnitStride(stride):
            # non-transpose case, unroll is in perp dim and should be scaled by unroll Stride
            module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(stmp), sgpr(stmp+1), sgpr(stmp+0), \
                stride, comment="tlu=1, scaled unroll-offset by stride"))

        module.add(SAddU32(dst=sgpr(tileStart+0), src0=sgpr(tileStart+0), src1=sgpr(stmp+0), comment="accum GsuOffset term to tilestart"))
        module.add(SAddCU32(dst=sgpr(tileStart+1), src0=sgpr(tileStart+1), src1=sgpr(stmp+1), comment="accum GsuOffset term to tilestart"))

        return module

    def graIncrements(self, writer, kernel, loopIdx, tP):
        module = Module("GSU On graIncrements")

        tc = tP["tensorChar"]
        tIdx: int = tP["idx"]
        dimIdx = kernel["ProblemType"]["IndicesSummation"][loopIdx] # dimension index
        loopChar = writer.states.indexChars[dimIdx]
        stride = writer.strideRef(tc, dimIdx)
        isMirrorIdx = dimIdx in kernel["ProblemType"]["MirrorDims%s"%tc]

        if writer.states.globalReadIncsUseVgpr:
            with writer.allocTmpSgpr(3, tag="GSUOn_graIncrements_tmpSgprInfo") as tmpSgprInfo:
                tmpSgpr = tmpSgprInfo.idx
                gsuSgpr = tmpSgpr + 2
                du = kernel["_DepthU%s"%tc]
                duBpe = int(du * tP["bpeGR"])
                module.add(SAndB32(dst=sgpr(tmpSgpr), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
                module.add(SMulI32(dst=sgpr(gsuSgpr), src0=sgpr(tmpSgpr), src1=duBpe, comment="GSU*DepthU*Bpe"))
                module.add(SAndB32(dst=sgpr(tmpSgpr), src0=sgpr("GSU"), src1=hex(0x8000), comment="SCC = (GSUC == 1) ?"))
                module.add(SCMovB32(dst=sgpr(gsuSgpr), src=duBpe, comment="DepthU*Bpe if GSUC = 1"))
                module.add(SMulI32(dst=sgpr(tmpSgpr+0), src0=sgpr(gsuSgpr), src1=stride, \
                    comment="incr%s%s = %s*DepthU*bpeGR (unrollIdx)"%(tc, loopChar, stride) ))
                # TODO - this should be mul-H??
                module.add(SMovB32(
                    dst=sgpr(tmpSgpr+1), \
                    src=0, \
                    comment="(carry)"))
                module.add(VMovB32(
                    dst=vgpr("GlobalReadIncs%s+%u+0"%(tc, 2*loopIdx)), \
                    src=sgpr(tmpSgpr+0)))
                module.add(VMovB32(
                    dst=vgpr("GlobalReadIncs%s+%u+1"%(tc, 2*loopIdx)), \
                    src=sgpr(tmpSgpr+1)))
        else:
            with writer.allocTmpSgpr(2, tag="GSUOn_graIncrements_tmpSgprInfo2") as tmpSgprInfo:
                incr = sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx))
                tmpSgpr = tmpSgprInfo.idx
                gsuSgpr = tmpSgpr + 1
                # GlobalReadIncs doubles as scratch for the intermediate duBpe value;
                # final SCSelectB32/SMulI32 writes the result back to the same register.
                incSgpr = incr

                tcGR = tc if tc == "Metadata" else (tc + "GR")

                # swizzle: resolve MI dimension to numeric value
                mi_dim = 1
                if tc == "A" and kernel["ProblemType"]["SwizzleTensorA"]:
                    mi_dim = kernel["MatrixInstM"]
                elif tc == "B" and kernel["ProblemType"]["SwizzleTensorB"]:
                    mi_dim = kernel["MatrixInstN"]

                du = kernel["_DepthU%s"%tc]
                duBpe = int(du * tP["bpeGR"]) * mi_dim
                module.add(SAndB32(dst=sgpr(gsuSgpr), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))

                # MX scale GSU SRD increment, gated by MXScaleFormat:
                #   - Swizzled (HostPreSwizzle/InMemorySwizzle): the K-block is
                #     M*K_inner contiguous, so the increment scales with Size{tile}.
                #   - NoSwizzle: canonical layout, same step as A/B (duBpe only).
                mxFmt = kernel.get("MXScaleFormat", "NoSwizzle")
                isMxSwizzledScale = ('MXS' in tc) and mxFmt in ("InMemorySwizzle", "HostPreSwizzle")
                if isMxSwizzledScale:
                    module.add(SMulI32(dst=incSgpr, src0=sgpr("Size%s"%INDEX_CHARS[tIdx]), src1=duBpe, comment="GSU*DepthU*Bpe*MI_dim(%d) (swizzled MX scale layout)"%(mi_dim)))
                elif kernel["enableTDMMetadata"] and tP["isM"] and kernel["ProblemType"]["MetadataLayout"]:
                    ia = kernel["ProblemType"]["IndexAssignmentsMetadata"]
                    metadataStrideSgpr = f"Stride{tc}{writer.states.indexChars[ia[1]]}"
                    module.add(SMovB32(dst=incSgpr, src=sgpr(metadataStrideSgpr), comment="incSgpr = stride"))
                    module.add(SMulI32(dst=incSgpr, src0=incSgpr, src1=(du), comment="incSgpr = stride * du"))
                else:
                    module.add(SMovB32(dst=incSgpr, src=duBpe, comment="GSU*DepthU*Bpe*MI_dim(%d)"%(mi_dim)))
                module.add(SMulI32(dst=sgpr(gsuSgpr), src0=sgpr(gsuSgpr), src1=incSgpr, comment="GSU*DepthU*Bpe*MI_dim(%d)"%(mi_dim)))
                module.add(SAndB32(dst=sgpr(tmpSgpr), src0=sgpr("GSU"), src1=hex(0x8000), comment="SCC = (GSUC == 1) ?"))

                m = sgpr(gsuSgpr)

                if isMirrorIdx:
                    m.setMinus(True)

                duBpe = int(du * tP["bpeGR"]) * mi_dim
                # multiply by stride, optimizing if unit stride
                if writer.isConstUnitStride(stride) or (kernel["enableTDMMetadata"] and tP["isM"]):
                    module.add(SCSelectB32(dst=incr, src0=incSgpr, src1=m, comment="incr%s (unrollIdx)"%(tc)))
                else:
                    module.add(SCMovB32(dst=m, src=duBpe, comment="DepthU*Bpe if GSUC = 1"))
                    module.add(SMulI32(dst=incr, src0=m, src1=stride, comment="incr%s unrollIdx)"%(tc) ))

        return module

    def graIncrementsRestore(self, writer, kernel, loopCounterName):
        module = Module("GSU On graIncrementsRestore")

        with writer.allocTmpSgpr(1, tag="GSU On graIncrementsRestore_tmpSgprInfo") as tmpSgprInfo:
            gsuSgpr = tmpSgprInfo.idx
            module.add(SAndB32(dst=sgpr(gsuSgpr), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
            module.add(SMulI32(dst=sgpr(gsuSgpr), src0=sgpr(gsuSgpr), src1=kernel["DepthU"]))
            module.add(SMulI32(dst=sgpr(loopCounterName), src0=sgpr(loopCounterName), \
                               src1=sgpr(gsuSgpr), comment="=loopCounterName*DepthU"))

        return module

    def calculateLoopNumIter(self, writer, kernel, loopCounterName, tmpSgprInfo):
        module = Module("GSU On calculateLoopNumIter")

        tmpSgpr = tmpSgprInfo.idx
        # if GSU numIter++ if gsuSumIdx < remainder
        gsuLabel = Label(label=writer.labels.getNameInc("GSU"), comment="")
        module.add(SAndB32(dst=sgpr(tmpSgpr), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
        module.add(SCmpEQU32(src0=sgpr(tmpSgpr), src1=1, comment="GSU == 1 ?"))
        module.add(SCBranchSCC1(labelName=gsuLabel.getLabelName(), comment="branch if GSU == 1"))
        module.add(writer.calculateLoopNumIterGsu(kernel, loopCounterName, tmpSgprInfo))
        module.add(gsuLabel)

        return module

    ##############################################################################
    # Emit code to compute loop iterations for GSU.
    # See same function in KernelWriterSource.py for background explanation
    # This function is used to compute number of loop iters and also
    # for computing the global read increment for GSU case.
    # For multiple summation, the number of loop iterations needs to be reset
    # for each iteration so replicate the code in addr inc and at open of unroll loop

    # tmpSgpr is allocation of at least 3 tmpSgpr

    # Output: SGPR(destName) contains the number of unroll iterations for
    # this workgroup.
    ##############################################################################
    def calculateLoopNumIterGsu(self, writer, kernel, destName, tmpSgprRes: ContinuousRegister):
        module = Module("calculateLoopNumIterGsu")

        loopCounter = sgpr(destName)
        quotient = destName
        remainder = "GSUSumIdx+1" # numIterPerWgRemainder
        dividend = destName

        tmpVgpr = writer.vgprPool.checkOut(2, tag="GSUOn_calculateLoopNumIter_tmpVgpr")
        tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
        module.add(scalarUInt32DivideAndRemainder(quotient, dividend, "GSU", remainder, tmpVgprRes, wavewidth=kernel["WavefrontSize"]))
        writer.vgprPool.checkIn(tmpVgpr)

        # if gsuSumIdx < numIterPerWgRemainder
        module.add(SAddU32(dst=sgpr(tmpSgprRes.idx), src0=1, \
            src1=loopCounter, comment="tmp<-numIterMyWg+" ))
        module.add(SCmpLtU32(src0=sgpr("GSUSumIdx"), src1=sgpr("GSUSumIdx+1"), \
            comment="gsuSumIdx < numIterPerWgRemainder" ))
        module.add(SCMovB32(dst=loopCounter, src=sgpr(tmpSgprRes.idx), comment="numIterMyWg++ if needed"))

        return module

    def calculateIncrementMetadata(self, writer, kernel, sgprOut):
        module = Module("GSU On calculateLoopNumIter")
        with writer.allocTmpSgpr(1, tag="GSU On calculateIncrementMetadata_tmpSgprGSU") as tmpSgprGSU:
            module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
            module.add(SMulI32(dst=sgpr(sgprOut), src0=kernel["DepthU"], src1=sgpr(tmpSgprGSU.idx), comment="IncsMetadata = GSU*DepthU"))
            module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=hex(0x8000), comment="SCC = (GSUC == 1) ?"))
        module.add(SCMovB32(dst=sgpr(sgprOut), src=kernel["DepthU"], comment="IncsMetadata = DepthU if GSUC == 1"))
        module.add(SLShiftRightB32(dst=sgpr(sgprOut), shiftHex=hex(log2(8)), src=sgpr(sgprOut)))
        return module

    def computeStoreSrdStart(self, writer, kernel):
        module = Module("GSU On computeStoreSrdStart")
        module.add(self.computeStoreSrdStartCommon(writer, kernel))
        return module

    def noLoadLoop(self, writer, kernel, tensorParametersA, tensorParametersB, pack, packPre):
        module = Module("GSU On noLoadLoop")

        isDTV = (kernel["DirectToVgprA"] or kernel["DirectToVgprB"])
        needSecondNLL  = isDTV # need 2 NLL for 2 buffers (PGR1/2)
        NLLnum = 2 if needSecondNLL else 1
        gsuLabel = Label(label=writer.labels.getNameInc("GSU"), comment="")
        with writer.allocTmpSgpr(1, tag="GSU On noLoadLoop_tmpSgprGSU") as tmpSgprGSU:
            module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
            module.add(SCmpEQU32(src0=sgpr(tmpSgprGSU.idx), src1=1, comment="GSU == 1 ?"))
        noLoadLoopModules = None
        acclen = 0
        gsuBackup          = kernel["GlobalSplitU"]
        gsuAccumBackup     = kernel["_GlobalAccumulation"]
        bpeCexternalBackup = writer.states.bpeCexternal
        kernel["GlobalSplitU"] = 1
        kernel["_GlobalAccumulation"] = None
        writer.states.bpeCexternal = writer.states.bpeCexternalGSU1
        if kernel["KernelLanguage"] == "Assembly" and kernel["OptNoLoadLoop"] and \
            kernel["BufferLoad"] and kernel["BufferStore"] and writer.states.doShadowInit and \
            kernel["LocalSplitU"]==1 and \
            writer.states.actualSummationLoops==1:

            # two different noLoadLoops:
            # 1. OptNLL & PAP global-read interleaved (only for PAP=ON)
            # (2. OptNLL : No PAP global-read (For PAP=OFF, or PAP=ON but the last tile))
            #  -> this is unified with 1. global-read is invalidated at the last tile.
            # 3. OrdinaryNLL (Not Opt.)

            noLoadLoopModules = Module("noLoadLoop")
            for NLLindex in range(0, NLLnum):
              writer.saveLocalPointers(kernel, tensorParametersA, tensorParametersB)
              # copy pack
              if NLLindex == NLLnum - 1 or (writer.states.packDTVA or writer.states.packDTVB or writer.states.convDTVA or writer.states.convDTVB):
                # last NLL or  pack DTV case, no deep copy for pack
                # pack code for local prefetch is generated in noLoadLoopBody and used for DTV even
                deepCopyPack = pack
                deepCopyPackPre = packPre
              else:
                # deepCopy packCode for OptNLL noLoadLoop
                deepCopyPack = deepcopy(pack)
                deepCopyPackPre = deepcopy(packPre)
              noLoadLoopModules.add(writer.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=True, isNGLL=False, pack=deepCopyPack, packPre=deepCopyPackPre, NLLindex=NLLindex, NLLnum=NLLnum))
              writer.restoreLocalPointers(kernel, tensorParametersA, tensorParametersB)

            acclen = countInstruction(noLoadLoopModules)
        kernel["GlobalSplitU"] = gsuBackup
        kernel["_GlobalAccumulation"] = gsuAccumBackup
        writer.states.bpeCexternal = bpeCexternalBackup

        if acclen > 16384:
            with writer.allocTmpSgpr(3, tag="GSU On noLoadLoop_tmpSgprInfo") as tmpSgprInfo:
                module.add(writer.longBranchScc0(gsuLabel, posNeg=1, tmpSgprInfo=tmpSgprInfo, comment="branch if GSU != 1"))
        else:
            module.add(SCBranchSCC0(labelName=gsuLabel.getLabelName(), comment="branch if GSU != 1"))

        if noLoadLoopModules != None:
            module.add(noLoadLoopModules)
        module.add(gsuLabel)

        return module

    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("GSU On tailLoopNumIter")

        with writer.allocTmpSgpr(3, tag="GSU On tailLoopNumIter_tmpSgprInfo") as tmpSgprInfo:
            tmpSgpr = tmpSgprInfo.idx
            remainder    = "GSUSumIdx+1" # numIterPerWgRemainder
            gsucLabel    = Label(label=writer.labels.getNameInc("GSUC_TL"), comment="")
            gsucLabelEnd = Label(label=writer.labels.getNameInc("GSUC_TL_End"), comment="")
            module.add(SAndB32(dst=sgpr(tmpSgpr), src0=sgpr("GSU"), src1=hex(0x8000), comment="SCC = (GSUC == 1) ?"))
            module.add(SCBranchSCC1(labelName=gsucLabel.getLabelName(), comment="branch if GSUC == 1"))
            # if GSU numIter=0 if gsuSumIdx != numIterPerWgRemainder
            module.add(SCmpLgU32(src0=sgpr("GSUSumIdx"), src1=sgpr("GSUSumIdx+1"), comment="gsuSumIdx == numIterPerWgRemainder"))
            module.add(SCMovB32(dst=loopCounter, src=0, comment="numIter=0 if gsuSimIdx != numIterPerWgRemainder"))
            module.add(SBranch(gsucLabelEnd.getLabelName()))
            module.add(gsucLabel)
            # calculate the lastWg
            tmpVgpr = writer.vgprPool.checkOut(2, tag="GSUOn_tailLoopNumIter_tmpVgpr")
            tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
            module.add(SLShiftRightB32(dst=sgpr(tmpSgpr+1), src=sgpr("SizesSum"), shiftHex=log2(kernel["DepthU"]), \
                                            comment="s%s = s[sgprSizesSum] / %s"%(tmpSgpr+1,kernel["DepthU"])))
            module.add(SAndB32(dst=sgpr(tmpSgpr+2), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
            module.add(scalarUInt32DivideAndRemainder(tmpSgpr, tmpSgpr+1, tmpSgpr+2, remainder, tmpVgprRes, kernel["WavefrontSize"]))
            module.add(SSubU32(dst=sgpr(tmpSgpr+1), src0=sgpr(tmpSgpr+2), src1=1, comment="GSU-1"))
            writer.vgprPool.checkIn(tmpVgpr)
            module.add(SCmpEQU32(src0=sgpr(tmpSgpr), src1=0, comment="quotient == 0"))
            module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=sgpr("GSUSumIdx+1"), src1=sgpr(tmpSgpr+1), \
                                    comment="lastWg = (quotient==0) ? numIterPerWgRemainder : GSU-1"))
            # if GSU numIter=0 if gsuSumIdx != lastWg
            module.add(SCmpLgU32(src0=sgpr("GSUSumIdx"), src1=sgpr(tmpSgpr), comment="gsuSumIdx == lastWg"))
            module.add(SCMovB32(dst=loopCounter, src=0, comment="numIter=0 if gsuSumIdx != lastWg"))
            module.add(gsucLabelEnd)

        return module

    def setupNewTile(self, writer, kernel, tensorParametersA, tensorParametersB, tPM):
        module = Module("GSU On setupNewTile")

        addBranch = False
        for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
            if i != writer.states.unrollIdx:
                addBranch = True
                break
        if addBranch:
            gsuBackup   = kernel["GlobalSplitU"]
            gsuLabel    = Label(label=writer.labels.getNameInc("GSU"), comment="")
            gsuLabelEnd = Label(label=writer.labels.getNameInc("GSU_End"), comment="")
            with writer.allocTmpSgpr(1, tag="GSU On setupNewTile_tmpSgprGSU") as tmpSgprGSU:
                module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
                module.add(SCmpEQU32(src0=sgpr(tmpSgprGSU.idx), src1=1, comment="GSU == 1 ?"))
            module.add(SCBranchSCC1(labelName=gsuLabel.getLabelName(), comment="branch if GSU == 1"))
            module.addComment1("global read addresses: increments a")
            kernel["GlobalSplitU"] = 2
        module.add(self.graIncrementsAB(writer, kernel, tensorParametersA, tensorParametersB, tPM))
        if addBranch:
            module.add(SBranch(gsuLabelEnd.getLabelName()))
            module.add(gsuLabel)
            kernel["GlobalSplitU"] = 1
            module.add(self.graIncrementsAB(writer, kernel, tensorParametersA, tensorParametersB, tPM))
            kernel["GlobalSplitU"] = gsuBackup
            module.add(gsuLabelEnd)

        return module

    def globalWriteBatchProlog(self, writer, kernel, tmpVgpr, tmpVgprSize, tmpVgprDynamic, \
                               batchIdx, ss, gwvw, batchElements, \
                               beta, edge, sumIdxGSUSYNC, addrCalc):
        module = Module("GSU On globalWriteBatchProlog")

        if kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
            if writer.states.serializedStore:
                module.add(SNop(0, "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst"))

        return module

    def defineAndResources(self, writer, kernel, tmpSgpr0, tmpSgprM, tmpSgprN, tmpSgprNumWG0, tmpSgprAccumTiles):
        module = Module("GSU On defineAndResources")

        if (kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel" or kernel["AdaptiveGemmGSUA"] == 1):
            extReadEpilogueLabeltmp    = Label(label=writer.labels.getNameInc("LoadExternalEpilogueStruct"), comment="")
            module.addComment0("Check if custom structure pointer is null")
            if kernel["ProblemType"]["SupportUserArgs"]:
                module.add(SCmpEQU32(src0=sgpr("ArgType"), src1=2, comment="ArgType == 2 ?"))
                module.add(SCBranchSCC0(labelName=extReadEpilogueLabeltmp.getLabelName()))
            module.add(SMulI32(dst=sgpr(tmpSgpr0), src0=sgpr(tmpSgprAccumTiles), src1="MTOffset", comment="accumNumTiles*(MT0*MT1*bpeC)"))
            module.add(SAddU32(dst=sgpr("AddressTD"), src0=sgpr("AddressTD"), src1=sgpr(tmpSgpr0)))
            module.add(SAddCU32(dst=sgpr("AddressTD+1"), src0=sgpr("AddressTD+1"), src1=0))
            module.add(SAddU32(dst=sgpr("Synchronizer"), src0=sgpr("Synchronizer"), src1=hex(1638400)))
            module.add(SAddCU32(dst=sgpr("Synchronizer+1"), src0=sgpr("Synchronizer+1"), src1=0))
            module.add(extReadEpilogueLabeltmp)

        return module

    def writeBiasToGlobal(self, writer, kernel, biasDataType, tP, tmpSgprRes, biasBpe):
        module = Module("GSU On writeBiasToGlobal")

        if (kernel["GlobalSplitU"] > 1 or kernel["GlobalSplitU"] == -1) and not (kernel["GlobalSplitUAlgorithm"] == "SingleBuffer" and kernel["ProblemType"]["ComputeDataType"] == biasDataType):
            '''
            We use num_records to save the bias data, so we have to shift the global pointer.
            final offset = d_size * gsu + sizeI/J * gsuIdx
            '''
            assert tmpSgprRes.size >= 4
            tmpSgpr = tmpSgprRes.idx
            #Calculate tensor 2d size
            module.add(SMovB64(dst=sgpr(tmpSgpr+0, 2), src=0x1, comment="Init tensor size"))
            indices = [i for i in range(kernel["ProblemType"]["NumIndicesC"])]
            numDim = len(indices)
            for i in range(0, numDim):
                idx = indices[i]
                stride = writer.strideRef("D",idx)
                size =   writer.sizeRef(idx)
                module.add(SSubU32(dst=sgpr(tmpSgpr+2), src0=size, src1=0x1, comment="(size-1)"))
                module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgpr+2), sgpr(tmpSgpr+3), stride, \
                            sgpr(tmpSgpr+2), comment="stride x (size-1)"))
                module.add(SAddU32(dst=sgpr(tmpSgpr+0), src0=sgpr(tmpSgpr+0), src1=sgpr(tmpSgpr+2), comment="sum tensor size"))
                module.add(SAddCU32(dst=sgpr(tmpSgpr+1), src0=sgpr(tmpSgpr+1), src1=sgpr(tmpSgpr+3), comment="sum tensor size"))
            # SingleBuffer works on the same work space for every gsu
            if kernel["_GlobalAccumulation"] == "MultipleBuffer":
                module.add(SAndB32(dst=sgpr(tmpSgpr+2), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
                module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgpr+0), sgpr(tmpSgpr+1), sgpr(tmpSgpr+2), \
                                sgpr(tmpSgpr+0), comment="Recalculate gsu stride (size * gsu)"))
                module.add(SMovB32(dst=sgpr(tmpSgpr+2), src=sgpr("GSUSumIdx"), comment="Init tensor size"))
                module.add(SMovB32(dst=sgpr(tmpSgpr+3), src=0x0, comment="Init tensor size"))
                module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgpr+2), sgpr(tmpSgpr+3), writer.sizeRef(tP["idx"]), \
                                sgpr(tmpSgpr+2), comment="Reduction GSU offset *stride"))
                module.add(SAddU32(dst=sgpr(tmpSgpr+0), src0=sgpr(tmpSgpr+0), src1=sgpr(tmpSgpr+2), comment="sum gsu offset"))
                module.add(SAddCU32(dst=sgpr(tmpSgpr+1), src0=sgpr(tmpSgpr+1), src1=sgpr(tmpSgpr+3), comment="sum gsu offset"))
            module.add(scalarStaticMultiply64(sgpr(tmpSgpr, 2), sgpr(tmpSgpr, 2), biasBpe, None, comment="stride * bpe"))
            module.add(SAddU32(dst=sgpr("SrdBias+0"), src0=sgpr("SrdBias+0"), src1=sgpr(tmpSgpr), comment="Recalculate start address for GSU."))
            module.add(SAddCU32(dst=sgpr("SrdBias+1"), src0=sgpr("SrdBias+1"), src1=sgpr(tmpSgpr+1), comment="Recalculate start address for GSU."))

        return module

    def reductionBranches(self, writer, kernel, tPB, vectorWidths, elements, tmpVgpr, cvtVgprStruct, vectorDataTypes, factorDims, reductionEndLabel, endLabel):
        module = Module("GSU On reductionBranches")

        edges = [False] # no edge variant
        alphas = [False] # no ahpla variant
        betas = [False] # no beta variant
        edgeI = edges[0]
        alpha = alphas[0]
        beta = betas[0]
        gwvw = vectorWidths[edgeI]
        atomic = (kernel["GlobalSplitU"] > 1) and (kernel["_GlobalAccumulation"] != 'MultipleBuffer' and kernel["_GlobalAccumulation"] != 'MultipleBufferSingleKernel')

        module.add(self.reductionProcedure(writer, kernel, elements, alpha, beta, edges, atomic, gwvw, tmpVgpr, cvtVgprStruct, vectorDataTypes, reductionEndLabel, endLabel))

        return module

    def reductionProcedure(self, writer, kernel, elements, alpha, beta, edges, atomic, gwvw, tmpVgpr, cvtVgprStruct, vectorDataTypes, reductionEndLabel, endLabel):
        module = Module("GSU Common reductionProcedure")

        reductionLabels = {}
        for edge in edges:
            reductionLabels[edge] = Label(writer.labels.getNameInc("Reduction_B%u_E%u" % (1 if beta else 0, 1 if edge else 0)), comment="")

        for edge in edges:
            # write label for batch case
            module.add(reductionLabels[edge])

            # PreLoopVmcntCaseStr = ""
            # # not generate Case 2 if StoreCInUnroll with StoreVectorWidth==1 (Case 2 will be same as Case 3)
            # if self.canOptimizePreLoopLWVmcnt:
            #     if edge or (kernel["StoreCInUnroll"] and kernel["StoreVectorWidth"]==1):
            #         self.currPreLoopVmcntCase = PreLoopVmcntCase.OrdNLL_E1_Store
            #     else:
            #         self.currPreLoopVmcntCase = PreLoopVmcntCase.OptNLL_Store
            #     PreLoopVmcntCaseStr = inst("s_mov_b32", sgpr("PreLoopLWVmcntCase"), hex(self.currPreLoopVmcntCase.value), \
            #         "for optimizing next PreLoop LW vmcnt, set to Case%u"%self.currPreLoopVmcntCase.value)
            #     # reset vmcnt if the dict has this key (OptNLL_Store, OrdNLL_E1_Store),
            #     # OrdNLL_B1_Store is excluded
            #     if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
            #         self.preLoopVmcntDict[self.currPreLoopVmcntCase] = 0

            edgeI = edge # TODO: remove?

            ########################################
            # Calculate Vgprs for Write Batching
            ########################################
            writer.vgprPool.resetOccupancyLimit()
            writer.sgprPool.resetOccupancyLimit()

            # Temporarily grow pool for sgpr
            sgprList = []
            if kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
                sgprList.append(writer.sgprPool.checkOut(1, tag="reductionProcedure_sgprList1", preventOverflow=False))
                sgprList.append(writer.sgprPool.checkOut(1, tag="reductionProcedure_sgprList2", preventOverflow=False))
                sgprList.append(writer.sgprPool.checkOut(1, tag="reductionProcedure_sgprList3", preventOverflow=False))
                sgprList.append(writer.sgprPool.checkOutAligned(2, 2, tag="reductionProcedure_sgprList4", preventOverflow=False))
                sgprList.append(writer.sgprPool.checkOutAligned(2, 2, tag="reductionProcedure_sgprList5", preventOverflow=False))
                sgprList.append(writer.sgprPool.checkOutAligned(4, 4, tag="reductionProcedure_sgprList6", preventOverflow=False))
                for s in sgprList:
                    writer.sgprPool.checkIn(s)

            tmpVgprDynamic = None
            tmpVgprDynamicSize  = 0
            tmpVgprDynamicAlign = 0
            if kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
                GSUTotal = writer.getMBSKGSUTotal(kernel)
                vgprMbsk = (GSUTotal-1) * gwvw * max(1, kernel["ProblemType"]["DestDataType"].numRegisters())
                tmpVgprDynamicSize  = vgprMbsk
                tmpVgprDynamicAlign = 4
            if tmpVgprDynamicSize > 0:
                tmpVgprDynamic = ContinuousRegister(idx=writer.vgprPool.checkOutAligned(tmpVgprDynamicSize, tmpVgprDynamicAlign, tag="reductionProcedure_tmpVgprDynamic"), size=tmpVgprDynamicSize)

            ss = StoreState(writer, kernel, gwvw, edge, True, atomic, elements[edgeI], vectorDataTypes, dim=0, isWorkspace=True)

            # how many vgprs are needed for zero elements
            # 2 for addressC in vgpr for addition - already checked out
            # 2 for coord0,1 of thread - already checked out
            # 2 for tmp - already checked out

            # 5 = how many vgprs are needed per element (flat)
            #    - 2 for addr
            #    - 3 for GLOBAL_OFFSET_C calculation (can overlap below, therefore max)
            #    - if beta gwvw*rpe for new value
            #    - if atomic 2*rpe for old and cmp values

            # print("numVgprsPerAddr=%u, numVgprsPerDataPerVI=%u, numVgprPerValuC=%u"%(ss.cfg.numVgprsPerAddr, ss.cfg.numVgprsPerDataPerVI, ss.cfg.numVgprPerValuC))
            # numVgprsPerElement = ss.cfg.numVgprPerValuC*gwvw + ss.cfg.numVgprsPerAddr + int(ceil(ss.cfg.numVgprsPerDataPerVI * gwvw))

            # if kernel["GroupLoadStore"] and kernel["ProblemType"]["UseBeta"]:
            #     numVgprsPerElement += ss.cfg.numVgprsPerAddr

            # Get estimated numVgprAvailable
            # print("Max vgprs =", maxVgprs, writer.vgprPool.size(), writer.vgprPool.availableBlock(ss.numVgprsPerElement, ss.align))
            numVgprAvailable = writer.vgprPool.availableBlock(ss.numVgprsPerElement, ss.align)

            # Grow the register pool if needed - we need enough regs for at least one element
            # Unfortunate since this means the write logic is setting the VGPR requirement
            # for the entire kernel but at least we have a functional kernel.
            # Before growing the pool, see if we can shrink the write vector width instead?
            # TODO: the vgprSerial is needed for-ever and if we grow here will split the
            # range of the tmps.    Maybe want to move vgprSerial to first vgpr?

            # TODO: Minimum elems for StoreRemap
            # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
            minElements = int(4 / kernel["ProblemType"]["ComputeDataType"].numBytes())
            minNeeded = minElements * ss.numVgprsPerElement

            gsuDebug = 0
            if gsuDebug:
                print("numVgprAvailable=", numVgprAvailable, "minElements=", minElements, "minNeeded=", minNeeded)

            if numVgprAvailable < minNeeded:
                gwvwOrig = gwvw
                currentOccupancy = writer.getOccupancy(kernel["NumThreads"], writer.vgprPool.size(), \
                        writer.sgprPool.size(), writer.getLdsSize(kernel), writer.agprPool.size(), writer.states.doubleVgpr)
                futureOccupancy = writer.getOccupancy(kernel["NumThreads"], writer.vgprPool.size() - numVgprAvailable + minNeeded, \
                        writer.sgprPool.size(), writer.getLdsSize(kernel), writer.agprPool.size(), writer.states.doubleVgpr)

                if gsuDebug:
                    print("currentOccupancy=%u futureOccupancy=%u VGPRs=%u numVgprAvail=%u vgprPerElem=%u" \
                        % (currentOccupancy, futureOccupancy, writer.vgprPool.size(), \
                        numVgprAvailable, minNeeded))
                if futureOccupancy > currentOccupancy:
                    if gsuDebug:
                        print("warning: %s growing VGPR for GlobalWrite batching - this may bloat VGPR usage" % \
                            (writer.states.kernelName))
                        print("     numVgprAvailable=", numVgprAvailable, \
                            "numVgprsPerElement=", ss.numVgprsPerElement, \
                            "gwvw=", gwvw)
                elif gwvw != gwvwOrig:
                    ss.gwvw = gwvw # make both representations consistent
                    if gsuDebug:
                        print2("info: %s shrank gwvw from %u to %u but kept occupancy same=%u." \
                            % (writer.states.kernelName, gwvwOrig, gwvw, currentOccupancy))

                if numVgprAvailable < minNeeded:
                    print2("info: growing pool += %d * %d for GlobalWrite\n" \
                        % (minElements,ss.numVgprsPerElement))
                    print2(writer.vgprPool.state())
                    writer.vgprPool.growPool(0, minElements, ss.numVgprsPerElement, \
                        "grow-pool for GlobalWrite")
                    numVgprAvailable = writer.vgprPool.available()
                    print2(writer.vgprPool.state())

            # Use VGPR up to next occupancy threshold:
            maxVgprs, occupancy = writer.getMaxRegsForOccupancy(kernel["NumThreads"], writer.vgprPool.size(), writer.sgprPool.size(), \
                writer.getLdsSize(kernel), writer.agprPool.size(), writer.states.doubleVgpr)
            # Set occupancy limit for register pools
            # TODO: Support gfx12
            if kernel["ISA"][0] != 12:
                writer.vgprPool.setOccupancyLimit(writer.states.regCaps["MaxVgpr"], writer.states.regCaps["PhysicalMaxVgpr"] // occupancy)
                writer.sgprPool.setOccupancyLimit(writer.states.regCaps["MaxSgpr"], writer.states.regCaps["PhysicalMaxSgpr"] // occupancy)

            if ss.numVgprsPerElement:
                numElementsPerBatch = numVgprAvailable // ss.numVgprsPerElement
            else:
                numElementsPerBatch = len(elements[edgeI]) # max, do 'em all

            # assert(self.numVgprValuC % gwvw == 0) # sanity check

            numElementsPerBatch = numElementsPerBatch if not kernel["NumElementsPerBatchStore"] else min(kernel["NumElementsPerBatchStore"],numElementsPerBatch)

            if gsuDebug:
                print("NumElementsPerBatch=", numElementsPerBatch, "LimitedBySgprs=", ss.cfg.numElementsPerBatchLimitedBySgprs, \
                        "WARNING" if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch else "okay")

            if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch:
                numElementsPerBatch = ss.cfg.numElementsPerBatchLimitedBySgprs

            # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
            if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
                # only do an even number of halves - since these share hi/lo pieces of some registers?
                if numElementsPerBatch > 1:
                    numElementsPerBatch = int(numElementsPerBatch/2)*2
                elif not kernel["EnableMatrixInstruction"]:
                    # (excluding MFMA+LSU case. It can work without an issue)
                    # The globalWriteBatch routine below can't handle odd elements per batch
                    # and 0 elements per batch is illegal.
                    # so if we don't have *GPR resources to handle a larger batch then need
                    # to mark overflowedResources rather than generate a kernel that won't work.
                    # It might be possible to fix globalWriteBatch to handle this case but these
                    # are likely to be low-performing so likely not worth optimizing.
                    print("WARNING: half requires at least two elements per batch")
                    self.overflowedResources = 3
            # elif kernel["ProblemType"]["MacDataTypeA"].is8bitFloat():
            #    if numElementsPerBatch > 1:
            #        numElementsPerBatch = int(numElementsPerBatch/4)*4

            assert numElementsPerBatch > 0, "numElementsPerBatch=0 for %s"%writer.states.kernelName

            # if no atomics and no edge, then write whole vectors
            # if not atomic and not edge:
            #  numVectorsPerBatch = numElementsPerBatch / kernel["GlobalWriteVectorWidth"]
            #  #print "  NumVectorsPerBatch", numVectorsPerBatch
            #  numElementsPerBatch = numVectorsPerBatch * kernel["GlobalWriteVectorWidth"]
            numBatches = max(1, ceilDivide(len(elements[edgeI]),numElementsPerBatch))
            numSgprs = ss.cfg.numTempSgprPerBatch + ss.cfg.numMaskSgprPerBatch + ss.cfg.numMaskSgprPerElement * numElementsPerBatch

            if writer.db["PrintStoreRegisterDb"]:
                print("edgeI", edgeI, "NumBatches", numBatches, "NumElementsPerBatch", numElementsPerBatch, "numVgprsPerElement", ss.numVgprsPerElement, "len(elements[edgeI])", len(elements[edgeI]))
                print ("numSgprs=", numSgprs, "sgprPool.size()=", writer.sgprPool.size(), "numTempSgprPerBatch=", ss.cfg.numTempSgprPerBatch,
                    "numMaskSgprPerBatch=", ss.cfg.numMaskSgprPerBatch, "numMaskSgprPerElement=", ss.cfg.numMaskSgprPerElement)
                print(writer.sgprPool.state())
            module.addComment1("edge=%d, allocate %u sgpr. perBatchTmpS=%u perBatchMaskS=%u perElementMaskS=%u elementsPerBatch=%u" %
                    (edgeI, numSgprs, ss.cfg.numTempSgprPerBatch, ss.cfg.numMaskSgprPerBatch, ss.cfg.numMaskSgprPerElement, numElementsPerBatch))

            # GSU code below needs both tmpSgpr.idx (storeOffsetSgpr) and
            # tmpSgpr.idx+1 (loadOffsetSgpr), so reserve at least 2 sgprs.
            allocNumSgprs = max(numSgprs, 2)
            with writer.allocTmpSgpr(allocNumSgprs, 2, tag="GSU On globalWriteBatch_tmpSgpr") as tmpSgpr:
                elementSgprs = tmpSgpr.idx + ss.cfg.numTempSgprPerBatch
                codeAccVgprRead = deepcopy(writer.codes.accVgprRead) if writer.states.serializedStore else None
                codeAccVgprWrite = deepcopy(writer.codes.accVgprWrite) if writer.states.serializedStore else None
                codeAccVgprReadBackup = deepcopy(codeAccVgprRead)

                if kernel["MIArchVgpr"] and alpha and not kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
                    codeAccVgprRead = None
                    # Only apply when 2 wave optimization features are enabled
                    if (kernel["StorePriorityOpt"] or kernel["StoreSyncOpt"]) and beta:
                        self.alphaBeforeLoadC = True

                # calculate address in workspace
                module.add(self.computeWorkspaceSrd(writer, kernel, tmpSgpr))

                lastGsuWgBusyWaitingLabel = Label(writer.labels.getNameInc("last_gsu_wg_busy_waiting"), comment="")
                reductionBodyLabel = Label(writer.labels.getNameInc("reduction_body"), comment="")
                partialWriteLabel = Label(writer.labels.getNameInc("partial_write"), comment="")

                module.add(self.syncOffsetPreparation(writer, kernel, tmpSgpr, partialWriteLabel))

                for batchIdx in range(0, numBatches):
                    elementStartIdx = batchIdx * numElementsPerBatch
                    elementStopIdx = min(elementStartIdx + numElementsPerBatch, len(elements[edgeI]))
                    elementsThisBatch = elements[edgeI][elementStartIdx:elementStopIdx]
                    # print("BATCH[%u/%u]: elements[edgeI][%u:%u] VGPRs=%u" % (batchIdx, numBatches, elementStartIdx, elementStopIdx, ss.numVgprsPerElement ))
                    # elementVgprs can be large and should be perfectly tuned to the number of available
                    # VGPRS.    We do not want to accidentally overflow and grow the pool here:
                    module.add(self.partialWriteBatch(writer, kernel, ss, batchIdx, beta, edge, gwvw, elementsThisBatch, tmpVgpr, elementSgprs, \
                                                      tmpSgpr, codeAccVgprRead, lastGsuWgBusyWaitingLabel, reductionBodyLabel, partialWriteLabel))

                # synchronize GSUWG in reductionBatch. If is the last WG -> do reduction; else branch to GW_END
                ss.firstBatch = True
                for batchIdx in range(0, numBatches):
                    elementStartIdx = batchIdx * numElementsPerBatch
                    elementStopIdx = min(elementStartIdx + numElementsPerBatch, len(elements[edgeI]))
                    elementsThisBatch = elements[edgeI][elementStartIdx:elementStopIdx]
                    # print("BATCH[%u/%u]: elements[edgeI][%u:%u] VGPRs=%u" % (batchIdx, numBatches, elementStartIdx, elementStopIdx, ss.numVgprsPerElement ))
                    # elementVgprs can be large and should be perfectly tuned to the number of available
                    # VGPRS.    We do not want to accidentally overflow and grow the pool here:
                    module.add(self.reductionBatch(writer, kernel, ss, batchIdx, alpha, beta, edge, atomic, \
                            gwvw, elementsThisBatch, tmpVgpr, tmpVgprDynamic, elementSgprs, tmpSgpr, \
                            codeAccVgprReadBackup, codeAccVgprWrite, reductionBodyLabel, endLabel))

                module.add(SWaitCnt(vlcnt=0, comment="wait for buffer_load to finish"))
                if kernel["MbskPrefetchMethod"] == 0:
                    module.add(SAndB32(dst=sgpr(tmpSgpr.idx), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
                    module.add(SCmpGtI32(src0=sgpr(tmpSgpr.idx), src1=self.gsuThreshold, comment="GSU > %u ?" % self.gsuThreshold))
                    module.add(SCBranchSCC1(labelName=reductionEndLabel.getLabelName(), comment="branch if true"))
                    module.addComment("GSU > %u, no need to reset synchronizer" % self.gsuThreshold)
                    module.add(SWaitCnt(kmcnt=0, comment="wait for reset synchronizer to finish"))
                ss.resetState()

            # Free after final vgpr vcalculation
            if tmpVgprDynamic:
                writer.vgprPool.checkIn(tmpVgprDynamic.idx)

        return module

    def syncOffsetPreparation(self, writer, kernel, tmpSgpr, partialWriteLabel):
        module = Module("GSU Common syncOffsetPreparation")

        tmpS01 = writer.sgprPool.checkOut(1, tag="syncOffsetPreparation_tmpS01", preventOverflow=False)
        tmpS02 = writer.sgprPool.checkOut(1, tag="syncOffsetPreparation_tmpS02", preventOverflow=False)
        tmpS03 = writer.sgprPool.checkOut(1, tag="syncOffsetPreparation_tmpS03", preventOverflow=False)

        #####################################synchronizer offset cal and set synchronizer#####################################
        #####################################WaveId+WgId*WaveNum+WgNum*WaveNum*Batch#####################################
        #####################################WgId+WaveId*WgNum+WgNum*WaveNum*Batch#####################################
        module.addComment("synchronizer offset cal")
        module.add(SMulI32(dst=sgpr(tmpS03), src0=sgpr("NumWorkGroups1"), src1=sgpr("NumWorkGroups0"), comment=""))
        module.add(SMulI32(dst=sgpr(tmpS02), src0=sgpr(tmpS03), src1=sgpr("WorkGroup2"), comment=""))
        module.add(SMulI32(dst=sgpr(tmpS01), src0=sgpr("WorkGroup1"), src1=sgpr("NumWorkGroups0"), comment=""))
        module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1=sgpr("WorkGroup0")))
        module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1=sgpr(tmpS02)))
        module.add(VReadfirstlaneB32(dst=sgpr(tmpS02), src=vgpr("Serial")))
        module.add(SMulI32(dst=sgpr(tmpS03), src0=sgpr(tmpS03), src1=sgpr("SizeK"), comment="cal a wave offset"))
        module.add(SLShiftRightB32(dst=sgpr(tmpS02), shiftHex=hex(log2(kernel["WavefrontSize"])), src=sgpr(tmpS02)))
        module.add(SMulI32(dst=sgpr(tmpS02), src0=sgpr(tmpS03), src1=sgpr(tmpS02), comment="wave offset at batch")) # WaveId*WgNum
        module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS02), src1=sgpr(tmpS01))) # WaveId*WgNum+WgId
        module.add(SLShiftLeftB32(dst=sgpr(tmpS01), src=sgpr(tmpS01), shiftHex=hex(2), comment="")) # atomic 32bits
        #####################################set synchronizer#####################################
        module.add(SAddU32(dst=sgpr("SrdSync+0"), src0=sgpr("Synchronizer+0"), src1=sgpr(tmpS01), comment="" ))
        module.add(SAddCU32(dst=sgpr("SrdSync+1"), src0=sgpr("Synchronizer+1"), src1=hex(0), comment="" ))

        if kernel["MbskPrefetchMethod"] == 0:
            module.addSpaceLine()
            module.add(SAndB32(dst=sgpr(tmpS02), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
            module.add(SCmpGtI32(src0=sgpr(tmpS02), src1=self.gsuThreshold, comment="GSU > %u ?" % self.gsuThreshold))
            module.add(SCBranchSCC1(labelName=partialWriteLabel.getLabelName(), comment="branch if true"))
            module.addComment("GSU <= %u, last gsu wg do the reduction" % self.gsuThreshold)
            module.add(SSubU32(dst=sgpr(tmpS02), src0=sgpr(tmpS02), src1=1, comment=""))
            module.add(SCmpEQU32(src0=sgpr("GSUSumIdx"), src1=sgpr(tmpS02), comment="GSUSumIdx == GSU-1 ?"))
            module.add(SCBranchSCC0(labelName=partialWriteLabel.getLabelName(), comment="branch if false"))

        writer.sgprPool.checkIn(tmpS03)
        writer.sgprPool.checkIn(tmpS02)
        writer.sgprPool.checkIn(tmpS01)

        return module

    def lastGsuWgBusyWaiting(self, writer, kernel, ss, tmpSgpr, tmpVgpr, lastGsuWgBusyWaitingLabel, reductionBodyLabel):
        module = Module("GSU Common lastGsuWgBusyWaiting")

        tmpS01 = tmpSgpr.idx

        if writer.states.asmCaps["HasScalarStore"]:
            module.add(SLoadB32(dst=sgpr(tmpS01), base=sgpr("SrdSync",2), soffset=0, smem=SMEMModifiers(glc=True), comment="get atomic_dec value"))
            module.add(SWaitCnt(kmcnt=0, comment="wait for atomic_dec value load"))
            module.add(SCmpEQU32(src0=sgpr(tmpS01), src1=1, comment="last GSU WG?"))
            module.add(SCBranchSCC0(labelName=lastGsuWgBusyWaitingLabel.getLabelName(), comment="branch if false"))
            module.add(SMovB32(dst=sgpr(tmpS01), src=0, comment="reset synchronizer"))
            module.add(SStoreB32(src=sgpr(tmpS01), base=sgpr("SrdSync", 2), soffset=0, smem=SMEMModifiers(glc=True), comment="reset synchronizer"))
        else:
            # Poll+clear via the same atomic channel as the producer's atomic_dec: a plain
            # buffer load/store races with it on gfx1250's CU-partitioned L2. atomic_dec(wrap=0)
            # always writes 0 and returns old, so it atomically reads-and-resets; spin until old==1.
            addrVgpr = writer.vgprPool.checkOutAligned(2, 2, "syncDecAddr")
            dataVgpr = writer.vgprPool.checkOut(1, "syncDecData")  # wrap value = 0
            dstVgpr  = writer.vgprPool.checkOut(1, "syncDecDst")   # atomic return (old)
            # EXEC is wave-width: 1 sgpr on wave32, 2 on wave64
            numExecSgpr = 1 if kernel["WavefrontSize"] == 32 else 2
            SMovExec    = SMovB32 if numExecSgpr == 1 else SMovB64
            execSgpr    = writer.sgprPool.checkOutAligned(numExecSgpr, numExecSgpr, preventOverflow=False)

            module.add(VMovB64(dst=vgpr(addrVgpr, 2), src=sgpr("SrdSync", 2), comment="synchronizer atomic address"))
            module.add(VMovB32(dst=vgpr(dataVgpr), src=0, comment="atomic_dec wrap value = 0 (forces slot to 0)"))
            module.add(SMovExec(dst=sgpr(execSgpr, numExecSgpr), src=EXEC(), comment="save EXEC"))
            module.add(SMovExec(dst=EXEC(), src=1, comment="only lane 0 active"))

            atomicFlat = FLATModifiers(scope=CacheScope.SCOPE_DEV) \
                if writer.states.archCaps["DefaultScopeIsCULocal"] else None
            innerSpinLabel = Label(writer.labels.getNameInc("last_gsu_wg_busy_waiting_inner"), "")
            module.add(innerSpinLabel)
            if writer.states.archCaps["RequiresXCntForVolatileVMEM"]:
                module.add(SWaitXCnt(xcnt=0, comment="drain in-flight VMEM before flat atomic"))
            module.add(FlatAtomicDecU32(dst=vgpr(dstVgpr), addr=vgpr(addrVgpr, 2), data=vgpr(dataVgpr), \
                                        modifier=atomicFlat, \
                                        comment="atomic read+reset synchronizer (wrap=0 -> slot:=0)"))
            module.add(SWaitCnt(vlcnt=0, comment="wait for atomic return"))
            module.add(VReadfirstlaneB32(dst=sgpr(tmpS01), src=vgpr(dstVgpr), comment="read old synchronizer value"))
            module.add(SCmpEQU32(src0=sgpr(tmpS01), src1=1, comment="last GSU WG? (producer signalled)"))
            module.add(SCBranchSCC0(labelName=innerSpinLabel.getLabelName(), comment="branch if false (retry)"))

            module.add(SMovExec(dst=EXEC(), src=sgpr(execSgpr, numExecSgpr), comment="restore EXEC"))
            writer.vgprPool.checkIn(addrVgpr)
            writer.vgprPool.checkIn(dataVgpr)
            writer.vgprPool.checkIn(dstVgpr)
            writer.sgprPool.checkIn(execSgpr)
        module.add(SBranch(labelName=reductionBodyLabel.getLabelName(), comment=""))

        return module

    def partialWriteBatch(self, writer, kernel, ss, batchIdx, beta, edge, gwvw, batchElements, tmpVgpr, batchElementSgprs, tmpSgpr, codeAccVgprRead, \
                          lastGsuWgBusyWaitingLabel, reductionBodyLabel, partialWriteLabel):
        module = Module("GSU Common partialWriteBatch")

        # allow expanding vgpr pool for OptNLL
        # preventOverflow = True #(not isOptNLL)
        # ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, isOptNLL=False, factorDim=0, isWorkspace=True)
        ss.setupStoreElementsForBatchWihoutVgprCheckOut(kernel, gwvw, batchElements, batchElementSgprs, isOptNLL=True, factorDim=0, isWorkspace=True)

        if batchIdx == 0 and kernel["MbskPrefetchMethod"] == 0:
            module.add(lastGsuWgBusyWaitingLabel)
            module.add(self.lastGsuWgBusyWaiting(writer, kernel, ss, tmpSgpr, tmpVgpr, lastGsuWgBusyWaitingLabel, reductionBodyLabel))
            module.add(partialWriteLabel)

        module.addComment0("optSingleColVgpr=%u optSharedColVgpr=%u optSGPRUsage=%s optSrdIncForRow=%u" % \
            (ss.optSingleColVgpr, ss.optSharedColVgpr, ss.optSGPRUsage, ss.optSrdIncForRow))

        if kernel["StoreSyncOpt"]:
            module.add(SSleep(kernel["StoreSyncOpt"] - 1, "optimization: sync and wait"))
            module.add(SBarrier())

        # comment tt1, tt0, vc1, vc0
        # tt = thread tile, vc=vector component
        commentStr = "Partial Write%s%s Batch #%u (d1,d0,vc1,vc0) =\n   " \
            % (" Beta" if beta else "", " Edge" if edge else "", batchIdx)
        for elementIdx, element in enumerate(batchElements):
            commentStr += "(%u,%u,%u,%u:vw%u)" % (element[0], element[1], element[2], element[3], gwvw)
            if elementIdx < len(batchElements)-1:
                commentStr += "; "
        module.addComment2(commentStr)

        storeOffsetSgpr = tmpSgpr.idx
        loadOffsetSgpr = tmpSgpr.idx + 1
        storeOffsetSgprRes = ContinuousRegister(storeOffsetSgpr, 1)

        ########################################
        # On input, coord0 and coord1 are VGPRs computed in the pre-batch code, based
        # on the thread and tid number.    These are ELEMENT offsets from start of tensor C
        # for the top-left corner this thread will write.    These are not changed
        # across all the store loop iters.
        if writer.db["ConservativeWaitCnt"] & 0x10:
            module.add(SBarrier(comment="debug"))
            module.add(SWaitCnt(vlcnt=0, vscnt=0, comment="ConservativeWaitCnt"))
            module.add(SBarrier(comment="debug"))

        if not edge and writer.db["ForceEdgeStores"]>=2:
            module.add(writer.getBomb()) # should not get here
        if edge and writer.db["AssertNoEdge"]:
            module.add(writer.getBomb()) # should not get here

        if kernel["BufferStore"] and edge:
            bufferOOB = tmpVgpr.idx + tmpVgpr.size - 1
            module.add(VMovB32(dst=vgpr(bufferOOB), src="BufferOOB"))
        else:
            bufferOOB = None

        if beta and kernel["StoreSyncOpt"]:
            module.add(SSleep(kernel["StoreSyncOpt"] - 1, "optimization: sync and wait"))
            module.add(SBarrier())

        ########################################
        # accvgpr read
        module.addComment("accvgpr read")
        if batchIdx > 0:
            module.add(SWaitCnt(vlcnt=0, vscnt=0, comment="Wait previous batch write over"))
        if codeAccVgprRead is not None and kernel["LocalSplitU"] == 1:
            regsPerScalar = writer.states.bpeCinternal // writer.states.bpr # register per scalar
            # loop over store instructions within one batch
            for elementIdx in range(0, len(batchElements)):
                # loop over scalars within one store instruction
                for vi in range(0, gwvw):
                    # loop over registers within one scalar
                    for rIdx in range(0, regsPerScalar):
                        codeAccVgprReadInst = codeAccVgprRead.popFirstItem() # v_accvgpr_read_b32
                        codeAccVgprReadInst.dst = vgpr(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx)
                        module.add(codeAccVgprReadInst)
        elif kernel["LocalSplitU"] > 1:
            # read from LSU VGPRs
            regsPerScalar = writer.states.bpeCinternal // writer.states.bpr # register per scalar            
            if kernel["MIArchVgpr"]:
                tmpStartVgprValuC = writer.states.c.startVgprValu
                writer.states.c.startVgprValu = 0
                module.add(RegSet("v", "vgprValuC", 0))
            if ss.lsuStartVgprOffset >= 0:
                for elementIdx in range(0, len(batchElements)):
                    for vi in range(0, gwvw):
                        for rIdx in range(0, regsPerScalar):
                            codeAccVgprReadInst = codeAccVgprRead.popFirstItem() # v_mov_b32
                            codeAccVgprReadInst.dst = vgpr(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx)
                            module.add(codeAccVgprReadInst)
            ss.lsuStartVgprOffset += len(batchElements) * gwvw * regsPerScalar

            if kernel["MIArchVgpr"]:
                writer.states.c.startVgprValu = tmpStartVgprValuC
                module.add(RegSet("v", "vgprValuC", tmpStartVgprValuC))
            else:
                module.add(SNop(1, "2 wait states required before reading vgpr"))
        module.addSpaceLine()

        ########################################
        # Write to workspace
        module.addComment("write to workspace")
        if kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
            storeCodeGSUSK = Module("GroupLoadStore")
            storeWidth = kernel["StoreVectorWidth"]
            bps = kernel["StoreVectorWidth"] * writer.states.bpeCinternal
            rpv = bps / writer.states.bpr
            isGlc = True
            isSlc = True
            isNT  = False #bool(kernel["NonTemporalD"] & 0x4)
            for elementIdx in range(0, len(batchElements)):
                addrCalc = ss.elementAddr[elementIdx]
                data = ss.elementData[elementIdx]
                addr0 = vgpr(addrCalc.addrDVgpr)
                addr1 = sgpr("SrdD", 4)
                globalOffset = 0

                if batchIdx == 0 and elementIdx == 0:
                    addrDVgpr = addrCalc.addrDVgpr
                    storeCodeGSUSK.add(vectorStaticMultiply(vgpr(addrDVgpr), vgpr("Serial"), storeWidth * writer.states.bpeCinternal, storeOffsetSgprRes))
                    storeCodeGSUSK.add(SMovB32(dst=sgpr(storeOffsetSgpr), src=0, comment="Init sgpr offset for interleaved wave store"))
                else:
                    # Use "NumThreads" instead of "MIWaveGroup" because LSU will not show in "MIWaveGroup"
                    increment = kernel["NumThreads"] * storeWidth * writer.states.bpeCinternal
                    storeCodeGSUSK.add(SAddU32(dst=sgpr(storeOffsetSgpr), src0=sgpr(storeOffsetSgpr), src1=increment, comment="Increase sgpr offset for store"))

                sumIdx = ss.elementSumIdx[elementIdx]
                if not kernel["StoreRemapVectorWidth"]:
                    # Only GSU>1 MBSK write to workspace (GSU1 MBSK will write to output buffer)
                    # so we need wsOffset to coalesced store to workspace buffer
                    wsOffset = sgpr(storeOffsetSgpr)
                    wsScope = CacheScope.SCOPE_DEV \
                        if writer.states.archCaps["DefaultScopeIsCULocal"] else CacheScope.SCOPE_NONE
                    storeCodeGSUSK.add(writer.chooseGlobalWrite(True, bps, sumIdx, rpv, \
                        addr0, addr1, globalOffset, soffset=wsOffset, \
                        glc=isGlc, slc=isSlc, nt=isNT, hi16=0, scope=wsScope, comment="store WS"))
                else:
                    rpe = writer.states.bpeCinternal // writer.states.bpr
                    storeCodeGSUSK.add(writer.storeRemapAddLocalWrite(kernel, ss, addrCalc, sumIdx*rpe))
                    # Column Block Shape has been written to LDS
                    # Now read back and write out to global memory
            module.add(storeCodeGSUSK)

        # return registers to pool:
        lastDataD = -1
        for elementIdx in range(0, len(batchElements)):
            if not ss.sharedColDVgprs:
                addrCalc: AddrCalculation = ss.elementAddres[elementIdx]
                addrDVgpr = addrCalc.addrDVgpr
                addrGSUSyncVgprs    = addrCalc.addrGSUSyncVgprs
                addrCVgpr = addrCalc.addrCVgpr
                writer.vgprPool.checkIn(addrDVgpr)
                if addrCVgpr != addrDVgpr:
                    writer.vgprPool.checkIn(addrCVgpr)
                if addrGSUSyncVgprs != None:
                    writer.parentWriter.vgprPool.checkIn(addrGSUSyncVgprs)

            data = ss.elementData[elementIdx]
            if data != 0:
                if data != lastDataD:
                    writer.vgprPool.checkIn(data)
                lastDataD = data

        ss.firstBatch = False
        ss.checkInTempVgprC()

        return module

    def reductionBatch(self, writer, kernel, ss, batchIdx, alpha, beta, edge, atomic, gwvw, batchElements, tmpVgpr, tmpVgprDynamic, \
                       batchElementSgprs, tmpSgpr, codeAccVgprRead, codeAccVgprWrite, reductionBodyLabel, endLabel):
        module = Module("GSU Common reductionBatch")

        module.addComment0("optSingleColVgpr=%u optSharedColVgpr=%u optSGPRUsage=%s optSrdIncForRow=%u" % \
            (ss.optSingleColVgpr, ss.optSharedColVgpr, ss.optSGPRUsage, ss.optSrdIncForRow))

        if kernel["StoreSyncOpt"]:
            module.add(SSleep(kernel["StoreSyncOpt"] - 1, "optimization: sync and wait"))
            module.add(SBarrier())

        # comment tt1, tt0, vc1, vc0
        # tt = thread tile, vc=vector component
        commentStr = "Reduction%s%s Batch #%u (d1,d0,vc1,vc0) =\n   " \
            % (" Beta" if beta else "", " Edge" if edge else "", batchIdx)
        for elementIdx, element in enumerate(batchElements):
            commentStr += "(%u,%u,%u,%u:vw%u)" % (element[0], element[1], element[2], element[3], gwvw)
            if elementIdx < len(batchElements)-1:
                commentStr += "; "
        module.addComment2(commentStr)

        # allow expanding vgpr pool for OptNLL
        # preventOverflow = True #(not isOptNLL)
        # ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, isOptNLL=False, factorDim=0, isWorkspace=True)
        ss.setupStoreElementsForBatchWihoutVgprCheckOut(kernel, gwvw, batchElements, batchElementSgprs, isOptNLL=True, factorDim=0, isWorkspace=True)

        ########################################
        # On input, coord0 and coord1 are VGPRs computed in the pre-batch code, based
        # on the thread and tid number.    These are ELEMENT offsets from start of tensor C
        # for the top-left corner this thread will write.    These are not changed
        # across all the store loop iters.
        if writer.db["ConservativeWaitCnt"] & 0x10:
            module.add(SBarrier(comment="debug"))
            module.add(SWaitCnt(vlcnt=0, vscnt=0, comment="ConservativeWaitCnt"))
            module.add(SBarrier(comment="debug"))

        if not edge and writer.db["ForceEdgeStores"]>=2:
            module.add(writer.getBomb()) # should not get here
        if edge and writer.db["AssertNoEdge"]:
            module.add(writer.getBomb()) # should not get here

        if kernel["BufferStore"] and edge:
            bufferOOB = tmpVgpr.idx + tmpVgpr.size - 1
            module.add(VMovB32(dst=vgpr(bufferOOB), src="BufferOOB"))
        else:
            bufferOOB = None

        if beta and kernel["StoreSyncOpt"]:
            module.add(SSleep(kernel["StoreSyncOpt"] - 1, "optimization: sync and wait"))
            module.add(SBarrier())

        # do we really need this?
        sumIdxGSUSYNC = ss.elementSumIdx[len(batchElements)-1]
        addrCalc = ss.elementAddr[len(batchElements)-1]
        accvgprWriteLabel = Label(writer.labels.getNameInc("accvgpr_write"), comment="")
        
        if (kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel'):
            module.add(self.GSUSynccodegenOpt(kernel, writer, ss, batchIdx, tmpSgpr, tmpVgpr, tmpVgprDynamic, gwvw, batchElements,\
                                              endLabel, sumIdxGSUSYNC, addrCalc.addrDVgpr, reductionBodyLabel))
            module.addComment("synchronizer store end")
            if kernel["MbskPrefetchMethod"] == 0:
                module.add(SAndB32(dst=sgpr(tmpSgpr.idx), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
                module.add(SCmpGtI32(src0=sgpr(tmpSgpr.idx), src1=self.gsuThreshold, comment="GSU > %u ?" % self.gsuThreshold))
                module.add(SCBranchSCC1(labelName=accvgprWriteLabel.getLabelName(), comment="branch if true"))
                module.addComment("GSU <= %u, do accvgpr_read for the last gsu wg" % self.gsuThreshold)
                module.add(self.lastGsuWgReduction(kernel, writer, ss, batchIdx, tmpVgpr, tmpVgprDynamic, gwvw, batchElements, \
                                               codeAccVgprRead, addrCalc.globalOffset, addrCalc.addrDVgpr))

        ########################################
        # accvgpr write
        module.add(accvgprWriteLabel)
        module.addComment("accvgpr write")
        if codeAccVgprWrite is not None:
            regsPerScalar = writer.states.bpeCinternal // writer.states.bpr # register per scalar
            if kernel["MIArchVgpr"] and kernel["LocalSplitU"] > 1:
                tmpStartVgprValuC = writer.states.c.startVgprValu
                writer.states.c.startVgprValu = 0
                module.add(RegSet("v", "vgprValuC", 0))
            # loop over store instructions within one batch
            for elementIdx in range(0, len(batchElements)):
                # loop over scalars within one store instruction
                for vi in range(0, gwvw):
                    # loop over registers within one scalar
                    for rIdx in range(0, regsPerScalar):
                        codeAccVgprWriteInst = codeAccVgprWrite.popFirstItem()
                        codeAccVgprWriteInst.srcs[0] = vgpr(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx)
                        module.add(codeAccVgprWriteInst)

            # TODO: need 3 wait states if read accvgpr after write accvgpr?
            # if not kernel["MIArchVgpr"]:
            #     module.add(SNop(1, "2 wait states required before reading vgpr"))
            if kernel["MIArchVgpr"] and kernel["LocalSplitU"] > 1:
                writer.states.c.startVgprValu = tmpStartVgprValuC
                module.add(RegSet("v", "vgprValuC", tmpStartVgprValuC))

        if edge and (not kernel["BufferStore"]): # atomic or
            # subsequent batch must start with full exec mask
            # BufferStore doesn't need exec since it used buffer range checking when
            # possible
            module.add(self.getEdgeMovInstType()(EXEC(), -1, "full mask -> exec"))

        if writer.db["ConservativeWaitCnt"] & 0x40:
            module.add(SBarrier(comment="debug"))
            module.add(SWaitCnt(vlcnt=0, vscnt=0, comment="ConservativeWaitCnt"))
            module.add(SBarrier(comment="debug"))

        # return registers to pool:
        lastDataD = -1
        for elementIdx in range(0, len(batchElements)):
            if not ss.sharedColDVgprs:
                addrCalc: AddrCalculation = ss.elementAddres[elementIdx]
                addrDVgpr = addrCalc.addrDVgpr
                addrGSUSyncVgprs    = addrCalc.addrGSUSyncVgprs
                addrCVgpr = addrCalc.addrCVgpr
                writer.vgprPool.checkIn(addrDVgpr)
                if addrCVgpr != addrDVgpr:
                    writer.vgprPool.checkIn(addrCVgpr)
                if addrGSUSyncVgprs != None:
                    writer.parentWriter.vgprPool.checkIn(addrGSUSyncVgprs)

            data = ss.elementData[elementIdx]
            if data != 0:
                if data != lastDataD:
                    writer.vgprPool.checkIn(data)
                lastDataD = data

        ss.firstBatch = False
        ss.checkInTempVgprC()

        return module

    def computeWorkspaceSrd(self, writer, kernel, tmpSgpr):
        module = Module("GSU Common computeWorkspaceSrd")

        tmpS01 = tmpSgpr.idx
        tmpS02 = tmpSgpr.idx + 1

        # WS address calculation
        module.addComment("calculate the starting WG index of GSU WGs")
        module.add(SMulI32(dst=sgpr(tmpS01), src0=sgpr("NumWorkGroups0"), src1=sgpr("WorkGroup1"), comment="NumWorkGroups0*wg1"))
        module.add(SAndB32(dst=sgpr(tmpS02), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
        module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1=sgpr("WorkGroup0"), comment="NumWorkGroups0*wg1+wg0"))
        module.add(SMulI32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1=sgpr(tmpS02), comment="(NumWorkGroups0*wg1+wg0)*GSU"))
        module.add(SMulI32(dst=sgpr("GSUStartWGIdx"), src0=sgpr("NumWorkGroups0"), src1=sgpr("NumWorkGroups1"), comment="NumWgPerBatch"))
        module.add(SMulI32(dst=sgpr("GSUStartWGIdx"), src0=sgpr("GSUStartWGIdx"), src1=sgpr("WorkGroup2"), comment="NumWgPerBatch"))
        module.add(SMulI32(dst=sgpr("GSUStartWGIdx"), src0=sgpr("GSUStartWGIdx"), src1=sgpr(tmpS02), comment="NumWgPerBatch"))
        module.add(SAddU32(dst=sgpr("GSUStartWGIdx"), src0=sgpr("GSUStartWGIdx"), src1=sgpr(tmpS01), comment="starting WG index of each GSU WGs"))
        module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr("GSUStartWGIdx"), src1=sgpr("GSUSumIdx"), comment="(NumWorkGroups0*wg1+wg0)*GSU+NumWgPerBatch+GSUSumIdx"))

        assert kernel["BufferStore"]
        module.addComment("add offset to the base address of workspace buffer")
        module.add(SMulHIU32(dst=sgpr(tmpS02), src0=sgpr(tmpS01), src1="MTOffset", comment="(MT0*MT1*bpeC)*WGIdx"))
        module.add(SMulI32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1="MTOffset", comment="(MT0*MT1*bpeC)*WGIdx"))
        module.add(SAddU32(dst=sgpr("SrdD+0"), src0=sgpr("AddressD+0"), src1=sgpr(tmpS01), comment="add lo to SRD"))
        module.add(SAddCU32(dst=sgpr("SrdD+1"), src0=sgpr("AddressD+1"), src1=sgpr(tmpS02), comment="add hi to SRD"))
        module.addSpaceLine()

        return module

    def _generateAtomicDec(self, writer, kernel, dst, base):
        module = Module("atomicDec")
        if writer.states.asmCaps["HasSAtomic"]:
            module.add(SAtomicDec(dst=dst, base=base, smem=SMEMModifiers(glc=True)))
        else:
            # For architectures without scalar atomic (e.g., gfx1250), use flat atomic with EXEC mask
            addrVgpr = writer.vgprPool.checkOutAligned(2, 2, "addrVgpr")
            dataVgpr = writer.vgprPool.checkOut(1)
            dstVgpr = writer.vgprPool.checkOut(1)
            # EXEC is wave-width: 1 sgpr on wave32, 2 sgprs on wave64
            numSgpr = 1 if kernel["WavefrontSize"] == 32 else 2
            SMovExec = SMovB32 if numSgpr == 1 else SMovB64
            tmpSgpr = writer.sgprPool.checkOutAligned(numSgpr, numSgpr, preventOverflow=False)

            # dst contains GSU-1 (the wrap value), move it to VGPR
            module.add(VMovB32(dst=vgpr(dataVgpr), src=dst, comment="wrap value = GSU-1"))
            module.add(VMovB64(dst=vgpr(addrVgpr, 2), src=base, comment="atomic address"))

            # Save EXEC and set only lane 0 active
            module.add(SMovExec(dst=sgpr(tmpSgpr, numSgpr), src=EXEC(), comment="save EXEC"))
            module.add(SMovExec(dst=EXEC(), src=1, comment="only lane 0 active"))
            # Arches that mark RequiresXCntForVolatileVMEM (e.g. gfx1250) need
            # an explicit XNACK-replay drain before a volatile/atomic VMEM op.
            if writer.states.archCaps["RequiresXCntForVolatileVMEM"]:
                module.add(SWaitXCnt(xcnt=0, comment="drain in-flight VMEM before flat atomic"))
            atomicFlat = FLATModifiers(scope=CacheScope.SCOPE_DEV) \
                if writer.states.archCaps["DefaultScopeIsCULocal"] else None
            module.add(FlatAtomicDecU32(dst=vgpr(dstVgpr), addr=vgpr(addrVgpr, 2), data=vgpr(dataVgpr),
                                         modifier=atomicFlat))
            module.add(SMovExec(dst=EXEC(), src=sgpr(tmpSgpr, numSgpr), comment="restore EXEC"))
            module.add(SWaitCnt(vlcnt=0, comment="wait for atomic return"))

            # Read return value to destination SGPR
            module.add(VReadfirstlaneB32(dst=dst, src=vgpr(dstVgpr), comment="read atomic return value"))

            writer.vgprPool.checkIn(addrVgpr)
            writer.vgprPool.checkIn(dataVgpr)
            writer.vgprPool.checkIn(dstVgpr)
            writer.sgprPool.checkIn(tmpSgpr)
        return module

    def GSUSynccodegenOpt(self, kernel, writer, ss, batchIdx, tmpSgpr, tmpVgpr, tmpVgprDynamic, gwvw, batchElements, \
                          labelend, vgprstart, vgproffset, reductionBodyLabel):
        module = Module("GSUSYNC")

        reductionSkipLabel = Label(writer.labels.getNameInc("reduction_skip"), comment="")
        reductionAllGsuWgLabel = Label(writer.labels.getNameInc("reduction_all_gsu_wg"), comment="")

        tmpS01 = writer.sgprPool.checkOut(1, tag="GSUSynccodegenOpt_tmpS01", preventOverflow=False)
        tmpS02 = writer.sgprPool.checkOut(1, tag="GSUSynccodegenOpt_tmpS02", preventOverflow=False)
        tmpS05 = writer.sgprPool.checkOutAligned(2,2, tag="GSUSynccodegenOpt_tmpS05", preventOverflow=False)
        tmpS06 = writer.sgprPool.checkOutAligned(4,4, tag="GSUSynccodegenOpt_tmpS06", preventOverflow=False)
        tmpS06Res = ContinuousRegister(idx=tmpS06, size=4)

        bufferOOB = tmpVgpr.idx + tmpVgpr.size - 1
        storeOffsetSgpr = tmpSgpr.idx
        loadOffsetSgpr = tmpSgpr.idx + 1
        storeOffsetSgprRes = ContinuousRegister(storeOffsetSgpr, 1)
        
        addr1 = sgpr(tmpS06, 4)
        addr0 = vgpr(vgproffset)
        bps = kernel["ProblemType"]["ComputeDataType"].numBytes() * gwvw
        storeWidth = kernel["StoreVectorWidth"]
        increment = kernel["NumThreads"] * storeWidth * writer.states.bpeCinternal
        # On gfx1250 the reducer reads workspace data that other GSU WGs wrote
        # from a different CU; SCOPE_DEV forces the load to bypass L1 and match
        # the SCOPE_DEV partial-write store + flat_atomic_dec_u32 ordering.
        gsuReadScope = CacheScope.SCOPE_DEV \
            if writer.states.archCaps["DefaultScopeIsCULocal"] else CacheScope.SCOPE_NONE

        if batchIdx == 0:
            # wait for write to ws and do atomic dec to synchronizer
            module.addComment("check done start")
            module.add(SWaitCnt(waitAll=True, comment="wait store done before synchronizer start load and add"))
            module.add(SAndB32(dst=sgpr(tmpS02), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
            module.add(SSubU32(dst=sgpr(tmpS02), src0=sgpr(tmpS02), src1=1, comment=""))
            module.add(self._generateAtomicDec(writer, kernel, dst=sgpr(tmpS02), base=sgpr("SrdSync", 2)))
            if kernel["MbskPrefetchMethod"] == 0:
                module.add(SAndB32(dst=sgpr(tmpS01), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
                module.add(SCmpGtI32(src0=sgpr(tmpS01), src1=self.gsuThreshold, comment="GSU > %u ?" % self.gsuThreshold))
                module.add(SCBranchSCC0(labelName=reductionSkipLabel.getLabelName(), comment="branch if false"))
                module.addComment("GSU > %u, atomic_dec selects the gsu wg to do the reduction" % self.gsuThreshold)

            # wait for synchronizer and check whether to branch or not
            module.addComment("check synchronizer done")
            module.add(SWaitCnt(kmcnt=0, comment="Wait for synchronizer"))
            module.add(SCmpEQU32(src0=sgpr(tmpS02), src1=hex(1), comment=""))
            module.add(SCBranchSCC1(labelName=reductionBodyLabel.getLabelName(), comment="branch if true"))
            if kernel["MbskPrefetchMethod"] == 0:
                module.add(reductionSkipLabel)
            module.add(SEndpgm())
            module.addComment("check done end")
            module.add(reductionBodyLabel)

            # calculate the address for read from ws
            addrCalc = ss.elementAddr[0]
            addrDVgpr = addrCalc.addrDVgpr
            module.add(SMovB32(sgpr(loadOffsetSgpr), 0, "Init sgpr offset for interleaved wave load"))
            module.add(vectorStaticMultiply(vgpr(addrDVgpr), vgpr("Serial"), storeWidth * writer.states.bpeCinternal, storeOffsetSgprRes))
            module.add(VMovB32(dst=vgpr(bufferOOB), src="BufferOOB"))
            module.addComment("synchronizer sum offset is equal to MTOffset (=MT0*MT1*bpeC)")
            module.add(SMulHIU32(dst=sgpr(tmpS06+1), src0="MTOffset", src1=sgpr("GSUStartWGIdx"), comment="(MT0*MT1*bpeC)*WGIdx"))
            module.add(SMulI32(dst=sgpr(tmpS06), src0="MTOffset", src1=sgpr("GSUStartWGIdx"), comment="(MT0*MT1*bpeC)*WGIdx"))
            module.add(SAddU32(dst=sgpr(tmpS06+0), src0=sgpr("AddressD+0"), src1=sgpr(tmpS06), comment="add lo to SRD"))
            module.add(SAddCU32(dst=sgpr(tmpS06+1), src0=sgpr("AddressD+1"), src1=sgpr(tmpS06+1), comment="add hi to SRD"))
            module.add(SMovB64(sgpr(tmpS06+2, 2), sgpr("SrdD+2", 2), ""))

            # for GSU < self.gsuThreshold, GSU-1 is passed to the reduction body
            module.add(SAndB32(dst=sgpr(tmpS02), src0=sgpr("GSU"), src1=writer.gsuMaskHex(kernel), comment="Restore GSU"))
            if kernel["MbskPrefetchMethod"] == 0:
                module.add(SCmpGtI32(src0=sgpr(tmpS02), src1=self.gsuThreshold, comment="GSU > %u ?" % self.gsuThreshold))
                module.add(SCBranchSCC1(labelName=reductionAllGsuWgLabel.getLabelName(), comment="branch if true"))
                module.addComment("GSU <= %u, so we minus 1 from GSUSync at the beginning" % self.gsuThreshold)
                module.add(SSubI32(dst=sgpr(tmpS02), src0=sgpr(tmpS02), src1=1, comment="Use GSU-1"))
                module.add(reductionAllGsuWgLabel)

        if not kernel["MbskPrefetchMethod"]:
            for elementIdx in range(0, len(batchElements)):
                addrCalc: AddrCalculation = ss.elementAddr[elementIdx]
                addr0 = vgpr(addrCalc.addrDVgpr)

                # pre-load
                SyncloadedData = 0
                SynchronizerAddSkiplabelString = "Synchronizer_read_add_skip"
                SynchronizerAddSkipComment = "Synchronizer read add skip"
                SynchronizerAddSkiplabel = Label(writer.labels.getNameInc(SynchronizerAddSkiplabelString), SynchronizerAddSkipComment)

                GSUtotal = writer.getMBSKGSUTotal(kernel)
                SynchronizerAddEndlabel = [""] * GSUtotal

                for idx in range(0, GSUtotal):
                    SynchronizerAddEndlabelString = "Synchronizer_read_add_end_"+str(idx+1)
                    SynchronizerAddEndComment = "Synchronizer read add end_"+str(idx+1)
                    SynchronizerAddEndlabel[idx] = Label(writer.labels.getNameInc(SynchronizerAddEndlabelString), SynchronizerAddEndComment)

                #####################################load buffer#####################################
                module.addComment("buffer load start")
                for times in range(elementIdx, elementIdx+1):
                    if batchIdx != 0 or elementIdx != 0:
                        module.add(SAddU32(dst=sgpr(loadOffsetSgpr), src0=sgpr(loadOffsetSgpr), src1=increment, comment="Increase sgpr offset for load"))
                        module.add(SMulHIU32(dst=sgpr(tmpS06+1), src0="MTOffset", src1=sgpr("GSUStartWGIdx"), comment="(MT0*MT1*bpeC)*WGIdx"))
                        module.add(SMulI32(dst=sgpr(tmpS06), src0="MTOffset", src1=sgpr("GSUStartWGIdx"), comment="(MT0*MT1*bpeC)*WGIdx"))
                        module.add(SAddU32(dst=sgpr(tmpS06+0), src0=sgpr("AddressD+0"), src1=sgpr(tmpS06), comment="add lo to SRD"))
                        module.add(SAddCU32(dst=sgpr(tmpS06+1), src0=sgpr("AddressD+1"), src1=sgpr(tmpS06+1), comment="add hi to SRD"))

                vgprstart = ss.elementSumIdx[elementIdx]
                dataType  = kernel["ProblemType"]["DestDataType"]
                if dataType.isDouble() or dataType.isSingleComplex():
                    vgprstart = vgprstart*2
                module.add(writer.chooseGlobalRead(True, bps, vgprstart, \
                                addr0, addr1, soffset=sgpr(loadOffsetSgpr), offset=0, glc=True, slc=True, scope=gsuReadScope,\
                                comment="load GSU WG %d element %d" % (SyncloadedData, elementIdx)))
                SyncloadedData += 1

                # Init GSUSync for different batch
                module.add(SMovB32(dst=sgpr("GSUSync"), src=sgpr(tmpS02), comment="Init GSUSync to GSU for batch %u" % batchIdx))

                SynchronizerlabelString = "Synchronizer_read_add"
                SynchronizerComment = "Synchronizer read add"
                Synchronizerlabel = Label(writer.labels.getNameInc(SynchronizerlabelString), SynchronizerComment)
                tmpVAdd = tmpVgprDynamic.idx
                GSUMvgpr = tmpVgpr.idx
                GSUP1 = GSUtotal-1

                for i in range(0, GSUP1):
                    module.add(SSubI32(dst=sgpr("GSUSync"), src0=sgpr("GSUSync"), src1=1, comment="%u" % i))

                    module.add(SAddU32(dst=sgpr(tmpS06+0), src0=sgpr(tmpS06+0), src1="MTOffset", comment="" ))
                    module.add(SAddCU32(dst=sgpr(tmpS06+1), src0=sgpr(tmpS06+1), src1="MTOffsetH32", comment="" ))

                    module.add(SCmpEQI32(src0=sgpr("GSUSync"), src1=0, comment=""))#GSUSync+GSUP1==GSU
                    module.add(SCBranchSCC1(labelName=SynchronizerAddEndlabel[i].getLabelName(), comment="SyncAddbranchhere"))

                    if(kernel["ProblemType"]["DestDataType"].numRegisters() > 1):
                        module.add(writer.chooseGlobalRead(True, bps, tmpVAdd+gwvw*kernel["ProblemType"]["DestDataType"].numRegisters()*i, \
                                    addr0, addr1, soffset=0, offset=0, glc=True, slc=True, scope=gsuReadScope, \
                                    comment="load GSU WG %d element %d" % (SyncloadedData, elementIdx)))
                    else:
                        module.add(writer.chooseGlobalRead(True, bps, tmpVAdd+gwvw*i, \
                                    addr0, addr1, soffset=sgpr(loadOffsetSgpr), offset=0, glc=True, slc=True, scope=gsuReadScope, \
                                    comment="load GSU WG %d element %d" % (SyncloadedData, elementIdx)))
                    SyncloadedData += 1

                module.addComment("buffer load end")

                #####################################> GSUtotal reduction start#####################################
                module.addComment("buffer add start")

                module.add(Synchronizerlabel)
                vlcnt = SyncloadedData - 1

                for i in range(0, GSUP1):
                    vlcnt = vlcnt - 1 if vlcnt > 0 else 0
                    module.add(SWaitCnt(vlcnt=vlcnt, comment="(wait for buffer ready)"))
                    if kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].isInt32():
                        for j in range(0, int(gwvw)):
                            module.add(VAddI32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(tmpVAdd+0+gwvw*i+j), \
                                    comment="buffer add"))
                    else:
                        if ((gwvw % 2) == 1):
                            for j in range(0, int(gwvw)):
                                module.add(VAddF32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(tmpVAdd+0+gwvw*i+j), \
                                        comment="buffer add"))
                        else:
                            for j in range(0, int(gwvw/2)):
                                module.add(VAddPKF32(dst=vgpr(vgprstart+j*2, 2), src0=vgpr(vgprstart+j*2, 2), \
                                            src1=vgpr(tmpVAdd+0+gwvw*i+j*2, 2), comment="buffer pk"))

                    module.add(SSubI32(dst=sgpr("GSUSync"), src0=sgpr("GSUSync"), src1=1, comment="%u" % i))
                    module.add(SCmpLeI32(src0=sgpr("GSUSync"), src1=0-(GSUP1-1), comment=""))#GSUSync+GSUP1==GSU
                    module.add(SCBranchSCC1(labelName=SynchronizerAddSkiplabel.getLabelName(), comment="SyncAddbranch"))

                    module.add(SAddU32(dst=sgpr(tmpS06+0), src0=sgpr(tmpS06+0), src1="MTOffset", comment=""))
                    module.add(SAddCU32(dst=sgpr(tmpS06+1), src0=sgpr(tmpS06+1), src1="MTOffsetH32", comment=""))

                    numSgpr = 1 if kernel["WavefrontSize"] == 32 else 2
                    module.add(VCmpGEI32(dst=sgpr(tmpS05,numSgpr), src0=0, src1=sgpr("GSUSync"), comment=""))
                    module.add(VCndMaskB32(dst=vgpr(GSUMvgpr), src1=vgpr(bufferOOB), src0=addr0, src2=sgpr(tmpS05,numSgpr), comment="protect if OOB"))

                    if(kernel["ProblemType"]["DestDataType"].numRegisters() > 1):
                        module.add(writer.chooseGlobalRead(True, bps, tmpVAdd+gwvw*kernel["ProblemType"]["DestDataType"].numRegisters()*i, \
                                    vgpr(GSUMvgpr), addr1, soffset=0, offset=0, glc=True, slc=True, scope=gsuReadScope, \
                                    comment="prefetch GSU WG element %d" % elementIdx))
                    else:
                        module.add(writer.chooseGlobalRead(True, bps, tmpVAdd+gwvw*i, \
                                    vgpr(GSUMvgpr), addr1, soffset=sgpr(loadOffsetSgpr), offset=0, glc=True, slc=True, scope=gsuReadScope, \
                                    comment="prefetch GSU WG element %d" % elementIdx))
                    vlcnt += 1

                module.addComment("buffer add end")

                module.add(SCmpGtI32(src0=sgpr("GSUSync"), src1=hex(1-(GSUP1)), comment=""))
                module.add(SCBranchSCC1(labelName=Synchronizerlabel.getLabelName(), comment="Syncbranchhere"))

                #####################################< GSUtotal reduction start#####################################
                for k in range(GSUtotal-2, -1, -1):
                    module.addSpaceLine()
                    module.add(SynchronizerAddEndlabel[k])

                    vlcnt = k
                    for i in range(0, k):
                        vlcnt = vlcnt - 1 if vlcnt > 0 else 0
                        module.add(SWaitCnt(vlcnt=vlcnt, comment="(wait for buffer ready)"))
                        if kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].isInt32():
                            for j in range(0, int(gwvw)):
                                module.add(VAddI32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(tmpVAdd+0+gwvw*i+j), \
                                        comment="buffer add"))
                        else:
                            if ((gwvw % 2) == 1):
                                for j in range(0, int(gwvw)):
                                    module.add(VAddF32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(tmpVAdd+0+gwvw*i+j), \
                                                comment="buffer add"))
                            else:
                                for j in range(0, int(gwvw/2)):
                                    module.add(VAddPKF32(dst=vgpr(vgprstart+j*2, 2), src0=vgpr(vgprstart+j*2, 2), \
                                                    src1=vgpr(tmpVAdd+0+gwvw*i+j*2, 2), comment="buffer pk"))

                        if i == k-1:
                            module.add(SBranch(labelName=SynchronizerAddSkiplabel.getLabelName(), comment="SyncAddbranch"))

                module.add(SynchronizerAddSkiplabel)

                module.addComment("buffer add end2")
        else:
            tmpWSD = writer.sgprPool.checkOutAligned(4, 4, tag="GSUSynccodegenOpt_tmpWSD", preventOverflow=False)
            GSUtotal = writer.getMBSKGSUTotal(kernel)-1
            loadWidth = gwvw * int(max(1, kernel["ProblemType"]["DestDataType"].numRegisters()))
            unrolledWGs = GSUtotal // len(batchElements)
            tmpVidx = tmpVgprDynamic.idx
            tmpVAdd = [[0] * len(batchElements) for _ in range(unrolledWGs)]
            for i in range(0, unrolledWGs):
                for j in range(0, len(batchElements)):
                    tmpVAdd[i][j] = tmpVidx
                    tmpVidx += loadWidth

            SynchronizerAddEndlabel = [""] * (unrolledWGs+1)

            for uidx in range(0, unrolledWGs+1):
                SynchronizerAddEndlabelString = "Synchronizer_read_add_end_"+str(uidx+1)
                SynchronizerAddEndComment = "Synchronizer read add end_"+str(uidx+1)
                SynchronizerAddEndlabel[uidx] = Label(writer.labels.getNameInc(SynchronizerAddEndlabelString), SynchronizerAddEndComment)

            # set buffer load address for WG0
            module.add(SMulHIU32(dst=sgpr(tmpWSD+1), src0="MTOffset", src1=sgpr("GSUStartWGIdx"), comment="(MT0*MT1*bpeC)*WGIdx"))
            module.add(SMulI32(dst=sgpr(tmpWSD), src0="MTOffset", src1=sgpr("GSUStartWGIdx"), comment="(MT0*MT1*bpeC)*WGIdx"))
            module.add(SAddU32(dst=sgpr(tmpWSD), src0=sgpr("AddressD+0"), src1=sgpr(tmpWSD), comment="add lo to SRD"))
            module.add(SAddCU32(dst=sgpr(tmpWSD+1), src0=sgpr("AddressD+1"), src1=sgpr(tmpWSD+1), comment="add hi to SRD WSD"))
            module.add(SMovB64(sgpr(tmpWSD+2, 2), sgpr("SrdD+2", 2), ""))

            # set buffer load address for WG1
            module.add(SAddU32(dst=sgpr(tmpS06+0), src0=sgpr(tmpWSD+0), src1="MTOffset", comment="" ))
            module.add(SAddCU32(dst=sgpr(tmpS06+1), src0=sgpr(tmpWSD+1), src1="MTOffsetH32", comment="" ))
            module.add(SMovB64(sgpr(tmpS06+2, 2), sgpr("SrdD+2", 2), ""))

            accumulationStartlabel = Label(writer.labels.getNameInc("Accumulation_Start"), "Accumulation Start")
            accumulationEndlabel   = Label(writer.labels.getNameInc("Accumulation_End"), "Accumulation End")
            # Init GSUSync for different batch
            module.add(SMovB32(dst=sgpr("GSUSync"), src=sgpr(tmpS02), comment="Init GSUSync to GSU for batch %u" % batchIdx))

            # pre-load
            SyncloadedData = 0

            # first 2 WGs: read same element first for earlier reduction
            for elementIdx in range(0, len(batchElements)):
                addrCalc: AddrCalculation = ss.elementAddr[elementIdx]
                addr0    = vgpr(addrCalc.addrDVgpr)
                if batchIdx != 0 or elementIdx != 0:
                    module.add(SAddU32(dst=sgpr(loadOffsetSgpr), src0=sgpr(loadOffsetSgpr), src1=increment, comment="Increase sgpr offset for load"))
                if elementIdx == 0:
                    module.add(SMovB32(dst=sgpr(tmpS01), src=sgpr(loadOffsetSgpr), comment="save first element offset"))
                for uidx in range(0, 2):
                    if uidx == 0:
                        data = ss.elementSumIdx[elementIdx]
                        tmpAddr1 = tmpWSD
                    else:
                        data = tmpVAdd[-1][elementIdx]
                        tmpAddr1 = tmpS06

                    module.add(writer.chooseGlobalRead(True, bps, data, \
                                    addr0, sgpr(tmpAddr1, 4), soffset=sgpr(loadOffsetSgpr), offset=0, glc=True, slc=True, scope=gsuReadScope,\
                                    comment="load GSU WG %d element %d " % (uidx, elementIdx)))

                    SyncloadedData += 1

            module.add(SSubI32(dst=sgpr("GSUSync"), src0=sgpr("GSUSync"), src1=2))
            module.add(SCmpLeI32(src0=sgpr("GSUSync"), src1=0, comment="GSUSync <= 0?"))
            module.add(SCBranchSCC1(labelName=SynchronizerAddEndlabel[1].getLabelName(), comment=""))
            module.add(SAddU32(dst=sgpr(tmpS06+0), src0=sgpr(tmpS06+0), src1="MTOffset", comment="" ))
            module.add(SAddCU32(dst=sgpr(tmpS06+1), src0=sgpr(tmpS06+1), src1="MTOffsetH32", comment="" ))

            # other WGs: read all elements together
            for uidx in range(2, unrolledWGs+1):
                module.add(SMovB32(dst=sgpr(loadOffsetSgpr), src=sgpr(tmpS01), comment="restore offset for element0"))
                for elementIdx in range(0, len(batchElements)):
                    addrCalc: AddrCalculation = ss.elementAddr[elementIdx]
                    addr0    = vgpr(addrCalc.addrDVgpr)
                    data = tmpVAdd[uidx-2][elementIdx]

                    if elementIdx != 0:
                        module.add(SAddU32(dst=sgpr(loadOffsetSgpr), src0=sgpr(loadOffsetSgpr), src1=increment, comment="Increase sgpr offset for load"))

                    module.add(writer.chooseGlobalRead(True, bps, data, \
                                    addr0, addr1, soffset=sgpr(loadOffsetSgpr), offset=0, glc=True, slc=True, scope=gsuReadScope,\
                                    comment="load GSU WG %d element %d " % (uidx, elementIdx)))
                    SyncloadedData += 1

                module.add(SSubI32(dst=sgpr("GSUSync"), src0=sgpr("GSUSync"), src1=1))
                module.add(SCmpLeI32(src0=sgpr("GSUSync"), src1=0, comment="GSUSync <= 0?"))
                module.add(SCBranchSCC1(labelName=SynchronizerAddEndlabel[uidx].getLabelName(), comment=""))
                module.add(SAddU32(dst=sgpr(tmpS06+0), src0=sgpr(tmpS06+0), src1="MTOffset", comment="" ))
                module.add(SAddCU32(dst=sgpr(tmpS06+1), src0=sgpr(tmpS06+1), src1="MTOffsetH32", comment="" ))

            module.addComment("buffer load end\n")

            ##################################### reduction start #####################################
            module.addComment("buffer add start")

            vlcnt = SyncloadedData

            # reduce first 2 WGs
            module.add(SMovB32(dst=sgpr(loadOffsetSgpr), src=sgpr(tmpS01), comment="restore offset for element0"))
            for elementIdx in range(0, len(batchElements)):
                addrCalc: AddrCalculation = ss.elementAddr[elementIdx]
                addr0 = vgpr(addrCalc.addrDVgpr)
                data = tmpVAdd[-1][elementIdx]
                vgprstart   = ss.elementSumIdx[elementIdx]
                vlcnt       = vlcnt - 2 if vlcnt > 0 else 0

                module.add(SWaitCnt(vlcnt=vlcnt, comment="(wait for buffer ready)"))
                if kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].isInt32():
                    for j in range(0, int(gwvw)):
                        module.add(VAddI32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(data+j), \
                                comment="buffer add"))
                else:
                    if ((gwvw % 2) == 1):
                        for j in range(0, int(gwvw)):
                            module.add(VAddF32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(data+j), \
                                        comment="buffer add"))
                    else:
                        for j in range(0, int(gwvw/2)):
                            module.add(VAddPKF32(dst=vgpr(vgprstart+j*2, 2), src0=vgpr(vgprstart+j*2, 2), \
                                            src1=vgpr(data+j*2, 2), comment="buffer pk"))

                # prefetch
                if elementIdx != 0:
                    module.add(SAddU32(dst=sgpr(loadOffsetSgpr), src0=sgpr(loadOffsetSgpr), src1=increment, comment="Increase sgpr offset for load"))

                module.add(writer.chooseGlobalRead(True, bps, data, \
                                addr0, addr1, soffset=sgpr(loadOffsetSgpr), offset=0, glc=True, slc=True, scope=gsuReadScope,\
                                comment="prefetch element %d " % (elementIdx)))
                vlcnt += 1

            module.add(SSubI32(dst=sgpr("GSUSync"), src0=sgpr("GSUSync"), src1=1))
            module.add(SCmpLeI32(src0=sgpr("GSUSync"), src1=0-unrolledWGs, comment=""))
            module.add(SCBranchSCC1(labelName=accumulationEndlabel.getLabelName(), comment="Accumulation finished"))
            module.add(SAddU32(dst=sgpr(tmpS06+0), src0=sgpr(tmpS06+0), src1="MTOffset", comment="" ))
            module.add(SAddCU32(dst=sgpr(tmpS06+1), src0=sgpr(tmpS06+1), src1="MTOffsetH32", comment="" ))
            module.add(SCmpLeI32(src0=sgpr("GSUSync"), src1=0, comment="disable buffer load if GSUSync <= 0"))
            module.add(SCSelectB32(sgpr(tmpS06+2), 0, sgpr(tmpS06+2), ""))
            module.add(accumulationStartlabel)

            for uidx in range(0, unrolledWGs):
                module.add(SMovB32(dst=sgpr(loadOffsetSgpr), src=sgpr(tmpS01), comment="restore offset for element0"))
                for elementIdx in range(0, len(batchElements)):
                    addrCalc: AddrCalculation = ss.elementAddr[elementIdx]
                    addr0    = vgpr(addrCalc.addrDVgpr)
                    data     = tmpVAdd[uidx][elementIdx]
                    vgprstart   = ss.elementSumIdx[elementIdx]
                    vlcnt       = vlcnt - 1 if vlcnt > 0 else 0

                    module.add(SWaitCnt(vlcnt=vlcnt, comment="(wait for buffer ready)"))
                    if kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].isInt32():
                        for j in range(0, int(gwvw)):
                            module.add(VAddI32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(data+j), \
                                    comment="buffer add"))
                    else:
                        if ((gwvw % 2) == 1):
                            for j in range(0, int(gwvw)):
                                module.add(VAddF32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(data+j), \
                                                comment="buffer add"))
                        else:
                            for j in range(0, int(gwvw/2)):
                                module.add(VAddPKF32(dst=vgpr(vgprstart+j*2, 2), src0=vgpr(vgprstart+j*2, 2), \
                                                src1=vgpr(data+j*2, 2), comment="buffer pk"))

                    # prefetch
                    if elementIdx != 0:
                        module.add(SAddU32(dst=sgpr(loadOffsetSgpr), src0=sgpr(loadOffsetSgpr), src1=increment, comment="Increase sgpr offset for load"))

                    module.add(writer.chooseGlobalRead(True, bps, data, \
                                    addr0, addr1, soffset=sgpr(loadOffsetSgpr), offset=0, glc=True, slc=True, scope=gsuReadScope,\
                                    comment="prefetch element %d " % (elementIdx)))
                    vlcnt += 1

                module.add(SSubI32(dst=sgpr("GSUSync"), src0=sgpr("GSUSync"), src1=1))
                module.add(SCmpLeI32(src0=sgpr("GSUSync"), src1=0-unrolledWGs, comment=""))
                module.add(SCBranchSCC1(labelName=accumulationEndlabel.getLabelName(), comment="Accumulation finished"))
                module.add(SAddU32(dst=sgpr(tmpS06+0), src0=sgpr(tmpS06+0), src1="MTOffset", comment="" ))
                module.add(SAddCU32(dst=sgpr(tmpS06+1), src0=sgpr(tmpS06+1), src1="MTOffsetH32", comment="" ))
                module.add(SCmpLeI32(src0=sgpr("GSUSync"), src1=0, comment="disable buffer load if GSUSync <= 0"))
                module.add(SCSelectB32(sgpr(tmpS06+2), 0, sgpr(tmpS06+2), ""))

            module.add(SBranch(labelName=accumulationStartlabel.getLabelName(), comment=""))

            for k in range(unrolledWGs, 0, -1):
                module.addSpaceLine()
                module.add(SynchronizerAddEndlabel[k])
                vlcnt = (k+1) * len(batchElements)

                # reduce first 2 WGs
                for elementIdx in range(0, len(batchElements)):
                    vlcnt = vlcnt-2 if vlcnt > 0 else 0
                    vgprstart   = ss.elementSumIdx[elementIdx]
                    data  = tmpVAdd[-1][elementIdx]

                    module.add(SWaitCnt(vlcnt=vlcnt, comment="(wait for buffer ready)"))
                    if kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].isInt32():
                        for j in range(0, int(gwvw)):
                            module.add(VAddI32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(data+j), \
                                    comment="buffer add"))
                    else:
                        if ((gwvw % 2) == 1):
                            for j in range(0, int(gwvw)):
                                module.add(VAddF32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(data+j), \
                                                comment="buffer add"))
                        else:
                            for j in range(0, int(gwvw/2)):
                                module.add(VAddPKF32(dst=vgpr(vgprstart+j*2, 2), src0=vgpr(vgprstart+j*2, 2), \
                                                src1=vgpr(data+j*2, 2), comment="buffer pk"))

                for i in range(1, k):
                    for elementIdx in range(0, len(batchElements)):
                        vlcnt = vlcnt-1 if vlcnt > 0 else 0
                        vgprstart   = ss.elementSumIdx[elementIdx]
                        data  = tmpVAdd[i-1][elementIdx]

                        module.add(SWaitCnt(vlcnt=vlcnt, comment="(wait for buffer ready)"))
                        if kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].isInt32():
                            for j in range(0, int(gwvw)):
                                module.add(VAddI32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(data+j), \
                                        comment="buffer add"))
                        else:
                            if ((gwvw % 2) == 1):
                                for j in range(0, int(gwvw)):
                                    module.add(VAddF32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(data+j), \
                                                comment="buffer add"))
                            else:
                                for j in range(0, int(gwvw/2)):
                                    module.add(VAddPKF32(dst=vgpr(vgprstart+j*2, 2), src0=vgpr(vgprstart+j*2, 2), \
                                                src1=vgpr(data+j*2, 2), comment="buffer pk"))

                module.add(SBranch(labelName=accumulationEndlabel.getLabelName(), comment="Accumulation End"))

            module.addComment("buffer add end\n")
            module.add(accumulationEndlabel)
            writer.sgprPool.checkIn(tmpWSD)

        writer.sgprPool.checkIn(tmpS06)
        writer.sgprPool.checkIn(tmpS05)
        writer.sgprPool.checkIn(tmpS02)
        writer.sgprPool.checkIn(tmpS01)

        return module

    def lastGsuWgReduction(self, kernel, writer, ss, batchIdx, tmpVgpr, tmpVgprDynamic, gwvw, batchElements, codeAccVgprRead, vgproffset, soffset):
        module = Module("lastGsuWgReduction")

        accvgprReadLabel = Label(writer.labels.getNameInc("last_gsu_wg_accvgpr_read"), comment="")
        module.add(accvgprReadLabel)
        module.add(SWaitCnt(vlcnt=0, comment="wait for buffer_load to finish"))

        tmpVAdd = tmpVgprDynamic.idx
        # last gsu wg accvgpr read
        if codeAccVgprRead is not None and kernel["LocalSplitU"] == 1:
            regsPerScalar = writer.states.bpeCinternal // writer.states.bpr # register per scalar
            # loop over store instructions within one batch
            batchNumAccRegs = len(batchElements) * gwvw * regsPerScalar
            vmcntForSummation = int(batchNumAccRegs / gwvw)
            for elementIdx in range(0, len(batchElements)):
                # loop over scalars within one store instruction
                for vi in range(0, gwvw):
                    # loop over registers within one scalar
                    for rIdx in range(0, regsPerScalar):
                        codeAccVgprReadInst = codeAccVgprRead.popFirstItem()
                        codeAccVgprReadInst.dst = vgpr(tmpVAdd + regsPerScalar*vi + rIdx) # replace dst with temp sgpr
                        module.add(codeAccVgprReadInst)

                vgprstart = ss.elementSumIdx[elementIdx]
                dataType  = kernel["ProblemType"]["DestDataType"]
                if dataType.isDouble() or dataType.isSingleComplex():
                    vgprstart = vgprstart*2
                vmcntForSummation = vmcntForSummation - 1
                # module.add(SWaitCnt(vlcnt=vmcntForSummation, comment="wait for buffer_load to finish"))

                GSUP1 = 1 # do 1 element at a time
                for i in range(0, GSUP1):
                    if kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].isInt32():
                        for j in range(0, int(gwvw)):
                            module.add(VAddI32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(tmpVAdd+0+gwvw*i+j), \
                                    comment="buffer add"))
                    else:
                        if ((gwvw % 2) == 1):
                            for j in range(0, int(gwvw)):
                                module.add(VAddF32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(tmpVAdd+0+gwvw*i+j), \
                                        comment="buffer add"))
                        else:
                            for j in range(0, int(gwvw/2)):
                                module.add(VAddPKF32(dst=vgpr(vgprstart+j*2, 2), src0=vgpr(vgprstart+j*2, 2), \
                                            src1=vgpr(tmpVAdd+0+gwvw*i+j*2, 2), comment="buffer pk"))
        elif kernel["LocalSplitU"] > 1:
            # read from LSU VGPRs
            regsPerScalar = writer.states.bpeCinternal // writer.states.bpr # register per scalar
            if kernel["MIArchVgpr"]:
                tmpStartVgprValuC = writer.states.c.startVgprValu
                writer.states.c.startVgprValu = 0
                module.add(RegSet("v", "vgprValuC", 0))
            if ss.lsuStartVgprOffset > 0:
                for elementIdx in range(0, len(batchElements)):
                    for vi in range(0, gwvw):
                        for rIdx in range(0, regsPerScalar):
                            codeAccVgprReadInst = codeAccVgprRead.popFirstItem()
                            codeAccVgprReadInst.dst = vgpr(tmpVAdd + regsPerScalar*vi + rIdx) # replace dst with temp sgpr
                            codeAccVgprReadInst.comment = "copy acc to vreg[%u]" % (tmpVAdd + regsPerScalar*vi + rIdx)
                            module.add(codeAccVgprReadInst)

                    vgprstart = ss.elementSumIdx[elementIdx]
                    dataType  = kernel["ProblemType"]["DestDataType"]
                    if dataType.isDouble() or dataType.isSingleComplex():
                        vgprstart = vgprstart*2

                    GSUP1 = 1 # do 1 element at a time
                    for i in range(0, GSUP1):
                        if kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].isInt32():
                            for j in range(0, int(gwvw)):
                                module.add(VAddI32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(tmpVAdd+0+gwvw*i+j), \
                                        comment="buffer add"))
                        else:
                            if ((gwvw % 2) == 1):
                                for j in range(0, int(gwvw)):
                                    module.add(VAddF32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(tmpVAdd+0+gwvw*i+j), \
                                            comment="buffer add"))
                            else:
                                for j in range(0, int(gwvw/2)):
                                    module.add(VAddPKF32(dst=vgpr(vgprstart+j*2, 2), src0=vgpr(vgprstart+j*2, 2), \
                                                src1=vgpr(tmpVAdd+0+gwvw*i+j*2, 2), comment="buffer pk"))

            ss.lsuStartVgprOffset += len(batchElements) * gwvw * regsPerScalar
            
            if kernel["MIArchVgpr"]:
                writer.states.c.startVgprValu = tmpStartVgprValuC
                module.add(RegSet("v", "vgprValuC", tmpStartVgprValuC))
            else:
                module.add(SNop(1, "2 wait states required before reading vgpr"))

        module.addSpaceLine()

        return module

    def initializeSrd(self, writer, ArgTypeCheckLabel, GeneralBatchedGemmSrdInitiation_End, kernel, ch):
        module = Module("initializeSrd")
        # Special handling for "MultipleBuffer" and "MultipleBufferSingleKernel" for General Batched GEMM
        # ArgType == 3 (General Batched GEMM) but GSU == 1, then SrdC/D will be initialized to correct batch matrix address from pointer array (AddressC/D)
        # ArgType == 3 (General Batched GEMM) but GSU > 1, then SrdC/D will be initialized with workspace.
        # "MultipleBuffer" means both SrdC and SrdD are workspace pointers
        # "MultipleBufferSingleKernel" means only SrdD will be workspace pointer while SrdC will be initialized to correct batch matrix address from pointer array (AddressC)      
        if((kernel["_GlobalAccumulation"] == 'MultipleBuffer') or (kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel')):
            with writer.allocTmpSgpr(1, tag="initializeSrd_tmpSgprGSU") as tmpSgprGSU:
                module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
                module.add(SCmpEQU32(src0=sgpr(tmpSgprGSU.idx), src1=1, comment="GSU == 1 ?"))
                module.add(SCBranchSCC1(labelName=ArgTypeCheckLabel.getLabelName(), comment="Handling General Batched GEMM SRD initialization"))
                if((kernel["_GlobalAccumulation"] == 'MultipleBuffer') or (kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel' and ch == "D")):
                    module.add(SMovB64(dst=sgpr("Srd%s+0"%ch, 2), src=sgpr("Address%s+0"%ch, 2), comment="init SRD base address" )) 
                    module.add(SBranch(labelName=GeneralBatchedGemmSrdInitiation_End.getLabelName(), comment="End of handling General Batched GEMM SRD initialization"))
                module.add(ArgTypeCheckLabel)
        return module

    # GSU = 1, then all C and D will have pointer to device memory holding pointer array to batch matrices. 
    # So we need logic to get the correct batch matrix address into Srd.
    # GSU > 1, then AddressA/B will be pointer to device memory holding pointer array to batch matrices but AddressC/D will further depend on the following:
    # a) MultiBuffer case: AddressC/D will be pointer to workspace and will be handled similar to Strided Batched case.
    # b) MultiBufferSingleKernel case: AddressC will be pointer to device memory holding pointer array to batch matrices while AddressD will be pointer to workspace.
    def routeToGeneralBatchedOrStridedBatched(self, stridedBatchedGemmLoad, argTypeChecks, generalBatchedGemmLoad, mat, kernel, tmpS1):
        module = Module("routeToGeneralBatchedOrStridedBatched")
        module.add(SCmpEQU32(src0=sgpr(tmpS1), src1=1, comment="GSU == 1 ?"))
        if(kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel' and mat == "C"):
            module.add(SCBranchSCC0(labelName=argTypeChecks.getLabelName()))
        else:
            module.add(SCBranchSCC0(labelName=stridedBatchedGemmLoad.getLabelName()))
        if kernel["ProblemType"]["SupportUserArgs"]:
            module.add(SCmpEQU32(src0=sgpr("ArgType"), src1=3, comment="ArgType == 3 for General Batched GEMM"))
            module.add(SCBranchSCC1(labelName=generalBatchedGemmLoad.getLabelName())) 
        if(kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel' and mat == "C"):
            module.add(SBranch(labelName=stridedBatchedGemmLoad.getLabelName()))
            module.add(argTypeChecks)
            if kernel["ProblemType"]["SupportUserArgs"]:
                module.add(SCmpEQU32(src0=sgpr("ArgType"), src1=3, comment="ArgType == 3 for General Batched GEMM"))   
                module.add(SCBranchSCC1(labelName=generalBatchedGemmLoad.getLabelName()))
                module.add(SBranch(labelName=stridedBatchedGemmLoad.getLabelName()))  
        return module
