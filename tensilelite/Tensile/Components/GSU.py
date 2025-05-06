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

from rocisa import countInstruction
from rocisa.container import ContinuousRegister, SMEMModifiers
from rocisa.instruction import SAddCU32, SAddU32, SAndB32, SAtomicDec, SBranch, SCBranchSCC0, \
    SCBranchSCC1, SCMovB32, SCSelectB32, SCmpEQI32, SCmpEQU32, SCmpGtI32, SCmpLeI32, SCmpLgU32, SCmpLtU32, \
    SLShiftLeftB32, SLShiftLeftB64, SLShiftRightB32, SMovB32, SMovB64, SMulI32, SNop, VReadfirstlaneB32, SSubI32, SSubU32, \
    SWaitCnt, scalarStaticMultiply64, scalarUInt32DivideAndRemainder, VAddF32, VAddPKF32, VCmpGEI32, VCndMaskB32, VMovB32

from ..TensileInstructions import Module, Label, sgpr, vgpr, log2
from ..Component import Component
from ..AsmAddressCalculation import AddrCalculation
import abc

from copy import deepcopy

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

    @abc.abstractmethod
    def noLoadLoop(self, writer, kernel, tensorParametersA, tensorParametersB, pack):
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
    def defineAndResources(self, writer, kernel, tmpSgpr0, tmpSgprM, tmpSgprN, tmpSgprNumWG0):
        pass

    @abc.abstractmethod
    def writeBiasToGlobal(self, writer, kernel, biasDataType, tP, tmpSgprRes, biasBpe):
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
        tcGR = tc if tc == "Metadata" else (tc + "GR")
        dimIdx = kernel["ProblemType"]["IndicesSummation"][loopIdx] # dimension index
        loopChar = writer.states.indexChars[dimIdx]
        stride = writer.strideRef(tc, dimIdx)
        isMirrorIdx = dimIdx in kernel["ProblemType"]["MirrorDims%s"%tc]

        m = "DepthU*Bpe%s"%(tcGR)
        if isMirrorIdx:
          m = "-%s"%(m)

        if writer.states.globalReadIncsUseVgpr:
            with writer.allocTmpSgpr(2) as tmpSgprInfo:
                tmpSgpr = tmpSgprInfo.idx
                module.add(SMovB32(dst=sgpr(tmpSgpr+0), src="DepthU*%d"%(tP["bpeGR"]), comment="DepthU*Bpe"))
                module.add(SMulI32(dst=sgpr(tmpSgpr+0), src0=sgpr(tmpSgpr+0), src1=stride, \
                    comment="incr%s%s = %s*DepthU*bpeGR (unrollIdx)"%(tc, loopChar, stride) ))
                # TODO - this should be mul-H??
                module.add(SMovB32(dst=sgpr(tmpSgpr+1), src=hex(0), comment="(carry)"))
                module.add(VMovB32(dst=vgpr("GlobalReadIncs%s+%u+0"%(tc, 2*loopIdx)), src=sgpr(tmpSgpr+0)))
                module.add(VMovB32(dst=vgpr("GlobalReadIncs%s+%u+1"%(tc, 2*loopIdx)), src=sgpr(tmpSgpr+1)))
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

    def noLoadLoop(self, writer, kernel, tensorParametersA, tensorParametersB, pack):
        module = Module("GSU Off noLoadLoop")
        return module

    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("GSU Off tailLoopNumIter")
        return module

    def setupNewTile(self, writer, kernel, tensorParametersA, tensorParametersB, tPM):
        module = Module("GSU Off setupNewTile")

        module.add(self.graIncrementsAB(writer, kernel, tensorParametersA, tensorParametersB, tPM))

        return module

    def globalWriteBatchProlog(self, writer, kernel, tmpVgpr, tmpVgprSize, tmpVgprDynamic, \
                               batchIdx, ss, gwvw, batchElements, \
                               beta, edge, sumIdxGSUSYNC, addrCalc):
        module = Module("GSU Off globalWriteBatchProlog")
        return module

    def defineAndResources(self, writer, kernel, tmpSgpr0, tmpSgprM, tmpSgprN, tmpSgprNumWG0):
        module = Module("GSU Off defineAndResources")
        return module

    def writeBiasToGlobal(self, writer, kernel, biasDataType, tP, tmpSgprRes, biasBpe):
        module = Module("GSU Off writeBiasToGlobal")
        return module

class GSUOn(GSU):

    @classmethod
    def matches(cls, writer, debug=False):
        return writer.states.kernel["GlobalSplitU"] > 0 or writer.states.kernel["GlobalSplitU"] == -1

    def __call__(self):
        assert(0)

    def graWorkGroup(self, writer, kernel):
        module = Module("GSU On graWorkGroup")

        gsuLabel    = Label(label=writer.labels.getNameInc("GSU"), comment="")
        gsuLabelEnd = Label(label=writer.labels.getNameInc("GSU_End"), comment="")
        with writer.allocTmpSgpr(1) as tmpSgprGSU:
            module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
            module.add(SCmpEQU32(src0=sgpr(tmpSgprGSU.idx), src1=1, comment="GSU == 1 ?"))
            module.add(SCBranchSCC1(labelName=gsuLabel.getLabelName(), comment="branch if GSU == 1"))

        if ((kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel')):
            extReadEpilogueLabeltmp    = Label(label=writer.labels.getNameInc("LoadExternalEpilogueStruct"), comment="")
            module.addComment0("Check if custom structure pointer is null")
            if kernel["ProblemType"]["SupportUserArgs"]:
                module.add(SCmpEQU32(src0=sgpr("ArgType"), src1=2, comment="ArgType == 2 ?"))
                module.add(SCBranchSCC0(labelName=extReadEpilogueLabeltmp.getLabelName()))
            module.add(SMovB64(dst=sgpr("WSDstart",2), src=sgpr("AddressD",2)))
            module.add(SMovB64(dst=sgpr("AddressD",2), src=sgpr("AddressTD",2)))
            module.add(SMovB64(dst=sgpr("AddressTD",2), src=sgpr("WSDstart",2)))
            module.add(extReadEpilogueLabeltmp)

        module.addComment("GSU-not-WGMapRR :nwg1 = (size%s + MT%s - 1) / MT%s;" \
            % (writer.states.tileChar1, writer.states.tileChar1, writer.states.tileChar1))

        tmpVgpr = writer.vgprPool.checkOut(2, "tmp")
        tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
        gsuwgmrrLabel    = Label(label=writer.labels.getNameInc("GSUWGMRR"), comment="")
        gsuwgmrrLabelEnd = Label(label=writer.labels.getNameInc("GSUWGMRR_End"), comment="")
        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            module.add(SAndB32(dst=sgpr(tmpSgprInfo.idx), src0=sgpr("GSU"), src1=hex(0x4000), comment="SCC = (GSUWGMRR == 1) ?"))
            module.add(SCBranchSCC1(labelName=gsuwgmrrLabel.getLabelName(), comment="branch if GSUWGMRR == 1"))
            # wg1       = wg1 / GSU
            # gsuSumIdx = wg1 % GSU
            module.add(SAndB32(dst=sgpr(tmpSgprInfo.idx), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
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
        depthU = kernel["DepthU"]
        depthUDiv = kernel["DepthU"]
        # swizzle
        if (tP["isSwizzled"] and tc == 'A'):
            depthUDiv = "%s%s"%(kernel["DepthU"], "*MI_M")
        elif (tP["isSwizzled"] and tc == 'B'):
            depthUDiv = "%s%s"%(kernel["DepthU"], "*MI_N")

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
        gsucLabel    = Label(label=writer.labels.getNameInc("GSUC_A" if tP["isA"] else "GSUC_B"), comment="")
        gsucLabelEnd = Label(label=writer.labels.getNameInc("GSUC_A_End" if tP["isA"] else "GSUC_B_End"), comment="")
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
        dimIdx = kernel["ProblemType"]["IndicesSummation"][loopIdx] # dimension index
        loopChar = writer.states.indexChars[dimIdx]
        stride = writer.strideRef(tc, dimIdx)
        isMirrorIdx = dimIdx in kernel["ProblemType"]["MirrorDims%s"%tc]

        if writer.states.globalReadIncsUseVgpr:
            with writer.allocTmpSgpr(3) as tmpSgprInfo:
                tmpSgpr = tmpSgprInfo.idx
                gsuSgpr = tmpSgpr + 2
                module.add(SAndB32(dst=sgpr(tmpSgpr), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
                module.add(SMulI32(dst=sgpr(gsuSgpr), src0=sgpr(tmpSgpr), src1="DepthU*%d"%(tP["bpeGR"]), comment="GSU*DepthU*Bpe"))
                module.add(SAndB32(dst=sgpr(tmpSgpr), src0=sgpr("GSU"), src1=hex(0x8000), comment="SCC = (GSUC == 1) ?"))
                module.add(SCMovB32(dst=sgpr(gsuSgpr), src="DepthU*%d"%(tP["bpeGR"]), comment="DepthU*Bpe if GSUC = 1"))
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
            with writer.allocTmpSgpr(2) as tmpSgprInfo:
                tmpSgpr = tmpSgprInfo.idx
                gsuSgpr = tmpSgpr + 1

                tcGR = tc if tc == "Metadata" else (tc + "GR")

                # swizzle
                mult_MI_Dim = ""
                if tc == "A" and kernel["ProblemType"]["SwizzleTensorA"]:
                    mult_MI_Dim = "*MI_M"
                elif tc == "B" and kernel["ProblemType"]["SwizzleTensorB"]:
                    mult_MI_Dim = "*MI_N"

                module.add(SAndB32(dst=sgpr(gsuSgpr), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
                module.add(SMulI32(dst=sgpr(gsuSgpr), src0=sgpr(gsuSgpr), src1="DepthU*Bpe%s%s"%(tcGR, mult_MI_Dim), comment="GSU*DepthU*Bpe%s"%(mult_MI_Dim)))
                module.add(SAndB32(dst=sgpr(tmpSgpr), src0=sgpr("GSU"), src1=hex(0x8000), comment="SCC = (GSUC == 1) ?"))

                m = sgpr(gsuSgpr)

                if isMirrorIdx:
                    m.setMinus(True)

                incr = sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx))
                duBpe = "DepthU*Bpe%s%s"%(tcGR, mult_MI_Dim)
                # multiply by stride, optimizing if unit stride
                if writer.isConstUnitStride(stride):
                    module.add(SCSelectB32(dst=incr, src0=duBpe, src1=m, comment="incr%s (unrollIdx)"%(tc)))
                else:
                    module.add(SCMovB32(dst=m, src=duBpe, comment="DepthU*Bpe if GSUC = 1"))
                    module.add(SMulI32(dst=incr, src0=m, src1=stride, comment="incr%s unrollIdx)"%(tc) ))

                if kernel["ProblemType"]["Sparse"]:
                    if tP["is_sparse"]:
                        module.add(SLShiftRightB32(dst=incr, shiftHex=hex(log2(2)), src=incr))
                    elif tP["isM"]:
                        module.add(SLShiftRightB32(dst=incr, shiftHex=hex(log2(8)), src=incr))

        return module

    def graIncrementsRestore(self, writer, kernel, loopCounterName):
        module = Module("GSU On graIncrementsRestore")

        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            gsuSgpr = tmpSgprInfo.idx
            module.add(SAndB32(dst=sgpr(gsuSgpr), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
            module.add(SMulI32(dst=sgpr(gsuSgpr), src0=sgpr(gsuSgpr), src1=kernel["DepthU"]))
            module.add(SMulI32(dst=sgpr(loopCounterName), src0=sgpr(loopCounterName), \
                               src1=sgpr(gsuSgpr), comment="=loopCounterName*DepthU"))

        return module

    def calculateLoopNumIter(self, writer, kernel, loopCounterName, tmpSgprInfo):
        module = Module("GSU On calculateLoopNumIter")

        tmpSgpr = tmpSgprInfo.idx
        # if GSU numIter++ if gsuSumIdx < remainder
        gsuLabel = Label(label=writer.labels.getNameInc("GSU"), comment="")
        module.add(SAndB32(dst=sgpr(tmpSgpr), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
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

        tmpVgpr = writer.vgprPool.checkOut(2,"tmp")
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
        with writer.allocTmpSgpr(1) as tmpSgprGSU:
            module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
            module.add(SMulI32(dst=sgpr(sgprOut), src0=kernel["DepthU"], src1=sgpr(tmpSgprGSU.idx), comment="IncsMetadata = GSU*DepthU"))
            module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=hex(0x8000), comment="SCC = (GSUC == 1) ?"))
        module.add(SCMovB32(dst=sgpr(sgprOut), src=kernel["DepthU"], comment="IncsMetadata = DepthU if GSUC == 1"))
        module.add(SLShiftRightB32(dst=sgpr(sgprOut), shiftHex=hex(log2(8)), src=sgpr(sgprOut)))
        return module

    def computeStoreSrdStart(self, writer, kernel):
        module = Module("GSU On computeStoreSrdStart")

        indices = list(range(0, kernel["ProblemType"]["NumIndicesC"]))
        numDim = len(indices)

        if kernel["GlobalSplitUAlgorithm"] == 'MultipleBuffer' or kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
            gsuLabel = Label(label=writer.labels.getNameInc("GSU"), comment="")
            with writer.allocTmpSgpr(1) as tmpSgprGSU:
                module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
                module.add(SCmpEQU32(src0=sgpr(tmpSgprGSU.idx), src1=1, comment="GSU == 1 ?"))
                module.add(SCBranchSCC1(labelName=gsuLabel.getLabelName(), comment="branch if GSU == 1"))
            # GSU algorithm 2: adjust output buffer address to per GSU buffer
            with writer.allocTmpSgpr(4, alignment=1) as tmpSgprInfo:
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

    def noLoadLoop(self, writer, kernel, tensorParametersA, tensorParametersB, pack):
        module = Module("GSU On noLoadLoop")

        isDTV = (kernel["DirectToVgprA"] or kernel["DirectToVgprB"])
        needSecondNLL  = isDTV # need 2 NLL for 2 buffers (PGR1/2)
        NLLnum = 2 if needSecondNLL else 1
        gsuLabel = Label(label=writer.labels.getNameInc("GSU"), comment="")
        with writer.allocTmpSgpr(1) as tmpSgprGSU:
            module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
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
              else:
                # deepCopy packCode for OptNLL noLoadLoop
                deepCopyPack = deepcopy(pack)
              noLoadLoopModules.add(writer.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=True, isNGLL=False, pack=deepCopyPack, NLLindex=NLLindex, NLLnum=NLLnum))
              writer.restoreLocalPointers(kernel, tensorParametersA, tensorParametersB)

            acclen = countInstruction(noLoadLoopModules)
        kernel["GlobalSplitU"] = gsuBackup
        kernel["_GlobalAccumulation"] = gsuAccumBackup
        writer.states.bpeCexternal = bpeCexternalBackup

        if acclen > 16384:
            with writer.allocTmpSgpr(3) as tmpSgprInfo:
                module.add(writer.longBranchScc0(gsuLabel, posNeg=1, tmpSgprInfo=tmpSgprInfo, comment="branch if GSU != 1"))
        else:
            module.add(SCBranchSCC0(labelName=gsuLabel.getLabelName(), comment="branch if GSU != 1"))

        if noLoadLoopModules != None:
            module.add(noLoadLoopModules)
        module.add(gsuLabel)

        return module

    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("GSU On tailLoopNumIter")

        with writer.allocTmpSgpr(3) as tmpSgprInfo:
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
            tmpVgpr = writer.vgprPool.checkOut(2,"tmp")
            tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
            module.add(SLShiftRightB32(dst=sgpr(tmpSgpr+1), src=sgpr("SizesSum"), shiftHex=log2(kernel["DepthU"]), \
                                            comment="s%s = s[sgprSizesSum] / %s"%(tmpSgpr+1,kernel["DepthU"])))
            module.add(SAndB32(dst=sgpr(tmpSgpr+2), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
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
            with writer.allocTmpSgpr(1) as tmpSgprGSU:
                module.add(SAndB32(dst=sgpr(tmpSgprGSU.idx), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
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

    def GSUSynccodegen(self, writer, kernel, tmpVgpr, tmpVgprSize, tmpVgprDynamic, batchIdx, ss, gwvw, batchElements, labelend, vgprstart, globalOffset, vgproffset):
        module = Module("GSUSYNC")

        # dot2: enable GSU for non-MFMA mode
        if kernel["EnableMatrixInstruction"]:
            WaveNum = str(kernel["MIWaveGroup"][0]*kernel["MIWaveGroup"][1])
        else:
            WaveNum = str(kernel["NumThreads"] // kernel["WavefrontSize"])

        module.addComment("check done start")

        #####################################synchronizer offset cal and set synchronizer#####################################
        #####################################WaveId+WgId*WaveNum+WgNum*WaveNum*Batch
        #####################################WgId+WaveId*WgNum+WgNum*WaveNum*Batch
        module.addComment("synchronizer offset cal")

        tmpS02 = writer.sgprPool.checkOut(1, preventOverflow=False) #
        tmpS01 = writer.sgprPool.checkOut(1, preventOverflow=False) #
        tmpS03 = writer.sgprPool.checkOut(1, preventOverflow=False) #

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
        if batchIdx > 0:
            module.add(SMulI32(dst=sgpr(tmpS03), src0=sgpr(tmpS03), src1=int(WaveNum), comment="cal a batch offset")) # WgNum*WaveNum
            module.add(SMulI32(dst=sgpr(tmpS03), src0=sgpr(tmpS03), src1=batchIdx, comment="this batch offset")) # WgNum*WaveNum*Batch
            module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1=sgpr(tmpS03))) # WaveId*WgNum+WgId + WgNum*WaveNum*Batch
        module.add(SLShiftLeftB32(dst=sgpr(tmpS01), src=sgpr(tmpS01), shiftHex=hex(2), comment="")) # atomic 32bits
        #####################################set synchronizer
        module.add(SAddU32(dst=sgpr("SrdSync+0"), \
                           src0=sgpr("Synchronizer+0"), \
                           src1=sgpr(tmpS01), \
                           comment="" ))
        module.add(SAddCU32(dst=sgpr("SrdSync+1"), \
                            src0=sgpr("Synchronizer+1"), \
                            src1=0, \
                            comment="" ))

        module.add(SWaitCnt(waitAll=True, comment="wait store done before synchronizer start load and add"))
        module.add(SAndB32(dst=sgpr(tmpS02), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
        module.add(SSubU32(dst=sgpr(tmpS02), src0=sgpr(tmpS02), src1=hex(1), comment=""))
        module.add(SAtomicDec(dst=sgpr(tmpS02), base=sgpr("SrdSync", 2), smem=SMEMModifiers(glc=True)))
        module.addSpaceLine()
        #####################################cal synchronizer sum offset#####################################
        module.addComment("synchronizer sum offset cal")

        tmpS04 = writer.sgprPool.checkOutAligned(2,2, preventOverflow=False) #
        tmpS05 = writer.sgprPool.checkOutAligned(2,2, preventOverflow=False) #

        indices = list(range(0, kernel["ProblemType"]["NumIndicesC"]))
        numDim = len(indices)
        with writer.allocTmpSgpr(5) as tmpSgprInfo:
            tmpSgpr = tmpSgprInfo.idx
            module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgpr+0), sgpr(tmpSgpr+1), sgpr("SizesFree+0"), 1, tmpVgpr, "Free0"))
            for i in range(1, numDim):
                module.add(SSubU32(dst=sgpr(tmpSgpr+4), src0=sgpr("SizesFree+%u"%i), src1=1, comment="Free%u" % i))
                module.add(SMulI32(dst=sgpr(tmpSgpr+4), src0=sgpr(tmpSgpr+4), src1=1, comment="Free%u" % i))
                module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgpr+2), sgpr(tmpSgpr+3), sgpr(tmpSgpr+4), sgpr("StrideC%s"%writer.states.indexChars[i]), tmpVgpr, "Free%u" % i))
                module.add(SAddU32(dst=sgpr(tmpSgpr+0), src0=sgpr(tmpSgpr+0), src1=sgpr(tmpSgpr+2), comment="Free%u" % i))
                module.add(SAddCU32(dst=sgpr(tmpSgpr+1), src0=sgpr(tmpSgpr+1), src1=sgpr(tmpSgpr+3), comment="Free%u" % i))

            bpetmp = int(writer.states.bpr * kernel["ProblemType"]["DestDataType"].numRegisters()) # self.states.bpeCinternal
            module.add(SLShiftLeftB64(dst=sgpr(tmpS04,2), src=sgpr(tmpSgpr+0,2), shiftHex=log2(writer.states.bpeCexternal), comment="scale by bpe"))

        module.addSpaceLine()
        #####################################cal synchronizer sum start#####################################
        # no need to do because we have workspace start sgpr
        #####################################check synchronizer done#####################################
        checkSyncCode = Module("check synchronizer done")
        checkSyncCode.addComment("check synchronizer done")

        checkSyncCode.add(SWaitCnt(lgkmcnt=0, comment="Wait for synchronizer"))
        checkSyncCode.add(SCmpEQU32(
            src0=sgpr(tmpS02), \
            src1=hex(1), \
            comment=""))

        # checkSyncCode.add(SCBranchSCC0(labelName=labelendname, comment=""))
        branch: Module = writer.longBranchScc0(label=labelend, posNeg=1, comment="long branch sync")
        rowIncrement = Module('Skip and row increment')
        totalNumRows = sum(addrCalc.rowInc for addrCalc in ss.elementAddr)

        if totalNumRows > 0 and ss.optSrdIncForRow:
            dstBpe = int(writer.states.bpr * kernel["ProblemType"]["DestDataType"].numRegisters())
            wsBpe = int(writer.states.bpr * kernel["ProblemType"]["ComputeDataType"].numRegisters())
            packedC1 = kernel["PackedC1IndicesX"]
            pIdx = writer.states.indexChars[packedC1[0]]
            rowIncrement.add(AddrCalculation.incrementSrdMultipleRows("SrdTD", f"StrideD{pIdx}", tmpS01, totalNumRows, dstBpe))
            rowIncrement.add(AddrCalculation.incrementSrdMultipleRows("WSDstart", f"StrideD{pIdx}", tmpS01, totalNumRows, wsBpe))

        insertIndex = branch.findIndexByType(SCBranchSCC1)

        if insertIndex is not None and rowIncrement.count():
            branch.add(rowIncrement, insertIndex+1)

        checkSyncCode.add(branch)

        checkSyncCode.addComment("check done end")
        checkSyncCode.addSpaceLine()
        #####################################load buffer#####################################
        checkSyncCode.addComment("buffer load start")
        # common variables
        SyncloadedData = 0
        tmpS06 = writer.sgprPool.checkOutAligned(4,4, preventOverflow=False) #overflow?
        addr1 = sgpr(tmpS06, 4)
        addr0 = vgpr(vgproffset)
        bps = kernel["ProblemType"]["ComputeDataType"].numBytes() * gwvw

        if not kernel["MbskPrefetchOpt"]:
            for elementIdx in range(0, len(batchElements)):
                mask     = ss.elementMask[elementIdx]
                addrCalc: AddrCalculation = ss.elementAddr[elementIdx]
                SyncloadedData = 0

                SynchronizerAddSkiplabelString = "Synchronizer_read_add_skip"
                SynchronizerAddSkipComment = "Synchronizer read add skip"
                SynchronizerAddSkiplabel = Label(writer.labels.getNameInc(SynchronizerAddSkiplabelString), SynchronizerAddSkipComment)

                addr0 = vgpr(addrCalc.addrDVgpr)

                GSUtotal = writer.getMBSKGSUTotal(kernel)
                SynchronizerAddEndlabel = [""] * GSUtotal

                for idx in range(0, GSUtotal):
                    SynchronizerAddEndlabelString = "Synchronizer_read_add_end_"+str(idx+1)
                    SynchronizerAddEndComment = "Synchronizer read add end_"+str(idx+1)
                    SynchronizerAddEndlabel[idx] = Label(writer.labels.getNameInc(SynchronizerAddEndlabelString), SynchronizerAddEndComment)

                bufferOOB = tmpVgpr + tmpVgprSize - 1
                module.add(VMovB32(dst=vgpr(bufferOOB), src="BufferOOB"))

                module.add(SMovB32(sgpr(tmpS06+0), sgpr("WSDstart+0"), "Move workspace start"))
                module.add(SMovB32(sgpr(tmpS06+1), sgpr("WSDstart+1"), "Move workspace start"))
                module.add(SMovB32(sgpr(tmpS06+2), sgpr("SrdD+2"), ""))
                module.add(SMovB32(sgpr(tmpS06+3), sgpr("SrdD+3"), ""))

                if elementIdx == 0:
                    # Insert check synchronizer done code here for better scheduling
                    module.add(checkSyncCode)

                for times in range(elementIdx, elementIdx+1):
                    addrCalctmp: AddrCalculation = ss.elementAddr[times]
                    if ss.optSrdIncForRow and addrCalctmp.rowInc:
                        module.add(addrCalctmp.incrementToNextRow(kernel, "D", ss, tmpS05, dst=tmpS06))
                        module.add(SAddU32(dst=sgpr("WSDstart+0"), \
                                           src0=sgpr("WSDstart+0"), \
                                           src1=sgpr(tmpS05), \
                                           comment="" ))
                        module.add(SAddCU32(dst=sgpr("WSDstart+1"), \
                                            src0=sgpr("WSDstart+1"), \
                                            src1=0, \
                                            comment="" ))

                vgprstart = ss.elementSumIdx[elementIdx] #here
                dataType  = kernel["ProblemType"]["DestDataType"]
                if dataType.isDouble() or dataType.isSingleComplex():
                    vgprstart = vgprstart*2
                module.add(writer.chooseGlobalRead(True, bps, vgprstart, \
                           addr0, addr1, soffset=0, offset=addrCalc.globalOffset, glc=True, slc=True,\
                           comment="load GSU D 0 "+str(vgprstart)))
                SyncloadedData += 1

                module.add(SAndB32(dst=sgpr("GSUSync"), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))

                SynchronizerlabelString = "Synchronizer_read_add"
                SynchronizerComment = "Synchronizer read add"
                Synchronizerlabel = Label(writer.labels.getNameInc(SynchronizerlabelString), SynchronizerComment)

                tmpVAdd = tmpVgprDynamic
                GSUMvgpr = tmpVgpr

                GSUP1 = GSUtotal-1

                for i in range(0,GSUP1):
                    module.add(SSubI32(dst=sgpr("GSUSync"), src0=sgpr("GSUSync"), src1=1, comment="%u" % i))

                    module.add(SAddU32(dst=sgpr(tmpS06+0), \
                                       src0=sgpr(tmpS06+0), \
                                       src1=sgpr(tmpS04+0), \
                                       comment="" ))
                    module.add(SAddCU32(dst=sgpr(tmpS06+1), \
                                        src0=sgpr(tmpS06+1), \
                                        src1=sgpr(tmpS04+1), \
                                        comment="" ))

                    module.add(SCmpEQI32(src0=sgpr("GSUSync"), src1=0, comment=""))#GSUSync+GSUP1==GSU
                    module.add(SCBranchSCC1(labelName=SynchronizerAddEndlabel[i].getLabelName(), comment="SyncAddbranchhere"))

                    if(kernel["ProblemType"]["DestDataType"].numRegisters() > 1):
                        module.add(writer.chooseGlobalRead(True, bps, tmpVAdd+gwvw*kernel["ProblemType"]["DestDataType"].numRegisters()*i, \
                                   addr0, addr1, soffset=0, offset=addrCalc.globalOffset, glc=True, slc=True, \
                                   comment="load GSU DD %u %u %u" % (bps, gwvw, kernel["ProblemType"]["DestDataType"].numRegisters())))
                    else:
                        module.add(writer.chooseGlobalRead(True, bps, tmpVAdd+gwvw*i, \
                                   addr0, addr1, soffset=0, offset=addrCalc.globalOffset, glc=True, slc=True, \
                                   comment="load GSU DD %u %u %u" % (bps, gwvw, kernel["ProblemType"]["DestDataType"].numRegisters())))

                    SyncloadedData += 1
                module.addComment("buffer load end\n")

                #####################################> GSUtotal reduction start#####################################
                module.addComment("buffer add start")
                vscnt = 0
                lgkmcnt = -1
                vmcnt = SyncloadedData = SyncloadedData -1

                module.add(Synchronizerlabel)

                for i in range(0, GSUP1):
                    module.addSpaceLine()
                    vmcnt = SyncloadedData = SyncloadedData -1
                    module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=vscnt, comment="(wait for buffer ready)"))

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

                    module.add(SAddU32(dst=sgpr(tmpS06+0), \
                                       src0=sgpr(tmpS06+0), \
                                       src1=sgpr(tmpS04+0), \
                                       comment="" ))
                    module.add(SAddCU32(dst=sgpr(tmpS06+1), \
                                        src0=sgpr(tmpS06+1), \
                                        src1=sgpr(tmpS04+1), \
                                        comment="" ))

                    module.add(VCmpGEI32(dst=sgpr(tmpS05,2), src0=0, src1=sgpr("GSUSync"), comment=""))
                    module.add(VCndMaskB32(
                            dst=vgpr(GSUMvgpr), \
                            src1=vgpr(bufferOOB), \
                            src0=addr0, \
                            src2=sgpr(tmpS05,2), \
                            comment="protect if OOB"))

                    if(kernel["ProblemType"]["DestDataType"].numRegisters() > 1):
                        module.add(writer.chooseGlobalRead(True, bps, tmpVAdd+gwvw*kernel["ProblemType"]["DestDataType"].numRegisters()*i, \
                                   vgpr(GSUMvgpr), addr1, soffset=0, offset=addrCalc.globalOffset, glc=True, slc=True, \
                                   comment="load GSU DD %u" % bps))
                    else:
                        module.add(writer.chooseGlobalRead(True, bps, tmpVAdd+gwvw*i, \
                                   vgpr(GSUMvgpr), addr1, soffset=0, offset=addrCalc.globalOffset, glc=True, slc=True, \
                                   comment="load GSU DD %u" % bps))

                    SyncloadedData += 1

                module.addComment("buffer add end\n")

                module.add(SCmpGtI32(
                    src0=sgpr("GSUSync"), \
                    src1=hex(1-(GSUP1)), \
                    comment=""))
                module.add(SCBranchSCC1(labelName=Synchronizerlabel.getLabelName(), comment="Syncbranchhere"))

                #####################################< GSUtotal reduction start#####################################
                for k in range(GSUtotal-2, -1, -1):
                    module.addSpaceLine()
                    module.add(SynchronizerAddEndlabel[k])

                    vmcnt = k
                    for i in range(0, k):
                        module.addSpaceLine()
                        vmcnt = vmcnt -1 if vmcnt > 0 else 0
                        module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=vscnt, comment="(wait for buffer ready)"))

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
                module.addComment("buffer add end2\n")

        else:
            tmpWSD = writer.sgprPool.checkOutAligned(4, 4, preventOverflow=False)
            GSUtotal = writer.getMBSKGSUTotal(kernel)-1
            loadWidth = gwvw * int(max(1, kernel["ProblemType"]["DestDataType"].numRegisters()))
            unrolledWGs = GSUtotal // len(batchElements)
            tmpVidx = tmpVgprDynamic
            tmpVAdd = [[0] * len(batchElements) for _ in range(unrolledWGs)]
            for i in range(0, unrolledWGs):
                for j in range(0, len(batchElements)):
                    tmpVAdd[i][j] = tmpVidx
                    tmpVidx += loadWidth

            SynchronizerAddEndlabel = [""] * (unrolledWGs+1)

            for idx in range(0, unrolledWGs+1):
                SynchronizerAddEndlabelString = "Synchronizer_read_add_end_"+str(idx+1)
                SynchronizerAddEndComment = "Synchronizer read add end_"+str(idx+1)
                SynchronizerAddEndlabel[idx] = Label(writer.labels.getNameInc(SynchronizerAddEndlabelString), SynchronizerAddEndComment)

            # set buffer load address for WG0
            module.add(SMovB64(sgpr(tmpWSD+0, 2), sgpr("WSDstart+0", 2), "Move workspace start"))
            module.add(SMovB64(sgpr(tmpWSD+2, 2), sgpr("SrdD+2", 2), ""))

            # Insert check synchronizer done code here for better scheduling
            module.add(checkSyncCode)

            # set buffer load address for WG1
            module.add(SAddU32(dst=sgpr("WSDstart+0"), src0=sgpr("WSDstart+0"), src1=sgpr(tmpS04+0), comment="" ))
            module.add(SAddCU32(dst=sgpr("WSDstart+1"), src0=sgpr("WSDstart+1"), src1=sgpr(tmpS04+1), comment="" ))
            module.add(SMovB64(sgpr(tmpS06+0, 2), sgpr("WSDstart+0", 2), "Move workspace start"))
            module.add(SMovB64(sgpr(tmpS06+2, 2), sgpr("SrdD+2", 2), ""))

            ReductionStartlabel = Label(writer.labels.getNameInc("Reduction_Start"), "Reduction Start")
            ReductionEndlabel   = Label(writer.labels.getNameInc("Reduction_End"), "Reduction End")
            module.add(SAndB32(dst=sgpr("GSUSync"), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))

            # pre-load
            SyncloadedData = 0

            # first 2 WGs: read same element first for earlier reduction
            for elementIdx in range(0, len(batchElements)):
                addrCalc: AddrCalculation = ss.elementAddr[elementIdx]
                addr0    = vgpr(addrCalc.addrDVgpr)
                for uidx in range(0, 2):
                    if uidx == 0:
                        data = ss.elementSumIdx[elementIdx]
                        tmpAddr1 = tmpWSD
                    else:
                        data = tmpVAdd[-1][elementIdx]
                        tmpAddr1 = tmpS06

                    if ss.optSrdIncForRow and addrCalc.rowInc:
                        module.add(addrCalc.incrementToNextRow(kernel, "D", ss, tmpS05, dst=tmpAddr1))

                    module.add(writer.chooseGlobalRead(True, bps, data, \
                               addr0, sgpr(tmpAddr1, 4), soffset=0, offset=addrCalc.globalOffset, glc=True, slc=True,\
                               comment="load GSU WG %d element %d " % (uidx, elementIdx)))

                    SyncloadedData += 1

            module.add(SSubI32(dst=sgpr("GSUSync"), src0=sgpr("GSUSync"), src1=2))
            module.add(SCmpLeI32(src0=sgpr("GSUSync"), src1=0, comment="GSUSync <= 0?"))
            module.add(SCBranchSCC1(labelName=SynchronizerAddEndlabel[1].getLabelName(), comment=""))
            module.add(SAddU32(dst=sgpr("WSDstart+0"), src0=sgpr("WSDstart+0"), src1=sgpr(tmpS04+0), comment="" ))
            module.add(SAddCU32(dst=sgpr("WSDstart+1"), src0=sgpr("WSDstart+1"), src1=sgpr(tmpS04+1), comment="" ))
            module.add(SMovB64(sgpr(tmpS06+0, 2), sgpr("WSDstart+0", 2), "Move workspace start"))

            # other WGs: read all elements together
            for uidx in range(2, unrolledWGs+1):
                for elementIdx in range(0, len(batchElements)):
                    addrCalc: AddrCalculation = ss.elementAddr[elementIdx]
                    addr0    = vgpr(addrCalc.addrDVgpr)
                    data = tmpVAdd[uidx-2][elementIdx]

                    if ss.optSrdIncForRow and addrCalc.rowInc:
                        module.add(addrCalc.incrementToNextRow(kernel, "D", ss, tmpS05, dst=tmpS06))

                    module.add(writer.chooseGlobalRead(True, bps, data, \
                               addr0, addr1, soffset=0, offset=addrCalc.globalOffset, glc=True, slc=True,\
                               comment="load GSU WG %d element %d " % (uidx, elementIdx)))
                    SyncloadedData += 1

                module.add(SSubI32(dst=sgpr("GSUSync"), src0=sgpr("GSUSync"), src1=1))
                module.add(SCmpLeI32(src0=sgpr("GSUSync"), src1=0, comment="GSUSync <= 0?"))
                module.add(SCBranchSCC1(labelName=SynchronizerAddEndlabel[uidx].getLabelName(), comment=""))
                module.add(SAddU32(dst=sgpr("WSDstart+0"), src0=sgpr("WSDstart+0"), src1=sgpr(tmpS04+0), comment="" ))
                module.add(SAddCU32(dst=sgpr("WSDstart+1"), src0=sgpr("WSDstart+1"), src1=sgpr(tmpS04+1), comment="" ))
                module.add(SMovB64(sgpr(tmpS06+0, 2), sgpr("WSDstart+0", 2), "Move workspace start"))

            module.addComment("buffer load end\n")

            ##################################### reduction start #####################################
            module.addComment("buffer add start")

            vscnt = 0
            lgkmcnt = -1
            vmcnt = SyncloadedData

            # reduce first 2 WGs
            for elementIdx in range(0, len(batchElements)):
                addrCalc: AddrCalculation = ss.elementAddr[elementIdx]
                addr0 = vgpr(addrCalc.addrDVgpr)
                data = tmpVAdd[-1][elementIdx]
                vgprstart   = ss.elementSumIdx[elementIdx]
                vmcnt       = vmcnt - 2 if vmcnt > 0 else 0

                module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=vscnt, comment="(wait for buffer ready)"))
                if ((gwvw % 2) == 1):
                    for j in range(0, int(gwvw)):
                        module.add(VAddF32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(data+j), \
                                   comment="buffer add"))
                else:
                    for j in range(0, int(gwvw/2)):
                        module.add(VAddPKF32(dst=vgpr(vgprstart+j*2, 2), src0=vgpr(vgprstart+j*2, 2), \
                                   src1=vgpr(data+j*2, 2), comment="buffer pk"))

                # prefetch 
                if ss.optSrdIncForRow and addrCalc.rowInc:
                    module.add(addrCalc.incrementToNextRow(kernel, "D", ss, tmpS05, dst=tmpS06))

                module.add(writer.chooseGlobalRead(True, bps, data, \
                           addr0, addr1, soffset=0, offset=addrCalc.globalOffset, glc=True, slc=True,\
                           comment="prefetch element %d " % (elementIdx)))
                vmcnt += 1

            module.add(SSubI32(dst=sgpr("GSUSync"), src0=sgpr("GSUSync"), src1=1))
            module.add(SCmpLeI32(src0=sgpr("GSUSync"), src1=0-unrolledWGs, comment=""))
            module.add(SCBranchSCC1(labelName=ReductionEndlabel.getLabelName(), comment="Reduction finished"))
            module.add(SAddU32(dst=sgpr("WSDstart+0"), src0=sgpr("WSDstart+0"), src1=sgpr(tmpS04+0), comment="" ))
            module.add(SAddCU32(dst=sgpr("WSDstart+1"), src0=sgpr("WSDstart+1"), src1=sgpr(tmpS04+1), comment="" ))
            module.add(SMovB64(sgpr(tmpS06+0, 2), sgpr("WSDstart+0", 2), "Move workspace start"))
            module.add(SCmpLeI32(src0=sgpr("GSUSync"), src1=0, comment="disable buffer load if GSUSync <= 0"))
            module.add(SCSelectB32(sgpr(tmpS06+2), 0, sgpr(tmpS06+2), ""))
            module.add(ReductionStartlabel)

            for uidx in range(0, unrolledWGs):
                for elementIdx in range(0, len(batchElements)):
                    addrCalc: AddrCalculation = ss.elementAddr[elementIdx]
                    addr0    = vgpr(addrCalc.addrDVgpr)
                    data     = tmpVAdd[uidx][elementIdx]
                    vgprstart   = ss.elementSumIdx[elementIdx]
                    vmcnt       = vmcnt -1 if vmcnt > 0 else 0

                    module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=vscnt, comment="(wait for buffer ready)"))
                    if ((gwvw % 2) == 1):
                        for j in range(0, int(gwvw)):
                            module.add(VAddF32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(data+j), \
                                               comment="buffer add"))
                    else:
                        for j in range(0, int(gwvw/2)):
                            module.add(VAddPKF32(dst=vgpr(vgprstart+j*2, 2), src0=vgpr(vgprstart+j*2, 2), \
                                                 src1=vgpr(data+j*2, 2), comment="buffer pk"))

                    # prefetch 
                    if ss.optSrdIncForRow and addrCalc.rowInc:
                        module.add(addrCalc.incrementToNextRow(kernel, "D", ss, tmpS05, dst=tmpS06))

                    module.add(writer.chooseGlobalRead(True, bps, data, \
                               addr0, addr1, soffset=0, offset=addrCalc.globalOffset, glc=True, slc=True,\
                               comment="prefetch element %d " % (elementIdx)))
                    vmcnt += 1

                module.add(SSubI32(dst=sgpr("GSUSync"), src0=sgpr("GSUSync"), src1=1))
                module.add(SCmpLeI32(src0=sgpr("GSUSync"), src1=0-unrolledWGs, comment=""))
                module.add(SCBranchSCC1(labelName=ReductionEndlabel.getLabelName(), comment="Reduction finished"))
                module.add(SAddU32(dst=sgpr("WSDstart+0"), src0=sgpr("WSDstart+0"), src1=sgpr(tmpS04+0), comment="" ))
                module.add(SAddCU32(dst=sgpr("WSDstart+1"), src0=sgpr("WSDstart+1"), src1=sgpr(tmpS04+1), comment="" ))
                module.add(SMovB64(sgpr(tmpS06+0, 2), sgpr("WSDstart+0", 2), "Move workspace start"))
                module.add(SCmpLeI32(src0=sgpr("GSUSync"), src1=0, comment="disable buffer load if GSUSync <= 0"))
                module.add(SCSelectB32(sgpr(tmpS06+2), 0, sgpr(tmpS06+2), ""))

            module.add(SBranch(labelName=ReductionStartlabel.getLabelName(), comment=""))

            for k in range(unrolledWGs, 0, -1):
                module.addSpaceLine()
                module.add(SynchronizerAddEndlabel[k])
                vmcnt = (k+1) * len(batchElements)

                # reduce first 2 WGs
                module.addSpaceLine()
                for elementIdx in range(0, len(batchElements)):
                    vmcnt = vmcnt-2 if vmcnt > 0 else 0
                    vgprstart   = ss.elementSumIdx[elementIdx]
                    data  = tmpVAdd[-1][elementIdx]

                    module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=vscnt, comment="(wait for buffer ready)"))

                    if ((gwvw % 2) == 1):
                        for j in range(0, int(gwvw)):
                            module.add(VAddF32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(data+j), \
                                               comment="buffer add"))
                    else:
                        for j in range(0, int(gwvw/2)):
                            module.add(VAddPKF32(dst=vgpr(vgprstart+j*2, 2), src0=vgpr(vgprstart+j*2, 2), \
                                                 src1=vgpr(data+j*2, 2), comment="buffer pk"))

                for i in range(1, k):
                    module.addSpaceLine()
                    for elementIdx in range(0, len(batchElements)):
                        vmcnt = vmcnt-1 if vmcnt > 0 else 0
                        vgprstart   = ss.elementSumIdx[elementIdx]
                        data  = tmpVAdd[i-1][elementIdx]

                        module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=vscnt, comment="(wait for buffer ready)"))

                        if ((gwvw % 2) == 1):
                            for j in range(0, int(gwvw)):
                                module.add(VAddF32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(data+j), \
                                                   comment="buffer add"))
                        else:
                            for j in range(0, int(gwvw/2)):
                                module.add(VAddPKF32(dst=vgpr(vgprstart+j*2, 2), src0=vgpr(vgprstart+j*2, 2), \
                                                     src1=vgpr(data+j*2, 2), comment="buffer pk"))

                module.add(SBranch(labelName=ReductionEndlabel.getLabelName(), comment="Reduction End"))

            module.addComment("buffer add end\n")
            module.add(ReductionEndlabel)
            module.add(SMovB64(sgpr("WSDstart+0", 2), sgpr(tmpWSD+0, 2), "restore for next batch"))
            writer.sgprPool.checkIn(tmpWSD)

        writer.sgprPool.checkIn(tmpS06)
        writer.sgprPool.checkIn(tmpS05)
        writer.sgprPool.checkIn(tmpS04)
        writer.sgprPool.checkIn(tmpS03)
        writer.sgprPool.checkIn(tmpS02)
        writer.sgprPool.checkIn(tmpS01)

        return module

    def globalWriteBatchProlog(self, writer, kernel, tmpVgpr, tmpVgprSize, tmpVgprDynamic, \
                               batchIdx, ss, gwvw, batchElements, \
                               beta, edge, sumIdxGSUSYNC, addrCalc):
        module = Module("GSU On globalWriteBatchProlog")

        if kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
            if writer.states.serializedStore:
                module.add(SNop(0, "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst"))

            ########################################################

            module.addSpaceLine()
            SynchronizerEndlabelString = "Sync_EDN%s%s" % ("_Beta" if beta else "", "_Edge" if edge else "" )
            SynchronizerEndlabelComment = "Sync_EDN"
            SynchronizerEndlabel = Label(writer.labels.getNameInc(SynchronizerEndlabelString), SynchronizerEndlabelComment)
            SynchronizerEndlabel = Label(writer.labels.getName(SynchronizerEndlabelString), SynchronizerEndlabelComment)

            module.addCommentAlign("source store done, GSU:"+str(kernel["GlobalSplitU"])) #GSUSYNC
            module.addSpaceLine()

            module.add(self.GSUSynccodegen(writer, kernel, tmpVgpr, tmpVgprSize, tmpVgprDynamic, \
                                           batchIdx, ss, gwvw, batchElements, \
                                           SynchronizerEndlabel, sumIdxGSUSYNC, addrCalc.globalOffset, addrCalc.addrDVgpr))

        return module

    def defineAndResources(self, writer, kernel, tmpSgpr0, tmpSgprM, tmpSgprN, tmpSgprNumWG0):
        module = Module("GSU On defineAndResources")

        if kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel':
            extReadEpilogueLabeltmp    = Label(label=writer.labels.getNameInc("LoadExternalEpilogueStruct"), comment="")
            module.addComment0("Check if custom structure pointer is null")
            if kernel["ProblemType"]["SupportUserArgs"]:
                module.add(SCmpEQU32(src0=sgpr("ArgType"), src1=2, comment="ArgType == 2 ?"))
                module.add(SCBranchSCC0(labelName=extReadEpilogueLabeltmp.getLabelName()))
            module.add(SMulI32(dst=sgpr(tmpSgpr0), src0=sgpr(tmpSgprM), src1=sgpr(tmpSgprN)))
            module.add(SAndB32(dst=sgpr(tmpSgprNumWG0), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
            module.add(SMulI32(dst=sgpr(tmpSgpr0), src0=sgpr(tmpSgpr0), src1=sgpr(tmpSgprNumWG0)))
            module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr0), src=sgpr(tmpSgpr0), shiftHex=(2)))
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
            if kernel["GlobalSplitUAlgorithm"] == "MultipleBuffer":
                module.add(SAndB32(dst=sgpr(tmpSgpr+2), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
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