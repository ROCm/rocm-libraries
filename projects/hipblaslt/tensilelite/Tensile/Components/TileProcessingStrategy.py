# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Persistent tile-processing contracts, geometry, and full-K processing."""

from dataclasses import dataclass
from typing import Optional
from rocisa.code import Module, Label
from rocisa.container import sgpr, vgpr, ContinuousRegister
from rocisa.instruction import VMovB32, SBranch, SAndB32, SCSelectB32, SCBranchSCC0, SCmpEQU32, SMaxI32, SMovB32, SMulI32, SNop, SSubU32
from rocisa.functions import scalarStaticDivideAndRemainder, BranchIfNotZero, scalarUInt32DivideAndRemainder
from ..Component import Component
import abc
from .Subtile.SubtileLREmit import localReadResetOffsetsSubtile
from .WorkAssignment import StaticPartition

@dataclass(frozen=True)
class TileWork:
    """Codegen view; names refer to existing registers, never a device struct."""
    tile: str
    k_start: Optional[str] = None
    k_end: Optional[str] = None
    completion_identity: tuple = ()

    def borrowedIdentity(self, kernel):
        """Registers materializing another tile may borrow until compute resumes."""
        names = ["WorkGroup0", "WorkGroup1", "WorkGroup2"]
        names.extend(name for name in (self.k_start, self.k_end) if name is not None)
        if kernel["SpaceFillingAlgo"]:
            names.append(self.tile)
        names.extend(self.completion_identity)
        return names

class TileProcessingStrategy(Component):
    """Interpret assigned work, compute K bounds and choose completion behavior."""
    emitsParallelReductionSgprAliases = False
    borrowsSrdWsInEpilogue = False
    emitsWorkspaceReductionBpe = False
    requiresWorkspaceReductionStorePath = False
    keepsConstantsInSgpr = False
    supportsSubtileImpl = True

    def tileWork(self, kernel):
        return TileWork("PersistentTileID")

    def prefetchEligibility(self, writer, kernel, skip):
        return Module("Full tile allows persistent prefetch")

    @abc.abstractmethod
    def __call__(self):
        pass

    def computeTotalTiles(self, writer, kernel, dstSgpr):
        module = Module('Persistent computeTotalTiles')
        module.add(SMulI32(dst=sgpr(dstSgpr), src0=sgpr('NumWorkGroups0'), src1=sgpr('NumWorkGroups1'), comment='totalTiles = nwg0 * nwg1'))
        for i in range(kernel['ProblemType']['NumIndicesC'] - kernel['ProblemType']['NumIndicesFree']):
            batchIdx = kernel['ProblemType']['NumIndicesFree'] + i
            module.add(SMulI32(dst=sgpr(dstSgpr), src0=sgpr(dstSgpr), src1=sgpr('SizesFree+%u' % batchIdx), comment='totalTiles *= batch dim %u' % i))
        return module

    def tileIndexToWorkGroup(self, writer, kernel, sTmp):
        module = Module('Persistent tileIndexToWorkGroup')
        module.addComment0('Map persistent tile index to wg0/1/2')
        module.add(SMulI32(dst=sgpr(sTmp + 1), src0=sgpr('NumWorkGroups0'), src1=sgpr('NumWorkGroups1'), comment='Total tiles'))
        tmpVgpr = writer.vgprPool.checkOut(2, 'div')
        tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
        module.add(scalarUInt32DivideAndRemainder(qReg='WorkGroup2', dReg=sTmp, divReg=sTmp + 1, rReg=sTmp + 2, tmpVgprRes=tmpVgprRes, wavewidth=kernel['WavefrontSize'], doRemainder=True, comment='TileID // nWG0*nWG1'))
        if kernel['SpaceFillingAlgo']:
            module.add(SNop(waitState=1, comment=''))
            module.add(SMovB32(dst=sgpr('PersistentTileID'), src=sgpr(sTmp + 2), comment=''))
        module.add(scalarUInt32DivideAndRemainder(qReg='WorkGroup1', dReg=sTmp + 2, divReg='NumWorkGroups0', rReg='WorkGroup0', tmpVgprRes=tmpVgprRes, wavewidth=kernel['WavefrontSize'], doRemainder=True, comment='TileID // nWG0'))
        tmpVgprRes = None
        writer.vgprPool.checkIn(tmpVgpr)
        module.addSpaceLine()
        return module

class DataParallel(TileProcessingStrategy):
    """Persistent full-K tiles with deterministic grid-stride assignment."""
    kernel = {"TileProcessingStrategy": "DataParallel", "WorkAssignment": "StaticGrid"}

    def __call__(self):
        raise NotImplementedError

    def tileWork(self, kernel):
        return TileWork("PersistentTileID")

    def persistentTileRegisters(self, kernel):
        return []

    def persistentWorkspaceRegisters(self, kernel):
        return []

    def staticPartition(self):
        return StaticPartition("NextTile", "TotalTiles", "PersistentGrid", "NextTile", tile_units=True)

    def materializeTile(self, writer, kernel, tile, tPA, tPB, skipLroReset=False):
        """Map a full-K tile; assignment owns all cursor changes."""
        module = Module("DataParallel materializeTile")
        if kernel["PrefetchGlobalRead"] and not skipLroReset:
            if kernel["UseSubtileImpl"]:
                module.add(localReadResetOffsetsSubtile(writer, kernel))
            else:
                module.add(writer.localReadResetOffsets(kernel, tPA))
                if kernel["ProblemType"]["MXBlockA"] and "MX" in tPA:
                    module.add(writer.localReadResetOffsets(kernel, tPA["MX"]))
                if kernel["ProblemType"]["MXBlockB"] and "MX" in tPB:
                    module.add(writer.localReadResetOffsets(kernel, tPB["MX"]))
                module.add(writer.localReadResetOffsets(kernel, tPB))
        with writer.allocTmpSgpr(3, 2, "PersistentTileMapping") as tmp:
            module.add(SMovB32(dst=sgpr(tmp.idx), src=sgpr(tile), comment="Assigned output tile"))
            module.add(self.tileIndexToWorkGroup(writer, kernel, tmp.idx))
        return module

    def prefetchAcrossPersistentSetupNextTile(self, writer, kernel, tPA, tPB, skipLroReset=False):
        from Tensile.Components.WorkGroupMappingAlgos import DefaultWGM, SpaceFillingCurveWalk
        module = self.materializeTile(writer, kernel, self.staticPartition().cursor, tPA, tPB, skipLroReset=skipLroReset)
        if kernel["SpaceFillingAlgo"]:
            writer.states.WGMTransformLevels = len(kernel["SpaceFillingAlgo"])
            module.add(SpaceFillingCurveWalk(writer, kernel, "WGM"))
        else:
            module.add(DefaultWGM(writer, kernel, "WGM"))
        return module

    def calculateLoopNumIter(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        module = Module('Persistent Common calculateLoopNumIter')
        module.add(SMovB32(dst=sgpr(loopCounterName), src=sgpr("ItersPerTile"), comment="Full-tile K loop count"))
        # The scheduling ABI reserves one iteration for K=0 so every output
        # tile is visited. Its compute loop must still skip the empty sum.
        module.add(SCmpEQU32(src0=sgpr("SizesSum+%u" % writer.states.unrollIdx), src1=0, comment="Empty summation"))
        module.add(SCSelectB32(dst=sgpr(loopCounterName), src0=0, src1=sgpr(loopCounterName), comment="K=0 still stores the tile but issues no compute"))
        alphaLabel2 = Label(writer.labels.getNameInc('PersistentAlphaCheck'), '')
        module.add(BranchIfNotZero('Alpha', kernel['ProblemType']['ComputeDataType'].toEnum(), alphaLabel2))
        module.add(SMovB32(dst=sgpr(loopCounterName), src=0, comment='Skip iterations'))
        module.add(alphaLabel2)
        if not kernel['NoTailLoop']:
            tmpSgpr = tmpSgprInfo.idx
            unrollIdx = writer.states.unrollIdx
            loopChar = writer.states.indexChars[kernel['ProblemType']['IndicesSummation'][unrollIdx]]
            assert kernel['DepthU'] % 2 == 0
            maxUnit = writer.states.tailloopInNllmaxUnit
            if not (writer.states.tailloopInNll and maxUnit == 1):
                if kernel['DepthU'] & kernel['DepthU'] - 1 == 0:
                    module.add(scalarStaticDivideAndRemainder(qReg=tmpSgpr, rReg=tmpSgpr + 1, dReg='SizesSum+%u' % unrollIdx, divisor=kernel['DepthU'], tmpSgprRes=None, doRemainder=2))
                else:
                    with writer.allocTmpSgpr(4, tag='calculateLoopNumIterCommon_tmpSgpr1') as tmpSgpr1:
                        module.add(scalarStaticDivideAndRemainder(qReg=tmpSgpr, rReg=tmpSgpr + 1, dReg='SizesSum+%u' % unrollIdx, divisor=kernel['DepthU'], tmpSgprRes=tmpSgpr1, doRemainder=2))
                module.add(SCmpEQU32(src0=sgpr(tmpSgpr + 1), src1=0, comment='numIter%s == 0' % loopChar))
                module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=0, src1=1, comment='check if size uses tail loop'))
                if writer.states.tailloopInNll and maxUnit > 1:
                    module.add(SAndB32(dst=sgpr(tmpSgpr + 2), src0=sgpr('SizesSum+%u' % unrollIdx), src1=maxUnit - 1, comment='if summation is not multiple of %u, skip tailloopInNll' % maxUnit))
                    module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=0, comment='do not decrement in tailloopInNll case'))
                module.add(SSubU32(dst=sgpr(loopCounterName), src0=sgpr(loopCounterName), src1=sgpr(tmpSgpr), comment='Adjust loop counter for tail loop'))
                module.add(SMaxI32(dst=sgpr(loopCounterName), src0=sgpr(loopCounterName), src1=0, comment='Avoid setting negative value to loopCounter'))
        return module

    def computeLoadSrd(self, writer, kernel, tP, sTmp):
        module = Module("DataParallel computeLoadSrd")
        return module


    def computeStoreSrdStart(self, writer, kernel):
        module = Module("DataParallel computeStoreSrdStart")
        return module


    def graAddresses(self, writer, kernel, tP, vTmp):
        module = Module("DataParallel graAddresses")

        tc = tP["tensorChar"]
        module.add(VMovB32(dst=vgpr(vTmp+0), src=sgpr("Address%s+0" % tc)))
        module.add(VMovB32(dst=vgpr(vTmp+1), src=sgpr("Address%s+1" % tc)))

        return module


    def declareStaggerParms(self, writer, kernel):
        module = Module("DataParallel declareStaggerParms")
        return module


    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("DataParallel tailLoopNumIter")
        return module


    def storeBranches(self, writer, kernel, partialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        module = Module("DataParallel storeBranches")
        return module


    def writePartials(self, writer, kernel, partialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
        module = Module("DataParallel writePartials")
        return module


    def initializeSrdAddressFlagsCheck(self, GeneralBatchedGemmSrdInitiation):
        module = Module("DataParallel initializeSrdAddressFlagsCheck")
        module.add(SBranch(labelName=GeneralBatchedGemmSrdInitiation.getLabelName(), comment="General Batched GEMM, Srd initialized to 0"))
        return module


    def routeToGeneralBatchedOrStridedBatched(self, writer, stridedBatchedGemmLoad, generalBatchedGemmLoad, kernel):
        module = Module("DataParallel routeToGeneralBatchedOrStridedBatched")
        if kernel["ProblemType"]["SupportUserArgs"]:
            writer.cmpNamedArgTypeEq(module, 3, "ArgType == 3 for General Batched GEMM")
            module.add(SCBranchSCC0(labelName=stridedBatchedGemmLoad.getLabelName()))
            module.add(SBranch(labelName=generalBatchedGemmLoad.getLabelName(),
                               comment="General batched output uses the pointer array"))
        return module


    def kernelEnd(self, writer, kernel):
        module = Module("DataParallel kernelEnd")
        return module
