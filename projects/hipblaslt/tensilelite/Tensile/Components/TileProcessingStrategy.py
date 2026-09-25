# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Persistent tile-processing contracts, geometry, and full-K processing."""

from dataclasses import dataclass
from typing import Optional
from rocisa.code import Module, Label
from rocisa.container import sgpr, vgpr, ContinuousRegister
from rocisa.instruction import VMovB32, SBranch, SAndB32, SCSelectB32, SCBranchSCC0, SCmpEQU32, SMaxI32, SMovB32, SMulI32, SNop, SSubU32, SAddU32, SCmpLtU32, VReadfirstlaneB32
from rocisa.functions import scalarStaticDivideAndRemainder, BranchIfNotZero, scalarUInt32DivideAndRemainder, sMagicDiv2
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
        return StaticPartition("PersistentIteration", "PersistentIterationEnd", "skGrid", "PersistentWorkGroupIndex")


    def prefetchAcrossPersistentSetupNextTile(self, writer, kernel, tPA, tPB, skipLroReset=False):
        """Recompute Persistent tile locals and map tile index to WorkGroup* for the *next* tile.

        After each persistent iteration's main body, ``PersistentIteration`` already holds the starting
        global iteration index for the next chunk (set at the beginning of ``graWorkGroup``).
        Running ``mapIterationToTile`` + ``tileIndexToWorkGroup`` + WGM remapping here matches the start of the
        next ``setupNewTile`` / ``graWorkGroup`` (without advancing ``PersistentIteration`` again), so
        SGPRs are warm before the persistent back-edge.

        When ``skipLroReset`` is True the local-read-offset reset inside
        ``mapIterationToTile`` is suppressed.  This is needed when PAP runs *before*
        the NLL body: the NLL still needs the current tile's read pointers."""
        from Tensile.Components.WorkGroupMappingAlgos import DefaultWGM, SpaceFillingCurveWalk
        module = Module('Persistent prefetchAcrossPersistentSetupNextTile')
        with writer.allocTmpSgpr(4, 2, 'PersistentPrefetchTemp') as sTmpRes:
            sTmp = sTmpRes.idx
            module.add(self.mapIterationToTile(writer, kernel, sTmp, tPA, tPB, skipLroReset=skipLroReset))
            module.add(self.tileIndexToWorkGroup(writer, kernel, sTmp))
        if len(kernel['SpaceFillingAlgo']):
            writer.states.WGMTransformLevels = len(kernel['SpaceFillingAlgo'])
            module.add(SpaceFillingCurveWalk(writer, kernel, 'WGM'))
        else:
            module.add(DefaultWGM(writer, kernel, 'WGM'))
        return module

    def calculateLoopNumIter(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        module = Module('Persistent Common calculateLoopNumIter')
        sIpt = writer.acquirePersistentConstSgpr(kernel, 'ItersPerTile')
        if writer.isPersistentConstantsToVgprEnabled(kernel):
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.persistentConstVgprs['ItersPerTile'])))
        module.add(SMovB32(dst=sgpr(loopCounterName), src=sgpr(sIpt), comment='Persistent loop counter = ItersPerTile (DP-only full tile)'))
        writer.releasePersistentConstSgpr(sIpt)
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

    def initializePartition(self, writer, kernel):
        module = Module("DataParallel iteration partition")
        constantsInVgprs = writer.isPersistentConstantsToVgprEnabled(kernel)
        sIdx = writer.acquirePersistentConstSgpr(kernel, 'PersistentWorkGroupIndex')
        sIpt = writer.acquirePersistentConstSgpr(kernel, 'ItersPerTile')
        if constantsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sIdx), src=vgpr(writer.states.persistentConstVgprs['PersistentWorkGroupIndex'])))
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.persistentConstVgprs['ItersPerTile'])))
        module.add(SMulI32(dst=sgpr('PersistentIteration'), src0=sgpr(sIdx), src1=sgpr(sIpt), comment='DP starting iteration'))
        writer.releasePersistentConstSgpr(sIdx)
        with writer.allocTmpSgpr(1, tag='TotalIters') as sTmpRes:
            sTmp = sTmpRes.idx
            module.add(self.computeTotalTiles(writer, kernel, sTmp))
            module.add(SMulI32(dst=sgpr(sTmp), src0=sgpr(sTmp), src1=sgpr(sIpt), comment='totalIters = totalTiles * itersPerTile'))
            module.add(SMovB32(dst=sgpr('PersistentIterationEnd'), src=sgpr(sTmp), comment='DP ending iteration'))
            module.add(SCmpLtU32(src0=sgpr('PersistentIteration'), src1=sgpr(sTmp), comment="Make sure there's work to do"))
        writer.releasePersistentConstSgpr(sIpt)
        module.add(writer.longBranchScc0(Label('KernelEnd', ''), posNeg=1))
        return module

    def mapIterationToTile(self, writer, kernel, sTmp, tPA, tPB, skipLroReset=False):
        module = Module('Persistent mapIterationToTile')
        constantsInVgprs = writer.isPersistentConstantsToVgprEnabled(kernel)
        if kernel['PrefetchGlobalRead'] and (not skipLroReset):
            if not kernel['UseSubtileImpl']:
                module.add(writer.localReadResetOffsets(kernel, tPA))
                if kernel['ProblemType']['MXBlockA'] and 'MX' in tPA:
                    module.add(writer.localReadResetOffsets(kernel, tPA['MX']))
                if kernel['ProblemType']['MXBlockB'] and 'MX' in tPB:
                    module.add(writer.localReadResetOffsets(kernel, tPB['MX']))
                module.add(writer.localReadResetOffsets(kernel, tPB))
            else:
                module.add(localReadResetOffsetsSubtile(writer, kernel))
        module.addComment0('Persistent calculate tile idx and map to WG')
        sMagicNum = writer.acquirePersistentConstSgpr(kernel, 'MagicNumberItersPerTile')
        sMagicShift = writer.acquirePersistentConstSgpr(kernel, 'MagicShiftItersPerTile')
        if constantsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sMagicNum), src=vgpr(writer.states.persistentConstVgprs['MagicNumberItersPerTile'])))
            module.add(VReadfirstlaneB32(dst=sgpr(sMagicShift), src=vgpr(writer.states.persistentConstVgprs['MagicShiftItersPerTile'])))
        sMaskedShift = None
        sMagicShiftForDiv = sMagicShift
        module.add(sMagicDiv2(sgpr(sTmp), sgpr(sTmp + 1), sgpr('PersistentIteration'), sgpr(sMagicNum), sgpr(sMagicShiftForDiv), sgpr(sTmp + 2)))
        if sMaskedShift is not None:
            writer.sgprPool.checkIn(sMaskedShift)
        writer.releasePersistentConstSgpr(sMagicNum)
        writer.releasePersistentConstSgpr(sMagicShift)
        sIpt = writer.acquirePersistentConstSgpr(kernel, 'ItersPerTile')
        if constantsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.persistentConstVgprs['ItersPerTile'])))
        module.add(SMulI32(dst=sgpr(sTmp + 1), src0=sgpr(sTmp), src1=sgpr(sIpt), comment='Tile start iteration'))
        module.add(SAddU32(dst=sgpr(sTmp + 2), src0=sgpr(sTmp + 1), src1=sgpr(sIpt), comment='Tile end iteration'))
        writer.releasePersistentConstSgpr(sIpt)
        return module
