# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Persistent tile-processing contracts, shared geometry, and full-K processing."""

from rocisa.code import Module, Label
from rocisa.container import sgpr, vgpr, ContinuousRegister
from rocisa.instruction import VMovB32, SBranch, SAndB32, SCSelectB32, SAddU32, SCBranchSCC0, SCBranchSCC1, SCmpEQU32, SCmpGeU32, SCmpLtU32, SLShiftRightB32, SMaxI32, SMovB32, SMulI32, SNop, SSubU32, VReadfirstlaneB32
from rocisa.functions import scalarStaticDivideAndRemainder, sMagicDiv2, BranchIfNotZero, scalarUInt32DivideAndRemainder
from ..Component import Component
import abc
from .Subtile.SubtileLREmit import localReadResetOffsetsSubtile
from ..Common import clusterEnabled, persistentSpatialCluster

class TileProcessingStrategy(Component):
    """Interpret assigned work, compute K bounds and choose completion behavior."""
    emitsParallelReductionSgprAliases = False
    borrowsSrdWsInEpilogue = False
    emitsWorkspaceReductionBpe = False
    requiresWorkspaceReductionStorePath = False
    keepsConstantsInSgpr = False
    supportsSubtileImpl = True

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


    def preLoop(self, writer, kernel):
        module = Module('Persistent TwoTileDPFirst openLoop')
        constantsInVgprs = writer.isPersistentConstantsToVgprEnabled(kernel)
        xccMapping = Component.XCCMapping.find(writer)
        module.add(xccMapping(writer, kernel))
        if writer.states.archCaps['WorkGroupIdFromTTM'] and (not clusterEnabled(kernel['ClusterDim'])):
            module.add(SMovB32(dst=sgpr('WorkGroup0'), src='ttmp9', comment='workaround'))
            module.add(SAndB32(dst=sgpr('WorkGroup1'), src0=hex(65535), src1='ttmp7', comment='workaround'))
            module.add(SLShiftRightB32(dst=sgpr('WorkGroup2'), shiftHex=hex(16), src='ttmp7', comment='workaround'))
        module.add(Component.WorkAssignment.find(writer).persistentClusterPadEarlyExit(writer, kernel))
        if persistentSpatialCluster(kernel):
            with writer.allocTmpSgpr(2, tag='ClusterDPFold') as tRes:
                t0 = tRes.idx
                t1 = tRes.idx + 1
                module.add(SMulI32(dst=sgpr(t0), src0=sgpr('WorkGroup1'), src1=sgpr('NumWorkGroups0'), comment='DP fold: WorkGroup1 * nWG0 (N-tile row)'))
                module.add(SMulI32(dst=sgpr(t1), src0=sgpr('NumWorkGroups0'), src1=sgpr('NumWorkGroups1'), comment='DP fold: nWG0 * nWG1 (tiles per batch)'))
                module.add(SMulI32(dst=sgpr(t1), src0=sgpr(t1), src1=sgpr('WorkGroup2'), comment='DP fold: batch * (nWG0*nWG1)'))
                module.add(SAddU32(dst=sgpr('WorkGroup0'), src0=sgpr('WorkGroup0'), src1=sgpr(t0), comment='DP fold: + WorkGroup1*nWG0'))
                module.add(SAddU32(dst=sgpr('WorkGroup0'), src0=sgpr('WorkGroup0'), src1=sgpr(t1), comment='DP fold: PersistentWorkGroupIndex = batch*(nWG0*nWG1) + N*nWG0 + M'))
        if constantsInVgprs:
            module.add(VMovB32(dst=vgpr(self._constantVgpr(writer, 'PersistentWorkGroupIndex')), src=sgpr('WorkGroup0'), comment='Save original Persistent index to VGPR'))
        else:
            module.add(SMovB32(dst=sgpr('PersistentWorkGroupIndex'), src=sgpr('WorkGroup0'), comment='Save original Persistent index'))
        if persistentSpatialCluster(kernel):
            module.add(Component.WorkAssignment.find(writer).persistentMulticastPrologueSignal(writer, kernel))
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

    def graWorkGroup(self, writer, kernel, tPA, tPB):
        module = Module('Persistent TwoTileDPFirst graWorkGroup')
        constantsInVgprs = writer.isPersistentConstantsToVgprEnabled(kernel)
        sTmp = writer.sgprPool.checkOutAligned(4, 2, 'PersistentMappingTemp', preventOverflow=False)
        sIpt = writer.acquirePersistentConstSgpr(kernel, 'ItersPerTile')
        if constantsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.persistentConstVgprs['ItersPerTile'])))
        module.add(self.computeTotalTiles(writer, kernel, sTmp + 3))
        module.add(SMulI32(dst=sgpr(sTmp + 3), src0=sgpr(sTmp + 3), src1=sgpr(sIpt), comment='dpSectionSize = totalTiles * ItersPerTile'))
        module.add(SCmpLtU32(src0=sgpr('PersistentIteration'), src1=sgpr(sTmp + 3), comment="Make sure there's DP work to do"))
        module.add(writer.longBranchScc0(Label('KernelEnd', ''), posNeg=1))
        writer.releasePersistentConstSgpr(sIpt)
        module.add(self.mapIterationToTile(writer, kernel, sTmp, tPA, tPB))
        sIpt = writer.acquirePersistentConstSgpr(kernel, 'ItersPerTile')
        sGrid = writer.acquirePersistentConstSgpr(kernel, 'skGrid')
        if constantsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.persistentConstVgprs['ItersPerTile'])))
            module.add(VReadfirstlaneB32(dst=sgpr(sGrid), src=vgpr(writer.states.persistentConstVgprs['skGrid'])))
        module.add(SMulI32(dst=sgpr(sTmp + 1), src0=sgpr(sGrid), src1=sgpr(sIpt), comment='DP iterations shift'))
        writer.releasePersistentConstSgpr(sGrid)
        writer.releasePersistentConstSgpr(sIpt)
        module.add(SAddU32(dst=sgpr(sTmp + 1), src0=sgpr(sTmp + 1), src1=sgpr('PersistentIteration'), comment='Add DP shift'))
        module.add(SMovB32(dst=sgpr('PersistentIteration'), src=sgpr(sTmp + 1), comment='Store next DP iteration'))
        module.add(self.tileIndexToWorkGroup(writer, kernel, sTmp))
        writer.sgprPool.checkIn(sTmp)
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

    def rapTileBatch(self, writer, kernel, dstSgpr):
        """Batch index of the tile ``PersistentIteration`` currently points at, into ``dstSgpr``.

        Repeats the arithmetic ``mapIterationToTile`` and ``tileIndexToWorkGroup`` perform -- tile
        index from ``PersistentIteration``, then tile index over the tiles per batch -- but
        writes only ``dstSgpr`` and temporaries. ``mapIterationToTile`` also resets the
        local-read offsets and ``tileIndexToWorkGroup`` claims ``WorkGroup0/1/2``, neither of
        which may happen at a point that might still branch away.

        Valid only at a persistent-loop entry, where ``PersistentIteration`` still names the
        tile about to be computed; ``graWorkGroup`` advances it past that point.

        ReuseAcrossPersistent calls this at both entries to decide whether the
        resident A belongs to the tile this iteration will compute.
        """
        module = Module('Persistent rapTileBatch')
        constantsInVgprs = writer.isPersistentConstantsToVgprEnabled(kernel)
        with writer.allocTmpSgpr(4, 2, 'RAPTileBatchTemp') as sTmpRes:
            sTmp = sTmpRes.idx
            sMagicNum = writer.acquirePersistentConstSgpr(kernel, 'MagicNumberItersPerTile')
            sMagicShift = writer.acquirePersistentConstSgpr(kernel, 'MagicShiftItersPerTile')
            if constantsInVgprs:
                module.add(VReadfirstlaneB32(dst=sgpr(sMagicNum), src=vgpr(writer.states.persistentConstVgprs['MagicNumberItersPerTile'])))
                module.add(VReadfirstlaneB32(dst=sgpr(sMagicShift), src=vgpr(writer.states.persistentConstVgprs['MagicShiftItersPerTile'])))
            module.add(sMagicDiv2(sgpr(sTmp), sgpr(sTmp + 1), sgpr('PersistentIteration'), sgpr(sMagicNum), sgpr(sMagicShift), sgpr(sTmp + 2)))
            writer.releasePersistentConstSgpr(sMagicNum)
            writer.releasePersistentConstSgpr(sMagicShift)
            module.add(SMulI32(dst=sgpr(sTmp + 1), src0=sgpr('NumWorkGroups0'), src1=sgpr('NumWorkGroups1'), comment='RAP: tiles per batch'))
            tmpVgpr = writer.vgprPool.checkOut(2, 'rapTileBatchDiv')
            tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
            module.add(scalarUInt32DivideAndRemainder(qReg=dstSgpr, dReg=sTmp, divReg=sTmp + 1, rReg=sTmp + 3, tmpVgprRes=tmpVgprRes, wavewidth=kernel['WavefrontSize'], doRemainder=False, comment='RAP: batch of the tile at PersistentIteration'))
            writer.vgprPool.checkIn(tmpVgpr)
        return module

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

    def papHasNextPersistentIteration(self, writer, kernel, skipLabel):
        """Emit the PAP "skip if there is no next persistent iteration" predicate.

        This is the variant-specific back-edge test that decides whether the
        PAP next-tile prefetch may run at all. The default (static Persistent:
        Persistent==3 TwoTileDPFirst, and the Persistent3/static path of Persistent==5) tests
        the deterministically-advanced ``PersistentIteration`` against ``PersistentIterationEnd``
        — identical to the historical inline compare in
        ``prefetchAcrossPersistent`` — so the persistent loop's own back-edge
        (``PersistentLoop.closePersistentLoop``) and the PAP skip agree on when
        the current tile is the last one.

        Variants whose next tile comes from a stateful source (e.g. Persistent==4
        PersistentDynamic's per-XCD work-queue pop) override this because they
        cannot cheaply predict the next iteration without consuming queue state.
        """
        module = Module('papHasNextPersistentIteration')
        module.add(SCmpGeU32(src0=sgpr('PersistentIteration'), src1=sgpr('PersistentIterationEnd'), comment='No next persistent iteration'))
        module.add(SCBranchSCC1(labelName=skipLabel.getLabelName(), comment=''))
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


    def storeBranches(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        module = Module("DataParallel storeBranches")
        return module


    def writePartials(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
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


    def _constantVgpr(self, writer, name):
        return writer.states.persistentConstVgprs[name]
