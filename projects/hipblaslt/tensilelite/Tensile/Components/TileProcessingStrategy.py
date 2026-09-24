# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Persistent tile-processing contracts, geometry, and full-K processing."""

from dataclasses import dataclass
from typing import Optional
from rocisa.code import Module, Label
from rocisa.container import sgpr, vgpr, ContinuousRegister
from rocisa.instruction import VMovB32, SBranch, SAndB32, SCSelectB32, SCBranchSCC0, SCmpEQU32, SMaxI32, SMovB32, SMulI32, SNop, SSubU32, \
    SAddU32, SCMovB32, SCmpGeU32, SLShiftLeftB32, SLShiftRightB32, SMinU32
from rocisa.functions import scalarStaticDivideAndRemainder, BranchIfNotZero, scalarUInt32DivideAndRemainder
from ..Common import log2, persistentSpatialCluster
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

    def skipPhantomTileStore(self, writer, kernel):
        return Module("No phantom tiles")

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
        return ["PersistentPhantomTile"] if persistentSpatialCluster(kernel) else []

    def persistentWorkspaceRegisters(self, kernel):
        return []

    def staticPartition(self):
        return StaticPartition("NextTile", "TotalTiles", "PersistentGrid", "NextTile", tile_units=True)

    # Spatial clusters walk the tile space in Cs x Ck blocks: a cluster owns whole
    # blocks, so every peer has the same trip count and its M/N neighbours stay the
    # multicast partners fixed by its hardware position. The linear index is
    # block * (Cs*Ck) + peerY*Cs + peerX, and the bound is padded to whole blocks.
    def computeTotalTiles(self, writer, kernel, dstSgpr):
        if not persistentSpatialCluster(kernel):
            return super().computeTotalTiles(writer, kernel, dstSgpr)
        cs, ck = kernel["ClusterDim"]
        module = Module("DataParallel cluster computeTotalTiles")
        module.add(SAddU32(dst=sgpr(dstSgpr), src0=sgpr("NumWorkGroups0"), src1=hex(cs - 1)))
        module.add(SLShiftRightB32(dst=sgpr(dstSgpr), shiftHex=hex(log2(cs)), src=sgpr(dstSgpr), comment="blocksM = ceil(nwg0 / Cs)"))
        with writer.allocTmpSgpr(1, tag="ClusterTotalTiles") as tmp:
            module.add(SAddU32(dst=sgpr(tmp.idx), src0=sgpr("NumWorkGroups1"), src1=hex(ck - 1)))
            module.add(SLShiftRightB32(dst=sgpr(tmp.idx), shiftHex=hex(log2(ck)), src=sgpr(tmp.idx), comment="blocksN = ceil(nwg1 / Ck)"))
            module.add(SMulI32(dst=sgpr(dstSgpr), src0=sgpr(dstSgpr), src1=sgpr(tmp.idx), comment="blocks = blocksM * blocksN"))
        for i in range(kernel['ProblemType']['NumIndicesC'] - kernel['ProblemType']['NumIndicesFree']):
            batchIdx = kernel['ProblemType']['NumIndicesFree'] + i
            module.add(SMulI32(dst=sgpr(dstSgpr), src0=sgpr(dstSgpr), src1=sgpr('SizesFree+%u' % batchIdx), comment='blocks *= batch dim %u' % i))
        module.add(SLShiftLeftB32(dst=sgpr(dstSgpr), shiftHex=hex(log2(cs * ck)), src=sgpr(dstSgpr), comment="totalTiles = blocks * Cs*Ck"))
        return module

    def tileIndexToWorkGroup(self, writer, kernel, sTmp):
        if not persistentSpatialCluster(kernel):
            return super().tileIndexToWorkGroup(writer, kernel, sTmp)
        cs, ck = kernel["ClusterDim"]
        module = Module("DataParallel cluster tileIndexToWorkGroup")
        module.addComment0("Map cluster-block index to wg0/1/2")
        with writer.allocTmpSgpr(4, tag="ClusterTileMapping") as tmp:
            sPeer, sBlocksM, sBlocksMN, sRem = tmp.idx, tmp.idx + 1, tmp.idx + 2, tmp.idx + 3
            module.add(SAndB32(dst=sgpr(sPeer), src0=sgpr(sTmp), src1=hex(cs * ck - 1), comment="peer = index % (Cs*Ck)"))
            module.add(SLShiftRightB32(dst=sgpr(sTmp), shiftHex=hex(log2(cs * ck)), src=sgpr(sTmp), comment="block = index / (Cs*Ck)"))
            module.add(SAddU32(dst=sgpr(sBlocksM), src0=sgpr("NumWorkGroups0"), src1=hex(cs - 1)))
            module.add(SLShiftRightB32(dst=sgpr(sBlocksM), shiftHex=hex(log2(cs)), src=sgpr(sBlocksM), comment="blocksM = ceil(nWG0 / Cs)"))
            module.add(SAddU32(dst=sgpr(sBlocksMN), src0=sgpr("NumWorkGroups1"), src1=hex(ck - 1)))
            module.add(SLShiftRightB32(dst=sgpr(sBlocksMN), shiftHex=hex(log2(ck)), src=sgpr(sBlocksMN), comment="blocksN = ceil(nWG1 / Ck)"))
            module.add(SMulI32(dst=sgpr(sBlocksMN), src0=sgpr(sBlocksMN), src1=sgpr(sBlocksM), comment="blocks per batch"))
            tmpVgpr = writer.vgprPool.checkOut(2, 'div')
            tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
            module.add(scalarUInt32DivideAndRemainder(qReg='WorkGroup2', dReg=sTmp, divReg=sBlocksMN, rReg=sRem, tmpVgprRes=tmpVgprRes, wavewidth=kernel['WavefrontSize'], doRemainder=True, comment='block // blocksM*blocksN'))
            module.add(scalarUInt32DivideAndRemainder(qReg='WorkGroup1', dReg=sRem, divReg=sBlocksM, rReg='WorkGroup0', tmpVgprRes=tmpVgprRes, wavewidth=kernel['WavefrontSize'], doRemainder=True, comment='block // blocksM'))
            tmpVgprRes = None
            writer.vgprPool.checkIn(tmpVgpr)
            module.add(SLShiftLeftB32(dst=sgpr("WorkGroup0"), shiftHex=hex(log2(cs)), src=sgpr("WorkGroup0"), comment="blockM * Cs"))
            module.add(SAndB32(dst=sgpr(sRem), src0=sgpr(sPeer), src1=hex(cs - 1), comment="peerX"))
            module.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(sRem), comment="M tile = blockM*Cs + peerX"))
            module.add(SLShiftLeftB32(dst=sgpr("WorkGroup1"), shiftHex=hex(log2(ck)), src=sgpr("WorkGroup1"), comment="blockN * Ck"))
            module.add(SLShiftRightB32(dst=sgpr(sPeer), shiftHex=hex(log2(cs)), src=sgpr(sPeer), comment="peerY"))
            module.add(SAddU32(dst=sgpr("WorkGroup1"), src0=sgpr("WorkGroup1"), src1=sgpr(sPeer), comment="N tile = blockN*Ck + peerY"))
            # A peer past the tile edge of a boundary block still issues the same
            # multicast loads as its partners, so it aliases the edge tile and
            # only skips the store.
            module.add(SCmpGeU32(src0=sgpr("WorkGroup0"), src1=sgpr("NumWorkGroups0"), comment="M tile past the edge?"))
            module.add(SCSelectB32(dst=sgpr("PersistentPhantomTile"), src0=1, src1=0))
            module.add(SCmpGeU32(src0=sgpr("WorkGroup1"), src1=sgpr("NumWorkGroups1"), comment="N tile past the edge?"))
            module.add(SCMovB32(dst=sgpr("PersistentPhantomTile"), src=1, comment="phantom tile: compute, do not store"))
            module.add(SSubU32(dst=sgpr(sRem), src0=sgpr("NumWorkGroups0"), src1=1))
            module.add(SMinU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(sRem), comment="clamp M tile to the edge"))
            module.add(SSubU32(dst=sgpr(sRem), src0=sgpr("NumWorkGroups1"), src1=1))
            module.add(SMinU32(dst=sgpr("WorkGroup1"), src0=sgpr("WorkGroup1"), src1=sgpr(sRem), comment="clamp N tile to the edge"))
        module.addSpaceLine()
        return module

    def skipPhantomTileStore(self, writer, kernel):
        module = Module("DataParallel skipPhantomTileStore")
        if not persistentSpatialCluster(kernel):
            return module
        module.add(SCmpEQU32(src0=sgpr("PersistentPhantomTile"), src1=0, comment="phantom tiles skip the store"))
        module.add(writer.longBranchScc0(Label("PersistentLoopClose", ""), posNeg=1))
        return module

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
