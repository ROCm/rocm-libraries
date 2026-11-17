# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared spatial block geometry and logical-worker/physical-peer boundaries."""

from rocisa.code import Module, Label
from rocisa.container import sgpr, ContinuousRegister
from rocisa.instruction import SAddU32, SAndB32, SCSelectB32, SCMovB32, SCmpEQU32, SCmpGeU32, SLShiftLeftB32, SLShiftRightB32, SMinU32, SMovB32, SMulI32, SSubU32
from rocisa.functions import scalarUInt32DivideAndRemainder
from ..Common import log2, persistentSpatialCluster


class ClusterTileMapping:
    @staticmethod
    def blockCount(writer, kernel, dstSgpr, *, paddedPeers=False):
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
        if paddedPeers:
            module.add(SLShiftLeftB32(dst=sgpr(dstSgpr), shiftHex=hex(log2(cs * ck)), src=sgpr(dstSgpr), comment="totalTiles = blocks * Cs*Ck"))
        return module

    @staticmethod
    def materialize(writer, kernel, sTmp, *, peer=None):
        cs, ck = kernel["ClusterDim"]
        module = Module("DataParallel cluster tileIndexToWorkGroup")
        module.addComment0("Map cluster-block index to wg0/1/2")
        with writer.allocTmpSgpr(4, tag="ClusterTileMapping") as tmp:
            sPeer, sBlocksM, sBlocksMN, sRem = tmp.idx, tmp.idx + 1, tmp.idx + 2, tmp.idx + 3
            if peer is None:
                module.add(SAndB32(dst=sgpr(sPeer), src0=sgpr(sTmp), src1=hex(cs * ck - 1), comment="peer = index % (Cs*Ck)"))
                module.add(SLShiftRightB32(dst=sgpr(sTmp), shiftHex=hex(log2(cs * ck)), src=sgpr(sTmp), comment="block = index / (Cs*Ck)"))
            else:
                module.add(SMovB32(dst=sgpr(sPeer), src=sgpr(peer), comment="spatial peer of this logical worker"))
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

    @staticmethod
    def skipPhantomCompletion(writer, kernel):
        module = Module("DataParallel skipPhantomTileStore")
        if not persistentSpatialCluster(kernel):
            return module
        module.add(SCmpEQU32(src0=sgpr("PersistentPhantomTile"), src1=0, comment="phantom tiles skip the store"))
        module.add(writer.longBranchScc0(Label("PersistentLoopClose", ""), posNeg=1))
        return module

    @staticmethod
    def physicalSlot(kernel, dst, logicalProducer):
        """Lift a logical reduction partner without changing tree arithmetic."""
        module = Module("Cluster physical partial slot")
        cs, cn = kernel["ClusterDim"]
        module.add(SLShiftLeftB32(dst=sgpr(dst), src=logicalProducer,
                                  shiftHex=log2(cs * cn), comment="producer cluster * peers"))
        module.add(SAddU32(dst=sgpr(dst), src0=sgpr(dst), src1=sgpr("StreamKClusterPeer"),
                           comment="physical partial slot = cluster * peers + peer"))
        return module
