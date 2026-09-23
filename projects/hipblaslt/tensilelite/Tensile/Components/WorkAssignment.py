# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Persistent assignment progress and worker mapping."""

import abc
from rocisa.code import Module, Label
from rocisa.container import sgpr, vgpr, ContinuousRegister
from rocisa.instruction import SBarrier, SBranch, SCmpEQU32, SCmpGeU32, SCBranchSCC0, SLongBranchNegative, SAddU32, SAndB32, SCBranchSCC1, SCSelectB32, SCmpLtU32, SEndpgm, SMulI32, VReadfirstlaneB32
from ..Component import Component
from rocisa.functions import scalarStaticDivideAndRemainder
from ..Common import clusterEnabled, persistentSpatialCluster, persistentMulticast

class WorkAssignment(Component):
    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

    def _clusterElectArriveSignal(self, writer, module, *, labelBase, electTag, wait=False):
        skipSignal = Label(label=writer.labels.getNameInc(labelBase), comment='')
        elect = writer.sgprPool.checkOut(1, electTag)
        module.add(VReadfirstlaneB32(dst=sgpr(elect), src=vgpr('Serial'), comment='wave 0 signals the cluster'))
        module.add(SCmpEQU32(src0=sgpr(elect), src1=0, comment='Check for wave 0'))
        module.add(SCBranchSCC0(labelName=skipSignal.getLabelName(), comment='only wave 0 signals the cluster'))
        module.add(SBarrier(True, False, True, comment='cluster_barrier signal (arrive)'))
        module.add(skipSignal)
        if wait:
            module.add(SBarrier(True, True, True, comment='cluster_barrier wait'))
        writer.sgprPool.checkIn(elect)
        return module

    def persistentMulticastPrologueSignal(self, writer, kernel):
        module = Module('Persistent multicast prologue signal')
        if not persistentSpatialCluster(kernel):
            return module
        assert writer.states.asmCaps.get('HasClusterBarrier', False), 'cluster B-multicast requires the HasClusterBarrier asm capability'
        module.addComment0('cluster B-multicast: elect wave 0 to signal the cluster barrier (pairs first-load wait)')
        self._clusterElectArriveSignal(writer, module, labelBase='PersistentMC_SkipSignal', electTag='PersistentMulticastElect')
        return module

    def persistentMulticastProloguePrefetchHandshake(self, writer, kernel):
        module = Module('Persistent multicast prologue prefetch cluster handshake')
        if not persistentMulticast(kernel):
            return module
        assert writer.states.asmCaps.get('HasClusterBarrier', False), 'cluster B-multicast requires the HasClusterBarrier asm capability'
        module.addComment0('cluster B-multicast: bracket prologue double-buffer prefetch load with cluster handshake')
        self._clusterElectArriveSignal(writer, module, labelBase='PersistentMC_SkipPrefetchSignal', electTag='PersistentMulticastPrefetchElect', wait=True)
        return module

    def persistentMulticastZeroIterClusterWait(self, writer, kernel):
        module = Module('Persistent multicast zero-iteration cluster wait')
        if not persistentSpatialCluster(kernel):
            return module
        assert writer.states.asmCaps.get('HasClusterBarrier', False), 'cluster B-multicast requires the HasClusterBarrier asm capability'
        module.addComment0('cluster B-multicast: zero-iteration skip path consumes the prologue cluster arrive (pairs prologue arrive)')
        skipWait = Label(label=writer.labels.getNameInc('PersistentMC_SkipZeroIterClusterWait'), comment='')
        module.add(SCBranchSCC0(labelName=skipWait.getLabelName(), comment='>=1 full iteration: the first-load cluster wait pairs the arrive'))
        module.add(SBarrier(True, True, True, comment='cluster_barrier wait'))
        module.add(skipWait)
        return module

    def persistentClusterPadEarlyExit(self, writer, kernel):
        module = Module('Persistent cluster pad early-exit')
        if not persistentSpatialCluster(kernel):
            return module
        assert clusterEnabled(kernel['ClusterDim']), 'persistentClusterPadEarlyExit requires an enabled cluster'
        module.addComment1('Persistent cluster multicast: exit padded boundary-cluster peers before the cluster barrier (grid rounded up to ClusterDim)')
        padExit = Label(writer.labels.getNameInc('PersistentClusterPad_EarlyStop'), '')
        padNoExit = Label(writer.labels.getNameInc('PersistentClusterPad_NoEarlyStop'), '')
        module.add(SCmpGeU32(src0=sgpr('WorkGroup0'), src1=sgpr('NumWorkGroups0'), comment='padded if WorkGroup0 (M-tile) >= tilesM'))
        module.add(SCBranchSCC1(labelName=padExit.getLabelName()))
        with writer.allocTmpSgpr(1, tag='persistentClusterPad_tmpSgpr') as padTmp:
            boundN = 'NumWorkGroups1'
            if kernel['GlobalSplitU'] != 0:
                module.add(SAndB32(dst=sgpr(padTmp.idx), src0=sgpr('GSU'), src1=writer.gsuMaskHex(kernel), comment='Restore GSU'))
                module.add(SMulI32(dst=sgpr(padTmp.idx), src0=sgpr('NumWorkGroups1'), src1=sgpr(padTmp.idx), comment='tilesN * GSU'))
                boundN = padTmp.idx
            module.add(SCmpGeU32(src0=sgpr('WorkGroup1'), src1=sgpr(boundN), comment='padded if WorkGroup1 (N-tile) >= tilesN*GSU'))
            module.add(SCBranchSCC1(labelName=padExit.getLabelName()))
            module.add(SBranch(labelName=padNoExit.getLabelName()))
            module.add(padExit)
            module.add(SEndpgm(comment='padded work-group: exit before any cluster barrier/load (WAVEDONE frees -3 barrier slot)'))
            module.add(padNoExit)
        return module

class StaticGrid(WorkAssignment):
    def __call__(self):
        raise NotImplementedError

    kernel = {"WorkAssignment": "StaticGrid"}

    def closeLoop(self, writer, kernel):
        module = Module("StaticGrid closeLoop")
        module.add(SCmpGeU32(src0=sgpr("PersistentIteration"), src1=sgpr("PersistentIterationEnd"),
                            comment="Check whether assigned work is exhausted"))
        module.add(writer.longBranchScc0(Label(writer.rapPersistentLoopEntryLabel(kernel), ""), posNeg=-1))
        return module

class DynamicWorkQueue(WorkAssignment):
    def __call__(self):
        raise NotImplementedError

    kernel = {"WorkAssignment": "DynamicWorkQueue"}

    def closeLoop(self, writer, kernel):
        module = Module("DynamicWorkQueue closeLoop")
        module.add(SBarrier(comment="Sync before work-queue persistent re-entry"))
        with writer.allocTmpSgpr(3, tag="DynamicWorkQueue_closeLoop") as tmp:
            module.add(SLongBranchNegative(Label("PersistentLoopStart", ""), tmp))
        return module

class Hybrid(WorkAssignment):
    def __call__(self):
        raise NotImplementedError

    kernel = {"WorkAssignment": "Hybrid"}

    def closeLoop(self, writer, kernel):
        module = Module("Hybrid closeLoop")
        dynamic = Label("Hybrid_DynamicClose", "")
        done = Label("Hybrid_CloseDone", "")
        module.add(SCmpEQU32(src0=sgpr("WorkAssignmentMode"), src1=0,
                            comment="Hybrid: static assignment selected"))
        module.add(SCBranchSCC0(labelName=dynamic.getLabelName()))
        module.add(StaticGrid().closeLoop(writer, kernel))
        module.add(SBranch(labelName=done.getLabelName()))
        module.add(dynamic)
        module.add(DynamicWorkQueue().closeLoop(writer, kernel))
        module.add(done)
        return module

class XCCMapping(Component):
    """
    XCC mapping code.
    """

class XCCMappingOff(XCCMapping):
    kernel = {"PersistentXCCMapping": 0}

    def __call__(self, writer, kernel):
        module = Module("XCCMapping Off")
        return module

class XCCMappingOn(XCCMapping):

    @classmethod
    def matches(cls, writer, debug=False):
        return writer.states.kernel["PersistentXCCMapping"] > 0

    def __call__(self, writer, kernel):
        module = Module("XCCMapping On")

        with writer.allocTmpSgpr(4, tag="PersistentXCCMappingOn_tmpSgprRes") as tmpSgprRes:
            sXCC   = tmpSgprRes.idx
            sGridC = tmpSgprRes.idx + 1
            sGridF = tmpSgprRes.idx + 2
            sGridM = tmpSgprRes.idx + 3
            sTmp = None
            sTmpRes = None
            sqTmp = writer.sgprPool.checkOut(1, "sqTmp")
            divisor = kernel["PersistentXCCMapping"]
            if ((divisor & (divisor - 1)) != 0): # Need temp registers if not power of 2
                sTmp = writer.sgprPool.checkOutAligned(2, 2, "sTmp", preventOverflow=not kernel.get("UseSubtileImpl", False))
                sTmpRes  = ContinuousRegister(idx=sTmp, size=2)

            # sGridC = ceil(grid / xccm)
            sGrid = writer.acquirePersistentConstSgpr(kernel, "skGrid")
            if writer.isPersistentConstantsToVgprEnabled(kernel):
                module.add(VReadfirstlaneB32(dst=sgpr(sGrid), src=vgpr(writer.states.persistentConstVgprs["skGrid"])))
            module.add(SAddU32(dst=sgpr(sGridC), src0=sgpr(sGrid), src1=hex(kernel["PersistentXCCMapping"] - 1), comment="ceil(grid/xccm)"))
            module.add(scalarStaticDivideAndRemainder(qReg=sGridC, rReg=-1, dReg=sGridC, divisor=kernel["PersistentXCCMapping"], tmpSgprRes=sTmpRes, doRemainder=0))
            # sGridF = floor(grid / xccm)
            # sGridM = grid % xccm
            module.add(scalarStaticDivideAndRemainder(qReg=sGridF, rReg=sGridM, dReg=sGrid, divisor=kernel["PersistentXCCMapping"], tmpSgprRes=sTmpRes))
            writer.releasePersistentConstSgpr(sGrid)
            # sXCC = wg0 % xccm
            # sqtmp is temp register for quotient for non-power-of-2 case
            # sqtmp overlaps temp registers, works in this case and output is discarded
            module.add(scalarStaticDivideAndRemainder(qReg=sqTmp, rReg=sXCC, dReg="WorkGroup0", divisor=kernel["PersistentXCCMapping"], tmpSgprRes=sTmpRes, doRemainder=2))
            # Check if current XCC requires a remainder WG or not
            module.add(SCmpLtU32(src0=sgpr(sXCC), src1=sgpr(sGridM), comment="XCCM < Remainder"))
            module.add(SCSelectB32(dst=sgpr(sGridC), src0=sgpr(sGridC), src1=sgpr(sGridF), comment="Select multiplier"))
            module.add(SCSelectB32(dst=sgpr(sGridM), src0=0, src1=sgpr(sGridM), comment="Select remainder"))
            # WG = floor(wg0 / xccm) * xccm + XCCoffset + optional remainder
            module.add(scalarStaticDivideAndRemainder(qReg="WorkGroup0", rReg=-1, dReg="WorkGroup0", divisor=kernel["PersistentXCCMapping"], tmpSgprRes=sTmpRes, doRemainder=0))
            module.add(SMulI32(dst=sgpr(sXCC), src0=sgpr(sXCC), src1=sgpr(sGridC), comment="XCC group id"))
            module.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(sXCC), comment="Add XCC group offset"))
            module.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(sGridM), comment="Add remainder offset"))

            writer.sgprPool.checkIn(sqTmp)
            if sTmp is not None:
                writer.sgprPool.checkIn(sTmp)

        return module
