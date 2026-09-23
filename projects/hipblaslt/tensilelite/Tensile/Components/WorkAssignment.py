# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Work assignment, reservation, worker mapping, and synchronization."""

from dataclasses import dataclass
from rocisa.code import Module, Label
from rocisa.container import sgpr, vgpr, SMEMModifiers, GLOBALModifiers, EXEC, ContinuousRegister, DSModifiers, MemTokenData
from rocisa.instruction import SMovB32, VMovB32, VReadfirstlaneB32, SCmpEQU32, SCBranchSCC1, SBranch, SAddCU32, SAddU32, SAndB32, SBarrier, SCBranchSCC0, SCSelectB32, SCmpGeU32, SCmpLtU32, SLShiftLeftB32, SLShiftRightB32, VLShiftLeftB32, SMovB64, SMulI32, SNop, SSubU32, SXorB32, SWaitAlu, SWaitCnt, SWaitXCnt, VSubU32, VCmpXEqU32, GlobalAtomicIncU32Saddr, SLongBranchNegative, SAtomicInc, DSLoadB32, DSStoreB32, SEndpgm
import abc
from ..Component import Component
from ..Common import log2, clusterEnabled, persistentSpatialCluster, persistentMulticast
from rocisa.enum import CacheScope
from rocisa.functions import scalarUInt32DivideAndRemainder, scalarStaticDivideAndRemainder, sMagicDiv2
from ..ExecutionPolicy import hasDynamicAssignment, hasHybridAssignment

@dataclass(frozen=True)
class QueuePartition:
    """Live ABI names supplied by the K-partition interpretation."""
    grid: str
    total_items: str
    address: str
    raw_rank: str
    sticky_empty: str
    launch_rank: str = "PersistentWorkGroupIndex"
    unique_labels: bool = False

@dataclass(frozen=True)
class StaticPartition:
    """Cursor storage and units supplied by the selected work interpretation."""
    cursor: str
    bound: str
    grid: str
    initial_rank: str

def _mailboxLds0Token(writer):
    return MemTokenData([writer.states.memTokenLdsBuffer0])


def _emitMailboxAddressAndWave0Skip(writer, module, vLocalAddress, skipLabel,
                                    preventOverflow=True):
    # Per-wave mailbox slot in LDS[0..124]: (Serial<<2) - (tid0<<2).
    # tid0 is firstlane(Serial). Serial is written at kernel start.
    sTid0 = writer.sgprPool.checkOut(1, "MailboxFirstTid", preventOverflow=preventOverflow)
    sBase = writer.sgprPool.checkOut(1, "MailboxWaveBase", preventOverflow=preventOverflow)
    module.add(VReadfirstlaneB32(dst=sgpr(sTid0), src=vgpr("Serial"),
                                 comment="wave first thread id from Serial"))
    module.add(VLShiftLeftB32(dst=vgpr(vLocalAddress), src=vgpr("Serial"), shiftHex=log2(4),
                              comment="Serial * 4"))
    # firstlane dest is an SGPR; wait before the wave-base SALU.
    module.add(SNop(waitState=2, comment="wait after readfirstlane before SALU"))
    module.add(SWaitAlu(va_sdst=0, comment="va_sdst: firstlane(Serial) ready for wave-base SALU"))
    module.add(SLShiftLeftB32(dst=sgpr(sBase), src=sgpr(sTid0), shiftHex=log2(4),
                              comment="wave base in bytes"))
    module.add(VSubU32(dst=vgpr(vLocalAddress), src0=vgpr(vLocalAddress), src1=sgpr(sBase)))
    writer.sgprPool.checkIn(sBase)
    module.add(SCmpEQU32(src0=sgpr(sTid0), src1=0, comment="Check for wave 0"))
    writer.sgprPool.checkIn(sTid0)
    module.add(SCBranchSCC0(labelName=skipLabel.getLabelName(), comment="Skip work item"))


def _emitWorkItemMailbox(writer, module, vLocalAddress, vWaveWorkItemIdx, skipLabel,
                         sWorkItemIdx=None):
    # Mailbox DS ops occupy LDS[0..124] (TDM buffer 0). Token them as LDS0
    # so the scheduler places a publish fence between store and load, and a
    # release before the next LDS0 write. WG barriers stay untokened.
    mailboxToken = _mailboxLds0Token(writer)
    storeInst = DSStoreB32(dstAddr=vgpr(vLocalAddress), src=vgpr(vWaveWorkItemIdx),
                           ds=DSModifiers(offset=0))
    storeInst.setMemToken(mailboxToken)
    module.add(storeInst)
    module.add(SWaitCnt(dscnt=0))
    module.add(skipLabel)
    module.add(SBarrier(comment="mailbox publish: wave 0 store visible"))
    loadInst = DSLoadB32(dst=vgpr(vWaveWorkItemIdx), src=vgpr(vLocalAddress),
                         ds=DSModifiers(offset=0))
    loadInst.setMemToken(_mailboxLds0Token(writer))
    module.add(loadInst)
    module.add(SWaitCnt(dscnt=0))
    if sWorkItemIdx is not None:
        module.add(VReadfirstlaneB32(dst=sgpr(sWorkItemIdx), src=vgpr(vWaveWorkItemIdx),
                                     comment="Read work item index from vgpr"))
        module.add(SBarrier(comment="mailbox index visible to all waves"))


def _extract_hybrid_mode():
    """Extract SK5 mode bit 30 into WorkAssignmentMode; clear it in MagicShiftItersPerTile."""
    module = Module("SK5 mode extraction")
    module.add(SLShiftRightB32(dst=sgpr("WorkAssignmentMode"),
                               src=sgpr("MagicShiftItersPerTile"),
                               shiftHex=hex(30),
                               comment="SK5: shift mode bit (bit 30) down"))
    module.add(SAndB32(dst=sgpr("WorkAssignmentMode"),
                       src0=sgpr("WorkAssignmentMode"),
                       src1=hex(0x1),
                       comment="SK5: isolate mode bit -> WorkAssignmentMode"))
    # Clear MagicShift bit 30 by XOR with (HybridMode << 30). HybridMode
    # is the extracted mode, so the clear is RAW on extract. Restore
    # HybridMode to 0/1 afterward.
    module.add(SLShiftLeftB32(dst=sgpr("WorkAssignmentMode"),
                              src=sgpr("WorkAssignmentMode"),
                              shiftHex=hex(30),
                              comment="SK5: mode bit back to bit 30 for XOR-clear"))
    module.add(SXorB32(dst=sgpr("MagicShiftItersPerTile"),
                       src0=sgpr("MagicShiftItersPerTile"),
                       src1=sgpr("WorkAssignmentMode"),
                       comment="SK5: clear bit 30 via XOR of extracted mode"))
    module.add(SLShiftRightB32(dst=sgpr("WorkAssignmentMode"),
                               src=sgpr("WorkAssignmentMode"),
                               shiftHex=hex(30),
                               comment="SK5: restore HybridMode to 0/1"))
    return module


class WorkAssignment(Component):
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

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

    def registerRequirements(self, writer, kernel, unaligned, aligned):
        processing = Component.TileProcessingStrategy.find(writer)
        if kernel["WorkAssignment"] == "StaticGrid":
            partition = processing.staticPartition()
            if not writer.isPersistentConstantsToVgprEnabled(kernel) and partition.initial_rank != partition.cursor:
                unaligned.append(partition.initial_rank)
            unaligned.extend([partition.cursor, partition.bound])
            unaligned.extend(processing.persistentTileRegisters(kernel))
            if kernel["SpaceFillingAlgo"]:
                unaligned.append("PersistentTileID")
            if writer.isPrefetchAcrossPersistentEnabled(kernel):
                unaligned.append("PersistentPrefetchState")
            if kernel["ReuseAcrossPersistent"]:
                unaligned.append("RAPResidentBatch")
        else:
            partition = processing.queuePartition()
            unaligned.append(partition.launch_rank)
            unaligned.extend(processing.persistentTileRegisters(kernel))
            hybrid = kernel["WorkAssignment"] == "Hybrid"
            if hybrid:
                unaligned.append("WorkAssignmentMode")
                if kernel["WorkQueueStealing"]:
                    unaligned.append(partition.sticky_empty)
                if kernel["SpaceFillingAlgo"]:
                    unaligned.append("PersistentTileID")
            if writer.isPrefetchAcrossPersistentEnabled(kernel):
                unaligned.extend(["PersistentPrefetchState", "NextWorkItem"])
                writer.states.numSgprPersistent += 2
            if not hybrid and kernel["WorkQueueStealing"]:
                unaligned.append(partition.sticky_empty)
        aligned.extend(processing.persistentWorkspaceRegisters(kernel))

    def gridRegister(self, writer, kernel):
        processing = Component.TileProcessingStrategy.find(writer)
        return processing.staticPartition().grid if kernel["WorkAssignment"] == "StaticGrid" else processing.queuePartition().grid

    def activateStaticRange(self, writer, kernel, processing, tPA, tPB):
        module = Module("StreamK TwoTileDPFirst graWorkGroup")

        # StreamK workgroup mapping. This is short-lived scratch, so grow the pool
        # rather than reject the solution when no 4-register hole is free: MX TDM
        # kernels can be left without one while still far below MaxSgpr. Growth only
        # happens where the pinned checkout would have failed, so kernels that fit a
        # hole keep the same register assignment, and checkResources still rejects
        # anything that ends up over MaxSgpr.
        sTmp = writer.sgprPool.checkOutAligned(4, 2, "SKMappingTemp", preventOverflow=False)


        module.add(processing.skTileIndex(writer, kernel, sTmp, tPA, tPB))

        transition, sTmp = processing.nextStaticCursor(writer, kernel, sTmp)
        module.add(transition)
        module.add(SMovB32(dst=sgpr("PersistentIteration"), src=sgpr(sTmp+1), comment="Store current iteration"))

        module.add(processing.finishStaticTile(writer, kernel, tPA, tPB, sTmp))
        return module

    def activateHybridStaticRange(self, writer, kernel, processing, tPA, tPB):
        mod = Module("StreamK hybrid partition activation")

        sTmp = writer.sgprPool.checkOutAligned(4, 2, "SKMappingTemp")

        mod.add(processing.skTileIndex(writer, kernel, sTmp, tPA, tPB))

        transition, sTmp = processing.nextStaticCursor(writer, kernel, sTmp)
        mod.add(transition)
        mod.add(SMovB32(dst=sgpr("PersistentIteration"), src=sgpr(sTmp+1),
                           comment="Store current iteration"))

        mod.add(processing.finishStaticTile(writer, kernel, tPA, tPB, sTmp))
        return mod

    def fetchNextWorkItem(self, writer, kernel, sWorkItemIdx, sAddress) -> Module:
        """Atomic fetch-and-increment for the dynamic work-queue counter.

        Targets with scalar memory atomics use ``s_atomic_inc`` directly; targets
        without them (e.g. gfx12/gfx1250, where ``HasSAtomic`` is false) issue a
        returning vector atomic from lane 0 instead.
        """
        module = Module("fetchNextWorkItem")

        if writer.states.asmCaps["HasSAtomic"]:
            module.add(SAtomicInc(dst=sgpr(sWorkItemIdx), base=sgpr(sAddress, 2), soffset=0,
                                  smem=SMEMModifiers(glc=True), comment="Fetch next work item index"))
            module.add(SWaitCnt(kmcnt=0, comment="Wait for scalar memory op"))
            return module

        # No scalar memory atomic: issue the wrapping fetch-and-increment as a returning
        # vector atomic from lane 0 only. Use ``global_atomic_inc_u32`` in SADDR form
        # (scalar 64-bit base + per-lane 32-bit offset).
        vZeroOffset = writer.vgprPool.checkOut(1, "AtomicZeroOffset")
        vWrapValue  = writer.vgprPool.checkOut(1, "AtomicWrapValue")
        vFetchedIdx = writer.vgprPool.checkOut(1, "AtomicFetchedIdx")
        sSavedExec  = writer.sgprPool.checkOutAligned(writer.states.laneSGPRCount,
                                                      writer.states.laneSGPRCount,
                                                      "SavedExec")
        execMovInst = SMovB32 if kernel["WavefrontSize"] == 32 else SMovB64

        module.add(VMovB32(dst=vgpr(vZeroOffset), src=0,
                           comment="Zero per-lane offset; queue base stays in saddr"))
        module.add(self.preVolatileVmem(writer, comment="drain xnacks before dynamic queue atomic"))
        module.add(execMovInst(dst=sgpr(sSavedExec, writer.states.laneSGPRCount),
                               src=EXEC(), comment="save exec mask"))
        module.add(VCmpXEqU32(dst=EXEC(), src0=vgpr("Serial"), src1=0,
                              comment="lane 0 fetches next work item"))
        # Wrap VGPR is the atomic data operand; emit it immediately before
        # the increment so va_vdst covers VALU to atomic.
        module.add(VMovB32(dst=vgpr(vWrapValue), src=sgpr(sWorkItemIdx),
                           comment="Queue wrap threshold (atomic_inc src)"))
        module.add(GlobalAtomicIncU32Saddr(dst=vgpr(vFetchedIdx),
                                      vaddr=vgpr(vZeroOffset),
                                      data=vgpr(vWrapValue),
                                      saddr=sgpr(sAddress, 2),
                                      modifier=GLOBALModifiers(scope=CacheScope.SCOPE_DEV),
                                      comment="Fetch next work item index"))
        module.add(SWaitCnt(vlcnt=0, comment="Wait for VMEM atomic return (loadcnt; global needs no dscnt)"))
        module.add(VReadfirstlaneB32(dst=sgpr(sWorkItemIdx), src=vgpr(vFetchedIdx),
                                     comment="Read fetched work item index"))
        module.add(execMovInst(dst=EXEC(),
                               src=sgpr(sSavedExec, writer.states.laneSGPRCount),
                               comment="restore exec mask"))
        writer.sgprPool.checkIn(sSavedExec)
        writer.vgprPool.checkIn(vFetchedIdx)
        writer.vgprPool.checkIn(vWrapValue)
        writer.vgprPool.checkIn(vZeroOffset)
        return module


    def queueConstants(self, writer, kernel):
        """Return (numQueues, mask, log2Queues, cacheLineLog2) for this arch.

        ``numQueues`` = archCaps["NumXCD"] and the counter stride =
        archCaps["CacheLineBytes"]; both must be powers of two for the shift/AND
        queue masking and queue-address shift to be valid (asserted below).
        """
        numQueues = writer.states.archCaps["NumXCD"]
        assert numQueues > 0 and (numQueues & (numQueues - 1)) == 0, (
            "StreamK dynamic-queue fast masking requires a power-of-two queue count "
            "(got %d for ISA %s)" % (numQueues, tuple(kernel["ISA"][:2])))
        strideBytes = writer.states.archCaps["CacheLineBytes"]
        assert strideBytes > 0 and (strideBytes & (strideBytes - 1)) == 0, (
            "StreamK per-queue counter stride must be a power-of-two cache-line "
            "size (got %d for ISA %s)" % (strideBytes, tuple(kernel["ISA"][:2])))
        return numQueues, numQueues - 1, log2(numQueues), log2(strideBytes)


    def flagsBaseOffset(self, writer, kernel):
        """Byte offset where the partials/fixup ready flags begin.

        The flags region starts right after the per-queue counters, i.e. after
        ``numQueues * strideBytes`` bytes (8 * 128 = 1024 on gfx942/gfx950).
        """
        numQueues, _, _, cacheLineLog2 = self.queueConstants(writer, kernel)
        return numQueues << cacheLineLog2


    @staticmethod
    def usesRawQueueRank(writer, kernel):
        """True when the per-XCD queue index is taken from the raw pre-remap
        launch rank snapshotted into the reused, in-window-dead persistent
        ``StreamKTileIdx`` carrier (zero extra SGPR -- see the prologue snapshot
        in KernelWriterAssembly).

        The auto-reset wrap bound (tiles_q + W_q [+ W_p]) assumes each queue's
        home-workgroup count equals ``distribute(skGrid, q)`` -- i.e. that the
        set of workgroup ids mapped to queue q is ``{i in [0,skGrid) : i %%
        numQueues == q}``.  That holds only if the value feeding ``% numQueues``
        densely covers ``[0, skGrid)``.  ``PersistentWorkGroupIndex`` is the *remapped* id
        (wgmXCC CU-count remap and/or the PersistentXCCMapping chiplet remap), and
        neither remap is a ``% numQueues``-count-preserving permutation when the
        grid does not block evenly, so ``PersistentWorkGroupIndex %% numQueues`` skews the
        per-queue count away from W_q and the counter no longer wraps back to 0
        each launch.  Using the raw launch rank (a dense bijection onto
        ``[0, skGrid)`` == physical XCD rank) restores the invariant.

        Two disjoint remap regimes need the raw rank:
          * WorkGroupMappingXCC == -1 (dynamic auto-WGM) -- host picks
            WGMXCC = NUM_XCD > 1 and the wgmXCC remap skews the count.
          * PersistentXCCMapping != 0 with WorkGroupMappingXCC > 1 (SKXCC) -- the
            SKXCC chiplet remap (plus fixed WGMXCC > 1) skews the count.  SKXCC
            with WGMXCC == 1 is already count-preserving and stays on the cheap
            ``PersistentWorkGroupIndex %% numQueues`` else-branch; WGMXCC == -1 is mutually
            exclusive with SKXCC so the disjuncts never overlap.

        Fixed non-SKXCC WGMXCC == 1 needs no fix (PersistentWorkGroupIndex is already the raw
        rank).  On WorkGroupIdFromTTM targets (gfx12) PersistentWorkGroupIndex is re-read from
        the raw hardware id (ttmp9); single-queue arches (NumXCD <= 1) are
        trivially balanced. The writer prologue uses this same predicate."""
        return ((hasDynamicAssignment(kernel) or hasHybridAssignment(kernel))
                and writer.states.archCaps["NumXCD"] > 1
                and not writer.states.archCaps["WorkGroupIdFromTTM"]
                and (kernel["WorkGroupMappingXCC"] == -1
                     or (kernel["PersistentXCCMapping"] != 0
                         and kernel["WorkGroupMappingXCC"] > 1)))


    def emitQueueIndex(self, writer, kernel, sQueueIdx, wsLog2Queues) -> Module:
        """Compute the per-XCD work-queue index into ``sQueueIdx``.

        Zero-overhead accounting fix: the queue must come from the raw
        round-robin launch rank so its ``% numQueues`` count equals the
        ``distribute(skGrid, q)`` the auto-reset bound assumes (see
        ``usesRawQueueRank``).  On gfx9 that raw rank is snapshotted once, before
        wgmXCC / the SKXCC XCCMapping remap rewrites WorkGroup0, into the reused,
        in-window-dead persistent ``StreamKTileIdx`` carrier (KernelWriterAssembly
        prologue -- zero extra SGPR); here it is read back and reduced
        ``% numQueues``.  Otherwise (WGMXCC no-op, or gfx12) ``PersistentWorkGroupIndex``
        already holds the raw id, so fall back to ``PersistentWorkGroupIndex %% numQueues``.
        """
        partition = Component.TileProcessingStrategy.find(writer).queuePartition()
        module = Module("StreamK queue index")
        if self.usesRawQueueRank(writer, kernel):
            # The queue index is the RAW pre-wgmXCC launch WG rank modulo
            # numQueues. This raw rank densely covers [0, skGrid), so the number
            # of home workgroups mapped to queue q equals distribute(skGrid, q) =
            # W_q -- exactly the count the auto-reset wrap bound (tiles_q + W_q)
            # assumes -- and the atomic counter self-resets to 0 every launch.
            # (PersistentWorkGroupIndex is the wgmXCC CU-count-remapped id, whose % numQueues is
            # NOT count-preserving and skews the per-queue count.) Uniform for SK4
            # and SK5 -- the snapshot lives in the reused, in-window-dead
            # persistent StreamKTileIdx carrier (zero extra SGPR; see
            # KernelWriterAssembly prologue and usesRawQueueRank).
            _, numQueuesMask, _, _ = self.queueConstants(writer, kernel)
            module.add(SAndB32(dst=sgpr(sQueueIdx), src0=sgpr(partition.raw_rank), src1=hex(numQueuesMask),
                               comment="queue = rawWG %% numQueues (dense round-robin => home-WG count == distribute(skGrid,q))"))
        else:
            module.add(SLShiftRightB32(dst=sgpr(sQueueIdx), src=sgpr(partition.launch_rank), shiftHex=wsLog2Queues))
            module.add(SLShiftLeftB32(dst=sgpr(sQueueIdx), src=sgpr(sQueueIdx), shiftHex=wsLog2Queues))
            module.add(SSubU32(dst=sgpr(sQueueIdx), src0=sgpr(partition.launch_rank), src1=sgpr(sQueueIdx),
                               comment="Default queue index"))
        return module


    def emitStructuralCount(self, mod, mask, log2Queues, sDst, sTotal, sQueue, sTmp, comment):
        """Emit sDst = (sTotal >> log2Queues) + [sQueue < (sTotal & mask)].

        Reuses the shift/and(mask)/cmp/cselect idiom already used for
        tilesInQueue / workgroupsInQueue in graWorkGroup so the per-queue
        structural share (tiles or workgroups) can be recomputed for an
        arbitrary queue index. ``sTmp`` is a caller-owned scratch SGPR.
        """
        mod.add(SLShiftRightB32(dst=sgpr(sDst), src=sgpr(sTotal), shiftHex=log2Queues, comment=comment))
        mod.add(SAndB32(dst=sgpr(sTmp), src0=sgpr(sTotal), src1=mask, comment="Remainder"))
        mod.add(SCmpLtU32(src0=sgpr(sQueue), src1=sgpr(sTmp), comment="Queue gets a structural extra?"))
        mod.add(SCSelectB32(dst=sgpr(sTmp), src0=1, src1=0))
        mod.add(SAddU32(dst=sgpr(sDst), src0=sgpr(sDst), src1=sgpr(sTmp)))


    def foldHomeBound(self, writer, mod, kernel, sBound, sQueueIdx, sGrid):
        """Fold the predecessor's workgroup count into the home auto-reset bound.

        Adds the predecessor term W_p to the ``tiles_q + W_q - 1`` already in
        ``sBound``, giving the stealing bound ``tiles_q + W_q + W_p - 1`` (queue q
        also absorbs W_p increments from its one predecessor p = (q-1) & mask).
        Caller gates on kernel["WorkQueueStealing"] and passes the grid SGPR
        name ("skGrid" for SK4, "SKGrid" for SK5-dynamic); ``sQueueIdx`` is
        preserved. Exact only when W_q >= 1 whenever tiles_q >= 1 (skGrid >=
        numQueues); the Solution layer rejects debug overrides that break this.
        """
        _, mask, log2Queues, _ = self.queueConstants(writer, kernel)
        sPred = writer.sgprPool.checkOut(1, "wsPredQueue")
        sWp = writer.sgprPool.checkOut(1, "wsPredWorkgroups")
        sTmp = writer.sgprPool.checkOut(1, "wsPredTmp")
        # p = (q - 1) & mask  (wraps 0 -> numQueues-1 for unsigned subtract)
        mod.add(SSubU32(dst=sgpr(sPred), src0=sgpr(sQueueIdx), src1=1, comment="Predecessor queue (q-1)"))
        mod.add(SAndB32(dst=sgpr(sPred), src0=sgpr(sPred), src1=mask, comment="Wrap predecessor index"))
        # W_p = (skGrid >> log2) + [p < (skGrid & mask)]
        self.emitStructuralCount(mod, mask, log2Queues, sWp, sGrid, sPred, sTmp,
                                comment="Predecessor workgroups W_(q-1)")
        mod.add(SAddU32(dst=sgpr(sBound), src0=sgpr(sBound), src1=sgpr(sWp),
                        comment="Home auto-reset bound += predecessor workgroups (next-neighbor steal)"))
        writer.sgprPool.checkIn(sTmp)
        writer.sgprPool.checkIn(sWp)
        writer.sgprPool.checkIn(sPred)


    def stealFromNeighbor(self, writer, mod, kernel, sQueueIdx, sWorkItemIdx, sGrid, mkLabel):
        """Single-hop next-neighbor steal on the per-XCD queue topology.

        On entry sQueueIdx holds home queue q and sWorkItemIdx holds the home
        fetch result (both live). If the home fetch was valid (index <
        TotalItems) this is a no-op; otherwise one s_atomic_inc steals from the
        next neighbor s = (q+1) & mask and the global tile index is recomputed
        from s. A lost race leaves sWorkItemIdx >= TotalItems, so the downstream
        valid-index check turns this WG into a no-op. sQueueIdx is clobbered
        (advanced to s). Caller gates on kernel["WorkQueueStealing"] and passes
        the grid SGPR name ("skGrid" for SK4, "SKGrid" for SK5-dynamic). The
        steal atomic uses the stolen queue's bound ``tiles_s + W_s + W_q - 1``.
        """
        partition = Component.TileProcessingStrategy.find(writer).queuePartition()
        _, mask, log2Queues, cacheLineLog2 = self.queueConstants(writer, kernel)
        skFetchDone = mkLabel("SK_FetchDone")
        mod.add(SCmpLtU32(src0=sgpr(sWorkItemIdx), src1=sgpr(partition.total_items), comment="Home fetch valid?"))
        mod.add(SCBranchSCC1(labelName=skFetchDone.getLabelName(), comment="Valid work fetched; no steal"))

        # Build the steal auto-reset bound tiles_s + W_s + W_q - 1 into
        # sWorkItemIdx (dead here). W_q is the stealer's own workgroup count and
        # must be computed while sQueueIdx still holds q, before advancing to s.
        sTmp = writer.sgprPool.checkOut(1, "wsStealTmp")
        sWq = writer.sgprPool.checkOut(1, "wsStealerWorkgroups")
        self.emitStructuralCount(mod, mask, log2Queues, sWq, sGrid, sQueueIdx, sTmp,
                                comment="Stealer workgroups W_q")

        # Walk to the immediate next queue (wrap within the per-XCD queues, single-hop next-neighbor).
        mod.add(SAddU32(dst=sgpr(sQueueIdx), src0=sgpr(sQueueIdx), src1=1, comment="Next queue"))
        mod.add(SAndB32(dst=sgpr(sQueueIdx), src0=sgpr(sQueueIdx), src1=mask, comment="Wrap queue index"))

        # tiles_s into sWorkItemIdx, then += W_s and += W_q, then -1.
        self.emitStructuralCount(mod, mask, log2Queues, sWorkItemIdx, partition.total_items, sQueueIdx, sTmp,
                                comment="Stolen-queue tiles tiles_s")
        sWs = writer.sgprPool.checkOut(1, "wsStolenWorkgroups")
        self.emitStructuralCount(mod, mask, log2Queues, sWs, sGrid, sQueueIdx, sTmp,
                                comment="Stolen-queue workgroups W_s")
        mod.add(SAddU32(dst=sgpr(sWorkItemIdx), src0=sgpr(sWorkItemIdx), src1=sgpr(sWs), comment="tiles_s + W_s"))
        writer.sgprPool.checkIn(sWs)
        mod.add(SAddU32(dst=sgpr(sWorkItemIdx), src0=sgpr(sWorkItemIdx), src1=sgpr(sWq), comment="+ W_q (stealer)"))
        mod.add(SSubU32(dst=sgpr(sWorkItemIdx), src0=sgpr(sWorkItemIdx), src1=1, comment="Steal auto-reset bound"))
        writer.sgprPool.checkIn(sWq)
        writer.sgprPool.checkIn(sTmp)

        # One atomic on the neighbor's counter with the static self-reset bound.
        sAddress = writer.sgprPool.checkOutAligned(2, 2, "wsStealAddress")
        mod.add(SLShiftLeftB32(dst=sgpr(sAddress), src=sgpr(sQueueIdx), shiftHex=cacheLineLog2, comment="Stride queues to cache lines (stolen queue)"))
        mod.add(SAddU32(dst=sgpr(sAddress+0), src0=sgpr(sAddress+0), src1=sgpr(partition.address + "+0")))
        mod.add(SAddCU32(dst=sgpr(sAddress+1), src0=0, src1=sgpr(partition.address + "+1")))
        mod.add(SAtomicInc(dst=sgpr(sWorkItemIdx), base=sgpr(sAddress, 2), soffset=0, smem=SMEMModifiers(glc=True), comment="Fetch stolen work item index"))
        mod.add(SWaitCnt(kmcnt=0, comment="Wait for scalar memory op"))
        writer.sgprPool.checkIn(sAddress)
        # Recompute global tile index from the neighbor's queue.
        mod.add(SLShiftLeftB32(dst=sgpr(sWorkItemIdx), src=sgpr(sWorkItemIdx), shiftHex=log2Queues))
        mod.add(SAddU32(dst=sgpr(sWorkItemIdx), src0=sgpr(sWorkItemIdx), src1=sgpr(sQueueIdx)))
        mod.add(skFetchDone)


    def preVolatileVmem(self, writer, comment="") -> Module:
        """Drain in-flight VMEM (XNACK-replay) before a volatile/atomic VMEM op.

        Required on arches with `RequiresXCntForVolatileVMEM` or
        `EnableXnackReplay`. No-op elsewhere.
        """
        module = Module("Work queue pre-volatile VMEM drain")
        if writer.states.archCaps["RequiresXCntForVolatileVMEM"] or \
                writer.states.archCaps["EnableXnackReplay"]:
            module.add(SWaitXCnt(xcnt=0, comment=comment))
        return module


    def fetchAndBroadcast(self, writer, kernel, preventOverflow=True, uniqueLabels=False):
        """Pop the next work item from this WG's per-XCD queue and broadcast it.

        Wave 0 performs the stateful atomic-increment pop from the dynamic
        work-queue and shares the resulting *global* work-item index with all
        waves via LDS. Returns ``(module, sWorkItemIdx)``; the caller owns
        ``sWorkItemIdx`` and must check it back in.

        This is the exact fetch sequence that used to live inline at the top of
        ``graWorkGroup``; it is factored out unchanged so PAP can reuse it (pop
        once per tile) while keeping non-PAP SK4 codegen byte-identical.

        ``preventOverflow`` is forwarded to the scratch SGPR check-outs. The
        default (True) matches the historical graWorkGroup behavior, where the
        pool has free headroom so allocation never overflows (byte-identical).
        When PAP calls this inside the OptNLL window the pool is near its
        high-water mark, so callers pass preventOverflow=False to let the pool
        grow gracefully (and signal occupancy pressure) instead of hitting the
        preventOverflow guard. The flag does not change the register indices of
        allocations that already fit, so non-PAP output is unaffected.
        """
        module = Module("DynamicWorkQueue fetchAndBroadcast")
        partition = Component.TileProcessingStrategy.find(writer).queuePartition()

        # Local address for sharing work id
        vLocalAddress = writer.vgprPool.checkOut(1, "LocalAddress")
        # Only first wave reads next work item index. When PAP hoists this pop
        # into the NLL window the same helper is also emitted in graWorkGroup, so
        # PAP callers request a unique label to avoid a duplicate-symbol clash;
        # the graWorkGroup (non-PAP) path keeps the historical name.
        skSkipWorkItem = Label(writer.labels.getNameInc("SK_SkipWorkItem") if partition.unique_labels else writer.labels.getNameInc("SK_PAP_SkipWorkItem") if uniqueLabels else "SK_SkipWorkItem", "")
        _emitMailboxAddressAndWave0Skip(writer, module, vLocalAddress, skSkipWorkItem,
                                        preventOverflow=preventOverflow)

        # Per-arch dynamic-queue fast-mask constants: log2(numQueues) for the
        # PersistentWorkGroupIndex/queue divisions, log2(cache-line size) for the counter stride.
        _, _, wsLog2Queues, wsCacheLineLog2 = self.queueConstants(writer, kernel)

        # Default queue index
        sQueueIdx = writer.sgprPool.checkOut(1, "QueueIdx", preventOverflow=preventOverflow)
        module.add(self.emitQueueIndex(writer, kernel, sQueueIdx, wsLog2Queues))

        # Queue address
        sAddress = writer.sgprPool.checkOutAligned(2, 2, "Address", preventOverflow=preventOverflow)
        module.add(SLShiftLeftB32(dst=sgpr(sAddress), src=sgpr(sQueueIdx), shiftHex=wsCacheLineLog2, comment="Stride queues to different cache lines"))
        module.add(SAddU32(dst=sgpr(sAddress+0), src0=sgpr(sAddress+0), src1=sgpr(partition.address + "+0")))
        module.add(SAddCU32(dst=sgpr(sAddress+1), src0=0, src1=sgpr(partition.address + "+1")))

        # Tiles in queue
        sTilesInQueue = writer.sgprPool.checkOut(1, "tilesInQueue", preventOverflow=preventOverflow)
        module.add(SLShiftRightB32(dst=sgpr(sTilesInQueue), src=sgpr(partition.total_items), shiftHex=wsLog2Queues))
        sRemainder = writer.sgprPool.checkOut(1, "remainder tiles", preventOverflow=preventOverflow)
        module.add(SLShiftLeftB32(dst=sgpr(sRemainder), src=sgpr(sTilesInQueue), shiftHex=wsLog2Queues))
        module.add(SSubU32(dst=sgpr(sRemainder), src0=sgpr(partition.total_items), src1=sgpr(sRemainder), comment="Remainder tiles"))
        module.add(SCmpLtU32(src0=sgpr(sQueueIdx), src1=sgpr(sRemainder), comment="Check if queue gets an extra tile"))
        module.add(SCSelectB32(dst=sgpr(sRemainder), src0=1, src1=0))
        module.add(SAddU32(dst=sgpr(sTilesInQueue), src0=sgpr(sTilesInQueue), src1=sgpr(sRemainder)))
        writer.sgprPool.checkIn(sRemainder)

        # Workgroups in queue
        sWorkgroupsInQueue = writer.sgprPool.checkOut(1, "workgroupsInQueue", preventOverflow=preventOverflow)
        module.add(SLShiftRightB32(dst=sgpr(sWorkgroupsInQueue), src=sgpr(partition.grid), shiftHex=wsLog2Queues))
        sRemainder = writer.sgprPool.checkOut(1, "remainder workgroups", preventOverflow=preventOverflow)
        module.add(SLShiftLeftB32(dst=sgpr(sRemainder), src=sgpr(sWorkgroupsInQueue), shiftHex=wsLog2Queues))
        module.add(SSubU32(dst=sgpr(sRemainder), src0=sgpr(partition.grid), src1=sgpr(sRemainder), comment="Remainder workgroups"))
        module.add(SCmpLtU32(src0=sgpr(sQueueIdx), src1=sgpr(sRemainder), comment="Check if queue gets an extra tile"))
        module.add(SCSelectB32(dst=sgpr(sRemainder), src0=1, src1=0))
        module.add(SAddU32(dst=sgpr(sWorkgroupsInQueue), src0=sgpr(sWorkgroupsInQueue), src1=sgpr(sRemainder)))
        writer.sgprPool.checkIn(sRemainder)

        # Fetch next work item index
        sWorkItemIdx = writer.sgprPool.checkOut(1, "nextWorkItemIdx", preventOverflow=preventOverflow)
        module.add(SAddU32(dst=sgpr(sWorkItemIdx), src0=sgpr(sTilesInQueue), src1=sgpr(sWorkgroupsInQueue), comment="Queue reset"))
        module.add(SSubU32(dst=sgpr(sWorkItemIdx), src0=sgpr(sWorkItemIdx), src1=1))
        writer.sgprPool.checkIn(sTilesInQueue)
        writer.sgprPool.checkIn(sWorkgroupsInQueue)

        # Work stealing: fold the predecessor's workgroup count into the home
        # auto-reset bound so the counter still self-resets under next-neighbor stealing.
        if kernel["WorkQueueStealing"]:
            self.foldHomeBound(writer, module, kernel, sWorkItemIdx, sQueueIdx, partition.grid)

        # Work stealing: once this WG has seen its home queue empty (sticky), it
        # never touches the home counter again -- skip the home fetch and force
        # the steal path with an invalid sentinel index (>= TotalItems).
        if kernel["WorkQueueStealing"]:
            skStealOnly = Label(writer.labels.getNameInc("SK_StealOnly"), "")
            skHomeFetched = Label(writer.labels.getNameInc("SK_HomeFetched"), "")
            module.add(SCmpEQU32(src0=sgpr(partition.sticky_empty), src1=0, comment="Home not yet empty?"))
            module.add(SCBranchSCC0(labelName=skStealOnly.getLabelName(), comment="Sticky: skip home fetch, steal only"))

        # Fetch next work item
        module.add(self.fetchNextWorkItem(writer, kernel, sWorkItemIdx, sAddress))
        writer.sgprPool.checkIn(sAddress)

        # Convert to global work item index
        module.add(SLShiftLeftB32(dst=sgpr(sWorkItemIdx), src=sgpr(sWorkItemIdx), shiftHex=wsLog2Queues))
        module.add(SAddU32(dst=sgpr(sWorkItemIdx), src0=sgpr(sWorkItemIdx), src1=sgpr(sQueueIdx)))

        # Work stealing: latch the sticky-empty flag on the first empty home
        # fetch, then fall through to the steal (or, when already sticky, jump
        # straight to the steal with the sentinel index).
        if kernel["WorkQueueStealing"]:
            module.add(SCmpGeU32(src0=sgpr(sWorkItemIdx), src1=sgpr(partition.total_items), comment="Home fetch empty?"))
            module.add(SCSelectB32(dst=sgpr(partition.sticky_empty), src0=1, src1=0, comment="Latch sticky-empty on empty home"))
            module.add(SBranch(labelName=skHomeFetched.getLabelName(), comment="Home fetched; try one steal"))
            module.add(skStealOnly)
            module.add(SMovB32(dst=sgpr(sWorkItemIdx), src=sgpr(partition.total_items), comment="Sentinel index (>= TotalItems) forces steal"))
            module.add(skHomeFetched)
            self.stealFromNeighbor(writer, module, kernel, sQueueIdx, sWorkItemIdx, partition.grid, lambda base: Label(writer.labels.getNameInc(base), ""))
        writer.sgprPool.checkIn(sQueueIdx)

        # Share work item index with all waves
        vWaveWorkItemIdx = writer.vgprPool.checkOut(1, "WaveWorkItemIdx")
        module.add(VMovB32(dst=vgpr(vWaveWorkItemIdx), src=sgpr(sWorkItemIdx), comment="Move work item index to vgpr"))
        _emitWorkItemMailbox(writer, module, vLocalAddress, vWaveWorkItemIdx, skSkipWorkItem,
                             sWorkItemIdx=sWorkItemIdx)

        writer.vgprPool.checkIn(vLocalAddress)
        writer.vgprPool.checkIn(vWaveWorkItemIdx)

        return module, sWorkItemIdx


    def acquireQueueItem(self, writer, kernel):
        partition = Component.TileProcessingStrategy.find(writer).queuePartition()
        module = Module("DynamicWorkQueue activateReservedOrAcquire")
        # PrefetchAcrossPersistent (PAP): the dynamic work-queue pop is stateful
        # (an atomic increment that consumes a queue slot / termination token),
        # so it must happen exactly once per tile. When PAP is enabled, the
        # prior persistent iteration's NLL already popped this iteration's work
        # item (PersistentPrefetchState != 0) and stashed it in NextWorkItem; reuse
        # it here instead of popping again (which would double-consume). On the
        # first iteration (and whenever not primed) we pop normally.
        papEnabled = writer.isPrefetchAcrossPersistentEnabled(kernel)
        if papEnabled:
            skPapFetchDone = Label(writer.labels.getNameInc("SK_PAP_FetchDone"), "")
            skPapUsePrimed = Label(writer.labels.getNameInc("SK_PAP_UsePrimedWorkItem"), "")
            module.add(SCmpEQU32(src0=sgpr("PersistentPrefetchState"), src1=0, comment="PAP: was next work item already popped?"))
            module.add(SCBranchSCC0(labelName=skPapUsePrimed.getLabelName(), comment="primed: reuse stashed work item"))
            moduleFetch, sWorkItemIdx = self.fetchAndBroadcast(writer, kernel)
            module.add(moduleFetch)
            module.add(SBranch(labelName=skPapFetchDone.getLabelName(), comment="popped this iteration's work item"))
            module.add(skPapUsePrimed)
            module.add(SMovB32(dst=sgpr(sWorkItemIdx), src=sgpr("NextWorkItem"), comment="PAP: reuse work item popped by prior NLL"))
            module.add(skPapFetchDone)
        else:
            moduleFetch, sWorkItemIdx = self.fetchAndBroadcast(writer, kernel)
            module.add(moduleFetch)

        # Check if work item index is valid
        module.add(SCmpLtU32(src0=sgpr(sWorkItemIdx), src1=sgpr(partition.total_items), comment="Check if work item index is valid"))
        # If work item index is not valid, skip to end of kernel
        module.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))

        return module, sWorkItemIdx

    def activateFullTileRange(self, writer, kernel, processing, tPA, tPB):
        module = Module('Persistent TwoTileDPFirst graWorkGroup')
        constantsInVgprs = writer.isPersistentConstantsToVgprEnabled(kernel)
        sTmp = writer.sgprPool.checkOutAligned(4, 2, 'PersistentMappingTemp', preventOverflow=False)
        sIpt = writer.acquirePersistentConstSgpr(kernel, 'ItersPerTile')
        if constantsInVgprs:
            module.add(VReadfirstlaneB32(dst=sgpr(sIpt), src=vgpr(writer.states.persistentConstVgprs['ItersPerTile'])))
        module.add(processing.computeTotalTiles(writer, kernel, sTmp + 3))
        module.add(SMulI32(dst=sgpr(sTmp + 3), src0=sgpr(sTmp + 3), src1=sgpr(sIpt), comment='dpSectionSize = totalTiles * ItersPerTile'))
        module.add(SCmpLtU32(src0=sgpr('PersistentIteration'), src1=sgpr(sTmp + 3), comment="Make sure there's DP work to do"))
        module.add(writer.longBranchScc0(Label('KernelEnd', ''), posNeg=1))
        writer.releasePersistentConstSgpr(sIpt)
        module.add(processing.mapIterationToTile(writer, kernel, sTmp, tPA, tPB))
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
        module.add(processing.tileIndexToWorkGroup(writer, kernel, sTmp))
        writer.sgprPool.checkIn(sTmp)
        return module


class StaticGrid(WorkAssignment):
    kernel = {"WorkAssignment": "StaticGrid"}
    def initialize(self, writer, kernel, processing):
        module = Module("StaticGrid initialize")
        skConstsInVgprs = writer.isPersistentConstantsToVgprEnabled(kernel)
        partition = processing.staticPartition()

        xccMapping = Component.XCCMapping.find(writer)
        module.add(xccMapping(writer, kernel))

        # Skip the gfx12 ttmp reread under clustering: defineAndResources already left
        # the cluster-decoded rank in WorkGroup0/1/2, and rereading ttmp9 (cluster_x here)
        # would collide PersistentWorkGroupIndex across the cluster.
        if writer.states.archCaps["WorkGroupIdFromTTM"] and not clusterEnabled(kernel["ClusterDim"]):
            module.add(SMovB32(dst=sgpr("WorkGroup0"), src="ttmp9", comment="workaround"))
            module.add(SAndB32(dst=sgpr("WorkGroup1"), src0=hex(0xFFFF), src1="ttmp7", comment="workaround"))
            module.add(SLShiftRightB32(dst=sgpr("WorkGroup2"), shiftHex=hex(0x10), src="ttmp7", comment="workaround"))

        # No USO prologue: bit 29 is tested in place at each divergence site.

        # Cluster multicast: exit padded boundary-cluster peers here, before the
        # fold overwrites WorkGroup0 with the linear index and before the prologue
        # cluster-barrier arrive, so their WAVEDONE frees the -3 barrier slot for
        # the present peers. No-op unless this is the ForceDPOnly cluster path;
        # the two-tile cluster instead defers the prologue arrive to AFTER the
        # StreamK work-check (see below).
        module.add(self.persistentClusterPadEarlyExit(writer, kernel))

        # Cluster multicast: fold the 2-D (+batch) HW workgroup coords into the
        # linear DP tile index the DP decode expects. WorkGroup0 = global M-tile
        # (Cs B-peers M-adjacent), WorkGroup1 = global N-tile (Ck A-peers
        # N-adjacent), WorkGroup2 = batch, so (M-fastest, matching tileIndexToWorkGroup):
        #   PersistentWorkGroupIndex = WorkGroup2*(nWG0*nWG1) + WorkGroup1*nWG0 + WorkGroup0
        # written into WorkGroup0 so the save below copies the final index. A 1-D
        # [Cs, 1] cluster launches the same 2-D grid, so it folds identically.
        if persistentSpatialCluster(kernel):
            with writer.allocTmpSgpr(2, tag="ClusterDPFold") as tRes:
                t0 = tRes.idx
                t1 = tRes.idx + 1
                module.add(SMulI32(dst=sgpr(t0), src0=sgpr("WorkGroup1"), src1=sgpr("NumWorkGroups0"),
                                   comment="DP fold: WorkGroup1 * nWG0 (N-tile row)"))
                module.add(SMulI32(dst=sgpr(t1), src0=sgpr("NumWorkGroups0"), src1=sgpr("NumWorkGroups1"),
                                   comment="DP fold: nWG0 * nWG1 (tiles per batch)"))
                module.add(SMulI32(dst=sgpr(t1), src0=sgpr(t1), src1=sgpr("WorkGroup2"),
                                   comment="DP fold: batch * (nWG0*nWG1)"))
                module.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(t0),
                                   comment="DP fold: + WorkGroup1*nWG0"))
                module.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(t1),
                                   comment="DP fold: PersistentWorkGroupIndex = batch*(nWG0*nWG1) + N*nWG0 + M"))

        if skConstsInVgprs:
            module.add(VMovB32(dst=vgpr(writer.states.persistentConstVgprs["PersistentWorkGroupIndex"]), src=sgpr("WorkGroup0"),
                               comment="Save original StreamK index to VGPR"))
        else:
            module.add(SMovB32(dst=sgpr(partition.initial_rank), src=sgpr("WorkGroup0"),
                               comment="Save mapped persistent rank"))

        # Cluster multicast: arrive once per workgroup at the cluster split barrier
        # here in the prologue, before the first tensor_load_to_lds, so it pairs
        # the cluster-barrier pass's first-load wait.
        #
        # EXCEPTION -- the two-tile (StreamKForceDPOnly==0) cluster: its no-work
        # peers only reveal themselves at the StreamK work-check below, so arriving
        # here would over-count the -3 barrier. DEFER that arrive to just after the
        # work-check. The ForceDPOnly cluster has already dropped its no-work peers
        # in persistentClusterPadEarlyExit above, so it arrives here.
        if persistentSpatialCluster(kernel):
            module.add(self.persistentMulticastPrologueSignal(writer, kernel))


        module.add(processing.initializePartition(writer, kernel))
        return module

    def activateReservedOrAcquire(self, writer, kernel, processing, tPA, tPB):
        writer.states.currentTileWork = processing.tileWork(kernel)
        if writer.states.currentTileWork.k_start is None:
            return self.activateFullTileRange(writer, kernel, processing, tPA, tPB)
        return self.activateStaticRange(writer, kernel, processing, tPA, tPB)

    def __call__(self):
        raise NotImplementedError

    def reserveNext(self, writer, kernel, skipLabel):
        module = Module("StaticGrid reserveNext")
        partition = Component.TileProcessingStrategy.find(writer).staticPartition()
        module.add(SCmpGeU32(src0=sgpr(partition.cursor), src1=sgpr(partition.bound), comment="No next persistent iteration"))
        module.add(SCBranchSCC1(labelName=skipLabel.getLabelName(), comment=""))
        return module

    def peekTileBatch(self, writer, kernel, dstSgpr):
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

    def closeLoop(self, writer, kernel):
        module = Module("StaticGrid closeLoop")
        partition = Component.TileProcessingStrategy.find(writer).staticPartition()
        module.add(SCmpGeU32(src0=sgpr(partition.cursor), src1=sgpr(partition.bound), comment="Check whether assigned work is exhausted"))
        module.add(writer.longBranchScc0(Label(writer.rapPersistentLoopEntryLabel(kernel), ""), posNeg=-1))
        return module


class DynamicWorkQueue(WorkAssignment):
    kernel = {"WorkAssignment": "DynamicWorkQueue"}
    def initialize(self, writer, kernel, processing):
        partition = Component.TileProcessingStrategy.find(writer).queuePartition()
        module = Module("StreamK Dynamic openLoop")

        xccMapping = Component.XCCMapping.find(writer)
        module.add(xccMapping(writer, kernel))

        # Skip the gfx12 ttmp reread under clustering: defineAndResources already left
        # the cluster-decoded rank in WorkGroup0/1/2, and rereading ttmp9 (cluster_x here)
        # would collide PersistentWorkGroupIndex across the cluster.
        if writer.states.archCaps["WorkGroupIdFromTTM"] and not clusterEnabled(kernel["ClusterDim"]):
            module.add(SMovB32(dst=sgpr("WorkGroup0"), src="ttmp9", comment="workaround"))
            module.add(SAndB32(dst=sgpr("WorkGroup1"), src0=hex(0xFFFF), src1="ttmp7", comment="workaround"))
            module.add(SLShiftRightB32(dst=sgpr("WorkGroup2"), shiftHex=hex(0x10), src="ttmp7", comment="workaround"))

        module.add(SMovB32(dst=sgpr(partition.launch_rank), src=sgpr("WorkGroup0"), comment="Save original StreamK index"))
        # Work stealing: this WG has not yet seen its home queue empty.
        if kernel["WorkQueueStealing"]:
            module.add(SMovB32(dst=sgpr(partition.sticky_empty), src=0, comment="WS: home not yet empty"))
        # Two-tile SK (DP first)
        # Do DP tiles before SK
        skInitDone = Label("SK_InitDone", "")
        module.add(skInitDone)

        return module

    def activateReservedOrAcquire(self, writer, kernel, processing, tPA, tPB):
        module, item = self.acquireQueueItem(writer, kernel)
        writer.states.currentTileWork = processing.tileWork(kernel)
        module.add(processing.activateWorkItem(writer, kernel, tPA, tPB, item))
        return module

    def __call__(self):
        raise NotImplementedError

    def reserveNext(self, writer, kernel, skipLabel):
        """SK4 PAP back-edge predicate: pop the next work item once, up front.

        Unlike static StreamK (which can predict the next iteration from
        PersistentIteration/PersistentIterationEnd), SK4's next tile comes from a stateful
        work-queue pop and cannot be predicted without consuming a slot. We
        therefore pop it here (inside the NLL PAP window), stash it in
        NextWorkItem for the persistent back-edge's graWorkGroup to reuse,
        and mark PersistentPrefetchState so that back-edge never pops again (even on
        the draining iteration -- avoiding a double-consume of a termination
        token). When the pop drains the queue (index >= TotalItems) we skip the
        rest of PAP; the back-edge graWorkGroup then exits via KernelEnd.
        """
        partition = Component.TileProcessingStrategy.find(writer).queuePartition()
        module = Module("StreamK Dynamic papHasNextPersistentIteration")
        # The OptNLL PAP window runs near the SGPR high-water mark, so let the
        # pop's scratch check-outs grow the pool (preventOverflow=False) instead
        # of tripping the preventOverflow guard.
        moduleFetch, sWorkItemIdx = self.fetchAndBroadcast(writer, kernel, preventOverflow=False, uniqueLabels=True)
        module.add(moduleFetch)
        module.add(SMovB32(dst=sgpr("NextWorkItem"), src=sgpr(sWorkItemIdx), comment="PAP: stash popped work item for persistent back-edge"))
        # Mark primed BEFORE the drain check so the back-edge reuses the stashed
        # work item (never re-pops) even when this pop drained the queue.
        module.add(SMovB32(dst=sgpr("PersistentPrefetchState"), src=1, comment="PAP: next work item already popped"))
        writer.sgprPool.checkIn(sWorkItemIdx)
        module.add(SCmpGeU32(src0=sgpr("NextWorkItem"), src1=sgpr(partition.total_items), comment="PAP: queue drained (no next tile)?"))
        module.add(SCBranchSCC1(labelName=skipLabel.getLabelName(), comment="drained: skip next-tile prefetch"))
        return module

    def closeLoop(self, writer, kernel):
        module = Module("DynamicWorkQueue closeLoop")
        module.add(SBarrier(comment="Sync before work-queue persistent re-entry"))
        with writer.allocTmpSgpr(3, tag="DynamicWorkQueue_closeLoop") as tmp:
            module.add(SLongBranchNegative(Label("PersistentLoopStart", ""), tmp))
        return module


class Hybrid(WorkAssignment):
    kernel = {"WorkAssignment": "Hybrid"}
    def initialize(self, writer, kernel, processing):
        partition = Component.TileProcessingStrategy.find(writer).queuePartition()
        module = Module("StreamK Hybrid openLoop")

        # ----- Common prologue: XCC mapping, gfx12 workaround, save WG0 -----
        xccMapping = Component.XCCMapping.find(writer)
        module.add(xccMapping(writer, kernel))

        # Skip the gfx12 ttmp reread under clustering: defineAndResources already left
        # the cluster-decoded rank in WorkGroup0/1/2, and rereading ttmp9 (cluster_x here)
        # would collide PersistentWorkGroupIndex across the cluster.
        if writer.states.archCaps["WorkGroupIdFromTTM"] and not clusterEnabled(kernel["ClusterDim"]):
            module.add(SMovB32(dst=sgpr("WorkGroup0"), src="ttmp9", comment="workaround"))
            module.add(SAndB32(dst=sgpr("WorkGroup1"), src0=hex(0xFFFF), src1="ttmp7", comment="workaround"))
            module.add(SLShiftRightB32(dst=sgpr("WorkGroup2"), shiftHex=hex(0x10), src="ttmp7", comment="workaround"))

        # SK5 always has isPersistentConstantsToVgprEnabled(kernel) == False,
        # so save directly to the PersistentWorkGroupIndex SGPR (no VGPR-cache path).
        module.add(SMovB32(dst=sgpr(partition.launch_rank), src=sgpr("WorkGroup0"),
                           comment="SK5: save original StreamK index"))
        # Work stealing: this WG has not yet seen its home queue empty.
        if kernel["WorkQueueStealing"]:
            module.add(SMovB32(dst=sgpr(partition.sticky_empty), src=0,
                               comment="WS: home not yet empty"))

        # ----- Extract the mode bit once for the whole kernel -----
        module.add(Component.WorkAssignment.find(writer).extractMode(writer, kernel))

        # No USO prologue. Bit 30 must still be extracted and cleared above:
        # WorkAssignmentMode's dispatch is a plain SCmpEQU32==0 and the SKTiles
        # alias needs a clean tile count. Bit 29 needs neither.

        module.add(processing.initializePartition(writer, kernel))
        return module

    def activateReservedOrAcquire(self, writer, kernel, processing, tPA, tPB):
        module = Module("Hybrid activateReservedOrAcquire")
        writer.states.currentTileWork = processing.tileWork(kernel)
        def dynamic(mod):
            acquired, item = self.acquireQueueItem(writer, kernel)
            mod.add(acquired)
            mod.add(processing.activateWorkItem(writer, kernel, tPA, tPB, item))
        self.dispatch(writer, module, "GRA", dynamic,
                      lambda mod: mod.add(self.activateHybridStaticRange(writer, kernel, processing, tPA, tPB)))
        return module

    def __call__(self):
        raise NotImplementedError

    def extractMode(self, writer, kernel):
        return _extract_hybrid_mode()

    def dispatch(self, writer, module, tag, emitDynamic, emitStatic):
        """Emit mode-gated dual path: dynamic (SK4) first, static (SK3) second."""
        sk5Static = Label(writer.labels.getNameInc(f"SK5_Static{tag}"), "")
        sk5Done   = Label(writer.labels.getNameInc(f"SK5_{tag}Done"), "")

        module.add(SCmpEQU32(src0=sgpr("WorkAssignmentMode"), src1=0,
                             comment=f"SK5: mode bit == 0 -> SK3 (static) {tag}"))
        module.add(SCBranchSCC1(labelName=sk5Static.getLabelName(),
                                comment=f"SK5: branch to static {tag}"))

        module.addComment2(f"SK5 dynamic (SK4) {tag}")
        emitDynamic(module)

        module.add(SBranch(labelName=sk5Done.getLabelName(),
                           comment=f"SK5: skip static {tag}"))

        module.add(sk5Static)
        module.addComment2(f"SK5 static (SK3) {tag}")
        emitStatic(module)

        module.add(sk5Done)

    def reserveNext(self, writer, kernel, skipLabel):
        module = Module("Hybrid reserveNext")
        self.dispatch(writer, module, "PapHasNext",
                      lambda mod: mod.add(DynamicWorkQueue.reserveNext(self, writer, kernel, skipLabel)),
                      lambda mod: mod.add(StaticGrid().reserveNext(writer, kernel, skipLabel)))
        return module

    def closeLoop(self, writer, kernel):
        module = Module("Hybrid closeLoop")
        dynamic = Label("Hybrid_DynamicClose", "")
        done = Label("Hybrid_CloseDone", "")
        module.add(SCmpEQU32(src0=sgpr("WorkAssignmentMode"), src1=0, comment="Hybrid: static assignment selected"))
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
            grid = Component.WorkAssignment.find(writer).gridRegister(writer, kernel)
            sGrid = writer.acquirePersistentConstSgpr(kernel, grid)
            if writer.isPersistentConstantsToVgprEnabled(kernel):
                module.add(VReadfirstlaneB32(dst=sgpr(sGrid), src=vgpr(writer.states.persistentConstVgprs[grid])))
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
