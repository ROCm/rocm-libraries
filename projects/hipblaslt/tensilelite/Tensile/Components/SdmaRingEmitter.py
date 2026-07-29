# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# SDMA ring-buffer producer emitter (ROCM-27524, SDMA offload codegen Task 4).
#
# Packet-INDEPENDENT rocisa translation of MORI's anvil device ring skeleton
# (mori/include/mori/core/transport/sdma/anvil_device.hpp:121-234, the
# WrapIntoRing / CanWriteUpto / ReserveQueueSpace / placePacket / submitPacket
# five). It generates the assembly a GPU producer runs to reserve space in a
# host-created SDMA ring, place already-built packet dwords, and ring the
# doorbell -- WITHOUT knowing what the packet is. Task 5 builds the COPY_SUBWIN
# + ATOMIC packet dwords and calls placePacket; Tasks 6/7 wire the whole thing
# into the GEMM epilogue store path. This round emits nothing into a live
# kernel; it is verified by rendering each method's Module to assembly text and
# asserting on the instruction sequence + scope bits
# (Tensile/Tests/unit/test_sdma_ring_emitter.py).
#
# The device handle it consumes is the W-element SdmaQueueDeviceHandle array
# produced by Task 2 (client/src/SdmaQueue.hpp) and passed in via the
# FusedSdmaQueues kernarg (Task 3, intra-segment offset 160). The 7x8-byte
# field layout below is the cross-task byte contract locked by the
# static_asserts in that header; do not reorder.
#
# gfx950 (CDNA4) scope -> instruction-bit mapping (SC[1:0]+NT, NOT gfx1250
# scope:/th:). Verified three ways: rocm-ref cache-policies / multi-gpu-
# communication (SC1=1 => device, SC1=SC0=1 => system), the in-repo fused-A2A
# handshake (GlobalWriteBatch._emitFusedA2AHandshake), and a live host render
# (gfx950 asmCaps HasSC0Modifier=1). Encoded through GLOBALModifiers(glc,slc):
#   AGENT  (ring/wptr/cachedWptr/committedWptr): glc=False slc=True  -> "sc1"
#   SYSTEM (doorbell store, rptr load)         : glc=True  slc=True  -> "sc0 sc1"
#   CAS    (cachedWptr reserve, device+return) : glc=True  slc=False -> "sc0"
# The A2A-local stores also carry sc1 to bypass L2 (Global Constraint 1: gfx950
# L2 is XCD-local only), which the AGENT mapping already provides.
################################################################################

from rocisa.container import vgpr, sgpr, VCC, GLOBALModifiers
from rocisa.code import Module, Label, TextBlock
from rocisa.instruction import (
    SMovB32, VMovB32,
    SAddU32, SAddCU32, SSubU32, SSubBU32,
    SAndB32, SLShiftRightB32,
    SCmpEQU32, SCmpLtU32,
    SCBranchSCC0, SCBranchSCC1, SBranch,
    SWaitCnt, SSleep,
    VReadfirstlaneB32, VAddCOU32, VAddCCOU32,
    GlobalStoreB32, GlobalStoreB64, GlobalLoadB64,
    GlobalAtomicCmpswapB64,
)


# 256 KB SDMA ring, matching client/src/SdmaQueue.hpp SDMA_QUEUE_SIZE and MORI's
# SDMA_QUEUE_SIZE. Power of two => WrapIntoRing is an AND mask, never a divide.
SDMA_QUEUE_SIZE = 256 * 1024
assert (SDMA_QUEUE_SIZE & (SDMA_QUEUE_SIZE - 1)) == 0, "ring size must be a power of two"

# Byte offsets of every SdmaQueueDeviceHandle field (contract: the static_asserts
# in client/src/SdmaQueue.hpp lock these). Pointers are 8 bytes; the last field
# is a VALUE seed, not a pointer.
OFF_queueBuf         = 0    # ring base (uint32_t*, dword-addressed)
OFF_rptr             = 8    # hardware read pointer  (SYSTEM-scope read)
OFF_wptr             = 16   # hardware write pointer (AGENT-scope write)
OFF_doorbell         = 24   # doorbell               (SYSTEM-scope write)
OFF_cachedWptr       = 32   # producer reservation cursor (AGENT-scope CAS)
OFF_committedWptr    = 40   # commit-serialization cursor (AGENT-scope)
OFF_cachedHwReadIndex = 48  # per-producer private cache SEED (value, never stored back)


class SdmaRingEmitter:
    """Packet-independent SDMA ring producer, emitted as rocisa Modules.

    One instance is stateless; every method takes the registers it operates on
    (the caller -- Task 6/7's KernelWriterAssembly -- owns the pools) plus a
    `w` context exposing `.sgprPool` / `.vgprPool` / `.labels`, mirroring the
    GL2PrefetchLoad component. Persistent per-producer state lives in caller-
    owned SGPRs:
      * handleBase (2 SGPRs): pointer to this peer's SdmaQueueDeviceHandle.
      * cachedHwReadIdx (2 SGPRs): the private CanWriteUpto cache. The caller
        seeds it ONCE from handle+48 at setup; this emitter reads and refreshes
        it in-register and NEVER stores it back to memory (treating it as shared
        state would add a needless, hard-to-find race -- see the long note in
        client/src/SdmaQueue.hpp).

    Field pointers (queueBuf/rptr/wptr/doorbell/cachedWptr/committedWptr) are
    loaded on demand with s_load_dwordx2 from handleBase+offset, matching the
    on-demand kernarg loads in _fusedA2ALoadRecvBase.
    """

    def __init__(self, queueSize: int = SDMA_QUEUE_SIZE):
        assert (queueSize & (queueSize - 1)) == 0, "ring size must be a power of two"
        self.queueSize = queueSize
        self.ringMask  = queueSize - 1

    # ---- small helpers -----------------------------------------------------

    def _loadFieldPtr(self, module, w, dstPairS, handleBaseS, byteOff):
        """Load an 8-byte handle field (pointer) at handleBase+byteOff into an
        aligned SGPR pair via s_load_dwordx2. Used for queueBuf/rptr/wptr/
        doorbell/cachedWptr/committedWptr. handleBase is a raw SGPR pointer pair
        and byteOff is a compile-time immediate, so this is a plain s_load with
        no argLoader dependency (keeps the emitter reusable outside a full
        KernelWriter). Caller waits (kmcnt) before use."""
        module.add(TextBlock("  s_load_dwordx2 s[%d:%d], s[%d:%d], 0x%x\n"
                             % (dstPairS, dstPairS + 1, handleBaseS, handleBaseS + 1, byteOff)))
        return module

    def _wrapIntoRing(self, module, dstS, srcS, comment=""):
        """dst = src & (queueSize-1). WrapIntoRing without a divide (ring size is
        a power of two). src/dst are the low dword of a byte index; the ring is
        <4 GiB so the high dword of a wrapped index is always 0."""
        module.add(SAndB32(dst=sgpr(dstS), src0=sgpr(srcS), src1=self.ringMask,
                           comment=comment or "WrapIntoRing: idx & (SDMA_QUEUE_SIZE-1)"))
        return module

    def _ptrToVgpr(self, module, vPairV, sPairS, comment=""):
        """Copy a 64-bit pointer from an SGPR pair to a VGPR pair (global_* on
        gfx950 take the address in VGPRs with a null saddr, per the fused-A2A
        idiom)."""
        module.add(VMovB32(dst=vgpr(vPairV + 0), src=sgpr(sPairS + 0), comment=comment + " lo"))
        module.add(VMovB32(dst=vgpr(vPairV + 1), src=sgpr(sPairS + 1), comment=comment + " hi"))
        return module

    # ---- CanWriteUpto ------------------------------------------------------

    def emitCanWriteUpto(self, module, w, handleBaseS, cachedHwReadIdxS,
                         uptoIdxS, resultS, tmpPairS):
        """MORI CanWriteUpto (anvil_device.hpp:126-135), two-level full check.

        Fast path uses the private cache only (no memory traffic):
            if (upto - cachedHwReadIndex) < queueSize: return true
        Slow path (cache says full) reads the hardware rptr at SYSTEM scope,
        refreshes the cache, and re-tests. Emits the SYSTEM-scope rptr load
        (glc/slc => sc0 sc1). `resultS` is set to 1 (can write) or 0 (full);
        the caller branches on it. All index math is 64-bit (idx pair = S:S+1).
        """
        canLabel  = Label(w.labels.getNameInc("sdma_canwrite_ok"),  "CanWriteUpto: room in ring")
        fullLabel = Label(w.labels.getNameInc("sdma_canwrite_full"), "CanWriteUpto: cache says full -> read rptr")
        doneLabel = Label(w.labels.getNameInc("sdma_canwrite_done"), "CanWriteUpto: done")

        # tmp = upto - cachedHwReadIndex (64-bit), then compare tmp < queueSize.
        # queueSize < 2^32 so if the high dword of the difference is nonzero the
        # gap is huge (>= 2^32) => definitely not < queueSize => full.
        self._emitU64Sub(module, tmpPairS, uptoIdxS, cachedHwReadIdxS,
                         "CanWriteUpto: upto - cachedHwReadIndex")
        module.add(SCmpEQU32(src0=sgpr(tmpPairS + 1), src1=0, comment="diff hi == 0? (gap < 2^32)"))
        module.add(SCBranchSCC0(labelName=fullLabel.getLabelName(), comment="hi != 0 -> gap huge -> full path"))
        module.add(SCmpLtU32(src0=sgpr(tmpPairS + 0), src1=self.queueSize,
                             comment="diff < queueSize? (fast-path room check)"))
        module.add(SCBranchSCC1(labelName=canLabel.getLabelName(), comment="room via cached index"))

        # Slow path: SYSTEM-scope read of the hardware rptr, refresh cache, retest.
        module.add(fullLabel)
        rptrPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_cw_rptrPtr", preventOverflow=False)
        self._loadFieldPtr(module, w, rptrPtrS, handleBaseS, OFF_rptr)
        module.add(SWaitCnt(kmcnt=0, comment="wait rptr pointer load"))
        vRptrAddr = w.vgprPool.checkOutAligned(2, 2, tag="sdma_cw_rptrAddr")
        vRptrVal  = w.vgprPool.checkOutAligned(2, 2, tag="sdma_cw_rptrVal")
        self._ptrToVgpr(module, vRptrAddr, rptrPtrS, "rptr addr")
        off = vgpr("off", 1, False, False, True)
        module.add(GlobalLoadB64(
            dst=vgpr(vRptrVal, 2), vaddr=vgpr(vRptrAddr, 2), saddr=off,
            modifier=GLOBALModifiers(glc=True, slc=True),
            comment="load hardware rptr (SYSTEM scope, sc0 sc1)"))
        module.add(SWaitCnt(vlcnt=0, comment="wait rptr load"))
        # refresh cachedHwReadIndex = rptr (into the caller's private SGPR pair).
        module.add(VReadfirstlaneB32(dst=sgpr(cachedHwReadIdxS + 0), src=vgpr(vRptrVal + 0),
                                     comment="cachedHwReadIndex lo = rptr"))
        module.add(VReadfirstlaneB32(dst=sgpr(cachedHwReadIdxS + 1), src=vgpr(vRptrVal + 1),
                                     comment="cachedHwReadIndex hi = rptr"))
        w.vgprPool.checkIn(vRptrAddr)
        w.vgprPool.checkIn(vRptrVal)
        w.sgprPool.checkIn(rptrPtrS)
        # retest with refreshed cache.
        self._emitU64Sub(module, tmpPairS, uptoIdxS, cachedHwReadIdxS,
                         "CanWriteUpto: upto - refreshed rptr")
        module.add(SCmpEQU32(src0=sgpr(tmpPairS + 1), src1=0, comment="diff hi == 0?"))
        module.add(SCBranchSCC0(labelName=doneLabel.getLabelName(), comment="hi != 0 -> full (result stays 0)"))
        module.add(SCmpLtU32(src0=sgpr(tmpPairS + 0), src1=self.queueSize, comment="diff < queueSize?"))
        module.add(SCBranchSCC1(labelName=canLabel.getLabelName(), comment="room after refresh"))
        # fall through to done with result=0 set below.
        module.add(SMovB32(dst=sgpr(resultS), src=0, comment="CanWriteUpto = false (full)"))
        module.add(SBranch(labelName=doneLabel.getLabelName(), comment="-> done"))
        module.add(canLabel)
        module.add(SMovB32(dst=sgpr(resultS), src=1, comment="CanWriteUpto = true (room)"))
        module.add(doneLabel)
        return module

    # ---- ReserveQueueSpace (CAS, NOT fetch_add) ----------------------------

    def emitReserveQueueSpace(self, module, w, handleBaseS, cachedHwReadIdxS,
                              sizeInBytes, outCurS, outOffsetS):
        """MORI ReserveQueueSpace (anvil_device.hpp:137-169): reserve `sizeInBytes`
        in the ring via a compare-exchange loop and compute the wrap-padding.

        MUST be CAS, not fetch_add: on wrap the reservation also pads the ring
        tail (offset = queueSize - WrapIntoRing(cur)), and that padding depends
        on the CURRENT cur_index -- so "compute new index" and "claim the slot"
        must be one atomic step. A fetch_add would let two producers compute
        different padding yet both believe they claimed the slot.

        Per iteration:
          cur   = load cachedWptr (AGENT)
          off   = (WrapIntoRing(cur) + size > queueSize) ? queueSize-WrapIntoRing(cur) : 0
          new   = cur + size + off
          if CanWriteUpto(new) and CAS(cachedWptr, cur -> new) succeeds: break
        Outputs: outCurS (2 SGPRs) = reserved base index; outOffsetS (1 SGPR) =
        pad bytes. sizeInBytes is a compile-time packet size (immediate).
        """
        loopLabel  = Label(w.labels.getNameInc("sdma_reserve_loop"), "ReserveQueueSpace: CAS retry loop")
        noPadLabel = Label(w.labels.getNameInc("sdma_reserve_nopad"), "ReserveQueueSpace: no wrap padding")
        doneLabel  = Label(w.labels.getNameInc("sdma_reserve_done"),  "ReserveQueueSpace: reserved")

        cachedWptrPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_rsv_cwPtr", preventOverflow=False)
        self._loadFieldPtr(module, w, cachedWptrPtrS, handleBaseS, OFF_cachedWptr)
        module.add(SWaitCnt(kmcnt=0, comment="wait cachedWptr pointer load"))

        newIdxS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_rsv_new", preventOverflow=False)
        wrapS   = w.sgprPool.checkOut(1, tag="sdma_rsv_wrap", preventOverflow=False)
        canS    = w.sgprPool.checkOut(1, tag="sdma_rsv_can", preventOverflow=False)
        tmpPair = w.sgprPool.checkOutAligned(2, 2, tag="sdma_rsv_tmp", preventOverflow=False)

        # VGPRs for the CAS: address + a 4-dword data reg [0:1]=swap(new), [2:3]=compare(cur).
        vCasAddr = w.vgprPool.checkOutAligned(2, 2, tag="sdma_rsv_casAddr")
        vCasData = w.vgprPool.checkOutAligned(4, 4, tag="sdma_rsv_casData")

        module.add(loopLabel)
        # cur = load cachedWptr (AGENT scope).
        self._ptrToVgpr(module, vCasAddr, cachedWptrPtrS, "cachedWptr addr")
        off = vgpr("off", 1, False, False, True)
        module.add(GlobalLoadB64(
            dst=vgpr(vCasData + 2, 2), vaddr=vgpr(vCasAddr, 2), saddr=off,
            modifier=GLOBALModifiers(glc=False, slc=True),
            comment="cur = load cachedWptr (AGENT scope, sc1)"))
        module.add(SWaitCnt(vlcnt=0, comment="wait cachedWptr load"))
        module.add(self._vReadfirstlane(outCurS + 0, vCasData + 2, "cur lo -> sgpr"))
        module.add(self._vReadfirstlane(outCurS + 1, vCasData + 3, "cur hi -> sgpr"))

        # off = 0 by default; if WrapIntoRing(cur)+size > queueSize -> pad tail.
        module.add(SMovB32(dst=sgpr(outOffsetS), src=0, comment="offset = 0 (no pad)"))
        self._wrapIntoRing(module, wrapS, outCurS + 0, "WrapIntoRing(cur)")
        module.add(SAddU32(dst=sgpr(tmpPair), src0=sgpr(wrapS), src1=sizeInBytes,
                           comment="WrapIntoRing(cur) + size"))
        module.add(SCmpLtU32(src0=sgpr(tmpPair), src1=self.queueSize + 1,
                             comment="wrap+size <= queueSize? (fits without wrap)"))
        module.add(SCBranchSCC1(labelName=noPadLabel.getLabelName(), comment="fits -> no padding"))
        module.add(SSubU32(dst=sgpr(outOffsetS), src0=self.queueSize, src1=sgpr(wrapS),
                           comment="offset = queueSize - WrapIntoRing(cur) (pad ring tail)"))
        module.add(noPadLabel)

        # new = cur + size + offset (64-bit).
        module.add(SAddU32(dst=sgpr(newIdxS + 0), src0=sgpr(outCurS + 0), src1=sizeInBytes,
                           comment="new lo = cur + size"))
        module.add(SAddCU32(dst=sgpr(newIdxS + 1), src0=sgpr(outCurS + 1), src1=0, comment="new hi (carry)"))
        module.add(SAddU32(dst=sgpr(newIdxS + 0), src0=sgpr(newIdxS + 0), src1=sgpr(outOffsetS),
                           comment="new lo += offset"))
        module.add(SAddCU32(dst=sgpr(newIdxS + 1), src0=sgpr(newIdxS + 1), src1=0, comment="new hi (carry)"))

        # CanWriteUpto(new)? if not, retry (a concurrent consumer may free space).
        self.emitCanWriteUpto(module, w, handleBaseS, cachedHwReadIdxS, newIdxS, canS, tmpPair)
        module.add(SCmpEQU32(src0=sgpr(canS), src1=0, comment="CanWriteUpto == false?"))
        module.add(SCBranchSCC1(labelName=loopLabel.getLabelName(), comment="full -> retry"))

        # CAS(cachedWptr, cur -> new): data[0:1]=new(swap), data[2:3]=cur(compare).
        module.add(VMovB32(dst=vgpr(vCasData + 0), src=sgpr(newIdxS + 0), comment="swap lo = new"))
        module.add(VMovB32(dst=vgpr(vCasData + 1), src=sgpr(newIdxS + 1), comment="swap hi = new"))
        # (vCasData+2/3 already hold cur from the load above = the compare value.)
        self._emitReserveCas(module, w, cachedWptrPtrS, vCasAddr, vCasData)
        module.add(SWaitCnt(vlcnt=0, vscnt=0, comment="wait CAS return"))
        # CAS returns the pre-op memory value in vCasData[0:1]; success iff it == cur.
        module.add(self._vReadfirstlane(tmpPair + 0, vCasData + 0, "CAS pre-op lo"))
        module.add(SCmpEQU32(src0=sgpr(tmpPair + 0), src1=sgpr(outCurS + 0),
                             comment="CAS pre-op lo == cur lo? (won the slot)"))
        module.add(SCBranchSCC0(labelName=loopLabel.getLabelName(), comment="lost race -> retry"))
        module.add(self._vReadfirstlane(tmpPair + 1, vCasData + 1, "CAS pre-op hi"))
        module.add(SCmpEQU32(src0=sgpr(tmpPair + 1), src1=sgpr(outCurS + 1), comment="pre-op hi == cur hi?"))
        module.add(SCBranchSCC0(labelName=loopLabel.getLabelName(), comment="lost race -> retry"))
        module.add(SBranch(labelName=doneLabel.getLabelName(), comment="won -> reserved"))
        module.add(doneLabel)

        w.vgprPool.checkIn(vCasAddr)
        w.vgprPool.checkIn(vCasData)
        w.sgprPool.checkIn(cachedWptrPtrS)
        w.sgprPool.checkIn(newIdxS)
        w.sgprPool.checkIn(wrapS)
        w.sgprPool.checkIn(canS)
        w.sgprPool.checkIn(tmpPair)
        return module

    def _emitReserveCas(self, module, w, cachedWptrPtrS, vCasAddr, vCasData):
        """64-bit compare-exchange of cachedWptr, device scope + return (sc0).

        Isolated so the CAS primitive can be swapped without touching the reserve
        logic. Uses global_atomic_cmpswap_x2 (the GlobalAtomicCmpswapB64 opcode,
        rendered "_x2" on gfx9 / "_b64" on gfx11+), taking the raw cachedWptr
        pointer in VGPRs directly -- the same bare-pointer atomic idiom the fused-
        A2A handshake uses (GlobalWriteBatch._emitFusedA2AHandshake). vCasData is
        4 dwords: [0:1]=swap(new), [2:3]=compare(cur); the pre-op value returns in
        [0:1]. glc=True selects return-of-pre-op (the assembler REQUIRES sc0 on
        this op: "instruction must use sc0"); slc=False selects device scope.
        """
        self._ptrToVgpr(module, vCasAddr, cachedWptrPtrS, "cachedWptr addr")
        off = vgpr("off", 1, False, False, True)
        module.add(GlobalAtomicCmpswapB64(
            dst=vgpr(vCasData, 2), vaddr=vgpr(vCasAddr, 2), data=vgpr(vCasData, 4), saddr=off,
            modifier=GLOBALModifiers(glc=True, slc=False),
            comment="CAS cachedWptr cur->new (device scope, return pre-op: sc0)"))
        return module

    # ---- placePacket -------------------------------------------------------

    def emitPlacePacket(self, module, w, handleBaseS, packetDwordsV, numDwords,
                        pendingWptrS, offsetS):
        """MORI placePacket (anvil_device.hpp:171-195): write `offsetS` bytes of
        zero-padding (NOPs) then `numDwords` packet dwords into the ring, all at
        AGENT scope (sc1). Advances pendingWptrS (2 SGPRs) by offset then by the
        packet size. `packetDwordsV` is the base VGPR of the already-built packet
        (Task 5 fills it); `numDwords` is compile-time.

        Ring addressing is per-dword: base_dword = WrapIntoRing(pending)/4, and
        each store targets queueBuf[base_dword + i]. queueBuf is a uint32_t*, so
        the byte address is queueBuf + WrapIntoRing(pending) (already a dword-
        aligned byte offset). Wrap padding is emitted as an unrolled run of
        zero stores when offset is a compile-time constant; when it is runtime
        (the general reserve result) a small loop covers it.
        """
        queueBufPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_pp_qbuf", preventOverflow=False)
        self._loadFieldPtr(module, w, queueBufPtrS, handleBaseS, OFF_queueBuf)
        module.add(SWaitCnt(kmcnt=0, comment="wait queueBuf pointer load"))

        wrapS   = w.sgprPool.checkOut(1, tag="sdma_pp_wrap", preventOverflow=False)
        cntS    = w.sgprPool.checkOut(1, tag="sdma_pp_cnt", preventOverflow=False)
        vAddr   = w.vgprPool.checkOutAligned(2, 2, tag="sdma_pp_addr")
        vZero   = w.vgprPool.checkOut(1, tag="sdma_pp_zero")
        off     = vgpr("off", 1, False, False, True)
        module.add(VMovB32(dst=vgpr(vZero), src=0, comment="padding NOP value = 0"))

        # ---- padding段: store `offset` bytes of zero at WrapIntoRing(pending). ----
        padLoop = Label(w.labels.getNameInc("sdma_pp_padloop"), "placePacket: zero-pad ring tail")
        padDone = Label(w.labels.getNameInc("sdma_pp_paddone"), "placePacket: padding done")
        # numOffsetDwords = offset / 4; if 0, skip the pad loop entirely.
        module.add(SLShiftRightB32(dst=sgpr(cntS), src=sgpr(offsetS), shiftHex=2,
                                   comment="numOffsetDwords = offset / 4"))
        module.add(SCmpEQU32(src0=sgpr(cntS), src1=0, comment="no padding?"))
        module.add(SCBranchSCC1(labelName=padDone.getLabelName(), comment="offset==0 -> skip pad"))
        module.add(padLoop)
        self._wrapIntoRing(module, wrapS, pendingWptrS + 0, "WrapIntoRing(pending) (pad)")
        self._emitRingByteAddr(module, vAddr, queueBufPtrS, wrapS)
        module.add(GlobalStoreB32(
            vaddr=vgpr(vAddr, 2), src=vgpr(vZero), saddr=off,
            modifier=GLOBALModifiers(glc=False, slc=True, isStore=True),
            comment="ring[wrap] = 0 padding NOP (AGENT scope, sc1)"))
        module.add(SAddU32(dst=sgpr(pendingWptrS + 0), src0=sgpr(pendingWptrS + 0), src1=4,
                           comment="pending += 4 (one padded dword)"))
        module.add(SAddCU32(dst=sgpr(pendingWptrS + 1), src0=sgpr(pendingWptrS + 1), src1=0, comment="pending hi carry"))
        module.add(SSubU32(dst=sgpr(cntS), src0=sgpr(cntS), src1=1, comment="numOffsetDwords -= 1"))
        module.add(SCmpEQU32(src0=sgpr(cntS), src1=0, comment="pad done?"))
        module.add(SCBranchSCC0(labelName=padLoop.getLabelName(), comment="more padding"))
        module.add(padDone)

        # ---- packet段: store numDwords packet dwords at WrapIntoRing(pending). ----
        # Recompute base after padding advanced pending. numDwords is compile-time,
        # so unroll (matches MORI's compile-time-bounded loop; one warp writes <=64).
        self._wrapIntoRing(module, wrapS, pendingWptrS + 0, "WrapIntoRing(pending) (packet base)")
        self._emitRingByteAddr(module, vAddr, queueBufPtrS, wrapS)
        for i in range(numDwords):
            module.add(GlobalStoreB32(
                vaddr=vgpr(vAddr, 2), src=vgpr(packetDwordsV + i), saddr=off,
                modifier=GLOBALModifiers(offset=i * 4, glc=False, slc=True, isStore=True),
                comment="ring[base + %d] = packet dword %d (AGENT scope, sc1)" % (i, i)))
        # pending += numDwords*4 (packet size).
        module.add(SAddU32(dst=sgpr(pendingWptrS + 0), src0=sgpr(pendingWptrS + 0), src1=numDwords * 4,
                           comment="pending += packet size"))
        module.add(SAddCU32(dst=sgpr(pendingWptrS + 1), src0=sgpr(pendingWptrS + 1), src1=0, comment="pending hi carry"))

        w.vgprPool.checkIn(vAddr)
        w.vgprPool.checkIn(vZero)
        w.sgprPool.checkIn(wrapS)
        w.sgprPool.checkIn(cntS)
        w.sgprPool.checkIn(queueBufPtrS)
        return module

    def _emitRingByteAddr(self, module, vAddrV, queueBufPtrS, wrapS):
        """Compute the 64-bit VGPR byte address queueBuf + WrapIntoRing(pending)
        into vAddrV[0:1]. queueBuf is a byte-addressable base; the wrapped index
        is already a byte offset (<4 GiB, so it adds only into the low dword with
        carry)."""
        module.add(VMovB32(dst=vgpr(vAddrV + 0), src=sgpr(queueBufPtrS + 0), comment="queueBuf lo"))
        module.add(VMovB32(dst=vgpr(vAddrV + 1), src=sgpr(queueBufPtrS + 1), comment="queueBuf hi"))
        module.add(VAddCOU32(dst=vgpr(vAddrV + 0), dst1=VCC(), src0=sgpr(wrapS), src1=vgpr(vAddrV + 0),
                             comment="addr lo = queueBuf + WrapIntoRing(pending)"))
        module.add(VAddCCOU32(dst=vgpr(vAddrV + 1), dst1=VCC(), src0=vgpr(vAddrV + 1), src1=0, src2=VCC(),
                              comment="addr hi (carry)"))
        return module

    # ---- submitPacket ------------------------------------------------------

    def emitSubmitPacket(self, module, w, handleBaseS, baseS, pendingWptrS):
        """MORI submitPacket (anvil_device.hpp:197-234): serialize this producer's
        commit behind earlier reservations, then publish the packet.

        (1) spin until committedWptr == base (this producer's turn; earlier
            reservations commit in order). Read committedWptr at AGENT scope.
        (2) Global Constraint 4 publish sequence (any bit wrong => timing hang):
              store wptr = pending        AGENT  (sc1)
              s_waitcnt vmcnt(0)
              store doorbell = pending    SYSTEM (sc0 sc1)   <-- rings the engine
              store committedWptr = pend  AGENT  (sc1)       <-- unblocks next producer
            The value written to wptr/doorbell/committedWptr is the new absolute
            byte wptr (pending), NOT an increment. A vmcnt(0) precedes the
            doorbell so the wptr store is globally ordered before the engine is
            told to read up to it.

        baseS / pendingWptrS are 2-SGPR byte indices from the reserve+place pair.
        Emitted by a single elected lane (Task 7 gates it), so NO s_barrier here
        -- MORI's wave_barrier is a C++ compiler fence; in single-lane assembly
        the s_waitcnt already orders memory and an s_barrier would deadlock.
        """
        # --- (1) spin: committedWptr == base ---
        commPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_sp_commPtr", preventOverflow=False)
        self._loadFieldPtr(module, w, commPtrS, handleBaseS, OFF_committedWptr)
        module.add(SWaitCnt(kmcnt=0, comment="wait committedWptr pointer load"))

        vAddr = w.vgprPool.checkOutAligned(2, 2, tag="sdma_sp_addr")
        vVal  = w.vgprPool.checkOutAligned(2, 2, tag="sdma_sp_val")
        tmpS  = w.sgprPool.checkOut(1, tag="sdma_sp_tmp", preventOverflow=False)
        off   = vgpr("off", 1, False, False, True)
        self._ptrToVgpr(module, vAddr, commPtrS, "committedWptr addr")

        spinLabel = Label(w.labels.getNameInc("sdma_submit_spin"), "submitPacket: wait committedWptr == base")
        spinDone  = Label(w.labels.getNameInc("sdma_submit_ready"), "submitPacket: our turn")
        module.add(spinLabel)
        module.add(GlobalLoadB64(
            dst=vgpr(vVal, 2), vaddr=vgpr(vAddr, 2), saddr=off,
            modifier=GLOBALModifiers(glc=False, slc=True),
            comment="load committedWptr (AGENT scope, sc1)"))
        module.add(SWaitCnt(vlcnt=0, comment="wait committedWptr load"))
        module.add(self._vReadfirstlane(tmpS, vVal + 0, "committedWptr lo"))
        module.add(SCmpEQU32(src0=sgpr(tmpS), src1=sgpr(baseS + 0), comment="committedWptr lo == base lo?"))
        module.add(SCBranchSCC0(labelName=spinLabel.getLabelName(), comment="not our turn -> spin"))
        module.add(self._vReadfirstlane(tmpS, vVal + 1, "committedWptr hi"))
        module.add(SCmpEQU32(src0=sgpr(tmpS), src1=sgpr(baseS + 1), comment="committedWptr hi == base hi?"))
        module.add(SCBranchSCC0(labelName=spinLabel.getLabelName(), comment="not our turn -> spin"))
        module.add(SSleep(simm16=1, comment="submitPacket: brief backoff between polls"))
        module.add(spinDone)
        module.add(SWaitCnt(vlcnt=0, vscnt=0, comment="ensure our packet stores are globally visible before wptr"))

        # value written to wptr / doorbell / committedWptr = pending (absolute byte wptr).
        module.add(VMovB32(dst=vgpr(vVal + 0), src=sgpr(pendingWptrS + 0), comment="publish value = pending lo"))
        module.add(VMovB32(dst=vgpr(vVal + 1), src=sgpr(pendingWptrS + 1), comment="publish value = pending hi"))

        # --- (2a) store wptr = pending  (AGENT, sc1) ---
        wptrPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_sp_wptrPtr", preventOverflow=False)
        self._loadFieldPtr(module, w, wptrPtrS, handleBaseS, OFF_wptr)
        module.add(SWaitCnt(kmcnt=0, comment="wait wptr pointer load"))
        self._ptrToVgpr(module, vAddr, wptrPtrS, "wptr addr")
        module.add(GlobalStoreB64(
            vaddr=vgpr(vAddr, 2), src=vgpr(vVal, 2), saddr=off,
            modifier=GLOBALModifiers(glc=False, slc=True, isStore=True),
            comment="store wptr = pending (AGENT scope, sc1)"))
        w.sgprPool.checkIn(wptrPtrS)

        # --- vmcnt(0): order the wptr store before the doorbell (Global Constraint 4) ---
        module.add(SWaitCnt(vscnt=0, comment="s_waitcnt vmcnt(0): wptr store visible before doorbell"))

        # --- (2b) store doorbell = pending  (SYSTEM, sc0 sc1) -> rings the engine ---
        dbPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_sp_dbPtr", preventOverflow=False)
        self._loadFieldPtr(module, w, dbPtrS, handleBaseS, OFF_doorbell)
        module.add(SWaitCnt(kmcnt=0, comment="wait doorbell pointer load"))
        self._ptrToVgpr(module, vAddr, dbPtrS, "doorbell addr")
        module.add(GlobalStoreB64(
            vaddr=vgpr(vAddr, 2), src=vgpr(vVal, 2), saddr=off,
            modifier=GLOBALModifiers(glc=True, slc=True, isStore=True),
            comment="ring doorbell = pending (SYSTEM scope, sc0 sc1)"))
        w.sgprPool.checkIn(dbPtrS)
        module.add(SWaitCnt(vscnt=0, comment="wait doorbell store issued"))

        # --- (2c) store committedWptr = pending  (AGENT, sc1) -> unblocks next producer ---
        self._ptrToVgpr(module, vAddr, commPtrS, "committedWptr addr")
        module.add(GlobalStoreB64(
            vaddr=vgpr(vAddr, 2), src=vgpr(vVal, 2), saddr=off,
            modifier=GLOBALModifiers(glc=False, slc=True, isStore=True),
            comment="store committedWptr = pending (AGENT scope, sc1)"))
        module.add(SWaitCnt(vscnt=0, comment="wait committedWptr store issued"))

        w.vgprPool.checkIn(vAddr)
        w.vgprPool.checkIn(vVal)
        w.sgprPool.checkIn(tmpS)
        w.sgprPool.checkIn(commPtrS)
        return module

    # ---- utility: 64-bit sub + readfirstlane (no direct rocisa 64-bit sub) --

    def _emitU64Sub(self, module, dstPairS, aPairS, bPairS, comment):
        """dst = a - b (64-bit) via s_sub_u32 / s_subb_u32."""
        module.add(SSubU32(dst=sgpr(dstPairS + 0), src0=sgpr(aPairS + 0), src1=sgpr(bPairS + 0),
                           comment=comment + " (lo)"))
        module.add(SSubBU32(dst=sgpr(dstPairS + 1), src0=sgpr(aPairS + 1), src1=sgpr(bPairS + 1),
                            comment=comment + " (hi, borrow)"))
        return module

    def _vReadfirstlane(self, dstS, srcV, comment):
        """v_readfirstlane_b32: move a lane-uniform VGPR value to an SGPR (the
        reserve/submit math is uniform, so lane 0 is representative)."""
        return VReadfirstlaneB32(dst=sgpr(dstS), src=vgpr(srcV), comment=comment)
