# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from rocisa.container import sgpr, SMEMModifiers
from rocisa.code import Label
from rocisa.instruction import (
    SMovB32, SMovB64, SLoadB64, SLoadB256,
    SAddU32, SAddCU32, SSubU32, SSubBU32,
    SAndB32, SLShiftRightB32,
    SCmpEQU32, SCmpEQU64, SCmpLeU32, SCmpLtU32,
    SCBranchSCC0, SCBranchSCC1, SBranch,
    SWaitCnt, SSleep,
    SStoreB32, SStoreB64, SStoreB128, SAtomicCmpswapX2,
)


# 256 KB SDMA ring, matching client/include/SdmaQueue.hpp SDMA_QUEUE_SIZE. Power of
# two => wrapping is an AND mask, never a divide; under 4 GiB => a wrapped index
# fits the low dword and its high dword is always 0.
SDMA_QUEUE_SIZE = 256 * 1024
assert (SDMA_QUEUE_SIZE & (SDMA_QUEUE_SIZE - 1)) == 0, "ring size must be a power of two"

# Byte offsets of every SdmaQueueDeviceHandle field (contract: the static_asserts
# in client/include/SdmaQueue.hpp lock these).
OFF_queueBuf         = 0    # ring base (uint32_t*, dword-addressed)
OFF_rptr             = 8    # hardware read pointer  (SYSTEM-scope read)
OFF_wptr             = 16   # hardware write pointer (AGENT-scope write)
OFF_doorbell         = 24   # doorbell               (SYSTEM-scope write)
OFF_cachedWptr       = 32   # producer reservation cursor (AGENT-scope CAS)
OFF_committedWptr    = 40   # commit-serialization cursor (AGENT-scope)
OFF_cachedHwReadIndex = 48  # per-producer private cache SEED (value, never stored back)


class SdmaRingEmitter:
    """Packet-independent SDMA ring producer, emitted as rocisa Modules.

    Persistent per-producer state lives in caller-owned SGPRs:
      * handleBase (2 SGPRs): pointer to this peer's SdmaQueueDeviceHandle.
      * cachedHwReadIdx (2 SGPRs): the private room-check cache. The caller
        seeds it ONCE from handle+48 at setup; this emitter reads and refreshes
        it in-register and NEVER stores it back to memory (see the note in
        client/include/SdmaQueue.hpp).
    """

    def __init__(self, queueSize: int = SDMA_QUEUE_SIZE):
        assert (queueSize & (queueSize - 1)) == 0, "ring size must be a power of two"
        self.queueSize = queueSize
        self.ringMask  = queueSize - 1

    # ---- room check -------------------------------------------------------

    def _emitRoomCheck(self, module, w, cachedHwReadIdxS, uptoIdxS, tmpPairS, what):
        """SCC = ((upto - cachedHwReadIndex) < queueSize). Registers only.

        Both exits leave SCC correct: the high-dword branch is taken exactly
        when SCC is already 0, and the fall-through path ends on the compare
        that decides it. There is no s_cmp_lt_u64 -- gfx950 has only the u64
        equality forms -- hence the split; and queueSize is under 2^32, so any
        nonzero high dword already means no room.
        """
        noRoom = Label(w.labels.getNameInc("sdma_roomcheck_no"), "RoomCheck: gap >= 2^32, SCC already 0")
        module.add(SSubU32(dst=sgpr(tmpPairS + 0), src0=sgpr(uptoIdxS + 0),
                           src1=sgpr(cachedHwReadIdxS + 0),
                           comment="RoomCheck: %s (lo)" % what))
        module.add(SSubBU32(dst=sgpr(tmpPairS + 1), src0=sgpr(uptoIdxS + 1),
                            src1=sgpr(cachedHwReadIdxS + 1),
                            comment="RoomCheck: %s (hi, borrow)" % what))
        module.add(SCmpEQU32(src0=sgpr(tmpPairS + 1), src1=0, comment="diff hi == 0? (gap < 2^32)"))
        module.add(SCBranchSCC0(labelName=noRoom.getLabelName(), comment="hi != 0 -> no room, SCC=0"))
        module.add(SCmpLtU32(src0=sgpr(tmpPairS + 0), src1=self.queueSize,
                             comment="SCC = diff < queueSize? (room)"))
        module.add(noRoom)

    def _emitRefreshCache(self, module, w, handleBaseS, cachedHwReadIdxS):
        """cachedHwReadIdx = *rptr, re-read past the caches with glc.

        Emits only s_load and s_waitcnt, neither of which writes SCC, so a room
        check on either side of this keeps its answer.

        The load lands STRAIGHT IN the caller's private pair: it is the refresh,
        so there is nothing to relay.
        """
        rptrPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_cw_rptrPtr", preventOverflow=False)
        module.add(SLoadB64(dst=sgpr(rptrPtrS, 2), base=sgpr(handleBaseS, 2),
                            soffset=hex(OFF_rptr), comment="load rptr pointer"))
        module.add(SWaitCnt(kmcnt=0, comment="wait rptr pointer load"))
        module.add(SLoadB64(dst=sgpr(cachedHwReadIdxS, 2), base=sgpr(rptrPtrS, 2),
                            soffset=hex(0), smem=SMEMModifiers(glc=True),
                            comment="cachedHwReadIndex = hardware rptr"))
        module.add(SWaitCnt(kmcnt=0, comment="wait rptr load"))
        w.sgprPool.checkIn(rptrPtrS)

    # ---- ReserveQueueSpace (CAS, NOT fetch_add) ----------------------------

    def emitReserveQueueSpace(self, module, w, handleBaseS, cachedHwReadIdxS,
                              sizeInBytes, outCurS, outOffsetS):
        """Reserve `sizeInBytes` in the ring via a compare-exchange loop and
        compute the wrap-padding. outCurS (2 SGPRs) = reserved base index,
        outOffsetS (1 SGPR) = pad bytes, sizeInBytes a compile-time immediate.

        MUST be CAS, not fetch_add: the wrap padding depends on the CURRENT
        cur_index, so computing the new index and claiming the slot have to be
        one atomic step, or two producers pad differently and both believe they
        won.

        The seeded compare value is a hint. A stale one loses the first race
        and comes back with the truth, and it cannot wedge the loop either:
        cachedWptr only increases, so a stale seed is smaller, which makes the
        room check more likely to pass rather than less.
        """
        loopLabel  = Label(w.labels.getNameInc("sdma_reserve_loop"), "ReserveQueueSpace: CAS retry loop")
        noPadLabel = Label(w.labels.getNameInc("sdma_reserve_nopad"), "ReserveQueueSpace: no wrap padding")
        doneLabel  = Label(w.labels.getNameInc("sdma_reserve_done"),  "ReserveQueueSpace: reserved")

        cachedWptrPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_rsv_cwPtr", preventOverflow=False)
        module.add(SLoadB64(dst=sgpr(cachedWptrPtrS, 2), base=sgpr(handleBaseS, 2),
                            soffset=hex(OFF_cachedWptr), comment="load cachedWptr pointer"))
        module.add(SWaitCnt(kmcnt=0, comment="wait cachedWptr pointer load"))

        wrapS   = w.sgprPool.checkOut(1, tag="sdma_rsv_wrap", preventOverflow=False)
        tmpPair = w.sgprPool.checkOutAligned(2, 2, tag="sdma_rsv_tmp", preventOverflow=False)

        # CAS data block: [0:1]=swap and the pre-op return, [2:3]=compare(cur).
        # 4-ALIGNED for the 4-dword SDATA (see _packetStoreWidths).
        casDataS = w.sgprPool.checkOutAligned(4, 4, tag="sdma_rsv_casData", preventOverflow=False)
        # The new index is accumulated straight into the swap half rather than
        # into its own pair: [0:1] is dead on entry to the loop body, because
        # the retry path copies the pre-op into the compare half before
        # branching back, and the swap value is what the CAS wants there anyway.
        newIdxS = casDataS

        # Seed the compare slot with the current value (see the docstring on why
        # a possibly-stale seed is safe here).
        module.add(SLoadB64(dst=sgpr(casDataS + 2, 2), base=sgpr(cachedWptrPtrS, 2),
                            soffset=hex(0), smem=SMEMModifiers(glc=True),
                            comment="seed cur = cachedWptr (hint; the CAS self-corrects)"))
        module.add(SWaitCnt(kmcnt=0, comment="wait cachedWptr seed"))

        module.add(loopLabel)
        module.add(SMovB64(dst=sgpr(outCurS, 2), src=sgpr(casDataS + 2, 2),
                           comment="cur = compare slot"))

        # off = 0 by default; if WrapIntoRing(cur)+size > queueSize -> pad tail.
        module.add(SMovB32(dst=sgpr(outOffsetS), src=0, comment="offset = 0 (no pad)"))
        module.add(SAndB32(dst=sgpr(wrapS), src0=sgpr(outCurS + 0), src1=self.ringMask,
                           comment="WrapIntoRing(cur)"))
        module.add(SAddU32(dst=sgpr(tmpPair), src0=sgpr(wrapS), src1=sizeInBytes,
                           comment="WrapIntoRing(cur) + size"))
        module.add(SCmpLeU32(src0=sgpr(tmpPair), src1=self.queueSize,
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

        # Room for `new`? Check the private cache first; if that says full,
        # re-read the hardware rptr and retest before giving up -- a concurrent
        # consumer may have freed space. Each check leaves its answer in SCC,
        # and nothing between a check and its branch writes SCC.
        haveRoomLabel = Label(w.labels.getNameInc("sdma_rsv_haveroom"),
                              "ReserveQueueSpace: room, go claim the slot")
        self._emitRoomCheck(module, w, cachedHwReadIdxS, newIdxS, tmpPair,
                            "new - cachedHwReadIndex")
        module.add(SCBranchSCC1(labelName=haveRoomLabel.getLabelName(), comment="room via cached index"))
        # ⚠ THE REFRESH IS ALMOST NEVER EXERCISED end to end: it runs only when
        # the room check fails, i.e. when the SDMA engine has fallen a whole
        # ring behind. A passing validation run is therefore NOT evidence about
        # these instructions.
        self._emitRefreshCache(module, w, handleBaseS, cachedHwReadIdxS)
        self._emitRoomCheck(module, w, cachedHwReadIdxS, newIdxS, tmpPair,
                            "new - refreshed rptr")
        module.add(SCBranchSCC0(labelName=loopLabel.getLabelName(), comment="still full -> retry"))
        module.add(haveRoomLabel)

        # CAS(cachedWptr, cur -> new): [0:1] already holds new, [2:3] holds cur.
        # casDataS is 4-ALIGNED: [0:1] is the swap value and the pre-op comes
        # back over it, [2:3] is the compare value. glc on an atomic selects
        # return-of-pre-op.
        #
        # An atomic must sit in a single-instruction clause (ISA 8.2); the
        # branch before and the s_waitcnt after keep it alone, so do not let
        # another SMEM op become adjacent. The retry path below rewrites this
        # instruction's own source registers, which that section explicitly
        # permits for an atomic returning its pre-op value.
        module.add(SAtomicCmpswapX2(
            dst=sgpr(casDataS, 4), base=sgpr(cachedWptrPtrS, 2), soffset=hex(0),
            smem=SMEMModifiers(glc=True),
            comment="CAS cachedWptr cur->new (glc = return pre-op)"))
        module.add(SWaitCnt(kmcnt=0, comment="wait CAS return"))
        # The pre-op memory value comes back in casDataS[0:1]; we won iff it == cur.
        module.add(SCmpEQU64(src0=sgpr(casDataS, 2), src1=sgpr(outCurS, 2),
                             comment="CAS pre-op == cur? (won the slot)"))
        module.add(SCBranchSCC1(labelName=doneLabel.getLabelName(), comment="won -> reserved"))

        # Fell through, so we lost the race. The pre-op value IS the current
        # cur, so move it into the compare slot and go round again. This is the
        # only path that updates cur, which is why the loop body never reloads
        # it.
        module.add(SMovB64(dst=sgpr(casDataS + 2, 2), src=sgpr(casDataS, 2),
                           comment="cur = CAS pre-op (refresh from the failed swap)"))
        module.add(SBranch(labelName=loopLabel.getLabelName(), comment="retry with the refreshed cur"))
        module.add(doneLabel)

        w.sgprPool.checkIn(casDataS)
        w.sgprPool.checkIn(cachedWptrPtrS)
        w.sgprPool.checkIn(wrapS)
        w.sgprPool.checkIn(tmpPair)

    # ---- placePacket -------------------------------------------------------

    def emitPlacePacket(self, module, w, handleBaseS, packetDwordsS, numDwords,
                        pendingWptrS, offsetS):
        """Write `offsetS` bytes of zero-padding (NOPs) then `numDwords` packet
        dwords into the ring, all with s_store...glc. Advances pendingWptrS
        (2 SGPRs) by offset then by the packet size. `packetDwordsS` is the base
        SGPR of the already-built packet (SdmaPacketEmitter fills it);
        `numDwords` is compile-time.

        BLOCK REUSE. The caller may pass the SAME packetDwordsS block for a
        later packet, but only AFTER this call has emitted the first one's
        stores -- rebuilding it earlier overwrites dwords not yet stored, and
        nothing here would notice.

        What that needs is a CLAUSE BOUNDARY, not an s_waitcnt: a scalar
        store's sources have to survive its clause, not its completion, and the
        rebuild starts with an s_mov, which breaks the clause. A wait here would
        only serialize the submit.
        """
        queueBufPtrS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_pp_qbuf", preventOverflow=False)
        module.add(SLoadB64(dst=sgpr(queueBufPtrS, 2), base=sgpr(handleBaseS, 2),
                            soffset=hex(OFF_queueBuf), comment="load queueBuf pointer"))
        module.add(SWaitCnt(kmcnt=0, comment="wait queueBuf pointer load"))

        wrapS   = w.sgprPool.checkOut(1, tag="sdma_pp_wrap", preventOverflow=False)
        cntS    = w.sgprPool.checkOut(1, tag="sdma_pp_cnt", preventOverflow=False)
        # 2-ALIGNED: this pair is the SBASE of every store below (see
        # _emitRingByteAddr).
        addrS   = w.sgprPool.checkOutAligned(2, 2, tag="sdma_pp_addr", preventOverflow=False)
        zeroS   = w.sgprPool.checkOut(1, tag="sdma_pp_zero", preventOverflow=False)
        module.add(SMovB32(dst=sgpr(zeroS), src=0, comment="padding NOP value = 0"))

        # ---- padding: store `offset` bytes of zero at WrapIntoRing(pending). ----
        padLoop = Label(w.labels.getNameInc("sdma_pp_padloop"), "placePacket: zero-pad ring tail")
        padDone = Label(w.labels.getNameInc("sdma_pp_paddone"), "placePacket: padding done")
        module.add(SLShiftRightB32(dst=sgpr(cntS), src=sgpr(offsetS), shiftHex=2,
                                   comment="numOffsetDwords = offset / 4"))
        module.add(SCmpEQU32(src0=sgpr(cntS), src1=0, comment="no padding?"))
        module.add(SCBranchSCC1(labelName=padDone.getLabelName(), comment="offset==0 -> skip pad"))
        module.add(padLoop)
        module.add(SAndB32(dst=sgpr(wrapS), src0=sgpr(pendingWptrS + 0), src1=self.ringMask,
                           comment="WrapIntoRing(pending) (pad)"))
        self._emitRingByteAddr(module, addrS, queueBufPtrS, wrapS)
        # glc on every ring store: past the scalar data cache and L2.
        module.add(SStoreB32(
            src=sgpr(zeroS), base=sgpr(addrS, 2), soffset=hex(0),
            smem=SMEMModifiers(glc=True, isStore=True),
            comment="ring[wrap] = 0 padding NOP"))
        module.add(SAddU32(dst=sgpr(pendingWptrS + 0), src0=sgpr(pendingWptrS + 0), src1=4,
                           comment="pending += 4 (one padded dword)"))
        module.add(SAddCU32(dst=sgpr(pendingWptrS + 1), src0=sgpr(pendingWptrS + 1), src1=0, comment="pending hi carry"))
        module.add(SSubU32(dst=sgpr(cntS), src0=sgpr(cntS), src1=1, comment="numOffsetDwords -= 1"))
        module.add(SCmpEQU32(src0=sgpr(cntS), src1=0, comment="pad done?"))
        module.add(SCBranchSCC0(labelName=padLoop.getLabelName(), comment="more padding"))
        module.add(padDone)

        # ---- packet: store numDwords packet dwords at WrapIntoRing(pending). ----
        # Recompute base after padding advanced pending. numDwords is compile-time,
        # so unroll (one warp writes <=64).
        module.add(SAndB32(dst=sgpr(wrapS), src0=sgpr(pendingWptrS + 0), src1=self.ringMask,
                           comment="WrapIntoRing(pending) (packet base)"))
        self._emitRingByteAddr(module, addrS, queueBufPtrS, wrapS)
        for i, width in self._packetStoreWidths(packetDwordsS, numDwords):
            op = {1: SStoreB32, 2: SStoreB64, 4: SStoreB128}[width]
            src = sgpr(packetDwordsS + i) if width == 1 else sgpr(packetDwordsS + i, width)
            module.add(op(
                src=src, base=sgpr(addrS, 2), soffset=hex(i * 4),
                smem=SMEMModifiers(glc=True, isStore=True),
                comment="ring[base + %d] = packet dword%s"
                        % (i, "" if width == 1 else "s %d..%d" % (i, i + width - 1))))
        # pending += numDwords*4 (packet size).
        module.add(SAddU32(dst=sgpr(pendingWptrS + 0), src0=sgpr(pendingWptrS + 0), src1=numDwords * 4,
                           comment="pending += packet size"))
        module.add(SAddCU32(dst=sgpr(pendingWptrS + 1), src0=sgpr(pendingWptrS + 1), src1=0, comment="pending hi carry"))

        w.sgprPool.checkIn(addrS)
        w.sgprPool.checkIn(zeroS)
        w.sgprPool.checkIn(wrapS)
        w.sgprPool.checkIn(cntS)
        w.sgprPool.checkIn(queueBufPtrS)

    @staticmethod
    def _packetStoreWidths(baseS, numDwords):
        """Split a run of `numDwords` consecutive SGPRs starting at `baseS` into
        the widest scalar stores gfx950 will take. Returns [(dwordIndex, width)].

        x4 is the ceiling: SMEM stores write 1-4 Dwords and there is no
        s_store_dwordx8. rocisa exposes SStoreB256 / SStoreB512 anyway -- those
        are load-shaped leftovers and must not be reached for here.

        The ring address needs only Dword alignment; SMEM ignores the two LSBs
        of the byte address and OFFSET has no alignment restriction. That an x4
        store may cross a 16-byte boundary is the ISA answering by omission,
        so the end-to-end run is what actually confirms it -- the 84-byte
        reservation stride puts packets at every 4-byte phase.
        """
        out, i = [], 0
        while i < numDwords:
            reg, left = baseS + i, numDwords - i
            if reg % 4 == 0 and left >= 4:
                width = 4
            elif reg % 2 == 0 and left >= 2:
                width = 2
            else:
                width = 1
            out.append((i, width)); i += width
        return out

    def _emitRingByteAddr(self, module, dstPairS, queueBufPtrS, wrapS):
        """Compute the 64-bit byte address queueBuf + WrapIntoRing(pending) into
        the 2-ALIGNED SGPR pair dstPairS. dstPairS becomes the SBASE of every
        store in the placement, and SMEM requires an even SBASE (measured at the
        assembler: s_store with s[3:4] is rejected for register alignment).

        Folding the wrap into the base here, once per placement, is what lets
        the stores address their dwords with plain IMMEDIATE offsets. The
        alternative -- leaving the base at queueBuf and passing the wrap as
        SOFFSET -- reads better but is a trap: rocisa's SMEMModifiers omits the
        `offset:` text entirely when the offset is 0, so the FIRST store of each
        packet would silently drop from the IMM=1/SOE=1 encoding to IMM=0/SOE=0
        and put the runtime SGPR in the OFFSET field, which Table 39 documents
        as immediate-or-M0 only for stores. One store per packet encoded
        differently from its neighbours is not a bug anyone finds by reading."""
        module.add(SAddU32(dst=sgpr(dstPairS + 0), src0=sgpr(queueBufPtrS + 0), src1=sgpr(wrapS),
                           comment="addr lo = queueBuf + WrapIntoRing(pending)"))
        module.add(SAddCU32(dst=sgpr(dstPairS + 1), src0=sgpr(queueBufPtrS + 1), src1=0,
                            comment="addr hi (carry)"))

    # ---- submitPacket ------------------------------------------------------

    def emitSubmitPacket(self, module, w, handleBaseS, baseS, pendingWptrS):
        """Serialize this producer's commit behind earlier reservations, then
        publish the packet.

        (1) spin until committedWptr == base (this producer's turn; earlier
            reservations commit in order), polling with s_load...glc.
        (2) Publish sequence (any bit wrong => timing hang):
              store wptr = pending        s_store_dwordx2 glc
              s_waitcnt lgkmcnt(0)
              store doorbell = pending    s_store_dwordx2 glc   <-- rings the engine
              store committedWptr = pend  s_store_dwordx2 glc   <-- unblocks next producer
            The value written to wptr/doorbell/committedWptr is the new absolute
            byte wptr (pending), NOT an increment. A wait precedes the doorbell
            so the wptr store is globally ordered before the engine is told to
            read up to it. All three are scalar, so the ordering waits are
            lgkmcnt rather than vmcnt, and pending is written straight from its
            SGPR pair (2-ALIGNED for the x2 SDATA; the caller allocates it so).

        baseS / pendingWptrS are 2-SGPR byte indices from the reserve+place pair.
        Emitted by a single elected lane, so NO s_barrier here: in single-lane
        assembly the s_waitcnt already orders memory and an s_barrier would
        deadlock.

        The election must yield exactly ONE submitting wave -- two would publish
        two packets against one reservation. That is a caller obligation;
        nothing here detects it.
        """
        # Every pointer this function needs lies in one 32-byte run of the
        # handle -- wptr at 16 through committedWptr at 40 -- so one load gets
        # all of them: [0:1] wptr, [2:3] doorbell, [6:7] committedWptr, with
        # [4:5] the cachedWptr pointer that is not used here. x8 SDATA wants a
        # multiple of four. Doing it once also keeps a load out of the publish
        # sequence below, where it would sit between the wptr store and the
        # doorbell it is meant to precede.
        ptrsS = w.sgprPool.checkOutAligned(8, 4, tag="sdma_sp_ptrs", preventOverflow=False)
        module.add(SLoadB256(dst=sgpr(ptrsS, 8), base=sgpr(handleBaseS, 2),
                             soffset=hex(OFF_wptr),
                             comment="load wptr / doorbell / committedWptr pointers"))
        module.add(SWaitCnt(kmcnt=0, comment="wait handle pointer load"))
        wptrPtrS = ptrsS + (OFF_wptr          - OFF_wptr) // 4
        dbPtrS   = ptrsS + (OFF_doorbell      - OFF_wptr) // 4
        commPtrS = ptrsS + (OFF_committedWptr - OFF_wptr) // 4

        # --- (1) spin: committedWptr == base ---
        # 2-ALIGNED: this pair is the x2 destination of the poll below.
        pollS = w.sgprPool.checkOutAligned(2, 2, tag="sdma_sp_poll", preventOverflow=False)

        spinLabel = Label(w.labels.getNameInc("sdma_submit_spin"), "submitPacket: wait committedWptr == base")
        module.add(spinLabel)
        module.add(SSleep(simm16=1, comment="submitPacket: backoff between polls (must stay INSIDE the spin body)"))
        # glc, so this is a poll and not one value read many times.
        module.add(SLoadB64(dst=sgpr(pollS, 2), base=sgpr(commPtrS, 2),
                            soffset=hex(0), smem=SMEMModifiers(glc=True),
                            comment="poll committedWptr"))
        module.add(SWaitCnt(kmcnt=0, comment="wait committedWptr load"))
        module.add(SCmpEQU64(src0=sgpr(pollS, 2), src1=sgpr(baseS, 2),
                             comment="committedWptr == base?"))
        module.add(SCBranchSCC0(labelName=spinLabel.getLabelName(), comment="not our turn -> spin"))
        # The packet stores are scalar, so what has to drain here is lgkmcnt,
        # not vmcnt. Getting this counter wrong is a timing-only failure: the
        # doorbell would reach the engine ahead of the packet it announces.
        module.add(SWaitCnt(kmcnt=0, comment="ensure our packet stores are globally visible before wptr"))

        # --- (2a) store wptr = pending ---
        module.add(SStoreB64(
            src=sgpr(pendingWptrS, 2), base=sgpr(wptrPtrS, 2), soffset=hex(0),
            smem=SMEMModifiers(glc=True, isStore=True),
            comment="store wptr = pending"))

        # --- order the wptr store before the doorbell ---
        module.add(SWaitCnt(kmcnt=0, comment="s_waitcnt lgkmcnt(0): wptr store visible before doorbell"))

        # --- (2b) store doorbell = pending -> rings the engine ---
        # An MMIO/BAR write, but no wider modifier than the other stores.
        module.add(SStoreB64(
            src=sgpr(pendingWptrS, 2), base=sgpr(dbPtrS, 2), soffset=hex(0),
            smem=SMEMModifiers(glc=True, isStore=True),
            comment="ring doorbell = pending"))
        module.add(SWaitCnt(kmcnt=0, comment="wait doorbell store issued"))

        # --- (2c) store committedWptr = pending -> unblocks next producer ---
        module.add(SStoreB64(
            src=sgpr(pendingWptrS, 2), base=sgpr(commPtrS, 2), soffset=hex(0),
            smem=SMEMModifiers(glc=True, isStore=True),
            comment="store committedWptr = pending"))
        module.add(SWaitCnt(kmcnt=0, comment="wait committedWptr store issued"))

        w.sgprPool.checkIn(pollS)
        w.sgprPool.checkIn(ptrsS)
