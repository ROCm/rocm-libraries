# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Accumulator/lane layout strategies for the native RMS epilogue.

RMSEpilogueEmitter runs one schedule (two-phase residual ring, gamma LDS reads,
partial reduction and write) for every kernel family. Everything that depends
on how a kernel distributes its accumulators over lanes lives here:

* SubtileRmsLayout: Subtile kernels. A "unit" is a (pair mp, column tile n) visited in
  the order given by unitCoords (M-pairs interleaved per column when T_M % 4 == 0, else
  column-fastest); a lane owns 8 rows spread over two 16-row tiles of one
  column, and the 4 row groups (g = lane >> 4) are reduced with permlane swaps.
* CmsRmsLayout: CMS kernels (SourceSwap, VW=8). A "unit" is one of 32 columns; a lane
  owns 8 contiguous rows of it. The 16 lanes sharing g (c = lane & 15) are
  reduce-scattered over c bits 0..1 with DPP quad_perm adds, then an LDS transpose
  lets every thread sum the remaining holders of its own column and store it.

The layouts hold no register lifetime of their own: they allocate into and read the
emitter's attributes, and the emitter's teardown frees them (see persistentAttrs).
"""

import math

from rocisa.container import DPPModifiers, DSModifiers, sgpr, vgpr
from rocisa.instruction import (
    DSLoadB32,
    DSLoadB64,
    DSStoreB32,
    SMulI32,
    SNop,
    SWaitCnt,
    VAddF32,
    VAddU32,
    VAndB32,
    VCmpNeU32,
    VCndMaskB32,
    VLShiftLeftB32,
    VLShiftRightB32,
    VMovB32,
    VMulLOU32,
    VOrB32,
    VPermlane16SwapB32,
    VPermlane32SwapB32,
)


class SubtileRmsLayout:
    """Layout of a Subtile kernel: T_N ssq slots, units are (mp, n) pairs."""

    needsPairShuffle = True

    def __init__(self, emitter, vgprTiles):
        self.emitter = emitter
        self.vgprTiles = vgprTiles
        self.numSsqSlots = emitter.T_N
        # Interleave M-pairs (2q, 2q+1) per column so both 64B halves of each 128B
        # residual line are requested back to back instead of T_N pairs apart. When
        # T_M//2 is odd the partner half belongs to the neighbouring wave, so the
        # order stays column-fastest.
        self.interleavePairs = (emitter.T_M // 2) % 2 == 0
        self.numGammaBanks = 2 if self.interleavePairs else 1
        self.unitOrder = self._unitOrder()

    def _unitOrder(self):
        """Return the (mp, n) visit order over all units.

        Interleaved: for each M-pair-pair q and column tile n, visit (2q, n) then
        (2q+1, n). Otherwise column-fastest: mp = u // T_N, n = u % T_N. Either way mp
        ascends for each n, so the ssq summation order is unchanged.
        """
        e = self.emitter
        if not self.interleavePairs:
            return [(u // e.T_N, u % e.T_N) for u in range(e.numUnits)]
        order = []
        for q in range(e.T_M // 4):
            for n in range(e.T_N):
                order.append((2 * q, n))
                order.append((2 * q + 1, n))
        assert len(order) == e.numUnits, "pair order must cover every pair"
        return order

    def unitSlot(self, u):
        return self.unitCoords(u)[1]

    def gammaKey(self, u):
        return self.unitCoords(u)[0]

    def unitCoords(self, u):
        """Return (mp, n) of unit ``u`` in the order chosen by _unitOrder."""
        return self.unitOrder[u]

    def accSources(self, u):
        """Return the 8 (regIndex, isAgpr) accumulator sources in row order j=0..7."""
        e = self.emitter
        mp, n = self.unitCoords(u)
        sources = []
        for j in range(8):
            tileIdx = 2 * mp if j < 4 else 2 * mp + 1
            tile = self.vgprTiles[n * e.T_M + tileIdx]
            reg = tile.regList.indices[j % 4]
            sources.append((reg, tile.regList.pool != e.writer.vgprPool))
        return sources

    def persistentAttrs(self):
        return ("laneV", "cV", "gV", "waveIdV", "waveMV", "waveNV",
                "rowOriginV", "colOriginV", "colBaseV", "colBaseDV", "pairRowOffV",
                "g4V", "oobV", "gammaReadBaseV")

    def emitAddrConstants(self, module):
        """Per-lane residual-address constants (precompute once per emission).

        Faithful to the #eoff column-major model of @mega_fused_epilogue: the
        residual byte address is (column*rowExtent + row) * 2. colBase, pairRowOff
        and g4 are the index-independent lane terms; the per-load code adds the
        pair/tile/column terms on top. These VGPRs are freed in the emitter teardown.
        """
        e = self.emitter
        vgprPool = e.writer.vgprPool
        e.colBaseV    = vgprPool.checkOut(1, tag="rms_colBase")
        e.colBaseDV   = vgprPool.checkOut(1, tag="rms_colBaseD")
        e.pairRowOffV = vgprPool.checkOut(1, tag="rms_pairRowOff")
        e.g4V         = vgprPool.checkOut(1, tag="rms_g4")
        e.oobV        = vgprPool.checkOut(1, tag="rms_oob")
        module.addComment1("residual load constants: colBase, colBaseD, pairRowOff, g4, oob.")
        # colBase = colOrigin + c (the lane's column within the wave tile).
        module.add(VAddU32(vgpr(e.colBaseV), vgpr(e.colOriginV), vgpr(e.cV),
                           comment="colBase = colOrigin + c."))
        # colBaseD = waveN*(T_N*16) + c: tile-relative (SrdD folds WG1*MT1, so the
        # D store must NOT re-add it, unlike the residual store's global colBase).
        cwSpanN = e.T_N * 16
        tmpColD = vgprPool.checkOut(1, tag="rms_colBaseDTmp")
        module.add(VMovB32(dst=vgpr(tmpColD), src=cwSpanN, comment=f"T_N*16={cwSpanN}."))
        module.add(VMulLOU32(dst=vgpr(e.colBaseDV), src0=vgpr(tmpColD), src1=vgpr(e.waveNV),
                             comment="colBaseD = waveN * (T_N*16)."))
        vgprPool.checkIn(tmpColD)
        module.add(VAddU32(vgpr(e.colBaseDV), vgpr(e.colBaseDV), vgpr(e.cV),
                           comment="colBaseD = waveN*(T_N*16) + c (tile-relative; SrdD folds WG1*MT1)."))
        # pairRowOff = (g&1)*16 + (g>>1)*8: tileSel*16 + half*8.
        tmpV = vgprPool.checkOut(1, tag="rms_pairRowTmp")
        module.add(VAndB32(dst=vgpr(tmpV), src0=vgpr(e.gV), src1=1, comment="tileSel = g & 1."))
        module.add(VLShiftLeftB32(dst=vgpr(tmpV), shiftHex=hex(4), src=vgpr(tmpV),
                                  comment="tileSel * 16."))
        module.add(VLShiftRightB32(dst=vgpr(e.pairRowOffV), shiftHex=hex(1), src=vgpr(e.gV),
                                   comment="half = g >> 1."))
        module.add(VLShiftLeftB32(dst=vgpr(e.pairRowOffV), shiftHex=hex(3),
                                  src=vgpr(e.pairRowOffV), comment="half * 8."))
        module.add(VAddU32(vgpr(e.pairRowOffV), vgpr(e.pairRowOffV), vgpr(tmpV),
                           comment="pairRowOff = tileSel*16 + half*8."))
        vgprPool.checkIn(tmpV)
        # g4 = g * 4 (native-tile row-group offset).
        module.add(VLShiftLeftB32(dst=vgpr(e.g4V), shiftHex=hex(2), src=vgpr(e.gV),
                                  comment="g4 = g * 4."))
        # oob = BufferOOB sentinel address that clamps an OOB row to a no-op load.
        module.add(VMovB32(dst=vgpr(e.oobV), src="BufferOOB", comment="oob = BufferOOB sentinel."))

    def emitWideRowBase(self, module, dstV, u):
        e = self.emitter
        mp, _ = self.unitCoords(u)
        module.add(VAddU32(vgpr(dstV), vgpr(e.rowOriginV), vgpr(e.pairRowOffV),
                           comment="rowBaseP = rowOrigin + pairRowOff."))
        e._vaddImm(module, dstV, dstV, (2 * mp) * 16, f"rowBaseP += m*16 (m={2 * mp}).")

    def emitNarrowRowBase(self, module, dstV, u, half):
        e = self.emitter
        mp, _ = self.unitCoords(u)
        ti = 2 * mp + half
        module.add(VAddU32(vgpr(dstV), vgpr(e.rowOriginV), vgpr(e.g4V),
                           comment="rowBaseT = rowOrigin + g4."))
        e._vaddImm(module, dstV, dstV, ti * 16, f"rowBaseT += ti*16 (ti={ti}).")

    def residualCol(self, u):
        return self.emitter.colBaseV, self.unitCoords(u)[1] * 16

    def dCol(self, u):
        return self.emitter.colBaseDV, self.unitCoords(u)[1] * 16

    def emitGammaReadBase(self, module):
        """Precompute this lane's gamma LDS read base byte offset.

        Faithful to the #rowg/#goff gamma LDS index of @mega_fused_body (MLIR
        280-284, 354-355): LDS[idx] = gamma[WorkGroup0*MT0 + idx] with
        idx = waveM*(T_M*16) + m*16 + g*4 + k. The index-independent per-lane
        term is waveM*(T_M*16) + g*4; its byte offset is that * 2 (bf16). The
        body adds the m*32 tile term per pair. Freed in the emitter teardown.
        """
        e = self.emitter
        vgprPool = e.writer.vgprPool
        if e.gammaReadBaseV is None:
            e.gammaReadBaseV = vgprPool.checkOut(1, tag="rms_gammaReadBase")
        module.addComment1("gamma read base = (waveM*(T_M*16) + g*4) * 2 (bf16 LDS byte offset).")
        rwSpanM = e.T_M * 16
        # waveMTerm = waveM * (T_M*16): shift when power of two, else materialized mul.
        if rwSpanM & (rwSpanM - 1) == 0:
            module.add(VLShiftLeftB32(dst=vgpr(e.gammaReadBaseV),
                                      shiftHex=hex(int(math.log2(rwSpanM))),
                                      src=vgpr(e.waveMV),
                                      comment=f"waveMTerm = waveM * {rwSpanM} (T_M*16)."))
        else:
            tmpV = vgprPool.checkOut(1, tag="rms_gammaReadTmp")
            module.add(VMovB32(dst=vgpr(tmpV), src=rwSpanM, comment=f"T_M*16={rwSpanM}."))
            module.add(VMulLOU32(dst=vgpr(e.gammaReadBaseV), src0=vgpr(tmpV), src1=vgpr(e.waveMV),
                                 comment=f"waveMTerm = waveM * {rwSpanM} (T_M*16)."))
            vgprPool.checkIn(tmpV)
        module.add(VAddU32(vgpr(e.gammaReadBaseV), vgpr(e.gammaReadBaseV), vgpr(e.g4V),
                           comment="+= g4 (native row-group offset)."))
        module.add(VLShiftLeftB32(dst=vgpr(e.gammaReadBaseV), shiftHex=hex(1),
                                  src=vgpr(e.gammaReadBaseV),
                                  comment="gamma read base byte offset = idx * 2 (bf16)."))
        if e.gammaLdsBase:
            module.add(VAddU32(vgpr(e.gammaReadBaseV), vgpr(e.gammaReadBaseV), e.gammaLdsBase,
                               comment=f"+= gammaLdsBase({e.gammaLdsBase})."))

    def issueGammaLds(self, module, g8Bank, key):
        """Issue the pair's two gamma DS reads early; ``key`` is the pair index mp.

        Called by _emitBody on the first visit to each mp, reading into that mp's parity
        bank (the single bank when not interleaved). Latency is hidden by the
        accumulator-stage, residual-cvt, H8 add, and ssq tree that follow before the
        matching wait in _waitConvertGammaLds on the first unit of the pair.
        """
        e = self.emitter
        vgprPool = e.writer.vgprPool
        m = 2 * key
        gldsAddr = vgprPool.checkOut(1, tag="rms_gldsAddr")
        e._vaddImm(module, gldsAddr, e.gammaReadBaseV, m * 16 * 2,
                   f"gldsAddr = gammaReadBase + m*32 (m={m}).")
        module.add(DSLoadB64(dst=vgpr(g8Bank + 0, 2), src=vgpr(gldsAddr),
                             ds=DSModifiers(offset=0), comment="tile m gamma: 4 bf16 (2 dwords)."))
        module.add(DSLoadB64(dst=vgpr(g8Bank + 4, 2), src=vgpr(gldsAddr),
                             ds=DSModifiers(offset=32), comment="tile m+1 gamma: 4 bf16 (+16 rows)."))
        vgprPool.checkIn(gldsAddr)

    def emitPartialReduction(self, module, ssqBase):
        """Fold the row groups with permlane swaps, then write the partials via LDS."""
        self.emitRowGroupReduce(module, ssqBase)
        self.emitter._writePartials(module, ssqBase)

    def emitRowGroupReduce(self, module, ssqBase):
        """Intra-wave row-group XOR butterfly over ssqAcc (permlane, no LDS).

        Faithful port of the updated @combine_rowgroups. Each ssqAcc[n] holds
        this lane's per-row-group partial sum of H^2; a 2-round XOR butterfly
        over the waveSize//mfmaN = 4 row-groups (g = lane>>4) folds them so every
        lane ends with the wavefront-complete partial for its column. Round 0
        exchanges with lane^16 via v_permlane16_swap_b32, round 1 with lane^32 via
        v_permlane32_swap_b32: the swap places the partner lane's value in the dst
        register while the src copy keeps this lane's own value, so dst+src is the
        pair reduction. Pure VALU -- no ds_bpermute, no s_waitcnt.
        """
        e = self.emitter
        module.addComment0("intra-wave row-group XOR butterfly over ssqAcc (permlane, no LDS).")
        vgprPool = e.writer.vgprPool
        numRounds = int(math.log2(e.waveSize // e.mfmaN))
        assert numRounds == 2, "combine_rowgroups butterfly expects waveSize/mfmaN == 4 (2 rounds)"
        swaps = [VPermlane16SwapB32, VPermlane32SwapB32]
        tmp = vgprPool.checkOut(1, tag="rms_combTmp")
        for n in range(self.numSsqSlots):
            s = ssqBase + n
            for r in range(numRounds):
                xorW = e.mfmaN << r
                module.add(VMovB32(dst=vgpr(tmp), src=vgpr(s),
                                   comment=f"ssqAcc[{n}] own copy (round {r})."))
                # gfx950 hazard: v_mov above writes tmp, which this v_permlane*_swap
                # reads; requires >= 2 wait states (hand-written asm is not auto-fixed).
                module.add(SNop(waitState=1, comment="gfx950 hazard: VALU write (v_mov) -> v_permlane read needs 2 wait states"))
                module.add(swaps[r](dst=vgpr(s), src=vgpr(tmp),
                                    comment=f"ssqAcc[{n}] <- partner^{xorW}; tmp keeps own."))
                module.add(VAddF32(dst=vgpr(s), src0=vgpr(s), src1=vgpr(tmp),
                                   comment=f"ssqAcc[{n}] = own + partner^{xorW}."))
        vgprPool.checkIn(tmp)


class CmsRmsLayout:
    """Layout of a CMS kernel (MT256x256, 2x2 waves, SourceSwap, VW=8).

    A lane owns 8 contiguous M rows (rb8 + j) of 32 N columns; unit u is column
    colOrigin + 32*g + u, and ssq slot(u) = u. Waves are linearized M-fastest
    (waveId = waveN*2 + waveM).
    """

    needsPairShuffle = False
    # One gamma key (always 0) serves every unit, so a single bank suffices.
    numGammaBanks = 1
    numUnits = 32
    numSsqSlots = 32
    # Slots each lane keeps after the reduce-scatter, and their LDS stride (256 threads * 4 bytes).
    numPublishedSlots = 8
    slotStrideBytes = 256 * 4

    def __init__(self, emitter):
        kernel = emitter.kernel
        assert kernel["MacroTile0"] == 256 and kernel["MacroTile1"] == 256, \
            "cms rms layout requires macro tile 256x256"
        assert list(kernel["MIWaveGroup"]) == [2, 2], "cms rms layout requires MIWaveGroup [2, 2]"
        assert kernel["VectorWidthA"] == 8 and kernel["VectorWidthB"] == 8, \
            "cms rms layout requires VectorWidthA == VectorWidthB == 8"
        assert kernel["SourceSwap"], "cms rms layout requires SourceSwap"
        assert not emitter.useMxfp8, "cms rms layout does not support mxfp8"
        self.emitter = emitter

    def unitSlot(self, u):
        return u

    def gammaKey(self, u):
        return 0

    def accSources(self, u):
        """Return the 8 (regIndex, isAgpr) sources of column ``u`` in row order j=0..7.

        AGPR index is tIdx + 4*j + 32*vw1; indices past maxLimitAgprs live in the VGPR
        spill area right after the VALU C base.
        """
        states = self.emitter.writer.states
        tIdx, vw1 = u // 8, u % 8
        sources = []
        for j in range(8):
            src = tIdx + 4 * j + 32 * vw1
            if src < states.maxLimitAgprs:
                sources.append((src, True))
                continue
            sources.append((states.c.startVgprValu + (src - states.maxLimitAgprs), False))
        return sources

    def persistentAttrs(self):
        return ("laneV", "cV", "gV", "waveIdV", "waveMV", "waveNV",
                "rowOriginV", "colOriginV", "rb8V", "cbResV", "cb2V", "oobV", "gammaReadBaseV")

    def emitAddrConstants(self, module):
        """Compute rb8 (row base of the 8 owned rows), global/tile-relative column bases, oob."""
        e = self.emitter
        vgprPool = e.writer.vgprPool
        e.rb8V   = vgprPool.checkOut(1, tag="rms_rb8")
        e.cbResV = vgprPool.checkOut(1, tag="rms_cbRes")
        e.cb2V   = vgprPool.checkOut(1, tag="rms_cb2")
        e.oobV   = vgprPool.checkOut(1, tag="rms_oob")
        module.addComment1("cms residual load constants: rb8, cbRes, cb2, oob.")
        module.add(VLShiftLeftB32(dst=vgpr(e.rb8V), shiftHex=hex(3), src=vgpr(e.cV),
                                  comment="8*c."))
        module.add(VAddU32(vgpr(e.rb8V), vgpr(e.rowOriginV), vgpr(e.rb8V),
                           comment="rb8 = rowOrigin + 8*c (M base of the 8 contiguous rows)."))
        tmpV = vgprPool.checkOut(1, tag="rms_gOff")
        module.add(VLShiftLeftB32(dst=vgpr(tmpV), shiftHex=hex(5), src=vgpr(e.gV),
                                  comment="g * 32."))
        module.add(VAddU32(vgpr(e.cbResV), vgpr(e.colOriginV), vgpr(tmpV),
                           comment="cbRes = colOrigin + g*32 (global N base)."))
        # SrdD already folds WG1*MT1, so the D column base is tile-relative.
        module.add(VLShiftLeftB32(dst=vgpr(e.cb2V), shiftHex=hex(7), src=vgpr(e.waveNV),
                                  comment="waveN * 128."))
        module.add(VAddU32(vgpr(e.cb2V), vgpr(e.cb2V), vgpr(tmpV),
                           comment="cb2 = waveN*128 + g*32 (D tile-relative N base)."))
        vgprPool.checkIn(tmpV)
        module.add(VMovB32(dst=vgpr(e.oobV), src="BufferOOB", comment="oob = BufferOOB sentinel."))

    def emitWideRowBase(self, module, dstV, u):
        e = self.emitter
        e._vaddImm(module, dstV, e.rb8V, 0, "rowBaseP = rb8.")

    def emitNarrowRowBase(self, module, dstV, u, half):
        e = self.emitter
        e._vaddImm(module, dstV, e.rb8V, 4 * half, f"rowBaseT = rb8 + {4 * half}.")

    def residualCol(self, u):
        return self.emitter.cbResV, u

    def dCol(self, u):
        return self.emitter.cb2V, u

    def emitGammaReadBase(self, module):
        """Gamma LDS read base byte offset = gammaLdsBase + waveM*256 + c*16."""
        e = self.emitter
        vgprPool = e.writer.vgprPool
        if e.gammaReadBaseV is None:
            e.gammaReadBaseV = vgprPool.checkOut(1, tag="rms_gammaReadBase")
        module.addComment1("gamma read base = waveM*256 + c*16 (+ gammaLdsBase).")
        module.add(VLShiftLeftB32(dst=vgpr(e.gammaReadBaseV), shiftHex=hex(8), src=vgpr(e.waveMV),
                                  comment="waveM * 256."))
        tmpV = vgprPool.checkOut(1, tag="rms_gammaReadTmp")
        module.add(VLShiftLeftB32(dst=vgpr(tmpV), shiftHex=hex(4), src=vgpr(e.cV),
                                  comment="c * 16."))
        module.add(VAddU32(vgpr(e.gammaReadBaseV), vgpr(e.gammaReadBaseV), vgpr(tmpV),
                           comment="gamma read base = waveM*256 + c*16."))
        vgprPool.checkIn(tmpV)
        if e.gammaLdsBase:
            module.add(VAddU32(vgpr(e.gammaReadBaseV), vgpr(e.gammaReadBaseV), e.gammaLdsBase,
                               comment=f"+= gammaLdsBase({e.gammaLdsBase})."))

    def issueGammaLds(self, module, g8Bank, key):
        """Read the 8 bf16 gamma rows of this lane (16 contiguous LDS bytes) as two b64."""
        e = self.emitter
        module.add(DSLoadB64(dst=vgpr(g8Bank + 0, 2), src=vgpr(e.gammaReadBaseV),
                             ds=DSModifiers(offset=0), comment="gamma rows 0..3: 4 bf16 (2 dwords)."))
        module.add(DSLoadB64(dst=vgpr(g8Bank + 4, 2), src=vgpr(e.gammaReadBaseV),
                             ds=DSModifiers(offset=8), comment="gamma rows 4..7: 4 bf16 (2 dwords)."))

    def emitPartialReduction(self, module, ssqBase):
        """Reduce ssqAcc over the 16 c lanes and 2 waveM waves, then store one column per thread.

        The 16 c lanes are reduce-scattered over c bits 0..1 in registers (each lane ends
        with 8 quad-summed columns), published to LDS, and transposed so thread t sums
        the 8 holders of WG-local column t. Cheaper than an all-reduce that leaves 16
        redundant copies and stores from only a few writer lanes.
        """
        module.addComment0("reduce-scatter over c bits 0..1, LDS transpose, one partial store per thread.")
        self._quadReduceScatter(module, ssqBase)
        self._publishQuadPartials(module, ssqBase)
        self._storeColumnPartials(module)

    def _quadReduceScatter(self, module, ssqBase):
        """Two quad_perm reduce-scatter rounds (c bit 0, then bit 1); VALU only, no LDS.

        Afterwards ssqAcc[j], j = 0..7, holds the sum over the 4 lanes sharing c bits 2..3
        for slot u = 16*b0 + 8*b1 + j, with b0 = c & 1 and b1 = (c >> 1) & 1.
        """
        e = self.emitter
        vgprPool = e.writer.vgprPool
        sgprPool = e.writer.sgprPool
        lsc = e.laneSGPRCount
        sendBank = vgprPool.checkOut(self.numSsqSlots // 2, tag="rms_rsSend")
        bitV = vgprPool.checkOut(1, tag="rms_rsBit")
        bitMask = sgprPool.checkOutAligned(lsc, lsc, tag="rms_rsMask", preventOverflow=False)
        numLive = self.numSsqSlots
        for bit, quadPerm in enumerate(([1, 0, 3, 2], [2, 3, 0, 1])):
            self._reduceScatterRound(module, ssqBase, sendBank, bitV, bitMask, numLive, bit, quadPerm)
            numLive //= 2
        sgprPool.checkIn(bitMask)
        vgprPool.checkIn(bitV)
        vgprPool.checkIn(sendBank)

    def _reduceScatterRound(self, module, ssqBase, sendBank, bitV, bitMask, numLive, bit, quadPerm):
        """Halve the live slots: keep the half matching c bit ``bit``, send the other half.

        Lanes with the bit set keep the upper half and send the lower; the quad partner
        (same bit flipped) does the opposite, so each lane adds its partner's copy of the
        half it kept.
        """
        e = self.emitter
        half = numLive // 2
        # Every cndmask precedes every DPP add, so each DPP-read register was written at
        # least `half` instructions earlier; this satisfies the gfx9 VALU write -> DPP
        # read hazard (2 wait states) without an s_nop.
        assert half >= 2, "reduce-scatter round too small to hide the dpp read hazard"
        lsc = e.laneSGPRCount
        module.add(VAndB32(dst=vgpr(bitV), src0=1 << bit, src1=vgpr(e.cV), comment=f"c & {1 << bit}."))
        module.add(VCmpNeU32(dst=sgpr(bitMask, lsc), src0=0, src1=vgpr(bitV),
                             comment=f"bitMask = c bit {bit} is set."))
        for i in range(half):
            module.add(VCndMaskB32(dst=vgpr(sendBank + i), src0=vgpr(ssqBase + i + half),
                                   src1=vgpr(ssqBase + i), src2=sgpr(bitMask, lsc),
                                   comment=f"send[{i}] = ssqAcc[{i} or {i + half}] (the half the quad partner keeps)."))
            module.add(VCndMaskB32(dst=vgpr(ssqBase + i), src0=vgpr(ssqBase + i),
                                   src1=vgpr(ssqBase + i + half), src2=sgpr(bitMask, lsc),
                                   comment=f"ssqAcc[{i}] = kept half (slot {i} or {i + half})."))
        for i in range(half):
            module.add(VAddF32(dst=vgpr(ssqBase + i), src0=vgpr(sendBank + i), src1=vgpr(ssqBase + i),
                               dpp=DPPModifiers(quad_perm=quadPerm),
                               comment=f"ssqAcc[{i}] += quad partner's send[{i}] (c bit {bit} flipped)."))

    def _publishQuadPartials(self, module, ssqBase):
        """Write ssqAcc[0..7] of every thread to LDS at dword j*256 + tid (conflict-free)."""
        e = self.emitter
        vgprPool = e.writer.vgprPool
        tidX4 = vgprPool.checkOut(1, tag="rms_pwTidX4")
        module.add(VLShiftLeftB32(dst=vgpr(tidX4), shiftHex=hex(2), src=vgpr("Serial"),
                                  comment="tid*4 (LDS byte base)."))
        # WAR barrier: gamma LDS reads done before reusing LDS base 0.
        module.add(e.writer._syncThreads(e.kernel, "cms partials: ensure gamma LDS free before ssq write."))
        for j in range(self.numPublishedSlots):
            module.add(DSStoreB32(dstAddr=vgpr(tidX4), src=vgpr(ssqBase + j),
                                  ds=DSModifiers(offset=j * self.slotStrideBytes),
                                  comment=f"lds[j={j}, tid] = ssqAcc[{j}]."))
        module.add(SWaitCnt(dscnt=0, comment="wait ssq LDS writes."))
        module.add(e.writer._syncThreads(e.kernel, "cms partials: publish ssq LDS."))
        vgprPool.checkIn(tidX4)

    def _storeColumnPartials(self, module):
        """Thread t sums the 8 LDS holders of WG-local column t and stores it to partialBuf.

        The LDS loads are in flight while the SRD and column address are built.
        """
        e = self.emitter
        vgprPool = e.writer.vgprPool
        sgprPool = e.writer.sgprPool
        lsc = e.laneSGPRCount
        module.addComment0("transpose-sum: thread t owns WG-local column t.")
        baseV = vgprPool.checkOut(1, tag="rms_holderBase")
        self._holderBaseBytes(module, baseV)
        numHolders = e.G_M * 4
        holders = vgprPool.checkOut(numHolders, tag="rms_holders")
        self._issueHolderLoads(module, baseV, holders)
        partialSrd, ntV = e._buildPartialSrd(module)
        colV = vgprPool.checkOut(1, tag="rms_pwCol")
        addrV = vgprPool.checkOut(1, tag="rms_pwAddr")
        colMask = sgprPool.checkOutAligned(lsc, lsc, tag="rms_pwColMask", preventOverflow=False)
        self._columnIndex(module, colV)
        self._sumHolders(module, holders, numHolders)
        e._storePartialAt(module, holders, "column sum", ntV, partialSrd, colV, addrV, colMask)
        module.add(SWaitCnt(vscnt=0, comment="drain partialBuf stores."))
        sgprPool.checkIn(colMask)
        vgprPool.checkIn(addrV)
        vgprPool.checkIn(colV)
        vgprPool.checkIn(ntV)
        sgprPool.checkIn(partialSrd)
        vgprPool.checkIn(holders)
        vgprPool.checkIn(baseV)

    def _holderBaseBytes(self, module, baseV):
        """baseV = LDS byte address of this thread's holder with q = 0, wm = 0.

        With t = Serial: u = t & 31, g = (t >> 5) & 3, wN = t >> 7, j = u & 7, b0 = u >> 4,
        b1 = (u >> 3) & 1. The holder lane c = b0 + 2*b1 + 4*q of wave wN*2 + wm is at
        dword j*256 + 64*(wN*2 + wm) + 16*g + c, so
        base = (j*256 + 128*wN + 16*g + b0 + 2*b1) * 4; q and wm become DS immediates.
        The five fields occupy disjoint bits, so they are combined with v_or.
        """
        vgprPool = self.emitter.writer.vgprPool
        serial = vgpr("Serial")
        tmp = vgprPool.checkOut(1, tag="rms_holderTmp")
        module.add(VAndB32(dst=vgpr(baseV), src0=7, src1=serial, comment="j = t & 7."))
        module.add(VLShiftLeftB32(dst=vgpr(baseV), shiftHex=hex(10), src=vgpr(baseV),
                                  comment="j * 256 dwords in bytes."))
        module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(2), src=serial, comment="t << 2."))
        module.add(VAndB32(dst=vgpr(tmp), src0=0x200, src1=vgpr(tmp), comment="wN * 128 dwords in bytes."))
        module.add(VOrB32(dst=vgpr(baseV), src0=vgpr(baseV), src1=vgpr(tmp), comment="base |= wN term."))
        module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(1), src=serial, comment="t << 1."))
        module.add(VAndB32(dst=vgpr(tmp), src0=0xC0, src1=vgpr(tmp), comment="g * 16 dwords in bytes."))
        module.add(VOrB32(dst=vgpr(baseV), src0=vgpr(baseV), src1=vgpr(tmp), comment="base |= g term."))
        module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=hex(2), src=serial, comment="t >> 2."))
        module.add(VAndB32(dst=vgpr(tmp), src0=4, src1=vgpr(tmp), comment="b0 in bytes."))
        module.add(VOrB32(dst=vgpr(baseV), src0=vgpr(baseV), src1=vgpr(tmp), comment="base |= b0 term."))
        module.add(VAndB32(dst=vgpr(tmp), src0=8, src1=serial, comment="2*b1 dwords in bytes."))
        module.add(VOrB32(dst=vgpr(baseV), src0=vgpr(baseV), src1=vgpr(tmp), comment="base |= b1 term."))
        vgprPool.checkIn(tmp)

    def _issueHolderLoads(self, module, baseV, holders):
        """Load the G_M waves x 4 quads of holders; the holder of (wm, q) is at +wm*256 + q*16 bytes."""
        e = self.emitter
        waveStrideBytes = e.waveSize * 4
        for wm in range(e.G_M):
            for q in range(4):
                module.add(DSLoadB32(dst=vgpr(holders + wm * 4 + q), src=vgpr(baseV),
                                     ds=DSModifiers(offset=wm * waveStrideBytes + q * 16),
                                     comment=f"lds holder: waveM {wm}, c quad {q}."))

    def _columnIndex(self, module, colV):
        e = self.emitter
        with e.writer.allocTmpSgpr(1, tag="rms_colBaseS") as colBase:
            module.add(SMulI32(dst=sgpr(colBase.idx), src0=sgpr("WorkGroup1"), src1=e.MT1,
                               comment="WorkGroup1 * MT1."))
            module.add(VAddU32(vgpr(colV), sgpr(colBase.idx), vgpr("Serial"),
                               comment="column = WorkGroup1*MT1 + tid (thread t owns WG-local column t)."))

    def _sumHolders(self, module, holders, numHolders):
        """Sum the holders into holders[0], draining the LDS reads with a staggered wait."""
        for k in range(1, numHolders):
            module.add(SWaitCnt(dscnt=numHolders - 1 - k,
                                comment=f"drain to {numHolders - 1 - k} outstanding lds reads."))
            module.add(VAddF32(dst=vgpr(holders), src0=vgpr(holders), src1=vgpr(holders + k),
                               comment=f"column sum += holder {k}."))
