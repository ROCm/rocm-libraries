# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Instruction emitter for LogicalScheduler.

Converts the logical schedule (EmittedModule chains) into concrete GPU
instructions by dispatching each opType to its emit method.
"""

from __future__ import annotations

from Tensile.Components.Subtile.Kernel import emitMfmaInstruction
from Tensile.Components.Subtile.SubtileGREmit import (
    emitSingleBufferLoad, globalReadPtrUpdates, globalReadLDSBufferSwap,
)
from Tensile.Components.Subtile.SubtileLREmit import (
    emitSingleDsRead, localReadLDSBufferSwap,
)
from Tensile.Components.Subtile.SubtileScaleEmit import (
    globalReadDoScaleSubtile, globalReadScalePtrUpdates,
)
from rocisa.code import Module
from rocisa.instruction import (
    SWaitCnt, SBarrier, DSLoadB32, SCmpEQU32, SCmpLeU32,
    SCBranchSCC1, SMovB32, VAddU32, VAndB32, VCmpGEI32, VCmpGTI32, VCmpLeI32,
    VCmpLtI32, VCndMaskB32, VLShiftLeftB32, VLShiftRightB32, VMovB32, VSubI32,
)
from rocisa.instruction import SWaitTensorcnt
from rocisa.container import vgpr, sgpr, DSModifiers, ContinuousRegister
from rocisa.code import Label


class SWaitCntEx(SWaitCnt):
    """SWaitCnt with adjustVmcnt flag for the instruction scheduler post-pass."""
    def __init__(self, adjustVmcnt=True, **kwargs):
        super().__init__(**kwargs)
        self._adjustVmcnt = adjustVmcnt

    @property
    def adjustVmcnt(self):
        return self._adjustVmcnt

    def __deepcopy__(self, memo):
        return SWaitCntEx(
            adjustVmcnt=self._adjustVmcnt,
            vlcnt=self.vlcnt, vscnt=self.vscnt,
            dscnt=self.dscnt, kmcnt=self.kmcnt,
            comment=self.comment)


class InstructionEmitter:
    """Emits GPU instructions for each opType in the LogicalScheduler output.

    VGPR tile indexing uses placement-level tile maps (tileId → vgprTileId)
    set by assign_vgpr_tiles(). Per-tensor VGPR tile lists are indexed by
    vgprTileId. All tensors (A, B, SA, SB) use the same tile-map approach.
    """

    def __init__(self, writer, kernel, config,
                 tileInfoA, tileInfoB, dtileInfo,
                 vgprTilesA, vgprTilesB,
                 scaleTileInfoA=None, scaleTileInfoB=None,
                 vgprTilesSA=None, vgprTilesSB=None,
                 tensorParametersA=None, tensorParametersB=None):
        self.writer = writer
        self.kernel = kernel
        self.config = config
        self.tileInfoA = tileInfoA
        self.tileInfoB = tileInfoB
        self.dtileInfo = dtileInfo
        self.vgprTilesA = vgprTilesA
        self.vgprTilesB = vgprTilesB
        self.vgprTilesSA = vgprTilesSA or []
        self.vgprTilesSB = vgprTilesSB or []
        self.tensorParametersMap = {}
        if tensorParametersA is not None:
            self.tensorParametersMap['A'] = tensorParametersA
        if tensorParametersB is not None:
            self.tensorParametersMap['B'] = tensorParametersB

        # Derived state
        self.hasScale = scaleTileInfoA is not None and scaleTileInfoB is not None
        self.subtileShapeK = tileInfoA.subtileShape[1]
        self.tileInfoMap = {'A': tileInfoA, 'B': tileInfoB}
        if self.hasScale:
            self.tileInfoMap['SA'] = scaleTileInfoA
            self.tileInfoMap['SB'] = scaleTileInfoB

        # Dispatch table — unroll_iter is passed for mfma/lr
        self._dispatch = {
            'mfma':         lambda em, ui: self.emit_mfma(em.source, ui),
            'lr':           lambda em, ui: self.emit_lr(em.source, ui),
            'gr':           lambda em, ui: self.emit_gr(em.source),
            'wait_gr':      lambda em, ui: self.emit_wait_gr(em.source),
            'wait_lr':      lambda em, ui: self.emit_wait_lr(),
            'sync':         lambda em, ui: self.emit_sync(),
            'lr_inc':           lambda em, ui: self.emit_lr_inc(em.source),
            'gr_inc':           lambda em, ui: self.emit_gr_inc(em.source),
            'skip':             lambda em, ui: self.emit_skip(em.source),
            'mask_k':       lambda em, ui: self.emit_mask_k(em.source),
            'inline':       lambda em, ui: self.emit_inline(em.source),
        }

        # Sentinel for the long-lived per-lane diff vgpr. Set by
        # emit_mask_k_init, consumed by every emit_mask_k call in the tail body.
        self._tail_vDiff = None

    def emit_mfma(self, placement, unroll_iter=0):
        """Emit MFMA instructions from MFMAPlacement."""
        module = Module()
        subIterK = placement.subIterK
        tile_maps = {t: placement.vgpr_tile_maps[t][unroll_iter]
                     for t in placement.vgpr_tile_maps}

        for a in placement.tileA.tileId_list:
            for b in placement.tileB.tileId_list:
                groupA = (a // self.config.lrA.mn) * self.config.lrA.mn
                groupB = (b // self.config.lrB.mn) * self.config.lrB.mn
                aTile = self.vgprTilesA[tile_maps['A'][groupA]]
                bTile = self.vgprTilesB[tile_maps['B'][groupB]]
                dTile = self.dtileInfo.vgprTiles[a + b * self.dtileInfo.localMMATileGrid[0]]

                if self.hasScale:
                    scaleGroupA = (a // self.config.lrSA.mn) * self.config.lrSA.mn
                    scaleGroupB = (b // self.config.lrSB.mn) * self.config.lrSB.mn
                    scaleATile = self.vgprTilesSA[tile_maps['SA'][scaleGroupA]]
                    scaleBTile = self.vgprTilesSB[tile_maps['SB'][scaleGroupB]]
                    scaleAVgpr = next(iter(scaleATile))
                    scaleBVgpr = next(iter(scaleBTile))
                    mShapeA = self.tileInfoMap['SA'].lrSubtileShape[0]
                    mShapeB = self.tileInfoMap['SB'].lrSubtileShape[0]
                    kShapeA = self.tileInfoMap['SA'].lrSubtileShape[1]
                    kShapeB = self.tileInfoMap['SB'].lrSubtileShape[1]
                    sAsel = (a % mShapeA) + mShapeA * (subIterK % kShapeA)
                    sBsel = (b % mShapeB) + mShapeB * (subIterK % kShapeB)
                else:
                    scaleAVgpr = scaleBVgpr = -1
                    sAsel = sBsel = 0

                module.add(emitMfmaInstruction(
                    self.writer, self.kernel, aTile, bTile, dTile, dTile,
                    scaleAVgpr=scaleAVgpr, scaleBVgpr=scaleBVgpr,
                    scaleAsel=sAsel, scaleBsel=sBsel,
                    comment=f"MFMA C[{a},{b}] += A[{a},K={subIterK}] * B[{b},K={subIterK}]"))
        return list(module.flatitems())

    def emit_lr(self, placement, unroll_iter=0):
        """Emit LR (ds_read) instructions from LRPlacement."""
        module = Module()
        tensor = placement.tensor
        tile_map = placement.vgpr_tile_map[unroll_iter] if placement.vgpr_tile_map else {}

        if tensor in ('A', 'B'):
            ti = self.tileInfoMap[tensor]
            vgprTiles = self.vgprTilesA if tensor == 'A' else self.vgprTilesB
            lrGran = self.config.lrA if tensor == 'A' else self.config.lrB
            for tileId in range(placement.tiles.tileId_start, placement.tiles.tileId_end, lrGran.mn):
                for k in range(placement.tiles.subIterK_start, placement.tiles.subIterK_end, lrGran.k):
                    subtileK = k // self.subtileShapeK
                    subIterK_within = k % self.subtileShapeK
                    dstTile = vgprTiles[tile_map[tileId]]
                    swizzled = self.writer.states.subtileLdsSwizzle
                    module.add(emitSingleDsRead(
                        ti, tileId, subtileK, subIterK_within, dstTile, swizzled=swizzled))
        elif tensor in ('SA', 'SB'):
            tc = 'MXSA' if tensor == 'SA' else 'MXSB'
            ti = self.tileInfoMap[tensor]
            lrGran = self.config.lrSA if tensor == 'SA' else self.config.lrSB
            vgprTilesScale = self.vgprTilesSA if tensor == 'SA' else self.vgprTilesSB
            for tileId in range(placement.tiles.tileId_start, placement.tiles.tileId_end, lrGran.mn):
                scaleGroupIdx = tileId // lrGran.mn
                groupKey = scaleGroupIdx * lrGran.mn
                kGroupIdx = placement.tiles.subIterK_start // ti.lrSubtileShape[1]
                numKGroups = ti.lrLocalSubtileGrid[1]
                dsOffset = int(ti.lrSubtileSize) * (scaleGroupIdx * numKGroups + kGroupIdx)
                vdst = next(iter(vgprTilesScale[tile_map[groupKey]]))
                module.add(DSLoadB32(
                    dst=vgpr(vdst),
                    src=vgpr(ti.sharedVgprLROffset[0]),
                    ds=DSModifiers(offset=dsOffset),
                    comment=f"scale{tc}[group{scaleGroupIdx},K={placement.tiles.subIterK_start}]: load 4B from LDS"))
        return list(module.flatitems())

    def emit_gr(self, placement):
        """Emit GR (buffer_load) instructions from GRPlacement."""
        module = Module()
        tensor = placement.tensor
        if tensor in ('A', 'B'):
            ti = self.tileInfoMap[tensor]
            grGran = self.config.grA if tensor == 'A' else self.config.grB
            for tileId in range(placement.tiles.tileId_start, placement.tiles.tileId_end, grGran.mn):
                for k in range(placement.tiles.subIterK_start, placement.tiles.subIterK_end, grGran.k):
                    subtileK = k // self.subtileShapeK
                    module.add(emitSingleBufferLoad(ti, self.kernel, tileId, subtileK))
        elif tensor in ('SA', 'SB'):
            tc = 'MXSA' if tensor == 'SA' else 'MXSB'
            module.add(globalReadDoScaleSubtile(tc, self.writer, self.kernel))
        return list(module.flatitems())

    def emit_wait_gr(self, source):
        """Emit SWaitCnt for wait_gr from BaseOp with wait_gr_counts."""
        counts = source.wait_gr_counts
        if counts is None:
            return []
        
        if self.kernel.get("enableTDMA", False) and self.kernel.get("enableTDMB", False):
            tdmCnt = counts.A + counts.B + counts.SA + counts.SB
            return [SWaitTensorcnt(tensorcnt=tdmCnt,
                                   comment=f"Wait TDM (tensor_load_to_lds): A={counts.A} B={counts.B} SA={counts.SA} SB={counts.SB}")]

        # TODO. Hardcoded for now, but we should just get this from atomic emit codes (emitSingleBufferLoad, ...)
        grMap = {'A': max(1,int(1.0/self.tileInfoA.loadRatioGR)),
                 'B':  max(1,int(1.0/self.tileInfoB.loadRatioGR)),
                 'SA': 1, 
                 'SB': 1}  
        grCnt = (counts.A * grMap['A'] +
                 counts.B * grMap['B'] +
                 counts.SA * grMap['SA'] +
                 counts.SB * grMap['SB'])
        swait = SWaitCntEx(vlcnt=grCnt, vscnt=-1,
                           adjustVmcnt=source.adjustVmcnt,
                           comment=f"Wait GR (per-subIterK): A={counts.A} B={counts.B} SA={counts.SA} SB={counts.SB}")
        return [swait]

    def emit_wait_lr(self):
        return [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1,
                         comment="Wait for LR to complete")]

    def emit_sync(self):
        return [SBarrier(comment="Barrier")]

    def emit_inline(self, source):
        """Emit a writer-built Module supplied by an InlineModuleOp callback."""
        if source.build is None:
            return []
        mod = source.build(self)
        return list(mod.flatitems()) if mod is not None else []

    def emit_lr_inc(self, source):
        """Emit localReadLDSBufferSwap for a single tensor."""
        tensor = source.tensor
        tc = {'A': 'A', 'B': 'B', 'SA': 'MXSA', 'SB': 'MXSB'}.get(tensor, tensor)
        module = Module()
        module.add(localReadLDSBufferSwap(tc, self.writer, self.kernel))
        return list(module.flatitems())

    def emit_gr_inc(self, source):
        """Emit globalReadPtrUpdates + globalReadLDSBufferSwap for a single tensor."""
        tensor = source.tensor
        tc = {'A': 'A', 'B': 'B', 'SA': 'MXSA', 'SB': 'MXSB'}.get(tensor, tensor)
        module = Module()
        if tensor in ('SA', 'SB'):
            module.add(globalReadScalePtrUpdates(tc, self.writer, self.kernel))
        else:
            module.add(globalReadPtrUpdates(tc, self.writer, self.kernel))
        module.add(globalReadLDSBufferSwap(tc, self.writer, self.kernel))
        return list(module.flatitems())

    def emit_skip(self, source):
        """Emit skip guard: compare LoopCounterL and branch."""
        labelName = source.target if source.rawLabel else f"SkipTo{source.target}"
        skipLabel = Label(labelName, "")
        cmpMap = {"EQ": SCmpEQU32, "LE": SCmpLeU32}
        cmpCls = cmpMap[source.compare]
        module = Module()
        # SCMP inline range is -16..64 (signed); stage non-inline via scratch sgpr.
        if -16 <= source.value <= 64:
            module.add(cmpCls(
                src0=sgpr("LoopCounterL"), src1=source.value,
                comment=f"LoopCounter {source.compare} {source.value}?"))
        else:
            with self.writer.allocTmpSgpr(1) as litSgprInfo:
                litSgpr = litSgprInfo.idx
                module.add(SMovB32(
                    dst=sgpr(litSgpr), src=hex(source.value),
                    comment=f"stage literal {source.value} (non-inline) for cmp src1"))
                module.add(cmpCls(
                    src0=sgpr("LoopCounterL"), src1=sgpr(litSgpr),
                    comment=f"LoopCounter {source.compare} {source.value}?"))
        module.add(SCBranchSCC1(
            labelName=skipLabel.getLabelName(),
            comment=source.branchComment or f"skip to {source.target}"))
        return list(module.flatitems())

    def _mfma_K_constants(self):
        """Constants used by both mask emitters.

        Returns (numMIInUnroll, dividerFortidInK):
          * numMIInUnroll    = MI_M * MI_K / WavefrontSize — per-lane K chunk
            consumed by one MFMA call (contiguous K-elements held by each lane).
          * dividerFortidInK = MI_M — lanes with the same
            (Serial % WavefrontSize) / MI_M share the same K-position;
            WavefrontSize / MI_M = number of K-groups in the wave.
        """
        kernel = self.kernel
        MI_M             = kernel["MatrixInstM"]
        MI_K             = kernel["MatrixInstK"]
        waveSize         = kernel["WavefrontSize"]
        numMIInUnroll    = MI_M * MI_K // waveSize
        dividerFortidInK = MI_M
        return numMIInUnroll, dividerFortidInK

    def emit_mask_k_init(self):
        """Stage the per-subIterK invariants for emit_mask_k.

        Only two vgprs survive past this routine and into emit_mask_k:
          * self._tail_vDiff           — rem - laneK_0 (for subIterK=n the
            effective diff is diff - n*MatrixInstK, folded into cmp constants)
          * self._tail_boundaryMask[4] — BF16 only; precomputed boundary masks
            indexed by vgpr i, derived from d = rem % 8

        """
        numMIInUnroll, dividerFortidInK = self._mfma_K_constants()
        writer = self.writer
        module = Module()

        # Compute K per lane (numMIInUnroll, dividerFortidInK are power of 2)
        waveSize = self.kernel["WavefrontSize"]
        divLog2  = (dividerFortidInK).bit_length() - 1
        mulLog2  = (numMIInUnroll).bit_length() - 1

        kReg = writer.vgprPool.checkOut(1, "tail_kReg")
        module.add(VAndB32(
            dst=vgpr(kReg), src0=waveSize - 1, src1=vgpr("Serial"),
            comment=f"tail_kReg = Serial & {waveSize - 1} (Serial % {waveSize})"))
        module.add(VLShiftRightB32(
            dst=vgpr(kReg), shiftHex=divLog2, src=vgpr(kReg),
            comment=f"tail_kReg >>= {divLog2} (tail_kReg / {dividerFortidInK})"))

        # laneK_0 and diff = rem - laneK_0 are shared across all subIterK.
        workKVgpr = writer.vgprPool.checkOut(1, "tail_workK")
        self._tail_vDiff = writer.vgprPool.checkOut(1, "tail_vDiff")
        loopCounterName = writer.loopCounterName(
            self.kernel, writer.states.unrollIdx)
        module.add(VLShiftLeftB32(
            dst=vgpr(workKVgpr), shiftHex=mulLog2, src=vgpr(kReg),
            comment=f"laneK_0 = tail_kReg << {mulLog2} (tail_kReg * {numMIInUnroll})"))
        writer.vgprPool.checkIn(kReg)
        module.add(VSubI32(
            dst=vgpr(self._tail_vDiff),
            src0=sgpr(loopCounterName), src1=vgpr(workKVgpr),
            comment="diff = rem - laneK_0 (shared across all subIterK)"))
        writer.vgprPool.checkIn(workKVgpr)

        self._tail_boundaryMask = None
        laneSGPRCount = writer.states.laneSGPRCount
        if self.kernel["ProblemType"]["DataTypeA"].isBFloat16():
            halfMaskVgpr = writer.vgprPool.checkOut(1, "tail_halfMask")
            module.add(VMovB32(
                dst=vgpr(halfMaskVgpr), src="0x0000FFFF",
                comment="BF16 half-mask: keep K0 (low 16b), zero K1 (high 16b)"))

            # Precompute the boundary masks from d = rem % numMIInUnroll.
            # The boundary-mask pattern (which vgprs are full/half/zero)
            # depends only on rem % numMIInUnroll: laneK_0 is always a
            # multiple of numMIInUnroll, and (assuming MatrixInstK is a
            # multiple of numMIInUnroll) every "boundary" lane in every
            # subIterK has effective_diff ≡ rem (mod numMIInUnroll). So all
            # boundary lanes share the same mask pattern (one per vgpr held
            # per lane: numMIInUnroll // kStride masks).
            kStride = 2  # BF16: 2 K-elements packed per 32-bit vgpr
            assert numMIInUnroll % kStride == 0, \
                f"numMIInUnroll ({numMIInUnroll}) must be a multiple of kStride ({kStride})"
            numBoundaryMasks = numMIInUnroll // kStride
            vDLaneRem = writer.vgprPool.checkOut(1, "tail_vDLaneRem")
            module.add(VAndB32(
                dst=vgpr(vDLaneRem),
                src0=sgpr(loopCounterName), src1=numMIInUnroll - 1,
                comment=f"d = rem % {numMIInUnroll} (boundary-mask pattern depends only on this)"))
            self._tail_boundaryMask = [
                writer.vgprPool.checkOut(1, f"tail_boundaryMask{i}")
                for i in range(numBoundaryMasks)
            ]
            with writer.allocTmpSgpr(laneSGPRCount, alignment=laneSGPRCount) as tmpSgprInfo:
                maskSgpr = tmpSgprInfo.idx
                for i in range(numBoundaryMasks):
                    bm = self._tail_boundaryMask[i]
                    hiBound = i * kStride + kStride  # d < hiBound → half-keep
                    loBound = i * kStride + 1        # d < loBound → zero
                    module.add(VCmpLtI32(
                        dst=sgpr(maskSgpr, laneSGPRCount),
                        src0=vgpr(vDLaneRem), src1=hiBound,
                        comment=f"boundary[{i}]: d < {hiBound} ? halfKeep : full"))
                    module.add(VCndMaskB32(
                        dst=vgpr(bm),
                        src0=-1,
                        src1=vgpr(halfMaskVgpr),
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment=f"boundaryMask[{i}] = (d<{hiBound}) ? halfKeep : full"))
                    module.add(VCmpLtI32(
                        dst=sgpr(maskSgpr, laneSGPRCount),
                        src0=vgpr(vDLaneRem), src1=loBound,
                        comment=f"boundary[{i}]: d < {loBound} ? 0 : prev"))
                    module.add(VCndMaskB32(
                        dst=vgpr(bm), src0=vgpr(bm), src1=0,
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment=f"boundaryMask[{i}] = (d<{loBound}) ? 0 : prev"))
            writer.vgprPool.checkIn(halfMaskVgpr)
            writer.vgprPool.checkIn(vDLaneRem)

        return list(module.flatitems())

    def emit_mask_k(self, source):
        """Per-lane K-mask for one subIterK, applied to A/B tiles via V_AND_B32.

        Uses diff = rem - laneK_0 from emit_mask_k_init. For subIterK=n the
        effective per-lane diff is diff - n*MatrixInstK, folded into the
        cmp immediates (no per-call sub).

        BF16 (kStride=2 K positions per vgpr) builds a per-vgpr 3-state mask
        from the precomputed boundary masks (full / boundary[i] / zero).
        Non-BF16 (e.g. FP4) builds a single 2-state mask shared across all
        vgprs, assuming rem aligns to the per-lane K stride (true for
        rem=32 with FP4 MIK=128). A boundary inside a non-BF16 vgpr would
        need per-byte/nibble handling.
        """
        assert self._tail_vDiff is not None, \
            "emit_mask_k_init must run before emit_mask_k"

        writer = self.writer
        kernel = self.kernel
        subIterK = source.subIterK
        kBaseConst = subIterK * kernel["MatrixInstK"]

        laneSGPRCount = writer.states.laneSGPRCount
        isBF16 = kernel["ProblemType"]["DataTypeA"].isBFloat16()
        kStride = 2  # BF16: 2 elements packed per 32-bit vgpr (low=K0, high=K1)

        module = Module()

        def _unique_ids(key):
            m = source.vgpr_tile_map.get(key, [{}])[0]
            return sorted(set(m.values()))

        aIds, bIds = _unique_ids('A'), _unique_ids('B')
        # All A/B tiles share the same vgprs-per-lane count and K layout,
        # so the i-th vgpr of every tile gets the same mask.
        refTiles = self.vgprTilesA if aIds else self.vgprTilesB
        refIds = aIds or bIds
        vgprPerInUnroll = len(list(refTiles[refIds[0]])) if refIds else 0

        with writer.allocTmpSgpr(laneSGPRCount, alignment=laneSGPRCount) as tmpSgprInfo:
            maskSgpr = tmpSgprInfo.idx

            def _emit_cmp(cmpCls, literal, comment):
                # VOPC inline range is -16..64; stage out-of-range via scratch sgpr
                # (e.g. BF16 MI_K=32 subIterK>=2, or FP4 MI_K=128 subIterK>=1).
                if -16 <= literal <= 64:
                    module.add(cmpCls(
                        dst=sgpr(maskSgpr, laneSGPRCount),
                        src0=vgpr(self._tail_vDiff), src1=literal,
                        comment=comment))
                else:
                    with writer.allocTmpSgpr(1) as litSgprInfo:
                        litSgpr = litSgprInfo.idx
                        module.add(SMovB32(
                            dst=sgpr(litSgpr), src=hex(literal),
                            comment=f"stage literal {literal} (non-inline)"))
                        module.add(cmpCls(
                            dst=sgpr(maskSgpr, laneSGPRCount),
                            src0=vgpr(self._tail_vDiff), src1=sgpr(litSgpr),
                            comment=comment))

            if isBF16:
                # 3-way per (lane, subIterK), 2 cmps shared across all i:
                #   sFull = effective_diff_n >= numMIInUnroll → -1
                #   sZero = effective_diff_n <= 0            → 0
                #   else                                     → boundary[i]
                numMIInUnroll = vgprPerInUnroll * kStride
                fullLit = kBaseConst + numMIInUnroll - 1
                zeroLit = kBaseConst
                maskVgprs = [writer.vgprPool.checkOut(1, f"mask_k_msk{i}_k{subIterK}")
                             for i in range(vgprPerInUnroll)]
                _emit_cmp(VCmpGTI32, fullLit,
                          f"sFull: diff > {fullLit} (effective_diff_{subIterK} >= {numMIInUnroll})")
                for i in range(vgprPerInUnroll):
                    module.add(VCndMaskB32(
                        dst=vgpr(maskVgprs[i]),
                        src0=vgpr(self._tail_boundaryMask[i]), src1=-1,
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment=f"mask[{i}] = sFull ? full : boundary[{i}]"))
                _emit_cmp(VCmpLeI32, zeroLit,
                          f"sZero: diff <= {zeroLit} (effective_diff_{subIterK} <= 0)")
                for i in range(vgprPerInUnroll):
                    module.add(VCndMaskB32(
                        dst=vgpr(maskVgprs[i]), src0=vgpr(maskVgprs[i]), src1=0,
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment=f"mask[{i}] = sZero ? 0 : prev"))
            else:
                # 2-state shared mask: diff < kBaseConst+1 → 0, else -1.
                literal = kBaseConst + 1
                sharedMask = writer.vgprPool.checkOut(1, f"mask_k_msk_k{subIterK}")
                maskVgprs = [sharedMask] * vgprPerInUnroll
                _emit_cmp(VCmpLtI32, literal,
                          f"mask: diff < {literal} (laneK_{subIterK} >= rem)")
                module.add(VCndMaskB32(
                    dst=vgpr(sharedMask), src0=-1, src1=0,
                    src2=sgpr(maskSgpr, laneSGPRCount),
                    comment=f"mask = (diff < {literal}) ? 0 : -1"))

            for label, ids, tilesDict in (("A", aIds, self.vgprTilesA),
                                          ("B", bIds, self.vgprTilesB)):
                for tid in ids:
                    for i, v in enumerate(list(tilesDict[tid])):
                        module.add(VAndB32(
                            dst=vgpr(v), src0=vgpr(v), src1=vgpr(maskVgprs[i]),
                            comment=f"mask {label}[{i}] (K=[{i*kStride},{i*kStride+kStride-1}])"))

            # MX scale mask (FP4 only): reuse the A/B mask we just built.
            # The A/B mask for subIterK=base is binary per-lane:
            #   laneK_base >= rem → 0   (zero A/B[base]; also OK to zero scale
            #                            for both byte-pairs — see below)
            #   else              → -1  (keep)
            # Scale vgpr packs lrSA.k subIterK pairs (bytes 0..1 → base,
            # bytes 2..3 → base+1). Applying the base mask to the scale:
            #   • lane fails at base → scale zeroed → MFMAs at base, base+1
            #     see scale=0; A/B[base]=0 by the same mask, and A/B[base+1]
            #     is masked by its own emit_mask_k → both products = 0. ✓
            #   • lane passes at base, fails at base+1 → scale bytes 2..3 left
            #     stale, but A/B[base+1] is masked to 0 by its own emit_mask_k
            #     call → product = 0 regardless of scale value. ✓
            # So a single VAnd with maskVgprs[0] suffices per scale-group
            # boundary; no separate compute or scratch vgpr needed.
            scaleStride = self.config.lrSA.k if self.hasScale else 0
            if (not isBF16) and self.hasScale \
                    and (self.vgprTilesSA or self.vgprTilesSB) \
                    and scaleStride > 0 and (subIterK % scaleStride == 0):
                for tensor, tilesList in (('SA', self.vgprTilesSA),
                                          ('SB', self.vgprTilesSB)):
                    liveIds = sorted(set(
                        source.vgpr_tile_map.get(tensor, [{}])[0].values()))
                    for tid in liveIds:
                        for v in list(tilesList[tid]):
                            module.add(VAndB32(
                                dst=vgpr(v), src0=vgpr(v),
                                src1=vgpr(maskVgprs[0]),
                                comment=f"mask scale vgpr (reuse A/B mask, subIterK={subIterK})"))

        for m in set(maskVgprs):
            writer.vgprPool.checkIn(m)
        return list(module.flatitems())

    def emit_mask_k_done(self):
        """Release the long-lived tail-loop vgprs (scratch ones are released
        at the end of emit_mask_k_init)."""
        if getattr(self, "_tail_vDiff", None) is not None:
            self.writer.vgprPool.checkIn(self._tail_vDiff)
            self._tail_vDiff = None
        if getattr(self, "_tail_boundaryMask", None) is not None:
            for bm in self._tail_boundaryMask:
                self.writer.vgprPool.checkIn(bm)
            self._tail_boundaryMask = None
        return []

    def populate(self, emitted, unroll_iter=0):
        """Walk emitted partitions and fill em.instructions."""
        for partition_emitted in emitted:
            for emitted_group in partition_emitted:
                for em in emitted_group:
                    handler = self._dispatch.get(em.opType)
                    if handler:
                        em.instructions = handler(em, unroll_iter)
