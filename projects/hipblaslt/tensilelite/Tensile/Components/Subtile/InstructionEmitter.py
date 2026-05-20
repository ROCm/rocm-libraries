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
    SWaitCnt, SBarrier, DSLoadB32, SCmpEQU32, SCmpLeU32, SCmpLtU32, SCmpGeU32,
    SCBranchSCC1, SMovB32, VAddU32, VAndB32, VCmpGEI32, VCmpGTI32, VCmpLeI32,
    VCmpLtI32, VCndMaskB32, VMovB32, VSubI32,
)
from rocisa.container import vgpr, sgpr, DSModifiers, ContinuousRegister
from rocisa.code import Label
from rocisa.functions import (
    vectorStaticRemainder, vectorStaticDivide, vectorStaticMultiply,
)


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
                 vgprTilesSA=None, vgprTilesSB=None):
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
            'tail_srd_advance': lambda em, ui: self.emit_tail_srd_advance(em.source),
            'tail_lr_inc':      lambda em, ui: self.emit_tail_lr_inc(em.source),
            'skip':             lambda em, ui: self.emit_skip(em.source),
            'mask_k_init':  lambda em, ui: self.emit_mask_k_init(),
            'mask_k':       lambda em, ui: self.emit_mask_k(em.source),
            'mask_k_done':  lambda em, ui: self.emit_mask_k_done(),
        }

        # Per-lane K-index vgpr for the tail-loop K mask. Set by emit_mask_k_init,
        # consumed by emit_mask_k for every subIterK in the tail body.
        self._tail_kReg = None

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
                    sAsel = (a % 2) + 2 * subIterK
                    sBsel = (b % 2) + 2 * subIterK
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
                    module.add(emitSingleDsRead(
                        ti, tileId, subtileK, subIterK_within, dstTile))
        elif tensor in ('SA', 'SB'):
            tc = 'MXSA' if tensor == 'SA' else 'MXSB'
            ti = self.tileInfoMap[tensor]
            lrGran = self.config.lrSA if tensor == 'SA' else self.config.lrSB
            vgprTilesScale = self.vgprTilesSA if tensor == 'SA' else self.vgprTilesSB
            groupStride = lrGran.mn * ti.subtileSize
            subtileK = placement.tiles.subIterK_start // self.subtileShapeK
            for tileId in range(placement.tiles.tileId_start, placement.tiles.tileId_end, lrGran.mn):
                scaleGroupIdx = tileId // lrGran.mn
                groupKey = scaleGroupIdx * lrGran.mn
                dsOffset = groupStride * (scaleGroupIdx * (self.config.numSubIterK // self.subtileShapeK) + subtileK)
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

    def emit_tail_srd_advance(self, source):
        """SRD-only advance for tail entry, gated by `K >= 2*DepthU`.

        Runs only on the NGLL path where PRELOOP's MT1 GR loaded without an
        accompanying SRD advance. NLL-only path (K < 2*DU) skips the body.
        """
        tensor = source.tensor
        tc = {'A': 'A', 'B': 'B', 'SA': 'MXSA', 'SB': 'MXSB'}.get(tensor, tensor)
        two_du = 2 * self.kernel["DepthU"]
        skipLabel = Label(self.writer.labels.getNameInc(f"TailSkipSrdAdv_{tc}"), "")
        module = Module()
        module.add(SCmpLtU32(src0=sgpr("SizesSum+0"), src1=two_du,
                             comment=f"K < 2*DepthU? skip tail SRD advance for {tc}"))
        module.add(SCBranchSCC1(labelName=skipLabel.getLabelName(),
                                comment="NLL-only path: SRD already in place"))
        if tensor in ('SA', 'SB'):
            module.add(globalReadScalePtrUpdates(tc, self.writer, self.kernel))
        else:
            module.add(globalReadPtrUpdates(tc, self.writer, self.kernel))
        module.add(skipLabel)
        return list(module.flatitems())

    def emit_tail_lr_inc(self, source):
        """LR LDS buffer swap for tail entry, gated by `DU <= K < 2*DepthU`.

        Runs only on the NLL-only path where PRELOOP swapped LW but not LR.
        Skipped on:
          - NGLL path (K >= 2*DU): NGLL already swapped LR.
          - K < DU path: PRELOOP was skipped entirely, LW never swapped.
        """
        tensor = source.tensor
        tc = {'A': 'A', 'B': 'B', 'SA': 'MXSA', 'SB': 'MXSB'}.get(tensor, tensor)
        du = self.kernel["DepthU"]
        two_du = 2 * du
        skipLabel = Label(self.writer.labels.getNameInc(f"TailSkipLrInc_{tc}"), "")
        module = Module()
        module.add(SCmpGeU32(src0=sgpr("SizesSum+0"), src1=two_du,
                             comment=f"K >= 2*DepthU? skip tail LR swap for {tc}"))
        module.add(SCBranchSCC1(labelName=skipLabel.getLabelName(),
                                comment="NGLL path: LR already aligned with LW"))
        module.add(SCmpLtU32(src0=sgpr("SizesSum+0"), src1=du,
                             comment=f"K < DepthU? skip tail LR swap for {tc}"))
        module.add(SCBranchSCC1(labelName=skipLabel.getLabelName(),
                                comment="preloop-skipped path: LW never swapped"))
        module.add(localReadLDSBufferSwap(tc, self.writer, self.kernel))
        module.add(skipLabel)
        return list(module.flatitems())

    def emit_skip(self, source):
        """Emit skip guard: compare LoopCounterL and branch."""
        skipLabel = Label(f"SkipTo{source.target}", "")
        cmpMap = {"EQ": SCmpEQU32, "LE": SCmpLeU32}
        return [
            cmpMap[source.compare](src0=sgpr("LoopCounterL"), src1=source.value,
                                   comment=f"LoopCounter {source.compare} {source.value}?"),
            SCBranchSCC1(labelName=skipLabel.getLabelName(),
                         comment=f"skip to {source.target}"),
        ]

    def _mfma_K_constants(self):
        """Constants used by both mask emitters.

        Returns (numMIInUnroll, dividerFortidInK). Assumes K % numMIInUnroll == 0
        so the tail boundary always lands between full per-lane K chunks — no
        intra-MFMA-operand group split or intra-vgpr partial masking needed.
        """
        kernel = self.kernel
        matrixInstT      = min(kernel["MatrixInstM"], kernel["MatrixInstN"])
        numTileInInstA   = kernel["MatrixInstM"] // matrixInstT
        numMIInUnroll    = kernel["MIInputPerThreadA"] // numTileInInstA
        dividerFortidInK = kernel["MatrixInstN"] * kernel["MatrixInstB"]
        return numMIInUnroll, dividerFortidInK

    def emit_mask_k_init(self):
        """Compute (Serial % WavefrontSize) / dividerFortidInK into self._tail_kReg.

        Also hoists the per-subIterK invariants into persistent vgprs so the
        per-subIterK emit_mask_k only needs to issue cmp/cndmask with bumped
        immediates (no laneK recomputation, no per-call diff sub):
          * self._tail_workKVgpr — laneK_0 = kReg * numMIInUnroll
          * self._tail_vDiff      — rem - laneK_0  (shared across all subIterK;
            for subIterK=n the effective diff is diff - n*MatrixInstK, which we
            fold into the cmp constants)
          * self._tail_halfMaskVgpr — 0x0000FFFF (BF16 only, low-BF16-keep)
          * self._tail_vNegOne    — 0xFFFFFFFF (BF16 only, full-mask vgpr;
            cndmask const-bus dodge)

        The half-mask and full-mask live in vgprs because v_cndmask_b32 carries
        the lane mask in src2 — adding another scalar source on src0/src1 would
        violate the GCN constant-bus restriction.

        Reused by every emit_mask_k call in the tail body. All vgprs are released
        in emit_mask_k_done.
        """
        _, dividerFortidInK = self._mfma_K_constants()
        numMIInUnroll, _ = self._mfma_K_constants()
        writer = self.writer
        module = Module()

        self._tail_kReg = writer.vgprPool.checkOut(1, "tail_kReg")
        tmpVgpr = writer.vgprPool.checkOut(2, "tail_kReg_tmp")
        tmpVgprRes = ContinuousRegister(idx=tmpVgpr, size=2)
        dummy = writer.vgprPool.checkOut(1, "tail_kReg_dummy")
        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            module.add(vectorStaticRemainder(
                dummy, self._tail_kReg, "Serial",
                self.kernel["WavefrontSize"], tmpVgprRes, tmpSgprInfo))
            module.add(vectorStaticDivide(
                self._tail_kReg, self._tail_kReg,
                dividerFortidInK, tmpVgprRes))
        writer.vgprPool.checkIn(tmpVgpr)
        writer.vgprPool.checkIn(dummy)

        # laneK_0 and diff = rem - laneK_0 are shared across all subIterK.
        self._tail_workKVgpr = writer.vgprPool.checkOut(1, "tail_workK")
        self._tail_vDiff = writer.vgprPool.checkOut(1, "tail_vDiff")
        loopCounterName = writer.loopCounterName(
            self.kernel, writer.states.unrollIdx)
        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            module.add(vectorStaticMultiply(
                vgpr(self._tail_workKVgpr), vgpr(self._tail_kReg),
                numMIInUnroll, tmpSgprInfo,
                comment=f"laneK_0 = tail_kReg * {numMIInUnroll}"))
        module.add(VSubI32(
            dst=vgpr(self._tail_vDiff),
            src0=sgpr(loopCounterName), src1=vgpr(self._tail_workKVgpr),
            comment="diff = rem - laneK_0 (shared across all subIterK)"))

        # _tail_vNegOne is needed by both BF16 (3-state mask src0) and non-BF16
        # (cndmask src0 to build a 0-or-(-1) lane mask). v_cndmask_b32 already
        # spends its const-bus on src2 (sgpr predicate); src0 must be a vgpr.
        self._tail_halfMaskVgpr = None
        self._tail_vNegOne = writer.vgprPool.checkOut(1, "tail_vNegOne")
        module.add(VMovB32(
            dst=vgpr(self._tail_vNegOne), src=-1,
            comment="full-mask vgpr (constant-bus dodge)"))
        self._tail_vD8 = None
        self._tail_boundaryMask = None
        if self.kernel["ProblemType"]["DataTypeA"].isBFloat16():
            self._tail_halfMaskVgpr = writer.vgprPool.checkOut(1, "tail_halfMask")
            module.add(VMovB32(
                dst=vgpr(self._tail_halfMaskVgpr), src="0x0000FFFF",
                comment="BF16 half-mask: keep K0 (low 16b), zero K1 (high 16b)"))

            # Precompute the 4 boundary masks from d = rem%8.
            # The boundary-mask pattern (which vgprs are full/half/zero) depends
            # only on rem%8: since laneK_0 is always a multiple of 8 and we mod
            # out MatrixInstK=32, every "boundary" lane in every subIterK has
            # effective_diff ≡ rem (mod 8). So all boundary lanes share the same
            # 4-vgpr mask pattern, and we can build it once here.
            laneSGPRCount = writer.states.laneSGPRCount
            self._tail_vD8 = writer.vgprPool.checkOut(1, "tail_vD8")
            module.add(VAndB32(
                dst=vgpr(self._tail_vD8),
                src0=sgpr(loopCounterName), src1=7,
                comment="d = rem % 8 (boundary-mask pattern depends only on this)"))
            self._tail_boundaryMask = [
                writer.vgprPool.checkOut(1, f"tail_boundaryMask{i}")
                for i in range(4)
            ]
            with writer.allocTmpSgpr(laneSGPRCount, alignment=laneSGPRCount) as tmpSgprInfo:
                maskSgpr = tmpSgprInfo.idx
                for i in range(4):
                    bm = self._tail_boundaryMask[i]
                    module.add(VCmpLtI32(
                        dst=sgpr(maskSgpr, laneSGPRCount),
                        src0=vgpr(self._tail_vD8), src1=2*i + 2,
                        comment=f"boundary[{i}]: d < {2*i+2} ? halfKeep : full"))
                    module.add(VCndMaskB32(
                        dst=vgpr(bm),
                        src0=vgpr(self._tail_vNegOne),
                        src1=vgpr(self._tail_halfMaskVgpr),
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment=f"boundaryMask[{i}] = (d<{2*i+2}) ? halfKeep : full"))
                    module.add(VCmpLtI32(
                        dst=sgpr(maskSgpr, laneSGPRCount),
                        src0=vgpr(self._tail_vD8), src1=2*i + 1,
                        comment=f"boundary[{i}]: d < {2*i+1} ? 0 : prev"))
                    module.add(VCndMaskB32(
                        dst=vgpr(bm), src0=vgpr(bm), src1=0,
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment=f"boundaryMask[{i}] = (d<{2*i+1}) ? 0 : prev"))
        return list(module.flatitems())

    def emit_mask_k(self, source):
        """Per-lane K-mask for one subIterK.

        Uses the shared diff = rem - laneK_0 staged in emit_mask_k_init.
        For subIterK=n the effective per-lane diff is `diff - n*MatrixInstK`,
        which we fold into the cmp immediates (no per-call sub).

        A/B tiles get per-vgpr V_AND_B32 with a mask vgpr. For BF16
        (stride = 2 K positions per vgpr), vgpr `i` in subIterK `n` uses a
        3-state mask:
            kOff = i*2 + n*MatrixInstK
            mask = (diff < kOff+1) ? 0
                 : (diff < kOff+2) ? 0x0000FFFF
                                   : 0xFFFFFFFF
        Non-BF16 (e.g. FP4) uses a single 2-state mask shared across all i:
            mask = (diff < n*MatrixInstK + 1) ? 0 : 0xFFFFFFFF
        This assumes rem aligns to the per-lane K stride per vgpr (true for
        rem=32 with FP4 MIK=128, 8 K/vgpr × 4 vgprs). A boundary inside a
        non-BF16 vgpr would need per-byte/nibble handling.
        MXSA/MXSB keep the whole-vgpr CNDMASK -> 0 path.
        """
        assert self._tail_kReg is not None, \
            "emit_mask_k_init must run before emit_mask_k"

        writer = self.writer
        kernel = self.kernel
        subIterK = source.subIterK
        matrixInstK = kernel["MatrixInstK"]
        kBaseConst = subIterK * matrixInstK

        laneSGPRCount = writer.states.laneSGPRCount
        isBF16 = kernel["ProblemType"]["DataTypeA"].isBFloat16()
        kStride = 2  # BF16: 2 elements packed per 32-bit vgpr (low=K0, high=K1)

        module = Module()

        def _unique_ids(key):
            m = source.vgpr_tile_map.get(key, [{}])[0]
            return sorted(set(m.values()))

        # All A/B tiles in this subIterK have the same vgprs-per-lane count
        # and the same K layout across vgprs, so the i-th vgpr of every tile
        # gets the same mask. Compute the masks once and reuse across all tiles.
        vgprPerInUnroll = 0
        for ids, tilesDict in ((_unique_ids('A'), self.vgprTilesA),
                               (_unique_ids('B'), self.vgprTilesB)):
            if ids:
                vgprPerInUnroll = len(list(tilesDict[ids[0]]))
                break

        # BF16 allocates one mask per i (boundary may land inside any vgpr).
        # Non-BF16 builds a single 2-state mask and replicates the index across
        # all i, since the boundary aligns to a vgpr edge.
        if isBF16:
            maskVgprs = [writer.vgprPool.checkOut(1, f"mask_k_msk{i}_k{subIterK}")
                         for i in range(vgprPerInUnroll)]
        else:
            sharedMask = writer.vgprPool.checkOut(1, f"mask_k_msk_k{subIterK}")
            maskVgprs = [sharedMask] * vgprPerInUnroll

        with writer.allocTmpSgpr(laneSGPRCount, alignment=laneSGPRCount) as tmpSgprInfo:
            maskSgpr = tmpSgprInfo.idx

            # Compare for non-BF16: laneK_n >= rem ⟺ diff <= n*MatrixInstK ⟺
            # diff < kBaseConst+1. Predicate is reused by MXSA/MXSB below (whole
            # -vgpr cndmask) and by the v_and mask vgpr we build next.
            # BF16 doesn't need this here — the isBF16 block below ends with
            # sZero (= diff <= kBaseConst) in maskSgpr, which MXSA/MXSB reuses.
            if not isBF16:
                literal = kBaseConst + 1
                if -16 <= literal <= 64:
                    module.add(VCmpLtI32(
                        dst=sgpr(maskSgpr, laneSGPRCount),
                        src0=vgpr(self._tail_vDiff), src1=literal,
                        comment=f"mask: diff < {literal} (laneK_{subIterK} >= rem)"))
                else:
                    # VOPC inline-constant range is -16..64; stage larger
                    # literals via a scratch sgpr (e.g. FP4 MI_K=128, subIterK>=1).
                    with writer.allocTmpSgpr(1) as litSgprInfo:
                        litSgpr = litSgprInfo.idx
                        module.add(SMovB32(
                            dst=sgpr(litSgpr), src=hex(literal),
                            comment=f"stage literal {literal} (non-inline)"))
                        module.add(VCmpLtI32(
                            dst=sgpr(maskSgpr, laneSGPRCount),
                            src0=vgpr(self._tail_vDiff), src1=sgpr(litSgpr),
                            comment=f"mask: diff < {literal} (laneK_{subIterK} >= rem)"))
                # Build the 2-state mask vgpr: predicate ? 0 : -1.
                module.add(VCndMaskB32(
                    dst=vgpr(sharedMask),
                    src0=vgpr(self._tail_vNegOne), src1=0,
                    src2=sgpr(maskSgpr, laneSGPRCount),
                    comment=f"mask = (diff < {literal}) ? 0 : -1"))

            if isBF16:
                # 3-way classifier per (lane, subIterK):
                #   sFull = effective_diff_n >= 8  → mask = -1
                #   sZero = effective_diff_n <= 0  → mask = 0
                #   else                           → mask = boundaryMask[i] (precomputed)
                # Only 2 cmps per subIterK (shared across all i), vs 8 in the
                # per-i version, because the boundary pattern is baked in.
                module.add(VCmpGTI32(
                    dst=sgpr(maskSgpr, laneSGPRCount),
                    src0=vgpr(self._tail_vDiff), src1=kBaseConst + 7,
                    comment=f"sFull: diff > {kBaseConst+7} (effective_diff_{subIterK} >= 8)"))
                for i in range(vgprPerInUnroll):
                    module.add(VCndMaskB32(
                        dst=vgpr(maskVgprs[i]),
                        src0=vgpr(self._tail_boundaryMask[i]),
                        src1=vgpr(self._tail_vNegOne),
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment=f"mask[{i}] = sFull ? full : boundary[{i}]"))
                module.add(VCmpLeI32(
                    dst=sgpr(maskSgpr, laneSGPRCount),
                    src0=vgpr(self._tail_vDiff), src1=kBaseConst,
                    comment=f"sZero: diff <= {kBaseConst} (effective_diff_{subIterK} <= 0)"))
                for i in range(vgprPerInUnroll):
                    module.add(VCndMaskB32(
                        dst=vgpr(maskVgprs[i]), src0=vgpr(maskVgprs[i]), src1=0,
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment=f"mask[{i}] = sZero ? 0 : prev"))

            def _mask_all_whole(tile_vgprs, label):
                for v in tile_vgprs:
                    module.add(VCndMaskB32(
                        dst=vgpr(v), src0=vgpr(v), src1=0,
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment=f"zero {label} if laneK >= rem"))

            def _mask_all_partial(tile_vgprs, label):
                for i, v in enumerate(tile_vgprs):
                    module.add(VAndB32(
                        dst=vgpr(v), src0=vgpr(v), src1=vgpr(maskVgprs[i]),
                        comment=f"mask {label}[{i}] (K=[{i*kStride},{i*kStride+kStride-1}])"))

            for tid in _unique_ids('A'):
                _mask_all_partial(list(self.vgprTilesA[tid]), "A")
            for tid in _unique_ids('B'):
                _mask_all_partial(list(self.vgprTilesB[tid]), "B")
            # MXSA/MXSB use the whole-vgpr zero mask. For BF16 the isBF16 block
            # above left sZero (= diff <= kBaseConst, semantically equal to
            # diff < kBaseConst+1) in maskSgpr — exactly what we need. For non-BF16
            # the needWholeMaskCmp at the top set the same predicate.
            # TEMP: scale mask disabled while debugging FP4 tail correctness.
            # if self.hasScale:
            #     for tid in _unique_ids('SA'):
            #         _mask_all_whole(list(self.vgprTilesSA[tid]), "MXSA")
            #     for tid in _unique_ids('SB'):
            #         _mask_all_whole(list(self.vgprTilesSB[tid]), "MXSB")

        for m in set(maskVgprs):
            writer.vgprPool.checkIn(m)
        return list(module.flatitems())

    def emit_mask_k_done(self):
        """Release the tail-loop vgprs allocated by emit_mask_k_init."""
        if self._tail_kReg is not None:
            self.writer.vgprPool.checkIn(self._tail_kReg)
            self._tail_kReg = None
        if getattr(self, "_tail_workKVgpr", None) is not None:
            self.writer.vgprPool.checkIn(self._tail_workKVgpr)
            self._tail_workKVgpr = None
        if getattr(self, "_tail_vDiff", None) is not None:
            self.writer.vgprPool.checkIn(self._tail_vDiff)
            self._tail_vDiff = None
        if getattr(self, "_tail_halfMaskVgpr", None) is not None:
            self.writer.vgprPool.checkIn(self._tail_halfMaskVgpr)
            self._tail_halfMaskVgpr = None
        if getattr(self, "_tail_vNegOne", None) is not None:
            self.writer.vgprPool.checkIn(self._tail_vNegOne)
            self._tail_vNegOne = None
        if getattr(self, "_tail_vD8", None) is not None:
            self.writer.vgprPool.checkIn(self._tail_vD8)
            self._tail_vD8 = None
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
