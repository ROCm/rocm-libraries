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
    SCBranchSCC1, SMovB32, VAddU32, VAndB32, VCmpGEI32, VCmpLtI32, VCndMaskB32,
    VMovB32, VSubI32,
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
        module.add(globalReadPtrUpdates(tc, self.writer, self.kernel))
        module.add(skipLabel)
        return list(module.flatitems())

    def emit_tail_lr_inc(self, source):
        """LR LDS buffer swap for tail entry, gated by `K < 2*DepthU`.

        Runs only on the NLL-only path where PRELOOP swapped LW but not LR.
        NGLL path (K >= 2*DU) skips the body — NGLL already swapped LR.
        """
        tensor = source.tensor
        tc = {'A': 'A', 'B': 'B', 'SA': 'MXSA', 'SB': 'MXSB'}.get(tensor, tensor)
        two_du = 2 * self.kernel["DepthU"]
        skipLabel = Label(self.writer.labels.getNameInc(f"TailSkipLrInc_{tc}"), "")
        module = Module()
        module.add(SCmpGeU32(src0=sgpr("SizesSum+0"), src1=two_du,
                             comment=f"K >= 2*DepthU? skip tail LR swap for {tc}"))
        module.add(SCBranchSCC1(labelName=skipLabel.getLabelName(),
                                comment="NGLL path: LR already aligned with LW"))
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

        Also stages the BF16 half-mask constant 0x0000FFFF into self._tail_halfMaskVgpr
        when the kernel datatype is BF16. The half-mask lives in a VGPR (not an SGPR)
        because v_cndmask_b32 takes the user mask in src2 — adding another scalar
        source in src1 would violate the GCN constant-bus restriction.

        Reused by every emit_mask_k call in the tail body. The vgprs are leaked;
        the tail's vgpr pool is released by deallocVgprTiles when the body ends.
        """
        _, dividerFortidInK = self._mfma_K_constants()
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

        self._tail_halfMaskVgpr = None
        if self.kernel["ProblemType"]["DataTypeA"].isBFloat16():
            self._tail_halfMaskVgpr = writer.vgprPool.checkOut(1, "tail_halfMask")
            module.add(VMovB32(
                dst=vgpr(self._tail_halfMaskVgpr), src="0x0000FFFF",
                comment="BF16 half-mask: keep K0 (low 16b), zero K1 (high 16b)"))
        return list(module.flatitems())

    def emit_mask_k(self, source):
        """Per-lane K-mask for one subIterK.

        Computes laneK = _tail_kReg * numMIInUnroll + subIterK * MatrixInstK.

        A/B tiles get per-vgpr V_AND_B32 with a 3-state mask, so the boundary
        can fall inside a vgpr without losing the valid element. The mask for
        vgpr i (BF16 stride = 2 K positions per vgpr) is:
            diff = rem - (laneK + 2*i)
            mask = (diff <= 0) ? 0x00000000              # both K invalid
                 : (diff == 1) ? 0x0000FFFF              # keep K0 only
                              : 0xFFFFFFFF               # both K valid
        Non-BF16 datatypes (and MXSA/MXSB scale vgprs) keep the legacy
        whole-vgpr CNDMASK -> 0 path; the assertion below catches misuse
        until the partial-mask table is generalized.
        """
        assert self._tail_kReg is not None, \
            "emit_mask_k_init must run before emit_mask_k"

        writer = self.writer
        kernel = self.kernel
        subIterK = source.subIterK
        matrixInstK = kernel["MatrixInstK"]

        numMIInUnroll, _ = self._mfma_K_constants()

        loopCounterName = writer.loopCounterName(kernel, writer.states.unrollIdx)
        laneSGPRCount = writer.states.laneSGPRCount
        isBF16 = kernel["ProblemType"]["DataTypeA"].isBFloat16()
        kStride = 2  # BF16: 2 elements packed per 32-bit vgpr (low=K0, high=K1)

        module = Module()
        workKVgpr = writer.vgprPool.checkOut(1, f"mask_k_work_k{subIterK}")
        vMask = writer.vgprPool.checkOut(1, f"mask_k_vmask_k{subIterK}") if isBF16 else None
        vDiff = writer.vgprPool.checkOut(1, f"mask_k_vdiff_k{subIterK}") if isBF16 else None
        vNegOne = writer.vgprPool.checkOut(1, f"mask_k_vneg1_k{subIterK}") if isBF16 else None
        with writer.allocTmpSgpr(laneSGPRCount, alignment=laneSGPRCount) as tmpSgprInfo:
            maskSgpr = tmpSgprInfo.idx

            module.add(vectorStaticMultiply(
                vgpr(workKVgpr), vgpr(self._tail_kReg),
                numMIInUnroll, tmpSgprInfo,
                comment=f"laneK = tail_kReg * {numMIInUnroll}"))
            kBaseConst = subIterK * matrixInstK
            if kBaseConst:
                module.add(VAddU32(
                    dst=vgpr(workKVgpr), src0=vgpr(workKVgpr),
                    src1=kBaseConst,
                    comment=f"laneK += subIterK({subIterK}) * MatrixInstK"))

            # Whole-vgpr compare is only needed for non-BF16 A/B and for MXSA/MXSB.
            # The BF16 partial-mask path computes its own per-vgpr mask below.
            needWholeMaskCmp = (not isBF16) or self.hasScale
            if needWholeMaskCmp:
                module.add(VCmpGEI32(
                    dst=sgpr(maskSgpr, laneSGPRCount),
                    src0=vgpr(workKVgpr), src1=sgpr(loopCounterName),
                    comment=f"mask: laneK >= rem (subIterK={subIterK})"))

            if isBF16:
                # Stage 0xFFFFFFFF in a vgpr; v_cndmask_b32 takes the carry in src2,
                # so src0/src1 must avoid a second scalar source (const-bus rule).
                module.add(VMovB32(
                    dst=vgpr(vNegOne), src=-1,
                    comment="BF16 full-mask vgpr (constant-bus dodge)"))

            def _unique_ids(key):
                m = source.vgpr_tile_map.get(key, [{}])[0]
                return sorted(set(m.values()))

            def _mask_all_whole(tile_vgprs, label):
                for v in tile_vgprs:
                    module.add(VCndMaskB32(
                        dst=vgpr(v), src0=vgpr(v), src1=0,
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment=f"zero {label} if laneK >= rem"))

            def _mask_all_partial(tile_vgprs, label):
                for i, v in enumerate(tile_vgprs):
                    kOff = i * kStride
                    # vDiff = rem - (laneK + kOff)
                    if kOff:
                        module.add(VAddU32(
                            dst=vgpr(vDiff), src0=vgpr(workKVgpr), src1=kOff,
                            comment=f"vgprK0 = laneK + {kOff}"))
                        module.add(VSubI32(
                            dst=vgpr(vDiff), src0=sgpr(loopCounterName), src1=vgpr(vDiff),
                            comment=f"diff = rem - vgprK0 ({label}[{i}])"))
                    else:
                        module.add(VSubI32(
                            dst=vgpr(vDiff), src0=sgpr(loopCounterName), src1=vgpr(workKVgpr),
                            comment=f"diff = rem - laneK ({label}[{i}])"))
                    # mask = (diff < 2) ? sHalf : 0xFFFFFFFF
                    module.add(VCmpLtI32(
                        dst=sgpr(maskSgpr, laneSGPRCount),
                        src0=vgpr(vDiff), src1=2,
                        comment="diff < 2 ?"))
                    module.add(VCndMaskB32(
                        dst=vgpr(vMask),
                        src0=vgpr(vNegOne), src1=vgpr(self._tail_halfMaskVgpr),
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment="mask = (diff<2) ? halfKeep : full"))
                    # if diff < 1: mask = 0
                    module.add(VCmpLtI32(
                        dst=sgpr(maskSgpr, laneSGPRCount),
                        src0=vgpr(vDiff), src1=1,
                        comment="diff < 1 ?"))
                    module.add(VCndMaskB32(
                        dst=vgpr(vMask), src0=vgpr(vMask), src1=0,
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment="mask = (diff<1) ? 0 : prev"))
                    module.add(VAndB32(
                        dst=vgpr(v), src0=vgpr(v), src1=vgpr(vMask),
                        comment=f"mask {label}[{i}] (K=[{kOff},{kOff+kStride-1}])"))

            mask_ab = _mask_all_partial if isBF16 else _mask_all_whole
            for tid in _unique_ids('A'):
                mask_ab(list(self.vgprTilesA[tid]), "A")
            for tid in _unique_ids('B'):
                mask_ab(list(self.vgprTilesB[tid]), "B")
            # Re-issue the laneK >= rem compare for MXSA/MXSB which use whole-vgpr mask;
            # the BF16 partial path overwrote maskSgpr.
            if self.hasScale and isBF16:
                module.add(VCmpGEI32(
                    dst=sgpr(maskSgpr, laneSGPRCount),
                    src0=vgpr(workKVgpr), src1=sgpr(loopCounterName),
                    comment=f"reload mask: laneK >= rem (for MXSA/MXSB)"))
            if self.hasScale:
                for tid in _unique_ids('SA'):
                    _mask_all_whole(list(self.vgprTilesSA[tid]), "MXSA")
                for tid in _unique_ids('SB'):
                    _mask_all_whole(list(self.vgprTilesSB[tid]), "MXSB")

        writer.vgprPool.checkIn(workKVgpr)
        if vMask is not None:
            writer.vgprPool.checkIn(vMask)
        if vDiff is not None:
            writer.vgprPool.checkIn(vDiff)
        if vNegOne is not None:
            writer.vgprPool.checkIn(vNegOne)
        return list(module.flatitems())

    def emit_mask_k_done(self):
        """Release the tail-loop vgprs allocated by emit_mask_k_init."""
        if self._tail_kReg is not None:
            self.writer.vgprPool.checkIn(self._tail_kReg)
            self._tail_kReg = None
        if getattr(self, "_tail_halfMaskVgpr", None) is not None:
            self.writer.vgprPool.checkIn(self._tail_halfMaskVgpr)
            self._tail_halfMaskVgpr = None
        return []

    def populate(self, emitted, unroll_iter=0):
        """Walk emitted partitions and fill em.instructions."""
        for partition_emitted in emitted:
            for emitted_group in partition_emitted:
                for em in emitted_group:
                    handler = self._dispatch.get(em.opType)
                    if handler:
                        em.instructions = handler(em, unroll_iter)
