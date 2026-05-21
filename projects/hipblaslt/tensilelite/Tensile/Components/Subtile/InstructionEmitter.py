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
from rocisa.instruction import SWaitCnt, SBarrier, DSLoadB32, SCmpEQU32, SCmpLeU32, SCBranchSCC1
from rocisa.container import vgpr, sgpr, DSModifiers
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

        # Dispatch table — unroll_iter is passed for mfma/lr/gr_inc/lr_inc
        # (gr_inc/lr_inc consume ui for the R-period swap-vs-advance split).
        self._dispatch = {
            'mfma':     lambda em, ui: self.emit_mfma(em.source, ui),
            'lr':       lambda em, ui: self.emit_lr(em.source, ui),
            'gr':       lambda em, ui: self.emit_gr(em.source),
            'wait_gr':  lambda em, ui: self.emit_wait_gr(em.source),
            'wait_lr':  lambda em, ui: self.emit_wait_lr(),
            'sync':     lambda em, ui: self.emit_sync(),
            'lr_inc':   lambda em, ui: self.emit_lr_inc(em.source, ui),
            'gr_inc':   lambda em, ui: self.emit_gr_inc(em.source, ui),
            'skip':     lambda em, ui: self.emit_skip(em.source),
        }

    def emit_mfma(self, placement, unroll_iter=0):
        """Emit MFMA instructions from MFMAPlacement."""
        module = Module()
        subIterK = placement.subIterK
        tile_maps = {t: placement.vgpr_tile_maps[t][unroll_iter]
                     for t in placement.vgpr_tile_maps}

        # Decouple-scale-DU op_sel: scaleSchedulingPeriod (R) > 1 means one
        # scale fetch covers R body copies' worth of K-bytes packed in the
        # 4 B scale VGPR.  numSubIterK is the data-side value (not widened),
        # so we add (ui % R) * numSubIterK_data to subIterK to advance the
        # scale byte index from one body copy to the next.  For R==1 this
        # collapses to the original `(a%2) + 2*subIterK` formula.
        R = self.config.scaleSchedulingPeriod
        numSubIterK_data = getattr(self.config, 'numSubIterK_data',
                                   self.config.numSubIterK)
        effectiveSubIterK = subIterK + (unroll_iter % R) * numSubIterK_data

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
                    sAsel = (a % 2) + 2 * effectiveSubIterK
                    sBsel = (b % 2) + 2 * effectiveSubIterK
                else:
                    scaleAVgpr = scaleBVgpr = -1
                    sAsel = sBsel = 0

                module.add(emitMfmaInstruction(
                    self.writer, self.kernel, aTile, bTile, dTile, dTile,
                    scaleAVgpr=scaleAVgpr, scaleBVgpr=scaleBVgpr,
                    scaleAsel=sAsel, scaleBsel=sBsel,
                    comment=f"MFMA C[{a},{b}] += A[{a},K={effectiveSubIterK}] * B[{b},K={effectiveSubIterK}]"))
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
            # One ds_read_b32 per scale group loads the whole (2,2) LR subtile
            # — 4 MMA scale tiles == 4 bytes per lane — into a single VGPR.
            # dsOffset strides by lrSubtileSize in the M dimension only; the K
            # position within the LR subtile is implicit because a single
            # b32 load covers all K within the subtile.  With
            # scaleSchedulingPeriod (R) > 1 the scheduler emits exactly one
            # scale LR per period covering [0, R*numSubIterK_data) K-slots,
            # which corresponds to one (2,2) LR subtile's K-extent under the
            # supported configs (lrSubtileShape[1] == R*numSubIterK_data),
            # so the existing dsOffset addressing stays valid.  The scale
            # LDS region itself is R times larger (driven by _DepthUMXS{tc}
            # widening in Solution.py); LR buffer-swap selects which half of
            # that region this fetch reads from.
            for tileId in range(placement.tiles.tileId_start, placement.tiles.tileId_end, lrGran.mn):
                scaleGroupIdx = tileId // lrGran.mn
                groupKey = scaleGroupIdx * lrGran.mn
                dsOffset = int(ti.lrSubtileSize) * scaleGroupIdx
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

    def emit_lr_inc(self, source, unroll_iter=0):
        """Emit localReadLDSBufferSwap for a single tensor.

        Decouple-scale-DU (data path): the LDS read-side swap is conceptually
        an outer-iter-boundary event, not a per-body-copy event.  With
        scaleSchedulingPeriod (R) > 1 the unroll factor expands to R body
        copies per outer iter; the LDS read half must flip exactly ONCE per
        outer iter so that all R copies in a given outer iter consume the
        SAME half (the one written by the PGR-prefetched outer iter).
        lr_inc is a preOp on the LR placement, so it fires immediately
        BEFORE the LR ds_read.  Firing it at ui%R == 0 means the swap
        happens between outer iter N's last copy and outer iter N+1's
        first copy — i.e. AFTER all R copies of outer iter N have consumed
        the current half (matching the design intent).  ui%R != 0 copies
        emit nothing.

        For R == 1 the gate (ui%1 == 0) is always true and the output is
        bit-identical to the pre-R-period emitter (lr_inc fires every body
        copy, same as today's per-body alternation).

        Scale lr_inc (SA/SB) ops are gated out of this handler entirely by
        InstructionEmitter.populate via _scale_op_gated_out (at ui%R != R-1
        for scale lr_inc; the asymmetry comes from scale lr/gr_inc being a
        single atomic op rather than the data path's split SRD-advance vs
        LDS-swap), so the gate below only governs data (A/B) tensors in
        practice.
        """
        tensor = source.tensor
        tc = {'A': 'A', 'B': 'B', 'SA': 'MXSA', 'SB': 'MXSB'}.get(tensor, tensor)
        R = self.config.scaleSchedulingPeriod
        module = Module()
        if tensor in ('A', 'B') and R > 1 and (unroll_iter % R) != 0:
            return []
        module.add(localReadLDSBufferSwap(tc, self.writer, self.kernel))
        return list(module.flatitems())

    def emit_gr_inc(self, source, unroll_iter=0):
        """Emit globalReadPtrUpdates + globalReadLDSBufferSwap for a single tensor.

        Decouple-scale-DU (data path): the data SRD must advance on EVERY
        body copy so the next GR reads the next data-DU of K, but the LDS
        write-side swap (XOR LocalWriteBaseAddr with Swap) is an outer-iter
        boundary event — it flips which LDS half the next R-period worth
        of GRs writes into.  With scaleSchedulingPeriod (R) > 1 the unroll
        factor expands to R body copies per outer iter; we want the LDS
        swap to fire exactly once per outer iter at ui%R == 0 (so the next
        period's R GRs target the OTHER half).  The SRD advance still fires
        every copy, cumulatively advancing by R*DU per outer iter as
        required by the data layout.

        For R == 1 the gate (ui%1 == 0) is always true so we emit both the
        SRD advance and the LDS swap on every copy — bit-identical to the
        pre-R-period emitter.

        Scale gr_inc (SA/SB) ops are gated out of this handler entirely by
        InstructionEmitter.populate via _scale_op_gated_out at ui%R != 0,
        so the scale path still emits both ScalePtrUpdate and ScaleLDSSwap
        atomically (when it does emit).
        """
        tensor = source.tensor
        tc = {'A': 'A', 'B': 'B', 'SA': 'MXSA', 'SB': 'MXSB'}.get(tensor, tensor)
        R = self.config.scaleSchedulingPeriod
        module = Module()
        if tensor in ('SA', 'SB'):
            module.add(globalReadScalePtrUpdates(tc, self.writer, self.kernel))
            module.add(globalReadLDSBufferSwap(tc, self.writer, self.kernel))
        else:
            module.add(globalReadPtrUpdates(tc, self.writer, self.kernel))
            if R == 1 or (unroll_iter % R) == 0:
                module.add(globalReadLDSBufferSwap(tc, self.writer, self.kernel))
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

    def populate(self, emitted, unroll_iter=0):
        """Walk emitted partitions and fill em.instructions.

        Decouple-scale-DU: when scaleSchedulingPeriod (R) > 1, the
        emission cadence within an R-period of body copies is:

          Scale (SA/SB) path  — gated atomically at populate time via
          _scale_op_gated_out (whole op is skipped in non-firing copies):
            - scale GR (DTL → LDS) and scale gr_inc (SRD advance + LDS
              write-side swap) fire in the FIRST copy of the period
              (ui%R == 0) so the DTL goes to the current half and the
              swap sets up the next period's write target.
            - scale lr_inc (LDS read-side swap) fires in the LAST copy
              of the period (ui%R == R-1) so the swap happens AFTER
              all R copies have read the same half (PGR=0 postOp
              cadence; the residual R=2 PGR=2 K>=512 regression
              traces to scale lr_inc being a preOp under PGR>0 but
              that is left as-is per the scope of this fix —
              touching the scale path globally regresses PGR=2 K=256
              and PGR=0 cases).
            - scale LR (ds_read) is NOT gated: it re-issues the same
              4 B load into a cycled VGPR each copy.  MFMA op_sel
              picks the right K byte per copy (see emit_mfma).

          Data (A/B) path  — gated INSIDE emit_gr_inc / emit_lr_inc
          because the SRD advance and the LDS swap have different
          cadences:
            - data SRD advance (s_add Srd{tc}, Srd{tc}, DU*bpe) fires
              every copy in emit_gr_inc — the SRD cumulatively walks
              R*DU per outer iter so the next GR reads the next K-DU.
            - data LDS write-side swap (s_xor LocalWriteBaseAddr,
              Swap) fires only at ui%R == 0 — the next period's R
              GRs all write into the OTHER half (preOp on GR, so the
              swap happens before the GR).
            - data LDS read-side swap (v_xor sharedVgprLROffset,
              sharedVgprLROffsetSwap) fires only at ui%R == 0 — same
              gate as the write-side swap.  lr_inc is a preOp on LR
              so firing at ui%R == 0 means the swap is sandwiched
              between outer iter N's last LR (ui=R-1) and outer iter
              N+1's first LR (ui=0).  Both bodies of a given outer
              iter then consume the SAME LDS half.

        For R == 1 every gate reduces to "fire every copy" and the
        output is bit-identical to the pre-R-period scheduler.
        """
        R = self.config.scaleSchedulingPeriod
        for partition_emitted in emitted:
            for emitted_group in partition_emitted:
                for em in emitted_group:
                    handler = self._dispatch.get(em.opType)
                    if handler is None:
                        continue
                    if R > 1 and self._scale_op_gated_out(em, unroll_iter, R):
                        em.instructions = []
                        continue
                    em.instructions = handler(em, unroll_iter)

    @staticmethod
    def _scale_op_gated_out(em, unroll_iter: int, R: int) -> bool:
        """Return True for scale ops that should NOT emit in this ui.

        See populate() docstring for the cadence rationale.
        """
        tensor = getattr(em.source, 'tensor', None)
        if tensor not in ('SA', 'SB'):
            return False
        if em.opType in ('gr', 'gr_inc'):
            return (unroll_iter % R) != 0
        if em.opType == 'lr_inc':
            return (unroll_iter % R) != (R - 1)
        return False
