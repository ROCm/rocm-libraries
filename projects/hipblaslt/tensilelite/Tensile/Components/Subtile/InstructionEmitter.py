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

        # Dispatch table — unroll_iter is passed only for mfma/lr where the
        # per-body-copy K-position selects the right scale byte / VGPR tile.
        # gr_inc / lr_inc fire identically across all body copies (data side)
        # and the R>1 scale gating is now decoupled from `ui` at this layer.
        self._dispatch = {
            'mfma':     lambda em, ui: self.emit_mfma(em.source, ui),
            'lr':       lambda em, ui: self.emit_lr(em.source, ui),
            'gr':       lambda em, ui: self.emit_gr(em.source),
            'wait_gr':  lambda em, ui: self.emit_wait_gr(em.source),
            'wait_lr':  lambda em, ui: self.emit_wait_lr(),
            'sync':     lambda em, ui: self.emit_sync(),
            'lr_inc':   lambda em, ui: self.emit_lr_inc(em.source),
            'gr_inc':   lambda em, ui: self.emit_gr_inc(em.source),
            'skip':     lambda em, ui: self.emit_skip(em.source),
        }

    def emit_mfma(self, placement, unroll_iter=0):
        """Emit MFMA instructions from MFMAPlacement."""
        module = Module()
        subIterK = placement.subIterK
        tile_maps = {t: placement.vgpr_tile_maps[t][unroll_iter]
                     for t in placement.vgpr_tile_maps}

        # Under R>1 one scale fetch packs R body copies' K-bytes into the 4B
        # scale VGPR; advance the op_sel byte index by (ui%R)*numSubIterK_data
        # per body copy.  R==1 collapses to the legacy (a%2)+2*subIterK formula.
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
            # (4 MMA scale tiles = 4 bytes/lane) into a single VGPR.  dsOffset
            # strides only in M; the K position is implicit because one b32
            # covers the subtile's K-extent.  Invariant under R>1:
            # lrSubtileShape[1] == R*numSubIterK_data, so the same addressing
            # walks the 2x-larger scale LDS region in Solution.py's
            # _DepthUMXS{tc} allocation.
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

    def emit_lr_inc(self, source):
        """Emit localReadLDSBufferSwap for a single tensor.

        Under R>1 the data LR-side swap must fire on EVERY body copy so
        consecutive copies alternate which LDS half they read (R toggles per
        outer iter = net identity).  Gating to ui%R==0 makes all R copies read
        the SAME half and C1+ MFMAs consume stale data — the K>=512
        subtile_mxfp8_mt256 PGR=2 wrong-result bug.  R==1 reduces to
        once-per-outer-iter (legacy).  Scale lr_inc (SA/SB) is gated out of
        this handler by populate via _scale_op_gated_out (fires at ui%R==R-1).
        """
        tensor = source.tensor
        tc = {'A': 'A', 'B': 'B', 'SA': 'MXSA', 'SB': 'MXSB'}.get(tensor, tensor)
        module = Module()
        module.add(localReadLDSBufferSwap(tc, self.writer, self.kernel))
        return list(module.flatitems())

    def emit_gr_inc(self, source):
        """Emit globalReadPtrUpdates + globalReadLDSBufferSwap for a single tensor.

        Mirror of emit_lr_inc on the write side: under R>1 BOTH the data SRD
        advance AND the LW swap fire on every body copy so consecutive copies
        prefetch to ALTERNATING LDS halves (otherwise copies 1..R-1 overwrite
        copy 0 at the same offset — the K>=512 PGR=2 wrong-result bug).
        R==1 is bit-identical to legacy.  Scale gr_inc (SA/SB) is gated to
        ui%R==0 by populate so scale SRD+LW swap stay one atomic op.
        """
        tensor = source.tensor
        tc = {'A': 'A', 'B': 'B', 'SA': 'MXSA', 'SB': 'MXSB'}.get(tensor, tensor)
        module = Module()
        if tensor in ('SA', 'SB'):
            module.add(globalReadScalePtrUpdates(tc, self.writer, self.kernel))
            module.add(globalReadLDSBufferSwap(tc, self.writer, self.kernel))
        else:
            module.add(globalReadPtrUpdates(tc, self.writer, self.kernel))
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

    def populate(self, emitted, unroll_iter=0, strip_prefetch=False):
        """Walk emitted partitions and fill em.instructions.

        Per-ui emission cadence under R>1 (R==1: every gate is "fire"):

          Scale (SA/SB) — gated here via _scale_op_gated_out:
            gr, gr_inc : ui%R == 0     (DTL + SRD advance + LW swap)
            lr_inc     : ui%R == R-1   (LR swap after all R reads;
                                        insert_gr_lr_inc rewrites MT-
                                        transition preOp->postOp under
                                        PGR>0; see LogicalScheduler.py)
            lr         : every copy    (same 4B reload, op_sel picks byte)

          Data (A/B) — gated INSIDE emit_gr_inc / emit_lr_inc because
          SRD advance and LDS swap have different cadences (see those
          docstrings): SRD/LW/LR swaps all fire every copy.

        strip_prefetch (R>1 + PGR>=2 last outer iter): only the scale `gr`
        DTL is stripped because after the last iter
            SrdMXSA = orig + K            (one M-row stride OOB)
            SrdA    = orig + K - 128      (still in-bounds)
        SRD advances and LW/LR XORs are NOT stripped so the NGLL/NLL
        fall-through sees the same SRD/LDS state as the un-fixed code.
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
                    if strip_prefetch and self._is_prefetch_only(em):
                        em.instructions = []
                        continue
                    em.instructions = handler(em, unroll_iter)

    @staticmethod
    def _is_prefetch_only(em) -> bool:
        """Return True for scale GR (SA/SB); see populate() strip_prefetch."""
        if em.opType != 'gr':
            return False
        tensor = getattr(em.source, 'tensor', None)
        return tensor in ('SA', 'SB')

    @staticmethod
    def _scale_op_gated_out(em, unroll_iter: int, R: int) -> bool:
        """Return True for scale ops that should NOT emit in this ui.

        Cadence rationale: see populate().
        """
        tensor = getattr(em.source, 'tensor', None)
        if tensor not in ('SA', 'SB'):
            return False
        if em.opType in ('gr', 'gr_inc'):
            return (unroll_iter % R) != 0
        if em.opType == 'lr_inc':
            return (unroll_iter % R) != (R - 1)
        return False
