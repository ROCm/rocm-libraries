# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Subtile-path tail-loop scaffold.

Stitches together (in order):

- PGR>0 tail-entry gating (origCounter==0 / 0<orig<PGR / orig>=PGR);
- ``calculateLoopNumIter`` + ``openLoop`` (loop-counter setup + early
  exit to ``SkipTailLoopL``);
- ``SrdNumRecords`` tighteners (see ``SubtileTailSrdTighten``);
- vgprTile alloc for A/B/MX scale operands;
- DepthU-shaped GR + MX scale GR;
- per-lane ``kPosBase = tidInK * numMIInUnroll``;
- per-(operand, mmak, ir) K-tail mask precompute (see
  ``SubtileTailMask``) hoisted above the GR drain so it co-issues
  with buffer-load latency;
- per-mmak A/B slice alloc + ds_read + lane mask + MFMA grid +
  slice free, with per-mmak early exit;
- final ``SkipTailLoopL`` label via ``closeLoop(emitEndLabelOnly=True)``.

The helper is a free function: it takes the live KernelWriter as
``kw``, mirroring ``SubtileTailSrdTighten`` /
``SubtileTailMask``. KernelWriter exposes a thin
``_emitTailLoopScaffoldSubtile`` instance-method delegate so the
existing unit-test API (``kwa._emitTailLoopScaffoldSubtile(...)``)
stays unchanged.
"""

from rocisa.code import Label, Module
from rocisa.container import sgpr, vgpr
from rocisa.instruction import (
    SAddCU32, SAddU32, SBarrier, SBranch, SCBranchSCC1,
    SCmpEQU32, SCmpLeU32, SCmpLtU32, SWaitCnt, SXorB32,
    VCndMaskB32,
)
from rocisa.functions import (
    vectorStaticDivide, vectorStaticMultiply, vectorStaticRemainder,
)

from .Kernel import MX_SCALE_TILES_PER_VGPR, emitMfmaInstruction
from .SubtileGREmit import globalReadDoSubtile
from .SubtileLREmit import emitSingleDsRead
from .SubtileScaleEmit import globalReadDoScaleSubtile, localReadDoScaleSubtile
from .SubtileTailMask import (
    emitTailKPosCmpSubtile,
    emitTailSubLaneMaskApplySubtile,
    emitTailSubLaneMaskPrecomputeSubtile,
    emitTailSubLaneMaskRefineSubtile,
    subtileTailByteShiftApplies,
)
from .SubtileTailSrdTighten import (
    emitTailSrdTightenSubtile,
    emitTailSrdTightenSubtileMX,
    emitTailSrdTightenSubtileMXData,
)


def emitTailLoopScaffoldSubtile(kw, kernel, tensorParametersA, tensorParametersB):
    """Subtile-path tail-loop scaffold (PGR=0 and PGR>0).

    Emits, when NoTailLoop is False:
      - LoopCounterL = K mod DU + early-exit to SkipTailLoopL;
      - PGR>0 tail-entry gating keyed off a snapshot of
        OrigLoopCounter (taken before calculateLoopNumIter resets it):
          - origCounter == 0: branch straight to the tail body. The
            upstream `SkipSubtileMainLoop<L>` gate in
            `kernelBodySubtile` skipped preLoop/mainloop/NGLL/NLL, so
            SRDs are at K=0, LWAs at buf 0, accD at zero -- no undo
            needed.
          - 0 < origCounter < PGR: small-counter LWA realign (XOR LWA
            back to match LR buffer).
          - origCounter >= PGR: SRD +1 DU advance to undo the per-iter
            GR_INC's off-by-one at loop exit.
      - per-lane kPosBase = tidInK * numMIInUnroll;
      - re-issued DepthU-shaped GR + LR (data + MX scale);
      - per-mmak v_cmp_ge_i32 + per-MFMA v_cndmask zeroing of
        A/B/MXSA/MXSB inputs against LoopCounterL;
      - MFMAs into the existing D accumulators.

    No `closeLoop(... finalLoop=True)` is emitted: the body processes
    all K_tail in a single pass and nothing branches to
    `TailLoopEndL`. Only the post-tail `SkipTailLoopL` label is
    emitted (via `closeLoop(... emitEndLabelOnly=True)`) so the
    `calculateLoopNumIter` early-exit branch resolves.
    """
    module = Module("tailLoopSubtile")
    kPosBaseVgpr = None
    if not kernel["NoTailLoop"]:
        module.addComment2("Tail Loop")
        kw.states.inTailLoop = True
        # closeLoop's needTailEndCode path reads kw.oriLra*/oriLwa*.
        # Subtile kernels never back up LRA/LWA (per-iter LR addressing
        # is self-contained in localReadDoSubtile), so default to None.
        for _attr in ("oriLraA", "oriLraB", "oriLraM",
                      "oriLwaA", "oriLwaB", "oriLwaM"):
            if not hasattr(kw, _attr):
                setattr(kw, _attr, None)

        # loopChar drives both the PGR>0 tail-entry gating below and the
        # per-mmak early-exit branch target (`SkipTailLoop<L>`).
        unrollIdx = kw.states.unrollIdx
        loopChar = kw.states.indexChars[
            kernel["ProblemType"]["IndicesSummation"][unrollIdx]]

        # PGR>0 tail-entry gating. Three paths keyed off OrigLoopCounter
        # (which still holds K//DU at this point -- calculateLoopNumIter
        # below is what zeroes it). The `origCounter == 0` case is
        # handled upstream by the `SkipSubtileMainLoop<L>` gate in
        # `kernelBodySubtile` (skips the entire scheduler-emitted preLoop
        # / mainloop / NGLL / NLL block, so SRDs/LWAs stay at
        # setupNewTile's defaults and accD stays at
        # `initVgprTilesToZero`'s zero -- no in-tail reset needed):
        #
        #   0 < origCounter < PGR (PGR=2 origCounter==1 only):
        #     SkipOp(LE 1, NLL) in the scheduler's build_preloop jumped
        #     past GR(MT 1) into NLL, so LR (at buf 0) drained the first
        #     prefetch correctly, but preLoop's GR_INC had already flipped
        #     LWA to buf 1. Re-XOR LWA back so the tail's DTL write lands
        #     in the same LDS buf the tail's LR reads from. SRD is already
        #     at K_aligned == DU.
        #   origCounter >= PGR:
        #     NLL/NGLL drained K=[0, origCounter*DU) cleanly; last GR_INC
        #     left SRD one DU short of K_aligned. Advance SRD by 1 DU.
        #
        # Hoisted ABOVE `calculateLoopNumIter`: the SRD advance / LWA
        # realign is the mainloop-exit "GR_INC + LW swap" undo, and the
        # main-and-exit-loops block runs BEFORE the tail-entry K%DU==0
        # cmp/branch. Running it here unconditionally is harmless on the
        # early-exit path (the post-tail write-out uses SrdC/SrdD, not
        # SrdA/B/MXSA/MXSB or LocalWriteBaseAddr), and it lets the SAddU32
        # / SXorB32 chain co-issue with calculateLoopNumIter's
        # divide-and-remainder rather than serialize behind its cmp+branch.
        # Bonus: reading OrigLoopCounter directly drops the snapshot
        # SGPR a prior version held across the calculateLoopNumIter
        # call.
        if kernel["PrefetchGlobalRead"] > 0:
            pgr = kernel["PrefetchGlobalRead"]

            # Per-tensor SRD advance: A/B by depthUBytes (matches
            # SubtileGREmit._emitGRPtrUpdate_TLU0); MXSA/MXSB by
            # lrSubtileSize * lrGlobalSubtileGrid[1] (matches
            # SubtileScaleEmit.emitScaleGRPtrUpdate).
            def _tailSrdAdvanceBytes(ti):
                tc = ti.tc
                if tc in ('A', 'B'):
                    return int(ti.depthUBytes)
                # MXSA / MXSB
                return int(ti.lrSubtileSize * ti.lrGlobalSubtileGrid[1])

            tailAdvanceTiles = [kw.states.a.tileInfo, kw.states.b.tileInfo]
            if kernel["ProblemType"].get("MXBlockA", 0) > 0:
                tailAdvanceTiles.append(kw.states.mxsa.tileInfo)
            if kernel["ProblemType"].get("MXBlockB", 0) > 0:
                tailAdvanceTiles.append(kw.states.mxsb.tileInfo)
            realignTcs = ['A', 'B']
            if kernel["ProblemType"].get("MXBlockA", 0) > 0:
                realignTcs.append('MXSA')
            if kernel["ProblemType"].get("MXBlockB", 0) > 0:
                realignTcs.append('MXSB')

            smallCounterLabel = Label(
                "PGRTailSmallCounterRealign%s" % loopChar, "")
            tailEntryLabel = Label("PGRTailEntry%s" % loopChar, "")

            # origCounter == 0 reaches here via the SkipSubtileMainLoop<L>
            # gate (preLoop/mainloop/NGLL/NLL were skipped). SRDs at K=0,
            # LWAs at buf 0, accD at 0 -- branch straight to the tail body.
            module.add(SCmpEQU32(
                src0=sgpr("OrigLoopCounter"), src1=0,
                comment="origCounter == 0 (mainLoop was skipped)?"))
            module.add(SCBranchSCC1(
                labelName=tailEntryLabel.getLabelName(),
                comment="skip realign/advance; tail body runs from setupNewTile defaults"))

            module.add(SCmpLtU32(
                src0=sgpr("OrigLoopCounter"),
                src1=pgr,
                comment="0 < origCounter < PGR?"))
            module.add(SCBranchSCC1(
                labelName=smallCounterLabel.getLabelName(),
                comment="branch to small-counter realign"))

            # Large-counter path (origCounter >= PGR): advance SRD by 1 DU
            # to undo the per-iter GR_INC's off-by-one at loop exit.
            for ti in tailAdvanceTiles:
                tc = ti.tc
                inc = _tailSrdAdvanceBytes(ti)
                module.add(SAddU32(
                    dst=sgpr("Srd%s" % tc), src0=sgpr("Srd%s" % tc), src1=inc,
                    comment="advance Srd%s by 1 DU (%dB)" % (tc, inc)))
                module.add(SAddCU32(
                    dst=sgpr("Srd%s+1" % tc), src0=sgpr("Srd%s+1" % tc), src1=0,
                    comment="%s: carry" % tc))
            module.add(SBranch(labelName=tailEntryLabel.getLabelName()))

            # Small-counter realign (PGR=2 origCounter==1 only): XOR LWA<tc>
            # back so it points at the LDS buffer LR reads from. MXSA/MXSB
            # have their own Swap{MXSA,MXSB} sgprs (see
            # SubtileScaleEmit.emitScaleGRLDSSwap).
            module.add(smallCounterLabel)
            for tc in realignTcs:
                module.add(SXorB32(
                    dst=sgpr("LocalWriteBaseAddr%s" % tc),
                    src0=sgpr("LocalWriteBaseAddr%s" % tc),
                    src1=sgpr("Swap%s" % tc),
                    comment="XOR LWA%s back to match LR buffer" % tc))

            module.add(tailEntryLabel)

        module.add(kw.calculateLoopNumIter(kernel, tensorParametersA, tensorParametersB, -1))

        module.add(kw.openLoop(kernel, tensorParametersA, tensorParametersB, -1, None))

        # Tighten Srd<tc>+2 (NumRecords) at tail entry against
        # last-m-row OOR reads: bf16/fp16/int8 A/B, then MX scales,
        # then MX data. Each helper has its own gating (DTV-swizzled
        # operands are deferred). See
        # `Components/Subtile/SubtileTailSrdTighten.py` for the
        # formulas.
        module.add(emitTailSrdTightenSubtile(kw, kernel))
        module.add(emitTailSrdTightenSubtileMX(kw, kernel))
        module.add(emitTailSrdTightenSubtileMXData(kw, kernel))

        # No SRD rewind here. For PGR=0 the mainloop's per-iter GR_INC
        # leaves Srd<tc> at the K-tail's first byte after K//DU iters,
        # so the tail GR reads correct data from current SRD. For PGR>0
        # the gating above (small-counter realign / +1 DU advance) lands
        # SRD at K_aligned -- and for `origCounter == 0` the upstream
        # `SkipSubtileMainLoop<L>` gate kept SRD at K=0 / accD at 0. The
        # per-MFMA lane mask below zeros lanes past
        # LoopCounterL = K mod DU; BufferLoad bounds-clips any DU-shaped
        # over-read.

        # Allocate tail-local vgprTiles. A/B use per-mmak slice alloc
        # (`allocVgprTileRegistersForMmak` inside the loop below) so
        # only one mmak's A+B is live at a time -- the bulk alloc kept
        # every mmak's slice live and pushed large MTs (e.g. 320x288)
        # past the 256-VGPR wave-64 occupancy budget on top of the
        # D-tile overflow. MX scale tiles keep the bulk alloc: their
        # LR subtile shape `(2, 2)` packs `_mma0` into the tile id and
        # cross-mmak VGPR sharing means a per-mmak slice would re-emit
        # the same ds_read into the same VGPR.
        kw.states.a.tileInfo.initVgprTileSlots(kw, kernel)
        kw.states.b.tileInfo.initVgprTileSlots(kw, kernel)
        mxAllocTiles = []
        if kernel["ProblemType"].get("MXBlockA", 0) > 0:
            mxAllocTiles.append(kw.states.mxsa.tileInfo)
        if kernel["ProblemType"].get("MXBlockB", 0) > 0:
            mxAllocTiles.append(kw.states.mxsb.tileInfo)
        for mxTile in mxAllocTiles:
            mxTile.allocVgprTileRegisters_legacy(kw, kernel)

        # Re-issue one DepthU-shaped GR + LR. Byte-layout matches a
        # mainloop iter; the tightener above clips the last m-row's K
        # reads to in-bounds bytes (buffer-OOB returns 0 for the rest)
        # and the per-lane mask below zeros bytes past K_tail.
        module.add(globalReadDoSubtile('A', kw, kernel))
        module.add(globalReadDoSubtile('B', kw, kernel))

        # No narrow trailing-element load: `buffer_load_*_d16 ... lds`
        # is rejected by the assembler on gfx950. Instead the wide DTL
        # load + buffer-engine OOB suppression keeps in-bounds bytes in
        # LDS and leaves OOB bytes stale; `emitTailSubLaneMaskRefineSubtile`
        # zeros those stale bytes in VGPR after the local read.

        # MX scale tail GR. Host pads MXSA/MXSB with zeros and
        # pre-swizzles them (`ContractionProblemGemm::setMXScale{A,B}`
        # with padScaleTensor=true), so over-read scale bytes past
        # K_tail/mxBlock contribute 0 to the MFMA -- no LDS pre-zero.
        if kernel["ProblemType"].get("MXBlockA", 0) > 0:
            module.add(globalReadDoScaleSubtile('MXSA', kw, kernel))
        if kernel["ProblemType"].get("MXBlockB", 0) > 0:
            module.add(globalReadDoScaleSubtile('MXSB', kw, kernel))

        # Per-lane K-position base: kPosBase = tidInK * numMIInUnroll.
        # Stripped from the legacy mfmaIter setup
        # (KernelWriterAssembly.py shiftK path); subtile has
        # numReadsIterCoalesced == 1 so the kStepForCoalesced add drops.
        # Emitted after the tail GR (and before the GR-wait) so the
        # remainder/divide/multiply chain overlaps with the buffer-load
        # memory latency rather than serializing in front of it.
        matrixInstT      = min(kernel["MatrixInstM"], kernel["MatrixInstN"])
        numTileInInstA   = kernel["MatrixInstM"] // matrixInstT
        numTileInInstB   = kernel["MatrixInstN"] // matrixInstT
        numMIInputA      = kernel["MIInputPerThreadA"]
        numMIInputB      = kernel["MIInputPerThreadB"]
        numMIInUnroll    = max(numMIInputA // numTileInInstA, numMIInputB // numTileInInstB)
        dividerFortidInK = kernel["MatrixInstN"] * kernel["MatrixInstB"]
        numReadsIterCoalesced = max(kw.states.numReadsIterCoalescedA,
                                    kw.states.numReadsIterCoalescedB)

        kPosBaseVgpr = kw.vgprPool.checkOut(1, "kReg_first")
        with kw.allocTmpSgpr(1) as tmpSgprInfo:
            module.add(vectorStaticRemainder(-1, kPosBaseVgpr, "Serial",
                                             kernel["WavefrontSize"], None, tmpSgprInfo,
                                             "kPosBase = Serial %% WavefrontSize (lane-id in wave)"))
            module.add(vectorStaticDivide(kPosBaseVgpr, kPosBaseVgpr,
                                          dividerFortidInK, None,
                                          "kPosBase = tidInK = lane-id / (MatrixInstN * MatrixInstB)"))
        with kw.allocTmpSgpr(1) as tmpSgprInfo:
            module.add(vectorStaticMultiply(vgpr(kPosBaseVgpr), vgpr(kPosBaseVgpr),
                                            numMIInUnroll * numReadsIterCoalesced,
                                            tmpSgprInfo,
                                            "kPosBase = tidInK * numMIInUnroll (K-element base)"))

        # Precompute every per-(operand, mmak, ir) K-tail byte mask
        # ONCE before the swait+sbarrier below (when the byte refine
        # path applies and the precompute is enabled). The mask chain
        # only reads `LoopCounterL` and `kPosBaseVgpr` -- it does NOT
        # consume any DTL/LDS data -- so hoisting it above the
        # `tail GR: wait for DTL writes to LDS` drain lets the
        # cmp/cndmask chain co-issue with the buffer-load latency
        # rather than serializing behind it.
        #
        # The per-mmak loop below collapses to a pure
        # `v_and_b32 vIdx, vMask[..], vIdx` apply step. Persistent
        # VGPR cost for the common bf16/bf16 case (shared A/B masks):
        # `numMmaks * (numMIInUnroll // elementsPerVgpr)` (e.g. 8 vgprs
        # on MT256/DU=64, scales linearly with DU); offset by removing
        # the inline kPosCur+maskVgpr+seedVgpr scratch the legacy helper
        # held per mmak iter and by hoisting `numMmaks` cmp+cndmask
        # chains out of the loop.
        #
        # Gated by `SubtileTailMaskPrecompute` (default True). Set to
        # False to fall back to the legacy per-mmak inline mask chain.
        useMaskPrecompute = kernel.get("SubtileTailMaskPrecompute", True)
        byteRefineApplies = subtileTailByteShiftApplies(kernel, numMIInUnroll)
        precomputedMaskMap = None
        precomputedMaskVgprs = []
        if useMaskPrecompute and byteRefineApplies:
            precomputeModule, precomputedMaskMap, precomputedMaskVgprs = \
                emitTailSubLaneMaskPrecomputeSubtile(
                    kw, kernel, kPosBaseVgpr,
                    kw.states.a.tileInfo.localMMATileGrid[1],
                    kernel["MatrixInstK"], numMIInUnroll)
            module.add(precomputeModule)

        module.add(SWaitCnt(vlcnt=0, vscnt=-1,
                            comment="tail GR: wait for DTL writes to LDS"))
        module.add(SBarrier(comment="tail GR: LDS sync before LR"))

        # MX scale LR fires once up front: scale VGPRs are mmak-shared
        # (LR subtile shape `(2, 2)` packs mmak//subtileKShape into the
        # scale group), so per-mmak re-issue would re-write the same
        # VGPR. A/B LR runs inside the per-mmak loop below for the
        # per-slice checkOut/release.
        if kernel["ProblemType"].get("MXBlockA", 0) > 0:
            module.add(localReadDoScaleSubtile('MXSA', kw, kernel))
        if kernel["ProblemType"].get("MXBlockB", 0) > 0:
            module.add(localReadDoScaleSubtile('MXSB', kw, kernel))

        # Per-mmak: lane mask + MFMA. Mirrors Subtile/Kernel.py:emitMfmaCode
        # including D-tile index resolution so the masked VGPRs feed the
        # same MFMA shape as the mainloop.
        #
        # Cndmask is HOISTED out of the (mma1, mma0) MFMA grid: for a given
        # mmak, atiles depend only on (mma0, mmak) and btiles only on
        # (mma1, mmak), so the same A vgprs would otherwise be cndmasked
        # `len(mma1)` times and the same B vgprs `len(mma0)` times. We
        # emit one cndmask per unique vgpr per mmak (tracked via
        # `seenVgpr`) and then run a cndmask-free MFMA grid.
        tiA = kw.states.a.tileInfo
        tiB = kw.states.b.tileInfo
        dtileInfo = kw.states.d.tileInfo
        tiMXSA = kw.states.mxsa.tileInfo if kernel["ProblemType"].get("MXBlockA", 0) > 0 else None
        tiMXSB = kw.states.mxsb.tileInfo if kernel["ProblemType"].get("MXBlockB", 0) > 0 else None
        hasScaleA = tiMXSA is not None and tiMXSA.mxBlock > 0
        hasScaleB = tiMXSB is not None and tiMXSB.mxBlock > 0

        lrSubtileShapeA = tiA.lr.subtileShape
        lrSubtileShapeB = tiB.lr.subtileShape
        miK = kernel["MatrixInstK"]

        numMmaTilePerSubtileA = lrSubtileShapeA[0] * lrSubtileShapeA[1]
        numMmaTilePerSubtileB = lrSubtileShapeB[0] * lrSubtileShapeB[1]
        lrLocalGridA0 = tiA.localMMATileGrid[0] // lrSubtileShapeA[0]
        lrLocalGridB0 = tiB.localMMATileGrid[0] // lrSubtileShapeB[0]
        subtileKShape = lrSubtileShapeA[1] if hasScaleA else 0
        subtileKGrid = tiA.localSubtileGrid[1] if hasScaleA else 0

        def _aTileId(mma0_, mmak_):
            aSId0 = mma0_ // lrSubtileShapeA[0]
            aSId1 = mmak_ // lrSubtileShapeA[1]
            _mmak = mmak_ % lrSubtileShapeA[1]
            return (aSId1 * lrLocalGridA0 + aSId0) * numMmaTilePerSubtileA + _mmak

        def _bTileId(mma1_, mmak_):
            bSId0 = mma1_ // lrSubtileShapeB[0]
            bSId1 = mmak_ // lrSubtileShapeB[1]
            _mmak = mmak_ % lrSubtileShapeB[1]
            return (bSId1 * lrLocalGridB0 + bSId0) * numMmaTilePerSubtileB + _mmak

        def _scaleAVgpr(mma0_, mmak_):
            scaleGroup = (mma0_ // 2) * subtileKGrid + mmak_ // subtileKShape
            return tiMXSA.vgprTiles[MX_SCALE_TILES_PER_VGPR * scaleGroup].regList.indices[0] \
                   if hasScaleA and tiMXSA.mxBlock else -1

        def _scaleBVgpr(mma1_, mmak_):
            scaleGroup = (mma1_ // 2) * subtileKGrid + mmak_ // subtileKShape
            return tiMXSB.vgprTiles[MX_SCALE_TILES_PER_VGPR * scaleGroup].regList.indices[0] \
                   if hasScaleB and tiMXSB.mxBlock else -1

        laneSGPRCount = kw.states.laneSGPRCount
        for mmak in range(tiA.localMMATileGrid[1]):
            # Per-mmak: alloc this mmak's A/B slice, ds_read its K-slice
            # from LDS, wait, mask + MFMA, then free the slice. Peak VGPR
            # pressure tracks one slice instead of the full vgprTiles
            # range.
            aMmakSlice = tiA.allocVgprTileRegistersForMmak(kw, kernel, mmak)
            bMmakSlice = tiB.allocVgprTileRegistersForMmak(kw, kernel, mmak)
            # ds_read slice (mmak -> (sId1, du)):
            #   sId1, du = divmod(mmak, subtileShape[1])
            #   mfmaId   = getSubtileShapeLinearId(du, 0)
            # Inlined here (no shared helper) because this iteration shape
            # is tail-scaffold-specific.
            for tc, ti in (('A', tiA), ('B', tiB)):
                subKShape = ti.subtileShape[1]
                sId1     = mmak // subKShape
                du       = mmak %  subKShape
                mfmaId   = ti.getSubtileShapeLinearId(du, 0)
                for sId0 in range(ti.localSubtileGrid[0]):
                    tileIdx = ti.lrTileIndexForSubtile(sId0, sId1, mfmaId)
                    dstTile = ti.vgprTiles[tileIdx]
                    module.add(emitSingleDsRead(ti, sId0, sId1, du, dstTile))
            module.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1,
                                comment="tail LR mmak=%u: wait for ds_reads before lane mask + MFMA" % mmak))

            # Two K-tail masking paths share this scaffold:
            #   - Sub-lane byte refine (`subtileTailByteShiftApplies`):
            #     owns the per-lane mask end-to-end. Its mod=0 step is
            #     equivalent to the coarse `kPos vs LoopCounterL` cmp +
            #     per-VGPR cndmask the legacy path emitted, so we skip
            #     the coarse path entirely (#5) when the byte refine
            #     fires. Predicate excludes MX, so MXSA/MXSB cndmasks
            #     are never needed in this branch.
            #   - Coarse path (default): per-mmak cmp + per-VGPR
            #     cndmask, including MXSA/MXSB when scales are present.
            if byteRefineApplies:
                # Group A/B operand VGPRs by ir slot within each
                # (mma0|1, mmak) tile -- one mask chain per (ir, operand).
                # Sub-lane refine assumes the same VGPR holds the same
                # K-slot across mma{0,1} tiles (matches the legacy hoisted
                # cndmask layout); the assert below pins that invariant.
                aIndicesByIr = {}
                bIndicesByIr = {}
                seenAByteVgpr = {}
                seenBByteVgpr = {}
                for mma0 in range(tiA.localMMATileGrid[0]):
                    aIndices = tiA.vgprTiles[_aTileId(mma0, mmak)].regList.indices
                    for ir, vIdx in enumerate(aIndices):
                        if vIdx in seenAByteVgpr:
                            assert seenAByteVgpr[vIdx] == ir, (
                                f"VGPR {vIdx} seen at ir {seenAByteVgpr[vIdx]} and "
                                f"{ir}; sub-lane refine dedup assumes one ir per vIdx")
                            continue
                        seenAByteVgpr[vIdx] = ir
                        aIndicesByIr.setdefault(ir, []).append(vIdx)
                for mma1 in range(tiB.localMMATileGrid[0]):
                    bIndices = tiB.vgprTiles[_bTileId(mma1, mmak)].regList.indices
                    for ir, vIdx in enumerate(bIndices):
                        if vIdx in seenBByteVgpr:
                            assert seenBByteVgpr[vIdx] == ir, (
                                f"VGPR {vIdx} seen at ir {seenBByteVgpr[vIdx]} and "
                                f"{ir}; sub-lane refine dedup assumes one ir per vIdx")
                            continue
                        seenBByteVgpr[vIdx] = ir
                        bIndicesByIr.setdefault(ir, []).append(vIdx)
                if useMaskPrecompute:
                    # Hot path: per-mmak step is just `v_and_b32` against
                    # the precomputed VGPR (chain emitted up front).
                    module.add(emitTailSubLaneMaskApplySubtile(
                        kw, mmak, precomputedMaskMap, aIndicesByIr, bIndicesByIr))
                else:
                    # `SubtileTailMaskPrecompute=False` reversibility path.
                    module.add(emitTailSubLaneMaskRefineSubtile(
                        kw, kernel, kPosBaseVgpr, mmak, miK, numMIInUnroll,
                        aIndicesByIr, bIndicesByIr))
            else:
                with kw.allocTmpSgpr(laneSGPRCount,
                                     alignment=laneSGPRCount) as tmpSgprInfo:
                    maskSgpr = tmpSgprInfo.idx
                    module.add(emitTailKPosCmpSubtile(
                        kw, kPosBaseVgpr, mmak, miK, maskSgpr))

                    # Hoisted lane-mask: cndmask each unique vgpr ONCE per
                    # mmak. atiles & scaleA vary only with mma0 (fixed
                    # across mma1); btiles & scaleB vary only with mma1
                    # (fixed across mma0).
                    seenVgpr = set()

                    def _cndmaskVgpr(idx, label):
                        if idx < 0 or idx in seenVgpr:
                            return
                        seenVgpr.add(idx)
                        module.add(VCndMaskB32(
                            dst=vgpr(idx), src0=vgpr(idx), src1=0,
                            src2=sgpr(maskSgpr, laneSGPRCount),
                            comment="zero %s[%u] if K_idx >= sizeL" % (label, idx)))

                    for mma0 in range(tiA.localMMATileGrid[0]):
                        for idx in tiA.vgprTiles[_aTileId(mma0, mmak)].regList.indices:
                            _cndmaskVgpr(idx, "ValuA")
                        if hasScaleA:
                            _cndmaskVgpr(_scaleAVgpr(mma0, mmak), "ValuMXSA")

                    for mma1 in range(tiB.localMMATileGrid[0]):
                        for idx in tiB.vgprTiles[_bTileId(mma1, mmak)].regList.indices:
                            _cndmaskVgpr(idx, "ValuB")
                        if hasScaleB:
                            _cndmaskVgpr(_scaleBVgpr(mma1, mmak), "ValuMXSB")

            # MFMA grid -- inputs already lane-masked above.
            for mma1 in range(tiB.localMMATileGrid[0]):
                for mma0 in range(tiA.localMMATileGrid[0]):
                    atiles = tiA.vgprTiles[_aTileId(mma0, mmak)]
                    btiles = tiB.vgprTiles[_bTileId(mma1, mmak)]
                    dtiles = dtileInfo.vgprTiles[mma0 + mma1 * dtileInfo.localMMATileGrid[0]]

                    if hasScaleA:
                        scaleAVgpr = _scaleAVgpr(mma0, mmak)
                        scaleBVgpr = _scaleBVgpr(mma1, mmak)
                        sAsel = (mma0 % 2) + 2 * (mmak % 2)
                        sBsel = (mma1 % 2) + 2 * (mmak % 2)
                    else:
                        scaleAVgpr = scaleBVgpr = -1
                        sAsel = sBsel = -1

                    module.add(emitMfmaInstruction(
                        kw, kernel, atiles, btiles, dtiles, dtiles,
                        scaleAVgpr=scaleAVgpr, scaleBVgpr=scaleBVgpr,
                        scaleAsel=sAsel, scaleBsel=sBsel,
                        comment="tail MFMA C[%u,%u] += A[%u,%u] * B[%u,%u] (mmak=%u)" %
                                (mma0, mma1, mma0, mmak, mmak, mma1, mmak)))

            # Release this mmak's A/B slice. The dealloc is emit-time
            # pool accounting only (no instruction), so a runtime
            # early-exit branch out below stays balanced: the branch
            # target downstream of the loop sees a clean pool either
            # way.
            tiA.freeVgprTileRegistersForMmak(kw, kernel, aMmakSlice)
            tiB.freeVgprTileRegistersForMmak(kw, kernel, bMmakSlice)

            # Per-mmak early exit: when K_tail (=LoopCounterL=K mod DU)
            # is fully consumed by the mmaks already issued
            # (K_tail <= MIK * (mmak+1)), every subsequent MFMA would
            # feed already-zeroed lanes. Skip them. Omit after the
            # final mmak: closeLoop's natural exit covers that case.
            if mmak + 1 < tiA.localMMATileGrid[1]:
                consumedK = miK * (mmak + 1)
                # `consumedK` exceeds the gfx950 VOPC/SOPC inline range
                # (-16..64) for bf16 (MIK=32) once subIterK>=2 and for fp4
                # (MIK=128) at every mmak; stage non-inline literals
                # through a scratch sgpr so the cmp always has inline-or-
                # sgpr src1.
                module.add(kw._emitSubtileScalarCmpLitOrStaged(
                    SCmpLeU32, sgpr("LoopCounterL"), consumedK,
                    "LoopCounterL <= MIK*(subIterK+1)?"))
                module.add(SCBranchSCC1(
                    labelName=Label.getFormatting("SkipTailLoop%s" % loopChar),
                    comment="early-exit tail after subIterK=%u (no valid K left)" % mmak))

        # Release the precomputed K-tail mask VGPRs (held across every
        # mmak iter). `SkipTailLoopL` sits past this cleanup, so the
        # emit-time pool stays balanced under either runtime exit.
        for vMask in precomputedMaskVgprs:
            kw.vgprPool.checkIn(vMask)
        precomputedMaskVgprs = []
        precomputedMaskMap = None

        # The per-mmak alloc/free above already returned every A/B
        # vgprTile to the pool; clear the slot lists.
        kw.states.a.tileInfo.vgprTiles = []
        kw.states.b.tileInfo.vgprTiles = []
        for mxTile in mxAllocTiles:
            mxTile.deallocVgprTileRegisters_legacy(kw, kernel)

        # No `closeLoop(... finalLoop=True)` here. The subtile tail body
        # processes the entire K_tail in a single pass via the `mmak`
        # loop above (every lane mask was emitted against the current
        # `LoopCounterL = K mod DU` snapshot), so the per-iter
        # `s_sub_i32 LoopCounterL, ..., MIK` + back-edge to
        # `TailLoopBeginL` that `closeLoop` would emit is dead code:
        # the only useful effect would be the `TailLoopEndL:` label,
        # which nothing branches to. The standalone `OrigLoopCounter`
        # increment closeLoop emits is for the legacy LRO-damage
        # recovery block, which is bypassed for `UseSubtileImpl=1`
        # kernels (see `closeLoop`'s `needTailEndCode`/subtile branch).
        kw.vgprPool.checkIn(kPosBaseVgpr)
        kw.states.inTailLoop = False
    # Always emit SkipTailLoopL so the early-exit branch resolves
    # regardless of NoTailLoop.
    module.add(kw.closeLoop(kernel, tensorParametersA, tensorParametersB, -1, finalLoop=True, emitEndLabelOnly=True))
    return module
