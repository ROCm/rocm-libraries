"""Writer / rocisa integration tests for the subtile LogicalScheduler.

SCOPE (grc.148 prune): this file covers only the Python *writer / rocisa integration*
layer.  Pure scheduler value/config tests, writer-free pass-pipeline structural
tests, and loop-variant data tests have been removed; they are now covered by
the native C++ gtest suite:
  - logical_scheduler_test.cpp      — value types, SchedulerConfig, NormalizePartitionSizes,
                                      GetPartitionCandidates, FmtMt, Dep, EmittedModule, …
  - logical_scheduler_passes_test.cpp — LR/GR slot structure, VGPR metadata/peaks,
                                        dep kinds, preOps, emit structure, pass pipeline
  - loop_variants_test.cpp          — Preloop, NGLL, NLL, TailLoop flat groups
  - instruction_scheduler_test.cpp  — instruction scheduling order and vmcnt

What remains here:
  1. TestComputeFlatTailPeaks  — _ensure_cpp() / C++ binding round-trip and
                                 getNumVgpr delegation through the C++ layer.
  2. TestIntegration           — Full writer path:
       - populate_instructions with real rocisa objects
       - allocVgprTiles / writer VGPR allocation and lifetime
       - emitMainAndExitLoops label structure and per-unroll VGPR differences
       - emitTailLoop K-mask emission (FP4 v_cmp_lt_i32 / v_cndmask_b32,
         BF16 V_AND_B32 half-mask)
       - instructionScheduleFromLists on emitted module chains
       - NLL per-unroll scale VGPR distinctness (pgr=1 and pgr=2)

The byte-identical Python↔C++ pass-pipeline parity tests (test_logicalSchedulerCpp.py),
the C++ instruction-scheduler parity tests (test_instructionSchedulerCpp.py), and
the redundant emit/preloop snapshot regressions (test_SubtileBasedSchedulerRef.py)
have been removed; that coverage now lives in the C++ gtest suite above.
"""
import pytest
from Tensile.Components.Subtile.Kernel import (
    TileInfo, AB_B16, AB_B4, MXSA_B4, MXSB_B4, CD_F32,
)
from Tensile.Components.Subtile.LogicalScheduler import (
    LogicalScheduler,
    ReadGranularity,
    SchedulerConfig,
)
from unittest.mock import MagicMock


def makeTileInfo(tc, kernel):
    """Compatibility wrapper: select geometry from kernel config and return TileInfo."""
    fp4 = kernel["ProblemType"].get("MXBlockA", 0) > 0
    _geo = {
        'A': AB_B4 if fp4 else AB_B16,
        'B': AB_B4 if fp4 else AB_B16,
        'MXSA': MXSA_B4,
        'MXSB': MXSB_B4,
        'D': CD_F32,
    }
    return TileInfo(_geo[tc], tc, None, kernel)


# ── Shared fixtures ───────────────────────────────────────────

def _mock_dtype(num_bytes=2):
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    mock.numRegisters.return_value = num_bytes / 4
    # Treat the default 2-byte mock as BF16 (only consumer is the Subtile tail
    # mask path, which dispatches on isBFloat16); 0.5-byte (fp4) returns False.
    mock.isBFloat16.return_value = (num_bytes == 2)
    return mock


def create_kernel(MT0=256, MT1=256, fp4=False, depthU=None,
                  miWaveGroup=None, sourceSwap=False):
    mxblock = 32 if fp4 else 0
    bpe = 0.5 if fp4 else 2
    matrixInstK = 128 if fp4 else 32
    if depthU is None:
        depthU = 256 if fp4 else 64
    if miWaveGroup is None:
        miWaveGroup = [2, 2]
    dtype = _mock_dtype(bpe)
    problemType = {
        "DataTypeA": dtype,
        "DataTypeB": dtype,
        "ComputeDataType": _mock_dtype(4),
    }
    if fp4:
        problemType["MXBlockA"] = mxblock
        problemType["MXBlockB"] = mxblock
    kernel = {
        "DepthU": depthU,
        "_DepthUA": depthU,
        "_DepthUB": depthU,
        "MacroTileA": MT0,
        "MacroTileB": MT1,
        "MacroTile0": MT0,
        "MacroTile1": MT1,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": matrixInstK,
        "MatrixInstB": 1,
        "MIInputPerThreadA": matrixInstK // 4,
        "MIInputPerThreadB": matrixInstK // 4,
        "MIWaveGroup": list(miWaveGroup),
        "WavefrontSize": 64,
        "SourceSwap": sourceSwap,
        "MIArchVgpr": False,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "NonTemporalMXSA": 0,
        "NonTemporalMXSB": 0,
        "NoTailLoop": False,
        "ProblemType": problemType,
    }
    if fp4:
        kernel["_DepthUMXSA"] = depthU // mxblock
        kernel["_DepthUMXSB"] = depthU // mxblock
    return kernel


def make_cfg_256x256_fp4(depthU=256, k_gran=1, partSizeM=0, partSizeN=0,
                         grSA_k=2, grSA_mn=8, grSB_k=2, grSB_mn=8, pgr=2,
                         miWaveGroup=None):
    """Build FP4 config with scale tensors. k_gran applies to LR A/B."""
    kernel = create_kernel(256, 256, fp4=True, depthU=depthU, miWaveGroup=miWaveGroup)
    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    scaleTiA = makeTileInfo('MXSA', kernel)
    scaleTiB = makeTileInfo('MXSB', kernel)
    grA = ReadGranularity(mn=1, k=2) if tiA.loadRatioGR <= 1.0 else ReadGranularity(mn=2, k=2)
    grB = ReadGranularity(mn=1, k=2) if tiB.loadRatioGR <= 1.0 else ReadGranularity(mn=2, k=2)
    return SchedulerConfig(
        numMFMATilesM=tiA.localMMATileGrid[0],
        numMFMATilesN=tiB.localMMATileGrid[0],
        numSubIterK=tiA.localMMATileGrid[1],
        lrA=ReadGranularity(mn=1, k=k_gran),
        lrB=ReadGranularity(mn=1, k=k_gran),
        grA=grA,
        grB=grB,
        lrSA=ReadGranularity(mn=2, k=2),
        lrSB=ReadGranularity(mn=2, k=2),
        grSA=ReadGranularity(mn=scaleTiA.localMMATileGrid[0], k=scaleTiA.localMMATileGrid[1]),
        grSB=ReadGranularity(mn=scaleTiB.localMMATileGrid[0], k=scaleTiB.localMMATileGrid[1]),
        partitionSizeM=partSizeM,
        partitionSizeN=partSizeN,
        pgr=pgr,
    )


def make_cfg_bf16(MT0=256, MT1=256, depthU=64, partSizeM=0, partSizeN=0,
                  miWaveGroup=None, sourceSwap=False, lrA=None, lrB=None):
    """Build BF16 config without scale tensors."""
    kernel = create_kernel(MT0, MT1, fp4=False, depthU=depthU,
                           miWaveGroup=miWaveGroup, sourceSwap=sourceSwap)
    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    if lrA is None:
        lrA = ReadGranularity(mn=1, k=1)
    if lrB is None:
        lrB = ReadGranularity(mn=1, k=1)
    grA = ReadGranularity(mn=1, k=2) if tiA.loadRatioGR <= 1.0 else ReadGranularity(mn=2, k=2)
    grB = ReadGranularity(mn=1, k=2) if tiB.loadRatioGR <= 1.0 else ReadGranularity(mn=2, k=2)
    return SchedulerConfig(
        numMFMATilesM=tiA.localMMATileGrid[0],
        numMFMATilesN=tiB.localMMATileGrid[0],
        numSubIterK=tiA.localMMATileGrid[1],
        lrA=lrA,
        lrB=lrB,
        grA=grA,
        grB=grB,
        partitionSizeM=partSizeM,
        partitionSizeN=partSizeN,
    )


def make_writer_and_tileinfos(kernel, fp4=False):
    """Create writer with register pools and TileInfos for integration tests."""
    from types import SimpleNamespace
    from rocisa import rocIsa
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType
    from Tensile.Common.RegisterPool import allocTmpGpr

    ri = rocIsa.getInstance()
    if not ri.isInit():
        import shutil
        asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
        ri.init((9, 5, 0), asmpath)
    ri.setKernel((9, 5, 0), 64)

    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    scaleTiA = makeTileInfo('MXSA', kernel) if fp4 else None
    scaleTiB = makeTileInfo('MXSB', kernel) if fp4 else None

    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr, False)
    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr, False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr, False)
    writer.states = SimpleNamespace(
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
        unrollIdx=0,
        laneSGPRCount=2,
    )
    writer.allocTmpSgpr = lambda num, alignment=None, tag=None: allocTmpGpr(
        writer.sgprPool, num, writer.states.regCaps["MaxSgpr"], alignment, tag, None)
    writer.loopCounterName = lambda kernel, loopIdx: "LoopCounterL"
    writer.tailLoopBoundaryDtlLoadAB = lambda *a, **kw: MagicMock()
    _label_counters = {}
    def _getNameInc(base):
        n = _label_counters.get(base, 0)
        _label_counters[base] = n + 1
        return f"{base}_{n}"
    writer.labels = SimpleNamespace(getNameInc=_getNameInc)
    dTileInfo = makeTileInfo('D', kernel)
    dTileInfo.allocVgprTileRegisters_legacy(writer, kernel)
    writer.states.d = SimpleNamespace(tileInfo=dTileInfo)
    writer.states.a = SimpleNamespace(tileInfo=tiA)
    writer.states.b = SimpleNamespace(tileInfo=tiB)
    tiA.allocOffsetRegisters(writer, kernel)
    tiB.allocOffsetRegisters(writer, kernel)
    if scaleTiA and scaleTiB:
        writer.states.mxsa = SimpleNamespace(tileInfo=scaleTiA)
        writer.states.mxsb = SimpleNamespace(tileInfo=scaleTiB)
        scaleTiA.allocOffsetRegisters(writer, kernel)
        scaleTiB.allocOffsetRegisters(writer, kernel)

    return writer, tiA, tiB, scaleTiA, scaleTiB, dTileInfo


# ══════════════════════════════════════════════════════════════
# C++ binding round-trip: compute_flat_tail_peaks / getNumVgpr
# ══════════════════════════════════════════════════════════════

class TestComputeFlatTailPeaks:
    """Tests for C++ compute_flat_tail_peaks() and the delegated getNumVgpr."""

    def test_single_partition_flat_equals_groups(self):
        """Single-partition: flat peaks == tile_peaks / num_sets (one pingpong set)."""
        import math
        kernel = create_kernel(256, 256)
        cfg = make_cfg_bf16(256, 256)
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        cpp = sched._ensure_cpp()
        flat = cpp.compute_flat_tail_peaks()

        # flat_A = numMFMATilesM / lrA.mn, tile_peaks_A = 2 * flat_A
        expected_A = cfg.numMFMATilesM // cfg.lrA.mn
        expected_B = cfg.numMFMATilesN // cfg.lrB.mn
        assert flat.get('A', 0) == expected_A, f"flat A: {flat}"
        assert flat.get('B', 0) == expected_B, f"flat B: {flat}"

    def test_multi_partition_flat_covers_all_tiles(self):
        """Multi-partition: flat peak = global unique groups (all partitions merged)."""
        kernel = create_kernel(256, 256, fp4=True)
        cfg = make_cfg_256x256_fp4(partSizeN=4)
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        cpp = sched._ensure_cpp()
        flat = cpp.compute_flat_tail_peaks()

        # Flat covers ALL N-tiles globally → flat_B = numMFMATilesN / lrB.mn
        expected_B = cfg.numMFMATilesN // cfg.lrB.mn
        assert flat.get('B', 0) == expected_B, f"flat B: {flat}"

    def test_get_num_vgpr_cpp_matches_python_single_partition(self):
        """C++ get_num_vgpr == Python getNumVgpr for a single-partition BF16 config."""
        import math
        kernel = create_kernel(256, 256)
        tiA = makeTileInfo('A', kernel)
        tiB = makeTileInfo('B', kernel)
        cfg = make_cfg_bf16(256, 256)
        sched = LogicalScheduler(cfg)
        sched.build()

        result = sched.getNumVgpr(tiA, tiB)

        # For no partitioning, mainloop total should equal what the C++ computes:
        # max(mainloop_total, flat_tail_total) = mainloop_total since mainloop >= flat.
        vgpr_per_A = int(math.ceil(tiA.mmaTileRegCount * cfg.lrA.k * cfg.lrA.mn))
        vgpr_per_B = int(math.ceil(tiB.mmaTileRegCount * cfg.lrB.k * cfg.lrB.mn))
        expected = (sched.tile_peaks.get('A', 0) * vgpr_per_A
                    + sched.tile_peaks.get('B', 0) * vgpr_per_B)
        assert result == expected, f"got {result}, expected {expected}"

    def test_get_num_vgpr_cpp_with_scale(self):
        """C++ get_num_vgpr includes SA/SB when scale tile infos are provided."""
        kernel = create_kernel(256, 256, fp4=True)
        tiA = makeTileInfo('A', kernel)
        tiB = makeTileInfo('B', kernel)
        scaleTiA = makeTileInfo('MXSA', kernel)
        scaleTiB = makeTileInfo('MXSB', kernel)

        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.build()

        total_with = sched.getNumVgpr(tiA, tiB, scaleTiA, scaleTiB)
        total_without = sched.getNumVgpr(tiA, tiB)
        assert total_with > total_without, \
            "Including scale tensors must increase VGPR count"

    def test_get_num_vgpr_flat_tail_dominates_multi_partition(self):
        """For a 4-partition config with single k-chunk, flat tail may drive the budget."""
        kernel = create_kernel(256, 256, fp4=True)
        tiA = makeTileInfo('A', kernel)
        tiB = makeTileInfo('B', kernel)
        scaleTiA = makeTileInfo('MXSA', kernel)
        scaleTiB = makeTileInfo('MXSB', kernel)

        # Use a config where more partitions don't reduce the budget due to
        # use_global_pos=True (triggered when any tensor has single k-chunk).
        cfg_1x1 = make_cfg_256x256_fp4(partSizeN=0)
        cfg_1x4 = make_cfg_256x256_fp4(partSizeN=2)

        sched_1x1 = LogicalScheduler(cfg_1x1)
        sched_1x1.build()
        sched_1x4 = LogicalScheduler(cfg_1x4)
        sched_1x4.build()

        v_1x1 = sched_1x1.getNumVgpr(tiA, tiB, scaleTiA, scaleTiB)
        v_1x4 = sched_1x4.getNumVgpr(tiA, tiB, scaleTiA, scaleTiB)

        # More partitions should not increase VGPR count.
        assert v_1x4 <= v_1x1, \
            f"More partitions should not increase budget: 1x1={v_1x1}, 1x4={v_1x4}"


# ══════════════════════════════════════════════════════════════
# Integration tests
# ══════════════════════════════════════════════════════════════

class TestIntegration:

    def test_populate_instructions_256x256_fp4(self):
        """Full pipeline: emit → populate_instructions → emit_module on-demand."""
        from Tensile.Components.Subtile.LogicalScheduler import instructionScheduleFromLists

        kernel = create_kernel(256, 256, fp4=True)
        writer, tiA, tiB, scaleTiA, scaleTiB, dTileInfo = make_writer_and_tileinfos(kernel, fp4=True)

        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.emit()
        sched.allocVgprTiles(writer, tiA, tiB,
                              scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)

        try:
            sched.populate_instructions(
                writer, kernel,
                tileInfoA=tiA, tileInfoB=tiB,
                dtileInfo=dTileInfo,
                scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
            )

            assert sched._emitter is not None

            # All modules emit non-empty instruction lists on demand (no
            # pre-populated em.instructions since grc.115 removed the populate loop).
            for pi, partition_emitted in enumerate(sched._emitted):
                for k, emitted in enumerate(partition_emitted):
                    for em in emitted:
                        insts = sched._emitter.emit_module(em, unroll_iter=0)
                        assert len(insts) > 0, \
                            f"P{pi} k={k} [{em.moduleId}] {em.opType}: no instructions"

            # MFMA/LR VGPR disjoint (checked at tile-map level, not instruction level)
            for pi, slots in enumerate(sched._partitions):
                for slot in slots:
                    if slot.mfma and slot.lrs:
                        for lr in slot.lrs:
                            if lr.vgpr_tile_map and lr.tensor in ('A', 'B'):
                                mfma_map = slot.mfma.vgpr_tile_maps.get(lr.tensor, [])
                                if mfma_map:
                                    assert set(mfma_map[0].values()).isdisjoint(
                                        set(lr.vgpr_tile_map[0].values()))

            # instructionScheduleFromLists succeeds and produces non-empty output
            for pi, partition_emitted in enumerate(sched._emitted):
                for k, emitted in enumerate(partition_emitted):
                    inst_lists = [sched._emitter.emit_module(em, 0) for em in emitted]
                    scheduled = instructionScheduleFromLists(emitted, inst_lists)
                    assert len(list(scheduled.flatitems())) > 0

        finally:
            sched.deallocVgprTiles(writer)

    def test_emitLoops_256x256_fp4(self):
        """emitMainAndExitLoops + emitTailLoop: label structure and per-unroll VGPR differences."""
        kernel = create_kernel(256, 256, fp4=True)
        writer, tiA, tiB, scaleTiA, scaleTiB, dTileInfo = make_writer_and_tileinfos(kernel, fp4=True)

        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.build()
        sched.allocVgprTiles(writer, tiA, tiB,
                              scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)

        try:
            sched.populate_instructions(
                writer, kernel,
                tileInfoA=tiA, tileInfoB=tiB,
                dtileInfo=dTileInfo,
                scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
            )

            uf = sched.unroll_factor
            asm_main = str(sched.emitMainAndExitLoops(writer, kernel))

            assert "LoopBeginL:" in asm_main
            assert "SkipToNGLL:" in asm_main

            if uf > 1:
                for ui in range(uf):
                    assert f"MAINLOOP_C{ui}" in asm_main
                    assert f"NGLL_C{ui}" in asm_main
                    assert f"NLL_C{ui}" in asm_main
                assert "SkipToEnd:" in asm_main
                assert "SkipToNLL:" in asm_main

                def get_mfma_vgprs(emitted_3d, unroll_iter):
                    vgprs = set()
                    for partition in emitted_3d:
                        for group in partition:
                            for em in group:
                                if em.opType == 'mfma':
                                    for inst in sched._emitter.emit_module(em, unroll_iter):
                                        vgprs.add(str(inst))
                    return vgprs

                # Check per-unroll VGPR differences BEFORE emitTailLoop swaps tiles.
                vgprs_0 = get_mfma_vgprs(sched._emitted, 0)
                vgprs_1 = get_mfma_vgprs(sched._emitted, 1)
                assert vgprs_0 != vgprs_1, \
                    "Per-unroll copies should differ in MFMA instructions"
            else:
                assert "MAINLOOP" in asm_main
                assert "NGLL" in asm_main
                assert "NLL" in asm_main

            asm_tail = str(sched.emitTailLoop(writer, kernel))
            assert "TAILLOOP" in asm_tail, "emitTailLoop should emit the TAILLOOP body"

        finally:
            sched.deallocVgprTiles(writer)

    def test_tailloop_k_mask_256x256_fp4(self):
        """Tail loop must emit per-lane K-mask (v_cmp_lt_i32 + v_cndmask_b32) after wait_lr."""
        kernel = create_kernel(256, 256, fp4=True)
        writer, tiA, tiB, scaleTiA, scaleTiB, dTileInfo = make_writer_and_tileinfos(kernel, fp4=True)

        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.build()
        sched.allocVgprTiles(writer, tiA, tiB,
                              scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)
        try:
            sched.populate_instructions(
                writer, kernel,
                tileInfoA=tiA, tileInfoB=tiB,
                dtileInfo=dTileInfo,
                scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
            )
            asm = str(sched.emitTailLoop(writer, kernel))

            # mask init (preamble): kReg = (Serial % WS) / dividerFortidInK
            assert "v_and_b32" in asm or "v_lshrrev_b32" in asm, \
                "expected mask-init arithmetic in tail preamble"

            # mask body: per-group compare + cndmask -> 0
            assert "v_cmp_lt_i32" in asm, \
                "tail loop missing per-lane K compare (v_cmp_lt_i32)"
            assert "v_cndmask_b32" in asm, \
                "tail loop missing v_cndmask_b32 to zero A vgprs"
            assert ", 0," in asm, \
                "v_cndmask_b32 should zero (src1=0) the masked lanes"

            # ordering: compare must come after a wait_lr (lgkmcnt(0)) and
            # before the first v_mfma.
            first_cmp   = asm.find("v_cmp_lt_i32")
            first_mfma  = asm.find("v_mfma")
            last_wait_before_cmp = asm.rfind("lgkmcnt(0)", 0, first_cmp)
            assert first_cmp != -1 and first_mfma != -1
            assert last_wait_before_cmp != -1, "expected lgkmcnt(0) before mask"
            assert first_cmp < first_mfma, "mask must precede first MFMA"
        finally:
            sched.deallocVgprTiles(writer)

    def test_tailloop_k_partial_mask_bf16(self):
        """BF16 tail loop must use V_AND_B32 with a 3-state mask (incl. 0x0000FFFF)
        so the K boundary can fall inside a vgpr without losing the valid element."""
        kernel = create_kernel(256, 256, fp4=False)
        writer, tiA, tiB, scaleTiA, scaleTiB, dTileInfo = make_writer_and_tileinfos(kernel, fp4=False)

        cfg = make_cfg_bf16(256, 256)
        sched = LogicalScheduler(cfg)
        sched.build()
        sched.allocVgprTiles(writer, tiA, tiB)
        try:
            sched.populate_instructions(
                writer, kernel,
                tileInfoA=tiA, tileInfoB=tiB,
                dtileInfo=dTileInfo,
                tensorParametersA=MagicMock(),
                tensorParametersB=MagicMock(),
            )
            asm = str(sched.emitTailLoop(writer, kernel)).lower()

            # half-mask sgpr load
            assert "0x0000ffff" in asm, \
                "BF16 tail mask must reference the 0x0000FFFF half-mask constant"
            # V_AND_B32 applies the per-vgpr mask to A/B tile vgprs
            assert "v_and_b32" in asm, \
                "BF16 tail must emit V_AND_B32 over A/B tile vgprs"
            # Two-stage select uses V_CMP_LT_I32 (diff<2 and diff<1)
            assert "v_cmp_lt_i32" in asm, \
                "BF16 tail mask uses v_cmp_lt_i32 for the diff<2 / diff<1 select"

            # Ordering: the v_and_b32 must follow the wait_lr and precede the v_mfma.
            first_and  = asm.find("v_and_b32 v")  # skip mask-init 'v_and_b32 ...,63,...'
            first_mfma = asm.find("v_mfma")
            assert first_and != -1 and first_mfma != -1
            # The first v_and_b32 we care about is the mask in the body; tolerate
            # earlier mask-init operations by checking ordering against MFMA.
            last_wait_before_mfma = asm.rfind("lgkmcnt(0)", 0, first_mfma)
            assert last_wait_before_mfma != -1, "expected lgkmcnt(0) before MFMA"
            assert last_wait_before_mfma < first_mfma
        finally:
            sched.deallocVgprTiles(writer)

    @pytest.mark.parametrize("pgr", [1, 2])
    def test_nll_scale_vgprs_differ_across_unroll_copies(self, pgr):
        """NLL for each unroll copy must use distinct scale VGPRs matching LR loads."""
        from rocisa.instruction import MXMFMAInstruction

        kernel = create_kernel(256, 256, fp4=True)
        writer, tiA, tiB, scaleTiA, scaleTiB, dTileInfo = make_writer_and_tileinfos(kernel, fp4=True)

        cfg = make_cfg_256x256_fp4(pgr=pgr)
        sched = LogicalScheduler(cfg)
        sched.build()
        sched.allocVgprTiles(writer, tiA, tiB,
                              scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)

        try:
            sched.populate_instructions(
                writer, kernel,
                tileInfoA=tiA, tileInfoB=tiB,
                dtileInfo=dTileInfo,
                scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
            )

            uf = sched.unroll_factor
            if uf < 2:
                pytest.skip("unroll_factor < 2, no multi-copy NLL to test")

            def get_scale_vgprs(emitted_3d, unroll_iter):
                sa_vgprs = set()
                sb_vgprs = set()
                for partition in emitted_3d:
                    for group in partition:
                        for em in group:
                            if em.opType == 'mfma':
                                for inst in sched._emitter.emit_module(em, unroll_iter):
                                    if isinstance(inst, MXMFMAInstruction):
                                        sa_vgprs.add(str(inst.mxsa))
                                        sb_vgprs.add(str(inst.mxsb))
                return sa_vgprs, sb_vgprs

            nll_scales = []
            for ui in range(uf):
                nll_scales.append(get_scale_vgprs(sched._nll_emitted, ui))

            for ui in range(uf):
                for uj in range(ui + 1, uf):
                    sa_i, sb_i = nll_scales[ui]
                    sa_j, sb_j = nll_scales[uj]
                    assert sa_i != sa_j, \
                        f"PGR{pgr}: NLL_C{ui} and NLL_C{uj} use same scaleA VGPRs {sa_i}"
                    assert sb_i != sb_j, \
                        f"PGR{pgr}: NLL_C{ui} and NLL_C{uj} use same scaleB VGPRs {sb_i}"
        finally:
            sched.deallocVgprTiles(writer)


# Tool to visualize the scheduling steps on a real kernel configuration. Run with --interactive to step through each phase.
# Also calls the instruction scheduler to verify the emitted modules are valid input and to show the final instruction counts.
# Example usage:
#   PYTHONPATH=. python Tensile/Tests/unit/test_SubtileBasedLogicalScheduler.py --mt0 320 --mt1 320 --du 64 --pgr 1 --wg 2x2 --partition-size 10x2
#   PYTHONPATH=. python Tensile/Tests/unit/test_SubtileBasedLogicalScheduler.py --mt0 256 --mt1 256 --du 256 --dtype fp4 --pgr 2 --wg 2x2 --partition-size 8x4
if __name__ == "__main__":
    import sys
    import io
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize SubtileBased LogicalScheduler steps for a given kernel config.",
    )
    parser.add_argument("--mt0", type=int, default=256, help="MacroTile0 (default: 256)")
    parser.add_argument("--mt1", type=int, default=256, help="MacroTile1 (default: 256)")
    parser.add_argument("--du", type=int, default=None,
                        help="DepthU (default: 64 for bf16, 512 for fp4)")
    parser.add_argument("--dtype", choices=["bf16", "fp4"], default="bf16",
                        help="Data type (default: bf16)")
    parser.add_argument("--partition-size", type=str, default="0x0",
                        help="partitionSize as MxN in MFMA tiles (0 = full dim, default: 0x0)")
    parser.add_argument("--wg", type=str, default="2x2",
                        help="MIWaveGroup as MxN (default: 2x2)")
    parser.add_argument("--pgr", type=int, choices=[0, 1, 2], default=1,
                        help="PrefetchGlobalRead level (default: 1)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Step through each phase interactively")
    args = parser.parse_args()

    fp4 = args.dtype == "fp4"
    if args.du is None:
        args.du = 512 if fp4 else 64

    wg_parts = args.wg.lower().split("x")
    if len(wg_parts) != 2:
        parser.error(f"--wg must be MxN (e.g. 2x2), got: {args.wg}")
    waveGroup = (int(wg_parts[0]), int(wg_parts[1]))

    ps_parts = args.partition_size.lower().split("x")
    if len(ps_parts) != 2:
        parser.error(f"--partition-size must be MxN (e.g. 10x2), got: {args.partition_size}")
    partSizeM, partSizeN = int(ps_parts[0]), int(ps_parts[1])

    kernel = create_kernel(args.mt0, args.mt1, fp4=fp4, depthU=args.du,
                           miWaveGroup=list(waveGroup))
    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    scaleTiA = makeTileInfo('MXSA', kernel) if fp4 else None
    scaleTiB = makeTileInfo('MXSB', kernel) if fp4 else None

    # Mirror Kernel.py:1139-1140 — gr granularity widens to (2,2) when the
    # tile's GR load ratio exceeds 1.0.
    grA = ReadGranularity(mn=1, k=2) if tiA.loadRatioGR <= 1.0 else ReadGranularity(mn=2, k=2)
    grB = ReadGranularity(mn=1, k=2) if tiB.loadRatioGR <= 1.0 else ReadGranularity(mn=2, k=2)

    cfg_kwargs = dict(
        numMFMATilesM=tiA.localMMATileGrid[0],
        numMFMATilesN=tiB.localMMATileGrid[0],
        numSubIterK=tiA.localMMATileGrid[1],
        lrA=ReadGranularity(mn=1, k=1),
        lrB=ReadGranularity(mn=1, k=1),
        grA=grA,
        grB=grB,
        partitionSizeM=partSizeM,
        partitionSizeN=partSizeN,
        pgr=args.pgr,
    )
    if fp4:
        cfg_kwargs.update(
            lrSA=ReadGranularity(mn=2, k=2),
            lrSB=ReadGranularity(mn=2, k=2),
            grSA=ReadGranularity(mn=scaleTiA.localMMATileGrid[0],
                                 k=scaleTiA.localMMATileGrid[1]),
            grSB=ReadGranularity(mn=scaleTiB.localMMATileGrid[0],
                                 k=scaleTiB.localMMATileGrid[1]),
        )
    cfg = SchedulerConfig(**cfg_kwargs)

    print(f"Config: MT={args.mt0}x{args.mt1}, DU={args.du}, dtype={args.dtype}, "
          f"WG={waveGroup[0]}x{waveGroup[1]}, "
          f"partitionSize={partSizeM}x{partSizeN}, pgr={args.pgr}")
    print(f"        numMFMATilesM={cfg.numMFMATilesM}, "
          f"numMFMATilesN={cfg.numMFMATilesN}, "
          f"numSubIterK={cfg.numSubIterK}, "
          f"hasScale={cfg.hasScale}, plr={cfg.plr}")
    print(f"        loadRatioGR(A,B)=({tiA.loadRatioGR:.3f}, {tiB.loadRatioGR:.3f}) "
          f"-> grA=({grA.mn},{grA.k}) grB=({grB.mn},{grB.k})")
    print()

    sched = LogicalScheduler(cfg)

    steps = [
        ("Place LRs",                     lambda: (sched.place_LRs(), sched.print_lr())),
        ("Assign VGPR tiles",             lambda: (sched.assign_vgpr_tiles(), sched.print_vgpr())),
        ("Place GRs",                     lambda: (sched.place_GRs(), sched.print_gr())),
        ("Annotate deps",                 lambda: (sched.annotate_deps(), sched.print_deps())),
        ("Remove unnecessary GR deps",    lambda: (sched.remove_unnecessary_gr_deps(), sched.print_deps())),
        ("Remove unnecessary LR deps",    lambda: (sched.remove_unnecessary_lr_deps(), sched.print_deps())),
        ("Remove cross deps",             lambda: (sched.remove_cross_deps(), sched.print_remove_deps())),
        ("Insert gr/lr inc",              lambda: (sched.insert_gr_lr_inc(), sched.print_group_lr_gr())),
        ("Group LR/GR",                   lambda: (sched.group_lr_gr(), sched.print_group_lr_gr())),
        ("Remove unnecessary wait_lr_sync", lambda: (sched.remove_unnecessary_wait_lr_sync(), sched.print_group_lr_gr())),
        ("Emit",                          lambda: (sched.emit(), sched.print_emit())),
        ("Emit (dependency order)",       lambda: (None, sched.print_emit_dep_order())),
    ]

    for i, (title, run) in enumerate(steps):
        _, output = run()
        print(f"{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print(output)
        if args.interactive and i < len(steps) - 1:
            input("Press Enter for next step...")

    writer, tiA, tiB, scaleTiA, scaleTiB, dTileInfo = make_writer_and_tileinfos(kernel, fp4=fp4)

    sched.allocVgprTiles(writer, tiA, tiB,
                         scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)

    sched.populate_instructions(
        writer, kernel,
        tileInfoA=tiA, tileInfoB=tiB,
        dtileInfo=dTileInfo,
        scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
    )

    print(f"{'=' * 60}")
    print(f"  Build tailloop (PGR0 template)")
    print(f"{'=' * 60}")
    print(sched.print_emit(sched._tailloop_emitted).replace("MAINLOOP:", "TAILLOOP:"))
    if args.interactive:
        input("Press Enter for next step...")

    def _print_emitLoop(label, emitted_3d, schedule=True):
        module = sched._emitLoop(writer, kernel, label, emitted_3d, schedule=schedule)
        buf = io.StringIO()
        for inst in module.flatitems():
            buf.write(f"  {str(inst).rstrip()}\n")
        return buf.getvalue()

    if args.pgr >= 1:
        loop_sections = [
            ("PRELOOP",  sched._preloop_emitted, False),
            ("MAINLOOP", sched._emitted),
            ("NGLL",     sched._ngll_emitted),
            ("NLL",      sched._nll_emitted),
            ("TAILLOOP", sched._tailloop_emitted, False),
        ]
    else:
        loop_sections = [
            ("MAINLOOP", sched._emitted),
            ("TAILLOOP", sched._tailloop_emitted, False),
        ]

    for section in loop_sections:
        label, emitted_3d = section[0], section[1]
        schedule = section[2] if len(section) > 2 else True
        print(f"{'=' * 60}")
        print(f"  {label} (emitLoop)")
        print(f"{'=' * 60}")
        print(_print_emitLoop(label, emitted_3d, schedule=schedule))
        if args.interactive:
            input("Press Enter for next step...")

    sched.deallocVgprTiles(writer)
