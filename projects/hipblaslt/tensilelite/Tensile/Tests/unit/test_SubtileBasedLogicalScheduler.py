"""Tests for LogicalScheduler — validates each scheduling pass.

Organized by pass:
  1. PlaceLRs        — MFMA + Local Read placement
  2. AssignVgprTiles — VGPR tile ID allocation
  3. PlaceGRs        — Global Read placement
  4. AnnotateDeps    — Raw dependency annotation
  5. RemoveUnnecessaryGrDeps / RemoveUnnecessaryLrDeps — Dep pruning
  6. RemoveCrossDeps — Cross-subIterK dep → preOp conversion
  7. InsertGrLrInc   — MT iteration increment insertion
  8. GroupLrGr       — Chain grouping with merged preOps
  9. RemoveUnnecessaryWaitLrSync — Sync downgrade
  10. Emit           — EmittedModule chain generation
  11. BuildPreloop / BuildNGLL / BuildNLL — Loop variant construction
  12. Integration    — Full pipeline with real instructions
"""
import pytest
from Tensile.Components.Subtile.Kernel import (
    TileInfo, AB_B8, AB_B16, AB_B16_W32, AB_B4, MXSA_B4, MXSB_B4, CD_F32, CD_F32_W32,
)
from Tensile.Components.Subtile.LogicalScheduler import (
    GRPlacementStrategy,
    LogicalScheduler,
    MFMATileRange,
    ReadGranularity,
    SchedulerConfig,
    EmittedModule,
    LRPlacement,
    GRPlacement,
    Dep,
    WaitGRCounts,
    fmt_mt,
)
from unittest.mock import MagicMock
from rocisa.code import Module


def makeTileInfo(tc, kernel):
    """Compatibility wrapper: select geometry from kernel config and return TileInfo."""
    fp4 = kernel["ProblemType"].get("MXBlockA", 0) > 0
    wave32 = kernel.get("WavefrontSize", 64) == 32
    ab_bf16 = AB_B16_W32 if wave32 else AB_B16
    cd_f32 = CD_F32_W32 if wave32 else CD_F32
    _geo = {
        'A': AB_B4 if fp4 else ab_bf16,
        'B': AB_B4 if fp4 else ab_bf16,
        'MXSA': MXSA_B4,
        'MXSB': MXSB_B4,
        'D': cd_f32,
    }
    # TDM path queries writer.states.subtileLdsSwizzle in TileInfo.__init__.
    # Provide a minimal stub so makeTileInfo can be called without a full writer.
    writer_stub = None
    if kernel.get("enableTDM%s" % tc, False):
        from types import SimpleNamespace
        writer_stub = SimpleNamespace(states=SimpleNamespace(subtileLdsSwizzle=False))
    return TileInfo(_geo[tc], tc, writer_stub, kernel)


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
                  miWaveGroup=None, sourceSwap=False, arch="gfx950"):
    mxblock = 32 if fp4 else 0
    bpe = 0.5 if fp4 else 2
    matrixInstK = 128 if fp4 else 32
    if depthU is None:
        depthU = 256 if fp4 else 64
    if miWaveGroup is None:
        miWaveGroup = [2, 2]
    is_gfx1250 = (arch == "gfx1250")
    waveSize = 32 if is_gfx1250 else 64
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
        "WavefrontSize": waveSize,
        "SourceSwap": sourceSwap,
        "MIArchVgpr": is_gfx1250,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "NonTemporalMXSA": 0,
        "NonTemporalMXSB": 0,
        "NoTailLoop": False,
        "ProblemType": problemType,
    }
    if is_gfx1250:
        kernel["ISA"] = (12, 5, 0)
        kernel["UseSubtileImpl"] = True
        kernel["enableTDMA"] = True
        kernel["enableTDMB"] = True
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


def make_cfg_bf16_pgr0(MT0=256, MT1=256, depthU=64):
    """Build BF16 config with pgr=0."""
    kernel = create_kernel(MT0, MT1, fp4=False, depthU=depthU)
    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    return SchedulerConfig(
        numMFMATilesM=tiA.localMMATileGrid[0],
        numMFMATilesN=tiB.localMMATileGrid[0],
        numSubIterK=tiA.localMMATileGrid[1],
        lrA=ReadGranularity(mn=1, k=1),
        lrB=ReadGranularity(mn=1, k=1),
        grA=ReadGranularity(mn=1, k=2),
        grB=ReadGranularity(mn=1, k=2),
        pgr=0,
    )


def make_cfg_bf16_pgr1(MT0=256, MT1=256, depthU=128, partSizeM=0, partSizeN=0):
    """Build BF16 config with pgr=1."""
    kernel = create_kernel(MT0, MT1, fp4=False, depthU=depthU)
    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    return SchedulerConfig(
        numMFMATilesM=tiA.localMMATileGrid[0],
        numMFMATilesN=tiB.localMMATileGrid[0],
        numSubIterK=tiA.localMMATileGrid[1],
        lrA=ReadGranularity(mn=1, k=1),
        lrB=ReadGranularity(mn=1, k=1),
        grA=ReadGranularity(mn=1, k=2),
        grB=ReadGranularity(mn=1, k=2),
        partitionSizeM=partSizeM,
        partitionSizeN=partSizeN,
        pgr=1,
    )


def create_kernel_fp8(MT0, MT1, waveGroup, depthU=128):
    """Create a plain FP8 kernel config (bpe=1, matrixInstK=128, no MX scale)."""
    dtype = _mock_dtype(1)
    return {
        "DepthU": depthU,
        "_DepthUA": depthU,
        "_DepthUB": depthU,
        "MacroTileA": MT0,
        "MacroTileB": MT1,
        "MacroTile0": MT0,
        "MacroTile1": MT1,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": 128,
        "MIWaveGroup": list(waveGroup),
        "WavefrontSize": 64,
        "SourceSwap": False,
        "MIArchVgpr": False,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "NonTemporalMXSA": 0,
        "NonTemporalMXSB": 0,
        "ProblemType": {
            "DataTypeA": dtype,
            "DataTypeB": dtype,
            "ComputeDataType": _mock_dtype(4),
        },
    }


def make_cfg_fp8(MT0, MT1, waveGroup, depthU=128, pgr=2):
    """Build plain FP8 config (AB_B8, matrixInstK=128, no scale tensors).

    GR granularity matches production: mn=subtileShape[0]=1, k=subtileShape[1]=1.
    With DU=128 and matrixInstK=128: numSubIterK=1, flat_len=numPartitions.
    """
    kernel = create_kernel_fp8(MT0, MT1, waveGroup, depthU)
    tiA = TileInfo(AB_B8, 'A', None, kernel)
    tiB = TileInfo(AB_B8, 'B', None, kernel)
    return SchedulerConfig(
        numMFMATilesM=tiA.localMMATileGrid[0],
        numMFMATilesN=tiB.localMMATileGrid[0],
        numSubIterK=tiA.localMMATileGrid[1],
        lrA=ReadGranularity(mn=1, k=1),
        lrB=ReadGranularity(mn=1, k=1),
        grA=ReadGranularity(mn=tiA.subtileShape[0], k=tiA.subtileShape[1]),
        grB=ReadGranularity(mn=tiB.subtileShape[0], k=tiB.subtileShape[1]),
        pgr=pgr,
    )


def make_example_granularities_1():
    """Example Granularities 1 from the design doc: LR A,B=1x1, LR SA,SB=2x2."""
    return SchedulerConfig(
        numMFMATilesM=2,
        numMFMATilesN=2,
        numSubIterK=2,
        lrA=ReadGranularity(mn=1, k=1),
        lrB=ReadGranularity(mn=1, k=1),
        lrSA=ReadGranularity(mn=2, k=2),
        lrSB=ReadGranularity(mn=2, k=2),
        grA=ReadGranularity(mn=1, k=2),
        grB=ReadGranularity(mn=1, k=2),
        grSA=ReadGranularity(mn=2, k=2),
        grSB=ReadGranularity(mn=2, k=2),
    )


def make_writer_and_tileinfos(kernel, fp4=False):
    """Create writer with register pools and TileInfos for integration tests."""
    from types import SimpleNamespace
    from rocisa import rocIsa
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType
    from Tensile.Common.RegisterPool import allocTmpGpr

    ri = rocIsa.getInstance()
    import shutil
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    # Pick ISA + wavesize from the kernel — rocisa is a process-wide singleton,
    # so re-init each call to override any prior state from other tests.
    isa = tuple(kernel.get("ISA", (9, 5, 0)))
    waveSize = kernel.get("WavefrontSize", 64)
    ri.init(isa, asmpath)
    ri.setKernel(isa, waveSize)
    is_gfx1250 = (isa == (12, 5, 0))

    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    scaleTiA = makeTileInfo('MXSA', kernel) if fp4 else None
    scaleTiB = makeTileInfo('MXSB', kernel) if fp4 else None

    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr, False)
    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr, False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr, False)
    writer.sgprs = {}
    writer.states = SimpleNamespace(
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
        archCaps={"LDSBankCount": 64, "LDSBankWidth": 4,
                  "HasWmmaArbStallBit": is_gfx1250},
        asmCaps={"HasMFMA": not is_gfx1250,
                 "HasWMMA_AccImmZero": is_gfx1250},
        unrollIdx=0,
        laneSGPRCount=1 if is_gfx1250 else 2,
        subtileLdsSwizzle=not is_gfx1250,
    )
    if is_gfx1250:
        writer.sgprPool.checkOut(12)
        writer.sgprs["StrideA0I"] = 10
        writer.sgprs["StrideB1J"] = 11
        for tc in ['A', 'B']:
            writer.sgprs["tdm%sGroup0" % tc] = writer.sgprPool.checkOutAligned(4, 4, preventOverflow=False)
            writer.sgprs["tdm%sGroup1" % tc] = writer.sgprPool.checkOutAligned(8, 4, preventOverflow=False)
            writer.sgprs["tdmLdsAddr%s" % tc] = writer.sgprPool.checkOut(1, preventOverflow=False)
            writer.sgprs["tdmLdsSwapMask%s" % tc] = writer.sgprPool.checkOut(1, preventOverflow=False)
            writer.sgprs["Address%s" % tc] = writer.sgprPool.checkOutAligned(2, 2, preventOverflow=False)
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


# ── Assertion helpers ─────────────────────────────────────────

def _get_lr(slot, tensor):
    """Get the LR placement for a given tensor in a slot."""
    matches = [lr for lr in slot.lrs if lr.tensor == tensor]
    assert len(matches) == 1, f"Expected 1 LR for {tensor}, got {len(matches)}"
    return matches[0]


def _assert_slot_lrs(slot, expected_tensors):
    """Assert the tensor list of LRs in a slot."""
    assert [lr.tensor for lr in slot.lrs] == expected_tensors


def _assert_slot_grs(slot, expected_tensors, msg=""):
    """Assert the tensor list of GRs in a slot."""
    assert [gr.tensor for gr in slot.grs] == expected_tensors, msg


def _assert_lr(slot, tensor, mt, k_start, k_end, tile_start, tile_end):
    """Assert an LR placement matches expected values."""
    lr = _get_lr(slot, tensor)
    assert lr.mtIteration == mt, \
        f"LR {tensor}: expected mt={mt}, got {lr.mtIteration}"
    assert lr.tiles.subIterK_start == k_start
    assert lr.tiles.subIterK_end == k_end
    assert lr.tiles.tileId_start == tile_start
    assert lr.tiles.tileId_end == tile_end


def _assert_gr(slot, tensor, k_start, k_end, tile_start, tile_end, mt=2, idx=0):
    """Assert a GR placement matches expected values."""
    grs = [gr for gr in slot.grs if gr.tensor == tensor]
    assert len(grs) > idx, \
        f"Expected at least {idx+1} GR(s) for {tensor} in slot {slot.subIterK}, got {len(grs)}"
    gr = grs[idx]
    assert gr.mtIteration == mt, \
        f"GR {tensor}[{idx}] in slot {slot.subIterK}: expected mt={mt}, got {gr.mtIteration}"
    assert gr.tiles.subIterK_start == k_start
    assert gr.tiles.subIterK_end == k_end
    assert gr.tiles.tileId_start == tile_start
    assert gr.tiles.tileId_end == tile_end


def _dep_refs(placement):
    """Return list of (type, tensor, partition, subIterK_slot, mt_offset) for deps."""
    result = []
    for dep in placement.deps:
        p = dep.ref
        kind = 'LR' if isinstance(p, LRPlacement) else 'GR'
        result.append((kind, p.tensor, p.partition, p.subIterK_slot, dep.mt_offset))
    return result


def _preop_kinds(placement):
    """Return list of (kind, has_sync, wait_gr_counts_dict_or_None) for preOps."""
    result = []
    for op in placement.preOps:
        has_sync = getattr(op, 'has_sync', False)
        counts = getattr(op, 'wait_gr_counts', None)
        if counts:
            result.append((op.kind, has_sync, {'A': counts.A, 'B': counts.B,
                                                'SA': counts.SA, 'SB': counts.SB}))
        else:
            result.append((op.kind, has_sync, None))
    return result


def _preop_inc_tensors(placement, kind):
    """Return list of tensor names for preOps of given kind."""
    return [op.tensor for op in placement.preOps if op.kind == kind]


# ══════════════════════════════════════════════════════════════
# Step 1: Place LRs
# ══════════════════════════════════════════════════════════════

class TestPlaceLRs:

    def test_1x1_k1_DU256(self):
        """MT=256x256, DU=256, FP4, LR k=1.
        numMFMATilesM=8, numMFMATilesN=8, numSubIterK=2.
        """
        cfg = make_cfg_256x256_fp4()
        assert cfg.numMFMATilesM == 8
        assert cfg.numMFMATilesN == 8
        assert cfg.numSubIterK == 2
        assert cfg.hasScale

        sched = LogicalScheduler(cfg)
        partitions = sched.place_LRs()
        slots = partitions[0]

        assert len(slots) == 2

        # subIterK=0: LR A, B (k=1, non-wrapping)
        s0 = slots[0]
        assert s0.mfma.subIterK == 0
        assert s0.mfma.tileA.tileId_start == 0
        assert s0.mfma.tileA.tileId_end == 8
        assert s0.mfma.tileB.tileId_end == 8
        _assert_slot_lrs(s0, ['A', 'B'])
        _assert_lr(s0, 'A', 0,1, 2, 0, 8)
        _assert_lr(s0, 'B', 0,1, 2, 0, 8)

        # subIterK=1: LR A, B (wrapping), SA, SB (wrapping)
        s1 = slots[1]
        assert s1.mfma.subIterK == 1
        _assert_slot_lrs(s1, ['A', 'B', 'SA', 'SB'])
        _assert_lr(s1, 'A', 1,0, 1, 0, 8)
        _assert_lr(s1, 'SA', 1,0, 2, 0, 8)
        _assert_lr(s1, 'SB', 1,0, 2, 0, 8)

    def test_1x1_k2_DU256(self):
        """MT=256x256, DU=256, FP4, LR A/B with k=2.
        With k=2 == numSubIterK, LR A/B load all subIterKs at once.
        Split: LR A at subIterK=0, LR B at subIterK=1.
        """
        cfg = make_cfg_256x256_fp4(k_gran=2)
        assert cfg.numMFMATilesM == 8
        assert cfg.numMFMATilesN == 8
        assert cfg.numSubIterK == 2

        sched = LogicalScheduler(cfg)
        partitions = sched.place_LRs()
        slots = partitions[0]
        assert len(slots) == 2

        # subIterK=0: LR A + LR SA (side grouping, all MT n+1)
        _assert_slot_lrs(slots[0], ['A', 'SA'])
        _assert_lr(slots[0], 'A', 1,0, 2, 0, 8)
        _assert_lr(slots[0], 'SA', 1,0, 2, 0, 8)

        # subIterK=1: LR B + LR SB
        _assert_slot_lrs(slots[1], ['B', 'SB'])
        _assert_lr(slots[1], 'B', 1,0, 2, 0, 8)
        _assert_lr(slots[1], 'SB', 1,0, 2, 0, 8)

    def test_1x1_k1_DU512(self):
        """MT=256x256, DU=512, FP4, LR k=1.
        DU=512 → numSubIterK=4. LR SA,SB: k=2 → 2 chunks.
        """
        cfg = make_cfg_256x256_fp4(depthU=512)
        assert cfg.numMFMATilesM == 8
        assert cfg.numMFMATilesN == 8
        assert cfg.numSubIterK == 4
        assert cfg.hasScale

        sched = LogicalScheduler(cfg)
        partitions = sched.place_LRs()
        slots = partitions[0]
        assert len(slots) == 4

        # subIterK=0: A, B (k=1) + SA (k=2, A-side)
        _assert_slot_lrs(slots[0], ['A', 'B', 'SA'])
        assert slots[0].mfma.subIterK == 0
        _assert_lr(slots[0], 'A', 0,1, 2, 0, 8)
        _assert_lr(slots[0], 'SA', 0,2, 4, 0, 8)

        # subIterK=1: A, B + SB (k=2, B-side)
        _assert_slot_lrs(slots[1], ['A', 'B', 'SB'])
        _assert_lr(slots[1], 'A', 0,2, 3, 0, 8)
        _assert_lr(slots[1], 'SB', 0,2, 4, 0, 8)

        # subIterK=2: A, B only
        _assert_slot_lrs(slots[2], ['A', 'B'])
        _assert_lr(slots[2], 'A', 0,3, 4, 0, 8)

        # subIterK=3: all wrapping → MT n+1
        _assert_slot_lrs(slots[3], ['A', 'B', 'SA', 'SB'])
        _assert_lr(slots[3], 'A', 1,0, 1, 0, 8)
        _assert_lr(slots[3], 'SA', 1,0, 2, 0, 8)
        _assert_lr(slots[3], 'SB', 1,0, 2, 0, 8)

    def test_1x1_k2_DU512(self):
        """MT=256x256, DU=512, FP4, LR A/B with k=2.
        2 chunks of 2 subIterKs. A/B split by side within each chunk.
        """
        cfg = make_cfg_256x256_fp4(depthU=512, k_gran=2)
        assert cfg.numSubIterK == 4

        sched = LogicalScheduler(cfg)
        partitions = sched.place_LRs()
        slots = partitions[0]
        assert len(slots) == 4

        # Chunk 0: A+SA at s0, B+SB at s1 (MT n, loading [2,3])
        _assert_slot_lrs(slots[0], ['A', 'SA'])
        _assert_lr(slots[0], 'A', 0,2, 4, 0, 8)
        _assert_lr(slots[0], 'SA', 0,2, 4, 0, 8)
        _assert_slot_lrs(slots[1], ['B', 'SB'])
        _assert_lr(slots[1], 'B', 0,2, 4, 0, 8)
        _assert_lr(slots[1], 'SB', 0,2, 4, 0, 8)

        # Chunk 1: A+SA at s2, B+SB at s3 (MT n+1, loading [0,1])
        _assert_slot_lrs(slots[2], ['A', 'SA'])
        _assert_lr(slots[2], 'A', 1,0, 2, 0, 8)
        _assert_slot_lrs(slots[3], ['B', 'SB'])
        _assert_lr(slots[3], 'B', 1,0, 2, 0, 8)

    def test_2x2_k1_DU256(self):
        """MT=256x256, DU=256, FP4, k=1, 2x2 partition.
        8x8 tiles → 4 partitions of 4x4. Column-major traversal.
        """
        cfg = make_cfg_256x256_fp4(partSizeM=4, partSizeN=4)
        assert cfg.numPartitions == 4
        assert cfg.partitionSizeM == 4
        assert cfg.partitionSizeN == 4

        sched = LogicalScheduler(cfg)
        partitions = sched.place_LRs()
        assert len(partitions) == 4

        # P0: A[0-3],B[0-3]
        p0 = partitions[0]
        assert len(p0) == 2
        assert p0[0].mfma.tileA.tileId_start == 0
        assert p0[0].mfma.tileA.tileId_end == 4
        assert p0[0].mfma.tileB.tileId_start == 0
        assert p0[0].mfma.tileB.tileId_end == 4

        # s0: A, B K-prefetch + SA wrapping for P1 tiles [4-7]
        _assert_slot_lrs(p0[0], ['A', 'B', 'SA'])
        _assert_lr(p0[0], 'A', 0,1, 2, 0, 4)
        _assert_lr(p0[0], 'B', 0,1, 2, 0, 4)
        lr_sa0 = _get_lr(p0[0], 'SA')
        assert lr_sa0.tiles.tileId_start == 4
        assert lr_sa0.tiles.tileId_end == 8
        assert lr_sa0.mtIteration == 0

        # s1: A wrapping only (B unchanged for P1)
        _assert_slot_lrs(p0[1], ['A'])
        lr_a1 = _get_lr(p0[1], 'A')
        assert lr_a1.tiles.tileId_start == 4
        assert lr_a1.tiles.tileId_end == 8
        assert lr_a1.mtIteration == 0

        # P1: A[4-7],B[0-3] → B wrapping for P2
        p1 = partitions[1]
        _assert_slot_lrs(p1[0], ['A', 'SB'])
        _assert_slot_lrs(p1[1], ['B'])
        lr_b1 = _get_lr(p1[1], 'B')
        assert lr_b1.tiles.tileId_start == 4
        assert lr_b1.tiles.tileId_end == 8

        # P2: A[0-3],B[4-7] → no wrapping (both already loaded)
        p2 = partitions[2]
        _assert_slot_lrs(p2[0], ['B'])
        assert len(p2[1].lrs) == 0

        # P3: last → load all for MT n+1
        p3 = partitions[3]
        _assert_slot_lrs(p3[0], ['SA'])
        lr_sa3 = _get_lr(p3[0], 'SA')
        assert lr_sa3.tiles.tileId_start == 0
        assert lr_sa3.tiles.tileId_end == 4
        assert lr_sa3.mtIteration == 1
        _assert_slot_lrs(p3[1], ['A', 'B', 'SB'])
        _assert_lr(p3[1], 'A', 1,0, 1, 0, 4)

    def test_2x2_k1_DU512(self):
        """MT=256x256, DU=512, FP4, k=1, 2x2 partition.
        numSubIterK=4. 4 partitions × 4 subIterKs = 16 slots.
        """
        cfg = make_cfg_256x256_fp4(depthU=512, partSizeM=4, partSizeN=4)
        assert cfg.numPartitions == 4
        assert cfg.numSubIterK == 4

        sched = LogicalScheduler(cfg)
        partitions = sched.place_LRs()
        assert len(partitions) == 4

        # P0: 4 subIterKs with K-prefetch and wrapping
        p0 = partitions[0]
        assert len(p0) == 4
        for s in p0:
            assert s.mfma.tileA.tileId_start == 0
            assert s.mfma.tileA.tileId_end == 4

        _assert_slot_lrs(p0[0], ['A', 'B', 'SA'])
        _assert_slot_lrs(p0[1], ['A', 'B', 'SB'])
        _assert_slot_lrs(p0[2], ['A', 'B', 'SA'])
        _assert_slot_lrs(p0[3], ['A'])

        # P3: last partition — last 2 slots load for MT n+1
        p3 = partitions[3]
        assert len(p3[0].lrs) == 0
        assert len(p3[1].lrs) == 0
        _assert_slot_lrs(p3[2], ['SA'])
        _assert_slot_lrs(p3[3], ['A', 'B', 'SB'])
        _assert_lr(p3[3], 'A', 1,0, 1, 0, 4)
        _assert_lr(p3[3], 'B', 1,0, 1, 0, 4)

    def test_2x2_k2_DU256(self):
        """MT=256x256, DU=256, FP4, k=2, 2x2 partition.
        Each partition: A/B loaded per side, wrapping across partitions.
        """
        cfg = make_cfg_256x256_fp4(k_gran=2, partSizeM=4, partSizeN=4)
        assert cfg.numPartitions == 4

        sched = LogicalScheduler(cfg)
        partitions = sched.place_LRs()
        assert len(partitions) == 4

        # P0: A+SA at s0 (A changes for P1)
        _assert_slot_lrs(partitions[0][0], ['A', 'SA'])
        lr_a = _get_lr(partitions[0][0], 'A')
        assert lr_a.tiles.tileId_start == 4
        assert lr_a.tiles.tileId_end == 8
        assert lr_a.mtIteration == 0
        assert len(partitions[0][1].lrs) == 0

        # P2: no LRs (both A and B already loaded)
        assert len(partitions[2][0].lrs) == 0
        assert len(partitions[2][1].lrs) == 0

        # P3: last → A+SA, B+SB for MT n+1
        _assert_slot_lrs(partitions[3][0], ['A', 'SA'])
        _assert_slot_lrs(partitions[3][1], ['B', 'SB'])
        _assert_lr(partitions[3][0], 'A', 1,0, 2, 0, 4)
        _assert_lr(partitions[3][1], 'B', 1,0, 2, 0, 4)

    def test_10x1_k1_bf16(self):
        """MT=320x320, BF16, DU=64, k=1, 10x1 partition. No scale.
        B never changes → only A wrapping LRs needed.
        """
        cfg = make_cfg_bf16(320, 320, partSizeM=1, partSizeN=10)
        assert cfg.numMFMATilesM == 10
        assert cfg.numMFMATilesN == 10
        assert cfg.numSubIterK == 2
        assert not cfg.hasScale
        assert cfg.numPartitions == 10

        sched = LogicalScheduler(cfg)
        partitions = sched.place_LRs()
        assert len(partitions) == 10

        # P0: A+B K-prefetch, then A wrapping
        _assert_slot_lrs(partitions[0][0], ['A', 'B'])
        _assert_slot_lrs(partitions[0][1], ['A'])
        lr_a0 = _get_lr(partitions[0][0], 'A')
        assert lr_a0.tiles.subIterK_start == 1
        assert lr_a0.mtIteration == 0

        # P1-P8: only A LRs (B deduped)
        for pi in range(1, 9):
            _assert_slot_lrs(partitions[pi][0], ['A'])
            _assert_slot_lrs(partitions[pi][1], ['A'])

        # P9: last → A+B for MT n+1
        _assert_slot_lrs(partitions[9][0], ['A'])
        _assert_slot_lrs(partitions[9][1], ['A', 'B'])
        _assert_lr(partitions[9][1], 'A', 1,0, 1, 0, 1)
        _assert_lr(partitions[9][1], 'B', 1,0, 1, 0, 10)

    def test_asymmetric_mt(self):
        """MT0=384, MT1=128 — asymmetric tile counts."""
        cfg = make_cfg_bf16(384, 128)
        assert cfg.numMFMATilesM == 12  # 384/16/2
        assert cfg.numMFMATilesN == 4   # 128/16/2
        assert cfg.numSubIterK == 2

        sched = LogicalScheduler(cfg)
        partitions = sched.place_LRs()
        slots = partitions[0]
        assert len(slots) == 2
        assert slots[0].mfma.tileA.tileId_end == 12
        assert slots[0].mfma.tileB.tileId_end == 4


# ══════════════════════════════════════════════════════════════
# Step 2: Assign VGPR tiles
# ══════════════════════════════════════════════════════════════

class TestAssignVgprTiles:

    def _assert_no_conflict_and_unrolling(self, sched):
        """Validate vgprTileId invariants: no overlap, bounds, alternation, continuity."""
        parts = sched._partitions
        num_iters = sched.unroll_factor

        # 1. No MFMA/LR vgprTileId overlap
        for pi, slots in enumerate(parts):
            for slot in slots:
                if not slot.mfma:
                    continue
                for lr in slot.lrs:
                    if lr.tensor not in ('A', 'B') or not lr.vgpr_tile_map:
                        continue
                    mfma_map_list = slot.mfma.vgpr_tile_maps.get(lr.tensor, [])
                    for ui in range(len(mfma_map_list)):
                        mfma_vids = set(mfma_map_list[ui].values())
                        lr_vids = set(lr.vgpr_tile_map[ui].values())
                        assert mfma_vids.isdisjoint(lr_vids), \
                            f"P{pi} k={slot.subIterK} iter={ui}: MFMA/LR {lr.tensor} overlap"

        # 2. All vgprTileIds within [0, tile_peaks)
        for pi, slots in enumerate(parts):
            for slot in slots:
                if slot.mfma:
                    for tensor, maps in slot.mfma.vgpr_tile_maps.items():
                        peak = sched.tile_peaks[tensor]
                        for ui, tile_map in enumerate(maps):
                            for tid, vid in tile_map.items():
                                assert 0 <= vid < peak, \
                                    f"P{pi} k={slot.subIterK} iter={ui}: " \
                                    f"MFMA {tensor} vid={vid} out of [0, {peak})"
                for lr in slot.lrs:
                    if not lr.vgpr_tile_map:
                        continue
                    peak = sched.tile_peaks[lr.tensor]
                    for ui, tile_map in enumerate(lr.vgpr_tile_map):
                        for tid, vid in tile_map.items():
                            assert 0 <= vid < peak, \
                                f"P{pi} k={slot.subIterK} iter={ui}: " \
                                f"LR {lr.tensor} vid={vid} out of [0, {peak})"

        # 3. Double-buffer alternation: MFMAs in different K-groups use different sets.
        cfg = sched.config
        lr_grans = {'A': cfg.lrA, 'B': cfg.lrB}
        if cfg.hasScale:
            lr_grans['SA'] = cfg.lrSA
            lr_grans['SB'] = cfg.lrSB

        for ui in range(num_iters):
            for pi, slots in enumerate(parts):
                by_k = {}
                for slot in slots:
                    if not slot.mfma:
                        continue
                    by_k.setdefault(slot.subIterK, []).append(slot)
                sorted_ks = sorted(by_k.keys())
                for i in range(len(sorted_ks) - 1):
                    k0, k1 = sorted_ks[i], sorted_ks[i + 1]
                    for s0 in by_k[k0]:
                        for s1 in by_k[k1]:
                            for tensor in sched.tile_peaks:
                                gran = lr_grans[tensor]
                                if k0 // gran.k == k1 // gran.k:
                                    continue
                                maps0 = s0.mfma.vgpr_tile_maps.get(tensor, [])
                                maps1 = s1.mfma.vgpr_tile_maps.get(tensor, [])
                                if ui >= len(maps0) or ui >= len(maps1):
                                    continue
                                common_tiles = (set(maps0[ui].keys())
                                                & set(maps1[ui].keys()))
                                for tid in common_tiles:
                                    assert maps0[ui][tid] != maps1[ui][tid], \
                                        f"P{pi} iter={ui}: {tensor} tile {tid} " \
                                        f"same vid at k={k0} and k={k1}"

        # 4. Unrolling continuity
        if sched.needs_unrolling:
            last_ui = num_iters - 1
            wrapping_writes = {}
            for pi, slots in enumerate(parts):
                for slot in slots:
                    for lr in slot.lrs:
                        if lr.mtIteration == 0 or lr.tensor not in ('A', 'B'):
                            continue
                        if not lr.vgpr_tile_map:
                            continue
                        tile_map = lr.vgpr_tile_map[last_ui]
                        for tileId, vid in tile_map.items():
                            for lk in lr.tiles.subIterK_list:
                                wrapping_writes[(lr.tensor, tileId, lk, pi)] = vid

            for pi, slots in enumerate(parts):
                for slot in slots:
                    if not slot.mfma:
                        continue
                    for tensor, tileRange in [('A', slot.mfma.tileA), ('B', slot.mfma.tileB)]:
                        mfma_map_0 = slot.mfma.vgpr_tile_maps[tensor][0]
                        for tileId in tileRange.tileId_list:
                            key = (tensor, tileId, slot.subIterK, pi)
                            if key in wrapping_writes:
                                assert mfma_map_0[tileId] == wrapping_writes[key], \
                                    f"P{pi} k={slot.subIterK}: unrolling continuity broken"

    def test_basic_with_scale(self):
        """VgprTile allocation with scale tensors (design doc example)."""
        cfg = make_example_granularities_1()
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        parts = sched._partitions
        s0, s1 = parts[0][0], parts[0][1]

        for tensor in ('A', 'B', 'SA', 'SB'):
            assert len(s0.mfma.vgpr_tile_maps[tensor]) > 0

        for tensor in ('A', 'B'):
            map_k0 = s0.mfma.vgpr_tile_maps[tensor][0]
            map_k1 = s1.mfma.vgpr_tile_maps[tensor][0]
            for tileId in map_k0:
                if tileId in map_k1:
                    assert map_k0[tileId] != map_k1[tileId]

        assert sched.tile_peaks['A'] > 0
        assert sched.tile_peaks['SA'] > 0
        assert sched.needs_unrolling
        self._assert_no_conflict_and_unrolling(sched)

    def test_no_scale_k1(self):
        """No scales, A/B k_gran=1 → tiles alternate every subIterK."""
        cfg = SchedulerConfig(
            numMFMATilesM=2, numMFMATilesN=2, numSubIterK=2,
            lrA=ReadGranularity(mn=1, k=1),
            lrB=ReadGranularity(mn=1, k=1),
            grA=ReadGranularity(mn=1, k=2),
            grB=ReadGranularity(mn=1, k=2),
        )
        assert not cfg.hasScale

        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        assert 'SA' not in sched.tile_peaks
        assert not sched.needs_unrolling
        self._assert_no_conflict_and_unrolling(sched)

    def test_DU512(self):
        """DU=512, FP4. numSubIterK=4, no unrolling needed."""
        cfg = make_cfg_256x256_fp4(depthU=512)
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        for slot in sched._partitions[0]:
            assert len(slot.mfma.vgpr_tile_maps['A']) > 0
            assert len(slot.mfma.vgpr_tile_maps['SA']) > 0

        assert not sched.needs_unrolling
        self._assert_no_conflict_and_unrolling(sched)

    def test_DU512_partition_2x2(self):
        """DU=512 + 2x2 partition. All partitions have tile maps."""
        cfg = make_cfg_256x256_fp4(depthU=512, partSizeM=4, partSizeN=4)
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        for pi in range(4):
            for slot in sched._partitions[pi]:
                assert len(slot.mfma.vgpr_tile_maps['A']) > 0

        assert not sched.needs_unrolling
        self._assert_no_conflict_and_unrolling(sched)

    # ── Group 1: BF16 2x2 ──

    @pytest.mark.parametrize("depthU", [64, 128])
    def test_bf16_2x2(self, depthU):
        """BF16 256x256 with default miWaveGroup=[2,2]."""
        cfg = make_cfg_bf16(depthU=depthU)
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        assert sched.tile_peaks == {'A': 16, 'B': 16}
        assert not sched.needs_unrolling
        self._assert_no_conflict_and_unrolling(sched)

    # ── Group 2: BF16 1x4 / 4x1 ──

    @pytest.mark.parametrize("depthU", [64, 128])
    def test_bf16_1x4(self, depthU):
        """BF16 256x256 miWaveGroup=[1,4]: 16 M-tiles, 4 N-tiles."""
        cfg = make_cfg_bf16(depthU=depthU, miWaveGroup=[1, 4])
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        assert sched.tile_peaks == {'A': 32, 'B': 8}
        assert not sched.needs_unrolling
        self._assert_no_conflict_and_unrolling(sched)

    @pytest.mark.parametrize("depthU", [64, 128])
    def test_bf16_4x1(self, depthU):
        """BF16 256x256 miWaveGroup=[4,1]: 4 M-tiles, 16 N-tiles."""
        cfg = make_cfg_bf16(depthU=depthU, miWaveGroup=[4, 1])
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        assert sched.tile_peaks == {'A': 8, 'B': 32}
        assert not sched.needs_unrolling
        self._assert_no_conflict_and_unrolling(sched)

    # ── Group 3: BF16 multi-partition along N ──

    @pytest.mark.parametrize("partSizeN,expected_peak_B", [
        (4, 8),
        (3, 6),
        (2, 4),
    ])
    def test_bf16_partition_N(self, partSizeN, expected_peak_B):
        """BF16 256x256 (tilesN=8) with partitions along N."""
        cfg = make_cfg_bf16(depthU=64, partSizeN=partSizeN)
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        assert sched.tile_peaks['A'] == 16
        assert sched.tile_peaks['B'] == expected_peak_B
        assert not sched.needs_unrolling
        self._assert_no_conflict_and_unrolling(sched)

    # ── Group 3b: BF16 multi-partition on asymmetric macro tiles ──

    @pytest.mark.parametrize("partSizeN,expected_peak_B,expected_partN", [
        (6, 12, [6, 6]),
        (4, 8,  [4, 4, 4]),
        (3, 6,  [3, 3, 3, 3]),
        (5, 10, [5, 2, 5]),
    ])
    def test_bf16_partition_256x384(self, partSizeN, expected_peak_B, expected_partN):
        """BF16 256x384 (tilesN=12) with partitions along N."""
        cfg = make_cfg_bf16(MT1=384, depthU=64, partSizeN=partSizeN)
        assert cfg.partitionSizesN == expected_partN
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        assert sched.tile_peaks['A'] == 16
        assert sched.tile_peaks['B'] == expected_peak_B
        assert not sched.needs_unrolling
        self._assert_no_conflict_and_unrolling(sched)

    @pytest.mark.parametrize("partSizeN,expected_peak_B,expected_partN", [
        (4, 8,  [4, 3, 4]),
        (3, 6,  [3, 2, 3, 3]),
        (6, 12, [6, 5]),
    ])
    def test_bf16_partition_256x352(self, partSizeN, expected_peak_B, expected_partN):
        """BF16 256x352 (tilesN=11, odd) with partitions along N."""
        cfg = make_cfg_bf16(MT1=352, depthU=64, partSizeN=partSizeN)
        assert cfg.partitionSizesN == expected_partN
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        assert sched.tile_peaks['A'] == 16
        assert sched.tile_peaks['B'] == expected_peak_B
        assert not sched.needs_unrolling
        self._assert_no_conflict_and_unrolling(sched)

    @pytest.mark.parametrize("partSizeN,expected_peak_B,expected_partN", [
        (4, 8,  [4, 4, 3, 4, 4, 4]),
        (6, 12, [6, 5, 6, 6]),
        (8, 16, [8, 7, 8]),
    ])
    def test_bf16_partition_256x368(self, partSizeN, expected_peak_B, expected_partN):
        """BF16 256x368 miWaveGroup=[4,1] (tilesM=4, tilesN=23) with partitions along N."""
        cfg = make_cfg_bf16(MT1=368, depthU=64, miWaveGroup=[4, 1],
                            partSizeN=partSizeN)
        assert cfg.partitionSizesN == expected_partN
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        assert sched.tile_peaks['A'] == 8
        assert sched.tile_peaks['B'] == expected_peak_B
        assert not sched.needs_unrolling
        self._assert_no_conflict_and_unrolling(sched)

    # ── Group 4: FP4 2x2 ──

    @pytest.mark.parametrize("depthU,expect_unrolling", [(256, True), (512, False)])
    def test_fp4_2x2(self, depthU, expect_unrolling):
        """FP4 256x256 with default miWaveGroup=[2,2]."""
        cfg = make_cfg_256x256_fp4(depthU=depthU)
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        assert sched.tile_peaks == {'A': 16, 'B': 16, 'SA': 8, 'SB': 8}
        assert sched.needs_unrolling == expect_unrolling

        for pi in range(cfg.numPartitions):
            for slot in sched._partitions[pi]:
                if slot.mfma:
                    assert len(slot.mfma.vgpr_tile_maps['SA']) > 0
                    assert len(slot.mfma.vgpr_tile_maps['SB']) > 0

        self._assert_no_conflict_and_unrolling(sched)

    # ── Group 5: FP4 1x4 / 4x1 ──

    @pytest.mark.parametrize("depthU,expect_unrolling", [(256, True), (512, False)])
    def test_fp4_1x4(self, depthU, expect_unrolling):
        """FP4 256x256 miWaveGroup=[1,4]: 16 M-tiles, 4 N-tiles."""
        cfg = make_cfg_256x256_fp4(depthU=depthU, miWaveGroup=[1, 4])
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        assert sched.tile_peaks == {'A': 32, 'B': 8, 'SA': 16, 'SB': 4}
        assert sched.needs_unrolling == expect_unrolling
        self._assert_no_conflict_and_unrolling(sched)

    @pytest.mark.parametrize("depthU,expect_unrolling", [(256, True), (512, False)])
    def test_fp4_4x1(self, depthU, expect_unrolling):
        """FP4 256x256 miWaveGroup=[4,1]: 4 M-tiles, 16 N-tiles."""
        cfg = make_cfg_256x256_fp4(depthU=depthU, miWaveGroup=[4, 1])
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        assert sched.tile_peaks == {'A': 8, 'B': 32, 'SA': 4, 'SB': 16}
        assert sched.needs_unrolling == expect_unrolling
        self._assert_no_conflict_and_unrolling(sched)

    # ── Group 6: BF16 with lrA/lrB k=2 granularity ──

    @pytest.mark.parametrize("depthU,expect_unrolling", [(64, True), (128, False)])
    def test_bf16_lr_k2(self, depthU, expect_unrolling):
        """BF16 with LR k=2 granularity — odd k-groups trigger unrolling."""
        cfg = make_cfg_bf16(
            depthU=depthU,
            lrA=ReadGranularity(mn=1, k=2),
            lrB=ReadGranularity(mn=1, k=2),
        )
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        assert sched.tile_peaks == {'A': 16, 'B': 16}
        assert sched.needs_unrolling == expect_unrolling
        self._assert_no_conflict_and_unrolling(sched)


# ══════════════════════════════════════════════════════════════
# Step 3: Place GRs
# ══════════════════════════════════════════════════════════════

class TestPlaceGRs:

    def test_1x1_k1_DU256(self):
        """256x256, DU256, FP4. GR order: A, B, SA, SB.
        Total loads: A(8)+B(8)+SA(1)+SB(1)=18, 9 per slot.
        """
        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        slots = sched.place_GRs()

        _assert_slot_grs(slots[0], ['A', 'B'])
        _assert_gr(slots[0], 'A', 0, 2, 0, 8)
        _assert_gr(slots[0], 'B', 0, 2, 0, 1)

        _assert_slot_grs(slots[1], ['B', 'SA', 'SB'])
        _assert_gr(slots[1], 'B', 0, 2, 1, 8)
        _assert_gr(slots[1], 'SA', 0, 2, 0, 8)
        _assert_gr(slots[1], 'SB', 0, 2, 0, 8)

    def test_1x1_k1_DU512(self):
        """256x256, DU512, FP4. GR k=2 → two k-chunks. grSA/SB k=4 → full MT.
        Spaced-out distribution: B splits across s0/s1, SA migrates to s1, A k=2-4 sits whole at s2.
        """
        cfg = make_cfg_256x256_fp4(depthU=512)
        sched = LogicalScheduler(cfg)
        slots = sched.place_GRs()

        _assert_slot_grs(slots[0], ['A', 'B'])
        _assert_gr(slots[0], 'A', 0, 2, 0, 8)
        _assert_gr(slots[0], 'B', 0, 2, 0, 1)
        _assert_slot_grs(slots[1], ['B', 'SA'])
        _assert_gr(slots[1], 'B', 0, 2, 1, 8)
        _assert_gr(slots[1], 'SA', 0, 4, 0, 8)
        _assert_slot_grs(slots[2], ['SB', 'A'])
        _assert_gr(slots[2], 'SB', 0, 4, 0, 8)
        _assert_gr(slots[2], 'A', 2, 4, 0, 8)
        _assert_slot_grs(slots[3], ['B'])
        _assert_gr(slots[3], 'B', 2, 4, 0, 8)

    def test_2x2_k1_DU256(self):
        """256x256, DU256, FP4, 2x2 partition. Cross-MT dedup removes n+1 duplicates.
        Spaced-out distribution: tensors split into smaller atoms across more slots;
        scale GRs migrate to the last slot of the last partition.
        """
        cfg = make_cfg_256x256_fp4(partSizeM=4, partSizeN=4)
        sched = LogicalScheduler(cfg)
        slots = sched.place_GRs()
        parts = sched._partitions

        # P0: A n+1 split, B n+1 starts at s1
        _assert_slot_grs(parts[0][0], ['A'], "P0 s0")
        _assert_gr(parts[0][0], 'A', 0, 2, 4, 7, mt=1)
        _assert_slot_grs(parts[0][1], ['A', 'B'], "P0 s1")
        _assert_gr(parts[0][1], 'A', 0, 2, 7, 8, mt=1)
        _assert_gr(parts[0][1], 'B', 0, 2, 4, 5, mt=1)

        # P1: B n+1 finishes, A n+2 starts at s1
        _assert_slot_grs(parts[1][0], ['B'], "P1 s0")
        _assert_gr(parts[1][0], 'B', 0, 2, 5, 7, mt=1)
        _assert_slot_grs(parts[1][1], ['B', 'A'], "P1 s1")
        _assert_gr(parts[1][1], 'B', 0, 2, 7, 8, mt=1)
        _assert_gr(parts[1][1], 'A', 0, 2, 0, 1, mt=2)

        # P2: A n+2 finishes, B n+2 starts
        _assert_slot_grs(parts[2][0], ['A'], "P2 s0")
        _assert_gr(parts[2][0], 'A', 0, 2, 1, 4, mt=2)
        _assert_slot_grs(parts[2][1], ['B'], "P2 s1")
        _assert_gr(parts[2][1], 'B', 0, 2, 0, 2, mt=2)

        # P3: B n+2 finishes at s0; SA/SB n+2 at s1 (last slot)
        _assert_slot_grs(parts[3][0], ['B'], "P3 s0")
        _assert_gr(parts[3][0], 'B', 0, 2, 2, 4, mt=2)
        _assert_slot_grs(parts[3][1], ['SA', 'SB'], "P3 s1")
        _assert_gr(parts[3][1], 'SA', 0, 2, 0, 8, mt=2)
        _assert_gr(parts[3][1], 'SB', 0, 2, 0, 8, mt=2)

    def test_2x2_k1_DU512(self):
        """256x256, DU512, FP4, 2x2 partition. GR k=2 × 2 chunks + scale.
        Spaced-out distribution: A k=0-2 and k=2-4 chunks split across slots within P0;
        B n+1 starts in P0 s3; SA/SB migrate to P3 s3 (last slot of last partition).
        """
        cfg = make_cfg_256x256_fp4(depthU=512, partSizeM=4, partSizeN=4)
        sched = LogicalScheduler(cfg)
        sched.place_GRs()
        parts = sched._partitions

        # P0 s0: A n+1 k=0-2 [4-7]
        _assert_slot_grs(parts[0][0], ['A'], "P0 s0")
        _assert_gr(parts[0][0], 'A', 0, 2, 4, 7, mt=1)
        # P0 s1: A n+1 k=0-2 tail [7-8] + A n+1 k=2-4 head [4-5]
        _assert_slot_grs(parts[0][1], ['A', 'A'], "P0 s1")
        _assert_gr(parts[0][1], 'A', 0, 2, 7, 8, mt=1, idx=0)
        _assert_gr(parts[0][1], 'A', 2, 4, 4, 5, mt=1, idx=1)
        # P0 s2: A n+1 k=2-4 [5-7]
        _assert_slot_grs(parts[0][2], ['A'], "P0 s2")
        _assert_gr(parts[0][2], 'A', 2, 4, 5, 7, mt=1)
        # P0 s3: A n+1 k=2-4 tail [7-8] + B n+1 k=0-2 head [4-5]
        _assert_slot_grs(parts[0][3], ['A', 'B'], "P0 s3")
        _assert_gr(parts[0][3], 'A', 2, 4, 7, 8, mt=1)
        _assert_gr(parts[0][3], 'B', 0, 2, 4, 5, mt=1)

        # P2 s3: SA+SB n+2 (migrated from P3 s0 to P2 s3)
        _assert_slot_grs(parts[2][3], ['SA', 'SB'], "P2 s3")
        _assert_gr(parts[2][3], 'SA', 0, 4, 0, 8, mt=2)
        _assert_gr(parts[2][3], 'SB', 0, 4, 0, 8, mt=2)

    def test_10x1_bf16(self):
        """320x320, BF16, 10x1 partition. No scales."""
        cfg = make_cfg_bf16(320, 320, partSizeM=1, partSizeN=10)
        sched = LogicalScheduler(cfg)
        sched.place_GRs()
        parts = sched._partitions

        # P0..P4: A atoms
        for pi in range(4):
            _assert_slot_grs(parts[pi][0], ['A'], f"P{pi} s0")
            _assert_gr(parts[pi][0], 'A', 0, 2, pi*2+1, pi*2+2, mt=1)

        # P5..P9: B atoms
        for pi in range(5, 10):
            b_idx = (pi - 5) * 2
            _assert_slot_grs(parts[pi][0], ['B'], f"P{pi} s0")
            _assert_gr(parts[pi][0], 'B', 0, 2, b_idx, b_idx+1, mt=2)

    def test_1x19_bf16(self):
        """320x304, BF16, 1x19 partition, wg=[4,1]. High partition count with sourceSwap."""
        cfg = make_cfg_bf16(320, 304, partSizeN=1,
                            miWaveGroup=[4, 1], sourceSwap=True)
        sched = LogicalScheduler(cfg)
        sched.place_GRs()
        parts = sched._partitions

        # Collect (partition, subIterK) for each tensor/mt.
        b_n1 = []
        a_n2 = []
        for pi in range(19):
            for slot in parts[pi]:
                for gr in slot.grs:
                    if gr.tensor == 'B' and gr.mtIteration == 1:
                        b_n1.append((pi, slot.subIterK))
                    elif gr.tensor == 'A' and gr.mtIteration == 2:
                        a_n2.append((pi, slot.subIterK))

        # 9 B n+1 atoms (grB.mn=2) spread across P0..P10
        assert b_n1 == [
            (0,0),(1,0),(2,1),(3,1),(5,0),(6,0),(7,1),(8,1),(10,0)]
        # 5 A n+2 atoms spread across P11..P16
        assert a_n2 == [(11,0),(12,1),(13,1),(15,0),(16,0)]

        # B n+2 at P17
        _assert_slot_grs(parts[17][1], ['B'], "P17 s1")
        _assert_gr(parts[17][1], 'B', 0, 2, 0, 2, mt=2)

    def test_pgr1_gr_before_corresponding_lr(self):
        """PGR=1: GR(T, mt=X) must be placed strictly before first LR(T, mt=X)."""
        cfg = make_cfg_bf16_pgr1()
        sched = LogicalScheduler(cfg)
        sched.place_GRs()
        parts = sched._partitions
        numK = cfg.numSubIterK

        # Build per-(tensor, mt) earliest LR flat index
        lr_first = {}
        for pi, slots in enumerate(parts):
            for slot in slots:
                flat = pi * numK + slot.subIterK
                for lr in slot.lrs:
                    key = (lr.tensor, lr.mtIteration)
                    if key not in lr_first or flat < lr_first[key]:
                        lr_first[key] = flat

        for pi, slots in enumerate(parts):
            for slot in slots:
                flat = pi * numK + slot.subIterK
                for gr in slot.grs:
                    key = (gr.tensor, gr.mtIteration)
                    if key in lr_first:
                        assert flat < lr_first[key], (
                            f"GR({gr.tensor}, mt={gr.mtIteration}) at flat={flat} "
                            f"must be before first LR at flat={lr_first[key]}")


# ══════════════════════════════════════════════════════════════
# Step 4: Annotate deps
# ══════════════════════════════════════════════════════════════

class TestAnnotateDeps:

    def test_1x1_DU256(self):
        """256x256, DU256, FP4. MFMA→LR, LR→GR, GR→LR collision deps."""
        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.annotate_deps()
        parts = sched._partitions
        s0, s1 = parts[0][0], parts[0][1]

        # MFMA(k=0): all deps MT-1
        mfma0_deps = _dep_refs(s0.mfma)
        assert ('LR', 'A',  0, 1, -1) in mfma0_deps
        assert ('LR', 'B',  0, 1, -1) in mfma0_deps
        assert ('LR', 'SA', 0, 1, -1) in mfma0_deps
        assert ('LR', 'SB', 0, 1, -1) in mfma0_deps
        assert len(mfma0_deps) == 4

        # LR A @s0 → GR A @s0 (MT-2)
        assert _dep_refs(_get_lr(s0, 'A')) == [('GR', 'A', 0, 0, -2)]
        # LR B @s0 → last GR B @s1 (MT-2)
        assert _dep_refs(_get_lr(s0, 'B')) == [('GR', 'B', 0, 1, -2)]

        # GR A @s0 → LR A collision (MT 0)
        gr_a0 = [gr for gr in s0.grs if gr.tensor == 'A'][0]
        assert _dep_refs(gr_a0) == [('LR', 'A', 0, 0, 0)]

        # MFMA(k=1): LR A/B @s0 (MT 0), SA/SB @s1 (MT-1)
        mfma1_deps = _dep_refs(s1.mfma)
        assert ('LR', 'A', 0, 0, 0) in mfma1_deps
        assert ('LR', 'SA', 0, 1, -1) in mfma1_deps
        assert len(mfma1_deps) == 4

    def test_2x2_DU512(self):
        """256x256, DU512, FP4, 2x2 partition. Per-partition deps."""
        cfg = make_cfg_256x256_fp4(depthU=512, partSizeM=4, partSizeN=4)
        sched = LogicalScheduler(cfg)
        sched.annotate_deps()
        parts = sched._partitions

        assert len(parts) == 4
        assert len(parts[0]) == 4

        # P0 MFMA(k=0): deps from P3
        mfma_p0_s0 = _dep_refs(parts[0][0].mfma)
        assert ('LR', 'A', 3, 3, -1) in mfma_p0_s0
        assert len(mfma_p0_s0) == 4

        # P3 MFMA(k=0): deps on LRs that loaded subIterK=0 data for P3 tiles
        mfma_p3_s0 = _dep_refs(parts[3][0].mfma)
        assert ('LR', 'A', 0, 3, 0) in mfma_p3_s0
        assert ('LR', 'SA', 0, 2, 0) in mfma_p3_s0


# ══════════════════════════════════════════════════════════════
# Step 4b: Remove unnecessary deps (individual passes)
# ══════════════════════════════════════════════════════════════

class TestRemoveUnnecessaryGrDeps:

    def test_removes_mt_minus_2_deps(self):
        """MT-2 GR deps are removed when an MT-1 dep from a later LR guarantees the data."""
        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.annotate_deps()
        s0 = sched._partitions[0][0]

        # Before: LR A @s0 has dep on GR A @s0 (MT-2)
        lr_a_before = _get_lr(s0, 'A')
        assert len(lr_a_before.deps) == 1
        assert lr_a_before.deps[0].mt_offset == -2

        sched.remove_unnecessary_gr_deps()

        # After: dep removed (guaranteed by LR A @s1's MT-1 dep)
        lr_a_after = _get_lr(s0, 'A')
        assert len(lr_a_after.deps) == 0


class TestRemoveUnnecessaryLrDeps:

    def test_removes_covered_collision_deps(self):
        """GR→LR collision deps removed when covered by earlier sync."""
        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.annotate_deps()
        sched.remove_unnecessary_gr_deps()

        # Before remove_unnecessary_lr_deps: GR B @s1 has collision dep
        s1 = sched._partitions[0][1]
        gr_b1 = [gr for gr in s1.grs if gr.tensor == 'B'][0]
        deps_before = len(gr_b1.deps)

        sched.remove_unnecessary_lr_deps()

        gr_b1_after = [gr for gr in sched._partitions[0][1].grs if gr.tensor == 'B'][0]
        assert len(gr_b1_after.deps) == 0


# ══════════════════════════════════════════════════════════════
# Step 5: Remove cross-subIterK deps
# ══════════════════════════════════════════════════════════════

class TestRemoveCrossDeps:

    def test_1x1_DU256(self):
        """256x256, DU256, FP4. Cross deps → preOps, same-subIterK deps preserved."""
        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.remove_cross_deps()
        parts = sched._partitions
        s0, s1 = parts[0][0], parts[0][1]

        # MFMA(k=0): all cross → wait_lr
        assert _preop_kinds(s0.mfma) == [('wait_lr', False, None)]
        assert len(s0.mfma.deps) == 0

        # LR A @s0: dep removed (guaranteed by prev MT)
        assert _preop_kinds(_get_lr(s0, 'A')) == []
        assert len(_get_lr(s0, 'A').deps) == 0

        # GR A @s0: same-subIterK dep preserved
        gr_a0 = [gr for gr in s0.grs if gr.tensor == 'A'][0]
        assert _preop_kinds(gr_a0) == [('wait_lr', True, None)]
        assert len(gr_a0.deps) == 1

        # MFMA(k=1): all cross → wait_lr
        assert _preop_kinds(s1.mfma) == [('wait_lr', False, None)]
        assert len(s1.mfma.deps) == 0

        # LR A @s1: wait_gr_sync with counts
        lr_a1 = _get_lr(s1, 'A')
        assert _preop_kinds(lr_a1) == [('wait_gr', True, {'A': 8, 'B': 9, 'SA': 1, 'SB': 1})]

        # LR B @s1: wait_gr with has_sync
        lr_b1 = _get_lr(s1, 'B')
        assert _preop_kinds(lr_b1) == [('wait_gr', True, {'A': 8, 'B': 1, 'SA': 1, 'SB': 1})]

        # LR SA @s1
        lr_sa1 = _get_lr(s1, 'SA')
        assert _preop_kinds(lr_sa1) == [('wait_gr', True, {'A': 8, 'B': 1, 'SA': 0, 'SB': 1})]

        # LR SB @s1
        lr_sb1 = _get_lr(s1, 'SB')
        assert _preop_kinds(lr_sb1) == [('wait_gr', True, {'A': 8, 'B': 1, 'SA': 0, 'SB': 0})]

        # GR B @s1: dep removed
        gr_b1 = [gr for gr in s1.grs if gr.tensor == 'B'][0]
        assert _preop_kinds(gr_b1) == []
        assert len(gr_b1.deps) == 0

    def test_2x2_DU512(self):
        """256x256, DU512, FP4, 2x2 partition. Spot checks."""
        cfg = make_cfg_256x256_fp4(depthU=512, partSizeM=4, partSizeN=4)
        sched = LogicalScheduler(cfg)
        sched.remove_cross_deps()
        parts = sched._partitions

        # All P0 MFMAs have wait_lr
        assert _preop_kinds(parts[0][0].mfma) == [('wait_lr', False, None)]

        # All P3 MFMAs have wait_lr
        for slot in parts[3]:
            assert _preop_kinds(slot.mfma) == [('wait_lr', False, None)]

        # LR A @P3:s3: wait_gr_sync with A=20
        lr_a_p3_s3 = _get_lr(parts[3][3], 'A')
        assert lr_a_p3_s3.preOps[0].wait_gr_counts.A == 20

        # LR SA @P3:s2: wait_gr_sync with SA=1
        lr_sa_p3_s2 = _get_lr(parts[3][2], 'SA')
        assert lr_sa_p3_s2.preOps[0].wait_gr_counts.SA == 1


# ══════════════════════════════════════════════════════════════
# Step 6: Insert gr/lr inc
# ══════════════════════════════════════════════════════════════

class TestInsertGrLrInc:

    def test_1x1_DU256(self):
        """256x256, DU256, FP4. gr_inc at MT transitions, lr_inc on wrap-around."""
        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.insert_gr_lr_inc()
        parts = sched._partitions
        s0, s1 = parts[0][0], parts[0][1]

        # GR A @s0: mt=n+2, A was n → gr_inc(A)
        gr_a0 = [gr for gr in s0.grs if gr.tensor == 'A'][0]
        assert _preop_inc_tensors(gr_a0, 'gr_inc') == ['A']
        gr_b0 = [gr for gr in s0.grs if gr.tensor == 'B'][0]
        assert _preop_inc_tensors(gr_b0, 'gr_inc') == ['B']

        # LR A @s1: mt=n+1, A was n+2 → lr_inc(A)
        assert _preop_inc_tensors(_get_lr(s1, 'A'), 'lr_inc') == ['A']
        assert _preop_inc_tensors(_get_lr(s1, 'B'), 'lr_inc') == ['B']
        assert _preop_inc_tensors(_get_lr(s1, 'SA'), 'lr_inc') == ['SA']
        assert _preop_inc_tensors(_get_lr(s1, 'SB'), 'lr_inc') == ['SB']

        # GR B @s1: same mt as GR B @s0 → no duplicate gr_inc
        gr_b1 = [gr for gr in s1.grs if gr.tensor == 'B'][0]
        assert _preop_inc_tensors(gr_b1, 'gr_inc') == []

        # GR SA, SB @s1: mt transition → gr_inc
        gr_sa1 = [gr for gr in s1.grs if gr.tensor == 'SA'][0]
        assert _preop_inc_tensors(gr_sa1, 'gr_inc') == ['SA']
        gr_sb1 = [gr for gr in s1.grs if gr.tensor == 'SB'][0]
        assert _preop_inc_tensors(gr_sb1, 'gr_inc') == ['SB']

        # Ordering: wait_gr_sync before lr_inc
        lr_a1 = _get_lr(s1, 'A')
        assert lr_a1.preOps[0].kind == 'wait_gr'
        assert lr_a1.preOps[1].kind == 'lr_inc'

    def test_multipartition_bf16_gr_inc_only_at_base_id_0(self):
        """128x128, BF16, 2x2 partition. gr_inc only at tileId_start=0."""
        cfg = make_cfg_bf16(128, 128, partSizeM=2, partSizeN=2)
        assert cfg.numPartitions == 4

        sched = LogicalScheduler(cfg)
        sched.insert_gr_lr_inc()
        parts = sched._partitions

        # P0/P1: GR with tileId_start > 0 → no gr_inc
        for pi in [0, 1]:
            tensor = 'A' if pi == 0 else 'B'
            for slot in parts[pi]:
                for gr in slot.grs:
                    if gr.tensor == tensor:
                        assert gr.tiles.tileId_start > 0
                        assert _preop_inc_tensors(gr, 'gr_inc') == []

        # P2: GR A with tileId_start=0 → gr_inc(A)
        gr_a_p2 = [gr for s in parts[2] for gr in s.grs
                    if gr.tensor == 'A' and gr.tiles.tileId_start == 0]
        assert len(gr_a_p2) == 1
        assert _preop_inc_tensors(gr_a_p2[0], 'gr_inc') == ['A']


# ══════════════════════════════════════════════════════════════
# Step 7: Compute inflight loads
# ══════════════════════════════════════════════════════════════

class TestComputeInflightLoads:

    def test_DU256(self):
        """Validate inflight load counts match remove_cross_deps output."""
        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.remove_cross_deps()

        s0 = sched._partitions[0][0]
        s1 = sched._partitions[0][1]

        # LR A @s0: dep removed (guaranteed by prev MT)
        assert _preop_kinds(_get_lr(s0, 'A')) == []

        # LR B @s1: counts
        lr_b1 = _get_lr(s1, 'B')
        assert lr_b1.preOps[0].wait_gr_counts.A == 8
        assert lr_b1.preOps[0].wait_gr_counts.B == 1
        assert lr_b1.preOps[0].wait_gr_counts.SA == 1

        # LR SA @s1: counts
        lr_sa1 = _get_lr(s1, 'SA')
        assert lr_sa1.preOps[0].wait_gr_counts.A == 8
        assert lr_sa1.preOps[0].wait_gr_counts.B == 1

        # LR A @s1: counts
        lr_a1 = _get_lr(s1, 'A')
        assert lr_a1.preOps[0].wait_gr_counts.A == 8
        assert lr_a1.preOps[0].wait_gr_counts.B == 9
        assert lr_a1.preOps[0].wait_gr_counts.SA == 1
        assert lr_a1.preOps[0].wait_gr_counts.SB == 1

    def test_fp8_DU128_asymmetric_A_lt_B(self):
        """FP8 DU=128, MT=128x192, waveGroup=(2,2), PGR=2.

        Regression for _compute_inflight_loads bug (commit 055ecc8):
        With flat_len=1 (1 partition × 1 subIterK), wraps_needed=1
        (cross-MT dep), consumer_flat==dep_flat==0, the old wrap-counting
        code walked the single slot twice instead of once — overcounting all
        GRs in the slot as extra inflight.

        Correct behavior: walk exactly wraps_needed*flat_len=1 step; on the
        final step stop immediately at the dep GR (A), yielding A=0.  The B
        GRs that were emitted after A (higher sort key) are still counted.

        Old (buggy) counts for LR A: A=4, B=12.
        New (correct) counts for LR A: A=0, B=6.
        """
        cfg = make_cfg_fp8(128, 192, waveGroup=(2, 2))
        assert cfg.numSubIterK == 1, "FP8 DU=128 / matrixInstK=128 → single subIterK"
        sched = LogicalScheduler(cfg)
        sched.remove_cross_deps()

        s0 = sched._partitions[0][0]

        # LR A: dep is GR A (emitted first → last in reverse order).
        # All B GRs (emitted after A, encountered first in backward walk) are inflight.
        lr_a = _get_lr(s0, 'A')
        assert lr_a.preOps[0].wait_gr_counts.A == 0
        assert lr_a.preOps[0].wait_gr_counts.B == 6  # 6 B tiles in MT192 / wg2

        # LR B: dep is GR B (emitted last → first in reverse order); nothing after it.
        lr_b = _get_lr(s0, 'B')
        assert lr_b.preOps[0].wait_gr_counts.A == 0
        assert lr_b.preOps[0].wait_gr_counts.B == 0

    def test_fp8_DU128_asymmetric_A_gt_B(self):
        """FP8 DU=128, MT=448x64, waveGroup=(4,1), PGR=2.

        Same _compute_inflight_loads regression as test_fp8_DU128_asymmetric_A_lt_B
        but with more A tiles than B tiles (7A vs 4B).

        Old (buggy) counts for LR A: A=7, B=8.
        New (correct) counts for LR A: A=0, B=4.
        """
        cfg = make_cfg_fp8(448, 64, waveGroup=(4, 1))
        assert cfg.numSubIterK == 1
        sched = LogicalScheduler(cfg)
        sched.remove_cross_deps()

        s0 = sched._partitions[0][0]

        lr_a = _get_lr(s0, 'A')
        assert lr_a.preOps[0].wait_gr_counts.A == 0
        assert lr_a.preOps[0].wait_gr_counts.B == 4  # 4 B tiles in MT64 / wg1

        lr_b = _get_lr(s0, 'B')
        assert lr_b.preOps[0].wait_gr_counts.A == 0
        assert lr_b.preOps[0].wait_gr_counts.B == 0


# ══════════════════════════════════════════════════════════════
# Step 8: Group LR/GR
# ══════════════════════════════════════════════════════════════

class TestGroupLrGr:

    def test_1x1_DU256(self):
        """256x256, DU256, FP4. Chain grouping with merged preOps."""
        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.group_lr_gr()
        parts = sched._partitions
        s0, s1 = parts[0][0], parts[0][1]

        # s0: LR chain A←B, no preOps (deps removed)
        lr_a0 = _get_lr(s0, 'A')
        lr_b0 = _get_lr(s0, 'B')
        assert lr_a0.preOps == []
        assert lr_b0.deps[0].ref is lr_a0

        # s0: GR chain A←B, first GR dep→last LR
        gr_a0 = [gr for gr in s0.grs if gr.tensor == 'A'][0]
        gr_b0 = [gr for gr in s0.grs if gr.tensor == 'B'][0]
        assert gr_a0.preOps[0].kind == 'wait_lr'
        assert gr_a0.preOps[1].kind == 'gr_inc'
        assert gr_a0.deps[0].ref is lr_b0
        assert gr_b0.deps[0].ref is gr_a0

        # s1: LR chain A←B←SA←SB, merged preOps on A
        lr_a1 = _get_lr(s1, 'A')
        lr_b1 = _get_lr(s1, 'B')
        lr_sa1 = _get_lr(s1, 'SA')
        lr_sb1 = _get_lr(s1, 'SB')
        assert lr_a1.preOps[0].kind == 'wait_gr'
        assert lr_a1.preOps[0].wait_gr_counts.A == 8
        assert len(lr_a1.preOps) == 5  # wait_gr_sync + 4 lr_inc
        assert lr_b1.deps[0].ref is lr_a1
        assert lr_sa1.deps[0].ref is lr_b1
        assert lr_sb1.deps[0].ref is lr_sa1

        # s1: GR chain B←SA←SB (collision deps removed — covered by
        # MFMA@s0 which already syncs on the same LR SA/SB)
        gr_b1 = [gr for gr in s1.grs if gr.tensor == 'B'][0]
        gr_sa1 = [gr for gr in s1.grs if gr.tensor == 'SA'][0]
        gr_sb1 = [gr for gr in s1.grs if gr.tensor == 'SB'][0]
        assert gr_b1.preOps == []
        assert gr_sa1.preOps[0].kind == 'gr_inc'
        assert gr_sb1.deps[0].ref is gr_sa1


# ══════════════════════════════════════════════════════════════
# Step 9: Remove unnecessary wait_lr_sync
# ══════════════════════════════════════════════════════════════

class TestRemoveUnnecessaryWaitLrSync:

    def test_bf16_1x1(self):
        """256x256, BF16, 1x1. Verify wait_lr_sync handling."""
        cfg = make_cfg_bf16(256, 256)
        sched = LogicalScheduler(cfg)
        sched.group_lr_gr()

        # After group_lr_gr, get the preOps state
        sched.remove_unnecessary_wait_lr_sync()
        parts = sched._partitions
        # Verify no crash and GR preOps are present
        for slot in parts[0]:
            for gr in slot.grs:
                for op in gr.preOps:
                    assert op.kind in ('wait_lr', 'sync', 'gr_inc', 'lr_inc',
                                        'wait_gr', 'wait_gr', 'wait_lr', 'skip')


# ══════════════════════════════════════════════════════════════
# Step 10: Emit
# ══════════════════════════════════════════════════════════════

class TestEmit:

    def test_1x1_DU256(self):
        """256x256, DU256, FP4. EmittedModule chains with correct links."""
        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        result = sched.emit()

        assert len(result) == 1       # 1 partition
        assert len(result[0]) == 2    # 2 subIterKs

        # subIterK=0: mfma + 2 LRs + 2 GRs = 5 primary
        em0 = result[0][0]
        primary0 = [e for e in em0 if e.opType in ('mfma', 'lr', 'gr')]
        assert len(primary0) == 5
        assert primary0[0].opType == 'mfma'

        # MFMA has wait_lr before-link
        assert em0[0].before is not None
        assert em0[em0[0].before].opType == 'wait_lr'

        # No wait_gr in s0
        assert len([e for e in em0 if e.opType == 'wait_gr']) == 0

        # subIterK=1: mfma + 4 LRs + 3 GRs = 8 primary
        em1 = result[0][1]
        primary1 = [e for e in em1 if e.opType in ('mfma', 'lr', 'gr')]
        assert len(primary1) == 8
        assert len([e for e in em1 if e.opType == 'wait_gr']) == 1

        # No self-loops
        for em_list in [em0, em1]:
            for e in em_list:
                if e.before is not None:
                    assert e.before != e.moduleId


# ══════════════════════════════════════════════════════════════
# Step 11: Build loop variants
# ══════════════════════════════════════════════════════════════

class TestBuildPreloop:

    def test_256x256_fp4(self):
        """Preloop contains GR(n), LR, GR(n+1). No GR(n+2) or gr_inc."""
        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.build()
        sched.build_preloop()

        preloop = sched._preloop_emitted
        assert len(preloop) > 0

        # Collect all opTypes across all partitions/subIterKs
        all_ops = [em.opType for partition in preloop
                   for group in partition for em in group]

        # Should have lr and gr ops
        assert 'lr' in all_ops
        assert 'gr' in all_ops

        # Preloop has gr_inc for the n→n+1 transition but not lr_inc
        # (LR data is loaded fresh, not from a previous MT iteration)
        assert 'lr_inc' not in all_ops


class TestBuildNGLL:

    def test_256x256_fp4(self):
        """NGLL removes GR(n+2). LR, MFMA, and gr_inc preserved (gr_inc
        still needed to swap LW for tail entry)."""
        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.build()
        sched.build_ngll()

        ngll = sched._ngll_emitted
        assert len(ngll) > 0

        all_ops = [em.opType for partition in ngll
                   for group in partition for em in group]

        assert 'mfma' in all_ops
        assert 'lr' in all_ops
        # NGLL must drop GR(n+2) loads
        for partition in ngll:
            for group in partition:
                for em in group:
                    if em.opType == 'gr':
                        assert em.source.mtIteration != 2, \
                            "NGLL should not contain GR(n+2)"


class TestBuildNLL:

    def test_256x256_fp4(self):
        """NLL removes GRs, LR(n+1), and (PGR=2) gr_inc. LR(n), MFMA, and
        lr_inc remain — lr_inc is required to swap LR base for the next MT."""
        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.build()
        sched.build_nll()

        nll = sched._nll_emitted
        assert len(nll) > 0

        all_ops = [em.opType for partition in nll
                   for group in partition for em in group]

        assert 'mfma' in all_ops
        # NLL drops global loads and (under PGR=2) gr_inc
        assert 'gr' not in all_ops
        assert 'gr_inc' not in all_ops
        # LR(n+1) must be gone — only LR(n) remains
        for partition in nll:
            for group in partition:
                for em in group:
                    if em.opType == 'lr':
                        assert em.source.mtIteration == 0, \
                            "NLL should only contain LR(n)"


# ══════════════════════════════════════════════════════════════
# SchedulerConfig from TileInfo
# ══════════════════════════════════════════════════════════════

class TestFromTileInfo:

    def test_64x64_fp4(self):
        """MT=64, FP4 — matches design doc Example Granularities 1 tile counts."""
        kernel = create_kernel(64, 64, fp4=True)
        tiA = makeTileInfo('A', kernel)
        tiB = makeTileInfo('B', kernel)
        scaleTiA = makeTileInfo('MXSA', kernel)
        scaleTiB = makeTileInfo('MXSB', kernel)

        cfg = SchedulerConfig(
            numMFMATilesM=tiA.localMMATileGrid[0],
            numMFMATilesN=tiB.localMMATileGrid[0],
            numSubIterK=tiA.localMMATileGrid[1],
            lrA=ReadGranularity(mn=1, k=1),
            lrB=ReadGranularity(mn=1, k=1),
            grA=ReadGranularity(mn=1, k=2),
            grB=ReadGranularity(mn=1, k=2),
            lrSA=ReadGranularity(mn=2, k=2),
            lrSB=ReadGranularity(mn=2, k=2),
            grSA=ReadGranularity(mn=scaleTiA.localMMATileGrid[0], k=scaleTiA.localMMATileGrid[1]),
            grSB=ReadGranularity(mn=scaleTiB.localMMATileGrid[0], k=scaleTiB.localMMATileGrid[1]),
        )

        assert cfg.numMFMATilesM == 2  # 64/16/2
        assert cfg.numMFMATilesN == 2
        assert cfg.numSubIterK == 2
        assert cfg.hasScale

        sched = LogicalScheduler(cfg)
        partitions = sched.place_LRs()
        assert len(partitions[0]) == 2
        assert partitions[0][0].mfma.tileA.tileId_end == 2

    def test_256x256_no_scale(self):
        """MT=256, no scale."""
        cfg = make_cfg_bf16(256, 256)
        assert cfg.numMFMATilesM == 8  # 256/16/2
        assert cfg.numMFMATilesN == 8
        assert cfg.numSubIterK == 2
        assert not cfg.hasScale


# ══════════════════════════════════════════════════════════════
# get_partition_candidates
# ══════════════════════════════════════════════════════════════

class TestPartitionCandidates:

    def test_square(self):
        """M==N: partitions N, full then divUp(N,2) down to 1."""
        kernel = create_kernel(256, 256)
        tiA = makeTileInfo('A', kernel)
        tiB = makeTileInfo('B', kernel)
        candidates = SchedulerConfig.get_partition_candidates(tiA, tiB)
        assert candidates == [(8, 8), (8, 4), (8, 3), (8, 2), (8, 1)]

    def test_n_larger(self):
        """N > M: partitions N."""
        kernel = create_kernel(64, 256)
        tiA = makeTileInfo('A', kernel)
        tiB = makeTileInfo('B', kernel)
        candidates = SchedulerConfig.get_partition_candidates(tiA, tiB)
        assert candidates == [(2, 8), (2, 4), (2, 3), (2, 2), (2, 1)]

    def test_m_larger(self):
        """M > N: partitions M."""
        kernel = create_kernel(256, 64)
        tiA = makeTileInfo('A', kernel)
        tiB = makeTileInfo('B', kernel)
        candidates = SchedulerConfig.get_partition_candidates(tiA, tiB)
        assert candidates == [(8, 2), (4, 2), (3, 2), (2, 2), (1, 2)]

    def test_prime_dim(self):
        """Prime-sized dimension: full then divUp down to 1."""
        cfg_tiA = MagicMock()
        cfg_tiB = MagicMock()
        cfg_tiA.localMMATileGrid = [5, 2]
        cfg_tiB.localMMATileGrid = [3, 2]
        candidates = SchedulerConfig.get_partition_candidates(cfg_tiA, cfg_tiB)
        assert candidates == [(5, 3), (3, 3), (2, 3), (1, 3)]

    def test_composite(self):
        """Composite dimension with mixed prime factors."""
        cfg_tiA = MagicMock()
        cfg_tiB = MagicMock()
        cfg_tiA.localMMATileGrid = [2, 2]
        cfg_tiB.localMMATileGrid = [6, 2]
        candidates = SchedulerConfig.get_partition_candidates(cfg_tiA, cfg_tiB)
        assert candidates == [(2, 6), (2, 3), (2, 2), (2, 1)]


# ══════════════════════════════════════════════════════════════
# _normalize_partition_sizes
# ══════════════════════════════════════════════════════════════

class TestNormalizePartitionSizes:
    """Pin the current behavior of SchedulerConfig._normalize_partition_sizes.

    These tests exercise the staticmethod directly; they bypass __post_init__
    so no full SchedulerConfig is constructed.
    """

    norm = staticmethod(SchedulerConfig._normalize_partition_sizes)

    # ── single-int spec ──────────────────────────────────────

    def test_zero_means_full_dim(self):
        assert self.norm(0, 16, 'N') == [16]

    def test_spec_equals_total(self):
        assert self.norm(16, 16, 'N') == [16]

    def test_divides_evenly(self):
        assert self.norm(8, 16, 'N') == [8, 8]
        assert self.norm(8, 24, 'N') == [8, 8, 8]
        assert self.norm(1, 5, 'N') == [1, 1, 1, 1, 1]

    def test_single_full_with_remainder(self):
        # num_full == 1 → [s, remainder]
        assert self.norm(8, 12, 'N') == [8, 4]
        assert self.norm(22, 23, 'N') == [22, 1]

    def test_remainder_placed_in_middle(self):
        # num_full=2, remainder=4, mid=1 → [s, rem, s]
        assert self.norm(8, 20, 'N') == [8, 4, 8]
        # num_full=2, remainder=7, mid=1 → [8,7,8]
        assert self.norm(8, 23, 'N') == [8, 7, 8]
        # num_full=5, remainder=3, mid=2 → [4,4,3,4,4,4]
        assert self.norm(4, 23, 'N') == [4, 4, 3, 4, 4, 4]
        # num_full=3, remainder=2, mid=1 → [4,2,4,4]
        assert self.norm(4, 14, 'N') == [4, 2, 4, 4]

    def test_spec_one(self):
        assert self.norm(1, 4, 'N') == [1, 1, 1, 1]

    # ── invalid single-int spec ──────────────────────────────

    def test_spec_negative_raises(self):
        with pytest.raises(AssertionError, match="must be in"):
            self.norm(-1, 16, 'N')

    def test_spec_larger_than_total_raises(self):
        with pytest.raises(AssertionError, match="must be in"):
            self.norm(20, 16, 'N')

    # ── explicit list spec ───────────────────────────────────

    def test_list_passthrough(self):
        assert self.norm([8, 8], 16, 'N') == [8, 8]
        assert self.norm([22, 1], 23, 'N') == [22, 1]
        assert self.norm([4, 4, 3, 4, 4, 4], 23, 'N') == [4, 4, 3, 4, 4, 4]

    def test_tuple_spec_returned_as_list(self):
        result = self.norm((8, 8), 16, 'N')
        assert result == [8, 8]
        assert isinstance(result, list)

    def test_list_wrong_sum_raises(self):
        with pytest.raises(AssertionError, match="must sum to 16"):
            self.norm([8, 7], 16, 'N')

    def test_list_with_zero_raises(self):
        with pytest.raises(AssertionError, match="must be >= 1"):
            self.norm([8, 0, 8], 16, 'N')

    def test_list_with_negative_raises(self):
        with pytest.raises(AssertionError, match="must be >= 1"):
            self.norm([10, -2, 8], 16, 'N')

    def test_dim_label_in_error(self):
        with pytest.raises(AssertionError, match="for M"):
            self.norm([1, 2], 16, 'M')

    # ── mn-aware behavior ────────────────────────────────────

    def test_mn_default_is_one(self):
        # Default mn=1: behavior identical to the pre-mn algorithm.
        assert self.norm(8, 23, 'N') == [8, 7, 8]

    def test_mn_already_aligned_passthrough(self):
        assert self.norm(8, 16, 'N', 2) == [8, 8]
        assert self.norm(8, 24, 'N', 4) == [8, 8, 8]

    def test_mn_snaps_spec_down(self):
        # 7 snaps down to 6 (largest mn-multiple <= 7), then split as usual.
        assert self.norm(7, 16, 'N', 2) == [6, 4, 6]

    def test_mn_snaps_spec_clamped_to_mn(self):
        # spec=1 with mn=2 snaps up to mn (the floor of valid sizes).
        assert self.norm(1, 16, 'N', 2) == [2] * 8

    def test_mn_remainder_stays_aligned(self):
        # total=22, s=8 → num_full=2, remainder=6 (even). mid=1 → [8,6,8].
        assert self.norm(8, 22, 'N', 2) == [8, 6, 8]
        # total=20, s=8 → [8,4,8]; remainder 4 is mn-aligned.
        assert self.norm(8, 20, 'N', 4) == [8, 4, 8]

    def test_mn_zero_spec_uses_full_dim(self):
        # spec=0 means "one partition for the whole dim"; if the whole dim
        # is mn-aligned the result is [total].
        assert self.norm(0, 16, 'N', 2) == [16]

    def test_mn_total_not_aligned_falls_back_to_single(self):
        # total=23, mn=2: no multi-partition split is mn-valid → [total].
        # Caller's vgpr-budget loop sees a single big partition and moves on.
        assert self.norm(0, 23, 'N', 2) == [23]
        assert self.norm(8, 23, 'N', 2) == [23]
        assert self.norm(22, 23, 'N', 2) == [23]

    def test_mn_spec_equals_total(self):
        assert self.norm(16, 16, 'N', 2) == [16]
        # Aligned total but spec snaps down → multi-partition.
        assert self.norm(16, 16, 'N', 4) == [16]

    def test_mn_one_full_with_aligned_remainder(self):
        # num_full=1 path with mn-aligned remainder.
        assert self.norm(8, 12, 'N', 4) == [8, 4]
        assert self.norm(12, 16, 'N', 4) == [12, 4]

    def test_mn_explicit_list_aligned_passes(self):
        assert self.norm([8, 4, 8], 20, 'N', 2) == [8, 4, 8]
        assert self.norm([4, 4, 4, 4], 16, 'N', 4) == [4, 4, 4, 4]

    def test_mn_explicit_list_unaligned_raises(self):
        # The [22,1] FP4 case from standalone: 1 is not a multiple of mn=2.
        with pytest.raises(AssertionError, match="multiples of mn=2"):
            self.norm([22, 1], 23, 'N', 2)

    def test_mn_explicit_list_one_unaligned_element_raises(self):
        with pytest.raises(AssertionError, match="multiples of mn=4"):
            self.norm([8, 6, 8], 22, 'N', 4)

    def test_mn_monotonic_candidate_sweep(self):
        # Mirrors how Kernel.py iterates get_partition_candidates from large
        # to small. Snapping DOWN must keep the post-normalization sequence
        # non-increasing in partition size, so the vgpr-budget loop never
        # re-tries a size it already rejected.
        candidates = [16, 8, 7, 6, 5, 4, 3, 2, 1]
        normalized_first = [self.norm(s, 16, 'N', 2)[0] for s in candidates]
        # Each step is <= the previous step.
        for prev, cur in zip(normalized_first, normalized_first[1:]):
            assert cur <= prev, f"non-monotonic: {normalized_first}"


# ══════════════════════════════════════════════════════════════
# getNumVgpr
# ══════════════════════════════════════════════════════════════

class TestGetNumVgpr:

    def test_no_scale(self):
        """A+B total without scale."""
        kernel = create_kernel(256, 256)
        tiA = makeTileInfo('A', kernel)
        tiB = makeTileInfo('B', kernel)
        cfg = make_cfg_bf16(256, 256)
        sched = LogicalScheduler(cfg)
        sched.build()

        total = sched.getNumVgpr(tiA, tiB)
        assert total > 0

        import math
        vgpr_per_A = int(math.ceil(tiA.mmaTileRegCount * cfg.lrA.k * cfg.lrA.mn))
        vgpr_per_B = int(math.ceil(tiB.mmaTileRegCount * cfg.lrB.k * cfg.lrB.mn))
        expected = sched.tile_peaks['A'] * vgpr_per_A + sched.tile_peaks['B'] * vgpr_per_B
        assert total == expected

    def test_with_scale(self):
        """Including SA+SB increases total."""
        kernel = create_kernel(256, 256, fp4=True)
        tiA = makeTileInfo('A', kernel)
        tiB = makeTileInfo('B', kernel)
        scaleTiA = makeTileInfo('MXSA', kernel)
        scaleTiB = makeTileInfo('MXSB', kernel)

        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.build()

        total = sched.getNumVgpr(tiA, tiB, scaleTiA, scaleTiB)
        total_no_scale = sched.getNumVgpr(tiA, tiB)
        assert total > total_no_scale

    def test_decreases_with_partitions(self):
        """More partitions → fewer VGPRs."""
        kernel = create_kernel(256, 256, fp4=True)
        tiA = makeTileInfo('A', kernel)
        tiB = makeTileInfo('B', kernel)
        scaleTiA = makeTileInfo('MXSA', kernel)
        scaleTiB = makeTileInfo('MXSB', kernel)

        def _build_and_count(partSizeM, partSizeN):
            cfg = make_cfg_256x256_fp4(partSizeM=partSizeM, partSizeN=partSizeN)
            sched = LogicalScheduler(cfg)
            sched.build()
            return sched.getNumVgpr(tiA, tiB, scaleTiA, scaleTiB)

        vgpr_1x1 = _build_and_count(0, 0)
        vgpr_1x2 = _build_and_count(0, 4)
        vgpr_1x4 = _build_and_count(0, 2)

        assert vgpr_1x1 >= vgpr_1x2
        assert vgpr_1x2 >= vgpr_1x4


# ══════════════════════════════════════════════════════════════
# Integration tests
# ══════════════════════════════════════════════════════════════

class TestIntegration:

    def test_populate_instructions_256x256_fp4(self):
        """Full pipeline: emit → populate_instructions → instructionSchedule."""
        from Tensile.Components.Subtile.InstructionScheduler import instructionSchedule

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

            assert len(sched._emitted_per_unroll) == sched.unroll_factor
            assert len(sched._ngll_per_unroll) == sched.unroll_factor
            assert len(sched._nll_per_unroll) == sched.unroll_factor

            # All modules have instructions
            for pi, partition_emitted in enumerate(sched._emitted_per_unroll[0]):
                for k, emitted in enumerate(partition_emitted):
                    for em in emitted:
                        assert len(em.instructions) > 0, \
                            f"P{pi} k={k} [{em.moduleId}] {em.opType}: no instructions"

            # MFMA/LR VGPR disjoint
            for pi, slots in enumerate(sched._partitions):
                for slot in slots:
                    if slot.mfma and slot.lrs:
                        for lr in slot.lrs:
                            if lr.vgpr_tile_map and lr.tensor in ('A', 'B'):
                                mfma_map = slot.mfma.vgpr_tile_maps.get(lr.tensor, [])
                                if mfma_map:
                                    assert set(mfma_map[0].values()).isdisjoint(
                                        set(lr.vgpr_tile_map[0].values()))

            # instructionSchedule succeeds
            for pi, partition_emitted in enumerate(sched._emitted_per_unroll[0]):
                for k, emitted in enumerate(partition_emitted):
                    scheduled = instructionSchedule(emitted)
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
            asm = str(sched.emitMainAndExitLoops(writer, kernel)) \
                + str(sched.emitTailLoop(writer, kernel))

            assert "LoopBeginL:" in asm
            assert "SkipToNGLL:" in asm

            if uf > 1:
                for ui in range(uf):
                    assert f"MAINLOOP_C{ui}" in asm
                    assert f"NGLL_C{ui}" in asm
                    assert f"NLL_C{ui}" in asm
                assert "SkipToEnd:" in asm
                assert "SkipToNLL:" in asm

                def get_mfma_vgprs(emitted_3d):
                    vgprs = set()
                    for partition in emitted_3d:
                        for group in partition:
                            for em in group:
                                if em.opType == 'mfma':
                                    for inst in em.instructions:
                                        vgprs.add(str(inst))
                    return vgprs

                vgprs_0 = get_mfma_vgprs(sched._emitted_per_unroll[0])
                vgprs_1 = get_mfma_vgprs(sched._emitted_per_unroll[1])
                assert vgprs_0 != vgprs_1, \
                    "Per-unroll copies should differ in MFMA instructions"
            else:
                assert "MAINLOOP" in asm
                assert "NGLL" in asm
                assert "NLL" in asm

        finally:
            sched.deallocVgprTiles(writer)

    def test_emitMainAndExitLoops_pap_mx_inserts_preloop_skip_and_nll_hooks(self):
        """Subtile PAP is scheduler-owned and applies to MX PRELOOP/NLL paths."""
        kernel = create_kernel(256, 256, fp4=True)
        kernel.update({
            "UseSubtileImpl": True,
            "PrefetchAcrossPersistent": 1,
            "PrefetchGlobalRead": 2,
            "StreamK": 3,
        })
        writer, tiA, tiB, scaleTiA, scaleTiB, dTileInfo = make_writer_and_tileinfos(kernel, fp4=True)

        def _pap_hook(_kernel, _tPA, _tPB, _preloop_gr, skipBarrier=False):
            module = Module("Subtile PAP test hook")
            module.addComment0("Subtile PAP test hook")
            return module

        writer.prefetchAcrossPersistentSubtile = MagicMock(side_effect=_pap_hook)

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

            module = sched.emitMainAndExitLoops(writer, kernel)
            asm = str(module)

            assert "Subtile PAP: first PRELOOP GR already issued?" in asm
            assert "SubtilePAPPreloopFirstGRMerge" in asm
            assert "Subtile PAP test hook" in asm
            assert writer.prefetchAcrossPersistentSubtile.call_count == sched.unroll_factor
            assert all(call.kwargs["skipBarrier"] for call in writer.prefetchAcrossPersistentSubtile.call_args_list)

            pap_idx = asm.index("Subtile PAP test hook")
            nll_idx = asm.rfind("NLL_C", 0, pap_idx)
            assert nll_idx != -1
            wait_gr_idx = asm.rfind("Wait GR", nll_idx, pap_idx)
            barrier_idx = asm.rfind("Barrier", nll_idx, pap_idx)
            assert nll_idx < wait_gr_idx < barrier_idx < pap_idx
            mfma_idx = asm.find("MFMA C[", pap_idx)
            assert mfma_idx != -1
            assert pap_idx < mfma_idx
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

            def get_scale_vgprs(emitted_3d):
                sa_vgprs = set()
                sb_vgprs = set()
                for partition in emitted_3d:
                    for group in partition:
                        for em in group:
                            if em.opType == 'mfma':
                                for inst in em.instructions:
                                    if isinstance(inst, MXMFMAInstruction):
                                        sa_vgprs.add(str(inst.mxsa))
                                        sb_vgprs.add(str(inst.mxsb))
                return sa_vgprs, sb_vgprs

            nll_scales = []
            for ui in range(uf):
                nll_scales.append(get_scale_vgprs(sched._nll_per_unroll[ui]))

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
    parser.add_argument("--arch", choices=["gfx950", "gfx1250"], default="gfx950",
                        help="Target arch (default: gfx950). gfx1250 enables wave32 + TDM.")
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
                           miWaveGroup=list(waveGroup), arch=args.arch)
    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    scaleTiA = makeTileInfo('MXSA', kernel) if fp4 else None
    scaleTiB = makeTileInfo('MXSB', kernel) if fp4 else None

    # Mirror Kernel.py mainLoop GR granularity selection:
    #   - gfx950 (buffer load): mn=subtileShape[0] (doubled when loadRatioGR>1),
    #                            k=subtileShape[1]
    #   - gfx1250 (TDM):        one tensor_load_to_lds covers the full local
    #                            MMA grid -> mn=localMMATileGrid[0], k=[1]
    if args.arch == "gfx1250":
        grA = ReadGranularity(mn=tiA.localMMATileGrid[0], k=tiA.localMMATileGrid[1])
        grB = ReadGranularity(mn=tiB.localMMATileGrid[0], k=tiB.localMMATileGrid[1])
    else:
        grA = ReadGranularity(mn=1, k=2) if tiA.loadRatioGR <= 1.0 else ReadGranularity(mn=2, k=2)
        grB = ReadGranularity(mn=1, k=2) if tiB.loadRatioGR <= 1.0 else ReadGranularity(mn=2, k=2)

    # gfx1250 enables TDM (tensor_load_to_lds); GR doesn't pressure SIMD issue
    # slots, so bunch all GR atoms into partition 0 / subIterK 0. Mirrors
    # Kernel.py's grPlacement selection.
    grPlacement = (GRPlacementStrategy.BUNCHED if args.arch == "gfx1250"
                   else GRPlacementStrategy.SPREAD)
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
        grPlacement=grPlacement,
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

    print(f"Config: arch={args.arch}, MT={args.mt0}x{args.mt1}, DU={args.du}, dtype={args.dtype}, "
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
            ("MAINLOOP", sched._emitted_per_unroll[0]),
            ("NGLL",     sched._ngll_per_unroll[0]),
            ("NLL",      sched._nll_per_unroll[0]),
            ("TAILLOOP", sched._tailloop_emitted, False),
        ]
    else:
        loop_sections = [
            ("MAINLOOP", sched._emitted_per_unroll[0]),
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

    sys.exit(0)

    sched.build_preloop()
    preloop_output = sched.print_emit(sched._preloop_emitted)
    print(f"{'=' * 60}")
    print(f"  Preloop")
    print(f"{'=' * 60}")
    print(preloop_output.replace("MAINLOOP:", "PRELOOP:"))
    if interactive:
        input("Press Enter for next step...")

    sched.build_ngll()
    ngll_output = sched.print_emit(sched._ngll_emitted)
    print(f"{'=' * 60}")
    print(f"  NGLL")
    print(f"{'=' * 60}")
    print(ngll_output.replace("MAINLOOP:", "NGLL:"))
    if interactive:
        input("Press Enter for next step...")

    sched.build_nll()
    nll_output = sched.print_emit(sched._nll_emitted)
    print(f"{'=' * 60}")
    print(f"  NLL")
    print(f"{'=' * 60}")
    print(nll_output.replace("MAINLOOP:", "NLL:"))
    if interactive:
        input("Press Enter for next step...")

    writer, tiA, tiB, scaleTiA, scaleTiB, dTileInfo = make_writer_and_tileinfos(kernel, fp4=not use_bf16)

    sched.allocVgprTiles(writer, tiA, tiB,
                         scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)

    sched.populate_instructions(
        writer, kernel,
        tileInfoA=tiA, tileInfoB=tiB,
        dtileInfo=dTileInfo,
        scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
    )

    def _print_emitLoop(label, emitted_3d, schedule=True):
        module = sched._emitLoop(writer, kernel, label, emitted_3d, schedule=schedule)
        buf = io.StringIO()
        for inst in module.flatitems():
            buf.write(f"  {str(inst).rstrip()}\n")
        return buf.getvalue()

    for section in [
        ("PRELOOP",  sched._preloop_emitted, False),
        ("MAINLOOP", sched._emitted_per_unroll[0]),
        ("NGLL",     sched._ngll_per_unroll[0]),
        ("NLL",      sched._nll_per_unroll[0]),
    ]:
        label, emitted_3d = section[0], section[1]
        schedule = section[2] if len(section) > 2 else True
        print(f"{'=' * 60}")
        print(f"  {label} (emitLoop)")
        print(f"{'=' * 60}")
        print(_print_emitLoop(label, emitted_3d, schedule=schedule))
        if interactive:
            input("Press Enter for next step...")

    sched.deallocVgprTiles(writer)


# ══════════════════════════════════════════════════════════════
# MT iteration integer representation
# ══════════════════════════════════════════════════════════════

class TestFmtMt:
    def test_fmt_mt_values(self):
        assert fmt_mt(0) == "n"
        assert fmt_mt(1) == "n+1"
        assert fmt_mt(2) == "n+2"
        assert fmt_mt(3) == "n+3"


class TestMtIterationTypes:
    def test_mt_iteration_is_int_bf16(self):
        cfg = make_cfg_bf16()
        sched = LogicalScheduler(cfg)
        sched.place_GRs()
        for slots in sched._partitions:
            for slot in slots:
                for lr in slot.lrs:
                    assert isinstance(lr.mtIteration, int), \
                        f"LR {lr.tensor} mtIteration is {type(lr.mtIteration)}"
                for gr in slot.grs:
                    assert isinstance(gr.mtIteration, int), \
                        f"GR {gr.tensor} mtIteration is {type(gr.mtIteration)}"

    def test_mt_iteration_is_int_fp4(self):
        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.place_GRs()
        for slots in sched._partitions:
            for slot in slots:
                for lr in slot.lrs:
                    assert isinstance(lr.mtIteration, int), \
                        f"LR {lr.tensor} mtIteration is {type(lr.mtIteration)}"
                for gr in slot.grs:
                    assert isinstance(gr.mtIteration, int), \
                        f"GR {gr.tensor} mtIteration is {type(gr.mtIteration)}"


class TestPreloopMtIntegers:
    def test_preloop_uses_int_mt(self):
        cfg = make_cfg_bf16()
        sched = LogicalScheduler(cfg)
        sched.emit()
        preloop = sched.build_preloop()
        for partition_emitted in preloop:
            for emitted in partition_emitted:
                for em in emitted:
                    src = em.source
                    if isinstance(src, GRPlacement):
                        assert isinstance(src.mtIteration, int), \
                            f"Preloop GR {src.tensor} mtIteration is {type(src.mtIteration)}"
                    elif isinstance(src, LRPlacement):
                        assert isinstance(src.mtIteration, int), \
                            f"Preloop LR {src.tensor} mtIteration is {type(src.mtIteration)}"


class TestBuildNll:
    def test_nll_removes_expected_ops(self):
        cfg = make_cfg_bf16()
        sched = LogicalScheduler(cfg)
        sched.emit()
        nll = sched.build_nll()
        for partition_emitted in nll:
            for emitted in partition_emitted:
                for em in emitted:
                    src = em.source
                    assert em.opType != 'gr', "NLL should have no GR ops"
                    # PGR=2 (default): gr_inc dropped; lr_inc kept to swap LR
                    # base for the next MT iteration on tail entry.
                    assert em.opType != 'gr_inc', "NLL should have no gr_inc ops"
                    if em.opType == 'lr' and isinstance(src, LRPlacement):
                        assert src.mtIteration == 0, \
                            f"NLL should only have LR(n=0), got mt={src.mtIteration}"
                    if em.opType == 'wait_gr':
                        from Tensile.Components.Subtile.LogicalScheduler import WaitGROp
                        if isinstance(src, WaitGROp) and src.wait_gr_counts:
                            cnts = src.wait_gr_counts
                            assert cnts.A == 0 and cnts.B == 0 and cnts.SA == 0 and cnts.SB == 0, \
                                f"NLL WaitGR should have zeroed counts, got {cnts}"


# ══════════════════════════════════════════════════════════════
# PGR=0 tests
# ══════════════════════════════════════════════════════════════

class TestPGR0Config:

    def test_pgr0_requires_single_partition(self):
        with pytest.raises(AssertionError, match="pgr=0 requires numPartitions=1"):
            SchedulerConfig(
                numMFMATilesM=8, numMFMATilesN=8, numSubIterK=2,
                lrA=ReadGranularity(mn=1, k=1), lrB=ReadGranularity(mn=1, k=1),
                grA=ReadGranularity(mn=1, k=2), grB=ReadGranularity(mn=1, k=2),
                pgr=0, partitionSizeN=4,
            )


class TestPlaceLRs_PLR0:

    def test_bf16_plr0_structure(self):
        cfg = make_cfg_bf16_pgr0()
        sched = LogicalScheduler(cfg)
        partitions = sched.place_LRs()

        assert len(partitions) == 1
        slots = partitions[0]
        numK = cfg.numSubIterK

        for k in range(numK):
            slot = slots[k]
            assert slot.mfma is not None
            assert slot.mfma.subIterK == k
            for lr in slot.lrs:
                assert lr.mtIteration == 0
                assert lr.tiles.subIterK_start == k
                assert lr.tiles.subIterK_end == k + 1

    def test_bf16_plr0_tensors(self):
        cfg = make_cfg_bf16_pgr0()
        sched = LogicalScheduler(cfg)
        partitions = sched.place_LRs()
        slots = partitions[0]

        for k in range(cfg.numSubIterK):
            _assert_slot_lrs(slots[k], ['A', 'B'])

    def test_bf16_plr0_print_lr(self):
        cfg = make_cfg_bf16_pgr0()
        sched = LogicalScheduler(cfg)
        sched.place_LRs()
        output = sched.print_lr()

        assert "MT n," in output
        assert "MT n+1" not in output


class TestPlaceGRs_PGR0:

    def test_bf16_pgr0_all_in_subIterK0(self):
        cfg = make_cfg_bf16_pgr0()
        sched = LogicalScheduler(cfg)
        sched.place_LRs()
        sched.place_GRs()
        slots = sched._partitions[0]

        for gr in slots[0].grs:
            assert gr.subIterK_slot == 0
            assert gr.mtIteration == 0

        for k in range(1, cfg.numSubIterK):
            assert len(slots[k].grs) == 0

    def test_bf16_pgr0_covers_full_k(self):
        cfg = make_cfg_bf16_pgr0()
        sched = LogicalScheduler(cfg)
        sched.place_LRs()
        sched.place_GRs()
        slots = sched._partitions[0]

        gr_a = [gr for gr in slots[0].grs if gr.tensor == 'A']
        all_k = set()
        for gr in gr_a:
            for k in range(gr.tiles.subIterK_start, gr.tiles.subIterK_end):
                all_k.add(k)
        assert all_k == set(range(cfg.numSubIterK))

    def test_bf16_pgr0_print_gr(self):
        cfg = make_cfg_bf16_pgr0()
        sched = LogicalScheduler(cfg)
        sched.place_LRs()
        sched.place_GRs()
        output = sched.print_gr()

        assert "GR A (MT n," in output
        assert "GR B (MT n," in output
        assert "MT n+1" not in output
        assert "MT n+2" not in output


class TestAnnotateDeps_PGR0:

    def _build(self):
        cfg = make_cfg_bf16_pgr0()
        sched = LogicalScheduler(cfg)
        sched.place_LRs()
        sched.place_GRs()
        sched.annotate_deps()
        return cfg, sched

    def test_no_crash(self):
        self._build()

    def test_gr_deps_on_lr_mt_minus2(self):
        _, sched = self._build()
        slots = sched._partitions[0]
        for slot in slots:
            for gr in slot.grs:
                assert gr.deps, f"GR {gr.tensor} should have collision deps"
                for dep in gr.deps:
                    assert isinstance(dep.ref, LRPlacement)
                    assert dep.mt_offset == -2

    def test_mfma_deps_on_lr_mt0(self):
        _, sched = self._build()
        slots = sched._partitions[0]
        for slot in slots:
            if slot.mfma:
                for dep in slot.mfma.deps:
                    assert isinstance(dep.ref, LRPlacement)
                    assert dep.ref.mtIteration == 0

    def test_lr_deps_on_gr_mt0(self):
        _, sched = self._build()
        slots = sched._partitions[0]
        for slot in slots:
            for lr in slot.lrs:
                for dep in lr.deps:
                    assert isinstance(dep.ref, GRPlacement)
                    assert dep.ref.mtIteration == 0


class TestEmit_PGR0:

    def test_emit_succeeds(self):
        cfg = make_cfg_bf16_pgr0()
        sched = LogicalScheduler(cfg)
        sched.emit()
        assert sched._emitted is not None
        assert len(sched._emitted) == 1
        assert len(sched._emitted[0]) == cfg.numSubIterK

    def test_gr_lr_inc_as_postOps(self):
        cfg = make_cfg_bf16_pgr0()
        sched = LogicalScheduler(cfg)
        sched.emit()
        # Each tensor's INC should be on its own placement (not merged)
        slot0 = sched._partitions[-1][0]  # subIterK=0
        slot1 = sched._partitions[-1][-1]  # subIterK=1

        # lr_inc(A) on LR A, lr_inc(B) on LR B
        lr_a = [lr for lr in slot1.lrs if lr.tensor == 'A'][0]
        lr_b = [lr for lr in slot1.lrs if lr.tensor == 'B'][0]
        assert any(op.kind == 'lr_inc' and op.tensor == 'A' for op in lr_a.postOps)
        assert any(op.kind == 'lr_inc' and op.tensor == 'B' for op in lr_b.postOps)

        # gr_inc(A) on GR A, gr_inc(B) on GR B
        gr_a = [gr for gr in slot0.grs if gr.tensor == 'A'][0]
        gr_b = [gr for gr in slot0.grs if gr.tensor == 'B'][0]
        assert any(op.kind == 'gr_inc' and op.tensor == 'A' for op in gr_a.postOps)
        assert any(op.kind == 'gr_inc' and op.tensor == 'B' for op in gr_b.postOps)


class TestBuildLoopVariants_PGR0:

    def test_preloop_empty(self):
        cfg = make_cfg_bf16_pgr0()
        sched = LogicalScheduler(cfg)
        sched.emit()
        preloop = sched.build_preloop()
        assert preloop == [[[]]]

    def test_ngll_empty(self):
        cfg = make_cfg_bf16_pgr0()
        sched = LogicalScheduler(cfg)
        sched.emit()
        ngll = sched.build_ngll()
        assert ngll == [[[]]]

    def test_nll_empty(self):
        cfg = make_cfg_bf16_pgr0()
        sched = LogicalScheduler(cfg)
        sched.emit()
        nll = sched.build_nll()
        assert nll == [[[]]]


class TestBuildTailloopPGR0:

    def test_assembly_fp4_256x256(self):
        """Full pipeline: display generated assembly for FP4 tailloop."""
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

            print("FP4 256x256 tailloop assembly:")
            module = sched._emitLoop(writer, kernel, "TAILLOOP",
                                      sched._tailloop_emitted,
                                      schedule=False)
            for inst in module.flatitems():
                print(f"  {str(inst).rstrip()}")

        finally:
            sched.deallocVgprTiles(writer)

    def test_assembly_bf16_320x320_1x5(self):
        """Full pipeline: BF16 320x320 tailloop with 1x5 partitions."""
        kernel = create_kernel(320, 320, fp4=False)
        writer, tiA, tiB, scaleTiA, scaleTiB, dTileInfo = make_writer_and_tileinfos(kernel, fp4=False)

        cfg = SchedulerConfig(
            numMFMATilesM=tiA.localMMATileGrid[0],
            numMFMATilesN=tiB.localMMATileGrid[0],
            numSubIterK=tiA.localMMATileGrid[1],
            lrA=ReadGranularity(mn=1, k=1),
            lrB=ReadGranularity(mn=1, k=1),
            grA=ReadGranularity(mn=1, k=2),
            grB=ReadGranularity(mn=1, k=2),
            partitionSizeN=2,
            pgr=2,
        )
        sched = LogicalScheduler(cfg)
        sched.build()
        sched.allocVgprTiles(writer, tiA, tiB)

        try:
            sched.populate_instructions(
                writer, kernel,
                tileInfoA=tiA, tileInfoB=tiB,
                dtileInfo=dTileInfo,
            )

            print("BF16 320x320 1x5 tailloop assembly:")
            module = sched._emitLoop(writer, kernel, "TAILLOOP",
                                      sched._tailloop_emitted,
                                      schedule=False)
            for inst in module.flatitems():
                print(f"  {str(inst).rstrip()}")

        finally:
            sched.deallocVgprTiles(writer)
