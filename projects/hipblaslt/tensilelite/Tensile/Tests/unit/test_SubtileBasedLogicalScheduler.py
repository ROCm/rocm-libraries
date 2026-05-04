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

from Tensile.Components.Subtile.Kernel import (
    TileInfo, AB_B16, AB_B4, MXSA_B4, MXSB_B4, CD_F32,
)
from Tensile.Components.Subtile.LogicalScheduler import (
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
    return mock


def create_kernel(MT0=256, MT1=256, fp4=False, depthU=None):
    mxblock = 32 if fp4 else 0
    bpe = 0.5 if fp4 else 2
    matrixInstK = 128 if fp4 else 32
    if depthU is None:
        depthU = 256 if fp4 else 64
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
        "MIWaveGroup": [2, 2],
        "WavefrontSize": 64,
        "SourceSwap": False,
        "MIArchVgpr": False,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "NonTemporalMXSA": 0,
        "NonTemporalMXSB": 0,
        "ProblemType": problemType,
    }
    if fp4:
        kernel["_DepthUMXSA"] = depthU // mxblock
        kernel["_DepthUMXSB"] = depthU // mxblock
    return kernel


def make_cfg_256x256_fp4(depthU=256, k_gran=1, numPartM=1, numPartN=1):
    """Build FP4 config with scale tensors. k_gran applies to LR A/B."""
    kernel = create_kernel(256, 256, fp4=True, depthU=depthU)
    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    scaleTiA = makeTileInfo('MXSA', kernel)
    scaleTiB = makeTileInfo('MXSB', kernel)
    return SchedulerConfig(
        numMFMATilesM=tiA.localMMATileGrid[0],
        numMFMATilesN=tiB.localMMATileGrid[0],
        numSubIterK=tiA.localMMATileGrid[1],
        lrA=ReadGranularity(mn=1, k=k_gran),
        lrB=ReadGranularity(mn=1, k=k_gran),
        grA=ReadGranularity(mn=1, k=2),
        grB=ReadGranularity(mn=1, k=2),
        lrSA=ReadGranularity(mn=2, k=2),
        lrSB=ReadGranularity(mn=2, k=2),
        grSA=ReadGranularity(mn=scaleTiA.localMMATileGrid[0], k=scaleTiA.localMMATileGrid[1]),
        grSB=ReadGranularity(mn=scaleTiB.localMMATileGrid[0], k=scaleTiB.localMMATileGrid[1]),
        numPartitionsM=numPartM,
        numPartitionsN=numPartN,
    )


def make_cfg_bf16(MT0=256, MT1=256, depthU=64, numPartM=1, numPartN=1):
    """Build BF16 config without scale tensors."""
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
        numPartitionsM=numPartM,
        numPartitionsN=numPartN,
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
    )
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
        cfg = make_cfg_256x256_fp4(numPartM=2, numPartN=2)
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
        cfg = make_cfg_256x256_fp4(depthU=512, numPartM=2, numPartN=2)
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
        cfg = make_cfg_256x256_fp4(k_gran=2, numPartM=2, numPartN=2)
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
        cfg = make_cfg_bf16(320, 320, numPartM=10, numPartN=1)
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
        """Validate no MFMA/LR vgprTileId overlap and unrolling continuity."""
        parts = sched._partitions
        num_iters = sched.unroll_factor

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
        cfg = make_cfg_256x256_fp4(depthU=512, numPartM=2, numPartN=2)
        sched = LogicalScheduler(cfg)
        sched.assign_vgpr_tiles()

        for pi in range(4):
            for slot in sched._partitions[pi]:
                assert len(slot.mfma.vgpr_tile_maps['A']) > 0

        assert not sched.needs_unrolling
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
        """256x256, DU512, FP4. GR k=2 → two k-chunks. grSA/SB k=4 → full MT."""
        cfg = make_cfg_256x256_fp4(depthU=512)
        sched = LogicalScheduler(cfg)
        slots = sched.place_GRs()

        _assert_slot_grs(slots[0], ['A'])
        _assert_gr(slots[0], 'A', 0, 2, 0, 8)
        _assert_slot_grs(slots[1], ['B'])
        _assert_gr(slots[1], 'B', 0, 2, 0, 8)
        _assert_slot_grs(slots[2], ['SA', 'SB', 'A'])
        _assert_gr(slots[2], 'A', 2, 4, 0, 6)
        _assert_slot_grs(slots[3], ['A', 'B'])
        _assert_gr(slots[3], 'A', 2, 4, 6, 8)
        _assert_gr(slots[3], 'B', 2, 4, 0, 8)

    def test_2x2_k1_DU256(self):
        """256x256, DU256, FP4, 2x2 partition. Cross-MT dedup removes n+1 duplicates."""
        cfg = make_cfg_256x256_fp4(numPartM=2, numPartN=2)
        sched = LogicalScheduler(cfg)
        slots = sched.place_GRs()
        parts = sched._partitions

        # P0: A n+1
        _assert_slot_grs(parts[0][0], ['A'], "P0 s0")
        _assert_gr(parts[0][0], 'A', 0, 2, 4, 6, mt=1)
        _assert_slot_grs(parts[0][1], ['A'], "P0 s1")
        _assert_gr(parts[0][1], 'A', 0, 2, 6, 8, mt=1)

        # P1: B n+1
        _assert_slot_grs(parts[1][0], ['B'], "P1 s0")
        _assert_gr(parts[1][0], 'B', 0, 2, 4, 6, mt=1)

        # P3: B n+2, SA n+2, SB n+2
        _assert_slot_grs(parts[3][1], ['B', 'SA', 'SB'], "P3 s1")
        _assert_gr(parts[3][1], 'SA', 0, 2, 0, 8, mt=2)
        _assert_gr(parts[3][1], 'SB', 0, 2, 0, 8, mt=2)

    def test_2x2_k1_DU512(self):
        """256x256, DU512, FP4, 2x2 partition. GR k=2 × 2 chunks + scale."""
        cfg = make_cfg_256x256_fp4(depthU=512, numPartM=2, numPartN=2)
        sched = LogicalScheduler(cfg)
        sched.place_GRs()
        parts = sched._partitions

        # P0: A n+1 across 4 slots
        for i, (k_s, k_e) in enumerate([(0,2),(0,2),(2,4),(2,4)]):
            _assert_slot_grs(parts[0][i], ['A'], f"P0 s{i}")
        _assert_gr(parts[0][0], 'A', 0, 2, 4, 6, mt=1)
        _assert_gr(parts[0][2], 'A', 2, 4, 4, 6, mt=1)

        # P3 s0: SA+SB n+2
        _assert_slot_grs(parts[3][0], ['SA', 'SB'], "P3 s0")
        _assert_gr(parts[3][0], 'SA', 0, 4, 0, 8, mt=2)

    def test_10x1_bf16(self):
        """320x320, BF16, 10x1 partition. No scales."""
        cfg = make_cfg_bf16(320, 320, numPartM=10, numPartN=1)
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
        cfg = make_cfg_256x256_fp4(depthU=512, numPartM=2, numPartN=2)
        sched = LogicalScheduler(cfg)
        sched.annotate_deps()
        parts = sched._partitions

        assert len(parts) == 4
        assert len(parts[0]) == 4

        # P0 MFMA(k=0): deps from P3
        mfma_p0_s0 = _dep_refs(parts[0][0].mfma)
        assert ('LR', 'A', 3, 3, -1) in mfma_p0_s0
        assert len(mfma_p0_s0) == 4

        # P3 MFMA(k=0): deps from earlier partitions
        mfma_p3_s0 = _dep_refs(parts[3][0].mfma)
        assert ('LR', 'A', 1, 2, 0) in mfma_p3_s0
        assert ('LR', 'SA', 1, 0, 0) in mfma_p3_s0


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
        assert _preop_kinds(lr_a1) == [('wait_gr', True, {'A': 8, 'B': 8, 'SA': 1, 'SB': 1})]

        # LR B @s1: wait_gr with has_sync
        lr_b1 = _get_lr(s1, 'B')
        assert _preop_kinds(lr_b1) == [('wait_gr', True, {'A': 8, 'B': 1, 'SA': 0, 'SB': 0})]

        # LR SA @s1
        lr_sa1 = _get_lr(s1, 'SA')
        assert _preop_kinds(lr_sa1) == [('wait_gr', True, {'A': 8, 'B': 8, 'SA': 0, 'SB': 0})]

        # LR SB @s1
        lr_sb1 = _get_lr(s1, 'SB')
        assert _preop_kinds(lr_sb1) == [('wait_gr', True, {'A': 8, 'B': 8, 'SA': 1, 'SB': 0})]

        # GR B @s1: dep removed
        gr_b1 = [gr for gr in s1.grs if gr.tensor == 'B'][0]
        assert _preop_kinds(gr_b1) == []
        assert len(gr_b1.deps) == 0

    def test_2x2_DU512(self):
        """256x256, DU512, FP4, 2x2 partition. Spot checks."""
        cfg = make_cfg_256x256_fp4(depthU=512, numPartM=2, numPartN=2)
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
        cfg = make_cfg_bf16(128, 128, numPartM=2, numPartN=2)
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
        assert lr_b1.preOps[0].wait_gr_counts.SA == 0

        # LR SA @s1: counts
        lr_sa1 = _get_lr(s1, 'SA')
        assert lr_sa1.preOps[0].wait_gr_counts.A == 8
        assert lr_sa1.preOps[0].wait_gr_counts.B == 8

        # LR A @s1: counts
        lr_a1 = _get_lr(s1, 'A')
        assert lr_a1.preOps[0].wait_gr_counts.A == 8
        assert lr_a1.preOps[0].wait_gr_counts.B == 8
        assert lr_a1.preOps[0].wait_gr_counts.SA == 1
        assert lr_a1.preOps[0].wait_gr_counts.SB == 1


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
        """NGLL removes GR(n+2) and gr_inc. LR and MFMA preserved."""
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
        # NGLL should not have gr_inc
        assert 'gr_inc' not in all_ops


class TestBuildNLL:

    def test_256x256_fp4(self):
        """NLL removes GRs, LR(n+1), increments. Only MFMA remains."""
        cfg = make_cfg_256x256_fp4()
        sched = LogicalScheduler(cfg)
        sched.build()
        sched.build_nll()

        nll = sched._nll_emitted
        assert len(nll) > 0

        all_ops = [em.opType for partition in nll
                   for group in partition for em in group]

        assert 'mfma' in all_ops
        # NLL should not have gr or gr_inc or lr_inc
        assert 'gr' not in all_ops
        assert 'gr_inc' not in all_ops
        assert 'lr_inc' not in all_ops


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
        """M==N: partitions N."""
        kernel = create_kernel(256, 256)
        tiA = makeTileInfo('A', kernel)
        tiB = makeTileInfo('B', kernel)
        candidates = SchedulerConfig.get_partition_candidates(tiA, tiB)
        assert candidates == [(1, 1), (1, 2), (1, 4), (1, 8)]

    def test_n_larger(self):
        """N > M: partitions N."""
        kernel = create_kernel(64, 256)
        tiA = makeTileInfo('A', kernel)
        tiB = makeTileInfo('B', kernel)
        candidates = SchedulerConfig.get_partition_candidates(tiA, tiB)
        assert candidates == [(1, 1), (1, 2), (1, 4), (1, 8)]

    def test_m_larger(self):
        """M > N: partitions M."""
        kernel = create_kernel(256, 64)
        tiA = makeTileInfo('A', kernel)
        tiB = makeTileInfo('B', kernel)
        candidates = SchedulerConfig.get_partition_candidates(tiA, tiB)
        assert candidates == [(1, 1), (2, 1), (4, 1), (8, 1)]

    def test_prime_dim(self):
        """Prime-sized dimension: only (1,1) and (prime,1)."""
        cfg_tiA = MagicMock()
        cfg_tiB = MagicMock()
        cfg_tiA.localMMATileGrid = [5, 2]
        cfg_tiB.localMMATileGrid = [3, 2]
        candidates = SchedulerConfig.get_partition_candidates(cfg_tiA, cfg_tiB)
        assert candidates == [(1, 1), (5, 1)]

    def test_composite(self):
        """Composite dimension with mixed prime factors."""
        cfg_tiA = MagicMock()
        cfg_tiB = MagicMock()
        cfg_tiA.localMMATileGrid = [2, 2]
        cfg_tiB.localMMATileGrid = [6, 2]
        candidates = SchedulerConfig.get_partition_candidates(cfg_tiA, cfg_tiB)
        assert candidates == [(1, 1), (1, 2), (1, 3), (1, 6)]


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

        def _build_and_count(numPartM, numPartN):
            cfg = make_cfg_256x256_fp4(numPartM=numPartM, numPartN=numPartN)
            sched = LogicalScheduler(cfg)
            sched.build()
            return sched.getNumVgpr(tiA, tiB, scaleTiA, scaleTiB)

        vgpr_1x1 = _build_and_count(1, 1)
        vgpr_1x2 = _build_and_count(1, 2)
        vgpr_1x4 = _build_and_count(1, 4)

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

    def test_emitAllLoops_256x256_fp4(self):
        """emitAllLoops: label structure and per-unroll VGPR differences."""
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
            module = sched.emitAllLoops(writer, kernel)
            asm = str(module)

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

# Tool to visualize the scheduling steps on a real kernel configuration. Run with --interactive to step through each phase.
# Also calls the instruction scheduler to verify the emitted modules are valid input and to show the final instruction counts.

if __name__ == "__main__":
    import sys
    import io

    use_bf16 = "--bf16" in sys.argv

    if use_bf16:
        kernel = create_kernel(384, 256, fp4=False, depthU=64)
        tiA = makeTileInfo('A', kernel)
        tiB = makeTileInfo('B', kernel)
        scaleTiA = None
        scaleTiB = None

        cfg = SchedulerConfig(
            numMFMATilesM=tiA.localMMATileGrid[0],
            numMFMATilesN=tiB.localMMATileGrid[0],
            numSubIterK=tiA.localMMATileGrid[1],
            lrA=ReadGranularity(mn=1, k=1),
            lrB=ReadGranularity(mn=1, k=1),
            grA=ReadGranularity(mn=1, k=2),
            grB=ReadGranularity(mn=1, k=2),
            numPartitionsM=2,
            numPartitionsN=1,
        )
    else:
        kernel = create_kernel(128, 128, fp4=True, depthU=512)
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

    print(f"Config: numMFMATilesM={cfg.numMFMATilesM}, "
          f"numMFMATilesN={cfg.numMFMATilesN}, "
          f"numSubIterK={cfg.numSubIterK}, "
          f"hasScale={cfg.hasScale}")
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

    interactive = "--interactive" in sys.argv or "-i" in sys.argv

    for i, (title, run) in enumerate(steps):
        _, output = run()
        print(f"{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print(output)
        if interactive and i < len(steps) - 1:
            input("Press Enter for next step...")

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

    def _print_emitLoop(label, emitted_3d):
        module = sched._emitLoop(writer, kernel, label, emitted_3d)
        buf = io.StringIO()
        for inst in module.flatitems():
            buf.write(f"  {str(inst).rstrip()}\n")
        return buf.getvalue()

    for label, emitted_3d in [
        ("PRELOOP",  sched._preloop_emitted),
        ("MAINLOOP", sched._emitted_per_unroll[0]),
        ("NGLL",     sched._ngll_per_unroll[0]),
        ("NLL",      sched._nll_per_unroll[0]),
    ]:
        print(f"{'=' * 60}")
        print(f"  {label} (emitLoop)")
        print(f"{'=' * 60}")
        print(_print_emitLoop(label, emitted_3d))
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
                    assert em.opType != 'gr_inc', "NLL should have no gr_inc ops"
                    assert em.opType != 'lr_inc', "NLL should have no lr_inc ops"
                    if em.opType == 'lr' and isinstance(src, LRPlacement):
                        assert src.mtIteration == 0, \
                            f"NLL should only have LR(n=0), got mt={src.mtIteration}"
                    if em.opType == 'wait_gr':
                        from Tensile.Components.Subtile.LogicalScheduler import WaitGROp
                        if isinstance(src, WaitGROp) and src.wait_gr_counts:
                            cnts = src.wait_gr_counts
                            assert cnts.A == 0 and cnts.B == 0 and cnts.SA == 0 and cnts.SB == 0, \
                                f"NLL WaitGR should have zeroed counts, got {cnts}"
