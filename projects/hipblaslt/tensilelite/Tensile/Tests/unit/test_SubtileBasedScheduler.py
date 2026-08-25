import io
import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock
from Tensile.Components.SubtileBasedKernel import TileInfo
from Tensile.Components.SubtileBasedScheduler import (
    SubtileBasedScheduler, SchedulerConfig, PrefetchMode,
    MFMAOp, GROp, LROp, WaitGROp, WaitLROp, SyncOp, GR_INCOp, LR_INCOp,
)
from rocisa.code import Module, Label
from rocisa import rocIsa
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType

# Initialize rocIsa for gfx950
ri = rocIsa.getInstance()
if not ri.isInit():
    import shutil
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    ri.init((9, 5, 0), asmpath)
ri.setKernel((9, 5, 0), 64)

def _mock_dtype(num_bytes=2):
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    return mock

def create_kernel(MT0=256, MT1=256, fp4=False):
    mxblock = 32 if fp4 else 0
    bpe = 0.5 if fp4 else 2
    matrixInstK = 128 if fp4 else 32
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

def create_mock_writer(kernel):
    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr, False)
    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr, False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr, False)
    writer.states = SimpleNamespace(
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
    )
    # Allocate D tileInfo (same as KernelWriter line 3843)
    dTileInfo = TileInfo('D', kernel)
    dTileInfo.allocVgprTileRegisters(writer, kernel)
    writer.states.d = SimpleNamespace(tileInfo=dTileInfo)
    return writer

def create_writer_with_tiles(kernel, tiA, tiB, scaleTiA=None, scaleTiB=None):
    writer = create_mock_writer(kernel)
    writer.states.a = SimpleNamespace(tileInfo=tiA)
    writer.states.b = SimpleNamespace(tileInfo=tiB)
    writer.states.mxsa = SimpleNamespace(tileInfo=scaleTiA) if scaleTiA else SimpleNamespace()
    writer.states.mxsb = SimpleNamespace(tileInfo=scaleTiB) if scaleTiB else SimpleNamespace()
    tiA.allocOffsetRegisters(writer, kernel)
    tiB.allocOffsetRegisters(writer, kernel)
    if scaleTiA:
        scaleTiA.allocOffsetRegisters(writer, kernel)
    if scaleTiB:
        scaleTiB.allocOffsetRegisters(writer, kernel)
    # Named sgprs for GR_INC / LR_INC emit paths
    writer.sgprs = {}
    writer.sgprs["SrdA"] = writer.sgprPool.checkOutAligned(4, 4, "SrdA", preventOverflow=False)
    writer.sgprs["SrdB"] = writer.sgprPool.checkOutAligned(4, 4, "SrdB", preventOverflow=False)
    writer.sgprs["LocalWriteBaseAddrA"] = writer.sgprPool.checkOut(1, preventOverflow=False)
    writer.sgprs["LocalWriteBaseAddrB"] = writer.sgprPool.checkOut(1, preventOverflow=False)
    writer.sgprs["SwapA"] = writer.sgprPool.checkOut(1, preventOverflow=False)
    writer.sgprs["SwapB"] = writer.sgprPool.checkOut(1, preventOverflow=False)
    writer.ldsTotalSize = 0
    if scaleTiA and scaleTiB:
        writer.sgprs["SrdMXSA"] = writer.sgprPool.checkOutAligned(4, 4, "SrdMXSA", preventOverflow=False)
        writer.sgprs["SrdMXSB"] = writer.sgprPool.checkOutAligned(4, 4, "SrdMXSB", preventOverflow=False)
        writer.sgprs["LocalWriteBaseAddrMXSA"] = writer.sgprPool.checkOut(1, preventOverflow=False)
        writer.sgprs["LocalWriteBaseAddrMXSB"] = writer.sgprPool.checkOut(1, preventOverflow=False)
        writer.sgprs["SwapMXSA"] = writer.sgprPool.checkOut(1, preventOverflow=False)
        writer.sgprs["SwapMXSB"] = writer.sgprPool.checkOut(1, preventOverflow=False)
        writer.sgprs["StridesMXSA"] = writer.sgprPool.checkOut(1, preventOverflow=False)
        writer.sgprs["StridesMXSB"] = writer.sgprPool.checkOut(1, preventOverflow=False)
    return writer


def test_PGR2_64_64_1x1():
    MT0=MT1=64
    kernel = create_kernel(MT0,MT1)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    # 2x2 partition grid
    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]

    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH)
    s = SubtileBasedScheduler(tiA, tiB, cfg)

    assert len(s.preloopSteps)  > 0
    assert len(s.mainloopSteps) > 0
    assert len(s.ngllSteps)     > 0
    assert len(s.nllSteps)      > 0

    writer = create_writer_with_tiles(kernel, tiA, tiB)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s.printSchedule(showVgpr=True, showDeps=True, showSubtiles=True)
    actual = buf.getvalue()

    expected = """\
SubtileGridA=2, SubtileGridB=2
Partition grid: 1 x 1
Partition size: 2 x 2
Prefetch: HALF_PREFETCH
Reuse: ACROSS_PARTITIONS
totalVGPRTiles: 8 (32 VGPRs)
totalScaleVGPRTiles: 0
hasScale: False

Ordering grid (COLUMN_MAJOR):
   0

PRELOOP:
  GR (MT 0):  A: [0, 1]  B: [0, 1]
  GR_INC
  WAIT_GR (MT 0) A: [0, 1]  B: [0, 1] — inflight SubtileLoads A=0 B=0 scaleA=0 scaleB=0
  SYNC
  LR (MT 0, subIterK 0) A: [0, 1]  B: [0, 1]
    - LOAD  A: {0: 0, 1: 1}  B: {0: 2, 1: 3}
  WAIT_LR
  SKIP_IF_LE(1, NLL)
  GR (MT 1):  A: [0, 1]  B: [0, 1]
  GR_INC
  SKIP_IF_LE(2, NGLL)

MAINLOOP:
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(0, 0), (0, 1), (1, 0), (1, 1)]
        - USING  A: {0: 0, 1: 1}  B: {0: 2, 1: 3}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: [0, 1]  B: [0, 1]
        - LOAD  A: {0: 4, 1: 5}  B: {0: 6, 1: 7}
        before: [none]  after: [WaitLROp]
      GR (MT n+2):  A: [0]  B: [0]
        before: [LR(MT n, sik 1), WaitLROp, SyncOp]  after: [none]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(0, 0), (0, 1), (1, 0), (1, 1)]
        - USING  A: {0: 4, 1: 5}  B: {0: 6, 1: 7}
        before: [none]  after: [none]
      LR (MT n+1, subIterK 0) A: [0, 1]  B: [0, 1]
        - LOAD  A: {0: 0, 1: 1}  B: {0: 2, 1: 3}
        before: [WaitGROp(A=1 B=1 SA=0 SB=0), SyncOp, LR_INCOp]  after: [WaitLROp]
      GR (MT n+2):  A: [1]  B: [1]
        before: [none]  after: [GR_INCOp]
"""

    assert expected in actual


def test_PGR2_64_64_1x1_fp4():
    kernel = create_kernel(64, 64, fp4=True)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    scaleTiA = TileInfo('MXSA', kernel)
    scaleTiB = TileInfo('MXSB', kernel)
    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]

    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH)
    s = SubtileBasedScheduler(tiA, tiB, cfg,
                              scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)

    assert s.totalVGPRTiles == 8
    assert s.totalScaleVGPRTiles == 2
    assert s.hasScale

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s.printSchedule(showVgpr=True, showDeps=True)
    actual = buf.getvalue()

    expected = """\
MAINLOOP:
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - USING  A: {0: 0, 1: 1}  B: {0: 2, 1: 3}
        - SCALE  A: {0: 0}  B: {0: 1}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: [0, 1]  B: [0, 1]
        - LOAD  A: {0: 4, 1: 5}  B: {0: 6, 1: 7}
        before: [none]  after: [WaitLROp]
      GR (MT n+2):  A: [0]  B: [0]
        before: [LR(MT n, sik 1), WaitLROp, SyncOp]  after: [none]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - USING  A: {0: 4, 1: 5}  B: {0: 6, 1: 7}
        - SCALE  A: {0: 0}  B: {0: 1}
        before: [none]  after: [none]
      LR (MT n+1, subIterK 0) A: [0, 1]  B: [0, 1]
        - LOAD  A: {0: 0, 1: 1}  B: {0: 2, 1: 3}
        - SCALE  A: {0: 0}  B: {0: 1}
        before: [WaitGROp(A=1 B=1 SA=1 SB=1), SyncOp, LR_INCOp]  after: [WaitLROp]
      GR (MT n+2):  A: [1]  B: [1]
        before: [none]  after: [GR_INCOp]
"""

    assert expected in actual


def test_PGR2_64_64_2x2():
    MT0=MT1=64
    kernel = create_kernel(MT0,MT1)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    # 2x2 partition grid
    lsgA = tiA.localSubtileGrid[0]//2
    lsgB = tiB.localSubtileGrid[0]//2

    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH)
    s = SubtileBasedScheduler(tiA, tiB, cfg)

    assert len(s.preloopSteps)  > 0
    assert len(s.mainloopSteps) > 0
    assert len(s.ngllSteps)     > 0
    assert len(s.nllSteps)      > 0

    writer = create_writer_with_tiles(kernel, tiA, tiB)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s.printSchedule(showVgpr=True, showDeps=True, showSubtiles=True)
    actual = buf.getvalue()

    expected = """\
SubtileGridA=2, SubtileGridB=2
Partition grid: 2 x 2
Partition size: 1 x 1
Prefetch: HALF_PREFETCH
Reuse: ACROSS_PARTITIONS
totalVGPRTiles: 8 (32 VGPRs)
totalScaleVGPRTiles: 0
hasScale: False

Ordering grid (COLUMN_MAJOR):
   0   2
   1   3

PRELOOP:
  GR (MT 0):  A: [0]  B: [0]
  GR (MT 0):  A: [1]  B: []
  GR (MT 0):  A: []  B: [1]
  GR_INC
  WAIT_GR (MT 0) A: [0, 1]  B: [0, 1] — inflight SubtileLoads A=0 B=0 scaleA=0 scaleB=0
  SYNC
  LR (MT 0, subIterK 0) A: [0]  B: [0]
    - LOAD  A: {0: 0}  B: {0: 1}
  WAIT_LR
  SKIP_IF_LE(1, NLL)
  GR (MT 1):  A: [0]  B: [0]
  SKIP_IF_LE(2, NGLL)

MAINLOOP:
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(0, 0)]
        - USING  A: {0: 0}  B: {0: 1}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: [0]  B: [0]
        - LOAD  A: {0: 2}  B: {0: 3}
        before: [none]  after: [WaitLROp]
      GR (MT n+1):  A: [1]  B: []
        before: [none]  after: [none]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(0, 0)]
        - USING  A: {0: 2}  B: {0: 3}
        before: [none]  after: [none]
      LR (MT n, subIterK 0) A: [1]  B: []
        - LOAD  A: {1: 4}  B: {}
        before: [WaitGROp(A=2 B=2 SA=0 SB=0), SyncOp]  after: [WaitLROp]
  Partition 1:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(1, 0)]
        - USING  A: {1: 4}  B: {0: 1}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: [1]  B: []
        - LOAD  A: {1: 5}  B: {}
        before: [none]  after: [WaitLROp]
      GR (MT n+1):  A: []  B: [1]
        before: [none]  after: [GR_INCOp]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(1, 0)]
        - USING  A: {1: 5}  B: {0: 3}
        before: [none]  after: [none]
      LR (MT n, subIterK 0) A: []  B: [1]
        - LOAD  A: {}  B: {1: 6}
        before: [WaitGROp(A=2 B=2 SA=0 SB=0), SyncOp]  after: [WaitLROp]
  Partition 2:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(0, 1)]
        - USING  A: {0: 0}  B: {1: 6}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: []  B: [1]
        - LOAD  A: {}  B: {1: 7}
        before: [none]  after: [WaitLROp]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(0, 1)]
        - USING  A: {0: 2}  B: {1: 7}
        before: [none]  after: [none]
      LR (MT n, subIterK 0) A: []  B: []
        - LOAD  A: {}  B: {}
        before: [none]  after: [WaitLROp]
  Partition 3:
    subIterK=0:
      MFMAs (MT n, subIterK 0):
        - [(1, 1)]
        - USING  A: {1: 4}  B: {1: 6}
        before: [none]  after: [none]
      LR (MT n, subIterK 1) A: []  B: []
        - LOAD  A: {}  B: {}
        before: [none]  after: [WaitLROp]
      GR (MT n+2):  A: [0]  B: [0]
        before: [LR(MT n, sik 1), WaitLROp, SyncOp]  after: [none]
    subIterK=1:
      MFMAs (MT n, subIterK 1):
        - [(1, 1)]
        - USING  A: {1: 5}  B: {1: 7}
        before: [none]  after: [none]
      LR (MT n+1, subIterK 0) A: [0]  B: [0]
        - LOAD  A: {0: 0}  B: {0: 1}
        before: [GR(MT n+1), WaitGROp(A=2 B=2 SA=0 SB=0), SyncOp, LR_INCOp]  after: [WaitLROp]
"""

    assert expected in actual


def _mod_sig(mod):
    """Compact signature of an AnnotatedModule: (opType, before_ops, after_ops)."""
    def _op_name(e):
        if e.module:
            op = e.module.op
            if isinstance(op, LROp):
                return f"LR(MT {op.mtIteration})"
            if isinstance(op, GROp):
                return f"GR(MT {op.mtIteration})"
            return type(op).__name__
        return type(e.op).__name__
    op = mod.op
    if isinstance(op, MFMAOp):
        name = f"MFMA(sik {op.subIterK})"
    elif isinstance(op, LROp):
        name = f"LR(MT {op.mtIteration}, sik {op.subIterK})"
    elif isinstance(op, GROp):
        name = f"GR(MT {op.mtIteration})"
    else:
        name = type(op).__name__
    before = [_op_name(e) for e in mod.before]
    after = [_op_name(e) for e in mod.after]
    return (name, before, after)


def _step_sig(steps):
    """Extract compact structural signature from loop steps."""
    result = []
    for pss in steps:
        for dus in pss.subIterKSteps:
            mods = [_mod_sig(m) for m in dus.modules]
            result.append((pss.partitionId, dus.subIterK, mods))
    return result


def _create_1x1_scheduler():
    MT0 = MT1 = 64
    kernel = create_kernel(MT0, MT1)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]
    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH)
    return SubtileBasedScheduler(tiA, tiB, cfg)


def test_PGR2_64_64_1x1_ngll():
    """NGLL removes GR(n+2) and GR_INC; orphaned deps from GR(n+2) are dropped."""
    s = _create_1x1_scheduler()
    mainloop_sig = _step_sig(s.mainloopSteps)
    ngll_sig = _step_sig(s.ngllSteps)

    # Same number of partitions and subIterK steps
    assert len(ngll_sig) == len(mainloop_sig)

    for (pid, sik, mods) in ngll_sig:
        # No GR(n+2) modules
        assert not any("GR(MT n+2)" in name for name, _, _ in mods), \
            f"NGLL partition {pid} sik {sik} should not have GR(n+2)"
        # No GR_INCOp in any after deps
        for name, before, after in mods:
            assert "GR_INCOp" not in after, \
                f"NGLL partition {pid} sik {sik} {name} should not have GR_INCOp in after"
        # No SyncOp orphaned from removed GR(n+2)
        for name, before, after in mods:
            assert "SyncOp" not in after, \
                f"NGLL partition {pid} sik {sik} {name} should not have orphaned SyncOp in after"


def test_PGR2_64_64_1x1_nll():
    """NLL removes all GR, LR(n+1), GR_INC, LR_INC, WaitGR(n+1) and paired SyncOps."""
    s = _create_1x1_scheduler()
    nll_sig = _step_sig(s.nllSteps)

    for (pid, sik, mods) in nll_sig:
        for name, before, after in mods:
            # No GR modules
            assert not name.startswith("GR("), \
                f"NLL partition {pid} sik {sik} should not have {name}"
            # No LR(n+1) modules
            assert "MT n+1" not in name, \
                f"NLL partition {pid} sik {sik} should not have {name}"
            # No GR_INC or LR_INC in deps
            assert "GR_INCOp" not in before and "GR_INCOp" not in after, \
                f"NLL {name} should not have GR_INCOp"
            assert "LR_INCOp" not in before, \
                f"NLL {name} should not have LR_INCOp in before"

    # subIterK=0: should still have MFMA + LR(MT n)
    _, _, mods_sik0 = nll_sig[0]
    op_names = [name for name, _, _ in mods_sik0]
    assert any("MFMA" in n for n in op_names)
    assert any("LR(MT n" in n for n in op_names)

    # subIterK=1: should have MFMA only (LR(n+1) removed, no GR)
    _, _, mods_sik1 = nll_sig[1]
    op_names = [name for name, _, _ in mods_sik1]
    assert any("MFMA" in n for n in op_names)
    assert not any(n.startswith("LR") for n in op_names), \
        f"NLL sik=1 should not have LR, got {op_names}"


def test_PGR2_64_64_1x1_emitted_modules_links(verbose=False):
    MT0 = MT1 = 64
    kernel = create_kernel(MT0, MT1)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]

    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH)
    s = SubtileBasedScheduler(tiA, tiB, cfg)
    writer = create_writer_with_tiles(kernel, tiA, tiB)

    s.allocVgprTiles(writer)
    try:
        dtileInfo = writer.states.d.tileInfo
        pss = s.mainloopSteps[0]
        emitted0 = s._buildEmittedModules(writer, kernel, pss.subIterKSteps[0].modules, dtileInfo)
        emitted1 = s._buildEmittedModules(writer, kernel, pss.subIterKSteps[1].modules, dtileInfo)
    finally:
        s.deallocVgprTiles(writer)

    sig = lambda ems: [(em.moduleId, em.opType, len(em.instructions), em.before) for em in ems]

    assert sig(emitted0) == [
        (0, "mfma", 4, None),
        (1, "lr", 4, None),
        (2, "gr", 4, 4),
        (3, "wait_lr", 1, 1),
        (4, "sync", 1, 3),
    ]
    assert sig(emitted1) == [
        (0, "mfma", 4, None),
        (1, "lr", 4, 5),
        (2, "gr", 4, None),
        (3, "wait_gr", 1, None),
        (4, "sync", 1, 3),
        (5, "lr_inc", 6, 4),
        (6, "wait_lr", 1, 1),
        (7, "gr_inc", 8, 2),
    ]

    if verbose:
        for label, emitted in [("subIterK=0", emitted0), ("subIterK=1", emitted1)]:
            print(f"  {label}:")
            for em in emitted:
                beforeStr = str(em.before) if em.before is not None else "-"
                print(f"    id={em.moduleId} {em.opType}: {len(em.instructions)} insts "
                      f"before=[{beforeStr}]")


def test_PGR2_256_256_1x1_extract_paths_from_before_deps():
    MT0 = MT1 = 256
    kernel = create_kernel(MT0, MT1)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]

    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH)
    s = SubtileBasedScheduler(tiA, tiB, cfg)
    writer = create_writer_with_tiles(kernel, tiA, tiB)

    s.allocVgprTiles(writer)
    try:
        dtileInfo = writer.states.d.tileInfo
        pss = s.mainloopSteps[0]
        dus0 = pss.subIterKSteps[0]
        dus1 = pss.subIterKSteps[1]

        emitted0 = s._buildEmittedModules(writer, kernel, dus0.modules, dtileInfo)
        emitted1 = s._buildEmittedModules(writer, kernel, dus1.modules, dtileInfo)
    finally:
        s.deallocVgprTiles(writer)

    sig0 = [(em.moduleId, em.opType, len(em.instructions), em.before) for em in emitted0]
    sig1 = [(em.moduleId, em.opType, len(em.instructions), em.before) for em in emitted1]

    assert sig0 == [
        (0, "mfma", 64, None),
        (1, "lr", 16, None),
        (2, "gr", 16, 4),
        (3, "wait_lr", 1, 1),
        (4, "sync", 1, 3),
    ]
    assert sig1 == [
        (0, "mfma", 64, None),
        (1, "lr", 16, 5),
        (2, "gr", 16, None),
        (3, "wait_gr", 1, None),
        (4, "sync", 1, 3),
        (5, "lr_inc", 6, 4),
        (6, "wait_lr", 1, 1),
        (7, "gr_inc", 8, 2),
    ]

    mfmaIdx0, pathOrders0 = SubtileBasedScheduler._extractPathsFromBeforeDeps(emitted0)
    mfmaIdx1, pathOrders1 = SubtileBasedScheduler._extractPathsFromBeforeDeps(emitted1)

    assert mfmaIdx0 == 0
    assert pathOrders0 == [[1, 3, 4, 2]]
    assert mfmaIdx1 == 0
    assert pathOrders1 == [[3, 4, 5, 1, 6], [2, 7]]


def _classify_inst(inst):
    """Classify an instruction into a single-char type tag."""
    from rocisa.instruction import GlobalReadInstruction, LocalReadInstruction, MFMAInstruction
    from Tensile.Components.SubtileBasedKernel import MXMFMAInstruction
    if isinstance(inst, (MFMAInstruction, MXMFMAInstruction)):
        return 'M'
    if isinstance(inst, LocalReadInstruction):
        return 'L'
    if isinstance(inst, GlobalReadInstruction):
        return 'G'
    return 'S'


def _get_scheduled_instructions(scheduler, writer, kernel, subIterK):
    """Build emitted modules for a subIterK and return the flat instruction list."""
    dtileInfo = writer.states.d.tileInfo
    pss = scheduler.mainloopSteps[0]
    dus = pss.subIterKSteps[subIterK]
    emitted = scheduler._buildEmittedModules(writer, kernel, dus.modules, dtileInfo)
    scheduled = SubtileBasedScheduler.instructionSchedule(emitted)
    return scheduled.flatitems()


def _get_scheduled_sequence(scheduler, writer, kernel, subIterK, scaleTiA=None, scaleTiB=None):
    """Build emitted modules for a subIterK and return the instruction-scheduled type sequence."""
    return ''.join(_classify_inst(i) for i in _get_scheduled_instructions(scheduler, writer, kernel, subIterK))

    """Compute scheduling quality metrics from a type-tagged sequence string.

    Returns (exposed, spacings) where:
      exposed  - number of instructions beyond 2 per MFMA slot (0 = ideal)
      spacings - list of MFMA-gap distances between consecutive buffer_loads
    """
    # Split into per-MFMA-slot buckets
    slots = []
    current = []
    for ch in seq:
        if ch == 'M':
            slots.append(current)
            current = []
        else:
            current.append(ch)
    slots.append(current)  # after last MFMA

    exposed = sum(max(0, len(s) - 2) for s in slots)

    # Buffer load spacing: distance in MFMA count between consecutive G's
    mfma_idx = 0
    gr_mfma_positions = []
    for ch in seq:
        if ch == 'M':
            mfma_idx += 1
        elif ch == 'G':
            gr_mfma_positions.append(mfma_idx)
    spacings = [gr_mfma_positions[i + 1] - gr_mfma_positions[i]
                for i in range(len(gr_mfma_positions) - 1)]
    return exposed, spacings

# help non-reg refactoring. To be replaced with a more relaxed test.
def test_PGR2_256_256_fp4_instruction_schedule_exact():
    """Exact regression test for the mainloop instruction schedule (fp4 256x256)."""
    kernel = create_kernel(256, 256, fp4=True)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    scaleTiA = TileInfo('MXSA', kernel)
    scaleTiB = TileInfo('MXSB', kernel)
    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]

    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH)
    s = SubtileBasedScheduler(tiA, tiB, cfg,
                              scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)
    writer = create_writer_with_tiles(kernel, tiA, tiB,
                                      scaleTiA=scaleTiA, scaleTiB=scaleTiB)
    s.allocVgprTiles(writer)
    try:
        seq0 = _get_scheduled_sequence(s, writer, kernel, 0)
        seq1 = _get_scheduled_sequence(s, writer, kernel, 1)
    finally:
        s.deallocVgprTiles(writer)

    # M=MFMA, L=LocalRead, G=GlobalRead(buffer_load), S=scalar ALU/wait/sync
    expected_sik0 = \
        "MLMLMLMLMLMLMLMLMLMLMLMLMLMLMLMLMMMMSSMSSMGSMSMMMGMSMMMGMSMMMGMSMMMGM" \
        "SMMMGMSMMMGMSMMMGMSMMMGMSMMMGMMMMMM"
    expected_sik1 = \
        "MSMGMSMMMMMGMSMMMMMMGMSMMMMMGMSMMMMMMGMSMSMSSMSSMSSMSSGMSSSMSLMLMLML" \
        "MLMGLMSLMLMLMLMLMLGMSLSMSLSMSLSMSLSMSLSMSLSMSLSMSLSMSLSMLMLMLMMMMSM"

    assert seq0 == expected_sik0, f"subIterK=0 mismatch:\n  got: {seq0}\n  exp: {expected_sik0}"
    assert seq1 == expected_sik1, f"subIterK=1 mismatch:\n  got: {seq1}\n  exp: {expected_sik1}"


def test_PGR2_256_256_fp4_vmcnt():
    """Verify vmcnt values in SWaitCnt instructions for 256x256 fp4 mainloop.

    The scheduler's post-pass sets vlcnt = initial_inflight + buffer_loads_before_wait.
    We verify that for each SWaitCnt with vlcnt >= 0, the value equals the number
    of buffer_load instructions that appear before it in the scheduled sequence.
    The initial inflight count (from prior subIterK GRs) is baked into the
    pre-adjustment vlcnt by emitWaitGR, so the final vlcnt must be at least
    the number of buffer_loads placed before the wait in this subIterK.
    """
    from rocisa.instruction import SWaitCnt, GlobalReadInstruction

    kernel = create_kernel(256, 256, fp4=True)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    scaleTiA = TileInfo('MXSA', kernel)
    scaleTiB = TileInfo('MXSB', kernel)
    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]

    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH)
    s = SubtileBasedScheduler(tiA, tiB, cfg,
                              scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)
    writer = create_writer_with_tiles(kernel, tiA, tiB,
                                      scaleTiA=scaleTiA, scaleTiB=scaleTiB)
    s.allocVgprTiles(writer)
    try:
        for sik in range(len(s.mainloopSteps[0].subIterKSteps)):
            insts = _get_scheduled_instructions(s, writer, kernel, sik)

            # Walk instructions: count buffer_loads before each SWaitCnt
            buf_loads_before = 0
            for inst in insts:
                if isinstance(inst, GlobalReadInstruction):
                    buf_loads_before += 1
                elif isinstance(inst, SWaitCnt) and inst.vlcnt >= 0:
                    # vlcnt must be >= buffer_loads placed before this wait
                    # (the difference is the initial inflight from prior GRs)
                    assert inst.vlcnt >= buf_loads_before, \
                        f"subIterK={sik}: SWaitCnt vlcnt={inst.vlcnt} < " \
                        f"buf_loads_before={buf_loads_before}"
    finally:
        s.deallocVgprTiles(writer)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp4", action="store_true", help="Enable FP4 path with MX scales")
    args = parser.parse_args()

    MT0=MT1=256
    kernel = create_kernel(MT0, MT1, fp4=args.fp4)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)

    scaleTiA = TileInfo('MXSA', kernel) if args.fp4 else None
    scaleTiB = TileInfo('MXSB', kernel) if args.fp4 else None

    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]

    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH)
    s = SubtileBasedScheduler(tiA, tiB, cfg,
                              scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)

    assert len(s.preloopSteps)  > 0
    assert len(s.mainloopSteps) > 0
    assert len(s.ngllSteps)     > 0
    assert len(s.nllSteps)      > 0

    writer = create_writer_with_tiles(kernel, tiA, tiB,
                                      scaleTiA=scaleTiA, scaleTiB=scaleTiB)

    print("=== DEFAULT ===")
    s.printSchedule()
    print("\n=== VGPR + DEPS ===")
    s.printSchedule(showVgpr=False, showDeps=True, showSubtiles=False)
    s.printSchedule()

    s.allocVgprTiles(writer)

    preloop  = s._emitLoop(writer, kernel, "PRELOOP", s.preloopSteps)
    mainloop = s._emitLoop(writer, kernel, "MAINLOOP", s.mainloopSteps)
    ngll = Module("NGLL")
    ngll.add(Label("SkipToNGLL", ""))
    ngll.add(s._emitLoop(writer, kernel, "NGLL", s.ngllSteps))
    nll = Module("NLL")
    nll.add(Label("SkipToNLL", ""))
    nll.add(s._emitLoop(writer, kernel, "NLL", s.nllSteps))
    print(mainloop)
    # kernel = create_kernel()
    # tiA = TileInfo('A', kernel)
    # tiB = TileInfo('B', kernel)
    # lsgA = tiA.localSubtileGrid[0]
    # lsgB = tiB.localSubtileGrid[0]

    # configs = [
    #     # (f"lsg {lsgA}x{lsgB}, group {lsgA}x{lsgB}, HALF_PREFETCH, ACROSS_PARTITIONS, COLUMN_MAJOR",
    #     #     SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.ACROSS_PARTITIONS, PartitionOrdering.COLUMN_MAJOR)),
    #     (f"lsg {lsgA}x{lsgB}, group {lsgA//2}x{lsgB//2}, HALF_PREFETCH, ACROSS_PARTITIONS, COLUMN_MAJOR",
    #         SchedulerConfig(lsgA//2, lsgB//2, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.ACROSS_PARTITIONS, PartitionOrdering.COLUMN_MAJOR)),
    # ]

    # for name, cfg in configs:
    #     print(f"=== {name} ===")
    #     s = SubtileBasedScheduler(tiA, tiB, cfg)
    #     s.printSchedule()
    #     writer = create_writer_with_tiles(kernel, tiA, tiB)
    #     # s.generateCode(writer, kernel)
    #     # print()
