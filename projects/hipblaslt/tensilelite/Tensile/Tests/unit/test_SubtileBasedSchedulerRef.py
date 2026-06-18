"""Non-regression tests for LogicalScheduler — exact output checks.

These tests capture the expected scheduling output at specific pipeline steps
to detect unintended regressions. 
"""

from Tensile.Components.Subtile.Kernel import (
    TileInfo, AB_B8, AB_B16, AB_B4, MXSA_B4, MXSB_B4, CD_F32,
)
from Tensile.Components.Subtile.LogicalScheduler import (
    GRPlacementStrategy,
    LogicalScheduler,
    ReadGranularity,
    SchedulerConfig,
)


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


def create_kernel(MT0=256, MT1=256, fp4=False, depthU=None, waveGroup=(2, 2)):
    from unittest.mock import MagicMock

    mxblock = 32 if fp4 else 0
    bpe = 0.5 if fp4 else 2
    matrixInstK = 128 if fp4 else 32
    if depthU is None:
        depthU = 256 if fp4 else 64
    dtype = MagicMock()
    dtype.numBytes.return_value = bpe
    problemType = {
        "DataTypeA": dtype,
        "DataTypeB": dtype,
        "ComputeDataType": MagicMock(**{"numBytes.return_value": 4}),
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
        "MIWaveGroup": list(waveGroup),
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


def make_256x256_bf16():
    kernel = create_kernel(256, 256, fp4=False, depthU=64)
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
    )


EXPECTED_EMIT_DEP_ORDER_256x256_BF16_1x1 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-7]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-7]
        [ 5] wait_lr    wait_lr
        [ 6] sync       sync
        [ 7] gr_inc     gr_inc(A)
        [ 3] gr         GR A (MT n+2, subIterK [0,1]) ids [0-7]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-7] , B : [0-7] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 5] wait_gr    wait_gr(A=8)
        [ 6] sync       sync
        [ 7] lr_inc     lr_inc(A)
        [ 8] lr_inc     lr_inc(B)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-7]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-7]
      path 1:
        [ 9] gr_inc     gr_inc(B)
        [ 3] gr         GR B (MT n+2, subIterK [0,1]) ids [0-7]
"""


def make_384x256_bf16():
    kernel = create_kernel(384, 256, fp4=False, depthU=64)
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
        partitionSizeM=6,
    )


EXPECTED_EMIT_DEP_ORDER_384x256_BF16_2x1 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-5] , B : [0-7] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-5]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-7]
      path 1:
        [ 3] gr         GR A (MT n+1, subIterK [0,1]) ids [6-10]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-5] , B : [0-7] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 5] wait_gr    wait_gr(A=11,B=8)
        [ 6] sync       sync
        [ 1] lr         LR A  (MT n, subIterK [0]) [6-11]
      path 1:
        [ 2] gr         GR A (MT n+1, subIterK [0,1]) ids [11-11]
        [ 7] sync       sync
        [ 8] gr_inc     gr_inc(A)
        [ 3] gr         GR A (MT n+2, subIterK [0,1]) ids [0-3]
  Partition 1:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [6-11] , B : [0-7] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [6-11]
      path 1:
        [ 2] gr         GR A (MT n+2, subIterK [0,1]) ids [4-5]
        [ 5] gr_inc     gr_inc(B)
        [ 3] gr         GR B (MT n+2, subIterK [0,1]) ids [0-2]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [6-11] , B : [0-7] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 5] wait_gr    wait_gr(A=12,B=3)
        [ 6] sync       sync
        [ 7] lr_inc     lr_inc(A)
        [ 8] lr_inc     lr_inc(B)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-5]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-7]
      path 1:
        [ 3] gr         GR B (MT n+2, subIterK [0,1]) ids [3-7]
"""


def test_384x256_bf16_partition_2x1():
    """Exact check of emit dependency order for 384x256 BF16, 2x1 partition."""
    cfg = make_384x256_bf16()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_384x256_BF16_2x1, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_384x256_BF16_2x1}\n"
        f"--- Actual ---\n{actual}"
    )


def test_256x256_bf16_partition_1x1():
    """Exact check of emit dependency order for 256x256 BF16, 1x1 partition."""
    cfg = make_256x256_bf16()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_256x256_BF16_1x1, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_256x256_BF16_1x1}\n"
        f"--- Actual ---\n{actual}"
    )


EXPECTED_EMIT_DEP_ORDER_256x256_BF16_GFX1250_TDM = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] <- [5]
      preMFMA path 0:
        [ 5] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-7]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-7]
        [ 6] wait_lr    wait_lr
        [ 7] sync       sync
        [ 8] gr_inc     gr_inc(A)
        [ 3] gr         GR A (MT n+2, subIterK [0,1]) ids [0-7]
        [ 9] gr_inc     gr_inc(B)
        [ 4] gr         GR B (MT n+2, subIterK [0,1]) ids [0-7]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-7] , B : [0-7] <- [3]
      preMFMA path 0:
        [ 3] wait_lr    wait_lr
      path 0:
        [ 4] wait_gr    wait_gr(A=8,B=8)
        [ 5] sync       sync
        [ 6] lr_inc     lr_inc(A)
        [ 7] lr_inc     lr_inc(B)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-7]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-7]
"""


def make_256x256_bf16_gfx1250_tdm():
    """gfx1250 TDM variant of make_256x256_bf16.

    Equivalent CLI: --arch gfx1250 --mt0 256 --mt1 256 --du 64 --dtype bf16
                    --pgr 2 --wg 2x2 --partition-size 0x0

    TDM enabled on both A and B (gfx1250). With TDM on, GR uses
    tensor_load_to_lds and does not need to be spread across subIterKs —
    GRPlacementStrategy.BUNCHED pins all GR atoms to partition 0 / subIterK 0.
    """
    kernel = create_kernel(256, 256, fp4=False, depthU=64)
    kernel["enableTDMA"] = True
    kernel["enableTDMB"] = True
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
        grPlacement=GRPlacementStrategy.BUNCHED,
    )


def test_256x256_bf16_gfx1250_tdm_partition_1x1():
    """Exact check of emit dep order for 256x256 BF16 gfx1250+TDM, 1x1 partition.

    With TDM, every GR atom is pinned to subIterK=0 (grAllInFirstSlot=True).
    The expected output below differs from the non-TDM case in that the GR B
    placement moves from subIterK=1 to subIterK=0.
    """
    cfg = make_256x256_bf16_gfx1250_tdm()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_256x256_BF16_GFX1250_TDM, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_256x256_BF16_GFX1250_TDM}\n"
        f"--- Actual ---\n{actual}"
    )


def make_320x320_bf16():
    kernel = create_kernel(320, 320, fp4=False, depthU=64)
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
        partitionSizeN=2,
    )


EXPECTED_EMIT_DEP_ORDER_320x320_BF16_1x5 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-9] , B : [0-1] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-9]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-1]
      path 1:
        [ 3] gr         GR B (MT n+1, subIterK [0,1]) ids [2-3]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-9] , B : [0-1] <- [3]
      preMFMA path 0:
        [ 3] wait_lr    wait_lr
      path 0:
        [ 4] wait_gr    wait_gr(A=10,B=10)
        [ 5] sync       sync
        [ 1] lr         LR B  (MT n, subIterK [0]) [2-3]
      path 1:
        [ 2] gr         GR B (MT n+1, subIterK [0,1]) ids [4-5]
  Partition 1:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-9] , B : [2-3] <- [3]
      preMFMA path 0:
        [ 3] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR B  (MT n, subIterK [1]) [2-3]
      path 1:
        [ 2] gr         GR B (MT n+1, subIterK [0,1]) ids [6-7]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-9] , B : [2-3] <- [3]
      preMFMA path 0:
        [ 3] wait_lr    wait_lr
      path 0:
        [ 4] wait_gr    wait_gr(A=10,B=12)
        [ 5] sync       sync
        [ 1] lr         LR B  (MT n, subIterK [0]) [4-5]
      path 1:
        [ 2] gr         GR B (MT n+1, subIterK [0,1]) ids [8-9]
  Partition 2:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-9] , B : [4-5] <- [3]
      preMFMA path 0:
        [ 3] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR B  (MT n, subIterK [1]) [4-5]
      path 1:
        [ 4] gr_inc     gr_inc(A)
        [ 2] gr         GR A (MT n+2, subIterK [0,1]) ids [0-1]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-9] , B : [4-5] <- [3]
      preMFMA path 0:
        [ 3] wait_lr    wait_lr
      path 0:
        [ 4] wait_gr    wait_gr(A=12,B=12)
        [ 5] sync       sync
        [ 1] lr         LR B  (MT n, subIterK [0]) [6-7]
      path 1:
        [ 2] gr         GR A (MT n+2, subIterK [0,1]) ids [2-3]
  Partition 3:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-9] , B : [6-7] <- [3]
      preMFMA path 0:
        [ 3] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR B  (MT n, subIterK [1]) [6-7]
      path 1:
        [ 2] gr         GR A (MT n+2, subIterK [0,1]) ids [4-5]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-9] , B : [6-7] <- [3]
      preMFMA path 0:
        [ 3] wait_lr    wait_lr
      path 0:
        [ 4] wait_gr    wait_gr(A=16,B=10)
        [ 5] sync       sync
        [ 1] lr         LR B  (MT n, subIterK [0]) [8-9]
      path 1:
        [ 2] gr         GR A (MT n+2, subIterK [0,1]) ids [6-7]
  Partition 4:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-9] , B : [8-9] <- [3]
      preMFMA path 0:
        [ 3] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR B  (MT n, subIterK [1]) [8-9]
      path 1:
        [ 2] gr         GR A (MT n+2, subIterK [0,1]) ids [8-9]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-9] , B : [8-9] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 5] wait_gr    wait_gr(A=10,B=8)
        [ 6] sync       sync
        [ 7] lr_inc     lr_inc(A)
        [ 8] lr_inc     lr_inc(B)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-9]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-1]
      path 1:
        [ 9] gr_inc     gr_inc(B)
        [ 3] gr         GR B (MT n+2, subIterK [0,1]) ids [0-1]
"""


def test_320x320_bf16_partition_1x5():
    """Exact check of emit dependency order for 320x320 BF16, 1x5 partition."""
    cfg = make_320x320_bf16()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_320x320_BF16_1x5, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_320x320_BF16_1x5}\n"
        f"--- Actual ---\n{actual}"
    )


def make_256x256_bf16_pgr0():
    kernel = create_kernel(256, 256, fp4=False, depthU=64)
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


EXPECTED_EMIT_DEP_ORDER_256x256_BF16_PGR0 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] <- [7]
      preMFMA path 0:
        [ 3] gr         GR A (MT n, subIterK [0,1]) ids [0-7]
        [ 5] gr_inc     gr_inc(A)
        [ 4] gr         GR B (MT n, subIterK [0,1]) ids [0-7]
        [ 6] gr_inc     gr_inc(B)
        [ 8] wait_gr    wait_gr(0)
        [ 9] sync       sync
        [ 1] lr         LR A  (MT n, subIterK [0]) [0-7]
        [ 2] lr         LR B  (MT n, subIterK [0]) [0-7]
        [ 7] wait_lr    wait_lr
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-7] , B : [0-7] <- [5]
      preMFMA path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-7]
        [ 3] lr_inc     lr_inc(A)
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-7]
        [ 4] lr_inc     lr_inc(B)
        [ 5] wait_lr    wait_lr
"""


def test_256x256_bf16_pgr0():
    """Exact check of emit dependency order for 256x256 BF16, PGR0."""
    cfg = make_256x256_bf16_pgr0()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_256x256_BF16_PGR0, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_256x256_BF16_PGR0}\n"
        f"--- Actual ---\n{actual}"
    )


def make_256x256_bf16_pgr1():
    kernel = create_kernel(256, 256, fp4=False, depthU=64)
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
        pgr=1,
    )


EXPECTED_EMIT_DEP_ORDER_256x256_BF16_PGR1 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] <- [5]
      preMFMA path 0:
        [ 5] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-7]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-7]
      path 1:
        [ 6] gr_inc     gr_inc(A)
        [ 3] gr         GR A (MT n+1, subIterK [0,1]) ids [0-7]
        [ 7] gr_inc     gr_inc(B)
        [ 4] gr         GR B (MT n+1, subIterK [0,1]) ids [0-7]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-7] , B : [0-7] <- [3]
      preMFMA path 0:
        [ 3] wait_lr    wait_lr
      path 0:
        [ 4] wait_gr    wait_gr(0)
        [ 5] sync       sync
        [ 6] lr_inc     lr_inc(A)
        [ 7] lr_inc     lr_inc(B)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-7]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-7]
"""


def test_256x256_bf16_pgr1():
    """Exact check of emit dependency order for 256x256 BF16, PGR1."""
    cfg = make_256x256_bf16_pgr1()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_256x256_BF16_PGR1, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_256x256_BF16_PGR1}\n"
        f"--- Actual ---\n{actual}"
    )


def make_256x256_fp4():
    kernel = create_kernel(256, 256, fp4=True, depthU=256)
    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    scaleTiA = makeTileInfo('MXSA', kernel)
    scaleTiB = makeTileInfo('MXSB', kernel)
    return SchedulerConfig(
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


EXPECTED_EMIT_DEP_ORDER_256x256_FP4_1x1 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] <- [5]
      preMFMA path 0:
        [ 5] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-7]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-7]
        [ 6] wait_lr    wait_lr
        [ 7] sync       sync
        [ 8] gr_inc     gr_inc(A)
        [ 3] gr         GR A (MT n+2, subIterK [0,1]) ids [0-7]
        [ 9] gr_inc     gr_inc(B)
        [ 4] gr         GR B (MT n+2, subIterK [0,1]) ids [0-0]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-7] , B : [0-7] <- [8]
      preMFMA path 0:
        [ 8] wait_lr    wait_lr
      path 0:
        [ 9] wait_gr    wait_gr(A=8,B=1)
        [10] sync       sync
        [11] lr_inc     lr_inc(A)
        [12] lr_inc     lr_inc(B)
        [13] lr_inc     lr_inc(SA)
        [14] lr_inc     lr_inc(SB)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-7]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-7]
        [ 3] lr         LR SA (MT n+1, subIterK [0,1]) [0-7]
        [ 4] lr         LR SB (MT n+1, subIterK [0,1]) [0-7]
      path 1:
        [ 5] gr         GR B (MT n+2, subIterK [0,1]) ids [1-7]
        [15] gr_inc     gr_inc(SA)
        [ 6] gr         GR SA (MT n+2, subIterK [0,1]) ids [0-7]
        [16] gr_inc     gr_inc(SB)
        [ 7] gr         GR SB (MT n+2, subIterK [0,1]) ids [0-7]
"""


def test_256x256_fp4_partition_1x1():
    """Exact check of emit dependency order for 256x256 FP4, 1x1 partition."""
    cfg = make_256x256_fp4()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_256x256_FP4_1x1, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_256x256_FP4_1x1}\n"
        f"--- Actual ---\n{actual}"
    )


def make_256x256_fp8(pgr=2):
    from unittest.mock import MagicMock
    dtype = MagicMock()
    dtype.numBytes.return_value = 1
    kernel = {
        "DepthU": 128, "_DepthUA": 128, "_DepthUB": 128,
        "MacroTileA": 256, "MacroTileB": 256,
        "MacroTile0": 256, "MacroTile1": 256,
        "MatrixInstM": 16, "MatrixInstN": 16, "MatrixInstK": 128,
        "MIWaveGroup": [2, 2], "WavefrontSize": 64,
        "SourceSwap": False, "MIArchVgpr": False,
        "NonTemporalA": 0, "NonTemporalB": 0,
        "NonTemporalMXSA": 0, "NonTemporalMXSB": 0,
        "ProblemType": {"DataTypeA": dtype, "DataTypeB": dtype,
                        "ComputeDataType": MagicMock(**{"numBytes.return_value": 4})},
    }
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


EXPECTED_EMIT_DEP_ORDER_256x256_FP8_PGR1 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] <- [5]
      preMFMA path 0:
        [ 5] wait_lr    wait_lr
      path 0:
        [10] gr_inc     gr_inc(A)
        [ 3] gr         GR A (MT n+1, subIterK [0]) ids [0-7]
        [11] gr_inc     gr_inc(B)
        [ 4] gr         GR B (MT n+1, subIterK [0]) ids [0-7]
        [ 6] wait_gr    wait_gr(0)
        [ 7] sync       sync
        [ 8] lr_inc     lr_inc(A)
        [ 9] lr_inc     lr_inc(B)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-7]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-7]
"""

EXPECTED_EMIT_DEP_ORDER_256x256_FP8_PGR2 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] <- [5]
      preMFMA path 0:
        [ 5] wait_lr    wait_lr
      path 0:
        [ 6] wait_gr    wait_gr(0)
        [ 7] sync       sync
        [ 8] lr_inc     lr_inc(A)
        [ 9] lr_inc     lr_inc(B)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-7]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-7]
      path 1:
        [10] sync       sync
        [11] gr_inc     gr_inc(A)
        [ 3] gr         GR A (MT n+2, subIterK [0]) ids [0-7]
        [12] gr_inc     gr_inc(B)
        [ 4] gr         GR B (MT n+2, subIterK [0]) ids [0-7]
"""


def test_256x256_fp8_partition_1x1_pgr1():
    """Exact check of emit dependency order for 256x256 FP8, 1x1 partition, PGR=1.

    PGR=1: GR and LR are in the same path — GR issues for MT n+1 then
    wait_gr(0) then LR, all in one sequential chain.
    """
    cfg = make_256x256_fp8(pgr=1)
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_256x256_FP8_PGR1, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_256x256_FP8_PGR1}\n"
        f"--- Actual ---\n{actual}"
    )


def test_256x256_fp8_partition_1x1_pgr2():
    """Exact check of emit dependency order for 256x256 FP8, 1x1 partition, PGR=2.

    PGR=2: GR and LR split into two independent paths — LR in path 0
    (wait_gr(0) then LR for MT n+1), GR in path 1 (GR for MT n+2).
    wait_gr(0) confirms _compute_inflight_loads yields vmcnt=0 for the
    symmetric case after the inflight-loads bug fix.
    """
    cfg = make_256x256_fp8(pgr=2)
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_256x256_FP8_PGR2, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_256x256_FP8_PGR2}\n"
        f"--- Actual ---\n{actual}"
    )


def make_128x128_bf16():
    kernel = create_kernel(128, 128, fp4=False, depthU=128)
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
    )


EXPECTED_EMIT_DEP_ORDER_128x128_BF16_1x1 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-3] , B : [0-3] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-3]
        [ 5] wait_lr    wait_lr
        [ 6] sync       sync
        [ 7] gr_inc     gr_inc(A)
        [ 3] gr         GR A (MT n+2, subIterK [0,1]) ids [0-3]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-3] , B : [0-3] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 5] wait_gr    wait_gr(A=12,B=8)
        [ 6] sync       sync
        [ 1] lr         LR A  (MT n, subIterK [2]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [2]) [0-3]
      path 1:
        [ 7] gr_inc     gr_inc(B)
        [ 3] gr         GR B (MT n+2, subIterK [0,1]) ids [0-3]
    subIterK=2:
      MFMA: [ 0] MFMAs (MT n, subIterK 2  ) A : [0-3] , B : [0-3] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [3]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [3]) [0-3]
        [ 5] wait_lr    wait_lr
        [ 6] sync       sync
        [ 3] gr         GR A (MT n+2, subIterK [2,3]) ids [0-3]
    subIterK=3:
      MFMA: [ 0] MFMAs (MT n, subIterK 3  ) A : [0-3] , B : [0-3] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 5] wait_gr    wait_gr(A=12,B=8)
        [ 6] sync       sync
        [ 7] lr_inc     lr_inc(A)
        [ 8] lr_inc     lr_inc(B)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-3]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-3]
      path 1:
        [ 3] gr         GR B (MT n+2, subIterK [2,3]) ids [0-3]
"""


def test_128x128_bf16_partition_1x1():
    """Exact check of emit dependency order for 128x128 BF16, 1x1 partition."""
    cfg = make_128x128_bf16()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_128x128_BF16_1x1, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_128x128_BF16_1x1}\n"
        f"--- Actual ---\n{actual}"
    )


def make_128x128_fp4():
    kernel = create_kernel(128, 128, fp4=True, depthU=512)
    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    scaleTiA = makeTileInfo('MXSA', kernel)
    scaleTiB = makeTileInfo('MXSB', kernel)
    return SchedulerConfig(
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


EXPECTED_EMIT_DEP_ORDER_128x128_FP4_1x1 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-3] , B : [0-3] <- [6]
      preMFMA path 0:
        [ 6] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-3]
        [ 3] lr         LR SA (MT n, subIterK [2,3]) [0-3]
        [ 7] wait_lr    wait_lr
        [ 8] sync       sync
        [ 9] gr_inc     gr_inc(A)
        [ 4] gr         GR A (MT n+2, subIterK [0,1]) ids [0-3]
        [10] gr_inc     gr_inc(B)
        [ 5] gr         GR B (MT n+2, subIterK [0,1]) ids [0-0]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-3] , B : [0-3] <- [6]
      preMFMA path 0:
        [ 6] wait_lr    wait_lr
      path 0:
        [ 7] wait_gr    wait_gr(A=12,B=9,SA=1,SB=1)
        [ 8] sync       sync
        [ 1] lr         LR A  (MT n, subIterK [2]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [2]) [0-3]
        [ 3] lr         LR SB (MT n, subIterK [2,3]) [0-3]
      path 1:
        [ 4] gr         GR B (MT n+2, subIterK [0,1]) ids [1-3]
        [ 9] sync       sync
        [10] gr_inc     gr_inc(SA)
        [ 5] gr         GR SA (MT n+2, subIterK [0,3]) ids [0-3]
    subIterK=2:
      MFMA: [ 0] MFMAs (MT n, subIterK 2  ) A : [0-3] , B : [0-3] <- [5]
      preMFMA path 0:
        [ 5] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [3]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [3]) [0-3]
        [ 6] wait_lr    wait_lr
        [ 7] sync       sync
        [ 8] gr_inc     gr_inc(SB)
        [ 3] gr         GR SB (MT n+2, subIterK [0,3]) ids [0-3]
        [ 4] gr         GR A (MT n+2, subIterK [2,3]) ids [0-3]
    subIterK=3:
      MFMA: [ 0] MFMAs (MT n, subIterK 3  ) A : [0-3] , B : [0-3] <- [6]
      preMFMA path 0:
        [ 6] wait_lr    wait_lr
      path 0:
        [ 7] wait_gr    wait_gr(A=12,B=8,SA=1,SB=1)
        [ 8] sync       sync
        [ 9] lr_inc     lr_inc(A)
        [10] lr_inc     lr_inc(B)
        [11] lr_inc     lr_inc(SA)
        [12] lr_inc     lr_inc(SB)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-3]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-3]
        [ 3] lr         LR SA (MT n+1, subIterK [0,1]) [0-3]
        [ 4] lr         LR SB (MT n+1, subIterK [0,1]) [0-3]
      path 1:
        [ 5] gr         GR B (MT n+2, subIterK [2,3]) ids [0-3]
"""


def test_128x128_fp4_partition_1x1():
    """Exact check of emit dependency order for 128x128 FP4, 1x1 partition."""
    cfg = make_128x128_fp4()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_128x128_FP4_1x1, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_128x128_FP4_1x1}\n"
        f"--- Actual ---\n{actual}"
    )


def make_256x256_fp4_pgr0():
    kernel = create_kernel(256, 256, fp4=True, depthU=256)
    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    scaleTiA = makeTileInfo('MXSA', kernel)
    scaleTiB = makeTileInfo('MXSB', kernel)
    return SchedulerConfig(
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
        pgr=0,
    )


EXPECTED_EMIT_DEP_ORDER_256x256_FP4_PGR0 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] <- [15]
      preMFMA path 0:
        [ 5] gr         GR A (MT n, subIterK [0,1]) ids [0-7]
        [11] gr_inc     gr_inc(A)
        [ 6] gr         GR B (MT n, subIterK [0,1]) ids [0-7]
        [12] gr_inc     gr_inc(B)
        [ 7] gr         GR SA (MT n, subIterK [0,1]) ids [0-7]
        [13] gr_inc     gr_inc(SA)
        [ 8] gr         GR SB (MT n, subIterK [0,1]) ids [0-7]
        [14] gr_inc     gr_inc(SB)
        [16] wait_gr    wait_gr(0)
        [17] sync       sync
        [ 1] lr         LR A  (MT n, subIterK [0]) [0-7]
        [ 2] lr         LR B  (MT n, subIterK [0]) [0-7]
        [ 3] lr         LR SA (MT n, subIterK [0,1]) [0-7]
        [ 9] lr_inc     lr_inc(SA)
        [ 4] lr         LR SB (MT n, subIterK [0,1]) [0-7]
        [10] lr_inc     lr_inc(SB)
        [15] wait_lr    wait_lr
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-7] , B : [0-7] <- [5]
      preMFMA path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-7]
        [ 3] lr_inc     lr_inc(A)
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-7]
        [ 4] lr_inc     lr_inc(B)
        [ 5] wait_lr    wait_lr
"""


def test_256x256_fp4_pgr0():
    """Exact check of emit dependency order for 256x256 FP4, PGR0."""
    cfg = make_256x256_fp4_pgr0()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_256x256_FP4_PGR0, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_256x256_FP4_PGR0}\n"
        f"--- Actual ---\n{actual}"
    )


def make_256x256_fp4_pgr1():
    kernel = create_kernel(256, 256, fp4=True, depthU=256)
    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    scaleTiA = makeTileInfo('MXSA', kernel)
    scaleTiB = makeTileInfo('MXSB', kernel)
    return SchedulerConfig(
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
        pgr=1,
    )


EXPECTED_EMIT_DEP_ORDER_256x256_FP4_PGR1 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] <- [7]
      preMFMA path 0:
        [ 7] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-7]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-7]
      path 1:
        [ 8] gr_inc     gr_inc(A)
        [ 3] gr         GR A (MT n+1, subIterK [0,1]) ids [0-7]
        [ 9] gr_inc     gr_inc(B)
        [ 4] gr         GR B (MT n+1, subIterK [0,1]) ids [0-7]
        [10] gr_inc     gr_inc(SA)
        [ 5] gr         GR SA (MT n+1, subIterK [0,1]) ids [0-7]
        [11] gr_inc     gr_inc(SB)
        [ 6] gr         GR SB (MT n+1, subIterK [0,1]) ids [0-7]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-7] , B : [0-7] <- [5]
      preMFMA path 0:
        [ 5] wait_lr    wait_lr
      path 0:
        [ 6] wait_gr    wait_gr(0)
        [ 7] sync       sync
        [ 8] lr_inc     lr_inc(A)
        [ 9] lr_inc     lr_inc(B)
        [10] lr_inc     lr_inc(SA)
        [11] lr_inc     lr_inc(SB)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-7]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-7]
        [ 3] lr         LR SA (MT n+1, subIterK [0,1]) [0-7]
        [ 4] lr         LR SB (MT n+1, subIterK [0,1]) [0-7]
"""


def test_256x256_fp4_pgr1():
    """Exact check of emit dependency order for 256x256 FP4, PGR1."""
    cfg = make_256x256_fp4_pgr1()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_256x256_FP4_PGR1, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_256x256_FP4_PGR1}\n"
        f"--- Actual ---\n{actual}"
    )


def make_128x128_bf16_pgr1():
    kernel = create_kernel(128, 128, fp4=False, depthU=128)
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
        pgr=1,
    )


EXPECTED_EMIT_DEP_ORDER_128x128_BF16_PGR1 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-3] , B : [0-3] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-3]
      path 1:
        [ 5] gr_inc     gr_inc(A)
        [ 3] gr         GR A (MT n+1, subIterK [0,1]) ids [0-3]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-3] , B : [0-3] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 5] wait_gr    wait_gr(A=4)
        [ 6] sync       sync
        [ 1] lr         LR A  (MT n, subIterK [2]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [2]) [0-3]
      path 1:
        [ 7] gr_inc     gr_inc(B)
        [ 3] gr         GR B (MT n+1, subIterK [0,1]) ids [0-3]
    subIterK=2:
      MFMA: [ 0] MFMAs (MT n, subIterK 2  ) A : [0-3] , B : [0-3] <- [5]
      preMFMA path 0:
        [ 5] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [3]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [3]) [0-3]
      path 1:
        [ 3] gr         GR A (MT n+1, subIterK [2,3]) ids [0-3]
        [ 4] gr         GR B (MT n+1, subIterK [2,3]) ids [0-3]
    subIterK=3:
      MFMA: [ 0] MFMAs (MT n, subIterK 3  ) A : [0-3] , B : [0-3] <- [3]
      preMFMA path 0:
        [ 3] wait_lr    wait_lr
      path 0:
        [ 4] wait_gr    wait_gr(A=4,B=4)
        [ 5] sync       sync
        [ 6] lr_inc     lr_inc(A)
        [ 7] lr_inc     lr_inc(B)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-3]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-3]
"""

# FIXME : this schedule has issues for now. bad SA/SB counts + bad ordering of AB/SASB for performance
def test_128x128_bf16_pgr1():
    """Exact check of emit dependency order for 128x128 BF16, DU=128, PGR1."""
    cfg = make_128x128_bf16_pgr1()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_128x128_BF16_PGR1, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_128x128_BF16_PGR1}\n"
        f"--- Actual ---\n{actual}"
    )


def make_128x96_bf16_pgr1_wg4x1():
    """MT128x96, MIWT 2x6, WG 4x1, DU=128, PGR1.

    Mirrors subtile_bf16_failing.yaml. With this geometry, multiple GR_B
    placements end up in the same subIterK slot (slot 2: ids [4-5] and
    [0-5]). loadRatioGR_B > 1.0 so grB.mn = 2 (mirrors Kernel.py:1139-1140).
    """
    kernel = create_kernel(128, 96, fp4=False, depthU=128, waveGroup=(4, 1))
    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    grA = ReadGranularity(mn=1, k=2) if tiA.loadRatioGR <= 1.0 else ReadGranularity(mn=2, k=2)
    grB = ReadGranularity(mn=1, k=2) if tiB.loadRatioGR <= 1.0 else ReadGranularity(mn=2, k=2)
    return SchedulerConfig(
        numMFMATilesM=tiA.localMMATileGrid[0],
        numMFMATilesN=tiB.localMMATileGrid[0],
        numSubIterK=tiA.localMMATileGrid[1],
        lrA=ReadGranularity(mn=1, k=1),
        lrB=ReadGranularity(mn=1, k=1),
        grA=grA,
        grB=grB,
        pgr=1,
    )


# Regression: pins the post-fix behavior of remove_unnecessary_gr_deps when
# multiple GRs of the same tensor share a (mt_offset, partition, slot).
# The pass used to collapse those into one key and drop deps on later-rank
# GRs, leaving the surviving wait anchored on an earlier-rank GR — under-
# counting in-flight loads and causing stale-LDS reads on the next iter.
# Diff vs the buggy version: subIterK=1 wait_gr changes from
# "wait_gr(A=2,B=3)" to "wait_gr(A=2)" (B-wait pushed to wrap-around).
EXPECTED_EMIT_DEP_ORDER_128x96_BF16_PGR1_WG4x1 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-1] , B : [0-5] <- [5]
      preMFMA path 0:
        [ 5] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-1]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-5]
      path 1:
        [ 6] gr_inc     gr_inc(A)
        [ 3] gr         GR A (MT n+1, subIterK [0,1]) ids [0-1]
        [ 7] gr_inc     gr_inc(B)
        [ 4] gr         GR B (MT n+1, subIterK [0,1]) ids [0-1]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-1] , B : [0-5] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 5] wait_gr    wait_gr(A=2,B=1)
        [ 6] sync       sync
        [ 1] lr         LR A  (MT n, subIterK [2]) [0-1]
        [ 2] lr         LR B  (MT n, subIterK [2]) [0-5]
      path 1:
        [ 3] gr         GR B (MT n+1, subIterK [0,1]) ids [2-5]
    subIterK=2:
      MFMA: [ 0] MFMAs (MT n, subIterK 2  ) A : [0-1] , B : [0-5] <- [5]
      preMFMA path 0:
        [ 5] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [3]) [0-1]
        [ 2] lr         LR B  (MT n, subIterK [3]) [0-5]
      path 1:
        [ 3] gr         GR A (MT n+1, subIterK [2,3]) ids [0-1]
        [ 4] gr         GR B (MT n+1, subIterK [2,3]) ids [0-5]
    subIterK=3:
      MFMA: [ 0] MFMAs (MT n, subIterK 3  ) A : [0-1] , B : [0-5] <- [3]
      preMFMA path 0:
        [ 3] wait_lr    wait_lr
      path 0:
        [ 4] wait_gr    wait_gr(A=2,B=3)
        [ 5] sync       sync
        [ 6] lr_inc     lr_inc(A)
        [ 7] lr_inc     lr_inc(B)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-1]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-5]
"""


def test_128x96_bf16_pgr1_wg4x1():
    """Regression: multi-GR-per-slot anchoring in remove_unnecessary_gr_deps.

    Mirrors subtile_bf16_failing.yaml. Pins the post-fix behavior so a future
    change that re-collapses GRs sharing (mt_offset, partition, slot)
    will diff against this expected output.
    """
    cfg = make_128x96_bf16_pgr1_wg4x1()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_128x96_BF16_PGR1_WG4x1, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_128x96_BF16_PGR1_WG4x1}\n"
        f"--- Actual ---\n{actual}"
    )


def make_128x128_fp4_pgr1():
    kernel = create_kernel(128, 128, fp4=True, depthU=512)
    tiA = makeTileInfo('A', kernel)
    tiB = makeTileInfo('B', kernel)
    scaleTiA = makeTileInfo('MXSA', kernel)
    scaleTiB = makeTileInfo('MXSB', kernel)
    return SchedulerConfig(
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
        pgr=1,
    )


EXPECTED_EMIT_DEP_ORDER_128x128_FP4_PGR1 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-3] , B : [0-3] <- [6]
      preMFMA path 0:
        [ 6] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-3]
        [ 3] lr         LR SA (MT n, subIterK [2,3]) [0-3]
      path 1:
        [ 7] gr_inc     gr_inc(A)
        [ 4] gr         GR A (MT n+1, subIterK [0,1]) ids [0-3]
        [ 8] gr_inc     gr_inc(B)
        [ 5] gr         GR B (MT n+1, subIterK [0,1]) ids [0-0]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-3] , B : [0-3] <- [6]
      preMFMA path 0:
        [ 6] wait_lr    wait_lr
      path 0:
        [ 7] wait_gr    wait_gr(A=4,B=1)
        [ 8] sync       sync
        [ 1] lr         LR A  (MT n, subIterK [2]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [2]) [0-3]
        [ 3] lr         LR SB (MT n, subIterK [2,3]) [0-3]
      path 1:
        [ 4] gr         GR B (MT n+1, subIterK [0,1]) ids [1-3]
        [ 9] gr_inc     gr_inc(SA)
        [ 5] gr         GR SA (MT n+1, subIterK [0,3]) ids [0-3]
    subIterK=2:
      MFMA: [ 0] MFMAs (MT n, subIterK 2  ) A : [0-3] , B : [0-3] <- [6]
      preMFMA path 0:
        [ 6] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [3]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [3]) [0-3]
      path 1:
        [ 7] gr_inc     gr_inc(SB)
        [ 3] gr         GR SB (MT n+1, subIterK [0,3]) ids [0-3]
        [ 4] gr         GR A (MT n+1, subIterK [2,3]) ids [0-3]
        [ 5] gr         GR B (MT n+1, subIterK [2,3]) ids [0-3]
    subIterK=3:
      MFMA: [ 0] MFMAs (MT n, subIterK 3  ) A : [0-3] , B : [0-3] <- [5]
      preMFMA path 0:
        [ 5] wait_lr    wait_lr
      path 0:
        [ 6] wait_gr    wait_gr(A=4,B=4)
        [ 7] sync       sync
        [ 8] lr_inc     lr_inc(A)
        [ 9] lr_inc     lr_inc(B)
        [10] lr_inc     lr_inc(SA)
        [11] lr_inc     lr_inc(SB)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-3]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-3]
        [ 3] lr         LR SA (MT n+1, subIterK [0,1]) [0-3]
        [ 4] lr         LR SB (MT n+1, subIterK [0,1]) [0-3]
"""

def test_128x128_fp4_pgr1():
    """Exact check of emit dependency order for 128x128 FP4, DU=512, PGR1."""
    cfg = make_128x128_fp4_pgr1()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_128x128_FP4_PGR1, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_128x128_FP4_PGR1}\n"
        f"--- Actual ---\n{actual}"
    )


EXPECTED_PRELOOP_256x256_FP4_1x1 = """\
MAINLOOP:
  Partition 0:
    subIterK=0:
      [ 0] gr         GR A (MT n, subIterK [0,1]) ids [0-7]
      [ 1] gr         GR B (MT n, subIterK [0,1]) ids [0-7]
      [ 2] gr         GR SA (MT n, subIterK [0,1]) ids [0-7]
      [ 3] gr         GR SB (MT n, subIterK [0,1]) ids [0-7]
      [ 4] gr_inc     gr_inc(A)
      [ 5] gr_inc     gr_inc(B)
      [ 6] gr_inc     gr_inc(SA)
      [ 7] gr_inc     gr_inc(SB)
      [ 8] wait_gr    wait_gr(0)
      [ 9] sync       sync
      [10] lr         LR A  (MT n, subIterK [0]) [0-7]
      [11] lr         LR B  (MT n, subIterK [0]) [0-7]
      [12] lr         LR SA (MT n, subIterK [0,1]) [0-7]
      [13] lr         LR SB (MT n, subIterK [0,1]) [0-7]
      [14] skip       skip(LE:1:NLL)
      [15] gr         GR A (MT n+1, subIterK [0,1]) ids [0-7]
      [16] gr         GR B (MT n+1, subIterK [0,1]) ids [0-7]
      [17] gr         GR SA (MT n+1, subIterK [0,1]) ids [0-7]
      [18] gr         GR SB (MT n+1, subIterK [0,1]) ids [0-7]
      [19] skip       skip(LE:2:NGLL)
"""


def test_256x256_fp4_preloop_1x1():
    """Exact check of preloop for 256x256 FP4, 1x1 partition."""
    cfg = make_256x256_fp4()
    sched = LogicalScheduler(cfg)
    sched.emit()
    preloop = sched.build_preloop()
    actual = sched.print_emit(preloop)
    assert actual == EXPECTED_PRELOOP_256x256_FP4_1x1, (
        f"Preloop mismatch.\n"
        f"--- Expected ---\n{EXPECTED_PRELOOP_256x256_FP4_1x1}\n"
        f"--- Actual ---\n{actual}"
    )


EXPECTED_PRELOOP_256x256_FP4_PGR1_1x1 = """\
MAINLOOP:
  Partition 0:
    subIterK=0:
      [ 0] gr         GR A (MT n, subIterK [0,1]) ids [0-7]
      [ 1] gr         GR B (MT n, subIterK [0,1]) ids [0-7]
      [ 2] gr         GR SA (MT n, subIterK [0,1]) ids [0-7]
      [ 3] gr         GR SB (MT n, subIterK [0,1]) ids [0-7]
      [ 4] wait_gr    wait_gr(0)
      [ 5] sync       sync
      [ 6] lr         LR A  (MT n, subIterK [0]) [0-7]
      [ 7] lr         LR B  (MT n, subIterK [0]) [0-7]
      [ 8] lr         LR SA (MT n, subIterK [0,1]) [0-7]
      [ 9] lr         LR SB (MT n, subIterK [0,1]) [0-7]
      [10] skip       skip(LE:1:NLL)
"""


def test_256x256_fp4_preloop_pgr1_1x1():
    """Exact check of preloop for 256x256 FP4, PGR1, 1x1 partition.
    PGR1 preloop: GR(MT0) + wait + sync + LR + skip(NLL). No MT1 GRs."""
    cfg = make_256x256_fp4_pgr1()
    sched = LogicalScheduler(cfg)
    sched.emit()
    preloop = sched.build_preloop()
    actual = sched.print_emit(preloop)
    assert actual == EXPECTED_PRELOOP_256x256_FP4_PGR1_1x1, (
        f"Preloop mismatch.\n"
        f"--- Expected ---\n{EXPECTED_PRELOOP_256x256_FP4_PGR1_1x1}\n"
        f"--- Actual ---\n{actual}"
    )


EXPECTED_PRELOOP_320x320_BF16_1x5_OFFSET1 = """\
MAINLOOP:
  Partition 0:
    subIterK=0:
      [ 0] gr         GR A (MT n, subIterK [0,1]) ids [0-9]
      [ 1] gr         GR B (MT n, subIterK [0,1]) ids [0-9]
      [ 2] gr_inc     gr_inc(A)
      [ 3] gr_inc     gr_inc(B)
      [ 4] wait_gr    wait_gr(0)
      [ 5] sync       sync
      [ 6] lr         LR A  (MT n, subIterK [0]) [0-9]
      [ 7] lr         LR B  (MT n, subIterK [0]) [0-1]
      [ 8] skip       skip(LE:1:NLL)
      [ 9] gr         GR A (MT n+1, subIterK [0,1]) ids [0-9]
      [10] gr         GR B (MT n+1, subIterK [0,1]) ids [0-1]
      [11] skip       skip(LE:2:NGLL)
"""

EXPECTED_PRELOOP_320x320_BF16_1x5_OFFSET_ALL = """\
MAINLOOP:
  Partition 0:
    subIterK=0:
      [ 0] gr         GR A (MT n, subIterK [0,1]) ids [0-9]
      [ 1] gr         GR B (MT n, subIterK [0,1]) ids [0-9]
      [ 2] gr_inc     gr_inc(A)
      [ 3] gr_inc     gr_inc(B)
      [ 4] wait_gr    wait_gr(0)
      [ 5] sync       sync
      [ 6] lr         LR A  (MT n, subIterK [0]) [0-9]
      [ 7] lr         LR B  (MT n, subIterK [0]) [0-1]
      [ 8] skip       skip(LE:1:NLL)
      [ 9] gr         GR A (MT n+1, subIterK [0,1]) ids [0-9]
      [10] gr         GR B (MT n+1, subIterK [0,1]) ids [0-1]
      [11] gr         GR B (MT n+1, subIterK [0,1]) ids [2-3]
      [12] gr         GR B (MT n+1, subIterK [0,1]) ids [4-5]
      [13] gr         GR B (MT n+1, subIterK [0,1]) ids [6-7]
      [14] gr         GR B (MT n+1, subIterK [0,1]) ids [8-9]
      [15] skip       skip(LE:2:NGLL)
"""


def test_320x320_bf16_preloop_1x5_offset1():
    """Preloop with offsetPartition=1: MT1 GRs cover only partition 0."""
    cfg = make_320x320_bf16()
    cfg.offsetPartition = 1
    sched = LogicalScheduler(cfg)
    sched.emit()
    preloop = sched.build_preloop()
    actual = sched.print_emit(preloop)
    assert actual == EXPECTED_PRELOOP_320x320_BF16_1x5_OFFSET1, (
        f"Preloop mismatch.\n"
        f"--- Expected ---\n{EXPECTED_PRELOOP_320x320_BF16_1x5_OFFSET1}\n"
        f"--- Actual ---\n{actual}"
    )


def test_320x320_bf16_preloop_1x5_offset_all():
    """Preloop with offsetPartition=numPartitions: MT1 GRs cover all partitions."""
    cfg = make_320x320_bf16()
    # Forcing offsetPartition for testing. Not exposed yet.
    cfg.offsetPartition = cfg.numPartitions
    sched = LogicalScheduler(cfg)
    sched.emit()
    preloop = sched.build_preloop()
    actual = sched.print_emit(preloop)
    assert actual == EXPECTED_PRELOOP_320x320_BF16_1x5_OFFSET_ALL, (
        f"Preloop mismatch.\n"
        f"--- Expected ---\n{EXPECTED_PRELOOP_320x320_BF16_1x5_OFFSET_ALL}\n"
        f"--- Actual ---\n{actual}"
    )
