"""Non-regression tests for LogicalScheduler — exact output checks.

These tests capture the expected scheduling output at specific pipeline steps
to detect unintended regressions. 
"""

from Tensile.Components.SubtileBasedKernel import TileInfo
from Tensile.Components.SubtileBasedLogicalScheduler import (
    LogicalScheduler,
    ReadGranularity,
    SchedulerConfig,
)


def create_kernel(MT0=256, MT1=256, fp4=False, depthU=None):
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


def make_256x256_bf16():
    kernel = create_kernel(256, 256, fp4=False, depthU=64)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    return SchedulerConfig(
        numMFMATilesM=tiA.localMMATileGrid[0],
        numMFMATilesN=tiB.localMMATileGrid[0],
        numSubIterK=tiA.localMMATileGrid[1],
        lrA=ReadGranularity(mn=1, k=1),
        lrB=ReadGranularity(mn=1, k=1),
        grA=ReadGranularity(mn=1, k=2),
        grB=ReadGranularity(mn=1, k=2),
        numPartitionsM=1,
        numPartitionsN=1,
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
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    return SchedulerConfig(
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
        [ 5] wait_gr    wait_gr(A=7,B=8)
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


def make_320x320_bf16():
    kernel = create_kernel(320, 320, fp4=False, depthU=64)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    return SchedulerConfig(
        numMFMATilesM=tiA.localMMATileGrid[0],
        numMFMATilesN=tiB.localMMATileGrid[0],
        numSubIterK=tiA.localMMATileGrid[1],
        lrA=ReadGranularity(mn=1, k=1),
        lrB=ReadGranularity(mn=1, k=1),
        grA=ReadGranularity(mn=1, k=2),
        grB=ReadGranularity(mn=1, k=2),
        numPartitionsM=1,
        numPartitionsN=5,
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
        [ 4] sync       sync
        [ 5] gr_inc     gr_inc(A)
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


def make_256x256_fp4():
    kernel = create_kernel(256, 256, fp4=True, depthU=256)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    scaleTiA = TileInfo('MXSA', kernel)
    scaleTiB = TileInfo('MXSB', kernel)
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
        numPartitionsM=1,
        numPartitionsN=1,
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


def make_128x128_bf16():
    kernel = create_kernel(128, 128, fp4=False, depthU=128)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    return SchedulerConfig(
        numMFMATilesM=tiA.localMMATileGrid[0],
        numMFMATilesN=tiB.localMMATileGrid[0],
        numSubIterK=tiA.localMMATileGrid[1],
        lrA=ReadGranularity(mn=1, k=1),
        lrB=ReadGranularity(mn=1, k=1),
        grA=ReadGranularity(mn=1, k=2),
        grB=ReadGranularity(mn=1, k=2),
        numPartitionsM=1,
        numPartitionsN=1,
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
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    scaleTiA = TileInfo('MXSA', kernel)
    scaleTiB = TileInfo('MXSB', kernel)
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
        numPartitionsM=1,
        numPartitionsN=1,
    )


EXPECTED_EMIT_DEP_ORDER_128x128_FP4_1x1 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-3] , B : [0-3] <- [5]
      preMFMA path 0:
        [ 5] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-3]
        [ 3] lr         LR SA (MT n, subIterK [2,3]) [0-3]
        [ 6] wait_lr    wait_lr
        [ 7] sync       sync
        [ 8] gr_inc     gr_inc(A)
        [ 4] gr         GR A (MT n+2, subIterK [0,1]) ids [0-3]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-3] , B : [0-3] <- [5]
      preMFMA path 0:
        [ 5] wait_lr    wait_lr
      path 0:
        [ 6] wait_gr    wait_gr(A=12,B=8,SA=1,SB=1)
        [ 7] sync       sync
        [ 1] lr         LR A  (MT n, subIterK [2]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [2]) [0-3]
        [ 3] lr         LR SB (MT n, subIterK [2,3]) [0-3]
      path 1:
        [ 8] gr_inc     gr_inc(B)
        [ 4] gr         GR B (MT n+2, subIterK [0,1]) ids [0-3]
    subIterK=2:
      MFMA: [ 0] MFMAs (MT n, subIterK 2  ) A : [0-3] , B : [0-3] <- [6]
      preMFMA path 0:
        [ 6] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [3]) [0-3]
        [ 2] lr         LR B  (MT n, subIterK [3]) [0-3]
        [ 9] wait_lr    wait_lr
        [10] sync       sync
        [ 5] gr         GR A (MT n+2, subIterK [2,3]) ids [0-1]
        [ 7] gr_inc     gr_inc(SA)
        [ 3] gr         GR SA (MT n+2, subIterK [0,3]) ids [0-3]
        [ 8] gr_inc     gr_inc(SB)
        [ 4] gr         GR SB (MT n+2, subIterK [0,3]) ids [0-3]
    subIterK=3:
      MFMA: [ 0] MFMAs (MT n, subIterK 3  ) A : [0-3] , B : [0-3] <- [7]
      preMFMA path 0:
        [ 7] wait_lr    wait_lr
      path 0:
        [ 8] wait_gr    wait_gr(A=8,B=8,SA=1,SB=1)
        [ 9] sync       sync
        [10] lr_inc     lr_inc(A)
        [11] lr_inc     lr_inc(B)
        [12] lr_inc     lr_inc(SA)
        [13] lr_inc     lr_inc(SB)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-3]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-3]
        [ 3] lr         LR SA (MT n+1, subIterK [0,1]) [0-3]
        [ 4] lr         LR SB (MT n+1, subIterK [0,1]) [0-3]
      path 1:
        [ 5] gr         GR A (MT n+2, subIterK [2,3]) ids [2-3]
        [14] sync       sync
        [ 6] gr         GR B (MT n+2, subIterK [2,3]) ids [0-3]
"""


def test_128x128_fp4_partition_1x1():
    """Exact check of emit dependency order for 128x128 FP4, 1x1 partition."""
    cfg = make_128x128_fp4()
    sched = LogicalScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    # Note: [14] sync       sync is not needed because of the grouping of LRs. We could add an extra pass to detect those. TDB.
    assert actual == EXPECTED_EMIT_DEP_ORDER_128x128_FP4_1x1, (
        f"Emit dependency order mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_128x128_FP4_1x1}\n"
        f"--- Actual ---\n{actual}"
    )
