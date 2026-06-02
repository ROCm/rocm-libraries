"""Non-regression tests for assign_vgpr_tiles output.

These tests capture the expected VGPR tile allocation produced by
LogicalScheduler.assign_vgpr_tiles() for all non-FP8 configurations
defined in test_SubtileBasedSchedulerRef.py. They guard against any
unintended changes to vgpr_tile_maps for BF16 / FP4 schedules.

pgr only affects scheduling/wait ordering, not VGPR allocation, so each
(default-pgr, pgr1) pair is collapsed into one parametrized test that
asserts both factories produce the same map.
"""

import pytest

from test_SubtileBasedSchedulerRef import (
    make_256x256_bf16,
    make_384x256_bf16,
    make_320x320_bf16,
    make_256x256_bf16_pgr0,
    make_256x256_bf16_pgr1,
    make_128x128_bf16,
    make_128x128_bf16_pgr1,
    make_128x96_bf16_pgr1_wg4x1,
    make_256x256_fp4,
    make_128x128_fp4,
    make_256x256_fp4_pgr0,
    make_256x256_fp4_pgr1,
    make_128x128_fp4_pgr1,
)
from Tensile.Components.Subtile.LogicalScheduler import LogicalScheduler


def _vgpr_map(make):
    cfg = make()
    sched = LogicalScheduler(cfg)
    sched.assign_vgpr_tiles()
    return sched.print_vgpr()


def _check(make, expected):
    actual = _vgpr_map(make)
    assert actual == expected, (
        f"VGPR tile allocation mismatch.\n"
        f"--- Expected ---\n{expected}\n"
        f"--- Actual ---\n{actual}"
    )


EXPECTED_VGPR_256X256_BF16 = """\
needsUnrolling: False, unrollFactor: 1
vgprTiles: A: 16, B: 16
MAINLOOP:
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] A:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}, B:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR A  (MT n, subIterK [1]) [0-7] tiles:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}
      LR B  (MT n, subIterK [1]) [0-7] tiles:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-7] , B : [0-7] A:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}, B:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}
      LR A  (MT n+1, subIterK [0]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR B  (MT n+1, subIterK [0]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
"""


def test_vgpr_256x256_bf16():
    _check(make_256x256_bf16, EXPECTED_VGPR_256X256_BF16)


EXPECTED_VGPR_384X256_BF16 = """\
needsUnrolling: False, unrollFactor: 1
vgprTiles: A: 12, B: 16
MAINLOOP:
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-5] , B : [0-7] A:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}, B:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR A  (MT n, subIterK [1]) [0-5] tiles:{0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11}
      LR B  (MT n, subIterK [1]) [0-7] tiles:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-5] , B : [0-7] A:{0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11}, B:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}
      LR A  (MT n, subIterK [0]) [6-11] tiles:{6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5}
  Partition 1:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [6-11] , B : [0-7] A:{6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5}, B:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR A  (MT n, subIterK [1]) [6-11] tiles:{6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [6-11] , B : [0-7] A:{6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11}, B:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}
      LR A  (MT n+1, subIterK [0]) [0-5] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
      LR B  (MT n+1, subIterK [0]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
"""


def test_vgpr_384x256_bf16():
    _check(make_384x256_bf16, EXPECTED_VGPR_384X256_BF16)


EXPECTED_VGPR_320X320_BF16 = """\
needsUnrolling: False, unrollFactor: 1
vgprTiles: A: 20, B: 4
MAINLOOP:
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-9] , B : [0-1] A:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}, B:{0: 0, 1: 1}
      LR A  (MT n, subIterK [1]) [0-9] tiles:{0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18, 9: 19}
      LR B  (MT n, subIterK [1]) [0-1] tiles:{0: 2, 1: 3}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-9] , B : [0-1] A:{0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18, 9: 19}, B:{0: 2, 1: 3}
      LR B  (MT n, subIterK [0]) [2-3] tiles:{2: 0, 3: 1}
  Partition 1:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-9] , B : [2-3] A:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}, B:{2: 0, 3: 1}
      LR B  (MT n, subIterK [1]) [2-3] tiles:{2: 2, 3: 3}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-9] , B : [2-3] A:{0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18, 9: 19}, B:{2: 2, 3: 3}
      LR B  (MT n, subIterK [0]) [4-5] tiles:{4: 0, 5: 1}
  Partition 2:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-9] , B : [4-5] A:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}, B:{4: 0, 5: 1}
      LR B  (MT n, subIterK [1]) [4-5] tiles:{4: 2, 5: 3}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-9] , B : [4-5] A:{0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18, 9: 19}, B:{4: 2, 5: 3}
      LR B  (MT n, subIterK [0]) [6-7] tiles:{6: 0, 7: 1}
  Partition 3:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-9] , B : [6-7] A:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}, B:{6: 0, 7: 1}
      LR B  (MT n, subIterK [1]) [6-7] tiles:{6: 2, 7: 3}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-9] , B : [6-7] A:{0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18, 9: 19}, B:{6: 2, 7: 3}
      LR B  (MT n, subIterK [0]) [8-9] tiles:{8: 0, 9: 1}
  Partition 4:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-9] , B : [8-9] A:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}, B:{8: 0, 9: 1}
      LR B  (MT n, subIterK [1]) [8-9] tiles:{8: 2, 9: 3}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-9] , B : [8-9] A:{0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18, 9: 19}, B:{8: 2, 9: 3}
      LR A  (MT n+1, subIterK [0]) [0-9] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
      LR B  (MT n+1, subIterK [0]) [0-1] tiles:{0: 0, 1: 1}
"""


def test_vgpr_320x320_bf16():
    _check(make_320x320_bf16, EXPECTED_VGPR_320X320_BF16)


EXPECTED_VGPR_256X256_BF16_PGR0 = """\
needsUnrolling: False, unrollFactor: 1
vgprTiles: A: 8, B: 8
MAINLOOP:
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] A:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}, B:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR A  (MT n, subIterK [0]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR B  (MT n, subIterK [0]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-7] , B : [0-7] A:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}, B:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR A  (MT n, subIterK [1]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR B  (MT n, subIterK [1]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
"""


def test_vgpr_256x256_bf16_pgr0():
    _check(make_256x256_bf16_pgr0, EXPECTED_VGPR_256X256_BF16_PGR0)


EXPECTED_VGPR_128X128_BF16 = """\
needsUnrolling: False, unrollFactor: 1
vgprTiles: A: 8, B: 8
MAINLOOP:
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-3] , B : [0-3] A:{0: 0, 1: 1, 2: 2, 3: 3}, B:{0: 0, 1: 1, 2: 2, 3: 3}
      LR A  (MT n, subIterK [1]) [0-3] tiles:{0: 4, 1: 5, 2: 6, 3: 7}
      LR B  (MT n, subIterK [1]) [0-3] tiles:{0: 4, 1: 5, 2: 6, 3: 7}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-3] , B : [0-3] A:{0: 4, 1: 5, 2: 6, 3: 7}, B:{0: 4, 1: 5, 2: 6, 3: 7}
      LR A  (MT n, subIterK [2]) [0-3] tiles:{0: 0, 1: 1, 2: 2, 3: 3}
      LR B  (MT n, subIterK [2]) [0-3] tiles:{0: 0, 1: 1, 2: 2, 3: 3}
    subIterK=2:
      MFMAs (MT n, subIterK 2  ) A : [0-3] , B : [0-3] A:{0: 0, 1: 1, 2: 2, 3: 3}, B:{0: 0, 1: 1, 2: 2, 3: 3}
      LR A  (MT n, subIterK [3]) [0-3] tiles:{0: 4, 1: 5, 2: 6, 3: 7}
      LR B  (MT n, subIterK [3]) [0-3] tiles:{0: 4, 1: 5, 2: 6, 3: 7}
    subIterK=3:
      MFMAs (MT n, subIterK 3  ) A : [0-3] , B : [0-3] A:{0: 4, 1: 5, 2: 6, 3: 7}, B:{0: 4, 1: 5, 2: 6, 3: 7}
      LR A  (MT n+1, subIterK [0]) [0-3] tiles:{0: 0, 1: 1, 2: 2, 3: 3}
      LR B  (MT n+1, subIterK [0]) [0-3] tiles:{0: 0, 1: 1, 2: 2, 3: 3}
"""


def test_vgpr_128x128_bf16():
    _check(make_128x128_bf16, EXPECTED_VGPR_128X128_BF16)


EXPECTED_VGPR_128X96_BF16_PGR1_WG4X1 = """\
needsUnrolling: False, unrollFactor: 1
vgprTiles: A: 4, B: 12
MAINLOOP:
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-1] , B : [0-5] A:{0: 0, 1: 1}, B:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
      LR A  (MT n, subIterK [1]) [0-1] tiles:{0: 2, 1: 3}
      LR B  (MT n, subIterK [1]) [0-5] tiles:{0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-1] , B : [0-5] A:{0: 2, 1: 3}, B:{0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11}
      LR A  (MT n, subIterK [2]) [0-1] tiles:{0: 0, 1: 1}
      LR B  (MT n, subIterK [2]) [0-5] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    subIterK=2:
      MFMAs (MT n, subIterK 2  ) A : [0-1] , B : [0-5] A:{0: 0, 1: 1}, B:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
      LR A  (MT n, subIterK [3]) [0-1] tiles:{0: 2, 1: 3}
      LR B  (MT n, subIterK [3]) [0-5] tiles:{0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11}
    subIterK=3:
      MFMAs (MT n, subIterK 3  ) A : [0-1] , B : [0-5] A:{0: 2, 1: 3}, B:{0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11}
      LR A  (MT n+1, subIterK [0]) [0-1] tiles:{0: 0, 1: 1}
      LR B  (MT n+1, subIterK [0]) [0-5] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
"""


def test_vgpr_128x96_bf16_pgr1_wg4x1():
    _check(make_128x96_bf16_pgr1_wg4x1, EXPECTED_VGPR_128X96_BF16_PGR1_WG4X1)


EXPECTED_VGPR_256X256_FP4 = """\
needsUnrolling: True, unrollFactor: 2
vgprTiles: A: 16, B: 16, SA: 8, SB: 8
MAINLOOP (unroll 0):
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] A:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}, B:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}, SA:{0: 0, 2: 1, 4: 2, 6: 3}, SB:{0: 0, 2: 1, 4: 2, 6: 3}
      LR A  (MT n, subIterK [1]) [0-7] tiles:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}
      LR B  (MT n, subIterK [1]) [0-7] tiles:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-7] , B : [0-7] A:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}, B:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}, SA:{0: 0, 2: 1, 4: 2, 6: 3}, SB:{0: 0, 2: 1, 4: 2, 6: 3}
      LR A  (MT n+1, subIterK [0]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR B  (MT n+1, subIterK [0]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR SA (MT n+1, subIterK [0,1]) [0-7] tiles:{0: 4, 2: 5, 4: 6, 6: 7}
      LR SB (MT n+1, subIterK [0,1]) [0-7] tiles:{0: 4, 2: 5, 4: 6, 6: 7}
MAINLOOP (unroll 1):
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] A:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}, B:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}, SA:{0: 4, 2: 5, 4: 6, 6: 7}, SB:{0: 4, 2: 5, 4: 6, 6: 7}
      LR A  (MT n, subIterK [1]) [0-7] tiles:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}
      LR B  (MT n, subIterK [1]) [0-7] tiles:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-7] , B : [0-7] A:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}, B:{0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}, SA:{0: 4, 2: 5, 4: 6, 6: 7}, SB:{0: 4, 2: 5, 4: 6, 6: 7}
      LR A  (MT n+1, subIterK [0]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR B  (MT n+1, subIterK [0]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR SA (MT n+1, subIterK [0,1]) [0-7] tiles:{0: 0, 2: 1, 4: 2, 6: 3}
      LR SB (MT n+1, subIterK [0,1]) [0-7] tiles:{0: 0, 2: 1, 4: 2, 6: 3}
"""


def test_vgpr_256x256_fp4():
    _check(make_256x256_fp4, EXPECTED_VGPR_256X256_FP4)


EXPECTED_VGPR_128X128_FP4 = """\
needsUnrolling: False, unrollFactor: 1
vgprTiles: A: 8, B: 8, SA: 4, SB: 4
MAINLOOP:
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-3] , B : [0-3] A:{0: 0, 1: 1, 2: 2, 3: 3}, B:{0: 0, 1: 1, 2: 2, 3: 3}, SA:{0: 0, 2: 1}, SB:{0: 0, 2: 1}
      LR A  (MT n, subIterK [1]) [0-3] tiles:{0: 4, 1: 5, 2: 6, 3: 7}
      LR B  (MT n, subIterK [1]) [0-3] tiles:{0: 4, 1: 5, 2: 6, 3: 7}
      LR SA (MT n, subIterK [2,3]) [0-3] tiles:{0: 2, 2: 3}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-3] , B : [0-3] A:{0: 4, 1: 5, 2: 6, 3: 7}, B:{0: 4, 1: 5, 2: 6, 3: 7}, SA:{0: 0, 2: 1}, SB:{0: 0, 2: 1}
      LR A  (MT n, subIterK [2]) [0-3] tiles:{0: 0, 1: 1, 2: 2, 3: 3}
      LR B  (MT n, subIterK [2]) [0-3] tiles:{0: 0, 1: 1, 2: 2, 3: 3}
      LR SB (MT n, subIterK [2,3]) [0-3] tiles:{0: 2, 2: 3}
    subIterK=2:
      MFMAs (MT n, subIterK 2  ) A : [0-3] , B : [0-3] A:{0: 0, 1: 1, 2: 2, 3: 3}, B:{0: 0, 1: 1, 2: 2, 3: 3}, SA:{0: 2, 2: 3}, SB:{0: 2, 2: 3}
      LR A  (MT n, subIterK [3]) [0-3] tiles:{0: 4, 1: 5, 2: 6, 3: 7}
      LR B  (MT n, subIterK [3]) [0-3] tiles:{0: 4, 1: 5, 2: 6, 3: 7}
    subIterK=3:
      MFMAs (MT n, subIterK 3  ) A : [0-3] , B : [0-3] A:{0: 4, 1: 5, 2: 6, 3: 7}, B:{0: 4, 1: 5, 2: 6, 3: 7}, SA:{0: 2, 2: 3}, SB:{0: 2, 2: 3}
      LR A  (MT n+1, subIterK [0]) [0-3] tiles:{0: 0, 1: 1, 2: 2, 3: 3}
      LR B  (MT n+1, subIterK [0]) [0-3] tiles:{0: 0, 1: 1, 2: 2, 3: 3}
      LR SA (MT n+1, subIterK [0,1]) [0-3] tiles:{0: 0, 2: 1}
      LR SB (MT n+1, subIterK [0,1]) [0-3] tiles:{0: 0, 2: 1}
"""


def test_vgpr_128x128_fp4():
    _check(make_128x128_fp4, EXPECTED_VGPR_128X128_FP4)


EXPECTED_VGPR_256X256_FP4_PGR0 = """\
needsUnrolling: False, unrollFactor: 1
vgprTiles: A: 8, B: 8, SA: 4, SB: 4
MAINLOOP:
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] A:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}, B:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}, SA:{0: 0, 2: 1, 4: 2, 6: 3}, SB:{0: 0, 2: 1, 4: 2, 6: 3}
      LR A  (MT n, subIterK [0]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR B  (MT n, subIterK [0]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR SA (MT n, subIterK [0,1]) [0-7] tiles:{0: 0, 2: 1, 4: 2, 6: 3}
      LR SB (MT n, subIterK [0,1]) [0-7] tiles:{0: 0, 2: 1, 4: 2, 6: 3}
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-7] , B : [0-7] A:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}, B:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}, SA:{0: 0, 2: 1, 4: 2, 6: 3}, SB:{0: 0, 2: 1, 4: 2, 6: 3}
      LR A  (MT n, subIterK [1]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
      LR B  (MT n, subIterK [1]) [0-7] tiles:{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
"""


def test_vgpr_256x256_fp4_pgr0():
    _check(make_256x256_fp4_pgr0, EXPECTED_VGPR_256X256_FP4_PGR0)


@pytest.mark.parametrize(
    "default_make, pgr1_make",
    [
        (make_256x256_bf16, make_256x256_bf16_pgr1),
        (make_128x128_bf16, make_128x128_bf16_pgr1),
        (make_256x256_fp4, make_256x256_fp4_pgr1),
        (make_128x128_fp4, make_128x128_fp4_pgr1),
    ],
    ids=["256x256_bf16", "128x128_bf16", "256x256_fp4", "128x128_fp4"],
)
def test_vgpr_pgr_invariant(default_make, pgr1_make):
    """pgr only affects scheduling/wait ordering, not VGPR allocation."""
    assert _vgpr_map(default_make) == _vgpr_map(pgr1_make)


