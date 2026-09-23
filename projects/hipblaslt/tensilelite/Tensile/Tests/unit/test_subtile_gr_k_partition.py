################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

# TODO: TEMPORARY FIX. These tests cover the trigger for the subtile global-read
# K-partition bug (see fix_subtile_gr_missing_k_partition). Remove together with
# _subtileGRKPartitionIsBuggy once the emit-side fix lands.

import pytest

from Tensile.SolutionStructs.Solution import _subtileGRKPartitionIsBuggy


# (loadRatioGR, localSubtileGrid) -> expected buggy?
CASES = [
    # loadRatioGR <= 1: cooperative M-grouping never spans multiple subtiles.
    (1, [7, 2], False),
    (0.5, [7, 2], False),
    # loadRatioGR > 1 but M-subtile count divisible -> no partial "ghost" group.
    (2, [6, 2], False),
    (2, [8, 4], False),
    # loadRatioGR > 1, M-subtile count NOT divisible, but a single K-partition
    # (localSubtileGrid[1] == 1) -> no cross-partition overlap, no bug.
    (2, [7, 1], False),
    (2, [3, 1], False),
    # The buggy combination: loadRatioGR > 1, odd/indivisible M-subtile count,
    # AND more than one K-partition.
    (2, [7, 2], True),
    (2, [3, 4], True),
    (4, [6, 2], True),   # 6 % 4 != 0
]


@pytest.mark.parametrize("load_ratio,local_subtile_grid,expected", CASES)
def test_subtile_gr_k_partition_trigger(load_ratio, local_subtile_grid, expected):
    assert _subtileGRKPartitionIsBuggy(load_ratio, local_subtile_grid) is expected


def test_divisible_m_count_is_not_buggy():
    # 8 M-subtiles cover exactly 4 groups of loadRatioGR=2 -> no ghost slot.
    assert _subtileGRKPartitionIsBuggy(2, [8, 3]) is False


def test_single_k_partition_is_never_buggy():
    # Regression: many shipped bf16 kernels have odd M-subtile counts but only a
    # single K-partition; they must NOT be flagged.
    for m in range(1, 16):
        assert _subtileGRKPartitionIsBuggy(2, [m, 1]) is False
