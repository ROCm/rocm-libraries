# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.SolutionStructs.Solution import (
    _SUBTILE_STACK_MAX_PAD_DEN,
    _SUBTILE_STACK_MAX_PAD_NUM,
    _subtileStackForTile,
)


# (MFMA-M tiles in the macro tile, expected stack). At MatrixInstM=16 the tile
# is 16x the first column, so 12 -> MT192, 14 -> MT224, 16 -> MT256.
CASES = [
    (1, 2),     # degenerate: no rounding considered at all
    (2, 2),     # divides exactly, and is already the minimum
    (3, 4),     # pads 3 -> 4, exactly at the cap
    (4, 4),     # divides exactly; 4 -> 8 would exceed the cap
    (5, 2),     # 5 -> 8 is 1.6x, too much padding
    (6, 8),     # pads 6 -> 8, exactly at the cap
    (7, 8),     # pads 7 -> 8, comfortably under
    (8, 8),     # divides exactly
    (9, 2),     # 9 -> 16 is 1.78x
    (10, 2),    # 10 -> 16 is 1.6x
    (11, 2),    # 11 -> 16 is 1.45x, still over
    (12, 16),   # MT192: pads 12 -> 16, exactly at the cap
    (13, 16),   # MT208
    (14, 16),   # MT224: pads 14 -> 16
    (15, 16),   # MT240
    (16, 16),   # MT256: divides exactly, one full cache line
]


@pytest.mark.parametrize("mtTiles,expected", CASES)
def test_stack_for_tile(mtTiles, expected):
    assert _subtileStackForTile(mtTiles) == expected


@pytest.mark.parametrize("mtTiles", [3, 6, 12])
def test_boundary_cases_sit_exactly_on_the_cap(mtTiles):
    # These are the tiles the cap admits by equality rather than by margin, so a
    # float cap would decide them on binary rounding. Pin that the comparison is
    # exact: padding to the next power of two is worth exactly the cap ratio.
    stack = _subtileStackForTile(mtTiles)
    assert stack * _SUBTILE_STACK_MAX_PAD_DEN == mtTiles * _SUBTILE_STACK_MAX_PAD_NUM


def test_stack_never_shrinks_below_an_exact_divisor():
    # Rounding may only move the stack up; a tile that divides a taller stack
    # exactly must never be given a shorter one.
    for mtTiles in range(1, 17):
        exact = next((s for s in (16, 8, 4, 2) if mtTiles % s == 0), 2)
        assert _subtileStackForTile(mtTiles) >= exact
