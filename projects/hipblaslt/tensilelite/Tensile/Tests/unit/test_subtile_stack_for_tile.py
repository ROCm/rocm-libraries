# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.SolutionStructs.Solution import (
    _SUBTILE_STACK_MAX_PAD_DEN,
    _SUBTILE_STACK_MAX_PAD_NUM,
    _SUBTILE_STACK_SIZES,
    _subtileStackForTile,
    _subtileStackForTLU1,
    _subtileTLU1StackReason,
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


# --- geometry-aware fallback -------------------------------------------------
#
# _subtileStackForTile picks on cache-line utilization alone. _subtileStackForTLU1
# additionally backs off to a shorter stack when the preferred one cannot be laid
# out for the wave group, instead of leaving the solution to be rejected.

MI_M = 16
WAVE_GROUPS = [(1, 1), (1, 2), (2, 1), (1, 4), (2, 2), (4, 1)]


def _state(mtTilesM, mtTilesN, waveGroup, isa=(9, 5, 0)):
    """Minimal solution state carrying just what the TLU=1 stack rules read."""
    return {
        "ISA": isa,
        "MIWaveGroup": list(waveGroup),
        "MIWaveTile": [mtTilesM // waveGroup[0], mtTilesN // waveGroup[1]],
        "MacroTile0": mtTilesM * MI_M,
        "MacroTile1": mtTilesN * MI_M,
        "MatrixInstM": MI_M,
        "MatrixInstK": 128,
        "WavefrontSize": 64,
        "DepthU": 256,
    }


def test_mt192x192_wg2x2_falls_back_to_a_layout_that_works():
    # 12 tiles on both operands. The preferred stack of 16 is not a multiple of
    # the wave's MIWaveTile of 6, so the strip cannot be shared; stack 2 tiles
    # the dim exactly and each wave owns whole strips.
    state = _state(12, 12, (2, 2))
    for tc in ("A", "B"):
        assert _subtileStackForTile(12) == 16
        assert _subtileStackForTLU1(state, tc, 12) == 2
        assert _subtileTLU1StackReason(state, tc, 12, 2) is None


@pytest.mark.parametrize("waveGroup", [(1, 4), (4, 1)])
def test_mt192x192_keeps_rejecting_the_three_tile_wave_groups(waveGroup):
    # These give a wave 3 of the 12 tiles, and no power-of-two stack tiles 3:
    # 2 and 4 straddle, 8 and 16 are not multiples of 3. No fallback exists, so
    # the preferred stack is returned and the caller still rejects.
    state = _state(12, 12, waveGroup)
    tc = "A" if waveGroup[0] == 4 else "B"
    stack = _subtileStackForTLU1(state, tc, 12)
    assert stack == _subtileStackForTile(12)
    assert _subtileTLU1StackReason(state, tc, 12, stack) is not None


def test_fallback_never_moves_a_stack_that_already_works():
    # The whole safety argument for the fallback: it may only change geometries
    # that are rejected today. If the preferred stack is viable it must be kept,
    # so no currently-valid solution changes its LDS layout.
    for mtTiles in range(2, 17):
        for waveGroup in WAVE_GROUPS:
            if mtTiles % waveGroup[0] or mtTiles % waveGroup[1]:
                continue
            state = _state(mtTiles, mtTiles, waveGroup)
            preferred = _subtileStackForTile(mtTiles)
            for tc in ("A", "B"):
                if _subtileTLU1StackReason(state, tc, mtTiles, preferred) is None:
                    assert _subtileStackForTLU1(state, tc, mtTiles) == preferred


def test_strip_sharing_rules_stay_gfx950_only():
    # _validateSubtileGRKPartition polices strip sharing on gfx950 alone. The
    # chooser has to use the same gate, or it would reject on other ISAs a
    # geometry the validator there would accept.
    tiles, waveGroup = 12, (2, 2)
    gfx950 = _state(tiles, tiles, waveGroup)
    other = _state(tiles, tiles, waveGroup, isa=(12, 5, 0))
    preferred = _subtileStackForTile(tiles)
    assert _subtileTLU1StackReason(gfx950, "A", tiles, preferred) is not None
    assert _subtileTLU1StackReason(other, "A", tiles, preferred) is None
    assert _subtileStackForTLU1(other, "A", tiles) == preferred


def test_fallback_only_ever_returns_a_known_stack_height():
    # A returned height is used to index the _ABTilePair map, so it must always
    # be one of the four geometries that exist.
    for mtTiles in range(2, 17):
        for waveGroup in WAVE_GROUPS:
            if mtTiles % waveGroup[0] or mtTiles % waveGroup[1]:
                continue
            state = _state(mtTiles, mtTiles, waveGroup)
            for tc in ("A", "B"):
                assert _subtileStackForTLU1(state, tc, mtTiles) in _SUBTILE_STACK_SIZES
