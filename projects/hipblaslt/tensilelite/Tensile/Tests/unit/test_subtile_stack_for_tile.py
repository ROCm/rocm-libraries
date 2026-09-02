# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.SolutionStructs.Validators.Subtile import (
    _subtileStackFullLine,
    _subtileStackLadder,
    _subtileStackForTile,
    subtileStackForTLU1,
    subtileTLU1StackReason,
)


MI_M = 16
FP4_BPE = 0.5
B16_BPE = 2.0
# fp4 fills a 128B line at 16 MFMA-M tiles, bf16 at 4.
FP4_FULL_LINE = _subtileStackFullLine(MI_M, FP4_BPE)
B16_FULL_LINE = _subtileStackFullLine(MI_M, B16_BPE)


def test_full_line_is_derived_from_the_dtype():
    assert FP4_FULL_LINE == 16
    assert B16_FULL_LINE == 4
    # The fp4 ladder must reproduce the hardcoded table it replaced.
    assert _subtileStackLadder(FP4_FULL_LINE) == (16, 8, 4, 2)
    assert _subtileStackLadder(B16_FULL_LINE) == (4, 2)


# (MFMA-M tiles in the macro tile, expected stack). At MatrixInstM=16 the tile
# is 16x the first column, so 12 -> MT192, 14 -> MT224, 16 -> MT256.
CASES = [
    (1, 2),     # degenerate: no rounding considered at all
    (2, 2),     # divides exactly, and is already the minimum
    (3, 4),     # pads 3 -> 4
    (4, 4),     # divides exactly; 8 would not cover a 4-tile dim any better
    (5, 8),     # MT80:  pads 5 -> 8
    (6, 8),     # pads 6 -> 8
    (7, 8),     # pads 7 -> 8
    (8, 8),     # divides exactly
    (9, 16),    # MT144: pads 9 -> 16
    (10, 16),   # MT160: pads 10 -> 16, the case a 2-tile stack served worst
    (11, 16),   # MT176: pads 11 -> 16
    (12, 16),   # MT192: pads 12 -> 16
    (13, 16),   # MT208
    (14, 16),   # MT224: pads 14 -> 16
    (15, 16),   # MT240
    (16, 16),   # MT256: divides exactly, one full cache line
]


# bf16's cache line caps the stack at 4, so it rounds up only while the rounded
# stack still covers the tile: 3 -> 4, but 5..7 fall back to the exact divisor.
B16_CASES = [
    (1, 2), (2, 2), (3, 4), (4, 4),
    (5, 2), (6, 2), (7, 2), (8, 4),
    (9, 2), (10, 2), (11, 2), (12, 4),
    (13, 2), (14, 2), (15, 2), (16, 4),
]


@pytest.mark.parametrize("mtTiles,expected", CASES)
def test_stack_for_tile(mtTiles, expected):
    assert _subtileStackForTile(mtTiles, FP4_FULL_LINE) == expected


@pytest.mark.parametrize("mtTiles,expected", B16_CASES)
def test_stack_for_tile_b16(mtTiles, expected):
    assert _subtileStackForTile(mtTiles, B16_FULL_LINE) == expected


@pytest.mark.parametrize("mtTiles", [18, 20, 24, 32, 40, 48, 64])
def test_stack_above_the_full_line_never_strands_a_trailing_strip(mtTiles):
    # Even tiles only: an odd one has no power-of-two divisor and falls back to
    # _SUBTILE_STACK_MIN, but MX rejects odd MIWaveTile before a layout is built.
    stack = _subtileStackForTile(mtTiles, FP4_FULL_LINE)
    assert stack <= FP4_FULL_LINE
    assert stack >= mtTiles or mtTiles % stack == 0


@pytest.mark.parametrize("fullLine", [FP4_FULL_LINE, B16_FULL_LINE])
def test_rounding_only_happens_when_one_strip_covers_the_tile(fullLine):
    ladder = _subtileStackLadder(fullLine)
    for mtTiles in range(1, 17):
        exact = next((s for s in ladder if mtTiles % s == 0), 2)
        stack = _subtileStackForTile(mtTiles, fullLine)
        if stack > exact:
            assert stack >= mtTiles


@pytest.mark.parametrize("fullLine", [FP4_FULL_LINE, B16_FULL_LINE])
def test_stack_never_shrinks_below_an_exact_divisor(fullLine):
    # Rounding may only move the stack up; a tile that divides a taller stack
    # exactly must never be given a shorter one.
    ladder = _subtileStackLadder(fullLine)
    for mtTiles in range(1, 17):
        exact = next((s for s in ladder if mtTiles % s == 0), 2)
        assert _subtileStackForTile(mtTiles, fullLine) >= exact


# --- geometry-aware fallback -------------------------------------------------
#
# _subtileStackForTile picks on cache-line utilization alone. subtileStackForTLU1
# additionally backs off to a shorter stack when the preferred one cannot be laid
# out for the wave group, instead of leaving the solution to be rejected.

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
        assert _subtileStackForTile(12, FP4_FULL_LINE) == 16
        assert subtileStackForTLU1(state, tc, 12, FP4_BPE) == 2
        assert subtileTLU1StackReason(state, tc, 12, 2, FP4_BPE) is None


@pytest.mark.parametrize("waveGroup", [(1, 4), (4, 1)])
def test_mt192x192_keeps_rejecting_the_three_tile_wave_groups(waveGroup):
    # These give a wave 3 of the 12 tiles, and no power-of-two stack tiles 3:
    # 2 and 4 straddle, 8 and 16 are not multiples of 3. No fallback exists, so
    # the preferred stack is returned and the caller still rejects.
    state = _state(12, 12, waveGroup)
    tc = "A" if waveGroup[0] == 4 else "B"
    stack = subtileStackForTLU1(state, tc, 12, FP4_BPE)
    assert stack == _subtileStackForTile(12, FP4_FULL_LINE)
    assert subtileTLU1StackReason(state, tc, 12, stack, FP4_BPE) is not None


def test_fallback_never_moves_a_stack_that_already_works():
    # The whole safety argument for the fallback: it may only change geometries
    # that are rejected today. If the preferred stack is viable it must be kept,
    # so no currently-valid solution changes its LDS layout.
    for mtTiles in range(2, 17):
        for waveGroup in WAVE_GROUPS:
            if mtTiles % waveGroup[0] or mtTiles % waveGroup[1]:
                continue
            state = _state(mtTiles, mtTiles, waveGroup)
            preferred = _subtileStackForTile(mtTiles, FP4_FULL_LINE)
            for tc in ("A", "B"):
                if subtileTLU1StackReason(state, tc, mtTiles, preferred, FP4_BPE) is None:
                    assert subtileStackForTLU1(state, tc, mtTiles, FP4_BPE) == preferred


def test_strip_sharing_rules_stay_gfx950_only():
    # _validateSubtileGRKPartition polices strip sharing on gfx950 alone. The
    # chooser has to use the same gate, or it would reject on other ISAs a
    # geometry the validator there would accept.
    tiles, waveGroup = 12, (2, 2)
    gfx950 = _state(tiles, tiles, waveGroup)
    other = _state(tiles, tiles, waveGroup, isa=(12, 5, 0))
    preferred = _subtileStackForTile(tiles, FP4_FULL_LINE)
    assert subtileTLU1StackReason(gfx950, "A", tiles, preferred, FP4_BPE) is not None
    assert subtileTLU1StackReason(other, "A", tiles, preferred, FP4_BPE) is None
    assert subtileStackForTLU1(other, "A", tiles, FP4_BPE) == preferred


def test_fallback_only_ever_returns_a_known_stack_height():
    # A returned height is used to index the _ABTilePair map, so it must always
    # be one of the four geometries that exist.
    for mtTiles in range(2, 17):
        for waveGroup in WAVE_GROUPS:
            if mtTiles % waveGroup[0] or mtTiles % waveGroup[1]:
                continue
            state = _state(mtTiles, mtTiles, waveGroup)
            for tc in ("A", "B"):
                assert subtileStackForTLU1(state, tc, mtTiles, FP4_BPE) in _subtileStackLadder(FP4_FULL_LINE)
