# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Independent oracle for the two Stream-K K-split work mappings.

Pure Python: it imports nothing from Tensile and needs no GPU and no build. It
re-derives both mappings from first principles so that a regression in the
generator cannot also silently "fix" the expectation.

The two mappings
----------------
* ``global_range``  -- the historical "first-E workgroups get one extra
  iteration" mapping. It splits the *global* iteration space
  ``[0, skTiles * itersPerTile)`` evenly across ``skGrid`` workgroups,
  ignoring where output-tile boundaries fall.
* ``pertile_range`` -- the USO (uniform summation order) per-tile mapping,
  defined only when ``skGrid % skTiles == 0``: each tile gets exactly
  ``F = skGrid // skTiles`` workgroups that split *that tile's*
  ``itersPerTile`` iterations, so no workgroup ever straddles a tile boundary.

Worked divergence example (pinned by ``test_worked_divergence_example``)
-----------------------------------------------------------------------
  tiles=128, skGrid=256, I=5 => W=2, E=128, F=2, I%F=1.
  Global: w0=[0,3), w1=[3,6) -- w1 straddles the tile boundary at 5.
  Per-tile: w0=[0,3), w1=[3,5), w2=[5,8).

That is the regression vector: the global mapping gives workgroup 1 iterations
3 and 4 of tile 0 *and* iteration 0 of tile 1, which the per-tile mapping never
does.
"""

import pytest


# ---------------------------------------------------------------------------
# Oracle. `tiles` is unused in the bodies; it stays in the signature so both
# mappings share one call form.
# ---------------------------------------------------------------------------


def global_range(w, tiles, I, skGrid, skTiles):      # historical first-E mapping
    W = skTiles * I // skGrid
    E = skTiles * I - W * skGrid
    if w < E:  start = w * (W + 1);  end = start + W + 1
    else:      start = w * W + E;    end = start + W
    return start, end


def pertile_range(w, tiles, I, skGrid, skTiles):     # USO per-tile mapping
    F = skGrid // skTiles
    q, s = divmod(w, F)
    W = skTiles * I // skGrid
    r = I % F
    start = q * I + s * W + min(s, r)
    return start, start + W + (1 if s < r else 0)


def pertile_active(skTiles, skGrid, uso):
    """Gate on the per-tile mapping; mirrors the device predicate
    ``perTileActive`` (StreamK.py, ContractionSolution.cpp).

    ``skTiles == 0`` is the forceDPOnly short-circuit: no Stream-K work to
    split. ``skGrid % skTiles != 0`` means skTiles does not divide skGrid, so
    there is no integer F = skGrid // skTiles and the per-tile mapping is
    undefined.
    """
    return uso and skTiles != 0 and skGrid % skTiles == 0


# ---------------------------------------------------------------------------
# Vector table
# ---------------------------------------------------------------------------

DIFFER = "differ"
AGREE = "agree"
GATE_OFF = "gate-off"
DP_ONLY = "dp-only"

# (id, skTiles, skGrid, itersPerTile, F, verdict)
CASES = [
    ("regression-128x256-I5", 128, 256, 5, 2, DIFFER),
    ("divisible-128x256-I4", 128, 256, 4, 2, AGREE),
    ("nondivisible-100x256-I7", 100, 256, 7, None, GATE_OFF),
    ("F1-256x256-I9", 256, 256, 9, 1, AGREE),
    ("96x384-I5", 96, 384, 5, 4, DIFFER),
    ("forceDPOnly", 0, 256, 5, None, DP_ONLY),
]


def _params(*verdicts):
    """Select vector-table rows by verdict, as pytest parametrize args."""
    return [
        pytest.param(t, g, i, f, v, id=name)
        for (name, t, g, i, f, v) in CASES
        if v in verdicts
    ]


def _ranges(fn, skTiles, skGrid, I):
    return [fn(w, skTiles, I, skGrid, skTiles) for w in range(skGrid)]


def _assert_exact_partition(ranges, total):
    """Contiguous, pairwise disjoint, and exactly covering [0, total)."""
    assert ranges, "expected at least one workgroup range"
    assert ranges[0][0] == 0, f"first range must start at 0, got {ranges[0]}"
    assert ranges[-1][1] == total, f"last range must end at {total}, got {ranges[-1]}"

    prev_end = 0
    for w, (start, end) in enumerate(ranges):
        assert start <= end, f"w={w}: inverted range {(start, end)}"
        assert start == prev_end, (
            f"w={w}: range {(start, end)} is not contiguous with previous end {prev_end}"
        )
        assert 0 <= start <= total and 0 <= end <= total, (
            f"w={w}: range {(start, end)} escapes [0, {total})"
        )
        prev_end = end

    # Contiguity + start 0 + end total already implies disjointness, but assert
    # it directly so a future edit to the oracle cannot lose the property.
    covered = []
    for start, end in ranges:
        covered.extend(range(start, end))
    assert len(covered) == total, f"covered {len(covered)} iterations, expected {total}"
    assert len(set(covered)) == total, "workgroup ranges overlap"
    assert set(covered) == set(range(total)), "workgroup ranges do not cover [0, total)"


# ---------------------------------------------------------------------------
# The pinned worked example
# ---------------------------------------------------------------------------


class TestWorkedExample:
    def test_worked_divergence_example(self):
        tiles, skGrid, I = 128, 256, 5
        W = tiles * I // skGrid
        E = tiles * I - W * skGrid
        F = skGrid // tiles

        assert (W, E, F, I % F) == (2, 128, 2, 1)

        assert global_range(0, tiles, I, skGrid, tiles) == (0, 3)
        assert global_range(1, tiles, I, skGrid, tiles) == (3, 6)

        assert pertile_range(0, tiles, I, skGrid, tiles) == (0, 3)
        assert pertile_range(1, tiles, I, skGrid, tiles) == (3, 5)
        assert pertile_range(2, tiles, I, skGrid, tiles) == (5, 8)

    def test_worked_example_global_straddles_tile_boundary(self):
        tiles, skGrid, I = 128, 256, 5
        start, end = global_range(1, tiles, I, skGrid, tiles)
        # Tile boundary sits at iteration 5; w1 = [3, 6) crosses it.
        assert start // I != (end - 1) // I, (
            "the regression vector requires the global mapping to straddle a tile boundary"
        )

        pstart, pend = pertile_range(1, tiles, I, skGrid, tiles)
        assert pstart // I == (pend - 1) // I, (
            "the per-tile mapping must never straddle a tile boundary"
        )


# ---------------------------------------------------------------------------
# Partition properties
# ---------------------------------------------------------------------------


class TestGlobalMapping:
    @pytest.mark.parametrize(
        "skTiles, skGrid, I, F, verdict", _params(DIFFER, AGREE, GATE_OFF)
    )
    def test_global_mapping_is_an_exact_partition(self, skTiles, skGrid, I, F, verdict):
        ranges = _ranges(global_range, skTiles, skGrid, I)
        _assert_exact_partition(ranges, skTiles * I)


class TestPerTileMapping:
    @pytest.mark.parametrize("skTiles, skGrid, I, F, verdict", _params(DIFFER, AGREE))
    def test_pertile_mapping_is_an_exact_partition(self, skTiles, skGrid, I, F, verdict):
        assert skGrid % skTiles == 0
        assert skGrid // skTiles == F
        ranges = _ranges(pertile_range, skTiles, skGrid, I)
        _assert_exact_partition(ranges, skTiles * I)

    @pytest.mark.parametrize("skTiles, skGrid, I, F, verdict", _params(DIFFER, AGREE))
    def test_pertile_mapping_never_straddles_a_tile(self, skTiles, skGrid, I, F, verdict):
        for w in range(skGrid):
            start, end = pertile_range(w, skTiles, I, skGrid, skTiles)
            if end == start:
                continue  # empty range owns no iterations, cannot straddle
            assert start // I == (end - 1) // I, (
                f"w={w}: per-tile range {(start, end)} straddles a tile boundary "
                f"(itersPerTile={I})"
            )

    @pytest.mark.parametrize("skTiles, skGrid, I, F, verdict", _params(DIFFER, AGREE))
    def test_pertile_gate_is_active(self, skTiles, skGrid, I, F, verdict):
        assert pertile_active(skTiles, skGrid, uso=True) is True
        # The gate is off whenever USO itself is off, regardless of shape.
        assert pertile_active(skTiles, skGrid, uso=False) is False


# ---------------------------------------------------------------------------
# Divergence / agreement verdicts
# ---------------------------------------------------------------------------


class TestVerdicts:
    @pytest.mark.parametrize("skTiles, skGrid, I, F, verdict", _params(DIFFER))
    def test_mappings_actually_differ(self, skTiles, skGrid, I, F, verdict):
        # A bug that collapsed the two mappings onto each other would make this
        # suite pass while testing nothing.
        assert I % F != 0, "a differing row requires itersPerTile % F != 0"
        diffs = [
            (
                w,
                global_range(w, skTiles, I, skGrid, skTiles),
                pertile_range(w, skTiles, I, skGrid, skTiles),
            )
            for w in range(skGrid)
            if global_range(w, skTiles, I, skGrid, skTiles)
            != pertile_range(w, skTiles, I, skGrid, skTiles)
        ]
        assert diffs, (
            f"expected the two mappings to diverge for skTiles={skTiles}, "
            f"skGrid={skGrid}, itersPerTile={I}, but they were identical for all "
            f"{skGrid} workgroups"
        )

    @pytest.mark.parametrize("skTiles, skGrid, I, F, verdict", _params(DIFFER))
    def test_global_mapping_straddles_somewhere(self, skTiles, skGrid, I, F, verdict):
        straddlers = [
            w
            for w in range(skGrid)
            for (start, end) in [global_range(w, skTiles, I, skGrid, skTiles)]
            if end > start and start // I != (end - 1) // I
        ]
        assert straddlers, (
            "on a differing row the global mapping must straddle at least one "
            "tile boundary -- that is the whole reason the per-tile mapping exists"
        )

    @pytest.mark.parametrize("skTiles, skGrid, I, F, verdict", _params(AGREE))
    def test_mappings_agree_everywhere(self, skTiles, skGrid, I, F, verdict):
        assert I % F == 0, "an agreeing row requires itersPerTile % F == 0"
        for w in range(skGrid):
            g = global_range(w, skTiles, I, skGrid, skTiles)
            p = pertile_range(w, skTiles, I, skGrid, skTiles)
            assert g == p, (
                f"w={w}: expected the mappings to agree for skTiles={skTiles}, "
                f"skGrid={skGrid}, itersPerTile={I}, got global={g} per-tile={p}"
            )


# ---------------------------------------------------------------------------
# Gate short-circuits
# ---------------------------------------------------------------------------


class TestGateOff:
    @pytest.mark.parametrize("skTiles, skGrid, I, F, verdict", _params(GATE_OFF))
    def test_pertile_gate_never_fires(self, skTiles, skGrid, I, F, verdict):
        assert skGrid % skTiles != 0
        assert pertile_active(skTiles, skGrid, uso=True) is False
        assert pertile_active(skTiles, skGrid, uso=False) is False

    @pytest.mark.parametrize("skTiles, skGrid, I, F, verdict", _params(GATE_OFF))
    def test_global_mapping_still_tiles_correctly(self, skTiles, skGrid, I, F, verdict):
        ranges = _ranges(global_range, skTiles, skGrid, I)
        _assert_exact_partition(ranges, skTiles * I)


class TestForceDPOnly:
    @pytest.mark.parametrize("skTiles, skGrid, I, F, verdict", _params(DP_ONLY))
    def test_no_streamk_work_short_circuits_the_gate(
        self, skTiles, skGrid, I, F, verdict
    ):
        assert skTiles == 0
        assert pertile_active(0, skGrid, True) is False
        assert pertile_active(0, skGrid, False) is False

    def test_force_dp_only_has_no_streamk_iterations(self):
        # forceDPOnly means skTiles == 0, so the Stream-K iteration space is
        # empty and neither mapping is ever consulted.
        skTiles, I = 0, 5
        assert skTiles * I == 0
