# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Split-K selection for wgrad: the ungrouped contract and the grouped path.

``select_split_k_wgrad`` is shared with the dgrad builder, which reaches it
through the benchmark driver's ``--split-k -1``. The ungrouped branch therefore
has to stay bit-for-bit stable independently of anything the grouped path does,
and that is what most of this file pins.
"""

from __future__ import annotations

import math
import unittest

from rocke.helpers.split_k import (
    MAX_GRID_DIM_Z,
    WGRAD_ASSUMED_WAVES_PER_CU,
    WGRAD_TARGET_WAVES_PER_CU,
    select_split_k_wgrad,
)

# Per-group wgrad GEMM dims for the grouped conv shapes the library benchmarks
# cover: (wg_M, wg_N, wg_K, groups). All are depthwise (cpg == kpg == 1) except
# ``grouped_cpg8``, which is grouped-but-not-depthwise and is the regime a
# groups-aware change is most likely to regress.
_SHAPES = {
    #                wg_M  wg_N              wg_K  groups
    "dw_g192_s2": (1, 9, 42 * 60 * 80, 192),
    "dw_g256_s1": (1, 9, 42 * 60 * 80, 256),
    "dw_g3_11x11": (1, 121, 42 * 480 * 640, 3),
    "dw_g512_s1": (1, 9, 42 * 30 * 40, 512),
    "grouped_cpg8": (8, 72, 42 * 60 * 80, 32),
    "ungrouped": (256, 2304, 64 * 28 * 28, 1),
    # Shallow wg_K: the cap (min(wg_K, z-limit)) binds, which is where a clamp
    # applied after the XCD snap would silently undo the alignment.
    "dw_shallow_k_g8": (1, 9, 4, 8),
    "dw_shallow_k_g16": (1, 9, 5, 16),
    "dw_shallow_k_g8b": (1, 9, 12, 8),
}

_TILES = [(64, 64, 64), (16, 16, 32), (32, 32, 32), (128, 128, 32), (16, 128, 32)]


def _pick(shape, tile, **kw):
    wg_M, wg_N, wg_K, groups = _SHAPES[shape]
    return select_split_k_wgrad(
        wg_M=wg_M,
        wg_N=wg_N,
        wg_K=wg_K,
        tile_m=tile[0],
        tile_n=tile[1],
        tile_k=tile[2],
        **kw,
    )


class TestUngroupedUnchanged(unittest.TestCase):
    """The legacy branch. These values predate the groups-aware path."""

    def test_default_groups_is_the_ck_formula(self):
        # floor(waves_per_cu * num_cus / base_grid), num_cus(gfx950) = 256.
        for tile, expected in [((64, 64, 64), 3), ((128, 128, 32), 14)]:
            d = _pick("ungrouped", tile)
            base_grid = math.ceil(256 / tile[0]) * math.ceil(2304 / tile[1])
            self.assertEqual(d.base_grid, base_grid)
            self.assertEqual(
                d.split_k,
                max(1, (WGRAD_ASSUMED_WAVES_PER_CU * 256) // base_grid),
                f"ungrouped decision moved at tile {tile}",
            )
            self.assertEqual(d.split_k, expected)

    def test_groups_one_ignores_block_size(self):
        # block_size only feeds the grouped occupancy target. If it ever leaks
        # into the ungrouped branch, dgrad geometry moves silently.
        for tile in _TILES:
            ref = _pick("ungrouped", tile).split_k
            for bs in (64, 128, 256, 512, 1024):
                self.assertEqual(
                    _pick("ungrouped", tile, groups=1, block_size=bs).split_k,
                    ref,
                    f"block_size={bs} changed the ungrouped decision at {tile}",
                )

    def test_dgrad_shaped_call_is_stable(self):
        # The dgrad caller passes M=N*Hi*Wi, N=cpg, K=Y*X*kpg and no groups.
        # Grouped dgrad (G=4, C=64, K=128) is live and GPU-tested; these are
        # the degrees it resolves today.
        for tile_m, expected in [(32, 5), (64, 10), (128, 20), (256, 39)]:
            d = select_split_k_wgrad(
                wg_M=4 * 28 * 28,
                wg_N=64 // 4,
                wg_K=3 * 3 * (128 // 4),
                tile_m=tile_m,
                tile_n=32,
                tile_k=32,
            )
            self.assertEqual(d.split_k, expected, f"dgrad moved at tile_m={tile_m}")


class TestGroupedPath(unittest.TestCase):
    def test_groups_factor_enters_the_grid(self):
        # Without the groups term a depthwise base_grid of 1 sized the degree
        # for a single CTA and asked for hundreds.
        for shape in ("dw_g192_s2", "dw_g256_s1", "dw_g512_s1"):
            groups = _SHAPES[shape][3]
            ungrouped = _pick(shape, (64, 64, 64)).split_k
            grouped = _pick(shape, (64, 64, 64), groups=groups).split_k
            self.assertLess(grouped, ungrouped)

    def test_total_ctas_stay_within_capacity(self):
        # Floor division is the invariant: rounding up crosses a CTA
        # quantisation cliff. Snap-up is the one sanctioned exception, and only
        # fires when the raw degree is below a single XCD step.
        for shape, (_, _, _, groups) in _SHAPES.items():
            if groups <= 1:
                continue
            for tile in _TILES:
                for bs in (64, 256):
                    d = _pick(shape, tile, groups=groups, block_size=bs)
                    ctas = d.base_grid * groups * d.split_k
                    xcds = 8
                    step = xcds // math.gcd(d.base_grid, xcds)
                    if d.split_k > step:
                        self.assertLessEqual(
                            ctas,
                            d.target_ctas,
                            f"{shape} {tile} bs={bs}: {ctas} CTAs over capacity",
                        )

    def test_xcd_snap_aligns_the_group_stride(self):
        # Consecutive conv groups sit base_grid*split_k apart in flat workgroup
        # id; that stride is what has to clear the XCD count.
        for shape, (_, _, _, groups) in _SHAPES.items():
            if groups <= 1:
                continue
            for tile in _TILES:
                d = _pick(shape, tile, groups=groups, block_size=256)
                stride = d.base_grid * d.split_k
                step = 8 // math.gcd(d.base_grid, 8)
                cap = min(_SHAPES[shape][2], MAX_GRID_DIM_Z // groups)
                if step > cap:
                    # One XCD step does not fit under the cap, so alignment is
                    # unreachable; the degree just has to stay legal.
                    self.assertLessEqual(d.split_k, cap)
                    continue
                if d.split_k > 1:
                    self.assertEqual(
                        stride % 8,
                        0,
                        f"{shape} {tile}: group stride {stride} not XCD-aligned",
                    )

    def test_cap_is_applied_before_the_snap(self):
        # Clamping after snapping would pull the degree back off a multiple of
        # the step. Every shape whose cap admits at least one full step must
        # come back aligned, shallow wg_K included.
        for shape, (_, _, wg_K, groups) in _SHAPES.items():
            if groups <= 1:
                continue
            for tile in _TILES:
                d = _pick(shape, tile, groups=groups, block_size=256)
                cap = min(wg_K, MAX_GRID_DIM_Z // groups)
                self.assertLessEqual(d.split_k, cap, f"{shape} {tile} over cap")
                step = 8 // math.gcd(d.base_grid, 8)
                if step <= cap and d.split_k > 1:
                    self.assertEqual(
                        (d.base_grid * d.split_k) % 8,
                        0,
                        f"{shape} {tile}: cap undid the XCD snap",
                    )

    def test_never_overflows_grid_dim_z(self):
        # groups * split_k rides gridDim.z, a 16-bit dispatch-packet field.
        # Overflowing it fails the launch with hipErrorInvalidValue.
        for shape, (_, _, _, groups) in _SHAPES.items():
            for tile in _TILES:
                for bs in (64, 128, 256, 512):
                    d = _pick(shape, tile, groups=groups, block_size=bs)
                    self.assertGreaterEqual(d.split_k, 1)
                    self.assertLessEqual(
                        groups * d.split_k,
                        MAX_GRID_DIM_Z,
                        f"{shape} {tile} bs={bs} overflows gridDim.z",
                    )

    def test_block_size_scales_the_occupancy_target(self):
        # The target is in waves, so a tile with half the waves per block may
        # have twice the blocks per CU.
        wide = _pick("grouped_cpg8", (64, 64, 64), groups=32, block_size=256)
        narrow = _pick("grouped_cpg8", (64, 64, 64), groups=32, block_size=64)
        self.assertGreaterEqual(narrow.split_k, wide.split_k)
        self.assertEqual(WGRAD_TARGET_WAVES_PER_CU % 4, 0)

    def test_single_xcd_arch_does_not_snap(self):
        # gfx1151 has one shader-engine group, so there is no stride to align.
        d = _pick(
            "dw_g512_s1", (16, 16, 32), groups=512, block_size=256, arch="gfx1151"
        )
        self.assertGreaterEqual(d.split_k, 1)
        self.assertIn("multiple of 1", d.reason)


if __name__ == "__main__":
    unittest.main()
