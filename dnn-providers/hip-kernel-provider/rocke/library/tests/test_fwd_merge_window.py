# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The sweep's copy of the fwd merge-degree policy must not drift from dispatch.

``benchmark_implicit_gemm_conv`` deliberately does not import the dispatcher --
the same separation ``_MAX_GRID_DIM_Z`` is kept under, and it matters more for
the merge policy than anywhere else in that file: the sweep is the ground truth
the policy is *tuned against*, so importing the policy into the sweep would make
the measurement depend on the thing being measured.

The cost of that separation is a hand-copied ``_fwd_merge_pick``. This test is
what makes the copy safe: it imports both and asserts they agree over the whole
(groups, M, tile_m, tile_n) grid the sweep can actually reach. A change to
``_FWD_MERGE_MAX`` / ``_FWD_MERGE_MIN_CTAS`` / ``_FWD_MERGE_DEGREES`` on the
dispatch side that is not mirrored into the benchmark fails here rather than
silently producing a sweep that brackets the wrong degree.

CPU-only: both modules import nothing but stdlib at module scope.
"""

import unittest

from benchmarks.common import benchmark_implicit_gemm_conv as B
from dispatch import grouped_convolution as D

# Groups the depthwise corpus and the held-out set actually contain, plus the
# awkward ones: primes, 2-adic-poor values (72 -> caps at 8), and 1/3 which have
# no admissible degree at all.
_GROUPS = (
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    12,
    16,
    24,
    32,
    36,
    48,
    64,
    72,
    80,
    96,
    112,
    128,
    144,
    160,
    192,
    240,
    288,
    320,
    384,
    416,
    448,
    512,
    576,
    640,
    672,
    768,
    960,
    1024,
    1152,
    1280,
    1536,
    2048,
)

# M = N*Ho*Wo. Spans the occupancy cap's whole range: 0 ("unknown", the cap is
# dropped), the single-tile floor, and up to a batch-256 112x112 feature map.
_M = (0, 1, 64, 256, 1024, 3136, 12544, 50176, 82944, 200704, 3211264)


class TestFwdMergePickMirror(unittest.TestCase):
    def test_constants_match(self):
        self.assertEqual(B._FWD_MERGE_DEGREES_MIRROR, D._FWD_MERGE_DEGREES)
        self.assertEqual(B._FWD_MERGE_MAX_MIRROR, D._FWD_MERGE_MAX)
        self.assertEqual(B._FWD_MERGE_MIN_CTAS_MIRROR, D._FWD_MERGE_MIN_CTAS)

    def test_pick_matches_dispatch_over_full_grid(self):
        checked = 0
        for groups in _GROUPS:
            for m in _M:
                for tile_m in B._TILE_MN:
                    for tile_n in B._TILE_MN:
                        got = B._fwd_merge_pick(groups, m, tile_m, tile_n)
                        want = D.fwd_group_merge_for_geometry(groups, m, tile_m, tile_n)
                        self.assertEqual(
                            got,
                            want,
                            f"drift at groups={groups} M={m} "
                            f"tile={tile_m}x{tile_n}: sweep={got} dispatch={want}",
                        )
                        checked += 1
        # Guard against the loops silently collapsing to nothing.
        self.assertEqual(checked, len(_GROUPS) * len(_M) * len(B._TILE_MN) ** 2)


class TestGroupMergeWindow(unittest.TestCase):
    """The window is over *admissible* degrees, not ``_FWD_MERGE_DEGREES`` slots.

    Indexing the raw degree tuple would make the bracket collapse whenever the
    neighbours are inadmissible -- groups=72 caps at 8 and admits only (8, 4, 2),
    so a raw-tuple window around 8 would hand back 32 and 16, which the sweep
    then drops, leaving a window of one.
    """

    def _admissible(self, groups, tile_n):
        return [
            gm
            for gm in sorted(B._GROUP_MERGE_SWEEP, reverse=True)
            if groups % gm == 0 and gm <= tile_n
        ]

    def test_window_is_admissible_ordered_and_bounded(self):
        for groups in _GROUPS:
            for m in _M:
                for tile_m in B._TILE_MN:
                    for tile_n in B._TILE_MN:
                        for radius in (0, 1, 2):
                            adm = self._admissible(groups, tile_n)
                            win = B._group_merge_window(
                                groups, m, tile_m, tile_n, radius
                            )
                            ctx = (
                                f"groups={groups} M={m} tile={tile_m}x{tile_n} "
                                f"r={radius}"
                            )
                            if not adm:
                                self.assertEqual(win, (), ctx)
                                continue
                            self.assertTrue(set(win) <= set(adm), ctx)
                            self.assertEqual(len(win), len(set(win)), ctx)
                            # Contiguous descending run of the admissible list.
                            self.assertEqual(
                                list(win),
                                sorted(win, reverse=True),
                                ctx,
                            )
                            lo = adm.index(win[0])
                            self.assertEqual(list(win), adm[lo : lo + len(win)], ctx)
                            self.assertEqual(
                                len(win), min(2 * radius + 1, len(adm)), ctx
                            )

    def test_window_brackets_the_dispatch_pick(self):
        """r>=1 must contain the picked degree whenever that degree is swept.

        This is the property the pruning rests on: the sweep still contains the
        configuration dispatch would actually select, so a pruned sweep can
        never report a winner the dispatcher cannot reach.
        """
        for groups in _GROUPS:
            for m in _M:
                for tile_m in B._TILE_MN:
                    for tile_n in B._TILE_MN:
                        adm = self._admissible(groups, tile_n)
                        pick = B._fwd_merge_pick(groups, m, tile_m, tile_n)
                        if not adm or pick not in adm:
                            continue
                        for radius in (0, 1, 2):
                            win = B._group_merge_window(
                                groups, m, tile_m, tile_n, radius
                            )
                            self.assertIn(
                                pick,
                                win,
                                f"groups={groups} M={m} "
                                f"tile={tile_m}x{tile_n} r={radius} "
                                f"pick={pick} win={win}",
                            )

    def test_unmergeable_shapes_yield_empty_window(self):
        # groups=1 is not grouped at all; 3 and 5 have no degree in the sweep.
        for groups in (1, 3, 5, 7):
            self.assertEqual(B._group_merge_window(groups, 1024, 64, 64, 1), ())

    def test_tile_n_cap_is_respected(self):
        # tile_n=16 cannot host Gm=32 or 64 however divisible the group count.
        win = B._group_merge_window(2048, 12544, 64, 16, 2)
        self.assertTrue(all(gm <= 16 for gm in win), win)

    def test_known_windows(self):
        # groups=512 at a large M is ceiling-bound: cap is _FWD_MERGE_MAX=32.
        self.assertEqual(B._group_merge_window(512, 12544, 64, 64, 1), (64, 32, 16))
        self.assertEqual(B._group_merge_window(512, 12544, 64, 64, 0), (32,))
        self.assertEqual(
            B._group_merge_window(512, 12544, 64, 64, 2), (64, 32, 16, 8, 4)
        )
        # groups=72 admits only 8/4/2 -- the window must not name 32 or 16.
        self.assertEqual(B._group_merge_window(72, 82944, 64, 64, 1), (8, 4, 2))
        # groups=448 at M=64 is occupancy-bound; the pick falls through to 1,
        # so the window anchors on the smallest admissible degree.
        self.assertEqual(B._group_merge_window(448, 64, 64, 64, 1), (8, 4, 2))


if __name__ == "__main__":
    unittest.main()
