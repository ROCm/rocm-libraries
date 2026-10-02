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
(groups, M, tile_m, tile_n, Y, X, stride) grid the sweep can actually reach. The
policy is a cost model now rather than a pair of caps, so the drift this guards
against is no longer only a changed constant -- a reordered ``_FWD_MERGE_CONSTS``
tuple or a term dropped from one body and not the other fails here rather than
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

# M = N*Ho*Wo. Spans the launch range the CTA-count term can see: 0 ("unknown",
# clamped to 1), the single-tile floor, and up to a batch-256 112x112 map.
_M = (0, 1, 64, 256, 1024, 3136, 12544, 50176, 82944, 200704, 3211264)

# (Y, X, stride). The filter area drives both the K-padding recovery and the
# footprint brake, so the grid has to span it: pointwise (which the gate rejects
# outright), the 3x3/5x5/7x7 depthwise mainstream at both strides, the
# asymmetric separable pair, and a filter big enough that merging never pays.
_YXS = (
    (1, 1, 1),
    (3, 3, 1),
    (3, 3, 2),
    (5, 5, 1),
    (5, 5, 2),
    (7, 7, 1),
    (7, 1, 1),
    (1, 7, 1),
    (15, 15, 1),
)


class TestFwdMergePickMirror(unittest.TestCase):
    def test_constants_match(self):
        self.assertEqual(B._FWD_MERGE_DEGREES_MIRROR, D._FWD_MERGE_DEGREES)
        self.assertEqual(
            B._FWD_MERGE_CONSTS_MIRROR,
            (
                D._FWD_MERGE_B_ALOAD_ISSUE,
                D._FWD_MERGE_W_MEMORY,
                D._FWD_MERGE_R_REUSE,
                D._FWD_MERGE_D_CTA_FIXED,
                D._FWD_MERGE_CUPAR,
                D._FWD_MERGE_C_FOOTPRINT,
                D._FWD_MERGE_A_UTIL,
                D._FWD_MERGE_Q_BRAKE,
            ),
        )
        self.assertEqual(B._FWD_MERGE_LINE_BYTES_MIRROR, D._FWD_MERGE_LINE_BYTES)
        self.assertEqual(B._FWD_MERGE_VEC_BYTES_MIRROR, D._FWD_MERGE_VEC_BYTES)

    def test_pick_matches_dispatch_over_full_grid(self):
        checked = 0
        for groups in _GROUPS:
            for m in _M:
                for tile_m in B._TILE_MN:
                    for tile_n in B._TILE_MN:
                        for y, x, stride in _YXS:
                            got = B._fwd_merge_pick(
                                groups, m, tile_m, tile_n, y, x, stride
                            )
                            want = D.fwd_group_merge_for_geometry(
                                groups,
                                m,
                                tile_m,
                                tile_n,
                                y=y,
                                x=x,
                                stride=stride,
                            )
                            self.assertEqual(
                                got,
                                want,
                                f"drift at groups={groups} M={m} "
                                f"tile={tile_m}x{tile_n} {y}x{x}s{stride}: "
                                f"sweep={got} dispatch={want}",
                            )
                            checked += 1
        # Guard against the loops silently collapsing to nothing.
        self.assertEqual(
            checked, len(_GROUPS) * len(_M) * len(B._TILE_MN) ** 2 * len(_YXS)
        )

    def test_pick_matches_dispatch_at_fp32(self):
        """esize is a policy input, not a constant -- mirror it too.

        ``support()`` admits only fp16/bf16 today so dispatch always passes 2,
        but the sweep passes 4 for fp32 shapes and the two bodies have to stay
        one body.
        """
        for groups in (64, 96, 512, 1280):
            for m in (1024, 50176):
                for y, x, stride in ((3, 3, 1), (5, 5, 2), (7, 7, 1)):
                    got = B._fwd_merge_pick(groups, m, 64, 64, y, x, stride, 4)
                    want = D.fwd_group_merge_for_geometry(
                        groups, m, 64, 64, y=y, x=x, stride=stride, esize=4
                    )
                    self.assertEqual(got, want, f"{groups} {m} {y}x{x}s{stride}")

    def test_pointwise_never_merges(self):
        """Y==X==1 is outside the emitter's fwd merge gate -- the policy agrees.

        Not a cost comparison: the cost model has no term that would rule 1x1
        out, so the gate is reproduced as an explicit early return and this is
        what pins it.
        """
        for groups in _GROUPS:
            for m in (1, 12544, 3211264):
                self.assertEqual(B._fwd_merge_pick(groups, m, 64, 64, 1, 1, 1), 1)
                self.assertEqual(
                    D.fwd_group_merge_for_geometry(
                        groups, m, 64, 64, y=1, x=1, stride=1
                    ),
                    1,
                )


class TestFwdMergePickShape(unittest.TestCase):
    """Properties the cost model must hold regardless of how it is refitted.

    The eight constants are empirical and will move when the corpus grows. These
    are the structural facts a refit must not break -- a fit that violates one of
    them is mis-specified, not merely retuned.
    """

    def test_pick_is_admissible(self):
        for groups in _GROUPS:
            for tile_n in B._TILE_MN:
                for y, x, stride in _YXS:
                    gm = B._fwd_merge_pick(groups, 12544, 64, tile_n, y, x, stride)
                    ctx = f"groups={groups} tile_n={tile_n} {y}x{x}s{stride} -> {gm}"
                    self.assertIn(gm, B._FWD_MERGE_DEGREES_MIRROR, ctx)
                    self.assertLessEqual(gm, tile_n, ctx)
                    self.assertEqual(groups % gm, 0, ctx)

    def test_large_filters_merge_no_harder_than_small_ones(self):
        """The footprint brake is linear in Y*X, so the degree must not rise
        with the filter area at a fixed launch."""
        for groups in (128, 512, 1280):
            for m in (3136, 50176):
                picks = [
                    B._fwd_merge_pick(groups, m, 64, 64, y, y, 1) for y in (3, 5, 7, 15)
                ]
                self.assertEqual(
                    picks,
                    sorted(picks, reverse=True),
                    f"groups={groups} M={m}: {picks}",
                )

    def test_tiny_filters_merge_at_least_as_hard_as_the_mainstream(self):
        # 3x3 wastes 7.1x of a 64-wide K tile unmerged; it must merge at least
        # as far as the 7x7 that has far less padding to recover.
        for groups in (128, 512, 1280):
            for m in (3136, 50176, 3211264):
                self.assertGreaterEqual(
                    B._fwd_merge_pick(groups, m, 64, 64, 3, 3, 1),
                    B._fwd_merge_pick(groups, m, 64, 64, 7, 7, 1),
                    f"groups={groups} M={m}",
                )


class TestGroupMergeWindow(unittest.TestCase):
    """The window is over *admissible* degrees, not ``_FWD_MERGE_DEGREES`` slots.

    Indexing the raw degree tuple would make the bracket collapse whenever the
    neighbours are inadmissible -- groups=72 admits only (8, 4, 2), so a raw-tuple
    window around 8 would hand back 32 and 16, which the sweep then drops, leaving
    a window of one.
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
                        for y, x, stride in _YXS:
                            for radius in (0, 1, 2):
                                adm = self._admissible(groups, tile_n)
                                win = B._group_merge_window(
                                    groups, m, tile_m, tile_n, radius, y, x, stride
                                )
                                ctx = (
                                    f"groups={groups} M={m} tile={tile_m}x{tile_n} "
                                    f"{y}x{x}s{stride} r={radius}"
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
                                self.assertEqual(
                                    list(win), adm[lo : lo + len(win)], ctx
                                )
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
                        for y, x, stride in _YXS:
                            adm = self._admissible(groups, tile_n)
                            pick = B._fwd_merge_pick(
                                groups, m, tile_m, tile_n, y, x, stride
                            )
                            if not adm or pick not in adm:
                                continue
                            for radius in (0, 1, 2):
                                win = B._group_merge_window(
                                    groups, m, tile_m, tile_n, radius, y, x, stride
                                )
                                self.assertIn(
                                    pick,
                                    win,
                                    f"groups={groups} M={m} "
                                    f"tile={tile_m}x{tile_n} {y}x{x}s{stride} "
                                    f"r={radius} pick={pick} win={win}",
                                )

    def test_unmergeable_shapes_yield_empty_window(self):
        # groups=1 is not grouped at all; 3, 5 and 7 have no degree in the sweep.
        for groups in (1, 3, 5, 7):
            self.assertEqual(
                B._group_merge_window(groups, 1024, 64, 64, 1, 3, 3, 1), ()
            )

    def test_tile_n_cap_is_respected(self):
        # tile_n=16 cannot host Gm=32 or 64 however divisible the group count.
        win = B._group_merge_window(2048, 12544, 64, 16, 2, 3, 3, 1)
        self.assertTrue(all(gm <= 16 for gm in win), win)

    def test_known_windows(self):
        # A 3x3 at a full-machine launch merges hard; 7x7 at the same launch
        # gives back two doublings to the footprint brake.
        self.assertEqual(B._group_merge_window(512, 12544, 64, 64, 0, 3, 3, 1), (32,))
        self.assertEqual(
            B._group_merge_window(512, 12544, 64, 64, 1, 3, 3, 1), (64, 32, 16)
        )
        self.assertEqual(
            B._group_merge_window(512, 12544, 64, 64, 2, 3, 3, 1),
            (64, 32, 16, 8, 4),
        )
        self.assertEqual(
            B._group_merge_window(512, 12544, 64, 64, 1, 7, 7, 1), (32, 16, 8)
        )
        # groups=72 admits only 8/4/2 -- the window must not name 32 or 16.
        self.assertEqual(
            B._group_merge_window(72, 82944, 64, 64, 1, 3, 3, 1), (8, 4, 2)
        )
        # A 15x15 filter is all brake and no padding to recover, so even 1280
        # groups at a tiny launch stop well short of the ceiling.
        self.assertEqual(
            B._group_merge_window(1280, 100, 64, 64, 1, 15, 15, 1), (16, 8, 4)
        )


if __name__ == "__main__":
    unittest.main()
