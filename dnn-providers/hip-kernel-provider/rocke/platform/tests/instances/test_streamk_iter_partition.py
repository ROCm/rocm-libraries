# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side math of the iteration-balanced stream-K partitioner.

The partition drives the launch grid, which lives outside the IR, so the
byte-identity gate cannot see a disagreement here. The reference vectors
below are mirrored verbatim by ``streamk_iter_partition.cpp`` for the C
engine; the invariant sweep checks the properties the fixup relies on.
"""

from __future__ import annotations

import unittest

from rocke.helpers.streamk import (
    StreamKIterPartition,
    streamk_end_iter,
    streamk_iter_partition,
    streamk_start_iter,
)

# (m_tiles, n_tiles, iters_per_tile, max_active_wgs, persistent) ->
# (sk_tiles, sk_ctas, iters_per_sk_cta, extra_iters, dp_tiles, grid_size,
#  max_linear_partners, tree_rounds, flags_bytes(n_waves=1))
# iters_per_sk_cta is clamped to 1 for an all-DP partition (a kernel divisor).
_VECTORS = (
    ((4, 4, 8, 16, False), (0, 0, 1, 0, 16, 16, 0, 0, 0)),  # tiles % W == 0
    ((2, 3, 10, 8, False), (6, 8, 7, 4, 0, 8, 2, 2, 256)),  # tiles < W
    ((5, 7, 12, 16, False), (19, 16, 14, 4, 16, 32, 1, 1, 256)),  # tiles > W
    ((1, 3, 2, 8, False), (0, 0, 1, 0, 3, 3, 0, 0, 0)),  # too little SK work
    ((5, 7, 12, 16, True), (19, 16, 14, 4, 16, 16, 1, 1, 256)),  # persistent
    ((3, 3, 64, 4, False), (5, 4, 80, 0, 4, 8, 1, 1, 256)),
    ((7, 5, 3, 16, False), (19, 16, 3, 9, 16, 32, 1, 1, 256)),
)


def _plan(p: StreamKIterPartition) -> tuple:
    return (
        p.sk_tiles,
        p.sk_ctas,
        p.iters_per_sk_cta,
        p.extra_iters,
        p.dp_tiles,
        p.grid_size,
        p.max_linear_partners,
        p.tree_rounds,
        p.flags_bytes(1),
    )


def _make(m, n, ipt, w, persistent=False) -> StreamKIterPartition:
    return streamk_iter_partition(
        m_tiles=m,
        n_tiles=n,
        iters_per_tile=ipt,
        max_active_wgs=w,
        persistent=persistent,
    )


class TestStreamKIterPartitionVectors(unittest.TestCase):
    def test_reference_vectors(self):
        for args, want in _VECTORS:
            with self.subTest(args=args):
                self.assertEqual(_plan(_make(*args)), want)

    def test_start_end_iter_vectors(self):
        p = _make(5, 7, 12, 16)
        self.assertEqual(
            [streamk_start_iter(p, c) for c in range(5)], [192, 207, 222, 237, 252]
        )
        self.assertEqual(
            [streamk_end_iter(p, c) for c in range(5)], [207, 222, 237, 252, 266]
        )

    def test_flags_bytes_per_wave(self):
        # One i32 flag per (stream-K CTA, wave), padded to 256 bytes.
        for args, n_waves, want in (
            ((5, 7, 12, 16), 4, 256),  # 4 * 16 * 4 = 256
            ((5, 7, 12, 16), 8, 512),
            ((2, 3, 10, 8), 9, 512),  # 4 * 8 * 9 = 288
            ((4, 4, 8, 16), 4, 0),  # all-DP: no flags
        ):
            with self.subTest(args=args, n_waves=n_waves):
                self.assertEqual(_make(*args).flags_bytes(n_waves), want)
        with self.assertRaisesRegex(ValueError, "n_waves must be > 0"):
            _make(5, 7, 12, 16).flags_bytes(0)

    def test_rejects_non_positive_counts(self):
        for kw in ("m_tiles", "n_tiles", "iters_per_tile", "max_active_wgs"):
            args = {
                "m_tiles": 2,
                "n_tiles": 2,
                "iters_per_tile": 4,
                "max_active_wgs": 8,
            }
            args[kw] = 0
            with (
                self.subTest(field=kw),
                self.assertRaisesRegex(ValueError, f"{kw} must be > 0"),
            ):
                streamk_iter_partition(**args)


class TestStreamKIterPartitionInvariants(unittest.TestCase):
    """Exhaustive sweep of small shapes against the fixup's assumptions."""

    def _check(self, p: StreamKIterPartition) -> None:
        ipt = p.iters_per_tile
        self.assertEqual(p.dp_tiles + p.sk_tiles, p.num_tiles)
        # A kernel divisor even when all-DP.
        self.assertGreaterEqual(p.iters_per_sk_cta, 1)
        if not p.sk_ctas:
            self.assertEqual(p.sk_tiles, 0)
            return
        # Every SK CTA owns >= 1 iteration and the ranges tile the SK span
        # contiguously.
        cursor = p.total_dp_iters
        contributors = {}
        for c in range(p.sk_ctas):
            lo, hi = streamk_start_iter(p, c), streamk_end_iter(p, c)
            self.assertEqual(lo, cursor)
            self.assertGreater(hi, lo)
            cursor = hi
            tiles = range(lo // ipt, (hi - 1) // ipt + 1)
            # A CTA is a non-owner in at most its first tile: one partial slot
            # per SK CTA is enough.
            non_owner = [t for t in tiles if lo > t * ipt]
            self.assertLessEqual(len(non_owner), 1)
            for t in tiles:
                contributors.setdefault(t, []).append(c)
        self.assertEqual(cursor, p.total_dp_iters + p.total_sk_iters)
        for t, ctas in contributors.items():
            # Contributors to one tile are consecutive SK CTAs.
            self.assertEqual(ctas, list(range(ctas[0], ctas[0] + len(ctas))))
            self.assertLessEqual(len(ctas) - 1, p.max_linear_partners)
            self.assertLessEqual(len(ctas), 1 << p.tree_rounds)

    def test_sweep(self):
        for m in range(1, 7):
            for n in range(1, 5):
                for ipt in (1, 2, 3, 5, 8, 13):
                    for w in (1, 2, 3, 4, 7, 8, 16):
                        with self.subTest(m=m, n=n, ipt=ipt, w=w):
                            self._check(_make(m, n, ipt, w))


if __name__ == "__main__":
    unittest.main()
