# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the short-K small-tile dispatch rule.

``common.small_tile_shortk_tile_trait`` gives the gfx950 small ring
non-power-of-two tiles (192x128 / 128x192 at 8 waves, 256x128x32 at 4 waves)
for 256 <= K <= 768 on half a wave to ``SMALL_TILE_SHORTK_MAX_WAVES`` waves of its
own tiles. Every body is a combination of existing levers, so no kernel code
changes; the rule must leave every other shape (and the pinned parity
shapes) on the 128x128 / 256x128 geometry, and ``SMALL_TILE_SHORTK = False``
restores it.
"""

from __future__ import annotations

import unittest
from unittest import mock

from rocke.dispatch import dispatch_gemm_bf16, dispatch_gemm_fp16
from rocke.dispatch.gemm import common as C
from rocke.dispatch.gemm.common import GemmRequest
from rocke.instances.common.gemm_universal import is_valid_spec


def _pick(dt, M, N, K, arch="gfx950"):
    fn = dispatch_gemm_fp16 if dt == "fp16" else dispatch_gemm_bf16
    return fn(GemmRequest(M=M, N=N, K=K, arch=arch, dtype=dt))


def _geo(r):
    t = r.spec.tile
    return (t.tile_m, t.tile_n, t.tile_k, t.warp_m, t.warp_n)


class TestSmallTileShortKPicks(unittest.TestCase):
    def test_canonical_rows(self):
        # few-tile short-K contraction shapes the rule was calibrated on
        for dt in ("fp16", "bf16"):
            with self.subTest(dt=dt, shape="3072x1920x576"):
                r = _pick(dt, 3072, 1920, 576)  # 240 tiles of 192x128: one wave
                tr = r.spec.trait
                self.assertEqual(r.kernel_id.spec_id, "cdna_dtl_ring_small")
                self.assertEqual(_geo(r), (192, 128, 64, 2, 4))
                self.assertEqual(
                    (tr.dtl_stages, tr.mfma_interleave, tr.ilv_merge), (3, True, True)
                )
                self.assertEqual(tr.ilv_barrier_after, 0)
                self.assertEqual(r.spec.block_size, 512)
            with self.subTest(dt=dt, shape="4608x3840x576"):
                r = _pick(dt, 4608, 3840, 576)  # 720 tiles of 128x192
                tr = r.spec.trait
                self.assertEqual(_geo(r), (128, 192, 64, 4, 2))
                self.assertEqual(tr.dtl_stages, 2)
                self.assertFalse(tr.chiplet_swizzle or tr.mfma_interleave)
            with self.subTest(dt=dt, shape="4608x4608x288"):
                r = _pick(dt, 4608, 4608, 288)  # K % 64 == 32
                tr = r.spec.trait
                self.assertEqual(_geo(r), (256, 128, 32, 4, 1))
                self.assertTrue(tr.acc_agpr and tr.mfma_interleave)
                self.assertEqual(tr.dtl_stages, 3)
                self.assertFalse(tr.interwave_setprio)
            # exact tiles: no bounds, soffset-0 DTLs, the cshuffle epilogue
            for M, N, K in ((3072, 1920, 576), (4608, 3840, 576), (4608, 4608, 288)):
                with self.subTest(dt=dt, shape=(M, N, K), what="exact"):
                    tr = _pick(dt, M, N, K).spec.trait
                    self.assertFalse(tr.pad_m or tr.pad_n or tr.dtl_tight_rsrc)
                    self.assertFalse(tr.pad_fast_tile)
                    self.assertTrue(tr.dtl_soff0)
                    self.assertEqual(tr.epilogue, "cshuffle")

    def test_ragged_takes_the_padded_body(self):
        for dt in ("fp16", "bf16"):
            for M, N, K in ((4608, 2880, 288), (4608, 4616, 288), (3072, 1928, 576)):
                with self.subTest(dt=dt, shape=(M, N, K)):
                    r = _pick(dt, M, N, K)
                    tr = r.spec.trait
                    self.assertEqual(r.kernel_id.spec_id, "cdna_dtl_ring_small")
                    self.assertIsNotNone(
                        C.small_tile_shortk_tile_trait("gfx950", M, N, K, dt)
                    )
                    self.assertIn(r.spec.tile.tile_n, (128, 192))
                    self.assertTrue(tr.pad_m and tr.pad_n and tr.dtl_tight_rsrc)
                    self.assertTrue(tr.pad_fast_tile)
                    self.assertFalse(tr.dtl_soff0)
                    self.assertEqual(tr.epilogue, "default")

    def test_orientation_follows_divisibility(self):
        # one wave: 192x128 when it divides, else 128x192 when that divides
        r = _pick("fp16", 2048, 2304, 512)  # 2048 % 192 != 0, 2304 % 192 == 0
        self.assertEqual(_geo(r)[:2], (128, 192))
        r = _pick("fp16", 2304, 2304, 512)
        self.assertEqual(_geo(r)[:2], (192, 128))

    def test_every_pick_is_valid(self):
        shapes = [
            (M, N, K)
            for M in (2048, 2304, 3072, 3840, 4608)
            for N in (1920, 2304, 2880, 3072, 3840, 4608)
            for K in (256, 288, 320, 384, 448, 512, 576, 640, 704, 768)
        ]
        n_r3 = 0
        for dt in ("fp16", "bf16"):
            for M, N, K in shapes:
                r3 = C.small_tile_shortk_tile_trait("gfx950", M, N, K, dt)
                r = _pick(dt, M, N, K)
                ok, why = is_valid_spec(r.spec, "gfx950")
                with self.subTest(dt=dt, shape=(M, N, K)):
                    self.assertTrue(ok, why)
                    if r3 is not None and r.kernel_id.spec_id == "cdna_dtl_ring_small":
                        n_r3 += 1
                        self.assertEqual((r.spec.tile, r.spec.trait), r3)
        self.assertGreater(n_r3, 50)


class TestSmallTileShortKScope(unittest.TestCase):
    def test_k_range_and_tile_bounds(self):
        f = C.small_tile_shortk_tile_trait
        self.assertIsNone(f("gfx950", 4608, 4608, 160, "fp16"))  # K < 256
        self.assertIsNone(f("gfx950", 2048, 2048, 2048, "fp16"))  # K > 768
        self.assertIsNone(f("gfx950", 2048, 2048, 832, "fp16"))
        self.assertIsNone(f("gfx950", 3072, 1920, 560, "fp16"))  # K % 32 != 0
        # fewer than half a wave of 192x128 tiles (96): the 128x128 small ring
        self.assertIsNone(f("gfx950", 1536, 1440, 576, "fp16"))
        # more than SMALL_TILE_SHORTK_MAX_WAVES waves of its tiles (ragged K = 512 on
        # a big grid stays on the 128x128 small ring)
        self.assertIsNone(f("gfx950", 8192, 8200, 512, "fp16"))
        self.assertIsNotNone(f("gfx950", 3840, 4608, 576, "fp16"))  # 720 tiles
        # more than one wave of 128x192 tiles with a poorly filled last wave
        # (held out: the 128x128 small ring keeps them)
        for M, N, K in (
            (3072, 2304, 768),
            (4096, 2048, 512),
            (2816, 2816, 512),
            (2048, 4096, 768),
        ):
            self.assertIsNone(f("gfx950", M, N, K, "fp16"), (M, N, K))
        self.assertIsNotNone(f("gfx950", 3072, 3072, 384, "fp16"))  # fill 0.75
        self.assertIsNone(f("gfx942", 3072, 1920, 576, "fp16"))  # gfx950 only

    def test_knob_off_restores_128x128_geometry(self):
        with mock.patch.object(C, "SMALL_TILE_SHORTK", False):
            for dt in ("fp16", "bf16"):
                r = _pick(dt, 3072, 1920, 576)
                self.assertEqual(_geo(r), (128, 128, 64, 2, 4))
                self.assertTrue(r.spec.trait.pad_m and r.spec.trait.dtl_tight_rsrc)
                r = _pick(dt, 4608, 4608, 288)
                self.assertEqual(_geo(r), (256, 128, 32, 2, 2))

    def test_parity_pinned_shapes_unchanged(self):
        # the parity-pinned dispatched shapes (test_gemm_persistent_streamk) stay on
        # their DTL ring bodies
        for dt, (M, N, K) in (
            ("fp16", (8192, 8200, 64)),
            ("bf16", (2048, 2048, 2048)),
            ("fp16", (2048, 2048, 96)),
            ("bf16", (4608, 4608, 160)),
        ):
            with self.subTest(dt=dt, shape=(M, N, K)):
                self.assertIsNone(C.small_tile_shortk_tile_trait("gfx950", M, N, K, dt))
                with mock.patch.object(C, "SMALL_TILE_SHORTK", False):
                    off = _pick(dt, M, N, K).spec
                self.assertEqual(_pick(dt, M, N, K).spec, off)

    def test_short_k_rilv_class_unchanged(self):
        # the short-K row-interleave class keeps its candidate slot; its
        # poorly filled exact-128x192 shapes (e.g. 4608^2 x 384) take the
        # small-tile body (``RILV_SHORTK_SMALL_TILE``),
        # the well-filled ones keep the merged-interleave body
        r = _pick("fp16", 4608, 4608, 384)
        self.assertEqual(r.kernel_id.spec_id, "cdna_dtl_pingpong_256x256")
        self.assertEqual((r.spec.tile.tile_m, r.spec.tile.tile_n), (128, 192))
        with mock.patch.object(C, "RILV_SHORTK_SMALL_TILE", False):
            r = _pick("fp16", 4608, 4608, 384)
        self.assertTrue(r.spec.trait.lds_row_interleave)
        r = _pick("fp16", 4608, 6144, 384)  # fill 0.84: the 256x256 body
        self.assertTrue(r.spec.trait.lds_row_interleave)


if __name__ == "__main__":
    unittest.main()
