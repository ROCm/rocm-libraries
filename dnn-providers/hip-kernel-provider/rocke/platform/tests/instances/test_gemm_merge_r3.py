# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Rules for the interactions of the GEMM dispatch levers (native-form model, K-proxy, split ring, gemm_skinny
bounds). CPU only.

* gfx942: the native-form model (g5_native, trained on the plan model's unscaled ranking) ranks the RCR plans
  without the main-loop scale (g5_plan.G5P_MAINLOOP_SCALE; the native forms do not run those traits),
  and its time of an RCR body that runs them gets that scale (g5_native.R3M_G5N_MAINLOOP_SCALE);
* gfx950 NT: a K-rounded K-proxy pick that is the split-K ring (no K tail with split-K) is re-dispatched without
  that candidate (common.R3M_KPROXY_RING_SPLIT_FALLBACK); the selector's decision cache keys the ring's state.
"""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest import mock

from rocke.dispatch.gemm import GemmRequest, dispatch_gemm_fp16
from rocke.dispatch.gemm import common as C
from rocke.dispatch.gemm import g5_native as GN
from rocke.dispatch.gemm import g5_plan as GP
from rocke.dispatch.gemm import selector as S
from rocke.dispatch.gemm import skinny as SK
from rocke.instances.common.gemm_universal import is_valid_spec, launch_shape_error

GEMM_SELECTOR_TESTS = True  # conftest: keep the shipped selector defaults


def _req(M, N, K, lay="RCR", dt="fp16", arch="gfx942"):
    return GemmRequest(
        M=M, N=N, K=K, arch=arch, dtype=dt, layout=lay,
        trans_a=lay[0] == "C", trans_b=lay[1] == "C",
    )  # fmt: skip


def _clear():
    S.clear_cache()
    C._G5P_CACHE.clear()
    C._R2_PLAN_CACHE.clear()
    GN._RANK_CACHE.clear()


class _Clean(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ)
        self._env.start()
        for k in (S.SELECTOR_ENV, S.NATIVE_SELECTOR_ENV):
            os.environ.pop(k, None)
        _clear()

    def tearDown(self):
        self._env.stop()
        _clear()


class TestDefaults(unittest.TestCase):
    def test_merge_knob_defaults(self):
        self.assertIs(GN.R3M_G5N_MAINLOOP_SCALE, True)
        self.assertIs(C.R3M_KPROXY_RING_SPLIT_FALLBACK, True)
        self.assertEqual(SK._R3_RING_SPLIT_BYPASS[0], 0)


class TestTiles2LongK(_Clean):
    """gfx950 tiles2: gemm_skinny below 48 128x128 tiles with K >= 2048."""

    def test_bound(self):
        self.assertEqual(SK.SKINNY_NATIVE_T2_LONGK_TILES, 48)
        for lay, M, N, K, want in (
            ("CRR", 192, 2048, 8192, True),  # 2 x 16 = 32 tiles, long K
            ("RRR", 256, 2944, 4096, True),  # 2 x 23 = 46 tiles, K >= 2048
            ("RRR", 256, 3072, 4096, False),  # 2 x 24 = 48 tiles
            ("RRR", 256, 2944, 1024, False),  # 46 tiles, short K
            ("CCR", 1000, 640, 2048, True),  # 8 x 5 = 40 tiles, K = 2048
        ):
            ok, _ = SK.skinny_admit_region(_req(M, N, K, lay, "bf16", arch="gfx950"))
            self.assertEqual(ok, want, (lay, M, N, K))


class TestOddLongK(_Clean):
    """gfx950 tiles2, odd MN leading dimension: gemm_skinny below 64 tiles when K >= 4096."""

    def test_rule(self):
        self.assertEqual((SK.SKINNY_NATIVE_T2_ODD_LONGK_TILES, SK.SKINNY_NATIVE_T2_ODD_LONGK_MIN), (64, 4096))
        for lay, M, N, K, want in (
            ("CRR", 513, 769, 4096, True),  # 5 x 7 = 35 tiles, K = 4096 (otherwise the DTL odd forms)
            ("CRR", 513, 769, 4088, False),  # K < 4096
            ("CRR", 1001, 500, 8192, True),  # 8 x 4 = 32 tiles
            ("CCR", 1001, 1001, 16384, False),  # 8 x 8 = 64 tiles
            ("RRR", 640, 1601, 24608, False),  # 5 x 13 = 65 tiles
            ("CRR", 1001, 380, 1024, True),  # 24 tiles: the < 32 tiles rule
        ):
            ok, _ = SK.skinny_admit_region(_req(M, N, K, lay, "bf16", arch="gfx950"))
            self.assertEqual(ok, want, (lay, M, N, K))
        with mock.patch.object(SK, "SKINNY_NATIVE_T2_ODD_LONGK_TILES", 0):
            self.assertFalse(SK.skinny_admit_region(_req(513, 769, 4096, "CRR", "bf16", arch="gfx950"))[0])


class TestSkinnyOverVgprNativeMerge(_Clean):
    """gfx942 NN / TN / TT: gemm_skinny ahead of the g5 native forms below 32 tiles with K < 16384, or with
    min(M, N) < 64 (the 48-tile rule refitted against the g5 native forms)."""

    def test_rule(self):
        rule = SK.SKINNY_OVER_VGPR_NATIVE["gfx942"]
        self.assertEqual(
            (rule["max_tiles"], rule["max_small"], rule["longk_tiles"], rule["native_min_k"]), (32, 64, 0, 16384)
        )
        g = lambda M, N, K, lay="RRR": SK.skinny_native_over_vgpr(_req(M, N, K, lay, arch="gfx942"))  # noqa: E731
        self.assertTrue(g(640, 768, 4096))  # 5 x 6 = 30 tiles
        self.assertFalse(g(640, 1152, 4096))  # 45 tiles (the 48-tile rule: gemm_skinny)
        self.assertFalse(g(640, 768, 16384))  # 30 tiles, K >= 16384: g5 native
        self.assertTrue(g(640, 768, 16376, "CCR"))
        self.assertTrue(g(48, 26112, 4096))  # min(M, N) < 64
        self.assertTrue(g(48, 1024, 20480))  # min(M, N) < 64 at long K too
        self.assertFalse(g(64, 26112, 4096))
        self.assertFalse(g(1024, 1024, 16384))  # 64 tiles
        self.assertFalse(SK.skinny_native_over_vgpr(_req(640, 768, 4096, arch="gfx942")))  # RCR


class TestNativeRankingUnscaled(_Clean):
    """g5_native ranks the RCR plans with mainloop=False (the scale would reorder in-scope plans)."""

    def test_rank_native_calls_unscaled_ranking(self):
        req = _req(4096, 4096, 4096, lay="RRR")
        if not GN.active(req):
            self.skipTest("native-form model inactive")
        seen = []
        real = C._g5p_ranked

        def spy(r, mainloop=None):
            seen.append(mainloop)
            return real(r, mainloop=mainloop)

        with mock.patch.object(C, "_g5p_ranked", spy):
            GN._RANK_CACHE.clear()
            GN.rank_native(req)
        self.assertTrue(seen)
        self.assertTrue(all(m is False for m in seen), seen)

    def test_native_forms_independent_of_mainloop_scale(self):
        req = _req(4096, 4096, 4096, lay="CCR")
        if not GN.active(req):
            self.skipTest("native-form model inactive")
        GN._RANK_CACHE.clear()
        C._G5P_CACHE.clear()
        a = [(round(u, 6), p) for u, p, _f in GN.rank_native(req)]
        with mock.patch.object(GP, "G5P_MAINLOOP_SCALE", 0.2):
            GN._RANK_CACHE.clear()
            C._G5P_CACHE.clear()
            b = [(round(u, 6), p) for u, p, _f in GN.rank_native(req)]
        self.assertTrue(a)
        self.assertEqual(a, b)


class TestRound2NativeG5pUnscaled(_Clean):
    """The native g5p pick (the comparison base of the native-form model) ranks without the main-loop scale
    (NN / TT reach it only through g5_native's comparison, so the dispatch does not show it)."""

    def test_independent_of_mainloop_scale(self):
        for lay in ("RRR", "CCR"):
            req = _req(8192, 8192, 8192, lay=lay)
            C._G5P_CACHE.clear()
            a = C.merge2_g5p_native_tile_trait(req)
            got = [a]
            for sc in (1.0, 0.2):
                with mock.patch.object(GP, "G5P_MAINLOOP_SCALE", sc):
                    C._G5P_CACHE.clear()
                    got.append(C.merge2_g5p_native_tile_trait(req))
            self.assertEqual(got, [a, a, a], lay)
        C._G5P_CACHE.clear()
        self.assertIsNotNone(C.merge2_g5p_native_tile_trait(_req(8192, 8192, 8192, lay="RRR")))


class TestPredictSpecMainloopScale(_Clean):
    def _ml_spec(self):
        r = dispatch_gemm_fp16(_req(8192, 8192, 8192))
        tr = getattr(r.spec, "trait", None)
        if tr is None or not int(getattr(tr, "vp_pair_reads", 0)):
            self.skipTest("the gfx942 8192^3 NT pick carries no main-loop traits")
        return r.spec

    def test_scale_applied_to_mainloop_rcr_body(self):
        spec = self._ml_spec()
        req = _req(8192, 8192, 8192)
        with mock.patch.object(GN, "R3M_G5N_MAINLOOP_SCALE", False):
            raw = GN.predict_spec(req, spec)
        with mock.patch.object(GN, "R3M_G5N_MAINLOOP_SCALE", True):
            scaled = GN.predict_spec(req, spec)
        self.assertIsNotNone(raw)
        self.assertAlmostEqual(scaled / raw, float(GP.G5P_MAINLOOP_SCALE), places=9)

    def test_no_scale_without_mainloop_traits(self):
        spec = self._ml_spec()
        plain = replace(
            spec, trait=replace(spec.trait, vp_pair_reads=0, vp_mfma_first=0, vp_tail_pack=0, vp_release_atoms=1)
        )
        req = _req(8192, 8192, 8192)
        with mock.patch.object(GN, "R3M_G5N_MAINLOOP_SCALE", False):
            raw = GN.predict_spec(req, plain)
        with mock.patch.object(GN, "R3M_G5N_MAINLOOP_SCALE", True):
            on = GN.predict_spec(req, plain)
        self.assertEqual(raw, on)


class TestKproxyRingSplit(_Clean):
    # 1024 x 1024 (64 tiles) and K >= 6144 is the split ring's region; K % 64 != 0 is the K-proxy's
    SHAPES = ((1024, 1024, 8200), (1536, 1024, 12296), (1024, 768, 6152))

    def test_rounded_pick_is_the_split_ring(self):
        for M, N, K in self.SHAPES:
            kp = -(-K // 64) * 64
            r = dispatch_gemm_fp16(_req(M, N, kp, arch="gfx950"))
            self.assertEqual(r.candidate.spec_id, "cdna_ring_split", (M, N, K))

    def test_fallback_takes_unsplit_ring_with_k_tail(self):
        for M, N, K in self.SHAPES:
            r = dispatch_gemm_fp16(_req(M, N, K, arch="gfx950"))
            self.assertTrue(any("k-proxy" in e for e in r.explanation), (M, N, K, r.explanation))
            self.assertNotEqual(r.candidate.spec_id, "cdna_ring_split")
            self.assertEqual(int(r.spec.trait.split_k), 1)
            self.assertTrue(r.spec.trait.dtl_k_tail)
            self.assertTrue(is_valid_spec(r.spec, arch="gfx950")[0])
            self.assertIsNone(launch_shape_error(r.spec, M, N, K))
            self.assertEqual(SK._R3_RING_SPLIT_BYPASS[0], 0)

    def test_knob_off_keeps_the_normal_dispatch(self):
        with mock.patch.object(C, "R3M_KPROXY_RING_SPLIT_FALLBACK", False):
            for M, N, K in self.SHAPES:
                _clear()
                r = dispatch_gemm_fp16(_req(M, N, K, arch="gfx950"))
                self.assertFalse(any("k-proxy" in e for e in r.explanation), (M, N, K))
                self.assertEqual(SK._R3_RING_SPLIT_BYPASS[0], 0)

    def test_reentry_guard(self):
        req = _req(1024, 1024, 8200, arch="gfx950")
        self.assertIsNotNone(C.r3_kproxy_dispatch(req, dispatch_gemm_fp16))
        self.assertEqual(C._R3_KPROXY_ACTIVE[0], 0)
        C._R3_KPROXY_ACTIVE[0] += 1
        try:
            self.assertIsNone(C.r3_kproxy_dispatch(req, dispatch_gemm_fp16))
        finally:
            C._R3_KPROXY_ACTIVE[0] -= 1
        # the rounded-K dispatch inside a K-proxy never re-applies it
        seen = []

        def spy(r):
            seen.append(C._R3_KPROXY_ACTIVE[0])
            return dispatch_gemm_fp16(r)

        _clear()
        C.r3_kproxy_dispatch(req, spy)
        self.assertTrue(seen)
        self.assertTrue(all(x == 1 for x in seen), seen)

    def test_split_ring_unchanged_at_aligned_k(self):
        r = dispatch_gemm_fp16(_req(1024, 1024, 8192, arch="gfx950"))
        self.assertEqual(r.candidate.spec_id, "cdna_ring_split")
        self.assertGreater(int(r.spec.trait.split_k), 1)

    def test_bypass_refuses_the_split_ring(self):
        req = _req(1024, 1024, 8192, arch="gfx950")
        spec = dispatch_gemm_fp16(req).spec
        self.assertTrue(SK.r3_ring_split_admit(req, spec)[0])
        SK._R3_RING_SPLIT_BYPASS[0] += 1
        try:
            self.assertFalse(SK.r3_ring_split_admit(req, spec)[0])
            self.assertFalse(SK.r3_ring_split_state())
        finally:
            SK._R3_RING_SPLIT_BYPASS[0] -= 1
        self.assertTrue(SK.r3_ring_split_state())

    def test_selector_cache_key_tracks_ring_state(self):
        from rocke.dispatch.gemm import GEMM_FP16_REGISTRY

        req = _req(1024, 1024, 8192, arch="gfx950")
        k0 = S._cache_key(req, GEMM_FP16_REGISTRY)
        SK._R3_RING_SPLIT_BYPASS[0] += 1
        try:
            k1 = S._cache_key(req, GEMM_FP16_REGISTRY)
        finally:
            SK._R3_RING_SPLIT_BYPASS[0] -= 1
        with mock.patch.object(SK, "R3_RING_SPLIT", False):
            k2 = S._cache_key(req, GEMM_FP16_REGISTRY)
        self.assertNotEqual(k0, k1)
        self.assertEqual(k1, k2)


if __name__ == "__main__":
    unittest.main()
