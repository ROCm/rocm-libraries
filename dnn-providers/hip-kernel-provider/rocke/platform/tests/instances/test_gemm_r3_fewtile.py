# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Few-tile / long-K GEMM levers (CPU only).

* ``split_k_serial`` 1 (fenced partials) is refused by ``is_valid_spec`` (measured wrong results on the gfx942
  register-staged body and the gfx950 DTL ring) unless ``SPLIT_K_SERIAL_FENCED_VALID``;
* ``split_k_serial`` 2 and ``split_k_aligned`` on the non-persistent direct_to_lds ring (no ping-pong / asm loop /
  K tail), with the empty-slice launch rule;
* the gfx950 split-K ring candidate (``skinny.R3_RING_SPLIT``): degree rule, spec, admission, registration,
  selector keeps it;
* the learned gemm_skinny configuration choice (``skinny_select``): features, the packed tree walk, the guard,
  the knob / env switch;
* gfx942 NN / TN / TT through the plan model's serial split-K plans (``common.R3_NATIVE_G5P_SERIAL``).
"""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest import mock

from rocke.dispatch.gemm import (
    GEMM_BF16_REGISTRY,
    GEMM_FP16_REGISTRY,
    GemmRequest,
    dispatch_gemm_fp16,
)
from rocke.dispatch.gemm import common as C
from rocke.dispatch.gemm import fp16_rcr as F16
from rocke.dispatch.gemm import selector as S
from rocke.dispatch.gemm import skinny as SK
from rocke.dispatch.gemm import skinny_select as SS
from rocke.instances.common import gemm_universal as GU
from rocke.instances.common.gemm_universal import is_valid_spec, launch_shape_error

GEMM_SKINNY_SELECT_TESTS = True  # conftest: keep the shipped learned skinny choice


def _req(M, N, K, lay="RCR", dt="fp16", arch="gfx950"):
    return GemmRequest(
        M=M, N=N, K=K, arch=arch, dtype=dt, layout=lay,
        trans_a=lay[0] == "C", trans_b=lay[1] == "C",
    )  # fmt: skip


def _ring(M=1024, N=768, K=10752, lay="RCR"):
    """The gfx950 small DTL ring spec of a request (the split ring's base)."""
    return F16._spec_cdna_dtl_ring_small(_req(M, N, K, lay), "t")


def _tr(spec, **kv):
    return replace(spec, trait=replace(spec.trait, **kv))


class _Clean(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ)
        self._env.start()
        for k in (S.SELECTOR_ENV, S.NATIVE_SELECTOR_ENV, SS.SKINNY_SELECT_ENV):
            os.environ.pop(k, None)
        S.clear_cache()
        SS.clear_cache()

    def tearDown(self):
        self._env.stop()
        S.clear_cache()
        SS.clear_cache()


class TestSerialValidity(_Clean):
    def test_fenced_mode_refused_everywhere(self):
        ring = _tr(_ring(), split_k=4, split_k_serial=1, pad_fast_tile=False)
        ok, why = is_valid_spec(ring, arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("split_k_serial 1", why)
        with mock.patch.object(GU, "SPLIT_K_SERIAL_FENCED_VALID", True):
            self.assertTrue(is_valid_spec(ring, arch="gfx950")[0])

    def test_write_through_on_ring(self):
        ring = _tr(_ring(), split_k=4, split_k_serial=2, pad_fast_tile=False)
        self.assertEqual(is_valid_spec(ring, arch="gfx950"), (True, "ok"))
        self.assertIsNone(launch_shape_error(ring, 1024, 768, 10752))
        with mock.patch.object(GU, "SPLIT_K_SERIAL_DTL", False):
            self.assertFalse(is_valid_spec(ring, arch="gfx950")[0])
        # ping-pong / persistent rings keep the vgpr_prefetch-only rule
        self.assertFalse(is_valid_spec(_tr(ring, pingpong=4), arch="gfx950")[0])
        self.assertFalse(
            is_valid_spec(_tr(ring, persistent_ctas=256), arch="gfx950")[0]
        )

    def test_aligned_slices_on_ring(self):
        ring = _ring()
        # 10752 / 64 = 168 K tiles: 5 slices of 34 tiles (the last 32) need aligned slices
        s5 = _tr(ring, split_k=5, pad_fast_tile=False)
        self.assertIsNotNone(launch_shape_error(s5, 1024, 768, 10752))
        a5 = _tr(s5, split_k_aligned=True)
        self.assertEqual(is_valid_spec(a5, arch="gfx950"), (True, "ok"))
        self.assertIsNone(launch_shape_error(a5, 1024, 768, 10752))
        self.assertIn("ska", a5.kernel_name())
        with mock.patch.object(GU, "SPLIT_K_ALIGNED_DTL", False):
            self.assertFalse(is_valid_spec(a5, arch="gfx950")[0])
        # no K tail with aligned ring slices; no empty slice
        self.assertIn("tile_k", launch_shape_error(a5, 1024, 768, 10752 + 8))
        self.assertIn("empty", launch_shape_error(_tr(a5, split_k=16), 1024, 768, 640))
        self.assertIsNone(launch_shape_error(_tr(a5, split_k=10), 1024, 768, 640))
        # the ring's K tail (pad_k / dtl_k_tail) keeps refusing aligned slices
        self.assertFalse(
            is_valid_spec(_tr(a5, pad_k=True, dtl_k_tail=1), arch="gfx950")[0]
        )
        self.assertFalse(is_valid_spec(_tr(a5, pad_k=True), arch="gfx950")[0])


class TestTicketMode(_Clean):
    """split_k_serial 3: atomic partials into W [M, N] + the per-wave ticket (measured slower than the atomic
    split on the gfx950 ring; a knob value, not dispatched)."""

    def test_valid_and_workspace(self):
        ring = _tr(_ring(), split_k=8, split_k_serial=3, pad_fast_tile=False)
        self.assertEqual(is_valid_spec(ring, arch="gfx950"), (True, "ok"))
        self.assertIn("sks3", ring.kernel_name())
        # W = M x N f32 (zeroed once), Cnt = tiles x waves
        self.assertEqual(
            GU.split_k_serial_workspace(ring, 1024, 768), (1024 * 768, 48 * 8)
        )
        two = _tr(ring, split_k_serial=2)
        self.assertEqual(
            GU.split_k_serial_workspace(two, 1024, 768), (8 * 48 * 128 * 128, 48 * 8)
        )
        self.assertFalse(is_valid_spec(_tr(ring, split_k_serial=4), arch="gfx950")[0])

    def test_binding_zeroes_w(self):
        from rocke.dispatch.gemm.binding import gemm_rcr_binding

        class _Rt:
            def __init__(self):
                self.n, self.zeroed = 0, []

            def alloc(self, n):
                self.n += 1
                return 4096 * self.n

            def memcpy_h2d(self, *a):
                pass

            def memset(self, p, v, n):
                self.zeroed.append((p, n))

        class _Res:
            pass

        for mode, want in ((3, True), (2, False)):
            res = _Res()
            res.request = _req(256, 256, 4096)
            res.grid, res.block = (2, 2, 4), (512, 1, 1)
            res.spec = _tr(
                _ring(256, 256, 4096),
                split_k=4,
                split_k_serial=mode,
                pad_fast_tile=False,
            )
            rt = _Rt()
            _packed, ptrs = gemm_rcr_binding(res, True, dtype="fp16").make_args(rt)
            w_dev = ptrs[3]
            self.assertEqual(any(p == w_dev for p, _n in rt.zeroed), want, mode)


class TestRingSplit(_Clean):
    def test_degree_rule(self):
        # 192 / tiles <= 6: serial (write-through), capped by the 512-deep slices
        self.assertEqual(SK.r3_ring_split_degree(1024, 768, 10752, 128, 128), (4, 2))
        self.assertEqual(SK.r3_ring_split_degree(2304, 256, 28416, 128, 128), (5, 2))
        # few tiles: atomic, 128 / tiles, at most 24
        self.assertEqual(SK.r3_ring_split_degree(512, 256, 19968, 128, 128), (16, 0))
        self.assertEqual(SK.r3_ring_split_degree(128, 128, 32768, 128, 128), (24, 0))
        self.assertEqual(SK.r3_ring_split_degree(128, 128, 4096, 128, 128), (8, 0))
        # enough tiles: no split
        self.assertEqual(SK.r3_ring_split_degree(4096, 2048, 8192, 128, 128), (1, 0))
        with mock.patch.object(SK, "R3_RING_SPLIT_SERIAL_MAX", 3):
            self.assertEqual(SK.r3_ring_split_degree(1024, 768, 10752, 128, 128)[1], 0)

    def test_spec_and_admission(self):
        req = _req(1024, 768, 10752)
        sp = SK.r3_ring_split_spec(req, _ring())
        self.assertEqual((sp.trait.split_k, sp.trait.split_k_serial), (4, 2))
        self.assertFalse(sp.trait.split_k_aligned)  # 168 tiles / 4
        self.assertEqual(SK.r3_ring_split_admit(req, sp), (True, "ok"))
        req5 = _req(1024, 640, 10752)  # 40 tiles -> 5 slices: aligned
        sp5 = SK.r3_ring_split_spec(req5, _ring(1024, 640))
        self.assertTrue(sp5.trait.split_k_aligned)
        self.assertTrue(SK.r3_ring_split_admit(req5, sp5)[0])
        for r, why in (
            (_req(1024, 768, 2048), "K="),
            (_req(4096, 4096, 8192), "tiles"),
            (_req(128, 128, 8192), "tiles"),
        ):
            with self.subTest(r=(r.M, r.N, r.K)):
                ok, msg = SK.r3_ring_split_admit(
                    r, SK.r3_ring_split_spec(r, _ring(r.M, r.N, r.K))
                )
                self.assertFalse(ok)
                self.assertIn(why, msg)
        # few tiles (the learned gemm_skinny configuration serves them), a small side below 64 and an
        # MN-contiguous leading dimension that is not a multiple of 8 stay out
        for r, why in (
            (_req(512, 256, 19968), "tiles"),
            (_req(48, 8192, 31488), "min(M, N)"),
            (_req(1028, 768, 10752, "CRR"), "multiple of 8"),
        ):
            with self.subTest(r=(r.M, r.N, r.K, r.layout)):
                ok, msg = SK.r3_ring_split_admit(
                    r, SK.r3_ring_split_spec(r, _ring(r.M, r.N, r.K))
                )
                self.assertFalse(ok)
                self.assertIn(why, msg)
        # enough tiles for one slice: refused (the unsplit bodies serve it)
        r = _req(1600, 1536, 8192)  # 13 x 12 = 156 tiles -> 192 / 156 rounds to 1
        ok, msg = SK.r3_ring_split_admit(
            r, SK.r3_ring_split_spec(r, _ring(1600, 1536, 8192))
        )
        self.assertFalse(ok)
        self.assertIn("one slice", msg)
        # K % 64: no split (no K tail with split-K)
        r = _req(1024, 768, 10752 + 8)
        self.assertFalse(
            SK.r3_ring_split_admit(
                r, SK.r3_ring_split_spec(r, _ring(1024, 768, 10760))
            )[0]
        )
        with mock.patch.object(SK, "R3_RING_SPLIT", False):
            self.assertFalse(SK.r3_ring_split_admit(req, sp)[0])
        self.assertFalse(
            SK.r3_ring_split_admit(_req(1024, 768, 10752, arch="gfx942"), sp)[0]
        )

    def test_registered_and_dispatched(self):
        for reg, dt in ((GEMM_FP16_REGISTRY, "fp16"), (GEMM_BF16_REGISTRY, "bf16")):
            c = reg.get(f"universal_gemm_{dt}_cdna_ring_split")
            self.assertIsNotNone(c)
            self.assertEqual(c.priority, SK.R3_RING_SPLIT_PRIORITY)
            self.assertEqual(tuple(c.capability.arches), ("gfx950",))
        for lay in ("RCR", "RRR", "CRR", "CCR"):
            with self.subTest(lay=lay):
                r = dispatch_gemm_fp16(_req(2304, 256, 28416, lay))
                self.assertEqual(
                    r.candidate.name, "universal_gemm_fp16_cdna_ring_split"
                )
                self.assertEqual(r.spec.data.layout, lay)
                self.assertEqual(int(r.spec.trait.split_k_serial), 2)
        with mock.patch.object(SK, "R3_RING_SPLIT", False):
            r = dispatch_gemm_fp16(_req(2304, 256, 28416))
            self.assertNotEqual(r.candidate.name, "universal_gemm_fp16_cdna_ring_split")

    def test_selector_keeps_the_rule_pick(self):
        req = _req(2304, 256, 28416)
        with mock.patch.dict(os.environ, {S.SELECTOR_ENV: "1"}):
            S.clear_cache()
            self.assertIsNone(S.decide(req, GEMM_FP16_REGISTRY))
            with mock.patch.object(S, "SELECTOR_KEEP_R3_RING_SPLIT", False):
                S.clear_cache()
                self.assertIsNotNone(S.decide(req, GEMM_FP16_REGISTRY))


class _FakeModel:
    """predict(X) = a fixed score per (warp_k, split) of the rows."""

    def __init__(self, fn):
        self.fn = fn

    def predict(self, X):
        import numpy as np

        return np.asarray([self.fn(x) for x in X], dtype=np.float64)


class TestSkinnySelect(_Clean):
    def test_features_and_packed_walk(self):
        req = _req(256, 512, 20224, "RRR", arch="gfx942")
        plan = SK.skinny_table_plan_of(req)
        f = SS.features("gfx942", 256, 512, 20224, plan)
        self.assertEqual(len(f), len(SS.FEATURES))
        self.assertEqual(f[SS.FEATURES.index("a942")], 1.0)
        self.assertEqual(f[SS.FEATURES.index("b_mn")], 1.0)
        loaded = SS.load("gfx942")
        self.assertIsNotNone(loaded)
        meta, model, bags = loaded
        self.assertEqual(meta["feature_version"], SS.FEATURE_VERSION)
        from pathlib import Path

        ens = S.TreeEnsemble.from_text(
            (Path(SS.__file__).with_name("selector_models") / meta["model"]).read_text()
        )
        rows = [
            SS.features("gfx942", 256, 512, 20224, c)
            for c in SS.menu(req, plan, meta)[:24]
        ]
        want = ens.predict(rows)
        got = model.predict(rows)
        for a, b in zip(want, got):
            self.assertAlmostEqual(a, b, places=9)

    def test_feature_values(self):
        from rocke.instances.common.gemm_skinny import SkinnyGemmSpec

        sp = SkinnyGemmSpec(dtype="bf16", wave_m_atoms=4, wave_n_atoms=2, warp_m=1, warp_n=2, warp_k=4, kc=8,
                            prefetch=2, split=8, red_sync="sc1")  # fmt: skip
        f = dict(zip(SS.FEATURES, SS.features("gfx950", 256, 512, 8192, sp)))
        self.assertEqual(
            (f["ma"], f["na"], f["wm"], f["wn"], f["wk"], f["kc"], f["pf"]),
            (4, 2, 1, 2, 4, 8, 2),
        )
        self.assertEqual(
            (f["tile_m"], f["tile_n"], f["lsplit"], f["a942"], f["bf16"]),
            (64, 64, 3.0, 0.0, 1.0),
        )
        # 4 x 8 tiles x 4 waves x 8 slices
        self.assertEqual(f["waves_cu"], 4 * 8 * 8 * 8 / 256)

    def test_packed_walk_threshold(self):
        text = "\n".join([
            "tree", "feature_names=a b", "", "Tree=0", "num_leaves=2", "split_feature=0", "threshold=2",
            "decision_type=2", "left_child=-1", "right_child=-2", "leaf_value=1 5", "", "end of trees",
        ])  # fmt: skip
        ens = S.TreeEnsemble.from_text(text)
        got = SS._Packed(ens).predict([[2.0, 0.0], [2.5, 0.0], [1.0, 0.0]])
        self.assertEqual(list(got), [1.0, 5.0, 1.0])

    def test_menu_starts_with_the_plan(self):
        req = _req(256, 512, 20224, "RRR", arch="gfx942")
        plan = SK.skinny_table_plan_of(req)
        m = SS.menu(req, plan, {"menu": [[4, 4, 1, 1, 2, 8, 2]]})
        self.assertIs(m[0], plan)
        self.assertGreater(len(m), 1)

    def _with(self, fn_main, fns_bags):
        meta = {"menu": [[4, 4, 1, 1, 2, 8, 2], [4, 4, 2, 2, 1, 16, 2]]}
        return mock.patch.object(
            SS,
            "load",
            lambda arch: (
                meta,
                _FakeModel(fn_main),
                tuple(_FakeModel(f) for f in fns_bags),
            ),
        )

    def test_guard(self):
        req = _req(256, 512, 20224, "RRR", arch="gfx942")
        plan = SK.skinny_table_plan_of(req)
        wk = SS.FEATURES.index("wk")
        fast = lambda x: 1.0 if x[wk] == 2 else 2.0  # noqa: E731
        with self._with(fast, (fast, fast)):
            sp, why = SS.select(req, plan)
            self.assertEqual(sp.warp_k, 2)
            self.assertIn("faster", why)
        SS.clear_cache()
        # one dissenting bag keeps the table plan
        with self._with(fast, (fast, lambda x: 3.0 if x[wk] == 2 else 2.0)):
            self.assertEqual(SS.select(req, plan)[0], plan)
        SS.clear_cache()
        # below the margin keeps the table plan
        tiny = lambda x: 2.0 - 0.01 if x[wk] == 2 else 2.0  # noqa: E731
        with self._with(tiny, (tiny, tiny)):
            self.assertEqual(SS.select(req, plan)[0], plan)

    def test_switches(self):
        req = _req(256, 512, 20224, "RRR", arch="gfx942")
        plan = SK.skinny_table_plan_of(req)
        with mock.patch.dict(os.environ, {SS.SKINNY_SELECT_ENV: "0"}):
            self.assertEqual(SS.select(req, plan), (plan, "SKINNY_SELECT off"))
            self.assertEqual(SK.skinny_plan(req), plan)
        with mock.patch.object(SS, "SKINNY_SELECT", False):
            self.assertEqual(SS.select(req, plan)[0], plan)
        self.assertEqual(
            SS.select(_req(256, 512, 20224, arch="gfx1201"), plan)[0], plan
        )
        # many tiles (outside the model's cells) keep the table plan
        breq = _req(4096, 8192, 8192, "CCR", arch="gfx942")
        self.assertIn("outside", SS.select(breq, SK.skinny_table_plan_of(breq))[1])
        # decode-like shapes keep the table plan
        dreq = _req(16, 4096, 8192, arch="gfx942")
        dplan = SK.skinny_table_plan_of(dreq)
        self.assertTrue(SS.select(dreq, dplan)[1].startswith("min(M, N)"))

    def test_shipped_choice_is_launchable(self):
        from rocke.instances.common.gemm_skinny import (
            is_valid_spec as skv,
            skinny_launch_error,
        )

        for arch in ("gfx950", "gfx942"):
            for M, N, K, lay in ((512, 256, 19968, "CCR"), (256, 512, 20992, "CRR"), (64, 4096, 4096, "RCR"),
                                 (16, 8192, 8192, "RCR"), (3, 777, 1001, "RRR")):  # fmt: skip
                with self.subTest(arch=arch, shape=(M, N, K, lay)):
                    r = _req(M, N, K, lay, arch=arch)
                    sp = SK.skinny_plan(r)
                    self.assertTrue(skv(sp, arch)[0])
                    self.assertIsNone(skinny_launch_error(sp, M, N, K, sp.split))
                    self.assertEqual(sp.layout, lay)


G3_SOV = {
    "gfx942": {"max_tiles": 48, "longk_tiles": 0, "longk_min": 1 << 62, "max_small": 64}
}


class TestSkinnyOverVgprNative(_Clean):
    def test_refit_rule(self):
        # the 48-tile rule (the shipped rule is tested in test_gemm_merge_r3)
        with mock.patch.dict(SK.SKINNY_OVER_VGPR_NATIVE, G3_SOV, clear=True):
            self._refit_rule()

    def _refit_rule(self):
        rule = SK.SKINNY_OVER_VGPR_NATIVE["gfx942"]
        self.assertEqual(
            (rule["max_tiles"], rule["max_small"], rule["longk_tiles"]), (48, 64, 0)
        )
        g = lambda M, N, K: SK.skinny_native_over_vgpr(
            _req(M, N, K, "RRR", arch="gfx942")
        )  # noqa: E731
        self.assertTrue(g(640, 1152, 4096))  # 5 x 9 = 45 tiles
        self.assertFalse(g(1024, 768, 4096))  # 48 tiles
        self.assertFalse(g(1024, 1024, 16384))  # 64 tiles: no long-K extension any more
        self.assertTrue(g(48, 26112, 4096))  # min(M, N) < 64
        self.assertFalse(g(64, 26112, 4096))
        self.assertFalse(
            SK.skinny_native_over_vgpr(_req(640, 768, 4096, arch="gfx942"))
        )  # RCR


class TestNativeG5pSerial(_Clean):
    def test_serial_plans_reach_native_layouts(self):
        req = _req(1024, 768, 10752, "RRR", arch="gfx942")
        tt = C.merge2_g5p_native_tile_trait(req)
        self.assertIsNotNone(tt)
        self.assertEqual(int(tt[1].split_k_serial), 2)
        with mock.patch.object(C, "R3_NATIVE_G5P_SERIAL", False):
            tt0 = C.merge2_g5p_native_tile_trait(req)
            self.assertTrue(tt0 is None or int(tt0[1].split_k_serial) == 0)


if __name__ == "__main__":
    unittest.main()
