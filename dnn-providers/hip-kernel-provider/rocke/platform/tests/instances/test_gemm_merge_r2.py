# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Rules where the GEMM levers interact (gemm_skinny, plan models, native forms). CPU only.

* gfx942 RCR: the plan model (g5_plan) and the gemm_skinny region rule (skinny.SKINNY_OVER_VGPR) both
  arbitrate gemm_skinny vs the register-staged body; the model decides when it is on
  (common.MERGE2_SKINNY_OVER_VGPR_WITH_MODEL = True applies the region rule too);
* the native (NN / TN / TT) selector ships off on gfx942 (its model does not score the register-staged native forms);
* the selector's model kinds: RCR models accept every supported feature version, native models only their own;
  a native request keeps a gemm_skinny rule pick;
* plan.plan_gemm routes gfx942 to the register-staged cost-model chooser (gemm_plan), other arches to the native-selector
  chooser; the pre-pass RCR estimate of the gfx942 chooser (plan model, ratio-calibrated, bounded by the fixed-menu
  plan's fitted estimate) and the slow gemm_skinny fallback rule;
* common.MERGE2_NATIVE_G5P: NN / TN / TT through the native form of the plan model's RCR pick.
"""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest import mock

from rocke.dispatch.gemm import (
    GEMM_FP16_REGISTRY,
    GemmRequest,
    dispatch_gemm_bf16,
    dispatch_gemm_fp16,
)
from rocke.dispatch.gemm import common as C
from rocke.dispatch.gemm import g5_plan as GP
from rocke.dispatch.gemm import plan as P
from rocke.dispatch.gemm import selector as S
from rocke.dispatch.gemm import skinny as SK
from rocke.instances.common.gemm_universal import is_valid_spec, launch_shape_error

GEMM_SELECTOR_TESTS = True  # conftest: keep the shipped selector defaults


def _req(M, N, K, lay="RCR", dt="fp16", arch="gfx942"):
    return GemmRequest(
        M=M, N=N, K=K, arch=arch, dtype=dt, layout=lay,
        trans_a=lay[0] == "C", trans_b=lay[1] == "C",
    )  # fmt: skip


class _Clean(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ)
        self._env.start()
        for k in (S.SELECTOR_ENV, S.NATIVE_SELECTOR_ENV, P.GEMM_PLAN_PREPASS_ENV):
            os.environ.pop(k, None)
        S.clear_cache()
        C._G5P_CACHE.clear()

    def tearDown(self):
        self._env.stop()
        S.clear_cache()
        C._G5P_CACHE.clear()


class TestSkinnyVsG5Arbitration(_Clean):
    # 24 x 1856 x 23104 bf16 RCR: inside SKINNY_OVER_VGPR's region; the plan model keeps the register-staged body
    REQ = (24, 1856, 23104, "RCR", "bf16")

    def test_model_decides_by_default(self):
        self.assertFalse(C.MERGE2_SKINNY_OVER_VGPR_WITH_MODEL)
        self.assertTrue(SK.skinny_takes_over_vgpr(_req(*self.REQ)))
        r = dispatch_gemm_bf16(_req(*self.REQ))
        self.assertIn("cdna_vgpr", r.candidate.name)

    def test_region_rule_with_model_knob(self):
        with mock.patch.object(C, "MERGE2_SKINNY_OVER_VGPR_WITH_MODEL", True):
            r = dispatch_gemm_bf16(_req(*self.REQ))
        self.assertIn("cdna_skinny", r.candidate.name)

    def test_region_rule_without_model(self):
        with mock.patch.object(GP, "G5P_ON", False):
            r = dispatch_gemm_bf16(_req(*self.REQ))
        self.assertIn("cdna_skinny", r.candidate.name)

    def test_native_layout_not_arbitrated(self):
        # the register-staged native forms (R2_VP_MN) admit before any skinny arbitration
        spec_req = _req(2048, 2048, 2048, "RRR")
        r = dispatch_gemm_fp16(spec_req)
        self.assertIn("cdna_vgpr", r.candidate.name)
        self.assertEqual(r.spec.data.layout, "RRR")
        self.assertTrue(C.g5_vgpr_admit(spec_req, r.spec)[0])


class TestNativeSelectorDefaults(_Clean):
    def test_gfx942_native_selector_ships_off(self):
        self.assertFalse(S.native_selector_enabled("gfx942"))
        self.assertTrue(S.native_selector_enabled("gfx950"))
        with mock.patch.dict(os.environ, {S.NATIVE_SELECTOR_ENV: "1"}):
            self.assertTrue(S.native_selector_enabled("gfx942"))

    def test_gfx942_native_pick_is_rule_order(self):
        req = _req(1000, 712, 3000, "RRR")
        rule = GEMM_FP16_REGISTRY.select(req)
        self.assertEqual(dispatch_gemm_fp16(req).spec, rule.select_spec(req))
        self.assertIsNone(S.decide(req, GEMM_FP16_REGISTRY))

    def test_model_kind_versions(self):
        self.assertIn(S.FEATURE_VERSION, S.SUPPORTED_FEATURE_VERSIONS)
        # the RCR kind loads every supported version; the native kind only its own
        with mock.patch.dict(
            S.MODEL_KINDS, {"native": ("gemm_native", S.FEATURE_VERSION)}
        ):
            S.clear_cache()
            self.assertIsNone(S.load_model("gfx950", "native"))
        S.clear_cache()
        self.assertIsNotNone(S.load_model("gfx950", "native"))
        self.assertIsNotNone(S.load_model("gfx950", "rcr"))
        self.assertIsNotNone(S.load_model("gfx942", "rcr"))  # g4-v1 sidecar

    def test_native_request_keeps_skinny_rule_pick(self):
        req = _req(384, 512, 3000, "CCR", arch="gfx950")
        rule = GEMM_FP16_REGISTRY.select(req)
        self.assertEqual(type(rule.select_spec(req)).__name__, "SkinnyGemmSpec")
        self.assertIsNone(S.decide(req, GEMM_FP16_REGISTRY))

    def test_rcr_skinny_rule_pick_is_ranked(self):
        # an RCR gemm_skinny rule pick on gfx950 is ranked by the r2c-v2 model (it knows the family)
        req = _req(24, 1856, 23104, "RCR", arch="gfx950")
        rule = GEMM_FP16_REGISTRY.select(req)
        if type(rule.select_spec(req)).__name__ != "SkinnyGemmSpec":
            self.skipTest("rule pick is not gemm_skinny here")
        self.assertTrue(S.load_model("gfx950").knows_skinny)
        self.assertIsNotNone(S.decide(req, GEMM_FP16_REGISTRY))


class TestSkinnyNativeThresholds(_Clean):
    """gfx950 native gemm_skinny below 16 128x128 tiles (or an odd MN
    leading dimension); gfx942 native gemm_skinny ahead of the register-staged native forms
    below 32 tiles."""

    def test_gfx950_tiles(self):
        # (the rule with every odd MN leading dimension on gemm_skinny; the
        # DTL-ring odd forms off -- TestOddLdThresholds)
        with mock.patch.object(
            SK, "SKINNY_NATIVE_T2_ODD_MAX_TILES", 0
        ), mock.patch.object(
            SK, "SKINNY_NATIVE_T2_LONGK_TILES", 32
        ):  # the shipped long-K tile bound is 48 (test_gemm_merge_r3)
            self._gfx950_tiles()

    def _gfx950_tiles(self):
        self.assertEqual(SK.SKINNY_NATIVE_RULE["gfx950"], "tiles2")
        self.assertEqual(
            (
                SK.SKINNY_NATIVE_T2_SMALL,
                SK.SKINNY_NATIVE_T2_LONGK_TILES,
                SK.SKINNY_NATIVE_T2_LONGK_MIN,
            ),
            (8, 32, 2048),
        )
        for lay, M, N, K, want in (
            ("RRR", 333, 8001, 4096, True),  # odd MN leading dimension (N of B)
            ("RRR", 256, 384, 512, True),  # 2 x 3 = 6 tiles
            ("RRR", 360, 360, 1024, False),  # 3 x 3 = 9 tiles, short K
            ("RRR", 360, 360, 2048, True),  # 9 tiles, K >= 2048
            ("RRR", 1000, 1000, 4096, False),  # 64 tiles
            ("CRR", 192, 2048, 8192, False),  # 2 x 16 = 32 tiles
            ("CRR", 192, 1920, 8192, True),  # 2 x 15 = 30 tiles, long K
            ("CRR", 192, 1920, 1024, False),  # 30 tiles, short K
        ):
            ok, _ = SK.skinny_admit_region(_req(M, N, K, lay, "bf16", arch="gfx950"))
            self.assertEqual(ok, want, (lay, M, N, K))

    def test_gfx942_skinny_ahead_of_g5_native(self):
        # the earlier rule (since re-fitted: test_gemm_r3_fewtile pins the shipped values)
        r2 = {"gfx942": {"max_tiles": 32, "longk_tiles": 128, "longk_min": 16384}}
        with mock.patch.dict(SK.SKINNY_OVER_VGPR_NATIVE, r2, clear=True):
            self._check_r2_rule()

    def _check_r2_rule(self):
        few = _req(640, 768, 4096, "RRR")  # 5 x 6 = 30 tiles
        many = _req(1024, 1024, 4096, "RRR")  # 64 tiles
        self.assertTrue(SK.skinny_native_over_vgpr(few))
        self.assertFalse(SK.skinny_native_over_vgpr(many))
        self.assertFalse(
            SK.skinny_native_over_vgpr(_req(512, 1024, 4096, "RRR"))
        )  # 32 tiles
        # long K: below 128 tiles with K >= 16384
        self.assertTrue(
            SK.skinny_native_over_vgpr(_req(1024, 1024, 16384, "RRR"))
        )  # 64
        self.assertFalse(
            SK.skinny_native_over_vgpr(_req(1024, 2048, 16384, "RRR"))
        )  # 128
        self.assertFalse(SK.skinny_native_over_vgpr(_req(1024, 1024, 16376, "RRR")))
        self.assertFalse(SK.skinny_native_over_vgpr(_req(640, 768, 4096)))  # RCR
        self.assertIn("cdna_skinny", dispatch_gemm_fp16(few).candidate.name)
        self.assertIn("cdna_vgpr", dispatch_gemm_fp16(many).candidate.name)
        r = dispatch_gemm_fp16(few)
        self.assertFalse(
            C.g5_vgpr_admit(few, r.spec)[0] if hasattr(r.spec, "trait") else False
        )
        with mock.patch.dict(SK.SKINNY_OVER_VGPR_NATIVE, clear=True):
            self.assertIn("cdna_vgpr", dispatch_gemm_fp16(few).candidate.name)


class TestPlanRouting(_Clean):
    def test_gfx942_uses_g5_chooser(self):
        req = _req(1149, 4396, 3304, "CRR")
        pl = P.plan_gemm_fp16(req)
        self.assertEqual(pl.kind, "prepass_a")
        self.assertEqual(pl.request.layout, "RRR")
        self.assertEqual(pl.dispatch, pl.gemm)
        self.assertIsNotNone(pl.est_us)
        self.assertEqual(pl.workspace_bytes, 2 * 3304 * 1149)
        g = P.gemm_plan(req)
        self.assertEqual((g.kind, g.request.layout), (pl.kind, pl.request.layout))

    def test_gfx942_chooser_knob_off(self):
        req = _req(1149, 4396, 3304, "CRR")
        with mock.patch.object(P, "R2_PLAN", False):
            pl = P.plan_gemm_fp16(req)
        # the native-selector chooser: not tile-eligible -> native
        self.assertEqual(pl.kind, "native")

    def test_policy_applies_on_gfx942(self):
        req = _req(4096, 4096, 1024, "RRR")
        with mock.patch.dict(os.environ, {P.GEMM_PLAN_PREPASS_ENV: "off"}):
            self.assertEqual(P.plan_gemm_fp16(req).kind, "native")
        with mock.patch.dict(os.environ, {P.GEMM_PLAN_PREPASS_ENV: "always"}):
            pl = P.plan_gemm_fp16(req)
            self.assertEqual(pl.kind, "prepass")
            self.assertEqual(pl.gemm.spec.data.layout, "RCR")

    def test_gfx950_odd_skinny_prepass(self):
        # (the routing with every odd MN leading dimension on gemm_skinny; by
        # default the DTL-ring odd forms take it from 32 tiles, TestOddLdThresholds)
        with mock.patch.object(SK, "SKINNY_NATIVE_T2_ODD_MAX_TILES", 0):
            self._odd_skinny_prepass()

    def _odd_skinny_prepass(self):
        big = _req(
            6131, 9158, 17696, "CRR", "bf16", arch="gfx950"
        )  # odd M, N; 3456 tiles
        small = _req(1949, 1125, 29360, "CRR", "bf16", arch="gfx950")  # 144 tiles
        pl = P.plan_gemm_bf16(big)
        self.assertEqual(pl.kind, "prepass")
        self.assertEqual(
            [(t.operand, t.spec) for t in pl.transposes], [("A", None), ("B", None)]
        )
        self.assertEqual(pl.gemm.spec.data.layout, "RCR")
        self.assertEqual(pl.request.layout, "RCR")
        self.assertEqual(P.plan_gemm_bf16(small).kind, "native")
        with mock.patch.dict(P.GEMM_PLAN_ODD_SKINNY_PREPASS_TILES, {"gfx950": 0}):
            self.assertEqual(P.plan_gemm_bf16(big).kind, "native")
        with mock.patch.dict(os.environ, {P.GEMM_PLAN_PREPASS_ENV: "off"}):
            self.assertEqual(P.plan_gemm_bf16(big).kind, "native")

    def test_gfx950_uses_selector_chooser(self):
        req = _req(8192, 4096, 1024, "CRR", "bf16", arch="gfx950")
        pl = P.plan_gemm_bf16(req)
        self.assertIn("native", pl.predicted_us)
        self.assertIsNone(pl.est_us)

    def test_rcr_estimate_bounded_by_round1_plan(self):
        rreq = _req(22912, 192, 2880, "RCR", "bf16")
        res = dispatch_gemm_bf16(rreq)
        old = C.g5_vgpr_plan(22912, 192, 2880, "gfx942")
        bound = P._geo_est(rreq, *old)
        for mode in ("ratio", "abs", "off"):
            with mock.patch.object(P, "R2_PLAN_G5P_RCR", mode):
                est = P._rcr_est(rreq, res)
            self.assertIsNotNone(est, mode)
            if mode != "abs":
                self.assertLessEqual(est, bound + 1e-9, mode)
        with mock.patch.object(P, "R2_PLAN_G5P_RCR", "off"):
            self.assertAlmostEqual(P._rcr_est(rreq, res), bound, places=6)

    def test_ratio_calibration(self):
        rreq = _req(8192, 8192, 8192, "RCR")
        res = dispatch_gemm_fp16(rreq)
        with mock.patch.object(P, "R2_PLAN_G5P_RCR", "abs"):
            a = P._g5p_est(rreq, res)
        with mock.patch.object(P, "R2_PLAN_G5P_RCR", "ratio"):
            r = P._g5p_est(rreq, res)
        with mock.patch.object(P, "R2_PLAN_G5P_RCR", "off"):
            self.assertIsNone(P._g5p_est(rreq, res))
        self.assertIsNotNone(a)
        self.assertIsNotNone(r)
        self.assertNotAlmostEqual(a, r, places=3)

    def test_native_estimate_of_g5p_forms(self):
        # NN through a plan-model body (MERGE2_NATIVE_G5P): calibrated plan-model
        # time of the RCR body x the native cost factor
        req = _req(3072, 3072, 3072, "RRR")
        with mock.patch.object(C, "MERGE2_NATIVE_G5P_COMPARE", False):
            res = dispatch_gemm_fp16(req)
        self.assertIn("RRR", tuple(C.MERGE2_NATIVE_G5P))
        t = res.spec.tile
        dims = (t.tile_m, t.tile_n, t.tile_k, t.warp_m, t.warp_n)
        # a geometry outside the fixed (G5_VGPR_GEOS) menu
        self.assertFalse(any(tuple(g[:5]) == dims for g in C.G5_VGPR_GEOS.values()))
        rreq = replace(req, layout="RCR", trans_a=False, trans_b=True)
        pred = P._g5p_plan_pred(rreq, res.spec)
        self.assertIsNotNone(pred)
        est = P._g5_est(req, res)
        want = pred * P._g5p_cal(rreq) * C.r2_native_cost_factor("RRR", res.spec)
        self.assertAlmostEqual(est, want, places=6)
        with mock.patch.object(P, "R2_PLAN_G5P_RCR", "off"):
            self.assertIsNone(P._g5_est(req, res))

    def test_slow_skinny_fallback(self):
        req = _req(1149, 4396, 3304, "CRR")
        self.assertTrue(P._slow_native(req, "universal_gemm_fp16_cdna_skinny_fallback"))
        self.assertFalse(
            P._slow_native(_req(512, 4396, 3304, "CRR"), "x_skinny_fallback")
        )
        with mock.patch.object(P, "R2_PLAN_SLOW_SKINNY_FALLBACK_MIN_MN", 0):
            self.assertFalse(
                P._slow_native(req, "universal_gemm_fp16_cdna_skinny_fallback")
            )
            # without the rule the native skinny fallback keeps no cost model: native
            self.assertEqual(P.gemm_plan(req).kind, "native")


class TestNativeG5pForms(_Clean):
    SHAPES = (
        (4096, 4096, 4096),
        (1440, 4104, 7176),
        (3072, 1920, 576),
        (1000, 712, 3000),
    )

    ALL = ("RRR", "CRR", "CCR")

    def test_default_layouts(self):
        self.assertEqual(tuple(C.MERGE2_NATIVE_G5P), ("RRR", "CCR"))
        # TN (CRR) keeps the fitted 18-geometry plan
        req = _req(4096, 4096, 4096, "CRR")
        a = dispatch_gemm_fp16(req).spec
        with mock.patch.object(C, "MERGE2_NATIVE_G5P", ()):
            b = dispatch_gemm_fp16(req).spec
        self.assertEqual(a, b)

    def setUp(self):
        super().setUp()
        p = mock.patch.object(C, "MERGE2_NATIVE_G5P_COMPARE", False)
        p.start()
        self.addCleanup(p.stop)
        # these tests pin the native machinery without the native-form model;
        # the model (g5_native, tested in test_gemm_r3_g5_native.py) takes
        # precedence when it is on
        from rocke.dispatch.gemm import g5_native as _gn

        _p = mock.patch.object(_gn, "G5N_ON", False)
        _p.start()
        self.addCleanup(_p.stop)

    def test_guards(self):
        # 8-wave 256x192 plan-model bodies had only 2-element MN loads
        # natively; the guards keep the fitted-menu form there
        with mock.patch.object(C, "MERGE2_NATIVE_G5P_COMPARE", True):
            for lay, M, N, K in (
                ("RRR", 8704, 13568, 4864),
                ("CCR", 13056, 7680, 8192),
            ):
                req = _req(M, N, K, lay)
                r = dispatch_gemm_fp16(req)
                self.assertGreaterEqual(
                    r.spec.trait.mn_vec, C.MERGE2_NATIVE_G5P_MIN_VEC
                )
        req = _req(8704, 13568, 4864, "RRR")
        tt = C.merge2_g5p_native_tile_trait(req)
        if tt is not None:
            r = dispatch_gemm_fp16(req)
            self.assertGreaterEqual(r.spec.trait.mn_vec, C.MERGE2_NATIVE_G5P_MIN_VEC)
        with mock.patch.object(C, "MERGE2_NATIVE_G5P_COMPARE", True):
            with mock.patch.object(P, "_g5p_native_est", lambda *a, **k: 1e9):
                self.assertIsNone(
                    C.merge2_g5p_native_tile_trait(_req(1440, 4104, 7176, "RRR"))
                )
            with mock.patch.object(P, "_g5p_native_est", lambda *a, **k: 1e-9):
                self.assertIsNotNone(
                    C.merge2_g5p_native_tile_trait(_req(1440, 4104, 7176, "RRR"))
                )

    def test_forms_valid_and_from_the_plan_menu(self):
        menu = GP.load_menu("gfx942")
        n_forms = 0
        for lay in ("RRR", "CRR", "CCR"):
            for M, N, K in self.SHAPES:
                req = _req(M, N, K, lay)
                with mock.patch.object(C, "MERGE2_NATIVE_G5P", self.ALL):
                    tt = C.merge2_g5p_native_tile_trait(req)
                    r = dispatch_gemm_fp16(req)
                if tt is None:  # no plan-model form with >= MIN_VEC-element MN loads
                    continue
                n_forms += 1
                tile, trait = tt
                geo = (
                    f"{tile.tile_m}x{tile.tile_n}x{tile.tile_k}_w{tile.warp_m}x{tile.warp_n}"
                    f"_m{trait.vgpr_prefetch}_r{trait.lds_swizzle_r}"
                )
                self.assertIn(geo, menu)
                self.assertFalse(int(getattr(trait, "split_k_serial", 0)))
                self.assertIn("cdna_vgpr", r.candidate.name)
                self.assertEqual(r.spec.data.layout, lay)
                self.assertEqual(
                    (r.spec.tile.tile_m, r.spec.tile.tile_n), (tile.tile_m, tile.tile_n)
                )
                self.assertTrue(r.spec.trait.vp_mn_kv > 0)
                self.assertTrue(is_valid_spec(r.spec, arch="gfx942")[0])
                self.assertIsNone(launch_shape_error(r.spec, M, N, K))
        self.assertGreater(n_forms, 6)

    def test_model_off_falls_back(self):
        req = _req(4096, 4096, 4096, "RRR")
        with mock.patch.object(GP, "G5P_ON", False):
            self.assertIsNone(C.merge2_g5p_native_tile_trait(req))
            with mock.patch.object(C, "MERGE2_NATIVE_G5P", self.ALL):
                a = dispatch_gemm_fp16(req).spec
            with mock.patch.object(C, "MERGE2_NATIVE_G5P", ()):
                b = dispatch_gemm_fp16(req).spec
        self.assertEqual(a, b)

    def test_rcr_untouched(self):
        req = _req(4096, 4096, 4096)
        a = dispatch_gemm_fp16(req).spec
        with mock.patch.object(C, "MERGE2_NATIVE_G5P", self.ALL):
            b = dispatch_gemm_fp16(req).spec
        self.assertEqual(a, b)


class TestMergeR2Edges(_Clean):
    """Targeted edge cases of the interaction rules (found by host mutation testing)."""

    def test_native_g5p_skips_serial_plans(self):
        # long-K few-tile: the plan model's top plans reduce split-K in-launch (serial 2)
        req = _req(256, 256, 20736, "RRR")
        rreq = replace(req, layout="RCR", trans_a=False, trans_b=True)
        top = [p for _t, p in C._g5p_ranked(rreq) if p.geo != "skinny"][
            : C.MERGE2_NATIVE_G5P_TOP
        ]
        self.assertTrue(any(p.serial for p in top))
        # (with R3_NATIVE_G5P_SERIAL off they are skipped; on, they are served natively, tested in
        # test_gemm_r3_fewtile)
        with mock.patch.object(C, "R3_NATIVE_G5P_SERIAL", False):
            tt = C.merge2_g5p_native_tile_trait(req)
        self.assertTrue(tt is None or not int(getattr(tt[1], "split_k_serial", 0)))

    def test_native_model_rejects_rcr_versions(self):
        import json
        import shutil
        import tempfile
        from pathlib import Path

        src = Path(S.__file__).with_name("selector_models")
        tmp = tempfile.mkdtemp(prefix="m2sel-")
        try:
            for f in src.glob("gemm_native_gfx950.*"):
                shutil.copy(f, tmp)
            meta = Path(tmp) / "gemm_native_gfx950.meta.json"
            m = json.loads(meta.read_text())
            m["feature_version"] = S.FEATURE_VERSION  # an RCR feature version
            meta.write_text(json.dumps(m))
            with mock.patch.dict(os.environ, {S.MODEL_DIR_ENV: tmp}):
                S.clear_cache()
                self.assertIsNone(S.load_model("gfx950", "native"))
        finally:
            S.clear_cache()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_native_switch_independent_of_rcr_switch(self):
        # gfx942: the RCR selector ships off, the native one is switched on by env:
        # the override follows the native switch
        self.assertFalse(S.selector_enabled("gfx942"))
        req = _req(1000, 712, 3000, "RRR")
        with mock.patch.dict(os.environ, {S.NATIVE_SELECTOR_ENV: "1"}):
            S.clear_cache()
            self.assertIn("cdna_selected", dispatch_gemm_fp16(req).candidate.name)
        S.clear_cache()

    def test_native_g5p_compare_default(self):
        self.assertTrue(C.MERGE2_NATIVE_G5P_COMPARE)
        # the fitted-menu plan is modelled cheaper here: the comparison keeps it
        self.assertIsNone(C.merge2_g5p_native_tile_trait(_req(2048, 6144, 4096, "RRR")))
        with mock.patch.object(C, "MERGE2_NATIVE_G5P_COMPARE", False):
            self.assertIsNotNone(
                C.merge2_g5p_native_tile_trait(_req(2048, 6144, 4096, "RRR"))
            )

    def test_old_menu_key(self):
        self.assertEqual(P._old_menu_key("128x128x128"), "128x128x128_w2x2_m4_r1")
        self.assertEqual(P._old_menu_key("128x128x32"), "128x128x32_w2x4_m3_r1")

    def test_skinny_rcr_estimate_not_bounded(self):
        # the fixed-menu bound applies to register-staged picks only, not to an RCR gemm_skinny pick
        rreq = _req(16, 1024, 16384)
        res = dispatch_gemm_fp16(rreq)
        self.assertIn("cdna_skinny", res.candidate.name)
        with mock.patch.object(P, "_g5p_est", lambda *a, **k: 50.0):
            with mock.patch.object(P, "_geo_est", lambda *a, **k: 1.0):
                self.assertEqual(P._rcr_est(rreq, res), 50.0)

    def test_skinny_takeover_admits_without_table(self):
        with mock.patch.object(SK, "SKINNY_NATIVE_MARGIN", 10**9):
            self.assertTrue(SK.skinny_admit_region(_req(640, 768, 4096, "RRR"))[0])
            self.assertFalse(SK.skinny_admit_region(_req(1024, 1024, 4096, "RRR"))[0])

    def test_plan_pred_keys_on_split(self):
        from rocke.dispatch.gemm.g5_plan import plan_tile_trait
        from rocke.instances.common.gemm_universal import DataSpec, UniversalGemmSpec

        rreq = _req(256, 256, 20736)
        by_geo = {}
        for t, p in C._g5p_ranked(rreq):
            if p.geo != "skinny":
                by_geo.setdefault(p.geo, []).append((p, t))
        geo, plans = next(
            (g, v) for g, v in by_geo.items() if len({p.split for p, _ in v}) > 1
        )
        for p, t in plans[:2]:
            tile, trait = plan_tile_trait(p, 256, 256, 20736)
            spec = UniversalGemmSpec(
                name="x", tile=tile, trait=trait,
                data=DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", layout="RCR"),
            )  # fmt: skip
            self.assertAlmostEqual(P._g5p_plan_pred(rreq, spec), t, places=9)


if __name__ == "__main__":
    unittest.main()
