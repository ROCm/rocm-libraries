# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 NN / TN / TT: the learned native-form model over the g5 plan-model menu
(``rocke.dispatch.gemm.g5_native``) and its dispatcher / plan-chooser integration. CPU only.

* the shipped model's feature list, the knob / arch / layout gate (``active``);
* the form space (``plan_forms``): launchable forms only, the listed micro-blocks, the lane wrap only when the
  plain form does not launch, the forms without ``vp_unroll`` of an unrolled plan, the tuned trait set;
* ``rank_native`` ordering and knob-keyed caching; ``form_for`` / ``predicted_us`` / ``predict_spec`` agree;
* dispatch: the model's best form (or the model-off pick when the model predicts it faster), every pick valid and
  launchable; ``G5N_ON = False`` restores the model-off dispatch exactly;
* the plan chooser estimates both plans with the model (``plan.R3_PLAN_G5N``) and applies ``R3_PLAN_MARGIN``.
"""

from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace
from unittest import mock

from rocke.dispatch.gemm import (
    GemmRequest,
    dispatch_gemm_bf16,
    dispatch_gemm_fp16,
)
from rocke.dispatch.gemm import common as C
from rocke.dispatch.gemm import g5_native as GN
from rocke.dispatch.gemm import g5_plan as GP
from rocke.dispatch.gemm import plan as P
from rocke.instances.common.gemm_universal import (
    DataSpec,
    UniversalGemmSpec,
    is_valid_spec,
    launch_shape_error,
)

GEMM_SELECTOR_TESTS = True  # conftest: keep the shipped selector defaults

SHAPES = [
    (4096, 4096, 4096),
    (1640, 6216, 3000),
    (768, 1024, 13312),
    (2240, 1984, 2944),
    (5344, 232, 240),
    (14848, 4352, 5120),
    (1000, 3000, 2048),
]
LAYS = ("RRR", "CRR", "CCR")


def _req(M, N, K, lay="RRR", dt="fp16", arch="gfx942"):
    return GemmRequest(
        M=M, N=N, K=K, arch=arch, dtype=dt, layout=lay,
        trans_a=lay[0] == "C", trans_b=lay[1] == "C",
    )  # fmt: skip


def _disp(req):
    return (dispatch_gemm_fp16 if req.dtype == "fp16" else dispatch_gemm_bf16)(req)


def _clear():
    GN._RANK_CACHE.clear()
    C._G5P_CACHE.clear()
    C._R2_PLAN_CACHE.clear()


class _Clean(unittest.TestCase):
    def setUp(self):
        _clear()

    def tearDown(self):
        _clear()


class TestModelAndGate(_Clean):
    def test_shipped_defaults(self):
        self.assertTrue(GN.G5N_ON)
        self.assertTrue(GN.G5N_SERIAL)
        self.assertIn((2, 8), GN.G5N_MICRO)
        self.assertEqual(tuple(GN.G5N_LAYOUTS), LAYS)
        self.assertTrue(C.G5N_COMPARE_R2)
        self.assertEqual(
            tuple(GN.G5N_TRAIT_VARIANTS),
            ({"vp_mn_pipe": 0}, {"vp_mn_worder": 1}, {"vp_mn_order": 2}),
        )
        self.assertTrue(P.R3_PLAN_G5N)
        req = _req(2048, 2048, 2048, "RRR")
        forms = GN.plan_forms(req, GP.G5Plan("128x128x64_w4x2_m4_r0"))
        self.assertIn((2, 8), {(f.trait.vp_mn_kv, f.trait.mn_vec) for f in forms})

    def test_shipped_model_features(self):
        meta = json.loads((GN.MODEL_DIR / "g5_native_gfx942.meta.json").read_text())
        self.assertEqual(tuple(meta["features"]), GN.FEATURES)
        self.assertEqual(GN.FEATURES[: len(GP.FEATURES)], tuple(GP.FEATURES))
        self.assertIsNotNone(GN.load_model("gfx942"))
        self.assertIsNone(GN.load_model("gfx950"))

    def test_active_gate(self):
        for lay in LAYS:
            self.assertTrue(GN.active(_req(4096, 4096, 4096, lay)))
        self.assertFalse(GN.active(_req(4096, 4096, 4096, "RCR")))
        self.assertFalse(GN.active(_req(4096, 4096, 4096, "RRR", arch="gfx950")))
        with mock.patch.object(GN, "G5N_ON", False):
            self.assertFalse(GN.active(_req(4096, 4096, 4096, "RRR")))
        with mock.patch.object(GN, "G5N_ARCHES", ()):
            self.assertFalse(GN.active(_req(4096, 4096, 4096, "RRR")))
        with mock.patch.object(GN, "G5N_LAYOUTS", ("RRR",)):
            self.assertTrue(GN.active(_req(4096, 4096, 4096, "RRR")))
            self.assertFalse(GN.active(_req(4096, 4096, 4096, "CCR")))

    def test_feature_row(self):
        menu = GP.load_menu("gfx942")
        p = GP.G5Plan("256x256x64_w2x2_m4_r0")
        tr = {"vp_mn_kv": 8, "mn_vec": 8, "vp_mn_wrap": 0, "sched_strategy": "iterative-minreg",
              "vp_mn_pipe": 1, "vp_mn_order": 2, "vp_mn_worder": 1, "lds_swizzle_r": 0, "vp_unroll": 1}  # fmt: skip
        row = GN.form_features(
            4096, 4000, 2048, True, "CRR", p, tr, menu, 304, 1.5, sku=0.0
        )
        f = dict(zip(GN.FEATURES, row))
        self.assertEqual(len(row), len(GN.FEATURES))
        self.assertEqual((f["a_mn"], f["b_mn"], f["n_mn"]), (1.0, 1.0, 2.0))
        self.assertEqual((f["kv"], f["mv"], f["mb"], f["wrap"]), (8.0, 8.0, 64.0, 0.0))
        self.assertEqual(
            (f["minreg"], f["pipe"], f["order"], f["worder"]), (1.0, 1.0, 2.0, 1.0)
        )
        self.assertEqual(
            (f["ld_a8"], f["ld_b8"], f["ld_a16"], f["ld_b16"]), (1.0, 1.0, 1.0, 1.0)
        )
        self.assertEqual(
            (f["ln_rcr"], f["sku"], f["unroll"], f["bf16"]), (1.5, 0.0, 1.0, 1.0)
        )
        # 256 x 64 tile / (8 x 8) = 256 micro-blocks over 256 threads: one round, nothing repeated
        self.assertEqual((f["rounds_a"], f["rounds_b"], f["rep"]), (1.0, 1.0, 0.0))
        rcr = dict(
            zip(
                GN.FEATURES,
                GN.form_features(
                    4096, 4000, 2048, False, "RCR", p, tr, menu, 304, None
                ),
            )
        )
        self.assertEqual(
            (rcr["a_mn"], rcr["b_mn"], rcr["kv"], rcr["ln_rcr"]),
            (0.0, 0.0, GN.G5N_NA, GN.G5N_NA),
        )
        self.assertEqual(rcr["sku"], GN.G5N_SKU)
        nn = dict(
            zip(
                GN.FEATURES,
                GN.form_features(
                    4100, 4004, 2048, False, "RRR", p, tr, menu, 304, None
                ),
            )
        )
        self.assertEqual(
            (nn["a_mn"], nn["b_mn"], nn["ld_b8"], nn["ld_b16"]), (0.0, 1.0, 0.0, 0.0)
        )

    def test_rounds_and_rep(self):
        self.assertEqual(GN._rounds(256, 64, 8, 8, 256), (1, 0.0))
        self.assertEqual(GN._rounds(128, 64, 8, 8, 512), (1, 3.0))
        r, rep = GN._rounds(224, 64, 8, 8, 256)
        self.assertEqual(r, 1)
        self.assertAlmostEqual(rep, (256 - 224) / 224)
        # 56 x 16 = 896 micro-blocks over 256 threads: 4 rounds, the last one partial
        r, rep = GN._rounds(224, 64, 4, 4, 256)
        self.assertEqual(r, 4)
        self.assertAlmostEqual(rep, (1024 - 896) / 896)

    def test_feature_rep_of_wrapped_form(self):
        menu = GP.load_menu("gfx942")
        p = GP.G5Plan("224x256x64_w2x2_m4_r0")
        tr = {"vp_mn_kv": 8, "mn_vec": 8, "vp_mn_wrap": 1}
        f = dict(
            zip(
                GN.FEATURES,
                GN.form_features(
                    4096, 4096, 4096, False, "CCR", p, tr, menu, 304, None
                ),
            )
        )
        # TT: A (224 x 64) is the MN operand: 28 x 8 = 224 micro-blocks over 256 threads
        self.assertEqual((f["rounds_a"], f["rounds_b"], f["wrap"]), (1.0, 0.0, 1.0))
        self.assertAlmostEqual(f["rep"], 32 / 224)

    def test_ln_rcr_of(self):
        un = next(g for g in GP.load_menu("gfx942") if GP.plan_unroll(GP.G5Plan(g)))
        nu = next(g for g in GP.load_menu("gfx942") if not GP.plan_unroll(GP.G5Plan(g)))
        self.assertAlmostEqual(GN.ln_rcr_of(10.0, GP.G5Plan(nu)), math.log(10.0))
        self.assertAlmostEqual(
            GN.ln_rcr_of(10.0, GP.G5Plan(un)), math.log(10.0 / GP.G5P_VP_UNROLL_SCALE)
        )
        self.assertIsNone(GN.ln_rcr_of(0.0, GP.G5Plan(nu)))
        self.assertIsNone(GN.ln_rcr_of(None, GP.G5Plan(nu)))

    def test_old_geo_of_menu(self):
        self.assertEqual(C._old_geo_of_menu("256x256x64_w2x2_m4_r0"), "256x256x64")
        self.assertIsNone(C._old_geo_of_menu("112x112x32_w1x1_m4_r0"))


class TestForms(_Clean):
    def test_forms_launchable_and_in_space(self):
        menu = sorted(GP.load_menu("gfx942"))
        for (M, N, K), lay in zip(SHAPES, LAYS * 3):
            req = _req(M, N, K, lay)
            for geo in menu[::9]:
                for f in GN.plan_forms(req, GP.G5Plan(geo)):
                    self.assertTrue(is_valid_spec(f, arch="gfx942")[0], f.kernel_name())
                    self.assertIsNone(launch_shape_error(f, M, N, K))
                    self.assertEqual(f.data.layout, lay)
                    t = f.trait
                    tuned = C.R2_VP_MN_TUNED.get((lay, C._old_geo_of_menu(geo) or ""))
                    mic = (t.vp_mn_kv, t.mn_vec)
                    self.assertTrue(
                        mic in GN.G5N_MICRO or (tuned and mic == tuple(tuned[0])), mic
                    )
                    pipes = {C.R2_VP_MN_TRAIT["vp_mn_pipe"]} | {
                        v["vp_mn_pipe"]
                        for v in GN.G5N_TRAIT_VARIANTS
                        if "vp_mn_pipe" in v
                    }
                    self.assertIn(t.vp_mn_pipe, pipes)

    def test_wrap_only_when_needed(self):
        req = _req(4096, 4096, 4096, "RRR")
        for geo in (
            "256x256x64_w2x2_m4_r0",
            "192x256x64_w4x2_m4_r0",
            "112x112x32_w1x1_m4_r0",
        ):
            for f in GN.plan_forms(req, GP.G5Plan(geo)):
                if f.trait.vp_mn_wrap:
                    plain = replace(f, trait=replace(f.trait, vp_mn_wrap=0))
                    ok = is_valid_spec(plain, arch="gfx942")[
                        0
                    ] and not launch_shape_error(plain, 4096, 4096, 4096)
                    self.assertFalse(ok, f.kernel_name())

    def test_unroll_both(self):
        req = _req(2048, 2048, 2048, "CCR")
        un = next(g for g in sorted(GP.load_menu("gfx942")) if GP.plan_unroll(GP.G5Plan(g))
                  and GN.plan_forms(req, GP.G5Plan(g)))  # fmt: skip
        fs = GN.plan_forms(req, GP.G5Plan(un))
        self.assertEqual({f.trait.vp_unroll for f in fs}, {0, 1})
        with mock.patch.object(GN, "G5N_UNROLL_BOTH", False):
            self.assertEqual(
                {f.trait.vp_unroll for f in GN.plan_forms(req, GP.G5Plan(un))}, {1}
            )

    def test_tuned_form(self):
        menu = GP.load_menu("gfx942")
        hits = 0
        for (lay, old), (mic, trs, _f) in sorted(C.R2_VP_MN_TUNED.items()):
            g = C.G5_VGPR_GEOS[old]
            key = f"{g[0]}x{g[1]}x{g[2]}_w{g[3]}x{g[4]}_m{g[9]}_r{g[10]}"
            if key not in menu:
                continue
            req = _req(4096, 4096, 4096, lay)
            fs = GN.plan_forms(req, GP.G5Plan(key))
            tuned = [f for f in fs if all(getattr(f.trait, k) == v for k, v in trs.items())
                     and (f.trait.vp_mn_kv, f.trait.mn_vec) == tuple(mic)]  # fmt: skip
            if not tuned:
                continue  # the tuned micro-block does not launch at this shape
            hits += 1
            with mock.patch.object(GN, "G5N_TUNED_FORMS", False), mock.patch.object(
                GN, "G5N_TRAIT_VARIANTS", ()
            ):
                fs2 = GN.plan_forms(req, GP.G5Plan(key))
            self.assertFalse(
                [f for f in fs2 if GN.body_key(f) == GN.body_key(tuned[0])]
            )
            # a split-K plan of the geometry gets no tuned form (tuned on split-free bodies)
            if GP.parse_geo(key)[2] * 4 <= 4096:
                with mock.patch.object(GN, "G5N_TRAIT_VARIANTS", ()):
                    fs3 = GN.plan_forms(req, GP.G5Plan(key, 2, 0))
                self.assertFalse([f for f in fs3 if all(getattr(f.trait, k) == v for k, v in trs.items())
                                  and trs])  # fmt: skip
        self.assertGreater(hits, 0)

    def test_form_for_matches_plan_not_trait(self):
        # a form whose trait differs from its plan's body (a tuned swizzle shift) is still the plan's form
        req = _req(4096, 4096, 4096, "RRR")
        _u, p, f = GN.best_form(req)
        tile, trait = GP.plan_tile_trait(p, 4096, 4096, 4096)
        body = UniversalGemmSpec(name="x", tile=tile, trait=trait,
                                 data=DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", layout="RCR"),
                                 wave_size=64)  # fmt: skip
        fake = [
            (
                1.0,
                p,
                replace(
                    f, trait=replace(f.trait, lds_swizzle_r=f.trait.lds_swizzle_r + 1)
                ),
            )
        ]
        with mock.patch.object(GN, "rank_native", lambda r: fake):
            self.assertIs(GN.form_for(req, body), fake[0][2])
            other = replace(
                body,
                trait=replace(body.trait, lds_swizzle_r=body.trait.lds_swizzle_r + 1),
            )
            self.assertIsNone(GN.form_for(req, other))

    def test_trait_variants_knob(self):
        req = _req(2048, 2048, 2048, "RRR")
        p = GP.G5Plan("128x128x64_w4x2_m4_r0")
        with mock.patch.object(GN, "G5N_TRAIT_VARIANTS", ()):
            base = GN.plan_forms(req, p)
            r_off = GN.rank_native(req)
        tv = ({"vp_mn_pipe": 0}, {"vp_mn_worder": 1})
        with mock.patch.object(GN, "G5N_TRAIT_VARIANTS", tv):
            fs = GN.plan_forms(req, p)
            r_on = GN.rank_native(req)
        self.assertGreater(len(fs), len(base))
        for f in fs:
            self.assertTrue(is_valid_spec(f, arch="gfx942")[0])
        self.assertTrue(any(f.trait.vp_mn_pipe == 0 for f in fs))
        self.assertTrue(any(f.trait.vp_mn_worder == 1 for f in fs))
        self.assertFalse(any(f.trait.vp_mn_pipe == 0 for f in base))
        # the rank cache is keyed by the variants
        self.assertNotEqual(len(r_on), len(r_off))

    def test_micro_knob(self):
        req = _req(4096, 4096, 4096, "RRR")
        p = GP.G5Plan("256x256x64_w2x2_m4_r0")
        with mock.patch.object(GN, "G5N_MICRO", ((4, 4),)), mock.patch.object(
            GN, "G5N_TUNED_FORMS", False
        ):
            fs = GN.plan_forms(req, p)
        self.assertTrue(fs)
        self.assertEqual({(f.trait.vp_mn_kv, f.trait.mn_vec) for f in fs}, {(4, 4)})


class TestRank(_Clean):
    def test_sorted_and_consistent(self):
        for (M, N, K), lay in zip(SHAPES, LAYS * 3):
            req = _req(M, N, K, lay, dt="bf16" if M % 2 else "fp16")
            r = GN.rank_native(req)
            self.assertTrue(r, (M, N, K, lay))
            ts = [u for u, _p, _f in r]
            self.assertEqual(ts, sorted(ts))
            u, p, f = r[0]
            self.assertEqual(GN.best_form(req)[2], f)
            self.assertAlmostEqual(GN.predicted_us(req, f), u)
            self.assertAlmostEqual(GN.predict_spec(req, f), u, delta=1e-6 * u)
            tile, trait = GP.plan_tile_trait(p, M, N, K)
            body = UniversalGemmSpec(name="x", tile=tile, trait=trait,
                                     data=DataSpec(dtype_a=req.dtype, dtype_b=req.dtype, dtype_c=req.dtype,
                                                   layout="RCR"), wave_size=64)  # fmt: skip
            self.assertEqual(GN.body_key(GN.form_for(req, body)), GN.body_key(f))
            plans = {(q.geo, q.split) for _u, q, _f in r}
            self.assertLessEqual(len(plans), GN.G5N_TOP)
            self.assertTrue(all(q.geo != "skinny" for _u, q, _f in r))
            if not GN.G5N_SERIAL:
                self.assertTrue(all(not q.serial for _u, q, _f in r))

    def test_serial_plans_and_twins(self):
        # long K, few tiles: the plan model ranks serial split-K plans
        req = _req(768, 1024, 13312, "RRR")
        rreq = replace(req, layout="RCR", trans_a=False, trans_b=True)
        ser = [p for _t, p in C._g5p_ranked(rreq) if p.geo != "skinny" and p.serial]
        self.assertTrue(ser)
        with mock.patch.object(GN, "G5N_SERIAL", True):
            on = {q for _u, q, _f in GN.rank_native(req)}
        with mock.patch.object(GN, "G5N_SERIAL", False):
            off = {q for _u, q, _f in GN.rank_native(req)}
        self.assertTrue(any(q.serial for q in on))
        self.assertFalse(any(q.serial for q in off))
        for q in on:
            if q.serial:
                self.assertIn(GP.G5Plan(q.geo, q.split, 0), on)
        # a serial form launches and keeps the in-launch reduction
        f = (
            next(f for _u, q, f in GN.rank_native(req) if q.serial)
            if GN.G5N_SERIAL
            else None
        )
        if f is not None:
            self.assertEqual(int(f.trait.split_k_serial), int(next(q for _u, q, _f in GN.rank_native(req)
                                                                   if q.serial).serial))  # fmt: skip
            self.assertTrue(is_valid_spec(f, arch="gfx942")[0])

    def test_inactive_empty_and_cache_keyed(self):
        req = _req(4096, 4096, 4096, "RRR")
        self.assertTrue(GN.rank_native(req))
        with mock.patch.object(GN, "G5N_ON", False):
            self.assertEqual(GN.rank_native(req), [])
            self.assertIsNone(GN.best_form(req))
        with mock.patch.object(GN, "G5N_TOP", 1):
            self.assertEqual(len({q.geo for _u, q, _f in GN.rank_native(req)}), 1)
        self.assertEqual(GN.rank_native(_req(4096, 4096, 4096, "RCR")), [])

    def test_sku_knob_changes_prediction(self):
        req = _req(4096, 4096, 4096, "CCR")
        u1 = GN.best_form(req)[0]
        with mock.patch.object(GN, "G5N_SKU", 0.0):
            # no cache clear: the rank cache is keyed by the knob
            u0 = GN.best_form(req)[0]
        self.assertNotEqual(u0, u1)

    def test_spec_plan_swizzle_fallback(self):
        menu = GP.load_menu("gfx942")
        key = next(
            k for k in sorted(menu) if k.endswith("_r0") and k[:-1] + "1" not in menu
        )
        tile, trait = GP.plan_tile_trait(GP.G5Plan(key), 4096, 4096, 4096)
        body = UniversalGemmSpec(name="x", tile=tile, trait=replace(trait, lds_swizzle_r=1),
                                 data=DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", layout="RCR"),
                                 wave_size=64)  # fmt: skip
        self.assertEqual(GN.spec_plan(body, "gfx942").geo, key)
        self.assertEqual(GN.spec_plan(body, "gfx942", exact=True).geo, key[:-1] + "1")

    def test_predict_spec_layout_and_rank(self):
        req = _req(4096, 4096, 4096, "RRR")
        r = GN.rank_native(req)
        u, p, f = r[-1]
        self.assertAlmostEqual(GN.predict_spec(req, f), u, delta=1e-6 * u)
        # off the ranking (another shape): the form's own layout, not the RCR body
        req2 = _req(3968, 4096, 4096, "RRR")
        nat = GN.predict_spec(req2, f)
        rcr = GN.predict_spec(
            replace(req2, layout="RCR"), replace(f, data=replace(f.data, layout="RCR"))
        )
        self.assertNotAlmostEqual(nat, rcr)
        # a ranked form's prediction is the ranking's (its plan), even when the trait-derived plan differs
        fake = [(123.0, p, f)]
        with mock.patch.object(GN, "rank_native", lambda q: fake):
            self.assertEqual(GN.predict_spec(req, f), 123.0)

    def test_spec_plan(self):
        req = _req(4096, 4096, 4096, "RRR")
        _u, p, f = GN.best_form(req)
        self.assertEqual(GN.spec_plan(f, "gfx942"), p)
        off = replace(f, tile=replace(f.tile, tile_m=96, tile_n=96))
        self.assertTrue(GN.spec_plan(off, "gfx942").geo.startswith("96x96x"))
        rcr = replace(f, trait=replace(f.trait, vgpr_prefetch=0))
        self.assertIsNone(GN.spec_plan(rcr, "gfx942"))
        self.assertIsNone(GN.predict_spec(req, rcr))


class TestDispatch(_Clean):
    def test_picks_valid_and_from_model(self):
        for M, N, K in SHAPES:
            for lay in LAYS:
                req = _req(M, N, K, lay)
                res = _disp(req)
                s = res.spec
                if not int(getattr(getattr(s, "trait", None), "vgpr_prefetch", 0) or 0):
                    continue
                self.assertTrue(is_valid_spec(s, arch="gfx942")[0])
                self.assertIsNone(launch_shape_error(s, M, N, K))
                self.assertEqual(s.data.layout, lay)
                best = GN.best_form(req)
                if GN.body_key(replace(s, name="g5n")) == GN.body_key(best[2]):
                    continue
                # otherwise the model-off pick, predicted faster by the model
                r2 = C._r2_native_pick(req)
                self.assertIsNotNone(r2)
                self.assertLess(GN.predict_spec(req, r2), best[0] * GN.G5N_MARGIN)

    def test_compare_r2_off_takes_model_best(self):
        with mock.patch.object(C, "G5N_COMPARE_R2", False):
            for M, N, K in SHAPES[:4]:
                for lay in LAYS:
                    req = _req(M, N, K, lay)
                    s = _disp(req).spec
                    if int(getattr(s.trait, "vgpr_prefetch", 0)):
                        self.assertEqual(
                            GN.body_key(replace(s, name="g5n")),
                            GN.body_key(GN.best_form(req)[2]),
                        )

    def test_margin_prefers_r2(self):
        req = _req(4096, 4096, 4096, "RRR")
        with mock.patch.object(GN, "G5N_MARGIN", 1e6):
            tt = C.g5n_tile_trait(req)
        self.assertIsNone(tt)
        with mock.patch.object(GN, "G5N_MARGIN", 1e-6):
            tt = C.g5n_tile_trait(req)
        self.assertIsNotNone(tt)

    def test_off_restores_round2(self):
        for M, N, K in SHAPES:
            for lay in LAYS:
                req = _req(M, N, K, lay)
                with mock.patch.object(GN, "G5N_ON", False):
                    _clear()
                    off = _disp(req).spec
                    tt_off = C.g5_vgpr_tile_trait(req)
                C._G5N_BYPASS[0] += 1
                try:
                    _clear()
                    byp = _disp(req).spec
                    tt_byp = C.g5_vgpr_tile_trait(req)
                finally:
                    C._G5N_BYPASS[0] -= 1
                self.assertEqual(GN.body_key(off), GN.body_key(byp))
                self.assertEqual(tt_off, tt_byp)

    def test_r2_cache_keyed_by_model_state(self):
        req = _req(4096, 4096, 4096, "CRR")
        a = C.r2_vp_mn_plans(req)
        C._G5N_BYPASS[0] += 1
        try:
            b = C.r2_vp_mn_plans(req)
        finally:
            C._G5N_BYPASS[0] -= 1
        self.assertEqual(len(C._R2_PLAN_CACHE), 2)
        # the bypassed costs are the model-off costs
        with mock.patch.object(GN, "G5N_ON", False):
            C._R2_PLAN_CACHE.clear()
            off = C.r2_vp_mn_plans(req)
        self.assertEqual(off, b)
        self.assertTrue(a)

    def test_sweep_distinct_model_bodies(self):
        req = _req(4096, 4096, 4096, "CCR")
        tts = C.g5_vgpr_sweep_tile_traits(req)
        self.assertTrue(tts)
        self.assertLessEqual(len(tts), C.G5_VGPR_SWEEP)
        self.assertEqual(len({repr(t) for t in tts}), len(tts))
        u, p, _f = GN.best_form(req)
        self.assertEqual(tts[0], GP.plan_tile_trait(p, 4096, 4096, 4096))
        seen = []
        for _u, q, _f in GN.rank_native(req):
            if q not in seen:
                seen.append(q)
        want = [GP.plan_tile_trait(q, 4096, 4096, 4096) for q in seen][
            : C.G5_VGPR_SWEEP
        ]
        self.assertEqual(tts, want)


class TestPlan(_Clean):
    def test_model_estimates(self):
        for M, N, K in SHAPES[:4]:
            for lay in LAYS:
                req = _req(M, N, K, lay)
                pl = P.gemm_plan(req)
                nat = _disp(req)
                rreq = replace(req, layout="RCR", trans_a=False, trans_b=True)
                rcr = _disp(rreq)
                en, er = P._g5n_est(req, nat), P._g5n_est(rreq, rcr)
                if en is None or er is None:
                    continue
                if pl.kind == "native":
                    self.assertAlmostEqual(pl.est_us, en, delta=1e-6 * en)
                elif pl.kind == "prepass":
                    self.assertAlmostEqual(pl.alt_est_us, en, delta=1e-6 * en)
                    self.assertLess(pl.est_us * P.R3_PLAN_MARGIN, en)

    def test_prepass_and_partial_use_model(self):
        # sentinel model estimates: native slow, RCR pre-pass fast, TN partial forms fastest
        req = _req(4096, 4096, 4096, "CRR")

        def fake(r, res):
            lay = str(r.layout).upper()
            return {"CRR": 1000.0, "RCR": 400.0, "RRR": 100.0, "CCR": 150.0}[lay]

        with mock.patch.object(P, "_g5n_est", fake):
            pl = P.gemm_plan(req)
        self.assertEqual(pl.kind, "prepass_a")
        tp = P.r2_transpose_cost_us(4096, 4096, fast=True)
        self.assertAlmostEqual(pl.est_us, 100.0 + tp)
        self.assertAlmostEqual(pl.alt_est_us, 1000.0)
        with mock.patch.object(P, "_g5n_est", fake), mock.patch.object(
            P, "R2_PLAN_PARTIAL", False
        ):
            pl = P.gemm_plan(req)
        self.assertEqual(pl.kind, "prepass")
        self.assertAlmostEqual(pl.est_us, 400.0 + 2 * tp)

    def test_margin_and_knob(self):
        req = _req(4096, 4096, 4096, "CRR")
        with mock.patch.object(P, "R3_PLAN_MARGIN", 1e6):
            self.assertEqual(P.gemm_plan(req).kind, "native")
        with mock.patch.object(P, "R3_PLAN_MARGIN", 1e-6):
            self.assertNotEqual(P.gemm_plan(req).kind, "native")
        with mock.patch.object(P, "R3_PLAN_G5N", False):
            pl = P.gemm_plan(req)
            self.assertNotAlmostEqual(pl.est_us, P._g5n_est(req, _disp(req)) or -1.0)


if __name__ == "__main__":
    unittest.main()
