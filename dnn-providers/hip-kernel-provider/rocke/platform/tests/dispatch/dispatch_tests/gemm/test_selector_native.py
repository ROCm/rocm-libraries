# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The native (NN / TN / TT) learned GEMM selection (``rocke.dispatch.gemm.selector``).

CPU only. Covers the layout features (and that the RCR model's feature vector
does not change), the native model files, the native candidate set (every
candidate in the request's layout and inside the dispatcher's gates; generated
ones inside ``generated_spec_error``), the decision wiring (the native switch,
the global switch, pinned requests, the ``selected`` candidate in every layout)
and the two-stage neighbour knob.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from rocke.dispatch.gemm import (
    GEMM_BF16_REGISTRY,
    GEMM_FP16_REGISTRY,
    GemmRequest,
    dispatch_gemm_bf16,
    dispatch_gemm_fp16,
)
from rocke.dispatch.gemm import selector as S
from rocke.instances.common.gemm_universal import is_valid_spec, launch_shape_error

GEMM_SELECTOR_TESTS = True  # conftest: keep the shipped selector default on

_LAYOUTS = {"RRR": (False, False), "CRR": (True, False), "CCR": (True, True)}
_SHAPES = (
    (1000, 712, 3000),
    (256, 8192, 8192),
    (130, 4000, 20000),
    (2048, 3000, 5000),
    (512, 512, 512),
)


def _req(M, N, K, lay, dt="fp16", arch="gfx950", **kw):
    ta, tb = _LAYOUTS.get(lay, (False, True))
    return GemmRequest(
        M=M, N=N, K=K, arch=arch, dtype=dt, layout=lay, trans_a=ta, trans_b=tb, **kw
    )


def _reg(dt):
    return GEMM_FP16_REGISTRY if dt == "fp16" else GEMM_BF16_REGISTRY


def _disp(dt):
    return dispatch_gemm_fp16 if dt == "fp16" else dispatch_gemm_bf16


class _Clean(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ)
        self._env.start()
        os.environ.pop(S.NATIVE_SELECTOR_ENV, None)
        os.environ.pop(S.SELECTOR_ENV, None)
        # gemm_skinny's native forms (skinny.SKINNY_NATIVE) take the
        # few-tile shapes of these tests ahead of the UniversalGemm rule pick
        # (a gemm_skinny rule pick keeps the rule: TestSkinnyRulePick)
        from rocke.dispatch.gemm import skinny as SK

        self._sk = mock.patch.object(SK, "SKINNY_NATIVE", False)
        self._sk.start()
        S.clear_cache()

    def tearDown(self):
        self._sk.stop()
        self._env.stop()
        S.clear_cache()


class TestSkinnyRulePick(unittest.TestCase):
    """A native gemm_skinny rule pick is kept by the native selector
    (its model ranks UniversalGemm specs only)."""

    def test_skinny_rule_pick_kept(self):
        req = _req(384, 512, 3000, "CCR")
        S.clear_cache()
        rule = GEMM_FP16_REGISTRY.select(req)
        spec = rule.select_spec(req)
        self.assertEqual(type(spec).__name__, "SkinnyGemmSpec")
        self.assertIsNone(S.decide(req, GEMM_FP16_REGISTRY))
        self.assertEqual(dispatch_gemm_fp16(req).spec, spec)
        S.clear_cache()


class TestLayoutFeatures(_Clean):
    def test_layout_features_per_layout(self):
        ring = {"direct_to_lds": True, "dtl_prefetch": True, "dtl_stages": 2}
        cases = {
            "RCR": (0, 0, 0),
            "RRR": (0, 1, 1),
            "CRR": (1, 1, 2),
            "CCR": (1, 0, 1),
        }
        for lay, (a, bb, n) in cases.items():
            f = S._layout_features({"layout": lay, **ring})
            self.assertEqual((f["x_a_mn"], f["x_b_mn"], f["x_n_mn"]), (a, bb, n), lay)
            self.assertEqual(f["x_mn_dtl"], float(n > 0), lay)
            self.assertEqual(f["x_mn_vgpr"], 0.0, lay)
            g = S._layout_features({"layout": lay})
            self.assertEqual(g["x_mn_vgpr"], float(n > 0), lay)
            self.assertEqual(g["x_mn_dtl"], 0.0, lay)

    def test_rcr_model_vector_ignores_layout(self):
        m = S.load_model("gfx950")
        self.assertIsNotNone(m)
        self.assertFalse(any(n.startswith("x_") for n in m.features))
        req = _req(8192, 8192, 8192, "RCR")
        spec = GEMM_FP16_REGISTRY.select(req).select_spec(req)
        flat = S.flatten_spec(spec)
        self.assertEqual(flat["layout"], "RCR")
        no_lay = {k: v for k, v in flat.items() if k != "layout"}
        d = S.trait_defaults(m.meta)
        v1 = S.feature_vector(m.features, "gfx950", "fp16", 8192, 8192, 8192, flat, trait_defaults=d)
        v2 = S.feature_vector(m.features, "gfx950", "fp16", 8192, 8192, 8192, no_lay, trait_defaults=d)
        self.assertEqual(v1, v2)

    def test_origin_class_form(self):
        self.assertEqual(S.origin_class("form:same"), "form")
        self.assertEqual(S.origin_class("neighbour:mn_chunk=64"), "neighbour")
        self.assertEqual(S.origin_class("registered"), "registered")


class TestNativeModel(_Clean):
    def test_models_load(self):
        for arch in ("gfx950", "gfx942"):
            m = S.load_model(arch, "native")
            self.assertIsNotNone(m, arch)
            self.assertEqual(m.meta["feature_version"], S.NATIVE_FEATURE_VERSION)
            self.assertTrue(any(n.startswith("x_") for n in m.features), arch)
            self.assertEqual(m.meta["library"], [])

    def test_wrong_version_is_ignored(self):
        with mock.patch.object(S, "NATIVE_FEATURE_VERSION", "nope"):
            with mock.patch.dict(S.MODEL_KINDS, {"native": ("gemm_native", "nope-x")}):
                S.clear_cache()
                self.assertIsNone(S.load_model("gfx950", "native"))
        S.clear_cache()

    def test_sidecar_default_on_and_switches(self):
        for arch in ("gfx950", "gfx942"):
            m = S.load_model(arch, "native")
            self.assertEqual(S.native_selector_enabled(arch), bool(m.meta["default_on"]))
            with mock.patch.dict(os.environ, {S.NATIVE_SELECTOR_ENV: "0"}):
                self.assertFalse(S.native_selector_enabled(arch))
            with mock.patch.dict(os.environ, {S.NATIVE_SELECTOR_ENV: "1"}):
                self.assertTrue(S.native_selector_enabled(arch))
            with mock.patch.dict(os.environ, {S.SELECTOR_ENV: "0", S.NATIVE_SELECTOR_ENV: "1"}):
                self.assertFalse(S.native_selector_enabled(arch))
        with mock.patch.dict(S.MODEL_KINDS, {"native": ("gemm_native", "never")}):
            S.clear_cache()
            self.assertEqual(S.native_selector_enabled("gfx950"), S.SELECTOR_NATIVE_DEFAULT_ON)
        S.clear_cache()


class TestNativeCandidates(_Clean):
    def test_candidates_gated_and_in_layout(self):
        for lay in _LAYOUTS:
            for M, N, K in _SHAPES[:3]:
                for dt in ("fp16", "bf16"):
                    req = _req(M, N, K, lay, dt)
                    reg = _reg(dt)
                    cand = reg.select(req)
                    rule = (cand, cand.select_spec(req))
                    cs = S.native_candidates(req, reg, rule_pick=rule)
                    self.assertTrue(cs, (lay, M, N, K))
                    # a gemm_skinny native rule pick (skinny.SKINNY_NATIVE)
                    # is not a UniversalGemm candidate (decide keeps it)
                    want = 1 if S._universal(rule[1]) else 0
                    self.assertEqual(sum(c.is_rule_pick for c in cs), want)
                    for c in cs:
                        self.assertEqual(c.spec.data.layout, lay)
                        self.assertIsNone(S._gate(req, c.spec), c.origin)
                        ok, why = is_valid_spec(c.spec, arch="gfx950")
                        self.assertTrue(ok, why)
                        self.assertIsNone(launch_shape_error(c.spec, M, N, K))
                        if c.origin.startswith("neighbour"):
                            parent = next(
                                (
                                    p
                                    for p in cs
                                    if p.candidate_name == c.candidate_name
                                    and not p.origin.startswith("neighbour")
                                ),
                                None,
                            )
                            self.assertIsNotNone(parent)

    def test_forms_cover_native_alternatives(self):
        from rocke.dispatch.gemm.common import native_layout_alternatives

        req = _req(8192, 8192, 8192, "RRR")
        rreq = _req(8192, 8192, 8192, "RCR")
        rspec = GEMM_FP16_REGISTRY.select(rreq).select_spec(rreq)
        alts, _ = native_layout_alternatives(req, rspec)
        cand = GEMM_FP16_REGISTRY.select(req)
        cs = S.native_candidates(
            req, GEMM_FP16_REGISTRY, rule_pick=(cand, cand.select_spec(req))
        )
        keys = {S._spec_key(c.spec) for c in cs}
        for nm, s in alts:
            if S._gate(req, s) is None:
                self.assertIn(S._spec_key(s), keys, nm)

    def test_mn_levers_only_on_mn_ring(self):
        req = _req(130, 4000, 20000, "CRR")
        cand = GEMM_FP16_REGISTRY.select(req)
        cs = S.native_candidates(
            req, GEMM_FP16_REGISTRY, rule_pick=(cand, cand.select_spec(req))
        )
        for c in cs:
            if c.origin.startswith(("neighbour:mn_chunk", "neighbour:mn_pad")):
                self.assertTrue(S._mn_ring(c.spec), c.origin)

    def test_wrong_layout_specs_dropped(self):
        req = _req(1000, 712, 3000, "RRR")
        rreq = _req(1000, 712, 3000, "RCR")
        cand = GEMM_FP16_REGISTRY.select(rreq)
        rspec = cand.select_spec(rreq)

        class _Reg:
            def combos(self, r, include_opt_in=False):
                return [(cand, rspec)]

        cs = S.native_candidates(req, _Reg(), sources=("registered",))
        self.assertEqual(cs, [])

    def test_generated_need_multiples_of_8(self):
        # (SELECTOR_LD_ANY admits DTL-ring mn_ld_any forms; off here)
        for lay in _LAYOUTS:
            req = _req(130, 4000, 20000, lay)
            cand = GEMM_FP16_REGISTRY.select(req)
            with mock.patch.object(S, "SELECTOR_LD_ANY", False):
                cs = S.native_candidates(
                    req, GEMM_FP16_REGISTRY, rule_pick=(cand, cand.select_spec(req))
                )
            self.assertTrue(cs)
            self.assertFalse([c for c in cs if c.origin.startswith("neighbour")], lay)

    def test_generated_ld_any_forms(self):
        # generated forms on a misaligned M / N are DTL-ring mn_ld_any forms
        # with exact bounds only; a misaligned K keeps every generated spec out
        for lay in _LAYOUTS:
            req = _req(130, 4000, 20000, lay)
            cand = GEMM_FP16_REGISTRY.select(req)
            cs = S.native_candidates(
                req, GEMM_FP16_REGISTRY, rule_pick=(cand, cand.select_spec(req))
            )
            for c in cs:
                if c.origin.startswith("neighbour"):
                    tr = c.spec.trait
                    self.assertTrue(
                        tr.mn_ld_any and tr.direct_to_lds and tr.dtl_tight_rsrc,
                        (lay, c.origin),
                    )
            req = _req(130, 4000, 20004, lay)
            cand = GEMM_FP16_REGISTRY.select(req)
            cs = S.native_candidates(
                req, GEMM_FP16_REGISTRY, rule_pick=(cand, cand.select_spec(req))
            )
            self.assertFalse([c for c in cs if c.origin.startswith("neighbour")], lay)

    def test_mn_levers_not_on_vgpr_bases(self):
        vgpr_bases = 0
        for lay in _LAYOUTS:
            for M, N, K in ((1000, 712, 3000), (512, 512, 512), (2048, 3000, 5000)):
                req = _req(M, N, K, lay)
                cand = GEMM_FP16_REGISTRY.select(req)
                with mock.patch.object(S, "SELECTOR_NATIVE_NEIGHBOUR_BASES", 0):
                    cs = S.native_candidates(
                        req, GEMM_FP16_REGISTRY, rule_pick=(cand, cand.select_spec(req))
                    )
                vgpr_bases += sum(
                    1 for c in cs if not c.origin.startswith("neighbour") and not S._mn_ring(c.spec)
                )
                for c in cs:
                    if c.origin.startswith(("neighbour:mn_chunk", "neighbour:mn_pad")):
                        self.assertTrue(S._mn_ring(c.spec), (lay, M, N, K, c.origin))
        self.assertGreater(vgpr_bases, 0)

    def test_native_requests_use_native_model(self):
        req = _req(1000, 712, 3000, "CCR")
        dec = S.decide(req, GEMM_FP16_REGISTRY)
        m = S.load_model("gfx950", "native")
        sc, c = dec.ranked[0]
        self.assertAlmostEqual(
            m.score("fp16", 1000, 712, 3000, [S.flatten_spec(c.spec)])[0], sc, places=9
        )
        self.assertEqual(S.model_kind(req), "native")
        self.assertEqual(S.model_kind(_req(1000, 712, 3000, "RCR")), "rcr")

    def test_neighbour_bases_knob(self):
        req = _req(2048, 3000, 5000, "RRR")
        n = {}
        for nb in (0, 2):
            with mock.patch.object(S, "SELECTOR_NATIVE_NEIGHBOUR_BASES", nb):
                S.clear_cache()
                n[nb] = len(S.decide(req, GEMM_FP16_REGISTRY).ranked)
        self.assertLess(n[2], n[0])
        S.clear_cache()


class TestNativeDecision(_Clean):
    def test_switches(self):
        req = _req(1000, 712, 3000, "RRR")
        on = dispatch_gemm_fp16(req)
        with mock.patch.dict(os.environ, {S.NATIVE_SELECTOR_ENV: "0"}):
            S.clear_cache()
            off = dispatch_gemm_fp16(req)
        with mock.patch.dict(os.environ, {S.SELECTOR_ENV: "0"}):
            S.clear_cache()
            off2 = dispatch_gemm_fp16(req)
        S.clear_cache()
        rule = GEMM_FP16_REGISTRY.select(req)
        self.assertEqual(off.spec, rule.select_spec(req))
        self.assertEqual(off2.spec, off.spec)
        self.assertEqual(on.spec.data.layout, "RRR")
        self.assertTrue(any(e.startswith("selector") for e in on.explanation))

    def test_rcr_untouched_by_native_switch(self):
        req = _req(1000, 712, 3000, "RCR")
        a = dispatch_gemm_fp16(req)
        with mock.patch.dict(os.environ, {S.NATIVE_SELECTOR_ENV: "0"}):
            S.clear_cache()
            b = dispatch_gemm_fp16(req)
        S.clear_cache()
        self.assertEqual(a.spec, b.spec)

    def test_selected_candidate_reselects(self):
        for lay in _LAYOUTS:
            for M, N, K in _SHAPES:
                req = _req(M, N, K, lay)
                r = dispatch_gemm_fp16(req)
                if r.candidate.name != "universal_gemm_fp16_cdna_selected":
                    continue
                pinned = dispatch_gemm_fp16(
                    _req(M, N, K, lay, algorithm="universal_gemm", spec_id="cdna_selected")
                )
                self.assertEqual(pinned.spec, r.spec)
                self.assertEqual(r.spec.data.layout, lay)
                return
        self.skipTest("no selector override among the probe shapes")

    def test_pinned_request_skips_selector(self):
        req = _req(1000, 712, 3000, "RRR")
        cand = GEMM_FP16_REGISTRY.select(req)
        pinned = dispatch_gemm_fp16(
            _req(1000, 712, 3000, "RRR", algorithm=cand.algorithm, spec_id=cand.spec_id)
        )
        self.assertEqual(pinned.spec, cand.select_spec(req))

    def test_selected_hidden_when_off(self):
        req = _req(1000, 712, 3000, "RRR")
        sel = GEMM_FP16_REGISTRY.get("universal_gemm_fp16_cdna_selected")
        with mock.patch.dict(os.environ, {S.NATIVE_SELECTOR_ENV: "0"}):
            S.clear_cache()
            ok, why = sel.admits(req)
        S.clear_cache()
        self.assertFalse(ok)
        self.assertIn("selector off", why)

    def test_bf16_native(self):
        r = dispatch_gemm_bf16(_req(256, 8192, 8192, "CCR", "bf16"))
        self.assertEqual(r.spec.data.layout, "CCR")
        self.assertIsNone(launch_shape_error(r.spec, 256, 8192, 8192))


if __name__ == "__main__":
    unittest.main()
