# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The learned RCR GEMM selector (``rocke.dispatch.gemm.selector``).

CPU only. Covers the feature / model contract the C++ mirror depends on, the
generated-spec launch gate, the decision rule (margins, rule-pick classes,
pinned requests, the off switch), the opt-in ``selected`` candidate, and the
autotune entry point with a synthetic measurement.
"""

from __future__ import annotations

import dataclasses
import math
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
from rocke.dispatch.gemm import autotune as A
from rocke.dispatch.gemm import selector as S
from rocke.instances.common.gemm_universal import launch_shape_error

GEMM_SELECTOR_TESTS = True  # conftest: keep the shipped selector default on

_SHAPES = (
    (8192, 8192, 8192),
    (128, 2048, 2048),
    (5376, 256, 18944),
    (1536, 1440, 576),
    (32768, 128, 32768),
    (256, 256, 8192),
    (4096, 4096, 512),
    (64, 8192, 4096),
    (768, 1792, 768),
)


def _reg(dt):
    return GEMM_FP16_REGISTRY if dt == "fp16" else GEMM_BF16_REGISTRY


def _disp(dt):
    return dispatch_gemm_fp16 if dt == "fp16" else dispatch_gemm_bf16


class TestConfigAndFeatures(unittest.TestCase):
    def test_config_string_round_trip(self):
        req = GemmRequest(M=8192, N=8192, K=8192, arch="gfx950", dtype="fp16")
        spec = GEMM_FP16_REGISTRY.select(req).select_spec(req)
        cfg = S.config_string(spec)
        tile, trait = S.parse_config(cfg)
        self.assertEqual(tile["tile_m"], spec.tile.tile_m)
        for k, v in trait.items():
            self.assertEqual(getattr(spec.trait, k), v, k)
        flat = S.flat_from_config(cfg)
        full = S.flatten_spec(spec)
        for k, v in flat.items():
            self.assertEqual(full[k], v, k)

    def test_known_feature_values(self):
        names = [
            "d_tiles_m",
            "d_tiles_n",
            "d_split",
            "d_kslice_tail",
            "d_fill",
            "tr_split_k",
            "tr_pipeline",
        ]
        flat = S.flat_from_config(
            "t16x64x32_w1x4x1_a16x16x32|epilogue=default,pad_m=True,split_k=2"
        )
        x = S.feature_vector(
            names,
            "gfx950",
            "bf16",
            5,
            4736,
            2624,
            flat,
            trait_defaults=S.trait_defaults(),
        )
        self.assertEqual(x[:3], [1.0, 74.0, 2.0])
        self.assertEqual(x[3], 0.0)  # 2624 % (2 * 32) == 0
        self.assertAlmostEqual(x[4], 5 / 16)
        self.assertEqual(x[5:], [2.0, 2.0])  # compv4 -> 2

    def test_k_tail_features(self):
        names = ["d_ktail", "d_kslice_tail", "d_ln_kiters"]
        flat = S.flat_from_config("t128x128x32_w2x2x1_a32x32x16|")
        d = S.trait_defaults()
        whole = S.feature_vector(
            names, "gfx950", "fp16", 1024, 1024, 1024, flat, trait_defaults=d
        )
        tail = S.feature_vector(
            names, "gfx950", "fp16", 1024, 1024, 1040, flat, trait_defaults=d
        )
        self.assertEqual(whole[:2], [0.0, 0.0])
        self.assertEqual(tail[:2], [1.0, 1.0])
        self.assertAlmostEqual(whole[2], math.log(32))
        self.assertAlmostEqual(tail[2], math.log(33))

    def test_feature_names_cover_every_model_feature(self):
        for arch in ("gfx950", "gfx942"):
            m = S.load_model(arch)
            self.assertIsNotNone(m, arch)
            d = S.feature_dict(
                arch,
                "fp16",
                1024,
                1024,
                1024,
                {},
                trait_fields=[n[3:] for n in m.features if n.startswith("tr_")],
                trait_defaults=S.trait_defaults(m.meta),
            )
            missing = [n for n in m.features if n not in d]
            self.assertEqual(missing, [], arch)


class TestModel(unittest.TestCase):
    def test_vectorised_walk_equals_row_walk(self):
        m = S.load_model("gfx950")
        req = GemmRequest(M=4096, N=2048, K=1024, arch="gfx950", dtype="fp16")
        ranked = S.rank(req, GEMM_FP16_REGISTRY, model=m)
        xs = [
            S.feature_vector(
                m.features,
                "gfx950",
                "fp16",
                4096,
                2048,
                1024,
                S.flatten_spec(c.spec),
                trait_defaults=S.trait_defaults(m.meta),
            )
            for _, c in ranked
        ]
        self.assertGreater(len(xs), 8)
        a = m.ensemble.predict(xs)
        b = [m.ensemble.predict_one(x) for x in xs]
        self.assertEqual(a, b)

    def test_no_model_for_other_arches(self):
        self.assertIsNone(S.load_model("gfx1201"))
        req = GemmRequest(M=1024, N=1024, K=1024, arch="gfx1201", dtype="fp16")
        r = dispatch_gemm_fp16(req)
        self.assertNotIn("selected", r.candidate.name)

    def test_tree_text_parser(self):
        txt = (
            "tree\nfeature_names=a b\n\nTree=0\nnum_leaves=2\nsplit_feature=1\nthreshold=0.5\n"
            "decision_type=2\nleft_child=-1\nright_child=-2\nleaf_value=1 2\n\nend of trees\n"
        )
        e = S.TreeEnsemble.from_text(txt)
        self.assertEqual(e.feature_names, ["a", "b"])
        self.assertEqual(e.predict_one([0.0, 0.4]), 1.0)
        self.assertEqual(e.predict_one([0.0, 0.6]), 2.0)
        # a value equal to the threshold goes left (LightGBM: x <= threshold)
        self.assertEqual(e.predict_one([0.0, 0.5]), 1.0)
        rows = [[0.0, 0.5], [0.0, 0.4], [0.0, 0.6], [0.0, 0.5]]
        self.assertEqual(e.predict(rows), [1.0, 1.0, 2.0, 1.0])
        with self.assertRaises(ValueError):
            S.TreeEnsemble.from_text(txt.replace("decision_type=2", "decision_type=1"))

    def test_zero_as_missing_goes_to_the_default_side(self):
        # decision_type 6 = missing type Zero (bits 2-3 = 1) + default left (bit 1)
        nl = chr(10)
        txt = nl.join(
            [
                "tree",
                "feature_names=a",
                "",
                "Tree=0",
                "num_leaves=2",
                "split_feature=0",
                "threshold=-1",
                "decision_type=6",
                "left_child=-1",
                "right_child=-2",
                "leaf_value=1 2",
                "",
                "end of trees",
                "",
            ]
        )
        e = S.TreeEnsemble.from_text(txt)
        rows = [[0.0], [5.0], [-3.0], [0.0], [7.0]]
        self.assertEqual([e.predict_one(r) for r in rows], [1.0, 2.0, 1.0, 1.0, 2.0])
        self.assertEqual(e.predict(rows), [1.0, 2.0, 1.0, 1.0, 2.0])


class TestGeneratedSpecGate(unittest.TestCase):
    def _req(self, M, N, K):
        return GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype="fp16")

    def _cshuffle(self, req):
        c = next(
            c
            for c in GEMM_FP16_REGISTRY.candidates()
            if c.name.endswith("cdna_cshuffle")
        )
        return c.select_spec(req)

    def test_split_k_needs_whole_tile_slices(self):
        req = self._req(256, 256, 8192)
        base = self._cshuffle(req)
        ok = dataclasses.replace(base, trait=dataclasses.replace(base.trait, split_k=4))
        self.assertIsNone(S.generated_spec_error(req, ok, base))
        req2 = self._req(256, 256, 8192 + 32)
        bad = S._fix_pads(ok, 256, 256, 8192 + 32)
        self.assertIsNotNone(S.generated_spec_error(req2, bad, base))
        padk = dataclasses.replace(ok, trait=dataclasses.replace(ok.trait, pad_k=True))
        self.assertIsNotNone(S.generated_spec_error(req, padk, base))

    def test_no_ragged_rows_on_a_generated_global_load_tiling(self):
        req = self._req(1000, 1024, 1024)
        base = self._cshuffle(self._req(1024, 1024, 1024))
        ragged = S._fix_pads(base, 1000, 1024, 1024)
        self.assertTrue(ragged.trait.pad_m)
        self.assertIsNotNone(S.generated_spec_error(req, ragged, ragged))
        self.assertIsNotNone(S.generated_spec_error(req, ragged, None))

    def test_extents_multiple_of_8(self):
        req = self._req(1024, 1024, 1020)
        base = self._cshuffle(self._req(1024, 1024, 1024))
        base = dataclasses.replace(
            base, trait=dataclasses.replace(base.trait, split_k=1)
        )
        padded = S._fix_pads(base, 1024, 1024, 1020)
        self.assertTrue(padded.trait.pad_k)
        why = S.generated_spec_error(req, padded, padded)
        self.assertIsNotNone(why)
        self.assertIn("multiples of 8", why)
        # the same spec on K % 8 == 0 passes this rule
        self.assertIsNone(
            S.generated_spec_error(self._req(1024, 1024, 1016), padded, padded)
        )

    def test_split_k_never_on_dtl_bodies(self):
        req = self._req(8192, 8192, 8192)
        rule = GEMM_FP16_REGISTRY.select(req).select_spec(req)
        self.assertTrue(rule.trait.direct_to_lds)
        sk = dataclasses.replace(rule, trait=dataclasses.replace(rule.trait, split_k=2))
        why = S.generated_spec_error(req, sk, rule)
        self.assertIsNotNone(why)
        self.assertIn("split_k", why)

    def test_stream_k_and_pools_are_never_generated(self):
        req = self._req(4096, 4096, 4096)
        base = self._cshuffle(req)
        sk = dataclasses.replace(
            base, trait=dataclasses.replace(base.trait, persistent_ctas=256)
        )
        self.assertIsNotNone(S.generated_spec_error(req, sk, base))

    def test_every_offered_candidate_passes_the_gates(self):
        for dt in ("fp16", "bf16"):
            for M, N, K in _SHAPES:
                req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt)
                for c in S.selector_candidates(req, _reg(dt)):
                    if S._skinny(c.spec):
                        # gemm_skinny configurations: the body's own gates
                        # (test_selector_skinny_guard covers them in detail)
                        self.assertIsNone(
                            S.skinny_variant_error(req, c.spec), (dt, M, N, K, c.origin)
                        )
                        continue
                    self.assertIsNone(
                        launch_shape_error(c.spec, M, N, K), (dt, M, N, K, c.origin)
                    )
                    self.assertIsNone(S._gate(req, c.spec), (dt, M, N, K, c.origin))
                    if c.origin != "registered":
                        self.assertFalse(c.spec.trait.streamk)
                        if c.spec.trait.split_k > 1:
                            self.assertEqual(
                                K % (c.spec.trait.split_k * c.spec.tile.tile_k), 0
                            )
                            self.assertFalse(c.spec.trait.direct_to_lds)


class TestDecision(unittest.TestCase):
    def setUp(self):
        S.clear_cache()

    def test_off_switch_is_the_rule_layer(self):
        for dt in ("fp16", "bf16"):
            for M, N, K in _SHAPES:
                req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt)
                cand = _reg(dt).select(req)
                with mock.patch.dict(os.environ, {"ROCKE_GEMM_SELECTOR": "0"}):
                    r = _disp(dt)(req)
                self.assertEqual(r.candidate.name, cand.name)
                self.assertEqual(r.spec, cand.select_spec(req))
                self.assertFalse(any(e.startswith("selector") for e in r.explanation))

    def test_unvalidated_arch_ships_off(self):
        m = S.load_model("gfx942")
        self.assertFalse(m.meta.get("default_on", True))
        self.assertFalse(S.selector_enabled("gfx942"))
        self.assertTrue(S.selector_enabled("gfx950"))
        req = GemmRequest(M=128, N=2048, K=2048, arch="gfx942", dtype="fp16")
        rule = GEMM_FP16_REGISTRY.select(req)
        self.assertEqual(dispatch_gemm_fp16(req).candidate.name, rule.name)
        with mock.patch.dict(os.environ, {"ROCKE_GEMM_SELECTOR": "1"}):
            self.assertTrue(S.selector_enabled("gfx942"))
            r = dispatch_gemm_fp16(req)
        self.assertTrue(any(e.startswith("selector") for e in r.explanation))

    def test_pinned_requests_bypass_the_selector(self):
        req = GemmRequest(
            M=128,
            N=2048,
            K=2048,
            arch="gfx950",
            dtype="fp16",
            algorithm="universal_gemm",
            spec_id="cdna_cshuffle_default",
        )
        r = dispatch_gemm_fp16(req)
        self.assertTrue(r.candidate.name.endswith("cdna_cshuffle"))

    def test_beat_anchors_keep_the_rule_pick(self):
        for dt in ("fp16", "bf16"):
            for M, N, K in (
                (8192, 8192, 8192),
                (23040, 24576, 12288),
                (15360, 15360, 768),
            ):
                req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt)
                r = _disp(dt)(req)
                cand = _reg(dt).select(req)
                self.assertEqual(r.spec, cand.select_spec(req), (dt, M, N, K))

    def test_margin_floor_and_origin_margin(self):
        m = S.load_model("gfx950")
        self.assertGreaterEqual(
            m.margin_for("dtl:beat"), S.SELECTOR_MIN_MARGIN["dtl:beat"]
        )
        self.assertGreaterEqual(
            m.margin_for("anything"), S.SELECTOR_MIN_MARGIN["default"]
        )
        self.assertEqual(S.origin_class("neighbour:split_k=4"), "neighbour")
        self.assertEqual(S.origin_class("library"), "library")
        self.assertEqual(S.origin_class("registered"), "registered")

    def test_selected_candidate_reselects_the_pick(self):
        hits = 0
        for dt in ("fp16", "bf16"):
            for M, N, K in _SHAPES:
                req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt)
                r = _disp(dt)(req)
                if not r.candidate.name.endswith("cdna_selected"):
                    continue
                hits += 1
                pinned = dataclasses.replace(
                    req, algorithm="universal_gemm", spec_id="cdna_selected"
                )
                r2 = _disp(dt)(pinned)
                self.assertEqual(r2.spec, r.spec)
                self.assertEqual(r2.grid, r.grid)
                self.assertIsNone(launch_shape_error(r.spec, M, N, K))
        self.assertGreater(hits, 0)

    def test_decision_is_deterministic(self):
        req = GemmRequest(M=5376, N=256, K=18944, arch="gfx950", dtype="bf16")
        a = dispatch_gemm_bf16(req)
        S.clear_cache()
        b = dispatch_gemm_bf16(req)
        self.assertEqual(a.spec, b.spec)
        self.assertEqual(a.kernel_id, b.kernel_id)

    def test_other_families_and_layouts_keep_the_rule_pick(self):
        self.assertFalse(S._universal(object()))
        self.assertFalse(
            S._rcr(GemmRequest(M=64, N=64, K=64, arch="gfx950", layout="RRR"))
        )
        self.assertTrue(S._rcr(GemmRequest(M=64, N=64, K=64, arch="gfx950")))

        class _Reg:
            family = "fake"

            def select(self, req):
                class _C:
                    name = "fake"

                    @staticmethod
                    def select_spec(r):
                        return object()

                return _C()

            def combos(self, req, include_opt_in=False):
                return ()

            def candidates(self):
                return ()

        req = GemmRequest(M=1024, N=1024, K=1024, arch="gfx950", dtype="fp16")
        self.assertIsNone(S.decide(req, _Reg()))

    def test_diverse_top(self):
        c = S.SelectorCandidate("library", "", None)
        ranked = [(1.0, c), (1.0, c), (1.5, c), (2.0, c)]
        self.assertEqual([s for s, _ in S.diverse_top(ranked, 3)], [1.0, 1.5, 2.0])


class TestAutotune(unittest.TestCase):
    def test_lite_and_search_pick_the_measured_best(self):
        req = GemmRequest(M=5376, N=256, K=18944, arch="gfx950", dtype="bf16")
        ranked = A.gemm_candidates(req, pool=24)
        truth = {S._spec_key(c.spec): 1.0 + 0.1 * i for i, (_, c) in enumerate(ranked)}
        best_key = S._spec_key(ranked[2][1].spec)
        truth[best_key] = 0.5  # the measured best is not the predicted best

        def measure(spec, _fid):
            return [truth[S._spec_key(spec)]] * 3

        r = A.tune_gemm(
            req,
            measure=measure,
            check=lambda s: True,
            mode="search",
            budget=40,
            pool=24,
        )
        self.assertEqual(S._spec_key(r.spec), best_key)
        d = r.dispatch_result(req)
        self.assertEqual(d.spec, r.spec)
        lite = A.tune_gemm(
            req, measure=measure, check=lambda s: True, mode="lite", lite_k=3, pool=24
        )
        self.assertIn(
            S._spec_key(lite.spec),
            {S._spec_key(c.spec) for _, c in ranked[:4]} | {best_key},
        )

    def test_lite_always_retimes_the_rule_pick(self):
        # (the ring_small_any rule pick of 5376x256x18944 ranks
        # in the top 3; this shape keeps the rule pick outside it)
        req = GemmRequest(M=1024, N=1024, K=8192, arch="gfx950", dtype="bf16")
        ranked = A.gemm_candidates(req, pool=24)
        rule = [c for _, c in ranked if c.is_rule_pick]
        self.assertEqual(len(rule), 1)
        top = {S._spec_key(c.spec) for _, c in S.diverse_top(ranked, 3)}
        self.assertNotIn(S._spec_key(rule[0].spec), top)
        rk = S._spec_key(rule[0].spec)

        def measure(spec, _fid):
            return [0.1 if S._spec_key(spec) == rk else 1.0]

        r = A.tune_gemm(
            req, measure=measure, check=lambda s: True, mode="lite", lite_k=3, pool=24
        )
        self.assertEqual(S._spec_key(r.spec), rk)

    def test_failing_check_never_wins(self):
        req = GemmRequest(M=5376, N=256, K=18944, arch="gfx950", dtype="bf16")
        ranked = A.gemm_candidates(req, pool=8)
        bad = S._spec_key(ranked[0][1].spec)

        def measure(spec, _fid):
            return [0.1 if S._spec_key(spec) == bad else 1.0]

        r = A.tune_gemm(
            req,
            measure=measure,
            check=lambda s: S._spec_key(s) != bad,
            mode="lite",
            pool=8,
        )
        self.assertNotEqual(S._spec_key(r.spec), bad)
        self.assertTrue(math.isfinite(1.0))


if __name__ == "__main__":
    unittest.main()
