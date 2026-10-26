# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The learned GEMM selector's gemm_skinny family and portfolio guard.

CPU only. A synthetic r2c-v2 model (hand-written LightGBM text) prefers
gemm_skinny configurations so the routing can be checked without the shipped
model: skinny config strings and features, the generated-variant gate
(``skinny_variant_error``), the opt-in ``cdna_selected_skinny`` candidate that
serves a generated skinny pick, the bootstrap-bag portfolio guard, and that a
model of the older feature version never ranks gemm_skinny configurations.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
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
from rocke.instances.common.gemm_skinny import (
    SkinnyGemmSpec,
    is_valid_spec as skinny_valid,
    skinny_kps,
    skinny_launch_error,
    skinny_steps,
)

GEMM_SELECTOR_TESTS = True  # conftest: keep the selector default on

_NL = chr(10)


def _tree(feature: int, threshold: float, left: float, right: float) -> str:
    return _NL.join(
        [
            "num_leaves=2",
            f"split_feature={feature}",
            f"threshold={threshold}",
            "decision_type=2",
            "left_child=-1",
            "right_child=-2",
            f"leaf_value={left} {right}",
            "",
        ]
    )


def _model_text(features, trees) -> str:
    out = ["tree", "feature_names=" + " ".join(features), ""]
    for i, t in enumerate(trees):
        out += [f"Tree={i}", t]
    out += ["end of trees", ""]
    return _NL.join(out)


# s_fam (0 universal, 1 skinny), s_split (-1 universal), p_ln_m
_FEATURES = ["s_fam", "s_split", "p_ln_m"]


def _write_model(d, arch, trees, *, version="r2c-v2", skinny=True, bags=(), min_gain=0.0, guard_enabled=None):
    meta = {
        "feature_version": version,
        "arch": arch,
        "features": _FEATURES,
        "trait_defaults": {},
        "margins": {"default": 0.0},
        "default_on": True,
        "library": [],
        "neighbour_levers": [],
        "families": {"skinny": skinny},
        "guard": {},
    }
    names = []
    for i, b in enumerate(bags):
        nm = f"gemm_rcr_{arch}.bag{i}.model.txt"
        with open(os.path.join(d, nm), "w", newline=_NL) as f:
            f.write(_model_text(_FEATURES, b))
        names.append(nm)
    if names:
        meta["guard"] = {"bags": names, "min_gain": min_gain}
        if guard_enabled is not None:
            meta["guard"]["enabled"] = guard_enabled
    with open(os.path.join(d, f"gemm_rcr_{arch}.model.txt"), "w", newline=_NL) as f:
        f.write(_model_text(_FEATURES, trees))
    with open(os.path.join(d, f"gemm_rcr_{arch}.meta.json"), "w", newline=_NL) as f:
        json.dump(meta, f)


# skinny configurations predicted 1.0 faster; among them split 2 the fastest
# (split <= 2.5: -0.3, split <= 1.5: +0.5; universal rows have s_split = -1)
_PREFER_SKINNY = [
    _tree(0, 0.5, 0.0, -1.0),
    _tree(1, 2.5, -0.3, 0.0),
    _tree(1, 1.5, 0.5, 0.0),
]
# a bag that predicts every skinny configuration slower than any universal one
_BAG_AGAINST = [_tree(0, 0.5, 0.0, 1.0)]
_BAG_FOR = [_tree(0, 0.5, 0.0, -1.0)]

# RCR shapes whose rule pick is a UniversalGemm body and where gemm_skinny
# admits through its fallback registration
_SHAPES = ((448, 128, 14912), (512, 64, 8000), (256, 2048, 4096), (1024, 1024, 2048))


class _ModelDir(unittest.TestCase):
    """Run each test against a temporary model dir (ROCKE_GEMM_SELECTOR_MODEL_DIR)."""

    trees = _PREFER_SKINNY
    bags = ()
    version = "r2c-v2"
    skinny = True
    min_gain = 0.0
    guard_enabled = None

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="rocke_sel_")
        for arch in ("gfx950", "gfx942"):
            _write_model(
                self.d,
                arch,
                self.trees,
                version=self.version,
                skinny=self.skinny,
                bags=self.bags,
                min_gain=self.min_gain,
                guard_enabled=self.guard_enabled,
            )
        self.env = mock.patch.dict(
            os.environ,
            {S.MODEL_DIR_ENV: self.d, S.SELECTOR_ENV: "1"},
        )
        self.env.start()
        S.clear_cache()

    def tearDown(self):
        self.env.stop()
        S.clear_cache()
        shutil.rmtree(self.d, ignore_errors=True)


class TestSkinnyConfigAndFeatures(unittest.TestCase):
    def test_config_round_trip(self):
        spec = SkinnyGemmSpec(
            dtype="fp16",
            wave_m_atoms=2,
            wave_n_atoms=4,
            warp_k=8,
            kc=8,
            prefetch=1,
            red_sync="sc1",
            red_unroll=8,
            split=18,
        )
        cfg = S.config_string(spec)
        self.assertEqual(
            cfg,
            "sk|kc=8,prefetch=1,red_sync=sc1,red_unroll=8,split=18,warp_k=8,wave_m_atoms=2",
        )
        self.assertTrue(S.is_skinny_config(cfg))
        back = SkinnyGemmSpec(dtype="fp16", **S.parse_skinny_config(cfg))
        self.assertEqual(back, spec)
        flat = S.flat_from_config(cfg)
        self.assertEqual(flat, S.flatten_spec(spec))
        self.assertEqual(flat["family"], "skinny")
        # the split is always written, even at its default
        self.assertTrue(S.config_string(SkinnyGemmSpec()).endswith("split=1"))
        with self.assertRaises(ValueError):
            S.parse_skinny_config("sk|nope=1")
        with self.assertRaises(ValueError):
            S.parse_skinny_config("t128x128x32_w2x2x1_a32x32x16|")

    def test_skinny_feature_values(self):
        spec = SkinnyGemmSpec(
            dtype="fp16", wave_m_atoms=2, wave_n_atoms=4, warp_m=1, warp_n=2, warp_k=4,
            kc=16, prefetch=2, red_sync="sc1", split=9,
        )
        names = [
            "s_fam", "s_ma", "s_na", "s_wk", "s_split", "s_rsc1", "s_regs",
            "d_tm", "d_tn", "d_tk", "d_split", "d_tiles_m", "d_tiles_n", "tr_split_k",
        ]
        x = S.feature_vector(
            names, "gfx950", "fp16", 448, 128, 14912, S.flatten_spec(spec),
            trait_defaults=S.trait_defaults(),
        )
        acc, frag = 2 * 4 * 4, (2 + 4) * 8
        self.assertEqual(
            x,
            [1.0, 2.0, 4.0, 4.0, 9.0, 1.0, float(acc + 3 * frag),
             32.0, 128.0, 256.0, 9.0, 14.0, 1.0, 9.0],
        )

    def test_universal_rows_have_no_family_features(self):
        flat = S.flat_from_config("t128x128x32_w2x2x1_a32x32x16|")
        x = S.feature_vector(
            list(S.SKINNY_FEATURES), "gfx950", "fp16", 1024, 1024, 1024, flat,
            trait_defaults=S.trait_defaults(),
        )
        self.assertEqual(x[0], 0.0)
        self.assertEqual(set(x[1:]), {-1.0})

    def test_spec_class_and_origin(self):
        self.assertEqual(S.spec_class(S.flatten_spec(SkinnyGemmSpec())), "skinny")
        self.assertEqual(S.origin_class("skinny:split=4"), "skinny")


class TestSkinnyVariantGate(unittest.TestCase):
    def _req(self, M, N, K):
        return GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype="fp16")

    def test_empty_k_slice_is_refused(self):
        spec = SkinnyGemmSpec(dtype="fp16", kc=16, split=1)   # 64 K per step
        req = self._req(64, 64, 64 * 5)                         # 5 steps
        self.assertIsNone(S.skinny_variant_error(req, spec))
        self.assertIsNone(S.skinny_variant_error(req, dataclasses.replace(spec, split=5)))
        # 8 slices of ceil(5 / 8) = 1 step: slices 5..7 empty
        self.assertIn("empty", S.skinny_variant_error(req, dataclasses.replace(spec, split=8)))
        # 4 slices of 2 steps: the last one starts at step 6 > 5
        self.assertIn("empty", S.skinny_variant_error(req, dataclasses.replace(spec, split=4)))

    def test_body_contract_is_enforced(self):
        # K not a multiple of the configuration's K alignment (launch contract)
        spec = SkinnyGemmSpec(dtype="fp16", kalign=8)
        self.assertIn("kalign", S.skinny_variant_error(self._req(64, 64, 1001), spec))
        self.assertIsNone(S.skinny_variant_error(self._req(64, 64, 1000), spec))
        # static validity (register budget) of the body
        big = SkinnyGemmSpec(dtype="fp16", wave_m_atoms=8, wave_n_atoms=8, kc=32, prefetch=4)
        self.assertIn("register", S.skinny_variant_error(self._req(256, 256, 4096), big))

    def test_every_generated_variant_passes_the_body_gates(self):
        for M, N, K in _SHAPES + ((5, 4736, 2624), (130, 9000, 30000), (17, 33, 777)):
            req = self._req(M, N, K)
            base = SkinnyGemmSpec(dtype="fp16", kalign=8 if K % 8 == 0 else (2 if K % 2 == 0 else 1))
            for lab, v in S._skinny_variants(base):
                err = S.skinny_variant_error(req, v)
                if err is not None:
                    continue
                self.assertTrue(skinny_valid(v, "gfx950")[0], lab)
                self.assertIsNone(skinny_launch_error(v, M, N, K, v.split), lab)
                steps = skinny_steps(v, K)
                self.assertLess((v.split - 1) * skinny_kps(v, K, v.split), steps, lab)


class TestDtlOnlyLevers(unittest.TestCase):
    def test_dtl_stages_never_mutates_a_global_load_body(self):
        req = GemmRequest(M=4096, N=4096, K=4096, arch="gfx950", dtype="fp16")
        levers = [("dtl_stages", (2, 3))]
        cs = next(c for c in GEMM_FP16_REGISTRY.candidates() if c.name.endswith("cdna_cshuffle"))
        base = cs.select_spec(req)
        self.assertFalse(base.trait.direct_to_lds)
        self.assertEqual(list(S._mutations(base, levers)), [])
        ring = GemmRequest(M=2048, N=2048, K=2048, arch="gfx950", dtype="fp16")
        dtl = next(
            sp for c, sp in GEMM_FP16_REGISTRY.combos(ring, include_opt_in=False)
            if hasattr(sp, "trait") and sp.trait.direct_to_lds
        )
        muts = list(S._mutations(dtl, levers))
        self.assertTrue(muts)
        self.assertTrue(all(m.trait.direct_to_lds for _, m in muts))
        self.assertIn("dtl_stages", S._DTL_ONLY_LEVERS)


class TestSkinnyRouting(_ModelDir):
    def test_model_knows_skinny(self):
        m = S.load_model("gfx950")
        self.assertTrue(m.knows_skinny)
        self.assertEqual(m.meta["feature_version"], S.FEATURE_VERSION)

    def test_generated_skinny_pick_is_served_and_reselected(self):
        hits = 0
        for disp, reg, dt in ((dispatch_gemm_fp16, GEMM_FP16_REGISTRY, "fp16"),
                              (dispatch_gemm_bf16, GEMM_BF16_REGISTRY, "bf16")):
            for M, N, K in _SHAPES:
                req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt)
                r = disp(req)
                self.assertIsInstance(r.spec, SkinnyGemmSpec, (dt, M, N, K))
                self.assertEqual(r.spec.split, 2)
                self.assertIsNone(S.skinny_variant_error(req, r.spec))
                self.assertTrue(any(e.startswith("selector") for e in r.explanation))
                if r.candidate.spec_id == "cdna_selected_skinny":
                    hits += 1
                    pinned = dataclasses.replace(
                        req, algorithm="gemm_skinny", spec_id="cdna_selected_skinny"
                    )
                    r2 = disp(pinned)
                    self.assertEqual(r2.spec, r.spec)
                    self.assertEqual(r2.grid, r.grid)
                else:
                    # a registered skinny candidate's own configuration
                    self.assertEqual(r.spec, r.candidate.select_spec(req))
                # the universal selected candidate never serves a skinny pick
                ok, _ = S.selected_supports(req, reg, family="universal")
                self.assertFalse(ok)
                for s in S.selected_sweep(req, reg, family="skinny"):
                    self.assertIsInstance(s, SkinnyGemmSpec)
        self.assertGreater(hits, 0)

    def test_off_switch_and_knob(self):
        for M, N, K in _SHAPES:
            req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype="fp16")
            rule = GEMM_FP16_REGISTRY.select(req)
            with mock.patch.dict(os.environ, {S.SELECTOR_ENV: "0"}):
                r = dispatch_gemm_fp16(req)
            self.assertEqual(r.candidate.name, rule.name)
            self.assertEqual(r.spec, rule.select_spec(req))
            if isinstance(rule.select_spec(req), SkinnyGemmSpec):
                continue
            # SELECTOR_SKINNY off: no skinny configuration is ranked, so the
            # synthetic model (which only rewards skinny) keeps the rule pick
            with mock.patch.object(S, "SELECTOR_SKINNY", False):
                S.clear_cache()
                r = dispatch_gemm_fp16(req)
                self.assertNotIsInstance(r.spec, SkinnyGemmSpec)
                self.assertEqual(r.spec, rule.select_spec(req))
            S.clear_cache()

    def test_autotune_serving_candidate(self):
        req = GemmRequest(M=448, N=128, K=14912, arch="gfx950", dtype="fp16")
        ranked = A.gemm_candidates(req, pool=8)
        self.assertTrue(any(isinstance(c.spec, SkinnyGemmSpec) for _, c in ranked))
        for _, c in ranked:
            k = A._serving_candidate(GEMM_FP16_REGISTRY, c, req)
            if isinstance(c.spec, SkinnyGemmSpec) and c.origin != "registered":
                self.assertEqual(k.spec_id, "cdna_selected_skinny")


class TestOldVersionNeverRanksSkinny(_ModelDir):
    version = "g4-v1"

    def test_v1_model_has_no_skinny_candidates(self):
        m = S.load_model("gfx950")
        self.assertIsNotNone(m)
        self.assertFalse(m.knows_skinny)
        for M, N, K in _SHAPES:
            req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype="fp16")
            for _, c in S.rank(req, GEMM_FP16_REGISTRY, model=m):
                self.assertNotIsInstance(c.spec, SkinnyGemmSpec)


class TestFamilyFlagOff(_ModelDir):
    skinny = False

    def test_model_without_skinny_training_data(self):
        self.assertFalse(S.load_model("gfx950").knows_skinny)


class TestPortfolioGuardKeepsRule(_ModelDir):
    bags = (_BAG_FOR, _BAG_AGAINST)

    def test_one_dissenting_bag_keeps_the_rule_pick(self):
        for M, N, K in _SHAPES:
            req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype="fp16")
            rule = GEMM_FP16_REGISTRY.select(req)
            rspec = rule.select_spec(req)
            if isinstance(rspec, SkinnyGemmSpec):
                continue
            dec = S.decide(req, GEMM_FP16_REGISTRY)
            self.assertTrue(dec.pick.is_rule_pick, (M, N, K))
            self.assertIn("portfolio guard", dec.reason)
            r = dispatch_gemm_fp16(req)
            self.assertEqual(r.spec, rspec)
            with mock.patch.object(S, "SELECTOR_PORTFOLIO_GUARD", False):
                S.clear_cache()
                self.assertFalse(S.decide(req, GEMM_FP16_REGISTRY).pick.is_rule_pick)
            S.clear_cache()


class TestPortfolioGuardAgrees(_ModelDir):
    bags = (_BAG_FOR, _BAG_FOR)

    def test_agreeing_bags_allow_the_switch(self):
        req = GemmRequest(M=448, N=128, K=14912, arch="gfx950", dtype="fp16")
        dec = S.decide(req, GEMM_FP16_REGISTRY)
        self.assertFalse(dec.pick.is_rule_pick)
        self.assertNotIn("portfolio guard", dec.reason)


class TestPortfolioGuardDisabledBySidecar(_ModelDir):
    bags = (_BAG_AGAINST,)
    guard_enabled = False

    def test_disabled_guard_never_reverts_a_switch(self):
        m = S.load_model("gfx950")
        self.assertEqual(len(m.bags), 1)
        self.assertFalse(m.guard_enabled)
        req = GemmRequest(M=448, N=128, K=14912, arch="gfx950", dtype="fp16")
        dec = S.decide(req, GEMM_FP16_REGISTRY)
        self.assertFalse(dec.pick.is_rule_pick)
        self.assertNotIn("portfolio guard", dec.reason)


class TestPortfolioGuardMinGain(_ModelDir):
    bags = (_BAG_FOR,)
    min_gain = 1.0   # the bag's gain is exactly 1.0: not above the threshold

    def test_gain_must_exceed_the_threshold(self):
        req = GemmRequest(M=448, N=128, K=14912, arch="gfx950", dtype="fp16")
        dec = S.decide(req, GEMM_FP16_REGISTRY)
        if not isinstance(GEMM_FP16_REGISTRY.select(req).select_spec(req), SkinnyGemmSpec):
            self.assertTrue(dec.pick.is_rule_pick)


class TestMissingBagFile(unittest.TestCase):
    def test_sidecar_naming_a_missing_bag_raises(self):
        d = tempfile.mkdtemp(prefix="rocke_sel_")
        try:
            _write_model(d, "gfx950", _PREFER_SKINNY, bags=(_BAG_FOR,))
            os.remove(os.path.join(d, "gemm_rcr_gfx950.bag0.model.txt"))
            S.clear_cache()
            with self.assertRaises(ValueError):
                S._load_model_cached("gfx950", d)
        finally:
            S.clear_cache()
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
