# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Config surface of the fp8 sweep harness: validation, derivation, selection.

Every assertion here guards a failure that is **silent** in the shape this
replaced, which is why they are worth having:

* a trait the enumerator did not loop over could be written in the config and
  was then ignored -- a sweep could ask for four TDM depths and confidently
  rank four copies of one kernel;
* the stage-1 cut kept a fixed top-N chosen under one pinned trait config,
  while the stage-2 gain varies enough across geometries to reorder them;
* a wrong-but-fast kernel inside that top-N shrank the finalist pool instead
  of being replaced, because rank N+1 was never verified.

None of those raised, logged, or showed up in the results. They changed what
the search looked at.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM / "python"))
sys.path.insert(0, str(PLATFORM / "python/rocke/examples/gfx1250/gemm_perf_fp8"))

CONFIGS = PLATFORM / "python/rocke/examples/gfx1250/gemm_perf_fp8/configs"


def _sweep():
    import fp8_gemm_sweep

    return fp8_gemm_sweep


def _base_traits():
    """A minimal valid ``traits`` block: one pinned key, one swept."""
    return {
        "scheduler": "intrawave",
        "epilogue": ["default", "cshuffle"],
        "screening": {"epilogue": "default"},
    }


def _cfg(traits=None, selection=None):
    # load_config, not json.loads: the shipped shapes carry only what differs
    # from _base.json, so a raw read has no target/traits/selection at all.
    cfg = _sweep().load_config(CONFIGS / "sq4k.json")
    if traits is not None:
        cfg["traits"] = traits
    if selection:
        cfg["selection"].update(selection)
    return cfg


class TestTraitValidation(unittest.TestCase):
    """Unusable configs must fail loudly rather than be ignored."""

    def _rejects(self, traits, needle):
        m = _sweep()
        with self.assertRaises(SystemExit) as cm:
            m.resolve_traits(_cfg(traits))
        self.assertIn(needle, str(cm.exception))

    def test_unknown_trait_key_is_rejected(self):
        t = _base_traits()
        t["lds_swizzel"] = False  # typo
        self._rejects(t, "not a TraitSpec field")

    def test_shape_derived_traits_are_rejected(self):
        for key in ("pad_m", "pad_n", "pad_k"):
            with self.subTest(key=key):
                t = _base_traits()
                t[key] = True
                self._rejects(t, "derived from the problem shape")

    def test_swept_key_without_a_screening_value_is_rejected(self):
        t = _base_traits()
        t["lds_k_pad"] = [0, 48]  # swept, but absent from screening
        self._rejects(t, "needs an explicit stage-1 value")

    def test_screening_key_naming_an_undeclared_trait_is_rejected(self):
        t = _base_traits()
        t["screening"]["lds_k_pad"] = 48  # not declared in traits at all
        self._rejects(t, "not declared")

    def test_screening_entry_for_a_pinned_key_is_ignored_not_rejected(self):
        """An explicit scalar IS the pin; a leftover screening entry is noise.

        This is what command-line narrowing produces -- ``--epilogues
        cshuffle`` rewrites the list to one element while the inherited
        ``screening.epilogue`` still names the value that was dropped.
        Erroring there would make the obvious invocation fail.
        """
        m = _sweep()
        t = _base_traits()
        t["epilogue"] = "cshuffle"          # narrowed to a pin
        t["screening"]["epilogue"] = "default"   # stale, names a dropped value
        pin, swept = m.resolve_traits(_cfg(t))
        self.assertEqual(pin["epilogue"], "cshuffle")
        self.assertNotIn("epilogue", swept)

    def test_single_element_sweep_becomes_a_pin(self):
        m = _sweep()
        t = _base_traits()
        t["epilogue"] = ["cshuffle"]
        pin, swept = m.resolve_traits(_cfg(t))
        self.assertEqual(pin["epilogue"], "cshuffle")
        self.assertEqual(swept, {})

    def test_screening_value_outside_the_sweep_is_rejected(self):
        """Else stage 1 ranks geometry under a kernel stage 2 never builds."""
        t = _base_traits()
        t["screening"]["epilogue"] = "nonesuch"
        self._rejects(t, "not among the swept values")

    def test_empty_sweep_list_is_rejected(self):
        t = _base_traits()
        t["lds_k_pad"] = []
        self._rejects(t, "empty list")

    def test_scalar_pins_and_list_sweeps(self):
        m = _sweep()
        pin, swept = m.resolve_traits(_cfg(_base_traits()))
        self.assertEqual(pin["scheduler"], "intrawave")
        self.assertEqual(pin["epilogue"], "default")  # from screening
        self.assertEqual(swept, {"epilogue": ["default", "cshuffle"]})


class TestGenericEnumeration(unittest.TestCase):
    def test_any_traitspec_field_can_be_swept(self):
        """``tdm_depth`` is the case that motivated this: it was ignored.

        The old enumerator was a fixed nest of six loops, so a config asking
        for four depths produced four specs all carrying the screening pin's
        depth.
        """
        m = _sweep()
        cfg = _cfg()
        cfg["traits"] = {
            "scheduler": "intrawave", "epilogue": "default", "pipeline": "mem",
            "lds_swizzle": False, "waves_per_eu": None, "lds_k_pad": 48,
            "tdm": True,
            "tdm_depth": [1, 2, 3, 4],
            "screening": {"tdm_depth": 2},
        }
        tiles = m.enumerate_tile_configs(cfg)
        specs = m.enumerate_trait_configs(cfg, tiles[:3])
        depths = sorted({s.trait.tdm_depth for s in specs})
        self.assertEqual(depths, [1, 2, 3, 4], f"got {depths}; the sweep was ignored")

    def test_dtl_prefetch_without_direct_to_lds_is_not_generated(self):
        """A pairing ``is_valid_spec`` rejects should never enter the grid."""
        m = _sweep()
        combos = m._trait_combos(
            {"direct_to_lds": [False, True], "dtl_prefetch": [False, True]}
        )
        self.assertTrue(combos)
        for c in combos:
            self.assertFalse(c["dtl_prefetch"] and not c["direct_to_lds"], c)

    def test_validity_changing_sweep_is_warned_about(self):
        """Depth changes the admissible tile pool, so arms are not comparable."""
        m = _sweep()
        cfg = _cfg()
        cfg["traits"] = {
            "scheduler": "intrawave", "epilogue": "default", "pipeline": "mem",
            "lds_swizzle": False, "waves_per_eu": None, "lds_k_pad": 48,
            "tdm": True, "tdm_depth": [2, 4], "screening": {"tdm_depth": 2},
        }
        warnings = m.warn_if_sweep_changes_validity(cfg, [])
        self.assertTrue(warnings, "expected a warning for a validity-changing sweep")
        self.assertIn("tdm_depth", warnings[0])
        self.assertIn("admissible tile pool", warnings[0])

    def test_no_warning_when_the_pool_is_unchanged(self):
        m = _sweep()
        self.assertEqual(m.warn_if_sweep_changes_validity(_cfg(_base_traits()), []), [])


class TestInertSweepDetection(unittest.TestCase):
    """A trait whose values compile to the same kernel must be called out.

    Three such traits were found by hand on this target, each only after a
    result looked wrong. The sweep built, timed and ranked identical kernels
    and reported a winner among them; nothing in the config said so.
    """

    def _big_finalists(self):
        """Champion-class geometry -- big enough that LDS pins occupancy."""
        from dataclasses import replace

        m = _sweep()
        cfg = m.load_config(CONFIGS / "sq4k.json")
        tiles = m.enumerate_tile_configs(cfg)
        big = [s for s in tiles if s.tile.tile_m == 256 and s.tile.tile_n == 256]
        self.assertTrue(big, "expected some 256x256 geometries")
        return cfg, big[:4]

    def test_waves_per_eu_is_flagged_on_lds_pinned_tiles(self):
        """Backend-level inertness: the IR differs, the machine code does not."""
        m = _sweep()
        cfg, fin = self._big_finalists()
        warns = [w for w in m.warn_if_sweep_is_inert(cfg, fin) if "waves_per_eu" in w]
        self.assertTrue(warns, "waves_per_eu is a no-op where LDS pins occupancy")
        self.assertIn("SAME kernel", warns[0])

    def test_ab_load_elem_bytes_is_flagged_under_tdm(self):
        """Emitter-level inertness: TDM replaces the per-lane global load."""
        m = _sweep()
        cfg, fin = self._big_finalists()
        m.apply_set_overrides(cfg, [
            "traits.ab_load_elem_bytes=[2,null]",
            "traits.screening.ab_load_elem_bytes=null",
        ])
        warns = [w for w in m.warn_if_sweep_is_inert(cfg, fin)
                 if "ab_load_elem_bytes" in w]
        self.assertTrue(warns)

    def test_a_degenerate_sweep_is_reported_separately(self):
        """One valid value is not a comparison either, but it is not inertness."""
        m = _sweep()
        cfg, fin = self._big_finalists()
        m.apply_set_overrides(cfg, [
            "traits.lds_swizzle=[false,true]",
            "traits.screening.lds_swizzle=false",
        ])
        warns = [w for w in m.warn_if_sweep_is_inert(cfg, fin) if "lds_swizzle" in w]
        self.assertTrue(warns)
        self.assertIn("degenerate", warns[0])

    def test_a_live_trait_is_not_flagged(self):
        """Guards against a detector that simply warns about everything."""
        m = _sweep()
        cfg, fin = self._big_finalists()
        warns = [w for w in m.warn_if_sweep_is_inert(cfg, fin) if "lds_k_pad" in w]
        self.assertEqual(warns, [], "lds_k_pad changes the kernel; it is live")

    def test_diagnostics_never_break_the_run(self):
        m = _sweep()
        self.assertEqual(m.warn_if_sweep_is_inert(_cfg(_base_traits()), []), [])


class TestConfigDerivation(unittest.TestCase):
    def test_extends_deep_merges(self):
        """Both shipped shapes inherit from _base.json.

        27 of their 34 keys were identical before this; the shapes now carry
        only their problem and tile_config. Even ``benchmark.workers``, which
        used to differ 16 vs 64, is shared -- build time measured flat from 16
        to 128 workers, so the split was never meaningful.
        """
        m = _sweep()
        for name, k in (("sq4k.json", 8192), ("skinny.json", 4096)):
            with self.subTest(config=name):
                cfg = m.load_config(CONFIGS / name)
                self.assertEqual(cfg["problem"]["k"], k)        # own
                self.assertEqual(cfg["benchmark"]["workers"], 64)   # inherited
                self.assertTrue(cfg["traits"]["tdm"])           # inherited
                self.assertEqual(cfg["traits"]["tdm_depth"], 2)
                self.assertIn("pipeline", cfg["traits"])
                self.assertIn("arch", cfg["target"])
                self.assertNotIn("extends", cfg)

    def test_the_base_is_what_ships(self):
        """TDM depth 2 is pinned in the base, not left to a variant file.

        The pre-TDM default measured 1.43x worse on the square shape, so a
        base config without it describes a kernel nobody would run.
        """
        m = _sweep()
        pin, _ = m.resolve_traits(m.load_config(CONFIGS / "sq4k.json"))
        self.assertTrue(pin["tdm"])
        self.assertEqual(pin["tdm_depth"], 2)

    def test_extends_cycle_is_detected(self):
        import tempfile

        m = _sweep()
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.json", Path(d) / "b.json"
            a.write_text(json.dumps({"extends": "b.json"}))
            b.write_text(json.dumps({"extends": "a.json"}))
            with self.assertRaises(SystemExit) as cm:
                m.load_config(a)
            self.assertIn("cycle", str(cm.exception))

    def test_set_override_parses_json_and_bare_strings(self):
        m = _sweep()
        cfg = {"traits": {"tdm": False}, "selection": {}}
        applied = m.apply_set_overrides(cfg, [
            "traits.tdm=true",
            "traits.tdm_depth=3",
            "traits.lds_k_pad=[0,48]",
            "traits.scheduler=intrawave",
            "traits.waves_per_eu=null",
        ])
        self.assertEqual(cfg["traits"]["tdm"], True)
        self.assertEqual(cfg["traits"]["tdm_depth"], 3)
        self.assertEqual(cfg["traits"]["lds_k_pad"], [0, 48])
        self.assertEqual(cfg["traits"]["scheduler"], "intrawave")
        self.assertIsNone(cfg["traits"]["waves_per_eu"])
        self.assertEqual(len(applied), 5)

    def test_set_creates_missing_sections_and_rejects_malformed(self):
        m = _sweep()
        cfg = {}
        m.apply_set_overrides(cfg, ["a.b.c=1"])
        self.assertEqual(cfg["a"]["b"]["c"], 1)
        for bad in ("noequals", "=5", ".=5"):
            with self.assertRaises(SystemExit):
                m.apply_set_overrides({}, [bad])


class TestFinalistSelection(unittest.TestCase):
    """The cut sat at a 0.9% margin while stage 2 reorders by up to 12pp."""

    @staticmethod
    def _ranked(times):
        return [{"best_ms": t, "spec": {}, "verified": True} for t in times]

    def test_crowded_boundary_keeps_more_than_n(self):
        m = _sweep()
        ranked = self._ranked([1.00 + 0.001 * i for i in range(40)])  # 0.1% apart
        kept = m.select_finalists(ranked, {"tile_finalists": 20})
        self.assertGreater(len(kept), 20)

    def test_separated_boundary_keeps_exactly_n(self):
        """Separation is *relative*: each step must exceed the margin.

        An arithmetic ladder is not separated -- at rank 20 of ``1.0 + i`` the
        next two entries are still inside a 12% margin. Geometric spacing is
        what makes neighbours more than one margin apart.
        """
        m = _sweep()
        ranked = self._ranked([1.2 ** i for i in range(40)])  # 20% apart each
        kept = m.select_finalists(ranked, {"tile_finalists": 20})
        self.assertEqual(len(kept), 20)

    def test_zero_margin_reproduces_the_fixed_cut(self):
        m = _sweep()
        ranked = self._ranked([1.00 + 0.001 * i for i in range(40)])
        kept = m.select_finalists(
            ranked, {"tile_finalists": 20, "finalist_margin_pct": 0}
        )
        self.assertEqual(len(kept), 20)

    def test_margin_is_measured_from_rank_n_not_from_the_best(self):
        """Anchoring on rank 1 would keep everything on a flat distribution."""
        m = _sweep()
        ranked = self._ranked([1.0] * 20 + [1.05, 1.5, 2.0])
        kept = m.select_finalists(
            ranked, {"tile_finalists": 20, "finalist_margin_pct": 10}
        )
        self.assertEqual(len(kept), 21)  # 1.05 is within 10% of rank-20's 1.0

    def test_budget_allows_backfill_but_is_bounded(self):
        """The verification walk must be able to replace rejects, not run away."""
        m = _sweep()
        self.assertEqual(m._finalist_budget({"tile_finalists": 20}), 40)
        self.assertEqual(
            m._finalist_budget({"tile_finalists": 20, "finalist_margin_pct": 0}), 20
        )


class TestShippedConfigs(unittest.TestCase):
    def test_every_shipped_config_validates(self):
        m = _sweep()
        found = sorted(CONFIGS.glob("*.json"))
        self.assertTrue(found)
        for path in found:
            with self.subTest(config=path.name):
                cfg = m.load_config(path)
                pin, swept = m.resolve_traits(cfg)
                self.assertTrue(pin)
                # the pin must itself be a buildable trait dict
                from rocke.instances.common.gemm_universal import TraitSpec

                TraitSpec(**pin)

    def test_no_shipped_config_still_uses_the_old_shape(self):
        for path in sorted(CONFIGS.glob("*.json")):
            raw = json.loads(path.read_text())
            with self.subTest(config=path.name):
                self.assertNotIn("screening_traits", raw)
                self.assertNotIn("trait_config", raw)


if __name__ == "__main__":
    unittest.main()
