# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The --split-k 0 sweep must be a strict superset of --split-k -1.
#
# --split-k -1 ("auto") resolves the degree per tile config through CK's
# formula, and that is what dispatch ships. --split-k 0 ("sweep") used to walk
# only the fixed power-of-two ladder _SPLIT_K_AUTO. The CK formula is a floor
# division that essentially never lands on a power of two, so the two sets did
# not intersect and a sweep run could not answer "did the heuristic pick well?".
#
# These tests pin the two properties that fix buys us:
#   1. the merge is order-preserving, deduplicating, and superset-forming;
#   2. the degree the sweep adds is produced by the *same* helper the -1 build
#      path calls, so "what we sweep" cannot drift from "what we ship".
#
# All CPU-only: the helpers are pure arithmetic over the problem shape, no GPU
# and no kernel build.

from __future__ import annotations

import unittest

from benchmarks.common.benchmark_implicit_gemm_conv import (
    _SPLIT_K_AUTO,
    _ck_split_k_dgrad,
    _ck_split_k_wgrad,
    _dgrad_sweep_split_k_values,
    _is_wmma_arch,
    _prune_skips_wgrad_combo,
    _sweep_degrees,
    _wgrad_runtime_split_k_degrees,
    _wgrad_sweep_split_k_values,
)

_ARCH = "gfx950"


def _problem(N=8, H=56, W=56, C=64, K=64, Y=3, X=3):
    """A ConvProblem for the shape the sweep-vs-ship gap was measured on."""
    from kernels.common.conv_implicit_gemm import ConvProblem

    return ConvProblem(N=N, Hi=H, Wi=W, C=C, K=K, Y=Y, X=X)


def _problem_3d(N=2, Di=16, H=32, W=32, C=64, K=64, Z=3, Y=3, X=3):
    """A 3-D ConvProblem. ConvProblem requires all of Di/Z/sD/pD/dD together."""
    from kernels.common.conv_implicit_gemm import ConvProblem

    return ConvProblem(
        N=N,
        Di=Di,
        Hi=H,
        Wi=W,
        C=C,
        K=K,
        Z=Z,
        Y=Y,
        X=X,
        sD=1,
        pD=1,
        dD=1,
        pH=1,
        pW=1,
    )


class TestSweepDegrees(unittest.TestCase):
    """_sweep_degrees: the ladder merged with one CK degree."""

    def test_off_ladder_degree_is_inserted_in_descending_position(self):
        # 56 is a real CK degree for this shape and is not a power of two.
        got = _sweep_degrees(56)
        self.assertEqual(got, (128, 64, 56, 32, 16, 8, 4, 2, 1))

    def test_descending_order_is_preserved(self):
        # --split-k-prune walks degrees high -> low and stops at the first
        # regression; the pending re-sort keys on -split_k to reproduce it.
        # An out-of-order ladder would prune the wrong tail.
        for ck in (1, 2, 3, 56, 64, 170, 1000):
            got = _sweep_degrees(ck)
            self.assertEqual(
                list(got), sorted(got, reverse=True), f"not descending for ck={ck}"
            )

    def test_on_ladder_degree_is_not_duplicated(self):
        # When the heuristic lands on the ladder we must reuse that entry, not
        # time the same degree twice.
        got = _sweep_degrees(64)
        self.assertEqual(got, _SPLIT_K_AUTO)
        self.assertEqual(len(got), len(set(got)))

    def test_result_is_always_a_superset_of_the_ladder(self):
        for ck in (1, 3, 7, 14, 25, 28, 42, 51, 56, 64, 85, 102, 170):
            self.assertTrue(
                set(_SPLIT_K_AUTO).issubset(set(_sweep_degrees(ck))),
                f"ladder not preserved for ck={ck}",
            )

    def test_ck_degree_is_always_present(self):
        for ck in (1, 3, 56, 64, 170):
            self.assertIn(ck, _sweep_degrees(ck))

    def test_adds_at_most_one_entry(self):
        for ck in (1, 3, 56, 64, 170):
            self.assertLessEqual(len(_sweep_degrees(ck)), len(_SPLIT_K_AUTO) + 1)

    def test_non_positive_degree_falls_back_to_the_bare_ladder(self):
        # A degenerate/empty grid can only ever yield >=1 from the helpers, but
        # the merge must not inject 0 or a negative into the launch ladder if
        # one ever arrives -- those are not legal split-K degrees.
        for ck in (0, -1):
            self.assertEqual(_sweep_degrees(ck), _SPLIT_K_AUTO)


class TestCkDegreeMatchesTheShippedPath(unittest.TestCase):
    """The sweep's degree must come from the same code path as --split-k -1."""

    def test_wgrad_helper_uses_the_kernels_own_gemm_dims(self):
        # Not a re-derivation of the formula -- that would be tautological. The
        # point is that the benchmark feeds select_split_k_wgrad the SAME GEMM
        # dims the kernel builder does, so the swept degree is the shipped one.
        from kernels.common.conv_implicit_gemm_wgrad import _wg_K, _wg_M, _wg_N
        from rocke.helpers.split_k import select_split_k_wgrad

        for p in (_problem(), _problem_3d()):
            for tile_m, tile_n, tile_k in (
                (128, 128, 64),
                (64, 256, 32),
                (256, 64, 128),
            ):
                expected = select_split_k_wgrad(
                    wg_M=_wg_M(p),
                    wg_N=_wg_N(p),
                    wg_K=_wg_K(p),
                    tile_m=tile_m,
                    tile_n=tile_n,
                    tile_k=tile_k,
                    arch=_ARCH,
                ).split_k
                self.assertEqual(
                    _ck_split_k_wgrad(p, tile_m, tile_n, tile_k, _ARCH), expected
                )

    def test_wgrad_helper_honours_the_3d_depth_terms(self):
        # Regression guard for a real defect: the open-coded dims this helper
        # replaced dropped Z from GEMM-N and Do from GEMM-K, so on a 3-D conv
        # the benchmark resolved a degree several times larger than the one the
        # builder ships -- and the change marks that row '*' as "what ships".
        from kernels.common.conv_implicit_gemm_wgrad import _wg_K, _wg_N

        p = _problem_3d()
        self.assertTrue(p.is_3d)
        # The dropped terms really are load-bearing on this shape.
        self.assertNotEqual(_wg_N(p), p.Y * p.X * p.cpg)
        self.assertNotEqual(_wg_K(p), p.N * p.Ho * p.Wo)
        # And they really do change the resolved degree.
        from rocke.helpers.split_k import select_split_k_wgrad

        for tile_m, tile_n, tile_k in ((128, 128, 64), (256, 64, 64)):
            two_d_only = select_split_k_wgrad(
                wg_M=p.kpg,
                wg_N=p.Y * p.X * p.cpg,
                wg_K=p.N * p.Ho * p.Wo,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                arch=_ARCH,
            ).split_k
            self.assertNotEqual(
                _ck_split_k_wgrad(p, tile_m, tile_n, tile_k, _ARCH), two_d_only
            )

    def test_dgrad_helper_uses_the_kernels_own_gemm_dims(self):
        from kernels.common.conv_implicit_gemm_dgrad import _dg_K, _dg_M, _dg_N
        from rocke.helpers.split_k import select_split_k_wgrad

        p = _problem()
        for tile_m, tile_n, tile_k in ((128, 128, 64), (64, 256, 32)):
            expected = select_split_k_wgrad(
                wg_M=_dg_M(p),
                wg_N=_dg_N(p),
                wg_K=_dg_K(p),
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                arch=_ARCH,
            ).split_k
            self.assertEqual(
                _ck_split_k_dgrad(p, tile_m, tile_n, tile_k, _ARCH, is_wmma=False),
                expected,
            )

    def test_dgrad_wmma_pins_to_one(self):
        # WMMA has no split-K on the dgrad path; the build site pinned this
        # before the helper existed and the helper must keep the pin.
        p = _problem()
        self.assertEqual(_ck_split_k_dgrad(p, 128, 128, 64, "gfx1201", is_wmma=True), 1)

    def test_is_wmma_arch_agrees_with_wave_size(self):
        from rocke.core.arch import ArchTarget

        for arch in ("gfx942", "gfx950", "gfx1201"):
            self.assertEqual(
                _is_wmma_arch(arch), ArchTarget.from_gfx(arch).wave_size == 32
            )

    def test_degree_varies_with_tile_config(self):
        # The whole design rests on this: the CK degree is per tile config (it
        # depends on tile_m/tile_n via base_grid), not one extra value per run.
        # If this ever collapses to a constant, a single global extra degree
        # would have been enough and the per-config plumbing is dead weight.
        p = _problem()
        degrees = {
            _ck_split_k_wgrad(p, tm, tn, 64, _ARCH)
            for tm in (64, 128, 256)
            for tn in (64, 128, 256)
        }
        self.assertGreater(len(degrees), 1)


class TestSweepClosesTheGapOnTheMeasuredShape(unittest.TestCase):
    """The regression this change exists to prevent."""

    def test_ck_degrees_are_mostly_off_ladder(self):
        # Documents the motivating measurement: on this shape essentially none
        # of the CK degrees are powers of two, so before the merge the sweep and
        # the shipped degree had (near) zero overlap.
        p = _problem()
        ck = [
            _ck_split_k_wgrad(p, tm, tn, tk, _ARCH)
            for tm in (64, 128, 256)
            for tn in (64, 128, 256)
            for tk in (32, 64, 128)
        ]
        off_ladder = [d for d in ck if d not in _SPLIT_K_AUTO]
        self.assertGreater(
            len(off_ladder),
            len(ck) // 2,
            "expected most CK degrees off the power-of-two ladder",
        )

    def test_every_ck_degree_is_covered_after_the_merge(self):
        p = _problem()
        for tm in (64, 128, 256):
            for tn in (64, 128, 256):
                for tk in (32, 64, 128):
                    d = _ck_split_k_wgrad(p, tm, tn, tk, _ARCH)
                    self.assertIn(d, _sweep_degrees(d))


class TestSweptDegreeEqualsShippedDegree(unittest.TestCase):
    """The property the whole change exists for.

    Not "the helper reproduces the formula" (tautological) but "the degree the
    sweep injects is the degree the -1 build path actually resolves". These go
    through _build_wgrad_one / _build_dgrad_one, i.e. the real build entry
    points, and compare their resolved_split_k against the swept ladder.
    """

    # Geometries that actually pass spec validation on the problem below. wgrad
    # needs the cshuffle epilogue here; dgrad rejects cshuffle for split_k > 1
    # and so uses default.
    _TILES = ((16, 16, 16), (16, 16, 32), (16, 16, 64), (32, 16, 16))

    def test_wgrad_minus_one_degree_is_in_the_sweep_ladder(self):
        from benchmarks.common.benchmark_implicit_gemm_conv import _build_wgrad_one

        p = _problem(N=2, H=28, W=28, C=32, K=32)
        checked = 0
        for tile_m, tile_n, tile_k in self._TILES:
            combo = (tile_m, tile_n, tile_k, 1, 1, 16, "basic", "cshuffle", False, -1)
            got = _build_wgrad_one((combo, p, "fp16", _ARCH))
            if got is None:
                continue  # combo rejected by spec validation, not by split-K
            resolved = got[2]
            self.assertEqual(
                resolved,
                _ck_split_k_wgrad(p, tile_m, tile_n, tile_k, _ARCH),
                "the -1 build path and the sweep helper disagree",
            )
            self.assertIn(
                resolved,
                _sweep_degrees(_ck_split_k_wgrad(p, tile_m, tile_n, tile_k, _ARCH)),
                "--split-k 0 would not measure the degree --split-k -1 ships",
            )
            checked += 1
        self.assertGreater(checked, 0, "no combo built; test proved nothing")

    def test_dgrad_minus_one_degree_is_in_the_sweep_ladder(self):
        from benchmarks.common.benchmark_implicit_gemm_conv import _build_dgrad_one

        p = _problem(N=2, H=28, W=28, C=32, K=32)
        is_wmma = _is_wmma_arch(_ARCH)
        checked = 0
        for tile_m, tile_n, tile_k in self._TILES:
            combo = (tile_m, tile_n, tile_k, 1, 1, 16, "basic", "default", -1)
            got = _build_dgrad_one((combo, p, "fp16", _ARCH, 1, 1, 1))
            if got is None:
                continue
            resolved = got[2]
            expected = _ck_split_k_dgrad(
                p, tile_m, tile_n, tile_k, _ARCH, is_wmma=is_wmma
            )
            self.assertEqual(resolved, expected)
            self.assertIn(resolved, _sweep_degrees(expected))
            checked += 1
        self.assertGreater(checked, 0, "no combo built; test proved nothing")


class TestSweepCallSitesMergeTheCkDegree(unittest.TestCase):
    """The sweep AXES themselves, not just the pure merge helper.

    These are the tests that fail if someone reverts a call site back to a bare
    _SPLIT_K_AUTO while leaving _sweep_degrees intact -- the exact regression the
    helper-only tests above cannot see.
    """

    _GEOM = (128, 128, 64)

    def _ck_wgrad(self, p):
        return _ck_split_k_wgrad(p, *self._GEOM, _ARCH)

    def _ck_dgrad(self, p):
        return _ck_split_k_dgrad(p, *self._GEOM, _ARCH, is_wmma=_is_wmma_arch(_ARCH))

    # -- wgrad compile-time legs (async, basic) ----------------------------

    def test_wgrad_async_leg_sweeps_the_merged_ladder(self):
        p = _problem()
        got = _wgrad_sweep_split_k_values(0, "mem", True, p, *self._GEOM, _ARCH)
        self.assertEqual(got, _sweep_degrees(self._ck_wgrad(p)))
        self.assertIn(self._ck_wgrad(p), got)
        self.assertTrue(set(_SPLIT_K_AUTO).issubset(got))

    def test_wgrad_basic_leg_sweeps_the_merged_ladder(self):
        # 'basic' is runtime-incapable too; a run-level predicate that tested
        # only async_dma once dropped every basic combo from the sweep.
        p = _problem()
        got = _wgrad_sweep_split_k_values(0, "basic", False, p, *self._GEOM, _ARCH)
        self.assertEqual(got, _sweep_degrees(self._ck_wgrad(p)))
        self.assertIn(self._ck_wgrad(p), got)

    def test_wgrad_runtime_capable_leg_still_uses_the_runtime_sentinel(self):
        # These compile one runtime-degree kernel and sweep at launch time, so
        # the build axis must stay the single sentinel 0 -- widening it here
        # would compile the whole ladder for nothing.
        p = _problem()
        for pipeline in ("mem", "compv3", "compv4", "wavelet"):
            self.assertEqual(
                _wgrad_sweep_split_k_values(0, pipeline, False, p, *self._GEOM, _ARCH),
                (0,),
            )

    def test_wgrad_fixed_degree_is_not_widened(self):
        p = _problem()
        for fixed in (-1, 1, 2, 8):
            for pipeline, async_dma in (
                ("mem", False),
                ("basic", False),
                ("mem", True),
            ):
                self.assertEqual(
                    _wgrad_sweep_split_k_values(
                        fixed, pipeline, async_dma, p, *self._GEOM, _ARCH
                    ),
                    (fixed,),
                )

    # -- wgrad runtime (launch-time) leg -----------------------------------

    def test_wgrad_runtime_degrees_sweep_the_merged_ladder(self):
        p = _problem()
        ck = self._ck_wgrad(p)
        got = _wgrad_runtime_split_k_degrees(True, 0, ck)
        self.assertEqual(got, _sweep_degrees(ck))
        self.assertIn(ck, got)
        self.assertTrue(set(_SPLIT_K_AUTO).issubset(got))

    def test_wgrad_fixed_kernel_launches_exactly_its_own_degree(self):
        for resolved in (1, 4, 56):
            self.assertEqual(
                _wgrad_runtime_split_k_degrees(False, resolved, 56), (resolved,)
            )

    # -- dgrad -------------------------------------------------------------

    def test_dgrad_sweep_merges_the_ck_degree(self):
        p = _problem()
        got = _dgrad_sweep_split_k_values(
            0, p, *self._GEOM, _ARCH, is_wmma=_is_wmma_arch(_ARCH)
        )
        self.assertEqual(got, _sweep_degrees(self._ck_dgrad(p)))
        self.assertIn(self._ck_dgrad(p), got)
        self.assertTrue(set(_SPLIT_K_AUTO).issubset(got))

    def test_dgrad_fixed_degree_is_not_widened(self):
        p = _problem()
        for fixed in (-1, 1, 2, 8):
            self.assertEqual(
                _dgrad_sweep_split_k_values(
                    fixed, p, *self._GEOM, _ARCH, is_wmma=False
                ),
                (fixed,),
            )

    def test_dgrad_wmma_sweep_still_pins_the_ck_degree_to_one(self):
        p = _problem()
        got = _dgrad_sweep_split_k_values(0, p, *self._GEOM, _ARCH, is_wmma=True)
        # WMMA's CK degree is 1, which is already on the ladder -> no widening.
        self.assertEqual(got, _SPLIT_K_AUTO)

    # -- the property, stated directly -------------------------------------

    def test_every_sweeping_leg_covers_the_shipped_degree(self):
        # One assertion for the headline contract, across every leg that
        # actually enumerates degrees, on several tile configs.
        p = _problem()
        for tile in ((128, 128, 64), (64, 256, 32), (256, 64, 128), (32, 32, 16)):
            ck_w = _ck_split_k_wgrad(p, *tile, _ARCH)
            self.assertIn(
                ck_w,
                _wgrad_sweep_split_k_values(0, "basic", False, p, *tile, _ARCH),
                f"wgrad basic leg misses the shipped degree at {tile}",
            )
            self.assertIn(
                ck_w,
                _wgrad_sweep_split_k_values(0, "mem", True, p, *tile, _ARCH),
                f"wgrad async leg misses the shipped degree at {tile}",
            )
            self.assertIn(
                ck_w,
                _wgrad_runtime_split_k_degrees(True, 0, ck_w),
                f"wgrad runtime leg misses the shipped degree at {tile}",
            )
            ck_d = _ck_split_k_dgrad(p, *tile, _ARCH, is_wmma=False)
            self.assertIn(
                ck_d,
                _dgrad_sweep_split_k_values(0, p, *tile, _ARCH, is_wmma=False),
                f"dgrad leg misses the shipped degree at {tile}",
            )


class TestPruneNeverSkipsTheShippedDegree(unittest.TestCase):
    """--split-k-prune must not prune away the row the sweep exists to show.

    This guard is what protects the async and 'basic' legs, where every degree
    is compiled as its own combo. The per-launch guard in the measure loop only
    covers the runtime leg (one combo, whole ladder), so without this one a
    pruned config silently dropped its CK combo and no '*' row was ever emitted
    for that config -- defeating the point of the change on two of three legs.
    """

    _KEY = ("cfg", 128, 128, 64)
    _CK = 56

    def test_unpruned_config_is_never_skipped(self):
        for degree in (128, 64, 56, 32, 1):
            self.assertFalse(
                _prune_skips_wgrad_combo(self._KEY, set(), degree, self._CK)
            )

    def test_pruned_config_skips_ordinary_ladder_degrees(self):
        pruned = {self._KEY}
        for degree in (128, 64, 32, 16, 8, 4, 2, 1):
            self.assertTrue(
                _prune_skips_wgrad_combo(self._KEY, pruned, degree, self._CK),
                f"ladder degree {degree} should be pruned",
            )

    def test_pruned_config_still_measures_the_ck_degree(self):
        self.assertFalse(
            _prune_skips_wgrad_combo(self._KEY, {self._KEY}, self._CK, self._CK),
            "the shipped degree was pruned away",
        )

    def test_exemption_also_applies_when_the_ck_degree_is_on_the_ladder(self):
        # Dedup case: the rule must not change meaning just because the
        # heuristic happened to land on a power of two.
        self.assertFalse(_prune_skips_wgrad_combo(self._KEY, {self._KEY}, 64, 64))
        self.assertTrue(_prune_skips_wgrad_combo(self._KEY, {self._KEY}, 32, 64))

    def test_other_configs_are_unaffected(self):
        other = ("cfg", 64, 64, 32)
        self.assertFalse(_prune_skips_wgrad_combo(other, {self._KEY}, 32, self._CK))


class TestFixedSplitKIsUnaffected(unittest.TestCase):
    """--split-k values other than 0 must not pick up the merged ladder."""

    def test_builders_pass_fixed_degrees_through_untouched(self):
        from benchmarks.common.benchmark_implicit_gemm_conv import _build_wgrad_one

        p = _problem(N=2, H=16, W=16, C=32, K=32)
        for fixed in (1, 2, 8):
            combo = (128, 128, 64, 2, 2, 32, "basic", "default", False, fixed)
            got = _build_wgrad_one((combo, p, "fp16", _ARCH))
            if got is not None:
                self.assertEqual(
                    got[2], fixed, f"fixed --split-k {fixed} was not passed through"
                )

    def test_sweep_merge_is_never_consulted_for_a_fixed_degree(self):
        # _sweep_degrees widens the ladder; a fixed degree must not be widened.
        # Guards the shape of the contract rather than a call site: if a future
        # edit routes a fixed degree through the merge, the result would be a
        # 9-element ladder instead of the single requested degree.
        for fixed in (1, 2, 8, 64):
            self.assertNotEqual(_sweep_degrees(fixed), (fixed,))


if __name__ == "__main__":
    unittest.main()
