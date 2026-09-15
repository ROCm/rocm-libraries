# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the launch-overhead example's pure helpers (no GPU, no rocKE).

The helpers under test are the ones that turn raw timings into a claim:
``summarize_ab`` decides whether an A/B difference is real, and
``substitute_share`` re-expresses a measured share for a different kernel
signature. A bug in either produces a number that looks authoritative and is
wrong, which is worse than no measurement -- so they are tested directly rather
than only through a GPU run nobody can rerun in CI.

The negative cases matter most: a harness that cannot REFUSE to call noise an
effect will eventually manufacture one.
"""
import unittest

from rocke.benchmark.perf.examples import profile_launch_overhead as ex


class TestSigShape(unittest.TestCase):
    """Kernarg layout must match what ``packing.py`` actually emits."""

    def test_all_pointer_signature_packs_without_padding(self):
        sig = [{"name": f"p{i}", "type": "ptr<f16>"} for i in range(3)]
        self.assertEqual(
            ex.sig_shape(sig),
            {"nargs": 3, "nptr": 3, "nscalar": 0, "kernarg_bytes": 24},
        )

    def test_trailing_scalar_needs_no_interior_padding(self):
        # 11 pointers (88 B, 8-aligned) then one i32 at offset 88: already
        # 4-aligned, so the blob is 92 B with no hole. This is GDN decode's
        # signature shape.
        sig = [{"name": f"p{i}", "type": "ptr<bf16>"} for i in range(11)]
        sig.append({"name": "n", "type": "i32"})
        self.assertEqual(
            ex.sig_shape(sig),
            {"nargs": 12, "nptr": 11, "nscalar": 1, "kernarg_bytes": 92},
        )

    def test_scalar_before_pointer_forces_alignment_padding(self):
        # i32 at 0..4, then a pointer must start at 8 -- 4 bytes of padding.
        # Getting this wrong understates the blob and silently mis-sizes the
        # memcpy term in the substitution arithmetic.
        sig = [
            {"name": "n", "type": "i32"},
            {"name": "p", "type": "ptr<f32>"},
        ]
        self.assertEqual(ex.sig_shape(sig)["kernarg_bytes"], 16)


class TestSummarizeAB(unittest.TestCase):
    """The effect/noise decision, which is what makes a delta quotable."""

    @staticmethod
    def _samples(us, n=40):
        """n identical per-launch samples, in seconds."""
        return [us * 1e-6] * n

    def test_effect_larger_than_repeat_spread_is_called_real(self):
        out = ex.summarize_ab(
            self._samples(10.0),
            self._samples(10.02),  # A' -- same arm, tiny drift
            self._samples(8.7),
            self._samples(8.72),
        )["p10_us"]
        self.assertTrue(out["delta_exceeds_noise"])
        self.assertAlmostEqual(out["delta_us_per_launch"], 1.3, places=6)
        self.assertAlmostEqual(out["noise_floor_us"], 0.02, places=6)

    def test_effect_smaller_than_repeat_spread_is_refused(self):
        # The case that stops the harness inventing a win: arms differ by
        # 0.05 us while the SAME arm repeats 0.40 us apart. Nothing is
        # measurable here and the summary must say so.
        out = ex.summarize_ab(
            self._samples(10.0),
            self._samples(10.4),
            self._samples(9.95),
            self._samples(10.3),
        )["p10_us"]
        self.assertFalse(out["delta_exceeds_noise"])

    def test_noise_floor_takes_the_worse_of_the_two_arms(self):
        out = ex.summarize_ab(
            self._samples(10.0),
            self._samples(10.01),  # A spread 0.01
            self._samples(8.0),
            self._samples(8.30),  # B spread 0.30 -- the binding one
        )["p10_us"]
        self.assertAlmostEqual(out["noise_floor_us"], 0.30, places=6)

    def test_every_estimator_is_reported(self):
        out = ex.summarize_ab(
            self._samples(10.0),
            self._samples(10.0),
            self._samples(9.0),
            self._samples(9.0),
        )
        self.assertEqual(set(out), {"min_us", "p10_us", "median_us"})


class TestSubstituteShare(unittest.TestCase):
    """Re-expressing a measured share for another signature."""

    def test_share_without_substitution_is_plain_arithmetic(self):
        out = ex.substitute_share(
            total_armA_us=10.0,
            total_armB_us=9.0,
            pack_armA_us=2.0,
            pack_armB_us=1.0,
        )
        self.assertAlmostEqual(out["nonpacking_us"], 8.0)
        self.assertAlmostEqual(out["packing_share_armA_pct"], 20.0)
        self.assertAlmostEqual(out["packing_share_armB_pct"], 11.111111, places=5)
        self.assertAlmostEqual(out["saving_us"], 1.0)

    def test_swapping_a_denominator_term_lowers_the_substituted_share(self):
        # Scaling only the numerator is the bias this argument exists to avoid:
        # the target signature also makes two denominator terms more expensive,
        # so crediting packing with all of the growth overstates its share.
        common = dict(
            total_armA_us=10.0,
            total_armB_us=9.0,
            pack_armA_us=2.0,
            pack_armB_us=1.0,
            pack_target_armA_us=3.0,
            pack_target_armB_us=1.5,
        )
        numerator_only = ex.substitute_share(**common)
        with_swap = ex.substitute_share(**common, swap=[(0.14, 0.20), (0.69, 0.90)])
        self.assertGreater(
            numerator_only["packing_share_armA_pct"],
            with_swap["packing_share_armA_pct"],
        )
        # The remainder grows by exactly the swapped deltas: 0.06 + 0.21.
        self.assertAlmostEqual(
            with_swap["nonpacking_us"], numerator_only["nonpacking_us"] + 0.27
        )


class TestStepupCheck(unittest.TestCase):
    """Back-pressure detector: async enqueue stops being a host clock if the
    device falls behind, and that shows up as the second half of a block
    running slower than the first."""

    def test_flat_block_has_unit_ratio(self):
        out = ex.stepup_check([10e-6] * 20)
        self.assertTrue(out["checked"])
        self.assertAlmostEqual(out["ratio"], 1.0)

    def test_block_that_doubles_halfway_is_detected(self):
        out = ex.stepup_check([10e-6] * 10 + [20e-6] * 10)
        self.assertAlmostEqual(out["ratio"], 2.0)

    def test_too_few_samples_reports_unchecked_rather_than_guessing(self):
        self.assertEqual(ex.stepup_check([10e-6] * 4), {"checked": False})


if __name__ == "__main__":
    unittest.main()
