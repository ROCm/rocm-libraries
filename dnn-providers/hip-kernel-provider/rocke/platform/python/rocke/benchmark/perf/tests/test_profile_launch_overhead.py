# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the launch-overhead example's pure helpers (no GPU, no rocke).

Covers the helpers that turn raw timings into a claim. Each test guards a
specific way the harness could report a confident, wrong number: a share that
ignores its own measurement, a delta smaller than the drift it was measured
against, or a queue that blocks inside every chunk while the across-chunk
series stays flat.
"""
import unittest

from rocke.benchmark.perf.examples import profile_launch_overhead as ex


def _samples(us, n=40):
    """n identical per-launch samples, in seconds."""
    return [us * 1e-6] * n


def _saturating(cost_us=8.0, free=50, blocked_us=40.0):
    """Enqueue is cheap until the queue fills, then it blocks."""
    state = {"n": 0}

    def fn():
        state["n"] += 1
        delay = cost_us if state["n"] <= free else blocked_us
        end = ex.PERF() + delay * 1e-6
        while ex.PERF() < end:
            pass

    return fn, state


def _fixed_cost(us):
    """A call whose cost does not depend on how often it is called."""

    def fn():
        end = ex.PERF() + us * 1e-6
        while ex.PERF() < end:
            pass

    return fn


class TestSigShape(unittest.TestCase):
    def test_kernarg_layout_matches_the_alignment_rule(self):
        # 11 pointers (88 B) then an i32 at offset 88: already 4-aligned, so
        # 92 B with no hole. This is GDN decode's signature shape.
        sig = [{"name": f"p{i}", "type": "ptr<bf16>"} for i in range(11)]
        sig.append({"name": "n", "type": "i32"})
        self.assertEqual(
            ex.sig_shape(sig),
            {"nargs": 12, "nptr": 11, "nscalar": 1, "kernarg_bytes": 92},
        )
        # i32 at 0..4 then a pointer must start at 8 -- 4 bytes of padding.
        self.assertEqual(
            ex.sig_shape(
                [{"name": "n", "type": "i32"}, {"name": "p", "type": "ptr<f32>"}]
            )["kernarg_bytes"],
            16,
        )


class TestSummarizeAB(unittest.TestCase):
    def test_effect_above_repeat_spread_is_real(self):
        out = ex.summarize_ab(
            _samples(10.0), _samples(10.02), _samples(8.7), _samples(8.72)
        )["p10_us"]
        self.assertTrue(out["delta_exceeds_noise"])
        self.assertAlmostEqual(out["delta_us_per_launch"], 1.3, places=6)

    def test_effect_below_repeat_spread_is_refused(self):
        # The case that stops the harness inventing a win: arms differ by
        # 0.05 us while the SAME arm repeats 0.40 us apart.
        out = ex.summarize_ab(
            _samples(10.0), _samples(10.4), _samples(9.95), _samples(10.3)
        )["p10_us"]
        self.assertFalse(out["delta_exceeds_noise"])


class TestSubstituteShare(unittest.TestCase):
    def test_each_arm_share_uses_its_own_measured_total(self):
        """Regression: arm A's total was rebuilt from arm B's remainder, so a
        measured arm A could change tenfold without moving the reported
        share."""
        low = ex.substitute_share(10.0, 9.0, 2.0, 1.0)
        high = ex.substitute_share(100.0, 9.0, 2.0, 1.0)
        self.assertFalse(low["model_estimate"])
        self.assertAlmostEqual(low["packing_share_armA_pct"], 20.0)
        self.assertAlmostEqual(high["total_us_armA"], 100.0)
        self.assertAlmostEqual(high["packing_share_armA_pct"], 2.0)
        # The arms differ only in their packer, so the remainders should agree;
        # the residual is what exposes it when they do not.
        self.assertAlmostEqual(low["remainder_residual_us"], 0.0)
        self.assertAlmostEqual(
            ex.substitute_share(12.5, 9.0, 2.0, 1.0)["remainder_residual_us"], 2.5
        )

    def test_substitution_swaps_denominator_terms_and_says_it_is_a_model(self):
        # Scaling only the numerator is the bias this exists to avoid: the
        # target signature also makes two denominator terms more expensive.
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
        self.assertTrue(with_swap["model_estimate"])
        self.assertGreater(
            numerator_only["packing_share_armA_pct"],
            with_swap["packing_share_armA_pct"],
        )
        self.assertAlmostEqual(
            with_swap["nonpacking_us"], numerator_only["nonpacking_us"] + 0.27
        )

    def test_residual_tolerance_does_not_tighten_as_the_run_gets_quieter(self):
        """A quieter run drives the noise floor toward zero. Judged against the
        floor alone, a residual of 0.1% of the remainder then reads as a 4x
        violation -- the check would get harder to pass the better the
        measurement. The relative term floors it."""
        quiet = ex.substitute_share(
            9.009, 9.0, 1.0, 1.0, noise_floor_us=0.0023  # residual 0.009 us on 8 us
        )
        self.assertTrue(quiet["remainder_within_tol"])
        self.assertGreater(quiet["remainder_tolerance_us"], 0.0023)
        # A residual that is genuinely large still fails.
        skewed = ex.substitute_share(10.0, 9.0, 1.0, 1.0, noise_floor_us=0.0023)
        self.assertFalse(skewed["remainder_within_tol"])
        # Without a noise floor the helper reports the residual and judges
        # nothing, so a caller cannot read a verdict that was never computed.
        self.assertIsNone(
            ex.substitute_share(10.0, 9.0, 1.0, 1.0)["remainder_within_tol"]
        )


class TestBackPressureDetection(unittest.TestCase):
    """Async enqueue is only a host clock while the host outruns the device."""

    def test_across_chunk_check_is_blind_to_uniform_saturation(self):
        # Why the other two checks exist: the queue is drained between chunks,
        # so every chunk saturates identically and the series is flat.
        self.assertLess(abs(ex.stepup_check([12.0e-6] * 20)["ratio"] - 1.0), 0.01)
        self.assertAlmostEqual(
            ex.stepup_check([10e-6] * 10 + [20e-6] * 10)["ratio"], 2.0
        )

    def test_in_chunk_stepup_catches_a_queue_filling_mid_chunk(self):
        fn, _ = _saturating()
        self.assertTrue(ex.in_chunk_stepup(fn, 200, segments=4)["back_pressured"])
        self.assertFalse(
            ex.in_chunk_stepup(lambda: None, 400, segments=4)["back_pressured"]
        )

    def test_chunk_size_sensitivity_catches_cost_growing_with_chunk(self):
        # Pure host work cannot care how long we go between drains.
        fn, state = _saturating(free=30)
        out = ex.chunk_size_sensitivity(fn, lambda: state.update(n=0), 20, 200, reps=1)
        self.assertTrue(out["back_pressured"])
        # A path whose cost does not depend on the drain interval stays quiet.
        # The fixture spends a fixed 3 us per call so per-call cost is set by
        # the work, not by loop overhead that scales with iteration count.
        self.assertFalse(
            ex.chunk_size_sensitivity(_fixed_cost(3.0), lambda: None, 50, 200)[
                "back_pressured"
            ]
        )

    def test_one_stall_trips_the_single_sample_gate_but_not_min_of_reps(self):
        """Both gates invalidate a run, so a false positive throws away a good
        measurement. Timing each size exactly once, one scheduler stall in the
        large-chunk timing is enough to fire -- which is what was observed on a
        shared node. Min-of-reps keeps a systematic effect and drops a one-off,
        the same reason micro() takes a minimum.

        Call 100 is chosen so the stall lands in the LARGE timing at reps=1
        (small takes calls 1-50, large 51-250) and in a discarded SMALL rep at
        reps=5 (small takes 1-250 across five reps, large starts at 251)."""

        def stalling_fixture(stall_on=100, stall_us=400.0):
            state = {"call": 0}

            def fn():
                state["call"] += 1
                if state["call"] == stall_on:
                    end = ex.PERF() + stall_us * 1e-6
                    while ex.PERF() < end:
                        pass

            return fn

        single = ex.chunk_size_sensitivity(
            stalling_fixture(), lambda: None, 50, 200, reps=1
        )
        self.assertTrue(single["back_pressured"])
        repeated = ex.chunk_size_sensitivity(
            stalling_fixture(), lambda: None, 50, 200, reps=5
        )
        self.assertFalse(repeated["back_pressured"])
        self.assertEqual(repeated["reps"], 5)


if __name__ == "__main__":
    unittest.main()
