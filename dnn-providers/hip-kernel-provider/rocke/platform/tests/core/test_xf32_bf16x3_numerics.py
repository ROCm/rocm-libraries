# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What the BF16x3 recipe is worth numerically (XF32 plan, step 9).

Steps 5-8 prove the recipe is *emitted* and *codegen-able*. Neither says
anything about what it computes. This pins the arithmetic itself, as a model
in plain numpy, so the accuracy claims the feature rests on are checked on
every host rather than only on a gfx950 node:

* the split is an error-free transform -- ``big + small`` recovers roughly
  twice bf16's significand, which is where the precision comes from;
* three terms land far closer to an fp32 reference than the single ``big*big``
  term a plain BF16 MMA would give, and closer than real TF32 -- the emulation
  is *more* accurate than the thing it stands in for, not a cheaper
  approximation of it;
* dropping ``small*small`` costs nothing measurable, which is why the recipe
  is three MMAs and not four;
* the term order is load-bearing, because fp32 accumulation is not
  associative.

This is a model of the arithmetic, not of the hardware. It uses the same
round-to-nearest-even narrowing the ISA's ``v_cvt_pk_bf16_f32`` does (pinned
by ``test_xf32_comgr_compile.py``), but accumulates over K in plain sequential
fp32, which is not necessarily the order an MFMA uses internally. So the
tolerances here are relative-accuracy properties; nothing claims bit equality
with a device. The on-device half of step 9 lives in
``rocke/examples/gfx950/xf32/bf16x3_parity.py``.
"""

from __future__ import annotations

import unittest

import numpy as np

_K = 256
_TRIALS = 64
_SEED = 20_208


def _to_bf16(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 -> fp32, rounding to nearest even.

    Returned as fp32 so the value can be used in fp32 arithmetic directly.
    RNE, not truncation: LLVM's ``fptrunc float to bfloat`` is RNE and the
    backend selects the packed RNE convert, so a truncating model here would
    be measuring a recipe nobody ships.
    """
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    # Add half an ULP of the bf16 grid, biased by the retained LSB so that
    # exact ties go to even, then drop the low 16 bits.
    bumped = (
        u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    ) & np.uint32(0xFFFF0000)
    return bumped.view(np.float32)


def _to_tf32(x: np.ndarray) -> np.ndarray:
    """fp32 -> real TF32 (10 explicit significand bits) -> fp32, RNE."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    bumped = (
        u + np.uint32(0x0FFF) + ((u >> np.uint32(13)) & np.uint32(1))
    ) & np.uint32(0xFFFFE000)
    return bumped.view(np.float32)


def _split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Dekker-style split the lowering emits: ``big``, then the residual."""
    big = _to_bf16(x)
    small = _to_bf16((x - big).astype(np.float32))
    return big, small


def _fp32_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sequential fp32 accumulation -- the model's notion of one MMA."""
    acc = np.float32(0.0)
    for i in range(a.shape[0]):
        acc = np.float32(acc + np.float32(a[i] * b[i]))
    return acc


def _reference(a: np.ndarray, b: np.ndarray) -> np.float64:
    """Ground truth, in fp64. Not reachable by any of the models."""
    return np.dot(a.astype(np.float64), b.astype(np.float64))


def _bf16x3_dot(a: np.ndarray, b: np.ndarray, *, order=(0, 1, 2)) -> np.float32:
    """The emulation: three BF16 MMAs threading one fp32 accumulator.

    ``order`` permutes the three terms so a test can show the order matters.
    """
    a_big, a_small = _split(a)
    b_big, b_small = _split(b)
    terms = ((a_small, b_big), (a_big, b_small), (a_big, b_big))
    acc = np.float32(0.0)
    for idx in order:
        x, y = terms[idx]
        acc = np.float32(acc + _fp32_dot(x, y))
    return acc


def _inputs(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return (
        rng.standard_normal(_K, dtype=np.float32),
        rng.standard_normal(_K, dtype=np.float32),
    )


def _scale(a: np.ndarray, b: np.ndarray) -> float:
    """``sum |a_i b_i|``: the magnitude the accumulator actually carries.

    Dot-product error is normalised by this rather than by the result. A
    random length-256 dot product cancels down to a fraction of the terms
    that went into it, so dividing by the result would report the luck of
    that cancellation instead of the arithmetic under test.
    """
    return float(np.sum(np.abs(a.astype(np.float64) * b.astype(np.float64))))


def _err(got: float, want: float, scale: float) -> float:
    return abs(float(got) - float(want)) / scale


class TestTheSplit(unittest.TestCase):
    def test_big_and_small_are_representable_in_bf16(self):
        """Both halves must survive the narrowing the MMA will do to them."""
        rng = np.random.default_rng(_SEED)
        x = rng.standard_normal(4096, dtype=np.float32) * np.float32(1e3)
        big, small = _split(x)
        np.testing.assert_array_equal(big, _to_bf16(big))
        np.testing.assert_array_equal(small, _to_bf16(small))

    def test_the_split_roughly_doubles_the_significand(self):
        """``big + small`` is good to ~2^-16, against bf16's own ~2^-8.

        This is the whole source of the precision: two bf16 values carry
        about sixteen significand bits between them, two thirds of fp32's
        twenty-four.
        """
        rng = np.random.default_rng(_SEED + 1)
        x = rng.standard_normal(4096, dtype=np.float32)
        big, small = _split(x)
        rebuilt = (big + small).astype(np.float32)
        err = np.abs(rebuilt - x) / np.maximum(np.abs(x), np.float32(1e-30))
        self.assertLess(float(err.max()), 2.0**-16)
        # And the single-bf16 alternative really is ~256x worse, so the
        # bound above is not just a loose number that anything would pass.
        coarse = np.abs(big - x) / np.maximum(np.abs(x), np.float32(1e-30))
        self.assertGreater(float(coarse.max()), 2.0**-9)

    def test_the_residual_is_smaller_than_the_leading_term(self):
        rng = np.random.default_rng(_SEED + 2)
        x = rng.standard_normal(4096, dtype=np.float32)
        big, small = _split(x)
        nonzero = big != 0
        ratio = np.abs(small[nonzero]) / np.abs(big[nonzero])
        self.assertLess(float(ratio.max()), 2.0**-8)


class TestAccuracyAgainstFp32(unittest.TestCase):
    """The claim that justifies the feature, stated as a measurement."""

    def _errors(self) -> dict[str, float]:
        rng = np.random.default_rng(_SEED + 3)
        worst = {"bf16": 0.0, "tf32": 0.0, "bf16x3": 0.0, "fp32": 0.0}
        for _ in range(_TRIALS):
            a, b = _inputs(rng)
            want = _reference(a, b)
            scale = _scale(a, b)
            a_big, _ = _split(a)
            b_big, _ = _split(b)
            worst["bf16"] = max(
                worst["bf16"], _err(_fp32_dot(a_big, b_big), want, scale)
            )
            worst["tf32"] = max(
                worst["tf32"], _err(_fp32_dot(_to_tf32(a), _to_tf32(b)), want, scale)
            )
            worst["bf16x3"] = max(worst["bf16x3"], _err(_bf16x3_dot(a, b), want, scale))
            worst["fp32"] = max(worst["fp32"], _err(_fp32_dot(a, b), want, scale))
        return worst

    def test_three_terms_beat_one(self):
        """An emulated atom must be worth more than the BF16 MMA it is built from."""
        worst = self._errors()
        self.assertLess(worst["bf16x3"], worst["bf16"] / 100.0)

    def test_the_emulation_beats_real_tf32(self):
        """XF32 is not an approximation of TF32 -- it is strictly better.

        Real TF32 keeps ten significand bits; the split keeps about sixteen.
        A caller reaching for xf32 in place of TF32 therefore never loses
        accuracy by doing so, which is what makes the substitution safe.
        """
        worst = self._errors()
        self.assertLess(worst["bf16x3"], worst["tf32"] / 10.0)

    def test_the_emulation_lands_near_the_fp32_accumulation_noise_floor(self):
        """Within a small constant factor of an honest fp32 MMA's own error.

        An fp32 MMA is not exact either -- it rounds once per K term, and over
        K=256 that is already within about one order of magnitude of where the
        emulation lands. The gap being this narrow is the ceiling the recipe
        was designed to reach, and reaching it is why the fourth term is not
        worth issuing. The factor is deliberately loose: the point is the
        order of magnitude, not a bound that a reseed would flip.
        """
        worst = self._errors()
        self.assertLess(worst["bf16x3"], 2.0**-18)
        self.assertLess(worst["bf16x3"], 32.0 * max(worst["fp32"], 2.0**-24))


class TestTheRecipeShape(unittest.TestCase):
    def test_dropping_small_times_small_costs_nothing(self):
        """Why the recipe is three MMAs and not four.

        The fourth term is the product of two residuals, each already down at
        2^-8 of its operand, so it sits below where fp32 accumulation can
        carry it. Adding it back must not move the answer meaningfully.
        """
        rng = np.random.default_rng(_SEED + 5)
        for _ in range(_TRIALS):
            a, b = _inputs(rng)
            _, a_small = _split(a)
            _, b_small = _split(b)
            dropped = _fp32_dot(a_small, b_small)
            self.assertLess(abs(float(dropped)) / _scale(a, b), 2.0**-19)

    def test_the_term_order_is_load_bearing(self):
        """fp32 accumulation is not associative, so the order is part of the spec.

        If no permutation changed the answer, the order in the lowering would
        be arbitrary and free to drift. It is not: this shows a reordering is
        observable, which is what makes the fixed order in
        ``_expand_emulated_mma`` worth pinning.
        """
        rng = np.random.default_rng(_SEED + 6)
        differing = 0
        for _ in range(_TRIALS):
            a, b = _inputs(rng)
            if _bf16x3_dot(a, b) != _bf16x3_dot(a, b, order=(2, 1, 0)):
                differing += 1
        self.assertGreater(differing, 0, "term order turned out not to matter")

    def test_smallest_first_is_the_better_order(self):
        """The order the lowering uses: ``small*big``, ``big*small``, ``big*big``.

        Summing smallest-magnitude first is the standard way to limit
        cancellation loss in a fixed-precision accumulator, and it is what CK
        does. Over a sample it should not be worse than the reverse.
        """
        rng = np.random.default_rng(_SEED + 7)
        ours = reverse = 0.0
        for _ in range(_TRIALS):
            a, b = _inputs(rng)
            want, scale = _reference(a, b), _scale(a, b)
            ours += _err(_bf16x3_dot(a, b), want, scale)
            reverse += _err(_bf16x3_dot(a, b, order=(2, 1, 0)), want, scale)
        self.assertLessEqual(ours, reverse)


class TestTheNarrowingModel(unittest.TestCase):
    """The model's bf16 must be RNE, or every number above is off.

    ``v_cvt_pk_bf16_f32`` rounds to nearest even. CK's *scalar* fallback
    truncates, and the difference is a half-ULP bias on every input -- so this
    checks the model against the path that actually ships.
    """

    def test_exact_values_are_unchanged(self):
        exact = np.array([0.0, 1.0, -2.0, 0.5, 256.0, -1.5], dtype=np.float32)
        np.testing.assert_array_equal(_to_bf16(exact), exact)

    def test_ties_go_to_even(self):
        # 1.0 + 2^-9 is exactly halfway between two bf16 neighbours (1.0 and
        # 1.0 + 2^-8). RNE picks 1.0, whose stored significand is even;
        # truncation would also pick 1.0, so the next case is the one that
        # tells them apart.
        halfway_down = np.float32(1.0 + 2.0**-9)
        self.assertEqual(float(_to_bf16(halfway_down)), 1.0)
        # 1.0 + 3*2^-9 is halfway between 1.0 + 2^-8 and 1.0 + 2^-7. RNE
        # rounds *up* to the even neighbour; truncation would round down.
        halfway_up = np.float32(1.0 + 3.0 * 2.0**-9)
        self.assertEqual(float(_to_bf16(halfway_up)), 1.0 + 2.0**-7)

    def test_rounding_is_not_truncation(self):
        """Just above a bf16 grid point, RNE goes up and truncation stays."""
        x = np.float32(1.0 + 2.0**-8 + 2.0**-9 + 2.0**-12)
        self.assertGreater(float(_to_bf16(x)), float(x))

    def test_tf32_keeps_ten_significand_bits(self):
        """The TF32 comparison is only fair if the TF32 model is right."""
        self.assertEqual(float(_to_tf32(np.float32(1.0 + 2.0**-10))), 1.0 + 2.0**-10)
        self.assertEqual(float(_to_tf32(np.float32(1.0 + 2.0**-13))), 1.0)


if __name__ == "__main__":
    unittest.main()
