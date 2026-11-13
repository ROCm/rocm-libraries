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
  term a plain BF16 MMA would give, and closer than real TF32 *on raw fp32
  operands*: the emulation tracks the payload it is given, where TF32 first
  rounds most of it away. On operands already rounded to TF32 precision the
  comparison reverses -- TF32's product is exact there and the emulation is off
  by the omitted term -- so this is a statement about raw payloads, not a
  guarantee of never doing worse;
* dropping ``small*small`` costs little on typical data, which is why the
  recipe is three MMAs and not four; its worst case is 2^-16 of a product,
  reached at an operand exactly halfway between two bf16 values;
* the split stays finite, and the product with it, for every finite payload --
  including the top 2^-9 of the range where bf16 rounding overflows -- and an
  infinity or NaN in an operand makes every output it feeds non-finite;
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
    # exact ties go to even, then drop the low 16 bits. The carry runs into the
    # exponent, so a payload at or above 0x7f7f8000 comes out as infinity --
    # which is what the hardware narrowing does, and is the overflow the
    # lowering caps.
    bumped = (
        u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    ) & np.uint32(0xFFFF0000)
    # A NaN keeps being one: the add would carry a payload held only in the low
    # bits into the exponent and hand back an infinity.
    return np.where(np.isnan(x), np.float32("nan"), bumped.view(np.float32))


def _to_tf32(x: np.ndarray) -> np.ndarray:
    """fp32 -> real TF32 (10 explicit significand bits) -> fp32, RNE."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    bumped = (
        u + np.uint32(0x0FFF) + ((u >> np.uint32(13)) & np.uint32(1))
    ) & np.uint32(0xFFFFE000)
    return bumped.view(np.float32)


def _cap(big: np.ndarray) -> np.ndarray:
    """Hold ``big`` at the largest finite bf16, keeping its sign.

    The lowering's saturation, on the same bit patterns it uses: an unsigned
    compare of each pattern against ``0x7f7f`` with the value's own sign
    attached, which catches +-inf and every NaN and nothing finite.
    """
    bits = np.ascontiguousarray(big, dtype=np.float32).view(np.uint32) >> np.uint32(16)
    cap = (bits & np.uint32(0x8000)) | np.uint32(0x7F7F)
    return (np.where(bits > cap, cap, bits).astype(np.uint32) << np.uint32(16)).view(
        np.float32
    )


def _split_uncapped(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The split as first written, with no cap. Kept to show what the cap is for."""
    x = np.asarray(x, dtype=np.float32)
    big = _to_bf16(x)
    with np.errstate(invalid="ignore", over="ignore"):
        small = _to_bf16((x - big).astype(np.float32))
    return big, small


def _split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Dekker-style split the lowering emits: ``big``, then the residual.

    ``big`` is the RNE rounding held at the largest finite bf16; the residual is
    taken from ``x`` itself, so whatever the cap threw away -- or, for a
    non-finite ``x``, the non-finite value -- is left in ``small``.
    """
    x = np.asarray(x, dtype=np.float32)
    big = _cap(_to_bf16(x))
    with np.errstate(invalid="ignore", over="ignore"):
        small = _to_bf16((x - big).astype(np.float32))
    return big, small


def _fp32_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sequential fp32 accumulation -- the model's notion of one MMA."""
    acc = np.float32(0.0)
    # Infinities and NaNs are legitimate operands here, and overflow is an
    # answer rather than an accident; IEEE arithmetic is the model, so the
    # warnings numpy attaches to it say nothing.
    with np.errstate(all="ignore"):
        for i in range(a.shape[0]):
            acc = np.float32(acc + np.float32(a[i] * b[i]))
    return acc


def _reference(a: np.ndarray, b: np.ndarray) -> np.float64:
    """Ground truth, in fp64. Not reachable by any of the models."""
    return np.dot(a.astype(np.float64), b.astype(np.float64))


def _bf16x3_dot(
    a: np.ndarray, b: np.ndarray, *, order=(0, 1, 2), split=None
) -> np.float32:
    """The emulation: three BF16 MMAs threading one fp32 accumulator.

    ``order`` permutes the three terms so a test can show the order matters;
    ``split`` swaps in another split (the default is the lowering's).
    """
    split = split or _split
    a_big, a_small = split(a)
    b_big, b_small = split(b)
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


def _f32(bits: int) -> np.float32:
    """The fp32 value with this bit pattern -- how a TF32 payload is written down."""
    return np.array([bits], dtype=np.uint32).view(np.float32)[0]


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

    def test_the_emulation_beats_real_tf32_on_raw_fp32_operands(self):
        """On raw fp32 payloads XF32 is not an approximation of TF32 -- it is better.

        Real TF32 keeps ten significand bits of an operand; the split keeps about
        sixteen. Measured here on operands that carry all twenty-four, so the
        emulation tracks the payload where TF32 first rounds most of it away.

        That is the whole claim. On operands already rounded to TF32 precision
        the native atom is exact and this recipe is off by the omitted
        ``small*small`` term -- up to 2^-16 per product (see
        ``TestTheOmittedTerm``) -- so a caller gains accuracy by choosing it
        for raw operands and can lose a little by choosing it for prepared ones.
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
    def test_dropping_small_times_small_costs_little_on_typical_data(self):
        """Why the recipe is three MMAs and not four.

        The fourth term is the product of two residuals, each at most 2^-8 of its
        operand and typically far less, so on ordinary data it sits below where
        fp32 accumulation can carry it. This is the typical case, not the bound:
        ``TestTheOmittedTerm`` is the worst case, which is 2^-16 of a product.
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


class TestTheOmittedTerm(unittest.TestCase):
    """The recipe drops ``small*small``; this is what that costs, worst case.

    A residual is at most half a bf16 ulp -- 2^-8 of its operand -- so the term
    is below 2^-16 of the product. It gets almost all the way there: an operand
    exactly halfway between two bf16 values, the one nearest a power of two,
    leaves ties-to-even with a residual of exactly half an ulp, and against the
    product that is ``2^-16 / (1 + 2^-8)^2``, about 1.514e-5 -- 99% of the bound,
    and eight times the 2^-18 this was once stated as.
    """

    _BOUND = 2.0**-16
    _MIDPOINT = 2.0**-16 / (1.0 + 2.0**-8) ** 2

    def test_a_midpoint_operand_reaches_the_bound(self):
        x = np.array([1.0 + 2.0**-8], dtype=np.float32)  # halfway: 1.0 | 1 + 2^-7
        big, small = _split(x)
        self.assertEqual((float(big[0]), float(small[0])), (1.0, 2.0**-8))
        # Exact, and exactly representable: 1 + 2^-7 + 2^-16.
        exact = float(x[0]) ** 2
        self.assertEqual(exact, 1.0078277587890625)
        got = float(_bf16x3_dot(x, x))
        # The three terms give 1 + 2^-8 + 2^-8; the omitted 2^-8 * 2^-8 is the
        # entire difference.
        self.assertEqual(got, 1.0078125)
        self.assertEqual(exact - got, 2.0**-16)
        rel = (exact - got) / exact
        self.assertAlmostEqual(rel, 1.514e-5, delta=1e-8)
        self.assertAlmostEqual(rel, self._MIDPOINT, delta=1e-12)
        self.assertGreater(rel, 2.0**-18)  # the bound this was once stated as
        self.assertLess(rel, self._BOUND)
        self.assertGreater(rel, 0.99 * self._BOUND)

    def test_no_operand_pair_exceeds_the_bound(self):
        rng = np.random.default_rng(_SEED + 8)
        random = rng.standard_normal(100_000, dtype=np.float32)
        # Every exact midpoint on the bf16 grid in [1, 2), both signs.
        steps = np.arange(0x3F800000, 0x40000000, 0x10000, dtype=np.uint32)
        mid = (steps + np.uint32(0x8000)).view(np.float32)
        # A zero has no residual and no ratio; drop the rare exact one.
        random = random[random != 0]
        for name, payload in (
            ("random", random),
            ("midpoint", mid),
            ("-midpoint", -mid),
        ):
            with self.subTest(name):
                _, small = _split(payload)
                ratio = np.abs(small.astype(np.float64)) / np.abs(
                    payload.astype(np.float64)
                )
                self.assertLess(float(ratio.max()) ** 2, self._BOUND)
        # And it is nearly attained: the midpoint nearest a power of two.
        _, small = _split(mid)
        ratio = np.abs(small.astype(np.float64)) / np.abs(mid.astype(np.float64))
        self.assertGreater(float(ratio.max()) ** 2, 0.99 * self._BOUND)


class TestTopBinadeSaturation(unittest.TestCase):
    """bf16 rounding overflows for magnitudes of 0x7f7f8000 or more; the cap is what
    keeps those payloads finite.

    That is the top 2^-9 of the finite fp32 range, and it holds TF32's own largest
    finite value, 0x7f7fe000 -- so these are legitimate operands, not noise.
    Uncapped, ``big`` rounds to +inf and ``small`` becomes the opposite infinity,
    and the ``inf * 0`` that falls out in the cross terms turns a finite product
    into NaN.
    """

    _PAYLOADS = (0x7F7F8000, 0x7F7FA000, 0x7F7FC000, 0x7F7FE000, 0x7F7FFFFF)
    _BF16_MAX = _f32(0x7F7F0000)

    def test_rounding_overflows_exactly_at_0x7f7f8000(self):
        self.assertTrue(np.isfinite(_to_bf16(_f32(0x7F7F7FFF))))
        self.assertTrue(np.isposinf(_to_bf16(_f32(0x7F7F8000))))
        self.assertTrue(np.isneginf(_to_bf16(_f32(0xFF7F8000))))

    def test_without_the_cap_a_finite_payload_splits_into_infinities(self):
        """The defect, reproduced on the model."""
        for bits in self._PAYLOADS:
            with self.subTest(hex(bits)):
                big, small = _split_uncapped(np.array([_f32(bits)]))
                self.assertTrue(np.isposinf(big[0]))
                self.assertTrue(np.isneginf(small[0]))

    def test_with_the_cap_the_split_is_finite_and_recovers_the_payload(self):
        for sign in (0, 0x80000000):
            for bits in self._PAYLOADS:
                with self.subTest(hex(bits | sign)):
                    x = np.array([_f32(bits | sign)])
                    big, small = _split(x)
                    self.assertEqual(abs(float(big[0])), float(self._BF16_MAX))
                    self.assertEqual(np.sign(big[0]), np.sign(x[0]))
                    self.assertTrue(np.isfinite(small[0]))
                    rebuilt = float(big[0]) + float(small[0])
                    self.assertLessEqual(
                        abs(rebuilt - float(x[0])) / abs(float(x[0])), 2.0**-16
                    )

    def test_payloads_below_the_threshold_split_exactly_as_before(self):
        """The cap must not move a single bit of anything that never needed it."""
        rng = np.random.default_rng(_SEED + 9)
        bits = rng.integers(0, 2**32, size=200_000, dtype=np.uint64).astype(np.uint32)
        keep = np.isfinite(bits.view(np.float32)) & (
            (bits & np.uint32(0x7FFFFFFF)) < np.uint32(0x7F7F8000)
        )
        x = bits[keep].view(np.float32)
        for capped, uncapped in zip(_split(x), _split_uncapped(x)):
            np.testing.assert_array_equal(
                capped.view(np.uint32), uncapped.view(np.uint32)
            )

    def test_the_boundary_product_stays_finite(self):
        """TF32's largest finite value against the smallest normal: exactly 3.998046875.

        The exact product is finite, so the result must be. The capped split
        lands on it exactly; the uncapped one gives NaN."""
        a = np.array([_f32(0x7F7FE000)])
        b = np.array([_f32(0x00800000)])
        self.assertEqual(float(a[0]) * float(b[0]), 3.998046875)
        self.assertEqual(float(_bf16x3_dot(a, b)), 3.998046875)
        self.assertTrue(np.isnan(_bf16x3_dot(a, b, split=_split_uncapped)))

    def test_a_product_that_overflows_is_an_infinity_not_a_nan(self):
        x = np.array([_f32(0x7F7FE000)])
        self.assertTrue(np.isposinf(_bf16x3_dot(x, x)))
        self.assertTrue(np.isneginf(_bf16x3_dot(x, -x)))


class TestSubnormalPayloads(unittest.TestCase):
    """bf16 has fp32's exponent range but only 7 mantissa bits, so its smallest
    positive value is 2^-133 and the split of a subnormal payload has an absolute
    error floor instead of a relative one.

    ``big`` lands on the bf16 grid, whose spacing down there is 2^-133, so it is
    within 2^-134 of the payload, and ``small`` can only be 0 or one grid step; the
    split is therefore good to an absolute 2^-134 per operand, and a payload no
    larger than 2^-134 contributes nothing at all.
    """

    def test_a_subnormal_payload_splits_to_an_absolute_error_of_two_to_the_minus_134(
        self,
    ):
        rng = np.random.default_rng(_SEED + 11)
        bits = rng.integers(1, 0x00800000, size=100_000, dtype=np.uint64).astype(
            np.uint32
        )
        x = bits.view(np.float32)
        big, small = _split(x)
        error = np.abs((big.astype(np.float64) + small) - x.astype(np.float64))
        self.assertLessEqual(float(error.max()), 2.0**-134)

    def test_a_payload_no_larger_than_two_to_the_minus_134_contributes_nothing(self):
        partner = np.array([2.0**100], dtype=np.float32)
        for bits in (0x00000001, 0x00001000, 0x00008000):
            with self.subTest(hex(bits)):
                x = np.array([_f32(bits)])
                big, small = _split(x)
                self.assertEqual((float(big[0]), float(small[0])), (0.0, 0.0))
                self.assertEqual(float(_bf16x3_dot(x, partner)), 0.0)
                # ... although the exact product is not zero.
                self.assertGreater(float(x[0]) * float(partner[0]), 0.0)
        # One step above that, the payload is represented.
        big, small = _split(np.array([_f32(0x00008001)]))
        self.assertEqual(float(big[0]), 2.0**-133)

    def test_a_subnormal_the_bf16_grid_holds_exactly_is_exact(self):
        """2^-131 and 2^-127 are on the grid, so nothing is lost."""
        for bits in (0x00040000, 0x00200000, 0x80200000):
            with self.subTest(hex(bits)):
                x = np.array([_f32(bits)])
                big, small = _split(x)
                self.assertEqual(float(big[0]) + float(small[0]), float(x[0]))


class TestNonFinitePayloads(unittest.TestCase):
    """An infinity or NaN in an operand makes every output it feeds non-finite.

    The cap makes ``big`` finite for these as well, and ``small`` -- taken from
    the payload -- keeps the non-finite value, so ``small*big``, the first term
    issued, is non-finite whatever the other operand holds.
    """

    _NON_FINITE = (0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001)

    def test_the_leading_part_is_finite_and_the_residual_is_not(self):
        for bits in self._NON_FINITE:
            with self.subTest(hex(bits)):
                big, small = _split(np.array([_f32(bits)]))
                self.assertTrue(np.isfinite(big[0]))
                self.assertFalse(np.isfinite(small[0]))

    def test_an_infinity_against_an_ordinary_partner_is_that_infinity(self):
        one = np.array([1.0], dtype=np.float32)
        for bits, sign in ((0x7F800000, 1.0), (0xFF800000, -1.0)):
            with self.subTest(hex(bits)):
                x = np.array([_f32(bits)])
                self.assertEqual(float(_bf16x3_dot(x, one)), sign * np.inf)
                self.assertEqual(float(_bf16x3_dot(one, x)), sign * np.inf)
                # The partner's sign decides which infinity.
                self.assertEqual(float(_bf16x3_dot(x, -one)), -sign * np.inf)

    def test_infinity_times_infinity_is_an_infinity(self):
        inf = np.array([_f32(0x7F800000)])
        self.assertTrue(np.isposinf(_bf16x3_dot(inf, inf)))
        self.assertTrue(np.isneginf(_bf16x3_dot(inf, -inf)))

    def test_infinity_times_zero_is_nan(self):
        inf, zero = np.array([_f32(0x7F800000)]), np.array([0.0], dtype=np.float32)
        self.assertTrue(np.isnan(_bf16x3_dot(inf, zero)))
        self.assertTrue(np.isnan(_bf16x3_dot(zero, inf)))

    def test_nan_in_either_operand_gives_nan(self):
        nan, one = np.array([_f32(0x7FC00000)]), np.array([1.0], dtype=np.float32)
        self.assertTrue(np.isnan(_bf16x3_dot(nan, one)))
        self.assertTrue(np.isnan(_bf16x3_dot(one, nan)))

    def test_a_large_partner_can_turn_an_infinity_into_a_nan_but_never_a_finite_number(
        self,
    ):
        """The limit of "an infinity stays an infinity".

        Against a partner of 512 or more the partner's own residual can reach
        the cap's magnitude and overflow to an infinity of the *opposite* sign,
        and the two infinities sum to NaN. IEEE would give the infinity; what
        is guaranteed is that the output is not finite.
        """
        inf = np.array([_f32(0x7F800000)])
        for partner in (700.5, -700.3, 1000.7):
            with self.subTest(partner):
                got = _bf16x3_dot(inf, np.array([partner], dtype=np.float32))
                self.assertFalse(np.isfinite(got))

    def test_only_the_dot_products_that_consume_a_non_finite_payload_are_affected(self):
        rng = np.random.default_rng(_SEED + 10)
        a, b = _inputs(rng)
        poisoned = a.copy()
        poisoned[3] = np.float32("inf")
        self.assertFalse(np.isfinite(_bf16x3_dot(poisoned, b)))
        # A dot product that does not contain element 3 never sees it.
        keep = np.arange(a.shape[0]) != 3
        self.assertEqual(
            float(_bf16x3_dot(poisoned[keep], b[keep])),
            float(_bf16x3_dot(a[keep], b[keep])),
        )


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

    def test_a_nan_stays_a_nan(self):
        """The integer carry would turn a NaN held in the low bits into infinity."""
        self.assertTrue(np.isnan(_to_bf16(_f32(0x7F800001))))
        self.assertTrue(np.isnan(_to_bf16(_f32(0xFFC00000))))

    def test_tf32_keeps_ten_significand_bits(self):
        """The TF32 comparison is only fair if the TF32 model is right."""
        self.assertEqual(float(_to_tf32(np.float32(1.0 + 2.0**-10))), 1.0 + 2.0**-10)
        self.assertEqual(float(_to_tf32(np.float32(1.0 + 2.0**-13))), 1.0)


if __name__ == "__main__":
    unittest.main()
