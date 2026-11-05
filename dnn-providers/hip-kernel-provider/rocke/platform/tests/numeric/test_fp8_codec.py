# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocke.numeric.fp8: the OCP e4m3 / e5m2 host codecs every fp8 harness shares.

The one home for the codec's own facts: the decode tables against the format
definitions (and ``ml_dtypes`` where the ``dev`` extra is installed), the exact
encoders, and the rounding encoder. How a harness uses the codec is tested with
that harness (test_universal_gemm_verify.py, test_fp8_trait_ab.py). CPU-only.
"""

from __future__ import annotations

import unittest

import numpy as np

from rocke.numeric.fp8 import (
    bf8e5m2_decode_table,
    bf8e5m2_encode,
    fp8e4m3_decode_table,
    fp8e4m3_encode,
    fp8e4m3_round,
)


class TestE4m3Table(unittest.TestCase):
    def test_known_codes_decode_to_their_format_values(self):
        # Anchors spanning the encoding: both zeros, the implicit-leading-one
        # boundary, a mantissa step, and the largest finite magnitude. Hand-
        # computed from s.eeee.mmm / bias 7, not read back out of the table.
        table = fp8e4m3_decode_table()
        for code, want in (
            (0x00, 0.0),
            (0x80, -0.0),
            (0x38, 1.0),
            (0x3C, 1.5),
            (0x40, 2.0),
            (0xB8, -1.0),
            (0x7E, 448.0),
            (0xFE, -448.0),
        ):
            with self.subTest(code=hex(code)):
                self.assertEqual(float(table[code]), want)

    def test_subnormals_use_the_e_eq_0_rule(self):
        # e == 0 drops the implicit leading one and pins the exponent at 2^-6,
        # so the smallest step is 2^-6/8 == 2^-9 and the steps are uniform.
        table = fp8e4m3_decode_table()
        self.assertEqual(float(table[0x01]), 2.0**-9)
        self.assertEqual(float(table[0x07]), 7 * 2.0**-9)
        # Subnormal-to-normal must be continuous: 0x08 is the first normal.
        self.assertEqual(float(table[0x08]), 8 * 2.0**-9)

    def test_nan_only_at_the_two_all_ones_codes(self):
        # e4m3 spends no codes on infinities, so 0x7f/0xff are the only NaNs
        # and every other code is finite -- which is what lets the encoders
        # treat the finite set as the whole representable set.
        table = fp8e4m3_decode_table()
        self.assertEqual(set(np.flatnonzero(np.isnan(table)).tolist()), {0x7F, 0xFF})
        self.assertFalse(np.isinf(table).any())


class TestE5m2Table(unittest.TestCase):
    def test_known_codes_decode_to_their_format_values(self):
        t = bf8e5m2_decode_table()
        for code, want in (
            (0x3C, 1.0),
            (0xBC, -1.0),
            (0x45, 5.0),
            (0x7B, 57344.0),  # largest finite
            (0x04, 2.0**-14),  # smallest normal
            (0x01, 2.0**-16),  # smallest subnormal
        ):
            with self.subTest(code=hex(code)):
                self.assertEqual(float(t[code]), want)

    def test_inf_and_nan_codes_decode_to_nan(self):
        # e == 0x1F is inf / NaN; decoding them to NaN keeps them out of the
        # encoder's table, so none of those codes is ever produced.
        t = bf8e5m2_decode_table()
        self.assertTrue(np.isnan(t[0x7C:0x80]).all())
        self.assertTrue(np.isnan(t[0xFC:0x100]).all())
        self.assertEqual(int(np.isnan(t).sum()), 8)


class TestAgainstMlDtypes(unittest.TestCase):
    def test_tables_match_ml_dtypes(self):
        ml_dtypes = __import__("pytest").importorskip("ml_dtypes")
        codes = np.arange(256, dtype=np.uint8)
        for table, dt in (
            (fp8e4m3_decode_table(), ml_dtypes.float8_e4m3fn),
            (bf8e5m2_decode_table(), ml_dtypes.float8_e5m2),
        ):
            ref = codes.view(dt).astype(np.float32)
            finite = np.isfinite(ref)
            np.testing.assert_array_equal(table[finite], ref[finite])


class TestExactEncoders(unittest.TestCase):
    def test_every_finite_code_round_trips(self):
        # -0.0 encodes to +0.0's code: they are numerically equal, so compare
        # the decoded values rather than the codes.
        for table, encode in (
            (fp8e4m3_decode_table(), fp8e4m3_encode),
            (bf8e5m2_decode_table(), bf8e5m2_encode),
        ):
            vals = table[np.isfinite(table)]
            np.testing.assert_array_equal(table[encode(vals)], vals)

    def test_inexact_values_are_refused_and_named(self):
        # A silent round here would demote an exact compare to a tolerance
        # check without changing any output.
        for bad in (0.3, 1.1, 2.0**-10, 449.0, 1e30, float("nan")):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "fp8e4m3 encode: .*element 1"):
                    fp8e4m3_encode(np.array([1.0, bad], dtype=np.float32))

    def test_each_format_refuses_what_it_cannot_hold(self):
        # 1.125 needs three significand bits: exact in e4m3, not in e5m2.
        x = np.array([1.125], dtype=np.float32)
        self.assertEqual(int(fp8e4m3_encode(x)[0]), 0x39)
        with self.assertRaisesRegex(ValueError, "bf8e5m2 encode: 1.125"):
            bf8e5m2_encode(x)

    def test_shape_and_dtype(self):
        x = np.zeros((3, 5), dtype=np.float32)
        for encode in (fp8e4m3_encode, bf8e5m2_encode):
            out = encode(x)
            self.assertEqual(out.shape, (3, 5))
            self.assertEqual(out.dtype, np.uint8)


class TestE4m3Round(unittest.TestCase):
    def test_every_finite_value_round_trips(self):
        table = fp8e4m3_decode_table()
        codes = np.flatnonzero(np.isfinite(table) & (np.arange(256) != 0x80))
        np.testing.assert_array_equal(
            fp8e4m3_round(table[codes]), codes.astype(np.uint8)
        )

    def test_ties_go_to_the_even_code(self):
        # 1.0 = 0x38, 1.125 = 0x39, 1.25 = 0x3A: midpoints go to 0x38 and 0x3A.
        x = np.array([1.0625, 1.1875, -1.0625], dtype=np.float32)
        self.assertEqual(list(fp8e4m3_round(x)), [0x38, 0x3A, 0xB8])

    def test_rounds_to_nearest_and_saturates_at_the_largest_finite(self):
        x = np.array([1.05, 1.08, 1000.0, -1000.0, 0.0, -0.0], dtype=np.float32)
        self.assertEqual(list(fp8e4m3_round(x)), [0x38, 0x39, 0x7E, 0xFE, 0x00, 0x00])

    def test_negative_values_that_round_to_zero_give_plus_zero(self):
        # The table holds one zero (+0, 0x00), as for the exact encoders.
        x = np.array([-(2.0**-11), -0.0, 2.0**-11], dtype=np.float32)
        self.assertEqual(list(fp8e4m3_round(x)), [0x00, 0x00, 0x00])

    def test_nan_is_rejected_and_its_element_named(self):
        with self.assertRaisesRegex(ValueError, "NaN at element 2"):
            fp8e4m3_round(np.array([1.0, 2.0, np.nan, np.nan], dtype=np.float32))

    def test_shape_and_dtype(self):
        out = fp8e4m3_round(np.zeros((2, 3), dtype=np.float32))
        self.assertEqual((out.shape, out.dtype), ((2, 3), np.uint8))


if __name__ == "__main__":
    unittest.main()
