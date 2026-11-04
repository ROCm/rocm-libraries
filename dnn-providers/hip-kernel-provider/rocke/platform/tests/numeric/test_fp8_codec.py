# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocke.numeric.fp8: the OCP e4m3 / e5m2 host codecs every fp8 harness shares.

CPU-only. The format facts are checked against the OCP definitions, and, where
the ``dev`` extra is installed, the decode tables against ``ml_dtypes`` as an
independent oracle.
"""

from __future__ import annotations

import unittest

import numpy as np

from rocke.numeric.fp8 import (
    bf8e5m2_decode_table,
    bf8e5m2_encode,
    fp8e4m3_decode_table,
    fp8e4m3_encode,
)


class TestDecodeTables(unittest.TestCase):
    def test_e4m3_format_facts(self):
        t = fp8e4m3_decode_table()
        self.assertEqual(t.shape, (256,))
        self.assertEqual(float(np.nanmax(t)), 448.0)
        self.assertEqual(float(t[0x38]), 1.0)  # s0 e0111 m000
        self.assertEqual(float(t[0x01]), 2.0**-9)  # smallest subnormal
        self.assertTrue(np.isnan(t[0x7F]) and np.isnan(t[0xFF]))
        self.assertEqual(int(np.isnan(t).sum()), 2)  # no infinities in e4m3fn

    def test_e5m2_format_facts(self):
        t = bf8e5m2_decode_table()
        self.assertEqual(float(np.nanmax(t)), 57344.0)
        self.assertEqual(float(t[0x3C]), 1.0)  # s0 e01111 m00
        self.assertEqual(float(t[0x01]), 2.0**-16)
        # e == 0x1F (inf and NaN) decodes to NaN, keeping it out of the encoder.
        self.assertEqual(int(np.isnan(t).sum()), 8)

    def test_against_ml_dtypes(self):
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
    def test_round_trip_every_finite_code(self):
        for table, encode in (
            (fp8e4m3_decode_table(), fp8e4m3_encode),
            (bf8e5m2_decode_table(), bf8e5m2_encode),
        ):
            vals = table[np.isfinite(table)]
            np.testing.assert_array_equal(table[encode(vals)], vals)

    def test_inexact_values_are_rejected(self):
        for encode, bad in ((fp8e4m3_encode, 1.0625), (bf8e5m2_encode, 1.125)):
            with self.assertRaises(ValueError):
                encode(np.array([bad], dtype=np.float32))

    def test_shape_and_dtype(self):
        x = np.zeros((3, 5), dtype=np.float32)
        for encode in (fp8e4m3_encode, bf8e5m2_encode):
            out = encode(x)
            self.assertEqual(out.shape, (3, 5))
            self.assertEqual(out.dtype, np.uint8)

    def test_the_manifest_runner_uses_this_codec(self):
        from rocke.instances.common.manifest_runner import gemm

        self.assertIs(gemm._F8_ENCODERS["fp8e4m3"], fp8e4m3_encode)
        self.assertIs(gemm._F8_ENCODERS["bf8e5m2"], bf8e5m2_encode)


if __name__ == "__main__":
    unittest.main()
