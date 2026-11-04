# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import unittest

import hostnumerics as hn


class ScalarMetadataTests(unittest.TestCase):
    def test_binary16(self):
        info = hn.scalar_type_info(hn.ScalarType.Float16)
        self.assertEqual(info.name, "f16")
        self.assertEqual(info.category, hn.ScalarCategory.FloatingPoint)
        self.assertEqual(info.storage_bits, 16)
        self.assertEqual(
            (info.exponent_bits, info.mantissa_bits, info.exponent_bias), (5, 10, 15)
        )
        self.assertTrue(info.is_signed)
        self.assertTrue(info.supports_nan)
        self.assertTrue(info.supports_infinity)
        self.assertFalse(info.is_packed())
        with self.assertRaises(AttributeError):
            info.storage_bits = 32

    def test_packed_and_complex_storage(self):
        packed = hn.scalar_type_info(hn.ScalarType.Float6E3M2)
        self.assertEqual(packed.storage_bits, 6)
        self.assertTrue(packed.is_packed())
        self.assertFalse(packed.supports_nan)
        self.assertFalse(packed.supports_infinity)
        complex_type = hn.scalar_type_info(hn.ScalarType.ComplexFloat32)
        self.assertEqual(complex_type.category, hn.ScalarCategory.Complex)
        self.assertEqual(complex_type.storage_bits, 64)

    def test_unsigned_floating_point_formats(self):
        for scalar_type in (
            hn.ScalarType.E4M3,
            hn.ScalarType.E5M3,
            hn.ScalarType.E8M0,
            hn.ScalarType.E8M0Zero,
        ):
            with self.subTest(scalar_type=scalar_type):
                info = hn.scalar_type_info(scalar_type)
                self.assertEqual(info.category, hn.ScalarCategory.FloatingPoint)
                self.assertFalse(info.is_signed)
                self.assertEqual(info.storage_bits, 8)
        self.assertFalse(hn.scalar_type_info(hn.ScalarType.Boolean).is_signed)
        self.assertFalse(hn.scalar_type_info(hn.ScalarType.UInt8).is_signed)
        self.assertTrue(hn.scalar_type_info(hn.ScalarType.Int4).is_signed)
        self.assertTrue(hn.scalar_type_info(hn.ScalarType.ComplexFloat32).is_signed)
