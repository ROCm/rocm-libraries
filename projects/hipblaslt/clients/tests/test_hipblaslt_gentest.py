# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only tests: python -m unittest discover -s clients/tests -p 'test_hipblaslt_gentest.py'."""

import ctypes
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hipblaslt_gentest as gentest


class CharArrayBytesTests(unittest.TestCase):
    FIELD = ctypes.c_char * 32

    def test_value_that_fits_round_trips(self):
        data = gentest.char_array_bytes("gpu_arch", "9(0a|42|50)|1250", self.FIELD)
        self.assertEqual(data, b"9(0a|42|50)|1250")
        self.assertEqual(self.FIELD(*data).value, b"9(0a|42|50)|1250")

    def test_longest_terminated_value_is_accepted(self):
        value = "x" * 31
        self.assertEqual(gentest.char_array_bytes("gpu_arch", value, self.FIELD), value.encode())

    def test_value_that_fills_the_array_is_rejected(self):
        # ctypes would store all 32 bytes with no terminator.
        with self.assertRaises(SystemExit) as ctx:
            gentest.char_array_bytes("gpu_arch", "x" * 32, self.FIELD)
        self.assertIn("gpu_arch", str(ctx.exception.code))
        self.assertIn("at most 31", str(ctx.exception.code))

    def test_longer_value_is_rejected(self):
        with self.assertRaises(SystemExit):
            gentest.char_array_bytes("gpu_arch_exclude", "x" * 40, self.FIELD)


if __name__ == "__main__":
    unittest.main()
