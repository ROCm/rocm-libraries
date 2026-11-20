# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only tests: python -m unittest discover -s clients/tests -p 'test_hipblaslt_gentest.py'."""

import ctypes
import io
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


class UnknownTestArgumentTests(unittest.TestCase):
    def setUp(self):
        fields = [("name", ctypes.c_char * 64), ("K", ctypes.c_int64)]
        self.saved = dict(gentest.param)
        self.saved_args = dict(gentest.args)
        self.saved_testcases = set(gentest.testcases)
        gentest.param["Arguments"] = type("Arguments", (ctypes.Structure,), {"_fields_": fields})

    def tearDown(self):
        gentest.param.clear()
        gentest.param.update(self.saved)
        gentest.args.clear()
        gentest.args.update(self.saved_args)
        gentest.testcases.clear()
        gentest.testcases.update(self.saved_testcases)

    def test_unknown_key_names_the_test_and_key(self):
        # A YAML flow mapping written "K:1" yields the key "K:1", not K.
        with self.assertRaises(SystemExit) as ctx:
            gentest.write_test({"name": "grid_limit", "K:1": None})
        self.assertIn("grid_limit", str(ctx.exception.code))
        self.assertIn("K:1", str(ctx.exception.code))

    def test_profile_log_keys_are_ignored(self):
        out = io.BytesIO()
        gentest.args.update(outfile=out, signature_written=True)
        record = {"name": "replay", "K": 3}
        record.update({key: True for key in gentest.PROFILE_LOG_KEYS})
        gentest.write_test(record)
        self.assertEqual(out.getvalue(), bytes(gentest.param["Arguments"](b"replay", 3)))

    def test_unknown_key_is_rejected_alongside_profile_log_keys(self):
        with self.assertRaises(SystemExit) as ctx:
            gentest.write_test({"name": "replay", "K": 3, "call_count": 1, "swizzle_A": True})
        self.assertIn("swizzle_A", str(ctx.exception.code))
        self.assertNotIn("call_count", str(ctx.exception.code))


if __name__ == "__main__":
    unittest.main()
