# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the headless rocGDB physical-register decoder."""

from __future__ import annotations

import importlib.util
import json
import math
import unittest
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools/rocke_debug.py"
_SPEC = importlib.util.spec_from_file_location("rocke_debug", _TOOL)
rocke_debug = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rocke_debug)


class TestF32(unittest.TestCase):
    def test_normal_signed_zero_infinity_and_nan(self):
        cases = (
            (0x3F800000, "normal", 1.0, "1.0"),
            (0x80000000, "zero", -0.0, "-0"),
            (0x7F800000, "infinity", None, "inf"),
            (0xFFC00001, "nan", None, "nan"),
        )
        for raw, classification, value, text in cases:
            with self.subTest(raw=hex(raw)):
                element = rocke_debug.decode_word(raw, "f32")[0]
                self.assertEqual(element["class"], classification)
                self.assertEqual(element["value"], value)
                self.assertEqual(element["value_text"], text)
                self.assertEqual(element["raw_bits"], raw)


class TestPacked16(unittest.TestCase):
    def test_f16_is_little_endian_and_preserves_special_values(self):
        elements = rocke_debug.decode_word(0xFC007C00, "f16x2")
        self.assertEqual(
            [element["raw_hex"] for element in elements], ["0x7c00", "0xfc00"]
        )
        self.assertEqual(
            [element["value_text"] for element in elements], ["inf", "-inf"]
        )

    def test_f16_subnormal(self):
        element = rocke_debug.decode_word(0x00000001, "f16x2")[0]
        self.assertEqual(element["class"], "subnormal")
        self.assertEqual(element["value"], math.ldexp(1.0, -24))

    def test_bf16_is_little_endian(self):
        elements = rocke_debug.decode_word(0xBF803F80, "bf16x2")
        self.assertEqual([element["value"] for element in elements], [1.0, -1.0])
        self.assertEqual([element["index"] for element in elements], [0, 1])


class TestPacked8(unittest.TestCase):
    def test_fp8_e4m3_finite_only_values(self):
        elements = rocke_debug.decode_word(0x7F7EB838, "fp8e4m3x4")
        self.assertEqual(
            [element["value"] for element in elements[:3]], [1.0, -1.0, 448.0]
        )
        self.assertEqual(elements[3]["class"], "nan")
        self.assertIsNone(elements[3]["value"])

    def test_fp8_preserves_negative_zero(self):
        element = rocke_debug.decode_word(0x00000080, "fp8e4m3x4")[0]
        self.assertEqual(element["class"], "zero")
        self.assertEqual(element["sign"], -1)
        self.assertEqual(element["value_text"], "-0")

    def test_fp8_fnuz_uses_bias_eight_and_0x80_nan(self):
        one = rocke_debug.decode_word(0x00000040, "fp8e4m3x4", "fnuz")[0]
        nan = rocke_debug.decode_word(0x00000080, "fp8e4m3x4", "fnuz")[0]
        largest = rocke_debug.decode_word(0x0000007F, "fp8e4m3x4", "fnuz")[0]
        self.assertEqual(one["value"], 1.0)
        self.assertEqual(nan["class"], "nan")
        self.assertEqual(largest["value"], 240.0)

    def test_bf8_e5m2_special_values(self):
        elements = rocke_debug.decode_word(0x7DFC7C3C, "bf8e5m2x4")
        self.assertEqual(elements[0]["value"], 1.0)
        self.assertEqual(elements[1]["value_text"], "inf")
        self.assertEqual(elements[2]["value_text"], "-inf")
        self.assertEqual(elements[3]["class"], "nan")

    def test_bf8_fnuz_has_no_infinity(self):
        elements = rocke_debug.decode_word(0x0000807C, "bf8e5m2x4", "fnuz")
        self.assertEqual(elements[0]["class"], "normal")
        self.assertEqual(elements[0]["value"], 32768.0)
        self.assertEqual(elements[1]["class"], "nan")


class TestRecords(unittest.TestCase):
    def test_lane_identity_and_exec_state(self):
        records = rocke_debug.decode_register(
            "$v40", [0x3F800000, 0x40000000, 0x40400000], "f32", exec_mask=0b101
        )
        self.assertEqual([record["lane"] for record in records], [0, 1, 2])
        self.assertEqual([record["active"] for record in records], [True, False, True])
        self.assertTrue(
            all(record["schema"] == "rocke-register-v1" for record in records)
        )

    def test_float8_format_is_part_of_record_provenance(self):
        record = rocke_debug.decode_register(
            "$v2", [0x40], "fp8e4m3x4", exec_mask=1, float8_format="fnuz"
        )[0]
        self.assertEqual(record["float8_format"], "fnuz")

    def test_unknown_exec_state_is_explicit(self):
        record = rocke_debug.decode_register("$s4", [0], "f32")[0]
        self.assertIsNone(record["active"])

    def test_jsonl_is_strict_and_round_trips_special_values(self):
        records = rocke_debug.decode_register("$v1", [0x7FC00000], "f32", exec_mask=1)
        output = rocke_debug.records_jsonl(records)
        self.assertNotIn("NaN", output)
        decoded = json.loads(output)
        self.assertIsNone(decoded["elements"][0]["value"])
        self.assertEqual(decoded["elements"][0]["value_text"], "nan")

    def test_human_output_has_lane_raw_dtype_and_values(self):
        record = rocke_debug.decode_register("$v7", [0x3F800000], "f32", exec_mask=1)
        output = rocke_debug.records_human(record)
        self.assertIn("$v7", output)
        self.assertIn("0x3f800000", output)
        self.assertIn("f32", output)
        self.assertIn("[1.0]", output)

    def test_rejects_unsupported_dtype_and_out_of_range_word(self):
        with self.assertRaises(ValueError):
            rocke_debug.decode_word(0, "f64")
        with self.assertRaises(ValueError):
            rocke_debug.decode_word(1 << 32, "f32")


if __name__ == "__main__":
    unittest.main()
