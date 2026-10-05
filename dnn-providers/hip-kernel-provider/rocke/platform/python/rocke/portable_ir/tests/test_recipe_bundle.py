# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Tests for the productization plumbing: CBOR codec + recipe bundle format.
#
# These guard the SHIPPING artifact path: a recipe (concrete or rolled) is
# encoded to CBOR, packed into a bundle keyed by (key, arch), and the C runtime
# decodes the SAME values back. If the codec drifts (type handling, ordering,
# float precision) or the bundle layout changes, these fail. The byte-identical
# equivalence to the C VM is covered by rocke.portable_ir.drivers.parity_matrix.
#
#   python3 -m unittest rocke.portable_ir.tests.test_recipe_bundle
import math
import unittest

from rocke.portable_ir.src.recipe_bundle import (
    BUNDLE_SCHEMA,
    bundle_lookup,
    build_bundle,
    cbor_decode,
    cbor_encode,
)


class TestCborCodec(unittest.TestCase):
    def _roundtrip(self, obj):
        self.assertEqual(cbor_decode(cbor_encode(obj)), obj)

    def test_scalars(self):
        for v in [
            None,
            True,
            False,
            0,
            1,
            23,
            24,
            255,
            256,
            65535,
            65536,
            2**31,
            2**32,
            -1,
            -24,
            -256,
            -(2**32),
        ]:
            self._roundtrip(v)

    def test_floats(self):
        for v in [0.0, -0.0, 1.5, -1e30, 3.141592653589793, 1e-12]:
            out = cbor_decode(cbor_encode(v))
            self.assertEqual(out, v)
        # large/odd float values survive exactly (float64)
        self.assertTrue(math.isclose(cbor_decode(cbor_encode(1e308)), 1e308))

    def test_bool_not_int(self):
        # bool is an int subclass in Python; must encode as CBOR true/false.
        self.assertEqual(cbor_encode(True), b"\xf5")
        self.assertEqual(cbor_encode(False), b"\xf4")
        self.assertIs(cbor_decode(cbor_encode(True)), True)

    def test_strings_unicode(self):
        for s in ["", "f32", "rocke_mini_attn_norm0_f32", "Δμ→λ", "a" * 300]:
            self._roundtrip(s)

    def test_containers(self):
        self._roundtrip([])
        self._roundtrip({})
        self._roundtrip([1, "a", True, None, [2, 3], {"k": "v"}])

    def test_recipe_like(self):
        recipe = {
            "schema": "rocke.recipe/v1",
            "kernel_name_fmt": "k_{D}",
            "spec": [{"name": "D", "kind": "int"}],
            "attrs": {"max_workgroup_size": {"t": "i", "v": 64}},
            "program": [
                {
                    "op": "param",
                    "name": "Q",
                    "type": {"kind": "ptr", "pointee": "f32", "space": "global"},
                    "bind": "Q",
                    "attrs": {"noalias": True, "align": 16},
                },
                {
                    "op": "emit",
                    "opcode": "tile.inline_asm",
                    "outs": [
                        {"bind": "r0", "type": "i32"},
                        {"bind": "r1", "type": "i32"},
                    ],
                },
                {
                    "op": "emit",
                    "opcode": "arith.constant",
                    "out": {"bind": "c", "type": "f32"},
                    "attrs": {"value": {"t": "f", "v": -1e30}},
                },
                {"op": "ret"},
            ],
        }
        self._roundtrip(recipe)


class TestBundle(unittest.TestCase):
    def test_build_and_lookup(self):
        entries = [
            {
                "key": "k_a",
                "arch": "gfx950",
                "recipe": {"schema": "rocke.recipe/v1", "x": 1},
            },
            {
                "key": "k_a",
                "arch": "gfx942",
                "recipe": {"schema": "rocke.recipe/v1", "x": 2},
            },
            {
                "key": "k_b",
                "arch": "gfx950",
                "recipe": {"schema": "rocke.recipe/v1", "x": 3},
            },
        ]
        bundle = cbor_decode(cbor_encode(build_bundle(entries)))
        self.assertEqual(bundle["schema"], BUNDLE_SCHEMA)
        self.assertEqual(bundle_lookup(bundle, "k_a", "gfx950")["x"], 1)
        self.assertEqual(bundle_lookup(bundle, "k_a", "gfx942")["x"], 2)
        self.assertEqual(bundle_lookup(bundle, "k_b")["x"], 3)
        self.assertIsNone(bundle_lookup(bundle, "k_a", "gfx_nope"))
        self.assertIsNone(bundle_lookup(bundle, "missing"))


class TestConcreteRecordBundle(unittest.TestCase):
    def test_record_concrete_bundle_roundtrip(self):
        from rocke.portable_ir.examples import mini_attn
        from rocke.portable_ir.src.recipe_bundle import record_concrete_bundle

        entries = record_concrete_bundle(
            [
                (lambda: mini_attn.build_mini_attn(0, "f32"), "gfx950"),
                (lambda: mini_attn.build_mini_attn(1, "f32"), "gfx950"),
            ]
        )
        self.assertEqual(
            [e["key"] for e in entries],
            ["rocke_mini_attn_norm0_f32", "rocke_mini_attn_norm1_f32"],
        )
        bundle = cbor_decode(cbor_encode(build_bundle(entries)))
        r = bundle_lookup(bundle, "rocke_mini_attn_norm1_f32", "gfx950")
        self.assertEqual(r["schema"], "rocke.recipe/v1")
        self.assertEqual(r, entries[1]["recipe"])  # CBOR preserved the recipe exactly


if __name__ == "__main__":
    unittest.main()
