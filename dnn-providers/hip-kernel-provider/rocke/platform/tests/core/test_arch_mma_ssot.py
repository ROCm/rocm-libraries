# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SSOT guards for the bare-op_id MMA result-dtype lookup.

``IRBuilder.mma`` uses ``target._op_id_d_dtype()`` to size a ``tile.mma`` result
vector's accumulator element without an ``ArchTarget`` in hand. These tests pin
the first-wins / raise-on-drift contract of that lookup so it stays deterministic
across the arches that list a given op_id.
"""

from __future__ import annotations

import unittest
from unittest import mock

from rocke.core.arch.target import (
    MmaCatalog,
    _FragInfo,
    _build_mma_op,
    _load_specs,
    _op_id_d_dtype,
    _validate_specs_doc,
    normalize_dtype,
)


class TestOpIdDDtype(unittest.TestCase):
    def test_matches_catalog_first_hit(self):
        # Every op_id in the catalog resolves to its normalized result dtype,
        # taking the first arch that lists it (dict preserves catalog order).
        expected: dict = {}
        for row in _load_specs().values():
            for o in row["mma"]:
                expected.setdefault(o["op_id"], normalize_dtype(o["d"]))
        self.assertEqual(_op_id_d_dtype(), expected)

    def test_d_dtype_invariant_across_arches(self):
        # The whole premise of the bare-op_id lookup: an op_id's result dtype
        # is invariant across the arches that list it, so building the map must not
        # raise on the real catalog. (The raise path is exercised below.)
        try:
            _op_id_d_dtype()
        except ValueError as exc:  # pragma: no cover - only hit on real drift
            self.fail(f"_op_id_d_dtype() raised on the shipped catalog: {exc}")

    def test_raises_on_cross_arch_disagreement(self):
        specs = _load_specs()
        # Find an op_id and clone its row into a fake arch with a different D dtype.
        sample = next(o for row in specs.values() for o in row["mma"])
        original_d = normalize_dtype(sample["d"])
        other_d = "i32" if original_d != "i32" else "f32"
        clash = dict(sample)
        clash["d"] = other_d
        drifted = dict(specs)
        drifted["_synthetic_drift"] = {"mma": [clash]}

        _op_id_d_dtype.cache_clear()
        try:
            with mock.patch("rocke.core.arch.target._load_specs", return_value=drifted):
                with self.assertRaises(ValueError):
                    _op_id_d_dtype()
        finally:
            _op_id_d_dtype.cache_clear()


class TestFourRoleSchema(unittest.TestCase):
    def test_every_catalog_row_has_explicit_abcd_roles(self):
        for gfx, arch in _load_specs().items():
            for index, op in enumerate(arch["mma"]):
                with self.subTest(gfx=gfx, index=index, op_id=op["op_id"]):
                    self.assertTrue({"a", "b", "c", "d"}.issubset(op))

    def test_rejects_legacy_schema_version(self):
        with self.assertRaisesRegex(ValueError, "schema version 2"):
            _validate_specs_doc({"version": 1, "arches": {}})

    def test_rejects_mma_row_without_c_or_d(self):
        base = {
            "family": "mma",
            "a": "fp16",
            "b": "fp16",
            "c": "fp32",
            "d": "fp32",
            "m": 16,
            "n": 16,
            "k": 16,
            "op_id": "synthetic",
        }
        for role in ("c", "d"):
            row = dict(base)
            del row[role]
            with self.subTest(role=role), self.assertRaisesRegex(
                ValueError, f"required fields: {role}"
            ):
                _validate_specs_doc(
                    {"version": 2, "arches": {"synthetic": {"mma": [row]}}}
                )

    def test_catalog_distinguishes_c_input_from_d_result(self):
        row = {
            "family": "mma",
            "a": "fp16",
            "b": "fp16",
            "c": "i32",
            "d": "fp32",
            "m": 1,
            "n": 1,
            "k": 1,
            "op_id": "synthetic_four_role",
        }
        op = _build_mma_op(row)
        catalog = MmaCatalog([op])
        self.assertEqual(op.c_dtype, "i32")
        self.assertEqual(op.d_dtype, "fp32")
        self.assertEqual(
            catalog.enumerate(
                a_dtype="fp16", b_dtype="fp16", c_dtype="i32", d_dtype="fp32"
            ),
            [op],
        )
        self.assertEqual(
            catalog.enumerate(
                a_dtype="fp16", b_dtype="fp16", c_dtype="fp32", d_dtype="fp32"
            ),
            [],
        )

    def test_fragment_metadata_can_model_distinct_c_and_d(self):
        c_fn = lambda *_: ("c0", "c1")
        d_fn = lambda *_: ("d0", "d1")
        info = _FragInfo(
            1,
            2,
            4,
            64,
            d_fn=d_fn,
            c_frag_len=3,
            c_fn=c_fn,
        )
        self.assertEqual((info.c_frag_len, info.d_frag_len), (3, 4))
        self.assertIs(info.c_fn, c_fn)
        self.assertIs(info.d_fn, d_fn)

    def test_shipped_c_and_d_layout_objects_have_distinct_roles(self):
        sample = next(
            op
            for arch in _load_specs().values()
            for row in arch["mma"]
            if (op := _build_mma_op(row))._c_layout is not None
        )
        self.assertIsNot(sample.c_layout(), sample.d_layout())
        self.assertEqual(sample.c_layout().role, "c")
        self.assertEqual(sample.d_layout().role, "d")
        self.assertIs(sample.acc_layout(), sample.d_layout())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
