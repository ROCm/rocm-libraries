# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SSOT guards for the bare-op_id MMA accumulator-dtype lookup.

``IRBuilder.mma`` uses ``target._op_id_c_dtype()`` to size a ``tile.mma`` result
vector's accumulator element without an ``ArchTarget`` in hand. These tests pin
the first-wins / raise-on-drift contract of that lookup so it stays deterministic
across the arches that list a given op_id.
"""

from __future__ import annotations

import unittest
from unittest import mock

from rocke.core.arch.target import (
    _load_specs,
    _op_id_c_dtype,
    normalize_dtype,
)


class TestOpIdCDtype(unittest.TestCase):
    def test_matches_catalog_first_hit(self):
        # Every op_id in the catalog resolves to its normalized accumulator dtype,
        # taking the first arch that lists it (dict preserves catalog order).
        expected: dict = {}
        for row in _load_specs().values():
            for o in row["mma"]:
                expected.setdefault(o["op_id"], normalize_dtype(o["c"]))
        self.assertEqual(_op_id_c_dtype(), expected)

    def test_c_dtype_invariant_across_arches(self):
        # The whole premise of the bare-op_id lookup: an op_id's accumulator dtype
        # is invariant across the arches that list it, so building the map must not
        # raise on the real catalog. (The raise path is exercised below.)
        try:
            _op_id_c_dtype()
        except ValueError as exc:  # pragma: no cover - only hit on real drift
            self.fail(f"_op_id_c_dtype() raised on the shipped catalog: {exc}")

    def test_raises_on_cross_arch_disagreement(self):
        specs = _load_specs()
        # Find an op_id and clone its row into a fake arch with a different c dtype.
        sample = next(o for row in specs.values() for o in row["mma"])
        original_c = normalize_dtype(sample["c"])
        other_c = "i32" if original_c != "i32" else "f32"
        clash = dict(sample)
        clash["c"] = other_c
        drifted = dict(specs)
        drifted["_synthetic_drift"] = {"mma": [clash]}

        _op_id_c_dtype.cache_clear()
        try:
            with mock.patch("rocke.core.arch.target._load_specs", return_value=drifted):
                with self.assertRaises(ValueError):
                    _op_id_c_dtype()
        finally:
            _op_id_c_dtype.cache_clear()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
