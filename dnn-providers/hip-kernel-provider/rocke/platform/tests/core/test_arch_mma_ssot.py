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
from dataclasses import replace
from unittest import mock

import pytest

from rocke.core.arch import ArchTarget, MmaCatalog, MmaOp, MmaScaleOperand
from rocke.core.arch.wmma_scale import gfx1250_scaled_wmma

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


def _contracts():
    base = MmaOp(
        family="wmma_scaled",
        a_dtype="fp8e4m3",
        b_dtype="bf8e5m2",
        c_dtype="fp32",
        a_scale=MmaScaleOperand("e8m0", 32),
        b_scale=MmaScaleOperand("e8m0", 32),
        m=16,
        n=16,
        k=128,
        op_id="fixture",
    )
    rows = [base]
    for field in ("a_scale", "b_scale"):
        for scale in (MmaScaleOperand("e4m3", 32), MmaScaleOperand("e8m0", 16), None):
            rows.append(replace(base, **{field: scale}))
    rows.append(replace(base, a_scale=None, b_scale=None))
    return [replace(row, op_id=row.semantic_id()) for row in rows]


def test_full_contract_queries_distinguish_each_input_scale():
    rows = _contracts()
    assert len({row.op_id for row in rows}) == len(rows)
    catalog = MmaCatalog(rows)
    for row in rows:
        query = dict(
            family=row.family,
            a_dtype=row.a_dtype,
            b_dtype=row.b_dtype,
            c_dtype=row.c_dtype,
            scales=(row.a_scale, row.b_scale),
            m=row.m,
            n=row.n,
        )
        assert catalog.enumerate(**query) == [row]
        assert catalog.has_shape(**query, k=row.k)
        assert catalog.op_for_shape(**query, k=row.k) is row
        assert catalog.select_largest_k(**query) is row
        assert catalog.select_largest_k(**query, k_max=64) is None


def test_partial_query_rejects_ambiguity_but_enumeration_and_existence_are_valid():
    catalog = MmaCatalog(_contracts())
    query = dict(
        family="wmma_scaled",
        a_dtype="fp8",
        b_dtype="bf8",
        c_dtype="fp32",
        m=16,
        n=16,
    )
    assert len(catalog.enumerate(**query)) == 8
    assert catalog.has_shape(**query, k=128)
    with pytest.raises(ValueError, match="ambiguous MMA query"):
        catalog.op_for_shape(**query, k=128)
    with pytest.raises(ValueError, match="ambiguous MMA query"):
        catalog.select_largest_k(**query)
    assert len(catalog.enumerate(**query, scales=(None, None))) == 1
    scale = MmaScaleOperand("e5m3", 32)
    assert catalog.op_for_shape(**query, scales=(scale, scale), k=128) is None
    with pytest.raises(ValueError, match="exactly 2"):
        catalog.enumerate(**query, scales=(None,))


def test_scale_alias_selects_the_same_contract():
    catalog = MmaCatalog(_contracts())
    query = dict(
        family="wmma_scaled",
        a_dtype="fp8",
        b_dtype="bf8",
        c_dtype="f32",
        m=16,
        n=16,
        k=128,
    )
    scales = (MmaScaleOperand("fp8e4m3", 32), MmaScaleOperand("e8m0", 32))
    row = catalog.op_for_shape(**query, scales=scales)
    assert row is not None and row.a_scale.dtype == "e4m3"


def test_scaled_catalog_identity_and_backend_contract():
    catalog = ArchTarget.from_gfx("gfx1250").mma
    rows = [row for row in catalog.ops if row.family == "wmma_scaled"]
    assert len(rows) == 4
    for row in rows:
        assert row.op_id == row.semantic_id()
        packing = gfx1250_scaled_wmma(row.op_id)
        assert packing.atom is row
        assert packing.matrix_formats == (
            (0, 0) if row.a_dtype == "fp8e4m3" else (1, 1)
        )
        assert packing.scales.count * packing.scales.block_k == row.k
        assert packing.matrix_llvm_types == ("<16 x i32>", "<16 x i32>")
    for family in ("wmma_scale", "wmma_scale16"):
        old_id = f"{family}_f32_16x16x128_fp8_fp8"
        assert catalog.by_op_id(old_id) is None
        assert gfx1250_scaled_wmma(old_id) is None


@pytest.mark.parametrize("dtype", ["i32", "fp4", "e5m2", "", None])
def test_invalid_scale_format(dtype):
    with pytest.raises(ValueError, match="e8m0, e4m3, or e5m3"):
        MmaScaleOperand(dtype, 16)


@pytest.mark.parametrize("block", [0, 8, 64, 16.0, True, "32", None])
def test_invalid_scale_block_size(block):
    with pytest.raises(ValueError, match="integer equal to 16 or 32"):
        MmaScaleOperand("e8m0", block)


def test_largest_k_only_rejects_ties_at_the_selected_k():
    rows = _contracts()
    unique = replace(rows[0], k=256)
    catalog = MmaCatalog([*rows, unique])
    query = dict(
        family="wmma_scaled", a_dtype="fp8", b_dtype="bf8", c_dtype="fp32", m=16, n=16
    )
    assert catalog.select_largest_k(**query) is unique
    with pytest.raises(ValueError, match="ambiguous MMA query"):
        catalog.select_largest_k(**query, k_max=128)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
