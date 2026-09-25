# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SSOT guards for the bare-op_id MMA accumulator-dtype lookup.

``IRBuilder.mma`` uses ``target._op_id_c_dtype()`` to size a ``tile.mma`` result
vector's accumulator element without an ``ArchTarget`` in hand. These tests pin
the first-wins / raise-on-drift contract of that lookup so it stays deterministic
across the arches that list a given op_id.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import asdict, replace
from unittest import mock

import pytest

from rocke.core.arch import ArchTarget, MmaCatalog, MmaOp, MmaScaleBlockK, MmaScaleDType
from rocke.core.arch.wmma_scale import gfx1250_scaled_wmma

from rocke.core.arch.target import (
    _load_specs,
    _op_id_c_dtype,
    _op_id_family,
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
        a_scale_dtype="e8m0",
        b_scale_dtype="e8m0",
        scale_block_k=32,
        m=16,
        n=16,
        k=128,
        op_id="fixture",
    )
    rows = [base]
    for field in ("a_scale_dtype", "b_scale_dtype"):
        for dtype in ("e4m3", "e5m3"):
            rows.append(replace(base, **{field: dtype}))
    rows.append(replace(base, scale_block_k=16))
    rows.append(
        replace(base, a_scale_dtype=None, b_scale_dtype=None, scale_block_k=None)
    )
    return [replace(row, op_id=f"fixture_{i}") for i, row in enumerate(rows)]


@pytest.mark.parametrize("use_strings", [False, True])
def test_full_contract_queries_distinguish_each_scale_type_and_shared_block(
    use_strings,
):
    rows = _contracts()
    assert len({row.op_id for row in rows}) == len(rows)
    catalog = MmaCatalog(rows)
    for row in rows:
        query = dict(
            family=row.family,
            a_dtype=row.a_dtype,
            b_dtype=row.b_dtype,
            c_dtype=row.c_dtype,
            scales=(
                (
                    str(row.a_scale_dtype)
                    if use_strings and row.a_scale_dtype
                    else row.a_scale_dtype
                ),
                (
                    str(row.b_scale_dtype)
                    if use_strings and row.b_scale_dtype
                    else row.b_scale_dtype
                ),
                row.scale_block_k,
            ),
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
    assert len(catalog.enumerate(**query)) == 7
    assert catalog.has_shape(**query, k=128)
    with pytest.raises(ValueError, match="ambiguous MMA query"):
        catalog.op_for_shape(**query, k=128)
    with pytest.raises(ValueError, match="ambiguous MMA query"):
        catalog.select_largest_k(**query)
    assert len(catalog.enumerate(**query, scales=(None, None, None))) == 1
    assert catalog.op_for_shape(**query, scales=("e5m3", "e5m3", 32), k=128) is None
    with pytest.raises(ValueError, match="exactly 3"):
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
    for scales, field in (
        (("fp8e4m3", "e8m0", 32), "a_scale_dtype"),
        (("e8m0", "fp8e4m3", 32), "b_scale_dtype"),
    ):
        row = catalog.op_for_shape(**query, scales=scales)
        assert row is not None and getattr(row, field) is MmaScaleDType.E4M3
        assert getattr(replace(row, **{field: "fp8e4m3"}), field) is MmaScaleDType.E4M3


@pytest.mark.parametrize("dtype", list(MmaScaleDType))
def test_scale_dtype_preserves_string_and_json_behavior(dtype):
    assert isinstance(dtype, str)
    assert dtype == dtype.value
    assert str(dtype) == f"{dtype}" == dtype.value
    assert {dtype: 1}[dtype.value] == 1
    assert json.dumps(dtype) == json.dumps(dtype.value)
    assert MmaScaleDType(json.loads(json.dumps(dtype))) is dtype
    row = replace(_contracts()[0], a_scale_dtype=dtype.value, b_scale_dtype=dtype)
    assert row.a_scale_dtype is row.b_scale_dtype is dtype
    decoded = json.loads(json.dumps(asdict(row)))
    assert decoded["a_scale_dtype"] == decoded["b_scale_dtype"] == dtype.value
    restored = MmaOp(**decoded)
    assert restored == row
    assert restored.a_scale_dtype is restored.b_scale_dtype is dtype


def test_scale_dtype_members_and_alias():
    assert [dtype.value for dtype in MmaScaleDType] == ["e8m0", "e4m3", "e5m3"]
    assert MmaScaleDType("fp8e4m3") is MmaScaleDType.E4M3


@pytest.mark.parametrize("dtype", ["e5m2", "fp8e5m2", "bf8e5m2", "bf8"])
@pytest.mark.parametrize("field", ["a_scale_dtype", "b_scale_dtype"])
def test_e5m2_is_not_a_scale_dtype_alias(dtype, field):
    with pytest.raises(ValueError):
        MmaScaleDType(dtype)
    scales = ["e8m0", "e8m0", 32]
    scales[0 if field == "a_scale_dtype" else 1] = dtype
    catalog = MmaCatalog(_contracts())
    query = dict(
        family="wmma_scaled",
        a_dtype="fp8",
        b_dtype="bf8",
        c_dtype="fp32",
        scales=tuple(scales),
        m=16,
        n=16,
    )
    for method, kwargs in (
        (catalog.enumerate, query),
        (catalog.has_shape, {**query, "k": 128}),
        (catalog.op_for_shape, {**query, "k": 128}),
        (catalog.select_largest_k, query),
    ):
        with pytest.raises(ValueError, match="MMA scale dtype"):
            method(**kwargs)


def test_scaled_catalog_identity_and_backend_contract():
    catalog = ArchTarget.from_gfx("gfx1250").mma
    rows = [row for row in catalog.ops if row.family == "wmma_scaled"]
    assert len(rows) == 6
    assert len({row.op_id for row in rows}) == 6
    for row in rows:
        dtype = {"fp8e4m3": "fp8", "bf8e5m2": "bf8", "fp4e2m1": "fp4"}[row.a_dtype]
        assert row.op_id == (
            f"wmma_gfx1250_f32_16x16x128_{dtype}_{dtype}"
            f"_scale_e8m0_e8m0_k{row.scale_block_k}"
        )
        assert row.a_scale_dtype == row.b_scale_dtype == "e8m0"
        assert row.a_scale_dtype is row.b_scale_dtype is MmaScaleDType.E8M0
        assert isinstance(row.scale_block_k, MmaScaleBlockK)
        packing = gfx1250_scaled_wmma(row.op_id)
        assert packing.atom is row
        selector = {"fp8e4m3": 0, "bf8e5m2": 1, "fp4e2m1": 4}[row.a_dtype]
        assert packing.matrix_formats == (selector, selector)
        assert packing.scales.count * packing.scales.block_k == row.k
        assert packing.matrix_llvm_types == ("<16 x i32>", "<16 x i32>")
    for family in ("wmma_scale", "wmma_scale16"):
        old_id = f"{family}_f32_16x16x128_fp8_fp8"
        assert catalog.by_op_id(old_id) is None
        assert gfx1250_scaled_wmma(old_id) is None


def test_scale_block_k_is_a_two_value_enum():
    assert list(MmaScaleBlockK) == [MmaScaleBlockK.K16, MmaScaleBlockK.K32]
    for block in MmaScaleBlockK:
        assert replace(_contracts()[0], scale_block_k=block).scale_block_k is block
        assert replace(_contracts()[0], scale_block_k=int(block)).scale_block_k is block
    with pytest.raises(ValueError):
        MmaScaleBlockK(64)


@pytest.mark.parametrize("field", ["a_scale_dtype", "b_scale_dtype"])
@pytest.mark.parametrize(
    "dtype", ["i32", "fp4", "e5m2", "fp8e5m2", "bf8e5m2", "bf8", "", None]
)
def test_invalid_scale_format(field, dtype):
    with pytest.raises(ValueError, match="e8m0, e4m3, or e5m3"):
        replace(_contracts()[0], **{field: dtype})


@pytest.mark.parametrize("block", [0, 8, 64, 16.0, True, "32", None])
def test_invalid_scale_block_size(block):
    with pytest.raises(ValueError, match="integer equal to 16 or 32"):
        replace(_contracts()[0], scale_block_k=block)


@pytest.mark.parametrize(
    "scales",
    [
        ("e8m0", None, 32),
        (None, "e8m0", 32),
        (None, None, 32),
        ("e8m0", "e8m0", None),
        ("i32", "e8m0", 32),
        ("e8m0", "e8m0", 0),
        ("e8m0", "e8m0", 16.0),
        ("e8m0", "e8m0", True),
    ],
)
def test_invalid_scale_query_even_for_empty_catalog(scales):
    with pytest.raises(ValueError, match="MMA scale"):
        MmaCatalog([]).enumerate(
            family="wmma_scaled",
            a_dtype="fp8",
            b_dtype="fp8",
            c_dtype="fp32",
            scales=scales,
        )


def test_unscaled_defaults_have_no_scale_metadata():
    row = MmaOp("wmma", "fp8e4m3", "fp8e4m3", "fp32", 16, 16, 64, "fixture")
    assert (row.a_scale_dtype, row.b_scale_dtype, row.scale_block_k) == (
        None,
        None,
        None,
    )
    assert (
        MmaCatalog([row]).op_for_shape(
            family="wmma",
            a_dtype="fp8",
            b_dtype="fp8",
            c_dtype="fp32",
            m=16,
            n=16,
            k=64,
            scales=(None, None, None),
        )
        is row
    )


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


def test_op_id_family_matches_all_catalogs_and_rejects_conflicts():
    specs = _load_specs()
    for row in specs.values():
        for op in row["mma"]:
            assert _op_id_family()[op["op_id"]] == op["family"]
    assert _op_id_family().get("unknown") is None
    sample = next(op for row in specs.values() for op in row["mma"])
    drifted = {**specs, "synthetic": {"mma": [{**sample, "family": "conflicting"}]}}
    _op_id_family.cache_clear()
    try:
        with mock.patch("rocke.core.arch.target._load_specs", return_value=drifted):
            with pytest.raises(ValueError, match="inconsistent family"):
                _op_id_family()
    finally:
        _op_id_family.cache_clear()


def test_mma_naming_without_target_lookup():
    from rocke.core.ir import F16, F32, I32, IRBuilder

    atoms = [
        (
            ArchTarget.from_gfx("gfx950").mma.by_op_id("mfma_f32_16x16x16_f16"),
            F16,
            "acc",
        ),
        (
            ArchTarget.from_gfx("gfx1151").mma.by_op_id("wmma_i32_16x16x16_iu8"),
            I32,
            "acc",
        ),
        (
            next(
                op
                for op in ArchTarget.from_gfx("gfx1250").mma.ops
                if op.family == "wmma_scaled"
            ),
            I32,
            "mxacc",
        ),
    ]
    _op_id_family.cache_clear()
    with mock.patch.object(
        ArchTarget,
        "from_gfx",
        side_effect=AssertionError("target lookup in generic MMA"),
    ):
        for atom, elem, hint in atoms:
            assert atom is not None
            for arg in (atom, atom.op_id):
                b = IRBuilder("neutral_mma")
                a = b.zero_vec(elem, atom.a_frag_len)
                c = b.zero_vec(I32 if atom.c_dtype == "i32" else F32, atom.c_frag_len)
                extra = (
                    (b.const_i32(0), b.const_i32(0))
                    if atom.family == "wmma_scaled"
                    else ()
                )
                result = b.mma(arg, a, a, c, *extra)
                assert result.name.lstrip("%").startswith(hint)
