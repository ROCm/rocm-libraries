# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SSOT guards for indexed MMA ``srcs`` and ``dst`` metadata.

``IRBuilder.mma`` uses ``dst`` metadata to size a ``tile.mma`` result. These
tests pin the ``dst`` lookup, indexed catalog query, and optional per-source
scale parsing without assuming ``src2`` and ``dst`` are always identical.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import asdict, replace
from unittest import mock

import pytest

from rocke.core.arch import MmaScaleBlockK, MmaScaleDType
from rocke.core.arch.wmma_scale import gfx1250_scaled_wmma

from rocke.core.arch.target import (
    ArchTarget,
    MmaCatalog,
    MmaOp,
    MmaDst,
    MmaScaleOperand,
    MmaSrc,
    _FragInfo,
    _FragOperand,
    _MMA_FRAGMENT_INFO,
    _build_mma_op,
    _load_specs,
    _op_id_dst_dtype,
    _op_id_c_dtype,
    _op_id_family,
    normalize_dtype,
)
from rocke.core.ir import IRBuilder


class TestOpIdDstDtype(unittest.TestCase):
    def test_matches_catalog_first_hit(self):
        # Every op_id in the catalog resolves to its normalized dst dtype,
        # taking the first arch that lists it (dict preserves catalog order).
        expected: dict = {}
        for row in _load_specs().values():
            for o in row["mma"]:
                expected.setdefault(o["op_id"], normalize_dtype(o["dst"]["dtype"]))
        self.assertEqual(_op_id_dst_dtype(), expected)

    def test_dst_dtype_invariant_across_arches(self):
        # The whole premise of the bare-op_id lookup: an op_id's dst dtype
        # is invariant across the arches that list it, so building the map must not
        # raise on the real catalog. (The raise path is exercised below.)
        try:
            _op_id_dst_dtype()
        except ValueError as exc:  # pragma: no cover - only hit on real drift
            self.fail(f"_op_id_dst_dtype() raised on the shipped catalog: {exc}")

    def test_raises_on_cross_arch_disagreement(self):
        specs = _load_specs()
        # Find an op_id and clone its row into a fake arch with a different dst dtype.
        sample = next(o for row in specs.values() for o in row["mma"])
        original_dst = normalize_dtype(sample["dst"]["dtype"])
        other_dst = "i32" if original_dst != "i32" else "f32"
        clash = dict(sample)
        clash["dst"] = {"dtype": other_dst}
        drifted = dict(specs)
        drifted["_synthetic_drift"] = {"mma": [clash]}

        _op_id_dst_dtype.cache_clear()
        try:
            with mock.patch("rocke.core.arch.target._load_specs", return_value=drifted):
                with self.assertRaises(ValueError):
                    _op_id_dst_dtype()
        finally:
            _op_id_dst_dtype.cache_clear()


class TestIndexedMmaOperands(unittest.TestCase):
    def test_cpu_catalog_indexed_queries_cover_gfx950_and_gfx1250(self):
        for gfx in ("gfx950", "gfx1250"):
            catalog = ArchTarget.from_gfx(gfx).mma
            for op in catalog.ops:
                with self.subTest(gfx=gfx, op_id=op.op_id):
                    self.assertEqual(len(op.srcs), 3)
                    self.assertIn(
                        op,
                        catalog.enumerate(
                            family=op.family,
                            src_dtypes=tuple(src.dtype for src in op.srcs),
                            dst_dtype=op.dst.dtype,
                            m=op.m,
                            n=op.n,
                        ),
                    )

    def test_indexed_query_distinguishes_src2_and_dst(self):
        distinct = MmaOp(
            family="mma",
            srcs=(MmaSrc("xf32"), MmaSrc("xf32"), MmaSrc("fp32")),
            dst=MmaDst("i32", frag_len=7),
            m=16,
            n=16,
            k=8,
            op_id="synthetic_distinct_dst",
        )
        catalog = MmaCatalog([distinct])
        self.assertEqual(
            catalog.enumerate(src_dtypes=("xf32", "xf32", "fp32"), dst_dtype="i32"),
            [distinct],
        )
        self.assertEqual(
            catalog.enumerate(a_dtype="xf32", b_dtype="xf32", c_dtype="fp32"),
            [],
        )

        builder = IRBuilder("distinct_dst")
        value = builder.const_i32(0)
        result = builder.mma(distinct, value, value, value)
        self.assertEqual(result.type.count, distinct.dst.frag_len)
        self.assertEqual(result.type.elem.name, "i32")

    def test_fragment_metadata_distinguishes_src2_and_dst(self):
        def src2_coord(builder, lane, slot):
            return builder.const_i32(slot), lane

        def dst_coord(builder, lane, slot):
            return lane, builder.const_i32(slot)

        op_id = "synthetic_distinct_fragment_metadata"
        info = _FragInfo(
            srcs=(
                _FragOperand(1),
                _FragOperand(1),
                _FragOperand(3, src2_coord),
            ),
            dst=_FragOperand(7, dst_coord),
            wave_size=32,
        )
        row = {
            "family": "mma",
            "srcs": [
                {"dtype": "xf32"},
                {"dtype": "xf32"},
                {"dtype": "fp32"},
            ],
            "dst": {"dtype": "i32"},
            "m": 16,
            "n": 16,
            "k": 8,
            "op_id": op_id,
        }

        with mock.patch.dict(_MMA_FRAGMENT_INFO, {op_id: info}):
            op = _build_mma_op(row)

        self.assertEqual(op.srcs[2].frag_len, 3)
        self.assertEqual(op.dst.frag_len, 7)
        self.assertIs(op.src_layout(2).fn, src2_coord)
        self.assertIs(op.dst_layout().fn, dst_coord)
        self.assertEqual(op.c_frag_len, op.dst.frag_len)
        self.assertIs(op.c_layout(), op.dst_layout())

        builder = IRBuilder("distinct_fragment_metadata")
        value = builder.const_i32(0)
        result = builder.mma(op, value, value, value)
        self.assertEqual(result.type.count, op.dst.frag_len)
        self.assertEqual(result.type.elem.name, "i32")

    def test_scaled_wmma_catalog_preserves_scale_contracts(self):
        catalog = ArchTarget.from_gfx("gfx1250").mma
        for block_size in (32, 16):
            family = "wmma_scaled"
            with self.subTest(family=family):
                op = catalog.by_op_id(
                    f"wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k{block_size}"
                )
                self.assertIsNotNone(op)
                self.assertEqual(op.family, family)
                self.assertEqual(op.shape, (16, 16, 128))
                for src in op.srcs[:2]:
                    self.assertEqual(src.dtype, "fp8e4m3")
                    self.assertEqual(src.frag_len, 16)
                    self.assertEqual(
                        (src.scale.dtype, src.scale.block_size), ("e8m0", block_size)
                    )
                self.assertIsNone(op.srcs[2].scale)
                self.assertEqual(op.srcs[2].frag_len, 8)
                self.assertEqual(op.dst.frag_len, 8)
                self.assertEqual(op.src_layout(2).role, "src2")
                self.assertEqual(op.dst_layout().role, "dst")

    def test_scale_format_and_block_size_are_independent_per_source(self):
        # Synthetic rows exercise the descriptor without admitting new hardware
        # operations into the shipped catalog.
        for dtype, block_size in (
            ("e8m0", 32),
            ("e8m0", 16),
            ("e4m3", 16),
            ("fp8e4m3", 16),
            ("e5m3", 16),
        ):
            with self.subTest(dtype=dtype, block_size=block_size):
                op = _build_mma_op(
                    {
                        "family": "mma",
                        "srcs": [
                            {
                                "dtype": "fp4",
                                "scale": {"dtype": dtype, "block_size": block_size},
                            },
                            {
                                "dtype": "fp4",
                                "scale": {"dtype": "e8m0", "block_size": 32},
                            },
                            {"dtype": "fp32"},
                        ],
                        "dst": {"dtype": "fp32"},
                        "m": 16,
                        "n": 16,
                        "k": 128,
                        "op_id": "synthetic_scaled_mma",
                    }
                )
                expected_dtype = "e4m3" if dtype == "fp8e4m3" else dtype
                self.assertEqual(
                    op.srcs[0].scale, MmaScaleOperand(expected_dtype, block_size)
                )
                self.assertEqual(op.srcs[1].scale, MmaScaleOperand("e8m0", 32))
                self.assertIsNone(op.srcs[2].scale)
                self.assertFalse(hasattr(op.dst, "scale"))

    def test_scale_dtype_must_be_a_supported_value_format(self):
        for dtype in ("i32", "fp4", "e5m2", "", None):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "e8m0, e4m3, or e5m3"):
                    MmaScaleOperand(dtype, 16)

    def test_scale_dtype_alias_is_canonicalized(self):
        for block_size in (16, 32):
            with self.subTest(block_size=block_size):
                alias = MmaScaleOperand("fp8e4m3", block_size)
                canonical = MmaScaleOperand("e4m3", block_size)
                self.assertEqual(alias.dtype, "e4m3")
                self.assertEqual(alias, canonical)
                self.assertEqual(hash(alias), hash(canonical))

    def test_scale_block_size_must_be_16_or_32(self):
        for block_size in (0, -16, 1, 8, 64, 16.0, 32.0, 16.5, "16", True, None):
            with self.subTest(block_size=block_size):
                with self.assertRaisesRegex(ValueError, "integer equal to 16 or 32"):
                    MmaScaleOperand("e8m0", block_size)

    def test_scale_identity_includes_format_and_block_size(self):
        self.assertEqual(
            len(
                {
                    MmaScaleOperand("e8m0", 32),
                    MmaScaleOperand("e8m0", 16),
                    MmaScaleOperand("e4m3", 16),
                }
            ),
            3,
        )


def _replace_scales(row, **changes):
    srcs = list(row.srcs)
    for i, key in enumerate(("a_scale_dtype", "b_scale_dtype")):
        scale = srcs[i].scale
        if changes == dict(a_scale_dtype=None, b_scale_dtype=None, scale_block_k=None):
            scale = None
        else:
            if key in changes:
                scale = replace(scale, dtype=changes[key])
            if "scale_block_k" in changes:
                scale = replace(scale, block_size=changes["scale_block_k"])
        srcs[i] = replace(srcs[i], scale=scale)
    return replace(row, srcs=tuple(srcs))


def _contracts():
    base = MmaOp(
        family="wmma_scaled",
        srcs=(
            MmaSrc("fp8e4m3", scale=MmaScaleOperand("e8m0", 32)),
            MmaSrc("bf8e5m2", scale=MmaScaleOperand("e8m0", 32)),
            MmaSrc("fp32"),
        ),
        dst=MmaDst("fp32"),
        m=16,
        n=16,
        k=128,
        op_id="fixture",
    )
    rows = [base]
    for field in ("a_scale_dtype", "b_scale_dtype"):
        for dtype in ("e4m3", "e5m3"):
            rows.append(_replace_scales(base, **{field: dtype}))
    rows.append(_replace_scales(base, scale_block_k=16))
    rows.append(
        _replace_scales(
            base, a_scale_dtype=None, b_scale_dtype=None, scale_block_k=None
        )
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
        assert (
            getattr(_replace_scales(row, **{field: "fp8e4m3"}), field)
            is MmaScaleDType.E4M3
        )


@pytest.mark.parametrize("dtype", list(MmaScaleDType))
def test_scale_dtype_preserves_string_and_json_behavior(dtype):
    assert isinstance(dtype, str)
    assert dtype == dtype.value
    assert str(dtype) == f"{dtype}" == dtype.value
    assert {dtype: 1}[dtype.value] == 1
    assert json.dumps(dtype) == json.dumps(dtype.value)
    assert MmaScaleDType(json.loads(json.dumps(dtype))) is dtype
    row = _replace_scales(
        _contracts()[0], a_scale_dtype=dtype.value, b_scale_dtype=dtype
    )
    assert row.a_scale_dtype is row.b_scale_dtype is dtype
    decoded = json.loads(json.dumps(asdict(row)))
    assert (
        decoded["srcs"][0]["scale"]["dtype"]
        == decoded["srcs"][1]["scale"]["dtype"]
        == dtype.value
    )
    decoded["srcs"] = tuple(
        MmaSrc(
            **{
                **src,
                "scale": MmaScaleOperand(**src["scale"]) if src["scale"] else None,
            }
        )
        for src in decoded["srcs"]
    )
    decoded["dst"] = MmaDst(**decoded["dst"])
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
    assert len(rows) == 4
    assert len({row.op_id for row in rows}) == 4
    for row in rows:
        dtype = {"fp8e4m3": "fp8", "bf8e5m2": "bf8"}[row.a_dtype]
        assert row.op_id == (
            f"wmma_gfx1250_f32_16x16x128_{dtype}_{dtype}"
            f"_scale_e8m0_e8m0_k{row.scale_block_k}"
        )
        assert row.a_scale_dtype == row.b_scale_dtype == "e8m0"
        assert row.a_scale_dtype is row.b_scale_dtype is MmaScaleDType.E8M0
        assert isinstance(row.scale_block_k, MmaScaleBlockK)
        packing = gfx1250_scaled_wmma(row.op_id)
        assert packing.atom is row
        assert packing.matrix_formats == (
            (0, 0) if row.a_dtype == "fp8e4m3" else (1, 1)
        )
        assert packing.scales.count * packing.scales.block_k == row.k
        assert (row.a_frag_len, row.b_frag_len) == (16, 16)
    for family in ("wmma_scale", "wmma_scale16"):
        old_id = f"{family}_f32_16x16x128_fp8_fp8"
        assert catalog.by_op_id(old_id) is None
        assert gfx1250_scaled_wmma(old_id) is None


def test_scale_block_k_is_a_two_value_enum():
    assert list(MmaScaleBlockK) == [MmaScaleBlockK.K16, MmaScaleBlockK.K32]
    for block in MmaScaleBlockK:
        assert (
            _replace_scales(_contracts()[0], scale_block_k=block).scale_block_k is block
        )
        assert (
            _replace_scales(_contracts()[0], scale_block_k=int(block)).scale_block_k
            is block
        )
    with pytest.raises(ValueError):
        MmaScaleBlockK(64)


@pytest.mark.parametrize("field", ["a_scale_dtype", "b_scale_dtype"])
@pytest.mark.parametrize(
    "dtype", ["i32", "fp4", "e5m2", "fp8e5m2", "bf8e5m2", "bf8", "", None]
)
def test_invalid_scale_format(field, dtype):
    with pytest.raises(ValueError, match="e8m0, e4m3, or e5m3"):
        _replace_scales(_contracts()[0], **{field: dtype})


@pytest.mark.parametrize("block", [0, 8, 64, 16.0, True, "32", None])
def test_invalid_scale_block_size(block):
    with pytest.raises(ValueError, match="integer equal to 16 or 32"):
        _replace_scales(_contracts()[0], scale_block_k=block)


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
    row = MmaOp(
        "wmma",
        (MmaSrc("fp8e4m3"), MmaSrc("fp8e4m3"), MmaSrc("fp32")),
        MmaDst("fp32"),
        16,
        16,
        64,
        "fixture",
    )
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


@pytest.mark.parametrize("block_size", [16, 32])
def test_indexed_queries_combine_distinct_result_and_scale_contract(block_size):
    base = _replace_scales(_contracts()[0], scale_block_k=block_size)
    distinct = replace(base, dst=MmaDst("i32", frag_len=7))
    other_block = _replace_scales(distinct, scale_block_k=48 - block_size)
    catalog = MmaCatalog([base, distinct, other_block])
    query = dict(
        family="wmma_scaled",
        src_dtypes=("fp8", "bf8", "f32"),
        dst_dtype="i32",
        scales=("e8m0", "e8m0", block_size),
        m=16,
        n=16,
    )
    assert catalog.enumerate(**query) == [distinct]
    assert catalog.has_shape(**query, k=128)
    assert catalog.op_for_shape(**query, k=128) is distinct
    assert catalog.select_largest_k(**query) is distinct
    assert catalog.op_for_shape(**{**query, "dst_dtype": "fp32"}, k=128) is base
    with pytest.raises(ValueError, match="ambiguous MMA query"):
        catalog.op_for_shape(**{**query, "scales": None}, k=128)


def test_legacy_scaled_json_maps_to_the_same_indexed_contract():
    row = dict(
        family="wmma_scaled",
        a="fp8",
        b="fp8",
        c="fp32",
        m=16,
        n=16,
        k=128,
        op_id="wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k32",
        a_scale_dtype="e8m0",
        b_scale_dtype="e8m0",
        scale_block_k=32,
    )
    atom = _build_mma_op(row)
    assert atom == ArchTarget.from_gfx("gfx1250").mma.by_op_id(row["op_id"])
    assert atom.srcs[0].scale.layout.role == "src0_scale"
    assert atom.srcs[1].scale.layout.role == "src1_scale"
    conflict = {
        **row,
        "srcs": [
            {"dtype": "fp8", "scale": {"dtype": "e4m3", "block_size": 32}},
            {"dtype": "fp8"},
            {"dtype": "fp32"},
        ],
    }
    with pytest.raises(ValueError, match="conflicting indexed and legacy"):
        _build_mma_op(conflict)


def test_shared_scale_projection_does_not_hide_independent_blocks():
    atom = _contracts()[0]
    src0 = replace(atom.srcs[0], scale=replace(atom.srcs[0].scale, block_size=16))
    with pytest.raises(ValueError, match="shared scale block size"):
        replace(atom, srcs=(src0, *atom.srcs[1:]))
    # General metadata can still describe independently scaled sources.
    independent = replace(atom, family="mma", srcs=(src0, *atom.srcs[1:]))
    assert independent.srcs[0].scale.block_size == 16
    assert independent.srcs[1].scale.block_size == 32
    catalog = MmaCatalog([independent])
    query = dict(src_dtypes=("fp8", "bf8", "fp32"), dst_dtype="fp32")
    assert catalog.enumerate(**query) == [independent]
    assert catalog.enumerate(**query, scales=("e8m0", "e8m0", 16)) == []
    assert catalog.enumerate(**query, scales=("e8m0", "e8m0", 32)) == []
