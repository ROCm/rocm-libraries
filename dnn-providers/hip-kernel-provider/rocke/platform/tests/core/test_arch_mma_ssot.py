# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SSOT guards for indexed MMA ``srcs`` and ``dst`` metadata.

``IRBuilder.mma`` uses ``dst`` metadata to size a ``tile.mma`` result. These
tests pin the ``dst`` lookup, indexed catalog query, and optional per-source
scale parsing without assuming ``src2`` and ``dst`` are always identical.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
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
    _expand_mma_rows,
    _mma_op_by_id,
    _GFX1250_SCALED_FAMILY,
    _mma_operand_rows,
    _op_id_c_dtype,
    _op_id_family,
    normalize_dtype,
)
from rocke.core.ir import IRBuilder


@pytest.mark.parametrize("width", [-1, 1.5, "4", None, True, False])
def test_mma_rejects_invalid_destination_width(width):
    atom = ArchTarget.from_gfx("gfx942").mma.op_for_shape(
        family="mma", a_dtype="fp32", b_dtype="fp32", c_dtype="fp32", m=16, n=16, k=4
    )
    atom = replace(atom, dst=replace(atom.dst, frag_len=width))
    b = IRBuilder("invalid_dst_width")
    value = b.const_i32(0)
    with pytest.raises(ValueError, match="destination fragment length"):
        b.mma(atom, value, value, value)


@pytest.mark.parametrize("dtype", ["garbage", "", None, 32])
def test_mma_rejects_unsupported_destination_dtype(dtype):
    atom = ArchTarget.from_gfx("gfx942").mma.op_for_shape(
        family="mma", a_dtype="fp32", b_dtype="fp32", c_dtype="fp32", m=16, n=16, k=4
    )
    atom = replace(atom, dst=replace(atom.dst, dtype=dtype))
    b = IRBuilder("invalid_dst_dtype")
    value = b.const_i32(0)
    with pytest.raises(ValueError, match="destination dtype"):
        b.mma(atom, value, value, value)


@pytest.mark.parametrize(
    "dtype,element",
    [("fp32", "f32"), (" F32 ", "f32"), ("i32", "i32"), (" INT32 ", "i32")],
)
def test_mma_destination_aliases_and_unspecified_width(dtype, element):
    atom = ArchTarget.from_gfx("gfx942").mma.op_for_shape(
        family="mma", a_dtype="fp32", b_dtype="fp32", c_dtype="fp32", m=16, n=16, k=4
    )
    unspecified = replace(atom, dst=MmaDst(dtype))
    b = IRBuilder("unspecified_dst_width")
    value = b.const_i32(0)
    result = b.mma(unspecified, value, value, value)
    assert result.type.count == atom.dst.frag_len
    assert result.type.elem.name == element
    assert b.mma(atom.op_id, value, value, value).type.count == atom.dst.frag_len
    unknown = replace(unspecified, op_id="unknown_atom")
    with pytest.raises(ValueError, match="unknown MMA op_id"):
        b.mma(unknown, value, value, value)


def test_gemm_zero_accumulator_uses_source_width():
    from rocke.instances.common.gemm_universal import _emit_zero_acc_op

    atom = ArchTarget.from_gfx("gfx942").mma.op_for_shape(
        family="mma", a_dtype="fp32", b_dtype="fp32", c_dtype="fp32", m=16, n=16, k=4
    )
    atom = replace(
        atom,
        srcs=(*atom.srcs[:2], MmaSrc("fp32", frag_len=4)),
        dst=MmaDst("fp32", frag_len=7),
    )
    b = IRBuilder("independent_accumulator")
    acc = _emit_zero_acc_op(b, atom)
    assert acc.type.count == 4
    assert acc.type.elem.name == "f32"
    assert b.mma(atom, acc, acc, acc).type.count == 7


def test_filtered_catalog_preserves_supported_order_and_fields():
    # Independent snapshot of the supported rows at 72c28a69, filtered with
    # the LLVM legality table. Preserve all remaining fields and relative order.
    rows = _load_specs()["gfx1250"]["mma"]
    assert len(rows) == 93
    legacy = []
    for row in rows:
        row = deepcopy(row)
        if "srcs" in row:
            srcs, dst = row.pop("srcs"), row.pop("dst")
            assert dst["dtype"] == srcs[2]["dtype"]
            for key, src in zip(("a", "b", "c"), srcs):
                row[key] = src["dtype"]
                if "scale" in src:
                    row[f"{key}_scale_dtype"] = src["scale"]["dtype"]
                    row["scale_block_k"] = src["scale"]["block_size"]
        legacy.append(row)
    encoded = json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == (
        "d31fe60f6172c7c2172a420cc4558591fa1389d41cc2829eafc99bceb66e858a"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("generate", "gfx1250_wmma_scaled_v2"),
        ("family", "wmma"),
        ("input_dtypes", ["fp8", "bf8", "fp6", "bf6", "fp8"]),
        ("input_dtypes", ["fp8"]),
        ("scale_dtypes", ["e8m0", "e5m2", "e4m3"]),
        ("scale_dtypes", ["e8m0", "e4m3", "e5m3"]),
        ("block_k", [32, 8]),
        ("block_k", [32.0, 16]),
        ("shape", [16, 16, 64]),
        ("c", "fp16"),
        ("unexpected", True),
    ],
)
def test_family_declaration_rejects_unsupported_axes(field, value):
    declaration = deepcopy(_GFX1250_SCALED_FAMILY)
    declaration[field] = value
    with pytest.raises(ValueError, match="unsupported MMA family declaration"):
        _expand_mma_rows("gfx1250", [declaration])


def test_family_declaration_is_admitted_by_its_content_on_any_arch():
    # The declaration is checked against the one supported family, not against
    # the name of the arch whose row carries it.
    declaration = deepcopy(_GFX1250_SCALED_FAMILY)
    expected = _expand_mma_rows("gfx1250", [declaration])
    assert expected
    for gfx in ("gfx950", "gfx1201"):
        assert _expand_mma_rows(gfx, [deepcopy(declaration)]) == expected


def test_family_declaration_rejects_missing_fields_and_duplicates():
    declaration = deepcopy(_GFX1250_SCALED_FAMILY)
    for field in declaration:
        if field == "generate":
            with pytest.raises(ValueError, match="invalid MMA declaration"):
                _expand_mma_rows(
                    "gfx1250", [{k: v for k, v in declaration.items() if k != field}]
                )
            continue
        missing = {k: v for k, v in declaration.items() if k != field}
        with pytest.raises(ValueError, match="unsupported MMA family declaration"):
            _expand_mma_rows("gfx1250", [missing])
    rows = _expand_mma_rows("gfx1250", [declaration])
    for declarations in (
        [declaration, declaration],
        [rows[0], declaration],
        [declaration, rows[-1]],
    ):
        with pytest.raises(ValueError, match="duplicate MMA op_id"):
            _expand_mma_rows("gfx1250", declarations)


def test_atom_lookup_by_op_id_needs_no_target():
    # Every catalogued atom is found from its op_id alone, whichever arch lists it.
    for gfx in _load_specs():
        for op in ArchTarget.from_gfx(gfx).mma.ops:
            found = _mma_op_by_id(op.op_id)
            assert found is not None
            assert (found.op_id, found.family) == (op.op_id, op.family)
    assert _mma_op_by_id("unknown") is None


@pytest.mark.parametrize("declarations", [None, {}, [None], [42], [{}]])
def test_invalid_family_schema_fails_loudly(declarations):
    with pytest.raises(ValueError, match="MMA declaration"):
        _expand_mma_rows("gfx1250", declarations)


class TestOpIdDstDtype(unittest.TestCase):
    def test_matches_catalog_first_hit(self):
        # Every op_id in the catalog resolves to its normalized dst dtype,
        # taking the first arch that lists it (dict preserves catalog order).
        expected: dict = {}
        for row in _load_specs().values():
            for o in row["mma"]:
                expected.setdefault(
                    o["op_id"], normalize_dtype(_mma_operand_rows(o)[1]["dtype"])
                )
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
    def test_cpu_catalog_indexed_queries_cover_mfma_and_wmma(self):
        for gfx in ("gfx942", "gfx950", "gfx1250"):
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
            srcs=(MmaSrc("tf32"), MmaSrc("tf32"), MmaSrc("fp32")),
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
    assert len(rows) == 86
    assert len({row.op_id for row in rows}) == 86
    for row in rows:
        tokens = {
            "fp8e4m3": "fp8",
            "bf8e5m2": "bf8",
            "fp6e2m3": "fp6",
            "fp6e3m2": "bf6",
            "fp4e2m1": "fp4",
        }
        assert row.op_id == (
            f"wmma_gfx1250_f32_16x16x128_{tokens[row.a_dtype]}_{tokens[row.b_dtype]}"
            f"_scale_{row.a_scale_dtype}_{row.b_scale_dtype}_k{row.scale_block_k}"
        )
        assert isinstance(row.a_scale_dtype, MmaScaleDType)
        assert isinstance(row.b_scale_dtype, MmaScaleDType)
        assert isinstance(row.scale_block_k, MmaScaleBlockK)
        packing = gfx1250_scaled_wmma(row.op_id)
        assert packing.atom is row
        selectors = {
            "fp8e4m3": 0,
            "bf8e5m2": 1,
            "fp6e2m3": 2,
            "fp6e3m2": 3,
            "fp4e2m1": 4,
        }
        assert packing.matrix_formats == (
            selectors[row.a_dtype],
            selectors[row.b_dtype],
        )
        assert packing.scale_formats == tuple(
            {"e8m0": 0, "e5m3": 1, "e4m3": 2}[dtype]
            for dtype in (row.a_scale_dtype, row.b_scale_dtype)
        )
        assert packing.scales.count * packing.scales.block_k == row.k
        assert (row.a_frag_len, row.b_frag_len) == (16, 16)
    for family in ("wmma_scale", "wmma_scale16"):
        old_id = f"{family}_f32_16x16x128_fp8_fp8"
        assert catalog.by_op_id(old_id) is None
        assert gfx1250_scaled_wmma(old_id) is None


@pytest.mark.parametrize("dtype", ["fp8", "bf8", "fp4"])
@pytest.mark.parametrize("block_k,scale_count", [(16, 8), (32, 4)])
def test_scaled_indexed_query_preserves_source_scale_contract(
    dtype, block_k, scale_count
):
    catalog = ArchTarget.from_gfx("gfx1250").mma
    atom = catalog.op_for_shape(
        family="wmma_scaled",
        src_dtypes=(dtype, dtype, "fp32"),
        dst_dtype="fp32",
        m=16,
        n=16,
        k=128,
        scales=(MmaScaleDType.E8M0, MmaScaleDType.E8M0, block_k),
    )
    assert atom is not None
    for src in atom.srcs[:2]:
        assert src.scale.dtype is MmaScaleDType.E8M0
        assert src.scale.block_size == block_k
        assert src.scale.frag_len == scale_count
    assert atom.srcs[2].scale is None
    assert atom.scale_block_k == block_k
    assert atom.k == 128


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
    assert atom.srcs[0].scale.layout.role == "scale_src0"
    assert atom.srcs[1].scale.layout.role == "scale_src1"
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


@pytest.mark.parametrize("dst_dtype", [None, "", "fp32"])
def test_indexed_queries_preserve_explicit_destination(dst_dtype):
    atom = _contracts()[0]
    catalog = MmaCatalog([atom])
    query = dict(
        family=atom.family,
        src_dtypes=("fp8", "bf8", "fp32"),
        dst_dtype=dst_dtype,
        m=16,
        n=16,
    )
    expected = None if dst_dtype == "" else atom
    assert catalog.enumerate(**query) == ([] if expected is None else [atom])
    assert catalog.has_shape(**query, k=128) == (expected is not None)
    assert catalog.op_for_shape(**query, k=128) is expected
    assert catalog.select_largest_k(**query) is expected


def test_omitted_destination_excludes_distinct_result_contracts():
    base = _contracts()[0]
    distinct = replace(base, dst=replace(base.dst, dtype="i32"))
    catalog = MmaCatalog([distinct, base])
    for dtypes in (
        dict(src_dtypes=("fp8", "bf8", "fp32")),
        dict(a_dtype="fp8", b_dtype="bf8", c_dtype="fp32"),
    ):
        query = dict(family=base.family, m=16, n=16, **dtypes)
        assert catalog.enumerate(**query) == [base]
        assert catalog.has_shape(**query, k=128)
        assert catalog.op_for_shape(**query, k=128) is base
        assert catalog.select_largest_k(**query) is base
        assert catalog.op_for_shape(**query, dst_dtype="i32", k=128) is distinct


def test_indexed_json_without_destination_inherits_source_two():
    row = dict(
        family="mma",
        m=16,
        n=16,
        k=4,
        op_id="mfma_f32_16x16x4_f32",
        srcs=[{"dtype": "fp32"}, {"dtype": "fp32"}, {"dtype": "fp32"}],
    )
    implicit = _build_mma_op(row)
    explicit = _build_mma_op({**row, "dst": {"dtype": "fp32"}})
    assert implicit == explicit
    assert implicit.srcs[2].dtype == implicit.dst.dtype == "fp32"
    assert implicit.srcs[2].layout.role == "src2"
    assert implicit.dst.layout.role == "dst"
    with pytest.raises(ValueError, match="dst must define a string dtype"):
        _build_mma_op({**row, "dst": None})


@pytest.mark.parametrize("source", [0, 1])
@pytest.mark.parametrize("legacy_dtype", [None, "e4m3"])
@pytest.mark.parametrize("indexed_dtype", [None, "fp8e4m3"])
def test_explicit_legacy_and_indexed_scales_must_agree(
    source, legacy_dtype, indexed_dtype
):
    legacy_scale = {"dtype": legacy_dtype, "block_size": 32} if legacy_dtype else None
    srcs = [{"dtype": "fp8", "scale": legacy_scale} for _ in range(2)]
    srcs.append({"dtype": "fp32"})
    srcs[source]["scale"] = (
        {"dtype": indexed_dtype, "block_size": 32} if indexed_dtype else None
    )
    row = dict(
        family="mma",
        op_id="fixture",
        srcs=srcs,
        dst={"dtype": "fp32"},
        m=16,
        n=16,
        k=128,
        a_scale_dtype=legacy_dtype,
        b_scale_dtype=legacy_dtype,
        scale_block_k=32 if legacy_dtype else None,
    )
    if (legacy_dtype is None) != (indexed_dtype is None):
        with pytest.raises(ValueError, match="conflicting indexed and legacy"):
            _build_mma_op(row)
    else:
        atom = _build_mma_op(row)
        assert all(
            src.scale == (MmaScaleOperand(legacy_dtype, 32) if legacy_dtype else None)
            for src in atom.srcs[:2]
        )


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


@pytest.mark.parametrize(
    "method", ["enumerate", "has_shape", "op_for_shape", "select_largest_k"]
)
@pytest.mark.parametrize("source", [0, 1, 2])
@pytest.mark.parametrize("legacy", [None, "alias", "conflict"])
def test_duplicate_matrix_query_contracts(method, source, legacy):
    atom = ArchTarget.from_gfx("gfx942").mma.op_for_shape(
        a_dtype="tf32", b_dtype="tf32", c_dtype="fp32", m=16, n=16, k=8
    )
    query = dict(src_dtypes=("tf32", "tf32", "fp32"), m=16, n=16)
    query[("a_dtype", "b_dtype", "c_dtype")[source]] = (
        None
        if legacy is None
        else (" XF32 ", " XF32 ", " F32 ")[source] if legacy == "alias" else "i32"
    )
    if method in ("has_shape", "op_for_shape"):
        query["k"] = 8
    call = getattr(MmaCatalog([atom]), method)
    if legacy == "conflict":
        with pytest.raises(ValueError, match="conflicting indexed and legacy matrix"):
            call(**query)
    else:
        result = call(**query)
        assert result == (
            [atom] if method == "enumerate" else True if method == "has_shape" else atom
        )


def _duplicate_matrix_row():
    return dict(
        family="mma",
        op_id="fixture",
        m=16,
        n=16,
        k=8,
        a="tf32",
        b="tf32",
        c="fp32",
        srcs=[{"dtype": " XF32 "}, {"dtype": "tf32"}, {"dtype": " F32 "}],
        dst={"dtype": "i32"},
    )


def _read_fixture_row(row, reader):
    if reader == "builder":
        return _build_mma_op(row)
    _op_id_dst_dtype.cache_clear()
    try:
        with mock.patch(
            "rocke.core.arch.target._load_specs",
            return_value={"fixture": {"mma": [row]}},
        ):
            return _op_id_dst_dtype()[row["op_id"]]
    finally:
        _op_id_dst_dtype.cache_clear()


@pytest.mark.parametrize("reader", ["builder", "raw"])
@pytest.mark.parametrize("source", [0, 1, 2])
def test_duplicate_matrix_rows_reject_conflicting_dtypes(reader, source):
    row = _duplicate_matrix_row()
    row[("a", "b", "c")[source]] = "fp16"
    with pytest.raises(ValueError, match="conflicting indexed and legacy matrix"):
        _read_fixture_row(row, reader)


@pytest.mark.parametrize("reader", ["builder", "raw"])
def test_duplicate_matrix_rows_preserve_aliases_and_independent_result(reader):
    row = _duplicate_matrix_row()
    result = _read_fixture_row(row, reader)
    if reader == "builder":
        assert result.srcs[2].dtype == "fp32"
        assert result.dst.dtype == result.c_dtype == "i32"
    else:
        assert result == "i32"
    del row["dst"]
    result = _read_fixture_row(row, reader)
    assert (result.dst.dtype if reader == "builder" else result) == "fp32"


@pytest.mark.parametrize("reader", ["builder", "raw"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("srcs", None),
        ("srcs", []),
        ("srcs", [{"dtype": "tf32"}]),
        ("srcs", [{"dtype": "tf32"}, None, {"dtype": "fp32"}]),
        ("srcs", [{"dtype": "tf32"}, {}, {"dtype": "fp32"}]),
        ("srcs", [{"dtype": "tf32"}, {"dtype": None}, {"dtype": "fp32"}]),
        ("dst", None),
        ("dst", {}),
        ("dst", {"dtype": None}),
    ],
)
def test_matrix_row_structure_rejected_before_scale_merge(reader, field, value):
    row = _duplicate_matrix_row()
    row[field] = value
    row.update(a_scale_dtype="e8m0", b_scale_dtype="e8m0", scale_block_k=32)
    with pytest.raises(ValueError, match="MMA catalog"):
        _read_fixture_row(row, reader)


@pytest.mark.parametrize("dtype", ["garbage", ""])
def test_raw_mma_rejects_present_unsupported_destination_dtype(dtype):
    atom = ArchTarget.from_gfx("gfx942").mma.op_for_shape(
        a_dtype="fp32", b_dtype="fp32", c_dtype="fp32", m=16, n=16, k=4
    )
    row = dict(
        family=atom.family,
        op_id=atom.op_id,
        m=atom.m,
        n=atom.n,
        k=atom.k,
        srcs=[{"dtype": s.dtype} for s in atom.srcs],
        dst={"dtype": dtype},
    )
    # Generic catalog metadata recognizes types independently of emission support.
    assert _build_mma_op(row).dst.dtype == dtype
    _op_id_dst_dtype.cache_clear()
    try:
        with mock.patch(
            "rocke.core.arch.target._load_specs",
            return_value={"fixture": {"mma": [row]}},
        ):
            b = IRBuilder("raw_bad_destination")
            value = b.const_i32(0)
            before = len(b.kernel.body.ops)
            with pytest.raises(ValueError, match="MMA destination dtype"):
                b.mma(atom.op_id, value, value, value)
            assert len(b.kernel.body.ops) == before
    finally:
        _op_id_dst_dtype.cache_clear()


@pytest.mark.parametrize(
    "dtype,element", [(" F32 ", "f32"), (" INT32 ", "i32"), (None, "f32")]
)
def test_raw_mma_preserves_aliases_and_absent_catalog_fallback(dtype, element):
    atom = ArchTarget.from_gfx("gfx942").mma.op_for_shape(
        a_dtype="fp32", b_dtype="fp32", c_dtype="fp32", m=16, n=16, k=4
    )
    lookup = {} if dtype is None else {atom.op_id: dtype}
    with mock.patch("rocke.core.arch.target._op_id_dst_dtype", return_value=lookup):
        b = IRBuilder("raw_destination")
        value = b.const_i32(0)
        result = b.mma(atom.op_id, value, value, value)
        assert result.type.elem.name == element
        assert result.type.count == atom.dst.frag_len


@pytest.mark.parametrize("reader", ["builder", "raw"])
def test_indexed_row_missing_destination_defaults_to_source_two(reader):
    row = _duplicate_matrix_row()
    del row["dst"]
    del row["c"]
    implicit = _read_fixture_row(row, reader)
    explicit = _read_fixture_row({**row, "dst": {"dtype": "fp32"}}, reader)
    assert implicit == explicit


def test_rdna_integer_wmma_rows_name_i8_and_i4():
    # The instruction names keep "iu8"/"iu4"; the operand dtypes are i8/i4, and only
    # i8/i4 select the ops ("iu8"/"iu4" are not dtypes).
    for gfx in ("gfx1151", "gfx11-generic"):
        mma = ArchTarget.from_gfx(gfx).mma
        for op_id, dtype, old in (
            ("wmma_i32_16x16x16_iu8", "i8", "iu8"),
            ("wmma_i32_16x16x16_iu4", "i4", "iu4"),
        ):
            op = mma.by_op_id(op_id)
            assert (op.a_dtype, op.b_dtype, op.c_dtype) == (dtype, dtype, "i32")
            for spelling, want in ((dtype, op_id), (old, None)):
                found = mma.op_for_shape(
                    family="wmma",
                    a_dtype=spelling,
                    b_dtype=spelling,
                    c_dtype="i32",
                    m=16,
                    n=16,
                    k=16,
                )
                assert (found and found.op_id) == want
