# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SSOT guards for indexed MMA ``srcs`` and ``dst`` metadata.

``IRBuilder.mma`` uses ``dst`` metadata to size a ``tile.mma`` result. These
tests pin the ``dst`` lookup, indexed catalog query, and optional per-source
scale parsing without assuming ``src2`` and ``dst`` are always identical.
"""

from __future__ import annotations

import unittest
from unittest import mock

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
        for family, block_size in (("wmma_scale", 32), ("wmma_scale16", 16)):
            with self.subTest(family=family):
                op = catalog.by_op_id(f"{family}_f32_16x16x128_fp8_fp8")
                self.assertIsNotNone(op)
                self.assertEqual(op.family, family)
                self.assertEqual(op.shape, (16, 16, 128))
                for src in op.srcs[:2]:
                    self.assertEqual(src.dtype, "fp8e4m3")
                    self.assertEqual(src.frag_len, 16)
                    self.assertEqual(src.scale, MmaScaleOperand("e8m0", block_size))
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
