# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SSOT guards for indexed MMA sources and destination metadata.

``IRBuilder.mma`` uses destination metadata to size a ``tile.mma`` result. These
tests pin the destination lookup, indexed catalog query, and optional per-source
scale parsing without assuming source 2 and destination are always identical.
"""

from __future__ import annotations

import unittest
from unittest import mock

from rocke.core.arch.target import (
    ArchTarget,
    MmaCatalog,
    MmaOp,
    MmaOperand,
    MmaScaleOperand,
    MmaSource,
    _build_mma_op,
    _load_specs,
    _op_id_dst_dtype,
    normalize_dtype,
)
from rocke.core.ir import IRBuilder


class TestOpIdDstDtype(unittest.TestCase):
    def test_matches_catalog_first_hit(self):
        # Every op_id in the catalog resolves to its normalized accumulator dtype,
        # taking the first arch that lists it (dict preserves catalog order).
        expected: dict = {}
        for row in _load_specs().values():
            for o in row["mma"]:
                expected.setdefault(o["op_id"], normalize_dtype(o["dst"]["dtype"]))
        self.assertEqual(_op_id_dst_dtype(), expected)

    def test_dst_dtype_invariant_across_arches(self):
        # The whole premise of the bare-op_id lookup: an op_id's destination dtype
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

    def test_indexed_query_distinguishes_source2_and_destination(self):
        distinct = MmaOp(
            family="mma",
            srcs=(MmaSource("xf32"), MmaSource("xf32"), MmaSource("fp32")),
            dst=MmaOperand("i32", frag_len=7),
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

    def test_scale_is_optional_and_attached_to_its_source(self):
        op = _build_mma_op(
            {
                "family": "mma",
                "srcs": [
                    {"dtype": "fp4", "scale": {"dtype": "i32"}},
                    {"dtype": "fp4", "scale": {"dtype": "i32"}},
                    {"dtype": "fp32"},
                ],
                "dst": {"dtype": "fp32"},
                "m": 16,
                "n": 16,
                "k": 128,
                "op_id": "mfma_scale_f32_16x16x128_f8f6f4",
            }
        )
        self.assertEqual(op.srcs[0].scale, MmaScaleOperand("i32"))
        self.assertEqual(op.srcs[1].scale, MmaScaleOperand("i32"))
        self.assertIsNone(op.srcs[2].scale)
        self.assertFalse(hasattr(op.dst, "scale"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
