# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection must distinguish logical operands even when carriers agree."""

from dataclasses import replace

import pytest

from rocke.core.arch import (
    ArchTarget,
    MmaCatalog,
    MmaDst,
    MmaOp,
    MmaScaleOperand,
    MmaSrc,
)
from rocke.core.arch.wmma_scale import gfx1250_scaled_wmma


def _contracts():
    base = MmaOp(
        family="wmma_scaled",
        srcs=(
            MmaSrc("fp8", scale=MmaScaleOperand("e8m0", 32)),
            MmaSrc("bf8", scale=MmaScaleOperand("e8m0", 32)),
            MmaSrc("fp32"),
        ),
        dst=MmaDst("fp32"),
        m=16,
        n=16,
        k=128,
        op_id="fixture",
    )
    rows = [base]
    for index in (0, 1):
        for scale in (MmaScaleOperand("e4m3", 32), MmaScaleOperand("e8m0", 16), None):
            srcs = list(base.srcs)
            srcs[index] = replace(srcs[index], scale=scale)
            rows.append(replace(base, srcs=tuple(srcs)))
    rows.append(replace(base, srcs=(*base.srcs[:2], MmaSrc("fp16"))))
    rows.append(replace(base, dst=MmaDst("fp16")))
    rows.append(
        replace(base, srcs=tuple(replace(src, scale=None) for src in base.srcs))
    )
    return [replace(row, op_id=row.semantic_id()) for row in rows]


def test_full_contract_queries_distinguish_each_source_scale_and_result():
    rows = _contracts()
    assert len({row.op_id for row in rows}) == len(rows)
    catalog = MmaCatalog(rows)
    for row in rows:
        query = dict(
            family=row.family,
            src_dtypes=tuple(src.dtype for src in row.srcs),
            dst_dtype=row.dst.dtype,
            src_scales=tuple(src.scale for src in row.srcs),
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
        src_dtypes=("fp8", "bf8", "fp32"),
        dst_dtype="fp32",
        m=16,
        n=16,
    )
    assert len(catalog.enumerate(**query)) == 8
    assert catalog.has_shape(**query, k=128)
    with pytest.raises(ValueError, match="ambiguous MMA query"):
        catalog.op_for_shape(**query, k=128)
    with pytest.raises(ValueError, match="ambiguous MMA query"):
        catalog.select_largest_k(**query)
    assert len(catalog.enumerate(**query, src_scales=(None, None, None))) == 1
    scale = MmaScaleOperand("e5m3", 32)
    assert catalog.op_for_shape(**query, src_scales=(scale, scale, None), k=128) is None
    with pytest.raises(ValueError, match="exactly 3"):
        catalog.enumerate(**query, src_scales=(None, None))


def test_scale_alias_selects_the_same_contract():
    catalog = MmaCatalog(_contracts())
    query = dict(
        family="wmma_scaled",
        src_dtypes=("fp8", "bf8", "f32"),
        dst_dtype="fp32",
        m=16,
        n=16,
        k=128,
    )
    scales = (MmaScaleOperand("fp8e4m3", 32), MmaScaleOperand("e8m0", 32), None)
    row = catalog.op_for_shape(**query, src_scales=scales)
    assert row is not None and row.srcs[0].scale.dtype == "e4m3"


def test_scaled_catalog_identity_and_backend_contract():
    catalog = ArchTarget.from_gfx("gfx1250").mma
    rows = [row for row in catalog.ops if row.family == "wmma_scaled"]
    assert len(rows) == 4
    for row in rows:
        assert row.op_id == row.semantic_id()
        packing = gfx1250_scaled_wmma(row.op_id)
        assert packing.atom is row
        assert packing.matrix_formats == (
            (0, 0) if row.srcs[0].dtype == "fp8e4m3" else (1, 1)
        )
        assert packing.scales.count * packing.scales.block_k == row.k
        assert packing.matrix_llvm_types == ("<16 x i32>", "<16 x i32>")
    for family in ("wmma_scale", "wmma_scale16"):
        old_id = f"{family}_f32_16x16x128_fp8_fp8"
        assert catalog.by_op_id(old_id) is None
        assert gfx1250_scaled_wmma(old_id) is None
