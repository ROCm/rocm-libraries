# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Public dtype aliases resolve to explicit catalog keys without changing ops."""

import pytest
from rocke.core.arch import ArchTarget
from rocke.core.dtypes import normalize_dtype


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("fp8", "fp8e4m3"),
        ("bf8", "bf8e5m2"),
        ("fp6", "fp6e2m3"),
        ("bf6", "fp6e3m2"),
        ("fp4", "fp4e2m1"),
    ],
)
def test_alias_catalog_lookup(alias, canonical):
    assert normalize_dtype(alias) == canonical
    assert normalize_dtype(" " + alias.upper() + " ") == canonical
    assert normalize_dtype(canonical) == canonical
    found = False
    for gfx in ("gfx950", "gfx1250"):
        catalog = ArchTarget.from_gfx(gfx).mma
        for op in catalog.ops:
            assert op.a_dtype not in ("fp8", "bf8", "fp6", "bf6", "fp4")
            assert op.b_dtype not in ("fp8", "bf8", "fp6", "bf6", "fp4")
            if op.a_dtype != canonical:
                continue
            found = True
            assert catalog.has_shape(
                family=op.family,
                scales=(op.a_scale_dtype, op.b_scale_dtype, op.scale_block_k),
                a_dtype=alias,
                b_dtype=op.b_dtype,
                c_dtype=op.c_dtype,
                m=op.m,
                n=op.n,
                k=op.k,
            )
    # BF6 name recognition does not require a corresponding catalog atom.
    if alias != "bf6":
        assert found


@pytest.mark.parametrize("dtype", ["fp4", "fp4e2m1", "fp6", "fp6e2m3", "fp16"])
def test_fp8_example_rejects_other_families(dtype):
    from rocke.examples.gfx1250.gemm import mxfp8_gemm

    with pytest.raises(SystemExit) as exc:
        mxfp8_gemm.main(["--dtype", dtype])
    assert exc.value.code == 2


def test_supported_aliases_preserve_lowered_kernel():
    from dataclasses import replace
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.core.lower_hip import lower_kernel_to_hip
    from rocke.instances.gfx1250.block_scaled_gemm import (
        _LOWBIT_DTYPES,
        BlockScaledGemmSpec,
        build_block_scaled_gemm,
        block_scaled_gemm_signature,
    )

    for short in ("fp8", "bf8", "fp6", "bf6", "fp4"):
        canonical = normalize_dtype(short)
        if canonical not in _LOWBIT_DTYPES:
            continue
        for mode, block_k in (("wmma_scale", 32), ("wmma_scale16", 16)):
            spec = BlockScaledGemmSpec(
                name="alias_identity",
                M=32,
                N=48,
                K=256,
                dtype_a=short,
                dtype_b=short,
                scale_dtype="e8m0",
                matrix_path=mode,
                block_k=block_k,
            )
            explicit = replace(spec, dtype_a=canonical, dtype_b=canonical)
            mixed_case = replace(spec, dtype_a=" " + short.upper() + " ", dtype_b=short)
            assert spec == explicit == mixed_case
            assert spec.dtype_a == spec.dtype_b == canonical
            assert hash(spec) == hash(explicit) == hash(mixed_case)
            a, b = build_block_scaled_gemm(spec), build_block_scaled_gemm(explicit)
            assert block_scaled_gemm_signature(spec) == block_scaled_gemm_signature(
                explicit
            )
            assert lower_kernel_to_llvm(
                a, arch="gfx1250", llvm_flavor="llvm23"
            ) == lower_kernel_to_llvm(b, arch="gfx1250", llvm_flavor="llvm23")
            assert lower_kernel_to_hip(a, arch="gfx1250") == lower_kernel_to_hip(
                b, arch="gfx1250"
            )
