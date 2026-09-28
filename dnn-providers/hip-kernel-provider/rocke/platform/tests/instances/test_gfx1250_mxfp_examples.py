# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Focused examples build real kernels and constrain their input contracts."""

import argparse
import importlib

import pytest

from rocke.core.arch.target import normalize_dtype
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.gfx1250.block_scaled_gemm import build_block_scaled_gemm


@pytest.mark.parametrize("family,dtype", [("mxfp8", "fp8")])
@pytest.mark.parametrize("path,block_k", [("wmma_scale", 32), ("wmma_scale16", 16)])
def test_example_contract_and_lowering(family, dtype, path, block_k):
    example = importlib.import_module(f"rocke.examples.gfx1250.gemm.{family}_gemm")
    args = argparse.Namespace(m=32, n=48, k=256, matrix_path=path, dtype=dtype)
    spec = example.make_spec(args)
    assert (spec.dtype_a, spec.dtype_b, spec.scale_dtype) == (
        normalize_dtype(dtype),
        normalize_dtype(dtype),
        "e8m0",
    )
    assert spec.block_k == block_k
    llvm = lower_kernel_to_llvm(
        build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
    )
    assert "@llvm.amdgcn.wmma.scale" in llvm
