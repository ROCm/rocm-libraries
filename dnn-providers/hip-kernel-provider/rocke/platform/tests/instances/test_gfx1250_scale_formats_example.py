# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Scale example preserves independent selectors through real lowering."""

import argparse
import pytest
from rocke.examples.gfx1250.gemm.scale_formats_gemm import make_spec
from rocke.instances.gfx1250.block_scaled_gemm import build_block_scaled_gemm, is_valid_spec
from rocke.core.lower_llvm import lower_kernel_to_llvm

@pytest.mark.parametrize("a,b,sa,sb", [("fp4", "fp4", "e4m3", "e4m3"), ("fp6", "fp4", "e8m0", "e5m3")])
def test_scale_example_contract(a, b, sa, sb):
    spec = make_spec(argparse.Namespace(m=32, n=48, k=256, matrix_path="wmma_scale16", dtype_a=a, dtype_b=b, scale_dtype_a=sa, scale_dtype_b=sb))
    assert spec.resolved_scale_dtypes() == (sa, sb)
    assert is_valid_spec(spec)[0]
    assert "@llvm.amdgcn.wmma.scale16" in lower_kernel_to_llvm(build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor="llvm23")
