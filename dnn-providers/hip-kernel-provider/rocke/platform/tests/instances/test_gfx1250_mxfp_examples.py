# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Focused examples build real kernels and constrain their input contracts."""

import argparse
import importlib

import pytest

from rocke.core.arch.target import normalize_dtype
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.examples.gfx1250.gemm import _scaled_gemm_example
from rocke.instances.gfx1250.block_scaled_gemm import build_block_scaled_gemm


@pytest.mark.parametrize("family,dtype", [("mxfp8", "fp8"), ("mxfp4", "fp4")])
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


def test_example_all_cases_and_hip_route(monkeypatch):
    from rocke.examples.gfx1250.gemm import mxfp4_gemm

    seen = []

    def run(spec, cases, **kwargs):
        seen.append((spec, cases, kwargs))
        return len(cases)

    monkeypatch.setattr(_scaled_gemm_example, "run_cases", run)
    assert (
        mxfp4_gemm.main(
            ["--matrix-path", "wmma_scale16", "--compile-route", "hip", "--case", "all"]
        )
        == 0
    )
    spec, cases, kwargs = seen[0]
    assert spec.block_k == 16
    assert cases == ("neutral", "a-only", "b-only", "mixed") + tuple(
        f"group-{g}" for g in range(16)
    )
    assert kwargs == {"compile_route": "hip"}
