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


@pytest.mark.parametrize(
    "family,dtype",
    [("mxfp8", "fp8"), ("mxfp4", "fp4"), ("mxfp6", "fp6"), ("mxfp6", "bf6")],
)
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


@pytest.mark.parametrize(
    "a,b", [("fp8e4m3", "fp4"), ("fp4", "fp8e4m3"), ("fp6", "bf6")]
)
def test_mixed_example_independent_operands(a, b):
    from rocke.examples.gfx1250.gemm.mixed_scaled_gemm import make_spec

    args = argparse.Namespace(
        m=32, n=48, k=256, matrix_path="wmma_scale16", dtype_a=a, dtype_b=b
    )
    spec = make_spec(args)
    assert (spec.dtype_a, spec.dtype_b, spec.scale_dtype) == (
        normalize_dtype(a),
        normalize_dtype(b),
        "e8m0",
    )
    assert "@llvm.amdgcn.wmma.scale16" in lower_kernel_to_llvm(
        build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
    )


@pytest.mark.parametrize("a,b", [("fp4", "fp4e2m1"), ("fp6", "fp6e2m3")])
def test_mixed_example_rejects_equal_canonical_formats(a, b):
    from rocke.examples.gfx1250.gemm.mixed_scaled_gemm import make_spec

    with pytest.raises(ValueError, match="homogeneous"):
        make_spec(argparse.Namespace(dtype_a=a, dtype_b=b))


@pytest.mark.parametrize(
    "dtype",
    [
        "fp8",
        "bf8",
        "fp8e4m3",
        "bf8e5m2",
        "fp6",
        "bf6",
        "fp6e2m3",
        "fp6e3m2",
        "fp4",
        "fp4e2m1",
    ],
)
@pytest.mark.parametrize("operand", ["a", "b"])
def test_mixed_scaled_gemm_cli_normalizes_matrix_names(monkeypatch, dtype, operand):
    from rocke.examples.gfx1250.gemm import mixed_scaled_gemm as example

    calls = []

    def verify(spec, args):
        calls.append(spec)
        llvm = lower_kernel_to_llvm(
            build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
        )
        assert "@llvm.amdgcn.wmma.scale" in llvm
        return 0

    monkeypatch.setattr(example, "verify", verify)
    other = "fp8" if normalize_dtype(dtype) == "fp4e2m1" else "fp4"
    argv = ["--dtype-a", other, "--dtype-b", other]
    argv += [f"--dtype-{operand}", dtype]
    assert example.main(argv) == 0
    assert len(calls) == 1
    assert getattr(calls[0], f"dtype_{operand}") == normalize_dtype(dtype)


@pytest.mark.parametrize("dtype,alias", [("fp8", "fp8e4m3"), ("bf8", "bf8e5m2")])
def test_mixed_cli_rejects_equal_eight_bit_aliases(capsys, dtype, alias):
    from rocke.examples.gfx1250.gemm.mixed_scaled_gemm import main

    with pytest.raises(SystemExit) as error:
        main(["--dtype-a", dtype, "--dtype-b", alias])
    assert error.value.code == 2
    assert "choose different A/B formats" in capsys.readouterr().err
