# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Focused examples build real kernels and constrain their input contracts."""

import argparse
import importlib
import subprocess
import sys

import pytest

from rocke.core.arch.target import normalize_dtype
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.examples.gfx1250.gemm import _scaled_gemm_example
from rocke.instances.gfx1250.block_scaled_gemm import build_block_scaled_gemm


@pytest.mark.parametrize(
    "family,dtype", [("mxfp8", "fp8"), ("mxfp6", "fp6"), ("mxfp4", "fp4")]
)
@pytest.mark.parametrize("path,block_k", [("wmma_scale", 32), ("wmma_scale16", 16)])
@pytest.mark.parametrize("output_dtype", ["bf16", "fp16"])
def test_example_contract_and_lowering(family, dtype, path, block_k, output_dtype):
    example = importlib.import_module(f"rocke.examples.gfx1250.gemm.{family}_gemm")
    args = argparse.Namespace(
        m=32, n=48, k=256, matrix_path=path, dtype=dtype, output_dtype=output_dtype
    )
    spec = example.make_spec(args)
    assert (spec.dtype_a, spec.dtype_b, spec.scale_dtype) == (
        normalize_dtype(dtype),
        normalize_dtype(dtype),
        "e8m0",
    )
    assert spec.block_k == block_k
    assert spec.dtype_c == output_dtype
    llvm = lower_kernel_to_llvm(
        build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
    )
    assert "@llvm.amdgcn.wmma.scale" in llvm
    storage_type = "half" if output_dtype == "fp16" else "bfloat"
    assert f"store {storage_type} " in llvm


@pytest.mark.parametrize("family", ["mxfp4", "mxfp6", "mxfp8"])
@pytest.mark.parametrize("output_dtype", [None, "bf16", "fp16"])
def test_example_cli_selects_output(monkeypatch, family, output_dtype):
    example = importlib.import_module(f"rocke.examples.gfx1250.gemm.{family}_gemm")
    seen = []

    def run(spec, cases, **kwargs):
        seen.append(spec.dtype_c)
        return len(cases)

    monkeypatch.setattr(_scaled_gemm_example, "run_cases", run)
    argv = [] if output_dtype is None else ["--output-dtype", output_dtype]
    assert example.main(argv) == 0
    assert seen == [output_dtype or "bf16"] * (1 if family == "mxfp4" else 2)


@pytest.mark.parametrize("family", ["mxfp4", "mxfp6", "mxfp8"])
def test_example_cli_rejects_unsupported_output(monkeypatch, family, capsys):
    example = importlib.import_module(f"rocke.examples.gfx1250.gemm.{family}_gemm")
    monkeypatch.setattr(example, "verify", lambda *args: pytest.fail("must not launch"))
    with pytest.raises(SystemExit) as exc:
        example.main(["--output-dtype", "fp32"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "family,dtype",
    [
        ("mxfp4", None),
        ("mxfp6", "fp6"),
        ("mxfp6", "bf6"),
        ("mxfp8", "fp8"),
        ("mxfp8", "bf8"),
    ],
)
@pytest.mark.parametrize("path", ["wmma_scale", "wmma_scale16"])
@pytest.mark.parametrize("route", ["comgr", "hip"])
def test_example_fp16_output_numeric(gpu_env, family, dtype, path, route):
    command = [
        sys.executable,
        "-m",
        f"rocke.examples.gfx1250.gemm.{family}_gemm",
        "--output-dtype",
        "fp16",
        "--matrix-path",
        path,
        "--compile-route",
        route,
        "--m",
        "32",
        "--n",
        "48",
        "--k",
        "256",
        "--case",
        "mixed",
    ]
    if dtype is not None:
        command.extend(["--dtype", dtype])
    result = subprocess.run(
        command, env=gpu_env, capture_output=True, text=True, timeout=300
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count("bad=0") == 1, output
    assert "PASS: verified 1 cases" in output, output
    print(output, end="")


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
