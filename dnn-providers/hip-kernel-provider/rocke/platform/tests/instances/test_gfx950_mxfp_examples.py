# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import argparse
import importlib
from itertools import product

import numpy as np
import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.examples.common.scaled_gemm_reference import reference_result
from rocke.examples.gfx950.gemm.block_scaled_gemm_verify import (
    assert_scale_coverage,
    make_inputs,
)
from rocke.instances.gfx950.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
)

FORMATS = ("fp8", "bf8", "fp6", "bf6", "fp4")


@pytest.mark.parametrize(
    "family,dtype",
    [
        ("mxfp8", "fp8"),
        ("mxfp8", "bf8"),
        ("mxfp6", "fp6"),
        ("mxfp6", "bf6"),
        ("mxfp4", "fp4"),
    ],
)
@pytest.mark.parametrize("output", ["bf16", "fp16"])
def test_example_builds_real_kernel(family, dtype, output):
    ex = importlib.import_module(f"rocke.examples.gfx950.gemm.{family}_gemm")
    spec = ex.make_spec(
        argparse.Namespace(m=32, n=64, k=128, dtype=dtype, output_dtype=output)
    )
    assert spec.block_k == 32 and spec.block_size == 64
    assert "call <16 x float> @llvm.amdgcn.mfma.scale" in _lower_kernel_to_llvm_python(
        build_block_scaled_gemm(spec), arch="gfx950", llvm_flavor="llvm23"
    )


@pytest.mark.parametrize(
    "family,count", [("mxfp8", 2), ("mxfp6", 2), ("mxfp4", 1), ("mixed_scaled", 1)]
)
def test_default_and_all_case_cli(monkeypatch, family, count):
    from rocke.examples.gfx950.gemm import _scaled_gemm_example as common

    ex = importlib.import_module(
        f"rocke.examples.gfx950.gemm.{family}_gemm"
        if family != "mixed_scaled"
        else "rocke.examples.gfx950.gemm.mixed_scaled_gemm"
    )
    seen = []
    monkeypatch.setattr(
        common,
        "run_cases",
        lambda spec, cases, **kw: seen.append((spec, cases, kw)) or len(cases),
    )
    assert (
        ex.main(["--case", "all", "--output-dtype", "fp16", "--compile-route", "hip"])
        == 0
    )
    assert len(seen) == count
    for spec, cases, kw in seen:
        assert (spec.N, spec.dtype_c) == (64, "fp16")
        assert set(cases) >= {
            "a-only",
            "b-only",
            "mixed",
            "positive",
            "group-0",
            "group-7",
        }
        assert kw == {"compile_route": "hip"}


@pytest.mark.parametrize("a,b", list(product(FORMATS, repeat=2)))
@pytest.mark.parametrize("output", ["bf16", "fp16"])
def test_nontrivial_scale_coverage_and_oracle_sensitivity(a, b, output):
    spec = BlockScaledGemmSpec("fixture", 64, 96, 256, a, b, output)

    def reference(inputs):
        return reference_result(
            *inputs,
            32,
            native=True,
            dtype_a=spec.dtype_a,
            dtype_b=spec.dtype_b,
            dtype_c=spec.dtype_c,
        )

    x = make_inputs(spec, "positive")
    assert_scale_coverage(spec, "positive", *x[2:])
    sa, sb = x[2:]
    assert np.all(sa[:, 0::2] != sa[:, 1::2]) and np.all(sb[0::2] != sb[1::2])
    expected = reference(x)
    faults = [
        (np.full_like(sa, 127), sb),
        (sa, np.full_like(sb, 127)),
        (np.full_like(sa, 127), np.full_like(sb, 127)),
        (np.repeat(sa[:, 0::2], 2, axis=1), sb),
        (sa, np.repeat(sb[0::2], 2, axis=0)),
        (np.roll(sa, 1, axis=1), sb),
        (sa, np.roll(sb, 1, axis=0)),
        (np.roll(sa, 1, axis=0), sb),
        (sa, np.roll(sb, 1, axis=1)),
    ]
    for bad_a, bad_b in faults:
        assert np.any(reference((x[0], x[1], bad_a, bad_b)) != expected)
    for case in ("a-only", "b-only", "mixed", "positive", "group-0", "group-7"):
        y = make_inputs(spec, case)
        assert_scale_coverage(spec, case, *y[2:])
        assert np.isfinite(reference(y)).all()


def test_mixed_rejects_equal_aliases():
    from rocke.examples.gfx950.gemm.mixed_scaled_gemm import make_spec

    with pytest.raises(ValueError, match="different A/B"):
        make_spec(argparse.Namespace(dtype_a="fp8", dtype_b="fp8e4m3"))
