# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Scale legality, independent selectors, and encoded-byte reference checks."""

import hashlib
from itertools import product
import json
from pathlib import Path
from dataclasses import replace
import re

import numpy as np
import pytest

from rocke.core.arch import ArchTarget
from rocke.core.arch.wmma_scale import gfx1250_scaled_wmma
from rocke.core.backend import resolve_backend
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.examples.gfx1250.gemm.block_scaled_gemm_verify import (
    decode_scale,
    make_case_inputs,
    reference_result,
)
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
    is_valid_spec,
)

FORMATS = ("fp8", "bf8", "fp6", "bf6", "fp4")
SCALES = ("e8m0", "e5m3", "e4m3")
PAIRS = (
    [(d, "fp4", "e8m0", s) for d in FORMATS[:-1] for s in SCALES[1:]]
    + [("fp4", d, s, "e8m0") for d in FORMATS[:-1] for s in SCALES[1:]]
    + [("fp4", "fp4", s, s) for s in SCALES[1:]]
)


def spec_for(a="fp4", b="fp4", sa="e4m3", sb="e4m3", block_k=32, **kwargs):
    return BlockScaledGemmSpec(
        name="scale_formats_test",
        M=32,
        N=48,
        K=256,
        dtype_a=a,
        dtype_b=b,
        scale_dtype="e8m0",
        scale_dtype_a=sa,
        scale_dtype_b=sb,
        matrix_path="wmma_scale16" if block_k == 16 else "wmma_scale",
        block_k=block_k,
        **kwargs,
    )


@pytest.mark.parametrize("a,b,sa,sb", list(product(FORMATS, FORMATS, SCALES, SCALES)))
@pytest.mark.parametrize("block_k", [32, 16])
def test_catalog_and_builder_agree_on_legal_scale_combinations(a, b, sa, sb, block_k):
    expected = (
        (sa == "e8m0" or a == "fp4")
        and (sb == "e8m0" or b == "fp4")
        and (a != "fp4" or b != "fp4" or sa == sb)
    )
    atom = ArchTarget.from_gfx("gfx1250").mma.op_for_shape(
        family="wmma_scaled",
        a_dtype=a,
        b_dtype=b,
        c_dtype="fp32",
        m=16,
        n=16,
        k=128,
        scales=(sa, sb, block_k),
    )
    assert (atom is not None) == expected
    assert is_valid_spec(spec_for(a, b, sa, sb, block_k))[0] == expected


@pytest.mark.parametrize("a,b,sa,sb", PAIRS)
@pytest.mark.parametrize("block_k", [32, 16])
@pytest.mark.parametrize("dtype_c", ["bf16", "fp16"])
def test_scale_formats_lowering_and_reference(a, b, sa, sb, block_k, dtype_c):
    spec = spec_for(a, b, sa, sb, block_k, dtype_c=dtype_c)
    atom = ArchTarget.from_gfx("gfx1250").mma.op_for_shape(
        family="wmma_scaled",
        a_dtype=a,
        b_dtype=b,
        c_dtype="fp32",
        m=16,
        n=16,
        k=128,
        scales=(sa, sb, block_k),
    )
    assert (
        atom.op_id == f"wmma_gfx1250_f32_16x16x128_{a}_{b}_scale_{sa}_{sb}_k{block_k}"
    )
    contract = gfx1250_scaled_wmma(atom.op_id)
    assert contract.scale_formats == (SCALES.index(sa), SCALES.index(sb))
    kernel = build_block_scaled_gemm(spec)
    llvm = lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor="llvm23")
    calls = [
        line
        for line in llvm.splitlines()
        if "call <8 x float> @llvm.amdgcn.wmma.scale" in line
    ]
    assert len(calls) == 2
    for call in calls:
        assert re.findall(r"i32 0, i32 (\d+), i(?:32|64) %", call) == [
            str(SCALES.index(sa)),
            str(SCALES.index(sb)),
        ]
    hip = lower_kernel_to_hip(kernel, arch="gfx1250")
    hip_calls = [
        line for line in hip.splitlines() if "__builtin_amdgcn_wmma_scale" in line
    ]
    assert len(hip_calls) == 2
    for call in hip_calls:
        assert re.findall(r", 0, (\d+), ", call) == [
            str(SCALES.index(sa)),
            str(SCALES.index(sb)),
        ]
    for flavor in ("llvm20", "llvm22"):
        error = RuntimeError if resolve_backend() == "cpp" else NotImplementedError
        with pytest.raises(error, match="requires llvm23"):
            lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor=flavor)
    inputs = make_case_inputs(spec, "mixed")
    result = reference_result(
        *inputs,
        block_k,
        native=True,
        dtype_a=a,
        dtype_b=b,
        dtype_c=dtype_c,
        scale_dtype_a=sa,
        scale_dtype_b=sb,
    )
    assert np.isfinite(result).all() and np.any(result)


@pytest.mark.parametrize(
    "dtype,bias,max_code,max_value", [("e4m3", 7, 126, 448), ("e5m3", 15, 254, 114688)]
)
def test_scale_decoding(dtype, bias, max_code, max_value):
    codes = np.array([0, 1, 8, bias * 8, bias * 8 + 4, max_code], dtype=np.uint8)
    np.testing.assert_array_equal(
        decode_scale(codes, dtype),
        [0, 2.0 ** (-bias - 2), 2.0 ** (1 - bias), 1, 1.5, max_value],
    )
    with pytest.raises(ValueError):
        decode_scale(np.array([max_code + 1], dtype=np.uint8), dtype)


def test_e4m3_decoder_against_independent_dtype():
    ml = pytest.importorskip("ml_dtypes")
    codes = np.arange(127, dtype=np.uint8)
    np.testing.assert_array_equal(
        decode_scale(codes, "e4m3"), codes.view(ml.float8_e4m3fn).astype(np.float64)
    )


@pytest.mark.parametrize("dtype", ["e5m2", "bf8e5m2", "fp8e5m2", "bf8", "", "i32"])
@pytest.mark.parametrize("field", ["scale_dtype_a", "scale_dtype_b"])
def test_invalid_scale_spellings(dtype, field):
    spec = replace(spec_for(), **{field: dtype})
    assert not is_valid_spec(spec)[0]
    with pytest.raises(ValueError):
        build_block_scaled_gemm(spec)


def test_per_operand_overrides_preserve_defaults_and_aliases():
    implicit = spec_for(sa=None, sb=None)
    explicit = replace(implicit, scale_dtype_a="e8m0", scale_dtype_b="i8")
    assert implicit.kernel_name() == explicit.kernel_name()
    lower = lambda spec: lower_kernel_to_llvm(
        build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
    )
    assert lower(implicit) == lower(explicit)
    alias = replace(
        spec_for(), scale_dtype_a="fp8e4m3", scale_dtype_b=None, scale_dtype="e4m3"
    )
    assert alias.resolved_scale_dtypes() == ("e4m3", "e4m3")
    assert alias.kernel_name() == spec_for().kernel_name()
    assert lower(alias) == lower(spec_for())
    assert lower(implicit) != lower(spec_for())


def test_legacy_wmma_rejects_per_operand_scale_overrides():
    spec = replace(
        spec_for(),
        dtype_a="fp8",
        dtype_b="fp8",
        matrix_path="wmma",
        scale_dtype="fp32",
        block_k=128,
    )
    assert is_valid_spec(spec) == (
        False,
        "per-operand scale types require native scaled WMMA",
    )


@pytest.mark.parametrize(
    "case,expected_sha",
    json.loads(
        Path(__file__).with_name("gfx1250_scale_formats_llvm23.json").read_text()
    ).items(),
)
def test_scale_formats_golden(case, expected_sha):
    a, b, sa, sb, block_k = case.split("/")
    llvm = lower_kernel_to_llvm(
        build_block_scaled_gemm(spec_for(a, b, sa, sb, int(block_k))),
        arch="gfx1250",
        llvm_flavor="llvm23",
    )
    assert hashlib.sha256(llvm.encode()).hexdigest() == expected_sha


@pytest.mark.parametrize(
    "a,b,sa,sb", [("fp8", "fp4", "e4m3", "e8m0"), ("fp4", "fp4", "e4m3", "e5m3")]
)
def test_backend_rejects_invalid_contract_even_if_catalog_injected(
    monkeypatch, a, b, sa, sb
):
    from rocke.core.arch.target import MmaCatalog

    atom = ArchTarget.from_gfx("gfx1250").mma.op_for_shape(
        family="wmma_scaled",
        a_dtype=a,
        b_dtype=b,
        c_dtype="fp32",
        m=16,
        n=16,
        k=128,
        scales=("e8m0", "e8m0", 32),
    )
    invalid = replace(atom, a_scale_dtype=sa, b_scale_dtype=sb)
    monkeypatch.setattr(MmaCatalog, "by_op_id", lambda self, op_id: invalid)
    with pytest.raises(ValueError, match="unsupported scaled WMMA backend contract"):
        gfx1250_scaled_wmma(atom.op_id)
