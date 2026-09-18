# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""FP6 packing and mixed-format native WMMA with E8M0 scales."""

import hashlib
import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from rocke.core.arch import ArchTarget
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.examples.gfx1250.gemm.block_scaled_gemm_verify import (
    decode_fp6,
    pack_fp6_codes,
)
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    block_scaled_gemm_signature,
    build_block_scaled_gemm,
    is_valid_spec,
)

FORMATS = ("fp8", "bf8", "fp6", "bf6", "fp4")


def spec_for(a="fp6", b="bf6", mode="wmma_scale", **kwargs):
    return BlockScaledGemmSpec(
        name="fp6_test",
        M=32,
        N=48,
        K=256,
        dtype_a=a,
        dtype_b=b,
        scale_dtype="e8m0",
        matrix_path=mode,
        block_k=16 if mode == "wmma_scale16" else 32,
        **kwargs,
    )


@pytest.mark.parametrize(
    "a,b,mode", list(product(FORMATS, FORMATS, ("wmma_scale", "wmma_scale16")))
)
def test_all_native_matrix_pairs(a, b, mode):
    spec = spec_for(a, b, mode)
    assert is_valid_spec(spec)[0]
    atom = ArchTarget.from_gfx("gfx1250").mma.by_op_id(f"{mode}_f32_16x16x128_{a}_{b}")
    assert (atom.a_frag_len, atom.b_frag_len, atom.c_frag_len) == (16, 16, 8)
    kernel = build_block_scaled_gemm(spec)
    llvm = lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor="llvm23")
    call = next(
        l for l in llvm.splitlines() if "call <8 x float> @llvm.amdgcn.wmma.scale" in l
    )
    assert f"i32 {FORMATS.index(a)}, <16 x i32>" in call
    assert f"i32 {FORMATS.index(b)}, <16 x i32>" in call
    hip = lower_kernel_to_hip(kernel, arch="gfx1250")
    assert f"__builtin_amdgcn_{mode}_f32_16x16x128_f8f6f4({FORMATS.index(a)}," in hip
    for flavor in ("llvm20", "llvm22"):
        with pytest.raises(NotImplementedError, match="requires llvm23"):
            lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor=flavor)
    if a in ("fp6", "bf6"):
        assert block_scaled_gemm_signature(spec)[0]["type"] == "ptr<i8, global>"


@pytest.mark.parametrize(
    "case,expected_sha",
    json.loads(
        Path(__file__).with_name("gfx1250_scaled_wmma_llvm23.json").read_text()
    ).items(),
)
def test_scaled_wmma_llvm23_golden(case, expected_sha):
    mode, a, b, sa, sb = case.split("/")
    assert sa == sb == "e8m0"
    spec = spec_for(a, b, mode)
    llvm = lower_kernel_to_llvm(
        build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
    )
    assert hashlib.sha256(llvm.encode()).hexdigest() == expected_sha
