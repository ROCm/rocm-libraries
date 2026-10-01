# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Ordered mixed-format atoms retain independent operand storage contracts."""

import hashlib
from itertools import product
import json
from pathlib import Path
import re

import pytest

from rocke.core.arch import ArchTarget
from rocke.core.arch.wmma_scale import gfx1250_scaled_wmma
from rocke.core.backend import resolve_backend
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
    is_valid_spec,
)

FORMATS = ("fp8", "bf8", "fp6", "bf6", "fp4")
MIXED_PAIRS = [(a, b) for a, b in product(FORMATS, repeat=2) if a != b]


@pytest.mark.parametrize("a,b", MIXED_PAIRS)
@pytest.mark.parametrize("block_k", [32, 16])
def test_mixed_catalog_storage_and_lowering(a, b, block_k):
    mode = "wmma_scale16" if block_k == 16 else "wmma_scale"
    spec = BlockScaledGemmSpec(
        name="mixed_test",
        M=32,
        N=48,
        K=256,
        dtype_a=a,
        dtype_b=b,
        matrix_path=mode,
        scale_dtype="e8m0",
        block_k=block_k,
    )
    assert is_valid_spec(spec) == (True, f"ok: gfx1250 K=128 native {mode} {a} GEMM")
    atom = ArchTarget.from_gfx("gfx1250").mma.op_for_shape(
        family="wmma_scaled",
        a_dtype=a,
        b_dtype=b,
        c_dtype="fp32",
        m=16,
        n=16,
        k=128,
        scales=("e8m0", "e8m0", block_k),
    )
    assert (
        atom.op_id == f"wmma_gfx1250_f32_16x16x128_{a}_{b}_scale_e8m0_e8m0_k{block_k}"
    )
    contract = gfx1250_scaled_wmma(atom.op_id)
    widths = {"fp8": 8, "bf8": 8, "fp6": 6, "bf6": 6, "fp4": 4}
    for operand, dtype in (("a", a), ("b", b)):
        layout = contract.matrix_layout(operand)
        bits = widths[dtype]
        assert layout.fragment.packing.element_bits == bits
        assert layout.fragment.live_carriers == bits * 2
        assert layout.fragment.padding_bits == (8 - bits) * 64
        assert layout.chunk_bytes == (16 if bits in (4, 8) else 24)
        assert contract.scale_packing(operand).count == 128 // block_k
    kernel = build_block_scaled_gemm(spec)
    llvm = lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor="llvm23")
    calls = [
        line
        for line in llvm.splitlines()
        if "call <8 x float> @llvm.amdgcn.wmma.scale" in line
    ]
    assert len(calls) == 2
    for call in calls:
        # Match selector positions in order; a set-membership check misses swaps.
        assert re.findall(r"i32 (\d+), <16 x i32>", call) == [
            str(FORMATS.index(a)),
            str(FORMATS.index(b)),
        ]
    hip = lower_kernel_to_hip(kernel, arch="gfx1250")
    builtin = f"__builtin_amdgcn_{mode}_f32_16x16x128_f8f6f4"
    assert f"{builtin}({FORMATS.index(a)}," in hip
    for flavor in ("llvm20", "llvm22"):
        error = RuntimeError if resolve_backend() == "cpp" else NotImplementedError
        with pytest.raises(error, match="requires llvm23"):
            lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor=flavor)


@pytest.mark.parametrize("a,b", MIXED_PAIRS)
def test_mixed_atoms_do_not_leak_to_other_targets(a, b):
    for arch in ("gfx942", "gfx950", "gfx1201"):
        assert (
            ArchTarget.from_gfx(arch).mma.op_for_shape(
                family="wmma_scaled",
                a_dtype=a,
                b_dtype=b,
                c_dtype="fp32",
                m=16,
                n=16,
                k=128,
                scales=("e8m0", "e8m0", 32),
            )
            is None
        )


@pytest.mark.parametrize("scales", [("e4m3", "e8m0", 32), ("e5m3", "e8m0", 16)])
def test_mixed_atoms_do_not_admit_other_scale_contracts(scales):
    assert (
        ArchTarget.from_gfx("gfx1250").mma.op_for_shape(
            family="wmma_scaled",
            a_dtype="fp8",
            b_dtype="fp4",
            c_dtype="fp32",
            m=16,
            n=16,
            k=128,
            scales=scales,
        )
        is None
    )


@pytest.mark.parametrize(
    "case,expected_sha",
    json.loads(
        Path(__file__).with_name("gfx1250_mixed_scaled_wmma_llvm23.json").read_text()
    ).items(),
)
def test_mixed_llvm23_golden(case, expected_sha):
    a, b, block_k = case.split("/")
    spec = BlockScaledGemmSpec(
        name="mixed_golden",
        M=32,
        N=48,
        K=256,
        dtype_a=a,
        dtype_b=b,
        matrix_path="wmma_scale16" if block_k == "16" else "wmma_scale",
        scale_dtype="e8m0",
        block_k=int(block_k),
    )
    llvm = lower_kernel_to_llvm(
        build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
    )
    assert hashlib.sha256(llvm.encode()).hexdigest() == expected_sha
