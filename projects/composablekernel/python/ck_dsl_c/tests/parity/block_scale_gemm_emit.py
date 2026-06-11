#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/block_scale_gemm_emit.py -- Python reference emitter for the
# block_scale_gemm parity harness. Selects one of N sampled spec configs by
# argv[1], builds the BlockScaleGemmSpec, builds the kernel via
# build_block_scale_gemm and prints lower_kernel_to_llvm(arch='gfx950') to stdout
# so it can be byte-compared with the C emitter block_scale_gemm_emit.c.
import sys

from ck_dsl.instances.common.block_scale_gemm import (
    BlockScaleGemmSpec,
    build_block_scale_gemm,
)
from ck_dsl import lower_kernel_to_llvm


def _spec(idx: int) -> BlockScaleGemmSpec:
    common = dict(quant_mode="abquant", block_tile_m=16, block_tile_n=16)
    if idx == 0:
        return BlockScaleGemmSpec(
            M=32,
            N=32,
            K=64,
            mantissa_dtype="fp8e4m3",
            group_size_mnk=(1, 1, 64),
            **common,
        )
    if idx == 1:
        return BlockScaleGemmSpec(
            M=64,
            N=64,
            K=128,
            mantissa_dtype="fp8e4m3",
            group_size_mnk=(1, 1, 128),
            **common,
        )
    if idx == 2:
        return BlockScaleGemmSpec(
            M=16,
            N=16,
            K=128,
            mantissa_dtype="bf8e5m2",
            group_size_mnk=(1, 1, 64),
            **common,
        )
    if idx == 3:
        return BlockScaleGemmSpec(
            M=128,
            N=128,
            K=256,
            mantissa_dtype="fp8e4m3",
            group_size_mnk=(1, 1, 256),
            **common,
        )
    if idx == 4:
        return BlockScaleGemmSpec(
            M=48,
            N=48,
            K=96,
            mantissa_dtype="bf8e5m2",
            group_size_mnk=(1, 1, 96),
            **common,
        )
    if idx == 5:
        return BlockScaleGemmSpec(
            M=80,
            N=80,
            K=160,
            mantissa_dtype="fp8e4m3",
            group_size_mnk=(1, 1, 160),
            **common,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: block_scale_gemm_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    spec = _spec(idx)
    kernel = build_block_scale_gemm(spec, arch="gfx950")
    text = lower_kernel_to_llvm(kernel, arch="gfx950")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
