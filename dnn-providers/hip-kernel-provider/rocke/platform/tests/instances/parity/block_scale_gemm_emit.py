#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/block_scale_gemm_emit.py -- Python reference emitter for the
# block_scale_gemm parity harness. Selects one of N sampled spec configs by
# argv[1], builds the BlockScaleGemmSpec, builds the kernel via
# build_block_scale_gemm and prints lower_kernel_to_llvm(arch='gfx950') to stdout
# so it can be byte-compared with the C emitter block_scale_gemm_emit.c.
from rocke.instances.common.block_scale_gemm import (
    BlockScaleGemmSpec,
    build_block_scale_gemm,
)
from _emit_common import run_emit


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
    return run_emit(
        _spec,
        build_block_scale_gemm,
        usage="usage: block_scale_gemm_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
