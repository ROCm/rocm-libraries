#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1250_block_scaled_gemm_emit.py -- Python reference emitter for
# the gfx1250 K=64 FP8/BF8 block-scaled GEMM parity harness.
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
)
from _emit_common import run_emit


def _spec(idx: int) -> BlockScaledGemmSpec:
    if idx == 0:
        return BlockScaledGemmSpec(name="g", M=128, N=128, K=128)
    if idx == 1:
        return BlockScaledGemmSpec(
            name="g", M=128, N=128, K=128, dtype_a="bf8", dtype_b="bf8"
        )
    if idx == 2:
        return BlockScaledGemmSpec(
            name="g", M=256, N=128, K=256, dtype_a="fp8", dtype_b="bf8", dtype_c="fp16"
        )
    if idx == 3:
        return BlockScaledGemmSpec(
            name="g",
            M=128,
            N=256,
            K=128,
            dtype_a="bf8",
            dtype_b="fp8",
            scale_dtype="fp16",
        )
    if idx == 4:
        return BlockScaledGemmSpec(
            name="g", M=64, N=64, K=192, matrix_path="wmma", block_k=64, tile_k=64
        )
    if idx == 5:
        return BlockScaledGemmSpec(
            name="g", M=16, N=32, K=256, dtype_a="fp8e4m3", dtype_b="bf8e5m2"
        )
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch="gfx1250"):
    return build_block_scaled_gemm(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: gfx1250_block_scaled_gemm_emit.py <config_index 0..5>\n",
        arch="gfx1250",
    )


if __name__ == "__main__":
    raise SystemExit(main())
