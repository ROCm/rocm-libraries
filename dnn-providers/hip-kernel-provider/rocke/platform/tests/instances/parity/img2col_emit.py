#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/img2col_emit.py -- Python reference emitter for the img2col
# parity harness. Selects one of N sampled spec configs by argv[1], builds the
# Img2ColSpec, builds the kernel via build_img2col(arch='gfx950') and prints
# lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared with
# the C emitter img2col_emit.c.
from rocke.instances.common.img2col import Img2ColSpec, build_img2col
from rocke.instances.common.conv_implicit_gemm import ConvProblem
from _emit_common import run_emit


def _spec(idx: int) -> Img2ColSpec:
    if idx == 0:
        return Img2ColSpec(
            problem=ConvProblem(N=1, Hi=8, Wi=8, C=16, K=16, Y=3, X=3),
            block_tile_m=4,
            block_tile_k=64,
            vec_k=1,
        )
    if idx == 1:
        return Img2ColSpec(
            problem=ConvProblem(N=2, Hi=16, Wi=16, C=32, K=32, Y=3, X=3, pH=1, pW=1),
            block_tile_m=8,
            block_tile_k=128,
            vec_k=4,
        )
    if idx == 2:
        return Img2ColSpec(
            problem=ConvProblem(N=4, Hi=32, Wi=32, C=64, K=64, Y=3, X=3, pH=1, pW=1),
            block_tile_m=16,
            block_tile_k=256,
            vec_k=8,
        )
    if idx == 3:
        return Img2ColSpec(
            problem=ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1),
            block_tile_m=8,
            block_tile_k=128,
            vec_k=8,
        )
    if idx == 4:
        return Img2ColSpec(
            problem=ConvProblem(N=2, Hi=16, Wi=16, C=15, K=32, Y=3, X=3),
            block_tile_m=8,
            block_tile_k=120,
            vec_k=8,
        )
    if idx == 5:
        return Img2ColSpec(
            problem=ConvProblem(
                N=2, Hi=32, Wi=32, C=32, K=32, Y=3, X=3, dH=2, dW=2, pH=2, pW=2
            ),
            block_tile_m=8,
            block_tile_k=128,
            vec_k=4,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_img2col,
        usage="usage: img2col_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
