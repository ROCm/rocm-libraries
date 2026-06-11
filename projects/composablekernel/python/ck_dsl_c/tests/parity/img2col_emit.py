#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/img2col_emit.py -- Python reference emitter for the img2col
# parity harness. Selects one of N sampled spec configs by argv[1], builds the
# Img2ColSpec, builds the kernel via build_img2col(arch='gfx950') and prints
# lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared with
# the C emitter img2col_emit.c.
import sys

from ck_dsl.instances.common.img2col import Img2ColSpec, build_img2col
from ck_dsl.instances.common.conv_implicit_gemm import ConvProblem
from ck_dsl import lower_kernel_to_llvm


def _spec(idx: int) -> Img2ColSpec:
    if idx == 0:
        return Img2ColSpec(
            problem=ConvProblem(N=1, Hi=8, Wi=8, C=16, K=16, R=3, S=3),
            block_tile_m=4,
            block_tile_k=64,
            vec_k=1,
        )
    if idx == 1:
        return Img2ColSpec(
            problem=ConvProblem(N=2, Hi=16, Wi=16, C=32, K=32, R=3, S=3, pH=1, pW=1),
            block_tile_m=8,
            block_tile_k=128,
            vec_k=4,
        )
    if idx == 2:
        return Img2ColSpec(
            problem=ConvProblem(N=4, Hi=32, Wi=32, C=64, K=64, R=3, S=3, pH=1, pW=1),
            block_tile_m=16,
            block_tile_k=256,
            vec_k=8,
        )
    if idx == 3:
        return Img2ColSpec(
            problem=ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, R=3, S=3, pH=1, pW=1),
            block_tile_m=8,
            block_tile_k=128,
            vec_k=8,
        )
    if idx == 4:
        return Img2ColSpec(
            problem=ConvProblem(N=2, Hi=16, Wi=16, C=15, K=32, R=3, S=3),
            block_tile_m=8,
            block_tile_k=120,
            vec_k=8,
        )
    if idx == 5:
        return Img2ColSpec(
            problem=ConvProblem(
                N=2, Hi=32, Wi=32, C=32, K=32, R=3, S=3, dH=2, dW=2, pH=2, pW=2
            ),
            block_tile_m=8,
            block_tile_k=128,
            vec_k=4,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: img2col_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    spec = _spec(idx)
    kernel = build_img2col(spec, arch="gfx950")
    text = lower_kernel_to_llvm(kernel, arch="gfx950")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
