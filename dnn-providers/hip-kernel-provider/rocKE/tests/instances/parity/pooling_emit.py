#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/pooling_emit.py -- Python reference emitter for the pooling2d
# instance parity harness. Selects one of the sampled configs by argv[1] (the
# config index), builds the Pooling2DSpec, builds the kernel via
# build_pooling2d and prints lower_kernel_to_llvm(arch='gfx950') to stdout so
# it can be byte-compared with the C emitter pooling_emit.c.
from rocke.instances.common.pooling import (
    Pooling2DSpec,
    PoolingProblem,
    build_pooling2d,
)
from _emit_common import run_emit


def _spec(idx: int) -> Pooling2DSpec:
    if idx == 0:
        return Pooling2DSpec(
            problem=PoolingProblem(
                N=1, H=28, W=28, C=64, Y=2, X=2, sH=2, sW=2, pH=0, pW=0, dH=1, dW=1
            ),
            dtype="f16",
            op="max",
            block_size=256,
            vec=1,
        )
    if idx == 1:
        return Pooling2DSpec(
            problem=PoolingProblem(
                N=2, H=56, W=56, C=128, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
            ),
            dtype="f16",
            op="avg",
            block_size=256,
            vec=2,
        )
    if idx == 2:
        return Pooling2DSpec(
            problem=PoolingProblem(
                N=4, H=112, W=112, C=256, Y=7, X=7, sH=7, sW=7, pH=0, pW=0, dH=1, dW=1
            ),
            dtype="f16",
            op="sum",
            block_size=256,
            vec=4,
        )
    if idx == 3:
        return Pooling2DSpec(
            problem=PoolingProblem(
                N=1, H=224, W=224, C=64, Y=2, X=2, sH=2, sW=2, pH=0, pW=0, dH=1, dW=1
            ),
            dtype="bf16",
            op="max",
            block_size=256,
            vec=1,
        )
    if idx == 4:
        return Pooling2DSpec(
            problem=PoolingProblem(
                N=2, H=32, W=32, C=256, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
            ),
            dtype="f16",
            op="avg",
            block_size=128,
            vec=8,
        )
    if idx == 5:
        return Pooling2DSpec(
            problem=PoolingProblem(
                N=1, H=64, W=64, C=128, Y=2, X=2, sH=2, sW=2, pH=0, pW=0, dH=1, dW=1
            ),
            dtype="bf16",
            op="sum",
            block_size=512,
            vec=4,
        )
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch=None):
    return build_pooling2d(spec)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: pooling_emit.py <config_index 0..5> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
