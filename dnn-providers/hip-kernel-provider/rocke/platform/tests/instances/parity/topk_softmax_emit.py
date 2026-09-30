#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/topk_softmax_emit.py -- Python reference emitter for the
# topk-softmax parity harness. Selects one of the sampled configs by argv[1]
# (the config index), builds the TopkSoftmaxSpec, builds the kernel via
# build_topk_softmax and prints lower_kernel_to_llvm(arch='gfx950') to stdout
# so it can be byte-compared with the C emitter topk_softmax_emit.c.
from rocke.instances.common.topk_softmax import (
    TopkSoftmaxSpec,
    build_topk_softmax,
)
from _emit_common import run_emit


def _spec(idx: int) -> TopkSoftmaxSpec:
    if idx == 0:
        return TopkSoftmaxSpec(
            n_per_row=32, k=1, dtype="f32", out_dtype="f32", block_size=32
        )
    if idx == 1:
        return TopkSoftmaxSpec(
            n_per_row=64, k=4, dtype="f16", out_dtype="f32", block_size=64
        )
    if idx == 2:
        return TopkSoftmaxSpec(
            n_per_row=128, k=8, dtype="bf16", out_dtype="bf16", block_size=64
        )
    if idx == 3:
        return TopkSoftmaxSpec(
            n_per_row=4096, k=16, dtype="f32", out_dtype="f32", block_size=128
        )
    if idx == 4:
        return TopkSoftmaxSpec(
            n_per_row=16384, k=32, dtype="f32", out_dtype="f32", block_size=256
        )
    if idx == 5:
        return TopkSoftmaxSpec(
            n_per_row=768, k=2, dtype="f32", out_dtype="f32", block_size=64
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_topk_softmax,
        usage="usage: topk_softmax_emit.py <config_index 0..5> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
