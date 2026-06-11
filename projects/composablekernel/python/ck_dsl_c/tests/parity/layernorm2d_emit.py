#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/layernorm2d_emit.py -- Python reference emitter for the
# LayerNorm2D parity harness. Selects one of 6 sampled configs by argv[1]
# (the config index 0..5), builds the LayerNorm2DSpec, builds the kernel via
# build_layernorm2d and prints lower_kernel_to_llvm(arch='gfx950') to stdout
# so it can be byte-compared with the C emitter layernorm2d_emit.c.
import sys

from ck_dsl.instances.common.layernorm2d import (
    LayerNorm2DSpec,
    build_layernorm2d,
)
from ck_dsl import lower_kernel_to_llvm


def _spec(idx: int) -> LayerNorm2DSpec:
    if idx == 0:
        return LayerNorm2DSpec(
            n_per_block=4096, block_size=256, vec=4, dtype="f16", save_mean_invstd=False
        )
    if idx == 1:
        return LayerNorm2DSpec(
            n_per_block=4096, block_size=256, vec=8, dtype="f16", save_mean_invstd=False
        )
    if idx == 2:
        return LayerNorm2DSpec(
            n_per_block=4096,
            block_size=256,
            vec=4,
            dtype="bf16",
            save_mean_invstd=False,
        )
    if idx == 3:
        return LayerNorm2DSpec(
            n_per_block=2048, block_size=128, vec=4, dtype="f16", save_mean_invstd=True
        )
    if idx == 4:
        return LayerNorm2DSpec(
            n_per_block=8192, block_size=256, vec=8, dtype="f16", save_mean_invstd=False
        )
    if idx == 5:
        return LayerNorm2DSpec(
            n_per_block=1024, block_size=256, vec=2, dtype="bf16", save_mean_invstd=True
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: layernorm2d_emit.py <config_index 0..5>\n")
        return 2
    idx = int(sys.argv[1])
    spec = _spec(idx)
    kernel = build_layernorm2d(spec)
    text = lower_kernel_to_llvm(kernel, arch="gfx950")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
