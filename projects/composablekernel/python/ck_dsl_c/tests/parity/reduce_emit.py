#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/reduce_emit.py -- Python reference emitter for the reduce2d
# instance parity harness. Selects one of the sampled configs by argv[1] (the
# config index), builds the Reduce2DSpec, builds the kernel via build_reduce2d
# and prints lower_kernel_to_llvm(arch='gfx950') to stdout so it can be
# byte-compared with the C emitter reduce_emit.c.
import sys

from ck_dsl.instances.common.reduce import (
    Reduce2DSpec,
    build_reduce2d,
)
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


def _spec(idx: int) -> Reduce2DSpec:
    if idx == 0:
        return Reduce2DSpec(
            n_per_block=4096, op="sum", block_size=256, vec=4, dtype="f16", wave_size=64
        )
    if idx == 1:
        return Reduce2DSpec(
            n_per_block=4096, op="max", block_size=256, vec=4, dtype="f16", wave_size=64
        )
    if idx == 2:
        return Reduce2DSpec(
            n_per_block=4096,
            op="mean",
            block_size=256,
            vec=4,
            dtype="f16",
            wave_size=64,
        )
    if idx == 3:
        return Reduce2DSpec(
            n_per_block=2048,
            op="sum",
            block_size=128,
            vec=4,
            dtype="bf16",
            wave_size=64,
        )
    if idx == 4:
        return Reduce2DSpec(
            n_per_block=4096, op="sum", block_size=512, vec=2, dtype="f16", wave_size=64
        )
    if idx == 5:
        return Reduce2DSpec(
            n_per_block=3072,
            op="max",
            block_size=256,
            vec=8,
            dtype="bf16",
            wave_size=64,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: reduce_emit.py <config_index 0..5> [mode]\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    spec = _spec(idx)
    kernel = build_reduce2d(spec)
    if mode == "ll":
        text = lower_kernel_to_llvm(kernel, arch="gfx950")
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    elif mode == "verify":
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    else:
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
