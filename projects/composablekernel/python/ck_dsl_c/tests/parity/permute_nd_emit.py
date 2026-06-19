#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/permute_nd_emit.py -- Python reference emitter for the permute_nd
# instance parity harness. Selects one of the sampled configs by argv[1] (the
# config index), builds the PermuteSpec, builds the kernel via build_permute and
# prints lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared
# with the C emitter permute_nd_emit.c.
import sys

from ck_dsl.instances.common.permute_nd import (
    PermuteSpec,
    build_permute,
)
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


def _spec(idx: int) -> PermuteSpec:
    if idx == 0:
        return PermuteSpec(
            x_shape=(4, 8, 16), perm=(2, 0, 1), dtype="f16", block_size=256
        )
    if idx == 1:
        return PermuteSpec(
            x_shape=(16, 8, 4), perm=(2, 1, 0), dtype="f16", block_size=256
        )
    if idx == 2:
        return PermuteSpec(x_shape=(32, 32), perm=(1, 0), dtype="bf16", block_size=256)
    if idx == 3:
        return PermuteSpec(
            x_shape=(8, 8, 8, 8), perm=(3, 2, 1, 0), dtype="f16", block_size=128
        )
    if idx == 4:
        return PermuteSpec(
            x_shape=(64, 64, 64), perm=(1, 2, 0), dtype="bf16", block_size=512
        )
    if idx == 5:
        return PermuteSpec(x_shape=(256,), perm=(0,), dtype="f16", block_size=256)
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: permute_nd_emit.py <config_index 0..5> [mode]\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    spec = _spec(idx)
    kernel = build_permute(spec, arch="gfx950")
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
