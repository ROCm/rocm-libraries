#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/smoothquant_emit.py -- Python reference emitter for the
# smoothquant parity harness. Selects one of the sampled SmoothQuantSpec configs
# by argv[1], builds the kernel via build_smoothquant and prints
# lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared with
# the C emitter smoothquant_emit.c.
import sys

from ck_dsl.instances.common.smoothquant import (
    SmoothQuantSpec,
    build_smoothquant,
)
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


def _spec(idx: int) -> SmoothQuantSpec:
    if idx == 0:
        return SmoothQuantSpec(
            n_per_block=1024, dtype="f16", out_dtype="i8", block_size=256, vec=4
        )
    if idx == 1:
        return SmoothQuantSpec(
            n_per_block=2048, dtype="bf16", out_dtype="i8", block_size=256, vec=8
        )
    if idx == 2:
        return SmoothQuantSpec(
            n_per_block=512, dtype="f16", out_dtype="fp8e4m3", block_size=128, vec=4
        )
    if idx == 3:
        return SmoothQuantSpec(
            n_per_block=1024, dtype="bf16", out_dtype="bf8e5m2", block_size=64, vec=2
        )
    if idx == 4:
        return SmoothQuantSpec(
            n_per_block=2048, dtype="f16", out_dtype="i8", block_size=512, vec=4
        )
    if idx == 5:
        return SmoothQuantSpec(
            n_per_block=256, dtype="bf16", out_dtype="fp8e4m3", block_size=256, vec=2
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: smoothquant_emit.py <config_index 0..5> [mode]\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    spec = _spec(idx)
    kernel = build_smoothquant(spec)
    if mode == "ir":
        sys.stdout.write(serialize(kernel))
    elif mode == "verify":
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    else:
        text = lower_kernel_to_llvm(kernel, arch="gfx950")
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
