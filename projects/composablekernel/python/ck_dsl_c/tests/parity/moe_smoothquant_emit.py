#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/moe_smoothquant_emit.py -- Python reference emitter for the
# moe_smoothquant parity harness. Selects one of the sampled
# MoeSmoothQuantSpec configs by argv[1], builds the kernel via
# build_moe_smoothquant and prints lower_kernel_to_llvm(arch='gfx950') to
# stdout so it can be byte-compared with the C emitter moe_smoothquant_emit.c.
import sys

from ck_dsl.instances.common.moe_smoothquant import (
    MoeSmoothQuantSpec,
    build_moe_smoothquant,
)
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


def _spec(idx: int) -> MoeSmoothQuantSpec:
    if idx == 0:
        return MoeSmoothQuantSpec(
            n_per_block=512,
            topk=2,
            experts=64,
            dtype="f16",
            out_dtype="i8",
            block_size=256,
            vec=4,
        )
    if idx == 1:
        return MoeSmoothQuantSpec(
            n_per_block=1024,
            topk=4,
            experts=128,
            dtype="bf16",
            out_dtype="fp8e4m3",
            block_size=256,
            vec=4,
        )
    if idx == 2:
        return MoeSmoothQuantSpec(
            n_per_block=2048,
            topk=8,
            experts=256,
            dtype="f16",
            out_dtype="i8",
            block_size=256,
            vec=4,
            tokens=256,
        )
    if idx == 3:
        return MoeSmoothQuantSpec(
            n_per_block=4096,
            topk=1,
            experts=8,
            dtype="f16",
            out_dtype="i8",
            block_size=512,
            vec=8,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: moe_smoothquant_emit.py <config_index 0..3> [mode]\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    spec = _spec(idx)
    kernel = build_moe_smoothquant(spec)
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
