#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/moe_smoothquant_emit.py -- Python reference emitter for the
# moe_smoothquant parity harness. Selects one of the sampled
# MoeSmoothQuantSpec configs by argv[1], builds the kernel via
# build_moe_smoothquant and prints lower_kernel_to_llvm(arch='gfx950') to
# stdout so it can be byte-compared with the C emitter moe_smoothquant_emit.c.
from rocke.instances.common.moe_smoothquant import (
    MoeSmoothQuantSpec,
    build_moe_smoothquant,
)
from _emit_common import run_emit


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


def _build(spec, arch=None):
    return build_moe_smoothquant(spec)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: moe_smoothquant_emit.py <config_index 0..3> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
