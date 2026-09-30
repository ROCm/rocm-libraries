#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/reduce_emit.py -- Python reference emitter for the reduce2d
# instance parity harness. Selects one of the sampled configs by argv[1] (the
# config index), builds the Reduce2DSpec, builds the kernel via build_reduce2d
# and prints lower_kernel_to_llvm(arch='gfx950') to stdout so it can be
# byte-compared with the C emitter reduce_emit.c.
from rocke.instances.common.reduce import (
    Reduce2DSpec,
    build_reduce2d,
)
from _emit_common import run_emit


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
    if idx == 6:
        # gfx1151 (RDNA3.5): wave32 reduce (cross-lane reduction over 32 lanes).
        return (
            Reduce2DSpec(
                n_per_block=4096,
                op="sum",
                block_size=256,
                vec=4,
                dtype="f16",
                wave_size=32,
            ),
            "gfx1151",
        )
    if idx == 7:
        # gfx1201 (RDNA4): wave32 reduce.
        return (
            Reduce2DSpec(
                n_per_block=4096,
                op="sum",
                block_size=256,
                vec=4,
                dtype="f16",
                wave_size=32,
            ),
            "gfx1201",
        )
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch=None):
    return build_reduce2d(spec)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: reduce_emit.py <config_index 0..7> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
