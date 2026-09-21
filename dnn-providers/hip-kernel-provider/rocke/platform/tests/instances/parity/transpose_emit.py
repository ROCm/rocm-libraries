#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/transpose_emit.py -- Python reference emitter for the 2D
# transpose instance parity harness. Selects one of the sampled configs by
# argv[1] (the config index), builds the Transpose2DSpec, builds the kernel via
# build_transpose2d and prints lower_kernel_to_llvm(arch='gfx950') to stdout so
# it can be byte-compared with the C emitter transpose_emit.c.
from rocke.instances.common.transpose import Transpose2DSpec, build_transpose2d
from _emit_common import run_emit


def _spec(idx: int) -> Transpose2DSpec:
    if idx == 0:
        return Transpose2DSpec(
            tile_m=16, tile_n=16, vec=2, dtype="f16", lds_pad=8, grid_order="row"
        )
    if idx == 1:
        return Transpose2DSpec(
            tile_m=32, tile_n=32, vec=4, dtype="f16", lds_pad=8, grid_order="row"
        )
    if idx == 2:
        return Transpose2DSpec(
            tile_m=64, tile_n=64, vec=8, dtype="f16", lds_pad=8, grid_order="row"
        )
    if idx == 3:
        return Transpose2DSpec(
            tile_m=64, tile_n=64, vec=8, dtype="bf16", lds_pad=8, grid_order="row"
        )
    if idx == 4:
        return Transpose2DSpec(
            tile_m=32, tile_n=32, vec=4, dtype="bf16", lds_pad=8, grid_order="morton"
        )
    if idx == 5:
        return Transpose2DSpec(
            tile_m=16, tile_n=16, vec=4, dtype="f16", lds_pad=8, grid_order="row"
        )
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch=None):
    return build_transpose2d(spec)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: transpose_emit.py <config_index 0..5> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
