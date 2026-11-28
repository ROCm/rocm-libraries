#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/transpose_bc_emit.py -- Python reference emitter for the no-LDS
# in-register transpose instance parity harness. Selects one of the sampled
# configs by argv[1] (the config index), builds the TransposeBcSpec, builds the
# kernel via build_transpose_bc(spec, arch) and prints lower_kernel_to_llvm(arch)
# to stdout so it can be byte-compared with the C emitter transpose_bc_emit.c.
#
# The configs cover every vec width (2, 4, 8), both dtypes, block sizes of one
# and several waves, the flat-global and buffer-resource IO paths (including
# bf16 + use_buffer_io, which names the kernel "_buf" but keeps the global
# path), a non-default kernel name and one wave32 (gfx1151) build.
#
# Configs 10..17 are rejected: a block below one wave, a block that is not a
# whole number of wave32 waves, an unknown arch, a vec outside {2, 4, 8}, a
# non-square per-lane sub-tile, an unsupported dtype, a tile that is not a
# multiple of vec and a block over the hardware thread cap. A rejected config
# prints "invalid spec: <reason>" to stdout and exits 1 on both sides, so the
# gate compares the reject reasons byte for byte.
import sys

from rocke.instances.common.transpose_bc import (
    TransposeBcSpec,
    build_transpose_bc,
    is_valid_spec,
)
from _emit_common import run_emit


def _spec(idx: int):
    if idx == 0:
        return TransposeBcSpec(tile_m=16, tile_n=16, vec=2, dtype="f16")
    if idx == 1:
        return TransposeBcSpec(tile_m=32, tile_n=32, vec=4, dtype="f16")
    if idx == 2:
        return TransposeBcSpec(tile_m=64, tile_n=64, vec=8, dtype="f16")
    if idx == 3:
        return TransposeBcSpec(tile_m=64, tile_n=64, vec=8, dtype="bf16")
    if idx == 4:
        return TransposeBcSpec(
            tile_m=64, tile_n=64, vec=8, dtype="f16", use_buffer_io=True
        )
    if idx == 5:
        return TransposeBcSpec(
            tile_m=64, tile_n=64, vec=8, dtype="bf16", use_buffer_io=True
        )
    if idx == 6:
        return TransposeBcSpec(
            tile_m=32, tile_n=32, vec=2, dtype="f16", use_buffer_io=True
        )
    if idx == 7:
        return TransposeBcSpec(tile_m=128, tile_n=128, vec=8, dtype="bf16")
    if idx == 8:
        return TransposeBcSpec(
            tile_m=96,
            tile_n=96,
            vec=4,
            dtype="f16",
            use_buffer_io=True,
            name="tbc_custom",
        )
    if idx == 9:
        # 64 lanes: two wave32 waves.
        return TransposeBcSpec(), "gfx1151"
    if idx == 10:
        return TransposeBcSpec(tile_m=16, tile_n=16, vec=8)
    if idx == 11:
        return TransposeBcSpec(tile_m=48, tile_n=48, vec=8), "gfx1151"
    if idx == 12:
        return TransposeBcSpec(), "gfx000"
    if idx == 13:
        return TransposeBcSpec(vec=3)
    if idx == 14:
        return TransposeBcSpec(tile_m=64, tile_n=32, vec=8)
    if idx == 15:
        return TransposeBcSpec(dtype="f32")
    if idx == 16:
        return TransposeBcSpec(tile_m=36, tile_n=64, vec=8)
    if idx == 17:
        return TransposeBcSpec(tile_m=512, tile_n=512, vec=8)
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch="gfx950"):
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        sys.stdout.write(f"invalid spec: {why}\n")
        raise SystemExit(1)
    return build_transpose_bc(spec, arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: transpose_bc_emit.py <config_index 0..17> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
