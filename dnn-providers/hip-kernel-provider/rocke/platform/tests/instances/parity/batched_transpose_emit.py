#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/batched_transpose_emit.py -- Python reference emitter for the
# batched 2D transpose instance parity harness. Selects one of the sampled
# configs by argv[1] (the config index), builds the BatchedTranspose2DSpec,
# builds the kernel via build_batched_transpose2d and prints
# lower_kernel_to_llvm(arch) to stdout so it can be byte-compared with the C
# emitter batched_transpose_emit.c.
#
# The configs cover every square tile (16/32/64), every vec width (2/4/8),
# both dtypes plus the "fp16" alias, LDS pads 0/4/8/16, a non-default kernel
# name, the largest block (64x64 v4 = 1024 threads) and one gfx1151 build
# (the kernel has no wave-width dependence; arch only feeds the validator).
#
# Configs 9..13 are rejected: an unknown arch, an unsupported dtype, a vec
# outside {2, 4, 8}, a non-square tile and a negative tile whose block_size
# floors to -2. A rejected config prints "invalid spec: <reason>" to stdout and
# exits 1 on both sides, so the gate compares the reject reasons byte for byte.
import sys

from rocke.instances.common.batched_transpose import (
    BatchedTranspose2DSpec,
    build_batched_transpose2d,
    is_valid_spec,
)
from _emit_common import run_emit


def _spec(idx: int):
    if idx == 0:
        return BatchedTranspose2DSpec(
            tile_m=16, tile_n=16, vec=2, dtype="f16", lds_pad=8
        )
    if idx == 1:
        return BatchedTranspose2DSpec(
            tile_m=32, tile_n=32, vec=4, dtype="f16", lds_pad=8
        )
    if idx == 2:
        # Every field at the dataclass default.
        return BatchedTranspose2DSpec()
    if idx == 3:
        return BatchedTranspose2DSpec(
            tile_m=64, tile_n=64, vec=8, dtype="bf16", lds_pad=8
        )
    if idx == 4:
        return BatchedTranspose2DSpec(
            tile_m=32, tile_n=32, vec=8, dtype="bf16", lds_pad=0
        )
    if idx == 5:
        return BatchedTranspose2DSpec(
            tile_m=16, tile_n=16, vec=4, dtype="f16", lds_pad=4
        )
    if idx == 6:
        return BatchedTranspose2DSpec(
            tile_m=64, tile_n=64, vec=4, dtype="bf16", lds_pad=16
        )
    if idx == 7:
        return BatchedTranspose2DSpec(
            tile_m=32,
            tile_n=32,
            vec=2,
            dtype="fp16",
            lds_pad=8,
            name="rocke_bxpose_custom",
        )
    if idx == 8:
        return (
            BatchedTranspose2DSpec(tile_m=32, tile_n=32, vec=4, dtype="f16", lds_pad=8),
            "gfx1151",
        )
    if idx == 9:
        return BatchedTranspose2DSpec(), "gfx000"
    if idx == 10:
        return BatchedTranspose2DSpec(dtype="f32")
    if idx == 11:
        return BatchedTranspose2DSpec(tile_m=32, tile_n=32, vec=3)
    if idx == 12:
        return BatchedTranspose2DSpec(tile_m=32, tile_n=16, vec=4)
    if idx == 13:
        # (tile_m * tile_n) // vec floors to -2.
        return BatchedTranspose2DSpec(tile_m=-3, tile_n=1, vec=2)
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch="gfx950"):
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        sys.stdout.write(f"invalid spec: {why}\n")
        raise SystemExit(1)
    return build_batched_transpose2d(spec, arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: batched_transpose_emit.py <config_index 0..13> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
