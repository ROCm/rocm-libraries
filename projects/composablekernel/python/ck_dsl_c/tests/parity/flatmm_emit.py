#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/flatmm_emit.py -- Python reference emitter for the FlatMM
# parity harness. Selects one of 6 sampled FlatMMSpec configs by argv[1]
# (0..5), builds it via build_flatmm and prints lower_kernel_to_llvm(
# arch='gfx950') to stdout so it can be byte-compared with the C emitter
# flatmm_emit.c.
import sys

from ck_dsl.instances.common.flatmm import FlatMMSpec, build_flatmm
from ck_dsl.instances.common.gemm_universal import TileSpec, TraitSpec
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


def _spec(idx: int) -> FlatMMSpec:
    if idx == 0:
        return FlatMMSpec(
            tile=TileSpec(
                128,
                128,
                64,
                warp_m=1,
                warp_n=4,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"
            ),
            wave_size=64,
            block_size=256,
            batch_size=0,
            preshuffle_b=False,
            name="ck_dsl_flatmm",
        )
    if idx == 1:
        return FlatMMSpec(
            tile=TileSpec(
                128,
                128,
                64,
                warp_m=1,
                warp_n=4,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="mem", scheduler="intrawave", epilogue="default"),
            wave_size=64,
            block_size=256,
            batch_size=0,
            preshuffle_b=False,
            name="ck_dsl_flatmm",
        )
    if idx == 2:
        return FlatMMSpec(
            tile=TileSpec(
                128,
                128,
                64,
                warp_m=1,
                warp_n=4,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            ),
            trait=TraitSpec(
                pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"
            ),
            wave_size=64,
            block_size=256,
            batch_size=0,
            preshuffle_b=False,
            name="ck_dsl_flatmm",
        )
    if idx == 3:
        return FlatMMSpec(
            tile=TileSpec(
                64,
                64,
                32,
                warp_m=1,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"
            ),
            wave_size=64,
            block_size=128,
            batch_size=0,
            preshuffle_b=False,
            name="ck_dsl_flatmm",
        )
    if idx == 4:
        return FlatMMSpec(
            tile=TileSpec(
                128,
                128,
                64,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"
            ),
            wave_size=64,
            block_size=512,
            batch_size=0,
            preshuffle_b=False,
            name="ck_dsl_flatmm",
        )
    if idx == 5:
        return FlatMMSpec(
            tile=TileSpec(
                256,
                256,
                64,
                warp_m=2,
                warp_n=4,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"
            ),
            wave_size=64,
            block_size=512,
            batch_size=0,
            preshuffle_b=False,
            name="ck_dsl_flatmm",
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: flatmm_emit.py <config_index 0..5>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    spec = _spec(idx)
    kernel = build_flatmm(spec, arch="gfx950")
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
