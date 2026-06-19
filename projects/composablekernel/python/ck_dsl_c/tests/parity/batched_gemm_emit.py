#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/batched_gemm_emit.py -- Python reference emitter for the
# batched-GEMM parity harness. Selects one of 5 sampled BatchedGemmSpec
# configs by argv[1] (0..4), builds it via build_batched_gemm and prints
# lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared
# with the C emitter batched_gemm_emit.c.
import sys

from ck_dsl.instances.common.batched_gemm import (
    BatchedGemmSpec,
    build_batched_gemm,
)
from ck_dsl.instances.common.gemm_universal import TileSpec, TraitSpec
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


def _spec(idx: int) -> BatchedGemmSpec:
    if idx == 0:
        return BatchedGemmSpec(
            name="bgm_64x64x32_2x2",
            tile=TileSpec(
                64,
                64,
                32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="mem", epilogue="default"),
            dtype="fp16",
            wave_size=64,
        )
    if idx == 1:
        return BatchedGemmSpec(
            name="bgm_128x128x32_2x2_cshuffle",
            tile=TileSpec(
                128,
                128,
                32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="compv3", epilogue="cshuffle"),
            dtype="fp16",
            wave_size=64,
        )
    if idx == 2:
        return BatchedGemmSpec(
            name="bgm_256x256x64_4x4",
            tile=TileSpec(
                256,
                256,
                64,
                warp_m=4,
                warp_n=4,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="compv4", epilogue="default"),
            dtype="fp16",
            wave_size=64,
        )
    if idx == 3:
        return BatchedGemmSpec(
            name="bgm_128x128x64_2x2_wide",
            tile=TileSpec(
                128,
                128,
                64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            ),
            trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
            dtype="fp16",
            wave_size=64,
        )
    if idx == 4:
        return BatchedGemmSpec(
            name="bgm_64x128x32_1x2_skip",
            tile=TileSpec(
                64,
                128,
                32,
                warp_m=1,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv3", epilogue="default", active_tile_skip=True
            ),
            dtype="fp16",
            wave_size=64,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: batched_gemm_emit.py <config_index 0..4> [ll|ir|verify]\n"
        )
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    spec = _spec(idx)
    kernel = build_batched_gemm(spec, arch="gfx950")
    if mode == "ll":
        text = lower_kernel_to_llvm(kernel, arch="gfx950")
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    else:  # verify
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
