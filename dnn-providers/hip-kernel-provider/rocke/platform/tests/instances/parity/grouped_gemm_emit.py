#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/grouped_gemm_emit.py -- Python reference emitter for the
# grouped-GEMM parity harness. Selects one of the sampled GroupedGemmSpec
# configs by argv[1], builds the kernel via build_grouped_gemm and prints
# lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared
# with the C emitter grouped_gemm_emit.c.
from rocke.instances.common.grouped_gemm import (
    GroupedGemmSpec,
    build_grouped_gemm,
)
from rocke.instances.common.gemm_universal import TileSpec, TraitSpec
from _emit_common import run_emit


def _spec(idx: int) -> GroupedGemmSpec:
    if idx == 0:
        return GroupedGemmSpec(
            name="ggemm_fp16_m128n128k32",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
            dtype="fp16",
            wave_size=64,
        )
    if idx == 1:
        return GroupedGemmSpec(
            name="ggemm_bf16_m32n32k32",
            tile=TileSpec(
                tile_m=32,
                tile_n=32,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            ),
            trait=TraitSpec(
                pipeline="mem", epilogue="cshuffle", pad_m=True, pad_n=True
            ),
            dtype="bf16",
            wave_size=64,
        )
    if idx == 2:
        return GroupedGemmSpec(
            name="ggemm_fp16_m64n64k64",
            tile=TileSpec(
                tile_m=64,
                tile_n=64,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="compv3", epilogue="default"),
            dtype="fp16",
            wave_size=64,
        )
    if idx == 3:
        return GroupedGemmSpec(
            name="ggemm_fp16_m256n256k128",
            tile=TileSpec(
                tile_m=256,
                tile_n=256,
                tile_k=128,
                warp_m=4,
                warp_n=4,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv4", epilogue="cshuffle", chiplet_swizzle=True
            ),
            dtype="fp16",
            wave_size=64,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_grouped_gemm,
        usage="usage: grouped_gemm_emit.py <config_index 0..3>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
