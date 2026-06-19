#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/matmul_nbits_emit.py -- Python reference emitter for the
# MatMulNBits parity harness. Selects one of 6 configs by argv[1] (index 0..5),
# builds the MatMulNBitsSpec, builds the kernel via build_matmul_nbits(arch=
# 'gfx1201') and prints lower_kernel_to_llvm(kernel, arch='gfx1201') to stdout so
# it can be byte-compared with the C emitter matmul_nbits_emit.c. gfx1201 is one
# of the matmul_nbits SUPPORTED_ARCHES (gfx1151/gfx1201); gfx950 is rejected by
# the validator on both sides, so it must NOT be used here.
import sys

from ck_dsl.instances.common.matmul_nbits import build_matmul_nbits
from ck_dsl.instances.common._matmul_nbits_common import MatMulNBitsSpec
from ck_dsl.instances.common.gemm_universal import TileSpec
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


def _spec(idx: int) -> MatMulNBitsSpec:
    if idx == 0:
        return MatMulNBitsSpec(
            name="matmul_nbits_gfx950",
            N=4096,
            K=4096,
            tile=TileSpec(
                tile_m=64,
                tile_n=128,
                tile_k=16,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            group_size=32,
            scale_dtype="fp16",
            family="large_n",
            optimized=False,
        )
    if idx == 1:
        return MatMulNBitsSpec(
            name="matmul_nbits_gfx950_skinny",
            N=32,
            K=4096,
            tile=TileSpec(
                tile_m=64,
                tile_n=32,
                tile_k=16,
                warp_m=2,
                warp_n=1,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            group_size=32,
            scale_dtype="fp16",
            family="skinny_n",
            optimized=False,
        )
    if idx == 2:
        return MatMulNBitsSpec(
            name="matmul_nbits_gfx950_gemv",
            N=248320,
            K=4096,
            tile=TileSpec(
                tile_m=1,
                tile_n=256,
                tile_k=16,
                warp_m=1,
                warp_n=8,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            group_size=32,
            scale_dtype="fp16",
            family="decode_gemv",
            optimized=False,
        )
    if idx == 3:
        return MatMulNBitsSpec(
            name="matmul_nbits_gfx950_large_8k",
            N=8192,
            K=4096,
            tile=TileSpec(
                tile_m=64,
                tile_n=128,
                tile_k=16,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            group_size=32,
            scale_dtype="fp32",
            family="large_n",
            optimized=False,
        )
    if idx == 4:
        return MatMulNBitsSpec(
            name="matmul_nbits_gfx950_opt",
            N=4096,
            K=4096,
            tile=TileSpec(
                tile_m=64,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            group_size=32,
            scale_dtype="fp16",
            family="large_n",
            optimized=True,
        )
    if idx == 5:
        return MatMulNBitsSpec(
            name="matmul_nbits_gfx950_12k",
            N=12288,
            K=4096,
            tile=TileSpec(
                tile_m=64,
                tile_n=128,
                tile_k=16,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            group_size=32,
            scale_dtype="fp16",
            family="large_n",
            optimized=False,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: matmul_nbits_emit.py <config_index 0..5> [ll|ir|verify]\n"
        )
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    spec = _spec(idx)
    kernel = build_matmul_nbits(spec, arch="gfx1201")
    if mode == "ll":
        text = lower_kernel_to_llvm(kernel, arch="gfx1201")
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    else:  # verify
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
