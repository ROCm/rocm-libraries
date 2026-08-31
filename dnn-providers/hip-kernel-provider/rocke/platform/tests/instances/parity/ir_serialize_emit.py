#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/ir_serialize_emit.py -- Python reference emitter for the
# ck.dsl.ir/v1 serialization parity harness. Selects one of the 7 sampled
# universal-GEMM configs by argv[1] (config index 0..6), builds the
# UniversalGemmSpec IDENTICALLY to gemm_emit.py, builds the kernel via
# build_universal_gemm and prints ir_serialize.serialize(kernel) to stdout so it
# can be byte-compared with the C emitter ir_serialize_emit.c.
import sys

from rocke.instances.common.gemm_universal import (
    UniversalGemmSpec,
    TileSpec,
    TraitSpec,
    DataSpec,
    build_universal_gemm,
)
from rocke.core.ir_serialize import serialize


def _spec(idx: int) -> UniversalGemmSpec:
    if idx == 0:
        return UniversalGemmSpec(
            name="test1",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="compv3", epilogue="default"),
            data=DataSpec(
                dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32"
            ),
            wave_size=64,
            block_size=256,
            batched=False,
        )
    if idx == 1:
        return UniversalGemmSpec(
            name="test2",
            tile=TileSpec(
                tile_m=256,
                tile_n=256,
                tile_k=64,
                warp_m=4,
                warp_n=4,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
            data=DataSpec(dtype_a="fp16"),
            wave_size=64,
            block_size=1024,
            batched=False,
        )
    if idx == 2:
        return UniversalGemmSpec(
            name="test3",
            tile=TileSpec(
                tile_m=256,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=4,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=8,
            ),
            trait=TraitSpec(pipeline="compv4", epilogue="default"),
            data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
            wave_size=64,
            block_size=512,
            batched=False,
        )
    if idx == 3:
        return UniversalGemmSpec(
            name="test4",
            tile=TileSpec(
                tile_m=128,
                tile_n=256,
                tile_k=64,
                warp_m=4,
                warp_n=1,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            ),
            trait=TraitSpec(pipeline="mem", epilogue="default"),
            data=DataSpec(dtype_a="fp16"),
            wave_size=64,
            block_size=256,
            batched=False,
        )
    if idx == 4:
        return UniversalGemmSpec(
            name="test5",
            tile=TileSpec(
                tile_m=64,
                tile_n=64,
                tile_k=64,
                warp_m=1,
                warp_n=1,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv3", epilogue="cshuffle", chiplet_swizzle=True
            ),
            data=DataSpec(dtype_a="fp16"),
            wave_size=64,
            block_size=64,
            batched=False,
        )
    if idx == 5:
        return UniversalGemmSpec(
            name="test6",
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
            trait=TraitSpec(pipeline="compv4", epilogue="default", direct_to_lds=True),
            data=DataSpec(dtype_a="fp16"),
            wave_size=64,
            block_size=1024,
            batched=True,
        )
    if idx == 6:
        return UniversalGemmSpec(
            name="test7",
            tile=TileSpec(
                tile_m=192,
                tile_n=192,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=32,
            ),
            trait=TraitSpec(pipeline="compv4", epilogue="cshuffle", lds_swizzle=True),
            data=DataSpec(dtype_a="fp16"),
            wave_size=64,
            block_size=256,
            batched=False,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: ir_serialize_emit.py <config_index 0..6>\n")
        return 2
    idx = int(sys.argv[1])
    spec = _spec(idx)
    kernel = build_universal_gemm(spec, arch="gfx950")
    sys.stdout.write(serialize(kernel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
