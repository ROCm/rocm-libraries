#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gemm_emit.py -- Python reference emitter for the universal-GEMM
# parity harness. Selects one of 7 sampled (spec,knobs) configs by argv[1] (the
# config index 0..6), builds the UniversalGemmSpec, builds the kernel via
# build_universal_gemm and prints lower_kernel_to_llvm(arch='gfx950') to stdout
# so it can be byte-compared with the C emitter gemm_emit.c.
from rocke.instances.common.gemm_universal import (
    UniversalGemmSpec,
    TileSpec,
    TraitSpec,
    DataSpec,
    build_universal_gemm,
)
from _emit_common import run_emit


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
    if idx == 7:
        # Split-K regression: split_k=1 must stay byte-identical to the
        # single-K-pass body (same shape as idx 0).
        return UniversalGemmSpec(
            name="test8",
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
            trait=TraitSpec(pipeline="compv3", epilogue="default", split_k=1),
            data=DataSpec(
                dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32"
            ),
            wave_size=64,
            block_size=256,
            batched=False,
        )
    if idx == 8:
        # Split-K active: split_k=8 on the M=2 N=4096 K=4096 bf16 decode
        # shape, tile 16x64x64. Exercises the f32 Cf32 workspace + atomic-add
        # epilogue (atomicrmw fadd).
        return UniversalGemmSpec(
            name="test9",
            tile=TileSpec(
                tile_m=16,
                tile_n=64,
                tile_k=64,
                warp_m=1,
                warp_n=4,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="compv4", epilogue="default", split_k=8),
            data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
            wave_size=64,
            block_size=256,
            batched=False,
        )
    if idx == 9:
        # gfx942 (CDNA) coverage of the universal-GEMM common family: the test1
        # shape, but built+lowered for gfx942 so the gate exercises gfx942's
        # datalayout / waitcnt / MFMA-catalog lowering, not only gfx950. Same
        # wave64 MFMA spec is valid on both CDNA archs; both engines must agree.
        return (
            UniversalGemmSpec(
                name="test1_gfx942",
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
            ),
            "gfx942",
        )
    if idx == 10:
        # cshuffle epilogue with cshuffle_no_alias=True: the C staging tile gets
        # its own exclusive LDS bytes (not aliased onto A/B) and the step-0 reuse
        # barrier is elided. Same shape as idx 1 (the aliased cshuffle config) so
        # the two configs isolate the no-alias knob. Both engines must agree.
        return UniversalGemmSpec(
            name="test_noalc",
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
            trait=TraitSpec(
                pipeline="compv4", epilogue="cshuffle", cshuffle_no_alias=True
            ),
            data=DataSpec(dtype_a="fp16"),
            wave_size=64,
            block_size=1024,
            batched=False,
        )
    if idx == 11:
        # WMMA (RDNA3.5 / gfx1151, wave32) cshuffle epilogue. The wave32
        # accumulator scatter is driven by the op's c_layout() map rather than
        # the MFMA lane math, so this is the config that byte-validates the
        # WMMA branch of _emit_epilogue_cshuffle. pad_m/pad_n stay off so the
        # step-4 wide vector store (store_vec=8) is the path exercised.
        return (
            UniversalGemmSpec(
                name="test_wmma_cshuffle_1151",
                tile=TileSpec(
                    tile_m=32,
                    tile_n=32,
                    tile_k=16,
                    warp_m=2,
                    warp_n=2,
                    warp_k=1,
                    warp_tile_m=16,
                    warp_tile_n=16,
                    warp_tile_k=16,
                ),
                trait=TraitSpec(pipeline="mem", epilogue="cshuffle"),
                data=DataSpec(dtype_a="fp16"),
                wave_size=32,
                block_size=128,
                batched=False,
            ),
            "gfx1151",
        )
    if idx == 12:
        # Same WMMA cshuffle config on gfx1201 (RDNA4). gfx1201 uses the
        # column-distributed gfx12 accumulator map -- the same map gfx1250
        # bf16 uses -- so this covers the layout the gfx1250 GEMM depends on
        # with an arch the C++ engine can actually lower.
        return (
            UniversalGemmSpec(
                name="test_wmma_cshuffle_1201",
                tile=TileSpec(
                    tile_m=32,
                    tile_n=32,
                    tile_k=16,
                    warp_m=2,
                    warp_n=2,
                    warp_k=1,
                    warp_tile_m=16,
                    warp_tile_n=16,
                    warp_tile_k=16,
                ),
                trait=TraitSpec(pipeline="mem", epilogue="cshuffle"),
                data=DataSpec(dtype_a="fp16"),
                wave_size=32,
                block_size=128,
                batched=False,
            ),
            "gfx1201",
        )
    if idx == 13:
        # Persistent (grid-stride) tile loop, plain decode: the whole
        # K-loop + cshuffle epilogue lives inside a scf.for strided by
        # persistent_ctas, and the tile origin comes from the induction
        # variable instead of blockIdx. Same shape as idx 1 so the two
        # configs isolate the persistent knob.
        return UniversalGemmSpec(
            name="test_persistent",
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
            trait=TraitSpec(
                pipeline="compv4",
                epilogue="cshuffle",
                persistent=True,
                persistent_ctas=304,
            ),
            data=DataSpec(dtype_a="fp16"),
            wave_size=64,
            block_size=1024,
            batched=False,
        )
    if idx == 14:
        # Persistent composed with the chiplet super-tile swizzle: the tile
        # index feeding the XCD remap is the loop's induction variable rather
        # than the flattened blockIdx, which is the other half of the
        # persistent tile-decode branch.
        return UniversalGemmSpec(
            name="test_persistent_chiplet",
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
            trait=TraitSpec(
                pipeline="compv3",
                epilogue="default",
                persistent=True,
                persistent_ctas=256,
                chiplet_swizzle=True,
            ),
            data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
            wave_size=64,
            block_size=256,
            batched=False,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_universal_gemm,
        usage="usage: gemm_emit.py <config_index 0..14>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
