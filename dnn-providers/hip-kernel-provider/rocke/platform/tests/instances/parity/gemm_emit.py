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
        # gfx1250 (RDNA) fp8 coverage: fp8e4m3 A/B into a bf16 C on the
        # 16x16x64 WMMA atom. The only config in this family whose A/B and C
        # dtypes differ, and the only one on a wave32 arch -- it exercises the
        # K=64 low-bit intrinsic ABI (no leading sign-extend immargs), the
        # dword-carried LDS fragment load, and the truncated i8 padding zero,
        # none of which the fp16/bf16 configs above reach. Padding is on so the
        # masked-load path is covered too.
        return (
            UniversalGemmSpec(
                name="test_fp8_gfx1250",
                tile=TileSpec(
                    tile_m=64, tile_n=64, tile_k=64,
                    warp_m=2, warp_n=2, warp_k=1,
                    warp_tile_m=16, warp_tile_n=16, warp_tile_k=64,
                ),
                trait=TraitSpec(
                    pipeline="mem",
                    scheduler="intrawave",
                    epilogue="default",
                    pad_m=True,
                    pad_n=True,
                    pad_k=True,
                ),
                data=DataSpec(
                    dtype_a="fp8e4m3", dtype_b="fp8e4m3",
                    dtype_c="bf16", dtype_acc="fp32", layout="RCR",
                ),
                wave_size=32,
                block_size=128,
                batched=False,
            ),
            "gfx1250",
        )
    if idx == 12:
        # Config 11 with the global A/B load width pinned to the old 2-byte
        # ladder. The resolved default gives a 1-byte operand all 16 bytes;
        # this is the only config that exercises the override, and it is the
        # arm the lever harness measures the default against. Unpadded so the
        # whole-vector load path is the one compared -- config 11 already
        # covers the padded per-element gather at the wider width.
        return (
            UniversalGemmSpec(
                name="test_fp8_gfx1250_abeb2",
                tile=TileSpec(
                    tile_m=64, tile_n=64, tile_k=64,
                    warp_m=2, warp_n=2, warp_k=1,
                    warp_tile_m=16, warp_tile_n=16, warp_tile_k=64,
                ),
                trait=TraitSpec(
                    pipeline="mem",
                    scheduler="intrawave",
                    epilogue="default",
                    pad_m=False,
                    pad_n=False,
                    pad_k=False,
                    ab_load_elem_bytes=2,
                ),
                data=DataSpec(
                    dtype_a="fp8e4m3", dtype_b="fp8e4m3",
                    dtype_c="bf16", dtype_acc="fp32", layout="RCR",
                ),
                wave_size=32,
                block_size=128,
                batched=False,
            ),
            "gfx1250",
        )
    if idx == 13:
        # Config 11 with an lds_k_pad that is NOT a multiple of the 16-element
        # LDS vector width. The staged tile's row stride becomes 64+56=120
        # elements, so alternate rows start off a 16-byte boundary and the
        # access can only claim align 8. Every other config in this family has a
        # stride the vector width divides, which makes this the only one where
        # the alignment is bounded by the stride rather than by the vector --
        # i.e. the only config that would catch the alignment attribute being
        # derived from the vector shape alone.
        return (
            UniversalGemmSpec(
                name="test_fp8_gfx1250_ldspad",
                tile=TileSpec(
                    tile_m=64, tile_n=64, tile_k=64,
                    warp_m=2, warp_n=2, warp_k=1,
                    warp_tile_m=16, warp_tile_n=16, warp_tile_k=64,
                ),
                trait=TraitSpec(
                    pipeline="mem",
                    scheduler="intrawave",
                    epilogue="default",
                    pad_m=True,
                    pad_n=True,
                    pad_k=True,
                    lds_k_pad=56,
                ),
                data=DataSpec(
                    dtype_a="fp8e4m3", dtype_b="fp8e4m3",
                    dtype_c="bf16", dtype_acc="fp32", layout="RCR",
                ),
                wave_size=32,
                block_size=128,
                batched=False,
            ),
            "gfx1250",
        )
    # gfx1250 TDM load path, one config per K-loop shape. Appended last (configs
    # 14–17) so that inserting earlier does not renumber existing configs.
    # Ported from the upstream parity suite; deliberately bf16 so the path is
    # not gated by the fp8/bf8 atom and the spec builder stays simple.
    if idx in _TDM_CONFIGS:
        # depth 1 = single-buffer loop, depth 2 = ping-pong, depth 3/4 = ring.
        # Multi-wave configs gate each operand's descriptor to one wave; the
        # single-wave config (depth 4) issues both unconditionally.
        name, depth, epilogue, pad, single_wave = _TDM_CONFIGS[idx]
        tile = (
            TileSpec(
                tile_m=64,
                tile_n=64,
                tile_k=64,
                warp_m=1,
                warp_n=1,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            )
            if single_wave
            else TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            )
        )
        return (
            UniversalGemmSpec(
                name=name,
                tile=tile,
                trait=TraitSpec(
                    pipeline="mem",
                    epilogue=epilogue,
                    tdm=True,
                    tdm_depth=depth,
                    lds_k_pad=pad,
                ),
                data=DataSpec(
                    dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32"
                ),
                wave_size=32,
                block_size=32 if single_wave else 128,
                batched=False,
            ),
            "gfx1250",
        )
    if idx in _WMMA_CSHUFFLE_CONFIGS:
        # gfx1250 WMMA cshuffle coverage. Every other cshuffle config in this
        # family is MFMA, and the WMMA path rejected the epilogue outright
        # until it was ported, so without these two the gate says nothing
        # about the staging scatter or the C staging tile's dtype.
        #
        # The pair is chosen to split the two halves that can regress
        # independently:
        #   * fp8, unpadded -- A/B (8-bit) and C (16-bit) disagree, so it pins
        #     the staging tile to the *C* dtype; unpadded keeps the wide
        #     global store the epilogue exists to produce.
        #   * bf16, padded -- A/B and C agree, and pad_n decomposes that wide
        #     store back into guarded per-element stores, which is the other
        #     branch of step 4.
        name, dtype, pad = _WMMA_CSHUFFLE_CONFIGS[idx]
        warp_tile_k = 64 if dtype.startswith(("fp8", "bf8")) else 32
        return (
            UniversalGemmSpec(
                name=name,
                tile=TileSpec(
                    tile_m=64,
                    tile_n=64,
                    tile_k=warp_tile_k,
                    warp_m=2,
                    warp_n=2,
                    warp_k=1,
                    warp_tile_m=16,
                    warp_tile_n=16,
                    warp_tile_k=warp_tile_k,
                ),
                trait=TraitSpec(
                    pipeline="mem",
                    scheduler="intrawave",
                    epilogue="cshuffle",
                    pad_m=pad,
                    pad_n=pad,
                    pad_k=True,
                ),
                data=DataSpec(
                    dtype_a=dtype,
                    dtype_b=dtype,
                    dtype_c="bf16" if dtype.startswith(("fp8", "bf8")) else dtype,
                    dtype_acc="fp32",
                    layout="RCR",
                ),
                wave_size=32,
                block_size=128,
                batched=False,
            ),
            "gfx1250",
        )
    if idx == 20:
        # The config the LDS budget used to reject. A 256x256 C staging tile is
        # 128 KiB; charged on top of a depth-2 TDM ring's A/B it overflowed the
        # gfx1250 cap, so this geometry -- the champion of the fp8 square sweep
        # -- was never admitted with cshuffle. The packer aliases C onto the A/B
        # pool, so real usage is max(AB, C) and it fits. Here to keep the gate
        # honest about that: the previous budget rejects this spec outright, and
        # a rejection on one engine and not the other is an ASYMMETRIC drift.
        return (
            UniversalGemmSpec(
                name="test_cshuffle_wmma_aliased_lds",
                tile=TileSpec(
                    tile_m=256,
                    tile_n=256,
                    tile_k=256,
                    warp_m=4,
                    warp_n=8,
                    warp_k=1,
                    warp_tile_m=16,
                    warp_tile_n=16,
                    warp_tile_k=64,
                ),
                trait=TraitSpec(
                    pipeline="mem",
                    scheduler="intrawave",
                    epilogue="cshuffle",
                    tdm=True,
                    tdm_depth=2,
                    lds_k_pad=48,
                ),
                data=DataSpec(
                    dtype_a="fp8e4m3",
                    dtype_b="fp8e4m3",
                    dtype_c="bf16",
                    dtype_acc="fp32",
                    layout="RCR",
                ),
                wave_size=32,
                block_size=1024,
                batched=False,
            ),
            "gfx1250",
        )
    if idx == 21:
        # The only config on the dense 16x16x128 f8f6f4 atom. Its operand
        # carrier is <16 x i32> rather than <8 x i32> and its intrinsic takes a
        # format immediate per matrix with NO trailing reuse flags, so neither
        # the K=64 configs above nor the scaled 16x16x128 atoms (a different
        # opcode and a different arity) would catch a drift in this call shape.
        # tile_k is two atom steps so the K loop is emitted, not peeled to one.
        return (
            UniversalGemmSpec(
                name="test_wmma_f8f6f4_k128",
                tile=TileSpec(
                    tile_m=128,
                    tile_n=128,
                    tile_k=256,
                    warp_m=2,
                    warp_n=2,
                    warp_k=1,
                    warp_tile_m=16,
                    warp_tile_n=16,
                    warp_tile_k=128,
                ),
                trait=TraitSpec(
                    pipeline="mem",
                    scheduler="intrawave",
                    epilogue="cshuffle",
                    tdm=True,
                    tdm_depth=2,
                    lds_k_pad=48,
                ),
                data=DataSpec(
                    dtype_a="fp8e4m3",
                    dtype_b="fp8e4m3",
                    dtype_c="bf16",
                    dtype_acc="fp32",
                    layout="RCR",
                ),
                wave_size=32,
                block_size=128,
                batched=False,
            ),
            "gfx1250",
        )
    raise SystemExit(f"unknown config index {idx}")


# idx -> (name, dtype_ab, pad_m/pad_n). Appended after the TDM block for the
# same reason it was: a new index must not renumber the existing ones.
_WMMA_CSHUFFLE_CONFIGS = {
    18: ("test_cshuffle_wmma_fp8", "fp8e4m3", False),
    19: ("test_cshuffle_wmma_bf16_pad", "bf16", True),
}


# idx -> (name, tdm_depth, epilogue, lds_k_pad, single_wave)
_TDM_CONFIGS = {
    14: ("test_tdm_d1", 1, "default", 8, False),
    15: ("test_tdm_d2", 2, "default", 8, False),
    16: ("test_tdm_d3_ring", 3, "default", 8, False),
    17: ("test_tdm_d4_ring_1wave", 4, "default", 0, True),
}


def main() -> int:
    return run_emit(
        _spec,
        build_universal_gemm,
        usage="usage: gemm_emit.py <config_index 0..21>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
