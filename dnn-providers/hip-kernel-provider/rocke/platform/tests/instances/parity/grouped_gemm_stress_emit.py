#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# grouped_gemm_stress_emit.py -- WIDE adversarial config set for the
# grouped-GEMM Python-vs-C parity stress test. Selects a config by argv[1]
# and prints _native_lower(arch='gfx950') of build_grouped_gemm.
import sys

from rocke.instances.common.grouped_gemm import (
    GroupedGemmSpec,
    build_grouped_gemm,
)
from rocke.instances.common.gemm_universal import TileSpec, TraitSpec

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify

# Each entry: (name, dtype, TileSpec kwargs, TraitSpec kwargs)
# The grouped_gemm IR is a function of (tile, trait, dtype, wave_size,
# block_size) only -- M/N/K problem shapes are runtime kernel args, so the
# "edge shapes" we push are TILE-geometry edges (smallest legal tile, prime-ish
# warp counts via large warp_m/n, deep tile_k, wide tiles near LDS cap) plus
# every dtype/atom/pipeline/epilogue/knob combination.
CONFIGS = [
    # 0-3: mirror the original sampled set (regression anchor)
    (
        "g_fp16_m128n128k32",
        "fp16",
        dict(
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
        dict(pipeline="compv4", epilogue="cshuffle"),
    ),
    (
        "g_bf16_m32n32k32",
        "bf16",
        dict(
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
        dict(pipeline="mem", epilogue="cshuffle", pad_m=True, pad_n=True),
    ),
    (
        "g_fp16_m64n64k64",
        "fp16",
        dict(
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
        dict(pipeline="compv3", epilogue="default"),
    ),
    (
        "g_fp16_m256n256k128",
        "fp16",
        dict(
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
        dict(pipeline="compv4", epilogue="cshuffle", chiplet_swizzle=True),
    ),
    # 4-7: smallest legal tiles per atom (edge: tile == warp*warp_tile, 1 warp)
    (
        "g_fp16_smallest_16atom",
        "fp16",
        dict(
            tile_m=16,
            tile_n=16,
            tile_k=16,
            warp_m=1,
            warp_n=1,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        dict(pipeline="mem", epilogue="default"),
    ),
    (
        "g_fp16_smallest_32atom",
        "fp16",
        dict(
            tile_m=32,
            tile_n=32,
            tile_k=8,
            warp_m=1,
            warp_n=1,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=8,
        ),
        dict(pipeline="mem", epilogue="default"),
    ),
    (
        "g_bf16_smallest_16x16x16",
        "bf16",
        dict(
            tile_m=16,
            tile_n=16,
            tile_k=16,
            warp_m=1,
            warp_n=1,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        dict(pipeline="mem", epilogue="default"),
    ),
    (
        "g_bf16_smallest_16x16x32",
        "bf16",
        dict(
            tile_m=16,
            tile_n=16,
            tile_k=32,
            warp_m=1,
            warp_n=1,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
        ),
        dict(pipeline="mem", epilogue="default"),
    ),
    # 8-11: all four f16 atoms at a common 64x64 tile shape
    (
        "g_fp16_atom16x16x16",
        "fp16",
        dict(
            tile_m=64,
            tile_n=64,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        dict(pipeline="compv4", epilogue="cshuffle"),
    ),
    (
        "g_fp16_atom16x16x32",
        "fp16",
        dict(
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
        ),
        dict(pipeline="compv4", epilogue="cshuffle"),
    ),
    (
        "g_fp16_atom32x32x8",
        "fp16",
        dict(
            tile_m=64,
            tile_n=64,
            tile_k=16,
            warp_m=1,
            warp_n=1,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=8,
        ),
        dict(pipeline="compv4", epilogue="cshuffle"),
    ),
    (
        "g_fp16_atom32x32x16",
        "fp16",
        dict(
            tile_m=64,
            tile_n=64,
            tile_k=32,
            warp_m=1,
            warp_n=1,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        dict(pipeline="compv4", epilogue="cshuffle"),
    ),
    # 12-15: pipeline sweep at a fixed shape (mem/compv3/compv4/wsp3)
    (
        "g_fp16_pipe_mem",
        "fp16",
        dict(
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
        dict(pipeline="mem", epilogue="default"),
    ),
    (
        "g_fp16_pipe_compv3",
        "fp16",
        dict(
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
        dict(pipeline="compv3", epilogue="default"),
    ),
    (
        "g_fp16_pipe_compv4_dir",
        "fp16",
        dict(
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
        dict(pipeline="compv4", epilogue="default"),
    ),
    (
        "g_fp16_pipe_wsp3",
        "fp16",
        dict(
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
        dict(pipeline="wsp3", epilogue="default"),
    ),
    # 16-18: scheduler + epilogue cross
    (
        "g_fp16_interwave",
        "fp16",
        dict(
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
        dict(pipeline="compv4", epilogue="cshuffle", scheduler="interwave"),
    ),
    (
        "g_fp16_intrawave_explicit",
        "fp16",
        dict(
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
        dict(pipeline="compv4", epilogue="cshuffle", scheduler="intrawave"),
    ),
    (
        "g_fp16_compv3_cshuffle",
        "fp16",
        dict(
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
        dict(pipeline="compv3", epilogue="cshuffle"),
    ),
    # 19-22: pad flags individually + all
    (
        "g_fp16_pad_k",
        "fp16",
        dict(
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
        dict(pipeline="compv4", epilogue="cshuffle", pad_k=True),
    ),
    (
        "g_fp16_pad_all",
        "fp16",
        dict(
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
        dict(
            pipeline="compv4", epilogue="cshuffle", pad_m=True, pad_n=True, pad_k=True
        ),
    ),
    (
        "g_bf16_pad_mem_default",
        "bf16",
        dict(
            tile_m=64,
            tile_n=64,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        dict(pipeline="mem", epilogue="default", pad_m=True, pad_n=True, pad_k=True),
    ),
    (
        "g_fp16_persistent",
        "fp16",
        dict(
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
        dict(pipeline="compv4", epilogue="cshuffle", persistent=True),
    ),
    # 23-25: chiplet swizzle param variations
    (
        "g_fp16_chiplet_wgm4",
        "fp16",
        dict(
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
        dict(
            pipeline="compv4", epilogue="cshuffle", chiplet_swizzle=True, chiplet_wgm=4
        ),
    ),
    (
        "g_fp16_chiplet_xcd4_chunk32",
        "fp16",
        dict(
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
        dict(
            pipeline="compv4",
            epilogue="cshuffle",
            chiplet_swizzle=True,
            chiplet_num_xcds=4,
            chiplet_chunk_size=32,
        ),
    ),
    (
        "g_fp16_waves_per_eu2",
        "fp16",
        dict(
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
        dict(pipeline="compv4", epilogue="cshuffle", waves_per_eu=2),
    ),
    # 26-28: LDS knobs
    (
        "g_fp16_lds_swizzle",
        "fp16",
        dict(
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
        dict(pipeline="compv4", epilogue="cshuffle", lds_swizzle=True),
    ),
    (
        "g_fp16_lds_k_pad8",
        "fp16",
        dict(
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
        dict(pipeline="compv4", epilogue="cshuffle", lds_k_pad=8),
    ),
    (
        "g_fp16_active_tile_skip",
        "fp16",
        dict(
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
        dict(pipeline="compv4", epilogue="cshuffle", active_tile_skip=True),
    ),
    # 29-31: deep K (many k-atoms per tile -> long unrolled loop -> SSA stress)
    (
        "g_fp16_deepk_256",
        "fp16",
        dict(
            tile_m=64,
            tile_n=64,
            tile_k=256,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        dict(pipeline="compv4", epilogue="cshuffle"),
    ),
    (
        "g_fp16_deepk_512",
        "fp16",
        dict(
            tile_m=64,
            tile_n=64,
            tile_k=512,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        dict(pipeline="mem", epilogue="default"),
    ),
    (
        "g_bf16_deepk_256_16x16x32",
        "bf16",
        dict(
            tile_m=32,
            tile_n=32,
            tile_k=256,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
        ),
        dict(pipeline="compv4", epilogue="cshuffle"),
    ),
    # 32-34: large/wide tiles (near LDS cap, many MFMAs/warp -> SSA numbering)
    (
        "g_fp16_wide_m256n128",
        "fp16",
        dict(
            tile_m=256,
            tile_n=128,
            tile_k=32,
            warp_m=4,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        dict(pipeline="compv4", epilogue="cshuffle"),
    ),
    (
        "g_fp16_tall_m128n256",
        "fp16",
        dict(
            tile_m=128,
            tile_n=256,
            tile_k=32,
            warp_m=2,
            warp_n=4,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        dict(pipeline="compv4", epilogue="cshuffle"),
    ),
    (
        "g_fp16_big_warpcount_8x8",
        "fp16",
        dict(
            tile_m=256,
            tile_n=256,
            tile_k=32,
            warp_m=8,
            warp_n=8,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        dict(pipeline="mem", epilogue="default"),
    ),
    # 35-37: asymmetric warp counts + non-square mfmas_per_warp
    (
        "g_fp16_asym_warp4x1",
        "fp16",
        dict(
            tile_m=256,
            tile_n=32,
            tile_k=32,
            warp_m=4,
            warp_n=1,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        dict(pipeline="compv4", epilogue="cshuffle"),
    ),
    (
        "g_fp16_asym_warp1x4",
        "fp16",
        dict(
            tile_m=32,
            tile_n=256,
            tile_k=32,
            warp_m=1,
            warp_n=4,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        dict(pipeline="compv4", epilogue="cshuffle"),
    ),
    (
        "g_fp16_multi_mfma_4x4",
        "fp16",
        dict(
            tile_m=256,
            tile_n=256,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        dict(pipeline="compv4", epilogue="cshuffle"),
    ),
    # 38-39: f16 alias "f16" string (dtype-normalization edge) + bf16 compv3
    (
        "g_f16alias_m128",
        "f16",
        dict(
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
        dict(pipeline="compv4", epilogue="cshuffle"),
    ),
    (
        "g_bf16_compv3_default",
        "bf16",
        dict(
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
        ),
        dict(pipeline="compv3", epilogue="default"),
    ),
]


def _spec(idx: int) -> GroupedGemmSpec:
    name, dtype, tk, trk = CONFIGS[idx]
    return GroupedGemmSpec(
        name=name,
        tile=TileSpec(**tk),
        trait=TraitSpec(**trk),
        dtype=dtype,
        wave_size=64,
    )


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: grouped_gemm_stress_emit.py <config_index> [ll|ir|verify]\n"
        )
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    if idx < 0 or idx >= len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    spec = _spec(idx)
    kernel = build_grouped_gemm(spec, arch="gfx950")
    if mode == "ll":
        text = _native_lower(kernel, arch="gfx950")
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    else:  # verify
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
