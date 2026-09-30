#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gemm_stress_emit.py -- WIDE adversarial Python reference emitter
# for the universal-GEMM parity harness. Mirrors gemm_stress_emit.c config table
# 1:1. Selects a config by argv[1] and prints _native_lower(gfx950).
import sys

from rocke.instances.common.gemm_universal import (
    UniversalGemmSpec,
    TileSpec,
    TraitSpec,
    DataSpec,
    build_universal_gemm,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


def _specs():
    S = []

    def add(
        name, tile, trait=None, data=None, wave_size=64, block_size=0, batched=False
    ):
        S.append(
            UniversalGemmSpec(
                name=name,
                tile=TileSpec(*tile),
                trait=(
                    trait
                    if trait is not None
                    else TraitSpec(pipeline="compv4", epilogue="default")
                ),
                data=data if data is not None else DataSpec(),
                wave_size=wave_size,
                block_size=block_size,
                batched=batched,
            )
        )

    # tile tuple = (tm,tn,tk, wm,wn,wk, wtm,wtn,wtk)
    # ---- atom coverage f16 (all 4 atoms) ----
    add(
        "s00",
        (64, 64, 16, 1, 1, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=64,
    )
    add(
        "s01",
        (64, 64, 32, 1, 1, 1, 16, 16, 32),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=64,
    )
    add(
        "s02",
        (64, 64, 16, 1, 1, 1, 32, 32, 8),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=64,
    )
    add(
        "s03",
        (64, 64, 16, 1, 1, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=64,
    )
    # ---- atom coverage bf16 (2 atoms) ----
    add(
        "s04",
        (64, 64, 16, 1, 1, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
        block_size=64,
    )
    add(
        "s05",
        (128, 128, 32, 2, 2, 1, 16, 16, 32),
        TraitSpec(pipeline="compv4", epilogue="cshuffle"),
        DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
        block_size=256,
    )
    # ---- warp grid coverage ----
    add(
        "s06",
        (32, 64, 16, 1, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=128,
    )
    add(
        "s07",
        (64, 32, 16, 2, 1, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=128,
    )
    add(
        "s08",
        (64, 128, 16, 1, 4, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=256,
    )
    add(
        "s09",
        (128, 64, 16, 4, 1, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=256,
    )
    add(
        "s10",
        (64, 128, 16, 2, 4, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=512,
    )
    add(
        "s11",
        (128, 64, 16, 4, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=512,
    )
    # ---- pipeline coverage ----
    add(
        "s12",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="mem", epilogue="default"),
        block_size=256,
    )
    add(
        "s13",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv3", epilogue="default"),
        block_size=256,
    )
    add(
        "s14",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv3", epilogue="cshuffle"),
        block_size=256,
    )
    # ---- epilogue cshuffle wide ----
    add(
        "s15",
        (256, 128, 32, 2, 4, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="cshuffle"),
        block_size=512,
    )
    # ---- non-square / "prime-ish" tile dims ----
    add(
        "s16",
        (192, 64, 16, 2, 1, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=128,
    )
    add(
        "s17",
        (96, 96, 16, 1, 1, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=64,
    )
    add(
        "s18",
        (224, 32, 16, 1, 1, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=64,
    )
    add(
        "s19",
        (320, 64, 16, 1, 2, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=128,
    )
    # ---- large K (many k-atoms) ----
    add(
        "s20",
        (128, 128, 128, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=256,
    )
    add(
        "s21",
        (128, 128, 256, 2, 2, 1, 16, 16, 32),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=256,
    )
    # ---- knob: chiplet_swizzle (non-default chiplet params) ----
    add(
        "s22",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(
            pipeline="compv4",
            epilogue="default",
            chiplet_swizzle=True,
            chiplet_wgm=4,
            chiplet_num_xcds=4,
            chiplet_chunk_size=32,
        ),
        block_size=256,
    )
    # ---- knob: lds_swizzle ----
    add(
        "s23",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", lds_swizzle=True),
        block_size=256,
    )
    # ---- knob: lds_k_pad ----
    add(
        "s24",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", lds_k_pad=16),
        block_size=256,
    )
    # ---- knob: direct_to_lds ----
    add(
        "s25",
        (128, 128, 64, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", direct_to_lds=True),
        block_size=256,
    )
    # ---- knob: direct_to_lds + dtl_prefetch ----
    add(
        "s26",
        (128, 128, 64, 2, 2, 1, 16, 16, 16),
        TraitSpec(
            pipeline="compv4", epilogue="default", direct_to_lds=True, dtl_prefetch=True
        ),
        block_size=256,
    )
    # ---- knob: persistent ----
    add(
        "s27",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", persistent=True),
        block_size=256,
    )
    # ---- knob: pad_m/n/k ----
    add(
        "s28",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(
            pipeline="compv4", epilogue="default", pad_m=True, pad_n=True, pad_k=True
        ),
        block_size=256,
    )
    # ---- knob: waves_per_eu ----
    add(
        "s29",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", waves_per_eu=2),
        block_size=256,
    )
    # ---- knob: scheduler interwave ----
    add(
        "s30",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", scheduler="interwave", epilogue="default"),
        block_size=256,
    )
    # ---- batched ----
    add(
        "s31",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=256,
        batched=True,
    )
    # ---- batched + active_tile_skip (MoE) ----
    add(
        "s32",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", active_tile_skip=True),
        block_size=256,
        batched=True,
    )
    # ---- knob: preshuffle_b ----
    add(
        "s33",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", preshuffle_b=True),
        block_size=256,
    )
    # ---- wsp3 pipeline ----
    add(
        "s34",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="wsp3", epilogue="default"),
        block_size=256,
    )
    # ---- biggest valid square ----
    add(
        "s35",
        (256, 256, 64, 4, 4, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=1024,
    )
    # ---- 32x32x8 atom with bigger tile ----
    add(
        "s36",
        (128, 128, 32, 2, 2, 1, 32, 32, 8),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=256,
    )
    # ---- gfx942-style wave (still gfx950 lower) edge: warp_tile 16x16x32 + cshuffle 32 ----
    add(
        "s37",
        (256, 256, 32, 4, 4, 1, 16, 16, 32),
        TraitSpec(pipeline="compv4", epilogue="cshuffle"),
        block_size=1024,
    )
    return S


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gemm_stress_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    specs = _specs()
    if idx < 0 or idx >= len(specs):
        sys.stderr.write(f"unknown config index {idx}\n")
        return 2
    spec = specs[idx]
    kernel = build_universal_gemm(spec, arch="gfx950")
    if mode == "ll":
        text = _native_lower(kernel, arch="gfx950")
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
