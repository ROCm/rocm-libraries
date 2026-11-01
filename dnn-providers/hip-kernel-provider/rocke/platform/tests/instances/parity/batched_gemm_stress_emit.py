#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/batched_gemm_stress_emit.py -- WIDE adversarial Python reference
# emitter for the batched-GEMM parity harness. Mirrors
# batched_gemm_stress_emit.c config table 1:1. Selects a config by argv[1] and
# prints _native_lower(gfx950).
#
# batched_gemm reuses the universal-GEMM body in batched mode (batched=True),
# so this sweep exercises the same atom / warp-grid / pipeline / epilogue /
# knob matrix as gemm_stress, but routed through BatchedGemmSpec ->
# build_batched_gemm so the batch-offset prologue + active_tile_skip (MoE)
# paths are covered too.
import sys

from rocke.instances.common.batched_gemm import (
    BatchedGemmSpec,
    build_batched_gemm,
)
from rocke.instances.common.gemm_universal import TileSpec, TraitSpec

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


def _specs():
    S = []

    def add(name, tile, trait=None, dtype="fp16", block_size=0):
        S.append(
            BatchedGemmSpec(
                name=name,
                tile=TileSpec(*tile),
                trait=(
                    trait
                    if trait is not None
                    else TraitSpec(pipeline="compv4", epilogue="default")
                ),
                dtype=dtype,
                wave_size=64,
                block_size=block_size,
            )
        )

    # tile tuple = (tm,tn,tk, wm,wn,wk, wtm,wtn,wtk)
    # ---- atom coverage f16 (all 4 atoms) ----
    add(
        "b00",
        (64, 64, 16, 1, 1, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=64,
    )
    add(
        "b01",
        (64, 64, 32, 1, 1, 1, 16, 16, 32),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=64,
    )
    add(
        "b02",
        (64, 64, 16, 1, 1, 1, 32, 32, 8),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=64,
    )
    add(
        "b03",
        (64, 64, 16, 1, 1, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=64,
    )
    # ---- atom coverage bf16 (2 atoms) ----
    add(
        "b04",
        (64, 64, 16, 1, 1, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        dtype="bf16",
        block_size=64,
    )
    add(
        "b05",
        (128, 128, 32, 2, 2, 1, 16, 16, 32),
        TraitSpec(pipeline="compv4", epilogue="cshuffle"),
        dtype="bf16",
        block_size=256,
    )
    # ---- warp grid coverage ----
    add(
        "b06",
        (32, 64, 16, 1, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=128,
    )
    add(
        "b07",
        (64, 32, 16, 2, 1, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=128,
    )
    add(
        "b08",
        (64, 128, 16, 1, 4, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=256,
    )
    add(
        "b09",
        (128, 64, 16, 4, 1, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=256,
    )
    add(
        "b10",
        (64, 128, 16, 2, 4, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=512,
    )
    add(
        "b11",
        (128, 64, 16, 4, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=512,
    )
    # ---- pipeline coverage ----
    add(
        "b12",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="mem", epilogue="default"),
        block_size=256,
    )
    add(
        "b13",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv3", epilogue="default"),
        block_size=256,
    )
    add(
        "b14",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv3", epilogue="cshuffle"),
        block_size=256,
    )
    # ---- epilogue cshuffle wide ----
    add(
        "b15",
        (256, 128, 32, 2, 4, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="cshuffle"),
        block_size=512,
    )
    # ---- non-square / "prime-ish" tile dims ----
    add(
        "b16",
        (192, 64, 16, 2, 1, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=128,
    )
    add(
        "b17",
        (96, 96, 16, 1, 1, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=64,
    )
    add(
        "b18",
        (224, 32, 16, 1, 1, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=64,
    )
    add(
        "b19",
        (320, 64, 16, 1, 2, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=128,
    )
    # ---- large K (many k-atoms) ----
    add(
        "b20",
        (128, 128, 128, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=256,
    )
    add(
        "b21",
        (128, 128, 256, 2, 2, 1, 16, 16, 32),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=256,
    )
    # ---- knob: chiplet_swizzle (non-default chiplet params) ----
    add(
        "b22",
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
        "b23",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", lds_swizzle=True),
        block_size=256,
    )
    # ---- knob: lds_k_pad ----
    add(
        "b24",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", lds_k_pad=16),
        block_size=256,
    )
    # ---- knob: direct_to_lds ----
    add(
        "b25",
        (128, 128, 64, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", direct_to_lds=True),
        block_size=256,
    )
    # ---- knob: direct_to_lds + dtl_prefetch ----
    add(
        "b26",
        (128, 128, 64, 2, 2, 1, 16, 16, 16),
        TraitSpec(
            pipeline="compv4", epilogue="default", direct_to_lds=True, dtl_prefetch=True
        ),
        block_size=256,
    )
    # ---- knob: persistent ----
    add(
        "b27",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", persistent=True),
        block_size=256,
    )
    # ---- knob: pad_m/n/k ----
    add(
        "b28",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(
            pipeline="compv4", epilogue="default", pad_m=True, pad_n=True, pad_k=True
        ),
        block_size=256,
    )
    # ---- knob: waves_per_eu ----
    add(
        "b29",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", waves_per_eu=2),
        block_size=256,
    )
    # ---- knob: scheduler interwave ----
    add(
        "b30",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", scheduler="interwave", epilogue="default"),
        block_size=256,
    )
    # ---- active_tile_skip (MoE batched) ----
    add(
        "b31",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", active_tile_skip=True),
        block_size=256,
    )
    # ---- active_tile_skip + cshuffle ----
    add(
        "b32",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="cshuffle", active_tile_skip=True),
        block_size=256,
    )
    # ---- knob: preshuffle_b ----
    add(
        "b33",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", preshuffle_b=True),
        block_size=256,
    )
    # ---- compv3 + active_tile_skip (MoE batched, mem-light pipeline) ----
    # (wsp3 is a separate, not-yet-ported C emitter; excluded from the
    # batched-GEMM parity sweep so every listed config exercises the
    # batched_gemm.c emission path that IS port-complete.)
    add(
        "b34",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv3", epilogue="default", active_tile_skip=True),
        block_size=256,
    )
    # ---- biggest valid square ----
    add(
        "b35",
        (256, 256, 64, 4, 4, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=1024,
    )
    # ---- 32x32x8 atom with bigger tile ----
    add(
        "b36",
        (128, 128, 32, 2, 2, 1, 32, 32, 8),
        TraitSpec(pipeline="compv4", epilogue="default"),
        block_size=256,
    )
    # ---- 16x16x32 warp-tile + cshuffle 32 (wide-K edge) ----
    add(
        "b37",
        (256, 256, 32, 4, 4, 1, 16, 16, 32),
        TraitSpec(pipeline="compv4", epilogue="cshuffle"),
        block_size=1024,
    )
    # ---- bf16 + active_tile_skip ----
    add(
        "b38",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", active_tile_skip=True),
        dtype="bf16",
        block_size=256,
    )
    # ---- bf16 wide-K atom 16x16x32 ----
    add(
        "b39",
        (128, 128, 64, 2, 2, 1, 16, 16, 32),
        TraitSpec(pipeline="compv4", epilogue="default"),
        dtype="bf16",
        block_size=256,
    )
    # ---- compv3 + cshuffle + 32x32x16 ----
    add(
        "b40",
        (128, 128, 32, 2, 2, 1, 32, 32, 16),
        TraitSpec(pipeline="compv3", epilogue="cshuffle"),
        block_size=256,
    )
    # ---- mem pipeline + 1x1 warp ----
    add(
        "b41",
        (64, 64, 32, 1, 1, 1, 32, 32, 16),
        TraitSpec(pipeline="mem", epilogue="default"),
        block_size=64,
    )
    # ---- direct_to_lds + cshuffle ----
    add(
        "b42",
        (128, 128, 64, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="cshuffle", direct_to_lds=True),
        block_size=256,
    )
    # ---- persistent + cshuffle ----
    add(
        "b43",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="cshuffle", persistent=True),
        block_size=256,
    )
    # ---- pad_m/n/k + cshuffle ----
    add(
        "b44",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(
            pipeline="compv4", epilogue="cshuffle", pad_m=True, pad_n=True, pad_k=True
        ),
        block_size=256,
    )
    # ---- chiplet_swizzle defaults (wgm=8, xcds=8, chunk=64) ----
    add(
        "b45",
        (128, 128, 32, 2, 2, 1, 16, 16, 16),
        TraitSpec(pipeline="compv4", epilogue="default", chiplet_swizzle=True),
        block_size=256,
    )
    # ---- asymmetric warp 4x2 + cshuffle ----
    add(
        "b46",
        (256, 128, 32, 4, 2, 1, 32, 32, 16),
        TraitSpec(pipeline="compv4", epilogue="cshuffle"),
        block_size=512,
    )
    return S


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: batched_gemm_stress_emit.py <config_index> [ll|ir|verify]\n"
        )
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    specs = _specs()
    if idx < 0 or idx >= len(specs):
        sys.stderr.write(f"unknown config index {idx}\n")
        return 2
    spec = specs[idx]
    kernel = build_batched_gemm(spec, arch="gfx950")
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
