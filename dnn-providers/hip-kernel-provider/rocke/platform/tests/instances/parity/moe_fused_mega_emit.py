#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/moe_fused_mega_emit.py -- Python reference emitter for the
# moe_fused_mega parity harness. Selects one of N sampled spec configs by
# argv[1], builds the FusedMegaKernelSpec, builds the kernel via
# build_moe_fused_mega_gemm and prints lower_kernel_to_llvm(arch='gfx950') to
# stdout so it can be byte-compared with the C emitter moe_fused_mega_emit.c.
from rocke.instances.common.moe_fused_mega import (
    FusedMegaKernelSpec,
    build_moe_fused_mega_gemm,
)
from _emit_common import run_emit


def _spec(idx: int) -> FusedMegaKernelSpec:
    if idx == 0:
        return FusedMegaKernelSpec(
            name="moe_mega_baseline",
            tile_m=32,
            tile_n_inter=256,
            tile_k_gu=32,
            tile_n_down=256,
            tile_k_down=64,
            dtype="fp16",
        )
    if idx == 1:
        return FusedMegaKernelSpec(
            name="moe_mega_tuned_m16",
            tile_m=16,
            tile_n_inter=256,
            tile_k_gu=32,
            tile_n_down=256,
            tile_k_down=64,
            dtype="fp16",
        )
    if idx == 2:
        return FusedMegaKernelSpec(
            name="moe_mega_large_k",
            tile_m=32,
            tile_n_inter=256,
            tile_k_gu=64,
            tile_n_down=256,
            tile_k_down=128,
            dtype="fp16",
        )
    if idx == 3:
        return FusedMegaKernelSpec(
            name="moe_mega_wide_n",
            tile_m=32,
            tile_n_inter=512,
            tile_k_gu=32,
            tile_n_down=512,
            tile_k_down=64,
            dtype="fp16",
        )
    if idx == 4:
        return FusedMegaKernelSpec(
            name="moe_mega_fp8",
            tile_m=32,
            tile_n_inter=256,
            tile_k_gu=32,
            tile_n_down=256,
            tile_k_down=64,
            dtype="fp8e4m3",
        )
    if idx == 5:
        return FusedMegaKernelSpec(
            name="moe_mega_bf16",
            tile_m=32,
            tile_n_inter=256,
            tile_k_gu=32,
            tile_n_down=256,
            tile_k_down=64,
            dtype="bf16",
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_moe_fused_mega_gemm,
        usage="usage: moe_fused_mega_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
