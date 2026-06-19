#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/moe_fused_mega_emit.py -- Python reference emitter for the
# moe_fused_mega parity harness. Selects one of N sampled spec configs by
# argv[1], builds the FusedMegaKernelSpec, builds the kernel via
# build_moe_fused_mega_gemm and prints lower_kernel_to_llvm(arch='gfx950') to
# stdout so it can be byte-compared with the C emitter moe_fused_mega_emit.c.
import sys

from ck_dsl.instances.common.moe_fused_mega import (
    FusedMegaKernelSpec,
    build_moe_fused_mega_gemm,
)
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


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
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: moe_fused_mega_emit.py <config_index> [ll|ir|verify]\n"
        )
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    spec = _spec(idx)
    kernel = build_moe_fused_mega_gemm(spec, arch="gfx950")
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
