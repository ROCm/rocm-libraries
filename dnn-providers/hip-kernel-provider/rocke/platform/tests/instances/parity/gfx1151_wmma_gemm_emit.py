#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1151_wmma_gemm_emit.py -- Python reference emitter for the
# gfx1151 (RDNA3.5 / Strix Halo) WMMA GEMM parity harness. Selects one of 6
# sampled WmmaGemmSpec configs by argv[1] (0..5), builds it via build_wmma_gemm
# and prints lower_kernel_to_llvm(arch='gfx1151') to stdout so it can be
# byte-compared with the C emitter gfx1151_wmma_gemm_emit.c.
from rocke.instances.gfx1151.wmma_gemm import (
    WmmaGemmSpec,
    build_wmma_gemm,
)
from _emit_common import run_emit


def _spec(idx: int) -> WmmaGemmSpec:
    if idx == 0:
        return WmmaGemmSpec()
    if idx == 1:
        return WmmaGemmSpec(name="wmma_probe_gfx1151", block_x_is_m=True)
    if idx == 2:
        return WmmaGemmSpec(dtype="fp16", block_x_is_m=True)
    if idx == 3:
        return WmmaGemmSpec(name="rocke_wmma_gemm_v2", dtype="fp16", block_x_is_m=True)
    if idx == 4:
        return WmmaGemmSpec(name="wmma_gemm_tile16x16x16", block_x_is_m=False)
    if idx == 5:
        return WmmaGemmSpec(dtype="fp16", name="wmma_f16_16x16x16", block_x_is_m=False)
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_wmma_gemm,
        usage="usage: gfx1151_wmma_gemm_emit.py <config_index 0..5>\n",
        arch="gfx1151",
    )


if __name__ == "__main__":
    raise SystemExit(main())
