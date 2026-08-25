#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1151_wmma_gemm_iu8_dequant_emit.py -- Python reference emitter
# for the gfx1151 (RDNA3.5) true-INT8 WMMA GEMM with f16 dequant output parity
# harness. Selects one of 6 sampled configs by argv[1] (0..5), builds it via
# build_wmma_gemm_iu8_dequant and prints lower_kernel_to_llvm(arch='gfx1151') to
# stdout so it can be byte-compared with the C emitter
# gfx1151_wmma_gemm_iu8_dequant_emit.c.
#
# NOTE: M/N/K in each config are runtime kernel parameters (they drive the
# launch grid, not the build), so the emitted IR is identical across configs.
# The config index is kept so the two emitters stay structurally in lock-step.
from rocke.instances.gfx1151.wmma_gemm_iu8_dequant import (
    WmmaGemmIu8DequantSpec,
    build_wmma_gemm_iu8_dequant,
)
from _emit_common import run_emit


def _spec(idx: int) -> WmmaGemmIu8DequantSpec:
    if 0 <= idx <= 5:
        return WmmaGemmIu8DequantSpec(name="rocke_wmma_gemm_iu8_dequant")
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_wmma_gemm_iu8_dequant,
        usage="usage: gfx1151_wmma_gemm_iu8_dequant_emit.py <config_index 0..5>\n",
        arch="gfx1151",
    )


if __name__ == "__main__":
    raise SystemExit(main())
