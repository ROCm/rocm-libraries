#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1151_wmma_gemm_int8_emit.py -- Python reference emitter for the
# gfx1151 (RDNA3.5 / Strix Halo) INT8-storage / f16-compute WMMA GEMM parity
# harness. Selects one of 6 sampled WmmaGemmInt8Spec configs by argv[1] (0..5),
# builds it via build_wmma_gemm_int8 and prints
# lower_kernel_to_llvm(arch='gfx1151') to stdout so it can be byte-compared with
# the C emitter gfx1151_wmma_gemm_int8_emit.c.
import sys

from ck_dsl.instances.gfx1151.wmma_gemm_int8 import (
    WmmaGemmInt8Spec,
    build_wmma_gemm_int8,
)
from ck_dsl import lower_kernel_to_llvm


def _spec(idx: int) -> WmmaGemmInt8Spec:
    if idx == 0:
        return WmmaGemmInt8Spec(name="ck_dsl_wmma_gemm_int8", dtype="i8")
    if idx == 1:
        return WmmaGemmInt8Spec(name="wmma_int8_probe_gfx1151", dtype="i8")
    if idx == 2:
        return WmmaGemmInt8Spec(name="ck_dsl_wmma_gemm_int8_v2", dtype="i8")
    if idx == 3:
        return WmmaGemmInt8Spec(name="wmma_gemm_int8_tile16x16x16", dtype="i8")
    if idx == 4:
        return WmmaGemmInt8Spec(name="wmma_int8_dequant_f16_out", dtype="i8")
    if idx == 5:
        return WmmaGemmInt8Spec(name="wmma_path_b_int8_f16", dtype="i8")
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gfx1151_wmma_gemm_int8_emit.py <config_index 0..5>\n")
        return 2
    idx = int(sys.argv[1])
    spec = _spec(idx)
    kernel = build_wmma_gemm_int8(spec, arch="gfx1151")
    text = lower_kernel_to_llvm(kernel, arch="gfx1151")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
