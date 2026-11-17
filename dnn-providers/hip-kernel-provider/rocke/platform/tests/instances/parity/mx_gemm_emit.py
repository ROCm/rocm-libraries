#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/mx_gemm_emit.py -- Python reference emitter for the MX-GEMM
# parity harness. Selects one of 6 sampled mx_gemm configs by argv[1] (index
# 0..5), builds the MxGemmSpec, builds the kernel via build_mx_gemm and prints
# lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared with
# the C emitter mx_gemm_emit.c.
from rocke.instances.common.mx_gemm import MxGemmSpec, build_mx_gemm
from _emit_common import run_emit


def _spec(idx: int) -> MxGemmSpec:
    common = {"group_k": 32, "name": "rocke_mx_gemm"}
    # (M, N, K, mantissa, block_tile_m, block_tile_n): whole and partial
    # workgroup tiles, a K that is not a multiple of 64 (one group added
    # without the scaled MFMA) and an N that is not a multiple of 32 (stores
    # past N moved past the end of C).
    configs = (
        (16, 16, 64, "fp8e4m3", 32, 32),
        (32, 32, 64, "fp8e4m3", 32, 256),
        (16, 16, 32, "bf8e5m2", 32, 32),
        (32, 32, 128, "fp8e4m3", 64, 64),
        (48, 64, 96, "fp8e4m3", 32, 64),
        (64, 48, 160, "bf8e5m2", 64, 32),
    )
    if not 0 <= idx < len(configs):
        raise SystemExit(f"unknown config index {idx}")
    M, N, K, mantissa, tile_m, tile_n = configs[idx]
    return MxGemmSpec(
        M=M,
        N=N,
        K=K,
        mantissa_dtype=mantissa,
        block_tile_m=tile_m,
        block_tile_n=tile_n,
        **common,
    )


def main() -> int:
    return run_emit(
        _spec,
        build_mx_gemm,
        usage="usage: mx_gemm_emit.py <config_index 0..5> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
