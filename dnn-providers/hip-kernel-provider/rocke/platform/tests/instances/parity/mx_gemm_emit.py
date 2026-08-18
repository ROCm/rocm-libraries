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
    common = dict(
        mantissa_dtype="fp8e4m3",
        group_k=32,
        block_tile_m=16,
        block_tile_n=16,
        name="rocke_mx_gemm",
        per_input_row=True,
    )
    if idx == 0:
        return MxGemmSpec(M=16, N=16, K=64, **{**common, "mantissa_dtype": "fp8e4m3"})
    if idx == 1:
        return MxGemmSpec(M=32, N=32, K=64, **{**common, "mantissa_dtype": "fp8e4m3"})
    if idx == 2:
        return MxGemmSpec(M=16, N=16, K=32, **{**common, "mantissa_dtype": "bf8e5m2"})
    if idx == 3:
        return MxGemmSpec(M=32, N=32, K=128, **{**common, "mantissa_dtype": "fp8e4m3"})
    if idx == 4:
        return MxGemmSpec(M=48, N=64, K=96, **{**common, "mantissa_dtype": "fp8e4m3"})
    if idx == 5:
        return MxGemmSpec(M=64, N=48, K=160, **{**common, "mantissa_dtype": "bf8e5m2"})
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_mx_gemm,
        usage="usage: mx_gemm_emit.py <config_index 0..5> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
