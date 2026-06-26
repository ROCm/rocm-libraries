#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1250_wmma_gemm_emit.py -- Python reference emitter for the
# gfx1250 WMMA GEMM (K=32) parity harness.
from rocke.instances.gfx1250.wmma_gemm import (
    WmmaGemmSpec,
    build_wmma_gemm,
)
from _emit_common import run_emit


def _spec(idx: int) -> WmmaGemmSpec:
    if idx == 0:
        return WmmaGemmSpec()
    if idx == 1:
        return WmmaGemmSpec(name="wmma_probe_gfx1250")
    if idx == 2:
        return WmmaGemmSpec(dtype="fp16")
    if idx == 3:
        return WmmaGemmSpec(name="rocke_wmma_gemm_gfx1250_v2", dtype="fp16")
    if idx == 4:
        return WmmaGemmSpec(name="wmma_gemm_tile16x16x32")
    if idx == 5:
        return WmmaGemmSpec(dtype="fp16", name="wmma_f16_16x16x32")
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_wmma_gemm,
        usage="usage: gfx1250_wmma_gemm_emit.py <config_index 0..5>\n",
        arch="gfx1250",
    )


if __name__ == "__main__":
    raise SystemExit(main())
