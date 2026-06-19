#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1201_wmma_gemm_emit.py -- Python reference emitter for the
# gfx1201 (RDNA4 / Navi48) WMMA GEMM parity harness. Selects one of 6 sampled
# WmmaGemmSpec configs by argv[1] (0..5), builds it via build_wmma_gemm and
# prints lower_kernel_to_llvm(arch='gfx1201') to stdout so it can be
# byte-compared with the C emitter gfx1201_wmma_gemm_emit.c.
import sys

from ck_dsl.instances.gfx1201.wmma_gemm import (
    WmmaGemmSpec,
    build_wmma_gemm,
)
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


def _spec(idx: int) -> WmmaGemmSpec:
    if idx == 0:
        return WmmaGemmSpec()
    if idx == 1:
        return WmmaGemmSpec(name="wmma_probe_gfx1201")
    if idx == 2:
        return WmmaGemmSpec(dtype="fp16")
    if idx == 3:
        return WmmaGemmSpec(name="ck_dsl_wmma_gemm_gfx12_v2", dtype="fp16")
    if idx == 4:
        return WmmaGemmSpec(name="wmma_gemm_tile16x16x16")
    if idx == 5:
        return WmmaGemmSpec(dtype="fp16", name="wmma_f16_16x16x16")
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gfx1201_wmma_gemm_emit.py <config_index 0..5>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    spec = _spec(idx)
    kernel = build_wmma_gemm(spec, arch="gfx1201")
    if mode == "ll":
        text = lower_kernel_to_llvm(kernel, arch="gfx1201")
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
