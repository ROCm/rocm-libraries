#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/mfma_gemm_emit.py -- Python reference emitter for the MFMA-GEMM
# instance parity harness. Selects one of the sampled configs by argv[1] (the
# config index), builds the MfmaGemmSpec, builds the kernel via build_mfma_gemm
# and prints lower_kernel_to_llvm(arch='gfx950') to stdout so it can be
# byte-compared with the C emitter mfma_gemm_emit.c.
import sys

from ck_dsl.instances.common.mfma_gemm import MfmaGemmSpec, build_mfma_gemm
from ck_dsl import lower_kernel_to_llvm


def _spec(idx: int) -> MfmaGemmSpec:
    if idx == 0:
        return MfmaGemmSpec(
            M=256, N=256, K=256, dtype="f16", tile_m=16, tile_n=16, kpack=True
        )
    if idx == 1:
        return MfmaGemmSpec(
            M=512, N=512, K=256, dtype="f16", tile_m=32, tile_n=32, kpack=True
        )
    if idx == 2:
        return MfmaGemmSpec(
            M=256, N=256, K=512, dtype="bf16", tile_m=16, tile_n=16, kpack=True
        )
    if idx == 3:
        return MfmaGemmSpec(
            M=256, N=256, K=256, dtype="f16", tile_m=16, tile_n=16, kpack=False
        )
    if idx == 4:
        return MfmaGemmSpec(
            M=512, N=512, K=512, dtype="f16", tile_m=32, tile_n=32, kpack=False
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: mfma_gemm_emit.py <config_index 0..4>\n")
        return 2
    idx = int(sys.argv[1])
    spec = _spec(idx)
    kernel = build_mfma_gemm(spec, arch="gfx950")
    text = lower_kernel_to_llvm(kernel, arch="gfx950")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
