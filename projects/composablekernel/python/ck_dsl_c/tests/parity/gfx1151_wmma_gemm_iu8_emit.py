#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1151_wmma_gemm_iu8_emit.py -- Python reference emitter for the
# gfx1151 (RDNA3.5) native-integer WMMA GEMM (iu8) parity harness. Selects one
# of 6 sampled configs by argv[1] (0..5), builds it via build_wmma_gemm_iu8 and
# prints lower_kernel_to_llvm(arch='gfx1151') to stdout so it can be
# byte-compared with the C emitter gfx1151_wmma_gemm_iu8_emit.c.
#
# NOTE: M/N/K in each config are runtime kernel parameters (they drive the
# launch grid, not the build), so the emitted IR is identical across configs.
# The config index is kept so the two emitters stay structurally in lock-step.
import sys

from ck_dsl.instances.gfx1151.wmma_gemm_iu8 import (
    WmmaGemmIu8Spec,
    build_wmma_gemm_iu8,
)
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


def _spec(idx: int) -> WmmaGemmIu8Spec:
    if 0 <= idx <= 5:
        return WmmaGemmIu8Spec(name="ck_dsl_wmma_gemm_iu8")
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gfx1151_wmma_gemm_iu8_emit.py <config_index 0..5>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    spec = _spec(idx)
    kernel = build_wmma_gemm_iu8(spec, arch="gfx1151")
    if mode == "ll":
        text = lower_kernel_to_llvm(kernel, arch="gfx1151")
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
