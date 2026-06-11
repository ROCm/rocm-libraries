#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/moe_fused_mega_fp8_emit.py -- Python reference emitter for the FP8
# fused-MoE MEGA-kernel parity harness. Selects one of N sampled spec configs by
# argv[1], builds FusedMegaKernelSpecFp8, builds via build_moe_fused_mega_gemm_fp8
# and prints lower_kernel_to_llvm(arch='gfx950') to stdout so it can be
# byte-compared with the C emitter moe_fused_mega_fp8_emit.c.
import sys

from ck_dsl.instances.common.moe_fused_mega_fp8 import (
    FusedMegaKernelSpecFp8,
    build_moe_fused_mega_gemm_fp8,
)
from ck_dsl import lower_kernel_to_llvm


def _spec(idx: int):
    common = dict(tile_m=16, tile_n_inter=256)
    if idx == 0:
        return FusedMegaKernelSpecFp8(
            name="moe_fused_mega_fp8_baseline",
            gate_up_k=32,
            down_k=32,
            use_dtla=False,
            sched_cadence=None,
            **common,
        ), False
    if idx == 1:
        return FusedMegaKernelSpecFp8(
            name="moe_fused_mega_fp8_l7_hero",
            gate_up_k=128,
            down_k=128,
            use_dtla=False,
            sched_cadence=None,
            **common,
        ), False
    if idx == 2:
        return FusedMegaKernelSpecFp8(
            name="moe_fused_mega_fp8_l8_dtla",
            gate_up_k=128,
            down_k=128,
            use_dtla=True,
            sched_cadence="none",
            **common,
        ), False
    if idx == 3:
        return FusedMegaKernelSpecFp8(
            name="moe_fused_mega_fp8_l9_iglp",
            gate_up_k=128,
            down_k=128,
            use_dtla=True,
            sched_cadence="iglp1",
            **common,
        ), False
    if idx == 4:
        return FusedMegaKernelSpecFp8(
            name="moe_fused_mega_fp8_prod",
            gate_up_k=128,
            down_k=128,
            use_dtla=True,
            sched_cadence="iglp1",
            **common,
        ), False
    if idx == 5:
        return FusedMegaKernelSpecFp8(
            name="moe_fused_mega_fp8_persistent",
            gate_up_k=128,
            down_k=128,
            use_dtla=True,
            sched_cadence="iglp1",
            **common,
        ), True
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: moe_fused_mega_fp8_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    spec, persistent = _spec(idx)
    kernel = build_moe_fused_mega_gemm_fp8(spec, arch="gfx950", persistent=persistent)
    text = lower_kernel_to_llvm(kernel, arch="gfx950")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
