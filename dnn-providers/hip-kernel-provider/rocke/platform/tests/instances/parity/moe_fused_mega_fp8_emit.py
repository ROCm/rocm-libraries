#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/moe_fused_mega_fp8_emit.py -- Python reference emitter for the FP8
# fused-MoE MEGA-kernel parity harness. Selects one of N sampled spec configs by
# argv[1], builds FusedMegaKernelSpecFp8, builds via build_moe_fused_mega_gemm_fp8
# and prints _native_lower(arch='gfx950') to stdout so it can be
# byte-compared with the C emitter moe_fused_mega_fp8_emit.c.
import sys

from rocke.instances.common.moe_fused_mega_fp8 import (
    FusedMegaKernelSpecFp8,
    build_moe_fused_mega_gemm_fp8,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


def _spec(idx: int):
    common = dict(tile_m=16, tile_n_inter=256)
    if idx == 0:
        return (
            FusedMegaKernelSpecFp8(
                name="moe_fused_mega_fp8_baseline",
                gate_up_k=32,
                down_k=32,
                use_dtla=False,
                sched_cadence=None,
                **common,
            ),
            False,
        )
    if idx == 1:
        return (
            FusedMegaKernelSpecFp8(
                name="moe_fused_mega_fp8_l7_hero",
                gate_up_k=128,
                down_k=128,
                use_dtla=False,
                sched_cadence=None,
                **common,
            ),
            False,
        )
    if idx == 2:
        return (
            FusedMegaKernelSpecFp8(
                name="moe_fused_mega_fp8_l8_dtla",
                gate_up_k=128,
                down_k=128,
                use_dtla=True,
                sched_cadence="none",
                **common,
            ),
            False,
        )
    if idx == 3:
        return (
            FusedMegaKernelSpecFp8(
                name="moe_fused_mega_fp8_l9_iglp",
                gate_up_k=128,
                down_k=128,
                use_dtla=True,
                sched_cadence="iglp1",
                **common,
            ),
            False,
        )
    if idx == 4:
        return (
            FusedMegaKernelSpecFp8(
                name="moe_fused_mega_fp8_prod",
                gate_up_k=128,
                down_k=128,
                use_dtla=True,
                sched_cadence="iglp1",
                **common,
            ),
            False,
        )
    if idx == 5:
        return (
            FusedMegaKernelSpecFp8(
                name="moe_fused_mega_fp8_persistent",
                gate_up_k=128,
                down_k=128,
                use_dtla=True,
                sched_cadence="iglp1",
                **common,
            ),
            True,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: moe_fused_mega_fp8_emit.py <config_index> [ll|ir|verify]\n"
        )
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    spec, persistent = _spec(idx)
    kernel = build_moe_fused_mega_gemm_fp8(spec, arch="gfx950", persistent=persistent)
    if mode == "ll":
        text = _native_lower(kernel, arch="gfx950")
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    else:  # verify
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
