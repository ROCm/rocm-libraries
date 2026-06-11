#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/add_rmsnorm2d_bf16_emit.py -- Python reference emitter for the
# fused add + RMSNorm (bf16/f16 output) parity harness. Selects one of the
# sampled AddRMSNorm2DBF16Spec configs by argv[1], builds it via
# build_add_rmsnorm2d_bf16 and prints lower_kernel_to_llvm(arch='gfx950') to
# stdout so it can be byte-compared with the C emitter
# add_rmsnorm2d_bf16_emit.c.
import sys

from ck_dsl.instances.common.add_rmsnorm2d_bf16 import (
    AddRMSNorm2DBF16Spec,
    build_add_rmsnorm2d_bf16,
)
from ck_dsl import lower_kernel_to_llvm


def _spec(idx: int) -> AddRMSNorm2DBF16Spec:
    if idx == 0:
        return AddRMSNorm2DBF16Spec(
            n_per_block=1024,
            block_size=256,
            vec=4,
            dtype="bf16",
            save_residual=True,
            wave_size=64,
        )
    if idx == 1:
        return AddRMSNorm2DBF16Spec(
            n_per_block=2048,
            block_size=256,
            vec=4,
            dtype="bf16",
            save_residual=True,
            wave_size=64,
        )
    if idx == 2:
        return AddRMSNorm2DBF16Spec(
            n_per_block=4096,
            block_size=256,
            vec=4,
            dtype="bf16",
            save_residual=True,
            wave_size=64,
        )
    if idx == 3:
        return AddRMSNorm2DBF16Spec(
            n_per_block=8192,
            block_size=256,
            vec=4,
            dtype="bf16",
            save_residual=True,
            wave_size=64,
        )
    if idx == 4:
        return AddRMSNorm2DBF16Spec(
            n_per_block=1024,
            block_size=256,
            vec=2,
            dtype="f16",
            save_residual=False,
            wave_size=64,
        )
    if idx == 5:
        return AddRMSNorm2DBF16Spec(
            n_per_block=2048,
            block_size=128,
            vec=4,
            dtype="bf16",
            save_residual=True,
            wave_size=64,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: add_rmsnorm2d_bf16_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    spec = _spec(idx)
    kernel = build_add_rmsnorm2d_bf16(spec, arch="gfx950")
    text = lower_kernel_to_llvm(kernel, arch="gfx950")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
