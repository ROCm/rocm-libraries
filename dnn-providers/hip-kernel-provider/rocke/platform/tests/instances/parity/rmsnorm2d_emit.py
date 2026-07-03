#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/rmsnorm2d_emit.py -- Python reference emitter for the rmsnorm2d
# parity harness. Selects one of the sampled RMSNorm2DSpec configs by argv[1],
# builds the kernel via build_rmsnorm2d and prints
# lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared with
# the C emitter rmsnorm2d_emit.c.
from rocke.instances.common.rmsnorm2d import (
    RMSNorm2DSpec,
    build_rmsnorm2d,
)
from _emit_common import run_emit


def _spec(idx: int) -> RMSNorm2DSpec:
    if idx == 0:
        return RMSNorm2DSpec(
            n_per_block=1024, block_size=256, vec=4, dtype="f16", save_inv_rms=False
        )
    if idx == 1:
        return RMSNorm2DSpec(
            n_per_block=2048, block_size=256, vec=4, dtype="bf16", save_inv_rms=False
        )
    if idx == 2:
        return RMSNorm2DSpec(
            n_per_block=4096, block_size=256, vec=4, dtype="f16", save_inv_rms=False
        )
    if idx == 3:
        return RMSNorm2DSpec(
            n_per_block=8192, block_size=256, vec=8, dtype="bf16", save_inv_rms=False
        )
    if idx == 4:
        return RMSNorm2DSpec(
            n_per_block=16384, block_size=256, vec=8, dtype="f16", save_inv_rms=True
        )
    if idx == 5:
        return RMSNorm2DSpec(
            n_per_block=131072, block_size=256, vec=4, dtype="f16", save_inv_rms=False
        )
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch=None):
    return build_rmsnorm2d(spec)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: rmsnorm2d_emit.py <config_index 0..5> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
