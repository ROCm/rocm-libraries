# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from _emit_common import run_emit
from rocke.instances.gfx950.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
)

FORMATS = ("fp8e4m3", "bf8e5m2", "fp6e2m3", "fp6e3m2", "fp4e2m1")


def _spec(idx):
    if not 0 <= idx < 50:
        raise SystemExit(f"unknown config index {idx}")
    return BlockScaledGemmSpec(
        "scaled_parity",
        32 if idx < 25 else 64,
        64 if idx < 25 else 96,
        128 if idx < 25 else 256,
        FORMATS[(idx % 25) // 5],
        FORMATS[idx % 5],
        "bf16" if idx < 25 else "fp16",
    )


def main():
    return run_emit(_spec, build_block_scaled_gemm)


if __name__ == "__main__":
    raise SystemExit(main())
