# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Packed FP4 E2M1 x E2M1 GEMM with E8M0 block scales on gfx1250."""

from __future__ import annotations

from ....instances.gfx1250.block_scaled_gemm import BlockScaledGemmSpec
from ._scaled_gemm_example import argument_parser, verify


def make_spec(args) -> BlockScaledGemmSpec:
    return BlockScaledGemmSpec(
        name="mxfp4_gemm",
        M=args.m,
        N=args.n,
        K=args.k,
        dtype_a="fp4",
        dtype_b="fp4",
        dtype_c="bf16",
        scale_dtype="e8m0",
        matrix_path=args.matrix_path,
        block_k=16 if args.matrix_path == "wmma_scale16" else 32,
    )


def main(argv: list[str] | None = None) -> int:
    args = argument_parser(__doc__).parse_args(argv)
    return verify(make_spec(args), args)


if __name__ == "__main__":
    raise SystemExit(main())
