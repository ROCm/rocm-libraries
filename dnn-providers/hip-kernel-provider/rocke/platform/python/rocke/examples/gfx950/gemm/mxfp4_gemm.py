# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Packed FP4 E2M1 x E2M1 GEMM with E8M0 block scales on gfx950."""

from __future__ import annotations

from ....instances.gfx950.block_scaled_gemm import BlockScaledGemmSpec
from ._scaled_gemm_example import argument_parser, verify


def make_spec(args) -> BlockScaledGemmSpec:
    return BlockScaledGemmSpec(
        name="mxfp4_gemm",
        M=args.m,
        N=args.n,
        K=args.k,
        dtype_a="fp4",
        dtype_b="fp4",
        dtype_c=args.output_dtype,
        block_k=32,
    )


def main(argv: list[str] | None = None) -> int:
    args = argument_parser(__doc__).parse_args(argv)
    return verify(make_spec(args), args)


if __name__ == "__main__":
    raise SystemExit(main())
