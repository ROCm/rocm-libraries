# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Homogeneous packed FP6 E2M3 or E3M2 GEMM with packed E8M0 block scales on gfx1250."""

from __future__ import annotations

from ....core.arch.target import normalize_dtype
from ....instances.gfx1250.block_scaled_gemm import BlockScaledGemmSpec
from ._scaled_gemm_example import argument_parser, verify


def make_spec(args) -> BlockScaledGemmSpec:
    return BlockScaledGemmSpec(
        name="mxfp6_gemm",
        M=args.m,
        N=args.n,
        K=args.k,
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c="bf16",
        scale_dtype="e8m0",
        matrix_path=args.matrix_path,
        block_k=16 if args.matrix_path == "wmma_scale16" else 32,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argument_parser(__doc__)
    parser.add_argument(
        "--dtype",
        type=normalize_dtype,
        choices=("fp6e2m3", "fp6e3m2", "both"),
        default="both",
    )
    args = parser.parse_args(argv)
    dtypes = ("fp6e2m3", "fp6e3m2") if args.dtype == "both" else (args.dtype,)
    for dtype in dtypes:
        args.dtype = dtype
        verify(make_spec(args), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
