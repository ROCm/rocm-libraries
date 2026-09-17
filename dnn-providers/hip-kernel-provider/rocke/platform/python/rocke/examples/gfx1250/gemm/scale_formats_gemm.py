# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM with explicit A/B block-scale formats on gfx1250."""

from __future__ import annotations

from ....instances.gfx1250.block_scaled_gemm import BlockScaledGemmSpec
from ._scaled_gemm_example import argument_parser, verify

FORMATS = ("fp8e4m3", "bf8e5m2", "fp6", "bf6", "fp4")


def make_spec(args) -> BlockScaledGemmSpec:
    return BlockScaledGemmSpec(
        name="scale_formats_gemm",
        M=args.m,
        N=args.n,
        K=args.k,
        dtype_a=args.dtype_a,
        dtype_b=args.dtype_b,
        dtype_c="bf16",
        scale_dtype="e8m0",
        scale_dtype_a=args.scale_dtype_a,
        scale_dtype_b=args.scale_dtype_b,
        matrix_path=args.matrix_path,
        block_k=16 if args.matrix_path == "wmma_scale16" else 32,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argument_parser(__doc__)
    parser.add_argument("--dtype-a", choices=FORMATS, default="fp4")
    parser.add_argument("--dtype-b", choices=FORMATS, default="fp4")
    parser.add_argument(
        "--scale-dtype-a", choices=("e8m0", "e4m3", "e5m3"), default="e4m3"
    )
    parser.add_argument(
        "--scale-dtype-b", choices=("e8m0", "e4m3", "e5m3"), default="e4m3"
    )
    args = parser.parse_args(argv)
    try:
        spec = make_spec(args)
    except ValueError as exc:
        parser.error(str(exc))
    return verify(spec, args)


if __name__ == "__main__":
    raise SystemExit(main())
