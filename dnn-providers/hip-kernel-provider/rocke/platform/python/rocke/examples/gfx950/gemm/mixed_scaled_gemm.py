# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM with different A/B matrix formats and E8M0 scales on gfx950."""

from __future__ import annotations

from ....core.dtypes import normalize_dtype
from ....instances.gfx950.block_scaled_gemm import LOWBIT_DTYPES, BlockScaledGemmSpec
from ._scaled_gemm_example import argument_parser, verify

FORMATS = tuple(sorted(LOWBIT_DTYPES))


def make_spec(args) -> BlockScaledGemmSpec:
    if normalize_dtype(args.dtype_a) == normalize_dtype(args.dtype_b):
        raise ValueError(
            "choose different A/B formats or use a homogeneous mxfp example"
        )
    return BlockScaledGemmSpec(
        name="mixed_scaled_gemm",
        M=args.m,
        N=args.n,
        K=args.k,
        dtype_a=args.dtype_a,
        dtype_b=args.dtype_b,
        dtype_c=args.output_dtype,
        block_k=32,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argument_parser(__doc__)
    parser.add_argument(
        "--dtype-a", type=normalize_dtype, choices=FORMATS, default="fp8e4m3"
    )
    parser.add_argument(
        "--dtype-b", type=normalize_dtype, choices=FORMATS, default="fp4e2m1"
    )
    args = parser.parse_args(argv)
    try:
        spec = make_spec(args)
    except ValueError as exc:
        parser.error(str(exc))
    return verify(spec, args)


if __name__ == "__main__":
    raise SystemExit(main())
