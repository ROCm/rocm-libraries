# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared arguments and execution for focused scaled GEMM examples."""

from __future__ import annotations

import argparse

from .block_scaled_gemm_verify import run_cases


def argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--k", type=int, choices=(128, 256), default=256)
    parser.add_argument(
        "--matrix-path",
        choices=("wmma_scale", "wmma_scale16"),
        default="wmma_scale",
        help="SCALE uses K=32 scale blocks; SCALE16 uses K=16 blocks",
    )
    parser.add_argument("--compile-route", choices=("comgr", "hip"), default="comgr")
    parser.add_argument(
        "--case",
        default="mixed",
        help="neutral, a-only, b-only, mixed, group-N, or all",
    )
    return parser


def verify(spec, args: argparse.Namespace) -> int:
    """Prepare bounded inputs, compile, launch, and compare with the reference.

    The shared verifier owns packing, independent decoding, HIP allocations,
    and result checks. Family examples select the public input contract.
    """
    cases = (
        ("neutral", "a-only", "b-only", "mixed")
        + tuple(f"group-{g}" for g in range(spec.K // spec.block_k))
        if args.case == "all"
        else (args.case,)
    )
    count = run_cases(spec, cases, compile_route=args.compile_route)
    print(f"PASS: verified {count} cases", flush=True)
    return 0
