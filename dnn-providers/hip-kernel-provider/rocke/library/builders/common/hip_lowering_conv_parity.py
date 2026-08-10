# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Audit HIP debug lowering parity for convolution kernels (library layer).

The convolution companion to ``rocke.examples.common.hip_lowering_parity``
(platform layer), and the direct analogue of
``hip_lowering_attention_parity`` for the SDPA/MHA vertical. It provides
``_conv_cases(arch)`` for the implicit-GEMM conv, grouped direct-conv and
img2col kernels imported from the ``kernels`` package.

Platform code must NOT import from ``kernels``; this module isolates those
imports in the library layer. It imports ``Case``, ``_selected`` and
``audit_cases`` from the platform module (library -> platform is legal per
the one-way layering rule).

Usage::

    python -m builders.common.hip_lowering_conv_parity
    python -m builders.common.hip_lowering_conv_parity --compile-hip
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from kernels import (  # noqa: E402
    ConvProblem,
    DirectConv4cSpec,
    DirectConv16cSpec,
    DirectConvProblem,
    Img2ColSpec,
    ImplicitGemmConvSpec,
    build_direct_conv_4c,
    build_direct_conv_16c,
    build_img2col,
    build_implicit_gemm_conv,
)

# Import shared audit infrastructure from platform (library->platform is legal).
from rocke.examples.common.hip_lowering_parity import (  # noqa: E402
    Case,
    _selected,
    audit_cases,
)


def _conv_problem() -> ConvProblem:
    return ConvProblem(
        N=1,
        Hi=8,
        Wi=8,
        C=16,
        K=16,
        Y=3,
        X=3,
        sH=1,
        sW=1,
        pH=1,
        pW=1,
        dH=1,
        dW=1,
    )


def _conv_cases(arch: str = "gfx950") -> List[Case]:
    """Return the convolution ``Case`` instances for the given arch.

    These are the cases that lived in the platform harness's ``make_cases``
    before the convolution vertical was carved into the library. Specs are
    reproduced verbatim so the audit output is unchanged.
    """
    del arch  # conv cases are arch-independent; kept for signature parity
    convp = _conv_problem()

    cases: List[Case] = [
        Case(
            "img2col",
            "small",
            lambda: build_img2col(
                Img2ColSpec(problem=convp, block_tile_m=8, block_tile_k=64)
            ),
        ),
        Case(
            "implicit_gemm_conv",
            "conv",
            lambda: build_implicit_gemm_conv(
                ImplicitGemmConvSpec(
                    problem=convp,
                    name="hip_audit_conv",
                    tile_m=64,
                    tile_n=64,
                    tile_k=64,
                    warp_m=2,
                    warp_n=2,
                    warp_tile_m=32,
                    warp_tile_n=32,
                    warp_tile_k=16,
                    pipeline="mem",
                    epilogue="cshuffle",
                )
            ),
        ),
        Case(
            "direct_conv_16c",
            "conv",
            lambda: build_direct_conv_16c(
                DirectConv16cSpec(
                    DirectConvProblem(N=1, H=8, W=8, groups=8, cpg=16, kpg=16)
                )
            ),
        ),
        Case(
            "direct_conv_4c",
            "conv",
            lambda: build_direct_conv_4c(
                DirectConv4cSpec(
                    DirectConvProblem(N=1, H=8, W=8, groups=16, cpg=4, kpg=4)
                )
            ),
        ),
    ]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", help="substring or group filter")
    parser.add_argument(
        "--compile-hip", action="store_true", help="compile HIP source to HSACO"
    )
    parser.add_argument("--compile-timeout-s", type=int, default=180)
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--emit-dir", type=Path, default=None)
    args = parser.parse_args()

    cases = _selected(_conv_cases(arch=args.arch), args.case)
    if not cases:
        print(f"no cases matched {args.case!r}")
        return 2

    results = audit_cases(
        cases,
        compile_hip=args.compile_hip,
        arch=args.arch,
        emit_dir=args.emit_dir,
        compile_timeout_s=args.compile_timeout_s,
    )
    for r in results:
        compile_status = ""
        if args.compile_hip:
            compile_status = f" hipcc={'OK' if r.hip_compile_ok else 'FAIL'}"
        status = "OK" if r.ok else "FAIL"
        print(
            f"{status:4} {r.group:9} {r.name:28} "
            f"llvm={'OK' if r.llvm_ok else 'FAIL'} hip={'OK' if r.hip_ok else 'FAIL'} "
            f"chars={r.hip_chars}{compile_status}"
        )
        if r.error:
            print(f"     {r.error}")

    failures = [r for r in results if not r.ok]
    print(
        f"SUMMARY total={len(results)} ok={len(results) - len(failures)} fail={len(failures)}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
