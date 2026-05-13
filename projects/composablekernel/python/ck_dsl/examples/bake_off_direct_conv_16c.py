# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DSL example 09: direct grouped convolution 16c (bake-off 2).

A streaming direct convolution for `C=K=16` per group, processed
row-by-row with an MFMA 16x16x16 fp16 pipeline and a 3-acc circular
output accumulator. Targets the bake-off problem shape:

  N=32, H=W=200, groups in {16, 32}, cpg=kpg=16, 3x3, pad=1

Algorithm:
  - 8 waves per workgroup, one wave per group, BLOCK_Q=16
  - Double-buffered LDS for the input row, ping-pong on y%2
  - 3-accumulator circular pipeline along H
  - Per output row: 9 MFMAs (R*S)

Reference numbers from the bake-off study
`convolution_direct_vs_implicit_gemm_study.md`:
  CK Tile streaming direct-conv (`v2_16c_*`):
    groups=8:  ~155 TFLOPS
    groups=16: ~250 TFLOPS
    groups=32: ~254 TFLOPS

This DSL emission uses K32 folding plus a wide vector direct epilogue
to land in the high-performance band for this shape on gfx950.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ck_dsl.helpers import compile_kernel, make_conv_manifest, write_artifact
from ck_dsl.instances.conv_direct_grouped import (
    DirectConv16cSpec,
    DirectConvProblem,
    build_direct_conv_16c,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--isa", default="amdgcn-amd-amdhsa--gfx950")
    parser.add_argument(
        "--groups",
        type=int,
        default=16,
        help="number of groups (16 or 32 in the bake-off)",
    )
    args = parser.parse_args()

    problem = DirectConvProblem(
        N=32,
        H=200,
        W=200,
        groups=args.groups,
        cpg=16,
        kpg=16,
        KH=3,
        KW=3,
        PAD=1,
        stride=1,
    )
    # Empirical policy:
    #   groups=16 -> block_groups=4 + K32 fold + wide vec epilogue: ~210 TFLOPS.
    #   groups=32 -> block_groups=8 is better for per-launch throughput (~103 TFLOPS).
    block_groups = 4 if args.groups <= 16 else 8
    use_k32 = args.groups <= 16
    spec = DirectConv16cSpec(
        problem=problem,
        name="ck_dsl_ex09_bake_off_direct_conv_16c",
        block_q=16,
        block_groups=block_groups,
        double_buffer=False,
        fold_k32=use_k32,
    )

    kernel = build_direct_conv_16c(spec)
    artifact = compile_kernel(kernel, isa=args.isa)

    p = problem
    q_tiles = (p.W + spec.block_q - 1) // spec.block_q
    g_tiles = p.groups // spec.block_groups
    manifest = make_conv_manifest(
        artifact=artifact,
        block_m=spec.block_q,
        block_n=spec.block_groups,
        block_k=p.cpg,
        threads_per_block=spec.threads_per_block,
        conv=[
            p.N,
            p.H,
            p.W,
            p.total_c,
            p.total_k,
            p.KH,
            p.KW,
            p.stride,
            p.stride,
            p.PAD,
            p.PAD,
            1,
            1,
        ],
        groups=p.groups,
        cpg=p.cpg,
        kpg=p.kpg,
        conv_layout="direct_grouped",
        grid_explicit=[q_tiles, g_tiles, p.N],
        warmup_iters=5,
        timed_iters=50,
        atoms=["tile.mfma_f32_16x16x16_f16"]
        + (["tile.mfma_f32_16x16x32_f16"] if use_k32 else []),
        notes=(
            "Direct grouped convolution 16c — bake-off 2. "
            "K32 folded main loop, BLOCK_Q=16, 3-acc circular pipeline, "
            "wide vector direct epilogue. Python-native runtime verifies "
            "against a grouped NumPy fp32 reference."
        ),
        extra={
            "default_shape": [p.N, p.H * p.W * p.N, p.total_k],
        },
    )
    paths = write_artifact(artifact, Path(args.output_dir), manifest)
    print(
        f"emitted {paths['hsaco']} ({artifact.hsaco_bytes} bytes) in "
        f"{artifact.timings['total']:.2f} ms total"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
