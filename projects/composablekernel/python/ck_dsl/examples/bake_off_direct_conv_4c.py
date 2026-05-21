# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DSL example 10: direct grouped convolution 4c (bake-off 2)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ck_dsl.helpers import compile_kernel, make_conv_manifest, write_artifact
from ck_dsl.instances.conv_direct_grouped import (
    DirectConv4cSpec,
    DirectConvProblem,
    build_direct_conv_4c,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--isa", default="amdgcn-amd-amdhsa--gfx950")
    parser.add_argument("--groups", type=int, default=64)
    args = parser.parse_args()

    problem = DirectConvProblem(
        N=32,
        H=200,
        W=200,
        groups=args.groups,
        cpg=4,
        kpg=4,
        KH=3,
        KW=3,
        PAD=1,
        stride=1,
    )
    spec = DirectConv4cSpec(
        problem=problem,
        name="ck_dsl_ex10_bake_off_direct_conv_4c",
        block_q=8,
        block_groups=16,
    )

    kernel = build_direct_conv_4c(spec)
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
        conv_layout="direct_grouped_4c",
        grid_explicit=[q_tiles, g_tiles, p.N],
        warmup_iters=10,
        timed_iters=100,
        atoms=["tile.mfma_f32_4x4x4_f16"],
        notes=(
            "Direct grouped convolution 4c — bake-off 2. "
            "MFMA 4x4x4 uses 16 independent groups per wave; "
            "block_q=8 processes two 4-wide Q tiles per wave. "
            "Epilogue is one vec2-dword (4 halves) buffer_store per "
            "lane, matching the lane->output mapping for 4x4x4."
        ),
    )
    paths = write_artifact(artifact, Path(args.output_dir), manifest)
    print(
        f"emitted {paths['hsaco']} ({artifact.hsaco_bytes} bytes) in "
        f"{artifact.timings['total']:.2f} ms total"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
