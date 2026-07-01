# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emit and verify the gfx1201 (RDNA4, wave32, WMMA) deep-fused conv + maxpool.

Same fused dataflow as the gfx950 prototype (virtual-concat -> conv0 3x3 -> ReLU
-> 1x1 conv1 -> ReLU -> 2x2 stride-2 maxpool -> store, no conv0/conv1 HBM
intermediates), but driven by the gfx12 WMMA 16x16x16 atom on wave32. The kernel
body is the arch-parametric one in ``instances/common``; this driver only pins
the WMMA geometry and reuses the gfx950 harness's numpy reference + verify/bench
helpers (they are spec-generic). Must run on a gfx1201 device.

  PYTHONPATH=Python python3 -m rocke.examples.gfx1201.deep_fused_conv_pool_verify \
    --arch gfx1201 --verify --h 16 --w 16 --c 8 --k0 32 --k1 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rocke.core.arch import ArchTarget
from rocke.helpers import compile_kernel, make_conv_manifest, write_artifact
from rocke.instances import ConvAccumulatorEpilogue, ConvProblem
from rocke.instances.gfx1201.deep_fused_conv_pool import (
    Gfx1201DeepFusedConvPoolSpec,
    FusedConvPoolProblem,
    build_deep_fused_conv_pool,
    deep_fused_conv_pool_grid,
    deep_fused_conv_pool_signature,
    is_valid_spec,
)

# The reference + launch helpers are arch-neutral (they only read ``spec.problem``
# / ``spec.block_size`` and the common grid), so reuse them verbatim.
from rocke.examples.gfx950.deep_conv_fusion.deep_fused_conv_pool_verify import (
    _verify_artifact,
    _benchmark_artifact,
)

_WMMA_ATOM = "wmma_gfx12_f32_16x16x16_f16"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "output" / "deep_fused_conv_pool"),
    )
    parser.add_argument("--arch", default="gfx1201")
    parser.add_argument("--isa", default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--tile-k", type=int, default=16)
    parser.add_argument(
        "--conv1-tile-k",
        type=int,
        default=0,
        help="K tile for the conv1 1x1 stage; 0 reuses --tile-k",
    )
    parser.add_argument("--tile-n", type=int, default=32)
    parser.add_argument("--pool-tile-h", type=int, default=4)
    parser.add_argument("--pool-tile-w", type=int, default=4)
    parser.add_argument("--warp-m", type=int, default=2)
    parser.add_argument("--warp-n", type=int, default=1)
    parser.add_argument(
        "--pipeline", default="mem", choices=["mem", "compv3", "compv4"]
    )
    parser.add_argument("--unroll-k", action="store_true")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--w", type=int, default=16)
    parser.add_argument("--c", type=int, default=8)
    parser.add_argument("--k0", type=int, default=32)
    parser.add_argument("--k1", type=int, default=24)
    args = parser.parse_args()

    arch = args.arch
    if args.isa is not None:
        from rocke.core.arch import arch_from_isa

        arch = arch_from_isa(args.isa) or arch
    if arch != "gfx1201":
        print("deep_fused_conv_pool WMMA path is gfx1201-only", file=sys.stderr)
        return 2

    target = ArchTarget.from_gfx(arch)
    atom = target.mma.by_op_id(_WMMA_ATOM)
    if atom is None:
        print(f"no {_WMMA_ATOM} on {arch}", file=sys.stderr)
        return 2

    conv = ConvProblem(
        N=args.n,
        Hi=args.h,
        Wi=args.w,
        C=args.c,
        K=args.k0,
        Y=3,
        X=3,
        sH=1,
        sW=1,
        pH=1,
        pW=1,
        dH=1,
        dW=1,
    )
    problem = FusedConvPoolProblem(conv=conv, conv1_k=args.k1)
    conv_tile_h = args.pool_tile_h * problem.pool_stride_h
    conv_tile_w = args.pool_tile_w * problem.pool_stride_w
    spec = Gfx1201DeepFusedConvPoolSpec(
        problem=problem,
        tile_m=conv_tile_h * conv_tile_w,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        conv1_tile_k=args.conv1_tile_k,
        pool_tile_h=args.pool_tile_h,
        pool_tile_w=args.pool_tile_w,
        warp_m=args.warp_m,
        warp_n=args.warp_n,
        warp_tile_m=atom.m,
        warp_tile_n=atom.n,
        warp_tile_k=atom.k,
        wave_size=target.wave_size,
        pipeline=args.pipeline,
        unroll_k=args.unroll_k,
        acc_epilogue=ConvAccumulatorEpilogue(relu=True),
    )
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        print(f"invalid spec: {why}", file=sys.stderr)
        return 2

    kernel = build_deep_fused_conv_pool(spec, arch=arch)
    artifact = (
        compile_kernel(kernel, isa=args.isa)
        if args.isa
        else compile_kernel(kernel, arch=arch)
    )

    grid = deep_fused_conv_pool_grid(spec)
    manifest = make_conv_manifest(
        artifact=artifact,
        block_m=spec.tile_m,
        block_n=spec.tile_n,
        block_k=spec.tile_k,
        threads_per_block=spec.block_size,
        conv=[
            conv.N,
            conv.Hi,
            conv.Wi,
            conv.C,
            conv.K,
            conv.Y,
            conv.X,
            conv.sH,
            conv.sW,
            conv.pH,
            conv.pW,
            conv.dH,
            conv.dW,
        ],
        groups=1,
        cpg=conv.C,
        kpg=conv.K,
        grid_explicit=grid,
        conv_layout="deep_fused_conv_pool",
        atoms=[f"tile.{_WMMA_ATOM}"],
        notes=(
            "Experimental gfx1201 (RDNA4 WMMA) deep-fusion prototype: implicit-GEMM "
            "conv0 + ReLU staged through LDS, 1x1 conv1 + ReLU, inline 2x2 stride-2 "
            "maxpool to global. No conv0/conv1 intermediate written to HBM."
        ),
        extra={
            "kind": "deep_fused_conv_pool_fp16",
            "args_signature": deep_fused_conv_pool_signature(spec),
            "pool": [
                problem.pool_y,
                problem.pool_x,
                problem.pool_stride_h,
                problem.pool_stride_w,
            ],
            "conv1": {
                "kernel": "1x1",
                "K1": problem.conv1_channels,
                "tile_k": spec.effective_conv1_tile_k,
            },
            "pool_output_shape": [
                conv.N,
                problem.pool_ho,
                problem.pool_wo,
                problem.conv1_channels,
            ],
            "default_shape": [
                conv.N,
                conv.Hi,
                conv.Wi,
                conv.C,
                conv.K,
                problem.conv1_channels,
            ],
            "experimental": True,
        },
    )

    paths = write_artifact(artifact, Path(args.output_dir), manifest)
    t = artifact.timings
    print(
        f"emitted {paths['hsaco']} ({artifact.hsaco_bytes} bytes) in "
        f"{t['total']:.2f} ms total"
    )
    print(f"  grid={grid} block=({spec.block_size}, 1, 1)")
    print(
        f"  pool_tile={spec.pool_tile_h}x{spec.pool_tile_w} "
        f"tile={spec.tile_m}x{spec.tile_n}x{spec.tile_k} "
        f"conv1_tile_k={spec.effective_conv1_tile_k} "
        f"warp={spec.warp_m}x{spec.warp_n} "
        f"atom={spec.warp_tile_m}x{spec.warp_tile_n}x{spec.warp_tile_k} "
        f"pipeline={spec.pipeline} unroll_k={spec.unroll_k}"
    )
    print(f"  conv0_input={[conv.N, conv.Hi, conv.Wi, conv.C]} conv0_K={conv.K}")
    print(
        f"  conv1_1x1={[conv.N, conv.Ho, conv.Wo, conv.K]} -> K1={problem.conv1_channels}"
    )
    print(
        f"  pool_output={[conv.N, problem.pool_ho, problem.pool_wo, problem.conv1_channels]}"
    )
    if args.verify:
        if not _verify_artifact(artifact, spec, seed=args.seed, tol=args.tol):
            return 1
    if args.bench:
        _benchmark_artifact(
            artifact, spec, seed=args.seed, warmup=args.warmup, iters=args.iters
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
