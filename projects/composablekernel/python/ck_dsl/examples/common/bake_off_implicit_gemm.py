# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DSL example 08: implicit-GEMM convolution (bake-off 1).

A DSL-native implicit-GEMM convolution targeting the bake-off problem
(NHWC × KRSC -> NHWK, `N=8 H=W=56 C=K=64 R=S=3 stride=pad=1`):
  - Implicit-GEMM with the convolution map encoded as a
    coordinate-transform DAG, fused so we never materialise the im2col
    tile.
  - Compute pipeline: 64x64x64 block tile, 2x2 warps,
    mfma_f32_32x32x16_f16 atom, single-buffer LDS for A and B,
    cshuffle epilogue.

Authoring style:
  - CK Tile-style transform DAG: every offset arithmetic step is
    expressed as `unmerge(...)`, `embed(...)`, `pad(...)`, or `merge(...)`
    over `ck_dsl.helpers.transforms.TensorDescriptor`. The kernel author never
    writes `gm // (Ho * Wo)` by hand; the same SSA gets generated, but
    via the algebra.
  - The kernel emits straight to AMDGPU LLVM IR + HSACO in-process
    (no hipcc / MLIR / clang). Codegen time for one kernel is
    typically <150 ms wall.

Reference: CK Tile's `cktile_fixed_lean` for this shape reaches
~250 TFLOPS in CUDA-graph mode; we aim to beat that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ck_dsl.core.arch import ArchTarget
from ck_dsl.helpers import compile_kernel, make_conv_manifest, write_artifact
from ck_dsl.instances.common.conv_implicit_gemm import (
    ConvProblem,
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "output" / "bake_off_implicit_gemm"),
    )
    parser.add_argument(
        "--arch", default="gfx950", help="gfx target (gfx942, gfx950, ...)"
    )
    parser.add_argument(
        "--isa",
        default=None,
        help="raw comgr triple; overrides --arch when set",
    )
    args = parser.parse_args()

    arch = args.arch
    if args.isa is not None:
        from ck_dsl.core.arch import arch_from_isa

        arch = arch_from_isa(args.isa) or arch
    target = ArchTarget.from_gfx(arch)
    # The winning 32x32 warp tile uses the largest legal K for the target:
    # gfx950 (CDNA4) carries the wide 32x32x16 atom, gfx942 (CDNA3) only the
    # 32x32x8 atom. Sourcing K from the catalog keeps gfx950 output unchanged
    # while degrading cleanly to the narrow atom on gfx942 (no comgr crash).
    atom = target.mma.select_largest_k(
        a_dtype="fp16", b_dtype="fp16", c_dtype="fp32", m=32, n=32
    )
    if atom is None:
        print(f"no fp16 32x32 MFMA atom for {arch}", file=sys.stderr)
        return 2
    warp_tile_k = atom.k

    problem = ConvProblem(
        N=8,
        Hi=56,
        Wi=56,
        C=64,
        K=64,
        R=3,
        S=3,
        sH=1,
        sW=1,
        pH=1,
        pW=1,
        dH=1,
        dW=1,
    )
    # Winning config from the sweep in `instances.conv_implicit_gemm`:
    #   tile (64, 64, 64)
    #   warp grid (2, 2)
    #   MFMA 32x32x16 (gfx950)
    #   pipeline=mem (single-buffer LDS — for K_gemm=576 the compv4
    #                double-buffer doesn't beat the LDS bandwidth)
    #   epilogue=cshuffle  (wide-vector global stores, runbook §9.3)
    # On MI300X for this conv shape:
    #   per-launch:   248 TFLOPS
    #   graph 5x200:  280 TFLOPS  (vs ~250 TFLOPS for CK Tile's own
    #                              `cktile_fixed_lean` on the same shape)
    spec = ImplicitGemmConvSpec(
        problem=problem,
        name="ck_dsl_ex08_bake_off_implicit_gemm",
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=warp_tile_k,
        pipeline="mem",
        epilogue="cshuffle",
    )

    kernel = build_implicit_gemm_conv(spec)
    if args.isa is not None:
        artifact = compile_kernel(kernel, isa=args.isa)
    else:
        artifact = compile_kernel(kernel, arch=arch)

    p = problem
    manifest = make_conv_manifest(
        artifact=artifact,
        block_m=spec.tile_m,
        block_n=spec.tile_n,
        block_k=spec.tile_k,
        threads_per_block=spec.block_size,
        conv=[p.N, p.Hi, p.Wi, p.C, p.K, p.R, p.S, p.sH, p.sW, p.pH, p.pW, p.dH, p.dW],
        groups=1,
        cpg=p.C,
        kpg=p.K,
        conv_layout="implicit_gemm",
        # The kernel reads block_id.x as the N-tile index and
        # block_id.y as the M-tile index (mirrors gemm_universal).
        # The runner computes (M_tiles, N_tiles) following the
        # "M-first" convention; we set grid_order="NM" so it swaps
        # (gx, gy) -> (N_tiles, M_tiles) before launch.
        grid_order="NM",
        warmup_iters=5,
        timed_iters=100,
        atoms=[f"tile.mfma_f32_32x32x{warp_tile_k}_f16"],
        notes=(
            "Bake-off 1: implicit-GEMM conv via the coord-transform "
            "DAG (ck_dsl.helpers.transforms.TensorDescriptor). A's address is "
            "computed as (m, k) -> unmerge -> (n, ho, wo, r, s, c) -> "
            "embed -> (n, hi, wi, c) -> naive NHWC offset, with the "
            "conv boundary check baked into the descriptor's validity "
            "predicate. Same algorithmic strategy as a hand-written "
            "implicit-GEMM kernel, but expressed through CK Tile's "
            "coordinate-transform algebra instead of inline arithmetic."
        ),
        extra={
            "default_shape": [p.M, p.N_gemm, p.K_gemm],
            "transform_dag": {
                "A_nhwc": [
                    {
                        "transform": "unmerge",
                        "upper": "m",
                        "into": ["n", "ho", "wo"],
                        "dims": [p.N, p.Ho, p.Wo],
                    },
                    {
                        "transform": "embed",
                        "upper": ["ho", "r"],
                        "into": "hi",
                        "strides": [p.sH, p.dH],
                        "offset": -p.pH,
                        "lo": 0,
                        "hi": p.Hi,
                    },
                    {
                        "transform": "embed",
                        "upper": ["wo", "s"],
                        "into": "wi",
                        "strides": [p.sW, p.dW],
                        "offset": -p.pW,
                        "lo": 0,
                        "hi": p.Wi,
                    },
                    {
                        "transform": "unmerge",
                        "upper": "k",
                        "into": ["r", "s", "c"],
                        "dims": [p.R, p.S, p.C],
                    },
                ],
                "B_krsc": [
                    {
                        "transform": "unmerge",
                        "upper": "k_gemm",
                        "into": ["r", "s", "c"],
                        "dims": [p.R, p.S, p.C],
                    },
                ],
                "D_nhwk": [
                    {
                        "transform": "unmerge",
                        "upper": "m",
                        "into": ["n", "ho", "wo"],
                        "dims": [p.N, p.Ho, p.Wo],
                    },
                ],
            },
        },
    )

    paths = write_artifact(artifact, Path(args.output_dir), manifest)
    t = artifact.timings
    print(
        f"emitted {paths['hsaco']} ({artifact.hsaco_bytes} bytes) in "
        f"{t['total']:.2f} ms total"
    )
    print(
        f"  ir_build={t['ir_build']:.1f}ms "
        f"lower={t['ir_lower_llvm']:.1f}ms "
        f"comgr_bc={t['comgr_bc']:.1f}ms "
        f"reloc={t['comgr_relocatable']:.1f}ms "
        f"exe={t['comgr_executable']:.1f}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
