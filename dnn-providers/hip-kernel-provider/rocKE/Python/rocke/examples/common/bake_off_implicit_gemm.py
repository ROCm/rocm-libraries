# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DSL example 08: implicit-GEMM convolution (bake-off 1).

A DSL-native implicit-GEMM convolution targeting the bake-off problem
(NHWC × KYXC -> NHWK, `N=8 H=W=56 C=K=64 Y=X=3 stride=pad=1`):
  - Implicit-GEMM with the convolution map encoded as a
    coordinate-transform DAG, fused so we never materialise the im2col
    tile.
  - Compute pipeline: 64x64x64 block tile, 2x2 warps,
    mfma_f32_32x32x16_f16 atom, single-buffer LDS for A and B,
    cshuffle epilogue.

Authoring style:
  - CK Tile-style transform DAG: every offset arithmetic step is
    expressed as `unmerge(...)`, `embed(...)`, `pad(...)`, or `merge(...)`
    over `rocke.helpers.transforms.TensorDescriptor`. The kernel author never
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

from rocke.core.arch import ArchTarget
from rocke.helpers import compile_kernel, make_conv_manifest, write_artifact
from rocke.instances.common.conv_implicit_gemm import (
    ConvDataSpec,
    ConvProblem,
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
)


def get_vector_sizes(C: int, K: int, dtype: str) -> tuple[int, int, int]:
    def _vec(n: int) -> int:
        sizes = [8, 4, 2, 1] if dtype != "fp32" else [4, 2, 1]
        return next(v for v in sizes if n % v == 0)

    vec_c = _vec(C)
    return vec_c, vec_c, _vec(K)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DSL example: implicit-GEMM convolution bake-off"
    )
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
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="data type for A/B/D tensors (default: fp16)",
    )

    # ConvProblem parameters — shared between 2-D and 3-D.
    # Pass --Di and --Z to activate the 3-D path; omit for 2-D.
    conv = parser.add_argument_group("ConvProblem", "convolution shape parameters")
    conv.add_argument("--N", type=int, default=8, help="batch size")
    conv.add_argument(
        "--Di", type=int, default=None, help="input depth  (3-D only; omit for 2-D)"
    )
    conv.add_argument("--Hi", type=int, default=56, help="input height")
    conv.add_argument("--Wi", type=int, default=56, help="input width")
    conv.add_argument("--C", type=int, default=64, help="input channels")
    conv.add_argument("--K", type=int, default=64, help="output channels / filters")
    conv.add_argument("--Z", type=int, default=None, help="filter depth  (3-D only)")
    conv.add_argument("--Y", type=int, default=3, help="filter height")
    conv.add_argument("--X", type=int, default=3, help="filter width")
    conv.add_argument("--sD", type=int, default=None, help="depth stride  (3-D only)")
    conv.add_argument("--sH", type=int, default=1, help="vertical stride")
    conv.add_argument("--sW", type=int, default=1, help="horizontal stride")
    conv.add_argument("--pD", type=int, default=None, help="depth padding  (3-D only)")
    conv.add_argument("--pH", type=int, default=1, help="vertical padding")
    conv.add_argument("--pW", type=int, default=1, help="horizontal padding")
    conv.add_argument("--dD", type=int, default=None, help="depth dilation  (3-D only)")
    conv.add_argument("--dH", type=int, default=1, help="vertical dilation")
    conv.add_argument("--dW", type=int, default=1, help="horizontal dilation")

    # ImplicitGemmConvSpec parameters
    spec_grp = parser.add_argument_group(
        "ImplicitGemmConvSpec", "kernel tile / pipeline configuration"
    )
    spec_grp.add_argument("--tile-m", type=int, default=64, help="block tile M")
    spec_grp.add_argument("--tile-n", type=int, default=64, help="block tile N")
    spec_grp.add_argument("--tile-k", type=int, default=64, help="block tile K")
    spec_grp.add_argument("--warp-m", type=int, default=2, help="warp grid M")
    spec_grp.add_argument("--warp-n", type=int, default=2, help="warp grid N")
    spec_grp.add_argument("--warp-tile-m", type=int, default=32, help="MFMA atom M")
    spec_grp.add_argument("--warp-tile-n", type=int, default=32, help="MFMA atom N")
    spec_grp.add_argument(
        "--warp-tile-k",
        type=int,
        default=None,
        help="MFMA atom K (default: largest legal K for the target atom)",
    )
    spec_grp.add_argument(
        "--pipeline",
        default="mem",
        choices=["mem", "compv3", "compv4"],
        help="pipeline variant",
    )
    spec_grp.add_argument(
        "--epilogue",
        default="cshuffle",
        choices=["default", "cshuffle"],
        help="epilogue variant",
    )

    args = parser.parse_args()

    if args.Di is not None and args.Z is None:
        print("--Z (filter depth) is required when --Di is set", file=sys.stderr)
        return 2
    if args.Z is not None and args.Di is None:
        print("--Di (input depth) is required when --Z is set", file=sys.stderr)
        return 2

    arch = args.arch
    if args.isa is not None:
        from rocke.core.arch import arch_from_isa

        arch = arch_from_isa(args.isa) or arch
    target = ArchTarget.from_gfx(arch)

    dtype = args.dtype
    # The winning 32x32 warp tile uses the largest legal K for the target:
    # gfx950 (CDNA4) carries the wide 32x32x16 atom, gfx942 (CDNA3) only the
    # 32x32x8 atom. Sourcing K from the catalog keeps gfx950 output unchanged
    # while degrading cleanly to the narrow atom on gfx942 (no comgr crash).
    dtype = args.dtype
    atom = target.mma.select_largest_k(
        a_dtype=dtype,
        b_dtype=dtype,
        c_dtype="fp32",
        m=args.warp_tile_m,
        n=args.warp_tile_n,
        k_max=args.tile_k,
    )
    if atom is None:
        print(
            f"no {dtype} {args.warp_tile_m}x{args.warp_tile_n} MFMA atom for {arch}",
            file=sys.stderr,
        )
        return 2
    warp_tile_k = args.warp_tile_k if args.warp_tile_k is not None else atom.k

    # Single ConvProblem for both 2-D and 3-D; Di/Z being None selects 2-D.
    problem = ConvProblem(
        N=args.N,
        Di=args.Di,
        Hi=args.Hi,
        Wi=args.Wi,
        C=args.C,
        K=args.K,
        Z=args.Z,
        Y=args.Y,
        X=args.X,
        sD=args.sD,
        sH=args.sH,
        sW=args.sW,
        pD=args.pD,
        pH=args.pH,
        pW=args.pW,
        dD=args.dD,
        dH=args.dH,
        dW=args.dW,
    )

    vector_size_a, vector_size_b, vector_size_c = get_vector_sizes(
        C=args.C, K=args.K, dtype=dtype
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
        name="rocke_ex08_bake_off_implicit_gemm",
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        warp_m=args.warp_m,
        warp_n=args.warp_n,
        warp_tile_m=args.warp_tile_m,
        warp_tile_n=args.warp_tile_n,
        warp_tile_k=warp_tile_k,
        pipeline=args.pipeline,
        epilogue=args.epilogue,
        vector_size_a=vector_size_a,
        vector_size_b=vector_size_b,
        vector_size_c=vector_size_c,
    )

    kernel = build_implicit_gemm_conv(spec)
    if args.isa is not None:
        artifact = compile_kernel(kernel, isa=args.isa)
    else:
        artifact = compile_kernel(kernel, arch=arch)

    p = problem
    # Build the manifest. For 2-D, `conv` holds 13 ints:
    #   [N, Hi, Wi, C, K, Y, X, sH, sW, pH, pW, dH, dW]
    # For 3-D, `conv` holds 18 ints and `conv_layout` selects the parser:
    #   [N, Di, Hi, Wi, C, K, Z, Y, X, sD, sH, sW, pD, pH, pW, dD, dH, dW]
    if p.is_3d:
        conv_field = [
            p.N,
            p.Di,
            p.Hi,
            p.Wi,
            p.C,
            p.K,
            p.Z,
            p.Y,
            p.X,
            p.sD,
            p.sH,
            p.sW,
            p.pD,
            p.pH,
            p.pW,
            p.dD,
            p.dH,
            p.dW,
        ]
        conv_layout = "implicit_gemm_3d"
        extra = {
            "dtype": dtype,
            "conv_3d": [
                p.N,
                p.Di,
                p.Hi,
                p.Wi,
                p.C,
                p.K,
                p.Z,
                p.Y,
                p.X,
                p.sD,
                p.sH,
                p.sW,
                p.pD,
                p.pH,
                p.pW,
                p.dD,
                p.dH,
                p.dW,
            ],
            "default_shape": [p.M, p.N_gemm, p.K_gemm],
        }
    else:
        conv_field = [
            p.N,
            p.Hi,
            p.Wi,
            p.C,
            p.K,
            p.Y,
            p.X,
            p.sH,
            p.sW,
            p.pH,
            p.pW,
            p.dH,
            p.dW,
        ]
        conv_layout = "implicit_gemm"
        extra = {
            "dtype": dtype,
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
                        "upper": ["ho", "y"],
                        "into": "hi",
                        "strides": [p.sH, p.dH],
                        "offset": -p.pH,
                        "lo": 0,
                        "hi": p.Hi,
                    },
                    {
                        "transform": "embed",
                        "upper": ["wo", "x"],
                        "into": "wi",
                        "strides": [p.sW, p.dW],
                        "offset": -p.pW,
                        "lo": 0,
                        "hi": p.Wi,
                    },
                    {
                        "transform": "unmerge",
                        "upper": "k",
                        "into": ["y", "x", "c"],
                        "dims": [p.Y, p.X, p.C],
                    },
                ],
                "B_kyxc": [
                    {
                        "transform": "unmerge",
                        "upper": "k_gemm",
                        "into": ["y", "x", "c"],
                        "dims": [p.Y, p.X, p.C],
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
        }

    manifest = make_conv_manifest(
        artifact=artifact,
        block_m=spec.tile_m,
        block_n=spec.tile_n,
        block_k=spec.tile_k,
        threads_per_block=spec.block_size,
        conv=conv_field,
        groups=1,
        cpg=p.C,
        kpg=p.K,
        conv_layout=conv_layout,
        dtype=dtype,
        # The kernel reads block_id.x as the N-tile index and
        # block_id.y as the M-tile index (mirrors gemm_universal).
        grid_order="NM",
        warmup_iters=5,
        timed_iters=100,
        extra=extra,
        atoms=[f"tile.mfma_f32_32x32x{warp_tile_k}_{dtype}"],
        notes=(
            "Bake-off 1: implicit-GEMM conv via the coord-transform "
            "DAG (rocke.helpers.transforms.TensorDescriptor). A's address is "
            "computed as (m, k) -> unmerge -> (n, ho, wo, r, s, c) -> "
            "embed -> (n, hi, wi, c) -> naive NHWC offset, with the "
            "conv boundary check baked into the descriptor's validity "
            "predicate. Same algorithmic strategy as a hand-written "
            "implicit-GEMM kernel, but expressed through CK Tile's "
            "coordinate-transform algebra instead of inline arithmetic."
        ),
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
