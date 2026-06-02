# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emit and verify the gfx950 deep-fused conv + maxpool prototype.

The v1 kernel maps CTAs over rectangular pooled-output tiles, computes the
required conv0 and 1x1 conv1 tile locally, then inline-pools from LDS into the
final output. It is still a prototype schedule, but it exercises multi-CTA 2D
spatial tiling.
"""

from __future__ import annotations

import argparse
import ctypes
import struct
import sys
from pathlib import Path

from ck_dsl.core.arch import ArchTarget
from ck_dsl.helpers import compile_kernel, make_conv_manifest, write_artifact
from ck_dsl.instances import ConvAccumulatorEpilogue, ConvProblem
from ck_dsl.instances.gfx950.deep_fused_conv_pool import (
    FusedConvPoolProblem,
    Gfx950DeepFusedConvPoolSpec,
    build_deep_fused_conv_pool,
    deep_fused_conv_pool_grid,
    deep_fused_conv_pool_signature,
    is_valid_spec,
)


def _as_u8_buffer(array):
    return (ctypes.c_uint8 * int(array.nbytes)).from_buffer(array)


def _reference_conv1x1_relu_pool(
    A, B0, W1, conv: ConvProblem, problem: FusedConvPoolProblem
):
    import numpy as np

    Ap = np.pad(A, ((0, 0), (conv.pH, conv.pH), (conv.pW, conv.pW), (0, 0)))
    C0 = np.zeros((conv.N, conv.Ho, conv.Wo, conv.K), dtype=np.float32)
    for r in range(conv.R):
        for s in range(conv.S):
            row_start = r * conv.dH
            col_start = s * conv.dW
            x = Ap[
                :,
                row_start : row_start + conv.Ho * conv.sH : conv.sH,
                col_start : col_start + conv.Wo * conv.sW : conv.sW,
                :,
            ].astype(np.float32)
            w = B0[:, r, s, :].astype(np.float32)
            C0 += np.einsum("nhwc,kc->nhwk", x, w, optimize=True)

    C0 = np.maximum(C0, 0.0).astype(np.float16).astype(np.float32)
    C1 = np.einsum("nhwk,ok->nhwo", C0, W1.astype(np.float32), optimize=True)
    C1 = np.maximum(C1, 0.0).astype(np.float16).astype(np.float32)
    ref = np.empty(
        (conv.N, problem.pool_ho, problem.pool_wo, problem.conv1_channels),
        dtype=np.float32,
    )
    for ho in range(problem.pool_ho):
        for wo in range(problem.pool_wo):
            h0 = ho * problem.pool_stride_h
            w0 = wo * problem.pool_stride_w
            patch = C1[:, h0 : h0 + problem.pool_y, w0 : w0 + problem.pool_x, :]
            ref[:, ho, wo, :] = patch.max(axis=(1, 2))
    return ref.astype(np.float16)


def _verify_artifact(
    artifact, spec: Gfx950DeepFusedConvPoolSpec, *, seed: int, tol: float
) -> bool:
    import numpy as np
    from ck_dsl.runtime.hip_module import Runtime

    conv = spec.problem.conv
    rng = np.random.default_rng(seed)
    A = (
        rng.standard_normal((conv.N, conv.Hi, conv.Wi, conv.C)).astype(np.float32)
        * 0.25
    ).astype(np.float16)
    B0 = (
        rng.standard_normal((conv.K, conv.R, conv.S, conv.C)).astype(np.float32) * 0.25
    ).astype(np.float16)
    W1 = (
        rng.standard_normal((spec.problem.conv1_channels, conv.K)).astype(np.float32)
        * 0.25
    ).astype(np.float16)
    Y = np.zeros(
        (
            conv.N,
            spec.problem.pool_ho,
            spec.problem.pool_wo,
            spec.problem.conv1_channels,
        ),
        dtype=np.float16,
    )

    rt = Runtime()
    mod = rt.load_module(artifact.hsaco)
    fn = mod.get_function(artifact.kernel_name)
    A_dev = rt.alloc(A.nbytes)
    B_dev = rt.alloc(B0.nbytes)
    Y_dev = rt.alloc(Y.nbytes)
    W1_dev = rt.alloc(W1.nbytes)
    try:
        rt.memcpy_h2d(A_dev, _as_u8_buffer(A), A.nbytes)
        rt.memcpy_h2d(B_dev, _as_u8_buffer(B0), B0.nbytes)
        rt.memcpy_h2d(W1_dev, _as_u8_buffer(W1), W1.nbytes)
        rt.memset(Y_dev, 0, Y.nbytes)
        args = struct.pack(
            "<QQQQiiii",
            A_dev,
            B_dev,
            Y_dev,
            W1_dev,
            W1.nbytes,
            A.nbytes,
            B0.nbytes,
            Y.nbytes,
        )
        rt.launch_blocking(
            fn,
            deep_fused_conv_pool_grid(spec),
            (spec.block_size, 1, 1),
            args,
        )
        rt.memcpy_d2h(_as_u8_buffer(Y), Y_dev, Y.nbytes)
    finally:
        rt.free(A_dev)
        rt.free(B_dev)
        rt.free(Y_dev)
        rt.free(W1_dev)

    ref = _reference_conv1x1_relu_pool(A, B0, W1, conv, spec.problem)
    diff = np.abs(Y.astype(np.float32) - ref.astype(np.float32))
    max_diff = float(diff.max()) if diff.size else 0.0
    bad = int(np.count_nonzero(diff > tol))
    print(f"verify: max_abs_diff={max_diff:.6g} bad_count={bad}/{Y.size} tol={tol:g}")
    if bad:
        idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
        print(
            "verify: worst_idx="
            f"{idx} got={float(Y[idx]):.6g} ref={float(ref[idx]):.6g} "
            f"diff={float(diff[idx]):.6g}",
            file=sys.stderr,
        )
    return bad == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "output" / "deep_fused_conv_pool"),
    )
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--isa", default=None)
    parser.add_argument(
        "--verify", action="store_true", help="launch and compare to numpy"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--w", type=int, default=16)
    parser.add_argument(
        "--c",
        type=int,
        default=8,
        help="logical post-concat input channels; target exercise uses 8",
    )
    parser.add_argument(
        "--k0",
        type=int,
        default=32,
        help="conv0 output channels; target exercise uses 32",
    )
    parser.add_argument(
        "--k1",
        type=int,
        default=24,
        help="1x1 conv1 output channels; target exercise uses 24",
    )
    args = parser.parse_args()

    arch = args.arch
    if args.isa is not None:
        from ck_dsl.core.arch import arch_from_isa

        arch = arch_from_isa(args.isa) or arch
    if arch != "gfx950":
        print("deep_fused_conv_pool v1 is gfx950-only", file=sys.stderr)
        return 2

    target = ArchTarget.from_gfx(arch)
    atom = target.mma.select_largest_k(
        a_dtype="fp16", b_dtype="fp16", c_dtype="fp32", m=32, n=32
    )
    if atom is None:
        print(f"no fp16 32x32 MFMA atom for {arch}", file=sys.stderr)
        return 2

    conv = ConvProblem(
        N=args.n,
        Hi=args.h,
        Wi=args.w,
        C=args.c,
        K=args.k0,
        R=3,
        S=3,
        sH=1,
        sW=1,
        pH=1,
        pW=1,
        dH=1,
        dW=1,
    )
    problem = FusedConvPoolProblem(conv=conv, conv1_k=args.k1)
    spec = Gfx950DeepFusedConvPoolSpec(
        problem=problem,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=atom.k,
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
            conv.R,
            conv.S,
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
        atoms=[f"tile.mfma_f32_32x32x{atom.k}_f16"],
        notes=(
            "Experimental gfx950 deep-fusion prototype: implicit-GEMM conv0 "
            "accumulators are transformed by a static ReLU epilogue and staged "
            "through C-shuffle LDS; a 1x1 conv1 MFMA consumes that on-chip tile, "
            "applies a second ReLU epilogue, stages through LDS, then inline "
            "2x2 stride-2 maxpool writes the final global output. No conv0 or "
            "conv1 intermediate tensor is written to HBM."
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
            "conv1": {"kernel": "1x1", "K1": problem.conv1_channels},
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
    print(f"  pool_tile={spec.pool_tile_h}x{spec.pool_tile_w}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
