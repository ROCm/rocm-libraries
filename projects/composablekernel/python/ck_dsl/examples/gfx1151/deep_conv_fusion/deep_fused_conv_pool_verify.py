# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emit and verify the gfx1151 genuine-int8/int4 deep-fused conv + maxpool.

Wave32/WMMA sibling of ``examples/gfx950/deep_conv_fusion``. The kernel computes

    conv0 3x3 (int8) -> Quant(i32->i8) -> ReLU -> Quant(i8->i4)
    -> conv1 1x1 (int4) -> Quant(i32->i4) -> ReLU
    -> 2x2/s2 MaxPool -> Quant(i4->i4) -> packed-int4 output

entirely on-chip, with genuine low-bit HBM storage (int8 X/W0, packed-int4 W1
and Y). The numpy reference here is *integer-exact*: it reproduces the exact
clamp/round-half-even quant chain the kernel emits, so a correct kernel matches
bit-for-bit (tol defaults to 0 mismatches).

Tensor layouts (all NHWC, N=1):

  * ``X``  : int8  ``(N, Hi, Wi, C)``         contiguous
  * ``W0`` : int8  ``(K0, R, S, C)``          contiguous (== ``(K0, K_gemm)``)
  * ``W1`` : packed int4 ``(K1, K0//2)``      byte = (hi<<4)|(lo&0xF), lo=even k0
  * ``Y``  : int32 ``(pool_ho, pool_wo, ceil(K1/8))``  3 words/pixel, each word
             holds 8 signed nibbles: channel ch -> word ch//8, shift 4*(ch%8)

Kernel ABI: params (X, W0, Y, W1), no scalar size args -> pack ``<QQQQ>``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import struct
import sys
from pathlib import Path

import numpy as np

from ck_dsl.core.arch import ArchTarget
from ck_dsl.helpers import compile_kernel, make_conv_manifest, write_artifact
from ck_dsl.instances.gfx1151.deep_fused_conv_pool import (
    Gfx1151DeepFusedConvPoolSpec,
    build_deep_fused_conv_pool,
    deep_fused_conv_pool_grid,
    is_valid_spec,
    make_deep_fused_conv_pool_spec,
)


def _as_u8_buffer(array):
    return (ctypes.c_uint8 * int(array.nbytes)).from_buffer(array)


# --------------------------------------------------------------------------
# Integer-exact reference (mirrors the kernel's quant chain bit-for-bit).
# --------------------------------------------------------------------------


def _q(scaled_f32: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """clamp-then-round-half-even, matching the kernel's
    ``cvt_f32_to_i8_sat(clamp_f32(v*inv, lo, hi))``. All scales are exact
    powers of two so the f32 multiply that produced ``scaled_f32`` is exact."""
    return np.rint(np.clip(scaled_f32, lo, hi))


def _reference(X, W0, W1, spec: Gfx1151DeepFusedConvPoolSpec) -> np.ndarray:
    """Return final int4 codes ``(pool_ho, pool_wo, K1)`` as int32 in [-8, 7]."""
    p = spec.problem
    conv = p.conv
    N, Hi, Wi, C = X.shape
    K0, R, S = conv.K, conv.R, conv.S
    Ho, Wo = conv.Ho, conv.Wo
    K1 = p.conv1_channels

    # conv0 -> exact int32 accumulation (int8 x int8 over K_gemm).
    Xp = np.pad(
        X.astype(np.int64),
        ((0, 0), (conv.pH, conv.pH), (conv.pW, conv.pW), (0, 0)),
    )
    P0 = np.zeros((N, Ho, Wo, K0), dtype=np.int64)
    for r in range(R):
        for s in range(S):
            x = Xp[
                :,
                r * conv.dH : r * conv.dH + Ho * conv.sH : conv.sH,
                s * conv.dW : s * conv.dW + Wo * conv.sW : conv.sW,
                :,
            ]
            w = W0[:, r, s, :].astype(np.int64)  # (K0, C)
            P0 += np.einsum("nhwc,kc->nhwk", x, w, optimize=True)

    f0 = P0.astype(np.float32)
    q0 = _q(f0 * np.float32(spec.m0), -127.0, 127.0)
    q0r = np.maximum(q0, 0.0)  # ReLU
    C0 = _q(q0r * np.float32(spec.m0b), -8.0, 7.0)  # int4 codes

    # conv1 1x1 -> exact int32 accumulation (int4 x int4 over K0).
    P1 = np.einsum("nhwk,ok->nhwo", C0, W1.astype(np.float32), optimize=True)
    q1 = _q(P1 * np.float32(spec.m1), -8.0, 7.0)
    C1 = np.maximum(q1, 0.0)  # ReLU -> int4 codes

    # 2x2/s2 maxpool over int4 codes, then final int4 quant.
    ref = np.empty((p.pool_ho, p.pool_wo, K1), dtype=np.int32)
    for ho in range(p.pool_ho):
        for wo in range(p.pool_wo):
            h0 = ho * p.pool_stride_h
            w0 = wo * p.pool_stride_w
            patch = C1[0, h0 : h0 + p.pool_y, w0 : w0 + p.pool_x, :]
            pooled = patch.max(axis=(0, 1)).astype(np.float32)
            ref[ho, wo, :] = _q(pooled * np.float32(spec.mf), -8.0, 7.0).astype(
                np.int32
            )
    return ref


# --------------------------------------------------------------------------
# Data generation + int4 packing / unpacking.
# --------------------------------------------------------------------------


def _pack_w1(W1_codes: np.ndarray) -> np.ndarray:
    """``(K1, K0)`` signed int4 codes -> ``(K1, K0//2)`` int8 packed bytes."""
    K1, K0 = W1_codes.shape
    lo = W1_codes[:, 0::2].astype(np.int32) & 0xF
    hi = W1_codes[:, 1::2].astype(np.int32) & 0xF
    return ((hi << 4) | lo).astype(np.uint8).view(np.int8)


def _unpack_y(Y_words: np.ndarray, K1: int) -> np.ndarray:
    """``(pool_ho, pool_wo, words)`` int32 -> ``(pool_ho, pool_wo, K1)`` int4."""
    ph, pw, _ = Y_words.shape
    out = np.empty((ph, pw, K1), dtype=np.int32)
    for ch in range(K1):
        word = Y_words[:, :, ch // 8].astype(np.uint32)
        nib = (word >> (4 * (ch % 8))) & 0xF
        signed = nib.astype(np.int32)
        signed = np.where(signed >= 8, signed - 16, signed)
        out[:, :, ch] = signed
    return out


def _make_inputs(spec: Gfx1151DeepFusedConvPoolSpec, *, seed: int):
    conv = spec.problem.conv
    K1 = spec.problem.conv1_channels
    rng = np.random.default_rng(seed)
    X = rng.integers(-3, 4, size=(conv.N, conv.Hi, conv.Wi, conv.C), dtype=np.int8)
    W0 = rng.integers(-3, 4, size=(conv.K, conv.R, conv.S, conv.C), dtype=np.int8)
    W1_codes = rng.integers(-3, 4, size=(K1, conv.K), dtype=np.int8)
    W1 = _pack_w1(W1_codes)
    Y = np.zeros(
        (spec.problem.pool_ho, spec.problem.pool_wo, (K1 + 7) // 8),
        dtype=np.int32,
    )
    return X, W0, W1, W1_codes, Y


def _pack_args(X_dev, W0_dev, Y_dev, W1_dev) -> bytes:
    return struct.pack("<QQQQ", X_dev, W0_dev, Y_dev, W1_dev)


def _useful_flops(spec: Gfx1151DeepFusedConvPoolSpec) -> int:
    conv = spec.problem.conv
    conv0 = conv.N * conv.Ho * conv.Wo * conv.K * conv.R * conv.S * conv.C
    conv1 = conv.N * conv.Ho * conv.Wo * spec.problem.conv1_channels * conv.K
    return 2 * (conv0 + conv1)


def _deep_fused_args_signature():
    return [
        {"name": "X", "type": "ptr<i8, global>", "size_bytes": 8},
        {"name": "W0", "type": "ptr<i8, global>", "size_bytes": 8},
        {"name": "Y", "type": "ptr<i32, global>", "size_bytes": 8},
        {"name": "W1", "type": "ptr<i8, global>", "size_bytes": 8},
    ]


def _make_manifest(artifact, spec, *, seed: int, tol: int, warmup: int, iters: int):
    conv = spec.problem.conv
    problem = spec.problem
    grid = deep_fused_conv_pool_grid(spec)
    atoms = ["wmma_i32_16x16x16_iu8"] if spec.native_int else []
    atoms.append("wmma_f32_16x16x16_f16")
    return make_conv_manifest(
        artifact=artifact,
        block_m=spec.tile_m,
        block_n=spec.tile_n,
        block_k=spec.kpad,
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
        conv_layout="deep_fused_conv_pool_i8i4",
        warmup_iters=warmup,
        timed_iters=iters,
        atoms=atoms,
        notes=(
            "gfx1151 deep-fused int8/int4 conv0 -> conv1 -> 2x2/s2 maxpool. "
            "The kernel ABI is four pointers (X, W0, Y, W1); W1 and Y use "
            "packed signed int4 storage."
        ),
        extra={
            "kind": "deep_fused_conv_pool_i8i4",
            "args_signature": _deep_fused_args_signature(),
            "sig_has_bytes": 0,
            "pool": [
                problem.pool_y,
                problem.pool_x,
                problem.pool_stride_h,
                problem.pool_stride_w,
            ],
            "pool_tile": [spec.pool_tile_h, spec.pool_tile_w],
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
            "quant": {
                "m0": spec.m0,
                "m0b": spec.m0b,
                "m1": spec.m1,
                "mf": spec.mf,
            },
            "seed": int(seed),
            "verify_tol": int(tol),
            "experimental": True,
        },
    )


def _launch(rt, fn, spec, X, W0, W1, Y, *, blocking: bool):
    X_dev = rt.alloc(X.nbytes)
    W0_dev = rt.alloc(W0.nbytes)
    Y_dev = rt.alloc(Y.nbytes)
    W1_dev = rt.alloc(W1.nbytes)
    try:
        rt.memcpy_h2d(X_dev, _as_u8_buffer(X), X.nbytes)
        rt.memcpy_h2d(W0_dev, _as_u8_buffer(W0), W0.nbytes)
        rt.memcpy_h2d(W1_dev, _as_u8_buffer(W1), W1.nbytes)
        rt.memset(Y_dev, 0, Y.nbytes)
        args = _pack_args(X_dev, W0_dev, Y_dev, W1_dev)
        grid = deep_fused_conv_pool_grid(spec)
        block = (spec.block_size, 1, 1)
        if blocking:
            rt.launch_blocking(fn, grid, block, args)
            rt.memcpy_d2h(_as_u8_buffer(Y), Y_dev, Y.nbytes)
            return None
        # caller drives timing
        return (X_dev, W0_dev, Y_dev, W1_dev, args, grid, block)
    finally:
        if blocking:
            rt.free(X_dev)
            rt.free(W0_dev)
            rt.free(Y_dev)
            rt.free(W1_dev)


def _verify_artifact(artifact, spec, *, seed, tol) -> bool:
    from ck_dsl.runtime.hip_module import Runtime

    K1 = spec.problem.conv1_channels
    X, W0, W1, W1_codes, Y = _make_inputs(spec, seed=seed)
    W1_signed = W1_codes  # already signed int4 codes

    rt = Runtime()
    mod = rt.load_module(artifact.hsaco)
    fn = mod.get_function(artifact.kernel_name)
    _launch(rt, fn, spec, X, W0, W1, Y, blocking=True)

    got = _unpack_y(Y, K1)
    ref = _reference(X, W0, W1_signed, spec)
    diff = np.abs(got - ref)
    max_diff = int(diff.max()) if diff.size else 0
    bad = int(np.count_nonzero(diff > tol))
    print(f"verify: max_abs_diff={max_diff} bad_count={bad}/{got.size} tol={tol:g}")
    if bad:
        idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
        print(
            f"verify: worst_idx={idx} got={int(got[idx])} ref={int(ref[idx])}",
            file=sys.stderr,
        )
    return bad == 0


def _benchmark_artifact(artifact, spec, *, seed, warmup, iters) -> float:
    from ck_dsl.runtime.hip_module import Runtime

    warmup = max(int(warmup), 100)
    iters = max(int(iters), 1)
    X, W0, W1, _codes, Y = _make_inputs(spec, seed=seed)

    rt = Runtime()
    mod = rt.load_module(artifact.hsaco)
    fn = mod.get_function(artifact.kernel_name)
    grid = deep_fused_conv_pool_grid(spec)
    block = (spec.block_size, 1, 1)
    X_dev = rt.alloc(X.nbytes)
    W0_dev = rt.alloc(W0.nbytes)
    Y_dev = rt.alloc(Y.nbytes)
    W1_dev = rt.alloc(W1.nbytes)
    try:
        rt.memcpy_h2d(X_dev, _as_u8_buffer(X), X.nbytes)
        rt.memcpy_h2d(W0_dev, _as_u8_buffer(W0), W0.nbytes)
        rt.memcpy_h2d(W1_dev, _as_u8_buffer(W1), W1.nbytes)
        rt.memset(Y_dev, 0, Y.nbytes)
        args = _pack_args(X_dev, W0_dev, Y_dev, W1_dev)

        for _ in range(warmup):
            rt.launch(fn, grid, block, args)
        rt.sync()

        start = rt.event()
        end = rt.event()
        start.record()
        for _ in range(iters):
            rt.launch(fn, grid, block, args)
        end.record()
        end.synchronize()
        ms = start.elapsed_to(end) / iters
        start.destroy()
        end.destroy()
        rt.sync()
    finally:
        rt.free(X_dev)
        rt.free(W0_dev)
        rt.free(Y_dev)
        rt.free(W1_dev)

    useful_tflops = _useful_flops(spec) / 1e9 / ms
    print(
        f"bench: warmup={warmup} iters={iters} mean_ms={ms:.6g} "
        f"useful_TFLOPS={useful_tflops:.3f}"
    )
    return ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="gfx1151")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for the emitted hsaco and manifest.json",
    )
    parser.add_argument(
        "--emit-hsaco",
        default=None,
        help="build+compile only, write the hsaco to this path plus a sibling "
        "manifest.json, then exit (local half of a cross-build)",
    )
    parser.add_argument(
        "--prebuilt",
        default=None,
        help="skip compilation; load this prebuilt hsaco and verify/bench it "
        "(board half of the cross-build; needs no ROCm toolchain). The "
        "spec is reconstructed from the same CLI args, so kernel_name / "
        "grid / block match the build.",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tol", type=int, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--w", type=int, default=16)
    parser.add_argument("--c", type=int, default=8)
    parser.add_argument("--k0", type=int, default=32)
    parser.add_argument("--k1", type=int, default=24)
    # Best measured tile for the native-int direct path on gfx1151: a
    # wide-short 2x32 pool tile (tile_m=256) is memory/latency-bound-optimal,
    # ~16.1 ms full-shape vs ~18.5 ms for 8x8 (both bit-exact).
    parser.add_argument("--pool-tile-h", type=int, default=2)
    parser.add_argument("--pool-tile-w", type=int, default=32)
    parser.add_argument(
        "--direct",
        action="store_true",
        help="use the direct-conv0 footprint-cache A operand (vs im2col)",
    )
    # multi-lever campaign toggles
    parser.add_argument(
        "--waves-per-eu",
        type=int,
        default=0,
        help="L1: emit amdgpu-waves-per-eu launch bound (0 = unset)",
    )
    parser.add_argument(
        "--sched",
        default="compv4",
        choices=["mem", "compv3", "compv4", "intrawave"],
        help="L2: instruction-schedule policy for the WMMA loops "
        "(compv4 is the best measured policy for the native-int path)",
    )
    parser.add_argument(
        "--mask-maxpool",
        action="store_true",
        help="L3: predicated (branch-free) maxpool tail",
    )
    parser.add_argument(
        "--scalar-rne",
        action="store_true",
        help="compatibility no-op: scalar per-slot RNE is the default after measurement",
    )
    parser.add_argument(
        "--vector-rne",
        action="store_true",
        help="Q1: use vectorized fixed-scale RNE/clamp over WMMA accumulator vectors",
    )
    parser.add_argument(
        "--interior-fastpath",
        action="store_true",
        help="Q2: use no-halo-predicate direct input staging for interior CTAs",
    )
    parser.add_argument(
        "--static-direct-kmap",
        action="store_true",
        help="Q3: specialize direct-conv A fragment Kg->(r,s,ci) mapping",
    )
    parser.add_argument(
        "--packed-c0",
        action="store_true",
        help="Q4: pack conv0 C0 int4 handoff two codes per byte using even-lane stores",
    )
    parser.add_argument(
        "--repack-c0",
        action="store_true",
        help="Lever 2: lane-local LDS->LDS repack of C0 (no bpermute) so conv1 "
        "loads A as a bitcast instead of packing nibbles on every fragment load",
    )
    parser.add_argument(
        "--butterfly",
        action="store_true",
        help="L4: in-register conv0->conv1 C-frag transpose (no c0_smem/barrier)",
    )
    parser.add_argument(
        "--native-int",
        action="store_true",
        help="conv0 uses the native wmma_i32_16x16x16_iu8 atom (raw int8 -> i8 "
        "LDS -> exact i32 acc) instead of the fp16-emulation path; conv1 uses "
        "native packed-int4 WMMA when available",
    )
    parser.add_argument(
        "--batch-loads",
        dest="batch_loads",
        action="store_true",
        default=True,
        help="Lever 1 (DEFAULT ON): register-level multi-buffered footprint "
        "staging -- issue all conv0 input global loads to distinct VGPRs before "
        "any ds_store so they coalesce under a single vmcnt(0) and overlap "
        "(~5%% on the full shape, bit-exact)",
    )
    parser.add_argument(
        "--no-batch-loads",
        dest="batch_loads",
        action="store_false",
        help="disable Lever 1 footprint load batching (A/B baseline)",
    )
    args = parser.parse_args()

    if args.arch not in ("gfx1151", "gfx11-generic"):
        print(
            "deep_fused_conv_pool (this file) needs the gfx1151 wave32/WMMA ABI "
            "(gfx1151 or gfx11-generic)",
            file=sys.stderr,
        )
        return 2

    try:
        ArchTarget.from_gfx(args.arch)
    except KeyError as e:
        print(f"unknown arch {args.arch}: {e}", file=sys.stderr)
        return 2

    # butterfly is built on the direct-conv0 path, so force it on when requested.
    direct = args.direct or args.butterfly
    spec = make_deep_fused_conv_pool_spec(
        n=args.n,
        h=args.h,
        w=args.w,
        c=args.c,
        k0=args.k0,
        k1=args.k1,
        pool_tile_h=args.pool_tile_h,
        pool_tile_w=args.pool_tile_w,
        direct_conv0=direct,
        waves_per_eu=args.waves_per_eu,
        sched_policy=args.sched,
        mask_maxpool=args.mask_maxpool,
        specialized_rne=args.vector_rne and not args.scalar_rne,
        interior_fastpath=args.interior_fastpath,
        static_direct_kmap=args.static_direct_kmap,
        packed_c0_handoff=args.packed_c0,
        repack_c0=args.repack_c0,
        butterfly_conv01=args.butterfly,
        native_int=args.native_int,
        batch_loads=args.batch_loads,
    )
    ok, why = is_valid_spec(spec, arch=args.arch)
    if not ok:
        print(f"invalid spec: {why}", file=sys.stderr)
        return 2

    if args.prebuilt:
        # Board half: no toolchain. Reconstruct the artifact from a prebuilt
        # hsaco; kernel_name is reproduced from the (identical) spec.
        from types import SimpleNamespace

        hsaco = Path(args.prebuilt).read_bytes()
        artifact = SimpleNamespace(
            hsaco=hsaco,
            kernel_name=spec.kernel_name(),
            hsaco_bytes=len(hsaco),
            timings={},
        )
    else:
        kernel = build_deep_fused_conv_pool(spec, arch=args.arch)
        artifact = compile_kernel(kernel, arch=args.arch)

    manifest = _make_manifest(
        artifact,
        spec,
        seed=args.seed,
        tol=args.tol,
        warmup=args.warmup,
        iters=args.iters,
    )

    if args.emit_hsaco:
        hsaco_path = Path(args.emit_hsaco)
        hsaco_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = dict(manifest)
        manifest["hsaco"] = hsaco_path.name
        hsaco_path.write_bytes(artifact.hsaco)
        manifest_path = hsaco_path.with_name("manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"emitted {artifact.kernel_name} -> {hsaco_path} and {manifest_path} "
            f"({artifact.hsaco_bytes} bytes) for arch {args.arch}"
        )
        return 0

    out = Path(args.output_dir or f"/tmp/deep_fused_conv_pool_verify_{args.arch}")
    paths = write_artifact(
        artifact,
        out,
        manifest,
        write_ir_text=hasattr(artifact, "ir_text"),
        write_llvm_text=hasattr(artifact, "llvm_text"),
    )

    conv = spec.problem.conv
    p = spec.problem
    grid = deep_fused_conv_pool_grid(spec)
    print(
        f"emitted {paths['hsaco']} and {paths['manifest']} "
        f"({artifact.hsaco_bytes} bytes); "
        f"grid={grid} block=({spec.block_size},1,1)"
    )
    print(
        f"  conv0_input=[{conv.N},{conv.Hi},{conv.Wi},{conv.C}] K0={conv.K} "
        f"-> conv1 1x1 -> K1={p.conv1_channels} "
        f"-> pool_out=[{p.pool_ho},{p.pool_wo},{p.conv1_channels}]"
    )
    print(
        f"  tile_m={spec.tile_m} tile_n={spec.tile_n} kpad={spec.kpad} "
        f"pool_tile={spec.pool_tile_h}x{spec.pool_tile_w} "
        f"warps={spec.warp_m}x{spec.warp_n}"
    )

    if args.verify or args.bench:
        cmd = [
            sys.executable,
            "-m",
            "ck_dsl.run_manifest",
            str(paths["hsaco"]),
            str(paths["manifest"]),
        ]
        if args.verify:
            cmd.append("--verify")
        r = subprocess.run(cmd, text=True, timeout=300)
        return r.returncode
    print(
        "run with: "
        f"{sys.executable} -m ck_dsl.run_manifest {paths['hsaco']} "
        f"{paths['manifest']} --verify"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
