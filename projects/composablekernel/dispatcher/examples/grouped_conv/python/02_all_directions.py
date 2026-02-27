#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 02: All Convolution Directions (Forward, BwdData, BwdWeight) x 2D/3D

GPU execution for all 6 kernel variants with CPU reference verification.

Usage:
    python3 02_all_directions.py
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from grouped_conv_utils import (
    GroupedConvKernelConfig,
    GroupedConvProblem,
    GpuGroupedConvRunner,
    setup_multiple_grouped_conv_dispatchers,
    validate_grouped_conv_config,
    detect_gpu_arch,
)


# =============================================================================
# CPU Reference Implementations
# =============================================================================

def ref_conv2d_fwd(inp, wei, prob):
    N, Hi, Wi, G, C = inp.shape
    _, Kpg, Y, X, _ = wei.shape
    Ho, Wo = prob.Ho, prob.Wo
    out = np.zeros((N, Ho, Wo, G, Kpg), dtype=np.float32)
    for n in range(N):
        for g in range(G):
            for ho in range(Ho):
                for wo in range(Wo):
                    for k in range(Kpg):
                        s = 0.0
                        for y in range(Y):
                            for x in range(X):
                                hi = ho * prob.stride_h - prob.pad_h + y
                                wi = wo * prob.stride_w - prob.pad_w + x
                                if 0 <= hi < Hi and 0 <= wi < Wi:
                                    for c in range(C):
                                        s += float(inp[n,hi,wi,g,c]) * float(wei[g,k,y,x,c])
                        out[n,ho,wo,g,k] = s
    return out


def ref_conv2d_bwd_data(dy, wei, prob):
    """CPU ref: compute dX from dY and W using transpose-conv logic."""
    N, Ho, Wo, G, Kpg = dy.shape
    _, _, Y, X, C = wei.shape
    Hi, Wi = prob.Hi, prob.Wi
    dx = np.zeros((N, Hi, Wi, G, C), dtype=np.float32)
    for n in range(N):
        for g in range(G):
            for hi in range(Hi):
                for wi in range(Wi):
                    for c in range(C):
                        s = 0.0
                        for y in range(Y):
                            for x in range(X):
                                ho = hi + prob.pad_h - y
                                wo = wi + prob.pad_w - x
                                if ho % prob.stride_h == 0 and wo % prob.stride_w == 0:
                                    ho //= prob.stride_h
                                    wo //= prob.stride_w
                                    if 0 <= ho < Ho and 0 <= wo < Wo:
                                        for k in range(Kpg):
                                            s += float(dy[n,ho,wo,g,k]) * float(wei[g,k,y,x,c])
                        dx[n,hi,wi,g,c] = s
    return dx


def ref_conv2d_bwd_weight(x, dy, prob):
    """CPU ref: compute dW from X and dY."""
    N, Hi, Wi, G, C = x.shape
    _, Ho, Wo, _, Kpg = dy.shape
    Y, X = prob.Y, prob.X
    dw = np.zeros((G, Kpg, Y, X, C), dtype=np.float32)
    for g in range(G):
        for k in range(Kpg):
            for y in range(Y):
                for xf in range(X):
                    for c in range(C):
                        s = 0.0
                        for n in range(N):
                            for ho in range(Ho):
                                for wo in range(Wo):
                                    hi = ho * prob.stride_h - prob.pad_h + y
                                    wi = wo * prob.stride_w - prob.pad_w + xf
                                    if 0 <= hi < Hi and 0 <= wi < Wi:
                                        s += float(x[n,hi,wi,g,c]) * float(dy[n,ho,wo,g,k])
                        dw[g,k,y,xf,c] = s
    return dw


def main():
    parser = argparse.ArgumentParser(description="All grouped-conv directions (2D/3D) with verification")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    args = parser.parse_args()

    arch = args.arch
    print("=" * 70)
    print("Example 02: All Convolution Directions x 2D/3D")
    print("=" * 70)
    print(f"\n  Arch: {arch}, Dtype: {args.dtype}")

    # Config validation for all directions
    print("\n--- Config Validation ---")
    for variant in ["forward", "bwd_data", "bwd_weight"]:
        for ndim in [2, 3]:
            cfg = GroupedConvKernelConfig(variant=variant, ndim_spatial=ndim, arch=arch)
            r = validate_grouped_conv_config(cfg.to_dict())
            print(f"  {variant:12s} {ndim}D: valid={r.is_valid}")

    key_order = [
        ("forward", 2),
        ("forward", 3),
        ("bwd_data", 2),
        ("bwd_data", 3),
        ("bwd_weight", 2),
        ("bwd_weight", 3),
    ]

    runner_by_key = {}
    jit_build_s = 0.0
    print("\n--- Python JIT Build ---")
    configs = [
        GroupedConvKernelConfig(
            variant=variant,
            ndim_spatial=ndim,
            arch=arch,
            dtype=args.dtype,
        )
        for variant, ndim in key_order
    ]
    t0 = time.perf_counter()
    jit_libs = setup_multiple_grouped_conv_dispatchers(configs, verbose=False)
    jit_build_s = time.perf_counter() - t0
    for i, key in enumerate(key_order):
        lib = jit_libs[i]
        if lib is None:
            print(f"  JIT {key[0]} {key[1]}D: FAILED")
            continue
        custom_runner = GpuGroupedConvRunner(lib_path=str(lib.path))
        if custom_runner.is_available():
            runner_by_key[key] = custom_runner
            print(f"  JIT {key[0]} {key[1]}D: {lib.path}")
        else:
            print(f"  JIT {key[0]} {key[1]}D: load failed")
    print(f"  JIT build time: {jit_build_s:.3f} s")

    missing = [key for key in key_order if key not in runner_by_key]
    if missing:
        print(f"\n  JIT unavailable for {len(missing)} configs: {missing}")
        return 1

    # GPU execution for all 6 variants
    print("\n--- GPU Execution (all 6 variants) ---")
    problems = {
        "fwd_2d": GroupedConvProblem(N=1, C=64, K=64, Hi=8, Wi=8, Y=3, X=3, pad_h=1, pad_w=1, direction="forward"),
        "fwd_3d": GroupedConvProblem(N=1, C=64, K=64, Di=8, Hi=8, Wi=8, Z=3, Y=3, X=3, pad_d=1, pad_h=1, pad_w=1, direction="forward"),
        "bwdd_2d": GroupedConvProblem(N=1, C=64, K=64, Hi=8, Wi=8, Y=3, X=3, pad_h=1, pad_w=1, direction="bwd_data"),
        "bwdd_3d": GroupedConvProblem(N=1, C=64, K=64, Di=8, Hi=8, Wi=8, Z=3, Y=3, X=3, pad_d=1, pad_h=1, pad_w=1, direction="bwd_data"),
        "bwdw_2d": GroupedConvProblem(N=1, C=64, K=64, Hi=8, Wi=8, Y=3, X=3, pad_h=1, pad_w=1, direction="bwd_weight"),
        "bwdw_3d": GroupedConvProblem(N=1, C=64, K=64, Di=8, Hi=8, Wi=8, Z=3, Y=3, X=3, pad_d=1, pad_h=1, pad_w=1, direction="bwd_weight"),
    }

    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32
    results = {}
    for name, prob in problems.items():
        d = prob.direction
        if d == "forward":
            a = np.random.uniform(-0.3, 0.3, prob.input_shape()).astype(np_dtype)
            b = np.random.uniform(-0.3, 0.3, prob.weight_shape()).astype(np_dtype)
        elif d == "bwd_data":
            a = np.random.uniform(-0.3, 0.3, prob.output_shape()).astype(np_dtype)  # dY
            b = np.random.uniform(-0.3, 0.3, prob.weight_shape()).astype(np_dtype)  # W
        elif d == "bwd_weight":
            a = np.random.uniform(-0.3, 0.3, prob.input_shape()).astype(np_dtype)   # X
            b = np.random.uniform(-0.3, 0.3, prob.output_shape()).astype(np_dtype)  # dY

        res = runner_by_key[(d, prob.ndim_spatial)].run(a, b, prob)
        nz = np.count_nonzero(res.output) if res.success else 0
        sz = res.output.size if res.success else 0
        results[name] = (res, a, b, prob)
        tag = "OK" if res.success else res.error
        print(f"  {name:10s}: {tag:12s}  time={res.time_ms:.4f}ms  TFLOPS={res.tflops:.2f}  nonzero={nz}/{sz}")

    # CPU reference verification for all 2D directions
    print("\n--- CPU Reference Verification (2D) ---")
    all_pass = True

    # Forward 2D: a=X, b=W
    res, x, w, prob = results["fwd_2d"]
    if res.success:
        ref = ref_conv2d_fwd(x, w, prob)
        d = np.abs(res.output.astype(np.float32) - ref)
        ok = np.allclose(res.output.astype(np.float32), ref, atol=0.05)
        print(f"  fwd_2d:  max_abs={d.max():.6f}  match={ok}")
        all_pass &= ok

    # BwdData 2D: a=dY, b=W -> c=dX
    res, dy, w, prob = results["bwdd_2d"]
    if res.success:
        ref = ref_conv2d_bwd_data(dy, w, prob)
        d = np.abs(res.output.astype(np.float32) - ref)
        ok = np.allclose(res.output.astype(np.float32), ref, atol=0.1)
        print(f"  bwdd_2d: max_abs={d.max():.6f}  match={ok}")
        all_pass &= ok

    # BwdWeight 2D: a=X, b=dY -> c=dW
    res, x, dy, prob = results["bwdw_2d"]
    if res.success:
        ref = ref_conv2d_bwd_weight(x, dy, prob)
        d = np.abs(res.output.astype(np.float32) - ref)
        ok = np.allclose(res.output.astype(np.float32), ref, atol=0.5)
        print(f"  bwdw_2d: max_abs={d.max():.6f}  match={ok}")
        all_pass &= ok

    for r in runner_by_key.values():
        r.cleanup()

    # Summary
    gpu_ok = all(r[0].success for r in results.values())
    status = "PASS" if gpu_ok and all_pass else "FAIL"
    print("\n" + "=" * 70)
    print(f"  GPU execution:  {sum(1 for r in results.values() if r[0].success)}/6 OK")
    print(f"  CPU ref match:  {'all pass' if all_pass else 'FAIL'}")
    if jit_build_s > 0.0:
        print(f"  JIT build time: {jit_build_s:.3f} s")
    print(f"  Status: {status}")
    print("=" * 70)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
