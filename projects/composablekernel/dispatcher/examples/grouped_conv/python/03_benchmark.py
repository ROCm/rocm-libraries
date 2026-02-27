#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 03: Multi-Problem GPU Benchmark

Runs actual GPU convolutions for common model architectures and reports TFLOPS.

Usage:
    python3 03_benchmark.py
    python3 03_benchmark.py --arch gfx950
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
    detect_gpu_arch,
)


def main():
    parser = argparse.ArgumentParser(description="Multi-Problem GPU Benchmark")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    args = parser.parse_args()

    print("=" * 70)
    print("Example 03: Multi-Problem GPU Benchmark")
    print("=" * 70)
    print(f"\n  Arch: {args.arch}, Dtype: {args.dtype}")

    # JIT is required for this example.
    key_order = [
        ("forward", 2),
        ("forward", 3),
        ("bwd_data", 2),
        ("bwd_weight", 2),
    ]
    print("\n--- Python JIT Build ---")
    configs = [
        GroupedConvKernelConfig(
            variant=variant,
            ndim_spatial=ndim,
            arch=args.arch,
            dtype=args.dtype,
        )
        for variant, ndim in key_order
    ]
    t0 = time.perf_counter()
    jit_libs = setup_multiple_grouped_conv_dispatchers(configs, verbose=False)
    jit_build_s = time.perf_counter() - t0

    runner_by_key = {}
    for i, key in enumerate(key_order):
        lib = jit_libs[i]
        if lib is None:
            print(f"  JIT {key[0]} {key[1]}D: FAILED")
            continue
        runner = GpuGroupedConvRunner(lib_path=str(lib.path))
        if runner.is_available():
            runner_by_key[key] = runner
            print(f"  JIT {key[0]} {key[1]}D: {lib.path}")
        else:
            print(f"  JIT {key[0]} {key[1]}D: load failed")

    missing = [key for key in key_order if key not in runner_by_key]
    print(f"  JIT build time: {jit_build_s:.3f} s")
    if missing:
        print(f"\n  ERROR: missing JIT runners for {missing}")
        return 1

    np_dtype = np.float16 if args.dtype in ["fp16", "bf16"] else np.float32

    # 2D benchmark problems
    problems_2d = [
        ("ResNet-stage2",  1,  64,  64,  56,  56, 3, 3, 1, 1),
        ("ResNet-stage3",  1, 128, 128,  28,  28, 3, 3, 1, 1),
        ("ResNet-stage4",  1, 256, 256,  14,  14, 3, 3, 1, 1),
        ("ResNet-stage5",  1, 512, 512,   7,   7, 3, 3, 1, 1),
        ("Pointwise-1x1",  1, 256, 256,  56,  56, 1, 1, 1, 0),
        ("Batch-8",         8,  64, 128,  56,  56, 3, 3, 1, 1),
        ("Batch-32",       32,  64, 128,  56,  56, 3, 3, 1, 1),
    ]

    print(f"\n{'Problem':<20} {'N':>4} {'C':>4} {'K':>4} {'H':>4} {'W':>4} "
          f"{'F':>3} {'Time(ms)':>10} {'TFLOPS':>8} {'Status':>8}")
    print("-" * 85)

    all_ok = True
    for label, n, c, k, h, w, y, x, s, p in problems_2d:
        prob = GroupedConvProblem(N=n, C=c, K=k, Hi=h, Wi=w, Y=y, X=x,
                                  stride_h=s, stride_w=s, pad_h=p, pad_w=p,
                                  direction="forward")
        inp = np.random.uniform(-0.3, 0.3, prob.input_shape()).astype(np_dtype)
        wei = np.random.uniform(-0.3, 0.3, prob.weight_shape()).astype(np_dtype)
        res = runner_by_key[("forward", 2)].run(inp, wei, prob)
        if res.success:
            print(f"{label:<20} {n:>4} {c:>4} {k:>4} {h:>4} {w:>4} "
                  f"{y}x{x} {res.time_ms:>10.4f} {res.tflops:>8.2f} {'OK':>8}")
        else:
            print(f"{label:<20} {n:>4} {c:>4} {k:>4} {h:>4} {w:>4} "
                  f"{y}x{x} {'---':>10} {'---':>8} {res.error:>8}")
            all_ok = False

    # 3D benchmark problems
    problems_3d = [
        ("3D-small",   1,  64,  64,  8, 16, 16, 3, 3, 3),
        ("3D-medium",  1,  64, 128, 16, 32, 32, 3, 3, 3),
    ]

    print(f"\n{'Problem':<20} {'N':>4} {'C':>4} {'K':>4} {'D':>4} {'H':>4} {'W':>4} "
          f"{'F':>5} {'Time(ms)':>10} {'TFLOPS':>8} {'Status':>8}")
    print("-" * 95)

    for label, n, c, k, d, h, w, z, y, x in problems_3d:
        prob = GroupedConvProblem(N=n, C=c, K=k, Di=d, Hi=h, Wi=w, Z=z, Y=y, X=x,
                                  direction="forward")
        inp = np.random.uniform(-0.3, 0.3, prob.input_shape()).astype(np_dtype)
        wei = np.random.uniform(-0.3, 0.3, prob.weight_shape()).astype(np_dtype)
        res = runner_by_key[("forward", 3)].run(inp, wei, prob)
        if res.success:
            print(f"{label:<20} {n:>4} {c:>4} {k:>4} {d:>4} {h:>4} {w:>4} "
                  f"{z}x{y}x{x} {res.time_ms:>10.4f} {res.tflops:>8.2f} {'OK':>8}")
        else:
            print(f"{label:<20} {n:>4} {c:>4} {k:>4} {d:>4} {h:>4} {w:>4} "
                  f"{z}x{y}x{x} {'---':>10} {'---':>8} {res.error:>8}")
            all_ok = False

    # Backward direction benchmarks
    print(f"\n--- Backward Directions ---")
    print(f"{'Problem':<20} {'Dir':>12} {'Time(ms)':>10} {'TFLOPS':>8} {'Status':>8}")
    print("-" * 65)

    for label, direction in [("ResNet-s3 bwdd", "bwd_data"), ("ResNet-s3 bwdw", "bwd_weight")]:
        prob = GroupedConvProblem(N=1, C=128, K=128, Hi=28, Wi=28, Y=3, X=3,
                                  stride_h=1, stride_w=1, pad_h=1, pad_w=1,
                                  direction=direction)
        inp = np.random.uniform(-0.3, 0.3, prob.input_shape()).astype(np_dtype)
        wei = np.random.uniform(-0.3, 0.3, prob.weight_shape()).astype(np_dtype)
        res = runner_by_key[(direction, 2)].run(inp, wei, prob)
        if res.success:
            print(f"{label:<20} {direction:>12} {res.time_ms:>10.4f} {res.tflops:>8.2f} {'OK':>8}")
        else:
            print(f"{label:<20} {direction:>12} {'---':>10} {'---':>8} {res.error:>8}")
            all_ok = False

    for runner in runner_by_key.values():
        runner.cleanup()

    status = "PASS" if all_ok else "FAIL"
    print("\n" + "=" * 70)
    print(f"  JIT build time: {jit_build_s:.3f} s")
    print(f"  Status: {status}")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
