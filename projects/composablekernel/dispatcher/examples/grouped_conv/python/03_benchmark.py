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
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from grouped_conv_utils import (
    GroupedConvProblem,
    GpuGroupedConvRunner,
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

    runner = GpuGroupedConvRunner()
    if not runner.is_available():
        print("\n  ERROR: GPU library not available. Build dispatcher_conv_lib first.")
        return 1

    print(f"  Library: {runner.library_path}")
    print(f"  Kernels: {runner.lib.kernel_names()}")

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

    print(f"\n{'Problem':<20} {'N':>3} {'C':>4} {'K':>4} {'H':>4} {'W':>4} "
          f"{'F':>3} {'GFLOPs':>8} {'ms':>8} {'TFLOPS':>8} {'Status':>8}")
    print("-" * 85)

    total_gflops = 0.0
    all_ok = True
    for label, n, c, k, h, w, y, x, s, p in problems_2d:
        prob = GroupedConvProblem(N=n, C=c, K=k, Hi=h, Wi=w, Y=y, X=x,
                                  stride_h=s, stride_w=s, pad_h=p, pad_w=p,
                                  direction="forward")
        inp = np.random.uniform(-0.3, 0.3, prob.input_shape()).astype(np.float16)
        wei = np.random.uniform(-0.3, 0.3, prob.weight_shape()).astype(np.float16)
        res = runner.run(inp, wei, prob)
        gf = prob.gflops
        total_gflops += gf
        if res.success:
            print(f"{label:<20} {n:>3} {c:>4} {k:>4} {h:>4} {w:>4} "
                  f"{y}x{x} {gf:>8.2f} {res.time_ms:>8.4f} {res.tflops:>8.2f} {'OK':>8}")
        else:
            print(f"{label:<20} {n:>3} {c:>4} {k:>4} {h:>4} {w:>4} "
                  f"{y}x{x} {gf:>8.2f} {'---':>8} {'---':>8} {res.error:>8}")
            all_ok = False

    print("-" * 85)
    print(f"{'Total 2D':<20} {'':>3} {'':>4} {'':>4} {'':>4} {'':>4} "
          f"{'':>3} {total_gflops:>8.2f}")

    # 3D benchmark problems
    problems_3d = [
        ("3D-small",   1,  64,  64,  8, 16, 16, 3, 3, 3),
        ("3D-medium",  1,  64, 128, 16, 32, 32, 3, 3, 3),
    ]

    print(f"\n{'Problem':<20} {'N':>3} {'C':>4} {'K':>4} {'D':>4} {'H':>4} {'W':>4} "
          f"{'F':>5} {'GFLOPs':>8} {'ms':>8} {'TFLOPS':>8} {'Status':>8}")
    print("-" * 95)

    total_3d = 0.0
    for label, n, c, k, d, h, w, z, y, x in problems_3d:
        prob = GroupedConvProblem(N=n, C=c, K=k, Di=d, Hi=h, Wi=w, Z=z, Y=y, X=x,
                                  direction="forward")
        inp = np.random.uniform(-0.3, 0.3, prob.input_shape()).astype(np.float16)
        wei = np.random.uniform(-0.3, 0.3, prob.weight_shape()).astype(np.float16)
        res = runner.run(inp, wei, prob)
        gf = prob.gflops
        total_3d += gf
        if res.success:
            print(f"{label:<20} {n:>3} {c:>4} {k:>4} {d:>4} {h:>4} {w:>4} "
                  f"{z}x{y}x{x} {gf:>8.2f} {res.time_ms:>8.4f} {res.tflops:>8.2f} {'OK':>8}")
        else:
            print(f"{label:<20} {n:>3} {c:>4} {k:>4} {d:>4} {h:>4} {w:>4} "
                  f"{z}x{y}x{x} {gf:>8.2f} {'---':>8} {'---':>8} {res.error:>8}")
            all_ok = False

    # Backward direction benchmarks
    print(f"\n--- Backward Directions ---")
    print(f"{'Problem':<20} {'Dir':>8} {'GFLOPs':>8} {'ms':>8} {'TFLOPS':>8} {'Status':>8}")
    print("-" * 60)

    for label, direction in [("ResNet-s3 bwdd", "bwd_data"), ("ResNet-s3 bwdw", "bwd_weight")]:
        prob = GroupedConvProblem(N=1, C=128, K=128, Hi=28, Wi=28, Y=3, X=3,
                                  stride_h=1, stride_w=1, pad_h=1, pad_w=1,
                                  direction=direction)
        inp = np.random.uniform(-0.3, 0.3, prob.input_shape()).astype(np.float16)
        wei = np.random.uniform(-0.3, 0.3, prob.weight_shape()).astype(np.float16)
        res = runner.run(inp, wei, prob)
        gf = prob.gflops
        if res.success:
            print(f"{label:<20} {direction:>8} {gf:>8.2f} {res.time_ms:>8.4f} {res.tflops:>8.2f} {'OK':>8}")
        else:
            print(f"{label:<20} {direction:>8} {gf:>8.2f} {'---':>8} {'---':>8} {res.error:>8}")

    runner.cleanup()

    status = "PASS" if all_ok else "PARTIAL"
    print("\n" + "=" * 70)
    print(f"  Total GFLOPs: {total_gflops + total_3d:.2f}")
    print(f"  Status: {status}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
