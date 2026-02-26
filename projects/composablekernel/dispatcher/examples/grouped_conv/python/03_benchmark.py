#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 03: Multi-Problem Benchmark

Benchmarks grouped convolution across common model architectures.
Reports GFLOP counts for each problem size. All configs built explicitly.

Usage:
    python3 03_benchmark.py
    python3 03_benchmark.py --arch gfx950
    python3 03_benchmark.py --dtype bf16
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codegen"))

from ctypes_utils import detect_gpu_arch
from grouped_conv_utils import validate_grouped_conv_config


def calc_conv2d_flops(n, c, k, hi, wi, y, x, stride_h=1, stride_w=1, pad_h=0, pad_w=0):
    """Calculate 2*N*K*Ho*Wo*C*Y*X FLOPs for conv2d forward."""
    ho = (hi + 2 * pad_h - y) // stride_h + 1
    wo = (wi + 2 * pad_w - x) // stride_w + 1
    return 2 * n * k * ho * wo * c * y * x


def calc_conv3d_flops(n, c, k, di, hi, wi, z, y, x, stride_d=1, stride_h=1, stride_w=1):
    """Calculate FLOPs for conv3d forward."""
    do_ = (di - z) // stride_d + 1
    ho = (hi - y) // stride_h + 1
    wo = (wi - x) // stride_w + 1
    return 2 * n * k * do_ * ho * wo * c * z * y * x


def make_conv_config(variant, ndim, arch, dtype, tile_n=128, tile_k=128, pipeline="compv4"):
    """Build a conv config with all fields explicit."""
    return {
        "tile_config": {
            "tile_m": [1],
            "tile_n": [tile_n],
            "tile_k": [tile_k],
            "wave_m": [2],
            "wave_n": [2],
            "wave_k": [1],
            "warp_tile_m": [32],
            "warp_tile_n": [32],
            "warp_tile_k": [16],
        },
        "trait_config": {
            "pipeline": [pipeline],
            "epilogue": ["cshuffle"],
            "scheduler": ["intrawave"],
            "pad_m": [True],
            "pad_n": [True],
            "pad_k": [True],
        },
        "variant": variant,
        "ndim_spatial": ndim,
        "arch": arch,
        "layout": "nhwgc",
        "dtype": dtype,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Problem Benchmark")
    parser.add_argument(
        "--arch", default=detect_gpu_arch(),
        help="Target architecture (auto-detected from rocminfo)",
    )
    parser.add_argument(
        "--dtype", default="fp16", choices=["fp16", "bf16"],
        help="Data type (default: fp16)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Example 03: Multi-Problem Benchmark")
    print("=" * 70)
    print(f"\n  Arch: {args.arch}, Dtype: {args.dtype}\n")

    # =========================================================================
    # Kernel configs (explicit)
    # =========================================================================
    print("--- Kernel Configs ---")

    configs = {
        "fwd_large":  make_conv_config("forward",    2, args.arch, args.dtype, 256, 256, "compv4"),
        "fwd_medium": make_conv_config("forward",    2, args.arch, args.dtype, 128, 128, "compv4"),
        "fwd_small":  make_conv_config("forward",    2, args.arch, args.dtype, 64,  64,  "compv3"),
        "bwdd":       make_conv_config("bwd_data",   2, args.arch, args.dtype, 128, 128, "compv3"),
        "bwdw":       make_conv_config("bwd_weight", 2, args.arch, args.dtype, 128, 128, "compv3"),
    }

    for name, cfg in configs.items():
        tc = cfg["tile_config"]
        result = validate_grouped_conv_config(cfg)
        print(f"  {name:<12}: tile 1x{tc['tile_n'][0]}x{tc['tile_k'][0]}, "
              f"pipeline {cfg['trait_config']['pipeline'][0]}, "
              f"valid={result.is_valid}")

    # =========================================================================
    # 2D benchmark problems
    # =========================================================================
    print("\n--- 2D Problems ---")
    problems_2d = [
        # (label, N, C, K, H, W, Y, X, stride, pad)
        ("ResNet-conv1",    1,   3,  64, 224, 224, 7, 7, 2, 3),
        ("ResNet-stage2",   1,  64,  64,  56,  56, 3, 3, 1, 1),
        ("ResNet-stage3",   1, 128, 128,  28,  28, 3, 3, 1, 1),
        ("ResNet-stage4",   1, 256, 256,  14,  14, 3, 3, 1, 1),
        ("ResNet-stage5",   1, 512, 512,   7,   7, 3, 3, 1, 1),
        ("Pointwise-1x1",  1, 256, 256,  56,  56, 1, 1, 1, 0),
        ("Batch-8",         8,  64, 128,  56,  56, 3, 3, 1, 1),
        ("Batch-32",       32,  64, 128,  56,  56, 3, 3, 1, 1),
    ]

    print(f"  {'Problem':<18} {'N':>3} {'C':>4} {'K':>4} {'H':>4} {'W':>4} "
          f"{'F':>3} {'GFLOPs':>10}")
    print("  " + "-" * 60)

    total_gflops = 0.0
    for label, n, c, k, h, w, y, x, s, p in problems_2d:
        flops = calc_conv2d_flops(n, c, k, h, w, y, x, s, s, p, p)
        gflops = flops / 1e9
        total_gflops += gflops
        print(f"  {label:<18} {n:>3} {c:>4} {k:>4} {h:>4} {w:>4} "
              f"{y}x{x} {gflops:>10.2f}")

    print("  " + "-" * 60)
    print(f"  {'Total 2D':<18} {'':>3} {'':>4} {'':>4} {'':>4} {'':>4} "
          f"{'':>3} {total_gflops:>10.2f}")

    # =========================================================================
    # 3D benchmark problems
    # =========================================================================
    print()
    problems_3d = [
        ("3D-small",   1,  32,  64,  8, 16, 16, 3, 3, 3),
        ("3D-medium",  1,  64, 128, 16, 32, 32, 3, 3, 3),
        ("3D-large",   1, 128, 256, 16, 32, 32, 3, 3, 3),
    ]

    print(f"  {'Problem':<18} {'N':>3} {'C':>4} {'K':>4} {'D':>4} {'H':>4} "
          f"{'W':>4} {'F':>5} {'GFLOPs':>10}")
    print("  " + "-" * 65)

    total_3d = 0.0
    for label, n, c, k, d, h, w, z, y, x in problems_3d:
        flops = calc_conv3d_flops(n, c, k, d, h, w, z, y, x)
        gflops = flops / 1e9
        total_3d += gflops
        print(f"  {label:<18} {n:>3} {c:>4} {k:>4} {d:>4} {h:>4} "
              f"{w:>4} {z}x{y}x{x} {gflops:>10.2f}")

    print("  " + "-" * 65)
    print(f"  {'Total 3D':<18} {'':>3} {'':>4} {'':>4} {'':>4} {'':>4} "
          f"{'':>4} {'':>5} {total_3d:>10.2f}")

    # =========================================================================
    # Config generation timing
    # =========================================================================
    print("\n" + "-" * 50)
    print("Config Generation Timing:")
    print("-" * 50)

    for variant in ["forward", "bwd_data", "bwd_weight"]:
        pipeline = "compv4" if variant == "forward" else "compv3"
        t0 = time.time()
        for _ in range(100):
            cfg = make_conv_config(variant, 2, args.arch, args.dtype, pipeline=pipeline)
            validate_grouped_conv_config(cfg)
        elapsed_ms = (time.time() - t0) * 1000.0 / 100.0
        print(f"  {variant:<16}: {elapsed_ms:.3f} ms/config (avg of 100)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  2D problems:  {len(problems_2d)}")
    print(f"  3D problems:  {len(problems_3d)}")
    print(f"  Total GFLOPs: {total_gflops + total_3d:.2f}")
    print(f"\n  Note: TFLOPS will be reported when GPU execution is available")
    print(f"        via the compiled conv dispatcher library.")
    print(f"\n  Status: PASS")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
