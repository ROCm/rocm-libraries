#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Grouped Convolution tile engine benchmark runner.

Tests various problem dimensions with different algorithms and pipelines for
forward, backward data, and backward weight convolutions.

Problem sizes are based on actual CK tile examples and dispatcher tests.

Usage:
    # Forward convolution with ResNet problems
    python grouped_conv_benchmark.py --variant forward --problems resnet

    # Backward data with custom problem
    python grouped_conv_benchmark.py --variant bwd_data --problems "1,64,128,1,28,28,3,3"

    # All three directions
    for variant in forward bwd_data bwd_weight; do
      python grouped_conv_benchmark.py --variant $variant --problems resnet --csv results_$variant.csv
    done
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

_DISPATCHER_ROOT = Path(__file__).resolve().parents[3] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))

from grouped_conv_utils import (  # noqa: E402
    GroupedConvKernelConfig,
    GroupedConvProblem,
)

from dispatcher_common import detect_gpu_arch  # noqa: E402


# =============================================================================
# Problem Presets (from dispatcher/examples/grouped_conv/cpp/*.cpp)
# =============================================================================

# From 07_multi_tile_benchmark.cpp
RESNET_PROBLEMS_2D = [
    # name, N, C, K, G, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w
    ("ResNet-stage2", 1, 64, 64, 1, 56, 56, 3, 3, 1, 1, 1, 1),
    ("ResNet-stage3", 1, 128, 128, 1, 28, 28, 3, 3, 1, 1, 1, 1),
    ("ResNet-stage4", 1, 256, 256, 1, 14, 14, 3, 3, 1, 1, 1, 1),
    ("ResNet-stage5", 1, 512, 512, 1, 7, 7, 3, 3, 1, 1, 1, 1),
    ("Pointwise-1x1", 1, 256, 256, 1, 56, 56, 1, 1, 1, 1, 0, 0),
    ("Batch-8", 8, 64, 128, 1, 56, 56, 3, 3, 1, 1, 1, 1),
]

# From 05_bwd_data.cpp (default: N=1, C=64, K=128, size=14, Y=3, X=3)
BACKWARD_DATA_PROBLEMS = [
    ("BwdData-small", 1, 64, 128, 1, 14, 14, 3, 3, 1, 1, 1, 1),
    ("BwdData-medium", 1, 128, 256, 1, 28, 28, 3, 3, 1, 1, 1, 1),
    ("BwdData-large", 1, 256, 512, 1, 56, 56, 3, 3, 1, 1, 1, 1),
]

# From 06_bwd_weight.cpp (default: N=1, C=64, K=64, size=14)
BACKWARD_WEIGHT_PROBLEMS = [
    ("BwdWeight-small", 1, 64, 64, 1, 14, 14, 3, 3, 1, 1, 1, 1),
    ("BwdWeight-medium", 1, 128, 128, 1, 28, 28, 3, 3, 1, 1, 1, 1),
    ("BwdWeight-large", 1, 256, 256, 1, 56, 56, 3, 3, 1, 1, 1, 1),
]

# Grouped convolutions (from CK patterns)
GROUPED_CONV_PROBLEMS = [
    ("MobileNet-G2", 1, 64, 64, 2, 56, 56, 3, 3, 1, 1, 1, 1),
    ("MobileNet-G4", 1, 128, 128, 4, 28, 28, 3, 3, 1, 1, 1, 1),
    ("MobileNet-G8", 1, 256, 256, 8, 14, 14, 3, 3, 1, 1, 1, 1),
    ("DepthwiseC64", 1, 64, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1),  # G=C (depthwise)
]


def parse_problems(spec: str, variant: str) -> List[Tuple[str, GroupedConvProblem]]:
    """
    Parse problem specs.

    Args:
        spec: 'resnet', 'grouped', 'all', or 'N,C,K,G,H,W,Y,X;...'
        variant: 'forward', 'bwd_data', or 'bwd_weight'

    Returns:
        List of (name, GroupedConvProblem) tuples
    """
    problems = []

    if spec == "resnet":
        return [(name, create_problem_from_spec(vals, variant))
                for name, *vals in RESNET_PROBLEMS_2D]
    elif spec == "grouped":
        return [(name, create_problem_from_spec(vals, variant))
                for name, *vals in GROUPED_CONV_PROBLEMS]
    elif spec == "bwd_data":
        return [(name, create_problem_from_spec(vals, variant))
                for name, *vals in BACKWARD_DATA_PROBLEMS]
    elif spec == "bwd_weight":
        return [(name, create_problem_from_spec(vals, variant))
                for name, *vals in BACKWARD_WEIGHT_PROBLEMS]
    elif spec == "all":
        return (
            [(name, create_problem_from_spec(vals, variant))
             for name, *vals in RESNET_PROBLEMS_2D] +
            [(name, create_problem_from_spec(vals, variant))
             for name, *vals in GROUPED_CONV_PROBLEMS]
        )

    # Custom specs: 'N,C,K,G,H,W,Y,X;...'
    for idx, part in enumerate(spec.split(";")):
        vals = [int(x) for x in part.split(",")]
        prob = create_problem_from_spec(vals, variant)
        name = f"Custom-{idx}"
        problems.append((name, prob))

    return problems


def create_problem_from_spec(vals: List[int], variant: str) -> GroupedConvProblem:
    """Create GroupedConvProblem from value list."""
    if len(vals) == 8:  # N,C,K,G,H,W,Y,X (default stride=1, pad=kernel//2)
        N, C, K, G, H, W, Y, X = vals
        return GroupedConvProblem(
            N=N, C=C, K=K, G=G,
            Hi=H, Wi=W,
            Y=Y, X=X,
            stride_h=1, stride_w=1,
            dilation_h=1, dilation_w=1,
            pad_h=Y//2, pad_w=X//2,
            direction=variant,
        )
    elif len(vals) == 12:  # N,C,K,G,H,W,Y,X,SH,SW,PH,PW
        N, C, K, G, H, W, Y, X, SH, SW, PH, PW = vals
        return GroupedConvProblem(
            N=N, C=C, K=K, G=G,
            Hi=H, Wi=W,
            Y=Y, X=X,
            stride_h=SH, stride_w=SW,
            dilation_h=1, dilation_w=1,
            pad_h=PH, pad_w=PW,
            direction=variant,
        )
    elif len(vals) == 10:  # 3D: N,C,K,G,D,H,W,Z,Y,X
        N, C, K, G, D, H, W, Z, Y, X = vals
        return GroupedConvProblem(
            N=N, C=C, K=K, G=G,
            Di=D, Hi=H, Wi=W,
            Z=Z, Y=Y, X=X,
            stride_d=1, stride_h=1, stride_w=1,
            dilation_d=1, dilation_h=1, dilation_w=1,
            pad_d=Z//2, pad_h=Y//2, pad_w=X//2,
            direction=variant,
        )
    else:
        raise ValueError(f"Invalid problem spec length {len(vals)}: {vals}")


def generate_kernel_configs(
    variant: str,
    arch: str,
    dtype: str,
    ndim_spatial: int,
) -> List[GroupedConvKernelConfig]:
    """
    Generate kernel configurations based on dispatcher patterns.

    From dispatcher examples:
    - Small tile: 1x64x64, wave 1x4x1, warp 16x16x32
    - Medium tile: 1x128x128, wave 2x2x1, warp 32x32x16
    - Large tile: 1x256x256, wave 2x2x1, warp 32x32x16

    Pipelines:
    - forward: compv3, compv4
    - backward: compv3, mem (compv4 auto-corrects to compv3)

    Schedulers: intrawave, interwave
    """
    configs = []

    # Tile configurations from dispatcher/examples/grouped_conv/cpp/07_multi_tile_benchmark.cpp
    tile_configs = [
        # (tile_m, tile_n, tile_k, wave_m, wave_n, wave_k, warp_m, warp_n, warp_k)
        (1, 64, 64, 1, 4, 1, 16, 16, 32),      # Small
        (1, 128, 128, 2, 2, 1, 32, 32, 16),    # Medium
        (1, 256, 256, 2, 2, 1, 32, 32, 16),    # Large
    ]

    # Pipelines depend on variant
    if variant == "forward":
        pipelines = ["compv3", "compv4"]
    else:  # bwd_data, bwd_weight
        pipelines = ["compv3", "mem"]

    schedulers = ["intrawave", "interwave"]

    for tile_m, tile_n, tile_k, wave_m, wave_n, wave_k, warp_m, warp_n, warp_k in tile_configs:
        for pipeline in pipelines:
            for scheduler in schedulers:
                config = GroupedConvKernelConfig(
                    variant=variant,
                    ndim_spatial=ndim_spatial,
                    dtype=dtype,
                    arch=arch,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    tile_k=tile_k,
                    wave_m=wave_m,
                    wave_n=wave_n,
                    wave_k=wave_k,
                    warp_tile_m=warp_m,
                    warp_tile_n=warp_n,
                    warp_tile_k=warp_k,
                    pipeline=pipeline,
                    scheduler=scheduler,
                )
                configs.append(config)

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Grouped Convolution Tile Engine Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Forward convolution with ResNet problems
  python grouped_conv_benchmark.py --variant forward --problems resnet

  # Backward data with custom problem
  python grouped_conv_benchmark.py --variant bwd_data --problems "1,64,128,1,28,28,3,3"

  # All three directions with CSV output
  for variant in forward bwd_data bwd_weight; do
    python grouped_conv_benchmark.py --variant $variant --problems resnet --csv results_$variant.csv
  done

  # Show best kernel per problem
  python grouped_conv_benchmark.py --variant forward --problems all --best
        """
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument(
        "--variant",
        default="forward",
        choices=["forward", "bwd_data", "bwd_weight"],
        help="Convolution direction"
    )
    parser.add_argument(
        "--problems",
        default="resnet",
        help="Problem preset: 'resnet', 'grouped', 'bwd_data', 'bwd_weight', 'all', or 'N,C,K,G,H,W,Y,X;...'",
    )
    parser.add_argument("--dtype", default="fp16", help="Data type (fp16, bf16, fp32, etc.)")
    parser.add_argument("--ndim", type=int, default=2, choices=[1, 2, 3], help="Spatial dimensions")
    parser.add_argument("--best", action="store_true", help="Show best kernel per problem")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV file")
    parser.add_argument("--json", type=str, default=None, help="Output JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    problems = parse_problems(args.problems, args.variant)

    print(f"\n{'=' * 80}")
    print(f"Grouped Convolution Tile Engine Benchmark")
    print(f"{'=' * 80}")
    print(f"  Arch:     {args.arch}")
    print(f"  Variant:  {args.variant}")
    print(f"  Dtype:    {args.dtype}")
    print(f"  Ndim:     {args.ndim}D")
    print(f"  Problems: {len(problems)}")

    # Generate kernel configurations
    print(f"\n--- Generating Kernel Configurations ---")
    all_configs = generate_kernel_configs(
        variant=args.variant,
        arch=args.arch,
        dtype=args.dtype,
        ndim_spatial=args.ndim,
    )
    print(f"  Generated {len(all_configs)} kernel configs")

    # Display config breakdown
    pipeline_counts = defaultdict(int)
    scheduler_counts = defaultdict(int)
    tile_counts = defaultdict(int)
    for cfg in all_configs:
        pipeline_counts[cfg.pipeline] += 1
        scheduler_counts[cfg.scheduler] += 1
        tile_counts[cfg.tile_str] += 1

    print(f"  Pipelines:  {dict(pipeline_counts)}")
    print(f"  Schedulers: {dict(scheduler_counts)}")
    print(f"  Tiles:      {dict(tile_counts)}")

    # Benchmark
    print(f"\n--- Benchmark ({len(all_configs)} kernels x {len(problems)} problems) ---")
    print(f"\nNOTE: This is a benchmark framework placeholder.")
    print(f"      Actual kernel compilation and execution would happen here.\n")

    all_results = []
    bench_t0 = time.perf_counter()

    print(f"  {'Problem':<20} {'Kernel':<45} {'Pipeline':<10} {'Tile':<15} {'FLOPS':>12}")
    print(f"  {'-' * 110}")

    for prob_name, prob in problems:
        for cfg in all_configs:
            # Placeholder for actual kernel execution
            # In production: compile kernel, run on GPU, measure time

            print(
                f"  {prob_name:<20} {cfg.name[:45]:<45} {cfg.pipeline:<10} "
                f"{cfg.tile_str:<15} {prob.flops:>12.2e}"
            )

            all_results.append({
                "problem_name": prob_name,
                "kernel": cfg.name,
                "variant": cfg.variant,
                "dtype": cfg.dtype,
                "pipeline": cfg.pipeline,
                "scheduler": cfg.scheduler,
                "tile": cfg.tile_str,
                "wave": cfg.wave_str,
                "warp": cfg.warp_str,
                "problem": {
                    "N": prob.N,
                    "C": prob.C,
                    "K": prob.K,
                    "G": prob.G,
                    "Hi": prob.Hi,
                    "Wi": prob.Wi,
                    "Ho": prob.Ho,
                    "Wo": prob.Wo,
                    "Y": prob.Y,
                    "X": prob.X,
                },
                "flops": prob.flops,
            })

    bench_time = time.perf_counter() - bench_t0

    # Report
    print(f"\n{'=' * 80}")
    print(f"  Benchmark: {bench_time:.1f}s")
    print(f"  Results:   {len(all_results)} measurements")

    if args.best and all_results:
        print(f"\n  Kernel configurations per problem:")
        by_problem = defaultdict(list)
        for r in all_results:
            by_problem[r["problem_name"]].append(r)

        for prob_name in sorted(by_problem.keys()):
            configs = by_problem[prob_name]
            print(f"\n    {prob_name}:")
            print(f"      Kernels: {len(configs)}")
            print(f"      Pipelines: {set(c['pipeline'] for c in configs)}")
            print(f"      Tiles: {set(c['tile'] for c in configs)}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            fieldnames = [
                "problem_name", "kernel", "variant", "dtype", "pipeline", "scheduler",
                "tile", "wave", "warp",
                "N", "C", "K", "G", "Hi", "Wi", "Ho", "Wo", "Y", "X",
                "flops",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                row = {**r, **r["problem"]}
                del row["problem"]
                writer.writerow(row)
        print(f"\n  CSV: {args.csv}")

    if args.json:
        report = {
            "metadata": {
                "arch": args.arch,
                "variant": args.variant,
                "dtype": args.dtype,
                "ndim_spatial": args.ndim,
                "bench_time_s": bench_time,
                "num_kernels": len(all_configs),
                "num_problems": len(problems),
            },
            "results": all_results,
        }
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  JSON: {args.json}")

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
