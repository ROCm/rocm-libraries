#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Full grouped convolution benchmark sweep.

JIT-compiles ALL grouped convolution kernels for a variant, then for EACH test problem
finds all matching kernels and benchmarks them, streaming results incrementally to CSV/JSON.

Results are printed live per-problem with the best kernel highlighted.
TFLOPS and latency come directly from CK's HIP event timing.

Usage:
    # Full sweep
    python grouped_conv_full_benchmark.py --variant forward --workers 256

    # Quick test
    python grouped_conv_full_benchmark.py --variant forward --category quick --max-kernels 10

    # Filter to large tiles
    python grouped_conv_full_benchmark.py --variant forward --filter "c.tile_n >= 128"
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[2] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))

from grouped_conv_utils import (  # noqa: E402
    GroupedConvProblem,
    GpuGroupedConvRunner,
    detect_gpu_arch,
    setup_multiple_grouped_conv_dispatchers,
)

from grouped_conv_instance_builder import expand_sweep, apply_filter  # noqa: E402


# Variant configs
VARIANT_CONFIGS = {
    "forward": "configs/forward.json",
    "bwd_data": "configs/bwd_data.json",
    "bwd_weight": "configs/bwd_weight.json",
}


@dataclass
class TestProblem:
    """Test problem specification for grouped convolution."""

    name: str
    category: str
    variant: str
    N: int
    C: int
    K: int
    G: int
    Hi: int
    Wi: int
    Y: int
    X: int
    stride_h: int = 1
    stride_w: int = 1
    pad_h: int = 0
    pad_w: int = 0
    dtype: str = "fp16"


# Problem definitions organized by category
def _expand_dtypes(problems, dtypes=["fp16", "bf16"]):
    """Expand problems to include multiple dtypes."""
    expanded = []
    for prob in problems:
        for dt in dtypes:
            p = TestProblem(
                prob.name,
                prob.category,
                prob.variant,
                prob.N,
                prob.C,
                prob.K,
                prob.G,
                prob.Hi,
                prob.Wi,
                prob.Y,
                prob.X,
                prob.stride_h,
                prob.stride_w,
                prob.pad_h,
                prob.pad_w,
                dt,
            )
            expanded.append(p)
    return expanded


FORWARD_PROBLEMS_QUICK = _expand_dtypes([
    TestProblem("ResNet-stage2", "quick", "forward", 1, 64, 64, 1, 56, 56, 3, 3, 1, 1, 1, 1),
    TestProblem("ResNet-stage3", "quick", "forward", 1, 128, 128, 1, 28, 28, 3, 3, 1, 1, 1, 1),
])

FORWARD_PROBLEMS_FULL = _expand_dtypes([
    TestProblem("ResNet-stage2", "full", "forward", 1, 64, 64, 1, 56, 56, 3, 3, 1, 1, 1, 1),
    TestProblem("ResNet-stage3", "full", "forward", 1, 128, 128, 1, 28, 28, 3, 3, 1, 1, 1, 1),
    TestProblem("ResNet-stage4", "full", "forward", 1, 256, 256, 1, 14, 14, 3, 3, 1, 1, 1, 1),
    TestProblem("ResNet-stage5", "full", "forward", 1, 512, 512, 1, 7, 7, 3, 3, 1, 1, 1, 1),
    TestProblem("ResNet-bottleneck", "full", "forward", 1, 256, 64, 1, 56, 56, 1, 1, 1, 1, 0, 0),
    TestProblem("ResNet-projection", "full", "forward", 1, 128, 256, 1, 28, 28, 1, 1, 2, 2, 0, 0),
    TestProblem("MobileNet-dw-112", "full", "forward", 1, 32, 32, 32, 112, 112, 3, 3, 1, 1, 1, 1),
    TestProblem("MobileNet-dw-56", "full", "forward", 1, 64, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1),
])

BWD_DATA_PROBLEMS_QUICK = _expand_dtypes([
    TestProblem("BwdData-small", "quick", "bwd_data", 1, 64, 64, 1, 28, 28, 3, 3, 1, 1, 1, 1),
])

BWD_DATA_PROBLEMS_FULL = _expand_dtypes([
    TestProblem("BwdData-small", "full", "bwd_data", 1, 64, 64, 1, 28, 28, 3, 3, 1, 1, 1, 1),
    TestProblem("BwdData-medium", "full", "bwd_data", 1, 128, 128, 1, 14, 14, 3, 3, 1, 1, 1, 1),
    TestProblem("BwdData-large", "full", "bwd_data", 1, 256, 256, 1, 7, 7, 3, 3, 1, 1, 1, 1),
])

BWD_WEIGHT_PROBLEMS_QUICK = _expand_dtypes([
    TestProblem("BwdWeight-small", "quick", "bwd_weight", 1, 64, 64, 1, 28, 28, 3, 3, 1, 1, 1, 1),
])

BWD_WEIGHT_PROBLEMS_FULL = _expand_dtypes([
    TestProblem("BwdWeight-small", "full", "bwd_weight", 1, 64, 64, 1, 28, 28, 3, 3, 1, 1, 1, 1),
    TestProblem("BwdWeight-medium", "full", "bwd_weight", 1, 128, 128, 1, 14, 14, 3, 3, 1, 1, 1, 1),
    TestProblem("BwdWeight-large", "full", "bwd_weight", 1, 256, 256, 1, 7, 7, 3, 3, 1, 1, 1, 1),
])

PROBLEM_SETS = {
    "forward": {
        "quick": FORWARD_PROBLEMS_QUICK,
        "full": FORWARD_PROBLEMS_FULL,
    },
    "bwd_data": {
        "quick": BWD_DATA_PROBLEMS_QUICK,
        "full": BWD_DATA_PROBLEMS_FULL,
    },
    "bwd_weight": {
        "quick": BWD_WEIGHT_PROBLEMS_QUICK,
        "full": BWD_WEIGHT_PROBLEMS_FULL,
    },
}


def parse_problems(variant: str, category: str) -> List[TestProblem]:
    """Get test problems for variant and category."""
    problems = []
    cats = ["quick"] if category == "quick" else ["quick", "full"]

    for cat in cats:
        problems.extend(PROBLEM_SETS[variant].get(cat, []))

    return problems


def compute_output_shape(prob: TestProblem) -> tuple:
    """Compute output spatial dimensions."""
    Ho = (prob.Hi + 2 * prob.pad_h - prob.Y) // prob.stride_h + 1
    Wo = (prob.Wi + 2 * prob.pad_w - prob.X) // prob.stride_w + 1
    return Ho, Wo


def main():
    parser = argparse.ArgumentParser(description="Full Grouped Convolution Benchmark Sweep")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument(
        "--variant", default="forward", choices=["forward", "bwd_data", "bwd_weight"]
    )
    parser.add_argument("--category", default="quick", choices=["quick", "full"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--build-dir", default="/tmp/grouped_conv_full_bench")
    parser.add_argument("--filter", dest="filter_expr", default="")
    parser.add_argument("--filter-file", default="")
    parser.add_argument("--csv", default="grouped_conv_sweep_results.csv")
    parser.add_argument("--json", default="grouped_conv_sweep_results.json")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--max-kernels", type=int, default=0)
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Parse problems
    print(f"\n{'=' * 80}")
    print("Phase 1: Parse test problems")
    print(f"{'=' * 80}")

    all_problems = parse_problems(args.variant, args.category)
    print(f"  Variant:  {args.variant}")
    print(f"  Category: {args.category}")
    print(f"  Total problems: {len(all_problems)}")

    # Phase 2: Compile kernels
    print(f"\n{'=' * 80}")
    print("Phase 2: Compile kernels")
    print(f"{'=' * 80}")

    cfg_path = str(_THIS_DIR / VARIANT_CONFIGS[args.variant])
    configs = expand_sweep(cfg_path, args.arch)

    if args.filter_expr or args.filter_file:
        configs = apply_filter(configs, args.filter_expr, args.filter_file)

    if args.max_kernels > 0:
        configs = configs[: args.max_kernels]

    if not configs:
        print("  No configs to compile. Exiting.")
        return

    print(f"\n  {args.variant}: {len(configs)} configs, {args.workers} workers...")
    t0 = time.perf_counter()

    libs = setup_multiple_grouped_conv_dispatchers(
        configs,
        verbose=True,
        max_workers=args.workers,
    )

    compile_time = time.perf_counter() - t0
    built = sum(1 for lib in libs if lib is not None)
    print(f"  Built {built}/{len(configs)} in {compile_time:.0f}s")

    # Create kernel index: (dtype, variant) -> list of (lib, config)
    kernel_index: Dict[tuple, List[tuple]] = {}
    for config, lib in zip(configs, libs):
        if lib is None:
            continue
        key = (config.dtype, config.variant)
        kernel_index.setdefault(key, []).append((lib, config))

    total_built = sum(len(v) for v in kernel_index.values())
    print(f"\n  Total compiled: {total_built}")
    print(f"  Unique (dtype, variant) groups: {len(kernel_index)}")

    if args.compile_only:
        print(f"\n  Compile-only. {total_built} kernels ready.")
        return

    # Phase 3: Benchmark
    print(f"\n{'=' * 80}")
    print("Phase 3: Benchmark (serial GPU execution)")
    print(f"{'=' * 80}")

    csv_path = Path(args.csv) if os.path.isabs(args.csv) else _THIS_DIR / args.csv
    csv_fields = [
        "problem_name",
        "N",
        "C",
        "K",
        "G",
        "Hi",
        "Wi",
        "Y",
        "X",
        "stride_h",
        "stride_w",
        "pad_h",
        "pad_w",
        "dtype",
        "kernel",
        "variant",
        "pipeline",
        "scheduler",
        "tile",
        "latency_ms",
        "tflops",
        "non_zero",
    ]

    # Resume support
    completed = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                completed.add(
                    (
                        row.get("kernel", ""),
                        row.get("problem_name", ""),
                        str(row.get("N", "")),
                        str(row.get("C", "")),
                    )
                )
        csv_file = open(csv_path, "a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        print(f"  Resuming: {len(completed)} measurements already in CSV")
    else:
        csv_file = open(csv_path, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

    dtype_map = {
        "fp16": np.float16,
        "bf16": np.float16,
        "fp32": np.float32,
        "fp8": np.float16,
        "bf8": np.float16,
    }

    np.random.seed(42)
    total_measurements = len(completed)
    bench_t0 = time.perf_counter()

    for prob in all_problems:
        np_dtype = dtype_map.get(prob.dtype, np.float16)
        key = (prob.dtype, prob.variant)
        kernel_entries = kernel_index.get(key, [])

        if not kernel_entries:
            print(f"\n  Problem: {prob.name} - No matching kernels for {key}")
            continue

        # Create input tensors
        if prob.variant == "forward":
            input_shape = (prob.N, prob.Hi, prob.Wi, prob.G, prob.C // prob.G)
            weight_shape = (prob.K, prob.Y, prob.X, prob.G, prob.C // prob.G)
        elif prob.variant == "bwd_data":
            Ho, Wo = compute_output_shape(prob)
            input_shape = (prob.N, Ho, Wo, prob.G, prob.K // prob.G)
            weight_shape = (prob.K, prob.Y, prob.X, prob.G, prob.C // prob.G)
        else:  # bwd_weight
            Ho, Wo = compute_output_shape(prob)
            input_shape = (prob.N, prob.Hi, prob.Wi, prob.G, prob.C // prob.G)
            weight_shape = (prob.N, Ho, Wo, prob.G, prob.K // prob.G)

        input_data = (np.random.randn(*input_shape) * 0.1).astype(np_dtype)
        weight_data = (np.random.randn(*weight_shape) * 0.1).astype(np_dtype)

        grouped_conv_prob = GroupedConvProblem(
            N=prob.N,
            C=prob.C,
            K=prob.K,
            G=prob.G,
            Hi=prob.Hi,
            Wi=prob.Wi,
            Y=prob.Y,
            X=prob.X,
            stride_h=prob.stride_h,
            stride_w=prob.stride_w,
            pad_h=prob.pad_h,
            pad_w=prob.pad_w,
            direction=prob.variant,
        )

        print(f"\n  Problem: {prob.name}")
        print(
            f"  N={prob.N} C={prob.C} K={prob.K} G={prob.G} "
            f"H={prob.Hi} W={prob.Wi} Y={prob.Y} X={prob.X}"
        )
        print(
            f"  {'Kernel':<55} {'Time(ms)':>10} {'TFLOPS':>10}"
            f" {'NonZero':>10}"
        )
        print(f"  {'-' * 90}")

        best_tflops = 0.0
        best_kernel = ""

        for lib, config in kernel_entries:
            resume_key = (config.name, prob.name, str(prob.N), str(prob.C))
            if resume_key in completed:
                continue

            runner = GpuGroupedConvRunner(lib_path=str(lib.path))
            if not runner.is_available():
                continue

            result = runner.run(input_data, weight_data, grouped_conv_prob)
            if not result.success:
                continue

            non_zero = 0
            if result.output is not None:
                non_zero = int(np.count_nonzero(result.output))

            if result.tflops > best_tflops:
                best_tflops = result.tflops
                best_kernel = config.name

            marker = " *BEST*" if config.name == best_kernel else ""
            print(
                f"  {config.name:<55} {result.time_ms:>10.3f}"
                f" {result.tflops:>10.2f} {non_zero:>10}{marker}"
            )

            row = {
                "problem_name": prob.name,
                "N": prob.N,
                "C": prob.C,
                "K": prob.K,
                "G": prob.G,
                "Hi": prob.Hi,
                "Wi": prob.Wi,
                "Y": prob.Y,
                "X": prob.X,
                "stride_h": prob.stride_h,
                "stride_w": prob.stride_w,
                "pad_h": prob.pad_h,
                "pad_w": prob.pad_w,
                "dtype": prob.dtype,
                "kernel": config.name,
                "variant": config.variant,
                "pipeline": config.pipeline,
                "scheduler": config.scheduler,
                "tile": config.tile_str,
                "latency_ms": result.time_ms,
                "tflops": result.tflops,
                "non_zero": non_zero,
            }
            writer.writerow(row)
            csv_file.flush()
            total_measurements += 1

        if best_kernel:
            print(f"\n  Best: {best_kernel} ({best_tflops:.2f} TFLOPS)")

    csv_file.close()
    bench_time = time.perf_counter() - bench_t0

    # Summary
    print(f"\n{'=' * 80}")
    print("Results")
    print(f"{'=' * 80}")
    print(f"  Measurements: {total_measurements}")
    print(f"  Compile time: {compile_time:.1f}s")
    print(f"  Benchmark time: {bench_time:.1f}s")
    print(f"  CSV: {csv_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
