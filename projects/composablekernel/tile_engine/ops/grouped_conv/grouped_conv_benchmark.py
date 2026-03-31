#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Grouped Convolution tile engine GPU benchmark runner.

Uses the dispatcher's setup_multiple_grouped_conv_dispatchers() for pipelined JIT
compilation, then runs GPU benchmarks and reports results.

Usage:
    python grouped_conv_gpu_benchmark.py configs/forward.json
    python grouped_conv_gpu_benchmark.py configs/forward.json --workers 256
    python grouped_conv_gpu_benchmark.py configs/bwd_data.json --problems custom --verify
    python grouped_conv_gpu_benchmark.py configs/receipt0_forward.json --best
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[2] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))

from grouped_conv_utils import (  # noqa: E402
    GroupedConvProblem,
    detect_gpu_arch,
    setup_multiple_grouped_conv_dispatchers,
)

from grouped_conv_instance_builder import expand_sweep, apply_filter  # noqa: E402


# =============================================================================
# Problem Presets
# =============================================================================

# ResNet problem sizes (from dispatcher/examples/grouped_conv/cpp/07_multi_tile_benchmark.cpp)
RESNET_PROBLEMS_2D = [
    ("ResNet-stage2", 1, 64, 64, 1, 56, 56, 3, 3, 1, 1, 1, 1),
    ("ResNet-stage3", 1, 128, 128, 1, 28, 28, 3, 3, 1, 1, 1, 1),
    ("ResNet-stage4", 1, 256, 256, 1, 14, 14, 3, 3, 1, 1, 1, 1),
    ("ResNet-stage5", 1, 512, 512, 1, 7, 7, 3, 3, 1, 1, 1, 1),
    ("ResNet-bottleneck", 1, 256, 64, 1, 56, 56, 1, 1, 1, 1, 0, 0),
    ("ResNet-projection", 1, 128, 256, 1, 28, 28, 1, 1, 2, 2, 0, 0),
]

# MobileNet grouped convolutions
MOBILENET_PROBLEMS_2D = [
    ("MobileNet-depthwise-112", 1, 32, 32, 32, 112, 112, 3, 3, 1, 1, 1, 1),
    ("MobileNet-depthwise-56", 1, 64, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1),
    ("MobileNet-depthwise-28", 1, 128, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1),
    ("MobileNet-depthwise-14", 1, 256, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1),
]

# Backward data problems (from dispatcher example 05)
BWD_DATA_PROBLEMS_2D = [
    ("BwdData-small", 1, 64, 64, 1, 28, 28, 3, 3, 1, 1, 1, 1),
    ("BwdData-medium", 1, 128, 128, 1, 14, 14, 3, 3, 1, 1, 1, 1),
    ("BwdData-large", 1, 256, 256, 1, 7, 7, 3, 3, 1, 1, 1, 1),
]

# Backward weight problems (from dispatcher example 06)
BWD_WEIGHT_PROBLEMS_2D = [
    ("BwdWeight-small", 1, 64, 64, 1, 28, 28, 3, 3, 1, 1, 1, 1),
    ("BwdWeight-medium", 1, 128, 128, 1, 14, 14, 3, 3, 1, 1, 1, 1),
    ("BwdWeight-large", 1, 256, 256, 1, 7, 7, 3, 3, 1, 1, 1, 1),
]

PROBLEM_PRESETS = {
    "resnet": RESNET_PROBLEMS_2D,
    "mobilenet": MOBILENET_PROBLEMS_2D,
    "grouped": MOBILENET_PROBLEMS_2D,
    "bwd_data": BWD_DATA_PROBLEMS_2D,
    "bwd_weight": BWD_WEIGHT_PROBLEMS_2D,
    "all": RESNET_PROBLEMS_2D + MOBILENET_PROBLEMS_2D,
}


def parse_problems(spec: str, variant: str) -> List[Tuple[str, GroupedConvProblem]]:
    """Parse problem specs or use presets.

    Args:
        spec: Either a preset name or "N,C,K,G,H,W,Y,X,..." custom format
        variant: forward, bwd_data, or bwd_weight

    Returns:
        List of (name, GroupedConvProblem) tuples
    """
    if spec in PROBLEM_PRESETS:
        raw = PROBLEM_PRESETS[spec]
        problems = []
        for row in raw:
            name, N, C, K, G, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w = row
            problems.append(
                (
                    name,
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=G,
                        Hi=Hi,
                        Wi=Wi,
                        Y=Y,
                        X=X,
                        stride_h=stride_h,
                        stride_w=stride_w,
                        pad_h=pad_h,
                        pad_w=pad_w,
                        direction=variant,
                    ),
                )
            )
        return problems

    # Custom format: "N,C,K,G,H,W,Y,X;..."
    problems = []
    for part in spec.split(";"):
        vals = [int(x) for x in part.split(",")]
        if len(vals) >= 8:
            N, C, K, G, H, W, Y, X = vals[:8]
            stride_h = vals[8] if len(vals) > 8 else 1
            stride_w = vals[9] if len(vals) > 9 else 1
            pad_h = vals[10] if len(vals) > 10 else 0
            pad_w = vals[11] if len(vals) > 11 else 0
            name = f"Custom-N{N}C{C}K{K}G{G}H{H}W{W}"
            problems.append(
                (
                    name,
                    GroupedConvProblem(
                        N=N,
                        C=C,
                        K=K,
                        G=G,
                        Hi=H,
                        Wi=W,
                        Y=Y,
                        X=X,
                        stride_h=stride_h,
                        stride_w=stride_w,
                        pad_h=pad_h,
                        pad_w=pad_w,
                        direction=variant,
                    ),
                )
            )
    return problems


def compute_output_shape(prob: GroupedConvProblem) -> Tuple[int, int]:
    """Compute output spatial dimensions."""
    Ho = (prob.Hi + 2 * prob.pad_h - prob.Y) // prob.stride_h + 1
    Wo = (prob.Wi + 2 * prob.pad_w - prob.X) // prob.stride_w + 1
    return Ho, Wo


def main():
    parser = argparse.ArgumentParser(
        description="Grouped Convolution Tile Engine GPU Benchmark"
    )
    parser.add_argument("configs", nargs="+", help="Sweep config JSON(s)")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--workers", type=int, default=8, help="Parallel JIT workers")
    parser.add_argument(
        "--problems",
        default="resnet",
        help="Problem preset or custom: N,C,K,G,H,W,Y,X,...",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify output is non-zero"
    )
    parser.add_argument(
        "--best", action="store_true", help="Show best kernel per problem"
    )
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument(
        "--build-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "build"),
        help="JIT build output directory",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--filter",
        dest="filter_expr",
        default="",
        help='Python expr per config, e.g. "c.tile_n >= 128"',
    )
    parser.add_argument(
        "--filter-file", default="", help="Path to .py with filter_config(c) -> bool"
    )
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()

    if args.clean and build_dir.exists():
        print(f"  Cleaning {build_dir} ...")
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0: Expand configs
    all_configs = []
    for cfg_path in args.configs:
        configs = expand_sweep(cfg_path, args.arch)
        all_configs.extend(configs)
        print(f"  {cfg_path}: {len(configs)} kernel configs")

    if args.filter_expr or args.filter_file:
        before = len(all_configs)
        all_configs = apply_filter(all_configs, args.filter_expr, args.filter_file)
        print(f"  Filter: {before} -> {len(all_configs)} configs")

    if not all_configs:
        print("  No kernel configs to benchmark. Exiting.")
        return

    # Determine variant from first config
    variant = all_configs[0].variant
    problems = parse_problems(args.problems, variant)

    print(f"\n{'=' * 80}")
    print("Grouped Convolution Tile Engine GPU Benchmark")
    print(f"{'=' * 80}")
    print(f"  Arch:     {args.arch}")
    print(f"  Variant:  {variant}")
    print(f"  Kernels:  {len(all_configs)}")
    print(f"  Problems: {len(problems)}")
    print(f"  Workers:  {args.workers}")
    print(f"  Build:    {build_dir}")

    # Phase 1: Pipelined JIT via the dispatcher
    print(
        f"\n--- Phase 1: JIT compile ({len(all_configs)} kernels,"
        f" {args.workers} workers) ---"
    )
    jit_t0 = time.perf_counter()

    # Returns library paths WITHOUT loading them (avoids GPU context during compilation)
    lib_paths = setup_multiple_grouped_conv_dispatchers(
        all_configs,
        verbose=True,
        max_workers=args.workers,
    )

    jit_time = time.perf_counter() - jit_t0
    built = sum(1 for p in lib_paths if p is not None)
    failed = len(all_configs) - built
    print(f"\n  Built {built}/{len(all_configs)} in {jit_time:.0f}s ({failed} failed)")

    if args.compile_only:
        print(f"\n{'=' * 80}")
        print(f"  Compile-only mode. {built} kernels ready.")
        print(f"{'=' * 80}")
        return

    # Phase 2: Benchmark
    print(f"\n--- Phase 2: Benchmark ({built} kernels x {len(problems)} problems) ---")

    dtype_map = {
        "fp16": np.float16,
        "bf16": np.float16,
        "fp32": np.float32,
        "fp8": np.float16,
        "bf8": np.float16,
        "int8": np.int8,
    }
    np.random.seed(42)
    all_results = []
    bench_t0 = time.perf_counter()

    for prob_idx, (prob_name, prob) in enumerate(problems):
        first_dtype = all_configs[0].dtype if all_configs else "fp16"
        np_dtype = dtype_map.get(first_dtype, np.float16)

        # Create input tensors based on variant
        if variant == "forward":
            # Forward: input (N, Hi, Wi, G, C/G), weight (K, Y, X, G, C/G)
            input_shape = (prob.N, prob.Hi, prob.Wi, prob.G, prob.C // prob.G)
            weight_shape = (prob.K, prob.Y, prob.X, prob.G, prob.C // prob.G)
        elif variant == "bwd_data":
            # Backward data: output_grad (N, Ho, Wo, G, K/G), weight (K, Y, X, G, C/G)
            Ho, Wo = compute_output_shape(prob)
            input_shape = (prob.N, Ho, Wo, prob.G, prob.K // prob.G)
            weight_shape = (prob.K, prob.Y, prob.X, prob.G, prob.C // prob.G)
        else:  # bwd_weight
            # Backward weight: input (N, Hi, Wi, G, C/G), output_grad (N, Ho, Wo, G, K/G)
            Ho, Wo = compute_output_shape(prob)
            input_shape = (prob.N, prob.Hi, prob.Wi, prob.G, prob.C // prob.G)
            weight_shape = (prob.N, Ho, Wo, prob.G, prob.K // prob.G)

        input_data = (np.random.randn(*input_shape) * 0.1).astype(np_dtype)
        weight_data = (np.random.randn(*weight_shape) * 0.1).astype(np_dtype)

        prob_str = (
            f"N={prob.N} C={prob.C} K={prob.K} G={prob.G} "
            f"H={prob.Hi} W={prob.Wi} Y={prob.Y} X={prob.X}"
        )
        print(f"\n  Problem [{prob_idx}]: {prob_name}")
        print(f"  {prob_str}")
        print(
            f"  {'Kernel':<55} {'Time(ms)':>10} {'TFLOPS':>10}"
            f" {'NonZero':>10} {'Status':>6}"
        )
        print(f"  {'-' * 95}")

        for config, lib_path in zip(all_configs, lib_paths):
            if lib_path is None:
                continue

            # Create runner from library path
            # GPU context is initialized lazily on first run() call
            from grouped_conv_utils import GpuGroupedConvRunner
            runner = GpuGroupedConvRunner(lib_path=str(lib_path))

            # First run() call triggers lazy GPU initialization
            result = runner.run(input_data, weight_data, prob)
            if not runner.is_available() or not result.success:
                continue
            if not result.success:
                continue

            non_zero = 0
            status = "OK"
            if result.output is not None:
                non_zero = int(np.count_nonzero(result.output))
                total = result.output.size
                status = "OK" if non_zero > 0 else "ZERO"
            else:
                status = "FAIL"

            print(
                f"  {config.name:<55} {result.time_ms:>10.3f}"
                f" {result.tflops:>10.2f} {non_zero:>10} {status:>6}"
            )

            all_results.append(
                {
                    "kernel": config.name,
                    "variant": config.variant,
                    "dtype": config.dtype,
                    "pipeline": config.pipeline,
                    "scheduler": config.scheduler,
                    "tile": config.tile_str,
                    "problem_name": prob_name,
                    "problem": {
                        "N": prob.N,
                        "C": prob.C,
                        "K": prob.K,
                        "G": prob.G,
                        "Hi": prob.Hi,
                        "Wi": prob.Wi,
                        "Y": prob.Y,
                        "X": prob.X,
                    },
                    "latency_ms": result.time_ms,
                    "tflops": result.tflops,
                    "non_zero": non_zero,
                }
            )

    bench_time = time.perf_counter() - bench_t0

    # Cleanup - no explicit cleanup needed for GroupedConvDispatcherLib

    # Report
    print(f"\n{'=' * 80}")
    print(f"  JIT:       {jit_time:.0f}s ({built} kernels)")
    print(f"  Benchmark: {bench_time:.1f}s")
    print(f"  Results:   {len(all_results)} measurements")

    if args.best and all_results:
        from collections import defaultdict

        by_problem = defaultdict(list)
        for r in all_results:
            key = json.dumps(r["problem"], sort_keys=True)
            by_problem[key].append(r)

        print("\n  Best kernel per problem:")
        for key, results in by_problem.items():
            best = max(results, key=lambda x: x["tflops"])
            prob = json.loads(key)
            print(
                f"    N={prob['N']} C={prob['C']} K={prob['K']} "
                f"H={prob['Hi']} W={prob['Wi']}"
                f" -> {best['kernel']} ({best['tflops']:.2f} TFLOPS)"
            )

    if args.csv:
        csv_path = Path(args.csv)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "kernel",
                    "variant",
                    "dtype",
                    "pipeline",
                    "scheduler",
                    "tile",
                    "problem_name",
                    "N",
                    "C",
                    "K",
                    "G",
                    "Hi",
                    "Wi",
                    "Y",
                    "X",
                    "latency_ms",
                    "tflops",
                    "non_zero",
                ],
            )
            writer.writeheader()
            for r in all_results:
                row = {**r, **r["problem"]}
                del row["problem"]
                writer.writerow(row)
        print(f"  CSV: {csv_path}")

    if args.json:
        json_path = Path(args.json)
        with open(json_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "arch": args.arch,
                        "variant": variant,
                        "num_kernels": len(all_configs),
                        "num_problems": len(problems),
                        "jit_time_s": jit_time,
                        "bench_time_s": bench_time,
                    },
                    "results": all_results,
                },
                f,
                indent=2,
            )
        print(f"  JSON: {json_path}")

    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
