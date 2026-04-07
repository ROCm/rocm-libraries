#!/usr/bin/env python3
"""Full grouped convolution benchmark sweep.

Architecture mirrors FMHA's fmha_full_benchmark.py:
  Phase 1: Compile all kernels (parallel, returns .so paths only)
  Phase 2: Benchmark via subprocess isolation (serial GPU access)

Each kernel runs in a subprocess to avoid Python ctypes library loading limits.
Subprocess batching (default 20) balances overhead vs fault isolation.

Usage:
    python grouped_conv_full_benchmark.py --workers 8 --csv results.csv
    python grouped_conv_full_benchmark.py --batch-size 20 --problems forward_training
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[2] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_THIS_DIR))

from grouped_conv_utils import setup_multiple_grouped_conv_dispatchers  # noqa: E402
from grouped_conv_instance_builder import expand_sweep  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Grouped Conv Benchmark Sweep")
    parser.add_argument("configs", nargs="+", help="Config JSON files")
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--problems", default="forward_training")
    parser.add_argument("--csv", type=str, default="grouped_conv_results.csv")
    parser.add_argument("--workers", type=int, default=8, help="Parallel build workers")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Kernels per subprocess (balance overhead vs fault isolation)",
    )
    parser.add_argument(
        "--kernel-timeout",
        type=int,
        default=30,
        help="Per-kernel timeout in seconds",
    )
    parser.add_argument(
        "--max-kernels",
        type=int,
        default=0,
        help="Limit to first N kernels (0=all)",
    )
    args = parser.parse_args()

    # ========================================================================
    # Phase 1: Compile kernels (parallel)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 1: Compile kernels")
    print(f"{'=' * 80}")

    all_configs = []
    for cfg_path in args.configs:
        all_configs.extend(expand_sweep(cfg_path, args.arch))

    if args.max_kernels > 0:
        all_configs = all_configs[:args.max_kernels]

    print(f"  Expanded configs: {len(all_configs)}")
    print(f"  Build workers: {args.workers}")

    t0 = time.perf_counter()
    # CRITICAL: This returns Path objects only, does NOT load .so files
    lib_paths = setup_multiple_grouped_conv_dispatchers(
        all_configs, verbose=True, max_workers=args.workers
    )
    build_time = time.perf_counter() - t0

    built_kernels = [
        (cfg, lib) for cfg, lib in zip(all_configs, lib_paths) if lib is not None
    ]
    print(f"\n  Built {len(built_kernels)}/{len(all_configs)} kernels in {build_time:.0f}s")

    if not built_kernels:
        print("  ERROR: No kernels built successfully")
        return 1

    # ========================================================================
    # Phase 2: Load problems
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 2: Load test problems")
    print(f"{'=' * 80}")

    sys.path.insert(0, str(_THIS_DIR / "problems"))
    # Load problem sets for training and validation
    if args.problems == "bwd_data_synthetic_extended":
        from bwd_data_synthetic_extended import TRAINING_PROBLEMS_BWD_DATA_SYNTHETIC as problems
    elif args.problems == "bwd_data_test_validation":
        from bwd_data_test_validation import VALIDATION_PROBLEMS_BWD_DATA as problems
    elif args.problems == "bwd_weight_synthetic_extended":
        from bwd_weight_synthetic_extended import TRAINING_PROBLEMS_BWD_WEIGHT_SYNTHETIC as problems
    elif args.problems == "bwd_weight_test_validation":
        from bwd_weight_test_validation import VALIDATION_PROBLEMS_BWD_WEIGHT as problems
    elif args.problems == "forward_training":
        from forward_training import TRAINING_PROBLEMS_FORWARD as problems
    elif args.problems == "forward_validation_model_crawler_100":
        from forward_validation_model_crawler_100 import VALIDATION_PROBLEMS_FORWARD_MODEL_CRAWLER_100 as problems
    else:
        raise ValueError(f"Unknown problem set: {args.problems}. Available: forward_training, forward_validation_model_crawler_100, bwd_data_synthetic_extended, bwd_data_test_validation, bwd_weight_synthetic_extended, bwd_weight_test_validation")

    print(f"  Problems: {len(problems)}")
    print(f"  Total measurements: {len(built_kernels)} x {len(problems)} = {len(built_kernels) * len(problems)}")

    # ========================================================================
    # Phase 3: Benchmark via subprocess (serial GPU, batched subprocess)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("Phase 3: Benchmark (subprocess isolation, batched)")
    print(f"{'=' * 80}")
    print(f"  Batch size: {args.batch_size} kernels per subprocess")
    print(f"  Timeout: {args.kernel_timeout}s per kernel")
    print()

    csv_path = Path(args.csv)
    csv_fields = [
        "kernel",
        "problem_idx",
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
        "latency_ms",
        "tflops",
        "non_zero",
    ]

    # Open CSV for writing
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    worker_path = _THIS_DIR / "run_one_grouped_conv_kernel.py"
    worker_env = os.environ.copy()
    # Worker needs both dispatcher/python (for dispatcher_common) and current dir (for grouped_conv_utils)
    worker_env["GCONV_PYPATH"] = os.pathsep.join([
        str(_DISPATCHER_ROOT / "python"),
        str(_THIS_DIR)
    ])

    total_measurements = 0
    total_failures = 0
    bench_t0 = time.perf_counter()

    for prob_idx, prob in enumerate(problems):
        print(f"\nProblem [{prob_idx + 1}/{len(problems)}]: N={prob.N} C={prob.C} K={prob.K} H={prob.Hi} W={prob.Wi}")
        print(f"  {'Kernel':<60} {'Time(ms)':>10} {'TFLOPS':>10} {'Status':>10}")
        print(f"  {'-' * 95}")

        # Convert problem to dict once
        prob_dict = {
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
            "direction": prob.direction,
        }

        # Process kernels in batches
        for batch_start in range(0, len(built_kernels), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(built_kernels))
            batch = built_kernels[batch_start:batch_end]

            # Build JSON payload for this batch
            items = []
            for cfg, lib_path in batch:
                items.append({
                    "so_path": str(lib_path),  # CRITICAL: Only pass string path, not loaded library
                    "problem": prob_dict,
                    "kernel_name": cfg.name,
                })

            payload = json.dumps({"items": items})

            # Run subprocess with batch
            try:
                proc = subprocess.Popen(
                    [sys.executable, str(worker_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    env=worker_env,
                )

                timeout_total = args.kernel_timeout * len(batch)
                stdout_bytes, _ = proc.communicate(
                    input=payload.encode("utf-8"), timeout=timeout_total
                )

                # Parse results (one JSON line per kernel)
                for line in stdout_bytes.decode("utf-8").strip().split("\n"):
                    if not line:
                        continue

                    try:
                        result = json.loads(line)
                        batch_idx = result.get("idx", 0)
                        cfg, lib_path = batch[batch_idx]

                        if result.get("ok", False):
                            status = "OK" if result.get("non_zero", 0) > 0 else "ZERO"
                            print(
                                f"  {cfg.name:<60} {result['ms']:>10.3f} {result['tflops']:>10.2f} {status:>10}"
                            )

                            writer.writerow({
                                "kernel": cfg.name,
                                "problem_idx": prob_idx,
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
                                "latency_ms": result["ms"],
                                "tflops": result["tflops"],
                                "non_zero": result.get("non_zero", 0),
                            })
                            csv_file.flush()
                            total_measurements += 1
                        else:
                            error_msg = result.get("error", "unknown")
                            # Show full error for debugging (first 100 chars)
                            print(f"  {cfg.name:<60} FAILED")
                            print(f"    Error: {error_msg[:100]}")
                            total_failures += 1

                    except json.JSONDecodeError:
                        print(f"  Warning: Could not parse result line: {line[:50]}")
                        total_failures += 1

            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                print(f"  Batch timeout ({len(batch)} kernels)")
                total_failures += len(batch)

            except Exception as e:
                print(f"  Batch error: {e}")
                total_failures += len(batch)

    bench_time = time.perf_counter() - bench_t0
    csv_file.close()

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Build time: {build_time:.0f}s")
    print(f"  Benchmark time: {bench_time:.0f}s")
    print(f"  Total time: {build_time + bench_time:.0f}s")
    print(f"  Successful measurements: {total_measurements}")
    print(f"  Failed measurements: {total_failures}")
    print(f"  Output: {csv_path}")


if __name__ == "__main__":
    main()
