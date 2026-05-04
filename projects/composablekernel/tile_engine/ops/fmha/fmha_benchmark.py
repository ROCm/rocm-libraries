#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA tile engine benchmark runner.

Uses the dispatcher's setup_multiple_fmha_dispatchers() for pipelined JIT
compilation, then runs GPU benchmarks and reports results.

Usage:
    python fmha_benchmark.py configs/fwd.json
    python fmha_benchmark.py configs/receipt0_fwd.json --workers 256 --build-dir /tmp/fmha_build
    python fmha_benchmark.py configs/fwd.json --problems "2,8,1024,128" --verify
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

_DISPATCHER_ROOT = Path(__file__).resolve().parents[3] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))

from fmha_utils import (  # noqa: E402
    FmhaProblem,
    FmhaRunner,
    cpu_attention_fwd,
    detect_gpu_arch,
    setup_multiple_fmha_dispatchers,
)

from fmha_instance_builder import expand_sweep, apply_filter  # noqa: E402


def parse_problems(spec: str) -> List[FmhaProblem]:
    """Parse problem specs: 'batch,nhead,seqlen,hdim;...'"""
    problems = []
    for part in spec.split(";"):
        vals = [int(x) for x in part.split(",")]
        if len(vals) == 4:
            b, h, s, d = vals
            problems.append(
                FmhaProblem(
                    batch=b,
                    nhead_q=h,
                    nhead_k=h,
                    seqlen_q=s,
                    seqlen_k=s,
                    hdim_q=d,
                    hdim_v=d,
                )
            )
        elif len(vals) == 6:
            b, hq, hk, sq, sk, d = vals
            problems.append(
                FmhaProblem(
                    batch=b,
                    nhead_q=hq,
                    nhead_k=hk,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    hdim_q=d,
                    hdim_v=d,
                )
            )
    return problems


def main():
    parser = argparse.ArgumentParser(description="FMHA Tile Engine Benchmark")
    parser.add_argument("configs", nargs="+", help="Sweep config JSON(s)")
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument("--workers", type=int, default=8, help="Parallel JIT workers")
    parser.add_argument(
        "--problems",
        default="2,8,1024,128",
        help="Problem sizes: batch,nhead,seqlen,hdim",
    )
    parser.add_argument("--receipt", type=int, default=0)
    parser.add_argument(
        "--verify", action="store_true", help="Verify against CPU reference"
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
        help='Python expr per config, e.g. "c.hdim_q == 128"',
    )
    parser.add_argument(
        "--filter-file", default="", help="Path to .py with filter_config(c) -> bool"
    )
    args = parser.parse_args()

    problems = parse_problems(args.problems)
    build_dir = Path(args.build_dir).resolve()

    if args.clean and build_dir.exists():
        print(f"  Cleaning {build_dir} ...")
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0: Expand configs
    all_configs = []
    for cfg_path in args.configs:
        configs = expand_sweep(cfg_path, args.arch, args.receipt)
        all_configs.extend(configs)
        print(f"  {cfg_path}: {len(configs)} kernel configs")

    if args.filter_expr or args.filter_file:
        before = len(all_configs)
        all_configs = apply_filter(all_configs, args.filter_expr, args.filter_file)
        print(f"  Filter: {before} -> {len(all_configs)} configs")

    # Remove standalone combine configs -- they are auto-paired during JIT
    all_configs = [c for c in all_configs if c.family != "fwd_splitkv_combine"]

    print(f"\n{'=' * 70}")
    print("FMHA Tile Engine Benchmark")
    print(f"{'=' * 70}")
    print(f"  Arch:     {args.arch}")
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

    setups = setup_multiple_fmha_dispatchers(
        all_configs,
        output_dir=build_dir,
        verbose=True,
        max_workers=args.workers,
    )

    jit_time = time.perf_counter() - jit_t0
    built = sum(1 for s in setups if s.success)
    failed = len(all_configs) - built
    print(f"\n  Built {built}/{len(all_configs)} in {jit_time:.0f}s ({failed} failed)")

    # Load runners for successfully compiled kernels
    for setup in setups:
        if setup.success and setup.library_path and setup.runner is None:
            try:
                setup.runner = FmhaRunner.from_library(setup.library_path, args.arch)
            except Exception as e:
                print(f"  Warning: Failed to load runner: {e}")
                setup.success = False
                
    if args.compile_only:
        print(f"\n{'=' * 70}")
        print(f"  Compile-only mode. {built} kernels ready.")
        print(f"{'=' * 70}")
        return

    # Phase 2: Benchmark
    print(f"\n--- Phase 2: Benchmark ({built} kernels x {len(problems)} problems) ---")

    dtype_map = {
        "fp16": np.float16,
        "bf16": np.float32,
        "fp32": np.float32,
        "fp8bf16": np.float16,
        "fp8fp32": np.float16,
        "bf8": np.float16,
    }
    np.random.seed(42)
    all_results = []
    bench_t0 = time.perf_counter()

    for prob_idx, prob in enumerate(problems):
        first_dtype = all_configs[0].data_type if all_configs else "fp16"
        first_mask = all_configs[0].mask if all_configs else "no"
        np_dtype = dtype_map.get(first_dtype, np.float16)
        Q = (np.random.randn(*prob.q_shape()) * 0.1).astype(np_dtype)
        K = (np.random.randn(*prob.k_shape()) * 0.1).astype(np_dtype)
        V = (np.random.randn(*prob.v_shape()) * 0.1).astype(np_dtype)

        _MASK_INT = {"no": 0, "top_left": 1, "bottom_right": 2, "generic": 3}
        first_mask_int = _MASK_INT.get(first_mask, 0)

        ref = None
        if args.verify:
            ref = cpu_attention_fwd(
                Q.astype(np.float32),
                K.astype(np.float32),
                V.astype(np.float32),
                prob.scale,
                mask_type=first_mask_int,
            )

        prob_str = f"B={prob.batch} H={prob.nhead_q} S={prob.seqlen_q} D={prob.hdim_q}"
        print(f"\n  Problem [{prob_idx}]: {prob_str}")
        print(
            f"  {'Kernel':<50} {'Time(ms)':>10} {'TFLOPS':>10}"
            f" {'MaxErr':>10} {'Status':>6}"
        )
        print(f"  {'-' * 90}")

        _BIAS_INT = {"no": 0, "bias": 1, "alibi": 2}

        for config, setup in zip(all_configs, setups):
            if not setup.success or setup.runner is None:
                continue

            # Skip kernels whose hdim doesn't match the problem
            if config.hdim_q != prob.hdim_q or config.hdim_v != prob.hdim_v:
                continue

            mask_int = _MASK_INT.get(config.mask, 0)
            # Causal masks need window_right=0 (no future tokens visible)
            is_causal = config.mask in ("top_left", "bottom_right")
            is_group = config.mode == "group"

            # Map instance-builder family to runner api_family
            _FAMILY_TO_API = {
                "fwd_splitkv": "splitkv",
                "fwd_pagedkv": "pagedkv",
                "fwd_appendkv": "appendkv",
            }
            api_family = _FAMILY_TO_API.get(config.family, config.family)

            result = setup.runner.run(
                Q, K, V, prob,
                mask_type=mask_int,
                bias_type=_BIAS_INT.get(config.bias, 0),
                has_lse=int(config.lse),
                has_dropout=int(config.dropout),
                has_logits=int(config.logits),
                has_sink=int(config.sink),
                data_type=config.data_type,
                is_group_mode=int(is_group),
                is_v_rowmajor=int(config.vlayout == "r"),
                api_family=api_family,
                window_left=-1,
                window_right=0 if is_causal else -1,
            )
            if not result.success:
                continue

            # Adjust TFLOPS for causal mask (~half the ops)
            tflops = result.tflops
            if is_causal and result.time_ms > 0:
                sq, sk = prob.seqlen_q, prob.seqlen_k
                causal_ratio = (min(sq, sk) + 1) / (2.0 * sk)
                tflops = prob.num_ops * causal_ratio / (result.time_ms * 1e-3) / 1e12

            max_err = 0.0
            status = "OK"
            if ref is not None and result.output is not None:
                max_err = float(np.abs(result.output.astype(np.float32) - ref).max())
                status = "PASS" if max_err < 0.01 else "FAIL"

            print(
                f"  {config.name:<50} {result.time_ms:>10.3f}"
                f" {tflops:>10.2f} {max_err:>10.2e} {status:>6}"
            )

            all_results.append(
                {
                    "kernel": config.name,
                    "dtype": config.data_type,
                    "hdim": config.hdim_q,
                    "pipeline": config.pipeline,
                    "problem": {
                        "batch": prob.batch,
                        "nhead_q": prob.nhead_q,
                        "seqlen_q": prob.seqlen_q,
                        "hdim_q": prob.hdim_q,
                    },
                    "latency_ms": result.time_ms,
                    "tflops": tflops,
                    "max_err": max_err,
                }
            )

    bench_time = time.perf_counter() - bench_t0

    # Cleanup
    for setup in setups:
        if setup.success and setup.runner:
            try:
                setup.runner.cleanup()
            except Exception:
                pass

    # Report
    print(f"\n{'=' * 70}")
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
                f"    B={prob['batch']} H={prob['nhead_q']}"
                f" S={prob['seqlen_q']} D={prob['hdim_q']}"
                f" -> {best['kernel']} ({best['tflops']:.2f} TFLOPS)"
            )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "kernel",
                    "dtype",
                    "hdim",
                    "pipeline",
                    "batch",
                    "nhead_q",
                    "seqlen_q",
                    "hdim_q",
                    "latency_ms",
                    "tflops",
                    "max_err",
                ],
            )
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
                "jit_time_s": jit_time,
                "bench_time_s": bench_time,
                "num_kernels": len(all_configs),
                "num_built": built,
                "num_problems": len(problems),
            },
            "results": all_results,
        }
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  JSON: {args.json}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
