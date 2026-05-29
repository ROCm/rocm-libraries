#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Interleaved multi-config ATOM benchmark.

Spawns bench_atom.py --single-shot once per (round, config) in round-robin
order so all three configs experience the same GPU power/thermal state within
each round.  Each invocation pays the full ATOM startup cost (~30s) but
produces a single unbiased measurement.

Usage:
  python bench_atom_sweep.py --model <path> [--tokenizer <path>] [--rounds 30]
      [--batch-size 2] [--input-len 512] [--output-len 200]
      [--kv-cache-dtype bf16] [--max-model-len 16384] [--level 3]

Output:
  Per-round raw lines, then a summary table with mean ± stdev for each config.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from pathlib import Path

CONFIGS = ["baseline", "dsl_gemm", "dsl_all"]
SCRIPT = Path(__file__).parent / "bench_atom.py"


def run_single_shot(
    python: str, args: argparse.Namespace, config: str
) -> tuple[float, float, float] | None:
    """Launch one bench_atom.py --single-shot subprocess and parse its output."""
    cmd = [
        python,
        str(SCRIPT),
        "--model",
        args.model,
        "--config",
        config,
        "--batch-size",
        str(args.batch_size),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--level",
        str(args.level),
        "--single-shot",
    ]
    if args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    if args.max_model_len:
        cmd += ["--max-model-len", str(args.max_model_len)]

    env = os.environ.copy()
    env["AITER_LOG_LEVEL"] = "WARNING"
    if args.ck_path:
        env["PYTHONPATH"] = args.ck_path + ":" + env.get("PYTHONPATH", "")
    if args.aiter_path:
        env["AITER_PATH"] = args.aiter_path

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in result.stdout.splitlines():
        if line.startswith("SINGLE_SHOT"):
            _, cfg, total_ms, step_us, throughput = line.split()
            assert cfg == config
            return float(total_ms), float(step_us), float(throughput)
    # Print stderr for debugging if parse failed
    print(f"  [WARN] no SINGLE_SHOT line from {config}:", file=sys.stderr)
    for line in result.stderr.splitlines()[-5:]:
        print(f"    {line}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--rounds",
        type=int,
        default=30,
        help="Number of rounds (one rep per config per round, default 30)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=200)
    parser.add_argument("--kv-cache-dtype", dest="kv_cache_dtype", default="bf16")
    parser.add_argument("--max-model-len", dest="max_model_len", type=int, default=None)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (default: current interpreter)",
    )
    parser.add_argument(
        "--ck-path",
        dest="ck_path",
        default=None,
        help="Path to prepend to PYTHONPATH (CK DSL root)",
    )
    parser.add_argument(
        "--aiter-path", dest="aiter_path", default=None, help="AITER_PATH env var value"
    )
    args = parser.parse_args()

    data: dict[str, list[float]] = {c: [] for c in CONFIGS}  # step_us per config

    print("=" * 72)
    print("ATOM interleaved benchmark sweep")
    print("=" * 72)
    print(f"  Rounds:    {args.rounds}  (one rep per config per round, round-robin)")
    print(f"  Configs:   {', '.join(CONFIGS)}")
    print(f"  Model:     {args.model}")
    print(f"  bs={args.batch_size}  in={args.input_len}tok  out={args.output_len}tok")
    print()
    print(
        f"  {'Rnd':>4}  {'Config':<12}  {'Total(ms)':>10}  {'Step(µs)':>10}  {'Thru(tok/s)':>12}"
    )
    print(f"  {'-' * 4}  {'-' * 12}  {'-' * 10}  {'-' * 10}  {'-' * 12}")

    for rnd in range(1, args.rounds + 1):
        for config in CONFIGS:
            result = run_single_shot(args.python, args, config)
            if result is None:
                print(f"  {rnd:>4}  {config:<12}  {'ERROR':>10}")
                continue
            total_ms, step_us, throughput = result
            data[config].append(step_us)
            print(
                f"  {rnd:>4}  {config:<12}  {total_ms:>10.1f}  {step_us:>10.1f}  {throughput:>12.1f}"
            )
        sys.stdout.flush()

    print()
    print("=" * 72)
    print(f"RESULTS (mean ± stdev, n={args.rounds} interleaved rounds)")
    print("=" * 72)
    baseline_mean = (
        statistics.mean(data["baseline"]) if data["baseline"] else float("nan")
    )
    for config in CONFIGS:
        vals = data[config]
        if len(vals) < 2:
            print(f"  {config:<12}  insufficient data")
            continue
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals)
        delta = ""
        if config != "baseline" and baseline_mean:
            pct = (mean - baseline_mean) / baseline_mean * 100
            delta = f"  ({pct:+.1f}% vs baseline)"
        print(f"  {config:<12}  {mean:.1f} ± {stdev:.1f} µs/step{delta}")
    print()


if __name__ == "__main__":
    main()
