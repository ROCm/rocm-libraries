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
  Per-round raw lines, then a summary table with mean ± stdev for each config
  covering all five metrics: TTFT, decode step, TPOT, throughput, total time.
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

# Fields returned by SINGLE_SHOT: total_ms step_us tpot_ms ttft_ms throughput
FIELDS = ["total_ms", "step_us", "tpot_ms", "ttft_ms", "throughput"]


def run_single_shot(
    python: str, args: argparse.Namespace, config: str
) -> dict[str, float] | None:
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
        env["PYTHONPATH"] = args.aiter_path + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in result.stdout.splitlines():
        if line.startswith("SINGLE_SHOT"):
            parts = line.split()
            # format: SINGLE_SHOT <config> total_ms step_us tpot_ms ttft_ms throughput
            assert parts[1] == config
            return {f: float(v) for f, v in zip(FIELDS, parts[2:])}
    print(f"  [WARN] no SINGLE_SHOT line from {config}:", file=sys.stderr)
    for line in result.stderr.splitlines()[-5:]:
        print(f"    {line}", file=sys.stderr)
    return None


def _pct(mean: float, baseline: float) -> str:
    if baseline == 0:
        return ""
    return f"{(mean - baseline) / baseline * 100:+.1f}%"


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
        "--aiter-path",
        dest="aiter_path",
        default=None,
        help="AITER_PATH env var value",
    )
    args = parser.parse_args()

    # Accumulate all metrics per config
    data: dict[str, dict[str, list[float]]] = {
        c: {f: [] for f in FIELDS} for c in CONFIGS
    }

    print("=" * 80)
    print("ATOM interleaved benchmark sweep")
    print("=" * 80)
    print(f"  Rounds:  {args.rounds}  (round-robin: baseline → dsl_gemm → dsl_all)")
    print(f"  Model:   {args.model}")
    print(f"  bs={args.batch_size}  in={args.input_len}tok  out={args.output_len}tok")
    print()
    print(
        f"  {'Rnd':>4}  {'Config':<12}  {'Total(ms)':>10}  {'Step(µs)':>10}"
        f"  {'TPOT(ms)':>9}  {'TTFT(ms)':>9}  {'Thru(tok/s)':>12}"
    )
    print(
        f"  {'-' * 4}  {'-' * 12}  {'-' * 10}  {'-' * 10}"
        f"  {'-' * 9}  {'-' * 9}  {'-' * 12}"
    )

    for rnd in range(1, args.rounds + 1):
        for config in CONFIGS:
            r = run_single_shot(args.python, args, config)
            if r is None:
                print(f"  {rnd:>4}  {config:<12}  {'ERROR':>10}")
                continue
            for f in FIELDS:
                data[config][f].append(r[f])
            print(
                f"  {rnd:>4}  {config:<12}"
                f"  {r['total_ms']:>10.1f}  {r['step_us']:>10.1f}"
                f"  {r['tpot_ms']:>9.3f}  {r['ttft_ms']:>9.1f}  {r['throughput']:>12.1f}"
            )
        sys.stdout.flush()

    print()
    print("=" * 80)
    print(f"RESULTS — mean ± stdev  (n={args.rounds} interleaved rounds)")
    print("=" * 80)

    # Per-metric summary tables
    metrics = [
        (
            "Step (µs)",
            "step_us",
            ".1f",
            True,
        ),  # lower is better → negative % = improvement
        ("TTFT (ms)", "ttft_ms", ".1f", True),
        ("TPOT (ms)", "tpot_ms", ".3f", True),
        (
            "Throughput",
            "throughput",
            ".1f",
            False,
        ),  # higher is better → positive % = improvement
        ("Total (ms)", "total_ms", ".1f", True),
    ]

    baseline_means: dict[str, float] = {}
    for _, field, _, _ in metrics:
        vals = data["baseline"][field]
        baseline_means[field] = statistics.mean(vals) if vals else float("nan")

    for label, field, fmt, lower_better in metrics:
        print(f"\n  {label}")
        print(f"  {'Config':<14}  {'Mean':>10}  {'±Stdev':>8}  {'vs baseline':>12}")
        print(f"  {'-' * 14}  {'-' * 10}  {'-' * 8}  {'-' * 12}")
        bm = baseline_means[field]
        for config in CONFIGS:
            vals = data[config][field]
            if len(vals) < 2:
                print(f"  {config:<14}  insufficient data")
                continue
            mean = statistics.mean(vals)
            stdev = statistics.stdev(vals)
            if config == "baseline":
                delta = "—"
            else:
                pct = (mean - bm) / bm * 100 if bm else 0.0
                sign = "▼" if (pct < 0) == lower_better else "▲"
                delta = f"{sign} {abs(pct):.1f}%"
            print(f"  {config:<14}  {mean:{fmt}:>10}  ±{stdev:{fmt}:>7}  {delta:>12}")

    print()


if __name__ == "__main__":
    main()
