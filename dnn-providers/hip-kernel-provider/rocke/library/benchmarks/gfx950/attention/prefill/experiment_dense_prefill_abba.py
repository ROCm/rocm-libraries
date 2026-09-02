#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ABBA paired benchmark for gfx950 dense prefill (Llama-3-8B exact shape).

Runs repeated A-B-B-A rounds of baseline vs candidate source trees on one GPU,
recording per-sample latency, TFLOPS, parity error, and median ratios.

Usage (from rocke/library with PYTHONPATH set)::

    python benchmarks/gfx950/attention/prefill/experiment_dense_prefill_abba.py \\
        --baseline-root /ossci-storage/spur/$USER/src/rocke-baseline \\
        --candidate-root /ossci-storage/spur/$USER/src/rocke \\
        --rounds 5 --warmup 20 --iters 50 \\
        --output-json /ossci-storage/spur/$USER/results/exp_abba.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

import torch  # noqa: E402

_DEFAULT_SHAPE = Path(_HERE) / "llama3_8b_dense_prefill_shape.json"


def _git_info(root: Path) -> dict:
    def _run(*args: str) -> str:
        try:
            return (
                subprocess.check_output(
                    ["git", "-C", str(root), *args],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "root": str(root),
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _run("describe", "--always", "--dirty"),
    }


def _device_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "gcnArchName": props.gcnArchName,
        "multi_processor_count": props.multi_processor_count,
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
    }


def _median_ratio(candidate: list[float], baseline: list[float]) -> float:
    if len(candidate) != len(baseline) or not candidate:
        return float("nan")
    ratios = [c / b for c, b in zip(candidate, baseline) if b > 0]
    return statistics.median(ratios) if ratios else float("nan")


def _run_side(
    label: str,
    root: Path,
    shape: dict,
    *,
    warmup: int,
    iters: int,
    seed: int,
    check: bool,
) -> dict:
    """Benchmark using the rocke tree at ``root`` (directory containing library/)."""
    rocke = root
    if not (rocke / "library").is_dir():
        raise FileNotFoundError(f"expected rocke tree at {rocke} (missing library/)")

    lib = rocke / "library"
    plat = rocke / "platform" / "python"
    # Fresh import path for each staged tree.
    for p in (str(lib), str(plat)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import importlib

    import builders.gfx950.attention.prefill.attention_dense_prefill as adp

    importlib.reload(adp)

    spec = adp.make_spec_from_shape(shape)
    t0 = time.time()
    result = adp.run_benchmark(
        spec, warmup=warmup, iters=iters, seed=seed, check=check
    )
    result["label"] = label
    result["elapsed_s"] = round(time.time() - t0, 2)
    result["git"] = _git_info(root)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape-json", type=Path, default=_DEFAULT_SHAPE)
    ap.add_argument(
        "--baseline-shape-json",
        type=Path,
        default=None,
        help="shape for baseline tree (default: --shape-json)",
    )
    ap.add_argument(
        "--candidate-shape-json",
        type=Path,
        default=None,
        help="shape for candidate tree (default: --shape-json)",
    )
    ap.add_argument("--baseline-root", type=Path, required=True)
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=5, help="ABBA round count")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1

    shape_default = json.loads(args.shape_json.read_text())
    baseline_shape = json.loads(
        (args.baseline_shape_json or args.shape_json).read_text()
    )
    candidate_shape = json.loads(
        (args.candidate_shape_json or args.shape_json).read_text()
    )
    check = not args.no_check
    rounds = max(1, args.rounds)

    report: dict = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shape": shape_default,
        "baseline_shape": baseline_shape,
        "candidate_shape": candidate_shape,
        "device": _device_info(),
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "rounds": rounds,
        "samples": [],
    }

    baseline_tflops: list[float] = []
    candidate_tflops: list[float] = []

    for r in range(rounds):
        for label, root, bucket, side_shape in (
            ("A_baseline", args.baseline_root, baseline_tflops, baseline_shape),
            ("B_candidate", args.candidate_root, candidate_tflops, candidate_shape),
            ("B_candidate", args.candidate_root, candidate_tflops, candidate_shape),
            ("A_baseline", args.baseline_root, baseline_tflops, baseline_shape),
        ):
            print(f"\n=== round {r + 1}/{rounds} {label} ===", flush=True)
            sample = _run_side(
                label,
                root.resolve(),
                side_shape,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
                check=check,
            )
            report["samples"].append(sample)
            bucket.append(sample["tflops"])
            print(
                f"  {sample['ms']:.4f} ms  {sample['tflops']:.1f} TFLOPS  "
                f"max_abs={sample['max_abs']:.2e}  ok={sample['ok']}",
                flush=True,
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    report["summary"] = {
        "baseline_tflops_median": statistics.median(baseline_tflops),
        "candidate_tflops_median": statistics.median(candidate_tflops),
        "median_ratio_candidate_over_baseline": _median_ratio(
            candidate_tflops, baseline_tflops
        ),
        "target_ratio": 1.20,
        "meets_20pct": _median_ratio(candidate_tflops, baseline_tflops) >= 1.20,
        "all_ok": all(s["ok"] for s in report["samples"]),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.output_json}")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["all_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
