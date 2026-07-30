#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Aggregate rocprofv3 counter CSVs from multiple passes into one summary table.

Hardware that cannot collect every counter in a single pass forces you to run
several rocprofv3 invocations and stitch the results together. This probe reads
all ``*counter_collection.csv`` files under ``--csv-dir`` (recursively),
aggregates by ``Counter_Name``, and prints mean / stddev / min / max per counter.

``--drop-cold`` discards the first value per counter, which is the cold dispatch
(module load, first-touch page faults) and is not representative.

    python probe_counter_aggregate.py --csv-dir /tmp/pass_out --drop-cold

Use it after any manual rocprof run, not just the ones driven by the other
probes in this directory. ``aggregate()`` is the programmatic entry point.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def aggregate(csv_dir: Path, drop_cold: bool = False) -> dict[str, list[float]]:
    """Collect every counter value found under ``csv_dir`` keyed by name."""
    agg: dict[str, list[float]] = defaultdict(list)
    csv_files = sorted(csv_dir.glob("**/*counter_collection.csv"))
    if not csv_files:
        return {}

    for csv_path in csv_files:
        try:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
        except OSError as e:
            print(f"WARNING: failed to read {csv_path}: {e}", file=sys.stderr)
            continue
        for r in rows:
            name, val = r.get("Counter_Name", ""), r.get("Counter_Value", "")
            if not name or not val:
                continue
            try:
                agg[name].append(float(val))
            except ValueError:
                pass

    if drop_cold:
        agg = {k: v[1:] for k, v in agg.items() if len(v) > 1}
    return dict(agg)


def summarize(agg: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Reduce raw per-dispatch values to mean / stddev / min / max / n."""
    out = {}
    for name, vals in sorted(agg.items()):
        if not vals:
            continue
        out[name] = {
            "mean": statistics.fmean(vals),
            "stddev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv-dir", required=True, type=Path)
    p.add_argument(
        "--drop-cold",
        action="store_true",
        help="discard the first value per counter (the cold dispatch)",
    )
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = p.parse_args(argv)

    if not args.csv_dir.is_dir():
        raise SystemExit(f"--csv-dir {args.csv_dir} is not a directory")

    stats = summarize(aggregate(args.csv_dir, drop_cold=args.drop_cold))
    if not stats:
        raise SystemExit(f"no counter_collection.csv rows found under {args.csv_dir}")

    if args.json:
        print(json.dumps(stats, indent=2))
        return 0

    print(f"{'Counter':<32}{'Mean':>12}{'StdDev':>12}{'Min':>12}{'Max':>12}{'N':>5}")
    print("-" * 85)
    for name, s in stats.items():
        print(
            f"{name:<32}{s['mean']:>12.2f}{s['stddev']:>12.2f}"
            f"{s['min']:>12.2f}{s['max']:>12.2f}{s['n']:>5}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
