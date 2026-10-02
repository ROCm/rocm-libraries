#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run the degree sweep once per tile in tiles.py, into one CSV per tile.

Sequential on purpose: each measure_degrees.py run already saturates the box
with its own compile pool and then measures on the GPU, so overlapping two of
them would corrupt the timings that the whole corpus exists to collect.

Usage:
    python3 run_tiles.py shapes_fwd.txt --jobs 48 [--outdir tiles_out]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from tiles import TILES

HERE = Path(__file__).resolve().parent


def slug(spec):
    return spec.replace("/", "_").replace("x", "x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shapes")
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--outdir", default="tiles_out")
    ap.add_argument("--only", default=None, help="comma-separated tile specs")
    args = ap.parse_args()

    out = HERE / args.outdir
    out.mkdir(exist_ok=True)
    want = set(args.only.split(",")) if args.only else None

    for spec, src in TILES:
        if want and spec not in want:
            continue
        csv = out / f"{slug(spec)}.csv"
        log = out / f"{slug(spec)}.log"
        if csv.exists():
            print(f"[skip] {spec} — {csv.name} exists", flush=True)
            continue
        cmd = [
            sys.executable,
            str(HERE / "measure_degrees.py"),
            args.shapes,
            "--tile",
            spec,
            "-o",
            str(csv),
            "--jobs",
            str(args.jobs),
        ]
        print(f"[run ] {spec:<20} ({src})", flush=True)
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
        rows = sum(1 for _ in open(csv)) - 1 if csv.exists() else 0
        print(
            f"[done] {spec:<20} rc={rc} rows={rows} "
            f"{(time.time() - t0) / 60:.1f} min",
            flush=True,
        )


if __name__ == "__main__":
    main()
