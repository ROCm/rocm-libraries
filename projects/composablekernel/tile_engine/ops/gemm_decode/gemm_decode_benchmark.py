# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Sweep driver for gemm_decode benchmarks.

P0 stub. Discovers all `benchmark_gemm_decode_*` executables under a build
directory and runs each across a small problem-size grid; the full sweep
matrix and CSV emission land in P1+ together with the codegen.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


_DEFAULT_PROBLEMS: List[Tuple[int, int, int]] = [
    (1, 8192, 7168),
    (1, 4096, 7168),
    (8, 8192, 7168),
]


def _discover(build_dir: Path) -> List[Path]:
    bin_dir = build_dir / "bin"
    if not bin_dir.is_dir():
        return []
    return sorted(p for p in bin_dir.iterdir()
                  if p.is_file() and p.name.startswith("benchmark_gemm_decode_"))


def _run(exe: Path, m: int, n: int, k: int, split_k: int) -> str:
    cmd = [
        str(exe),
        f"-m={m}",
        f"-n={n}",
        f"-k={k}",
        f"-split_k={split_k}",
        "-warmup=20",
        "-repeat=50",
        "-metric=2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return res.stdout.strip() if res.returncode == 0 else f"FAILED: {res.stderr.strip()}"


def main() -> int:
    parser = argparse.ArgumentParser(description="gemm_decode sweep driver (P0 stub)")
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--csv", default=None, type=Path)
    parser.add_argument("--split-k", default=[1, 2, 4], type=int, nargs="+")
    args = parser.parse_args()

    exes = _discover(args.build_dir)
    if not exes:
        print(f"no benchmark_gemm_decode_* executables under {args.build_dir}/bin/", file=sys.stderr)
        return 1

    rows = []
    for exe in exes:
        for (m, n, k) in _DEFAULT_PROBLEMS:
            for sk in args.split_k:
                line = _run(exe, m, n, k, sk)
                print(f"{exe.name} M={m} N={n} K={k} split_k={sk}: {line}")
                rows.append({"kernel": exe.name, "M": m, "N": n, "K": k,
                             "split_k": sk, "output": line})

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
