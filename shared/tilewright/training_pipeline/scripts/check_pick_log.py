#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check a bench log's tilewright picks against the pipeline's kernel pool.

With `TILEWRIGHT_PICK_LOG=1` hipBLASLt prints a `[TILEWRIGHT_PICK]` line per
ranked problem whose `top1_index` is a position in the pool it ranked, and
hipblaslt-bench then prints the solution it ran (`--Solution index: N`). This
script builds the pool the pipeline uses (`lib/bank_pool.py`) and reports
every problem whose pool position does not name the solution that ran, and
every `n_configs` that differs from the pool size.

Usage:
  check_pick_log.py LOG [LOG ...] --library-dir DIR --library-stem STEM
      (--config-yaml CFG [--hardware-device N]
       | --processor gfx950 --cu-count 256 [--chip-id 0x75a0]
         [--bank-stem STEM ...])

With --config-yaml the bank is the one the config asks for on the device;
otherwise it is taken from --bank-stem files, or every file of the problem
type. Exit code 0 when every pick matches, 1 on a mismatch, 2 on bad input.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_THIS_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

import yaml  # noqa: E402

from lib import bank_pool  # noqa: E402

_PICK_RE = re.compile(
    r"\[TILEWRIGHT_PICK\]\s+m=(\d+)\s+n=(\d+)\s+k=(\d+)\s+b=(\d+)\s.*?"
    r"top1_index=(\d+)\s+n_configs=(\d+)"
)
_SOLUTION_RE = re.compile(r"^\s*--Solution index:\s*(-?\d+)")


@dataclass(frozen=True)
class Pick:
    shape: Tuple[int, int, int, int]
    top1_index: int
    n_configs: int
    solution: int


@dataclass
class PickCheck:
    picks: List[Pick] = field(default_factory=list)
    without_pick: int = 0
    n_configs: List[int] = field(default_factory=list)
    mismatches: List[Tuple[Pick, Optional[int]]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches


def parse_picks(lines: Iterable[str]) -> Tuple[List[Pick], int]:
    """(picks paired with the solution index printed after them, number of
    solution indices without a pick line before them)."""
    picks: List[Pick] = []
    pending: Optional[re.Match] = None
    without = 0
    for line in lines:
        m = _PICK_RE.search(line)
        if m:
            pending = m
            continue
        s = _SOLUTION_RE.match(line)
        if s is None:
            continue
        if pending is None:
            without += 1
            continue
        g = [int(x) for x in pending.groups()]
        picks.append(Pick(tuple(g[:4]), g[4], g[5], int(s.group(1))))
        pending = None
    return picks, without


def check_picks(lines: Iterable[str], pool: Sequence[int]) -> PickCheck:
    """Map every pick's `top1_index` through `pool` (solution indices in pool
    order)."""
    picks, without = parse_picks(lines)
    out = PickCheck(picks=picks, without_pick=without)
    out.n_configs = sorted({p.n_configs for p in picks})
    for p in picks:
        named = pool[p.top1_index] if p.top1_index < len(pool) else None
        if named != p.solution:
            out.mismatches.append((p, named))
    return out


def _pool(args: argparse.Namespace) -> List[int]:
    if args.config_yaml is not None:
        cfg = yaml.safe_load(args.config_yaml.read_text()) or {}
        return bank_pool.config_pool(
            cfg, args.library_dir, args.library_stem, args.hardware_device
        ).indices
    device = bank_pool.BankDevice(
        processor=args.processor, cu_count=args.cu_count, chip_id=args.chip_id
    )
    return bank_pool.library_pool(
        args.library_dir, args.library_stem, device, args.bank_stem or None
    ).indices


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--library-dir", type=Path, required=True)
    ap.add_argument("--library-stem", required=True)
    ap.add_argument("--config-yaml", type=Path)
    ap.add_argument("--hardware-device", type=int, default=0)
    ap.add_argument("--processor")
    ap.add_argument("--cu-count", type=int)
    ap.add_argument("--chip-id", type=lambda v: int(v, 0))
    ap.add_argument("--bank-stem", action="append", default=[])
    ap.add_argument("--show", type=int, default=20, help="mismatches to list")
    args = ap.parse_args(argv)
    if args.config_yaml is None and (args.processor is None or args.cu_count is None):
        ap.error("give --config-yaml, or --processor and --cu-count")
    try:
        pool = _pool(args)
        lines: List[str] = []
        for log in args.logs:
            lines += log.read_text(errors="replace").splitlines()
    except (OSError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    result = check_picks(lines, pool)
    print(f"pool: {len(pool)} kernels")
    print(f"picks: {len(result.picks)} with a solution index after them")
    print(f"solution indices without a pick line: {result.without_pick}")
    sizes = ", ".join(str(n) for n in result.n_configs) or "-"
    print(f"n_configs in the log: {sizes}")
    if any(n != len(pool) for n in result.n_configs):
        print("n_configs differs from the pool size")
    print(f"mismatches: {len(result.mismatches)}")
    for p, named in result.mismatches[: args.show]:
        m, n, k, b = p.shape
        print(
            f"  m={m} n={n} k={k} b={b} top1_index={p.top1_index}: pool names "
            f"{named}, the bench ran {p.solution}"
        )
    return 0 if result.ok and all(n == len(pool) for n in result.n_configs) else 1


if __name__ == "__main__":
    sys.exit(main())
