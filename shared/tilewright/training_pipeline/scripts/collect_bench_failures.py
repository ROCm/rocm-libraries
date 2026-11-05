#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""List every hipblaslt-bench failure in stage03 block logs.

For each failure it reports the GEMM shape and the kernel in flight when the
failure was logged, so the crash can be reproduced from the shape and the
solution index alone, and it lists the blocks whose logs end problems early.

Usage:
  collect_bench_failures.py <stage03 logs dir> [more dirs ...] [--out CSV]

The CSV goes to --out, by default `bench_failures.csv` in the parent of the
first logs directory (the stage03 directory).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_THIS_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from lib.bench_log import (  # noqa: E402
    HEADER_RE,
    SKIP_RE,
    SOL_INDEX_RE,
    SOL_NAME_RE,
    iter_problems,
)

# Anything that means the GPU or the process died, as opposed to a slow kernel.
FATAL_RE = re.compile(
    r"(HSA_STATUS_ERROR[A-Z_]*"
    r"|Memory access fault"
    r"|hang analysis"
    r"|GPU core dump"
    r"|Segmentation fault"
    r"|terminate called"
    r"|rocblaslt error)",
    re.I,
)
# stage03's annotations. A crash annotation names the shape the recovery
# skipped (`M= N= K= B= transA= transB=`); an abort annotation marks a block
# given up on.
CRASH_RE = re.compile(
    r"\*\*\* SKIPPED CRASHED GEMM at block-offset (\d+) \(yaml line (\d+)\):"
    r"(.*?)\(rc=(-?\d+)\)"
)
_KV_RE = re.compile(r"(\w+)=(\S+)")
ABORT_RE = re.compile(r"\*\*\* ABORT: (.*?)\s*\*\*\*")

COLUMNS = [
    "block",
    "gpu",
    "m",
    "n",
    "k",
    "batch",
    "transA",
    "transB",
    "sol_idx",
    "sol_name",
    "error",
    "detail",
]


def _block_and_gpu(path: str) -> Tuple[str, str]:
    base = os.path.basename(path)
    m = re.search(r"_gpu(\d+)\.log$", base)
    gpu = m.group(1) if m else ""
    stem = re.sub(r"(_gpu\d+)?\.log$", "", base)
    return stem.split("_lines")[0], gpu


def _event(block: str, gpu: str, line_no: int, **kw) -> Dict[str, object]:
    ev: Dict[str, object] = {c: "" for c in COLUMNS}
    ev.update(block=block, gpu=gpu, line_no=line_no)
    ev.update(kw)
    return ev


def parse_log(path: str) -> Tuple[List[Dict[str, object]], int, int]:
    """(fatal events, complete problems, problems cut short) of one log."""
    block, gpu = _block_and_gpu(path)
    shape: Dict[str, object] = {}
    sol_idx = sol_name = ""
    pending: Optional[List[str]] = None
    events: List[Dict[str, object]] = []

    with open(path, errors="replace") as fh:
        lines = fh.readlines()

    for i, raw in enumerate(lines):
        s = raw.strip()
        if pending is not None:
            values = [v.strip() for v in s.split(",")]
            cols, pending = pending, None
            if len(values) == len(cols):
                row = dict(zip(cols, values))
                shape = {
                    "transA": row.get("transA", ""),
                    "transB": row.get("transB", ""),
                    "batch": row.get("batch_count", ""),
                    "m": row.get("m", ""),
                    "n": row.get("n", ""),
                    "k": row.get("k", ""),
                }
                continue
        hm = HEADER_RE.match(s)
        if hm:
            pending = [c.strip() for c in hm.group(2).split(",")]
            sol_idx = sol_name = ""
            continue
        mi = SOL_INDEX_RE.match(s)
        if mi:
            sol_idx = mi.group(1)
            continue
        mn = SOL_NAME_RE.match(s)
        if mn:
            sol_name = mn.group(1)
            continue
        ms = SKIP_RE.match(s)
        if ms:
            sol_idx = ms.group("index") or ""
            sol_name = "(skipped)"
            continue
        mc = CRASH_RE.search(s)
        if mc:
            kv = dict(_KV_RE.findall(mc.group(3)))
            events.append(
                _event(
                    block,
                    gpu,
                    i,
                    m=kv.get("M", ""),
                    n=kv.get("N", ""),
                    k=kv.get("K", ""),
                    batch=kv.get("B", ""),
                    transA=kv.get("transA", ""),
                    transB=kv.get("transB", ""),
                    error=f"crash_skipped(rc={mc.group(4)})",
                    detail=f"yaml line {mc.group(2)}, block-offset {mc.group(1)}",
                )
            )
            continue
        ma = ABORT_RE.search(s)
        if ma:
            events.append(
                _event(
                    block,
                    gpu,
                    i,
                    error="block_abandoned",
                    detail=ma.group(1),
                )
            )
            continue
        mf = FATAL_RE.search(s)
        if mf:
            last = events[-1] if events else None
            if (
                last is not None
                and int(last["line_no"]) > i - 25
                and last["m"] == shape.get("m", "")
                and last["sol_idx"] == sol_idx
            ):
                last["detail"] = f"{last['detail']} | {s[:120]}"
                continue
            events.append(
                _event(
                    block,
                    gpu,
                    i,
                    sol_idx=sol_idx,
                    sol_name=sol_name,
                    error=mf.group(1),
                    detail=s[:160],
                    **shape,
                )
            )

    problems = list(iter_problems(lines))
    n_complete = sum(1 for p in problems if p.complete)
    n_cut = sum(1 for p in problems if not p.finished)
    return events, n_complete, n_cut


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log_dirs", nargs="+", help="stage03 logs directories")
    ap.add_argument(
        "--out",
        default=None,
        help="CSV path (default: <parent of the first logs dir>/bench_failures.csv)",
    )
    args = ap.parse_args(argv)

    all_events: List[Dict[str, object]] = []
    block_rows: List[Tuple[str, int, int, int]] = []
    for d in args.log_dirs:
        for name in sorted(os.listdir(d)):
            if not name.startswith("block_") or not name.endswith(".log"):
                continue
            events, n_complete, n_cut = parse_log(os.path.join(d, name))
            all_events += events
            block_rows.append((name, n_complete, len(events), n_cut))

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.log_dirs[0])), "bench_failures.csv"
    )
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for e in all_events:
            w.writerow(e)

    print(f"fatal events : {len(all_events)}   -> {out}")
    print(
        f"blocks parsed: {len(block_rows)}   with problems cut short: "
        f"{sum(1 for r in block_rows if r[3])}"
    )
    if all_events:
        print("\n-- failures (shape / kernel / error) " + "-" * 40)
        for e in all_events:
            print(
                f"  {e['block']} gpu{e['gpu']}  {e['transA']}{e['transB']} "
                f"m={e['m']} n={e['n']} k={e['k']} b={e['batch']}  "
                f"sol={e['sol_idx']}  {e['error']}"
            )
            if e["sol_name"] and e["sol_name"] != "(skipped)":
                print(f"        kernel: {str(e['sol_name'])[:130]}")
        print("\n-- by solution index " + "-" * 56)
        for idx, c in Counter(e["sol_idx"] for e in all_events).most_common():
            print(f"  sol {idx or '(unknown)'}: {c}")
        print("\n-- by error " + "-" * 64)
        for err, c in Counter(e["error"] for e in all_events).most_common():
            print(f"  {err}: {c}")
    print("\n-- blocks with failures or problems cut short " + "-" * 31)
    for name, n_complete, n_events, n_cut in block_rows:
        if n_cut or n_events:
            print(
                f"  {name}  complete={n_complete}  fatal_events={n_events}  "
                f"cut_short={n_cut}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
