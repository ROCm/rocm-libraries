#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check picks against a library's kernel guards.

A model trained with kernel guards is served with the guards sidecar of its
library (`<stem>.tilewright.guards`, lib/correctness.py): TensileLite offers a
guarded kernel only for problems whose free sizes reach its guards. A ranking
that reads no sidecar (the analytical ranking hipBLASLt uses without
tilewright) can still pick such a kernel below its guard. This script reads
picks and lists every pick of a guarded kernel on a problem its guards deny,
so that a caller can leave those problems out of a run before it starts
(predicted picks) or stop on them after it (the solutions a bench ran).

Usage:
  check_guarded_picks.py --guards FILE [--any-free-size]
      (--picks TSV [TSV ...] | --bench-log LOG [LOG ...])
      [--denied-out JSON]

--picks reads tab-separated files whose header line (a leading `#` is
dropped) names the columns m, n, k, b (or batch) and the picked kernel's
solution name (the first of served_solution_name, solution_name,
served_kernel_name and kernel_name that it has), one pick per row.
--bench-log reads hipblaslt-bench logs (lib/bench_log.py), the solution
each problem ran. Names match as the blocklist and TensileLite match them
(`blocklist_key`). With --any-free-size a guard denies a problem whose M or
N is below its minimum, whichever free size it reads: a conservative rule
for rankings that do not know which operand a kernel reads past.
--denied-out writes the denied picks as JSON (m, n, k, batch, name,
guards). Exit code 0 when no pick is denied, 1 when some are (each
listed), 2 on bad input.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_THIS_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from lib import correctness as cx  # noqa: E402

NAME_COLUMNS: Tuple[str, ...] = (
    "served_solution_name",
    "solution_name",
    "served_kernel_name",
    "kernel_name",
)


@dataclass(frozen=True)
class Pick:
    m: int
    n: int
    k: int
    batch: int
    name: str
    source: str = ""


def strict_guards(guards: Sequence[cx.Guard]) -> Tuple[cx.Guard, ...]:
    """`guards` with each minimum applied to both free sizes (the largest
    minimum of the kernel's guards on M and on N)."""
    low = max((g.minimum for g in guards), default=0)
    return (cx.Guard("M", low), cx.Guard("N", low)) if low else ()


def denied_picks(
    picks: Iterable[Pick],
    guards: Mapping[str, Sequence[cx.Guard]],
    any_free_size: bool = False,
) -> List[Tuple[Pick, Tuple[cx.Guard, ...]]]:
    """(pick, the guards that apply) of every pick of a kernel `guards`
    ({kernel name: guards}) guards whose guards deny its problem; with
    `any_free_size` each guard reads both free sizes (`strict_guards`)."""
    by_key = {cx.blocklist_key(n): tuple(g) for n, g in guards.items()}
    out = []
    for p in picks:
        found = by_key.get(cx.blocklist_key(p.name))
        if not found:
            continue
        applied = strict_guards(found) if any_free_size else found
        if not cx.guards_admit(applied, p.m, p.n):
            out.append((p, applied))
    return out


def read_pick_table(path: Path) -> List[Pick]:
    """The picks of a tab-separated file (see the module docstring).
    ValueError naming the file when its header lacks a column."""
    lines = Path(path).read_text().splitlines()
    if not lines:
        return []
    header = lines[0].lstrip("#").split("\t")
    rows = csv.DictReader(lines[1:], fieldnames=header, delimiter="\t")
    batch = "b" if "b" in header else "batch"
    name = next((c for c in NAME_COLUMNS if c in header), None)
    missing = [c for c in ("m", "n", "k", batch) if c not in header]
    if name is None or missing:
        raise ValueError(
            f"{path}: its header names no "
            + (" or ".join(missing) if missing else " or ".join(NAME_COLUMNS))
        )
    out = []
    for i, r in enumerate(rows, 2):
        try:
            out.append(
                Pick(
                    int(r["m"]),
                    int(r["n"]),
                    int(r["k"]),
                    int(r[batch]),
                    str(r[name] or ""),
                    f"{path}:{i}",
                )
            )
        except (TypeError, ValueError):
            raise ValueError(f"{path}:{i}: not a pick row") from None
    return out


def read_bench_picks(path: Path) -> List[Pick]:
    """The solution each complete problem of a hipblaslt-bench log ran (its
    first tested row)."""
    from lib.bench_log import parse_log_file

    out = []
    for prob in parse_log_file(path):
        shape = prob.shape()
        if shape is None or not prob.tested:
            continue
        m, n, k, b = shape
        out.append(Pick(m, n, k, b, prob.tested[0].sol_name, str(path)))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--guards", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--picks", type=Path, nargs="+")
    src.add_argument("--bench-log", type=Path, nargs="+")
    ap.add_argument("--any-free-size", action="store_true")
    ap.add_argument("--denied-out", type=Path)
    ap.add_argument("--show", type=int, default=20, help="denied picks to list")
    args = ap.parse_args(argv)
    try:
        guards = cx.read_guards(args.guards)
        picks: List[Pick] = []
        for path in args.picks or []:
            picks += read_pick_table(path)
        for path in args.bench_log or []:
            picks += read_bench_picks(path)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    denied = denied_picks(picks, guards, args.any_free_size)
    rule = "M or N" if args.any_free_size else "the free size each reads"
    print(f"guards: {len(guards)} kernel(s), {rule} below the minimum denies")
    print(f"picks: {len(picks)}; of a guarded kernel: ", end="")
    keys = {cx.blocklist_key(n) for n in guards}
    print(sum(1 for p in picks if cx.blocklist_key(p.name) in keys))
    print(f"denied: {len(denied)}")
    for p, applied in denied[: args.show]:
        text = " and ".join(g.text() for g in applied)
        print(f"  m={p.m} n={p.n} k={p.k} b={p.batch}: {p.name} ({text}) {p.source}")
    if args.denied_out is not None:
        doc = [
            {
                "m": p.m,
                "n": p.n,
                "k": p.k,
                "batch": p.batch,
                "name": p.name,
                "guards": [g.to_dict() for g in applied],
                "source": p.source,
            }
            for p, applied in denied
        ]
        args.denied_out.write_text(json.dumps(doc, indent=1) + "\n")
    return 1 if denied else 0


if __name__ == "__main__":
    sys.exit(main())
