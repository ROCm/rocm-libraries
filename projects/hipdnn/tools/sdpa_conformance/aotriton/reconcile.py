#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Offline reconciliation of a conformance run (login-node tool, needs no GPU or torch).

    reconcile.py <run-dir> [--out reconciled.csv]

``<run-dir>`` holds one ``shard-<i>of<N>/`` directory per shard, each with ``collected.txt`` (every
node id the shard collected) and ``results.csv`` (one row per finished case). The union over all
shards gets one row per node id: rows as recorded, and ``not-run`` for every collected id without
a row (killed job, exhausted walltime). A truncated trailing row (no newline, or the wrong number
of fields) is dropped, so its case becomes ``not-run`` too.
"""

import argparse
import collections
import csv
import io
import sys
from pathlib import Path

FIELDS = ("nodeid", "outcome", "mode", "fwd", "bwd", "reason")
OUTCOMES = ("pass", "fail", "crash", "not-run", "declined", "native-skip", "no-hipdnn")
CSV_NAME = "results.csv"
COLLECTED_NAME = "collected.txt"


def format_row(row):
    """One CSV line (with its newline); fields never contain line breaks."""
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerow(
        " ".join(str(row.get(f, "")).split("\n")).replace("\r", " ") for f in FIELDS
    )
    return out.getvalue()


def read_rows(path):
    """Rows of one shard's CSV; a malformed trailing line is dropped, any other raises."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    rows = []
    for number, line in enumerate(lines, start=1):
        last = number == len(lines)
        fields = next(csv.reader([line])) if line.strip() else []
        well_formed = line.endswith("\n") and len(fields) == len(FIELDS)
        if number == 1 and fields == list(FIELDS):
            continue
        if not well_formed:
            if last:
                continue
            raise ValueError(f"{path}:{number}: malformed row {line!r}")
        rows.append(dict(zip(FIELDS, fields)))
    return rows


def reconcile(run_dir):
    """Union of every shard's rows plus ``not-run`` rows; returns a list of row dicts."""
    seen, rows, collected = set(), [], []
    for shard in sorted(Path(run_dir).glob("shard-*")):
        for row in read_rows(shard / CSV_NAME):
            if row["nodeid"] not in seen:
                seen.add(row["nodeid"])
                rows.append(row)
        listing = shard / COLLECTED_NAME
        if listing.exists():
            collected += [ln for ln in listing.read_text().splitlines() if ln.strip()]
    for nodeid in dict.fromkeys(collected):
        if nodeid not in seen:
            seen.add(nodeid)
            rows.append(
                dict(zip(FIELDS, (nodeid, "not-run", "", "", "", "no row recorded")))
            )
    return rows


def summarize(rows):
    """Text summary: outcome totals, outcomes by reason, failing node ids with their reason."""
    by_outcome = collections.Counter(r["outcome"] for r in rows)
    lines = [f"{len(rows)} cases"]
    lines += [f"  {o:12} {by_outcome[o]}" for o in OUTCOMES if by_outcome[o]]
    reasons = collections.Counter(
        (r["outcome"], r["reason"])
        for r in rows
        if r["outcome"] not in ("pass", "fail", "no-hipdnn")
    )
    if reasons:
        lines.append("by reason:")
        lines += [f"  {n:7} {o}: {why}" for (o, why), n in reasons.most_common()]
    modes = collections.Counter(r["mode"] for r in rows if r["outcome"] == "pass")
    if modes:
        lines.append(
            "passes by mode: "
            + ", ".join(f"{m or 'unrecorded'}={n}" for m, n in modes.items())
        )
    failing = [r for r in rows if r["outcome"] in ("fail", "crash")]
    if failing:
        lines.append("failing:")
        lines += [f"  {r['outcome']} {r['nodeid']}  {r['reason']}" for r in failing]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="write the reconciled CSV here")
    args = parser.parse_args(argv)
    rows = reconcile(args.run_dir)
    if args.out:
        args.out.write_text(
            ",".join(FIELDS) + "\n" + "".join(format_row(r) for r in rows)
        )
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
