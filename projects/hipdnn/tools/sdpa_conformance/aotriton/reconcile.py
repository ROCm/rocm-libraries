#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Offline reconciliation of a conformance run (login-node tool, needs no GPU or torch).

    reconcile.py <run-dir> [--out reconciled.csv]
    reconcile.py <run-dir> --engines <out>

``<run-dir>`` holds one ``shard-<i>of<N>/`` directory per shard, each with ``collected.txt`` (every
node id the shard collected) and ``results.csv`` (one row per finished case). The union over all
shards gets one row per node id: rows as recorded, and ``not-run`` for every collected id without
a row (killed job, exhausted walltime). A truncated trailing row (no newline, or the wrong number
of fields) is dropped, so its case becomes ``not-run`` too. CSVs written before the
``fwd_engines`` / ``bwd_engines`` columns existed read with those fields empty.

``--engines <out>`` takes one shard's output directory of a ``run_aotriton.py --all-engines`` run
and writes ``<out>/engines.csv`` (``nodeid,direction,engine,source,outcome,reason``): per case the
engine that served it under the heuristic (``source=heuristic``, ``<out>/results.csv``) and one
row per pinned engine from ``<out>/all-engines/<fwd|bwd>/<engine-slug>/`` (``source=pinned``), with
per-(direction, engine) outcome counts printed. A pinned row whose served engine is not the pinned
one is recorded as ``declined`` and never counts as that engine's pass.
"""

import argparse
import collections
import csv
import io
import sys
from pathlib import Path

LEGACY_FIELDS = ("nodeid", "outcome", "mode", "fwd", "bwd", "reason")
FIELDS = LEGACY_FIELDS + ("fwd_engines", "bwd_engines")
OUTCOMES = ("pass", "fail", "crash", "not-run", "declined", "native-skip", "no-hipdnn")
CSV_NAME = "results.csv"
COLLECTED_NAME = "collected.txt"
ENGINES_CSV_NAME = "engines.csv"
ENGINES_DIR = "all-engines"
ENGINE_NAME_FILE = (
    "engine.txt"  # the pinned engine's name, beside a sweep's results.csv
)
ENGINES_FIELDS = ("nodeid", "direction", "engine", "source", "outcome", "reason")
FORWARD_TESTS = "test_forward.py"


def direction_of(nodeid):
    """``'fwd'`` for a ``test_forward.py`` case, else ``'bwd'``: the direction an engine sweep covers."""
    return "fwd" if Path(nodeid.split("::")[0]).name == FORWARD_TESTS else "bwd"


def format_row(row):
    """One CSV line (with its newline); fields never contain line breaks."""
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerow(
        " ".join(str(row.get(f, "")).split("\n")).replace("\r", " ") for f in FIELDS
    )
    return out.getvalue()


def read_rows(path):
    """Rows of one shard's CSV; a malformed trailing line is dropped, any other raises.

    A CSV from before the engine-list columns (legacy header or no header, six fields) reads with
    those fields empty.
    """
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    rows = []
    for number, line in enumerate(lines, start=1):
        last = number == len(lines)
        fields = next(csv.reader([line])) if line.strip() else []
        well_formed = line.endswith("\n") and len(fields) in (
            len(LEGACY_FIELDS),
            len(FIELDS),
        )
        if number == 1 and fields in (list(FIELDS), list(LEGACY_FIELDS)):
            continue
        if not well_formed:
            if last:
                continue
            raise ValueError(f"{path}:{number}: malformed row {line!r}")
        rows.append({**dict.fromkeys(FIELDS, ""), **dict(zip(FIELDS, fields))})
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
                {
                    **dict.fromkeys(FIELDS, ""),
                    "nodeid": nodeid,
                    "outcome": "not-run",
                    "reason": "no row recorded",
                }
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


def served_engines(state):
    """Engine names in a ``fwd`` / ``bwd`` state text (``hipdnn:a+b``); empty for any other state."""
    kind, _, detail = state.partition(":")
    return set(detail.split("+")) - {""} if kind == "hipdnn" else set()


def engine_rows(out):
    """Rows of ``engines.csv`` for the all-engines run in ``out`` (one shard's directory)."""
    out = Path(out)
    rows = []
    for row in read_rows(out / CSV_NAME):
        direction = direction_of(row["nodeid"])
        for engine in sorted(served_engines(row[direction])):
            rows.append(
                (
                    row["nodeid"],
                    direction,
                    engine,
                    "heuristic",
                    row["outcome"],
                    row["reason"],
                )
            )
    for sweep in sorted((out / ENGINES_DIR).glob(f"*/*/{ENGINE_NAME_FILE}")):
        direction, pinned = sweep.parent.parent.name, sweep.read_text().strip()
        for row in read_rows(sweep.parent / CSV_NAME):
            outcome, reason = row["outcome"], row["reason"]
            served = served_engines(row[direction])
            if outcome in ("pass", "fail") and served != {pinned}:
                outcome = "declined"
                reason = f"served {'+'.join(sorted(served)) or 'no engine'}, not the pinned {pinned}"
            rows.append((row["nodeid"], direction, pinned, "pinned", outcome, reason))
    return [dict(zip(ENGINES_FIELDS, r)) for r in rows]


def engine_counts(rows):
    """Text: per (direction, engine) the pass / fail / declined counts (other outcomes appended)."""
    counts = collections.defaultdict(collections.Counter)
    for r in rows:
        counts[(r["direction"], r["engine"])][r["outcome"]] += 1
    lines = []
    for (direction, engine), c in sorted(counts.items()):
        extra = [
            f"{o}={n}"
            for o, n in sorted(c.items())
            if o not in ("pass", "fail", "declined")
        ]
        lines.append(
            f"{direction} {engine}: pass={c['pass']} fail={c['fail']} declined={c['declined']}"
            + "".join(f" {e}" for e in extra)
        )
    return "\n".join(lines)


def write_engines(out):
    """Write ``<out>/engines.csv``; returns the count text."""
    rows = engine_rows(out)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(ENGINES_FIELDS)
    for r in rows:
        writer.writerow(
            " ".join(str(r[f]).split("\n")).replace("\r", " ") for f in ENGINES_FIELDS
        )
    (Path(out) / ENGINES_CSV_NAME).write_text(buffer.getvalue())
    return engine_counts(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="write the reconciled CSV here")
    parser.add_argument(
        "--engines",
        type=Path,
        metavar="OUT",
        help="write OUT/engines.csv for the --all-engines run in OUT (a shard directory)",
    )
    args = parser.parse_args(argv)
    rows = reconcile(args.run_dir)
    if args.out:
        args.out.write_text(
            ",".join(FIELDS) + "\n" + "".join(format_row(r) for r in rows)
        )
    print(summarize(rows))
    if args.engines:
        print(write_engines(args.engines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
