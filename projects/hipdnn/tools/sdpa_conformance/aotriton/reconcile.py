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
and writes ``<out>/engines.csv`` (``nodeid,direction,engine,source,outcome,reason,pin_fwd,pin_bwd,
kernel,stage``): per case the engine that served it under the heuristic (``source=heuristic``,
``<out>/results.csv``; for a ``crash`` row the engine in flight), one row per case and pass of every
pass directory under ``<out>/all-engines/`` (``pinned``, ``pair``, ``refwd``, ``discovery``), and a
``not-run`` row for every node id a pass listed (``nodeids.txt``, ``collected.txt`` for pass 1) but
never recorded. ``capture/`` directories are never read. A pinned row whose served engine is not
the pinned one is recorded as ``declined`` and never counts as that engine's pass. The command
prints outcome counts per direction and engine, per pass expected against recorded, and every
engine of ``<out>/engine-inventory.txt`` with its counts (zeros included) and, for an engine no case
reached, its first decline reasons from ``<out>/declines/``.
"""

import argparse
import collections
import csv
import io
import json
import re
import sys
from pathlib import Path

import conformance_contract as contract
from conformance_contract import ENGINES_EXTRA_FIELDS, RESULTS_EXTRA_FIELDS

LEGACY_FIELDS = ("nodeid", "outcome", "mode", "fwd", "bwd", "reason")
ENGINE_LIST_FIELDS = LEGACY_FIELDS + ("fwd_engines", "bwd_engines")
FIELDS = ENGINE_LIST_FIELDS + RESULTS_EXTRA_FIELDS
OUTCOMES = ("pass", "fail", "crash", "not-run", "declined", "native-skip", "no-hipdnn")
CSV_NAME = "results.csv"
COLLECTED_NAME = "collected.txt"
ENGINES_CSV_NAME = "engines.csv"
ENGINES_DIR = "all-engines"
ENGINE_NAME_FILE = (
    "engine.txt"  # the pinned engine's name, beside a sweep's results.csv
)
ENGINES_FIELDS = (
    "nodeid",
    "direction",
    "engine",
    "source",
    "outcome",
    "reason",
) + ENGINES_EXTRA_FIELDS
PASS_META_NAME = "pass.json"
NODEIDS_NAME = "nodeids.txt"
DECLINES_DIR = "declines"  # the job stages the workers' decline files here
MAIN_PASS = "main"
HEURISTIC = "heuristic"
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

    A CSV from before the engine-list columns (six fields) or before the kernel/side/repro
    columns (eight fields) reads with the missing fields empty.
    """
    known = {len(LEGACY_FIELDS), len(ENGINE_LIST_FIELDS), len(FIELDS)}
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    rows = []
    for number, line in enumerate(lines, start=1):
        last = number == len(lines)
        fields = next(csv.reader([line])) if line.strip() else []
        well_formed = line.endswith("\n") and len(fields) in known
        if number == 1 and fields in (
            list(FIELDS),
            list(ENGINE_LIST_FIELDS),
            list(LEGACY_FIELDS),
        ):
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


def pass_slug(out, pass_dir):
    """Name of a pass for the in-flight directory and repro case names: ``main`` for pass 1,
    else its path under ``all-engines/`` joined with ``-`` (``pair-A__X``, ``refwd-discovery``).
    """
    parts = Path(pass_dir).relative_to(out).parts
    if parts[:1] == (ENGINES_DIR,):
        parts = parts[1:]
    return "-".join(parts) or MAIN_PASS


def pass_dirs(out):
    """``[(directory, meta)]`` of every pass under ``<out>/all-engines``; ``capture/`` directories never.

    A pass directory holds ``pass.json`` (``kind`` sweep|pair|refwd|discovery, ``direction``,
    ``pin_fwd``, ``pin_bwd``). A directory with only ``engine.txt`` (a sweep of a run before
    ``pass.json``) reads as a sweep pinning that engine in the direction named by its parent.
    """
    root = Path(out) / ENGINES_DIR
    found = {}
    for engine_file in root.glob(f"*/*/{ENGINE_NAME_FILE}"):
        d = engine_file.parent
        direction = d.parent.name
        found[d] = {
            "kind": "sweep",
            "direction": direction,
            f"pin_{direction}": engine_file.read_text().strip(),
        }
    for meta_file in root.glob(f"**/{PASS_META_NAME}"):
        found[meta_file.parent] = json.loads(meta_file.read_text())
    return sorted(found.items())


def _listing(path):
    path = Path(path)
    if not path.exists():
        return []
    return list(dict.fromkeys(ln for ln in path.read_text().splitlines() if ln.strip()))


def _passes(out):
    """``(directory, meta, expected node ids)`` of pass 1 and every pass under ``all-engines``."""
    out = Path(out)
    yield out, {"kind": HEURISTIC}, _listing(out / COLLECTED_NAME)
    for d, meta in pass_dirs(out):
        yield d, meta, _listing(d / NODEIDS_NAME)


_SOURCE = {"sweep": "pinned"}


def _pass_engine_rows(d, meta, expected):
    """Engine rows of one pass: one per served engine and case, ``not-run`` per unreached case."""
    kind = meta["kind"]
    pin = {"fwd": meta.get("pin_fwd", ""), "bwd": meta.get("pin_bwd", "")}
    rows, recorded = [], set()

    def add(nodeid, direction, engine, outcome, reason, row=None):
        row = row or dict.fromkeys(FIELDS, "")
        rows.append(
            {
                "nodeid": nodeid,
                "direction": direction,
                "engine": engine,
                "source": _SOURCE.get(kind, kind),
                "outcome": outcome,
                "reason": reason,
                "pin_fwd": pin["fwd"],
                "pin_bwd": pin["bwd"],
                "kernel": row[f"{direction}_kernel"],
                "stage": row["stage"],
            }
        )

    def mismatch(row, wanted):
        """``wanted`` is ``{direction: engine}``; the declined reason when a served engine differs."""
        wrong = [
            f"served {'+'.join(sorted(served_engines(row[d]))) or 'no engine'}, not the pinned {e}"
            for d, e in wanted.items()
            if served_engines(row[d]) != {e}
        ]
        return "; ".join(wrong)

    for row in read_rows(d / CSV_NAME):
        nodeid, outcome, reason = row["nodeid"], row["outcome"], row["reason"]
        recorded.add(nodeid)
        crashed = row["fail_side"] if outcome == "crash" else ""
        failed = row["fail_side"] if outcome == "fail" else ""
        if kind == HEURISTIC:
            direction = crashed or direction_of(nodeid)
            for engine in sorted(served_engines(row[direction])):
                add(nodeid, direction, engine, outcome, reason, row)
        elif kind == "discovery":
            for engine in sorted(served_engines(row["bwd"])):
                add(nodeid, "bwd", engine, outcome, reason, row)
        elif kind == "sweep":
            direction = meta["direction"]
            if crashed and crashed != direction:
                for engine in sorted(served_engines(row[crashed])):
                    add(nodeid, crashed, engine, outcome, reason, row)
                continue
            if outcome in ("pass", "fail"):
                wrong = mismatch(row, {direction: pin[direction]})
                if wrong:
                    outcome, reason = "declined", wrong
            add(nodeid, direction, pin[direction], outcome, reason, row)
        else:  # pair, refwd: the backward is pinned; a pair also pins the forward
            wanted = {"bwd": pin["bwd"]}
            if kind == "pair":
                wanted["fwd"] = pin["fwd"]
            if outcome in ("pass", "fail"):
                wrong = mismatch(row, wanted)
                if wrong:
                    outcome, reason = "declined", wrong
            side = crashed or failed or "bwd"
            add(nodeid, side, wanted.get(side, "reference"), outcome, reason, row)
    direction = (
        meta.get("direction")
        if kind == "sweep"
        else "bwd" if kind != HEURISTIC else None
    )
    for nodeid in expected:
        if nodeid not in recorded:
            engine = pin[direction] if kind in ("sweep", "pair", "refwd") else ""
            add(
                nodeid,
                direction or direction_of(nodeid),
                engine,
                "not-run",
                "no row recorded",
            )
    return rows


def engine_rows(out):
    """Rows of ``engines.csv`` for the all-engines run in ``out`` (one shard's directory)."""
    return [
        row
        for d, meta, expected in _passes(out)
        for row in _pass_engine_rows(d, meta, expected)
    ]


def _count_label(r):
    if r["source"] in ("heuristic", "pinned"):
        return f"{r['direction']} {r['engine']}"
    label = f"{r['source']} {r['direction']} {r['engine']}"
    return label + (f" (fwd {r['pin_fwd']})" if r["source"] == "pair" else "")


def engine_counts(rows):
    """Text: per (direction, engine) the pass / fail / declined counts (other outcomes appended)."""
    counts = collections.defaultdict(collections.Counter)
    for r in rows:
        counts[_count_label(r)][r["outcome"]] += 1
    lines = []
    for label, c in sorted(counts.items()):
        extra = [
            f"{o}={n}"
            for o, n in sorted(c.items())
            if o not in ("pass", "fail", "declined")
        ]
        lines.append(
            f"{label}: pass={c['pass']} fail={c['fail']} declined={c['declined']}"
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


def pass_accounting(out):
    """Text, one line per pass: node ids expected against rows recorded."""
    lines = []
    for d, _, expected in _passes(out):
        recorded = {r["nodeid"] for r in read_rows(d / CSV_NAME)}
        missing = len([n for n in expected if n not in recorded])
        name = (
            MAIN_PASS
            if d == Path(out)
            else d.relative_to(Path(out) / ENGINES_DIR).as_posix()
        )
        lines.append(
            f"{name}: expected {len(expected)}, recorded {len(recorded & set(expected))}"
            + (f" ({missing} not-run)" if missing else "")
        )
    return "\n".join(lines)


def _decline_lines(out):
    """``{engine as the backend logged it: [distinct decline lines, in file order]}``."""
    found = {}
    for path in sorted((Path(out) / DECLINES_DIR).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            lines = found.setdefault(entry["engine"], [])
            if entry["line"] not in lines:
                lines.append(entry["line"])
    return found


def _same_engine(inventory_name, logged):
    """The backend logs an ingestor engine by name and an ASM builder by class name."""
    a, b = inventory_name.lower(), logged.lower()
    squash = lambda t: re.sub(r"[^a-z0-9]", "", t)
    return (
        a == b
        or a.endswith(":" + b)
        or squash(b) in squash(a)
        or squash(a) in squash(b)
    )


def inventory_summary(out, rows, reasons=3):
    """Text: every engine of ``<out>/engine-inventory.txt`` with its outcome counts, zeros included,
    and for an engine no case reached the first decline reasons logged under ``<out>/declines/``.
    """
    path = Path(out) / contract.INVENTORY_NAME
    if not path.exists():
        return ""
    declines = _decline_lines(out)
    lines = []
    for name in _listing(path):
        counts = collections.Counter(r["outcome"] for r in rows if r["engine"] == name)
        lines.append(
            f"{name}: "
            + " ".join(
                f"{o}={counts[o]}"
                for o in ("pass", "fail", "crash", "declined", "not-run")
            )
        )
        if not counts:
            logged = [
                t for e, ts in declines.items() if _same_engine(name, e) for t in ts
            ]
            lines += [f"    decline: {t}" for t in logged[:reasons]] or [
                "    no applicable case and no decline reason logged"
            ]
    return "\n".join(lines)


_PID = re.compile(r"\(pid (\d+)\)$")
_KILL = re.compile(r"watchdog: sent \w+ to pid (\d+)")
_STACK = re.compile(
    r"--- pytest_gpu_lease\.watchdog: stack of pid (\d+) on GPU .*? ---\n.*?--- end of stack ---",
    re.S,
)


def mark_hangs(pass_dir, out, slug):
    """Rewrite the ``crash`` rows of ``pass_dir`` whose worker pid the lease watchdog signalled
    (``<pass_dir>/watchdog.log``) to reason ``hang``; the relayed stack goes to the case's repro
    directory as ``watchdog-stack.txt``. Returns the number of rows changed (idempotent).
    """
    pass_dir = Path(pass_dir)
    log = pass_dir / "watchdog.log"
    if not log.exists():
        return 0
    text = log.read_text()
    killed = set(_KILL.findall(text))
    stacks = {m.group(1): m.group(0) for m in _STACK.finditer(text)}
    rows, changed = read_rows(pass_dir / CSV_NAME), 0
    for row in rows:
        match = (
            _PID.search(row["reason"])
            if row["outcome"] == "crash" and not row["reason"].startswith("hang")
            else None
        )
        if not match or match.group(1) not in killed:
            continue
        pid = match.group(1)
        row["reason"] = f"hang: killed by the lease watchdog (pid {pid})"
        changed += 1
        if pid in stacks:
            case = (
                Path(out)
                / contract.REPRO_DIR
                / contract.repro_case(row["nodeid"], slug)
            )
            case.mkdir(parents=True, exist_ok=True)
            (case / "watchdog-stack.txt").write_text(stacks[pid] + "\n")
    if changed:
        temp = pass_dir / (CSV_NAME + ".tmp")
        temp.write_text(",".join(FIELDS) + "\n" + "".join(format_row(r) for r in rows))
        temp.replace(pass_dir / CSV_NAME)
    return changed


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
        print(pass_accounting(args.engines))
        print(inventory_summary(args.engines, engine_rows(args.engines)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
