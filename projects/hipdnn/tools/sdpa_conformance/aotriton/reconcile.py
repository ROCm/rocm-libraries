#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Offline reconciliation of a conformance run (login-node tool, needs no GPU or torch).

    reconcile.py <run-dir> [--out reconciled.csv]
    reconcile.py <run-dir> --report [--shards-file FILE] [--baseline FILE] [--out reconciled.csv]
    reconcile.py <run-dir> --engines <out>
    reconcile.py <run-dir> --compare <other-run-dir>

``<run-dir>`` holds one ``<suite>-<k>of<N>/`` directory per shard (``default-3of16``, ``forward-1of4``;
``shard-<k>of<N>`` too), each with ``collected.txt`` (every
node id the shard collected) and ``results.csv`` (one row per finished case). A run directory holds
each node id once: the union keeps the first row of an id recorded by two shard directories (``--report``
lists such ids under ``== duplicate node ids``). The union gets one row per node id: rows as recorded,
and ``not-run`` for every collected id without
a row (killed job, exhausted walltime). A truncated trailing row (no newline, or the wrong number
of fields) is dropped, so its case becomes ``not-run`` too. A ``results.csv`` written by an earlier
version (before the ``fwd_engines`` / ``bwd_engines`` columns, the kernel/side/repro columns or the
candidate columns) still reads, with the missing fields empty.

``--report`` reads the whole run: shard accounting (``--shards-file`` lists the expected shards,
``<suite> <k> <N>`` per line, so a shard that left no directory shows as missing), outcome totals (with
the ``native-skip`` rows by class: ``kernel-limit``, ``api-rule``, ``resource``, ``other``), per-
pass and per-(direction, engine) counts over every shard (each engine credited from the ``fwd`` and
``bwd`` columns of every passing row in pass 1 and both pinned engines of a passing pair-pass row;
the pinned and reference-forward passes credit the engine they pin, the discovery pass the backward
engine that served), then split by mode and by test file), the engines no case passed
with their first decline reasons, every ``fail`` / ``crash`` row with engine (the failing side's, else
the other side's), side, stage and repro path, and with ``--baseline FILE`` (one failing node id per
line) the failing node ids new to this run and the baseline ids that no longer fail here (outcome
``pass`` only); every other baseline id (``declined``, ``native-skip``, ``no-hipdnn``, ``not-run`` or
not in the run) is listed as not confirmed fixed with its outcome.

``--compare <other-run-dir>`` compares the reconciled rows of ``<other-run-dir>`` (the earlier run) with
``<run-dir>`` case by case: every outcome transition with its count and example node ids, and the
cases whose served engine changed with the same outcome. Exit status 2 if a run directory is missing.

``--engines <out>`` takes one shard's output directory of a ``run_aotriton.py --all-engines`` run
and writes ``<out>/engines.csv`` (``nodeid,direction,engine,source,outcome,reason,pin_fwd,pin_bwd,
kernel,stage,fwd_candidate,bwd_candidate``): per case the engine that served it under the heuristic (``source=heuristic``,
``<out>/results.csv``; for a ``crash`` row the engine in flight), one row per case and pass of every
pass directory under ``<out>/all-engines/`` (``pinned``, ``pair``, ``refwd``, ``discovery``), and a
``not-run`` row for every node id a pass listed (``nodeids.txt``, ``collected.txt`` for pass 1) but
never recorded. ``capture/`` directories are never read. A pinned row whose served engine is not
the pinned one is recorded as ``declined`` and never counts as that engine's pass; a ``fail`` row
whose failing side served its pinned engine stays a ``fail`` against that side, and only a side that
ran is compared with its pin. A state ``error:<engine>`` (a hipDNN call that raised) names the engine
like ``hipdnn:<engine>``. The command
prints outcome counts per direction and engine, per pass expected against recorded, and every
engine of ``<out>/engine-inventory.txt`` with its counts (zeros included) and, for an engine no case
reached, its first decline reasons from ``<out>/declines/``.

``--engines <out>`` also reads the candidate passes of ``run_aotriton.py --exhaustive`` under
``<out>/cand/`` (``candidate`` and ``candidate-pair`` passes; ``engines.csv`` gains
``fwd_candidate``, ``bwd_candidate``; a case an engine cannot enumerate is a ``not-enumerable``
row with the message). ``results.csv`` gains the same two columns after ``repro``.
"""

import argparse
import collections
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import conformance_contract as contract
from conformance_contract import (
    CANDIDATE_FIELDS,
    ENGINES_EXTRA_FIELDS,
    RESULTS_EXTRA_FIELDS,
)

LEGACY_FIELDS = ("nodeid", "outcome", "mode", "fwd", "bwd", "reason")
ENGINE_LIST_FIELDS = LEGACY_FIELDS + ("fwd_engines", "bwd_engines")
PRE_CANDIDATE_FIELDS = ENGINE_LIST_FIELDS + RESULTS_EXTRA_FIELDS
FIELDS = PRE_CANDIDATE_FIELDS + CANDIDATE_FIELDS
OUTCOMES = (
    "pass",
    "fail",
    "crash",
    "not-run",
    "declined",
    "native-skip",
    "no-hipdnn",
    contract.NOT_ENUMERABLE,
)
CSV_NAME = "results.csv"
COLLECTED_NAME = "collected.txt"
ENGINES_CSV_NAME = "engines.csv"
ENGINES_DIR = "all-engines"
ENGINE_NAME_FILE = (
    "engine.txt"  # the pinned engine's name, beside a sweep's results.csv
)
ENGINES_FIELDS = (
    (
        "nodeid",
        "direction",
        "engine",
        "source",
        "outcome",
        "reason",
    )
    + ENGINES_EXTRA_FIELDS
    + CANDIDATE_FIELDS
)
PASS_META_NAME = "pass.json"
NODEIDS_NAME = "nodeids.txt"
DECLINES_DIR = "declines"  # the job stages the workers' decline files here
MAIN_PASS = "main"
HEURISTIC = "heuristic"
SHARD_DIR = re.compile(r"^([a-z]+)-(\d+)of(\d+)$")

SKIP_CLASSES = ("kernel-limit", "api-rule", "resource", "other")
# First match wins; built from the ``pytest.skip`` messages under vendor/ (a native-skip's reason).
_SKIP_PATTERNS = (
    # "Skip large UT with bias to avoid OOM", "Skip large tests (...) due to low VRAM.",
    # "[Adiffs] Skip due to known OOM."
    ("resource", re.compile(r"\bOOM\b|VRAM|out of memory", re.I)),
    # "..._scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True"
    ("api-rule", re.compile(r"attn_mask should not be set|is_causal", re.I)),
    # "Skip hdim > 256 on gfx11 arch due to register pressure.", "hdim < 64 AITER ASM kernel does not
    # exist.", "Bias is not supported in AITER ASM backend", "GQA is not exposed in AITER ASM backend."
    (
        "kernel-limit",
        re.compile(r"\bhdim\b|register pressure|AITER|kernel does not exist", re.I),
    ),
)


def skip_class(reason):
    """``kernel-limit`` (an AOTriton kernel limit), ``api-rule`` (an API rule), ``resource`` (a low-VRAM
    or OOM guard) or ``other``, for the reason of a ``native-skip`` row."""
    return next((c for c, p in _SKIP_PATTERNS if p.search(reason)), "other")


def direction_of(nodeid):
    """``'fwd'`` for a ``test_forward.py`` case, else ``'bwd'``: the direction an engine sweep covers."""
    return (
        "fwd" if Path(nodeid.split("::")[0]).name == contract.FORWARD_TESTS else "bwd"
    )


def format_row(row):
    """One CSV line (with its newline); fields never contain line breaks."""
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerow(
        " ".join(str(row.get(f, "")).split("\n")).replace("\r", " ") for f in FIELDS
    )
    return out.getvalue()


def read_rows(path):
    """Rows of one shard's CSV; a malformed trailing line is dropped, any other raises.

    Reads the ``results.csv`` of run directories written by earlier versions of ``run_aotriton.py``
    (six fields: before the engine-list columns; eight: before the kernel/side/repro columns;
    thirteen: before the candidate columns), so those runs still reconcile; missing fields read empty.
    """
    known = {
        len(LEGACY_FIELDS),
        len(ENGINE_LIST_FIELDS),
        len(PRE_CANDIDATE_FIELDS),
        len(FIELDS),
    }
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
            list(PRE_CANDIDATE_FIELDS),
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


def open_rows_for_append(path):
    """``path`` (a results CSV) opened for appending rows, its parent created. A last line without
    its newline, which a writer killed mid-row leaves and :func:`read_rows` drops, is cut first, so
    a resumed run never glues its first row onto it; a file left empty gets the header.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            with open(path, "r+b") as f:
                f.truncate(data.rfind(b"\n") + 1)
    empty = not path.exists() or not path.stat().st_size
    f = open(path, "a")
    if empty:
        f.write(",".join(FIELDS) + "\n")
        f.flush()
    return f


def shard_dirs(run_dir):
    """The shard directories of a run (``<suite>-<k>of<N>``), ordered by suite, then ``k`` numerically."""
    found = []
    for d in Path(run_dir).iterdir():
        match = SHARD_DIR.match(d.name)
        if match and d.is_dir():
            found.append(((match[1], int(match[2]), int(match[3])), d))
    return [d for _, d in sorted(found)]


def read_shards_file(path):
    """Shard directory names of a ``<suite> <k> <N>`` file (``#`` comments and blank lines are skipped)."""
    names = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and not fields[0].startswith("#"):
            suite, k, n = fields[:3]
            names.append(f"{suite}-{k}of{n}")
    return names


def _union(run_dir):
    """``(rows, duplicates)``: the rows of :func:`reconcile` and ``{node id: [shard directory names]}``
    of every id recorded by more than one shard directory (the first row wins)."""
    seen, rows, collected, where = set(), [], [], {}
    shards = shard_dirs(run_dir)
    if not shards:
        raise FileNotFoundError(
            f"no shard directories (<suite>-<k>of<N>) under {run_dir}"
        )
    for shard in shards:
        for row in read_rows(shard / CSV_NAME):
            owners = where.setdefault(row["nodeid"], [])
            if shard.name not in owners:
                owners.append(shard.name)
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
    return rows, {n: dirs for n, dirs in where.items() if len(dirs) > 1}


def reconcile(run_dir):
    """Union of every shard's rows plus ``not-run`` rows; returns a list of row dicts.

    A run directory holds each node id once; an id recorded by two shard directories keeps its first
    row (shards in :func:`shard_dirs` order). ``run_report`` lists those ids.
    """
    return _union(run_dir)[0]


def summarize(rows, failing=True):
    """Text summary: outcome totals, outcomes by reason, failing node ids with their reason
    (unless ``failing`` is false)."""
    by_outcome = collections.Counter(r["outcome"] for r in rows)
    lines = [f"{len(rows)} cases"]
    lines += [f"  {o:12} {by_outcome[o]}" for o in OUTCOMES if by_outcome[o]]
    skips = collections.Counter(
        skip_class(r["reason"]) for r in rows if r["outcome"] == "native-skip"
    )
    if skips:
        lines.append("native-skip by class:")
        lines += [f"  {c:12} {skips[c]}" for c in SKIP_CLASSES if skips[c]]
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
    failed = [r for r in rows if r["outcome"] in ("fail", "crash")] if failing else []
    if failed:
        lines.append("failing:")
        lines += [f"  {r['outcome']} {r['nodeid']}  {r['reason']}" for r in failed]
    return "\n".join(lines)


def served_engines(state):
    """Engine names in a ``fwd`` / ``bwd`` state text (``hipdnn:a+b``, or ``error:a`` for a hipDNN call
    that raised); empty for any other state."""
    kind, _, detail = state.partition(":")
    return set(detail.split("+")) - {""} if kind in ("hipdnn", "error") else set()


def pass_slug(out, pass_dir):
    """Name of a pass for the in-flight directory and repro case names: ``main`` for pass 1,
    else its path under ``all-engines/`` joined with ``-`` (``pair-A__X``, ``refwd-discovery``);
    a candidate pass keeps its ``cand`` prefix (``cand-fwd-<engine>-0``).
    """
    parts = Path(pass_dir).relative_to(out).parts
    if parts[:1] == (ENGINES_DIR,):
        parts = parts[1:]
    return "-".join(parts) or MAIN_PASS


def pass_dirs(out):
    """``[(directory, meta)]`` of every pass under ``<out>/all-engines`` and ``<out>/cand``;
    ``capture/`` directories never.

    A pass directory holds ``pass.json`` (``kind`` sweep|pair|refwd|discovery|candidate|
    candidate-pair, ``direction``, ``pin_fwd``, ``pin_bwd``, ``fwd_candidate``, ``bwd_candidate``).
    A directory with only ``engine.txt`` (a sweep written before ``pass.json`` existed, still on disk
    in older run directories) reads as a sweep pinning that engine in the direction named by its parent.
    """
    root = Path(out) / ENGINES_DIR
    found = {}
    for engine_file in root.glob(f"*/*/{ENGINE_NAME_FILE}"):
        d = engine_file.parent
        direction = d.parent.name
        found[d] = {
            "kind": contract.PASS_SWEEP,
            "direction": direction,
            f"pin_{direction}": engine_file.read_text().strip(),
        }
    for top in (root, Path(out) / contract.CANDIDATES_DIR):
        for meta_file in top.glob(f"**/{PASS_META_NAME}"):
            found[meta_file.parent] = json.loads(meta_file.read_text())
    return sorted(found.items())


def listing(path):
    path = Path(path)
    if not path.exists():
        return []
    return list(dict.fromkeys(ln for ln in path.read_text().splitlines() if ln.strip()))


def _passes(out):
    """``(directory, meta, expected node ids)`` of pass 1 and every pass under ``all-engines``."""
    out = Path(out)
    yield out, {"kind": HEURISTIC}, listing(out / COLLECTED_NAME)
    for d, meta in pass_dirs(out):
        yield d, meta, listing(d / NODEIDS_NAME)


_SOURCE = {contract.PASS_SWEEP: "pinned"}  # every other kind is its own source
# one direction pinned; the other unpinned
_PINNED_KINDS = (contract.PASS_SWEEP, contract.PASS_CANDIDATE)
# backward cases, both directions pinned
_PAIR_KINDS = (contract.PASS_PAIR, contract.PASS_CANDIDATE_PAIR)


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
                "mode": row["mode"],
                "pin_fwd": pin["fwd"],
                "pin_bwd": pin["bwd"],
                "kernel": row[f"{direction}_kernel"],
                "stage": row["stage"],
                "fwd_candidate": meta.get("fwd_candidate", ""),
                "bwd_candidate": meta.get("bwd_candidate", ""),
            }
        )

    def mismatch(row, wanted, only_ran=False):
        """``wanted`` is ``{direction: engine}``; the declined reason when a served engine differs.
        With ``only_ran`` a direction whose state is empty (it never ran) is not compared, unless
        none ran."""
        if only_ran:
            wanted = {d: e for d, e in wanted.items() if row[d]} or wanted
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
            # A passing case credits the engine of each side that ran (a backward case's forward
            # engine too); a failing or crashed one only the failing side's; any other outcome
            # the direction of the test file.
            failing = crashed or failed
            if failing:
                sides = (failing,)
            elif outcome == "pass":
                sides = ("fwd", "bwd")
            else:
                sides = (direction_of(nodeid),)
            for direction in sides:
                for engine in sorted(served_engines(row[direction])):
                    add(nodeid, direction, engine, outcome, reason, row)
        elif kind == contract.PASS_DISCOVERY:
            for engine in sorted(served_engines(row["bwd"])):
                add(nodeid, "bwd", engine, outcome, reason, row)
        elif kind in _PINNED_KINDS:
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
            if kind in _PAIR_KINDS:
                wanted["fwd"] = pin["fwd"]
            if outcome in ("pass", "fail"):
                wrong = mismatch(row, wanted, only_ran=outcome == "fail")
                if wrong:
                    outcome, reason = "declined", wrong
            if kind in _PAIR_KINDS and outcome == "pass":
                # both pins were honoured and the case passed: credit both engines
                for direction in ("fwd", "bwd"):
                    add(nodeid, direction, pin[direction], outcome, reason, row)
                continue
            # a failing or crashed row credits the failing side; a declined row stays one row
            side = crashed or failed or "bwd"
            add(nodeid, side, wanted.get(side, "reference"), outcome, reason, row)
    direction = (
        meta.get("direction")
        if kind in _PINNED_KINDS
        else "bwd" if kind != HEURISTIC else None
    )
    for nodeid in expected:
        if nodeid not in recorded:
            engine = (
                pin[direction]
                if kind in _PINNED_KINDS + _PAIR_KINDS + (contract.PASS_REFWD,)
                else ""
            )
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
    engine = r["engine"] or "-"
    if r["source"] in ("heuristic", "pinned"):
        return f"{r['direction']} {engine}"
    label = f"{r['source']} {r['direction']} {engine}"
    if r["source"] == contract.PASS_CANDIDATE:
        return f"{label} #{r[r['direction'] + '_candidate'] or '-'}"
    if r["source"] == contract.PASS_CANDIDATE_PAIR:
        return (
            f"{label} (fwd {r['pin_fwd']} #{r['fwd_candidate'] or '-'}, "
            f"bwd {r['pin_bwd']} #{r['bwd_candidate'] or '-'})"
        )
    if r["source"] == contract.PASS_PAIR:
        # the partner pin: a backward credit names its forward, a forward credit its backward
        other = "bwd" if r["direction"] == "fwd" else "fwd"
        return f"{label} ({other} {r['pin_' + other]})"
    return label


def _tally(rows, counts, splits=None):
    """Add the engine ``rows`` to ``counts``, ``{label: Counter of outcomes}``, and (with ``splits``,
    ``{(label, "mode" | "file", value): Counter of outcomes}``) to the per-mode and per-test-file
    counts."""
    for r in rows:
        counts[_count_label(r)][r["outcome"]] += 1
        if contract.CANDIDATE_NOT_HONORED in r["reason"]:
            counts[_count_label(r)]["not_honored"] += 1
        if splits is not None:
            label = _count_label(r)
            file = Path(r["nodeid"].split("::")[0]).name
            for kind, value in (("mode", r["mode"] or "unrecorded"), ("file", file)):
                splits[(label, kind, value)][r["outcome"]] += 1
    return counts


def _outcome_text(c):
    extra = [
        f"{o}={n}"
        for o, n in sorted(c.items())
        if o not in ("pass", "fail", "declined")
    ]
    return f"pass={c['pass']} fail={c['fail']} declined={c['declined']}" + "".join(
        f" {e}" for e in extra
    )


def engine_splits(rows, splits=None):
    """Text: under each engine label of :func:`engine_counts`, its counts per mode (``hipdnn``: the
    hipDNN forward ran; ``ref-fwd``: a reference forward) and per test file. ``splits`` is a
    ``_tally`` already filled, for a run whose rows come shard by shard."""
    if splits is None:
        splits = collections.defaultdict(collections.Counter)
        _tally(rows, collections.defaultdict(collections.Counter), splits)
    lines = []
    labels = sorted({label for label, _, _ in splits})
    for label in labels:
        lines.append(label)
        lines += [
            f"  {kind} {value}: {_outcome_text(c)}"
            for (lab, kind, value), c in sorted(splits.items())
            if lab == label
        ]
    return "\n".join(lines)


def engine_counts(rows, counts=None):
    """Text: per (direction, engine) the pass / fail / declined counts (other outcomes appended;
    ``not_honored`` counts the declines whose candidate the backend did not build). ``counts``
    is a ``_tally`` already filled, for a run whose rows come shard by shard."""
    if counts is None:
        counts = _tally(rows, collections.defaultdict(collections.Counter))
    lines = [f"{label}: {_outcome_text(c)}" for label, c in sorted(counts.items())]
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


def pass_name(out, d):
    out = Path(out)
    if d == out:
        return MAIN_PASS
    base = out if d.is_relative_to(out / contract.CANDIDATES_DIR) else out / ENGINES_DIR
    return d.relative_to(base).as_posix()


def pass_totals(out):
    """``[(pass name, node ids expected, of those recorded)]``, pass 1 first (named ``main``)."""
    totals = []
    for d, _, expected in _passes(out):
        recorded = {r["nodeid"] for r in read_rows(d / CSV_NAME)}
        totals.append((pass_name(out, d), len(expected), len(recorded & set(expected))))
    return totals


def _accounting_line(name, expected, recorded):
    missing = expected - recorded
    return f"{name}: expected {expected}, recorded {recorded}" + (
        f" ({missing} not-run)" if missing else ""
    )


def pass_accounting(out):
    """Text, one line per pass: node ids expected against rows recorded."""
    return "\n".join(_accounting_line(*t) for t in pass_totals(out))


def _decline_lines(*outs):
    """``{engine as the backend logged it: [distinct decline lines, in file order]}`` over the
    ``declines/`` of every ``outs`` directory."""
    found = {}
    for out in outs:
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
    """The backend logs an ingestor engine by its name and an ASM builder by class name.

    Rules: equal names, or one a ``:``-suffix of the other; or the logged name is a
    ``...PlanBuilder`` class and the inventory name is a built-in engine (no ``:``) naming SDPA
    (``ASM_SDPA_ENGINE`` against ``SdpaFwdPlanBuilder`` / ``SdpaBwdPlanBuilder``).
    """
    a, b = inventory_name.lower(), logged.lower()
    if a == b or a.endswith(":" + b) or b.endswith(":" + a):
        return True
    return b.endswith("planbuilder") and ":" not in a and "sdpa" in a and "sdpa" in b


def _engine_outcomes(rows, by_engine):
    """Add the engine ``rows`` to ``by_engine``, ``{engine: Counter of outcomes}``."""
    for r in rows:
        by_engine.setdefault(r["engine"], collections.Counter())[r["outcome"]] += 1
    return by_engine


def inventory_summary(out, rows, reasons=3, declines_from=None, by_engine=None):
    """Text: every engine of ``<out>/engine-inventory.txt`` with its outcome counts, zeros included,
    and for an engine no case reached the first decline reasons logged under ``<out>/declines/``
    (or under every directory of ``declines_from``). ``by_engine`` is an ``_engine_outcomes``
    already filled, for a run whose rows come shard by shard.
    """
    path = Path(out) / contract.INVENTORY_NAME
    if not path.exists():
        return ""
    declines = _decline_lines(*(declines_from or [out]))
    if by_engine is None:
        by_engine = _engine_outcomes(rows, {})
    lines = []
    for name in listing(path):
        counts = by_engine.get(name, collections.Counter())
        lines.append(
            f"{name}: "
            + " ".join(
                f"{o}={counts[o]}"
                for o in ("pass", "fail", "crash", "declined", "not-run")
                + (
                    (contract.NOT_ENUMERABLE,)
                    if contract.NOT_ENUMERABLE in counts
                    else ()
                )
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


def failure_rows(out):
    """Every ``fail`` / ``crash`` row of every pass of the shard ``out``, as dicts: ``pass``, ``nodeid``,
    ``outcome``, ``direction`` (the failing side, else the test file's), ``engine`` (the engine served
    on that side, else the other side's, else ``-``; ``a+b`` for several), ``fail_side``, ``stage``,
    ``repro`` (relative to the shard directory, empty when none was saved), ``reason``. No such row is
    dropped, whatever engine its pass pinned.
    """
    found = []
    for d, _, _ in _passes(out):
        for r in read_rows(d / CSV_NAME):
            if r["outcome"] not in ("fail", "crash"):
                continue
            side = r["fail_side"] or direction_of(r["nodeid"])
            other = "bwd" if side == "fwd" else "fwd"
            engines = served_engines(r[side]) or served_engines(r[other])
            found.append(
                {
                    "pass": pass_name(out, d),
                    "nodeid": r["nodeid"],
                    "outcome": r["outcome"],
                    "direction": side,
                    "engine": "+".join(sorted(engines)) or "-",
                    "fail_side": r["fail_side"],
                    "stage": r["stage"],
                    "repro": r["repro"],
                    "reason": r["reason"],
                }
            )
    return found


def _shard_accounting(name, shard):
    if shard is None:
        return f"{name}: MISSING (no shard directory)"
    (_, collected, recorded), *others = pass_totals(shard)
    if not collected:
        return f"{name}: collected.txt missing or empty, case count unknown"
    line = f"{name}: collected {collected}, recorded {recorded}, not-run {collected - recorded}"
    behind = sum(e - r for _, e, r in others)
    return line + (f", other passes not-run {behind}" if behind else "")


def run_report(run_dir, expected=(), baseline=None):
    """Text report of the whole run in ``run_dir``: shard accounting (``expected`` names the shards
    that must exist), outcome totals, passes and engines summed over every shard, the engines
    no case passed, every failure, and with ``baseline`` (a file of failing node ids) the diff.
    """
    shards = {d.name: d for d in shard_dirs(run_dir)}
    rows, duplicates = _union(run_dir)
    names = list(dict.fromkeys([*expected, *shards]))
    passes, counts, by_engine, failures, splits = (
        {},
        collections.defaultdict(collections.Counter),
        {},
        [],
        collections.defaultdict(collections.Counter),
    )
    for name, shard in shards.items():
        for pass_name, want, got in pass_totals(shard):
            total = passes.setdefault(pass_name, [0, 0])
            total[0] += want
            total[1] += got
        engines = engine_rows(shard)
        _tally(engines, counts, splits)
        _engine_outcomes(engines, by_engine)
        failures += [{"shard": name, **f} for f in failure_rows(shard)]
    lines = ["== shards"] + [_shard_accounting(n, shards.get(n)) for n in names]
    lines += ["", "== duplicate node ids"]
    lines += [f"{n}: {', '.join(dirs)}" for n, dirs in duplicates.items()] or ["none"]
    lines += ["", "== outcomes", summarize(rows, failing=False)]
    lines += ["", "== passes (all shards)"] + [
        _accounting_line(n, *t) for n, t in passes.items()
    ]
    lines += ["", "== engines (all shards)", engine_counts((), counts)]
    lines += [
        "",
        "== engines by mode and test file (all shards)",
        engine_splits((), splits),
    ]
    inventories = [d for d in shards.values() if (d / contract.INVENTORY_NAME).exists()]
    lines += ["", "== inventory"]
    if inventories:
        first = inventories[0]
        listed = listing(first / contract.INVENTORY_NAME)
        lines += [
            f"inventory of {d.name} differs from {first.name}"
            for d in inventories[1:]
            if listing(d / contract.INVENTORY_NAME) != listed
        ]
        zero = [n for n in listed if not by_engine.get(n, {}).get("pass")]
        lines.append("engines with no pass: " + (", ".join(zero) or "none"))
        lines.append(
            inventory_summary(
                first, (), declines_from=list(shards.values()), by_engine=by_engine
            )
        )
    else:
        lines.append("no engine-inventory.txt in any shard")
    lines += ["", f"== failures ({len(failures)})"]
    for f in failures:
        repro = f"{f['shard']}/{f['repro']}" if f["repro"] else "-"
        lines.append(
            f"{f['outcome']} {f['shard']} {f['pass']} {f['direction']} {f['engine'] or '-'} "
            f"side={f['fail_side'] or '-'} stage={f['stage'] or '-'} repro={repro} "
            f"{f['nodeid']} :: {f['reason'][:160]}"
        )
    if baseline:
        lines += [
            "",
            f"== baseline {Path(baseline).name}",
            _baseline_diff(baseline, rows, failures),
        ]
    return "\n".join(lines)


def _baseline_diff(baseline, rows, failures):
    known = {
        ln.strip()
        for ln in Path(baseline).read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    }
    here = {f["nodeid"]: f for f in failures}
    outcome = {r["nodeid"]: r["outcome"] for r in rows}
    new = sorted(here.keys() - known)
    fixed = sorted(n for n in known - here.keys() if outcome.get(n) == "pass")
    unconfirmed = sorted(known - here.keys() - set(fixed))
    lines = [
        f"baseline {len(known)} ids, failing here {len(here)}, still failing {len(known & here.keys())}",
        f"new failures (not in baseline): {len(new)}",
        *[
            f"  {n}  [{here[n]['outcome']} {here[n]['direction']} {here[n]['engine'] or '-'}]"
            for n in new
        ],
        "",
        f"no longer failing (in baseline, outcome pass here): {len(fixed)}",
        *[f"  {n}: {outcome[n]}" for n in fixed],
        "",
        f"not confirmed fixed (in baseline, no failure row here, outcome other than pass or not in this run): {len(unconfirmed)}",
        *[f"  {n}: {outcome.get(n, 'not in this run')}" for n in unconfirmed],
    ]
    return "\n".join(lines)


ABSENT = "not in run"
COMPARE_EXAMPLES = 3


def _served(row):
    """``fwd <engines>, bwd <engines>`` of a reconciled row (``-`` for a side that served none)."""
    return ", ".join(
        f"{d} {'+'.join(sorted(served_engines(row[d]))) or '-'}" for d in ("fwd", "bwd")
    )


def compare_runs(run_dir, other_dir):
    """Text: the reconciled rows of ``other_dir`` (earlier) against ``run_dir`` (later), case by case:
    every outcome transition that happened (``other -> run``, with a count and example node ids;
    ``not in run`` for an id only one run has) and the cases whose served engine changed with the
    same outcome."""
    mine = {r["nodeid"]: r for r in reconcile(run_dir)}
    theirs = {r["nodeid"]: r for r in reconcile(other_dir)}
    moves = collections.defaultdict(list)
    swaps = []
    for nodeid in sorted(mine.keys() | theirs.keys()):
        a, b = theirs.get(nodeid), mine.get(nodeid)
        old, new = (a or {}).get("outcome", ABSENT), (b or {}).get("outcome", ABSENT)
        if old != new:
            moves[(old, new)].append(nodeid)
        elif _served(a) != _served(b):
            swaps.append((nodeid, new, _served(a), _served(b)))
    both = len(mine.keys() & theirs.keys())
    changed = sum(map(len, moves.values()))
    lines = [
        f"== compare {Path(other_dir).name} -> {Path(run_dir).name}",
        f"cases: {both} in both, "
        f"{len(theirs.keys() - mine.keys())} only in {Path(other_dir).name}, "
        f"{len(mine.keys() - theirs.keys())} only in {Path(run_dir).name}",
        f"outcome transitions: {changed} cases changed outcome",
    ]
    for (old, new), ids in sorted(moves.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        lines.append(f"  {old} -> {new}: {len(ids)}")
        lines += [f"      {n}" for n in ids[:COMPARE_EXAMPLES]]
    lines.append(f"served engine changed, same outcome: {len(swaps)}")
    for nodeid, outcome, old, new in swaps[:COMPARE_EXAMPLES]:
        lines.append(f"  {nodeid} [{outcome}]: {old} -> {new}")
    return "\n".join(lines)


def claim_repro(root, engine, nodeid):
    """Whether a repro for ``nodeid`` may be saved under ``root`` (``<out>/repro``): at most
    ``REPRO_MAX_PER_KEY`` per (engine, test function) and ``REPRO_MAX_BYTES`` on disk. A slot is
    claimed atomically, so concurrent workers never exceed the per-key cap."""
    root = Path(root)
    used = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    if used >= contract.REPRO_MAX_BYTES:
        return False
    function = re.sub(r"\[.*\]$", "", nodeid)
    key = hashlib.sha1(f"{engine}\0{function}".encode()).hexdigest()[:12]
    claims = root / ".claims"
    claims.mkdir(parents=True, exist_ok=True)
    for slot in range(contract.REPRO_MAX_PER_KEY):
        try:
            (claims / f"{key}-{slot}").mkdir()
            return True
        except FileExistsError:
            continue
    return False


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
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the report of the whole run (every shard of <run-dir>) instead of the summary",
    )
    parser.add_argument(
        "--shards-file",
        type=Path,
        help="with --report: the expected shards, one '<suite> <k> <N>' per line",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="with --report: failing node ids to diff against, one per line",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        metavar="OTHER",
        help="also print a case-by-case comparison of OTHER (an earlier run directory) with <run-dir>",
    )
    args = parser.parse_args(argv)
    try:
        rows = reconcile(args.run_dir)
        compared = compare_runs(args.run_dir, args.compare) if args.compare else None
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    if args.out:
        args.out.write_text(
            ",".join(FIELDS) + "\n" + "".join(format_row(r) for r in rows)
        )
    if args.report:
        expected = read_shards_file(args.shards_file) if args.shards_file else ()
        print(run_report(args.run_dir, expected, args.baseline))
    else:
        print(summarize(rows))
    if compared:
        print(compared)
    if args.engines:
        print(write_engines(args.engines))
        print(engine_splits(engine_rows(args.engines)))
        print(pass_accounting(args.engines))
        print(inventory_summary(args.engines, engine_rows(args.engines)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
