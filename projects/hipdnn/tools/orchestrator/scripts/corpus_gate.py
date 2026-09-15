#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Prove the ingestor's production run accounted for every graph in the corpus.

The ingestor runbook requires an attributable outcome for every input, not just a
count of how many it served. A coverage number is only as good as its denominator,
and the contract the ingestor agent writes is exactly where a friendly denominator
would come from: report outcomes for the graphs that went well and simply omit the
ones that did not build, and "N served, 0 errors" reads as success while the corpus
itself says nothing about the graphs that produced no row at all. A graph with no
timing result and no decline reason is a missing outcome, not a zero, and it has to
show up as one.

So this gate does not read the contract's own tallies. It walks `--corpus-root`
itself to build the denominator (every file matching `--glob`, independent of
anything the agent reported), then joins the contract's `corpus.inputs` against that
list by resolved path. Anything that does not form a clean one-to-one correspondence
-- a contract entry naming a path that is not in the corpus, a corpus file with no
contract entry, the same path reported twice, an outcome string outside the five the
runbook recognizes -- fails the join, and the join failing is reported as its own
number rather than folded silently into "coverage".

    corpus_gate.py --contract ingestor.json --corpus-root corpus/ --out gate.json

Exit codes: 0 the report was written (an incomplete join or zero coverage is a
verdict, not an error -- the flow asserts on `complete_join` and `meets_target`),
1 `--contract` could not be read/parsed, or `--corpus-root` is not a directory.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

KNOWN_OUTCOMES = ("served", "declined", "error", "missing", "ambiguous")


def _resolve_graph_path(raw: object, corpus_root: Path) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = corpus_root / candidate
    return candidate.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, help="ingestor.json to read")
    parser.add_argument(
        "--corpus-root",
        required=True,
        help="directory the corpus denominator is built from",
    )
    parser.add_argument("--out", required=True, help="where to write the JSON report")
    parser.add_argument(
        "--glob",
        default="**/*.json",
        help="pattern (relative to --corpus-root) that defines the corpus (default: **/*.json)",
    )
    args = parser.parse_args()

    corpus_root = Path(args.corpus_root).resolve()
    if not corpus_root.is_dir():
        print(
            f"error: --corpus-root is not a directory: {corpus_root}", file=sys.stderr
        )
        return 1

    try:
        contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(
            f"error: could not read contract {args.contract}: {error}", file=sys.stderr
        )
        return 1
    if not isinstance(contract, dict):
        print(
            f"error: contract {args.contract}'s top level is not a JSON object",
            file=sys.stderr,
        )
        return 1

    corpus_block = contract.get("corpus")
    inputs = corpus_block.get("inputs") if isinstance(corpus_block, dict) else None
    if not isinstance(inputs, list):
        inputs = []

    found: set[Path] = set()
    # include_hidden=True matches guard_scope.py's rationale: without it `glob`
    # silently drops dot-prefixed names, so a dot-prefixed corpus file would be
    # missing from the denominator without ever showing up as an error.
    for match in glob.glob(
        args.glob, root_dir=corpus_root, recursive=True, include_hidden=True
    ):
        candidate = corpus_root / match
        if candidate.is_file():
            found.add(candidate.resolve())
    denominator = len(found)

    reported = len(inputs)
    joined = 0
    unjoined_paths: list[Path | str] = []
    seen: Counter[Path] = Counter()
    outcome_counts = {name: 0 for name in KNOWN_OUTCOMES}
    unknown_outcome = 0

    for entry in inputs:
        if not isinstance(entry, dict):
            unknown_outcome += 1
            continue
        outcome = entry.get("outcome")
        if outcome in outcome_counts:
            outcome_counts[outcome] += 1
        else:
            unknown_outcome += 1

        resolved = _resolve_graph_path(entry.get("graph"), corpus_root)
        if resolved is None:
            unjoined_paths.append(f"<malformed graph field: {entry.get('graph')!r}>")
            continue
        seen[resolved] += 1
        if resolved in found:
            joined += 1
        else:
            unjoined_paths.append(resolved)

    duplicates = sum(count - 1 for count in seen.values() if count > 1)
    unaccounted_paths = sorted(found - set(seen), key=lambda p: p.as_posix())
    unjoined = len(unjoined_paths)
    unaccounted = len(unaccounted_paths)

    complete_join = (
        1
        if (
            unjoined == 0
            and unaccounted == 0
            and duplicates == 0
            and unknown_outcome == 0
            and reported == denominator
        )
        else 0
    )
    meets_target = (
        1
        if (
            complete_join == 1
            and outcome_counts["error"] == 0
            and outcome_counts["missing"] == 0
            and outcome_counts["ambiguous"] == 0
            and outcome_counts["served"] > 0
        )
        else 0
    )

    notes: list[str] = []
    if denominator == 0:
        notes.append(
            f"the corpus at {corpus_root.as_posix()} (pattern {args.glob!r}) contains "
            f"zero graphs. A corpus of zero graphs proves nothing about the "
            f"ingestor's coverage, whatever the contract reports."
        )
    if unjoined:
        sample = ", ".join(
            p.as_posix() if isinstance(p, Path) else str(p) for p in unjoined_paths[:10]
        )
        notes.append(
            f"{unjoined} contract entry/entries name a graph that is not in the "
            f"corpus under {corpus_root.as_posix()}: {sample}. A reported outcome "
            f"for a graph outside the corpus is a join error, not extra credit."
        )
    if unaccounted:
        sample = ", ".join(p.as_posix() for p in unaccounted_paths[:10])
        notes.append(
            f"{unaccounted} corpus file(s) have no contract entry at all: {sample}. "
            f"Every input under --corpus-root needs an attributable outcome; a graph "
            f"with no timing row and no decline reason is a missing outcome, not a zero."
        )
    if duplicates:
        notes.append(
            f"{duplicates} duplicate report(s): a graph path was reported more than "
            f"once. A graph contributes exactly one outcome to the join."
        )
    if unknown_outcome:
        notes.append(
            f"{unknown_outcome} entry/entries report an outcome outside "
            f"{KNOWN_OUTCOMES}."
        )
    for name in ("error", "missing", "ambiguous"):
        if outcome_counts[name]:
            notes.append(f"{outcome_counts[name]} graph(s) reported outcome '{name}'.")
    if denominator > 0 and outcome_counts["served"] == 0:
        notes.append(
            "no graph in the corpus was reported as 'served'; nothing was served."
        )

    report = {
        "denominator": denominator,
        "reported": reported,
        "joined": joined,
        "unjoined": unjoined,
        "unaccounted": unaccounted,
        "duplicates": duplicates,
        "served": outcome_counts["served"],
        "declined": outcome_counts["declined"],
        "error": outcome_counts["error"],
        "missing": outcome_counts["missing"],
        "ambiguous": outcome_counts["ambiguous"],
        "unknown_outcome": unknown_outcome,
        "complete_join": complete_join,
        "meets_target": meets_target,
        "unaccounted_sample": [p.as_posix() for p in unaccounted_paths[:10]],
        "unjoined_sample": [
            p.as_posix() if isinstance(p, Path) else str(p) for p in unjoined_paths[:10]
        ],
        "feedback": "\n\n".join(notes),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"corpus gate: denominator={denominator} reported={reported} joined={joined} "
        f"unjoined={unjoined} unaccounted={unaccounted} duplicates={duplicates} "
        f"complete_join={complete_join} meets_target={meets_target}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
