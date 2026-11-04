#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A shape set's evaluation slice: the graphs held out of every model trained from it.

`choose` picks the slice once, from the set's corpus_gen manifests (one per operation), and
writes it beside the set: per regime, `round(fraction * n)` graphs (at least one where the regime has two or more,
never all of a regime), chosen by a seeded hash of the graph id, so the slice depends on the
set alone. Every model trained from the set holds those graphs out
(`uhd_gen generate --eval-benchmarks`), and every one is scored on the same graphs.

`sample` draws validation graphs from one or more slices, spread evenly over their regimes:
one from each regime in turn before any regime gives a second. With `--trained` (a model
run's generation_manifest.json) it refuses a slice graph the model trained on: scoring it
would be recall, not validation.

`holdout` writes the union of several slices as the JSON list `--eval-benchmarks` reads.

`rows` keeps the rows of an exported corpus (`python -m uhd_gen.dataset export`) whose graph is
in the slices: the stored measurements of the evaluation set, for `uhd_gen evaluate`.

Standard library only.

    eval_slice.py choose <set>/<operation>/manifest.json ... [--fraction 0.1] [--seed 0]
    eval_slice.py holdout <set>/eval_slice.json ... --out eval_benchmarks.json
    eval_slice.py sample <set>/eval_slice.json ... --count 200 --out sample.json
        [--trained <run>/generation_manifest.json]
    eval_slice.py rows exported.json <set>/eval_slice.json ... --out slice_rows.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

#: The share of each regime held out.
DEFAULT_FRACTION = 0.1

SCHEMA = "hipdnn_uhd.eval_slice/1"


def _rank(benchmark: str, seed: int) -> str:
    return hashlib.sha256(f"{benchmark}|seed={seed}".encode("utf-8")).hexdigest()


def choose(
    graphs: Iterable[Mapping[str, Any]],
    fraction: float = DEFAULT_FRACTION,
    seed: int = 0,
) -> dict[str, Any]:
    """The slice of a set's manifest graphs held out of training."""
    if not 0 < fraction < 1:
        raise ValueError(
            f"an evaluation slice is a share of the set: 0 < f < 1, not {fraction}"
        )
    by_regime: dict[str, dict[str, Mapping[str, Any]]] = {}
    for graph in graphs:
        benchmark = graph.get("benchmark")
        if benchmark:
            regime = str(graph.get("regime") or "unlabelled")
            by_regime.setdefault(regime, {})[str(benchmark)] = graph
    if not by_regime:
        raise ValueError("the manifest names no graph ids")
    chosen: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for regime, members in sorted(by_regime.items()):
        ids = sorted(members)
        take = round(fraction * len(ids))
        take = min(max(take, 1), len(ids) - 1) if len(ids) >= 2 else 0
        held = sorted(ids, key=lambda b: _rank(b, seed))[:take]
        chosen.extend(
            {
                "benchmark": b,
                "regime": regime,
                "file": members[b].get("file"),
                **({"path": members[b]["path"]} if members[b].get("path") else {}),
            }
            for b in held
        )
        counts[regime] = {"graphs": len(ids), "held_out": len(held)}
    return {
        "schema": SCHEMA,
        "fraction": fraction,
        "seed": seed,
        "benchmarks": sorted(g["benchmark"] for g in chosen),
        "graphs": sorted(chosen, key=lambda g: g["benchmark"]),
        "by_regime": counts,
    }


def sample_evenly(
    graphs: Iterable[Mapping[str, Any]], count: int, seed: int = 0
) -> list[Mapping[str, Any]]:
    """Up to `count` of `graphs`, one from each regime in turn, each regime's in the order of
    a seeded hash of the graph id; all of them when there are no more than `count`."""
    by_regime: dict[str, list[Mapping[str, Any]]] = {}
    for graph in graphs:
        by_regime.setdefault(str(graph.get("regime") or "unlabelled"), []).append(graph)
    queues = [
        sorted(members, key=lambda g: _rank(str(g["benchmark"]), seed))
        for _, members in sorted(by_regime.items())
    ]
    chosen: list[Mapping[str, Any]] = []
    depth = 0
    while len(chosen) < count and any(depth < len(q) for q in queues):
        for queue in queues:
            if depth < len(queue) and len(chosen) < count:
                chosen.append(queue[depth])
        depth += 1
    return chosen


def _slice_graphs(paths: list[Path]) -> list[dict[str, Any]]:
    """Every graph of the named slices, with its file as an absolute path; one entry per id."""
    graphs: dict[str, dict[str, Any]] = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        root = path.resolve().parent
        for graph in document.get("graphs") or []:
            entry = dict(graph)
            if entry.get("file") and not entry.get("path"):
                entry["path"] = str((root / entry["file"]).resolve())
            entry["slice"] = str(path)
            graphs.setdefault(entry["benchmark"], entry)
    return list(graphs.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser(
        "choose", help="choose a set's slice from its manifest.json files"
    )
    pick.add_argument("manifests", type=Path, nargs="+")
    pick.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    pick.add_argument("--seed", type=int, default=0)
    pick.add_argument(
        "--out",
        type=Path,
        help="default: eval_slice.json in the set's root (beside the operations)",
    )
    hold = sub.add_parser(
        "holdout", help="the union of slices, as --eval-benchmarks reads it"
    )
    hold.add_argument("slices", type=Path, nargs="+")
    hold.add_argument("--out", type=Path, required=True)
    draw = sub.add_parser(
        "sample", help="graphs spread evenly over the slices' regimes"
    )
    draw.add_argument("slices", type=Path, nargs="+")
    draw.add_argument("--count", type=int, default=200)
    draw.add_argument("--seed", type=int, default=0)
    draw.add_argument("--out", type=Path, required=True)
    draw.add_argument(
        "--trained",
        type=Path,
        help="a model run's generation_manifest.json: refuse graphs it trained on",
    )
    keep = sub.add_parser("rows", help="the exported rows of the slices' graphs")
    keep.add_argument("exported", type=Path)
    keep.add_argument("slices", type=Path, nargs="+")
    keep.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "choose":
        graphs = []
        for manifest in args.manifests:
            root = manifest.resolve().parent
            for graph in (
                json.loads(manifest.read_text(encoding="utf-8")).get("graphs") or []
            ):
                entry = dict(graph)
                if entry.get("file"):
                    entry["path"] = str(root / entry["file"])
                graphs.append(entry)
        chosen = choose(graphs, args.fraction, args.seed)
        chosen["manifests"] = [str(m.resolve()) for m in args.manifests]
        sets = {m.resolve().parent.parent for m in args.manifests}
        if args.out is None and len(sets) != 1:
            raise SystemExit(
                "the manifests are of different sets: name the slice with --out"
            )
        out = args.out or sets.pop() / "eval_slice.json"
        if out.exists():
            raise SystemExit(f"{out} exists: a set's slice is chosen once and kept")
        out.write_text(json.dumps(chosen, indent=1) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "out": str(out),
                    "held_out": len(chosen["benchmarks"]),
                    "by_regime": chosen["by_regime"],
                }
            )
        )
        return 0
    graphs = _slice_graphs(args.slices)
    if args.command == "rows":
        held = {g["benchmark"] for g in graphs}
        rows = [
            r
            for r in json.loads(args.exported.read_text(encoding="utf-8"))
            if str(r.get("benchmark")) in held
        ]
        args.out.write_text(json.dumps(rows) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "rows": len(rows),
                    "graphs": len({r.get("benchmark") for r in rows}),
                }
            )
        )
        return 0
    if args.command == "holdout":
        args.out.write_text(
            json.dumps(sorted(g["benchmark"] for g in graphs)) + "\n", encoding="utf-8"
        )
        print(json.dumps({"out": str(args.out), "held_out": len(graphs)}))
        return 0
    if args.trained:
        manifest = json.loads(args.trained.read_text(encoding="utf-8"))
        trained = {
            str(key[0])
            for model in manifest.get("models") or []
            for key in model.get("training_problem_keys") or []
        }
        overlap = sorted(g["benchmark"] for g in graphs if g["benchmark"] in trained)
        if overlap:
            raise SystemExit(
                f"{len(overlap)} slice graph(s) are among the model's training "
                f"graphs (e.g. {overlap[0]}): it was not trained with these slices "
                "held out"
            )
    drawn = sample_evenly(graphs, args.count, args.seed)
    args.out.write_text(
        json.dumps({"schema": SCHEMA, "seed": args.seed, "graphs": drawn}, indent=1)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "drawn": len(drawn), "of": len(graphs)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
