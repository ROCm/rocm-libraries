"""Cross-engine comparison of L1 collections over one corpus.

Each engine's `uhd_gen generate --role predict_engine_tflops` run writes a `corpus.csv`
of measured immediate throughput, one row per (graph, device). Joined on the `benchmark`
id -- the UUID5 the corpus mints from the canonical graph document, which is stable
across engines and runs -- those files answer the question the heuristics exist for:
for each problem, which engine is actually fastest, and by how much.

The manifest of the corpus supplies the regime, so the answer can be read per population
rather than as one aggregate that hides a model excellent on prefill and useless on decode
(RFC 0019.13 §11.2).

Usage:
    python compare_engines.py --manifest corpus/manifest.json \
        --engine rocKE=uhd-gen-dense/l1/corpus.csv \
        --engine flyDSL=uhd-gen-fly/l1/corpus.csv \
        --engine AITER=uhd-gen-aiter/l1/corpus.csv \
        [--report out.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import pandas as pd


def load(spec: str) -> tuple[str, pd.DataFrame]:
    label, _, path = spec.partition("=")
    frame = pd.read_csv(path)
    frame["engine_label"] = label
    return label, frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--engine", action="append", required=True,
                        help="LABEL=path/to/corpus.csv (repeatable)")
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    regime = {row["benchmark"]: row["regime"] for row in manifest["graphs"]}
    name = {row["benchmark"]: row["name"] for row in manifest["graphs"]}
    total = len(manifest["graphs"])

    frames = dict(load(spec) for spec in args.engine)
    best: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for label, frame in frames.items():
        measured = frame[frame.get("tflops").notna()] if "tflops" in frame else frame.iloc[0:0]
        # One engine can carry several rows per graph (per device, per repeat). The engine's
        # number for a problem is its best measured throughput, which is what a selector that
        # picks this engine would deliver once its own knobs are settled.
        for benchmark, group in measured.groupby("benchmark"):
            best[str(benchmark)][label] = float(group["tflops"].max())

    print(f"corpus {total} graphs; {len(best)} carry at least one measurement\n")
    print(f"{'engine':<12} {'measured':>9} {'coverage':>9} {'wins':>6} {'win rate':>9}")
    wins: collections.Counter = collections.Counter()
    for benchmark, scores in best.items():
        wins[max(scores, key=scores.get)] += 1
    for label in frames:
        served = sum(1 for scores in best.values() if label in scores)
        served_wins = wins[label]
        rate = 100.0 * served_wins / served if served else 0.0
        print(f"{label:<12} {served:>9} {100.0 * served / max(1, total):>8.1f}% "
              f"{served_wins:>6} {rate:>8.1f}%")

    contested = {b: s for b, s in best.items() if len(s) > 1}
    print(f"\ncontested: {len(contested)} graphs have two or more engines measured")
    if contested:
        print(f"\n{'regime':<26} {'graphs':>7}  winners")
        by_regime: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for benchmark, scores in contested.items():
            by_regime[regime.get(benchmark, "?")][max(scores, key=scores.get)] += 1
        for label, counter in sorted(by_regime.items()):
            spread = ", ".join(f"{k} {v}" for k, v in counter.most_common())
            print(f"{label:<26} {sum(counter.values()):>7}  {spread}")

        margins = []
        for benchmark, scores in contested.items():
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            margin = (ranked[0][1] - ranked[1][1]) / ranked[1][1] if ranked[1][1] else 0.0
            margins.append((margin, benchmark, ranked))
        margins.sort(reverse=True)
        print("\nwidest margins (winner over runner-up):")
        for margin, benchmark, ranked in margins[:8]:
            spread = "  ".join(f"{label} {value:.1f}" for label, value in ranked)
            print(f"  {margin * 100:6.1f}%  {name.get(benchmark, benchmark)[:58]:<58} {spread}")
        close = [m for m, _, _ in margins if m < 0.05]
        print(f"\n{len(close)} of {len(contested)} contested graphs are within 5% -- "
              "the population where a wrong pick costs least and a heuristic is hardest to train")

    if args.report:
        args.report.write_text(json.dumps({
            "corpus": str(args.manifest),
            "graphs": total,
            "measured": {label: sum(1 for s in best.values() if label in s) for label in frames},
            "wins": dict(wins),
            "contested": len(contested),
            "per_graph": {name.get(b, b): s for b, s in best.items()},
        }, indent=2), encoding="utf-8")
        print(f"\nwritten {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
