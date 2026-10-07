"""Cross-engine comparison of L1 collections over one corpus, per ranking metric.

Joins each engine's `predict_engine` corpus CSVs on `benchmark` (stable across engines)
to show which engine is best per problem, per manifest regime (RFC 0019.13 §11.2). A row
counts only toward the metric its `binding.metric` was collected under: the engine picks
its kernel for the requested metric, so a `time` collection's throughput is not the
engine's `tflops` answer. One label may name several corpora of one engine.

Usage:
    python compare_engines.py --manifest corpus/manifest.json \
        --engine rocKE=uhd-gen-dense/l1/corpus_tflops.csv \
        --engine rocKE=uhd-gen-dense/l1/corpus_time.csv \
        --engine AITER=uhd-gen-aiter/l1/corpus.csv \
        [--metric tflops] [--report out.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import pandas as pd

# The runtime's metric registry, from this checkout rather than a second copy.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from uhd_gen.ranking_metrics import RANKING_METRICS, is_valid_metric_value  # noqa: E402


def measure(specs: list[str], names: dict[str, str], metrics: list[str]):
    """(labels in the order named, metric -> benchmark -> label -> best valid value).

    An engine's number for a problem is its best valid measurement in that metric, in the
    metric's direction. A label whose corpora disagree on the engine or its selector
    contract within one metric is refused: its numbers would not describe one engine.
    """
    labels: list[str] = []
    best = {metric: collections.defaultdict(dict) for metric in metrics}
    contracts: dict[tuple[str, str], tuple] = {}
    for spec in specs:
        label, separator, path = spec.partition("=")
        if not separator or not label or not path:
            raise SystemExit(f"--engine {spec!r}: expected LABEL=path/to/corpus.csv")
        if label not in labels:
            labels.append(label)
        frame = pd.read_csv(path)
        if "binding" not in frame:
            raise SystemExit(
                f"{path}: no binding column, so the metric each row was collected under "
                "is unknown"
            )
        for record in frame.to_dict(orient="records"):
            binding = json.loads(record["binding"])
            metric_name = binding.get("metric")
            benchmark = str(record["benchmark"])
            if metric_name not in best or benchmark not in names:
                continue
            contract = (
                binding.get("engine"),
                json.dumps(binding.get("trained_against"), sort_keys=True),
            )
            if contracts.setdefault((label, metric_name), contract) != contract:
                raise SystemExit(
                    f"{path}: {label} mixes engines or selector contracts in its "
                    f"{metric_name} collections"
                )
            metric = RANKING_METRICS[metric_name]
            value = pd.to_numeric(record.get(metric.label), errors="coerce")
            if pd.isna(value) or not is_valid_metric_value(metric_name, float(value)):
                continue
            value = float(value)
            scores = best[metric_name][benchmark]
            if label in scores:
                choose = max if metric.objective == "max" else min
                value = choose(value, scores[label])
            scores[label] = value
    return labels, best


def compare(metric_name, measurements, labels, regime, names, total) -> dict:
    """Print one metric's comparison and return its report section."""
    metric = RANKING_METRICS[metric_name]
    choose = max if metric.objective == "max" else min
    # Sorted, so an exact tie goes to the same label whatever order corpora were named in.
    winners = {
        benchmark: choose(sorted(scores), key=scores.get)
        for benchmark, scores in measurements.items()
    }
    wins = collections.Counter(winners.values())
    print(f"\n=== {metric_name} ({metric.units}, {metric.objective}) ===")
    print(
        f"corpus {total} graphs; {len(measurements)} carry at least one measurement\n"
    )
    print(f"{'engine':<12} {'measured':>9} {'coverage':>9} {'wins':>6} {'win rate':>9}")
    measured = {
        label: sum(1 for scores in measurements.values() if label in scores)
        for label in labels
    }
    for label in labels:
        served = measured[label]
        rate = 100.0 * wins[label] / served if served else 0.0
        print(
            f"{label:<12} {served:>9} {100.0 * served / max(1, total):>8.1f}% "
            f"{wins[label]:>6} {rate:>8.1f}%"
        )

    contested = {b: s for b, s in measurements.items() if len(s) > 1}
    print(f"\ncontested: {len(contested)} graphs have two or more engines measured")
    if contested:
        print(f"\n{'regime':<26} {'graphs':>7}  winners")
        by_regime: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        for benchmark in contested:
            by_regime[regime.get(benchmark, "?")][winners[benchmark]] += 1
        for label, counter in sorted(by_regime.items()):
            spread = ", ".join(f"{k} {v}" for k, v in counter.most_common())
            print(f"{label:<26} {sum(counter.values()):>7}  {spread}")

        # How far the winner beats the runner-up, relative to the slower of the two.
        margins = []
        for benchmark, scores in contested.items():
            ranked = sorted(
                scores.items(),
                key=lambda kv: kv[1],
                reverse=metric.objective == "max",
            )
            winner, runner_up = ranked[0][1], ranked[1][1]
            slower = runner_up if metric.objective == "max" else winner
            if slower:
                margin = abs(winner - runner_up) / slower
            else:
                margin = 0.0 if winner == runner_up else float("inf")
            margins.append((margin, benchmark, ranked))
        margins.sort(reverse=True)
        print("\nwidest margins (winner over runner-up):")
        for margin, benchmark, ranked in margins[:8]:
            spread = "  ".join(f"{label} {value:.4g}" for label, value in ranked)
            print(
                f"  {margin * 100:6.1f}%  {names.get(benchmark, benchmark)[:58]:<58} {spread}"
            )
        close = [m for m, _, _ in margins if m < 0.05]
        print(
            f"\n{len(close)} of {len(contested)} contested graphs are within 5% -- "
            "the population where a wrong pick costs least and a heuristic is hardest to train"
        )
    return {
        "units": metric.units,
        "objective": metric.objective,
        "measured": measured,
        "wins": dict(wins),
        "contested": len(contested),
        "per_graph": {names.get(b, b): s for b, s in measurements.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument(
        "--engine",
        action="append",
        required=True,
        help="LABEL=path/to/corpus.csv (repeatable; one label may name several corpora)",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=tuple(RANKING_METRICS),
        help="Ranking metric to compare in (repeatable; default: every registered metric)",
    )
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    regime = {str(row["benchmark"]): row["regime"] for row in manifest["graphs"]}
    names = {str(row["benchmark"]): row["name"] for row in manifest["graphs"]}
    total = len(manifest["graphs"])
    metrics = list(dict.fromkeys(args.metric or RANKING_METRICS))

    labels, best = measure(args.engine, names, metrics)
    reports = {
        metric: compare(metric, best[metric], labels, regime, names, total)
        for metric in metrics
        if best[metric]
    }
    if not reports:
        print(
            f"no corpus row measures a graph of {args.manifest} in {', '.join(metrics)}"
        )
        return 1

    if args.report:
        args.report.write_text(
            json.dumps(
                {"corpus": str(args.manifest), "graphs": total, "metrics": reports},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwritten {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
