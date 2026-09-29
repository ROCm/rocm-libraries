"""Does the engine whose model predicts the better value actually run better?

L1 is the one score compared ACROSS engines, so the question it must answer is not "how
close is the number" but "does it pick the winner". This joins a bake-off's predictions with
the per-engine measured L1 corpora on the graph and reports, per ranking metric, over the
graphs where two or more engines both predicted and measured:

  * agreement -- predicted winner == measured winner
  * the cost of disagreeing, as the regret of the predicted pick relative to the measured
    best in that metric's own direction (throughput given up for tflops, time added for time)
  * both, per regime, because an aggregate hides a selector that is right on prefill and
    wrong on every decode-shaped problem

Engines are named as the runtime names them, so any engine's model scores -- there is no
allowlist. A measured row counts toward the metric its collection binding was taken under:
an engine's own selector picks its kernel in the requested metric, so a `time` model is
scored against `time`-selector measurements only.

Usage:
    python score_predictions.py --manifest corpus/manifest.json --predictions predictions.json \
        --measured out-miopen/l1/corpus_tflops.csv --measured out-miopen/l1/corpus_time.csv \
        --measured out-other/l1/corpus.csv
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import pandas as pd

# The registry the runtime enforces (label column and direction per metric), from the
# checkout this script ships in rather than a second copy of it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from uhd_gen.ranking_metrics import RANKING_METRICS  # noqa: E402


def _regret(metric, best: float, picked: float) -> float:
    """How much worse the pick is than the best, as a fraction, in the metric's direction."""
    return (best - picked) / best if metric.objective == "max" else (picked - best) / best


def _measured(paths, by_benchmark, metrics):
    """metric -> graph name -> engine -> the engine's measured value in that metric."""
    measured = {name: collections.defaultdict(dict) for name in metrics}
    for path in paths:
        frame = pd.read_csv(path)
        if "binding" in frame:
            frame["_metric"] = frame["binding"].map(lambda value: json.loads(value).get("metric"))
        else:
            raise SystemExit(f"{path}: no binding column, so the metric it was collected under is unknown")
        for (metric_name, engine, benchmark), group in frame.groupby(["_metric", "engine_name", "benchmark"]):
            if metric_name not in measured:
                continue
            metric = RANKING_METRICS[metric_name]
            values = pd.to_numeric(group[metric.label], errors="coerce").dropna()
            name = by_benchmark.get(str(benchmark))
            if name and not values.empty:
                best = values.max() if metric.objective == "max" else values.min()
                measured[metric_name][name][engine] = float(best)
    return measured


def _report(metric, rows) -> None:
    frame = pd.DataFrame(rows)
    agree = int(frame["agree"].sum())
    print(f"\n=== {metric.name} ({metric.units}, {metric.objective}) ===")
    print(f"contested graphs scored by two or more engines: {len(frame)}")
    print(f"  predicted winner == measured winner : {agree} ({100 * agree / len(frame):.1f}%)")
    wrong = frame[~frame["agree"]]
    if not wrong.empty:
        print(f"  wrong pick                         : {len(wrong)}")
        print(f"  regret when wrong                  : mean {100 * wrong['regret'].mean():.1f}%"
              f", p95 {100 * wrong['regret'].quantile(0.95):.1f}%"
              f", max {100 * wrong['regret'].max():.1f}%")
    # The whole-corpus figure a selector delivers: zero where it picks right, the shortfall
    # where it does not. This is what a user feels, unlike the agreement rate.
    print(f"  mean regret over ALL scored        : {100 * frame['regret'].mean():.2f}%")

    print(f"\n{'regime':<26} {'graphs':>7} {'agree':>7}   measured winners")
    for name, group in frame.groupby("regime"):
        spread = ", ".join(f"{k} {v}" for k, v in
                           collections.Counter(group["measured_winner"]).most_common())
        print(f"{name:<26} {len(group):>7} {100 * group['agree'].mean():>6.0f}%   {spread}")

    print(f"\n{'pair':<48} {'graphs':>7} {'agree':>7}")
    for engines, group in frame.groupby(frame.apply(
            lambda row: " vs ".join(sorted({row["measured_winner"], row["predicted_winner"]})), axis=1)):
        print(f"{engines:<48} {len(group):>7} {100 * group['agree'].mean():>6.0f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--predictions", required=True, type=pathlib.Path)
    parser.add_argument("--measured", action="append", required=True, type=pathlib.Path,
                        help="An L1 corpus.csv (repeatable); rows are attributed by engine_name "
                             "and by the metric their binding was collected under")
    parser.add_argument("--metric", action="append", choices=tuple(RANKING_METRICS),
                        help="Metric(s) to score (default: every metric the predictions carry)")
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    regime = {row["name"]: row["regime"] for row in manifest["graphs"]}
    by_benchmark = {row["benchmark"]: row["name"] for row in manifest["graphs"]}

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    metrics = list(dict.fromkeys(args.metric or [row["metric"] for row in predictions
                                                 if row.get("metric") in RANKING_METRICS]))
    predicted = {name: collections.defaultdict(dict) for name in metrics}
    for row in predictions:
        if row.get("metric") in predicted and row.get("value") is not None:
            predicted[row["metric"]][row["graph"]][row["engine"]] = float(row["value"])
    measured = _measured(args.measured, by_benchmark, metrics)

    scored = []
    for name in metrics:
        metric = RANKING_METRICS[name]
        better = max if metric.objective == "max" else min
        rows = []
        for graph, scores in measured[name].items():
            guesses = predicted[name].get(graph, {})
            shared = set(scores) & set(guesses)
            if len(shared) < 2:
                continue
            measured_best = better(sorted(shared), key=lambda engine: scores[engine])
            predicted_best = better(sorted(shared), key=lambda engine: guesses[engine])
            rows.append({"metric": name, "graph": graph, "regime": regime.get(graph, "?"),
                         "engines": len(shared), "measured_winner": measured_best,
                         "predicted_winner": predicted_best, "agree": measured_best == predicted_best,
                         "regret": _regret(metric, scores[measured_best], scores[predicted_best])})
        if not rows:
            print(f"\n=== {name} ===\nno graph has two engines with both a prediction and a measurement")
            continue
        _report(metric, rows)
        scored.extend(rows)

    if not scored:
        return 1
    if args.report:
        args.report.write_text(pd.DataFrame(scored).to_json(orient="records", indent=1), encoding="utf-8")
        print(f"\nwritten {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
