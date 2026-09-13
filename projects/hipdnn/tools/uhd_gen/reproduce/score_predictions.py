"""Does the engine whose model predicts higher actually run faster?

L1 is the one score compared ACROSS engines, so the question it must answer is not "how
close is the number" but "does it pick the winner". This joins a bake-off's predictions with
the per-engine measured corpora on the graph name and reports, over the graphs where two or
more engines both predicted and measured:

  * agreement -- predicted winner == measured winner
  * the cost of disagreeing, as the throughput given up relative to the measured best
  * both, per regime, because an aggregate hides a selector that is right on prefill and
    wrong on every decode-shaped problem

Usage:
    python score_predictions.py --manifest corpus/manifest.json --predictions predictions.json \
        --measured rocKE=out-dense/l1/corpus.csv --measured flyDSL=out-fly/l1/corpus.csv
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import pandas as pd

ENGINE_LABELS = {
    "hipkernel:Gfx950AttentionDense": "rocKE",
    "hipkernel:Gfx942AttentionDense": "rocKE",
    "hipkernel:FlydslAttention": "flyDSL",
    "ASM_SDPA_ENGINE": "AITER",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--predictions", required=True, type=pathlib.Path)
    parser.add_argument("--measured", action="append", required=True, help="LABEL=corpus.csv")
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    regime = {row["name"]: row["regime"] for row in manifest["graphs"]}
    by_benchmark = {row["benchmark"]: row["name"] for row in manifest["graphs"]}

    measured: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for spec in args.measured:
        label, _, path = spec.partition("=")
        frame = pd.read_csv(path)
        for benchmark, group in frame[frame["tflops"].notna()].groupby("benchmark"):
            name = by_benchmark.get(str(benchmark))
            if name:
                measured[name][label] = float(group["tflops"].max())

    predicted: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for row in json.loads(args.predictions.read_text(encoding="utf-8")):
        label = ENGINE_LABELS.get(row["engine"])
        if label and row.get("tflops") is not None:
            predicted[row["graph"]][label] = float(row["tflops"])

    rows = []
    for name, scores in measured.items():
        guesses = predicted.get(name, {})
        shared = set(scores) & set(guesses)
        if len(shared) < 2:
            continue
        measured_best = max(shared, key=lambda label: scores[label])
        predicted_best = max(shared, key=lambda label: guesses[label])
        lost = (scores[measured_best] - scores[predicted_best]) / scores[measured_best]
        rows.append({"graph": name, "regime": regime.get(name, "?"), "engines": len(shared),
                     "measured_winner": measured_best, "predicted_winner": predicted_best,
                     "agree": measured_best == predicted_best, "throughput_lost": lost})

    if not rows:
        print("no graph has two engines with both a prediction and a measurement")
        return 1
    frame = pd.DataFrame(rows)
    agree = int(frame["agree"].sum())
    print(f"contested graphs scored by two or more engines: {len(frame)}")
    print(f"  predicted winner == measured winner : {agree} ({100 * agree / len(frame):.1f}%)")
    wrong = frame[~frame["agree"]]
    if not wrong.empty:
        print(f"  wrong pick                         : {len(wrong)}")
        print(f"  throughput given up when wrong     : mean {100 * wrong['throughput_lost'].mean():.1f}%"
              f", p95 {100 * wrong['throughput_lost'].quantile(0.95):.1f}%"
              f", max {100 * wrong['throughput_lost'].max():.1f}%")
    # The whole-corpus figure a selector delivers: zero where it picks right, the shortfall
    # where it does not. This is what a user feels, unlike the agreement rate.
    print(f"  mean throughput lost over ALL scored : {100 * frame['throughput_lost'].mean():.2f}%")

    print(f"\n{'regime':<26} {'graphs':>7} {'agree':>7}   measured winners")
    for name, group in frame.groupby("regime"):
        spread = ", ".join(f"{k} {v}" for k, v in
                           collections.Counter(group["measured_winner"]).most_common())
        print(f"{name:<26} {len(group):>7} {100 * group['agree'].mean():>6.0f}%   {spread}")

    print(f"\n{'pair':<24} {'graphs':>7} {'agree':>7}")
    for engines, group in frame.groupby(frame.apply(
            lambda row: " vs ".join(sorted({row["measured_winner"], row["predicted_winner"]})), axis=1)):
        print(f"{engines:<24} {len(group):>7} {100 * group['agree'].mean():>6.0f}%")

    if args.report:
        args.report.write_text(frame.to_json(orient="records", indent=1), encoding="utf-8")
        print(f"\nwritten {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
