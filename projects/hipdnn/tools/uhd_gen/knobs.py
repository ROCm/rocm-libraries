# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Which knobs earn the kernels they cost.

An AOT pack builds one kernel per knob combination per geometry, so every knob
multiplies the build. A knob earns that only if varying it changes which kernel is
fastest. This measures that directly, against the same oracle the regret report uses
(RFC 0019.13 §11.2): the best *measured* candidate for a problem.

Three questions, in the order they are worth asking:

  1. **Does the knob vary at all?** A field the pack declares but builds one value of
     costs nothing and teaches nothing. It is still worth naming, because it sits in
     the KMD claiming to be a variant axis and a model may be ranking on it.

  2. **If it were pinned to one value, what would that cost?** For each value, restrict
     the catalog to candidates carrying it and re-ask the oracle question. Two distinct
     failures come out and must not be averaged together: a problem the value still
     serves but more slowly (regret), and a problem it cannot serve at all (coverage
     loss). The second is not a slower kernel; it is no kernel.

  3. **How few kernels per geometry actually suffice?** Greedy over knob combinations,
     reporting the regret curve as variants are added. This is the number the AOT budget
     is spent on, and the one worth arguing about.

Deliberately not a model: it reads measurements and reports what they say. Nothing here
trains, ranks, or predicts.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field

import pandas as pd

from .evaluate import regret_of, resolve_grouping

__all__ = [
    "KnobAblation",
    "ValueAblation",
    "add_knob_arguments",
    "analyse_knobs",
    "knob_columns",
    "run_knobs",
]

logger = logging.getLogger(__name__)

#: Knob columns live under this prefix. `$kernel.*` is the KMD's variant space, which is
#: exactly what an AOT build enumerates; `$q.*` describes the problem and cannot be
#: chosen away.
_KERNEL_PREFIX = "kernel."

#: Identity columns that share the prefix without being knobs.
_NOT_KNOBS = frozenset({"kernel"})


def knob_columns(df: pd.DataFrame) -> list[str]:
    """Every `kernel.*` column that is a candidate axis, in corpus order."""
    return [
        c
        for c in df.columns
        if c.startswith(_KERNEL_PREFIX) and c not in _NOT_KNOBS and not c.endswith(".uid")
    ]


@dataclass
class ValueAblation:
    """What pinning one knob to one value would cost."""

    value: object
    #: Problems with at least one candidate carrying this value.
    covered: int
    #: Problems with none -- the engine would stop serving these entirely.
    uncovered: int
    mean_regret: float
    p50_regret: float
    p95_regret: float
    max_regret: float

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "covered": self.covered,
            "uncovered": self.uncovered,
            "mean_regret": self.mean_regret,
            "p50_regret": self.p50_regret,
            "p95_regret": self.p95_regret,
            "max_regret": self.max_regret,
        }


@dataclass
class KnobAblation:
    """What one knob is worth across the corpus."""

    name: str
    values: list = field(default_factory=list)
    per_value: list[ValueAblation] = field(default_factory=list)

    @property
    def is_constant(self) -> bool:
        return len(self.values) <= 1

    @property
    def best(self) -> ValueAblation | None:
        """The value that would hurt least if the knob were pinned to it.

        Ordered on coverage first: a value that cannot serve a problem is not
        comparable to one that serves it slowly, and no amount of low regret on the
        problems it does cover makes up for the ones it drops.
        """
        if not self.per_value:
            return None
        return min(self.per_value, key=lambda v: (v.uncovered, v.p95_regret, v.mean_regret))

    def to_dict(self) -> dict:
        best = self.best
        return {
            "name": self.name,
            "distinct_values": len(self.values),
            "values": list(self.values),
            "constant": self.is_constant,
            "per_value": [v.to_dict() for v in self.per_value],
            "best_value": None if best is None else best.value,
            "cost_of_pinning": None if best is None else best.p95_regret,
            "problems_lost_by_pinning": None if best is None else best.uncovered,
        }


def _oracle_by_problem(df: pd.DataFrame, group: list[str], target: str, objective: str):
    agg = "min" if objective == "min" else "max"
    return df.groupby(group, dropna=False)[target].agg(agg)


def analyse_knobs(
    df: pd.DataFrame,
    target: str = "robustMeanMs",
    objective: str = "min",
    device_column: str | None = None,
) -> dict:
    """Ablate every knob against the full-catalog oracle."""
    grouping = resolve_grouping(df, device_column)
    group = list(grouping.columns)

    usable = df[df[target].notna() & (df[target] > 0)]
    if usable.empty:
        raise ValueError(f"no rows carry a positive {target}")

    oracle = _oracle_by_problem(usable, group, target, objective)
    total_problems = len(oracle)

    knobs = []
    for name in knob_columns(usable):
        values = sorted(usable[name].dropna().unique().tolist(), key=repr)
        ablation = KnobAblation(name=name, values=values)
        if len(values) > 1:
            for value in values:
                subset = usable[usable[name] == value]
                if subset.empty:
                    ablation.per_value.append(
                        ValueAblation(value, 0, total_problems, 0.0, 0.0, 0.0, 0.0)
                    )
                    continue
                restricted = _oracle_by_problem(subset, group, target, objective)
                joined = oracle.to_frame("oracle").join(
                    restricted.to_frame("restricted"), how="left"
                )
                served = joined[joined["restricted"].notna()]
                regrets = [
                    regret_of(row.restricted, row.oracle, objective)
                    for row in served.itertuples()
                ]
                series = pd.Series(regrets, dtype="float64")
                ablation.per_value.append(
                    ValueAblation(
                        value=value,
                        covered=len(served),
                        uncovered=total_problems - len(served),
                        mean_regret=float(series.mean()) if len(series) else 0.0,
                        p50_regret=float(series.quantile(0.50)) if len(series) else 0.0,
                        p95_regret=float(series.quantile(0.95)) if len(series) else 0.0,
                        max_regret=float(series.max()) if len(series) else 0.0,
                    )
                )
        knobs.append(ablation)

    return {
        "problems": total_problems,
        "measurements": int(len(usable)),
        "grouped_by": group,
        "grouping_degraded": grouping.degraded,
        "target": target,
        "objective": objective,
        "knobs": [k.to_dict() for k in knobs],
        "variant_curve": _variant_curve(usable, group, target, objective, oracle),
    }


def _variant_curve(df, group, target, objective, oracle) -> list[dict]:
    """How regret falls as knob combinations are added, best-first.

    Greedy, not exhaustive: the exhaustive answer is a set-cover over 2^combinations and
    the greedy one is what a build budget is actually spent -- "if I can afford N
    variants, which N, and what do they cost me".

    Reported as a curve rather than a single number because the interesting quantity is
    where it flattens: the first N after which another kernel per geometry buys nothing.
    """
    knobs = knob_columns(df)
    varying = [k for k in knobs if df[k].nunique(dropna=False) > 1]
    if not varying:
        return []

    combo = df[varying].astype(str).agg("|".join, axis=1)
    work = df.assign(_combo=combo)
    combos = sorted(work["_combo"].unique().tolist())

    chosen: list[str] = []
    curve = []
    remaining = set(combos)
    while remaining:
        scored = []
        for candidate in sorted(remaining):
            trial = chosen + [candidate]
            subset = work[work["_combo"].isin(trial)]
            restricted = _oracle_by_problem(subset, group, target, objective)
            joined = oracle.to_frame("oracle").join(
                restricted.to_frame("restricted"), how="left"
            )
            served = joined[joined["restricted"].notna()]
            regrets = [
                regret_of(r.restricted, r.oracle, objective) for r in served.itertuples()
            ]
            series = pd.Series(regrets, dtype="float64")
            scored.append(
                (
                    len(oracle) - len(served),
                    float(series.mean()) if len(series) else 0.0,
                    candidate,
                    float(series.quantile(0.95)) if len(series) else 0.0,
                    len(served),
                )
            )
        uncovered, mean_regret, candidate, p95, served = min(scored)
        chosen.append(candidate)
        remaining.discard(candidate)
        curve.append(
            {
                "variants": len(chosen),
                "added": candidate,
                "problems_covered": served,
                "problems_uncovered": uncovered,
                "mean_regret": mean_regret,
                "p95_regret": p95,
            }
        )
        # The curve is only interesting until it is flat and complete.
        if uncovered == 0 and mean_regret <= 1e-12:
            break
    return curve


def add_knob_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="benchmark corpus CSV")
    parser.add_argument("--target", default="robustMeanMs", help="timing column to rank on")
    parser.add_argument(
        "--objective", default="min", choices=("min", "max"), help="direction of --target"
    )
    parser.add_argument(
        "--device-column",
        default=None,
        help="column naming the device; joins the problem key so one corpus may span GPUs",
    )
    parser.add_argument("--output", default=None, help="write the full report as JSON here")


def run_knobs(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.input)
    try:
        report = analyse_knobs(df, args.target, args.objective, args.device_column)
    except (KeyError, ValueError) as error:
        logger.error("%s", error)
        return 1

    print(f"\nKnob value over {report['problems']} problem(s), "
          f"{report['measurements']} measurement(s)")
    print(f"  grouped by: {', '.join(report['grouped_by'])}")
    print(f"  target:     {report['target']} ({report['objective']})\n")

    constant = [k for k in report["knobs"] if k["constant"]]
    varying = [k for k in report["knobs"] if not k["constant"]]

    if constant:
        print("  Declared but never varied -- no kernels to save, and nothing to rank on:")
        for k in constant:
            only = k["values"][0] if k["values"] else "<none>"
            print(f"    {k['name']:28} always {only}")
        print()

    print("  Cost of pinning each knob to its best single value:")
    print(f"    {'knob':28} {'values':>6} {'best':>10} {'lost':>6} {'mean':>9} {'p95':>9} {'max':>9}")
    for k in varying:
        best = next((v for v in k["per_value"] if v["value"] == k["best_value"]), None)
        if best is None:
            continue
        print(f"    {k['name']:28} {k['distinct_values']:>6} {str(k['best_value']):>10} "
              f"{best['uncovered']:>6} {best['mean_regret']:>8.2%} "
              f"{best['p95_regret']:>8.2%} {best['max_regret']:>8.2%}")

    curve = report["variant_curve"]
    if curve:
        print("\n  Variants per geometry, added greedily:")
        print(f"    {'#':>3} {'covered':>8} {'uncovered':>10} {'mean':>9} {'p95':>9}  combination")
        for row in curve:
            print(f"    {row['variants']:>3} {row['problems_covered']:>8} "
                  f"{row['problems_uncovered']:>10} {row['mean_regret']:>8.2%} "
                  f"{row['p95_regret']:>8.2%}  {row['added']}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
            handle.write("\n")
        print(f"\n  report: {args.output}")
    return 0
