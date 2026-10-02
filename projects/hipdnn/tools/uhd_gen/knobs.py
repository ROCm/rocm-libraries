# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Which knobs earn the kernels they cost.

Every knob multiplies an AOT build, so each is measured against the best-measured
oracle (RFC 0019.13 §11.2): does it vary, what would pinning it cost (regret and lost
coverage, kept separate), and how few combinations per geometry suffice.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .catalog import candidate_density
from .corpus_io import read_corpus_frame
from .evaluate import regret_of, resolve_grouping

__all__ = [
    "KnobAblation",
    "ValueAblation",
    "add_knob_arguments",
    "analyse_knobs",
    "graph_bound_twins",
    "knob_columns",
    "run_knobs",
]

logger = logging.getLogger(__name__)

#: Prefix of the KMD's variant space, i.e. what an AOT build enumerates.
_KERNEL_PREFIX = "kernel."

#: Roots that are not the problem; the problem root is the op name, so it is found by
#: exclusion (same line as `uhd_gen.dataset.publish._query_columns`).
_RESERVED_PREFIXES = (_KERNEL_PREFIX, "device.")

#: Identity columns that share the prefix without being knobs.
_NOT_KNOBS = frozenset({"kernel"})


def knob_columns(df: pd.DataFrame) -> list[str]:
    """Every `kernel.*` column that is a candidate axis, in corpus order."""
    return [
        c
        for c in df.columns
        if c.startswith(_KERNEL_PREFIX)
        and c not in _NOT_KNOBS
        and not c.endswith(".uid")
    ]


def graph_bound_twins(columns) -> dict[str, str]:
    """Short name -> the problem column carrying it, for a field the matcher binds.

    Matched on the short name: `kernel.seqlen_q` twins `attention_dense.seqlen_q`.
    """
    return {
        column.split(".", 1)[1]: column
        for column in columns
        if "." in column and not column.startswith(_RESERVED_PREFIXES)
    }


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
    #: Varies among one problem's candidates; if False its pin cost is meaningless.
    tunable: bool = True
    #: A problem column shares the short name (matcher-bound, not generator-pinned).
    graph_bound: bool = False
    #: That column's full published name, as the advice must name a real binding.
    bound_as: str | None = None

    @property
    def is_constant(self) -> bool:
        return len(self.values) <= 1

    @property
    def best(self) -> ValueAblation | None:
        """The value that would hurt least if the knob were pinned to it.

        Coverage first: no regret saving makes up for a dropped problem.
        """
        if not self.per_value:
            return None
        return min(
            self.per_value, key=lambda v: (v.uncovered, v.p95_regret, v.mean_regret)
        )

    def to_dict(self) -> dict:
        best = self.best
        return {
            "name": self.name,
            "distinct_values": len(self.values),
            "values": list(self.values),
            "constant": self.is_constant,
            "tunable": self.tunable,
            "graph_bound": self.graph_bound,
            "bound_as": self.bound_as,
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

    # Only a column that varies within a problem is a real choice; otherwise its pin cost
    # is meaningless (orphaned problems just leave the comparison). Non-varying ones are
    # graph-bound if a problem column twins them, else pinned by the pack's generator.
    graph_bound = graph_bound_twins(usable.columns)
    within = usable.groupby(group, dropna=False)
    knobs = []
    for name in knob_columns(usable):
        short = name[len(_KERNEL_PREFIX) :]
        values = sorted(usable[name].dropna().unique().tolist(), key=repr)
        tunable = bool((within[name].nunique(dropna=False) > 1).any())
        ablation = KnobAblation(
            name=name,
            values=values,
            tunable=tunable,
            graph_bound=short in graph_bound,
            bound_as=graph_bound.get(short),
        )
        if len(values) > 1 and tunable:
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
    """How regret falls as knob combinations are added greedily, best-first."""
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
                regret_of(r.restricted, r.oracle, objective)
                for r in served.itertuples()
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
        if uncovered == 0 and mean_regret <= 1e-12:
            break
    return curve


#: Pinning costs below this are measurement noise.
FREE_THRESHOLD = 0.005

#: Above noise but small; worth kernels only if the author's build budget allows.
CHEAP_THRESHOLD = 0.02


def load_importance(manifest_path: str | Path) -> dict[str, dict]:
    """Optional `feature_importance` from a training manifest; ranking never needs it."""
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("no feature importance (%s): %s", manifest_path, error)
        return {}
    return manifest.get("feature_importance") or {}


def rank_knobs(report: dict, importance: dict[str, dict] | None = None) -> list[dict]:
    """Every declared field by descending pin cost, with the decision it implies.

    Matched/pinned fields, fields whose best value orphans problems, and constants are
    never judged on cost: there it does not measure what it seems to.
    """
    importance = importance or {}
    ranked = []
    for knob in report["knobs"]:
        short = knob["name"].removeprefix(_KERNEL_PREFIX)
        imp = importance.get(knob["name"]) or importance.get(short) or {}
        row = {
            "name": knob["name"],
            "short_name": short,
            "distinct_values": knob["distinct_values"],
            "best_value": knob["best_value"],
            "cost": knob["cost_of_pinning"],
            "problems_lost": knob["problems_lost_by_pinning"],
            "tunable": knob.get("tunable", True),
            "graph_bound": knob.get("graph_bound", False),
            "bound_as": knob.get("bound_as"),
            "gain": imp.get("gain"),
            "split": imp.get("split"),
        }
        lost = row["problems_lost"] or 0
        if knob["constant"]:
            row["verdict"] = "CONSTANT"
            only = knob["values"][0] if knob["values"] else "<none>"
            row["advice"] = (
                f"never varies (always {only}); remove from the KMD -- it also blocks "
                f"the model, see RFC 0019 6.3"
            )
        elif not row["tunable"] and row["graph_bound"]:
            row["verdict"] = "MATCHED"
            row["advice"] = (
                f"bound to the graph: all {knob['distinct_values']} values exist across "
                f"the corpus, but the matcher fixes it per problem, so every candidate "
                f"shares one. Not an AOT choice. A model reading it from `$kernel.` "
                f"needs a knob it should not have -- read "
                f"`${row['bound_as'] or row['short_name']}` instead"
            )
        elif not row["tunable"]:
            row["verdict"] = "PINNED"
            row["advice"] = (
                f"the pack builds ONE value per geometry ({knob['distinct_values']} exist "
                f"across the corpus), so the model is never offered the choice and no "
                f"measurement here can say whether it matters. Nothing binds it -- this "
                f"is the generator's decision. Build both values per geometry and "
                f"re-sweep to find out, or drop it from the KMD as unearned"
            )
        elif row["cost"] is None:
            row["verdict"] = "UNMEASURED"
            row["advice"] = "no measurement covered this field"
        elif lost > 0:
            row["verdict"] = "KEEP"
            row["advice"] = (
                f"cannot be pinned: the best value ({row['best_value']}) leaves {lost} "
                f"problem(s) with no kernel at all"
            )
        elif row["cost"] <= FREE_THRESHOLD:
            row["verdict"] = "DROP"
            row["advice"] = (
                f"pinning to {row['best_value']} costs {row['cost']:.2%} and orphans "
                f"nothing -- its kernels are not earning their build"
            )
        elif row["cost"] <= CHEAP_THRESHOLD:
            row["verdict"] = "CHEAP"
            row["advice"] = (
                f"pinning to {row['best_value']} costs {row['cost']:.2%}; worth kernels "
                f"only if the build budget allows"
            )
        else:
            row["verdict"] = "KEEP"
            row["advice"] = (
                f"decides the winner -- pinning to {row['best_value']} costs "
                f"{row['cost']:.2%}"
            )
        ranked.append(row)

    # Non-choices sort below the real choices so they do not bury the ranking.
    def _order(row: dict) -> tuple:
        rank = {"PINNED": 1, "MATCHED": 2, "CONSTANT": 3}.get(row["verdict"], 0)
        return (rank, -(row["cost"] or 0.0), row["name"])

    return sorted(ranked, key=_order)


def format_author_report(
    report: dict, ranked: list[dict], engine: str | None = None
) -> str:
    """The ranking as something a kernel author can act on without reading JSON."""
    lines = [
        f"# Knob value report{f' -- {engine}' if engine else ''}",
        "",
        f"- problems: {report['problems']}",
        f"- measurements: {report['measurements']}",
        f"- target: `{report['target']}` ({report['objective']})",
        f"- grouped by: {', '.join(f'`{c}`' for c in report['grouped_by'])}",
        "",
        "Cost is what pinning the knob to its best single value would lose, p95 across",
        "problems. Gain and splits are what the trained trees did with it -- a second",
        "opinion only: a feature can be split on heavily and still be free to pin,",
        "because predicting time is not the same as changing which candidate wins.",
        "",
        "| knob | values | verdict | cost of pinning | best value | tree gain | splits |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in ranked:
        cost = "--" if row["cost"] is None else f"{row['cost']:.2%}"
        gain = "--" if row["gain"] is None else f"{row['gain']:,.0f}"
        split = "--" if row["split"] is None else f"{row['split']:,}"
        lines.append(
            f"| `{row['short_name']}` | {row['distinct_values']} | **{row['verdict']}** "
            f"| {cost} | {row['best_value']} | {gain} | {split} |"
        )

    actionable = [r for r in ranked if r["verdict"] in ("CONSTANT", "DROP", "PINNED")]
    lines += ["", "## What to change", ""]
    if actionable:
        for row in actionable:
            lines.append(f"- `{row['short_name']}`: {row['advice']}")
    else:
        lines.append("- Nothing: every declared knob varies and earns its kernels.")

    curve = report.get("variant_curve") or []
    if curve:
        lines += [
            "",
            "## How few variants per geometry would do",
            "",
            "| variants | mean regret | p95 |",
            "|---|---|---|",
        ]
        for row in curve[:5]:
            lines.append(
                f"| {row['variants']} | {row['mean_regret']:.2%} | {row['p95_regret']:.2%} |"
            )
    return "\n".join(lines) + "\n"


def add_knob_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        required=True,
        help="benchmark corpus: the published .parquet dataset, a collected "
        ".csv, or .json records -- the same three forms train takes",
    )
    parser.add_argument(
        "--target", default="robustMeanMs", help="timing column to rank on"
    )
    parser.add_argument(
        "--objective",
        default="min",
        choices=("min", "max"),
        help="direction of --target",
    )
    parser.add_argument(
        "--device-column",
        default=None,
        help="column naming the device; joins the problem key so one corpus may span GPUs",
    )
    parser.add_argument(
        "--output", default=None, help="write the full report as JSON here"
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="train_manifest.json, to add what the trees split on as a second opinion",
    )
    parser.add_argument(
        "--author-report",
        default=None,
        help="write the ranking as markdown for the kernel author",
    )
    parser.add_argument(
        "--engine", default=None, help="engine name, for the report title"
    )
    parser.add_argument(
        "--curve-rows",
        type=int,
        default=12,
        help="how many rows of the variant curve to print; the tail is flat and long, "
        "and printing all of it has already pushed the ranking out of a log",
    )


def run_knobs(args: argparse.Namespace) -> int:
    # Same reader as train/evaluate, so `--input` and device-id typing agree.
    try:
        df = read_corpus_frame(Path(args.input))
    except (OSError, ValueError, ImportError) as error:
        logger.error("cannot read corpus %s: %s", args.input, error)
        return 1
    # A deterministic catalog offers no choice at all: say so instead of a table of
    # MATCHED/PINNED rows. Exit 0: "no knob is worth keeping" is a valid answer.
    density = candidate_density(df, device_column=args.device_column)
    if density.deterministic:
        print(f"\nKnob value over {density.problems} problem(s)")
        print(f"  {density.diagnosis()}")
        return 0
    try:
        report = analyse_knobs(df, args.target, args.objective, args.device_column)
    except (KeyError, ValueError) as error:
        logger.error("%s", error)
        return 1

    importance = load_importance(args.manifest) if args.manifest else {}
    ranked = rank_knobs(report, importance)
    report["ranked"] = ranked

    print(
        f"\nKnob value over {report['problems']} problem(s), "
        f"{report['measurements']} measurement(s)"
    )
    print(f"  grouped by: {', '.join(report['grouped_by'])}")
    print(f"  target:     {report['target']} ({report['objective']})\n")

    print("  Fields, most consequential first:")
    # Zero cost with non-zero orphans is not free to pin; this column shows which.
    header = (
        f"    {'field':22} {'values':>6} {'verdict':>9} {'cost':>8} "
        f"{'orphans':>8} {'best':>8}"
    )
    if importance:
        header += f" {'gain':>12} {'splits':>7}"
    print(header)
    for row in ranked:
        cost = "     --" if row["cost"] is None else f"{row['cost']:7.2%}"
        lost = (
            "      --" if row["problems_lost"] is None else f"{row['problems_lost']:8,}"
        )
        line = (
            f"    {row['short_name']:22} {row['distinct_values']:>6} "
            f"{row['verdict']:>9} {cost} {lost} {str(row['best_value']):>8}"
        )
        if importance:
            gain = "          --" if row["gain"] is None else f"{row['gain']:12,.0f}"
            split = "     --" if row["split"] is None else f"{row['split']:7,}"
            line += f" {gain} {split}"
        print(line)

    matched = [r for r in ranked if r["verdict"] in ("MATCHED", "PINNED")]
    if matched:
        print(
            "\n  No AOT choice exists -- every candidate for a problem shares one value:"
        )
        for row in matched:
            cause = "graph-bound" if row["graph_bound"] else "pinned by the pack"
            print(
                f"    {row['short_name']:22} {row['distinct_values']} values, "
                f"one per problem ({cause})"
            )

    print("\n  What to change:")
    actionable = [r for r in ranked if r["verdict"] in ("CONSTANT", "DROP", "PINNED")]
    if actionable:
        for row in actionable:
            print(f"    {row['short_name']:22} {row['advice']}")
    else:
        print("    nothing -- every declared knob varies and earns its kernels")

    curve = report["variant_curve"]
    if curve:
        shown = curve[: max(1, args.curve_rows)]
        print("\n  Variants per geometry, added greedily:")
        print(
            f"    {'#':>3} {'covered':>8} {'uncovered':>10} {'mean':>9} {'p95':>9}  combination"
        )
        for row in shown:
            print(
                f"    {row['variants']:>3} {row['problems_covered']:>8} "
                f"{row['problems_uncovered']:>10} {row['mean_regret']:>8.2%} "
                f"{row['p95_regret']:>8.2%}  {row['added']}"
            )
        if len(curve) > len(shown):
            last = curve[-1]
            print(
                f"    ... {len(curve) - len(shown)} more, to "
                f"{last['variants']} variants at {last['mean_regret']:.2%} mean"
            )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
            handle.write("\n")
        print(f"\n  report: {args.output}")

    if args.author_report:
        with open(args.author_report, "w", encoding="utf-8") as handle:
            handle.write(format_author_report(report, ranked, args.engine))
        print(f"  author report: {args.author_report}")
    return 0
