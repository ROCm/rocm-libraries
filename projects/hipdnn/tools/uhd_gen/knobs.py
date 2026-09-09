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
from pathlib import Path

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

#: The problem namespace. A `kernel.*` field with a twin here was bound by the matcher
#: to the graph; one without was pinned by whoever generated the pack.
_QUERY_PREFIX = "q."
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
    #: Whether the column varies among the candidates of a single problem. False means
    #: no AOT choice exists for it, and its pin cost is meaningless.
    tunable: bool = True
    #: Whether the problem namespace carries the same name. Separates a field the
    #: matcher bound to the graph from one the pack's generator pinned per geometry --
    #: identical in the data, different in what the author can do about it.
    graph_bound: bool = False

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
            "tunable": self.tunable,
            "graph_bound": self.graph_bound,
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

    # A `kernel.*` column is not automatically something a caller tunes. The KMD
    # declares one variant space, and it holds both: fields the matcher binds to the
    # problem (head_size, seqlen_q, dtype -- the kernel was built for that shape) and
    # fields a caller genuinely chooses among the candidates that survive matching
    # (block_m, use_exp2_fast).
    #
    # They are told apart by whether the column varies *within* a problem. A field that
    # cannot offer a choice there is not something an AOT build decides: pinning it has
    # no measurable cost, because the problems it would orphan simply leave the
    # comparison rather than scoring badly in it. A report ranking on cost alone
    # therefore recommends dropping seqlen_q, which means shipping kernels for one
    # sequence length.
    #
    # Two different causes produce that, and they need different answers, so they are
    # distinguished by whether the problem namespace carries the same name:
    #
    #   * `q.seqlen_q` exists beside `kernel.seqlen_q`  -> the matcher binds it to the
    #     graph. Nothing to do; the kernel was built for that shape.
    #   * no `q.waves_per_eu` exists                    -> nothing bound it. The pack's
    #     generator simply chose one value per geometry, so the model was never offered
    #     the choice. That is a decision the author can revisit: build both and re-sweep
    #     to find out whether it matters, or drop it from the KMD as unearned.
    graph_bound = {
        column[len(_QUERY_PREFIX):]
        for column in usable.columns
        if column.startswith(_QUERY_PREFIX)
    }
    within = usable.groupby(group, dropna=False)
    knobs = []
    for name in knob_columns(usable):
        short = name[len(_KERNEL_PREFIX):]
        values = sorted(usable[name].dropna().unique().tolist(), key=repr)
        tunable = bool((within[name].nunique(dropna=False) > 1).any())
        ablation = KnobAblation(
            name=name, values=values, tunable=tunable, graph_bound=short in graph_bound
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


#: Pinning costs below this read as measurement noise, not a real loss. A knob this
#: cheap buys nothing that survives a re-run, so its kernels are not earning their
#: build.
FREE_THRESHOLD = 0.005

#: Above noise but small. Whether it is worth kernels is the author's call, not the
#: tool's: it depends on how much build budget the engine has, which this tool cannot
#: see.
CHEAP_THRESHOLD = 0.02


def load_importance(manifest_path: str | Path) -> dict[str, dict]:
    """Read `feature_importance` from a training manifest, if it carries one.

    Absent is normal -- a manifest written before the field existed, or a report run
    without a model. The ranking never depends on it; it is a second opinion.
    """
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("no feature importance (%s): %s", manifest_path, error)
        return {}
    return manifest.get("feature_importance") or {}


def rank_knobs(report: dict, importance: dict[str, dict] | None = None) -> list[dict]:
    """Every declared field, most consequential first, with the decision it implies.

    Ordering is by what pinning the field would cost, descending, because that is the
    question an AOT build asks: the row at the top is the one whose kernels are buying
    the most, and the row at the bottom is the one to delete first.

    Three things are deliberately never ranked on cost:

    * A **matched** field -- one the matcher binds to the problem's shape, so it does
      not vary among the candidates of any single problem. Its pin cost measures
      nothing, because the problems it would orphan leave the comparison rather than
      scoring badly in it. Ranked on cost, `seqlen_q` reads 0.00% and the report
      recommends shipping kernels for one sequence length.
    * A field whose best value still **orphans problems**. Zero regret over the
      problems it can serve says nothing about the ones it cannot.
    * A **constant**, which costs no kernels at all -- there is only one value to
      build -- but is not harmless: training drops a column that cannot separate
      candidates, and §6.3 then refuses a model whose axes no longer match the
      engine's knobs.
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
                f"needs a knob it should not have -- read `$q.{row['short_name']}` instead"
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

    # Matched fields and constants sort below the real choices: neither is something an
    # AOT build decides, and either one reported above the knob that actually decides
    # the winner would bury the ranking this exists to give.
    def _order(row: dict) -> tuple:
        rank = {"PINNED": 1, "MATCHED": 2, "CONSTANT": 3}.get(row["verdict"], 0)
        return (rank, -(row["cost"] or 0.0), row["name"])

    return sorted(ranked, key=_order)


def format_author_report(report: dict, ranked: list[dict], engine: str | None = None) -> str:
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
        lines += ["", "## How few variants per geometry would do", "", "| variants | mean regret | p95 |", "|---|---|---|"]
        for row in curve[:5]:
            lines.append(
                f"| {row['variants']} | {row['mean_regret']:.2%} | {row['p95_regret']:.2%} |"
            )
    return "\n".join(lines) + "\n"


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
    parser.add_argument("--engine", default=None, help="engine name, for the report title")
    parser.add_argument(
        "--curve-rows",
        type=int,
        default=12,
        help="how many rows of the variant curve to print; the tail is flat and long, "
             "and printing all of it has already pushed the ranking out of a log",
    )


def run_knobs(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.input)
    try:
        report = analyse_knobs(df, args.target, args.objective, args.device_column)
    except (KeyError, ValueError) as error:
        logger.error("%s", error)
        return 1

    importance = load_importance(args.manifest) if args.manifest else {}
    ranked = rank_knobs(report, importance)
    report["ranked"] = ranked

    print(f"\nKnob value over {report['problems']} problem(s), "
          f"{report['measurements']} measurement(s)")
    print(f"  grouped by: {', '.join(report['grouped_by'])}")
    print(f"  target:     {report['target']} ({report['objective']})\n")

    # The ranking first and unconditionally: it is the answer, and everything below is
    # its supporting detail.
    print("  Fields, most consequential first:")
    # `orphans` is not decoration: a zero cost beside a non-zero orphan count is the
    # difference between "free to pin" and "pinning it ships no kernel for those
    # problems at all", and a table without it cannot be checked by its reader.
    header = (f"    {'field':22} {'values':>6} {'verdict':>9} {'cost':>8} "
              f"{'orphans':>8} {'best':>8}")
    if importance:
        header += f" {'gain':>12} {'splits':>7}"
    print(header)
    for row in ranked:
        cost = "     --" if row["cost"] is None else f"{row['cost']:7.2%}"
        lost = "      --" if row["problems_lost"] is None else f"{row['problems_lost']:8,}"
        line = (f"    {row['short_name']:22} {row['distinct_values']:>6} "
                f"{row['verdict']:>9} {cost} {lost} {str(row['best_value']):>8}")
        if importance:
            gain = "          --" if row["gain"] is None else f"{row['gain']:12,.0f}"
            split = "     --" if row["split"] is None else f"{row['split']:7,}"
            line += f" {gain} {split}"
        print(line)

    matched = [r for r in ranked if r["verdict"] in ("MATCHED", "PINNED")]
    if matched:
        print("\n  No AOT choice exists -- every candidate for a problem shares one value:")
        for row in matched:
            cause = "graph-bound" if row["graph_bound"] else "pinned by the pack"
            print(f"    {row['short_name']:22} {row['distinct_values']} values, "
                  f"one per problem ({cause})")

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
        print(f"    {'#':>3} {'covered':>8} {'uncovered':>10} {'mean':>9} {'p95':>9}  combination")
        for row in shown:
            print(f"    {row['variants']:>3} {row['problems_covered']:>8} "
                  f"{row['problems_uncovered']:>10} {row['mean_regret']:>8.2%} "
                  f"{row['p95_regret']:>8.2%}  {row['added']}")
        if len(curve) > len(shown):
            last = curve[-1]
            print(f"    ... {len(curve) - len(shown)} more, to "
                  f"{last['variants']} variants at {last['mean_regret']:.2%} mean")

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
