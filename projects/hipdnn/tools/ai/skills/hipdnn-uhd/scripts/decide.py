#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stage V for a UHD: decide, from what the engine answered, whether a model may be installed.

validate_l1.py / validate_l2.py ask the engine about every held-out graph; this turns those
answers into a verdict. Standard library only.

An engine predictor (L1) has these gates, each guarding a failure that is otherwise silent:

* **identity** -- the candidate carries the id the engine declares, and in the tree the
  engine read it was the *only* document carrying it. Anything else and the numbers below
  describe some other model.
* **available** -- with the candidate installed, the engine returns a prediction for the
  graphs it executes. An engine that reports UNAVAILABLE falls back to no L1 at all.
* **metric** -- the candidate's descriptor (`score.metric`) and its answers name the metric
  its id is declared for. The bench echoes the metric it was asked for, so the descriptor is
  what shows a model of another metric.
* **selector_revision** -- the revision the engine reports now is the one the model was
  trained against. The engine refuses a stale model at load time on a user's machine;
  refusing it here is cheaper.
* **accuracy** -- candidate median |error| against measured is no worse than the incumbent
  model's by more than `max_regression` (5% by default when measured). Accuracy for both
  models is reported side by side, over the same graphs.

Where the checkout binds no model of the role for the arch yet (a new engine, or a new arch
of one), there is nothing to regress against. Say so (`--no-incumbent`), never infer it from
the incumbent side
answering nothing -- that is a failure to compare, and stays inconclusive. The accuracy gate
then asks only that the candidate was scored, and the report says `baseline: "none"`.

A catalog ranker (L2) has `available`, `read`, `winner_records` and `regret`
(`decide_catalog`).

    decide.py l1 --rows rows.json --candidate-dir <model dir> --out report.json
        [--max-regression 0.05] [--min-available 1.0] [--no-incumbent]
    decide.py l2 --rows rows.json --candidate-report <eval_report.json>
        [--incumbent-report <eval_report.json>] [--baseline model|native|none]
        [--max-regression 0.0] --out report.json

Exits 0 when every gate passes, 1 otherwise; the report says which gate failed and why.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

AVAILABLE = "available"

#: The default tolerance an L1 is held to against the incumbent model.
DEFAULT_L1_MAX_REGRESSION = 0.05


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    """Rank correlation; None when either side is constant or there are under 3 points."""
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def accuracy(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """(predicted, measured) in one metric -> error summary. Relative, because an L1
    score is compared across engines whose values differ by orders of magnitude."""
    usable = [
        (p, m)
        for p, m in pairs
        if isinstance(p, (int, float))
        and isinstance(m, (int, float))
        and math.isfinite(p)
        and math.isfinite(m)
        and m > 0
    ]
    if not usable:
        return {"graphs": 0}
    relative = [(p - m) / m for p, m in usable]
    absolute = [abs(r) for r in relative]
    return {
        "graphs": len(usable),
        "median_abs_pct_error": 100.0 * statistics.median(absolute),
        "mean_abs_pct_error": 100.0 * sum(absolute) / len(absolute),
        "mean_pct_bias": 100.0 * sum(relative) / len(relative),
        "within_10pct": sum(a <= 0.10 for a in absolute) / len(absolute),
        "spearman": spearman([p for p, _ in usable], [m for _, m in usable]),
    }


def decide(
    document: dict[str, Any],
    *,
    declared_id: str,
    candidate_id: Optional[str],
    candidate_selector_revision: Optional[str],
    min_available: float = 1.0,
    max_regression: Optional[float] = None,
    min_graphs: int = 1,
    incumbent_baseline: bool = True,
    candidate_metric: Optional[str] = None,
) -> dict[str, Any]:
    """The validation report. `passed` is True only if every enabled gate holds."""
    rows = document.get("rows") or []
    measured = bool(document.get("measured"))
    tree = document.get("tree") or {}
    declared_id = declared_id.lower()
    gates: dict[str, dict[str, Any]] = {}

    identity_problems = []
    if (candidate_id or "").lower() != declared_id:
        identity_problems.append(
            f"candidate id {candidate_id} is not the declared {declared_id}"
        )
    if tree.get("candidate_id") and tree["candidate_id"] != declared_id:
        identity_problems.append(f"the tree held candidate {tree['candidate_id']}")
    if not tree.get("unique"):
        identity_problems.append(
            f"documents carrying {declared_id} in the validation tree: "
            f"{tree.get('carrying_declared_id')}; the engine may have read any of them"
        )
    if not tree.get("byte_equal"):
        identity_problems.append(
            "the installed candidate differs from the model under test"
        )
    gates["identity"] = {"passed": not identity_problems, "problems": identity_problems}

    # Graphs the engine executed. Without measurement every graph counts: an engine that
    # declines a graph also has nothing to predict for it, so a pure prediction run can
    # under-report availability, never over-report it.
    executed = [r for r in rows if not measured or (r.get("measured") or {}).get("ok")]
    available = [
        r for r in executed if (r.get("candidate") or {}).get("status") == AVAILABLE
    ]
    fraction = len(available) / len(executed) if executed else 0.0
    unavailable = [
        {
            "graph": r["graph"],
            "status": (r.get("candidate") or {}).get("status"),
            "reason": (r.get("candidate") or {}).get("reason")
            or (r.get("candidate") or {}).get("error"),
        }
        for r in executed
        if r not in available
    ][:20]
    gates["available"] = {
        "passed": len(executed) >= min_graphs and fraction >= min_available,
        "executed_graphs": len(executed),
        "available_graphs": len(available),
        "fraction": fraction,
        "required": min_available,
        "min_graphs": min_graphs,
        "unavailable_sample": unavailable,
    }

    # RFC 0019 §11.4: an answer names its metric, and one in another metric is not
    # converted -- it is the wrong answer. A candidate that answers in a metric other than
    # the one its engine declares its id for has been bound where no request will use it.
    # An answer that names no metric cannot show it is in the declared one, so it fails too.
    metric = document.get("metric") or "tflops"
    answered = sorted(
        {str((r.get("candidate") or {}).get("metric")) for r in available}
    )
    wrong = [m for m in answered if m not in ("None", metric)]
    unnamed = sum((r.get("candidate") or {}).get("metric") is None for r in available)
    metric_problems = []
    # The bench names the metric it was asked for, so the answers alone cannot show the
    # model's metric; the candidate's descriptor declares it (`score.metric`).
    if candidate_metric is not None and candidate_metric != metric:
        metric_problems.append(
            f"the candidate model predicts {candidate_metric}, but the engine declares "
            f"this id for {metric}"
        )
    if wrong:
        metric_problems.append(f"candidate answered in {wrong}, not {metric}")
    if unnamed:
        metric_problems.append(f"{unnamed} candidate answer(s) named no metric")
    gates["metric"] = {
        "passed": not metric_problems,
        "declared": metric,
        "answered": [m for m in answered if m != "None"],
        "problems": metric_problems,
    }

    revisions = sorted(
        {(r.get("candidate") or {}).get("selector_revision") for r in available}
        - {None}
    )
    revision_problems = []
    if not candidate_selector_revision:
        revision_problems.append(
            "the candidate records no trained_against.selector_revision"
        )
    elif not revisions:
        revision_problems.append(
            "no prediction reported the engine's selector_revision"
        )
    elif revisions != [candidate_selector_revision]:
        revision_problems.append(
            f"engine reports {revisions}; the candidate was trained against "
            f"{candidate_selector_revision}"
        )
    gates["selector_revision"] = {
        "passed": not revision_problems,
        "engine_reports": revisions,
        "trained_against": candidate_selector_revision,
        "problems": revision_problems,
    }

    metrics: dict[str, Any] = {}
    if measured:
        # Each model against the run it predicts. A descriptor engine's candidate L1 predicts the
        # selection its L2 makes, the incumbent one the selection the build makes, and validate_l1.py
        # measures both (`measured_incumbent`); an opaque engine has one selection for both.
        def truth(row: dict[str, Any], side: str) -> Optional[float]:
            own = row.get("measured_incumbent") if side == "incumbent" else None
            if own is not None:
                return own.get("value") if own.get("ok") else None
            return row["measured"]["value"]

        def scored(side: str, among: list) -> dict[str, Any]:
            return accuracy(
                [
                    ((r.get(side) or {}).get("value"), truth(r, side))
                    for r in among
                    if (r.get(side) or {}).get("status") == AVAILABLE
                    and truth(r, side) is not None
                ]
            )

        for side in ("candidate", "incumbent"):
            metrics[side] = scored(side, executed)
        # The comparison on the graphs both models answered.
        both = [
            r
            for r in executed
            if (r.get("candidate") or {}).get("status") == AVAILABLE
            and (r.get("incumbent") or {}).get("status") == AVAILABLE
            and truth(r, "incumbent") is not None
        ]
        metrics["paired"] = {
            side: scored(side, both) for side in ("candidate", "incumbent")
        }
        metrics["incumbent_truth"] = (
            "its own selection"
            if any("measured_incumbent" in r for r in rows)
            else "the candidate's selection"
        )
        if not incumbent_baseline:
            # No incumbent to pair with: the candidate on every graph it answered.
            metrics["paired"] = {
                "candidate": metrics["candidate"],
                "incumbent": None,
                "baseline": "none",
            }

    if max_regression is not None:
        paired = metrics.get("paired") or {}
        new = (paired.get("candidate") or {}).get("median_abs_pct_error")
        old = (paired.get("incumbent") or {}).get("median_abs_pct_error")
        if not measured:
            gate = {
                "passed": False,
                "problems": ["max_regression needs measured graphs"],
            }
        elif not incumbent_baseline:
            scored_graphs = (paired.get("candidate") or {}).get("graphs") or 0
            gate = {
                "passed": new is not None and scored_graphs >= min_graphs,
                "baseline": "none",
                "candidate_median_abs_pct_error": new,
                "scored_graphs": scored_graphs,
                "problems": (
                    []
                    if new is not None and scored_graphs >= min_graphs
                    else ["the candidate predicted none of the measured graphs"]
                ),
            }
        elif old is None:
            gate = {
                "passed": False,
                "problems": [
                    "the incumbent model could not be scored on paired graphs; regression is inconclusive"
                ],
            }
        else:
            allowed = old * (1.0 + max_regression)
            gate = {
                "passed": new is not None and new <= allowed,
                "candidate_median_abs_pct_error": new,
                "incumbent_median_abs_pct_error": old,
                "allowed": allowed,
                "problems": (
                    []
                    if new is not None and new <= allowed
                    else [
                        f"candidate median error {new} exceeds {allowed:.3f} "
                        f"(incumbent {old:.3f} + {100 * max_regression:.0f}%)"
                    ]
                ),
            }
        gates["accuracy"] = gate

    return {
        "schema": "hipdnn_uhd.validation_report/1",
        "passed": all(g["passed"] for g in gates.values()),
        "gates": gates,
        "metrics": metrics,
        "graphs": len(rows),
        "measured": measured,
        "metric": metric,
        "declared_id": declared_id,
        "baseline": "incumbent" if incumbent_baseline else "none",
    }


#: Absolute slack on mean top-1 regret before a candidate counts as worse than the incumbent
#: ranker: regret is a fraction of the best measured time, and two rankers within this of
#: each other are within measurement noise of the same choices.
CATALOG_REGRET_SLACK = 0.005


def decide_catalog(
    rows_doc: dict[str, Any],
    *,
    candidate_report: Optional[dict[str, Any]],
    incumbent_report: Optional[dict[str, Any]],
    min_available: float = 1.0,
    max_regression: Optional[float] = None,
    baseline_kind: Optional[str] = None,
) -> dict[str, Any]:
    """The verdict on a catalog ranker (L2).

    Four gates:

    * `available` -- the engine ran under the candidate tree on at least `min_available` of the
      held-out graphs it runs under the root as it stands. A graph that tree cannot run either
      is the engine's failure, not the candidate's: it is reported (`engine_failures`) and left
      out, and an engine that runs none of them fails the gate, since then nothing was tested;
    * `read` -- the runtime did not say, on any graph, that it is not using the model. A refused
      ranker still lets the plan build (it ranks by priority, then id), so this is the only
      evidence a promoted ranker is the one choosing;
    * `winner_records` -- no run had its catalog ordered by a benchmark winner record (the
      runtime says so when one does): a record takes precedence over any ranker, and a pass it
      decided says nothing about the model. validate_l2.py gives every run an empty cache, so
      this catches a record the build itself carries rather than one an earlier run wrote;
    * `regret` -- on a held-out catalog collection (every candidate timed), the candidate's mean
      top-1 regret is no worse than the incumbent ranker's by more than `max_regression` (relative)
      plus a noise slack. Native code requires paired actual GPU times; missing or failed
      baseline evaluation is inconclusive and cannot pass. With no incumbent ranker at all
      (`baseline_kind="none"`, decided from the build) the candidate need only be scored.
    """
    rows = rows_doc.get("rows") or []
    runnable = [r for r in rows if (r.get("incumbent") or {}).get("ok")]
    ran = [r for r in runnable if (r.get("candidate") or {}).get("ok")]

    # A refusal both runs report is about another engine's model in the shared root; only one
    # the candidate run alone reports is about the candidate.
    def candidate_only(row: dict) -> list[str]:
        theirs = set((row.get("incumbent") or {}).get("refusals") or [])
        return [
            line
            for line in (row.get("candidate") or {}).get("refusals") or []
            if line not in theirs
        ]

    refused = [r for r in rows if candidate_only(r)]
    gates: dict[str, Any] = {}
    fraction = len(ran) / len(runnable) if runnable else 0.0
    passed = bool(runnable) and fraction >= min_available
    problems = []
    if not runnable:
        problems.append(
            f"the engine ran none of the {len(rows)} held-out graph(s) even under the "
            f"root as it stands; nothing about the candidate was tested"
        )
    elif not passed:
        problems.append(
            f"{len(runnable) - len(ran)} of {len(runnable)} graph(s) the root as it stands "
            f"runs failed under the candidate tree"
        )
    gates["available"] = {
        "passed": passed,
        "fraction": fraction,
        "graphs": len(rows),
        "runnable": len(runnable),
        "ran": len(ran),
        "engine_failures": len(rows) - len(runnable),
        "problems": problems,
    }
    gates["read"] = {
        "passed": not refused,
        "problems": [
            f"{Path(r['graph']).name}: {candidate_only(r)[0]}" for r in refused[:5]
        ],
    }
    records = rows_doc.get("winner_records") or {}
    gates["winner_records"] = {
        "passed": not any(records.values()),
        "found": records,
        "problems": (
            []
            if not any(records.values())
            else [
                "benchmark winner records decided some selections; the pass "
                "is inconclusive about the ranker"
            ]
        ),
    }

    def mean_regret(report: Optional[dict[str, Any]]) -> Optional[float]:
        return (((report or {}).get("metrics") or {}).get("top1_regret") or {}).get(
            "mean"
        )

    baseline_kind = baseline_kind or (
        "model" if incumbent_report is not None else "missing"
    )
    new, old = mean_regret(candidate_report), mean_regret(incumbent_report)
    valid_new = isinstance(new, (int, float)) and math.isfinite(new) and new >= 0
    valid_old = isinstance(old, (int, float)) and math.isfinite(old) and old >= 0
    if not valid_new:
        gate = {
            "passed": False,
            "problems": ["the candidate was not scored on the held-out catalog"],
        }
    elif baseline_kind == "native":
        # Native scoring code cannot be evaluated offline. Its actual paired GPU timings
        # are the baseline, not the absence of an offline report.
        ratios = []
        for row in ran:
            candidate = row["candidate"].get("avgTimeMs")
            incumbent = row["incumbent"].get("avgTimeMs")
            if all(
                isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                for v in (candidate, incumbent)
            ):
                ratios.append(candidate / incumbent)
        ratio = sum(ratios) / len(ratios) if ratios else None
        allowed = 1.0 + (max_regression or 0.0) + CATALOG_REGRET_SLACK
        complete = bool(ran) and len(ratios) == len(ran)
        gate = {
            "passed": complete and ratio <= allowed,
            "baseline": "native",
            "paired_graphs": len(ratios),
            "candidate_mean_regret": new,
            "mean_time_ratio": ratio,
            "allowed_time_ratio": allowed,
            "problems": (
                [
                    "native comparison requires positive, finite paired GPU timings "
                    "for every graph"
                ]
                if not complete
                else (
                    []
                    if ratio <= allowed
                    else [
                        f"candidate mean paired time ratio {ratio:.4f} exceeds {allowed:.4f}"
                    ]
                )
            ),
        }
    elif baseline_kind == "none":
        gate = {
            "passed": True,
            "baseline": "none",
            "candidate_mean_regret": new,
            "problems": [],
        }
    elif baseline_kind != "model" or not valid_old:
        gate = {
            "passed": False,
            "candidate_mean_regret": new,
            "baseline": baseline_kind,
            "problems": [
                "the incumbent ranker is absent or could not be evaluated; "
                "regression is inconclusive"
            ],
        }
    else:
        allowed = old * (1.0 + (max_regression or 0.0)) + CATALOG_REGRET_SLACK
        gate = {
            "passed": new <= allowed,
            "candidate_mean_regret": new,
            "incumbent_mean_regret": old,
            "allowed": allowed,
            "problems": (
                []
                if new <= allowed
                else [
                    f"candidate mean top-1 regret {new:.4f} exceeds {allowed:.4f} "
                    f"(incumbent {old:.4f})"
                ]
            ),
        }
    gates["regret"] = gate
    return {
        "schema": "hipdnn_uhd.catalog_validation_report/1",
        "passed": all(g["passed"] for g in gates.values()),
        "gates": gates,
        "metrics": {
            "candidate": candidate_report and candidate_report.get("metrics"),
            "incumbent": incumbent_report and incumbent_report.get("metrics"),
        },
        "graphs": len(rows),
        "baseline": baseline_kind,
        "install": rows_doc.get("install"),
    }


def comparison_warning(role: str, metrics: Optional[dict[str, Any]]) -> Optional[str]:
    """What a user installing the model should hear about it against the incumbent model,
    from the report's `metrics`; None when the candidate is no worse.

    Worse at all is said, not only worse past the gate's tolerance: the user is choosing to
    replace that model. A comparison that could not be made is said too -- the incumbent model
    answered nothing, or there is none -- because a pass then says nothing about regression.
    """
    if not metrics:
        return None
    if metrics.get("baseline") == "none":
        return (
            "not compared: the checkout binds no model of this role for this arch yet"
        )
    if role == "sort_kernel_catalog":
        if metrics.get("baseline") == "native":
            ratio = (metrics.get("paired") or {}).get("mean_time_ratio")
            if ratio is None:
                return "native ranker comparison is inconclusive: no paired GPU timings"
            return (
                f"slower than native ranker: mean paired time ratio {ratio:.4f}"
                if ratio > 1
                else None
            )
        new = (metrics.get("candidate") or {}).get("mean")
        old = (metrics.get("incumbent") or {}).get("mean")
        if new is None:
            return None
        if old is None:
            return "not compared with the incumbent ranker: there was none to score"
        if new > old:
            return (
                f"worse than the incumbent ranker: mean top-1 regret {new:.4f} vs "
                f"{old:.4f}"
            )
        return None
    new = (metrics.get("candidate") or {}).get("median_abs_pct_error")
    old = (metrics.get("incumbent") or {}).get("median_abs_pct_error")
    if new is None:
        return None
    if old is None:
        return (
            "not compared with the incumbent model: it answered none of the held-out "
            "graphs the candidate did"
        )
    if new > old:
        return (
            f"less accurate than the incumbent model: median |error| {new:.2f}% vs "
            f"{old:.2f}%"
        )
    return None


def _candidate(model_dir: Path) -> dict[str, Any]:
    descriptors = sorted(model_dir.glob("*.uhd.json"))
    if len(descriptors) != 1:
        raise SystemExit(
            f"{model_dir} must hold exactly one *.uhd.json, found {len(descriptors)}"
        )
    return json.loads(descriptors[0].read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="role", required=True)
    l1 = sub.add_parser("l1", help="an engine predictor")
    l1.add_argument("--rows", type=Path, required=True)
    l1.add_argument("--candidate-dir", type=Path, required=True)
    l1.add_argument("--min-available", type=float, default=1.0)
    l1.add_argument(
        "--min-graphs",
        type=int,
        default=1,
        help="fewest graphs the engine must execute (and, with no incumbent, the candidate "
        "must predict) for the verdict to count",
    )
    l1.add_argument("--max-regression", type=float, default=DEFAULT_L1_MAX_REGRESSION)
    l1.add_argument(
        "--no-incumbent",
        action="store_true",
        help="the checkout binds no model of this role for this arch",
    )
    l1.add_argument("--out", type=Path, required=True)
    l2 = sub.add_parser("l2", help="a catalog ranker")
    l2.add_argument("--rows", type=Path, required=True)
    l2.add_argument("--candidate-report", type=Path, required=True)
    l2.add_argument("--incumbent-report", type=Path)
    l2.add_argument("--baseline", choices=("model", "native", "none"))
    l2.add_argument("--min-available", type=float, default=1.0)
    l2.add_argument("--max-regression", type=float)
    l2.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    if args.role == "l1":
        candidate = _candidate(args.candidate_dir)
        report = decide(
            rows,
            declared_id=rows["declared_id"],
            candidate_id=candidate.get("id"),
            candidate_selector_revision=(candidate.get("trained_against") or {}).get(
                "selector_revision"
            ),
            min_available=args.min_available,
            max_regression=args.max_regression if rows.get("measured") else None,
            min_graphs=args.min_graphs,
            incumbent_baseline=not args.no_incumbent,
            candidate_metric=(candidate.get("score") or {}).get("metric"),
        )
        note = comparison_warning("predict_engine", report["metrics"].get("paired"))
    else:
        load = lambda path: (
            json.loads(path.read_text(encoding="utf-8")) if path else None
        )
        report = decide_catalog(
            rows,
            candidate_report=load(args.candidate_report),
            incumbent_report=load(args.incumbent_report),
            min_available=args.min_available,
            max_regression=args.max_regression,
            baseline_kind=args.baseline,
        )
        regret = report["gates"]["regret"]
        note = comparison_warning(
            "sort_kernel_catalog",
            {
                "baseline": report["baseline"],
                "candidate": {"mean": regret.get("candidate_mean_regret")},
                "incumbent": {"mean": regret.get("incumbent_mean_regret")},
                "paired": {"mean_time_ratio": regret.get("mean_time_ratio")},
            },
        )
    report["comparison"] = note
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "gates": {k: g["passed"] for k, g in report["gates"].items()},
                "comparison": note,
                "report": str(args.out),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
