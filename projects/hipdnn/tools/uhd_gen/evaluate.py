#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Score a trained UHD against the best kernel that was actually measured.

RFC 0019.13 §11.2. Training reports RMSE on `log1p(target)`; the runtime decision is
*pick the best kernel for this problem*, and RMSE can improve while that decision gets
worse. This module computes the metrics that answer the decision question directly:

    top-1 regret     how much slower the model's pick is than the oracle's
    regret tail      the fraction of problems where that shortfall exceeds 5%
    top-k recall     how often the oracle is in the model's top k
    per-regime       the same regret, broken out, because an aggregate hides a model
                     that is excellent on the dense middle and useless on the edges

The oracle `v*(p)` is the best *measured* candidate for a problem -- argmin of the
target under `objective: min`, argmax under `max`. Regret is measured in the target
metric, never in rank position (§11.2): ranks are meaningless between variants whose
timings overlap within noise, and choosing the second of two indistinguishable kernels
costs nothing and MUST NOT be scored as an error.

Everything here is computed on a held-out slice split BY PROBLEM (§5.6.4 "folds are by
problem"). A row-wise split would put some of a problem's candidates in training and
the rest in evaluation; the evaluation-side oracle would then be the best of a subset,
the model's pick would frequently be that same row, and the reported regret would be a
number far better than the truth. The split therefore moves whole problems.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Report format identity. Written into every report so a consumer can tell one
#: vintage from another rather than guessing from which keys happen to be present.
REPORT_SCHEMA = "uhd_gen.eval_report/1"

#: §11.2: "fraction of problems with regret > 5%".
DEFAULT_REGRET_TAIL_THRESHOLD = 0.05

#: §11.2: "k = 1, 3, 5".
TOP_K_VALUES = (1, 3, 5)

DEFAULT_EVAL_FRACTION = 0.2
DEFAULT_SEED = 0

#: The device identity column of the §8.3 envelope: lowercase hex of `DeviceKey::hash()`,
#: the device half of the winner key, exactly as `benchmark` is the graph half. Dotless,
#: so it is an envelope column and not a feature; the `device.*` feature columns (device
#: properties) are a different thing and are NOT part of problem identity.
DEVICE_COLUMN = "device"

#: The graph-content hash. Half of a problem's identity on its own, all of it on a
#: corpus collected before the `device` column existed.
BENCHMARK_COLUMN = "benchmark"

#: Where a corpus might carry §5.2's regime label. Discovered rather than required: a
#: sweep exported by `export-benchmarks` carries no regime at all today, and demanding
#: one would make `evaluate` unusable on every corpus that exists.
REGIME_COLUMN_CANDIDATES = ("regime", "corpus_regime", "q.regime", "problem.regime")

#: Two candidates whose measured times differ by less than this are treated as the same
#: choice for top-k recall. See `_tie_mask` for why recall needs it and regret does not.
DEFAULT_TIE_REL_TOLERANCE = 0.01

#: Half-width, in standard errors, of the noise band used for the same purpose when the
#: target is a millisecond timing column and `stddevMs`/`iters` are present.
DEFAULT_TIE_SIGMA = 2.0

#: Targets whose units are milliseconds, and so are comparable with `stddevMs`. Applying
#: a millisecond spread to a TFLOPS target would be a units error that silently widens
#: or narrows the tie band, so the noise rule is restricted to these by name.
MILLISECOND_TARGETS = frozenset({"minTimeMs", "avgTimeMs", "robustMeanMs"})

#: Float slack for the non-negativity assertion. Regret is a ratio of two measured
#: values; when the pick IS the oracle the ratio is exactly 1.0 in binary, but a
#: tolerance costs nothing and a spurious failure would cost a run.
_REGRET_EPSILON = 1e-9

#: A scorer maps candidate rows to a predicted score in the target's direction: higher
#: is better under `objective: max`, lower under `min`. Only the ORDER it induces is
#: used -- regret is computed from measured values, never predicted ones -- so a
#: monotone reparameterisation of the score cannot change any number in the report.
Scorer = Callable[[pd.DataFrame], np.ndarray]


class ObjectiveDirectionError(RuntimeError):
    """A regret came out negative, which is arithmetically impossible under §11.2.

    Both directions are defined so that regret is non-negative. A negative one means
    the oracle was chosen under the opposite direction to the one the metric applied,
    i.e. `objective` is backwards -- and every number in the report is then inverted.
    Raised rather than clamped: a report that prints wrong-way-round regret as a
    plausible small positive number is worse than no report.
    """


@dataclass(frozen=True)
class Grouping:
    """How rows were collapsed into problems, and whether that was the full identity."""

    columns: tuple[str, ...]
    degraded: bool
    detail: str


@dataclass(frozen=True)
class Split:
    """A reproducible partition of problems into training and evaluation slices."""

    seed: int
    fraction: float
    train_problems: tuple[tuple[str, ...], ...]
    eval_problems: tuple[tuple[str, ...], ...]

    @property
    def method(self) -> str:
        return "full_corpus" if self.fraction >= 1.0 else "group_holdout_by_problem"


@dataclass
class ProblemResult:
    """One problem's contribution to every metric in the report."""

    key: tuple[str, ...]
    regime: str | None
    candidates: int
    oracle_value: float
    picked_value: float
    regret: float
    #: 0-based position of the oracle in the model's ranking.
    oracle_rank: int
    #: Best position, in the same ranking, of any candidate indistinguishable from the
    #: oracle. Equals `oracle_rank` when nothing ties with it.
    tied_rank: int
    tied_candidates: int


@dataclass
class Exclusions:
    """Rows and problems that could not contribute, counted by reason.

    §5.6.3 warns that dropping a configuration from the evaluation slice removes it
    from the oracle -- the corruption the prune of §5.5 is otherwise warned about. So
    nothing is dropped silently here, and what IS dropped is only ever a row that
    carries no measurement to be best with.
    """

    invalid_rows: int = 0
    missing_target_rows: int = 0
    problems_no_measured_candidate: int = 0
    problems_single_candidate: int = 0
    problems_non_positive_oracle: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "invalid_rows": self.invalid_rows,
            "missing_target_rows": self.missing_target_rows,
            "problems_no_measured_candidate": self.problems_no_measured_candidate,
            "problems_single_candidate": self.problems_single_candidate,
            "problems_non_positive_oracle": self.problems_non_positive_oracle,
            "policy": (
                "Only rows that carry no usable measurement are dropped: is_valid=False "
                "(a candidate that never ran has no time and cannot be the best) and "
                "rows whose target is empty or non-numeric. Every measured candidate of "
                "an evaluated problem stays in the oracle set, because dropping one "
                "would corrupt the oracle (RFC 0019.13 §5.6.3). Problems left with one "
                "measured candidate are excluded from the metrics rather than scored as "
                "regret 0: with nothing to choose between, a correct pick is not "
                "evidence and averaging it in flatters the model."
            ),
        }


@dataclass
class EvaluationResult:
    """The report, plus the per-problem rows behind it for tests and debugging."""

    report: dict[str, Any]
    problems: list[ProblemResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Problem identity
# --------------------------------------------------------------------------------------


def _blank(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return text.eq("") | text.str.lower().isin({"nan", "none"})


def resolve_grouping(df: pd.DataFrame, device_column: str | None = None) -> Grouping:
    """Decide what identifies a problem in this corpus.

    A problem is `(benchmark, device)`. The same graph on two GPUs is two problems with
    two different best kernels, and grouping on `benchmark` alone would put both GPUs'
    candidates in one pool: the oracle becomes "fastest kernel on the faster card", and
    every problem from the slower card is charged a regret it never had any way to
    avoid. The resulting number is not a smaller version of the truth, it is a
    different quantity.

    Corpora collected before the `device` column existed cannot be grouped that way, so
    this degrades to `benchmark` alone -- and says so loudly everywhere the number is
    shown, because the degraded figure is the conflated one described above.
    """
    if BENCHMARK_COLUMN not in df.columns:
        raise ValueError(
            f"corpus has no {BENCHMARK_COLUMN!r} column, so rows cannot be grouped into "
            "problems at all; regret is undefined without problem identity"
        )

    column = device_column or DEVICE_COLUMN
    if column not in df.columns:
        if device_column is not None:
            raise ValueError(
                f"--device-column {device_column!r} is not a column of this corpus "
                f"(columns: {', '.join(map(str, df.columns))})"
            )
        return Grouping(
            (BENCHMARK_COLUMN,),
            True,
            f"no {DEVICE_COLUMN!r} column in this corpus (it predates the device "
            "identity field in the §8.3 envelope)",
        )

    blanks = _blank(df[column])
    if blanks.all():
        return Grouping(
            (BENCHMARK_COLUMN,),
            True,
            f"the {column!r} column is present but empty on all {len(df)} row(s) (the "
            "sweep that produced this corpus logged no device identity)",
        )
    if blanks.any():
        return Grouping(
            (BENCHMARK_COLUMN, column),
            False,
            f"{int(blanks.sum())} of {len(df)} row(s) carry no device identity; those "
            "group under the empty device id, which is its own problem bucket and is "
            "NOT merged with the identified rows",
        )
    return Grouping(
        (BENCHMARK_COLUMN, column),
        False,
        f"{df[column].nunique()} distinct device(s) in the corpus",
    )


def problem_keys(df: pd.DataFrame, grouping: Grouping) -> pd.Series:
    """A hashable problem key per row, as a tuple of the grouping columns' values."""
    columns = [df[name].astype(str).str.strip() for name in grouping.columns]
    return pd.Series(list(zip(*columns)), index=df.index, dtype=object)


# --------------------------------------------------------------------------------------
# The split
# --------------------------------------------------------------------------------------


def _problem_digest(key: Sequence[str], seed: int) -> str:
    payload = "\x1f".join(str(part) for part in key) + f"|seed={seed}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_problems(
    keys: Iterable[tuple[str, ...]], fraction: float, seed: int
) -> Split:
    """Hold out `fraction` of the PROBLEMS, reproducibly.

    Assignment is by a seeded hash of the problem key, not by shuffling row order, so
    the same corpus and seed give the same slice no matter what order the rows arrived
    in or which rows were appended since -- a regret figure that moves because the log
    was concatenated differently is not comparable round over round (§5.6.3).

    The unit is the problem. Splitting rows would scatter one problem's candidates
    across both sides; see this module's docstring for why the number that comes out of
    that is not the one anyone wants.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"--eval-fraction must be in (0, 1]; got {fraction}")

    ordered = sorted({tuple(key) for key in keys})
    if not ordered:
        return Split(seed, fraction, (), ())

    ranked = sorted(ordered, key=lambda key: _problem_digest(key, seed))
    if fraction >= 1.0:
        count = len(ranked)
    else:
        # At least one problem, always: a fraction that rounds to zero on a small
        # corpus would produce an empty slice and a report full of nulls, which reads
        # like a broken tool rather than like "your corpus is too small to hold out
        # from". A one-problem slice is visibly imprecise instead, and the problem
        # count is in the report so the imprecision is legible (§5.6.3).
        count = max(1, round(fraction * len(ranked)))
    held_out = set(ranked[:count])
    return Split(
        seed,
        fraction,
        tuple(key for key in ordered if key not in held_out),
        tuple(key for key in ordered if key in held_out),
    )


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def regret_of(picked: float, oracle: float, objective: str) -> float:
    """§11.2's top-1 regret: the fractional shortfall of `picked` against `oracle`."""
    if objective == "min":
        value = picked / oracle - 1.0
    elif objective == "max":
        value = 1.0 - picked / oracle
    else:
        raise ValueError(f"objective must be 'min' or 'max'; got {objective!r}")
    if value < -_REGRET_EPSILON:
        raise ObjectiveDirectionError(
            f"regret came out negative ({value:.6g}) with objective={objective!r}: the "
            f"model's pick measured {picked:g} against an oracle of {oracle:g}, which "
            "under this direction means the 'oracle' is not the best candidate. The "
            "objective is almost certainly backwards -- check the descriptor's "
            "`objective` field, or pass --objective explicitly. Every regret and "
            "recall figure would be inverted, so this fails instead of printing."
        )
    return max(value, 0.0)


def _tie_mask(
    values: np.ndarray,
    oracle_position: int,
    objective: str,
    rel_tolerance: float,
    sigma: float,
    stddev: np.ndarray | None,
    iters: np.ndarray | None,
) -> np.ndarray:
    """Which candidates are indistinguishable from the oracle.

    §11.2 measures regret in the target metric precisely so that picking one of two
    statistically indistinguishable kernels costs nothing. Regret gets that for free:
    two candidates a fraction of a percent apart produce a regret a fraction of a
    percent from zero. Top-k recall does NOT -- it is a rank test, and it scores the
    second of two coin-flip-equivalent kernels as an outright miss, so a model that is
    behaving perfectly can post a recall@1 of 0.5 on a corpus of near-ties.

    Two candidates count as tied here when EITHER holds:

    - the regret of choosing one over the other is within `rel_tolerance`. Unit-free,
      works for any target and either direction, and it is the same quantity the regret
      metric already reports -- so "tied" means exactly "costs less than 1%", which is
      a claim a reader can check against the regret column;
    - their measurement intervals overlap within `sigma` standard errors. Only applied
      when the target is a millisecond timing column, because `stddevMs` is in
      milliseconds and comparing it against a TFLOPS target would be a units error.
      For `avgTimeMs` the standard error is exactly `stddevMs/sqrt(iters)`; for
      `minTimeMs` and `robustMeanMs` the sample spread is only a scale for the noise
      rather than that estimator's own error, so this band is approximate -- and
      deliberately so, since the alternative is to have no noise notion at all for the
      §8.5 default statistic.

    The report carries strict recall alongside the tie-aware one, so nothing is hidden
    by this choice: a reader who distrusts the tolerance can read the strict column.
    """
    oracle_value = float(values[oracle_position])
    if objective == "min":
        cost = values / oracle_value - 1.0
    else:
        cost = 1.0 - values / oracle_value
    tied = cost <= rel_tolerance

    if stddev is not None and iters is not None:
        counts = np.where(np.isfinite(iters) & (iters > 0), iters, 1.0)
        spread = np.where(np.isfinite(stddev) & (stddev >= 0), stddev, 0.0)
        standard_error = spread / np.sqrt(counts)
        band = sigma * np.sqrt(standard_error**2 + standard_error[oracle_position] ** 2)
        tied = tied | (np.abs(values - oracle_value) <= band)

    tied[oracle_position] = True

    return tied


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")


def _summarise(regrets: list[float]) -> dict[str, float | None]:
    if not regrets:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(regrets)),
        "p50": _percentile(regrets, 50),
        "p95": _percentile(regrets, 95),
        "max": float(np.max(regrets)),
    }


# --------------------------------------------------------------------------------------
# The evaluation itself
# --------------------------------------------------------------------------------------


def evaluate_corpus(
    df: pd.DataFrame,
    scorer: Scorer,
    *,
    target: str,
    objective: str,
    grouping: Grouping | None = None,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    seed: int = DEFAULT_SEED,
    regime_column: str | None = None,
    tie_rel_tolerance: float = DEFAULT_TIE_REL_TOLERANCE,
    tie_sigma: float = DEFAULT_TIE_SIGMA,
    regret_tail_threshold: float = DEFAULT_REGRET_TAIL_THRESHOLD,
    device_column: str | None = None,
) -> EvaluationResult:
    """Compute §11.2's metrics for `scorer` over the held-out slice of `df`."""
    if target not in df.columns:
        raise ValueError(f"target column {target!r} is not in the corpus")
    if objective not in ("min", "max"):
        raise ValueError(f"objective must be 'min' or 'max'; got {objective!r}")

    warnings: list[str] = []
    if grouping is None:
        grouping = resolve_grouping(df, device_column)
    if grouping.degraded:
        warnings.append(
            "DEGRADED PROBLEM GROUPING: problems are identified by "
            f"{BENCHMARK_COLUMN!r} ALONE because {grouping.detail}. The same graph "
            "measured on two devices is being scored as ONE problem: the oracle "
            "becomes the best kernel on whichever device is faster, and the model's "
            "pick is drawn from that merged pool too. The figures below can come out "
            "either too high -- a correct pick on the slower device charged the faster "
            "device's time as its target -- or too low, when a problem that should "
            "have been scored twice collapses into one easy comparison. They are not "
            "comparable with figures from a corpus that carries device identity. "
            "Re-export from a sweep that logs the `device` column."
        )
    elif "carry no device identity" in grouping.detail:
        warnings.append(
            f"PARTIAL DEVICE IDENTITY: {grouping.detail}. Regret for those problems is "
            "computed against an oracle drawn from a bucket that may span devices."
        )

    keys = problem_keys(df, grouping)
    split = split_problems(keys.unique().tolist(), eval_fraction, seed)
    if split.fraction >= 1.0:
        warnings.append(
            "NO HELD-OUT SLICE: --eval-fraction 1.0 scores every problem in the "
            "corpus. Regret measured on problems the model trained on is optimistic "
            "and is not the number to ship against (RFC 0019.13 §5.6.4)."
        )

    held_out = set(split.eval_problems)
    slice_mask = keys.isin(held_out)
    eval_df = df[slice_mask]
    eval_keys = keys[slice_mask]

    if regime_column is None:
        regime_column = next(
            (name for name in REGIME_COLUMN_CANDIDATES if name in df.columns), None
        )
    elif regime_column not in df.columns:
        raise ValueError(
            f"--regime-column {regime_column!r} is not a column of this corpus"
        )

    if "is_valid" in eval_df.columns:
        valid = eval_df["is_valid"].astype(str).str.strip().str.lower() == "true"
    else:
        valid = pd.Series(True, index=eval_df.index)
    values = pd.to_numeric(eval_df[target], errors="coerce")

    exclusions = Exclusions()
    exclusions.invalid_rows = int((~valid).sum())
    exclusions.missing_target_rows = int((valid & ~np.isfinite(values)).sum())
    usable = valid & np.isfinite(values)

    if target in MILLISECOND_TARGETS and "stddevMs" in eval_df.columns:
        stddev_all = pd.to_numeric(eval_df["stddevMs"], errors="coerce")
        iters_all = (
            pd.to_numeric(eval_df["iters"], errors="coerce")
            if "iters" in eval_df.columns
            else pd.Series(1.0, index=eval_df.index)
        )
        noise_available = True
    else:
        stddev_all = iters_all = None
        noise_available = False

    results: list[ProblemResult] = []
    # Grouped by hand rather than through `Series.groupby`: the keys here are tuples,
    # and pandas treats a tuple key as a multi-column selector in several places. This
    # keeps the key exactly as it was built and the row index exactly as it was read.
    groups: dict[tuple[str, ...], list] = {}
    for index, key in eval_keys.items():
        groups.setdefault(key, []).append(index)

    for key, indices in groups.items():
        rows = eval_df.loc[indices]
        keep = usable.loc[indices]
        candidates = rows[keep.values]
        if candidates.empty:
            exclusions.problems_no_measured_candidate += 1
            continue

        measured = values.loc[candidates.index].to_numpy(dtype=float)
        oracle_position = int(np.argmin(measured) if objective == "min" else np.argmax(measured))
        oracle_value = float(measured[oracle_position])
        if not oracle_value > 0.0:
            # Both regret formulas divide by the oracle value, and under `max` a
            # non-positive oracle also flips the ratio's sense, so the metric would
            # come out negative for a correct pick. Excluded and counted rather than
            # producing a number whose sign no longer means what the report says.
            exclusions.problems_non_positive_oracle += 1
            continue
        if len(measured) < 2:
            exclusions.problems_single_candidate += 1
            continue

        predictions = np.asarray(scorer(candidates), dtype=float)
        if predictions.shape != measured.shape:
            raise ValueError(
                f"scorer returned {predictions.shape} scores for {measured.shape} "
                "candidate rows"
            )
        if not np.all(np.isfinite(predictions)):
            raise ValueError(
                f"scorer produced a non-finite score for problem {key}; a NaN score "
                "would sort arbitrarily and silently randomise the model's pick"
            )

        # Rank in the objective's direction. `argsort` is stable, so predicted ties
        # keep corpus order rather than depending on the sort implementation.
        order = np.argsort(-predictions if objective == "max" else predictions, kind="stable")
        picked_position = int(order[0])
        picked_value = float(measured[picked_position])
        regret = regret_of(picked_value, oracle_value, objective)

        rank_of = np.empty(len(order), dtype=int)
        rank_of[order] = np.arange(len(order))
        tied = _tie_mask(
            measured,
            oracle_position,
            objective,
            tie_rel_tolerance,
            tie_sigma,
            stddev_all.loc[candidates.index].to_numpy(dtype=float) if noise_available else None,
            iters_all.loc[candidates.index].to_numpy(dtype=float) if noise_available else None,
        )

        regime = None
        if regime_column is not None:
            labels = rows[regime_column].astype(str).str.strip()
            regime = labels.iloc[0] if labels.nunique() == 1 else "<mixed>"

        results.append(
            ProblemResult(
                key=tuple(key),
                regime=regime,
                candidates=len(measured),
                oracle_value=oracle_value,
                picked_value=picked_value,
                regret=regret,
                oracle_rank=int(rank_of[oracle_position]),
                tied_rank=int(rank_of[tied].min()),
                tied_candidates=int(tied.sum()),
            )
        )

    report = _build_report(
        results,
        exclusions=exclusions,
        grouping=grouping,
        split=split,
        target=target,
        objective=objective,
        regime_column=regime_column,
        tie_rel_tolerance=tie_rel_tolerance,
        tie_sigma=tie_sigma,
        noise_available=noise_available,
        regret_tail_threshold=regret_tail_threshold,
        corpus_rows=len(df),
        corpus_problems=len(split.train_problems) + len(split.eval_problems),
        warnings=warnings,
    )
    return EvaluationResult(report=report, problems=results, warnings=warnings)


def _build_report(
    results: list[ProblemResult],
    *,
    exclusions: Exclusions,
    grouping: Grouping,
    split: Split,
    target: str,
    objective: str,
    regime_column: str | None,
    tie_rel_tolerance: float,
    tie_sigma: float,
    noise_available: bool,
    regret_tail_threshold: float,
    corpus_rows: int,
    corpus_problems: int,
    warnings: list[str],
) -> dict[str, Any]:
    regrets = [item.regret for item in results]
    tail = [item for item in results if item.regret > regret_tail_threshold]

    recall: dict[str, dict[str, float | None]] = {"strict": {}, "tie_aware": {}, "trivial": {}}
    for k in TOP_K_VALUES:
        if results:
            recall["strict"][str(k)] = sum(item.oracle_rank < k for item in results) / len(results)
            recall["tie_aware"][str(k)] = sum(item.tied_rank < k for item in results) / len(results)
            # A problem with no more than k measured candidates scores a hit whatever
            # the model does. Counted so a recall@5 of 1.0 on a corpus of 4-candidate
            # problems is legible as the tautology it is.
            recall["trivial"][str(k)] = sum(item.candidates <= k for item in results) / len(results)
        else:
            recall["strict"][str(k)] = recall["tie_aware"][str(k)] = recall["trivial"][str(k)] = None

    if regime_column is None:
        per_regime = None
        per_regime_status = (
            "UNAVAILABLE: this corpus carries no regime column (looked for "
            f"{', '.join(REGIME_COLUMN_CANDIDATES)}; pass --regime-column to name a "
            "different one). §11.2 requires the per-regime table as the PRIMARY form, "
            "because an aggregate hides a model that is excellent on the dense middle "
            "of the corpus and useless on decode-shaped or prime-dimension problems -- "
            "frequently the shapes the heuristic exists to get right. The aggregate "
            "below is therefore the whole report, and it is weaker than §11.2 asks for."
        )
    else:
        buckets: dict[str, list[float]] = {}
        for item in results:
            buckets.setdefault(item.regime or "<unset>", []).append(item.regret)
        per_regime = {
            name: {"problems": len(values), "mean_regret": float(np.mean(values))}
            for name, values in sorted(buckets.items())
        }
        per_regime_status = f"from column {regime_column!r}"

    return {
        "schema": REPORT_SCHEMA,
        "rfc": "0019.13 §11.2",
        "generated": datetime.now(timezone.utc).isoformat(),
        "corpus": {"rows": corpus_rows, "problems": corpus_problems},
        "target": target,
        "objective": objective,
        "grouping": {
            "columns": list(grouping.columns),
            "degraded": grouping.degraded,
            "detail": grouping.detail,
        },
        "split": {
            "method": split.method,
            "unit": "problem",
            "seed": split.seed,
            "eval_fraction": split.fraction,
            "train_problems": len(split.train_problems),
            "eval_problems": len(split.eval_problems),
            "eval_problem_keys": [list(key) for key in split.eval_problems],
            "note": (
                "Problems, not rows, are held out: every candidate of an evaluated "
                "problem is on the evaluation side, so the oracle is the best of the "
                "full measured set. Assignment is a seeded SHA-256 of the problem key, "
                "so the slice is reproducible from (corpus, seed) alone."
            ),
        },
        "slice": {
            "name": "held-out evaluation slice",
            "exhaustive": False,
            "caveat": (
                "V(p) here is what the sweep happened to measure, not every applicable "
                "configuration, so v*(p) is a lower bound on the best and this regret "
                "is optimistic (RFC 0019.13 §11.1). It becomes exact only on a slice "
                "measured exhaustively per §5.6.3."
            ),
        },
        "exclusions": exclusions.as_dict(),
        "metrics": {
            "problems_scored": len(results),
            "top1_regret": _summarise(regrets),
            "regret_tail": {
                "threshold": regret_tail_threshold,
                "problems": len(tail),
                "fraction": (len(tail) / len(results)) if results else None,
            },
            "topk_recall": recall,
            "per_regime": per_regime,
            "per_regime_status": per_regime_status,
        },
        "ties": {
            "rel_tolerance": tie_rel_tolerance,
            "sigma": tie_sigma,
            "noise_band_applied": noise_available,
            "policy": (
                "Regret needs no tie rule: it is measured in the target metric, so two "
                "indistinguishable kernels differ by an indistinguishable regret "
                "(§11.2). Top-k recall is a rank test and does need one, so it is "
                "reported twice -- `strict` requires the exact oracle row in the top k, "
                "`tie_aware` accepts any candidate within "
                f"{tie_rel_tolerance:.1%} of the oracle's measured value"
                + (
                    f", or within {tie_sigma:g} standard errors of it using stddevMs/iters"
                    if noise_available
                    else " (no stddevMs noise band: the target is not a millisecond "
                    "timing column, so the corpus spread is not in comparable units)"
                )
                + "."
            ),
        },
        "not_implemented": [
            "§11.2 weighted aggregates: regret weighted by declared regime weights "
            "(§5.2). Nothing in this pipeline declares weights, so only the unweighted "
            "figure is computed -- which §11.2 says is the whole report for a blindly "
            "generated corpus, but a corpus that does declare weights needs both.",
            "§11.2 calibration metrics (relative absolute error, signed bias, "
            "selected-candidate calibration). Required only when score.calibrated is "
            "true; they measure the score's absolute value, not its ranking.",
            "§11.3 shape extrapolation (leave-one-regime-out) and variant "
            "extrapolation (leave-variants-out). Both need retraining per fold; this "
            "command scores one already-trained model.",
            "§5.6.3/§5.6.4 round-0 core versus full slice, and the steering/reserved "
            "portions of the evaluation slice. Those are properties of a corpus "
            "collected by the campaign loop, which does not exist yet; the split here "
            "is over whatever corpus it is given.",
        ],
        "warnings": warnings,
    }


# --------------------------------------------------------------------------------------
# Scoring a trained model
# --------------------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class ModelBundle:
    """A trained UHD, loaded well enough to rank candidates."""

    scorer: Scorer
    features: list[str]
    target: str | None
    objective: str | None
    source: str
    trained_on: str | None
    training_rows: int | None


def _flatbuffer_scorer(artifact: Path, features: list[str]) -> Scorer:
    """Score with the artifact that actually ships.

    `train` deletes `model.lgbm` unless `--keep-lgbm`, so on an ordinary model
    directory `model.bin` is the only thing left to rank with -- and it is also the
    file the engine loads, which makes it the right thing to measure anyway.

    Only the induced ORDER matters (see `Scorer`), so the ensemble is summed exactly as
    `TreeDataAdapter::score()` does -- unit learning rate, LightGBM having folded the
    shrinkage into the dumped leaf values -- and the log1p inverse is applied for
    callers who want the value, not because ranking needs it.
    """
    import uhd_gen  # noqa: F401  puts _generated/ on sys.path

    from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT

    from .train_uhd import build_feature_matrix

    with open(artifact, "rb") as handle:
        model = GbdtModelT.InitFromPackedBuf(bytearray(handle.read()), 0)

    trees = [
        (
            np.asarray(tree.featureIndices, dtype=np.int64),
            np.asarray(tree.thresholds, dtype=np.float64),
            np.asarray(tree.leftChildren, dtype=np.int64),
            np.asarray(tree.rightChildren, dtype=np.int64),
            np.asarray(tree.leafValues, dtype=np.float64),
            np.asarray(tree.defaultLeft, dtype=bool),
            np.asarray(tree.decisionLte, dtype=bool),
        )
        for tree in model.trees
    ]
    base = float(model.baseScore)

    def score(frame: pd.DataFrame) -> np.ndarray:
        matrix = build_feature_matrix(frame, features)
        total = np.full(len(frame), base, dtype=np.float64)
        for feature_index, threshold, left, right, leaf, default_left, lte in trees:
            node = np.zeros(len(frame), dtype=np.int64)
            # Descend every row one level per iteration rather than one row at a time:
            # a 500-tree model over a corpus is otherwise minutes of Python.
            while True:
                internal = left[node] >= 0
                if not internal.any():
                    break
                rows = np.flatnonzero(internal)
                here = node[rows]
                x = matrix[rows, feature_index[here]]
                go_left = np.where(lte[here], x <= threshold[here], x > threshold[here])
                go_left = np.where(np.isnan(x), default_left[here], go_left)
                node[rows] = np.where(go_left, left[here], right[here])
            total += leaf[node]
        return np.expm1(total)

    return score


def _booster_scorer(model_file: Path, features: list[str]) -> Scorer:
    import lightgbm as lgb

    from .train_uhd import build_feature_matrix, predict

    booster = lgb.Booster(model_file=str(model_file))

    def score(frame: pd.DataFrame) -> np.ndarray:
        return predict(booster, build_feature_matrix(frame, features))

    return score


def load_model(model_dir: Path, model_file: Path | None = None) -> ModelBundle:
    """Load a `train --output-dir` result: features, direction, and something to rank with."""
    descriptor_paths = sorted(model_dir.glob("*.uhd.json"))
    descriptor = _load_json(descriptor_paths[0]) if descriptor_paths else {}
    manifest_path = model_dir / "train_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}

    features = manifest.get("features") or [
        name.lstrip("$") for name in descriptor.get("features_signature", [])
    ]
    if not features:
        raise ValueError(
            f"{model_dir} carries neither a train_manifest.json with `features` nor a "
            "descriptor with `features_signature`; there is no way to know which "
            "columns the model was trained on"
        )

    # The objective is READ, never assumed: it decides which end of the measured range
    # is the oracle, and getting it backwards inverts every number in the report.
    objective = descriptor.get("objective") or manifest.get("objective")

    if model_file is not None:
        candidate = model_file
    elif (model_dir / "model.lgbm").exists():
        candidate = model_dir / "model.lgbm"
    else:
        artifact = descriptor.get("tree_data", {}).get("artifact", "model.bin")
        candidate = model_dir / artifact
    if not candidate.exists():
        raise ValueError(f"no model artifact at {candidate}")

    if candidate.suffix in (".lgbm", ".txt"):
        scorer = _booster_scorer(candidate, features)
    else:
        scorer = _flatbuffer_scorer(candidate, features)

    return ModelBundle(
        scorer=scorer,
        features=list(features),
        target=manifest.get("target"),
        objective=objective,
        source=str(candidate),
        trained_on=manifest.get("input_file"),
        training_rows=manifest.get("num_samples"),
    )


def _holdout_integrity(corpus: Path, bundle: ModelBundle) -> dict[str, str]:
    """Was the model kept away from the problems it is about to be scored on?

    A model fitted on the whole corpus has already seen every evaluation problem, and
    §5.6.4 is explicit that scoring a problem with a model that saw it is a leak. This
    cannot be proved from the artifacts -- only the training input's path is recorded --
    but the common case, evaluating the same CSV that was trained on, IS detectable,
    and it is exactly the case that produces a flattering number.
    """
    if bundle.trained_on is None:
        return {
            "status": "unknown",
            "detail": "the model directory has no train_manifest.json, so what it was "
            "trained on cannot be checked; if it was this corpus, the regret below is "
            "optimistic (RFC 0019.13 §5.6.4).",
        }
    try:
        same = Path(bundle.trained_on).resolve() == corpus.resolve()
    except OSError:
        same = str(bundle.trained_on) == str(corpus)
    if same:
        return {
            "status": "COMPROMISED",
            "detail": f"the model was trained on this very corpus ({bundle.trained_on}), "
            "so it has already seen every evaluation problem. §5.6.4: scoring a problem "
            "with a model that trained on it is a leak, and the regret below is "
            "optimistic by an unknown amount. Use --emit-train-slice to write the "
            "training side of this split, train on THAT, then evaluate again with the "
            "same --seed and --eval-fraction.",
        }
    return {
        "status": "held_out",
        "detail": f"the model was trained on {bundle.trained_on}, which is not this "
        "corpus file. Whether that file overlaps this one's evaluation problems is not "
        "checkable from the artifacts; with --emit-train-slice it does not overlap by "
        "construction.",
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def add_evaluate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Benchmark CSV/JSON to evaluate on")
    parser.add_argument(
        "--model-dir",
        required=True,
        dest="model_dir",
        help="A `train --output-dir` result: descriptor, model artifact, manifest",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model artifact to rank with (default: model.lgbm if present, else the "
        "descriptor's tree_data.artifact -- the file the engine itself loads)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write eval_report.json (default: <model-dir>/eval_report.json)",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=DEFAULT_EVAL_FRACTION,
        dest="eval_fraction",
        help=f"Fraction of PROBLEMS held out and scored (default: {DEFAULT_EVAL_FRACTION}). "
        "1.0 scores the whole corpus and says loudly that the figure is optimistic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for the problem-level split (default: {DEFAULT_SEED}). Recorded in "
        "the report; the same corpus and seed always give the same slice.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Measured column regret is computed in (default: the manifest's target)",
    )
    parser.add_argument(
        "--objective",
        choices=("min", "max"),
        default=None,
        help="Override the direction read from the descriptor/manifest. Both are legal "
        "and getting it backwards inverts every number, so it is read, not assumed.",
    )
    parser.add_argument(
        "--device-column",
        default=None,
        dest="device_column",
        help=f"Column holding device identity (default: {DEVICE_COLUMN!r}). A problem is "
        "(benchmark, device); without device identity the grouping degrades and says so.",
    )
    parser.add_argument(
        "--regime-column",
        default=None,
        dest="regime_column",
        help="Column holding the corpus regime for the §11.2 per-regime table "
        f"(default: the first of {', '.join(REGIME_COLUMN_CANDIDATES)} that is present)",
    )
    parser.add_argument(
        "--tie-rel-tolerance",
        type=float,
        default=DEFAULT_TIE_REL_TOLERANCE,
        dest="tie_rel_tolerance",
        help="Candidates within this fraction of the oracle's measured value count as "
        f"tied with it for tie-aware top-k recall (default: {DEFAULT_TIE_REL_TOLERANCE})",
    )
    parser.add_argument(
        "--tie-sigma",
        type=float,
        default=DEFAULT_TIE_SIGMA,
        dest="tie_sigma",
        help="Width, in standard errors, of the noise band that also counts as a tie "
        f"when the target is a millisecond timing (default: {DEFAULT_TIE_SIGMA})",
    )
    parser.add_argument(
        "--regret-tail-threshold",
        type=float,
        default=DEFAULT_REGRET_TAIL_THRESHOLD,
        dest="regret_tail_threshold",
        help=f"Regret above which a problem counts in the tail (default: {DEFAULT_REGRET_TAIL_THRESHOLD})",
    )
    parser.add_argument(
        "--include-per-problem",
        action="store_true",
        dest="include_per_problem",
        help="Write every problem's oracle, pick and regret into the report",
    )
    parser.add_argument(
        "--emit-train-slice",
        default=None,
        dest="emit_train_slice",
        help="Write the TRAINING side of this split to a CSV. Train on that file and "
        "evaluate with the same --seed and --eval-fraction, and the model provably "
        "never saw an evaluation problem.",
    )


def _read_corpus(path: Path) -> pd.DataFrame:
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def run_evaluate(args: argparse.Namespace) -> int:
    corpus_path = Path(args.input)
    model_dir = Path(args.model_dir)

    try:
        bundle = load_model(model_dir, Path(args.model) if args.model else None)
    except (ValueError, OSError) as error:
        logger.error("%s", error)
        return 1

    df = _read_corpus(corpus_path)
    logger.info("Loaded %d row(s) from %s", len(df), corpus_path)

    target = args.target or bundle.target
    if target is None:
        logger.error(
            "no target column: %s records none and --target was not passed. Regret is "
            "measured in the target metric, so there is nothing to measure without it.",
            model_dir / "train_manifest.json",
        )
        return 1
    objective = args.objective or bundle.objective
    if objective is None:
        logger.error(
            "no objective: neither the descriptor nor the manifest in %s declares one "
            "and --objective was not passed. min and max are both legal and the wrong "
            "one inverts every number, so this is never guessed.",
            model_dir,
        )
        return 1

    try:
        result = evaluate_corpus(
            df,
            bundle.scorer,
            target=target,
            objective=objective,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
            regime_column=args.regime_column,
            device_column=args.device_column,
            tie_rel_tolerance=args.tie_rel_tolerance,
            tie_sigma=args.tie_sigma,
            regret_tail_threshold=args.regret_tail_threshold,
        )
    except (ValueError, ObjectiveDirectionError) as error:
        logger.error("%s", error)
        return 1

    report = result.report
    report["corpus"]["path"] = str(corpus_path)
    report["model"] = {
        "model_dir": str(model_dir),
        "artifact": bundle.source,
        "features": bundle.features,
        "objective_source": "--objective" if args.objective else "descriptor/manifest",
        "trained_on": bundle.trained_on,
        "training_rows": bundle.training_rows,
    }
    report["holdout_integrity"] = _holdout_integrity(corpus_path, bundle)
    if report["holdout_integrity"]["status"] == "COMPROMISED":
        report["warnings"].append(
            "HELD-OUT SLICE COMPROMISED: " + report["holdout_integrity"]["detail"]
        )
    if args.include_per_problem:
        report["per_problem"] = [
            {
                "key": list(item.key),
                "regime": item.regime,
                "candidates": item.candidates,
                "oracle_value": item.oracle_value,
                "picked_value": item.picked_value,
                "regret": item.regret,
                "oracle_rank": item.oracle_rank,
                "tied_rank": item.tied_rank,
                "tied_candidates": item.tied_candidates,
            }
            for item in result.problems
        ]

    if args.emit_train_slice:
        grouping = Grouping(
            tuple(report["grouping"]["columns"]),
            report["grouping"]["degraded"],
            report["grouping"]["detail"],
        )
        keys = problem_keys(df, grouping)
        held_out = {tuple(key) for key in report["split"]["eval_problem_keys"]}
        slice_path = Path(args.emit_train_slice)
        slice_path.parent.mkdir(parents=True, exist_ok=True)
        df[~keys.isin(held_out)].to_csv(slice_path, index=False)
        report["split"]["train_slice"] = str(slice_path)
        logger.info("Wrote training slice to %s", slice_path)

    output_path = Path(args.output) if args.output else model_dir / "eval_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    _print_summary(report, output_path)
    return 0


def _print_summary(report: dict[str, Any], output_path: Path) -> None:
    metrics = report["metrics"]
    regret = metrics["top1_regret"]

    # Warnings first and unmissable. A degraded grouping or a compromised holdout makes
    # every figure below mean something other than what it is labelled, so it cannot be
    # a footnote under the numbers it invalidates.
    for warning in report["warnings"]:
        print(f"\n!! {warning}")

    print(f"\nRegret report ({report['rfc']}) -- {output_path}")
    print(f"  target/objective:   {report['target']} ({report['objective']})")
    print(f"  problems grouped by: {', '.join(report['grouping']['columns'])}")
    print(
        f"  split:              {report['split']['method']}, seed {report['split']['seed']}, "
        f"{report['split']['eval_problems']} eval / {report['split']['train_problems']} train problem(s)"
    )
    print(f"  problems scored:    {metrics['problems_scored']}")
    if regret["mean"] is None:
        print("  top-1 regret:       n/a (no problem had two measured candidates)")
    else:
        print(
            "  top-1 regret:       mean {mean:.4f}  p50 {p50:.4f}  p95 {p95:.4f}  max {max:.4f}".format(
                **{key: value for key, value in regret.items()}
            )
        )
        print(
            f"  regret tail (>{metrics['regret_tail']['threshold']:.0%}): "
            f"{metrics['regret_tail']['fraction']:.4f} "
            f"({metrics['regret_tail']['problems']} problem(s))"
        )
        strict = metrics["topk_recall"]["strict"]
        tie_aware = metrics["topk_recall"]["tie_aware"]
        for k in TOP_K_VALUES:
            print(
                f"  top-{k} recall:       strict {strict[str(k)]:.4f}   "
                f"tie-aware {tie_aware[str(k)]:.4f}"
            )
    if metrics["per_regime"] is None:
        print(f"  per-regime regret:  {metrics['per_regime_status'].splitlines()[0]}")
    else:
        print(f"  per-regime regret:  ({metrics['per_regime_status']})")
        for name, values in metrics["per_regime"].items():
            print(
                f"    {name:<24} mean {values['mean_regret']:.4f}  "
                f"({values['problems']} problem(s))"
            )
    integrity = report.get("holdout_integrity", {}).get("status")
    if integrity:
        print(f"  holdout integrity:  {integrity}")
    dropped = report["exclusions"]
    print(
        "  excluded:           {invalid_rows} invalid row(s), {missing_target_rows} "
        "row(s) with no target, {problems_single_candidate} single-candidate "
        "problem(s), {problems_no_measured_candidate} problem(s) with nothing "
        "measured, {problems_non_positive_oracle} with a non-positive oracle".format(
            **{key: dropped[key] for key in dropped if key != "policy"}
        )
    )
