#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Score a trained UHD against the best measured kernel (RFC 0019.13 §11.2).

RMSE on `log(target)` can improve while the pick-the-best-kernel decision gets worse, so
this reports top-1 regret, the regret tail, top-k recall, and per-regime regret.
The oracle is the best *measured* candidate; regret is in the target metric, not rank,
so picking one of two indistinguishable kernels costs nothing. The held-out slice is
split by problem (§5.6.4): a row-wise split would make the oracle the best of a subset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .correctness import known_wrong
from .corpus_io import read_corpus_frame
from .ranking_metrics import RANKING_METRICS
from .score_transform import INVERTIBLE as INVERTIBLE_TRANSFORMS
from .score_transform import TRAINED as TRAINED_TRANSFORM
from .score_transform import inverse as invert_score

logger = logging.getLogger(__name__)

#: Report format identity, written into every report.
REPORT_SCHEMA = "uhd_gen.eval_report/1"

#: §11.2: "fraction of problems with regret > 5%".
DEFAULT_REGRET_TAIL_THRESHOLD = 0.05

#: §11.2: "k = 1, 3, 5".
TOP_K_VALUES = (1, 3, 5)

#: Advisory only: §11.2 fixes no calibration threshold. Above benchmark noise, below a
#: difference that would reorder engines of genuinely different speed.
CALIBRATION_BIAS_WARN = 0.10

DEFAULT_EVAL_FRACTION = 0.2
DEFAULT_SEED = 0

#: Device identity column of the §8.3 envelope: hex of `DeviceKey::hash()`. Not the
#: `device.*` feature columns, which are not part of problem identity.
DEVICE_COLUMN = "device"

#: The graph-content hash; the whole problem identity on corpora without `device`.
BENCHMARK_COLUMN = "benchmark"

#: Where a corpus might carry §5.2's regime label. Optional: exported sweeps carry none.
REGIME_COLUMN_CANDIDATES = ("regime", "corpus_regime", "q.regime", "problem.regime")


def _regime_column(columns: Iterable[str]) -> str | None:
    """The corpus's regime label: a named candidate, else any namespaced `*.regime`."""
    columns = list(columns)
    named = next((name for name in REGIME_COLUMN_CANDIDATES if name in columns), None)
    if named is not None:
        return named
    return next((c for c in columns if c.endswith(".regime")), None)


#: Relative regret below which two candidates tie for top-k recall (see `_tie_mask`).
DEFAULT_TIE_REL_TOLERANCE = 0.01

#: Tie noise band half-width in standard errors, for millisecond targets with `stddevMs`.
DEFAULT_TIE_SIGMA = 2.0

#: Millisecond targets, comparable with `stddevMs`; the noise band applies only to these.
MILLISECOND_TARGETS = frozenset({"minTimeMs", "avgTimeMs", "robustMeanMs"})

#: Float slack for the regret non-negativity check.
_REGRET_EPSILON = 1e-9

#: Maps candidate rows to a score in the target's direction. Only the induced order is
#: used; regret comes from measured values.
Scorer = Callable[[pd.DataFrame], np.ndarray]


class ObjectiveDirectionError(RuntimeError):
    """A regret came out negative: `objective` is backwards, so every figure is inverted.

    Raised rather than clamped so an inverted report is never printed.
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
    #: Best rank of any candidate tied with the oracle; `oracle_rank` if none.
    tied_rank: int
    tied_candidates: int
    #: §11.4 static-order reference: corpus rows are in engine enumeration order, so the
    #: static pick is the first usable row and the static ranking is the row order.
    static_order_value: float
    static_order_regret: float
    static_order_oracle_rank: int
    static_order_tied_rank: int
    #: §11.4 random reference: exact expectations over the candidate set, not sampled.
    random_regret: float
    random_tail_fraction: float
    #: Two-layer models: `regret` split into wrong-group and wrong-member parts, which
    #: sum to `regret`. None for single-layer models or without the group column.
    group_regret: float | None = None
    in_group_regret: float | None = None


@dataclass
class Exclusions:
    """Rows and problems that could not contribute, counted by reason.

    Only rows with no usable measurement are dropped; anything else would corrupt the
    oracle (§5.6.3).
    """

    invalid_rows: int = 0
    numerically_invalid_rows: int = 0
    missing_target_rows: int = 0
    problems_no_measured_candidate: int = 0
    problems_single_candidate: int = 0
    problems_non_positive_oracle: int = 0

    def deterministic_catalog(self, scored: int) -> bool:
        """Nothing was scored, solely because every problem had one candidate.

        That means kernel choice is a function of the problem (train L1), unlike an empty
        report caused by unmeasured problems (fix the sweep).
        """
        return (
            scored == 0
            and self.problems_single_candidate > 0
            and self.problems_no_measured_candidate == 0
            and self.problems_non_positive_oracle == 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "invalid_rows": self.invalid_rows,
            "numerically_invalid_rows": self.numerically_invalid_rows,
            "missing_target_rows": self.missing_target_rows,
            "problems_no_measured_candidate": self.problems_no_measured_candidate,
            "problems_single_candidate": self.problems_single_candidate,
            "problems_non_positive_oracle": self.problems_non_positive_oracle,
            "policy": (
                "Only rows that carry no usable measurement are dropped: is_valid=False "
                "(a candidate that never ran has no time and cannot be the best), "
                "numerically_valid=False (a candidate shown to compute the wrong answer "
                "has no time for the right one, RFC 0019 §13.2) and rows whose target is "
                "empty or non-numeric. Every measured candidate of "
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
    """Decide what identifies a problem: `(benchmark, device)`.

    Each GPU has its own best kernel; pooling them charges the slower card regret it
    could not avoid. Corpora without `device` degrade to `benchmark` and are flagged.
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
    """Hold out `fraction` of the problems, reproducibly.

    Assigned by a seeded hash of the problem key, so row order or appended rows do not
    move the split (§5.6.3).
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
        # At least one problem, so a small corpus gets an imprecise slice, not an empty one.
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


#: Transforms whose inverse is a physical, strictly positive quantity whatever the metric.
_PHYSICAL_TRANSFORMS = frozenset({"log", "log1p", "sqrt"})


def score_is_physical(score_declaration: dict | None) -> bool:
    """Whether the runtime holds this score to `> 0` (`scoreFromRaw`, UhdKernelHeuristic.hpp).

    True for a declared metric or a log/log1p/sqrt transform; a metric-less identity or
    exp score is an ordering key and may be signed.
    """
    declaration = score_declaration or {}
    return (
        bool(declaration.get("metric"))
        or declaration.get("transform") in _PHYSICAL_TRANSFORMS
    )


def rankable_scores(recovered: np.ndarray, physical: bool) -> np.ndarray:
    """The runtime's score admission: finite, and positive when the score is physical.

    Anything else the engine ranks last in declared order.
    """
    finite = np.isfinite(recovered)
    return finite & (recovered > 0) if physical else finite


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


def _regret_vector(
    values: np.ndarray, oracle_value: float, objective: str
) -> np.ndarray:
    """`regret_of`, vectorised over one problem's whole candidate set."""
    cost = (
        values / oracle_value - 1.0
        if objective == "min"
        else 1.0 - values / oracle_value
    )
    return np.maximum(cost, 0.0)


def _tie_mask(
    values: np.ndarray,
    oracle_position: int,
    objective: str,
    rel_tolerance: float,
    sigma: float,
    stddev: np.ndarray | None,
    iters: np.ndarray | None,
) -> np.ndarray:
    """Which candidates are indistinguishable from the oracle, for top-k recall.

    Regret already scores near-ties near zero; rank-based recall does not. Tied means
    regret within `rel_tolerance`, or, for millisecond targets, values within `sigma`
    standard errors (`stddevMs/sqrt(iters)`; approximate for min/robust-mean targets).
    The report also carries strict recall.
    """
    oracle_value = float(values[oracle_position])
    tied = _regret_vector(values, oracle_value, objective) <= rel_tolerance

    if stddev is not None and iters is not None:
        counts = np.where(np.isfinite(iters) & (iters > 0), iters, 1.0)
        spread = np.where(np.isfinite(stddev) & (stddev >= 0), stddev, 0.0)
        standard_error = spread / np.sqrt(counts)
        band = sigma * np.sqrt(standard_error**2 + standard_error[oracle_position] ** 2)
        tied = tied | (np.abs(values - oracle_value) <= band)

    tied[oracle_position] = True

    return tied


def _percentile(values: list[float], q: float) -> float:
    return (
        float(np.percentile(np.asarray(values, dtype=float), q))
        if values
        else float("nan")
    )


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
    group_column: str | None = None,
    score_declaration: dict | None = None,
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
        regime_column = _regime_column(df.columns)
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
    # A known-wrong row may still carry a timing; it must never become the oracle.
    wrong = known_wrong(eval_df)
    exclusions.numerically_invalid_rows = int((valid & wrong).sum())
    valid = valid & ~wrong
    exclusions.missing_target_rows = int((valid & ~np.isfinite(values)).sum())
    usable = valid & np.isfinite(values)

    if target not in MILLISECOND_TARGETS:
        stddev_all = iters_all = None
        noise_available = False
        noise_absent_reason = (
            f"the target {target!r} is not a millisecond timing column, so the corpus "
            "spread is not in comparable units"
        )
    elif "stddevMs" not in eval_df.columns:
        # §8.3 requires `stddevMs`/`iters`; say so rather than reporting a units mismatch.
        stddev_all = iters_all = None
        noise_available = False
        noise_absent_reason = (
            "this corpus carries no `stddevMs` column, though the target is a "
            "millisecond timing. §8.3 lists `stddevMs` and `iters` as required columns "
            "of the result envelope"
        )
        warnings.append(
            "NO MEASUREMENT NOISE: the corpus has no `stddevMs` column, so the "
            f"tie-aware recall for target {target!r} falls back to the "
            f"{tie_rel_tolerance:.1%} relative tolerance alone and the "
            "--tie-sigma band never applies. §8.3 requires `stddevMs` and `iters`; "
            "re-collect with a producer that emits them."
        )
    else:
        stddev_all = pd.to_numeric(eval_df["stddevMs"], errors="coerce")
        iters_all = (
            pd.to_numeric(eval_df["iters"], errors="coerce")
            if "iters" in eval_df.columns
            else pd.Series(1.0, index=eval_df.index)
        )
        noise_available = True
        noise_absent_reason = None

    results: list[ProblemResult] = []
    calibration_predicted: list[float] = []
    calibration_measured: list[float] = []
    calibration_picked_predicted: list[float] = []
    calibration_picked_measured: list[float] = []
    # Finite scores the runtime discards as non-positive; reported beside calibration.
    discarded_predictions = 0
    physical = score_is_physical(score_declaration)
    # Grouped by hand: pandas treats tuple keys as multi-column selectors in places.
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
        oracle_position = int(
            np.argmin(measured) if objective == "min" else np.argmax(measured)
        )
        oracle_value = float(measured[oracle_position])
        if not oracle_value > 0.0:
            # Regret divides by the oracle, and a non-positive one flips its sign.
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
        if np.isnan(predictions).any():
            raise ValueError(
                f"scorer produced a NaN score for problem {key}; a NaN score "
                "would sort arbitrarily and silently randomise the model's pick"
            )

        # Rank in the objective's direction; stable sort keeps corpus order on ties.
        # Unrankable candidates (`rankable_scores`) are forced last explicitly: under
        # `min` a negative or -inf score would otherwise be picked. With nothing
        # rankable, the pick is §5 step 7's declared order, as in the engine.
        rankable = rankable_scores(predictions, physical)
        discarded_predictions += int(
            np.count_nonzero(np.isfinite(predictions) & ~rankable)
        )
        ranking_key = -predictions if objective == "max" else predictions
        ranking_key = np.where(rankable, ranking_key, np.inf)
        order = np.argsort(ranking_key, kind="stable")
        picked_position = int(order[0])
        picked_value = float(measured[picked_position])
        regret = regret_of(picked_value, oracle_value, objective)

        # §11.2 calibration inputs: ranking ignores a constant-factor score error, but
        # cross-engine arbitration (RFC 0019 §11.3) compares absolute values. Only
        # rankable scores count: -inf declines and non-positive scores are never used.
        calibration_predicted.extend(predictions[rankable].tolist())
        calibration_measured.extend(measured[rankable].tolist())
        if rankable[picked_position]:
            calibration_picked_predicted.append(float(predictions[picked_position]))
            calibration_picked_measured.append(picked_value)

        rank_of = np.empty(len(order), dtype=int)
        rank_of[order] = np.arange(len(order))
        tied = _tie_mask(
            measured,
            oracle_position,
            objective,
            tie_rel_tolerance,
            tie_sigma,
            (
                stddev_all.loc[candidates.index].to_numpy(dtype=float)
                if noise_available
                else None
            ),
            (
                iters_all.loc[candidates.index].to_numpy(dtype=float)
                if noise_available
                else None
            ),
        )

        # §11.4 references. Static order: rows follow engine enumeration (§2.1), so
        # the static pick is the first usable row.
        static_order_value = float(measured[0])
        # Random: exact expectations over V(p), so they do not move between runs.
        candidate_regrets = _regret_vector(measured, oracle_value, objective)

        regime = None
        if regime_column is not None:
            labels = rows[regime_column].astype(str).str.strip()
            regime = labels.iloc[0] if labels.nunique() == 1 else "<mixed>"

        # Split regret into wrong-group and wrong-member parts; they sum to `regret`.
        group_regret = in_group_regret = None
        if group_column is not None and group_column in candidates.columns:
            same_group = (
                candidates[group_column].to_numpy()
                == candidates[group_column].iloc[picked_position]
            )
            in_group = measured[same_group]
            group_best = float(in_group.min() if objective == "min" else in_group.max())
            group_regret = regret_of(group_best, oracle_value, objective)
            in_group_regret = regret - group_regret

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
                static_order_value=static_order_value,
                static_order_regret=regret_of(
                    static_order_value, oracle_value, objective
                ),
                static_order_oracle_rank=oracle_position,
                static_order_tied_rank=int(np.flatnonzero(tied)[0]),
                random_regret=float(candidate_regrets.mean()),
                random_tail_fraction=float(
                    (candidate_regrets > regret_tail_threshold).mean()
                ),
                group_regret=group_regret,
                in_group_regret=in_group_regret,
            )
        )

    calibration = _calibration_block(
        score_declaration,
        target,
        calibration_predicted,
        calibration_measured,
        calibration_picked_predicted,
        calibration_picked_measured,
        discarded_predictions,
    )
    # Not a gate (§11.4), but a systematic bias is invisible in ranking figures, so warn.
    # Over-reading a throughput or under-reading a time wins arbitrations it should lose.
    selected = calibration.get("selected_candidate")
    if selected is not None:
        bias = selected["signed_relative_bias"]
        if abs(bias) > CALIBRATION_BIAS_WARN:
            warnings.append(
                f"CALIBRATED SCORE IS BIASED: on the candidates this model picks it "
                f"{'OVER' if bias > 0 else 'UNDER'}-predicts by {abs(bias):.1%} on average "
                f"(threshold {CALIBRATION_BIAS_WARN:.0%}). Ranking within this engine is "
                "unaffected -- a constant factor cannot reorder a catalog -- but RFC 0019 "
                "§11.3 compares this number against other engines' predictions, so the "
                "quick and thorough policies will "
                f"{'favour' if (bias > 0) == (objective == 'max') else 'avoid'} this engine "
                "by roughly that margin."
            )

    report = _build_report(
        results,
        exclusions=exclusions,
        grouping=grouping,
        split=split,
        target=target,
        objective=objective,
        metric=(score_declaration or {}).get("metric"),
        regime_column=regime_column,
        tie_rel_tolerance=tie_rel_tolerance,
        tie_sigma=tie_sigma,
        noise_available=noise_available,
        noise_absent_reason=noise_absent_reason,
        regret_tail_threshold=regret_tail_threshold,
        corpus_rows=len(df),
        corpus_problems=len(split.train_problems) + len(split.eval_problems),
        calibration=calibration,
        warnings=warnings,
        group_column=group_column,
    )
    return EvaluationResult(report=report, problems=results, warnings=warnings)


def _calibration_summary(predicted: np.ndarray, measured: np.ndarray) -> dict[str, Any]:
    """§11.2's absolute-value figures for one set of (prediction, measurement) pairs."""
    error = predicted - measured
    relative = error / measured
    return {
        "rows": int(predicted.size),
        "signed_bias": float(np.mean(error)),
        "signed_relative_bias": float(np.mean(relative)),
        "mean_absolute_error": float(np.mean(np.abs(error))),
        "relative_absolute_error": float(np.mean(np.abs(relative))),
        "rmse": float(np.sqrt(np.mean(error * error))),
    }


def _calibration_block(
    score_declaration: dict | None,
    target: str,
    predicted: list[float],
    measured: list[float],
    picked_predicted: list[float],
    picked_measured: list[float],
    discarded_predictions: int,
) -> dict[str, Any]:
    """RFC 0019.13 §11.2's calibration metrics, or why they were not computed.

    Nothing else checks the absolute scale (`train --calibrated` only stamps the flag).
    Reported over all scored candidates and over the picked one, which is what
    cross-engine selection consumes.
    """
    declaration = score_declaration or {}
    if declaration.get("calibrated") is not True:
        return {
            "status": "not applicable",
            "detail": (
                "The descriptor does not declare `score.calibrated: true`, so its score "
                "is an ordering key and not an absolute quantity. §11.2 requires these "
                "metrics only of a calibrated score; ranking this engine's own catalog "
                "needs no absolute accuracy."
            ),
        }
    if not measured:
        return {
            "status": "UNAVAILABLE",
            "detail": "No problem produced a scored candidate, so there is nothing to compare.",
            "excluded_runtime_discarded_predictions": discarded_predictions,
        }

    predicted_array = np.asarray(predicted, dtype=float)
    measured_array = np.asarray(measured, dtype=float)
    picked_predicted_array = np.asarray(picked_predicted, dtype=float)
    picked_measured_array = np.asarray(picked_measured, dtype=float)
    # A `max` target can still have a zero candidate (only zero oracles are dropped).
    usable = measured_array > 0.0
    picked_usable = picked_measured_array > 0.0
    if not usable.any():
        return {
            "status": "UNAVAILABLE",
            "detail": "No scored candidate carries a positive measurement to compare against.",
        }

    metric = declaration.get("metric")
    block: dict[str, Any] = {
        "status": "computed",
        "metric": metric,
        "target": target,
        "all_candidates": _calibration_summary(
            predicted_array[usable], measured_array[usable]
        ),
        "selected_candidate": (
            _calibration_summary(
                picked_predicted_array[picked_usable],
                picked_measured_array[picked_usable],
            )
            if picked_usable.any()
            else None
        ),
        "excluded_non_positive_rows": int((~usable).sum()),
        # Runtime-discarded scores are never consumed, so not calibrated, but counted.
        "excluded_runtime_discarded_predictions": discarded_predictions,
    }
    label = RANKING_METRICS[metric].label if metric in RANKING_METRICS else None
    if label != target:
        # The figures subtract prediction from measurement; a registered metric fixes its
        # label column (RFC 0019 §13.4), so any other target is a different quantity.
        block["warning"] = (
            f"score.metric {metric!r} is measured by the {label!r} column but the target "
            f"column is {target!r}. These figures subtract the prediction from the "
            "measurement, so they mean nothing unless both are the same physical quantity."
        )
    return block


def _regime_means(
    results: list[ProblemResult],
    value: Callable[[ProblemResult], float],
    regime_column: str | None,
) -> dict[str, dict[str, float]] | None:
    """§11.2's per-regime table for whichever per-problem quantity `value` reads."""
    if regime_column is None:
        return None
    buckets: dict[str, list[float]] = {}
    for item in results:
        buckets.setdefault(item.regime or "<unset>", []).append(value(item))
    return {
        name: {"problems": len(items), "mean_regret": float(np.mean(items))}
        for name, items in sorted(buckets.items())
    }


def _random_tie_recall(candidates: int, tied: int, k: int) -> float:
    """Chance a uniformly drawn k-subset of `candidates` holds one of the `tied` rows."""
    if k >= candidates:
        return 1.0
    return 1.0 - math.comb(candidates - tied, k) / math.comb(candidates, k)


def _references(
    results: list[ProblemResult],
    *,
    regime_column: str | None,
    regret_tail_threshold: float,
) -> dict[str, Any]:
    """§11.4's oracle, static-order and random references, over the scored problems.

    Static order is the corpus row order (engine enumeration order). Random has no
    ranking, so its figures are exact expectations over `V(p)`, not sampled draws.
    """
    count = len(results)

    def tail(problems: float) -> dict[str, Any]:
        return {
            "threshold": regret_tail_threshold,
            "problems": problems,
            "fraction": (problems / count) if count else None,
        }

    def recall(rank: Callable[[ProblemResult], int]) -> dict[str, float | None]:
        return {
            str(k): (sum(rank(item) < k for item in results) / count) if count else None
            for k in TOP_K_VALUES
        }

    def expected(
        value: Callable[[ProblemResult, int], float],
    ) -> dict[str, float | None]:
        return {
            str(k): (
                float(np.mean([value(item, k) for item in results])) if count else None
            )
            for k in TOP_K_VALUES
        }

    return {
        "definition": (
            "RFC 0019.13 §11.4: the model's figures above are only readable against "
            "the ordering it replaces and the floor it must clear, so both are "
            "recomputed here over the same scored problems."
        ),
        "oracle": {
            "problems": count,
            "top1_regret": _summarise([0.0] * count),
            "regret_tail": tail(0),
            "topk_recall": {
                "strict": {str(k): (1.0 if count else None) for k in TOP_K_VALUES},
                "tie_aware": {str(k): (1.0 if count else None) for k in TOP_K_VALUES},
            },
            "per_regime": _regime_means(results, lambda item: 0.0, regime_column),
            "note": (
                "v*(p) by construction: regret 0, and the oracle is its own top pick. "
                "The upper bound of the scale, not an achievable model."
            ),
        },
        "static_order": {
            "problems": count,
            "top1_regret": _summarise([item.static_order_regret for item in results]),
            "regret_tail": tail(
                sum(
                    item.static_order_regret > regret_tail_threshold for item in results
                )
            ),
            "topk_recall": {
                "strict": recall(lambda item: item.static_order_oracle_rank),
                "tie_aware": recall(lambda item: item.static_order_tied_rank),
            },
            "per_regime": _regime_means(
                results, lambda item: item.static_order_regret, regime_column
            ),
            "note": (
                "Stage 1's shipped priority/id ordering (§2.1), read off the corpus: a "
                "problem's rows are in the order the engine enumerated its candidates, "
                "so the static pick is the first row that carries a usable measurement "
                "and the static ranking is the row order itself. A corpus whose rows "
                "were re-sorted after collection does not carry that order, and this "
                "reference is then a permutation rather than the shipped one."
            ),
        },
        "random": {
            "problems": count,
            "top1_regret": _summarise([item.random_regret for item in results]),
            "regret_tail": tail(
                float(sum(item.random_tail_fraction for item in results))
            ),
            "topk_recall": {
                "strict": expected(
                    lambda item, k: min(k, item.candidates) / item.candidates
                ),
                "tie_aware": expected(
                    lambda item, k: _random_tie_recall(
                        item.candidates, item.tied_candidates, k
                    )
                ),
            },
            "per_regime": _regime_means(
                results, lambda item: item.random_regret, regime_column
            ),
            "note": (
                "Uniform choice from V(p), §11.4's sanity floor. Every figure is an "
                "exact expectation over the candidate set rather than a sampled draw, "
                "so `regret_tail.problems` is an expected count and can be fractional."
            ),
        },
    }


def _build_report(
    results: list[ProblemResult],
    *,
    exclusions: Exclusions,
    grouping: Grouping,
    split: Split,
    target: str,
    objective: str,
    metric: str | None,
    regime_column: str | None,
    tie_rel_tolerance: float,
    tie_sigma: float,
    noise_available: bool,
    noise_absent_reason: str | None,
    regret_tail_threshold: float,
    corpus_rows: int,
    corpus_problems: int,
    calibration: dict[str, Any],
    warnings: list[str],
    group_column: str | None = None,
) -> dict[str, Any]:
    regrets = [item.regret for item in results]
    tail = [item for item in results if item.regret > regret_tail_threshold]

    recall: dict[str, dict[str, float | None]] = {
        "strict": {},
        "tie_aware": {},
        "trivial": {},
    }
    for k in TOP_K_VALUES:
        if results:
            recall["strict"][str(k)] = sum(
                item.oracle_rank < k for item in results
            ) / len(results)
            recall["tie_aware"][str(k)] = sum(
                item.tied_rank < k for item in results
            ) / len(results)
            # Problems with at most k candidates hit regardless of the model.
            recall["trivial"][str(k)] = sum(
                item.candidates <= k for item in results
            ) / len(results)
        else:
            recall["strict"][str(k)] = recall["tie_aware"][str(k)] = recall["trivial"][
                str(k)
            ] = None

    # Absent section means "not a two-layer model", not "the split came out zero".
    split_results = [item for item in results if item.group_regret is not None]
    two_stage = None
    if split_results:
        two_stage = {
            "group_column": group_column,
            "problems": len(split_results),
            "group_regret": _summarise([item.group_regret for item in split_results]),
            "in_group_regret": _summarise(
                [item.in_group_regret for item in split_results]
            ),
            "note": (
                "Both parts are measured against the same oracle, so they sum to "
                "top1_regret. group_regret is what choosing the group cost -- the best "
                "member of the chosen group against the best anywhere -- and "
                "in_group_regret is what choosing within it cost."
            ),
        }

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
        per_regime = _regime_means(results, lambda item: item.regret, regime_column)
        per_regime_status = f"from column {regime_column!r}"

    references = _references(
        results,
        regime_column=regime_column,
        regret_tail_threshold=regret_tail_threshold,
    )
    static_regret = references["static_order"]["top1_regret"]["mean"]
    model_regret = _summarise(regrets)["mean"]
    # §11.4 MUST 2: merely matching static order is not beating it; epsilon absorbs
    # float wobble on an exact tie.
    if model_regret is not None and model_regret >= static_regret - _REGRET_EPSILON:
        warnings.append(
            "MODEL DOES NOT BEAT STATIC ORDER: mean top-1 regret "
            f"{model_regret:.4f} against the shipped priority/id ordering's "
            f"{static_regret:.4f} over the same {len(results)} problem(s). RFC 0019.13 "
            "§11.4: a model that loses to the ordering it replaces is the one case "
            "where shipping is almost certainly wrong. This does not gate emission -- "
            "§11.4 leaves that judgement to the author."
        )
    # §11.4 MUST 3: an aggregate gain can hide a regression in one regime.
    static_per_regime = references["static_order"]["per_regime"]
    if per_regime is not None and static_per_regime is not None:
        losing = [
            name
            for name, values in per_regime.items()
            if values["mean_regret"]
            > static_per_regime[name]["mean_regret"] + _REGRET_EPSILON
        ]
        if losing:
            warnings.append(
                "REGIME REGRESSION AGAINST STATIC ORDER: "
                + ", ".join(
                    f"{name} {per_regime[name]['mean_regret']:.4f} vs "
                    f"{static_per_regime[name]['mean_regret']:.4f}"
                    for name in losing
                )
                + ". RFC 0019.13 §11.4: aggregate improvement can hide a regression "
                "confined to the workloads a heuristic was built for."
            )

    return {
        "schema": REPORT_SCHEMA,
        "rfc": "0019.13 §11.2, §11.4",
        "generated": datetime.now(timezone.utc).isoformat(),
        "corpus": {"rows": corpus_rows, "problems": corpus_problems},
        # The ranking metric the score estimates (RFC 0019 §4.4); null for a metric-less
        # ranker. Regret is in `target`.
        "metric": metric,
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
            # Recorded even when false, to distinguish from reports predating the check.
            "deterministic_catalog": exclusions.deterministic_catalog(len(results)),
            "top1_regret": _summarise(regrets),
            "regret_tail": {
                "threshold": regret_tail_threshold,
                "problems": len(tail),
                "fraction": (len(tail) / len(results)) if results else None,
            },
            "topk_recall": recall,
            "two_stage": two_stage,
            "per_regime": per_regime,
            "calibration": calibration,
            "per_regime_status": per_regime_status,
            "references": references,
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
                    else f" (no stddevMs noise band: {noise_absent_reason})"
                )
                + "."
            ),
        },
        "not_implemented": [
            "§11.2 weighted aggregates: regret weighted by declared regime weights "
            "(§5.2). Nothing in this pipeline declares weights, so only the unweighted "
            "figure is computed -- which §11.2 says is the whole report for a blindly "
            "generated corpus, but a corpus that does declare weights needs both.",
            "§11.3 shape extrapolation (leave-one-regime-out) and variant "
            "extrapolation (leave-variants-out). Both need retraining per fold; this "
            "command scores one already-trained model.",
            "§5.6.3/§5.6.4 round-0 core versus full slice, and the steering/reserved "
            "portions of the evaluation slice. Those are properties of a corpus "
            "collected by the campaign loop, which does not exist yet; the split here "
            "is over whatever corpus it is given.",
            "§11.4 MUST 4: regret regression against a PREVIOUS UHD for the same engine "
            "and arch. `evaluate` scores one --model-dir and has no loader for an "
            "already-promoted descriptor, so there is no prior figure to compare "
            "against; the comparison is declared here rather than silently skipped.",
            "§11.4 item 5's per-candidate scoring time against RFC 0019 §9's `native` "
            "baseline. §11.4 itself notes the measurement harness of §11.6 (B5) does "
            "not exist yet, so the number has no reference to be reported against.",
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
    descriptor: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    role: str | None = None
    #: The feature layer 1 grouped on; the artifact only has a slot index, not a column.
    group_feature: str | None = None


def _flatbuffer_scorer(
    artifact: Path,
    features: list[str],
    categorical_encoding: dict[str, dict[str, int]] | None = None,
    *,
    signature: list | None = None,
    feature_evaluator: str | None = None,
    expected_hash: str | None = None,
    model_hash: str | None = None,
    objective: str | None = None,
    score_transform: str = TRAINED_TRANSFORM,
    engine_prediction: bool = False,
    physical: bool,
) -> Scorer:
    """Score with the shipped `model.bin`, summed as `TreeDataAdapter::score()` does.

    `train` deletes `model.lgbm` by default. The bytes pass the loader's own checks
    (`verify_tree_artifact`, features hash) first, so a file the engine would refuse is
    never reported with a regret.
    """
    import uhd_gen  # noqa: F401  puts _generated/ on sys.path

    from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT

    from .artifact import verify_tree_artifact
    from .train_uhd import build_feature_matrix

    # The signature's slot count is the arity the runtime holds the artifact to.
    model = GbdtModelT.InitFromPackedBuf(
        bytearray(
            verify_tree_artifact(
                artifact,
                model_hash,
                feature_count=len(signature) if signature else None,
            )
        ),
        0,
    )
    if expected_hash is not None:
        stored_hash = (
            model.featuresHash.decode("utf-8")
            if isinstance(model.featuresHash, bytes)
            else model.featuresHash
        )
        if stored_hash != expected_hash:
            raise ValueError(
                "descriptor features_hash does not match the shipped model artifact"
            )

    def routing(values, count: int, absent: bool) -> np.ndarray:
        """A per-node flag vector as `prepareTrees` reads it: supplied entries, then `absent`."""
        flags = np.full(count, absent, dtype=bool)
        supplied = np.asarray([] if values is None else values, dtype=bool)[:count]
        flags[: len(supplied)] = supplied
        return flags

    def arrays_of(trees) -> list[tuple[np.ndarray, ...]]:
        # `default_left`/`decision_lte` may be short, read as TreeDataAdapter does: missing
        # `default_left` is false; absent or empty `decision_lte` is `<=` everywhere, but a
        # nonempty short one is `<` past its end.
        arrays = []
        for tree in trees or []:
            count = len(tree.leftChildren)
            lte = tree.decisionLte
            arrays.append(
                (
                    np.asarray(tree.featureIndices, dtype=np.int64),
                    np.asarray(tree.thresholds, dtype=np.float64),
                    np.asarray(tree.leftChildren, dtype=np.int64),
                    np.asarray(tree.rightChildren, dtype=np.int64),
                    np.asarray(tree.leafValues, dtype=np.float64),
                    routing(tree.defaultLeft, count, False),
                    routing(lte, count, lte is None or len(lte) == 0),
                )
            )
        return arrays

    trees = arrays_of(model.trees)
    base = float(model.baseScore)

    # Two-layer artifact: reading only `trees` would silently score layer 1 alone.
    group_slot = int(
        model.groupByFeatureIndex if model.groupByFeatureIndex is not None else -1
    )
    groups = {
        float(group.value): arrays_of(group.trees) for group in (model.groups or [])
    }
    # EnginePredictor refuses to bind a grouped artifact: an L1 score is per row.
    if groups and engine_prediction:
        raise ValueError(
            f"{artifact}: a grouped tree_data artifact cannot be bound to the predict_engine "
            "role; grouped L1 models have no per-row contract"
        )
    if groups and objective not in ("max", "min"):
        raise ValueError(
            f"{artifact}: a grouped artifact chooses its group in the objective's "
            f"direction, and the descriptor declares none ({objective!r})"
        )

    def ensemble(
        arrays: list[tuple[np.ndarray, ...]], matrix: np.ndarray, rows: np.ndarray
    ) -> np.ndarray:
        total = np.full(len(rows), base, dtype=np.float64)
        for feature_index, threshold, left, right, leaf, default_left, lte in arrays:
            node = np.zeros(len(rows), dtype=np.int64)
            # Vectorised: descend every row one level per iteration.
            while True:
                internal = left[node] >= 0
                if not internal.any():
                    break
                at = np.flatnonzero(internal)
                here = node[at]
                x = matrix[rows[at], feature_index[here]]
                # `decision_lte` false means `<` (schema, TreeDataAdapter), not `>`.
                go_left = np.where(lte[here], x <= threshold[here], x < threshold[here])
                go_left = np.where(np.isnan(x), default_left[here], go_left)
                node[at] = np.where(go_left, left[here], right[here])
            total += leaf[node]
        return total

    def recover(raw: np.ndarray) -> np.ndarray:
        return invert_score(raw, score_transform)

    def score(frame: pd.DataFrame) -> np.ndarray:
        matrix = build_feature_matrix(
            frame,
            features,
            categorical_encoding,
            signature=signature,
            feature_evaluator=feature_evaluator,
        )
        every = np.arange(len(frame))
        layer_one = ensemble(trees, matrix, every)
        if group_slot < 0 or not groups:
            return recover(layer_one)

        # The group is chosen per problem (this frame), as TreeDataAdapter::scoreBatch
        # does. Only rows naming a group with a rankable layer-1 score may choose; best raw
        # score in the objective's direction wins, exact ties go to the smaller group value.
        # No eligible row: every candidate is unusable, i.e. declared order.
        group_values = matrix[:, group_slot]
        eligible = ~np.isnan(group_values) & rankable_scores(
            recover(layer_one), physical
        )
        raw = np.full(len(frame), -np.inf, dtype=np.float64)
        if not eligible.any():
            return raw
        best = (
            layer_one[eligible].min()
            if objective == "min"
            else layer_one[eligible].max()
        )
        chosen = group_values[eligible & (layer_one == best)].min()
        inside = np.flatnonzero(group_values == chosen)
        # A group layer 2 does not describe is ranked by layer 1, matching the adapter.
        within = groups.get(float(chosen))
        raw[inside] = ensemble(within, matrix, inside) if within else layer_one[inside]

        # Rejected groups' -inf survives the inverse, so they stay unusable (`rankScored`).
        return recover(raw)

    return score


def _booster_scorer(
    model_file: Path,
    features: list[str],
    categorical_encoding: dict[str, dict[str, int]] | None = None,
    *,
    signature: list | None = None,
    feature_evaluator: str | None = None,
    score_transform: str = TRAINED_TRANSFORM,
) -> Scorer:
    import lightgbm as lgb

    from .train_uhd import build_feature_matrix

    booster = lgb.Booster(model_file=str(model_file))

    def score(frame: pd.DataFrame) -> np.ndarray:
        values = booster.predict(
            build_feature_matrix(
                frame,
                features,
                categorical_encoding,
                signature=signature,
                feature_evaluator=feature_evaluator,
            )
        )
        return invert_score(values, score_transform)

    return score


def load_model(
    model_dir: Path,
    model_file: Path | None = None,
    *,
    feature_evaluator: str | None = None,
    runtime_predictions: list[dict] | None = None,
) -> ModelBundle:
    """Load a `train --output-dir` result: features, direction, and something to rank with."""
    descriptor_paths = sorted(model_dir.glob("*.uhd.json"))
    if len(descriptor_paths) > 1:
        raise ValueError(
            f"{model_dir} has multiple UHD descriptors; use a directory containing one model"
        )
    descriptor = _load_json(descriptor_paths[0]) if descriptor_paths else {}
    manifest_path = model_dir / "train_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    role = manifest.get("role")
    if role is None:
        # Promoted models ship without a manifest; recover the role from the install
        # layout `<ued-id>/<role>/<arch>/[<metric>/]` (RFC 0019 §3.1).
        from .provenance import ROLES

        for ancestor in model_dir.resolve().parents[:2]:
            if ancestor.name in ROLES:
                role = ancestor.name
                break
    from .immediate import ROLE, validate_model

    immediate = role == ROLE
    l1_metric = validate_model(descriptor) if immediate else None

    from .features import (
        build_features_signature,
        compute_features_hash,
        evaluator_feature_semantics_revision,
        signature_references,
    )
    from .artifact import is_contained_relative_path
    from .provenance import require_feature_semantics

    signature = descriptor.get("features_signature") or manifest.get(
        "features_signature"
    )
    if not signature and manifest.get("features"):
        signature = build_features_signature(manifest["features"])
    if not signature and not (
        immediate and descriptor.get("adapter") in ("native", "custom_library")
    ):
        raise ValueError(f"{model_dir} carries no features_signature")
    signature = signature or []
    features = [reference[1:] for reference in signature_references(signature)]

    # Read, never assumed: a backwards objective inverts every number in the report.
    objective = descriptor.get("objective") or manifest.get("objective")

    group_feature = manifest.get("group_by_feature")

    categorical_encoding = descriptor.get(
        "categorical_encoding", manifest.get("categorical_encoding", {})
    )
    expected_hash = descriptor.get("features_hash", manifest.get("features_hash"))
    if expected_hash is not None:
        # RFC 0019 §6.3: recompute the digest with the shared evaluator generation used.
        if (
            compute_features_hash(signature, categorical_encoding, feature_evaluator)
            != expected_hash
        ):
            raise ValueError(
                "features_signature/categorical_encoding does not match features_hash"
            )
        # The loader refuses models trained under other feature semantics
        # (FeatureSemantics.hpp), so report nothing for one.
        if signature:
            require_feature_semantics(
                descriptor.get("trained_against", manifest.get("trained_against")),
                evaluator_feature_semantics_revision(feature_evaluator),
            )

    if descriptor:
        # The engine reads an absent or empty `score.transform` as identity
        # (ScoreTransform.hpp); the descriptor wins over the manifest.
        transform = (descriptor.get("score") or {}).get("transform") or "identity"
    else:
        # A manifest without the key predates `log`; those models were all log1p.
        transform = manifest.get("score_transform", "log1p")
    if transform not in INVERTIBLE_TRANSFORMS:
        # The engine supports more transforms; this module just lacks their inverse.
        raise ValueError(
            f"uhd_gen can only score {', '.join(INVERTIBLE_TRANSFORMS)} transforms; this "
            f"descriptor declares {transform!r}, which the runtime loads but `evaluate` cannot invert"
        )
    if runtime_predictions is not None:
        if not immediate:
            raise ValueError(
                "--predictions is only supported for engine-immediate evaluation"
            )
        from .immediate import prediction_scorer

        scorer = prediction_scorer(descriptor, runtime_predictions)
        candidate = Path("<runtime-predictions>")
    else:
        if descriptor.get("adapter") in ("native", "custom_library"):
            raise ValueError(
                "native/custom models require --predictions from hipdnn_bench --predict-engine"
            )
        declared = descriptor.get("tree_data", {})
        named = None
        if "artifact" in declared:
            artifact = declared["artifact"]
            # UhdParser's rule, so an artifact the runtime would never open is not scored.
            if (
                not isinstance(artifact, str)
                or not artifact
                or not is_contained_relative_path(artifact)
            ):
                raise ValueError(
                    f"tree_data.artifact {artifact!r} must be a relative path inside the "
                    "descriptor's directory"
                )
            named = model_dir / artifact
        if model_file is not None:
            candidate = model_file
        elif named is not None:
            candidate = named
        elif (model_dir / "model.lgbm").exists() and not group_feature:
            # A grouped model's `model.lgbm` holds layer 1 only; use `model.bin` instead.
            candidate = model_dir / "model.lgbm"
        else:
            candidate = model_dir / "model.bin"
        if not candidate.exists():
            raise ValueError(f"no model artifact at {candidate}")
        if candidate.suffix in (".lgbm", ".txt"):
            scorer = _booster_scorer(
                candidate,
                features,
                categorical_encoding,
                signature=signature,
                feature_evaluator=feature_evaluator,
                score_transform=transform,
            )
        else:
            # The declared digest covers only the artifact the descriptor names.
            model_hash = (
                declared.get("hash")
                if named is not None and candidate.resolve() == named.resolve()
                else None
            )
            scorer = _flatbuffer_scorer(
                candidate,
                features,
                categorical_encoding,
                signature=signature,
                feature_evaluator=feature_evaluator,
                expected_hash=expected_hash,
                model_hash=model_hash,
                objective=objective,
                score_transform=transform,
                engine_prediction=immediate,
                physical=score_is_physical(descriptor.get("score")),
            )

    return ModelBundle(
        scorer=scorer,
        features=list(features),
        target=manifest.get("target", l1_metric.label if l1_metric else None),
        objective=objective,
        source=str(candidate),
        trained_on=manifest.get("input_file"),
        training_rows=manifest.get("num_samples"),
        descriptor=descriptor,
        manifest=manifest,
        role=role,
        group_feature=group_feature,
    )


def _holdout_integrity(
    bundle: ModelBundle, evaluated: Iterable[Sequence[str]]
) -> dict[str, str]:
    """Whether the model's recorded training problem keys overlap the evaluated ones.

    Without recorded keys the answer is `unknown`: a file name proves nothing.
    """
    evaluated = {tuple(str(part) for part in key) for key in evaluated}
    recorded = bundle.manifest.get("training_problem_keys")
    if recorded is None:
        trained_on = getattr(bundle, "trained_on", None)
        trained_on = f" ({trained_on})" if trained_on else ""
        return {
            "status": "unknown",
            "detail": f"the model records no training problem keys{trained_on}, so whether it "
            "saw this corpus's evaluation problems cannot be shown; if it did, the regret "
            "below is optimistic (RFC 0019.13 §5.6.4).",
        }
    trained = {tuple(str(part) for part in key) for key in recorded}
    # Keys of different widths (one corpus lacks device identity) are compared on the
    # graph alone; a shared graph may then be from another device, so that is unknown.
    narrowed = len({len(key) for key in trained | evaluated}) > 1
    if narrowed:
        trained = {key[:1] for key in trained}
        evaluated = {key[:1] for key in evaluated}
    overlap = trained & evaluated
    if overlap and narrowed:
        return {
            "status": "unknown",
            "detail": "training and evaluation keys differ in width (one corpus has no device "
            "identity) and share graphs, so disjointness cannot be shown",
        }
    if overlap:
        return {
            "status": "COMPROMISED",
            "detail": f"{len(overlap)} of {len(evaluated)} evaluated problem(s) are among the "
            "model's recorded training problems. §5.6.4: scoring a problem with a model that "
            "trained on it is a leak, and the regret below is optimistic by an unknown "
            "amount. Use --emit-train-slice to write the training side of this split, train "
            "on THAT, then evaluate again with the same --seed and --eval-fraction.",
        }
    return {
        "status": "held_out",
        "detail": f"none of the {len(evaluated)} evaluated problem(s) is among the "
        f"{len(trained)} recorded training problem(s)",
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def add_evaluate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        required=True,
        help="Corpus to evaluate on: the published .parquet dataset, or a collected .csv/.json corpus",
    )
    parser.add_argument(
        "--feature-evaluator", help="Path to the shared hipdnn_uhd_features executable"
    )
    parser.add_argument(
        "--additional-model-dir",
        action="append",
        default=[],
        help="Another engine's L1 model; repeat for cross-engine immediate comparison",
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        help="Runtime --predict-engine JSON responses for common native/custom model evaluation",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        dest="model_dir",
        help="A `train --output-dir` result: descriptor, model artifact, manifest",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model artifact override (default: the descriptor's tree_data.artifact, "
        "the file the engine itself loads)",
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
        "--group-column",
        default=None,
        dest="group_column",
        help="Column naming the group a candidate belongs to, for the two-stage regret "
        "split of a grouped model (default: the manifest's `group_by_feature`). Only "
        "affects reporting; the model's own grouping is read from the artifact.",
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


def run_evaluate(args: argparse.Namespace) -> int:
    corpus_path = Path(args.input)
    model_dir = Path(args.model_dir)

    predictions = None
    if args.predictions:
        try:
            predictions = []
            for path in args.predictions:
                content = _load_json(Path(path))
                predictions.extend(content if isinstance(content, list) else [content])
        except (OSError, ValueError) as error:
            logger.error("%s", error)
            return 1
    try:
        bundle = load_model(
            model_dir,
            Path(args.model) if args.model else None,
            feature_evaluator=args.feature_evaluator,
            runtime_predictions=predictions,
        )
    except (ValueError, OSError) as error:
        logger.error("%s", error)
        return 1
    from .immediate import ROLE

    # A registered metric fixes the direction (RFC 0019 §4.4); refuse a contradiction.
    declared = bundle.descriptor.get("score", {}).get("metric")
    if declared in RANKING_METRICS and args.objective not in (
        None,
        RANKING_METRICS[declared].objective,
    ):
        logger.error(
            "score.metric %r ranks %s; --objective %s contradicts it",
            declared,
            RANKING_METRICS[declared].objective,
            args.objective,
        )
        return 1
    if bundle.role == ROLE:
        from .immediate import evaluate_immediate, read_corpus

        try:
            if args.target not in (None, RANKING_METRICS[declared].label):
                raise ValueError(
                    f"L1 evaluation of {declared!r} cannot override its calibrated label"
                )
            if args.device_column not in (None, "device"):
                raise ValueError(
                    "L1 evaluation groups by the recorded graph/device identity"
                )
            bundles = [bundle] + [
                load_model(
                    Path(path),
                    feature_evaluator=args.feature_evaluator,
                    runtime_predictions=predictions,
                )
                for path in args.additional_model_dir
            ]
            frame = read_corpus(corpus_path)
            report = evaluate_immediate(
                frame,
                bundles,
                eval_fraction=args.eval_fraction,
                seed=args.seed,
                include_per_problem=args.include_per_problem,
                feature_evaluator=args.feature_evaluator,
            )
            report["corpus"]["path"] = str(corpus_path)
            report["models"] = [
                {"artifact": item.source, "uhd_id": item.descriptor.get("id")}
                for item in bundles
            ]
            if args.emit_train_slice:
                keys = problem_keys(frame, resolve_grouping(frame))
                held_out = {tuple(key) for key in report["split"]["eval_problem_keys"]}
                slice_path = Path(args.emit_train_slice)
                slice_path.parent.mkdir(parents=True, exist_ok=True)
                frame[~keys.isin(held_out)].to_csv(
                    slice_path, index=False, lineterminator="\n"
                )
                report["split"]["train_slice"] = str(slice_path)
            output_path = (
                Path(args.output) if args.output else model_dir / "eval_report.json"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            metrics = report["metrics"]
            print(f"Immediate prediction report: {output_path}")
            print(f"  calibration: {json.dumps(metrics['calibration'])}")
            print(
                f"  cross-engine selection: {json.dumps(metrics['immediate_selection'])}"
            )
            print(f"  holdout integrity: {report['holdout_integrity']['status']}")
            return 0
        except (OSError, TypeError, ValueError, KeyError) as error:
            logger.error("%s", error)
            return 1
    if args.additional_model_dir or args.predictions:
        logger.error(
            "additional models and runtime predictions require %s models", ROLE
        )
        return 1

    # Same reader as the trainer, so a .parquet-trained model is scored against the
    # same typed dataset.
    df = read_corpus_frame(corpus_path)
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
            group_column=args.group_column or bundle.group_feature,
            tie_rel_tolerance=args.tie_rel_tolerance,
            tie_sigma=args.tie_sigma,
            regret_tail_threshold=args.regret_tail_threshold,
            score_declaration=bundle.descriptor.get("score"),
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
    report["holdout_integrity"] = _holdout_integrity(
        bundle, report["split"]["eval_problem_keys"]
    )
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
        df[~keys.isin(held_out)].to_csv(slice_path, index=False, lineterminator="\n")
        report["split"]["train_slice"] = str(slice_path)
        logger.info("Wrote training slice to %s", slice_path)

    output_path = Path(args.output) if args.output else model_dir / "eval_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    _print_summary(report, output_path)
    return 0


def _print_summary(report: dict[str, Any], output_path: Path) -> None:
    metrics = report["metrics"]
    regret = metrics["top1_regret"]

    # Warnings first: a degraded grouping or compromised holdout invalidates the figures.
    for warning in report["warnings"]:
        print(f"\n!! {warning}")

    print(f"\nRegret report ({report['rfc']}) -- {output_path}")
    print(
        f"  metric:             {report.get('metric') or '(none: ranks its own catalog only)'}"
    )
    print(f"  target/objective:   {report['target']} ({report['objective']})")
    print(f"  problems grouped by: {', '.join(report['grouping']['columns'])}")
    print(
        f"  split:              {report['split']['method']}, seed {report['split']['seed']}, "
        f"{report['split']['eval_problems']} eval / {report['split']['train_problems']} train problem(s)"
    )
    print(f"  problems scored:    {metrics['problems_scored']}")
    if regret["mean"] is None:
        print("  top-1 regret:       n/a (no problem had two measured candidates)")
        if metrics.get("deterministic_catalog"):
            # Explain why it is empty, so a working sweep is not debugged needlessly.
            print(
                "                      all "
                f"{report['exclusions']['problems_single_candidate']} problem(s) had a "
                "single candidate: this engine's kernel choice is a total function of the\n"
                "                      problem, so there is no ordering for a ranking "
                "model to learn. Train --role predict_engine instead."
            )
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
    references = metrics["references"]
    if regret["mean"] is not None:
        # §11.4: the model's regret is only meaningful beside the references.
        print(
            "  vs §11.4 references: static order "
            f"{references['static_order']['top1_regret']['mean']:.4f}   "
            f"random {references['random']['top1_regret']['mean']:.4f}   "
            "oracle 0.0000 (mean top-1 regret)"
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
        "  excluded:           {invalid_rows} invalid row(s), "
        "{numerically_invalid_rows} numerically wrong row(s), {missing_target_rows} "
        "row(s) with no target, {problems_single_candidate} single-candidate "
        "problem(s), {problems_no_measured_candidate} problem(s) with nothing "
        "measured, {problems_non_positive_oracle} with a non-positive oracle".format(
            **{key: dropped[key] for key in dropped if key != "policy"}
        )
    )
