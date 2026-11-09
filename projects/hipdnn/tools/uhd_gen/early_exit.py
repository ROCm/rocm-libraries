# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Early-exit schedules for `tree_data` rankers (gbdt_model.fbs `early_exit`).

A schedule is a list of (trees, keep) stages. Ranking a catalog, the runtime
(`TreeDataAdapter::scoreBatch`) sums the first `trees` trees for every candidate still in,
keeps the `keep` best in the objective's direction (ties keep the earlier row), and goes on;
the survivors of the last stage are scored by the whole model. A pruned candidate carries
its partial sum, ranks below every fully scored usable candidate (later stages first, then
by that sum) and reports no estimate.

This module holds the runtime's checks, the same staged ranking for offline evaluation, and
the trainer's choice of a schedule from out-of-fold predictions: the cheapest schedule whose
held-out regret stays within a tolerance of scoring every candidate fully.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

Schedule = Sequence[tuple[int, int]]

#: Prefix lengths and shortlist sizes the trainer tries.
PREFIXES = (5, 10, 20, 30, 40, 60, 100)
KEEPS = (8, 16, 32)
#: A schedule may lose at most this much mean and p95 top-1 regret against full scoring.
MEAN_REGRET_TOLERANCE = 0.0002
P95_REGRET_TOLERANCE = 0.002
#: A schedule saving less than this fraction of tree walks is not worth shipping.
MAX_WORK_FRACTION = 0.8
#: scoreBatch reports a row's stage in a uint8.
MAX_STAGES = 255


def check_schedule(schedule: Schedule, tree_count: int) -> None:
    """Refuse what TreeDataAdapter refuses, with its reasons."""
    if len(schedule) > MAX_STAGES:
        raise ValueError(
            f"early_exit stage {len(schedule)}: the schedule has more than 255 stages"
        )
    previous: tuple[int, int] | None = None
    for index, (trees, keep) in enumerate(schedule):
        if not 1 <= trees < tree_count:
            raise ValueError(
                f"early_exit stage {index}: trees must be at least 1 and fewer than the "
                "model's trees"
            )
        if keep < 1:
            raise ValueError(f"early_exit stage {index}: keep must be at least 1")
        if previous is not None and (trees <= previous[0] or keep >= previous[1]):
            raise ValueError(
                f"early_exit stage {index}: stages must increase in trees and decrease in keep"
            )
        previous = (trees, keep)


def parse_schedule(text: str) -> list[tuple[int, int]]:
    """`"30:16"` or `"10:32,40:8"` -> [(trees, keep), ...]."""
    stages = []
    for part in text.split(","):
        trees, sep, keep = part.strip().partition(":")
        if not sep:
            raise ValueError(f"early-exit stage {part!r} is not TREES:KEEP")
        stages.append((int(trees), int(keep)))
    return stages


def _keep_best(
    values: np.ndarray, alive: np.ndarray, keep: int, lower_is_better: bool
) -> np.ndarray:
    """The `keep` best of `alive` (row indices, ascending) by `values`; ties keep the
    earlier row, as the runtime's nth_element with an index tie-break does."""
    if len(alive) <= keep:
        return alive
    key = values[alive] if lower_is_better else -values[alive]
    order = np.lexsort((alive, key))
    return np.sort(alive[order[:keep]])


@dataclass
class StagedScores:
    """`raw`: the full score for rows `stage == stages`, otherwise the partial sum over the
    trees of the stage the row was pruned at. `stage`: stages each row survived."""

    raw: np.ndarray
    stage: np.ndarray
    stages: int

    @property
    def pruned(self) -> np.ndarray:
        return self.stage < self.stages


def staged_scores(
    add_trees: Callable[[np.ndarray, np.ndarray, int, int], None],
    count: int,
    tree_count: int,
    schedule: Schedule,
    base_score: float,
    lower_is_better: bool,
) -> StagedScores:
    """Score `count` rows as TreeDataAdapter::scoreBatch does with `schedule`.

    `add_trees(acc, rows, begin, end)` adds, in tree order, the leaves of trees
    [begin, end) to `acc[rows]`; summing tree by tree from 0 is what makes a survivor's
    score the full model's bit for bit.
    """
    partial = np.zeros(count, dtype=np.float64)
    stage = np.zeros(count, dtype=np.int64)
    alive = np.arange(count)
    done = 0
    for index, (trees, keep) in enumerate(schedule):
        add_trees(partial, alive, done, trees)
        done = trees
        survivors = _keep_best(partial, alive, keep, lower_is_better)
        stage[np.setdiff1d(alive, survivors, assume_unique=True)] = index
        alive = survivors
    add_trees(partial, alive, done, tree_count)
    raw = partial.copy()
    raw[alive] = base_score + partial[alive]
    stage[alive] = len(schedule)
    return StagedScores(raw=raw, stage=stage, stages=len(schedule))


def staged_pick(
    prefix: Mapping[int, np.ndarray],
    full: np.ndarray,
    schedule: Schedule,
    lower_is_better: bool,
) -> int:
    """The row a ranker picks with `schedule`, from prefix sums of one catalog: the best
    survivor by full score (ties: earlier row). `prefix[t]` holds every row's sum of the
    first t trees."""
    alive = np.arange(len(full))
    for trees, keep in schedule:
        alive = _keep_best(prefix[trees], alive, keep, lower_is_better)
    return int(_keep_best(full, alive, 1, lower_is_better)[0])


def work_fraction(
    catalog_sizes: Sequence[int], schedule: Schedule, tree_count: int
) -> float:
    """Tree walks with `schedule` over walks scoring every row, mean over catalogs."""
    fractions = []
    for size in catalog_sizes:
        live, done, walks = size, 0, 0
        for trees, keep in schedule:
            walks += live * (trees - done)
            done, live = trees, min(live, keep)
        walks += live * (tree_count - done)
        fractions.append(walks / (size * tree_count))
    return float(np.mean(fractions)) if fractions else 1.0


def candidate_schedules(tree_count: int) -> list[list[tuple[int, int]]]:
    """Single stages (p, k), and two stages (p1, 32) -> (p2, 8) with p1 < p2."""
    prefixes = [p for p in PREFIXES if p < tree_count]
    schedules: list[list[tuple[int, int]]] = [[(p, k)] for p in prefixes for k in KEEPS]
    schedules += [
        [(p1, 32), (p2, 8)] for i, p1 in enumerate(prefixes) for p2 in prefixes[i + 1 :]
    ]
    return schedules


@dataclass
class ScheduleChoice:
    """The chosen schedule (empty: none) and every candidate's held-out figures."""

    schedule: list[tuple[int, int]]
    full: dict[str, float]
    candidates: list[dict] = field(default_factory=list)

    def to_manifest(self) -> dict:
        return {
            "schedule": [list(stage) for stage in self.schedule],
            "full_model": self.full,
            "tolerance": {
                "mean_regret": MEAN_REGRET_TOLERANCE,
                "p95_regret": P95_REGRET_TOLERANCE,
                "max_work_fraction": MAX_WORK_FRACTION,
            },
            "candidates": self.candidates,
        }


def _regret(picked: float, best: float, objective: str) -> float:
    return picked / best - 1.0 if objective == "min" else 1.0 - picked / best


def choose_schedule(
    catalogs: Sequence[tuple[Mapping[int, np.ndarray], np.ndarray, np.ndarray]],
    tree_count: int,
    objective: str,
) -> ScheduleChoice:
    """Pick the cheapest schedule within tolerance of full scoring.

    `catalogs`: per held-out problem, (prefix sums by tree count, full scores, measured
    target). Raw scores rank in `objective`'s direction (every score transform is
    increasing). Prefix sums must cover every prefix in `candidate_schedules(tree_count)`.
    """
    lower = objective == "min"

    def regrets(schedule: Schedule) -> np.ndarray:
        values = []
        for prefix, full, measured in catalogs:
            best = measured.min() if lower else measured.max()
            pick = staged_pick(prefix, full, schedule, lower)
            values.append(
                max(_regret(float(measured[pick]), float(best), objective), 0.0)
            )
        return np.asarray(values)

    def figures(values: np.ndarray, work: float) -> dict[str, float]:
        return {
            "mean_regret": float(values.mean()) if len(values) else 0.0,
            "p95_regret": float(np.quantile(values, 0.95)) if len(values) else 0.0,
            "work_fraction": work,
        }

    sizes = [len(full) for _, full, _ in catalogs]
    full = figures(regrets([]), 1.0)
    choice = ScheduleChoice(schedule=[], full=full)
    best_work = MAX_WORK_FRACTION
    for schedule in candidate_schedules(tree_count):
        work = work_fraction(sizes, schedule, tree_count)
        result = {
            "schedule": [list(stage) for stage in schedule],
            **figures(regrets(schedule), work),
        }
        choice.candidates.append(result)
        within = (
            result["mean_regret"] <= full["mean_regret"] + MEAN_REGRET_TOLERANCE
            and result["p95_regret"] <= full["p95_regret"] + P95_REGRET_TOLERANCE
        )
        if within and work < best_work:
            best_work = work
            choice.schedule = list(schedule)
    return choice


def out_of_fold_catalogs(
    cross_validation: Mapping,
    problem_ids: np.ndarray,
    measured: np.ndarray,
) -> list[tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]]:
    """Held-out catalogs for `choose_schedule` from train_uhd.train_model's
    `cross_validation` record: each fold's booster scores only its validation rows, at
    every candidate prefix and with the final model's tree count. Rows are grouped into
    catalogs by `problem_ids`; `measured` is the target, in the objective's units."""
    folds = cross_validation.get("folds")
    boosters = cross_validation.get("boosters") or []
    if not folds or len(folds) != len(boosters):
        raise ValueError(
            "choosing an early-exit schedule needs problem-grouped cross-validation folds"
        )
    matrix = cross_validation["matrix"]
    tree_count = int(cross_validation["trees"])
    prefixes = sorted(
        {p for schedule in candidate_schedules(tree_count) for p, _ in schedule}
    )
    catalogs = []
    for (_, valid), booster in zip(folds, boosters):
        rows = np.asarray(valid)
        x = matrix[rows]
        full = booster.predict(x, num_iteration=tree_count, raw_score=True)
        prefix = {
            p: booster.predict(x, num_iteration=p, raw_score=True) for p in prefixes
        }
        ids = problem_ids[rows]
        for problem in np.unique(ids):
            at = np.flatnonzero(ids == problem)
            if len(at) < 2:
                continue
            catalogs.append(
                (
                    {p: values[at] for p, values in prefix.items()},
                    full[at],
                    measured[rows][at],
                )
            )
    return catalogs
