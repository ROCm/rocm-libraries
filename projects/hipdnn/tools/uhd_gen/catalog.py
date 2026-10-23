# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Whether an engine's catalog leaves anything to rank.

A catalog is *deterministic* when every problem has one candidate: `sort_kernel_catalog`
has nothing to learn, but `predict_engine` still does. Measured from the corpus, not the
pack, because the native matcher may discriminate on fields the KMD does not declare.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .evaluate import Grouping, problem_keys, resolve_grouping


class DeterministicCatalogError(ValueError):
    """A ranking role was asked of a catalog that ranks nothing."""


@dataclass(frozen=True)
class CatalogDensity:
    """How many candidates the engine offered per problem, across a corpus."""

    problems: int
    single_candidate_problems: int
    max_candidates: int
    #: candidates -> number of problems offering that many.
    histogram: dict[int, int]
    #: The columns a problem was identified by, and whether that identity is degraded.
    grouping: Grouping | None = None

    @property
    def deterministic(self) -> bool:
        """Every problem had exactly one candidate; False for an empty corpus."""
        return self.problems > 0 and self.max_candidates <= 1

    @property
    def rankable_problems(self) -> int:
        """Problems with something to choose between -- the only ones L2 can learn from."""
        return self.problems - self.single_candidate_problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "problems": self.problems,
            "single_candidate_problems": self.single_candidate_problems,
            "rankable_problems": self.rankable_problems,
            "max_candidates": self.max_candidates,
            "deterministic": self.deterministic,
            # JSON object keys are strings; the histogram is emitted into the report.
            "histogram": {
                str(key): value for key, value in sorted(self.histogram.items())
            },
            "grouped_by": list(self.grouping.columns) if self.grouping else None,
        }

    def diagnosis(self, engine: str | None = None) -> str:
        """Why a ranking role cannot be trained here, and what to do instead."""
        subject = f"engine {engine!r}" if engine else "this engine"
        return (
            f"deterministic catalog: all {self.problems} problem(s) in this corpus had "
            f"exactly one candidate, so {subject}'s kernel choice is a total function of "
            "the problem and there is nothing for a ranking model to order. This is a "
            "property of the engine, not a defect in the corpus -- a matcher that pins "
            "every distinguishing field leaves a singleton, and the engine's own score "
            "hook is inert by construction. Train --role predict_engine instead: "
            "the kernel is already chosen, and its measured throughput or time is the "
            "quantity cross-engine arbitration actually compares."
        )

    def near_deterministic_warning(self, threshold: float = 0.95) -> str | None:
        """A warning when nearly all problems have a single candidate; None otherwise."""
        if self.deterministic or self.problems == 0:
            return None
        excluded = self.single_candidate_problems / self.problems
        if excluded < threshold:
            return None
        return (
            f"{self.single_candidate_problems} of {self.problems} problem(s) "
            f"({excluded:.1%}) have a single candidate and are excluded from every "
            f"ranking metric; only {self.rankable_problems} problem(s) carry a choice. "
            "The reported regret describes that minority, not the corpus."
        )


def candidate_density(
    df: pd.DataFrame,
    *,
    grouping: Grouping | None = None,
    device_column: str | None = None,
) -> CatalogDensity:
    """Census the candidates-per-problem distribution of a corpus.

    Problem identity is `evaluate`'s. Rows are counted as given, so unfiltered input
    counts candidates offered rather than measured.
    """
    if df.empty:
        return CatalogDensity(0, 0, 0, {}, grouping)
    grouping = grouping or resolve_grouping(df, device_column)
    per_problem = collections.Counter(problem_keys(df, grouping))
    histogram = collections.Counter(per_problem.values())
    return CatalogDensity(
        problems=len(per_problem),
        single_candidate_problems=histogram.get(1, 0),
        max_candidates=max(per_problem.values()),
        histogram=dict(histogram),
        grouping=grouping,
    )


def require_rankable(
    df: pd.DataFrame,
    *,
    engine: str | None = None,
    grouping: Grouping | None = None,
    device_column: str | None = None,
) -> CatalogDensity:
    """Census the corpus and raise DeterministicCatalogError if it has nothing to rank."""
    density = candidate_density(df, grouping=grouping, device_column=device_column)
    if density.deterministic:
        raise DeterministicCatalogError(density.diagnosis(engine))
    return density
