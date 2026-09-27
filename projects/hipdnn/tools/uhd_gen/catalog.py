# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Whether an engine's catalog leaves anything to rank.

An engine's kernel matcher decides which of its kernels a graph may run on. When that
matcher pins every distinguishing field, the surviving set is a singleton: kernel
identity becomes a total function of the problem, and the engine's own ranking hook has
nothing left to order. The gfx950 dense attention pack is the worked example -- 150
variants, and its `scoreKernel` returns a constant with a comment saying why.

Call such a catalog *deterministic*. The property is worth naming because the two roles
fail differently on one:

  - `sort_kernel_catalog` has no learnable content whatsoever. `evaluate` already
    declines to score a single-candidate problem as regret 0 -- with nothing to choose
    between, a correct pick is not evidence -- so every problem is excluded and the
    report comes back empty. That is the right arithmetic paired with an unhelpful
    diagnosis: an empty report reads as a broken corpus or a bad split, and an operator
    will go and debug one.
  - `predict_engine_tflops` is unaffected, and is the role such an engine actually
    wants. The kernel is already chosen; what nobody knows is how fast it will run, and
    that number is exactly what cross-engine arbitration compares.

So determinism is not a defect to route around. It is a property to detect and report,
so that the operator is sent to the role with something to learn rather than left
staring at a corpus that is behaving correctly.

Measured from the corpus, never from the pack. The matcher is native code this tooling
cannot introspect, and pack metadata only exposes the fields a KMD happens to declare --
a field that discriminates but is not declared reads as overlap that does not exist,
which is the same trap that produced a ranking model for an engine whose UED could not
express the field it ranked on. Counting candidate rows per problem asks the engine what
it did, not what its metadata implies it might have done.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .evaluate import Grouping, problem_keys, resolve_grouping


class DeterministicCatalogError(ValueError):
    """A ranking role was asked of a catalog that ranks nothing.

    Distinct from a plain `ValueError` so a caller that wants to route on the condition
    -- offering L1 rather than aborting, say -- can catch it without pattern-matching a
    message string.
    """


@dataclass(frozen=True)
class CatalogDensity:
    """How many candidates the engine offered per problem, across a corpus."""

    problems: int
    single_candidate_problems: int
    max_candidates: int
    #: candidates -> number of problems offering that many. Kept whole rather than
    #: reduced to a flag: "148 of 150 problems had one candidate" is a different
    #: situation from "all 150 did", and only the histogram tells them apart.
    histogram: dict[int, int]
    #: The columns a problem was identified by, and whether that identity is degraded.
    grouping: Grouping | None = None

    @property
    def deterministic(self) -> bool:
        """Every problem in this corpus had exactly one candidate.

        False for an empty corpus: nothing was observed, so nothing is concluded. A
        caller that treats "no problems" as "deterministic" would refuse a ranking role
        on the strength of a corpus that failed to load.
        """
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
            "histogram": {str(key): value for key, value in sorted(self.histogram.items())},
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
            "hook is inert by construction. Train --role predict_engine_tflops instead: "
            "the kernel is already chosen, and its throughput is the quantity "
            "cross-engine arbitration actually compares."
        )

    def near_deterministic_warning(self, threshold: float = 0.95) -> str | None:
        """A catalog that ranks almost nothing, which L2 will score on almost nothing.

        Not an error: a handful of contested problems is still a real, if thin, ranking
        signal, and refusing it would be this tool deciding a contest it is meant to
        measure. But a model fitted on 5% of its corpus and reported as though it were
        fitted on all of it is a number a reader will misread, so the shortfall is said
        out loud where the model is trained.
        """
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

    Problem identity is `evaluate`'s, reused rather than restated: the same corpus must
    not be one shape here and another there, or a role refused by this census would be
    scored by that report.

    Rows are counted as they arrive. Callers that intend to train have already dropped
    the rows carrying no measurement, and a caller that has not will see the candidates
    the engine *offered* rather than the ones that ran -- which is the right count for
    deciding whether a catalog has ranking density, and the wrong one for deciding
    whether that density survived measurement. `evaluate` answers the second question.
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
    """Census the corpus and refuse a ranking role if it has nothing to rank.

    Returns the census so the caller can report it whether or not it raised -- the
    histogram is worth printing on the way past, not only on the way out.
    """
    density = candidate_density(df, grouping=grouping, device_column=device_column)
    if density.deterministic:
        raise DeterministicCatalogError(density.diagnosis(engine))
    return density
