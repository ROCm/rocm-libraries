# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Optional build-reuse preference over a model's scored, legal candidate pool."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence, Set

from .params import Genome, SearchSpace


@dataclass
class BuildPreference:
    """Evaluator-owned context, refreshed before ask; never an observation.

    Only the best ``rank_fraction`` of the acquisition pool (including boundary
    ties) may displace its leader. ``explore_probability`` keeps a cost-blind
    selection path. Ranks are a heuristic, not calibrated latency differences.
    """

    rank_fraction: float = 0.1
    explore_probability: float = 0.2
    keys: Set[str] = field(default_factory=set)
    selections: int = 0

    def __post_init__(self):
        if not 0 < self.rank_fraction <= 1:
            raise ValueError("rank_fraction must be in (0, 1]")
        if not 0 <= self.explore_probability <= 1:
            raise ValueError("explore_probability must be in [0, 1]")

    def choose(self, space: SearchSpace, rng: random.Random,
               genomes: Sequence[Genome], scores: Sequence[float], best: int) -> Genome:
        if not self.keys or space.key(genomes[best]) in self.keys:
            return genomes[best]
        if not all(math.isfinite(s) for s in scores):
            raise ValueError("build preference requires finite acquisition scores")
        order = sorted(range(len(genomes)), key=lambda i: (-scores[i], i))
        cutoff = scores[order[max(1, math.ceil(len(order) * self.rank_fraction)) - 1]]
        reusable = [i for i in order if scores[i] >= cutoff
                    and space.key(genomes[i]) in self.keys]
        if not reusable or rng.random() < self.explore_probability:
            return genomes[best]
        self.selections += 1
        return genomes[reusable[0]]


def attach_build_preference(strategy, preference: BuildPreference) -> int:
    """Attach to supported model members, returning how many accept it.

    Startup queues and non-model members are intentionally unchanged. Context
    is not checkpointed: the evaluator must restore availability before ask.
    """
    from .portfolio import Portfolio
    from .surrogate import GPEI, TPE

    if isinstance(strategy, (TPE, GPEI)):
        strategy.build_preference = preference
        return 1
    if isinstance(strategy, Portfolio):
        return sum(attach_build_preference(m, preference) for m in strategy.members)
    return 0
