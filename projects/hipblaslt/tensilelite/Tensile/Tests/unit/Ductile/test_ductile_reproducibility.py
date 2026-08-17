# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from itertools import product

import numpy as np
import pytest

from Tensile.ductile.algorithm import GeneticAlgorithm
from Tensile.ductile.core import SearchSpace, Selection, Crossover, Mutation, Mating, Survival
from Tensile.ductile.core.population import Individual, Population

pytestmark = pytest.mark.unit


def _build_population(space, size):
    keys = list(space.sizes.keys())
    indices = [range(space.sizes[k]) for k in keys]
    all_inds = []
    for idx in product(*indices):
        all_inds.append(Individual(dict(zip(keys, idx))))
        if len(all_inds) == size:
            break
    return Population(all_inds)


def _run_ga(seed, monkeypatch):
    space = SearchSpace({"DepthU": [32, 64, 128], "PrefetchGlobalRead": [1, 2], "SourceSwap": [0, 1]}, max_iters=4)
    initial_pop = _build_population(space, 6)
    monkeypatch.setattr(space, "sample", lambda *args, **kwargs: initial_pop.copy())

    selection = Selection.get("tournament", k=2, ratio=0.5, elitism=0.0, replacement=True)
    crossover = Crossover.get("ux", prob=0.9, mode="random")
    mutation = Mutation(space, prob=0.2)
    mating = Mating(space, selection, crossover, mutation, max_iters=8)
    # Keep flow deterministic but seed-sensitive via NumPy RNG.
    monkeypatch.setattr(mating, "__call__", lambda pop, n_offsprings=None: pop.shuffle())
    survival = Survival.get("fitness")

    def evaluate(individuals):
        vals = []
        for ind in individuals:
            vals.append(ind["DepthU"] * 0.01 + ind["PrefetchGlobalRead"] * 0.3 + ind["SourceSwap"] * 0.1)
        return np.array([vals], dtype=np.float32)

    ga = GeneticAlgorithm(
        space,
        mating,
        evaluate=evaluate,
        survival=survival,
        pop_size=6,
        n_gen=3,
        soo=False,
        period=0,
        seed=seed,
        verbose=0,
    )

    best, fitness = ga.optimize()
    return best, fitness, ga.stats


def test_ga_reproducible_with_same_seed(monkeypatch):
    best_a, fit_a, stats_a = _run_ga(seed=11, monkeypatch=monkeypatch)
    best_b, fit_b, stats_b = _run_ga(seed=11, monkeypatch=monkeypatch)

    assert best_a == best_b
    assert np.allclose(fit_a, fit_b)
    assert np.allclose(stats_a["f_avg"], stats_b["f_avg"])
    assert np.allclose(stats_a["f_max"], stats_b["f_max"])


def test_ga_seed_change_can_change_trajectory(monkeypatch):
    _best_a, _fit_a, stats_a = _run_ga(seed=13, monkeypatch=monkeypatch)
    _best_b, _fit_b, stats_b = _run_ga(seed=29, monkeypatch=monkeypatch)

    # Allow occasional collisions, but usually different RNG trajectories diverge.
    assert stats_a["f_avg"] != stats_b["f_avg"] or stats_a["f_max"] != stats_b["f_max"]
