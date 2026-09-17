# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from itertools import product

import numpy as np
import pytest

from Tensile.ductile.algorithm import GeneticAlgorithm
from Tensile.ductile.core import SearchSpace, Selection, Crossover, Mutation, Mating, Survival
from Tensile.ductile.core.population import Individual, Population

pytestmark = pytest.mark.unit


def _build_space():
    return SearchSpace(
        {
            "DepthU": [32, 64, 128, 256, 512],
            "PrefetchGlobalRead": [1, 2],
            "PrefetchLocalRead": [1],
            "SourceSwap": [0, 1],
            "1LDSBuffer": [0, 1],
            "WorkGroupMapping": [1, 6, 8],
        },
        max_iters=4,
    )


def _build_population(space, size):
    keys = list(space.sizes.keys())
    indices = [range(space.sizes[k]) for k in keys]
    all_inds = []
    for idx in product(*indices):
        all_inds.append(Individual(dict(zip(keys, idx))))
        if len(all_inds) == size:
            break
    return Population(all_inds)


def test_search_space_uses_parameters():
    space = _build_space()
    assert min(space.map["DepthU"]) == 32 and max(space.map["DepthU"]) == 512
    assert space.map["PrefetchGlobalRead"] == [1, 2]
    assert space.map["PrefetchLocalRead"] == [1]
    assert space.map["SourceSwap"] == [0, 1]
    assert space.map["1LDSBuffer"] == [0, 1]
    assert space.map["WorkGroupMapping"] == [1, 6, 8]


def test_ga_noisy_multiobjective_shows_early_favg_gain(monkeypatch):
    space = _build_space()

    pop_size = 8
    initial_pop = _build_population(space, pop_size)
    monkeypatch.setattr(space, "sample", lambda *args, **kwargs: initial_pop.copy())

    selection = Selection.get("tournament", k=2, ratio=0.5, elitism=0.0, replacement=True)
    crossover = Crossover.get("ux", prob=0.9, mode="random")
    mutation = Mutation(space, prob=0.25)
    mating = Mating(space, selection, crossover, mutation, max_iters=8)
    monkeypatch.setattr(mating, "__call__", lambda pop, n_offsprings=None: pop.copy())
    survival = Survival.get("fitness")

    generation = {"idx": 0}
    noise = np.array(
        [
            [0.10, -0.08, 0.05, -0.07, 0.03, -0.02, 0.04, -0.03],
            [-0.06, 0.09, -0.05, 0.08, -0.02, 0.03, -0.04, 0.02],
            [0.07, -0.04, 0.06, -0.01, 0.02, -0.03, 0.05, -0.06],
        ],
        dtype=np.float32,
    )

    def evaluate(individuals):
        g = generation["idx"]
        generation["idx"] += 1

        # Trend chosen to emulate realistic behavior: early f_avg increase,
        # while the best score can dip in a later generation.
        trend = [3.0, 5.5, 4.9][min(g, 2)]
        scores = np.zeros((3, len(individuals)), dtype=np.float32)
        for j, ind in enumerate(individuals):
            base = (
                ind["DepthU"] * 0.008
                + ind["PrefetchGlobalRead"] * 0.20
                + ind["PrefetchLocalRead"] * 0.05
                + ind["SourceSwap"] * 0.07
                + ind["1LDSBuffer"] * 0.09
                + ind["WorkGroupMapping"] * 0.015
            )
            scores[:, j] = base + trend + noise[:, j]

        if g == 2:
            scores[:, 0] -= 1.0

        return scores

    ga = GeneticAlgorithm(
        space,
        mating,
        evaluate=evaluate,
        survival=survival,
        pop_size=pop_size,
        n_gen=3,
        soo=False,
        period=0,
        seed=7,
        verbose=0,
    )

    best, fitness = ga.optimize()

    assert len(best) >= 1
    assert fitness.shape[0] >= 1
    assert len(ga.stats["f_avg"]) == 3
    assert len(ga.stats["f_max"]) == 3
    assert ga.stats["f_avg"][1] > ga.stats["f_avg"][0]
    assert ga.stats["f_max"][2] <= ga.stats["f_max"][1]
