# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.ductile.algorithm import GeneticAlgorithm
from Tensile.ductile.core import SearchSpace, Selection, Crossover, Mutation, Mating, Survival

pytestmark = pytest.mark.unit


def _build_ga(period=2, tol=0.0, div_thr=0.5):
    space = SearchSpace({"DepthU": [32, 64, 128], "SourceSwap": [0, 1]}, max_iters=2)
    selection = Selection.get("tournament", k=2, ratio=0.5, elitism=0.0, replacement=True)
    crossover = Crossover.get("ux", prob=0.9, mode="random")
    mutation = Mutation(space, prob=0.2)
    mating = Mating(space, selection, crossover, mutation, max_iters=4)
    survival = Survival.get("fitness")

    return GeneticAlgorithm(
        space,
        mating,
        evaluate=lambda _x: [1.0],
        survival=survival,
        pop_size=6,
        n_gen=3,
        soo=False,
        period=period,
        tol=tol,
        div_thr=div_thr,
        seed=1,
        verbose=0,
    )


def test_termination_raises_stop_iteration_on_plateau():
    ga = _build_ga(period=2, tol=0.0)
    ga.termination(f_avg=1.0, f_max=1.5, diversity=1.0)
    ga.termination(f_avg=1.0, f_max=1.5, diversity=1.0)

    with pytest.raises(StopIteration, match="did not increase"):
        ga.termination(f_avg=1.0, f_max=1.5, diversity=1.0)


def test_termination_sets_low_diversity_decay_and_updates_pop_size():
    ga = _build_ga(period=0, tol=0.0, div_thr=0.6)
    original = ga.pop_size

    ga.termination(f_avg=1.0, f_max=1.2, diversity=0.1)

    assert ga._decay_type == "low_diversity"
    assert ga.pop_size < original
