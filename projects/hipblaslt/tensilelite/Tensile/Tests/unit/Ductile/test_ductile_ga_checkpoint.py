# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from itertools import product
import random

import numpy as np
import pytest

from Tensile.ductile.algorithm import GeneticAlgorithm
from Tensile.ductile.core import SearchSpace, Selection, Crossover, Mutation, Mating, Survival
from Tensile.ductile.core.population import Individual, Population

pytestmark = pytest.mark.unit


def _build_space():
    return SearchSpace(
        {
            "DepthU": [32, 64, 128],
            "PrefetchGlobalRead": [1, 2],
            "SourceSwap": [0, 1],
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


def _build_ga(space, checkpoint_path, n_gen):
    selection = Selection.get("tournament", k=2, ratio=0.5, elitism=0.0, replacement=True)
    crossover = Crossover.get("ux", prob=0.9, mode="random")
    mutation = Mutation(space, prob=0.20)
    mating = Mating(space, selection, crossover, mutation, max_iters=8)
    survival = Survival.get("fitness")

    def evaluate(individuals):
        scores = []
        for ind in individuals:
            score = ind["DepthU"] * 0.01 + ind["PrefetchGlobalRead"] * 0.3 + ind["SourceSwap"] * 0.1
            scores.append(score)
        return np.array([scores], dtype=np.float32)

    ga = GeneticAlgorithm(
        space,
        mating,
        evaluate=evaluate,
        survival=survival,
        pop_size=6,
        n_gen=n_gen,
        soo=False,
        period=0,
        seed=3,
        verbose=0,
        checkpoint_path=checkpoint_path,
    )
    return ga, mating


def test_ga_checkpoint_resume_roundtrip(monkeypatch, tmp_path):
    checkpoint_path = str(tmp_path / "ductile_ga.checkpoint")
    space = _build_space()
    initial_pop = _build_population(space, 6)
    monkeypatch.setattr(space, "sample", lambda *args, **kwargs: initial_pop.copy())

    ga, mating = _build_ga(space, checkpoint_path=checkpoint_path, n_gen=2)
    monkeypatch.setattr(mating, "__call__", lambda pop, n_offsprings=None: pop.copy())

    best_first, _ = ga.optimize()
    assert len(best_first) >= 1
    assert len(ga.stats["f_max"]) == 2

    ga_resume, mating_resume = _build_ga(space, checkpoint_path=checkpoint_path, n_gen=3)
    monkeypatch.setattr(mating_resume, "__call__", lambda pop, n_offsprings=None: pop.copy())

    ga_resume.load(checkpoint_path)
    best_resumed, _ = ga_resume.optimize()

    assert len(best_resumed) >= 1
    assert len(ga_resume.stats["f_max"]) == 3


def test_ga_load_rejects_mismatched_space_map(tmp_path):
    space = _build_space()
    ga, _ = _build_ga(space, checkpoint_path=str(tmp_path / "unused.checkpoint"), n_gen=2)

    bad_checkpoint = {
        "gen": 1,
        "soo": False,
        "space_map": {
            "DepthU": [32, 64],
            "PrefetchGlobalRead": [1, 2],
            "SourceSwap": [0, 1],
        },
        "stats": {},
        "best": None,
        "n_evals": 0,
        "old_pop": [],
        "pop": [],
        "pop_size": 6,
        "_pop_size": 6,
        "decay_type": "none",
        "random_state": random.getstate(),
        "np_random_state": np.random.get_state(),
    }

    with pytest.raises(ValueError, match="space.map mismatch"):
        ga.load(bad_checkpoint)


def test_ga_load_rejects_missing_required_fields(tmp_path):
    space = _build_space()
    ga, _ = _build_ga(space, checkpoint_path=str(tmp_path / "unused.checkpoint"), n_gen=2)

    incomplete_checkpoint = {
        "gen": 1,
        "soo": False,
        "space_map": space.map,
        # Intentionally missing several required keys.
    }

    with pytest.raises(ValueError, match="missing required fields"):
        ga.load(incomplete_checkpoint)


def test_ga_load_rejects_soo_mismatch(tmp_path):
    space = _build_space()
    ga, _ = _build_ga(space, checkpoint_path=str(tmp_path / "unused.checkpoint"), n_gen=2)

    bad_checkpoint = {
        "gen": 1,
        "soo": True,
        "space_map": space.map,
        "stats": {},
        "best": None,
        "n_evals": 0,
        "old_pop": [],
        "pop": [],
        "pop_size": 6,
        "_pop_size": 6,
        "decay_type": "none",
        "random_state": random.getstate(),
        "np_random_state": np.random.get_state(),
    }

    with pytest.raises(ValueError, match="soo mismatch"):
        ga.load(bad_checkpoint)


def test_ga_load_rejects_unknown_decay_type(tmp_path):
    space = _build_space()
    ga, _ = _build_ga(space, checkpoint_path=str(tmp_path / "unused.checkpoint"), n_gen=2)

    bad_checkpoint = {
        "gen": 1,
        "soo": False,
        "space_map": space.map,
        "stats": {},
        "best": None,
        "n_evals": 0,
        "old_pop": [],
        "pop": [],
        "pop_size": 6,
        "_pop_size": 6,
        "decay_type": "mystery",
        "random_state": random.getstate(),
        "np_random_state": np.random.get_state(),
    }

    with pytest.raises(ValueError, match="unknown decay_type"):
        ga.load(bad_checkpoint)
