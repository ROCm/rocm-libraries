# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Extended tests for GeneticAlgorithm — targeting uncovered code paths.

Covers: pop_size validation error paths, soo=True reduce_fn path, n_gen/period
validation, large-space and small-space pop_size decay at init, weights/weight_beta
processing, MaxIterationsReached fallback during initial sampling, optimize() with
resume from checkpoint (load/save), soo best selection at end, invalid type
logging paths.
"""

import pickle
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

from Tensile.ductile.algorithm import GeneticAlgorithm
from Tensile.ductile.core import SearchSpace, Selection, Crossover, Mutation, Mating, Survival
from Tensile.ductile.core.mating import MatingExhaustedError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_space(keys=None, max_iters=3):
    if keys is None:
        keys = {"DepthU": [32, 64, 128, 256], "SourceSwap": [0, 1]}
    return SearchSpace(keys, max_iters=max_iters if max_iters > 3 else 50)


def _make_mating(space, max_iters=3):
    selection = Selection.get("tournament", k=2, ratio=0.5, elitism=0.0, replacement=True)
    crossover = Crossover.get("ux", prob=0.9, mode="random")
    mutation = Mutation(space, prob=0.2)
    return Mating(space=space, selection=selection, crossover=crossover, mutation=mutation, max_iters=50)


def _make_ga(space=None, pop_size=8, n_gen=2, soo=False, period=0, tol=0.0,
             div_thr=0.5, seed=1, evaluate=None, verbose=0, weights=None,
             weight_beta=0.25, checkpoint_path=None):
    if space is None:
        space = _make_space()
    mating = _make_mating(space)
    survival = Survival.get("fitness")
    evaluate = evaluate or (lambda _x: np.ones((1, len(_x)), dtype=np.float32))
    return GeneticAlgorithm(
        space,
        mating,
        evaluate=evaluate,
        survival=survival,
        pop_size=pop_size,
        n_gen=n_gen,
        soo=soo,
        period=period,
        tol=tol,
        div_thr=div_thr,
        seed=seed,
        verbose=verbose,
        weights=weights,
        weight_beta=weight_beta,
        checkpoint_path=checkpoint_path,
    )


# ---------------------------------------------------------------------------
# Constructor validation — invalid type logging
# ---------------------------------------------------------------------------

class TestGAConstructorValidation:
    def test_invalid_pop_size_falls_back_to_default(self):
        # Use a large enough space so DEFAULTS["pop_size"]=512 fits
        large_space = SearchSpace({f"P{i}": list(range(10)) for i in range(4)}, max_iters=3)
        mating = _make_mating(large_space)
        ga = GeneticAlgorithm(
            large_space, mating,
            evaluate=lambda x: np.ones((1, len(x))),
            pop_size=2,  # invalid — must be > 2; fallback to DEFAULTS["pop_size"]
            n_gen=1, seed=1, verbose=0
        )
        assert ga.pop_size > 2

    def test_invalid_n_gen_falls_back_to_default(self):
        large_space = SearchSpace({f"P{i}": list(range(10)) for i in range(4)}, max_iters=3)
        mating = _make_mating(large_space)
        ga = GeneticAlgorithm(
            large_space, mating,
            evaluate=lambda x: np.ones((1, len(x))),
            pop_size=512,
            n_gen=0,  # invalid; fallback to DEFAULTS["n_gen"]
            seed=1, verbose=0
        )
        assert ga.n_gen > 0

    def test_invalid_soo_type_falls_back_to_default(self):
        large_space = SearchSpace({f"P{i}": list(range(10)) for i in range(4)}, max_iters=3)
        mating = _make_mating(large_space)
        ga = GeneticAlgorithm(
            large_space, mating,
            evaluate=lambda x: np.ones((1, len(x))),
            pop_size=512,
            n_gen=1,
            soo="yes",  # invalid — not bool; fallback to DEFAULTS["soo"]
            seed=1, verbose=0
        )
        assert isinstance(ga.soo, bool)

    def test_invalid_period_falls_back_to_default(self):
        large_space = SearchSpace({f"P{i}": list(range(10)) for i in range(4)}, max_iters=3)
        mating = _make_mating(large_space)
        ga = GeneticAlgorithm(
            large_space, mating,
            evaluate=lambda x: np.ones((1, len(x))),
            pop_size=512,
            n_gen=1,
            period=-1,  # invalid; fallback to DEFAULTS["period"]
            seed=1, verbose=0
        )
        assert ga.period >= 0

    def test_invalid_tol_falls_back_to_default(self):
        large_space = SearchSpace({f"P{i}": list(range(10)) for i in range(4)}, max_iters=3)
        mating = _make_mating(large_space)
        ga = GeneticAlgorithm(
            large_space, mating,
            evaluate=lambda x: np.ones((1, len(x))),
            pop_size=512,
            n_gen=1,
            tol=-1.0,  # invalid; fallback to DEFAULTS["tol"]
            seed=1, verbose=0
        )
        assert ga.tol >= 0

    def test_space_not_search_space_raises(self):
        mating = _make_mating(_make_space())
        with pytest.raises(ValueError, match="space must be of type SearchSpace"):
            GeneticAlgorithm(
                {"DepthU": [32, 64]}, mating,
                evaluate=lambda x: np.ones((1, len(x))),
                pop_size=8, n_gen=1, seed=1, verbose=0
            )

    def test_mating_not_mating_raises(self):
        space = _make_space()
        with pytest.raises(ValueError, match="mating must be of type Mating"):
            GeneticAlgorithm(
                space, object(),
                evaluate=lambda x: np.ones((1, len(x))),
                pop_size=8, n_gen=1, seed=1, verbose=0
            )

    def test_evaluate_not_callable_raises(self):
        space = _make_space()
        mating = _make_mating(space)
        with pytest.raises(ValueError, match="'evaluate' must be a callable"):
            GeneticAlgorithm(
                space, mating,
                evaluate="not-callable",
                pop_size=8, n_gen=1, seed=1, verbose=0
            )

    def test_pop_size_exceeds_n_perms_raises(self):
        # Space with only 2 permutations
        space = _make_space(keys={"A": [0, 1]})
        mating = _make_mating(space)
        with pytest.raises(ValueError, match="search space too small"):
            GeneticAlgorithm(
                space, mating,
                evaluate=lambda x: np.ones((1, len(x))),
                pop_size=100, n_gen=1, seed=1, verbose=0
            )

    def test_weights_not_list_of_dicts_raises(self):
        space = _make_space()
        mating = _make_mating(space)
        with pytest.raises(ValueError, match="weights must be an iterable"):
            GeneticAlgorithm(
                space, mating,
                evaluate=lambda x: np.ones((1, len(x))),
                pop_size=8, n_gen=1, seed=1, verbose=0,
                weights="bad"
            )


# ---------------------------------------------------------------------------
# Population size decay at init
# ---------------------------------------------------------------------------

class TestGAPopSizeDecay:
    def test_large_space_increases_pop_size(self):
        # max var size (256) > pop_size (8) → large_space decay
        space = _make_space(keys={"DepthU": list(range(256)), "SourceSwap": [0, 1]})
        mating = _make_mating(space)
        ga = GeneticAlgorithm(
            space, mating,
            evaluate=lambda x: np.ones((1, len(x))),
            pop_size=8,
            n_gen=1, seed=1, verbose=0
        )
        assert ga._decay_type == "large_space"
        assert ga.pop_size > 8

    def test_small_space_reduces_pop_size(self):
        # max var size (18) < pop_size (100)/5=20 → halved to 50
        # n_perms = 3*18*2 = 108 >= 50, so passes the n_perms check
        keys = {"A": list(range(3)), "B": list(range(18)), "C": list(range(2))}
        space = _make_space(keys=keys)
        mating = _make_mating(space)
        ga = GeneticAlgorithm(
            space, mating,
            evaluate=lambda x: np.ones((1, len(x))),
            pop_size=100,
            n_gen=1, seed=1, verbose=0
        )
        assert ga.pop_size < 100


# ---------------------------------------------------------------------------
# Weights processing
# ---------------------------------------------------------------------------

class TestGAWeights:
    def test_valid_weights_set_probs(self):
        space = _make_space()
        mating = _make_mating(space)
        # DepthU has 4 values → weights list of length 4
        weights = [{"DepthU": [1.0, 2.0, 3.0, 4.0]}]
        ga = GeneticAlgorithm(
            space, mating,
            evaluate=lambda x: np.ones((1, len(x))),
            pop_size=8, n_gen=1, seed=1, verbose=0,
            weights=weights, weight_beta=0.25
        )
        assert "DepthU" in ga.probs
        assert abs(ga.probs["DepthU"].sum() - 1.0) < 1e-5

    def test_unknown_param_in_weights_is_skipped(self):
        space = _make_space()
        mating = _make_mating(space)
        weights = [{"NonExistentParam": [1.0, 2.0]}]
        ga = GeneticAlgorithm(
            space, mating,
            evaluate=lambda x: np.ones((1, len(x))),
            pop_size=8, n_gen=1, seed=1, verbose=0,
            weights=weights
        )
        assert "NonExistentParam" not in ga.probs

    def test_wrong_weight_length_raises(self):
        space = _make_space()
        mating = _make_mating(space)
        # DepthU has 4 values, but weight list has 3
        weights = [{"DepthU": [1.0, 2.0, 3.0]}]
        with pytest.raises(ValueError):
            GeneticAlgorithm(
                space, mating,
                evaluate=lambda x: np.ones((1, len(x))),
                pop_size=8, n_gen=1, seed=1, verbose=0,
                weights=weights
            )


# ---------------------------------------------------------------------------
# optimize() — basic run, soo=True, MaxIterationsReached fallback
# ---------------------------------------------------------------------------

class TestGAOptimize:
    def test_basic_optimize_returns_x_and_f(self):
        ga = _make_ga(n_gen=2, pop_size=8)
        X, F = ga.optimize()
        assert isinstance(X, list)
        assert len(X) > 0
        assert F.size > 0

    def test_soo_true_returns_single_best(self):
        space = _make_space()
        mating = _make_mating(space)
        evaluate = lambda _x: np.ones((1, len(_x)), dtype=np.float32)
        ga = GeneticAlgorithm(
            space, mating,
            evaluate=evaluate,
            pop_size=8, n_gen=2, soo=True, seed=1, verbose=0
        )
        X, F = ga.optimize()
        assert len(X) == 1

    def test_max_iterations_reached_fallback(self, monkeypatch):
        """When initial sampling raises MaxIterationsReached, GA retries with half pop."""
        from Tensile.ductile.core.space import MaxIterationsReached

        call_count = [0]
        original_sample = SearchSpace.sample

        def patched_sample(self, size, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MaxIterationsReached("test", 2)
            # Return a small population for the retry
            return original_sample(self, min(size, 4), **kwargs)

        monkeypatch.setattr(SearchSpace, "sample", patched_sample)

        ga = _make_ga(n_gen=1, pop_size=8)
        X, F = ga.optimize()
        assert call_count[0] >= 2  # at least one retry happened

    def test_convergence_stops_early(self):
        ga = _make_ga(n_gen=20, period=2, tol=0.0)
        # With constant fitness, termination fires after period+1 steps
        X, F = ga.optimize()
        assert isinstance(X, list)

    def test_checkpoint_saved_each_generation(self, tmp_path):
        ckpt = tmp_path / "ga.checkpoint"
        ga = _make_ga(n_gen=2, pop_size=8, checkpoint_path=str(ckpt))
        ga.optimize()
        assert ckpt.exists()

    def test_resume_from_checkpoint(self, tmp_path):
        ckpt = tmp_path / "ga.checkpoint"
        ga = _make_ga(n_gen=2, pop_size=8, checkpoint_path=str(ckpt))
        ga.optimize()

        # Resume from checkpoint
        ga2 = _make_ga(n_gen=4, pop_size=8, checkpoint_path=str(ckpt))
        ga2.load(ckpt)
        X, F = ga2.optimize()
        assert isinstance(X, list)

    def test_mating_max_iters_failure_stops_gracefully(self, monkeypatch):
        ga = _make_ga(n_gen=3, pop_size=8)

        def _raise_stop(*_a, **_kw):
            raise MatingExhaustedError("max iters reached while generating offsprings")

        # Force mating failure after first generation update path.
        monkeypatch.setattr(
            ga,
            "mating",
            _raise_stop,
        )

        X, F = ga.optimize()
        assert isinstance(X, list)
        assert len(X) > 0
        assert F.size > 0


# ---------------------------------------------------------------------------
# save() and load()
# ---------------------------------------------------------------------------

class TestGASaveLoad:
    def test_load_from_path(self, tmp_path):
        ga = _make_ga(n_gen=1, pop_size=8)
        X, F = ga.optimize()

        ckpt = tmp_path / "ckpt.pkl"
        # Manually build a valid checkpoint dict
        from Tensile.ductile.core.population import Population, Individual
        import random as _random
        pop = Population([Individual({"DepthU": 0, "SourceSwap": 0}, F=1.0)])
        state = {
            "gen": 1,
            "soo": False,
            "space_map": ga.space.map,
            "stats": {},
            "best": pop.tolist(),
            "n_evals": 1,
            "old_pop": [],
            "pop": pop.tolist(),
            "pop_size": ga.pop_size,
            "_pop_size": ga._pop_size,
            "decay_type": "none",
            "random_state": _random.getstate(),
            "np_random_state": np.random.get_state(),
        }
        with open(ckpt, "wb") as f:
            pickle.dump(state, f)

        ga.load(ckpt)
        assert ga._resume_state is not None

    def test_load_non_dict_raises(self):
        ga = _make_ga(n_gen=1, pop_size=8)
        with pytest.raises(FileNotFoundError):
            ga.load("not-a-dict-or-path")

    def test_load_missing_fields_raises(self):
        ga = _make_ga(n_gen=1, pop_size=8)
        with pytest.raises(ValueError, match="missing required fields"):
            ga.load({"gen": 1})  # missing many required fields

    def test_load_soo_mismatch_raises(self):
        ga = _make_ga(n_gen=1, pop_size=8, soo=False)
        with pytest.raises(ValueError, match="soo mismatch"):
            ga.load({
                "gen": 1, "soo": True,  # mismatch
                "space_map": ga.space.map,
                "stats": {}, "best": None, "n_evals": 0,
                "old_pop": [], "pop": [], "pop_size": 8, "_pop_size": 8,
                "decay_type": "none", "random_state": None, "np_random_state": None,
            })

    def test_load_space_map_mismatch_raises(self):
        ga = _make_ga(n_gen=1, pop_size=8)
        with pytest.raises(ValueError, match="space.map mismatch"):
            ga.load({
                "gen": 1, "soo": False,
                "space_map": {"OtherParam": [0, 1]},  # mismatch
                "stats": {}, "best": None, "n_evals": 0,
                "old_pop": [], "pop": [], "pop_size": 8, "_pop_size": 8,
                "decay_type": "none", "random_state": None, "np_random_state": None,
            })

    def test_load_unknown_decay_type_raises(self):
        ga = _make_ga(n_gen=1, pop_size=8)
        with pytest.raises(ValueError, match="unknown decay_type"):
            ga.load({
                "gen": 1, "soo": False,
                "space_map": ga.space.map,
                "stats": {}, "best": None, "n_evals": 0,
                "old_pop": [], "pop": [], "pop_size": 8, "_pop_size": 8,
                "decay_type": "weird_decay",  # unknown
                "random_state": None, "np_random_state": None,
            })

    def test_load_large_space_decay_type_sets_lambda(self):
        import random as _random
        ga = _make_ga(n_gen=1, pop_size=8)
        ga.load({
            "gen": 1, "soo": False,
            "space_map": ga.space.map,
            "stats": {}, "best": None, "n_evals": 0,
            "old_pop": [], "pop": [], "pop_size": 8, "_pop_size": 8,
            "decay_type": "large_space",
            "random_state": _random.getstate(),
            "np_random_state": np.random.get_state(),
        })
        assert ga._decay_type == "large_space"
        assert hasattr(ga, "decay")

    def test_load_low_diversity_decay_type_sets_lambda(self):
        import random as _random
        ga = _make_ga(n_gen=1, pop_size=8)
        ga.load({
            "gen": 1, "soo": False,
            "space_map": ga.space.map,
            "stats": {}, "best": None, "n_evals": 0,
            "old_pop": [], "pop": [], "pop_size": 8, "_pop_size": 8,
            "decay_type": "low_diversity",
            "random_state": _random.getstate(),
            "np_random_state": np.random.get_state(),
        })
        assert ga._decay_type == "low_diversity"
        assert hasattr(ga, "decay")


# ---------------------------------------------------------------------------
# update() method
# ---------------------------------------------------------------------------

class TestGAUpdate:
    def test_update_with_none_best_initializes_best(self):
        ga = _make_ga(n_gen=1, pop_size=8)
        space = ga.space
        from Tensile.ductile.core.population import Population, Individual
        pop = Population([
            Individual({"DepthU": 0, "SourceSwap": 0}, F=0.0),
            Individual({"DepthU": 1, "SourceSwap": 1}, F=0.0),
        ])
        old_pop = Population()
        scores = np.array([[1.0, 2.0]], dtype=np.float32)

        best, f_max = ga.update(None, pop, old_pop, scores)
        assert best is not None
        assert f_max > 0

    def test_update_replaces_best_on_improvement(self):
        ga = _make_ga(n_gen=1, pop_size=8)
        from Tensile.ductile.core.population import Population, Individual
        pop = Population([
            Individual({"DepthU": 0, "SourceSwap": 0}, F=1.0),
            Individual({"DepthU": 1, "SourceSwap": 1}, F=2.0),
        ])
        old_pop = Population()
        scores1 = np.array([[1.0, 2.0]], dtype=np.float32)
        best, _ = ga.update(None, pop, old_pop, scores1)

        scores2 = np.array([[3.0, 2.0]], dtype=np.float32)
        best2, _ = ga.update(best, pop, old_pop, scores2)
        assert best2.F.max() >= best.F.max()

    def test_negative_scores_clipped_to_minus_one(self):
        ga = _make_ga(n_gen=1, pop_size=8)
        from Tensile.ductile.core.population import Population, Individual
        pop = Population([Individual({"DepthU": 0, "SourceSwap": 0})])
        scores = np.array([[-5.0]], dtype=np.float32)
        best, _ = ga.update(None, pop, Population(), scores)
        # negative clamped to -1
        assert pop.G[0][0] == -1.0


# ---------------------------------------------------------------------------
# termination() method
# ---------------------------------------------------------------------------

class TestGATermination:
    def test_period_zero_never_fires(self):
        ga = _make_ga(period=0)
        for _ in range(100):
            ga.termination(f_avg=1.0, f_max=1.0, diversity=1.0)
        # No StopIteration raised

    def test_period_nonzero_fires_on_plateau(self):
        ga = _make_ga(period=2, tol=0.0)
        ga.termination(f_avg=1.0, f_max=1.0, diversity=1.0)
        ga.termination(f_avg=1.0, f_max=1.0, diversity=1.0)
        with pytest.raises(StopIteration):
            ga.termination(f_avg=1.0, f_max=1.0, diversity=1.0)

    def test_diversity_below_threshold_triggers_low_diversity_decay(self):
        ga = _make_ga(period=0, div_thr=0.5)
        ga.termination(f_avg=1.0, f_max=1.0, diversity=0.1)
        assert ga._decay_type == "low_diversity"

    def test_repr_contains_expected_fields(self):
        ga = _make_ga()
        r = repr(ga)
        assert "GeneticAlgorithm" in r
        assert "pop_size" in r
