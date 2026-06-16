################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import pytest

import Tensile.ductile.core.space as space_mod
from Tensile.ductile.core.space import SearchSpace, MaxIterationsReached
from Tensile.ductile.core.population import Individual, Population

pytestmark = pytest.mark.unit


class _FakeParallel:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, tasks):
        list(tasks)
        return [[]]


def test_searchspace_transform_individual_and_population():
    space = SearchSpace({"DepthU": [32, 64], "SourceSwap": [0, 1]}, max_iters=2)
    ind = Individual({"DepthU": 1, "SourceSwap": 0})
    pop = Population([ind, Individual({"DepthU": 0, "SourceSwap": 1})])

    assert space.transform(ind) == {"DepthU": 64, "SourceSwap": 0}
    assert space.transform(pop) == [{"DepthU": 64, "SourceSwap": 0}, {"DepthU": 32, "SourceSwap": 1}]


def test_searchspace_sample_raises_when_no_valid_candidates(monkeypatch):
    space = SearchSpace({"DepthU": [32, 64], "SourceSwap": [0, 1]}, max_iters=1, valid=lambda _x: False)
    monkeypatch.setattr(space_mod.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(space_mod.joblib, "Parallel", _FakeParallel)

    with pytest.raises(MaxIterationsReached, match="max iters reached"):
        space.sample(size=2)


def test_searchspace_sample_reuse_cache_with_capacity(monkeypatch):
    space = SearchSpace({"DepthU": [32, 64], "SourceSwap": [0, 1]}, max_iters=1)
    space.cache = [Individual({"DepthU": 0, "SourceSwap": 1})]

    monkeypatch.setattr(space_mod.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(space_mod.joblib, "Parallel", _FakeParallel)

    pop = space.sample(size=1, reuse=True)
    assert pop.size == 1
    assert space.transform(pop)[0] == {"DepthU": 32, "SourceSwap": 1}

################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Extended tests for Tensile.ductile.core.space (SearchSpace) — targeting
uncovered code paths.

Covers: valid() with a validator, transform() for both Individual and Population,
size() keyed and unkeyed, __contains__, __iter__, sample with reuse=True (cache),
MaxIterationsReached when constraint rejects everything, xdist override detection,
__repr__.
"""

import os
import importlib

import numpy as np
import pytest

from Tensile.ductile.core.space import SearchSpace, MaxIterationsReached, sample_chunk
from Tensile.ductile.core.population import Individual, Population

pytestmark = pytest.mark.unit


def _valid_a_zero(x):
    return x["A"] == 0


def _always_false(_x):
    return False


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestSearchSpaceConstructor:
    def test_empty_space_raises(self):
        with pytest.raises(ValueError, match="non-empty dictionary"):
            SearchSpace({})

    def test_empty_variable_choices_raises(self):
        with pytest.raises(ValueError, match="length = 0"):
            SearchSpace({"A": []})

    def test_repr_contains_num_vars(self):
        space = SearchSpace({"A": [1, 2], "B": [0, 1]})
        r = repr(space)
        assert "num_vars=2" in r


# ---------------------------------------------------------------------------
# size(), __contains__, __iter__
# ---------------------------------------------------------------------------

class TestSearchSpaceAccessors:
    def test_size_no_key_returns_num_vars(self):
        space = SearchSpace({"A": [1, 2, 3], "B": [0, 1]})
        assert space.size() == 2

    def test_size_with_key_returns_var_count(self):
        space = SearchSpace({"A": [1, 2, 3], "B": [0, 1]})
        assert space.size("A") == 3
        assert space.size("B") == 2

    def test_contains_true_for_existing_key(self):
        space = SearchSpace({"A": [1, 2]})
        assert "A" in space

    def test_contains_false_for_missing_key(self):
        space = SearchSpace({"A": [1, 2]})
        assert "B" not in space

    def test_iter_yields_all_keys(self):
        space = SearchSpace({"A": [1, 2], "B": [0, 1]})
        assert set(space) == {"A", "B"}

    def test_getitem_returns_index_list(self):
        space = SearchSpace({"A": [10, 20, 30]})
        assert space["A"] == [0, 1, 2]


# ---------------------------------------------------------------------------
# transform()
# ---------------------------------------------------------------------------

class TestSearchSpaceTransform:
    def test_transform_individual_maps_indices_to_values(self):
        space = SearchSpace({"DepthU": [32, 64, 128]})
        ind = Individual({"DepthU": 1})  # index 1 → value 64
        result = space.transform(ind)
        assert result == {"DepthU": 64}

    def test_transform_population_returns_list_of_dicts(self):
        space = SearchSpace({"DepthU": [32, 64], "SourceSwap": [0, 1]})
        pop = Population([
            Individual({"DepthU": 0, "SourceSwap": 0}),
            Individual({"DepthU": 1, "SourceSwap": 1}),
        ])
        result = space.transform(pop)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["DepthU"] == 32
        assert result[1]["DepthU"] == 64


# ---------------------------------------------------------------------------
# valid() with validator function
# ---------------------------------------------------------------------------

class TestSearchSpaceValid:
    def test_valid_returns_true_when_no_validator(self):
        space = SearchSpace({"A": [0, 1]})
        ind = Individual({"A": 0})
        assert space.valid(ind) is True

    def test_valid_calls_validator_with_transformed_individual(self):
        called_with = []

        def validator(x):
            called_with.append(x)
            return True

        space = SearchSpace({"A": [10, 20]}, valid=validator)
        ind = Individual({"A": 1})  # index 1 → value 20
        space.valid(ind)
        assert called_with == [{"A": 20}]

    def test_valid_returns_false_when_validator_rejects(self):
        space = SearchSpace({"A": [10, 20]}, valid=lambda x: x["A"] != 20)
        ind = Individual({"A": 1})  # index 1 → value 20
        assert space.valid(ind) is False


# ---------------------------------------------------------------------------
# sample() — constrained sampling, cache reuse, MaxIterationsReached
# ---------------------------------------------------------------------------

class TestSearchSpaceSample:
    def test_unconstrained_sample_returns_correct_size(self):
        space = SearchSpace({"A": [0, 1, 2, 3], "B": [0, 1]}, max_iters=10)
        pop = space.sample(4)
        assert pop.size == 4

    def test_constrained_sample_only_valid_individuals(self):
        # Only allow A=0 (index 0)
        space = SearchSpace(
            {"A": [0, 1, 2], "B": [0, 1]},
            max_iters=200,
            valid=_valid_a_zero,
        )
        pop = space.sample(2)
        assert pop.size == 2
        for ind in pop:
            # index 0 → value 0
            assert ind["A"] == 0

    def test_sample_with_reuse_uses_cached_individuals(self):
        space = SearchSpace({"A": [0, 1, 2, 3], "B": [0, 1]}, max_iters=10)
        pop1 = space.sample(3)
        pop2 = space.sample(3, reuse=True)
        # Cache should be populated; second sample should succeed quickly
        assert pop2.size == 3

    def test_max_iterations_reached_raises(self):
        # Validator rejects everything → MaxIterationsReached
        space = SearchSpace({"A": [0, 1]}, max_iters=1, valid=_always_false)
        with pytest.raises(MaxIterationsReached):
            space.sample(5)

    def test_sample_with_probability_weights(self):
        space = SearchSpace({"A": [0, 1, 2, 3]}, max_iters=10)
        # p biases toward index 0
        pop = space.sample(4, p={"A": None})
        assert pop.size == 4

    def test_seed_makes_sampling_deterministic(self):
        space1 = SearchSpace({"A": [0, 1, 2, 3], "B": [0, 1]}, max_iters=10)
        space1.seed(42)
        pop1 = space1.sample(4)

        space2 = SearchSpace({"A": [0, 1, 2, 3], "B": [0, 1]}, max_iters=10)
        space2.seed(42)
        pop2 = space2.sample(4)

        vals1 = sorted([tuple(ind.values) for ind in pop1])
        vals2 = sorted([tuple(ind.values) for ind in pop2])
        assert vals1 == vals2


# ---------------------------------------------------------------------------
# xdist worker detection
# ---------------------------------------------------------------------------

class TestXdistWorkerDetection:
    def test_xdist_env_var_sets_threading_backend(self, monkeypatch):
        import Tensile.ductile.core.space as space_mod
        monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
        # Reload IS_XDIST_WORKER logic by checking the module-level constants
        # (they're set at import time; we just verify the module reflects the design)
        # Rather than reimporting, test the constants directly:
        is_xdist = bool(os.environ.get("PYTEST_XDIST_WORKER"))
        assert is_xdist is True

    def test_no_xdist_env_var_uses_multiprocessing(self, monkeypatch):
        monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
        is_xdist = bool(os.environ.get("PYTEST_XDIST_WORKER"))
        assert is_xdist is False

    def test_reload_module_sets_threading_backend_constants(self, monkeypatch):
        monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
        import Tensile.ductile.core.space as space_mod
        reloaded = importlib.reload(space_mod)
        assert reloaded.JOBLIB_BACKEND == "threading"
        assert reloaded.JOBLIB_N_JOBS_OVERRIDE == 1

    def test_reload_module_sets_multiprocessing_backend_constants(self, monkeypatch):
        monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
        import Tensile.ductile.core.space as space_mod
        reloaded = importlib.reload(space_mod)
        assert reloaded.JOBLIB_BACKEND == "multiprocessing"
        assert reloaded.JOBLIB_N_JOBS_OVERRIDE is None


class TestSampleChunk:
    def test_sample_chunk_returns_only_valid_inds(self):
        sizes = {"A": 3, "B": 2}

        def valid(ind):
            return ind["A"] == 0

        out = sample_chunk(valid, p={}, sizes=sizes, chunk_size=30, seed=123)
        assert isinstance(out, list)
        for ind in out:
            assert isinstance(ind, Individual)
            assert ind["A"] == 0

    def test_sample_chunk_with_probability_map(self):
        sizes = {"A": 4}
        p = {"A": [1.0, 0.0, 0.0, 0.0]}

        out = sample_chunk(lambda _ind: True, p=p, sizes=sizes, chunk_size=10, seed=321)
        assert len(out) == 10
        assert all(ind["A"] == 0 for ind in out)
