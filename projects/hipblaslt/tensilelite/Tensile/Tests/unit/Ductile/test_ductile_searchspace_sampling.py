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
