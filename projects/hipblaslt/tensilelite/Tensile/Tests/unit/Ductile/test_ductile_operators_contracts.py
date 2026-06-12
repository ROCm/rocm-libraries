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

from Tensile.ductile.core import SearchSpace, Selection, Crossover, Mutation
from Tensile.ductile.core.population import Individual, Population

pytestmark = pytest.mark.unit


def _space():
    return SearchSpace({"DepthU": [32, 64, 128], "SourceSwap": [0, 1]}, max_iters=2)


def _population():
    pop = Population(
        [
            Individual({"DepthU": 0, "SourceSwap": 0}, F=1.0),
            Individual({"DepthU": 1, "SourceSwap": 0}, F=2.0),
            Individual({"DepthU": 2, "SourceSwap": 1}, F=3.0),
            Individual({"DepthU": 1, "SourceSwap": 1}, F=4.0),
        ]
    )
    return pop


def test_selection_get_rejects_unknown_name():
    with pytest.raises(ValueError, match="selection must be"):
        Selection.get("not-a-selection")


def test_crossover_get_rejects_unknown_name():
    with pytest.raises(ValueError, match="crossover must be"):
        Crossover.get("not-a-crossover")


def test_crossover_rejects_unknown_pairing_mode():
    with pytest.raises(ValueError, match="unknown pairing mode"):
        Crossover.get("ux", mode="unknown")


def test_mutation_rejects_invalid_probability_and_weight_type():
    space = _space()
    with pytest.raises(ValueError, match="probabilities must be a float"):
        Mutation(space, prob=2.0)
    with pytest.raises(ValueError, match="weights must be a dictionary"):
        Mutation(space, prob=0.2, weights=[1, 2])


def test_selection_and_crossover_emit_expected_sizes_and_schema():
    pop = _population()

    selection = Selection.get("tournament", k=2, ratio=0.5, elitism=0.0, replacement=True)
    parents = selection(pop)
    assert parents.size == 2

    crossover = Crossover.get("ux", prob=1.0, mode="random")
    offspring_pairs = list(crossover(parents, n_off=2))
    assert len(offspring_pairs) >= 1

    a, b = offspring_pairs[0]
    assert set(a.names) == set(pop.names)
    assert set(b.names) == set(pop.names)


def test_mutation_never_introduces_out_of_space_values():
    space = _space()
    mutation = Mutation(space, prob=1.0)
    mutated = mutation(Individual({"DepthU": 1, "SourceSwap": 0}))

    assert mutated["DepthU"] in space["DepthU"]
    assert mutated["SourceSwap"] in space["SourceSwap"]
