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

from Tensile.ductile.core.population import Individual, Population, IndividualSet, ExceedsCapacity

pytestmark = pytest.mark.unit


def test_individual_diff_and_assignment_reset_fitness():
    inda = Individual({"DepthU": 0, "SourceSwap": 0}, F=3.0)
    indb = Individual({"DepthU": 1, "SourceSwap": 0}, F=2.0)

    assert inda.diff(indb) == ["DepthU"]
    inda["DepthU"] = 1
    assert inda.F == 0.0


def test_population_unique_and_merge_for_compatible_individuals():
    pop = Population(
        [
            Individual({"DepthU": 0, "SourceSwap": 0}),
            Individual({"DepthU": 0, "SourceSwap": 0}),
            Individual({"DepthU": 1, "SourceSwap": 1}),
        ]
    )
    uniq = pop.unique()

    assert uniq.size == 2

    merged = uniq.merge(Population([Individual({"DepthU": 2, "SourceSwap": 1})]))
    assert merged.size == 3


def test_population_merge_rejects_mismatched_variable_sets():
    pop_a = Population([Individual({"DepthU": 0, "SourceSwap": 0})])
    pop_b = Population([Individual({"DepthU": 0, "PrefetchGlobalRead": 1})])

    with pytest.raises(ValueError, match="same variables"):
        pop_a.merge(pop_b)


def test_individual_set_capacity_limit():
    s = IndividualSet(capacity=1)
    s.add(Individual({"DepthU": 0, "SourceSwap": 0}))

    with pytest.raises(ExceedsCapacity):
        s.add(Individual({"DepthU": 1, "SourceSwap": 1}))
