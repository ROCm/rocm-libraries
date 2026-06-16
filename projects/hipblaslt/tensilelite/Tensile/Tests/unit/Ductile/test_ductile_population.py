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

"""Extended tests for Tensile.ductile.core.population — targeting uncovered paths.

Covers: Individual arithmetic ops (+,-,*,/), __radd__/__rsub__/__rmul__, __hash__,
comparison (__eq__, __lt__, __le__, __gt__, __ge__), __copy__, IndividualSet
init from IndividualSet, IndividualSet capacity with iterable, Population.get(),
diversity(reduce=False), F.setter, G.getter/G.setter, argsort, shuffle,
Population.copy(), nunique(key=None and key=k), Population with set input,
Individual.copy() independence, individual __repr__, Population.__str__.
"""

import copy

import numpy as np
import pytest

from Tensile.ductile.core.population import (
    ExceedsCapacity,
    Individual,
    IndividualSet,
    Population,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Individual — constructor validation
# ---------------------------------------------------------------------------

class TestIndividualConstructor:
    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="non-empty dictionary"):
            Individual({})

    def test_non_numeric_values_raises(self):
        with pytest.raises(ValueError, match="basic types"):
            Individual({"A": "string"})

    def test_repr_contains_x_and_f(self):
        ind = Individual({"A": 1}, F=2.5)
        r = repr(ind)
        assert "Individual" in r
        assert "2.5" in r


# ---------------------------------------------------------------------------
# Individual — arithmetic operators
# ---------------------------------------------------------------------------

class TestIndividualArithmetic:
    def _ind(self, f=2.0):
        return Individual({"A": 0, "B": 1}, F=f)

    def test_add_two_individuals(self):
        a = self._ind(2.0)
        b = self._ind(3.0)
        assert a + b == 5.0

    def test_add_scalar(self):
        a = self._ind(2.0)
        assert a + 1.0 == 3.0

    def test_radd_scalar(self):
        a = self._ind(2.0)
        assert 1.0 + a == 3.0

    def test_sub_two_individuals(self):
        a = self._ind(5.0)
        b = self._ind(3.0)
        assert a - b == 2.0

    def test_sub_scalar(self):
        a = self._ind(5.0)
        assert a - 1.0 == 4.0

    def test_rsub_scalar(self):
        a = self._ind(5.0)
        # __rsub__ delegates to __sub__, so semantics are (a.F - scalar)
        assert 10.0 - a == -5.0

    def test_mul_two_individuals(self):
        a = self._ind(2.0)
        b = self._ind(3.0)
        assert a * b == 6.0

    def test_mul_scalar(self):
        a = self._ind(2.0)
        assert a * 3.0 == 6.0

    def test_rmul_scalar(self):
        a = self._ind(2.0)
        assert 3.0 * a == 6.0

    def test_truediv_two_individuals(self):
        a = self._ind(6.0)
        b = self._ind(3.0)
        assert a / b == 2.0

    def test_truediv_scalar(self):
        a = self._ind(6.0)
        assert a / 2.0 == 3.0


# ---------------------------------------------------------------------------
# Individual — comparison operators (via @total_ordering)
# ---------------------------------------------------------------------------

class TestIndividualComparisons:
    def test_lt(self):
        a = Individual({"A": 0}, F=1.0)
        b = Individual({"A": 1}, F=2.0)
        assert a < b

    def test_gt(self):
        a = Individual({"A": 0}, F=3.0)
        b = Individual({"A": 1}, F=1.0)
        assert a > b

    def test_le_equal(self):
        a = Individual({"A": 0}, F=2.0)
        b = Individual({"A": 1}, F=2.0)
        # __eq__ compares X, so same F but different X is not <= under total_ordering
        assert not (a <= b)

    def test_ge_greater(self):
        a = Individual({"A": 0}, F=3.0)
        b = Individual({"A": 1}, F=2.0)
        assert a >= b

    def test_eq_different_x(self):
        a = Individual({"A": 0})
        b = Individual({"A": 1})
        assert a != b

    def test_eq_same_x(self):
        a = Individual({"A": 0})
        b = Individual({"A": 0})
        assert a == b

    def test_lt_with_scalar(self):
        a = Individual({"A": 0}, F=1.0)
        assert a < 2.0

    def test_eq_non_individual(self):
        a = Individual({"A": 0})
        assert a != "not-an-individual"


# ---------------------------------------------------------------------------
# Individual — hash, copy
# ---------------------------------------------------------------------------

class TestIndividualHashAndCopy:
    def test_hash_is_stable(self):
        a = Individual({"A": 0, "B": 1})
        h1 = hash(a)
        h2 = hash(a)
        assert h1 == h2

    def test_hash_differs_for_different_values(self):
        a = Individual({"A": 0})
        b = Individual({"A": 1})
        assert hash(a) != hash(b)

    def test_copy_is_independent(self):
        a = Individual({"A": 0}, F=1.0)
        b = a.copy()
        b["A"] = 1
        assert a["A"] == 0

    def test_dunder_copy_creates_independent_copy(self):
        a = Individual({"A": 0, "B": 1}, F=2.0)
        b = copy.copy(a)
        b["A"] = 99
        assert a["A"] == 0

    def test_usable_as_dict_key(self):
        a = Individual({"A": 0})
        d = {a: "value"}
        assert d[a] == "value"


# ---------------------------------------------------------------------------
# Individual — items property and iter
# ---------------------------------------------------------------------------

class TestIndividualAccessors:
    def test_items_property(self):
        a = Individual({"A": 1, "B": 2})
        assert ("A", 1) in a.items
        assert ("B", 2) in a.items

    def test_iter_yields_keys(self):
        a = Individual({"A": 1, "B": 2})
        assert set(a) == {"A", "B"}

    def test_setitem_unknown_key_raises(self):
        a = Individual({"A": 0})
        with pytest.raises(KeyError):
            a["NonExistent"] = 1

    def test_update_resets_fitness(self):
        a = Individual({"A": 0, "B": 1}, F=3.0)
        a.update({"A": 1})
        assert a.F == 0.0


# ---------------------------------------------------------------------------
# IndividualSet
# ---------------------------------------------------------------------------

class TestIndividualSet:
    def test_init_from_individual_set_copies_capacity(self):
        s1 = IndividualSet(capacity=5)
        s1.add(Individual({"A": 0}))
        s2 = IndividualSet(s1)  # copy from IndividualSet
        assert s2.capacity == 5

    def test_init_exceeds_capacity_raises(self):
        inds = [Individual({"A": i}) for i in range(3)]
        with pytest.raises(ExceedsCapacity):
            IndividualSet(inds, capacity=2)

    def test_add_at_capacity_raises(self):
        s = IndividualSet(capacity=1)
        s.add(Individual({"A": 0}))
        with pytest.raises(ExceedsCapacity):
            s.add(Individual({"A": 1}))


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

class TestPopulationExtended:
    def _pop(self, n=4):
        return Population([
            Individual({"DepthU": i, "SourceSwap": i % 2}, F=float(i + 1))
            for i in range(n)
        ])

    def test_init_from_set_of_individuals(self):
        inds = {Individual({"A": 0}), Individual({"A": 1})}
        pop = Population(inds)
        assert pop.size == 2

    def test_init_single_individual_wraps_in_list(self):
        ind = Individual({"A": 0})
        pop = Population(ind)
        assert pop.size == 1

    def test_init_rejects_non_individuals(self):
        with pytest.raises(ValueError, match="must be of type Individual"):
            Population(["not-an-individual"])

    def test_init_rejects_mismatched_variable_sets(self):
        with pytest.raises(ValueError, match="same variables"):
            Population([
                Individual({"A": 0}),
                Individual({"B": 0}),
            ])

    def test_get_single_key(self):
        pop = self._pop(4)
        vals = pop.get("DepthU")
        assert vals.shape == (4,)

    def test_get_multiple_keys(self):
        pop = self._pop(4)
        vals = pop.get("DepthU", "SourceSwap")
        assert vals.shape == (4, 2)

    def test_ary_returns_ndarray(self):
        pop = self._pop(3)
        ary = pop.ary
        assert isinstance(ary, np.ndarray)
        assert ary.shape == (3, 2)

    def test_diversity_reduce_false_returns_dict(self):
        pop = self._pop(4)
        div = pop.diversity(reduce=False)
        assert isinstance(div, dict)
        assert set(div.keys()) == set(pop[0].names)

    def test_diversity_empty_population_returns_zero(self):
        pop = Population()
        assert pop.diversity() == 0

    def test_f_setter_assigns_scores(self):
        pop = self._pop(3)
        pop.F = [10.0, 20.0, 30.0]
        assert pop[0].F == 10.0
        assert pop[1].F == 20.0
        assert pop[2].F == 30.0

    def test_g_getter_and_setter(self):
        pop = Population([
            Individual({"A": 0}, G=[0]),
            Individual({"A": 1}, G=[0]),
        ])
        scores = np.array([[1.0, 2.0]])  # shape (1, 2) → G[i] = scores[:, i]
        pop.G = scores
        assert pop[0].G[0] == 1.0
        assert pop[1].G[0] == 2.0
        g_arr = pop.G
        assert g_arr.shape[0] == 2

    def test_argsort_descending_fitness(self):
        pop = Population([
            Individual({"A": 0}, F=1.0),
            Individual({"A": 1}, F=3.0),
            Individual({"A": 2}, F=2.0),
        ])
        indices = pop.argsort()
        assert indices[0] == 1  # highest F first

    def test_sort_returns_descending_order(self):
        pop = Population([
            Individual({"A": 0}, F=1.0),
            Individual({"A": 1}, F=3.0),
            Individual({"A": 2}, F=2.0),
        ])
        sorted_pop = pop.sort()
        assert sorted_pop[0].F >= sorted_pop[1].F >= sorted_pop[2].F

    def test_shuffle_returns_same_size(self):
        pop = self._pop(5)
        shuffled = pop.shuffle()
        assert len(shuffled) == 5

    def test_copy_independence(self):
        pop = self._pop(3)
        copied = pop.copy()
        copied[0]["DepthU"] = 999
        assert pop[0]["DepthU"] != 999

    def test_nunique_no_key_returns_series(self):
        import pandas as pd
        pop = self._pop(4)
        result = pop.nunique()
        assert isinstance(result, pd.Series)

    def test_nunique_with_key_returns_int(self):
        pop = self._pop(4)
        result = pop.nunique("SourceSwap")
        assert isinstance(result, int)
        assert result <= 2  # SourceSwap only has 0 and 1

    def test_unique_with_key_returns_array(self):
        pop = self._pop(4)
        unique_vals = pop.unique("SourceSwap")
        assert len(unique_vals) <= 2

    def test_unique_with_key_return_index(self):
        pop = self._pop(4)
        unique_vals, indices = pop.unique("SourceSwap", return_index=True)
        assert len(unique_vals) == len(indices)

    def test_str_delegates_to_repr(self):
        pop = self._pop(2)
        # __str__ calls __repr__; just ensure it doesn't crash
        s = str(pop)
        assert isinstance(s, str)
