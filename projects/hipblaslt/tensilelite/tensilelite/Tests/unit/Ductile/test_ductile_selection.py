# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Extended tests for tensilelite.ductile.core.selection targeting uncovered paths.

Covers: Beta, Random, Rank, Tournament, RouletteWheel, Truncation; elitism paths;
replacement=False; __repr__ methods; and Selection base __call__ size contract.
"""

import numpy as np
import pytest

from tensilelite.ductile.core import Selection
from tensilelite.ductile.core.population import Individual, Population

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pop(n=8, distinct=True):
    """Return a Population of n individuals with distinct DepthU values."""
    inds = [
        Individual({"DepthU": i, "SourceSwap": i % 2, "MatrixInstruction": i % 3}, F=float(i + 1))
        for i in range(n)
    ]
    return Population(inds)


def _pop_uniform_fitness(n=6):
    """Population where all individuals share the same fitness (edge case for rank/kde)."""
    return Population([
        Individual({"DepthU": i, "SourceSwap": 0, "MatrixInstruction": 0}, F=1.0)
        for i in range(n)
    ])


# ---------------------------------------------------------------------------
# Selection base __call__ — elitism + size contracts
# ---------------------------------------------------------------------------

class TestSelectionBaseCall:
    def test_output_size_equals_ceil_ratio_times_pop(self):
        pop = _pop(8)
        sel = Selection.get("random", ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 4

    def test_elitism_preserves_best_individual(self):
        pop = _pop(8)
        sel = Selection.get("tournament", k=2, ratio=0.5, elitism=0.5, replacement=True)
        parents = sel(pop)
        # Best individual (highest F) must be in output when elitism > 0
        best_F = pop.sort()[0].F
        assert any(p.F == best_F for p in parents)

    def test_elitism_zero_gives_no_guaranteed_best(self):
        """With elitism=0.0, no individual is forcibly kept — just size contract."""
        pop = _pop(8)
        sel = Selection.get("random", ratio=0.25, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 2

    def test_repr_contains_name(self):
        sel = Selection.get("random", ratio=0.5, elitism=0.0)
        assert "random" in repr(sel)


# ---------------------------------------------------------------------------
# Beta selection
# ---------------------------------------------------------------------------

class TestBetaSelection:
    def test_with_replacement_returns_correct_size(self):
        pop = _pop(10)
        sel = Selection.get("beta", ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 5

    def test_without_replacement_unique_parents(self):
        pop = _pop(10)
        sel = Selection.get("beta", ratio=0.4, elitism=0.0, replacement=False)
        parents = sel(pop)
        assert parents.size == 4
        # All parents are unique
        assert len(set(p.values for p in parents)) == parents.size

    def test_custom_a_b_params(self):
        pop = _pop(8)
        sel = Selection.get("beta", a=0.5, b=1.0, ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 4

    def test_repr_includes_a_b(self):
        sel = Selection.get("beta", a=2.0, b=3.0)
        r = repr(sel)
        assert "a=2.0" in r
        assert "b=3.0" in r


# ---------------------------------------------------------------------------
# Random selection
# ---------------------------------------------------------------------------

class TestRandomSelection:
    def test_with_replacement(self):
        pop = _pop(6)
        sel = Selection.get("random", ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 3

    def test_without_replacement_unique(self):
        pop = _pop(8)
        sel = Selection.get("random", ratio=0.5, elitism=0.0, replacement=False)
        parents = sel(pop)
        assert parents.size == 4
        assert len(set(p.values for p in parents)) == 4

    def test_schema_preserved(self):
        pop = _pop(6)
        sel = Selection.get("random", ratio=0.5, elitism=0.0)
        parents = sel(pop)
        assert set(parents[0].names) == set(pop[0].names)


# ---------------------------------------------------------------------------
# Rank selection
# ---------------------------------------------------------------------------

class TestRankSelection:
    def test_returns_correct_size(self):
        pop = _pop(8)
        sel = Selection.get("rank", ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 4

    def test_without_replacement(self):
        pop = _pop(8)
        sel = Selection.get("rank", ratio=0.5, elitism=0.0, replacement=False)
        parents = sel(pop)
        assert parents.size == 4
        assert len(set(p.values for p in parents)) == 4

    def test_all_parents_from_source_population(self):
        pop = _pop(6)
        sel = Selection.get("rank", ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        pop_values = {p.values for p in pop}
        for parent in parents:
            assert parent.values in pop_values


# ---------------------------------------------------------------------------
# Tournament selection
# ---------------------------------------------------------------------------

class TestTournamentSelection:
    def test_with_replacement_size(self):
        pop = _pop(8)
        sel = Selection.get("tournament", k=2, ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 4

    def test_without_replacement_unique(self):
        pop = _pop(8)
        sel = Selection.get("tournament", k=2, ratio=0.5, elitism=0.0, replacement=False)
        parents = sel(pop)
        assert parents.size == 4
        assert len(set(p.values for p in parents)) == 4

    def test_large_k_still_selects(self):
        pop = _pop(8)
        sel = Selection.get("tournament", k=4, ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 4

    def test_repr_includes_k(self):
        sel = Selection.get("tournament", k=3)
        assert "k=3" in repr(sel)


# ---------------------------------------------------------------------------
# Truncation selection — boundary condition coverage
# ---------------------------------------------------------------------------

class TestTruncationSelection:
    def test_basic(self):
        pop = _pop(8)
        sel = Selection.get("truncation", elite_ratio=0.5, ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 4

    def test_small_elite_ratio_clamps_to_one_member(self):
        """elite_ratio=0.1 on 8 members → int(8*0.1)=0, clamped to 1."""
        pop = _pop(8)
        sel = Selection.get("truncation", elite_ratio=0.1, ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 4
        # All parents should be the best individual (only 1 in pool)
        best = pop.sort()[0]
        assert all(p.F == best.F for p in parents)

    def test_elite_ratio_truncation_with_replacement(self):
        """elite_ratio=0.5 with replacement=True samples from top half."""
        pop = _pop(8)
        sel = Selection.get("truncation", elite_ratio=0.5, ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        # All parents must be from top 4 (int(8*0.5)=4)
        top4_fitnesses = {ind.F for ind in pop.sort()[:4]}
        assert all(p.F in top4_fitnesses for p in parents)

    def test_undersized_pool_without_replacement_raises(self):
        """Pool smaller than n_parents with replacement=False raises clear error."""
        pop = _pop(8)
        # elite_ratio=0.25 → int(8*0.25)=2 members, but ratio=0.5 needs 4 parents
        sel = Selection.get("truncation", elite_ratio=0.25, ratio=0.5, elitism=0.0, replacement=False)
        with pytest.raises(ValueError, match=r"elite pool \(2 members\) is smaller than n_parents \(4\)"):
            sel(pop)

    def test_exact_size_pool_without_replacement(self):
        """Pool size == n_parents with replacement=False works."""
        pop = _pop(8)
        # elite_ratio=0.5 → 4 members, ratio=0.5 → 4 parents
        sel = Selection.get("truncation", elite_ratio=0.5, ratio=0.5, elitism=0.0, replacement=False)
        parents = sel(pop)
        assert parents.size == 4
        # All parents are unique
        assert len(set(p.values for p in parents)) == 4

    def test_large_pool_without_replacement(self):
        """Pool larger than n_parents with replacement=False works."""
        pop = _pop(8)
        # elite_ratio=0.75 → 6 members, ratio=0.5 → 4 parents
        sel = Selection.get("truncation", elite_ratio=0.75, ratio=0.5, elitism=0.0, replacement=False)
        parents = sel(pop)
        assert parents.size == 4
        assert len(set(p.values for p in parents)) == 4

    def test_non_integer_product_floors_correctly(self):
        """elite_ratio that produces non-integer pool size floors via int()."""
        pop = _pop(7)
        # elite_ratio=0.4 → int(7*0.4) = int(2.8) = 2
        sel = Selection.get("truncation", elite_ratio=0.4, ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 4
        # Parents from top 2 individuals
        top2_fitnesses = {ind.F for ind in pop.sort()[:2]}
        assert all(p.F in top2_fitnesses for p in parents)

    def test_elite_ratio_one_includes_entire_population(self):
        """elite_ratio=1.0 includes all members in pool."""
        pop = _pop(6)
        sel = Selection.get("truncation", elite_ratio=1.0, ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 3
        all_fitnesses = {ind.F for ind in pop}
        assert all(p.F in all_fitnesses for p in parents)

    def test_zero_elite_ratio_clamps_to_one(self):
        """elite_ratio=0.0 → int(8*0) = 0, clamped to 1."""
        pop = _pop(8)
        sel = Selection.get("truncation", elite_ratio=0.0, ratio=0.25, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 2
        # All parents from single best individual
        best = pop.sort()[0]
        assert all(p.F == best.F for p in parents)

# ---------------------------------------------------------------------------
# RouletteWheel selection (bonus — also uncovered)
# ---------------------------------------------------------------------------

class TestRouletteWheelSelection:
    def test_basic(self):
        pop = _pop(8)
        sel = Selection.get("roulette_wheel", ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 4


class TestSelectionRegistry:
    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="selection must be"):
            Selection.get("not-a-selection")
