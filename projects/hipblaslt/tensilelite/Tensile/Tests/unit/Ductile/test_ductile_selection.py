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

"""Extended tests for Tensile.ductile.core.selection targeting uncovered paths.

Covers: Beta, Random, Rank, Tournament, RouletteWheel, Truncation; elitism paths;
replacement=False; __repr__ methods; and Selection base __call__ size contract.
"""

import numpy as np
import pytest

from Tensile.ductile.core import Selection
from Tensile.ductile.core.population import Individual, Population

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
# Truncation selection (bonus — also uncovered)
# ---------------------------------------------------------------------------

class TestTruncationSelection:
    def test_basic(self):
        pop = _pop(8)
        sel = Selection.get("truncation", elite_ratio=0.5, ratio=0.5, elitism=0.0, replacement=True)
        parents = sel(pop)
        assert parents.size == 4


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
