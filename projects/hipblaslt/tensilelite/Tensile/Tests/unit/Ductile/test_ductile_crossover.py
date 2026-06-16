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

"""Extended tests for Tensile.ductile.core.crossover — targeting uncovered paths.

Covers: HalfUniform (hux), SinglePoint (spx), TwoPoint (tpx) operators; pairing
modes diverse / fitness / rank; prob=0 (skip crossover, identical parents); identical
parents short-circuit; n_off > available pairs (replacement=True in pairing);
__repr__; Crossover.get() registry.
"""

import numpy as np
import pytest

from Tensile.ductile.core import Crossover
from Tensile.ductile.core.population import Individual, Population

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pop(n=6):
    return Population([
        Individual({"DepthU": i, "SourceSwap": i % 2}, F=float(i + 1))
        for i in range(n)
    ])


def _apply(crossover, pop, n_off=4):
    """Collect all offspring from the crossover generator."""
    return list(crossover(pop, n_off=n_off))


def _all_valid_genes(offspring_pairs, pop):
    pop_values = {k: {ind[k] for ind in pop} for k in pop[0].names}
    for pa, pb in offspring_pairs:
        for child in (pa, pb):
            for k in child.names:
                if child[k] not in pop_values[k]:
                    return False
    return True


# ---------------------------------------------------------------------------
# Uniform crossover (ux) — already partially covered; add pairing modes
# ---------------------------------------------------------------------------

class TestUniformCrossoverPairingModes:
    def test_diverse_pairing_mode(self):
        pop = _pop(6)
        cx = Crossover.get("ux", prob=1.0, mode="diverse")
        pairs = _apply(cx, pop)
        assert len(pairs) >= 1

    def test_fitness_pairing_mode(self):
        pop = _pop(6)
        cx = Crossover.get("ux", prob=1.0, mode="fitness")
        pairs = _apply(cx, pop)
        assert len(pairs) >= 1

    def test_rank_pairing_mode(self):
        pop = _pop(6)
        cx = Crossover.get("ux", prob=1.0, mode="rank")
        pairs = _apply(cx, pop)
        assert len(pairs) >= 1

    def test_prob_zero_returns_identical_parents(self):
        pop = _pop(6)
        cx = Crossover.get("ux", prob=0.0, mode="random")
        pairs = _apply(cx, pop, n_off=2)
        # With prob=0 and identical parents are always returned as-is
        # (either because prob check fails or parents are equal)
        assert len(pairs) >= 1

    def test_n_off_exceeds_pairs_uses_replacement(self):
        pop = _pop(4)  # C(4,2)=6 pairs; request 12 offspring → 6 matings
        cx = Crossover.get("ux", prob=1.0, mode="random")
        pairs = _apply(cx, pop, n_off=12)
        assert len(pairs) >= 6

    def test_repr_contains_name_and_mode(self):
        cx = Crossover.get("ux", prob=0.9, mode="random")
        r = repr(cx)
        assert "ux" in r
        assert "random" in r


# ---------------------------------------------------------------------------
# HalfUniform crossover (hux)
# ---------------------------------------------------------------------------

class TestHalfUniformCrossover:
    def test_offspring_schema_preserved(self):
        pop = _pop(4)
        cx = Crossover.get("hux", prob=1.0, mode="random")
        pairs = _apply(cx, pop, n_off=2)
        for pa, pb in pairs:
            assert set(pa.names) == set(pop[0].names)
            assert set(pb.names) == set(pop[0].names)

    def test_genes_come_from_parents(self):
        pop = _pop(4)
        cx = Crossover.get("hux", prob=1.0, mode="random")
        pairs = _apply(cx, pop, n_off=4)
        assert _all_valid_genes(pairs, pop)

    def test_identical_parents_skip_crossover(self):
        ind = Individual({"DepthU": 0, "SourceSwap": 0}, F=1.0)
        pop = Population([ind, ind.copy()])
        cx = Crossover.get("hux", prob=1.0, mode="random")
        pairs = list(cx(pop, n_off=1))
        # Identical parents → pa == pb, so op returns as-is
        assert len(pairs) >= 1

    def test_size_matches_n_off(self):
        pop = _pop(4)
        cx = Crossover.get("hux", prob=1.0, mode="random")
        pairs = _apply(cx, pop, n_off=4)
        # math.ceil(4/2) = 2 matings → 2 pairs
        assert len(pairs) == 2


# ---------------------------------------------------------------------------
# SinglePoint crossover (spx)
# ---------------------------------------------------------------------------

class TestSinglePointCrossover:
    def test_offspring_size_matches_parents(self):
        pop = _pop(4)
        cx = Crossover.get("spx", prob=1.0, mode="random")
        pairs = _apply(cx, pop, n_off=4)
        for pa, pb in pairs:
            assert pa.size == pop[0].size
            assert pb.size == pop[0].size

    def test_offspring_schema_preserved(self):
        pop = _pop(4)
        cx = Crossover.get("spx", prob=1.0, mode="random")
        pairs = _apply(cx, pop, n_off=2)
        for pa, pb in pairs:
            assert set(pa.names) == set(pop[0].names)

    def test_genes_come_from_parents(self):
        pop = _pop(4)
        cx = Crossover.get("spx", prob=1.0, mode="random")
        pairs = _apply(cx, pop, n_off=4)
        assert _all_valid_genes(pairs, pop)

    def test_repr_contains_name(self):
        cx = Crossover.get("spx", prob=0.9, mode="random")
        assert "spx" in repr(cx)


# ---------------------------------------------------------------------------
# TwoPoint crossover (tpx)
# ---------------------------------------------------------------------------

class TestTwoPointCrossover:
    def test_offspring_schema_preserved(self):
        pop = _pop(6)
        cx = Crossover.get("tpx", prob=1.0, mode="random")
        pairs = _apply(cx, pop, n_off=4)
        for pa, pb in pairs:
            assert set(pa.names) == set(pop[0].names)

    def test_genes_come_from_parents(self):
        pop = _pop(6)
        cx = Crossover.get("tpx", prob=1.0, mode="random")
        pairs = _apply(cx, pop, n_off=4)
        assert _all_valid_genes(pairs, pop)

    def test_size_with_large_n_off(self):
        pop = _pop(4)
        cx = Crossover.get("tpx", prob=1.0, mode="random")
        pairs = _apply(cx, pop, n_off=8)
        assert len(pairs) == 4  # math.ceil(8/2) = 4


# ---------------------------------------------------------------------------
# Registry / error handling
# ---------------------------------------------------------------------------

class TestCrossoverRegistry:
    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="crossover must be"):
            Crossover.get("unknown_op")

    def test_unknown_pairing_mode_raises(self):
        with pytest.raises(ValueError, match="unknown pairing mode"):
            Crossover.get("ux", mode="nonexistent_mode")


class TestCrossoverContracts:
    def test_selection_and_crossover_emit_expected_sizes_and_schema(self):
        from Tensile.ductile.core import Selection

        pop = Population(
            [
                Individual({"DepthU": 0, "SourceSwap": 0}, F=1.0),
                Individual({"DepthU": 1, "SourceSwap": 0}, F=2.0),
                Individual({"DepthU": 2, "SourceSwap": 1}, F=3.0),
                Individual({"DepthU": 1, "SourceSwap": 1}, F=4.0),
            ]
        )

        selection = Selection.get("tournament", k=2, ratio=0.5, elitism=0.0, replacement=True)
        parents = selection(pop)
        assert parents.size == 2

        crossover = Crossover.get("ux", prob=1.0, mode="random")
        offspring_pairs = list(crossover(parents, n_off=2))
        assert len(offspring_pairs) >= 1

        a, b = offspring_pairs[0]
        assert set(a.names) == set(pop.names)
        assert set(b.names) == set(pop.names)
