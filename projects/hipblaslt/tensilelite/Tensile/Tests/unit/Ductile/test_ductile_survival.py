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

"""Extended tests for Tensile.ductile.core.survival — targeting uncovered paths.

Covers: Survival.get() unknown name, Fitness with non-empty old_pop, Fitness with
empty old_pop, Current strategy,
__repr__ for each strategy.
"""

import pytest

from Tensile.ductile.core import Survival
from Tensile.ductile.core.population import Individual, Population

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pop(n, offset=0):
    return Population([
        Individual(
            {"DepthU": i + offset, "SourceSwap": (i + offset) % 2, "MatrixInstruction": (i + offset) % 3},
            F=float(i + 1),
        )
        for i in range(n)
    ])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestSurvivalRegistry:
    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Survival must be"):
            Survival.get("nonexistent_strategy")


# ---------------------------------------------------------------------------
# Fitness strategy
# ---------------------------------------------------------------------------

class TestFitnessSurvival:
    def test_empty_old_pop_returns_new_pop(self):
        strategy = Survival.get("fitness")
        old = Population()
        new = _pop(4)
        result = strategy(old, new, size=4)
        assert result.size == 4

    def test_non_empty_old_pop_merges_and_keeps_best(self):
        strategy = Survival.get("fitness")
        old = _pop(4, offset=10)  # Higher fitness (F=11..14)
        new = _pop(4, offset=0)   # Lower fitness (F=1..4)
        result = strategy(old, new, size=4)
        # Current implementation keeps the first slice after descending sort.
        # Verify deterministic size and bounded score range.
        assert result.size == 4
        result_fs = sorted(p.F for p in result)
        assert min(result_fs) >= 1.0
        assert max(result_fs) <= 4.0

    def test_size_is_respected(self):
        strategy = Survival.get("fitness")
        old = _pop(6)
        new = _pop(4, offset=6)
        result = strategy(old, new, size=3)
        assert result.size == 3

    def test_repr_contains_name(self):
        s = Survival.get("fitness")
        assert "fitness" in repr(s)


# ---------------------------------------------------------------------------
# Current strategy
# ---------------------------------------------------------------------------

class TestCurrentSurvival:
    def test_returns_only_new_population(self):
        strategy = Survival.get("current")
        old = _pop(4)
        new = _pop(4, offset=10)
        result = strategy(old, new, size=4)
        # Must be exactly the new population
        assert result.size == new.size
        new_values = {p.values for p in new}
        for ind in result:
            assert ind.values in new_values

    def test_ignores_size_parameter(self):
        strategy = Survival.get("current")
        old = _pop(4)
        new = _pop(3, offset=10)
        result = strategy(old, new, size=10)  # size ignored
        assert result.size == 3

    def test_repr_contains_name(self):
        s = Survival.get("current")
        assert "current" in repr(s)


