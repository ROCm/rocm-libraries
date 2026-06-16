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

import pytest

from Tensile.ductile.core import Mutation, SearchSpace
from Tensile.ductile.core.population import Individual

pytestmark = pytest.mark.unit


def _space():
    return SearchSpace({"DepthU": [32, 64, 128], "SourceSwap": [0, 1]}, max_iters=2)


class TestMutationContracts:
    def test_rejects_invalid_probability_and_weight_type(self):
        space = _space()
        with pytest.raises(ValueError, match="probabilities must be a float"):
            Mutation(space, prob=2.0)
        with pytest.raises(ValueError, match="weights must be a dictionary"):
            Mutation(space, prob=0.2, weights=[1, 2])

    def test_never_introduces_out_of_space_values(self):
        space = _space()
        mutation = Mutation(space, prob=1.0)
        mutated = mutation(Individual({"DepthU": 1, "SourceSwap": 0}))

        assert mutated["DepthU"] in space["DepthU"]
        assert mutated["SourceSwap"] in space["SourceSwap"]
