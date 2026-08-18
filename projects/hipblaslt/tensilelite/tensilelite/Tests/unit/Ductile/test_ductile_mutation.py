# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
