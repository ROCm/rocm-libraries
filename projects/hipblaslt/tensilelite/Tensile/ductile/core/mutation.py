# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from .space import SearchSpace
from .population import Individual
import numpy as np


class Mutation:
    def __init__(self,
                 space: SearchSpace,
                 prob: float = None,
                 weights: dict = None):

        if not isinstance(space, SearchSpace):
            raise ValueError("space must be of type SearchSpace")
        self.space = space

        prob = 1 / len(self.space) if prob is None else prob
        if not (isinstance(prob, float) and (0 <= prob <= 1)):
            raise ValueError(f"probabilities must be a float between 0 and 1.")

        self.weights = weights if weights else {}
        if not isinstance(self.weights, dict):
            raise ValueError("weights must be a dictionary")

        self.weights.update({k: 0 for k in self.space if self.space.size(k) < 2})
        self.probs = {k: np.clip(prob * self.weights.get(k, 1.0), 0, 1) for k in self.space}

    def __call__(self, ind: Individual):
        ind = ind.copy()
        mask = [k for p, k in zip(np.random.random(ind.size), ind.names) if p < self.probs[k]]
        ind.update({k: np.random.choice([v for v in self.space[k] if v != ind[k]]) for k in mask})
        return ind

    def __repr__(self):
        prob = next(iter(self.probs.values()))
        if all(v == prob for v in self.probs.values()):
            return f"Mutation(prob={prob:.5f})"
        return f"Mutation(probs={self.probs}, weights={self.weights})"
