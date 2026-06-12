################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
