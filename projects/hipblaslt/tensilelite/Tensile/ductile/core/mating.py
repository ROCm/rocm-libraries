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
from .selection import Selection
from .crossover import Crossover
from .mutation import Mutation
from .population import Population, IndividualSet, ExceedsCapacity
from .space import SearchSpace
from typing import Callable


class Mating:
    def __init__(self,
                 space: SearchSpace,
                 selection: Selection,
                 crossover: Crossover,
                 mutation: Mutation,
                 max_iters: int = 100):
        self.space = space
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.max_iters = max_iters

    def __call__(self, pop: Population, n_offsprings: int = None) -> Population:
        n_offsprings = n_offsprings if n_offsprings else pop.size
        parents = self.selection(pop)

        offsprings, it = IndividualSet(capacity=n_offsprings), 0
        while it < self.max_iters:
            it += 1
            try:
                for inda, indb in self.crossover(parents, n_offsprings):  # TODO
                    inda = self.mutation(inda)
                    if self.space.valid(inda):
                        offsprings.add(inda)
                    indb = self.mutation(indb)
                    if self.space.valid(indb):
                        offsprings.add(indb)
            except ExceedsCapacity as e:
                break

        if it == self.max_iters:
            raise ValueError("max iters reached while generating offsprings")

        return Population(offsprings)

    def __repr__(self):
        msg = f"{self.selection}\n{self.crossover}\n{self.mutation}"
        return msg
