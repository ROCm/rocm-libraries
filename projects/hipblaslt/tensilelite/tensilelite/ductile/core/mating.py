# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from .selection import Selection
from .crossover import Crossover
from .mutation import Mutation
from .population import Population, IndividualSet, ExceedsCapacity
from .space import SearchSpace
from typing import Callable


class MatingExhaustedError(Exception):
    pass


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
                for inda, indb in self.crossover(parents, n_offsprings):
                    inda = self.mutation(inda)
                    if self.space.valid(inda):
                        offsprings.add(inda)
                    indb = self.mutation(indb)
                    if self.space.valid(indb):
                        offsprings.add(indb)
            except ExceedsCapacity as e:
                break

        if it == self.max_iters:
            raise MatingExhaustedError("max iters reached while generating offsprings")

        return Population(offsprings)

    def __repr__(self):
        msg = f"{self.selection}\n{self.crossover}\n{self.mutation}"
        return msg
