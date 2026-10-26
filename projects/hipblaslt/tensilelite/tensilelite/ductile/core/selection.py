# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from .population import Population
from abc import abstractmethod

import math
import numpy as np
import random


class Selection:
    __registry__ = {}
    name = ""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__registry__[cls.name] = cls

    @classmethod
    def get(cls, name: str, *args, **kwargs):
        if name not in cls.__registry__:
            raise ValueError(f"selection must be on of {list(cls.__registry__.keys())}")
        return cls.__registry__[name](*args, **kwargs)

    def __init__(self,
                 ratio: float = 0.5,
                 elitism: float = 1 / 7,
                 replacement: bool = True):
        self.ratio = ratio
        self.elitism = elitism
        self.replace = replacement

    @abstractmethod
    def op(self, pop: Population, n_parents: int):
        raise NotImplementedError("Subclasses should implement this!")

    def __call__(self,
                 pop: Population,
                 *args,
                 **kwargs) -> Population:
        n_parents = math.ceil(pop.size * self.ratio)
        n_elites = round(n_parents * self.elitism)

        sorted_indices = pop.argsort()
        elites = pop[sorted_indices[:n_elites]]
        remaining = pop[sorted_indices[n_elites:]]

        return elites.merge(self.op(remaining, n_parents - n_elites))

    def __repr__(self):
        return f"Selection(name={self.name}, ratio={self.ratio}, elitism={self.elitism:.4f})"


class Beta(Selection):
    name = "beta"

    def __init__(self,
                 a: float = 1.0,
                 b: float = 2.5,
                 **kwargs):
        self.a = a
        self.b = b
        super(Beta, self).__init__(**kwargs)

    def sample(self, size):
        return int(random.betavariate(self.a, self.b) * size)

    def op(self, pop: Population, n_parents: int) -> Population:
        pop = pop.sort()

        if self.replace:
            return pop[[self.sample(pop.size) for _ in range(n_parents)]]

        chosen = set()
        while len(chosen) < n_parents:
            idx = self.sample(pop.size)
            chosen.add(idx)
        return pop[list(chosen)]

    def __repr__(self):
        msg = super().__repr__()
        msg = msg.split(")")[0]
        return f"{msg}, a={self.a}, b={self.b})"


class Random(Selection):
    name = "random"

    def __init__(self, **kwargs):
        super(Random, self).__init__(**kwargs)

    def op(self, pop: Population, n_parents: int) -> Population:
        return pop[np.random.choice(np.arange(pop.size), n_parents, replace=self.replace)]


class Rank(Selection):
    name = "rank"

    def op(self, pop: Population, n_parents: int) -> Population:
        pop = pop.sort()
        n = pop.size
        p = [(2 * (n - r + 1)) / (n * (n + 1)) for r in range(1, n + 1)]
        return pop[np.random.choice(np.arange(n), n_parents, p=p, replace=self.replace)]


class Tournament(Selection):
    name = "tournament"

    def __init__(self, k: int = 2, **kwargs):
        self.k = k
        super(Tournament, self).__init__(**kwargs)

    def op(self, pop: Population, n_parents: int) -> Population:
        pop = pop.sort()
        indices = np.arange(pop.size)
        # TODO use diversity to regulate k maybe
        if self.replace:
            return pop[[np.random.choice(indices, self.k, replace=False).min() for _ in range(n_parents)]]

        chosen = set()
        while len(chosen) < n_parents:
            chosen.add(np.random.choice(indices, self.k, replace=False).min())
        return pop[list(chosen)]

    def __repr__(self):
        msg = super().__repr__()
        msg = msg.split(")")[0]
        return f"{msg}, k={self.k})"


class RouletteWheel(Selection):
    name = "roulette_wheel"

    def op(self, pop: Population, n_parents: int) -> Population:
        cs = pop.cumsum()
        p = pop.F / cs[-1]
        return np.random.choice(pop, n_parents, p=p, replace=self.replace)


class Truncation(Selection):
    name = "truncation"

    def __init__(self, elite_ratio: float, **kwargs):
        self.elite_ratio = elite_ratio
        super(Truncation, self).__init__(**kwargs)

    def op(self, pop: Population, n_parents: int) -> Population:
        pop = pop.sort()
        pop = pop[:int(pop.size * self.elite_ratio)]
        return np.random.choice(pop, n_parents, replace=self.replace)
