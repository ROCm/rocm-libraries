# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from .population import Population, Individual
from typing import Sequence
from itertools import combinations
from abc import abstractmethod

import math
import numpy as np
import random


class Crossover:
    __registry__ = {}
    name = "base"

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__registry__[cls.name] = cls

    @classmethod
    def get(cls, name: str, *args, **kwargs):
        if name not in cls.__registry__:
            raise ValueError(f"crossover must be on of {list(cls.__registry__.keys())}")
        return cls.__registry__[name](*args, **kwargs)

    def __init__(self,
                 prob: float = 0.9,
                 mode: str = "random"):
        if not (hasattr(self, mode) and callable(getattr(self, mode))):
            raise ValueError(f"unknown pairing mode '{mode}'")

        self.prob = prob
        self.mode = getattr(self, mode)

    def pairing(self, parents, n_matings):
        pairs = np.fromiter(combinations(range(parents.size), 2), dtype=(int, 2))
        p = self.mode(parents, pairs)
        indices = np.random.choice(a=len(pairs), size=n_matings, p=p, replace=n_matings > len(pairs))
        return parents.view(np.ndarray)[pairs[indices]]

    def random(self, *args):
        return None

    def diverse(self, parents, pairs):
        distances = np.array([Population(parents[pair]).diversity() for pair in pairs])
        return distances / distances.sum()

    def fitness(self, parents, pairs):
        F = np.array([sum(parents[pair]) for pair in pairs])
        return F / F.sum()

    def rank(self, parents, pairs):
        order = np.argsort(np.array([sum(parents[pair]) for pair in pairs]))[::-1]
        ranks = order.argsort() + 1
        n = len(pairs)
        p = [(2 * (n - r + 1)) / (n * (n + 1)) for r in ranks]
        return np.array(p)

    @abstractmethod
    def op(self, pa: Individual, pb: Individual) -> Sequence[Individual]:
        raise NotImplementedError("Subclasses should implement this!")

    def __call__(self,
                 parents: Population,
                 n_off: int) -> Sequence[Individual]:
        n_matings = math.ceil(n_off / 2)
        for pa, pb in self.pairing(parents, n_matings):
            yield self.op(pa, pb) if random.random() < self.prob and pa != pb else (pa, pb)

    def __repr__(self):
        return f"Crossover(name={self.name}, prob={self.prob}, mode={self.mode.__name__})"


class Uniform(Crossover):
    name = "ux"

    def op(self, pa: Individual, pb: Individual) -> Sequence[Individual]:
        mask = np.random.random(pa.size) > 0.5
        a = Individual({k: pb[k] if m else pa[k] for m, k in zip(mask, pa.names)})
        b = Individual({k: pa[k] if m else pb[k] for m, k in zip(mask, pa.names)})
        return a, b


class HalfUniform(Uniform):
    name = "hux"

    def op(self, pa: Individual, pb: Individual) -> Sequence[Individual]:
        diff = pa.diff(pb)
        swaps = len(diff) // 2
        a, b = pa.copy(), pb.copy()
        mask = np.random.random(len(diff)) > 0.5
        while mask.sum() < swaps:
            mask = np.random.random(len(diff)) > 0.5
        a.update({k: pb[k] if m else pa[k] for m, k in zip(mask, diff)})
        b.update({k: pa[k] if m else pb[k] for m, k in zip(mask, diff)})
        return a, b


class SinglePoint(Crossover):
    name = "spx"

    def op(self, pa: Individual, pb: Individual) -> Sequence[Individual]:
        p = int(random.random() * pa.size)
        a = Individual(dict(zip(pa.names, pa.values[:p] + pb.values[p:])))
        b = Individual(dict(zip(pa.names, pb.values[:p] + pa.values[p:])))
        return a, b


class TwoPoint(Crossover):
    name = "tpx"

    def op(self, pa: Individual, pb: Individual) -> Sequence[Individual]:
        p0, p1 = sorted(np.random.choice(pa.size, 2, replace=False))
        a = Individual(dict(zip(pa.names, pa.values[:p0] + pb.values[p0:p1] + pa.values[p1:])))
        b = Individual(dict(zip(pa.names, pb.values[:p0] + pa.values[p0:p1] + pb.values[p1:])))
        return a, b
