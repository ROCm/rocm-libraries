# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from typing import Sequence, Union
from numbers import Number
from typing import Sequence
import functools
import numpy as np


@functools.total_ordering
class Individual:
    def __init__(self, X: dict, F: float = 0.0, G: Sequence = [0]):
        if not (isinstance(X, dict) and len(X) > 0):
            raise ValueError("X must be a non-empty dictionary.")
        if not all(isinstance(v, Number) for v in X.values()):
            raise ValueError("X must be a dictionary of basic types.")
        self.X = dict(sorted(X.items()))
        self.F = float(F)  # Convert to Python float to ensure consistency
        self.G = G

    @property
    def names(self):
        return tuple(self.X.keys())

    @property
    def values(self):
        return tuple(self.X.values())

    @property
    def items(self):
        return tuple(zip(self.names, self.values))

    @property
    def size(self):
        return len(self.X)

    def copy(self):
        return Individual(self.X, self.F, self.G)  # init already makes a copy of X

    def update(self, X: dict):
        self.X.update(X)
        self.F = 0.0

    def diff(self, other):
        return sorted({k for k in self.names if self[k] != other[k]})

    def __getitem__(self, key: str):
        return self.X[key]

    def __iter__(self):
        return dict.__iter__(self.X)

    def __setitem__(self, key: str, value):
        if key not in self.X:
            raise KeyError(key)
        self.X[key] = value
        self.F = 0.0

    def __eq__(self, other):
        if isinstance(other, Individual):
            return self.X == other.X
        return False

    def __lt__(self, other):
        if isinstance(other, Individual):
            return self.F < other.F
        return self.F < other

    def __add__(self, other):
        return self.F + other.F if isinstance(other, Individual) else self.F + other

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.F - other.F if isinstance(other, Individual) else self.F - other

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        return self.F * other.F if isinstance(other, Individual) else self.F * other

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.F / other.F if isinstance(other, Individual) else self.F / other

    def __copy__(self):
        return self.copy()

    def __hash__(self):
        return hash(self.values)

    def __repr__(self):
        return f"Individual(X={self.X}, F={self.F:.4f})"


class ExceedsCapacity(Exception):
    pass


class IndividualSet(set):
    def __init__(self, s=(), capacity=np.inf):
        self.capacity = capacity

        if isinstance(s, IndividualSet):
            self.capacity = s.capacity
        self.msg = f"{self.capacity}"

        if self.capacity and len(s) > self.capacity:
            raise ExceedsCapacity(self.msg)
        super().__init__(s)

    def add(self, el):
        new_size = len(self) + (len(el) if hasattr(el, "__len__") else 1)
        if new_size > self.capacity:
            raise ExceedsCapacity(self.msg)
        super().add(el)


class Population(np.ndarray, Sequence):
    def __new__(cls, individuals=[]):
        if isinstance(individuals, Individual):
            individuals = [individuals]
        if isinstance(individuals, set):
            individuals = list(individuals)
        if not all([isinstance(individual, Individual) for individual in individuals]):
            raise ValueError("must be of type Individual")
        if len(set(tuple(ind.names for ind in individuals))) > 1:
            raise ValueError("all individuals must have the same variables")
        return np.array(individuals).view(cls)

    def __array_finalize__(self, obj):
        self.names = ()
        if hasattr(obj, "__len__") and len(obj) == 0:
            return
        if isinstance(obj[0], Individual):
            self.names = obj[0].names

    @property
    def shape(self):
        return self.size, len(self.names)

    def get(self, *keys):
        return np.array([[p[k] for k in keys] for p in self]).squeeze()

    @property
    def ary(self):
        return np.array([ind.values for ind in self])

    def unique(self,
               key: str = None,
               return_index=False):
        if key:
            return np.unique([p[key] for p in self], return_index=return_index)

        seen = set()
        indices = []
        for i, p in enumerate(self):
            key_tuple = p.values
            if key_tuple not in seen:
                seen.add(key_tuple)
                indices.append(i)

        uniq_pop = Population([self[i].copy() for i in indices])
        if not return_index:
            return uniq_pop
        return uniq_pop, np.array(indices)

    def nunique(self, key: str = None) -> Union[dict, int]:
        if key:
            return len(np.unique([p[key] for p in self]))
        return {name: len(set(p[name] for p in self)) for name in self.names}

    def merge(self, other):
        merged = np.concatenate((self, other)).view(Population)
        if len(set(tuple(ind.names for ind in merged))) > 1:
            raise ValueError("all individuals must have the same variables")
        return merged

    def diversity(self, reduce=True):
        if self.size == 0:
            return 0

        X = self.ary
        n = len(X)
        if n < 2:
            return 0 if reduce else dict.fromkeys(self.names, 0)

        if reduce:
            diffs = X[:, None, :] != X[None, :, :]
            hamming = diffs.mean(axis=2)
            return hamming[np.triu_indices(n, k=1)].mean()

        idx = np.triu_indices(n, k=1)
        div = {}
        for i, name in enumerate(self.names):
            col = X[:, i]
            div[name] = (col[:, None] != col[None, :])[idx].mean()
        return div

    @property
    def F(self):
        return np.array([ind.F for ind in self])

    @F.setter
    def F(self, scores):
        for ind, score in zip(self, scores):
            ind.F = score

    @property
    def G(self):
        return np.array([ind.G for ind in self])

    @G.setter
    def G(self, scores):
        for i, ind in enumerate(self):
            ind.G = scores[:, i]

    def argsort(self):
        return np.array(np.argsort(self))[::-1]

    def sort(self):
        return self[self.argsort()]

    def shuffle(self):
        return np.random.permutation(self)

    def copy(self):
        return Population([ind.copy() for ind in self])

    def __str__(self):
        return self.__repr__()
