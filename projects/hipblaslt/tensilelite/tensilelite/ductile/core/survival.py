# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from .population import Population
from abc import abstractmethod


class Survival:
    __registry__ = {}
    name = ""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__registry__[cls.name] = cls

    @classmethod
    def get(cls, name: str, *args, **kwargs):
        if name not in cls.__registry__:
            raise ValueError(f"Survival must be on of {list(cls.__registry__.keys())}")
        return cls.__registry__[name](*args, **kwargs)

    @abstractmethod
    def __call__(self,
                 old_pop: Population,
                 pop: Population,
                 size: int,
                 ratio: float = 1.0,  # TODO make op as others
                 *args,
                 **kwargs) -> Population:
        raise NotImplementedError("Subclasses should implement this!")


class Fitness(Survival):
    name = "fitness"

    def __call__(self,
                 old_pop: Population,
                 pop: Population,
                 size: int,
                 *args,
                 **kwargs) -> Population:
        if old_pop.size == 0:
            return pop
        pop = old_pop.merge(pop).sort()
        return pop[:size]

    def __repr__(self):
        return f"Survival(name={self.name})"


class Current(Survival):
    name = "current"

    def __call__(self,
                 old_pop: Population,
                 pop: Population,
                 *args,
                 **kwargs) -> Population:
        return pop

    def __repr__(self):
        return f"Survival(name={self.name})"
