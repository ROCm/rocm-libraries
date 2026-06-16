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
