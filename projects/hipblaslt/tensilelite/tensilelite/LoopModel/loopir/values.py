# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Immutable non-node semantic values used by LoopIR."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IRTypeError(TypeError):
    """A LoopIR value is malformed."""


class TransformRefused(ValueError):
    """A well-formed structural edit cannot satisfy its local contract."""


class OperandRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


INPUT = OperandRole.INPUT
OUTPUT = OperandRole.OUTPUT


class Scope(str, Enum):
    WAVE = "wave"
    BLOCK = "block"
    CLUSTER = "cluster"
    GRID = "grid"


WAVE = Scope.WAVE
BLOCK = Scope.BLOCK
CLUSTER = Scope.CLUSTER
GRID = Scope.GRID


class _Sentinel:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name

    __str__ = __repr__


IMMEDIATE = _Sentinel("IMMEDIATE")


@dataclass(frozen=True, eq=False)
class CompletionClass:
    name: str
    counter_scope: Scope

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise IRTypeError("a CompletionClass needs a non-empty string name")
        if not isinstance(self.counter_scope, Scope):
            raise IRTypeError("CompletionClass.counter_scope must be a Scope")
