# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The facts stated for a Value, read without a builder.

A fact about an integer Value is stated in one of two places:

1. A kernel parameter's declaration (``IRBuilder.param``), kept in
   ``Param.attrs``. ``KernelLauncher.from_artifact`` checks it on the host
   before each launch.
2. An ``arith.assume_range`` marker, written only by a caller that checked or
   proved its facts first, such as ``IRBuilder.gpu_checked_value``, whose
   facts the GPU checks.

:func:`stated_facts_of` reads both and nothing else: it never runs the IR
range walk, so it returns exactly what was stated. A bound that names another
Value -- a relation such as ``ldc ge N``, or a marker whose upper end is a
Value -- is resolved to a number through that Value's own stated facts,
transitively: ``start lt T`` with ``T le 4096`` gives ``start <= 4095``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import MarkerEnd, Op, Value, has_stated_facts, marker_facts
from .param_facts import (
    UNSTATED,
    checked_end,
    param_interval_resolver,
    stated_param_interval,
)
from .value_sources import ASSUME_RANGE_OP

__all__ = ["StatedFacts", "has_stated_facts", "stated_facts_of"]


@dataclass(frozen=True)
class StatedFacts:
    """The facts stated for one Value, with every bound resolved to a number.

    ``lower`` / ``upper`` are the inclusive ends, ``None`` when nothing states
    that side or its bound does not resolve to a number. The Value is congruent
    to ``residue`` modulo ``multiple_of``.
    """

    lower: int | None = None
    upper: int | None = None
    multiple_of: int = 1
    residue: int = 0


_NONE = StatedFacts()


def stated_facts_of(value: Value | int) -> StatedFacts:
    """The facts stated for ``value``: by its parameter declaration, by the
    ``arith.assume_range`` marker that produced it, or, for an int, the int
    itself. A Value with neither has no stated facts."""
    if isinstance(value, bool) or not isinstance(value, (Value, int)):
        raise TypeError(f"stated_facts_of: expected a Value or an int, got {value!r}")
    if isinstance(value, int):
        return StatedFacts(value, value)
    if value.param is not None:
        return _declared_facts(value)
    op = value.op
    if op is not None and op.name == ASSUME_RANGE_OP:
        return _marker_facts(op)
    return _NONE


def _declared_facts(value: Value) -> StatedFacts:
    param = value.param
    interval = param_interval_resolver(param.declared_before)
    lower, upper = stated_param_interval(param, interval)
    return StatedFacts(
        None if lower is UNSTATED else lower,
        None if upper is UNSTATED else upper,
        param.attrs.get("multiple_of", 1),
        0,
    )


def _marker_facts(op: Op) -> StatedFacts:
    stated = marker_facts(op)
    lower = None if stated.lower is None else _inclusive_end(stated.lower, "lower", 0)
    upper = None if stated.upper is None else _inclusive_end(stated.upper, "upper", -1)
    return StatedFacts(lower, upper, stated.multiple_of, stated.residue)


def _inclusive_end(end: MarkerEnd, side: str, offset: int) -> int | None:
    """A marker end as an inclusive number, ``offset`` added (-1 turns the
    exclusive upper end into an inclusive one). An int end is taken as
    stated; a Value end through that Value's own stated facts on ``side``
    (``"lower"`` or ``"upper"``), ``None`` when they do not resolve or the end
    leaves int64."""
    if isinstance(end.bound, Value):
        resolved = getattr(stated_facts_of(end.bound), side)
        if resolved is None:
            return None
        return checked_end(resolved + end.plus + offset)
    return end.bound + end.plus + offset
