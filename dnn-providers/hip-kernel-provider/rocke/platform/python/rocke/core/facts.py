# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The facts stated for a Value, read without a builder.

A fact about an integer Value is stated in one of two places:

1. A kernel parameter's declaration (``IRBuilder.param``), kept in
   ``Param.attrs``. The host checks it before every launch.
2. An ``arith.assume_range`` marker, written only by a caller that checked or
   proved its facts first.

:func:`stated_facts_of` reads both and nothing else: it never runs the IR
range walk, so it returns exactly what was stated. A bound that names another
Value -- a relation such as ``ldc ge N``, or a marker whose upper end is a
Value -- is resolved to a number through that Value's own stated facts,
transitively: ``start lt T`` with ``T le 4096`` gives ``start <= 4095``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import (
    _UNSTATED,
    FACT_WORDS,
    Value,
    _checked_end,
    _param_interval_resolver,
    _stated_param_interval,
)

__all__ = ["StatedFacts", "stated_facts_of"]


@dataclass(frozen=True)
class StatedFacts:
    """The facts stated for one Value, with every bound resolved to a number.

    ``words`` names the comparison words that were stated (``FACT_WORDS``
    order). ``lower`` / ``upper`` are the inclusive ends they give, ``None``
    when no word states that side or its bound does not resolve to a number.
    The Value is congruent to ``residue`` modulo ``multiple_of``.
    """

    words: tuple[str, ...] = ()
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
        return StatedFacts(("eq",), value, value)
    if value.param is not None:
        return _declared_facts(value)
    op = value.op
    if op is not None and op.name == "arith.assume_range":
        return _marker_facts(op)
    return _NONE


def _declared_facts(value: Value) -> StatedFacts:
    param = value.param
    interval = _param_interval_resolver(param.declared_before)
    lower, upper = _stated_param_interval(param, interval)
    return StatedFacts(
        tuple(w for w in FACT_WORDS if w in param.attrs),
        None if lower is _UNSTATED else lower,
        None if upper is _UNSTATED else upper,
        param.attrs.get("multiple_of", 1),
        0,
    )


def _marker_facts(op) -> StatedFacts:
    attrs = op.attrs
    words = []
    if "ge" in attrs:
        words.append("ge")
    upper: int | None = None
    if len(op.operands) == 2:
        words.append("lt")
        end = stated_facts_of(op.operands[1]).upper
        if end is not None:
            upper = _checked_end(end + attrs.get("lt_plus", 0) - 1)
    elif "lt" in attrs:
        words.append("lt")
        upper = attrs["lt"] - 1
    multiple_of = attrs.get("multiple_of", 1)
    if multiple_of > 1:
        words.append("multiple_of")
    return StatedFacts(
        tuple(words),
        attrs.get("ge"),
        upper,
        multiple_of,
        attrs.get("residue", 0),
    )
