# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Knob axes: one tuning decision over kernel-spec fields, as data.

A family declares its space as a tuple of :class:`KnobAxis`, ordered
prerequisites-first with enabler axes leading (see :mod:`.walk`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

Knobs = Tuple[Tuple[str, object], ...]


def sorted_items(values: Mapping[str, object]) -> Knobs:
    return tuple(sorted(values.items()))


@dataclass(frozen=True)
class KnobAxis:
    """One tuning decision over kernel-spec fields; ``choices[0]`` is ``()``,
    "leave the base value".

    ``enabler`` marks knobs that can turn an otherwise-illegal setting legal
    (an LDS-saving knob that lets a larger tile fit). They must lead the axis
    order; see :func:`rocke.dispatch.tuning.walk.iter_knob_sets`.
    """

    name: str
    choices: Tuple[Knobs, ...]
    enabler: bool = False


def flag(name: str, *, enabler: bool = False) -> KnobAxis:
    """Off, or on."""
    return KnobAxis(name, ((), ((name, True),)), enabler)


def values(
    name: str, default: object, options: Sequence[object], *, enabler: bool = False
) -> KnobAxis:
    """Every option except ``default``."""
    return KnobAxis(
        name, ((),) + tuple(((name, v),) for v in options if v != default), enabler
    )


def gated(
    gate: str, sub: Mapping[str, Sequence[object]], *, enabler: bool = False
) -> KnobAxis:
    """``gate`` off, or on with every combination of the sub-knobs it gates.

    The kernel reads the sub-knobs only when ``gate`` is set, so varying them
    while it is off would emit byte-identical duplicate kernels.
    """
    combos: list[Knobs] = [((gate, True),)]
    for name, options in sub.items():
        combos = [c + ((name, v),) for c in combos for v in options]
    return KnobAxis(gate, ((),) + tuple(combos), enabler)


def choices(name: str, options: Sequence[object]) -> KnobAxis:
    """Every option, the base's included, for axes whose base comes from the
    request or a policy; canonicalization drops the base-equal choice."""
    return KnobAxis(name, ((),) + tuple(((name, v),) for v in options))


def knob_requirements(axes: Tuple[KnobAxis, ...]) -> Mapping[str, str]:
    """``{sub_knob: gate}`` for every multi-field choice on ``axes``.

    A gated or combined axis puts its gate first in each choice; the fields
    after it are read only while the gate is on, so setting one without the
    gate re-emits the base kernel.
    """
    requires: dict[str, str] = {}
    for axis in axes:
        for choice in axis.choices:
            if len(choice) < 2:
                continue
            gate = choice[0][0]
            for name, _value in choice[1:]:
                requires.setdefault(name, gate)
    return requires


def axis_knob_names(axes: Tuple[KnobAxis, ...]) -> frozenset:
    """Every kernel-spec field some choice on ``axes`` sets."""
    return frozenset(name for axis in axes for choice in axis.choices for name, _ in choice)
