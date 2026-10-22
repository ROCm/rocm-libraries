# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Enumerating a knob space: sweep levels, the pruned depth-first walk, the
random sampler, and the one-knob-at-a-time production walk.

Family-neutral: callers pass the axes and an ``is_valid`` predicate.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, Mapping, Optional, Tuple

from .axes import KnobAxis

SWEEP_LEVELS: Tuple[str, ...] = ("production", "full")

_SWEEP_LEVEL: contextvars.ContextVar[str] = contextvars.ContextVar(
    "rocke_sweep_level", default="production"
)


def current_sweep_level() -> str:
    return _SWEEP_LEVEL.get()


def configure_sweep(level: str, tuning_sample: int = 0) -> int:
    """Select the sweep level and return the sample count for ``iter_combos``.

    ``production`` walks each candidate's curated set and ignores
    ``tuning_sample``. ``full`` samples ``tuning_sample`` specs per candidate
    (0 walks the full stream, which can be millions of specs).
    """
    count = sample_count(level, tuning_sample)
    _SWEEP_LEVEL.set(level)
    return count


def sample_count(level: str, tuning_sample: int) -> int:
    """The per-candidate sample count a sweep at ``level`` uses."""
    if level not in SWEEP_LEVELS:
        raise ValueError(f"sweep level must be one of {SWEEP_LEVELS}, got {level!r}")
    if int(tuning_sample) < 0:
        raise ValueError(f"tuning_sample must be >= 0, got {tuning_sample!r}")
    return 0 if level == "production" else int(tuning_sample)


@contextmanager
def sweep_level(level: str):
    """Run the body at ``level``; the caller's level is restored on exit."""
    sample_count(level, 0)
    token = _SWEEP_LEVEL.set(level)
    try:
        yield
    finally:
        _SWEEP_LEVEL.reset(token)


def iter_at_level(level: str, make: Callable[[], Iterable]) -> Iterator:
    """Iterate ``make()`` with the sweep level set only while it runs.

    The level is active inside each ``next()`` and never while this generator
    is suspended, so a driver that interleaves two sweeps at different levels
    cannot leak one into the other. Candidates read the level when they create
    a stream, which always happens inside a ``next()``.
    """
    iterator = None
    while True:
        with sweep_level(level):
            if iterator is None:
                iterator = iter(make())
            try:
                item = next(iterator)
            except StopIteration:
                return
        yield item


def _enabler_count(axes: Tuple[KnobAxis, ...]) -> int:
    n = next((i for i, a in enumerate(axes) if not a.enabler), len(axes))
    if any(a.enabler for a in axes[n:]):
        raise ValueError("enabler axes must lead the axis order")
    return n


def iter_knob_sets(
    axes: Tuple[KnobAxis, ...],
    root: Mapping[str, object],
    is_valid,
) -> Iterable[dict]:
    """Every legal assignment over ``axes`` on top of ``root``, depth first.

    ``is_valid`` sees the assignment so far with undecided axes at their
    defaults. Every "requires" relation in the kernel validators points at an
    earlier axis, so a failing prefix cannot be repaired later and its subtree
    is pruned. Only while leading ``enabler`` axes are undecided is an invalid
    prefix kept, because those knobs can make the setting itself legal.
    """
    n_enablers = _enabler_count(axes)

    def walk(i: int, knobs: dict, valid: bool):
        if i == len(axes):
            if valid:
                yield knobs
            return
        for choice in axes[i].choices:
            if choice:
                child = dict(knobs)
                child.update(choice)
                child_valid = is_valid(child)
            else:
                child, child_valid = knobs, valid
            if child_valid or i < n_enablers:
                yield from walk(i + 1, child, child_valid)

    start = dict(root)
    yield from walk(0, start, is_valid(start))


def random_knob_set(axes, root, is_valid, rng) -> Optional[dict]:
    """One random walk down the pruned axis tree; ``None`` on a dead end.

    At every axis the walk picks uniformly among the choices the validator
    accepts, so every legal assignment is reachable. It is not uniform over
    the whole legal set, which would need subtree sizes the full walk is too
    slow to count; it does sample each knob value at a useful rate.
    """
    n_enablers = _enabler_count(axes)
    knobs, valid = dict(root), is_valid(root)
    for i, axis in enumerate(axes):
        for choice in rng.sample(axis.choices, len(axis.choices)):
            if choice:
                child = dict(knobs)
                child.update(choice)
                child_valid = is_valid(child)
            else:
                child, child_valid = knobs, valid
            if child_valid or i < n_enablers:
                knobs, valid = child, child_valid
                break
    return knobs if valid else None


def one_knob_at_a_time(axes: Tuple[KnobAxis, ...], is_valid) -> Iterable[dict]:
    """The base, then every non-default choice of each axis on its own.

    A choice that is only legal once an earlier knob is set is paired with the
    first earlier single choice that makes it legal, so every knob still
    appears in the production set.
    """
    yield {}
    for i, axis in enumerate(axes):
        for choice in axis.choices:
            if not choice:
                continue
            if is_valid(dict(choice)):
                yield dict(choice)
                continue
            prerequisite = next(
                (
                    {**dict(earlier), **dict(choice)}
                    for prior in axes[:i]
                    for earlier in prior.choices
                    if earlier and is_valid({**dict(earlier), **dict(choice)})
                ),
                None,
            )
            if prerequisite is not None:
                yield prerequisite
