#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Seeded sampler for gauntlet spaces (RFC 0023, sections 5 and 7.3).

One sample is fully determined by its shape seed: ``sample()`` seeds a fresh
``random.Random`` with it, draws every axis (a boundary value with probability 1/2, else
a uniform value), redraws when a constraint is violated, and renders the space's case
block into a row in ``sweep.json``'s own ``{id, values, metadata}`` format (plus
``tensor_patches`` when the space declares them).

The data seed is separate. By default it is derived from the shape seed; it becomes the
case's ``metadata.seed``, which the harness uses for input fills.

An axis marked ``late`` in its space draws from a stream of its own (the shape seed, the
attempt and the axis name), so axes added to a space later do not shift the shapes that seeds
recorded earlier fix. ``sample()`` draws every other axis from the one seeded generator.
"""

import hashlib
import json
import random

SEED_LIMIT = (
    2**32
)  # data seeds are in [0, 2^32): the harness narrows them to unsigned int
MAX_SHAPE_SEED = 2**63  # shape seeds are written as int64-safe case ids
DEFAULT_MAX_ATTEMPTS = 5000
GENERATOR = "gauntlet-fuzz"
DEFAULT_PURPOSE = "probe"


class InfeasibleSpaceError(Exception):
    """No legal sample was found within the attempt budget."""


def case_id(shape_seed):
    return f"s{shape_seed}"


def derive_data_seed(shape_seed):
    """Data seed for a shape seed: a stable hash, always in ``[0, 2^32)``."""
    digest = hashlib.sha256(f"gauntlet-data-seed:{shape_seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def check_shape_seed(shape_seed):
    if (
        not isinstance(shape_seed, int)
        or isinstance(shape_seed, bool)
        or not 0 <= shape_seed < MAX_SHAPE_SEED
    ):
        raise ValueError(
            f"shape seed must be an integer in [0, 2^63), got {shape_seed!r}"
        )
    return shape_seed


def check_data_seed(data_seed):
    if (
        not isinstance(data_seed, int)
        or isinstance(data_seed, bool)
        or not 0 <= data_seed < SEED_LIMIT
    ):
        raise ValueError(
            f"data seed must be an integer in [0, 2^32), got {data_seed!r}"
        )
    return data_seed


class Sample:
    """One legal sample of a space."""

    def __init__(self, space, shape_seed, data_seed, axis_values, row, attempts):
        self.space = space
        self.shape_seed = shape_seed
        self.data_seed = data_seed
        self.axis_values = axis_values
        self.row = row
        self.attempts = attempts

    @property
    def case_id(self):
        return self.row["id"]


def sample(space, shape_seed, data_seed=None, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """The sample of ``space`` that ``shape_seed`` fixes.

    ``data_seed`` defaults to ``derive_data_seed(shape_seed)``. Raises
    ``InfeasibleSpaceError`` naming the space and every constraint reason (with how often
    each rejected a draw) when no legal sample appears in ``max_attempts`` draws.
    """
    check_shape_seed(shape_seed)
    data_seed = (
        derive_data_seed(shape_seed)
        if data_seed is None
        else check_data_seed(data_seed)
    )
    rng = random.Random(shape_seed)
    rejected = {}
    for attempt in range(1, max_attempts + 1):
        axis_values = {
            name: axis.draw(
                # A late axis has a stream per (shape seed, attempt, axis) of its own, so adding
                # one never moves a draw of the axes before it, retries included.
                random.Random(f"gauntlet-late-axis:{shape_seed}:{attempt}:{name}")
                if axis.late
                else rng
            )
            for name, axis in space.axes.items()
        }
        env = space.derive_env(axis_values)
        violated = space.first_violation(env)
        if violated is not None:
            key = (violated.rule, violated.reason)
            rejected[key] = rejected.get(key, 0) + 1
            continue
        case = space.render(env)
        return Sample(
            space,
            shape_seed,
            data_seed,
            axis_values,
            _row(space, shape_seed, data_seed, axis_values, case),
            attempt,
        )
    detail = "; ".join(
        f"'{rule}' ({reason}) rejected {n} draws"
        for (rule, reason), n in rejected.items()
    )
    raise InfeasibleSpaceError(
        f"space '{space.space_id}' is infeasible: no legal sample in {max_attempts} draws"
        f" for shape seed {shape_seed}: {detail}"
    )


def _row(space, shape_seed, data_seed, axis_values, case):
    notes = (
        f"gauntlet-fuzz sample: space {space.space_id}, shape seed {shape_seed},"
        f" axes {json.dumps(axis_values, sort_keys=True, separators=(',', ':'))}"
    )
    metadata = {"format_version": 1, "generator": GENERATOR, "notes": notes}
    metadata.update(case["metadata"])
    metadata.setdefault("purpose", DEFAULT_PURPOSE)
    metadata["seed"] = data_seed
    row = {"id": case_id(shape_seed), "values": case["values"], "metadata": metadata}
    if "tensor_patches" in case:
        row["tensor_patches"] = case["tensor_patches"]
    return row
