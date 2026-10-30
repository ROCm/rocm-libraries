#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Seeded sampler for gauntlet spaces (RFC 0023, sections 5 and 7.3).

One sample is fully determined by its shape seed: ``sample()`` seeds a fresh
``random.Random`` with it, draws every axis (a boundary value with probability 1/2, else
a uniform value), redraws when a constraint is violated, and renders the space's case
block into a row in ``sweep.json``'s own ``{id, values, metadata}`` format (plus
``add_tensors``, ``node_patches`` and ``tensor_patches`` when the space declares them).

The data seed is separate. By default it is derived from the shape seed; it becomes the
case's ``metadata.seed``, which the harness uses for input fills.

An axis marked ``late`` in its space draws from a stream of its own (the shape seed, the
attempt and the axis name), so axes added to a space later do not shift the shapes that seeds
recorded earlier fix. ``sample()`` draws every other axis from the one seeded generator.

``sample_from_axes()`` is the other way in: the axis values are given, not drawn, and so are the
data seed and the case id. It is what ``run_gauntlet_fuzz.py --row`` runs. It re-checks the
space's constraints, so a row that no sample could be raises ``InvalidRowError``.

``shrink()`` searches for simpler axis values that keep a failure: one axis at a time, a ``range``
axis binary-searched toward its low end (its declared boundaries below the value first), a
``values`` axis moved to an earlier entry of its list (list order is "simpler first"). It renders
each candidate with ``sample_from_axes`` and asks an oracle (the driver's candidate run) whether
the failure stayed; a candidate the space does not allow costs the oracle nothing.
"""

import copy
import hashlib
import json
import random
import re

# A range-axis candidate that violates a constraint is replaced by the nearest legal value; this
# bounds how far the search looks before it gives the axis up.
SHRINK_SCAN_LIMIT = 1024

SEED_LIMIT = (
    2**32
)  # data seeds are in [0, 2^32): the harness narrows them to unsigned int
MAX_SHAPE_SEED = 2**63  # shape seeds are written as int64-safe case ids
DEFAULT_MAX_ATTEMPTS = 5000
GENERATOR = "gauntlet-fuzz"
DEFAULT_PURPOSE = "probe"
DEFAULT_ROW_CASE_ID = "row"
_CASE_ID_RE = re.compile(r"[A-Za-z0-9_]+")


class InfeasibleSpaceError(Exception):
    """No legal sample was found within the attempt budget."""


class InvalidRowError(ValueError):
    """Explicit axis values that are not a legal sample of the space."""


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
    """One legal sample of a space. ``shape_seed`` is None for one built from explicit axis
    values (``sample_from_axes``)."""

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
            _row(
                space,
                case_id(shape_seed),
                f"shape seed {shape_seed}",
                data_seed,
                axis_values,
                case,
            ),
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


def _check_axis_value(axis, value):
    """``InvalidRowError`` text when ``value`` is not one the axis can draw, else None."""
    if not axis.is_range:
        if not any(type(value) is type(v) and value == v for v in axis.values):
            return f"{value!r} is not one of {axis.values}"
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        return f"{value!r} is not an integer"
    if not axis.lo <= value <= axis.hi:
        return f"{value} is outside the range [{axis.lo}, {axis.hi}]"
    return None


def sample_from_axes(space, axis_values, data_seed, row_case_id=DEFAULT_ROW_CASE_ID):
    """The sample of ``space`` with these axis values, as ``sample()`` would render it.

    ``axis_values`` names every axis of the space, late ones included, and nothing else; each
    value must be one the axis can draw. The constraints are checked again on the derived
    environment, so a row that violates one is an error that names the rule and its reason.
    ``data_seed`` becomes ``metadata.seed`` and ``row_case_id`` is the row's id, which fixes the
    GTest name. Raises ``InvalidRowError`` for any of these; ``ValueError`` for a bad data seed.
    """
    check_data_seed(data_seed)
    if not isinstance(row_case_id, str) or not _CASE_ID_RE.fullmatch(row_case_id):
        raise InvalidRowError(
            f"case id must be a non-empty string of [A-Za-z0-9_], got {row_case_id!r}"
        )
    if not isinstance(axis_values, dict):
        raise InvalidRowError(f"axes must be an object, got {axis_values!r}")
    missing = [name for name in space.axes if name not in axis_values]
    unknown = sorted(name for name in axis_values if name not in space.axes)
    if missing or unknown:
        detail = "; ".join(
            part
            for part in (
                f"missing axes {missing}" if missing else "",
                f"unknown axes {unknown}" if unknown else "",
            )
            if part
        )
        raise InvalidRowError(f"space '{space.space_id}': {detail}")
    for name, axis in space.axes.items():
        problem = _check_axis_value(axis, axis_values[name])
        if problem is not None:
            raise InvalidRowError(f"space '{space.space_id}': axis '{name}': {problem}")
    axis_values = {name: copy.deepcopy(axis_values[name]) for name in space.axes}
    env = space.derive_env(axis_values)
    violated = space.first_violation(env)
    if violated is not None:
        raise InvalidRowError(
            f"space '{space.space_id}': the row violates '{violated.rule}'"
            f" ({violated.reason})"
        )
    case = space.render(env)
    return Sample(
        space,
        None,
        data_seed,
        axis_values,
        _row(space, row_case_id, "explicit row", data_seed, axis_values, case),
        1,
    )


def _row(space, row_id, origin, data_seed, axis_values, case):
    notes = (
        f"gauntlet-fuzz sample: space {space.space_id}, {origin},"
        f" axes {json.dumps(axis_values, sort_keys=True, separators=(',', ':'))}"
    )
    metadata = {"format_version": 1, "generator": GENERATOR, "notes": notes}
    metadata.update(case["metadata"])
    metadata.setdefault("purpose", DEFAULT_PURPOSE)
    metadata["seed"] = data_seed
    row = {"id": row_id, "values": case["values"], "metadata": metadata}
    for key in ("add_tensors", "node_patches", "tensor_patches"):
        if key in case:
            row[key] = case[key]
    return row


class ShrinkStop(Exception):
    """Raised by a shrink oracle that wants no more candidates (budget spent, run stopping)."""


def _same_value(a, b):
    return type(a) is type(b) and a == b


def shrink(space, axis_values, data_seed, row_case_id, oracle):
    """Simpler axis values for which ``oracle`` still answers true.

    ``oracle(sample)`` gets a legal ``Sample`` rendered from candidate axis values (same data
    seed and case id as the original) and answers whether the failure it stands for stayed; it
    may raise ``ShrinkStop`` to end the search. A candidate that violates a constraint is never
    shown to the oracle, and one already answered for the same axis values is not asked again.

    Axes are visited in the space's order, one at a time, with the others held at their current
    values, until no axis can move. A ``values`` axis tries the entries before its current one,
    first entry first. A ``range`` axis tries its low end and the declared boundaries below the
    value, then binary-searches what lies between the highest value refused and the lowest
    accepted (a value the constraints forbid gives way to the nearest legal one). Returns
    ``(axis_values, accepted_steps)``: the original when nothing could move.
    """
    current = copy.deepcopy(axis_values)
    steps = 0
    answers = {}

    def attempt(name, value):
        """The oracle's answer for ``value`` of axis ``name`` (True or False); None when the
        candidate is not a legal sample."""
        candidate = dict(current)
        candidate[name] = value
        key = json.dumps(candidate, sort_keys=True)
        if key in answers:
            return answers[key]
        try:
            rendered = sample_from_axes(space, candidate, data_seed, row_case_id)
        except InvalidRowError:
            answers[key] = None
            return None
        answers[key] = bool(oracle(rendered))
        return answers[key]

    def accept(name, value):
        nonlocal steps
        current[name] = copy.deepcopy(value)
        steps += 1

    def shrink_values(name, axis):
        tried = []
        for earlier in axis.values:
            if _same_value(earlier, current[name]):
                break  # from the current entry on, nothing is simpler
            if any(_same_value(earlier, t) for t in tried):
                continue
            tried.append(earlier)
            if attempt(name, earlier):
                accept(name, earlier)
                return True
        return False

    def shrink_range(name, axis):
        moved = False
        lower = axis.lo - 1  # every value up to here is refused (or below the range)
        upper = current[name]  # the lowest value known to keep the failure
        for probe in sorted({axis.lo, *axis.boundaries}):
            if not axis.lo <= probe < upper:
                continue
            answer = attempt(name, probe)
            if answer:
                accept(name, probe)
                moved = True
                upper = probe
                break
            if answer is False:
                lower = probe
        while upper - lower > 1:
            mid = (lower + upper) // 2
            found = None
            for offset in range(min(upper - lower, SHRINK_SCAN_LIMIT)):
                for candidate in (mid + offset, mid - offset) if offset else (mid,):
                    if lower < candidate < upper:
                        answer = attempt(name, candidate)
                        if answer is not None:
                            found = (candidate, answer)
                            break
                if found is not None:
                    break
            if found is None:
                break  # no legal value is left between the two
            candidate, answer = found
            if answer:
                accept(name, candidate)
                moved = True
                upper = candidate
            else:
                lower = candidate
        return moved

    # An axis is settled once it was searched with the other axes where they are now; one that
    # moves unsettles all the others, and the search ends when every axis is settled.
    settled = set()
    try:
        while len(settled) < len(space.axes):
            for name, axis in space.axes.items():
                if name in settled:
                    continue
                moved = (shrink_range if axis.is_range else shrink_values)(name, axis)
                if moved:
                    settled = {name}
                else:
                    settled.add(name)
    except ShrinkStop:
        pass
    return current, steps
