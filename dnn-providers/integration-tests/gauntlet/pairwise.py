#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Seeded pairwise generation over gauntlet spaces (RFC 0023, section 6).

``build(space, seed)`` returns rows (axis-value dicts, ready for ``sampler.sample_from_axes``) in
which every value of every axis meets every value of every other axis, in a legal row where both
values *matter*. Three definitions carry the whole module.

**Levels.** A ``values`` axis has its values, deduplicated in order. A ``range`` axis has its
declared boundaries plus one interior level drawn from the seed: strictly between the lowest and
the highest boundary, and equal to none. The range ends are levels only if they are declared
boundaries. A range with no room has no interior level.

**Live.** A value of axis A is *live* in a row when some other level of A gives a legal row whose
rendered case differs from the row's. The rendered case is what ``sampler.sample_from_axes``
returns, restricted to what shapes the graph: ``values``, ``tensor_patches``, ``add_tensors``,
``node_patches`` and the ``purpose``, ``inputs``, ``validator``, ``placement`` and
``placement_uid`` metadata (the id, notes, seed and generator only name the row). A value that is
not live is inert there: ``wl`` while the window is ``none``, ``heads_d128`` while ``d`` is 64,
``under`` placement at alignment 1. A tuple (a pair of values) is *covered* by a row only if both
values are live in it. Liveness is exact: every other level is tried before a value is called inert.

**Uncovered tuples are proven, never dropped.** When no row can cover a tuple, an exhaustive
search over the axes that can matter for it (the *relevant axes*) decides why:

* ``illegal``: no legal row holds both values;
* ``dead``: legal rows hold both, but in each of them at least one of the two is inert;
* ``unproven``: the search hit its cap first, and no later row covered the tuple. A space with
  many axes tied together by constraints can leave some; the tuple is never counted covered.

The relevant axes of the pair (A, B) are found by closure, reading the names of expressions with
``ast`` and expanding derived names to the axes they read. The *support* of an axis is itself plus
every axis read by a constraint that reads it and by a case-template group that reads it (a ``${}``
value, or one ``add_tensors`` / ``node_patches`` / ``tensor_patches`` entry with its ``when``):
whether the axis is live in a legal row depends on those axes alone. The relevant axes of a pair
are the support of A and of B, closed under constraints (a constraint that reads a relevant axis
joins all its axes; so does a derived value that can fail to evaluate, such as a division by an
axis). An axis outside them cannot change whether the pair is legal or live, so it is filled from
a legal base row. The search is depth first over the relevant axes in a *static plan*: an order
fixed per search shape, and for each depth the checks that become decidable there (a constraint;
a derived value that can fail; and, for an axis that must be live, necessary conditions for it:
some other level that keeps a constraint true, some derived value that reads it directly that
changes, and both together). Each check reads a few axes only, so its answer is cached by their
levels and shared by every search of the build; a prefix that fails one is refuted at once.

Construction is AETG-style: take the first uncovered tuple, fix its two axes, draw ``candidates``
completions (each fills the other axes in a random order with the level that covers the most
uncovered tuples so far), keep the legal one whose live gain is largest and that covers the start
tuple live, and credit the tuples that are live in it. No such completion: run the exhaustive search.
"""

import ast
import copy
import itertools
import json
import random
import time
from bisect import bisect_right
from collections import Counter
from typing import NamedTuple

from space import SpaceError

DEFAULT_CANDIDATES = 50
DEFAULT_EXHAUSTIVE_CAP = 200_000
# Attempts to find the legal base row that fills the axes a pair does not depend on.
BASE_ROW_ATTEMPTS = 20_000
# Completions tried around a row the exhaustive search found, to put the free axes to use.
FALLBACK_COMPLETIONS = 8
# A range this narrow has its interior candidates listed; wider, one is drawn by rejection.
INTERIOR_LIST_LIMIT = 4096

# What rejects a prefix of a search besides a constraint.
EVALUATION_FAILS = "an expression of the space fails to evaluate"
INERT = "a value is inert"

# What of a rendered case shapes the graph. The rest of the metadata is the row's name.
GRAPH_CASE_KEYS = ("values", "tensor_patches", "add_tensors", "node_patches")
GRAPH_METADATA_KEYS = ("purpose", "inputs", "validator", "placement", "placement_uid")

# The outcomes of a gauntlet run that `pair_coverage` tells apart.
PASSED_OUTCOMES = ("PASS", "UNCLAIMED_PASS")
DECLINED_OUTCOME = "DECLINED"


class Finding(NamedTuple):
    """A tuple that no row covers: ``pair`` is ``((axisA, valueA), (axisB, valueB))`` and ``detail``
    says why (the constraint that rejected most completions for ``illegal``, which of the two
    values was never live for ``dead``, how far the search got for ``unproven``)."""

    pair: tuple
    detail: str


class Pairwise:
    """The result of ``build``: ``rows`` (axis-value dicts, in generation order), ``levels``
    (axis -> level values), ``tuples_total``, the ``illegal`` / ``dead`` / ``unproven`` findings,
    ``lower_bound`` (the largest product of two axes' level counts: no row set is smaller),
    ``timings`` and the ``liveness`` the build used (reusable for ``pair_coverage``)."""

    def __init__(self, space, seed, strength):
        self.space_id = space.space_id
        self.seed = seed
        self.strength = strength
        self.rows = []
        self.levels = {}
        self.tuples_total = 0
        self.illegal = []
        self.dead = []
        self.unproven = []
        self.lower_bound = 0
        self.timings = {}
        self.liveness = None


def _same(a, b):
    return type(a) is type(b) and a == b


def _dedupe(values):
    out = []
    for v in values:
        if not any(_same(v, o) for o in out):
            out.append(v)
    return out


def _interior_level(space, axis, seed):
    bounds = sorted(set(axis.boundaries))
    if len(bounds) >= 2:
        lo, hi = bounds[0] + 1, bounds[-1] - 1
    elif bounds:
        lo, hi = axis.lo, axis.hi
    else:
        lo, hi = axis.lo + 1, axis.hi - 1
    skip = set(bounds)
    if hi < lo or (hi - lo + 1) <= sum(1 for b in skip if lo <= b <= hi):
        return None
    rng = random.Random(f"gauntlet-pairwise-level:{space.space_id}:{seed}:{axis.name}")
    if hi - lo + 1 <= INTERIOR_LIST_LIMIT:
        return rng.choice([v for v in range(lo, hi + 1) if v not in skip])
    while True:
        v = rng.randint(lo, hi)
        if v not in skip:
            return v


def levels(space, axis, seed):
    """The levels of one axis (an ``Axis`` or its name) for a pairwise seed, in a fixed order:
    a ``values`` axis's values deduplicated; a ``range`` axis's declared boundaries, then one
    interior level. A range with no boundary and no room is its low end alone."""
    if isinstance(axis, str):
        axis = space.axes[axis]
    if not axis.is_range:
        return _dedupe(axis.values)
    found = _dedupe(axis.boundaries)
    interior = _interior_level(space, axis, seed)
    if interior is not None:
        found.append(interior)
    return found or [axis.lo]


def _names_in(source, known):
    return {
        n.id
        for n in ast.walk(ast.parse(source.strip(), mode="eval"))
        if isinstance(n, ast.Name) and n.id in known
    }


def _is_subst(text):
    return isinstance(text, str) and text.startswith("${") and text.endswith("}")


def _subst_sources(node):
    """The expression of every ``${...}`` string under a JSON tree of a space's case block."""
    if _is_subst(node):
        yield node[2:-1]
    elif isinstance(node, list):
        for v in node:
            yield from _subst_sources(v)
    elif isinstance(node, dict):
        for v in node.values():
            yield from _subst_sources(v)


def _constant(node, *, nonzero=False):
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and (not nonzero or node.value != 0)
    )


def _fallible(source):
    """Can an expression raise because of the values it is given: a division or modulo by
    anything but a non-zero constant, a power with a computed exponent, or an index that is
    computed? (Operand types are taken to be right.)"""
    for n in ast.walk(ast.parse(source.strip(), mode="eval")):
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                if not _constant(n.right, nonzero=True):
                    return True
            elif isinstance(n.op, ast.Pow) and not _constant(n.right):
                return True
        elif isinstance(n, ast.Subscript) and not isinstance(n.slice, ast.Constant):
            return True
    return False


class _Dependencies:
    """Which axes the liveness of an axis and the legality of a pair can depend on."""

    def __init__(self, space):
        axes = list(space.axes)
        self.expanded = {a: frozenset([a]) for a in axes}
        # The names (axes or earlier derived values) each derived value reads directly.
        self.reads = {}
        for name, expr in space.derive.items():
            reads = _names_in(expr.source, self.expanded)
            self.reads[name] = reads
            self.expanded[name] = frozenset().union(*(self.expanded[n] for n in reads))
        self.constraints = [self._axes_of(c.expr.source) for c in space.constraints]
        # A derived value that can fail to evaluate makes a row "no row", so it acts like a
        # constraint on the axes it reads.
        self.fallible = {
            name: self.expanded[name]
            for name, expr in space.derive.items()
            if _fallible(expr.source)
        }
        self.reach = self.constraints + list(self.fallible.values())
        case = json.loads(space.path.read_text(encoding="utf-8"))["case"]
        self.groups = []
        # The names (axes or derived) the case template names.
        named_in_case = set()
        for src in _subst_sources(case):
            named_in_case |= _names_in(src, self.expanded)
        self.named_in_case = named_in_case
        # What changes in a row when one axis does: the derived names that depend on it, and the
        # names the case template reads that depend on it.
        self.dependents = {
            a: [n for n in space.derive if a in self.expanded[n]] for a in axes
        }
        self.case_inputs = {
            a: sorted(n for n in named_in_case if a in self.expanded[n]) for a in axes
        }
        for key in ("values", "metadata"):
            self.groups += [
                self._axes_of(src) for src in _subst_sources(case.get(key, {}))
            ]
        for key in ("add_tensors", "node_patches", "tensor_patches"):
            for entry in case.get(key, []):
                self.groups.append(
                    frozenset().union(
                        *(self._axes_of(src) for src in _subst_sources(entry))
                    )
                )
        self.support = {a: self._support(a, self.groups, self.reach) for a in axes}
        self.touching = {
            a: [i for i, c in enumerate(self.constraints) if a in c] for a in axes
        }

    def _axes_of(self, source):
        return frozenset().union(
            *(self.expanded[n] for n in _names_in(source, self.expanded))
        )

    @staticmethod
    def _support(axis, *families):
        support = {axis}
        for part in itertools.chain(*families):
            if axis in part:
                support |= part
        return frozenset(support)

    def close(self, axes):
        """``axes`` plus, until nothing changes, every axis of each constraint (or fallible
        derived value) that touches them."""
        closed = set(axes)
        grew = True
        while grew:
            grew = False
            for c in self.reach:
                if c & closed and not c <= closed:
                    closed |= c
                    grew = True
        return frozenset(closed)


def _rendered(case):
    metadata = case["metadata"]
    graph = [case.get(k) for k in GRAPH_CASE_KEYS]
    graph.append({k: metadata[k] for k in GRAPH_METADATA_KEYS if k in metadata})
    return json.dumps(graph, sort_keys=True, default=repr)


class Liveness:
    """Is a value live in a row (see the module docstring)? The row must be legal. A row differs
    from its alternatives only in the axis, so the answer depends on the axis's *support* alone:
    its own value and the values of the axes its constraints and case groups read. Answers are
    cached by those values, so the cost is one search per distinct neighbourhood."""

    def __init__(self, space, level_map):
        self.space = space
        self.levels = level_map
        self.names = list(space.axes)
        self.position = {n: p for p, n in enumerate(self.names)}
        self.deps = _Dependencies(space)
        self.support = self._positions(self.deps.support)
        self.touching = {
            a: [space.constraints[i] for i in idx]
            for a, idx in self.deps.touching.items()
        }
        self.level_values = [self.levels[n] for n in self.names]
        self._live = {}
        self._live_by_value = {}
        self.renders = 0
        self.reasons = {c.rule: c.reason for c in space.constraints}

    def _positions(self, by_name):
        return {
            self.position[a]: sorted(self.position[x] for x in axes)
            for a, axes in by_name.items()
        }

    def violation(self, values):
        """The rule that rejects an axis-value dict, None when it is legal."""
        try:
            c = self.space.first_violation(self.space.derive_env(values))
        except SpaceError:
            return EVALUATION_FAILS
        return None if c is None else c.rule

    def _render(self, env):
        self.renders += 1
        return _rendered(self.space.render(env))

    def _alternative(self, env, axis, level):
        """``env`` with ``axis`` at ``level``: only the derived values that depend on it are
        evaluated again. Raises ``SpaceError`` as ``derive_env`` would."""
        alt = dict(env)
        alt[axis] = level
        for name in self.deps.dependents[axis]:
            alt[name] = self.space.derive[name].evaluate(alt)
        return alt

    def _differing(self, values, axis):
        """The environments of the rows that differ from ``values`` only in ``axis`` and render a
        different case, legal or not, one at a time. Raises ``_NoRow`` if ``values`` cannot be
        rendered (a mix of values that is no row)."""
        try:
            env = self.space.derive_env(values)
        except SpaceError:
            raise _NoRow from None
        inputs = self.deps.case_inputs[axis]
        seen = [json.dumps(env[n]) for n in inputs]
        base = None
        own = values[axis]
        for level in self.levels[axis]:
            if _same(level, own):
                continue
            try:
                alt = self._alternative(env, axis, level)
            except SpaceError:
                continue  # no such row
            if all(json.dumps(alt[n]) == v for n, v in zip(inputs, seen)):
                continue  # the case template reads the same values: the same case
            try:
                if base is None:
                    base = self._render(env)
            except SpaceError:
                raise _NoRow from None
            try:
                if self._render(alt) != base:
                    yield alt
            except SpaceError:
                continue

    def _is_live(self, values, axis):
        legal = self.touching[axis]
        for alt in self._differing(values, axis):
            try:
                if all(c.expr.evaluate(alt) for c in legal):
                    return True
            except SpaceError:
                continue
        return False

    def _row_values(self, row):
        return {n: self.level_values[p][row[p]] for p, n in enumerate(self.names)}

    def live_index(self, row, pos):
        """Liveness of axis ``pos`` in a row given as one level index per axis; None if the row
        cannot be rendered (never for a legal row)."""
        key = (pos, tuple(row[p] for p in self.support[pos]))
        hit = self._live.get(key)
        if hit is None:
            try:
                hit = self._live[key] = self._is_live(
                    self._row_values(row), self.names[pos]
                )
            except _NoRow:
                return None
        return hit

    def live(self, values, axis):
        """Liveness of axis ``axis`` in a legal row given as an axis-value dict (a range axis
        need not hold a level)."""
        pos = self.position[axis]
        key = (
            pos,
            tuple(json.dumps(values[self.names[p]]) for p in self.support[pos]),
        )
        hit = self._live_by_value.get(key)
        if hit is None:
            try:
                hit = self._is_live(values, axis)
            except _NoRow:
                hit = False
            self._live_by_value[key] = hit
        return hit


class _NoRow(Exception):
    """An axis-value mix that cannot be rendered."""


class _Capped(Exception):
    """A search visited more nodes than its cap."""


class _Check:
    """A test a prefix of a search can fail once it has set the axes at ``positions``: ``fn``
    takes their level indices and returns None when they pass, else what refuted them. The
    answers are cached by those indices, for every search of the build."""

    __slots__ = ("positions", "members", "fn", "cache")

    def __init__(self, positions, fn):
        self.positions = positions
        self.members = frozenset(positions)
        self.fn = fn
        self.cache = {}


_UNSET = object()


class _Search(NamedTuple):
    kind: str  # "found", "illegal", "dead" or "unproven"
    row: tuple
    detail: str


class _Builder:
    def __init__(self, space, seed, candidates, cap):
        self.space = space
        self.seed = seed
        self.candidates = candidates
        self.cap = cap
        self.names = list(space.axes)
        self.n = len(self.names)
        self.level_map = {n: levels(space, n, seed) for n in self.names}
        self.size = [len(self.level_map[n]) for n in self.names]
        self.liveness = Liveness(space, self.level_map)
        self.base_row = None
        self._plans = {}
        self._single = {}
        self._checks = {}
        self.exhaustive_seconds = 0.0
        self.unproven_at = (
            {}
        )  # tuple -> Finding, for the tuples whose search passed the cap
        # The tuples of axes x < y occupy one block each, indexed lx * size[y] + ly.
        self.block_start = {}
        self.starts = []
        self.pairs = []
        total = 0
        for x in range(self.n):
            for y in range(x + 1, self.n):
                self.block_start[x, y] = total
                self.starts.append(total)
                self.pairs.append((x, y))
                total += self.size[x] * self.size[y]
        self.total = total
        self.uncovered = bytearray(b"\x01") * total
        # slot[x][y] = (offset, step of x's level, step of y's level) for either order.
        self.slot = [[None] * self.n for _ in range(self.n)]
        for (x, y), start in self.block_start.items():
            self.slot[x][y] = (start, self.size[y], 1)
            self.slot[y][x] = (start, 1, self.size[y])

    def values(self, row):
        return {
            n: self.liveness.level_values[p][row[p]] for p, n in enumerate(self.names)
        }

    def decode(self, index):
        k = bisect_right(self.starts, index) - 1
        x, y = self.pairs[k]
        local = index - self.starts[k]
        return x, y, local // self.size[y], local % self.size[y]

    def tuple_of(self, x, y, lx, ly):
        lv = self.liveness.level_values
        return (
            (self.names[x], copy.deepcopy(lv[x][lx])),
            (self.names[y], copy.deepcopy(lv[y][ly])),
        )

    def row_pairs(self, row):
        """The uncovered tuples a row holds, as (x, y) axis positions."""
        unc = self.uncovered
        return [
            (x, y)
            for (x, y), start in self.block_start.items()
            if unc[start + row[x] * self.size[y] + row[y]]
        ]

    def fill(self, fixed, rng):
        """A row with the axes of ``fixed`` (position -> level) set and every other axis put, in
        a random order, at the level that covers most uncovered tuples with the axes set so far.
        """
        row = [-1] * self.n
        for p, level in fixed.items():
            row[p] = level
        order = [p for p in range(self.n) if row[p] < 0]
        rng.shuffle(order)
        placed = [p for p in range(self.n) if row[p] >= 0]
        unc = self.uncovered
        for x in order:
            best, ties = -1, []
            slots = [(self.slot[x][y], row[y]) for y in placed]
            for level in range(self.size[x]):
                gain = 0
                for (start, step_x, step_y), ly in slots:
                    gain += unc[start + level * step_x + ly * step_y]
                if gain > best:
                    best, ties = gain, [level]
                elif gain == best:
                    ties.append(level)
            row[x] = ties[0] if len(ties) == 1 else rng.choice(ties)
            placed.append(x)
        return tuple(row)

    def choose(self, rows, start):
        """The legal row of ``rows`` with the most newly covered tuples that cover tuple ``start``
        (axes ``a`` and ``b``) live; None if none does."""
        live = self.liveness
        scored = []
        for k, row in enumerate(rows):
            if live.violation(self.values(row)) is None:
                pairs = self.row_pairs(row)
                scored.append((-len(pairs), k, row, pairs))
        scored.sort(key=lambda s: (s[0], s[1]))
        a, b = start
        best, best_gain = None, 0
        for neg, _k, row, pairs in scored:
            if -neg <= best_gain:
                break
            if not (live.live_index(row, a) and live.live_index(row, b)):
                continue
            gain = sum(
                1
                for x, y in pairs
                if live.live_index(row, x) and live.live_index(row, y)
            )
            if gain > best_gain:
                best, best_gain = row, gain
        return best

    def credit(self, row):
        live = self.liveness
        for x, y in self.row_pairs(row):
            if live.live_index(row, x) and live.live_index(row, y):
                self.uncovered[
                    self.block_start[x, y] + row[x] * self.size[y] + row[y]
                ] = 0
        for x, y, lx, ly in list(self.unproven_at):
            if (
                row[x] == lx
                and row[y] == ly
                and live.live_index(row, x)
                and live.live_index(row, y)
            ):
                del self.unproven_at[x, y, lx, ly]

    def legal_base_row(self):
        if self.base_row is None:
            rng = random.Random(
                f"gauntlet-pairwise-base:{self.space.space_id}:{self.seed}"
            )
            for _ in range(BASE_ROW_ATTEMPTS):
                row = tuple(rng.randrange(s) for s in self.size)
                if self.liveness.violation(self.values(row)) is None:
                    self.base_row = row
                    break
            else:
                raise ValueError(
                    f"space '{self.space.space_id}': no legal row among the levels"
                    f" in {BASE_ROW_ATTEMPTS} draws"
                )
        return self.base_row

    def relevant(self, *axes):
        """The positions of the axes that a search fixing the axes ``axes`` depends on."""
        deps = self.liveness.deps
        position = self.liveness.position
        return sorted(
            position[n]
            for n in deps.close(
                frozenset().union(*(deps.support[self.names[a]] for a in axes))
            )
        )

    def _environment(self, positions, key):
        values = self.liveness.level_values
        return {self.names[p]: values[p][i] for p, i in zip(positions, key)}

    def _positions(self, axes):
        position = self.liveness.position
        return tuple(sorted(position[a] for a in axes))

    def _chain(self, names):
        """The derived values, in declaration order, that evaluating ``names`` needs."""
        derive = self.space.derive
        reads = self.liveness.deps.reads
        needed, stack = set(), list(names)
        while stack:
            n = stack.pop()
            if n in derive and n not in needed:
                needed.add(n)
                stack.extend(reads[n])
        return [n for n in derive if n in needed]

    def _constraint_check(self, index):
        space = self.space
        deps = self.liveness.deps
        constraint = space.constraints[index]
        steps = self._chain(_names_in(constraint.rule, deps.expanded))
        positions = self._positions(deps.constraints[index])

        def fn(key):
            env = self._environment(positions, key)
            try:
                for n in steps:
                    env[n] = space.derive[n].evaluate(env)
                return None if constraint.expr.evaluate(env) else constraint.rule
            except SpaceError:
                return EVALUATION_FAILS

        return _Check(positions, fn)

    def _fallible_check(self, name):
        space = self.space
        steps = self._chain([name])
        positions = self._positions(self.liveness.deps.fallible[name])

        def fn(key):
            env = self._environment(positions, key)
            try:
                for n in steps:
                    env[n] = space.derive[n].evaluate(env)
            except SpaceError:
                return EVALUATION_FAILS
            return None

        return _Check(positions, fn)

    def _move_check(self, key):
        """What a prefix that sets the axes of a ``scope`` must still allow of axis ``pos``: some
        other level that gives a legal row (as far as the constraints in ``part`` tell) in which
        a value the case template reads changes. ``part`` is ``"full"`` (the constraints that
        read the axis and the case inputs it moves), ``"differs"`` (only that some derived value
        that reads the axis directly changes: whatever changes in a row changes first in one) or
        the index of one constraint that reads the axis (legality alone). Each is necessary for
        the axis to be live, so a prefix that fails one is refuted early; the render decides the
        rest."""
        pos, part = key
        space = self.space
        deps = self.liveness.deps
        axis = self.names[pos]
        inputs = deps.case_inputs[axis]
        level_values = self.liveness.level_values[pos]
        if len(level_values) < 2:  # no other level to move to
            return _Check((pos,), lambda key: INERT)
        direct = [n for n in space.derive if axis in deps.reads[n]]
        if part == "full":
            legal, watched = deps.touching[axis], inputs
        elif part == "differs":
            if (
                axis in deps.named_in_case
            ):  # the case names the axis itself: it always differs
                return _Check((pos,), lambda key: None)
            legal, watched = [], direct
        else:
            legal, watched = [part], []
        differing = isinstance(part, str)
        rules = [space.constraints[i].expr for i in legal]
        reads = list(watched)
        for i in legal:
            reads += _names_in(space.constraints[i].rule, deps.expanded)
        needed = self._chain(reads)
        moved = [n for n in needed if axis in deps.expanded[n]]
        scope = frozenset([axis]).union(*(deps.expanded[n] for n in reads))
        positions = self._positions(scope)
        own_at = positions.index(pos)

        def fn(key):
            env = self._environment(positions, key)
            try:
                for n in needed:
                    env[n] = space.derive[n].evaluate(env)
            except SpaceError:
                return EVALUATION_FAILS
            before = [repr(env[n]) for n in watched]
            for level, value in enumerate(level_values):
                if level == key[own_at]:
                    continue
                alt = dict(env)
                alt[axis] = value
                try:
                    for n in moved:
                        alt[n] = space.derive[n].evaluate(alt)
                    if not all(c.evaluate(alt) for c in rules):
                        continue
                except SpaceError:
                    continue
                if not differing or any(
                    repr(alt[n]) != s for n, s in zip(watched, before)
                ):
                    return None
            return INERT

        return _Check(positions, fn)

    def _check(self, kind, key):
        found = self._checks.get((kind, key))
        if found is None:
            make = {
                "constraint": self._constraint_check,
                "fallible": self._fallible_check,
                "move": self._move_check,
            }[kind]
            found = self._checks[kind, key] = make(key)
        return found

    def _plan(self, fixed, need):
        """The static plan of a search with the axes ``fixed`` set in which the axes of ``need``
        are live: ``(order, stages)``. ``order`` lists the relevant axes not fixed in the order
        the search sets them; ``stages[d]`` holds the checks that become decidable once the
        first ``d`` of them are set (each runs once, at the first node that can decide it).
        """
        plan = self._plans.get((fixed, need))
        if plan is not None:
            return plan
        deps = self.liveness.deps
        relevant = self.relevant(*fixed)
        inside = set(relevant)
        pending = [
            self._check("constraint", i) for i in range(len(self.space.constraints))
        ]
        pending += [self._check("fallible", name) for name in deps.fallible]
        for p in need:
            parts = ["full", "differs", *deps.touching[self.names[p]]]
            pending += [self._check("move", (p, part)) for part in parts]
        pending = [c for c in pending if c.members <= inside]
        assigned = set(fixed)
        stages = [[c for c in pending if c.members <= assigned]]
        pending = [c for c in pending if not c.members <= assigned]
        remaining = [p for p in relevant if p not in assigned]
        order = []
        while remaining:

            def score(p):
                grown = assigned | {p}
                done = sum(1 for c in pending if c.members <= grown)
                progress = sum(
                    1.0 / len(c.members - assigned) for c in pending if p in c.members
                )
                return (-done, -progress, self.size[p], p)

            p = min(remaining, key=score)
            remaining.remove(p)
            assigned.add(p)
            order.append(p)
            stages.append([c for c in pending if c.members <= assigned])
            pending = [c for c in pending if not c.members <= assigned]
        plan = self._plans[fixed, need] = (order, stages)
        return plan

    def _dfs(self, fixed, need):
        """Depth-first search for a legal row with the axes of ``fixed`` (``(axis, level)``
        pairs) at their levels in which every axis of ``need`` (a tuple) is live, over the
        relevant axes (the others are taken from a legal base row), following the static plan: a
        node that fails a check of its stage is refuted, whatever the axes still to be set.
        Returns the row or None, what refuted the nodes, and the nodes visited; raises
        ``_Capped`` past ``cap`` nodes."""
        live = self.liveness
        order, stages = self._plan(tuple(p for p, _ in fixed), need)
        base_row = self.legal_base_row()
        row = list(base_row)
        for p, level in fixed:
            row[p] = level
        rejected = Counter()
        nodes = 0

        def passes(stage):
            for check in stage:
                key = tuple([row[p] for p in check.positions])
                verdict = check.cache.get(key, _UNSET)
                if verdict is _UNSET:
                    verdict = check.cache[key] = check.fn(key)
                if verdict is not None:
                    rejected[verdict] += 1
                    return False
            return True

        def descend(depth):
            nonlocal nodes
            if depth == len(order):
                full = tuple(row)
                if live.violation(self.values(full)) is None and all(
                    live.live_index(full, p) for p in need
                ):
                    return full
                return None
            p = order[depth]
            stage = stages[depth + 1]
            for level in range(self.size[p]):
                nodes += 1
                if nodes > self.cap:
                    raise _Capped
                row[p] = level
                if passes(stage):
                    found = descend(depth + 1)
                    if found is not None:
                        return found
            row[p] = base_row[p]
            return None

        if not passes(stages[0]):
            return None, rejected, nodes
        return descend(0), rejected, nodes

    def search(self, a, la, b, lb):
        """Exhaustive search for a legal row with axis ``a`` at level ``la`` and ``b`` at ``lb``
        where both are live: a row, or the proof that none exists."""
        started = time.perf_counter()
        try:
            return self._search(a, la, b, lb)
        finally:
            self.exhaustive_seconds += time.perf_counter() - started

    def _possible(self, pos, level):
        """Is axis ``pos`` at ``level`` live in some legal row? (Cached; ``_Capped`` if the
        search for it passed the cap.)"""
        found = self._single.get((pos, level), _UNSET)
        if found is _UNSET:
            try:
                found = self._dfs(((pos, level),), (pos,))[0] is not None
            except _Capped:
                found = None
            self._single[pos, level] = found
        if found is None:
            raise _Capped
        return found

    def _search(self, a, la, b, lb):
        capped = _Search(
            "unproven", None, f"the search passed its cap of {self.cap} nodes"
        )
        both = ((a, la), (b, lb))
        try:
            row, rejected, nodes = self._dfs(both, ())
            if row is None:
                if not rejected:
                    return _Search("illegal", None, "no legal row")
                rule, count = rejected.most_common(1)[0]
                reason = self.liveness.reasons.get(rule)
                return _Search(
                    "illegal",
                    None,
                    f"'{rule}' rejected {count} partial rows of {nodes + 1}"
                    + (f" ({reason})" if reason else ""),
                )
            seen_a = self._possible(a, la)
            seen_b = self._possible(b, lb)
            if seen_a and seen_b:
                row = self._dfs(both, (a, b))[0]
                if row is not None:
                    return _Search("found", row, "")
        except _Capped:
            return capped
        if not seen_a and not seen_b:
            detail = "neither value is ever live"
        elif not seen_a:
            detail = f"{self.names[a]} is never live"
        elif not seen_b:
            detail = f"{self.names[b]} is never live"
        else:
            detail = "never both live in one row"
        return _Search("dead", None, detail)

    def next_row(self, start_tuple, row_number):
        """A row covering the first uncovered tuple live, or the search verdict that none exists."""
        x, y, lx, ly = start_tuple
        space_id = self.space.space_id
        fixed = {x: lx, y: ly}
        rows = [
            self.fill(
                fixed,
                random.Random(
                    f"gauntlet-pairwise:{space_id}:{self.seed}:{row_number}:{k}"
                ),
            )
            for k in range(self.candidates)
        ]
        best = self.choose(rows, (x, y))
        if best is not None:
            return best, None
        verdict = self.search(x, lx, y, ly)
        if verdict.kind != "found":
            return None, verdict
        fixed = {p: verdict.row[p] for p in self.relevant(x, y)}
        rows = [verdict.row] + [
            self.fill(
                fixed,
                random.Random(
                    f"gauntlet-pairwise:{space_id}:{self.seed}:{row_number}:fallback{k}"
                ),
            )
            for k in range(FALLBACK_COMPLETIONS)
        ]
        return self.choose(rows, (x, y)), None


def build(
    space,
    seed,
    strength=2,
    candidates=DEFAULT_CANDIDATES,
    cap=DEFAULT_EXHAUSTIVE_CAP,
):
    """A seeded pairwise covering set of ``space`` (see the module docstring).

    ``candidates`` is how many completions each row is chosen from; ``cap`` bounds the
    combinations the exhaustive search enumerates per tuple. Only strength 2 is supported.
    Deterministic in ``(space, seed, candidates, cap)``."""
    if strength != 2:
        raise ValueError(f"only pairwise (strength 2) is supported, got {strength}")
    started = time.perf_counter()
    b = _Builder(space, seed, candidates, cap)
    result = Pairwise(space, seed, strength)
    result.levels = {n: copy.deepcopy(b.level_map[n]) for n in b.names}
    result.tuples_total = b.total
    result.lower_bound = max(
        (b.size[x] * b.size[y] for x, y in b.pairs), default=max(b.size, default=0)
    )
    result.liveness = b.liveness
    kinds = {"illegal": result.illegal, "dead": result.dead}
    fallbacks = 0
    cursor = 0
    while True:
        while cursor < b.total and not b.uncovered[cursor]:
            cursor += 1
        if cursor == b.total:
            break
        x, y, lx, ly = b.decode(cursor)
        before = b.exhaustive_seconds
        row, verdict = b.next_row((x, y, lx, ly), len(result.rows))
        fallbacks += b.exhaustive_seconds > before
        if row is not None:
            b.credit(row)
            result.rows.append(copy.deepcopy(b.values(row)))
        elif verdict.kind == "unproven":
            # a row built for a later tuple may still cover it; `credit` then drops it
            b.unproven_at[x, y, lx, ly] = Finding(
                b.tuple_of(x, y, lx, ly), verdict.detail
            )
            b.uncovered[cursor] = 0
        else:
            kinds[verdict.kind].append(
                Finding(b.tuple_of(x, y, lx, ly), verdict.detail)
            )
            b.uncovered[cursor] = 0
    result.unproven = list(b.unproven_at.values())
    result.timings = {
        "total_s": time.perf_counter() - started,
        "exhaustive_s": b.exhaustive_seconds,
        "exhaustive_searches": fallbacks,
        "renders": b.liveness.renders,
    }
    return result


def pair_coverage(space, records, level_map, liveness=None):
    """How much of the pair universe a run drew and passed.

    ``records`` is an iterable of ``(axis_values, outcome)``; ``level_map`` is the run's
    ``Pairwise.levels``. A tuple is *drawn* by a record only if both its values are levels (a
    range axis holding an interior value that is not a level maps to none) and live in the
    record's row; it is *passed* if such a record's outcome is ``PASS`` or ``UNCLAIMED_PASS``,
    and *declined only* if every record that drew it was ``DECLINED``.

    Returns ``tuples`` (the whole universe), ``drawn``, ``passed``, ``declined_only``,
    ``never_passed`` (drawn but not passed) and ``not_passed`` (every tuple of the universe that
    was not passed, drawn or not, as ``((axisA, valueA), (axisB, valueB))``, in universe order).
    """
    liveness = liveness if liveness is not None else Liveness(space, level_map)
    names = list(space.axes)
    drawn, passed, declined, other = set(), set(), set(), set()
    for values, outcome in records:
        held = {}
        for p, name in enumerate(names):
            value = values[name]
            for index, level in enumerate(level_map[name]):
                if _same(level, value):
                    if liveness.live(values, name):
                        held[p] = index
                    break
        for x, y in itertools.combinations(sorted(held), 2):
            key = (x, held[x], y, held[y])
            drawn.add(key)
            if outcome in PASSED_OUTCOMES:
                passed.add(key)
            elif outcome == DECLINED_OUTCOME:
                declined.add(key)
            else:
                other.add(key)
    not_passed = []
    total = 0
    for x, y in itertools.combinations(range(len(names)), 2):
        for lx, vx in enumerate(level_map[names[x]]):
            for ly, vy in enumerate(level_map[names[y]]):
                total += 1
                if (x, lx, y, ly) not in passed:
                    not_passed.append(
                        ((names[x], copy.deepcopy(vx)), (names[y], copy.deepcopy(vy)))
                    )
    return {
        "tuples": total,
        "drawn": len(drawn),
        "passed": len(passed),
        "declined_only": len(declined - passed - other),
        "never_passed": len(drawn - passed),
        "not_passed": not_passed,
    }
