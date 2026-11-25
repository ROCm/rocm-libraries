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
* ``budget_exceeded``: not settled at all. A build or repair given a deadline (``deadline``, an
  absolute reading of ``clock``) ends at it, checked at every node of the search and between
  tuples, and every tuple still uncovered goes here: it neither has a row nor a proof. These count
  as not covered (unlike ``unproven`` they are not "decided as far as the search could"), the rows
  built so far are kept, and ``generation_complete`` is false. ``progress`` is called with a
  record (space, phase, tuples settled and total, rows, elapsed) at most every
  ``progress_interval`` seconds, also from inside a long search.

The search that decides this is exact and has three parts (``_Under``, ``_Builder._problem``,
``_Builder._solve``).

*What the expressions read.* The pair fixes two axes, and ``_Under`` evaluates the space's
expressions with them fixed to say which axes each one *effectively* reads: the branch of
``d3 if ndim == 5 else 1`` that a fixed ``ndim`` rules out is not read, nor are the operands of an
``and`` / ``or`` after the one that ends it, nor the body of an ``add_tensors`` / ``node_patches``
entry whose ``when`` is false. With ``ndim`` at 4 nothing reads ``win_d``, so it is inert and every
pair that holds it is dead at once. The *relevant axes* are the axes that the constraints (and
derived values that can fail to evaluate, such as a division by an axis) which read a fixed axis
effectively read, those that the liveness checks below read, and, by closure, those of every such
expression that reads a relevant axis. Every other axis cannot change whether the pair is legal or
live, so it is filled from a legal base row.

*Checks.* A constraint, a fallible derived value, and the liveness of each axis that must be live
are each one check, reading a few axes; its answer is cached by their levels, for every search of
the build. An axis is live when another of its levels gives a legal row (the constraints and derived
values that read it hold) in which one *group* of the case renders differently: a case differs
exactly where one of its groups does (``_template_groups``: each ``${}`` of ``values``, of the graph
keys of ``metadata`` and of ``tensor_patches``, and each ``add_tensors`` / ``node_patches`` entry as
a whole, present only while its ``when`` holds). Two cheaper checks, both necessary for liveness,
refute a prefix early: the case must still be able to read the axis once the axes that decide what
is read (the tests of conditionals and guards) are set, and some other level must keep a constraint
true and change a value the case reads.

*Search.* Depth first with forward checking: whenever every axis of a check but one is set, only
the levels of that one that pass it stay in its domain (one level is set at once and propagates; none
cuts the branch), and the axis with the smallest domain is set next. After the pair is set and
propagated, the axes left split into the connected components of the checks that read two of them.
Components share no check, so a row exists iff every component has a completion, and each is
searched alone. A liveness check reads every axis its groups read, so it joins them; where the
case reads all axes through the axis that must be live, they stay one component. A cap on the nodes
visited (``DEFAULT_EXHAUSTIVE_CAP``) stays as the fallback: past it the tuple is ``unproven``.

Construction is AETG-style: take the first uncovered tuple, fix its two axes, draw ``candidates``
completions (each fills the other axes in a random order with the level that covers the most
uncovered tuples so far), keep the legal one whose live gain is largest and that covers the start
tuple live, and credit the tuples that are live in it. No such completion: run the exhaustive search.

A run that declines many rows leaves tuples no passing row ever drew. ``repair_targets`` lists the
tuples whose every drawing record was ``DECLINED`` or ``UNEXERCISED`` and the levels that some
``PASS`` record held; ``repair`` builds new rows for them (the same construction and liveness,
only the targets left to cover) from those levels alone. A target with a value that never passed
is not retried.
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

from space import SpaceError, _Expr, _Subst


def _json_encoder(**options):
    """A ``dumps`` with the given ``json.dumps`` options whose encoder is built once: the
    search encodes millions of small values, and ``json.dumps`` with options builds an encoder
    (and the C encoder behind it) at every call. The text is ``json.dumps(value, **options)``'s.
    """
    encoder = json.JSONEncoder(**options)
    try:
        from json.encoder import c_make_encoder, encode_basestring_ascii

        fast = c_make_encoder(
            None,  # no circular-reference check: the values are rendered JSON, never cyclic
            encoder.default,
            encode_basestring_ascii,
            encoder.indent,
            encoder.key_separator,
            encoder.item_separator,
            encoder.sort_keys,
            encoder.skipkeys,
            encoder.allow_nan,
        )
    except (ImportError, TypeError):
        # no C encoder (or another signature): encode() is public
        return encoder.encode
    return lambda value: "".join(fast(value, 0))


_dumps = _json_encoder()
_dumps_sorted = _json_encoder(sort_keys=True, default=repr)

DEFAULT_CANDIDATES = 50
DEFAULT_EXHAUSTIVE_CAP = 200_000
# Attempts to find the legal base row that fills the axes a pair does not depend on.
BASE_ROW_ATTEMPTS = 20_000
# Completions tried around a row the exhaustive search found, to put the free axes to use.
FALLBACK_COMPLETIONS = 8
# A range this narrow has its interior candidates listed; wider, one is drawn by rejection.
INTERIOR_LIST_LIMIT = 4096
# Seconds between two progress records of a build or repair.
DEFAULT_PROGRESS_INTERVAL = 10.0

# What rejects a prefix of a search besides a constraint.
EVALUATION_FAILS = "an expression of the space fails to evaluate"
INERT = "a value is inert"
NO_LEVEL = "no level of an axis is left"

# What of a rendered case shapes the graph. The rest of the metadata is the row's name.
GRAPH_CASE_KEYS = ("values", "tensor_patches", "add_tensors", "node_patches")
GRAPH_METADATA_KEYS = ("purpose", "inputs", "validator", "placement", "placement_uid")

# The outcomes of a gauntlet run that `pair_coverage` tells apart.
PASSED_OUTCOMES = ("PASS", "UNCLAIMED_PASS")
DECLINED_OUTCOME = "DECLINED"
UNEXERCISED_OUTCOME = "UNEXERCISED"


class Finding(NamedTuple):
    """A tuple that no row covers: ``pair`` is ``((axisA, valueA), (axisB, valueB))`` and ``detail``
    says why (the constraint that rejected most completions for ``illegal``, which of the two
    values was never live for ``dead``, how far the search got for ``unproven``)."""

    pair: tuple
    detail: str


class Pairwise:
    """The result of ``build``: ``rows`` (axis-value dicts, in generation order), ``levels``
    (axis -> level values), ``tuples_total``, the ``illegal`` / ``dead`` / ``unproven`` findings,
    the ``budget_exceeded`` findings (tuples left unsettled when the deadline passed) and
    ``generation_complete`` (false then), ``lower_bound`` (the largest product of two axes' level
    counts: no row set is smaller), ``timings`` and the ``liveness`` the build used (reusable for
    ``pair_coverage``)."""

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
        self.budget_exceeded = []
        self.generation_complete = True
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


def _rendered(case):
    metadata = case["metadata"]
    graph = [case.get(k) for k in GRAPH_CASE_KEYS]
    graph.append({k: metadata[k] for k in GRAPH_METADATA_KEYS if k in metadata})
    return _dumps_sorted(graph)


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
        seen = [_dumps(env[n]) for n in inputs]
        base = None
        own = values[axis]
        for level in self.levels[axis]:
            if _same(level, own):
                continue
            try:
                alt = self._alternative(env, axis, level)
            except SpaceError:
                continue  # no such row
            if all(_dumps(alt[n]) == v for n, v in zip(inputs, seen)):
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
            tuple(_dumps(values[self.names[p]]) for p in self.support[pos]),
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


class _Budget(Exception):
    """The deadline of a build or repair passed."""


class _Unread(Exception):
    """An expression read a name its dependency analysis said it would not read."""


class _LazyEnv(dict):
    """Axis values, with each derived value computed the first time it is read. A name that is
    neither is an ``_Unread``: the search evaluates an expression with only the axes it
    depends on, so a read outside them is a bug of the analysis, never a value to guess.
    """

    def __init__(self, derive, values):
        super().__init__(values)
        self._derive = derive

    def __missing__(self, name):
        expr = self._derive.get(name)
        if expr is None:
            raise _Unread(name)
        value = self[name] = expr.evaluate(self)
        return value


class _Group(NamedTuple):
    """One part of a rendered case that stands or falls with its values: a ``${...}`` of
    ``values``, of a graph key of ``metadata`` or of ``tensor_patches``, or one ``add_tensors`` /
    ``node_patches`` entry (all the ``${...}`` of its body, present only while its ``when``
    holds; ``guard`` is that ``when``, None for the others)."""

    guard: object
    parts: tuple


def _compiled_substitutions(tree):
    """The ``${...}`` of a compiled case block (``Space.case_values`` and the like)."""
    if isinstance(tree, _Subst):
        yield tree
    elif isinstance(tree, list):
        for v in tree:
            yield from _compiled_substitutions(v)
    elif isinstance(tree, dict):
        for v in tree.values():
            yield from _compiled_substitutions(v)


def _template_groups(space):
    """The parts of a space's case that decide whether two rendered cases differ: exactly what
    ``_rendered`` compares (the ``values``, the added tensors, the node and tensor patches and the
    ``GRAPH_METADATA_KEYS`` of the metadata)."""
    groups = [_Group(None, (s,)) for s in _compiled_substitutions(space.case_values)]
    for key, tree in space.case_metadata.items():
        if key in GRAPH_METADATA_KEYS:
            groups += [_Group(None, (s,)) for s in _compiled_substitutions(tree)]
    for entries in (space.case_add_tensors, space.case_node_patches):
        for entry in entries or ():
            body = {k: v for k, v in entry.items() if k != "when"}
            groups.append(
                _Group(entry.get("when"), tuple(_compiled_substitutions(body)))
            )
    groups += [
        _Group(None, (s,)) for s in _compiled_substitutions(space.case_tensor_patches)
    ]
    return groups


def _group_value(group, env):
    """What a group renders to in ``env`` as text, None while its entry is left out."""
    guard = group.guard
    if isinstance(guard, _Subst):
        if not guard.expr.evaluate(env):
            return None
    elif guard is not None and not guard:
        return None
    return _dumps_sorted([part.expr.evaluate(env) for part in group.parts])


_UNKNOWN = object()
_NO_AXES = frozenset()


def _parse(source):
    return ast.parse(source.strip(), mode="eval").body


class _KnownEnv(dict):
    """The environment ``_Under`` evaluates a fully known subexpression in."""

    def __init__(self, under):
        super().__init__()
        self._under = under

    def __missing__(self, name):
        value = self._under.value_of(name)
        if value is _UNKNOWN:
            raise _Unread(name)
        self[name] = value
        return value


class _Partial:
    """The expressions of a space as syntax trees, for ``_Under``: derived values, constraints and
    the groups of the case (see ``_template_groups``)."""

    def __init__(self, space, expanded):
        self.axes = frozenset(space.axes)
        self.evaluator = _Expr("0", space.space_id, "dependency analysis", ())
        self.derive_nodes = {n: _parse(e.source) for n, e in space.derive.items()}
        self.constraint_nodes = [_parse(c.rule) for c in space.constraints]
        self.groups = _template_groups(space)
        self.guard_nodes = [
            _parse(g.guard.expr.source) if isinstance(g.guard, _Subst) else None
            for g in self.groups
        ]
        self.part_nodes = [
            [_parse(p.expr.source) for p in g.parts] for g in self.groups
        ]
        # The axes some conditional, ``and`` / ``or`` operand or ``when`` that the groups of the
        # case read tests: the only ones whose values can decide what the case reads.
        deciders = set()

        def tests(node):
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and n.id in expanded:
                    deciders.update(expanded[n.id])

        roots = [n for nodes in self.part_nodes for n in nodes]
        roots += [g for g in self.guard_nodes if g is not None]
        stack = [n.id for r in roots for n in ast.walk(r) if isinstance(n, ast.Name)]
        reached = set()
        while stack:
            name = stack.pop()
            if name in self.derive_nodes and name not in reached:
                reached.add(name)
                stack.extend(
                    n.id
                    for n in ast.walk(self.derive_nodes[name])
                    if isinstance(n, ast.Name)
                )
        roots += [self.derive_nodes[name] for name in reached]
        for root in roots:
            for n in ast.walk(root):
                if isinstance(n, ast.IfExp):
                    tests(n.test)
                elif isinstance(n, ast.BoolOp):
                    for operand in n.values[:-1]:
                        tests(operand)
        for guard in self.guard_nodes:
            if guard is not None:
                tests(guard)
        self.deciders = frozenset(deciders)

    def is_read(self, under, axis):
        """Does some group of the case effectively read ``axis`` under ``under``?"""
        return any(axis in self.group_reads(under, i) for i in range(len(self.groups)))

    def group_reads(self, under, i):
        """The axes the group ``i`` effectively reads under ``under``: none while its guard is
        known to be false."""
        axes = _NO_AXES
        if self.guard_nodes[i] is not None:
            axes, value = under.visit(self.guard_nodes[i])
            if value is not _UNKNOWN and not value:
                return _NO_AXES
        elif self.groups[i].guard is not None and not self.groups[i].guard:
            return _NO_AXES
        for node in self.part_nodes[i]:
            axes = axes | under.visit(node)[0]
        return axes


class _Under:
    """What the expressions of a space read once some axes are fixed (``fixed``: axis name ->
    value). ``visit(node)`` returns ``(axes, value)``: the axes the expression *effectively*
    reads, and its value if the fixed axes decide it (else ``_UNKNOWN``). It follows the
    evaluator's own short circuits: the branch of a conditional whose test is decided, the
    operands of ``and`` / ``or`` after one that ends the evaluation, are not read; everything
    else reads all it names. So ``d3 if ndim == 5 else 1`` reads nothing once ``ndim`` is 4, and
    the expression's value cannot depend on any axis outside ``axes``, whatever the other axes
    are, which is the whole claim the search relies on."""

    def __init__(self, partial, fixed):
        self.partial = partial
        self.fixed = fixed
        self._memo = {}
        self._env = _KnownEnv(self)

    def value_of(self, name):
        if name in self.fixed:
            return self.fixed[name]
        if name in self.partial.axes:
            return _UNKNOWN
        return self.visit(self.partial.derive_nodes[name])[1]

    def reads(self, node):
        return self.visit(node)[0]

    def visit(self, node):
        found = self._memo.get(id(node))
        if found is None:
            found = self._memo[id(node)] = self._visit(node)
        return found

    def _visit(self, node):
        kind = type(node)
        if kind is ast.Constant:
            return _NO_AXES, node.value
        if kind is ast.Name:
            name = node.id
            if name in self.fixed:
                return _NO_AXES, self.fixed[name]
            if name in self.partial.axes:
                return frozenset([name]), _UNKNOWN
            return self.visit(self.partial.derive_nodes[name])
        if kind is ast.IfExp:
            tested, value = self.visit(node.test)
            if value is not _UNKNOWN:
                return self.visit(node.body if value else node.orelse)
            axes = tested | self.visit(node.body)[0] | self.visit(node.orelse)[0]
            return axes, _UNKNOWN
        if kind is ast.BoolOp:
            return self._bool_op(node)
        if kind is ast.Call:
            parts = node.args
        else:
            parts = [c for c in ast.iter_child_nodes(node) if isinstance(c, ast.expr)]
        axes, known = _NO_AXES, True
        for part in parts:
            part_axes, value = self.visit(part)
            axes = axes | part_axes
            known = known and value is not _UNKNOWN
        return axes, (self._known(node) if known else _UNKNOWN)

    def _bool_op(self, node):
        is_and = isinstance(node.op, ast.And)
        axes, unknown, value = _NO_AXES, False, None
        for operand in node.values:
            operand_axes, operand_value = self.visit(operand)
            if unknown or operand_value is _UNKNOWN:
                # an operand whose value is open: the rest is read, and may be what is returned
                unknown = True
                axes = axes | operand_axes
            elif bool(operand_value) != is_and:  # this operand ends the evaluation
                return _NO_AXES, operand_value
            else:
                value = operand_value
        return (axes, _UNKNOWN) if unknown else (_NO_AXES, value)

    def _known(self, node):
        try:
            return self.partial.evaluator._eval(node, self._env)
        except (
            _Unread,
            SpaceError,
            TypeError,
            ValueError,
            ZeroDivisionError,
            IndexError,
            OverflowError,
            KeyError,
        ):
            return _UNKNOWN


class _LiveSpec(NamedTuple):
    """What the liveness of one axis reads, given the fixed axes: ``members`` (positions of the
    free axes), the groups, constraints and fallible derived values that read the axis.
    """

    members: tuple
    groups: tuple
    rules: tuple
    derived: tuple


class _Problem(NamedTuple):
    """The checks of one search, and ``free`` / ``touch``: the axes (positions) it sets, and for
    each the checks (indices) that read it. ``searched`` is every axis the search decides the
    answer from: the fixed ones and ``free``."""

    checks: list
    free: tuple
    touch: dict
    searched: tuple


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
    searched: tuple = ()  # the axes (positions) a "found" row was decided on


class _Builder:
    def __init__(
        self,
        space,
        seed,
        candidates,
        cap,
        liveness=None,
        allowed=None,
        targets=None,
        avoid=None,
        seed_prefix=None,
        witness=None,
        deadline=None,
        clock=time.monotonic,
        progress=None,
        progress_interval=DEFAULT_PROGRESS_INTERVAL,
        phase="build",
    ):
        """``allowed`` (axis -> level values) restricts every row to those levels, ``targets``
        (tuples as ``((axisA, valueA), (axisB, valueB))``) makes those the only tuples to cover,
        ``avoid`` lists tuples a row should not hold, ``witness`` is a legal row (axis -> value)
        to fill the axes a pair does not depend on from; ``liveness`` shares the answers of an
        earlier build. These are what ``repair`` sets.
        ``deadline`` (an absolute reading of ``clock``) ends the build: ``tick`` raises
        ``_Budget`` at every node of a search and between tuples once it has passed.
        ``progress(record)`` is called by ``tick`` at most every ``progress_interval`` seconds.
        """
        self.space = space
        self.seed = seed
        self.candidates = candidates
        self.cap = cap
        self.seed_prefix = seed_prefix or f"gauntlet-pairwise:{space.space_id}:{seed}"
        self.witness = witness
        self.deadline = deadline
        self.clock = clock
        self.progress = progress
        self.progress_interval = progress_interval
        self.phase = phase
        self._ticking = deadline is not None or progress is not None
        self._started = clock()
        self._reported = self._started
        self.rows_built = 0
        self.names = list(space.axes)
        self.n = len(self.names)
        self.level_map = {n: levels(space, n, seed) for n in self.names}
        self.size = [len(self.level_map[n]) for n in self.names]
        self.liveness = liveness or Liveness(space, self.level_map)
        self.allowed = [list(range(s)) for s in self.size]
        if allowed is not None:
            for p, n in enumerate(self.names):
                self.allowed[p] = [
                    i
                    for i, level in enumerate(self.level_map[n])
                    if any(_same(level, v) for v in allowed.get(n, ()))
                ]
        self.base_row = None
        self._single = {}
        self._checks = {}
        self.partial = _Partial(space, self.liveness.deps.expanded)
        self._unders = {}
        self._specs = {}
        self._problems = {}
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
        self.settled_base = 0  # tuples settled by retargets before the current one
        self.phase_total = total  # tuples the current target set holds
        self.report_total = total  # the total a progress record gives
        self.uncovered = bytearray(b"\x01") * total
        # slot[x][y] = (offset, step of x's level, step of y's level) for either order.
        self.slot = [[None] * self.n for _ in range(self.n)]
        for (x, y), start in self.block_start.items():
            self.slot[x][y] = (start, self.size[y], 1)
            self.slot[y][x] = (start, 1, self.size[y])
        self.avoid = bytearray(total)
        self.has_avoid = bool(avoid)
        for pair in avoid or ():
            self.avoid[self.index_of(pair)] = 1
        if targets is not None:
            self.retarget(targets)

    def index_of(self, pair):
        """The position in ``uncovered`` of a tuple ``((axisA, valueA), (axisB, valueB))``."""
        (a, va), (b, vb) = pair
        x, y = self.names.index(a), self.names.index(b)
        if x > y:
            x, y, va, vb = y, x, vb, va
        lx = self._level_index(x, va)
        ly = self._level_index(y, vb)
        return self.block_start[x, y] + lx * self.size[y] + ly

    def retarget(self, pairs):
        """Make these tuples the only ones left to cover."""
        self.uncovered = bytearray(self.total)
        for pair in pairs:
            self.uncovered[self.index_of(pair)] = 1
        self.phase_total = len(pairs)

    def tick(self):
        """Between tuples and at every node of a search: ``_Budget`` once the deadline has
        passed, else a progress record if the interval has."""
        if not self._ticking:
            return
        now = self.clock()
        if self.deadline is not None and now >= self.deadline:
            raise _Budget
        if self.progress is not None and now - self._reported >= self.progress_interval:
            self._reported = now
            self.progress(
                {
                    "space": self.space.space_id,
                    "phase": self.phase,
                    "settled": self.settled_base
                    + self.phase_total
                    - self.uncovered.count(1),
                    "total": self.report_total,
                    "rows": self.rows_built,
                    "elapsed": now - self._started,
                }
            )

    def _level_index(self, pos, value):
        for i, level in enumerate(self.level_map[self.names[pos]]):
            if _same(level, value):
                return i
        raise ValueError(f"{value!r} is not a level of axis '{self.names[pos]}'")

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
        a random order, at the level that covers most uncovered tuples with the axes set so far
        (fewest tuples to avoid first: when every allowed level of an axis holds one, the gain of
        the best is negative, and it is still the level taken). Every axis needs an allowed level.
        """
        row = [-1] * self.n
        for p, level in fixed.items():
            row[p] = level
        order = [p for p in range(self.n) if row[p] < 0]
        rng.shuffle(order)
        placed = [p for p in range(self.n) if row[p] >= 0]
        unc = self.uncovered
        avoid = self.avoid if self.has_avoid else None
        for x in order:
            best, ties = None, []
            slots = [(self.slot[x][y], row[y]) for y in placed]
            for level in self.allowed[x]:
                gain = 0
                if avoid is None:
                    for (start, step_x, step_y), ly in slots:
                        gain += unc[start + level * step_x + ly * step_y]
                else:  # a tuple to avoid outweighs any number of tuples to cover
                    for (start, step_x, step_y), ly in slots:
                        at = start + level * step_x + ly * step_y
                        gain += unc[at] - (self.n + 1) * avoid[at]
                if best is None or gain > best:
                    best, ties = gain, [level]
                elif gain == best:
                    ties.append(level)
            row[x] = ties[0] if len(ties) == 1 else rng.choice(ties)
            placed.append(x)
        return tuple(row)

    def choose(self, rows, start):
        """The legal row of ``rows`` with the most newly covered tuples that cover tuple ``start``
        (axes ``a`` and ``b``) live; None if none does. A row that holds fewer tuples of
        ``avoid`` comes first."""
        live = self.liveness
        scored = []
        for k, row in enumerate(rows):
            if live.violation(self.values(row)) is None:
                pairs = self.row_pairs(row)
                scored.append((self.avoided(row), -len(pairs), k, row, pairs))
        scored.sort(key=lambda s: s[:3])
        a, b = start
        best, best_gain, best_bad = None, 0, 0
        for bad, neg, _k, row, pairs in scored:
            if best is not None and bad > best_bad:
                break
            if -neg <= best_gain:
                continue
            if not (live.live_index(row, a) and live.live_index(row, b)):
                continue
            gain = sum(
                1
                for x, y in pairs
                if live.live_index(row, x) and live.live_index(row, y)
            )
            if gain > best_gain:
                best, best_gain, best_bad = row, gain, bad
        return best

    def avoided(self, row):
        """How many tuples of ``avoid`` a row holds (0 when there are none)."""
        if not self.has_avoid:
            return 0
        return sum(
            self.avoid[start + row[x] * self.size[y] + row[y]]
            for (x, y), start in self.block_start.items()
        )

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
        if self.base_row is None and self.witness is not None:
            try:
                row = tuple(
                    self._level_index(p, self.witness[n])
                    for p, n in enumerate(self.names)
                )
            except (KeyError, ValueError):
                row = None
            if row is not None and self.liveness.violation(self.values(row)) is None:
                self.base_row = row
        if self.base_row is None:
            rng = random.Random(
                f"gauntlet-pairwise-base:{self.space.space_id}:{self.seed}"
            )
            for _ in range(BASE_ROW_ATTEMPTS):
                row = tuple(rng.choice(a) for a in self.allowed)
                if self.liveness.violation(self.values(row)) is None:
                    self.base_row = row
                    break
            else:
                raise ValueError(
                    f"space '{self.space.space_id}': no legal row among the levels"
                    f" in {BASE_ROW_ATTEMPTS} draws"
                )
        return self.base_row

    def _under(self, fixed):
        """What the expressions of the space read with the axes of ``fixed`` (a sorted tuple of
        ``(position, level)``) at those levels."""
        found = self._unders.get(fixed)
        if found is None:
            values = self.liveness.level_values
            found = self._unders[fixed] = _Under(
                self.partial, {self.names[p]: values[p][i] for p, i in fixed}
            )
        return found

    def _scope(self, fixed, axes):
        """The part of ``fixed`` that sits on the axes (names) ``axes``."""
        return tuple(f for f in fixed if self.names[f[0]] in axes)

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

    def _eff_check(self, kind, ident, static, members, fixed):
        """The check that constraint number ``ident`` (``kind`` "constraint") holds, or that the
        derived value ``ident`` (``kind`` "fallible") evaluates, from the levels of ``members``
        (the axes it effectively reads with ``fixed`` set) and of the axes of ``fixed`` that sit
        among the ``static`` ones. The answers are cached by those levels, shared by every search
        that fixes the same of those axes at the same levels."""
        scope = self._scope(fixed, static)
        found = self._checks.get((kind, ident, scope))
        if found is None:
            space = self.space
            names = self.names
            values = self.liveness.level_values
            base = {names[p]: values[p][i] for p, i in scope}
            if kind == "constraint":
                expr = space.constraints[ident].expr
                rule = expr.source
            else:
                expr, rule = space.derive[ident], None

            def fn(key):
                env = _LazyEnv(space.derive, base)
                for p, i in zip(members, key):
                    env[names[p]] = values[p][i]
                try:
                    held = expr.evaluate(env)
                except SpaceError:
                    return EVALUATION_FAILS
                return rule if rule is not None and not held else None

            found = self._checks[kind, ident, scope] = _Check(members, fn)
        return found

    def _live_spec(self, pos, fixed):
        """What the liveness of axis ``pos`` (one of the ``fixed`` axes) reads: the groups of the
        case, constraints and fallible derived values that read it with every other fixed axis
        set (the axis itself stays open: another level of it is what liveness tries), and, with
        the axes they read, the ``members`` that are free."""
        found = self._specs.get((pos, fixed))
        if found is None:
            partial = self.partial
            axis = self.names[pos]
            position = self.liveness.position
            under = self._under(tuple(f for f in fixed if f[0] != pos))
            support = {axis}
            groups, rules, derived = [], [], []
            for i in range(len(partial.groups)):
                reads = partial.group_reads(under, i)
                if axis in reads:
                    support |= reads
                    groups.append(i)
            for i, node in enumerate(partial.constraint_nodes):
                reads = under.reads(node)
                if axis in reads:
                    support |= reads
                    rules.append(i)
            for name in self.liveness.deps.fallible:
                reads = under.reads(partial.derive_nodes[name])
                if axis in reads:
                    support |= reads
                    derived.append(name)
            if not groups:  # no group of the case reads the axis: it is never live
                support, rules, derived = {axis}, [], []
            members = tuple(sorted(position[a] for a in support if a != axis))
            found = self._specs[pos, fixed] = _LiveSpec(
                members, tuple(groups), tuple(rules), tuple(derived)
            )
        return found

    def _live_check(self, pos, spec, fixed):
        """The check that axis ``pos`` is live: some other level of it gives a legal row (the
        constraints and derived values of ``spec`` hold) whose case differs (one of the groups of
        ``spec`` renders differently), from the levels of the ``members`` of ``spec``. Liveness
        is read from the groups alone: a case differs exactly where one of its groups does.
        """
        axis = self.names[pos]
        scope = self._scope(fixed, self.liveness.deps.support[axis])
        found = self._checks.get(("live", pos, scope))
        if found is None:
            space = self.space
            names = self.names
            values = self.liveness.level_values
            base = {names[p]: values[p][i] for p, i in scope}
            own = base[axis]
            members = spec.members
            groups = [self.partial.groups[i] for i in spec.groups]
            rules = [space.constraints[i].expr for i in spec.rules]
            derived = [space.derive[n] for n in spec.derived]
            others = [v for v in values[pos] if not _same(v, own)]

            def env_at(key, level):
                env = _LazyEnv(space.derive, base)
                for p, i in zip(members, key):
                    env[names[p]] = values[p][i]
                env[axis] = level
                return env

            def fn(key):
                try:
                    before = [_group_value(g, env_at(key, own)) for g in groups]
                except SpaceError:
                    return EVALUATION_FAILS
                for level in others:
                    alt = env_at(key, level)
                    try:
                        for expr in derived:
                            expr.evaluate(alt)
                        if not all(rule.evaluate(alt) for rule in rules):
                            continue
                        after = [_group_value(g, alt) for g in groups]
                    except SpaceError:
                        continue
                    if after != before:
                        return None
                return INERT

            found = self._checks["live", pos, scope] = _Check(members, fn)
        return found

    def _reach_check(self, pos, fixed, relevant):
        """The check that the case can still read axis ``pos`` (it can only be live then): with
        the other fixed axes and the free relevant axes that decide what is read (``deciders``:
        the tests of conditionals and guards) at their levels, some group of the case reads it.
        Fewer axes set only means more is read, so this refutes a prefix as soon as its deciders
        are set, long before every axis the liveness reads is. None when there is no such axis to
        set."""
        deps = self.liveness.deps
        position = self.liveness.position
        axis = self.names[pos]
        fixed_positions = {p for p, _ in fixed}
        support = deps.support[axis]
        members = tuple(
            sorted(
                position[a]
                for a in support & self.partial.deciders
                if a != axis
                and position[a] in relevant
                and position[a] not in fixed_positions
            )
        )
        if not members:
            return None
        scope = self._scope(fixed, support)
        found = self._checks.get(("reach", pos, scope, members))
        if found is None:
            names = self.names
            values = self.liveness.level_values
            base = {names[p]: values[p][i] for p, i in scope if p != pos}
            partial = self.partial

            def fn(key):
                fixed_values = dict(base)
                for p, i in zip(members, key):
                    fixed_values[names[p]] = values[p][i]
                return (
                    None
                    if partial.is_read(_Under(partial, fixed_values), axis)
                    else INERT
                )

            found = self._checks["reach", pos, scope, members] = _Check(members, fn)
        return found

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
        """The checks that need no analysis of the fixed axes (``kind`` "move")."""
        found = self._checks.get((kind, key))
        if found is None:
            found = self._checks[kind, key] = self._move_check(key)
        return found

    def _problem(self, fixed, need):
        """The search for a legal row with the axes of ``fixed`` (a sorted tuple of ``(position,
        level)``) set in which the axes of ``need`` (a sorted tuple of those positions) are live.

        It depends on the fixed axes alone, so what the expressions read *with them set* is what
        counts (``_Under``): the constraints (and derived values that can fail) that read a fixed
        axis in some branch, the axes they effectively read, and, for each ``need`` axis, the
        members of its liveness (``_live_spec``) are the *relevant* axes, closed under the
        constraints that read one of them. A constraint whose effective axes are all outside
        keeps the value it has in the legal base row, which every other axis is taken from.
        Each constraint, each fallible derived value and each liveness is one check; a check of a
        constraint that reads no fixed axis in any branch and no relevant axis is not needed.
        """
        found = self._problems.get((fixed, need))
        if found is not None:
            return found
        space = self.space
        deps = self.liveness.deps
        position = self.liveness.position
        names = self.names
        partial = self.partial
        under = self._under(fixed)
        fixed_axes = frozenset(names[p] for p, _ in fixed)
        fixed_positions = {p for p, _ in fixed}

        def where(axes):
            return tuple(sorted(position[a] for a in axes))

        legal = [
            (
                "constraint",
                i,
                deps.constraints[i],
                under.reads(partial.constraint_nodes[i]),
            )
            for i in range(len(space.constraints))
        ]
        legal += [
            ("fallible", name, static, under.reads(partial.derive_nodes[name]))
            for name, static in deps.fallible.items()
        ]
        lives = {p: self._live_spec(p, fixed) for p in need}
        relevant = set(fixed_positions)
        for _kind, _ident, static, reads in legal:
            if static & fixed_axes:
                relevant.update(where(reads))
        for spec in lives.values():
            relevant.update(spec.members)
        grew = True
        while grew:
            grew = False
            for _kind, _ident, _static, reads in legal:
                touched = where(reads)
                if relevant.intersection(touched) and not relevant.issuperset(touched):
                    relevant.update(touched)
                    grew = True
        checks = []
        for kind, ident, static, reads in legal:
            members = where(reads)
            if static & fixed_axes or relevant.intersection(members):
                checks.append(self._eff_check(kind, ident, static, members, fixed))
        for p, spec in lives.items():
            checks.append(self._live_check(p, spec, fixed))
            reach = self._reach_check(p, fixed, relevant)
            if reach is not None:
                checks.append(reach)
            for part in ("full", "differs", *deps.touching[names[p]]):
                check = self._check("move", (p, part))
                if check.members <= relevant:
                    checks.append(check)
        touch = {}
        for index, check in enumerate(checks):
            for p in check.positions:
                touch.setdefault(p, []).append(index)
        free = tuple(p for p in sorted(relevant - fixed_positions) if p in touch)
        found = self._problems[fixed, need] = _Problem(
            checks, free, touch, tuple(sorted(relevant))
        )
        return found

    def _solve(self, fixed, need):
        """Search for a legal row with the axes of ``fixed`` (``(position, level)`` pairs) at
        their levels in which every axis of ``need`` is live, over the relevant axes of
        ``_problem`` (every other axis is taken from a legal base row). Returns ``(row, rejected,
        nodes, searched)``: the row or None, what refuted the nodes, the nodes visited and the
        relevant axes; raises ``_Capped`` past ``cap`` nodes.

        **Forward checking.** Whenever every axis but one of a check is set, the levels of that
        one that pass the check are all that stay in its domain (the answer is cached by the
        levels of the others); an empty domain cuts the branch, a single level is set at once
        and propagates further (it costs no node). Each node sets the axis with the smallest
        domain.

        **Components.** Once the fixed axes (and what propagates from them) are set, the axes
        left are split into the connected components of the checks that read two of them (a
        constraint, a derived value that can fail, a liveness: one link between all the axes it
        reads). The components share no check, so a row exists iff every component has a
        completion, and each is searched alone: a hopeless component is refuted without
        enumerating the others' levels. A liveness reads several axes, so it links them all,
        and a space whose cases read every axis through it stays one component.
        """
        fixed = tuple(sorted(fixed))
        need = tuple(sorted(need))
        problem = self._problem(fixed, need)
        checks, touch = problem.checks, problem.touch
        lv = [-1] * self.n
        for p, level in fixed:
            lv[p] = level
        dom = {p: list(self.allowed[p]) for p in problem.free}
        rejected = Counter()
        nodes = 0
        trail = []
        searched = problem.searched

        def undo(mark):
            while len(trail) > mark:
                p, old = trail.pop()
                if old is None:
                    lv[p] = -1
                else:
                    dom[p] = old

        def verdict(check, key):
            found = check.cache.get(key, _UNSET)
            if found is _UNSET:
                found = check.cache[key] = check.fn(key)
            return found

        def process(index, queue):
            check = checks[index]
            positions = check.positions
            open_ = [m for m in positions if lv[m] < 0]
            if not open_:
                found = verdict(check, tuple([lv[m] for m in positions]))
                if found is not None:
                    rejected[found] += 1
                    return False
                return True
            if len(open_) > 1:
                return True
            p = open_[0]
            at = positions.index(p)
            key = [lv[m] for m in positions]
            keep, why = [], None
            for level in dom[p]:
                key[at] = level
                found = verdict(check, tuple(key))
                if found is None:
                    keep.append(level)
                else:
                    why = found
            if len(keep) == len(dom[p]):
                return True
            if not keep:
                rejected[why or NO_LEVEL] += 1
                return False
            trail.append((p, dom[p]))
            dom[p] = keep
            if len(keep) == 1:
                lv[p] = keep[0]
                trail.append((p, None))
                queue.append(p)
            return True

        def propagate(queue):
            while queue:
                for index in touch[queue.pop()]:
                    if not process(index, queue):
                        return False
            return True

        def solve(component):
            nonlocal nodes
            open_ = [p for p in component if lv[p] < 0]
            if not open_:
                return True
            p = min(open_, key=lambda q: (len(dom[q]), -len(touch[q]), q))
            for level in list(dom[p]):
                if self._ticking:
                    self.tick()
                nodes += 1
                if nodes > self.cap:
                    raise _Capped
                mark = len(trail)
                lv[p] = level
                trail.append((p, None))
                if propagate([p]) and solve(component):
                    return True
                undo(mark)
            return False

        queue = []
        for p in problem.free:
            if not dom[p]:
                rejected[NO_LEVEL] += 1
                return None, rejected, nodes, searched
            if len(dom[p]) == 1:
                lv[p] = dom[p][0]
                queue.append(p)
        for index in range(len(checks)):
            if not process(index, queue):
                return None, rejected, nodes, searched
        if not propagate(queue):
            return None, rejected, nodes, searched
        parent = {p: p for p in problem.free if lv[p] < 0}

        def find(p):
            while parent[p] != p:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p

        for check in checks:
            head = None
            for m in check.positions:
                if m in parent:
                    if head is None:
                        head = m
                    else:
                        parent[find(m)] = find(head)
        components = {}
        for p in parent:
            components.setdefault(find(p), []).append(p)
        for component in components.values():
            if not solve(component):
                return None, rejected, nodes, searched
        row = list(self.legal_base_row())
        for p, level in fixed:
            row[p] = level
        for p in problem.free:
            if lv[p] >= 0:
                row[p] = lv[p]
        full = tuple(row)
        live = self.liveness
        if live.violation(self.values(full)) is not None or not all(
            live.live_index(full, p) for p in need
        ):
            raise RuntimeError(
                f"space '{self.space.space_id}': a row the search found is not legal and"
                f" live: {self.values(full)} (fixed {fixed}, live {need})"
            )
        return full, rejected, nodes, searched

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
                found = self._solve(((pos, level),), (pos,))[0] is not None
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
            row, rejected, nodes, _ = self._solve(both, ())
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
                row, _, _, searched = self._solve(both, (a, b))
                if row is not None:
                    return _Search("found", row, "", searched)
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
        rows = []
        for k in range(self.candidates):
            self.tick()
            rows.append(
                self.fill(
                    fixed,
                    random.Random(f"{self.seed_prefix}:{row_number}:{k}"),
                )
            )
        best = self.choose(rows, (x, y))
        if best is not None:
            return best, None
        verdict = self.search(x, lx, y, ly)
        if verdict.kind != "found":
            return None, verdict
        fixed = {p: verdict.row[p] for p in verdict.searched}
        rows = [verdict.row] + [
            self.fill(
                fixed,
                random.Random(f"{self.seed_prefix}:{row_number}:fallback{k}"),
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
    deadline=None,
    clock=time.monotonic,
    progress=None,
    progress_interval=DEFAULT_PROGRESS_INTERVAL,
):
    """A seeded pairwise covering set of ``space`` (see the module docstring).

    ``candidates`` is how many completions each row is chosen from; ``cap`` bounds the
    combinations the exhaustive search enumerates per tuple. Only strength 2 is supported.
    Deterministic in ``(space, seed, candidates, cap)`` as long as the deadline does not pass.

    ``deadline`` is an absolute reading of ``clock`` (``time.monotonic`` by default): at it the
    build stops, keeps its rows, and puts every tuple not settled yet in ``budget_exceeded``
    (``generation_complete`` false). ``progress(record)`` gets ``{"space", "phase", "settled",
    "total", "rows", "elapsed"}`` at most every ``progress_interval`` seconds."""
    if strength != 2:
        raise ValueError(f"only pairwise (strength 2) is supported, got {strength}")
    started = time.perf_counter()
    b = _Builder(
        space,
        seed,
        candidates,
        cap,
        deadline=deadline,
        clock=clock,
        progress=progress,
        progress_interval=progress_interval,
    )
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
    try:
        while True:
            while cursor < b.total and not b.uncovered[cursor]:
                cursor += 1
            if cursor == b.total:
                break
            b.tick()
            x, y, lx, ly = b.decode(cursor)
            before = b.exhaustive_seconds
            row, verdict = b.next_row((x, y, lx, ly), len(result.rows))
            fallbacks += b.exhaustive_seconds > before
            if row is not None:
                b.credit(row)
                result.rows.append(copy.deepcopy(b.values(row)))
                b.rows_built = len(result.rows)
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
    except _Budget:
        result.generation_complete = False
        result.budget_exceeded = _unsettled(b)
    result.unproven = list(b.unproven_at.values())
    result.timings = {
        "total_s": time.perf_counter() - started,
        "exhaustive_s": b.exhaustive_seconds,
        "exhaustive_searches": fallbacks,
        "renders": b.liveness.renders,
    }
    return result


BUDGET_DETAIL = "the generation budget ended before this tuple was settled"


def _unsettled(b):
    """The tuples a builder still has to cover, as ``budget_exceeded`` findings, in universe
    order."""
    return [
        Finding(b.tuple_of(*b.decode(i)), BUDGET_DETAIL)
        for i in range(b.total)
        if b.uncovered[i]
    ]


def _universe(names, level_map):
    """Every tuple of the levels as ``((x, lx, y, ly), ((axisX, valueX), (axisY, valueY)))``,
    axes in space order (x before y)."""
    for x, y in itertools.combinations(range(len(names)), 2):
        for lx, vx in enumerate(level_map[names[x]]):
            for ly, vy in enumerate(level_map[names[y]]):
                yield (
                    (x, lx, y, ly),
                    ((names[x], copy.deepcopy(vx)), (names[y], copy.deepcopy(vy))),
                )


def _tally(space, records, level_map, liveness):
    """Read a run's records: ``(drawn, passing, witness)``. ``drawn`` maps each tuple key
    ``(x, lx, y, ly)`` that some record drew (both values levels, both live in its row) to the set
    of outcomes of the records that drew it; ``passing`` is every ``(axis position, level index)``
    that some passed record held, live or not; ``witness`` is the axis values of the first passed
    record whose every value is a level (None if there is none)."""
    names = list(space.axes)
    drawn, passing, witness = {}, set(), None
    for values, outcome in records:
        index = {}
        for p, name in enumerate(names):
            value = values[name]
            for i, level in enumerate(level_map[name]):
                if _same(level, value):
                    index[p] = i
                    break
        if outcome in PASSED_OUTCOMES:
            passing.update(index.items())
            if witness is None and len(index) == len(names):
                witness = copy.deepcopy(values)
        held = sorted(p for p in index if liveness.live(values, names[p]))
        for x, y in itertools.combinations(held, 2):
            drawn.setdefault((x, index[x], y, index[y]), set()).add(outcome)
    return drawn, passing, witness


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
    drawn, _, _ = _tally(space, records, level_map, liveness)
    passed = {k for k, outcomes in drawn.items() if outcomes & set(PASSED_OUTCOMES)}
    declined = {k for k, outcomes in drawn.items() if outcomes == {DECLINED_OUTCOME}}
    not_passed = []
    total = 0
    for key, pair in _universe(list(space.axes), level_map):
        total += 1
        if key not in passed:
            not_passed.append(pair)
    return {
        "tuples": total,
        "drawn": len(drawn),
        "passed": len(passed),
        "declined_only": len(declined),
        "never_passed": len(drawn) - len(passed),
        "not_passed": not_passed,
    }


class Targets(NamedTuple):
    """What a run leaves for a repair round: ``targets`` (tuples every drawing record of which
    ended ``DECLINED`` or ``UNEXERCISED``, in universe order), ``allowed`` (axis -> the levels
    some passed record held) and ``witness`` (the axis values of a passed record, or None).
    """

    targets: list
    allowed: dict
    witness: dict


def repair_targets(space, records, level_map, liveness=None):
    """The tuples of a run worth a second try, and the levels to try them with (see ``repair``).
    ``records`` and ``level_map`` are those of ``pair_coverage``."""
    liveness = liveness if liveness is not None else Liveness(space, level_map)
    names = list(space.axes)
    drawn, passing, witness = _tally(space, records, level_map, liveness)
    waiting = {DECLINED_OUTCOME, UNEXERCISED_OUTCOME}
    targets = [
        pair
        for key, pair in _universe(names, level_map)
        if drawn.get(key, ()) and drawn[key] <= waiting
    ]
    allowed = {
        name: [
            copy.deepcopy(level)
            for i, level in enumerate(level_map[name])
            if (p, i) in passing
        ]
        for p, name in enumerate(names)
    }
    return Targets(targets, allowed, witness)


class Repair:
    """The result of ``repair``: ``rows`` (axis-value dicts), ``retried`` (the targets a row was
    built for), ``never_passed`` (targets not retried because a value of theirs is no level in
    ``allowed``), the ``unreachable`` findings (retried targets no row of the allowed levels
    covers, all of them when ``allowed`` has no level for some axis), the ``budget_exceeded``
    findings (retried targets left unsettled when the deadline passed) with
    ``generation_complete`` (false then) and ``timings``."""

    def __init__(self, space, seed, round_number):
        self.space_id = space.space_id
        self.seed = seed
        self.round = round_number
        self.rows = []
        self.retried = []
        self.never_passed = []
        self.unreachable = []
        self.budget_exceeded = []
        self.generation_complete = True
        self.timings = {}


def _pair_key(pair):
    return (pair[0][0], repr(pair[0][1]), pair[1][0], repr(pair[1][1]))


def repair(
    space,
    seed,
    targets,
    allowed,
    *,
    round_number,
    stubborn=(),
    witness=None,
    liveness=None,
    candidates=DEFAULT_CANDIDATES,
    cap=DEFAULT_EXHAUSTIVE_CAP,
    deadline=None,
    clock=time.monotonic,
    progress=None,
    progress_interval=DEFAULT_PROGRESS_INTERVAL,
):
    """Rows that cover ``targets`` (tuples as ``repair_targets`` lists them) using only the levels
    of ``allowed`` (axis -> level values) on every axis: the construction and the liveness
    of ``build``, with the targets as the only tuples left to cover. A target with a value
    outside ``allowed`` is not retried (``never_passed``).

    ``stubborn`` are targets an earlier round already retried without success. A row built for
    other targets avoids holding them (a tuple that always declines would otherwise decline every
    row it is packed into), and each gets a row of its own that avoids the other stubborn ones.

    ``seed`` is the pairwise seed of the run whose levels ``targets`` are in; the rows are
    deterministic in ``(space, seed, round_number, targets, stubborn, allowed, witness)``.
    ``witness`` is a legal row of allowed levels (axis -> value) that fills the axes a tuple does
    not depend on; ``liveness`` shares the answers of the build.

    ``deadline``, ``clock``, ``progress`` and ``progress_interval`` are those of ``build``: at the
    deadline the targets not settled yet are ``budget_exceeded`` and the rows so far are kept.
    """
    started = time.perf_counter()
    result = Repair(space, seed, round_number)

    def usable(axis, value):
        return any(_same(value, v) for v in allowed.get(axis, ()))

    retry = []
    for pair in targets:
        (a, va), (b, vb) = pair
        (retry if usable(a, va) and usable(b, vb) else result.never_passed).append(pair)
    result.retried = retry
    if not retry:
        return result
    stubborn_keys = {_pair_key(p) for p in stubborn}
    fresh = [p for p in retry if _pair_key(p) not in stubborn_keys]
    isolated = [p for p in retry if _pair_key(p) in stubborn_keys]
    b = _Builder(
        space,
        seed,
        candidates,
        cap,
        liveness=liveness,
        allowed=allowed,
        targets=fresh,
        avoid=isolated,
        seed_prefix=f"gauntlet-pairwise-repair:{space.space_id}:{seed}:{round_number}",
        witness=witness,
        deadline=deadline,
        clock=clock,
        progress=progress,
        progress_interval=progress_interval,
        phase="repair",
    )
    b.report_total = len(retry)
    barred = [name for p, name in enumerate(b.names) if not b.allowed[p]]
    if (
        barred
    ):  # a row has a value on every axis: with no level to give one, no row exists
        detail = "no level of " + ", ".join(f"'{name}'" for name in barred) + " passed"
        result.unreachable = [Finding(pair, detail) for pair in retry]
        return result

    def cover(cursor):
        x, y, lx, ly = b.decode(cursor)
        row, verdict = b.next_row((x, y, lx, ly), len(result.rows))
        if row is not None:
            b.credit(row)
            result.rows.append(copy.deepcopy(b.values(row)))
            b.rows_built = len(result.rows)
            return
        detail = (
            f"{verdict.kind}: {verdict.detail}"
            if verdict is not None
            else "no completion kept both values live"
        )
        result.unreachable.append(Finding(b.tuple_of(x, y, lx, ly), detail))
        b.uncovered[cursor] = 0

    cursor = 0
    next_isolated = 0
    try:
        while True:
            while cursor < b.total and not b.uncovered[cursor]:
                cursor += 1
            if cursor == b.total:
                break
            b.tick()
            cover(cursor)
        for next_isolated, pair in enumerate(isolated):
            b.settled_base = len(fresh) + next_isolated
            b.retarget([pair])
            b.tick()
            cover(b.index_of(pair))
        next_isolated = len(isolated)
    except _Budget:
        result.generation_complete = False
        unsettled = {_pair_key(f.pair): f for f in _unsettled(b)}
        for pair in isolated[next_isolated:]:
            unsettled.setdefault(_pair_key(pair), Finding(pair, BUDGET_DETAIL))
        result.budget_exceeded = list(unsettled.values())
    result.timings = {
        "total_s": time.perf_counter() - started,
        "exhaustive_s": b.exhaustive_seconds,
        "renders": b.liveness.renders,
    }
    return result
