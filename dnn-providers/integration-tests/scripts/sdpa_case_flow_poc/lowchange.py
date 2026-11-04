# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Low-change pairwise: cover pairs with rows that change at most N settings from a base row.

Counting rule: a setting counts as changed when it is set in both the base row and the row and
the two values differ. A setting that is unset in the base row and set in the row (a child that
a parent change switched on), or the reverse, does not count: it follows from its parent. The
axes of one group (a super-setting) count as one change when any of them counts.

Features: a feature is (key, value). For a plain axis the key is the axis and the value its value;
for a group the key is the group id and the value the tuple of its axes' values (None for an
inactive child), in the group's axis order.
"""

from __future__ import annotations

import random

from .model import Model
from .pairs import axis_pos, make_pair
from .solver import Solver


def counts_as_change(base: dict, axis: str, value) -> bool:
    return base[axis] is not None and value is not None and value != base[axis]


def change_count(
    model: Model, base: dict, row: dict, groups: dict | None = None
) -> int:
    """`groups` maps a group id to its axes; each group counts at most once."""
    groups = groups or {}
    grouped = {a for axes in groups.values() for a in axes}
    n = sum(counts_as_change(base, a, row[a]) for a in model.axes if a not in grouped)
    n += sum(
        any(counts_as_change(base, a, row[a]) for a in axes) for axes in groups.values()
    )
    return n


class Projection:
    """Row features with each group folded into one feature."""

    def __init__(self, model: Model, groups: dict | None = None):
        self.model = model
        self.groups = dict(groups or {})
        self.group_of = {a: g for g, axes in self.groups.items() for a in axes}
        pos = axis_pos(model)
        self.key_pos = {a: pos[a] for a in model.axes if a not in self.group_of}
        for g, axes in self.groups.items():
            self.key_pos[g] = min(pos[a] for a in axes)
        self.keys = sorted(self.key_pos, key=self.key_pos.get)

    def feature(self, row: dict, key: str):
        if key in self.groups:
            return (key, tuple(row[a] for a in self.groups[key]))
        return None if row[key] is None else (key, row[key])

    def features(self, row: dict) -> list:
        return [f for f in (self.feature(row, k) for k in self.keys) if f is not None]

    def pairs(self, row: dict):
        feats = self.features(row)
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                yield feats[i], feats[j]

    def assignment(self, feature) -> dict:
        key, value = feature
        if key in self.groups:
            return dict(zip(self.groups[key], value))
        return {key: value}

    def make_pair(self, f, g):
        return (f, g) if self.key_pos[f[0]] < self.key_pos[g[0]] else (g, f)

    def count(self, base: dict, row: dict) -> int:
        return change_count(self.model, base, row, self.groups)


class LowChange:
    """Rows with the fewest counted changes from `base` that hold given values, under `solver`."""

    def __init__(
        self,
        model: Model,
        solver: Solver,
        base: dict,
        max_changes: int,
        groups: dict | None = None,
    ):
        self.model = model
        self.solver = solver
        self.base = base
        self.max_changes = max_changes
        self.groups = dict(groups or {})
        self.group_of = {a: g for g, axes in self.groups.items() for a in axes}
        for g, axes in self.groups.items():
            if len({solver.comp_of[a] for a in axes}) > 1:
                raise ValueError(
                    f"group {g}: its axes lie in several solver components; link them in one"
                )
        self._conjs = {}
        for comp in solver.components:
            seen, conjs = set(), []
            for a in comp.axes:
                for c in comp.conjs_by_axis[a]:
                    if id(c) not in seen and c and all(c.values()):
                        seen.add(id(c))
                        conjs.append(c)
            self._conjs[id(comp)] = conjs
        self._cache = {}

    def complete(self, fixed: dict, budget: int | None = None):
        """Return (row, changes) with the fewest counted changes, at most `budget`; or None."""
        budget = self.max_changes if budget is None else budget
        by_comp = {}
        for a, v in fixed.items():
            by_comp.setdefault(self.solver.comp_of[a], {})[a] = v
        row = {a: self.base[a] for a in self.model.axes}
        spent = 0
        for ci in sorted(by_comp):
            got = self._min_comp(ci, by_comp[ci], budget - spent)
            if got is None:
                return None
            sol, cost = got
            row.update(sol)
            spent += cost
        return row, spent

    def feasible(self, fixed: dict) -> bool:
        return self.complete(fixed) is not None

    def _conflict_bound(
        self, comp, asg: dict, doms: dict, lb: int, bound: int, used: set
    ) -> int:
        """Raise `lb` by one for each rule conjunction that the row breaks if every open axis keeps its
        base value, and that only a counted change can leave; the conjunctions counted share no unit.
        """
        base = self.base
        for conj in self._conjs[id(comp)]:
            units = set()
            for x, vs in conj.items():
                if x in asg:
                    if asg[x] not in vs:
                        break
                    continue
                b = base[x]
                d = doms[x]
                if b is None or b not in vs or b not in d:
                    break
                if any(y not in vs and not counts_as_change(base, x, y) for y in d):
                    break
                units.add(self.group_of.get(x, x))
            else:
                if units and not (units & used):
                    used |= units
                    lb += 1
                    if lb > bound:
                        return lb
        return lb

    def _min_comp(self, ci: int, part: dict, budget: int):
        if budget < 0:
            return None
        key = (ci, tuple(sorted(part.items(), key=lambda kv: (kv[0], kv[1] or ""))))
        hit = self._cache.get(key)
        first = 0
        if hit is not None:
            best_cost, sol = hit
            if sol is None:
                if budget <= best_cost:
                    return None
                first = best_cost + 1
            else:
                return (sol, best_cost) if best_cost <= budget else None
        comp = self.solver.components[ci]
        doms = {}
        for a in comp.axes:
            dom = self.solver.domain(a)
            if a in part:
                if part[a] not in dom:
                    self._cache[key] = (budget, None)
                    return None
                doms[a] = [part[a]]
            else:
                b = self.base[a]
                doms[a] = sorted(
                    dom, key=lambda x: (x != b, x is not None, dom.index(x))
                )
        for bound in range(first, budget + 1):
            sol, _ = self._dfs(comp, {}, doms, 0, bound, frozenset(), {})
            if sol is not None:
                self._cache[key] = (bound, sol)
                return sol, bound
        self._cache[key] = (budget, None)
        return None

    def _dfs(
        self,
        comp,
        asg: dict,
        doms: dict,
        cost: int,
        bound: int,
        changed_groups: frozenset,
        why: dict,
    ):
        """(row, None) for the first row within `bound` changes, or (None, conflict set).

        A rule conflict names the assignments that caused it, so the search can jump back over the
        others (see Solver._dfs). A failure of the change bound depends on every assignment, so its
        conflict set is every assigned axis."""
        if len(asg) == len(comp.axes):
            return dict(asg), None
        base = self.base
        lb = cost
        # A unit is an axis, or a group (its axes change together for one count).
        used = set(changed_groups)
        # Branch first on an axis that cannot keep its base value, then on the smallest domain.
        best, bestn = None, None
        for a in comp.axes:
            if a in asg:
                continue
            d = doms[a]
            if all(counts_as_change(base, a, x) for x in d):
                unit = self.group_of.get(a, a)
                if unit not in used:
                    used.add(unit)
                    lb += 1
            n = (base[a] in d, len(d))
            if best is None or n < bestn:
                best, bestn = a, n
        if lb > bound:
            return None, set(asg)
        lb = self._conflict_bound(comp, asg, doms, lb, bound, used)
        if lb > bound:
            return None, set(asg)
        g = self.group_of.get(best)
        conflict = set(why.get(best, ()))
        for v in doms[best]:
            changed = counts_as_change(base, best, v)
            groups_now = changed_groups
            if g is not None and changed:
                changed = g not in changed_groups
                groups_now = changed_groups | {g}
            c = cost + changed
            if c > bound:
                conflict |= set(asg)
                continue
            asg[best] = v
            nd, nwhy, failed = self.solver._propagate(comp, best, v, asg, doms, why)
            if nd is None:
                conflict |= failed - {best}
            else:
                sol, sub = self._dfs(comp, asg, nd, c, bound, groups_now, nwhy)
                if sol is not None:
                    del asg[best]
                    return sol, None
                if best not in sub:
                    del asg[best]
                    return None, sub
                conflict |= sub - {best}
            del asg[best]
        return None, conflict


def lowchange_pairwise(
    model: Model,
    lc: LowChange,
    pairs,
    rng: random.Random,
    extend_tries: int = 40,
    projection: Projection | None = None,
) -> list:
    """Greedy covering set of `pairs` in which every row changes at most `lc.max_changes` settings.

    `pairs` must hold only pairs that some such row can hold, as features of `projection` (plain
    axis features when it is None). Each row starts from the feature with the most uncovered
    partners and one uncovered partner; further features are added while some completion stays
    within the limit, the one covering the most uncovered pairs first.
    """
    proj = projection or Projection(model, lc.groups)
    base = lc.base
    uncovered = set(pairs)
    partners = {}
    for f, g in sorted(uncovered, key=lambda p: (_order(p[0]), _order(p[1]))):
        partners.setdefault(f, set()).add(g)
        partners.setdefault(g, set()).add(f)
    rows = []

    def row_cov(row):
        return [p for p in proj.pairs(row) if p in uncovered]

    while uncovered:
        top = max(len(s) for s in partners.values() if s)
        seeds = sorted((f for f, s in partners.items() if len(s) == top), key=_order)
        f = rng.choice(seeds)
        g = rng.choice(sorted(partners[f], key=_order))
        chosen = [f, g]
        fixed = {**proj.assignment(f), **proj.assignment(g)}
        got = lc.complete(fixed)
        if got is None:
            raise AssertionError(
                f"pair {f} {g} has no row within {lc.max_changes} changes"
            )
        row, _ = got
        while True:
            keys = {x[0] for x in chosen}
            scores = {}
            for h, hp in partners.items():
                if h[0] in keys or not hp:
                    continue
                s = sum(1 for x in chosen if x in hp)
                if s:
                    scores[h] = s
            if not scores:
                break
            order = sorted(scores, key=lambda h: (-scores[h], _order(h)))[:extend_tries]
            added = False
            for h in order:
                trial = {**fixed, **proj.assignment(h)}
                got = lc.complete(trial)
                if got is not None:
                    fixed, row = trial, got[0]
                    chosen.append(h)
                    added = True
                    break
            if not added:
                break
        cov = row_cov(row)
        if not cov:
            raise AssertionError(f"row for {f} {g} covers no uncovered pair")
        for p, q in cov:
            uncovered.discard((p, q))
            partners[p].discard(q)
            partners[q].discard(p)
        rows.append(({"changes": proj.count(base, row)}, row))
    return rows


def _order(feature):
    """Sort key for features whose values may be tuples holding None."""
    key, value = feature
    if isinstance(value, tuple):
        return (key, tuple("" if x is None else x for x in value))
    return (key, (value,))
