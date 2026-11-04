# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Find a valid full row that contains given values, or prove that none exists.

A row gives every active axis one value. A child axis is active only when its parent is active
and has one of the listed parent values; an inactive child holds None. A row is valid when it
matches no conjunction of any exclusion rule.

The axes split into components: axes linked by a rule conjunction or by a parent/child link
share a component. Components are independent, so each one is solved alone. The search picks
the axis with the fewest remaining values first and removes values that would complete a
forbidden conjunction (forward checking).

When a branch fails, the search records which earlier assignments caused the failure (the
conflict set) and returns straight to the latest of them (conflict-directed backjumping). The
branches it skips hold no valid row, so the search still returns the first valid row in value
order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import Model

CACHE_LIMIT = 300_000
WITNESSES_PER_COMPONENT = 16


@dataclass
class Component:
    axes: list
    conjs_by_axis: dict = field(default_factory=dict)


class Solver:
    def __init__(self, model: Model, rules=None):
        self.model = model
        self.rules = list(model.rules if rules is None else rules)
        conjs = [c for r in self.rules for c in r.conjs]

        uf = {a: a for a in model.axes}

        def find(x):
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                uf[rb] = ra

        for c in conjs:
            axes = list(c)
            for other in axes[1:]:
                union(axes[0], other)
        for child, (pax, _) in model.parent.items():
            union(child, pax)

        groups = {}
        for a in model.axes:
            groups.setdefault(find(a), []).append(a)
        self.components = []
        self.comp_of = {}
        for axes in groups.values():
            comp = Component(axes=axes)
            for a in axes:
                comp.conjs_by_axis[a] = []
            ci = len(self.components)
            self.components.append(comp)
            for a in axes:
                self.comp_of[a] = ci
        for c in conjs:
            ci = self.comp_of[next(iter(c))]
            for a in c:
                self.components[ci].conjs_by_axis[a].append(c)

        self._cache = {}
        self._witnesses = [[] for _ in self.components]
        for ci in range(len(self.components)):
            if self._solve(ci, {}, None) is None:
                raise ValueError(
                    f"component {self.components[ci].axes} has no valid row"
                )

    # ------------------------------------------------------------------ public
    def feasible(self, fixed: dict) -> bool:
        by_comp = {}
        for a, v in fixed.items():
            by_comp.setdefault(self.comp_of[a], {})[a] = v
        for ci, part in by_comp.items():
            if not self._comp_feasible(ci, part):
                return False
        return True

    def complete(self, fixed: dict, prefs: dict | None = None, rng=None):
        """Return a valid full row containing `fixed`, or None.

        `prefs` maps an axis to its preferred value (None for inactive). With `rng`, every other
        value order is shuffled; without it, model order is used and inactive comes last.
        """
        by_comp = {}
        for a, v in fixed.items():
            by_comp.setdefault(self.comp_of[a], {})[a] = v
        row = {}
        for ci in range(len(self.components)):
            sol = self._solve(ci, by_comp.get(ci, {}), (prefs or {}, rng))
            if sol is None:
                return None
            row.update(sol)
        return {a: row[a] for a in self.model.axes}

    def domain(self, axis: str) -> list:
        d = list(self.model.values[axis])
        if axis in self.model.parent:
            d.append(None)
        return d

    # ------------------------------------------------------------------ internals
    def _comp_feasible(self, ci: int, part: dict) -> bool:
        for w in self._witnesses[ci]:
            if all(w[a] == v for a, v in part.items()):
                return True
        key = (ci, tuple(sorted(part.items(), key=lambda kv: (kv[0], kv[1] or ""))))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        sol = self._solve(ci, part, None)
        ok = sol is not None
        if ok:
            ws = self._witnesses[ci]
            ws.insert(0, sol)
            del ws[WITNESSES_PER_COMPONENT:]
        if len(self._cache) >= CACHE_LIMIT:
            self._cache.clear()
        self._cache[key] = ok
        return ok

    def _value_order(self, axis: str, order):
        dom = self.domain(axis)
        if order is None:
            return dom
        prefs, rng = order
        rest = list(dom)
        if rng is not None:
            rng.shuffle(rest)
        if axis in prefs:
            p = prefs[axis]
            if p in rest:
                rest.remove(p)
                rest.insert(0, p)
        return rest

    def _solve(self, ci: int, fixed: dict, order):
        comp = self.components[ci]
        doms = {}
        for a in comp.axes:
            if a in fixed:
                if fixed[a] not in self.domain(a):
                    return None
                doms[a] = [fixed[a]]
            else:
                doms[a] = self._value_order(a, order)
        sol, _ = self._dfs(comp, {}, doms, {})
        return sol

    def _dfs(self, comp: Component, asg: dict, doms: dict, why: dict):
        """(row, None) for the first valid row below this node, or (None, conflict set).

        `why` maps an unassigned axis to the assigned axes whose values removed some of its values.
        A conflict set names assigned axes only."""
        if len(asg) == len(comp.axes):
            return dict(asg), None
        best, bestn = None, None
        for a in comp.axes:
            if a in asg:
                continue
            n = len(doms[a])
            if best is None or n < bestn:
                best, bestn = a, n
                if n <= 1:
                    break
        conflict = set(why.get(best, ()))
        for v in doms[best]:
            asg[best] = v
            nd, nwhy, failed = self._propagate(comp, best, v, asg, doms, why)
            if nd is None:
                conflict |= failed - {best}
            else:
                sol, sub = self._dfs(comp, asg, nd, nwhy)
                if sol is not None:
                    del asg[best]
                    return sol, None
                if best not in sub:
                    del asg[best]
                    return None, sub
                conflict |= sub - {best}
            del asg[best]
        return None, conflict

    def _propagate(self, comp: Component, a: str, v, asg: dict, doms: dict, why: dict):
        """(domains, reasons, None) after assigning `a`, or (None, None, conflict set)."""
        parent = self.model.parent
        nd = dict(doms)
        nwhy = dict(why)
        nd[a] = [v]

        def prune(axis, keep, reasons):
            nd[axis] = keep
            nwhy[axis] = nwhy.get(axis, frozenset()) | reasons

        if a in parent:
            pax, pvals = parent[a]
            if pax in asg:
                active = asg[pax] in pvals
                if (v is None) == active:
                    return None, None, {a, pax}
            else:
                keep = [p for p in nd[pax] if (p in pvals) == (v is not None)]
                if not keep:
                    return None, None, set(nwhy.get(pax, ())) | {a}
                if len(keep) != len(nd[pax]):
                    prune(pax, keep, frozenset({a}))

        for c in self.model.children[a]:
            cp = parent[c][1]
            active = v in cp
            if c in asg:
                if (asg[c] is None) == active:
                    return None, None, {a, c}
            else:
                keep = [x for x in nd[c] if (x is None) != active]
                if not keep:
                    return None, None, set(nwhy.get(c, ())) | {a}
                if len(keep) != len(nd[c]):
                    prune(c, keep, frozenset({a}))

        if v is None:
            return nd, nwhy, None
        for conj in comp.conjs_by_axis[a]:
            if v not in conj[a]:
                continue
            unassigned = None
            open_count = 0
            matched = True
            for x, vs in conj.items():
                if x == a:
                    continue
                if x in asg:
                    if asg[x] not in vs:
                        matched = False
                        break
                else:
                    open_count += 1
                    unassigned = x
            if not matched:
                continue
            fixed = frozenset(x for x in conj if x != unassigned)
            if open_count == 0:
                return None, None, set(fixed)
            if open_count == 1:
                vs = conj[unassigned]
                keep = [x for x in nd[unassigned] if x is None or x not in vs]
                if not keep:
                    return None, None, set(nwhy.get(unassigned, ())) | fixed
                if len(keep) != len(nd[unassigned]):
                    prune(unassigned, keep, fixed)
        return nd, nwhy, None
