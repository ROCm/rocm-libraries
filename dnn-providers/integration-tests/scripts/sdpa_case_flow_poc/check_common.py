# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared code of the checkers. It does not import the generator.

The checkers read model.yaml and the rule files themselves and decide row validity and pair
validity with their own search, so a generator fault cannot hide in shared code.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import yaml

ABSENT = "-"
MEMO_LIMIT = 500_000
MEMO_SETS = 4


class Spec:
    """Axes, values, parent links and forbidden conjunctions, read from the model files."""

    def __init__(self, model_dir: Path, extra_rule_files=()):
        model_dir = Path(model_dir)
        with open(model_dir / "model.yaml", encoding="utf-8") as f:
            m = yaml.safe_load(f)
        self.axes = [a["id"] for a in m["axes"]]
        self.values = {a["id"]: [v["id"] for v in a["values"]] for a in m["axes"]}
        self.parent = {}
        for a in m["axes"]:
            if a["kind"] == "child":
                ((pax, pvals),) = a["parent"].items()
                self.parent[a["id"]] = (pax, set(pvals))
        self.children = {
            a: [c for c, (p, _) in self.parent.items() if p == a] for a in self.axes
        }
        self.bases = {n: b["values"] for n, b in (m.get("base_cases") or {}).items()}
        self.hints = m.get("generator_hints") or {}
        self.nogoods = []  # (rule id, {axis: set(values)})
        files = [model_dir / "rules.yaml", *map(Path, extra_rule_files)]
        self.rule_files = files
        for path in files:
            with open(path, encoding="utf-8") as f:
                doc = yaml.safe_load(f)
            for r in doc.get("rules") or []:
                conjs = ([r["forbid"]] if "forbid" in r else []) + list(
                    r.get("forbid_any") or []
                )
                for c in conjs:
                    for ax, vs in c.items():
                        if ax not in self.values or any(
                            v not in self.values[ax] for v in vs
                        ):
                            raise ValueError(
                                f"rule {r['id']}: unknown axis or value in {c}"
                            )
                    self.nogoods.append(
                        (r["id"], {ax: set(vs) for ax, vs in c.items()})
                    )
        self.rule_ids = sorted({rid for rid, _ in self.nogoods})
        self._memos = {}

    def base_row(self, name: str) -> dict:
        b = self.bases[name]
        return {a: b.get(a) for a in self.axes}

    # ------------------------------------------------------------------ one row
    def active(self, row: dict, axis: str) -> bool:
        while axis in self.parent:
            pax, pvals = self.parent[axis]
            if row.get(pax) not in pvals:
                return False
            axis = pax
        return True

    def row_problems(self, row: dict, rule_ids=None) -> list:
        """Structure problems and broken rules of one full row (None = inactive axis)."""
        probs = []
        for a in self.axes:
            v = row.get(a)
            if self.active(row, a):
                if v is None:
                    probs.append(f"structure: {a} active but absent")
                elif v not in self.values[a]:
                    probs.append(f"structure: {a} has unknown value {v}")
            elif v is not None:
                probs.append(f"structure: {a}={v} but its parent does not activate it")
        seen = {a: (row.get(a) if self.active(row, a) else None) for a in self.axes}
        for rid, c in self.nogoods:
            if rule_ids is not None and rid not in rule_ids:
                continue
            if all(seen[ax] in vs for ax, vs in c.items()):
                probs.append(f"rule {rid}: matches {fmt_conj(c)}")
        return probs

    # ------------------------------------------------------------------ search
    def satisfiable(
        self, fixed: dict, rule_ids=None, base=None, max_changes=None, groups=None
    ) -> bool:
        """True when some full row contains `fixed` and breaks none of the selected rules.

        With `base` and `max_changes`, the row must also change at most `max_changes` settings from
        `base`; a setting counts when it is set in both rows with different values, and the axes of
        one group in `groups` ({id: axes}) count once together.
        """
        nogoods = [c for rid, c in self.nogoods if rule_ids is None or rid in rule_ids]
        doms = {}
        for a in self.axes:
            dom = set(self.values[a])
            if a in self.parent:
                dom.add(None)
            if a in fixed:
                if fixed[a] not in dom:
                    return False
                dom = {fixed[a]}
            doms[a] = dom
        doms = self._fixpoint(doms, nogoods)
        bound = None if max_changes is None else (base, max_changes, dict(groups or {}))
        return doms is not None and self._search(doms, nogoods, bound)

    def _fixpoint(self, doms: dict, nogoods: list):
        doms = {a: set(d) for a, d in doms.items()}
        changed = True
        while changed:
            changed = False
            for child, (pax, pvals) in self.parent.items():
                pd, cd = doms[pax], doms[child]
                can_on = any(p in pvals for p in pd)
                can_off = any(p not in pvals for p in pd)
                new_c = {
                    x
                    for x in cd
                    if (x is None and can_off) or (x is not None and can_on)
                }
                child_on = any(x is not None for x in new_c)
                child_off = None in new_c
                new_p = {
                    p
                    for p in pd
                    if (p in pvals and child_on) or (p not in pvals and child_off)
                }
                if new_c != cd or new_p != pd:
                    doms[child], doms[pax] = new_c, new_p
                    changed = True
                if not new_c or not new_p:
                    return None
            for c in nogoods:
                open_axes = [ax for ax in c if len(doms[ax]) != 1]
                fixed_match = all(
                    next(iter(doms[ax])) in c[ax] for ax in c if len(doms[ax]) == 1
                )
                if not fixed_match:
                    continue
                if not open_axes:
                    return None
                if len(open_axes) == 1:
                    u = open_axes[0]
                    new = {x for x in doms[u] if x not in c[u]}
                    if not new:
                        return None
                    if new != doms[u]:
                        doms[u] = new
                        changed = True
        return doms

    def _memo_for(self, nogoods: list, base=None, unit_of=None) -> dict:
        """Results of searches over one set of conjunctions, one base and one set of units."""
        base_values = None if base is None else tuple(base.get(a) for a in self.axes)
        key = (
            tuple(map(id, nogoods)),
            base_values,
            tuple(sorted((unit_of or {}).items())),
        )
        if key not in self._memos:
            if len(self._memos) >= MEMO_SETS:
                del self._memos[next(iter(self._memos))]
            self._memos[key] = (nogoods, {})
        memo = self._memos[key][1]
        if len(memo) > MEMO_LIMIT:
            memo.clear()
        return memo

    def _open_groups(self, doms: dict, nogoods: list, scope, unit_of=None) -> list:
        """Open axes of `scope`, split into groups that no live conjunction, parent link or unit joins.

        A conjunction is live when every axis it fixes holds a forbidden value; only its open axes
        can still decide it. A unit ({axis: unit id}) is a set of axes that count as one change.
        Returns (axes, indices of the live conjunctions on them) per group. Groups can be searched
        one after another."""
        open_axes = [a for a in self.axes if a in scope and len(doms[a]) > 1]
        uf = {a: a for a in open_axes}

        def find(x):
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        def join(xs):
            xs = [x for x in xs if x in uf]
            for x in xs[1:]:
                uf[find(x)] = find(xs[0])

        for child, (pax, _) in self.parent.items():
            join([child, pax])
        by_unit = {}
        for a in open_axes:
            if unit_of and a in unit_of:
                by_unit.setdefault(unit_of[a], []).append(a)
        for axes in by_unit.values():
            join(axes)
        live = []
        for i, c in enumerate(nogoods):
            if all(next(iter(doms[ax])) in c[ax] for ax in c if len(doms[ax]) == 1):
                axes = [ax for ax in c if ax in uf]
                if axes:
                    join(axes)
                    live.append((i, axes[0]))
        groups = {}
        for a in open_axes:
            groups.setdefault(find(a), []).append(a)
        lives = {r: set() for r in groups}
        for i, a in live:
            lives[find(a)].add(i)
        return [(axes, frozenset(lives[r])) for r, axes in groups.items()]

    @staticmethod
    def _state(doms: dict, axes: list) -> tuple:
        return tuple((a, frozenset(doms[a])) for a in axes)

    def _search_split(self, doms: dict, nogoods: list, scope, memo: dict) -> bool:
        groups = self._open_groups(doms, nogoods, scope)
        if not groups:
            return True
        if len(groups) > 1:
            return all(
                self._search_split(doms, nogoods, set(g), memo) for g, _ in groups
            )
        ((group, live),) = groups
        key = ("sat", self._state(doms, group), live)
        if key in memo:
            return memo[key]
        a = min(group, key=lambda x: len(doms[x]))
        found = False
        for v in sorted(doms[a], key=lambda x: (x is None, x or "")):
            trial = dict(doms)
            trial[a] = {v}
            nd = self._fixpoint(trial, nogoods)
            if nd is not None and self._search_split(nd, nogoods, set(group), memo):
                found = True
                break
        memo[key] = found
        return found

    def _search(self, doms: dict, nogoods: list, bound=None) -> bool:
        if bound is None:
            memo = self._memo_for(nogoods)
            return self._search_split(doms, nogoods, set(self.axes), memo)
        base, limit, groups = bound
        unit_of = {a: g for g, axes in groups.items() for a in axes}
        spent = self._changed_units(doms, base, unit_of)
        if len(spent) > limit:
            return False
        memo = self._memo_for(nogoods, base, unit_of)
        extra = self._min_extra(
            doms, nogoods, base, unit_of, set(self.axes), limit - len(spent), memo
        )
        return extra is not None

    def _changed_units(self, doms: dict, base: dict, unit_of: dict) -> set:
        """Units with a decided axis whose value differs from `base` (both values set)."""
        return {
            unit_of.get(a, a)
            for a in self.axes
            if len(doms[a]) == 1 and _differs(base[a], next(iter(doms[a])))
        }

    def _min_extra(
        self,
        doms: dict,
        nogoods: list,
        base: dict,
        unit_of: dict,
        scope,
        budget: int,
        memo: dict,
    ):
        """Fewest units that a valid completion of the open axes of `scope` changes in addition to
        the units changed already, or None when every completion needs more than `budget`.

        The open axes are split into independent groups. Their counts add up, because a live
        conjunction, a parent link and a unit never cross two groups."""
        comps = self._open_groups(doms, nogoods, scope, unit_of)
        if not comps:
            return 0
        if len(comps) > 1:
            total = 0
            for comp, _ in comps:
                r = self._min_extra(
                    doms, nogoods, base, unit_of, set(comp), budget - total, memo
                )
                if r is None:
                    return None
                total += r
            return total
        ((comp, live),) = comps
        spent = self._changed_units(doms, base, unit_of)
        units = {unit_of.get(a, a) for a in comp}
        key = ("min", self._state(doms, comp), live, frozenset(units & spent))
        hit = memo.get(key)
        if hit is not None:
            exact, value = hit
            if exact:
                return value if value <= budget else None
            if budget <= value:
                return None
        forced = {
            unit_of.get(a, a)
            for a in comp
            if base[a] is not None
            and all(x is not None and x != base[a] for x in doms[a])
        }
        best = None
        if len(forced - spent) <= budget:
            a = min(comp, key=lambda x: (base[x] in doms[x], len(doms[x])))
            for v in sorted(doms[a], key=lambda x: (x != base[a], x is None, x or "")):
                cap = budget if best is None else best - 1
                if cap < 0:
                    break
                trial = dict(doms)
                trial[a] = {v}
                nd = self._fixpoint(trial, nogoods)
                if nd is None:
                    continue
                d = len(self._changed_units(nd, base, unit_of) - spent)
                if d > cap:
                    continue
                r = self._min_extra(
                    nd, nogoods, base, unit_of, set(comp), cap - d, memo
                )
                if r is not None:
                    best = d + r
        memo[key] = (True, best) if best is not None else (False, budget)
        return best

    # ------------------------------------------------------------------ pairs
    def candidate_pairs(self, skip=()):
        feats = [(a, v) for a in self.axes if a not in skip for v in self.values[a]]
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                if feats[i][0] != feats[j][0]:
                    yield feats[i], feats[j]


def _differs(base_value, value) -> bool:
    return base_value is not None and value is not None and value != base_value


def fmt_conj(c: dict) -> str:
    return ", ".join(f"{ax} in {sorted(vs)}" for ax, vs in c.items())


def read_cases(path: Path, spec: Spec):
    """Yield (case_id, meta, row) from a generated CSV; row maps every axis to a value or None."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [a for a in spec.axes if a not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing axis columns {missing}")
        meta_cols = [
            c for c in reader.fieldnames if c not in spec.values and c != "case_id"
        ]
        for rec in reader:
            row = {a: (None if rec[a] == ABSENT else rec[a]) for a in spec.axes}
            yield rec["case_id"], {c: rec[c] for c in meta_cols}, row


def change_count(spec: Spec, base: dict, row: dict, groups=None) -> int:
    """Settings set in both rows with different values (a child switched on or off is not counted).

    The axes of one group in `groups` ({id: axes}) count once together."""
    groups = groups or {}
    grouped = {a for axes in groups.values() for a in axes}

    def changed(a):
        return base[a] is not None and row[a] is not None and row[a] != base[a]

    return sum(1 for a in spec.axes if a not in grouped and changed(a)) + sum(
        1 for axes in groups.values() if any(changed(a) for a in axes)
    )


def row_features(spec: Spec, row: dict, groups=None) -> list:
    """(key, value) features in axis order; each group is one feature whose value is the tuple of
    its axes' values (None for an inactive child), placed at its first axis."""
    groups = groups or {}
    group_of = {a: g for g, axes in groups.items() for a in axes}
    feats = []
    for a in spec.axes:
        g = group_of.get(a)
        if g is None:
            if row[a] is not None:
                feats.append((a, row[a]))
        elif a == min(groups[g], key=spec.axes.index):
            feats.append((g, tuple(row[x] for x in groups[g])))
    return feats


def row_pairs(spec: Spec, row: dict, groups=None):
    feats = row_features(spec, row, groups)
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            yield feats[i], feats[j]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
