#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Count combinations and valid value pairs per research layer, by arithmetic.
#
# Inputs: L0-axes.yaml, L1-definition.yaml, L2-frontend.yaml (one folder, test-plans/sdpa-fwd/layers).
# Output: a JSON summary on stdout.
#
# Counting model
#   - A combination assigns one value to every ACTIVE axis. A child axis is active only when its
#     parent axis is active and has one of the listed parent values. An inactive child adds no
#     value and no factor.
#   - A pair is two (axis, value) features of different axes. A pair is valid at a layer when at
#     least one valid combination of that layer contains both features.
#   - A combination is valid at a layer when no conjunction of the layer's rules matches it.
#
# Method
#   The six size axes (HUBS) are enumerated. Every other top-level axis is a tree (the axis and its
#   descendants). Trees that share a rule conjunction (ignoring hub axes) form one component. Given
#   the hub values, components are independent, so the count is a sum over hub assignments of a
#   product of per-component counts. Nothing is listed beyond one component's states.

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict

import yaml

HUBS = ["batch_size", "seq_q", "seq_kv", "q_heads", "head_dim_qk", "head_dim_v"]
MAX_COMPONENT_STATES = 2_000_000


def load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Space:
    def __init__(self, l0):
        self.axes = {a["id"]: a for a in l0["axes"]}
        self.values = {
            aid: [v["id"] for v in a["values"]] for aid, a in self.axes.items()
        }
        self.parent = {}
        self.children = defaultdict(list)
        for aid, a in self.axes.items():
            if a["kind"] == "child":
                ((pax, pvals),) = a["parent"].items()
                assert pax in self.axes, (aid, pax)
                for pv in pvals:
                    assert pv in self.values[pax], (aid, pax, pv)
                self.parent[aid] = (pax, set(pvals))
                self.children[pax].append(aid)
        self.roots = [aid for aid in self.axes if aid not in self.parent]
        for h in HUBS:
            assert h in self.roots and not self.children[h], h
        self._states = {}

    def root(self, aid):
        while aid in self.parent:
            aid = self.parent[aid][0]
        return aid

    def states(self, aid):
        if aid in self._states:
            return self._states[aid]
        out = []
        for v in self.values[aid]:
            active = [c for c in self.children[aid] if v in self.parent[c][1]]
            for combo in itertools.product(*(self.states(c) for c in active)):
                s = {aid: v}
                for part in combo:
                    s.update(part)
                out.append(s)
        self._states[aid] = out
        return out


def parse_rules(space, docs):
    rules = []
    for doc in docs:
        for r in doc["rules"]:
            conjs = []
            if "forbid" in r:
                conjs.append(r["forbid"])
            conjs.extend(r.get("forbid_any", []))
            if not conjs:
                # 2026-10-29: a rule with no Layer 0 forbid (e.g. F-SCALE-TENSOR-BY-VALUE, whose forbid
                # names only a model value, `forbid_in_model`) removes nothing at these layers.
                assert "forbid_in_model" in r or "layer0_effect" in r, r["id"]
                continue
            parsed = []
            for c in conjs:
                pc = {}
                for ax, vals in c.items():
                    assert ax in space.axes, (r["id"], ax)
                    for v in vals:
                        assert v in space.values[ax], (r["id"], ax, v)
                    pc[ax] = frozenset(vals)
                parsed.append(pc)
            rules.append((r["id"], r["kind"], parsed))
    return rules


def matches(conj, state):
    for ax, vals in conj.items():
        if state.get(ax) not in vals:
            return False
    return True


def pairs_of_state(s):
    feats = sorted(s.items())
    return {(a, b) for a, b in itertools.combinations(feats, 2)}


class Layer:
    """Exact combination count and valid-pair set for one rule set."""

    def __init__(self, space, rules):
        self.space = space
        conjs = [c for _, _, cs in rules for c in cs]
        hubset = set(HUBS)
        nonhub_roots = [r for r in space.roots if r not in hubset]
        uf = {r: r for r in nonhub_roots}

        def find(x):
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        hub_only = []
        touched = set()
        conj_trees = []
        for c in conjs:
            trees = {space.root(ax) for ax in c} - hubset
            conj_trees.append(trees)
            if not trees:
                hub_only.append(c)
                continue
            touched |= trees
            t0 = next(iter(trees))
            for t in trees:
                uf[find(t)] = find(t0)
        comps = defaultdict(list)
        for t in nonhub_roots:
            if t in touched:
                comps[find(t)].append(t)
        self.components = []
        for trees in comps.values():
            tset = set(trees)
            size = math.prod(len(space.states(t)) for t in trees)
            assert size <= MAX_COMPONENT_STATES, (trees, size)
            cstates = []
            for combo in itertools.product(*(space.states(t) for t in trees)):
                s = {}
                for part in combo:
                    s.update(part)
                cstates.append(s)
            cconjs = [c for c, tr in zip(conjs, conj_trees) if tr and tr <= tset]
            self.components.append((sorted(trees), cstates, cconjs))
        self.free = [t for t in nonhub_roots if t not in touched]
        self.hub_only = hub_only

    def evaluate(self):
        space = self.space
        free_count = math.prod(len(space.states(t)) for t in self.free)
        total = 0
        # Per group: map hub-assignment index -> set of features; within-group pairs.
        comp_feats = [dict() for _ in self.components]
        comp_pairs = [set() for _ in self.components]
        valid_h = []
        # 2026-10-29 speed-up (same results): a component's valid states depend only on the hub axes
        # its conjunctions name, so each component is evaluated once per value of those hubs.
        rel_hubs = [
            sorted({ax for c in cconjs for ax in c if ax in HUBS})
            for _, _, cconjs in self.components
        ]
        cache = [dict() for _ in self.components]
        merged = [set() for _ in self.components]
        for hvals in itertools.product(*(space.values[h] for h in HUBS)):
            h = dict(zip(HUBS, hvals))
            if any(matches(c, h) for c in self.hub_only):
                continue
            keys = []
            for i, (trees, cstates, cconjs) in enumerate(self.components):
                key = tuple(h[x] for x in rel_hubs[i])
                if key not in cache[i]:
                    ok = []
                    for s in cstates:
                        full = dict(h)
                        full.update(s)
                        if not any(matches(c, full) for c in cconjs):
                            ok.append(s)
                    feats = set()
                    pairs_here = set()
                    for s in ok:
                        feats |= set(s.items())
                        pairs_here |= pairs_of_state(s)
                    cache[i][key] = (len(ok), frozenset(feats), pairs_here)
                keys.append(key)
            prod = math.prod(cache[i][k][0] for i, k in enumerate(keys))
            if prod == 0:
                continue
            total += prod * free_count
            hi = len(valid_h)
            valid_h.append(h)
            for i, key in enumerate(keys):
                n, feats, pairs_here = cache[i][key]
                if key not in merged[i]:
                    comp_pairs[i] |= pairs_here
                    merged[i].add(key)
                comp_feats[i][hi] = feats

        groups = []  # (name, dict hi->feats or None for constant, constant feats)
        for h in HUBS:
            groups.append(
                (
                    "hub:" + h,
                    {hi: frozenset({(h, hv[h])}) for hi, hv in enumerate(valid_h)},
                )
            )
        for i, (trees, _, _) in enumerate(self.components):
            groups.append(("comp:" + "+".join(trees), comp_feats[i]))
        within = set()
        for p in comp_pairs:
            within |= p
        for t in self.free:
            feats = set()
            for s in space.states(t):
                feats |= set(s.items())
                within |= pairs_of_state(s)
            groups.append(
                ("free:" + t, {hi: frozenset(feats) for hi in range(len(valid_h))})
            )

        def union(d):
            u = set()
            for f in d.values():
                u |= f
            return u

        info = []
        for name, d in groups:
            u = union(d)
            const = all(f == frozenset(u) for f in d.values())
            info.append((name, d, frozenset(u), const))
        cross = set()
        for (n1, d1, u1, c1), (n2, d2, u2, c2) in itertools.combinations(info, 2):
            if c1 or c2:
                for a in u1:
                    for b in u2:
                        cross.add((a, b) if a < b else (b, a))
            else:
                for hi in d1:
                    for a in d1[hi]:
                        for b in d2[hi]:
                            cross.add((a, b) if a < b else (b, a))
        pairs = within | cross
        live = defaultdict(set)
        for _, _, u, _ in info:
            for ax, v in u:
                live[ax].add(v)
        return total, pairs, dict(live)


def summarize(space, rules):
    total, pairs, live = Layer(space, rules).evaluate()
    multi = {ax for ax, vs in live.items() if len(vs) >= 2}
    pairs_multi = {p for p in pairs if p[0][0] in multi and p[1][0] in multi}
    return {
        "combinations": total,
        "log10_combinations": round(math.log10(total), 3) if total else None,
        "valid_pairs": len(pairs),
        "valid_pairs_live_multivalue_axes": len(pairs_multi),
        "live_axes": len(live),
        "live_multivalue_axes": len(multi),
        "live_values": sum(len(v) for v in live.values()),
    }, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("research_dir")
    ap.add_argument(
        "--pairs-out",
        metavar="CSV",
        help="also write the valid pairs of every layer to this CSV "
        "(layer,axis_a,value_a,axis_b,value_b; axis_a < axis_b)",
    )
    args = ap.parse_args()
    d = args.research_dir.rstrip("/")
    files = {
        n: f"{d}/{n}"
        for n in ("L0-axes.yaml", "L1-definition.yaml", "L2-frontend.yaml")
    }
    l0 = load(files["L0-axes.yaml"])
    l1 = load(files["L1-definition.yaml"])
    l2 = load(files["L2-frontend.yaml"])
    space = Space(l0)
    r1 = parse_rules(space, [l1])
    r2 = parse_rules(space, [l2])
    kinds = {"child": 0}
    for a in space.axes.values():
        kinds[a["kind"]] = kinds.get(a["kind"], 0) + 1

    layers = {}
    pairsets = {}
    layers[0], pairsets[0] = summarize(space, [])
    layers[1], pairsets[1] = summarize(space, r1)
    layers[2], pairsets[2] = summarize(space, r1 + r2)
    raw_pairs_l0 = 0
    roots_vals = {}
    # Unconstrained pair-slot count over every axis pair, ignoring parent/child co-occurrence.
    vals = [len(space.values[a]) for a in space.axes]
    raw_pairs_l0 = (sum(vals) ** 2 - sum(v * v for v in vals)) // 2

    per_rule = {}
    for layer, base_rules, new_rules in ((1, [], r1), (2, r1, r2)):
        base_total = layers[layer - 1]["combinations"]
        base_pairs = pairsets[layer - 1]
        lost = base_pairs - pairsets[layer]
        restored_by = {}
        for rule in new_rules:
            alone_total, alone_pairs, _ = Layer(space, base_rules + [rule]).evaluate()
            others = [r for r in new_rules if r is not rule]
            _, without_pairs, _ = Layer(space, base_rules + others).evaluate()
            restored = lost & without_pairs
            restored_by[rule[0]] = restored
            per_rule[rule[0]] = {
                "layer": layer,
                "kind": rule[1],
                "combinations_removed_alone": base_total - alone_total,
                "pairs_removed_alone": len(base_pairs - alone_pairs),
                "pairs_only_this_rule_removes": len(restored),
            }
        sole = set()
        for s in restored_by.values():
            sole |= s
        layers[layer]["pairs_lost_from_previous_layer"] = len(lost)
        layers[layer]["pairs_lost_jointly_only"] = len(lost - sole)
        layers[layer]["combinations_removed_from_previous_layer"] = (
            base_total - layers[layer]["combinations"]
        )

    out = {
        "inputs": {n: sha256(p) for n, p in files.items()},
        "script_sha256": sha256(__file__),
        "axes": {"total": len(space.axes), **kinds},
        "hubs": HUBS,
        "raw_value_pair_slots_ignoring_children": raw_pairs_l0,
        "rules": {"layer1": len(r1), "layer2": len(r2)},
        "layers": {str(k): v for k, v in layers.items()},
        "per_rule": per_rule,
    }
    if args.pairs_out:
        with open(args.pairs_out, "w", encoding="utf-8", newline="") as f:
            f.write("layer,axis_a,value_a,axis_b,value_b\n")
            for layer in (0, 1, 2):
                for (a, va), (b, vb) in sorted(pairsets[layer]):
                    f.write(f"{layer},{a},{va},{b},{vb}\n")
    json.dump(out, sys.stdout, indent=1, default=str)
    print()


if __name__ == "__main__":
    main()
