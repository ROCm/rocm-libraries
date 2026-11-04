# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Chain check: Layer 0 -> Layer 1 rules -> Layer 2 rules must give exactly the model's axes and values.

The API (Layer 2) drives the model; this check is the safety net. A model value missing from
Layer 0 means Layer 0 must be fixed; a Layer 0 value missing from the model needs a rule id that
says why.

Usage: python3 chain_check.py <plan_dir> <chain-map.yaml>
       (plan_dir holds model.yaml and rules.yaml, and layers/ holds the three Layer files)
Exit code 0 = no ERROR lines. Reads YAML only; changes nothing.

Definitions
  removed value   a Layer 0 value that an L1 or L2 rule removes ON ITS OWN (a forbid conjunction
                  naming only that axis), or a child value whose parent can never take an activating
                  value after those removals.
  L2 values       Layer 0 values minus removed values: what the frontend can state and means something.
  model values    model.yaml values minus those a single-axis rule of rules.yaml removes (values kept
                  only so the rejection suite can name them).
Checks
  A  every model value (incl. removed ones) traces to a Layer 0 value (same id, `class:`, or chain-map);
  B  every model value the model keeps traces to an L2 value (the model keeps nothing L1/L2 removes);
  C  every L2 value appears in the model, or chain-map names a rule id (in rules.yaml) explaining why not
     (`l0_values_refused_by_policy` for values a policy rule removes: meaningful, stated by the frontend,
     but hipDNN should refuse them);
  D  every value an L1/L2 rule removes is either absent from the model or removed there by the same rule id;
  E  rule ids: every L1/L2 rule is in rules.yaml or touches only values outside the model; every model
     definition/frontend rule is in L1/L2 (reported, not an error: some are model-only adaptations);
  F  policy rules (kind policy): none is an L1 or L2 rule; each names a `ticket`; a `covered` one names an
     exclusion rule in `covered_by`; every value one removes on its own is in chain-map
     `l0_values_refused_by_policy` under that rule, and every entry there is removed by the rule it names.
"""
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def load(p):
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def conjs(rule, key="forbid"):
    out = []
    if key in rule:
        out.append(rule[key])
    if key == "forbid":
        out.extend(rule.get("forbid_any") or [])
    return out


def space(axes):
    values = {a["id"]: [v["id"] for v in a["values"]] for a in axes}
    parent = {}
    for a in axes:
        if a["kind"] == "child":
            ((pax, pv),) = a["parent"].items()
            parent[a["id"]] = (pax, set(pv))
    classes = {a["id"]: {v["id"]: v.get("class") for v in a["values"]} for a in axes}
    return values, parent, classes


def single_removals(rules, values):
    removed = {}
    for r in rules:
        for c in conjs(r):
            if len(c) == 1:
                ((ax, vs),) = c.items()
                for v in vs:
                    if ax in values and v in values[ax]:
                        removed.setdefault((ax, v), r["id"])
    return removed


def close_children(values, parent, removed):
    changed = True
    while changed:
        changed = False
        for ax, (pax, pvs) in parent.items():
            live_parent = {v for v in values[pax] if (pax, v) not in removed}
            if not (live_parent & pvs):
                for v in values[ax]:
                    if (ax, v) not in removed:
                        removed[(ax, v)] = f"parent {pax} never {sorted(pvs)}"
                        changed = True
    return removed


def main():
    plan, mapfile = Path(sys.argv[1]), Path(sys.argv[2])
    l0 = load(plan / "layers/L0-axes.yaml")
    l1 = load(plan / "layers/L1-definition.yaml")
    l2 = load(plan / "layers/L2-frontend.yaml")
    model = load(plan / "model.yaml")
    rules = load(plan / "rules.yaml")["rules"]
    cmap = load(mapfile)

    v0, p0, _ = space(l0["axes"])
    set_aside = {s["id"] for s in l0.get("set_aside", [])}
    vm, pm, cm = space(model["axes"])
    rule_ids = {r["id"] for r in rules}
    rule_by_id = {r["id"]: r for r in rules}
    l1_rules, l2_rules = l1["rules"], l2["rules"]
    l1_ids = {r["id"] for r in l1_rules} | {s["id"] for s in l1.get("structural", [])}
    l2_ids = {r["id"] for r in l2_rules}

    errors, notes = [], []
    E = errors.append
    N = notes.append

    # ---- removals
    rem_l1 = close_children(v0, p0, dict(single_removals(l1_rules, v0)))
    both = dict(rem_l1)
    for k, v in single_removals(l2_rules, v0).items():
        both.setdefault(k, v)
    rem_l2 = close_children(v0, p0, both)
    l2_values = {(a, v) for a in v0 for v in v0[a] if (a, v) not in rem_l2}
    rem_model = close_children(vm, pm, single_removals(rules, vm))

    # ---- model value -> Layer 0 preimage
    src = cmap.get("model_axis_sources") or {}
    pre = {}
    for a, vals in vm.items():
        for v in vals:
            if a in src:
                s = src[a]
                if "l0_axis" in s:
                    pre[(a, v)] = {(s["l0_axis"], x) for x in v0[s["l0_axis"]]}
                elif s.get("l0_set_aside") in set_aside:
                    pre[(a, v)] = {("set_aside", s["l0_set_aside"])}
                else:
                    pre[(a, v)] = set()
                continue
            if a not in v0:
                pre[(a, v)] = set()
                continue
            if v in v0[a]:
                pre[(a, v)] = {(a, v)}
            elif cm[a].get(v) in v0[a]:
                pre[(a, v)] = {(a, cm[a][v])}
            else:
                pre[(a, v)] = set()
    # folded Layer 0 values (chain-map folded_into) also count as images
    fates = cmap.get("l0_axis_fates") or {}

    # A
    for (a, v), p in sorted(pre.items()):
        if not p:
            E(
                f"A  model value {a}={v} has no Layer 0 value (fix Layer 0, or add a chain-map entry)"
            )

    # B
    for (a, v), p in sorted(pre.items()):
        if (a, v) in rem_model:
            continue
        if any(x[0] == "set_aside" for x in p):
            continue
        if p and not (p & l2_values):
            why = sorted({rem_l2.get(x, "?") for x in p})
            E(
                f"B  model keeps {a}={v}, but Layer 1/2 remove its Layer 0 value(s) {sorted(p)} by {why}"
            )

    # C
    image = defaultdict(set)
    for mv, p in pre.items():
        if mv in rem_model:
            continue
        for x in p:
            image[x].add(mv)
    not_in = cmap.get("l0_values_not_in_model") or {}
    refused = cmap.get("l0_values_refused_by_policy") or {}
    explained = defaultdict(list)
    for a, v in sorted(l2_values):
        if image.get((a, v)):
            continue
        rid = None
        if a in fates:
            f = fates[a].get(v) or fates[a].get("all_values")
            if f:
                rid = f.get("explained_by")
                if (
                    f.get("folded_into") is not None
                    or f.get("to_model_axis")
                    or f.get("derived_from")
                ):
                    explained["folded or derived"].append(f"{a}={v} ({rid})")
                else:
                    explained["not modelled"].append(f"{a}={v} ({rid})")
        elif v in (not_in.get(a) or {}):
            rid = not_in[a][v]
            explained["not modelled"].append(f"{a}={v} ({rid})")
        elif v in (refused.get(a) or {}):
            rid = refused[a][v]
            explained["refused by policy (rejection suite)"].append(f"{a}={v} ({rid})")
        if rid is None:
            E(f"C  Layer 2 value {a}={v} is not in the model and no rule explains why")
        elif rid not in rule_ids:
            E(f"C  Layer 2 value {a}={v}: explaining rule {rid} is not in rules.yaml")
    # every chain-map entry must point at a real Layer 0 value
    for a, vs in not_in.items():
        for v in vs:
            if a not in v0 or v not in v0[a]:
                E(
                    f"C  chain-map l0_values_not_in_model names {a}={v}, which is not in Layer 0"
                )
    for a, vs in refused.items():
        for v in vs:
            if a not in v0 or v not in v0[a]:
                E(
                    f"C  chain-map l0_values_refused_by_policy names {a}={v}, which is not in Layer 0"
                )
    for a, f in fates.items():
        if a not in v0:
            E(f"C  chain-map l0_axis_fates names axis {a}, which is not in Layer 0")
            continue
        for v in f:
            if v != "all_values" and v not in v0[a]:
                E(f"C  chain-map l0_axis_fates names {a}={v}, which is not in Layer 0")

    # D
    for (a, v), rid in sorted(rem_l2.items()):
        mvs = [mv for mv, p in pre.items() if (a, v) in p]
        if not mvs:
            continue
        for mv in mvs:
            if mv not in rem_model:
                # a model value may also stand for other, live L0 values (e.g. dtype_k_vs_q)
                if pre[mv] & l2_values:
                    continue
                E(
                    f"D  Layer 0 {a}={v} removed by {rid}, but the model keeps {mv[0]}={mv[1]}"
                )
            elif not rid.startswith("parent ") and rem_model[mv] != rid:
                N(
                    f"D  {a}={v}: removed by {rid} in the layers, by {rem_model[mv]} in the model"
                )

    # E
    model_values = {(a, v) for a in vm for v in vm[a]}
    for r in l1_rules + l2_rules:
        if r["id"] in rule_ids:
            continue
        # A conjunction can match a model row only if, for every axis it names, one of its values has
        # a model image. A rule none of whose conjunctions can match is moot in the model.
        live = [
            c
            for c in conjs(r)
            if all(any(image.get((ax, x)) for x in vs) for ax, vs in c.items())
        ]
        if live:
            E(
                f"E  Layer rule {r['id']} is not in rules.yaml, yet {len(live)} of its conjunctions can match model rows: {live}"
            )
        else:
            N(
                f"E  Layer rule {r['id']} is not in rules.yaml: every conjunction names a value outside the model (moot there)"
            )
    for r in rules:
        if r["kind"] == "definition" and r["id"] not in l1_ids:
            N(f"E  model definition rule {r['id']} has no Layer 1 counterpart")
        if r["kind"] == "frontend" and r["id"] not in l2_ids:
            N(f"E  model frontend rule {r['id']} has no Layer 2 counterpart")

    # F
    for r in rules:
        if r["kind"] != "policy":
            continue
        rid = r["id"]
        if rid in l1_ids or rid in l2_ids:
            E(f"F  policy rule {rid} is also a Layer 1/2 rule; a policy is neither")
        if not r.get("ticket"):
            E(f"F  policy rule {rid} names no ticket")
        if r.get("form") == "covered":
            cov = rule_by_id.get(r.get("covered_by"))
            if cov is None or not conjs(cov):
                E(
                    f"F  policy rule {rid}: covered_by {r.get('covered_by')!r} is not an exclusion rule in rules.yaml"
                )
            if conjs(r):
                E(f"F  policy rule {rid} is `covered` but has its own forbid")
        for c in conjs(r):
            if len(c) != 1:
                continue
            ((ax, vs),) = c.items()
            for v in vs:
                for x in pre.get((ax, v), set()):
                    if x[0] in v0 and refused.get(x[0], {}).get(x[1]) != rid:
                        E(
                            f"F  policy rule {rid} removes {ax}={v}; chain-map l0_values_refused_by_policy must list {x[0]}={x[1]}: {rid}"
                        )
    for a, vs in refused.items():
        for v, rid in vs.items():
            r = rule_by_id.get(rid)
            if r is None or r["kind"] != "policy":
                E(
                    f"F  chain-map l0_values_refused_by_policy {a}={v}: {rid} is not a policy rule in rules.yaml"
                )
                continue
            mvs = [mv for mv, p in pre.items() if (a, v) in p]
            if not mvs or any(rem_model.get(mv) != rid for mv in mvs):
                N(
                    f"F  {a}={v}: listed under {rid}, removed in the model by {[rem_model.get(mv) for mv in mvs]}"
                )
                if not mvs or any(mv not in rem_model for mv in mvs):
                    E(
                        f"F  chain-map l0_values_refused_by_policy {a}={v}: the model keeps it"
                    )
    pol = [r["id"] for r in rules if r["kind"] == "policy"]

    # ---- report
    n0 = sum(len(x) for x in v0.values())
    print(
        f"Layer 0: {len(v0)} axes, {n0} values; removed by L1 alone {len(rem_l1)}, by L1+L2 {len(rem_l2)}; "
        f"L2 values {len(l2_values)}"
    )
    print(
        f"Model:   {len(vm)} axes, {len(model_values)} values; kept {len(model_values) - len(rem_model)}, "
        f"kept only for the rejection suite {len(rem_model)}"
    )
    print(
        f"Policy rules (should be refused by hipDNN, accepted by the frontend today): {', '.join(pol) or 'none'}"
    )
    for k in (
        "folded or derived",
        "not modelled",
        "refused by policy (rejection suite)",
    ):
        print(f"L2 values not in the model, {k}: {len(explained[k])}")
        for line in explained[k]:
            print(f"    {line}")
    print(f"NOTES ({len(notes)})")
    for n in notes:
        print(f"  {n}")
    print(f"ERRORS ({len(errors)})")
    for e in errors:
        print(f"  ERROR {e}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
