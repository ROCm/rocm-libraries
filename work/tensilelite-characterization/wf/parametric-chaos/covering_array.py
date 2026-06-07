#!/usr/bin/env python3
"""Stdlib pairwise covering-array generator over the reduced per-branch domains.

Run 1 mandatory fallback (no ACTS/PICT needed). Reads every Domain fragment under
<fragdir> (each a DOMAIN record {branch_id, domains:{sym:{type,min,max,values}}}),
reduces each symbol to a small value set, and emits a 2-way (pairwise) covering array.

Writes:
  <outdir>/covering_array/model.json   (params -> value lists, provenance, method)
  <outdir>/covering_array/cases.csv    (one row per test case)
Prints a JSON summary to stdout. Stdlib only; deterministic (no randomness).
"""
import argparse
import csv
import glob
import itertools
import json
import os


def reduce_domain(sym, dom):
    """Reduce one symbol's domain to a small, ordered, deduped value list."""
    t = (dom or {}).get("type", "str")
    vals = (dom or {}).get("values") or []
    if vals:
        out = vals[:4]
    elif t == "bool":
        out = [True, False]
    elif t == "int":
        lo = dom.get("min")
        hi = dom.get("max")
        if lo is not None and hi is not None and hi > lo:
            mid = (lo + hi) // 2
            out = sorted({lo, mid, hi})
        elif lo is not None:
            out = [lo, lo + 1, lo + 2]
        elif hi is not None:
            out = [hi - 2, hi - 1, hi]
        else:
            out = [0, 1, 2]
    elif t == "float":
        lo = dom.get("min")
        hi = dom.get("max")
        out = [lo if lo is not None else 0.0, hi if hi is not None else 1.0]
    elif t == "enum":
        out = vals[:4] or ["a", "b"]
    else:  # str / unknown
        out = ["", "x"]
    # de-dup preserving order
    seen, ded = set(), []
    for v in out:
        key = repr(v)
        if key not in seen:
            seen.add(key)
            ded.append(v)
    return ded


def build_model(fragdir, max_params):
    """Union symbols across all Domain fragments into a bounded parameter model."""
    params = {}            # name -> value list
    provenance = {}        # name -> [branch_ids]
    frags = sorted(glob.glob(os.path.join(fragdir, "*.json")))
    for fp in frags:
        try:
            rec = json.load(open(fp))
        except Exception:
            continue
        bid = rec.get("branch_id", os.path.basename(fp))
        for sym, dom in (rec.get("domains") or {}).items():
            vals = reduce_domain(sym, dom)
            if sym not in params:
                params[sym] = vals
            else:
                # widen with any new values, keep bounded
                merged = params[sym] + [v for v in vals if repr(v) not in {repr(x) for x in params[sym]}]
                params[sym] = merged[:4]
            provenance.setdefault(sym, [])
            if bid not in provenance[sym]:
                provenance[sym].append(bid)
    # bound parameter count: prefer params seen in the most branches (highest signal)
    if len(params) > max_params:
        ranked = sorted(params, key=lambda s: (-len(provenance.get(s, [])), s))[:max_params]
        params = {s: params[s] for s in ranked}
        provenance = {s: provenance[s] for s in ranked}
    return params, provenance, len(frags)


def pairwise(params):
    """Deterministic 2-way covering array via seed-pair greedy fill.

    Each new row is seeded with a still-uncovered pair, then the remaining cells are
    filled to cover the most additional uncovered pairs. Seeding guarantees every row
    removes at least one pair, so the loop terminates in <= |needed| rows."""
    names = list(params.keys())
    if not names:
        return [], []
    if len(names) == 1:
        return names, [[v] for v in params[names[0]]]

    # ordered list of uncovered (i, j, vi, vj) pairs (deterministic)
    needed = []
    for i, j in itertools.combinations(range(len(names)), 2):
        for vi in params[names[i]]:
            for vj in params[names[j]]:
                needed.append((i, j, vi, vj))
    needed_set = set((i, j, repr(vi), repr(vj)) for (i, j, vi, vj) in needed)

    def covered_by(row):
        cov = set()
        for i, j in itertools.combinations(range(len(names)), 2):
            cov.add((i, j, repr(row[i]), repr(row[j])))
        return cov

    rows = []
    while needed_set:
        # pick the first still-uncovered pair as the seed
        seed = None
        for (i, j, vi, vj) in needed:
            if (i, j, repr(vi), repr(vj)) in needed_set:
                seed = (i, j, vi, vj)
                break
        i, j, vi, vj = seed
        row = [None] * len(names)
        row[i], row[j] = vi, vj
        # fill remaining cells greedily (set cells exist -> gain is meaningful)
        for k in sorted(range(len(names)), key=lambda k: -len(params[names[k]])):
            if row[k] is not None:
                continue
            best_val, best_gain = params[names[k]][0], -1
            for v in params[names[k]]:
                gain = 0
                for a in range(len(names)):
                    if a == k or row[a] is None:
                        continue
                    if a < k:
                        key = (a, k, repr(row[a]), repr(v))
                    else:
                        key = (k, a, repr(v), repr(row[a]))
                    if key in needed_set:
                        gain += 1
                if gain > best_gain:
                    best_gain, best_val = gain, v
            row[k] = best_val
        rows.append(row)
        needed_set -= covered_by(row)
    return names, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragdir", required=True, help="dir of Domain fragments (*.json)")
    ap.add_argument("--outdir", required=True, help="PublicInputSurface output root")
    ap.add_argument("--max-params", type=int, default=12)
    a = ap.parse_args()

    ca_dir = os.path.join(a.outdir, "covering_array")
    os.makedirs(ca_dir, exist_ok=True)

    params, provenance, n_frags = build_model(a.fragdir, a.max_params)
    names, rows = pairwise(params)

    model = {
        "method": "stdlib-greedy-pairwise-2way",
        "strength": 2,
        "domain_fragments_read": n_frags,
        "parameters": {n: params[n] for n in names},
        "provenance": {n: provenance.get(n, []) for n in names},
        "constraints": [],
        "note": ("Run 1 fallback covering array (no ACTS/PICT). Constraints not yet wired into "
                 "row generation; impossible combos are a known blind spot for Run 1 "
                 "(see README-analysis.md)."),
        "case_count": len(rows),
    }
    with open(os.path.join(ca_dir, "model.json"), "w") as f:
        json.dump(model, f, indent=2)

    with open(os.path.join(ca_dir, "cases.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id"] + names)
        for idx, row in enumerate(rows):
            w.writerow([idx] + [json.dumps(v) for v in row])

    print(json.dumps({
        "parameters": len(names),
        "cases": len(rows),
        "domain_fragments_read": n_frags,
        "param_names": names,
    }, indent=0))


if __name__ == "__main__":
    main()
