#!/usr/bin/env python3
"""Emit EVERY grid row of the TN HHS catalog as a sweep shape, in stratified round-robin order.

WHY EVERY ROW, AND NOT A RANKED SUBSET
--------------------------------------
An earlier version of this campaign proposed ranking rows by how often a log-uniform random
(M,N,K) sampler landed on them, and sweeping only the high-traffic head. That was wrong, and
checking it rather than asserting it is what showed why:

    all 9680 keys are distinct, and within every one of the 1067 (M,N,B) groups the rows
    differ only in K

so a query at a row's exact key resolves to that row uniquely. EVERY ROW IS REACHABLE. The
"38.8% of rows are ever reached" number measured the hit rate of a sampler I invented, not a
property of the grid -- and ranking by it would have tuned the catalog to a synthetic workload
while calling it coverage. There is no defensible prior over query shapes here, so this asserts
none: it measures all of them.

ORDER IS ORDER ONLY
-------------------
The round-robin over (stratum x size-decade) cells cannot bias the result, because every row is
measured either way. Its only job is that an interrupted or still-running sweep leaves coverage
spread evenly over shape space, rather than concentrated wherever table order happens to point
(table order is sorted by M, so a truncated in-order sweep would know everything about small M
and nothing about large).

est_us comes from the table's own recorded GFlop/s. It is a navi31 96-CU warm number and so
underestimates cold time at 60 CU -- which is the safe direction: it buys MORE iterations than
strictly needed, never fewer. It is used only to pick an iteration count, never as a result.
"""
import argparse, collections, json, math
import yaml
try:    from yaml import CSafeLoader as Loader
except ImportError: from yaml import SafeLoader as Loader


def stratum(M, N):
    """Same definition used by the re-map and every report in this campaign."""
    lo, hi = min(M, N), max(M, N)
    if lo <= 8:      return "gemv"
    if hi <= 128:    return "tiny"
    if N * 4 <= M:   return "skinny_N"
    if M * 4 <= N:   return "skinny_M"
    if hi >= 4096:   return "large"
    return "med"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logic", default="/home/vmijovic/navi32/arms/hhs_lean100/x.yaml")
    ap.add_argument("--out", default="/home/vmijovic/navi32/state/full_grid_shapes.json")
    a = ap.parse_args()

    d = yaml.load(open(a.logic), Loader=Loader)
    assert d[11] == "GridBased", f"element[11] is {d[11]!r}"
    tab = d[7]

    keys = [tuple(e[0]) for e in tab]
    assert len(set(keys)) == len(keys), "duplicate grid keys -- exact-key reachability is not 1:1"

    cells = collections.defaultdict(list)
    for e in tab:
        M, N, B, K = e[0]
        gf = e[1][1]
        flops = 2 * M * N * K * max(1, B)
        est = flops / (gf * 1e3) if gf > 0 else 0.0
        st = stratum(M, N)
        decade = int(math.log10(max(flops, 1)))          # size band
        cells[(st, decade)].append(
            {"M": M, "N": N, "B": B, "K": K, "stratum": st, "est_us": est})

    # round-robin across cells, so any prefix of the output is a balanced sample of shape space
    order = sorted(cells)
    for c in order:
        cells[c].sort(key=lambda s: (s["M"], s["N"], s["B"], s["K"]))
    out, i = [], 0
    while any(cells[c] for c in order):
        for c in order:
            if cells[c]:
                out.append(cells[c].pop(0))
        i += 1

    assert len(out) == len(tab), f"row loss: {len(tab)} -> {len(out)}"
    assert {(s['M'], s['N'], s['B'], s['K']) for s in out} == set(keys), "key set changed"

    json.dump(out, open(a.out, "w"))
    print(f"  wrote {a.out}: {len(out)} shapes over {len(order)} (stratum x decade) cells")
    by = collections.Counter(s["stratum"] for s in out)
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(by.items())))
    print(f"  first 12 strata (round-robin check): "
          f"{[s['stratum'] for s in out[:12]]}")


if __name__ == "__main__":
    main()
