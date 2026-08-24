#!/usr/bin/env python3
"""Phase 0 analysis: tile structure, owned-time weighting, reroute candidates.

Produces the candidate set that reroute_probe.py then MEASURES. Nothing here is a
performance claim -- the `gflops` column in the logic table is navi31's tuning-time
measurement, used only to (a) weight kernels by owned time and (b) decide which rows
are above the dispatch floor and therefore worth probing.
"""
import argparse, collections, json, math, sys
import yaml
try:    from yaml import CSafeLoader as Loader
except ImportError: from yaml import SafeLoader as Loader

FLOOR_US = 40.0          # above this, the row is kernel-bound rather than dispatch-bound


def tile_key(s):
    return (s.get("MacroTile0"), s.get("MacroTile1"), s.get("DepthU"),
            tuple(s.get("WorkGroup", [])), tuple(s.get("MIWaveTile", [])))


def implied_us(row):
    (M, N, B, K), (sol, gf) = row[0], row[1]
    if gf <= 0: return float("nan")
    return 2.0 * M * N * K * max(B, 1) / (gf * 1e9) * 1e6


def stratum(row):
    M, N, B, K = row[0]
    lo, hi = min(M, N), max(M, N)
    if lo <= 8:                    return "gemv"
    if hi <= 128:                  return "tiny"
    if N * 4 <= M:                 return "skinny_N"   # N much smaller than M
    if M * 4 <= N:                 return "skinny_M"
    if hi >= 4096:                 return "large"
    return "med"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logic", required=True)
    ap.add_argument("--out", default="/home/vmijovic/navi32/state/lean_candidates.json")
    a = ap.parse_args()

    d = yaml.load(open(a.logic), Loader=Loader)
    assert d[11] == "GridBased", f"not GridBased: {d[11]}"
    sols = {s["SolutionIndex"]: s for s in d[5]}
    tab = d[7]

    rows_us = [implied_us(r) for r in tab]
    owned_time = collections.Counter()
    owned_rows = collections.Counter()
    for r, us in zip(tab, rows_us):
        sol = r[1][0]
        owned_rows[sol] += 1
        if us == us:
            owned_time[sol] += us

    groups = collections.defaultdict(list)
    for si, s in sols.items():
        groups[tile_key(s)].append(si)

    rep_time = {tk: max(m, key=lambda si: (owned_time.get(si, 0.0), owned_rows.get(si, 0), -si))
                for tk, m in groups.items()}
    rep_rows = {tk: max(m, key=lambda si: (owned_rows.get(si, 0), -si))
                for tk, m in groups.items()}
    differs = sum(1 for tk in groups if rep_time[tk] != rep_rows[tk])

    tile_of = {si: tile_key(s) for si, s in sols.items()}
    reroute = {si: rep_time[tile_of[si]] for si in sols}      # strict lean (81 reps)

    # what strict lean would cost, by stratum, weighted by rows and by time
    per_str = collections.defaultdict(lambda: dict(rows=0, rr_rows=0, us=0.0, rr_us=0.0,
                                                   af_rows=0, af_rr=0))
    probe = []
    for r, us in zip(tab, rows_us):
        sol = r[1][0]; st = stratum(r); rr = reroute[sol] != sol
        b = per_str[st]
        b["rows"] += 1; b["rr_rows"] += rr
        if us == us:
            b["us"] += us; b["rr_us"] += us if rr else 0.0
            if us > FLOOR_US:
                b["af_rows"] += 1; b["af_rr"] += rr
                if rr:
                    probe.append(dict(M=r[0][0], N=r[0][1], B=r[0][2], K=r[0][3],
                                      orig=sol, rep=reroute[sol], us=us, stratum=st))

    print(f"logic: {a.logic.split('/')[-1]}")
    print(f"  kernels {len(sols)}  rows {len(tab)}  tiles {len(groups)}  "
          f"strict-lean reps {len(set(rep_time.values()))}")
    print(f"  rep-by-TIME differs from rep-by-ROWS in {differs}/{len(groups)} tiles")
    tot_us = sum(b['us'] for b in per_str.values())
    tot_rr = sum(b['rr_us'] for b in per_str.values())
    print(f"  strict lean reroutes {100*sum(b['rr_rows'] for b in per_str.values())/len(tab):.1f}% of rows, "
          f"{100*tot_rr/tot_us:.1f}% of implied time")
    print()
    print(f"{'stratum':<10}{'rows':>7}{'reroute%':>10}{'time%':>8}{'>40us rows':>12}{'>40us rr%':>11}")
    print("-" * 58)
    for st in ("gemv", "tiny", "skinny_M", "skinny_N", "med", "large"):
        b = per_str.get(st)
        if not b: continue
        print(f"{st:<10}{b['rows']:>7}{100*b['rr_rows']/b['rows']:>9.1f}%"
              f"{100*b['rr_us']/max(b['us'],1e-9):>7.1f}%{b['af_rows']:>12}"
              f"{100*b['af_rr']/max(b['af_rows'],1):>10.1f}%")
    print("-" * 58)
    print(f"  probe candidates (above-floor rows that strict lean reroutes): {len(probe)}")
    pairs = {(p['orig'], p['rep']) for p in probe}
    print(f"  distinct (orig,rep) kernel pairs among them: {len(pairs)}")

    json.dump(dict(logic=a.logic, floor_us=FLOOR_US,
                   tiles={str(k): v for k, v in groups.items()},
                   rep_time={str(k): v for k, v in rep_time.items()},
                   owned_time=dict(owned_time), owned_rows=dict(owned_rows),
                   probe=probe), open(a.out, "w"))
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
