#!/usr/bin/env python3
"""Choose the lean kernel set from MEASURED data, and price every alternative.

Consumes the shape x kernel matrix from matrix_sweep.py. For a kept-set S the catalog
behaves as:  kernel(row) = row's own kernel if kept, else its tile's representative.
So adding a kernel back exactly recovers the loss on the rows that originally used it,
which makes a greedy selection both natural and measurable.

Everything here is priced in MEASURED microseconds on this machine at the target CU
count. The logic file's own `gflops` column is used only to weight sampled shapes back
to their stratum population -- never as a performance number.
"""
import argparse, collections, json, math, os
import yaml
try:    from yaml import CSafeLoader as Loader
except ImportError: from yaml import SafeLoader as Loader


def tile_key(s):
    return (s.get("MacroTile0"), s.get("MacroTile1"), s.get("DepthU"),
            tuple(s.get("WorkGroup", [])), tuple(s.get("MIWaveTile", [])))


def load(logic, matrix):
    d = yaml.load(open(logic), Loader=Loader)
    assert d[11] == "GridBased"
    sols = {s["SolutionIndex"]: s for s in d[5]}
    tab = d[7]
    tile_of = {si: tile_key(s) for si, s in sols.items()}
    groups = collections.defaultdict(list)
    for si in sols: groups[tile_of[si]].append(si)

    # original kernel + implied time per grid row, keyed by shape
    row_sol, row_us = {}, {}
    for r in tab:
        key = (r[0][0], r[0][1], r[0][2], r[0][3])
        gf = r[1][1]
        row_sol[key] = r[1][0]
        row_us[key] = 2.0*key[0]*key[1]*key[3]*max(key[2],1)/(gf*1e9)*1e6 if gf > 0 else float("nan")

    meas = []
    for line in open(matrix):
        try: m = json.loads(line)
        except Exception: continue
        key = (m["M"], m["N"], m["B"], m["K"])
        if key not in row_sol: continue
        gf = {int(k): v for k, v in m["gf"].items()}
        meas.append(dict(key=key, stratum=m["stratum"], gf=gf, orig=row_sol[key]))
    return sols, tab, tile_of, groups, row_sol, row_us, meas


def stratum_of(key):
    M, N, B, K = key; lo, hi = min(M, N), max(M, N)
    if lo <= 8: return "gemv"
    if hi <= 128: return "tiny"
    if N*4 <= M: return "skinny_N"
    if M*4 <= N: return "skinny_M"
    if hi >= 4096: return "large"
    return "med"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logic", required=True)
    ap.add_argument("--matrix", default="/home/vmijovic/navi32/results/P0_matrix.jsonl")
    ap.add_argument("--budgets", default="81,90,100,110,120,140")
    ap.add_argument("--out", default="/home/vmijovic/navi32/state/lean_pick.json")
    a = ap.parse_args()

    sols, tab, tile_of, groups, row_sol, row_us, meas = load(a.logic, a.matrix)
    print(f"  pool {len(sols)}  tiles {len(groups)}  measured shapes {len(meas)}")
    if not meas:
        print("  no measured shapes yet"); return

    # weight each measured shape back to its stratum's above-floor population
    pop = collections.Counter()
    for r in tab:
        key = (r[0][0], r[0][1], r[0][2], r[0][3]); u = row_us.get(key, float("nan"))
        if u == u and u > 40.0: pop[stratum_of(key)] += 1
    samp = collections.Counter(m["stratum"] for m in meas)
    w = {s: pop[s]/samp[s] for s in samp if samp[s]}

    # Representative per tile. Prefer MEASURED owned time; fall back to the table's own
    # owned time for tiles this sweep did not reach, so an unmeasured tile still gets the
    # kernel with the largest tuned footprint rather than an arbitrary one.
    owned = collections.Counter()
    for m in meas:
        g = m["gf"].get(m["orig"], 0)
        if g > 0:
            us = 2.0*m["key"][0]*m["key"][1]*m["key"][3]*max(m["key"][2],1)/(g*1e9)*1e6
            owned[m["orig"]] += us * w[m["stratum"]]
    tbl_owned = collections.Counter()
    for r in tab:
        key = (r[0][0], r[0][1], r[0][2], r[0][3]); u = row_us.get(key, float("nan"))
        if u == u: tbl_owned[r[1][0]] += u
    for si in sols:
        owned.setdefault(si, 0.0); tbl_owned.setdefault(si, 0.0)
    rep = {tk: max(mem, key=lambda si: (owned[si], tbl_owned[si], -si))
           for tk, mem in groups.items()}
    n_meas_tiles = sum(1 for tk, mem in groups.items() if any(owned[si] > 0 for si in mem))
    print(f"  tiles with >=1 measured row: {n_meas_tiles}/{len(groups)}")

    def usec(m, si):
        g = m["gf"].get(si, 0)
        if g <= 0: return None
        return 2.0*m["key"][0]*m["key"][1]*m["key"][3]*max(m["key"][2],1)/(g*1e9)*1e6

    def score(S):
        """weighted measured time for kept-set S, and per-stratum ratios vs full"""
        tot_f = tot_l = 0.0
        per = collections.defaultdict(lambda: [0.0, 0.0])
        for m in meas:
            uf = usec(m, m["orig"])
            k = m["orig"] if m["orig"] in S else rep[tile_of[m["orig"]]]
            ul = usec(m, k)
            if uf is None or ul is None: continue
            ww = w[m["stratum"]]
            tot_f += uf*ww; tot_l += ul*ww
            per[m["stratum"]][0] += uf*ww; per[m["stratum"]][1] += ul*ww
        return tot_f, tot_l, per

    base = set(rep.values())
    print(f"  strict lean = {len(base)} kernels")
    tf, tl, per = score(base)
    print(f"  strict lean measured time exposure: {100*(tl-tf)/tf:+.1f}%  "
          f"(above-floor, weighted)")

    # Greedy add-back, priced in measured weighted microseconds.
    #
    # TWO PHASES, deliberately. Optimising total weighted time alone under-protects small
    # strata: gemv has only 13 above-floor rows against med's 1712, so a 2-3x regression
    # there is invisible in the mean. 87% of grid rows sit in tiles that mix the pool's two
    # tuning campaigns, and a cross-campaign reroute is exactly where those blow-ups happen.
    # So: first spend budget repairing the WORST stratum, only then optimise the mean.
    # ("Select on the tail, not the mean" -- the lesson from the SK3 catalog campaign.)
    TOL = 1.02
    S = set(base); curve = [(len(S), 100*(tl-tf)/tf)]
    cand = [si for si in sols if si not in S]
    budgets = sorted(int(x) for x in a.budgets.split(","))
    picks = {}

    def gain_of(si):
        """weighted us recovered by keeping si instead of rerouting it, and per stratum"""
        tot = 0.0; per_s = collections.Counter()
        for m in meas:
            if m["orig"] != si: continue
            cur = usec(m, rep[tile_of[si]]); new = usec(m, si)
            if cur is None or new is None: continue
            g = (cur - new) * w[m["stratum"]]
            tot += g; per_s[m["stratum"]] += g
        return tot, per_s

    while len(S) < max(budgets) and cand:
        tf, tl, per = score(S)
        worst, wr = None, 1.0
        for s, (f, l) in per.items():
            if f > 0 and l/f > wr: worst, wr = s, l/f
        best = None
        if worst and wr > TOL:                       # phase 1: repair the tail
            bg = 0.0
            for si in cand:
                _, ps = gain_of(si)
                if ps[worst] > bg: best, bg = si, ps[worst]
        if best is None:                             # phase 2: optimise the mean
            bg = 0.0
            for si in cand:
                g, _ = gain_of(si)
                if g > bg: best, bg = si, g
        if best is None: break
        S.add(best); cand.remove(best)
        tf, tl, per = score(S)
        curve.append((len(S), 100*(tl-tf)/tf))
        if len(S) in budgets: picks[len(S)] = sorted(S)

    print(f"\n{'kernels':>8}{'measured exposure vs full':>28}")
    print("-"*38)
    for n, e in curve:
        if n in budgets or n == curve[0][0]:
            print(f"{n:>8}{e:>26.2f}%")
    tf, tl, per = score(set(picks.get(100, sorted(S)[:100])))
    print(f"\nper-stratum at J=100 (measured ratio full/lean, >1 = lean slower):")
    for s in ("gemv","tiny","skinny_M","skinny_N","med","large"):
        if s in per and per[s][0]:
            print(f"  {s:<10}{per[s][1]/per[s][0]:>8.4f}")
    json.dump(dict(rep={str(k): v for k, v in rep.items()}, curve=curve,
                   picks={str(k): v for k, v in picks.items()}), open(a.out, "w"))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
