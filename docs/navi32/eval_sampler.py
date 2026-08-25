#!/usr/bin/env python3
"""Draw an evaluation set that can actually SEE a coverage extension -- with no workload prior.

WHY A NEW EVAL SET IS REQUIRED
------------------------------
The existing 207-shape set lands 100% on rows that were already measured, so extending coverage
to the other 7541 rows is invisible to it *by construction*. Re-using it would have produced a
confident "no effect" that measured nothing.

WHY EQUAL WEIGHT PER ROW
------------------------
There is no defensible prior over query shapes. A previous attempt ranked rows by how often a
log-uniform sampler hit them; that ranks rows by a distribution I invented, and tuning against it
would be overfitting dressed up as coverage. Equal weight per row is the honest default precisely
because it declines to assert which shapes matter.

WHY OFF-LATTICE
---------------
0 of the 207 old eval queries landed on a grid key exactly, so GridBased is never lookup in
practice -- it is always snap-up from between lattice points. An eval set sitting exactly on keys
would measure the easy case and overstate the gain. So each query is jittered strictly *within
the catchment* of its target row and then CHECKED, by running the selection rule, that it still
resolves to that row. A query that jitters out of its catchment is silently testing a different
row, which is exactly the kind of bug that looks like a result.

SELECTION RULE (verified against the C++): staged lower_bound on M, then N clamped within the
M-block, then batch; each snapping UPWARD. Ties broken by |K - gridK|. M and N are not in the
distance metric.
"""
import argparse, bisect, collections, hashlib, json, random
import yaml
try:    from yaml import CSafeLoader as Loader
except ImportError: from yaml import SafeLoader as Loader


def stratum(M, N):
    lo, hi = min(M, N), max(M, N)
    if lo <= 8:      return "gemv"
    if hi <= 128:    return "tiny"
    if N * 4 <= M:   return "skinny_N"
    if M * 4 <= N:   return "skinny_M"
    if hi >= 4096:   return "large"
    return "med"


class Grid:
    """The shipped selection rule, reimplemented so queries can be verified rather than assumed."""

    def __init__(self, tab):
        self.rows = [tuple(e[0]) for e in tab]
        self.by_m = collections.defaultdict(lambda: collections.defaultdict(
            lambda: collections.defaultdict(list)))
        for i, (M, N, B, K) in enumerate(self.rows):
            self.by_m[M][N][B].append((K, i))
        self.ms = sorted(self.by_m)
        self.ns = {m: sorted(self.by_m[m]) for m in self.ms}
        self.bs = {(m, n): sorted(self.by_m[m][n]) for m in self.ms for n in self.ns[m]}
        for m in self.ms:
            for n in self.ns[m]:
                for b in self.by_m[m][n]:
                    self.by_m[m][n][b].sort()

    @staticmethod
    def _snap(vals, q):
        """first value >= q, clamped to the last if q exceeds every entry"""
        i = bisect.bisect_left(vals, q)
        return vals[i] if i < len(vals) else vals[-1]

    def select(self, m, n, b, k):
        M = self._snap(self.ms, m)
        N = self._snap(self.ns[M], n)
        B = self._snap(self.bs[(M, N)], b)
        cand = self.by_m[M][N][B]
        return min(cand, key=lambda kk: (abs(kk[0] - k), kk[0]))[1]

    def catchment(self, idx, rng):
        """A query strictly inside row `idx`'s catchment: at or just below its key on each axis."""
        M, N, B, K = self.rows[idx]
        i = self.ms.index(M)
        lo_m = self.ms[i - 1] + 1 if i else 1
        ns = self.ns[M]; j = ns.index(N)
        lo_n = ns[j - 1] + 1 if j else 1
        bs = self.bs[(M, N)]; z = bs.index(B)
        lo_b = bs[z - 1] + 1 if z else 1
        # K is nearest-match, not snap-up: stay inside the half-way band to the neighbours
        ks = sorted(kk for kk, _ in self.by_m[M][N][B])
        p = ks.index(K)
        lo_k = (ks[p - 1] + K) // 2 + 1 if p else max(1, K // 2)
        hi_k = (ks[p + 1] + K - 1) // 2 if p + 1 < len(ks) else K * 2
        return (rng.randint(lo_m, M), rng.randint(lo_n, N),
                rng.randint(lo_b, B), rng.randint(max(1, lo_k), max(1, hi_k)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logic", default="/home/vmijovic/navi32/arms/hhs_lean100/x.yaml")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out", default="/home/vmijovic/navi32/state/eval_fullcov.json")
    ap.add_argument("--holdout-out",
                    default="/home/vmijovic/navi32/state/holdout_fullcov.json")
    a = ap.parse_args()

    d = yaml.load(open(a.logic), Loader=Loader)
    assert d[11] == "GridBased"
    g = Grid(d[7])
    # per-row time estimate from the table's own recorded GFlop/s; used only to pick an
    # iteration count, never reported as a result
    est_us = [2 * e[0][0] * e[0][1] * e[0][3] * max(1, e[0][2]) / (e[1][1] * 1e3)
              if e[1][1] > 0 else 0.0 for e in d[7]]

    # Self-test: every row's own key must resolve to that row. This is what makes the claim
    # "every row is reachable" a measurement rather than an assertion.
    bad = [i for i in range(len(g.rows)) if g.select(*g.rows[i]) != i]
    assert not bad, f"{len(bad)} rows do not resolve to themselves, e.g. {bad[:5]}"
    print(f"  self-test: all {len(g.rows)} rows resolve to themselves under the selection rule")

    # Treated / held-out split of ROWS, decided before any re-map, by a stable hash of the key.
    treated, held = [], []
    for i, key in enumerate(g.rows):
        h = hashlib.sha256(repr(key).encode()).hexdigest()
        (treated if int(h[:8], 16) % 2 == 0 else held).append(i)
    print(f"  split: {len(treated)} treated rows / {len(held)} held out")

    rng = random.Random(a.seed)
    shapes, drops = [], 0
    for label, pool in (("treated", treated), ("control", held)):
        picks = rng.sample(pool, min(a.n // 2, len(pool)))
        for idx in picks:
            for _ in range(20):
                m, n, b, k = g.catchment(idx, rng)
                if g.select(m, n, b, k) == idx:      # verified, not assumed
                    R = g.rows[idx]
                    shapes.append({"M": m, "N": n, "B": b, "K": k,
                                   "stratum": stratum(m, n), "row": idx,
                                   # the ROW's stratum can differ from the QUERY's: jitter is
                                   # always downward, so a large row is often served a med
                                   # query. Reporting both keeps that visible instead of
                                   # letting one silently stand in for the other.
                                   "row_stratum": stratum(R[0], R[1]),
                                   "row_key": list(R), "arm": label})
                    break
            else:
                drops += 1

    exact = sum(1 for s in shapes if tuple(s["row_key"]) == (s["M"], s["N"], s["B"], s["K"]))
    print(f"  {len(shapes)} queries ({drops} rows dropped: no verifiable catchment sample)")
    print(f"  landing exactly on a grid key: {exact} ({100*exact/len(shapes):.0f}%)")
    for tag in ("stratum", "row_stratum"):
        by = collections.Counter(s[tag] for s in shapes)
        thin = [k for k, v in by.items() if v < 15]
        print(f"  by {tag:<12}" + "  ".join(f"{k}={v}" for k, v in sorted(by.items())))
        if thin:
            print(f"    NOT CLAIMABLE (n<15, resolution is +-2.5% at n~10): {sorted(thin)}")

    # bench_arms.py expects {"shapes":[...]} with a stable shape_id, and derives its iteration
    # count from est_us (see --target-us there): a fixed count would put the small end of this
    # set back under the 8-10% cold noise floor that time-derived iterations were adopted to fix.
    for s in shapes:
        s["shape_id"] = "cov-" + hashlib.sha256(
            f"{s['M']}x{s['N']}x{s['B']}x{s['K']}".encode()).hexdigest()[:24]
        s["est_us"] = est_us[s["row"]]
    json.dump({"count": len(shapes), "seed": a.seed,
               "source": "equal-weight-per-row, jittered off-lattice, catchment-verified",
               "shapes": shapes}, open(a.out, "w"))
    json.dump([list(g.rows[i]) for i in held], open(a.holdout_out, "w"))
    print(f"  wrote {a.out} and {a.holdout_out}")


if __name__ == "__main__":
    main()
