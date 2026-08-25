#!/usr/bin/env python3
"""Re-map a GridBased catalog's element[7]: point each row at its MEASURED-best kernel.

Only the solution INDEX of a row changes. Keys, row count and ordering are untouched -- the
same invariant the lean reduction had, and asserted the same way, because a bug that dropped or
reordered rows would look exactly like a successful re-map.

Why this exists: the shipped grid was tuned on navi31 at 96 CU. Measured cold at 60 CU, its own
picks reach only ~78% of what the same pool can do -- and ~12% on tiny shapes. No new kernels
are needed; the grid just points at the wrong one.

TWO THINGS THIS DELIBERATELY DOES NOT DO:

  * It does not re-map rows it has no measurement for. Those keep their existing index. A
    guessed index is worse than an inherited one.
  * It does not optimise each row in isolation and call that a win. Selection snaps UPWARD on
    (M,N) and clamps N within the M-block, so a row is consulted by queries BETWEEN lattice
    points, not just its own shape. Per-row optimisation is a local optimum for the region the
    row serves; --holdout exists to measure how much of it survives on unseen queries.
"""
import argparse, collections, copy, json, os
import yaml
try:    from yaml import CSafeLoader as Loader, CSafeDumper as Dumper
except ImportError: from yaml import SafeLoader as Loader, SafeDumper as Dumper


def load_matrix(path, keep_pool):
    """shape -> (per-kernel GFlop/s restricted to keep_pool, stratum, iters).

    Keeps the whole row, not just the argmax, so the gain of a candidate repoint can be
    computed against the kernel the grid currently uses.
    """
    out = {}
    for line in open(path):
        try: m = json.loads(line)
        except Exception: continue
        gf = {int(k): v for k, v in m["gf"].items() if v > 0 and int(k) in keep_pool}
        if not gf: continue
        out[(m["M"], m["N"], m["B"], m["K"])] = (gf, m.get("stratum", "?"), m.get("iters", 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logic", required=True, help="the LEAN catalog to re-map")
    ap.add_argument("--matrix", required=True, help="cold measurement jsonl")
    ap.add_argument("--src-pool", required=True,
                    help="the FULL logic the matrix was measured against (index space)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", default=None,
                    help="json list of shape keys to EXCLUDE from re-mapping (the test half)")
    ap.add_argument("--robust", action="store_true",
                    help="Plateau-aware pick. Measured: the top-5 kernels sit within ~2%% of each "
                         "other and on 73%% of rows the winner is not even clear of the noise "
                         "floor, so the per-row argmax is largely a coin-flip among near-ties. "
                         "This instead prefers, among kernels within --plateau of the row's best, "
                         "the one most often near-best ACROSS the stratum -- a broadly good kernel "
                         "should survive the off-lattice snap better than a per-row winner.")
    ap.add_argument("--plateau", type=float, default=0.02)
    ap.add_argument("--skip-strata", default="",
                    help="comma-separated strata to leave untouched. MEASURED: re-mapping "
                         "tiny (64.8%% of baseline) and gemv (94.9%%) REGRESSES them badly -- "
                         "their kernel time sits at the dispatch floor, so the 'best kernel' at "
                         "a grid key is chosen from near-noise and does not transfer to the "
                         "smaller queries that snap to it. Gating them out costs ~0.2pt of "
                         "aggregate wall-clock and removes the tail regression.")
    ap.add_argument("--min-gain", type=float, default=0.02,
                    help="only re-map a row if the measured gain exceeds this, to avoid "
                         "churning rows for differences inside the noise floor")
    a = ap.parse_args()

    lean = yaml.load(open(a.logic), Loader=Loader)
    src = yaml.load(open(a.src_pool), Loader=Loader)
    assert lean[11] == "GridBased", f"element[11] is {lean[11]!r}"
    n_rows_in = len(lean[7])

    # The matrix was measured against the FULL pool's index space; the lean catalog re-indexes
    # 0..N-1. Map between them by kernel identity (KernelNameMin), not by index.
    src_name = {s["SolutionIndex"]: s.get("KernelNameMin") for s in src[5]}
    lean_by_name = {s.get("KernelNameMin"): s["SolutionIndex"] for s in lean[5]}
    src_to_lean = {si: lean_by_name[n] for si, n in src_name.items() if n in lean_by_name}
    keep_pool = set(src_to_lean)
    lean_to_src = {v: k for k, v in src_to_lean.items()}   # precomputed; the inverse lookup
                                                            # was O(n) inside the row loop
    print(f"  lean pool {len(lean[5])} kernels; {len(keep_pool)} of the full pool map into it")

    best = load_matrix(a.matrix, keep_pool)
    print(f"  measured shapes usable: {len(best)}")

    # popularity = how often a kernel lands inside the plateau of a row's best, per stratum.
    pop = collections.defaultdict(collections.Counter)
    if a.robust:
        for (gf_src, st, _) in best.values():
            top = max(gf_src.values())
            for si, v in gf_src.items():
                if v >= top * (1 - a.plateau): pop[st][si] += 1
        print(f"  robust mode: plateau={100*a.plateau:.0f}%, "
              f"popularity tallied over {len(best)} rows")

    hold = set()
    if a.holdout:
        hold = {tuple(k) for k in json.load(open(a.holdout))}
        print(f"  holdout: {len(hold)} shapes excluded from re-mapping")

    skip = {x.strip() for x in a.skip_strata.split(",") if x.strip()}
    if skip: print(f"  leaving untouched: {sorted(skip)}")
    tab = lean[7]
    changed = skipped_hold = skipped_gain = unmeasured = skipped_strat = 0
    per = collections.Counter()
    new_tab = []
    for e in tab:
        ne = copy.deepcopy(e)
        key = tuple(e[0]); cur = e[1][0]
        if key in hold:
            skipped_hold += 1; new_tab.append(ne); continue
        hit = best.get(key)
        if not hit:
            unmeasured += 1; new_tab.append(ne); continue
        gf_src, st, _ = hit
        if st in skip:
            skipped_strat += 1; new_tab.append(ne); continue
        # translate the measured row into the lean index space
        gf = {src_to_lean[si]: v for si, v in gf_src.items()}
        if a.robust:
            top = max(gf.values())
            near = [si for si, v in gf.items() if v >= top * (1 - a.plateau)]
            # among near-ties on this row, take the one most often near-best in this stratum;
            # break remaining ties by this row's own measurement
            tgt = max(near, key=lambda si: (pop[st].get(lean_to_src[si], 0), gf[si]))
        else:
            tgt = max(gf, key=gf.get)
        if tgt == cur:
            new_tab.append(ne); continue
        # Only repoint if the measured gain clears --min-gain. The cold noise floor is ~0.7%
        # median / ~2% p90, so repointing for a 0.5% difference is churn, not improvement.
        cur_gf = gf.get(cur)
        if cur_gf and gf[tgt] / cur_gf - 1.0 < a.min_gain:
            skipped_gain += 1; new_tab.append(ne); continue
        ne[1][0] = tgt; changed += 1; per[st] += 1
        new_tab.append(ne)

    assert len(new_tab) == n_rows_in, \
        f"GRID SHRANK: {n_rows_in} -> {len(new_tab)}. Rows may be repointed, never dropped."
    assert all(0 <= e[1][0] < len(lean[5]) for e in new_tab), "row points outside the pool"
    assert [tuple(e[0]) for e in tab] == [tuple(e[0]) for e in new_tab], \
        "grid KEYS changed; only the solution index may be rewritten"

    out = list(lean); out[7] = new_tab

    class NoAlias(Dumper):
        def ignore_aliases(self, data): return True
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        yaml.dump(out, fh, Dumper=NoAlias, default_flow_style=None, width=1_000_000)

    print(f"  wrote {a.out}")
    print(f"    rows {n_rows_in} (unchanged)   repointed {changed} ({100*changed/n_rows_in:.1f}%)")
    print(f"    left alone: {unmeasured} unmeasured, {skipped_hold} held out, "
          f"{skipped_gain} below the {100*a.min_gain:.0f}% gain threshold, "
          f"{skipped_strat} in skipped strata")
    if per:
        print("    repointed by stratum: " + "  ".join(f"{k}={v}" for k, v in sorted(per.items())))


if __name__ == "__main__":
    main()
