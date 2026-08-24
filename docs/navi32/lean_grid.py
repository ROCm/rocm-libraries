#!/usr/bin/env python3
"""Emit a lean-Grid catalog: keep a chosen kernel set, REROUTE every grid row, delete none.

Follows /home/vmijovic/lengrid_plan.md with two deliberate deviations, both measured:

  1. The representative is chosen by MEASURED owned time on the target, not by the source
     SKU's table row counts. Reps picked from navi31's tuning numbers measured a median
     0.72x on this machine (25-28% loss) because they route PGR=0/PLR=0 rows onto
     PGR=2/PLR=1 kernels; reps picked from measurement do not.
  2. Budget is filled tail-first (repair the worst stratum, then the mean), because 87% of
     rows sit in tiles mixing the pool's two tuning campaigns and cross-campaign reroutes
     are where the blow-ups are.

Selection is done by lean_select.py; this script only applies it and asserts the invariants.
"""
import argparse, collections, copy, json, math, os, sys
import yaml
try:    from yaml import CSafeLoader as Loader, CSafeDumper as Dumper
except ImportError: from yaml import SafeLoader as Loader, SafeDumper as Dumper


def tile_key(s):
    return (s.get("MacroTile0"), s.get("MacroTile1"), s.get("DepthU"),
            tuple(s.get("WorkGroup", [])), tuple(s.get("MIWaveTile", [])))


def logtile(s):
    return (math.log2(max(1, s.get("MacroTile0", 1))),
            math.log2(max(1, s.get("MacroTile1", 1))),
            math.log2(max(1, s.get("DepthU", 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logic", required=True)
    ap.add_argument("--keep", required=True,
                    help="json: either a list of SolutionIndex, or {'tiles':[tile_key strs]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--identity", action="store_true",
                    help="null surgery: keep every kernel, still round-trip through this "
                         "tooling. The control that separates lean from serialization.")
    a = ap.parse_args()

    d = yaml.load(open(a.logic), Loader=Loader)
    assert d[11] == "GridBased", f"element[11] is {d[11]!r}, not GridBased"
    sols = {s["SolutionIndex"]: s for s in d[5]}
    tab = d[7]
    n_rows_in = len(tab)

    if a.identity:
        keep = sorted(sols)
    else:
        spec = json.load(open(a.keep))
        if isinstance(spec, dict) and "tiles" in spec:
            want = set(spec["tiles"])
            keep = sorted(si for si, s in sols.items() if str(tile_key(s)) in want)
        else:
            keep = sorted(spec)
    keep_set = set(keep)
    assert keep_set <= set(sols), "keep-set references unknown SolutionIndex"

    # every tile must retain at least one member, else rows fall to a foreign tile
    groups = collections.defaultdict(list)
    for si, s in sols.items(): groups[tile_key(s)].append(si)
    rep_of_tile = {}
    orphan = []
    for tk, mem in groups.items():
        kept = [si for si in mem if si in keep_set]
        if kept:
            rep_of_tile[tk] = kept[0]
        else:
            orphan.append(tk)
    if orphan:
        # reroute an orphaned tile to the nearest SURVIVING tile in log(MT0,MT1,DU)
        alive = [tk for tk in groups if tk in rep_of_tile]
        for tk in orphan:
            probe = sols[groups[tk][0]]
            near = min(alive, key=lambda t: sum(
                (x-y)**2 for x, y in zip(logtile(probe), logtile(sols[groups[t][0]]))))
            rep_of_tile[tk] = rep_of_tile[near]
        print(f"  WARNING {len(orphan)} tiles had no kept member; rerouted to nearest tile")

    # map every kernel -> a surviving kernel: itself if kept, else its tile's representative
    to_keep = {si: (si if si in keep_set else rep_of_tile[tile_key(s)])
               for si, s in sols.items()}
    remap = {old: new for new, old in enumerate(keep)}

    new_sols = []
    for old in keep:
        s = copy.deepcopy(sols[old]); s["SolutionIndex"] = remap[old]
        new_sols.append(s)
    new_tab = []
    for e in tab:
        ne = copy.deepcopy(e)
        ne[1][0] = remap[to_keep[e[1][0]]]
        new_tab.append(ne)

    # ---- invariants. These are asserts, not checks-after-the-fact: a bug that silently
    # ---- drops grid rows would look exactly like a successful lean.
    assert len(new_tab) == n_rows_in, \
        f"GRID SHRANK: {n_rows_in} -> {len(new_tab)} rows. Rows must be rerouted, never deleted."
    assert all(0 <= e[1][0] < len(new_sols) for e in new_tab), \
        "a grid row points outside the kept kernel set"
    assert {e[0] for e in map(lambda r: (tuple(r[0]),), tab)} == \
           {e[0] for e in map(lambda r: (tuple(r[0]),), new_tab)}, \
        "grid KEYS changed; only the solution index may be rewritten"

    out = list(d); out[5] = new_sols; out[7] = new_tab

    class NoAlias(Dumper):
        def ignore_aliases(self, data): return True

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        yaml.dump(out, fh, Dumper=NoAlias, default_flow_style=None, width=1_000_000)

    rerouted = sum(1 for e, ne in zip(tab, new_tab) if remap.get(e[1][0]) != ne[1][0])
    print(f"  wrote {a.out}")
    print(f"    kernels {len(sols)} -> {len(new_sols)}   grid rows {n_rows_in} -> {len(new_tab)} (unchanged)")
    print(f"    rows rerouted to a different kernel: {rerouted} ({100*rerouted/len(new_tab):.1f}%)")


if __name__ == "__main__":
    main()
