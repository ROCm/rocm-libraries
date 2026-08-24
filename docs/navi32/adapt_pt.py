#!/usr/bin/env python3
"""Adapt the HHS-chosen lean kernel set to another ProblemType.

The four TN catalogs are one tuning adapted per ProblemType (they share 9 680 grid keys
with byte-identical GFlops), so the reduction is done ONCE on HHS and carried across by
matching the full recipe -- tile plus the parameters measurement showed actually matter
(PGR, PLR, ScheduleIterAlg, and which of the pool's two tuning campaigns the kernel is
from). Matching only the tile would let a cross-campaign substitution back in, which is
exactly what measured 2-3x slower on N=1 shapes.

Fails loudly if a ProblemType cannot supply a kernel for a kept recipe, rather than
silently substituting a neighbour.
"""
import argparse, collections, json
import yaml
try:    from yaml import CSafeLoader as Loader
except ImportError: from yaml import SafeLoader as Loader


def sig(s):
    return (s.get("MacroTile0"), s.get("MacroTile1"), s.get("DepthU"),
            tuple(s.get("WorkGroup", [])), tuple(s.get("MIWaveTile", [])),
            s.get("PrefetchGlobalRead"), s.get("PrefetchLocalRead"),
            s.get("ScheduleIterAlg"), s.get("CUOccupancy") == -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-logic", required=True, help="the HHS parent logic")
    ap.add_argument("--src-keep", required=True, help="json list of kept HHS SolutionIndex")
    ap.add_argument("--dst-logic", required=True, help="the ProblemType to adapt to")
    ap.add_argument("--out", required=True, help="json list of kept SolutionIndex for dst")
    a = ap.parse_args()

    src = yaml.load(open(a.src_logic), Loader=Loader)
    dst = yaml.load(open(a.dst_logic), Loader=Loader)
    S = {s["SolutionIndex"]: s for s in src[5]}
    keep = json.load(open(a.src_keep))
    want = {sig(S[i]) for i in keep}

    by = collections.defaultdict(list)
    for s in dst[5]: by[sig(s)].append(s["SolutionIndex"])

    # tiebreak within a matched recipe: the kernel this ProblemType's own grid uses most
    refs = collections.Counter(e[1][0] for e in dst[7])

    missing = sorted(w for w in want if w not in by)
    if missing:
        raise SystemExit(
            f"ADAPTATION FAILED: {len(missing)} kept recipes have no kernel in "
            f"{a.dst_logic}. First: {missing[0]}. Report this as a real difference; "
            f"do not substitute a neighbour.")

    out = {max(by[w], key=lambda si: (refs.get(si, 0), -si)) for w in want}

    # A ProblemType may carry TILES that HHS does not (the aux variants add skinny ones:
    # MT256x16, MT224x32, MT64x160, MT32x224). Rows on such a tile would otherwise fall
    # back to a DIFFERENT macro-tile -- coverage loss, not a scheduling substitution, and
    # the documented catastrophic failure mode. Give every tile a representative.
    def tile(s): return (s.get("MacroTile0"), s.get("MacroTile1"), s.get("DepthU"),
                         tuple(s.get("WorkGroup", [])), tuple(s.get("MIWaveTile", [])))
    grp = collections.defaultdict(list)
    for s in dst[5]: grp[tile(s)].append(s["SolutionIndex"])
    added = []
    for tk, mem in grp.items():
        if not (set(mem) & out):
            pick = max(mem, key=lambda si: (refs.get(si, 0), -si))
            out.add(pick); added.append((tk, refs.get(pick, 0)))
    if added:
        print(f"    + {len(added)} kernels to cover tiles this ProblemType has and HHS does not:")
        for tk, n in added:
            print(f"        MT{tk[0]}x{tk[1]}x{tk[2]} WG{tk[3]} MIWT{tk[4]}  (rows={n})")

    out = sorted(out)
    json.dump(out, open(a.out, "w"))
    print(f"  {a.dst_logic.split('/')[-1]}")
    print(f"    pool {len(dst[5])} -> keeping {len(out)} kernels "
          f"({len(want)} recipes, all matched)   grid rows {len(dst[7])}")


if __name__ == "__main__":
    main()
