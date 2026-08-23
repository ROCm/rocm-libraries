#!/usr/bin/env python3
"""
Rewrite WorkGroupMapping across an entire Tensile logic file.

WHY. Every one of navi32's 73 shipped TN HHS solutions uses `WorkGroupMapping: 8`, inherited
from navi31 where it is a clean divisor: 48 WGPs / 8 = 6.00. On navi32's 30 WGPs it is
ragged (3.75), and 8 divides neither 30 WGPs nor 60 CUs. The tuning wiki names CU count as
the *only* tuning-relevant difference between navi31 and navi32, and lists the 60-CU factors
as 1,2,3,4,5,6,10,12,15,20,30,60 -- 8 is absent. 6 and 10 are the nearest factors.

This produces a complete, self-consistent logic file per WGM value so each can be built and
benchmarked as a whole library, rather than splicing variants into one file (which would
require remapping the shape table's solution indices).

    python3 wgm_variant.py IN.yaml OUT.yaml --wgm 6 [--isa gfx1100]
"""

import argparse
import collections
import pathlib
import sys

import yaml

ISA_OF = {"gfx1100": [11, 0, 0], "gfx1101": [11, 0, 1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--wgm", type=int, required=True)
    ap.add_argument("--isa", default=None, choices=sorted(ISA_OF))
    args = ap.parse_args()

    doc = yaml.safe_load(pathlib.Path(args.src).open())
    before = collections.Counter(s.get("WorkGroupMapping") for s in doc[5])

    touched = 0
    for sol in doc[5]:
        if "WorkGroupMapping" in sol:
            sol["WorkGroupMapping"] = args.wgm
            touched += 1
            # The solution name encodes WGM as a _WGM<n>_ token. Leaving it stale would make
            # two different kernels share a name, and the benchmark harness identifies
            # kernels BY NAME -- so a stale token silently merges distinct arms.
            for key in ("SolutionNameMin", "SolutionName"):
                if key in sol and isinstance(sol[key], str) and "_WGM" in sol[key]:
                    head, _, tail = sol[key].rpartition("_WGM")
                    rest = tail.lstrip("0123456789")
                    sol[key] = f"{head}_WGM{args.wgm}{rest}"

    if touched != len(doc[5]):
        sys.exit(f"ERROR: {len(doc[5]) - touched} solutions lack WorkGroupMapping")

    if args.isa:
        doc[2] = args.isa
        for sol in doc[5]:
            sol["ISA"] = list(ISA_OF[args.isa])

    pathlib.Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    with pathlib.Path(args.dst).open("w") as fh:
        yaml.safe_dump(doc, fh, default_flow_style=None, width=10**6, sort_keys=False)

    print(f"  {pathlib.Path(args.src).name} -> WGM{args.wgm}")
    print(f"    was: {dict(before)}   now: all {touched} solutions at WGM{args.wgm}")
    print(f"    isa={doc[2]}  -> {args.dst}")


if __name__ == "__main__":
    main()
