#!/usr/bin/env python3
"""
Retarget a Tensile logic file between gfx1100 (navi31) and gfx1101 (navi32).

WHY THIS EXISTS. The campaign runs in two directions and both need this:

  navi32 -> gfx1100   to RUN navi32's shipped catalog on this gfx1100 card, so it can serve
                      as the measured baseline. Without it the logic declares gfx1101 and
                      the build skips it entirely.
  <any>  -> gfx1101   to BUILD-GATE the final catalog for navi32, proving every shipped
                      solution actually compiles for the real target.

TWO PLACES CARRY THE ISA and missing either produces a file that looks retargeted but is not:
  * top level  element [2]  -- the string the build filters on ("gfx1100" / "gfx1101")
  * EVERY solution in [5]   -- its own `ISA: [11, 0, N]` list
Element [1] is the arch *name* ("navi31"/"navi32"), which must match the directory the build
enumerates.

Solution names (`SolutionNameMin`) do NOT encode the ISA, so they stay valid across a
retarget -- which is what makes the gfx1101 build gate a like-for-like check.

    python3 retarget_logic.py IN.yaml OUT.yaml --isa gfx1101 [--name navi32]
"""

import argparse
import pathlib
import sys

import yaml

ISA_OF = {"gfx1100": [11, 0, 0], "gfx1101": [11, 0, 1]}
NAME_OF = {"gfx1100": "navi31", "gfx1101": "navi32"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--isa", required=True, choices=sorted(ISA_OF))
    ap.add_argument("--name", default=None, help="arch name for element [1]; defaults per --isa")
    ap.add_argument("--device", default=None, help="override the device filter in element [3]")
    args = ap.parse_args()

    doc = yaml.safe_load(pathlib.Path(args.src).open())
    if not isinstance(doc, list) or len(doc) < 12:
        sys.exit(f"{args.src}: not a Tensile logic list of >=12 elements")

    before = (doc[1], doc[2], doc[11], len(doc[5]), len(doc[7]) if doc[7] else 0)

    doc[1] = args.name or NAME_OF[args.isa]
    doc[2] = args.isa
    if args.device:
        doc[3] = [args.device]

    isa = ISA_OF[args.isa]
    touched = 0
    for sol in doc[5]:
        if "ISA" in sol:
            sol["ISA"] = list(isa)
            touched += 1

    # A solution without an ISA key would silently keep the old target.
    if touched != len(doc[5]):
        sys.exit(f"ERROR: {len(doc[5]) - touched} of {len(doc[5])} solutions have no ISA key; "
                 "retarget would be incomplete")

    pathlib.Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    with pathlib.Path(args.dst).open("w") as fh:
        yaml.safe_dump(doc, fh, default_flow_style=None, width=10**6, sort_keys=False)

    print(f"  in : name={before[0]} isa={before[1]} type={before[2]} "
          f"solutions={before[3]} rows={before[4]}")
    print(f"  out: name={doc[1]} isa={doc[2]} type={doc[11]} "
          f"solutions={len(doc[5])} rows={len(doc[7]) if doc[7] else 0}  "
          f"({touched} solution ISA fields rewritten)")
    print(f"  -> {args.dst}")


if __name__ == "__main__":
    main()
