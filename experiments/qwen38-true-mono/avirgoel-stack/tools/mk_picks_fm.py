#!/usr/bin/env python3
"""picks_final.json -> a copy whose cells' HC stage knob strings (hc_knobs a1/a2/m1/m2) carry extra knobs (e.g. invmath=rsq32,sigm=hw).

    mk_picks_fm.py SRC DST EXTRA [KINDS]

KINDS (optional, comma list of layer kinds, e.g. "delta" or "delta,qsa"): only cells of those kinds (the part of the cell name before the
dot: delta.r4s -> delta, delta_ple.r4s -> delta_ple) get the extra knobs; absent or empty = every cell (the broad build).
Nothing else changes (sched strings, labels, canonical, ...). Refuses a stage that already names one of the extra keys, and a KINDS entry
that names no cell."""
import json
import sys

src, dst, extra = sys.argv[1:4]
kinds = {k for k in (sys.argv[4] if len(sys.argv) > 4 else "").split(",") if k}
keys = {kv.split("=")[0] for kv in extra.split(",")}
d = json.load(open(src))
missing = kinds - {cell.split(".")[0] for cell in d}
if missing:
    raise SystemExit(f"no cell of kind {sorted(missing)} in {src}")
n, cells = 0, []
for cell, p in d.items():
    if kinds and cell.split(".")[0] not in kinds:
        continue
    cells.append(cell)
    for st, spec in p["hc_knobs"].items():
        if keys & {kv.split("=")[0] for kv in spec.split(",") if kv}:
            raise SystemExit(f"{cell}.{st} already sets one of {keys}")
        p["hc_knobs"][st] = spec + "," + extra
        n += 1
json.dump(d, open(dst, "w"), indent=1)
print(f"mk_picks_fm: {n} stage strings of {len(cells)} cells ({','.join(cells)}) + {extra} -> {dst}")
