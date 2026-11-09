#!/usr/bin/env python3
"""picksH.json (lookback picks: list of {"cell", "canonical", ...}) -> a copy where named cells' fused-MoE knobs carry extra canonical changes.

    mk_picks_moe.py SRC DST "cell:knob=value,...;cell:..."      e.g. "delta.r4s:dn_inflight=1"

gen_tuned.py builds every pick's MoE functions from its CANONICAL changes on the served base (space.moe_adapter), so a knob added here as
``canonical["moe.<knob>"]`` reaches stage_moe_fused through the served code path and nothing else changes (sched groups, labels, B3 knobs).
Values are parsed as int when they look like one, else kept as strings. Refuses an unknown cell and a knob the cell's canonical already sets, unless the knob is written with a
leading ``!`` ("qsa.r4s:!router_knobs=mono"): then the canonical MUST already set it and its value is replaced (reported as ``(was old)``)."""
import json
import sys


def parse_val(v: str):
    try:
        return int(v)
    except ValueError:
        return v


def main(argv) -> int:
    src, dst, spec = argv
    d = json.load(open(src))
    picks = d if isinstance(d, list) else d["picks"]
    by_cell = {p["cell"]: p for p in picks}
    done = []
    for grp in filter(None, spec.split(";")):
        cell, _, body = grp.partition(":")
        if cell not in by_cell:
            raise SystemExit(f"no pick for cell {cell!r} in {src}")
        canon = by_cell[cell]["canonical"]
        for kv in filter(None, body.split(",")):
            k, _, v = kv.partition("=")
            over = k.startswith("!")
            key = "moe." + k.lstrip("!")
            if over != (key in canon):
                raise SystemExit(f"{cell}: {key}: " + (f"override '!' needs a canonical that sets it" if over else f"canonical already sets {key}={canon[key]!r} (write !{k} to override)"))
            old = canon.get(key)
            canon[key] = parse_val(v)
            done.append(f"{cell}:{key}={canon[key]!r}" + (f" (was {old!r})" if over else ""))
    if not done:
        raise SystemExit("no MoE change given")
    json.dump(d, open(dst, "w"), indent=1)
    print(f"mk_picks_moe: {'; '.join(done)} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
