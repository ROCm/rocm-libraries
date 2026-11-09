"""WP2b: write the rocKE GEMV-family functions (``qwen38_mono/stages2/gemv.py``) next to a served stack's functions.

    python wp2b_gen.py --stk <gen_tuned out dir> --out <dir> [--variants r1,r4s,...] [--knobs k=v,...]
                       [--knobs-json picks.json]

For every layer kind and variant: ``<out>/k/<kind>/gemv/<stage>_<kind>_<variant>.{ll,json}`` for the stages the kind
runs (B1 / C1 in delta, B1 / C1 / P1 in delta_ple, Q1 / C1 in qsa), built for the stack's own protocol:

- ``numa=1`` (the stack's ``numa=1`` lever: every function addresses ATOM's NUMA area);
- the poll pause of the stack's ``ATOM_Q38_SCHED`` (``<stk>/sched.txt``) for that build (``pause.<stage>`` /
  ``pause``), as the FlyDSL stage would poll;
- B1 under a B3 whose sidecar knobs have ``pre=dma`` (``<stk>/k/<kind>/b3/b3_<variant>.json``): ``host_b3pre`` = that
  spec (the FlyDSL B1 calls ``b3pre``; the rocKE B1 emits the same prologue).

``--knobs``: GemvKnobs on top (every stage); ``--knobs-json``: per cell / stage knob sets (``{"delta.r4s": {"c1":
"asop=1,..."}}``). Serve with ``ATOM_Q38_EXTERN_STAGES=...,b1=<out>/k/{kind}/gemv/b1_{kind}_{variant}.ll,c1=...``
(``extern_spec``). Prints the spec string.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace


def applicable(stage: str, rows: int, group: int, ks: str) -> str:
    """``ks`` without the knobs this stage / build does not take: ``bsop`` / ``store_out`` are C1's, ``cprio`` is
    not C1's, ``epi`` is B1's / Q1's in ungrouped builds of up to 4 rows (one knob set can go to every function)."""
    out = []
    for x in filter(None, (ks or "").split(",")):
        k = x.split("=")[0]
        if k in ("bsop", "store_out") and stage != "c1":
            continue
        if k == "cprio" and stage == "c1":
            continue
        if k == "epi" and (stage not in ("b1", "q1") or group or rows > 4):
            continue
        out.append(x)
    return ",".join(out)


def extern_spec(out: str, stages=("b1", "c1", "q1", "p1")) -> str:
    return ",".join(f"{s}={out}/k/{{kind}}/gemv/{s}_{{kind}}_{{variant}}.ll" for s in stages)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stk", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="r1,r2,r3,r4,r2s,r3s,r4s")
    ap.add_argument("--stages", default="b1,c1,q1,p1")
    ap.add_argument("--knobs", default="")
    ap.add_argument("--knobs-json", default=None)
    ap.add_argument("--json-only", action="store_true",
                    help="write only the (kind, variant) cells --knobs-json lists (the others keep the FlyDSL stage)")
    ap.add_argument("--flavor", default=os.environ.get("ROCKE_LLVM_FLAVOR", "llvm23"))
    a = ap.parse_args(argv)

    from atom.models.qwen4_exp_mono.kernels import sched as SCH

    from kernels.gfx950.qwen38_mono.stages2 import gemv as G

    sched = open(os.path.join(a.stk, "sched.txt")).read().strip()
    per_cell = json.load(open(a.knobs_json)) if a.knobs_json else {}
    g = G.geom_from_atom()
    written = {}
    for kind in ("delta", "delta_ple", "qsa"):
        shapes, knobs = [], {}
        for v in a.variants.split(","):
            if a.json_only and f"{kind}.{v}" not in per_cell:
                continue
            rows, seq, group = G.parse_variant(v)
            levers = dict(SCH.parse_levers(SCH.select(sched, kind, v)))
            b3side = os.path.join(a.stk, "k", kind, "b3", f"b3_{v}.json")
            b3spec = json.load(open(b3side)).get("stage_knobs", "") if os.path.isfile(b3side) else ""
            for st in a.stages.split(","):
                if kind not in G.STAGES[st]:
                    continue
                ks = per_cell.get(f"{kind}.{v}", {}).get(st, a.knobs)
                # knobs another stage owns (``bsop`` / ``store_out`` are C1's) are dropped for this stage
                ks = applicable(st, rows, group, ks)
                k = G.knobs_from_spec(ks)
                pause = levers.get(f"pause.{st}", levers.get("pause", 0))
                pause = k.pause if "pause=" in ks else int(pause)
                k = replace(k, numa=1 if rows <= 4 else 0, pause=pause)  # ATOM drops numa above 4 rows
                if st == "b1" and "pre=dma" in b3spec:
                    k = replace(k, host_b3pre=b3spec)
                sh = G.GemvShape(st, kind, rows, seq, group)
                shapes.append(sh)
                knobs[sh] = k
        d = os.path.join(a.out, "k", kind, "gemv")
        G.write_gemv_functions(d, shapes, lambda s: knobs[s], g, flavor=a.flavor)
        written[kind] = {s.stem: G.knobs_spec(k) + (f" host_b3pre={k.host_b3pre}" if k.host_b3pre else "")
                         for s, k in knobs.items()}  # fmt: skip
    with open(os.path.join(a.out, "gemv.json"), "w") as f:
        json.dump({"sched": sched, "functions": written}, f, indent=1)
    print("WP2B_GEN", sum(len(v) for v in written.values()), "functions")
    print("EXTERN", extern_spec(a.out, a.stages.split(",")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
