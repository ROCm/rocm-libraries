"""Served functions + schedule string of lookback picks (truemono-quick 3a; q38 container, rocKE on PYTHONPATH).

    python gen_tuned.py --base <e2ecomp2 gen dir> --picks picks.json --out <dir>

``<dir>/k/<kind>/{moe,b3}``: per layer kind, a copy of the hybrid's functions (``e2ecomp2/gen.sh``: BEST_KNOBS MoE,
TUNED_SPECS_NUMA B3) with every pick of that kind (``{"cell": "delta.r4s", "moe_knobs": {...}, "b3_spec": "..."}``,
run.py's PICK line) written over its variant. ``<dir>/sched.txt``: ATOM's ``sched.BEST`` with each pick's group
(``"sched_group": "delta.r4s:..."``) in place of the cell's. Serve with
``ATOM_Q38_EXTERN_STAGES=m3=<dir>/k/{kind}/moe/m3_{variant}.ll,...,b3=<dir>/k/{kind}/b3,b3pre=<dir>/k/{kind}/b3``
and ``ATOM_Q38_SCHED=$(cat <dir>/sched.txt)``. Prints what it wrote; a manifest goes to ``<dir>/tuned.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import space as S  # noqa: E402

KINDS = ("delta", "delta_ple", "qsa")


def moe_knobs(d: dict):
    from kernels.gfx950.qwen38_mono import stage_moe_fused as SMF

    flat, nested = {}, {}
    for k, v in d.items():
        head, _, rest = k.partition(".")
        if rest:
            nested.setdefault(head, {})[rest] = v
        else:
            flat[k] = v
    flat.update(nested)
    return SMF.knobs_from_json(flat)


def sched_with(picks) -> str:
    groups = {c: dict(v) for c, v in S.SCHED_BEST.items()}
    for p in picks:
        cell, _, body = p["sched_group"].partition(":")
        if cell != p["cell"]:
            raise ValueError(f"pick {p['cell']}: sched group of {cell}")
        groups[cell] = dict(kv.split("=", 1) for kv in body.split(",") if kv)
    parts = ["numa=1"]
    for cell, lev in groups.items():
        body = ",".join(f"{k}={v}" for k, v in sorted(lev.items()))
        if body:
            parts.append(f"{cell}:{body}")
    return ";".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flavor", default="llvm23")
    ap.add_argument("--kind-extra", default="", help="kind-level groups appended to the schedule string, e.g. "
                    "'qsa:qdense=3' (P13 N4a; ATOM applies it to every narrow ungrouped build of that kind)")
    a = ap.parse_args(argv)
    from kernels.gfx950.qwen38_mono import stage_b3 as SB
    from kernels.gfx950.qwen38_mono import stage_moe_fused as SMF
    from kernels.gfx950.qwen38_mono.mono_layout import from_atom

    picks = json.load(open(a.picks))
    picks = picks if isinstance(picks, list) else picks["picks"]
    for kind in KINDS:
        for part in ("moe", "b3"):
            d = f"{a.out}/k/{kind}/{part}"
            if os.path.isdir(d):
                shutil.rmtree(d)
            shutil.copytree(f"{a.base}/{part}", d)
    wrote = []
    for p in picks:
        kind, variant, rows, seq = S.parse_cell(p["cell"])
        if S.group_of(variant):
            wrote.append(f"{kind}/{variant}: schedule only (FlyDSL MoE / B3)")
            continue
        # built from the pick's CANONICAL changes on the rebased served set (space.moe_adapter / b3_adapter: numa=1,
        # BEST_KNOBS / TUNED_SPECS_NUMA), never from the flat knob dict (it lacks the fixed numa field)
        canon = p["canonical"]
        kn = S.moe_adapter(rows, seq).build({k[4:]: v for k, v in canon.items() if k.startswith("moe.")})
        if not kn.numa:
            raise ValueError(f"{p['cell']}: the MoE set lost numa=1")
        SMF.write_functions(f"{a.out}/k/{kind}/moe", [variant], kn, arena_words="atom", flavor=a.flavor)
        wrote.append(f"{kind}/moe/{variant}: {kn.suffix()}")
        if kind != "qsa":
            shape = SB.B3Shape(rows, seq, 0)
            kb = S.b3_adapter(variant, rows, seq).build({k[3:]: v for k, v in canon.items() if k.startswith("b3.")})
            if not kb.numa or (p.get("b3_spec") and SB.knobs_spec(kb) != p["b3_spec"]):
                raise ValueError(f"{p['cell']}: B3 set {SB.knobs_spec(kb)} != the pick's {p.get('b3_spec')}")
            SB.write_b3_functions(f"{a.out}/k/{kind}/b3", [shape], lambda s, kb=kb: kb, from_atom(None), flavor=a.flavor)
            wrote.append(f"{kind}/b3/{variant}: {SB.knobs_spec(kb)}")
    sched = sched_with(picks)
    extra = {}
    for grp in filter(None, a.kind_extra.split(";")):
        k, _, body = grp.partition(":")
        extra[k] = dict(kv.split("=", 1) for kv in body.split(",") if kv)
        sched += ";" + grp
    with open(f"{a.out}/sched.txt", "w") as f:
        f.write(sched)
    # the server-log build line each pick must produce (ATOM sched.log_build: levers sorted, then " | extern")
    exp = {"0": [], "1": [], "2": [], "3": []}
    for p in picks:
        kind, variant, prow, pseq = S.parse_cell(p["cell"])
        body = p["sched_group"].partition(":")[2]
        lev = dict(kv.split("=", 1) for kv in body.split(",") if kv)
        g = S.group_of(variant)
        if not g:
            lev.update(extra.get(kind, {}))
        if not (g and prow > 4):  # ATOM drops the numa lever above 4 rows
            lev["numa"] = "1"
        line = f"[ATOM_Q38_SCHED] build {kind} {variant}: " + " ".join(f"{k}={v}" for k, v in sorted(lev.items()))
        line += "" if g else " | extern"  # grouped builds link no functions
        # K of the cell: seq (r<k+1>s) or grouped (g<k+1>) builds belong to K = S - 1, r<n> decode builds to K = 0
        kk = str(g - 1) if g else (str(prow - 1) if pseq else "0")
        exp.setdefault(kk, []).append(line)
    for k, lines in exp.items():
        with open(f"{a.out}/expect_k{k}.txt", "w") as f:
            f.write("".join(x + "\n" for x in lines))
    json.dump({"picks": picks, "sched": sched, "wrote": wrote}, open(f"{a.out}/tuned.json", "w"), indent=1)
    print("gen_tuned:", "; ".join(wrote))
    print("gen_tuned: sched", sched)
    return 0


if __name__ == "__main__":
    sys.exit(main())
