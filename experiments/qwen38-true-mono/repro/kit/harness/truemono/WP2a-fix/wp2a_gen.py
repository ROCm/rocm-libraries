"""WP2a fix round: write the rocKE HC stage functions (``qwen38_mono/stages2/hc.py``) of the WP2a picks next to a
served stack's functions, for the quick end-to-end (PLAN decision 14).

    python wp2a_gen.py --stk <gen_tuned out dir> --picks picks_final.json [--cells delta.r1,...] [--port]

For every pick cell ``<kind>.<variant>``: ``<stk>/hc/<kind>/<stage>_<variant>.{ll,json}`` for A1 / A2 / M1 / M2 with
the pick's resolved per-stage ``HcKnobs`` (the exact strings the in-situ confirmation measured), and the cell's group of
``<stk>/sched.txt`` replaced by the pick's (its HC schedule levers: hold / prio / head_sleep / spread). ``--port``:
the FlyDSL-choice defaults instead (the port arm; the stack's schedule and its spread levers kept, numa=1, pre=args
on a memory-read A1). The server-log expectations ``<stk>/expect_k<K>.txt`` get the cell's new build line and one
``[ATOM_Q38_EXTERN_STAGES]`` line per linked HC function. Serve with ``ATOM_Q38_EXTERN_STAGES=...,a1=<stk>/hc/{kind},
a2=<stk>/hc/{kind},m1=<stk>/hc/{kind},m2=<stk>/hc/{kind}`` (``EXTERN`` line). Variants without a function file (wide,
grouped, r2-r4) keep the FlyDSL HC (ATOM_Q38_EXTERN_MISSING=mono).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HC = ("a1", "a2", "m1", "m2")


def parse_sched(s: str):
    head, groups = [], {}
    for part in filter(None, s.strip().split(";")):
        cell, sep, body = part.partition(":")
        if not sep:
            head.append(part)
            continue
        groups[cell] = dict(kv.split("=", 1) for kv in body.split(",") if kv)
    return head, groups


def fmt_sched(head, groups) -> str:
    parts = list(head)
    for cell, lev in groups.items():
        body = ",".join(f"{k}={v}" for k, v in sorted(lev.items()))
        if body:
            parts.append(f"{cell}:{body}")
    return ";".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stk", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--cells", default="", help="subset of the pick cells (default: all)")
    ap.add_argument("--port", action="store_true", help="the FlyDSL-choice defaults on the stack's schedule")
    ap.add_argument("--flavor", default=os.environ.get("ROCKE_LLVM_FLAVOR", "llvm23"))
    a = ap.parse_args(argv)

    from kernels.gfx950.qwen38_mono.stages2 import hc as H

    picks = json.load(open(a.picks))
    cells = [c for c in picks if not a.cells or c in a.cells.split(",")]
    sp = os.path.join(a.stk, "sched.txt")
    sched0 = open(sp).read().strip()
    head, groups = parse_sched(sched0)
    kind_extra = {c: g for c, g in groups.items() if "." not in c}  # e.g. qsa:qdense=3
    specs, written = [], {}
    new_lines = {}
    for cell in cells:
        p = picks[cell]
        kind, variant = cell.split(".")
        rows, seq = int(variant[1:].rstrip("s")), variant.endswith("s")
        if a.port:
            from atom.models.qwen4_exp_mono.kernels import sched as SCH

            lev = {k: str(v) for k, v in SCH.parse_levers(SCH.select(sched0, kind, variant))}
            per = {}
            for st in HC:
                down = "m1" if st in ("m1", "m2") else "a1"
                spread = int(lev.get(f"spread.{down}", lev.get("spread", 0)))
                kn = f"numa=1,spread={spread}" if spread else "numa=1"
                if st == "a1" and kind != "delta_ple":
                    kn += ",pre=args"
                per[st] = kn
        else:
            _h, pg = parse_sched(p["sched"])
            if list(pg) != [cell]:
                raise ValueError(f"{cell}: pick sched groups {list(pg)}")
            lev = pg[cell]
            per = p["hc_knobs"]
            groups[cell] = dict(lev)
        for st in HC:
            k = H.knobs_from_spec(per[st])
            if k.tolerance:
                raise ValueError(f"{cell} {st}: tolerance-class knobs are not served")
            hs = H.HcSpec(st, kind, rows, seq, k)
            why = H.supported(hs)
            if why:
                raise ValueError(f"{cell} {st}: {why}")
            specs.append(hs)
            written.setdefault(cell, {})[st] = H.knobs_spec(k)
        # the build line ATOM logs (sched.log_build: levers sorted; qsa's kind-level qdense; numa=1 at <= 4 rows)
        ll = dict(groups.get(cell, {}))
        ll.update(kind_extra.get(kind, {}))
        ll["numa"] = "1"
        kk = str(rows - 1) if seq else "0"
        line = f"[ATOM_Q38_SCHED] build {kind} {variant}: " + " ".join(f"{k}={v}" for k, v in sorted(ll.items()))
        line += " | extern"
        ext = [f"[ATOM_Q38_EXTERN_STAGES] {st} {variant}: rocke_q38_{st}_{variant} from {a.stk}/hc/{kind}/{st}_{variant}"
               for st in HC]  # fmt: skip
        new_lines.setdefault(kk, []).append((f"[ATOM_Q38_SCHED] build {kind} {variant}:", line, ext))
    H.write_hc_functions(os.path.join(a.stk, "hc"), specs, flavor=a.flavor)
    sched = fmt_sched(head, groups)
    with open(sp + ".stack", "w", newline="\n") as f:
        f.write(sched0)
    with open(sp, "w", newline="\n") as f:
        f.write(sched)
    for kk, items in new_lines.items():
        ef = os.path.join(a.stk, f"expect_k{kk}.txt")
        old = open(ef).read().splitlines() if os.path.isfile(ef) else []
        prefixes = [x[0] for x in items]
        keep = [ln for ln in old if not any(ln.startswith(pf) for pf in prefixes)]
        for _pf, line, ext in items:
            keep.append(line)
            keep.extend(ext)
        with open(ef, "w", newline="\n") as f:
            f.write("".join(x + "\n" for x in keep))
    with open(os.path.join(a.stk, "wp2a.json"), "w", newline="\n") as f:
        json.dump({"port": a.port, "sched_stack": sched0, "sched": sched, "functions": written}, f, indent=1)
    print("WP2A_GEN", len(specs), "functions", "port" if a.port else "picks", "cells", ",".join(cells))
    print("WP2A_SCHED", sched)
    print("EXTERN", ",".join(f"{st}={a.stk}/hc/{{kind}}" for st in HC))
    return 0


if __name__ == "__main__":
    sys.exit(main())
