"""RT coverage census: which true-mono kernels the direct compiler lowers, and what it still refuses.

    python rt_census.py --tc <target_compile> --out DIR [--labels L ...] [--no-neutralize]

For each ``rt_builds`` label: ``DIR/<label>.ir`` (serialized rocKE IR) and one ``target_compile gfx950`` run; a
refused build (``ROCKE_DIRECT_UNSUPPORTED``) goes through ``direct_pilot.neutralize_census`` to list every later
refusal too. Prints one line per label and writes ``DIR/census.json``. Compile-only, one process at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_tc(tc: str, ir_text: str) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "k.ir")
        with open(p, "w", newline="\n") as fh:
            fh.write(ir_text)
        r = subprocess.run([tc, "gfx950", d, p], capture_output=True, text=True)
        line = (r.stdout.strip().splitlines() or ["k\terror\tno output " + r.stderr[-400:]])[-1]
        parts = line.split("\t")
        return parts[1], "\t".join(parts[2:])


# target_compile prints the message without the error kind: the refusals (ROCKE_DIRECT_UNSUPPORTED) by text
REFUSAL_TEXT = ("is not lowered by the direct backend", "tile.inline_asm other than", "kernel symbol",
                "only global pointers are supported", "kernarg preload", "pointer to", "agpr_alloc",
                "unsupported kernel argument", "unknown kernel attribute", "unknown element type")


def unsupported(msg: str) -> bool:
    return "ROCKE_DIRECT_UNSUPPORTED" in msg or any(t in msg for t in REFUSAL_TEXT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--no-neutralize", action="store_true")
    a = ap.parse_args(argv)
    import rt_builds
    from rocke.core.ir_serialize import serialize
    from rocke.runtime.code_object import CodeObjectError, DirectUnsupported

    from kernels.gfx950.qwen38_mono import direct_pilot as dp

    os.makedirs(a.out, exist_ok=True)
    res = {}
    for label in a.labels or list(rt_builds.BUILDS):
        t0 = time.perf_counter()
        try:
            k = rt_builds.build(label)
            from rocke.runtime.direct_lift import lift

            lk = lift(k)
            text = serialize(lk)
        except Exception as e:  # noqa: BLE001
            res[label] = {"build_error": f"{type(e).__name__}: {e}"[:600]}
            print(f"{label}: BUILD ERROR {res[label]['build_error'][:200]}", flush=True)
            continue
        with open(os.path.join(a.out, f"{label}.ir"), "w", newline="\n") as fh:
            fh.write(text)
        st, msg = run_tc(a.tc, text)
        r = {"status": st, "msg": msg[:600], "ir_bytes": len(text), "lifted": getattr(lk, "lifted", {})}
        if st == "ok":
            print(f"{label}: OK {msg}", flush=True)
        else:
            print(f"{label}: {msg[:220]}", flush=True)
            if unsupported(msg) and not a.no_neutralize:

                def cf(t: str) -> None:
                    s, m = run_tc(a.tc, t)
                    if s == "ok":
                        return
                    m = "direct rocKE IR compiler rejected the kernel: ROCKE_DIRECT_UNSUPPORTED: " + m
                    if unsupported(m):
                        raise DirectUnsupported(m)
                    raise CodeObjectError(m)

                c = dp.neutralize_census(text, cf)
                r["census"] = c
                for x in c["refusals"][1:]:
                    print(f"    then: {x[:200]}", flush=True)
                print(f"    final: {c['final'][:200]}", flush=True)
        r["s"] = round(time.perf_counter() - t0, 2)
        res[label] = r
    with open(os.path.join(a.out, "census.json"), "w", newline="\n") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
