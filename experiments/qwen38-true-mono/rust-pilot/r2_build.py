"""R2: build the pilot kernels and write their IR, LLVM text and code objects.

    python r2_build.py --out DIR [--labels L ...] [--backends comgr direct] [--max-vgprs N]

Runs with PYTHONPATH on the rust tree (platform/python + library). For every
``direct_pilot.pilot_builds()`` label: ``DIR/<label>.ir`` (serialized rocKE
IR), ``<label>.ll`` (the Python lowering, what COMGR compiles) and one code
object per backend, ``<label>.<backend>.hsaco``. A refused direct build
writes ``<label>.direct.refused`` with the refusal instead (the run fails
unless ``--allow-refusal``). ``build.json`` records names, sizes and sha256
of everything written. ``--max-vgprs N`` compiles the direct objects at a
VGPR budget of N (forced spills) through the crate's ``target_compile``
example (``--target-compile PATH``), writing ``<label>.direct_v<N>.hsaco``;
``--spill-fracs F ...`` picks N as fractions of each kernel's own direct VGPR
count instead (rounded down to 8, at least 24, only below that count).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--backends", nargs="+", default=["comgr", "direct"])
    ap.add_argument("--allow-refusal", action="store_true")
    ap.add_argument("--max-vgprs", type=int, nargs="*", default=[])
    ap.add_argument("--spill-fracs", type=float, nargs="*", default=[])
    ap.add_argument("--target-compile", default="")
    a = ap.parse_args(argv)
    from kernels.gfx950.qwen38_mono import direct_pilot as dp
    from rocke.core.ir_serialize import serialize
    from rocke.helpers.compile import compile_kernel

    os.makedirs(a.out, exist_ok=True)
    rec, bad = {}, 0
    for label, build in dp.pilot_builds():
        if a.labels and label not in a.labels:
            continue
        k = build()
        text = serialize(k)
        r = {"name": k.name, "ir_sha": _sha(text.encode())}
        with open(os.path.join(a.out, f"{label}.ir"), "w", newline="\n") as fh:
            fh.write(text)
        for be in a.backends:
            t0 = time.perf_counter()
            try:
                art = compile_kernel(k, arch="gfx950", backend="python", code_object_backend=be)
            except Exception as e:  # noqa: BLE001 (record any refusal or failure)
                r[be] = {"error": f"{type(e).__name__}: {e}"[:2000]}
                bad += 1
                print(f"{label} {be}: ERROR {r[be]['error'][:300]}", flush=True)
                continue
            used = getattr(art, "code_object_backend", be)
            fb = getattr(art, "direct_fallback", None)
            if be == "direct" and used != "direct":
                with open(os.path.join(a.out, f"{label}.direct.refused"), "w") as fh:
                    fh.write(str(fb))
                r[be] = {"refused": str(fb)}
                bad += not a.allow_refusal
                print(f"{label} direct: REFUSED {fb}", flush=True)
                continue
            path = os.path.join(a.out, f"{label}.{be}.hsaco")
            with open(path, "wb") as fh:
                fh.write(art.hsaco)
            if be == "comgr":
                with open(os.path.join(a.out, f"{label}.ll"), "w", newline="\n") as fh:
                    fh.write(art.llvm_text)
            r[be] = {"hsaco_sha": _sha(art.hsaco), "bytes": len(art.hsaco), "s": round(time.perf_counter() - t0, 3)}
            print(f"{label} {be}: {len(art.hsaco)} bytes {r[be]['hsaco_sha'][:16]} ({r[be]['s']} s)", flush=True)
        budgets = list(a.max_vgprs)
        if a.spill_fracs and a.target_compile:
            sub0 = os.path.join(a.out, "v0")
            os.makedirs(sub0, exist_ok=True)
            p0 = subprocess.run([a.target_compile, "gfx950", sub0, os.path.join(a.out, f"{label}.ir")], capture_output=True, text=True)
            cols = p0.stdout.split()
            if p0.returncode == 0 and len(cols) >= 5 and cols[1] == "ok":
                v = int(cols[4])
                r["direct_vgprs"] = v
                budgets += [n for n in sorted({max(24, int(v * f) // 8 * 8) for f in a.spill_fracs}) if n < v]
        for n in budgets:
            ir_path = os.path.join(a.out, f"{label}.ir")
            sub = os.path.join(a.out, f"v{n}")
            os.makedirs(sub, exist_ok=True)
            p = subprocess.run([a.target_compile, "gfx950", sub, "--max-vgprs", str(n), ir_path],
                               capture_output=True, text=True)  # fmt: skip
            src = os.path.join(sub, f"{label}.hsaco")
            ok = p.returncode == 0 and os.path.exists(src)
            if ok:
                os.replace(src, os.path.join(a.out, f"{label}.direct_v{n}.hsaco"))
            r[f"direct_v{n}"] = {"ok": ok, "out": (p.stdout + p.stderr)[-600:]}
            bad += not ok
            print(f"{label} direct_v{n}: {'ok' if ok else 'FAIL'} {(p.stdout + p.stderr).strip()[-200:]}", flush=True)
        rec[label] = r
    with open(os.path.join(a.out, "build.json"), "w", newline="\n") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
