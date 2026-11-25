# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Usage: python -I -B sweep2.py <rocke_root> <golden_json> <out_json> <fams_file> <threads>
import hashlib, json, pathlib, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8")
root, gold, outp, famsf, nthreads = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
    int(sys.argv[5]),
)
g = json.load(open(gold, encoding="utf-8"))["ll"]
here = pathlib.Path(__file__).parent
jobs = []
for line in open(famsf, encoding="utf-8").read().split():
    fam, src = line.split("=", 1)
    for idx in sorted(g.get(fam, {}), key=int):
        jobs.append((fam, pathlib.Path(src), idx))


def run(j):
    fam, src, idx = j
    p = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(here / "run_emit_wrap.py"),
            root,
            str(src / f"{fam}_emit.py"),
            idx,
        ],
        capture_output=True,
        cwd=str(here),
    )
    out = p.stdout.replace(b"\r\n", b"\n")
    return fam, idx, p.returncode, hashlib.sha256(out).hexdigest()


res = {}
with ThreadPoolExecutor(nthreads) as ex:
    for fam, idx, rc, sha in ex.map(run, jobs):
        res.setdefault(fam, {})[idx] = {
            "rc": rc,
            "sha": sha,
            "match": g[fam].get(idx) == sha,
        }
json.dump(res, open(outp, "w", encoding="utf-8"), indent=1)
tot = sum(len(d) for d in res.values())
m = sum(v["match"] for d in res.values() for v in d.values())
print(
    "configs",
    tot,
    "match",
    m,
    "rc!=0",
    sum(v["rc"] != 0 for d in res.values() for v in d.values()),
)
