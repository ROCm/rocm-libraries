# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Derive a pins file (case -> kernel + launch) from benchmark_conv_compare results.

usage: python derive_conv_pins.py --out pins.json RUN_DIR [RUN_DIR ...]

Each RUN_DIR holds ``<dir>_s<N>.jsonl`` result rows and their ``.log`` (as
31_rocke_bench_par.sh writes them). With one run the pin is that run's winner.
With several, every finalist time a log printed (``[i/n]  TFLOPS  ms  label``) is
collected per case and label; the pin is the label with the lowest median over
the runs that measured it, among labels measured in at least ``--min-runs`` runs
(default: half of the runs that measured the case, rounded up). The ``ms`` of a pin is that median.
"""
import argparse
import json
import os
import re
import statistics
import sys

SHAPE_CALLS = {1: 2, 2: 1, 3: 1, 4: 2, 5: 1, 6: 2, 7: 2, 8: 1, 9: 1, 10: 2}
FIN_RE = re.compile(r"^\s+\[\s*\d+/\d+\]\s+([\d.]+) TFLOPS\s+([\d.]+) ms\s+(\S.*\S)\s*$")


def read_run(d):
    out = {}
    for name in sorted(os.listdir(d)):
        m = re.fullmatch(r"(fwd|dgrad|wgrad)_s(\d+)\.jsonl", name)
        if not m:
            continue
        key = f"{m.group(1)}_s{m.group(2)}"
        rows = [json.loads(line) for line in open(os.path.join(d, name)) if line.strip()]
        if not rows:
            continue
        row = rows[-1]
        fin = {}
        log = os.path.join(d, name + ".log")
        if os.path.exists(log):
            for line in open(log):
                f = FIN_RE.match(line)
                if f and "probe" not in line:
                    fin.setdefault(f.group(3), []).append(float(f.group(2)))
        out[key] = {
            "kernel": row.get("implicit_kernel") or row.get("kernel"),
            "ms": row.get("implicit_ms") or row.get("ms"),
            "cmd": row.get("miopen_cmd"),
            "finalists": {k: min(v) for k, v in fin.items()},
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-runs", type=int, default=0)
    ap.add_argument("--alts", type=int, default=0,
                    help="also list up to N alternative launches per case (the best finalists by median, "
                         "within --alts-window of the pin) as 'alts' -- the candidates of an in-step selection")
    ap.add_argument("--alts-window", type=float, default=0.03)
    a = ap.parse_args()
    runs = [read_run(d) for d in a.runs]
    cases = {}
    for key in sorted(set().union(*runs)):
        have = [r[key] for r in runs if key in r]
        shape = int(key.split("_s")[1])
        base = {"cmd": have[0]["cmd"], "calls": SHAPE_CALLS.get(shape, 1)}
        if len(have) == 1:
            cases[key] = dict(base, kernel=have[0]["kernel"], ms=have[0]["ms"])
            continue
        need_k = a.min_runs or -(-len(have) // 2)
        times = {}
        for h in have:
            for lab, ms in h["finalists"].items():
                times.setdefault(lab, []).append(ms)
            times.setdefault(h["kernel"], [])
        best = None
        for lab, v in times.items():
            if len(v) < need_k:
                continue
            med = statistics.median(v)
            if best is None or med < best[1]:
                best = (lab, med, len(v))
        if best is None:
            best = (have[0]["kernel"], have[0]["ms"], 1)
        cases[key] = dict(base, kernel=best[0], ms=round(best[1], 4), n_runs=best[2],
                          winners=[h["kernel"] for h in have])
    if a.alts:
        for key, pin in cases.items():
            pool = {}
            for r in runs:
                if key in r:
                    for lab, ms in r[key]["finalists"].items():
                        pool.setdefault(lab, []).append(ms)
            ranked = sorted(pool, key=lambda lab: statistics.median(pool[lab]))
            best = statistics.median(pool[ranked[0]]) if ranked else None
            alts = [pin["kernel"]] + [
                lab for lab in ranked
                if lab != pin["kernel"] and statistics.median(pool[lab]) <= best * (1 + a.alts_window)
            ]
            pin["alts"] = alts[: a.alts]
    json.dump({"source": a.runs, "cases": cases}, open(a.out, "w"), indent=1)
    print(f"{len(cases)} pins -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
