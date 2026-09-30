#!/usr/bin/env python3
"""Alternate kernel variants round by round on one GPU and report paired timing.

usage: alt_timing.py ROUNDS name=logdir [name=logdir ...]
"""
import csv, glob, os, re, statistics as st, subprocess, sys

BENCH = "/home/mchirila/rocm-libraries/projects/hipblaslt/tensilelite/bench/gfx1250_mxf4"
CLIENT = BENCH + "/../../build_tmp/tensilelite/client/tensilelite-client"

rounds = int(sys.argv[1])
variants = [a.split("=", 1) for a in sys.argv[2:]]
inis = {n: glob.glob(f"{BENCH}/{d}/**/ClientParameters.ini", recursive=True)[0] for n, d in variants}
env = dict(os.environ, HIP_VISIBLE_DEVICES="3",
           LD_LIBRARY_PATH="/opt/rocm-10.1.0a20260908+bkc.20260917/lib/llvm/lib/x86_64-unknown-linux-gnu")
env.pop("ROCR_VISIBLE_DEVICES", None)

times = {n: [] for n, _ in variants}
for r in range(rounds):
    order = variants if r % 2 == 0 else variants[::-1]
    for n, _ in order:
        out = subprocess.run([CLIENT, "--config-file", inis[n]], env=env, cwd=BENCH,
                             capture_output=True, text=True).stdout
        rows = [l for l in out.splitlines() if re.match(r"^\d+,", l)]
        hdr = next(l for l in out.splitlines() if l.startswith("run,"))
        rec = dict(zip(next(csv.reader([hdr])), next(csv.reader([rows[-1]]))))
        times[n].append(float(rec["time-us"]))
    print(f"round {r}: " + "  ".join(f"{n} {times[n][-1]:.1f}" for n, _ in variants), flush=True)

flops = 2 * 4096 * 4096 * 65536
base = variants[0][0]
for n, _ in variants:
    t = times[n]
    print(f"{n:8s} mean {st.mean(t):7.2f} us  sd {st.stdev(t):.2f}  median {st.median(t):7.2f}  "
          f"{flops / st.mean(t) / 1e6:8.0f} TFLOPS")
for n, _ in variants[1:]:
    d = [a - b for a, b in zip(times[base], times[n])]
    print(f"paired {base}-{n}: mean {st.mean(d):.2f} us sd {st.stdev(d):.2f}  "
          f"{n} faster in {sum(x > 0 for x in d)}/{len(d)}  speedup {st.mean(times[base]) / st.mean(times[n]):.4f}x")
