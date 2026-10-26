#!/usr/bin/env python3
"""Alternate kernel variants round by round on one GPU and report paired timing.

Matches the original cl24 baseline client settings:
  num-warmups=5, num-enqueues-per-sync=20

usage: alt_timing.py ROUNDS name=logdir [name=logdir ...]
"""
import csv, glob, os, re, statistics as st, subprocess, sys, tempfile

BENCH = "/home/mchirila/rocm-libraries/projects/hipblaslt/tensilelite/bench/gfx1250_mxf4"
CLIENT = BENCH + "/../../build_tmp/tensilelite/client/tensilelite-client"
# Same as logs/20261022-baseline-cl24-gpu0 / configs used for the ~13.4 PF number.
WARMUPS = 5
ENQUEUES = 20

rounds = int(sys.argv[1])
variants = [a.split("=", 1) for a in sys.argv[2:]]
src_inis = {n: glob.glob(f"{BENCH}/{d}/**/ClientParameters.ini", recursive=True)[0]
            for n, d in variants}
env = dict(os.environ, HIP_VISIBLE_DEVICES=os.environ.get("GPU_ID", "0"),
           LD_LIBRARY_PATH=os.environ.get("LIBOMP_DIR", "/opt/rocm/lib/llvm/lib"))
env.pop("ROCR_VISIBLE_DEVICES", None)

tmpdir = tempfile.mkdtemp(prefix="alt_timing_")
inis = {}
for n, src in src_inis.items():
    text = open(src).read()
    text = re.sub(r"^num-warmups=.*$", f"num-warmups={WARMUPS}", text, flags=re.M)
    text = re.sub(r"^num-enqueues-per-sync=.*$", f"num-enqueues-per-sync={ENQUEUES}",
                  text, flags=re.M)
    path = os.path.join(tmpdir, f"{n}.ini")
    open(path, "w").write(text)
    inis[n] = path
print(f"client: num-warmups={WARMUPS} num-enqueues-per-sync={ENQUEUES}  tmp={tmpdir}",
      flush=True)

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
    print(f"round {r}: " + "  ".join(f"{n} {times[n][-1]:.1f}" for n, _ in variants),
          flush=True)

flops = 2 * 4096 * 4096 * 65536
base = variants[0][0]
for n, _ in variants:
    t = times[n]
    tflops = flops / st.mean(t) / 1e6   # time in us → TFLOPS
    pflops = tflops / 1e3
    print(f"{n:8s} mean {st.mean(t):7.2f} us  sd {st.stdev(t):.2f}  "
          f"median {st.median(t):7.2f}  {tflops:8.0f} TFLOPS  {pflops:5.2f} PF")
for n, _ in variants[1:]:
    d = [a - b for a, b in zip(times[base], times[n])]
    print(f"paired {base}-{n}: mean {st.mean(d):.2f} us sd {st.stdev(d):.2f}  "
          f"{n} faster in {sum(x > 0 for x in d)}/{len(d)}  "
          f"speedup {st.mean(times[base]) / st.mean(times[n]):.4f}x")
