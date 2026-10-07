#!/usr/bin/env python3
"""Validate one build at a list of problem sizes on physical GPU $GPU_ID (default 3).

usage: validate.py LOGDIR [--jitter N] [--print-max N] [--timeout S] [SIZE ...]

LOGDIR holds the timing ClientParameters.ini of a build (logs/*-gen-vN). SIZE is M,N,B,K;
the default is the 10-size edge/StreamK set. --jitter sets mx-scale-jitter (default 4), which
gives every scale block a different value so scale addressing errors show up. Prints one
PASSED/FAILED line per size and writes LOGDIR/validate.{ini,log}.
"""
import argparse, glob, os, re, subprocess

BENCH = os.path.dirname(os.path.abspath(__file__))
ROCM = "/opt/rocm-10.1.0a20260908+bkc.20260917"
CLIENT = f"{BENCH}/../../build_tmp/tensilelite/client/tensilelite-client"
DEFAULT = ["256,256,1,256", "256,256,1,2048", "512,256,1,2048", "256,512,1,2048",
           "512,512,1,2048", "2048,1024,1,384", "4096,4096,1,8192", "384,320,1,2048",
           "300,200,1,1024", "1000,1500,1,2048"]

ap = argparse.ArgumentParser()
ap.add_argument("logdir")
ap.add_argument("sizes", nargs="*", default=DEFAULT)
ap.add_argument("--jitter", type=int, default=4)
ap.add_argument("--print-max", type=int, default=4)
ap.add_argument("--timeout", type=int, default=120, help="seconds before the client is killed")
args = ap.parse_args()

logdir = os.path.join(BENCH, args.logdir) if not os.path.isabs(args.logdir) else args.logdir
src = glob.glob(f"{logdir}/**/ClientParameters.ini", recursive=True)[0]
lines = [l for l in open(src).read().splitlines()
         if not re.match(r"(problem-size|[a-d]-strides|mx-scale-jitter|num-elements-to-validate"
                         r"|print-max|num-benchmarks|num-enqueues-per-sync|num-warmups)=", l)]
for s in args.sizes:
    m, n, b, k = s.split(",")
    lines += [f"problem-size={m},{n},{b},{k}", f"a-strides=-1,{k},-1", f"b-strides=-1,{k},-1",
              f"c-strides=-1,{m},-1", f"d-strides=-1,{m},-1"]
lines += ["num-elements-to-validate=-1", f"print-max={args.print_max}", "num-benchmarks=1",
          "num-enqueues-per-sync=1", "num-warmups=0", f"mx-scale-jitter={args.jitter}"]
ini, log = f"{logdir}/validate.ini", f"{logdir}/validate.log"
open(ini, "w").write("\n".join(lines) + "\n")

env = dict(os.environ, HIP_VISIBLE_DEVICES=os.environ.get("GPU_ID", "3"),
           LD_LIBRARY_PATH=f"{ROCM}/lib/llvm/lib/x86_64-unknown-linux-gnu")
env.pop("ROCR_VISIBLE_DEVICES", None)
with open(log, "w") as f:
    try:
        rc = subprocess.run([CLIENT, "--config-file", ini], stdout=f, stderr=subprocess.STDOUT,
                            env=env, timeout=args.timeout).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout (killed)"
out = open(log).read()
res = re.findall(r'"\(([0-9,]+)\)".*?,(PASSED|FAILED),', out)
for size, r in res:
    print(f"{size:>20}  {r}")
missing = len(args.sizes) - len(res)
if missing:
    print(f"{missing} size(s) without a result (client rc={rc}, see {log})")
