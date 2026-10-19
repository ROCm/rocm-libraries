"""Split custom_mainloop_scheduling.yaml into N shards (round-robin by size group) and
build-only them concurrently, to test whether group-level parallelism recovers the idle cores.
Usage: python split_experiment.py N   (run from the tensilelite dir, caches cleared by caller)
"""
import copy, os, subprocess, sys, time
import yaml

N = int(sys.argv[1])
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
TL = "/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite"
SRC = f"{TL}/tensilelite/Tests/common/gemm/gfx950/custom_mainloop_scheduling.yaml"
OUT = f"{TL}/timing_runs/cms/split_{N}{TAG}"
os.makedirs(OUT, exist_ok=True)

doc = yaml.safe_load(open(SRC))
flat = [(pt[0], g) for pt in doc["BenchmarkProblems"] for g in pt[1:]]
print("total groups", len(flat))
shards = [[] for _ in range(N)]
for i, item in enumerate(flat):
    shards[i % N].append(item)

procs = []
cores_per = 25 // N
t0 = time.time()
for k, items in enumerate(shards):
    d = copy.deepcopy(doc)
    d["BenchmarkProblems"] = [[pt, g] for pt, g in items]
    d.pop("TestParameters", None)
    cfg = f"{OUT}/shard{k}.yaml"
    yaml.safe_dump(d, open(cfg, "w"), sort_keys=False)
    cores = f"{k*cores_per}-{(k+1)*cores_per-1}"
    cmd = ["taskset", "-c", cores, "python", "-c",
           "import sys; from Tensile import Tensile; Tensile.Tensile(sys.argv[1:])",
           cfg, f"/tmp/tl_split_out_{N}{TAG}_{k}", "--build-only", "--gpu-targets", "gfx950",
           "--cxx-compiler", "/usr/bin/amdclang++",
           "--global-parameters", "TimingInstrumentation=True", "LibraryFormat='msgpack'"]
    log = open(f"{OUT}/shard{k}.log", "w")
    procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), time.time()))
rcs = []
for p, st in procs:
    rcs.append(p.wait())
print("rcs", rcs, "wall %.1f s" % (time.time() - t0))
