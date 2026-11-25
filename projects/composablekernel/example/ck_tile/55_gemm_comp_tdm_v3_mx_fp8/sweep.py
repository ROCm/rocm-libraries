#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""EXPERIMENT: sweep the comp_tdm_v3 MX FP8 example (55) over compile-time variants and
run-time arms; time first, verify the winners last.

Compile-time variants are -D flag sets (pipeline policy: A-row prefetch, refill point, scale
path, ...). Each becomes its own binary, built in parallel from the example's compile command
in <build>/compile_commands.json and its link line in the target's link.txt, so any configured
CMake build tree works (Unix Makefiles; configure it for gfx125x, CMAKE_EXPORT_COMPILE_COMMANDS
is on by default in CK). Build with -DCK_EX55_FOCUS in the variant's defines (or the default
defines below) to keep each build to the sweep set: 256x256x256, 2x2 waves, warp tile 16 or 32,
-cluster 1x1 / 2x1 / 4x1 (-place default or 8x4) / 2x2 / 4x4.

Run-time arms pick an instance inside a binary: warp tile, cluster, placement; shapes are
swept too. A binary whose kernel for an arm spills (VGPR or SGPR) is never run. Timing:
ROUNDS interleaved rounds (order reversed every other round), each arm 1 s idle, then the
example's sustained mode (-warmup/-repeat launches, one event pair). Correctness is checked
afterwards, for the --verify-top best arms only (verify-sparse and verify-rand at K = 8192).

    python3 sweep.py --build-dir <build> --config sweep_mx.json --gpu 0 --out <dir> [--jobs 8]
                     [--runner "gpu-guard run --"] [--verify-top 3] [--no-build]

--gpu auto (partitioned hosts, e.g. two partitions per physical GPU): samples `amd-smi
process` (--free-samples x --free-interval s), takes the first pair of --pairs (default
6,7 0,1 2,3 4,5) whose partitions both stayed idle and runs on the pair's first partition.
Before every timed batch one more sample is taken; a batch during which another process
used the pair is tagged and left out of the ranking. --allow-single falls back to a single
idle partition (quick checks only; the summary says so).

Writes <out>/results.csv (one row per timed launch batch) and prints a ranked summary.
"""
import argparse
import concurrent.futures as cf
import csv
import glob
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

SRC = "gemm_comp_tdm_v3_mx_fp8_bench.cpp"
TARGET = "tile_example_gemm_comp_tdm_v3_mx_fp8_bench"
# Experiment defines owned by the variants: stripped from the base compile command first, so
# a build tree configured with its own CK_EX55_* flags does not redefine them.
OWNED = re.compile(r"^-D(CK_EX55_|CK_TILE_GFX125_|CK_TILE_EXP_)")
LLVM = os.environ.get("ROCM_LLVM", "/opt/rocm/llvm/bin")


def log(msg):
    print(msg, flush=True)


def base_compile(build_dir: Path):
    db = json.loads((build_dir / "compile_commands.json").read_text())
    ent = [e for e in db if e["file"].endswith(SRC)]
    if not ent:
        sys.exit(f"{SRC} not in {build_dir}/compile_commands.json (configure the example first)")
    e = ent[0]
    args = e.get("arguments") or shlex.split(e["command"])
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a in ("-o", "-c"):
            i += 2
            continue
        if not OWNED.match(a):
            out.append(a)
        i += 1
    return out, e["file"], e["directory"]


def base_link(build_dir: Path):
    hits = glob.glob(str(build_dir / "**" / f"{TARGET}.dir" / "link.txt"), recursive=True)
    if not hits:
        sys.exit(f"no link.txt for {TARGET} under {build_dir} (Unix Makefiles generator needed)")
    link = Path(hits[0])
    args = shlex.split(link.read_text())
    return args, link.parent.parent.parent  # link.txt runs from the target's directory


def build_variant(v, out: Path, comp, src, cdir, link, ldir):
    exe = out / "bin" / v["name"]
    obj = out / "obj" / f"{v['name']}.o"
    exe.parent.mkdir(parents=True, exist_ok=True)
    obj.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cc = comp + list(v["defines"]) + ["-o", str(obj), "-c", src]
    r = subprocess.run(cc, cwd=cdir, capture_output=True, text=True)
    if r.returncode:
        (out / "obj" / f"{v['name']}.log").write_text(r.stdout + r.stderr)
        return v["name"], None, f"compile failed (see obj/{v['name']}.log)"
    la, i = [], 0
    while i < len(link):
        a = link[i]
        if a == "-o":
            la += ["-o", str(exe)]
            i += 2
            continue
        la.append(str(obj) if a.endswith(".o") else a)
        i += 1
    r = subprocess.run(la, cwd=ldir, capture_output=True, text=True)
    if r.returncode:
        return v["name"], None, "link failed: " + (r.stderr or r.stdout)[-400:]
    return v["name"], exe, f"built in {time.time() - t0:.0f} s"


def kernel_table(exe: Path):
    """Per kernel in the binary: (demangled name, vgpr spills, sgpr spills, vgprs)."""
    for f in glob.glob(f"{exe}.0.*"):
        os.remove(f)
    subprocess.run([f"{LLVM}/llvm-objdump", "--offloading", str(exe)], cwd=exe.parent,
                   capture_output=True)
    co = [f for f in glob.glob(f"{exe}.0.hip-amdgcn*") if "gfx125" in f]
    if not co:
        return []
    notes = subprocess.run([f"{LLVM}/llvm-readelf", "--notes", co[0]], capture_output=True,
                           text=True).stdout
    rows, cur = [], {}
    for line in notes.splitlines():
        m = re.match(r"\s+\.(name|vgpr_spill_count|sgpr_spill_count|vgpr_count):\s+(\S+)", line)
        if not m:
            continue
        cur[m.group(1)] = m.group(2)
        if "name" in cur and "vgpr_spill_count" in cur and "sgpr_spill_count" in cur and "vgpr_count" in cur:
            rows.append(cur)
            cur = {}
    names = subprocess.run([f"{LLVM}/llvm-cxxfilt"], input="\n".join(r["name"] for r in rows),
                           capture_output=True, text=True).stdout.splitlines()
    return [(n, int(r["vgpr_spill_count"]), int(r["sgpr_spill_count"]), int(r["vgpr_count"]))
            for n, r in zip(names, rows)]


def arm_kernel(table, arm):
    """The kernel an arm launches, matched on warp tile, cluster and placement."""
    wt, cl, pl = arm["wt"], arm["cluster"], arm.get("place", "default")
    cm, cn = (int(x) for x in cl.split("x"))
    hits = []
    for name, vs, ss, vg in table:
        if f"sequence<{wt}, {wt}, 128>" not in name or "256, 256, 256" not in name:
            continue
        clustered = "ClusterTileGemmShape" in name
        if (cm * cn > 1) != clustered:
            continue
        if clustered and f"ClusterTileGemmShape<ck_tile::sequence<{cm}, {cn}, 1>" not in name:
            continue
        if ("XcdRegion" in name) != (pl == "8x4"):
            continue
        hits.append((vs, ss, vg))
    return hits


def arm_args(arm, shape, cfg):
    m, n, k = shape
    a = [f"-m={m}", f"-n={n}", f"-k={k}", "-tile=256x256x256", "-epilogue=tdm", "-warps=2x2",
         f"-wt={arm['wt']}", f"-cluster={arm['cluster']}", f"-place={arm.get('place', 'default')}",
         f"-c_dtype={cfg.get('c_dtype', 'bf16')}"]
    return a


def _amdsmi(args):
    r = subprocess.run(["timeout", "30", "amd-smi"] + args, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"amd-smi {' '.join(args)} failed or timed out")
    return r.stdout


def hip_index_map():
    """amd-smi GPU id -> HIP device index, matched on PCI bus id. The HIP query runs in its own
    short-lived process, before any sampling, so its context is never mistaken for a job."""
    bdf_of = {}
    for line in _amdsmi(["list", "--csv"]).splitlines()[1:]:
        f = line.split(",")
        if len(f) >= 2 and f[0].strip().isdigit():
            bdf_of[f[0].strip()] = f[1].strip().lower()
    code = ("import ctypes,sys\nh=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')\n"
            "for i in range(int(sys.argv[1])):\n b=ctypes.create_string_buffer(64)\n"
            " h.hipDeviceGetPCIBusId(b,64,i)\n print(i,b.value.decode().lower())\n")
    r = subprocess.run(["timeout", "60", sys.executable, "-c", code, str(len(bdf_of))],
                       capture_output=True, text=True)
    hip_of_bdf = {}
    for line in r.stdout.splitlines():
        i, b = line.split()
        hip_of_bdf[b] = int(i)
    return {g: hip_of_bdf[b] for g, b in bdf_of.items() if b in hip_of_bdf}


def busy_sample(gmap, own_pid=None, detail=None):
    """HIP indices where another process is computing right now (one amd-smi process sample).

    A process counts only with cu_occupancy > 0: one that merely holds memory does not slow
    our kernels. Columns are found by header name. detail (a list) collects the culprits."""
    lines = _amdsmi(["process", "--csv"]).splitlines()
    if not lines:
        return set()
    head = [h.strip() for h in lines[0].split(",")]
    col = {h: i for i, h in enumerate(head)}
    busy = set()
    if "pid" not in col or "gpu" not in col:
        return busy  # no process running: amd-smi then prints just "gpu,process_info"
    for line in lines[1:]:
        f = [x.strip() for x in line.split(",")]
        if len(f) < len(head) or not f[col["pid"]].isdigit() or f[col["gpu"]] not in gmap:
            continue
        if own_pid is not None and int(f[col["pid"]]) == own_pid:
            continue
        occ = f[col["cu_occupancy"]] if "cu_occupancy" in col else "1"
        if not occ.isdigit() or int(occ) > 0:
            busy.add(gmap[f[col["gpu"]]])
            if detail is not None:
                detail.append(f"HIP {gmap[f[col['gpu']]]} pid {f[col['pid']]} "
                              f"{f[col['name']].split('/')[-1] if 'name' in col else '?'} cu_occ {occ}")
    return busy


def pick_gpu(o):
    gmap = hip_index_map()
    if not gmap:
        sys.exit("--gpu auto: could not map amd-smi GPUs to HIP indices; pass --gpu N")
    busy = {}
    for s in range(o.free_samples):
        for i in busy_sample(gmap):
            busy[i] = busy.get(i, 0) + 1
        if s + 1 < o.free_samples:
            time.sleep(o.free_interval)
    pairs = [tuple(int(x) for x in p.split(",")) for p in o.pairs.split()]
    for p in pairs:
        log(f"  pair {p}: " + ", ".join(f"HIP {i} " + (f"busy ({busy[i]} samples)" if i in busy else "free")
                                         for i in p))
    for p in pairs:
        if all(i in gmap.values() and i not in busy for i in p):
            log(f"--gpu auto: pair {p} free, running on HIP {p[0]}")
            return p[0], p, gmap, False
    if o.allow_single:
        for p in pairs:
            for i in p:
                if i in gmap.values() and i not in busy:
                    log(f"--gpu auto: no free pair; SINGLE partition HIP {i} (quick check only)")
                    return i, (i,), gmap, True
    sys.exit("--gpu auto: no free pair" + ("" if o.allow_single else " (--allow-single takes one partition)"))


def run(cmd, env, runner):
    full = (shlex.split(runner) if runner else []) + cmd
    r = subprocess.run(full, env=env, capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--gpu", required=True, help="HIP device index (HIP_VISIBLE_DEVICES), or auto")
    ap.add_argument("--pairs", default="6,7 0,1 2,3 4,5", help="--gpu auto: partition pairs, in order")
    ap.add_argument("--free-samples", type=int, default=6, help="--gpu auto: amd-smi samples")
    ap.add_argument("--free-interval", type=float, default=10.0, help="--gpu auto: seconds between")
    ap.add_argument("--allow-single", action="store_true", help="--gpu auto: fall back to one partition")
    ap.add_argument("--no-pair-check", action="store_true", help="--gpu auto: do not tag busy batches")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--runner", default="", help='launch prefix, e.g. "gpu-guard run --"')
    ap.add_argument("--verify-top", type=int, default=3, help="verify the N best arms (0: none)")
    ap.add_argument("--no-build", action="store_true", help="reuse <out>/bin binaries")
    ap.add_argument("--dry-run", action="store_true", help="build and match arms; no GPU runs")
    ap.add_argument("--allow-spills", type=int, default=0,
                    help="run arms whose kernel spills at most this many VGPRs + SGPRs (default 0)")
    o = ap.parse_args()

    cfg = json.loads(o.config.read_text())
    out = o.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    defaults = cfg.get("default_defines", [])
    variants = [dict(v, defines=defaults + v.get("defines", [])) for v in cfg["variants"]]

    # --- build
    if not o.no_build:
        comp, src, cdir = base_compile(o.build_dir)
        link, ldir = base_link(o.build_dir)
        log(f"building {len(variants)} variant(s), {o.jobs} at a time")
        with cf.ThreadPoolExecutor(o.jobs) as ex:
            for name, exe, msg in ex.map(lambda v: build_variant(v, out, comp, src, cdir, link, ldir),
                                         variants):
                log(f"  {name}: {msg}")

    # --- arms: (variant, arm) pairs whose kernel exists and does not spill
    plan = []
    for v in variants:
        exe = out / "bin" / v["name"]
        if not exe.exists():
            log(f"SKIP variant {v['name']}: no binary")
            continue
        table = kernel_table(exe)
        for arm in cfg["arms"]:
            tag = f"{v['name']}|wt{arm['wt']}|c{arm['cluster']}|{arm.get('place', 'default')}"
            hits = arm_kernel(table, arm)
            if not hits:
                log(f"SKIP {tag}: no such instance in the binary")
                continue
            spill = [(vs, ss) for vs, ss, _ in hits if vs or ss]
            if spill and max(vs + ss for vs, ss in spill) > o.allow_spills:
                log(f"SKIP {tag}: spills (vgpr/sgpr spills {[(vs, ss) for vs, ss, _ in hits]})")
                continue
            if spill:
                log(f"NOTE {tag}: runs with spills {spill} (--allow-spills {o.allow_spills})")
            plan.append(dict(tag=tag, exe=exe, arm=arm, vgpr=max(vg for _, _, vg in hits)))
    log(f"{len(plan)} arm(s) x {len(cfg['shapes'])} shape(s) to time")
    for p in plan:
        log(f"  {p['tag']}  vgpr={p['vgpr']}")
    if o.dry_run or not plan:
        return

    # --- time first
    gpu, pair, gmap, single = o.gpu, None, None, False
    if o.gpu == "auto":
        gpu, pair, gmap, single = pick_gpu(o)
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
    rounds, warm, rep = cfg.get("rounds", 5), cfg.get("warmup", 50), cfg.get("repeat", 100)
    timing = ["-mode=sustained", "-data=rand", "-sigma=0.1", f"-warmup={warm}", f"-repeat={rep}"]
    res = {}
    with open(out / "results.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["variant_arm", "M", "N", "K", "round", "median_tflops", "pair_busy"])
        for shape in cfg["shapes"]:
            for r in range(rounds):
                order = plan if r % 2 == 0 else plan[::-1]
                for p in order:
                    time.sleep(1.0)
                    tainted = False
                    if pair and not o.no_pair_check:
                        why = []
                        try:
                            tainted = bool(busy_sample(gmap, os.getpid(), why) & set(pair))
                        except RuntimeError:
                            tainted, why = True, ["amd-smi failed"]
                        if tainted:
                            log("  pair busy: " + "; ".join(w for w in why if any(f"HIP {i} " in w for i in pair) or w == "amd-smi failed"))
                    rc, txt = run([str(p["exe"])] + arm_args(p["arm"], shape, cfg) + timing, env, o.runner)
                    m = re.search(r'"median_tflops":([0-9.]+)', txt)
                    tf = float(m.group(1)) if (m and rc == 0) else None
                    res.setdefault((p["tag"], tuple(shape)), []).append(None if tainted else tf)
                    w.writerow([p["tag"], *shape, r + 1, tf if tf is not None else "ERR", int(tainted)])
                    fcsv.flush()
                    log(f"r{r + 1} {shape} {p['tag']}: {tf if tf is not None else 'ERROR rc=%d' % rc}"
                        + ("  [pair busy: not ranked]" if tainted else ""))

    # --- summary, best first per shape
    best = []
    for shape in cfg["shapes"]:
        rows = []
        for p in plan:
            v = [x for x in res.get((p["tag"], tuple(shape)), []) if x is not None]
            if v:
                rows.append((statistics.median(v), min(v), max(v), p))
        rows.sort(key=lambda x: -x[0])
        log(f"\n== {shape[0]}x{shape[1]}x{shape[2]}: median TF over clean rounds of {rounds}"
            + (f" on HIP {gpu}" if gpu is not None else "")
            + ("  [SINGLE partition: quick check only]" if single else ""))
        top = rows[0][0] if rows else 1.0
        for med, lo, hi, p in rows:
            n_ok = len([x for x in res.get((p["tag"], tuple(shape)), []) if x is not None])
            log(f"  {med:8.1f}  ({lo:.0f}-{hi:.0f})  {med / top:.3f}x  {p['tag']}  [{n_ok}/{rounds} clean]")
        best += [(shape, p) for _, _, _, p in rows[: o.verify_top]]

    # --- verify the winners last
    if o.verify_top:
        log("\n== verify (K = 8192) of the best arms")
        for shape, p in best:
            vshape = [shape[0], shape[1], 8192]
            for mode in ("verify-sparse", "verify-rand"):
                rc, txt = run([str(p["exe"])] + arm_args(p["arm"], vshape, cfg) + [f"-mode={mode}"],
                              env, o.runner)
                m = re.search(r'"bad":(\d+).*?"result":"([A-Z]+)"', txt)
                log(f"  {p['tag']} {shape[0]}x{shape[1]} {mode}: "
                    + (f"{m.group(2)} bad={m.group(1)}" if m else f"NO RESULT rc={rc}"))


if __name__ == "__main__":
    main()
