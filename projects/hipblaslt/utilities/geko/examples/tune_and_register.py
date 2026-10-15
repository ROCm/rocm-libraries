#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run Geko, then add the kernels that win to the user kernel library.

The library is the directory HIPBLASLT_USER_KERNEL_LIBRARY names, the same variable
a workload sets to use it. Geko runs exactly as you would run it yourself: give its
command line, --tune or --search with any of its options, after `--`. Then:

  1. geko      the command, unchanged, in the current directory, without the
               variable, so Geko compares against the shipped kernels
  2. measure   in a process whose HIPBLASLT_USER_KERNEL_LIBRARY names a scratch
               copy of the library, register Geko's payloads, check each tuned
               kernel's output against a host reference, and compare it with what
               hipBLASLt selects today. Against the shipped kernel that is Geko's
               own hipblaslt-bench measurement; against a user kernel an earlier
               run added, both are timed the same way, with hipblaslt-bench.
  3. add       sizes Geko kept whose tuned kernel is correct and at least
               --min-speedup faster than today's selection are registered and
               mapped into the library; nothing else is written there
  4. confirm   a fresh process with only HIPBLASLT_USER_KERNEL_LIBRARY set reports
               what the heuristic now selects for every size

Registering, mapping and checking go through geko.user_kernels (ctypes), so --lib,
or the libhipblaslt.so the loader finds, must be built with
HIPBLASLT_ENABLE_USER_KERNELS=ON. Timing uses build/release/clients/hipblaslt-bench
in the hipBLASLt checkout Geko uses (--hipblaslt), run as Geko runs it. Any process
started with the variable naming the library gets the kernels, with no API call:

    HIPBLASLT_USER_KERNEL_LIBRARY=/abs/path/user_kernel_lib python train.py

Rerunning with the same library is safe: a size is only remapped when the new
kernel beats the one already there.

With --replay nothing is tuned and the library is only read: a fresh process names
it and, for every size it maps, reports whether the heuristic returns the user
kernel, how fast it is against the shipped kernel in hipblaslt-bench, and whether
both match the host reference.

Usage:
    export HIPBLASLT_USER_KERNEL_LIBRARY=/abs/path/user_kernel_lib
    python tune_and_register.py [OPTIONS] -- geko --tune --arch gfx950 --list sizes.yaml \\
        --devices 0,1 --workdir W
    python tune_and_register.py [OPTIONS] -- geko --search --list sizes.yaml --workdir W
    python tune_and_register.py [OPTIONS] --geko-workdir W    # a Geko run that finished
    python tune_and_register.py --replay
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geko.user_kernels import (  # noqa: E402
    LIBRARY_ENV, TENSILE_TYPES, HipBLASLt, MatmulAlgo, Problem, UserKernels,
    geko_payloads, problem_of, read_payload, saved_in, saved_mappings, user_kernel_library,
)

GEKO_TYPES = {v[1]: k for k, v in TENSILE_TYPES.items()}

# LD_LIBRARY_PATH at an install tree makes Geko's tensilelite client load a host
# library that cannot read the YAML Geko writes; an inherited HIP_VISIBLE_DEVICES
# remaps the GPU ids Geko is given; HIPBLASLT_TENSILE_LIBPATH redirects the shipped
# baseline, and a user kernel library would change it.
GEKO_ENV_DROP = ("LD_LIBRARY_PATH", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                 "CUDA_VISIBLE_DEVICES", "HIPBLASLT_TENSILE_LIBPATH", LIBRARY_ENV)

BENCH = "build/release/clients/hipblaslt-bench"


def size_key(p: Problem) -> str:
    return f"{p.trans}:{p.a_type}:{p.c_type}:{p.m}x{p.n}x{p.batch}x{p.k}"


def label(p: Problem) -> str:
    return f"M={p.m} N={p.n} B={p.batch} K={p.k} {p.trans} {TENSILE_TYPES[p.a_type][1]}"


def step(n, title):
    print(f"\n[{n}/4] {title}", flush=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def geko_option(cmd, name):
    """The value of `name` in a Geko command line, or None."""
    for i, a in enumerate(cmd):
        if a == name and i + 1 < len(cmd):
            return cmd[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def run_geko(cmd, log):
    """Run Geko's command line as given; return its exit code and run directory."""
    env = dict(os.environ)
    dropped = [v for v in GEKO_ENV_DROP if env.pop(v, None) is not None]
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    print(f"      {' '.join(cmd)}")
    if dropped:
        print(f"      (without {', '.join(dropped)})")
    print(f"      log: {log}", flush=True)

    workdir = geko_option(cmd, "--workdir")
    before = set(Path.cwd().glob("geko_*"))
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(5)
            if int(time.time() - t0) % 120 < 5:
                print(f"      ... still running ({time.time() - t0:.0f}s)", flush=True)
    print(f"      geko finished in {time.time() - t0:.0f}s (exit {proc.returncode})")
    if workdir:
        return proc.returncode, Path(workdir).resolve()
    # Without --workdir, Geko makes ./geko_<timestamp>.
    new = sorted(d for d in set(Path.cwd().glob("geko_*")) - before if d.is_dir())
    return proc.returncode, (new[-1].resolve() if new else None)


def read_geko(geko_dir):
    """Geko's verdict for every size it benchmarked, keyed by size_key()."""
    raw = geko_dir / "results/raw_results.csv"
    if not raw.is_file():
        return None
    final = geko_dir / "results/final_results.csv"
    # Geko writes no final_results.csv when it keeps nothing.
    kept_rows = list(csv.DictReader(open(final))) if final.is_file() else []

    def problem(r):
        return Problem(r["transA"] + r["transB"], int(r["m"]), int(r["n"]),
                       int(r["batch_count"]), int(r["k"]), GEKO_TYPES.get(r["a_type"], r["a_type"]),
                       GEKO_TYPES.get(r["b_type"], r["b_type"]),
                       GEKO_TYPES.get(r["d_type"] or r["c_type"], r["c_type"]))

    metrics = geko_dir / "results/metrics.json"
    up_thr = (json.loads(metrics.read_text()).get("config", {}).get("up_thr")
              if metrics.is_file() else None)
    kept = {size_key(problem(r)) for r in kept_rows}
    out = {}
    for r in csv.DictReader(open(raw)):
        p = problem(r)
        out[size_key(p)] = {
            "problem": asdict(p), "shipped_us": float(r["us_reference"]),
            "tuned_us": float(r["us_tuned"]), "uplift_pct": float(r["uplift_pct"]),
            "kept": size_key(p) in kept, "kernel": r["solution_tuned"],
            "same_as_shipped": r["kernel_reference"] == r["kernel_tuned"],
            "valid": r.get("valid", "True") == "True", "error": r.get("error_tuned"),
            "up_thr": up_thr, "shipped_source": r["lib_source_reference"],
            "supported": p.a_type in TENSILE_TYPES and TENSILE_TYPES[p.a_type][2] is not None,
        }
    return out


def not_kept_reason(g):
    """Why Geko left this size out of final_libs."""
    if g["same_as_shipped"]:
        return "Geko's best kernel is the shipped one"
    if not g["valid"]:
        return f"Geko rejected its output (error {g['error']})"
    if g["uplift_pct"] <= 0:
        return f"Geko found nothing faster than shipped ({g['uplift_pct']:+.1f}%)"
    thr = f" of {(g['up_thr'] - 1) * 100:+.0f}%" if g["up_thr"] else ""
    return f"Geko's {g['uplift_pct']:+.1f}% vs shipped is below its threshold{thr}"


def bench_root(args):
    """The hipBLASLt checkout whose hipblaslt-bench times kernels, as Geko's does, or None."""
    for c in (args.hipblaslt, geko_option(args.geko, "--hipblaslt") if args.geko else None,
              os.environ.get("GEKO_HIPBLASLT_PATH")):
        if c and (Path(c).expanduser() / BENCH).is_file():
            return str(Path(c).expanduser().resolve())
    return None


class Children:
    """Runs hipBLASLt stages in child processes, each naming its own library."""

    def __init__(self, args, outdir):
        self.outdir = outdir
        env = dict(os.environ)
        env.pop(LIBRARY_ENV, None)
        env["HIP_VISIBLE_DEVICES"] = str(args.eval_gpu)
        if args.lib:
            env["GEKO_HIPBLASLT_LIB"] = str(Path(args.lib).resolve())
        self.env = env
        self.common = {"hipblaslt": bench_root(args), "bench_duration": args.bench_duration,
                       "bench_dir": str(outdir / "bench"), "max_workspace": args.max_workspace,
                       "tolerance": args.tolerance, "min_speedup": args.min_speedup}

    def run(self, stage, library, plan):
        plan_file = self.outdir / f"{stage}_in.json"
        out_file = self.outdir / f"{stage}_out.json"
        plan_file.write_text(json.dumps(dict(self.common, **plan), indent=2))
        out_file.unlink(missing_ok=True)
        env = dict(self.env)
        env[LIBRARY_ENV] = str(library)
        log = self.outdir / f"{stage}.log"
        with open(log, "w") as f:
            proc = subprocess.Popen([sys.executable, __file__, "--_stage", stage, "--_plan",
                                     str(plan_file), "--_out", str(out_file)],
                                    env=env, stdout=subprocess.PIPE, stderr=f, text=True)
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
            proc.wait()
        if proc.returncode != 0 or not out_file.exists():
            tail = log.read_text().strip().splitlines()[-5:]
            print(f"      ERROR: {stage} failed (exit {proc.returncode}); see {log}")
            for line in tail:
                print(f"        {line}")
            return None
        return json.loads(out_file.read_text())


def check_library(library_root):
    if library_root is None:
        return [f"{LIBRARY_ENV} is unset; set it to the user kernel library to add winners "
                "to, e.g. export HIPBLASLT_USER_KERNEL_LIBRARY=$PWD/user_kernel_lib"]
    if not library_root.is_absolute():
        return [f"{LIBRARY_ENV}={library_root} is not absolute; hipBLASLt would not use it"]
    problems = []
    for d in (library_root, library_root / "objects"):
        if d.exists() and d.stat().st_mode & 0o022:
            problems.append(f"{d} is group- or world-writable; hipBLASLt refuses to use it "
                            f"(chmod go-w {d})")
    return problems


def scratch_copy(library_root, scratch):
    """A throwaway copy of the library, so measuring never writes to the real one."""
    shutil.rmtree(scratch, ignore_errors=True)
    if library_root.is_dir():
        shutil.copytree(library_root, scratch)
    else:
        scratch.mkdir(parents=True)
    for d in [scratch, *(p for p in scratch.rglob("*") if p.is_dir())]:
        d.chmod(0o700)


def fmt(v, f="{:.2f}"):
    return f.format(v) if isinstance(v, (int, float)) else "-"


def report(entries, outdir, dry_run):
    print("\n" + "=" * 110)
    print(f"{'size':40} {'shipped':>9} {'tuned':>9} {'tuned vs':>9}  "
          f"{'before':13} {'now':12} result")
    print(f"{'':40} {'us':>9} {'us':>9} {'shipped':>9}  "
          f"{'this run':13} {'selected':12}")
    for e in entries:
        g, x = e["geko"], e.get("eval") or {}
        before = "-" if not x.get("current") else (
            f"user {fmt(x.get('current_us'))}" if x["current"] == "user kernel" else "shipped")
        print(f"{label(Problem(**g['problem'])):40} {fmt(g['shipped_us']):>9} "
              f"{fmt(g['tuned_us']):>9} {fmt(g['shipped_us'] / g['tuned_us'], '{:.2f}x'):>9}  "
              f"{before:13} {e.get('selected', '-'):12} {e['result']}")
    print("shipped and tuned times are Geko's hipblaslt-bench measurements. 'before this run' "
          "is what a tuned kernel\nhas to beat: the shipped kernel, or a user kernel an earlier "
          "run added, with its hipblaslt-bench time\nfrom this run; the result then compares "
          "it with the tuned kernel timed the same way.")

    counts = {}
    for e in entries:
        counts[e["outcome"]] = counts.get(e["outcome"], 0) + 1
    order = ["added", "would add", "kept", "rejected", "skipped", "error"]
    print("\n" + ", ".join(f"{counts[o]} {o}" for o in order if o in counts)
          + f"  (of {len(entries)} sizes)" + ("  [dry run: library not modified]" if dry_run else ""))

    flat = []
    for e in entries:
        g, x = e["geko"], e.get("eval") or {}
        flat.append(dict(
            g["problem"], geko_shipped_us=g["shipped_us"], geko_tuned_us=g["tuned_us"],
            geko_uplift_pct=g["uplift_pct"], geko_kept=g["kept"],
            shipped_us=x.get("shipped_us"), current=x.get("current"),
            current_us=x.get("current_us"), tuned_us=x.get("tuned_us"),
            tuned_bench_us=x.get("tuned_bench_us"),
            speedup_vs_current=x.get("speedup_vs_current"),
            speedup_vs_shipped=x.get("speedup_vs_shipped"), rel_diff=x.get("rel_diff"),
            tuned_index=x.get("tuned_index"), outcome=e["outcome"], result=e["result"],
            selected=e.get("selected"), tuned_kernel=g["kernel"]))
    with open(outdir / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    (outdir / "report.json").write_text(json.dumps(flat, indent=2))
    print(f"report: {outdir / 'report.csv'}")


def replay(args, library_root, children, outdir):
    mapped = saved_mappings(library_root) if library_root.is_dir() else []
    if not mapped:
        return f"{library_root} maps no sizes; nothing to replay"
    print(f"library   {library_root} ({len(mapped)} mapped sizes)")
    print(f"output    {outdir}")
    t0 = time.time()
    print(f"\n[replay] fresh process with only {LIBRARY_ENV} set (GPU {args.eval_gpu})")
    root = children.common["hipblaslt"]
    print(f"      timing with {Path(root) / BENCH}" if root else
          "      no hipblaslt-bench found (--hipblaslt); reporting selection and accuracy only")
    sizes = [{"key": size_key(p), "arch": arch, "problem": asdict(p)}
             for arch, p, _, _ in mapped]
    res = children.run("replay", library_root, {"sizes": sizes})
    if res is None:
        return "replay failed"

    print("\n" + "=" * 118)
    print(f"{'size':40} {'arch':8} {'selected':12} {'shipped':>9} {'selected':>9} {'vs':>8}  result")
    print(f"{'':40} {'':8} {'':12} {'us':>9} {'us':>9} {'shipped':>8}")
    rows = []
    for s in sizes:
        r, p = res.get(s["key"], {}), Problem(**s["problem"])
        if r.get("user"):
            bad = [n for n in ("rel_diff", "shipped_rel_diff")
                   if r.get(n) is not None and r[n] > args.tolerance]
            if "rel_diff" in bad:
                result = f"WARNING: output differs from the reference (rel {r['rel_diff']:.3g})"
            elif r.get("shipped_rel_diff") is None:
                result = "user kernel (hipBLASLt ships no kernel to compare with)"
            elif r.get("speedup") is None:
                result = "user kernel (not timed)"
            else:
                result = f"user kernel, {r['speedup']:.2f}x vs shipped"
        else:
            result = r.get("error") or ("WARNING: mapped in the library, but the heuristic "
                                        "returns the shipped kernel")
        sel = "user kernel" if r.get("user") else ("shipped" if r else "-")
        print(f"{label(p):40} {s['arch']:8} {sel:12} {fmt(r.get('shipped_us')):>9} "
              f"{fmt(r.get('selected_us')):>9} {fmt(r.get('speedup'), '{:.2f}x'):>8}  {result}")
        rows.append(dict(s["problem"], arch=s["arch"], selected=sel,
                         selected_index=r.get("index"), shipped_us=r.get("shipped_us"),
                         selected_us=r.get("selected_us"), speedup_vs_shipped=r.get("speedup"),
                         rel_diff=r.get("rel_diff"), result=result))
    print("times are hipblaslt-bench us/call, measured as Geko measures; outputs are compared "
          "with a host reference.")
    with open(outdir / "replay.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (outdir / "replay.json").write_text(json.dumps(rows, indent=2))
    print(f"report: {outdir / 'replay.csv'}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


def main(args):
    library_root = user_kernel_library()
    problems = check_library(library_root)
    if problems:
        return "setup problems:\n  " + "\n  ".join(problems)
    devices = geko_option(args.geko, "--devices") if args.geko else None
    args.eval_gpu = args.eval_gpu or (devices or "0").split(",")[0]

    if args.replay:
        outdir = Path.cwd() / f"replay_{library_root.name}"
        outdir.mkdir(parents=True, exist_ok=True)
        return replay(args, library_root, Children(args, outdir), outdir)

    print(f"library   {library_root}" + ("" if library_root.is_dir() else " (new)"))
    t0 = time.time()
    step(1, "Geko")
    if args.geko:
        log = Path.cwd() / f"geko_{time.strftime('%Y%m%d_%H%M%S')}.log"
        rc, geko_dir = run_geko(args.geko, log)
        if geko_dir is None:
            return f"cannot tell where Geko wrote its run; see {log}"
        if rc != 0:
            print(f"      WARNING: geko exited {rc}; using whatever results it produced")
    else:
        geko_dir = args.geko_workdir.resolve()
        print(f"      reusing the Geko run in {geko_dir}")
    outdir = geko_dir / "user_kernels"
    outdir.mkdir(parents=True, exist_ok=True)
    if args.geko:
        shutil.move(log, outdir / "geko.log")
    print(f"      geko run  {geko_dir}")
    print(f"      output    {outdir}")

    geko = read_geko(geko_dir)
    if geko is None:
        return f"Geko produced no results/raw_results.csv in {geko_dir}"
    payloads = geko_payloads(geko_dir)
    entries = []
    for key, g in geko.items():
        e = {"key": key, "geko": g}
        if not g["supported"]:
            e.update(outcome="skipped", result=f"skipped: {g['problem']['a_type']} buffers "
                     "cannot be checked by this script")
        entries.append(e)
    kept_n = sum(1 for e in entries if e["geko"]["kept"])
    print(f"      Geko kept faster kernels for {kept_n} of {len(entries)} sizes, "
          f"in {len(payloads)} payload(s)")

    children = Children(args, outdir)
    step(2, f"checking each tuned kernel against today's selection (GPU {args.eval_gpu})")
    root = children.common["hipblaslt"]
    print(f"      timing with {Path(root) / BENCH}" if root else
          "      no hipblaslt-bench found (--hipblaslt); a size that already has a user "
          "kernel cannot be compared")
    candidates = [e for e in entries if "outcome" not in e]
    if kept_n and not payloads:
        return f"Geko kept kernels but left no payload under {geko_dir / 'build/library'}"
    if candidates and payloads:
        scratch = outdir / "scratch_lib"
        scratch_copy(library_root, scratch)
        res = children.run("measure", scratch, {
            "payloads": [str(p.dat) for p in payloads],
            "entries": [{"key": e["key"], "geko": e["geko"]} for e in candidates]})
        if res is None:
            return "measuring failed"
        for e in candidates:
            x = e["eval"] = res[e["key"]]
            e["outcome"], e["result"] = x["outcome"], x["result"]
            if args.dry_run and x["outcome"] == "add":
                e["outcome"], e["result"] = "would add", "would add: " + x["result"]
    else:
        for e in candidates:
            e.update(outcome="skipped", result="skipped: no payload to measure")
        print("      nothing to measure")

    step(3, f"updating {library_root}")
    to_add = [e for e in entries if e["outcome"] == "add"]
    if args.dry_run:
        print("      dry run: not modifying the library")
    elif not to_add:
        print("      nothing to add")
    else:
        library_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        res = children.run("add", library_root, {
            "entries": [{"key": e["key"], "problem": e["geko"]["problem"],
                         "payload": e["eval"]["payload"], "shard": e["eval"]["shard"]}
                        for e in to_add]})
        if res is None:
            return "adding to the library failed"
        for e in to_add:
            r = res.get(e["key"], {})
            if r.get("mapped"):
                e.update(outcome="added", result="added: " + e["result"])
            else:
                e.update(outcome="error", result="error: " + r.get("error", "not mapped"))

    step(4, "checking what a fresh process selects")
    if library_root.is_dir():
        res = children.run("confirm", library_root, {
            "entries": [{"key": e["key"], "problem": e["geko"]["problem"]}
                        for e in entries if e["geko"]["supported"]]}) or {}
        for e in entries:
            sel = res.get(e["key"])
            if sel:
                e["selected"] = "user kernel" if sel["user"] else "shipped"
                if e["outcome"] == "added" and not sel["user"]:
                    print(f"      WARNING: {label(Problem(**e['geko']['problem']))} was added but "
                          "the heuristic still returns the shipped kernel")
    else:
        print(f"      {library_root} does not exist yet; every size uses the shipped kernel")
        for e in entries:
            e["selected"] = "shipped"

    report(entries, outdir, args.dry_run)
    print(f"\nto use the library:  {LIBRARY_ENV}={library_root} <your workload>")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# hipBLASLt stages (child processes)
# ---------------------------------------------------------------------------

def worker(stage, plan_file, out_file):
    import numpy as np

    plan = json.loads(Path(plan_file).read_text())
    max_ws = plan["max_workspace"]
    hb = HipBLASLt()
    info = hb.info()
    print(f"      hipBLASLt {info['version']} (git {info['git_revision']}) at {info['realpath']}",
          flush=True)

    def rel_diff(got, ref):
        return float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))

    def bench_us(name, kernels):
        """hipblaslt-bench us/call of each (problem, solution index), or None where it
        reports nothing. Geko's own routine: a short probe sets the iterations so the
        warmup and the timed run each last bench_duration. Runs in this process's
        environment, so the bench sees this stage's library and GPU.
        """
        from geko import bench
        from geko.bench import log

        rows = log.update([{
            "transA": p.trans[0], "transB": p.trans[1], "batch_count": p.batch,
            "M": p.m, "N": p.n, "K": p.k, "a_type": TENSILE_TYPES[p.a_type][1],
            "b_type": TENSILE_TYPES[p.b_type][1], "c_type": TENSILE_TYPES[p.c_type][1],
            "d_type": TENSILE_TYPES[p.c_type][1], "compute_type": "f32_r", "alpha": 1.0,
        } for p, _ in kernels], beta=False, flush=True)[0]
        for row, (_, index) in zip(rows, kernels):
            row.update(algo_method=2, solution_index=index)  # 2: by index
        d = Path(plan["bench_dir"])
        d.mkdir(parents=True, exist_ok=True)
        log.dump(rows, d / f"{name}.yaml")
        df = bench.standard_benchmark(plan["hipblaslt"], d / f"{name}.yaml", d / f"{name}.out",
                                      devices=[0], duration=plan["bench_duration"])
        us = log.realign_rows(rows, df)["us"]
        return [float(u) if u == u and u > 0 else None for u in us]

    def choices(uk, p):
        """(today's selection, whether it is a user kernel, the shipped selection)."""
        top = uk.heuristics(p, 2, max_ws)
        if not top:
            return None, False, None
        if uk.is_user_kernel(top[0].algo):
            return top[0].algo, True, (top[1].algo if len(top) > 1 else None)
        return top[0].algo, False, top[0].algo

    def register(uk, dats):
        """dat path -> (Payload, {shard index: user kernel index})."""
        out = {}
        for dat in dats:
            pl = read_payload(Path(dat))
            idx = uk.register(pl.dat, pl.co)
            if len(idx) != len(pl.shard_indices):
                raise RuntimeError(f"{pl.dat.name}: registered {len(idx)} kernels, "
                                   f"payload lists {len(pl.shard_indices)}")
            out[dat] = (pl, dict(zip(pl.shard_indices, idx)))
        return out

    def find(registered, p, name):
        """(payload, shard) of Geko's tuned kernel for p: by name, else the size table."""
        for dat, (pl, _) in registered.items():
            for shard, kname in pl.names.items():
                q = problem_of(pl.problem_types[shard], p.m, p.n, p.batch, p.k)
                if kname == name and size_key(q) == size_key(p):
                    return dat, shard
        for dat, (pl, _) in registered.items():
            for q, shard in pl.sizes:
                if size_key(q) == size_key(p):
                    return dat, shard
        return None

    out = {}
    with UserKernels(hb) as uk:
        if stage == "measure":
            registered = register(uk, plan["payloads"])
            n = sum(len(v[1]) for v in registered.values())
            print(f"      registered {n} tuned kernel(s) from {len(registered)} payload(s) "
                  "in a scratch library", flush=True)
            pending = []  # sizes whose current kernel is a user kernel: timed below

            def decide(r, current_us, tuned_us):
                sp = r["speedup_vs_current"] = current_us / tuned_us
                r["current_us"] = current_us
                if sp < plan["min_speedup"]:
                    r.update(outcome="kept", result=f"kept {r['current']}: tuned is {sp:.4f}x "
                             f"vs it, below --min-speedup {plan['min_speedup']}")
                else:
                    r.update(outcome="add", result=f"{sp:.2f}x faster than {r['current']}")

            def show(p, r):
                now = (f"  user kernel {r['current_us']:8.2f}"
                       if r.get("current") == "user kernel" and "current_us" in r else "")
                tuned = r.get("tuned_bench_us", r.get("tuned_us"))
                print(f"      {label(p):40} shipped {fmt(r.get('shipped_us')):>8}{now}  tuned "
                      f"{fmt(tuned):>8} us  -> {r['result']}", flush=True)

            for e in plan["entries"]:
                g, p = e["geko"], Problem(**e["geko"]["problem"])
                r = out[e["key"]] = {}
                cur, cur_user, _ = choices(uk, p)
                hit = find(registered, p, g["kernel"])
                if cur is None:
                    r.update(outcome="skipped",
                             result="skipped: hipBLASLt has no kernel for this size")
                    continue
                if hit is None:
                    r.update(outcome="error" if g["kept"] else "skipped",
                             result="error: Geko's tuned kernel is not in its payloads"
                             if g["kept"] else "skipped: no tuned kernel")
                    continue
                dat, shard = hit
                idx = registered[dat][1][shard]
                try:
                    ref, (got,) = uk.run(p, [MatmulAlgo.from_index(idx, max_ws)], max_ws)
                except Exception as ex:  # noqa: BLE001
                    r.update(outcome="error", result=f"error: {ex}")
                    print(f"      {label(p):40} {r['result']}", flush=True)
                    continue
                rel = rel_diff(got, ref)
                r.update(shipped_us=g["shipped_us"], tuned_us=g["tuned_us"],
                         speedup_vs_shipped=g["shipped_us"] / g["tuned_us"],
                         current="user kernel" if cur_user else "shipped",
                         current_index=cur.index, tuned_index=idx, payload=dat, shard=shard,
                         rel_diff=rel)
                if not g["kept"]:
                    r.update(outcome="skipped", result=f"skipped: {not_kept_reason(g)}")
                elif rel > plan["tolerance"]:
                    r.update(outcome="rejected",
                             result=f"rejected: output differs from the reference (rel {rel:.3g})")
                elif cur_user and cur.index == idx:
                    r.update(outcome="kept", result="kept: this kernel is already in the library")
                elif cur_user:
                    pending.append((e["key"], p, cur.index, idx))
                    continue
                else:
                    decide(r, g["shipped_us"], g["tuned_us"])
                show(p, r)

            if pending and not plan["hipblaslt"]:
                for key, p, _, _ in pending:
                    out[key].update(outcome="error", result="error: comparing with the user "
                                    "kernel already there needs hipblaslt-bench; pass --hipblaslt")
                    show(p, out[key])
            elif pending:
                print(f"      timing {len(pending)} size(s) that already have a user kernel "
                      "with hipblaslt-bench", flush=True)
                try:
                    before = bench_us("current", [(p, c) for _, p, c, _ in pending])
                    after = bench_us("tuned", [(p, t) for _, p, _, t in pending])
                except Exception as ex:  # noqa: BLE001
                    before = after = [None] * len(pending)
                    print(f"      hipblaslt-bench failed: {ex}", flush=True)
                for (key, p, _, _), b, a in zip(pending, before, after):
                    r = out[key]
                    if b is None or a is None:
                        r.update(outcome="error", result="error: hipblaslt-bench timed nothing; "
                                 f"see {plan['bench_dir']}")
                    else:
                        r["tuned_bench_us"] = a
                        decide(r, b, a)
                    show(p, r)

        elif stage == "add":
            root = user_kernel_library()
            registered = register(uk, sorted({e["payload"] for e in plan["entries"]}))
            for dat, (pl, _) in registered.items():
                if not saved_in(root, pl.co):
                    raise RuntimeError(f"{pl.dat.name} was not saved to {root}; see hipBLASLt's "
                                       "warnings in this log")
            for e in plan["entries"]:
                p = Problem(**e["problem"])
                idx = registered[e["payload"]][1][e["shard"]]
                uk.set_exact_match(idx, p.m, p.n, p.batch, p.k)
                now = uk.heuristic(p, max_ws)
                ok = now is not None and now.algo.index == idx
                out[e["key"]] = {"mapped": ok, "index": idx} if ok else \
                    {"mapped": False, "error": "mapped, but the heuristic does not return it"}
                print(f"      mapped {label(p)} to user kernel {idx}", flush=True)
            print(f"      {len(registered)} payload(s) saved, "
                  f"{sum(v['mapped'] for v in out.values())} size(s) mapped", flush=True)

        elif stage == "confirm":
            for e in plan["entries"]:
                best = uk.heuristic(Problem(**e["problem"]), max_ws)
                if best is not None:
                    out[e["key"]] = {"user": bool(uk.is_user_kernel(best.algo)),
                                     "index": best.algo.index}
            registered, mapped = uk.counts()
            print(f"      loaded {registered} kernel(s) and {mapped} mapping(s) at device init",
                  flush=True)

        elif stage == "replay":
            registered, mapped = uk.counts()
            to_time = []
            for s in plan["sizes"]:
                p = Problem(**s["problem"])
                r = out[s["key"]] = {"user": False}
                cur, cur_user, shipped = choices(uk, p)
                if cur is None:
                    r["error"] = "hipBLASLt returns no kernel for this size"
                    continue
                r["index"] = cur.index
                if not cur_user:
                    continue
                algos = [cur] + ([shipped] if shipped else [])
                try:
                    ref, outs = uk.run(p, algos, max_ws)
                except Exception as ex:  # noqa: BLE001
                    r["error"] = f"error: {ex}"
                    continue
                r.update(user=True, rel_diff=rel_diff(outs[0], ref))
                if shipped:
                    r["shipped_rel_diff"] = rel_diff(outs[1], ref)
                    to_time.append((s["key"], p, cur.index, shipped.index))
                print(f"      {label(p):40} user kernel {cur.index}", flush=True)
            print(f"      loaded {registered} kernel(s) and {mapped} mapping(s) at device init",
                  flush=True)
            if to_time and plan["hipblaslt"]:
                print(f"      timing {len(to_time)} size(s) with hipblaslt-bench", flush=True)
                try:
                    sel = bench_us("selected", [(p, u) for _, p, u, _ in to_time])
                    ship = bench_us("shipped", [(p, s) for _, p, _, s in to_time])
                except Exception as ex:  # noqa: BLE001
                    sel = ship = [None] * len(to_time)
                    print(f"      hipblaslt-bench failed: {ex}", flush=True)
                for (key, _, _, _), u, s in zip(to_time, sel, ship):
                    out[key].update(selected_us=u, shipped_us=s,
                                    speedup=s / u if u and s else None)

    Path(out_file).write_text(json.dumps(out, indent=2))
    return 0


def parse_args():
    argv = sys.argv[1:]
    geko = []
    if "--" in argv:
        i = argv.index("--")
        argv, geko = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geko-workdir", type=Path, default=None,
                    help="use a Geko run that already finished instead of running Geko")
    ap.add_argument("--lib", default=None,
                    help="libhipblaslt.so to measure and register with (default: "
                         "$GEKO_HIPBLASLT_LIB, else the loader's search path)")
    ap.add_argument("--eval-gpu", default=None,
                    help="GPU for the hipBLASLt measurements (default: first of Geko's "
                         "--devices, else 0)")
    ap.add_argument("--min-speedup", type=float, default=1.03,
                    help="add a kernel only if it is at least this much faster than what "
                         "hipBLASLt selects today (default: 1.03)")
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="max relative difference from the host reference")
    ap.add_argument("--dry-run", action="store_true",
                    help="run Geko and measure, but do not modify the library")
    ap.add_argument("--replay", action="store_true",
                    help="tune nothing and only read the library: in a fresh process, report "
                         "what the heuristic selects for every size the library maps")
    ap.add_argument("--hipblaslt", default=None, metavar="PATH",
                    help="hipBLASLt checkout whose build/release/clients/hipblaslt-bench "
                         "times kernels (default: Geko's --hipblaslt, else "
                         "$GEKO_HIPBLASLT_PATH)")
    ap.add_argument("--bench-duration", type=float, default=1.0, metavar="S",
                    help="seconds hipblaslt-bench spends warming up, and again timing, each "
                         "kernel (default: 1.0, as Geko does)")
    ap.add_argument("--max-workspace", type=int, default=128 * 1024 * 1024)
    ap.add_argument("--_stage", help=argparse.SUPPRESS)
    ap.add_argument("--_plan", help=argparse.SUPPRESS)
    ap.add_argument("--_out", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.geko = geko
    if not args._stage and not args.replay and bool(geko) == bool(args.geko_workdir):
        ap.error("give a Geko command after `--`, or --geko-workdir, or --replay")
    if geko and not any(Path(t).name == "geko" for t in geko[:3]):
        ap.error(f"expected a geko command after `--`, got {' '.join(geko[:3])!r}")
    return args


if __name__ == "__main__":
    a = parse_args()
    r = worker(a._stage, a._plan, a._out) if a._stage else main(a)
    sys.exit(f"\nERROR: {r}" if isinstance(r, str) else r)
