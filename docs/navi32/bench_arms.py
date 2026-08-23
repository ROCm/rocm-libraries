#!/usr/bin/env python3
"""
Interleaved multi-arm benchmark for the navi32 campaign.

An arm is a (library, extra-env) pair. Arms are interleaved per shape so machine drift
cancels within a (shape, rep) rather than across the run.

NAVI32 EMULATION -- both halves are applied to every arm:
  HIPBLASLT_BENCH_CU_MASK=60   real 60-CU execution (30 of 48 WGPs). Verified: 62.4% of
                               unmasked throughput vs an ideal 62.5%.
  --sm_count_target 60         the selector/Origami sees 60 CUs, so kernel CHOICE and the
                               StreamK grid match a 60-CU part.
Applying only the second gives correct choices timed on the wrong machine.

--fixed-iters IS MANDATORY HERE. The arms differ in library size (navi32's 60 kernels vs
navi31's 238) and tiered iteration counts charge one-time library init unevenly -- an effect
worth ~5 points in an earlier campaign on this workspace.

Put an A/A arm (a duplicate of the baseline) LAST so the pair brackets the whole interleave
and measures the maximum arm-position drift, which an A/A in the middle cannot see.
"""

import argparse
import csv
import json
import os
import pathlib
import subprocess
import statistics
import sys
import time

BENCH = "/home/vmijovic/navi32/src/projects/hipblaslt/build/release/clients/hipblaslt-bench"
LOCK = "/home/vmijovic/navi32/.gpu.lock"


def parse(out):
    """-> (gflops, us, kernel) from a bench run."""
    hdr, g, us, kern = None, 0.0, 0.0, ""
    for line in out.splitlines():
        s = line.strip()
        if "hipblaslt-Gflops" in s:
            hdr = [x.strip() for x in s.split(":", 1)[-1].split(",")]
        elif hdr and (s.startswith("T,") or s.startswith("N,")):
            f = [x.strip() for x in s.split(",")]
            try:
                g = float(f[hdr.index("hipblaslt-Gflops")])
                us = float(f[hdr.index("us")])
            except (ValueError, IndexError):
                pass
        elif s.startswith("--Solution name:"):
            kern = s.split(":", 1)[1].strip()
    return g, us, kern


def run_one(lib, env_extra, shape, cus, iters, timeout, no_mask=False):
    m, n, k = shape["M"], shape["N"], shape["K"]
    cmd = [BENCH, "--api_method", "c", "-m", str(m), "-n", str(n), "-k", str(k),
           "--transA", "T", "--transB", "N", "--lda", str(k), "--ldb", str(k),
           "--ldc", str(m), "--ldd", str(m),
           "--a_type", "f16_r", "--b_type", "f16_r", "--c_type", "f16_r",
           "--d_type", "f16_r", "--compute_type", "f32_r",
           "--algo_method", "heuristic", "--requested_solution", "1",
           "--initialization", "trig_float", "--print_kernel_info",
           "--cold_iters", str(max(1, iters // 3)), "--iters", str(iters),
           "--sm_count_target", str(cus)]
    env = dict(os.environ, HIPBLASLT_TENSILE_LIBPATH=lib, **env_extra)
    if not no_mask:
        env["HIPBLASLT_BENCH_CU_MASK"] = str(cus)
    try:
        p = subprocess.run(["flock", "-w", "300", LOCK] + cmd, env=env,
                           capture_output=True, text=True, timeout=timeout)
        g, us, kern = parse(p.stdout)
        return (g, us, kern, "ok" if g > 0 else "error")
    except subprocess.TimeoutExpired:
        # INTERMITTENT HANG, observed on this workspace: a run occasionally emits its
        # result row and then never exits, holding the GPU lock. Isolated retries of the
        # same (shape, arm) always pass, so it is a teardown race rather than a bad shape.
        # Kill the whole process group -- abandoning it wedges every later arm.
        subprocess.run(["pkill", "-9", "-f", "hipblaslt-bench"], capture_output=True)
        return (0.0, 0.0, "", "timeout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="name=libdir[:VAR=VAL,...]")
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cus", type=int, default=60)
    ap.add_argument("--fixed-iters", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=35)
    ap.add_argument("--no-cu-mask", action="store_true",
                    help="skip HIPBLASLT_BENCH_CU_MASK. The masked stream hangs ~37%% of "
                         "runs (measured: 0/8 timeouts unmasked vs 3/8 masked), which is "
                         "unusable for a 5000-run sweep. Without it selection is still "
                         "navi32-correct via --sm_count_target, but execution is on all "
                         "96 CUs, so absolute throughput is optimistic and only ARM RATIOS "
                         "are meaningful.")
    a = ap.parse_args()

    arms = []
    for spec in a.arms:
        name, _, rest = spec.partition("=")
        lib, _, envs = rest.partition(":")
        env = {}
        for pair in filter(None, envs.split(",")):
            k, _, v = pair.partition("=")
            env[k.strip()] = v.strip()
        arms.append((name, lib, env))

    shapes = json.load(open(a.shapes))["shapes"]
    if a.limit:
        shapes = shapes[:a.limit]

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():  # resume: skip (shape, arm, rep) already recorded
        with out.open() as fh:
            for r in csv.DictReader(fh):
                done.add((r["shape_id"], r["arm"], r["rep"]))
        print(f"resuming: {len(done)} rows already present")

    new = not out.exists()
    fh = out.open("a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["shape_id", "M", "N", "K", "stratum", "arm", "rep",
                    "gflops", "us", "kernel", "status", "ts"])
        fh.flush()

    total = len(shapes) * len(arms) * a.reps
    t0, cnt = time.time(), 0
    print(f"{len(arms)} arms x {len(shapes)} shapes x {a.reps} reps = {total} measurements")
    print(f"emulation: sm_count_target={a.cus}"
          f"{'' if a.no_cu_mask else f', CU mask={a.cus}'}"
          f"  fixed-iters={a.fixed_iters}")

    for si, sh in enumerate(shapes):
        for rep in range(a.reps):
            for name, lib, env in arms:
                key = (sh["shape_id"], name, str(rep))
                if key in done:
                    continue
                g, us, kern, st = run_one(lib, env, sh, a.cus, a.fixed_iters, a.timeout,
                                          a.no_cu_mask)
                w.writerow([sh["shape_id"], sh["M"], sh["N"], sh["K"],
                            sh.get("stratum", ""), name, rep,
                            f"{g:.2f}", f"{us:.3f}", kern, st, f"{time.time():.0f}"])
                cnt += 1
        fh.flush()
        if (si + 1) % 25 == 0:
            el = time.time() - t0
            rate = cnt / el if el else 0
            print(f"[{si+1}/{len(shapes)}] {cnt} runs  {el/60:.1f}m  "
                  f"{rate:.2f}/s  eta {(total-len(done)-cnt)/rate/60:.0f}m", flush=True)
    fh.close()
    print(f"done: {cnt} new rows -> {out}")


if __name__ == "__main__":
    main()
