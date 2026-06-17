#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Isolated-process discovery runner for the tile-boundary StreamK stress suite.

Each candidate test instance is launched in its *own* ``hipblaslt-test``
process so that a single hard GPU fault (ROCM-26298-class out-of-bounds store)
cannot poison the HIP context and mask every later case. Per case we record:

  PASS       - test passed (exit 0, ``[  PASSED  ]``)
  SKIP       - skipped at runtime (unsupported config / no solution)
  FAIL       - graceful gtest failure (e.g. NaN guard-pad mismatch or
               norm/unit check) -> a likely silent overrun or wrong numerics
  GPU_FAULT  - process aborted/crashed or "Memory access fault" -> a likely
               hard out-of-bounds store (the ROCM-26298 signature)
  TIMEOUT    - exceeded --timeout

Use the small stress-only data file (fast startup) generated with:
  python3 hipblaslt_gentest.py -I ../common/include data/matmul_stress_gtest.yaml -o stress_only.data

Then:
  python3 run_tile_boundary_stress.py \
      --binary  <build>/clients/hipblaslt-test \
      --data    stress_only.data \
      --report  tile_boundary_stress_findings.md

This is a TEST-ONLY tool; it does not modify any library/runtime code.
"""

import argparse
import concurrent.futures as cf
import csv
import os
import re
import signal
import subprocess
import sys
import time

GUARD_RE = re.compile(r"guard|pad", re.IGNORECASE)
FAULT_RE = re.compile(r"Memory access fault|HSA_STATUS_ERROR|page fault|"
                      r"GPU.*fault|aborting", re.IGNORECASE)
NORM_RE = re.compile(r"norm[_ ]?error[^\n]*", re.IGNORECASE)


def list_tests(binary, data, gtest_filter):
    """Return fully-qualified Suite.Test names matching the filter."""
    out = subprocess.run(
        [binary, "--data", data, "--gtest_list_tests", f"--gtest_filter={gtest_filter}"],
        capture_output=True, text=True).stdout
    names, suite = [], None
    for line in out.splitlines():
        if not line or line[0] in "#":
            continue
        if not line[0].isspace() and line.rstrip().endswith("."):
            suite = line.strip()
            continue
        if suite and line[:1].isspace():
            test = line.strip().split("  #")[0].strip()
            if test:
                names.append(suite + test)
    return names


def classify(rc, out):
    if rc == 0:
        if "[  SKIPPED ]" in out or " SKIPPED " in out:
            return "SKIP"
        if "[  PASSED  ]" in out:
            return "PASS"
        return "PASS"  # exit 0, no explicit marker
    if rc is None:
        return "TIMEOUT"
    if rc < 0:  # killed by signal (abort/segv) -> hard fault
        return "GPU_FAULT"
    if FAULT_RE.search(out):
        return "GPU_FAULT"
    if "[  FAILED  ]" in out:
        return "FAIL"
    return "GPU_FAULT"  # nonzero without graceful failure marker


def run_one(binary, data, name, timeout, extra):
    cmd = [binary, "--data", data, f"--gtest_filter={name}"] + extra
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc, out = p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired as e:
        rc = None
        out = (e.stdout or "") + (e.stderr or "") if isinstance(e.stdout, str) else ""
    dt = time.time() - t0
    status = classify(rc, out)
    norm = NORM_RE.search(out)
    note = norm.group(0).strip() if norm else ""
    if status in ("GPU_FAULT", "FAIL") and not note:
        fm = FAULT_RE.search(out)
        note = fm.group(0).strip() if fm else (out.strip().splitlines() or [""])[-1][:160]
    return name, status, (rc if rc is not None else "timeout"), round(dt, 2), note


def short(name):
    # matmul/stress_stress_TN_MT128x192_legA_bf16_..._TN_2816_2096_2048_...
    m = re.search(r"(stress_[A-Z]+_MT\d+x\d+_leg[AB])_.*?_(?:TN|NN|NT|TT)_"
                  r"(\d+)_(\d+)_(\d+)", name)
    if m:
        return f"{m.group(1)} M={m.group(2)} N={m.group(3)} K={m.group(4)}"
    return name


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--filter", default="*stress*")
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel isolated processes (default 4)")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--report", default="tile_boundary_stress_findings.md")
    ap.add_argument("--csv", default=None)
    ap.add_argument("gtest_args", nargs="*",
                    help="extra args passed to hipblaslt-test (after --)")
    args = ap.parse_args(argv)

    names = list_tests(args.binary, args.data, args.filter)
    if not names:
        print("error: no tests matched filter", file=sys.stderr)
        return 1
    print(f"Discovered {len(names)} instances; running isolated "
          f"(jobs={args.jobs}, timeout={args.timeout}s)...", file=sys.stderr)

    results = []
    counts = {}
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, args.binary, args.data, n, args.timeout,
                          args.gtest_args): n for n in names}
        done = 0
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            counts[r[1]] = counts.get(r[1], 0) + 1
            done += 1
            if r[1] in ("GPU_FAULT", "FAIL", "TIMEOUT") or done % 50 == 0:
                print(f"[{done}/{len(names)}] {r[1]:9} {short(r[0])} {r[4]}",
                      file=sys.stderr)

    results.sort(key=lambda r: (r[1], r[0]))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["test", "status", "rc", "seconds", "note"])
            w.writerows(results)

    order = ["GPU_FAULT", "FAIL", "TIMEOUT", "SKIP", "PASS"]
    with open(args.report, "w") as fh:
        fh.write("# Tile-boundary StreamK stress - discovery findings\n\n")
        fh.write(f"- Binary: `{args.binary}`\n- Data: `{args.data}`\n")
        fh.write(f"- Total instances: {len(names)}\n\n")
        fh.write("## Summary\n\n| status | count |\n| --- | --- |\n")
        for s in order:
            if s in counts:
                fh.write(f"| {s} | {counts[s]} |\n")
        fh.write("\n")
        for s in order:
            sub = [r for r in results if r[1] == s]
            if not sub or s == "PASS":
                continue
            fh.write(f"## {s} ({len(sub)})\n\n")
            fh.write("| case | rc | s | note |\n| --- | --- | --- | --- |\n")
            for (name, st, rc, dt, note) in sub:
                fh.write(f"| {short(name)} | {rc} | {dt} | {note[:120]} |\n")
            fh.write("\n")
        npass = counts.get("PASS", 0)
        fh.write(f"## PASS ({npass})\n\nOmitted for brevity; "
                 f"see CSV for the full list.\n")

    print("\n=== SUMMARY ===", file=sys.stderr)
    for s in order:
        if s in counts:
            print(f"  {s:9} {counts[s]}", file=sys.stderr)
    print(f"Report: {args.report}" + (f"  CSV: {args.csv}" if args.csv else ""),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
