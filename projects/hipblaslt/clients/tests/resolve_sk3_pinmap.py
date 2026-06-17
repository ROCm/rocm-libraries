#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Resolve per-build StreamK (SK) solution indices into a pin-map.

hipBLASLt solution indices are library-build-specific, so the tile-boundary
stress suite cannot hard-code them. This tool enumerates the build's solutions
with ``hipblaslt-bench --algo_method all --print_kernel_info`` (which prints
``--Solution index`` / ``--kernel name`` triples) and groups every SK kernel by
transpose + macro-tile shape, emitting a JSON pin-map:

    { "TN_MT128x192": [17499, 18543, 18800, 19057], "NN_MT256x256": [...], ... }

The pin-map is consumed by:
  * generate_tile_boundary_stress.py --pin-map  (Leg B: pin SK3 at K<=256)
  * the pinned large-K fault hunt (Leg A pinned), which exercises each SK
    kernel's store edge directly (the heuristic tends not to pick SK kernels,
    so without pinning the overrunning kernel is never run).

``getAllAlgos`` returns all solutions for a (transpose, type) combo regardless
of M/N/K, so one bench run per transpose enumerates every SK index. Run at a
tile-aligned (safe) size to avoid faulting mid-enumeration.

Usage (run bench):
  python3 resolve_sk3_pinmap.py --bench <build>/clients/hipblaslt-bench \
      --out sk3_pinmap.json
Usage (parse existing --print_kernel_info logs):
  python3 resolve_sk3_pinmap.py --tn-log tn.log --nn-log nn.log --out sk3_pinmap.json
"""

import argparse
import json
import re
import subprocess
import sys

MT_RE = re.compile(r"_MT(\d+)x(\d+)x\d+_")
SK_RE = re.compile(r"_SK[1-9]\d*_")  # SK1+ (StreamK); SK0 is data-parallel

BENCH_TYPES = ["--a_type", "bf16_r", "--b_type", "bf16_r", "--c_type", "bf16_r",
               "--d_type", "bf16_r", "--compute_type", "f32_r"]


def run_bench(bench, transA, transB, size):
    m, n, k = size
    cmd = [bench, "--transA", transA, "--transB", transB,
           "-m", str(m), "-n", str(n), "-k", str(k),
           "--algo_method", "all", "--print_kernel_info"] + BENCH_TYPES
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def parse_kernel_info(text):
    """Yield (index, kernel_name) from --print_kernel_info output."""
    idx = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("--Solution index:"):
            idx = s.split(":", 1)[1].strip()
        elif s.startswith("--kernel name:") and idx is not None:
            yield idx, s.split(":", 1)[1].strip()
            idx = None


def collect(text, tag, pinmap):
    for idx, kn in parse_kernel_info(text):
        if not SK_RE.search(kn):
            continue
        m = MT_RE.search(kn)
        if not m:
            continue
        key = f"{tag}_MT{m.group(1)}x{m.group(2)}"
        try:
            i = int(idx)
        except ValueError:
            continue
        pinmap.setdefault(key, set()).add(i)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", help="path to hipblaslt-bench (to run enumeration)")
    ap.add_argument("--tn-log", help="existing TN --print_kernel_info log to parse")
    ap.add_argument("--nn-log", help="existing NN --print_kernel_info log to parse")
    ap.add_argument("--tn-size", default="2816,2112,2048",
                    help="safe (tile-aligned) M,N,K for the TN bench run")
    ap.add_argument("--nn-size", default="4096,4096,2048",
                    help="safe M,N,K for the NN bench run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    pinmap = {}
    if args.tn_log:
        collect(open(args.tn_log).read(), "TN", pinmap)
    if args.nn_log:
        collect(open(args.nn_log).read(), "NN", pinmap)
    if args.bench:
        tn = tuple(int(x) for x in args.tn_size.split(","))
        nn = tuple(int(x) for x in args.nn_size.split(","))
        collect(run_bench(args.bench, "T", "N", tn), "TN", pinmap)
        collect(run_bench(args.bench, "N", "N", nn), "NN", pinmap)
    if not pinmap:
        print("error: nothing resolved; pass --bench or --tn-log/--nn-log",
              file=sys.stderr)
        return 1

    out = {k: sorted(v) for k, v in sorted(pinmap.items())}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(f"Wrote {args.out}: {len(out)} shapes, "
          f"{sum(len(v) for v in out.values())} SK indices", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
