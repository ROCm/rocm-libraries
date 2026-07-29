#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Best achieved swapqk performance per sequence length.

Runs the production chunked harness (all H heads, real launches) over a short
list of known-good configs at each L and reports the winner. chunk_sweep already
sweeps the chunk count C and o_nt internally, so this only varies the kernel
shape knobs (q_block, block_n, O-carry, V layout).

Two numbers are reported per L because they answer different questions:
  * wall-clock TF -- what the part actually delivers, INCLUDING the power
    throttle (2405 MHz / 87 W -> ~1600 MHz / 43 W within ~5 s of sustained
    load). This is the honest end-to-end figure.
  * TF @ fixed clock -- from GRBM_GUI_ACTIVE, what the kernel would deliver if
    the clock held. Use this to compare kernels; the throttle swing (20-40%) is
    larger than most config differences.

C is capped per L so the concurrent KV working set stays sane against the 32 MB
MALL (per-head KV is L*D*2*2 bytes = 8 MB at L=16K).

Usage:
    python3 l_sweep.py --lens 512,1024,2048,4096,8192,16384
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# (label, chunk_sweep args)
CONFIGS = [
    ("mq2 of16 bn32", ["--mq", "2", "--of16", "1", "--block-n", "32"]),
    ("mq2 of16 bn32 vt", ["--mq", "2", "--of16", "1", "--block-n", "32", "--vt", "1"]),
    ("mq1 of32 bn32", ["--mq", "1", "--of16", "0", "--block-n", "32"]),
    ("mq1 of32 bn32 vt", ["--mq", "1", "--of16", "0", "--block-n", "32", "--vt", "1"]),
    ("mq1 of32 bn64", ["--mq", "1", "--of16", "0", "--block-n", "64"]),
    ("mq1 of32 bn64 vt", ["--mq", "1", "--of16", "0", "--block-n", "64", "--vt", "1"]),
]

# per-head KV grows with L, so the head-chunk count has to shrink to keep the
# concurrent KV footprint near the MALL rather than thrashing it.
CHUNKS = {
    512: "6,12,24",
    1024: "6,12,24",
    2048: "4,6,12,24",
    4096: "2,4,6,12",
    8192: "1,2,4,6",
    16384: "1,2,4",
}

BEST = re.compile(r"^best:\s*([\d.]+)\s*TF at C=(\d+)\s*o_nt=(\d)", re.M)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens", default="512,1024,2048,4096,8192,16384")
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    sweep = here.parent / "chunk_sweep.py"
    lens = [int(x) for x in args.lens.split(",")]

    print(
        f"H={args.heads} D={args.head_size} B=1, wall-clock (throttled), "
        f"best over C and o_nt\n"
    )
    hdr = f"{'L':>7}" + "".join(f"{c[0][:15]:>17}" for c in CONFIGS)
    print(hdr)
    print("-" * len(hdr))

    winners = {}
    for L in lens:
        # long L takes far longer per iteration; keep the run bounded.
        iters = args.iters if L <= 4096 else max(3, args.iters // 2)
        row, best = f"{L:>7}", None
        for label, extra in CONFIGS:
            cmd = [
                sys.executable,
                str(sweep),
                "--seqlen",
                str(L),
                "--heads",
                str(args.heads),
                "--head-size",
                str(args.head_size),
                "--chunks",
                CHUNKS.get(L, "1,2,4"),
                "--ont",
                "0,1",
                "--iters",
                str(iters),
                "--warmup",
                "3",
            ] + extra
            try:
                p = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=args.timeout
                )
                m = BEST.search(p.stdout)
            except subprocess.TimeoutExpired:
                m = None
            if m is None:
                row += f"{'FAIL':>17}"
                if args.verbose:
                    print(f"\n  {L} {label}: {p.stderr.strip()[-300:]}\n")
                continue
            tf, c, ont = float(m.group(1)), int(m.group(2)), int(m.group(3))
            row += f"{tf:>12.2f} C{c:<3}"
            if best is None or tf > best[0]:
                best = (tf, label, c, ont)
        print(row, flush=True)
        if best:
            winners[L] = best

    print(f"\n{'L':>7}{'best TF':>10}{'config':>19}{'C':>4}{'o_nt':>6}")
    print("-" * 46)
    for L, (tf, label, c, ont) in winners.items():
        print(f"{L:>7}{tf:>10.2f}{label:>19}{c:>4}{ont:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
