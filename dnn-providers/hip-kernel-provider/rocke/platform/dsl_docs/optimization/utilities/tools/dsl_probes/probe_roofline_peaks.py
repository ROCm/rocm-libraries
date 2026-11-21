#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measure the per-instruction issue costs that a cycle budget has to be priced in.

Builds and runs ``probe_roofline_peaks.hip`` under rocprofv3 and divides the
dispatch's GPU-active cycles by the operations each SIMD retired. The result is
cycles per instruction per SIMD -- clock-independent, and therefore usable on
parts that power-throttle, where wall-clock FLOPS numbers are not reproducible.

Feed the numbers to ``probe_cycle_budget.py --anchor-mma/--anchor-valu/
--anchor-transcendental``.

    python probe_roofline_peaks.py --build --cus <compute-units-on-the-part>

Each class is measured twice: at one wave per SIMD (the issue cost with the SIMD
to itself) and saturated (the real ceiling once many waves compete for the port).
A kernel that is nowhere near the saturated number is stalled, not issue-bound.

``--build`` compiles the .hip next to this file with hipcc; pass ``--exe`` to use
a binary you built yourself. ``measure()`` is the programmatic entry point.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

# NACC in the .hip: independent accumulator chains per iteration.
NACC = 8
FLOP_PER_OP = {"mma": 8192.0, "valu": 64.0, "exp": 32.0}
HERE = Path(__file__).resolve().parent


def build(exe: Path, arch: str) -> Path:
    """Compile the companion .hip for ``arch``."""
    src = HERE / "probe_roofline_peaks.hip"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    if shutil.which("hipcc") is None:
        raise SystemExit("hipcc not found on PATH")
    subprocess.run(
        ["hipcc", f"--offload-arch={arch}", "-O3", "-o", str(exe), str(src)],
        check=True,
    )
    return exe


def _dispatch_cycles(outdir: Path) -> float | None:
    """Largest GRBM_GUI_ACTIVE among the peak dispatches (the timed one)."""
    per: dict[str, float] = collections.defaultdict(float)
    for path in outdir.glob("**/*counter_collection.csv"):
        with open(path) as f:
            for row in csv.DictReader(f):
                if "peak" not in row.get("Kernel_Name", ""):
                    continue
                if row.get("Counter_Name") != "GRBM_GUI_ACTIVE":
                    continue
                try:
                    per[row["Dispatch_Id"]] += float(row["Counter_Value"])
                except (KeyError, ValueError):
                    continue
    return max(per.values()) if per else None


def measure(
    exe: Path,
    *,
    cus: int,
    clock_mhz: float = 2400.0,
    timeout: int = 900,
) -> list[dict]:
    """Run every (mode, class) combination and return one record each."""
    if shutil.which("rocprofv3") is None:
        raise SystemExit("rocprofv3 not found on PATH")
    simds = cus * 2
    runs = [
        ("1 wave/SIMD", "mma", 20000, simds, 32),
        ("1 wave/SIMD", "valu", 200000, simds, 32),
        ("1 wave/SIMD", "exp", 200000, simds, 32),
        ("saturated", "mma", 4000, cus * 8, 256),
        ("saturated", "valu", 40000, cus * 8, 256),
        ("saturated", "exp", 40000, cus * 8, 256),
    ]

    out = []
    with tempfile.TemporaryDirectory(prefix="peaks_") as tmp:
        for mode, which, iters, blocks, thr in runs:
            outdir = Path(tmp) / f"{which}_{blocks}"
            proc = subprocess.run(
                [
                    "rocprofv3",
                    "--pmc",
                    "GRBM_GUI_ACTIVE",
                    "-d",
                    str(outdir),
                    "-f",
                    "csv",
                    "--",
                    str(exe),
                    which,
                    str(iters),
                    str(blocks),
                    str(thr),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            cyc = _dispatch_cycles(outdir)
            if cyc is None:
                out.append(
                    {
                        "mode": mode,
                        "op": which,
                        "ok": False,
                        "error": proc.stderr.strip()[-200:],
                    }
                )
                continue
            waves_per_simd = blocks * (thr // 32) / simds
            ops_per_simd = waves_per_simd * iters * NACC
            cyc_per_op = cyc / ops_per_simd
            ops_per_clk_cu = 2.0 / cyc_per_op  # 2 SIMDs per CU
            out.append(
                {
                    "mode": mode,
                    "op": which,
                    "ok": True,
                    "waves_per_simd": waves_per_simd,
                    "cyc_per_op_per_simd": cyc_per_op,
                    "ops_per_clk_per_cu": ops_per_clk_cu,
                    "tflops_at_ref_clock": ops_per_clk_cu
                    * FLOP_PER_OP[which]
                    * clock_mhz
                    * 1e6
                    * cus
                    / 1e12,
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cus", type=int, required=True, help="compute units on the part")
    p.add_argument("--arch", default="gfx1151", help="offload arch for --build")
    p.add_argument("--build", action="store_true", help="compile the .hip first")
    p.add_argument("--exe", type=Path, default=None, help="prebuilt binary to run")
    p.add_argument("--clock-mhz", type=float, default=2400.0)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    exe = args.exe
    if args.build or exe is None:
        with tempfile.TemporaryDirectory(prefix="peaks_build_") as tmp:
            exe = build(Path(tmp) / "probe_roofline_peaks", args.arch)
            results = measure(
                exe, cus=args.cus, clock_mhz=args.clock_mhz, timeout=args.timeout
            )
    else:
        results = measure(
            exe, cus=args.cus, clock_mhz=args.clock_mhz, timeout=args.timeout
        )

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(
        f"{args.arch}, {args.cus} CUs / {args.cus * 2} SIMDs, "
        f"reference clock {args.clock_mhz:.0f} MHz\n"
    )
    print(
        f"{'mode':<13}{'op':<6}{'waves/SIMD':>11}{'cyc/op/SIMD':>13}"
        f"{'ops/clk/CU':>12}{'TFLOP/s@ref':>13}"
    )
    print("-" * 68)
    for r in results:
        if not r["ok"]:
            print(f"{r['mode']:<13}{r['op']:<6}   FAIL  {r['error']}")
            continue
        print(
            f"{r['mode']:<13}{r['op']:<6}{r['waves_per_simd']:>11.1f}"
            f"{r['cyc_per_op_per_simd']:>13.2f}{r['ops_per_clk_per_cu']:>12.3f}"
            f"{r['tflops_at_ref_clock']:>13.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
