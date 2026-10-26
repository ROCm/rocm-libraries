#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regenerate the buildable-MI lists MI Designer reads (geko/config_generator/mi_lists/).

For each GEMM type (layout x data types), every candidate MatrixInstruction is checked with
TensileLite code generation (CPU only; no GPU is used) and the ones that build within the generic
(GA) search space's fork ranges are written to mi_lists/<arch>/<GEMM name>.yaml. MI Designer uses
the list for every search space except subtile. See geko/config_generator/buildability.py for
the search. Rerun whenever TensileLite code generation or geko's fork ranges change.

Usage:
    python3 scripts/generate_buildable_mis.py --arch gfx950 --types B:B:S H:H:S F8:B:S \\
        --layouts TN NN NT TT --workers 64
"""

import argparse
import datetime
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from geko.config_generator.mi_candidates import MI_LISTS_DIR, buildable_list_path
from geko.paths import resolve_hipblaslt_path
from geko.utils import ensure_tensile_importable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--arch", required=True, help="ARCH key, e.g. gfx950 or gfx942")
    p.add_argument(
        "--types",
        nargs="+",
        required=True,
        metavar="A:D:C",
        help="DataType:DestDataType:ComputeDataType, e.g. B:B:S or F8:B:S",
    )
    p.add_argument(
        "--layouts", nargs="+", default=["TN", "NN", "NT", "TT"], help="TRANSA+TRANSB, e.g. TN"
    )
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--hipblaslt", default=None, help="hipBLASLt checkout (default: auto-detect)")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=MI_LISTS_DIR,
        help="root of the lists (default: geko's own mi_lists/)",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="also write <arch>/<GEMM>.csv with every MI's verdict, witness setting and checks",
    )
    return p.parse_args()


def _source(hipblaslt: Path) -> str:
    commit = (
        subprocess.run(
            ["git", "-C", str(hipblaslt), "rev-parse", "--short=10", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        or "unknown"
    )
    return f"rocm-libraries {commit}, {datetime.date.today().isoformat()}"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s: %(message)s", stream=sys.stdout
    )
    hipblaslt = Path(resolve_hipblaslt_path(args.hipblaslt))
    ensure_tensile_importable(hipblaslt)
    from geko.config_generator import buildability as B  # imports TensileLite
    from geko.config_generator import mi_candidates

    mi_candidates.MI_LISTS_DIR = args.output_dir
    source = _source(hipblaslt)
    for types in args.types:
        dt, dd, cd = types.split(":")
        for layout in args.layouts:
            config = dict(
                TRANSA=layout[0],
                TRANSB=layout[1],
                DataType=dt,
                DestDataType=dd,
                ComputeDataType=cd,
                ARCH=args.arch,
            )
            t0 = time.time()
            print(
                f"{args.arch} {types} {layout}: writing probe configs, setting up TensileLite",
                flush=True,
            )

            def progress(done, total, buildable):
                if done % 100 == 0 or done == total:
                    print(
                        f"  {done}/{total} MIs, {buildable} buildable, {time.time() - t0:.0f}s",
                        flush=True,
                    )

            try:
                verdicts, space, name = B.classify_candidates(config, hipblaslt, args.workers, progress)
            except B.SubtileOnly as e:
                print(f"  skipped: {e}", flush=True)
                continue
            path = buildable_list_path(args.arch, name)
            B.write_buildable_list(path, verdicts, arch=args.arch, gemm_name=name, source=source)
            stages = {}
            for v in verdicts:
                stages[v.stage or "unbuildable"] = stages.get(v.stage or "unbuildable", 0) + 1
            print(
                f"  -> {path} ({sum(v.buildable for v in verdicts)} of {len(verdicts)} buildable; "
                f"decided by {stages}; {sum(v.checks for v in verdicts):,} code generations, "
                f"{time.time() - t0:.0f}s)",
                flush=True,
            )
            if args.report_dir:
                B.write_report(args.report_dir / args.arch / f"{name}.csv", verdicts)


if __name__ == "__main__":
    main()
