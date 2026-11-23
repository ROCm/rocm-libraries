#!/usr/bin/env python
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pre-warm ``norm_kernel_cache.py``'s on-disk cache across many CPU workers, no GPU.

Companion to ``run_norm_shards.py``: that runner's 8-way GPU-pinned shards
each compile their own cases' candidates inline, which serializes CPU-bound
compile time behind however many GPUs the box has. For most batchnorm2d/
layernorm2d shapes (a few dozen candidates) that is fine; for batchnorm2d's
``n_groups > 1`` masked tail on large-C shapes (candidate counts into the
hundreds, see ``norm_kernel_cache.py``'s module docstring) it starves compile
of every core beyond the GPU count. This script runs ``--compile-only
--cache-dir DIR`` for every in-scope case in a MIOpenDriver file, across as
many parallel subprocesses as the box has cores -- compilation needs no GPU,
so it is not bound by device count. Run it once per case file before
``run_norm_shards.py -- --cache-dir DIR ...`` so the GPU-pinned pass hits a
warm cache (a near-instant unpickle, see ``compile_candidate_cached``)
instead of compiling inline.

Usage (from ``library/``)::

    python benchmarks/common/norm_compile_all.py \\
        --miopen-file ~/bnorm_infer.txt --family bnorm \\
        --cache-dir /scratch/norm_cache --arch gfx950 --workers 64
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmarks.common.parse_norm_miopen import parse_file

_HERE = Path(__file__).resolve().parent
_SCRIPTS = {
    "bnorm": _HERE / "benchmark_batchnorm2d_aot.py",
    "lnorm": _HERE / "benchmark_layernorm2d_aot.py",
}


def case_argv(case, *, arch: str) -> list:
    dims = dict(case.dims)
    if case.family == "bnorm":
        return [
            "--n", str(dims["n"]), "--c", str(dims["c"]),
            "--h", str(dims["h"]), "--w", str(dims["w"]),
            "--training" if case.mode == "training" else "--inference",
            "--dtype", case.dtype, "--arch", arch,
        ]
    return [
        "--rows", str(dims["rows"]), "--cols", str(dims["cols"]),
        "--kind", "layernorm", "--dtype", case.dtype, "--arch", arch,
    ]


def compile_one(case, *, script: Path, arch: str, cache_dir: str, timeout: int) -> tuple:
    argv = [
        sys.executable, str(script), *case_argv(case, arch=arch),
        "--compile-only", "--cache-dir", cache_dir,
    ]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return (case, 1, "timeout")
    tail = "\n".join(proc.stderr.splitlines()[-10:]) if proc.returncode else ""
    return (case, proc.returncode, tail)


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--miopen-file", required=True, dest="miopen_file", metavar="FILE")
    ap.add_argument("--family", choices=("bnorm", "lnorm"), required=True)
    ap.add_argument("--cache-dir", required=True, dest="cache_dir", metavar="DIR")
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--workers", type=int, default=64,
                     help="parallel compile-only subprocesses (CPU-bound, no GPU needed)")
    ap.add_argument("--timeout", type=int, default=1800, metavar="SECONDS",
                     help="per-case wall clock budget (default 30 min)")
    return ap


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    cases, skips = parse_file(args.miopen_file, args.family)
    total = sum(skips.values()) + len(cases)
    print(f"{args.miopen_file}: {len(cases)}/{total} line(s) in scope", flush=True)
    for reason, n in sorted(skips.items(), key=lambda kv: -kv[1]):
        print(f"  skipped {n}: {reason}", flush=True)
    if not cases:
        print("error: no in-scope cases", file=sys.stderr)
        return 2

    script = _SCRIPTS[args.family]
    started = time.monotonic()
    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(compile_one, case, script=script, arch=args.arch,
                        cache_dir=args.cache_dir, timeout=args.timeout)
            for case in cases
        ]
        for fut in as_completed(futures):
            case, rc, tail = fut.result()
            done += 1
            if rc != 0:
                failed.append((case, tail))
                print(f"[{done}/{len(cases)}] FAIL rc={rc} {case.family}/{case.dtype}/"
                      f"{case.mode}/{dict(case.dims)}\n{tail}", flush=True)
            elif done % 10 == 0 or done == len(cases):
                elapsed = time.monotonic() - started
                print(f"[{done}/{len(cases)}] compiled ({elapsed / 60:.1f} min elapsed)",
                      flush=True)

    print(f"\n{len(cases) - len(failed)}/{len(cases)} case(s) cached in "
          f"{(time.monotonic() - started) / 60:.1f} min", flush=True)
    if failed:
        print(f"{len(failed)} case(s) failed to compile -- see FAIL lines above",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
