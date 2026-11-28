#!/usr/bin/env python
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run a bnorm/lnorm MIOpenDriver case file across several GPUs, one process per card.

Sibling of ``run_conv_shards.py`` for the batchnorm2d/layernorm2d AOT
benchmarks. Not built on that runner's ``SCRIPTS``/``shard_command`` path: both
conv AOT scripts take ``--miopen-file`` and loop over the whole thing inside
one process, appending rows as they go. Neither
``benchmark_batchnorm2d_aot.py`` nor ``benchmark_layernorm2d_aot.py`` does
that -- each invocation times exactly one shape
(``--n/--c/--h/--w`` or ``--rows/--cols``) and
``norm_bench_common.JsonlResultsWriter`` opens ``--results`` in ``"w"`` mode,
so one process can't accumulate rows across shapes. This runner instead:
parses the MIOpenDriver file itself (:mod:`benchmarks.common.parse_norm_miopen`,
which also drops what the current kernel families don't cover -- see its
module docstring), deals the resulting cases round-robin into per-shard JSON
Lines files, and has each shard's worker
(:mod:`benchmarks.common.norm_shard_worker`) run its cases one subprocess at a
time. The device-pinned scheduler loop, device resolution, and progress
reporting are reused unchanged from ``run_conv_shards.py`` -- they are generic
over "a shard is a child process that eventually writes case rows to a path",
which holds here too.

Usage (from ``library/``)::

    python benchmarks/common/run_norm_shards.py \\
        --miopen-file ~/bnorm_infer.txt --family bnorm --devices 8 \\
        --arch gfx950 --log-dir /scratch/bnorm_infer -- --fast-rank

Everything after ``--`` is forwarded verbatim to the per-case AOT script
invocation (e.g. ``--fast-rank``, ``--warmup``, ``--iters``, ``--timing``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from benchmarks.common import case_results
from benchmarks.common.parse_norm_miopen import ParsedCase, parse_file
from benchmarks.common.run_conv_shards import (
    POLL_SECONDS,
    PROGRESS_SECONDS,
    SHARDS_PER_DEVICE,
    _duration,
    measured_keys,
    progress_line,
    resolve_devices,
    run_shards,
    shard_paths,
)

_HERE = Path(__file__).resolve().parent
_WORKER = _HERE / "norm_shard_worker.py"


def _case_to_dict(case: ParsedCase) -> dict:
    return {
        "family": case.family,
        "dims": list(case.dims),
        "dtype": case.dtype,
        "mode": case.mode,
    }


def deal_cases(cases: Sequence[ParsedCase], shards: int) -> List[List[ParsedCase]]:
    """Round-robin ``cases`` into ``shards`` lists, dropping empty ones.

    Same dealing rule as ``run_conv_shards.split_cases`` (line *i* to shard
    *i mod shards*), reimplemented here because it deals ``ParsedCase``
    objects rather than raw MIOpenDriver text lines.
    """
    if shards < 1:
        raise ValueError("--shards needs at least one shard")
    dealt: List[List[ParsedCase]] = [[] for _ in range(shards)]
    for i, case in enumerate(cases):
        dealt[i % shards].append(case)
    return [shard for shard in dealt if shard]


def worker_command(
    family: str, arch: str, forwarded: Sequence[str], paths: Dict[str, Path]
) -> List[str]:
    return [
        sys.executable, "-u", str(_WORKER),
        "--family", family,
        "--cases", str(paths["cases"]),
        "--results", str(paths["rows"]),
        "--arch", arch,
        "--", *forwarded,
    ]


def _tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Everything after -- is forwarded to the per-case AOT invocation.",
    )
    ap.add_argument("--miopen-file", required=True, dest="miopen_file", metavar="FILE")
    ap.add_argument("--family", choices=("bnorm", "lnorm"), required=True)
    ap.add_argument("--devices", default="auto",
                    help="device count, comma-separated ids, or auto (default: auto)")
    ap.add_argument("--shards", type=int, default=None, metavar="N")
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--log-dir", default=None, dest="log_dir", metavar="DIR")
    ap.add_argument("--results", default=None, metavar="FILE",
                    help="merged per-case table (default: results.jsonl in --log-dir)")
    ap.add_argument("forwarded", nargs=argparse.REMAINDER)
    return ap


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    forwarded = list(args.forwarded)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    cases, skips = parse_file(args.miopen_file, args.family)
    total_lines = sum(skips.values()) + len(cases)
    print(f"{args.miopen_file}: {len(cases)}/{total_lines} line(s) in scope", flush=True)
    for reason, n in sorted(skips.items(), key=lambda kv: -kv[1]):
        print(f"  skipped {n}: {reason}", flush=True)
    if not cases:
        print("error: no in-scope cases", file=sys.stderr)
        return 2

    devices = resolve_devices(args.devices)
    shards = args.shards if args.shards is not None else SHARDS_PER_DEVICE * len(devices)
    dealt = deal_cases(cases, shards)

    import tempfile

    log_dir = Path(args.log_dir or tempfile.mkdtemp(prefix="rocke-norm-shards-"))
    log_dir.mkdir(parents=True, exist_ok=True)

    width = len(str(len(dealt) - 1)) or 1
    plan: List[Tuple[int, Dict[str, Path]]] = []
    for index, shard in enumerate(dealt):
        paths = shard_paths(log_dir, index, width)
        paths["cases"].write_text(
            "".join(json.dumps(_case_to_dict(c)) + "\n" for c in shard),
            encoding="utf-8",
        )
        plan.append((index, paths))

    print(
        f"{len(cases)} case(s) over {len(dealt)} shard(s) on {len(devices)} device(s)",
        flush=True,
    )
    print(f"logs: {log_dir}", flush=True)

    def _report(index: int, device: str, rc: int, seconds: float) -> None:
        paths = shard_paths(log_dir, index, width)
        rows = len(case_results.read_rows(paths["rows"]))
        print(
            f"shard {index} device={device} rc={rc} {rows} row(s) {seconds / 60:.1f} min",
            flush=True,
        )

    started = time.monotonic()
    row_files = [paths["rows"] for _index, paths in plan]

    def _progress(shards_done: int, busy: int) -> None:
        print(
            progress_line(
                measured=len(measured_keys(row_files)),
                total=len(cases),
                baseline=0,
                shards_done=shards_done,
                shards=len(dealt),
                busy=busy,
                devices=len(devices),
                elapsed=time.monotonic() - started,
            ),
            flush=True,
        )

    done = run_shards(
        plan,
        devices=devices,
        command=lambda index, paths: worker_command(args.family, args.arch, forwarded, paths),
        poll_seconds=POLL_SECONDS,
        on_done=_report,
        on_progress=_progress,
        progress_seconds=PROGRESS_SECONDS,
    )

    rc_all = 0
    for index, _device, rc, _seconds in done:
        if rc:
            rc_all = rc_all or rc
            print(f"\nshard {index} failed (rc={rc}); last lines of its log:")
            print(_tail(shard_paths(log_dir, index, width)["log"]))

    shard_rows = [
        row for _index, paths in plan for row in case_results.read_rows(paths["rows"])
    ]
    best_by_key: Dict[str, dict] = {}
    for row in shard_rows:
        key = row.get("key")
        ms = row.get("ms")
        if key is None or ms is None:
            continue
        current = best_by_key.get(key)
        if current is None or ms < current["ms"]:
            best_by_key[key] = row

    out = Path(args.results or (log_dir / "results.jsonl"))
    if shard_rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in shard_rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        csv_path = out.with_suffix(out.suffix + ".best.csv") if out.suffix else out.with_suffix(".best.csv")
        import csv as csv_mod

        fieldnames = sorted({k for row in best_by_key.values() for k in row})
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv_mod.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for key in sorted(best_by_key):
                writer.writerow(best_by_key[key])
        print(f"\n{len(best_by_key)}/{len(cases)} case(s) measured", flush=True)
        print(f"results: {out} (all candidates) and {csv_path} (best per case)", flush=True)
    else:
        print("\n[info] no per-case rows were written by any shard")
        rc_all = rc_all or 1
    print(f"wall: {(time.monotonic() - started) / 60:.1f} min", flush=True)
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
