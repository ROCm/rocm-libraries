#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage03d_sustained_finalists -- time the finalists of every power-capped
GEMM again in long self-soaked windows (lib/sustained.py).

Reads the block logs of the round's exclusive bench (`<stage03-dir>/logs`:
stage03b's merged logs with the near-winner re-reads, else stage03's) and
the bench yaml it benched (`<in-dir>/shapes.yaml`), picks the power-capped
GEMMs and their finalists by the config's `bench.sustained` block (kernels
the config's blocklist names are no finalists and set no winner,
lib/correctness.py), and benches them --runs times through stage03 (every
--devices GPU, one GPU per bench process, crash recovery, retry pass): one
line per GEMM (`sustained.sustained_line`: an explicit list, no skip
screen, the long windows), each list shuffled with the config's
`bench.order_seed` and the shape, then rotated by one place per run, and the
GEMMs in a shuffled order per run. A block holds about `gemms_per_block`
GEMMs, run back to back on one GPU. GEMMs of other execution contexts
(stage03c) and of the planner's seed bench are not re-measured.

The runs' rows are a label source of their own: stage04 reads
`runs/run_<r>/stage03/logs` with --sustained-log-dir and flags them
`is_sustained`, and lib/repeats.py labels a finalist by them (their median
over the runs) and scales the GEMM's other kernels to match. Each run's
stage03 records the kernel identity of --identity-stem's files for stage04;
the stage refuses when the logs it reads were benched with other contents
of them.

Outputs in <out-dir>/, removed and rewritten by every run:
  runs/run_<r>/         the run's bench yaml (shapes.yaml), stage03 outputs
                        (stage03/) and log (stage03.log)
  sustained.json        the settings and per GEMM: why it is power-capped,
                        its short-window winner, its finalists with their
                        short-window and sustained latencies
  sustained_summary.json
                        counts, wall time and quarantined attempts per run,
                        and the kernel identity record

Exit status: 0; 1 on setup errors, when no run timed any finalist or when
the logs were benched with other library files; 2 when a GPU's lease is
refused; 128 + N when stopped by signal N.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import AbstractSet, Any, Dict, Iterable, List, Mapping, Optional
from typing import Sequence, Tuple

import yaml

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib import correctness as cx  # noqa: E402
from lib import gpu_lease  # noqa: E402
from lib import kernel_identity as ki  # noqa: E402
from lib import procs  # noqa: E402
from lib import rebench as rb  # noqa: E402
from lib import sustained as sus  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_log import iter_problems  # noqa: E402
from lib.bench_yaml import (  # noqa: E402
    adaptive_line,
    kernel_order_seed,
    read_bench_lines,
    rotating_mb_for,
    shape_of_line,
)
from lib.fs import atomic_write_text, write_json  # noqa: E402
from lib.planner import stable_rng  # noqa: E402
from lib.shapes import Shape  # noqa: E402

STAGE03 = THIS_DIR / "stage03_load_balance_offline_tuning.py"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="stage03d -- time the finalists of power-capped GEMMs again "
        "in long self-soaked windows."
    )
    ap.add_argument(
        "--in-dir", type=Path, required=True, help="the bench yaml's dir (shapes.yaml)"
    )
    ap.add_argument(
        "--stage03-dir",
        type=Path,
        required=True,
        help="the stage whose logs/ hold the round's exclusive bench (stage03b "
        "or stage03)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config-yaml", type=Path, required=True)
    ap.add_argument("--bench-binary", type=Path, required=True)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--stall-timeout-s", type=float, default=600.0)
    ap.add_argument("--startup-grace-s", type=float, default=900.0)
    ap.add_argument(
        "--pool-only-library",
        action="append",
        default=[],
        metavar="STEM",
        help="the runs bench a pool-only library directory of these stems "
        "(with --library-dir), as the round's stage03",
    )
    ap.add_argument("--library-dir", type=Path, default=None)
    ap.add_argument("--identity-dir", type=Path, default=None)
    ap.add_argument("--identity-stem", action="append", default=[], metavar="STEM")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if bool(args.identity_stem) != (args.identity_dir is not None):
        ap.error("--identity-dir and --identity-stem go together")
    if args.pool_only_library and args.library_dir is None:
        ap.error("--pool-only-library needs --library-dir")
    return args


def library_record(
    args: argparse.Namespace,
) -> Tuple[Optional[ki.LibraryIdentity], Optional[Dict[str, Any]]]:
    """(the kernel identity of the files the logs were benched with, as
    stage03 recorded them, else of --identity-stem's files; the kernel
    identity record of --identity-stem's files, no kernels listed); (None,
    None) without them. ValueError when the logs were benched with other
    contents of them, or one cannot be read."""
    if args.identity_dir is None:
        return None, None
    try:
        identity = ki.library_identity(args.identity_dir, args.identity_stem)
        then = ki.bench_record(args.stage03_dir / "logs")
        now = (
            ki.library_identity(args.identity_dir, list(then["files"]))
            if then is not None
            else None
        )
    except OSError as e:
        raise ValueError(str(e)) from None
    if then is not None and now is not None:
        changed = ki.changed_files(then, now)
        if changed:
            raise ValueError(
                f"{args.stage03_dir} benched other library files than "
                f"{args.identity_dir} holds now ({', '.join(changed)}; library "
                f"digest {then['digest']} -> {now.digest}): its logs name other "
                f"kernels than a sustained run would time"
            )
    return now or identity, ki.record(identity, ())


def select(
    logs: Iterable[Sequence[str]],
    st: Mapping[str, Any],
    blocked: AbstractSet[int] = frozenset(),
) -> Dict[Shape, sus.Finalists]:
    """{shape: finalists} of the power-capped GEMMs of the logs (their
    lines), the `blocked` solutions left out; a shape benched twice keeps
    its first complete problem."""
    out: Dict[Shape, sus.Finalists] = {}
    for lines in logs:
        for prob in iter_problems(lines):
            f = sus.finalists(prob, st, blocked)
            if f is not None:
                out.setdefault(f.shape, f)
    return out


def _run_stage03(
    run_dir: Path,
    args: argparse.Namespace,
    blocks: int,
    children: procs.ChildGroups,
) -> int:
    cmd = [
        sys.executable,
        str(STAGE03),
        "--in-dir",
        str(run_dir),
        "--out-dir",
        str(run_dir / "stage03"),
        "--bench-binary",
        str(args.bench_binary),
        "--devices",
        str(args.devices),
        "--blocks-per-gpu",
        str(blocks),
        "--stall-timeout-s",
        str(args.stall_timeout_s),
        "--startup-grace-s",
        str(args.startup_grace_s),
        "--no-warmup-pass",
        "--config-yaml",
        str(args.config_yaml),
    ]
    if args.pool_only_library:
        cmd += ["--library-dir", str(args.library_dir)]
        for stem in args.pool_only_library:
            cmd += ["--pool-only-library", stem]
    if args.identity_dir is not None:
        cmd += ["--identity-dir", str(args.identity_dir)]
        for stem in args.identity_stem:
            cmd += ["--identity-stem", stem]
    with (run_dir / "stage03.log").open("w") as log:
        proc = children.popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=procs.die_with_parent(),
        )
        return proc.wait()


def _summary_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    children = procs.ChildGroups()
    devices = [int(d) for d in str(args.devices).split(",") if d.strip()]
    with procs.stop_children_on_exit(children):
        try:
            with gpu_lease.hold_usable(
                devices,
                "stage03d sustained finalists",
                args.out_dir,
                report=lambda m: ui.warn("lease", m),
            ) as leases:
                return _run(args, leases.devices, children)
        except gpu_lease.LeaseError as e:
            ui.err("lease", str(e))
            return 2


def _run(
    args: argparse.Namespace, devices: List[int], children: procs.ChildGroups
) -> int:
    t0 = time.time()
    cfg = yaml.safe_load(args.config_yaml.read_text()) or {}
    st = sus.settings(cfg)
    errors = sus.config_errors(cfg)
    if errors:
        for e in errors:
            ui.err("cfg", e)
        return 1
    in_yaml = args.in_dir / "shapes.yaml"
    if not in_yaml.is_file():
        ui.err("err", f"no shapes.yaml in {args.in_dir}")
        return 1
    log_paths = sorted((args.stage03_dir / "logs").glob("block_*.log"))
    if not log_paths:
        ui.err("err", f"no block_*.log in {args.stage03_dir / 'logs'}")
        return 1
    if not args.bench_binary.exists():
        ui.err("err", f"hipblaslt-bench not found: {args.bench_binary}")
        return 1
    try:
        library, identity = library_record(args)
    except ValueError as e:
        ui.err("identity", str(e))
        return 1
    logs = [rb.read_log_lines(p.read_text(errors="replace")) for p in log_paths]
    try:
        blocked = rb.blocked_solutions(
            logs, cx.config_blocklist(cfg), library.ids if library is not None else None
        )
    except ValueError as e:
        ui.err("blocklist", str(e))
        return 1

    chosen = select(logs, st, blocked)
    lines_by_shape: Dict[Shape, str] = {}
    for line in read_bench_lines(in_yaml):
        shape = shape_of_line(line)
        if shape is not None:
            lines_by_shape.setdefault(shape, line)
    jobs = [s for s in lines_by_shape if s in chosen]
    n_unmatched = len(chosen) - len(jobs)
    order_seed = kernel_order_seed(cfg)
    blocks = rb.blocks_per_gpu(len(jobs), len(devices), int(st["gemms_per_block"]))
    if not args.quiet:
        ui.banner("stage03d sustained finalists")
        ui.info(
            "cfg", f"logs         : {len(log_paths)} in {args.stage03_dir / 'logs'}"
        )
        ui.info("cfg", f"settings     : {st}")
        ui.info(
            "select",
            f"{len(jobs)} power-capped GEMMs, "
            f"{sum(len(chosen[s].candidates) for s in jobs)} finalists; "
            f"{blocks} block(s) per GPU and run",
        )
    if n_unmatched:
        ui.warn("select", f"{n_unmatched} GEMM(s) have no line in {in_yaml}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(args.out_dir / "runs", ignore_errors=True)
    for name in ("sustained.json", "sustained_summary.json"):
        (args.out_dir / name).unlink(missing_ok=True)
    samples = rb.Samples()
    run_records: List[Dict[str, Any]] = []
    for r in range(int(st["runs"]) if jobs else 0):
        run_dir = args.out_dir / "runs" / f"run_{r}"
        run_dir.mkdir(parents=True)
        order = list(jobs)
        if order_seed is not None:
            stable_rng(order_seed, "sustained", r).shuffle(order)
        lines = []
        for shape in order:
            rot = rotating_mb_for(Shape(*shape), cfg) if cfg.get("hipblaslt") else None
            kernels = rb.run_order(
                chosen[shape].sol_indices, r, order_seed, ("sustained", *shape)
            )
            line = adaptive_line(lines_by_shape[shape], cfg)
            line = sus.sustained_line(line, kernels, st, rot)
            lines.append(line.rstrip("\n") + "\n")
        atomic_write_text(run_dir / "shapes.yaml", "".join(lines))
        t_run = time.time()
        rc = _run_stage03(run_dir, args, blocks, children)
        n_rows = 0
        for p in sorted((run_dir / "stage03" / "logs").glob("block_*.log")):
            n_rows += samples.add_log(
                rb.read_log_lines(p.read_text(errors="replace")), r
            )
        s03 = _summary_json(run_dir / "stage03" / "stage03_summary.json")
        rec = {
            "run": r,
            "rc": rc,
            "wall_s": time.time() - t_run,
            "rows": n_rows,
            "quarantined": len(s03.get("quarantined") or []),
        }
        run_records.append(rec)
        (ui.ok if rc == 0 else ui.warn)(
            "run",
            f"run {r}: rc={rc}  {n_rows} timed rows  ({ui.fmt_dur(rec['wall_s'])})",
        )
    if jobs and not any(rec["rows"] for rec in run_records):
        ui.err("run", "no sustained run timed any finalist; see runs/*/stage03.log")
        return 1

    per_gemm = []
    for shape in jobs:
        f = chosen[shape]
        timed = samples.by_shape.get(shape, {})
        per_gemm.append(
            {
                "m": shape[0],
                "n": shape[1],
                "k": shape[2],
                "batch": shape[3],
                "reason": f.reason,
                "winner_us": f.winner_us,
                "tflops": sus.tflops(shape, f.winner_us),
                "finalists": [
                    {
                        "sol_index": c.sol_index,
                        "skipped": c.skipped,
                        "us": c.us,
                        "runs_us": [s.us for s in timed.get(c.sol_index, [])],
                        "median_us": (
                            statistics.median(s.us for s in timed[c.sol_index])
                            if timed.get(c.sol_index)
                            else None
                        ),
                    }
                    for c in f.candidates
                ],
            }
        )
    write_json(
        args.out_dir / "sustained.json",
        {
            "settings": st,
            "library_digest": identity["digest"] if identity else None,
            "gemms": per_gemm,
        },
    )
    reasons = [chosen[s].reason for s in jobs]
    summary = {
        "in_yaml": str(in_yaml),
        "settings": st,
        "n_gemms": len(jobs),
        "n_gemms_by_reason": {r: reasons.count(r) for r in sorted(set(reasons))},
        "n_finalists": sum(len(chosen[s].candidates) for s in jobs),
        "n_unmatched_gemms": n_unmatched,
        "blocks_per_gpu": blocks,
        "runs": run_records,
        "wall_s": time.time() - t0,
        **({ki.RECORD_KEY: identity} if identity else {}),
    }
    write_json(args.out_dir / "sustained_summary.json", summary)
    if not args.quiet:
        ui.banner("Done", ui.C.GREEN)
        ui.ok(
            "done",
            f"{len(jobs)} GEMMs, {summary['n_finalists']} finalists, "
            f"{len(run_records)} run(s) in {ui.fmt_dur(summary['wall_s'])}",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
