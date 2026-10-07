#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage03b_rebench_near_ties -- time each GEMM's near-tie candidates again
and write stage03's logs with their median latencies (lib/rebench.py).

Reads stage03's block logs (`<stage03-dir>/logs/block_*.log`) and the bench
yaml stage03 benched (`<in-dir>/shapes_probed.yaml`, else `shapes.yaml`),
finds the near-tie set of every GEMM (--tolerance, --max-kernels) and
benches the sets --runs times through stage03 (every --devices GPU, crash
recovery, retry pass): one bench line per GEMM (`rebench.rebench_line`,
iteration counts times --iters-factor), each list rotated by one place per
run. The bench must know `solution_index_list`. A run that fails keeps the
timings it logged; the stage fails only when no run timed anything.

Options left out on the command line come from the config's
`bench.rebench` block (tolerance, runs, iters_factor, max_kernels).

Outputs in <out-dir>/, removed and rewritten by every run:
  logs/block_*.log      stage03's logs with the median rows merged in;
                        stage04 reads these instead of stage03's
  runs/run_<r>/         the run's bench yaml (shapes.yaml), stage03 outputs
                        (stage03/) and log (stage03.log)
  rebench.json          per GEMM: candidates with their original and
                        re-bench latencies, winner before and after
  rebench_summary.json  counts, winner changes, wall time per run

Exit status: 0; 1 on setup errors or when no run timed any candidate;
128 + N when stopped by signal N.
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib import procs  # noqa: E402
from lib import rebench as rb  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_yaml import (
    read_bench_lines,
    rotating_mb_for,
    shape_of_line,
)  # noqa: E402
from lib.fs import atomic_write_text, write_json  # noqa: E402
from lib.shapes import Shape  # noqa: E402

STAGE03 = THIS_DIR / "stage03_load_balance_offline_tuning.py"
CONFIG_KEYS = ("tolerance", "runs", "iters_factor", "max_kernels")
DEFAULTS = {
    "tolerance": rb.DEFAULT_TOLERANCE,
    "runs": rb.DEFAULT_RUNS,
    "iters_factor": rb.DEFAULT_ITERS_FACTOR,
    "max_kernels": rb.DEFAULT_MAX_KERNELS,
}


def bench_lists_supported(bench_binary: Path) -> bool:
    """The bench's argument schema (hipblaslt_common.yaml next to it, or
    next to the binary a wrapper script execs) knows
    `solution_index_list`. A bench without the file is assumed to."""
    for d in (bench_binary.parent, bench_binary.resolve().parent):
        schema = d / "hipblaslt_common.yaml"
        if schema.is_file():
            return "solution_index_list" in schema.read_text(errors="replace")
    return True


def resolve_options(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    """Fill the options left out on the command line from the config's
    `bench.rebench` block, else DEFAULTS."""
    block = ((cfg.get("bench") or {}).get("rebench") or {}) if cfg else {}
    for key in CONFIG_KEYS:
        if getattr(args, key) is None:
            setattr(args, key, block.get(key, DEFAULTS[key]))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="stage03b -- re-bench each GEMM's near-tie candidates and "
        "merge their median latencies into stage03's logs."
    )
    ap.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="stage03's input dir (shapes_probed.yaml, else shapes.yaml)",
    )
    ap.add_argument("--stage03-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bench-binary", type=Path, required=True)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--config-yaml", type=Path, default=None)
    ap.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="relative latency gap to the winner within which a candidate is "
        f"re-benched (default {rb.DEFAULT_TOLERANCE})",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=None,
        help=f"re-bench runs; an odd count has a true median (default {rb.DEFAULT_RUNS})",
    )
    ap.add_argument(
        "--iters-factor",
        type=float,
        default=None,
        help="factor on the bench line's iteration counts (default "
        f"{rb.DEFAULT_ITERS_FACTOR})",
    )
    ap.add_argument(
        "--max-kernels",
        type=int,
        default=None,
        help="largest near-tie set per GEMM, the fastest candidates (default "
        f"{rb.DEFAULT_MAX_KERNELS})",
    )
    ap.add_argument(
        "--blocks-per-gpu",
        type=int,
        default=2,
        help="stage03 work blocks per GPU and run",
    )
    ap.add_argument("--stall-timeout-s", type=float, default=300.0)
    ap.add_argument("--startup-grace-s", type=float, default=900.0)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def _clean(out_dir: Path) -> None:
    for name in ("logs", "runs"):
        shutil.rmtree(out_dir / name, ignore_errors=True)
    for name in ("rebench.json", "rebench_summary.json"):
        (out_dir / name).unlink(missing_ok=True)


def _bench_yaml(in_dir: Path) -> Optional[Path]:
    for name in ("shapes_probed.yaml", "shapes.yaml"):
        if (in_dir / name).is_file():
            return in_dir / name
    return None


def _run_stage03(
    run_dir: Path, args: argparse.Namespace, children: procs.ChildGroups
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
        str(args.blocks_per_gpu),
        "--stall-timeout-s",
        str(args.stall_timeout_s),
        "--startup-grace-s",
        str(args.startup_grace_s),
        "--no-warmup-pass",
    ]
    with (run_dir / "stage03.log").open("w") as log:
        proc = children.popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        return proc.wait()


def _summary_stats(
    changes: List[rb.ProblemChange],
    tie_sets: Mapping[rb.Shape, rb.TieSet],
    n_complete: int,
) -> Dict[str, Any]:
    shifts = [
        abs(math.log(new / old))
        for ch in changes
        for old, new in ch.replaced.values()
        if old > 0 and new > 0 and math.isfinite(old)
    ]
    winner_shift = [
        math.log(ch.replaced[ch.winner_before][1] / ch.replaced[ch.winner_before][0])
        for ch in changes
        if ch.winner_before in ch.replaced
    ]
    n_changed = sum(1 for ch in changes if ch.winner_changed)
    return {
        "n_problems": n_complete,
        "n_gemms_with_near_ties": len(tie_sets),
        "n_candidates": sum(len(t.candidates) for t in tie_sets.values()),
        "n_skip_candidates": sum(
            1 for t in tie_sets.values() for c in t.candidates if c.skipped
        ),
        "n_gemms_merged": len({ch.shape for ch in changes}),
        "n_rows_replaced": sum(len(ch.replaced) for ch in changes),
        "n_winner_changed": n_changed,
        "winner_changed_frac_of_merged": (
            n_changed / len(changes) if changes else None
        ),
        "winner_changed_frac_of_problems": (
            n_changed / n_complete if n_complete else None
        ),
        "median_abs_log_latency_change": (
            sorted(shifts)[len(shifts) // 2] if shifts else None
        ),
        "mean_log_winner_latency_change": (
            sum(winner_shift) / len(winner_shift) if winner_shift else None
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    children = procs.ChildGroups()
    with procs.stop_children_on_exit(children):
        return _run(args, children)


def _run(args: argparse.Namespace, children: procs.ChildGroups) -> int:
    t0 = time.time()
    cfg: Dict[str, Any] = {}
    if args.config_yaml is not None:
        with args.config_yaml.open() as f:
            cfg = yaml.safe_load(f) or {}
    resolve_options(args, cfg)
    if not 0 < float(args.tolerance) < 1 or int(args.runs) < 1:
        ui.err("args", "--tolerance must be in (0, 1) and --runs at least 1")
        return 1
    in_yaml = _bench_yaml(args.in_dir)
    if in_yaml is None:
        ui.err("err", f"no shapes_probed.yaml or shapes.yaml in {args.in_dir}")
        return 1
    log_paths = sorted((args.stage03_dir / "logs").glob("block_*.log"))
    if not log_paths:
        ui.err("err", f"no block_*.log in {args.stage03_dir / 'logs'}")
        return 1
    if not args.bench_binary.exists():
        ui.err("err", f"hipblaslt-bench not found: {args.bench_binary}")
        return 1
    if not bench_lists_supported(args.bench_binary):
        ui.err(
            "err",
            f"{args.bench_binary} does not know solution_index_list; the re-bench "
            f"needs a hipblaslt-bench that does",
        )
        return 1

    logs = {p.name: rb.read_log_lines(p.read_text(errors="replace")) for p in log_paths}
    tie_sets, n_complete = rb.tie_sets_of_logs(
        logs.values(), float(args.tolerance), int(args.max_kernels)
    )
    lines_by_shape: Dict[rb.Shape, str] = {}
    for line in read_bench_lines(in_yaml):
        shape = shape_of_line(line)
        if shape is not None:
            lines_by_shape.setdefault(shape, line)
    jobs = [s for s in lines_by_shape if s in tie_sets]
    n_unmatched = len(tie_sets) - len(jobs)
    if not args.quiet:
        ui.banner("stage03b near-tie re-bench")
        ui.info(
            "cfg", f"stage03 logs : {len(log_paths)} in {args.stage03_dir / 'logs'}"
        )
        ui.info("cfg", f"bench yaml   : {in_yaml}")
        ui.info(
            "cfg",
            f"tolerance={args.tolerance}  runs={args.runs}  "
            f"iters_factor={args.iters_factor}  max_kernels={args.max_kernels}",
        )
        ui.info(
            "ties",
            f"{len(tie_sets)} of {n_complete} complete problems have near ties "
            f"({sum(len(tie_sets[s].candidates) for s in jobs)} candidates)",
        )
    if n_unmatched:
        ui.warn("ties", f"{n_unmatched} near-tie GEMM(s) have no line in {in_yaml}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _clean(args.out_dir)
    samples = rb.Samples()
    run_records: List[Dict[str, Any]] = []
    for r in range(int(args.runs) if jobs else 0):
        run_dir = args.out_dir / "runs" / f"run_{r}"
        run_dir.mkdir(parents=True)
        lines = []
        for shape in jobs:
            rot = None
            if cfg.get("hipblaslt"):
                rot = rotating_mb_for(Shape(*shape), cfg)
            lines.append(
                rb.rebench_line(
                    lines_by_shape[shape],
                    rb.rotate(tie_sets[shape].sol_indices, r),
                    iters_factor=float(args.iters_factor),
                    rotating_mib=rot,
                ).rstrip("\n")
            )
        atomic_write_text(run_dir / "shapes.yaml", "".join(ln + "\n" for ln in lines))
        t_run = time.time()
        rc = _run_stage03(run_dir, args, children)
        n_rows = 0
        for p in sorted((run_dir / "stage03" / "logs").glob("block_*.log")):
            n_rows += samples.add_log(
                rb.read_log_lines(p.read_text(errors="replace")), r
            )
        rec = {"run": r, "rc": rc, "wall_s": time.time() - t_run, "rows": n_rows}
        run_records.append(rec)
        (ui.ok if rc == 0 else ui.warn)(
            "run",
            f"run {r}: rc={rc}  {n_rows} timed rows  ({ui.fmt_dur(rec['wall_s'])})",
        )
    if jobs and not any(rec["rows"] for rec in run_records):
        ui.err("run", "no re-bench run timed any candidate; see runs/*/stage03.log")
        return 1

    (args.out_dir / "logs").mkdir(parents=True, exist_ok=True)
    changes: List[rb.ProblemChange] = []
    for name, lines in logs.items():
        merged, ch = rb.merge_log(lines, tie_sets, samples)
        atomic_write_text(args.out_dir / "logs" / name, "".join(merged))
        changes.extend(ch)

    by_shape = {ch.shape: ch for ch in changes}
    per_gemm = []
    for shape in jobs:
        ties = tie_sets[shape]
        ch = by_shape.get(shape)
        per_gemm.append(
            {
                "m": shape[0],
                "n": shape[1],
                "k": shape[2],
                "batch": shape[3],
                "winner_before": ties.winner_sol,
                "winner_after": ch.winner_after if ch else ties.winner_sol,
                "candidates": [
                    {
                        "sol_index": c.sol_index,
                        "rank": c.rank,
                        "skipped": c.skipped,
                        "us": c.us,
                        "runs_us": [
                            s.us
                            for s in samples.by_shape.get(shape, {}).get(
                                c.sol_index, []
                            )
                        ],
                        "median_us": (
                            (ch.replaced.get(c.sol_index, (None, None))[1])
                            if ch
                            else None
                        ),
                    }
                    for c in ties.candidates
                ],
            }
        )
    write_json(args.out_dir / "rebench.json", {"gemms": per_gemm})
    summary = {
        "in_yaml": str(in_yaml),
        "options": {k: getattr(args, k) for k in CONFIG_KEYS},
        "n_unmatched_gemms": n_unmatched,
        **_summary_stats(changes, {s: tie_sets[s] for s in jobs}, n_complete),
        "runs": run_records,
        "wall_s": time.time() - t0,
    }
    write_json(args.out_dir / "rebench_summary.json", summary)
    if not args.quiet:
        ui.banner("Done", ui.C.GREEN)
        ui.ok(
            "done",
            f"merged {summary['n_rows_replaced']} rows of "
            f"{summary['n_gemms_merged']} GEMMs; winner changed in "
            f"{summary['n_winner_changed']}",
        )
        ui.ok(
            "done",
            f"wall {ui.fmt_dur(summary['wall_s'])}  logs: {args.out_dir / 'logs'}",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
