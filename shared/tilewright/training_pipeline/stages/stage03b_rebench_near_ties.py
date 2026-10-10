#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage03b_rebench_near_ties -- read each GEMM's near-winners again in
bench processes of their own and write stage03's logs with their medians
(lib/rebench.py).

Reads stage03's block logs (`<stage03-dir>/logs/block_*.log`) and the bench
yaml stage03 benched (`<in-dir>/shapes.yaml`), finds the near-tie set of
every GEMM (--tolerance, --max-kernels; the kernels the config's blocklist
names take no part, lib/correctness.py) and reads the sets again, as often
as the band of the GEMM's winner latency asks (`rebench.Bands`: `fast_runs`
below `fast_below_us`, `slow_runs` from `slow_from_us` on, in windows of at
least `slow_min_iters` iterations, `runs` in between). Run r benches the
GEMMs that ask more than r reads through stage03 (every --devices GPU,
crash recovery, retry pass, as few blocks as the lines per GPU need:
`rebench.blocks_per_gpu`): one bench line per GEMM (`rebench.reread_line`,
timed adaptively like the GEMM's own line), each list shuffled with the
config's `bench.order_seed` and the shape (unless `bench.kernel_order` is
`listed`), then rotated by one place per run (`rebench.run_order`), led by
a sacrificial copy of its first kernel (`sacrificial_kernel`), and the
GEMMs in a seeded shuffle per run (`gemm_order: shuffled`). A candidate's
label becomes the median of its original timing and its re-reads
(`with_original`, but in a set with a skip row; else the lower median of the
re-reads alone). The bench must know `solution_index_list`. A run that
fails keeps the timings it logged; the stage fails only when no run timed
anything. The stage leases the GPUs of --devices for all its runs
(lib/gpu_lease.py); a run's stage03 quarantines blocks whose GPU another
process used, and their rows take no part in the medians.

Options left out on the command line come from the config's
`bench.rebench` block (the keys of `rebench.REREAD_DEFAULTS`), else their
defaults; `rebench.PREDATING_REREADS` holds the settings of the stage
before the bands, and `rebench.PREDATING_SLOW_BAND` the slow band's before
it became three re-reads in the line's own windows.

--identity-dir / --identity-stem name the library files whose solution
indices the bench lines name, as for stage03: the stage refuses when the
files stage03 recorded hold other kernels under their solution indices now
(the logs it merges and the runs it adds would time other kernels), and
records their kernel identity (lib/kernel_identity.py) for stage04; its
runs name them to stage03, whose warm-up line takes a kernel of each. When
stage03 recorded none, nothing checked its logs against a library, and the
merged logs record none either.

Outputs in <out-dir>/, removed and rewritten by every run:
  logs/block_*.log      stage03's logs with the merged rows; stage04 reads
                        these instead of stage03's
  runs/run_<r>/         the run's bench yaml (shapes.yaml), stage03 outputs
                        (stage03/) and log (stage03.log)
  rebench.json          per GEMM: its band and runs, whether its labels
                        merge the original timings, candidates with their
                        original and re-read latencies and the label, winner
                        before and after, each named by signature and kernel
                        id next to its solution index when stage03 recorded
                        its library; the library digest of their solution
                        indices
  rebench_summary.json  the options, counts per band, winner changes, wall
                        time and quarantined attempts per run; the kernel
                        identity record, when stage03 recorded one; how
                        stage03's and the runs' benches launched the kernels
                        (`launch`, lib/launch_settings.py), which stage04
                        reads for the merged logs

Exit status: 0; 1 on setup errors, when no run timed any candidate or when
stage03 benched other library files; 2 when a GPU's lease is refused;
128 + N when stopped by signal N.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib import correctness as cx  # noqa: E402
from lib import gpu_lease  # noqa: E402
from lib import kernel_identity as ki  # noqa: E402
from lib import launch_settings  # noqa: E402
from lib import procs  # noqa: E402
from lib import rebench as rb  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_yaml import (  # noqa: E402
    DEFAULT_ORDER_SEED,
    kernel_order_seed,
    read_bench_lines,
    rotating_mb_for,
    shape_of_line,
)
from lib.fs import atomic_write_text, write_json  # noqa: E402
from lib.planner import stable_rng  # noqa: E402
from lib.shapes import Shape  # noqa: E402

STAGE03 = THIS_DIR / "stage03_load_balance_offline_tuning.py"
CONFIG_KEYS = tuple(rb.REREAD_DEFAULTS)


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
    `bench.rebench` block, else `rebench.REREAD_DEFAULTS`."""
    block = ((cfg.get("bench") or {}).get("rebench") or {}) if cfg else {}
    for key in CONFIG_KEYS:
        if getattr(args, key, None) is None:
            setattr(args, key, block.get(key, rb.REREAD_DEFAULTS[key]))


def option_errors(args: argparse.Namespace) -> List[str]:
    """What is wrong with the resolved options."""
    errors = []
    if not 0 < float(args.tolerance) < 1:
        errors.append("--tolerance must be in (0, 1)")
    if min(int(args.runs), int(args.fast_runs), int(args.slow_runs)) < 0:
        errors.append("the runs of a band must be at least 0")
    if not 0 <= float(args.fast_below_us) <= float(args.slow_from_us):
        errors.append("fast_below_us must be in [0, slow_from_us]")
    if int(args.slow_min_iters) < 0:
        errors.append("slow_min_iters must be at least 0")
    if args.gemm_order not in rb.GEMM_ORDERS:
        errors.append(f"gemm_order must be one of {list(rb.GEMM_ORDERS)}")
    return errors


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="stage03b -- read each GEMM's near-winners again in bench "
        "processes of their own and merge their medians into stage03's logs."
    )
    ap.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="stage03's input dir (shapes.yaml)",
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
        f"read again (default {rb.DEFAULT_TOLERANCE})",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=None,
        help="re-reads of a GEMM whose winner is in neither the fast nor the "
        f"slow band (default {rb.Bands.runs})",
    )
    ap.add_argument(
        "--fast-runs",
        type=int,
        default=None,
        help="re-reads of a GEMM whose winner is faster than fast_below_us "
        f"(default {rb.Bands.fast_runs})",
    )
    ap.add_argument(
        "--slow-runs",
        type=int,
        default=None,
        help="re-reads of a GEMM whose winner takes slow_from_us or more "
        f"(default {rb.Bands.slow_runs})",
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
        default=None,
        help="stage03 work blocks per GPU and run (default: one per "
        f"{rb.REBENCH_LINES_PER_BLOCK} lines of a GPU's share, at most "
        f"{rb.MAX_BLOCKS_PER_GPU})",
    )
    ap.add_argument("--stall-timeout-s", type=float, default=600.0)
    ap.add_argument("--startup-grace-s", type=float, default=900.0)
    ap.add_argument(
        "--identity-dir",
        type=Path,
        default=None,
        help="Tensile library directory of the bench's build, with " "--identity-stem",
    )
    ap.add_argument(
        "--identity-stem",
        action="append",
        default=[],
        metavar="STEM",
        help="library file whose solution indices the bench lines name " "(repeatable)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if bool(args.identity_stem) != (args.identity_dir is not None):
        ap.error("--identity-dir and --identity-stem go together")
    return args


def library_record(
    args: argparse.Namespace,
) -> Tuple[Optional[ki.LibraryIdentity], Optional[Dict[str, Any]]]:
    """(identity, kernel identity record without kernels listed) of the
    library files stage03 recorded, as --identity-dir holds them now. (None,
    None) without --identity-dir, or when stage03 recorded none: nothing
    checked its logs against a library, so the merged logs record none
    either. ValueError when a file stage03 recorded holds other kernels
    under its solution indices now, or cannot be read."""
    if args.identity_dir is None:
        return None, None
    try:
        then = ki.bench_record(args.stage03_dir / "logs")
        if then is None:
            return None, None
        now = ki.library_identity(args.identity_dir, list(then["files"]))
    except OSError as e:
        raise ValueError(str(e)) from None
    changed = ki.changed_files(then, now)
    if changed:
        why = (
            "its logs name other kernels than a re-bench would time"
            if then.get(ki.INDEX_DIGESTS)
            else "its record, written before index digests were recorded, cannot "
            "tell whether its logs name other kernels than a re-bench would time"
        )
        raise ValueError(
            f"stage03 in {args.stage03_dir} benched other library files than "
            f"{args.identity_dir} holds now ({', '.join(changed)}; library "
            f"digest {then['digest']} -> {now.digest}): {why}"
        )
    return now, ki.record(now, ())


def _clean(out_dir: Path) -> None:
    for name in ("logs", "runs"):
        shutil.rmtree(out_dir / name, ignore_errors=True)
    for name in ("rebench.json", "rebench_summary.json"):
        (out_dir / name).unlink(missing_ok=True)


def _read_summary(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _run_stage03(
    run_dir: Path,
    args: argparse.Namespace,
    blocks: int,
    children: procs.ChildGroups,
    library: Optional[ki.LibraryIdentity] = None,
) -> int:
    """stage03 over the run's lines; with `library` (the files stage03
    recorded, as --identity-dir holds them now) it names them, so that the
    warm-up line takes a kernel of each."""
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
    ]
    if args.config_yaml is not None:
        cmd += ["--config-yaml", str(args.config_yaml)]
    if library is not None and args.identity_dir is not None:
        cmd += ["--identity-dir", str(args.identity_dir)]
        for stem in library.stems:
            cmd += ["--identity-stem", stem]
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


def band_counts(
    jobs: List[rb.Shape], tie_sets: Mapping[rb.Shape, rb.TieSet], bands: rb.Bands
) -> Dict[str, Dict[str, int]]:
    """Per band: its GEMMs, their candidates and the runs each is read."""
    runs = {"fast": bands.fast_runs, "mid": bands.runs, "slow": bands.slow_runs}
    out = {b: {"gemms": 0, "candidates": 0, "runs": n} for b, n in runs.items()}
    for shape in jobs:
        ties = tie_sets[shape]
        entry = out[bands.band(ties.winner_us)]
        entry["gemms"] += 1
        entry["candidates"] += len(ties.candidates)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    children = procs.ChildGroups()
    devices = [int(d) for d in str(args.devices).split(",") if d.strip()]
    with procs.stop_children_on_exit(children):
        try:
            with gpu_lease.hold_usable(
                devices,
                "stage03b near-winner re-reads",
                args.out_dir,
                report=lambda m: ui.warn("lease", m),
            ) as leases:
                args.devices = ",".join(str(d) for d in leases.devices)
                return _run(args, children)
        except gpu_lease.LeaseError as e:
            ui.err("lease", str(e))
            return 2


def _run(args: argparse.Namespace, children: procs.ChildGroups) -> int:
    t0 = time.time()
    cfg: Dict[str, Any] = {}
    if args.config_yaml is not None:
        with args.config_yaml.open() as f:
            cfg = yaml.safe_load(f) or {}
    resolve_options(args, cfg)
    errors = option_errors(args)
    if errors:
        for e in errors:
            ui.err("args", e)
        return 1
    bands = rb.bands_of(vars(args))
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
        library, record = library_record(args)
    except ValueError as e:
        ui.err("identity", str(e))
        return 1
    if args.identity_dir is not None and record is None:
        ui.warn(
            "identity",
            f"stage03 in {args.stage03_dir} recorded no library files: the merged "
            f"logs record none either, so stage04 checks them by their solution "
            f"names as it checks logs without a record",
        )
    if not bench_lists_supported(args.bench_binary):
        ui.err(
            "err",
            f"{args.bench_binary} does not know solution_index_list; the re-bench "
            f"needs a hipblaslt-bench that does",
        )
        return 1

    logs = {p.name: rb.read_log_lines(p.read_text(errors="replace")) for p in log_paths}
    try:
        blocked = rb.blocked_solutions(
            logs.values(),
            cx.config_blocklist(cfg),
            library.ids if library is not None else None,
        )
    except ValueError as e:
        ui.err("blocklist", str(e))
        return 1
    tie_sets, n_complete = rb.tie_sets_of_logs(
        logs.values(), float(args.tolerance), int(args.max_kernels), blocked
    )
    lines_by_shape: Dict[rb.Shape, str] = {}
    for line in read_bench_lines(in_yaml):
        shape = shape_of_line(line)
        if shape is not None:
            lines_by_shape.setdefault(shape, line)
    jobs = [s for s in lines_by_shape if s in tie_sets]
    n_unmatched = len(tie_sets) - len(jobs)
    runs_of = {s: bands.runs_of(tie_sets[s].winner_us) for s in jobs}
    per_band = band_counts(jobs, tie_sets, bands)
    if not args.quiet:
        ui.banner("stage03b near-winner re-reads")
        ui.info(
            "cfg", f"stage03 logs : {len(log_paths)} in {args.stage03_dir / 'logs'}"
        )
        ui.info("cfg", f"bench yaml   : {in_yaml}")
        options = ", ".join(f"{k}={getattr(args, k)}" for k in CONFIG_KEYS)
        ui.info("cfg", f"options      : {options}")
        ui.info(
            "ties",
            f"{len(tie_sets)} of {n_complete} complete problems have near ties "
            f"({sum(len(tie_sets[s].candidates) for s in jobs)} candidates)",
        )
        if blocked:
            ui.info(
                "blocklist",
                f"{len(blocked)} kernel(s) of the library compute wrong results "
                f"(lib/correctness.py): none is a near-winner or the winner the "
                f"tolerance counts from",
            )
        for band, c in per_band.items():
            ui.info(
                "ties",
                f"{band:5s}: {c['gemms']} GEMMs, {c['candidates']} candidates, "
                f"{c['runs']} run(s) each",
            )
    if n_unmatched:
        ui.warn("ties", f"{n_unmatched} near-tie GEMM(s) have no line in {in_yaml}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _clean(args.out_dir)
    samples = rb.Samples()
    run_records: List[Dict[str, Any]] = []
    order_seed = kernel_order_seed(cfg)
    seed = (cfg.get("bench") or {}).get("order_seed")
    gemm_seed = int(DEFAULT_ORDER_SEED if seed is None else seed)
    n_devices = len([d for d in str(args.devices).split(",") if d.strip()])
    for r in range(max(runs_of.values(), default=0)):
        shapes = [s for s in jobs if runs_of[s] > r]
        if args.gemm_order == "shuffled":
            stable_rng(gemm_seed, "near-tie", r).shuffle(shapes)
        run_dir = args.out_dir / "runs" / f"run_{r}"
        run_dir.mkdir(parents=True)
        lines = []
        for shape in shapes:
            rot = None
            if cfg.get("hipblaslt"):
                rot = rotating_mb_for(Shape(*shape), cfg)
            ties = tie_sets[shape]
            lines.append(
                rb.reread_line(
                    lines_by_shape[shape],
                    rb.run_order(ties.sol_indices, r, order_seed, ("near-tie", *shape)),
                    rotating_mib=rot,
                    sacrificial=bool(args.sacrificial_kernel),
                    min_iters=bands.min_iters_of(ties.winner_us),
                ).rstrip("\n")
            )
        atomic_write_text(run_dir / "shapes.yaml", "".join(ln + "\n" for ln in lines))
        blocks = args.blocks_per_gpu or rb.blocks_per_gpu(len(lines), n_devices)
        t_run = time.time()
        rc = _run_stage03(run_dir, args, blocks, children, library)
        n_rows = 0
        for p in sorted((run_dir / "stage03" / "logs").glob("block_*.log")):
            n_rows += samples.add_log(
                rb.read_log_lines(p.read_text(errors="replace")), r
            )
        s03 = _read_summary(run_dir / "stage03" / "stage03_summary.json")
        rec = {
            "run": r,
            "rc": rc,
            "gemms": len(lines),
            "blocks_per_gpu": blocks,
            "wall_s": time.time() - t_run,
            "rows": n_rows,
            "quarantined": len(s03.get("quarantined") or []),
        }
        run_records.append(rec)
        (ui.ok if rc == 0 else ui.warn)(
            "run",
            f"run {r}: rc={rc}  {len(lines)} GEMMs  {n_rows} timed rows  "
            f"({ui.fmt_dur(rec['wall_s'])})"
            + (
                f"  {rec['quarantined']} quarantined attempt(s)"
                if rec["quarantined"]
                else ""
            ),
        )
    if run_records and not any(rec["rows"] for rec in run_records):
        ui.err("run", "no re-bench run timed any candidate; see runs/*/stage03.log")
        return 1

    (args.out_dir / "logs").mkdir(parents=True, exist_ok=True)
    changes: List[rb.ProblemChange] = []
    for name, lines in logs.items():
        merged, ch = rb.merge_log(
            lines,
            tie_sets,
            samples,
            with_original=bool(args.with_original),
            blocked=blocked,
        )
        atomic_write_text(args.out_dir / "logs" / name, "".join(merged))
        changes.extend(ch)

    by_shape = {ch.shape: ch for ch in changes}
    per_gemm = []
    for shape in jobs:
        ties = tie_sets[shape]
        ch = by_shape.get(shape)
        after = ch.winner_after if ch else ties.winner_sol
        per_gemm.append(
            {
                "m": shape[0],
                "n": shape[1],
                "k": shape[2],
                "batch": shape[3],
                "band": bands.band(ties.winner_us),
                "runs": runs_of[shape],
                "with_original": bool(args.with_original) and not ties.has_skip_row,
                "winner_before": ties.winner_sol,
                **ki.naming(library, ties.winner_sol, "winner_before_"),
                "winner_after": after,
                **ki.naming(library, after, "winner_after_"),
                "candidates": [
                    {
                        "sol_index": c.sol_index,
                        **ki.naming(library, c.sol_index),
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
    write_json(
        args.out_dir / "rebench.json",
        {
            "library_digest": record["digest"] if record else None,
            "with_original": bool(args.with_original),
            "gemms": per_gemm,
        },
    )
    launch = launch_settings.summarize(
        [launch_settings.read_record(args.stage03_dir / "logs")]
        + [
            launch_settings.read_record(
                args.out_dir / "runs" / f"run_{rec['run']}" / "stage03" / "logs"
            )
            for rec in run_records
        ]
    )
    if len(launch["heuristics"]) > 1:
        ui.warn(
            "launch",
            f"stage03 and the re-reads were benched with launch heuristics "
            f"{launch_settings.describe_digests(launch)}: their builds' "
            f"launch-rule sources differ",
        )
    summary = {
        "in_yaml": str(in_yaml),
        "options": {k: getattr(args, k) for k in CONFIG_KEYS},
        "bands": per_band,
        "n_unmatched_gemms": n_unmatched,
        **_summary_stats(changes, {s: tie_sets[s] for s in jobs}, n_complete),
        "runs": run_records,
        "wall_s": time.time() - t0,
        **({ki.RECORD_KEY: record} if record else {}),
        launch_settings.RECORD_KEY: launch,
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
