#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage04_convert_to_enriched_dataset -- hipblaslt-bench block logs ->
chunked CSV of enriched training rows.

Reads every `block_*.log` / `line*.log` in --log-dir: stage03's block logs,
including its retry pass (`block_retry_gpuD.log`). The log format is
described in lib/bench_log.py. Per problem:

  * Only complete problems are kept: the bench printed `Winner:` or reported
    every rank of `Is supported N`. A problem cut short by a crash would get a
    winner chosen from a prefix of its candidates.
  * Tested rows: bench columns are mapped by the header printed with each row;
    the identity is `--Solution index`.
  * Skip rows (`Skip solution: <rank> (... warm-up = <us> us, ...,
    solution index = <index>)`): the identity is the printed index, `us` is the
    single cold warm-up call, and the problem columns are copied from the
    problem's first tested row. Skip lines that do not parse are counted and
    reported.
  * Kernel parameters come from the library `.dat`, keyed by solution index.
    With --library-stem only that library is loaded and rows of solutions
    outside it are dropped and counted.
  * One row per (problem, solution): solutions with identical kernel
    parameters keep their own rows and latencies.
  * is_winner marks the fastest tested row. is_origami_pick marks the tested
    row of rank 0, the library's own pick: the Origami ranking, because
    training benches run with tilewright off (lib/bench_env.py).

Output in --out-dir (previous outputs are removed first):
  chunk_NNNN.csv        --rows-per-chunk rows each, columns OUT_FIELDS
  kernels.json          the kernel attributes (lib/dat.py) of every solution
                        the rows name
  enrich_summary.json   row and problem counts and every drop reason
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_THIS_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from lib import ui  # noqa: E402
from lib.bench_log import LogStats, Problem, iter_problems  # noqa: E402
from lib.dat import (  # noqa: E402
    KERNELS_FILE,
    KernelIndex,
    library_logic_path,
    load_kernel_index,
    parse_scale_mode,
    write_kernel_attributes,
)
from lib.fs import write_json  # noqa: E402

# Bench columns kept, by header name.
DATA_FIELDS = [
    "transA",
    "transB",
    "grouped_gemm",
    "batch_count",
    "m",
    "n",
    "k",
    "alpha",
    "lda",
    "stride_a",
    "beta",
    "ldb",
    "stride_b",
    "ldc",
    "stride_c",
    "ldd",
    "stride_d",
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "compute_type",
    "scaleA",
    "scaleB",
    "rotating_buffer",
    "flush",
    "use_gpu_timer",
    "hipblaslt-Gflops",
    "hipblaslt-GB/s",
    "us",
]
PERF_FIELDS = ("hipblaslt-Gflops", "hipblaslt-GB/s", "us")
PROBLEM_FIELDS = [f for f in DATA_FIELDS if f not in PERF_FIELDS]
# Adaptive-timing statistics of a tested row; empty when the bench ran with a
# fixed iteration count, and on skip rows.
NOISE_FIELDS = ["samples", "cv", "rel_iqr", "status"]
CONFIG_FIELDS = [
    "mt_m",
    "mt_n",
    "mt_k",
    "mi_m",
    "mi_n",
    "mi_k",
    "occupancy",
    "cache_hints_a",
    "cache_hints_b",
    "grvw_a",
    "grvw_b",
    "gwvw_d",
]
OUT_FIELDS = (
    [
        "is_winner",
        "is_skip",
        "is_origami_pick",
        "rank",
        "sol_idx_global",
        "sol_idx_local",
    ]
    + DATA_FIELDS
    + NOISE_FIELDS
    + CONFIG_FIELDS
)

_MT_NAME_RE = re.compile(r"_MT(\d+)x(\d+)x(\d+)_")
# A kernel name disagreeing with the .dat entry of its solution index means the
# index space does not match the library the bench loaded; a few are tolerated
# before giving up, a correct setup has none.
_MT_MISMATCH_FATAL_AFTER = 10


class Enricher:
    """Kernel parameters by solution index, with a check of each tested row's
    kernel name against the .dat macro tile."""

    def __init__(self, index: KernelIndex, quiet: bool = False) -> None:
        self.by_index = index.by_index
        self.quiet = quiet
        self.mt_mismatches = 0

    def config_for(
        self, sol_index: int, kernel_name: str = ""
    ) -> Optional[Dict[str, Any]]:
        info = self.by_index.get(int(sol_index))
        if info is not None and kernel_name:
            self._check_macro_tile(int(sol_index), kernel_name, info)
        return info

    def _check_macro_tile(
        self, sol_index: int, kernel_name: str, info: Dict[str, Any]
    ) -> None:
        m = _MT_NAME_RE.search(kernel_name)
        if not m:
            return
        named = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        dat_mt = (info["mt_m"], info["mt_n"], info["mt_k"])
        if named == dat_mt:
            return
        self.mt_mismatches += 1
        if self.mt_mismatches <= 5 and not self.quiet:
            ui.warn(
                "mt",
                f"solution {sol_index}: kernel name MT={named} but .dat MT={dat_mt}",
            )
        if self.mt_mismatches > _MT_MISMATCH_FATAL_AFTER:
            raise RuntimeError(
                f"{self.mt_mismatches} kernel names disagree with the .dat macro "
                f"tile of their solution index (e.g. solution {sol_index}: name "
                f"{named}, .dat {dat_mt}); the .dat directory does not match the "
                f"library the bench loaded"
            )


def _us_value(row: Dict[str, Any]) -> Optional[float]:
    try:
        v = float(row["us"])
    except (KeyError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def _blank_if_none(v: Any) -> Any:
    return "" if v is None else v


def problem_rows(
    prob: Problem, enricher: Enricher, counters: Counter
) -> List[Dict[str, Any]]:
    """Enriched rows of one complete problem."""
    base = next((t.fields for t in prob.tested if t.fields), None)
    if base is None:
        counters["problems_without_tested_row"] += 1
        return []
    problem_cols = {f: base.get(f, "") for f in PROBLEM_FIELDS}
    rows: List[Dict[str, Any]] = []
    for t in prob.tested:
        if t.sol_index is None:
            counters["tested_without_index"] += 1
            continue
        cfg = enricher.config_for(t.sol_index, t.sol_name)
        if cfg is None:
            counters["tested_outside_library"] += 1
            continue
        row: Dict[str, Any] = {f: t.fields.get(f, "") for f in DATA_FIELDS}
        row.update({f: t.fields.get(f, "") for f in NOISE_FIELDS})
        row.update(
            is_winner="False",
            is_skip="False",
            is_origami_pick=1 if t.rank == 0 else 0,
            rank=t.rank,
            sol_idx_global=t.sol_index,
            sol_idx_local=_blank_if_none(cfg.get("sol_idx_local")),
        )
        row.update({f: cfg[f] for f in CONFIG_FIELDS})
        rows.append(row)
    for s in prob.skipped:
        if s.sol_index is None:
            counters["skip_without_index"] += 1
            continue
        cfg = enricher.config_for(s.sol_index)
        if cfg is None:
            counters["skip_outside_library"] += 1
            continue
        if s.warm_up_us is None:
            counters["skip_nonfinite_warmup"] += 1
        row = dict(problem_cols)
        row.update({f: "" for f in NOISE_FIELDS})
        row.update(
            {
                "hipblaslt-Gflops": "",
                "hipblaslt-GB/s": "",
                "us": "" if s.warm_up_us is None else repr(s.warm_up_us),
                "is_winner": "False",
                "is_skip": "True",
                "is_origami_pick": 0,
                "rank": s.rank,
                "sol_idx_global": s.sol_index,
                "sol_idx_local": _blank_if_none(cfg.get("sol_idx_local")),
            }
        )
        row.update({f: cfg[f] for f in CONFIG_FIELDS})
        rows.append(row)
    rows.sort(key=lambda r: int(r["rank"]))
    tested = [r for r in rows if r["is_skip"] == "False" and _us_value(r)]
    if not tested:
        counters["problems_without_measured_tested_row"] += 1
        return []
    min(tested, key=_us_value)["is_winner"] = "True"
    return rows


def process_log_file(
    path: str,
    enricher: Enricher,
    counters: Counter,
    stats: LogStats,
    max_problems: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Enriched rows of every complete problem in one log, and the number of
    problems that produced rows (at most `max_problems`)."""
    out: List[Dict[str, Any]] = []
    kept = 0
    with open(path, errors="replace") as f:
        for prob in iter_problems(f, stats):
            if max_problems is not None and kept >= max_problems:
                break
            counters["problems"] += 1
            if not prob.complete:
                counters[
                    (
                        "problems_no_solution"
                        if prob.no_solution
                        else "problems_incomplete"
                    )
                ] += 1
                continue
            rows = problem_rows(prob, enricher, counters)
            if not rows:
                counters["problems_without_rows"] += 1
                continue
            counters["problems_kept"] += 1
            kept += 1
            out.extend(rows)
    return out, kept


# ── output ───────────────────────────────────────────────────────────────────


class ChunkWriter:
    def __init__(self, out_dir: str, rows_per_chunk: int, prefix: str = "chunk_"):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.rows_per_chunk = max(1, int(rows_per_chunk))
        self.prefix = prefix
        self.n_chunks = 0
        self.total_rows = 0
        self._rows_in_chunk = 0
        self._writer: Optional[csv.DictWriter] = None
        self._fh = None

    def _open(self) -> None:
        path = os.path.join(self.out_dir, f"{self.prefix}{self.n_chunks:04d}.csv")
        self._fh = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=OUT_FIELDS)
        self._writer.writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self._open()
        assert self._writer is not None
        self._writer.writerow(row)
        self._rows_in_chunk += 1
        self.total_rows += 1
        if self._rows_in_chunk >= self.rows_per_chunk:
            self._finish_chunk()

    def _finish_chunk(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
            self.n_chunks += 1
            self._rows_in_chunk = 0

    def close(self) -> Tuple[int, int]:
        self._finish_chunk()
        return self.n_chunks, self.total_rows


def _remove_previous_outputs(out_dir: str) -> int:
    if not os.path.isdir(out_dir):
        return 0
    removed = 0
    for name in os.listdir(out_dir):
        if (name.startswith("chunk_") and name.endswith(".csv")) or (
            name in ("enrich_summary.json", KERNELS_FILE)
        ):
            os.unlink(os.path.join(out_dir, name))
            removed += 1
    return removed


class _ColumnCheck:
    """Distinct values of the kernel-parameter columns over tested rows."""

    def __init__(self) -> None:
        self.n_tested = 0
        self.values: Dict[str, set] = {f: set() for f in CONFIG_FIELDS}

    def add(self, row: Dict[str, Any]) -> None:
        if row["is_skip"] != "False":
            return
        self.n_tested += 1
        for f in CONFIG_FIELDS:
            if len(self.values[f]) < 2:
                self.values[f].add(row[f])

    def constant_columns(self) -> List[str]:
        return [f for f in CONFIG_FIELDS if len(self.values[f]) == 1]


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="hipblaslt-bench block logs -> chunked enriched CSV folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--log-dir", required=True, help="directory with block_*.log")
    ap.add_argument("--out-dir", required=True, help="output dir for chunk_NNNN.csv")
    ap.add_argument(
        "--dat-dir",
        required=True,
        help="Tensile library dir with the .dat / .dat.zlib logic files "
        "(<build>/Tensile/library/<arch>)",
    )
    ap.add_argument(
        "--library-stem",
        default=None,
        help="Tensile library stem (TensileLibrary_..._<arch>); load only "
        "<dat-dir>/<stem>.dat[.zlib] and drop rows of solutions outside it. "
        "Without it every contraction library in --dat-dir is loaded.",
    )
    ap.add_argument(
        "--scale-mode",
        type=int,
        default=None,
        help="hipblaslt scaleA mode of the run (0 none, 1 Scalar, 3 MX "
        "block-32). Without --library-stem, only libraries expecting this mode "
        "are loaded (the gfx1250 MX and non-MX F8 TN libraries share one "
        "dtype/layout); with it, a conflicting library is an error.",
    )
    ap.add_argument("--rows-per-chunk", type=int, default=10_000)
    ap.add_argument(
        "--max-gemms",
        type=int,
        default=None,
        help="stop after this many enriched GEMMs (small-subset test)",
    )
    ap.add_argument(
        "--progress-every", type=float, default=10.0, help="seconds between reports"
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if not args.quiet:
        ui.banner("hipblaslt-bench logs  ->  enriched chunked CSV")
        ui.info("cfg", f"log_dir        : {args.log_dir}")
        ui.info("cfg", f"out_dir        : {args.out_dir}")
        ui.info("cfg", f"dat_dir        : {args.dat_dir}")
        ui.info("cfg", f"library_stem   : {args.library_stem}")
        ui.info("cfg", f"scale_mode     : {args.scale_mode}")
        ui.info("cfg", f"rows_per_chunk : {args.rows_per_chunk:,}")

    if not os.path.isdir(args.log_dir):
        ui.err("err", f"--log-dir does not exist: {args.log_dir}")
        return 1
    log_files = sorted(
        f
        for f in os.listdir(args.log_dir)
        if f.endswith(".log") and (f.startswith("block_") or f.startswith("line"))
    )
    if not log_files:
        ui.err("err", f"no block_*.log files in {args.log_dir}")
        return 1

    try:
        if args.library_stem:
            lib_path = library_logic_path(args.dat_dir, args.library_stem)
            lib_mode = parse_scale_mode(lib_path.name)
            if (
                args.scale_mode is not None
                and lib_mode is not None
                and lib_mode != int(args.scale_mode)
            ):
                ui.err(
                    "err",
                    f"--scale-mode {args.scale_mode} conflicts with library "
                    f"{args.library_stem} (expects scale mode {lib_mode})",
                )
                return 1
        index = load_kernel_index(
            args.dat_dir, library_stem=args.library_stem, scale_mode=args.scale_mode
        )
    except (FileNotFoundError, ValueError) as exc:
        ui.err("err", str(exc))
        return 1
    if not index.by_index:
        ui.err("err", f"no solutions loaded from {args.dat_dir}")
        return 1
    if not args.quiet:
        ui.ok(
            "load",
            f"kernel index: {ui.fmt_int(len(index.by_index))} solutions from "
            f"{len(index.files)} logic file(s)",
        )
        if index.skipped_scale_mode:
            ui.info(
                "load", f"scale-mode filter skipped {index.skipped_scale_mode} file(s)"
            )
        if not args.library_stem:
            ui.warn(
                "load",
                "no --library-stem: rows of every loaded library are kept",
            )
    for name in index.unreadable:
        ui.warn("load", f"unreadable Tensile logic file skipped: {name}")
    for reason in index.invalid:
        ui.warn("load", f"Tensile logic file TensileLite cannot read skipped: {reason}")
    if index.missing_keys:
        ui.warn(
            "load",
            "solutions without sizeMapping keys TensileLite requires (their "
            "kernel attributes take the C++ defaults): "
            + ", ".join(f"{k} x{n}" for k, n in index.missing_keys.items()),
        )
    if index.duplicate_indices:
        ui.warn(
            "load",
            f"{index.duplicate_indices} solution index(es) appear in more than "
            f"one logic file; the first file's entry is used",
        )

    removed = _remove_previous_outputs(args.out_dir)
    if removed and not args.quiet:
        ui.info("out", f"removed {removed} output file(s) of a previous run")
    enricher = Enricher(index, quiet=args.quiet)
    writer = ChunkWriter(args.out_dir, args.rows_per_chunk)
    counters: Counter = Counter()
    stats = LogStats()
    columns = _ColumnCheck()
    written: Set[int] = set()
    last_progress = time.time()
    total_kept = 0
    try:
        for i, fname in enumerate(log_files, 1):
            remaining = (
                None if args.max_gemms is None else max(0, args.max_gemms - total_kept)
            )
            if remaining == 0:
                break
            t_file = time.time()
            rows, kept = process_log_file(
                os.path.join(args.log_dir, fname),
                enricher,
                counters,
                stats,
                max_problems=remaining,
            )
            for r in rows:
                writer.write(r)
                columns.add(r)
                written.add(int(r["sol_idx_global"]))
                counters["rows_skip" if r["is_skip"] == "True" else "rows_tested"] += 1
            total_kept += kept
            if not args.quiet:
                ui.grey(
                    f"file {i:>3d}/{len(log_files)}",
                    f"{fname}  gemms={kept:>5d}  rows={ui.fmt_int(len(rows)):>10s}  "
                    f"({ui.fmt_dur(time.time() - t_file)})",
                )
                if time.time() - last_progress >= args.progress_every:
                    ui.info(
                        "progress",
                        f"gemms={ui.fmt_int(total_kept)}  rows={ui.fmt_int(writer.total_rows)}"
                        f"  elapsed={ui.fmt_dur(time.time() - t0)}",
                    )
                    last_progress = time.time()
    finally:
        n_chunks, total_rows = writer.close()
    write_kernel_attributes(
        os.path.join(args.out_dir, KERNELS_FILE),
        {sid: index.by_index[sid] for sid in written},
        args.library_stem,
    )

    summary = {
        "log_files": len(log_files),
        "library_stem": args.library_stem,
        "library_files": index.files,
        "invalid_library_files": index.invalid,
        "n_library_solutions": len(index.by_index),
        "missing_size_mapping_keys": index.missing_keys,
        "gemms": total_kept,
        "rows": total_rows,
        "chunks": n_chunks,
        "n_kernels": len(written),
        "skip_lines": stats.skip_lines,
        "unparsed_skip_lines": stats.unparsed_skip_lines,
        "unparsed_skip_examples": stats.unparsed_skip_examples,
        "malformed_tested_rows": stats.malformed_rows,
        "mt_mismatches": enricher.mt_mismatches,
        "counters": dict(sorted(counters.items())),
    }
    write_json(os.path.join(args.out_dir, "enrich_summary.json"), summary)

    if stats.unparsed_skip_lines:
        ui.warn(
            "parse",
            f"{stats.unparsed_skip_lines} 'Skip solution' line(s) did not parse "
            f"and were dropped; first: {stats.unparsed_skip_examples[0]!r}",
        )
    outside = counters["tested_outside_library"] + counters["skip_outside_library"]
    if outside:
        ui.info(
            "filter",
            f"dropped {outside} row(s) of solutions outside the loaded library",
        )
    for f in columns.constant_columns():
        ui.warn(
            "validate",
            f"kernel column '{f}' is constant (value={next(iter(columns.values[f]))}) "
            f"across {ui.fmt_int(columns.n_tested)} tested rows -- verify this is "
            f"expected for this library",
        )
    if not args.quiet:
        ui.banner("Done", ui.C.GREEN)
        ui.ok("done", f"gemms          : {ui.fmt_int(total_kept)}")
        ui.ok(
            "done",
            f"rows           : {ui.fmt_int(total_rows)} in {n_chunks} chunk(s)  "
            f"(tested {ui.fmt_int(counters['rows_tested'])}, "
            f"skip {ui.fmt_int(counters['rows_skip'])})",
        )
        ui.ok(
            "done",
            f"problems       : {ui.fmt_int(counters['problems'])} seen, "
            f"{ui.fmt_int(counters['problems_incomplete'])} incomplete, "
            f"{ui.fmt_int(counters['problems_no_solution'])} without solution",
        )
        ui.ok("done", f"kernels        : {ui.fmt_int(len(written))} in {KERNELS_FILE}")
        ui.info("done", f"summary        : {json.dumps(summary['counters'])}")
        ui.ok("done", f"elapsed        : {ui.fmt_dur(time.time() - t0)}")
    if columns.n_tested == 0:
        ui.err("validate", "no tested rows were produced; the dataset is empty")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
