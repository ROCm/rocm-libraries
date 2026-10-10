#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage04_convert_to_enriched_dataset -- hipblaslt-bench block logs ->
chunked CSV of enriched training rows.

Reads every `block_*.log` / `line*.log` in each --log-dir (the option
repeats): stage03's block logs, including its retry pass
(`block_retry_gpuD.log`). The log format is described in lib/bench_log.py.
Per problem:

  * Only complete problems are kept: the bench printed `Winner:` or reported
    every rank of `Is supported N`. A problem cut short by a crash would get a
    winner chosen from a prefix of its candidates.
  * Tested rows: bench columns are mapped by the header printed with each row;
    the identity is `--Solution index`. With --config-yaml every tested row
    must name the data types, compute type and scale modes of the config's
    `hipblaslt` section as the bench prints them (lib/bench_yaml.py
    `printed_types`); the first that does not stops the stage with 1, since
    the logs then hold another problem type than the run trains.
  * Skip rows (`Skip solution: <rank> (... warm-up = <us> us, ...,
    solution index = <index>)`): the identity is the printed index, `us` is the
    single cold warm-up call, and the problem columns are copied from the
    problem's first tested row. Skip lines that do not parse are counted and
    reported.
  * Rows of a kernel the config's blocklist names (lib/correctness.py: one
    that computes wrong results) are dropped (`tested_rows_blocked`,
    `skip_rows_blocked`), so it is never a GEMM's winner or a label.
  * Kernel parameters come from the library `.dat`, keyed by solution index.
    With --library-stem only that library is loaded and rows of solutions
    outside it are dropped and counted. A log directory whose bench recorded
    the identity of its library (lib/kernel_identity.py `bench_record`:
    stage03 and stage03b do) is read only when none of those files holds
    other kernels under its solution indices in --dat-dir now (a Prediction
    table that changed alone does not count), and then a tested row whose
    `--Solution name` is not the `.dat` name of its index is fatal. Without
    a record, a problem with such a row is dropped (its indices are not this
    library's), and more than a few such rows mean the bench loaded another
    hipBLASLt build than --dat-dir, which is fatal.
  * One row per (problem, solution): solutions with identical kernel
    parameters keep their own rows and latencies. The sacrificial copy a
    screened line leads with (lib/bench_log.py) gets no row
    (`sacrificial_rows` counts them), so its reading never becomes a label;
    the other rows keep their bench rank.
  * is_winner marks the fastest tested row. is_origami_pick marks the tested
    row of rank 0, the library's own pick: the Origami ranking, because
    training benches run with tilewright off (lib/bench_env.py). GEMMs of a
    --plan were benched in the planner's order, so none of their rows is
    marked.
  * canary_flagged is 1 on every row of a log holding stage03's
    `*** CANARY FLAGGED` line (lib/bench_log.py CANARY_FLAG_MARK): its
    block's data was kept although the bench's canary read the host as
    loaded; 0 elsewhere, or the column's absence in older CSVs.
  * cu_budget / schedule: the execution context the bench ran under
    (--cu-budget, --schedule); 0 / default, or the columns' absence in older
    CSVs, mean exclusive use of the device. --context-log-dir BUDGET SCHEDULE
    DIR reads DIR's logs as benched under that context instead, so one run
    enriches the logs of several contexts (each (shape, context) is its own
    GEMM downstream); --context-plan BUDGET SCHEDULE PLAN gives a context
    its own plan instead of --plan.
  * With --plan (stage02's plan.json), per GEMM of the plan: is_planned is 1
    for a solution on its kernel list and n_planned is the list length; a
    GEMM benched exhaustively (no list) or outside the plan gets is_planned 1
    and an empty n_planned, as does every row without --plan. A plan that
    records kernel identities (lib/kernel_identity.py) and was made with
    other library files than --dat-dir holds now names each listed kernel by
    the index the kernel has now; a listed kernel the library no longer
    holds under one index is fatal.
  * --extra-library-stem adds libraries whose rows are kept as well (the
    planner's bank kernels); kernels.json still names --library-stem.
  * --sustained-log-dir DIR reads DIR's logs as the sustained finalist
    re-measurement of stage03d (lib/sustained.py): their rows get
    is_sustained True, a label source lib/repeats.py prefers, and none is
    an Origami pick. Every other row has is_sustained False.

Output in --out-dir (previous outputs are removed first):
  chunk_NNNN.csv        --rows-per-chunk rows each, columns OUT_FIELDS
  kernels.json          the kernel attributes (lib/dat.py), kernel ids
                        (lib/kernel_identity.py, with the digests of the
                        library files; without --library-stem, of the files
                        whose rows were written) and signatures of every
                        solution the rows name
  enrich_summary.json   row and problem counts, every drop reason, the
                        blocklist whose kernels' rows were dropped
                        (`blocklist`: lib/correctness.py `applied_record`)
                        and how the benches of the logs launched the kernels
                        (`launch`: lib/launch_settings.py `summarize` of
                        the record each log directory's bench left)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_THIS_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from lib import kernel_identity as ki  # noqa: E402
from lib import launch_settings  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_log import (  # noqa: E402
    CANARY_FLAG_MARK,
    LogStats,
    Problem,
    iter_problems,
)
from lib.bench_yaml import SCALE_COLUMNS, printed_types, scale_mode_code  # noqa: E402
from lib.correctness import applied_record, config_blocklist  # noqa: E402
from lib.dat import (  # noqa: E402
    KERNELS_FILE,
    KernelIndex,
    library_logic_path,
    load_kernel_index,
    logic_stem,
    parse_scale_mode,
    write_kernel_attributes,
)
from lib.fs import write_json  # noqa: E402
from lib.planner import ShapeKey, plan_lookup  # noqa: E402

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
FLAG_FIELDS = ["canary_flagged"]
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
CONTEXT_FIELDS = ["cu_budget", "schedule"]
SCHEDULES = ("default", "dynamic", "auto")
PLAN_FIELDS = ["is_planned", "n_planned"]
OUT_FIELDS = (
    [
        "is_winner",
        "is_skip",
        "is_sustained",
        "is_origami_pick",
        "rank",
        "sol_idx_global",
        "sol_idx_local",
    ]
    + DATA_FIELDS
    + NOISE_FIELDS
    + FLAG_FIELDS
    + CONFIG_FIELDS
    + CONTEXT_FIELDS
    + PLAN_FIELDS
)

# A solution name disagreeing with the .dat entry of its solution index means
# the index space does not match the library the bench loaded; a few are
# tolerated before giving up, a correct setup has none.
_NAME_MISMATCH_FATAL_AFTER = 10


def _short_name(kernel: Any) -> str:
    """A kernel the bench printed (its solution name) or the .dat holds (its
    `kernel_dat_info`), named by signature and name hash."""
    if isinstance(kernel, Mapping):
        return ki.kernel_id(kernel)
    return ki.label_of_name(kernel)


class TypeMismatch(RuntimeError):
    """A tested row of another problem type than the config's."""


class Enricher:
    """Kernel parameters by solution index, and the check that a tested
    row's solution name is the .dat name of its index (`strict`: logs whose
    bench recorded this library, where any other name is fatal) and, with
    `expect` (`printed_types` of the config), that it names the config's
    problem type."""

    def __init__(self, index: KernelIndex, quiet: bool = False) -> None:
        self.by_index = index.by_index
        self.quiet = quiet
        self.expect: Optional[Dict[str, Any]] = None
        self.name_mismatches = 0
        self.strict = False
        self.blocked: Set[int] = set()
        self._by_name: Optional[Dict[str, List[int]]] = None

    def config_for(self, sol_index: int) -> Optional[Dict[str, Any]]:
        return self.by_index.get(int(sol_index))

    def ran(self, solution_name: str) -> str:
        """The kernel the bench printed `solution_name` of: its kernel id and
        solution index here when the library holds a solution of that name,
        else its macro tile and name hash."""
        if self._by_name is None:
            self._by_name = {}
            for sid, info in sorted(self.by_index.items()):
                self._by_name.setdefault(str(info["kernel_name"]), []).append(sid)
        sids = self._by_name.get(solution_name)
        if not sids:
            return f"{_short_name(solution_name)}, which this library does not hold"
        here = ", ".join(str(s) for s in sids)
        return f"{ki.kernel_id(self.by_index[sids[0]])} (solution {here} here)"

    def name_matches(self, sol_index: int, solution_name: str) -> bool:
        """False when the .dat names solution `sol_index` other than the
        bench did (`solution_name`; an empty one or an index outside the
        library is not checked); RuntimeError at once when `strict`, else
        after too many."""
        info = self.by_index.get(int(sol_index))
        if info is None or not solution_name or solution_name == info["kernel_name"]:
            return True
        self.name_mismatches += 1
        named, dat_name = self.ran(solution_name), _short_name(info)
        if self.strict:
            raise RuntimeError(
                f"solution {sol_index}: the bench ran {named}, the .dat names "
                f"{dat_name}, although the bench recorded these library files: "
                f"the .dat directory is not the library the bench loaded"
            )
        if self.name_mismatches <= 5 and not self.quiet:
            ui.warn(
                "identity",
                f"solution {sol_index}: the bench ran {named}, the .dat names "
                f"{dat_name}",
            )
        if self.name_mismatches > _NAME_MISMATCH_FATAL_AFTER:
            raise RuntimeError(
                f"{self.name_mismatches} solution names the bench printed disagree "
                f"with the .dat entry of their solution index (e.g. solution "
                f"{sol_index}: bench {named}, .dat {dat_name}); the .dat directory "
                f"is not the library the bench loaded: the hipBLASLt build changed"
            )
        return False

    def check_types(self, prob: Problem) -> None:
        """TypeMismatch for the first tested row of `prob` whose problem
        type differs from `expect` (a column the row lacks, or a scale mode
        it names otherwise than by number or name, is not compared)."""
        for t in prob.tested if self.expect else ():
            wrong = []
            for key, want in self.expect.items():
                got = t.fields.get(key)
                if got in (None, ""):
                    continue
                if key in SCALE_COLUMNS:
                    code = scale_mode_code(got)
                    if code is None or code == int(want):
                        continue
                elif got.strip() == want:
                    continue
                wrong.append(f"{key} {got.strip()} (the config's: {want})")
            if wrong:
                raise TypeMismatch(
                    f"problem {prob.shape()} at line {t.line_no} ran "
                    f"{', '.join(wrong)}: the bench ran another problem type than "
                    f"the config's"
                )


def _us_value(row: Dict[str, Any]) -> Optional[float]:
    try:
        v = float(row["us"])
    except (KeyError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def _blank_if_none(v: Any) -> Any:
    return "" if v is None else v


def _plan_entry(
    prob: Problem, plan: Optional[Dict[ShapeKey, Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    if not plan:
        return None
    shape = prob.shape()
    return plan.get(shape) if shape is not None else None


def problem_rows(
    prob: Problem,
    enricher: Enricher,
    counters: Counter,
    plan: Optional[Dict[ShapeKey, Dict[str, Any]]] = None,
    sustained: bool = False,
) -> List[Dict[str, Any]]:
    """Enriched rows of one complete problem (of a sustained re-measurement
    with `sustained`)."""
    if prob.sacrificial is not None:
        counters["sacrificial_rows"] += 1
    base = next((t.fields for t in prob.tested if t.fields), None)
    if base is None:
        counters["problems_without_tested_row"] += 1
        return []
    enricher.check_types(prob)
    problem_cols = {f: base.get(f, "") for f in PROBLEM_FIELDS}
    entry = _plan_entry(prob, plan)
    listed: Optional[Set[int]] = None
    if entry is not None and entry.get("kernels") is not None:
        listed = {int(s) for s in entry["kernels"]}
    n_planned = "" if listed is None else len(listed)
    rank0_is_origami = entry is None and not sustained
    flag = "True" if sustained else "False"

    def plan_cols(sol_index: int) -> Dict[str, Any]:
        planned = listed is None or int(sol_index) in listed
        if not planned:
            counters["rows_unplanned"] += 1
        return {"is_planned": 1 if planned else 0, "n_planned": n_planned}

    named = [
        enricher.name_matches(t.sol_index, t.sol_name)
        for t in prob.tested
        if t.sol_index is not None
    ]
    if not all(named):
        counters["problems_naming_other_kernels"] += 1
        return []
    rows: List[Dict[str, Any]] = []
    for t in prob.tested:
        if t.sol_index is None:
            counters["tested_without_index"] += 1
            continue
        cfg = enricher.config_for(t.sol_index)
        if cfg is None:
            counters["tested_outside_library"] += 1
            continue
        if int(t.sol_index) in enricher.blocked:
            counters["tested_rows_blocked"] += 1
            continue
        row: Dict[str, Any] = {f: t.fields.get(f, "") for f in DATA_FIELDS}
        row.update({f: t.fields.get(f, "") for f in NOISE_FIELDS})
        row.update(
            is_winner="False",
            is_skip="False",
            is_sustained=flag,
            is_origami_pick=1 if (rank0_is_origami and t.rank == 0) else 0,
            rank=t.rank,
            sol_idx_global=t.sol_index,
            sol_idx_local=_blank_if_none(cfg.get("sol_idx_local")),
        )
        row.update({f: cfg[f] for f in CONFIG_FIELDS})
        row.update(plan_cols(t.sol_index))
        rows.append(row)
    for s in prob.skipped:
        if s.sol_index is None:
            counters["skip_without_index"] += 1
            continue
        cfg = enricher.config_for(s.sol_index)
        if cfg is None:
            counters["skip_outside_library"] += 1
            continue
        if int(s.sol_index) in enricher.blocked:
            counters["skip_rows_blocked"] += 1
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
                "is_sustained": flag,
                "is_origami_pick": 0,
                "rank": s.rank,
                "sol_idx_global": s.sol_index,
                "sol_idx_local": _blank_if_none(cfg.get("sol_idx_local")),
            }
        )
        row.update({f: cfg[f] for f in CONFIG_FIELDS})
        row.update(plan_cols(s.sol_index))
        rows.append(row)
    rows.sort(key=lambda r: int(r["rank"]))
    tested = [r for r in rows if r["is_skip"] == "False" and _us_value(r)]
    if not tested:
        counters["problems_without_measured_tested_row"] += 1
        return []
    min(tested, key=_us_value)["is_winner"] = "True"
    if entry is not None:
        counters["gemms_in_plan"] += 1
    return rows


def process_log_file(
    path: str,
    enricher: Enricher,
    counters: Counter,
    stats: LogStats,
    max_problems: Optional[int] = None,
    plan: Optional[Dict[ShapeKey, Dict[str, Any]]] = None,
    sustained: bool = False,
) -> Tuple[List[Dict[str, Any]], int]:
    """Enriched rows of every complete problem in one log, and the number of
    problems that produced rows (at most `max_problems`). The rows of a log
    holding a CANARY_FLAG_MARK line have `canary_flagged` 1."""
    out: List[Dict[str, Any]] = []
    kept = 0
    flagged = False
    with open(path, errors="replace") as f:

        def lines():
            nonlocal flagged
            for line in f:
                flagged = flagged or line.startswith(CANARY_FLAG_MARK)
                yield line

        for prob in iter_problems(lines(), stats):
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
            rows = problem_rows(prob, enricher, counters, plan, sustained)
            if not rows:
                counters["problems_without_rows"] += 1
                continue
            counters["problems_kept"] += 1
            kept += 1
            out.extend(rows)
        flagged = flagged or any(line.startswith(CANARY_FLAG_MARK) for line in f)
    for row in out:
        row["canary_flagged"] = 1 if flagged else 0
    if flagged:
        counters["logs_canary_flagged"] += 1
        counters["problems_canary_flagged"] += kept
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


def _context_arg(values: List[str]) -> Tuple[str, int, str]:
    """(path, cu_budget, schedule) of a --context-log-dir / --context-plan
    value (CU_BUDGET SCHEDULE PATH)."""
    budget, schedule, path = values
    try:
        b = int(budget)
    except ValueError:
        raise ValueError(f"cu budget {budget!r} is not an integer") from None
    if b < 0:
        raise ValueError(f"cu budget {b} must be >= 0")
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule {schedule!r} is not one of {SCHEDULES}")
    return path, b, schedule


def _read_plan(path: str, dat_dir: str) -> Dict[ShapeKey, Dict[str, Any]]:
    """The shapes of a plan.json, their kernel lists as indices of the
    library in `dat_dir`: a plan that records its kernels and was made with
    other library files maps each listed kernel to its index now
    (lib/kernel_identity.py); BuildChangedError when a listed kernel is no
    longer in the library under one index. The lists of a plan that records
    no kernels are taken as they are, with a warning."""
    with open(path) as f:
        doc = json.load(f)
    lookup = plan_lookup(doc)
    rec = ki.read_record(doc)
    listed = {
        int(s)
        for e in lookup.values()
        if e.get("kernels") is not None
        for s in e["kernels"]
    }
    if rec is None and listed:
        ki.warn_unrecorded(f"plan {path}")
    if rec is None or not listed:
        return lookup
    remap = ki.Remap.of(
        rec, ki.library_identity(dat_dir, list(rec["files"])), f"plan {path}"
    )
    if remap.same:
        return lookup
    gone = sorted(s for s in listed if remap.index(s) is None)
    if gone:
        raise ki.BuildChangedError(
            f"{remap.explain()}; {len(gone)} of the kernels its lists name are "
            f"among them (e.g. {remap.kernels_then.get(gone[0], gone[0])})"
        )
    ui.info(
        "identity",
        f"{remap.explain()}; its kernel lists are read with their kernels' "
        f"current indices",
    )
    return {
        shape: (
            dict(e, kernels=remap.indices(e["kernels"]))
            if e.get("kernels") is not None
            else e
        )
        for shape, e in lookup.items()
    }


def _read_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg if isinstance(cfg, dict) else {}


def config_types(path: str) -> Dict[str, Any]:
    """`printed_types` of the `hipblaslt` section of the config at `path`;
    ValueError when it has none."""
    cfg = _read_config(path)
    hl = cfg.get("hipblaslt")
    if not isinstance(hl, dict):
        raise ValueError("it has no hipblaslt section")
    return printed_types(hl)


def _bench_strictness(log_dirs: List[str], dat_dir: str) -> Dict[str, bool]:
    """{log dir: its bench recorded the library files it ran with} of
    `log_dirs`; RuntimeError when a recorded file holds other kernels under
    its solution indices in `dat_dir` now (the logs name kernels of another
    hipBLASLt build) or cannot be read. A Prediction table that changed
    alone changes nothing the logs name, except under a record written
    before index digests were recorded."""
    out: Dict[str, bool] = {}
    for d in log_dirs:
        try:
            rec = ki.bench_record(d)
        except ValueError as e:
            raise RuntimeError(f"{d}: {e}") from None
        if rec is None:
            out[d] = False
            continue
        try:
            now = ki.library_identity(dat_dir, list(rec["files"]))
        except (OSError, ValueError) as e:
            raise RuntimeError(
                f"cannot read the library files the bench of {d} recorded in "
                f"{dat_dir}: {e}"
            ) from None
        changed = ki.changed_files(rec, now)
        if changed:
            why = (
                "so its solution indices name other kernels here"
                if rec.get(ki.INDEX_DIGESTS)
                else "and its record, written before index digests were recorded, "
                "cannot tell solution indices that name other kernels here from "
                "a Prediction table that changed alone"
            )
            raise RuntimeError(
                f"the bench of {d} ran with other library files than {dat_dir} "
                f"holds now ({', '.join(changed)}; library digest "
                f"{rec['digest']} -> {now.digest}): the hipBLASLt build changed, "
                f"{why}; enrich its logs with the build that benched them"
            )
        out[d] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="hipblaslt-bench block logs -> chunked enriched CSV folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--log-dir",
        action="append",
        default=[],
        help="directory with block_*.log; repeat to read several",
    )
    ap.add_argument(
        "--context-log-dir",
        nargs=3,
        action="append",
        default=[],
        metavar=("CU_BUDGET", "SCHEDULE", "DIR"),
        help="directory with block_*.log benched under this execution "
        "context (repeatable)",
    )
    ap.add_argument(
        "--sustained-log-dir",
        action="append",
        default=[],
        help="directory with block_*.log of stage03d's sustained finalist "
        "re-measurement (repeatable; one per run); its rows get is_sustained",
    )
    ap.add_argument(
        "--context-plan",
        nargs=3,
        action="append",
        default=[],
        metavar=("CU_BUDGET", "SCHEDULE", "PLAN"),
        help="plan.json of one context's benches, instead of --plan " "(repeatable)",
    )
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
    ap.add_argument(
        "--cu-budget",
        type=int,
        default=0,
        help="CUs the benched kernels could use (0: every CU)",
    )
    ap.add_argument(
        "--schedule",
        choices=SCHEDULES,
        default="default",
        help="tile scheduling the bench requested",
    )
    ap.add_argument(
        "--extra-library-stem",
        action="append",
        default=[],
        help="with --library-stem: also keep rows of this library's solutions "
        "(repeatable)",
    )
    ap.add_argument(
        "--plan",
        default=None,
        help="stage02 plan.json: fill is_planned / n_planned and leave "
        "is_origami_pick unset for its GEMMs",
    )
    ap.add_argument(
        "--config-yaml",
        default=None,
        help="run config: every tested row must name its hipblaslt section's "
        "problem type",
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
    if args.cu_budget < 0:
        ap.error("--cu-budget must be >= 0")
    if not args.log_dir and not args.context_log_dir:
        ap.error("give --log-dir or --context-log-dir")
    try:
        log_sources = [(d, args.cu_budget, args.schedule) for d in args.log_dir]
        log_sources += [_context_arg(v) for v in args.context_log_dir]
        log_sources += [(d, 0, "default") for d in args.sustained_log_dir]
        context_plans = {_context_arg(v)[1:]: v[2] for v in args.context_plan}
    except ValueError as e:
        ap.error(str(e))
    context = {"cu_budget": args.cu_budget, "schedule": args.schedule}

    t0 = time.time()
    if not args.quiet:
        ui.banner("hipblaslt-bench logs  ->  enriched chunked CSV")
        ui.info("cfg", f"log_dir        : {args.log_dir}")
        for budget, schedule, d in args.context_log_dir:
            ui.info("cfg", f"context logs   : {budget}/{schedule} {d}")
        ui.info("cfg", f"out_dir        : {args.out_dir}")
        ui.info("cfg", f"dat_dir        : {args.dat_dir}")
        ui.info("cfg", f"library_stem   : {args.library_stem}")
        if args.extra_library_stem:
            ui.info("cfg", f"extra stems    : {args.extra_library_stem}")
        if args.plan:
            ui.info("cfg", f"plan           : {args.plan}")
        ui.info("cfg", f"scale_mode     : {args.scale_mode}")
        ui.info("cfg", f"rows_per_chunk : {args.rows_per_chunk:,}")
        ui.info("cfg", f"context        : {context}")

    log_files: List[str] = []
    file_context: Dict[str, Tuple[int, str]] = {}
    file_dir: Dict[str, str] = {}
    sustained_dirs = set(args.sustained_log_dir)
    for log_dir, budget, schedule in log_sources:
        if not os.path.isdir(log_dir):
            ui.err("err", f"log directory does not exist: {log_dir}")
            return 1
        for f in sorted(os.listdir(log_dir)):
            if f.endswith(".log") and (f.startswith("block_") or f.startswith("line")):
                path = os.path.join(log_dir, f)
                log_files.append(path)
                file_context[path] = (budget, schedule)
                file_dir[path] = log_dir
    if not log_files:
        dirs = ", ".join(d for d, _, _ in log_sources)
        ui.err("err", f"no block_*.log files in {dirs}")
        return 1
    if args.extra_library_stem and not args.library_stem:
        ui.err("err", "--extra-library-stem needs --library-stem")
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
        for stem in args.extra_library_stem:
            extra = load_kernel_index(args.dat_dir, library_stem=stem)
            index.files += extra.files
            for sid, info in extra.by_index.items():
                if sid in index.by_index:
                    index.duplicate_indices += 1
                else:
                    index.by_index[sid] = info
                    index.file_of[sid] = extra.file_of[sid]
        identity = (
            ki.library_identity(
                args.dat_dir, [args.library_stem, *args.extra_library_stem]
            )
            if args.library_stem
            else None
        )
    except (FileNotFoundError, ValueError) as exc:
        ui.err("err", str(exc))
        return 1
    if not index.by_index:
        ui.err("err", f"no solutions loaded from {args.dat_dir}")
        return 1
    plan: Optional[Dict[ShapeKey, Dict[str, Any]]] = None
    plans_by_context: Dict[Tuple[int, str], Dict[ShapeKey, Dict[str, Any]]] = {}
    try:
        if args.plan:
            plan = _read_plan(args.plan, args.dat_dir)
        for ctx_key, path in context_plans.items():
            plans_by_context[ctx_key] = _read_plan(path, args.dat_dir)
    except ki.BuildChangedError as exc:
        ui.err("identity", f"{exc}; plan the round again with this build")
        return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        ui.err("err", f"cannot read a plan: {exc}")
        return 1
    try:
        strict = _bench_strictness(list(dict.fromkeys(file_dir.values())), args.dat_dir)
    except RuntimeError as exc:
        ui.err("identity", str(exc))
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
    blocklist = None
    if args.config_yaml:
        try:
            enricher.expect = config_types(args.config_yaml)
        except (OSError, ValueError, KeyError) as exc:
            ui.err("err", f"cannot read the problem type of {args.config_yaml}: {exc}")
            return 1
        if not args.quiet:
            ui.info("cfg", f"problem type   : {enricher.expect}")
        try:
            blocklist = config_blocklist(_read_config(args.config_yaml))
        except (OSError, ValueError) as exc:
            ui.err("blocklist", str(exc))
            return 1
        if blocklist:
            enricher.blocked = {
                sid for sid, info in index.by_index.items() if blocklist.blocks(info)
            }
            if not args.quiet:
                ui.info(
                    "blocklist",
                    f"{len(enricher.blocked)} kernel(s) of the library compute "
                    f"wrong results; their rows are dropped",
                )
    writer = ChunkWriter(args.out_dir, args.rows_per_chunk)
    counters: Counter = Counter()
    stats = LogStats()
    columns = _ColumnCheck()
    written: Set[int] = set()
    last_progress = time.time()
    total_kept = 0
    gemms_by_context: Counter = Counter()
    try:
        for i, path in enumerate(log_files, 1):
            fname = os.path.basename(path)
            remaining = (
                None if args.max_gemms is None else max(0, args.max_gemms - total_kept)
            )
            if remaining == 0:
                break
            t_file = time.time()
            budget, schedule = file_context[path]
            enricher.strict = strict[file_dir[path]]
            sustained = file_dir[path] in sustained_dirs
            rows, kept = process_log_file(
                path,
                enricher,
                counters,
                stats,
                max_problems=remaining,
                plan=plans_by_context.get((budget, schedule), plan),
                sustained=sustained,
            )
            for r in rows:
                r.update({"cu_budget": budget, "schedule": schedule})
                writer.write(r)
                columns.add(r)
                written.add(int(r["sol_idx_global"]))
                counters["rows_skip" if r["is_skip"] == "True" else "rows_tested"] += 1
            if sustained:
                counters["sustained_problems"] += kept
                counters["rows_sustained"] += len(rows)
            else:
                gemms_by_context[f"{budget}/{schedule}"] += kept
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
    except TypeMismatch as exc:
        ui.err("types", f"{path}: {exc}")
        return 1
    finally:
        n_chunks, total_rows = writer.close()
    if identity is None:
        try:
            identity = ki.library_identity(
                args.dat_dir,
                sorted({logic_stem(index.file_of[sid]) for sid in written}),
                tolerate_tables=True,
            )
        except (FileNotFoundError, ValueError) as exc:
            ui.err("err", str(exc))
            return 1
    write_kernel_attributes(
        os.path.join(args.out_dir, KERNELS_FILE),
        {sid: index.by_index[sid] for sid in written},
        args.library_stem,
        extra={
            ki.RECORD_KEY: ki.record(identity, written),
            "signatures": {
                str(sid): ki.signature_label(index.by_index[sid])
                for sid in sorted(written)
            },
        },
    )

    summary = {
        "log_files": len(log_files),
        "library_stem": args.library_stem,
        "extra_library_stems": args.extra_library_stem,
        "plan": args.plan,
        "library_files": index.files,
        "invalid_library_files": index.invalid,
        "n_library_solutions": len(index.by_index),
        "missing_size_mapping_keys": index.missing_keys,
        "context": context,
        "gemms_by_context": dict(sorted(gemms_by_context.items())),
        "gemms": total_kept,
        "rows": total_rows,
        "chunks": n_chunks,
        "n_kernels": len(written),
        "skip_lines": stats.skip_lines,
        "unparsed_skip_lines": stats.unparsed_skip_lines,
        "unparsed_skip_examples": stats.unparsed_skip_examples,
        "malformed_tested_rows": stats.malformed_rows,
        "name_mismatches": enricher.name_mismatches,
        "kernels_blocked": len(enricher.blocked),
        "blocklist": applied_record(blocklist),
        "library_digest": identity.digest,
        "bench_records": sum(strict.values()),
        "counters": dict(sorted(counters.items())),
    }
    launch_by_dir = {
        d: launch_settings.read_record(d)
        for d in dict.fromkeys(d for d, _, _ in log_sources)
    }
    summary["launch"] = launch_settings.summarize(launch_by_dir.values())
    write_json(os.path.join(args.out_dir, "enrich_summary.json"), summary)
    if len(summary["launch"]["heuristics"]) > 1:
        ui.warn(
            "launch",
            f"the logs were benched with {len(summary['launch']['heuristics'])} "
            f"launch heuristics: their builds' launch-rule sources differ, so "
            f"their labels may measure different launches: "
            + "; ".join(
                f"{d}: {launch_settings.describe_digests(r)}"
                for d, r in launch_by_dir.items()
            ),
        )

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
    if counters["logs_canary_flagged"]:
        ui.warn(
            "canary",
            f"{counters['problems_canary_flagged']} GEMM(s) of "
            f"{counters['logs_canary_flagged']} log(s) were benched while the "
            f"canary read the host as loaded; their rows have canary_flagged 1",
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
