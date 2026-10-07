# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage07: selection time of the deployed model and C++/Python pick parity.

Runs the hipblaslt-bench of the deploy checkout's build on every bench yaml
(filtered to the config's data types and layout) and checks the runtime
against stage06's deploy record. Report-only: no training or deploy decision
reads its numbers.

Pick parity (one run per yaml, `requested_solution_num: 1`, with
`TILEWRIGHT_DIAG` and `TILEWRIGHT_PICK_LOG` on):
  * every `[TILEWRIGHT_DIAG FILE]` line of the deployed library must name the
    co-located copy of the deployed file, with the recorded cell and split
    counts, feature hash and weight type, and that file must hash to the
    recorded sha256;
  * the engine's top-1 pick (its position in the library's kernel pool, the
    pick log's `top1_index`) and serving cell of every problem are compared
    with `lib.evaluate.top1_picks` on the same model bytes, the library's
    kernel pool read from its .dat and the device hardware; the picked
    kernel's signature is reported but does not decide. Both sides run the
    same engine, so a mismatch points at how hipBLASLt maps its problem,
    kernels, pool order or hardware to tilewright. A pick-log line without
    `top1_index` fails the check;
  * the runtime's selection efficiency: each runtime pick joined to the bench
    timings of its problem (--timings-csv-dir, enriched chunk_*.csv such as
    the run's stage04 data), against the fastest timed kernel; a pick with
    only a skip row is a miss under --pick-measurement timed (lib/evaluate.py),
    a pick without any row is counted apart (the runtime then ran a later
    kernel whose predicates pass).

Order parity (one more run per yaml with `requested_solution_num:
stage07.parity_top_k`, 8 by default): the runtime's top-1 at that request
size against the engine's, and the solutions the runtime returned, in order,
against the engine's ranking of the same request size. The kernels the
engine scored must come first and in the engine's order; the engine's kernels
the runtime left out must be ones it may reject by their predicates (a
kernel the timings show measured for the problem may not be left out), and
the rest of the list (filled by the analytical ranking) must be unscored.

Selection time (separate runs, pick log off): for every request size in
`stage07.timing_request_sizes`, `stage07.timing_repetitions` repetitions of
an ML-on and an ML-off run over the identical problem list, the order
alternating between repetitions. Each problem's `Solution selection time`
is paired across the two runs (median over repetitions per problem). The
report gives the ML-on times and the ML-on minus ML-off deltas as median,
percentiles and maximum, with percentile-bootstrap 95% intervals of the
medians, overall and split into the first call into each serving cell
(which builds that cell's caches) and the warm calls after it. The first
problem of every run, which carries the library load, is reported
separately. ML-on runs must show the deployed model loading; ML-off runs get
no tilewright variables at all. The ML-off times are the analytical
ranking's and are printed only with --origami-reference.

Everything is written below --out-dir. The exit status is non-zero when a
bench fails, the deployed model is not the one the runtime loaded, or pick
or order parity is below `stage07.parity_min_match_rate`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = THIS_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from lib import evaluate as ev  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_log import iter_problems  # noqa: E402
from run_orchestrator import arch_base, cfg_get  # noqa: E402

_RE_SEL_TIME = re.compile(r"Solution selection time:\s*(\S+)\s*us")
_RE_SUPPORTED = re.compile(r"^Is supported\s+(\d+)\s*/\s*Total solutions:\s*(\d+)")
_RE_HEADER = re.compile(r"^\[(\d+)\]:(.*)$")
_RE_DIAG_FILE = re.compile(r"\[TILEWRIGHT_DIAG FILE\]\s*(.*)$")
_RE_DIAG_FAIL = re.compile(r"\[TILEWRIGHT_DIAG FAIL\]\s*(.*)$")
_PICK_TAG = "[TILEWRIGHT_PICK]"
_RE_PICK = re.compile(
    r"\[TILEWRIGHT_PICK\]\s+m=(\d+)\s+n=(\d+)\s+k=(\d+)\s+b=(\d+)\s+"
    r"tA=([NT])\s+tB=([NT])\s+leaf=(\S*)\s+"
    r"top1_sig=\(mt_m=(\d+),mt_n=(\d+),mt_k=(\d+),mi_m=(\d+),mi_n=(\d+),"
    r"mi_k=(\d+),cha=(-?\d+),chb=(-?\d+)\)\s+top1_score=(\S+)\s+"
    r"top1_index=(\d+)\s+n_configs=(\d+)"
)

FILTER_FIELDS = (
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "compute_type",
    "transA",
    "transB",
)
DROP_KEYS = (
    "adaptive",
    "warmup_time",
    "sample_time",
    "measure_time",
    "max_measure_time",
    "min_iters",
    "max_iters",
    "noise_threshold",
    "stability_threshold",
    "stability_window",
    "stability_interval",
    "skip_slow_solution_ratio",
    "print_kernel_info",
    "rotating",
)

Key = Tuple[int, int, int, int, str, str]


# ── bench yaml ──────────────────────────────────────────────────────────────


def _norm_field(name: str, value: Any) -> str:
    s = str(value).strip()
    if name == "compute_type" and s.lower().startswith("c_"):
        s = s[2:]
    return s


def workload_filter(cfg: Mapping[str, Any]) -> Dict[str, str]:
    hbl = cfg.get("hipblaslt") or {}
    wf = {f: _norm_field(f, hbl[f]) for f in FILTER_FIELDS if hbl.get(f) is not None}
    for f in ("scaleA", "scaleB"):
        if hbl.get(f) is not None:
            wf[f] = str(int(hbl[f]))
    return wf


def _entry_matches(entry: Mapping[str, Any], wf: Mapping[str, str]) -> bool:
    for name, want in wf.items():
        if name in ("scaleA", "scaleB"):
            got = str(int(entry.get(name, 0) or 0))
        else:
            got = _norm_field(name, entry.get(name, ""))
        if got != want:
            return False
    return True


def load_bench_entries(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: a bench yaml is a list of problem mappings")
    return [dict(e) for e in data if isinstance(e, dict)]


def normalized_entries(
    entries: Sequence[Mapping[str, Any]],
    requested_solutions: int,
    kernel_info: bool = False,
) -> List[Dict[str, Any]]:
    """Copies set to run one timed iteration of `requested_solutions`
    solutions, without the training-bench timing and pruning knobs; with
    `kernel_info` every result names its solution index."""
    out = []
    for e in entries:
        d = {k: v for k, v in e.items() if k not in DROP_KEYS}
        d["iters"] = 1
        d["cold_iters"] = 0
        d["requested_solution_num"] = int(requested_solutions)
        if kernel_info:
            d["print_kernel_info"] = 1
        out.append(d)
    return out


def write_bench_yaml(entries: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in entries:
            line = yaml.safe_dump(
                dict(e), default_flow_style=True, width=1 << 20, sort_keys=False
            ).strip()
            f.write(f"- {line}\n")


def entry_key(e: Mapping[str, Any]) -> Key:
    return (
        int(e.get("M", e.get("m", 0))),
        int(e.get("N", e.get("n", 0))),
        int(e.get("K", e.get("k", 0))),
        int(e.get("batch_count", 1) or 1),
        str(e.get("transA", "N")),
        str(e.get("transB", "N")),
    )


# ── bench runs ──────────────────────────────────────────────────────────────


def run_bench(
    bench: Path,
    yaml_path: Path,
    log_path: Path,
    device: int,
    env: Dict[str, str],
    *,
    startup_grace_s: float,
    stall_timeout_s: float,
    quiet: bool,
) -> int:
    """Run one bench, output to `log_path`. A bench silent for longer than
    the budget is killed (startup gets its own, longer budget)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(bench), "--yaml", str(yaml_path), "--device", str(device)]
    if not quiet:
        ui.info("bench", " ".join(cmd))
    t0 = time.time()
    with log_path.open("w", buffering=1, errors="replace") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            errors="replace",
            bufsize=1,
            env=env,
            cwd=str(log_path.parent),
        )
        assert proc.stdout is not None
        last = [time.time(), False]

        def _drain() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                last[0], last[1] = time.time(), True

        pump = threading.Thread(target=_drain, daemon=True)
        pump.start()
        while proc.poll() is None:
            time.sleep(0.5)
            budget = stall_timeout_s if last[1] else startup_grace_s
            if budget > 0 and time.time() - last[0] > budget:
                ui.err("bench", f"no output for {budget:.0f}s; killing {bench.name}")
                proc.kill()
                break
        rc = proc.wait()
        pump.join(timeout=30)
    if not quiet:
        (ui.ok if rc == 0 else ui.err)(
            "bench", f"rc={rc} after {ui.fmt_dur(time.time() - t0)}  -> {log_path}"
        )
    return rc


# ── log parsing ─────────────────────────────────────────────────────────────


@dataclass
class Call:
    sel_us: float
    n_lines: int
    supported: Optional[int] = None
    key: Optional[Key] = None


@dataclass
class BenchLog:
    calls: List[Call] = field(default_factory=list)
    diag_files: List[Dict[str, str]] = field(default_factory=list)
    diag_fails: List[str] = field(default_factory=list)
    picks: List[Dict[str, Any]] = field(default_factory=list)
    unparsed_picks: List[str] = field(default_factory=list)
    unattributed_sel_lines: int = 0


def _kv(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def _float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_bench_log(path: Path) -> BenchLog:
    """Selection-time calls (grouped per problem by the bench's `Is
    supported` line, identified by the first result row after it), tilewright
    diagnostics and pick-log lines of one bench log."""
    out = BenchLog()
    pending: List[float] = []
    awaiting: Optional[Tuple[Call, List[str]]] = None
    current: Optional[Call] = None
    if not path.is_file():
        return out
    with path.open(errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if awaiting is not None:
                call, names = awaiting
                awaiting = None
                values = [v.strip() for v in line.strip().split(",")]
                row = dict(zip(names, values))
                try:
                    call.key = (
                        int(row["m"]),
                        int(row["n"]),
                        int(row["k"]),
                        int(row.get("batch_count", "1")),
                        row["transA"],
                        row["transB"],
                    )
                except (KeyError, ValueError):
                    pass
                continue
            m = _RE_SEL_TIME.search(line)
            if m:
                pending.append(_float(m.group(1)))
                continue
            m = _RE_SUPPORTED.match(line.strip())
            if m:
                current = Call(
                    sel_us=float(sum(pending)) if pending else float("nan"),
                    n_lines=len(pending),
                    supported=int(m.group(1)),
                )
                out.calls.append(current)
                pending = []
                continue
            m = _RE_HEADER.match(line.strip())
            if m and current is not None and current.key is None:
                names = [n.strip() for n in m.group(2).split(",")]
                if names and names[0] == "function":
                    names = names[1:]
                awaiting = (current, names)
                continue
            m = _RE_PICK.search(line)
            if m:
                g = m.groups()
                out.picks.append(
                    {
                        "key": (int(g[0]), int(g[1]), int(g[2]), int(g[3]), g[4], g[5]),
                        "cell": g[6],
                        "top1_sig": tuple(int(v) for v in g[7:15]),
                        "top1_score": _float(g[15]),
                        "top1_index": int(g[16]),
                        "n_configs": int(g[17]),
                    }
                )
                continue
            if _PICK_TAG in line:
                out.unparsed_picks.append(line.strip())
                continue
            m = _RE_DIAG_FILE.search(line)
            if m:
                out.diag_files.append(_kv(m.group(1)))
                continue
            m = _RE_DIAG_FAIL.search(line)
            if m:
                out.diag_fails.append(m.group(1))
    out.unattributed_sel_lines = len(pending)
    return out


# ── checks ──────────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_diag(
    log: BenchLog, expected_path: Path, record: Mapping[str, Any]
) -> Tuple[bool, List[str]]:
    """(ok, problems): the deployed model loaded from `expected_path` with
    the recorded contents."""
    problems: List[str] = []
    want = os.path.realpath(expected_path)
    mine = [d for d in log.diag_files if os.path.realpath(d.get("path", "")) == want]
    for msg in log.diag_fails:
        problems.append(f"engine load failure: {msg}")
    if not mine:
        seen = sorted({d.get("path", "?") for d in log.diag_files})
        problems.append(
            f"the runtime never loaded {expected_path}"
            + (f" (it loaded {seen})" if seen else " (no tilewright model loaded)")
        )
        return False, problems
    d = mine[0]
    expect = {
        "n_cells": str(record.get("n_cells")),
        "n_splits": str(record.get("n_splits")),
        "qhash": str(record.get("feature_catalog_hash")),
        "weights": str(record.get("weight_dtype")),
        "arch": str(record.get("arch")),
    }
    for k, v in expect.items():
        if d.get(k) != v:
            problems.append(f"loaded model {k}={d.get(k)} but the deploy recorded {v}")
    return not problems, problems


def compare_picks(
    problems: Sequence[Mapping[str, Any]],
    cpp: Sequence[Mapping[str, Any]],
    py: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Join engine pick-log lines and offline picks by problem. A pick
    matches when both sides pick the same pool position from the same cell.
    The signatures only diagnose: `pick_mismatch_same_sig` counts mismatches
    between kernels of one signature (pool order or duplicate kernels),
    `same_index_other_sig` picks of one position that describe different
    kernels (the pools or the kernel parameters differ)."""
    cpp_by: Dict[Key, Mapping[str, Any]] = {}
    inconsistent = 0
    for p in cpp:
        k = tuple(p["key"])
        if k in cpp_by:
            prev = cpp_by[k]
            if (prev["top1_index"], prev["cell"]) != (p["top1_index"], p["cell"]):
                inconsistent += 1
            continue
        cpp_by[k] = p
    py_by: Dict[Key, Mapping[str, Any]] = {}
    for p in py:
        k = (
            int(p["m"]),
            int(p["n"]),
            int(p["k"]),
            int(p["batch"]),
            str(p["transA"]),
            str(p["transB"]),
        )
        py_by.setdefault(k, p)
    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)
    seen = set()
    for e in problems:
        key = entry_key(e)
        if key in seen:
            continue
        seen.add(key)
        c, p = cpp_by.get(key), py_by.get(key)
        row: Dict[str, Any] = {"m": key[0], "n": key[1], "k": key[2], "batch": key[3]}
        row["transA"], row["transB"] = key[4], key[5]
        py_ok = p is not None and p.get("reason") == "ok"
        if c is not None:
            row["cpp_index"], row["cpp_cell"] = c["top1_index"], c["cell"]
            row["cpp_sig"], row["cpp_score"] = list(c["top1_sig"]), c["top1_score"]
            if p is not None and p.get("n_configs") is not None:
                if int(c["n_configs"]) != int(p["n_configs"]):
                    row["cpp_n_configs"] = c["n_configs"]
                    counts["pool_size_differs"] += 1
        if p is not None:
            row["py_reason"] = p.get("reason")
            row["py_cell"] = p.get("cell")
            if py_ok:
                row["py_index"] = p["top1_index"]
                row["py_sig"], row["py_score"] = list(p["top1_sig"]), p["top1_score"]
        if c is None and not py_ok:
            row["result"] = "agree_unscored"
        elif c is None:
            row["result"] = "no_cpp_pick"
        elif not py_ok:
            row["result"] = "no_py_pick"
        elif int(c["top1_index"]) != int(p["top1_index"]):
            row["result"] = "pick_mismatch"
            if tuple(c["top1_sig"]) == tuple(p["top1_sig"]):
                counts["pick_mismatch_same_sig"] += 1
        else:
            if tuple(c["top1_sig"]) != tuple(p["top1_sig"]):
                counts["same_index_other_sig"] += 1
            row["result"] = "match" if c["cell"] == p.get("cell") else "cell_mismatch"
        counts[row["result"]] += 1
        rows.append(row)
    n = len(rows)
    agree = counts.get("match", 0) + counts.get("agree_unscored", 0)
    pool_sizes = sorted(
        {int(p["n_configs"]) for p in py if p.get("n_configs") is not None}
    )
    return {
        "n_problems": n,
        "n_cpp_picks": len(cpp_by),
        "n_py_picks": sum(1 for p in py_by.values() if p.get("reason") == "ok"),
        "n_agree": agree,
        "match_rate": agree / n if n else None,
        "counts": dict(counts),
        "library_pool_size": pool_sizes[0] if len(pool_sizes) == 1 else pool_sizes,
        "n_inconsistent_cpp_picks": inconsistent,
        "per_problem": rows,
    }


def runtime_lists(path: Path) -> Dict[Key, List[int]]:
    """{problem: the solution indices the runtime returned, in its order} of
    a bench log run with `print_kernel_info`; a repeated problem keeps its
    first list."""
    out: Dict[Key, List[int]] = {}
    if not path.is_file():
        return out
    with path.open(errors="replace") as f:
        for prob in iter_problems(f):
            rows = sorted(prob.tested, key=lambda r: r.rank)
            if not rows:
                continue
            first = rows[0].fields
            try:
                key = (
                    int(first["m"]),
                    int(first["n"]),
                    int(first["k"]),
                    int(first.get("batch_count", "1")),
                    first["transA"],
                    first["transB"],
                )
            except (KeyError, ValueError):
                continue
            out.setdefault(key, [r.sol_index for r in rows if r.sol_index is not None])
    return out


def _subsequence(part: Sequence[int], whole: Sequence[int]) -> bool:
    it = iter(whole)
    return all(x in it for x in part)


def compare_orders(
    problems: Sequence[Mapping[str, Any]],
    runtime: Mapping[Key, Sequence[int]],
    py: Sequence[Mapping[str, Any]],
    position_of: Mapping[int, int],
    measured: Optional[Mapping[Key, Iterable[int]]] = None,
) -> Dict[str, Any]:
    """Join the runtime's returned lists (solution indices) with the engine's
    rankings (`top1_picks(..., with_ranking=True)`, pool positions) by
    problem. `match`: the scored kernels the runtime returned are the
    engine's first ones in its order, followed only by unscored kernels;
    `predicate_skips`: the same, but the runtime left some of the engine's
    kernels out, none of which `measured` (solutions with bench rows per
    problem) shows to pass the predicates; `order_mismatch` otherwise.
    `agree_unscored`: no ranking on either side."""
    py_by: Dict[Key, Mapping[str, Any]] = {}
    for p in py:
        k = (
            int(p["m"]),
            int(p["n"]),
            int(p["k"]),
            int(p["batch"]),
            str(p["transA"]),
            str(p["transB"]),
        )
        py_by.setdefault(k, p)
    sol_at = {pos: sid for sid, pos in position_of.items()}
    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)
    seen = set()
    for e in problems:
        key = entry_key(e)
        if key in seen:
            continue
        seen.add(key)
        p = py_by.get(key) or {}
        ranked = list(p.get("ranked") or []) if p.get("reason") == "ok" else []
        sids = runtime.get(key)
        row: Dict[str, Any] = {"m": key[0], "n": key[1], "k": key[2], "batch": key[3]}
        row["transA"], row["transB"] = key[4], key[5]
        if sids is None:
            row["result"] = "no_runtime_list"
        else:
            got = [position_of.get(int(s), -1) for s in sids]
            row["runtime"], row["engine"] = got, ranked[: max(len(got), 1)]
            scored = set(ranked)
            ours = [x for x in got if x in scored]
            fill_at = next((i for i, x in enumerate(got) if x not in scored), len(got))
            if not ranked:
                row["result"] = "agree_unscored" if not ours else "order_mismatch"
            elif any(x in scored for x in got[fill_at:]):
                row["result"] = "order_mismatch"
            elif ours == ranked[: len(ours)]:
                row["result"] = "match"
            elif ours and _subsequence(ours, ranked):
                left_out = [
                    x for x in ranked[: ranked.index(ours[-1]) + 1] if x not in ours
                ]
                row["left_out"] = left_out
                passes = set((measured or {}).get(key, ()))
                row["result"] = (
                    "order_mismatch"
                    if any(sol_at.get(x) in passes for x in left_out)
                    else "predicate_skips"
                )
            else:
                row["result"] = "order_mismatch"
            row["n_filled"] = len(got) - fill_at
        counts[row["result"]] += 1
        rows.append(row)
    n = len(rows)
    agree = sum(
        counts.get(r, 0) for r in ("match", "predicate_skips", "agree_unscored")
    )
    return {
        "n_problems": n,
        "n_agree": agree,
        "match_rate": agree / n if n else None,
        "counts": dict(counts),
        "per_problem": rows,
    }


def runtime_selection(
    problems: Sequence[Mapping[str, Any]],
    cpp: Sequence[Mapping[str, Any]],
    pool: Sequence[Mapping[str, Any]],
    timings: Mapping[Key, Mapping[str, Any]],
    pick_measurement: str,
    bootstrap: int = 0,
) -> Dict[str, Any]:
    """Selection efficiency of the runtime's top-1 picks (pick-log
    `top1_index`, a pool position) against the bench timings of the same
    problems (`lib.evaluate.load_timings`)."""
    by_key: Dict[Key, Mapping[str, Any]] = {}
    for p in cpp:
        by_key.setdefault(tuple(p["key"]), p)
    ratios: List[float] = []
    counts: Dict[str, int] = defaultdict(int)
    seen = set()
    for e in problems:
        key = entry_key(e)
        if key in seen:
            continue
        seen.add(key)
        pick = by_key.get(key)
        if pick is None:
            counts["n_without_runtime_pick"] += 1
            continue
        g = timings.get(key)
        idx = int(pick["top1_index"])
        if g is None:
            counts["n_without_timings"] += 1
            continue
        if not 0 <= idx < len(pool):
            counts["n_outside_the_pool"] += 1
            continue
        timed, screened = ev.measurements(g)
        runs, cost, miss = ev.pick_cost(
            int(pool[idx]["sol_idx_global"]), timed, screened, None, pick_measurement
        )
        winner = float(g.get("winner_us") or 0)
        if not runs:
            counts["n_top1_without_row"] += 1
            continue
        if cost is None or cost <= 0 or winner <= 0:
            counts["n_without_timings"] += 1
            continue
        ratios.append(winner / cost)
        if miss is not None:
            counts["n_miss"] += 1
    logs = [math.log(r) for r in ratios]
    ci = ev.bootstrap_ci(logs, resamples=bootstrap)
    return {
        "pick_measurement": pick_measurement,
        "n_problems": len(seen),
        "n_evaluated": len(ratios),
        "n_miss": counts.get("n_miss", 0),
        "n_without_runtime_pick": counts.get("n_without_runtime_pick", 0),
        "n_without_timings": counts.get("n_without_timings", 0),
        "n_top1_without_row": counts.get("n_top1_without_row", 0),
        "n_outside_the_pool": counts.get("n_outside_the_pool", 0),
        "sel_eff": ev.geomean(ratios),
        "sel_eff_ci95": None if ci is None else [math.exp(ci[0]), math.exp(ci[1])],
        "percentiles": {
            f"p{q}": percentile(ratios, q) if ratios else None
            for q in (1, 5, 10, 25, 50)
        },
    }


def json_clean(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_clean(v) for v in obj]
    return obj


# ── statistics ──────────────────────────────────────────────────────────────


def percentile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = (len(s) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def describe(xs: Sequence[float]) -> Dict[str, Any]:
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    if not xs:
        return {"n": 0}
    out: Dict[str, Any] = {"n": len(xs), "mean": statistics.fmean(xs)}
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        out[f"p{q}"] = percentile(xs, q)
    out["min"], out["max"] = min(xs), max(xs)
    return out


def _occurrence_keys(calls: Sequence[Call]) -> List[Tuple[Any, ...]]:
    seen: Dict[Any, int] = defaultdict(int)
    out = []
    for pos, c in enumerate(calls):
        base: Any = c.key if c.key is not None else ("#pos", pos)
        out.append((base, seen[base]))
        seen[base] += 1
    return out


def _with_ci(xs: Sequence[float], bootstrap: int) -> Dict[str, Any]:
    out = describe(xs)
    finite = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    out["p50_ci95"] = ev.bootstrap_ci(finite, statistic=np.median, resamples=bootstrap)
    return out


def first_calls_into_cells(calls: Sequence[Call], cell_of: Mapping[Key, str]) -> set:
    """Occurrence keys of the calls that are the first of their run into
    their serving cell."""
    seen: set = set()
    out = set()
    for okey, c in zip(_occurrence_keys(calls), calls):
        cell = cell_of.get(c.key) if c.key is not None else None
        if cell is not None and cell not in seen:
            seen.add(cell)
            out.add(okey)
    return out


def paired_timing(
    reps: Sequence[Tuple[BenchLog, BenchLog]],
    cell_of: Optional[Mapping[Key, str]] = None,
    bootstrap: int = 0,
) -> Dict[str, Any]:
    """Per-problem ML-on vs ML-off selection times over repetitions. The
    first call of every run (library load) is kept out of the pairing. With
    `cell_of` (problem -> serving cell) the problems are also split into the
    first call into each cell of an ML-on run and the warm calls after it;
    `*_ci95` are bootstrap intervals of the medians."""
    on_t: Dict[Any, List[float]] = defaultdict(list)
    off_t: Dict[Any, List[float]] = defaultdict(list)
    delta_t: Dict[Any, List[float]] = defaultdict(list)
    first_in_cell: set = set()
    cold_on: List[float] = []
    cold_off: List[float] = []
    unpaired = 0
    for on, off in reps:
        if on.calls:
            cold_on.append(on.calls[0].sel_us)
        if off.calls:
            cold_off.append(off.calls[0].sel_us)
        on_map = dict(zip(_occurrence_keys(on.calls), on.calls))
        off_map = dict(zip(_occurrence_keys(off.calls), off.calls))
        cold = set()
        if on.calls:
            cold.add(_occurrence_keys(on.calls)[0])
        if off.calls:
            cold.add(_occurrence_keys(off.calls)[0])
        if cell_of:
            first_in_cell |= first_calls_into_cells(on.calls, cell_of)
        keys = (set(on_map) | set(off_map)) - cold
        for k in keys:
            a, b = on_map.get(k), off_map.get(k)
            if (
                a is None
                or b is None
                or not (math.isfinite(a.sel_us) and math.isfinite(b.sel_us))
            ):
                unpaired += 1
                continue
            on_t[k].append(a.sel_us)
            off_t[k].append(b.sel_us)
            delta_t[k].append(a.sel_us - b.sel_us)
    med = {k: statistics.median(v) for k, v in on_t.items()}
    med_off = {k: statistics.median(v) for k, v in off_t.items()}
    med_delta = {k: statistics.median(v) for k, v in delta_t.items()}
    ratios = [med[k] / med_off[k] for k in med if med_off[k] > 0]
    out = {
        "repetitions": len(reps),
        "n_problems_paired": len(med_delta),
        "n_unpaired": unpaired,
        "ml_on_us": _with_ci(list(med.values()), bootstrap),
        "ml_off_us": _with_ci(list(med_off.values()), bootstrap),
        "delta_us": _with_ci(list(med_delta.values()), bootstrap),
        "ratio_on_over_off": _with_ci(ratios, bootstrap),
        "first_call_us": {"ml_on": cold_on, "ml_off": cold_off},
    }
    if cell_of:
        first = [k for k in med if k in first_in_cell]
        warm = [k for k in med if k not in first_in_cell]
        out["first_in_cell"] = {
            "n_problems": len(first),
            "ml_on_us": _with_ci([med[k] for k in first], bootstrap),
            "delta_us": _with_ci([med_delta[k] for k in first], bootstrap),
        }
        out["warm"] = {
            "n_problems": len(warm),
            "ml_on_us": _with_ci([med[k] for k in warm], bootstrap),
            "delta_us": _with_ci([med_delta[k] for k in warm], bootstrap),
        }
    return out


# ── stage ───────────────────────────────────────────────────────────────────


@dataclass
class Context:
    cfg: Dict[str, Any]
    record: Dict[str, Any]
    bench: Path
    library_dir: Path
    stem: str
    weights_path: Path
    device: int
    out_dir: Path
    startup_grace_s: float
    stall_timeout_s: float
    quiet: bool
    failures: List[str] = field(default_factory=list)
    timing_dirs: List[Path] = field(default_factory=list)
    pick_measurement: str = "warmup"
    bootstrap: int = 0
    top_k: int = 0
    origami_reference: bool = False
    cells: Dict[str, Dict[Key, str]] = field(default_factory=dict)
    _pool: Optional[List[Dict[str, Any]]] = None
    _timings: Optional[Dict[Key, Dict[str, Any]]] = None

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        ui.err("check", msg)

    def pool(self) -> List[Dict[str, Any]]:
        """The library's kernel pool, in the order the runtime ranks it."""
        if self._pool is None:
            self._pool = ev.library_pool(
                self.library_dir, self.stem, self.cfg, self.device
            )
        return self._pool

    def timings(self, keys: Iterable[Key]) -> Dict[Key, Dict[str, Any]]:
        """Bench timings of `keys` from --timings-csv-dir (read once)."""
        if self._timings is None:
            self._timings = {}
        want = [k for k in keys if k not in self._timings]
        if want and self.timing_dirs:
            got = ev.load_timings(self.timing_dirs, wanted=want)
            self._timings.update(got)
        return {k: self._timings[k] for k in keys if k in self._timings}


def _env(
    ctx: Context, tilewright: Optional[Dict[str, str]], timing: bool
) -> Dict[str, str]:
    from lib.bench_env import bench_env_with

    extra = {"HIPBLASLT_TENSILE_LIBPATH": str(ctx.library_dir)}
    if timing:
        extra["TENSILE_DB"] = "0x10000"
    return bench_env_with(extra, tilewright=tilewright)


def _bench(
    ctx: Context, yaml_path: Path, log: Path, env: Dict[str, str]
) -> Optional[BenchLog]:
    rc = run_bench(
        ctx.bench,
        yaml_path,
        log,
        ctx.device,
        env,
        startup_grace_s=ctx.startup_grace_s,
        stall_timeout_s=ctx.stall_timeout_s,
        quiet=ctx.quiet,
    )
    if rc != 0:
        ctx.fail(f"hipblaslt-bench failed (rc={rc}); see {log}")
        return None
    return parse_bench_log(log)


def _offline_picks(
    ctx: Context,
    problems: Sequence[Mapping[str, Any]],
    min_scored: int = 1,
    with_ranking: bool = False,
) -> List[Dict[str, Any]]:
    from lib.hardware import device_hardware

    data = Path(ctx.record["staged_model"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != ctx.record["sha256"]:
        raise RuntimeError(
            f"{ctx.record['staged_model']} differs from the deploy record"
        )
    hw = device_hardware(ctx.cfg, ctx.device)
    rows = []
    for e in problems:
        m, n, k, b, ta, tb = entry_key(e)
        row = {"m": m, "n": n, "k": k, "batch_count": b, "transA": ta, "transB": tb}
        row.update(
            {f: e[f] for f in ("a_type", "b_type", "c_type", "d_type", "compute_type")}
        )
        rows.append(row)
    return ev.top1_picks(
        data,
        rows,
        ctx.pool(),
        hardware=hw,
        min_scored=int(min_scored),
        with_ranking=with_ranking,
    )


def serving_cells(
    picks: Sequence[Mapping[str, Any]], key: str = "cell"
) -> Dict[Key, str]:
    """{problem: serving cell} of pick-log lines (`key` "cell") or offline
    picks."""
    out: Dict[Key, str] = {}
    for p in picks:
        if "key" in p:
            k = tuple(p["key"])
        else:
            k = (
                int(p["m"]),
                int(p["n"]),
                int(p["k"]),
                int(p["batch"]),
                str(p["transA"]),
                str(p["transB"]),
            )
        if p.get(key):
            out.setdefault(k, str(p[key]))
    return out


def _parity_env(ctx: Context) -> Dict[str, str]:
    return _env(
        ctx,
        {
            "TENSILE_USE_TILEWRIGHT": "1",
            "TILEWRIGHT_DIAG": "1",
            "TILEWRIGHT_PICK_LOG": "1",
        },
        timing=False,
    )


def _ci_text(
    ci: Optional[Sequence[float]], scale: float = 1.0, fmt: str = ".2f"
) -> str:
    if not ci:
        return ""
    return f" (95% CI {ci[0] * scale:{fmt}} .. {ci[1] * scale:{fmt}})"


def run_parity(
    ctx: Context, tag: str, entries: List[Dict[str, Any]], min_rate: float
) -> Dict[str, Any]:
    out = ctx.out_dir / tag / "parity"
    yaml_path = out / "bench.yaml"
    write_bench_yaml(normalized_entries(entries, 1), yaml_path)
    log = _bench(ctx, yaml_path, out / "bench.log", _parity_env(ctx))
    if log is None:
        return {"ok": False, "error": "bench failed"}
    diag_ok, diag_problems = check_diag(log, ctx.weights_path, ctx.record)
    for p in diag_problems:
        ctx.fail(f"{tag}: {p}")
    try:
        py = _offline_picks(ctx, entries)
    except Exception as e:
        ctx.fail(f"{tag}: offline ranking failed: {e!r}")
        return {"ok": False, "diag_ok": diag_ok, "error": repr(e)}
    result = compare_picks(entries, log.picks, py)
    result["diag_ok"] = diag_ok
    result["min_match_rate"] = min_rate
    result["n_unparsed_pick_lines"] = len(log.unparsed_picks)
    ctx.cells[tag] = {**serving_cells(py), **serving_cells(log.picks)}
    keys = list(dict.fromkeys(entry_key(e) for e in entries))
    sel = runtime_selection(
        entries,
        log.picks,
        ctx.pool(),
        ctx.timings(keys),
        ctx.pick_measurement,
        ctx.bootstrap,
    )
    result["runtime_selection"] = sel
    rate = result["match_rate"]
    result["ok"] = (
        diag_ok and not log.unparsed_picks and rate is not None and rate >= min_rate
    )
    with (out / "parity.json").open("w") as f:
        json.dump(json_clean(result), f, indent=2)
    if log.unparsed_picks:
        ctx.fail(
            f"{tag}: {len(log.unparsed_picks)} {_PICK_TAG} line(s) do not carry "
            f"the fields parity compares (top1_index, the pool position of the "
            f"pick); first: {log.unparsed_picks[0]!r}"
        )
    if result["n_inconsistent_cpp_picks"]:
        ui.warn(
            "parity",
            f"{tag}: the runtime logged different picks for the same problem "
            f"{result['n_inconsistent_cpp_picks']} times; the first one is compared",
        )
    if result["counts"].get("pool_size_differs"):
        ui.warn(
            "parity",
            f"{tag}: the runtime ranked a pool of a different size than the "
            f"library file holds ({result['library_pool_size']}) for "
            f"{result['counts']['pool_size_differs']} problems",
        )
    if result["counts"].get("pick_mismatch_same_sig"):
        ui.warn(
            "parity",
            f"{tag}: {result['counts']['pick_mismatch_same_sig']} mismatched picks "
            f"are kernels of the same signature: the runtime orders its kernel "
            f"pool, or breaks score ties, differently from the library file",
        )
    if result["counts"].get("same_index_other_sig"):
        ui.warn(
            "parity",
            f"{tag}: for {result['counts']['same_index_other_sig']} problems both "
            f"sides pick the same pool position but describe different kernels: "
            f"the runtime's pool or its kernel parameters differ from the library "
            f"file's",
        )
    if rate is None:
        ctx.fail(f"{tag}: no problem to compare")
    elif rate < min_rate:
        ctx.fail(
            f"{tag}: pick parity {rate * 100:.2f}% < {min_rate * 100:.2f}% "
            f"({result['counts']}); see {out / 'parity.json'}"
        )
    else:
        ui.ok(
            "parity",
            f"{tag}: {result['n_agree']}/{result['n_problems']} problems agree",
        )
    if sel["n_evaluated"]:
        ui.ok(
            "parity",
            f"{tag}: the runtime's picks on {sel['n_evaluated']} problems with bench "
            f"timings: {sel['sel_eff'] * 100:.2f}% of the fastest"
            f"{_ci_text(sel['sel_eff_ci95'], 100)}, {sel['n_miss']} misses, "
            f"{sel['n_top1_without_row']} top-1 kernels without a row",
        )
    return {k: v for k, v in result.items() if k != "per_problem"}


def run_order_parity(
    ctx: Context, tag: str, entries: List[Dict[str, Any]], k: int, min_rate: float
) -> Dict[str, Any]:
    """The runtime at `requested_solution_num: k`: its top-1 and the order
    of the solutions it returns, against the engine."""
    out = ctx.out_dir / tag / f"parity_rsn{k}"
    yaml_path = out / "bench.yaml"
    write_bench_yaml(normalized_entries(entries, k, kernel_info=True), yaml_path)
    log_path = out / "bench.log"
    log = _bench(ctx, yaml_path, log_path, _parity_env(ctx))
    if log is None:
        return {"ok": False, "error": "bench failed"}
    diag_ok, diag_problems = check_diag(log, ctx.weights_path, ctx.record)
    for p in diag_problems:
        ctx.fail(f"{tag} rsn={k}: {p}")
    try:
        py = _offline_picks(ctx, entries, min_scored=k, with_ranking=True)
    except Exception as e:
        ctx.fail(f"{tag} rsn={k}: offline ranking failed: {e!r}")
        return {"ok": False, "diag_ok": diag_ok, "error": repr(e)}
    keys = list(dict.fromkeys(entry_key(e) for e in entries))
    measured = {
        key: {int(c["sol_idx_global"]) for c in g["candidates"] if c["us"] > 0}
        for key, g in ctx.timings(keys).items()
    }
    position_of = {int(kk["sol_idx_global"]): i for i, kk in enumerate(ctx.pool())}
    top1 = compare_picks(entries, log.picks, py)
    order = compare_orders(entries, runtime_lists(log_path), py, position_of, measured)
    result = {
        "requested_solution_num": k,
        "diag_ok": diag_ok,
        "min_match_rate": min_rate,
        "top1": top1,
        "order": order,
    }
    rates = {"top-1": top1["match_rate"], "order": order["match_rate"]}
    result["ok"] = diag_ok and all(
        r is not None and r >= min_rate for r in rates.values()
    )
    with (out / "parity.json").open("w") as f:
        json.dump(json_clean(result), f, indent=2)
    for what, rate in rates.items():
        counts = (top1 if what == "top-1" else order)["counts"]
        if rate is None:
            ctx.fail(f"{tag} rsn={k}: no problem to compare ({what})")
        elif rate < min_rate:
            ctx.fail(
                f"{tag} rsn={k}: {what} parity {rate * 100:.2f}% < "
                f"{min_rate * 100:.2f}% ({counts}); see {out / 'parity.json'}"
            )
    if result["ok"]:
        ui.ok(
            "parity",
            f"{tag} rsn={k}: top-1 {top1['n_agree']}/{top1['n_problems']}, order "
            f"{order['n_agree']}/{order['n_problems']} problems agree "
            f"({order['counts'].get('predicate_skips', 0)} with kernels the "
            f"runtime may have rejected)",
        )
    return {
        "requested_solution_num": k,
        "ok": result["ok"],
        "diag_ok": diag_ok,
        "top1": {kk: v for kk, v in top1.items() if kk != "per_problem"},
        "order": {kk: v for kk, v in order.items() if kk != "per_problem"},
    }


def _print_timing(ctx: Context, tag: str, rsn: int, stats: Mapping[str, Any]) -> None:
    on = stats["ml_on_us"]
    ui.ok(
        "timing",
        f"{tag} rsn={rsn}: ML-on selection time per problem: median {on['p50']:.2f} us"
        f"{_ci_text(on.get('p50_ci95'))}, p99 {on['p99']:.2f}, max {on['max']:.2f} "
        f"over {on['n']} problems",
    )
    for part, what in (("first_in_cell", "first call into a cell"), ("warm", "warm")):
        st = (stats.get(part) or {}).get("ml_on_us") or {}
        if st.get("n"):
            ui.ok(
                "timing",
                f"{tag} rsn={rsn}:   {what}: median {st['p50']:.2f} us"
                f"{_ci_text(st.get('p50_ci95'))}, max {st['max']:.2f} over "
                f"{st['n']} problems",
            )
    d = stats["delta_us"]
    if ctx.origami_reference and d.get("n"):
        ui.info(
            "timing",
            f"{tag} rsn={rsn}: [reference] ML-on minus ML-off (the analytical "
            f"ranking) per problem: median {d['p50']:+.2f} us"
            f"{_ci_text(d.get('p50_ci95'))}, p10 {d['p10']:+.2f}, "
            f"p90 {d['p90']:+.2f}, p99 {d['p99']:+.2f}",
        )


def run_timing(
    ctx: Context,
    tag: str,
    entries: List[Dict[str, Any]],
    sizes: Sequence[int],
    reps: int,
    cell_of: Optional[Mapping[Key, str]] = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    on_env = _env(
        ctx, {"TENSILE_USE_TILEWRIGHT": "1", "TILEWRIGHT_DIAG": "1"}, timing=True
    )
    off_env = _env(ctx, None, timing=True)
    for rsn in sizes:
        out = ctx.out_dir / tag / "timing" / f"rsn{rsn}"
        yaml_path = out / "bench.yaml"
        write_bench_yaml(normalized_entries(entries, rsn), yaml_path)
        pairs: List[Tuple[BenchLog, BenchLog]] = []
        for rep in range(reps):
            order = ("on", "off") if rep % 2 == 0 else ("off", "on")
            logs: Dict[str, Optional[BenchLog]] = {}
            for mode in order:
                env = on_env if mode == "on" else off_env
                logs[mode] = _bench(
                    ctx, yaml_path, out / f"rep{rep}_ml_{mode}.log", env
                )
            on, off = logs["on"], logs["off"]
            if on is None or off is None:
                continue
            ok, problems = check_diag(on, ctx.weights_path, ctx.record)
            if not ok:
                for p in problems:
                    ctx.fail(f"{tag} rsn={rsn} rep {rep}: ML-on run: {p}")
                continue
            if off.diag_files:
                ctx.fail(
                    f"{tag} rsn={rsn} rep {rep}: tilewright loaded in the ML-off run"
                )
                continue
            for name, lg in (("ML-on", on), ("ML-off", off)):
                if len(lg.calls) != len(entries):
                    ui.warn(
                        "timing",
                        f"{tag} rsn={rsn} rep {rep}: {name} run has {len(lg.calls)} "
                        f"selections for {len(entries)} problems",
                    )
            pairs.append((on, off))
        stats = paired_timing(pairs, cell_of=cell_of, bootstrap=ctx.bootstrap)
        results[f"rsn{rsn}"] = stats
        if stats["delta_us"].get("n"):
            _print_timing(ctx, tag, rsn, stats)
        else:
            ctx.fail(f"{tag} rsn={rsn}: no paired selection times")
    return results


def yaml_tags(paths: Sequence[Path]) -> List[str]:
    """Output-directory names of the bench yamls: the file stem, with `_2`,
    `_3`, ... appended to repeats so that yamls of the same name keep
    separate results."""
    tags: List[str] = []
    used = set()
    for p in paths:
        tag, i = p.stem, 1
        while tag in used:
            i += 1
            tag = f"{p.stem}_{i}"
        used.add(tag)
        tags.append(tag)
    return tags


def _parse_sizes(text: Optional[str], default: Sequence[int]) -> List[int]:
    if not text:
        return [int(v) for v in default]
    out = [int(v) for v in text.split(",") if v.strip()]
    if not out or min(out) < 1:
        raise ValueError(f"bad request sizes {text!r}")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="stage07: selection time and C++/Python pick parity of the deployed model"
    )
    ap.add_argument("--config-yaml", type=Path, required=True)
    ap.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="hipBLASLt build directory of the deploy checkout",
    )
    ap.add_argument(
        "--deploy-record", type=Path, required=True, help="stage06 deploy_record.json"
    )
    ap.add_argument(
        "--bench-yaml",
        "--yaml",
        type=Path,
        action="append",
        required=True,
        dest="bench_yaml",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--bench-binary", type=Path, default=None)
    ap.add_argument(
        "--library-dir",
        type=Path,
        default=None,
        help="default: <build-dir>/Tensile/library/<arch>",
    )
    ap.add_argument(
        "--request-sizes",
        default=None,
        help="comma list; default stage07.timing_request_sizes",
    )
    ap.add_argument("--repetitions", type=int, default=None)
    ap.add_argument("--parity-min-match-rate", type=float, default=None)
    ap.add_argument(
        "--parity-top-k",
        type=int,
        default=None,
        help="request size of the order-parity run (<= 1: none); default "
        "stage07.parity_top_k",
    )
    ap.add_argument(
        "--timings-csv-dir",
        type=Path,
        action="append",
        default=None,
        help="enriched chunk_*.csv dirs with bench timings of the problems "
        "(repeatable): the runtime's selection efficiency",
    )
    ap.add_argument(
        "--pick-measurement",
        choices=ev.PICK_MEASUREMENTS,
        default=None,
        help="default validate.pick_measurement",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=None,
        help="bootstrap resamples of the intervals; default stage07.bootstrap",
    )
    ap.add_argument(
        "--origami-reference",
        action="store_true",
        help="also print the ML-off (analytical ranking) selection times",
    )
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    with args.config_yaml.open() as f:
        cfg = yaml.safe_load(f) or {}
    with args.deploy_record.open() as f:
        record = json.load(f)
    arch_dir = arch_base(str(cfg.get("arch", "")))
    target = next(
        (t for t in record.get("targets", []) if t.get("dir") == arch_dir), None
    )
    if target is None:
        ui.err("cfg", f"the deploy record has no target for {arch_dir}")
        return 2
    library_dir = (
        args.library_dir or args.build_dir / "Tensile" / "library" / arch_dir
    ).resolve()
    bench = (
        args.bench_binary or args.build_dir / "clients" / "hipblaslt-bench"
    ).resolve()
    sizes = _parse_sizes(
        args.request_sizes, cfg_get(cfg, "stage07.timing_request_sizes")
    )
    reps = int(args.repetitions or cfg_get(cfg, "stage07.timing_repetitions"))
    min_rate = float(
        args.parity_min_match_rate
        if args.parity_min_match_rate is not None
        else cfg_get(cfg, "stage07.parity_min_match_rate")
    )
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = Context(
        cfg=cfg,
        record=record,
        bench=bench,
        library_dir=library_dir,
        stem=str(target["library_stem"]),
        weights_path=library_dir / str(target["weights_file"]),
        device=int(args.device),
        out_dir=out_dir,
        startup_grace_s=float(cfg_get(cfg, "bench.startup_grace_s")),
        stall_timeout_s=float(cfg_get(cfg, "bench.stall_timeout_s")),
        quiet=args.quiet,
        timing_dirs=list(args.timings_csv_dir or []),
        pick_measurement=str(
            args.pick_measurement or cfg_get(cfg, "validate.pick_measurement")
        ),
        bootstrap=int(
            args.bootstrap
            if args.bootstrap is not None
            else cfg_get(cfg, "stage07.bootstrap")
        ),
        top_k=int(
            args.parity_top_k
            if args.parity_top_k is not None
            else cfg_get(cfg, "stage07.parity_top_k")
        ),
        origami_reference=args.origami_reference,
    )
    ui.banner("stage07  selection time + pick parity")
    ui.info("cfg", f"bench        : {bench}")
    ui.info("cfg", f"library      : {library_dir}")
    ui.info("cfg", f"library stem : {ctx.stem}")
    ui.info(
        "cfg",
        f"model        : {ctx.weights_path}  (sha256 {str(record.get('sha256'))[:16]})",
    )
    ui.info(
        "cfg",
        f"timing       : {'off' if args.no_timing else f'rsn {sizes}, {reps} repetitions'}",
    )
    ui.info(
        "cfg",
        "parity       : "
        + (
            "off"
            if args.no_parity
            else f"min match rate {min_rate}, order at rsn {ctx.top_k}"
        ),
    )

    summary: Dict[str, Any] = {
        "deploy_record": str(args.deploy_record.resolve()),
        "model": str(ctx.weights_path),
        "library_stem": ctx.stem,
        "per_yaml": {},
    }
    if not bench.is_file():
        ctx.fail(f"hipblaslt-bench not found: {bench}")
    elif not ctx.weights_path.is_file():
        ctx.fail(
            f"{ctx.weights_path} does not exist: build the hipblaslt-tilewright-models target "
            f"of {args.build_dir}"
        )
    elif sha256_file(ctx.weights_path) != record.get("sha256"):
        ctx.fail(
            f"{ctx.weights_path} is not the deployed model (sha256 differs); build "
            f"the hipblaslt-tilewright-models target of {args.build_dir}"
        )
    else:
        wf = workload_filter(cfg)
        for yp, tag in zip(args.bench_yaml, yaml_tags(args.bench_yaml)):
            try:
                entries = load_bench_entries(yp)
            except (OSError, ValueError, yaml.YAMLError) as e:
                ctx.fail(f"{yp}: {e}")
                continue
            kept = [e for e in entries if _entry_matches(e, wf)]
            ui.banner(
                f"{tag}: {len(kept)} of {len(entries)} problems match {wf}", ui.C.BLUE
            )
            res: Dict[str, Any] = {
                "yaml": str(yp),
                "n_entries": len(entries),
                "n_kept": len(kept),
            }
            if not kept:
                ctx.fail(
                    f"{tag}: no problem matches the config's data types and layout"
                )
            else:
                if not args.no_parity:
                    res["parity"] = run_parity(ctx, tag, kept, min_rate)
                    if ctx.top_k > 1:
                        res["parity_top_k"] = run_order_parity(
                            ctx, tag, kept, ctx.top_k, min_rate
                        )
                if not args.no_timing:
                    cell_of = ctx.cells.get(tag)
                    if cell_of is None:
                        try:
                            cell_of = serving_cells(_offline_picks(ctx, kept))
                        except Exception as e:
                            ui.warn(
                                "timing",
                                f"{tag}: no serving cells ({e!r}); first calls "
                                f"into a cell are not told apart",
                            )
                    res["timing"] = run_timing(ctx, tag, kept, sizes, reps, cell_of)
            summary["per_yaml"][tag] = res
    summary["failures"] = ctx.failures
    summary["ok"] = not ctx.failures
    with (out_dir / "summary.json").open("w") as f:
        json.dump(json_clean(summary), f, indent=2, default=str)
    if ctx.failures:
        ui.err(
            "done",
            f"{len(ctx.failures)} check(s) failed; see {out_dir / 'summary.json'}",
        )
        return 1
    ui.ok("done", f"all checks passed; {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
