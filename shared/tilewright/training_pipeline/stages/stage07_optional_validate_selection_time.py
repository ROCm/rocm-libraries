# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage07: selection time of the deployed model and C++/Python pick parity.

Runs the hipblaslt-bench of a hipBLASLt build (--build-dir; the deploy
checkout's, with the model co-located in its library, or any build whose
TensileLite serves the model's format, with --library-dir naming a library
directory that holds the model, such as the orchestrator's
validate/stage07_lib) on every bench yaml (filtered to the config's data
types and layout) and checks the runtime against stage06's deploy record.
Report-only: no training or deploy decision reads its numbers.

Every ML-on run states TENSILE_TILEWRIGHT_BANK: 1 when the config's planner
has a bank (lib/bank_pool.py), so that the runtime ranks the pool the model
was trained on, the library's Prediction table plus the bank, and 0
otherwise, since TensileLite's adapter left without the setting serves the
bank to a model that reads kernel attributes. The summary records it
(`bank_served`).

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
    `top1_index` fails the check, and so does a runtime that ranks a pool of
    another size than the model's (`n_configs`), naming both sizes: a bank
    the runtime does not serve, or one the config does not have;
  * the pool leaves out the kernels of the validated bundle's own blocklist,
    the sidecar its deploy record names (lib/correctness.py
    `deployed_blocklist`; the config's for a record from before sidecars),
    as the runtime does with the sidecar next to the model: a library
    directory that serves another sidecar fails, and an invocation whose
    blocklist names other kernels of the pool is warned about (summary.json
    `blocklist`); the timings below leave the same kernels out;
  * the runtime's selection efficiency: each runtime pick joined to the bench
    timings of its problem (--timings-csv-dir, enriched chunk_*.csv such as
    the run's stage04 data), against the fastest timed kernel; a pick with
    only a skip row is a miss under --pick-measurement timed (lib/evaluate.py),
    a pick without any row is counted apart (the runtime then ran a later
    kernel whose predicates pass). Timings enriched with another hipBLASLt
    build are read with their solution indices mapped to this library's by
    the kernel ids their kernels.json records (lib/kernel_identity.py).

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
ranking's and are printed only with --origami-reference. Every bench gets
the config's runtime.env (lib/bench_env.py), which platform.json records,
with how the benches launch the kernels (lib/launch_settings.py).

A yaml given with --held-out-yaml holds problems no round trained on, such
as the orchestrator's sample of the audit set, and its summary entry says so
(`held_out`). With --origami-reference such a yaml with bench timings gets
one more run, `requested_solution_num: 1` with tilewright off and
`print_kernel_info` on: the solution the runtime returns with the analytical
ranking (Origami's pick), joined to the same bench timings as the runtime's
ML-on picks of the pick-parity run and paired with them problem by problem
(`paired_selection`, the paired statistics of lib/evaluate.py `summarize`).
Only held-out problems are paired: the timings of trained problems come from
planned benches, which time Origami's pick only where a list held it. Like
every Origami number, report-only.

The stage leases its GPU (lib/gpu_lease.py) and records the platform
(lib/fingerprint.py) in platform.json and in the summary's `platform`: the
GPU's settings, and the CPU affinity the benches inherit, whether it pins
them, the governors, boost, SMT and the load average. --cpus pins the stage
and its benches to a CPU list first. Over each yaml's timing runs it
measures how busy the SMT siblings of pinned CPUs were (`timing_cpu`; null
when nothing is pinned).

Everything is written below --out-dir. The exit status is non-zero when a
bench fails, the deployed model is not the one the runtime loaded, or pick
or order parity is below `stage07.parity_min_match_rate`; 2 when the lease
or the platform check refuses the run.
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

from lib import bank_pool  # noqa: E402
from lib import evaluate as ev  # noqa: E402
from lib import fingerprint  # noqa: E402
from lib import gpu_lease  # noqa: E402
from lib import kfd  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_env import TILEWRIGHT_BANK_VAR  # noqa: E402
from lib.bench_log import iter_problems  # noqa: E402
from lib.correctness import (  # noqa: E402
    Blocklist,
    applied_record,
    config_blocklist,
    deployed_blocklist,
    sidecar_blocklist,
    sidecar_name,
)
from lib.kernel_identity import kernel_id, signature_text  # noqa: E402
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
    bench_device, env = kfd.one_gpu_launch(device, env)
    cmd = [str(bench), "--yaml", str(yaml_path), "--device", bench_device]
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


def paired_selection(
    problems: Sequence[Mapping[str, Any]],
    cpp: Sequence[Mapping[str, Any]],
    origami: Mapping[Key, int],
    pool: Sequence[Mapping[str, Any]],
    timings: Mapping[Key, Mapping[str, Any]],
    pick_measurement: str,
    bootstrap: int = 0,
) -> Dict[str, Any]:
    """[reference] The runtime's ML-on top-1 picks (pick-log `top1_index`, a
    pool position) and its picks with tilewright off (`origami`: problem ->
    solution index), each costed against the bench timings of its problem as
    `runtime_selection` costs a pick, and paired problem by problem by
    lib/evaluate.py `summarize` (`paired`: the problems where both picks have
    a cost). `per_problem` names both picks by signature."""
    by_key: Dict[Key, Mapping[str, Any]] = {}
    for p in cpp:
        by_key.setdefault(tuple(p["key"]), p)
    position_of = {int(k["sol_idx_global"]): i for i, k in enumerate(pool)}

    def signature(sid: Optional[int]) -> Optional[str]:
        at = position_of.get(sid) if sid is not None else None
        return None if at is None else signature_text(pool[at])

    results: List[ev.GemmResult] = []
    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)
    seen = set()
    for e in problems:
        key = entry_key(e)
        if key in seen:
            continue
        seen.add(key)
        g, off, pick = timings.get(key), origami.get(key), by_key.get(key)
        if g is None:
            counts["n_without_timings"] += 1
            continue
        if off is None:
            counts["n_without_origami_pick"] += 1
            continue
        timed, screened = ev.measurements(g)
        on: Optional[int] = None
        if pick is not None and 0 <= int(pick["top1_index"]) < len(pool):
            on = int(pool[int(pick["top1_index"])]["sol_idx_global"])
        on_runs, on_us, miss = (
            ev.pick_cost(on, timed, screened, None, pick_measurement)
            if on is not None
            else (False, None, None)
        )
        off_runs, off_us, off_miss = ev.pick_cost(
            int(off), timed, screened, None, pick_measurement
        )
        if not off_runs:
            counts["n_origami_without_row"] += 1
        elif off_miss is not None:
            counts["n_origami_miss"] += 1
        cell = str(pick["cell"]) if pick is not None else ""
        r = ev.GemmResult(
            index=len(results),
            m=key[0],
            n=key[1],
            k=key[2],
            batch=key[3],
            leaf=cell,
            model_cell=cell or None,
            served_by="model" if pick is not None else None,
            winner_us=float(g.get("winner_us") or 0),
            pick_sol_idx=on,
            pick_us=on_us if on_runs else None,
            origami_sol_idx=int(off),
            origami_us=off_us if off_runs else None,
            miss=miss,
        )
        results.append(r)
        rows.append(
            {
                "m": key[0],
                "n": key[1],
                "k": key[2],
                "batch": key[3],
                "transA": key[4],
                "transB": key[5],
                "winner_us": r.winner_us,
                "model_signature": signature(on),
                "model_us": r.pick_us,
                "model_sel_eff": r.sel_eff,
                "origami_signature": signature(int(off)),
                "origami_us": r.origami_us,
                "origami_sel_eff": r.origami_sel_eff,
            }
        )
    summ = ev.summarize(results, bootstrap=bootstrap)
    return {
        "pick_measurement": pick_measurement,
        "n_problems": len(seen),
        "n_without_timings": counts["n_without_timings"],
        "n_without_origami_pick": counts["n_without_origami_pick"],
        "n_origami_without_row": counts["n_origami_without_row"],
        "n_origami_miss": counts["n_origami_miss"],
        "n_model_evaluated": summ["n_evaluated"],
        "n_origami": summ["n_origami"],
        "paired": summ["paired"],
        "per_problem": rows,
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
    serve_bank: bool = False
    blocklist: Optional[Any] = None
    blocklist_source: str = "the config's blocklist"
    cells: Dict[str, Dict[Key, str]] = field(default_factory=dict)
    picks: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    _pool: Optional[List[Dict[str, Any]]] = None
    n_blocked: int = 0
    _identity: Optional[Any] = None
    _identity_error: Optional[str] = None
    _timings: Optional[Dict[Key, Dict[str, Any]]] = None

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        ui.err("check", msg)

    def served_blocklist(self) -> Optional[Any]:
        """The kernels the runtime leaves out: those of the validated bundle's
        own blocklist (`blocklist`, from its deploy record), else of the
        config's (a deploy record from before blocklist sidecars)."""
        if self.blocklist is not None:
            return self.blocklist
        return config_blocklist(self.cfg)

    def pool(self) -> List[Dict[str, Any]]:
        """The library's kernel pool, in the order the runtime ranks it,
        without the kernels of `served_blocklist`."""
        if self._pool is None:
            whole = bank_pool.config_pool(
                self.cfg,
                self.library_dir,
                self.stem,
                self.device,
                apply_blocklist=False,
            )
            served = bank_pool.without_blocked(whole, self.served_blocklist())
            self.n_blocked = len(whole.kernels) - len(served.kernels)
            self._pool = ev.pool_from_kernels(served.kernels)
        return self._pool

    def identity(self) -> Optional[Any]:
        """The kernel identity of the library's files
        (lib/kernel_identity.py); None, with one warning, when they cannot
        be read: timing CSVs are then read by their solution indices."""
        if self._identity is None and self._identity_error is None:
            try:
                self._identity = ev.library_identity(
                    self.library_dir, self.stem, self.cfg, self.device
                )
            except (OSError, ValueError, RuntimeError) as e:
                self._identity_error = str(e)
                ui.warn(
                    "identity",
                    f"no kernel identity of {self.library_dir}: {e}; timing CSVs "
                    f"are read by their solution indices",
                )
        return self._identity

    def timings(self, keys: Iterable[Key]) -> Dict[Key, Dict[str, Any]]:
        """Bench timings of `keys` from --timings-csv-dir (read once), their
        solution indices this library's."""
        if self._timings is None:
            self._timings = {}
        want = [k for k in keys if k not in self._timings]
        if want and self.timing_dirs:
            got = ev.load_timings(
                self.timing_dirs,
                wanted=want,
                identity=self.identity(),
                blocklist=self.served_blocklist(),
            )
            self._timings.update(got)
        return {k: self._timings[k] for k in keys if k in self._timings}


def _env(
    ctx: Context, tilewright: Optional[Dict[str, str]], timing: bool
) -> Dict[str, str]:
    from lib.bench_env import bench_env_with, runtime_env

    extra = {**runtime_env(ctx.cfg), "HIPBLASLT_TENSILE_LIBPATH": str(ctx.library_dir)}
    if timing:
        extra["TENSILE_DB"] = "0x10000"
    if tilewright is not None:
        tilewright = {**tilewright, TILEWRIGHT_BANK_VAR: bank_setting(ctx)}
    return bench_env_with(extra, tilewright=tilewright)


def bank_setting(ctx: Context) -> str:
    """TILEWRIGHT_BANK_VAR of every ML-on run."""
    return "1" if ctx.serve_bank else "0"


def _recorded_env(ctx: Context) -> Dict[str, Optional[str]]:
    """What platform.json records of the benches' environment."""
    from lib.bench_env import recorded_env

    return recorded_env(ctx.cfg, _env(ctx, None, False))


def _launch_record(ctx: Context) -> Dict[str, Any]:
    """What platform.json records of how the benches launch the kernels."""
    from lib import launch_settings

    return launch_settings.bench_record(
        ctx.bench, ctx.cfg, _env(ctx, None, False), os.environ
    )


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
    ctx.picks[tag] = log.picks
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
    runtime_sizes = sorted({int(p["n_configs"]) for p in log.picks})
    result["runtime_pool_sizes"] = runtime_sizes
    other_pool = [n for n in runtime_sizes if n != len(ctx.pool())]
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
        diag_ok
        and not log.unparsed_picks
        and not other_pool
        and rate is not None
        and rate >= min_rate
    )
    with (out / "parity.json").open("w") as f:
        json.dump(json_clean(result), f, indent=2)
    if other_pool:
        ctx.fail(f"{tag}: {pool_size_problem(other_pool, ctx)}")
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


def pool_size_problem(runtime_sizes: Sequence[int], ctx: Context) -> str:
    """Why the runtime ranked `runtime_sizes` kernels for a model whose pool
    (`ctx.pool()`) holds another number."""
    have = len(ctx.pool())
    sizes = "/".join(str(n) for n in runtime_sizes)
    less = (
        f", without the {ctx.n_blocked} kernel(s) of {ctx.blocklist_source}"
        if ctx.n_blocked
        else ""
    )
    text = (
        f"the runtime ranks {sizes} kernels with {TILEWRIGHT_BANK_VAR}="
        f"{bank_setting(ctx)}, the model's pool has {have} (the library's "
        f"Prediction table" + (" and the bank" if ctx.serve_bank else "") + less + ")"
    )
    if ctx.serve_bank and min(runtime_sizes) < have:
        return (
            f"{text}: the bank is not served; the build's TensileLite must rank "
            f"the Equality kernels of the problem type with {TILEWRIGHT_BANK_VAR}=1"
        )
    if not ctx.serve_bank and max(runtime_sizes) > have:
        return (
            f"{text}: the runtime serves a bank the config does not have "
            f"(planner.bank, planner.bank_library_stems), so its TensileLite does "
            f"not read {TILEWRIGHT_BANK_VAR}"
        )
    return f"{text}: the runtime's library is not the one the model was trained on"


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


def run_origami_picks(
    ctx: Context, tag: str, entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """[reference] The solution the runtime returns for each problem with
    tilewright off at `requested_solution_num: 1`, paired with the ML-on
    picks of the yaml's pick-parity run on the bench timings
    (`paired_selection`)."""
    out = ctx.out_dir / tag / "origami"
    yaml_path = out / "bench.yaml"
    write_bench_yaml(normalized_entries(entries, 1, kernel_info=True), yaml_path)
    log_path = out / "bench.log"
    log = _bench(ctx, yaml_path, log_path, _env(ctx, None, timing=False))
    if log is None:
        return {"error": "bench failed"}
    if log.diag_files:
        ctx.fail(f"{tag}: tilewright loaded in the run with tilewright off")
        return {"error": "tilewright loaded"}
    picks = {key: sids[0] for key, sids in runtime_lists(log_path).items() if sids}
    keys = list(dict.fromkeys(entry_key(e) for e in entries))
    result = paired_selection(
        entries,
        ctx.picks.get(tag, []),
        picks,
        ctx.pool(),
        ctx.timings(keys),
        ctx.pick_measurement,
        ctx.bootstrap,
    )
    with (out / "origami.json").open("w") as f:
        json.dump(json_clean(result), f, indent=2)
    p = result["paired"]
    if p["n"]:
        ui.info(
            "parity",
            f"{tag}: [reference] the runtime's picks with tilewright on and off "
            f"(Origami) on the same {p['n']} problems: model "
            f"{p['model_sel_eff'] * 100:.2f}%, Origami "
            f"{p['origami_sel_eff'] * 100:.2f}% ({p['wins']}/{p['ties']}/"
            f"{p['losses']} wins/ties/losses)",
        )
    else:
        ui.warn(
            "parity",
            f"{tag}: [reference] no problem has bench timings of both the "
            f"runtime's pick and Origami's ({out / 'origami.json'})",
        )
    return {k: v for k, v in result.items() if k != "per_problem"}


def _print_timing(ctx: Context, tag: str, rsn: int, stats: Mapping[str, Any]) -> None:
    on = stats["ml_on_us"]
    ui.ok(
        "timing",
        f"{tag} rsn={rsn}: ML-on selection time per problem: min {on['min']:.2f}, "
        f"mean {on['mean']:.2f}, median {on['p50']:.2f} us"
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
        help="hipBLASLt build directory: the deploy checkout's, or with "
        "--library-dir any build whose TensileLite serves the model's format",
    )
    ap.add_argument(
        "--deploy-record", type=Path, required=True, help="stage06 deploy_record.json"
    )
    ap.add_argument(
        "--bench-yaml",
        "--yaml",
        type=Path,
        action="append",
        default=[],
        dest="bench_yaml",
    )
    ap.add_argument(
        "--held-out-yaml",
        type=Path,
        action="append",
        default=[],
        help="a bench yaml of problems no round trained on, such as a sample "
        "of the audit set (repeatable); checked like --bench-yaml, marked "
        "held_out in the summary and, with --origami-reference, paired with "
        "Origami's pick",
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
    ap.add_argument(
        "--cpus",
        default=None,
        metavar="LIST",
        help="pin the stage and its benches to these CPUs (a list such as 4,5 "
        "or 8-11) before the platform is recorded",
    )
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if not args.bench_yaml and not args.held_out_yaml:
        ap.error("give at least one --bench-yaml or --held-out-yaml")
    if args.cpus is not None:
        cpus = fingerprint.parse_cpulist(args.cpus)
        if not cpus:
            ap.error(f"--cpus {args.cpus!r} is not a CPU list")
        try:
            os.sched_setaffinity(0, cpus)
        except OSError as e:
            ui.err("cfg", f"cannot pin stage07 to CPUs {args.cpus}: {e}")
            return 1

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
        return 1
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
        serve_bank=bank_pool.bank_source(cfg).enabled,
    )
    try:
        bundle_blocklist = deployed_blocklist(record, ctx.stem)
        invocation_blocklist = config_blocklist(cfg)
    except ValueError as e:
        ui.err("blocklist", str(e))
        return 1
    if bundle_blocklist is not None:
        ctx.blocklist = bundle_blocklist
        ctx.blocklist_source = "the bundle's blocklist sidecar"
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
        "bank         : "
        + ("served" if ctx.serve_bank else "none in the config, not served")
        + f" ({TILEWRIGHT_BANK_VAR}={bank_setting(ctx)} on ML-on runs)",
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

    blocklists = blocklist_check(ctx, bundle_blocklist, invocation_blocklist)
    summary: Dict[str, Any] = {
        "deploy_record": str(args.deploy_record.resolve()),
        "model": str(ctx.weights_path),
        "library_stem": ctx.stem,
        "bank_served": ctx.serve_bank,
        "origami_reference": ctx.origami_reference,
        "blocklist": blocklists,
        "per_yaml": {},
    }
    identity = ctx.identity()
    summary["library_digest"] = identity.digest if identity is not None else None
    refused = None
    try:
        with gpu_lease.hold([ctx.device], "stage07 selection time", out_dir):
            summary["platform"] = fingerprint.start_session(
                [ctx.device],
                out_dir,
                cpu=True,
                environment=_recorded_env(ctx),
                launch=_launch_record(ctx),
            )
            _run_checks(ctx, args, sizes, reps, min_rate, summary)
    except gpu_lease.LeaseError as e:
        refused = str(e)
    except fingerprint.PlatformError as e:
        refused = f"refused: {e}"
    if refused:
        ctx.fail(refused)
    summary["failures"] = ctx.failures
    summary["ok"] = not ctx.failures
    with (out_dir / "summary.json").open("w") as f:
        json.dump(json_clean(summary), f, indent=2, default=str)
    if ctx.failures:
        ui.err(
            "done",
            f"{len(ctx.failures)} check(s) failed; see {out_dir / 'summary.json'}",
        )
        return 2 if refused else 1
    ui.ok("done", f"all checks passed; {out_dir / 'summary.json'}")
    return 0


def blocklist_check(
    ctx: Context, bundle: Optional[Any], invocation: Optional[Any]
) -> Dict[str, Any]:
    """What summary.json records of the blocklists: the validated bundle's
    (`bundle`, from its deploy record; None for a record from before
    blocklist sidecars, when the invocation's applies), the one the runtime's
    library directory serves next to the model, and the invocation's
    (`invocation`), each by digest and kernel count. A runtime that serves
    another sidecar than the bundle's fails; an invocation whose blocklist
    names other kernels of the pool than the bundle's is warned about: stage
    7 validates the bundle as the runtime serves it."""
    side = ctx.library_dir / sidecar_name(ctx.stem)
    try:
        served = sidecar_blocklist(side) if side.is_file() else Blocklist()
    except ValueError as e:
        ctx.fail(str(e))
        served = None
    out = {
        "bundle": None if bundle is None else applied_record(bundle),
        "served": None if served is None else applied_record(served),
        "invocation": applied_record(invocation),
    }
    if bundle is None:
        return out
    if served is not None and served.names != bundle.names:
        ctx.fail(
            f"{side} names {len(served)} kernel(s), the deployed bundle's "
            f"blocklist sidecar {len(bundle)}: the runtime leaves out other "
            f"kernels than the bundle was deployed with; deploy it again, or "
            f"remove a sidecar a former deploy left"
        )
    try:
        by_name = {
            str(k["kernel_name"]): k
            for k in bank_pool.config_pool(
                ctx.cfg, ctx.library_dir, ctx.stem, ctx.device, apply_blocklist=False
            ).kernels
        }
    except (OSError, ValueError, RuntimeError):
        return out
    mine = (invocation.names if invocation else frozenset()) & set(by_name)
    theirs = bundle.names & set(by_name)
    if mine != theirs:
        served_wrong = sorted(mine - theirs)
        ui.warn(
            "blocklist",
            f"this invocation's blocklist names {len(mine)} kernel(s) of the "
            f"pool, the bundle's sidecar {len(theirs)}: stage 7 validates the "
            f"bundle as the runtime serves it, with its sidecar"
            + (
                f"; the runtime still offers {len(served_wrong)} kernel(s) this "
                f"invocation's blocklist says compute wrong results (e.g. "
                f"{kernel_id(by_name[served_wrong[0]])}): deploy the bundle "
                f"again to ship this blocklist"
                if served_wrong
                else ""
            ),
        )
    return out


def _run_checks(
    ctx: Context,
    args: argparse.Namespace,
    sizes: Sequence[int],
    reps: int,
    min_rate: float,
    summary: Dict[str, Any],
) -> None:
    """The parity and timing runs of every bench yaml, into `summary`. The
    SMT siblings outside the CPU affinity, when it pins the process, are
    watched through each yaml's timing runs (`timing_cpu`)."""
    cpu = (summary.get("platform") or {}).get("cpu") or {}
    siblings = fingerprint.parse_cpulist(cpu.get("smt_siblings_outside_affinity"))
    remedy = (
        f"the library directory {ctx.library_dir} must hold the staged model"
        if args.library_dir is not None
        else f"build the hipblaslt-tilewright-models target of {args.build_dir}"
    )
    if not ctx.bench.is_file():
        ctx.fail(f"hipblaslt-bench not found: {ctx.bench}")
    elif not ctx.weights_path.is_file():
        ctx.fail(f"{ctx.weights_path} does not exist: {remedy}")
    elif sha256_file(ctx.weights_path) != ctx.record.get("sha256"):
        ctx.fail(
            f"{ctx.weights_path} is not the deployed model (sha256 differs); "
            f"{remedy}"
        )
    else:
        wf = workload_filter(ctx.cfg)
        yamls = [(p, False) for p in args.bench_yaml]
        yamls += [(p, True) for p in args.held_out_yaml]
        for (yp, held_out), tag in zip(yamls, yaml_tags([p for p, _ in yamls])):
            try:
                entries = load_bench_entries(yp)
            except (OSError, ValueError, yaml.YAMLError) as e:
                ctx.fail(f"{yp}: {e}")
                continue
            kept = [e for e in entries if _entry_matches(e, wf)]
            ui.banner(
                f"{tag}: {len(kept)} of {len(entries)} problems match {wf}"
                + (" (held out of training)" if held_out else ""),
                ui.C.BLUE,
            )
            res: Dict[str, Any] = {
                "yaml": str(yp),
                "held_out": held_out,
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
                    if (
                        held_out
                        and ctx.origami_reference
                        and ctx.timing_dirs
                        and tag in ctx.picks
                    ):
                        res["origami_reference"] = run_origami_picks(ctx, tag, kept)
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
                    window = fingerprint.CpuWindow(siblings)
                    res["timing"] = run_timing(ctx, tag, kept, sizes, reps, cell_of)
                    res["timing_cpu"] = window.stop()
            summary["per_yaml"][tag] = res


if __name__ == "__main__":
    sys.exit(main())
