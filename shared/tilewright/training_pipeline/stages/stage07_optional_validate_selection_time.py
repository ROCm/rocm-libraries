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

Every ML-on run states TENSILE_TILEWRIGHT_BANK: 1 when the config's kernel
pool has a bank (lib/library_pool.py), so that the runtime ranks the pool
the model was trained on, the library's Prediction table plus the bank, and
0 otherwise, since TensileLite's adapter left without the setting serves the
bank to a model that reads kernel attributes. The summary records it
(`bank_served`).

Every bench line runs one iteration on zero operands (`initialization:
zero`): the stage reads selection times and picks, which no operand value
changes, and the bench generates block-scaled (MX) operands on the host, on
the CPUs --cpus pins the stage to, where values cost far more time to
generate than zeros.

Bench deaths: hipblaslt-bench allocates the scale buffers of block-scaled
(MX) problems smaller than the library reads them, and whether the read
faults depends on what the process has mapped after the buffer, so such a
problem can take one bench process down (an illegal memory access,
CLIENT_FAULT) and run in the next. The screen, pick and order parity, the
Origami reference and the timing bench their problems with
`bench_problems`: a process that dies of that fault is charged to the
problem after the last one it finished (a problem without a solution
finishes too; of entries equal in every argument the bench runs the first
alone), and that problem is left out of the run instead of failing the
stage; the problems after it run in a new process, whose output follows in
the run's log. The run's summary entry names the problems it left out
(`left_out`, with the exit status and log; the timing's per repetition and
mode), and its checks cover the problems it ran: the timing pairs a problem
where both runs have it and keeps the first call of every process out. Any
other bench failure fails the stage: a death on a problem without block
scaling or of another fault (a stall the stall timeout ends, too), output
that does not line up with the yaml, and MAX_IDLE_PROCESSES processes in a
row that die before finishing any problem, which no single problem
explains.

Screen (one run per yaml that holds a block-scaled problem, before its
other runs when it has any; ML on at `requested_solution_num: 1`, without
timing): every problem runs once, and the problems the screen leaves out
are left out of every other run of the yaml (`screen`), so that those run
one list and a problem that takes every process down is charged once. A
yaml whose problems the screen all left out fails the stage.

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
call of every bench process, which carries the library load, is reported
separately. ML-on runs must show the deployed model loading; ML-off runs get
no tilewright variables at all. The ML-off times are the analytical
ranking's and are printed only with --origami-reference. Every bench gets
the config's runtime.env (lib/bench_env.py), which platform.json records,
with how the benches launch the kernels (lib/launch_settings.py).

Production route (one more run per yaml, `requested_solution_num: 1` with
tilewright, `TILEWRIGHT_DIAG`, `TILEWRIGHT_PICK_LOG` and `print_kernel_info`
on, and without TENSILE_PREDICTION_LIB, which every other bench sets):
TensileLite then searches every row of the problem's library files in their
order, as production does, and an Equality row of the problem's size or a
GridBased row ahead of the Prediction row answers before the model is asked.
The summary entry `production_route` counts the problems that printed a
pick-log line (the model served them; `n_tilewright`, `tilewright_share`),
those the model's Prediction row served without one (`n_unscored`: the
model, loaded, scored nothing for them and the analytical ranking filled
the pick) and, for the others, names the row that served them (`served_by`:
its kind, library file and kernel signature, from the solution index and
kernel name the bench prints and the rows of the library's files of the
same data types and operation, lib/library_pool.py `read_type_rows`;
`n_ahead` those a row other than the model's answered); `route.json` has
them per problem, and the summary's `production_route` adds the yamls up. A
problem that did not reach the model, or a runtime that never loaded it
(`model_loaded`, from a `[TILEWRIGHT_DIAG FILE]` line naming the deployed
file), is a warning (`warned`), not a failure, and so is a bench of the
route that fails: training benches set TENSILE_PREDICTION_LIB, so the rows
ahead of the Prediction row are what no other stage sees. --no-route
(`stage07.production_route: false`) leaves the run out.

The rows ahead of the model can hold kernels outside the pool, which the
hazard canary never ran alone: with the canary on (`hazards.enabled`), a
kernel of a hazard class (lib/hazards.py) that such a row holds, and that is
none of the pool's kernels the served blocklist leaves in (by solution index
or name), keeps the route from launching it (summary `hazards.route`). A problem whose size an Equality row answers with
one is left out of the route (each yaml's `skipped`
`production_route_problems` names it with the kernel's signature), and a
row that answers every size with one (any other kind) skips the route, as
do library files whose rows cannot be read.

A yaml given with --held-out-yaml holds problems no round trained on, such
as the orchestrator's sample of the audit set, and its summary entry says so
(`held_out`). With --origami-reference such a yaml with bench timings gets
one more run, `requested_solution_num: 1` with tilewright off and
`print_kernel_info` on: the solution the runtime returns with the analytical
ranking (Origami's pick), joined to the same bench timings as the runtime's
ML-on picks of the pick-parity run and paired with them problem by problem
(`paired_selection`, the paired statistics of lib/evaluate.py `summarize`).
That run sets TENSILE_PREDICTION_LIB, so Origami's pick comes from the
Prediction row, which `origami_rows` names (the kind of the row of each
pick). Only held-out problems are paired: the timings of trained problems
come from planned benches, which time Origami's pick only where a list held
it. Like every Origami number, report-only.

A kernel the hazard canary keeps out (lib/hazards.py) never runs among
others. When the library's pool holds one, the runs in which the analytical
ranking picks kernels (the ML-off timing runs, and with them the timing,
the Origami-reference run, the order-parity run, whose list it fills, and
the production route) are skipped, and when the validated bundle's
blocklist does not leave all of them out, every run is: the summary's
`hazards` and each yaml's `skipped` say which and why.

The stage leases its GPU (lib/gpu_lease.py) and records the platform
(lib/fingerprint.py) in platform.json and in the summary's `platform`: the
GPU's settings, and the CPU affinity the benches inherit, whether it pins
them, the governors, boost, SMT and the load average. --cpus pins the stage
and its benches to a CPU list first. Over each yaml's timing runs it
measures how busy the SMT siblings of pinned CPUs were (`timing_cpu`; null
when nothing is pinned).

Everything is written below --out-dir. The exit status is non-zero when a
bench other than the production route's fails (a death of the bench's
fault on a block-scaled problem leaves the problem out instead; see Bench
deaths), the deployed model is not the one the runtime loaded, or pick or
order parity is below `stage07.parity_min_match_rate`; 2 when the lease or
the platform check refuses the run.
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
from lib import hazards  # noqa: E402
from lib import kfd  # noqa: E402
from lib import library_pool  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_env import PREDICTION_LIB_VAR, TILEWRIGHT_BANK_VAR  # noqa: E402
from lib.bench_log import TestedRow, iter_problems  # noqa: E402
from lib.correctness import (  # noqa: E402
    Blocklist,
    applied_record,
    config_blocklist,
    deployed_blocklist,
    sidecar_blocklist,
    sidecar_name,
)
from lib.kernel_identity import (  # noqa: E402
    kernel_id,
    label_of_name,
    signature_label,
    signature_text,
)
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
    solutions on zero operands, without the training-bench timing and
    pruning knobs; with `kernel_info` every result names its solution index.
    The stage reads selection times and picks, which no operand value
    changes, and zero operands are the cheapest to generate: the bench
    generates block-scaled (MX) operands on the host, on the CPUs the stage
    is pinned to."""
    out = []
    for e in entries:
        d = {k: v for k, v in e.items() if k not in DROP_KEYS}
        d["iters"] = 1
        d["cold_iters"] = 0
        d["initialization"] = "zero"
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
    # Positions in `calls` where a later bench process starts: a log that
    # resumed after a bench death (`bench_problems`) holds several.
    process_starts: List[int] = field(default_factory=list)


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


def _row_names(header: str) -> List[str]:
    """The column names of a result row's header (the text after `[r]:`)."""
    names = [n.strip() for n in header.split(",")]
    return names[1:] if names and names[0] == "function" else names


def _row_key(names: Sequence[str], line: str) -> Optional[Key]:
    """The problem a result row names: `line` its values, `names` the
    columns of its header; None when it names none."""
    row = dict(zip(names, (v.strip() for v in line.strip().split(","))))
    try:
        return (
            int(row["m"]),
            int(row["n"]),
            int(row["k"]),
            int(row.get("batch_count", "1")),
            row["transA"],
            row["transB"],
        )
    except (KeyError, ValueError):
        return None


def parse_bench_log(path: Path) -> BenchLog:
    """Selection-time calls (grouped per problem by the bench's `Is
    supported` line, identified by the first result row after it), tilewright
    diagnostics and pick-log lines of one bench log. In a log that resumed
    after a bench death a process starts at its `hipBLASLt version` banner
    or a `***` line (`process_starts`), and the problem a process died in
    before its result row is no call."""
    from lib.bench_log import ANNOTATION_PREFIX, PROCESS_BANNER

    out = BenchLog()
    pending: List[float] = []
    awaiting: Optional[Tuple[Call, List[str]]] = None
    current: Optional[Call] = None
    fresh = False
    if not path.is_file():
        return out
    with path.open(errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip().startswith((PROCESS_BANNER, ANNOTATION_PREFIX)):
                if current is not None and current.key is None:
                    out.calls.pop()
                    if out.process_starts and out.process_starts[-1] == len(out.calls):
                        out.process_starts.pop()
                out.unattributed_sel_lines += len(pending)
                pending, awaiting, current, fresh = [], None, None, True
                continue
            if awaiting is not None:
                call, names = awaiting
                awaiting = None
                call.key = _row_key(names, line)
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
                if fresh and out.calls:
                    out.process_starts.append(len(out.calls))
                fresh = False
                out.calls.append(current)
                pending = []
                continue
            m = _RE_HEADER.match(line.strip())
            if m and current is not None and current.key is None:
                awaiting = (current, _row_names(m.group(2)))
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
    out.unattributed_sel_lines += len(pending)
    return out


# ── checks ──────────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _loaded_from(log: BenchLog, path: Path) -> List[Dict[str, str]]:
    """The `[TILEWRIGHT_DIAG FILE]` lines of `log` that name `path`."""
    want = os.path.realpath(path)
    return [d for d in log.diag_files if os.path.realpath(d.get("path", "")) == want]


def model_loaded(log: BenchLog, path: Path) -> bool:
    """Whether the runtime loaded the model file `path`, whatever its
    contents: a `[TILEWRIGHT_DIAG FILE]` line names it."""
    return bool(_loaded_from(log, path))


def check_diag(
    log: BenchLog, expected_path: Path, record: Mapping[str, Any]
) -> Tuple[bool, List[str]]:
    """(ok, problems): the deployed model loaded from `expected_path` with
    the recorded contents."""
    problems: List[str] = []
    mine = _loaded_from(log, expected_path)
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


def runtime_rows(path: Path) -> Dict[Key, List[TestedRow]]:
    """{problem: the result rows the runtime returned, in its order, each
    naming its solution index and kernel} of a bench log run with
    `print_kernel_info`; a repeated problem keeps its first rows."""
    out: Dict[Key, List[TestedRow]] = {}
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
            out.setdefault(key, rows)
    return out


def runtime_lists(path: Path) -> Dict[Key, List[int]]:
    """{problem: the solution indices the runtime returned, in its order} of
    a bench log run with `print_kernel_info`; a repeated problem keeps its
    first list."""
    return {
        key: [r.sol_index for r in rows if r.sol_index is not None]
        for key, rows in runtime_rows(path).items()
    }


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


def kernel_label(kernel: Optional[Mapping[str, Any]], name: str = "") -> str:
    """How stage 7 names a kernel in its lines: its signature
    (lib/kernel_identity.py `signature_label`) where a library file gives
    it (`kernel`, lib.dat.kernel_dat_info), else what its solution name
    spells (`label_of_name`); empty without either."""
    if kernel is not None:
        try:
            return signature_label(kernel)
        except (KeyError, TypeError, ValueError):
            name = name or str(kernel.get("kernel_name") or "")
    return label_of_name(name) if name else ""


def _kernel_text(served: Mapping[str, Any]) -> str:
    """The kernel of a `served_by` entry as the lines name it."""
    if served.get("signature"):
        return str(served["signature"])
    return label_of_name(served["kernel"]) if served.get("kernel") else ""


def production_route(
    problems: Sequence[Mapping[str, Any]],
    picks: Sequence[Mapping[str, Any]],
    served: Mapping[Key, Sequence[TestedRow]],
    rows: Mapping[int, Tuple[str, str]],
    kernels: Optional[Mapping[int, Mapping[str, Any]]] = None,
    host: Optional[str] = None,
    loaded: bool = True,
) -> Dict[str, Any]:
    """The problems of a production-route run by what served them: the
    model, when the run printed a pick-log line for the problem (`picks`),
    else the library row of the solution the runtime returned (`served`,
    `runtime_rows`; `rows`, {solution index: (library stem, row kind)}, the
    kind `unknown` for an index of no row read and `none` when the runtime
    returned no solution). The model's row is a Prediction row of the
    library file `host` (of any file without `host`): a problem it served
    without a pick-log line reached the model, which scored nothing for it
    (`n_unscored`), when the runtime `loaded` the model. `served_by` counts
    the problems the model did not serve by row, library file and kernel
    (as the bench names it, with its signature from `kernels`, {solution
    index: lib.dat.kernel_dat_info}), most first; `n_ahead` those a row
    other than the model's answered, the rows searched before it.
    `warned`: a problem did not reach the model (`n_reached` <
    `n_problems`), or the runtime never loaded it."""
    pick_keys = {tuple(p["key"]) for p in picks}
    kernels = kernels or {}
    per_problem: List[Dict[str, Any]] = []
    kinds: Dict[str, int] = defaultdict(int)
    by: Dict[Tuple[Any, ...], int] = defaultdict(int)
    n_unscored = n_ahead = 0
    seen = set()
    for e in problems:
        key = entry_key(e)
        if key in seen:
            continue
        seen.add(key)
        got = list(served.get(key) or [])
        sid = got[0].sol_index if got else None
        kernel = (got[0].kernel_name or got[0].sol_name) if got else ""
        stem, kind = rows.get(sid, (None, "unknown")) if got else (None, "none")
        signature = kernel_label(kernels.get(sid), kernel) if got else ""
        tilewright = key in pick_keys
        model_row = kind == "Prediction" and (host is None or stem == host)
        unscored = not tilewright and loaded and model_row
        per_problem.append(
            {
                "m": key[0],
                "n": key[1],
                "k": key[2],
                "batch": key[3],
                "transA": key[4],
                "transB": key[5],
                "tilewright": tilewright,
                "unscored": unscored,
                "solution_index": sid,
                "kernel": kernel,
                "signature": signature,
                "row": kind,
                "library": stem,
            }
        )
        if tilewright:
            continue
        kinds[kind] += 1
        by[(kind, stem, kernel, signature, sid, unscored)] += 1
        n_unscored += int(unscored)
        n_ahead += int(kind not in ("none", "unknown") and not model_row)
    n = len(per_problem)
    n_tilewright = n - sum(kinds.values())
    served_by = [
        {
            "row": kind,
            "library": stem,
            "kernel": kernel,
            "signature": signature,
            "solution_index": sid,
            "unscored": unscored,
            "n": c,
        }
        for (kind, stem, kernel, signature, sid, unscored), c in sorted(
            by.items(), key=lambda kv: (-kv[1], [str(v) for v in kv[0]])
        )
    ]
    n_reached = n_tilewright + n_unscored
    return {
        "n_problems": n,
        "n_tilewright": n_tilewright,
        "n_unscored": n_unscored,
        "n_reached": n_reached,
        "tilewright_share": n_tilewright / n if n else None,
        "n_ahead": n_ahead,
        "warned": n_reached < n or not loaded,
        "rows": dict(sorted(kinds.items())),
        "served_by": served_by,
        "per_problem": per_problem,
    }


def route_summary(
    per_yaml: Mapping[str, Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """The production-route runs of every bench yaml together: their
    problems, those the model served and its share, those it scored nothing
    for and those that reached it, those rows ahead of it answered, the
    yamls whose route warned (`warned_yamls`) and the problems the yamls
    left out of their routes for kernels of a hazard class (`n_left_out`);
    None without a run."""
    runs = {
        tag: y["production_route"]
        for tag, y in per_yaml.items()
        if isinstance(y.get("production_route"), dict)
        and y["production_route"].get("n_problems") is not None
    }
    if not runs:
        return None
    n = sum(r["n_problems"] for r in runs.values())
    served = sum(r["n_tilewright"] for r in runs.values())
    left_out = sum(
        len((y.get("skipped") or {}).get("production_route_problems") or [])
        for y in per_yaml.values()
    )
    return {
        "n_problems": n,
        "n_tilewright": served,
        "n_unscored": sum(r.get("n_unscored", 0) for r in runs.values()),
        "n_reached": sum(r.get("n_reached", r["n_tilewright"]) for r in runs.values()),
        "n_ahead": sum(r.get("n_ahead", 0) for r in runs.values()),
        "tilewright_share": served / n if n else None,
        "warned_yamls": sorted(tag for tag, r in runs.items() if r.get("warned")),
        "n_left_out": left_out,
    }


def route_text(route: Mapping[str, Any], most: int = 3) -> str:
    """The rows that served the problems that did not reach the model, most
    first, for a warning line: the model's row, which served the problems
    the model scored nothing for, is left out."""
    others = [s for s in route.get("served_by") or [] if not s.get("unscored")]
    parts = []
    for s in others[:most]:
        where = (
            f"the {s['row']} row of {s['library']}"
            if s.get("library")
            else {"none": "no solution", "unknown": "a row of no file read"}.get(
                s["row"], f"a {s['row']} row"
            )
        )
        kernel = _kernel_text(s)
        parts.append(f"{s['n']} by {where}" + (f" ({kernel})" if kernel else ""))
    if len(others) > most:
        parts.append(f"{len(others) - most} more kernel(s)")
    return "; ".join(parts)


def route_status(route: Mapping[str, Any]) -> str:
    """What a production-route run says of the problems the model served,
    those it scored nothing for and those rows ahead of it answered."""
    n, n_tw = route["n_problems"], route["n_tilewright"]
    n_unscored, n_ahead = route.get("n_unscored", 0), route.get("n_ahead", 0)
    if not route.get("warned"):
        if n_tw == n:
            return f"tilewright served all {n} problems"
        return (
            f"every problem reached the model: tilewright served {n_tw} of {n} "
            f"and scored nothing for the other {n_unscored}, which its "
            f"Prediction row served"
        )
    others = route_text(route)
    return (
        f"tilewright served {n_tw} of {n} problems"
        + (f" ({n_tw / n * 100:.2f}%)" if n else "")
        + (
            f", scored nothing for {n_unscored} its Prediction row served"
            if n_unscored
            else ""
        )
        + (f", rows ahead of the Prediction row answered {n_ahead}" if n_ahead else "")
        + (f"; the others: {others}" if others else "")
        + (
            ""
            if route.get("model_loaded", True)
            else "; the runtime never loaded the model"
        )
    )


def unvetted_hazards(
    rows: library_pool.TypeRows,
    pool: Sequence[Mapping[str, Any]],
    host: Optional[str] = None,
    blocked: Optional[Blocklist] = None,
) -> List[Tuple[library_pool.RowEntry, Dict[str, Any]]]:
    """The row entries of `rows` (lib/library_pool.py `read_type_rows`) whose
    kernel is in a hazard class (lib/hazards.py `hazard_classes`) and is
    none of the kernels of `pool` that `blocked` leaves in, by solution
    index or name: the hazard canary runs those alone, no other. The
    Prediction rows of the file `host`, the model's, serve the pool without
    the blocked kernels and are left out. Each with what the summary records
    of it: its row kind, library file, size ((M, N, batch, K) of an Equality
    row, null for a size its key does not hold; null for a row that answers
    every size), kernel signature and classes."""
    vetted = [k for k in pool if not (blocked and blocked.blocks(k))]
    indices = {k.get("sol_idx_global") for k in vetted}
    names = {str(k.get("kernel_name") or "") for k in vetted} - {""}
    out = []
    for entry in rows.entries:
        if entry.kind == "Prediction" and entry.stem == host:
            continue
        kernel = rows.kernels.get(entry.index)
        if (
            kernel is None
            or entry.index in indices
            or str(kernel.get("kernel_name") or "") in names
        ):
            continue
        classes = hazards.hazard_classes(kernel)
        if classes:
            out.append(
                (
                    entry,
                    {
                        "row": entry.kind,
                        "library": entry.stem,
                        "size": (
                            None if entry.answers_every_size() else list(entry.size)
                        ),
                        "kernel": kernel_label(kernel),
                        "classes": list(classes),
                    },
                )
            )
    return out


def route_problems(
    problems: Sequence[Mapping[str, Any]],
    held: Sequence[Tuple[library_pool.RowEntry, Mapping[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(the problems the production route benches, those it leaves out): a
    problem whose size a row of `held` (`unvetted_hazards`) answers is left
    out, once, named with that row's kind, file, kernel and classes."""
    routed: List[Dict[str, Any]] = []
    left_out: Dict[Key, Dict[str, Any]] = {}
    for e in problems:
        key = entry_key(e)
        m, n, k, b = key[:4]
        hit = next((rec for row, rec in held if row.answers(m, n, b, k)), None)
        if hit is None:
            routed.append(dict(e))
        elif key not in left_out:
            left_out[key] = {
                "m": m,
                "n": n,
                "k": k,
                "batch": b,
                "transA": key[4],
                "transB": key[5],
                **{f: hit[f] for f in ("row", "library", "kernel", "classes")},
            }
    return routed, list(left_out.values())


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


def first_calls_into_cells(
    calls: Sequence[Call], cell_of: Mapping[Key, str], starts: Iterable[int] = ()
) -> set:
    """Occurrence keys of the calls that are the first of their bench
    process into their serving cell; a process starts at call 0 and at each
    position of `starts`."""
    seen: set = set()
    out = set()
    fresh = set(starts)
    for pos, (okey, c) in enumerate(zip(_occurrence_keys(calls), calls)):
        if pos in fresh:
            seen = set()
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
    first call of every bench process (library load; a run that resumed
    after a bench death has several, `BenchLog.process_starts`) is kept out
    of the pairing, and a problem pairs only where both runs have it. With
    `cell_of` (problem -> serving cell) the problems are also split into the
    first call into each cell of an ML-on process and the warm calls after
    it; `*_ci95` are bootstrap intervals of the medians."""
    on_t: Dict[Any, List[float]] = defaultdict(list)
    off_t: Dict[Any, List[float]] = defaultdict(list)
    delta_t: Dict[Any, List[float]] = defaultdict(list)
    first_in_cell: set = set()
    cold_on: List[float] = []
    cold_off: List[float] = []
    unpaired = 0
    for on, off in reps:
        on_keys, off_keys = _occurrence_keys(on.calls), _occurrence_keys(off.calls)
        on_map = dict(zip(on_keys, on.calls))
        off_map = dict(zip(off_keys, off.calls))
        cold = set()
        for log, okeys, times in ((on, on_keys, cold_on), (off, off_keys, cold_off)):
            for pos in sorted({0, *log.process_starts}) if log.calls else ():
                times.append(log.calls[pos].sel_us)
                cold.add(okeys[pos])
        if cell_of:
            first_in_cell |= first_calls_into_cells(
                on.calls, cell_of, on.process_starts
            )
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
    skip_analytical: Optional[str] = None
    skip_all: Optional[str] = None
    route_skip: Optional[str] = None
    route_held: List[Tuple[library_pool.RowEntry, Dict[str, Any]]] = field(
        default_factory=list
    )
    cells: Dict[str, Dict[Key, str]] = field(default_factory=dict)
    picks: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    _pool: Optional[List[Dict[str, Any]]] = None
    n_blocked: int = 0
    _identity: Optional[Any] = None
    _identity_error: Optional[str] = None
    _timings: Optional[Dict[Key, Dict[str, Any]]] = None
    _type_rows: Optional[library_pool.TypeRows] = None
    type_rows_error: Optional[str] = None

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

    def type_rows(self) -> library_pool.TypeRows:
        """The rows of the library's files of the stem's data types and
        operation (lib/library_pool.py `read_type_rows`), the rows that can
        serve its problems, and their kernels, read once; empty, with one
        warning (`type_rows_error`), when they cannot be read."""
        if self._type_rows is None:
            try:
                self._type_rows = library_pool.read_type_rows(
                    self.library_dir, self.stem
                )
            except (OSError, ValueError) as e:
                self._type_rows = library_pool.TypeRows()
                self.type_rows_error = str(e)
                ui.warn(
                    "route",
                    f"no rows of the library files in {self.library_dir}: {e}; the "
                    f"rows that served are named unknown",
                )
        return self._type_rows

    def serving_rows(self) -> Dict[int, Tuple[str, str]]:
        """{solution index: (library stem, row kind)} of `type_rows`."""
        return self.type_rows().rows

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
    ctx: Context,
    tilewright: Optional[Dict[str, str]],
    timing: bool,
    prediction_lib: bool = True,
) -> Dict[str, str]:
    from lib.bench_env import bench_env_with, runtime_env

    extra = {**runtime_env(ctx.cfg), "HIPBLASLT_TENSILE_LIBPATH": str(ctx.library_dir)}
    if timing:
        extra["TENSILE_DB"] = "0x10000"
    if tilewright is not None:
        tilewright = {**tilewright, TILEWRIGHT_BANK_VAR: bank_setting(ctx)}
    return bench_env_with(extra, tilewright=tilewright, prediction_lib=prediction_lib)


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
    ctx: Context, yaml_path: Path, log: Path, env: Dict[str, str], fatal: bool = True
) -> Optional[BenchLog]:
    """The parsed log of one bench; None when it fails, which fails the
    stage, or with `fatal` false is warned about."""
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
        msg = f"hipblaslt-bench failed (rc={rc}); see {log}"
        if fatal:
            ctx.fail(msg)
        else:
            ui.warn("bench", f"{msg} (a warning, not a failure)")
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


def _parity_env(ctx: Context, prediction_lib: bool = True) -> Dict[str, str]:
    return _env(
        ctx,
        {
            "TENSILE_USE_TILEWRIGHT": "1",
            "TILEWRIGHT_DIAG": "1",
            "TILEWRIGHT_PICK_LOG": "1",
        },
        timing=False,
        prediction_lib=prediction_lib,
    )


# The bench's own fault on block-scaled (MX) problems: hipblaslt-bench
# allocates their scale buffers smaller than the library reads them, and
# whether the read faults depends on what the process has mapped after the
# buffer, so a problem can take one process down and run in the next.
CLIENT_FAULT = "an illegal memory access was encountered"

# Bench processes in a row that die before finishing any problem: a fault
# that no single problem explains, which fails the stage.
MAX_IDLE_PROCESSES = 5


def block_scaled(entry: Mapping[str, Any]) -> bool:
    """Whether a bench yaml entry scales an operand by blocks (MX): its
    `scaleA` or `scaleB` names a block format."""
    from lib.bench_yaml import scale_mode_code
    from lib.dat import mx_block_size_for_scale_mode

    return any(
        mx_block_size_for_scale_mode(scale_mode_code(entry.get(name, 0)))
        for name in ("scaleA", "scaleB")
    )


def _arguments(entry: Mapping[str, Any]) -> Tuple[Any, ...]:
    """A bench yaml entry as hipblaslt_gentest compares entries: field by
    field, numbers and booleans as one type, lists element by element."""

    def value(v: Any) -> Any:
        if isinstance(v, (bool, int, float)):
            return float(v)
        if isinstance(v, (list, tuple)):
            return tuple(value(x) for x in v)
        return str(v)

    return tuple(sorted((str(k), value(v)) for k, v in entry.items()))


def gentest_firsts(entries: Sequence[Mapping[str, Any]]) -> List[int]:
    """For each entry of a bench yaml, the position of the first entry equal
    to it in every argument, which runs in its place: the bench converts its
    yaml with clients/hipblaslt_gentest.py, which skips an entry whose
    arguments equal an earlier one's."""
    first: Dict[Tuple[Any, ...], int] = {}
    return [first.setdefault(_arguments(e), i) for i, e in enumerate(entries)]


@dataclass
class Progress:
    """What the log of one bench process shows of its problems."""

    finished: int = 0
    keys: List[Optional[Key]] = field(default_factory=list)
    fault: bool = False


def bench_progress(path: Path) -> Progress:
    """The problems one bench process got past (`finished`), the problem of
    each one it started, from its first result row (`keys`; None without
    one), and whether it printed CLIENT_FAULT (`fault`). A problem starts
    with its `Is supported` line, or with the `NO solution found` line the
    bench prints in its place, which also finishes it; the bench got past
    every problem another one follows, and past the last one when it
    reported all of its solutions (lib/bench_log.py `Problem.finished`)."""
    from lib.bench_log import NO_SOLUTION_MARK

    out = Progress()
    if not path.is_file():
        return out
    with path.open(errors="replace") as f:
        lines = [line.strip() for line in f]
    starts = [
        i
        for i, s in enumerate(lines)
        if _RE_SUPPORTED.match(s) or NO_SOLUTION_MARK in s
    ]
    out.fault = any(CLIENT_FAULT in s for s in lines)
    for a, b in zip(starts, starts[1:] + [len(lines)]):
        key = None
        for i in range(a, b - 1):
            m = _RE_HEADER.match(lines[i])
            if m:
                key = _row_key(_row_names(m.group(2)), lines[i + 1])
                break
        out.keys.append(key)
    if starts:
        last = list(iter_problems(lines[starts[-1] :]))
        out.finished = len(starts) - 1 + int(bool(last) and last[0].finished)
    return out


@dataclass
class Benched:
    """What `bench_problems` ran: the entries whose problems the bench ran,
    in their order (`ran`), a record of each problem it left out, the bench
    processes, and whether it ran without failing the stage (`ok`)."""

    ran: List[Dict[str, Any]] = field(default_factory=list)
    left_out: List[Dict[str, Any]] = field(default_factory=list)
    processes: int = 0
    ok: bool = True


def bench_problems(
    ctx: Context,
    tag: str,
    entries: Sequence[Dict[str, Any]],
    requested: int,
    yaml_path: Path,
    log_path: Path,
    env: Dict[str, str],
    kernel_info: bool = False,
) -> Benched:
    """Bench `entries` (`normalized_entries` at `requested` solutions) in
    one process, its yaml `yaml_path` and its log `log_path`. A process that
    dies of the bench's fault (CLIENT_FAULT) on a block-scaled problem is
    charged to the problem after the last one it finished: that problem is
    left out, with the entries equal to it, and the problems after it run
    in a new process, whose yaml is `<log stem>.<n>.yaml` and whose output
    follows in `log_path` after a `***` line naming the problem. Any other
    bench failure fails the stage, as a failing bench does, with `ok` false:
    a death of another fault or on a problem without block scaling, output
    that does not line up with the yaml's problems (a problem of another
    shape, more problems than entries), or MAX_IDLE_PROCESSES processes in
    a row that die before finishing any problem. Each process runs fewer
    problems than the one before it."""
    out = Benched()
    todo = list(range(len(entries)))
    done: set = set()
    idle = 0
    while todo:
        proc = out.processes
        part, yml = log_path, yaml_path
        if proc:
            part = log_path.with_name(f"{log_path.stem}.{proc}{log_path.suffix}")
            yml = log_path.with_name(f"{log_path.stem}.{proc}.yaml")
        run = [entries[i] for i in todo]
        norm = normalized_entries(run, requested, kernel_info)
        write_bench_yaml(norm, yml)
        rc = run_bench(
            ctx.bench,
            yml,
            part,
            ctx.device,
            env,
            startup_grace_s=ctx.startup_grace_s,
            stall_timeout_s=ctx.stall_timeout_s,
            quiet=ctx.quiet,
        )
        out.processes += 1
        progress = bench_progress(part) if rc else Progress()
        if proc:
            with log_path.open("a", errors="replace") as f:
                f.write(part.read_text(errors="replace"))
            part.unlink()
        if rc == 0:
            done.update(todo)
            break
        same = gentest_firsts(norm)
        firsts = [p for p, q in enumerate(same) if p == q]
        lined_up = len(progress.keys) <= len(firsts) and all(
            k is None or k == entry_key(run[firsts[i]])
            for i, k in enumerate(progress.keys)
        )
        at = firsts[progress.finished] if progress.finished < len(firsts) else None
        blamed = run[firsts[-1] if at is None else at]
        if not (lined_up and progress.fault and block_scaled(blamed)):
            ctx.fail(
                f"hipblaslt-bench failed (rc={rc}); see {log_path}"
                + ("" if lined_up else ": its output does not line up with its yaml")
            )
            out.ok = False
            break
        if at is None:
            done.update(todo)
            ui.warn(
                "bench",
                f"{tag}: the bench exited with {rc} after its last problem "
                f"({log_path}); every problem ran",
            )
            break
        finished = set(firsts[: progress.finished])
        m, n, k, b, ta, tb = entry_key(run[at])
        for p, q in enumerate(same):
            if q == at:
                out.left_out.append(
                    {"problem": [m, n, k, b, ta, tb], "rc": rc, "log": str(log_path)}
                )
        done.update(todo[p] for p, q in enumerate(same) if q in finished)
        todo = [todo[p] for p, q in enumerate(same) if q not in finished and q != at]
        with log_path.open("a") as f:
            f.write(
                f"\n*** LEFT OUT M={m} N={n} K={k} batch={b} transA={ta} "
                f"transB={tb} (rc={rc}): the bench died of its fault on a "
                f"block-scaled problem; a new process runs any problems after "
                f"it ***\n"
            )
        ui.warn(
            "bench",
            f"{tag}: the bench died (rc={rc}) on M={m} N={n} K={k} batch={b} "
            f"trans {ta}{tb}, which this run leaves out; see {log_path}",
        )
        idle = idle + 1 if progress.finished == 0 else 0
        if idle >= MAX_IDLE_PROCESSES:
            ctx.fail(
                f"{tag}: {idle} bench processes in a row died before finishing "
                f"any problem: not one problem's fault; see {log_path}"
            )
            out.ok = False
            break
    out.ran = [e for i, e in enumerate(entries) if i in done]
    return out


def later_runs(ctx: Context, args: argparse.Namespace) -> bool:
    """Whether a yaml has runs after the screen: pick parity, or the timing
    or a production route (`--no-route`) unless the hazards skip them."""
    analytical = not (args.no_timing and getattr(args, "no_route", True))
    return not args.no_parity or (analytical and not ctx.skip_analytical)


def screen_problems(
    ctx: Context, tag: str, entries: List[Dict[str, Any]], res: Dict[str, Any]
) -> bool:
    """When a problem of `entries` is block-scaled, run every problem once
    (ML on, `requested_solution_num: 1`, no timing; `bench_problems`) and
    keep in `entries`, in place and in order, the ones the bench ran, so
    that the yaml's other runs share one list without the problems that
    took the screen down. `res["screen"]` records the problems left out.
    False, after failing the stage, when the screen fails or leaves no
    problem."""
    if not any(block_scaled(e) for e in entries):
        return True
    out = ctx.out_dir / tag / "screen"
    env = _env(ctx, {"TENSILE_USE_TILEWRIGHT": "1"}, timing=False)
    got = bench_problems(
        ctx, tag, entries, 1, out / "bench.yaml", out / "bench.log", env
    )
    res["screen"] = {
        "n": len(entries),
        "n_ran": len(got.ran),
        "left_out": got.left_out,
        "runs": got.processes,
    }
    if not got.ok:
        return False
    entries[:] = got.ran
    if not got.ran:
        ctx.fail(f"{tag}: the bench ran none of the problems; see {out}")
        return False
    return True


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
    got = bench_problems(
        ctx, tag, entries, 1, out / "bench.yaml", out / "bench.log", _parity_env(ctx)
    )
    if not got.ok:
        return {"ok": False, "error": "bench failed"}
    entries = got.ran
    log = parse_bench_log(out / "bench.log")
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
    if got.left_out:
        result["left_out"] = got.left_out
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
            f"{text}: the runtime serves a bank the config's pool does not have "
            f"(library.kind, or planner.bank and planner.bank_library_stems), so "
            f"its TensileLite does not read {TILEWRIGHT_BANK_VAR}"
        )
    return f"{text}: the runtime's library is not the one the model was trained on"


def run_order_parity(
    ctx: Context, tag: str, entries: List[Dict[str, Any]], k: int, min_rate: float
) -> Dict[str, Any]:
    """The runtime at `requested_solution_num: k`: its top-1 and the order
    of the solutions it returns, against the engine."""
    out = ctx.out_dir / tag / f"parity_rsn{k}"
    log_path = out / "bench.log"
    got = bench_problems(
        ctx, tag, entries, k, out / "bench.yaml", log_path, _parity_env(ctx), True
    )
    if not got.ok:
        return {"ok": False, "error": "bench failed"}
    entries = got.ran
    log = parse_bench_log(log_path)
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
    if got.left_out:
        result["left_out"] = got.left_out
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
        **({"left_out": got.left_out} if got.left_out else {}),
    }


def run_origami_picks(
    ctx: Context, tag: str, entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """[reference] The solution the runtime returns for each problem with
    tilewright off at `requested_solution_num: 1`, paired with the ML-on
    picks of the yaml's pick-parity run on the bench timings
    (`paired_selection`)."""
    out = ctx.out_dir / tag / "origami"
    log_path = out / "bench.log"
    env = _env(ctx, None, timing=False)
    got = bench_problems(ctx, tag, entries, 1, out / "bench.yaml", log_path, env, True)
    if not got.ok:
        return {"error": "bench failed"}
    entries = got.ran
    left = {"left_out": got.left_out} if got.left_out else {}
    log = parse_bench_log(log_path)
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
    rows = ctx.serving_rows()
    kinds: Dict[str, int] = defaultdict(int)
    for sid in picks.values():
        kinds[rows.get(sid, (None, "unknown"))[1]] += 1
    result["origami_rows"] = dict(sorted(kinds.items()))
    with (out / "origami.json").open("w") as f:
        json.dump(json_clean(dict(result, **left)), f, indent=2)
    p = result["paired"]
    if p["n"]:
        ui.info(
            "parity",
            f"{tag}: [reference] the runtime's picks with tilewright on and off "
            f"(Origami, from {rows_text(result['origami_rows'])} with "
            f"{PREDICTION_LIB_VAR}=1) on the same {p['n']} problems: model "
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
    return {k: v for k, v in dict(result, **left).items() if k != "per_problem"}


def rows_text(kinds: Mapping[str, int]) -> str:
    """`{row kind: problems}` as text, such as `the Prediction row`."""
    if not kinds:
        return "no row"
    if len(kinds) == 1:
        return f"the {next(iter(kinds))} row"
    return "rows " + ", ".join(f"{k} ({n})" for k, n in kinds.items())


def run_production_route(
    ctx: Context, tag: str, entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """The runtime as production runs it: tilewright on, with the pick log
    and the diagnostics, at `requested_solution_num: 1` with
    `print_kernel_info`, and without TENSILE_PREDICTION_LIB, so that
    TensileLite searches every row of the problem's library files in their
    order. Counts the problems the model served (`production_route`), those
    it scored nothing for, and names the row and kernel that served each
    other one; a problem that did not reach the model, a runtime that never
    loaded it and a bench that fails are warned about, not failed."""
    out = ctx.out_dir / tag / "route"
    yaml_path = out / "bench.yaml"
    write_bench_yaml(normalized_entries(entries, 1, kernel_info=True), yaml_path)
    log_path = out / "bench.log"
    env = _parity_env(ctx, prediction_lib=False)
    log = _bench(ctx, yaml_path, log_path, env, fatal=False)
    if log is None:
        return {"error": "bench failed"}
    loaded = model_loaded(log, ctx.weights_path)
    _, problems = check_diag(log, ctx.weights_path, ctx.record)
    rows = ctx.type_rows()
    result = production_route(
        entries,
        log.picks,
        runtime_rows(log_path),
        rows.rows,
        rows.kernels,
        host=ctx.stem,
        loaded=loaded,
    )
    result["prediction_lib"] = False
    result["model_loaded"] = loaded
    if problems:
        result["model_problems"] = problems
    with (out / "route.json").open("w") as f:
        json.dump(json_clean(result), f, indent=2)
    said = f"{tag}: without {PREDICTION_LIB_VAR}, as production runs, "
    if not result["n_problems"]:
        ui.warn("route", f"{tag}: no problem to route")
    elif result["warned"]:
        ui.warn("route", said + route_status(result))
    else:
        ui.ok("route", said + route_status(result))
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
        pairs: List[Tuple[BenchLog, BenchLog]] = []
        left_out: List[Dict[str, Any]] = []
        for rep in range(reps):
            order = ("on", "off") if rep % 2 == 0 else ("off", "on")
            logs: Dict[str, Optional[BenchLog]] = {}
            ran: Dict[str, int] = {}
            for mode in order:
                env = on_env if mode == "on" else off_env
                log_path = out / f"rep{rep}_ml_{mode}.log"
                got = bench_problems(ctx, tag, entries, rsn, yaml_path, log_path, env)
                logs[mode] = parse_bench_log(log_path) if got.ok else None
                ran[f"ML-{mode}"] = len(got.ran)
                left_out += [dict(r, rep=rep, mode=mode) for r in got.left_out]
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
                if len(lg.calls) != ran[name]:
                    ui.warn(
                        "timing",
                        f"{tag} rsn={rsn} rep {rep}: {name} run has {len(lg.calls)} "
                        f"selections for {ran[name]} problems",
                    )
            pairs.append((on, off))
        stats = paired_timing(pairs, cell_of=cell_of, bootstrap=ctx.bootstrap)
        if left_out:
            stats["left_out"] = left_out
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
    ap.add_argument(
        "--no-route",
        action="store_true",
        help=f"no production-route run (tilewright on without {PREDICTION_LIB_VAR})",
    )
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
    ui.info(
        "cfg",
        "route        : "
        + (
            "off"
            if args.no_route
            else f"tilewright on without {PREDICTION_LIB_VAR}, as production runs"
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
        "hazards": hazard_check(ctx),
        "per_yaml": {},
    }
    if not (args.no_route or ctx.skip_analytical or ctx.skip_all):
        summary["hazards"]["route"] = route_hazard_check(ctx)
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
    summary["production_route"] = route_summary(summary["per_yaml"])
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


def hazard_check(ctx: Context) -> Dict[str, Any]:
    """What summary.json records of the kernels of the library's pool (the
    table and the bank) that the hazard canary keeps out (lib/hazards.py
    `quarantined_names`), and the runs `ctx` skips for them: those in which
    the analytical ranking picks kernels while the pool holds one
    (`skip_analytical`), every run while the served blocklist leaves one in
    (`skip_all`)."""
    held = hazards.quarantined_names(ctx.cfg)
    if held:
        try:
            whole = bank_pool.config_pool(
                ctx.cfg,
                ctx.library_dir,
                ctx.stem,
                ctx.device,
                apply_blocklist=False,
            )
            held &= {str(k.get("kernel_name") or "") for k in whole.kernels}
        except (OSError, ValueError, RuntimeError):
            pass
    if not held:
        return {"kernels": 0, "skipped": None}
    served = ctx.served_blocklist()
    left_in = sorted(n for n in held if not (served and served.blocks_name(n)))
    ctx.skip_analytical = (
        f"the library's pool holds {len(held)} kernel(s) the hazard canary keeps "
        f"out, which the analytical ranking may pick"
    )
    if left_in:
        ctx.skip_all = (
            f"{ctx.blocklist_source} leaves in {len(left_in)} kernel(s) the hazard "
            f"canary keeps out (e.g. {label_of_name(left_in[0])}), which the runtime "
            f"would serve; deploy the bundle again to ship them in its blocklist"
        )
    why = ctx.skip_all or ctx.skip_analytical
    ui.warn("hazards", f"skipping {'every run' if left_in else 'runs'}: {why}")
    return {
        "kernels": len(held),
        "left_in": len(left_in),
        "skipped": "every run" if left_in else "analytical",
    }


def route_hazard_check(ctx: Context) -> Optional[Dict[str, Any]]:
    """What summary.json records of the kernels of a hazard class that rows
    of the library's files of its data types and operation hold outside the
    pool (`unvetted_hazards`: the table and the bank without the kernels of
    the served blocklist, by solution index or name), which the hazard
    canary never ran alone, and what the production route leaves out for
    them: a row that answers every size (any kind but
    Equality) skips the route (`route_skip`), an Equality row the problems
    of the sizes it answers (`route_held`). Library files whose rows cannot
    be read skip the route too; None, checking nothing, with the canary off
    (`hazards.enabled`)."""
    if not hazards.Settings.of(ctx.cfg).enabled:
        return None
    rows = ctx.type_rows()
    if ctx.type_rows_error is not None:
        ctx.route_skip = (
            f"the rows of the library's files cannot be read "
            f"({ctx.type_rows_error}): stage 7 cannot tell whether a row ahead of "
            f"the model holds a kernel of a hazard class the canary never ran alone"
        )
        ui.warn("hazards", f"skipping the production route: {ctx.route_skip}")
        return {"kernels": [], "skipped": "route"}
    try:
        pool = bank_pool.config_pool(
            ctx.cfg, ctx.library_dir, ctx.stem, ctx.device, apply_blocklist=False
        ).kernels
    except (OSError, ValueError, RuntimeError):
        pool = []
    held = unvetted_hazards(rows, pool, host=ctx.stem, blocked=ctx.served_blocklist())
    if not held:
        return {"kernels": [], "skipped": None}
    every = [(entry, rec) for entry, rec in held if entry.answers_every_size()]
    if every:
        entry, rec = every[0]
        ctx.route_skip = (
            f"the {entry.kind} row of {entry.stem} answers problems of every size "
            f"with {rec['kernel']} ({hazards.class_text(rec['classes'])}), a "
            f"kernel of a hazard class outside the pool, which the hazard canary "
            f"never ran alone"
        )
        ui.warn("hazards", f"skipping the production route: {ctx.route_skip}")
    else:
        ctx.route_held = held
        entry, rec = held[0]
        ui.warn(
            "hazards",
            f"the production route leaves out the problems of the sizes "
            f"{len(held)} Equality row entr{'y' if len(held) == 1 else 'ies'} "
            f"answer with a kernel of a hazard class outside the pool, which the "
            f"hazard canary never ran alone (e.g. {rec['kernel']} of {entry.stem}: "
            f"{hazards.class_text(rec['classes'])})",
        )
    return {
        "kernels": [rec for _, rec in held],
        "skipped": "route" if every else "sizes",
    }


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
    mine = {n for n, k in by_name.items() if invocation and invocation.blocks(k)}
    theirs = {n for n, k in by_name.items() if bundle.blocks(k)}
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


def _route(
    ctx: Context,
    tag: str,
    kept: List[Dict[str, Any]],
    res: Dict[str, Any],
    skipped: Dict[str, Any],
) -> None:
    """The production route of a yaml's problems into `res`, or why it is
    skipped, in full or for some problems (`route_hazard_check`), into
    `skipped`."""
    if ctx.skip_analytical or ctx.route_skip:
        skipped["production_route"] = ctx.skip_analytical or ctx.route_skip
        return
    routed, left_out = route_problems(kept, ctx.route_held)
    if left_out:
        skipped["production_route_problems"] = left_out
        first = left_out[0]
        ui.warn(
            "route",
            f"{tag}: {len(left_out)} problem(s) left out of the production route: "
            f"an Equality row answers their size with a kernel of a hazard class "
            f"the canary never ran alone (e.g. {first['m']}x{first['n']}x"
            f"{first['k']} batch {first['batch']}: {first['kernel']} of "
            f"{first['library']})",
        )
    if routed:
        res["production_route"] = run_production_route(ctx, tag, routed)
    else:
        skipped["production_route"] = (
            "Equality rows answer every problem's size with a kernel of a hazard "
            "class the canary never ran alone"
        )


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
            skipped: Dict[str, Any] = {}
            if not kept:
                ctx.fail(
                    f"{tag}: no problem matches the config's data types and layout"
                )
            elif ctx.skip_all:
                skipped["every run"] = ctx.skip_all
            elif not later_runs(ctx, args) or screen_problems(ctx, tag, kept, res):
                if not args.no_parity:
                    res["parity"] = run_parity(ctx, tag, kept, min_rate)
                    if (
                        held_out
                        and ctx.origami_reference
                        and ctx.timing_dirs
                        and tag in ctx.picks
                    ):
                        if ctx.skip_analytical:
                            skipped["origami_reference"] = ctx.skip_analytical
                        else:
                            res["origami_reference"] = run_origami_picks(ctx, tag, kept)
                    if ctx.top_k > 1:
                        if ctx.skip_analytical:
                            skipped["parity_top_k"] = ctx.skip_analytical
                        else:
                            res["parity_top_k"] = run_order_parity(
                                ctx, tag, kept, ctx.top_k, min_rate
                            )
                if not args.no_route:
                    _route(ctx, tag, kept, res, skipped)
                if not args.no_timing and ctx.skip_analytical:
                    skipped["timing"] = ctx.skip_analytical
                elif not args.no_timing:
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
            if skipped:
                res["skipped"] = skipped
            summary["per_yaml"][tag] = res


if __name__ == "__main__":
    sys.exit(main())
