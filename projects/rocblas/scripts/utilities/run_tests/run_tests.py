#!/usr/bin/env python3

# Copyright (C) 2016-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
run_tests.py — Parallel rocBLAS test runner

Replaces run_separate_tests.sh with concurrent execution, persistent state,
and live progress display.  All dependencies are Python 3.8+ stdlib only.

Usage examples
--------------
# Run everything (resume automatically if state file exists):
    python3 run_tests.py

# Run with 10 parallel jobs, custom executable, custom output dir:
    python3 run_tests.py -j 10 -e /opt/rocm/bin/rocblas-test -o /tmp/results

# Re-run from scratch (ignore previous state):
    python3 run_tests.py --reset

# Run only the L2_BLAS group:
    python3 run_tests.py --group L2_BLAS

# Run two specific jobs:
    python3 run_tests.py --job L1_BLAS.dot --job L2_BLAS.gemv_batched

# Skip tests matching a GTest pattern (repeatable):
    python3 run_tests.py --exclude-pattern "*f32_c_LNN*"
    python3 run_tests.py --exclude-pattern "*f32_c_LNN*" --exclude-pattern "*f64_c_LTN*"

# List all valid job IDs:
    python3 run_tests.py --list-jobs

# Plain output (no ANSI, safe for tee / CI logs):
    python3 run_tests.py --no-color

Resume behaviour
----------------
Interrupted runs (SSH drop, OOM kill, Ctrl+C) are resumed automatically.
  - Jobs whose recorded PID is still alive are waited on (reattached).
  - Jobs whose recorded PID is dead are reset to "not_started" and re-run;
    the old log is preserved as a .prev archive.
  - On re-run, only tests that did not pass in the previous log are
    re-executed (partial resume).  Passed tests are appended to the GTest
    filter negative section so they are skipped.  This applies to both
    interrupted jobs and completed-but-failed jobs.
  - With --skip-failed, previously failed tests are also skipped (only
    tests that have not been attempted at all are re-run).
  - --skip-failed never skips whole jobs; it only affects individual test
    selection within the partial resume filter.
  - Use --reset to force a completely clean start.

Caveats
-------
- PID reuse: os.kill(pid, 0) may hit a recycled PID on resume.  Use --reset
  if you suspect PID recycling.
- Two concurrent script instances in the same output dir will corrupt state.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from subprocess import Popen, STDOUT, run as _subprocess_run
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# JOB DEFINITIONS
# ---------------------------------------------------------------------------

@dataclass
class JobSpec:
    job_id: str        # e.g. "L2_BLAS.gemv_batched"
    group_id: str      # "AUXILIARY" | "L1_BLAS" | "L1_BLAS_EX" | "L2_BLAS"
    gtest_filter: str  # verbatim --gtest_filter value
    log_file: str      # absolute path, set after output_dir is known


def build_all_groups(output_dir: str) -> Dict[str, List[JobSpec]]:
    """Return ordered dict: group_id -> list of JobSpec.

    Filter strings replicate run_separate_tests.sh exactly.
    """
    groups: Dict[str, List[JobSpec]] = {}

    def make(group_id: str, variant: str, gtest_filter: str) -> JobSpec:
        job_id = f"{group_id}.{variant}"
        log_file = os.path.join(output_dir, f"{job_id}.txt")
        return JobSpec(job_id=job_id, group_id=group_id,
                       gtest_filter=gtest_filter, log_file=log_file)

    # -- AUXILIARY ----------------------------------------------------------
    aux_tests = [
        "half_operators", "complex_operators", "helper_utilities",
        "check_numerics_vector", "check_numerics_matrix",
        "check_numerics_matrix_batched", "set_get_pointer_mode",
        "set_get_atomics_mode", "logging", "set_get_vector",
        "set_get_vector_async", "set_get_matrix", "set_get_matrix_async",
    ]
    groups["AUXILIARY"] = [
        make("AUXILIARY", t, f"*{t}*quick*") for t in aux_tests
    ]

    # -- L1_BLAS ------------------------------------------------------------
    # Extra negative clauses for functions whose name is a prefix of another.
    # Without these, e.g. *rot* matches rotg/rotm/rotmg, *rotm* matches rotmg.
    l1_extra_neg: Dict[str, str] = {
        "dot":  "*dotc*",
        "rot":  "*rotg*:*rotm*",
        "rotm": "*rotmg*",
    }
    l1_functions = [
        "asum", "axpy", "copy", "dot", "dotc",
        "iamax", "iamin", "nrm2", "rot", "rotg",
        "rotm", "rotmg", "scal", "swap",
    ]
    jobs: List[JobSpec] = []
    for fn in l1_functions:
        xneg = (":" + l1_extra_neg[fn]) if fn in l1_extra_neg else ""
        jobs.append(make("L1_BLAS", fn,
                         f"*{fn}*quick*-*_batched*:*_ex*{xneg}"))
        jobs.append(make("L1_BLAS", f"{fn}_batched",
                         f"*{fn}_batched*quick*-*_ex*"))
        jobs.append(make("L1_BLAS", f"{fn}_strided_batched",
                         f"*{fn}_strided_batched*quick*-*_ex*"))
    groups["L1_BLAS"] = jobs

    # -- L1_BLAS_EX ---------------------------------------------------------
    l1_ex_functions = ["axpy", "dot", "dotc", "nrm2", "rot", "scal"]
    jobs = []
    for fn in l1_ex_functions:
        jobs.append(make("L1_BLAS_EX", f"{fn}_ex",
                         f"*{fn}_ex*quick*-*_batched*"))
        jobs.append(make("L1_BLAS_EX", f"{fn}_batched_ex",
                         f"*{fn}_batched_ex*quick*"))
        jobs.append(make("L1_BLAS_EX", f"{fn}_strided_batched_ex",
                         f"*{fn}_strided_batched_ex*quick*"))
    groups["L1_BLAS_EX"] = jobs

    # -- L2_BLAS ------------------------------------------------------------
    # Extra negative clauses for prefix-ambiguous L2 function names.
    l2_extra_neg: Dict[str, str] = {
        "her":  "*her2*",
        "hpr":  "*hpr2*",
        "spr":  "*spr2*",
        "syr":  "*syr2*",
        "ger":  "*geru*:*gerc*",
        "trsv": "*tbsv*",
        "trmv": "*tpmv*:*tbmv*",
    }
    l2_functions = [
        "trsv", "gbmv", "gemv", "hbmv", "hemv",
        "her", "her2", "hpmv", "hpr", "hpr2",
        "trmv", "tpmv", "tbmv", "tbsv", "ger",
        "geru", "gerc", "spr", "spr2", "syr",
        "syr2", "sbmv", "spmv", "symv",
    ]
    jobs = []
    for fn in l2_functions:
        xneg = (":" + l2_extra_neg[fn]) if fn in l2_extra_neg else ""
        jobs.append(make("L2_BLAS", fn,
                         f"*{fn}*quick*-*_batched*:*_ex*{xneg}"))
        jobs.append(make("L2_BLAS", f"{fn}_batched",
                         f"*{fn}_batched*quick*-*_ex*"))
        jobs.append(make("L2_BLAS", f"{fn}_strided_batched",
                         f"*{fn}_strided_batched*quick*-*_ex*"))
    groups["L2_BLAS"] = jobs

    return groups


# ---------------------------------------------------------------------------
# STATE I/O
# ---------------------------------------------------------------------------

@dataclass
class JobRecord:
    status: str     # "not_started" | "running" | "finished"
    result: str     # "unknown" | "pass" | "fail" | "incomplete"
    pid: Optional[int]
    start_time: Optional[float]
    end_time: Optional[float]
    exit_code: Optional[int]
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0           # total tests in this job (sticky max, set once)
    tests_passed_offset: int = 0   # cumulative passed from all prior interrupted runs
    tests_failed_offset: int = 0   # cumulative failed from all prior interrupted runs
    current_test: str = ""         # last [ RUN      ] name seen; runtime-only, not persisted
    excluded_tests: List[str] = field(default_factory=list)  # cumulative GTest exclusion set
    prev_log: Optional[str] = None  # path to archived log from most recent interrupted run


@dataclass
class RunState:
    version: int
    executable: str
    output_dir: str
    max_parallel: int
    records: Dict[str, JobRecord] = field(default_factory=dict)


_STATE_VERSION = 1


def _record_from_dict(d: dict) -> JobRecord:
    return JobRecord(
        status=d.get("status", "not_started"),
        result=d.get("result", "unknown"),
        pid=d.get("pid"),
        start_time=d.get("start_time"),
        end_time=d.get("end_time"),
        exit_code=d.get("exit_code"),
        tests_passed=d.get("tests_passed", 0),
        tests_failed=d.get("tests_failed", 0),
        tests_total=d.get("tests_total", 0),
        tests_passed_offset=d.get("tests_passed_offset", 0),
        tests_failed_offset=d.get("tests_failed_offset", 0),
        excluded_tests=d.get("excluded_tests", []),
        prev_log=d.get("prev_log"),
    )


def load_state(state_path: str) -> Optional[RunState]:
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r") as f:
            raw = json.load(f)
        records = {k: _record_from_dict(v) for k, v in raw.get("records", {}).items()}
        return RunState(
            version=raw.get("version", _STATE_VERSION),
            executable=raw.get("executable", ""),
            output_dir=raw.get("output_dir", ""),
            max_parallel=raw.get("max_parallel", 4),
            records=records,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"[warn] Could not parse state file {state_path}: {exc}", file=sys.stderr)
        return None


def save_state(state: RunState, state_path: str, lock: threading.Lock) -> None:
    tmp_path = state_path + ".tmp"
    with lock:
        raw = {
            "version": state.version,
            "executable": state.executable,
            "output_dir": state.output_dir,
            "max_parallel": state.max_parallel,
            "records": {
                k: {
                    "status": v.status,
                    "result": v.result,
                    "pid": v.pid,
                    "start_time": v.start_time,
                    "end_time": v.end_time,
                    "exit_code": v.exit_code,
                    "tests_passed": v.tests_passed,
                    "tests_failed": v.tests_failed,
                    "tests_total": v.tests_total,
                    "tests_failed_offset": v.tests_failed_offset,
                    "tests_passed_offset": v.tests_passed_offset,
                    "excluded_tests": v.excluded_tests,
                    "prev_log": v.prev_log,
                }
                for k, v in state.records.items()
            },
        }
        try:
            with open(tmp_path, "w") as f:
                json.dump(raw, f, indent=2)
            os.replace(tmp_path, state_path)
        except OSError as exc:
            print(f"[warn] Could not save state: {exc}", file=sys.stderr)


def _archive_log(log_file: str) -> Optional[str]:
    """Rename log_file to a sequenced .prev path and return the new path.

    Sequence: log.txt -> log.txt.prev -> log.txt.prev.1 -> log.txt.prev.2 ...
    Returns the archive path, or None if log_file does not exist.
    """
    if not os.path.exists(log_file):
        return None
    candidate = log_file + ".prev"
    if not os.path.exists(candidate):
        try:
            os.rename(log_file, candidate)
            return candidate
        except OSError:
            return None
    # Find next free sequence number
    n = 1
    while os.path.exists(f"{candidate}.{n}"):
        n += 1
    try:
        os.rename(log_file, f"{candidate}.{n}")
        return f"{candidate}.{n}"
    except OSError:
        return None


def _parse_completed_tests(log_file: str) -> tuple:
    """Parse a GTest log and return (passed_set, failed_set) of test names.

    Looks for lines produced by GTEST_LISTENER=PASS_LINE_IN_LOG:
      [       OK ] Suite/TestName (X ms)     -> passed
      [  FAILED  ] Suite/TestName (X ms)     -> failed (test-level, has timing suffix)

    Suite-level lines like '[  FAILED  ] 5 tests.' are skipped (no '(' in them).
    """
    passed: set = set()
    failed: set = set()
    try:
        with open(log_file, "r", errors="replace") as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("[       OK ]"):
                    # "[       OK ] Suite/Test (N ms)"
                    rest = line[len("[       OK ]"):].strip()
                    name = rest.split(" (")[0].strip()
                    if name:
                        passed.add(name)
                elif line.startswith("[  FAILED  ]") and " (" in line:
                    # Test-level failure: "[  FAILED  ] Suite/Test (N ms)"
                    rest = line[len("[  FAILED  ]"):].strip()
                    name = rest.split(" (")[0].strip()
                    if name:
                        failed.add(name)
    except OSError:
        pass
    return passed, failed


def _build_exclusion_filter(original_filter: str, excluded: set) -> Optional[str]:
    """Return an augmented GTest filter that skips the given test names.

    GTest filter format: positive[-negative]
    Negative patterns are colon-separated.  Appends fully-qualified test names
    as exact patterns to the negative section.

    Returns None if excluded is empty (caller should use original_filter as-is).
    """
    if not excluded:
        return None

    dash_pos = original_filter.find("-")
    if dash_pos == -1:
        positive = original_filter
        negative = ""
    else:
        positive = original_filter[:dash_pos]
        negative = original_filter[dash_pos + 1:]

    extra = ":".join(sorted(excluded))
    new_negative = f"{negative}:{extra}" if negative else extra
    return f"{positive}-{new_negative}"


_SUITE_LINE_RE = re.compile(r"^\S+\.$")

# Inline filter length threshold: use --gtest_list_tests enumeration when the
# filter string would exceed this many bytes.  ARG_MAX is ~128 KiB, but
# GTEST_FILTER env var is also not supported by all GTest builds, so instead we
# enumerate remaining tests with --gtest_list_tests and build a compact
# positive-only filter from those names (typically a few KB).
_FILTER_LIST_THRESHOLD = 32768


def _list_remaining_tests(
    executable: str,
    original_filter: str,
    excluded: set,
    exclude_patterns: List[str],
) -> Optional[List[str]]:
    """Return the list of test names that will actually run after filtering.

    Runs ``executable --gtest_filter=original_filter --gtest_list_tests`` to
    enumerate all test names matched by the *original* (pre-exclusion) filter,
    then removes names in ``excluded`` and names matching any ``exclude_patterns``
    glob via :func:`fnmatch.fnmatch`.

    Returns None if the subprocess fails (caller falls back to inline filter).

    Note on parameterized tests: ``--gtest_list_tests`` annotates each test name
    with ``  # GetParam() = ...`` describing the parameter values.  These
    annotations are NOT part of the actual test name and must be stripped before
    constructing the filter.  No timeout is set because slow targets (e.g. the
    FFM simulator) can take several minutes to initialize before listing tests.
    """
    try:
        result = _subprocess_run(
            [executable, f"--gtest_filter={original_filter}", "--gtest_list_tests"],
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    suite = ""
    remaining: List[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.rstrip()
        if _SUITE_LINE_RE.match(line):
            suite = line  # e.g. "blas2_tensile/quick_trsv."
        elif line.startswith("  ") and suite:
            # Strip the "  # GetParam() = ..." annotation that GTest appends to
            # parameterized test entries; keep only the bare test name.
            stripped = line.strip()
            hash_pos = stripped.find("#")
            if hash_pos != -1:
                stripped = stripped[:hash_pos].rstrip()
            if not stripped:
                continue
            test_name = suite + stripped  # fully-qualified: "Suite.TestName"
            if test_name in excluded:
                continue
            if any(fnmatch.fnmatch(test_name, pat) for pat in exclude_patterns):
                continue
            remaining.append(test_name)

    return remaining


def recover_interrupted_jobs(
    state: RunState,
    all_jobs: Dict[str, JobSpec],
    reattach_list: List[tuple],  # filled in-place: (job_id, pid)
) -> None:
    """Inspect jobs recorded as 'running'; reset dead ones, queue live ones."""
    for job_id, rec in state.records.items():
        if rec.status != "running":
            continue
        pid = rec.pid
        alive = False
        if pid is not None:
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                # pid exists but we lack permission — treat as alive
                alive = True

        if alive:
            reattach_list.append((job_id, pid))
        else:
            # Archive old log so _execute can inspect it on resume
            spec = all_jobs.get(job_id)
            prev = None
            if spec:
                prev = _archive_log(spec.log_file)
            rec.status = "not_started"
            rec.result = "unknown"
            rec.pid = None
            rec.start_time = None
            rec.end_time = None
            rec.exit_code = None
            rec.tests_passed = 0
            rec.prev_log = prev  # may be None if no log existed


# ---------------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------------

def _parse_test_counts(log_file: str) -> tuple:
    """Return (passed, failed, total, current_test) from a GTest log file.

    passed:       authoritative from '[  PASSED  ] N tests.' if present, otherwise
                  the count of '[       OK ]' lines seen so far (in-progress log).
    failed:       count of '[  FAILED  ] Suite.Test (N ms)' lines (test-level, with timing).
    total:        total tests scheduled, from '[==========] Running N tests from'.
                  0 if the header has not been written yet.
    current_test: name from the last '[ RUN      ] Suite.Test' line seen; "" if none.
    """
    ok_count = 0
    fail_count = 0
    passed_final: Optional[int] = None
    total = 0
    current_test = ""
    try:
        with open(log_file, "r", errors="replace") as f:
            for line in f:
                if "[  PASSED  ]" in line:
                    # "[  PASSED  ] 212 tests."
                    parts = line.split()
                    try:
                        passed_final = int(parts[3])
                    except (IndexError, ValueError):
                        pass
                elif line.startswith("[       OK ]"):
                    ok_count += 1
                elif line.startswith("[  FAILED  ]") and "(" in line:
                    # Test-level failure: "[  FAILED  ] Suite.Test (N ms)"
                    # Suite-level summary lines have no "(" and are skipped.
                    fail_count += 1
                elif line.startswith("[ RUN      ]"):
                    # "[ RUN      ] Suite.TestName" — prefix is exactly 13 chars
                    current_test = line[13:].strip()
                elif "[==========]" in line and " Running " in line and " tests " in line:
                    # "[==========] Running 149 tests from 53 test cases."
                    parts = line.split()
                    # parts: ['[==========]', 'Running', '149', 'tests', ...]
                    try:
                        total = int(parts[2])
                    except (IndexError, ValueError):
                        pass
    except OSError:
        pass
    passed = passed_final if passed_final is not None else ok_count
    return passed, fail_count, total, current_test


# ANSI color helpers
_BOLD  = "\033[1m"
_GREEN = "\033[32m"
_RED   = "\033[31m"
_RESET = "\033[0m"


def _c(text: str, code: str, use_color: bool) -> str:
    """Wrap text in an ANSI escape if use_color is True."""
    return f"{code}{text}{_RESET}" if use_color else text


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class LiveDisplay:
    """Thread-safe progress display.  TTY: ANSI refresh.  Non-TTY: plain lines."""

    def __init__(self, state: RunState, all_groups: Dict[str, List[JobSpec]],
                 selected_jobs: List[JobSpec],
                 start_time: float, tty: bool, lock: threading.Lock) -> None:
        self._state = state
        self._start = start_time
        self._tty = tty
        self._lock = lock
        self._last_lines = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # Build a groups dict scoped to only the selected jobs, preserving
        # group order from all_groups.
        selected_ids = {sp.job_id for sp in selected_jobs}
        self._selected_groups: Dict[str, List[JobSpec]] = {
            gid: [sp for sp in specs if sp.job_id in selected_ids]
            for gid, specs in all_groups.items()
            if any(sp.job_id in selected_ids for sp in specs)
        }
        self._selected_total = len(selected_jobs)

    def start(self) -> None:
        if self._tty:
            self._thread = threading.Thread(target=self._refresh_loop,
                                            daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def log_plain(self, msg: str) -> None:
        if not self._tty:
            elapsed = _fmt_elapsed(time.monotonic() - self._start)
            print(f"[{elapsed}] {msg}", flush=True)

    def refresh(self) -> None:
        if self._tty:
            self._render_tty()

    def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            self._render_tty()
            time.sleep(0.5)
        self._render_tty()  # final frame

    def _render_tty(self) -> None:
        lines = self._build_frame()
        out = []
        if self._last_lines:
            out.append(f"\033[{self._last_lines}A\033[J")
        out.append("\n".join(lines))
        out.append("\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self._last_lines = len(lines)

    def _build_frame(self) -> List[str]:
        state = self._state
        now_mono = time.monotonic()
        now_wall = time.time()
        elapsed = _fmt_elapsed(now_mono - self._start)

        records = state.records
        selected_ids = {
            sp.job_id
            for specs in self._selected_groups.values()
            for sp in specs
        }
        running_count = sum(
            1 for jid, r in records.items()
            if jid in selected_ids and r.status == "running"
        )

        lines = [
            f"=== rocBLAS Test Runner  "
            f"[running {running_count}/{self._selected_total} jobs]  "
            f"elapsed: {elapsed} ===",
            "",
            _c(f"    {'Group':<16} {'done/total':>10}  {'pass ( tests)':>14}  {'fail ( tests)':>14}  {'running':>8}", _BOLD, True),
        ]

        for gid, specs in self._selected_groups.items():
            done = run = 0
            pass_ = fail = 0
            tests_pass = tests_fail = 0
            for sp in specs:
                rec = records.get(sp.job_id)
                if rec is None:
                    continue
                if rec.status == "finished":
                    done += 1
                    tests_pass += rec.tests_passed
                    tests_fail += rec.tests_failed
                    if rec.result == "pass":
                        pass_ += 1
                    else:
                        fail += 1
                elif rec.status == "running":
                    run += 1
                    # Include in-progress test counts so the group totals reflect
                    # work done so far, not just what completed jobs contributed.
                    tests_pass += rec.tests_passed
                    tests_fail += rec.tests_failed
            arrow = "► " if run else "  "
            pass_raw = f"{pass_} ({tests_pass:>6})"
            fail_raw = f"{fail} ({tests_fail:>6})"
            pass_cell = _c(f"{pass_raw:>14}", _GREEN, pass_ > 0 or tests_pass > 0)
            fail_cell = _c(f"{fail_raw:>14}", _RED,   fail > 0 or tests_fail > 0)
            lines.append(
                f"  {arrow}{gid:<16} {f'{done}/{len(specs)}':>10}"
                f"  {pass_cell}"
                f"  {fail_cell}"
                f"  {run:>8}"
            )

        # Running now section
        running_jobs = [
            (jid, rec)
            for jid, rec in records.items()
            if jid in selected_ids and rec.status == "running"
        ]
        if running_jobs:
            _TEST_COL = 36  # max width for the current test name

            def _fmt_pair(total: int, new: int) -> str:
                t = str(total) if total else "-"
                n = str(new)   if new   else "-"
                return f"{t:>6} ({n:>6})"

            lines.append("")
            lines.append(
                _c(
                    f"  {'Running now':<36} {'elapsed':>11}"
                    f"  {'passed (   new)':>16}  {'failed (   new)':>16}  {'total':>8}"
                    f"  {'current test'}",
                    _BOLD, True,
                )
            )
            for jid, rec in running_jobs[:self._state.max_parallel]:
                elapsed_job = (
                    f"[{_fmt_elapsed(now_wall - rec.start_time)}]"
                    if rec.start_time else ""
                )
                short = jid.split(".", 1)[1] if "." in jid else jid

                passed_total = rec.tests_passed
                passed_new   = rec.tests_passed - rec.tests_passed_offset
                failed_total = rec.tests_failed_offset + rec.tests_failed
                failed_new   = rec.tests_failed
                total_str    = str(rec.tests_total) if rec.tests_total else "-"
                cur = rec.current_test
                if len(cur) > _TEST_COL:
                    cur = "…" + cur[-(_TEST_COL - 1):]

                pass_pair = _c(f"{_fmt_pair(passed_total, passed_new):>16}", _GREEN, passed_total > 0)
                fail_pair = _c(f"{_fmt_pair(failed_total, failed_new):>16}", _RED,   failed_total > 0)
                lines.append(
                    f"    {short:<34} {elapsed_job:>11}"
                    f"  {pass_pair}"
                    f"  {fail_pair}"
                    f"  {total_str:>8}"
                    f"  {cur}"
                )

        return lines


# ---------------------------------------------------------------------------
# EXECUTOR
# ---------------------------------------------------------------------------

# Global shutdown event — set by signal handler
shutdown_event = threading.Event()
_active_procs: Dict[str, Popen] = {}
_active_procs_lock = threading.Lock()


class JobRunner:
    def __init__(self, state: RunState, all_jobs: Dict[str, JobSpec],
                 state_path: str, state_lock: threading.Lock,
                 display: LiveDisplay, max_parallel: int,
                 count_interval: int, skip_failed: bool = False,
                 exclude_patterns: Optional[List[str]] = None) -> None:
        self._state = state
        self._all_jobs = all_jobs
        self._state_path = state_path
        self._state_lock = state_lock
        self._display = display
        self._semaphore = threading.Semaphore(max_parallel)
        self._count_interval = count_interval
        self._skip_failed = skip_failed
        self._exclude_patterns: List[str] = exclude_patterns or []
        self._poll_stop = threading.Event()

    def run_all(self, jobs: List[JobSpec],
                reattach: List[tuple]) -> None:
        threads = []
        for spec in jobs:
            t = threading.Thread(target=self._run_one, args=(spec,), daemon=True)
            threads.append(t)
        # Reattach threads for already-running PIDs
        for job_id, pid in reattach:
            t = threading.Thread(target=self._reattach_one,
                                 args=(job_id, pid), daemon=True)
            threads.append(t)

        poller = threading.Thread(target=self._poll_test_counts, daemon=True)
        poller.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self._poll_stop.set()
        poller.join(timeout=2)

    def _poll_test_counts(self) -> None:
        """Periodically re-count tests in running job logs and refresh display."""
        while not self._poll_stop.wait(timeout=self._count_interval):
            if shutdown_event.is_set():
                return
            with self._state_lock:
                running = [
                    (jid, rec)
                    for jid, rec in self._state.records.items()
                    if rec.status == "running"
                ]
            updated = False
            for jid, rec in running:
                spec = self._all_jobs.get(jid)
                if spec:
                    current_passed, current_failed, current_total, cur_test = _parse_test_counts(spec.log_file)
                    with self._state_lock:
                        offset = rec.tests_passed_offset
                        rec.tests_passed = current_passed + offset
                        rec.tests_failed = current_failed
                        rec.current_test = cur_test
                        # Refresh total from the current run's header.  We update
                        # only when GTest has actually written the count line
                        # (current_total > 0); before that, _execute() seeds
                        # rec.tests_total with an estimate, so the display shows
                        # something reasonable instead of stale values from
                        # previous runs.
                        if current_total > 0:
                            rec.tests_total = current_total + offset
                    updated = True
            if updated:
                self._display.refresh()

    def _run_one(self, spec: JobSpec) -> None:
        if shutdown_event.is_set():
            return
        self._semaphore.acquire()
        try:
            if shutdown_event.is_set():
                return
            self._execute(spec)
        finally:
            self._semaphore.release()
            self._display.refresh()

    def _execute(self, spec: JobSpec) -> None:
        os.makedirs(os.path.dirname(spec.log_file), exist_ok=True)
        env = {**os.environ, "GTEST_LISTENER": "PASS_LINE_IN_LOG", "ROCBLAS_TEST_TIMEOUT": "36000"}
        start = time.time()

        # Read cumulative exclusion state and prev_log atomically, then release
        # the lock before doing any file I/O.
        with self._state_lock:
            rec = self._state.records[spec.job_id]
            prev_log = rec.prev_log
            rec.prev_log = None                        # consumed
            prev_excluded: set = set(rec.excluded_tests)  # accumulated across all runs
            prev_passed_offset: int = rec.tests_passed_offset
            # tests_total is NOT reset here: the poll thread updates it whenever
            # GTest reports a non-zero count, preserving the last known value
            # across 0-test runs so false-pass detection can use it.
            prev_failed_offset: int = rec.tests_failed_offset

        # For completed-but-failed jobs, rec.prev_log is None but the log file
        # still exists — archive it now so we can parse it for partial resume.
        if prev_log is None and os.path.exists(spec.log_file):
            prev_log = _archive_log(spec.log_file)

        # Parse new completions from the most-recently-interrupted log and merge
        # them into the cumulative exclusion set.  This ensures every restart
        # extends the set rather than replacing it.
        gtest_filter = spec.gtest_filter
        if self._exclude_patterns:
            gtest_filter = _build_exclusion_filter(gtest_filter, set(self._exclude_patterns)) or gtest_filter
        preamble_lines: List[str] = []
        new_passed_count = 0
        new_failed_count = 0
        partial_total = 0  # tests GTest ran in prev_log (< original total if filtered)

        if prev_log and os.path.exists(prev_log):
            new_passed, new_failed = _parse_completed_tests(prev_log)
            _, _, partial_total, _ = _parse_test_counts(prev_log)
            new_passed_count = len(new_passed)
            new_failed_count = len(new_failed)
            new_excluded = new_passed | (new_failed if self._skip_failed else set())
            all_excluded = prev_excluded | new_excluded
        else:
            all_excluded = prev_excluded

        cumulative_passed = prev_passed_offset + new_passed_count
        cumulative_failed = prev_failed_offset + new_failed_count

        # Build per-test exclusion filter on top of the already-modified
        # gtest_filter (which may include --exclude-pattern additions) so
        # that user-supplied patterns are not silently dropped.
        augmented = _build_exclusion_filter(gtest_filter, all_excluded)
        if augmented is not None:
            gtest_filter = augmented
            n_excluded = len(all_excluded)
            n_failed_excl = n_excluded - cumulative_passed
            skip_desc = f"{cumulative_passed} passed" + (
                f", {n_failed_excl} failed" if n_failed_excl > 0 else ""
            )
            preamble_lines = [
                f"[run_tests.py] PARTIAL RESUME — excluding {n_excluded} tests total "
                f"({skip_desc} across all runs)",
                f"[run_tests.py] Filter: {gtest_filter[:200]}"
                + ("..." if len(gtest_filter) > 200 else ""),
                "",
            ]

        # When the accumulated exclusion filter exceeds _FILTER_LIST_THRESHOLD
        # bytes, passing it inline risks hitting ARG_MAX (~128 KiB on Linux).
        # The @file and GTEST_FILTER env var alternatives are not supported by
        # this GTest build.  Instead: enumerate all test names that match the
        # *original* job filter with --gtest_list_tests, subtract the excluded
        # set and apply any --exclude-pattern globs, then build a compact
        # positive-only filter from the remaining names (typically ~8 KB).
        if len(gtest_filter) > _FILTER_LIST_THRESHOLD:
            remaining = _list_remaining_tests(
                self._state.executable,
                spec.gtest_filter,
                all_excluded,
                self._exclude_patterns,
            )
            if remaining is not None:
                if not remaining:
                    # All tests already excluded — nothing to run.
                    with self._state_lock:
                        rec.status = "finished"
                        rec.result = "pass"
                        rec.end_time = time.time()
                        rec.exit_code = 0
                    save_state(self._state, self._state_path, self._state_lock)
                    self._display.log_plain(f"SKIP (all excluded) {spec.job_id}")
                    return
                gtest_filter = ":".join(remaining)
                if preamble_lines:
                    # Update filter line to show the compact form
                    preamble_lines[-2] = (
                        f"[run_tests.py] Filter (compact, {len(remaining)} tests): "
                        f"{gtest_filter[:200]}"
                        + ("..." if len(gtest_filter) > 200 else "")
                    )
        cmd = [self._state.executable, f"--gtest_filter={gtest_filter}"]

        # Compute best estimate of the original (unfiltered) test count.
        # partial_total is what GTest ran under the *previous* exclusion set,
        # so original = partial_total + len(prev_excluded).
        estimated_original = partial_total + len(prev_excluded) if partial_total else 0

        # Update state: running — seed cumulative counts and persist exclusion set.
        with self._state_lock:
            rec.status = "running"
            rec.start_time = start
            rec.result = "unknown"
            rec.excluded_tests = sorted(all_excluded)
            rec.tests_passed_offset = cumulative_passed
            rec.tests_passed = cumulative_passed
            rec.tests_failed_offset = cumulative_failed
            rec.tests_failed = 0   # reset; poller will fill with current-run count
            # tests_total: seed with our best estimate of the original (unfiltered)
            # count.  This is reset (not "only-grow") because a stale value from a
            # previously-buggy run (e.g. unfiltered binary execution) would
            # otherwise leak into the new run's display until the poller caught up.
            rec.tests_total = estimated_original

        try:
            with open(spec.log_file, "w") as log_fh:
                for line in preamble_lines:
                    log_fh.write(line + "\n")
                proc = Popen(cmd, stdout=log_fh, stderr=STDOUT, env=env)
            with self._state_lock:
                rec.pid = proc.pid
            save_state(self._state, self._state_path, self._state_lock)

            with _active_procs_lock:
                _active_procs[spec.job_id] = proc

            self._display.log_plain(f"STARTED  {spec.job_id}")
            exit_code = proc.wait()
        except OSError as exc:
            exit_code = -1
            with open(spec.log_file, "a") as log_fh:
                log_fh.write(f"\n[run_tests.py] Failed to launch: {exc}\n")
        finally:
            with _active_procs_lock:
                _active_procs.pop(spec.job_id, None)

        end = time.time()
        _passed, _failed, _total, _ = _parse_test_counts(spec.log_file)
        with self._state_lock:
            _known_total = rec.tests_total
            _offset      = rec.tests_passed_offset
        if exit_code == 0:
            # Guard against false pass: GTest exits 0 when ALL remaining tests
            # are excluded by the filter, even if many tests were never run.
            # Detect this by comparing the known total against the exclusion
            # offset — if the offset doesn't cover the full job, mark incomplete.
            if _passed == 0 and _total == 0 and _known_total > 0 and _offset < _known_total:
                result = "incomplete"
            else:
                result = "pass"
        elif _failed > 0:
            result = "fail"
        else:
            result = "incomplete"
        self._display.log_plain(
            f"{'PASS' if result == 'pass' else result.upper():<4}     {spec.job_id}"
            f"  exit={exit_code}  log={spec.log_file}"
        )

        with self._state_lock:
            rec.status = "finished"
            rec.result = result
            rec.end_time = end
            rec.exit_code = exit_code
            rec.pid = None
            offset = rec.tests_passed_offset
            rec.tests_passed = _passed + offset
            rec.tests_failed = _failed
            if _total > 0:
                rec.tests_total = _total + offset
        save_state(self._state, self._state_path, self._state_lock)

    def _reattach_one(self, job_id: str, pid: int) -> None:
        """Wait on an already-running process without consuming a semaphore slot."""
        self._display.log_plain(f"REATTACH {job_id}  pid={pid}")
        try:
            _, wait_status = os.waitpid(pid, 0)
            exit_code = os.waitstatus_to_exitcode(wait_status)
        except ChildProcessError:
            # Not our child — we can't waitpid on it.  Mark unknown.
            exit_code = -1
        except OSError:
            exit_code = -1

        end = time.time()
        spec = self._all_jobs.get(job_id)
        if spec:
            _, _failed, _, _ = _parse_test_counts(spec.log_file)
        else:
            _failed = 0
        if exit_code == 0:
            result = "pass"
        elif _failed > 0:
            result = "fail"
        else:
            result = "incomplete"
        self._display.log_plain(
            f"{'PASS' if result == 'pass' else result.upper():<4}     {job_id}"
            f"  exit={exit_code}  (reattached)"
        )
        with self._state_lock:
            rec = self._state.records.get(job_id)
            if rec:
                rec.status = "finished"
                rec.result = result
                rec.end_time = end
                rec.exit_code = exit_code
                rec.pid = None
        save_state(self._state, self._state_path, self._state_lock)
        self._display.refresh()


# ---------------------------------------------------------------------------
# SIGNAL HANDLING
# ---------------------------------------------------------------------------

def _install_signal_handler(state: RunState, state_path: str,
                             state_lock: threading.Lock,
                             display: "LiveDisplay",
                             requested: List["JobSpec"],
                             skipped_passed: int) -> None:
    def handler(_signum, _frame):
        shutdown_event.set()
        with _active_procs_lock:
            for proc in list(_active_procs.values()):
                try:
                    proc.terminate()
                except OSError:
                    pass
        save_state(state, state_path, state_lock)
        display.stop()
        print()
        print_summary(state, requested, skipped_passed)
        sys.exit(130)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_VALID_GROUPS = ["AUXILIARY", "L1_BLAS", "L1_BLAS_EX", "L2_BLAS"]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_tests.py",
        description="Parallel rocBLAS test runner with persistent state.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-e", "--executable",
                   default="/opt/rocm/bin/rocblas-test",
                   help="Path to rocblas-test binary (default: %(default)s)")
    p.add_argument("-o", "--output-dir",
                   default=os.path.join(os.getcwd(), "tests_output"),
                   help="Directory for log files and state file (default: <cwd>/tests_output)")
    p.add_argument("-j", "--max-parallel", type=int, default=8,
                   help="Maximum concurrent test jobs (default: %(default)s)")
    p.add_argument("--group", action="append", dest="groups",
                   choices=_VALID_GROUPS, metavar="GROUP",
                   help=f"Limit to one group; repeatable. Choices: {_VALID_GROUPS}")
    p.add_argument("--job", action="append", dest="jobs", metavar="JOB_ID",
                   help="Run a specific job by ID (e.g. L2_BLAS.gemv_batched); repeatable")
    p.add_argument("--list-jobs", action="store_true",
                   help="Print all job IDs grouped by group and exit")
    p.add_argument("--reset", action="store_true",
                   help="Delete existing state file and start fresh")
    p.add_argument("--count-interval", type=int, default=30, metavar="SECONDS",
                   help="How often (in seconds) to re-count in-progress test results (default: %(default)s)")
    p.add_argument("--skip-failed", action="store_true",
                   help="On partial resume, also skip previously failed tests "
                        "(re-run only tests that have not been attempted yet)")
    p.add_argument("--exclude-pattern", action="append", dest="exclude_patterns",
                   metavar="PATTERN",
                   help="GTest pattern to exclude from every job's filter (repeatable). "
                        "Appended to the negative section of each job's --gtest_filter. "
                        "Example: --exclude-pattern '*f32_c_LNN*'")
    p.add_argument("--no-color", action="store_true",
                   help="Plain output, no ANSI escape codes")
    return p


def select_jobs(all_groups: Dict[str, List[JobSpec]],
                requested_groups: Optional[List[str]],
                requested_jobs: Optional[List[str]]) -> List[JobSpec]:
    all_flat: Dict[str, JobSpec] = {
        sp.job_id: sp
        for specs in all_groups.values()
        for sp in specs
    }

    if requested_jobs:
        result = []
        for jid in requested_jobs:
            if jid not in all_flat:
                print(f"[error] Unknown job ID: {jid}", file=sys.stderr)
                sys.exit(1)
            result.append(all_flat[jid])
        return result

    if requested_groups:
        result = []
        for gid in requested_groups:
            result.extend(all_groups[gid])
        return result

    # All jobs
    result = []
    for specs in all_groups.values():
        result.extend(specs)
    return result


def list_jobs(all_groups: Dict[str, List[JobSpec]]) -> None:
    for gid, specs in all_groups.items():
        print(f"\n{gid} ({len(specs)} jobs):")
        for sp in specs:
            print(f"  {sp.job_id}")


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def print_summary(state: RunState, all_jobs: List[JobSpec],
                  skipped_passed: int) -> int:
    color = sys.stdout.isatty()
    total = len(all_jobs)
    _default = JobRecord("", "", None, None, None, None)
    passed = sum(1 for sp in all_jobs
                 if state.records.get(sp.job_id, _default).result == "pass")
    failed_specs = [
        sp for sp in all_jobs
        if state.records.get(sp.job_id, _default).result == "fail"
    ]
    incomplete_specs = [
        sp for sp in all_jobs
        if state.records.get(sp.job_id, _default).result == "incomplete"
    ]
    failed = len(failed_specs)
    incomplete = len(incomplete_specs)

    print(_c("\n=== Summary ===", _BOLD, color))
    print(f"Total:      {total} jobs")
    print(f"Passed:     {_c(str(passed),    _GREEN, color and passed > 0)}")
    print(f"Failed:     {_c(str(failed),    _RED,   color and failed > 0)}")
    if incomplete:
        print(f"Incomplete: {_c(str(incomplete), _RED, color)}  "
              f"(interrupted/crashed before any test failed)")
    if skipped_passed:
        print(f"Skipped:    {skipped_passed}  (already passed from previous run)")

    if failed_specs:
        print(_c("\nFailed jobs:", _RED, color))
        for sp in failed_specs:
            rec = state.records[sp.job_id]
            print(f"  {sp.job_id:<45}  exit_code={rec.exit_code}  "
                  f"log: {sp.log_file}")

    if incomplete_specs:
        print(_c("\nIncomplete jobs (crashed/interrupted, no test failures recorded):", _RED, color))
        for sp in incomplete_specs:
            rec = state.records[sp.job_id]
            print(f"  {sp.job_id:<45}  exit_code={rec.exit_code}  "
                  f"log: {sp.log_file}")

    print(f"\nState file: {os.path.join(state.output_dir, 'run_state.json')}")
    return 1 if (failed or incomplete) else 0


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_groups = build_all_groups(output_dir)
    all_flat: Dict[str, JobSpec] = {
        sp.job_id: sp
        for specs in all_groups.values()
        for sp in specs
    }

    if args.list_jobs:
        list_jobs(all_groups)
        return 0

    state_path = os.path.join(output_dir, "run_state.json")
    state_lock = threading.Lock()

    if args.reset and os.path.exists(state_path):
        os.remove(state_path)
        print(f"[info] Removed state file: {state_path}")

    # Load or create state
    existing = load_state(state_path)
    if existing is not None:
        if existing.executable != args.executable:
            print(
                f"[warn] State file was created with executable "
                f"'{existing.executable}' but current --executable is "
                f"'{args.executable}'.  Use --reset if you changed the binary.",
                file=sys.stderr,
            )
        state = existing
        state.executable = args.executable
        state.output_dir = output_dir
        state.max_parallel = args.max_parallel
    else:
        state = RunState(
            version=_STATE_VERSION,
            executable=args.executable,
            output_dir=output_dir,
            max_parallel=args.max_parallel,
            records={},
        )

    # Ensure all known jobs have a record
    for job_id in all_flat:
        if job_id not in state.records:
            state.records[job_id] = JobRecord(
                status="not_started", result="unknown",
                pid=None, start_time=None, end_time=None, exit_code=None,
            )

    # Back-fill test counts for finished jobs that were loaded from an older
    # state file (tests_passed == 0) or run before PASS_LINE_IN_LOG was added.
    if existing is not None:
        for job_id, rec in state.records.items():
            if rec.status == "finished" and rec.tests_passed == 0:
                spec = all_flat.get(job_id)
                if spec and os.path.exists(spec.log_file):
                    passed, failed, total, _ = _parse_test_counts(spec.log_file)
                    rec.tests_passed = passed
                    rec.tests_failed = failed
                    if total:
                        rec.tests_total = total

    # Recover any jobs recorded as "running" from a previous interrupted run
    reattach_list: List[tuple] = []
    if existing is not None:
        recover_interrupted_jobs(state, all_flat, reattach_list)

    # Determine which jobs to run this session
    requested = select_jobs(all_groups, args.groups, args.jobs)

    # Filter to pending jobs — only skip jobs that have already fully passed.
    # Failed jobs are always re-run; --skip-failed controls individual test
    # filtering inside the job (skip previously failed tests), not whole jobs.
    to_run = []
    skipped_passed = 0
    for sp in requested:
        rec = state.records[sp.job_id]
        if rec.status == "finished" and rec.result == "pass":
            skipped_passed += 1
        else:
            to_run.append(sp)

    if skipped_passed:
        print(f"[info] Skipping {skipped_passed} already-passed job(s).  Use --reset to re-run them.")

    if not to_run and not reattach_list:
        print("[info] Nothing to do — all selected jobs have already passed.")
        return print_summary(state, requested, skipped_passed)

    save_state(state, state_path, state_lock)

    tty = sys.stdout.isatty() and not args.no_color
    start_time = time.monotonic()
    display = LiveDisplay(state, all_groups, requested, start_time, tty, state_lock)
    display.start()
    _install_signal_handler(state, state_path, state_lock, display, requested, skipped_passed)

    runner = JobRunner(
        state=state,
        all_jobs=all_flat,
        state_path=state_path,
        state_lock=state_lock,
        display=display,
        max_parallel=args.max_parallel,
        count_interval=args.count_interval,
        skip_failed=args.skip_failed,
        exclude_patterns=args.exclude_patterns,
    )

    try:
        runner.run_all(to_run, reattach_list)
    finally:
        display.stop()

    return print_summary(state, requested, skipped_passed)


if __name__ == "__main__":
    sys.exit(main())
