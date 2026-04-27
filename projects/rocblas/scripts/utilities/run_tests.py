#!/usr/bin/env python3

"""Copyright (C) 2016-2026 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

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

# List all valid job IDs:
    python3 run_tests.py --list-jobs

# Plain output (no ANSI, safe for tee / CI logs):
    python3 run_tests.py --no-color

Resume behaviour
----------------
Interrupted runs (SSH drop, OOM kill, Ctrl+C) are resumed automatically.
  - Jobs whose recorded PID is still alive are waited on (reattached).
  - Jobs whose recorded PID is dead are reset to "not_started" and re-run.
  - Failed jobs are always re-run on resume.
  - Use --reset to force a completely clean start.

Caveats
-------
- PID reuse: os.kill(pid, 0) may hit a recycled PID on resume.  Use --reset
  if you suspect PID recycling.
- Two concurrent script instances in the same output dir will corrupt state.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from subprocess import Popen, STDOUT
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
    l1_functions = [
        "asum", "axpy", "copy", "dot", "dotc",
        "iamax", "iamin", "nrm2", "rot", "rotg",
        "rotm", "rotmg", "scal", "swap",
    ]
    jobs: List[JobSpec] = []
    for fn in l1_functions:
        jobs.append(make("L1_BLAS", fn,
                         f"*{fn}*quick*-*_batched*:*_ex*"))
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
    l2_functions = [
        "trsv", "gbmv", "gemv", "hbmv", "hemv",
        "her", "her2", "hpmv", "hpr", "hpr2",
        "trmv", "tpmv", "tbmv", "tbsv", "ger",
        "geru", "gerc", "spr", "spr2", "syr",
        "syr2", "sbmv", "spmv", "symv",
    ]
    jobs = []
    for fn in l2_functions:
        jobs.append(make("L2_BLAS", fn,
                         f"*{fn}*quick*-*_batched*:*_ex*"))
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
    result: str     # "unknown" | "pass" | "fail"
    pid: Optional[int]
    start_time: Optional[float]
    end_time: Optional[float]
    exit_code: Optional[int]


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
            # Preserve old log for debugging
            spec = all_jobs.get(job_id)
            if spec and os.path.exists(spec.log_file):
                try:
                    os.rename(spec.log_file, spec.log_file + ".prev")
                except OSError:
                    pass
            rec.status = "not_started"
            rec.result = "unknown"
            rec.pid = None
            rec.start_time = None
            rec.end_time = None
            rec.exit_code = None


# ---------------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------------

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
            f"  {'Group':<16} {'done/total':>10}  {'pass':>5}  {'fail':>5}  {'running':>8}",
        ]

        for gid, specs in self._selected_groups.items():
            done = pass_ = fail = run = 0
            for sp in specs:
                rec = records.get(sp.job_id)
                if rec is None:
                    continue
                if rec.status == "finished":
                    done += 1
                    if rec.result == "pass":
                        pass_ += 1
                    else:
                        fail += 1
                elif rec.status == "running":
                    run += 1
            marker = " <<<" if run else ""
            lines.append(
                f"  {gid:<16} {done:>4}/{len(specs):<4}   {pass_:>4}   {fail:>4}   {run:>4}{marker}"
            )

        # Running now section
        running_jobs = [
            (jid, rec)
            for jid, rec in records.items()
            if jid in selected_ids and rec.status == "running"
        ]
        if running_jobs:
            lines.append("")
            lines.append("  Running now:")
            for jid, rec in running_jobs[:8]:
                elapsed_job = ""
                if rec.start_time:
                    elapsed_job = f"[{_fmt_elapsed(now_wall - rec.start_time)}]"
                short = jid.split(".", 1)[1] if "." in jid else jid
                lines.append(f"    {short:<40} {elapsed_job}")

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
                 display: LiveDisplay, max_parallel: int) -> None:
        self._state = state
        self._all_jobs = all_jobs
        self._state_path = state_path
        self._state_lock = state_lock
        self._display = display
        self._semaphore = threading.Semaphore(max_parallel)

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

        for t in threads:
            t.start()
        for t in threads:
            t.join()

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
        cmd = [self._state.executable,
               f"--gtest_filter={spec.gtest_filter}"]
        start = time.time()

        # Update state: running
        with self._state_lock:
            rec = self._state.records[spec.job_id]
            rec.status = "running"
            rec.start_time = start
            rec.result = "unknown"

        try:
            with open(spec.log_file, "w") as log_fh:
                proc = Popen(cmd, stdout=log_fh, stderr=STDOUT)
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
        result = "pass" if exit_code == 0 else "fail"
        self._display.log_plain(
            f"{'PASS' if result == 'pass' else 'FAIL'}     {spec.job_id}"
            f"  exit={exit_code}  log={spec.log_file}"
        )

        with self._state_lock:
            rec.status = "finished"
            rec.result = result
            rec.end_time = end
            rec.exit_code = exit_code
            rec.pid = None
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
        result = "pass" if exit_code == 0 else "fail"
        self._display.log_plain(
            f"{'PASS' if result == 'pass' else 'FAIL'}     {job_id}"
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
                             state_lock: threading.Lock) -> None:
    def handler(_signum, _frame):
        shutdown_event.set()
        with _active_procs_lock:
            for proc in list(_active_procs.values()):
                try:
                    proc.terminate()
                except OSError:
                    pass
        save_state(state, state_path, state_lock)
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
                  skipped_count: int) -> int:
    total = len(all_jobs)
    passed = sum(1 for sp in all_jobs
                 if state.records.get(sp.job_id, JobRecord("", "", None, None, None, None)).result == "pass")
    failed_specs = [
        sp for sp in all_jobs
        if state.records.get(sp.job_id, JobRecord("", "", None, None, None, None)).result == "fail"
    ]
    failed = len(failed_specs)

    print("\n=== Summary ===")
    print(f"Total:    {total} jobs")
    print(f"Passed:   {passed}")
    print(f"Failed:   {failed}")
    if skipped_count:
        print(f"Skipped:  {skipped_count}  (already passed from previous run)")

    if failed_specs:
        print("\nFailed jobs:")
        for sp in failed_specs:
            rec = state.records[sp.job_id]
            print(f"  {sp.job_id:<45}  exit_code={rec.exit_code}  "
                  f"log: {sp.log_file}")

    print(f"\nState file: {os.path.join(state.output_dir, 'run_state.json')}")
    return 1 if failed else 0


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

    # Recover any jobs recorded as "running" from a previous interrupted run
    reattach_list: List[tuple] = []
    if existing is not None:
        recover_interrupted_jobs(state, all_flat, reattach_list)

    # Determine which jobs to run this session
    requested = select_jobs(all_groups, args.groups, args.jobs)

    # Filter to pending (not yet passed) — skip already-passed jobs
    to_run = []
    skipped = 0
    for sp in requested:
        rec = state.records[sp.job_id]
        if rec.status == "finished" and rec.result == "pass":
            skipped += 1
        else:
            to_run.append(sp)

    if skipped:
        print(f"[info] Skipping {skipped} already-passed job(s).  Use --reset to re-run them.")

    if not to_run and not reattach_list:
        print("[info] Nothing to do — all selected jobs have already passed.")
        return print_summary(state, requested, skipped)

    save_state(state, state_path, state_lock)
    _install_signal_handler(state, state_path, state_lock)

    tty = sys.stdout.isatty() and not args.no_color
    start_time = time.monotonic()
    display = LiveDisplay(state, all_groups, requested, start_time, tty, state_lock)
    display.start()

    runner = JobRunner(
        state=state,
        all_jobs=all_flat,
        state_path=state_path,
        state_lock=state_lock,
        display=display,
        max_parallel=args.max_parallel,
    )

    try:
        runner.run_all(to_run, reattach_list)
    finally:
        display.stop()

    return print_summary(state, requested, skipped)


if __name__ == "__main__":
    sys.exit(main())
