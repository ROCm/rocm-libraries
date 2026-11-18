#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Sample memory use of a command and its whole process tree.

Runs ``cmd`` and, every ``--interval`` seconds, walks ``/proc`` for every
descendant of it, reading ``/proc/<pid>/smaps_rollup`` for RSS, PSS and
private (USS) footprints.  Reports the peak of the summed values across the
tree, which process class held the peak, and the phase of the build that was
running at the time (phases are inferred from the command's own stdout).

PSS counts a page shared by N processes as 1/N in each, so the sum over a
tree counts each physical page once.  Summed RSS double-counts pages shared
between processes (copy-on-write pages after fork, shared libraries), so
RSS over-states footprint whenever processes share memory.

Usage::

    memwatch.py [--interval 0.5] [--out summary.json] [--csv timeline.csv]
                [--log child.log] -- python -m Tensile.TensileCreateLibrary ...

Exit status is the child's exit status; a child killed by a signal gives
128 + the signal number, as a shell would report.

Linux only: it reads /proc/<pid>/smaps_rollup (kernel 4.14 or later), which is
readable for processes of the same user or with CAP_SYS_PTRACE.

The default interval of 0.5 s is about what one sample costs on a tree of 100
processes: reading smaps_rollup walks each process's page tables, and on a 96
worker build a sample took 0.25 to 0.39 s, so a shorter interval just runs
samples back to back. On the gfx942 Equality subset (98 workers, 21 s) the
peak summed PSS agreed within 2% at 1, 0.5, 0.25 and 0.1 s (9.32, 9.52, 9.48,
9.46 GiB) and the sampled build took no longer than the unsampled one
(21.5 to 21.8 s versus 21.0 to 22.8 s).
"""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict

# Process classes, matched against /proc/<pid>/cmdline (NUL-joined argv) in
# order. The tool classes come before the interpreter so that a compiler
# invoked with a Python path among its arguments is not taken for the
# driver; "python-main" then only matches an interpreter in argv[0]. The
# root process is always reported as "python-main" regardless.
CLASSIFIERS = [
    (
        "loky-worker",
        re.compile(
            r"loky|joblib\.externals|multiprocessing\.(spawn|forkserver|resource_tracker)|concurrent\.futures"
        ),
    ),
    ("clang", re.compile(r"(^|/)(amd)?clang(\+\+)?(-\d+)?(\x00|$)")),
    ("lld", re.compile(r"(^|/)(ld\.lld|lld)(\x00|$)")),
    ("bundler", re.compile(r"clang-offload-bundler")),
    ("sh", re.compile(r"(^|/)(ba)?sh(\x00|$)")),
    ("python-main", re.compile(r"^[^\x00]*python[0-9.]*(\x00|$)")),
]

CLASS_NAMES = [name for name, _ in CLASSIFIERS] + ["other"]

# Lines on the child's stdout that mark a phase boundary. ParallelMap2 prints
# "<phase>: Launching N threads..." when a phase starts and "<phase>: Done."
# when it returns; run() prints "Time to <phase> (s): ..." after a phase has
# finished. Any line that marks a completion maps to the label "after <phase>",
# so the memory sampled until the next start marker is attributed to work
# that follows the completed phase, not to the phase itself.
PHASE_START = re.compile(r"^(?P<phase>.*?): Launching \d+ threads")
PHASE_DONE = re.compile(r"^(?P<phase>.*?): Done\.")
PHASE_TIME = re.compile(r"^Time to (?P<phase>[^(]+)")
PHASE_WRITER_DONE = re.compile(r"^# Tensile(Lite)? Library Writer DONE")
PHASE_TOTAL = re.compile(r"^Total time")

# Phases whose ParallelMap2 call returns a generator: "Done." is printed when
# the generator is handed back, before any result has been consumed, so it
# does not mark the end of the work. The phase ends at its "Time to" line.
# Today that is only logic loading (generateLogicDataAndSolutions in
# create_library/run.py); a new return_as="generator" caller belongs here too.
GENERATOR_PHASES = {"Loading Logics..."}


def classify(cmdline: str) -> str:
    for name, rx in CLASSIFIERS:
        if rx.search(cmdline):
            return name
    return "other"


def read_file(path: str):
    try:
        with open(path, "rb") as f:
            return f.read()
    except (OSError, ProcessLookupError):
        return None


def proc_table():
    """Return {pid: (ppid, starttime, session, state)} for all processes.

    starttime (clock ticks since boot, /proc/<pid>/stat field 22) tells a
    reused pid apart from the process that held it before; session is the
    session id, which every descendant of the command shares because the
    command is started as a session leader.
    """
    table = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        data = read_file(f"/proc/{entry}/stat")
        if not data:
            continue
        # comm may contain spaces/parens; the fields after ')' start at field 3,
        # so state is index 0, ppid index 1, session index 3, starttime index 19.
        rpar = data.rfind(b")")
        fields = data[rpar + 2 :].split()
        try:
            table[int(entry)] = (
                int(fields[1]),
                int(fields[19]),
                int(fields[3]),
                fields[0].decode(),
            )
        except (IndexError, ValueError):
            continue
    return table


def tree_members(root: int, table):
    """Pids of the command's tree: the root's descendants by parent pid, plus
    every process in the root's session, which keeps a worker that init
    adopted after the root died."""
    children = defaultdict(list)
    for pid, info in table.items():
        children[info[0]].append(pid)
    out = []
    seen = set()
    stack = [root]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        stack.extend(children.get(pid, ()))
    for pid, info in table.items():
        if pid not in seen and len(info) > 2 and info[2] == root:
            seen.add(pid)
            out.append(pid)
    return out


_SMAPS_KEYS = (
    b"Rss:",
    b"Pss:",
    b"Private_Clean:",
    b"Private_Dirty:",
    b"Swap:",
    b"Pss_Anon:",
    b"Pss_File:",
)


def smaps_rollup(pid: int):
    data = read_file(f"/proc/{pid}/smaps_rollup")
    if not data:
        return None
    vals = {}
    for line in data.splitlines():
        for key in _SMAPS_KEYS:
            if line.startswith(key):
                vals[key[:-1].decode()] = int(line.split()[1]) * 1024
                break
    if "Rss" not in vals:
        return None
    vals["Uss"] = vals.get("Private_Clean", 0) + vals.get("Private_Dirty", 0)
    return vals


def mem_available():
    data = read_file("/proc/meminfo")
    if not data:
        return None
    for line in data.splitlines():
        if line.startswith(b"MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


class Sampler(threading.Thread):
    def __init__(self, root_pid: int, interval: float, csv_path):
        super().__init__(daemon=True)
        self.root_pid = root_pid
        self.interval = interval
        self.csv_path = csv_path
        self.stop_event = threading.Event()
        self.t0 = time.monotonic()
        self.phase = "startup"
        self.phase_lock = threading.Lock()
        self.rows = []
        self.readable = 0  # processes whose smaps_rollup could be read, ever
        self.peaks = {
            "rss_sum": (0, 0.0, ""),
            "pss_sum": (0, 0.0, ""),
            "uss_sum": (0, 0.0, ""),
        }
        self.per_class_peak = defaultdict(lambda: {"rss": 0, "pss": 0, "count": 0})
        # Largest RSS and largest PSS of any single process in a class, tracked
        # separately: with shared pages the two need not be the same process.
        self.max_single = defaultdict(
            lambda: {"rss": 0, "rss_pid": 0, "pss": 0, "pss_pid": 0}
        )
        self.main_peak = {"rss": 0, "pss": 0}
        self.phase_peaks = defaultdict(lambda: {"rss_sum": 0, "pss_sum": 0})
        self.mem_avail0 = mem_available()
        self.min_mem_avail = self.mem_avail0
        self.sample_cost = []

    def set_phase(self, phase: str):
        with self.phase_lock:
            self.phase = phase

    def sample(self):
        t = time.monotonic() - self.t0
        table = proc_table()
        pids = tree_members(self.root_pid, table)
        by_class = defaultdict(lambda: {"rss": 0, "pss": 0, "uss": 0, "count": 0})
        rss_sum = pss_sum = uss_sum = 0
        # The root is listed even when it exited between proc_table() and here.
        pids = [pid for pid in pids if pid in table]
        for pid in pids:
            vals = smaps_rollup(pid)
            if vals is None:
                continue
            self.readable += 1
            # Classified afresh every sample: a cache keyed on the pid would
            # survive pid reuse (a build spawns 200k assemblers) and one keyed
            # on (pid, starttime) would survive exec, which changes the command
            # but neither value. One small /proc read per process is cheap next
            # to smaps_rollup.
            if pid == self.root_pid:
                cls = "python-main"
            else:
                raw = read_file(f"/proc/{pid}/cmdline") or b""
                cls = classify(raw.decode("utf-8", "replace"))
            rss, pss, uss = vals["Rss"], vals["Pss"], vals["Uss"]
            rss_sum += rss
            pss_sum += pss
            uss_sum += uss
            c = by_class[cls]
            c["rss"] += rss
            c["pss"] += pss
            c["uss"] += uss
            c["count"] += 1
            ms = self.max_single[cls]
            if rss > ms["rss"]:
                ms["rss"], ms["rss_pid"] = rss, pid
            if pss > ms["pss"]:
                ms["pss"], ms["pss_pid"] = pss, pid
            if pid == self.root_pid:
                self.main_peak["rss"] = max(self.main_peak["rss"], rss)
                self.main_peak["pss"] = max(self.main_peak["pss"], pss)
        with self.phase_lock:
            phase = self.phase
        for key, val in (
            ("rss_sum", rss_sum),
            ("pss_sum", pss_sum),
            ("uss_sum", uss_sum),
        ):
            if val > self.peaks[key][0]:
                self.peaks[key] = (val, t, phase)
        for cls, c in by_class.items():
            pk = self.per_class_peak[cls]
            pk["rss"] = max(pk["rss"], c["rss"])
            pk["pss"] = max(pk["pss"], c["pss"])
            pk["count"] = max(pk["count"], c["count"])
        pp = self.phase_peaks[phase]
        pp["rss_sum"] = max(pp["rss_sum"], rss_sum)
        pp["pss_sum"] = max(pp["pss_sum"], pss_sum)
        avail = mem_available()
        if avail is not None and (
            self.min_mem_avail is None or avail < self.min_mem_avail
        ):
            self.min_mem_avail = avail
        row = {
            "t": round(t, 3),
            "phase": phase,
            "nproc": len(pids),
            "rss_sum": rss_sum,
            "pss_sum": pss_sum,
            "uss_sum": uss_sum,
            "mem_avail": avail,
        }
        for cls in CLASS_NAMES:
            c = by_class.get(cls)
            row[f"{cls}.rss"] = c["rss"] if c else 0
            row[f"{cls}.pss"] = c["pss"] if c else 0
            row[f"{cls}.n"] = c["count"] if c else 0
        self.rows.append(row)

    def run(self):
        while not self.stop_event.is_set():
            start = time.monotonic()
            try:
                self.sample()
            except Exception as e:  # keep sampling no matter what
                print(f"memwatch: sample failed: {e}", file=sys.stderr)
            cost = time.monotonic() - start
            self.sample_cost.append(cost)
            self.stop_event.wait(max(0.0, self.interval - cost))
        # One final sample so the tail is recorded.
        try:
            self.sample()
        except Exception:
            pass

    def summary(self, elapsed: float, returncode: int, cmd):
        gib = 1024.0**3
        return {
            "cmd": cmd,
            "returncode": returncode,
            "elapsed_s": round(elapsed, 2),
            "interval_s": self.interval,
            "samples": len(self.rows),
            "mean_sample_cost_s": round(
                sum(self.sample_cost) / max(1, len(self.sample_cost)), 4
            ),
            "peak": {
                k: {
                    "bytes": v[0],
                    "GiB": round(v[0] / gib, 3),
                    "t": round(v[1], 1),
                    "phase": v[2],
                }
                for k, v in self.peaks.items()
            },
            "python_main_peak": {
                k: {"bytes": v, "GiB": round(v / gib, 3)}
                for k, v in self.main_peak.items()
            },
            "per_class_peak": {
                cls: {
                    "rss_GiB": round(v["rss"] / gib, 3),
                    "pss_GiB": round(v["pss"] / gib, 3),
                    "max_count": v["count"],
                    "largest_single_rss_MiB": round(
                        self.max_single[cls]["rss"] / 2**20, 1
                    ),
                    "largest_single_pss_MiB": round(
                        self.max_single[cls]["pss"] / 2**20, 1
                    ),
                }
                for cls, v in sorted(self.per_class_peak.items())
            },
            "phase_peak": {
                ph: {
                    "rss_sum_GiB": round(v["rss_sum"] / gib, 3),
                    "pss_sum_GiB": round(v["pss_sum"] / gib, 3),
                }
                for ph, v in self.phase_peaks.items()
            },
            "system_mem_available_drop_GiB": (
                round((self.mem_avail0 - self.min_mem_avail) / gib, 3)
                if self.mem_avail0 is not None and self.min_mem_avail is not None
                else None
            ),
            "warnings": self.warnings(),
        }

    def warnings(self):
        out = []
        if self.rows and self.readable == 0:
            out.append(
                "no process in the tree had a readable /proc/<pid>/smaps_rollup "
                "(needs Linux >= 4.14 and the same user as the command, or "
                "CAP_SYS_PTRACE); every memory figure above is 0 for that reason"
            )
        return out

    def write_csv(self):
        if not self.csv_path or not self.rows:
            return
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


def phase_from_line(line: str):
    """The phase label a stdout line switches to, or None if it is not a marker."""
    m = PHASE_START.match(line)
    if m:
        return m.group("phase").strip()
    m = PHASE_DONE.match(line)
    if m:
        phase = m.group("phase").strip()
        if phase in GENERATOR_PHASES:
            return None
        return f"after {phase}"
    m = PHASE_TIME.match(line)
    if m:
        return f"after {m.group('phase').strip()}"
    if PHASE_WRITER_DONE.match(line):
        return "after library writer"
    if PHASE_TOTAL.match(line):
        return "done"
    return None


def exit_status(returncode: int) -> int:
    """Shell-style exit status for a Popen return code: 128 + signal for a signal death."""
    if returncode < 0:
        return 128 - returncode
    return returncode


class OutputPump(threading.Thread):
    """Forwards the command's stdout lines to the phase tracker, the log and stdout.

    Runs on its own thread so the main thread can wait for the root process
    independently of the pipe, which a surviving worker may hold open after
    the root is gone. If handling a line fails (the log's filesystem is full,
    say) the error is kept, the command's group is terminated through
    ``on_error`` and the pipe is still drained so no writer blocks on it.
    """

    def __init__(self, stream, sampler, log, echo, on_error):
        super().__init__(daemon=True)
        self.stream = stream
        self.sampler = sampler
        self.log = log
        self.echo = echo
        self.on_error = on_error
        self.error = None

    def run(self):
        for line in self.stream:
            if self.error is not None:
                continue
            try:
                phase = phase_from_line(line)
                if phase:
                    self.sampler.set_phase(phase)
                if self.log:
                    self.log.write(line)
                    self.log.flush()
                if self.echo:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            except Exception as e:  # noqa: BLE001 - any output failure ends the run
                self.error = e
                self.on_error()


class RunResult:
    def __init__(self, sampler, proc, elapsed, error, leftover):
        self.sampler = sampler
        self.proc = proc
        self.elapsed = elapsed
        self.error = error
        self.leftover = leftover  # {pid: starttime} that survived the cleanup


def live_members(root: int):
    """{pid: starttime} of the command's tree members still alive (not zombies), root excluded."""
    table = proc_table()
    return {
        pid: table[pid][1]
        for pid in tree_members(root, table)
        if pid != root
        and pid in table
        and (len(table[pid]) < 4 or table[pid][3] != "Z")
    }


def _signal_group(pgid: int, sig) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _signal_tree(root: int, sig) -> None:
    """Send ``sig`` to the root's process group and to every live member of its
    tree individually: a descendant may have made its own process group while
    staying in the session. Each pid is checked against its recorded start
    time first so a pid recycled since discovery is never signalled."""
    _signal_group(root, sig)
    members = live_members(root)
    table = proc_table()
    for pid, start in members.items():
        if pid in table and table[pid][1] == start:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass


def _terminate_tree(root: int, root_alive, timeout: float = 5.0):
    """SIGTERM the tree, wait up to ``timeout``, then SIGKILL and wait again.

    ``root_alive()`` reports whether the root still runs; once it is gone the
    wait is on the other members. Returns the members still alive at the end,
    empty when the cleanup succeeded.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        _signal_tree(root, sig)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not root_alive() and not live_members(root):
                return {}
            time.sleep(0.1)
    return live_members(root)


def run_command(cmd, interval, csv_path, log_path, echo, grace):
    """Run ``cmd`` under the sampler and see it, and everything it started, to the end.

    The command leads its own session, so one process group covers every
    descendant. The sequence is:

    1. open the log (an unusable destination means nothing is launched);
    2. start the command, the sampler and the output pump;
    3. wait for the root, forwarding SIGINT/SIGTERM/SIGHUP to the group
       (a signal memwatch inherited as ignored, as SIGHUP under nohup, stays
       ignored by memwatch and the command alike); if the pump fails to write
       the log or stdout, terminate the group with a bounded
       SIGTERM-then-SIGKILL;
    4. once the root is gone, give the processes still alive in its session
       ``grace`` seconds (they are not part of a build that finished
       normally), then terminate them the same way, each signalled directly
       in case it left the root's process group; any that survive SIGKILL are
       reported and make the exit status nonzero;
    5. close the log; a flush or close failure counts as an output error.

    Raises OSError if the log cannot be opened or the command cannot start.
    """
    log = open(log_path, "w") if log_path else None
    try:
        # Unbuffered child stdout so phase markers arrive when they are
        # printed, not when a pipe buffer happens to fill.
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            env=env,
            start_new_session=True,
        )
    except OSError:
        if log:
            log.close()
        raise
    pgid = proc.pid
    sampler = Sampler(proc.pid, interval, csv_path)
    sampler.start()
    t0 = time.monotonic()
    error = None
    leftover = {}
    try:
        pump = OutputPump(
            proc.stdout,
            sampler,
            log,
            echo=echo,
            on_error=lambda: _signal_group(pgid, signal.SIGTERM),
        )
        pump.start()

        def forward(sig, _frame):
            _signal_group(pgid, sig)

        # A signal that reached memwatch already ignored (SIGHUP under nohup
        # or a CI agent) stays ignored: the command inherited that disposition
        # across exec, so forwarding it would do nothing, and the caller asked
        # for it to be ignored. The others are forwarded to the whole group.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            if signal.getsignal(sig) != signal.SIG_IGN:
                signal.signal(sig, forward)

        # Wait for the root itself, not for end-of-file on the pipe, and keep
        # an eye on the pump: a failed write must end the command even if it
        # ignores the first SIGTERM.
        while True:
            try:
                proc.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if pump.error is not None:
                    leftover = _terminate_tree(pgid, lambda: proc.poll() is None)
                    proc.wait()
                    break
        elapsed = time.monotonic() - t0

        # Survivors are judged by what is alive in the session, not by who
        # still holds the pipe: a worker that redirected its output would
        # otherwise slip past.
        deadline = time.monotonic() + grace
        while live_members(pgid) and time.monotonic() < deadline:
            time.sleep(0.1)
        if live_members(pgid):
            print(
                f"memwatch: command exited but processes of its session are still "
                f"running after {grace:g}s; terminating them",
                file=sys.stderr,
            )
            leftover = _terminate_tree(pgid, lambda: False)
        pump.join(5)
        error = pump.error
    finally:
        sampler.stop_event.set()
        sampler.join()
        if log:
            try:
                log.close()
            except OSError as e:
                if error is None:
                    error = e
    return RunResult(sampler, proc, elapsed, error, leftover)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--interval", type=float, default=0.5, help="sampling interval in seconds"
    )
    ap.add_argument("--out", default="memwatch.json", help="JSON summary path")
    ap.add_argument("--csv", default=None, help="CSV timeline path")
    ap.add_argument(
        "--log", default=None, help="file to copy the child's stdout/stderr into"
    )
    ap.add_argument(
        "--quiet", action="store_true", help="do not echo the child's output"
    )
    ap.add_argument(
        "--grace",
        type=float,
        default=10.0,
        help="seconds to wait for processes the command leaves running after it "
        "exits before terminating them (default 10)",
    )
    ap.add_argument(
        "cmd", nargs=argparse.REMAINDER, help="command to run (prefix with --)"
    )
    args = ap.parse_args(argv)
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given")

    try:
        run = run_command(
            cmd, args.interval, args.csv, args.log, not args.quiet, args.grace
        )
    except OSError as e:
        # Nothing was started: the log destination was unusable or the
        # command could not be executed.
        print(f"memwatch: error: {e}", file=sys.stderr)
        return 1
    sampler, proc, elapsed = run.sampler, run.proc, run.elapsed

    summary = sampler.summary(elapsed, proc.returncode, cmd)
    if run.leftover:
        pids = ", ".join(str(pid) for pid in sorted(run.leftover))
        summary["warnings"].append(
            f"could not terminate {len(run.leftover)} process(es) of the command's "
            f"tree after SIGKILL: pid {pids}; they are still running"
        )
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    sampler.write_csv()
    for w in summary["warnings"]:
        print(f"memwatch: warning: {w}", file=sys.stderr)

    pk = summary["peak"]
    print("\n==== memwatch summary ====")
    print(
        f"elapsed: {summary['elapsed_s']} s   exit: {proc.returncode}   samples: {summary['samples']}"
    )
    print(
        f"peak sum RSS : {pk['rss_sum']['GiB']:8.3f} GiB  at t={pk['rss_sum']['t']}s  [{pk['rss_sum']['phase']}]"
    )
    print(
        f"peak sum PSS : {pk['pss_sum']['GiB']:8.3f} GiB  at t={pk['pss_sum']['t']}s  [{pk['pss_sum']['phase']}]"
    )
    print(
        f"peak sum USS : {pk['uss_sum']['GiB']:8.3f} GiB  at t={pk['uss_sum']['t']}s  [{pk['uss_sum']['phase']}]"
    )
    mp = summary["python_main_peak"]
    print(
        f"main process peak: RSS {mp['rss']['GiB']:.3f} GiB  PSS {mp['pss']['GiB']:.3f} GiB"
    )
    print("per-class peaks (summed over class):")
    for cls, v in summary["per_class_peak"].items():
        print(
            f"  {cls:13s} rss {v['rss_GiB']:8.3f} GiB  pss {v['pss_GiB']:8.3f} GiB  "
            f"max n={v['max_count']:4d}  largest single rss {v['largest_single_rss_MiB']:8.1f} MiB"
        )
    print("per-phase peak of summed PSS:")
    for ph, v in summary["phase_peak"].items():
        print(
            f"  {ph[:50]:50s} rss {v['rss_sum_GiB']:8.3f} GiB  pss {v['pss_sum_GiB']:8.3f} GiB"
        )
    print(f"summary written to {args.out}")
    if run.error is not None:
        print(
            f"memwatch: error: could not handle the command's output ({run.error}); "
            "the command was terminated",
            file=sys.stderr,
        )
        return exit_status(proc.returncode) or 1
    if run.leftover:
        return exit_status(proc.returncode) or 1
    return exit_status(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
