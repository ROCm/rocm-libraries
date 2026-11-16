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

PAGE = os.sysconf("SC_PAGE_SIZE")

# Process classes, matched against /proc/<pid>/cmdline (NUL-joined argv) in order.
CLASSIFIERS = [
    (
        "loky-worker",
        re.compile(
            r"loky|joblib\.externals|multiprocessing\.(spawn|forkserver|resource_tracker)|concurrent\.futures"
        ),
    ),
    ("python-main", re.compile(r"TensileCreateLibrary|Tensile\.|python")),
    ("clang", re.compile(r"(^|/)(amd)?clang(\+\+)?(-\d+)?(\x00|$)")),
    ("lld", re.compile(r"(^|/)(ld\.lld|lld)(\x00|$)")),
    ("bundler", re.compile(r"clang-offload-bundler")),
    ("sh", re.compile(r"(^|/)(ba)?sh(\x00|$)")),
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
    """Return {pid: ppid} for all processes."""
    table = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        data = read_file(f"/proc/{entry}/stat")
        if not data:
            continue
        # comm may contain spaces/parens; ppid is the 2nd field after ')'.
        rpar = data.rfind(b")")
        fields = data[rpar + 2 :].split()
        try:
            table[int(entry)] = int(fields[1])
        except (IndexError, ValueError):
            continue
    return table


def descendants(root: int, table):
    children = defaultdict(list)
    for pid, ppid in table.items():
        children[ppid].append(pid)
    out = []
    stack = [root]
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(children.get(pid, ()))
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
        self.cmdline_cache = {}
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
        pids = descendants(self.root_pid, table)
        by_class = defaultdict(lambda: {"rss": 0, "pss": 0, "uss": 0, "count": 0})
        rss_sum = pss_sum = uss_sum = 0
        for pid in pids:
            vals = smaps_rollup(pid)
            if vals is None:
                continue
            cls = self.cmdline_cache.get(pid)
            if cls is None:
                raw = read_file(f"/proc/{pid}/cmdline") or b""
                cls = classify(raw.decode("utf-8", "replace"))
                if pid == self.root_pid:
                    cls = "python-main"
                self.cmdline_cache[pid] = cls
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
        }

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
        "cmd", nargs=argparse.REMAINDER, help="command to run (prefix with --)"
    )
    args = ap.parse_args(argv)
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given")

    log = open(args.log, "w") if args.log else None
    # Unbuffered child stdout so phase markers arrive when they are printed,
    # not when a pipe buffer happens to fill.
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    # The child leads its own process group so that a signal to memwatch can
    # be forwarded to every descendant: signalling only the child would leave
    # its workers and assemblers alive, holding stdout open, and the read loop
    # below would wait for them.
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
    sampler = Sampler(proc.pid, args.interval, args.csv)
    sampler.start()
    t0 = time.monotonic()

    def forward(sig, _frame):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)

    for line in proc.stdout:
        phase = phase_from_line(line)
        if phase:
            sampler.set_phase(phase)
        if log:
            log.write(line)
        if not args.quiet:
            sys.stdout.write(line)
    proc.wait()
    elapsed = time.monotonic() - t0
    sampler.stop_event.set()
    sampler.join()
    if log:
        log.close()

    summary = sampler.summary(elapsed, proc.returncode, cmd)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    sampler.write_csv()

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
    return exit_status(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
