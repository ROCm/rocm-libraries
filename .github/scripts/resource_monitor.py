#!/usr/bin/env python3
"""Run a Linux CI command while recording its CPU and memory consumption.

Inside a job container, cgroup counters measure the whole container. On a
hardware runner without a job container, /proc samples the command's process
tree. Neither mode requires third-party Python packages.
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if sys.platform.startswith("linux"):
    import resource
else:
    resource = None


CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_ROOT = Path("/proc")
GIB = 1024**3


def read_int(path):
    return int(path.read_text(encoding="ascii").strip())


class CgroupCounters:
    def __init__(self, root=CGROUP_ROOT):
        # Some hosts expose the host cgroup tree, while a private cgroup
        # namespace mounts this container's group directly at root.
        if root == CGROUP_ROOT:
            try:
                entries = (PROC_ROOT / "self/cgroup").read_text(encoding="ascii")
                relative = next(
                    line.split("::", 1)[1].lstrip("/")
                    for line in entries.splitlines()
                    if line.startswith("0::")
                )
                candidate = root / relative
                if (candidate / "cpu.stat").is_file() and (
                    candidate / "memory.current"
                ).is_file():
                    root = candidate
            except (OSError, StopIteration):
                pass
        self.root = root
        if (root / "cpu.stat").is_file() and (root / "memory.current").is_file():
            self.version = 2
        elif (root / "cpuacct/cpuacct.usage").is_file() and (
            root / "memory/memory.usage_in_bytes"
        ).is_file():
            self.version = 1
        else:
            raise FileNotFoundError("CPU and memory cgroup counters are unavailable")

    def read(self):
        if self.version == 2:
            cpu_stat = (self.root / "cpu.stat").read_text(encoding="ascii")
            usage = next(
                int(line.split()[1])
                for line in cpu_stat.splitlines()
                if line.startswith("usage_usec ")
            )
            return usage / 1_000_000, read_int(self.root / "memory.current")
        return (
            read_int(self.root / "cpuacct/cpuacct.usage") / 1_000_000_000,
            read_int(self.root / "memory/memory.usage_in_bytes"),
        )


def in_container():
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    try:
        cgroup = (PROC_ROOT / "self/cgroup").read_text(encoding="ascii")
    except OSError:
        return False
    return any(name in cgroup for name in ("docker", "kubepods", "containerd"))


def read_processes(proc_root=PROC_ROOT):
    """Return PID, parent, start time, CPU ticks, and RSS pages for live PIDs."""
    processes = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="ascii")
            fields = stat[stat.rfind(")") + 2 :].split()
            processes[int(entry.name)] = (
                int(fields[1]),
                int(fields[19]),
                int(fields[11]) + int(fields[12]),
                int(fields[21]),
            )
        except (OSError, ValueError, IndexError):
            continue  # A short-lived process can disappear during the scan.
    return processes


class ProcessTreeCounters:
    def __init__(self, root_pid):
        self.root_pid = root_pid
        self.last_ticks = {}
        self.cpu_seconds = 0.0
        self.ticks_per_second = os.sysconf("SC_CLK_TCK")
        self.page_size = os.sysconf("SC_PAGE_SIZE")

    def read(self):
        processes = read_processes()
        descendants = {self.root_pid}
        while True:
            new = {
                pid
                for pid, (parent, *_rest) in processes.items()
                if parent in descendants and pid not in descendants
            }
            if not new:
                break
            descendants.update(new)

        ticks = {}
        rss_pages = 0
        for pid in descendants:
            if pid not in processes:
                continue
            _parent, start, cpu_ticks, rss = processes[pid]
            ticks[(pid, start)] = cpu_ticks
            rss_pages += rss
            self.cpu_seconds += max(
                0, cpu_ticks - self.last_ticks.get((pid, start), 0)
            ) / self.ticks_per_second
        self.last_ticks = ticks
        return self.cpu_seconds, rss_pages * self.page_size, len(ticks)


def sample(now, started, cpu_seconds, memory_bytes, process_rss_bytes, process_count, previous):
    elapsed = now - started
    cpu_cores = 0.0
    if previous is not None and now > previous[0]:
        cpu_cores = max(0.0, cpu_seconds - previous[1]) / (now - previous[0])
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "cpu_cores": round(cpu_cores, 3),
        "memory_bytes": memory_bytes,
        "process_rss_bytes": process_rss_bytes,
        "process_count": process_count,
    }


def write_summary(path, summary):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(
            f"\n### Resource use: {summary['phase']}\n\n"
            "| Scope | Duration | CPU time | Average CPU | Peak CPU | "
            "Peak RAM | Peak process RSS | Exit |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
            f"| {summary['scope']} | {summary['duration_seconds']:.1f} s | "
            f"{summary['cpu_seconds']:.1f} s | "
            f"{summary['average_cpu_cores']:.2f} cores | "
            f"{summary['peak_cpu_cores']:.2f} cores | "
            f"{summary['peak_memory_bytes'] / GIB:.2f} GiB | "
            f"{summary['peak_process_rss_bytes'] / GIB:.2f} GiB | "
            f"{summary['exit_code']} |\n\n"
        )


def run(args):
    if resource is None:
        raise OSError("resource monitoring requires Linux")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("a command is required after --")
    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    cgroup = None
    if args.scope == "cgroup" or (args.scope == "auto" and in_container()):
        try:
            cgroup = CgroupCounters()
        except (OSError, ValueError):
            if args.scope == "cgroup":
                raise
    scope = "cgroup" if cgroup else "process tree"
    cgroup_start_cpu = cgroup.read()[0] if cgroup else 0.0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    started = time.monotonic()
    child_usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    child = subprocess.Popen(command, start_new_session=True)
    processes = ProcessTreeCounters(child.pid)

    def forward_signal(signum, _frame):
        if child.poll() is None:
            os.killpg(child.pid, signum)

    old_term = signal.signal(signal.SIGTERM, forward_signal)
    old_int = signal.signal(signal.SIGINT, forward_signal)
    try:
        with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=(
                    "timestamp",
                    "elapsed_seconds",
                    "cpu_cores",
                    "memory_bytes",
                    "process_rss_bytes",
                    "process_count",
                ),
            )
            writer.writeheader()
            previous = None
            while True:
                now = time.monotonic()
                process_cpu, process_rss, count = processes.read()
                if cgroup:
                    cgroup_cpu, memory = cgroup.read()
                    cpu_seconds = max(0.0, cgroup_cpu - cgroup_start_cpu)
                else:
                    cpu_seconds, memory = process_cpu, process_rss
                row = sample(now, started, cpu_seconds, memory, process_rss, count, previous)
                writer.writerow(row)
                csv_file.flush()  # Keep samples if CI cancels the job.
                samples.append(row)
                previous = (now, cpu_seconds)
                print(
                    f"[{args.phase}] CPU {row['cpu_cores']:.2f} cores, "
                    f"RAM {memory / GIB:.2f} GiB, process RSS {process_rss / GIB:.2f} GiB",
                    flush=True,
                )
                try:
                    child.wait(timeout=args.interval)
                    break
                except subprocess.TimeoutExpired:
                    pass
            # Capture the final CPU counter even when the command is shorter
            # than one sampling interval.
            now = time.monotonic()
            process_cpu, process_rss, count = processes.read()
            if cgroup:
                cgroup_cpu, memory = cgroup.read()
                cpu_seconds = max(0.0, cgroup_cpu - cgroup_start_cpu)
            else:
                cpu_seconds, memory = process_cpu, process_rss
            row = sample(now, started, cpu_seconds, memory, process_rss, count, previous)
            writer.writerow(row)
            samples.append(row)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)

    child_usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = (
        child_usage_after.ru_utime
        + child_usage_after.ru_stime
        - child_usage_before.ru_utime
        - child_usage_before.ru_stime
    )
    duration = time.monotonic() - started
    cpu_seconds = max(cpu_seconds, child_cpu)
    summary = {
        "phase": args.phase,
        "scope": scope,
        "exit_code": child.returncode if child.returncode >= 0 else 128 - child.returncode,
        "duration_seconds": round(duration, 3),
        "cpu_seconds": round(cpu_seconds, 3),
        "average_cpu_cores": round(cpu_seconds / duration, 3) if duration else 0,
        "peak_cpu_cores": max(row["cpu_cores"] for row in samples),
        "peak_memory_bytes": max(row["memory_bytes"] for row in samples),
        "peak_process_rss_bytes": max(row["process_rss_bytes"] for row in samples),
        "sample_count": len(samples),
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_summary(os.environ.get("GITHUB_STEP_SUMMARY"), summary)
    print(
        f"[{args.phase}] peak RAM {summary['peak_memory_bytes'] / GIB:.2f} GiB, "
        f"CPU time {summary['cpu_seconds']:.1f} s over {duration:.1f} s",
        flush=True,
    )
    return summary["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, help="Name shown in the CI summary")
    parser.add_argument("--output", required=True, help="Summary JSON path; CSV uses the same stem")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between samples")
    parser.add_argument("--scope", choices=("auto", "cgroup", "process"), default="auto")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
