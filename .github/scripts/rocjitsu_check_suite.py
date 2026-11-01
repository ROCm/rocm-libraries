# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fail when a rocjitsu suite did not finish, and record container memory state.

The supervisor marks incomplete runs itself, but if the whole step is killed it
writes nothing, and the runner can still report the killed step as successful.
"""

import argparse
import json
from pathlib import Path
import sys

MEMORY_FILES = ("memory.max", "memory.peak", "memory.current", "memory.events")


def cgroup_dir(cgroup_root):
    # Without a private cgroup namespace the memory files live in our own cgroup.
    if (cgroup_root / "memory.max").is_file():
        return cgroup_root
    own = Path("/proc/self/cgroup")
    for line in own.read_text().splitlines() if own.is_file() else ():
        if line.startswith("0::"):
            candidate = cgroup_root / line[3:].lstrip("/")
            if (candidate / "memory.max").is_file():
                return candidate
    return cgroup_root


def memory_lines(cgroup_root):
    cgroup_root = cgroup_dir(cgroup_root)
    lines = []
    for name in MEMORY_FILES:
        path = cgroup_root / name
        if path.is_file():
            for value in path.read_text().split("\n"):
                if value.strip():
                    lines.append(f"{name}: {value.strip()}")
    return lines or [f"no cgroup v2 memory files under {cgroup_root}"]


def oom_kills(lines):
    for line in lines:
        name, _, value = line.partition(": ")
        if name == "memory.events" and value.startswith("oom_kill "):
            return int(value.split()[1])
    return 0


def problems(report_dir):
    found = []
    progress_path = report_dir / "progress.json"
    if not progress_path.is_file():
        found.append("progress.json is missing; no test progress was recorded")
    else:
        progress = json.loads(progress_path.read_text())
        if progress.get("incomplete"):
            found.append(
                f"{progress['incomplete']} of {progress.get('selected')} tests did not finish"
            )
    execution_path = report_dir / "execution.json"
    if not execution_path.is_file():
        found.append(
            "execution.json is missing; the suite supervisor did not exit normally"
        )
    elif not json.loads(execution_path.read_text()).get("complete"):
        found.append("the suite supervisor reported an incomplete run")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    args = parser.parse_args()

    lines = memory_lines(args.cgroup_root)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "container-memory.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    kills = oom_kills(lines)
    if kills:
        print(f"::warning::the container recorded {kills} OOM kill(s)")

    found = problems(args.report_dir)
    for problem in found:
        print(f"::error::{problem}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
