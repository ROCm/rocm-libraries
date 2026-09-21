# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# What this host will actually let us have, as opposed to what it looks like it
# has. Shared by the tools under this directory that size a worker pool.
#
# Not a CLI entry point -- hence the leading underscore, since every other
# module here is a script.

from __future__ import annotations

import os
from pathlib import Path


def available_cpus(cgroup_root: Path = Path("/sys/fs/cgroup")) -> int:
    """CPUs this process may actually use, not the CPUs the host has.

    `os.cpu_count()` reports the machine, and on the hosts this tree is built
    on that is the wrong number by orders of magnitude -- a scheduler-pinned
    node reports 384 against an affinity of 2, and a container on a large host
    reports the host. Pools sized from it oversubscribe by the same factor.

    Three sources, narrowest first. Affinity covers cpuset pinning and
    scheduler placement; `cpu.max` covers a CFS quota (cgroup v2, then v1) --
    a quota is a *rate*, so a fractional share rounds up to one whole worker
    rather than down to zero, which would be a hang rather than a slow run.

    `cgroup_root` is a parameter so the quota branch is reachable from a test.
    Neither the build hosts nor the probe containers impose a quota, so left
    hardcoded it would be the one path nothing ever executes.
    """
    counts = []

    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            counts.append(len(getaffinity(0)))
        except OSError:
            pass

    for quota_path, period_path in (
        (cgroup_root / "cpu.max", None),
        (
            cgroup_root / "cpu" / "cpu.cfs_quota_us",
            cgroup_root / "cpu" / "cpu.cfs_period_us",
        ),
    ):
        try:
            if period_path is None:
                quota_s, period_s = quota_path.read_text().split()[:2]
            else:
                quota_s = quota_path.read_text().strip()
                period_s = period_path.read_text().strip()
            # "max" (v2) and a negative quota (v1) both mean unlimited.
            if quota_s == "max":
                continue
            quota, period = int(quota_s), int(period_s)
            if quota > 0 and period > 0:
                counts.append(max(1, -(-quota // period)))
        except (OSError, ValueError):
            continue

    counts.append(os.cpu_count() or 4)
    return max(1, min(counts))
