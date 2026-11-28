# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""How many CPUs a test process may use: the size of a worker pool.

``os.cpu_count()`` reports the machine. A scheduler-pinned job, a container on
a large host or a pod with a CPU quota may use far fewer CPUs, and a pool sized
from the machine oversubscribes them by the same factor. :func:`available_cpus`
takes the narrowest of the process's CPU affinity and every CPU quota of the
cgroups it belongs to (v2 ``cpu.max``, v1 ``cpu.cfs_quota_us``), each nested
cgroup included. A quota counts as the whole CPUs it grants, rounded down and
never below one: the rule the descriptor packer's pool uses
(``hkp_pack.pipeline._read_cpu_max``), so both pools agree under one quota.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["available_cpus"]


def _ancestors(path: Path, root: Path) -> list[Path]:
    """Return ``path`` through ``root`` without escaping the cgroup mount."""
    result = []
    current = path
    while current == root or root in current.parents:
        result.append(current)
        if current == root:
            break
        current = current.parent
    return result


def _cgroup_mounts(
    cgroup_root: Path, mountinfo: Path
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Return ``(hierarchy root, mount point)`` for v2 and CPU v1."""
    v2 = [(Path("/"), cgroup_root)]
    v1 = [
        (Path("/"), cgroup_root / "cpu"),
        (Path("/"), cgroup_root / "cpu,cpuacct"),
        (Path("/"), cgroup_root / "cpuacct,cpu"),
    ]
    try:
        lines = mountinfo.read_text(encoding="utf-8").splitlines()
    except OSError:
        return v2, v1
    discovered_v2 = []
    discovered_v1 = []
    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        filesystem = after.split()
        if len(fields) < 5 or len(filesystem) < 3:
            continue
        hierarchy_root = Path(fields[3])
        mount = Path(fields[4])
        kind = filesystem[0]
        options = set(filesystem[2].split(","))
        if kind == "cgroup2":
            discovered_v2.append((hierarchy_root, mount))
        elif kind == "cgroup" and "cpu" in options:
            discovered_v1.append((hierarchy_root, mount))
    return discovered_v2 or v2, discovered_v1 or v1


def _process_cgroup_dirs(
    cgroup_root: Path, proc_cgroup: Path, mountinfo: Path
) -> tuple[list[Path], list[Path]]:
    """Return active cgroup-v2 and CPU-controller v1 directories."""
    v2_mounts, v1_mounts = _cgroup_mounts(cgroup_root, mountinfo)
    v2 = [mount for _root, mount in v2_mounts]
    v1 = [mount for _root, mount in v1_mounts]
    try:
        lines = proc_cgroup.read_text(encoding="utf-8").splitlines()
    except OSError:
        return v2, v1
    for line in lines:
        try:
            _hierarchy, controllers, member = line.split(":", 2)
        except ValueError:
            continue
        membership = Path(member)
        if not controllers:
            v2 = _mounted_dirs(membership, v2_mounts)
        elif "cpu" in controllers.split(","):
            v1 = _mounted_dirs(membership, v1_mounts)
    return v2, v1


def _mounted_dirs(membership: Path, mounts: list[tuple[Path, Path]]) -> list[Path]:
    """The directories of cgroup ``membership`` under each ``(hierarchy root,
    mount point)`` of ``mounts`` that holds it, each with its ancestors up to
    the mount."""
    directories = []
    for hierarchy_root, mount in mounts:
        try:
            relative = membership.relative_to(hierarchy_root)
        except ValueError:
            continue
        directories.extend(_ancestors(mount / relative, mount))
    return directories


def _whole_cpus(quota: int, period: int) -> int:
    """The worker count a CPU quota of ``quota`` per ``period`` allows: its whole
    CPUs, rounded down so a pool never oversubscribes the quota, and at least one
    so a sub-CPU quota still runs."""
    return max(1, quota // period)


def available_cpus(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    mountinfo: Path = Path("/proc/self/mountinfo"),
) -> int:
    """Return the narrowest CPU limit visible through affinity or cgroups."""
    counts = []
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            counts.append(len(getaffinity(0)))
        except OSError:
            pass

    v2_dirs, v1_dirs = _process_cgroup_dirs(cgroup_root, proc_cgroup, mountinfo)
    for directory in v2_dirs:
        try:
            quota_s, period_s = (directory / "cpu.max").read_text().split()[:2]
            if quota_s != "max":
                quota, period = int(quota_s), int(period_s)
                if quota > 0 and period > 0:
                    counts.append(_whole_cpus(quota, period))
        except (OSError, ValueError):
            continue
    for directory in v1_dirs:
        try:
            quota = int((directory / "cpu.cfs_quota_us").read_text().strip())
            period = int((directory / "cpu.cfs_period_us").read_text().strip())
            if quota > 0 and period > 0:
                counts.append(_whole_cpus(quota, period))
        except (OSError, ValueError):
            continue

    counts.append(os.cpu_count() or 4)
    return max(1, min(counts))
