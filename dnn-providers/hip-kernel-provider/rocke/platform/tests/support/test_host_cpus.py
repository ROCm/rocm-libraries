# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.host_cpus.available_cpus takes the narrowest of the process's CPU
affinity and every CPU quota of the cgroups it belongs to, nested cgroups and a
cgroup mounted below its hierarchy root included. Each case writes a cgroup tree
and the process's cgroup and mount tables, and fixes the affinity, or writes
only the mount and reads it as the process's own cgroup."""

from __future__ import annotations

import sys

import pytest
from rocke.tests import host_cpus


@pytest.mark.skipif(sys.platform != "linux", reason="Linux cgroup contract")
def test_available_cpus_honors_nested_v2_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(host_cpus.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    monkeypatch.setattr(host_cpus.os, "cpu_count", lambda: 64)
    member = tmp_path / "team" / "job"
    member.mkdir(parents=True)
    (tmp_path / "cpu.max").write_text("max 100000")
    (member.parent / "cpu.max").write_text("300000 100000")
    (member / "cpu.max").write_text("150000 100000")
    proc = tmp_path / "self.cgroup"
    proc.write_text("0::/team/job\n")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"1 2 0:1 / {tmp_path} rw - cgroup2 cgroup rw\n")
    assert host_cpus.available_cpus(tmp_path, proc, mountinfo) == 2


@pytest.mark.skipif(sys.platform != "linux", reason="Linux cgroup contract")
def test_available_cpus_honors_nested_v1_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(host_cpus.os, "sched_getaffinity", lambda _pid: set(range(8)))
    monkeypatch.setattr(host_cpus.os, "cpu_count", lambda: 64)
    mount = tmp_path / "cpu,cpuacct"
    member = mount / "batch" / "job"
    member.mkdir(parents=True)
    (member.parent / "cpu.cfs_quota_us").write_text("100000")
    (member.parent / "cpu.cfs_period_us").write_text("100000")
    (member / "cpu.cfs_quota_us").write_text("800000")
    (member / "cpu.cfs_period_us").write_text("100000")
    proc = tmp_path / "self.cgroup"
    proc.write_text("2:cpu,cpuacct:/batch/job\n")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"1 2 0:1 / {mount} rw - cgroup cgroup rw,cpu,cpuacct\n")
    assert host_cpus.available_cpus(tmp_path, proc, mountinfo) == 1


@pytest.mark.skipif(sys.platform != "linux", reason="Linux cgroup contract")
@pytest.mark.parametrize("version", (1, 2))
def test_available_cpus_rebases_non_root_cgroup_mount(version, tmp_path, monkeypatch):
    monkeypatch.setattr(host_cpus.os, "sched_getaffinity", lambda _pid: set(range(8)))
    monkeypatch.setattr(host_cpus.os, "cpu_count", lambda: 64)
    mount = tmp_path / "mounted"
    member = mount / "team" / "job"
    member.mkdir(parents=True)
    if version == 2:
        (member / "cpu.max").write_text("150000 100000")
        cgroup_line = "0::/tenant.slice/team/job\n"
        mount_line = f"1 2 0:1 /tenant.slice {mount} rw - cgroup2 cgroup rw\n"
    else:
        (member / "cpu.cfs_quota_us").write_text("150000")
        (member / "cpu.cfs_period_us").write_text("100000")
        cgroup_line = "2:cpu,cpuacct:/tenant.slice/team/job\n"
        mount_line = (
            f"1 2 0:1 /tenant.slice {mount} rw - cgroup cgroup rw,cpu,cpuacct\n"
        )
    proc = tmp_path / "self.cgroup"
    proc.write_text(cgroup_line)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(mount_line)
    assert host_cpus.available_cpus(tmp_path, proc, mountinfo) == 2


@pytest.mark.skipif(sys.platform != "linux", reason="Linux affinity contract")
def test_available_cpus_honors_affinity_without_a_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(host_cpus.os, "sched_getaffinity", lambda _pid: {2, 3})
    monkeypatch.setattr(host_cpus.os, "cpu_count", lambda: 64)
    assert (
        host_cpus.available_cpus(
            tmp_path, tmp_path / "missing-cgroup", tmp_path / "missing-mountinfo"
        )
        == 2
    )


def _cgroup_root(tmp_path, files: dict[str, str]):
    """A cgroup mount at ``tmp_path`` holding ``files``, for a process whose own
    cgroup and mount tables are absent: the mount itself is read."""
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path, tmp_path / "no-proc-cgroup", tmp_path / "no-mountinfo"


@pytest.mark.parametrize(
    "files",
    [
        {"cpu.max": "100000 100000"},
        {"cpu/cpu.cfs_quota_us": "100000", "cpu/cpu.cfs_period_us": "100000"},
    ],
    ids=["v2", "v1"],
)
def test_a_cpu_quota_caps_the_count(tmp_path, files):
    # "<= 1", not "== 1": affinity also feeds the minimum, so a quota can only
    # ever lower the answer.
    assert host_cpus.available_cpus(*_cgroup_root(tmp_path, files)) <= 1


def test_a_fractional_quota_still_leaves_one_cpu(tmp_path):
    """Half a CPU is a rate, not half a process: rounding down would size a
    pool of zero workers, a hang rather than a slow run."""
    root = _cgroup_root(tmp_path, {"cpu.max": "50000 100000"})
    assert host_cpus.available_cpus(*root) == 1


@pytest.mark.parametrize(
    "files",
    [
        {"cpu.max": "max 100000"},
        {"cpu/cpu.cfs_quota_us": "-1", "cpu/cpu.cfs_period_us": "100000"},
        {},
    ],
    ids=["v2-max", "v1-negative", "absent"],
)
def test_no_quota_does_not_lower_the_count(tmp_path, files):
    """``max`` and a negative v1 quota both mean unlimited, not a cap of zero."""
    root = _cgroup_root(tmp_path, files)
    unlimited = _cgroup_root(tmp_path / "empty", {})
    assert host_cpus.available_cpus(*root) == host_cpus.available_cpus(*unlimited)


@pytest.mark.parametrize("files", [{}, {"cpu.max": "garbage"}, {"cpu.max": "0 0"}])
def test_the_count_is_always_a_usable_worker_count(tmp_path, files):
    assert host_cpus.available_cpus(*_cgroup_root(tmp_path, files)) >= 1
