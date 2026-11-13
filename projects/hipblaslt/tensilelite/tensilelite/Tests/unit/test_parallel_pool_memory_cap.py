# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Memory cap of ``tensilelite.Common.Parallel.poolSizeBudget``: the pools of all pytest-xdist
workers stay within POOL_MEMORY_FRACTION of the cgroup memory limit. The cgroup files live in a
temporary directory; affinity, loky's cpu_count and the environment are mocked."""

import importlib

import pytest

from tensilelite.Common.GlobalParameters import globalParameters

# `import tensilelite.Common.Parallel` resolves to the joblib `Parallel` class re-exported into the
# Common package namespace; load the real module.
P = importlib.import_module("tensilelite.Common.Parallel")

pytestmark = pytest.mark.unit

GIB = 2**30


def writeCgroup(tmp_path, selfCgroup, files):
    """A fake cgroup root under tmp_path: /proc/self/cgroup content and files relative to it."""
    root = tmp_path / "cgroup"
    for relative, value in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    proc = tmp_path / "self_cgroup"
    proc.write_text(selfCgroup)
    return str(root), str(proc)


@pytest.mark.parametrize(
    "selfCgroup, files, expected",
    [
        ("0::/\n", {"memory.max": "118111600640\n"}, 110 * GIB),  # v2, container's own namespace
        ("0::/\n", {"memory.max": "max\n"}, None),  # v2, no limit
        ("0::/job.slice\n", {"job.slice/memory.max": "8589934592\n"}, 8 * GIB),  # v2, nested
        ("0::/gone\n", {"memory.max": "4294967296\n"}, 4 * GIB),  # nested path absent: root
        ("4:memory:/\n", {"memory/memory.limit_in_bytes": "17179869184\n"}, 16 * GIB),  # v1
        ("4:memory:/\n", {"memory/memory.limit_in_bytes": "9223372036854771712\n"}, None),  # v1 none
        ("0::/\n", {}, None),  # no memory controller files
        ("0::/\n", {"memory.max": "garbage\n"}, None),
    ],
)
def test_cgroup_memory_limit(tmp_path, selfCgroup, files, expected):
    root, proc = writeCgroup(tmp_path, selfCgroup, files)
    assert P._cgroupMemoryLimit(root=root, selfCgroup=proc) == expected


def test_cgroup_memory_limit_without_proc_self_cgroup(tmp_path):
    root, _ = writeCgroup(tmp_path, "", {"memory.max": "1073741824\n"})
    assert P._cgroupMemoryLimit(root=root, selfCgroup=str(tmp_path / "missing")) == GIB


def test_cgroup_memory_limit_is_none_on_windows(monkeypatch):
    monkeypatch.setattr(P.os, "name", "nt")
    assert P._cgroupMemoryLimit() is None


@pytest.fixture
def host(monkeypatch):
    """Mocked host: affinity CPUs (loky agrees), pytest-xdist workers and cgroup memory limit."""
    monkeypatch.delenv("KUBE_CPU_REQUEST", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER_COUNT", raising=False)
    monkeypatch.setitem(globalParameters, "CpuThreads", -1)

    def setHost(cpus, workers=None, memory=None):
        monkeypatch.setattr(P.os, "sched_getaffinity", lambda pid: set(range(cpus)))
        monkeypatch.setattr(P, "lokyCpuCount", lambda: cpus)
        monkeypatch.setattr(P, "_cgroupMemoryLimit", lambda: memory)
        if workers is not None:
            monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", str(workers))

    return setHost


@pytest.mark.parametrize(
    "cpus, workers, memory, expected",
    [
        (256, 4, None, 128),  # no memory limit: the CPU budget, unchanged
        (256, 4, 110 * GIB, 75),  # 256-CPU host, 110 GiB container: memory caps 128 to 75
        (64, 4, 110 * GIB, 32),  # CPUs pinned to 64: the CPU budget is already lower
        (25, 4, 32 * GIB, 12),  # Math CI / TheRock pod: CPU budget below the memory cap
        (96, None, 8 * GIB, 21),  # standalone, 8 GiB container
        (64, 16, GIB, 1),  # tiny limit: never below one process
    ],
)
def test_budget_is_capped_by_memory(host, cpus, workers, memory, expected):
    host(cpus, workers=workers, memory=memory)
    assert P.poolSizeBudget() == expected
    assert P.CPUThreadCount() == expected


def test_all_pools_fit_the_memory_share(host):
    host(256, workers=4, memory=110 * GIB)
    total = P.poolSizeBudget() * P.concurrentWorkerCount() * P.POOL_PROCESS_PEAK_BYTES
    assert total <= P.POOL_MEMORY_FRACTION * 110 * GIB


def test_cpu_threads_still_caps_below_the_memory_cap(host):
    host(256, workers=4, memory=110 * GIB)
    globalParameters["CpuThreads"] = 8
    assert P.CPUThreadCount() == 8


def test_create_library_groups_share_the_capped_budget(host):
    """TensileCreateLibrary splits the same budget across its architecture groups."""
    from tensilelite.create_library import run as Run

    host(256, workers=4, memory=110 * GIB)
    assert Run._cpuCount() == 75
