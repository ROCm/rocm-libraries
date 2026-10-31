# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pool budget of ``Tensile.Common.Parallel.CPUThreadCount``: all pytest-xdist workers together
start at most 2x the available CPUs, a single process never more than 1x. Affinity mask, loky's
cpu_count (cgroup quota) and the environment are mocked."""

import importlib

import pytest

from Tensile.Common.GlobalParameters import globalParameters

# `import Tensile.Common.Parallel` resolves to the joblib `Parallel` class re-exported into the
# Common package namespace; load the real module.
P = importlib.import_module("Tensile.Common.Parallel")

pytestmark = pytest.mark.unit


@pytest.fixture
def host(monkeypatch):
    """Sets the mocked host: affinity CPUs, loky cpu_count, environment and CpuThreads."""
    monkeypatch.delenv("KUBE_CPU_REQUEST", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER_COUNT", raising=False)
    monkeypatch.setitem(globalParameters, "CpuThreads", -1)

    def setHost(affinity, loky=None, kube=None, workers=None, cpuThreads=-1):
        monkeypatch.setattr(P.os, "sched_getaffinity", lambda pid: set(range(affinity)))
        monkeypatch.setattr(P, "lokyCpuCount", lambda: affinity if loky is None else loky)
        if kube is not None:
            monkeypatch.setenv("KUBE_CPU_REQUEST", kube)
        if workers is not None:
            monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", workers)
        monkeypatch.setitem(globalParameters, "CpuThreads", cpuThreads)

    return setHost


@pytest.mark.parametrize(
    "affinity, kube, workers, expected",
    [
        (96, None, None, 96),  # standalone
        (25, None, "4", 12),  # -n 4 on 25 CPUs (TheRock / Math CI)
        (192, "25", "4", 12),  # -n 4, pod sees the whole node, KUBE_CPU_REQUEST=25
        (25, None, "2", 25),  # -n 2 on 25 CPUs: capped at the CPUs
        (128, None, "16", 16),  # -n 16 on 128 CPUs (FFM)
    ],
)
def test_budget_for_ci_setups(host, affinity, kube, workers, expected):
    host(affinity, kube=kube, workers=workers)
    assert P.CPUThreadCount() == expected


def test_cgroup_quota_below_affinity_limits_cpus(host):
    host(64, loky=8)
    assert P.availableCPUCount() == 8
    assert P.CPUThreadCount() == 8


def test_affinity_below_cgroup_quota_limits_cpus(host):
    host(8, loky=64)
    assert P.availableCPUCount() == 8


def test_kube_request_above_cpus_is_ignored(host):
    host(16, kube="25")
    assert P.availableCPUCount() == 16


@pytest.mark.parametrize("kube, expected", [("8", 8), ("8.0", 8), ("8.7", 8), ("0.5", 1)])
def test_kube_request_is_rounded_down_to_at_least_one(host, kube, expected):
    host(64, kube=kube)
    assert P.availableCPUCount() == expected


@pytest.mark.parametrize("kube", ["", "abc", "500m", "0", "-4", "nan", "inf"])
def test_unusable_kube_request_is_ignored(host, kube):
    host(64, kube=kube)
    assert P.availableCPUCount() == 64


@pytest.mark.parametrize("workers", ["", "four", "0", "-2"])
def test_unusable_worker_count_means_one_worker(host, workers):
    host(32, workers=workers)
    assert P.concurrentWorkerCount() == 1
    assert P.CPUThreadCount() == 32


def test_more_workers_than_twice_the_cpus_still_get_one_process(host):
    host(4, workers="16")
    assert P.CPUThreadCount() == 1


@pytest.mark.parametrize(
    "cpuThreads, expected",
    [(-1, 12), (1, 1), (8, 8), (12, 12), (32, 12), (0, 0)],
)
def test_cpu_threads_caps_the_budget(host, cpuThreads, expected):
    host(25, workers="4", cpuThreads=cpuThreads)
    assert P.CPUThreadCount() == expected


def test_disabled_is_one_thread(host):
    host(25, workers="4")
    assert P.CPUThreadCount(enable=False) == 1


def test_windows_caps_at_61_handles_without_loky(host, monkeypatch):
    host(256, loky=1)
    monkeypatch.setattr(P.os, "name", "nt")
    monkeypatch.setattr(P.os, "cpu_count", lambda: 128)
    assert P.availableCPUCount() == 61


@pytest.mark.parametrize(
    "cpus, workers",
    [(1, 1), (3, 2), (25, 4), (25, 3), (64, 7), (128, 16), (256, 32), (7, 13)],
)
def test_all_workers_together_stay_within_twice_the_cpus(host, cpus, workers):
    host(cpus, workers=str(workers))
    perProcess = P.CPUThreadCount()
    assert 1 <= perProcess <= cpus
    assert perProcess * workers <= max(workers, P.OVERSUBSCRIPTION_FACTOR * cpus)
