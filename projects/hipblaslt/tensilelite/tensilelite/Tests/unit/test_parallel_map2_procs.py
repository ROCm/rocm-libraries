# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import importlib

import pytest

from tensilelite.Common.GlobalParameters import globalParameters

# `import tensilelite.Common.Parallel` resolves to the joblib `Parallel` class re-exported into the
# Common package namespace; load the real module.
P = importlib.import_module("tensilelite.Common.Parallel")

pytestmark = pytest.mark.unit


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setitem(globalParameters, "CpuThreads", -1)

    def setHost(affinity, cpuThreads=-1):
        monkeypatch.setattr(P.os, "sched_getaffinity", lambda pid: set(range(affinity)))
        monkeypatch.setitem(globalParameters, "CpuThreads", cpuThreads)

    return setHost


@pytest.fixture
def poolSizes(monkeypatch):
    """Pool sizes ParallelMap2 hands joblib, one entry per call; no task is run."""
    sizes = []

    class RecordingParallel:
        def __init__(self, n_jobs, **kwargs):
            sizes.append(n_jobs)

        def __call__(self, tasks):
            return [None for _ in tasks]

    monkeypatch.setattr(P, "Parallel", RecordingParallel)
    monkeypatch.setitem(globalParameters, "ShowProgressBar", False)
    return sizes


@pytest.mark.parametrize(
    "affinity, cpuThreads, procs, expected",
    [
        (64, -1, 4, 4),  # below the budget: honoured (TensileLogic's --jobs)
        (64, -1, 64, 64),
        (64, 8, 16, 8),  # CpuThreads still caps
        (64, -1, None, 64),  # no procs: the budget
        (64, -1, 0, 64),
        (64, -1, -1, 64),
    ],
)
def test_parallel_map2_takes_procs_up_to_the_budget(
    host, poolSizes, capsys, affinity, cpuThreads, procs, expected
):
    host(affinity, cpuThreads=cpuThreads)
    # At least as many tasks as the largest pool: ParallelMap2 never starts more workers than tasks.
    tasks = [1] * 64
    assert P.ParallelMap2(abs, tasks, multiArg=False, procs=procs) == [None] * len(tasks)
    assert poolSizes == [expected]
    assert f"Launching {expected} threads for {len(tasks)} tasks" in capsys.readouterr().out


def test_parallel_map2_disabled_ignores_procs(host, poolSizes):
    host(64)
    P.ParallelMap2(abs, [1, 2], multiArg=False, enable=False, procs=4)
    assert poolSizes == [1]
