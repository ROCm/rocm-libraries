################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Common.Parallel``: thread-count logic,
global-parameter overwrite, the pcall wrappers, ``apply_print_exception``, and
the single-threaded / n_jobs=1 (in-process) paths of ``ParallelMap`` /
``ParallelMap2``. The real fork/spawn process-pool paths are not unit-driven
(flaky/slow) — see the suite's target.md."""

import contextlib
import importlib

import pytest

from Tensile.Common.GlobalParameters import globalParameters

# `import Tensile.Common.Parallel as P` resolves to the joblib `Parallel` class
# (re-exported into the Common package namespace, shadowing the submodule). Load
# the real module via import_module.
P = importlib.import_module("Tensile.Common.Parallel")

pytestmark = pytest.mark.unit


@contextlib.contextmanager
def _gp(**over):
    # Snapshot+restore the ENTIRE dict: ParallelMap2's joblib n_jobs=1 in-process
    # path calls OverwriteGlobalParameters(globalParameters), which does
    # clear()+update(self) and empties the live dict. Full restore keeps the
    # shared session intact regardless of such side effects.
    saved = dict(globalParameters)
    globalParameters.update(over)
    try:
        yield
    finally:
        globalParameters.clear()
        globalParameters.update(saved)


def _double(x):
    return x * 2


def _add(a, b):
    return a + b


# ---------------------------------------------------------------------------

def test_joblib_supports_generator_returns_bool():
    assert isinstance(P.joblibParallelSupportsGenerator(), bool)


def test_cpu_thread_count_disabled():
    assert P.CPUThreadCount(enable=False) == 1


def test_cpu_thread_count_all_cores():
    with _gp(CpuThreads=-1):
        assert P.CPUThreadCount(enable=True) >= 1


def test_cpu_thread_count_capped():
    with _gp(CpuThreads=1):
        assert P.CPUThreadCount(enable=True) == 1


def test_overwrite_global_parameters():
    from Tensile.Common import GlobalParameters as GPmod
    saved = dict(GPmod.globalParameters)
    try:
        P.OverwriteGlobalParameters({"OnlyKey": 1})
        assert dict(GPmod.globalParameters) == {"OnlyKey": 1}
    finally:
        GPmod.globalParameters.clear()
        GPmod.globalParameters.update(saved)


def test_pcall_multi_and_single_arg():
    from Tensile.Common import GlobalParameters as GPmod
    saved = dict(GPmod.globalParameters)
    try:
        assert P.pcallWithGlobalParamsMultiArg(_add, (2, 3), dict(saved)) == 5
        assert P.pcallWithGlobalParamsSingleArg(_double, 4, dict(saved)) == 8
    finally:
        GPmod.globalParameters.clear()
        GPmod.globalParameters.update(saved)


def test_apply_print_exception_paths():
    # args>0 form: item is the func, args[0] are its args.
    assert P.apply_print_exception(_add, (2, 3)) == 5
    # no-args form: item is (func, value).
    assert P.apply_print_exception((_double, 5)) == 10


def test_apply_print_exception_reraises(capsys):
    def boom(_):
        raise ValueError("x")
    with pytest.raises(ValueError):
        P.apply_print_exception((boom, 1))


def test_parallel_map_single_threaded_progressbar():
    with _gp(CpuThreads=1, ShowProgressBar=True):
        rv = P.ParallelMap(_double, [1, 2, 3], enable=False)
    assert list(rv) == [2, 4, 6]


def test_parallel_map_dummy_pool_no_progressbar():
    # ShowProgressBar False + threadCount 1 -> dummy (thread) pool .map, wrapped
    # with apply_print_exception; in-process, no fork.
    with _gp(CpuThreads=1, ShowProgressBar=False):
        rv = P.ParallelMap(_double, [1, 2, 3], enable=False)
    assert list(rv) == [2, 4, 6]


def test_parallel_map2_single_threaded_progressbar_multiarg():
    with _gp(CpuThreads=1, ShowProgressBar=True):
        rv = P.ParallelMap2(_add, [(1, 2), (3, 4)], multiArg=True)
    assert list(rv) == [3, 7]


def test_parallel_map2_single_threaded_progressbar_singlearg():
    with _gp(CpuThreads=1, ShowProgressBar=True):
        rv = P.ParallelMap2(_double, [1, 2, 3], multiArg=False)
    assert list(rv) == [2, 4, 6]


def test_parallel_map2_joblib_njobs1_in_process():
    # ShowProgressBar False + threadCount 1 -> joblib Parallel(n_jobs=1), which
    # runs sequentially in-process (no fork).
    with _gp(CpuThreads=1, ShowProgressBar=False):
        rv = P.ParallelMap2(_add, [(1, 2), (10, 20)], multiArg=True)
    assert list(rv) == [3, 30]
