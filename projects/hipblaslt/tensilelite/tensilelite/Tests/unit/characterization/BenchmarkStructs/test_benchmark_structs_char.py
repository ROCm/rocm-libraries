################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.BenchmarkStructs``: the pure parameter
helpers, the fork-permutation cartesian product, and ``BenchmarkStep``. The
``BenchmarkProcess`` config->steps integration builder needs full benchmark
configs (see target.md)."""

import importlib
import types

import pytest

B = importlib.import_module("Tensile.BenchmarkStructs")

pytestmark = pytest.mark.unit


def test_get_defaults_for_missing_parameters(snapshot):
    paramList = [{"A": [1]}]
    defaults = [{"A": [9]}, {"B": [2]}, {"ProblemSizes": [[1]]}]
    # A present -> skipped; B missing -> included; ProblemSizes always included.
    assert B.getDefaultsForMissingParameters(paramList, defaults) == snapshot


def test_separate_parameters(snapshot):
    single, multi = B.separateParameters({"S": [1], "M": [1, 2, 3], "ProblemSizes": [[1], [2]]})
    assert {"single": single, "multi": multi} == snapshot


def test_separate_parameters_none_exits():
    with pytest.raises(SystemExit):
        B.separateParameters({"bad": None})


def test_check_cd_buffer_ok():
    pt = {"OperationType": "GEMM", "IndexAssignmentsLD": [3, 4, 5, 6]}
    ps = types.SimpleNamespace(problems=[types.SimpleNamespace(sizes=(0, 0, 0, 7, 7, 0, 0))])
    B.checkCDBufferAndStrides(pt, ps, isCEqualD=True)  # ldd == ldc -> no exit


def test_check_cd_buffer_mismatch_exits():
    pt = {"OperationType": "GEMM", "IndexAssignmentsLD": [3, 4, 5, 6]}
    ps = types.SimpleNamespace(problems=[types.SimpleNamespace(sizes=(0, 0, 0, 7, 9, 0, 0))])
    with pytest.raises(SystemExit):
        B.checkCDBufferAndStrides(pt, ps, isCEqualD=True)


def test_check_cd_buffer_not_cequald():
    pt = {"OperationType": "GEMM", "IndexAssignmentsLD": [3, 4, 5, 6]}
    ps = types.SimpleNamespace(problems=[])
    B.checkCDBufferAndStrides(pt, ps, isCEqualD=False)  # no-op


def test_fork_permutations(snapshot):
    fork = {"X": [1, 2], "Y": [3]}
    groups = [[{"G": "a"}, {"G": "b"}]]
    cfp = B.constructForkPermutations(fork, groups)
    perms = list(cfp)
    assert {"len": len(cfp), "num_perms": len(perms), "perms": perms} == snapshot


def test_fork_permutations_next():
    cfp = B.constructForkPermutations({"X": [1, 2]}, [])
    first = next(cfp)
    assert "X" in first


def test_lazy_fork_permutations(snapshot):
    perms = list(B.constructLazyForkPermutations({"A": [1, 2]}, [[{"g": 0}, {"g": 1}]]))
    assert sorted(perms, key=lambda d: (d["A"], d["g"])) == snapshot


def _step(**over):
    kw = dict(forkParams={}, constantParams={}, paramGroups=[], customKernels=[],
              internalSupportParams={}, problemSizes=types.SimpleNamespace(totalProblemSizes=5),
              biasTypeArgs=None, factorDimArgs=None, activationArgs=None,
              icacheFlushArgs=None, idx=0)
    kw.update(over)
    return B.BenchmarkStep(**kw)


def test_benchmark_step_basic(snapshot):
    step = _step(idx=3)
    assert {"str": str(step), "repr": repr(step), "isFinal": step.isFinal(),
            "wildcard": step.customKernelWildcard} == snapshot


def test_benchmark_step_custom_kernel_wildcard(monkeypatch):
    monkeypatch.setattr(B, "getAllCustomKernelNames", lambda: ["k1", "k2"])
    step = _step(customKernels=["*"])
    assert step.customKernelWildcard is True
    assert step.customKernels == ["k1", "k2"]
