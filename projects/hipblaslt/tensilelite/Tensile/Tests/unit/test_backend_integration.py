################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import types

import numpy as np
import pytest

from Tensile.backends.base import BackendFactory, OptimizationBackend
import Tensile.backends.ductile_backend as ductile_backend_mod
from Tensile.backends.tensile_backend import TensileBackend
from Tensile.backends.ductile_backend import DuctileBackend

pytestmark = pytest.mark.unit


class _BackendForFactoryTest(OptimizationBackend):
    def run(self, backend_config, benchmark_config, benchmark_runner, cacheValid=False, buildOnly=False):
        return None

    def supports_solution_pool(self):
        return False


def test_backend_factory_rejects_non_backend_class(monkeypatch):
    monkeypatch.setattr(BackendFactory, "_backends", {})
    with pytest.raises(TypeError):
        BackendFactory.register("bad", dict)


def test_backend_factory_create_unknown_raises(monkeypatch):
    monkeypatch.setattr(BackendFactory, "_backends", {})
    with pytest.raises(ValueError, match="Unknown backend"):
        BackendFactory.create("unknown")


def test_backend_factory_register_and_create(monkeypatch):
    monkeypatch.setattr(BackendFactory, "_backends", {})
    BackendFactory.register("test", _BackendForFactoryTest)
    backend = BackendFactory.create("test")
    assert isinstance(backend, _BackendForFactoryTest)
    assert BackendFactory.get_available_backends() == ["test"]


def test_backend_factory_has_expected_backends():
    available = set(BackendFactory.get_available_backends())
    assert "tensile" in available
    if "ductile" in available:
        backend = BackendFactory.create("ductile")
        assert isinstance(backend, DuctileBackend)
    backend = BackendFactory.create("tensile")
    assert isinstance(backend, TensileBackend)


def test_tensile_backend_run_calls_benchmark_runner(monkeypatch):
    backend = TensileBackend()
    calls = {}

    monkeypatch.setattr(
        "Tensile.backends.tensile_backend.constructForkPermutations",
        lambda _fork_params, _param_groups: [{"x": 1}, {"x": 2}],
    )

    import Tensile.BenchmarkProblems as bp

    monkeypatch.setattr(bp, "_generateForkedSolutions", lambda *_args, **_kwargs: ["fork_a", "fork_b"])
    monkeypatch.setattr(bp, "_generateCustomKernelSolutions", lambda *_args, **_kwargs: ["ck_a"])

    def benchmark_runner(solutions, isCached=False, buildOnly=False):
        calls["solutions"] = solutions
        calls["isCached"] = isCached
        calls["buildOnly"] = buildOnly
        return "results.csv", 0

    benchmark_step = types.SimpleNamespace(
        forkParams={"x": [1, 2]},
        paramGroups=[],
        constantParams={},
        customKernels=[],
        internalSupportParams={},
        customKernelWildcard=False,
    )

    benchmark_config = {
        "forkParametersEnabled": True,
        "problemType": object(),
        "assembler": object(),
        "debugConfig": object(),
        "isaInfoMap": {"gfx942": {}},
        "benchmarkStep": benchmark_step,
        "solutionPoolIndex": {},
    }

    backend.run({}, benchmark_config, benchmark_runner, cacheValid=False, buildOnly=False)
    assert calls["solutions"] == ["fork_a", "fork_b", "ck_a"]
    assert calls["isCached"] is False
    assert calls["buildOnly"] is False


def test_ductile_backend_warns_when_cache_or_build_only(monkeypatch, tmp_path):
    warnings = []

    monkeypatch.setattr("Tensile.backends.ductile_backend.printWarning", lambda msg: warnings.append(msg))

    class FakeGA:
        def __init__(self, *args, **kwargs):
            self._evaluate = kwargs["evaluate"]

        def optimize(self):
            return [{"M": 32, "N": 32, "K": 32, "Batch": 1}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _best):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)

    backend = DuctileBackend()

    benchmark_step = types.SimpleNamespace(
        forkParams={
            "DepthU": [32, 64],
            "PrefetchGlobalRead": [1, 2],
            "PrefetchLocalRead": [1],
            "LocalReadVectorWidth": [4, 8],
            "SourceSwap": [0, 1],
            "1LDSBuffer": [0, 1],
        },
        paramGroups=[],
        constantParams={},
    )
    benchmark_config = {
        "forkParametersEnabled": True,
        "problemType": types.SimpleNamespace(state={}),
        "assembler": object(),
        "debugConfig": types.SimpleNamespace(splitGSU=False),
        "isaInfoMap": {"gfx942": {}},
        "benchmarkStep": benchmark_step,
        "rootPath": str(tmp_path),
        "configName": "ductile-test",
        "benchmarkStepIdx": 0,
        "totalBenchmarkSteps": 1,
    }

    backend.run({}, benchmark_config, lambda *_args, **_kwargs: ("unused.csv", 0), cacheValid=True, buildOnly=True)
    assert any("cacheValid is not supported" in msg for msg in warnings)
    assert any("buildOnly is not supported" in msg for msg in warnings)


def test_tensile_backend_supports_solution_pool():
    """Verify that TensileBackend supports solution pools."""
    backend = TensileBackend()
    assert backend.supports_solution_pool() is True


def test_tensile_backend_missing_required_key_raises():
    backend = TensileBackend()
    cfg = {
        "forkParametersEnabled": True,
        "problemType": object(),
        "assembler": object(),
        "debugConfig": object(),
        "isaInfoMap": {"gfx942": {}},
        "benchmarkStep": types.SimpleNamespace(
            forkParams={},
            paramGroups=[],
            constantParams={},
            customKernels=[],
            internalSupportParams={},
            customKernelWildcard=False,
        ),
    }

    with pytest.raises(ValueError, match="Missing required backend config key: solutionPoolIndex"):
        backend.run({}, cfg, lambda *_args, **_kwargs: ("unused.csv", 0))


def test_tensile_backend_solution_pool_path_uses_pool_entries(monkeypatch):
    backend = TensileBackend()
    calls = {}

    import Tensile.BenchmarkProblems as bp
    monkeypatch.setattr(
        bp,
        "_constructAllPoolSolutions",
        lambda pool_entries, *_a, **_kw: calls.setdefault("pool_entries", pool_entries) or ["pool_sol"],
    )
    monkeypatch.setattr(bp, "_generateForkedSolutions", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("no fork path")))
    monkeypatch.setattr(bp, "_generateCustomKernelSolutions", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("no custom path")))

    def benchmark_runner(solutions, isCached=False, buildOnly=False):
        calls["solutions"] = solutions
        calls["isCached"] = isCached
        calls["buildOnly"] = buildOnly
        return "results.csv", 0

    problem_type = object()
    benchmark_step = types.SimpleNamespace(
        forkParams={"x": [1, 2]},
        paramGroups=[],
        constantParams={},
        customKernels=[],
        internalSupportParams={},
        customKernelWildcard=False,
    )

    benchmark_config = {
        "forkParametersEnabled": True,
        "problemType": problem_type,
        "assembler": object(),
        "debugConfig": object(),
        "isaInfoMap": {"gfx942": {}},
        "benchmarkStep": benchmark_step,
        "solutionPoolIndex": {str(problem_type): [("pool.yaml", {"Solutions": [1]})]},
    }

    backend.run({}, benchmark_config, benchmark_runner, cacheValid=False, buildOnly=False)

    assert calls["pool_entries"] == [("pool.yaml", {"Solutions": [1]})]
    assert calls["solutions"] == [("pool.yaml", {"Solutions": [1]})]
    assert calls["isCached"] is False


def test_ductile_backend_does_not_support_solution_pool():
    """Verify that DuctileBackend does not support solution pools."""
    backend = DuctileBackend()
    assert backend.supports_solution_pool() is False
