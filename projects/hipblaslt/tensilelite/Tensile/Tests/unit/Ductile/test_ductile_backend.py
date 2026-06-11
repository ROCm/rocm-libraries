################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import types

import numpy as np
import pandas as pd
import pytest

import Tensile.backends.ductile_backend as ductile_backend_mod
from Tensile.backends.ductile_backend import DuctileBackend

pytestmark = pytest.mark.unit


class _FakeFactory:
    @staticmethod
    def get(*args, **kwargs):
        return object()


class _FakeMutation:
    def __init__(self, *args, **kwargs):
        pass


class _FakeMating:
    def __init__(self, *args, **kwargs):
        pass


class _FakeSearchSpace:
    def __init__(self, *args, **kwargs):
        pass


def _base_ductile_merged_config():
    return {
        "max_iters": 4,
        "selection": {"name": "tournament", "tournament": {"k": 2}, "common": {}},
        "crossover": {"name": "ux", "common": {}},
        "mutation": {"prob": 0.2},
        "survival": {"name": "fitness"},
        "pop_size": 4,
        "n_gen": 1,
        "soo": False,
        "period": 0,
        "tol": 0.0,
        "div_thr": 0.5,
        "seed": 1,
        "verbose": 0,
        "weights": None,
        "weight_beta": 0.25,
        "n_elements_to_validate": 0,
    }


def _make_benchmark_config(tmp_path):
    benchmark_step = types.SimpleNamespace(
        forkParams={"DepthU": [32, 64], "SourceSwap": [0, 1]},
        paramGroups=[],
        constantParams={},
    )

    return {
        "forkParametersEnabled": True,
        "problemType": types.SimpleNamespace(state={}),
        "assembler": object(),
        "debugConfig": types.SimpleNamespace(splitGSU=False),
        "isaInfoMap": {"gfx942": {}},
        "benchmarkStep": benchmark_step,
        "sourcePath": str(tmp_path / "source"),
        "rootPath": str(tmp_path),
        "configName": "ductile-eval",
        "benchmarkStepIdx": 0,
        "totalBenchmarkSteps": 1,
    }


def _patch_ductile_backend_primitives(monkeypatch, merged_config):
    monkeypatch.setattr("Tensile.backends.ductile_backend.SearchSpace", _FakeSearchSpace)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Selection", _FakeFactory)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Crossover", _FakeFactory)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Survival", _FakeFactory)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Mutation", _FakeMutation)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Mating", _FakeMating)
    monkeypatch.setattr("Tensile.backends.ductile_backend.ductile_config.update", lambda _cfg: merged_config)
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.ductile_config.populate",
        lambda _cfg, name: {"name": _cfg[name]["name"]},
    )


@pytest.mark.skipif(not ductile_backend_mod.DUCTILE_AVAILABLE, reason="Ductile modules are not available")
def test_ductile_backend_evaluate_missing_results_file_exits(monkeypatch, tmp_path):
    class FakeGA:
        def __init__(self, *args, **kwargs):
            self._evaluate = kwargs["evaluate"]

        def optimize(self):
            self._evaluate([{"a": 0}, {"a": 1}])
            return [{"a": 0}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _best):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend._generate_ga_solutions",
        lambda *_args, **_kwargs: [types.SimpleNamespace(), types.SimpleNamespace()],
    )
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.printExit",
        lambda msg: (_ for _ in ()).throw(RuntimeError(msg)),
    )
    _patch_ductile_backend_primitives(monkeypatch, _base_ductile_merged_config())

    backend = DuctileBackend()
    with pytest.raises(RuntimeError, match="Expected results file does not exist"):
        backend.run(
            {},
            _make_benchmark_config(tmp_path),
            lambda *_args, **_kwargs: (str(tmp_path / "missing.csv"), 0),
        )


@pytest.mark.skipif(not ductile_backend_mod.DUCTILE_AVAILABLE, reason="Ductile modules are not available")
def test_ductile_backend_evaluate_column_mismatch_exits(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    pd.DataFrame({"Cijk_0": [10.0, 11.0]}).to_csv(csv_path, index=False)

    class FakeGA:
        def __init__(self, *args, **kwargs):
            self._evaluate = kwargs["evaluate"]

        def optimize(self):
            self._evaluate([{"a": 0}, {"a": 1}, {"a": 2}])
            return [{"a": 0}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _best):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend._generate_ga_solutions",
        lambda *_args, **_kwargs: [types.SimpleNamespace(), types.SimpleNamespace(), None],
    )
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.printExit",
        lambda msg: (_ for _ in ()).throw(RuntimeError(msg)),
    )
    _patch_ductile_backend_primitives(monkeypatch, _base_ductile_merged_config())

    backend = DuctileBackend()
    with pytest.raises(RuntimeError, match="Mismatch between result columns and valid solutions"):
        backend.run({}, _make_benchmark_config(tmp_path), lambda *_args, **_kwargs: (str(csv_path), 0))


@pytest.mark.skipif(not ductile_backend_mod.DUCTILE_AVAILABLE, reason="Ductile modules are not available")
def test_ductile_backend_evaluate_preserves_solution_index_alignment(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    pd.DataFrame({"Cijk_0": [10.0, 11.0], "Cijk_1": [20.0, 21.0]}).to_csv(csv_path, index=False)

    captured = {}

    class FakeGA:
        def __init__(self, *args, **kwargs):
            self._evaluate = kwargs["evaluate"]

        def optimize(self):
            captured["nGFlops"] = self._evaluate([{"a": 0}, {"a": 1}, {"a": 2}])
            return [{"a": 0}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _best):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend._generate_ga_solutions",
        lambda *_args, **_kwargs: [types.SimpleNamespace(), None, types.SimpleNamespace()],
    )
    _patch_ductile_backend_primitives(monkeypatch, _base_ductile_merged_config())

    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    marker = source_dir / "dummy.txt"
    marker.write_text("x", encoding="utf-8")

    backend = DuctileBackend()
    backend.run({}, _make_benchmark_config(tmp_path), lambda *_args, **_kwargs: (str(csv_path), 0))

    assert not os.path.isdir(str(source_dir))
    fitness = captured["nGFlops"]
    assert fitness.shape == (2, 3)
    assert np.allclose(fitness[:, 0], [10.0, 11.0])
    assert np.allclose(fitness[:, 1], [0.0, 0.0])
    assert np.allclose(fitness[:, 2], [20.0, 21.0])