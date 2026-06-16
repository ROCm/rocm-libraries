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

################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Extended tests for DuctileBackend — targeting uncovered code paths.

Covers: cacheValid / buildOnly warnings, missing benchmarkStep, missing required
config keys, group parameter expansion (single-element → constantParams,
multi-element → fork_params), checkpoint loading (success + failure), post-
optimization verification stage (all-pass, partial-fail, all-fail), and the
supports_solution_pool API.
"""

import os
import types

import numpy as np
import pandas as pd
import pytest

import Tensile.backends.ductile_backend as ductile_backend_mod
from Tensile.backends.ductile_backend import DuctileBackend

pytestmark = pytest.mark.unit

skipif_no_ductile = pytest.mark.skipif(
    not ductile_backend_mod.DUCTILE_AVAILABLE,
    reason="Ductile modules not available",
)


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_ductile_backend.py helpers)
# ---------------------------------------------------------------------------

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


def _base_merged_config():
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


def _make_benchmark_config(tmp_path, fork_params=None, param_groups=None):
    benchmark_step = types.SimpleNamespace(
        forkParams=fork_params if fork_params is not None else {"DepthU": [32, 64], "SourceSwap": [0, 1]},
        paramGroups=param_groups if param_groups is not None else [],
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


def _patch_primitives(monkeypatch, merged_config):
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


def _make_simple_ga(csv_path, solutions_list):
    """Return a FakeGA class that exercises the optimize + evaluate path."""

    class FakeGA:
        def __init__(self, *args, **kwargs):
            self._evaluate = kwargs["evaluate"]

        def optimize(self):
            self._evaluate([{"a": 0}] * len(solutions_list))
            return [{"a": 0}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _best):
            return np.array([1.0], dtype=np.float32)

    return FakeGA


# ---------------------------------------------------------------------------
# API / construction
# ---------------------------------------------------------------------------

@skipif_no_ductile
def test_supports_solution_pool_returns_false():
    backend = DuctileBackend()
    assert backend.supports_solution_pool() is False


# ---------------------------------------------------------------------------
# Input validation — missing / bad configuration
# ---------------------------------------------------------------------------

@skipif_no_ductile
def test_run_raises_if_benchmark_step_missing(monkeypatch, tmp_path):
    _patch_primitives(monkeypatch, _base_merged_config())
    backend = DuctileBackend()
    cfg = _make_benchmark_config(tmp_path)
    cfg.pop("benchmarkStep")

    with pytest.raises(ValueError, match="Missing required backend config key: benchmarkStep"):
        backend.run({}, cfg, lambda *a, **kw: ("", 0))


@skipif_no_ductile
def test_run_raises_if_required_config_keys_missing(monkeypatch, tmp_path):
    _patch_primitives(monkeypatch, _base_merged_config())
    backend = DuctileBackend()
    cfg = _make_benchmark_config(tmp_path)
    cfg.pop("problemType")

    with pytest.raises(ValueError, match="Missing required config keys"):
        backend.run({}, cfg, lambda *a, **kw: ("", 0))


# ---------------------------------------------------------------------------
# Warnings for unsupported flags
# ---------------------------------------------------------------------------

@skipif_no_ductile
def test_run_warns_on_cache_valid(monkeypatch, tmp_path, capsys):
    csv_path = tmp_path / "results.csv"
    pd.DataFrame({"Cijk_0": [5.0]}).to_csv(csv_path, index=False)

    class FakeGA:
        def __init__(self, *a, **kw):
            self._evaluate = kw["evaluate"]

        def optimize(self):
            self._evaluate([{"a": 0}])
            return [{"a": 0}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _b):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend._generate_ga_solutions",
        lambda *a, **kw: [types.SimpleNamespace()],
    )
    warned = []
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.printWarning",
        lambda msg: warned.append(msg),
    )
    _patch_primitives(monkeypatch, _base_merged_config())

    backend = DuctileBackend()
    backend.run({}, _make_benchmark_config(tmp_path), lambda *a, **kw: (str(csv_path), 0), cacheValid=True)

    assert any("cacheValid" in w for w in warned)


@skipif_no_ductile
def test_run_warns_on_build_only(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    pd.DataFrame({"Cijk_0": [5.0]}).to_csv(csv_path, index=False)

    class FakeGA:
        def __init__(self, *a, **kw):
            self._evaluate = kw["evaluate"]

        def optimize(self):
            self._evaluate([{"a": 0}])
            return [{"a": 0}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _b):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend._generate_ga_solutions",
        lambda *a, **kw: [types.SimpleNamespace()],
    )
    warned = []
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.printWarning",
        lambda msg: warned.append(msg),
    )
    _patch_primitives(monkeypatch, _base_merged_config())

    backend = DuctileBackend()
    backend.run({}, _make_benchmark_config(tmp_path), lambda *a, **kw: (str(csv_path), 0), buildOnly=True)

    assert any("buildOnly" in w for w in warned)


# ---------------------------------------------------------------------------
# Group parameter expansion
# ---------------------------------------------------------------------------

@skipif_no_ductile
def test_single_element_param_group_folded_into_constant_params(monkeypatch, tmp_path):
    """A param_group with one item must be moved to constantParams, not fork_params."""
    csv_path = tmp_path / "results.csv"
    pd.DataFrame({"Cijk_0": [5.0]}).to_csv(csv_path, index=False)

    captured_space_kwargs = {}

    class _CapturingSearchSpace:
        def __init__(self, space, **kwargs):
            captured_space_kwargs["space"] = space

    class FakeGA:
        def __init__(self, *a, **kw):
            self._evaluate = kw["evaluate"]

        def optimize(self):
            self._evaluate([{"a": 0}])
            return [{"a": 0}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _b):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)
    monkeypatch.setattr("Tensile.backends.ductile_backend.SearchSpace", _CapturingSearchSpace)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Selection", _FakeFactory)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Crossover", _FakeFactory)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Survival", _FakeFactory)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Mutation", _FakeMutation)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Mating", _FakeMating)
    monkeypatch.setattr("Tensile.backends.ductile_backend.ductile_config.update", lambda _: _base_merged_config())
    monkeypatch.setattr("Tensile.backends.ductile_backend.ductile_config.populate", lambda c, n: {"name": c[n]["name"]})
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend._generate_ga_solutions",
        lambda *a, **kw: [types.SimpleNamespace()],
    )

    # single-element group: [{"PrefetchGlobalRead": 1}]
    cfg = _make_benchmark_config(tmp_path, param_groups=[[{"PrefetchGlobalRead": 1}]])
    backend = DuctileBackend()
    backend.run({}, cfg, lambda *a, **kw: (str(csv_path), 0))

    # single-element group must NOT appear as group_0 in fork space
    assert "group_0" not in captured_space_kwargs.get("space", {})


@skipif_no_ductile
def test_multi_element_param_group_becomes_fork_param(monkeypatch, tmp_path):
    """A param_group with >1 item must appear as group_N in the fork space."""
    csv_path = tmp_path / "results.csv"
    pd.DataFrame({"Cijk_0": [5.0]}).to_csv(csv_path, index=False)

    captured_space_kwargs = {}

    class _CapturingSearchSpace:
        def __init__(self, space, **kwargs):
            captured_space_kwargs["space"] = space

    class FakeGA:
        def __init__(self, *a, **kw):
            self._evaluate = kw["evaluate"]

        def optimize(self):
            self._evaluate([{"a": 0}])
            return [{"a": 0}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _b):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)
    monkeypatch.setattr("Tensile.backends.ductile_backend.SearchSpace", _CapturingSearchSpace)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Selection", _FakeFactory)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Crossover", _FakeFactory)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Survival", _FakeFactory)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Mutation", _FakeMutation)
    monkeypatch.setattr("Tensile.backends.ductile_backend.Mating", _FakeMating)
    monkeypatch.setattr("Tensile.backends.ductile_backend.ductile_config.update", lambda _: _base_merged_config())
    monkeypatch.setattr("Tensile.backends.ductile_backend.ductile_config.populate", lambda c, n: {"name": c[n]["name"]})
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend._generate_ga_solutions",
        lambda *a, **kw: [types.SimpleNamespace()],
    )

    # multi-element group: two items → goes to fork_params as group_0
    cfg = _make_benchmark_config(
        tmp_path,
        param_groups=[[{"PGR": 1}, {"PGR": 2}]],
    )
    backend = DuctileBackend()
    backend.run({}, cfg, lambda *a, **kw: (str(csv_path), 0))

    assert "group_0" in captured_space_kwargs.get("space", {})


# ---------------------------------------------------------------------------
# Checkpoint loading — success and failure paths
# ---------------------------------------------------------------------------

@skipif_no_ductile
def test_checkpoint_loading_success(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    pd.DataFrame({"Cijk_0": [5.0]}).to_csv(csv_path, index=False)

    checkpoint_file = tmp_path / "step-00__ductile.checkpoint"
    checkpoint_file.write_text("fake-checkpoint")

    load_called = []

    class FakeGA:
        def __init__(self, *a, **kw):
            self._evaluate = kw["evaluate"]

        def load(self, path):
            load_called.append(path)
            return self

        def optimize(self):
            self._evaluate([{"a": 0}])
            return [{"a": 0}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _b):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend._generate_ga_solutions",
        lambda *a, **kw: [types.SimpleNamespace()],
    )
    _patch_primitives(monkeypatch, _base_merged_config())

    backend = DuctileBackend()
    backend.run({}, _make_benchmark_config(tmp_path), lambda *a, **kw: (str(csv_path), 0))

    assert len(load_called) == 1


@skipif_no_ductile
def test_checkpoint_loading_failure_falls_back_to_fresh(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    pd.DataFrame({"Cijk_0": [5.0]}).to_csv(csv_path, index=False)

    checkpoint_file = tmp_path / "step-00__ductile.checkpoint"
    checkpoint_file.write_text("bad-checkpoint")

    warned = []

    class FakeGA:
        def __init__(self, *a, **kw):
            self._evaluate = kw["evaluate"]

        def load(self, path):
            raise ValueError("corrupt checkpoint")

        def optimize(self):
            self._evaluate([{"a": 0}])
            return [{"a": 0}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _b):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend._generate_ga_solutions",
        lambda *a, **kw: [types.SimpleNamespace()],
    )
    monkeypatch.setattr("Tensile.backends.ductile_backend.printWarning", lambda m: warned.append(m))
    _patch_primitives(monkeypatch, _base_merged_config())

    backend = DuctileBackend()
    backend.run({}, _make_benchmark_config(tmp_path), lambda *a, **kw: (str(csv_path), 0))

    assert any("Failed to load checkpoint" in w for w in warned)


# ---------------------------------------------------------------------------
# Post-optimization verification — partial / all-fail paths
# ---------------------------------------------------------------------------

@skipif_no_ductile
def test_verification_all_fail_calls_exit(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    pd.DataFrame({"Cijk_0": [5.0]}).to_csv(csv_path, index=False)

    exited = []

    class FakeGA:
        def __init__(self, *a, **kw):
            self._evaluate = kw["evaluate"]

        def optimize(self):
            self._evaluate([{"a": 0}])
            return [{"a": 0}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _b):
            # Return -1 → validation failed for all
            return np.array([-1.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend._generate_ga_solutions",
        lambda *a, **kw: [types.SimpleNamespace()],
    )
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.printExit",
        lambda msg: exited.append(msg),
    )
    _patch_primitives(monkeypatch, _base_merged_config())

    backend = DuctileBackend()
    backend.run({}, _make_benchmark_config(tmp_path), lambda *a, **kw: (str(csv_path), 0))

    assert any("No solutions passed" in e for e in exited)


@skipif_no_ductile
def test_verification_partial_fail_warns_and_reevaluates(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    pd.DataFrame({"Cijk_0": [5.0], "Cijk_1": [3.0]}).to_csv(csv_path, index=False)

    warned = []
    eval_calls = []

    class FakeGA:
        def __init__(self, *a, **kw):
            self._evaluate = kw["evaluate"]

        def optimize(self):
            self._evaluate([{"a": 0}, {"a": 1}])
            return [{"a": 0}, {"a": 1}], np.array([1.0, 0.8], dtype=np.float32)

        def evaluate(self, _b):
            eval_calls.append(True)
            # First call: second solution fails; second call: pass
            if len(eval_calls) == 1:
                return np.array([5.0, -1.0], dtype=np.float32)
            return np.array([5.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend._generate_ga_solutions",
        lambda *a, **kw: [types.SimpleNamespace(), types.SimpleNamespace()],
    )
    monkeypatch.setattr("Tensile.backends.ductile_backend.printWarning", lambda m: warned.append(m))
    _patch_primitives(monkeypatch, _base_merged_config())

    backend = DuctileBackend()
    backend.run({}, _make_benchmark_config(tmp_path), lambda *a, **kw: (str(csv_path), 0))

    assert any("passed verification" in w for w in warned)


# ---------------------------------------------------------------------------
# Multi-step log filename path
# ---------------------------------------------------------------------------

@skipif_no_ductile
def test_multistep_uses_step_indexed_log_filename(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    pd.DataFrame({"Cijk_0": [5.0]}).to_csv(csv_path, index=False)

    log_paths = []

    class FakeGA:
        def __init__(self, *a, **kw):
            log_paths.append(str(kw.get("log_file", "")))
            self._evaluate = kw["evaluate"]

        def optimize(self):
            self._evaluate([{"a": 0}])
            return [{"a": 0}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _b):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend._generate_ga_solutions",
        lambda *a, **kw: [types.SimpleNamespace()],
    )
    _patch_primitives(monkeypatch, _base_merged_config())

    cfg = _make_benchmark_config(tmp_path)
    cfg["totalBenchmarkSteps"] = 3
    cfg["benchmarkStepIdx"] = 1
    backend = DuctileBackend()
    backend.run({}, cfg, lambda *a, **kw: (str(csv_path), 0))

    assert any("step-01" in p for p in log_paths)
