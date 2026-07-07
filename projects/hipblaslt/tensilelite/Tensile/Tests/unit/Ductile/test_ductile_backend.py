# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import csv
import os
import types

import numpy as np
import pytest

import Tensile.backends.ductile_backend as ductile_backend_mod
from Tensile.backends.ductile_backend import DuctileBackend

pytestmark = pytest.mark.unit


def _write_csv(path, data: dict):
    """Write a dict of {col_name: [values]} as a CSV file."""
    cols = list(data.keys())
    rows = zip(*data.values())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)


def _fake_solution(name):
    return types.SimpleNamespace(name=name)


def _raise_runtime(msg):
    raise RuntimeError(msg)


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


def test_read_and_validate_scores_csv_success(monkeypatch, tmp_path):
    csv_path = tmp_path / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sol0", "sol1", "other"])
        writer.writerow([10.5, 20.25, 1])
        writer.writerow([11.5, 21.25, 2])

    monkeypatch.setattr(ductile_backend_mod, "printExit", _raise_runtime)

    scores = ductile_backend_mod._read_and_validate_scores_csv(str(csv_path), ["sol0", "sol1"])

    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.float32
    assert scores.shape == (2, 2)
    assert np.allclose(scores, np.array([[10.5, 20.25], [11.5, 21.25]], dtype=np.float32))


def test_read_and_validate_scores_csv_empty_file(monkeypatch, tmp_path):
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(ductile_backend_mod, "printExit", _raise_runtime)

    with pytest.raises(RuntimeError, match="Empty CSV results file"):
        ductile_backend_mod._read_and_validate_scores_csv(str(csv_path), ["sol0"])


def test_read_and_validate_scores_csv_missing_header_row(monkeypatch, tmp_path):
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text("\n", encoding="utf-8")

    monkeypatch.setattr(ductile_backend_mod, "printExit", _raise_runtime)

    with pytest.raises(RuntimeError, match="Missing CSV header row"):
        ductile_backend_mod._read_and_validate_scores_csv(str(csv_path), ["sol0"])


def test_read_and_validate_scores_csv_missing_expected_column(monkeypatch, tmp_path):
    csv_path = tmp_path / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sol0"])
        writer.writerow([10.0])

    monkeypatch.setattr(ductile_backend_mod, "printExit", _raise_runtime)

    with pytest.raises(RuntimeError, match="Missing expected result column"):
        ductile_backend_mod._read_and_validate_scores_csv(str(csv_path), ["sol0", "sol1"])


def test_read_and_validate_scores_csv_short_row(monkeypatch, tmp_path):
    csv_path = tmp_path / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sol0", "sol1"])
        writer.writerow([10.0])

    monkeypatch.setattr(ductile_backend_mod, "printExit", _raise_runtime)

    with pytest.raises(RuntimeError, match="Malformed CSV row"):
        ductile_backend_mod._read_and_validate_scores_csv(str(csv_path), ["sol0", "sol1"])


def test_read_and_validate_scores_csv_non_float(monkeypatch, tmp_path):
    csv_path = tmp_path / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sol0"])
        writer.writerow(["not-a-float"])

    monkeypatch.setattr(ductile_backend_mod, "printExit", _raise_runtime)

    with pytest.raises(RuntimeError, match="Non-float result value"):
        ductile_backend_mod._read_and_validate_scores_csv(str(csv_path), ["sol0"])


def test_read_and_validate_scores_csv_no_data_rows(monkeypatch, tmp_path):
    csv_path = tmp_path / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sol0"])

    monkeypatch.setattr(ductile_backend_mod, "printExit", _raise_runtime)

    with pytest.raises(RuntimeError, match="No data rows found"):
        ductile_backend_mod._read_and_validate_scores_csv(str(csv_path), ["sol0"])


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
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.getSolutionNameMin",
        lambda solution, _splitgsu: getattr(solution, "name", f"Cijk_{solution.solIdx}"),
    )


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


def test_ductile_backend_evaluate_column_mismatch_exits(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, {"sol0": [10.0, 11.0]})

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
        lambda *_args, **_kwargs: [_fake_solution("sol0"), _fake_solution("sol1"), None],
    )
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.getSolutionNameMin",
        lambda solution, _splitgsu: solution.name,
    )
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.printExit",
        lambda msg: (_ for _ in ()).throw(RuntimeError(msg)),
    )
    _patch_ductile_backend_primitives(monkeypatch, _base_ductile_merged_config())

    backend = DuctileBackend()
    with pytest.raises(RuntimeError, match="Missing expected result column"):
        backend.run({}, _make_benchmark_config(tmp_path), lambda *_args, **_kwargs: (str(csv_path), 0))


def test_ductile_backend_evaluate_empty_csv_exits(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("", encoding="utf-8")

    class FakeGA:
        def __init__(self, *args, **kwargs):
            self._evaluate = kwargs["evaluate"]

        def optimize(self):
            self._evaluate([{"a": 0}])
            return [{"a": 0}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _best):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend._generate_ga_solutions",
        lambda *_args, **_kwargs: [_fake_solution("sol0")],
    )
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.getSolutionNameMin",
        lambda solution, _splitgsu: solution.name,
    )
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.printExit",
        lambda msg: (_ for _ in ()).throw(RuntimeError(msg)),
    )
    _patch_ductile_backend_primitives(monkeypatch, _base_ductile_merged_config())

    backend = DuctileBackend()
    with pytest.raises(RuntimeError, match="Empty CSV results file"):
        backend.run({}, _make_benchmark_config(tmp_path), lambda *_args, **_kwargs: (str(csv_path), 0))


def test_ductile_backend_evaluate_short_row_exits(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sol0", "sol1"])
        writer.writerow([10.0])

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
        lambda *_args, **_kwargs: [_fake_solution("sol0"), _fake_solution("sol1")],
    )
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.getSolutionNameMin",
        lambda solution, _splitgsu: solution.name,
    )
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.printExit",
        lambda msg: (_ for _ in ()).throw(RuntimeError(msg)),
    )
    _patch_ductile_backend_primitives(monkeypatch, _base_ductile_merged_config())

    backend = DuctileBackend()
    with pytest.raises(RuntimeError, match="Malformed CSV row"):
        backend.run({}, _make_benchmark_config(tmp_path), lambda *_args, **_kwargs: (str(csv_path), 0))


def test_ductile_backend_evaluate_non_float_value_exits(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sol0"])
        writer.writerow(["not-a-float"])

    class FakeGA:
        def __init__(self, *args, **kwargs):
            self._evaluate = kwargs["evaluate"]

        def optimize(self):
            self._evaluate([{"a": 0}])
            return [{"a": 0}], np.array([1.0], dtype=np.float32)

        def evaluate(self, _best):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr("Tensile.backends.ductile_backend.GeneticAlgorithm", FakeGA)
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend._generate_ga_solutions",
        lambda *_args, **_kwargs: [_fake_solution("sol0")],
    )
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.getSolutionNameMin",
        lambda solution, _splitgsu: solution.name,
    )
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.printExit",
        lambda msg: (_ for _ in ()).throw(RuntimeError(msg)),
    )
    _patch_ductile_backend_primitives(monkeypatch, _base_ductile_merged_config())

    backend = DuctileBackend()
    with pytest.raises(RuntimeError, match="Non-float result value"):
        backend.run({}, _make_benchmark_config(tmp_path), lambda *_args, **_kwargs: (str(csv_path), 0))


def test_ductile_backend_evaluate_preserves_solution_index_alignment(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, {"sol0": [10.0, 11.0], "sol2": [20.0, 21.0]})

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
        lambda *_args, **_kwargs: [_fake_solution("sol0"), None, _fake_solution("sol2")],
    )
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.getSolutionNameMin",
        lambda solution, _splitgsu: solution.name,
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


def test_generate_single_solution_with_groups_expands_group_keys(monkeypatch):
    captured = {}

    def _fake_build(solution, *_a, **_kw):
        captured["solution"] = solution
        return types.SimpleNamespace()

    monkeypatch.setattr("Tensile.BenchmarkProblems._build_and_validate_solution", _fake_build)

    perm = {"DepthU": 64, "group_0": {"SourceSwap": 1, "PrefetchGlobalRead": 2}}
    result = ductile_backend_mod._generate_single_solution_with_groups(
        perm=perm,
        problemType=types.SimpleNamespace(state={"DataType": "s"}),
        constantParams={"WorkGroup": [16, 16, 1]},
        assembler=object(),
        debugConfig=types.SimpleNamespace(),
        isaInfoMap={"gfx942": {}},
        silent=True,
    )

    assert result is not None
    assert "group_0" not in captured["solution"]
    assert captured["solution"]["SourceSwap"] == 1
    assert captured["solution"]["PrefetchGlobalRead"] == 2
    assert captured["solution"]["DepthU"] == 64


def test_validate_solution_returns_false_when_kernel_init_fails(monkeypatch):
    monkeypatch.setattr(
        ductile_backend_mod,
        "_generate_single_solution_with_groups",
        lambda *_a, **_kw: types.SimpleNamespace(),
    )

    class _FailKW:
        def __init__(self, *_a, **_kw):
            pass

        def _initKernel(self, *_a, **_kw):
            raise RuntimeError("bad")

    monkeypatch.setattr(ductile_backend_mod, "KernelWriterAssembly", _FailKW)

    ok = ductile_backend_mod._validate_solution(
        problemType=types.SimpleNamespace(state={}),
        constantParams={},
        assembler=object(),
        debugConfig=types.SimpleNamespace(),
        isaInfoMap={"gfx942": {}},
        perm={"DepthU": 64},
        get_kernel_src=False,
    )

    assert ok is False


def test_validate_solution_calls_get_kernel_source_when_requested(monkeypatch):
    monkeypatch.setattr(
        ductile_backend_mod,
        "_generate_single_solution_with_groups",
        lambda *_a, **_kw: types.SimpleNamespace(),
    )
    seen = {"src": False}

    class _OkKW:
        def __init__(self, *_a, **_kw):
            pass

        def _initKernel(self, *_a, **_kw):
            return None

        def _getKernelSource(self, *_a, **_kw):
            seen["src"] = True

    monkeypatch.setattr(ductile_backend_mod, "KernelWriterAssembly", _OkKW)

    ok = ductile_backend_mod._validate_solution(
        problemType=types.SimpleNamespace(state={}),
        constantParams={},
        assembler=object(),
        debugConfig=types.SimpleNamespace(),
        isaInfoMap={"gfx942": {}},
        perm={"DepthU": 64},
        get_kernel_src=True,
    )

    assert ok is True
    assert seen["src"] is True


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
    monkeypatch.setattr(
        "Tensile.backends.ductile_backend.getSolutionNameMin",
        lambda solution, _splitgsu: getattr(solution, "name", f"Cijk_{solution.solIdx}"),
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

def test_supports_solution_pool_returns_false():
    backend = DuctileBackend()
    assert backend.supports_solution_pool() is False


# ---------------------------------------------------------------------------
# Input validation — missing / bad configuration
# ---------------------------------------------------------------------------

def test_run_raises_if_benchmark_step_missing(monkeypatch, tmp_path):
    _patch_primitives(monkeypatch, _base_merged_config())
    backend = DuctileBackend()
    cfg = _make_benchmark_config(tmp_path)
    cfg.pop("benchmarkStep")

    with pytest.raises(ValueError, match="Missing required backend config key: benchmarkStep"):
        backend.run({}, cfg, lambda *a, **kw: ("", 0))


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

def test_run_warns_on_cache_valid(monkeypatch, tmp_path, capsys):
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, {"Cijk_0": [5.0]})

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


def test_run_warns_on_build_only(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, {"Cijk_0": [5.0]})

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

def test_single_element_param_group_folded_into_constant_params(monkeypatch, tmp_path):
    """A param_group with one item must be moved to constantParams, not fork_params."""
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, {"Cijk_0": [5.0]})

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
        "Tensile.backends.ductile_backend.getSolutionNameMin",
        lambda solution, _splitgsu: getattr(solution, "name", f"Cijk_{solution.solIdx}"),
    )
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


def test_multi_element_param_group_becomes_fork_param(monkeypatch, tmp_path):
    """A param_group with >1 item must appear as group_N in the fork space."""
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, {"Cijk_0": [5.0]})

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
        "Tensile.backends.ductile_backend.getSolutionNameMin",
        lambda solution, _splitgsu: getattr(solution, "name", f"Cijk_{solution.solIdx}"),
    )
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

def test_checkpoint_loading_success(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, {"Cijk_0": [5.0]})

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


def test_checkpoint_loading_failure_falls_back_to_fresh(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, {"Cijk_0": [5.0]})

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

def test_verification_all_fail_calls_exit(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, {"Cijk_0": [5.0]})

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


def test_verification_partial_fail_warns_and_reevaluates(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, {"Cijk_0": [5.0], "Cijk_1": [3.0]})

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

def test_multistep_uses_step_indexed_log_filename(monkeypatch, tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_csv(csv_path, {"Cijk_0": [5.0]})

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
