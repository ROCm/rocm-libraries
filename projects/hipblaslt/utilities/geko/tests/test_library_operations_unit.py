# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import yaml

from geko.library import Library
from geko.library import operations


def _make_min_library() -> Library:
    data = [
        None,
        None,
        "gfx950",
        None,
        {"DataType": 0},
        [{"SolutionIndex": 0, "StaggerU": 0}],
        [2, 3, 0, 1],
        [[[16, 16, 1, 16], [0, 0.0]]],
        None,
        None,
        "DeviceEfficiency",
        "Equality",
    ]
    return Library(data, "test_lib.yaml")


def _make_min_dict_library() -> Library:
    data = {
        "ArchitectureName": "gfx950",
        "ProblemType": {"TransposeA": 0, "TransposeB": 0, "DataType": 0, "DestDataType": 0},
        "DefaultSolution": {"StaggerU": 0},
        "Solutions": [{"SolutionIndex": 0, "StaggerU": 0}],
        "IndexOrder": [2, 3, 0, 1],
        "ExactLogic": [[[16, 16, 1, 16], [0, 0.0]]],
        "PerfMetric": "DeviceEfficiency",
        "LibraryType": "Equality",
    }
    return Library(data, "dict_test_lib.yaml")


def _write_library_yaml(path: Path, lib_name: str = "lib.yaml") -> Path:
    data = [
        None,
        None,
        "gfx950",
        None,
        {
            "TransposeA": 0,
            "TransposeB": 0,
            "DataType": 0,
            "DestDataType": 0,
            "ComputeDataType": 0,
        },
        [{"SolutionIndex": 0, "StaggerU": 0}],
        [2, 3, 0, 1],
        [[[16, 16, 1, 16], [0, 100.0]]],
        None,
        None,
        "DeviceEfficiency",
        "Equality",
    ]
    p = path / lib_name
    yaml.safe_dump(data, p.open("w"), sort_keys=False)
    return p


def _write_match_table(path: Path, entries) -> Path:
    p = path / "MatchTable.yaml"
    yaml.safe_dump(entries, p.open("w"), sort_keys=False)
    return p


def test_load_collection_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        operations.load_collection(tmp_path / "missing")


def test_merge_solutions_no_files_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No valid YAML libraries"):
        operations.merge_solutions(tmp_path)


def test_merge_solutions_happy_path(tmp_path: Path) -> None:
    ldir = tmp_path / "b1" / "3_LibraryLogic"
    ldir.mkdir(parents=True)
    _write_library_yaml(ldir, "x.yaml")

    merged = operations.merge_solutions(tmp_path, epilogues=False)
    assert len(merged) == 1
    assert merged[0].name == "x.yaml"


def test_merge_rejects_missing_hipblaslt_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        operations.merge(tmp_path / "missing", "orig", "inc", "out")


def test_merge_invokes_tensile_merge_library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    (hip / "tensilelite/Tensile/bin").mkdir(parents=True)
    called = {}

    def _fake_run(cmd):
        called["cmd"] = cmd

    monkeypatch.setattr(operations, "run_silent_command", _fake_run)
    operations.merge(hip, "orig", "inc", "out", eff=False, force=True)
    assert "TensileMergeLibrary" in called["cmd"][0]
    assert "--no_eff" in called["cmd"]


def test_create_rejects_missing_hip_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        operations.create(tmp_path / "missing", tmp_path / "libs", tmp_path / "out")


def test_create_rejects_empty_library_dir(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    libs = tmp_path / "libs"
    libs.mkdir()
    with pytest.raises(ValueError, match="No valid libraries"):
        operations.create(hip, libs, tmp_path / "out")


def test_create_invokes_tensile_create_library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    (hip / "tensilelite/Tensile/bin").mkdir(parents=True)
    libs = tmp_path / "libs"
    libs.mkdir()
    _write_library_yaml(libs, "x.yaml")

    called = {}

    def _fake_run(cmd):
        called["cmd"] = cmd

    monkeypatch.setattr(operations, "run_silent_command", _fake_run)
    operations.create(hip, libs, tmp_path / "out")
    assert "TensileCreateLibrary" in called["cmd"][0]


def test_from_dataframe_requires_lib_column(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "M": 16,
                "N": 16,
                "K": 16,
                "batch_count": 1,
                "transA": "N",
                "transB": "N",
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "f32_r",
            }
        ]
    )
    with pytest.raises(ValueError, match="must contain the lib"):
        operations.from_dataframe(df, tmp_path)


def test_from_dataframe_requires_gemm_fields(tmp_path: Path) -> None:
    df = pd.DataFrame([{"lib": "a.yaml", "M": 16}])
    with pytest.raises(ValueError, match="missing fields"):
        operations.from_dataframe(df, tmp_path)


def test_from_dataframe_happy_path(tmp_path: Path) -> None:
    lib_dir = tmp_path / "libs"
    lib_dir.mkdir()
    _write_library_yaml(lib_dir, "a.yaml")

    df = pd.DataFrame(
        [
            {
                "lib": "a.yaml",
                "M": 16,
                "N": 16,
                "K": 16,
                "batch_count": 1,
                "transA": "N",
                "transB": "N",
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "f32_r",
            }
        ]
    )

    libs = operations.from_dataframe(df, lib_dir)
    assert len(libs) == 1
    assert libs[0].name == "a.yaml"
    assert len(libs[0].solutions) == 1
    assert len(libs[0].sizes) == 1


def test_from_dataframe_reuses_solution_for_multiple_sizes(tmp_path: Path) -> None:
    """Regression test: multiple sizes may map to the same source solution.

    run_search-style selections can include different GEMM sizes that resolve to the
    same source solution index. The generated library should keep one solution and
    reference it from each size mapping.
    """
    lib_dir = tmp_path / "libs"
    lib_dir.mkdir()

    data = [
        None,
        None,
        "gfx950",
        None,
        {
            "TransposeA": 0,
            "TransposeB": 0,
            "DataType": 0,
            "DestDataType": 0,
            "ComputeDataType": 0,
        },
        [{"SolutionIndex": 0, "StaggerU": 0}],
        [2, 3, 0, 1],
        [
            [[16, 16, 1, 16], [0, 100.0]],
            [[32, 16, 1, 16], [0, 95.0]],
        ],
        None,
        None,
        "DeviceEfficiency",
        "Equality",
    ]
    yaml.safe_dump(data, (lib_dir / "a.yaml").open("w"), sort_keys=False)

    df = pd.DataFrame(
        [
            {
                "lib": "a.yaml",
                "M": 16,
                "N": 16,
                "K": 16,
                "batch_count": 1,
                "transA": "N",
                "transB": "N",
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "f32_r",
            },
            {
                "lib": "a.yaml",
                "M": 32,
                "N": 16,
                "K": 16,
                "batch_count": 1,
                "transA": "N",
                "transB": "N",
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "f32_r",
            },
        ]
    )

    libs = operations.from_dataframe(df, lib_dir)
    assert len(libs) == 1
    assert libs[0].name == "a.yaml"
    assert len(libs[0].sizes) == 2

    # Expected behavior: both sizes should point to the same deduplicated solution.
    assert len(libs[0].solutions) == 1
    assert libs[0].sizes[0][1][0] == 0
    assert libs[0].sizes[1][1][0] == 0


def test_from_full_dataframe_reuses_tuned_solution_for_multiple_sizes(tmp_path: Path) -> None:
    lib_dir = tmp_path / "libs"
    lib_dir.mkdir()

    data = [
        None,
        None,
        "gfx950",
        None,
        {
            "TransposeA": 0,
            "TransposeB": 0,
            "DataType": 0,
            "DestDataType": 0,
            "ComputeDataType": 0,
        },
        [{"SolutionIndex": 0, "StaggerU": 0}],
        [2, 3, 0, 1],
        [
            [[16, 16, 1, 16], [0, 100.0]],
            [[32, 16, 1, 16], [0, 95.0]],
        ],
        None,
        None,
        "DeviceEfficiency",
        "Equality",
    ]
    yaml.safe_dump(data, (lib_dir / "a.yaml").open("w"), sort_keys=False)

    match_table = tmp_path / "MatchTable.yaml"
    yaml.safe_dump({}, match_table.open("w"), sort_keys=False)

    df = pd.DataFrame(
        [
            {
                "lib": "a.yaml",
                "M": 16,
                "N": 16,
                "K": 16,
                "batch_count": 1,
                "transA": "N",
                "transB": "N",
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "f32_r",
                "winner": "tuned",
                "solutionIdx_reference": 0,
            },
            {
                "lib": "a.yaml",
                "M": 32,
                "N": 16,
                "K": 16,
                "batch_count": 1,
                "transA": "N",
                "transB": "N",
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "f32_r",
                "winner": "tuned",
                "solutionIdx_reference": 0,
            },
        ]
    )

    libs = operations.from_full_dataframe(df, lib_dir, match_table)

    assert len(libs) == 1
    assert libs[0].name == "a.yaml"
    assert len(libs[0].sizes) == 2
    assert len(libs[0].solutions) == 1
    assert libs[0].sizes[0][1][0] == 0
    assert libs[0].sizes[1][1][0] == 0


def test_extract_solutions_requires_solution_idx(tmp_path: Path) -> None:
    df = pd.DataFrame([{"m": 16}])
    with pytest.raises(ValueError, match="solutionIdx"):
        operations.extract_solutions(df, tmp_path / "MatchTable.yaml")


def test_extract_solutions_happy_path(tmp_path: Path) -> None:
    lib_dir = tmp_path / "libs"
    lib_dir.mkdir()
    p = _write_library_yaml(lib_dir, "a.yaml")

    mt = _write_match_table(tmp_path, [[str(p), 0]])
    df = pd.DataFrame(
        [
            {
                "m": 16,
                "n": 16,
                "batch_count": 1,
                "k": 16,
                "solutionIdx": 0,
            }
        ]
    )

    libs = operations.extract_solutions(df, mt)
    assert len(libs) == 1
    assert len(libs[0].sizes) == 1


def test_prune_library_missing_hipblaslt_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        operations.prune_library(tmp_path / "missing", _make_min_library())


def test_prune_library_requires_library_type(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    with pytest.raises(TypeError, match="Must be of type 'Library'"):
        operations.prune_library(hip, base_lib={})


def test_prune_library_requires_sizes(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    lib = _make_min_library()
    lib.data[7] = None
    with pytest.raises(ValueError, match="size-solution mappings"):
        operations.prune_library(hip, base_lib=lib)


def test_prune_library_raises_on_cluster_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    lib = _make_min_library()

    monkeypatch.setattr(operations._bank, "cluster_solutions", lambda *_args, **_kwargs: {"k": {"solutions": lib.solutions, "sizes": lib.sizes}})

    class _FailRunner:
        def __init__(self, **_kwargs):
            return None

        def __call__(self, _workdir):
            return []

    monkeypatch.setattr(operations, "Runner", _FailRunner)

    with pytest.raises(RuntimeError, match="Cluster pruning failed"):
        operations.prune_library(hip, lib, workdir=tmp_path / "w")


def test_prune_library_uses_non_cluster_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    lib = _make_min_library()

    monkeypatch.setattr(operations, "create", lambda *_a, **_k: None)
    monkeypatch.setattr(
        operations._bank,
        "min_assigment",
        lambda *_a, **_k: (lib.solutions, lib.sizes),
    )

    class _PassRunner:
        def __init__(self, items, worker_impl, devices, n_slots=1):
            self.items = items
            self.worker_impl = worker_impl
            self.device = devices[0]

        def __call__(self, workdir):
            for item in self.items:
                w = self.worker_impl(item, self.device, 0, None, None)
                w.setup()
                w.run()
                w.teardown()
            return self.items

    monkeypatch.setattr(operations, "Runner", _PassRunner)
    monkeypatch.setattr(operations, "merge_solutions", lambda *_a, **_k: [lib])

    out = operations.prune_library(hip, lib, workdir=tmp_path / "w", cluster=False, devices=[0])
    assert out.name == lib.name


def test_prune_library_uses_non_cluster_path_with_dict_library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    lib = _make_min_dict_library()

    monkeypatch.setattr(operations, "create", lambda *_a, **_k: None)
    monkeypatch.setattr(
        operations._bank,
        "min_assigment",
        lambda *_a, **_k: (lib.solutions, lib.sizes),
    )

    class _PassRunner:
        def __init__(self, items, worker_impl, devices, n_slots=1):
            self.items = items
            self.worker_impl = worker_impl
            self.device = devices[0]

        def __call__(self, workdir):
            for item in self.items:
                w = self.worker_impl(item, self.device, 0, None, None)
                w.setup()
                w.run()
                w.teardown()
            return self.items

    monkeypatch.setattr(operations, "Runner", _PassRunner)
    monkeypatch.setattr(operations, "merge_solutions", lambda *_a, **_k: [lib])

    out = operations.prune_library(hip, lib, workdir=tmp_path / "w", cluster=False, devices=[0])
    assert out.name == lib.name


def test_prune_library_raises_with_error_details(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    lib = _make_min_library()
    sols = [lib.solutions[0], dict(lib.solutions[0], SolutionIndex=1)]
    sizes = [
        [[16, 16, 1, 16], [0, 0.0]],
        [[32, 32, 1, 32], [1, 0.0]],
    ]

    monkeypatch.setattr(
        operations._bank,
        "cluster_solutions",
        lambda *_args, **_kwargs: {
            (1, 2): {"solutions": sols, "sizes": sizes},
            (3, 4): {"solutions": sols, "sizes": sizes},
        },
    )

    monkeypatch.setattr(operations, "create", lambda *_a, **_k: None)
    monkeypatch.setattr(
        operations._bank,
        "min_assigment",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    class _PartialRunner:
        def __init__(self, items, worker_impl, devices, n_slots=1):
            self.items = items
            self.worker_impl = worker_impl
            self.device = devices[0]

        def __call__(self, workdir):
            for item in self.items:
                w = self.worker_impl(item, self.device, 0, None, None)
                w.setup()
                w.run()
                w.teardown()
            return []

    monkeypatch.setattr(operations, "Runner", _PartialRunner)

    with pytest.raises(RuntimeError, match="boom"):
        operations.prune_library(hip, lib, workdir=tmp_path / "w")


def test_prune_library_raises_when_merge_returns_multiple(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    lib = _make_min_library()

    monkeypatch.setattr(operations, "create", lambda *_a, **_k: None)
    monkeypatch.setattr(operations._bank, "min_assigment", lambda *_a, **_k: (lib.solutions, lib.sizes))

    class _PassRunner:
        def __init__(self, items, worker_impl, devices, n_slots=1):
            self.items = items
            self.worker_impl = worker_impl
            self.device = devices[0]

        def __call__(self, workdir):
            for item in self.items:
                w = self.worker_impl(item, self.device, 0, None, None)
                w.setup()
                w.run()
                w.teardown()
            return self.items

    monkeypatch.setattr(operations, "Runner", _PassRunner)
    monkeypatch.setattr(operations, "merge_solutions", lambda *_a, **_k: [lib, lib])

    with pytest.raises(RuntimeError, match="Found 2 libraries"):
        operations.prune_library(hip, lib, workdir=tmp_path / "w", cluster=False, devices=[0])


# ---------------------------------------------------------------------------
# normalize tests
# ---------------------------------------------------------------------------

def _make_tensile_mocks(calls: dict):
    """Return (sys.modules patch dict, mock modules) for Tensile imports."""
    tensile_mod = types.ModuleType("Tensile")
    library_io_mod = types.ModuleType("Tensile.LibraryIO")
    custom_yaml_mod = types.ModuleType("Tensile.CustomYamlLoader")
    merge_lib_mod = types.ModuleType("Tensile.TensileMergeLibrary")

    def _load_yaml_stream(path, loader):
        calls["load"] = path
        return [None, None, "gfx950"]  # must be a list – function validates this

    def _convert_to_dict(data, path):
        calls["convert"] = path
        return {"converted": True}

    def _normalize_dict(data):
        calls["normalize"] = data

    def _write_yaml(path, data, **kwargs):
        calls["write"] = (path, data)

    library_io_mod.writeYAML = _write_yaml
    custom_yaml_mod.load_yaml_stream = _load_yaml_stream
    merge_lib_mod.convertToDict = _convert_to_dict
    merge_lib_mod.normalizeDictLibraryLayout = _normalize_dict
    tensile_mod.LibraryIO = library_io_mod

    patch = {
        "Tensile": tensile_mod,
        "Tensile.LibraryIO": library_io_mod,
        "Tensile.CustomYamlLoader": custom_yaml_mod,
        "Tensile.TensileMergeLibrary": merge_lib_mod,
    }
    return patch


def test_normalize_raises_if_library_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Library path not found"):
        operations.normalize(tmp_path / "missing.yaml", tmp_path / "out.yaml")


def test_normalize_raises_if_hipblaslt_path_not_found(tmp_path: Path) -> None:
    lib = tmp_path / "lib.yaml"
    lib.touch()
    with pytest.raises(FileNotFoundError, match="hipBLASLt path not found"):
        operations.normalize(lib, tmp_path / "out.yaml", hipblaslt_path=tmp_path / "missing")


def test_normalize_raises_if_data_not_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lib = tmp_path / "lib.yaml"
    lib.touch()

    calls: dict = {}
    mocks = _make_tensile_mocks(calls)
    # Override load to return a non-list (dict format is unsupported as input)
    mocks["Tensile.CustomYamlLoader"].load_yaml_stream = lambda *_a, **_k: {"dict": "format"}
    for mod_name, mod in mocks.items():
        monkeypatch.setitem(sys.modules, mod_name, mod)

    with pytest.raises(ValueError, match="not in list format"):
        operations.normalize(lib, tmp_path / "out.yaml")


def test_normalize_calls_tensile_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lib = tmp_path / "lib.yaml"
    yaml.safe_dump([None, None, "gfx950"], lib.open("w"))
    out = tmp_path / "out.yaml"

    calls: dict = {}
    for mod_name, mod in _make_tensile_mocks(calls).items():
        monkeypatch.setitem(sys.modules, mod_name, mod)

    operations.normalize(lib, out)

    assert calls["load"] == lib
    assert calls["convert"] == str(lib)
    assert calls["normalize"] == {"converted": True}
    assert calls["write"] == (str(out), {"converted": True})


def test_normalize_appends_tensile_to_sys_path_when_hipblaslt_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lib = tmp_path / "lib.yaml"
    lib.touch()
    hip = tmp_path / "hip"
    hip.mkdir()
    expected = str(hip / "tensilelite")

    calls: dict = {}
    for mod_name, mod in _make_tensile_mocks(calls).items():
        monkeypatch.setitem(sys.modules, mod_name, mod)

    original_path = list(sys.path)
    operations.normalize(lib, tmp_path / "out.yaml", hipblaslt_path=hip)

    assert expected in sys.path
    # cleanup to avoid polluting other tests
    if expected not in original_path:
        sys.path.remove(expected)


# ---------------------------------------------------------------------------
# Helpers for default_solution mismatch tests
# ---------------------------------------------------------------------------

def _write_library_yaml_with_default(path: Path, lib_name: str, default_solution) -> Path:
    """Write a minimal library YAML, optionally setting DefaultSolution (index 12)."""
    data = [
        None,
        None,
        "gfx950",
        None,
        {
            "TransposeA": 0,
            "TransposeB": 0,
            "DataType": 0,
            "DestDataType": 0,
            "ComputeDataType": 0,
        },
        [{"SolutionIndex": 0, "StaggerU": 0}],
        [2, 3, 0, 1],
        [[[16, 16, 1, 16], [0, 100.0]]],
        None,
        None,
        "DeviceEfficiency",
        "Equality",
        default_solution,  # index 12 – DefaultSolution
    ]
    p = path / lib_name
    yaml.safe_dump(data, p.open("w"), sort_keys=False)
    return p


# ---------------------------------------------------------------------------
# merge_solutions – default_solution mismatch checks (line 181)
# ---------------------------------------------------------------------------

def test_merge_solutions_raises_on_default_solution_mismatch(tmp_path: Path) -> None:
    ldir1 = tmp_path / "b1" / "3_LibraryLogic"
    ldir2 = tmp_path / "b2" / "3_LibraryLogic"
    ldir1.mkdir(parents=True)
    ldir2.mkdir(parents=True)
    _write_library_yaml_with_default(ldir1, "x.yaml", {"StaggerU": 0})
    _write_library_yaml_with_default(ldir2, "x.yaml", {"StaggerU": 1})

    with pytest.raises(NotImplementedError, match="Default solution mismatch"):
        operations.merge_solutions(tmp_path, epilogues=False)


def test_merge_solutions_ok_with_matching_default_solution(tmp_path: Path) -> None:
    ldir1 = tmp_path / "b1" / "3_LibraryLogic"
    ldir2 = tmp_path / "b2" / "3_LibraryLogic"
    ldir1.mkdir(parents=True)
    ldir2.mkdir(parents=True)
    _write_library_yaml_with_default(ldir1, "x.yaml", {"StaggerU": 0})
    _write_library_yaml_with_default(ldir2, "x.yaml", {"StaggerU": 0})

    merged = operations.merge_solutions(tmp_path, epilogues=False)
    assert len(merged) == 1


def test_merge_solutions_ok_when_both_default_solution_none(tmp_path: Path) -> None:
    ldir1 = tmp_path / "b1" / "3_LibraryLogic"
    ldir2 = tmp_path / "b2" / "3_LibraryLogic"
    ldir1.mkdir(parents=True)
    ldir2.mkdir(parents=True)
    _write_library_yaml_with_default(ldir1, "x.yaml", None)
    _write_library_yaml_with_default(ldir2, "x.yaml", None)

    merged = operations.merge_solutions(tmp_path, epilogues=False)
    assert len(merged) == 1


# ---------------------------------------------------------------------------
# extract_solutions – default_solution mismatch checks (line 268)
# ---------------------------------------------------------------------------

def test_extract_solutions_raises_on_default_solution_mismatch(tmp_path: Path) -> None:
    lib_dir1 = tmp_path / "libs1"
    lib_dir2 = tmp_path / "libs2"
    lib_dir1.mkdir()
    lib_dir2.mkdir()
    p1 = _write_library_yaml_with_default(lib_dir1, "a.yaml", {"StaggerU": 0})
    p2 = _write_library_yaml_with_default(lib_dir2, "a.yaml", {"StaggerU": 1})

    # Two match-table entries pointing to different paths but same filename
    mt = _write_match_table(tmp_path, [[str(p1), 0], [str(p2), 0]])
    df = pd.DataFrame(
        [
            {"m": 16, "n": 16, "batch_count": 1, "k": 16, "solutionIdx": 0},
            {"m": 32, "n": 32, "batch_count": 1, "k": 32, "solutionIdx": 1},
        ]
    )

    with pytest.raises(NotImplementedError, match="Default solution mismatch"):
        operations.extract_solutions(df, mt)


def test_extract_solutions_ok_with_matching_default_solution(tmp_path: Path) -> None:
    lib_dir1 = tmp_path / "libs1"
    lib_dir2 = tmp_path / "libs2"
    lib_dir1.mkdir()
    lib_dir2.mkdir()
    p1 = _write_library_yaml_with_default(lib_dir1, "a.yaml", {"StaggerU": 0})
    p2 = _write_library_yaml_with_default(lib_dir2, "a.yaml", {"StaggerU": 0})

    mt = _write_match_table(tmp_path, [[str(p1), 0], [str(p2), 0]])
    df = pd.DataFrame(
        [
            {"m": 16, "n": 16, "batch_count": 1, "k": 16, "solutionIdx": 0},
            {"m": 32, "n": 32, "batch_count": 1, "k": 32, "solutionIdx": 1},
        ]
    )

    libs = operations.extract_solutions(df, mt)
    assert len(libs) == 1


def test_extract_solutions_ok_when_both_default_solution_none(tmp_path: Path) -> None:
    lib_dir1 = tmp_path / "libs1"
    lib_dir2 = tmp_path / "libs2"
    lib_dir1.mkdir()
    lib_dir2.mkdir()
    p1 = _write_library_yaml_with_default(lib_dir1, "a.yaml", None)
    p2 = _write_library_yaml_with_default(lib_dir2, "a.yaml", None)

    mt = _write_match_table(tmp_path, [[str(p1), 0], [str(p2), 0]])
    df = pd.DataFrame(
        [
            {"m": 16, "n": 16, "batch_count": 1, "k": 16, "solutionIdx": 0},
            {"m": 32, "n": 32, "batch_count": 1, "k": 32, "solutionIdx": 1},
        ]
    )

    libs = operations.extract_solutions(df, mt)
    assert len(libs) == 1
