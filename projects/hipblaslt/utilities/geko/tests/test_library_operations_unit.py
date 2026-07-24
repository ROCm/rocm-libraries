# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

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
