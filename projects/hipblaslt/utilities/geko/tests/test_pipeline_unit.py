# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from geko import pipeline
from geko.schemas import RunState


def _make_workload(path: Path) -> Path:
    path.write_text("[]\n")
    return path


def test_run_bench_returns_one_for_missing_hip_path(tmp_path: Path) -> None:
    workload = _make_workload(tmp_path / "wkld.yaml")
    rc = pipeline.run_bench(str(tmp_path / "missing"), str(workload), tmp_path / "out")
    assert rc == 1


def test_run_bench_returns_one_for_missing_workload(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    rc = pipeline.run_bench(str(hip), str(tmp_path / "missing.yaml"), tmp_path / "out")
    assert rc == 1


def test_run_bench_happy_path_invokes_standard_benchmark(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workload = _make_workload(tmp_path / "wkld.yaml")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(pipeline.bench.log, "parse", lambda *_args, **_kwargs: [{"M": 16}])
    monkeypatch.setattr(pipeline.bench.log, "update", lambda rows: (rows, None))

    dumped = {}

    def _fake_dump(_data, out_file):
        dumped["path"] = Path(out_file)
        dumped["path"].write_text("[]\n")

    monkeypatch.setattr(pipeline.bench.log, "dump", _fake_dump)

    called = {}

    def _fake_standard(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return pd.DataFrame()

    monkeypatch.setattr(pipeline.bench, "standard_benchmark", _fake_standard)

    rc = pipeline.run_bench(str(hip), str(workload), out_dir, devices=[0], benchmark_duration=0.1)
    assert rc == 0
    assert dumped["path"].is_file()
    assert called["kwargs"]["devices"] == [0]
    assert called["kwargs"]["duration"] == 0.1


def test_run_bench_device_alias_overrides_devices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workload = _make_workload(tmp_path / "wkld.yaml")

    monkeypatch.setattr(pipeline.bench.log, "parse", lambda *_args, **_kwargs: [{"M": 16}])
    monkeypatch.setattr(pipeline.bench.log, "update", lambda rows: (rows, None))
    monkeypatch.setattr(pipeline.bench.log, "dump", lambda _data, out_file: Path(out_file).write_text("[]\n"))

    called = {}

    def _fake_standard(*_args, **kwargs):
        called["devices"] = kwargs["devices"]
        return pd.DataFrame()

    monkeypatch.setattr(pipeline.bench, "standard_benchmark", _fake_standard)

    rc = pipeline.run_bench(str(hip), str(workload), tmp_path / "out", devices=[0, 1], device=2)
    assert rc == 0
    assert called["devices"] == [2]


def test_run_configure_rejects_invalid_device(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workload = _make_workload(tmp_path / "wkld.yaml")
    with pytest.raises(ValueError, match="Device ID"):
        pipeline.run_configure(str(hip), str(workload), devices=[8])


def test_run_configure_rejects_invalid_arch(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workload = _make_workload(tmp_path / "wkld.yaml")
    with pytest.raises(ValueError, match="Must be one of"):
        pipeline.run_configure(str(hip), str(workload), devices=[0], arch="gfx000")


def test_run_configure_rejects_invalid_backend(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workload = _make_workload(tmp_path / "wkld.yaml")
    with pytest.raises(ValueError, match="backend"):
        pipeline.run_configure(str(hip), str(workload), devices=[0], backend="bad")


def test_run_configure_device_alias_overrides_devices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workload = _make_workload(tmp_path / "wkld.yaml")

    called = {}

    def _fake_summarize(*_args, **kwargs):
        called["devices"] = kwargs["devices"]
        return pd.DataFrame(), pd.DataFrame([{"M": 16}])

    monkeypatch.setattr(pipeline.bench.log, "summarize", _fake_summarize)
    monkeypatch.setattr(pipeline, "gemm_configs_from_gemm_dataframe", lambda _df: [MagicMock()])
    monkeypatch.setattr(pipeline.optim, "configure", lambda *_a, **_k: None)

    pipeline.run_configure(str(hip), str(workload), devices=[0, 1], device=2, workdir=str(tmp_path / "w"))
    assert called["devices"] == [2]


def test_run_search_returns_when_filtered_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workload = _make_workload(tmp_path / "wkld.yaml")

    monkeypatch.setattr(
        pipeline.bench.log,
        "summarize",
        lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()),
    )

    pipeline.run_search(str(hip), str(workload), devices=[0], workdir=str(tmp_path / "w"))


def test_run_search_raises_when_no_winners(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workload = _make_workload(tmp_path / "wkld.yaml")

    summary = pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 16,
                "n": 16,
                "k": 16,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "f32_r",
            }
        ]
    )

    monkeypatch.setattr(pipeline.bench.log, "summarize", lambda *_a, **_k: (summary, summary))
    monkeypatch.setattr(pipeline.search, "configure", lambda *_a, **_k: [{"M": 16}])
    monkeypatch.setattr(pipeline.search, "run", lambda *_a, **_k: pd.DataFrame())

    with pytest.raises(ValueError, match="All benchmarks failed"):
        pipeline.run_search(str(hip), str(workload), devices=[0], workdir=str(tmp_path / "w"))


def test_run_search_success_writes_final_libs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    (hip / "build/release/device-library").mkdir(parents=True)
    workload = _make_workload(tmp_path / "wkld.yaml")
    workdir = tmp_path / "w"

    summary = pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 16,
                "n": 16,
                "k": 16,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "c_f32_r",
            }
        ]
    )

    winners = summary.copy()
    winners["solutionIdx"] = 0

    monkeypatch.setattr(pipeline.bench.log, "summarize", lambda *_a, **_k: (summary, summary))
    monkeypatch.setattr(pipeline.search, "configure", lambda *_a, **_k: [{"M": 16}])
    monkeypatch.setattr(pipeline.search, "run", lambda *_a, **_k: winners)

    class _FakeLibs(list):
        def dump(self, out_dir):
            p = Path(out_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "lib.yaml").write_text("x\n")

    monkeypatch.setattr(pipeline.library.operations, "extract_solutions", lambda *_a, **_k: _FakeLibs())
    monkeypatch.setattr(pipeline.optim, "analyze", lambda *_a, **_k: (pd.DataFrame([{"dummy": 1}]), pd.DataFrame([{"dummy": 1, "winner": "reference", "lib": "lib.yaml"}])))
    monkeypatch.setattr(
        pipeline.library,
        "from_dataframe",
        lambda *_a, **_k: _FakeLibs(),
    )
    monkeypatch.setattr(
        pipeline.library,
        "from_full_dataframe",
        lambda *_a, **_k: _FakeLibs(),
    )

    pipeline.run_search(str(hip), str(workload), devices=[0], workdir=str(workdir))
    assert (workdir / "final_libs" / "lib.yaml").is_file()


def test_run_search_success_without_analysis_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    (hip / "build/release/device-library").mkdir(parents=True)
    workload = _make_workload(tmp_path / "wkld.yaml")
    workdir = tmp_path / "w"

    summary = pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 16,
                "n": 16,
                "k": 16,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "c_f32_r",
            }
        ]
    )
    winners = summary.copy()
    winners["solutionIdx"] = 0

    monkeypatch.setattr(pipeline.bench.log, "summarize", lambda *_a, **_k: (summary, summary))
    monkeypatch.setattr(pipeline.search, "configure", lambda *_a, **_k: [{"M": 16}])
    monkeypatch.setattr(pipeline.search, "run", lambda *_a, **_k: winners)

    class _FakeLibs(list):
        def dump(self, out_dir):
            p = Path(out_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "lib.yaml").write_text("x\n")

    monkeypatch.setattr(pipeline.library.operations, "extract_solutions", lambda *_a, **_k: _FakeLibs())
    monkeypatch.setattr(pipeline.optim, "analyze", lambda *_a, **_k: (None, pd.DataFrame([{"dummy": 1, "winner": "reference", "lib": "lib.yaml"}])))
    monkeypatch.setattr(
        pipeline.library,
        "from_full_dataframe",
        lambda *_a, **_k: _FakeLibs(),
    )

    pipeline.run_search(str(hip), str(workload), devices=[0], workdir=str(workdir))
    assert not (workdir / "final_libs").exists()


def test_run_search_keeps_existing_artifacts_when_winners_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    (hip / "build/release/device-library").mkdir(parents=True)
    workload = _make_workload(tmp_path / "wkld.yaml")
    workdir = tmp_path / "w"
    workdir.mkdir()

    summary = pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 16,
                "n": 16,
                "k": 16,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "c_f32_r",
            }
        ]
    )
    winners = summary.copy()
    winners["solutionIdx"] = 0
    winners.to_csv(workdir / "winners.csv", index=False)

    lib_dir = workdir / "libs"
    build_dir = workdir / "build"
    bench_dir = workdir / "benchmarks"
    for p in (lib_dir, build_dir, bench_dir):
        p.mkdir(parents=True)
        (p / "keep.txt").write_text("keep\n")

    monkeypatch.setattr(pipeline.bench.log, "summarize", lambda *_a, **_k: (summary, summary))
    monkeypatch.setattr(pipeline.search, "configure", lambda *_a, **_k: [{"M": 16}])
    monkeypatch.setattr(pipeline.search, "run", lambda *_a, **_k: winners)

    class _FakeLibs(list):
        def dump(self, out_dir):
            p = Path(out_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "lib.yaml").write_text("x\n")

    monkeypatch.setattr(pipeline.library.operations, "extract_solutions", lambda *_a, **_k: _FakeLibs())
    monkeypatch.setattr(pipeline.optim, "analyze", lambda *_a, **_k: (None, pd.DataFrame([{"dummy": 1, "winner": "reference", "lib": "lib.yaml"}])))
    monkeypatch.setattr(
        pipeline.library,
        "from_full_dataframe",
        lambda *_a, **_k: _FakeLibs(),
    )

    pipeline.run_search(str(hip), str(workload), devices=[0], workdir=str(workdir))
    assert (lib_dir / "keep.txt").is_file()
    assert (build_dir / "keep.txt").is_file()
    assert (bench_dir / "keep.txt").is_file()


def test_run_optimize_requires_existing_workdir(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    with pytest.raises(FileNotFoundError, match="Working directory"):
        pipeline.run_optimize(str(hip), workdir=str(tmp_path / "missing"), devices=[0])


def test_run_optimize_requires_configured_state(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workdir = tmp_path / "w"
    workdir.mkdir()
    workload = _make_workload(tmp_path / "wkld.yaml")
    state = RunState.create(workload)
    state.configured = False
    state.dump(workdir / "run_state.json")
    with pytest.raises(ValueError, match="Configuration step"):
        pipeline.run_optimize(str(hip), workdir=str(workdir), devices=[0])


def test_run_optimize_requires_optimizations_dir(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workdir = tmp_path / "w"
    workdir.mkdir()
    workload = _make_workload(tmp_path / "wkld.yaml")
    state = RunState.create(workload)
    state.configured = True
    state.dump(workdir / "run_state.json")
    with pytest.raises(FileNotFoundError, match="Optimizations directory"):
        pipeline.run_optimize(str(hip), workdir=str(workdir), devices=[0])


def test_run_optimize_raises_for_no_completed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workdir = tmp_path / "w"
    (workdir / "optimizations").mkdir(parents=True)

    workload = _make_workload(tmp_path / "wkld.yaml")
    state = RunState.create(workload)
    state.configured = True
    state.dump(workdir / "run_state.json")

    monkeypatch.setattr(
        pipeline.optim.utils,
        "check_progress",
        MagicMock(side_effect=[(1, 0, 0), (1, 0, 1)]),
    )
    monkeypatch.setattr(pipeline.optim, "run", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="No GEMMs completed"):
        pipeline.run_optimize(str(hip), workdir=str(workdir), devices=[0])


def test_run_optimize_success_writes_final_libs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workdir = tmp_path / "w"
    (workdir / "optimizations").mkdir(parents=True)

    workload = _make_workload(tmp_path / "wkld.yaml")
    state = RunState.create(workload)
    state.configured = True
    state.dump(workdir / "run_state.json")

    monkeypatch.setattr(
        pipeline.optim.utils,
        "check_progress",
        MagicMock(side_effect=[(1, 1, 0), (1, 1, 0)]),
    )
    monkeypatch.setattr(pipeline.optim, "run", lambda *_a, **_k: None)

    class _FakeLibs(list):
        def dump(self, out_dir):
            p = Path(out_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "lib.yaml").write_text("x\n")

    monkeypatch.setattr(pipeline.library, "merge_solutions", lambda *_a, **_k: _FakeLibs())
    monkeypatch.setattr(pipeline.optim, "analyze", lambda *_a, **_k: (pd.DataFrame([{"dummy": 1}]), pd.DataFrame([{"dummy": 1, "winner": "reference", "lib": "lib.yaml"}])))
    monkeypatch.setattr(pipeline.library, "from_dataframe", lambda *_a, **_k: _FakeLibs())
    monkeypatch.setattr(pipeline.library, "from_full_dataframe", lambda *_a, **_k: _FakeLibs())

    pipeline.run_optimize(str(hip), workdir=str(workdir), devices=[0])
    assert (workdir / "final_libs" / "lib.yaml").is_file()


def test_run_optimize_success_without_analysis_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workdir = tmp_path / "w"
    (workdir / "optimizations").mkdir(parents=True)

    workload = _make_workload(tmp_path / "wkld.yaml")
    state = RunState.create(workload)
    state.configured = True
    state.dump(workdir / "run_state.json")

    monkeypatch.setattr(
        pipeline.optim.utils,
        "check_progress",
        MagicMock(side_effect=[(1, 1, 0), (1, 1, 0)]),
    )
    monkeypatch.setattr(pipeline.optim, "run", lambda *_a, **_k: None)

    class _FakeLibs(list):
        def dump(self, out_dir):
            p = Path(out_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "lib.yaml").write_text("x\n")

    monkeypatch.setattr(pipeline.library, "merge_solutions", lambda *_a, **_k: _FakeLibs())
    monkeypatch.setattr(pipeline.optim, "analyze", lambda *_a, **_k: (None, pd.DataFrame([{"dummy": 1, "winner": "reference", "lib": "lib.yaml"}])))
    monkeypatch.setattr(pipeline.library, "from_full_dataframe", lambda *_a, **_k: _FakeLibs())

    pipeline.run_optimize(str(hip), workdir=str(workdir), devices=[0])
    assert not (workdir / "final_libs").exists()


def test_run_optimize_skips_cleanup_when_not_needed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workdir = tmp_path / "w"
    (workdir / "optimizations").mkdir(parents=True)

    workload = _make_workload(tmp_path / "wkld.yaml")
    state = RunState.create(workload)
    state.configured = True
    state.dump(workdir / "run_state.json")

    for p in (workdir / "libs", workdir / "build", workdir / "benchmarks"):
        p.mkdir(parents=True)
        (p / "keep.txt").write_text("keep\n")

    monkeypatch.setattr(
        pipeline.optim.utils,
        "check_progress",
        MagicMock(side_effect=[(2, 2, 0), (2, 2, 0)]),
    )
    monkeypatch.setattr(pipeline.optim, "run", lambda *_a, **_k: None)

    class _FakeLibs(list):
        def dump(self, out_dir):
            p = Path(out_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "lib.yaml").write_text("x\n")

    monkeypatch.setattr(pipeline.library, "merge_solutions", lambda *_a, **_k: _FakeLibs())
    monkeypatch.setattr(pipeline.optim, "analyze", lambda *_a, **_k: (None, pd.DataFrame([{"dummy": 1, "winner": "reference", "lib": "lib.yaml"}])))
    monkeypatch.setattr(pipeline.library, "from_full_dataframe", lambda *_a, **_k: _FakeLibs())

    pipeline.run_optimize(str(hip), workdir=str(workdir), devices=[0], retry=False)
    assert (workdir / "build" / "keep.txt").is_file()
    assert (workdir / "benchmarks" / "keep.txt").is_file()
