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
    assert (out_dir / "wkld.csv").is_file()
    assert called["kwargs"]["devices"] == [0]
    assert called["kwargs"]["duration"] == 0.1
    assert called["kwargs"]["custom_lib_dir"] is None


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


def test_run_bench_uses_custom_lib_dir_over_custom_lib_src(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workload = _make_workload(tmp_path / "wkld.yaml")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    custom_src = tmp_path / "src_libs"
    custom_src.mkdir()
    custom_dir = tmp_path / "custom_build"
    custom_dir.mkdir()

    monkeypatch.setattr(pipeline.bench.log, "parse", lambda *_args, **_kwargs: [{"M": 16}])
    monkeypatch.setattr(pipeline.bench.log, "update", lambda rows: (rows, None))
    monkeypatch.setattr(pipeline.bench.log, "dump", lambda _data, out_file: Path(out_file).write_text("[]\n"))

    create_called = {"n": 0}
    monkeypatch.setattr(
        pipeline.library.operations,
        "create",
        lambda *_a, **_k: create_called.__setitem__("n", create_called["n"] + 1),
    )

    called = {}

    def _fake_standard(*_args, **kwargs):
        called["custom_lib_dir"] = kwargs["custom_lib_dir"]
        return pd.DataFrame()

    monkeypatch.setattr(pipeline.bench, "standard_benchmark", _fake_standard)

    rc = pipeline.run_bench(
        str(hip),
        str(workload),
        out_dir,
        devices=[0],
        custom_lib_src=str(custom_src),
        custom_lib_dir=str(custom_dir),
    )
    assert rc == 0
    assert create_called["n"] == 0
    assert called["custom_lib_dir"] == custom_dir


def test_run_bench_builds_custom_lib_from_src(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workload = _make_workload(tmp_path / "wkld.yaml")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    custom_src = tmp_path / "src_libs"
    custom_src.mkdir()

    monkeypatch.setattr(pipeline.bench.log, "parse", lambda *_args, **_kwargs: [{"M": 16}])
    monkeypatch.setattr(pipeline.bench.log, "update", lambda rows: (rows, None))
    monkeypatch.setattr(pipeline.bench.log, "dump", lambda _data, out_file: Path(out_file).write_text("[]\n"))

    created = {}

    def _fake_create(_hip, src, out):
        created["src"] = Path(src)
        created["out"] = Path(out)

    monkeypatch.setattr(pipeline.library.operations, "create", _fake_create)

    called = {}

    def _fake_standard(*_args, **kwargs):
        called["custom_lib_dir"] = kwargs["custom_lib_dir"]
        return pd.DataFrame()

    monkeypatch.setattr(pipeline.bench, "standard_benchmark", _fake_standard)

    rc = pipeline.run_bench(
        str(hip),
        str(workload),
        out_dir,
        devices=[0],
        custom_lib_src=str(custom_src),
    )
    assert rc == 0
    assert created["src"] == custom_src
    assert created["out"] == out_dir / "ref_build"
    assert called["custom_lib_dir"] == out_dir / "ref_build"


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


def test_run_configure_rejects_mx_on_unsupported_arch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from geko.schemas import GemmConfig, GemmType

    hip = tmp_path / "hip"
    hip.mkdir()
    workload = _make_workload(tmp_path / "wkld.yaml")

    monkeypatch.setattr(
        pipeline.bench.log,
        "summarize",
        lambda *_args, **kwargs: (pd.DataFrame(), pd.DataFrame([{"M": 16}])),
    )
    gt = GemmType.from_tensile("T", "N", "F8", "S", "S")
    gc = GemmConfig(gt, [[256, 256, 1, 256]], mx=True)
    monkeypatch.setattr(pipeline, "gemm_configs_from_gemm_dataframe", lambda _df: [gc])

    with pytest.raises(ValueError, match="MX .* is not supported on ARCH 'gfx942'"):
        pipeline.run_configure(str(hip), str(workload), devices=[0], arch="gfx942", workdir=str(tmp_path / "w"))


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


def test_run_search_forces_benchmark_on_summarize(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workload = _make_workload(tmp_path / "wkld.yaml")

    called = {}

    def _fake_summarize(*_args, **kwargs):
        called["force_benchmark"] = kwargs.get("force_benchmark")
        called["keep_thr"] = kwargs.get("keep_thr")
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(pipeline.bench.log, "summarize", _fake_summarize)

    pipeline.run_search(str(hip), str(workload), devices=[0], keep_thr=0.0, workdir=str(tmp_path / "w"))

    assert called["force_benchmark"] is True
    assert called["keep_thr"] == 0.0


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

    extracted = {}

    def _fake_extract(_winners, match_table, **_kwargs):
        extracted["match_table"] = Path(match_table)
        return _FakeLibs()

    monkeypatch.setattr(pipeline.library.operations, "extract_solutions", _fake_extract)

    analyze_called = {}

    def _fake_analyze(*_a, **kwargs):
        analyze_called["ref_custom_lib_dir"] = kwargs.get("ref_custom_lib_dir")
        return pd.DataFrame([{"dummy": 1}]), pd.DataFrame([{"dummy": 1, "winner": "reference", "lib": "lib.yaml"}])

    monkeypatch.setattr(pipeline.optim, "analyze", _fake_analyze)
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
    assert extracted["match_table"] == hip / "build/release/device-library/MatchTable.yaml"
    assert analyze_called["ref_custom_lib_dir"] is None


def test_run_search_with_custom_lib_src_builds_ref_lib_and_uses_ref_matchtable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hip = tmp_path / "hip"
    (hip / "build/release/device-library").mkdir(parents=True)
    workload = _make_workload(tmp_path / "wkld.yaml")
    workdir = tmp_path / "w"
    src_lib = tmp_path / "src_lib"
    src_lib.mkdir()

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

    cwd_match = tmp_path / "MatchTable.yaml"
    cwd_match.write_text("{}\n")

    created = {}

    def _fake_create(_hip, src, out):
        created["src"] = Path(src)
        created["out"] = Path(out)

    monkeypatch.setattr(pipeline.library.operations, "create", _fake_create)

    extracted = {}

    def _fake_extract(_winners, match_table, **_kwargs):
        extracted["match_table"] = Path(match_table)
        return _FakeLibs()

    monkeypatch.setattr(pipeline.library.operations, "extract_solutions", _fake_extract)

    analyze_called = {}

    def _fake_analyze(*_a, **kwargs):
        analyze_called["ref_custom_lib_dir"] = kwargs.get("ref_custom_lib_dir")
        return None, pd.DataFrame([{"dummy": 1, "winner": "reference", "lib": "lib.yaml"}])

    monkeypatch.setattr(pipeline.optim, "analyze", _fake_analyze)
    monkeypatch.setattr(pipeline.library, "from_full_dataframe", lambda *_a, **_k: _FakeLibs())

    prev_cwd = Path.cwd()
    try:
        import os
        os.chdir(tmp_path)
        pipeline.run_search(
            str(hip),
            str(workload),
            devices=[0],
            workdir=str(workdir),
            custom_lib_src=str(src_lib),
        )
    finally:
        import os
        os.chdir(prev_cwd)

    assert created["src"] == src_lib
    assert created["out"] == workdir / "ref_build"
    assert (workdir / "ref_matchtable.yaml").is_file()
    assert extracted["match_table"] == workdir / "ref_matchtable.yaml"
    assert analyze_called["ref_custom_lib_dir"] == workdir / "ref_build"


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


def _logic_payload(sizes: list) -> dict:
    return {
        "ArchitectureName": "gfx950",
        "ScheduleName": "gfx950",
        "ProblemType": {
            "DataType": 7,
            "DestDataType": 7,
            "ComputeDataType": 0,
            "TransposeA": True,
            "TransposeB": False,
            "HighPrecisionAccumulate": True,
        },
        "DefaultSolution": 0,
        "Solutions": [{"SolutionIndex": 0, "SolutionNameMin": "kernel"}],
        "ExactLogic": sizes,
        "LibraryType": "Equality",
    }


def _write_completed_config(optim_dir: Path, name: str, sizes: list) -> Path:
    (optim_dir / f"{name}.yaml").write_text("MatrixInstruction: [16, 16, 16, 1]\n")
    logic = optim_dir / f"build_{name}" / "3_LibraryLogic"
    logic.mkdir(parents=True)
    path = logic / f"{name}.yaml"
    from geko.library import Library

    Library(_logic_payload(sizes), path.name).dump(logic, name=path.name)
    return path


def test_process_duplicates_updates_shared_size_performance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    optim_dir = tmp_path / "optimizations"
    optim_dir.mkdir()
    shared = [1024, 1024, 1, 1024]
    unique = [8, 8, 1, 8]
    only_a = [16, 16, 1, 16]
    path_a = _write_completed_config(
        optim_dir,
        "BBS_TN_0",
        [[shared, [3, 100.0]], [unique, [3, 10.0]]],
    )
    path_b = _write_completed_config(optim_dir, "BBS_TN_1", [[shared, [5, 200.0]]])
    path_c = _write_completed_config(optim_dir, "BBS_TN_2", [[only_a, [1, 50.0]]])

    created = []

    def _create(_hip, logic_dir, output_dir):
        created.append((Path(logic_dir), Path(output_dir)))

    def _bench(_hip, bench_file, _output, custom_lib_dir=None, devices=None):
        text = Path(bench_file).read_text()
        rows = []
        if "M: 1024" in text:
            rows.append({"m": 1024, "n": 1024, "batch_count": 1, "k": 1024, "hipblaslt-Gflops": 999.0})
        if "M: 8" in text:
            rows.append({"m": 8, "n": 8, "batch_count": 1, "k": 8, "hipblaslt-Gflops": 1.0})
        return pd.DataFrame(rows)

    monkeypatch.setattr(pipeline.library, "create", _create)
    monkeypatch.setattr(pipeline.bench, "run", _bench)

    pipeline.process_duplicates(str(tmp_path / "hip"), optim_dir, devices=[0])

    built = {logic.parent.name for logic, _out in created}
    assert built == {"build_BBS_TN_0", "build_BBS_TN_1"}

    from geko.library import load_library

    lib_a = load_library(path_a)
    lib_b = load_library(path_b)
    lib_c = load_library(path_c)
    assert lib_a.sizes[0] == [shared, [3, 999.0]]
    assert lib_a.sizes[1] == [unique, [3, 10.0]]
    assert lib_b.sizes[0] == [shared, [5, 999.0]]
    assert lib_c.sizes[0] == [only_a, [1, 50.0]]


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


def test_mx_detected_from_scale_columns_in_workload(tmp_path: Path) -> None:
    """MX should be detected per-GEMM from scaleA/scaleB in workload log."""
    from geko.config_generator.load_input_config import gemm_configs_from_gemm_dataframe
    import pandas as pd

    df = pd.DataFrame([
        {"transA": "T", "transB": "N", "a_type": "f8_r", "b_type": "f8_r",
         "c_type": "f32_r", "compute_type": "f32_r", "m": 256, "n": 256,
         "batch_count": 1, "k": 256, "d_type": "f32_r", "scaleA": 3, "scaleB": 3},
        {"transA": "N", "transB": "N", "a_type": "bf16_r", "b_type": "bf16_r",
         "c_type": "bf16_r", "compute_type": "f32_r", "m": 512, "n": 512,
         "batch_count": 1, "k": 256, "d_type": "bf16_r", "scaleA": 1, "scaleB": 1},
    ])
    gcs = gemm_configs_from_gemm_dataframe(df)
    assert len(gcs) == 2
    mx_gcs = [gc for gc in gcs if gc.mx]
    non_mx_gcs = [gc for gc in gcs if not gc.mx]
    assert len(mx_gcs) == 1
    assert mx_gcs[0].gemm_type.data_type == "F8"
    assert len(non_mx_gcs) == 1
