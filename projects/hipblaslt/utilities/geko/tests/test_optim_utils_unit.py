# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from geko.optim import utils as outils


def test_get_checkpoint_file_and_multiple(tmp_path: Path) -> None:
    build = tmp_path / "build_a"
    build.mkdir()
    assert outils.get_checkpoint_file(build) is None

    cp = build / "a.checkpoint"
    cp.write_text("x", encoding="utf-8")
    assert outils.get_checkpoint_file(build) == cp

    (build / "b.checkpoint").write_text("y", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected at most 1 checkpoint"):
        outils.get_checkpoint_file(build)


def test_get_build_state_variants(tmp_path: Path) -> None:
    assert outils.get_build_state(tmp_path / "missing") == "missing"

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / ".running").write_text("", encoding="utf-8")
    assert outils.get_build_state(run_dir) == "running"

    comp = tmp_path / "comp"
    (comp / "3_LibraryLogic").mkdir(parents=True)
    (comp / "3_LibraryLogic/a.yaml").write_text("x", encoding="utf-8")
    assert outils.get_build_state(comp) == "completed"

    res = tmp_path / "res"
    res.mkdir()
    (res / "state.checkpoint").write_text("x", encoding="utf-8")
    assert outils.get_build_state(res) == "resumable"

    fail = tmp_path / "fail"
    fail.mkdir()
    (fail / "junk.txt").write_text("x", encoding="utf-8")
    assert outils.get_build_state(fail) == "failed"


def test_list_configs_and_failed_optimizations(tmp_path: Path) -> None:
    (tmp_path / "job_2.yaml").write_text("x", encoding="utf-8")
    (tmp_path / "job_10.yaml").write_text("x", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("x", encoding="utf-8")

    cfgs = outils.list_optimization_configs(tmp_path)
    assert [Path(c).name for c in cfgs] == ["job_2.yaml", "job_10.yaml"]

    build_fail = tmp_path / "build_job_2"
    build_fail.mkdir()
    (build_fail / "x.txt").write_text("x", encoding="utf-8")
    build_ok = tmp_path / "build_job_10/3_LibraryLogic"
    build_ok.mkdir(parents=True)
    (build_ok / "out.yaml").write_text("x", encoding="utf-8")

    failed = outils.get_failed_optimizations(tmp_path)
    assert failed == ["job_2"]


def test_clean_failed_build_variants(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    outils.clean_failed_build(missing)

    comp = tmp_path / "comp/3_LibraryLogic"
    comp.mkdir(parents=True)
    (comp / "x").write_text("x", encoding="utf-8")
    outils.clean_failed_build(comp.parent)
    assert comp.parent.is_dir()

    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "junk").write_text("x", encoding="utf-8")
    outils.clean_failed_build(failed)
    assert not failed.exists()

    failed_cp = tmp_path / "failed_cp"
    failed_cp.mkdir()
    cp = failed_cp / "keep.checkpoint"
    cp.write_text("x", encoding="utf-8")
    (failed_cp / "trash.txt").write_text("x", encoding="utf-8")
    (failed_cp / "trash_dir").mkdir()
    outils.clean_failed_build(failed_cp)
    assert failed_cp.is_dir()
    assert cp.exists()
    assert not (failed_cp / "trash.txt").exists()
    assert not (failed_cp / "trash_dir").exists()


def test_clean_failed_build_removes_stale_running_marker(tmp_path: Path) -> None:
    b = tmp_path / "build_x"
    b.mkdir()
    (b / ".running").write_text("", encoding="utf-8")
    (b / "junk").write_text("x", encoding="utf-8")
    outils.clean_failed_build(b)
    assert not b.exists()


def test_clean_failed_build_unsupported_state_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    b = tmp_path / "build_x"
    b.mkdir()
    monkeypatch.setattr(outils, "get_build_state", lambda _p: "unknown")
    with pytest.raises(ValueError, match="Unsupported build state"):
        outils.clean_failed_build(b)


def test_clean_failed_builds_calls_per_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "job_1.yaml").write_text("x", encoding="utf-8")
    (tmp_path / "job_2.yaml").write_text("x", encoding="utf-8")

    seen = []
    monkeypatch.setattr(outils, "clean_failed_build", lambda p: seen.append(Path(p).name))
    outils.clean_failed_builds(tmp_path)
    assert seen == ["build_job_1", "build_job_2"]


def test_check_progress_counts_states(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(outils, "list_optimization_configs", lambda _p: ["a.yaml", "b.yaml", "c.yaml"])

    def _state(path):
        n = Path(path).name
        if n == "build_a":
            return "completed"
        if n == "build_b":
            return "failed"
        return "running"

    monkeypatch.setattr(outils, "get_build_state", _state)
    total, done, failed = outils.check_progress(tmp_path)
    assert (total, done, failed) == (3, 1, 1)


def test_estimate_workload_happy_path_and_pop_adjustments(tmp_path: Path) -> None:
    conf = {
        "GlobalParameters": {"EnqueuesPerSync": 2, "NumWarmups": 1},
        "BenchmarkProblems": [
            [
                "x",
                {
                    "BenchmarkFinalParameters": [
                        {"ProblemSizes": [{"Exact": [16, 16, 1, 16]}, {"Exact": [32, 32, 1, 32]}]}
                    ],
                    "ForkParameters": [{"Groups": [[1, 2, 3], [4]]}, {"ParamX": [1, 2, 3, 4, 5]}],
                },
            ]
        ],
    }
    p = tmp_path / "cfg.yaml"
    yaml.safe_dump(conf, p.open("w"), sort_keys=False)
    w = outils.estimate_workload(p, pop_size=8)
    assert w > 0


def test_estimate_workload_validation_errors(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    yaml.safe_dump({"x": 1}, p.open("w"), sort_keys=False)
    with pytest.raises(ValueError, match="Missing required keys"):
        outils.estimate_workload(p)

    p2 = tmp_path / "bad2.yaml"
    yaml.safe_dump({"GlobalParameters": {}, "BenchmarkProblems": []}, p2.open("w"), sort_keys=False)
    with pytest.raises(ValueError, match="BenchmarkProblems must be non-empty list"):
        outils.estimate_workload(p2)

    p3 = tmp_path / "bad3.yaml"
    yaml.safe_dump(
        {
            "GlobalParameters": {},
            "BenchmarkProblems": [["x", {"BenchmarkFinalParameters": [{"NoProblemSizes": []}], "ForkParameters": [{"Groups": [[1]]}]}]],
        },
        p3.open("w"),
        sort_keys=False,
    )
    with pytest.raises(ValueError, match="No 'ProblemSizes' found"):
        outils.estimate_workload(p3)

    p4 = tmp_path / "bad4.yaml"
    yaml.safe_dump(
        {
            "GlobalParameters": {},
            "BenchmarkProblems": [["x", {"BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": [16, 16, 1, 16]}]}], "ForkParameters": [{"X": [1]}]}]],
        },
        p4.open("w"),
        sort_keys=False,
    )
    with pytest.raises(ValueError, match="No 'Groups' found"):
        outils.estimate_workload(p4)
