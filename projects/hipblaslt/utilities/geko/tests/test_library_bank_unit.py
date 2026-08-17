# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from geko.constants import GEMM_FIELDS
from geko.library import _bank


class _FakeLib:
    def __init__(self, solutions, sizes, bench_file: Path):
        self.solutions = solutions
        self.sizes = sizes
        self._bench_file = bench_file

    def create_bench_input(self, _cluster_dir):
        return (self._bench_file, None)


def _bench_rows() -> list[dict]:
    base = {
        "transA": "N",
        "transB": "N",
        "batch_count": 1,
        "a_type": "f16_r",
        "b_type": "f16_r",
        "c_type": "f16_r",
        "d_type": "f16_r",
        "compute_type": "c_f32_r",
    }
    rows = []
    for m in (16, 32):
        for sol, gflops in ((0, 100.0), (1, 81.0)):
            rows.append(
                {
                    **base,
                    "m": m,
                    "n": m,
                    "k": m,
                    "solutionIdx": sol,
                    "hipblaslt-Gflops": gflops,
                }
            )
    return rows


def test_cluster_solutions_groups_and_reindexes(tmp_path: Path) -> None:
    solutions = [
        {"MacroTile0": 64, "MacroTile1": 64, "DepthU": 16, "UseSgprForGRO": True},
        {"MacroTile0": 64, "MacroTile1": 64, "DepthU": 16, "UseSgprForGRO": True},
        {"MacroTile0": 128, "MacroTile1": 64, "DepthU": 16, "UseSgprForGRO": False},
    ]
    sizes = [
        [[16, 16, 1, 16], [0, 0.0]],
        [[32, 32, 1, 32], [1, 0.0]],
    ]
    lib = _FakeLib(solutions, sizes, tmp_path / "bench.yaml")

    out = _bank.cluster_solutions(lib, other_keys=["UseSgprForGRO"])
    assert len(out) == 1
    key = (64, 64, 16, 1)
    assert key in out
    assert len(out[key]["solutions"]) == 2
    assert out[key]["solutions"][0]["SolutionIndex"] == 0
    assert out[key]["solutions"][1]["SolutionIndex"] == 1
    assert out[key]["sizes"][0][1][0] == 0
    assert out[key]["sizes"][1][1][0] == 1


def test_solve_set_cover_paths() -> None:
    assert _bank.solve_set_cover({1, 2}, [(0, {1})]) is None
    cover = _bank.solve_set_cover({1, 2, 3}, [(0, {1, 2}), (1, {3}), (2, {2, 3})])
    assert set(cover) == {0, 1}


def test_scale_tol_keeps_or_increases_with_cap() -> None:
    assert _bank._scale_tol(3e9, 0.02) == pytest.approx(0.02)
    scaled = _bank._scale_tol(2e8, 0.02)
    assert scaled >= 0.02
    assert scaled <= 0.05


def test_min_assignment_uses_cached_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bench_file = tmp_path / "bench.yaml"
    bench_file.write_text("[]\n", encoding="utf-8")
    lib = _FakeLib(
        solutions=[
            {"SolutionIndex": 0, "MacroTile0": 64, "MacroTile1": 64, "DepthU": 16},
            {"SolutionIndex": 1, "MacroTile0": 64, "MacroTile1": 64, "DepthU": 16},
        ],
        sizes=[
            [[16, 16, 1, 16], [0, 0.0]],
            [[32, 32, 1, 32], [1, 0.0]],
        ],
        bench_file=bench_file,
    )

    bench_data = [
        {
            "M": 16,
            "N": 16,
            "K": 16,
            "batch_count": 1,
        },
        {
            "M": 32,
            "N": 32,
            "K": 32,
            "batch_count": 1,
        },
    ]
    monkeypatch.setattr(_bank.bench.log, "parse", lambda _p: bench_data)
    monkeypatch.setattr(_bank.bench.log, "update", lambda data, **_k: (data,))
    monkeypatch.setattr(_bank.bench.log, "dump", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _bank.bench.utils,
        "parse_benchmark_output",
        lambda _p: pd.DataFrame(_bench_rows()),
    )

    called = {"run": False}

    def _no_run(*_a, **_k):
        called["run"] = True
        return pd.DataFrame(_bench_rows())

    monkeypatch.setattr(_bank.bench, "run", _no_run)

    new_sols, new_sizes = _bank.min_assigment(
        hipblaslt_path=tmp_path,
        lib=lib,
        cluster_dir=tmp_path,
        custom_lib_dir=tmp_path,
        devices=[0],
        tol=0.2,
        scale_tol=False,
    )
    assert called["run"] is False
    assert len(new_sols) == 1
    assert len(new_sizes) == 2
    assert all(s[1][0] == 0 for s in new_sizes)


def test_min_assignment_falls_back_to_bench_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bench_file = tmp_path / "bench.yaml"
    bench_file.write_text("[]\n", encoding="utf-8")
    lib = _FakeLib(
        solutions=[
            {"SolutionIndex": 0, "MacroTile0": 64, "MacroTile1": 64, "DepthU": 16},
            {"SolutionIndex": 1, "MacroTile0": 64, "MacroTile1": 64, "DepthU": 16},
        ],
        sizes=[[[16, 16, 1, 16], [0, 0.0]]],
        bench_file=bench_file,
    )
    monkeypatch.setattr(_bank.bench.log, "parse", lambda _p: [{"M": 16, "N": 16, "K": 16, "batch_count": 1}])
    monkeypatch.setattr(_bank.bench.log, "update", lambda data, **_k: (data,))
    monkeypatch.setattr(_bank.bench.log, "dump", lambda *_a, **_k: None)
    monkeypatch.setattr(
        _bank.bench.utils,
        "parse_benchmark_output",
        lambda _p: (_ for _ in ()).throw(FileNotFoundError()),
    )

    called = {"run": 0}

    def _run(*_a, **_k):
        called["run"] += 1
        return pd.DataFrame(
            [
                {
                    **{k: v for k, v in _bench_rows()[0].items() if k in GEMM_FIELDS},
                    "m": 16,
                    "n": 16,
                    "k": 16,
                    "batch_count": 1,
                    "solutionIdx": 0,
                    "hipblaslt-Gflops": 100.0,
                }
            ]
        )

    monkeypatch.setattr(_bank.bench, "run", _run)
    new_sols, new_sizes = _bank.min_assigment(
        hipblaslt_path=tmp_path,
        lib=lib,
        cluster_dir=tmp_path,
        custom_lib_dir=tmp_path,
        devices=[0],
    )
    assert called["run"] == 1
    assert len(new_sols) == 1
    assert len(new_sizes) == 1


def test_min_assignment_infeasible_returns_original(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bench_file = tmp_path / "bench.yaml"
    bench_file.write_text("[]\n", encoding="utf-8")
    sols = [{"SolutionIndex": 0, "MacroTile0": 64, "MacroTile1": 64, "DepthU": 16}]
    sizes = [[[16, 16, 1, 16], [0, 0.0]]]
    lib = _FakeLib(solutions=sols, sizes=sizes, bench_file=bench_file)

    monkeypatch.setattr(_bank.bench.log, "parse", lambda _p: [{"M": 16, "N": 16, "K": 16, "batch_count": 1}])
    monkeypatch.setattr(_bank.bench.log, "update", lambda data, **_k: (data,))
    monkeypatch.setattr(_bank.bench.log, "dump", lambda *_a, **_k: None)
    monkeypatch.setattr(_bank.bench.utils, "parse_benchmark_output", lambda _p: pd.DataFrame(_bench_rows()[:1]))
    monkeypatch.setattr(_bank, "solve_set_cover", lambda *_a, **_k: None)

    out_sols, out_sizes = _bank.min_assigment(
        hipblaslt_path=tmp_path,
        lib=lib,
        cluster_dir=tmp_path,
        custom_lib_dir=tmp_path,
        devices=[0],
    )
    assert out_sols == sols
    assert out_sizes == sizes
    assert out_sols is not sols
    assert out_sizes is not sizes
