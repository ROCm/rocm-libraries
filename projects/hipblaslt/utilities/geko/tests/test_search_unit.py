# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from geko import search


def test_normalize_compute_type_prefixing() -> None:
    assert search._normalize_compute_type("f32_r") == "c_f32_r"
    assert search._normalize_compute_type("c_f32_r") == "c_f32_r"


def test_load_processed_df_missing_file_returns_empty(tmp_path: Path) -> None:
    out = search._load_processed_df(tmp_path / "missing.csv")
    assert out.empty


def test_load_processed_df_invalid_schema_deletes_file(tmp_path: Path) -> None:
    processed = tmp_path / "processed.csv"
    pd.DataFrame({"bad": [1]}).to_csv(processed, index=False)

    out = search._load_processed_df(processed)
    assert out.empty
    assert not processed.exists()


def test_configure_rejects_empty_df() -> None:
    with pytest.raises(ValueError, match="No GEMM operations"):
        search.configure(pd.DataFrame())


def test_configure_sets_skip_ratio_from_latency() -> None:
    df = pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 32,
                "n": 32,
                "k": 32,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "f32_r",
                "us": 10.0,
            },
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 4096,
                "n": 4096,
                "k": 2048,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "f32_r",
                "us": 30.0,
            },
        ]
    )

    out = search.configure(df)
    assert out[0]["skip_slow_solution_ratio"] == 0.3
    assert out[1]["skip_slow_solution_ratio"] == 0.6


def test_run_raises_for_missing_hipblaslt_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        search.run(tmp_path / "nohip", [{"M": 1}], tmp_path / "out", devices=[0])


def test_run_rejects_non_positive_chunk_size(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    with pytest.raises(ValueError, match="max_chunk_size"):
        search.run(
            hip,
            [
                {
                    "transA": "N",
                    "transB": "N",
                    "batch_count": 1,
                    "M": 32,
                    "N": 32,
                    "K": 32,
                    "a_type": "f16_r",
                    "b_type": "f16_r",
                    "c_type": "f16_r",
                    "d_type": "f16_r",
                    "compute_type": "f32_r",
                }
            ],
            tmp_path / "out",
            devices=[0],
            max_chunk_size=0,
        )


def test_run_empty_configs_returns_empty_df(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    out = search.run(hip, [], tmp_path / "out", devices=[0])
    assert out.empty


def test_run_uses_existing_data_when_all_configs_processed(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    outdir = tmp_path / "out"
    data_dir = outdir / "data"
    data_dir.mkdir(parents=True)

    cfg = {
        "transA": "N",
        "transB": "N",
        "batch_count": 1,
        "M": 32,
        "N": 32,
        "K": 32,
        "a_type": "f16_r",
        "b_type": "f16_r",
        "c_type": "f16_r",
        "d_type": "f16_r",
        "compute_type": "f32_r",
    }

    processed = pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 32,
                "n": 32,
                "k": 32,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "c_f32_r",
            }
        ]
    )
    processed.to_csv(outdir / "processed.csv", index=False)

    pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 32,
                "n": 32,
                "k": 32,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "c_f32_r",
                "us": 1.0,
            }
        ]
    ).to_csv(data_dir / "existing.csv", index=False)

    out = search.run(hip, [cfg], outdir, devices=[0])
    assert len(out) == 1
    assert out.iloc[0]["m"] == 32


def test_run_persists_winners_with_stubbed_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    outdir = tmp_path / "out"

    cfg = {
        "transA": "N",
        "transB": "N",
        "batch_count": 1,
        "M": 64,
        "N": 64,
        "K": 64,
        "a_type": "f16_r",
        "b_type": "f16_r",
        "c_type": "f16_r",
        "d_type": "f16_r",
        "compute_type": "f32_r",
    }

    monkeypatch.setattr(search.bench.log, "dump", lambda data, p: yaml.safe_dump(data, Path(p).open("w"), sort_keys=False))

    def _fake_bench_run(_hip, _bench_file, log_file, **_kwargs):
        Path(log_file).write_text("Winner\n")
        return pd.DataFrame()

    monkeypatch.setattr(search.bench, "run", _fake_bench_run)

    parse_df = pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 64,
                "n": 64,
                "k": 64,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "c_f32_r",
                "us": 2.0,
            }
        ]
    )
    monkeypatch.setattr(search.bench.utils, "parse_benchmark_output", lambda _p: parse_df)

    class _PassRunner:
        def __init__(self, items, worker_impl, devices, n_slots=1):
            self._items = items
            self._worker_impl = worker_impl
            self._devices = devices

        def __call__(self, _workdir):
            for item in self._items:
                w = self._worker_impl(item, self._devices[0], 0, None, None)
                w.setup()
                w.run()
                w.teardown()
            return self._items

    monkeypatch.setattr(search, "Runner", _PassRunner)

    out = search.run(hip, [cfg], outdir, devices=[0])
    assert len(out) == 1
    assert (outdir / "processed.csv").is_file()


def test_run_emits_failed_gemms_after_retry_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    outdir = tmp_path / "out"

    cfg = {
        "transA": "N",
        "transB": "N",
        "batch_count": 1,
        "M": 64,
        "N": 64,
        "K": 64,
        "a_type": "f16_r",
        "b_type": "f16_r",
        "c_type": "f16_r",
        "d_type": "f16_r",
        "compute_type": "f32_r",
    }

    monkeypatch.setattr(search.bench.log, "dump", lambda data, p: yaml.safe_dump(data, Path(p).open("w"), sort_keys=False))

    def _fake_bench_run(_hip, _bench_file, _log_file, **_kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(search.bench, "run", _fake_bench_run)

    class _FailRunner:
        def __init__(self, items, worker_impl, devices, n_slots=1):
            self._items = items
            self._worker_impl = worker_impl
            self._devices = devices

        def __call__(self, _workdir):
            for item in self._items:
                w = self._worker_impl(item, self._devices[0], 0, None, None)
                w.setup()
                w.run()
                w.teardown()
            return self._items

    monkeypatch.setattr(search, "Runner", _FailRunner)

    out = search.run(hip, [cfg], outdir, devices=[0])
    assert out.empty
    assert (outdir / "failed_gemms.yaml").is_file()
