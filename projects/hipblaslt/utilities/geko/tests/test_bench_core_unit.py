# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from geko.bench import bench as bcore


def _bench_row(**overrides):
    row = {
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
    row.update(overrides)
    return row


def test_run_raises_for_missing_paths(tmp_path: Path) -> None:
    bench_file = tmp_path / "bench.yaml"
    bench_file.write_text("[]\n")
    with pytest.raises(FileNotFoundError, match="hipBLASLt path not found"):
        bcore.run(tmp_path / "missing", bench_file, tmp_path / "out")

    hip = tmp_path / "hip"
    hip.mkdir()
    with pytest.raises(FileNotFoundError, match="Benchmark file does not exist"):
        bcore.run(hip, tmp_path / "missing.yaml", tmp_path / "out")


def test_run_cache_hit_returns_parsed_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    bench_file = tmp_path / "bench.yaml"
    bench_file.write_text("[]\n")
    out_file = tmp_path / "bench.out"

    called = {"verify": 0, "parse": 0}

    monkeypatch.setattr(bcore, "verify_output", lambda *_a, **_k: True)

    def _parse(_path):
        called["parse"] += 1
        return pd.DataFrame([{"ok": 1}])

    monkeypatch.setattr(bcore, "parse_benchmark_output", _parse)

    out = bcore.run(hip, bench_file, out_file, cache=True)
    assert len(out) == 1
    assert called["parse"] == 1


def test_run_single_device_sets_custom_lib_env_and_calls_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hip = tmp_path / "hip"
    (hip / "build/release/clients").mkdir(parents=True)
    bench_file = tmp_path / "bench.yaml"
    bench_file.write_text("[]\n")
    out_file = tmp_path / "bench.out"

    custom_lib = tmp_path / "custom"
    (custom_lib / "library/gfx950").mkdir(parents=True)

    monkeypatch.setattr(bcore, "verify_output", lambda *_a, **_k: False)
    monkeypatch.setattr(bcore, "parse_benchmark_output", lambda _p: pd.DataFrame([{"ok": 1}]))

    seen = {}

    def _run(cmd, stdout, stderr, env, check, text):
        seen["cmd"] = cmd
        seen["env"] = env
        stdout.write("ok\n")
        return 0

    monkeypatch.setattr(bcore.subprocess, "run", _run)

    out = bcore.run(
        hip,
        bench_file,
        out_file,
        custom_lib_dir=custom_lib,
        devices=[2],
        cache=False,
        bench_freq=True,
    )
    assert len(out) == 1
    assert seen["cmd"][1] == "--yaml"
    assert seen["cmd"][3] == "--device"
    assert seen["cmd"][4] == "2"
    assert seen["env"]["HIPBLASLT_BENCH_FREQ"] == "true"
    assert Path(seen["env"]["HIPBLASLT_TENSILE_LIBPATH"]).parts[-2:] == ("library", "gfx950")


def test_run_multi_device_chunk_path_aggregates_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    bench_file = tmp_path / "bench.yaml"
    data = [_bench_row(m=16), _bench_row(m=32), _bench_row(m=64), _bench_row(m=128)]
    yaml.safe_dump(data, bench_file.open("w"), sort_keys=False)
    out_file = tmp_path / "bench.out"

    def _fake_read(_p):
        return data

    def _fake_dump(rows, p):
        yaml.safe_dump(rows, p.open("w"), sort_keys=False)

    class _FakeRunner:
        def __init__(self, items, worker_impl, devices, n_slots, estimate_workload_fn, job_logger_fn):
            self.items = items
            self.worker_impl = worker_impl
            self.devices = devices

        def __call__(self, workdir, silent=False):
            for i, item in enumerate(self.items):
                w = self.worker_impl(item, self.devices[i % len(self.devices)], 0, None, None)
                w.setup()
                assert w.run() is True
                w.teardown()
            return self.items

    monkeypatch.setattr(bcore, "read_bench_yaml", _fake_read)
    monkeypatch.setattr(bcore, "dump_bench_yaml", _fake_dump)
    monkeypatch.setattr(bcore, "Runner", _FakeRunner)
    monkeypatch.setattr(bcore, "verify_output", lambda *_a, **_k: True)

    def _subprocess_run(cmd, stdout, stderr, env, check, text):
        stdout.write("bench output\n")
        return 0

    monkeypatch.setattr(bcore.subprocess, "run", _subprocess_run)

    monkeypatch.setattr(
        bcore,
        "parse_benchmark_output",
        lambda p: pd.DataFrame([{"rows": len(Path(p).read_text().strip().splitlines())}]),
    )

    out = bcore.run(hip, bench_file, out_file, devices=[0, 1], min_chunk_size=1)
    assert len(out) == 1
    combined = out_file.read_text()
    assert "bench output" in combined


def test_standard_benchmark_validation_and_scaling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    bench_file = tmp_path / "bench.yaml"
    yaml.safe_dump([_bench_row(), _bench_row(m=32)], bench_file.open("w"), sort_keys=False)
    out_file = tmp_path / "out.out"

    seen = {"calls": []}

    def _fake_run(_hip, yaml_file, _out, **kwargs):
        seen["calls"].append(Path(yaml_file))
        if "probe" in str(yaml_file):
            return pd.DataFrame([{"us": 10.0}, {"us": 20.0}])
        return pd.DataFrame([{"ok": 1}])

    monkeypatch.setattr(bcore, "run", _fake_run)
    out = bcore.standard_benchmark(hip, bench_file, out_file, duration=0.001)
    assert len(out) == 1
    assert len(seen["calls"]) == 2


def test_standard_benchmark_probe_mismatch_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    bench_file = tmp_path / "bench.yaml"
    yaml.safe_dump([_bench_row(), _bench_row(m=32)], bench_file.open("w"), sort_keys=False)

    def _fake_run(*_a, **_k):
        return pd.DataFrame([{"us": 10.0}])

    monkeypatch.setattr(bcore, "run", _fake_run)
    with pytest.raises(ValueError, match="Probe output rows"):
        bcore.standard_benchmark(hip, bench_file, tmp_path / "x.out")


def test_compare_raises_when_no_libraries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bcore.library, "load_collection", lambda _p: [])
    with pytest.raises(ValueError, match="No valid libraries"):
        bcore.compare(tmp_path / "hip", tmp_path / "libs")


def test_compare_happy_path_with_verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    (hip / "build/release/device-library").mkdir(parents=True)
    lib_dir = tmp_path / "libs"
    lib_dir.mkdir()
    custom = tmp_path / "custom"
    (custom / "library").mkdir(parents=True)

    class _Lib:
        name = "libA"

        def create_bench_input(self, benchmark_dir, duration, verify, beta):
            bf = Path(benchmark_dir) / "libA_bench.yaml"
            vf = Path(benchmark_dir) / "libA_verify.yaml"
            bf.write_text("[]\n")
            vf.write_text("[]\n")
            return bf, vf

    monkeypatch.setattr(bcore.library, "load_collection", lambda _p: [_Lib()])

    created = {"n": 0}

    def _create(*_a, **_k):
        created["n"] += 1

    monkeypatch.setattr(bcore.library.operations, "create", _create)

    def _update(df, _match):
        return df.assign(lib_source="default")

    monkeypatch.setattr(bcore, "update_lib_source", _update)

    common = {
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
        "kernel": "k",
        "solution": "s",
        "solutionIdx": 7,
        "hipblaslt-Gflops": 1.0,
        "hipblaslt-GB/s": 2.0,
        "us": 10.0,
        "error": 0.0,
        "lib": "x",
        "lib_source": "x",
        "lowest_avg_freq": 0.0,
        "lowest_median_freq": 0.0,
        "avg_MCLK": 0.0,
        "median_MCLK": 0.0,
    }

    def _fake_run(_hip, bench_in, _out, custom_lib_dir=None, **kwargs):
        p = str(bench_in)
        if p.endswith("verify.yaml"):
            return pd.DataFrame([
                {
                    **{k: v for k, v in common.items() if k not in {"kernel", "solution", "solutionIdx", "lib", "lib_source", "error", "hipblaslt-Gflops", "hipblaslt-GB/s", "us", "lowest_avg_freq", "lowest_median_freq", "avg_MCLK", "median_MCLK"}},
                    "norm_error": 1.0,
                    "atol": "failed",
                    "rtol": "ok",
                }
            ])
        base = dict(common)
        if custom_lib_dir is not None:
            base["us"] = 5.0
        return pd.DataFrame([base])

    monkeypatch.setattr(bcore, "run", _fake_run)

    out = bcore.compare(
        hip,
        lib_dir,
        custom_lib_dir=custom,
        benchmark_dir=tmp_path / "bench",
        verify=True,
        cache=False,
        devices=[0],
    )
    assert len(out) == 1
    assert created["n"] == 1
    assert out.iloc[0]["ratio"] == pytest.approx(2.0)
    assert out.iloc[0]["error_tuned"] == pytest.approx(1e6)
