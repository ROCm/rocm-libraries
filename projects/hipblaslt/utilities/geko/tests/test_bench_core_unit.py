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
        "scaleA": 1,
        "scaleB": 1,
    }
    row.update(overrides)
    return row


def _log_row(**overrides):
    """Build a bench yaml row (LOG_FIELDS convention, uppercase M/N/K)."""
    row = {
        "transA": "N",
        "transB": "N",
        "batch_count": 1,
        "M": 16,
        "N": 16,
        "K": 16,
        "a_type": "f16_r",
        "b_type": "f16_r",
        "c_type": "f16_r",
        "d_type": "f16_r",
        "compute_type": "c_f32_r",
    }
    row.update(overrides)
    return row


def _probe_row(row, us):
    """Build a raw-probe-output-style row (GEMM_FIELDS convention) from a bench yaml row."""
    out = {k.lower() if k in ("M", "N", "K") else k: v for k, v in row.items()}
    out["compute_type"] = out["compute_type"].lstrip("c_")
    out["us"] = us
    return out


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
    (custom_lib / "library/gfx950/TensileLibrary_lazy_gfx950.dat").write_text("dat\n")

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


def test_run_raises_for_unbuilt_custom_library(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    bench_file = tmp_path / "bench.yaml"
    bench_file.write_text("[]\n")
    out_file = tmp_path / "bench.out"
    custom_lib = tmp_path / "custom"
    custom_lib.mkdir()

    with pytest.raises(ValueError, match="is not built"):
        bcore.run(hip, bench_file, out_file, custom_lib_dir=custom_lib)


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
    row1, row2 = _log_row(M=16), _log_row(M=32)
    yaml.safe_dump([row1, row2], bench_file.open("w"), sort_keys=False)
    out_file = tmp_path / "out.out"

    seen = {"calls": []}

    def _fake_run(_hip, yaml_file, _out, **kwargs):
        seen["calls"].append(Path(yaml_file))
        if "probe" in str(yaml_file):
            # deliberately scrambled relative to the bench rows' order
            return pd.DataFrame([_probe_row(row2, 20.0), _probe_row(row1, 10.0)])
        return pd.DataFrame([{"ok": 1}])

    monkeypatch.setattr(bcore, "run", _fake_run)
    out = bcore.standard_benchmark(hip, bench_file, out_file, duration=0.001, adaptive=False)
    assert len(out) == 1
    assert len(seen["calls"]) == 2


def test_standard_benchmark_adaptive_single_run_without_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    bench_file = tmp_path / "bench.yaml"
    yaml.safe_dump([_log_row(M=16), _log_row(M=32)], bench_file.open("w"), sort_keys=False)
    out_file = tmp_path / "out.out"

    seen = {"calls": [], "kwargs": []}

    def _fake_run(_hip, yaml_file, _out, **kwargs):
        seen["calls"].append(Path(yaml_file))
        seen["kwargs"].append(kwargs)
        return pd.DataFrame([{"ok": 1}])

    monkeypatch.setattr(bcore, "run", _fake_run)
    out = bcore.standard_benchmark(hip, bench_file, out_file, duration=0.5, bench_freq=True)
    assert len(out) == 1
    assert len(seen["calls"]) == 1
    assert not any("probe" in str(p) for p in seen["calls"])
    assert seen["kwargs"][0]["cache"] is False
    assert seen["kwargs"][0]["bench_freq"] is True

    rows = yaml.safe_load(seen["calls"][0].read_text())
    assert len(rows) == 2
    for row in rows:
        assert row["adaptive"] is True
        assert row["measure_time"] == pytest.approx(500.0)
        assert row["warmup_time"] == pytest.approx(200.0)
        assert row["max_measure_time"] == pytest.approx(2000.0)
        assert row["sample_time"] == pytest.approx(1.0)


def test_standard_benchmark_adaptive_replaces_existing_adaptive_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    bench_file = tmp_path / "bench.yaml"
    row = _log_row()
    row.update({"adaptive": True, "measure_time": 1.0, "noise_threshold": 0.5})
    yaml.safe_dump([row], bench_file.open("w"), sort_keys=False)

    seen = {}

    def _fake_run(_hip, yaml_file, _out, **kwargs):
        seen["rows"] = yaml.safe_load(Path(yaml_file).read_text())
        return pd.DataFrame([{"ok": 1}])

    monkeypatch.setattr(bcore, "run", _fake_run)
    bcore.standard_benchmark(hip, bench_file, tmp_path / "out.out", duration=0.2)
    assert seen["rows"][0]["measure_time"] == pytest.approx(200.0)
    assert seen["rows"][0]["noise_threshold"] == pytest.approx(0.005)


def test_standard_benchmark_non_adaptive_strips_adaptive_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    bench_file = tmp_path / "bench.yaml"
    row = _log_row()
    row.update({"adaptive": True, "measure_time": 1.0, "noise_threshold": 0.5})
    yaml.safe_dump([row], bench_file.open("w"), sort_keys=False)

    seen = []

    def _fake_run(_hip, yaml_file, _out, **kwargs):
        rows = yaml.safe_load(Path(yaml_file).read_text())
        seen.append(rows)
        if "probe" in str(yaml_file):
            return pd.DataFrame([_probe_row(row, 10.0)])
        return pd.DataFrame([{"ok": 1}])

    monkeypatch.setattr(bcore, "run", _fake_run)
    bcore.standard_benchmark(hip, bench_file, tmp_path / "out.out", duration=0.001, adaptive=False)
    assert len(seen) == 2
    for rows in seen:
        for key in bcore.log.ADAPTIVE_ROW_KEYS:
            assert key not in rows[0]


def test_standard_benchmark_probe_mismatch_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    bench_file = tmp_path / "bench.yaml"
    yaml.safe_dump([_log_row(), _log_row(M=32)], bench_file.open("w"), sort_keys=False)

    def _fake_run(*_a, **_k):
        return pd.DataFrame([{"us": 10.0}])

    monkeypatch.setattr(bcore, "run", _fake_run)
    with pytest.raises(ValueError, match="Probe output rows"):
        bcore.standard_benchmark(hip, bench_file, tmp_path / "x.out", adaptive=False)


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

        def create_bench_input(self, benchmark_dir, duration, verify, beta, adaptive):
            assert adaptive is True
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
        "scaleA": 1,
        "scaleB": 1,
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


ADAPTIVE_OUTPUT_HEADER = (
    "transA,transB,batch_count,m,n,k,a_type,b_type,c_type,d_type,compute_type,"
    "hipblaslt-Gflops,hipblaslt-GB/s,us,batch,samples,hot_iters,mean_us,min_us,cv,rel_iqr,status"
)


def _adaptive_output(us: float, batch: int, samples: int, status: str) -> str:
    """hipblaslt-bench output with adaptive timing columns (their values vary per run)."""
    return (
        f"[0]:{ADAPTIVE_OUTPUT_HEADER}\n"
        f"N,N,1,16,16,16,f16_r,f16_r,f16_r,f16_r,c_f32_r,1.0,2.0,{us},{batch},{samples},"
        f"{samples * batch},{us + 0.1},{us - 0.1},0.01,0.02,{status}\n"
        "--kernel name: k0\n--Solution name: s0\n--Solution index: 7\n"
    )


def test_parse_output_only_adds_known_result_columns(tmp_path: Path) -> None:
    """Every parsed column is a benchmark input or a result listed in UNIQ_COLS.

    A new per-run result column in the hipblaslt-bench output (like the adaptive
    timing ones) that is not in UNIQ_COLS ends up as a merge key in compare().
    """
    f = tmp_path / "a.out"
    f.write_text(_adaptive_output(10.0, 4, 386, "converged"))
    df = bcore.parse_benchmark_output(f)

    inputs = set(ADAPTIVE_OUTPUT_HEADER.split(",hipblaslt-Gflops")[0].split(","))
    parser_added = {"kernel", "solution", "solutionIdx", "scaleA", "scaleB"}
    unclassified = set(df.columns) - inputs - parser_added - set(bcore.UNIQ_COLS)
    assert not unclassified


def _compare_with_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_df) -> pd.DataFrame:
    """Run compare() with reference and tuned outputs from make_df(is_tuned)."""
    hip = tmp_path / "hip"
    (hip / "build/release/device-library").mkdir(parents=True)
    lib_dir = tmp_path / "libs"
    lib_dir.mkdir()
    custom = tmp_path / "custom"
    (custom / "library").mkdir(parents=True)

    class _Lib:
        name = "libA"

        def create_bench_input(self, benchmark_dir, **_k):
            bf = Path(benchmark_dir) / "libA_bench.yaml"
            bf.write_text("[]\n")
            return bf, None

    monkeypatch.setattr(bcore.library, "load_collection", lambda _p: [_Lib()])
    monkeypatch.setattr(bcore.library.operations, "create", lambda *_a, **_k: None)
    monkeypatch.setattr(bcore, "update_lib_source", lambda df, _m: df.assign(lib_source="default"))
    monkeypatch.setattr(
        bcore, "run", lambda _h, _b, _o, custom_lib_dir=None, **_k: make_df(custom_lib_dir is not None)
    )

    return bcore.compare(
        hip, lib_dir, custom_lib_dir=custom, benchmark_dir=tmp_path / "bench", verify=False, devices=[0]
    )


def _result_row(is_tuned: bool, **extra) -> dict:
    return {
        **_bench_row(),
        "hipblaslt-Gflops": 1.0,
        "hipblaslt-GB/s": 2.0,
        "us": 5.0 if is_tuned else 10.0,
        "kernel": "k",
        "solution": "s",
        "solutionIdx": 7,
        **extra,
    }


def test_compare_ignores_adaptive_timing_columns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Adaptive timing outputs differ between runs and must not be merge keys."""

    def make_df(is_tuned: bool) -> pd.DataFrame:
        return pd.DataFrame([
            _result_row(
                is_tuned,
                batch=4 if is_tuned else 8,
                samples=386 if is_tuned else 400,
                hot_iters=1544 if is_tuned else 3200,
                mean_us=5.1 if is_tuned else 10.1,
                min_us=4.9 if is_tuned else 9.9,
                cv=0.01 if is_tuned else 0.2,
                rel_iqr=0.02 if is_tuned else 0.3,
                status="converged" if is_tuned else "noisy",
            )
        ])

    out = _compare_with_runs(monkeypatch, tmp_path, make_df)
    assert len(out) == 1
    assert out.iloc[0]["ratio"] == pytest.approx(2.0)
    for col in ("status", "cv", "samples"):
        assert f"{col}_reference" in out.columns and f"{col}_tuned" in out.columns


def test_compare_raises_when_unlisted_result_column_breaks_merge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A per-run result column missing from UNIQ_COLS must fail loudly, not give 0 rows."""

    def make_df(is_tuned: bool) -> pd.DataFrame:
        return pd.DataFrame([_result_row(is_tuned, new_metric=2.0 if is_tuned else 1.0)])

    with pytest.raises(ValueError, match=r"UNIQ_COLS.*new_metric"):
        _compare_with_runs(monkeypatch, tmp_path, make_df)
