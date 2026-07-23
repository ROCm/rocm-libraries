# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from geko.bench import log as blog


def _row(**overrides):
    row = {
        "function": "matmul",
        "M": 16,
        "N": 16,
        "K": 16,
        "transA": "N",
        "transB": "N",
        "batch_count": 1,
        "a_type": "f16_r",
        "b_type": "f16_r",
        "c_type": "f16_r",
        "d_type": "f16_r",
        "compute_type": "f32_r",
        "call_count": 1,
    }
    row.update(overrides)
    return row


def test_update_compute_type() -> None:
    assert blog.update_compute_type("f32_r") == "c_f32_r"
    assert blog.update_compute_type("c_f32_r") == "c_f32_r"
    assert blog.update_compute_type("f16_r") == "f16_r"


def test_parse_yaml_as_df_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "log.yaml"
    yaml.safe_dump([_row()], p.open("w"), sort_keys=False)

    df = blog.parse(p, as_df=True)
    assert len(df) == 1
    assert "call_count" in df.columns
    assert df.iloc[0]["compute_type"] == "c_f32_r"


def test_parse_raises_for_missing_required_fields(tmp_path: Path) -> None:
    p = tmp_path / "log.yaml"
    yaml.safe_dump([{"M": 16}], p.open("w"), sort_keys=False)
    with pytest.raises(ValueError, match="must have all fields"):
        blog.parse(p, as_df=True)


def test_parse_raises_for_missing_required_values(tmp_path: Path) -> None:
    p = tmp_path / "log.yaml"
    row = _row()
    row["M"] = None
    yaml.safe_dump([row], p.open("w"), sort_keys=False)
    with pytest.raises(ValueError, match="missing values"):
        blog.parse(p, as_df=True)


def test_parse_drops_columns_with_missing_values(tmp_path: Path) -> None:
    p = tmp_path / "log.yaml"
    row = _row(extra_field=None)
    yaml.safe_dump([row], p.open("w"), sort_keys=False)
    df = blog.parse(p, as_df=True)
    assert "extra_field" not in df.columns


def test_parse_aggregates_duplicate_rows(tmp_path: Path) -> None:
    p = tmp_path / "log.yaml"
    yaml.safe_dump([_row(call_count=2), _row(call_count=3)], p.open("w"), sort_keys=False)
    df = blog.parse(p, as_df=True)
    assert len(df) == 1
    assert int(df.iloc[0]["call_count"]) == 5


def test_read_handles_parser_error_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("a: ,\n")

    calls = {"n": 0}
    orig_safe_load = yaml.safe_load

    def _safe_load(x):
        calls["n"] += 1
        if calls["n"] == 1:
            raise yaml.parser.ParserError("ctx", None, "problem", None)
        return orig_safe_load(x)

    monkeypatch.setattr(yaml, "safe_load", _safe_load)
    out = blog.read(p)
    assert out == "a"


def test_update_rejects_unknown_format() -> None:
    with pytest.raises(TypeError, match="Unknown format"):
        blog.update(data=123)


def test_update_rejects_bad_latency_type() -> None:
    with pytest.raises(TypeError, match="latency"):
        blog.update([_row()], latency=1.0)


def test_update_rejects_latency_length_mismatch() -> None:
    with pytest.raises(ValueError, match="lengths must match"):
        blog.update([_row(), _row(M=32)], latency=[1.0])


def test_update_sets_iters_from_latency() -> None:
    out = blog.update([_row()], latency=[50.0], duration=0.01)[0]
    assert out[0]["iters"] == 200
    assert out[0]["cold_iters"] == 200
    assert out[0]["compute_type"] == "c_f32_r"


def test_update_from_path_returns_output_file_and_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "in.yaml"
    p.write_text("[]\n")

    monkeypatch.setattr(blog, "parse", lambda _p: [_row()])
    out = blog.update(str(p))
    assert len(out) == 2
    data, out_file = out
    assert data[0]["iters"] == 100
    assert Path(out_file).is_file()


def test_verify_output_true_and_false_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bench_file = tmp_path / "bench.yaml"
    yaml.safe_dump([_row()], bench_file.open("w"), sort_keys=False)

    match_df = pd.DataFrame(
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

    monkeypatch.setattr(blog.bench.utils, "parse_benchmark_output", lambda _p: match_df)
    assert blog.verify_output(tmp_path / "lat.out", bench_file) is True

    mismatch_df = match_df.copy()
    mismatch_df.loc[0, "m"] = 32
    monkeypatch.setattr(blog.bench.utils, "parse_benchmark_output", lambda _p: mismatch_df)
    assert blog.verify_output(tmp_path / "lat.out", bench_file) is False


def test_benchmark_forwards_bench_freq_and_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log_file = tmp_path / "log.yaml"
    log_file.write_text("[]\n")

    monkeypatch.setattr(blog, "parse", lambda _p: [_row()])

    called = {}

    def _fake_run(_hip, _bench, _out, **kwargs):
        called["kwargs"] = kwargs
        return pd.DataFrame([{"ok": 1}])

    monkeypatch.setattr(blog.bench, "run", _fake_run)
    out = blog.benchmark(tmp_path / "hip", log_file, tmp_path / "bench", device=3, bench_freq=True)
    assert len(out) == 1
    assert called["kwargs"]["devices"] == [3]
    assert called["kwargs"]["bench_freq"] is True


def test_summarize_keep_thr_zero_skips_benchmark(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workload = tmp_path / "w.yaml"
    workload.write_text("[]\n")

    rows = [_row(), _row(M=32)]
    monkeypatch.setattr(blog, "parse", lambda *_a, **_k: rows)
    monkeypatch.setattr(blog, "update", lambda data, **_k: (data,))

    called = {"run": False, "std": False}

    def _never_run(*_a, **_k):
        called["run"] = True
        return pd.DataFrame()

    def _never_std(*_a, **_k):
        called["std"] = True
        return pd.DataFrame()

    monkeypatch.setattr(blog.bench, "run", _never_run)
    monkeypatch.setattr(blog.bench, "standard_benchmark", _never_std)

    summary, uniq = blog.summarize(hip, workload, output_dir=tmp_path / "o", keep_thr=0.0, devices=[0])
    assert len(summary) == 2
    assert len(uniq) == 2
    assert called["run"] is False
    assert called["std"] is False


def test_summarize_keep_thr_positive_uses_standard_benchmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hip = tmp_path / "hip"
    (hip / "build/release/device-library").mkdir(parents=True)
    workload = tmp_path / "w.yaml"
    workload.write_text("[]\n")

    rows = [_row()]
    monkeypatch.setattr(blog, "parse", lambda *_a, **_k: rows)
    monkeypatch.setattr(blog, "update", lambda data, **_k: (data,))
    monkeypatch.setattr(blog, "update_lib_source", lambda df, _mt: df.assign(lib_source="x"))

    def _std(*_a, **_k):
        return pd.DataFrame(
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
                    "us": 1.0,
                }
            ]
        )

    monkeypatch.setattr(blog.bench, "standard_benchmark", _std)

    summary, uniq = blog.summarize(
        hip,
        workload,
        output_dir=tmp_path / "o",
        keep_thr=0.1,
        use_standard_benchmark=True,
        devices=[0],
    )
    assert not summary.empty
    assert len(uniq) == 1


def test_summarize_keep_thr_positive_uses_bench_run_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    (hip / "build/release/device-library").mkdir(parents=True)
    workload = tmp_path / "w.yaml"
    workload.write_text("[]\n")

    rows = [_row(call_count=2)]
    monkeypatch.setattr(blog, "parse", lambda *_a, **_k: rows)
    monkeypatch.setattr(blog, "update", lambda data, **_k: (data,))

    called = {}

    def _run(*_a, **kwargs):
        called["bench_freq"] = kwargs.get("bench_freq")
        return pd.DataFrame(
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
                    "us": 2.0,
                }
            ]
        )

    monkeypatch.setattr(blog.bench, "run", _run)
    monkeypatch.setattr(blog, "update_lib_source", lambda df, _mt: df)

    summary, uniq = blog.summarize(
        hip,
        workload,
        output_dir=tmp_path / "o",
        keep_thr=0.01,
        use_standard_benchmark=False,
        devices=[0],
        bench_freq=True,
    )
    assert not summary.empty
    assert len(uniq) == 1
    assert called["bench_freq"] is True


def test_summarize_device_alias_overrides_devices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    workload = tmp_path / "w.yaml"
    workload.write_text("[]\n")

    rows = [_row()]
    monkeypatch.setattr(blog, "parse", lambda *_a, **_k: rows)
    monkeypatch.setattr(blog, "update", lambda data, **_k: (data,))

    devices_seen = {}

    def _parse_devices(dev):
        devices_seen["value"] = list(dev)
        return list(dev)

    monkeypatch.setattr(blog, "parse_devices", _parse_devices)
    blog.summarize(hip, workload, output_dir=tmp_path / "o", keep_thr=0.0, devices=[0, 1], device=2)
    assert devices_seen["value"] == [2]
