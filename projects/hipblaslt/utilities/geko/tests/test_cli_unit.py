# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from geko import cli


def test_parse_cli_args_defaults_with_bench_and_workload(tmp_path: Path) -> None:
    workload = tmp_path / "wkld.yaml"
    workload.write_text("[]\n")

    args = cli.parse_cli_args(
        [
            "--bench",
            "--workload-log",
            str(workload),
            "--devices",
            "0,1",
        ]
    )

    assert args.bench is True
    assert args.search is False
    assert args.tune is False
    assert args.workload == str(workload)
    assert args.keep_thr == 0.0
    assert sorted(args.devices) == [0, 1]


def test_parse_cli_args_search_default_keep_thr(tmp_path: Path) -> None:
    workload = tmp_path / "wkld.yaml"
    workload.write_text("[]\n")

    args = cli.parse_cli_args(
        [
            "--search",
            "--workload-log",
            str(workload),
            "--devices",
            "0",
        ]
    )

    assert args.search is True
    assert args.keep_thr == 0.1


def test_parse_cli_args_requires_arch_for_tune(tmp_path: Path) -> None:
    workload = tmp_path / "wkld.yaml"
    workload.write_text("[]\n")

    with pytest.raises(SystemExit):
        cli.parse_cli_args(
            [
                "--tune",
                "--workload-log",
                str(workload),
                "--devices",
                "0",
            ]
        )


def test_parse_cli_args_inline_invalid_transpose(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.parse_cli_args(
            [
                "--bench",
                "--inline",
                "64",
                "64",
                "1",
                "64",
                "B",
                "B",
                "S",
                "X",
                "N",
                "--devices",
                "0",
            ]
        )


def test_parse_cli_args_rejects_missing_workload_file() -> None:
    with pytest.raises(SystemExit):
        cli.parse_cli_args([
            "--bench",
            "--workload-log",
            "missing.yaml",
            "--devices",
            "0",
        ])


def test_parse_cli_args_rejects_missing_list_file() -> None:
    with pytest.raises(SystemExit):
        cli.parse_cli_args([
            "--search",
            "--list",
            "missing.yaml",
            "--devices",
            "0",
        ])


def test_parse_cli_args_inline_requires_integer_dimensions() -> None:
    with pytest.raises(SystemExit):
        cli.parse_cli_args([
            "--bench",
            "--inline",
            "x",
            "64",
            "1",
            "64",
            "B",
            "B",
            "S",
            "N",
            "T",
            "--devices",
            "0",
        ])


def test_parse_cli_args_inline_normalizes_transpose_and_retry_flag() -> None:
    args = cli.parse_cli_args([
        "--bench",
        "--inline",
        "64",
        "64",
        "1",
        "64",
        " B ",
        "B",
        " S ",
        " t ",
        " n ",
        "--devices",
        "0",
        "--no_retry",
    ])
    assert args.inline == (64, 64, 1, 64, "B", "B", "S", "T", "N")
    assert args.retry is False


def test_parse_cli_args_surfaces_device_parse_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workload = tmp_path / "wkld.yaml"
    workload.write_text("[]\n")

    monkeypatch.setattr(cli, "parse_devices", lambda _s: (_ for _ in ()).throw(ValueError("bad devices")))
    with pytest.raises(SystemExit):
        cli.parse_cli_args([
            "--bench",
            "--workload-log",
            str(workload),
            "--devices",
            "bad",
        ])


def test_dispatch_bench_workflow_calls_run_bench(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workload = tmp_path / "wkld.yaml"
    workload.write_text("[]\n")

    called = {}

    def _fake_resolve(*_args, **_kwargs):
        return tmp_path

    def _fake_run_bench(hip_path, log_path, run_root, **kwargs):
        called["hip"] = hip_path
        called["log"] = log_path
        called["run_root"] = run_root
        called["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(cli, "resolve_hipblaslt_path", _fake_resolve)
    monkeypatch.setattr(cli, "run_bench", _fake_run_bench)

    args = cli.CliArgs(
        tune=False,
        bench=True,
        search=False,
        workload=str(workload),
        gemm_config=None,
        inline=None,
        arch=None,
        hipblaslt=str(tmp_path),
        verbose=0,
        devices=[0],
        n_slots=1,
        keep_thr=0.0,
        backend="ductile",
        workdir=str(tmp_path / "run"),
        up_thr=1.03,
        duration=0.04,
        benchmark_duration=0.5,
        retry=True,
        bench_freq=False,
    )

    rc = cli.dispatch(args, anchor=str(tmp_path))
    assert rc == 0
    assert called["log"] == str(workload)
    assert called["kwargs"]["devices"] == [0]


def test_dispatch_search_uses_generated_workload_from_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("dummy: true\n")

    monkeypatch.setattr(cli, "resolve_hipblaslt_path", lambda **_kwargs: tmp_path)
    monkeypatch.setattr(
        cli,
        "_rows_from_gemm_config_yaml",
        lambda _path, _arch: [{"M": 16, "N": 16, "K": 16}],
    )

    captured = {}

    def _fake_run_search(hip, workload, **kwargs):
        captured["hip"] = hip
        captured["workload"] = workload
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_search", _fake_run_search)

    args = cli.CliArgs(
        tune=False,
        bench=False,
        search=True,
        workload=None,
        gemm_config=str(cfg),
        inline=None,
        arch="gfx950",
        hipblaslt=str(tmp_path),
        verbose=0,
        devices=[0],
        n_slots=1,
        keep_thr=0.1,
        backend="ductile",
        workdir=str(tmp_path / "run"),
        up_thr=1.03,
        duration=0.04,
        benchmark_duration=0.5,
        retry=True,
        bench_freq=False,
    )

    rc = cli.dispatch(args, anchor=str(tmp_path))
    assert rc == 0
    generated = Path(captured["workload"])
    assert generated.is_file()
    with generated.open() as f:
        data = yaml.safe_load(f)
    assert data[0]["M"] == 16


def test_dispatch_tune_runs_configure_and_optimize(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workload = tmp_path / "wkld.yaml"
    workload.write_text("[]\n")

    order: list[str] = []

    monkeypatch.setattr(cli, "resolve_hipblaslt_path", lambda **_kwargs: tmp_path)

    def _fake_cfg(*_args, **_kwargs):
        order.append("configure")

    def _fake_opt(*_args, **_kwargs):
        order.append("optimize")

    monkeypatch.setattr(cli, "run_configure", _fake_cfg)
    monkeypatch.setattr(cli, "run_optimize", _fake_opt)

    args = cli.CliArgs(
        tune=True,
        bench=False,
        search=False,
        workload=str(workload),
        gemm_config=None,
        inline=None,
        arch="gfx950",
        hipblaslt=str(tmp_path),
        verbose=0,
        devices=[0],
        n_slots=1,
        keep_thr=0.0,
        backend="ductile",
        workdir=str(tmp_path / "run"),
        up_thr=1.03,
        duration=0.04,
        benchmark_duration=0.5,
        retry=True,
        bench_freq=False,
    )

    rc = cli.dispatch(args, anchor=str(tmp_path))
    assert rc == 0
    assert order == ["configure", "optimize"]


def test_main_maps_non_int_system_exit_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_non_int(_argv):
        raise SystemExit("bad")

    monkeypatch.setattr(cli, "parse_cli_args", _raise_non_int)
    rc = cli.main(["--help"])
    assert rc == 1


def test_dispatch_list_loader_error_returns_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("dummy: true\n")

    monkeypatch.setattr(cli, "resolve_hipblaslt_path", lambda **_kwargs: tmp_path)
    monkeypatch.setattr(cli, "_rows_from_gemm_config_yaml", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad cfg")))

    args = cli.CliArgs(
        tune=False,
        bench=False,
        search=True,
        workload=None,
        gemm_config=str(cfg),
        inline=None,
        arch="gfx950",
        hipblaslt=str(tmp_path),
        verbose=0,
        devices=[0],
        n_slots=1,
        keep_thr=0.1,
        backend="ductile",
        workdir=str(tmp_path / "run"),
        up_thr=1.03,
        duration=0.04,
        benchmark_duration=0.5,
        retry=True,
        bench_freq=False,
    )

    assert cli.dispatch(args, anchor=str(tmp_path)) == 1


def test_dispatch_inline_value_error_returns_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "resolve_hipblaslt_path", lambda **_kwargs: tmp_path)
    monkeypatch.setattr(cli.GemmType, "from_tensile", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad inline")))

    args = cli.CliArgs(
        tune=False,
        bench=True,
        search=False,
        workload=None,
        gemm_config=None,
        inline=(64, 64, 1, 64, "B", "B", "S", "N", "N"),
        arch=None,
        hipblaslt=str(tmp_path),
        verbose=0,
        devices=[0],
        n_slots=1,
        keep_thr=0.0,
        backend="ductile",
        workdir=str(tmp_path / "run"),
        up_thr=1.03,
        duration=0.04,
        benchmark_duration=0.5,
        retry=True,
        bench_freq=False,
    )

    assert cli.dispatch(args, anchor=str(tmp_path)) == 1


def test_dispatch_returns_one_for_missing_workload_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "resolve_hipblaslt_path", lambda **_kwargs: tmp_path)

    args = cli.CliArgs(
        tune=False,
        bench=True,
        search=False,
        workload=None,
        gemm_config=None,
        inline=None,
        arch=None,
        hipblaslt=str(tmp_path),
        verbose=0,
        devices=[0],
        n_slots=1,
        keep_thr=0.0,
        backend="ductile",
        workdir=str(tmp_path / "run"),
        up_thr=1.03,
        duration=0.04,
        benchmark_duration=0.5,
        retry=True,
        bench_freq=False,
    )

    assert cli.dispatch(args, anchor=str(tmp_path)) == 1


def test_dispatch_raises_when_mode_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workload = tmp_path / "wkld.yaml"
    workload.write_text("[]\n")

    monkeypatch.setattr(cli, "resolve_hipblaslt_path", lambda **_kwargs: tmp_path)

    args = cli.CliArgs(
        tune=False,
        bench=False,
        search=False,
        workload=str(workload),
        gemm_config=None,
        inline=None,
        arch=None,
        hipblaslt=str(tmp_path),
        verbose=0,
        devices=[0],
        n_slots=1,
        keep_thr=0.0,
        backend="ductile",
        workdir=str(tmp_path / "run"),
        up_thr=1.03,
        duration=0.04,
        benchmark_duration=0.5,
        retry=True,
        bench_freq=False,
    )

    with pytest.raises(NotImplementedError, match="Expected --tune"):
        cli.dispatch(args, anchor=str(tmp_path))
