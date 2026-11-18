# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only CLI delegation; synthetic timings must never enter correctness JSON."""

from __future__ import annotations

import json
from typing import Self

import pytest
from rocke.benchmark.gemm.peer_runner import PeerGemmMeasurement
from rocke.examples.common.peer_gemm import run


@pytest.fixture
def fake_runner(monkeypatch):
    calls = []

    class Runner:
        def __init__(self, spec, **kwargs):
            calls.append(("create", spec, kwargs))

        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type, exc, traceback):
            calls.append(("close",))

        def validate(self, *, fused, repeats):
            calls.append(("validate", fused, repeats))
            return {"status": "passed", "fused": fused}

        def enqueue(self, *, fused):
            calls.append(("enqueue", fused))

        def drain(self):
            calls.append(("drain",))

        def measure(self, *, fused):
            calls.append(("measure", fused))
            # Synthetic fixture fields, not software-achieved measurements.
            return PeerGemmMeasurement(fused, 0.0, (0.0, 0.0), (1, 1))

    monkeypatch.setattr(run, "PeerGemmRunner", Runner)
    return calls


def test_correctness_cli_never_requests_timing(fake_runner, tmp_path, capsys):
    output = tmp_path / "checks.json"
    assert (
        run.main(
            [
                "--devices",
                "1,0",
                "--arch",
                "gfx950",
                "--dtype",
                "f16",
                "--correctness-output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["devices"] == [1, 0]
    assert report["spec"]["dtype"] == "f16"
    assert [check["fused"] for check in report["checks"]] == [True, False]
    assert not any(call[0] in ("measure", "enqueue", "drain") for call in fake_runner)
    assert fake_runner[-1] == ("close",)
    assert "MEASUREMENTS" not in capsys.readouterr().out


def test_measurements_are_separate_and_order_is_balanced(fake_runner, tmp_path, capsys):
    output = tmp_path / "checks.json"
    assert (
        run.main(
            [
                "--devices",
                "0,1",
                "--arch",
                "gfx942",
                "--correctness-output",
                str(output),
                "--warmup",
                "2",
                "--measure-samples",
                "2",
            ]
        )
        == 0
    )
    assert [call[1] for call in fake_runner if call[0] == "enqueue"] == [
        True,
        True,
        False,
        False,
    ]
    assert [call[1] for call in fake_runner if call[0] == "measure"] == [
        True,
        False,
        False,
        True,
    ]
    text = output.read_text()
    assert "host_makespan_ms" not in text and "device_stream_ms" not in text
    stdout = capsys.readouterr().out
    assert "MEASUREMENTS" in stdout and "host_makespan_ms" in stdout


@pytest.mark.parametrize(
    "extra",
    [
        ["--devices", ""],
        ["--devices", "one"],
        ["--devices", "0,0"],
        ["--devices", "-1,0"],
        ["--repeats", "0"],
        ["--warmup", "-1"],
        ["--measure-samples", "-1"],
        ["--rows", "17"],
        ["--k", "0"],
    ],
)
def test_argument_errors_do_not_create_gpu_runner(fake_runner, extra):
    with pytest.raises(SystemExit) as error:
        run.main(["--devices", "0,1", "--arch", "gfx950", *extra])
    assert error.value.code == 2
    assert not fake_runner


def test_failed_teardown_cannot_publish_success_report(
    fake_runner, monkeypatch, tmp_path
):
    runner = run.PeerGemmRunner

    def fail(self, exc_type, exc, traceback):
        raise RuntimeError("synthetic teardown failure")

    monkeypatch.setattr(runner, "__exit__", fail)
    output = tmp_path / "checks.json"
    with pytest.raises(RuntimeError, match="teardown failure"):
        run.main(
            [
                "--devices",
                "0,1",
                "--arch",
                "gfx950",
                "--correctness-output",
                str(output),
            ]
        )
    assert not output.exists()
