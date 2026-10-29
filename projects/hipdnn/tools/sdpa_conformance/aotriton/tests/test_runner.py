# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""run_aotriton.py: command composition, collected.txt and result-directory rules (no pytest run)."""

import subprocess

import pytest

import aotriton_layout as layout
import run_aotriton


class FakePytest:
    """Stands in for subprocess.run: answers collection with canned ids, records every command."""

    def __init__(self, ids_per_pass):
        self.ids_per_pass = list(ids_per_pass)
        self.commands = []

    def __call__(self, cmd, env=None, capture_output=False, text=False):
        self.commands.append((cmd, env))
        if "--collect-only" in cmd:
            ids = self.ids_per_pass.pop(0)
            out = (
                "".join(f"{i}\n" for i in ids)
                + f"\n{len(ids)} tests collected in 0.1s\n"
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        return subprocess.CompletedProcess(cmd, 0)


@pytest.fixture(autouse=True)
def rocm_path(monkeypatch):
    """The setup_env.py activation sets ROCM_PATH; the runner's --hipdnn-prefix defaults to it."""
    monkeypatch.setenv("ROCM_PATH", "/opt/fake-rocm")


@pytest.fixture
def runner(monkeypatch):
    def go(ids_per_pass, *argv):
        fake = FakePytest(ids_per_pass)
        monkeypatch.setattr(run_aotriton.subprocess, "run", fake)
        status = run_aotriton.main(list(argv))
        return status, fake.commands

    return go


def test_forward_suite_names_test_forward_explicitly_and_exports_the_release(
    runner, tmp_path
):
    status, commands = runner(
        [["f::a", "f::b", "f::c"]],
        "--workers",
        "4",
        "--out",
        str(tmp_path),
        "--suite",
        "forward",
        "--release",
        "1",
        "--shard",
        "2/3",
        "--fake",
        "always-decline",
    )
    collect, run = commands
    assert status == 0
    assert str(layout.FLASH_TESTS / "test_forward.py") in collect[0]
    assert "--results-dir" not in " ".join(collect[0])
    for cmd, env in commands:
        assert {"-p", "hipdnn_aotriton_plugin"} <= set(cmd) and "--shard=2/3" in cmd
        assert "--fake=always-decline" in cmd and env["FOR_RELEASE"] == "1"
        assert env["PYTHONPATH"].split(":")[0] == str(layout.SHIM)
    assert f"--results-dir={tmp_path}" in run[0]
    assert run[0][run[0].index("-n") + 1] == "4" and "--max-worker-restart=3" in run[0]
    assert (tmp_path / "collected.txt").read_text() == "f::a\nf::b\nf::c\n"


def test_packed_gpu_adds_a_second_gpu_marked_pass_into_the_same_collected_list(
    runner, tmp_path
):
    status, commands = runner(
        [["v::a"], ["p::x", "p::y"]],
        "--workers",
        "1",
        "--out",
        str(tmp_path),
        "--packed-gpu",
    )
    assert len(commands) == 4  # collect + run, twice
    second = commands[2][0]
    assert (
        second[len(second) - 1 - second[::-1].index("-m") + 1] == "gpu"
        and str(layout.PACKED_TESTS) in second
    )
    assert "--max-worker-restart=2" in commands[3][0]
    assert (tmp_path / "collected.txt").read_text() == "v::a\np::x\np::y\n"


def test_default_out_is_arch_run_shard_directory(monkeypatch, runner):
    monkeypatch.setattr(run_aotriton, "DEFAULT_RESULTS", run_aotriton.Path("/nowhere"))
    args = run_aotriton.parse_args(
        ["--arch", "gfx950", "--run", "r1", "--shard", "2/4", "--workers", "1"]
    )
    assert (
        run_aotriton.resolve_out(args, None).as_posix()
        == "/nowhere/gfx950/r1/shard-2of4"
    )
    args = run_aotriton.parse_args(["--workers", "1"])
    with pytest.raises(SystemExit):
        run_aotriton.resolve_out(args, None)


def test_existing_results_are_never_appended_to(runner, tmp_path):
    (tmp_path / "results.csv").write_text(
        "nodeid,outcome,mode,fwd,bwd,reason\nx,pass,,,,\n"
    )
    with pytest.raises(SystemExit, match="already has rows"):
        runner([["a"]], "--workers", "1", "--out", str(tmp_path))


def test_empty_collection_runs_nothing_and_collection_errors_stop_the_run(
    runner, tmp_path, monkeypatch
):
    status, commands = runner([[]], "--workers", "1", "--out", str(tmp_path / "a"))
    assert status == 0 and len(commands) == 1

    def failing(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, 2, stdout="ERROR collecting x", stderr=""
        )

    monkeypatch.setattr(run_aotriton.subprocess, "run", failing)
    assert run_aotriton.main(["--workers", "1", "--out", str(tmp_path / "b")]) == 2


def test_every_pytest_command_checks_the_hipdnn_prefix(runner, tmp_path):
    status, commands = runner(
        [["t.py::a"]], "--workers", "1", "--out", str(tmp_path / "o")
    )
    assert status == 0
    assert commands and all(
        "--hipdnn-prefix=/opt/fake-rocm" in cmd for cmd, _ in commands
    )


def test_runner_refuses_to_start_without_a_hipdnn_prefix(monkeypatch, tmp_path):
    monkeypatch.delenv("ROCM_PATH")
    with pytest.raises(SystemExit, match="no hipDNN prefix"):
        run_aotriton.main(["--workers", "1", "--out", str(tmp_path / "o")])
