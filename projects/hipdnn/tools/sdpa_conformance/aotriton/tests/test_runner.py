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


def test_fwd_no_stats_is_refused_outside_test_forward(runner, tmp_path):
    for argv, ids in (
        (["--suite", "default"], []),
        (["--suite", "forward", "--packed-gpu"], []),
    ):
        with pytest.raises(SystemExit, match="--fwd-no-stats"):
            runner(
                ids,
                "--workers",
                "1",
                "--out",
                str(tmp_path / "o"),
                "--fwd-no-stats",
                *argv,
            )
    status, commands = runner(
        [["tests/test_forward.py::a", "tests/test_backward.py::b"]],
        "--workers", "1", "--out", str(tmp_path / "p"),
        "--suite", "forward", "--fwd-no-stats",
    )  # fmt: skip
    assert status == 2 and len(commands) == 1  # collection only; nothing ran


def test_fwd_no_stats_exports_the_env_var_only_when_asked(
    runner, tmp_path, monkeypatch
):
    monkeypatch.setenv(
        run_aotriton.FWD_NO_STATS_ENV, "1"
    )  # stale value from the caller
    _, plain = runner([["t/test_forward.py::a"]], "--workers", "1", "--out", str(tmp_path / "a"), "--suite", "forward")  # fmt: skip
    _, flagged = runner([["t/test_forward.py::a"]], "--workers", "1", "--out", str(tmp_path / "b"), "--suite", "forward", "--fwd-no-stats")  # fmt: skip
    assert all(run_aotriton.FWD_NO_STATS_ENV not in env for _, env in plain)
    assert len(flagged) == 2
    assert all(env[run_aotriton.FWD_NO_STATS_ENV] == "1" for _, env in flagged)


def test_layout_mode_is_exported_to_every_pytest_process(runner, tmp_path, monkeypatch):
    monkeypatch.setenv(
        layout.LAYOUT_MODE_ENV, "efficient"
    )  # stale value from the caller
    _, default = runner([["t/test_forward.py::a"]], "--workers", "1", "--out", str(tmp_path / "a"), "--suite", "forward")  # fmt: skip
    _, chosen = runner([["t/test_forward.py::a"]], "--workers", "1", "--out", str(tmp_path / "b"), "--suite", "forward", "--layout-mode", "aotriton")  # fmt: skip
    assert len(default) == len(chosen) == 2
    assert all(env[layout.LAYOUT_MODE_ENV] == "pytorch" for _, env in default)
    assert all(env[layout.LAYOUT_MODE_ENV] == "aotriton" for _, env in chosen)


def test_an_unknown_layout_mode_is_refused(runner, tmp_path):
    with pytest.raises(SystemExit):
        runner([], "--workers", "1", "--out", str(tmp_path), "--layout-mode", "bshd")


FWD = "vendor/test_forward.py"
BWD = "vendor/test_backward.py"


def pass1_row(nodeid, fwd_state="", bwd_state="", fwd_engines="", bwd_engines=""):
    return {
        "nodeid": nodeid,
        "outcome": "pass",
        "fwd": fwd_state,
        "bwd": bwd_state,
        "fwd_engines": fwd_engines,
        "bwd_engines": bwd_engines,
    }


def test_all_engines_sweeps_each_non_winning_engine_over_its_cases_only(
    monkeypatch, tmp_path
):
    pass1 = [
        # forward cases: winner excluded per case; a case a lone engine served adds no work
        pass1_row(f"{FWD}::f1", "hipdnn:A", fwd_engines="A;B;C"),
        pass1_row(f"{FWD}::f2", "hipdnn:B", fwd_engines="A;B"),
        pass1_row(f"{FWD}::f3", "hipdnn:A", fwd_engines="A"),
        # backward cases read bwd; their forward list is not swept
        pass1_row(f"{BWD}::b1", "hipdnn:X", "hipdnn:A", "X;Y", "A;B"),
        pass1_row(f"{BWD}::b2", "ref:why", "hipdnn:B", "", "A;B"),
        pass1_row(f"{BWD}::b3", "hipdnn:X", "declined:no", "X", ""),
    ]
    fake = FakePytest([[r["nodeid"] for r in pass1]])
    fake_call = fake.__call__

    def call(cmd, env=None, **kw):
        if "--collect-only" not in cmd and f"--results-dir={tmp_path}" in cmd:
            (tmp_path / "results.csv").write_text(
                ",".join(run_aotriton.reconcile.FIELDS)
                + "\n"
                + "".join(run_aotriton.reconcile.format_row(r) for r in pass1)
            )
        return fake_call(cmd, env=env, **kw)

    monkeypatch.setattr(run_aotriton.subprocess, "run", call)
    monkeypatch.setenv("ROCM_PATH", "/opt/fake-rocm")
    status = run_aotriton.main(
        ["--workers", "2", "--out", str(tmp_path), "--all-engines", "--shard", "1/2", "--layout-mode", "efficient"]  # fmt: skip
    )
    assert status == 0
    sweeps = {}
    for cmd, env in fake.commands[2:]:
        assert env[layout.LAYOUT_MODE_ENV] == "efficient"  # pass 2 inherits the mode
        (pin,) = [k for k in env if k in run_aotriton.ENGINE_ENV.values() and env[k]]
        listing = next(a for a in cmd if a.startswith("--hipdnn-nodeids-file=")).split(
            "="
        )[1]
        sweeps[(pin, env[pin])] = [
            line.split("::")[-1] for line in open(listing).read().splitlines()
        ]
        assert not any(a.startswith("--shard") for a in cmd)  # the ids are the shard
        assert f"--results-dir={tmp_path}" not in cmd
    assert sweeps == {
        ("HIPDNN_FWD_ENGINE", "A"): ["f2"],
        ("HIPDNN_FWD_ENGINE", "B"): ["f1"],
        ("HIPDNN_FWD_ENGINE", "C"): ["f1"],
        ("HIPDNN_BWD_ENGINE", "A"): ["b2"],
        ("HIPDNN_BWD_ENGINE", "B"): ["b1"],
    }
    assert (tmp_path / "all-engines" / "fwd" / "C" / "engine.txt").read_text() == "C\n"


def test_all_engines_is_refused_with_fake_or_a_preset_pin(
    runner, tmp_path, monkeypatch
):
    with pytest.raises(SystemExit, match="--fake"):
        runner([], "--workers", "1", "--out", str(tmp_path), "--all-engines", "--fake", "always-decline")  # fmt: skip
    monkeypatch.setenv("HIPDNN_BWD_ENGINE", "X")
    with pytest.raises(SystemExit, match="HIPDNN_BWD_ENGINE"):
        runner([], "--workers", "1", "--out", str(tmp_path), "--all-engines")
