# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What lets one command run every round unattended, with stub stages: a
dead orchestrator's stages, retried transient failures, cells stage05 could
not train, and the code, engine and config a run is pinned to."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

import run_orchestrator as ro
from gpu_fakes import wait_for
from lib import attempts, gpu_lease, pinning
from test_run_orchestrator import (
    arg,
    calls,
    env,
    make_config,
    run,
    stubs,
    visible_gpus,
)

# A full run of one round deploys no bundle, which leaves stage 7 nothing to
# validate.
ONE_ROUND = {"full.total_rounds": 1, "stage07.bench_yamls": []}


def stage05_runs(log):
    return [c for c in calls(log) if c["stage"] == "stage05"]


def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rpartition(")")[2].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _start(cfg, run_dir, stubs, pid_file, **env):
    """The orchestrator in a process of its own, a full run of `cfg`; returns
    it and the pid its sleeping stage writes to `pid_file`."""
    out = (pid_file.parent / "orchestrator.out").open("w")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(ro.__file__)),
            "--config",
            str(cfg),
            "--run-dir",
            str(run_dir),
            "--skip-preflight",
            "--no-report",
            "--stages-dir",
            str(stubs),
            "--mode",
            "full",
        ],
        env=dict(os.environ, STUB_PID_FILE=str(pid_file), **env),
        stdout=out,
        stderr=subprocess.STDOUT,
    )
    started = wait_for(lambda: pid_file.is_file() or proc.poll() is not None, 120)
    assert started and pid_file.is_file(), Path(out.name).read_text()[-2000:]
    return proc, int(pid_file.read_text())


def test_a_killed_orchestrator_takes_its_stage_with_it(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **ONE_ROUND)
    run_dir = tmp_path / "run"
    proc, stage = _start(cfg, run_dir, stubs, tmp_path / "pid", STUB_SLEEP="stage03")
    try:
        running = attempts.Running(run_dir).records()
        (token, rec), *_ = running.items()
        assert len(running) == 1 and rec["label"] == "round_0 stage03"
        assert rec["pid"] == rec["pgid"] == stage
        assert attempts.processes_of(token) == [stage]
        proc.kill()
        proc.wait()
        assert wait_for(lambda: not _alive(stage), 30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    assert not (run_dir / attempts.RUNNING_FILE).exists()
    assert list((run_dir / "round_0" / "stale").glob("*/stage03.log"))


def test_a_stage_that_outlives_its_orchestrator_is_stopped_before_the_resume(
    tmp_path, stubs, env, monkeypatch, capsys
):
    cfg = make_config(tmp_path, **ONE_ROUND)
    run_dir = tmp_path / "run"
    proc, stage = _start(
        cfg,
        run_dir,
        stubs,
        tmp_path / "pid",
        STUB_SLEEP="stage03",
        STUB_IGNORE_TERM="stage03",
    )
    try:
        proc.kill()
        proc.wait()
        time.sleep(1)
        assert _alive(stage)
        monkeypatch.setattr(attempts, "STOP_GRACE_S", 1.0)
        n = len(calls(env))
        assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    finally:
        if _alive(stage):
            os.kill(stage, signal.SIGKILL)
    assert not _alive(stage)
    err = capsys.readouterr().err
    assert "stopped round_0 stage03 of an earlier invocation" in err
    assert f"(pids [{stage}])" in err
    assert [c["stage"] for c in calls(env)[n:]][:2] == ["stage03", "stage04"]
    assert not (run_dir / attempts.RUNNING_FILE).exists()


def test_a_copy_of_a_running_run_directory_leaves_the_original_alone(
    tmp_path, stubs, env, capsys
):
    """A run directory copied while its stage runs carries .running.json:
    the orchestrator that resumes the copy must not stop the original's
    stage."""
    cfg = make_config(tmp_path, **ONE_ROUND)
    original, copy = tmp_path / "run", tmp_path / "copy"
    proc, stage = _start(cfg, original, stubs, tmp_path / "pid", STUB_SLEEP="stage03")
    try:
        shutil.copytree(original, copy, symlinks=True)
        assert attempts.Running(copy).records()
        assert run(stubs, cfg, copy, "--mode", "full") == 0
        assert _alive(stage) and proc.poll() is None
        assert list(attempts.Running(original).records())
        err = capsys.readouterr().err
        assert f"names round_0 stage03 of {original}, another run directory" in err
        assert (copy / "round_0" / ro.ROUND_MARKER).is_file()
        assert not (copy / attempts.RUNNING_FILE).exists()
    finally:
        proc.kill()
        proc.wait()
    assert wait_for(lambda: not _alive(stage), 30)


def test_a_live_orchestrator_on_the_same_path_stops_the_invocation(
    tmp_path, stubs, env, capsys
):
    """The run directory was replaced by a copy while an orchestrator runs
    in it: that orchestrator still writes into the path, so the copy is not
    resumed."""
    cfg = make_config(tmp_path, **ONE_ROUND)
    run_dir, moved = tmp_path / "run", tmp_path / "moved"
    proc, stage = _start(cfg, run_dir, stubs, tmp_path / "pid", STUB_SLEEP="stage03")
    try:
        run_dir.rename(moved)
        shutil.copytree(moved, run_dir, symlinks=True)
        n = len(calls(env))
        assert run(stubs, cfg, run_dir, "--mode", "full") == 2
        assert len(calls(env)) == n and _alive(stage)
        err = capsys.readouterr().err
        assert f"runs round_0 stage03 (pid {proc.pid}) is still alive" in err
    finally:
        proc.kill()
        proc.wait()
    assert wait_for(lambda: not _alive(stage), 30)


def test_processes_a_stage_leaves_behind_are_stopped(
    tmp_path, stubs, env, monkeypatch, capsys
):
    pid_file = tmp_path / "left.pid"
    monkeypatch.setenv("STUB_LEAVE", "stage04")
    monkeypatch.setenv("STUB_PID_FILE", str(pid_file))
    cfg = make_config(tmp_path, **ONE_ROUND)
    assert run(stubs, cfg, tmp_path / "run", "--mode", "full") == 0
    left = int(pid_file.read_text())
    assert wait_for(lambda: not _alive(left), 30)
    assert "stopped 1 process(es) round_0 stage04 left running" in (
        capsys.readouterr().err
    )


def test_a_stage_holds_its_gpu_leases_while_it_runs(tmp_path, stubs, env, visible_gpus):
    cfg = make_config(tmp_path, **ONE_ROUND)
    run_dir = tmp_path / "run"
    proc, stage = _start(cfg, run_dir, stubs, tmp_path / "pid", STUB_SLEEP="stage03")
    try:
        held = set()
        for fd in os.listdir(f"/proc/{stage}/fd"):
            try:
                held.add(os.path.basename(os.readlink(f"/proc/{stage}/fd/{fd}")))
            except OSError:
                pass
        assert {"gpu0.lock", "gpu1.lock"} <= held
    finally:
        proc.kill()
        proc.wait()
    assert wait_for(lambda: not _alive(stage), 30)


def test_a_stage_that_waits_for_a_gpu_says_so(
    tmp_path, stubs, env, visible_gpus, monkeypatch, capsys
):
    monkeypatch.setattr(gpu_lease, "WAIT_REPORT_S", 0.5)
    hold = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(ro.THIS_DIR)!r})\n"
        "from lib import gpu_lease\n"
        "with gpu_lease.hold_gpus([1], 'an experiment'):\n"
        "    print('held', flush=True)\n"
        "    time.sleep(2)\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", hold], stdout=subprocess.PIPE, text=True
    )
    cfg = make_config(
        tmp_path, **{"full.total_rounds": 1, "runtime.gpu_lease_wait_s": 60}
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        assert run(stubs, cfg, tmp_path / "run", "--mode", "initial") == 0
    finally:
        holder.kill()
        holder.wait()
    err = capsys.readouterr().err
    assert f"round_0 stage03: GPU 1 is leased by pid {holder.pid} " in err
    assert "for 'an experiment'" in err and "waiting up to 1m00s for it" in err
    assert "round_0 stage03: still waiting for GPU 1 (" in err
    assert "round_0 stage03: GPU 1 leased after " in err


# ── retries ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fail, retried",
    [
        ("stage03:2", True),
        ("stage03c:2", True),
        ("stage03:signal", True),
        ("stage04:signal", True),
        ("stage04:SIGTERM", True),
        ("stage03:1", False),
        ("stage05:2", False),
        ("stage04:2", False),
        ("stage03:SIGUSR1", False),
    ],
)
def test_transient_failures_are_retried_and_others_stop_the_run(
    tmp_path, stubs, env, monkeypatch, fail, retried
):
    monkeypatch.setenv("STUB_FAIL", fail)
    monkeypatch.setenv("STUB_FAIL_ONCE_DIR", str(tmp_path))
    key = fail.split(":")[0]
    cfg = make_config(
        tmp_path,
        **{
            **ONE_ROUND,
            "contexts.list": [{"cu_budget": 128, "cotenant": True}],
            "train.catalog": "v3",
        },
    )
    run_dir = tmp_path / "run"
    rc = run(stubs, cfg, run_dir, "--mode", "full")
    tried = [c for c in calls(env) if c["stage"] == key]
    if not retried:
        assert rc != 0 and len(tried) == 1
        return
    assert rc == 0 and len(tried) == 2
    marker = json.loads((run_dir / "round_0" / key / ro.STAGE_MARKER).read_text())
    assert marker["attempt"] == 2
    assert list((run_dir / "round_0" / "stale").glob(f"*/{key}/partial.txt"))
    (timing,) = (run_dir / "timing").glob("*.json")
    rows = [r for r in json.loads(timing.read_text())["rows"] if r["stage"] == key]
    assert [r["rc"] == 0 for r in rows] == [False, True]
    assert rows[1]["attempt"] == 2


def test_retries_wait_longer_each_time_and_stop_at_the_limit(
    tmp_path, stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_FAIL", "stage03:2")
    waits = []
    monkeypatch.setattr(ro.time, "sleep", waits.append)
    cfg = make_config(
        tmp_path,
        **{
            "full.total_rounds": 1,
            "runtime.stage_retries": 2,
            "runtime.stage_retry_backoff_s": 5,
        },
    )
    assert run(stubs, cfg, tmp_path / "run", "--mode", "full") == 2
    assert len([c for c in calls(env) if c["stage"] == "stage03"]) == 3
    assert waits == [5, 10]
    assert "stage03 failed on its GPUs in each of 3 attempts" in capsys.readouterr().err
    none = make_config(tmp_path, **{"runtime.stage_retries": 0})
    n = len(calls(env))
    assert run(stubs, none, tmp_path / "run2", "--mode", "full") == 2
    assert len(calls(env)) - n == 2


def test_devices_the_environment_does_not_show_are_not_retried(
    tmp_path, stubs, env, monkeypatch, capsys
):
    """A lease that cannot be taken for a reason other than another holder
    (here: devices the environment does not show) fails at once."""
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "5")
    waits = []
    monkeypatch.setattr(ro.time, "sleep", waits.append)
    cfg = make_config(tmp_path, **ONE_ROUND)
    assert run(stubs, cfg, tmp_path / "run", "--mode", "full") == 2
    assert waits == [] and not [c for c in calls(env) if c["stage"] == "stage03"]
    err = capsys.readouterr().err
    assert "stage03 not started: cannot lease the GPUs of devices [0, 1]" in err
    assert "busy" not in err and "[retry]" not in err


@pytest.mark.parametrize(
    "rc, retried",
    [
        (-signal.SIGKILL, "was killed by SIGKILL"),
        (128 + signal.SIGTERM, "was killed by SIGTERM"),
        (-signal.SIGHUP, "was killed by SIGHUP"),
        (-signal.SIGSEGV, None),
        (-signal.SIGABRT, None),
        (128 + signal.SIGSEGV, None),
        (1, None),
    ],
)
def test_only_deaths_by_a_stop_signal_are_retried(tmp_path, rc, retried):
    cfg_path = make_config(tmp_path)
    cfg, _ = ro.load_config(cfg_path)
    pipe = ro.Pipeline(
        cfg=cfg,
        snapshot=cfg_path,
        run_dir=tmp_path / "run",
        paths=ro.derive_paths(cfg),
        stages_dir=ro.THIS_DIR / "stages",
        timing=ro.Timing(),
        force_deploy=False,
    )
    assert pipe.retry_reason("stage05", [], rc) == retried
    assert pipe.retry_reason("stage03", ["--devices", "0"], 2) == "failed on its GPUs"
    assert pipe.retry_reason("stage03", ["--devices", "0"], 2, "setup") is None
    assert pipe.retry_reason("stage05", [], 2, "busy") == "found a GPU lease busy"


def test_lease_and_retry_defaults():
    assert ro.cfg_get({}, "runtime.gpu_lease_wait_s") == 3600
    assert ro.cfg_get({}, "runtime.stage_retries") == 1
    assert ro.cfg_get({}, "runtime.stage_retry_backoff_s") == 300
    assert ro.cfg_get({}, "train.on_cell_failure") == "stop"


# ── cells stage05 could not train ────────────────────────────────────────────


def test_carry_goes_on_with_the_bundle_stage05_wrote(
    tmp_path, stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_FAIL", "stage05@round_1:3")
    cfg = make_config(
        tmp_path, **{"full.total_rounds": 2, "train.on_cell_failure": "carry"}
    )
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    rd = run_dir / "round_1"
    record = json.loads((rd / "stage05" / ro.CARRIED_CELLS).read_text())
    assert record["cells"] == [
        {
            "cell": "X#M>1",
            "why": "not trained: too few gemms (2)",
            "model": "none of its own (served by its nearest trained cell)",
        }
    ]
    marker = json.loads((rd / ro.ROUND_MARKER).read_text())
    assert marker["cells_carried"] == ["X#M>1"]
    s05 = json.loads((rd / "stage05" / ro.STAGE_MARKER).read_text())
    assert "carried" in s05["note"] and s05["cmd"][2].endswith("stage05_train.py")
    assert record["n_cells_trained"] == 4
    deploys = [c for c in calls(env) if c["stage"] == "stage06"]
    assert arg(deploys[-1], "--train-dir") == str(rd / "stage05")
    err = capsys.readouterr().err
    assert "X#M>1: not trained: too few gemms (2); keeps none of its own" in err
    assert "round_1 went on without training 1 cell: X#M>1 (" in err


@pytest.mark.parametrize(
    "trained, fraction, carried",
    [("0", 1.0, False), ("1", 0.25, False), ("1", 0.5, True), ("3", 0.25, True)],
)
def test_carry_refuses_a_round_that_trained_too_few_cells(
    tmp_path, stubs, env, monkeypatch, capsys, trained, fraction, carried
):
    """One untrained cell of `trained` + 1: carry goes on without at most
    train.carry_max_fraction of them, and never without all."""
    monkeypatch.setenv("STUB_FAIL", "stage05@round_1:3")
    monkeypatch.setenv("STUB_TRAINED", trained)
    cfg = make_config(
        tmp_path,
        **{
            "full.total_rounds": 2,
            "train.on_cell_failure": "carry",
            "train.carry_max_fraction": fraction,
        },
    )
    run_dir = tmp_path / "run"
    rc = run(stubs, cfg, run_dir, "--mode", "full")
    rd = run_dir / "round_1"
    assert ro.stage_complete(rd / "stage05") is carried
    if carried:
        assert rc == 0 and (rd / "stage05" / ro.CARRIED_CELLS).is_file()
        return
    assert rc == ro.STAGE05_CELLS_NOT_TRAINED
    assert not (rd / "stage05" / ro.CARRIED_CELLS).exists()
    err = capsys.readouterr().err
    assert f"stage05 trained {trained} of the {int(trained) + 1} cells" in err


def test_carry_goes_on_with_the_bundle_a_stopped_round_left(
    tmp_path, stubs, env, monkeypatch, capsys
):
    """A round stopped on untrained cells; the same command with carry goes
    on with the bundle stage05 wrote instead of training it again, unless
    stage05 is asked for again."""
    monkeypatch.setenv("STUB_FAIL", "stage05@round_1:3")
    run_dir = tmp_path / "run"
    stop = make_config(tmp_path, **{"full.total_rounds": 2})
    assert run(stubs, stop, run_dir, "--mode", "full") == ro.STAGE05_CELLS_NOT_TRAINED
    err = capsys.readouterr().err
    assert "set train.on_cell_failure: carry and run the same command" in err
    rd = run_dir / "round_1"
    assert (rd / "stage05" / ro.UNTRAINED_BUNDLE).is_file()

    carry = make_config(
        tmp_path, **{"full.total_rounds": 2, "train.on_cell_failure": "carry"}
    )
    n = len(stage05_runs(env))
    assert run(stubs, carry, run_dir, "--mode", "full") == 0
    assert len(stage05_runs(env)) == n
    assert "going on with it" in capsys.readouterr().out
    marker = json.loads((rd / ro.ROUND_MARKER).read_text())
    assert marker["cells_carried"] == ["X#M>1"]

    again = ["--from-round", "1", "--from-stage", "5"]
    assert run(stubs, carry, run_dir, "--mode", "full", *again) == 0
    assert len(stage05_runs(env)) == n + 1


def test_a_round_with_another_command_trains_its_cells_again(
    tmp_path, stubs, env, monkeypatch
):
    monkeypatch.setenv("STUB_FAIL", "stage05@round_1:3")
    run_dir = tmp_path / "run"
    stop = make_config(tmp_path, **{"full.total_rounds": 2})
    assert run(stubs, stop, run_dir, "--mode", "full") == ro.STAGE05_CELLS_NOT_TRAINED
    other = make_config(
        tmp_path,
        **{
            "full.total_rounds": 2,
            "train.on_cell_failure": "carry",
            "train.epochs": 7,
        },
    )
    n = len(stage05_runs(env))
    assert run(stubs, other, run_dir, "--mode", "full", "--allow-code-change") == 0
    assert len(stage05_runs(env)) == n + 1


def test_carry_waits_for_the_code_it_ran_with(
    tmp_path, stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_FAIL", "stage05@round_1:3")
    _change_code_after(
        monkeypatch,
        lambda: any(
            c["stage"] == "stage05" and "round_1" in arg(c, "--output-dir")
            for c in calls(env)
        ),
    )
    cfg = make_config(
        tmp_path, **{"full.total_rounds": 2, "train.on_cell_failure": "carry"}
    )
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 2
    assert "stage05 is not marked complete: the code changed" in capsys.readouterr().err
    assert not ro.stage_complete(run_dir / "round_1" / "stage05")
    assert not (run_dir / "round_1" / "stage05" / ro.CARRIED_CELLS).exists()


def test_carry_needs_the_bundle_stage05_wrote(
    tmp_path, stubs, env, monkeypatch, capsys
):
    monkeypatch.setenv("STUB_FAIL", "stage05@round_1:3")
    monkeypatch.setenv("STUB_NO_BUNDLE", "1")
    cfg = make_config(
        tmp_path, **{"full.total_rounds": 2, "train.on_cell_failure": "carry"}
    )
    run_dir = tmp_path / "run"
    rc = run(stubs, cfg, run_dir, "--mode", "full")
    assert rc == ro.STAGE05_CELLS_NOT_TRAINED
    assert "holds no complete bundle to go on with" in capsys.readouterr().err
    assert not ro.stage_complete(run_dir / "round_1" / "stage05")
    assert not (run_dir / "round_1" / ro.ROUND_MARKER).exists()


# ── pin ──────────────────────────────────────────────────────────────────────


def test_a_changed_config_needs_allow_code_change(tmp_path, stubs, env, capsys):
    run_dir = tmp_path / "run"
    first = make_config(tmp_path, **{"full.total_rounds": 2, "train.epochs": 11})
    assert run(stubs, first, run_dir, "--mode", "initial") == 0
    rec = pinning.read_run_pin(run_dir)
    assert rec["pin"]["config"]["values"]["train.epochs"] == 11
    marker = json.loads((run_dir / "round_0" / "stage03" / ro.STAGE_MARKER).read_text())
    assert marker["pin"] == pinning.summary(rec["pin"])
    snap = sorted((run_dir / "configs").glob("*_initial.yaml"))[0].read_text()
    assert "# Produced with: pipeline code " in snap

    moved = make_config(
        tmp_path,
        **{
            "full.total_rounds": 3,
            "train.epochs": 11,
            "runtime.devices": [1],
            "train.on_cell_failure": "carry",
        },
    )
    assert run(stubs, moved, run_dir, "--mode", "active") == 0

    changed = make_config(tmp_path, **{"full.total_rounds": 3, "train.epochs": 22})
    capsys.readouterr()
    n = len(calls(env))
    assert run(stubs, changed, run_dir, "--mode", "active") == 2
    assert len(calls(env)) == n
    err = capsys.readouterr().err
    assert "config train.epochs: 11 -> 22" in err
    assert "--allow-code-change" in err
    assert run(stubs, changed, run_dir, "--mode", "active", "--allow-code-change") == 0
    rec = pinning.read_run_pin(run_dir)
    assert rec["history"][0]["changes"] == ["config train.epochs: 11 -> 22"]
    assert rec["pin"]["config"]["values"]["train.epochs"] == 22


def _change_code_after(monkeypatch, changed):
    """A pipeline file appears on disk once `changed()` is true."""
    real = pinning.code_file_digests

    def digests(pipeline_dir):
        files = real(pipeline_dir)
        if changed():
            files["lib/added_meanwhile.py"] = "0" * 64
        return files

    monkeypatch.setattr(pinning, "code_file_digests", digests)


def test_code_changed_while_the_run_goes_on_stops_it(
    tmp_path, stubs, env, monkeypatch, capsys
):
    _change_code_after(
        monkeypatch, lambda: any(c["stage"] == "stage03" for c in calls(env))
    )
    cfg = make_config(tmp_path, **ONE_ROUND)
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 2
    err = capsys.readouterr().err
    assert "stage03 is not marked complete: the code changed while" in err
    assert "files: lib/added_meanwhile.py (new)" in err
    assert not ro.stage_complete(run_dir / "round_0" / "stage03")
    rec = pinning.read_run_pin(run_dir)
    assert rec["first_stage"] == "round_0/stage01"
    assert "lib/added_meanwhile.py" not in rec["pin"]["code"]["files"]
    assert run(stubs, cfg, run_dir, "--mode", "full") == 2
    assert "pipeline code: " in capsys.readouterr().err
    assert run(stubs, cfg, run_dir, "--mode", "full", "--allow-code-change") == 0
    assert (run_dir / "round_0" / ro.ROUND_MARKER).is_file()


def test_validation_names_a_change_and_goes_on(
    tmp_path, stubs, env, monkeypatch, capsys
):
    """Validation produces no data or model of the run: another code or
    config is named, never refused, and the run's pin stays."""
    only8 = {"full.total_rounds": 1, "stage07.bench_yamls": []}
    cfg = make_config(tmp_path, **only8)
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    pin = (run_dir / pinning.RUN_PIN).read_text()
    _change_code_after(monkeypatch, lambda: True)
    other = make_config(tmp_path, **only8, **{"train.epochs": 9})
    capsys.readouterr()
    assert run(stubs, other, run_dir, "--mode", "validate") == 0
    err = capsys.readouterr().err
    assert "validating with a change of what the run used: pipeline code" in err
    assert "config train.epochs: 200 -> 9" in err
    assert (run_dir / pinning.RUN_PIN).read_text() == pin
    assert run(stubs, other, run_dir, "--mode", "full", "--from-round", "0") == 2


def test_an_audit_directory_keeps_its_config_keys_and_records_code(tmp_path, capsys):
    """Runs of other code share and extend an audit set: its pin keeps the
    config keys its benches read and records a change of code."""
    assert ro.pin_parts("audit") == ("config",)
    assert ro.pin_parts("validate") == () and ro.pin_parts("full") == pinning.PARTS
    old = {
        "code": {"digest": "a", "files": {"lib/a.py": "1"}},
        "engine": {"sha256": "e"},
        "config": pinning.config_record({"bench.skip_slow_solution_ratio": 0.45}),
    }
    pinning.write_run_pin(tmp_path, old, first_stage="stage01")
    code = {**old, "code": {"digest": "b", "files": {"lib/a.py": "2"}}}
    assert ro.settle_run_pin(tmp_path, code, False, ("config",)) == []
    assert "recorded: pipeline code: " in capsys.readouterr().err
    rec = pinning.read_run_pin(tmp_path)
    assert rec["pin"]["code"]["digest"] == "b" and rec["first_stage"] == "stage01"
    assert rec["history"][0]["changes"][0].endswith("files: lib/a.py")
    bench = {
        **code,
        "config": pinning.config_record({"bench.skip_slow_solution_ratio": 0.5}),
    }
    errs = ro.settle_run_pin(tmp_path, bench, False, ("config",))
    assert errs and "  config bench.skip_slow_solution_ratio: 0.45 -> 0.5" in errs


def test_a_change_the_invocation_does_not_pin_reaches_the_markers(
    tmp_path, monkeypatch, capsys
):
    cfg_path = make_config(tmp_path)
    cfg, _ = ro.load_config(cfg_path)

    def pipeline(parts):
        return ro.Pipeline(
            cfg=cfg,
            snapshot=cfg_path,
            run_dir=tmp_path / "run",
            paths=ro.derive_paths(cfg),
            stages_dir=ro.THIS_DIR / "stages",
            timing=ro.Timing(),
            force_deploy=False,
            pin=pinning.make_pin(ro.THIS_DIR, {}),
            pin_parts=parts,
        )

    training, audit = pipeline(pinning.PARTS), pipeline(("config",))
    _change_code_after(monkeypatch, lambda: True)
    assert training.pin_drift()[0].startswith("pipeline code: ")
    assert audit.pin_drift() == []
    assert "going on with a change this run does not pin" in capsys.readouterr().err
    assert "lib/added_meanwhile.py" in audit.pin["code"]["files"]
    assert audit.pin_drift() == []


def test_an_audit_directory_pins_what_its_benches_read():
    cfg = yaml.safe_load(
        """
config_id: a
arch: gfx950
hipblaslt: {a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r,
  scale_type: f32_r, compute_type: c_f32_r, transA: T, transB: N,
  library_stem: TensileLibrary_X_gfx950}
runtime: {devices: [0], rocm_libraries_root: /r}
train: {epochs: 3}
"""
    )
    full = ro.pinned_config(cfg)
    audit = ro.pinned_config(cfg, audit=True)
    assert full["train.epochs"] == 3 and "train.epochs" not in audit
    assert audit["bench.skip_slow_solution_ratio"] == 0.0
    assert "runtime.devices" not in full and "stage08.datasets" not in full
    assert "runtime.min_free_gb" not in full and "report.timeout_s" not in full
    assert full["runtime.rocm_libraries_root"] == "/r"
    read = {
        "planner.enabled",
        "planner.mode",
        "planner.ranker",
        "planner.model_file",
        "planner.bank",
        "planner.bank_library_stems",
    }
    assert {k for k in audit if k.startswith("planner.")} == read
    assert read < {k for k in full if k.startswith("planner.")}
    for key in ro.UNPINNED_KEYS + ro.AUDIT_UNPINNED_KEYS:
        ro._spec_at(key)
