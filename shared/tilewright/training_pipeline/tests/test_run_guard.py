# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The orchestrator's side of the run guard (config keys, excluded GPUs,
the health gate before a stage, the stages' environment) and
scripts/run_guard.py, on a fake sysfs."""

import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

import gpu_fakes as gf
import run_guard
import run_orchestrator as ro
from lib import gpu_health, kfd, procs
from test_run_orchestrator import _pipeline, make_config


@pytest.fixture
def gpus(tmp_path, monkeypatch):
    root = gf.write_sysfs(tmp_path / "sys")
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    for var in kfd.HIP_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv(kfd.ROCR_VAR, raising=False)
    return root


def _cfg(tmp_path, **overrides):
    return ro.load_config(make_config(tmp_path, **overrides))[0]


def test_the_guard_keys_are_checked(tmp_path):
    cfg = _cfg(
        tmp_path,
        **{
            "runtime.exclude_gpus": [3, "0000:4c:00.0", "GPU-00000000deadbeef"],
            "bench.quiet_host": {"every_s": 5, "abort_samples": 3},
            "bench.health": {"poll_s": 10, "canary": {"tolerance": 0.1}},
        },
    )
    assert ro.validate_config(cfg)[0] == []
    for value, needle in (
        ([-1], "runtime.exclude_gpus[0]: expected a GPU index"),
        ([True], "runtime.exclude_gpus[0]: expected a GPU index"),
        ([""], "runtime.exclude_gpus[0]: expected a GPU index"),
    ):
        bad = dict(cfg, runtime=dict(cfg["runtime"], exclude_gpus=value))
        assert any(e.startswith(needle) for e in ro.validate_config(bad)[0])
    bad = dict(cfg, bench={"quiet_host": {"quiet_busy": 0.5, "hot_busy": 0.2}})
    assert any(
        "quiet_busy 0.5 above hot_busy 0.2" in e for e in ro.validate_config(bad)[0]
    )
    bad = dict(cfg, bench={"health": {"canary": {"tolerance": 0}}})
    assert any("tolerance" in e for e in ro.validate_config(bad)[0])
    warn = dict(cfg, bench={"stall_timeout_s": 0})
    assert any("holds its GPU until" in w for w in ro.validate_config(warn)[1])
    assert ro.cfg_get({}, "bench.stall_timeout_s") == 600.0


def test_the_guard_keys_are_not_pinned(tmp_path):
    plain = _cfg(tmp_path)
    guarded = _cfg(
        tmp_path,
        **{
            "runtime.exclude_gpus": [3],
            "bench.quiet_host": {"enabled": False},
            "bench.health": {"max_wait_s": 60},
            "bench.stall_timeout_s": 900,
        },
    )
    assert ro.pinned_config(plain) == ro.pinned_config(guarded)
    assert ro.pinned_config(plain, audit=True) == ro.pinned_config(guarded, audit=True)
    assert not any(
        k.startswith(("bench.quiet_host", "bench.health", "runtime.exclude"))
        for k in ro.pinned_config(guarded)
    )


def test_excluded_gpus_leave_every_device_list(tmp_path, gpus):
    cfg = _cfg(
        tmp_path,
        **{
            "runtime.devices": [0, 1, 2],
            "runtime.exclude_gpus": [2, "0a:00.0", "ff:00.0"],
            "audit.devices": [1, 2],
            "hazards.devices": [0, 2],
        },
    )
    out, audit, notes, errs = ro.exclude_gpus(cfg, [0, 1])
    assert out["runtime"]["devices"] == [0] and out["audit"]["devices"] == []
    assert out["hazards"]["devices"] == [0]
    assert audit == [0] and cfg["runtime"]["devices"] == [0, 1, 2]
    assert "runtime.exclude_gpus: ff:00.0 names no GPU here" in notes
    assert errs == ["audit.devices: every device is excluded by runtime.exclude_gpus"]
    same, _, notes, errs = ro.exclude_gpus(_cfg(tmp_path))
    assert (notes, errs) == ([], []) and same["runtime"]["devices"] == [0, 1]
    text = ro.guard_text(cfg)
    assert "excluded GPUs 2, 0a:00.0, ff:00.0" in text
    assert "killed and quarantined in the run" in text
    assert "with the host's ttm workers" in text and "a canary of 4" in text
    off = _cfg(tmp_path, **{"bench.health": {"enabled": False, "ttm": False}})
    text = ro.guard_text(off)
    assert "no GPU health checks" in text and "no health canary" in text


def test_a_quoted_index_excludes_the_gpu_of_that_index(tmp_path, gpus):
    cfg = _cfg(tmp_path, **{"runtime.exclude_gpus": ["1"]})
    assert ro.validate_config(cfg)[0] == []
    out, _, notes, errs = ro.exclude_gpus(cfg)
    assert out["runtime"]["devices"] == [0] and (notes, errs) == (
        [
            "runtime.devices: devices [1] left out (runtime.exclude_gpus names their GPUs)"
        ],
        [],
    )


def test_stages_get_the_runs_guard_and_session(tmp_path):
    p = _pipeline(tmp_path, make_config(tmp_path))
    env = p.stage_env()
    assert env[gpu_health.GUARD_DIR_VAR] == str(tmp_path / "run" / "guard")
    assert env[gpu_health.SESSION_VAR] == p.invocation


def test_every_stage_of_an_invocation_gets_its_guard_and_session(
    tmp_path, monkeypatch, capsys
):
    import json

    from test_run_orchestrator import STUB, calls, run

    stubs = tmp_path / "stages"
    stubs.mkdir()
    for key, sdef in ro.STAGE_DEFS.items():
        (stubs / sdef.script).write_text(STUB.replace("__KEY__", repr(key)))
    seen = tmp_path / "env.jsonl"
    monkeypatch.setenv("STUB_LOG", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("STUB_ENV_LOG", str(seen))
    names = [gpu_health.GUARD_DIR_VAR, gpu_health.SESSION_VAR]
    monkeypatch.setenv("STUB_ENV", json.dumps(names))
    for k in ("STUB_FAIL", "STUB_RETRAIN", "STUB_GATE"):
        monkeypatch.delenv(k, raising=False)
    path = make_config(tmp_path, **{"full.total_rounds": 1})
    run_dir = tmp_path / "run"
    assert run(stubs, path, run_dir, "--mode", "initial") == 0
    first = calls(seen)
    assert {r["env"][names[0]] for r in first} == {str(run_dir.resolve() / "guard")}
    (session,) = {r["env"][names[1]] for r in first}
    assert session and session.endswith("_initial")
    assert "bench guard  : a part starts when" in capsys.readouterr().out
    seen.unlink()
    assert run(stubs, path, run_dir, "--mode", "active") == 0
    assert {r["env"][names[1]] for r in calls(seen)} - {session}


def _level(root, gpu):
    d = root / "class" / "drm" / f"renderD{gf.GPUS[gpu].render_minor}" / "device"
    return d / "power_dpm_force_performance_level"


def test_a_gpu_unlike_the_runs_start_holds_only_the_stages_that_need_it(
    tmp_path, gpus, monkeypatch, capsys
):
    monkeypatch.setattr(gpu_health, "CONFIRM_S", 0.01)
    path = make_config(tmp_path, **{"bench.health": {"poll_s": 1.0, "max_wait_s": 1.5}})
    p = _pipeline(tmp_path, path)
    assert p.wait_gpu_health([0, 1], "round_0 stage03", per_gpu=True)
    _level(gpus, 1).write_text("manual\n")
    assert p.wait_gpu_health([0, 1], "round_0 stage03", per_gpu=True)
    err = capsys.readouterr().err
    assert "GPU 1 (0000:0a:00.0): performance level 'manual'" in err
    assert "This holds 0000:0a:00.0 alone" in err
    assert "the run's other GPUs go on" in err
    assert not p.wait_gpu_health([1], "round_0 stage03", per_gpu=True)
    assert not p.wait_gpu_health([0, 1], "round_0 stage07")
    assert p.wait_gpu_health([0], "round_0 stage07")
    assert p.wait_gpu_health([], "round_0 stage05")
    _level(gpus, 1).write_text("auto\n")
    assert p.wait_gpu_health([0, 1], "round_0 stage07")
    doc = gpu_health.Guard(p.guard_dir, "x").read()
    assert doc["hold"] is None and len(doc["holds"]) == 1


def test_no_stage_starts_while_every_gpu_departs(tmp_path, gpus, monkeypatch, capsys):
    monkeypatch.setattr(gpu_health, "CONFIRM_S", 0.01)
    path = make_config(tmp_path, **{"bench.health": {"poll_s": 1.0, "max_wait_s": 1.5}})
    p = _pipeline(tmp_path, path)
    assert p.wait_gpu_health([0, 1], "round_0 stage03", per_gpu=True)
    for gpu in (0, 1):
        _level(gpus, gpu).write_text("manual\n")
    assert not p.wait_gpu_health([0, 1], "round_0 stage03", per_gpu=True)
    assert not p.wait_gpu_health([0], "round_0 stage03", per_gpu=True)
    err = capsys.readouterr().err
    assert "This holds every GPU of the run" in err
    hold = gpu_health.Guard(p.guard_dir, "x").hold()
    assert hold["host_wide"] and gpu_health.run_wide(hold)
    _level(gpus, 0).write_text("auto\n")
    assert p.wait_gpu_health([0, 1], "round_0 stage03", per_gpu=True)
    assert not gpu_health.run_wide(gpu_health.Guard(p.guard_dir, "x").hold())


def test_a_stage_on_held_gpus_is_not_started(tmp_path, gpus, monkeypatch, capsys):
    monkeypatch.setattr(gpu_health, "CONFIRM_S", 0.01)
    path = make_config(tmp_path, **{"bench.health": {"poll_s": 1.0, "max_wait_s": 1.0}})
    p = _pipeline(tmp_path, path)
    p.wait_gpu_health([0, 1], "x")
    _level(gpus, 0).write_text("low\n")
    rc, cause = p._stage_attempt(tmp_path, "stage07", [], 0, None, 1)
    assert (rc, cause) == (2, "health")
    assert (
        "stage07 not started: the run's GPUs fail their checks"
        in capsys.readouterr().err
    )
    _level(gpus, 1).write_text("low\n")
    rc, cause = p._stage_attempt(tmp_path, "stage03", ["--devices", "0,1"], 0, None, 1)
    assert (rc, cause) == (2, "health")
    assert p.retry_reason("stage03", [], 2, "health") == (
        "found the run's GPUs failing their checks"
    )


HOLDER = """
import sys, time
sys.path.insert(0, {root!r})
from lib import gpu_lease
with gpu_lease.hold_gpus([1], "another user"):
    print("held", flush=True)
    time.sleep(60)
"""


def test_a_gpu_another_process_tree_leases_does_not_hold_a_stage(
    tmp_path, gpus, monkeypatch, capsys
):
    monkeypatch.setattr(gpu_health, "CONFIRM_S", 0.01)
    monkeypatch.setenv(kfd.ROCR_VAR, "0,1")
    path = make_config(tmp_path, **{"bench.health": {"poll_s": 1.0, "max_wait_s": 1.0}})
    p = _pipeline(tmp_path, path)
    assert p.wait_gpu_health([0, 1], "x")
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER.format(root=str(ro.THIS_DIR))],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        _level(gpus, 1).write_text("manual\n")
        checked = []
        real = p.wait_gpu_health
        monkeypatch.setattr(
            p,
            "wait_gpu_health",
            lambda d, label, per_gpu=False: checked.append(list(d))
            or real(d, label, per_gpu),
        )
        rc, cause = p._stage_attempt(
            tmp_path, "stage03", ["--devices", "0,1"], 0, None, 1
        )
        assert cause != "health" and checked == [[0]]
        assert gpu_health.Guard(p.guard_dir, "x").hold() is None
        assert "GPU 1 left to its holder" in capsys.readouterr().err
    finally:
        holder.kill()
        holder.wait()


def test_the_correctness_gate_waits_for_the_runs_gpus(tmp_path, gpus, monkeypatch):
    monkeypatch.setattr(gpu_health, "CONFIRM_S", 0.01)
    path = make_config(
        tmp_path,
        **{
            "bench.health": {"poll_s": 1.0, "max_wait_s": 1.0},
            "correctness.devices": [1],
        },
    )
    p = _pipeline(tmp_path, path)
    assert p.health_devices("stage00", []) == [1]
    assert p.health_devices("stage00a", []) == [0]
    assert p.health_devices("stage03", [0]) == [0]
    assert p.wait_gpu_health([1], "x")
    _level(gpus, 1).write_text("manual\n")
    rc, cause = p._stage_attempt(tmp_path, "stage00", p.args_stage00(), None, None, 1)
    assert (rc, cause) == (2, "health")


def test_the_hazard_canary_waits_for_its_gpus(tmp_path, gpus, monkeypatch):
    monkeypatch.setattr(gpu_health, "CONFIRM_S", 0.01)
    path = make_config(
        tmp_path,
        **{
            "bench.health": {"poll_s": 1.0, "max_wait_s": 1.0},
            "hazards.devices": [1],
        },
    )
    p = _pipeline(tmp_path, path)
    assert p.health_devices("stage00a", []) == [1]
    assert p.wait_gpu_health([0, 1], "x")
    _level(gpus, 1).write_text("manual\n")
    rc, cause = p._stage_attempt(tmp_path, "stage00a", p.args_stage00a(), None, None, 1)
    assert (rc, cause) == (2, "health")


def test_the_watchdog_stops_a_bench_its_abort_probe_names():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    calls = []

    def probe():
        calls.append(1)
        return "contended" if len(calls) >= 3 else None

    rc, reason = procs.wait_with_stall_watchdog(
        proc, lambda: 0, 0, poll_s=0.05, probe_abort=probe
    )
    assert reason == "contended" and rc != 0 and len(calls) == 3


# ── scripts/run_guard.py ─────────────────────────────────────────────────────


def test_status_accept_and_release(tmp_path, gpus, capsys):
    run = tmp_path / "run"
    guard = gpu_health.Guard(run / gpu_health.GUARD_DIR, "s")
    g0, g1, _ = kfd.gpus()
    guard.baseline([g0, g1])
    guard.record_canary(
        gpu_health.gpu_key(g1),
        {"reference": True, "passed": True, "readings_us": [1.0], "taken": "t"},
    )
    guard.quarantine_lines(
        [{"shape": [1, 2, 3, 1], "in_flight": "kernel K", "gpu": 1, "stage": "s"}]
    )
    guard.quarantine_lines([{"shape": [4, 5, 6, 1]}])
    _level(gpus, 1).write_text("manual\n")
    guard.settle(
        gpu_health.Check(
            [gpu_health.gpu_key(g1)],
            {gpu_health.gpu_key(g1): [("perf_level", "GPU 1: performance level")]},
        )
    )

    assert run_guard.main(["status", str(run)]) == 0
    out = capsys.readouterr().out
    assert "HELD since" in out and "0000:0a:00.0 alone:" in out
    assert "0000:0a:00.0: GPU 1: performance level (since" in out
    assert "perf_level" in out and "'auto' -> 'manual'" in out
    assert "2 quarantined line(s)" in out and "1x2x3x1: hung in kernel K" in out
    assert "quarantined for bench" in out

    assert run_guard.main(["accept", str(run), "--gpu", gpu_health.gpu_key(g1)]) == 0
    out = capsys.readouterr().out
    assert "accepted: 0000:0a:00.0: performance level 'manual'" in out
    assert "the run's hold is lifted" in out
    doc = guard.read()
    assert doc["hold"] is None and doc["holds"][-1]["cleared_by"] == "accept"
    assert doc["baseline"]["gpus"]["0000:0a:00.0"]["readings"]["perf_level"] == "manual"
    assert doc["baseline"]["gpus"]["0000:0b:00.0"]["readings"]["perf_level"] == "auto"
    assert guard.reference("0000:0a:00.0") is None

    assert run_guard.main(["release", str(run), "--shape", "1,2,3,1"]) == 0
    assert guard.quarantined_shapes() == {(4, 5, 6, 1)}
    assert run_guard.main(["release", str(run / gpu_health.GUARD_DIR), "--all"]) == 0
    assert guard.quarantined_shapes() == set()
    assert "released 1 line(s); 0 stay quarantined" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run_guard.main(["status", str(tmp_path / "nothing")])


def test_accept_of_one_gpu_keeps_the_hold_of_another(tmp_path, gpus, capsys):
    run = tmp_path / "run"
    guard = gpu_health.Guard(run / gpu_health.GUARD_DIR, "s")
    g0, g1, _ = kfd.gpus()
    guard.baseline([g0, g1])
    d = gpus / "class" / "drm" / f"renderD{gf.GPUS[0].render_minor}" / "device"
    (d / "pp_features").write_text("00. DPM ( 0) : disabled\n")
    k0, k1 = gpu_health.gpu_key(g0), gpu_health.gpu_key(g1)
    guard.settle(
        gpu_health.Check(
            [k0, k1, gpu_health.HOST],
            {k0: [("power_management", "pm off")], k1: [("perf_level", "level")]},
        )
    )
    assert run_guard.main(["accept", str(run), "--gpu", k1]) == 0
    out = capsys.readouterr().out
    assert "the run stays held for: pm off" in out
    assert set(guard.hold()["entries"]) == {k0}
    assert run_guard.main(["accept", str(run)]) == 0
    out = capsys.readouterr().out
    assert f"{k0}: power management is off; the run's next check holds it" in out
    assert guard.hold() is None
    gone = "0000:4c:00.0"
    guard.settle(gpu_health.Check([gone], {gone: [("perf_level", "fell off")]}))
    assert run_guard.main(["accept", str(run), "--gpu", gone]) == 0
    out = capsys.readouterr().out
    assert f"{gone}: not found on this host" not in out and guard.hold() is None


@pytest.mark.parametrize("first", ["a hold", "a line"])
def test_releasing_lines_leaves_holds_settled_released_and_accepted(
    tmp_path, gpus, first, capsys
):
    run = tmp_path / "run"
    guard = gpu_health.Guard(run / gpu_health.GUARD_DIR, "s", by="t")
    g0, g1, _ = kfd.gpus()
    guard.baseline([g0, g1])
    quick = dict(ttm=lambda: [], sleep=lambda _s: None)

    def check():
        return guard.settle(gpu_health.check(guard, [g0, g1], **quick))[0]

    if first == "a hold":
        _level(gpus, 0).write_text("manual\n")
        assert check() == "raised"
        _level(gpus, 0).write_text("auto\n")
        assert check() == "cleared"
    guard.quarantine_lines([{"shape": [1, 2, 3, 1]}])
    assert run_guard.main(["release", str(run), "--all"]) == 0
    _level(gpus, 1).write_text("manual\n")
    assert check() == "raised"
    assert check() == "held"
    _level(gpus, 1).write_text("auto\n")
    assert check() == "cleared"
    _level(gpus, 1).write_text("manual\n")
    assert check() == "raised"
    capsys.readouterr()
    assert run_guard.main(["accept", str(run), "--gpu", g1.bdf]) == 0
    out = capsys.readouterr().out
    assert f"accepted: {g1.bdf}: performance level 'manual'" in out
    assert "the run's hold is lifted" in out
    doc = guard.read()
    assert doc["hold"] is None and g1.bdf in doc["released"]
    assert [e["shape"] for e in doc[gpu_health.RELEASED_LINES]] == [[1, 2, 3, 1]]
    assert check() == "" and guard.quarantined_shapes() == set()
    assert run_guard.main(["status", str(run)]) == 0
    assert "1 line(s) released (benched again)" in capsys.readouterr().out
    before = guard.read()["released"]
    time.sleep(1.1)
    assert run_guard.main(["accept", str(run)]) == 0
    after = guard.read()["released"]
    assert set(after) >= {g0.bdf, g1.bdf} and after[g1.bdf] > before[g1.bdf]


@pytest.mark.parametrize("setting", ["perf_level", "power_cap"])
def test_a_bench_stage_starts_beside_a_gpu_held_for_a_setting_the_gpus_share(
    tmp_path, gpus, monkeypatch, setting
):
    """The orchestrator starts stage 3 beside a GPU the run holds alone, and
    stage 3 leaves that GPU out of its platform check and benches on the
    other one, whatever setting of the check the held GPU departs in."""
    import json

    import bench_fakes as bf
    from lib import fingerprint

    monkeypatch.setattr(gpu_health, "CONFIRM_S", 0.01)
    bench = bf.write_fake_bench(tmp_path)
    (tmp_path / "state").mkdir()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"state_dir": str(tmp_path / "state")}))
    monkeypatch.setenv("FAKE_BENCH_PLAN", str(plan))
    health = {"poll_s": 1.0, "max_wait_s": 60.0}
    path = make_config(tmp_path, **{"bench.health": health})
    p = _pipeline(tmp_path, path)
    assert p.wait_gpu_health([0, 1], "x")
    if setting == "perf_level":
        _level(gpus, 1).write_text("manual\n")
    else:
        cap = sorted(_level(gpus, 1).parent.glob("hwmon/hwmon*/power1_cap"))[0]
        cap.write_text(str(int(cap.read_text()) - 50_000_000) + "\n")
    assert p.wait_gpu_health([0, 1], "x", per_gpu=True)
    gemms = [bf.Gemm(64 + 8 * i, 32 + 4 * i, 128 + 16 * i) for i in range(6)]
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "shapes.yaml").write_text("".join(bf.bench_line(g) for g in gemms))
    cfg = tmp_path / "stage_config.yaml"
    cfg.write_text(
        yaml.safe_dump({"bench": {"quiet_host": {"enabled": False}, "health": health}})
    )
    out = tmp_path / "round_0" / "stage03"
    args = ["--in-dir", str(in_dir), "--out-dir", str(out)]
    args += ["--bench-binary", str(bench), "--devices", "0,1", "--no-warmup-pass"]
    args += ["--quiet", "--config-yaml", str(cfg), "--no-health-canary"]
    rc, cause = p._stage_attempt(tmp_path / "round_0", "stage03", args, 0, None, 1)
    summary = json.loads((out / "stage03_summary.json").read_text())
    assert (rc, cause) == (0, "") and summary["problems"] == []
    assert summary["benched"] == len(gemms)
    assert {b["device"] for b in summary["blocks"]} == {0}
    platform = json.loads((out / fingerprint.PLATFORM_FILE).read_text())
    rows = {r["device"]: r for r in platform["gpus"]}
    assert platform["problems"] == [] and "held" not in rows[0]
    assert rows[1]["held"].startswith("GPU 1 (0000:0a:00.0): ")
    held = gpu_health.Guard(p.guard_dir, "x").hold()
    assert set(held["entries"]) == {"0000:0a:00.0"}


def test_the_script_runs_on_its_own(tmp_path):
    run = tmp_path / "run"
    gpu_health.Guard(run / gpu_health.GUARD_DIR, "s").quarantine_lines(
        [{"shape": [1, 1, 1, 1]}]
    )
    script = Path(run_guard.__file__)
    done = subprocess.run(
        [sys.executable, str(script), "status", str(run), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert yaml.safe_load(done.stdout)["quarantined_lines"][0]["shape"] == [1, 1, 1, 1]
