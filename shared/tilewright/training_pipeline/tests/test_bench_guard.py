# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage03 under the run's guard (lib/host_load.py, lib/gpu_health.py): quiet
cores, GPU health and its canary, hung lines and excluded GPUs, with the
fake bench, a fake `/proc/stat` and a fake sysfs."""

import argparse
import json
import os
import sys
import threading
import time
import types

import pytest
import yaml

import gpu_fakes as gf
import stage03_load_balance_offline_tuning as s03
from bench_fakes import (
    LIBRARY_STEM,
    Gemm,
    bench_line,
    solution,
    write_fake_bench,
    write_logic,
)
from lib import fingerprint, gpu_health, kfd
from lib import kernel_identity as ki
from lib.dat import kernel_dat_info

GEMMS = [Gemm(64 + 8 * i, 32 + 4 * i, 128 + 16 * i) for i in range(10)]


def _key(g):
    return f"{g.m}x{g.n}x{g.k}x{g.batch}"


@pytest.fixture
def plan(tmp_path, monkeypatch):
    """Writes the fake bench's plan (read by every bench process when it
    starts) and returns the bench."""
    bench = write_fake_bench(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    path = tmp_path / "plan.json"

    def write(**kw):
        kw.setdefault("state_dir", str(state))
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(kw))
        tmp.replace(path)
        monkeypatch.setenv("FAKE_BENCH_PLAN", str(path))
        return bench

    return write


@pytest.fixture
def gpus(tmp_path, monkeypatch):
    root = gf.write_sysfs(tmp_path / "sys")
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    for var in kfd.HIP_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(kfd.ROCR_VAR, "0,1")
    return root


@pytest.fixture
def fast(monkeypatch):
    """The watchdog polls and the health check's confirmation, short."""
    monkeypatch.setattr(kfd, "WATCH_POLL_S", 0.1)
    monkeypatch.setattr(gpu_health, "CONFIRM_S", 0.05)
    monkeypatch.setattr(s03, "HOLD_READ_S", 0.05)


def _config(tmp_path, **bench):
    cfg = {"bench": bench}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _identity(tmp_path):
    d = tmp_path / "library"
    if not d.is_dir():
        d.mkdir()
        write_logic(d / f"{LIBRARY_STEM}.dat", [solution(100 + i, i) for i in range(3)])
    return ["--identity-dir", d, "--identity-stem", LIBRARY_STEM]


def _stage03(monkeypatch, tmp_path, bench, *extra, lines=None, out="stage03"):
    in_dir = tmp_path / "in"
    in_dir.mkdir(exist_ok=True)
    text = lines if lines is not None else [bench_line(g) for g in GEMMS]
    (in_dir / "shapes.yaml").write_text("".join(text))
    out_dir = tmp_path / out
    argv = ["stage03", "--in-dir", in_dir, "--out-dir", out_dir, "--bench-binary"]
    argv += [bench, "--devices", "0,1", "--no-warmup-pass", "--quiet"]
    monkeypatch.setattr(sys, "argv", [str(a) for a in argv + list(extra)])
    rc = s03.main()
    summary = json.loads((out_dir / "stage03_summary.json").read_text())
    return rc, out_dir, summary


@pytest.fixture
def stat(tmp_path, monkeypatch):
    root = tmp_path / "proc"
    root.mkdir()
    monkeypatch.setenv("TILEWRIGHT_PROC_ROOT", str(root))
    with gf.FakeStat(root, sorted(os.sched_getaffinity(0))) as fake:
        yield fake


QUIET = {"every_s": 0.2}


# ── quiet cores ──────────────────────────────────────────────────────────────


def test_blocks_wait_for_quiet_cores_and_the_load_is_recorded(
    tmp_path, monkeypatch, plan, stat
):
    stat.set(0.6)
    cfg = _config(tmp_path, quiet_host=QUIET)
    series = tmp_path / "stage03" / "host_load.jsonl"

    def windows():
        try:
            return series.read_text().count("\n")
        except OSError:
            return 0

    def calm():
        assert gf.wait_for(lambda: windows() >= 4, 30)
        stat.set(0.02)

    t = threading.Thread(target=calm, daemon=True)
    t.start()
    rc, out, summary = _stage03(
        monkeypatch, tmp_path, plan(), "--config-yaml", cfg, "--no-health"
    )
    t.join(5)
    assert rc == 0 and summary["lost"] == 0 and summary["aborted"] == []
    q = summary["guard"]["quiet_host"]
    assert q["waits"] >= 1 and q["waited_s"] >= 0.5 and q["hot"] >= 3
    assert q["settings"]["every_s"] == 0.2 and q["unmeasurable"] is None
    platform = json.loads((out / fingerprint.PLATFORM_FILE).read_text())
    assert platform["host_load"] == q and platform["health"]["enabled"] is False
    assert (out / "host_load.jsonl").read_text().count("\n") >= 3
    assert all("host_load" in b for b in summary["blocks"])


def test_a_block_under_hot_cores_is_aborted_and_benched_again(
    tmp_path, monkeypatch, plan, stat, fast
):
    cfg = _config(tmp_path, quiet_host=QUIET)
    bench = plan(hang=[_key(GEMMS[3])])
    out = tmp_path / "stage03"

    def contend():
        assert gf.wait_for(lambda: list((out / "logs").glob("block_*.log")), 30)
        stat.set(0.6)
        assert gf.wait_for(lambda: (out / s03.ABORTED_DIR).is_dir(), 30)
        plan()
        stat.set(0.02)

    t = threading.Thread(target=contend, daemon=True)
    t.start()
    rc, out, summary = _stage03(
        monkeypatch,
        tmp_path,
        bench,
        "--config-yaml",
        cfg,
        "--no-health",
        "--blocks-per-gpu",
        2,
        "--devices",
        0,
    )
    t.join(5)
    assert rc == 0 and summary["lost"] == 0 and summary["problems"] == []
    (a,) = summary["aborted"]
    assert a["reason"] == "contended"
    assert a["attempt"] == 1 and a["host_load"]["hot"] >= 2
    moved = out / s03.ABORTED_DIR / f"{a['block']}_gpu{a['device']}_a1"
    assert json.loads((moved / s03.ABORTED_RECORD).read_text())["reason"] == "contended"
    log = (moved / f"{a['block']}_gpu{a['device']}.log").read_text()
    assert "*** ABORTED [contended]: other load on the stage's cores" in log
    again = next(b for b in summary["blocks"] if b["name"] == a["block"])
    assert again["aborts"] == 1


def test_a_block_aborted_at_the_tail_waits_for_quiet_cores_on_every_device(
    tmp_path, monkeypatch, plan, stat, fast
):
    cfg = _config(tmp_path, quiet_host=QUIET)
    bench = plan(hang=[_key(GEMMS[3])])
    out = tmp_path / "stage03"

    def finished(block, n):
        logs = list((out / "logs").glob(f"{block}_*.log"))
        return logs and logs[0].read_text().count("Winner:") == n

    def contend():
        assert gf.wait_for(
            lambda: finished("block_0001", 4) and finished("block_0002", 2), 30
        )
        stat.set(0.6)
        assert gf.wait_for(lambda: (out / s03.ABORTED_DIR).is_dir(), 30)
        time.sleep(3.0)
        plan()
        stat.set(0.02)

    t = threading.Thread(target=contend, daemon=True)
    t.start()
    rc, out, summary = _stage03(
        monkeypatch,
        tmp_path,
        bench,
        "--config-yaml",
        cfg,
        "--no-health",
        "--blocks-per-gpu",
        1,
        "--devices",
        "0,1,2",
        "--stall-timeout-s",
        60,
    )
    t.join(5)
    assert rc == 0 and summary["lost"] == 0 and summary["problems"] == []
    (a,) = summary["aborted"]
    assert a["block"].startswith("block_0000_") and a["reason"] == "contended"


def test_a_worker_puts_back_a_block_its_device_cannot_start_yet(tmp_path, monkeypatch):
    blocks = s03._build_blocks([bench_line(g) for g in GEMMS[:2]], 1)
    work = s03._Work(blocks, 2)
    calls = []

    class Guard:
        def ready(self, device, stop):
            calls.append("ready")
            return "go"

        def blocked(self, device):
            calls.append("blocked")
            return "its canary first" if calls.count("blocked") == 1 else None

    worker = s03._DeviceWorker(
        0, work, tmp_path / "bench", tmp_path / "logs", gpus=kfd.StageGpus([0])
    )
    worker.guard = Guard()
    benched = []

    def bench(block):
        benched.append(block.name)
        work.finish(block)

    monkeypatch.setattr(worker, "_bench", bench)
    worker.run()
    assert calls[:4] == ["ready", "blocked", "ready", "blocked"]
    assert [b.split("_lines")[0] for b in benched] == ["block_0001", "block_0000"]


def test_a_stage_that_never_finds_its_cores_quiet_gives_up(
    tmp_path, monkeypatch, plan, stat
):
    stat.set(0.9)
    cfg = _config(tmp_path, quiet_host=dict(QUIET, max_wait_s=1.0))
    rc, out, summary = _stage03(
        monkeypatch, tmp_path, plan(), "--config-yaml", cfg, "--no-health"
    )
    assert rc == 2
    assert any("cores were not quiet" in p for p in summary["problems"])
    assert summary["guard"]["gave_up"] and summary["benched"] == 0
    assert not list((out / "logs").glob("block_*.log"))


# ── GPU health ───────────────────────────────────────────────────────────────


def _level(root, gpu):
    d = root / "class" / "drm" / f"renderD{gf.GPUS[gpu].render_minor}" / "device"
    return d / "power_dpm_force_performance_level"


def _block_log(out, block):
    """(device, text) of the first log of `block` (a name prefix)."""
    logs = sorted((out / "logs").glob(f"{block}_*.log"))
    if not logs:
        return None, ""
    return int(logs[0].stem.rsplit("_gpu", 1)[1]), logs[0].read_text()


def test_a_departure_of_one_gpu_holds_its_blocks_while_the_other_goes_on(
    tmp_path, monkeypatch, plan, gpus, fast
):
    cfg = _config(tmp_path, quiet_host={"enabled": False}, health={"poll_s": 0.2})
    bench = plan(hang=[_key(GEMMS[3])])
    out = tmp_path / "stage03"
    seen = {}

    def reset():
        assert gf.wait_for(
            lambda: _block_log(out, "block_0000")[1].count("Winner:") == 3, 30
        )
        device, _ = _block_log(out, "block_0000")
        seen["held"] = device
        plan()
        _level(gpus, device).write_text("manual\n")
        other = 1 - device
        done = out / "logs" / f"block_0000_lines1-5_gpu{other}.log"
        assert gf.wait_for(
            lambda: done.is_file() and done.read_text().count("Winner:") == 5, 30
        )
        _level(gpus, device).write_text("auto\n")

    t = threading.Thread(target=reset, daemon=True)
    t.start()
    rc, out, summary = _stage03(
        monkeypatch,
        tmp_path,
        bench,
        "--config-yaml",
        cfg,
        "--no-health-canary",
        "--blocks-per-gpu",
        1,
    )
    t.join(5)
    held = seen["held"]
    assert rc == 0 and summary["lost"] == 0 and summary["problems"] == []
    (a,) = summary["aborted"]
    assert a["reason"] == "health" and a["device"] == held
    (problem,) = a["hold"]["problems"]
    assert f"GPU {held} (" in problem and "performance level 'manual'" in problem
    assert not gpu_health.run_wide(a["hold"])
    by_block = {b["name"]: b for b in summary["blocks"]}
    first = next(b for n, b in by_block.items() if n.startswith("block_0000"))
    second = next(b for n, b in by_block.items() if n.startswith("block_0001"))
    assert first["device"] == 1 - held and first["aborts"] == 1
    assert second["aborts"] == 0
    raised = summary["guard"]["health"]["holds"][0]
    assert raised["raised"] and not raised["every_gpu"]
    assert raised["gpus"] == [gpu_health.gpu_key(kfd.gpus()[held])]


def test_a_departure_of_every_gpu_holds_every_block_until_they_are_back(
    tmp_path, monkeypatch, plan, gpus, fast
):
    cfg = _config(tmp_path, quiet_host={"enabled": False}, health={"poll_s": 0.2})
    bench = plan(hang=[_key(GEMMS[3])])
    out = tmp_path / "stage03"

    def reset():
        assert gf.wait_for(lambda: list((out / "logs").glob("block_*.log")), 30)
        for gpu in (0, 1):
            _level(gpus, gpu).write_text("manual\n")
        assert gf.wait_for(lambda: (out / s03.ABORTED_DIR).is_dir(), 30)
        plan()
        for gpu in (0, 1):
            _level(gpus, gpu).write_text("auto\n")

    t = threading.Thread(target=reset, daemon=True)
    t.start()
    rc, out, summary = _stage03(
        monkeypatch,
        tmp_path,
        bench,
        "--config-yaml",
        cfg,
        "--no-health-canary",
        "--blocks-per-gpu",
        1,
    )
    t.join(5)
    assert rc == 0 and summary["lost"] == 0 and summary["problems"] == []
    assert {a["reason"] for a in summary["aborted"]} == {"health"}
    assert any(
        "performance level 'manual'" in p
        for a in summary["aborted"]
        for p in (a.get("hold") or {}).get("problems") or []
    )
    guard = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x")
    doc = guard.read()
    assert doc["hold"] is None and len(doc["holds"]) == 1
    assert doc["holds"][0]["cleared"] and doc["baseline"]["gpus"]["0000:0a:00.0"]
    health = summary["guard"]["health"]
    assert health["holds"][0].get("raised") and health["holds"][-1].get("cleared")
    assert any(h.get("every_gpu") for h in health["holds"])
    platform = json.loads((out / fingerprint.PLATFORM_FILE).read_text())
    assert platform["health"]["holds"] == health["holds"]


def test_a_gpu_unlike_the_runs_start_holds_the_stage_until_it_gives_up(
    tmp_path, monkeypatch, plan, gpus, fast
):
    guard = gpu_health.Guard(tmp_path / "stage03" / gpu_health.GUARD_DIR, "s0")
    guard.baseline(kfd.gpus()[:2])
    for gpu in (0, 1):
        _level(gpus, gpu).write_text("high\n")
    cfg = _config(
        tmp_path,
        quiet_host={"enabled": False},
        health={"poll_s": 0.2, "max_wait_s": 1.0},
    )
    rc, out, summary = _stage03(
        monkeypatch, tmp_path, plan(), "--config-yaml", cfg, "--no-health-canary"
    )
    assert rc == 2 and summary["benched"] == 0
    assert any(
        "held the work of every GPU for longer" in p for p in summary["problems"]
    )
    assert "performance level 'high'" in guard.hold()["problems"][0]
    assert gpu_health.run_wide(guard.hold())


def test_a_gpu_held_longer_than_the_limit_is_left_out_and_the_other_benches(
    tmp_path, monkeypatch, plan, gpus, fast
):
    for gpu in (0, 1):
        _pm(gpus, gpu)
    guard = gpu_health.Guard(tmp_path / "stage03" / gpu_health.GUARD_DIR, "s0")
    guard.baseline(kfd.gpus()[:2])
    _pm(gpus, 0, "disabled")
    cfg = _config(
        tmp_path,
        quiet_host={"enabled": False},
        health={"poll_s": 0.2, "max_wait_s": 1.0},
    )
    many = [Gemm(64 + 8 * i, 32 + 4 * i, 128 + 16 * i) for i in range(40)]
    rc, out, summary = _stage03(
        monkeypatch,
        tmp_path,
        plan(),
        "--config-yaml",
        cfg,
        "--no-health-canary",
        "--blocks-per-gpu",
        20,
        lines=[bench_line(g) for g in many],
    )
    assert rc == 0 and summary["lost"] == 0 and summary["problems"] == []
    assert summary["benched"] == len(many)
    assert {b["device"] for b in summary["blocks"]} == {1}
    (gone,) = summary["retired_devices"]
    assert gone["device"] == 0
    assert gone["why"].startswith("the run held its GPU for longer than")
    assert "power management is off" in gone["why"]
    assert set(guard.hold()["entries"]) == {gpu_health.gpu_key(kfd.gpus()[0])}


def _pm(root, gpu, state="enabled"):
    d = root / "class" / "drm" / f"renderD{gf.GPUS[gpu].render_minor}" / "device"
    (d / "pp_features").write_text(f"00. DPM ( 0) : {state}\n")


def test_a_stage_on_one_gpu_benches_beside_the_runs_hold_of_another(
    tmp_path, monkeypatch, plan, gpus, fast
):
    for gpu in (0, 1):
        _pm(gpus, gpu)
    guard = gpu_health.Guard(tmp_path / "stage03" / gpu_health.GUARD_DIR, "a", by="a")
    g0, g1, _ = kfd.gpus()
    guard.baseline([g0, g1])
    _pm(gpus, 0, "disabled")
    quick = dict(ttm=lambda: [], sleep=lambda _s: None)
    assert guard.settle(gpu_health.check(guard, [g0], **quick))[0] == "raised"
    cfg = _config(
        tmp_path,
        quiet_host={"enabled": False},
        health={"poll_s": 0.2, "max_wait_s": 1.5},
    )
    args = ("--config-yaml", cfg, "--no-health-canary", "--devices", 1)
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert rc == 0 and summary["lost"] == 0 and summary["benched"] == len(GEMMS)
    assert set(guard.hold()["entries"]) == {g0.bdf}
    assert not gpu_health.run_wide(guard.hold())
    assert summary["guard"]["health"]["gpus"] == [g1.bdf]
    _pm(gpus, 0)
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert rc == 0 and summary["lost"] == 0 and guard.hold() is None


def _depart(root, gpu, setting, back=False):
    """Move GPU `gpu`'s performance level or power cap off the fake sysfs's
    value (`back`: to it again)."""
    if setting == "perf_level":
        _level(root, gpu).write_text("auto\n" if back else "manual\n")
        return
    cap = sorted(_level(root, gpu).parent.glob("hwmon/hwmon*/power1_cap"))[0]
    step = 50_000_000 if back else -50_000_000
    cap.write_text(f"{int(cap.read_text()) + step}\n")


def _hold_alone(tmp_path, gpus, gpu, setting):
    """The run's guard of the stage, holding GPU `gpu` alone for `setting`."""
    guard = gpu_health.Guard(tmp_path / "stage03" / gpu_health.GUARD_DIR, "s0")
    g0, g1, _ = kfd.gpus()
    guard.baseline([g0, g1])
    _depart(gpus, gpu, setting)
    quick = dict(ttm=lambda: [], sleep=lambda _s: None)
    assert guard.settle(gpu_health.check(guard, [g0, g1], **quick))[0] == "raised"
    assert not gpu_health.run_wide(guard.hold())
    return guard


MANY = [Gemm(64 + 8 * i, 32 + 4 * i, 128 + 16 * i) for i in range(60)]


@pytest.mark.parametrize("setting", ["perf_level", "power_cap"])
def test_a_stage_starts_beside_a_gpu_held_for_a_setting_its_gpus_must_share(
    tmp_path, monkeypatch, plan, gpus, fast, setting
):
    _hold_alone(tmp_path, gpus, 1, setting)
    cfg = _config(tmp_path, quiet_host={"enabled": False}, health={"poll_s": 0.2})
    rc, out, summary = _stage03(
        monkeypatch, tmp_path, plan(), "--config-yaml", cfg, "--no-health-canary"
    )
    assert rc == 0 and summary["problems"] == [] and summary["benched"] == len(GEMMS)
    assert {b["device"] for b in summary["blocks"]} == {0}
    platform = json.loads((out / fingerprint.PLATFORM_FILE).read_text())
    rows = {r["device"]: r for r in platform["gpus"]}
    assert platform["problems"] == [] and "held" not in rows[0]
    assert rows[1]["held"].startswith(f"GPU 1 ({kfd.gpus()[1].bdf}): ")
    seen = summary["guard"]["health"]["platform"]
    assert [h["device"] for h in seen["held_at_start"]] == [1]
    assert seen["joined"] == [] and seen["unlike"] == []


def test_a_gpu_held_when_the_stage_starts_benches_once_it_is_back(
    tmp_path, monkeypatch, plan, gpus, fast
):
    guard = _hold_alone(tmp_path, gpus, 1, "perf_level")
    cfg = _config(tmp_path, quiet_host={"enabled": False}, health={"poll_s": 0.2})
    out = tmp_path / "stage03"

    def back():
        assert gf.wait_for((out / fingerprint.PLATFORM_FILE).is_file, 30)
        _depart(gpus, 1, "perf_level", back=True)

    t = threading.Thread(target=back, daemon=True)
    t.start()
    rc, out, summary = _stage03(
        monkeypatch,
        tmp_path,
        plan(),
        "--config-yaml",
        cfg,
        "--no-health-canary",
        "--blocks-per-gpu",
        30,
        lines=[bench_line(g) for g in MANY],
    )
    t.join(5)
    assert rc == 0 and summary["problems"] == [] and summary["benched"] == len(MANY)
    assert {b["device"] for b in summary["blocks"]} == {0, 1}
    seen = summary["guard"]["health"]["platform"]
    assert [j["device"] for j in seen["joined"]] == [1] and seen["unlike"] == []
    assert guard.hold() is None


def test_a_gpu_accepted_unlike_the_others_takes_no_block(
    tmp_path, monkeypatch, plan, gpus, fast
):
    import run_guard

    guard = _hold_alone(tmp_path, gpus, 1, "perf_level")
    cfg = _config(tmp_path, quiet_host={"enabled": False}, health={"poll_s": 0.2})
    out = tmp_path / "stage03"
    key = gpu_health.gpu_key(kfd.gpus()[1])

    def accept():
        assert gf.wait_for((out / fingerprint.PLATFORM_FILE).is_file, 30)
        assert run_guard.main(["accept", str(guard.dir), "--gpu", key]) == 0

    t = threading.Thread(target=accept, daemon=True)
    t.start()
    rc, out, summary = _stage03(
        monkeypatch,
        tmp_path,
        plan(),
        "--config-yaml",
        cfg,
        "--no-health-canary",
        "--blocks-per-gpu",
        30,
        lines=[bench_line(g) for g in MANY],
    )
    t.join(5)
    assert rc == 0 and summary["problems"] == [] and summary["benched"] == len(MANY)
    assert {b["device"] for b in summary["blocks"]} == {0}
    seen = summary["guard"]["health"]["platform"]
    (unlike,) = seen["unlike"]
    assert unlike["device"] == 1 and unlike["gpu"] == key
    assert "disagree on the performance level" in unlike["why"]
    assert seen["joined"] == [] and guard.hold() is None


def test_with_leave_held_gpus_a_gpu_held_alone_is_left_out_at_once(
    tmp_path, monkeypatch, plan, gpus, fast
):
    for gpu in (0, 1):
        _pm(gpus, gpu)
    guard = gpu_health.Guard(tmp_path / "stage03" / gpu_health.GUARD_DIR, "s0")
    g0, g1, _ = kfd.gpus()
    guard.baseline([g0, g1])
    _pm(gpus, 0, "disabled")
    quick = dict(ttm=lambda: [], sleep=lambda _s: None)
    assert guard.settle(gpu_health.check(guard, [g0, g1], **quick))[0] == "raised"
    cfg = _config(tmp_path, quiet_host={"enabled": False}, health={"poll_s": 0.2})
    args = ("--config-yaml", cfg, "--no-health-canary", "--leave-held-gpus")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert rc == 0 and summary["problems"] == [] and summary["benched"] == len(GEMMS)
    assert {b["device"] for b in summary["blocks"]} == {1}
    (gone,) = summary["retired_devices"]
    assert gone["device"] == 0 and gone["why"].startswith("the run holds its GPU (")
    assert gone["why"].endswith("and --leave-held-gpus leaves it out at once")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args, "--devices", 0)
    assert rc == 2 and summary["benched"] == 0
    assert not list((out / "logs").glob("block_*.log"))
    assert "every GPU stopped taking blocks" in summary["problems"][0]


def test_the_health_canary_follows_the_health_checks():
    def ns(**kw):
        base = {"quiet_host": None, "health": None, "health_canary": None}
        return argparse.Namespace(**{**base, **kw})

    off = {"bench": {"health": {"enabled": False}}}
    assert s03._guard_settings(ns(), {})[2] is True
    assert s03._guard_settings(ns(health=False), {})[2] is False
    assert s03._guard_settings(ns(), off)[2] is False
    assert s03._guard_settings(ns(health=False, health_canary=True), {})[2] is True
    no_canary = {"bench": {"health": {"canary": {"enabled": False}}}}
    assert s03._guard_settings(ns(), no_canary)[2] is False


# ── health canary ────────────────────────────────────────────────────────────


def test_the_canary_takes_a_reference_and_judges_later_sessions_by_it(
    tmp_path, monkeypatch, plan, gpus, fast
):
    cfg = _config(
        tmp_path,
        quiet_host={"enabled": False},
        health={
            "poll_s": 0.2,
            "max_wait_s": 2.0,
            "canary": {"retry_s": 0.3, "timings": 3},
        },
    )
    args = ("--config-yaml", cfg, *_identity(tmp_path))
    monkeypatch.setenv(gpu_health.SESSION_VAR, "first")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert rc == 0 and summary["lost"] == 0
    read = summary["guard"]["health"]["canary"]["read"]
    assert sorted(r["key"] for r in read) == ["0000:0a:00.0", "0000:0b:00.0"]
    assert all(r["reference"] and r["passed"] for r in read)
    guard = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x")
    setup = guard.canary_setup()
    assert len(setup["lines"]) == 4 and setup["timings"] == 3
    assert all("solution_index_list: [100]" in ln for ln in setup["lines"])
    logs = sorted((out / gpu_health.GUARD_DIR / "canary").glob("*.log"))
    assert len(logs) == 2

    monkeypatch.setenv(gpu_health.SESSION_VAR, "second")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    read = summary["guard"]["health"]["canary"]["read"]
    assert rc == 0 and len(read) == 2
    assert all(r["passed"] and not r["reference"] for r in read)
    assert all(r["ratio"] == pytest.approx(1.0) for r in read)

    monkeypatch.setenv(gpu_health.SESSION_VAR, "same")
    rc, _, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert len(summary["guard"]["health"]["canary"]["read"]) == 2
    rc, _, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert rc == 0 and summary["guard"]["health"]["canary"]["read"] == []

    monkeypatch.setenv(gpu_health.SESSION_VAR, "slow")
    slow = plan(us={_key(g): 13.0 for g in GEMMS})
    rc, out, summary = _stage03(monkeypatch, tmp_path, slow, *args)
    assert rc == 2 and summary["benched"] == 0
    read = summary["guard"]["health"]["canary"]["read"]
    assert read and not any(r["passed"] for r in read)
    assert all(r["ratio"] == pytest.approx(1.3) for r in read)
    assert any("failed its canary for longer" in p for p in summary["problems"])
    checks = guard.read()["canary"]["checks"]
    assert [c["passed"] for c in checks[:4]] == [True] * 4


def test_a_canary_of_one_line_takes_a_reference_and_judges_by_it(
    tmp_path, monkeypatch, plan, gpus, fast
):
    cfg = _config(
        tmp_path,
        quiet_host={"enabled": False},
        health={"poll_s": 0.2, "max_wait_s": 5.0, "canary": {"lines": 1}},
    )
    args = ("--config-yaml", cfg, *_identity(tmp_path))
    monkeypatch.setenv(gpu_health.SESSION_VAR, "first")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert rc == 0 and summary["lost"] == 0, summary["problems"]
    (line,) = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").canary_setup()["lines"]
    assert f"M: {GEMMS[2].m}," in line
    monkeypatch.setenv(gpu_health.SESSION_VAR, "second")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    read = summary["guard"]["health"]["canary"]["read"]
    assert rc == 0 and len(read) == 2
    assert all(r["judged"] and r["passed"] and r["lines_read"] == 1 for r in read)


def test_a_canary_that_cannot_be_read_takes_no_reference_and_holds_nothing(
    tmp_path, monkeypatch, plan, gpus, fast
):
    cfg = _config(tmp_path, quiet_host={"enabled": False}, health={"poll_s": 0.2})
    args = ("--config-yaml", cfg, *_identity(tmp_path))
    monkeypatch.setenv(gpu_health.SESSION_VAR, "crashing")
    rc, out, summary = _stage03(
        monkeypatch, tmp_path, plan(crash_text="hcanary"), *args
    )
    assert rc == 0 and summary["lost"] == 0
    read = summary["guard"]["health"]["canary"]["read"]
    assert len(read) == 2
    assert all(r["passed"] and not r["reference"] and not r["judged"] for r in read)
    assert all(r["why"].startswith("no reference taken (0 of 4") for r in read)
    assert all(r["unread"] for r in read)
    guard = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x")
    assert guard.read()["canary"].get("references", {}) == {}
    monkeypatch.setenv(gpu_health.SESSION_VAR, "fine")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    read = summary["guard"]["health"]["canary"]["read"]
    assert rc == 0 and all(r["reference"] for r in read)
    assert set(guard.read()["canary"]["references"]) == {"0000:0a:00.0", "0000:0b:00.0"}


def test_a_later_canary_that_cannot_be_read_gives_no_verdict_and_holds_nothing(
    tmp_path, monkeypatch, plan, gpus, fast
):
    cfg = _config(
        tmp_path,
        quiet_host={"enabled": False},
        health={"poll_s": 0.2, "max_wait_s": 30.0, "canary": {"retry_s": 60}},
    )
    args = ("--config-yaml", cfg, *_identity(tmp_path))
    monkeypatch.setenv(gpu_health.SESSION_VAR, "first")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert rc == 0
    guard = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x")
    refs = guard.read()["canary"]["references"]
    monkeypatch.setenv(gpu_health.SESSION_VAR, "second")
    rc, out, summary = _stage03(
        monkeypatch, tmp_path, plan(crash_text="hcanary"), *args
    )
    assert rc == 0 and summary["lost"] == 0 and summary["problems"] == []
    assert summary["benched"] == len(GEMMS)
    read = summary["guard"]["health"]["canary"]["read"]
    assert len(read) == 2
    for r in read:
        assert r["unread"] and r["passed"] and not r["judged"] and not r["reference"]
        assert r["why"].startswith("no verdict (0 of 4 canary lines read, bench rc")
    assert guard.read()["canary"]["references"] == refs
    monkeypatch.setenv(gpu_health.SESSION_VAR, "third")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    read = summary["guard"]["health"]["canary"]["read"]
    assert rc == 0 and len(read) == 2
    assert all(r["judged"] and r["passed"] and "unread" not in r for r in read)


def test_a_gpu_whose_canary_keeps_failing_is_left_out_and_the_other_finishes(
    tmp_path, monkeypatch, plan, gpus, fast
):
    many = [Gemm(64 + 8 * i, 32 + 4 * i, 128 + 16 * i) for i in range(40)]
    lines = [bench_line(g) for g in many]
    cfg = _config(
        tmp_path,
        quiet_host={"enabled": False},
        health={"poll_s": 0.2, "max_wait_s": 1.0, "canary": {"retry_s": 60}},
    )
    args = ("--config-yaml", cfg, *_identity(tmp_path), "--blocks-per-gpu", 20)
    monkeypatch.setenv(gpu_health.SESSION_VAR, "first")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args, lines=lines)
    assert rc == 0
    monkeypatch.setenv(gpu_health.SESSION_VAR, "second")
    bench = plan(slow_gpu={"0": 2.0})
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args, lines=lines)
    assert rc == 0 and summary["problems"] == [] and summary["lost"] == 0
    assert summary["benched"] == len(many) and summary["guard"]["gave_up"] is None
    assert {b["device"] for b in summary["blocks"]} == {1}
    (gone,) = summary["retired_devices"]
    assert gone == {
        "device": 0,
        "why": "failed its canary for longer than bench.health.max_wait_s 1 s",
    }


def test_the_retry_pass_passes_over_a_gpu_whose_canary_fails(
    tmp_path, monkeypatch, plan, gpus, fast
):
    cfg = _config(
        tmp_path,
        quiet_host={"enabled": False},
        health={
            "poll_s": 0.2,
            "max_wait_s": 20.0,
            "canary": {"retry_s": 0.3, "timings": 3},
        },
    )
    args = ("--config-yaml", cfg, *_identity(tmp_path))
    monkeypatch.setenv(gpu_health.SESSION_VAR, "first")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert rc == 0
    monkeypatch.setenv(gpu_health.SESSION_VAR, "second")
    bench = plan(crash_once=[_key(GEMMS[9])], slow_gpu={"0": 2.0})
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert rc == 0 and summary["problems"] == []
    assert (summary["retried"], summary["recovered"], summary["lost"]) == (1, 1, 0)
    retry = next(b for b in summary["blocks"] if b["name"] == "block_retry")
    assert retry["device"] == 1
    read = summary["guard"]["health"]["canary"]["read"]
    assert any(r["device"] == 0 and r["judged"] and not r["passed"] for r in read)
    assert all(b["device"] == 1 for b in summary["blocks"])


def test_a_canary_line_follows_its_kernel_to_its_new_solution_index(
    tmp_path, monkeypatch, plan, gpus, fast
):
    lines = [bench_line(g, solution_index_list="[101, 102]") for g in GEMMS]
    cfg = _config(tmp_path, quiet_host={"enabled": False}, health={"poll_s": 0.2})
    library = tmp_path / "library"
    library.mkdir()
    logic = library / f"{LIBRARY_STEM}.dat"
    write_logic(logic, [solution(100 + i, i, (64 + 64 * i, 128, 64)) for i in range(3)])
    args = ("--config-yaml", cfg, "--identity-dir", library)
    args += ("--identity-stem", LIBRARY_STEM)
    monkeypatch.setenv(gpu_health.SESSION_VAR, "first")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args, lines=lines)
    assert rc == 0
    guard = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x")
    setup = guard.canary_setup()
    assert all("solution_index_list: [101]" in ln for ln in setup["lines"])
    assert {k["signature"] for k in setup["kernels"]} == {
        "MT128x128x64 MI16x16x128 NTA0 NTB0"
    }
    moved = [solution(100 + i, i, (64 + 64 * i, 128, 64)) for i in (0, 2)]
    moved.append(solution(105, 3, (128, 128, 64)))
    write_logic(logic, moved)
    monkeypatch.setenv(gpu_health.SESSION_VAR, "second")
    yaml_dir = tmp_path / "yamls"
    yaml_dir.mkdir()
    rc, out, summary = _stage03(
        monkeypatch, tmp_path, plan(yaml_dir=str(yaml_dir)), *args, lines=lines
    )
    read = summary["guard"]["health"]["canary"]["read"]
    assert rc == 0 and len(read) == 2 and all(r["passed"] for r in read)
    assert all(r["lines_changed"][0].endswith("index 101 is 105 now") for r in read)
    canary_yamls = [
        p.read_text() for p in yaml_dir.glob("*.yaml") if "hcanary" in p.read_text()
    ]
    assert canary_yamls and all(
        "solution_index_list: [105]" in t and "[101]" not in t for t in canary_yamls
    )


def test_the_kernel_in_flight_follows_the_indices_the_log_reported():
    from lib.bench_log import Problem, SkipRow, TestedRow

    listed = [100, 100, 101, 102, 103]
    line = bench_line(GEMMS[0], solution_index_list="[100, 100, 101, 102, 103]")
    p = Problem(line_no=1)
    p.sacrificial = TestedRow(rank=0, fields={}, sol_index=100)
    p.tested = [TestedRow(rank=1, fields={}, sol_index=100)]
    p.skipped = [SkipRow(rank=2, warm_up_us=1.0, best_warm_up_us=1.0, sol_index=102)]
    assert s03._next_listed(listed, p) == 4
    guard = types.SimpleNamespace(kernel_label=lambda sid: f"K{sid}")
    got = s03._in_flight(line, p, guard)
    assert got["kernel"] == "K103" and got["rank"] == 3
    p.skipped[0].sol_index = None
    assert s03._next_listed(listed, p) == 3
    assert s03._next_listed(listed, None) == 0
    p.skipped = []
    p.tested[0].sol_index = 999
    assert s03._next_listed(listed, p) == 2


# ── hung lines ───────────────────────────────────────────────────────────────


def test_a_hung_line_is_quarantined_in_the_run_and_left_out_after(
    tmp_path, monkeypatch, plan
):
    lines = [bench_line(g, solution_index_list="[101, 102]") for g in GEMMS]
    hung = GEMMS[3]
    bench = plan(hang=[_key(hung)])
    args = ("--stall-timeout-s", 1, "--startup-grace-s", 1, *_identity(tmp_path))
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args, lines=lines)
    assert rc == 0 and summary["problems"] == []
    assert (summary["hung"], summary["lost"], summary["retried"]) == (1, 1, 0)
    rows = (out / "skipped_gemms.txt").read_text().splitlines()
    assert rows[1:] == [f"4,{hung.m},{hung.n},{hung.k},1,,hung"]
    (rec,) = summary["guard"]["hung_lines"]
    label = ki.signature_label(kernel_dat_info(solution(101, 1)))
    assert rec["shape"] == [hung.m, hung.n, hung.k, 1] and rec["yaml_line"] == 4
    assert rec["kernel"] == label and rec["in_flight"] == f"kernel {label}"
    log = next(
        p.read_text()
        for p in (out / "logs").glob("block_*.log")
        if "QUARANTINED HUNG LINE" in p.read_text()
    )
    assert f"(yaml line 4): M={hung.m} N={hung.n} K={hung.k} B=1, no output" in log
    guard = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x")
    assert guard.quarantined_shapes() == {(hung.m, hung.n, hung.k, 1)}

    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args, lines=lines)
    assert rc == 0 and summary["hung"] == 0
    assert summary["quarantined_lines_left_out"] == 1 and summary["lost"] == 1
    rows = (out / "skipped_gemms.txt").read_text().splitlines()
    assert rows[1:] == [f"4,{hung.m},{hung.n},{hung.k},1,,quarantined_line"]
    assert summary["benched"] == len(GEMMS) - 1


def test_a_stage_of_quarantined_lines_alone_records_them_left_out(
    tmp_path, monkeypatch, plan
):
    hung = GEMMS[3]
    bench = plan(hang=[_key(hung)])
    args = ("--stall-timeout-s", 1, "--startup-grace-s", 1)
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert rc == 0 and summary["hung"] == 1
    rc, out, summary = _stage03(
        monkeypatch, tmp_path, bench, *args, lines=[bench_line(hung)]
    )
    assert rc == 1 and summary["benched"] == 0 and summary["rc"] == 1
    assert summary["quarantined_lines_left_out"] == 1 and summary["lost"] == 1
    (left,) = summary["guard"]["quarantined_lines_left_out"]
    assert left == {"yaml_line": 1, "shape": [hung.m, hung.n, hung.k, 1]}
    assert summary["problems"] == [
        f"every line of {tmp_path / 'in' / 'shapes.yaml'} is quarantined in the "
        f"run (bench)"
    ]
    rows = (out / "skipped_gemms.txt").read_text().splitlines()
    assert rows[1:] == [f"1,{hung.m},{hung.n},{hung.k},1,,quarantined_line"]
    assert not list((out / "logs").glob("block_*.log"))


def test_a_hung_line_is_quarantined_only_for_the_protocol_it_hung_in(
    tmp_path, monkeypatch, plan
):
    hung = GEMMS[3]
    args = ("--stall-timeout-s", 1, "--startup-grace-s", 1)
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(hang=[_key(hung)]), *args)
    assert rc == 0 and summary["hung"] == 1
    (rec,) = summary["guard"]["hung_lines"]
    assert rec["scope"] == gpu_health.BENCH_SCOPE and rec["quarantined"]
    audit = (*args, "--hung-scope", "audit_v2")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *audit)
    assert rc == 0 and summary["lost"] == 0 and summary["benched"] == len(GEMMS)
    assert summary["quarantined_lines_left_out"] == 0
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(hang=[_key(hung)]), *audit)
    assert rc == 0 and summary["hung"] == 1
    guard = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x")
    shape = (hung.m, hung.n, hung.k, 1)
    assert guard.quarantined_shapes() == {shape}
    assert guard.quarantined_shapes("audit_v2") == {shape}
    assert guard.quarantined_shapes("audit_sustained") == set()
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *audit)
    assert summary["quarantined_lines_left_out"] == 1 and summary["lost"] == 1
    none = (*args, "--hung-scope", gpu_health.NO_SCOPE)
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *none)
    assert summary["quarantined_lines_left_out"] == 0 and summary["lost"] == 0


def test_with_no_scope_a_hung_line_only_leaves_its_block(tmp_path, monkeypatch, plan):
    hung = GEMMS[3]
    args = ("--stall-timeout-s", 1, "--startup-grace-s", 1, "--hung-scope", "none")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(hang=[_key(hung)]), *args)
    assert rc == 0 and (summary["hung"], summary["lost"]) == (1, 1)
    (rec,) = summary["guard"]["hung_lines"]
    assert rec["quarantined"] is False and rec["scope"] == gpu_health.NO_SCOPE
    guard = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x")
    assert guard.read().get("quarantined_lines") is None
    log = "".join(p.read_text() for p in (out / "logs").glob("block_*.log"))
    assert "*** HUNG LINE at block-offset" in log
    assert "not benched again in this stage" in log


def test_with_retry_hung_a_line_that_hangs_once_is_benched_in_the_retry_pass(
    tmp_path, monkeypatch, plan
):
    hung = GEMMS[3]
    args = ("--stall-timeout-s", 1, "--startup-grace-s", 1)
    args += ("--hung-scope", "audit_v2", "--retry-hung")
    rc, out, summary = _stage03(
        monkeypatch, tmp_path, plan(hang_once=[_key(hung)]), *args
    )
    assert rc == 0 and summary["problems"] == [] and summary["lost"] == 0
    assert (summary["hung"], summary["hung_recovered"]) == (0, 1)
    assert summary["retried"] == 1 and summary["recovered"] == 1
    (rec,) = summary["guard"]["hung_lines"]
    assert rec["retry"] and rec["retried"] == "benched" and not rec["quarantined"]
    guard = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x")
    assert guard.quarantined_shapes("audit_v2") == set()
    (retry_log,) = (out / "logs").glob("block_retry_gpu*.log")
    assert retry_log.read_text().count("Winner:") == 1
    log = "".join(p.read_text() for p in (out / "logs").glob("block_0*.log"))
    assert "*** HUNG LINE at block-offset" in log
    assert "benched once more in the retry pass" in log


def test_with_retry_hung_a_line_that_hangs_again_is_quarantined(
    tmp_path, monkeypatch, plan
):
    hung = GEMMS[3]
    args = ("--stall-timeout-s", 1, "--startup-grace-s", 1)
    args += ("--hung-scope", "audit_v2", "--retry-hung")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(hang=[_key(hung)]), *args)
    assert rc == 0 and summary["problems"] == []
    assert (summary["hung"], summary["hung_recovered"], summary["lost"]) == (1, 0, 1)
    rows = (out / "skipped_gemms.txt").read_text().splitlines()
    assert rows[1:] == [f"4,{hung.m},{hung.n},{hung.k},1,,hung"]
    first, again = summary["guard"]["hung_lines"]
    assert first["retry"] and first["retried"] == "hung again"
    assert not first["quarantined"] and again["quarantined"]
    assert again["block"] == "block_retry" and again["scope"] == "audit_v2"
    guard = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x")
    assert guard.quarantined_shapes("audit_v2") == {(hung.m, hung.n, hung.k, 1)}
    assert guard.quarantined_shapes() == set()


def test_with_retry_hung_a_line_the_retry_pass_does_not_bench_is_quarantined(
    tmp_path, monkeypatch, plan
):
    hung = GEMMS[3]
    args = ("--stall-timeout-s", 1, "--startup-grace-s", 1)
    args += ("--hung-scope", "audit_v2", "--retry-hung", "--max-retry-gemms", 0)
    rc, out, summary = _stage03(
        monkeypatch, tmp_path, plan(hang_once=[_key(hung)]), *args
    )
    assert rc == 0 and summary["retried"] == 0 and summary["lost"] == 1
    (rec,) = summary["guard"]["hung_lines"]
    assert rec["retried"] == "not benched to the end" and rec["quarantined"]
    guard = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x")
    assert guard.quarantined_shapes("audit_v2") == {(hung.m, hung.n, hung.k, 1)}


# ── excluded and leased GPUs ─────────────────────────────────────────────────


@pytest.mark.parametrize("ref", [1, "1", "0000:0a:00.0", "0a:00.0"])
def test_gpus_the_config_excludes_take_no_block(tmp_path, monkeypatch, plan, gpus, ref):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"runtime": {"exclude_gpus": [ref]}}))
    rc, out, summary = _stage03(
        monkeypatch, tmp_path, plan(), "--config-yaml", cfg, "--no-health-canary"
    )
    assert rc == 0 and summary["devices"] == [0]
    assert {b["device"] for b in summary["blocks"]} == {0}
