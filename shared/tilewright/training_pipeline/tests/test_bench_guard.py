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
from lib.bench_log import parse_log_file
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
    monkeypatch.setenv(gpu_health.SESSION_VAR, "unread")
    rc, out, summary = _stage03(
        monkeypatch, tmp_path, plan(silent_text="hcanary"), *args
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
        monkeypatch, tmp_path, plan(silent_text="hcanary"), *args
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


@pytest.mark.parametrize(
    "fault, why",
    [
        ({"crash_text": "hcanary"}, "its bench reported a fatal GPU error"),
        ({"hang_silent_text": "hcanary"}, "its bench printed nothing"),
    ],
    ids=["fault", "silent"],
)
def test_a_gpu_whose_canary_bench_faults_or_prints_nothing_is_held(
    tmp_path, monkeypatch, plan, gpus, fast, fault, why
):
    """A canary whose bench reports a fatal GPU error, or prints nothing
    until the watchdog kills it, fails, the first canary of a GPU too: the
    GPU benches no block until a canary passes, and the other GPU benches
    them all."""
    cfg = _config(
        tmp_path,
        quiet_host={"enabled": False},
        health={"poll_s": 0.2, "max_wait_s": 1.0, "canary": {"retry_s": 60}},
    )
    args = ("--config-yaml", cfg, *_identity(tmp_path))
    args += ("--stall-timeout-s", 0.5, "--startup-grace-s", 2)
    monkeypatch.setenv(gpu_health.SESSION_VAR, "first")
    bench = plan(on_gpu={"0": fault})
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert rc == 0 and summary["lost"] == 0 and summary["problems"] == []
    assert {b["device"] for b in summary["blocks"]} == {1}
    read = summary["guard"]["health"]["canary"]["read"]
    (bad,) = [r for r in read if r["device"] == 0]
    assert bad["faulted"] and not bad["passed"] and not bad["reference"]
    assert bad["why"].startswith(why)
    assert all(r["passed"] for r in read if r["device"] == 1)
    refs = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").read()["canary"]
    assert set(refs["references"]) == {gpu_health.gpu_key(kfd.gpus()[1])}


@pytest.mark.parametrize(
    "failure, why",
    [
        ({"crash_silent_text": "hcanary"}, "its bench crashed (SIGABRT)"),
        ({"exit_text": "hcanary"}, "its bench printed no problem (rc=1)"),
    ],
    ids=["crash", "exit"],
)
def test_a_canary_whose_bench_fails_as_on_any_gpu_gives_no_verdict(
    tmp_path, monkeypatch, plan, gpus, fast, failure, why
):
    """A canary whose bench crashes, or exits non-zero before any problem,
    without a fatal GPU error gives no verdict and holds nothing: a bench
    binary, yaml or library that fails does that on any GPU, and holding
    the GPU for it would hold every GPU for good."""
    cfg = _config(
        tmp_path,
        quiet_host={"enabled": False},
        health={"poll_s": 0.2, "canary": {"retry_s": 60}},
    )
    args = ("--config-yaml", cfg, *_identity(tmp_path))
    monkeypatch.setenv(gpu_health.SESSION_VAR, "first")
    rc, out, summary = _stage03(
        monkeypatch, tmp_path, plan(on_gpu={"0": failure}), *args
    )
    assert rc == 0 and summary["lost"] == 0 and summary["problems"] == []
    assert summary["benched"] == len(GEMMS) and summary["retired_devices"] == []
    read = summary["guard"]["health"]["canary"]["read"]
    (unread,) = [r for r in read if r["device"] == 0]
    assert unread["passed"] and unread["unread"] and not unread["reference"]
    assert "faulted" not in unread and "fault" not in unread
    assert unread["bench_error"] == why[len("its bench ") :]
    assert unread["why"].startswith(
        f"no reference taken (0 of 4 canary lines read, {why}"
    )
    hold = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").hold()
    assert hold is None


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
    assert left == {"yaml_line": 1, "shape": [hung.m, hung.n, hung.k, 1], "why": "hung"}
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


# ── wedged GPUs ──────────────────────────────────────────────────────────────

WEDGE_ARGS = ("--stall-timeout-s", 0.5, "--startup-grace-s", 3, "--no-health-canary")


def _wedge_config(tmp_path):
    return _config(tmp_path, quiet_host={"enabled": False}, health={"poll_s": 0.2})


def _logs_with(out, text):
    """The text of every block log under `out` (attempts set aside too) that
    holds `text`."""
    return [p.read_text() for p in out.rglob("block_*.log") if text in p.read_text()]


def test_a_bench_that_does_not_exit_after_its_shapes_has_its_gpu_probed_first(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """A process that finished every shape and then hung is killed and its
    data kept; its GPU benches the device probe before its next block, and
    benches on once the probe runs through."""
    bench = plan(teardown_hang=[_key(GEMMS[1])], start_s=0.3)
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    args += (*_identity(tmp_path), "--blocks-per-gpu", 5)
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert rc == 0 and summary["problems"] == [] and summary["lost"] == 0
    assert summary["benched"] == len(GEMMS)
    (event,) = [b for b in summary["blocks"] if b["health_events"]]
    assert "did not exit (killed: stall)" in event["health_events"][0]
    (log,) = _logs_with(out, "GPU HEALTH EVENT")
    assert "*** KILLED [stall]" in log and "SKIPPED CRASHED" not in log
    (probe,) = summary["guard"]["probes"]
    assert probe["device"] == event["device"] and probe["passed"] is True
    probe_log = out / s03.PROBE_DIR / os.path.basename(probe["log"])
    (timed,) = parse_log_file(probe_log)
    assert timed.complete and [t.sol_index for t in timed.tested] == [100]
    assert summary["guard"]["wedged"]["found"] == []


def test_a_gpu_that_fails_its_device_probe_is_held_as_wedged(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """A probe that prints nothing holds its GPU as wedged until the host
    reboots: it takes no block again, in this stage or a later one, and the
    other GPU benches every line."""
    bench = plan(
        teardown_hang=[_key(GEMMS[1])], hang_silent_text="device_probe", start_s=0.3
    )
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    args += (*_identity(tmp_path), "--blocks-per-gpu", 5)
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert rc == 0 and summary["lost"] == 0 and summary["benched"] == len(GEMMS)
    (probe,) = summary["guard"]["probes"]
    assert probe["passed"] is False and probe["fault"] == "printed nothing"
    (wedge,) = summary["guard"]["wedged"]["found"]
    key = probe["key"]
    assert wedge["devices"] == [probe["device"]]
    assert "its device probe printed nothing" in wedge["gpus"][key]["why"]
    (gone,) = summary["retired_devices"]
    assert gone["device"] == probe["device"] and gone["why"].startswith("wedged: ")
    assert summary["guard"]["wedged"]["stopped"] is None
    hold = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").hold()
    assert gpu_health.wedged_keys(hold) == [key] and not gpu_health.run_wide(hold)
    rc, _, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert rc == 0 and summary["lost"] == 0
    assert {b["device"] for b in summary["blocks"]} == {1 - probe["device"]}


def _stuck_first(monkeypatch, calls=None):
    """The first bench the stage kills reads as stuck in the GPU driver: for
    good, or for its first `calls` readings (a driver call that is only
    slow). Returns the pids it held stuck."""
    from lib import procs

    real = procs.stuck_members
    first = []
    seen = []

    def stuck(proc):
        if not first:
            first.append(proc.pid)
        if proc.pid != first[0] or (calls is not None and len(seen) >= calls):
            return real(proc)
        seen.append(proc.pid)
        return [proc.pid]

    monkeypatch.setattr(procs, "stuck_members", stuck)
    monkeypatch.setattr(procs, "STUCK_WAIT_S", 1.0 if calls is None else 30.0)
    return first


def test_a_bench_stuck_in_the_driver_is_not_killed_and_its_gpu_is_held(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """The watchdog does not SIGKILL a bench that stays stuck in the GPU
    driver: it is left running, its GPU is held as wedged and takes no new
    process, and its block goes back on the work queue for the other GPU."""
    refused = _stuck_first(monkeypatch)
    hung = GEMMS[3]
    bench = plan(hang=[_key(hung)])
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    args += ("--blocks-per-gpu", 1)
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert rc == 0 and refused
    assert (summary["hung"], summary["lost"]) == (1, 1)
    assert summary["benched"] == len(GEMMS) - 1
    (wedge,) = summary["guard"]["wedged"]["found"]
    assert wedge["reaped"] is False
    (info,) = wedge["gpus"].values()
    assert info["processes"] == [{"pid": refused[0], "comm": "bench", "ended": False}]
    (aborted,) = summary["aborted"]
    assert aborted["reason"] == "wedged" and aborted["device"] == wedge["device"]
    (again,) = [b for b in summary["blocks"] if b["aborts"]]
    assert again["device"] == 1 - wedge["device"]
    (log,) = _logs_with(out, "*** WEDGED:")
    assert "did not end" in log and "*** KILLED" not in log
    assert "*** LEFT RUNNING [stall]: no output for 0s, but it is stuck in" in log


def test_a_bench_in_a_slow_driver_call_is_killed_and_holds_nothing(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """A bench the watchdog kills while it is in a driver call only for a
    while gets SIGKILL once it leaves the call, ends, and leaves its GPU
    unheld: a slow call proves no wedge."""
    slow = _stuck_first(monkeypatch, calls=2)
    bench = plan(hang=[_key(GEMMS[3])])
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    args += ("--blocks-per-gpu", 1)
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert rc == 0 and slow
    assert (summary["hung"], summary["lost"]) == (1, 1)
    assert summary["benched"] == len(GEMMS) - 1
    assert summary["guard"]["wedged"]["found"] == [] and summary["aborted"] == []
    assert summary["retired_devices"] == []
    assert _logs_with(out, "*** WEDGED:") == []
    (log,) = _logs_with(out, "*** KILLED [stall]")
    assert "LEFT RUNNING" not in log
    hold = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").hold()
    assert gpu_health.wedged_keys(hold) == []


def _stays_stuck(monkeypatch, pid, readings=None):
    """`pid`, a process that does not run here, reads as in uninterruptible
    sleep: in every reading, or in its first `readings` and then as gone;
    the stage watches a stuck process other than its bench for REAP_S, here
    a second."""
    from lib import procs

    real = procs.process_state
    seen = []

    def state(p):
        if p != pid:
            return real(p)
        seen.append(p)
        return "D" if readings is None or len(seen) <= readings else None

    monkeypatch.setattr(procs, "process_state", state)
    monkeypatch.setattr(s03, "REAP_S", 1.0)
    return seen


@pytest.mark.parametrize("memory", ["two_gpus", "none"])
def test_a_wedge_of_two_gpus_or_of_the_host_stops_the_stage(
    tmp_path, monkeypatch, plan, gpus, fast, memory
):
    """Processes stuck in the GPU driver on two GPUs of the stage, or ones
    it started holding memory on none (the host), after a kill stop the
    stage at once with its own exit status, and the run holds what they
    wedged."""
    from lib import hazards, procs

    other = int(open("/proc/sys/kernel/pid_max").read()) - 1
    if os.path.exists(f"/proc/{other}"):
        pytest.skip(f"pid {other} runs here")
    held = {1001: 4096, 1002: 4096} if memory == "two_gpus" else {}
    gf.set_memory(gpus, other, held)
    calls = []

    def stuck(samples=0, interval_s=0.0):
        calls.append(samples)
        return [] if len(calls) == 1 else [(other, "other")]

    monkeypatch.setattr(hazards, "stuck_processes", stuck)
    monkeypatch.setattr(kfd, "pids_comparable", lambda: True)
    monkeypatch.setattr(procs, "descendants", lambda _pid: [other])
    _stays_stuck(monkeypatch, other)
    bench = plan(hang=[_key(GEMMS[3])])
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    t0 = time.monotonic()
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert rc == procs.WEDGED_EXIT == summary["rc"]
    assert time.monotonic() - t0 < 60
    assert set(calls) == {s03.WEDGE_SAMPLES}
    assert summary["guard"]["wedged"]["stopped"]
    assert any(
        p.startswith("the guard stopped the stage: device wedged, reboot")
        for p in summary["problems"]
    )
    hold = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").hold()
    keys = [gpu_health.gpu_key(g) for g in kfd.gpus()[:2]]
    want = sorted(keys) if memory == "two_gpus" else [gpu_health.HOST]
    assert gpu_health.wedged_keys(hold) == want
    assert gpu_health.wedge_holds(hold, keys)


@pytest.mark.parametrize("memory", ["other_gpu", "none"])
def test_a_process_stuck_elsewhere_that_the_stage_did_not_start_counts_for_nothing(
    tmp_path, monkeypatch, plan, gpus, fast, memory
):
    """A process stuck in the GPU driver on a GPU outside the stage, or
    holding memory on none, that the stage did not start is recorded and
    holds or stops nothing: another run's wedge is not this stage's."""
    from lib import hazards

    other = int(open("/proc/sys/kernel/pid_max").read()) - 1
    if os.path.exists(f"/proc/{other}"):
        pytest.skip(f"pid {other} runs here")
    (outside,) = [g for g in kfd.gpus() if g.gpu_id == 1003]
    gf.set_memory(gpus, other, {1003: 4096} if memory == "other_gpu" else {})
    calls = []

    def stuck(samples=0, interval_s=0.0):
        calls.append(samples)
        return [] if len(calls) == 1 else [(other, "other")]

    monkeypatch.setattr(hazards, "stuck_processes", stuck)
    monkeypatch.setattr(kfd, "pids_comparable", lambda: True)
    bench = plan(hang=[_key(GEMMS[3])])
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert rc == 0 and len(calls) >= 2
    wedged = summary["guard"]["wedged"]
    assert wedged["stopped"] is None and wedged["found"] == []
    gpus_held = [outside.index] if memory == "other_gpu" else []
    assert wedged["elsewhere"] == [{"pid": other, "comm": "other", "gpus": gpus_held}]
    assert summary["retired_devices"] == []
    hold = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").hold()
    assert gpu_health.wedged_keys(hold) == []


def test_a_process_stuck_only_for_a_while_holds_nothing(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """A process of the stage that a check finds stuck in the GPU driver,
    but that ends within REAP_S, as a bench whose teardown is slow does,
    wedges nothing: the stage neither holds a GPU or the host nor stops."""
    from lib import hazards, procs

    other = int(open("/proc/sys/kernel/pid_max").read()) - 1
    if os.path.exists(f"/proc/{other}"):
        pytest.skip(f"pid {other} runs here")
    gf.set_memory(gpus, other, {})
    calls = []

    def stuck(samples=0, interval_s=0.0):
        calls.append(samples)
        return [] if len(calls) == 1 else [(other, "other")]

    monkeypatch.setattr(hazards, "stuck_processes", stuck)
    monkeypatch.setattr(kfd, "pids_comparable", lambda: True)
    monkeypatch.setattr(procs, "descendants", lambda _pid: [other])
    seen = _stays_stuck(monkeypatch, other, readings=1)
    bench = plan(hang=[_key(GEMMS[3])])
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert rc == 0 and len(calls) >= 2 and len(seen) >= 2
    wedged = summary["guard"]["wedged"]
    assert wedged["stopped"] is None and wedged["found"] == []
    assert summary["retired_devices"] == [] and summary["lost"] == 1
    hold = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").hold()
    assert gpu_health.wedged_keys(hold) == []


def _failing_bench(tmp_path, *, after=0, real=None):
    """A bench that prints an error and exits 1 before any problem, as one
    whose binary, yaml or library is broken does on any GPU; with `real`,
    only from its `after`-th process on, the ones before it being `real`."""
    script = tmp_path / "failing-bench"
    count = tmp_path / "failing-bench.count"
    count.write_text("0")
    body = "#!/bin/sh\n"
    if real is not None:
        body += (
            f"n=$(cat {count}); n=$((n+1)); echo $n > {count}\n"
            f'if [ "$n" -le {after} ]; then exec {real} "$@"; fi\n'
        )
    body += "echo 'hipblaslt-bench: error: cannot open the library'\nexit 1\n"
    script.write_text(body)
    script.chmod(0o755)
    return script


@pytest.mark.parametrize("canary", [False, True], ids=["plain", "canary"])
def test_a_bench_that_fails_on_every_gpu_holds_none_and_the_stage_may_retry(
    tmp_path, monkeypatch, plan, gpus, fast, canary
):
    """Bench processes that exit 1 before any problem, on every GPU and
    with the GPUs' canaries read first and passed: each GPU abandons a
    block, and its device probe fails the same way, which a broken bench,
    yaml or library does on any GPU. The GPU leaves the stage, but the run
    does not hold it as wedged, and the stage exits 2, which another
    attempt may cure, not 3."""
    from lib import procs

    if canary:
        bench = _failing_bench(tmp_path, after=2, real=plan())
        cfg = _config(tmp_path, quiet_host={"enabled": False}, health={"poll_s": 0.2})
        args = ("--config-yaml", cfg, "--stall-timeout-s", 0.5)
        args += ("--startup-grace-s", 3)
    else:
        bench = _failing_bench(tmp_path)
        args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    many = [Gemm(64 + 8 * i, 32 + 4 * i, 128 + 16 * i) for i in range(40)]
    args += (*_identity(tmp_path), "--blocks-per-gpu", 2)
    rc, out, summary = _stage03(
        monkeypatch, tmp_path, bench, *args, lines=[bench_line(g) for g in many]
    )
    assert rc == 2 == summary["rc"] and rc != procs.WEDGED_EXIT
    if canary:
        read = summary["guard"]["health"]["canary"]["read"]
        assert len(read) == 2 and all(r["reference"] for r in read)
    wedged = summary["guard"]["wedged"]
    assert wedged["stopped"] is None and wedged["found"] == []
    probes = summary["guard"]["probes"]
    assert sorted(p["device"] for p in probes) == [0, 1]
    for p in probes:
        assert p["passed"] is False and "fault" not in p
        assert p["bench_error"] == "printed no problem (rc=1)"
    gone = summary["retired_devices"]
    assert [g["device"] for g in gone] == [0, 1]
    assert all("does not hold it as wedged" in g["why"] for g in gone)
    assert any("every GPU stopped taking blocks" in p for p in summary["problems"])
    assert procs.retry_status(rc, bench=True) == 2
    hold = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").hold()
    assert gpu_health.wedged_keys(hold) == []


def test_a_stage_whose_last_gpu_wedges_after_another_failed_is_retried(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """One GPU leaves the stage because its benches and its device probe
    fail as on any GPU, the other wedges: not every GPU of the stage is
    wedged, so the stage ends as one that ran out of GPUs does (exit 2, which
    another attempt may cure on the GPU the run does not hold), not with a
    wedge stop."""
    from lib import procs

    _stuck_first(monkeypatch)
    bench = plan(
        on_gpu={
            "0": {"exit_text": "matmul"},
            "1": {"hang_silent_text": "matmul"},
        }
    )
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    args += (*_identity(tmp_path), "--blocks-per-gpu", 1)
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert rc == 2 == summary["rc"]
    wedged = summary["guard"]["wedged"]
    assert wedged["stopped"] is None
    (wedge,) = wedged["found"]
    assert wedge["devices"] == [1]
    (probe,) = summary["guard"]["probes"]
    assert probe["device"] == 0 and probe["bench_error"]
    assert [g["device"] for g in summary["retired_devices"]] == [0, 1]
    hold = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").hold()
    assert gpu_health.wedged_keys(hold) == [gpu_health.gpu_key(kfd.gpus()[1])]
    assert procs.retry_status(rc, bench=True) == 2


def test_two_wedged_gpus_of_the_stage_stop_it_though_a_third_benches(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """Two of the stage's GPUs held as wedged, each by a wedge of its own,
    stop the stage as one wedge of both does, though its third GPU could
    bench: two wedged GPUs are a reboot's matter."""
    from lib import procs

    monkeypatch.setenv(kfd.ROCR_VAR, "0,1,2")
    keys = [gpu_health.gpu_key(g) for g in kfd.gpus()[:3]]
    guard = gpu_health.Guard(tmp_path / "stage03" / gpu_health.GUARD_DIR, "earlier")
    guard.hold_wedged({keys[0]: f"GPU 0 ({keys[0]}) is wedged: a test"})
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    args += ("--devices", "0,1,2")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert rc == 0 and summary["lost"] == 0
    assert {b["device"] for b in summary["blocks"]} <= {1, 2}
    guard.hold_wedged({keys[1]: f"GPU 1 ({keys[1]}) is wedged: another test"})
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert rc == procs.WEDGED_EXIT == summary["rc"]
    stopped = summary["guard"]["wedged"]["stopped"]
    assert stopped.startswith("2 GPUs of the stage are wedged (")
    assert 2 not in {d["device"] for d in summary["retired_devices"]}


def test_a_line_whose_process_wedges_its_gpu_is_benched_on_no_other_gpu(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """A line a bench reports a memory fault in, and is then stuck in the
    GPU driver, leaves the stage: its block goes back on the work queue
    without it, and neither the other GPU nor the retry pass benches it,
    so that a line that wedges every GPU it runs on wedges one."""
    from lib import procs

    stuck = _stuck_first(monkeypatch)
    bad = GEMMS[3]
    bench = plan(memory_fault=[_key(bad)])
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    args += ("--blocks-per-gpu", 1)
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert stuck and rc == 0, summary["problems"]
    (wedge,) = summary["guard"]["wedged"]["found"]
    assert (summary["lost"], summary["retried"]) == (1, 0)
    assert summary["benched"] == len(GEMMS) - 1
    (again,) = [b for b in summary["blocks"] if b["aborts"]]
    assert again["device"] == 1 - wedge["device"] and again["wedge_lines"] == [4]
    rows = (out / "skipped_gemms.txt").read_text().splitlines()[1:]
    assert rows == [f"4,{bad.m},{bad.n},{bad.k},1,{procs.STUCK_RC},wedged"]
    (log,) = _logs_with(out, "*** LEFT OUT WEDGING GEMM at block-offset 3")
    assert "*** ABORTED [wedged]" in log and "quarantined in the run (bench)" in log
    assert len(_logs_with(out, "MEMORY_FAULT")) == 1
    shape = [bad.m, bad.n, bad.k, 1]
    (w,) = summary["guard"]["wedge_lines"]
    assert (w["yaml_line"], w["shape"], w["why"]) == (4, shape, "wedged")
    assert (w["scope"], w["quarantined"], w["device"]) == (
        "bench",
        True,
        wedge["device"],
    )
    (entry,) = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").read()[
        "quarantined_lines"
    ]
    assert (entry["shape"], entry["why"], entry["scope"]) == (shape, "wedged", "bench")
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert rc == 0 and summary["lost"] == 1
    assert summary["benched"] == len(GEMMS) - 1
    (left,) = summary["guard"]["quarantined_lines_left_out"]
    assert left == {"yaml_line": 4, "shape": shape, "why": "wedged"}


def test_a_line_whose_warmup_shape_wedges_its_gpu_is_benched_on_no_other_gpu(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """With the warm-up pass, every process starts with its block's next
    line's shape over all its kernels: a bench that faults there, after the
    warm-up line, and is then stuck in the GPU driver wedged its GPU on that
    line, which is quarantined in the run as a wedging line, and its block
    goes back on the work queue without it."""
    from lib import procs

    stuck = _stuck_first(monkeypatch)
    bad = GEMMS[0]
    bench = plan(memory_fault=[_key(bad)])
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    args += ("--blocks-per-gpu", 1, "--warmup-pass", "--warmup-line")
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert stuck and rc == 0, summary["problems"]
    assert (summary["lost"], summary["retried"]) == (1, 0)
    assert summary["benched"] == len(GEMMS) - 1
    rows = (out / "skipped_gemms.txt").read_text().splitlines()[1:]
    assert rows == [f"1,{bad.m},{bad.n},{bad.k},1,{procs.STUCK_RC},wedged"]
    (w,) = summary["guard"]["wedge_lines"]
    assert (w["yaml_line"], w["why"], w["quarantined"]) == (1, "wedged", True)
    assert w["in_flight"].startswith("its warm-up shape (")
    (log,) = _logs_with(out, "*** LEFT OUT WEDGING GEMM at block-offset 0")
    assert "left the GPU wedged in its warm-up shape" in log
    assert "WEDGED BEFORE" not in log and "WARM-UP LINE FAILED" not in log
    (entry,) = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").read()[
        "quarantined_lines"
    ]
    assert (entry["yaml_line"], entry["why"]) == (1, "wedged")


def test_a_wedge_of_another_gpu_leaves_a_crash_on_this_one_a_crash(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """One GPU of the stage is wedged (a process the stage did not start
    stays stuck in the GPU driver holding memory on it) and the other GPU's
    benches fault on some lines. The check after such a fault holds the
    wedged GPU alone; the GPU that faulted benches on, and its faulted lines
    are skipped and benched again in the retry pass as any crash, not
    dropped as wedging lines. Later checks leave the stuck process they
    found out, so it is held once."""
    from lib import hazards

    pid = int(open("/proc/sys/kernel/pid_max").read()) - 3
    if os.path.exists(f"/proc/{pid}"):
        pytest.skip(f"pid {pid} runs here")
    calls = []

    def stuck(samples=0, interval_s=0.0):
        calls.append(samples)
        return [] if len(calls) == 1 else [(pid, "bench")]

    monkeypatch.setattr(hazards, "stuck_processes", stuck)
    monkeypatch.setattr(kfd, "pids_comparable", lambda: True)
    gf.set_memory(gpus, pid, {1001: 4096})
    _stays_stuck(monkeypatch, pid)
    faults = [_key(GEMMS[i]) for i in (1, 3, 6, 8)]
    bench = plan(on_gpu={"1": {"memory_fault": faults}})
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    args += ("--blocks-per-gpu", 1)
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert rc == 0 and summary["guard"]["wedged"]["stopped"] is None
    (wedge,) = summary["guard"]["wedged"]["found"]
    key0 = gpu_health.gpu_key(kfd.gpus()[0])
    assert wedge["device"] == 1 and list(wedge["gpus"]) == [key0]
    assert wedge["devices"] == [0] and wedge["host"] == []
    (gone,) = summary["retired_devices"]
    assert gone["device"] == 0 and gone["why"].startswith("wedged: ")
    assert summary["guard"]["wedge_lines"] == []
    assert all(b["wedge_lines"] == [] for b in summary["blocks"])
    assert all(a["reason"] != "wedged" for a in summary["aborted"])
    rows = (out / "skipped_gemms.txt").read_text().splitlines()[1:]
    faulted = {i + 1 for i in (1, 3, 6, 8)}
    assert rows and {int(r.split(",")[0]) for r in rows} <= faulted
    assert all(r.endswith(",crashed") for r in rows)
    assert summary["retried"] == summary["lost"] == len(rows)
    assert _logs_with(out, "*** WEDGE ELSEWHERE: wedged GPU 0")
    assert not _logs_with(out, "*** WEDGED:")
    hold = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").hold()
    assert gpu_health.wedged_keys(hold) == [key0]


def _wedge_elsewhere_once(monkeypatch, what):
    """The first wedge check after a process whose yaml is a `what`
    (device_probe, health_canary) finds the stage's other GPU wedged, and
    only it; the devices it was taken for."""
    real = s03._wedge_of
    fired = []

    def wedge_of(device, proc, rc, gpus, *a, **kw):
        args = " ".join(str(x) for x in getattr(proc, "args", None) or [])
        if f"/{what}_" not in args or fired:
            return real(device, proc, rc, gpus, *a, **kw)
        fired.append(device)
        other = gpus[1 - device]
        return {
            "device": device,
            "gpu": gpu_health.gpu_key(gpus[device]),
            "reaped": True,
            "gpus": {
                gpu_health.gpu_key(other): {
                    "gpu": other.index,
                    "processes": [{"pid": 4194300, "comm": "other"}],
                }
            },
            "host": [],
            "at": kfd.utc_iso(),
        }

    monkeypatch.setattr(s03, "_wedge_of", wedge_of)
    return fired


def test_a_failed_device_probe_holds_its_gpu_whatever_is_wedged_elsewhere(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """A device probe that prints nothing holds its GPU as wedged also when
    the wedge check after it finds the stage's other GPU wedged, which the
    run then holds too: a wedge elsewhere says nothing about the probed
    GPU. Two wedged GPUs of the stage stop it."""
    from lib import procs

    fired = _wedge_elsewhere_once(monkeypatch, "device_probe")
    bench = plan(
        teardown_hang=[_key(GEMMS[1])], hang_silent_text="device_probe", start_s=0.3
    )
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    args += (*_identity(tmp_path), "--blocks-per-gpu", 5)
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    (probe,) = summary["guard"]["probes"]
    assert fired == [probe["device"]]
    assert probe["passed"] is False and probe["fault"] == "printed nothing"
    assert rc == procs.WEDGED_EXIT == summary["rc"]
    found = summary["guard"]["wedged"]["found"]
    assert sorted(d for w in found for d in w["devices"]) == [0, 1]
    gone = {g["device"]: g["why"] for g in summary["retired_devices"]}
    assert set(gone) == {0, 1} and all(w.startswith("wedged: ") for w in gone.values())
    keys = [gpu_health.gpu_key(g) for g in kfd.gpus()[:2]]
    hold = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").hold()
    assert gpu_health.wedged_keys(hold) == sorted(keys)


def test_a_canary_without_a_verdict_stays_so_beside_a_wedge_elsewhere(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """A health canary whose bench fails as on any GPU (no verdict), while
    the wedge check after it finds the stage's other GPU wedged, still gives
    no verdict: the run holds the other GPU, and the canary's GPU benches
    every line."""
    fired = _wedge_elsewhere_once(monkeypatch, "health_canary")
    cfg = _config(
        tmp_path,
        quiet_host={"enabled": False},
        health={"poll_s": 0.2, "max_wait_s": 5.0, "canary": {"retry_s": 60}},
    )
    args = ("--config-yaml", cfg, *_identity(tmp_path))
    monkeypatch.setenv(gpu_health.SESSION_VAR, "first")
    bench = plan(on_gpu={"0": {"exit_text": "hcanary"}, "1": {"start_s": 1.0}})
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert fired == [0]
    assert rc == 0 and summary["lost"] == 0 and summary["problems"] == []
    read = summary["guard"]["health"]["canary"]["read"]
    (mine,) = [r for r in read if r["device"] == 0]
    assert mine["passed"] and mine["unread"] and "faulted" not in mine
    assert "wedged" not in mine and mine["bench_error"] == "printed no problem (rc=1)"
    (gone,) = summary["retired_devices"]
    assert gone["device"] == 1 and gone["why"].startswith("wedged: ")
    hold = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").hold()
    assert gpu_health.wedged_keys(hold) == [gpu_health.gpu_key(kfd.gpus()[1])]


def test_a_line_that_wedges_its_gpu_in_the_retry_pass_is_reported_wedged(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """A line that crashed in its block and, benched again in the retry
    pass, faults there with its process stuck in the GPU driver is without
    data as `wedged`, not as its block's crash, and quarantined in the run;
    the retry pass benches its other lines on the other GPU."""
    from lib import procs

    real = procs.stuck_members
    left = []

    def stuck(proc):
        args = " ".join(str(x) for x in getattr(proc, "args", None) or [])
        if "/block_retry_" not in args or proc.poll() is not None:
            return real(proc)
        if proc not in left:
            left.append(proc)
        return [proc.pid]

    monkeypatch.setattr(procs, "stuck_members", stuck)
    monkeypatch.setattr(procs, "STUCK_WAIT_S", 1.0)
    bad, once = GEMMS[3], GEMMS[7]
    bench = plan(memory_fault=[_key(bad)], crash_once=[_key(once)])
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    args += ("--blocks-per-gpu", 1)
    try:
        rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    finally:
        for proc in left:
            try:
                os.killpg(proc.pid, 9)
            except OSError:
                pass
    assert rc == 0 and len(left) == 1, summary["problems"]
    assert (summary["retried"], summary["recovered"], summary["lost"]) == (2, 1, 1)
    rows = (out / "skipped_gemms.txt").read_text().splitlines()[1:]
    assert rows == [f"4,{bad.m},{bad.n},{bad.k},1,{procs.STUCK_RC},wedged"]
    retry = next(b for b in summary["blocks"] if b["name"] == "block_retry")
    assert retry["wedge_lines"] == [4] and retry["aborts"] == 1
    (w,) = summary["guard"]["wedge_lines"]
    assert w["block"] == "block_retry" and w["quarantined"] is True
    guard = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x")
    assert guard.quarantined_lines() == {
        (bad.m, bad.n, bad.k, 1): guard.read()["quarantined_lines"][0]
    }
    assert len(summary["retired_devices"]) == 1


def test_a_gpu_whose_canary_keeps_faulting_leaves_the_stage(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """A health canary whose bench faults every time (its kernel faults at
    its shape, here on every GPU) does not hold the stage for good under
    the default bench.health.max_wait_s of 0: a GPU leaves the stage after
    CANARY_FAULT_LIMIT such canaries in a row, naming the way out, and a
    stage that loses every GPU so exits 2."""
    cfg = _config(
        tmp_path,
        quiet_host={"enabled": False},
        health={"poll_s": 0.2, "canary": {"retry_s": 0.2}},
    )
    args = ("--config-yaml", cfg, *_identity(tmp_path))
    monkeypatch.setenv(gpu_health.SESSION_VAR, "first")
    bench = plan(memory_fault_text="hcanary")
    rc, out, summary = _stage03(monkeypatch, tmp_path, bench, *args)
    assert rc == 2 and summary["benched"] == 0
    read = summary["guard"]["health"]["canary"]["read"]
    for d in (0, 1):
        mine = [r for r in read if r["device"] == d]
        assert len(mine) == s03.CANARY_FAULT_LIMIT
        assert all(r["faulted"] and not r["passed"] for r in mine)
    gone = {g["device"]: g["why"] for g in summary["retired_devices"]}
    assert set(gone) == {0, 1}
    for why in gone.values():
        assert f"in {s03.CANARY_FAULT_LIMIT} canaries in a row" in why
        assert "only turning the health canary off" in why
        assert "bench.health.canary.enabled: false" in why
    assert any("every GPU stopped taking blocks" in p for p in summary["problems"])
    hold = gpu_health.Guard(out / gpu_health.GUARD_DIR, "x").hold()
    assert gpu_health.wedged_keys(hold) == []


def test_gpus_the_run_holds_as_wedged_take_no_block(
    tmp_path, monkeypatch, plan, gpus, fast
):
    """A GPU the run holds as wedged leaves the stage at once, beside the
    others; when every GPU is, the stage stops with its own exit status."""
    from lib import procs

    keys = [gpu_health.gpu_key(g) for g in kfd.gpus()[:2]]
    guard = gpu_health.Guard(tmp_path / "stage03" / gpu_health.GUARD_DIR, "earlier")
    guard.hold_wedged({keys[0]: f"GPU 0 ({keys[0]}) is wedged: a test"})
    args = ("--config-yaml", _wedge_config(tmp_path), *WEDGE_ARGS)
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert rc == 0 and summary["lost"] == 0
    assert {b["device"] for b in summary["blocks"]} == {1}
    (gone,) = summary["retired_devices"]
    assert gone["device"] == 0 and "a test" in gone["why"]
    guard.hold_wedged({keys[1]: f"GPU 1 ({keys[1]}) is wedged: a test"})
    rc, out, summary = _stage03(monkeypatch, tmp_path, plan(), *args)
    assert rc == procs.WEDGED_EXIT and summary["benched"] == 0
    assert summary["guard"]["wedged"]["stopped"].startswith("every GPU of the stage")
