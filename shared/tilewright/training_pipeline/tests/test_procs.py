# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Process groups, interrupts and the stall watchdog of the bench stages,
with fake processes and the fake bench (no GPU)."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import stage03_load_balance_offline_tuning as s03
from bench_fakes import (
    LIBRARY_STEM,
    Gemm,
    bench_line,
    solution,
    write_fake_bench,
    write_logic,
)
from lib import kfd, procs

PIPELINE_DIR = Path(__file__).resolve().parent.parent
GEMMS = [Gemm(64 + 8 * i, 32 + 4 * i, 128 + 16 * i) for i in range(6)]


def _key(g):
    return f"{g.m}x{g.n}x{g.k}x{g.batch}"


# ── watchdog ─────────────────────────────────────────────────────────────────


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _Proc:
    def __init__(self, clock, exit_at=None, rc=0):
        self.clock = clock
        self.exit_at = exit_at
        self.rc = rc
        self.killed_at = None

    def wait(self, timeout=None):
        if self.killed_at is not None:
            return -9
        if timeout is None:
            return self.rc
        if self.exit_at is not None and self.clock.t + timeout >= self.exit_at:
            self.clock.t = self.exit_at
            return self.rc
        self.clock.t += timeout
        raise subprocess.TimeoutExpired("fake", timeout)

    def kill(self):
        self.killed_at = self.clock.t


def _watch(proc, clock, size, **kw):
    kw.setdefault("startup_grace_s", 100.0)
    return procs.wait_with_stall_watchdog(
        proc, size, 30.0, poll_s=10.0, clock=clock, **kw
    )


def test_watchdog_kills_a_silent_process_after_the_startup_grace():
    clock = _Clock()
    proc = _Proc(clock)
    assert _watch(proc, clock, lambda: 0) == (-9, "stall")
    assert proc.killed_at == 100.0


def test_watchdog_gives_a_writing_process_the_full_stall_window_after_grace():
    clock = _Clock()
    proc = _Proc(clock)
    assert _watch(proc, clock, lambda: int(min(clock.t, 95.0) * 10)) == (-9, "stall")
    assert proc.killed_at >= 125.0


def test_watchdog_after_progress_uses_only_the_stall_rule():
    clock = _Clock()
    proc = _Proc(clock)
    rc, reason = _watch(
        proc,
        clock,
        lambda: int(min(clock.t, 50.0) * 10),
        probe_progress=lambda: clock.t >= 20.0,
    )
    assert (rc, reason, proc.killed_at) == (-9, "stall", 80.0)


def test_watchdog_fatal_exit_and_disabled():
    clock = _Clock()
    proc = _Proc(clock)
    assert _watch(proc, clock, lambda: 0, probe_fatal=lambda: clock.t >= 30.0) == (
        -9,
        "fatal",
    )
    assert proc.killed_at == 30.0
    clock = _Clock()
    assert _watch(_Proc(clock, exit_at=45.0, rc=3), clock, lambda: 0) == (3, None)
    clock = _Clock()
    assert procs.wait_with_stall_watchdog(_Proc(clock, rc=5), lambda: 0, 0) == (
        5,
        None,
    )


def test_watchdog_kills_at_once_when_another_process_is_on_the_gpu():
    clock = _Clock()
    proc = _Proc(clock)
    seen = lambda: clock.t >= 20.0  # noqa: E731
    assert _watch(proc, clock, lambda: 0, probe_contaminated=seen) == (
        -9,
        "contaminated",
    )
    assert proc.killed_at == 20.0
    clock = _Clock()
    proc = _Proc(clock)
    rc = procs.wait_with_stall_watchdog(
        proc, lambda: 0, 0, poll_s=10.0, clock=clock, probe_contaminated=seen
    )
    assert rc == (-9, "contaminated") and proc.killed_at == 20.0


def test_poll_interval():
    assert procs.poll_interval(300) == 15.0
    assert procs.poll_interval(30) == 3.0
    assert procs.poll_interval(0.5) == 0.2


# ── process groups ───────────────────────────────────────────────────────────


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_dead(pids, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(_alive(p) for p in pids):
            return True
        time.sleep(0.05)
    return False


def _wait_for(predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


SPAWNER = textwrap.dedent(
    """
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    print(child.pid, flush=True)
    time.sleep(600)
    """
)


def test_stop_all_ends_each_group_and_refuses_new_children(tmp_path):
    children = procs.ChildGroups(grace_s=5.0)
    proc = children.popen(
        [sys.executable, "-c", SPAWNER], stdout=subprocess.PIPE, text=True
    )
    grandchild = int(proc.stdout.readline())
    assert os.getpgid(proc.pid) == proc.pid != os.getpgid(0)
    assert os.getpgid(grandchild) == proc.pid
    assert children.stop_all() == 1
    assert proc.returncode == -signal.SIGTERM
    assert _wait_dead([grandchild])
    with pytest.raises(procs.ChildrenStopped):
        children.popen([sys.executable, "-c", "pass"])
    assert children.stop_all() == 0


def test_kill_group_falls_back_to_kill_without_a_group():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    assert procs.kill_group(proc) == -signal.SIGKILL


GUARDED = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {root!r})
    from lib import procs

    children = procs.ChildGroups(grace_s=5.0)
    with procs.stop_children_on_exit(children):
        p = children.popen([sys.executable, "-c", "import time; time.sleep(600)"])
        print(p.pid, flush=True)
        time.sleep(600)
    """
)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_a_signal_stops_the_children_and_sets_the_exit_code(sig):
    stage = subprocess.Popen(
        [sys.executable, "-c", GUARDED.format(root=str(PIPELINE_DIR))],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child = int(stage.stdout.readline())
    stage.send_signal(sig)
    _out, err = stage.communicate(timeout=60)
    assert stage.returncode == 128 + sig
    assert signal.Signals(sig).name in err
    assert _wait_dead([child])


# ── stages ───────────────────────────────────────────────────────────────────


def _plan(tmp_path, **plan):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return {**os.environ, "FAKE_BENCH_PLAN": str(path)}


def _records(pid_dir):
    return {
        int(p.name): json.loads(p.read_text())
        for p in pid_dir.iterdir()
        if not p.name.startswith(".")
    }


def _stage_cmd(script, in_dir, out_dir, bench, *extra):
    return [
        sys.executable,
        str(PIPELINE_DIR / "stages" / script),
        "--in-dir",
        str(in_dir),
        "--out-dir",
        str(out_dir),
        "--bench-binary",
        str(bench),
        "--devices",
        "0,1",
        "--quiet",
        *extra,
    ]


def _interrupt(cmd, env, pid_dir, n_benches, sig):
    stage = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        assert _wait_for(lambda: len(_records(pid_dir)) >= n_benches)
        stage.send_signal(sig)
        stage.communicate(timeout=60)
    finally:
        if stage.poll() is None:
            stage.kill()
    records = _records(pid_dir)
    for pid, rec in records.items():
        assert rec["pgid"] == pid != os.getpgid(0)
    assert _wait_dead(list(records))
    return stage.returncode


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_stage03_interrupt_leaves_no_bench_running(tmp_path, sig):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "shapes.yaml").write_text("".join(bench_line(g) for g in GEMMS))
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    env = _plan(tmp_path, hang=[_key(g) for g in GEMMS], pid_dir=str(pid_dir))
    cmd = _stage_cmd(
        "stage03_load_balance_offline_tuning.py",
        in_dir,
        tmp_path / "out",
        write_fake_bench(tmp_path),
        "--no-warmup-pass",
        "--blocks-per-gpu",
        1,
    )
    assert _interrupt([str(c) for c in cmd], env, pid_dir, 2, sig) == 128 + sig
    assert not (tmp_path / "out" / "tmp").exists()


@pytest.mark.parametrize("canary", [False, True], ids=["plain", "canary"])
def test_stage03_watchdog_kills_a_hung_bench(tmp_path, monkeypatch, canary):
    """A shape hangs after its first lines: the watchdog kills the process
    and the shape is quarantined as hung, with the canary too (its output
    had ended before the shape started)."""
    in_dir = tmp_path / "stage01"
    in_dir.mkdir()
    (in_dir / "shapes.yaml").write_text("".join(bench_line(g) for g in GEMMS))
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    hung = GEMMS[1]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"hang": [_key(hung)], "pid_dir": str(pid_dir)}))
    monkeypatch.setenv("FAKE_BENCH_PLAN", str(plan))
    out = tmp_path / "stage03"
    extra = ["--no-canary"]
    if canary:
        library = tmp_path / "library"
        library.mkdir()
        write_logic(library / f"{LIBRARY_STEM}.dat", [solution(100, 0)])
        extra = ["--canary", "--identity-dir", library, "--identity-stem", LIBRARY_STEM]
    argv = _stage_cmd(
        "stage03_load_balance_offline_tuning.py",
        in_dir,
        out,
        write_fake_bench(tmp_path),
        "--blocks-per-gpu",
        1,
        "--no-warmup-pass",
        "--no-retry-skipped",
        "--stall-timeout-s",
        0.5,
        "--startup-grace-s",
        5,
        *extra,
    )[1:]
    monkeypatch.setattr(sys, "argv", [str(a) for a in argv])
    t0 = time.monotonic()
    assert s03.main() == 0
    assert time.monotonic() - t0 < 60
    assert _wait_dead(list(_records(pid_dir)))
    log = next((out / "logs").glob("block_0000_*.log")).read_text()
    assert "*** KILLED [stall]" in log and "*** QUARANTINED HUNG LINE" in log
    summary = json.loads((out / "stage03_summary.json").read_text())
    assert (summary["benched"], summary["skipped"], summary["lost"]) == (5, 0, 1)
    assert summary["hung"] == 1
    assert summary["canary"]["enabled"] is canary
    if canary:
        assert summary["canary"]["off"] == {} and "CANARY FAILED" not in log
    lost = (out / "skipped_gemms.txt").read_text().splitlines()[1:]
    assert lost == [f"2,{hung.m},{hung.n},{hung.k},1,,hung"]
    assert not (out / "tmp").exists()
