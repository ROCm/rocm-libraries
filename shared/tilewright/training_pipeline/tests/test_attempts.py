# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Stage attempts: the token every process of an attempt carries, the
run directory's record of running attempts, and stopping what survives an
orchestrator."""

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from gpu_fakes import wait_for
from lib import attempts, kfd, procs

PIPELINE_DIR = Path(__file__).resolve().parent.parent

DETACHED = textwrap.dedent(
    """
    import os, signal, subprocess, sys, time
    if sys.argv[1] == "ignore":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"],
        start_new_session=True,
    )
    print(os.getpid(), child.pid, flush=True)
    time.sleep(600)
    """
)


def _detached(token, how="default"):
    """A process carrying `token` and a child of it in another session."""
    proc = subprocess.Popen(
        [sys.executable, "-c", DETACHED, how],
        env=attempts.environ_with(token),
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    pids = [int(p) for p in proc.stdout.readline().split()]
    return proc, pids


def _alive(pid):
    # A process that exits between the open and the read of its stat file
    # fails the read with ESRCH (ProcessLookupError).
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rpartition(")")[2].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def test_processes_of_finds_every_process_carrying_the_token():
    token, other_token = attempts.new_token(), attempts.new_token()
    proc, pids = _detached(token)
    other, others = _detached(other_token)
    try:
        assert attempts.processes_of(token) == sorted(pids)
        assert set(attempts.processes_of(token)).isdisjoint(others)
    finally:
        for t in (token, other_token):
            attempts.stop(t, grace_s=5)
        for p in (proc, other):
            p.wait()


def test_stop_ends_a_tree_in_several_sessions():
    token = attempts.new_token()
    proc, pids = _detached(token)
    found, left = attempts.stop(token, grace_s=10)
    proc.wait(timeout=10)
    assert sorted(found) == sorted(pids) and left == []
    assert wait_for(lambda: not any(_alive(p) for p in pids), 10)


def test_stop_kills_what_ignores_sigterm_after_the_grace_period():
    token = attempts.new_token()
    proc, pids = _detached(token, "ignore")
    t0 = time.monotonic()
    found, left = attempts.stop(token, grace_s=1.0)
    proc.wait(timeout=10)
    assert found and left == []
    assert time.monotonic() - t0 >= 1.0
    assert proc.returncode == -signal.SIGKILL


def test_stop_sends_no_sigkill_to_a_process_stuck_in_a_driver_call(monkeypatch):
    token = attempts.new_token()
    proc, pids = _detached(token, "ignore")
    try:
        real = procs.process_state
        monkeypatch.setattr(
            procs, "process_state", lambda pid: "D" if pid == pids[0] else real(pid)
        )
        monkeypatch.setattr(procs, "STUCK_INTERVAL_S", 0.01)
        monkeypatch.setattr(attempts, "KILL_WAIT_S", 0.5)
        found, left = attempts.stop(token, grace_s=0.5)
        assert sorted(found) == sorted(pids) and left == [pids[0]]
        assert proc.poll() is None
        assert wait_for(lambda: not _alive(pids[1]), 10)
    finally:
        proc.kill()
        proc.wait()


def test_the_running_record_lists_attempts_until_they_end(tmp_path):
    running = attempts.Running(tmp_path)
    assert running.records() == {}
    running.add("a1", pid=os.getpid(), cmd=["x", 1], label="round_0 stage03")
    running.add("b2", pid=os.getpid(), cmd=["y"], label="round_0 stage03b")
    recs = json.loads((tmp_path / attempts.RUNNING_FILE).read_text())
    assert recs["a1"]["label"] == "round_0 stage03"
    assert recs["a1"]["pgid"] == os.getpgid(0) and recs["a1"]["cmd"] == ["x", "1"]
    assert recs["a1"]["run_dir"] == str(tmp_path.resolve())
    assert recs["a1"]["orchestrator_pid"] == os.getpid()
    assert recs["a1"]["orchestrator_start_s"] == kfd.proc_info(os.getpid()).start_s
    running.remove("a1")
    assert list(running.records()) == ["b2"]
    running.remove("b2")
    assert not (tmp_path / attempts.RUNNING_FILE).exists()


def test_survivors_of_a_dead_orchestrator_are_stopped(tmp_path):
    token = attempts.new_token()
    proc, pids = _detached(token, "ignore")
    attempts.Running(tmp_path).add(token, pid=pids[0], cmd=["s"], label="r stage")
    found = attempts.stop_survivors(tmp_path, grace_s=1.0)
    proc.wait(timeout=10)
    assert found.stopped == [("r stage", sorted(pids))] and found.stuck == []
    assert not (tmp_path / attempts.RUNNING_FILE).exists()
    assert attempts.stop_survivors(tmp_path) == attempts.Survivors()


def _orchestrated_by(run_dir, token, pid):
    """`run_dir`'s record of `token` as if the live process `pid` had
    written it."""
    path = Path(run_dir) / attempts.RUNNING_FILE
    recs = json.loads(path.read_text())
    recs[token].update(
        orchestrator_pid=pid, orchestrator_start_s=kfd.proc_info(pid).start_s
    )
    path.write_text(json.dumps(recs))
    return recs


def test_the_attempts_of_another_run_directory_are_left_alone(tmp_path):
    """A copy of a run directory made while a stage ran carries its record:
    an orchestrator on the copy drops it and signals nothing."""
    token = attempts.new_token()
    proc, pids = _detached(token)
    original, copy = tmp_path / "run", tmp_path / "copy"
    try:
        attempts.Running(original).add(token, pid=pids[0], cmd=["s"], label="r s3")
        _orchestrated_by(original, token, proc.pid)
        shutil.copytree(original, copy)
        found = attempts.stop_survivors(copy, grace_s=1.0)
        assert found == attempts.Survivors(elsewhere=[("r s3", str(original))])
        assert not (copy / attempts.RUNNING_FILE).exists()
        assert all(_alive(p) for p in pids)
        assert list(attempts.Running(original).records()) == [token]
    finally:
        attempts.stop(token, grace_s=5)
        proc.wait()


def test_the_attempts_of_a_live_orchestrator_are_never_signalled(tmp_path):
    token = attempts.new_token()
    proc, pids = _detached(token)
    try:
        attempts.Running(tmp_path).add(token, pid=pids[0], cmd=["s"], label="r s3")
        recs = _orchestrated_by(tmp_path, token, proc.pid)
        found = attempts.stop_survivors(tmp_path, grace_s=1.0)
        assert found == attempts.Survivors(alive=[("r s3", proc.pid)])
        assert all(_alive(p) for p in pids)
        assert list(attempts.Running(tmp_path).records()) == [token]
        recs[token]["orchestrator_start_s"] -= 1.0
        (tmp_path / attempts.RUNNING_FILE).write_text(json.dumps(recs))
        found = attempts.stop_survivors(tmp_path, grace_s=1.0)
        assert found.stopped == [("r s3", sorted(pids))] and not found.alive
    finally:
        attempts.stop(token, grace_s=5)
        proc.wait()


def test_a_zombie_orchestrator_counts_as_dead():
    zombie = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        assert wait_for(lambda: kfd.proc_info(zombie.pid).state == "Z", 10)
        info = kfd.proc_info(zombie.pid)
        assert info.exited and not info.gone
        rec = {"orchestrator_pid": zombie.pid, "orchestrator_start_s": info.start_s}
        assert not attempts.orchestrator_running(rec)
    finally:
        zombie.wait()
    assert kfd.proc_info(zombie.pid).exited
    me = {"orchestrator_pid": os.getpid(), "orchestrator_start_s": None}
    assert not attempts.orchestrator_running(me)
    parent = os.getppid()
    assert attempts.orchestrator_running(
        {
            "orchestrator_pid": parent,
            "orchestrator_start_s": kfd.proc_info(parent).start_s,
        }
    )


@pytest.mark.skipif(procs.die_with_parent() is None, reason="needs prctl")
def test_a_child_started_with_die_with_parent_ends_with_its_parent(tmp_path):
    parent = textwrap.dedent(
        f"""
        import subprocess, sys, time
        sys.path.insert(0, {str(PIPELINE_DIR)!r})
        from lib import procs
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)"],
            preexec_fn=procs.die_with_parent(),
            start_new_session=True,
        )
        print(child.pid, flush=True)
        time.sleep(600)
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", parent], stdout=subprocess.PIPE, text=True
    )
    child = int(proc.stdout.readline())
    assert _alive(child)
    proc.kill()
    proc.wait()
    assert wait_for(lambda: not _alive(child), 10)


POOL_PARENT = """
import concurrent.futures as cf
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, {root!r})
from lib import procs

if __name__ == "__main__":
    with cf.ProcessPoolExecutor(
        2,
        mp_context=mp.get_context("spawn"),
        initializer=procs.die_with,
        initargs=(os.getpid(),),
    ) as ex:
        print(*sorted({{ex.submit(os.getpid).result() for _ in range(8)}}), flush=True)
        time.sleep(600)
"""


@pytest.mark.skipif(procs.die_with_parent() is None, reason="needs prctl")
def test_pool_workers_that_call_die_with_end_with_their_parent(tmp_path):
    """stage05's training workers: spawned by a pool, they take no
    preexec_fn, and its initializer ties them to the stage."""
    script = tmp_path / "pool_parent.py"
    script.write_text(POOL_PARENT.format(root=str(PIPELINE_DIR)))
    proc = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, text=True
    )
    try:
        workers = [int(p) for p in proc.stdout.readline().split()]
        assert workers and all(_alive(w) for w in workers)
        proc.kill()
        proc.wait()
        assert wait_for(lambda: not any(_alive(w) for w in workers), 10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
