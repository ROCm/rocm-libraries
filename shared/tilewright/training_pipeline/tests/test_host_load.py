# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import threading
import time

import pytest

import gpu_fakes as gf
from gpu_fakes import FakeStat
from lib import host_load as hl


@pytest.fixture
def proc_root(tmp_path, monkeypatch):
    root = tmp_path / "proc"
    root.mkdir()
    monkeypatch.setenv(hl.PROC_ROOT_VAR, str(root))
    return root


def test_cpu_times_count_iowait_as_idle(proc_root):
    (proc_root / "stat").write_text(
        "cpu  9 9 9 9 9 9 9 9\ncpu0 10 0 5 80 5 0 0 0 0 0\ncpu3 1 1 1 1 1 1 1 1\n"
    )
    assert hl.cpu_times([0, 3, 7]) == {0: (15, 100), 3: (6, 8)}


def test_a_window_subtracts_the_stages_own_time_from_the_mean():
    before = {0: (0, 0), 1: (0, 0)}
    after = {0: (90, 100), 1: (10, 100)}
    s = hl.window_sample(before, after, own_jiffies=40, seconds=1.0)
    assert s.mean_busy == pytest.approx(0.5) and s.max_core_busy == pytest.approx(0.9)
    assert s.own_share == pytest.approx(0.2) and s.foreign == pytest.approx(0.3)
    assert hl.window_sample({}, after, 0, 1.0) is None


def _proc(root, pid, ppid, times, cmd=None, start=None):
    d = root / str(pid)
    d.mkdir(exist_ok=True)
    tail = ["0"] * 4 + [str(start if start is not None else pid)] + ["0"] * 25
    rest = ["S", str(ppid)] + ["0"] * 9 + [str(t) for t in times] + tail
    (d / "stat").write_text(f"{pid} (a (b) c) " + " ".join(rest) + "\n")
    if cmd is not None:
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode() + b"\0")


def test_the_own_tree_counts_live_descendants_and_what_they_reaped(proc_root):
    _proc(proc_root, 10, 1, (1, 2, 3, 4))
    _proc(proc_root, 11, 10, (5, 5, 0, 0))
    _proc(proc_root, 12, 11, (1, 0, 0, 0))
    _proc(proc_root, 20, 1, (100, 100, 100, 100))
    assert hl.tree_cpu(10) == 10 + 10 + 1
    assert hl.tree_cpu(12) == 1
    assert hl.tree_cpu(99) == 0


def test_the_runs_bench_stages_are_its_own_load_and_its_other_work_is_not(
    proc_root,
):
    stage = "python stages/stage03_bench.py --devices"
    _proc(proc_root, 10, 1, (1, 1, 0, 0), "python run_orchestrator.py")
    _proc(proc_root, 11, 10, (2, 0, 0, 0), f"{stage} 0")
    _proc(proc_root, 12, 11, (5, 0, 0, 0), "bench")
    _proc(proc_root, 20, 10, (3, 0, 1, 0), f"{stage} 1")
    _proc(proc_root, 21, 20, (7, 0, 0, 0), "bench")
    _proc(proc_root, 30, 10, (100, 0, 0, 0), "python stages/stage05_train.py")
    _proc(proc_root, 40, 1, (50, 0, 0, 0), f"{stage} 2")
    assert hl.bench_trees(11) == {(11, "11"): 7}
    assert hl.bench_trees(11, 10, "stage03_bench.py") == {
        (11, "11"): 7,
        (20, "20"): 11,
    }
    assert hl.bench_trees(10, 10, "stage03_bench.py") == {(10, "10"): 2 + 7 + 11 + 100}
    m = hl.Monitor([0], 11, hl.Settings(), run_root=10, marker="stage03_bench.py")
    assert m._own() == hl.bench_trees(11, 10, "stage03_bench.py")
    assert m.summary()["run_root"] == 10


def test_own_time_counts_growth_and_new_trees_but_takes_none_back():
    before = {(11, "a"): 7, (20, "b"): 10}
    assert hl.own_jiffies(before, {(11, "a"): 9, (20, "b"): 10, (40, "c"): 3}) == 5
    assert hl.own_jiffies(before, {(11, "a"): 9}) == 2
    assert hl.own_jiffies(before, {(11, "a"): 9, (20, "z"): 4}) == 6


def test_the_run_root_must_be_this_process_or_an_ancestor(monkeypatch):
    monkeypatch.delenv(hl.RUN_ROOT_VAR, raising=False)
    assert hl.run_root() is None
    monkeypatch.setenv(hl.RUN_ROOT_VAR, str(os.getppid()))
    assert hl.run_root() == os.getppid()
    assert hl.run_root(environ={hl.RUN_ROOT_VAR: str(os.getpid())}) == os.getpid()
    assert hl.run_root(environ={}) is None
    with gf.outside_process() as pid:
        assert hl.run_root(environ={hl.RUN_ROOT_VAR: str(pid)}) is None
    for bad in ("x", "-3", "0"):
        assert hl.run_root(environ={hl.RUN_ROOT_VAR: bad}) is None


def _monitor(cores, **kw):
    settings = hl.Settings(every_s=0.2, **kw)
    return hl.Monitor(cores, 999999, settings, first_window_s=0.1)


def test_a_quiet_window_lets_the_gate_through(proc_root):
    fake = FakeStat(proc_root, [0, 1], rate=0.6)
    with fake:
        m = _monitor([0, 1]).start()
        try:
            done = threading.Event()
            threading.Thread(target=lambda: (m.wait_quiet(), done.set())).start()
            assert not done.wait(0.8)
            mark = m.mark()
            assert m.hot_since(0) >= 2
            fake.set(0.05)
            assert not m.quiet_now()
            assert done.wait(5.0)
            assert m.latest().foreign <= 0.15 and m.quiet_now()
            assert m.hot_since(m.mark()) == 0 and mark >= 2
        finally:
            m.stop()
    s = m.summary()
    assert s["waits"] == 1 and s["waited_s"] > 0 and s["hot"] >= 2
    assert s["cores"] == "0-1" and s["settings"]["every_s"] == 0.2


def test_the_gate_gives_up_after_its_wait(proc_root):
    with FakeStat(proc_root, [0], rate=0.9):
        m = _monitor([0], max_wait_s=0.5).start()
        try:
            t0 = time.monotonic()
            assert m.wait_quiet() is False
            assert m.gave_up and time.monotonic() - t0 < 5.0
            assert m.wait_quiet() is False
        finally:
            m.stop()
    assert m.summary()["gave_up"] is True


def test_one_busy_core_holds_the_gate_only_while_nothing_of_the_stage_runs():
    assert hl.Monitor([0, 1], 1, hl.Settings()).quiet(
        hl.Sample("t", 1.0, 0.2, 0.99, 0.0, 0.1)
    )
    m = hl.Monitor([0, 1], 1, hl.Settings(quiet_core_busy=0.3))
    busy_core = hl.Sample("t", 1.0, 0.2, 0.4, 0.0, 0.2)
    assert not m.quiet(busy_core)
    assert m.quiet(hl.Sample("t", 1.0, 0.2, 0.4, 0.1, 0.1))
    assert m.hot(hl.Sample("t", 1.0, 0.5, 0.5, 0.0, 0.3))
    assert not m.hot(hl.Sample("t", 1.0, 0.5, 0.5, 0.3, 0.2))


def test_cores_the_stat_file_does_not_give_never_hold_the_gate(proc_root):
    said = []
    settings = hl.Settings(every_s=0.2)
    m = hl.Monitor([5], 999999, settings, report=said.append, first_window_s=0.1)
    m.start()
    try:
        assert m.wait_quiet() is True and m.quiet_now()
    finally:
        m.stop()
    assert "gives none of the cores 5" in m.summary()["unmeasurable"]
    assert said == [
        f"{proc_root / 'stat'} gives none of the cores 5: the load on them is not "
        f"measured; every part starts"
    ]


def test_settings_come_from_the_config_over_the_defaults():
    s = hl.Settings.of({"bench": {"quiet_host": {"hot_busy": 0.4, "enabled": False}}})
    assert (s.hot_busy, s.enabled, s.every_s) == (0.4, False, hl.DEFAULTS["every_s"])
    assert hl.Settings.of({}) == hl.Settings()
    assert "aborted and benched again after 2 window(s)" in hl.Settings().describe()


def test_every_window_goes_to_the_series(proc_root, tmp_path):
    with FakeStat(proc_root, [0], rate=0.1):
        series = tmp_path / "host_load.jsonl"
        m = hl.Monitor(
            [0],
            999999,
            hl.Settings(every_s=0.2),
            series_path=series,
            first_window_s=0.1,
        ).start()
        try:
            deadline = time.monotonic() + 5
            while m.mark() < 3 and time.monotonic() < deadline:
                time.sleep(0.05)
        finally:
            m.stop()
    rows = series.read_text().splitlines()
    assert len(rows) >= 3 and '"foreign": 0.1' in rows[-1]
    assert m.window(1)["samples"] == m.mark() - 1
