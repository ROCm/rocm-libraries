# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import shutil
import threading
import time

import pytest

import gpu_fakes as gf
from gpu_fakes import FakeStat
from lib import host_load as hl
from lib import kfd


@pytest.fixture
def proc_root(tmp_path, monkeypatch):
    root = tmp_path / "proc"
    root.mkdir()
    monkeypatch.setenv(hl.PROC_ROOT_VAR, str(root))
    return root


@pytest.fixture
def sys_root(tmp_path, monkeypatch):
    root = tmp_path / "sys"
    root.mkdir()
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    return root


def _smt(sys_root, cores):
    """A fake CPU topology: each CPU of `cores` with its SMT siblings."""
    for core in cores:
        for c in core:
            d = sys_root / "devices" / "system" / "cpu" / f"cpu{c}" / "topology"
            d.mkdir(parents=True, exist_ok=True)
            (d / "thread_siblings_list").write_text(",".join(map(str, core)) + "\n")


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


def _proc(root, pid, ppid, times, cmd=None, start=None, threads=None, cpus=None):
    """A fake process: with `threads`, a thread per letter in that state;
    with `cpus`, the CPU list it may run on."""
    d = root / str(pid)
    d.mkdir(exist_ok=True)
    tail = ["0"] * 4 + [str(start if start is not None else pid)] + ["0"] * 25
    rest = ["S", str(ppid)] + ["0"] * 9 + [str(t) for t in times] + tail
    (d / "stat").write_text(f"{pid} (a (b) c) " + " ".join(rest) + "\n")
    if cmd is not None:
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode() + b"\0")
    if threads is not None:
        _threads(root, pid, threads)
    if cpus is not None:
        (d / "status").write_text(f"Name:\ta\nCpus_allowed_list:\t{cpus}\n")


def _threads(root, pid, states):
    """Process `pid`'s threads, one per letter of `states`."""
    task = root / str(pid) / "task"
    shutil.rmtree(task, ignore_errors=True)
    for i, state in enumerate(states):
        t = task / str(pid * 100 + i)
        t.mkdir(parents=True)
        (t / "stat").write_text(f"{pid * 100 + i} (a (b) c) {state} 1 0 0\n")


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


# ── the run's benches beyond their CPUs ──────────────────────────────────────


def test_a_bench_running_more_threads_than_its_cpus_counts_the_excess(proc_root):
    _proc(proc_root, 10, 1, (1, 0, 0, 0), "python stages/stage03_bench.py")
    _proc(proc_root, 11, 10, (0, 0, 0, 0), "bench", threads="RRRRRRS", cpus="0-3")
    _proc(proc_root, 13, 11, (0, 0, 0, 0), "cotenant", threads="RRS")
    _proc(proc_root, 12, 10, (0, 0, 0, 0), "bench", threads="RRS", cpus="4-7")
    # bench 11 runs eight threads (its own six and its child's two) on four CPUs
    assert hl.excess_threads(range(8), 10) == 4
    # an even split of the cores among the benches caps a share
    _proc(proc_root, 14, 10, (0, 0, 0, 0), "bench", threads="RR", cpus="0-7")
    assert hl.excess_threads(range(8), 10) == 6
    # a bench that may run on none of the cores counts for nothing there
    assert hl.excess_threads([8, 9], 10) == 0
    # a bench whose CPUs are not known has all of the cores
    _proc(proc_root, 20, 1, (0, 0, 0, 0), "python stages/stage03_bench.py")
    _proc(proc_root, 21, 20, (0, 0, 0, 0), "bench", threads="RRRRR")
    assert hl.excess_threads([0, 1, 2, 3], 20) == 1
    assert hl.excess_threads(range(8), 99) == 0


def test_the_benches_of_the_runs_other_bench_stages_count_too(proc_root):
    stage = "python stages/stage03_bench.py --devices"
    _proc(proc_root, 10, 1, (0, 0, 0, 0), "python run_orchestrator.py")
    _proc(proc_root, 11, 10, (0, 0, 0, 0), f"{stage} 0")
    _proc(proc_root, 12, 11, (0, 0, 0, 0), "bench", threads="RRRR", cpus="0-3")
    _proc(proc_root, 20, 10, (0, 0, 0, 0), f"{stage} 1")
    _proc(proc_root, 21, 20, (0, 0, 0, 0), "bench", threads="RRRR", cpus="0-3")
    _proc(proc_root, 30, 10, (0, 0, 0, 0), "python stages/stage05_train.py")
    _proc(proc_root, 31, 30, (0, 0, 0, 0), "worker", threads="RRRRRRRR")
    # alone, the stage's bench has its four CPUs; beside the run's other
    # bench on the same CPUs each has half of them
    assert hl.excess_threads(range(4), 11) == 0
    assert hl.excess_threads(range(4), 11, 10, "stage03_bench.py") == 4


def test_the_benches_threads_beyond_their_cpus_are_other_load():
    m = hl.Monitor([0, 1, 2, 3], 1, hl.Settings())
    busy_by_design = hl.Sample("t", 1.0, 0.9, 0.9, 0.9, 0.0)
    assert m.quiet(busy_by_design) and not m.hot(busy_by_design)
    crowded = hl.Sample("t", 1.0, 0.9, 0.9, 0.9, 0.0, excess_threads=2, over=0.5)
    assert crowded.other == pytest.approx(0.5)
    assert not m.quiet(crowded) and m.hot(crowded)
    assert crowded.to_json()["over"] == 0.5 and crowded.to_json()["over_threads"] == 2
    assert "threads beyond their CPUs" in hl.Settings().describe()


def test_a_window_counts_the_excess_of_both_its_ends(proc_root):
    _proc(proc_root, 10, 1, (0, 0, 0, 0), "python stages/stage03_bench.py")
    _proc(proc_root, 11, 10, (0, 0, 0, 0), "bench", threads="RRRR", cpus="0-1")
    with FakeStat(proc_root, [0, 1], rate=0.0):
        m = hl.Monitor([0, 1], 10, hl.Settings(every_s=0.2), first_window_s=0.1)
        m.start()
        try:
            deadline = time.monotonic() + 5
            while m.mark() < 2 and time.monotonic() < deadline:
                time.sleep(0.05)
            s = m.latest()
            assert (s.excess_threads, s.over) == (2, pytest.approx(1.0))
            assert m.hot(s) and not m.quiet_now()
            _threads(proc_root, 11, "RS")
            mark = m.mark()
            while m.mark() < mark + 2 and time.monotonic() < deadline + 5:
                time.sleep(0.05)
            assert m.latest().excess_threads == 0 and m.quiet_now()
        finally:
            m.stop()
    assert m.summary()["over_max"] == pytest.approx(1.0)


# ── where the benches run ────────────────────────────────────────────────────


def test_cpu_cores_keep_smt_siblings_together(sys_root):
    _smt(sys_root, [(0, 4), (1, 5), (2, 6), (3, 7)])
    assert hl.cpu_cores(range(8)) == [(0, 4), (1, 5), (2, 6), (3, 7)]
    assert hl.cpu_cores([4, 1, 0]) == [(0, 4), (1,)]
    assert hl.cpu_cores([9, 8]) == [(8,), (9,)]


def test_each_gpu_gets_whole_cores_of_the_cpus_local_to_it(sys_root):
    _smt(sys_root, [(0, 4), (1, 5), (2, 6), (3, 7)])
    node_a, node_b = {0, 1, 4, 5}, {2, 3, 6, 7}
    local = {10: node_a, 11: node_a, 12: node_b, 13: node_b}
    assert hl.device_cpus([12, 10, 13, 11], range(8), local) == (
        {10: (0, 4), 11: (1, 5), 12: (2, 6), 13: (3, 7)},
        "local",
    )
    # the partitions of one node split its cores, the first a core more
    one_node = dict.fromkeys([10, 11, 12], set(range(8)))
    assert hl.device_cpus([10, 11, 12], range(8), one_node) == (
        {10: (0, 1, 4, 5), 11: (2, 6), 12: (3, 7)},
        "local",
    )
    # local CPUs outside the stage's mask leave it to an even split
    assert hl.device_cpus([10, 11], [2, 3, 6, 7], {10: node_a, 11: node_a}) == (
        {10: (2, 6), 11: (3, 7)},
        "even",
    )


def test_without_usable_local_cpus_the_mask_is_split_evenly(sys_root):
    node = {0, 1, 2, 3}
    for local in (
        None,
        {10: node},
        {10: node, 11: None},
        {10: node, 11: {3, 4, 5}},
        {10: {0}, 11: {0}},
    ):
        assert hl.device_cpus([10, 11], range(6), local) == (
            {10: (0, 1, 2), 11: (3, 4, 5)},
            "even",
        )
    assert hl.device_cpus([10, 11, 12], [0, 1]) == ({}, "none")
    assert hl.device_cpus([], range(4)) == ({}, "none")


def test_a_gpus_local_cpus_are_its_cards(sys_root):
    card = sys_root / "bus" / "pci" / "devices" / "0000:0d:00.0"
    card.mkdir(parents=True)
    (card / "local_cpulist").write_text("8-11,24-27\n")
    want = {8, 9, 10, 11, 24, 25, 26, 27}
    # a partition's own address names a function that is no GPU
    partition = kfd.Gpu(
        index=1, node=3, gpu_id=2, bdf="0000:0d:00.1", card="0000:0d:00.0"
    )
    assert hl.gpu_local_cpus(partition) == want
    assert (
        hl.gpu_local_cpus(kfd.Gpu(index=0, node=2, gpu_id=1, bdf="0000:0d:00.0"))
        == want
    )
    assert (
        hl.gpu_local_cpus(kfd.Gpu(index=2, node=4, gpu_id=3, bdf="0000:0e:00.0"))
        is None
    )
    assert hl.gpu_local_cpus(None) is None


def test_the_runs_gpus_come_from_the_environment():
    assert hl.run_gpus({hl.BENCH_GPUS_VAR: "6,0,2"}) == [0, 2, 6]
    for bad in ("", "x", "0,,1x", "-1"):
        assert hl.run_gpus({hl.BENCH_GPUS_VAR: bad}) is None
    assert hl.run_gpus({}) is None
