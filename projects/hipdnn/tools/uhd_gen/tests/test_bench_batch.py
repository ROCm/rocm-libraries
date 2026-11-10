# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`generate --bench-batch`: many graphs per hipdnn_bench process, against a stand-in bench
executable that answers `--graph` and `--graph-list` as hipdnn_bench does."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("pandas")

# One immediate response per graph, as hipdnn_bench --collect-immediate writes it. Timings
# depend on the metric and on the GPU HIP_VISIBLE_DEVICES selects, which the response names
# as its device. A graph whose id names "crash" kills a --graph-list process when reached;
# one naming "slow" takes SLOW_SECONDS in a --graph-list process; one naming "pause" holds a
# --graph-list process for PAUSE_SECONDS before it answers; one naming "stall" starts a
# child that holds the output pipes, writes its own and the child's pid to stall.pids, then
# holds a --graph-list process for STALL_SECONDS; one naming "escape" does the same with a
# holder outside its process tree (started through a process that exits at once), whose
# pid it writes to escape.pid; one naming "latin" writes a log line that is not UTF-8 to
# stdout before its answer in a --graph-list process; one naming "poison" is measured but
# fails every graph a --graph-list process measures after it, as a broken device or handle
# would; one naming "bad" fails on its own. Every process start is appended to calls.txt
# beside the script.
_BENCH = r"""
import json, os, subprocess, sys, time
from pathlib import Path

SLOW_SECONDS = 1.2
PAUSE_SECONDS = 6
STALL_SECONDS = 120
argv = sys.argv[1:]
here = Path(__file__).parent
with open(here / "calls.txt", "a", encoding="utf-8") as calls:
    calls.write(("list" if "--graph-list" in argv else "single") + "\n")
metric = argv[argv.index("--ranking-metric") + 1]
ordinal = int(os.environ.get("HIP_VISIBLE_DEVICES") or 0)
poisoned = False
SLEEPER = [sys.executable, "-c", f"import time; time.sleep({STALL_SECONDS})"]
# Started by a process that exits at once, in a session of its own on POSIX: outside the
# tree of the process that started it.
ESCAPE = (
    "import os, subprocess, sys; "
    f"p = subprocess.Popen({SLEEPER!r}, start_new_session=os.name != 'nt'); "
    "open(sys.argv[1], 'w').write(str(p.pid))"
)


def answer(path, listed):
    global poisoned
    graph = json.loads(Path(path).read_text(encoding="utf-8"))
    if "crash" in graph["id"] and listed:
        sys.stdout.flush()
        os._exit(3)
    if "slow" in graph["id"] and listed:
        time.sleep(SLOW_SECONDS)
    if "pause" in graph["id"] and listed:
        time.sleep(PAUSE_SECONDS)
    if "stall" in graph["id"] and listed:
        # Inherits stdout and stderr, as a helper the bench or a wrapper started would.
        child = subprocess.Popen(SLEEPER)
        (here / "stall.pids").write_text(f"{os.getpid()} {child.pid}", encoding="utf-8")
        time.sleep(STALL_SECONDS)
    if "escape" in graph["id"] and listed:
        subprocess.run([sys.executable, "-c", ESCAPE, str(here / "escape.pid")], check=True)
        time.sleep(STALL_SECONDS)
    if "latin" in graph["id"] and listed:
        # A provider's log line in a locale encoding, not UTF-8.
        sys.stdout.flush()
        sys.stdout.buffer.write(b"caf\xe9 warning\n")
        sys.stdout.buffer.flush()
    reason = "hipErrorIllegalAddress" if poisoned else None
    poisoned = poisoned or ("poison" in graph["id"] and listed)
    if "bad" in graph["id"]:
        reason = "no plan for this graph"
    if reason:
        line = {"engine_id": 7, "is_valid": False, "skip_reason": reason}
        if listed:
            line["graph_path"] = path
        print(json.dumps(line), flush=True)
        print(f"Engine query/collection failed: {reason}", file=sys.stderr)
        return False
    average = 1.0 + graph["size"] / 10 + (0.5 if metric == "time" else 0.0) + ordinal
    line = {
        "engine_id": 7, "engine_name": "provider:engine7", "graph_id": graph["id"],
        "graph_key": graph["id"], "device_id": f"board{ordinal}",
        "device_uuid": f"uuid-of-gpu-{ordinal}",
        "arch": "gfx942", "metric": metric,
        "binding": {"engine": "provider:engine7", "role": "predict_engine", "arch": "gfx942",
                    "metric": metric, "selector_revision": "provider-1",
                    "trained_against": {"selector_revision": "provider-1"}},
        "features": {"graph.flops": 2e9 * (graph["size"] + 1), "device.cu_count": 120},
        "avgTimeMs": average, "robustMeanMs": average * 0.9, "stddevMs": 0.01, "iters": 30,
        "is_valid": True, "selection_mode": "immediate", "timing_statistic": "robustMeanMs",
    }
    if listed:
        line["graph_path"] = path
    print(json.dumps(line), flush=True)
    return True


if "--graph-list" in argv:
    listed = Path(argv[argv.index("--graph-list") + 1]).read_text(encoding="utf-8").split()
    sys.exit(0 if all([answer(path, True) for path in listed]) else 1)
sys.exit(0 if answer(argv[argv.index("--graph") + 1], False) else 1)
"""


def _bench(directory: Path) -> str:
    script = directory / "bench.py"
    script.write_text(_BENCH, encoding="utf-8")
    if os.name == "nt":
        wrapper = directory / "bench.cmd"
        wrapper.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        wrapper = directory / "bench"
        wrapper.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
        )
        wrapper.chmod(0o755)
    return str(wrapper)


def _world(tmp_path: Path, ids: list[str]) -> dict:
    tree = tmp_path / "descriptors"
    tree.mkdir()
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    for size, graph_id in enumerate(ids):
        (graphs / f"{size:02d}.json").write_text(
            json.dumps({"id": graph_id, "size": size}), encoding="utf-8"
        )
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    return {
        "tree": tree,
        "graphs": graphs,
        "bench": _bench(bench_dir),
        "calls": bench_dir / "calls.txt",
    }


def _collect(world: dict, output: Path, batch: int, *extra: str) -> int:
    from uhd_gen.__main__ import main

    if world["calls"].exists():
        world["calls"].unlink()
    return main(
        [
            "generate",
            "--collect-only",
            "--graphs",
            str(world["graphs"]),
            "--descriptor-tree",
            str(world["tree"]),
            "--engine-id",
            "7",
            "--role",
            "predict_engine",
            "--bench",
            world["bench"],
            "--bench-batch",
            str(batch),
            "--output-dir",
            str(output),
            *extra,
        ]
    )


def _rows(output: Path) -> dict:
    corpus = json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    return {row["benchmark"]: (row["avgTimeMs"], row["robustMeanMs"]) for row in corpus}


def _alive(pid: int) -> bool:
    """Whether process `pid` is still running (a zombie counts as exited)."""
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.is_file() and stat.read_text().split(") ")[-1].startswith("Z"))


def _gone(pids: list[int], within: float = 10.0) -> bool:
    """Whether every process of `pids` exits within `within` seconds."""
    deadline = time.monotonic() + within
    while any(_alive(pid) for pid in pids):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.2)
    return True


def _kill(pids: list[int]) -> None:
    """End what a test left running, whatever the outcome."""
    import signal
    import subprocess

    for pid in pids:
        if not _alive(pid):
            continue
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(pid, signal.SIGKILL)


def _pids(path: Path) -> list[int]:
    return [int(pid) for pid in path.read_text(encoding="utf-8").split()]


def test_a_batched_collection_measures_what_one_process_per_graph_does(
    tmp_path, monkeypatch
):
    """Eight graphs in batches of four take two processes, and give the same rows."""
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    world = _world(tmp_path, [f"g{i}" for i in range(8)])

    assert _collect(world, tmp_path / "single", 1) == 0
    single_calls = world["calls"].read_text(encoding="utf-8").split()
    assert _collect(world, tmp_path / "batched", 4) == 0
    batched_calls = world["calls"].read_text(encoding="utf-8").split()

    assert single_calls == ["single"] * 8
    assert batched_calls == ["list"] * 2
    assert _rows(tmp_path / "batched") == _rows(tmp_path / "single")
    assert len(_rows(tmp_path / "batched")) == 8


def test_a_batch_that_crashes_is_finished_one_graph_per_process(tmp_path, monkeypatch):
    """The graphs before the crash keep their batch answers; the crashing graph and the rest
    are measured alone, where only the graph that fails on its own is recorded as failed.
    """
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    world = _world(tmp_path, ["g0", "crash1", "bad2", "g3"])

    assert _collect(world, tmp_path / "out", 4, "--max-graph-failures", "0.5") == 0

    assert (
        world["calls"].read_text(encoding="utf-8").split() == ["list"] + ["single"] * 3
    )
    assert set(_rows(tmp_path / "out")) == {"g0", "crash1", "g3"}
    manifest = json.loads(
        (tmp_path / "out" / "collection_manifest.json").read_text(encoding="utf-8")
    )
    assert [Path(failure["source"]).name for failure in manifest["failed_graphs"]] == [
        "02.json"
    ]
    assert "no plan for this graph" in manifest["failed_graphs"][0]["error"]


def test_graphs_a_poisoned_batch_fails_are_measured_alone(tmp_path, monkeypatch):
    """A batch whose device or handle breaks fails every later graph; measured alone, they
    succeed, so none of them is recorded as failed."""
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    world = _world(tmp_path, ["g0", "poison1", "g2", "g3"])

    assert _collect(world, tmp_path / "out", 4) == 0

    assert (
        world["calls"].read_text(encoding="utf-8").split() == ["list"] + ["single"] * 2
    )
    assert set(_rows(tmp_path / "out")) == {"g0", "poison1", "g2", "g3"}
    manifest = json.loads(
        (tmp_path / "out" / "collection_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["failed_graphs"] == []


def test_a_line_that_is_not_utf8_does_not_cut_the_batch(tmp_path, monkeypatch):
    """The bench's output is decoded with replacement, so one byte that is not UTF-8 neither
    ends the batch nor costs the graphs after it their batch answers."""
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    world = _world(tmp_path, ["g0", "latin1", "g2", "g3"])

    assert _collect(world, tmp_path / "out", 4) == 0

    assert world["calls"].read_text(encoding="utf-8").split() == ["list"]
    assert set(_rows(tmp_path / "out")) == {"g0", "latin1", "g2", "g3"}


def test_a_batch_that_stalls_is_killed_with_what_it_started_at_the_timeout(
    tmp_path, monkeypatch
):
    """A graph that stalls gets the batch killed after --bench-timeout together with the
    child it started, which holds the output pipes; the graphs the batch left are measured
    alone."""
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    world = _world(tmp_path, ["g0", "stall1", "g2", "g3"])
    pids_file = world["calls"].with_name("stall.pids")

    try:
        started = time.monotonic()
        assert _collect(world, tmp_path / "out", 4, "--bench-timeout", "2") == 0
        elapsed = time.monotonic() - started

        assert (
            elapsed < 45
        ), f"the run waited {elapsed:.0f} s for the stalled batch's child"
        assert _gone(
            _pids(pids_file)
        ), "the stalled bench or its child outlived the run"
        assert (
            world["calls"].read_text(encoding="utf-8").split()
            == ["list"] + ["single"] * 3
        )
        assert len(_rows(tmp_path / "out")) == 4
    finally:
        if pids_file.exists():
            _kill(_pids(pids_file))


def test_a_run_returns_while_a_process_outside_its_tree_holds_the_pipes(
    tmp_path, monkeypatch
):
    """A process the kill cannot reach (it left the bench's tree) keeps the output pipes
    open; the run still returns within seconds of the timeout and measures the rest."""
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    world = _world(tmp_path, ["g0", "escape1", "g2", "g3"])
    pid_file = world["calls"].with_name("escape.pid")

    try:
        started = time.monotonic()
        assert _collect(world, tmp_path / "out", 4, "--bench-timeout", "2") == 0
        elapsed = time.monotonic() - started

        assert elapsed < 45, f"the run waited {elapsed:.0f} s for a pipe holder"
        assert len(_rows(tmp_path / "out")) == 4
    finally:
        if pid_file.exists():
            _kill(_pids(pid_file))


def test_a_graph_that_stalls_past_its_own_timeout_ends_the_batch(tmp_path, monkeypatch):
    """A 6 s pause under a 2 s limit ends a batch of four, although the batch's share of
    the limit (8 s) would outlast it: the deadline is per graph."""
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    world = _world(tmp_path, ["g0", "pause1", "g2", "g3"])

    assert _collect(world, tmp_path / "out", 4, "--bench-timeout", "2") == 0

    assert (
        world["calls"].read_text(encoding="utf-8").split() == ["list"] + ["single"] * 3
    )
    assert len(_rows(tmp_path / "out")) == 4


def test_an_interrupted_run_kills_the_bench_and_what_it_started(tmp_path):
    """Ctrl+C during a run reaches the caller within seconds, long before the run's 60 s
    deadline, and kills the bench and its child before it propagates; on POSIX the bench is
    in a session of its own and would not receive it."""
    import _thread
    import threading

    from uhd_gen.generate import _run_bench

    world = _world(tmp_path, ["stall0"])
    listing = tmp_path / "graphs.txt"
    listing.write_text(f"{next(world['graphs'].iterdir())}\n", encoding="utf-8")
    pids_file = world["calls"].with_name("stall.pids")
    # The stand-in reads its GPU ordinal from HIP_VISIBLE_DEVICES; run it on the default.
    environment = {k: v for k, v in os.environ.items() if k != "HIP_VISIBLE_DEVICES"}
    finished = threading.Event()
    fired: list[float] = []

    def interrupt_once_stalled() -> None:
        deadline = time.monotonic() + 30
        while not pids_file.exists() and time.monotonic() < deadline:
            if finished.wait(0.1):
                return
        if not finished.is_set():
            fired.append(time.monotonic())
            _thread.interrupt_main()

    interrupter = threading.Thread(target=interrupt_once_stalled, daemon=True)
    interrupter.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            _run_bench(
                [
                    world["bench"],
                    "--graph-list",
                    str(listing),
                    "--ranking-metric",
                    "tflops",
                ],
                environment,
                tmp_path / "logs",
                0,
                [],
                timeout=60,
                per_line=True,
            )
        latency = time.monotonic() - fired[0]
        assert latency < 10, f"the interrupt took {latency:.0f} s to reach the caller"
        assert _gone(
            _pids(pids_file)
        ), "the interrupted bench or its child kept running"
    finally:
        finished.set()
        interrupter.join(timeout=5)
        if pids_file.exists():
            _kill(_pids(pids_file))


def test_a_batch_longer_than_the_timeout_runs_while_each_graph_answers_in_time(
    tmp_path, monkeypatch
):
    """--bench-timeout bounds each graph of a batch, not the batch: five graphs of 1.2 s
    each take 6 s under a 3 s limit and still finish in one process."""
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    world = _world(tmp_path, [f"slow{i}" for i in range(5)])

    assert _collect(world, tmp_path / "out", 5, "--bench-timeout", "3") == 0

    assert world["calls"].read_text(encoding="utf-8").split() == ["list"]
    assert len(_rows(tmp_path / "out")) == 5


def test_each_metric_of_a_batch_keeps_its_rows_on_their_graphs(tmp_path, monkeypatch):
    """A graph that fails on the first metric is left out of the second metric's batch;
    each metric's corpus holds every other graph's own row, measured for that metric."""
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    world = _world(tmp_path, ["g0", "bad1", "g2", "g3"])

    assert (
        _collect(
            world,
            tmp_path / "out",
            4,
            "--metric",
            "tflops",
            "time",
            "--max-graph-failures",
            "0.5",
        )
        == 0
    )

    # The stand-in adds 0.5 ms to every timing it takes for `time`.
    for metric, offset in (("tflops", 0.0), ("time", 0.5)):
        corpus = json.loads(
            (tmp_path / "out" / f"corpus_{metric}.json").read_text(encoding="utf-8")
        )
        assert {row["benchmark"]: row["avgTimeMs"] for row in corpus} == {
            "g0": 1.0 + offset,
            "g2": 1.2 + offset,
            "g3": 1.3 + offset,
        }, metric


def test_each_device_of_a_batch_measures_on_its_own_gpu(tmp_path, monkeypatch):
    """Each `--device` gets its own batch on that GPU: every graph has one row per GPU,
    with that GPU's timing."""
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    world = _world(tmp_path, ["g0", "g1", "g2", "g3"])

    assert _collect(world, tmp_path / "out", 4, "--device", "0", "--device", "1") == 0

    assert world["calls"].read_text(encoding="utf-8").split() == ["list"] * 2
    corpus = json.loads((tmp_path / "out" / "corpus.json").read_text(encoding="utf-8"))
    # The stand-in adds the GPU ordinal, in ms, to every timing.
    assert {(row["benchmark"], row["device"]): row["avgTimeMs"] for row in corpus} == {
        (graph, f"board{gpu}"): 1.0 + index / 10 + gpu
        for index, graph in enumerate(["g0", "g1", "g2", "g3"])
        for gpu in (0, 1)
    }


def test_a_catalog_sweep_is_not_batched(tmp_path, caplog):
    from uhd_gen.__main__ import main

    world = _world(tmp_path, ["g0"])
    code = main(
        [
            "generate",
            "--collect-only",
            "--graphs",
            str(world["graphs"]),
            "--descriptor-tree",
            str(world["tree"]),
            "--engine-id",
            "7",
            "--role",
            "sort_kernel_catalog",
            "--bench",
            world["bench"],
            "--bench-batch",
            "4",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    assert "--bench-batch batches engine-level" in caplog.text
