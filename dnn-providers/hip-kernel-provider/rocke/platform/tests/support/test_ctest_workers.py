# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.ctest_workers runs one CI shard's tests in worker processes:
every test the run selects (``-k``, ``-m``, the CI shard) runs in exactly one
worker, a run exits as the same run without workers does, a worker with no
share passes, a run that needs one process keeps it, each worker gets its own
temporary directory and share of the CPUs, and a worker stops when the process
that started it is killed.

The end-to-end cases run real pytest processes over a small test tree they
write, with ``rocke.tests.run_records`` recording what each process did. They
start two workers, so they run wherever this process may use two CPUs.
"""

from __future__ import annotations

import collections
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from rocke.tests import ci_shard
from rocke.tests.ctest_workers import (
    WORKERS_OPTION,
    combined_exit_status,
    worker_count,
    worker_cpus,
    worker_keys,
)
from rocke.tests.host_cpus import available_cpus
from rocke.tests.run_records import OTHER_SHARD, RECORDS_DIR_ENV, read_run_records
from rocke.tests.subprocess_launcher import rocke_subprocess_env

_CHILD_TIMEOUT_SECONDS = 30
_WORKERS = 2
_TREE = """
    import pytest

    @pytest.mark.parametrize("index", range(24))
    def test_case(index):
        pass

    @pytest.mark.rocke_shard_cost(5)
    @pytest.mark.parametrize("size", ["small", "large"])
    def test_heavy(size):
        pass

    @pytest.mark.integration
    def test_deselected():
        pass
"""

needs_two_cpus = pytest.mark.skipif(
    available_cpus() < _WORKERS, reason="starting two workers needs two CPUs"
)


def _write_tree(root: Path, source: str = _TREE) -> Path:
    tree = root / "tree"
    tree.mkdir()
    (tree / "test_cases.py").write_text(textwrap.dedent(source), encoding="utf-8")
    return tree


def _environment(records: Path | None = None, shard: str | None = None) -> dict:
    """A clean child environment: the plugins' variables unset, records into
    ``records`` (none when None), the CI shard ``"i/n"`` from the gtest pair
    (none when None)."""
    environment = rocke_subprocess_env()
    for name in ci_shard.CHILD_ENV_EXCLUDED:
        environment.pop(name, None)
    environment.pop(RECORDS_DIR_ENV, None)
    if records is not None:
        environment[RECORDS_DIR_ENV] = str(records)
        environment["PYTEST_PLUGINS"] = "rocke.tests.run_records"
    if shard is not None:
        index, _, total = shard.partition("/")
        environment[ci_shard.GTEST_INDEX_ENV] = index
        environment[ci_shard.GTEST_TOTAL_ENV] = total
    return environment


def _command(*arguments: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-p",
        "rocke.tests.ctest_shard",
        ci_shard.ACTIVATION_FLAG,
        "-p",
        "rocke.tests.ctest_workers",
        "-W",
        "ignore",
        *arguments,
    ]


def _run(
    tree: Path, records: Path | None, *arguments: str, shard: str | None = None
) -> subprocess.CompletedProcess:
    """pytest over ``tree`` with both shard plugins and ``arguments``."""
    return subprocess.run(
        _command(*arguments),
        cwd=tree,
        env=_environment(records, shard),
        capture_output=True,
        text=True,
        check=False,
        timeout=_CHILD_TIMEOUT_SECONDS,
    )


def _records(records: Path) -> collections.Counter[tuple[str, str, str]]:
    """How many times the run recorded each (test, event, reason)."""
    return collections.Counter(
        (record.node, record.event, record.reason)
        for record in read_run_records(records)
    )


def _ran(records: Path) -> collections.Counter[str]:
    """How many processes ran each test."""
    return collections.Counter(
        record.node for record in read_run_records(records) if record.event == "ran"
    )


@needs_two_cpus
@pytest.mark.parametrize("shard", [None, "1/3"])
def test_each_selected_test_runs_in_exactly_one_worker(tmp_path, shard):
    tree = _write_tree(tmp_path)
    alone = _run(tree, tmp_path / "alone", "-m", "not integration", shard=shard)
    split = _run(
        tree,
        tmp_path / "split",
        "-m",
        "not integration",
        WORKERS_OPTION,
        str(_WORKERS),
        shard=shard,
    )
    assert alone.returncode == split.returncode == 0, split.stdout[-3000:]
    ran_alone, ran_split = _ran(tmp_path / "alone"), _ran(tmp_path / "split")
    assert ran_alone and set(ran_split.values()) == {1}
    assert ran_split == ran_alone
    # Every record, each once: what collection found (the -m deselection with
    # its lane, the other CI shards' tests) as the one-process run has it.
    records_alone = _records(tmp_path / "alone")
    assert _records(tmp_path / "split") == records_alone
    assert set(records_alone.values()) == {1}
    events = {(node, event) for node, event, _ in records_alone}
    assert ("test_deselected", "deselected") in events
    if shard is not None:
        assert any(
            reason == OTHER_SHARD for _, event, reason in records_alone
        ), records_alone
    assert "rocke-ctest-workers: 2 workers exited [0, 0]" in split.stdout
    if shard is not None:
        # The shard header, unprefixed, as CI-log readers expect it.
        header = (
            f"rocke-ctest-shard: {shard} (GTEST_SHARD_INDEX=1, GTEST_TOTAL_SHARDS=3)"
        )
        assert header in split.stdout.splitlines()
    for index in range(_WORKERS):
        assert f"[w{index}] " in split.stdout
    assert split.stdout.count(" passed") == _WORKERS, split.stdout[-3000:]


@needs_two_cpus
def test_a_failing_test_fails_the_run_and_shows_its_output(tmp_path):
    tree = _write_tree(
        tmp_path,
        _TREE
        + """
    def test_fails():
        assert 1 == 2, "the failure this test reports"
    """,
    )
    result = _run(tree, None, WORKERS_OPTION, str(_WORKERS))
    assert result.returncode == pytest.ExitCode.TESTS_FAILED
    assert "the failure this test reports" in result.stdout
    assert "the run exits 1" in result.stdout


@needs_two_cpus
def test_a_worker_with_no_share_passes(tmp_path):
    tree = _write_tree(
        tmp_path,
        """
        def test_only():
            pass
        """,
    )
    result = _run(tree, tmp_path / "records", WORKERS_OPTION, str(_WORKERS))
    assert result.returncode == 0, result.stdout[-3000:]
    assert _ran(tmp_path / "records") == {"test_only": 1}
    assert result.stdout.count("kept 0 of 1") == _WORKERS - 1


@needs_two_cpus
def test_a_run_that_selects_nothing_exits_as_without_workers(tmp_path):
    tree = _write_tree(tmp_path)
    alone = _run(tree, None, "-k", "no_test_has_this_name")
    split = _run(
        tree, None, "-k", "no_test_has_this_name", WORKERS_OPTION, str(_WORKERS)
    )
    assert alone.returncode == split.returncode == pytest.ExitCode.NO_TESTS_COLLECTED


@needs_two_cpus
@pytest.mark.parametrize(
    "option",
    [["--collect-only"], ["-x"], ["--maxfail", "2"], ["--junitxml", "report.xml"]],
)
def test_a_run_that_needs_one_process_runs_in_it(tmp_path, option):
    tree = _write_tree(tmp_path)
    result = _run(tree, None, *option, WORKERS_OPTION, str(_WORKERS))
    assert result.returncode == 0, result.stdout[-3000:]
    assert "rocke-ctest-workers" not in result.stdout


@needs_two_cpus
def test_each_worker_gets_its_own_basetemp(tmp_path):
    tree = _write_tree(
        tmp_path,
        """
        import pytest
        from rocke.tests.ci_shard import current_worker

        @pytest.mark.parametrize("index", range(16))
        def test_writes_its_tmp_path(index, tmp_path):
            (tmp_path / "worker").write_text(str(current_worker()[0]))
        """,
    )
    basetemp = tmp_path / "basetemp"
    result = _run(tree, None, "--basetemp", str(basetemp), WORKERS_OPTION, "2")
    assert result.returncode == 0, result.stdout[-3000:]
    written = list(basetemp.glob("worker*/**/worker"))
    assert len(written) == 16
    for path in written:
        worker_dir = path.relative_to(basetemp).parts[0]
        assert path.read_text() == worker_dir.removeprefix("worker")
    assert sorted(path.name for path in basetemp.iterdir()) == ["worker0", "worker1"]


@needs_two_cpus
def test_a_process_a_worker_test_starts_is_no_worker(tmp_path):
    tree = _write_tree(
        tmp_path,
        """
        import os
        import pytest

        @pytest.mark.parametrize("index", range(8))
        def test_sees_no_worker_variables(index):
            assert not {
                name for name in os.environ if name.startswith("ROCKE_TEST_WORKER")
            }
        """,
    )
    result = _run(tree, None, WORKERS_OPTION, str(_WORKERS))
    assert result.returncode == 0, result.stdout[-3000:]


@needs_two_cpus
def test_each_worker_runs_on_its_share_of_the_cpus(tmp_path):
    tree = _write_tree(
        tmp_path,
        """
        import os
        import pytest

        @pytest.mark.parametrize("index", range(16))
        def test_reports_its_share(index, tmp_path_factory):
            cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
            report = tmp_path_factory.getbasetemp().parent / f"share-{os.getpid()}"
            report.write_text(
                f"{os.environ['OMP_NUM_THREADS']} {','.join(map(str, cpus))}"
            )
        """,
    )
    basetemp = tmp_path / "basetemp"
    result = _run(tree, None, "--basetemp", str(basetemp), WORKERS_OPTION, "2")
    assert result.returncode == 0, result.stdout[-3000:]
    share = max(1, available_cpus() // _WORKERS)
    inherited = os.environ.get("OMP_NUM_THREADS", "")
    threads = min(int(inherited), share) if inherited.isdigit() else share
    reports = [path.read_text().split(" ", 1) for path in basetemp.glob("share-*")]
    assert len(reports) == _WORKERS
    assert {reported for reported, _ in reports} == {str(threads)}
    if hasattr(os, "sched_getaffinity"):
        blocks = [cpus.split(",") for _, cpus in reports]
        assert all(len(cpus) == share for cpus in blocks)
        assert len({cpu for cpus in blocks for cpu in cpus}) == _WORKERS * share


@needs_two_cpus
@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc, kills with SIGKILL")
def test_a_worker_stops_when_the_run_is_killed(tmp_path):
    tree = _write_tree(
        tmp_path,
        """
        import os
        import time
        import pytest

        @pytest.mark.parametrize("index", range(32))
        def test_waits(index, tmp_path_factory):
            marker = tmp_path_factory.getbasetemp().parent / f"pid-{os.getpid()}"
            marker.write_text("")
            time.sleep(60)
        """,
    )
    basetemp = tmp_path / "basetemp"
    command = _command("--basetemp", str(basetemp), WORKERS_OPTION, str(_WORKERS))
    run = subprocess.Popen(
        command,
        cwd=tree,
        env=_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    workers: list[int] = []
    try:
        deadline = time.monotonic() + _CHILD_TIMEOUT_SECONDS
        while len(workers) < _WORKERS and time.monotonic() < deadline:
            time.sleep(0.2)
            workers = [int(p.name.removeprefix("pid-")) for p in basetemp.glob("pid-*")]
        assert len(workers) == _WORKERS
        run.send_signal(signal.SIGKILL)
        run.wait()
        deadline = time.monotonic() + 10
        while any(map(_running, workers)) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not [pid for pid in workers if _running(pid)]
    finally:
        run.kill()
        run.wait()
        for pid in workers:
            if _running(pid):
                os.kill(pid, signal.SIGKILL)


def _running(pid: int) -> bool:
    """Whether ``pid`` is a live process (a zombie of another parent counts as
    gone)."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


# --------------------------------------------------------------------- the rules


def test_the_keys_of_one_ci_shard_spread_over_the_workers():
    keys = [f"test_cases.py[{index}]" for index in range(400)]
    shard_keys = [key for key in keys if ci_shard.shard_of(key, 4) == 1]
    placement = worker_keys(shard_keys, {}, 4)
    counts = collections.Counter(placement.values())
    assert sorted(counts) == [0, 1, 2, 3]
    assert min(counts.values()) > len(shard_keys) // 8


def test_costed_keys_go_heaviest_first_to_the_least_loaded_worker():
    costs = {"heavy[a]": 9.0, "heavy[b]": 8.0, "heavy[c]": 1.0}
    placement = worker_keys(list(costs), costs, 2)
    assert placement["heavy[a]"] != placement["heavy[b]"]
    assert placement["heavy[c]"] == placement["heavy[b]"]


def test_the_worker_count_is_capped_by_the_usable_cpus():
    assert worker_count(4, 16) == 4
    assert worker_count(8, 3) == 3
    assert worker_count(1, 16) == 1
    with pytest.raises(
        pytest.UsageError,
        match="rocke-ctest-workers failed - --rocke-workers=0, expected at least 1",
    ):
        worker_count(0, 16)


def test_each_worker_runs_on_its_own_block_of_the_usable_cpus():
    cpus = list(range(32, 48))
    blocks = [worker_cpus(cpus, 16, index, 4) for index in range(4)]
    assert blocks == [cpus[0:4], cpus[4:8], cpus[8:12], cpus[12:16]]
    # Another run's rotation moves every block, keeping them apart.
    rotated = [worker_cpus(cpus, 16, index, 4, rotation=1) for index in range(4)]
    assert rotated == blocks[1:] + blocks[:1]
    # A quota narrower than the affinity sizes the blocks to the quota.
    assert [len(worker_cpus(range(64), 8, index, 4)) for index in range(4)] == [2] * 4
    assert worker_cpus([5, 6], 2, 1, 2) == [6]


@pytest.mark.parametrize(
    "statuses, combined",
    [
        ([0, 0, 0], 0),
        ([0, 1, 0], 1),
        ([1, 2, 0], 2),
        ([3, 1], 3),
        ([4, 1], 4),
        ([5, 5], 5),
        ([5, 0], 0),
        ([0, 7], 3),
        ([1, -11], 3),
    ],
)
def test_the_run_exits_with_the_most_severe_worker_status(statuses, combined):
    assert combined_exit_status(statuses) == combined


@pytest.mark.parametrize("text", ["1/1", "x/4", "2", "²/4", "-1/4"])
def test_a_malformed_worker_variable_is_refused(text):
    with pytest.raises(ValueError, match="current_worker failed - ROCKE_TEST_WORKER"):
        ci_shard.current_worker({ci_shard.WORKER_ENV: text})
    assert ci_shard.current_worker({ci_shard.WORKER_ENV: "1/4"}) == (1, 4)
    assert ci_shard.current_worker({}) is None
