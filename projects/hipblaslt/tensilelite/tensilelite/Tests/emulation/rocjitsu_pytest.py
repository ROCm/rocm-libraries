# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""pytest plugin for TensileLite tests under the rocjitsu emulator.

Load with ``-p rocjitsu_pytest`` (run_rocjitsu.py puts this directory on
PYTHONPATH). It records progress as tests finish, selects this CI shard's
tests by recorded duration (SHARD_INDEX/TOTAL_SHARDS, 1-based, as TheRock's
test jobs export them), and starts the longest tests first.
"""

import json
import os
from pathlib import Path

import pytest

DEFAULT_DURATIONS = Path(__file__).with_name("rocjitsu_tensilelite_durations.json")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Progress:
    def __init__(self, directory):
        self.path = directory / "progress.json"
        self.items = []
        self.reports = {}

    def save(self):
        counts = dict(passed=0, failed=0, skipped=0, incomplete=0)
        results = {}
        for nodeid in self.items:
            phases = self.reports.get(nodeid, {})
            if any(p["outcome"] == "failed" for p in phases.values()):
                outcome = "failed"
            elif "teardown" not in phases:
                outcome = "incomplete"
            elif any(p["outcome"] == "skipped" for p in phases.values()):
                outcome = "skipped"
            elif phases.get("call", {}).get("outcome") == "passed":
                outcome = "passed"
            else:
                outcome = "incomplete"
            counts[outcome] += 1
            results[nodeid] = dict(outcome=outcome, phases=phases)
        finished = sum("teardown" in self.reports.get(item, {}) for item in self.items)
        write_json(
            self.path,
            dict(selected=len(self.items), finished=finished, **counts, tests=results),
        )

    def pytest_collection_finish(self, session):
        if session.items:
            self.items = [item.nodeid for item in session.items]
            self.save()

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):
        self.items = ids
        self.save()

    def pytest_runtest_logreport(self, report):
        phases = self.reports.setdefault(report.nodeid, {})
        phases[report.when] = dict(outcome=report.outcome, seconds=report.duration)
        self.save()


def duration_key(nodeid):
    """Drop the file path so recorded durations survive a different rootdir."""
    return nodeid.rpartition("::")[2]


def longest_paired_with_shortest(items, durations):
    """Order tests as longest, shortest, next longest, next shortest, ...

    pytest-xdist hands each worker a run of consecutive tests up front. Pairing
    every long test with a short one starts the long tests first and keeps them
    from queuing behind each other, which otherwise sets the suite's wall time.
    Tests without a recorded duration count as the longest.
    """
    longest = max(durations.values(), default=0)
    ranked = sorted(
        items,
        key=lambda item: (
            -durations.get(duration_key(item.nodeid), longest),
            item.nodeid,
        ),
    )
    order = []
    while ranked:
        order.append(ranked.pop(0))
        if ranked:
            order.append(ranked.pop())
    return order


def shard_items(items, durations, index, total):
    """Return the items of 1-based shard `index` out of `total`.

    Longest-processing-time first: each test, longest first, joins the shard
    with the least recorded time so far, so the multi-hour tests land in
    different shards. Tests without a recorded duration count as the mean.
    Every xdist worker computes the same split from the same inputs.
    """
    known = [durations[k] for k in map(duration_key, (i.nodeid for i in items)) if k in durations]
    default = sum(known) / len(known) if known else 1
    ranked = sorted(
        items,
        key=lambda item: (-durations.get(duration_key(item.nodeid), default), item.nodeid),
    )
    loads = [0.0] * total
    selected = []
    for item in ranked:
        shard = min(range(total), key=lambda s: (loads[s], s))
        loads[shard] += durations.get(duration_key(item.nodeid), default)
        if shard == index - 1:
            selected.append(item)
    return selected


def env_shard():
    """(index, total) from SHARD_INDEX/TOTAL_SHARDS, or (1, 1) when unset."""
    total = int(os.environ.get("TOTAL_SHARDS") or 1)
    index = int(os.environ.get("SHARD_INDEX") or 1)
    if total < 1 or not 1 <= index <= total:
        raise pytest.UsageError(f"invalid shard {index} of {total}")
    return index, total


def pytest_addoption(parser):
    parser.addoption(
        "--rocjitsu-report-dir", default=os.environ.get("ROCJITSU_REPORT_DIR")
    )
    parser.addoption(
        "--rocjitsu-durations",
        default=os.environ.get("ROCJITSU_DURATIONS", str(DEFAULT_DURATIONS)),
        help="JSON of test id to seconds from an earlier run, used to shard and order tests",
    )


# Every xdist worker collects separately, so this runs in each of them and must
# produce the same selection and order; it runs last to see only selected tests.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    path = config.getoption("--rocjitsu-durations")
    durations = json.loads(Path(path).read_text()) if path and Path(path).is_file() else {}
    index, total = env_shard()
    if total > 1:
        keep = shard_items(items, durations, index, total)
        kept = set(map(id, keep))
        config.hook.pytest_deselected(items=[i for i in items if id(i) not in kept])
        items[:] = keep
    if durations:
        items[:] = longest_paired_with_shortest(items, durations)


def pytest_configure(config):
    directory = config.getoption("--rocjitsu-report-dir")
    if directory and not hasattr(config, "workerinput"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        progress = Progress(directory)
        progress.save()
        config.pluginmanager.register(progress)
