# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What each test did in a run, one record per event, for an audit of what no
CI lane runs (``tools/limbo_audit.py``).

A pytest plugin. It does nothing unless ``ROCKE_RUN_RECORDS_DIR`` names a
directory; then every pytest process writes its own file of JSON lines there.
Load it through the environment, so that every pytest process of a run loads
it, the child pytest processes some tests start included, with no change to
any command::

    ROCKE_RUN_RECORDS_DIR=<dir> PYTEST_PLUGINS=rocke.tests.run_records ctest ...

A pytest a test starts over a fixture tree of its own takes its environment
from :func:`fixture_session_environment`, so it records nothing; one that runs
the product's test files records them, as the entry's own sessions do.

A record names the test file by its absolute path in the tree that ran it
(an installed tree or a source checkout) and the test by its pytest node name
after the file (``[Class::]test[params]``). Its ``event`` is one of:

- ``collected``: the test was collected (before any filter removed it);
- ``deselected``: a filter removed it; ``reason`` is the reason the plugin that
  removed it gave (:func:`note_deselection`), else the ``-m`` or ``-k``
  expression that rejects it (declaring the lanes, ``rocke.tests.lanes``,
  whose markers are what it rejects);
- ``ran``: it ran, or erred in setup or teardown; ``reason`` is how it ended,
  one of :data:`RAN_OUTCOMES` (a test whose teardown raises after its body
  ran has two ``ran`` records), so runs of one test can be compared;
- ``skipped``: it was skipped; ``reason`` is pytest's skip reason;
- ``module_skipped``: its whole file was skipped while collecting (no node
  name); ``reason`` is pytest's skip reason;
- ``collect_error``: its file failed to collect (no node name);
- ``unfinished``: the process that was to run it ended before its session
  finished (a per-test time limit that ends the process, a crash), before the
  test ended; ``reason`` is :data:`UNFINISHED`.

A test a CI shard leaves to another shard of its entry is ``deselected`` with
the reason :data:`OTHER_SHARD` (``rocke.tests.ctest_shard``): the run of that
other shard has its verdict. A test with a ``collected`` record and no other
was dropped by another filter.

Besides those, each process writes the tests it is to run once every filter
has run, and a last line when its session finishes; each line is written as
it happens. :func:`read_run_records` reads these to give the ``unfinished``
records and does not yield them.

A run in worker processes (``rocke.tests.ctest_workers``) records each test as
the same run in one process does: worker 0 alone records what collection finds
(``collected``, ``deselected``, ``module_skipped``, ``collect_error``), and each
worker records how the tests it runs end (``ran``, ``skipped``).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
from _pytest.mark import KeywordMatcher, MarkMatcher
from _pytest.mark.expression import Expression

from .ci_shard import current_worker
from .lanes import LANES, skip_until_reason

__all__ = [
    "EVENTS",
    "OTHER_SHARD",
    "RAN_OUTCOMES",
    "RECORDS_DIR_ENV",
    "UNFINISHED",
    "RunRecord",
    "fixture_session_environment",
    "note_deselection",
    "read_run_records",
]

# The directory each pytest process writes its records into; unset, the plugin
# does nothing.
RECORDS_DIR_ENV = "ROCKE_RUN_RECORDS_DIR"
EVENTS = frozenset(
    {
        "collected",
        "deselected",
        "ran",
        "skipped",
        "module_skipped",
        "collect_error",
        "unfinished",
    }
)
# The reason of an ``unfinished`` record.
UNFINISHED = "not run: its pytest process ended before its session finished"
# The lines a process writes that are not records: a test it is to run, and
# the end of its session (no file, no node name).
_SELECTED = "selected"
_SESSION_FINISHED = "session_finished"
# The deselection reason of a test a CI shard leaves to another shard.
OTHER_SHARD = "deselected: another shard of this entry runs it"
# How a ``ran`` test ended: its body passed or failed, an expected failure
# failed or passed (non-strict), or its setup or teardown raised.
RAN_OUTCOMES = frozenset({"passed", "failed", "xfailed", "xpassed", "error"})
_SKIP_PREFIX = "Skipped: "
_DESELECTION_REASON = pytest.StashKey[str]()


def note_deselection(item: pytest.Item, reason: str) -> None:
    """Say why a plugin deselects ``item``, before it calls the
    ``pytest_deselected`` hook with it, so the run records the reason."""
    item.stash[_DESELECTION_REASON] = reason


@dataclass(frozen=True)
class RunRecord:
    """One event of one test (or one test file) in a run."""

    file: str  # absolute path of the test file in the tree that ran it
    node: str  # pytest node name after the file; "" for a file-level event
    event: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.event not in EVENTS:
            raise ValueError(
                f"RunRecord failed - event={self.event!r}, expected one of "
                f"{sorted(EVENTS)}"
            )
        file_level = self.event in ("module_skipped", "collect_error")
        if file_level == bool(self.node):
            raise ValueError(
                f"RunRecord failed - event={self.event!r}, node={self.node!r}, "
                "expected a node name exactly for a test-level event"
            )


def fixture_session_environment(
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """A copy of ``env`` (default ``os.environ``) without
    ``ROCKE_RUN_RECORDS_DIR``, for a pytest that a test starts over a fixture
    tree of its own: that session's outcomes belong to the test, not to the
    run, so it records nothing. A child pytest that runs the product's test
    files keeps the variable, so the run records those tests."""
    env = os.environ if env is None else env
    return {name: value for name, value in env.items() if name != RECORDS_DIR_ENV}


def read_run_records(directory: Path) -> Iterator[RunRecord]:
    """Every record the processes of one run wrote into ``directory``, and an
    ``unfinished`` record for each test a process was to run but did not
    end, when that process ended before its session finished."""
    if not directory.is_dir():
        raise ValueError(
            f"read_run_records failed - directory={str(directory)!r}, expected the "
            f"directory {RECORDS_DIR_ENV} named for the run"
        )
    for path in sorted(directory.glob("*.jsonl")):
        yield from _process_records(path)


def _process_records(path: Path) -> Iterator[RunRecord]:
    """The records of the one process that wrote ``path``."""
    selected: dict[tuple[str, str], None] = {}
    ended: set[tuple[str, str]] = set()
    finished = False
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if not line.strip():
                continue
            fields = json.loads(line)
            event = fields["event"]
            if event == _SESSION_FINISHED:
                finished = True
                continue
            test = (fields["file"], fields["node"])
            if event == _SELECTED:
                selected[test] = None
                continue
            if event in ("ran", "skipped"):
                ended.add(test)
            yield RunRecord(**fields)
    if not finished:
        for file, node in selected:
            if (file, node) not in ended:
                yield RunRecord(file, node, "unfinished", UNFINISHED)


class _Recorder:
    """The plugin object of one pytest session: appends its records to a file
    of its own in the run's records directory."""

    def __init__(self, directory: Path) -> None:
        worker = current_worker()
        # Every worker of a run collects the same tests; one records them.
        self._records_collection = worker is None or worker[0] == 0
        directory.mkdir(parents=True, exist_ok=True)
        # Line-buffered: a process that a per-test time limit ends with
        # os._exit keeps every line written before it.
        self._out = (directory / f"{os.getpid()}-{uuid.uuid4().hex}.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )

    def _write(self, file: Path, node: str, event: str, reason: str = "") -> None:
        record = RunRecord(str(file.resolve()), node, event, reason)
        self._write_line(asdict(record))

    def _write_line(self, fields: dict[str, str]) -> None:
        self._out.write(json.dumps(fields) + "\n")

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, items: list[pytest.Item]):
        # Before every other filter, so a test a filter removes is still
        # recorded; after them all, the tests this process is to run.
        if self._records_collection:
            for item in items:
                self._write(item.path, _node_name(item), "collected")
        yield
        for item in items:
            self._write_line(
                {
                    "file": str(item.path.resolve()),
                    "node": _node_name(item),
                    "event": _SELECTED,
                }
            )

    def pytest_sessionfinish(self) -> None:
        self._write_line({"file": "", "node": "", "event": _SESSION_FINISHED})

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        if not self._records_collection:
            return
        for item in items:
            self._write(item.path, _node_name(item), "deselected", _deselection(item))

    @pytest.hookimpl(hookwrapper=True)
    def pytest_make_collect_report(self, collector: pytest.Collector):
        report = (yield).get_result()
        if (
            report.passed
            or not isinstance(collector, pytest.Module)
            or not self._records_collection
        ):
            return
        if report.skipped:
            self._write(collector.path, "", "module_skipped", _skip_reason(report))
        else:
            self._write(collector.path, "", "collect_error")

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item):
        report = (yield).get_result()
        node = _node_name(item)
        if hasattr(report, "wasxfail"):
            if report.when == "call":
                outcome = "xfailed" if report.skipped else "xpassed"
                self._write(item.path, node, "ran", outcome)
        elif report.skipped:
            # A skip in setup or in the test body: the test did not check what
            # it is for. (A skip in teardown comes after the call's report.)
            if report.when != "teardown":
                self._write(item.path, node, "skipped", _skip_reason(report))
        elif report.when == "call":
            self._write(item.path, node, "ran", report.outcome)
        elif report.failed:
            self._write(item.path, node, "ran", "error")

    def pytest_unconfigure(self) -> None:
        self._out.close()


def _node_name(item: pytest.Item) -> str:
    return item.nodeid.partition("::")[2]


def _deselection(item: pytest.Item) -> str:
    """Why ``item`` was deselected: the reason the plugin that removed it gave,
    else the ``-m`` or ``-k`` expression that rejects it (evaluated as pytest
    does), else an unnamed plugin."""
    noted = item.stash.get(_DESELECTION_REASON, None)
    if noted is not None:
        return noted
    option = item.config.option
    if option.markexpr:
        expression = Expression.compile(option.markexpr)
        markers = list(item.iter_markers())
        if not expression.evaluate(MarkMatcher.from_markers(markers)):
            return _marker_deselection(expression, option.markexpr, markers)
    if option.keyword and not Expression.compile(option.keyword).evaluate(
        KeywordMatcher.from_item(item)
    ):
        return f"deselected by -k {option.keyword!r}"
    return "deselected by a plugin that gave no reason"


def _marker_deselection(
    expression: Expression, markexpr: str, markers: list[pytest.Mark]
) -> str:
    """Why ``-m markexpr`` rejects a test carrying ``markers``: the expression,
    declaring the lanes whose markers it rejects when the test without them
    would be selected."""
    reason = f"deselected by -m {markexpr!r}"
    names = {marker.name for marker in markers}
    lanes = [lane for lane in LANES if lane.marker and lane.marker in names]
    lane_markers = {lane.marker for lane in lanes}
    if lanes and expression.evaluate(
        MarkMatcher.from_markers(
            [marker for marker in markers if marker.name not in lane_markers]
        )
    ):
        return skip_until_reason(*lanes, reason=reason)
    return reason


def _skip_reason(report: pytest.TestReport | pytest.CollectReport) -> str:
    """pytest's reason for a skip, without its \"Skipped: \" prefix."""
    longrepr = report.longrepr
    reason = longrepr[2] if isinstance(longrepr, tuple) else str(longrepr or "")
    return reason.removeprefix(_SKIP_PREFIX)


def pytest_configure(config: pytest.Config) -> None:
    directory = os.environ.get(RECORDS_DIR_ENV)
    if directory:
        config.pluginmanager.register(_Recorder(Path(directory)), "rocke-run-records")
