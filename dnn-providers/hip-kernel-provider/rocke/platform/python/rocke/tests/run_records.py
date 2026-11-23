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

A record names the test file by its absolute path in the tree that ran it
(an installed tree or a source checkout) and the test by its pytest node name
after the file (``[Class::]test[params]``). Its ``event`` is one of:

- ``collected``: the test was collected (before any filter removed it);
- ``deselected``: a filter removed it; ``reason`` is the reason the plugin that
  removed it gave (:func:`note_deselection`), else the ``-m`` or ``-k``
  expression that rejects it (declaring the lanes, ``rocke.tests.lanes``,
  whose markers are what it rejects);
- ``ran``: it ran (passed, failed, xfailed or xpassed), or erred in setup or
  teardown, which a run reports as a failure;
- ``skipped``: it was skipped; ``reason`` is pytest's skip reason;
- ``module_skipped``: its whole file was skipped while collecting (no node
  name); ``reason`` is pytest's skip reason;
- ``collect_error``: its file failed to collect (no node name).

A test with a ``collected`` record and no other was dropped by another filter
(a CI shard runs only its share of an entry's tests).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
from _pytest.mark import KeywordMatcher, MarkMatcher
from _pytest.mark.expression import Expression

from .lanes import LANES, skip_until_reason

__all__ = [
    "EVENTS",
    "RECORDS_DIR_ENV",
    "RunRecord",
    "note_deselection",
    "read_run_records",
]

# The directory each pytest process writes its records into; unset, the plugin
# does nothing.
RECORDS_DIR_ENV = "ROCKE_RUN_RECORDS_DIR"
EVENTS = frozenset(
    {"collected", "deselected", "ran", "skipped", "module_skipped", "collect_error"}
)
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


def read_run_records(directory: Path) -> Iterator[RunRecord]:
    """Every record the processes of one run wrote into ``directory``."""
    if not directory.is_dir():
        raise ValueError(
            f"read_run_records failed - directory={str(directory)!r}, expected the "
            f"directory {RECORDS_DIR_ENV} named for the run"
        )
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                if line.strip():
                    yield RunRecord(**json.loads(line))


class _Recorder:
    """The plugin object of one pytest session: appends its records to a file
    of its own in the run's records directory."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._out = (directory / f"{os.getpid()}-{uuid.uuid4().hex}.jsonl").open(
            "a", encoding="utf-8"
        )

    def _write(self, file: Path, node: str, event: str, reason: str = "") -> None:
        record = RunRecord(str(file.resolve()), node, event, reason)
        self._out.write(json.dumps(asdict(record)) + "\n")

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, items: list[pytest.Item]):
        # Before every other filter, so a test a filter removes is still recorded.
        for item in items:
            self._write(item.path, _node_name(item), "collected")
        yield

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        for item in items:
            self._write(item.path, _node_name(item), "deselected", _deselection(item))

    @pytest.hookimpl(hookwrapper=True)
    def pytest_make_collect_report(self, collector: pytest.Collector):
        report = (yield).get_result()
        if report.passed or not isinstance(collector, pytest.Module):
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
                self._write(item.path, node, "ran")
        elif report.skipped:
            # A skip in setup or in the test body: the test did not check what
            # it is for. (A skip in teardown comes after the call's report.)
            if report.when != "teardown":
                self._write(item.path, node, "skipped", _skip_reason(report))
        elif report.when == "call" or report.failed:
            self._write(item.path, node, "ran")

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
