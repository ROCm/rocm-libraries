# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No test file in limbo: each is assigned to a CI entry or has a recorded reason.

Pull-request and push CI select rocKE's CTest entries by the quick and standard
tier labels the provider categories give them. A test file is assigned to a CI
entry when one of those entries selects it: a build-tree entry names source paths; an installed
entry names paths of the install layout, which the configured install rules map
back to the source (so a file the install leaves out is assigned to no CI entry
even when its directory is). A file a selected test runs as a child process counts too. A
module whose ``pytestmark`` the entry's ``-m`` expression rejects is not
assigned to that entry. The check works per file: marks on single tests, ``-k``
filters and a file whose every case skips in CI are not seen here, so
``rocke.tests.pending_lanes`` records those files under the lane their cases
declare where they skip (``rocke.tests.lanes``), and ``tools/limbo_audit.py``
checks each case against the record from the run records of CI lanes.

Every other test file must be listed in ``rocke.tests.pending_lanes``, and every
listed entry must be current: its file present, and assigned to no CI entry for
a whole-file entry, to one otherwise. A file is listed once per lane, and a
whole-file entry is its file's only entry. Each record's team owns its path in
CODEOWNERS. A file whose cases need a GPU no pull-request runner has (an arch
helper called with such arches) has a record for them, and no test writes a
lane reason by hand. An expired entry fails a change that touches it (its lines
in the registry, or its test file), and every run under ``--strict-expiry``;
otherwise it is reported as a warning. The registry's measured budgets are held
to the same rules: each names a present test file, is listed once, and is
reported or failed when expired the same way. A quarantined test names a test
that exists, is listed once, is reported or failed when expired the same way,
and the registry holds at most ``QUARANTINE_CAP`` of them.

The check configures the platform (nothing is built) and reads the provider
categories, so it runs in a source checkout of the monorepo only.
"""

from __future__ import annotations

import ast
import datetime
import functools
import itertools
import os
import posixpath
import re
import shutil
import subprocess
import sys
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest
from rocke.assets import platform_root
from rocke.tests import in_tree_suites, pending_lanes
from rocke.tests.comgr import CompilerBuild
from rocke.tests.ctest_listing import (
    configure_installed_ctest,
    list_ctest_entries,
    list_installed_ctest_entries,
)
from rocke.tests.entry_selection import (
    InstallLayout,
    Layout,
    SourceLayout,
    deselects_module,
    install_rules,
    module_marks,
    pytest_selection,
    selected_test_files,
)
from rocke.tests.entry_selection import (
    arguments_after_interpreter as _arguments_after_interpreter,
)
from rocke.tests.entry_selection import normalized_path as _normalized
from rocke.tests.lane_mentions import LaneMention, LaneMentionKind, lane_mentions
from rocke.tests.lanes import (
    BUILD_VARIANT,
    PR_RUNNER_ARCHES,
    RUNNER_HARDWARE,
    SOURCE_TREE,
    TICKET,
    Lane,
    skip_until_reason,
)
from rocke.tests.pending_lanes import (
    LANE_TICKETS,
    MEASURED_BUDGETS,
    PENDING,
    PENDING_MAX_DAYS,
    QUARANTINE_CAP,
    QUARANTINE_MAX_DAYS,
    QUARANTINE_TREES,
    QUARANTINED,
    TEAMS,
    Compilers,
    LaneTicket,
    MeasuredBudget,
    PendingTest,
    QuarantinedTest,
    measured_budget_kernels,
    measured_scratch_allowance,
    tree_test_files,
    utc_today,
    waiting_entry,
)
from rocke.tests.provider_categories import provider_category_labels
from rocke.tests.quarantine import register_quarantine, skip_quarantined_items
from rocke.tests.reference_artifacts import (
    ARTIFACT_CLI,
    published_reference_variants,
)

ROCKE = platform_root().parent
# The tiers pull-request and push CI select.
CI_TIERS = frozenset({"quick", "standard"})
# The provider build installs rocKE's CTest entries into bin/<this>.
PROVIDER_CTEST_SUBDIR = "hip_kernel_provider"
# The tests that run other test files as child pytest processes, so no CTest
# command names those files.
IN_TREE_RUNNER = "platform/tests/in_tree/test_in_tree_suites.py"
PORTABLE_IR_RUNNER = "platform/tests/portable_ir/test_portable_ir.py"
PORTABLE_IR_UNIT_TESTS = "platform/python/rocke/portable_ir/tests"
# platform/CMakeLists.txt (the ROCKE_INSTALL_TEST_GPU_REFERENCES block, through
# cmake/InstallGpuReferences.cmake) registers rocke_<operation>_gpu_<arch>_pytest
# over library/tests/test_<operation>_pinned_reference.py for each architecture
# that publishes a variant of the operation, only when the reference bundles
# are installed, which the provider build does and this check's configure
# (without the bundles) cannot. The library's artifact CLI lists those pairs
# from the committed locks (rocke.tests.reference_artifacts).
# The registry module, relative to the rocke directory.
PENDING_LANES_MODULE = "platform/python/rocke/tests/pending_lanes.py"
# A record of the registry: each kind names its test file and its expiry.
Record = PendingTest | MeasuredBudget | QuarantinedTest


def pinned_reference_entries(library: Path) -> dict[str, set[str]]:
    """Each pinned-reference test file, relative to the rocke directory -> the
    GPU CTest entries the provider build registers over it on a pull-request
    runner arch: one per architecture in ``PR_RUNNER_ARCHES`` that publishes a
    variant of the file's operation, as the artifact CLI under ``library``
    lists them. A file's name credits it with nothing."""
    entries: dict[str, set[str]] = {}
    for variant in published_reference_variants(library):
        operation, arch = variant["operation"], variant["architecture"]
        if arch in PR_RUNNER_ARCHES:
            path = f"library/tests/test_{operation}_pinned_reference.py"
            entries.setdefault(path, set()).add(f"rocke_{operation}_gpu_{arch}_pytest")
    return entries


def run_by_another_test() -> dict[str, str]:
    """Each test file a test runs as a child process -> the file that runs it."""
    runners = {}
    for package in in_tree_suites.SUITES:
        directory = ROCKE / "platform" / "python" / package.replace(".", "/")
        for file in directory.rglob("test_*.py"):
            if file.name not in in_tree_suites.EXCLUDED_FILES:
                runners[file.relative_to(ROCKE).as_posix()] = IN_TREE_RUNNER
    for file in (ROCKE / PORTABLE_IR_UNIT_TESTS).glob("test_*.py"):
        runners[file.relative_to(ROCKE).as_posix()] = PORTABLE_IR_RUNNER
    return runners


def ci_selected_files(scratch: Path) -> set[str]:
    """The test files, relative to the rocke directory, that pull-request and
    push CI run."""
    build, installed_ctest = configure_installed_ctest(
        scratch, f"-DROCKE_INSTALL_PROVIDER_CTEST_SUBDIR={PROVIDER_CTEST_SUBDIR}"
    )
    install = InstallLayout(
        [
            rule
            for script in sorted(build.rglob("cmake_install.cmake"))
            for rule in install_rules(script.read_text())
        ]
    )
    # Installed commands name paths relative to the CTest directory; build-tree
    # commands name absolute source paths.
    trees: list[tuple[list[dict], str, Layout]] = [
        (list_ctest_entries(build), "", SourceLayout()),
        (
            list_installed_ctest_entries(installed_ctest, scratch),
            f"bin/{PROVIDER_CTEST_SUBDIR}/",
            install,
        ),
    ]
    references = pinned_reference_entries(ROCKE / "library")
    reference_names = set().union(*references.values())
    entry_names = [entry["name"] for entries, _, _ in trees for entry in entries]
    labels = provider_category_labels([*entry_names, *reference_names], scratch)

    def ci_run(name: str) -> bool:
        return bool(labels.get(name, set()) & CI_TIERS)

    selected: set[Path] = set()
    for entries, base, layout in trees:
        # CTest leaves out the command of a program it cannot find: the C++
        # tests (nothing is built), which select no test file, but never the
        # Python ones.
        assert any("command" in entry for entry in entries), "CTest found no program"
        for entry in entries:
            if "command" in entry and ci_run(entry["name"]):
                selected |= selected_test_files(
                    _arguments_after_interpreter(entry["command"]), layout, base
                )
    rocke = ROCKE.resolve()
    in_ci = {
        file.relative_to(rocke).as_posix() for file in selected if rocke in file.parents
    }
    in_ci |= {file for file, runner in run_by_another_test().items() if runner in in_ci}
    in_ci |= {path for path, names in references.items() if any(map(ci_run, names))}
    return in_ci


def audit(
    test_files: set[str], in_ci: set[str], pending: Iterable[PendingTest]
) -> tuple[list[str], list[str]]:
    """(files in limbo, stale pending entries): the test files neither in CI
    nor listed, and the listed files whose entry's scope does not fit: a
    whole-file entry for a file in CI, or a case entry for a file in none."""
    pending = list(pending)
    in_limbo = sorted(test_files - in_ci - {entry.path for entry in pending})
    stale = sorted(
        {entry.path for entry in pending if (entry.path in in_ci) == entry.whole_file}
    )
    return in_limbo, stale


def listing_breaches(pending: Iterable[PendingTest]) -> list[str]:
    """The files or tests listed twice under one lane, and the whole-file
    entries that are not their file's only entry."""
    pending = list(pending)
    keys = [(entry.path, entry.node, entry.lane.name) for entry in pending]
    breaches = [
        f"listed twice under one lane: {path}" + (f"::{node}" if node else "")
        for path, node, _ in sorted({key for key in keys if keys.count(key) > 1})
    ]
    entries_of: dict[str, list[PendingTest]] = {}
    for entry in pending:
        entries_of.setdefault(entry.path, []).append(entry)
    for path, entries in sorted(entries_of.items()):
        if len(entries) > 1 and any(entry.whole_file for entry in entries):
            breaches.append(
                "a whole-file entry and other entries for one file: "
                f"{path} ({', '.join(entry.lane.name for entry in entries)})"
            )
    return breaches


def _days_from_today(days: int) -> str:
    return (utc_today() + datetime.timedelta(days=days)).isoformat()


def _pending(path: str, lane: Lane, **overrides: object) -> PendingTest:
    """A pending entry with a valid team, issue and expiry, overridden by
    ``overrides``."""
    fields = {
        "reason": "why",
        "team": TEAMS[0],
        "issue": "AICK-1234",
        "expires": _days_from_today(30),
    }
    return PendingTest(path, lane, **{**fields, **overrides})


def test_audit_finds_files_in_limbo_and_stale_entries():
    test_files = {
        "a/test_in_ci.py",
        "a/test_limbo.py",
        "a/test_whole_file.py",
        "a/test_skips.py",
    }
    in_ci = {"a/test_in_ci.py", "a/test_whole_file.py"}
    pending = [
        _pending("a/test_whole_file.py", TICKET, whole_file=True),
        _pending("a/test_skips.py", BUILD_VARIANT),
    ]
    # test_whole_file waits as a whole file but is in CI; test_skips has a
    # case entry (its cases declare a lane) but is in no CI entry.
    assert audit(test_files, in_ci, pending) == (
        ["a/test_limbo.py"],
        ["a/test_skips.py", "a/test_whole_file.py"],
    )


def test_a_file_may_wait_for_several_lanes_but_a_whole_file_entry_is_alone():
    hardware = _pending("a/test_x.py", RUNNER_HARDWARE)
    build = _pending("a/test_x.py", BUILD_VARIANT)
    assert listing_breaches([hardware, build]) == []
    assert listing_breaches([hardware, hardware]) == [
        "listed twice under one lane: a/test_x.py"
    ]
    whole_file = _pending("a/test_x.py", SOURCE_TREE, whole_file=True)
    assert listing_breaches([hardware, whole_file]) == [
        (
            "a whole-file entry and other entries for one file: "
            "a/test_x.py (runner-hardware, source-tree)"
        )
    ]
    # Each of a file's case entries must fit it: the file is in CI.
    entries = [hardware, build]
    assert audit({"a/test_x.py"}, {"a/test_x.py"}, entries) == ([], [])
    assert audit({"a/test_x.py"}, set(), entries) == ([], ["a/test_x.py"])


def test_paths_keep_their_root_when_normalized():
    assert _normalized("/src/./tests/../tests/core") == "/src/tests/core"
    assert _normalized("bin/p/./tests/../rocke") == "bin/p/rocke"
    assert _normalized("../tests") == "../tests"


def test_a_node_id_selects_its_file():
    selection = pytest_selection(
        ["tests/test_a.py::test_one", "-k", "x::y", "tests/test_b.py"]
    )
    assert selection.paths == ("tests/test_a.py", "tests/test_b.py")


def test_a_marker_expression_deselects_a_module_by_its_pytestmark(tmp_path):
    single = tmp_path / "test_single.py"
    single.write_text("import pytest\npytestmark = pytest.mark.integration\n")
    listed = tmp_path / "test_listed.py"
    listed.write_text(
        "import pytest\n"
        "pytestmark = [pytest.mark.gpu, pytest.mark.skipif(True, reason='x'), guard]\n"
    )
    plain = tmp_path / "test_plain.py"
    plain.write_text("def test_x():\n    pass\n")
    assert module_marks(listed) == {"gpu", "skipif"}
    assert deselects_module("not integration", single)
    assert not deselects_module("not integration", plain)
    assert deselects_module("gpu", plain)
    assert not deselects_module("gpu", listed)
    assert not deselects_module(None, single)


def test_a_file_entry_covers_the_cases_that_declare_its_lane_only():
    waiting = _pending("a/test_x.py", RUNNER_HARDWARE)
    declared = skip_until_reason(RUNNER_HARDWARE, reason="needs a gfx950 GPU")
    assert waiting.covers("a/test_x.py", "test_y", declared)
    both = skip_until_reason(RUNNER_HARDWARE, BUILD_VARIANT, reason="needs both")
    assert waiting.covers("a/test_x.py", "test_y", both)
    # A skip that only mentions the lane's arch is not a declaration: a defect
    # whose reason names gfx950 is not booked as waiting for the hardware.
    assert not waiting.covers("a/test_x.py", "test_y", "needs a gfx950 GPU")
    assert not waiting.covers("a/test_x.py", "test_y", "gfx950 kernel miscompares")
    other = skip_until_reason(BUILD_VARIANT, reason="no rocke_engine")
    assert not waiting.covers("a/test_x.py", "test_y", other)
    assert not waiting.covers("a/test_x.py", "test_y", None)
    assert not waiting.covers("a/test_z.py", "test_y", declared)
    # A whole-file entry: every case of its file waits for it.
    whole_file = _pending("a/test_x.py", SOURCE_TREE, whole_file=True)
    assert whole_file.covers("a/test_x.py", "test_y", "any reason")
    assert whole_file.covers("a/test_x.py", "test_y", None)


def test_an_entry_for_one_test_covers_its_unselected_or_declared_cases_only():
    with pytest.raises(ValueError, match="expected one test or the whole file"):
        _pending("a/test_x.py", TICKET, node="test_y", whole_file=True)
    with pytest.raises(ValueError, match="expected one of rocke.tests.lanes.LANES"):
        _pending("a/test_x.py", Lane("unlisted", "a lane"))
    test = _pending("a/test_x.py", TICKET, node="test_y")
    left_out = "deselected by -k 'not test_y'"
    assert test.covers("a/test_x.py", "test_y[case]", left_out, deselected=True)
    assert test.covers("a/test_x.py", "test_y", None)
    assert not test.covers("a/test_x.py", "test_yz", None)
    assert not test.covers("a/test_z.py", "test_y", None)
    group = _pending("a/test_x.py", TICKET, node="TestGroup")
    assert group.covers("a/test_x.py", "TestGroup::test_y[case]", None)
    assert not group.covers("a/test_x.py", "TestGroupTwo::test_y", None)
    # A test that ran to a skip is covered only when the skip declares the
    # entry's lane: a defect skip of a test listed under the hardware is not.
    runner = _pending("a/test_x.py", RUNNER_HARDWARE, node="test_y")
    declared = skip_until_reason(RUNNER_HARDWARE, reason="needs a gfx950 GPU")
    assert runner.covers("a/test_x.py", "test_y", declared)
    assert not runner.covers("a/test_x.py", "test_y", "invalid spec on gfx942")
    assert not runner.covers("a/test_x.py", "test_y", left_out)


def test_the_registry_accounts_for_a_case_by_its_lane_quarantine_or_runner():
    file_entry = next(
        entry
        for entry in PENDING
        if entry.lane is RUNNER_HARDWARE
        and not entry.node
        and sum(other.path == entry.path for other in PENDING) == 1
        and not any(q.path == entry.path for q in QUARANTINED)
    )
    path = file_entry.path
    declared = skip_until_reason(RUNNER_HARDWARE, reason="needs a gfx950 GPU")
    assert waiting_entry(path, "test_any", declared) is file_entry
    assert waiting_entry(path, "test_any", "needs a gfx950 GPU") is None
    assert waiting_entry(path, "test_any", None) is None
    everywhere = next(entry for entry in QUARANTINED if not entry.arches)
    assert waiting_entry(everywhere.path, everywhere.node_name, "any") is everywhere
    on_gfx950 = next(entry for entry in QUARANTINED if entry.arches)
    reason = f"quarantined: {on_gfx950.issue} on gfx950"
    assert waiting_entry(on_gfx950.path, on_gfx950.node_name, reason) is on_gfx950
    assert waiting_entry(on_gfx950.path, on_gfx950.node_name, "why") is None
    runner = next(entry for entry in QUARANTINED if entry.runs_files)
    child = runner.runs_files[0]
    assert waiting_entry(child, "TestAny::test_any", None) is runner


def test_the_record_of_a_test_comes_before_its_files():
    """A case both a record of its file and one of its test cover is booked
    to the test's: the more specific record carries its reason and issue."""
    node_entry = next(
        entry
        for entry in PENDING
        if entry.node
        and any(
            other.path == entry.path and other.lane is entry.lane and not other.node
            for other in PENDING
        )
    )
    declared = skip_until_reason(node_entry.lane, reason="why")
    assert waiting_entry(node_entry.path, node_entry.node, declared) is node_entry
    assert waiting_entry(node_entry.path, "test_another", declared) is not node_entry


def test_a_pending_test_needs_its_recorded_reason_team_and_issue():
    path = "library/tests/test_x.py"
    with pytest.raises(ValueError, match="reason is empty"):
        _pending(path, TICKET, reason=" ")
    with pytest.raises(ValueError, match="expected plain text"):
        _pending(path, TICKET, reason="see https://x")
    with pytest.raises(ValueError, match="expected an ISO date"):
        _pending(path, TICKET, expires="next month")
    with pytest.raises(ValueError, match="team='rocKE library', expected one of"):
        _pending(path, TICKET, team="rocKE library")
    for unfiled in ("none filed yet", "issue: none filed yet", "", "AICK"):
        with pytest.raises(ValueError, match="expected the issue that tracks it"):
            _pending(path, TICKET, issue=unfiled)
    assert _pending(path, TICKET, issue="ROCm/rocm-libraries#1").issue == (
        "ROCm/rocm-libraries#1"
    )


def lane_ticket_breaches(
    pending: Iterable[PendingTest], tickets: Iterable[LaneTicket]
) -> list[str]:
    """The lane tickets listed twice, the entries whose issue is no lane
    ticket's, and the entries that name a lane ticket's issue with another lane
    or expiry: every entry takes all three from its lane ticket, so each lane
    of an issue has one date."""
    tickets = list(tickets)
    pending = list(pending)
    issues = [ticket.issue for ticket in tickets]
    breaches = [
        f"lane ticket listed twice: {issue}"
        for issue in sorted({issue for issue in issues if issues.count(issue) > 1})
    ]
    ticket_of = {ticket.issue: ticket for ticket in tickets}
    breaches += [
        f"{entry.path}{'::' + entry.node if entry.node else ''} names "
        f"{entry.issue}, which is no lane ticket's issue: add a LaneTicket to "
        "LANE_TICKETS and write the entry with LaneTicket.waiting"
        for entry in pending
        if entry.issue not in ticket_of
    ]
    breaches += [
        f"{entry.path}{'::' + entry.node if entry.node else ''} names lane ticket "
        f"{entry.issue} with lane {entry.lane.name} and expiry {entry.expires}, "
        f"expected {ticket.lane.name} and {ticket.expires}: write it with "
        "LaneTicket.waiting"
        for entry in pending
        if (ticket := ticket_of.get(entry.issue)) is not None
        and (entry.lane, entry.expires) != (ticket.lane, ticket.expires)
    ]
    return breaches


def test_a_lane_ticket_gives_its_records_its_lane_issue_and_expiry():
    """A lane ticket's entries take its lane, issue and expiry; a ticket needs
    a tracked issue, a listed lane and an expiry within the pending window."""
    ticket = LaneTicket("AICK-1", RUNNER_HARDWARE, _days_from_today(30))
    entry = ticket.waiting("library/tests/test_x.py", "why", TEAMS[0], node="test_y")
    assert (entry.lane, entry.issue, entry.expires, entry.node) == (
        RUNNER_HARDWARE,
        "AICK-1",
        ticket.expires,
        "test_y",
    )
    assert ticket.waiting(
        "library/tests/test_x.py", "why", TEAMS[0], whole_file=True
    ).whole_file
    with pytest.raises(ValueError, match="expected the issue that tracks it"):
        LaneTicket("none filed yet", RUNNER_HARDWARE, _days_from_today(30))
    with pytest.raises(ValueError, match="expected one of rocke.tests.lanes.LANES"):
        LaneTicket("AICK-1", Lane("unlisted", "why"), _days_from_today(30))
    with pytest.raises(ValueError, match=f"expected at most {PENDING_MAX_DAYS} days"):
        LaneTicket("AICK-1", RUNNER_HARDWARE, _days_from_today(PENDING_MAX_DAYS + 1))


def test_a_record_that_names_a_lane_ticket_takes_its_lane_and_expiry():
    """An entry whose issue is no lane ticket's, or that names a lane ticket's
    issue with another lane or date, is a breach, so each lane of an issue
    keeps one date; a ticket is listed once."""
    ticket = LaneTicket("AICK-1", RUNNER_HARDWARE, _days_from_today(30))
    path = "library/tests/test_x.py"
    assert lane_ticket_breaches([ticket.waiting(path, "why", TEAMS[0])], [ticket]) == []
    other_date = _pending(
        path, RUNNER_HARDWARE, issue="AICK-1", expires=_days_from_today(20)
    )
    other_lane = _pending(path, TICKET, issue="AICK-1", expires=ticket.expires)
    other_issue = _pending(path, TICKET, issue="AICK-2")
    assert lane_ticket_breaches([other_date, other_lane, other_issue], [ticket]) == [
        (
            f"{path} names AICK-2, which is no lane ticket's issue: add a "
            "LaneTicket to LANE_TICKETS and write the entry with LaneTicket.waiting"
        ),
        (
            f"{path} names lane ticket AICK-1 with lane runner-hardware and expiry "
            f"{other_date.expires}, expected runner-hardware and {ticket.expires}: "
            "write it with LaneTicket.waiting"
        ),
        (
            f"{path} names lane ticket AICK-1 with lane ticket and expiry "
            f"{ticket.expires}, expected runner-hardware and {ticket.expires}: "
            "write it with LaneTicket.waiting"
        ),
    ]
    assert lane_ticket_breaches([], [ticket, ticket]) == [
        "lane ticket listed twice: AICK-1"
    ]


def test_every_record_of_a_lane_ticket_takes_its_lane_and_expiry():
    """The registry's entries agree with its lane tickets, and every lane
    ticket it defines is listed in LANE_TICKETS, which the check reads."""
    breaches = lane_ticket_breaches(PENDING, LANE_TICKETS)
    assert not breaches, breaches
    unlisted = sorted(
        ticket.issue
        for ticket in vars(pending_lanes).values()
        if isinstance(ticket, LaneTicket) and ticket not in LANE_TICKETS
    )
    assert not unlisted, f"lane tickets missing from LANE_TICKETS: {unlisted}"


def test_a_record_expires_within_its_maximum_window(monkeypatch):
    """A pending entry or measured budget expires at most PENDING_MAX_DAYS
    days away, a quarantine at most QUARANTINE_MAX_DAYS: a record is re-decided
    in that window, not set once and forgotten. A past date is a valid
    record, which the expiry check then reports. Today is read once and held
    for the records' checks too, so a run across midnight (UTC) does not move
    the window between the dates and the checks."""
    today = utc_today()
    monkeypatch.setattr(pending_lanes, "utc_today", lambda: today)

    def days_away(days: int) -> str:
        return (today + datetime.timedelta(days=days)).isoformat()

    path = "library/tests/test_x.py"
    assert _pending(path, TICKET, expires=days_away(PENDING_MAX_DAYS))
    assert _pending(path, TICKET, expires="2026-01-01")
    with pytest.raises(ValueError, match=f"expected at most {PENDING_MAX_DAYS} days"):
        _pending(path, TICKET, expires=days_away(PENDING_MAX_DAYS + 1))
    test_id = f"{path}::test_y"
    assert _quarantined(test_id, expires=days_away(QUARANTINE_MAX_DAYS))
    with pytest.raises(
        ValueError, match=f"expected at most {QUARANTINE_MAX_DAYS} days"
    ):
        _quarantined(test_id, expires=days_away(QUARANTINE_MAX_DAYS + 1))
    with pytest.raises(ValueError, match=f"expected at most {PENDING_MAX_DAYS} days"):
        _measured_budget(expires=days_away(PENDING_MAX_DAYS + 1))


def test_install_rules_map_installed_paths_back_to_the_source(tmp_path):
    for path in (
        "tests/core/test_a.py",
        "tests/test_skip.py",
        "tests/differential/test_b.py",
    ):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).touch()
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/test_moved.py").touch()
    script = (
        f'file(INSTALL DESTINATION "${{CMAKE_INSTALL_PREFIX}}/bin/p/tests" TYPE DIRECTORY '
        f'FILES "{tmp_path}/tests/" REGEX "/differential$" EXCLUDE '
        'REGEX "/test\\\\_skip\\\\.py$" EXCLUDE)\n'
        'file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/p/tests/bench" TYPE FILE '
        f'FILES\n    "{tmp_path}/pkg/test_moved.py"\n    )\n'
    )
    layout = InstallLayout(install_rules(script))
    assert (
        layout.source_of("bin/p/tests/core/test_a.py")
        == tmp_path / "tests/core/test_a.py"
    )
    # Left out by an exclude, of the file or of a directory above it.
    assert layout.source_of("bin/p/tests/test_skip.py") is None
    assert layout.source_of("bin/p/tests/differential/test_b.py") is None
    # A file installed into a directory another rule fills.
    assert (
        layout.source_of("bin/p/tests/bench/test_moved.py")
        == tmp_path / "pkg/test_moved.py"
    )
    assert layout.test_files_under("bin/p/tests") == {
        "bin/p/tests/core/test_a.py": tmp_path / "tests/core/test_a.py",
        "bin/p/tests/bench/test_moved.py": tmp_path / "pkg/test_moved.py",
    }


@dataclass(frozen=True)
class Change:
    """What a change touches against its base: the files (relative to the
    rocke directory, working tree included) and, in the registry module, the
    lines it adds or rewrites and the lines after which it deletes some."""

    files: frozenset[str]
    registry_lines: frozenset[int]
    registry_gaps: frozenset[int]


def change_base() -> str | None:
    """The commit this change is checked against: where it forks from
    ``ROCKE_CHANGE_BASE``, else from the ``--from-ref`` of the pre-commit run
    (``PRE_COMMIT_FROM_REF``; in CI, the tip of the pull request's base
    branch); None when neither is set or the base does not resolve. The fork
    point, so that the base's own newer edits are not counted as this
    change's; a base with no fork point with HEAD (a tree) is taken as is.

    No base is guessed: a run that names none (the CTest entry, develop's own
    CI, a plain local run) has no change, so an expired record only warns
    there. A guessed one, such as a stale local origin/develop, would count
    other changes' edits as this change's and fail a run that touched
    nothing."""
    base = os.environ.get("ROCKE_CHANGE_BASE") or os.environ.get("PRE_COMMIT_FROM_REF")
    if not base:
        return None
    return _git("merge-base", base, "HEAD") or (
        base if _git("rev-parse", "--verify", "--quiet", base) else None
    )


def _git(*arguments: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run(
        [git, *arguments], cwd=ROCKE, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


# A hunk header of a zero-context diff: its start and length in the new file.
_HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def registry_line_changes(diff: str) -> tuple[frozenset[int], frozenset[int]]:
    """(lines added or rewritten, lines after which lines were deleted) in the
    new file, from the zero-context diff of the registry module."""
    lines, gaps = set(), set()
    for match in _HUNK.finditer(diff):
        start, length = int(match.group(1)), int(match.group(2) or 1)
        if length:
            lines.update(range(start, start + length))
        else:
            gaps.add(start)
    return frozenset(lines), frozenset(gaps)


def current_change() -> Change | None:
    """This change against :func:`change_base`; None when git cannot tell."""
    base = change_base()
    if not base:
        return None
    options = ("--no-ext-diff", "--no-color", "--relative")
    names = _git("diff", *options, "--name-only", base)
    diff = _git("diff", *options, "-U0", base, "--", PENDING_LANES_MODULE)
    if names is None or diff is None:
        return None
    lines, gaps = registry_line_changes(diff)
    return Change(frozenset(names.splitlines()), lines, gaps)


def _lines_of(node: ast.AST) -> set[int]:
    return set(range(node.lineno, node.end_lineno + 1))


def _shared_argument_counts(module: ast.Module) -> dict[str, int]:
    """Each function of ``module`` that takes its records as ``*args`` -> the
    number of positional arguments before them, which every record shares."""
    return {
        node.name: len(node.args.posonlyargs) + len(node.args.args)
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.args.vararg is not None
    }


def record_lines(source: str, registry_name: str) -> list[frozenset[int]]:
    """The source lines of each record of the registry tuple ``registry_name``
    in ``source``, in order: a record written alone, all its lines; one of a
    group helper's records, the helper's shared arguments and its own."""
    module = ast.parse(source)
    shared_counts = _shared_argument_counts(module)
    statement = next(
        node
        for node in module.body
        if isinstance(node, ast.AnnAssign)
        and getattr(node.target, "id", None) == registry_name
    )
    spans = []
    for element in statement.value.elts:
        if not isinstance(element, ast.Starred):
            spans.append(frozenset(_lines_of(element)))
            continue
        call = element.value
        helper = getattr(call.func, "id", None)
        if helper not in shared_counts:
            raise ValueError(
                f"record_lines failed - {registry_name} unpacks a call of "
                f"{helper!r}, expected a function of the module that takes its "
                "records as *args"
            )
        shared_count = shared_counts[helper]
        shared = {call.lineno}.union(
            *(_lines_of(argument) for argument in call.args[:shared_count]),
            *(_lines_of(keyword.value) for keyword in call.keywords),
        )
        spans += [
            frozenset(shared | _lines_of(own)) for own in call.args[shared_count:]
        ]
    return spans


def touches(change: Change, path: str, lines: frozenset[int]) -> bool:
    """Whether ``change`` touches the record of ``path`` written on ``lines``
    of the registry: it changes the test file, or adds, rewrites or deletes
    one of the record's lines."""
    return (
        path in change.files
        or bool(lines & change.registry_lines)
        or any(gap in lines and gap + 1 in lines for gap in change.registry_gaps)
    )


@dataclass(frozen=True)
class ExpiredRecord:
    """An expired registry record: how a message names it, its test file and
    its lines in the registry module."""

    label: str
    path: str  # the test file
    lines: frozenset[int]  # its lines in the registry module


def expiry_failures(
    expired: Iterable[ExpiredRecord], change: Change | None, *, strict: bool
) -> list[str]:
    """The labels of the expired records that fail the run: all of them under
    ``--strict-expiry`` (a scheduled run), else those ``change`` touches, so
    a record reaching its date fails only the change that re-decides it. The
    rest are reported as warnings."""
    return [
        record.label
        for record in expired
        if strict or (change is not None and touches(change, record.path, record.lines))
    ]


_GROUP_SOURCE = """\
PENDING: tuple = (
    *_waiting(
        LANE,
        "reason",
        TEAM,
        "AICK-1",
        "2026-01-01",
        "a/test_x.py",
        "a/test_y.py",
    ),
    PendingTest(
        "a/test_z.py",
        LANE,
        "reason",
        TEAM,
        "AICK-1",
        "2026-01-01",
    ),
)


def _waiting(lane, reason, team, issue, expires, *paths):
    pass
"""


def test_each_record_owns_its_lines_and_its_group_lines():
    x, y, z = record_lines(_GROUP_SOURCE, "PENDING")
    shared = {2, 3, 4, 5, 6, 7}
    assert x == shared | {8} and y == shared | {9}
    assert z == set(range(11, 19))


def test_an_expired_record_fails_only_a_change_that_touches_it():
    x, y, z = (
        ExpiredRecord(f"a/test_{name}.py", f"a/test_{name}.py", lines)
        for name, lines in zip("xyz", record_lines(_GROUP_SOURCE, "PENDING"))
    )
    expired = [x, y, z]

    def change(files=(), lines=(), gaps=()):
        return Change(frozenset(files), frozenset(lines), frozenset(gaps))

    labels = [record.label for record in expired]
    assert expiry_failures(expired, None, strict=False) == []
    assert expiry_failures(expired, None, strict=True) == labels
    # Another record's line, or a record added next to them: none fails.
    assert expiry_failures(expired, change(lines={30}), strict=False) == []
    assert expiry_failures(expired, change(files={"other.py"}), strict=False) == []
    # Its own line, its test file, or its group's shared expiry line.
    assert expiry_failures(expired, change(lines={9}), strict=False) == [y.label]
    assert expiry_failures(expired, change(files={z.path}), strict=False) == [z.label]
    assert expiry_failures(expired, change(lines={7}), strict=False) == [
        x.label,
        y.label,
    ]
    # A line deleted between two of a record's lines touches it (here a line
    # of the group's reason); one deleted between two records, or a path
    # deleted from the group, touches neither.
    assert expiry_failures(expired, change(gaps={4}), strict=False) == [
        x.label,
        y.label,
    ]
    assert expiry_failures(expired, change(gaps={8}), strict=False) == []
    assert expiry_failures(expired, change(gaps={10}), strict=False) == []


def test_a_zero_context_diff_names_the_lines_it_changes():
    diff = (
        "diff --git a/x b/x\n"
        "@@ -10,2 +10,3 @@ PENDING\n-a\n-b\n+a\n+b\n+c\n"
        "@@ -40 +41 @@\n-x\n+y\n"
        "@@ -60,2 +60,0 @@\n-gone\n-gone\n"
    )
    assert registry_line_changes(diff) == (frozenset({10, 11, 12, 41}), {60})


def test_the_change_base_is_the_fork_point_of_the_first_base_given(monkeypatch):
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    if head is None or tree is None:
        pytest.skip("this checkout is not a git work tree")
    monkeypatch.delenv("ROCKE_CHANGE_BASE", raising=False)
    monkeypatch.setenv("PRE_COMMIT_FROM_REF", "HEAD")
    assert change_base() == head
    # ROCKE_CHANGE_BASE comes first; a tree has no fork point and is kept.
    monkeypatch.setenv("ROCKE_CHANGE_BASE", tree)
    assert change_base() == tree
    monkeypatch.setenv("ROCKE_CHANGE_BASE", "no-such-ref-for-this-test")
    assert change_base() is None


def test_a_run_that_names_no_base_has_no_change(monkeypatch):
    """With neither ROCKE_CHANGE_BASE nor PRE_COMMIT_FROM_REF set, no base is
    guessed (not origin/develop), so an expired record only warns."""
    monkeypatch.delenv("ROCKE_CHANGE_BASE", raising=False)
    monkeypatch.delenv("PRE_COMMIT_FROM_REF", raising=False)
    assert change_base() is None
    assert current_change() is None


def fail_or_report_expired(
    registry_name: str,
    records: Iterable[Record],
    label: Callable[[Record], str],
    *,
    source: str,
    change: Change | None,
    strict: bool,
    max_days: int,
    today: datetime.date,
) -> None:
    """Fail on the expired records of the registry tuple ``registry_name``,
    written in the registry module's ``source``, that :func:`expiry_failures`
    says fail this run; report the rest as a warning. ``label`` names a record
    in the message, and ``max_days`` is its kind's longest expiry window."""
    spans = record_lines(source, registry_name)
    records = list(records)
    assert len(spans) == len(records), (
        f"{registry_name} holds {len(records)} records, but its source spells "
        f"{len(spans)}: record_lines no longer reads the registry the way it is "
        "written"
    )
    expired = [
        ExpiredRecord(label(record), record.path, lines)
        for record, lines in zip(records, spans)
        if record.expiry_date < today
    ]
    if not expired:
        return
    failing = expiry_failures(expired, change, strict=strict)
    assert not failing, (
        f"expired {registry_name} entries this change touches, or any under "
        "--strict-expiry; re-decide them "
        f"(a new expiry at most {max_days} days away, or clear them): "
        f"{failing}"
    )
    message = (
        f"{registry_name} entries past their expiry: {sorted(r.label for r in expired)}"
    )
    print(message)
    warnings.warn(message, stacklevel=2)


def _check_expiry(
    request: pytest.FixtureRequest,
    registry_name: str,
    records: Iterable[Record],
    label: Callable[[Record], str],
    max_days: int,
) -> None:
    """:func:`fail_or_report_expired` on this tree's registry module, this
    change, today, and this run's ``--strict-expiry``."""
    fail_or_report_expired(
        registry_name,
        records,
        label,
        source=(ROCKE / PENDING_LANES_MODULE).read_text(encoding="utf-8"),
        change=current_change(),
        strict=request.config.getoption("--strict-expiry"),
        max_days=max_days,
        today=utc_today(),
    )


def test_every_pending_entry_is_current(request):
    breaches = listing_breaches(PENDING)
    assert not breaches, breaches
    missing = sorted(
        {pending.path for pending in PENDING if not (ROCKE / pending.path).is_file()}
    )
    assert not missing, f"listed but gone: {missing}"
    gone = sorted(
        f"{pending.path}::{pending.node}"
        for pending in PENDING
        if pending.node and not _defines_test(ROCKE / pending.path, pending.node)
    )
    assert not gone, f"tests listed but gone: {gone}"
    _check_expiry(
        request,
        "PENDING",
        PENDING,
        lambda p: f"{p.path}{'::' + p.node if p.node else ''} ({p.lane.name})",
        PENDING_MAX_DAYS,
    )


def test_every_measured_budget_is_current(request):
    keys = [(budget.path, budget.kernel_name) for budget in MEASURED_BUDGETS]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    assert not duplicates, f"measured budget listed more than once: {duplicates}"
    missing = sorted(
        {b.path for b in MEASURED_BUDGETS if not (ROCKE / b.path).is_file()}
    )
    assert not missing, f"measured budgets for test files that are gone: {missing}"
    _check_expiry(
        request,
        "MEASURED_BUDGETS",
        MEASURED_BUDGETS,
        lambda b: f"{b.path}::{b.kernel_name}",
        PENDING_MAX_DAYS,
    )


def _measured_budget(**overrides: object) -> MeasuredBudget:
    """A measured budget with valid fields, overridden by ``overrides``."""
    fields = {
        "path": "library/tests/test_x.py",
        "kernel_name": "rocke_x",
        "scratch_bytes": 8,
        "compilers": Compilers(llvm_major=24),
        "reason": "why",
        "team": TEAMS[0],
        "issue": "AICK-1234",
        "expires": _days_from_today(30),
    }
    return MeasuredBudget(**{**fields, **overrides})


def test_a_measured_budget_applies_only_on_its_compilers():
    def budget(compilers):
        return _measured_budget(compilers=compilers)

    ci_build = "424729dba65359a5e521ab2e729af131c8a5b0d4"
    other_build = "d6f6cb691863e9258034ab48c1dfce21689d5606"
    by_major = budget(Compilers(llvm_major=24))
    by_commit = budget(Compilers(llvm_commits=("424729dba653",)))
    by_both = budget(Compilers(llvm_major=24, llvm_commits=("424729dba653",)))
    assert by_major.compilers.include(CompilerBuild(24, other_build))
    assert by_major.compilers.include(CompilerBuild(24, None))
    assert not by_major.compilers.include(CompilerBuild(20, None))
    assert not by_major.compilers.include(CompilerBuild(None, ci_build))
    # Two builds of one LLVM major: only the measured one is allowed the bytes.
    assert by_commit.compilers.include(CompilerBuild(24, ci_build))
    assert not by_commit.compilers.include(CompilerBuild(24, other_build))
    assert not by_commit.compilers.include(CompilerBuild(24, None))
    assert by_both.compilers.include(CompilerBuild(24, ci_build))
    assert not by_both.compilers.include(CompilerBuild(23, ci_build))


def test_the_registry_allows_a_measured_budget_on_its_compilers_only():
    budget = MEASURED_BUDGETS[0]
    major = budget.compilers.llvm_major
    assert major is not None
    measured_on = CompilerBuild(major, None)
    assert (
        measured_scratch_allowance(budget.path, budget.kernel_name, measured_on)
        == budget.scratch_bytes
    )
    other = CompilerBuild(major - 1, None)
    assert measured_scratch_allowance(budget.path, budget.kernel_name, other) == 0
    unknown = CompilerBuild(None, None)
    assert measured_scratch_allowance(budget.path, budget.kernel_name, unknown) == 0
    assert measured_scratch_allowance(budget.path, "rocke_other", measured_on) == 0
    other_file = "library/tests/test_x.py"
    assert measured_scratch_allowance(other_file, budget.kernel_name, measured_on) == 0
    assert budget.kernel_name in measured_budget_kernels(budget.path)
    assert measured_budget_kernels(other_file) == frozenset()


def test_a_measured_budget_needs_a_positive_size_and_its_recorded_reason():
    budget = _measured_budget
    with pytest.raises(ValueError, match="expected a test file path"):
        budget(path="/abs/test_x.py")
    with pytest.raises(ValueError, match="kernel_name is empty"):
        budget(kernel_name=" ")
    with pytest.raises(ValueError, match="scratch_bytes=0, expected a positive"):
        budget(scratch_bytes=0)
    with pytest.raises(TypeError, match="compilers=24, expected a Compilers"):
        budget(compilers=24)
    with pytest.raises(ValueError, match="team='', expected one of"):
        budget(team="")
    with pytest.raises(ValueError, match="expected the issue that tracks it"):
        budget(issue="none filed yet")
    with pytest.raises(ValueError, match="expected plain text"):
        budget(reason="see https://x")
    with pytest.raises(ValueError, match="expected an ISO date"):
        budget(expires="next month")


def test_compilers_name_a_major_or_commits_of_the_right_shape():
    with pytest.raises(ValueError, match="expected an LLVM major, llvm-project"):
        Compilers()
    with pytest.raises(ValueError, match="llvm_major=True, expected a positive"):
        Compilers(llvm_major=True)
    with pytest.raises(ValueError, match="llvm_major=0, expected a positive"):
        Compilers(llvm_major=0)
    for llvm_commits in (("424729dba65",), ("424729DBA653",), "424729dba653"):
        with pytest.raises(ValueError, match="expected a tuple of llvm-project"):
            Compilers(llvm_commits=llvm_commits)


def _quarantined(test_id: str, **overrides: object) -> QuarantinedTest:
    fields = {
        "issue": "AICK-1234",
        "reason": "why",
        "team": TEAMS[0],
        "expires": _days_from_today(14),
    }
    return QuarantinedTest(test_id, **{**fields, **overrides})


def test_a_quarantined_test_needs_an_item_an_issue_and_its_recorded_reason():
    with pytest.raises(ValueError, match="expected <test file>::<pytest node name>"):
        _quarantined("library/tests/test_x.py")
    with pytest.raises(ValueError, match="expected a test function or one of its"):
        _quarantined("library/tests/test_x.py::TestX")
    with pytest.raises(ValueError, match="expected a test function or one of its"):
        _quarantined("library/tests/test_x.py::testing_helper")
    assert _quarantined("library/tests/test_x.py::TestX::test_y").node_name == (
        "TestX::test_y"
    )
    with pytest.raises(ValueError, match="expected a test under one of"):
        _quarantined("platform/python/rocke/tests/test_x.py::test_y")
    with pytest.raises(ValueError, match="expected the issue that tracks it"):
        _quarantined("library/tests/test_x.py::test_y", issue="filed soon")
    with pytest.raises(ValueError, match="expected the issue that tracks it"):
        _quarantined(
            "library/tests/test_x.py::test_y",
            issue="https://x/issues/1",
        )
    with pytest.raises(ValueError, match="expected plain text"):
        _quarantined("library/tests/test_x.py::test_y", reason="see https://x")
    with pytest.raises(ValueError, match="expected an ISO date"):
        _quarantined("library/tests/test_x.py::test_y", expires="next month")
    with pytest.raises(ValueError, match="expected a tuple of known gfx arches"):
        _quarantined("library/tests/test_x.py::test_y", arches=("gfx9999",))
    with pytest.raises(ValueError, match="expected a tuple of known gfx arches"):
        _quarantined("library/tests/test_x.py::test_y", arches="gfx950")
    with pytest.raises(TypeError, match="expected a Compilers or None"):
        _quarantined("library/tests/test_x.py::test_y", compilers=("424729dba653",))
    github = _quarantined(
        "library/tests/test_x.py::test_y", issue="ROCm/rocm-libraries#1"
    )
    assert github.issue == "ROCm/rocm-libraries#1"
    with pytest.raises(ValueError, match="expected the issue that tracks it"):
        _quarantined("library/tests/test_x.py::test_y", issue="none filed yet")


def test_a_quarantine_that_names_arches_holds_its_test_out_on_those_only():
    everywhere = _quarantined("library/tests/test_x.py::test_y")
    assert everywhere.holds_out_on("gfx942")
    assert everywhere.holds_out_on(None)
    on_gfx950 = _quarantined("library/tests/test_x.py::test_y", arches=("gfx950",))
    assert on_gfx950.holds_out_on("gfx950")
    assert not on_gfx950.holds_out_on("gfx942")
    assert not on_gfx950.holds_out_on(None)


def test_a_quarantine_that_names_compilers_holds_its_test_out_with_those_only():
    ci_build = CompilerBuild(24, "424729dba65359a5e521ab2e729af131c8a5b0d4")
    other_build = CompilerBuild(24, "d6f6cb691863e9258034ab48c1dfce21689d5606")
    unknown = CompilerBuild(None, None)
    any_compiler = _quarantined("library/tests/test_x.py::test_y")
    assert any_compiler.holds_out_with(ci_build)
    assert any_compiler.holds_out_with(unknown)
    with_ci_build = _quarantined(
        "library/tests/test_x.py::test_y",
        compilers=Compilers(llvm_commits=("424729dba653",)),
    )
    assert with_ci_build.holds_out_with(ci_build)
    assert not with_ci_build.holds_out_with(other_build)
    assert not with_ci_build.holds_out_with(unknown)


def test_a_quarantined_function_covers_its_parameters_and_a_case_only_itself():
    function = _quarantined("platform/tests/test_x.py::test_y")
    assert function.covers("platform/tests/test_x.py", "test_y")
    assert function.covers("platform/tests/test_x.py", "test_y[gfx942]")
    assert not function.covers("platform/tests/test_x.py", "test_yz")
    assert not function.covers("library/tests/test_x.py", "test_y")
    case = _quarantined("platform/tests/test_x.py::test_y[gfx942]")
    assert case.covers("platform/tests/test_x.py", "test_y[gfx942]")
    assert not case.covers("platform/tests/test_x.py", "test_y[gfx950]")
    assert not case.covers("platform/tests/test_x.py", "test_y")


def test_a_quarantine_narrowed_by_params_covers_the_cases_with_all_of_them():
    """One record for one defect across several cases: every case whose
    parameter ids include each of ``params``, matched as whole ids."""
    narrowed = _quarantined(
        "platform/tests/test_x.py::test_y", params=("window", "bf16")
    )
    path = "platform/tests/test_x.py"
    assert narrowed.covers(path, "test_y[window-bf16-mha]")
    assert narrowed.covers(path, "test_y[bf16-gqa-window]")
    assert not narrowed.covers(path, "test_y[window-fp16-mha]")
    assert not narrowed.covers(path, "test_y[window_1seq-bf16-mha]")
    assert not narrowed.covers(path, "test_y")
    with pytest.raises(ValueError, match="expected a tuple of whole parameter ids"):
        _quarantined("platform/tests/test_x.py::test_y", params=("a-b",))
    with pytest.raises(ValueError, match="expected a tuple of whole parameter ids"):
        _quarantined("platform/tests/test_x.py::test_y", params="bf16")
    with pytest.raises(ValueError, match="expected a test without parameters"):
        _quarantined("platform/tests/test_x.py::test_y[a]", params=("a",))


def test_only_a_quarantine_tree_skips_quarantined_tests(request):
    with pytest.raises(
        ValueError, match="skip_quarantined_items failed .* expected one of"
    ):
        skip_quarantined_items([], ROCKE / "platform/python", "platform/python")
    # register_quarantine checks the tree before it registers anything, so this
    # session is left as it was.
    with pytest.raises(
        ValueError, match="register_quarantine failed .* expected one of"
    ):
        register_quarantine(
            request.config, ROCKE / "platform/python", "platform/python"
        )


def test_a_tiling_test_can_be_quarantined():
    tiling = "platform/python/rocke/tests/helpers/tiling"
    assert _quarantined(f"{tiling}/test_x.py::test_y").path == f"{tiling}/test_x.py"


def _quarantine_hook_impls(config: pytest.Config) -> list:
    """The collection hook implementations ``rocke.tests.quarantine`` installed
    in this session."""
    return [
        impl
        for impl in config.pluginmanager.hook.pytest_collection_modifyitems.get_hookimpls()
        if type(impl.plugin).__module__ == "rocke.tests.quarantine"
    ]


def test_a_tree_registers_its_quarantine_once(request):
    """This session's root (``platform/tests``) registered its quarantine; a
    second registration of the same tree adds no second hook."""
    registered = _quarantine_hook_impls(request.config)
    assert len(registered) == 1, registered
    register_quarantine(request.config, ROCKE / "platform/tests", "platform/tests")
    assert _quarantine_hook_impls(request.config) == registered


def test_a_root_marks_its_items_before_the_quarantine_skips_them(request):
    """pluggy runs every ``tryfirst`` collection hook before every plain one.
    The quarantine's hook is plain, and the tiling root's ``gpu`` marking is
    ``tryfirst``, so a quarantined GPU test is marked ``gpu`` (and selected by
    ``-m gpu``) before it is skipped."""
    from rocke.tests.helpers.tiling import conftest as tiling_conftest

    (quarantine_impl,) = _quarantine_hook_impls(request.config)
    assert not quarantine_impl.tryfirst
    assert tiling_conftest.pytest_collection_modifyitems.pytest_impl["tryfirst"]


def _configured_pytest_root_paths(conftest: Path) -> set[str]:
    """The tree paths ``conftest`` passes to ``configure_pytest_root``, as its
    third positional argument or as ``tree_path=``."""
    if not conftest.is_file():
        return set()
    paths = set()
    for node in ast.walk(ast.parse(conftest.read_text(encoding="utf-8"))):
        if not (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", None))
            == "configure_pytest_root"
        ):
            continue
        arguments = node.args[2:3] + [
            keyword.value for keyword in node.keywords if keyword.arg == "tree_path"
        ]
        paths.update(
            argument.value
            for argument in arguments
            if isinstance(argument, ast.Constant)
        )
    return paths


def test_every_quarantine_tree_configures_itself_as_a_pytest_root():
    """pytest loads only the conftest files on a collected path's own ancestor
    chain, so a tree skips its quarantined tests, and gets the sweep guard,
    only when its own ``conftest.py`` calls
    ``rocke.tests.pytest_roots.configure_pytest_root`` with its path."""
    unconfigured = [
        tree
        for tree in QUARANTINE_TREES
        if tree not in _configured_pytest_root_paths(ROCKE / tree / "conftest.py")
    ]
    assert not unconfigured, (
        f"quarantine trees whose conftest.py does not call "
        f"configure_pytest_root with its path: {unconfigured}"
    )


@pytest.mark.skipif(not QUARANTINED, reason="no test is quarantined")
def test_a_quarantined_test_is_skipped_with_its_issue(tmp_path):
    entry = QUARANTINED[0]
    environment = {k: v for k, v in os.environ.items() if k != "ROCKE_QUARANTINE"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            f"--basetemp={tmp_path}",
            "-rs",
            entry.test_id,
        ],
        cwd=ROCKE,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert f"quarantined: {entry.issue}" in result.stdout, result.stdout[-4000:]
    assert " passed" not in result.stdout and " failed" not in result.stdout


def _defines_test(test_file: Path, node_name: str) -> bool:
    """Whether ``test_file`` defines the test function, or the test class,
    ``node_name`` names last."""
    name = node_name.split("::")[-1].partition("[")[0]
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == name
        for node in ast.walk(tree)
    )


def quarantine_breaches(entries: Iterable[QuarantinedTest]) -> list[str]:
    """What fails the quarantine record, whatever the change: two entries
    that may hold out the same item, or more items than ``QUARANTINE_CAP``. An
    expired entry fails as any expired record does
    (:func:`fail_or_report_expired`)."""
    entries = list(entries)
    breaches = [
        f"quarantined twice: {first.test_id} and {second.test_id} may hold out "
        "the same tests; keep one entry per test"
        for first, second in itertools.combinations(entries, 2)
        if first.overlaps(second)
    ]
    if len(entries) > QUARANTINE_CAP:
        breaches.append(
            f"{len(entries)} quarantined tests, over the cap of {QUARANTINE_CAP}; "
            "fix one before quarantining another"
        )
    return breaches


def test_a_quarantine_fails_when_listed_twice_or_over_the_cap():
    current = _quarantined("library/tests/test_x.py::test_y")
    assert quarantine_breaches([current]) == []
    # Expiry is not a breach whatever the change; an expired quarantine fails
    # as test_an_expired_quarantine_fails_a_change_that_touches_it says.
    expired = _quarantined("library/tests/test_x.py::test_z", expires="2026-01-01")
    assert quarantine_breaches([current, expired]) == []
    assert quarantine_breaches([current, current]) == [
        (
            "quarantined twice: library/tests/test_x.py::test_y and "
            "library/tests/test_x.py::test_y may hold out the same tests; keep one "
            "entry per test"
        )
    ]
    # A function and one of its cases, two narrowings of one function, or one
    # function on two arches: an item both cover would be held out only where
    # the first one says, so one entry names both arches.
    method = "library/tests/test_x.py::TestX::test_y"
    for first, second in (
        (_quarantined(method), _quarantined(f"{method}[a]", arches=("gfx950",))),
        (
            _quarantined(method, params=("bf16",)),
            _quarantined(method, params=("mha",)),
        ),
        (
            _quarantined(method, arches=("gfx942",)),
            _quarantined(method, arches=("gfx950",)),
        ),
    ):
        assert quarantine_breaches([first, second]) == [
            (
                f"quarantined twice: {first.test_id} and {second.test_id} may hold "
                "out the same tests; keep one entry per test"
            )
        ]
    assert (
        quarantine_breaches(
            [_quarantined(f"{method}[a]"), _quarantined(f"{method}[b]")]
        )
        == []
    )
    over_the_cap = [
        _quarantined(f"library/tests/test_x.py::test_{index}")
        for index in range(QUARANTINE_CAP + 1)
    ]
    assert quarantine_breaches(over_the_cap) == [
        (
            f"{QUARANTINE_CAP + 1} quarantined tests, over the cap of "
            f"{QUARANTINE_CAP}; fix one before quarantining another"
        )
    ]


# A quarantine registry as pending_lanes.py writes one: a group of two records
# (lines 2-6, 9 and 10 shared; 7 and 8 their own) and one record alone (lines
# 12-18).
_QUARANTINE_SOURCE = """\
QUARANTINED: tuple = (
    *_quarantined(
        "AICK-1",
        "reason",
        TEAM,
        "2026-01-01",
        "library/tests/test_x.py::test_a",
        "library/tests/test_x.py::test_b",
        arches=("gfx950",),
        compilers=COMPILERS,
    ),
    QuarantinedTest(
        "library/tests/test_y.py::test_c",
        "AICK-2",
        "reason",
        TEAM,
        "2026-01-01",
    ),
)


def _quarantined(issue, reason, team, expires, *test_ids, arches=(), compilers=None):
    pass
"""


def _expired_quarantines() -> list[QuarantinedTest]:
    """The records ``_QUARANTINE_SOURCE`` spells, expired."""
    return [
        _quarantined(test_id, expires="2026-01-01")
        for test_id in (
            "library/tests/test_x.py::test_a",
            "library/tests/test_x.py::test_b",
            "library/tests/test_y.py::test_c",
        )
    ]


def _check_quarantine_expiry(change: Change | None, *, strict: bool) -> None:
    fail_or_report_expired(
        "QUARANTINED",
        _expired_quarantines(),
        lambda entry: entry.test_id,
        source=_QUARANTINE_SOURCE,
        change=change,
        strict=strict,
        max_days=QUARANTINE_MAX_DAYS,
        today=datetime.date(2026, 1, 2),
    )


def _change(files=(), lines=()) -> Change:
    return Change(frozenset(files), frozenset(lines), frozenset())


def test_an_expired_quarantine_fails_a_change_that_touches_it():
    """Its own line in the registry, its group's shared lines, or its test
    file: the change that touches an expired quarantine re-decides it."""
    with pytest.raises(AssertionError, match=r"\['library/tests/test_y.py::test_c'\]"):
        _check_quarantine_expiry(_change(lines={15}), strict=False)
    with pytest.raises(AssertionError, match=r"\['library/tests/test_x.py::test_a'\]"):
        _check_quarantine_expiry(_change(lines={7}), strict=False)
    with pytest.raises(
        AssertionError,
        match=(
            r"\['library/tests/test_x.py::test_a', "
            r"'library/tests/test_x.py::test_b'\]"
        ),
    ):
        _check_quarantine_expiry(_change(lines={6}), strict=False)
    with pytest.raises(AssertionError, match=r"\['library/tests/test_y.py::test_c'\]"):
        _check_quarantine_expiry(
            _change(files={"library/tests/test_y.py"}), strict=False
        )
    with pytest.raises(
        AssertionError, match=f"at most {QUARANTINE_MAX_DAYS} days away"
    ):
        _check_quarantine_expiry(_change(lines={15}), strict=False)


def test_an_expired_quarantine_is_a_warning_for_a_change_that_does_not_touch_it():
    """An expired quarantine does not turn an unrelated change red, nor a run
    with no change to compare (develop's own CI): it is reported. The entry
    still covers its test, whose skip does not read the date."""
    for change in (_change(files={"library/tests/test_z.py"}, lines={30}), None):
        with pytest.warns(UserWarning, match="QUARANTINED entries past their expiry"):
            _check_quarantine_expiry(change, strict=False)
    expired = _quarantined("library/tests/test_x.py::test_a", expires="2026-01-01")
    assert expired.covers("library/tests/test_x.py", "test_a")


def test_every_expired_quarantine_fails_under_strict_expiry():
    """The scheduled run (--strict-expiry) fails on every expired quarantine,
    whatever the change."""
    with pytest.raises(AssertionError, match="test_a.*test_b.*test_c"):
        _check_quarantine_expiry(None, strict=True)


# The files that may name the runner lane or write a lane declaration by hand:
# the lanes module, the registry, and the tests of the test helpers (which
# assert the reasons the helpers write) and of the registry.
_LANE_HOMES = frozenset(
    {
        "platform/python/rocke/tests/lanes.py",
        "platform/python/rocke/tests/pending_lanes.py",
    }
)
_LANE_TESTS = (
    "platform/tests/support/",
    "platform/tests/tools/test_limbo_audit.py",
    "platform/tests/tools/test_pending_lanes.py",
)
# Out-of-tree build directories a developer may have made under a root.
_BUILD_DIR = re.compile(r"build([-_].*)?")


def _python_source_paths() -> list[str]:
    """Each Python file of the test file roots, outside the lane homes and
    their tests, relative to the rocke directory."""
    paths = []
    for root in ("platform", "library"):
        for file in (ROCKE / root).rglob("*.py"):
            parts = file.relative_to(ROCKE).parts
            if "__pycache__" in parts or any(_BUILD_DIR.fullmatch(p) for p in parts):
                continue
            path = file.relative_to(ROCKE).as_posix()
            if path in _LANE_HOMES or path.startswith(_LANE_TESTS):
                continue
            paths.append(path)
    return paths


@functools.cache
def _source_lane_mentions() -> tuple[LaneMention, ...]:
    """The lane mentions of :func:`_python_source_paths`, parsed once for the
    checks that read them."""
    return tuple(lane_mentions(ROCKE, _python_source_paths()))


def test_tests_state_the_arches_they_need_not_the_runner_lane():
    """A test names the arches or the GPU count it needs (``skip_needing_arch``,
    ``arch_skip_reason``, ``skip_needing_gpus``); that this waits for the
    runner hardware is a fact of ``rocke.tests.lanes`` alone, so the runner
    lane is named only in the lane homes and their tests. A skip says what CI
    lacks only through the lane helpers: a hand-written ``waits for ...``
    reason, or one about CI or a runner, is refused everywhere else. Only a
    reason written at the skip is read; one passed through a variable is not
    seen here (the limbo audit sees its case)."""
    named, hand_written = [], []
    for mention in _source_lane_mentions():
        if mention.kind is LaneMentionKind.RUNNER_LANE_NAME:
            named.append(f"{mention.path}:{mention.line} {mention.text}")
        elif mention.kind in (
            LaneMentionKind.HAND_WRITTEN_DECLARATION,
            LaneMentionKind.HAND_WRITTEN_SKIP_REASON,
        ):
            hand_written.append(f"{mention.path}:{mention.line}")
    assert not named, (
        "name the arches the case needs (skip_needing_arch / arch_skip_reason) "
        f"instead of the runner lane: {named}"
    )
    assert not hand_written, (
        "say what CI lacks with rocke.tests.lanes (skip_until, skip_until_reason, "
        "skip_needing_arch, arch_skip_reason, import_or_skip_until), not a "
        f"hand-written lane reason: {sorted(set(hand_written))}"
    )


def test_a_case_that_needs_runner_hardware_has_a_record():
    """A case that needs a GPU no pull-request runner has skips there as
    waiting for the runner hardware; its file needs a record for it (or a
    whole-file record), or it would be in limbo with no check on a pull
    request to say so."""
    recorded = {
        entry.path
        for entry in PENDING
        if entry.whole_file or entry.lane is RUNNER_HARDWARE
    }
    unrecorded = [
        f"{mention.path}:{mention.line} needs one of {list(mention.arches)}"
        for mention in _source_lane_mentions()
        if mention.kind is LaneMentionKind.RUNNER_HARDWARE_NEED
        and posixpath.basename(mention.path).startswith("test_")
        and mention.path not in recorded
    ]
    assert not unrecorded, (
        "cases that need a GPU no pull-request runner has, with no record: add a "
        "PendingTest(<file>, RUNNER_HARDWARE, ...) to rocke.tests.pending_lanes: "
        f"{unrecorded}"
    )


def codeowners_teams(codeowners: str, path: str) -> set[str]:
    """The teams the CODEOWNERS text ``codeowners`` gives the repository path
    ``path``: those of its last rule for a directory above it (rocKE's rules
    are directory rules)."""
    teams: set[str] = set()
    for line in codeowners.splitlines():
        words = line.partition("#")[0].split()
        if not words:
            continue
        pattern, *owners = words
        if pattern.endswith("/") and f"/{path}".startswith(pattern):
            teams = {owner.removeprefix("@") for owner in owners}
    return teams


def test_codeowners_rules_give_a_path_the_teams_of_its_last_directory_rule():
    codeowners = (
        "# a comment\n"
        "\n"
        "/group/project/ @ROCm/project-platform @ROCm/project-lib\n"
        "/group/project/library/ @ROCm/project-lib  # the library\n"
        "/group/project/library/README.md @ROCm/docs\n"
    )
    assert codeowners_teams(codeowners, "group/project/library/tests/test_x.py") == {
        "ROCm/project-lib"
    }
    assert codeowners_teams(codeowners, "group/project/tools/x.py") == {
        "ROCm/project-platform",
        "ROCm/project-lib",
    }
    assert codeowners_teams(codeowners, "group/other/x.py") == set()


def _codeowners() -> str | None:
    """The repository's CODEOWNERS, from the checkout or, in a sparse one,
    from git; None when neither has it."""
    file = ROCKE.parents[2] / ".github" / "CODEOWNERS"
    if file.is_file():
        return file.read_text(encoding="utf-8")
    return _git("show", "HEAD:.github/CODEOWNERS")


def test_every_record_names_a_team_that_owns_its_test():
    codeowners = _codeowners()
    if codeowners is None:
        pytest.skip("this checkout has no .github/CODEOWNERS, in the tree or in git")
    prefix = ROCKE.relative_to(ROCKE.parents[2]).as_posix()
    records = [
        *((entry.path, entry.team) for entry in PENDING),
        *((budget.path, budget.team) for budget in MEASURED_BUDGETS),
        *((entry.path, entry.team) for entry in QUARANTINED),
    ]
    unowned = sorted(
        f"{path} ({team})"
        for path, team in records
        if team not in codeowners_teams(codeowners, f"{prefix}/{path}")
    )
    assert (
        not unowned
    ), f"records whose team is not a CODEOWNERS team of their test: {unowned}"


def test_every_quarantined_test_is_current(request):
    breaches = quarantine_breaches(QUARANTINED)
    assert not breaches, breaches
    gone = [
        entry.test_id
        for entry in QUARANTINED
        if not (ROCKE / entry.path).is_file()
        or not _defines_test(ROCKE / entry.path, entry.node_name)
        or not all((ROCKE / child).is_file() for child in entry.runs_files)
    ]
    assert not gone, f"quarantined but gone: {gone}"
    _check_expiry(
        request,
        "QUARANTINED",
        QUARANTINED,
        lambda entry: entry.test_id,
        QUARANTINE_MAX_DAYS,
    )


@pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("ctest") is None,
    reason="CMake and CTest required",
)
# The provider's install paths this audit maps back to the source are POSIX
# paths only; the registration it audits is the same on every platform.
@pytest.mark.skipif(
    sys.platform == "win32", reason="the provider's install paths are POSIX paths"
)
def test_every_test_file_is_assigned_to_a_ci_entry_or_has_a_recorded_reason(tmp_path):
    in_limbo, stale = audit(
        tree_test_files(ROCKE), ci_selected_files(tmp_path), PENDING
    )
    assert not in_limbo, (
        f"{len(in_limbo)} test files are assigned to no CI entry and are not in "
        f"rocke.tests.pending_lanes: {in_limbo}"
    )
    assert not stale, (
        "records whose scope no longer fits (a whole-file record's file is now "
        "assigned to a CI entry, or a case record's file is assigned to none): "
        f"{stale}"
    )


# The template platform/CMakeLists.txt registers each published operation and
# architecture's GPU reference entry with, and the test file the entry runs.
_REFERENCE_ENTRY_TEMPLATE = re.compile(
    r"add_test\((rocke_\$\{_operation\}_gpu_\$\{_arch\}_pytest) \S+ "
    r'\\"\$\{ROCKE_TEST_CTEST_REL\}tests/'
    r'(library/tests/test_\$\{_operation\}_pinned_reference\.py)\\"'
)


def _render(template: str, operation: str, arch: str) -> str:
    return template.replace("${_operation}", operation).replace("${_arch}", arch)


def test_pinned_reference_entries_are_the_published_operations_registrations(
    tmp_path,
):
    """The audit credits each pinned-reference file with the entries the CMake
    template registers for the operations and pull-request runner arches the
    committed locks publish; a pinned-reference file planted beside them
    without a reference operation is credited with none."""
    name_template, file_template = _REFERENCE_ENTRY_TEMPLATE.search(
        (ROCKE / "platform/CMakeLists.txt").read_text(encoding="utf-8")
    ).groups()
    published = {
        (variant["operation"], variant["architecture"])
        for variant in published_reference_variants(ROCKE / "library")
    }
    expected: dict[str, set[str]] = {}
    for operation, arch in published:
        if arch in PR_RUNNER_ARCHES:
            expected.setdefault(_render(file_template, operation, arch), set()).add(
                _render(name_template, operation, arch)
            )
    assert expected, "no published pinned reference on a pull-request runner arch"
    assert pinned_reference_entries(ROCKE / "library") == expected

    # The same locks in a copy of the library tree, with a planted file.
    library = tmp_path / "library"
    shutil.copytree(
        ROCKE / "library/tests/reference_common", library / ARTIFACT_CLI.parent
    )
    for operation in {operation for operation, _ in published}:
        package = f"tests/{operation}_reference/architectures"
        shutil.copytree(ROCKE / "library" / package, library / package)
    planted = library / "tests/test_bogus_pinned_reference.py"
    planted.write_text("def test_x():\n    pass\n", encoding="utf-8")
    assert pinned_reference_entries(library) == expected
