# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No test file in limbo: each runs in a CI lane or has a recorded reason.

Pull-request and push CI select rocKE's CTest entries by the quick and standard
tier labels the provider categories give them. A test file is in CI when one of
those entries selects it: a build-tree entry names source paths; an installed
entry names paths of the install layout, which the configured install rules map
back to the source (so a file the install leaves out is not in CI even when its
directory is). A file a selected test runs as a child process counts too. A
module whose ``pytestmark`` the entry's ``-m`` expression rejects is not in CI
through that entry. The check works per file: marks on single tests, ``-k``
filters and a file whose every case skips in CI are not seen here, so
``rocke.tests.pending_lanes`` records those files under the lane their cases
declare where they skip (``rocke.tests.lanes``), and ``tools/limbo_audit.py``
checks each case against the record from the run records of CI lanes.

Every other test file must be listed in ``rocke.tests.pending_lanes``, and every
listed entry must be current: its file present, and in no CI-run entry for a
whole-file entry, in one otherwise. A file is listed once per lane, and a
whole-file entry is its file's only entry. Each record's team owns its path in
CODEOWNERS. A file whose cases need a GPU no pull-request runner has (an arch
helper called with such arches) has a record for them, and no test writes a
lane reason by hand. An expired entry fails a change that touches it (its lines
in the registry, or its test file), and every run under ``--strict-expiry``;
otherwise it is reported as a warning. The registry's measured budgets are held
to the same rules: each names a present test file, is listed once, and is
reported or failed when expired the same way. A quarantined test is held to
stricter rules: it names a test that exists, is listed once, fails the check as
soon as it expires, and the registry holds at most ``QUARANTINE_CAP`` of them.

The check configures the platform (nothing is built) and reads the provider
categories, so it runs in a source checkout of the monorepo only.
"""

from __future__ import annotations

import ast
import datetime
import inspect
import os
import posixpath
import re
import shutil
import subprocess
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pytest
from _pytest.mark.expression import Expression
from rocke.assets import platform_root
from rocke.tests import gpu as rocke_gpu
from rocke.tests import in_tree_suites
from rocke.tests import lanes as rocke_lanes
from rocke.tests.comgr import CompilerBuild
from rocke.tests.ctest_listing import configure_installed_ctest, list_ctest_entries
from rocke.tests.lanes import (
    BUILD_VARIANT,
    RUNNER_HARDWARE,
    SOURCE_TREE,
    TICKET,
    Lane,
    runner_lanes_for,
    skip_until_reason,
)
from rocke.tests.pending_lanes import (
    MEASURED_BUDGETS,
    PENDING,
    PENDING_MAX_DAYS,
    QUARANTINE_CAP,
    QUARANTINE_MAX_DAYS,
    QUARANTINE_TREES,
    QUARANTINED,
    TEAMS,
    Compilers,
    MeasuredBudget,
    PendingTest,
    QuarantinedTest,
    measured_budget_kernels,
    measured_scratch_allowance,
    tree_test_files,
    waiting_entry,
)
from rocke.tests.provider_categories import provider_category_labels
from rocke.tests.quarantine import register_quarantine, skip_quarantined_items

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
# over this file of each reference operation only when the reference bundles
# are installed, which the provider build does and this check's configure
# (without the bundles) cannot.
# The registry module, relative to the rocke directory.
PENDING_LANES_MODULE = "platform/python/rocke/tests/pending_lanes.py"
PINNED_REFERENCE_TEST = re.compile(r"library/tests/test_(\w+)_pinned_reference\.py")


def _normalized(path: str) -> str:
    """``path`` with its "." and ".." parts folded away; an absolute path stays
    absolute, and a ".." above a relative path's start stays in place."""
    return posixpath.normpath(path)


def _is_within(path: str, directory: str) -> bool:
    return path == directory or path.startswith(f"{directory}/")


def _is_test_file(path: Path) -> bool:
    return path.name.startswith("test_") and path.suffix == ".py"


@dataclass(frozen=True)
class InstallRule:
    """One ``file(INSTALL ...)`` call of a generated ``cmake_install.cmake``."""

    destination: str  # under the install prefix
    sources: tuple[str, ...]  # as the script spells them: "dir/" vs "dir"
    is_directory: bool
    excludes: tuple[re.Pattern[str], ...]

    def copies(self) -> list[tuple[str, Path]]:
        """(installed path, source path) of each file or directory it copies;
        "dir/" copies the contents of dir, "dir" copies dir itself."""
        return [
            (
                (
                    self.destination
                    if self.is_directory and source.endswith("/")
                    else f"{self.destination}/{Path(source).name}"
                ),
                Path(source),
            )
            for source in self.sources
        ]

    def source_of(self, installed: str) -> Path | None:
        """The existing source path this rule copies to ``installed``, or
        None. An excluded directory excludes everything beneath it."""
        for into, source in self.copies():
            if not _is_within(installed, into):
                continue
            if not self.is_directory:
                return source if installed == into else None
            candidate = source
            for part in Path(installed[len(into) :].lstrip("/")).parts:
                candidate = candidate / part
                if any(rule.search(candidate.as_posix()) for rule in self.excludes):
                    return None
            return candidate if candidate.exists() else None
        return None


_INSTALL_CALL = "file(INSTALL"
_INSTALL_PREFIX = "${CMAKE_INSTALL_PREFIX}/"
# A quoted argument, the closing parenthesis, or a bare argument.
_TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|(\))|([^\s")]+)')


def install_rules(cmake_install: str) -> list[InstallRule]:
    """The rules of a generated ``cmake_install.cmake``: each call reads
    ``DESTINATION <dir> TYPE <kind> [RENAME <name>] FILES <path>...`` and then
    ``REGEX <pattern> EXCLUDE`` pairs among other options."""
    rules = []
    start = cmake_install.find(_INSTALL_CALL)
    while start >= 0:
        tokens = []
        end = start + len(_INSTALL_CALL)
        for match in _TOKEN.finditer(cmake_install, end):
            end = match.end()
            quoted, close, bare = match.groups()
            if close:
                break
            # CMake escapes a backslash in a quoted argument as two.
            tokens.append(quoted.replace("\\\\", "\\") if quoted is not None else bare)
        start = cmake_install.find(_INSTALL_CALL, end)
        keywords = {
            token: index for index, token in enumerate(tokens) if token.isupper()
        }
        if not {"DESTINATION", "TYPE", "FILES"} <= keywords.keys():
            continue
        destination = tokens[keywords["DESTINATION"] + 1]
        if not destination.startswith(_INSTALL_PREFIX):
            continue
        files = []
        for token in tokens[keywords["FILES"] + 1 :]:
            if token.isupper():
                break
            files.append(token)
        rules.append(
            InstallRule(
                destination=destination.removeprefix(_INSTALL_PREFIX),
                sources=tuple(files),
                is_directory=tokens[keywords["TYPE"] + 1] == "DIRECTORY",
                excludes=tuple(
                    re.compile(tokens[index + 1])
                    for index, token in enumerate(tokens[:-2])
                    if token == "REGEX" and tokens[index + 2] == "EXCLUDE"
                ),
            )
        )
    return rules


class Layout(Protocol):
    """The path space a CTest command's arguments are in."""

    def source_of(self, path: str) -> Path | None:
        """The source file or directory at ``path``, or None."""

    def test_files_under(self, directory: str) -> dict[str, Path]:
        """Each test file under ``directory``: its path here -> its source."""


class SourceLayout:
    """A build-tree command's arguments: source paths themselves."""

    def source_of(self, path: str) -> Path | None:
        return Path(path)

    def test_files_under(self, directory: str) -> dict[str, Path]:
        root = Path(directory)
        return {
            f"{directory}/{file.relative_to(root).as_posix()}": file
            for file in root.rglob("test_*.py")
        }


class InstallLayout:
    """An installed command's arguments: install-layout paths, mapped back to
    the source through the install rules."""

    def __init__(self, rules: list[InstallRule]) -> None:
        self.rules = rules

    def source_of(self, path: str) -> Path | None:
        for rule in self.rules:
            source = rule.source_of(path)
            if source is not None:
                return source
        return None

    def test_files_under(self, directory: str) -> dict[str, Path]:
        """Every test file the install puts under ``directory``, from every rule
        that copies into it."""
        files = {}
        for rule in self.rules:
            for into, source in rule.copies():
                if _is_within(into, directory):
                    top, root = into, source
                elif rule.is_directory and _is_within(directory, into):
                    top, root = directory, rule.source_of(directory)
                else:
                    continue
                if root is None or not root.exists():
                    continue
                found = [root] if root.is_file() else root.rglob("test_*.py")
                for file in filter(_is_test_file, found):
                    there = (
                        top
                        if file == root
                        else f"{top}/{file.relative_to(root).as_posix()}"
                    )
                    if rule.source_of(there) == file:
                        files[there] = file
        return files


@dataclass(frozen=True)
class PytestSelection:
    """What a command's pytest arguments select, file by file."""

    paths: tuple[str, ...]
    ignored: tuple[str, ...]
    # The last -m expression, as pytest keeps it; None when there is none.
    marker_expression: str | None


def pytest_selection(arguments: list[str]) -> PytestSelection:
    """The paths a command's arguments name, the ones it ``--ignore``s, and
    its ``-m`` expression."""
    paths, ignored, marker_expression = [], [], None
    takes_value = ("-c", "-k", "-m", "-p", "--deselect", "--ignore")
    for index, argument in enumerate(arguments):
        option = arguments[index - 1] if index else None
        if option in takes_value:
            if option == "--ignore":
                ignored.append(argument)
            elif option == "-m":
                marker_expression = argument
        elif argument.startswith("--ignore="):
            ignored.append(argument.removeprefix("--ignore="))
        elif not argument.startswith("-"):
            # A node id (``file.py::test``) runs part of its file: the file is in.
            paths.append(argument.split("::", 1)[0])
    return PytestSelection(tuple(paths), tuple(ignored), marker_expression)


def module_marks(test_file: Path) -> set[str]:
    """The marker names a test module's ``pytestmark`` sets: each
    ``pytest.mark.<name>`` (called or not) it assigns, alone or in a list."""
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    marks = set()
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in targets
        ):
            continue
        value = statement.value
        for mark in value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]:
            if isinstance(mark, ast.Call):
                mark = mark.func
            if (
                isinstance(mark, ast.Attribute)
                and isinstance(mark.value, ast.Attribute)
                and mark.value.attr == "mark"
                and isinstance(mark.value.value, ast.Name)
                and mark.value.value.id == "pytest"
            ):
                marks.add(mark.attr)
    return marks


def deselects_module(marker_expression: str | None, test_file: Path) -> bool:
    """True when ``-m marker_expression`` deselects every test of the module
    through its ``pytestmark`` alone (marks on single tests are not read)."""
    if marker_expression is None:
        return False
    marks = module_marks(test_file)
    expression = Expression.compile(marker_expression)
    return not expression.evaluate(lambda name, **_: name in marks)


def _arguments_after_interpreter(command: list[str]) -> list[str]:
    """A command's arguments past the interpreter and its ``-m pytest`` (or
    ``-u``); a script entry keeps its script path."""
    arguments = command[1:]
    while arguments and arguments[0] in ("-u", "-I", "-S"):
        arguments = arguments[1:]
    if arguments[:2] == ["-m", "pytest"]:
        arguments = arguments[2:]
    return arguments


def selected_test_files(
    arguments: list[str], layout: Layout, base: str = ""
) -> set[Path]:
    """The source test files a command selects: the test files and directories
    its arguments name in ``layout`` (relative to ``base``), less the ones it
    ``--ignore``s and the modules its ``-m`` expression deselects whole."""
    selection = pytest_selection(arguments)
    ignored = [_normalized(base + path) for path in selection.ignored]
    selected = set()
    for path in (_normalized(base + path) for path in selection.paths):
        source = layout.source_of(path)
        if source is None or not source.exists():
            continue
        files = {path: source} if source.is_file() else layout.test_files_under(path)
        for there, file in files.items():
            if any(_is_within(there, skip) for skip in ignored):
                continue
            if file.suffix == ".py" and deselects_module(
                selection.marker_expression, file
            ):
                continue
            selected.add(file.resolve())
    return selected


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


def _python_on_path(scratch: Path) -> dict[str, str]:
    """An environment in which the bare ``python`` that installed entries run
    resolves, to this interpreter when no ``python`` is on PATH."""
    environment = dict(os.environ)
    if shutil.which("python") is None:
        interpreter_dir = scratch / "python-bin"
        interpreter_dir.mkdir()
        (interpreter_dir / "python").symlink_to(sys.executable)
        environment["PATH"] = os.pathsep.join(
            [str(interpreter_dir), environment.get("PATH", "")]
        )
    return environment


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
            list_ctest_entries(installed_ctest, env=_python_on_path(scratch)),
            f"bin/{PROVIDER_CTEST_SUBDIR}/",
            install,
        ),
    ]
    references = {
        path: f"rocke_{match.group(1)}_gpu_gfx942_pytest"
        for path in tree_test_files(ROCKE)
        if (match := PINNED_REFERENCE_TEST.fullmatch(path))
    }
    entry_names = [entry["name"] for entries, _, _ in trees for entry in entries]
    labels = provider_category_labels([*entry_names, *references.values()], scratch)

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
    in_ci |= {path for path, name in references.items() if ci_run(name)}
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
    return (_utc_today() + datetime.timedelta(days=days)).isoformat()


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


def test_a_record_expires_within_its_maximum_window():
    """A pending entry or measured budget expires at most PENDING_MAX_DAYS
    days away, a quarantine at most QUARANTINE_MAX_DAYS: a record is re-decided
    in that window, not set once and forgotten. A past date is a valid
    record, which the expiry check then reports."""
    path = "library/tests/test_x.py"
    assert _pending(path, TICKET, expires=_days_from_today(PENDING_MAX_DAYS))
    assert _pending(path, TICKET, expires="2026-01-01")
    with pytest.raises(ValueError, match=f"expected at most {PENDING_MAX_DAYS} days"):
        _pending(path, TICKET, expires=_days_from_today(PENDING_MAX_DAYS + 1))
    test_id = f"{path}::test_y"
    assert _quarantined(test_id, expires=_days_from_today(QUARANTINE_MAX_DAYS))
    with pytest.raises(
        ValueError, match=f"expected at most {QUARANTINE_MAX_DAYS} days"
    ):
        _quarantined(test_id, expires=_days_from_today(QUARANTINE_MAX_DAYS + 1))
    with pytest.raises(ValueError, match=f"expected at most {PENDING_MAX_DAYS} days"):
        _measured_budget(expires=_days_from_today(PENDING_MAX_DAYS + 1))


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
    """The commit this change is checked against: ``ROCKE_CHANGE_BASE``, else
    the ``--from-ref`` of the pre-commit run (``PRE_COMMIT_FROM_REF``, which CI
    sets to the pull request's base), else the merge base with origin/develop;
    None when none resolves."""
    explicit = os.environ.get("ROCKE_CHANGE_BASE") or os.environ.get(
        "PRE_COMMIT_FROM_REF"
    )
    return explicit or _git("merge-base", "HEAD", "origin/develop")


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
    names = _git("diff", "--name-only", "--relative", base)
    diff = _git("diff", "-U0", "--relative", base, "--", PENDING_LANES_MODULE)
    if names is None or diff is None:
        return None
    lines, gaps = registry_line_changes(diff)
    return Change(frozenset(names.splitlines()), lines, gaps)


# The registry helpers that build one record per trailing argument, by the
# number of leading arguments every record shares.
_GROUP_HELPERS = {"_waiting": 5, "_measured": 3, "_quarantined": 4}


def _lines_of(node: ast.AST) -> set[int]:
    return set(range(node.lineno, node.end_lineno + 1))


def record_lines(source: str, registry_name: str) -> list[frozenset[int]]:
    """The source lines of each record of the registry tuple ``registry_name``
    in ``source``, in order: a record written alone, all its lines; one of a
    group helper's records, the helper's shared arguments and its own."""
    statement = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.AnnAssign)
        and getattr(node.target, "id", None) == registry_name
    )
    spans = []
    for element in statement.value.elts:
        if not isinstance(element, ast.Starred):
            spans.append(frozenset(_lines_of(element)))
            continue
        call = element.value
        shared_count = _GROUP_HELPERS[call.func.id]
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


def test_the_change_base_follows_the_pre_commit_from_ref(monkeypatch):
    monkeypatch.delenv("ROCKE_CHANGE_BASE", raising=False)
    monkeypatch.setenv("PRE_COMMIT_FROM_REF", "origin/develop~3")
    assert change_base() == "origin/develop~3"
    monkeypatch.setenv("ROCKE_CHANGE_BASE", "abc123")
    assert change_base() == "abc123"


def _fail_or_report_expired(
    request, registry_name: str, records: Iterable, label
) -> None:
    """Fail on the expired records of the registry tuple ``registry_name``
    that :func:`expiry_failures` says fail this run; report the rest as a
    warning. ``label`` names a record in the message."""
    today = _utc_today()
    source = (ROCKE / PENDING_LANES_MODULE).read_text(encoding="utf-8")
    spans = record_lines(source, registry_name)
    records = list(records)
    assert len(spans) == len(records), (registry_name, len(spans), len(records))
    expired = [
        ExpiredRecord(label(record), record.path, lines)
        for record, lines in zip(records, spans)
        if record.expiry_date < today
    ]
    if not expired:
        return
    strict = request.config.getoption("--strict-expiry")
    failing = expiry_failures(expired, current_change(), strict=strict)
    assert not failing, (
        f"expired {registry_name} entries this change touches; re-decide them "
        f"(a new expiry at most {PENDING_MAX_DAYS} days away, or clear them): "
        f"{failing}"
    )
    message = (
        f"{registry_name} entries past their expiry: {sorted(r.label for r in expired)}"
    )
    print(message)
    warnings.warn(message, stacklevel=2)


def _utc_today() -> datetime.date:
    return datetime.datetime.now(datetime.timezone.utc).date()


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
    _fail_or_report_expired(
        request, "PENDING", PENDING, lambda p: f"{p.path} ({p.lane.name})"
    )


def test_every_measured_budget_is_current(request):
    keys = [(budget.path, budget.kernel_name) for budget in MEASURED_BUDGETS]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    assert not duplicates, f"measured budget listed more than once: {duplicates}"
    missing = sorted(
        {b.path for b in MEASURED_BUDGETS if not (ROCKE / b.path).is_file()}
    )
    assert not missing, f"measured budgets for test files that are gone: {missing}"
    _fail_or_report_expired(
        request,
        "MEASURED_BUDGETS",
        MEASURED_BUDGETS,
        lambda b: f"{b.path}::{b.kernel_name}",
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


def quarantine_breaches(
    entries: Iterable[QuarantinedTest], today: datetime.date
) -> list[str]:
    """What fails the quarantine record, whatever the change: an item listed
    twice, more items than ``QUARANTINE_CAP``, or an expired entry."""
    entries = list(entries)
    ids = [entry.test_id for entry in entries]
    breaches = [
        f"quarantined more than once: {test_id}"
        for test_id in sorted({test_id for test_id in ids if ids.count(test_id) > 1})
    ]
    if len(entries) > QUARANTINE_CAP:
        breaches.append(
            f"{len(entries)} quarantined tests, over the cap of {QUARANTINE_CAP}; "
            "fix one before quarantining another"
        )
    breaches += [
        f"quarantine expired: {entry.test_id}; fix the test, retire it, or extend "
        "the expiry with the reason recorded in its issue"
        for entry in entries
        if entry.expiry_date < today
    ]
    return breaches


def test_a_quarantine_fails_when_expired_listed_twice_or_over_the_cap():
    today = _utc_today()
    current = _quarantined("library/tests/test_x.py::test_y")
    assert quarantine_breaches([current], today) == []
    expired = _quarantined("library/tests/test_x.py::test_z", expires="2026-01-01")
    assert quarantine_breaches([current, expired], today) == [
        (
            "quarantine expired: library/tests/test_x.py::test_z; fix the test, "
            "retire it, or extend the expiry with the reason recorded in its issue"
        )
    ]
    assert quarantine_breaches([current, current], today) == [
        "quarantined more than once: library/tests/test_x.py::test_y"
    ]
    over_the_cap = [
        _quarantined(f"library/tests/test_x.py::test_{index}")
        for index in range(QUARANTINE_CAP + 1)
    ]
    assert quarantine_breaches(over_the_cap, today) == [
        (
            f"{QUARANTINE_CAP + 1} quarantined tests, over the cap of "
            f"{QUARANTINE_CAP}; fix one before quarantining another"
        )
    ]


# The files that may name the runner lane or write a lane declaration by hand:
# the lanes module, the registry, and the tests of both.
_LANE_HOMES = frozenset(
    {
        "platform/python/rocke/tests/lanes.py",
        "platform/python/rocke/tests/pending_lanes.py",
    }
)
_LANE_TEST_DIRS = ("platform/tests/support/", "platform/tests/tools/")


def _python_sources() -> Iterable[tuple[str, ast.Module]]:
    """Each Python file of the test file roots, outside the lane homes and
    their tests, relative to the rocke directory, with its syntax tree."""
    for root in ("platform", "library"):
        for file in (ROCKE / root).rglob("*.py"):
            parts = file.relative_to(ROCKE).parts
            if "__pycache__" in parts or any(p.startswith("build") for p in parts):
                continue
            path = file.relative_to(ROCKE).as_posix()
            if path in _LANE_HOMES or path.startswith(_LANE_TEST_DIRS):
                continue
            yield path, ast.parse(file.read_text(encoding="utf-8"))


def _declares_a_lane(text: str) -> bool:
    """Whether ``text`` reads as a lane declaration, a known lane or not."""
    try:
        return bool(rocke_lanes.lanes_waited_for(text))
    except ValueError:
        return True


# What a skip reason says when it claims CI lacks something: that the case
# waits for something, or a CI runner or CI itself.
_LANE_WORDS = re.compile(r"(?i:\bwait(?:s|ing)?\s+(?:for|on)\b|\brunners?\b)|\bCI\b")
# The pytest and unittest calls that skip, with the index of their first
# reason argument (``skipif(condition, reason)``); a keyword ``reason`` or
# ``msg`` is a reason too.
_SKIP_CALLS = {
    "skip": 0,
    "skipTest": 0,
    "skipif": 1,
    "skipIf": 1,
    "skipUnless": 1,
    "importorskip": 2,
}


def _written_text(node: ast.AST) -> str:
    """The text written in ``node`` itself: its string constants, f-string
    parts included, but none a call returns (a lane helper's reason)."""
    if isinstance(node, ast.Call):
        return ""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else ""
    return " ".join(_written_text(child) for child in ast.iter_child_nodes(node))


def hand_written_lane_reasons(tree: ast.Module) -> list[int]:
    """The lines of the skips in ``tree`` whose written reason says what CI
    lacks (:data:`_LANE_WORDS`) instead of naming a lane through
    ``rocke.tests.lanes``, or reads as a lane declaration."""
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        if name not in _SKIP_CALLS:
            continue
        reasons = node.args[_SKIP_CALLS[name] :] + [
            keyword.value
            for keyword in node.keywords
            if keyword.arg in ("reason", "msg")
        ]
        text = " ".join(_written_text(reason) for reason in reasons)
        if _LANE_WORDS.search(text) or _declares_a_lane(text):
            lines.append(node.lineno)
    return lines


def test_a_hand_written_lane_reason_is_found():
    tree = ast.parse(
        "import pytest, unittest\n"
        "pytest.skip('waits for a gfx950 runner in CI')\n"
        "pytest.mark.skipif(True, reason='no gfx950 runner')\n"
        "unittest.skipUnless(False, f'waiting on {x} in CI')\n"
        "pytest.skip('waits for ticket: torch')\n"
        "pytest.skip(msg='CI has no gfx1250')\n"
        "pytest.skip(skip_until_reason(TICKET, reason='waits for torch'))\n"
        "pytest.skip('needs ctest on PATH')\n"
        "pytest.mark.skipif(True, reason='the lane map is gfx942-only')\n"
        "pytest.importorskip('yaml', reason='needs PyYAML')\n"
    )
    assert hand_written_lane_reasons(tree) == [2, 3, 4, 5, 6]


def test_tests_state_the_arches_they_need_not_the_runner_lane():
    """A test names the arches it needs (``skip_needing_arch`` or
    ``arch_skip_reason``); that this waits for the runner hardware is a fact
    of ``rocke.tests.lanes`` alone, so the runner lane is named only in the
    lane homes. A skip says what CI lacks only through the lane helpers: a
    hand-written ``waits for ...`` reason, or one about CI or a runner, is
    refused everywhere else."""
    runner_lanes = {
        name
        for name, value in vars(rocke_lanes).items()
        if isinstance(value, Lane) and value.arches
    }
    named, hand_written = [], []
    for path, tree in _python_sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in runner_lanes:
                named.append(f"{path}:{node.lineno} {node.id}")
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and _declares_a_lane(node.value)
            ):
                hand_written.append(f"{path}:{node.lineno}")
        hand_written += [f"{path}:{line}" for line in hand_written_lane_reasons(tree)]
    assert not named, (
        "name the arches the case needs (skip_needing_arch / arch_skip_reason) "
        f"instead of the runner lane: {named}"
    )
    assert not hand_written, (
        "say what CI lacks with rocke.tests.lanes (skip_until, skip_until_reason, "
        "skip_needing_arch, arch_skip_reason, import_or_skip_until), not a "
        f"hand-written lane reason: {sorted(set(hand_written))}"
    )


# The helpers a test passes the arches its case needs, which declare the runner
# hardware when no pull-request runner has any of them.
_ARCH_NEED_HELPERS = {
    helper.__name__: list(inspect.signature(helper).parameters).index("arches")
    for helper in (
        rocke_lanes.skip_needing_arch,
        rocke_lanes.arch_skip_reason,
        rocke_gpu.current_device_arch_in,
        rocke_gpu.supported_device_arch,
        rocke_gpu.run_on_device,
    )
}


def _literal_strings(node: ast.AST | None) -> tuple[str, ...] | None:
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)) and all(
        isinstance(element, ast.Constant) and isinstance(element.value, str)
        for element in node.elts
    ):
        return tuple(element.value for element in node.elts)
    return None


def runner_hardware_needs(tree: ast.Module) -> list[tuple[int, tuple[str, ...]]]:
    """(line, arches) of each arch helper call in ``tree`` whose arches, a
    literal or a module-level name bound to one, no pull-request runner has.
    Arches computed at run time are not seen; the limbo audit sees their
    cases."""
    module_arches = {
        target.id: arches
        for statement in tree.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        and (arches := _literal_strings(statement.value)) is not None
        for target in (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if isinstance(target, ast.Name)
    }
    needs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        if name not in _ARCH_NEED_HELPERS:
            continue
        index = _ARCH_NEED_HELPERS[name]
        argument = (
            node.args[index]
            if len(node.args) > index
            else next((kw.value for kw in node.keywords if kw.arg == "arches"), None)
        )
        arches = _literal_strings(argument)
        if arches is None and isinstance(argument, ast.Name):
            arches = module_arches.get(argument.id)
        if arches and runner_lanes_for(arches):
            needs.append((node.lineno, arches))
    return needs


def test_runner_hardware_needs_are_found_from_literal_arches():
    tree = ast.parse(
        "ARCHES = ('gfx950',)\n"
        "BOTH = ('gfx942', 'gfx950')\n"
        "run_on_device(kernel, buffers, 64, ('gfx950',))\n"
        "run_on_device(kernel, buffers, 64, BOTH)\n"
        "gpu.current_device_arch_in(ARCHES, subject='x')\n"
        "skip_needing_arch(('gfx1250',), reason='x')\n"
        "supported_device_arch(kernel, arches=['gfx1151', 'gfx1201'])\n"
        "arch_skip_reason(computed_arches(), 'x')\n"
    )
    assert runner_hardware_needs(tree) == [
        (3, ("gfx950",)),
        (5, ("gfx950",)),
        (6, ("gfx1250",)),
        (7, ("gfx1151", "gfx1201")),
    ]


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
        f"{path}:{line} needs one of {list(arches)}"
        for path, tree in _python_sources()
        if posixpath.basename(path).startswith("test_") and path not in recorded
        for line, arches in runner_hardware_needs(tree)
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


def test_every_quarantined_test_is_current():
    breaches = quarantine_breaches(QUARANTINED, _utc_today())
    assert not breaches, breaches
    gone = [
        entry.test_id
        for entry in QUARANTINED
        if not (ROCKE / entry.path).is_file()
        or not _defines_test(ROCKE / entry.path, entry.node_name)
        or not all((ROCKE / child).is_file() for child in entry.runs_files)
    ]
    assert not gone, f"quarantined but gone: {gone}"


@pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("ctest") is None,
    reason="CMake and CTest required",
)
# The provider's install paths this audit maps back to the source are POSIX
# paths only; the registration it audits is the same on every platform.
@pytest.mark.skipif(
    sys.platform == "win32", reason="the provider's install paths are POSIX paths"
)
def test_every_test_file_runs_in_ci_or_has_a_recorded_reason(tmp_path):
    in_limbo, stale = audit(
        tree_test_files(ROCKE), ci_selected_files(tmp_path), PENDING
    )
    assert not in_limbo, (
        f"{len(in_limbo)} test files run in no CI-run entry and are not in "
        f"rocke.tests.pending_lanes: {in_limbo}"
    )
    assert not stale, (
        "listed under a lane whose kind no longer fits (a file waiting for a new "
        f"lane is now in CI, or one waiting for something CI lacks left CI): {stale}"
    )
