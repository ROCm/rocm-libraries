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
``rocke.tests.pending_lanes`` records those files under a lane by hand.

Every other test file must be listed in ``rocke.tests.pending_lanes``, and every
listed entry must be current: its file present and its lane's kind still
true of the file (a file waiting for a new lane is in no CI-run entry; a
file waiting for something CI lacks is in one). An expired entry fails only a
change that touches the registry or its file, and every run under
``--strict-expiry``; otherwise it is reported as a warning. The registry's
measured budgets are held to the same rules: each names a present test file, is
listed once, and is reported or failed when expired the same way. A
quarantined test is held to stricter rules: it names a test that exists, is
listed once, fails the check as soon as it expires, and the registry holds at
most ``QUARANTINE_CAP`` of them.

The check configures the platform (nothing is built) and reads the provider
categories, so it runs in a source checkout of the monorepo only.
"""

from __future__ import annotations

import ast
import datetime
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
from rocke.runtime.comgr import CompilerInfo
from rocke.tests import in_tree_suites
from rocke.tests.ctest_listing import configure_installed_ctest, list_ctest_entries
from rocke.tests.pending_lanes import (
    LANES,
    MEASURED_BUDGETS,
    PENDING,
    QUARANTINE_CAP,
    QUARANTINED,
    MeasuredBudget,
    PendingTest,
    QuarantinedTest,
    measured_budget_kernels,
    measured_scratch_allowance,
)
from rocke.tests.provider_categories import provider_category_labels
from rocke.tests.quarantine import skip_quarantined_items

ROCKE = platform_root().parent
# The tiers pull-request and push CI select.
CI_TIERS = frozenset({"quick", "standard"})
# The provider build installs rocKE's CTest entries into bin/<this>.
PROVIDER_CTEST_SUBDIR = "hip_kernel_provider"
# Where test files live, relative to the rocke directory.
TEST_FILE_ROOTS = ("platform", "library")
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
            paths.append(argument)
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


def tree_test_files() -> set[str]:
    """Every test file of the source tree, relative to the rocke directory."""
    return {
        file.relative_to(ROCKE).as_posix()
        for root in TEST_FILE_ROOTS
        for file in (ROCKE / root).rglob("test_*.py")
        if "__pycache__" not in file.parts
    }


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
        for path in tree_test_files()
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
    nor listed, and the listed files whose lane's kind does not fit them."""
    listed = {entry.path: entry for entry in pending}
    in_limbo = sorted(test_files - in_ci - listed.keys())
    stale = sorted(
        path for path, entry in listed.items() if (path in in_ci) != entry.lane.enrolled
    )
    return in_limbo, stale


def test_audit_finds_files_in_limbo_and_stale_entries():
    waiting = next(lane for lane in LANES if not lane.enrolled)
    lacking = next(lane for lane in LANES if lane.enrolled)

    def entry(path, lane):
        return PendingTest(path, lane, "why", "owner", "2027-01-31")

    test_files = {
        "a/test_in_ci.py",
        "a/test_limbo.py",
        "a/test_new_lane.py",
        "a/test_skips.py",
    }
    in_ci = {"a/test_in_ci.py", "a/test_new_lane.py"}
    pending = [entry("a/test_new_lane.py", waiting), entry("a/test_skips.py", lacking)]
    # test_new_lane waits for a lane but is in CI; test_skips waits for
    # something CI lacks but is in no CI entry.
    assert audit(test_files, in_ci, pending) == (
        ["a/test_limbo.py"],
        ["a/test_new_lane.py", "a/test_skips.py"],
    )


def test_paths_keep_their_root_when_normalized():
    assert _normalized("/src/./tests/../tests/core") == "/src/tests/core"
    assert _normalized("bin/p/./tests/../rocke") == "bin/p/rocke"
    assert _normalized("../tests") == "../tests"


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


def test_a_pending_test_needs_its_recorded_reason():
    lane = LANES[0]
    with pytest.raises(ValueError, match="reason is empty"):
        PendingTest("library/tests/test_x.py", lane, " ", "owner", "2027-01-31")
    with pytest.raises(ValueError, match="expected plain text"):
        PendingTest(
            "library/tests/test_x.py", lane, "see https://x", "owner", "2027-01-31"
        )
    with pytest.raises(ValueError, match="expected an ISO date"):
        PendingTest("library/tests/test_x.py", lane, "why", "owner", "next month")


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


def changed_files() -> set[str] | None:
    """The files, relative to the rocke directory, that this change touches
    against its base (``ROCKE_CHANGE_BASE``, else the merge base with
    origin/develop), working tree included; None when git cannot tell."""
    git = shutil.which("git")
    if git is None:
        return None

    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            [git, *arguments], cwd=ROCKE, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    base = os.environ.get("ROCKE_CHANGE_BASE") or run(
        "merge-base", "HEAD", "origin/develop"
    )
    if not base:
        return None
    names = run("diff", "--name-only", "--relative", base)
    return None if names is None else set(names.splitlines())


def expiry_failures(
    expired: list[str], changed: set[str] | None, *, strict: bool
) -> list[str]:
    """The expired entries that fail the run: all of them under
    ``--strict-expiry``, else only when the change touches the registry or one
    of their files (so an entry expiring does not turn an unrelated change
    red). The rest are reported as warnings."""
    if strict:
        return expired
    if not changed:
        return []
    if PENDING_LANES_MODULE in changed:
        return expired
    return [path for path in expired if path in changed]


def test_expired_entries_fail_only_a_change_that_touches_them():
    expired = ["a/test_x.py", "a/test_y.py"]
    assert expiry_failures(expired, None, strict=False) == []
    assert expiry_failures(expired, {"other.py"}, strict=False) == []
    assert expiry_failures(expired, {"a/test_y.py"}, strict=False) == ["a/test_y.py"]
    assert expiry_failures(expired, {PENDING_LANES_MODULE}, strict=False) == expired
    assert expiry_failures(expired, None, strict=True) == expired


def _fail_or_report_expired(request, registry: str, expired: dict[str, str]) -> None:
    """Fail on the ``expired`` entries (label -> its test file) that
    :func:`expiry_failures` says fail this run; report the rest as a warning."""
    if not expired:
        return
    strict = request.config.getoption("--strict-expiry")
    failing_paths = set(
        expiry_failures(sorted(set(expired.values())), changed_files(), strict=strict)
    )
    failing = sorted(label for label, path in expired.items() if path in failing_paths)
    assert not failing, f"expired {registry} entries; re-decide them: {failing}"
    message = f"{registry} entries past their expiry: {sorted(expired)}"
    print(message)
    warnings.warn(message, stacklevel=2)


def _utc_today() -> datetime.date:
    return datetime.datetime.now(datetime.timezone.utc).date()


def test_every_pending_entry_is_current(request):
    paths = [pending.path for pending in PENDING]
    duplicates = sorted({path for path in paths if paths.count(path) > 1})
    assert not duplicates, f"listed more than once: {duplicates}"
    missing = [
        pending.path for pending in PENDING if not (ROCKE / pending.path).is_file()
    ]
    assert not missing, f"listed but gone: {missing}"
    today = _utc_today()
    _fail_or_report_expired(
        request,
        "pending_lanes",
        {p.path: p.path for p in PENDING if p.expiry_date < today},
    )


def test_every_measured_budget_is_current(request):
    keys = [(budget.path, budget.kernel_name) for budget in MEASURED_BUDGETS]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    assert not duplicates, f"measured budget listed more than once: {duplicates}"
    missing = sorted(
        {b.path for b in MEASURED_BUDGETS if not (ROCKE / b.path).is_file()}
    )
    assert not missing, f"measured budgets for test files that are gone: {missing}"
    today = _utc_today()
    _fail_or_report_expired(
        request,
        "measured budget",
        {
            f"{b.path}::{b.kernel_name}": b.path
            for b in MEASURED_BUDGETS
            if b.expiry_date < today
        },
    )


def test_a_measured_budget_applies_only_on_its_compiler():
    budget = MEASURED_BUDGETS[0]

    def compiler(llvm_version):
        return CompilerInfo(llvm_version, "test", None, None, None)

    measured_on = compiler((budget.llvm_major, 0, 0))
    assert (
        measured_scratch_allowance(budget.path, budget.kernel_name, measured_on)
        == budget.scratch_bytes
    )
    other = compiler((budget.llvm_major - 1, 0, 0))
    assert measured_scratch_allowance(budget.path, budget.kernel_name, other) == 0
    assert measured_scratch_allowance(budget.path, budget.kernel_name, None) == 0
    unqueried = compiler(None)
    assert measured_scratch_allowance(budget.path, budget.kernel_name, unqueried) == 0
    assert measured_scratch_allowance(budget.path, "rocke_other", measured_on) == 0
    other_file = "library/tests/test_x.py"
    assert measured_scratch_allowance(other_file, budget.kernel_name, measured_on) == 0
    assert budget.kernel_name in measured_budget_kernels(budget.path)
    assert measured_budget_kernels(other_file) == frozenset()


def test_a_measured_budget_needs_a_positive_size_and_its_recorded_reason():
    def budget(**overrides):
        fields = {
            "path": "library/tests/test_x.py",
            "kernel_name": "rocke_x",
            "scratch_bytes": 8,
            "llvm_major": 24,
            "reason": "why",
            "owner": "owner",
            "expires": "2027-01-31",
        }
        return MeasuredBudget(**{**fields, **overrides})

    with pytest.raises(ValueError, match="expected a test file path"):
        budget(path="/abs/test_x.py")
    with pytest.raises(ValueError, match="kernel_name is empty"):
        budget(kernel_name=" ")
    with pytest.raises(ValueError, match="scratch_bytes=0, expected a positive"):
        budget(scratch_bytes=0)
    with pytest.raises(ValueError, match="llvm_major=True, expected a positive"):
        budget(llvm_major=True)
    with pytest.raises(ValueError, match="owner is empty"):
        budget(owner="")
    with pytest.raises(ValueError, match="expected plain text"):
        budget(reason="see https://x")
    with pytest.raises(ValueError, match="expected an ISO date"):
        budget(expires="next month")


def _quarantined(test_id: str, **overrides: str) -> QuarantinedTest:
    fields = {
        "issue": "AICK-1234",
        "reason": "why",
        "owner": "owner",
        "expires": "2027-01-31",
    }
    return QuarantinedTest(test_id, **{**fields, **overrides})


def test_a_quarantined_test_needs_an_item_an_issue_and_its_recorded_reason():
    with pytest.raises(ValueError, match="expected <test file>::<pytest node name>"):
        _quarantined("library/tests/test_x.py")
    with pytest.raises(ValueError, match="expected a test under one of"):
        _quarantined("platform/python/rocke/tests/test_x.py::test_y")
    with pytest.raises(ValueError, match="expected a Jira id"):
        _quarantined("library/tests/test_x.py::test_y", issue="none filed yet")
    with pytest.raises(ValueError, match="expected a Jira id"):
        _quarantined(
            "library/tests/test_x.py::test_y",
            issue="https://x/issues/1",
        )
    with pytest.raises(ValueError, match="expected plain text"):
        _quarantined("library/tests/test_x.py::test_y", reason="see https://x")
    with pytest.raises(ValueError, match="expected an ISO date"):
        _quarantined("library/tests/test_x.py::test_y", expires="next month")
    github = _quarantined(
        "library/tests/test_x.py::test_y", issue="ROCm/rocm-libraries#1"
    )
    assert github.issue == "ROCm/rocm-libraries#1"


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


def test_only_a_quarantine_tree_skips_quarantined_tests():
    with pytest.raises(ValueError, match="expected one of"):
        skip_quarantined_items([], ROCKE / "platform/python", "platform/python")


def _defines_test(test_file: Path, node_name: str) -> bool:
    """Whether ``test_file`` defines the test function ``node_name`` names."""
    function = node_name.split("::")[-1].partition("[")[0]
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function
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
    today = datetime.date(2027, 1, 1)
    current = _quarantined("library/tests/test_x.py::test_y")
    assert quarantine_breaches([current], today) == []
    expired = _quarantined("library/tests/test_x.py::test_z", expires="2026-12-31")
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


def test_every_quarantined_test_is_current():
    breaches = quarantine_breaches(QUARANTINED, _utc_today())
    assert not breaches, breaches
    gone = [
        entry.test_id
        for entry in QUARANTINED
        if not (ROCKE / entry.path).is_file()
        or not _defines_test(ROCKE / entry.path, entry.node_name)
    ]
    assert not gone, f"quarantined but gone: {gone}"


@pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("ctest") is None,
    reason="CMake and CTest required",
)
def test_every_test_file_runs_in_ci_or_has_a_recorded_reason(tmp_path):
    in_limbo, stale = audit(tree_test_files(), ci_selected_files(tmp_path), PENDING)
    assert not in_limbo, (
        f"{len(in_limbo)} test files run in no CI-run entry and are not in "
        f"rocke.tests.pending_lanes: {in_limbo}"
    )
    assert not stale, (
        "listed under a lane whose kind no longer fits (a file waiting for a new "
        f"lane is now in CI, or one waiting for something CI lacks left CI): {stale}"
    )
