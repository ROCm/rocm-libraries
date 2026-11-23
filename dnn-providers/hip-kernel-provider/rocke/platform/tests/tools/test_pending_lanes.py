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
listed entry must be current: its file present and its lane's kind still
true of the file (a file waiting for a new lane is in no CI-run entry; a
file waiting for something CI lacks is in one), and a file waiting for torch
imports it. A file is listed once per lane, and under several lanes only when
each waits for something CI lacks. An expired entry fails only a
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
from rocke.tests import lanes as rocke_lanes
from rocke.tests.ctest_listing import configure_installed_ctest, list_ctest_entries
from rocke.tests.lanes import (
    COMPREHENSIVE_CASES,
    GFX950_RUNNER,
    GFX1250_RUNNER,
    LANES,
    NIGHTLY_SOURCE,
    TORCH_IN_CI,
    Lane,
    skip_until_reason,
)
from rocke.tests.pending_lanes import (
    MEASURED_BUDGETS,
    NO_ISSUE_FILED,
    PENDING,
    QUARANTINE_CAP,
    QUARANTINED,
    MeasuredBudget,
    PendingTest,
    QuarantinedTest,
    measured_budget_kernels,
    measured_scratch_allowance,
    tree_test_files,
    waiting_entry,
)
from rocke.tests.provider_categories import provider_category_labels
from rocke.tests.quarantine import skip_quarantined_items

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
    nor listed, and the listed files a lane's kind does not fit."""
    pending = list(pending)
    in_limbo = sorted(test_files - in_ci - {entry.path for entry in pending})
    stale = sorted(
        {
            entry.path
            for entry in pending
            if (entry.path in in_ci) != entry.lane.enrolled
        }
    )
    return in_limbo, stale


def listing_breaches(pending: Iterable[PendingTest]) -> list[str]:
    """The files or tests listed twice under one lane, or files under several
    lanes when one of them is a lane that does not exist yet (its files are in
    no CI entry)."""
    pending = list(pending)
    keys = [(entry.path, entry.node, entry.lane.name) for entry in pending]
    breaches = [
        f"listed twice under one lane: {path}" + (f"::{node}" if node else "")
        for path, node, _ in sorted({key for key in keys if keys.count(key) > 1})
    ]
    lanes_of: dict[str, dict[str, Lane]] = {}
    for entry in pending:
        lanes_of.setdefault(entry.path, {})[entry.lane.name] = entry.lane
    for path, lanes in sorted(lanes_of.items()):
        if len(lanes) > 1 and not all(lane.enrolled for lane in lanes.values()):
            breaches.append(
                f"listed under several lanes, one of which no CI entry runs yet: "
                f"{path} ({', '.join(lanes)})"
            )
    return breaches


def imports_torch(test_file: Path) -> bool:
    """Whether a test module imports torch anywhere, a function body included:
    an ``import torch[.x]`` or ``from torch[.x] import`` statement, or a call of
    ``importorskip``, ``import_or_skip_until``, ``import_module`` or
    ``find_spec`` with ``"torch"`` (a module that checks for torch, for a child
    process that imports it)."""
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = [node.module or ""]
        elif isinstance(node, ast.Call):
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else None
            if isinstance(function, ast.Name):
                name = function.id
            first = node.args[0] if node.args else None
            if name not in (
                "importorskip",
                "import_or_skip_until",
                "import_module",
                "find_spec",
            ) or not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                continue
            modules = [first.value]
        else:
            continue
        if any(module.partition(".")[0] == "torch" for module in modules):
            return True
    return False


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


def test_a_file_may_wait_for_several_things_ci_lacks_but_one_new_lane():
    def entry(lane):
        return PendingTest("a/test_x.py", lane, "why", "owner", "2027-01-31")

    gfx1250_runner = next(
        lane for lane in LANES if lane.enrolled and lane is not GFX950_RUNNER
    )
    assert listing_breaches([entry(GFX950_RUNNER), entry(gfx1250_runner)]) == []
    assert listing_breaches([entry(GFX950_RUNNER), entry(GFX950_RUNNER)]) == [
        "listed twice under one lane: a/test_x.py"
    ]
    assert listing_breaches([entry(GFX950_RUNNER), entry(NIGHTLY_SOURCE)]) == [
        (
            "listed under several lanes, one of which no CI entry runs yet: "
            "a/test_x.py (gfx950-runner, nightly-source)"
        )
    ]
    # Each of a file's entries must fit it: in CI for both enrolled lanes.
    entries = [entry(GFX950_RUNNER), entry(gfx1250_runner)]
    assert audit({"a/test_x.py"}, {"a/test_x.py"}, entries) == ([], [])
    assert audit({"a/test_x.py"}, set(), entries) == ([], ["a/test_x.py"])


def test_a_torch_import_is_found_wherever_the_module_makes_it(tmp_path):
    sources = {
        "test_module_level.py": "import torch\n",
        "test_submodule.py": "import numpy as np\nfrom torch.nn import functional\n",
        "test_in_a_function.py": "def test_x():\n    import torch.cuda\n",
        "test_importorskip.py": (
            "import pytest\n"
            "def test_x():\n"
            "    torch = pytest.importorskip('torch', reason='graphs')\n"
        ),
        "test_import_module.py": (
            "import importlib\ntorch = importlib.import_module('torch')\n"
        ),
        "test_find_spec.py": (
            "import importlib.util\nHAS_TORCH = importlib.util.find_spec('torch')\n"
        ),
        "test_lane_import.py": (
            "from rocke.tests.lanes import TORCH_IN_CI, import_or_skip_until\n"
            "torch = import_or_skip_until('torch', TORCH_IN_CI)\n"
        ),
    }
    for name, source in sources.items():
        (tmp_path / name).write_text(source)
        assert imports_torch(tmp_path / name), name
    torch_free = {
        "test_numpy.py": "import numpy as np\n",
        "test_lookalike.py": "import torchvision\nfrom torchaudio import io\n",
        "test_relative.py": "from .torch import helper\n",
        "test_mention.py": "# torch is not needed: a numpy reference\nNAME = 'torch'\n",
        "test_other_skip.py": "import pytest\nyaml = pytest.importorskip('yaml')\n",
    }
    for name, source in torch_free.items():
        (tmp_path / name).write_text(source)
        assert not imports_torch(tmp_path / name), name


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
    def entry(lane):
        return PendingTest("a/test_x.py", lane, "why", "owner", "2027-01-31")

    waiting = entry(GFX950_RUNNER)
    declared = skip_until_reason(GFX950_RUNNER, reason="needs a gfx950 GPU")
    assert waiting.covers("a/test_x.py", "test_y", declared)
    both = skip_until_reason(GFX950_RUNNER, GFX1250_RUNNER, reason="needs either")
    assert waiting.covers("a/test_x.py", "test_y", both)
    # A skip that only mentions the lane's arch is not a declaration: a defect
    # whose reason names gfx950 is not booked as waiting for its runner.
    assert not waiting.covers("a/test_x.py", "test_y", "needs a gfx950 GPU")
    assert not waiting.covers("a/test_x.py", "test_y", "gfx950 kernel miscompares")
    other = skip_until_reason(GFX1250_RUNNER, reason="needs a gfx1250 GPU")
    assert not waiting.covers("a/test_x.py", "test_y", other)
    assert not waiting.covers("a/test_x.py", "test_y", None)
    assert not waiting.covers("a/test_z.py", "test_y", declared)
    # A lane no CI entry runs yet: every case of its files waits for it.
    new_lane = entry(NIGHTLY_SOURCE)
    assert new_lane.covers("a/test_x.py", "test_y", "any reason")
    assert new_lane.covers("a/test_x.py", "test_y", None)


def test_an_entry_for_one_test_covers_its_unselected_or_declared_cases_only():
    def entry(lane, node=""):
        return PendingTest("a/test_x.py", lane, "why", "owner", "2027-01-31", node)

    with pytest.raises(ValueError, match="expected a lane CI lacks for one test"):
        entry(NIGHTLY_SOURCE, "test_y")
    with pytest.raises(ValueError, match="expected one of rocke.tests.lanes.LANES"):
        PendingTest(
            "a/test_x.py",
            Lane("unlisted", "a lane", enrolled=True),
            "why",
            "owner",
            "2027-01-31",
        )
    test = entry(COMPREHENSIVE_CASES, "test_y")
    left_out = "deselected by -k 'not test_y'"
    assert test.covers("a/test_x.py", "test_y[case]", left_out, deselected=True)
    assert test.covers("a/test_x.py", "test_y", None)
    assert not test.covers("a/test_x.py", "test_yz", None)
    assert not test.covers("a/test_z.py", "test_y", None)
    group = entry(COMPREHENSIVE_CASES, "TestGroup")
    assert group.covers("a/test_x.py", "TestGroup::test_y[case]", None)
    assert not group.covers("a/test_x.py", "TestGroupTwo::test_y", None)
    # A test that ran to a skip is covered only when the skip declares the
    # entry's lane: a defect skip of a test listed under a runner is not.
    runner = entry(GFX950_RUNNER, "test_y")
    declared = skip_until_reason(GFX950_RUNNER, reason="needs a gfx950 GPU")
    assert runner.covers("a/test_x.py", "test_y", declared)
    assert not runner.covers("a/test_x.py", "test_y", "invalid spec on gfx942")
    assert not runner.covers("a/test_x.py", "test_y", left_out)


def test_the_registry_accounts_for_a_case_by_its_lane_quarantine_or_runner():
    file_entry = next(
        entry
        for entry in PENDING
        if entry.lane is GFX950_RUNNER
        and not entry.node
        and sum(other.path == entry.path for other in PENDING) == 1
        and not any(q.path == entry.path for q in QUARANTINED)
    )
    path = file_entry.path
    declared = skip_until_reason(GFX950_RUNNER, reason="needs a gfx950 GPU")
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
    today = _utc_today()
    _fail_or_report_expired(
        request,
        "pending_lanes",
        {f"{p.path} ({p.lane.name})": p.path for p in PENDING if p.expiry_date < today},
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


def _quarantined(test_id: str, **overrides: object) -> QuarantinedTest:
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
        _quarantined("library/tests/test_x.py::test_y", issue="filed soon")
    with pytest.raises(ValueError, match="expected a Jira id"):
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
    for llvm_commits in (("424729dba65",), ("424729DBA653",), "424729dba653"):
        with pytest.raises(ValueError, match="expected a tuple of llvm-project"):
            _quarantined("library/tests/test_x.py::test_y", llvm_commits=llvm_commits)
    github = _quarantined(
        "library/tests/test_x.py::test_y", issue="ROCm/rocm-libraries#1"
    )
    assert github.issue == "ROCm/rocm-libraries#1"
    unfiled = _quarantined("library/tests/test_x.py::test_y", issue=NO_ISSUE_FILED)
    assert unfiled.issue == NO_ISSUE_FILED


def test_a_quarantine_that_names_arches_holds_its_test_out_on_those_only():
    everywhere = _quarantined("library/tests/test_x.py::test_y")
    assert everywhere.holds_out_on("gfx942")
    assert everywhere.holds_out_on(None)
    on_gfx950 = _quarantined("library/tests/test_x.py::test_y", arches=("gfx950",))
    assert on_gfx950.holds_out_on("gfx950")
    assert not on_gfx950.holds_out_on("gfx942")
    assert not on_gfx950.holds_out_on(None)


def test_a_quarantine_that_names_compilers_holds_its_test_out_with_those_only():
    ci_build = "424729dba65359a5e521ab2e729af131c8a5b0d4"
    other_build = "d6f6cb691863e9258034ab48c1dfce21689d5606"
    any_compiler = _quarantined("library/tests/test_x.py::test_y")
    assert any_compiler.holds_out_with(ci_build)
    assert any_compiler.holds_out_with(None)
    with_ci_build = _quarantined(
        "library/tests/test_x.py::test_y", llvm_commits=("424729dba653",)
    )
    assert with_ci_build.holds_out_with(ci_build)
    assert not with_ci_build.holds_out_with(other_build)
    assert not with_ci_build.holds_out_with(None)


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


# The files that may name a runner lane or write a lane declaration by hand:
# the lanes module, the registry, and the tests of both.
_LANE_HOMES = frozenset(
    {
        "platform/python/rocke/tests/lanes.py",
        "platform/python/rocke/tests/pending_lanes.py",
    }
)
_LANE_TEST_DIRS = ("platform/tests/support/", "platform/tests/tools/")


def _python_sources() -> Iterable[tuple[str, ast.Module]]:
    """Each Python file of the test file roots, relative to the rocke
    directory, with its syntax tree."""
    for root in ("platform", "library"):
        for file in (ROCKE / root).rglob("*.py"):
            parts = file.relative_to(ROCKE).parts
            if "__pycache__" in parts or any(p.startswith("build") for p in parts):
                continue
            yield file.relative_to(ROCKE).as_posix(), ast.parse(
                file.read_text(encoding="utf-8")
            )


def _declares_a_lane(text: str) -> bool:
    """Whether ``text`` reads as a lane declaration, a known lane or not."""
    try:
        return bool(rocke_lanes.lanes_waited_for(text))
    except ValueError:
        return True


def test_tests_state_the_arches_they_need_not_the_runner_lane():
    """A test names the arches it needs (``skip_needing_arch`` or
    ``arch_skip_reason``); which runner lane that waits for, if any, is a fact
    of ``rocke.tests.lanes`` alone, so a runner lane is named only in the lane
    homes, and a hand-written ``waits for ...`` reason nowhere else."""
    runner_lanes = {
        name
        for name, value in vars(rocke_lanes).items()
        if isinstance(value, Lane) and value.arches
    }
    named, hand_written = [], []
    for path, tree in _python_sources():
        if path in _LANE_HOMES or path.startswith(_LANE_TEST_DIRS):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in runner_lanes:
                named.append(f"{path}:{node.lineno} {node.id}")
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and _declares_a_lane(node.value)
            ):
                hand_written.append(f"{path}:{node.lineno}")
    assert not named, (
        "name the arches the case needs (skip_needing_arch / arch_skip_reason) "
        f"instead of a runner lane: {named}"
    )
    assert not hand_written, (
        "declare a lane with rocke.tests.lanes.skip_until / skip_until_reason, "
        f"not a hand-written 'waits for' reason: {hand_written}"
    )


def test_every_file_waiting_for_torch_imports_it():
    assert TORCH_IN_CI.waits_for_torch
    torchless = sorted(
        f"{pending.path} ({pending.lane.name})"
        for pending in PENDING
        if pending.lane.waits_for_torch and not imports_torch(ROCKE / pending.path)
    )
    assert not torchless, (
        "listed under a lane that waits for torch, but they do not import it "
        f"(move them to the lane they wait for, or drop them): {torchless}"
    )


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
