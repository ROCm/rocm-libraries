# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: the test files a CTest entry's pytest command selects.

A registered CTest entry runs pytest over paths. A build-tree entry names
source paths; an installed entry names paths of the install layout, which the
configured install rules (each ``file(INSTALL ...)`` of a generated
``cmake_install.cmake``) map back to the source. ``selected_test_files`` turns
an entry's command arguments into the source test files it selects: the files
and directories it names, less the ones it ``--ignore``s and the modules its
``-m`` expression deselects through their ``pytestmark`` alone. Marks on
single tests and ``-k`` filters are not read.

    arguments = arguments_after_interpreter(entry["command"])
    files = selected_test_files(arguments, InstallLayout(rules), base)
"""

from __future__ import annotations

import ast
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from _pytest.mark.expression import Expression

__all__ = [
    "InstallLayout",
    "InstallRule",
    "Layout",
    "PytestSelection",
    "SourceLayout",
    "arguments_after_interpreter",
    "deselects_module",
    "install_rules",
    "module_marks",
    "normalized_path",
    "pytest_selection",
    "selected_test_files",
]


def normalized_path(path: str) -> str:
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


def arguments_after_interpreter(command: list[str]) -> list[str]:
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
    ignored = [normalized_path(base + path) for path in selection.ignored]
    selected = set()
    for path in (normalized_path(base + path) for path in selection.paths):
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
