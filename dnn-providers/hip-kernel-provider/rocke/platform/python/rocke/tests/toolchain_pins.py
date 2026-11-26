# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Find toolchain pins in rocKE's test and library code: the one checker behind
the toolchain-pins gate.

Tests use the compiler rocKE's discovery loads; the rule's one home is
``dsl_docs/reference/env_flags.md``, "Compiler detection and provenance". Code
that pins one LLVM flavor or one ROCm release, decides on the ROCm release of
the loaded comgr, or reaches into the runtime's or the lowering's private names
ties itself to one toolchain. A tree-wide gate
(``platform/tests/tools/test_toolchain_pins.py``) holds the remaining pins on an
allowlist that only shrinks, each line naming the issue that removes it.

A pin has a kind (:class:`ToolchainPinKind`). The checker reads the AST, so text
in strings, comments and docstrings is not code::

    GATE_FLAVOR = "llvm22"                    # FLAVOR_PIN: a constant
    if current_flavor() == "llvm22": ...      # FLAVOR_PIN: what the compiler reports
    for flavor in LLVM_FLAVORS:
        if flavor == "llvm20": ...            # nothing: one of every flavor, in turn
    GATE_COMGR_ROCM = "7.2"                   # ROCM_PIN: a constant named for ROCm
    if comgr_vintage < (7, 2): ...            # ROCM_PIN: compared with a release
    resolved_lib_rocm_version()               # ROCM_VERSION
    from rocke.core.lower_llvm import _detect_llvm_flavor  # PRIVATE_IMPORT

Test code is every file under a ``tests`` directory, and every ``test_*.py`` and
``conftest.py``; library code is every file under ``library/``. Both are held to
the first three kinds. PRIVATE_IMPORT applies to library code only: the
platform's own tests may test its private helpers.

    python -m rocke.tests.toolchain_pins      # each file's kinds, allowlist format
"""

from __future__ import annotations

import argparse
import ast
import enum
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..assets import platform_root
from . import source_allowlist

__all__ = [
    "ToolchainPin",
    "ToolchainPinKind",
    "allowlist_breaches",
    "checked_kinds",
    "read_allowlist",
    "toolchain_pins_in_python",
    "tree_files",
    "tree_toolchain_pins",
]


class ToolchainPinKind(enum.Enum):
    """What a pin ties to one toolchain; the gate allowlists kinds per file."""

    FLAVOR_PIN = "flavor_pin"
    """A module- or class-level constant set to one LLVM flavor (``"llvm22"``,
    or a collection holding only it), or a comparison of one, or of a collection
    of them, with what the loaded compiler or the environment reports: a call
    whose name says ``flavor`` and that is given no flavor, or a read of an
    environment variable whose name says ``FLAVOR``."""
    ROCM_PIN = "rocm_pin"
    """A module- or class-level constant set to one ROCm release (a string naming
    ``rocm<release>``, or a version set to a name that says ``rocm``), or a
    comparison with one."""
    ROCM_VERSION = "rocm_version"
    """A call of ``resolved_lib_rocm_version``: a decision on the ROCm release
    of the loaded comgr rather than on what the compiler reports."""
    PRIVATE_IMPORT = "private_import"
    """An underscore name of ``rocke.runtime`` (or a module under it) or of
    ``rocke.core.lower_llvm``, imported or read as an attribute (library code
    only)."""


@dataclass(frozen=True)
class ToolchainPin:
    """One pin: its 1-based ``line``, ``kind`` and the code it found."""

    line: int
    kind: ToolchainPinKind
    detail: str


_PINS_IN_TESTS = frozenset(
    {
        ToolchainPinKind.FLAVOR_PIN,
        ToolchainPinKind.ROCM_PIN,
        ToolchainPinKind.ROCM_VERSION,
    }
)
_PINS_IN_LIBRARY = _PINS_IN_TESTS | {ToolchainPinKind.PRIVATE_IMPORT}
_LIBRARY_ROOT = "library"
_FLAVOR_LITERAL = re.compile(r"llvm\d+")
_ROCM_RELEASE_TEXT = re.compile(r"rocm[-_]?\d+\.\d+", re.IGNORECASE)
_VERSION_TEXT = re.compile(r"\d+\.\d+(?:\.\d+)?")
_ROCM_WORD = re.compile(r"rocm", re.IGNORECASE)
_FLAVOR_WORD = re.compile(r"flavor", re.IGNORECASE)
_ROCM_VERSION_CALL = "resolved_lib_rocm_version"
_GUARDED_MODULES = ("rocke.runtime", "rocke.core.lower_llvm")
# Text every pin of a kind holds; a source without it is not parsed.
_MARKERS = {
    ToolchainPinKind.FLAVOR_PIN: "llvm",
    ToolchainPinKind.ROCM_PIN: "rocm",
    ToolchainPinKind.ROCM_VERSION: "rocm",
    ToolchainPinKind.PRIVATE_IMPORT: "rocke",
}
_ENVIRONMENT_READS = frozenset({"os.environ.get", "os.getenv", "environ.get", "getenv"})


def checked_kinds(path: str) -> frozenset[ToolchainPinKind]:
    """The kinds the gate checks in ``path`` (relative to the rocke directory,
    ``/`` separators): none outside test and library code."""
    parts = PurePosixPath(path)
    if parts.parts and parts.parts[0] == _LIBRARY_ROOT:
        return _PINS_IN_LIBRARY
    if "tests" in parts.parts[:-1] or parts.name.startswith("test_"):
        return _PINS_IN_TESTS
    if parts.name == "conftest.py":
        return _PINS_IN_TESTS
    return frozenset()


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_flavor_literal(node: ast.AST) -> bool:
    """One LLVM flavor (``"llvm22"``), or a collection of them."""
    text = _string(node)
    if text is not None:
        return bool(_FLAVOR_LITERAL.fullmatch(text))
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)) and node.elts:
        return all(_is_flavor_literal(element) for element in node.elts)
    return False


def _is_one_flavor(node: ast.AST) -> bool:
    """One LLVM flavor, alone or as the only member of a collection: a constant
    listing several is a copy of the flavor list, not a pin."""
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return len(node.elts) == 1 and _is_flavor_literal(node.elts[0])
    return _is_flavor_literal(node)


def _names_rocm_release(node: ast.AST) -> bool:
    text = _string(node)
    return text is not None and bool(_ROCM_RELEASE_TEXT.search(text))


def _is_version_literal(node: ast.AST) -> bool:
    text = _string(node)
    if text is not None:
        return bool(_VERSION_TEXT.fullmatch(text))
    return (
        isinstance(node, ast.Tuple)
        and 2 <= len(node.elts) <= 3
        and all(
            isinstance(element, ast.Constant) and type(element.value) is int
            for element in node.elts
        )
    )


def _reads_flavor(node: ast.AST) -> bool:
    """Whether ``node`` is what the loaded compiler or the environment says the
    flavor is: a call whose name says flavor, or an environment read of a
    variable whose name says FLAVOR."""
    if isinstance(node, ast.Call):
        function = _dotted(node.func)
        if function in _ENVIRONMENT_READS:
            key = _string(node.args[0]) if node.args else None
            return key is not None and bool(_FLAVOR_WORD.search(key))
        given_a_flavor = any(map(_is_flavor_literal, node.args)) or any(
            _is_flavor_literal(keyword.value) for keyword in node.keywords
        )
        return not given_a_flavor and bool(
            _FLAVOR_WORD.search(function.rsplit(".", 1)[-1])
        )
    if isinstance(node, ast.Subscript) and _dotted(node.value).endswith("environ"):
        key = _string(node.slice)
        return key is not None and bool(_FLAVOR_WORD.search(key))
    return False


def _constant_targets(
    tree: ast.Module, nodes: Sequence[ast.AST]
) -> list[tuple[str, ast.AST, int]]:
    """``(name, value, line)`` of each assignment to a name in a module or class
    body (``nodes`` are every node of ``tree``): the module's and its classes'
    constants."""
    bodies: list[list[ast.stmt]] = [tree.body]
    bodies.extend(node.body for node in nodes if isinstance(node, ast.ClassDef))
    found = []
    for body in bodies:
        for statement in body:
            if isinstance(statement, ast.Assign):
                targets, value = statement.targets, statement.value
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                targets, value = [statement.target], statement.value
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    found.append((target.id, value, statement.lineno))
    return found


def _constant_pins(tree: ast.Module, nodes: Sequence[ast.AST]) -> list[ToolchainPin]:
    pins = []
    for name, value, line in _constant_targets(tree, nodes):
        detail = f"{name} = {ast.unparse(value)}"
        if _is_one_flavor(value):
            pins.append(ToolchainPin(line, ToolchainPinKind.FLAVOR_PIN, detail))
        elif _names_rocm_release(value) or (
            _ROCM_WORD.search(name) and _is_version_literal(value)
        ):
            pins.append(ToolchainPin(line, ToolchainPinKind.ROCM_PIN, detail))
    return pins


def _comparison_pins(node: ast.Compare) -> list[ToolchainPin]:
    operands = [node.left, *node.comparators]
    detail = ast.unparse(node)
    pins = []
    if any(map(_is_flavor_literal, operands)) and any(map(_reads_flavor, operands)):
        pins.append(ToolchainPin(node.lineno, ToolchainPinKind.FLAVOR_PIN, detail))
    names_rocm = any(_ROCM_WORD.search(ast.unparse(operand)) for operand in operands)
    if any(map(_names_rocm_release, operands)) or (
        names_rocm and any(map(_is_version_literal, operands))
    ):
        pins.append(ToolchainPin(node.lineno, ToolchainPinKind.ROCM_PIN, detail))
    return pins


def _is_guarded(module: str) -> bool:
    return any(
        module == guarded or module.startswith(f"{guarded}.")
        for guarded in _GUARDED_MODULES
    )


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _module_bindings(nodes: Sequence[ast.AST]) -> dict[str, str]:
    """Each name a module (every node of it in ``nodes``) binds to a rocke
    module, with the module's dotted name (``{"comgr": "rocke.runtime.comgr"}``)."""
    bindings = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name.split(".")[0] != "rocke":
                    continue
                if imported.asname:
                    bindings[imported.asname] = imported.name
                else:
                    bindings["rocke"] = "rocke"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] != "rocke":
                continue
            for imported in node.names:
                bound = imported.asname or imported.name
                bindings[bound] = f"{node.module}.{imported.name}"
    return bindings


def _private_imports(nodes: Sequence[ast.AST]) -> list[ToolchainPin]:
    bindings = _module_bindings(nodes)
    pins = []
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for imported in node.names:
                full_name = f"{node.module}.{imported.name}"
                if _is_guarded(full_name) and _is_private(imported.name):
                    pins.append(
                        ToolchainPin(
                            node.lineno,
                            ToolchainPinKind.PRIVATE_IMPORT,
                            f"from {node.module} import {imported.name}",
                        )
                    )
        elif isinstance(node, ast.Import):
            for imported in node.names:
                if _is_guarded(imported.name) and any(
                    map(_is_private, imported.name.split("."))
                ):
                    pins.append(
                        ToolchainPin(
                            node.lineno,
                            ToolchainPinKind.PRIVATE_IMPORT,
                            f"import {imported.name}",
                        )
                    )
        elif isinstance(node, ast.Attribute) and _is_private(node.attr):
            owner = _dotted(node.value)
            root, _, rest = owner.partition(".")
            if root not in bindings:
                continue
            expanded = ".".join(filter(None, (bindings[root], rest)))
            if _is_guarded(expanded):
                pins.append(
                    ToolchainPin(
                        node.lineno, ToolchainPinKind.PRIVATE_IMPORT, _dotted(node)
                    )
                )
    return pins


def toolchain_pins_in_python(
    source: str,
    filename: str,
    kinds: frozenset[ToolchainPinKind] = frozenset(ToolchainPinKind),
) -> list[ToolchainPin]:
    """Every pin of a kind in ``kinds`` in Python ``source``, in line order.

    ``filename`` names the source in a ``SyntaxError``; a file that does not
    parse is an error, not a clean file.
    """
    lowered = source.lower()
    if not any(_MARKERS[kind] in lowered for kind in kinds):
        return []
    tree = ast.parse(source, filename=filename)
    nodes = list(ast.walk(tree))
    pins = _constant_pins(tree, nodes)
    for node in nodes:
        if isinstance(node, ast.Compare):
            pins.extend(_comparison_pins(node))
        elif (
            isinstance(node, ast.Call)
            and _dotted(node.func).rsplit(".", 1)[-1] == _ROCM_VERSION_CALL
        ):
            pins.append(
                ToolchainPin(
                    node.lineno, ToolchainPinKind.ROCM_VERSION, ast.unparse(node)
                )
            )
    if ToolchainPinKind.PRIVATE_IMPORT in kinds:
        pins.extend(_private_imports(nodes))
    return sorted(
        (pin for pin in pins if pin.kind in kinds),
        key=lambda pin: (pin.line, pin.kind.value, pin.detail),
    )


def tree_files(root: Path) -> list[str]:
    """The Python files under ``root`` that the gate reads (test and library
    code that git tracks or would track), relative POSIX paths, sorted."""
    return source_allowlist.tracked_files(
        root, lambda path: path.endswith(".py") and bool(checked_kinds(path))
    )


def tree_toolchain_pins(root: Path) -> dict[str, list[ToolchainPin]]:
    """The pins of every file :func:`tree_files` lists under ``root``, keyed by
    relative path; files with none are left out."""
    found = {}
    for path in tree_files(root):
        source = (root / path).read_text(encoding="utf-8")
        if pins := toolchain_pins_in_python(source, path, checked_kinds(path)):
            found[path] = pins
    return found


def read_allowlist(path: Path) -> dict[str, source_allowlist.AllowlistEntry]:
    """The allowlist at ``path``: one ``file: kind kind  # ISSUE`` line per file,
    the issue the one that removes the line
    (:func:`rocke.tests.source_allowlist.read_allowlist`)."""
    return source_allowlist.read_allowlist(path, ToolchainPinKind, require_issue=True)


def allowlist_breaches(
    found: Mapping[str, Sequence[ToolchainPin]],
    allowed: Mapping[str, source_allowlist.AllowlistEntry],
) -> list[str]:
    """What breaks the allowlist, one line per file, sorted by file
    (:func:`rocke.tests.source_allowlist.allowlist_breaches`)."""
    return source_allowlist.allowlist_breaches(found, allowed, subject="toolchain pin")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rocke.tests.toolchain_pins",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="the tree to read (default: the rocke source root)",
    )
    arguments = parser.parse_args(argv)
    root = arguments.root or platform_root().parent
    for path, pins in tree_toolchain_pins(root).items():
        print(
            source_allowlist.format_allowlist_entry(
                path, source_allowlist.file_kinds(pins)
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
