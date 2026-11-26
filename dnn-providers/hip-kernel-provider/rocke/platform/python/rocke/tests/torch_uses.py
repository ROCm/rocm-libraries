# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Find PyTorch uses in rocKE sources: the one checker behind the no-torch gate.

rocKE does not use torch. Existing uses are being removed, and a tree-wide gate
(``platform/tests/tools/test_no_torch.py``) holds them on an allowlist that only
shrinks. This module finds the uses; the gate compares them with the allowlist.
Other suites that must stay torch-free call :func:`torch_uses_in_python` on
their own files instead of carrying a checker of their own.

A use has a kind (:class:`TorchUseKind`). In Python, the checker reads the AST,
so a use is found at any nesting (a function body, a ``try`` block) and text in
strings, comments and docstrings is not code::

    import torch                         # IMPORT, TORCH_OBJECT where torch.x is used
    importlib.util.find_spec("torch")    # IMPORT
    sys.modules.get("torch")             # IMPORT: takes torch another module imported
    "torch" in sys.modules               # nothing: asks, does not take the module
    pytest.importorskip("torch")         # IMPORTORSKIP
    from rocke.runtime import torch_interop  # TORCH_INTEROP
    torch.cuda.synchronize()             # TORCH_API
    "torch.float16"                      # nothing: a string

In requirements files, ``pyproject.toml`` and CMake files, every line naming
torch as a word is a :attr:`TorchUseKind.CONFIG` use, comments included.

Identifiers that only carry torch in their name (``run_unified_attention_torch``,
``TorchBinding``) are not uses: :func:`torch_named_identifiers` lists them for the
renames that remove them.

    python -m rocke.tests.torch_uses            # each file's kinds, allowlist format
    python -m rocke.tests.torch_uses --names    # the torch-named identifiers
"""

from __future__ import annotations

import argparse
import ast
import enum
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..assets import platform_root
from . import source_allowlist
from .source_allowlist import AllowlistEntry
from .source_ast import dotted_name, string_constant
from .source_tree import source_files

__all__ = [
    "TorchUse",
    "TorchUseKind",
    "allowlist_breaches",
    "is_config_file",
    "read_allowlist",
    "torch_named_identifiers",
    "torch_uses_in_config",
    "torch_uses_in_file",
    "torch_uses_in_python",
    "tree_files",
    "tree_torch_uses",
]


class TorchUseKind(enum.Enum):
    """What a torch use does; the gate allowlists kinds per file."""

    IMPORT = "import"
    """An ``import`` of torch or a torch submodule, a call of ``__import__``,
    ``import_module`` or ``find_spec`` with a torch module name, or a read of
    one from ``sys.modules`` (``sys.modules.get("torch")``,
    ``sys.modules["torch"]``)."""
    IMPORTORSKIP = "importorskip"
    """``importorskip`` or ``import_or_skip_until`` with a torch module name."""
    TORCH_INTEROP = "torch_interop"
    """An import or attribute of rocKE's ``torch_interop`` module."""
    TORCH_API = "torch_api"
    """``torch.ops``, ``torch.fx``, ``torch.compile`` or ``torch.cuda``, by
    attribute or import."""
    TORCH_OBJECT = "torch_object"
    """Any other use of a torch module object (``torch.zeros``, ``nn.Linear`` after
    ``from torch import nn``, a ``torch`` argument)."""
    CONFIG = "config"
    """A requirements, ``pyproject.toml`` or CMake line naming torch."""


@dataclass(frozen=True)
class TorchUse:
    """One torch use: its 1-based ``line``, ``kind`` and the code it found."""

    line: int
    kind: TorchUseKind
    detail: str


_TORCH = "torch"
_INTEROP = "torch_interop"
_API_SUBMODULES = frozenset({"ops", "fx", "compile", "cuda"})
_DYNAMIC_IMPORTS = frozenset({"__import__", "import_module", "find_spec"})
_SKIP_IMPORTS = frozenset({"importorskip", "import_or_skip_until"})
_MODULE_TABLE = "sys.modules"
_CONFIG_MENTION = re.compile(rf"\b{_TORCH}\b", re.IGNORECASE)
_IDENTIFIER_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def _is_torch_module(module: str) -> bool:
    return module.split(".")[0] == _TORCH


def _is_torch_api(module: str) -> bool:
    parts = module.split(".")
    return len(parts) > 1 and parts[0] == _TORCH and parts[1] in _API_SUBMODULES


def _names_interop(module: str) -> bool:
    return _INTEROP in module.split(".")


def _module_kinds(module: str) -> list[TorchUseKind]:
    """The kinds an import of ``module`` (a dotted name) makes."""
    kinds = []
    if _is_torch_module(module):
        kinds.append(TorchUseKind.IMPORT)
    if _is_torch_api(module):
        kinds.append(TorchUseKind.TORCH_API)
    if _names_interop(module):
        kinds.append(TorchUseKind.TORCH_INTEROP)
    return kinds


def _torch_bindings(tree: ast.AST) -> dict[str, str]:
    """Each name the module binds to a torch module or member, with what it is
    (``{"nn": "torch.nn"}``). ``torch`` itself always counts, so a torch module
    obtained without an import (``importorskip``, an argument) is still seen."""
    bindings = {_TORCH: _TORCH}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if not _is_torch_module(imported.name):
                    continue
                if imported.asname:
                    bindings[imported.asname] = imported.name
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and _is_torch_module(node.module or "")
        ):
            for imported in node.names:
                bound = imported.asname or imported.name
                bindings[bound] = f"{node.module}.{imported.name}"
    return bindings


def _object_kind(expanded: str) -> TorchUseKind:
    return (
        TorchUseKind.TORCH_API if _is_torch_api(expanded) else TorchUseKind.TORCH_OBJECT
    )


def _import_uses(node: ast.Import | ast.ImportFrom) -> list[TorchUse]:
    uses = []
    if isinstance(node, ast.Import):
        for imported in node.names:
            uses.extend(
                TorchUse(node.lineno, kind, f"import {imported.name}")
                for kind in _module_kinds(imported.name)
            )
        return uses
    module = node.module or ""
    if node.level == 0 and _is_torch_module(module):
        for imported in node.names:
            full_name = f"{module}.{imported.name}"
            uses.append(TorchUse(node.lineno, TorchUseKind.IMPORT, f"from {full_name}"))
            if _is_torch_api(full_name):
                uses.append(
                    TorchUse(node.lineno, TorchUseKind.TORCH_API, f"from {full_name}")
                )
    if _names_interop(module) or any(item.name == _INTEROP for item in node.names):
        names = ", ".join(item.name for item in node.names)
        uses.append(
            TorchUse(
                node.lineno,
                TorchUseKind.TORCH_INTEROP,
                f"from {'.' * node.level}{module} import {names}",
            )
        )
    return uses


def _call_uses(node: ast.Call) -> list[TorchUse]:
    called = dotted_name(node.func)
    function = called.split(".")[-1]
    module = string_constant(node.args[0]) if node.args else None
    if module is None:
        return []
    detail = f"{called}({module!r})"
    if function in _SKIP_IMPORTS and _is_torch_module(module):
        return [TorchUse(node.lineno, TorchUseKind.IMPORTORSKIP, detail)]
    if function in _DYNAMIC_IMPORTS or called == f"{_MODULE_TABLE}.get":
        return [TorchUse(node.lineno, kind, detail) for kind in _module_kinds(module)]
    return []


def _module_table_uses(node: ast.Subscript) -> list[TorchUse]:
    """``sys.modules["torch"]`` read (not assigned or deleted): the module itself."""
    module = string_constant(node.slice)
    if not (
        isinstance(node.ctx, ast.Load)
        and dotted_name(node.value) == _MODULE_TABLE
        and module is not None
    ):
        return []
    detail = f"{_MODULE_TABLE}[{module!r}]"
    return [TorchUse(node.lineno, kind, detail) for kind in _module_kinds(module)]


def torch_uses_in_python(source: str, filename: str) -> list[TorchUse]:
    """Every torch use in Python ``source``, in line order.

    ``filename`` names the source in a ``SyntaxError``; a file that does not
    parse is an error, not a clean file.
    """
    if _TORCH not in source:
        return []
    tree = ast.parse(source, filename=filename)
    bindings = _torch_bindings(tree)
    uses: list[TorchUse] = []
    chained: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            uses.extend(_import_uses(node))
        elif isinstance(node, ast.Call):
            uses.extend(_call_uses(node))
        elif isinstance(node, ast.Subscript):
            uses.extend(_module_table_uses(node))
        elif isinstance(node, ast.Attribute):
            if node.attr == _INTEROP:
                uses.append(
                    TorchUse(node.lineno, TorchUseKind.TORCH_INTEROP, dotted_name(node))
                )
            root = node.value
            if isinstance(root, ast.Name) and root.id in bindings:
                chained.add(id(root))
                expanded = f"{bindings[root.id]}.{node.attr}"
                uses.append(
                    TorchUse(node.lineno, _object_kind(expanded), dotted_name(node))
                )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in bindings
            and id(node) not in chained
        ):
            kind = _object_kind(bindings[node.id])
            uses.append(TorchUse(node.lineno, kind, node.id))
    return sorted(uses, key=lambda use: (use.line, use.kind.value, use.detail))


def torch_uses_in_config(text: str) -> list[TorchUse]:
    """Every line of a requirements, ``pyproject.toml`` or CMake ``text`` that
    names torch as a word, comments included."""
    return [
        TorchUse(number, TorchUseKind.CONFIG, line.strip())
        for number, line in enumerate(text.splitlines(), 1)
        if _CONFIG_MENTION.search(line)
    ]


def is_config_file(path: Path) -> bool:
    """Whether ``path`` declares dependencies or build steps: a requirements file,
    ``pyproject.toml``, ``CMakeLists.txt`` or a ``.cmake`` file."""
    name = path.name
    return (
        name in ("pyproject.toml", "CMakeLists.txt")
        or path.suffix == ".cmake"
        or (name.startswith("requirements") and path.suffix == ".txt")
    )


def torch_uses_in_file(path: Path) -> list[TorchUse]:
    """The torch uses of one file: Python by AST, config files by line, others
    none."""
    if path.suffix == ".py":
        return torch_uses_in_python(path.read_text(encoding="utf-8"), str(path))
    if is_config_file(path):
        return torch_uses_in_config(path.read_text(encoding="utf-8"))
    return []


def torch_named_identifiers(source: str, filename: str) -> list[tuple[int, str]]:
    """Each identifier of Python ``source`` that carries torch as a word of its
    name, once per identifier, with its first line: names to rename, not uses.
    ``torch`` itself is a use (:attr:`TorchUseKind.TORCH_OBJECT`), not listed."""
    if _TORCH not in source.lower():
        return []
    first_line: dict[str, int] = {}
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Name):
            identifier = node.id
        elif isinstance(node, ast.Attribute):
            identifier = node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifier = node.name
        elif isinstance(node, ast.arg):
            identifier = node.arg
        elif isinstance(node, ast.alias):
            identifier = node.asname or node.name.split(".")[-1]
        else:
            continue
        if identifier == _TORCH or not any(
            word.lower() == _TORCH for word in _IDENTIFIER_WORD.findall(identifier)
        ):
            continue
        line = getattr(node, "lineno", 0)
        first_line[identifier] = min(first_line.get(identifier, line), line)
    return sorted((line, identifier) for identifier, line in first_line.items())


def tree_files(root: Path) -> list[str]:
    """The files under ``root`` that the gate reads, relative POSIX paths, sorted:
    the Python files and config files (:func:`is_config_file`) of
    :func:`~rocke.tests.source_tree.source_files`, so build trees, virtual
    environments and ignored scratch directories are never read.

    Raises ``RuntimeError`` when git is missing or ``root`` is not in a git
    work tree: the gate checks a source checkout.
    """
    return [
        path
        for path in source_files(root)
        if path.endswith(".py") or is_config_file(Path(path))
    ]


def tree_torch_uses(root: Path) -> dict[str, list[TorchUse]]:
    """The torch uses of every file :func:`tree_files` lists under ``root``,
    keyed by relative path; files with none are left out."""
    found = {}
    for path in tree_files(root):
        if uses := torch_uses_in_file(root / path):
            found[path] = uses
    return found


def read_allowlist(path: Path) -> dict[str, AllowlistEntry]:
    """The allowlist at ``path``: one ``file: kind kind`` line per file
    (:func:`rocke.tests.source_allowlist.read_allowlist`)."""
    return source_allowlist.read_allowlist(path, TorchUseKind)


def allowlist_breaches(
    found: Mapping[str, Sequence[TorchUse]],
    allowed: Mapping[str, AllowlistEntry],
) -> list[str]:
    """What breaks the allowlist, one line per file, sorted by file
    (:func:`rocke.tests.source_allowlist.allowlist_breaches`): a file that
    uses torch and is not listed, a kind of use its line does not list, and a
    line or kind the tree no longer uses."""
    return source_allowlist.allowlist_breaches(found, allowed, subject="torch use")


def _print_inventory(root: Path, *, names: bool) -> None:
    if not names:
        for path, uses in tree_torch_uses(root).items():
            print(
                source_allowlist.format_allowlist_entry(
                    path, source_allowlist.file_kinds(uses)
                )
            )
        return
    for path in tree_files(root):
        if not path.endswith(".py"):
            continue
        source = (root / path).read_text(encoding="utf-8")
        for line, identifier in torch_named_identifiers(source, path):
            print(f"{path}:{line}: {identifier}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rocke.tests.torch_uses", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="the tree to read (default: the rocke source root)",
    )
    parser.add_argument(
        "--names",
        action="store_true",
        help="list torch-named identifiers instead of each file's kinds of use",
    )
    arguments = parser.parse_args(argv)
    root = arguments.root or platform_root().parent
    _print_inventory(root, names=arguments.names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
