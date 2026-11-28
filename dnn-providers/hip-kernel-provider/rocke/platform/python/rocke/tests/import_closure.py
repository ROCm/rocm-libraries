# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The source files a Python script's imports can reach, found without running it.

One home for "which files can this script import": a check that runs a script
is affected by a change to any of them. The search reads each file's bytecode,
so every import statement counts, function-level and ``TYPE_CHECKING`` ones
included. It cannot follow a module named only at run time
(``importlib.import_module`` of a computed name, a file loaded by path) or a
file read as data. A namespace package (a directory without ``__init__.py``)
is searched as Python searches it, and holds no file of its own. An import it
cannot resolve in the search roots, the
standard library or the interpreter's built-in modules is reported by its
top-level name, so a caller can tell an installed package from code the search
did not reach (a module found through a ``sys.path`` entry the script adds at
run time, for one).
"""

from __future__ import annotations

import importlib.machinery
import modulefinder
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ImportClosure", "script_import_closure"]


@dataclass(frozen=True)
class ImportClosure:
    """What a script's imports reach: the files found, as resolved paths, and
    the top-level names of the imports found nowhere."""

    files: frozenset[Path]
    unresolved: frozenset[str]


class _NamespaceAwareFinder(modulefinder.ModuleFinder):
    """``modulefinder.ModuleFinder``, which fails with ``AttributeError`` on a
    namespace package, made to load one as Python does: a package whose search
    path is every directory of that name the search finds, with no file."""

    def __init__(self, path: list[str]) -> None:
        super().__init__(path=path)
        self._namespace_locations: dict[str, list[str]] = {}

    def find_module(self, name, path, parent=None):
        fullname = f"{parent.__name__}.{name}" if parent is not None else name
        spec = importlib.machinery.PathFinder.find_spec(
            name, self.path if path is None else path
        )
        if (
            fullname in self.excludes
            or spec is None
            or spec.loader is not None
            or not spec.submodule_search_locations
        ):
            return super().find_module(name, path, parent)
        locations = list(spec.submodule_search_locations)
        self._namespace_locations[fullname] = locations
        # The base class loads a package found this way through load_package;
        # its kind constant has no public name.
        return None, locations[0], ("", "", modulefinder._PKG_DIRECTORY)

    def load_package(self, fqname, pathname):
        locations = self._namespace_locations.get(fqname)
        if locations is None:
            return super().load_package(fqname, pathname)
        package = self.add_module(fqname)
        package.__path__ = locations
        return package


def _is_relative_import_name(
    name: str, callers: Iterable[str], modules: Mapping[str, object]
) -> bool:
    """Whether ModuleFinder listed ``name`` for a relative from-import: it lists
    ``from .core import x`` in package ``rocke`` as "core.x", which resolves
    against a package that holds each caller."""
    first = name.split(".", 1)[0]
    for caller in callers:
        parts = caller.split(".")
        if not any(
            ".".join([*parts[:depth], first]) in modules
            for depth in range(1, len(parts) + 1)
        ):
            return False
    return True


def _unresolved_names(finder: modulefinder.ModuleFinder) -> frozenset[str]:
    missing, _ = finder.any_missing_maybe()
    unresolved = set()
    for name in missing:
        if _is_relative_import_name(name, finder.badmodules[name], finder.modules):
            continue
        top = name.split(".", 1)[0]
        if top not in sys.stdlib_module_names and top not in sys.builtin_module_names:
            unresolved.add(top)
    return frozenset(unresolved)


def script_import_closure(script: Path, search_roots: Iterable[Path]) -> ImportClosure:
    """What the imports of ``script`` reach, transitively.

    An import resolves as it does for ``python script``: in the script's own
    directory first, then in ``search_roots`` in order. The files are
    ``script`` and every file found there; an import of the standard library
    is neither a file nor unresolved."""
    roots = [script.resolve().parent, *(Path(root).resolve() for root in search_roots)]
    finder = _NamespaceAwareFinder([str(root) for root in roots])
    finder.run_script(str(script))
    files = frozenset(
        Path(module.__file__).resolve()
        for module in finder.modules.values()
        if module.__file__
    )
    return ImportClosure(files, _unresolved_names(finder))
