# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Temporary support for aliasing one package namespace to another."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from types import ModuleType
from typing import Any


_MISSING = object()
_METADATA_ATTRIBUTES = (
    "__name__",
    "__package__",
    "__loader__",
    "__spec__",
    "__file__",
    "__cached__",
    "__path__",
)


class _AliasLoader(importlib.abc.InspectLoader):
    """Return a canonical module for a compatibility import name."""

    def __init__(
        self,
        alias_name: str,
        canonical_name: str,
        canonical_spec: importlib.machinery.ModuleSpec,
    ) -> None:
        self._alias_name = alias_name
        self._canonical_name = canonical_name
        self._canonical_spec = canonical_spec
        self._metadata: dict[str, Any] = {}

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module(self._canonical_name)
        self._metadata = {
            attribute: getattr(module, attribute, _MISSING)
            for attribute in _METADATA_ATTRIBUTES
        }
        return module

    def exec_module(self, module: ModuleType) -> None:
        # Import machinery applies the alias spec to the object returned by
        # create_module(). Restore the canonical metadata so resources, reload,
        # reprs, and serialized type names continue to describe one real tree.
        for attribute, value in self._metadata.items():
            if value is _MISSING:
                module.__dict__.pop(attribute, None)
            else:
                setattr(module, attribute, value)
        sys.modules[self._alias_name] = module

    def get_code(self, fullname: str):
        get_code = getattr(self._canonical_spec.loader, "get_code", None)
        return get_code(self._canonical_name) if get_code else None

    def get_source(self, fullname: str):
        get_source = getattr(self._canonical_spec.loader, "get_source", None)
        return get_source(self._canonical_name) if get_source else None

    def is_package(self, fullname: str) -> bool:
        return self._canonical_spec.submodule_search_locations is not None


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolve descendants of one package name through another package."""

    def __init__(self, alias: str, canonical: str) -> None:
        self.alias = alias
        self.canonical = canonical

    def find_spec(self, fullname: str, path=None, target=None):
        if not fullname.startswith(f"{self.alias}."):
            return None

        canonical_name = self.canonical + fullname[len(self.alias):]
        canonical_spec = importlib.util.find_spec(canonical_name)
        if canonical_spec is None:
            return None

        is_package = canonical_spec.submodule_search_locations is not None
        loader = _AliasLoader(fullname, canonical_name, canonical_spec)
        spec = importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=canonical_spec.origin,
            is_package=is_package,
        )
        if spec is None:
            return None

        spec.has_location = canonical_spec.has_location
        spec.cached = canonical_spec.cached
        if is_package:
            spec.submodule_search_locations = list(
                canonical_spec.submodule_search_locations or ()
            )
        return spec


def install_alias(*, alias: str, canonical: str) -> ModuleType:
    """Map ``alias`` and all of its descendants to ``canonical`` modules."""
    if not alias or not canonical or alias == canonical:
        raise ValueError("alias and canonical must be distinct package names")

    finder = next(
        (
            item
            for item in sys.meta_path
            if isinstance(item, _AliasFinder)
            and item.alias == alias
            and item.canonical == canonical
        ),
        None,
    )
    if finder is None:
        sys.meta_path.insert(0, _AliasFinder(alias, canonical))

    canonical_root = importlib.import_module(canonical)
    for name, module in tuple(sys.modules.items()):
        if module is None:
            continue
        if name == canonical or name.startswith(f"{canonical}."):
            alias_name = alias + name[len(canonical):]
            sys.modules[alias_name] = module

    return canonical_root


__all__ = ["install_alias"]
