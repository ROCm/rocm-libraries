# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Enrolled processors and their execution adapters, shared by every operation.

Each ``<operation>_reference/architectures`` package lists its processors in
``registry.json`` and delegates enrollment and lock lookup here. A processor's
package names it, and each of its variant subpackages adapts one kernel family
and holds that variant's lock once published. Qualification freezes this module
into each bundle's runner, whose worker resolves its adapter through it, so it
needs only the standard library.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType


class AdapterRegistry:
    """One operation's enrolled processors and their variant adapters.

    ``package`` names the architectures package, which holds one subpackage
    per processor, and ``root`` is its directory. ``label`` names the operation
    in errors. Enrollment is read once from ``registry.json``, so a name that is
    not listed there never reaches an import. A processor's variants are the
    subdirectories of its package that hold an ``__init__.py``; they are found
    without importing them.
    """

    def __init__(self, package: str, root: Path, *, label: str) -> None:
        self._package = package
        self._root = root
        self._label = label
        self._architectures = tuple(json.loads((root / "registry.json").read_text()))

    @property
    def architectures(self) -> tuple[str, ...]:
        """Enrolled processors in ``registry.json`` order."""
        return self._architectures

    def _check_enrolled(self, name: str) -> None:
        if name not in self._architectures:
            raise ValueError(
                f"{self._label} reference architecture is not enrolled: {name}"
            )

    def get_architecture(self, name: str) -> ModuleType:
        """Import a processor's package, which provides only its ``NAME``.

        Unenrolled names are rejected first. Replay needs only the name, so it
        never imports a variant adapter.
        """
        self._check_enrolled(name)
        return importlib.import_module(f"{self._package}.{name}")

    def variants(self, name: str) -> tuple[str, ...]:
        """A processor's variants in name order, found without importing them."""
        self._check_enrolled(name)
        adapters = (self._root / name).glob("*/__init__.py")
        return tuple(sorted(path.parent.name for path in adapters))

    def resolve_variant(self, name: str, variant: str | None = None) -> str:
        """Return ``variant`` if the processor enrolls it, else its only variant.

        Without ``variant``, a processor with several variants is ambiguous; the
        error lists them.
        """
        enrolled = self.variants(name)
        if variant is None and len(enrolled) == 1:
            return enrolled[0]
        if variant in enrolled:
            return variant
        problem = (
            "needs a selected reference variant"
            if variant is None
            else f"reference variant is not enrolled: {variant}"
        )
        choices = ", ".join(enrolled) or "none"
        raise ValueError(f"{self._label} {name} {problem}; enrolled: {choices}")

    def get_adapter(self, name: str, variant: str | None = None) -> ModuleType:
        """Import a variant's execution adapter, by default the processor's only one."""
        variant = self.resolve_variant(name, variant)
        return importlib.import_module(f"{self._package}.{name}.{variant}")

    def baseline_lock(self, name: str, variant: str | None = None) -> Path:
        """Resolve a variant's independently qualified lock without probing hardware.

        ``variant`` defaults to the processor's only variant. Nothing is
        imported; the lock exists once the variant is published.
        """
        variant = self.resolve_variant(name, variant)
        return self._root / name / variant / "baseline_lock.json"
