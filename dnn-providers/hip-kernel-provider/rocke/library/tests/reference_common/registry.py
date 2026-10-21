# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Enrolled processors and their execution adapters, shared by every operation.

Each ``<operation>_reference/architectures`` package lists its processors in
``registry.json`` and delegates enrollment and lock lookup here. Qualification
freezes this module into each bundle's runner, whose worker resolves its
adapter through it, so it needs only the standard library.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType


class AdapterRegistry:
    """One operation's enrolled processors and their adapter subpackages.

    ``package`` names the architectures package, which holds one adapter
    subpackage per processor, and ``root`` is its directory. ``label`` names
    the operation in errors. Enrollment is read once from ``registry.json``,
    so a name that is not listed there never reaches an import.
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

    def get_architecture(self, name: str) -> ModuleType:
        """Reject unenrolled architectures before importing any execution adapter."""
        if name not in self._architectures:
            raise ValueError(
                f"{self._label} reference architecture is not enrolled: {name}"
            )
        return importlib.import_module(f"{self._package}.{name}")

    def baseline_lock(self, name: str) -> Path:
        """Resolve a target's independently qualified lock without probing hardware."""
        self.get_architecture(name)
        return self._root / name / "baseline_lock.json"
