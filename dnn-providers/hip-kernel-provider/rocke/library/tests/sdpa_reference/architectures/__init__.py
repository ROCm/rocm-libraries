# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Explicit SDPA enrollments shared by qualification, verification, and CMake."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parent
ARCHITECTURES = tuple(json.loads((_ROOT / "registry.json").read_text()))


def get_architecture(name: str) -> ModuleType:
    """Reject unenrolled architectures before importing any execution adapter."""
    if name not in ARCHITECTURES:
        raise ValueError(f"SDPA reference architecture is not enrolled: {name}")
    return importlib.import_module(f"{__name__}.{name}")


def baseline_lock(name: str) -> Path:
    """Resolve a target's independently qualified lock without probing hardware."""
    get_architecture(name)
    return _ROOT / name / "baseline_lock.json"
