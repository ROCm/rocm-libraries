# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Architecture selection for installed, operation-specific reference CTests."""

from __future__ import annotations

import importlib
import os

import pytest


def architecture_available(
    architecture: str,
    families: tuple[str, ...],
    detected: str | None,
    declared: str,
    targets: str = "",
) -> bool:
    """Skip a different GPU lane; fail when the intended lane lacks its GPU."""
    # TheRock supplies comma-separated artifact targets independently of its
    # broader family name. Exact targets distinguish lanes within one family.
    exact_targets = {
        target.strip().split(":", 1)[0].lower()
        for target in targets.split(",")
        if target.strip()
    }
    if exact_targets and architecture not in exact_targets:
        return False
    if detected == architecture:
        return True
    if detected in exact_targets:
        # A job may fetch multiple target artifacts but run on one of them.
        return False
    expected = bool(exact_targets) or any(
        family.lower() in declared.lower() for family in families
    )
    if expected or (not detected and not declared):
        raise RuntimeError(
            f"required {architecture} GPU unavailable; detected {detected!r}"
        )
    return False


def start_reference_session(config) -> None:
    architecture = config.getoption("--rocke-reference-arch")
    operation = config.getoption("--rocke-reference-operation")
    if not architecture and not operation:
        return
    if not architecture or not operation:
        raise pytest.UsageError(
            "reference operation and architecture must be specified together"
        )
    registry = importlib.import_module(f"{operation}_reference.architectures")
    try:
        target = registry.get_architecture(architecture)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    from rocke.runtime.hip_module import get_device_arch

    try:
        available = architecture_available(
            architecture,
            target.FAMILIES,
            get_device_arch(),
            os.environ.get("AMDGPU_FAMILIES", ""),
            os.environ.get("AMDGPU_TARGETS", ""),
        )
    except RuntimeError as error:
        raise pytest.UsageError(str(error)) from error
    if not available:
        pytest.exit(
            f"reference suite requires {architecture}; different GPU lane",
            returncode=77,
        )


def select_reference_items(config, items) -> None:
    architecture = config.getoption("--rocke-reference-arch")
    if not architecture:
        return
    selected, deselected = [], []
    for item in items:
        callspec = getattr(item, "callspec", None)
        case_arch = callspec.params.get("architecture") if callspec else None
        (selected if case_arch in (None, architecture) else deselected).append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)
