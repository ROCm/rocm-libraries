# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Architecture selection for installed, operation-specific reference CTests."""

from __future__ import annotations

import importlib
import os

import pytest


def architecture_available(
    architecture: str,
    detected: str | None,
    targets: str = "",
) -> bool:
    """Select a processor's cohort within TheRock's requested artifact targets.

    TheRock owns family expansion. AMDGPU_FAMILIES describes build/artifact
    grouping, not reference enrollment, so do not infer coverage from its name.
    Without exact targets, select by the actual processor alone.
    """
    # TheRock supplies comma-separated artifact targets independently of its
    # broader family name. Exact targets distinguish lanes within one family.
    # Accept a CMake-style semicolon list too, so a local export cannot turn
    # matching hardware into a single unknown target and skip the suite.
    exact_targets = {
        target.strip().split(":", 1)[0].lower()
        for target in targets.replace(";", ",").split(",")
        if target.strip()
    }
    if exact_targets and architecture not in exact_targets:
        return False
    if not detected or (exact_targets and detected not in exact_targets):
        raise RuntimeError(
            f"required {architecture} GPU unavailable; detected {detected!r}"
        )
    # A supported processor may have no published reference for this operation.
    # Another processor's suite must skip before accessing its packaged bundle.
    return detected == architecture


def start_reference_session(config) -> None:
    architecture = config.getoption("--rocke-reference-arch")
    operation = config.getoption("--rocke-reference-operation")
    bundle = config.getoption("--rocke-reference-bundle")
    lock = config.getoption("--rocke-reference-lock")
    if (bundle or lock) and not (architecture and operation):
        raise pytest.UsageError(
            "reference bundle/lock overrides require explicit operation and architecture"
        )
    if lock and not bundle:
        raise pytest.UsageError("reference lock override requires a reference bundle")
    if not architecture and not operation:
        return
    if not architecture or not operation:
        raise pytest.UsageError(
            "reference operation and architecture must be specified together"
        )
    registry = importlib.import_module(f"{operation}_reference.architectures")
    try:
        registry.get_architecture(architecture)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    from .targets import get_device_processor

    # Match the processor against artifact targets. get_device_arch() returns
    # a catalog name (e.g. gfx12-generic for gfx1200), losing that identity.
    detected = get_device_processor()
    targets = os.environ.get("AMDGPU_TARGETS", "")

    try:
        available = architecture_available(architecture, detected, targets)
    except RuntimeError as error:
        raise pytest.UsageError(str(error)) from error
    if not available:
        pytest.exit(
            f"reference suite requires {architecture}; different GPU lane "
            f"(detected {detected}, AMDGPU_TARGETS={targets!r})",
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
