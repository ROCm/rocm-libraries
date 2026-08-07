# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validation for the ROCm-native artifacts required by TensileLite."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from _tensilelite_client_binding import (
    ClientBindingError,
    selected_client,
    validate_client,
)

from ._rocm import TensileLiteRuntimeError, validate_distribution


_client: Path | None = None
_custom = False


def _require_rocisa() -> None:
    """Require rocisa without interpreting its version or native layout."""
    try:
        import_module("rocisa")
    except (ImportError, OSError) as exc:
        raise TensileLiteRuntimeError(
            "TensileLite requires an independently packaged, importable rocisa dependency. "
            f"The rocisa import failed: {exc}"
        ) from exc


def initialize(distribution_version: str) -> None:
    """Validate and freeze the client selected by this installation."""
    global _client, _custom

    _require_rocisa()
    validated = validate_distribution("tensilelite", distribution_version)
    try:
        selected, custom = selected_client(validated.root)
        validate_client(selected, distribution_version)
    except ClientBindingError as exc:
        raise TensileLiteRuntimeError(str(exc)) from exc
    _client = selected
    _custom = custom


def client_executable() -> Path:
    """Return the client path frozen during package initialization."""
    if _client is None:
        raise TensileLiteRuntimeError("TensileLite runtime has not been initialized.")
    return _client
