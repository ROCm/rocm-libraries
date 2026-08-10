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
_root: Path | None = None
_root_source: str | None = None
_distribution_version: str | None = None


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
    """Validate generator prerequisites without requiring the optional client."""
    global _client, _root, _root_source, _distribution_version

    _require_rocisa()
    validated = validate_distribution("tensilelite", distribution_version)
    _client = None
    _root = validated.root
    _root_source = validated.source
    _distribution_version = distribution_version


def client_executable() -> Path:
    """Return the validated client selected by this installation on first use."""
    global _client

    if _client is not None:
        return _client
    if _root is None or _root_source is None or _distribution_version is None:
        raise TensileLiteRuntimeError("TensileLite runtime has not been initialized.")

    try:
        selected, custom = selected_client(_root)
        validate_client(selected, _distribution_version)
    except ClientBindingError as exc:
        raise TensileLiteRuntimeError(
            f"{exc}\n"
            f"  selected root: {_root}\n"
            f"  selected by: {_root_source}"
        ) from exc
    _client = selected
    return _client


def rocm_root() -> Path:
    """Return the ROCm root frozen during package initialization."""
    if _root is None:
        raise TensileLiteRuntimeError("TensileLite runtime has not been initialized.")
    return _root
